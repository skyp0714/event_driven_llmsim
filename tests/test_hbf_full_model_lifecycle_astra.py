import dataclasses
import json
import math
import unittest
from unittest.mock import patch

from serving.core.controller import Controller
from serving.core.hbf_full_model_astra import (
    build_full_model_hbf_astra_projection,
)
from serving.core.hbf_full_model_latency import (
    HBFModelBatchLatency,
    HBFParallelLayout,
    HBFServerHardware,
    qwen_logical_kv_bytes_per_token,
)
from serving.core.hbf_full_model_lifecycle import AppendJob, MigrationJob
from serving.core.hbf_full_model_lifecycle_astra import (
    HBFLifecycleAstraProjectionError,
    HBF_WRITE_FIXED_LATENCY_SEMANTICS,
    RDMA_FIXED_LATENCY_SEMANTICS,
    build_append_hbf_astra_projection,
    build_migration_hbf_astra_projection,
)


def migration_job(
        layout_key, *, logical_bytes=101, group_id=0, job_id=7,
        start_ns=123):
    layout = HBFParallelLayout.for_key(layout_key)
    physical = (
        logical_bytes * layout.physical_kv_replication_factor)
    return MigrationJob(
        job_id=job_id,
        session_id="session-a",
        generation=3,
        version=5,
        group_id=group_id,
        token_count=11,
        logical_bytes=logical_bytes,
        physical_bytes=physical,
        per_card_bytes=math.ceil(physical / layout.tp_size),
        start_ns=start_ns,
        completion_ns=start_ns + 999_999,
    )


def append_job(
        layout_key, *, logical_bytes=101, group_id=0, job_id=9,
        start_ns=456):
    layout = HBFParallelLayout.for_key(layout_key)
    physical = (
        logical_bytes * layout.physical_kv_replication_factor)
    return AppendJob(
        job_id=job_id,
        session_id="session-a",
        generation=3,
        version=6,
        group_id=group_id,
        token_count=7,
        logical_bytes=logical_bytes,
        physical_bytes=physical,
        per_card_bytes=math.ceil(physical / layout.tp_size),
        start_ns=start_ns,
        completion_ns=start_ns + 888_888,
    )


def foreground_latency(layout_key):
    layout = HBFParallelLayout.for_key(layout_key)
    collectives = (1, 1, 1) if layout.tp_size > 1 else (0, 0, 0)
    collective_ns = sum(collectives)
    return HBFModelBatchLatency(
        layout=layout.key,
        tp_size=layout.tp_size,
        replicas=layout.replicas,
        total_ns=6 + collective_ns,
        embedding_ns=1,
        dense_ns=1,
        attention_ns=1,
        router_ns=1,
        moe_ns=1,
        final_ns=1,
        collective_ns=collective_ns,
        tp_allreduce_ns=collectives[0],
        ep_allgather_ns=collectives[1],
        ep_reduce_scatter_ns=collectives[2],
        hbf_read_bytes_per_rank=103,
        lpddr_bytes_per_rank=31,
        collective_bytes_per_rank=(
            17 if layout.tp_size > 1 else 0),
        compute_roof_ns_sum=1,
        hbf_roof_ns_sum=1,
        lpddr_roof_ns_sum=1,
        attention_compute_roof_ns=1,
        attention_hbf_roof_ns=1,
        attention_lpddr_roof_ns=1,
        attention_dominant_roof="hbf_read",
        dominant_kernel_counts={
            "compute": 1,
            "hbf_read": 1,
            "lpddr": 1,
        },
    )


class MigrationLifecycleAstraProjectionTest(unittest.TestCase):
    def setUp(self):
        self.hardware = HBFServerHardware()

    def project(
            self, key, *, logical_bytes=101, group_id=0,
            chunk_bytes=37, server_id=0, hardware=None):
        layout = HBFParallelLayout.for_key(key)
        job = migration_job(
            key, logical_bytes=logical_bytes, group_id=group_id)
        projection = build_migration_hbf_astra_projection(
            job=job,
            hardware=hardware or self.hardware,
            layout=layout,
            chunk_bytes=chunk_bytes,
            server_id=server_id,
        )
        return job, projection

    def test_all_layouts_preserve_exact_hop_and_card_byte_ledgers(self):
        for key in ("dp8", "tp4", "tp8"):
            with self.subTest(layout=key):
                job, projection = self.project(key)
                ledger = projection.byte_ledger
                self.assertEqual(
                    ledger.gpu_source_pcie_bytes, job.logical_bytes)
                self.assertEqual(ledger.rdma_bytes, job.logical_bytes)
                self.assertEqual(
                    ledger.destination_pcie_root_bytes,
                    job.physical_bytes,
                )
                self.assertEqual(
                    ledger.destination_pcie_card_bytes,
                    job.physical_bytes,
                )
                self.assertEqual(
                    ledger.hbf_write_bytes, job.physical_bytes)
                self.assertEqual(
                    sum(row.hbf_write_bytes
                        for row in projection.card_ledgers),
                    job.physical_bytes,
                )
                self.assertLessEqual(
                    max(row.hbf_write_bytes
                        for row in projection.card_ledgers),
                    job.per_card_bytes,
                )
                self.assertEqual(
                    sum(stage.tensor_bytes
                        for stage in projection.stages
                        if stage.role == "hbf_write"),
                    job.physical_bytes,
                )
        dp8, _ = self.project("dp8")
        tp4, _ = self.project("tp4")
        tp8, _ = self.project("tp8")
        self.assertEqual(dp8.physical_bytes, dp8.logical_bytes)
        self.assertEqual(tp4.physical_bytes, tp4.logical_bytes)
        self.assertEqual(tp8.physical_bytes, 2 * tp8.logical_bytes)

    def test_pipeline_is_causal_and_has_exact_uncontended_critical_path(self):
        hardware = dataclasses.replace(
            self.hardware,
            hbf_write_bandwidth_gbps_per_card=1.0,
            hbf_write_latency_us=0.0,
            intra_fabric_bandwidth_gbps_per_card=1.0,
            pcie_root_bandwidth_gbps=1.0,
            rdma_bandwidth_gbps=1.0,
            rdma_one_way_latency_us=0.0,
        )
        _, projection = self.project(
            "dp8",
            logical_bytes=25,
            chunk_bytes=10,
            hardware=hardware,
        )
        self.assertEqual(projection.logical_chunks, (10, 10, 5))
        self.assertEqual(
            projection.dependency_critical_path_ns(), 65)
        by_id = {
            stage.stage_id: stage for stage in projection.stages
        }
        prefix = "migration:7:replica:0"
        self.assertEqual(
            by_id[f"{prefix}:chunk:1:rdma"].dependencies,
            (
                f"{prefix}:chunk:1:source-pcie",
                f"{prefix}:chunk:0:rdma",
            ),
        )
        self.assertEqual(
            by_id[f"{prefix}:chunk:1:pcie-root:0"].dependencies,
            (
                f"{prefix}:chunk:1:rdma",
                f"{prefix}:chunk:0:pcie-root:0",
            ),
        )
        self.assertEqual(
            by_id[
                f"{prefix}:chunk:1:card:0:hbf-write"
            ].dependencies,
            (
                f"{prefix}:chunk:1:card:0:pcie",
                f"{prefix}:chunk:0:card:0:hbf-write",
            ),
        )

    def test_fixed_latency_is_once_per_transfer_not_per_chunk(self):
        hardware = dataclasses.replace(
            self.hardware,
            rdma_bandwidth_gbps=3.0,
            rdma_one_way_latency_us=2.25,
            hbf_write_bandwidth_gbps_per_card=7.0,
            hbf_write_latency_us=1.25,
        )
        _, projection = self.project(
            "tp4",
            logical_bytes=101,
            chunk_bytes=13,
            hardware=hardware,
        )
        rdma = [
            stage for stage in projection.stages
            if stage.role == "rdma"
        ]
        self.assertEqual(rdma[0].fixed_latency_ns, 2_250)
        self.assertTrue(all(
            stage.fixed_latency_ns == 0 for stage in rdma[1:]))
        self.assertEqual(
            sum(stage.fixed_latency_ns for stage in rdma), 2_250)
        for card_id in projection.replica.card_ids:
            writes = [
                stage for stage in projection.stages
                if stage.role == "hbf_write"
                and stage.card_id == card_id
            ]
            self.assertTrue(writes)
            self.assertEqual(writes[0].fixed_latency_ns, 1_250)
            self.assertTrue(all(
                stage.fixed_latency_ns == 0 for stage in writes[1:]))
        for stage in projection.stages:
            self.assertEqual(
                stage.service_ns,
                math.ceil(stage.tensor_bytes / stage.bandwidth_gbps),
            )
            self.assertEqual(
                stage.runtime_ns,
                stage.service_ns + stage.fixed_latency_ns,
            )
            self.assertGreater(stage.runtime_ns, 0)
        audit = projection.audit_dict()
        self.assertEqual(
            audit["fixed_latency_semantics"]["rdma"],
            RDMA_FIXED_LATENCY_SEMANTICS,
        )
        self.assertEqual(
            audit["fixed_latency_semantics"]["hbf_write"],
            HBF_WRITE_FIXED_LATENCY_SEMANTICS,
        )

    def test_tp8_uses_two_destination_roots_and_exact_replication(self):
        job, projection = self.project(
            "tp8", logical_bytes=103, chunk_bytes=29, server_id=4)
        self.assertEqual(job.physical_bytes, 206)
        root_stages = [
            stage for stage in projection.stages
            if stage.role == "destination_pcie_root"
        ]
        self.assertEqual(
            {stage.root_id for stage in root_stages}, {0, 1})
        self.assertEqual(
            sum(stage.tensor_bytes for stage in root_stages),
            job.physical_bytes,
        )
        root_bytes = {
            root_id: sum(
                row.destination_pcie_bytes
                for row in projection.card_ledgers
                if row.card_id // self.hardware.cards_per_pcie_root
                == root_id
            )
            for root_id in (0, 1)
        }
        self.assertEqual(
            {
                root_id: sum(
                    stage.tensor_bytes for stage in root_stages
                    if stage.root_id == root_id
                )
                for root_id in (0, 1)
            },
            root_bytes,
        )
        for chunk_index in range(len(projection.logical_chunks)):
            self.assertEqual(
                {
                    stage.root_id for stage in root_stages
                    if stage.chunk_index == chunk_index
                },
                {0, 1},
            )
        self.assertTrue(all(
            resource.startswith("hbf-server:4:")
            for stage in projection.stages
            for resource in stage.resources
        ))

    def test_second_tp4_replica_cannot_escape_cards_or_root(self):
        _, projection = self.project(
            "tp4", group_id=1, server_id=6)
        self.assertEqual(projection.replica.card_ids, (4, 5, 6, 7))
        self.assertEqual(projection.replica.pcie_root_ids, (1,))
        for stage in projection.stages:
            if stage.card_id is not None:
                self.assertIn(stage.card_id, (4, 5, 6, 7))
            if stage.root_id is not None:
                self.assertEqual(stage.root_id, 1)
            self.assertFalse(any(
                f":card:{card_id}:" in resource
                for card_id in range(4)
                for resource in stage.resources
            ))

    def test_controller_accepts_exact_descriptor_and_deterministic_ids(self):
        job, projection = self.project(
            "tp4", group_id=1, server_id=8)
        command = Controller.hbf_background_command(
            *projection.controller_command_arguments())
        prefix, job_id, arrival, encoded = command.split("\t")
        self.assertEqual(prefix, "hbf-background")
        self.assertEqual(job_id, "hbf-migration.s8.r1.j7")
        self.assertEqual(arrival, str(job.start_ns))
        self.assertEqual(json.loads(encoded), projection.descriptor())
        self.assertEqual(
            set(projection.controller_stages()[0]),
            {"id", "runtime_ns", "tensor_bytes", "resources", "deps"},
        )
        repeated = build_migration_hbf_astra_projection(
            job=job,
            hardware=self.hardware,
            layout=HBFParallelLayout.for_key("tp4"),
            chunk_bytes=37,
            server_id=8,
        )
        self.assertEqual(
            projection.descriptor_json(), repeated.descriptor_json())
        self.assertEqual(job, migration_job(
            "tp4", logical_bytes=101, group_id=1))


class AppendLifecycleAstraProjectionTest(unittest.TestCase):
    def setUp(self):
        self.hardware = HBFServerHardware()

    def project(
            self, key, *, logical_bytes=101, group_id=0,
            chunk_bytes=37, server_id=0):
        layout = HBFParallelLayout.for_key(key)
        job = append_job(
            key, logical_bytes=logical_bytes, group_id=group_id)
        projection = build_append_hbf_astra_projection(
            job=job,
            hardware=self.hardware,
            layout=layout,
            chunk_bytes=chunk_bytes,
            server_id=server_id,
        )
        return job, projection

    def test_append_is_lpddr_then_hbf_and_uses_foreground_resources(self):
        job, projection = self.project(
            "tp4", logical_bytes=317, chunk_bytes=41, server_id=2)
        self.assertEqual(
            set(stage.role for stage in projection.stages),
            {"lpddr_read", "hbf_write"},
        )
        self.assertEqual(
            projection.byte_ledger.lpddr_read_bytes,
            job.physical_bytes,
        )
        self.assertEqual(
            projection.byte_ledger.hbf_write_bytes,
            job.physical_bytes,
        )
        self.assertEqual(
            projection.byte_ledger.rdma_bytes, 0)
        by_id = {
            stage.stage_id: stage for stage in projection.stages
        }
        for write in (
                stage for stage in projection.stages
                if stage.role == "hbf_write"):
            read_id = write.stage_id.replace("hbf-write", "lpddr-read")
            self.assertIn(read_id, write.dependencies)
            self.assertIn(read_id, by_id)
            self.assertEqual(
                write.resources,
                (f"hbf-server:2:card:{write.card_id}:hbf-read",),
            )
        for read in (
                stage for stage in projection.stages
                if stage.role == "lpddr_read"):
            self.assertEqual(
                read.resources,
                (f"hbf-server:2:card:{read.card_id}:lpddr",),
            )
        self.assertFalse(any(
            "ingress" in resource or "pcie-root" in resource
            for stage in projection.stages
            for resource in stage.resources
        ))

        foreground = build_full_model_hbf_astra_projection(
            latency=foreground_latency("tp4"),
            hardware=self.hardware,
            layout=HBFParallelLayout.for_key("tp4"),
            replica_id=0,
            batch_id=81,
            server_id=2,
        )
        foreground_resources = {
            resource
            for stage in foreground.stages
            for resource in stage.resources
        }
        lifecycle_hbf = {
            resource
            for stage in projection.stages
            if stage.role == "hbf_write"
            for resource in stage.resources
        }
        lifecycle_lpddr = {
            resource
            for stage in projection.stages
            if stage.role == "lpddr_read"
            for resource in stage.resources
        }
        self.assertTrue(lifecycle_hbf <= foreground_resources)
        self.assertTrue(lifecycle_lpddr <= foreground_resources)

    def test_tp8_append_ledger_includes_physical_kv_replication(self):
        job, projection = self.project(
            "tp8", logical_bytes=103, chunk_bytes=17)
        self.assertEqual(job.physical_bytes, 206)
        self.assertEqual(
            sum(row.lpddr_read_bytes
                for row in projection.card_ledgers),
            206,
        )
        self.assertEqual(
            sum(row.hbf_write_bytes
                for row in projection.card_ledgers),
            206,
        )
        self.assertEqual(
            sum(stage.tensor_bytes
                for stage in projection.stages
                if stage.role == "lpddr_read"),
            206,
        )

    def test_tp8_context_one_token_jobs_use_only_sequence_parity_cards(self):
        layout = HBFParallelLayout.for_key("tp8_context")
        logical_bytes = qwen_logical_kv_bytes_per_token()
        per_head = logical_bytes // 4
        for token_start, expected_cards in (
            (0, {0, 2, 4, 6}),
            (1, {1, 3, 5, 7}),
        ):
            expected_vector = {
                card_id: (
                    per_head if card_id in expected_cards else 0)
                for card_id in range(8)
            }
            for job_type, builder in (
                (MigrationJob, build_migration_hbf_astra_projection),
                (AppendJob, build_append_hbf_astra_projection),
            ):
                with self.subTest(
                        token_start=token_start,
                        kind=job_type.__name__):
                    job = job_type(
                        job_id=10 + token_start,
                        session_id=f"context-{token_start}",
                        generation=1,
                        version=2,
                        group_id=0,
                        token_count=1,
                        logical_bytes=logical_bytes,
                        physical_bytes=logical_bytes,
                        per_card_bytes=per_head,
                        start_ns=0,
                        completion_ns=1,
                        token_start=token_start,
                    )
                    projection = builder(
                        job=job,
                        hardware=self.hardware,
                        layout=layout,
                        chunk_bytes=7_001,
                    )
                    self.assertEqual(
                        dict(projection.card_bytes), expected_vector)
                    self.assertEqual(
                        {
                            row.card_id
                            for row in projection.card_ledgers
                            if row.hbf_write_bytes
                        },
                        expected_cards,
                    )
                    self.assertEqual(
                        {
                            stage.card_id
                            for stage in projection.stages
                            if stage.card_id is not None
                        },
                        expected_cards,
                    )
                    self.assertEqual(
                        sum(
                            stage.tensor_bytes
                            for stage in projection.stages
                            if stage.role == "hbf_write"
                        ),
                        logical_bytes,
                    )
                    projection.validate()

    def test_tiny_transfer_omits_empty_card_streams_without_losing_bytes(self):
        job, projection = self.project(
            "tp4", logical_bytes=2, chunk_bytes=1)
        nonempty_cards = {
            row.card_id for row in projection.card_ledgers
            if row.hbf_write_bytes
        }
        write_cards = {
            stage.card_id for stage in projection.stages
            if stage.role == "hbf_write"
        }
        self.assertEqual(nonempty_cards, write_cards)
        self.assertEqual(
            sum(stage.tensor_bytes
                for stage in projection.stages
                if stage.role == "hbf_write"),
            job.physical_bytes,
        )
        projection.validate()

    def test_append_controller_descriptor_is_accepted(self):
        job, projection = self.project(
            "dp8", group_id=6, server_id=3)
        command = Controller.hbf_background_command(
            *projection.controller_command_arguments(999))
        prefix, job_id, arrival, encoded = command.split("\t")
        self.assertEqual(
            (prefix, job_id, arrival),
            ("hbf-background", "hbf-append.s3.r6.j9", "999"),
        )
        self.assertEqual(json.loads(encoded), projection.descriptor())
        self.assertEqual(projection.arrival_ns, job.start_ns)


class LifecycleAstraProjectionValidationTest(unittest.TestCase):
    def setUp(self):
        self.hardware = HBFServerHardware()
        self.layout = HBFParallelLayout.for_key("tp4")

    def test_job_replication_per_card_group_and_chunk_are_strict(self):
        job = migration_job("tp4")
        with self.assertRaisesRegex(
                HBFLifecycleAstraProjectionError, "physical_bytes"):
            build_migration_hbf_astra_projection(
                job=dataclasses.replace(
                    job, physical_bytes=job.physical_bytes + 1),
                hardware=self.hardware,
                layout=self.layout,
                chunk_bytes=10,
            )
        with self.assertRaisesRegex(
                HBFLifecycleAstraProjectionError, "per_card_bytes"):
            build_migration_hbf_astra_projection(
                job=dataclasses.replace(
                    job, per_card_bytes=job.per_card_bytes + 1),
                hardware=self.hardware,
                layout=self.layout,
                chunk_bytes=10,
            )
        with self.assertRaisesRegex(
                ValueError, "replica_id|group"):
            build_migration_hbf_astra_projection(
                job=dataclasses.replace(job, group_id=2),
                hardware=self.hardware,
                layout=self.layout,
                chunk_bytes=10,
            )
        with self.assertRaisesRegex(
                HBFLifecycleAstraProjectionError, "chunk_bytes"):
            build_migration_hbf_astra_projection(
                job=job,
                hardware=self.hardware,
                layout=self.layout,
                chunk_bytes=0,
            )
        with self.assertRaisesRegex(TypeError, "MigrationJob"):
            build_migration_hbf_astra_projection(
                job=append_job("tp4"),
                hardware=self.hardware,
                layout=self.layout,
                chunk_bytes=10,
            )

    def test_pathological_chunking_is_rejected_before_dag_materialization(self):
        with patch(
                "serving.core.hbf_full_model_lifecycle_astra."
                "ASTRA_BACKGROUND_STAGE_LIMIT",
                10):
            with self.assertRaisesRegex(
                    HBFLifecycleAstraProjectionError,
                    "increase chunk_bytes"):
                build_migration_hbf_astra_projection(
                    job=migration_job("tp4", logical_bytes=8),
                    hardware=self.hardware,
                    layout=self.layout,
                    chunk_bytes=1,
                )
            with self.assertRaisesRegex(
                    HBFLifecycleAstraProjectionError,
                    "increase chunk_bytes"):
                build_append_hbf_astra_projection(
                    job=append_job("tp4", logical_bytes=8),
                    hardware=self.hardware,
                    layout=self.layout,
                    chunk_bytes=1,
                )

    def test_validation_rejects_cycles_and_resource_escape(self):
        projection = build_append_hbf_astra_projection(
            job=append_job("tp4"),
            hardware=self.hardware,
            layout=self.layout,
            chunk_bytes=37,
            server_id=4,
        )
        first = projection.stages[0]
        paired_write_id = first.stage_id.replace(
            "lpddr-read", "hbf-write")
        cycled_first = dataclasses.replace(
            first, dependencies=(paired_write_id,))
        cycled = dataclasses.replace(
            projection,
            stages=(cycled_first, *projection.stages[1:]),
        )
        with self.assertRaisesRegex(
                HBFLifecycleAstraProjectionError, "cycle"):
            cycled.validate()

        escaped_first = dataclasses.replace(
            first,
            resources=("hbf-server:4:card:7:lpddr",),
            card_id=7,
        )
        escaped = dataclasses.replace(
            projection,
            stages=(escaped_first, *projection.stages[1:]),
        )
        with self.assertRaisesRegex(
                HBFLifecycleAstraProjectionError, "ownership"):
            escaped.validate()


if __name__ == "__main__":
    unittest.main()

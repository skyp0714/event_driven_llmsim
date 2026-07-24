import json
import unittest
from dataclasses import replace

from serving.core.controller import Controller
from serving.core.hbf_full_model_astra import (
    AGGREGATE_V1_LIMITATIONS,
    HBFModelAstraProjectionError,
    PROJECTION_FIDELITY,
    build_full_model_hbf_astra_projection,
    build_hbf_server_placement,
)
from serving.core.hbf_full_model_latency import (
    HBFModelBatchLatency,
    HBFParallelLayout,
    HBFServerHardware,
)


def latency_for(layout_key: str) -> HBFModelBatchLatency:
    layout = HBFParallelLayout.for_key(layout_key)
    if layout.tp_size == 1:
        collectives = (0, 0, 0)
        collective_bytes = 0
    else:
        collectives = (71, 53, 37)
        collective_bytes = 10_003
    local = {
        "embedding_ns": 11,
        "dense_ns": 101,
        "attention_ns": 211,
        "router_ns": 17,
        "moe_ns": 307,
        "final_ns": 29,
    }
    collective_ns = sum(collectives)
    return HBFModelBatchLatency(
        layout=layout.key,
        tp_size=layout.tp_size,
        replicas=layout.replicas,
        total_ns=sum(local.values()) + collective_ns,
        **local,
        collective_ns=collective_ns,
        tp_allreduce_ns=collectives[0],
        ep_allgather_ns=collectives[1],
        ep_reduce_scatter_ns=collectives[2],
        hbf_read_bytes_per_rank=1_000_003,
        lpddr_bytes_per_rank=200_007,
        collective_bytes_per_rank=collective_bytes,
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


class HBFServerPlacementTest(unittest.TestCase):
    def setUp(self):
        self.hardware = HBFServerHardware()

    def test_dp8_tp4_tp8_are_exhaustive_and_disjoint(self):
        expectations = {
            "dp8": tuple((card,) for card in range(8)),
            "tp4": ((0, 1, 2, 3), (4, 5, 6, 7)),
            "tp8": (tuple(range(8)),),
        }
        expected_roots = {
            "dp8": ((0,), (0,), (0,), (0,),
                    (1,), (1,), (1,), (1,)),
            "tp4": ((0,), (1,)),
            "tp8": ((0, 1),),
        }
        for key in expectations:
            with self.subTest(layout=key):
                placement = build_hbf_server_placement(
                    hardware=self.hardware,
                    layout=HBFParallelLayout.for_key(key),
                    server_id=7,
                )
                self.assertEqual(
                    tuple(group.card_ids for group in placement.groups),
                    expectations[key],
                )
                self.assertEqual(
                    tuple(group.pcie_root_ids for group in placement.groups),
                    expected_roots[key],
                )
                flattened = [
                    card
                    for group in placement.groups
                    for card in group.card_ids
                ]
                self.assertEqual(flattened, list(range(8)))
                self.assertEqual(len(flattened), len(set(flattened)))

    def test_layout_name_cannot_hide_a_different_parallel_shape(self):
        with self.assertRaisesRegex(
                HBFModelAstraProjectionError, "must be tp=4"):
            build_hbf_server_placement(
                hardware=self.hardware,
                layout=HBFParallelLayout(
                    key="tp4", tp_size=1, replicas=8),
            )
        with self.assertRaisesRegex(
                HBFModelAstraProjectionError, "layout key"):
            build_hbf_server_placement(
                hardware=self.hardware,
                layout=HBFParallelLayout(
                    key="two_tp4", tp_size=4, replicas=2),
            )


class HBFModelAstraProjectionTest(unittest.TestCase):
    def setUp(self):
        self.hardware = HBFServerHardware()

    def project(self, key, replica=0, server=0, batch=19):
        layout = HBFParallelLayout.for_key(key)
        latency = latency_for(key)
        projection = build_full_model_hbf_astra_projection(
            latency=latency,
            hardware=self.hardware,
            layout=layout,
            replica_id=replica,
            batch_id=batch,
            server_id=server,
        )
        return latency, projection

    def test_runtime_and_physical_byte_contracts_are_exact(self):
        for key in ("dp8", "tp4", "tp8"):
            with self.subTest(layout=key):
                latency, projection = self.project(key)
                tp = latency.tp_size
                self.assertEqual(
                    projection.dependency_critical_path_ns(),
                    latency.total_ns,
                )
                self.assertEqual(
                    sum(stage.hbf_read_bytes
                        for stage in projection.stages),
                    latency.hbf_read_bytes_per_rank * tp,
                )
                self.assertEqual(
                    sum(stage.lpddr_bytes
                        for stage in projection.stages),
                    latency.lpddr_bytes_per_rank * tp,
                )
                self.assertEqual(
                    sum(stage.collective_bytes
                        for stage in projection.stages),
                    latency.collective_bytes_per_rank * tp,
                )
                self.assertEqual(
                    sum(stage.tensor_bytes
                        for stage in projection.stages),
                    (
                        latency.hbf_read_bytes_per_rank
                        + latency.lpddr_bytes_per_rank
                        + latency.collective_bytes_per_rank
                    ) * tp,
                )

    def test_two_tp4_replicas_have_no_card_or_fabric_aliases(self):
        _, left = self.project("tp4", replica=0, server=3)
        _, right = self.project("tp4", replica=1, server=3)
        left_resources = {
            resource
            for stage in left.stages
            for resource in stage.resources
        }
        right_resources = {
            resource
            for stage in right.stages
            for resource in stage.resources
        }
        self.assertTrue(left_resources.isdisjoint(right_resources))
        self.assertTrue(all(
            ":card:4:" not in resource
            and ":card:5:" not in resource
            and ":card:6:" not in resource
            and ":card:7:" not in resource
            for resource in left_resources
        ))
        self.assertTrue(all(
            ":card:0:" not in resource
            and ":card:1:" not in resource
            and ":card:2:" not in resource
            and ":card:3:" not in resource
            for resource in right_resources
        ))

    def test_collective_is_a_whole_group_causal_barrier(self):
        _, projection = self.project("tp4")
        by_id = {
            stage.stage_id: stage for stage in projection.stages
        }
        first_collective = by_id[
            "batch:19:replica:0:collective:tp_allreduce"]
        self.assertEqual(
            set(first_collective.dependencies),
            {
                f"batch:19:replica:0:card:{card}:moe"
                for card in range(4)
            },
        )
        self.assertEqual(
            set(first_collective.resources),
            {
                "hbf-server:0:pcie-root:0",
                *{
                    f"hbf-server:0:card:{card}:pcie"
                    for card in range(4)
                },
            },
        )
        self.assertEqual(first_collective.pcie_route, "root_local")
        for card in range(4):
            self.assertEqual(
                by_id[
                    f"batch:19:replica:0:card:{card}:final"
                ].dependencies,
                ("batch:19:replica:0:collective:ep_reduce_scatter",),
            )

    def test_dp8_projection_never_mentions_an_unselected_card(self):
        _, projection = self.project("dp8", replica=6, server=4)
        resources = {
            resource
            for stage in projection.stages
            for resource in stage.resources
        }
        self.assertTrue(resources)
        self.assertTrue(all(
            resource.startswith("hbf-server:4:card:6:")
            for resource in resources
        ))
        self.assertFalse(any(
            ":collective:" in stage.stage_id
            for stage in projection.stages
        ))

    def test_controller_command_accepts_descriptor_without_schema_loss(self):
        latency, projection = self.project("tp8", server=2)
        job_id, arrival_ns, stages = (
            projection.controller_command_arguments(123))
        command = Controller.hbf_background_command(
            job_id, arrival_ns, stages)
        prefix, job_id, arrival, encoded = command.split("\t")
        descriptor = json.loads(encoded)
        self.assertEqual(
            (prefix, job_id, arrival),
            ("hbf-background", "hbf-model.s2.r0.b19", "123"),
        )
        self.assertEqual(descriptor, projection.descriptor())
        self.assertEqual(
            sum(stage["runtime_ns"] for stage in descriptor["stages"]),
            (
                latency.total_ns * latency.tp_size
                - latency.collective_ns * (latency.tp_size - 1)
            ),
        )
        self.assertEqual(
            set(descriptor["stages"][0]),
            {"id", "runtime_ns", "tensor_bytes", "resources", "deps"},
        )

    def test_job_and_stage_ids_are_deterministic_and_batch_scoped(self):
        _, first = self.project(
            "tp4", replica=1, server=8, batch=41)
        _, repeated = self.project(
            "tp4", replica=1, server=8, batch=41)
        _, successor = self.project(
            "tp4", replica=1, server=8, batch=42)
        self.assertEqual(first.job_id, "hbf-model.s8.r1.b41")
        self.assertEqual(first.job_id, repeated.job_id)
        self.assertEqual(first.descriptor_json(), repeated.descriptor_json())
        self.assertNotEqual(first.job_id, successor.job_id)
        self.assertTrue(all(
            stage.stage_id.startswith("batch:41:replica:1:")
            for stage in first.stages
        ))
        self.assertEqual(
            tuple(stage.resources for stage in first.stages),
            tuple(stage.resources for stage in successor.stages),
        )

    def test_projection_discloses_aggregate_v1_limitations(self):
        _, projection = self.project("tp4")
        audit = projection.audit_dict()
        self.assertEqual(audit["fidelity"], PROJECTION_FIDELITY)
        self.assertEqual(
            audit["limitations"], list(AGGREGATE_V1_LIMITATIONS))
        self.assertIn("collectives use analytical", audit["limitations"][1])
        self.assertEqual(
            json.loads(projection.descriptor_json()),
            projection.descriptor(),
        )

    def test_source_metadata_and_accounting_are_fail_closed(self):
        layout = HBFParallelLayout.for_key("tp4")
        valid = latency_for("tp4")
        bad_cases = (
            (
                replace(valid, layout="tp8"),
                "layout metadata",
            ),
            (
                replace(valid, total_ns=valid.total_ns + 1),
                "sum of aggregate components",
            ),
            (
                replace(
                    valid,
                    collective_ns=valid.collective_ns + 1,
                    total_ns=valid.total_ns + 1,
                ),
                "collective component runtimes",
            ),
            (
                replace(valid, hbf_read_bytes_per_rank=-1),
                "non-negative integer",
            ),
        )
        for latency, message in bad_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                        HBFModelAstraProjectionError, message):
                    build_full_model_hbf_astra_projection(
                        latency=latency,
                        hardware=self.hardware,
                        layout=layout,
                        replica_id=0,
                        batch_id=19,
                    )

    def test_invalid_replica_is_rejected_before_descriptor_creation(self):
        with self.assertRaisesRegex(
                HBFModelAstraProjectionError, "outside"):
            self.project("tp4", replica=2)


if __name__ == "__main__":
    unittest.main()

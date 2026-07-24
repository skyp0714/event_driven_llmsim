import dataclasses
import os
from pathlib import Path
import unittest

from serving.core.hbf_astra_runner import (
    AstraHBFRunConfig,
    PersistentHBFAstraRunner,
)
from serving.core.hbf_full_model_astra import (
    HBFAstraTimingAccounting,
    build_hbf_server_placement,
    build_ordered_full_model_hbf_astra_projection,
)
from serving.core.hbf_full_model_latency import (
    HBFCollectiveExecutionOp,
    HBFModelBatchShape,
    HBFParallelLayout,
    HBFServerHardware,
    build_full_model_hbf_latency,
    load_hbf_server_config,
)
from serving.core.hbf_full_model_lifecycle import (
    MigrationJob,
    canonical_card_bytes,
    hbf_kv_range_card_bytes,
)
from serving.core.hbf_full_model_lifecycle_astra import (
    build_migration_hbf_astra_projection,
)
from serving.core.hbf_pcie_topology import HBFPCIeTopologyError


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "configs/wakekv_hbf/full_model_8card_server.json"
)


def migration_job(
        layout_key: str, *, group_id: int, logical_bytes: int = 4096,
) -> MigrationJob:
    layout = HBFParallelLayout.for_key(layout_key)
    physical = (
        logical_bytes * layout.physical_kv_replication_factor)
    card_ids = tuple(range(
        group_id * layout.tp_size,
        (group_id + 1) * layout.tp_size,
    ))
    if layout.is_context_striped:
        by_card = hbf_kv_range_card_bytes(
            layout=layout,
            card_ids=card_ids,
            kv_bytes_per_token=logical_bytes,
            token_start=0,
            token_count=1,
        )
    else:
        quotient, remainder = divmod(physical, layout.tp_size)
        by_card = {
            card_id: quotient + (1 if index < remainder else 0)
            for index, card_id in enumerate(card_ids)
        }
    return MigrationJob(
        job_id=17 + group_id,
        session_id=f"session-{layout_key}-{group_id}",
        generation=0,
        version=1,
        group_id=group_id,
        token_count=1,
        logical_bytes=logical_bytes,
        physical_bytes=physical,
        per_card_bytes=max(by_card.values()),
        start_ns=0,
        completion_ns=1,
        card_bytes=canonical_card_bytes(card_ids, by_card),
    )


class HBFSharedPCIeTopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware, cls.configured_layouts = load_hbf_server_config(
            CONFIG)
        cls.shape = HBFModelBatchShape(
            total_tokens=1,
            decode_hbf_k=(125_000,),
            decode_lpddr_k=(64,),
        )

    def model_projection(
            self, layout_key: str, *, hardware=None,
            replica_id: int = 0):
        hardware = self.hardware if hardware is None else hardware
        layout = HBFParallelLayout.for_key(layout_key)
        model = build_full_model_hbf_latency(
            repo_root=REPO_ROOT,
            hardware=hardware,
            layout=layout,
        )
        plan = model.batch_execution_plan(self.shape)
        projection = build_ordered_full_model_hbf_astra_projection(
            plan=plan,
            hardware=hardware,
            layout=layout,
            replica_id=replica_id,
            batch_id=31,
            server_id=5,
        )
        return plan, projection

    def migration_projection(
            self, layout_key: str, *, hardware=None,
            group_id: int = 0, logical_bytes: int = 4096,
            chunk_bytes: int = 1024):
        hardware = self.hardware if hardware is None else hardware
        return build_migration_hbf_astra_projection(
            job=migration_job(
                layout_key,
                group_id=group_id,
                logical_bytes=logical_bytes,
            ),
            hardware=hardware,
            layout=HBFParallelLayout.for_key(layout_key),
            chunk_bytes=chunk_bytes,
            server_id=5,
        )

    @staticmethod
    def collective_stages(plan, projection):
        operations = {
            index: operation
            for index, operation in enumerate(plan.operations)
            if isinstance(operation, HBFCollectiveExecutionOp)
        }
        return [
            (operations[stage.operation_index], stage)
            for stage in projection.stages
            if stage.operation_index in operations
        ]

    def test_publication_config_declares_complete_shared_topology(self):
        topology = build_hbf_server_placement(
            hardware=self.hardware,
            layout=self.configured_layouts["tp8_context"],
            server_id=5,
        ).pcie_topology
        self.assertEqual(topology.resource_mode, "shared")
        self.assertEqual(
            topology.card_to_root,
            (0, 0, 0, 0, 1, 1, 1, 1),
        )
        self.assertEqual(topology.nic_to_root, (0,))
        self.assertEqual(topology.p2p_mode, "cross_root")
        self.assertEqual(topology.root_bandwidth_gbps, 200.0)
        self.assertEqual(topology.inter_root_bandwidth_gbps, 100.0)
        self.assertIn("pcie_topology", build_hbf_server_placement(
            hardware=self.hardware,
            layout=self.configured_layouts["tp4"],
            server_id=5,
        ).as_dict())

    def test_tp4_replicas_are_root_local_independent_and_share_migration_links(self):
        _, left = self.model_projection("tp4", replica_id=0)
        _, right = self.model_projection("tp4", replica_id=1)
        left_collective = next(
            stage for stage in left.stages
            if stage.operation_kind == "collective")
        right_collective = next(
            stage for stage in right.stages
            if stage.operation_kind == "collective")
        self.assertEqual(left_collective.pcie_route, "root_local")
        self.assertEqual(right_collective.pcie_route, "root_local")
        self.assertIn(
            "hbf-server:5:pcie-root:0",
            left_collective.resources,
        )
        self.assertIn(
            "hbf-server:5:pcie-root:1",
            right_collective.resources,
        )
        self.assertTrue(
            set(left_collective.resources).isdisjoint(
                right_collective.resources))

        left_migration = self.migration_projection(
            "tp4", group_id=0)
        migration_resources = {
            resource
            for stage in left_migration.stages
            for resource in stage.resources
        }
        shared = (
            set(left_collective.resources) & migration_resources)
        self.assertIn("hbf-server:5:pcie-root:0", shared)
        self.assertTrue(any(
            resource.endswith(":pcie") for resource in shared))

        right_migration = self.migration_projection(
            "tp4", group_id=1)
        right_root_stage = next(
            stage for stage in right_migration.stages
            if stage.role == "destination_pcie_root")
        self.assertEqual(
            set(right_root_stage.resources),
            {
                "hbf-server:5:pcie-root:0",
                "hbf-server:5:pcie-inter-root:0-1",
                "hbf-server:5:pcie-root:1",
            },
        )

    def test_tp8_collectives_cross_roots_but_context_pairs_do_not(self):
        tp8_plan, tp8 = self.model_projection("tp8")
        for _, stage in self.collective_stages(tp8_plan, tp8):
            self.assertEqual(stage.pcie_route, "cross_root")
            self.assertIn(
                "hbf-server:5:pcie-inter-root:0-1",
                stage.resources,
            )

        context_plan, context = self.model_projection("tp8_context")
        pairs = []
        standard = []
        for operation, stage in self.collective_stages(
                context_plan, context):
            if operation.collective_type.startswith("PAIR_"):
                pairs.append(stage)
            else:
                standard.append(stage)
        self.assertTrue(pairs)
        self.assertTrue(standard)
        self.assertTrue(all(
            stage.pcie_route == "root_local"
            and not any(
                "pcie-inter-root" in resource
                for resource in stage.resources
            )
            for stage in pairs
        ))
        self.assertTrue(all(
            stage.pcie_route == "cross_root"
            and any(
                "pcie-inter-root" in resource
                for resource in stage.resources
            )
            for stage in standard
        ))

    def test_foreground_has_parity_but_shared_migration_has_internal_wait(self):
        legacy = dataclasses.replace(
            self.hardware,
            pcie_resource_mode="legacy_isolated",
        )
        shared_plan, shared_model = self.model_projection("tp8")
        legacy_plan, legacy_model = self.model_projection(
            "tp8", hardware=legacy)
        self.assertEqual(shared_plan.total_ns, legacy_plan.total_ns)
        self.assertEqual(
            shared_model.dependency_critical_path_ns(),
            legacy_model.dependency_critical_path_ns(),
        )
        self.assertEqual(
            shared_model.solo_resource_serialized_completion_ns(),
            shared_model.dependency_critical_path_ns(),
        )
        self.assertEqual(
            legacy_model.solo_resource_serialized_completion_ns(),
            legacy_model.dependency_critical_path_ns(),
        )
        self.assertEqual(
            shared_model.physical_collective_bytes,
            legacy_model.physical_collective_bytes,
        )
        self.assertNotEqual(
            tuple(stage.resources for stage in shared_model.stages),
            tuple(stage.resources for stage in legacy_model.stages),
        )

        shared_migration = self.migration_projection("tp8")
        legacy_migration = self.migration_projection(
            "tp8", hardware=legacy)
        self.assertEqual(
            shared_migration.dependency_critical_path_ns(),
            legacy_migration.dependency_critical_path_ns(),
        )
        self.assertGreater(
            shared_migration.solo_resource_serialized_completion_ns(),
            shared_migration.dependency_critical_path_ns(),
        )
        self.assertEqual(
            legacy_migration.solo_resource_serialized_completion_ns(),
            legacy_migration.dependency_critical_path_ns(),
        )
        self.assertEqual(
            shared_migration.byte_ledger,
            legacy_migration.byte_ledger,
        )
        timing = shared_migration.audit_dict()["timing_contract"]
        self.assertGreater(
            timing["solo_internal_resource_serialization_wait_ns"], 0)

    def test_inter_root_bandwidth_changes_only_cross_root_critical_path(self):
        slow_inter_root = dataclasses.replace(
            self.hardware,
            pcie_inter_root_bandwidth_gbps=1.0,
        )
        base_tp8, _ = self.model_projection("tp8")
        slow_tp8, _ = self.model_projection(
            "tp8", hardware=slow_inter_root)
        self.assertGreater(slow_tp8.total_ns, base_tp8.total_ns)

        base_tp4, _ = self.model_projection("tp4")
        slow_tp4, _ = self.model_projection(
            "tp4", hardware=slow_inter_root)
        self.assertEqual(slow_tp4.total_ns, base_tp4.total_ns)

        base_cross_migration = self.migration_projection(
            "tp4", group_id=1)
        slow_cross_migration = self.migration_projection(
            "tp4",
            group_id=1,
            hardware=dataclasses.replace(
                slow_inter_root,
                pcie_inter_root_fixed_latency_us=7.0,
            ),
        )
        self.assertGreater(
            slow_cross_migration.dependency_critical_path_ns(),
            base_cross_migration.dependency_critical_path_ns(),
        )

    def test_foreground_and_lifecycle_byte_conservation_survives_bridge(self):
        _, foreground = self.model_projection("tp8_context")
        migration = self.migration_projection("tp8_context")
        self.assertEqual(
            sum(stage.tensor_bytes for stage in foreground.stages),
            (
                foreground.physical_hbf_read_bytes
                + foreground.physical_lpddr_bytes
                + foreground.physical_collective_bytes
            ),
        )
        self.assertEqual(
            sum(
                stage.tensor_bytes for stage in migration.stages
                if stage.role == "destination_pcie_root"
            ),
            migration.physical_bytes,
        )
        self.assertEqual(
            sum(
                stage.tensor_bytes for stage in migration.stages
                if stage.role == "destination_pcie_card"
            ),
            migration.physical_bytes,
        )
        self.assertEqual(
            sum(
                stage.tensor_bytes for stage in migration.stages
                if stage.role == "hbf_write"
            ),
            migration.physical_bytes,
        )

    def test_invalid_or_ambiguous_topologies_fail_closed(self):
        bad_nic = dataclasses.replace(
            self.hardware,
            pcie_nic_to_root=(2,),
        )
        with self.assertRaisesRegex(
                HBFPCIeTopologyError, "unknown root"):
            bad_nic.validate()

        interleaved = dataclasses.replace(
            self.hardware,
            pcie_card_to_root=(0, 1, 0, 1, 0, 1, 0, 1),
        )
        with self.assertRaisesRegex(
                HBFPCIeTopologyError, "TP4 replica"):
            build_hbf_server_placement(
                hardware=interleaved,
                layout=HBFParallelLayout.for_key("tp4"),
            )

        cross_root_disabled = dataclasses.replace(
            self.hardware,
            pcie_p2p_mode="same_root_only",
        )
        with self.assertRaisesRegex(
                HBFPCIeTopologyError, "cross-root"):
            build_hbf_server_placement(
                hardware=cross_root_disabled,
                layout=HBFParallelLayout.for_key("tp8"),
            )

        unknown_mode = dataclasses.replace(
            self.hardware,
            pcie_resource_mode="implicit_magic",
        )
        with self.assertRaisesRegex(
                HBFPCIeTopologyError, "pcie_resource_mode"):
            unknown_mode.validate()

        mutable_mapping = dataclasses.replace(
            self.hardware,
            pcie_card_to_root=[0, 0, 0, 0, 1, 1, 1, 1],
        )
        with self.assertRaisesRegex(ValueError, "immutable tuple"):
            mutable_mapping.validate()

    def test_four_root_topology_is_rejected_for_every_layout(self):
        four_roots = dataclasses.replace(
            self.hardware,
            pcie_root_count=4,
            cards_per_pcie_root=2,
            pcie_card_to_root=(0, 0, 1, 1, 2, 2, 3, 3),
            pcie_nic_to_root=(0,),
        )
        for layout_key in ("dp8", "tp4", "tp8", "tp8_context"):
            with self.subTest(layout=layout_key):
                with self.assertRaisesRegex(
                        HBFPCIeTopologyError, "exactly two roots"):
                    build_hbf_server_placement(
                        hardware=four_roots,
                        layout=HBFParallelLayout.for_key(layout_key),
                    )

    def test_live_astra_matches_solo_internal_serialization(self):
        default_binary = (
            REPO_ROOT
            / "astra-sim/build/astra_analytical/build/bin"
            / "AstraSim_Analytical_Congestion_Aware"
        )
        binary = Path(os.environ.get(
            "LLMSIM_ASTRA_BINARY", default_binary))
        default_chakra = (
            REPO_ROOT
            / "astra-sim/extern/graph_frontend/chakra"
        )
        chakra = Path(os.environ.get(
            "LLMSIM_CHAKRA_ROOT", default_chakra))
        if not binary.is_file() or not chakra.is_dir():
            self.skipTest(
                "built congestion-aware ASTRA and Chakra are unavailable")

        observed = {}
        projected = {}
        for mode in ("legacy_isolated", "shared"):
            hardware = dataclasses.replace(
                self.hardware, pcie_resource_mode=mode)
            projection = self.migration_projection(
                "tp8",
                hardware=hardware,
                logical_bytes=64 * 1024 ** 2,
                chunk_bytes=16 * 1024 ** 2,
            )
            with PersistentHBFAstraRunner(
                num_npus=8,
                hbf_num_devices=8,
                repo_root=REPO_ROOT,
                binary_path=binary,
                chakra_root=chakra,
                config=AstraHBFRunConfig(timeout_seconds=30.0),
            ) as session:
                completion = session.submit_projection(
                    job_id=f"single-migration-{mode}",
                    arrival_ns=session.current_ns,
                    projection=projection,
                )
            observed[mode] = completion.elapsed_cycles
            projected[mode] = (
                projection.solo_resource_serialized_completion_ns())
            self.assertEqual(observed[mode], projected[mode])
            accounting = HBFAstraTimingAccounting(
                dependency_critical_path_ns=(
                    projection.dependency_critical_path_ns()),
                solo_resource_serialized_completion_ns=projected[mode],
                actual_resource_serialized_completion_ns=observed[mode],
            )
            self.assertGreaterEqual(
                observed[mode],
                projection.dependency_critical_path_ns(),
            )
            self.assertEqual(
                accounting.resource_delay_ns,
                (
                    accounting.internal_resource_serialization_wait_ns
                    + accounting.signed_interference_delta_ns
                ),
            )

        self.assertGreater(
            observed["shared"], observed["legacy_isolated"])
        self.assertEqual(projected["legacy_isolated"], 1_196_932)
        self.assertEqual(projected["shared"], 1_406_651)


if __name__ == "__main__":
    unittest.main()

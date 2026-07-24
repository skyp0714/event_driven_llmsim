import dataclasses
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from serving.core.controller import Controller
import serving.core.hbf_full_model_astra as astra_projection
from serving.core.hbf_astra_runner import (
    AstraHBFRunConfig,
    PersistentHBFAstraRunner,
)
from serving.core.hbf_full_model_astra import (
    HBFModelAstraProjectionError,
    ORDERED_V2_FIDELITY,
    ORDERED_V2_LIMITATIONS,
    ORDERED_V2_SCHEMA,
    ORDERED_V2_STAGE_LIMIT,
    PROJECTION_SCHEMA,
    build_full_model_hbf_astra_projection,
    build_hbf_server_placement,
    build_ordered_full_model_hbf_astra_projection,
)
from serving.core.hbf_full_model_latency import (
    HBFCollectiveExecutionOp,
    HBFKernelExecutionOp,
    HBFModelBatchShape,
    HBFParallelLayout,
    HBFServerHardware,
    build_full_model_hbf_latency,
)
from serving.core.hbf_full_model_lifecycle import AppendJob
from serving.core.hbf_full_model_lifecycle_astra import (
    build_append_hbf_astra_projection,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class OrderedFullModelHBFProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HBFServerHardware()
        cls.layouts = {
            key: HBFParallelLayout.for_key(key)
            for key in ("dp8", "tp4", "tp8", "tp8_context")
        }
        cls.models = {
            key: build_full_model_hbf_latency(
                repo_root=REPO_ROOT,
                hardware=cls.hardware,
                layout=layout,
            )
            for key, layout in cls.layouts.items()
        }
        cls.regular_shape = HBFModelBatchShape(
            total_tokens=4,
            prefill_q=(3,),
            prefill_hbf_k=(100,),
            prefill_lpddr_k=(2,),
            decode_hbf_k=(51,),
            decode_lpddr_k=(4,),
            lm_head_sequences=2,
        )
        # One committed token lives on even pair-rank 0; the current causal
        # token lives on odd pair-rank 1.  This makes both HBF and LPDDR
        # ledgers asymmetric while retaining a physically unique KV copy.
        cls.odd_context_shape = HBFModelBatchShape(
            total_tokens=1,
            prefill_q=(1,),
            prefill_hbf_k=(1,),
            prefill_lpddr_k=(0,),
        )

    def ordered(self, key, shape=None, *, with_latency=True):
        model = self.models[key]
        layout = self.layouts[key]
        shape = self.regular_shape if shape is None else shape
        plan = model.batch_execution_plan(shape)
        latency = model.batch_latency(shape)
        projection = build_ordered_full_model_hbf_astra_projection(
            plan=plan,
            latency=(latency if with_latency else None),
            hardware=self.hardware,
            layout=layout,
            replica_id=0,
            batch_id=71,
            server_id=3,
        )
        return plan, latency, projection

    def test_legacy_aggregate_v1_results_remain_unchanged(self):
        expected_stage_counts = {
            "dp8": 6,
            "tp4": 27,
            "tp8": 51,
        }
        for key in ("dp8", "tp4", "tp8"):
            with self.subTest(layout=key):
                model = self.models[key]
                layout = self.layouts[key]
                latency = model.batch_latency(self.regular_shape)
                projection = build_full_model_hbf_astra_projection(
                    latency=latency,
                    hardware=self.hardware,
                    layout=layout,
                    replica_id=0,
                    batch_id=11,
                )
                self.assertEqual(projection.schema, PROJECTION_SCHEMA)
                self.assertEqual(
                    len(projection.stages), expected_stage_counts[key])
                self.assertEqual(
                    projection.dependency_critical_path_ns(),
                    latency.total_ns,
                )
                self.assertEqual(
                    projection.physical_hbf_read_bytes,
                    latency.hbf_read_bytes_per_rank * layout.tp_size,
                )
                self.assertEqual(
                    projection.physical_lpddr_bytes,
                    latency.lpddr_bytes_per_rank * layout.tp_size,
                )
                self.assertEqual(
                    projection.physical_collective_bytes,
                    latency.collective_bytes_per_rank * layout.tp_size,
                )

    def test_context_placement_and_lifecycle_use_unique_tp8_contract(self):
        layout = self.layouts["tp8_context"]
        placement = build_hbf_server_placement(
            hardware=self.hardware,
            layout=layout,
            server_id=9,
        )
        self.assertEqual(placement.layout, "tp8_context")
        self.assertEqual(
            placement.groups[0].card_ids, tuple(range(8)))

        job = AppendJob(
            job_id=1,
            session_id="odd-context",
            generation=0,
            version=1,
            group_id=0,
            token_count=1,
            logical_bytes=17,
            physical_bytes=17,
            per_card_bytes=5,
            start_ns=0,
            completion_ns=1,
        )
        lifecycle = build_append_hbf_astra_projection(
            job=job,
            hardware=self.hardware,
            layout=layout,
            chunk_bytes=64,
            server_id=9,
        )
        self.assertEqual(lifecycle.placement.layout, "tp8_context")
        self.assertEqual(lifecycle.replica.card_ids, tuple(range(8)))
        self.assertEqual(
            lifecycle.byte_ledger.hbf_write_bytes, 17)
        self.assertEqual(
            sum(card.hbf_write_bytes for card in lifecycle.card_ledgers),
            17,
        )
        self.assertEqual(
            dict(lifecycle.card_bytes),
            {0: 5, 1: 0, 2: 4, 3: 0,
             4: 4, 5: 0, 6: 4, 7: 0},
        )
        self.assertEqual(
            {
                stage.card_id
                for stage in lifecycle.stages
                if stage.card_id is not None
            },
            {0, 2, 4, 6},
        )

    def test_every_operation_expands_in_order_with_exact_barriers(self):
        plan, _, projection = self.ordered("tp8_context")
        self.assertEqual(projection.schema, ORDERED_V2_SCHEMA)
        self.assertEqual(projection.fidelity, ORDERED_V2_FIDELITY)
        self.assertEqual(
            projection.audit_dict()["limitations"],
            list(ORDERED_V2_LIMITATIONS),
        )
        self.assertIn(
            "native ASTRA COMM_COLL",
            projection.audit_dict()["limitations"][0],
        )

        grouped = {index: [] for index in range(len(plan.operations))}
        for stage in projection.stages:
            self.assertIsNotNone(stage.operation_index)
            grouped[stage.operation_index].append(stage)
        self.assertTrue(all(grouped.values()))

        tails = {card_id: None for card_id in range(8)}
        for index, operation in enumerate(plan.operations):
            stages = grouped[index]
            self.assertEqual(
                {stage.operation_name for stage in stages},
                {operation.name},
            )
            if isinstance(operation, HBFKernelExecutionOp):
                self.assertEqual(
                    tuple(stage.card_id for stage in stages),
                    tuple(range(8)),
                )
                for stage in stages:
                    expected_deps = (
                        (tails[stage.card_id],)
                        if tails[stage.card_id] is not None else ()
                    )
                    self.assertEqual(stage.dependencies, expected_deps)
                    self.assertIn(
                        f"hbf-server:3:card:{stage.card_id}:npu",
                        stage.resources,
                    )
                    self.assertEqual(
                        any(resource.endswith(":hbf-read")
                            for resource in stage.resources),
                        stage.hbf_read_bytes > 0,
                    )
                    self.assertEqual(
                        any(resource.endswith(":lpddr")
                            for resource in stage.resources),
                        stage.lpddr_bytes > 0,
                    )
                    tails[stage.card_id] = stage.stage_id
            else:
                self.assertIsInstance(
                    operation, HBFCollectiveExecutionOp)
                self.assertEqual(len(stages), 1)
                stage = stages[0]
                self.assertIsNone(stage.card_id)
                self.assertEqual(
                    stage.dependencies,
                    tuple(tails[card_id] for card_id in range(8)),
                )
                self.assertEqual(
                    set(stage.resources),
                    {
                        "hbf-server:3:pcie-root:0",
                        "hbf-server:3:pcie-root:1",
                        *{
                            f"hbf-server:3:card:{card_id}:pcie"
                            for card_id in range(8)
                        },
                        *(
                            set()
                            if operation.collective_type.startswith("PAIR_")
                            else {
                                "hbf-server:3:"
                                "pcie-inter-root:0-1"
                            }
                        ),
                    },
                )
                self.assertEqual(
                    stage.pcie_route,
                    (
                        "root_local"
                        if operation.collective_type.startswith("PAIR_")
                        else "cross_root"
                    ),
                )
                for card_id in tails:
                    tails[card_id] = stage.stage_id

        expected_stage_count = (
            len(plan.kernel_operations) * 8
            + len(plan.collective_operations)
        )
        self.assertEqual(len(projection.stages), expected_stage_count)
        self.assertLess(len(projection.stages), ORDERED_V2_STAGE_LIMIT)
        self.assertEqual(
            projection.dependency_critical_path_ns(),
            plan.total_ns,
        )

    def test_odd_context_attention_uses_exact_card_runtime_and_bytes(self):
        plan, _, projection = self.ordered(
            "tp8_context", self.odd_context_shape)
        ranks = {
            rank.pair_rank: rank
            for rank in plan.context_attention_rank_executions
        }
        attention_indices = {
            index
            for index, operation in enumerate(plan.operations)
            if (
                isinstance(operation, HBFKernelExecutionOp)
                and operation.category == "attention"
            )
        }
        attention_stages = [
            stage
            for stage in projection.stages
            if stage.operation_index in attention_indices
        ]
        self.assertEqual(len(attention_indices), 48)
        self.assertEqual(len(attention_stages), 48 * 8)
        for stage in attention_stages:
            expected = ranks[stage.card_id % 2]
            self.assertEqual(stage.runtime_ns, expected.latency_ns)
            self.assertEqual(
                stage.hbf_read_bytes, expected.hbf_read_bytes)
            self.assertEqual(
                stage.lpddr_bytes, expected.lpddr_bytes)

        expected_attention_hbf = (
            plan.context_attention_physical_hbf_read_bytes_per_layer
            * len(attention_indices)
        )
        expected_attention_lpddr = (
            plan.context_attention_physical_lpddr_bytes_per_layer
            * len(attention_indices)
        )
        self.assertEqual(
            sum(stage.hbf_read_bytes for stage in attention_stages),
            expected_attention_hbf,
        )
        self.assertEqual(
            sum(stage.lpddr_bytes for stage in attention_stages),
            expected_attention_lpddr,
        )

        scalar_attention = next(
            operation
            for operation in plan.kernel_operations
            if operation.category == "attention"
        )
        self.assertNotEqual(
            scalar_attention.hbf_read_bytes_per_rank * 8 * 48,
            expected_attention_hbf,
        )
        self.assertNotEqual(
            scalar_attention.lpddr_bytes_per_rank * 8 * 48,
            expected_attention_lpddr,
        )
        self.assertEqual(
            projection.physical_hbf_read_bytes,
            sum(stage.hbf_read_bytes for stage in projection.stages),
        )
        self.assertEqual(
            projection.physical_lpddr_bytes,
            sum(stage.lpddr_bytes for stage in projection.stages),
        )
        self.assertEqual(
            projection.physical_collective_bytes,
            sum(stage.collective_bytes for stage in projection.stages),
        )
        self.assertEqual(
            projection.dependency_critical_path_ns(),
            plan.total_ns,
        )

    def test_conventional_ordered_ledgers_equal_scalar_times_tp(self):
        for key in ("dp8", "tp4", "tp8"):
            with self.subTest(layout=key):
                plan, latency, projection = self.ordered(key)
                tp = self.layouts[key].tp_size
                self.assertEqual(
                    projection.physical_hbf_read_bytes,
                    latency.hbf_read_bytes_per_rank * tp,
                )
                self.assertEqual(
                    projection.physical_lpddr_bytes,
                    latency.lpddr_bytes_per_rank * tp,
                )
                self.assertEqual(
                    projection.physical_collective_bytes,
                    latency.collective_bytes_per_rank * tp,
                )
                self.assertEqual(
                    projection.dependency_critical_path_ns(),
                    plan.total_ns,
                )

    def test_controller_accepts_full_ordered_descriptor(self):
        plan, _, projection = self.ordered(
            "tp8_context", self.odd_context_shape,
            with_latency=False,
        )
        job_id, arrival_ns, stages = (
            projection.controller_command_arguments(1234))
        command = Controller.hbf_background_command(
            job_id, arrival_ns, stages)
        prefix, encoded_job, encoded_arrival, descriptor_json = (
            command.split("\t"))
        self.assertEqual(prefix, "hbf-background")
        self.assertEqual(encoded_job, "hbf-model.s3.r0.b71")
        self.assertEqual(encoded_arrival, "1234")
        descriptor = json.loads(descriptor_json)
        self.assertEqual(descriptor, projection.descriptor())
        self.assertEqual(len(descriptor["stages"]), len(projection.stages))
        self.assertEqual(
            projection.source_plan_operation_count,
            len(plan.operations),
        )
        self.assertEqual(
            set(descriptor["stages"][0]),
            {"id", "runtime_ns", "tensor_bytes", "resources", "deps"},
        )

    def test_ordered_context_completes_on_persistent_astra(self):
        plan, _, projection = self.ordered(
            "tp8_context", self.odd_context_shape,
            with_latency=False,
        )
        with PersistentHBFAstraRunner(
                num_npus=8,
                hbf_num_devices=8,
                repo_root=REPO_ROOT,
                config=AstraHBFRunConfig(timeout_seconds=30.0),
        ) as session:
            completion = session.submit_projection(
                job_id="ordered-v2-context.71",
                arrival_ns=session.current_ns,
                projection=projection,
            )
            self.assertEqual(
                completion.elapsed_cycles, plan.total_ns)
            self.assertEqual(
                completion.stage_count, len(projection.stages))
            self.assertEqual(
                completion.projection_schema, ORDERED_V2_SCHEMA)
            self.assertTrue(completion.astra_cycles_used)

    def test_stage_limit_and_source_audit_fail_closed(self):
        model = self.models["tp4"]
        layout = self.layouts["tp4"]
        plan = model.batch_execution_plan(self.regular_shape)
        latency = model.batch_latency(self.regular_shape)
        with patch.object(
                astra_projection, "ORDERED_V2_STAGE_LIMIT", 10):
            with self.assertRaisesRegex(
                    HBFModelAstraProjectionError, "stage"):
                build_ordered_full_model_hbf_astra_projection(
                    plan=plan,
                    hardware=self.hardware,
                    layout=layout,
                    replica_id=0,
                    batch_id=1,
                )
        with self.assertRaisesRegex(
                HBFModelAstraProjectionError, "byte audit"):
            build_ordered_full_model_hbf_astra_projection(
                plan=plan,
                latency=dataclasses.replace(
                    latency,
                    hbf_read_bytes_per_rank=(
                        latency.hbf_read_bytes_per_rank + 1
                    ),
                ),
                hardware=self.hardware,
                layout=layout,
                replica_id=0,
                batch_id=1,
            )

    def test_aggregate_v1_rejects_asymmetric_context_bytes(self):
        model = self.models["tp8_context"]
        layout = self.layouts["tp8_context"]
        latency = model.batch_latency(self.odd_context_shape)
        with self.assertRaisesRegex(
                HBFModelAstraProjectionError, "ordered"):
            build_full_model_hbf_astra_projection(
                latency=latency,
                hardware=self.hardware,
                layout=layout,
                replica_id=0,
                batch_id=1,
            )


if __name__ == "__main__":
    unittest.main()

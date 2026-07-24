import dataclasses
import math
from pathlib import Path
import unittest

from serving.core.hbf_full_model_latency import (
    HBFCollectiveExecutionOp,
    HBFKernelExecutionOp,
    HBFModelBatchShape,
    HBFParallelLayout,
    build_full_model_hbf_latency,
    load_hbf_server_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT / "configs" / "wakekv_hbf"
    / "full_model_8card_server.json"
)


class FullModelHBFLatencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware, cls.layouts = load_hbf_server_config(CONFIG)
        cls.models = {
            key: build_full_model_hbf_latency(
                repo_root=REPO_ROOT,
                hardware=cls.hardware,
                layout=layout,
            )
            for key, layout in cls.layouts.items()
        }

    def test_layouts_use_exactly_eight_cards(self):
        self.assertEqual(
            set(self.layouts),
            {"dp8", "tp4", "tp8", "tp8_context"},
        )
        for layout in self.layouts.values():
            self.assertEqual(layout.tp_size * layout.replicas, 8)
        self.assertEqual(
            self.layouts["dp8"].physical_kv_replication_factor, 1)
        self.assertEqual(
            self.layouts["tp4"].physical_kv_replication_factor, 1)
        self.assertEqual(
            self.layouts["tp8"].physical_kv_replication_factor, 2)
        self.assertEqual(
            self.layouts["tp8_context"].physical_kv_replication_factor, 1)

    def test_invalid_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "every HBF card"):
            HBFParallelLayout("bad", 4, 1).validate(8)

    def test_batch_media_vectors_must_align(self):
        shape = HBFModelBatchShape(
            total_tokens=4,
            prefill_q=(4,),
            prefill_hbf_k=(),
            prefill_lpddr_k=(0,),
        )
        with self.assertRaisesRegex(ValueError, "must align"):
            shape.validate()

    def test_collectives_are_charged_only_for_tp_layouts(self):
        shape = HBFModelBatchShape(
            total_tokens=1,
            decode_hbf_k=(125_000,),
            decode_lpddr_k=(500,),
        )
        rows = {
            key: model.batch_latency(shape)
            for key, model in self.models.items()
        }
        self.assertEqual(rows["dp8"].collective_ns, 0)
        self.assertEqual(rows["dp8"].collective_bytes_per_rank, 0)
        self.assertGreater(rows["tp4"].collective_ns, 0)
        self.assertGreater(rows["tp8"].collective_ns, 0)
        self.assertGreater(
            rows["tp8"].collective_ns, rows["tp4"].collective_ns)

        tp = 4
        layers = 48
        fixed_seconds = (
            self.hardware.intra_fabric_fixed_latency_us * 1e-6)
        bandwidth = (
            self.hardware.intra_fabric_bandwidth_gbps_per_card * 1e9)
        hidden_payload = 1 * 2_048 * 2
        dispatch_local_chunk = 1 * (2_048 + 128) * 2
        expected_allreduce_per_layer = math.ceil(1e9 * (
            fixed_seconds
            + 2 * (tp - 1) / tp * hidden_payload / bandwidth
        ))
        expected_allgather_per_layer = math.ceil(1e9 * (
            fixed_seconds
            + (tp - 1) * dispatch_local_chunk / bandwidth
        ))
        expected_reduce_scatter_per_layer = math.ceil(1e9 * (
            fixed_seconds
            + (tp - 1) / tp * hidden_payload / bandwidth
        ))
        self.assertEqual(
            rows["tp4"].tp_allreduce_ns,
            layers * expected_allreduce_per_layer,
        )
        self.assertEqual(
            rows["tp4"].ep_allgather_ns,
            layers * expected_allgather_per_layer,
        )
        self.assertEqual(
            rows["tp4"].ep_reduce_scatter_ns,
            layers * expected_reduce_scatter_per_layer,
        )
        self.assertEqual(
            rows["tp4"].collective_ns,
            (
                rows["tp4"].tp_allreduce_ns
                + rows["tp4"].ep_allgather_ns
                + rows["tp4"].ep_reduce_scatter_ns
            ),
        )
        self.assertEqual(
            rows["tp4"].collective_bytes_per_rank,
            int(layers * (
                2 * (tp - 1) / tp * hidden_payload
                + (tp - 1) * dispatch_local_chunk
                + (tp - 1) / tp * hidden_payload
            )),
        )

    def test_collective_fixed_cost_is_per_logical_operation(self):
        shape = HBFModelBatchShape(
            total_tokens=17,
            prefill_q=(17,),
            prefill_hbf_k=(125_000,),
            prefill_lpddr_k=(0,),
        )
        base = self.models["tp4"].batch_latency(shape)
        slower_hardware = dataclasses.replace(
            self.hardware,
            intra_fabric_fixed_latency_us=(
                self.hardware.intra_fabric_fixed_latency_us + 1.0
            ),
        )
        slower_model = build_full_model_hbf_latency(
            repo_root=REPO_ROOT,
            hardware=slower_hardware,
            layout=self.layouts["tp4"],
        )
        slower = slower_model.batch_latency(shape)
        self.assertEqual(
            slower.collective_ns - base.collective_ns,
            48 * 3 * 1_000,
        )
        self.assertEqual(
            self.models["tp4"].metadata()[
                "collective_fixed_latency_semantics"],
            "end_to_end_fixed_per_logical_collective",
        )

    def test_slow_lpddr_is_visible_in_full_model_latency(self):
        shape = HBFModelBatchShape(
            total_tokens=32,
            prefill_q=(32,),
            prefill_hbf_k=(125_000,),
            prefill_lpddr_k=(0,),
        )
        normal = self.models["tp4"].batch_latency(shape)
        fast_hardware = dataclasses.replace(
            self.hardware,
            lpddr_bandwidth_gbps_per_card=3_350.0,
        )
        fast_model = build_full_model_hbf_latency(
            repo_root=REPO_ROOT,
            hardware=fast_hardware,
            layout=self.layouts["tp4"],
        )
        fast = fast_model.batch_latency(shape)
        self.assertGreater(normal.total_ns, fast.total_ns)
        self.assertGreater(normal.lpddr_roof_ns_sum, fast.lpddr_roof_ns_sum)

    def test_router_gemm_is_explicitly_charged(self):
        shape = HBFModelBatchShape(
            total_tokens=17,
            prefill_q=(17,),
            prefill_hbf_k=(125_000,),
            prefill_lpddr_k=(0,),
        )
        row = self.models["tp4"].batch_latency(shape)
        fit = self.models["tp4"].base_provider.calibration.fits["router"]
        flops = 2.0 * 17 * 2_048 * 128
        hbf_bytes = 2 * 2_048 * 128
        lpddr_bytes = 2 * (17 * 2_048 + 17 * 128)
        roof_seconds = max(
            flops / (self.hardware.npu_peak_tflops_per_card * 1e12),
            (
                self.hardware.hbf_read_latency_us * 1e-6
                + hbf_bytes
                / (
                    self.hardware.hbf_read_bandwidth_gbps_per_card
                    * 1e9
                )
            ),
            lpddr_bytes
            / (self.hardware.lpddr_bandwidth_gbps_per_card * 1e9),
        )
        expected_per_layer = max(1, math.ceil(1e9 * max(
            fit.launch_floor_seconds,
            roof_seconds * fit.eta("central"),
        )))
        self.assertEqual(row.router_ns, 48 * expected_per_layer)
        self.assertEqual(
            row.total_ns,
            (
                row.embedding_ns
                + row.dense_ns
                + row.attention_ns
                + row.router_ns
                + row.moe_ns
                + row.final_ns
                + row.collective_ns
            ),
        )
        attention = self.models["tp4"]._attention(shape)
        self.assertEqual(
            row.attention_compute_roof_ns,
            48 * attention.compute_roof_ns,
        )
        self.assertEqual(
            row.attention_hbf_roof_ns,
            48 * attention.hbf_roof_ns,
        )
        self.assertEqual(
            row.attention_lpddr_roof_ns,
            48 * attention.lpddr_roof_ns,
        )
        self.assertEqual(
            row.attention_dominant_roof,
            attention.dominant_roof,
        )

    def test_long_decode_responds_to_hbf_bandwidth(self):
        shape = HBFModelBatchShape(
            total_tokens=8,
            decode_hbf_k=(500_000,) * 8,
            decode_lpddr_k=(64,) * 8,
            lm_head_sequences=8,
        )
        normal = self.models["tp4"].batch_latency(shape)
        slow_hardware = dataclasses.replace(
            self.hardware,
            hbf_read_bandwidth_gbps_per_card=1_675.0,
        )
        slow_model = build_full_model_hbf_latency(
            repo_root=REPO_ROOT,
            hardware=slow_hardware,
            layout=self.layouts["tp4"],
        )
        slow = slow_model.batch_latency(shape)
        self.assertGreater(slow.attention_ns, normal.attention_ns)
        self.assertGreater(slow.hbf_roof_ns_sum, normal.hbf_roof_ns_sum)

    def test_hbf_read_latency_is_charged_and_configurable(self):
        shape = HBFModelBatchShape(
            total_tokens=1,
            decode_hbf_k=(1,),
            decode_lpddr_k=(1,),
        )
        normal = self.models["tp4"].batch_latency(shape)
        zero_latency_hardware = dataclasses.replace(
            self.hardware,
            hbf_read_latency_us=0.0,
        )
        zero_latency_model = build_full_model_hbf_latency(
            repo_root=REPO_ROOT,
            hardware=zero_latency_hardware,
            layout=self.layouts["tp4"],
        )
        zero_latency = zero_latency_model.batch_latency(shape)
        self.assertGreater(normal.total_ns, zero_latency.total_ns)

    def test_decode_envelope_charges_hbf_read_latency(self):
        shape = HBFModelBatchShape(
            total_tokens=8,
            decode_hbf_k=(125_000,) * 8,
            decode_lpddr_k=(0,) * 8,
            lm_head_sequences=8,
        )
        normal = self.models["tp4"].batch_latency(shape)
        zero_latency_hardware = dataclasses.replace(
            self.hardware,
            hbf_read_latency_us=0.0,
        )
        zero_latency_model = build_full_model_hbf_latency(
            repo_root=REPO_ROOT,
            hardware=zero_latency_hardware,
            layout=self.layouts["tp4"],
        )
        zero_latency = zero_latency_model.batch_latency(shape)
        self.assertGreater(
            normal.attention_ns,
            zero_latency.attention_ns,
        )

    def test_mixed_attention_pays_one_kernel_launch_floor(self):
        model = self.models["tp4"]
        mixed_shape = HBFModelBatchShape(
            total_tokens=2,
            prefill_q=(1,),
            prefill_hbf_k=(0,),
            prefill_lpddr_k=(0,),
            decode_hbf_k=(1,),
            decode_lpddr_k=(0,),
            lm_head_sequences=2,
        )
        prefill_shape = HBFModelBatchShape(
            total_tokens=1,
            prefill_q=(1,),
            prefill_hbf_k=(0,),
            prefill_lpddr_k=(0,),
        )
        decode_shape = HBFModelBatchShape(
            total_tokens=1,
            decode_hbf_k=(1,),
            decode_lpddr_k=(0,),
        )
        mixed = model._attention(mixed_shape)
        prefill = model._attention(prefill_shape)
        decode = model._attention(decode_shape)
        work = model._attention_work(
            q_lengths=mixed_shape.prefill_q,
            hbf_k_lengths=mixed_shape.prefill_hbf_k,
            lpddr_k_lengths=mixed_shape.prefill_lpddr_k,
            causal_prefill=True,
        )
        prefill_roof_seconds = max(
            (
                work[0]
                / (self.hardware.npu_peak_tflops_per_card * 1e12)
                * work[3]
            ),
            (
                (
                    self.hardware.hbf_read_latency_us * 1e-6
                    + work[1]
                    / (
                        self.hardware.hbf_read_bandwidth_gbps_per_card
                        * 1e9
                    )
                )
                if work[1] > 0 else 0.0
            ),
            work[2]
            / (self.hardware.lpddr_bandwidth_gbps_per_card * 1e9),
        )
        fit = model.base_provider.calibration.fits[
            "prefill_attention"]
        expected_ns = math.ceil(1e9 * (
            decode.latency_seconds
            + prefill_roof_seconds * fit.eta("central")
        ))
        self.assertEqual(mixed.latency_ns, expected_ns)
        self.assertLess(
            mixed.latency_ns,
            prefill.latency_ns + decode.latency_ns,
        )
        self.assertEqual(
            mixed.hbf_roof_ns,
            prefill.hbf_roof_ns + decode.hbf_roof_ns,
        )

    def test_mixed_attention_residual_dominated_matches_pure_sum(self):
        model = self.models["tp4"]
        mixed = model._attention(HBFModelBatchShape(
            total_tokens=17,
            prefill_q=(16,),
            prefill_hbf_k=(100_000,),
            prefill_lpddr_k=(0,),
            decode_hbf_k=(100_000,),
            decode_lpddr_k=(0,),
            lm_head_sequences=2,
        ))
        prefill = model._attention(HBFModelBatchShape(
            total_tokens=16,
            prefill_q=(16,),
            prefill_hbf_k=(100_000,),
            prefill_lpddr_k=(0,),
        ))
        decode = model._attention(HBFModelBatchShape(
            total_tokens=1,
            decode_hbf_k=(100_000,),
            decode_lpddr_k=(0,),
        ))
        self.assertLessEqual(
            abs(mixed.latency_ns - (
                prefill.latency_ns + decode.latency_ns
            )),
            1,
        )

    def test_tp8_gqa_replication_prevents_fake_attention_halving(self):
        shape = HBFModelBatchShape(
            total_tokens=1,
            decode_hbf_k=(750_000,),
            decode_lpddr_k=(1,),
        )
        tp4 = self.models["tp4"].batch_latency(shape)
        tp8 = self.models["tp8"].batch_latency(shape)
        ratio = tp8.attention_ns / tp4.attention_ns
        self.assertGreater(ratio, 0.75)
        self.assertLess(ratio, 1.25)

    def test_capacity_charges_weights_and_tp8_kv_replication(self):
        weights = 60_000_000_000
        per_replica = {
            key: model.logical_hbf_capacity_bytes_per_replica(
                model_weight_bytes=weights)
            for key, model in self.models.items()
        }
        server = {
            key: model.logical_hbf_capacity_bytes_server(
                model_weight_bytes=weights)
            for key, model in self.models.items()
        }
        self.assertLess(per_replica["dp8"], per_replica["tp4"])
        self.assertGreater(server["tp4"], server["dp8"])
        self.assertGreater(server["dp8"], server["tp8"])

    def test_latency_is_deterministic(self):
        shape = HBFModelBatchShape(
            total_tokens=17,
            prefill_q=(16,),
            prefill_hbf_k=(100_000,),
            prefill_lpddr_k=(10,),
            decode_hbf_k=(80_000,),
            decode_lpddr_k=(20,),
            lm_head_sequences=2,
        )
        first = self.models["tp4"].batch_latency(shape)
        second = self.models["tp4"].batch_latency(shape)
        self.assertEqual(first, second)

    def test_execution_plan_is_ordered_and_immutable(self):
        shape = HBFModelBatchShape(
            total_tokens=17,
            prefill_q=(16,),
            prefill_hbf_k=(100_000,),
            prefill_lpddr_k=(10,),
            decode_hbf_k=(80_000,),
            decode_lpddr_k=(20,),
            lm_head_sequences=2,
        )
        plan = self.models["tp4"].batch_execution_plan(shape)
        self.assertIs(
            plan,
            self.models["tp4"].batch_execution_plan(shape),
        )
        self.assertIsInstance(plan.operations, tuple)
        self.assertEqual(
            [operation.name for operation in plan.operations[:16]],
            [
                "embedding",
                "layer_0.input_layernorm",
                "layer_0.qkv_proj",
                "layer_0.q_norm",
                "layer_0.k_norm",
                "layer_0.rotary_emb",
                "layer_0.attention",
                "layer_0.o_proj",
                "layer_0.tp_allreduce",
                "layer_0.post_attention_layernorm",
                "layer_0.router",
                "layer_0.ep_allgather",
                "layer_0.expert_up",
                "layer_0.expert_activation",
                "layer_0.expert_down",
                "layer_0.ep_reduce_scatter",
            ],
        )
        self.assertEqual(
            [operation.name for operation in plan.operations[-3:]],
            ["final_layernorm", "lm_head", "sampler"],
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            plan.layout = "changed"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            plan.operations[0].latency_ns = 0

    def test_execution_plan_aggregates_to_latency_for_every_layout(self):
        shapes = {
            "prefill": HBFModelBatchShape(
                total_tokens=16,
                prefill_q=(16,),
                prefill_hbf_k=(100_000,),
                prefill_lpddr_k=(10,),
            ),
            "decode": HBFModelBatchShape(
                total_tokens=2,
                decode_hbf_k=(80_000, 90_000),
                decode_lpddr_k=(20, 30),
                lm_head_sequences=2,
            ),
            "mixed": HBFModelBatchShape(
                total_tokens=17,
                prefill_q=(16,),
                prefill_hbf_k=(100_000,),
                prefill_lpddr_k=(10,),
                decode_hbf_k=(80_000,),
                decode_lpddr_k=(20,),
                lm_head_sequences=2,
            ),
        }
        latency_fields = {
            "embedding": "embedding_ns",
            "dense": "dense_ns",
            "attention": "attention_ns",
            "router": "router_ns",
            "moe": "moe_ns",
            "final": "final_ns",
        }
        collective_fields = {
            "tp_allreduce": "tp_allreduce_ns",
            "ep_allgather": "ep_allgather_ns",
            "ep_reduce_scatter": "ep_reduce_scatter_ns",
        }
        for key, model in self.models.items():
            for shape_name, shape in shapes.items():
                with self.subTest(layout=key, shape=shape_name):
                    self._assert_plan_aggregates_to_latency(
                        model,
                        shape,
                        expect_collectives=(key != "dp8"),
                        latency_fields=latency_fields,
                        collective_fields=collective_fields,
                    )

    def _assert_plan_aggregates_to_latency(
            self, model, shape, *, expect_collectives,
            latency_fields, collective_fields):
        plan = model.batch_execution_plan(shape)
        row = model.batch_latency(shape)
        kernels = plan.kernel_operations
        collectives = plan.collective_operations
        self.assertEqual(
            len(kernels),
            628 if model.layout.is_context_striped else 580,
        )
        self.assertEqual(
            len(collectives),
            (
                240
                if model.layout.is_context_striped
                else (144 if expect_collectives else 0)
            ),
        )
        self.assertTrue(all(
            isinstance(operation, HBFKernelExecutionOp)
            for operation in kernels
        ))
        self.assertTrue(all(
            isinstance(operation, HBFCollectiveExecutionOp)
            for operation in collectives
        ))
        self.assertEqual(
            plan.total_ns,
            sum(
                operation.latency_ns
                for operation in kernels
            )
            + sum(
                operation.latency_ns
                for operation in collectives
            ),
        )
        self.assertEqual(plan.total_ns, row.total_ns)
        for category, field in latency_fields.items():
            categories = (
                {"attention", "attention_merge"}
                if category == "attention"
                else {category}
            )
            self.assertEqual(
                sum(
                    operation.latency_ns
                    for operation in kernels
                    if operation.category in categories
                ),
                getattr(row, field),
            )
        for category, field in collective_fields.items():
            self.assertEqual(
                sum(
                    operation.latency_ns
                    for operation in collectives
                    if operation.category == category
                ),
                getattr(row, field),
            )
        self.assertEqual(
            sum(
                operation.latency_ns
                for operation in collectives
            ),
            row.collective_ns,
        )
        self.assertEqual(
            math.ceil(sum(
                operation.hbf_read_bytes_per_rank
                for operation in kernels
            )),
            row.hbf_read_bytes_per_rank,
        )
        self.assertEqual(
            math.ceil(sum(
                operation.lpddr_bytes_per_rank
                for operation in kernels
            )),
            row.lpddr_bytes_per_rank,
        )
        self.assertEqual(
            sum(
                operation.transferred_bytes_per_rank
                for operation in collectives
            ),
            row.collective_bytes_per_rank,
        )
        self.assertEqual(
            sum(
                operation.compute_roof_ns
                for operation in kernels
            ),
            row.compute_roof_ns_sum,
        )
        self.assertEqual(
            sum(
                operation.hbf_roof_ns
                for operation in kernels
            ),
            row.hbf_roof_ns_sum,
        )
        self.assertEqual(
            sum(
                operation.lpddr_roof_ns
                for operation in kernels
            ),
            row.lpddr_roof_ns_sum,
        )

    def test_execution_plan_rooflines_are_not_separate_operations(self):
        shape = HBFModelBatchShape(
            total_tokens=1,
            decode_hbf_k=(125_000,),
            decode_lpddr_k=(500,),
        )
        plan = self.models["tp4"].batch_execution_plan(shape)
        self.assertEqual(
            {operation.kind for operation in plan.operations},
            {"kernel", "collective"},
        )
        self.assertTrue(all(
            operation.latency_semantics
            == "calibrated_max_roofline_inclusive"
            for operation in plan.kernel_operations
        ))
        self.assertEqual(
            plan.total_ns,
            self.models["tp4"].batch_latency(shape).total_ns,
        )

    def test_tp8_execution_plan_labels_replicated_kv_semantics(self):
        shape = HBFModelBatchShape(
            total_tokens=1,
            decode_hbf_k=(750_000,),
            decode_lpddr_k=(1,),
        )
        tp4_plan = self.models["tp4"].batch_execution_plan(shape)
        tp8_plan = self.models["tp8"].batch_execution_plan(shape)
        self.assertEqual(
            tp4_plan.kv_layout_semantics,
            "one_unique_kv_head_per_tp_rank",
        )
        self.assertEqual(tp4_plan.physical_kv_replication_factor, 1)
        self.assertEqual(
            tp8_plan.kv_layout_semantics,
            "one_kv_head_replicated_across_each_two_q_ranks",
        )
        self.assertEqual(tp8_plan.physical_kv_replication_factor, 2)
        attention_operations = tuple(
            operation
            for operation in tp8_plan.kernel_operations
            if operation.category == "attention"
        )
        self.assertEqual(len(attention_operations), 48)
        self.assertEqual(
            {
                operation.kv_layout_semantics
                for operation in attention_operations
            },
            {"one_kv_head_replicated_across_each_two_q_ranks"},
        )
        self.assertEqual(
            {
                operation.kv_layout_semantics
                for operation in tp8_plan.kernel_operations
                if operation.category != "attention"
            },
            {"not_applicable"},
        )


if __name__ == "__main__":
    unittest.main()

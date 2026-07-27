import dataclasses
import math

from serving.core.h100_kernel_calibrated_prompt import (
    LAUNCH_FLOOR_CAP_SECONDS,
    MEDIA_STREAMING_EFFICIENCY,
)
from pathlib import Path
import unittest

from serving.core.gpu_pd_latency import (
    P4D4LatencyModel,
    load_p4d4_gpu_config,
)
from serving.core.hbf_full_model_latency import (
    HBFModelBatchShape,
    HBFParallelLayout,
    build_full_model_hbf_latency,
    load_hbf_server_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_CONFIG = (
    REPO_ROOT / "configs" / "wakekv_hbf"
    / "p4d4_gpu_server.json"
)
HBF_CONFIG = (
    REPO_ROOT / "configs" / "wakekv_hbf"
    / "full_model_8card_server.json"
)


class P4D4LatencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = load_p4d4_gpu_config(GPU_CONFIG)
        cls.model = P4D4LatencyModel(
            repo_root=REPO_ROOT,
            hardware=cls.hardware,
        )

    def test_capacity_charges_exact_weights_and_workspace(self):
        self.assertEqual(
            self.hardware.model_weight_bytes_per_rank,
            15_285_252_096,
        )
        expected = (
            80_000_000_000
            - 15_285_252_096
            - 19_893_012_480
        )
        self.assertEqual(
            self.hardware.usable_hbm_bytes_per_rank, expected)
        self.assertEqual(
            self.hardware.kv_bytes_per_token_per_rank, 24_576)
        self.assertGreater(
            self.hardware.usable_hbm_bytes_per_rank,
            1_000_000
            * self.hardware.kv_bytes_per_token_per_rank,
        )
        self.assertLess(
            self.hardware.usable_hbm_bytes_per_rank,
            2_000_000
            * self.hardware.kv_bytes_per_token_per_rank,
        )
        self.assertEqual(
            self.hardware.kv_capacity_bytes_per_rank(1),
            16 * 24_576,
        )
        self.assertEqual(
            self.hardware.kv_capacity_bytes_per_rank(16),
            self.hardware.kv_capacity_bytes_per_rank(1),
        )
        self.assertEqual(
            self.hardware.ssd_capacity_bytes,
            30_720_000_000_000,
        )

    def test_handoff_is_pairwise_and_linear(self):
        zero = self.model.handoff_latency(0)
        short = self.model.handoff_latency(1_000)
        long = self.model.handoff_latency(2_000)
        self.assertEqual(zero.latency_ns, 0)
        self.assertEqual(short.bytes_per_rank, 24_576_000)
        self.assertEqual(
            short.aggregate_bytes, 4 * short.bytes_per_rank)
        self.assertGreater(long.latency_ns, short.latency_ns)
        self.assertEqual(
            long.latency_ns - short.latency_ns,
            int(round(
                short.bytes_per_rank
                / (450.0 * 1e9)
                * 1e9
            )),
        )

    def test_gpu_uses_same_compute_anchor_but_faster_memory_path(self):
        shape = HBFModelBatchShape(
            total_tokens=16,
            prefill_q=(16,),
            prefill_hbf_k=(125_000,),
            prefill_lpddr_k=(0,),
        )
        gpu = self.model.batch_latency(shape)
        hbf_hardware, layouts = load_hbf_server_config(HBF_CONFIG)
        hbf = build_full_model_hbf_latency(
            repo_root=REPO_ROOT,
            hardware=hbf_hardware,
            layout=layouts["tp4"],
        ).batch_latency(shape)
        self.assertEqual(
            self.model.provider.metadata()["hardware"], "H100")
        self.assertEqual(
            self.hardware.gpu_peak_tflops_per_gpu, 989.5)
        self.assertEqual(
            self.hardware.hbm_bandwidth_gbps_per_gpu, 3_350.0)
        self.assertLess(gpu.total_ns, hbf.total_ns)

    def test_batch_phases_preserve_exact_aggregate_latency(self):
        shapes = (
            HBFModelBatchShape(
                total_tokens=16,
                prefill_q=(16,),
                prefill_hbf_k=(125_000,),
                prefill_lpddr_k=(0,),
                lm_head_sequences=1,
            ),
            HBFModelBatchShape(
                total_tokens=4,
                decode_hbf_k=(1_000, 2_000, 3_000, 4_000),
                decode_lpddr_k=(0, 0, 0, 0),
                lm_head_sequences=4,
            ),
        )
        for shape in shapes:
            with self.subTest(shape=shape):
                latency = self.model.batch_latency(shape)
                phases = self.model.batch_phase_latency(shape)

                self.assertEqual(phases.layer_count, 48)
                self.assertEqual(
                    phases.total_ns, latency.total_ns)
                self.assertEqual(
                    phases.prologue_ns
                    + phases.layer_count * phases.layer_ns
                    + phases.epilogue_ns,
                    latency.total_ns,
                )
                self.assertEqual(
                    phases.layer_start_offset_ns(0),
                    phases.prologue_ns,
                )
                self.assertEqual(
                    phases.layer_start_offset_ns(47),
                    phases.prologue_ns + 47 * phases.layer_ns,
                )

    def test_gpu_has_one_hbm_path_not_independent_media_roofs(self):
        hbf_prefix = HBFModelBatchShape(
            total_tokens=16,
            prefill_q=(16,),
            prefill_hbf_k=(125_000,),
            prefill_lpddr_k=(0,),
        )
        alternate_label = HBFModelBatchShape(
            total_tokens=16,
            prefill_q=(16,),
            prefill_hbf_k=(0,),
            prefill_lpddr_k=(125_000,),
        )
        self.assertEqual(
            self.model.batch_latency(hbf_prefix),
            self.model.batch_latency(alternate_label),
        )

    def test_gpu_collectives_match_tp4_ag_rs_contract(self):
        shape = HBFModelBatchShape(
            total_tokens=17,
            prefill_q=(16,),
            prefill_hbf_k=(125_000,),
            prefill_lpddr_k=(0,),
            decode_hbf_k=(125_016,),
            decode_lpddr_k=(0,),
            lm_head_sequences=2,
        )
        row = self.model.batch_latency(shape)
        tp = 4
        layers = 48
        bandwidth = (
            self.hardware.nvlink_bandwidth_gbps_per_gpu * 1e9)
        fixed_seconds = (
            self.hardware.collective_fixed_latency_us * 1e-6)
        hidden_payload = 17 * 2_048 * 2
        dispatch_local_chunk = (
            max(1, 17 // tp) * (2_048 + 128) * 2)

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
            row.tp_allreduce_ns,
            layers * expected_allreduce_per_layer,
        )
        self.assertEqual(
            row.ep_allgather_ns,
            layers * expected_allgather_per_layer,
        )
        self.assertEqual(
            row.ep_reduce_scatter_ns,
            layers * expected_reduce_scatter_per_layer,
        )
        self.assertEqual(
            row.collective_ns,
            (
                row.tp_allreduce_ns
                + row.ep_allgather_ns
                + row.ep_reduce_scatter_ns
            ),
        )

    def test_gpu_collective_fixed_cost_is_per_logical_operation(self):
        shape = HBFModelBatchShape(
            total_tokens=17,
            prefill_q=(17,),
            prefill_hbf_k=(125_000,),
            prefill_lpddr_k=(0,),
        )
        base = self.model.batch_latency(shape)
        slower = P4D4LatencyModel(
            repo_root=REPO_ROOT,
            hardware=dataclasses.replace(
                self.hardware,
                collective_fixed_latency_us=(
                    self.hardware.collective_fixed_latency_us + 1.0
                ),
            ),
        ).batch_latency(shape)
        self.assertEqual(
            slower.collective_ns - base.collective_ns,
            48 * 3 * 1_000,
        )
        self.assertEqual(
            self.model.metadata()[
                "collective_fixed_latency_semantics"],
            "end_to_end_fixed_per_logical_collective",
        )

    def test_gpu_router_gemm_is_added_to_provider_comp(self):
        shape = HBFModelBatchShape(
            total_tokens=17,
            prefill_q=(17,),
            prefill_hbf_k=(125_000,),
            prefill_lpddr_k=(0,),
        )
        row = self.model.batch_latency(shape)
        fit = self.model.provider.calibration.fits["router"]
        flops = 2.0 * 17 * 2_048 * 128
        hbm_bytes = 2 * (
            17 * 2_048
            + 2_048 * 128
            + 17 * 128
        )
        # v2 semantics: eta on compute only, media streaming efficiency on
        # the HBM roofline, capped launch floor.
        expected_per_layer = math.ceil(1e9 * max(
            min(fit.launch_floor_seconds, LAUNCH_FLOOR_CAP_SECONDS),
            flops / (989.5 * 1e12) * fit.eta("central"),
            hbm_bytes / (
                3_350.0 * 1e9 * MEDIA_STREAMING_EFFICIENCY),
        ))
        self.assertEqual(row.router_ns, 48 * expected_per_layer)
        self.assertEqual(
            row.comp_ns,
            row.provider_comp_ns + row.router_ns,
        )
        self.assertEqual(
            row.total_ns,
            row.comp_ns + row.collective_ns,
        )

    def test_invalid_partition_and_roots_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "P4D4"):
            dataclasses.replace(
                self.hardware, prefill_gpu_count=3,
            ).validate()
        with self.assertRaisesRegex(ValueError, "PCIe"):
            dataclasses.replace(
                self.hardware, pcie_root_count=1,
            ).validate()
        with self.assertRaisesRegex(ValueError, "H100 calibration"):
            dataclasses.replace(
                self.hardware, gpu_peak_tflops_per_gpu=500.0,
            ).validate()


if __name__ == "__main__":
    unittest.main()

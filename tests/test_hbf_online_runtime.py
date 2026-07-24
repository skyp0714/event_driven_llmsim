from pathlib import Path
import unittest

from serving.core.hbf_full_model_latency import (
    qwen_logical_kv_bytes_per_token,
)
from serving.core.hbf_online_runtime import (
    FULL_MODEL_HBF_RUNTIME_SCHEMA,
    FullModelHBFRuntimeOptions,
    build_full_model_hbf_online_runtime,
    load_full_model_hbf_hardware,
    validate_full_model_hbf_gpu_cluster,
)
from serving.core.memory_model import MemoryModel
from serving.core.request import Request
from serving.core.scheduler import Scheduler


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "configs/wakekv_hbf/full_model_8card_server.json"
)
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LATENCY_MODEL = "h100-qwen3-tp4-kernel-calibrated"


class NullLogger:
    def info(self, *args, **kwargs):
        pass


def instance(instance_id, pd_type):
    return {
        "instance_id": instance_id,
        "node_id": 0,
        "pd_type": pd_type,
        "model_name": MODEL,
        "hardware": "H100",
        "num_npus": 4,
        "tp_size": 4,
        "pp_size": 1,
    }


def runtime_config():
    return {
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "block_size": 16,
        "enable_prefix_caching": False,
        "max_model_len": 1_010_000,
        "latency_model": LATENCY_MODEL,
    }


def finite_scheduler(instance_id, pd_type):
    per_rank_token_bytes = qwen_logical_kv_bytes_per_token() // 4
    memory = MemoryModel.__new__(MemoryModel)
    memory.instance_id = instance_id
    memory.node_id = 0
    memory.block_size = 16
    memory.kv_heads_per_tp_rank = 1
    memory.head_dim = per_rank_token_bytes // 2
    memory.layers_per_pp_rank = 1
    memory.kv_fp = 1
    memory.weight = 1
    memory.npu_used = 1
    memory.npu_peak_used = 1
    memory.npu_allocatable_mem = 10 ** 15
    memory.npu_mem = 10 ** 15
    memory.logger = NullLogger()
    memory.enable_prefix_caching = False

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.instance_id = instance_id
    scheduler.node_id = 0
    scheduler.pd_type = pd_type
    scheduler.model = MODEL
    scheduler.tp_size = 4
    scheduler.block_size = 16
    scheduler.enable_prefix_caching = False
    scheduler.memory = memory
    scheduler.request = []
    scheduler.agentic_kv_manager = None
    scheduler.pd_prefill_reclaimability_generation = 0
    scheduler.max_model_len = 1_010_000
    return scheduler


class FullModelHBFOnlineRuntimeTests(unittest.TestCase):
    def test_config_exposes_canonical_context_striping(self):
        hardware, layout = load_full_model_hbf_hardware(
            CONFIG, "tp8_context")
        self.assertEqual(hardware.card_count, 8)
        self.assertEqual(layout.key, "tp8_context")
        self.assertEqual(layout.tp_size, 8)
        self.assertEqual(layout.replicas, 1)

    def test_gpu_cluster_validation_is_fail_closed(self):
        instances = [instance(0, "prefill"), instance(1, "decode")]
        configs = [runtime_config(), runtime_config()]
        self.assertEqual(
            validate_full_model_hbf_gpu_cluster(
                instances,
                configs,
                {0: 0, 1: 0},
                network_backend="analytical-congestion-aware",
            ),
            (0, 1),
        )
        with self.assertRaisesRegex(
                ValueError, "analytical-congestion-aware"):
            validate_full_model_hbf_gpu_cluster(
                instances,
                configs,
                {0: 0, 1: 0},
                network_backend="analytical",
            )
        invalid = [dict(configs[0]), dict(configs[1])]
        invalid[1]["enable_prefix_caching"] = True
        with self.assertRaisesRegex(ValueError, "prefix caching"):
            validate_full_model_hbf_gpu_cluster(
                instances,
                invalid,
                {0: 0, 1: 0},
                network_backend="analytical-congestion-aware",
            )

    def test_builder_shares_exact_ledger_and_finite_gpu_bridge(self):
        instances = [instance(0, "prefill"), instance(1, "decode")]
        schedulers = [
            finite_scheduler(0, "prefill"),
            finite_scheduler(1, "decode"),
        ]
        options = FullModelHBFRuntimeOptions(
            layout_key="tp8_context",
            max_num_batched_tokens=16,
            max_num_seqs=4,
            max_prefill_chunk_tokens=16,
            prefill_drain_tail_tokens=7,
            prefill_drain_min_tokens=11,
            astra_chunk_bytes=1024 * 1024,
            server_id=3,
        )
        runtime = build_full_model_hbf_online_runtime(
            repo_root=REPO_ROOT,
            config_path=CONFIG,
            options=options,
            instances=instances,
            runtime_configs=[runtime_config(), runtime_config()],
            inst2node_mapping={0: 0, 1: 0},
            schedulers=schedulers,
            network_backend="analytical-congestion-aware",
        )
        self.assertIs(
            runtime.lifecycle.lpddr_ledger,
            runtime.pool.lpddr_ledger,
        )
        self.assertEqual(runtime.adapter.gpu_resume_mode, "recompute")
        self.assertEqual(
            runtime.gpu_hbm_bridge.pd_pairs, ((0, 1),))
        self.assertEqual(
            runtime.gpu_hbm_bridge.report()["adapter_contract"][
                "gpu_resume_mode"],
            "recompute",
        )
        self.assertFalse(runtime.pool.retain_detailed_history)
        self.assertTrue(
            runtime.pool.retain_token_completion_history)
        self.assertEqual(runtime.pool.prefill_drain_tail_tokens, 7)
        self.assertEqual(runtime.pool.prefill_drain_min_tokens, 11)
        report = runtime.report()
        self.assertEqual(report["schema"], FULL_MODEL_HBF_RUNTIME_SCHEMA)
        self.assertEqual(report["layout"]["key"], "tp8_context")
        self.assertEqual(report["gpu_pd_pair"], [0, 1])

        request = Request(9, MODEL, 17, 19, 0, 0)
        request.session_id = "session-reservation"
        self.assertTrue(runtime.gpu_hbm_bridge.try_reserve_pd_decode(
            request,
            prefill_instance_id=0,
            decode_instance_id=1,
        ))
        with self.assertRaisesRegex(
                RuntimeError, "pending_pd_decode_reservations"):
            runtime.assert_quiescent()
        runtime.gpu_hbm_bridge.cancel_pd_decode_reservation(request)
        runtime.assert_quiescent()

    def test_prefill_drain_options_are_non_negative(self):
        for field in (
            "prefill_drain_tail_tokens",
            "prefill_drain_min_tokens",
        ):
            values = {field: -1}
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    FullModelHBFRuntimeOptions(**values).validate()
        options = FullModelHBFRuntimeOptions(
            prefill_drain_tail_tokens=0,
            prefill_drain_min_tokens=0,
        )
        options.validate()


if __name__ == "__main__":
    unittest.main()

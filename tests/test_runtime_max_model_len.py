import json
from pathlib import Path
import unittest

from serving.__main__ import (
    _agentic_pd_layout,
    _resolve_instance_max_model_len,
)


class RuntimeMaxModelLenTest(unittest.TestCase):
    def test_model_config_is_default(self):
        instance = {
            "instance_id": 0,
            "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        }

        self.assertEqual(
            _resolve_instance_max_model_len(instance, None),
            262144,
        )

    def test_cli_fallback_and_instance_override(self):
        instance = {
            "instance_id": 0,
            "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        }
        self.assertEqual(
            _resolve_instance_max_model_len(instance, 524288),
            524288,
        )

        instance["max_model_len"] = 1010000
        self.assertEqual(
            _resolve_instance_max_model_len(instance, 524288),
            1010000,
        )

    def test_non_positive_or_non_integer_limit_is_rejected(self):
        base = {
            "instance_id": 0,
            "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        }
        for invalid in (0, -1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _resolve_instance_max_model_len(
                    {**base, "max_model_len": invalid}, None)
        for invalid in (True, 1010000.0, "1010000"):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                _resolve_instance_max_model_len(
                    {**base, "max_model_len": invalid}, None)

    def test_agentic_pd_layout_includes_context_limit(self):
        instance = {
            "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "tp_size": 4,
            "pp_size": 1,
        }
        runtime = {
            "block_size": 16,
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "max_model_len": 1010000,
            "latency_model": "h100-qwen3-tp4-kernel-calibrated",
            "latency_model_band": "central",
        }

        peer_runtime = {**runtime, "max_model_len": 262144}

        self.assertNotEqual(
            _agentic_pd_layout(instance, runtime),
            _agentic_pd_layout(instance, peer_runtime),
        )

    def test_qwen3_1m_p4d4_cluster_contract(self):
        repo_root = Path(__file__).resolve().parents[1]
        path = (
            repo_root
            / "configs/cluster/single_node_qwen3_1m_pd_p4d4_h100.json"
        )
        cluster = json.loads(path.read_text(encoding="utf-8"))
        node = cluster["nodes"][0]

        self.assertEqual(
            node["cpu_mem"]["mem_size"] * (1 << 30),
            512_000_000_000,
        )
        self.assertEqual(node["cpu_mem"]["mem_bw"], 200)
        self.assertEqual(
            {instance["pd_type"] for instance in node["instances"]},
            {"prefill", "decode"},
        )
        expected_max_num_seqs = {"prefill": 32, "decode": 128}
        for instance in node["instances"]:
            self.assertEqual(instance["num_npus"], 4)
            self.assertEqual(instance["tp_size"], 4)
            self.assertEqual(instance["ep_size"], 4)
            self.assertEqual(instance["max_model_len"], 1010000)
            self.assertEqual(instance["max_num_batched_tokens"], 131072)
            self.assertEqual(
                instance["max_num_seqs"],
                expected_max_num_seqs[instance["pd_type"]],
            )
            self.assertTrue(instance["enable_chunked_prefill"])
            self.assertFalse(instance["enable_prefix_caching"])
            self.assertEqual(instance["dtype"], "bfloat16")
            self.assertEqual(
                instance["npu_mem"]["mem_size"] * (1 << 30),
                80e9,
            )
            self.assertEqual(instance["npu_mem"]["mem_bw"], 3350)
            self.assertEqual(
                instance["npu_mem"]["runtime_reserve_bytes"],
                19_893_012_480,
            )
            self.assertEqual(
                instance["latency_model"],
                "h100-qwen3-tp4-kernel-calibrated",
            )


if __name__ == "__main__":
    unittest.main()

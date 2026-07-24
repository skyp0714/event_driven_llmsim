import json
import unittest
from pathlib import Path
from unittest.mock import patch

from serving.core.memory_model import Device, MemoryModel


class HBMRuntimeReserveTest(unittest.TestCase):
    _MODEL_CONFIG = {
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 16,
    }

    def memory(
            self, reserve=0, *, weight=256, prefix_caching=False,
            npu_mem=1, cpu_mem=1):
        with patch(
                "serving.core.memory_model.get_config",
                return_value=self._MODEL_CONFIG), patch.object(
                    MemoryModel, "get_weight", return_value=weight):
            return MemoryModel(
                model="test/model",
                instance_id=0,
                node_id=0,
                num_npus=1,
                tp_size=1,
                npu_mem=npu_mem,
                cpu_mem=cpu_mem,
                block_size=16,
                fp=16,
                enable_prefix_caching=prefix_caching,
                enable_prefix_sharing=False,
                prefix_pool=None,
                prefix_storage=None,
                npu_runtime_reserve_bytes=reserve,
            )

    def test_reserve_reduces_allocatable_but_preserves_physical_total(self):
        memory = self.memory(1024, prefix_caching=True)

        self.assertEqual(memory.npu_physical_mem, 1 << 30)
        self.assertEqual(memory.npu_runtime_reserve_bytes, 1024)
        self.assertEqual(memory.npu_allocatable_mem, (1 << 30) - 1024)
        self.assertEqual(memory.npu_mem, memory.npu_allocatable_mem)
        self.assertEqual(
            memory.mem_for_kv,
            memory.npu_allocatable_mem - memory.weight,
        )
        self.assertEqual(memory.npu_used, memory.weight)

    def test_allocate_availability_need_and_free_use_allocatable_capacity(self):
        memory = self.memory(4096)
        kv_capacity = memory.npu_allocatable_mem - memory.weight

        self.assertTrue(memory.is_avail(kv_capacity, Device.NPU))
        self.assertEqual(memory.need_size(kv_capacity, Device.NPU), 0)
        memory.allocate(kv_capacity, Device.NPU)

        self.assertFalse(memory.is_avail(1, Device.NPU))
        self.assertEqual(memory.need_size(1, Device.NPU), 1)
        with self.assertRaisesRegex(RuntimeError, "only 0.00MB is available"):
            memory.allocate(1, Device.NPU)

        memory.free(kv_capacity, Device.NPU)
        self.assertEqual(memory.npu_used, memory.weight)
        self.assertEqual(
            memory.npu_physical_mem - memory.npu_used,
            kv_capacity + memory.npu_runtime_reserve_bytes,
        )

    def test_invalid_reserve_is_rejected(self):
        for invalid in (True, 1.0, "1"):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                self.memory(invalid)
        for invalid in (-1, 1 << 30, (1 << 30) + 1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.memory(invalid)

    def test_weights_must_fit_after_reserve(self):
        reserve = 4096
        allocatable = (1 << 30) - reserve
        with self.assertRaisesRegex(
                RuntimeError, "exceeds allocatable NPU memory"):
            self.memory(reserve, weight=allocatable + 1)

    def test_zero_reserve_preserves_legacy_capacity(self):
        memory = self.memory()

        self.assertEqual(memory.npu_physical_mem, 1 << 30)
        self.assertEqual(memory.npu_allocatable_mem, 1 << 30)
        self.assertEqual(memory.npu_mem, 1 << 30)

    def test_qwen_hardware_decimal_capacities_are_exact_integer_bytes(self):
        memory = self.memory(
            19_893_012_480,
            npu_mem=74.50580596923828,
            cpu_mem=476.837158203125,
        )

        self.assertEqual(memory.npu_physical_mem, 80_000_000_000)
        self.assertEqual(memory.npu_allocatable_mem, 60_106_987_520)
        self.assertEqual(memory.cpu_mem, 512_000_000_000)

    def test_qwen_cluster_leaves_expected_per_rank_kv_capacity(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "configs/cluster/single_node_qwen3_1m_pd_p4d4_h100.json"
        )
        cluster = json.loads(path.read_text(encoding="utf-8"))
        node = cluster["nodes"][0]
        instance = node["instances"][0]

        memory = MemoryModel(
            model=instance["model_name"],
            instance_id=0,
            node_id=0,
            num_npus=instance["num_npus"],
            tp_size=instance["tp_size"],
            npu_mem=instance["npu_mem"]["mem_size"],
            cpu_mem=node["cpu_mem"]["mem_size"],
            block_size=instance["block_size"],
            fp=16,
            enable_prefix_caching=False,
            enable_prefix_sharing=False,
            prefix_pool=None,
            prefix_storage=None,
            ep_size=instance["ep_size"],
            pp_size=instance["pp_size"],
            kv_cache_dtype=instance["kv_cache_dtype"],
            npu_runtime_reserve_bytes=(
                instance["npu_mem"]["runtime_reserve_bytes"]),
        )

        self.assertEqual(memory.weight, 15_285_227_520)
        self.assertEqual(memory.npu_allocatable_mem, 60_106_987_520)
        self.assertEqual(
            memory.npu_allocatable_mem - memory.npu_used,
            44_821_760_000,
        )
        self.assertEqual(memory.get_kv(1_010_000), 24_821_760_000)
        self.assertGreater(
            memory.npu_allocatable_mem - memory.npu_used,
            memory.get_kv(1_010_000),
        )


if __name__ == "__main__":
    unittest.main()

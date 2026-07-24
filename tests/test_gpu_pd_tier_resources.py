import math
from pathlib import Path
import unittest

from serving.core.gpu_pd_latency import load_p4d4_gpu_config
from serving.core.gpu_pd_tier_resources import TierNodeResources
from serving.core.hbf_full_model_lifecycle import ResourceCalendar


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_CONFIG = (
    REPO_ROOT / "configs" / "wakekv_hbf"
    / "p4d4_gpu_server.json"
)


class TierNodeResourcesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = load_p4d4_gpu_config(GPU_CONFIG)
        cls.node = TierNodeResources(
            hardware=cls.hardware,
            node_id=0,
        )

    def test_exact_hbm_capacity_and_block_limits(self):
        self.assertEqual(
            self.node.hbm_kv_capacity_bytes_per_rank,
            44_821_735_424,
        )
        self.assertEqual(
            self.node.max_hbm_kv_blocks_per_rank,
            113_987,
        )
        self.assertEqual(
            self.node.max_hbm_kv_tokens_per_rank,
            1_823_792,
        )
        self.assertEqual(
            self.hardware.kv_capacity_bytes_per_rank(1),
            393_216,
        )
        self.assertEqual(
            self.hardware.kv_capacity_bytes_per_rank(16),
            393_216,
        )
        self.assertEqual(
            self.hardware.kv_capacity_bytes_per_rank(17),
            786_432,
        )

    def test_exact_sixteen_token_restore_stages(self):
        ssd = self.node.ssd_stage(16, direction="ssd_to_cpu")
        h2d = self.node.gpu_cpu_stage(
            16,
            gpu_role="p",
            direction="cpu_to_gpu",
        )
        self.assertEqual(ssd.bytes_per_rank, 393_216)
        self.assertEqual(ssd.aggregate_bytes, 1_572_864)
        self.assertEqual(ssd.latency_ns, 48_494)
        self.assertEqual(h2d.latency_ns, 12_865)
        self.assertEqual(
            ssd.latency_ns + h2d.latency_ns,
            61_359,
        )

    def test_one_million_token_restore_uses_configured_bandwidth(self):
        ssd = self.node.ssd_stage(
            1_000_000,
            direction="ssd_to_cpu",
        )
        h2d = self.node.gpu_cpu_stage(
            1_000_000,
            gpu_role="p",
            direction="cpu_to_gpu",
        )
        self.assertEqual(ssd.bytes_per_rank, 24_576_000_000)
        self.assertEqual(ssd.aggregate_bytes, 98_304_000_000)
        self.assertEqual(ssd.latency_ns, 1_780_889_566)
        self.assertEqual(h2d.latency_ns, 491_525_000)
        self.assertEqual(
            ssd.latency_ns + h2d.latency_ns,
            2_272_414_566,
        )

    def test_ssd_write_uses_slower_write_bandwidth(self):
        read = self.node.ssd_stage(16, direction="ssd_to_cpu")
        write = self.node.ssd_stage(16, direction="cpu_to_ssd")
        self.assertEqual(write.latency_ns, 66_812)
        self.assertGreater(write.latency_ns, read.latency_ns)

    def test_peer_copy_defaults_to_exact_wire_tokens(self):
        exact = self.node.peer_stage(
            17,
            direction="d_to_p",
        )
        blocked = self.node.peer_stage(
            17,
            direction="d_to_p",
            block_rounded=True,
        )
        self.assertEqual(exact.bytes_per_rank, 17 * 24_576)
        self.assertEqual(exact.latency_ns, 1_929)
        self.assertEqual(blocked.bytes_per_rank, 786_432)
        self.assertGreater(blocked.latency_ns, exact.latency_ns)

    def test_p_and_d_use_distinct_pcie_roots(self):
        p = self.node.gpu_cpu_stage(
            16, gpu_role="p", direction="cpu_to_gpu")
        d = self.node.gpu_cpu_stage(
            16, gpu_role="d", direction="cpu_to_gpu")
        self.assertIn("gpu-node-0-pcie-root-0", p.resources)
        self.assertNotIn("gpu-node-0-pcie-root-1", p.resources)
        self.assertIn("gpu-node-0-pcie-root-1", d.resources)
        self.assertNotIn("gpu-node-0-pcie-root-0", d.resources)

    def test_ssd_restore_is_two_serial_reservations(self):
        calendar = ResourceCalendar()
        ssd = self.node.ssd_stage(16, direction="ssd_to_cpu")
        h2d = self.node.gpu_cpu_stage(
            16, gpu_role="p", direction="cpu_to_gpu")
        ssd_start, ssd_end = ssd.reserve(
            calendar,
            ready_ns=0,
            job_id=1,
            namespace="restore",
        )
        h2d_start, h2d_end = h2d.reserve(
            calendar,
            ready_ns=ssd_end,
            job_id=2,
            namespace="restore",
        )
        self.assertEqual(ssd_start, 0)
        self.assertEqual(ssd_end, 48_494)
        self.assertEqual(h2d_start, ssd_end)
        self.assertEqual(h2d_end, 61_359)

    def test_two_nodes_do_not_share_resource_queues(self):
        node1 = TierNodeResources(
            hardware=self.hardware,
            node_id=1,
        )
        calendar = ResourceCalendar()
        stage0 = self.node.ssd_stage(16, direction="ssd_to_cpu")
        stage1 = node1.ssd_stage(16, direction="ssd_to_cpu")
        first = stage0.reserve(
            calendar, ready_ns=0, job_id=1, namespace="n0")
        other_node = stage1.reserve(
            calendar, ready_ns=0, job_id=1, namespace="n1")
        second_same_node = stage0.reserve(
            calendar, ready_ns=0, job_id=2, namespace="n0")
        self.assertEqual(first, other_node)
        self.assertEqual(second_same_node[0], first[1])
        self.assertTrue(
            set(stage0.resources).isdisjoint(stage1.resources))

    def test_latency_equations_include_one_fixed_cost(self):
        token_count = 16
        rank_bytes = 393_216
        aggregate = 4 * rank_bytes
        expected = math.ceil(1e9 * (
            5e-6 + max(
                rank_bytes / 50e9,
                aggregate / 200e9,
            )
        ))
        stage = self.node.gpu_cpu_stage(
            token_count,
            gpu_role="p",
            direction="gpu_to_cpu",
        )
        self.assertEqual(stage.latency_ns, expected)

    def test_invalid_role_direction_and_tokens_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "gpu_role"):
            self.node.gpu_cpu_stage(
                1, gpu_role="x", direction="cpu_to_gpu")
        with self.assertRaisesRegex(ValueError, "SSD direction"):
            self.node.ssd_stage(1, direction="invalid")
        with self.assertRaisesRegex(ValueError, "token_count"):
            self.node.peer_stage(-1, direction="p_to_d")


if __name__ == "__main__":
    unittest.main()

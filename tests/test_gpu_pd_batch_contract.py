from pathlib import Path
import unittest

from serving.core.gpu_pd_dual_oracle import (
    DualStrictInfiniteHBMOracle,
)
from serving.core.gpu_pd_dual_tiered import (
    DualFiniteHBMTieredBaseline,
)
from serving.core.gpu_pd_latency import load_p4d4_gpu_config
from serving.core.gpu_pd_oracle_node import StrictInfiniteHBMNode
from serving.core.gpu_pd_tiered_node import (
    FiniteHBMTieredP4D4Node,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_CONFIG = (
    REPO_ROOT / "configs" / "wakekv_hbf"
    / "p4d4_gpu_server.json"
)


class P4D4BatchContractPropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = load_p4d4_gpu_config(GPU_CONFIG)

    def assert_documented_limits(self, pool):
        self.assertEqual(pool.max_num_batched_tokens, 131_072)
        self.assertEqual(pool.max_prefill_chunk_tokens, 131_072)
        self.assertEqual(pool.max_num_seqs, 128)
        self.assertEqual(pool.p_max_num_seqs, 32)
        self.assertEqual(pool.d_max_num_seqs, 128)

    def assert_documented_report(self, report):
        self.assertEqual(report["max_num_batched_tokens"], 131_072)
        self.assertEqual(report["max_prefill_chunk_tokens"], 131_072)
        self.assertEqual(report["max_num_seqs"], 128)
        self.assertEqual(report["p_max_num_seqs"], 32)
        self.assertEqual(report["d_max_num_seqs"], 128)

    def test_single_oracle_propagates_stage_limits(self):
        node = StrictInfiniteHBMNode(
            repo_root=REPO_ROOT,
            hardware=self.hardware,
            node_id=0,
            max_num_batched_tokens=131_072,
            max_num_seqs=128,
            p_max_num_seqs=32,
            d_max_num_seqs=128,
            max_prefill_chunk_tokens=131_072,
        )
        self.assert_documented_limits(node.pool)
        self.assert_documented_report(node.report()["pool"])

    def test_single_tiered_propagates_stage_limits(self):
        node = FiniteHBMTieredP4D4Node(
            repo_root=REPO_ROOT,
            hardware=self.hardware,
            node_id=0,
            policy="hbm_lru_recompute",
            max_num_batched_tokens=131_072,
            max_num_seqs=128,
            p_max_num_seqs=32,
            d_max_num_seqs=128,
            max_prefill_chunk_tokens=131_072,
        )
        self.assert_documented_limits(node.pool)
        self.assert_documented_report(node.report()["pool"])

    def test_dual_oracle_propagates_stage_limits_to_both_nodes(self):
        runner = DualStrictInfiniteHBMOracle(
            repo_root=REPO_ROOT,
            hardware=self.hardware,
            max_num_batched_tokens=131_072,
            max_num_seqs=128,
            p_max_num_seqs=32,
            d_max_num_seqs=128,
            max_prefill_chunk_tokens=131_072,
        )
        self.assertEqual(len(runner.nodes), 2)
        for node in runner.nodes:
            self.assert_documented_limits(node.pool)
        for node_report in runner.report()["nodes"]:
            self.assert_documented_report(node_report["pool"])

    def test_dual_tiered_propagates_stage_limits_to_both_nodes(self):
        runner = DualFiniteHBMTieredBaseline(
            repo_root=REPO_ROOT,
            hardware=self.hardware,
            policy="hbm_lru_recompute",
            max_num_batched_tokens=131_072,
            max_num_seqs=128,
            p_max_num_seqs=32,
            d_max_num_seqs=128,
            max_prefill_chunk_tokens=131_072,
        )
        self.assertEqual(len(runner.nodes), 2)
        for node in runner.nodes:
            self.assert_documented_limits(node.pool)
        for node_report in runner.report()["nodes"]:
            self.assert_documented_report(node_report["pool"])


if __name__ == "__main__":
    unittest.main()

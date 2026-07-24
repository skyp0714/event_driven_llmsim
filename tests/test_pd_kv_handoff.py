import unittest
from types import SimpleNamespace
from unittest.mock import patch

from serving.core.request import Batch, Request
from serving.core.trace_generator import (
    _emit_pp_pd_power,
    _make_sub_batch,
    _pd_kv_handoff_bytes,
)


class PDKVHandoffTest(unittest.TestCase):
    def test_gqa_handoff_contains_only_k_and_v(self):
        ctx = SimpleNamespace(
            kv_head=8, tp_size=8, head_dim=128, kv_fp=2)
        batch = SimpleNamespace(total_len=100)

        per_rank = _pd_kv_handoff_bytes(ctx, batch)

        self.assertEqual(per_rank, 2 * 1 * 128 * 100 * 2)
        self.assertEqual(per_rank * ctx.tp_size, 2 * 8 * 128 * 100 * 2)

    def test_tp_larger_than_kv_heads_accounts_for_replication(self):
        ctx = SimpleNamespace(
            kv_head=4, tp_size=8, head_dim=128, kv_fp=1)
        batch = SimpleNamespace(total_len=16)

        per_rank = _pd_kv_handoff_bytes(ctx, batch)

        self.assertEqual(per_rank, 2 * 1 * 128 * 16)
        self.assertEqual(per_rank * ctx.tp_size, 2 * 8 * 128 * 16)

    def test_lower_tier_prefix_augments_only_the_staged_graph(self):
        ctx = SimpleNamespace(
            kv_head=8, tp_size=8, head_dim=128, kv_fp=2)
        first = SimpleNamespace(
            total_len=1, pd_restored_prefix_handoff_tokens=15)
        later = SimpleNamespace(
            total_len=1, pd_restored_prefix_handoff_tokens=0)

        self.assertEqual(
            _pd_kv_handoff_bytes(ctx, first),
            2 * 1 * 128 * 16 * 2,
        )
        self.assertEqual(
            _pd_kv_handoff_bytes(ctx, later),
            2 * 1 * 128 * 1 * 2,
        )

    def test_hbm_retained_prefix_remains_suffix_only(self):
        ctx = SimpleNamespace(
            kv_head=8, tp_size=8, head_dim=128, kv_fp=2)
        batch = SimpleNamespace(
            total_len=1, pd_restored_prefix_handoff_tokens=0)

        self.assertEqual(
            _pd_kv_handoff_bytes(ctx, batch),
            2 * 1 * 128 * 1 * 2,
        )

    def test_pd_power_uses_exact_graph_bytes_across_layers_and_ranks(self):
        class FakePower:
            def __init__(self):
                self.link_bytes = []

            def add_link_energy_consumption(self, node_id, num_bytes):
                self.link_bytes.append((node_id, num_bytes))

        power = FakePower()
        ctx = SimpleNamespace(
            power_model=power,
            pp_size=1,
            pd_type="prefill",
            config={"hidden_size": 4096, "num_hidden_layers": 48},
            kv_head=8,
            tp_size=8,
            head_dim=128,
            kv_fp=2,
            fp=2,
            node_id=3,
            perf_db={},
            model="model",
        )
        bctx = SimpleNamespace(
            total_len=1,
            lm_head_len=2,
            batch=SimpleNamespace(
                pd_restored_prefix_handoff_tokens=15),
        )
        per_rank_kv = 2 * 1 * 128 * 16 * 2
        sampler_output_per_rank = 2 * 4
        with patch(
                "serving.core.trace_generator._sequence",
                return_value=["sampler"]), patch(
                "serving.core.trace_generator.calculate_sizes",
                return_value=(0, 0, sampler_output_per_rank)):
            _emit_pp_pd_power(ctx, bctx)

        self.assertEqual(power.link_bytes, [(
            3,
            per_rank_kv * 8 * 48 + sampler_output_per_rank * 8,
        )])

    def test_sub_batch_split_preserves_restored_prefix_ownership(self):
        req_a = Request(0, "model", 16, 18, 0, 0)
        req_b = Request(1, "model", 16, 18, 0, 0)
        batch = Batch(
            0, "model", 2, 0,
            [1, 1], [], 2, 0, [1, 1], [15, 15], [],
            0, 0,
        )
        batch.requests.extend([req_a, req_b])
        batch.scheduled_tokens = {0: 1, 1: 1}
        batch.pd_restored_prefix_handoff_by_request = {0: 15}
        batch.pd_restored_prefix_handoff_tokens = 15
        batch.pd_new_kv_handoff_by_request = {0: 1, 1: 1}
        batch.pd_new_kv_handoff_tokens = 2

        sub_batches = _make_sub_batch(batch)
        by_request = {
            sub.requests[0].id: sub for sub in sub_batches
        }

        self.assertEqual(
            by_request[0].pd_restored_prefix_handoff_tokens, 15)
        self.assertEqual(
            by_request[0].pd_restored_prefix_handoff_by_request, {0: 15})
        self.assertEqual(
            by_request[1].pd_restored_prefix_handoff_tokens, 0)
        self.assertEqual(
            sum(sub.pd_new_kv_handoff_tokens for sub in sub_batches), 2)


if __name__ == "__main__":
    unittest.main()

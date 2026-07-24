import dataclasses
import math
from pathlib import Path
import unittest

from serving.core.h100_kernel_calibrated_prompt import (
    BF16_BYTES,
    QWEN_HEAD_DIM,
    QWEN_LAYERS,
)
from serving.core.hbf_full_model_latency import (
    HBFModelBatchShape,
    HBFParallelLayout,
    HBFServerHardware,
    ONLINE_SOFTMAX_ACCUMULATOR_BYTES,
    ONLINE_SOFTMAX_STATE_ELEMENTS,
    build_full_model_hbf_latency,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TP8ContextStripedLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HBFServerHardware()
        cls.tp8_layout = HBFParallelLayout.for_key("tp8")
        cls.context_layout = HBFParallelLayout.for_key("tp8_context")
        cls.tp8 = build_full_model_hbf_latency(
            repo_root=REPO_ROOT,
            hardware=cls.hardware,
            layout=cls.tp8_layout,
        )
        cls.context = build_full_model_hbf_latency(
            repo_root=REPO_ROOT,
            hardware=cls.hardware,
            layout=cls.context_layout,
        )

    def test_layout_is_additive_and_keeps_dense_tp8_sharding(self):
        layout = self.context_layout
        self.assertEqual((layout.tp_size, layout.replicas), (8, 1))
        self.assertTrue(layout.is_context_striped)
        self.assertEqual(layout.physical_kv_replication_factor, 1)
        self.assertEqual(layout.q_heads_per_rank, 4)
        self.assertEqual(layout.attention_q_heads_per_rank, 8)
        self.assertEqual(layout.kv_heads_per_rank, 1)
        self.assertEqual(layout.context_partition_factor, 2)
        self.assertEqual(
            layout.kv_layout_semantics,
            (
                "four_kv_heads_mapped_to_card_pairs_with_unique_"
                "even_odd_token_zigzag_context_q8_attention"
            ),
        )
        self.assertEqual(
            (self.tp8_layout.tp_size, self.tp8_layout.replicas),
            (8, 1),
        )
        self.assertEqual(self.tp8_layout.physical_kv_replication_factor, 2)
        self.assertEqual(self.tp8_layout.attention_q_heads_per_rank, 4)
        with self.assertRaisesRegex(ValueError, "one TP8 replica"):
            HBFParallelLayout(
                key="tp8_context", tp_size=4, replicas=2).validate(8)

    def test_attention_uses_q8_over_exact_token_granular_halves(self):
        kwargs = {
            "q_lengths": (16,),
            "hbf_k_lengths": (100_000,),
            "lpddr_k_lengths": (2_000,),
            "causal_prefill": True,
        }
        conventional = self.tp8._attention_work(**kwargs)
        striped = tuple(
            self.context._attention_work(**kwargs, context_rank=rank)
            for rank in (0, 1)
        )

        full_visible_pairs = 16 * 102_000 + 16 * 17 / 2
        odd_visible_prefixes = 8
        expected_visible_pairs = (
            (full_visible_pairs + odd_visible_prefixes) / 2,
            (full_visible_pairs - odd_visible_prefixes) / 2,
        )
        expected_hbf_bytes = (
            BF16_BYTES * 2 * (100_000 / 2) * QWEN_HEAD_DIM)
        expected_lpddr_bytes = BF16_BYTES * (
            2 * 16 * 8 * QWEN_HEAD_DIM
            + 2 * ((2_000 + 16) / 2) * QWEN_HEAD_DIM
        )

        for rank in (0, 1):
            self.assertEqual(
                striped[rank][0],
                expected_visible_pairs[rank]
                * 8
                * (4.0 * QWEN_HEAD_DIM + 5.0),
            )
            self.assertEqual(striped[rank][1], expected_hbf_bytes)
            self.assertEqual(striped[rank][2], expected_lpddr_bytes)
            self.assertEqual(
                striped[rank][3],
                math.ceil(8 / 132) * 132 / 8,
            )
        # Both physical ranks together preserve all causal work.  They do
        # not each round an odd visible-pair total up.
        self.assertEqual(
            sum(work[0] for work in striped),
            2 * conventional[0],
        )
        self.assertEqual(
            sum(work[1] for work in striped),
            conventional[1],
        )

    def test_unique_kv_bytes_and_capacity_are_not_rounded_into_replication(self):
        shape = HBFModelBatchShape(
            total_tokens=1,
            decode_hbf_k=(1,),
            decode_lpddr_k=(0,),
        )
        plan = self.context.batch_execution_plan(shape)
        logical_one_token_kv_bytes = (
            BF16_BYTES * 2 * 4 * QWEN_HEAD_DIM)
        self.assertEqual(
            plan.context_attention_physical_hbf_read_bytes_per_layer,
            logical_one_token_kv_bytes,
        )
        self.assertEqual(
            tuple(
                rank.hbf_read_bytes
                for rank in plan.context_attention_rank_executions
            ),
            (BF16_BYTES * 2 * QWEN_HEAD_DIM, 0),
        )
        self.assertEqual(plan.physical_kv_replication_factor, 1)

        weights = 60_000_000_000
        self.assertEqual(
            self.context.logical_hbf_capacity_bytes_per_replica(
                model_weight_bytes=weights),
            2 * self.tp8.logical_hbf_capacity_bytes_per_replica(
                model_weight_bytes=weights),
        )

    def test_odd_causal_context_preserves_exact_rank_work_and_byte_ledger(self):
        shape = HBFModelBatchShape(
            total_tokens=3,
            prefill_q=(3,),
            prefill_hbf_k=(1,),
            prefill_lpddr_k=(1,),
        )
        plan = self.context.batch_execution_plan(shape)
        ranks = plan.context_attention_rank_executions
        self.assertEqual(tuple(rank.pair_rank for rank in ranks), (0, 1))
        self.assertEqual(
            {rank.token_mapping for rank in ranks},
            {"sequence_local_even_odd_token_zigzag"},
        )

        flop_scale = 8 * (4.0 * QWEN_HEAD_DIM + 5.0)
        self.assertEqual(
            tuple(rank.flops for rank in ranks),
            (7 * flop_scale, 5 * flop_scale),
        )
        self.assertEqual(
            tuple(rank.visible_pairs for rank in ranks), (7, 5))
        self.assertEqual(
            tuple(rank.hbf_kv_tokens for rank in ranks), (1, 0))
        self.assertEqual(
            tuple(rank.lpddr_kv_tokens for rank in ranks), (2, 2))
        one_kv_token_bytes = BF16_BYTES * 2 * QWEN_HEAD_DIM
        self.assertEqual(
            tuple(rank.hbf_read_bytes for rank in ranks),
            (one_kv_token_bytes, 0),
        )

        query_io_per_rank = (
            BF16_BYTES * 2 * 3 * 8 * QWEN_HEAD_DIM)
        # Active LPDDR token position 1 belongs to rank 1.  Current causal
        # positions 2,3,4 split 2/1, so each rank owns two LPDDR KV tokens.
        lpddr_per_rank = (
            query_io_per_rank + 2 * one_kv_token_bytes)
        self.assertEqual(
            tuple(rank.lpddr_bytes for rank in ranks),
            (lpddr_per_rank, lpddr_per_rank),
        )
        self.assertEqual(
            plan.context_attention_hbf_read_bytes_per_layer,
            one_kv_token_bytes,
        )
        self.assertEqual(
            plan.context_attention_lpddr_bytes_per_layer,
            2 * lpddr_per_rank,
        )

        # Four independent KV-head pairs cover all 32 Q heads.  HBF KV is
        # physically unique; Q/partial IO is intentionally duplicated across
        # the two context ranks and remains visible in the LPDDR ledger.
        self.assertEqual(
            plan.context_attention_physical_hbf_read_bytes_per_layer,
            BF16_BYTES * 2 * 1 * 4 * QWEN_HEAD_DIM,
        )
        self.assertEqual(
            plan.context_attention_physical_lpddr_bytes_per_layer,
            4 * (
                2 * query_io_per_rank
                + BF16_BYTES * 2 * (1 + 3) * QWEN_HEAD_DIM
            ),
        )

        attention = next(
            operation
            for operation in plan.kernel_operations
            if operation.name == "layer_0.attention"
        )
        self.assertEqual(
            attention.latency_ns,
            max(rank.latency_ns for rank in ranks),
        )

    def test_odd_decode_context_is_exact_for_both_tiers_and_visible_pairs(self):
        shape = HBFModelBatchShape(
            total_tokens=1,
            decode_hbf_k=(3,),
            decode_lpddr_k=(2,),
        )
        plan = self.context.batch_execution_plan(shape)
        ranks = plan.context_attention_rank_executions
        flop_scale = 8 * (4.0 * QWEN_HEAD_DIM + 5.0)
        self.assertEqual(
            tuple(rank.flops for rank in ranks),
            (3 * flop_scale, 2 * flop_scale),
        )
        self.assertEqual(
            tuple(rank.visible_pairs for rank in ranks), (3, 2))
        self.assertEqual(
            tuple(rank.hbf_kv_tokens for rank in ranks), (2, 1))
        self.assertEqual(
            tuple(rank.lpddr_kv_tokens for rank in ranks), (1, 1))
        one_kv_token_bytes = BF16_BYTES * 2 * QWEN_HEAD_DIM
        self.assertEqual(
            tuple(rank.hbf_read_bytes for rank in ranks),
            (2 * one_kv_token_bytes, one_kv_token_bytes),
        )
        query_io_per_rank = (
            BF16_BYTES * 2 * 1 * 8 * QWEN_HEAD_DIM)
        self.assertEqual(
            tuple(rank.lpddr_bytes for rank in ranks),
            (
                query_io_per_rank + one_kv_token_bytes,
                query_io_per_rank + one_kv_token_bytes,
            ),
        )
        self.assertEqual(
            sum(rank.hbf_read_bytes for rank in ranks),
            3 * one_kv_token_bytes,
        )
        self.assertEqual(
            sum(
                rank.lpddr_bytes - query_io_per_rank
                for rank in ranks
            ),
            2 * one_kv_token_bytes,
        )

    def test_small_causal_zigzag_matches_exhaustive_token_mapping(self):
        token_bytes = BF16_BYTES * 2 * QWEN_HEAD_DIM
        flop_scale = 8 * (4.0 * QWEN_HEAD_DIM + 5.0)
        for hbf_tokens in range(5):
            for lpddr_tokens in range(5):
                for query_tokens in range(1, 6):
                    prior = hbf_tokens + lpddr_tokens
                    query_io = (
                        BF16_BYTES
                        * 2
                        * query_tokens
                        * 8
                        * QWEN_HEAD_DIM
                    )
                    for rank in (0, 1):
                        with self.subTest(
                                hbf=hbf_tokens,
                                lpddr=lpddr_tokens,
                                q=query_tokens,
                                rank=rank):
                            work = self.context._attention_work(
                                q_lengths=(query_tokens,),
                                hbf_k_lengths=(hbf_tokens,),
                                lpddr_k_lengths=(lpddr_tokens,),
                                causal_prefill=True,
                                context_rank=rank,
                            )
                            visible = sum(
                                sum(
                                    position % 2 == rank
                                    for position in range(
                                        prior + query_index + 1)
                                )
                                for query_index in range(query_tokens)
                            )
                            expected_hbf = sum(
                                position % 2 == rank
                                for position in range(hbf_tokens)
                            )
                            expected_lpddr = sum(
                                position % 2 == rank
                                for position in range(
                                    hbf_tokens,
                                    prior + query_tokens,
                                )
                            )
                            self.assertEqual(
                                work[0], visible * flop_scale)
                            self.assertEqual(
                                work[1], expected_hbf * token_bytes)
                            self.assertEqual(
                                work[2] - query_io,
                                expected_lpddr * token_bytes,
                            )

    def test_mixed_attention_uses_pair_rank_barrier_on_exact_combined_work(self):
        shape = HBFModelBatchShape(
            total_tokens=4,
            prefill_q=(3,),
            prefill_hbf_k=(1,),
            prefill_lpddr_k=(1,),
            decode_hbf_k=(3,),
            decode_lpddr_k=(2,),
            lm_head_sequences=2,
        )
        plan = self.context.batch_execution_plan(shape)
        ranks = plan.context_attention_rank_executions
        self.assertEqual(
            tuple(rank.visible_pairs for rank in ranks), (10, 7))
        self.assertEqual(
            tuple(rank.hbf_kv_tokens for rank in ranks), (3, 1))
        self.assertEqual(
            tuple(rank.lpddr_kv_tokens for rank in ranks), (3, 3))

        attention_operations = tuple(
            operation
            for operation in plan.kernel_operations
            if operation.category == "attention"
        )
        self.assertEqual(len(attention_operations), QWEN_LAYERS)
        critical_rank_ns = max(rank.latency_ns for rank in ranks)
        self.assertEqual(
            {operation.latency_ns for operation in attention_operations},
            {critical_rank_ns},
        )
        non_attention_total = sum(
            operation.latency_ns
            for operation in plan.operations
            if not (
                getattr(operation, "kind", None) == "kernel"
                and operation.category == "attention"
            )
        )
        # Every transformer layer has the same pair barrier; replacing the
        # layer-0 critical attention by either individual rank would be
        # incorrect when odd contexts make their work asymmetric.
        self.assertEqual(
            plan.total_ns,
            non_attention_total + QWEN_LAYERS * critical_rank_ns,
        )
        self.assertEqual(
            plan.total_ns,
            self.context.batch_latency(shape).total_ns,
        )

    def test_conventional_plan_has_no_context_rank_metadata(self):
        shape = HBFModelBatchShape(
            total_tokens=1,
            decode_hbf_k=(3,),
            decode_lpddr_k=(2,),
        )
        plan = self.tp8.batch_execution_plan(shape)
        self.assertEqual(plan.context_attention_rank_executions, ())
        self.assertEqual(
            plan.context_attention_hbf_read_bytes_per_layer, 0)
        self.assertEqual(
            plan.context_attention_lpddr_bytes_per_layer, 0)
        self.assertEqual(
            plan.context_attention_physical_hbf_read_bytes_per_layer, 0)
        self.assertEqual(
            plan.context_attention_physical_lpddr_bytes_per_layer, 0)

    def test_dense_model_and_standard_tp_collectives_remain_normal_tp8(self):
        shape = HBFModelBatchShape(
            total_tokens=18,
            prefill_q=(16,),
            prefill_hbf_k=(100_000,),
            prefill_lpddr_k=(10,),
            decode_hbf_k=(80_000, 90_000),
            decode_lpddr_k=(20, 30),
            lm_head_sequences=3,
        )
        conventional = self.tp8.batch_latency(shape)
        striped = self.context.batch_latency(shape)
        for field in (
            "embedding_ns",
            "dense_ns",
            "router_ns",
            "moe_ns",
            "final_ns",
            "tp_allreduce_ns",
            "ep_allgather_ns",
            "ep_reduce_scatter_ns",
        ):
            with self.subTest(field=field):
                self.assertEqual(
                    getattr(striped, field),
                    getattr(conventional, field),
                )

    def test_pair_cross_terms_have_exact_per_layer_bytes_and_fabric_latency(self):
        shape = HBFModelBatchShape(
            total_tokens=18,
            prefill_q=(16,),
            prefill_hbf_k=(100_000,),
            prefill_lpddr_k=(10,),
            decode_hbf_k=(80_000, 90_000),
            decode_lpddr_k=(20, 30),
            lm_head_sequences=3,
        )
        queries = shape.real_query_tokens
        query_bytes = queries * 4 * QWEN_HEAD_DIM * BF16_BYTES
        partial_bytes = (
            queries
            * 4
            * ONLINE_SOFTMAX_STATE_ELEMENTS
            * ONLINE_SOFTMAX_ACCUMULATOR_BYTES
        )
        bandwidth = (
            self.hardware.intra_fabric_bandwidth_gbps_per_card * 1e9)
        fixed = self.hardware.intra_fabric_fixed_latency_us * 1e-6
        query_ns = math.ceil(1e9 * (fixed + query_bytes / bandwidth))
        partial_ns = math.ceil(
            1e9 * (fixed + partial_bytes / bandwidth))

        row = self.context.batch_latency(shape)
        conventional = self.tp8.batch_latency(shape)
        self.assertEqual(
            row.pair_query_exchange_bytes_per_rank,
            QWEN_LAYERS * query_bytes,
        )
        self.assertEqual(
            row.pair_softmax_partial_exchange_bytes_per_rank,
            QWEN_LAYERS * partial_bytes,
        )
        self.assertEqual(
            row.pair_query_exchange_ns,
            QWEN_LAYERS * query_ns,
        )
        self.assertEqual(
            row.pair_softmax_partial_exchange_ns,
            QWEN_LAYERS * partial_ns,
        )
        self.assertEqual(
            row.collective_bytes_per_rank
            - conventional.collective_bytes_per_rank,
            QWEN_LAYERS * (query_bytes + partial_bytes),
        )
        self.assertEqual(
            row.collective_ns - conventional.collective_ns,
            row.pair_query_exchange_ns
            + row.pair_softmax_partial_exchange_ns,
        )

    def test_partial_merge_reads_both_fp32_states_and_writes_owned_q4(self):
        shape = HBFModelBatchShape(
            total_tokens=32,
            prefill_q=(32,),
            prefill_hbf_k=(100_000,),
            prefill_lpddr_k=(0,),
        )
        queries = shape.real_query_tokens
        state_bytes = (
            queries
            * 4
            * ONLINE_SOFTMAX_STATE_ELEMENTS
            * ONLINE_SOFTMAX_ACCUMULATOR_BYTES
        )
        output_bytes = (
            queries * 4 * QWEN_HEAD_DIM * BF16_BYTES)
        expected_per_layer = 2 * state_bytes + output_bytes
        row = self.context.batch_latency(shape)
        self.assertEqual(
            row.pair_attention_merge_lpddr_bytes_per_rank,
            QWEN_LAYERS * expected_per_layer,
        )
        self.assertGreater(row.pair_attention_merge_ns, 0)

        slower = build_full_model_hbf_latency(
            repo_root=REPO_ROOT,
            hardware=dataclasses.replace(
                self.hardware,
                lpddr_bandwidth_gbps_per_card=10.0,
            ),
            layout=self.context_layout,
        ).batch_latency(shape)
        self.assertGreater(
            slower.pair_attention_merge_ns,
            row.pair_attention_merge_ns,
        )

    def test_plan_orders_query_exchange_partial_exchange_and_merge(self):
        shape = HBFModelBatchShape(
            total_tokens=1,
            decode_hbf_k=(125_000,),
            decode_lpddr_k=(0,),
        )
        plan = self.context.batch_execution_plan(shape)
        names = [operation.name for operation in plan.operations]
        layer_zero = [
            name for name in names if name.startswith("layer_0.")]
        self.assertEqual(
            layer_zero[:11],
            [
                "layer_0.input_layernorm",
                "layer_0.qkv_proj",
                "layer_0.q_norm",
                "layer_0.k_norm",
                "layer_0.rotary_emb",
                "layer_0.pair_query_exchange",
                "layer_0.attention",
                "layer_0.pair_softmax_partial_exchange",
                "layer_0.pair_online_softmax_merge",
                "layer_0.o_proj",
                "layer_0.tp_allreduce",
            ],
        )
        self.assertEqual(len(plan.kernel_operations), 580 + QWEN_LAYERS)
        self.assertEqual(
            len(plan.collective_operations),
            144 + 2 * QWEN_LAYERS,
        )
        query_op = next(
            operation
            for operation in plan.collective_operations
            if operation.category == "pair_query_exchange"
        )
        partial_op = next(
            operation
            for operation in plan.collective_operations
            if operation.category == "pair_softmax_partial_exchange"
        )
        merge_op = next(
            operation
            for operation in plan.kernel_operations
            if operation.category == "attention_merge"
        )
        self.assertEqual(query_op.collective_type, "PAIR_QUERY_EXCHANGE")
        self.assertEqual(
            partial_op.collective_type,
            "PAIR_SOFTMAX_PARTIAL_EXCHANGE",
        )
        self.assertEqual(
            merge_op.family,
            "online_softmax_merge",
        )
        self.assertEqual(
            {
                operation.kv_layout_semantics
                for operation in (
                    next(
                        op for op in plan.kernel_operations
                        if op.name == "layer_0.attention"
                    ),
                    merge_op,
                )
            },
            {self.context_layout.kv_layout_semantics},
        )

    def test_conventional_tp8_has_no_new_pair_costs(self):
        shape = HBFModelBatchShape(
            total_tokens=1,
            decode_hbf_k=(125_000,),
            decode_lpddr_k=(0,),
        )
        row = self.tp8.batch_latency(shape)
        self.assertEqual(row.pair_query_exchange_ns, 0)
        self.assertEqual(row.pair_softmax_partial_exchange_ns, 0)
        self.assertEqual(row.pair_attention_merge_ns, 0)
        self.assertEqual(row.pair_query_exchange_bytes_per_rank, 0)
        self.assertEqual(
            row.pair_softmax_partial_exchange_bytes_per_rank, 0)
        self.assertEqual(row.pair_attention_merge_lpddr_bytes_per_rank, 0)
        self.assertFalse(any(
            "pair_" in operation.name
            for operation in self.tp8.batch_execution_plan(shape).operations
        ))

    def test_metadata_pins_context_partition_and_partial_format(self):
        metadata = self.context.metadata()
        self.assertEqual(
            metadata["kv_layout_semantics"],
            self.context_layout.kv_layout_semantics,
        )
        contract = metadata["context_parallel_attention"]
        self.assertTrue(contract["enabled"])
        self.assertEqual(contract["kv_head_pairs"], 4)
        self.assertEqual(contract["cards_per_kv_head"], 2)
        self.assertEqual(
            contract["card_pairs"],
            ((0, 1), (2, 3), (4, 5), (6, 7)),
        )
        self.assertEqual(contract["physical_kv_replication_factor"], 1)
        self.assertEqual(contract["attention_q_heads_per_rank"], 8)
        self.assertEqual(contract["dense_q_heads_per_rank"], 4)
        self.assertEqual(contract["softmax_partial"], "fp32_(O,max,sum)")
        self.assertIn(
            "odd-token rank asymmetry preserved",
            contract["partition"],
        )


if __name__ == "__main__":
    unittest.main()

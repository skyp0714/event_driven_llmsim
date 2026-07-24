import math
import unittest
from pathlib import Path

from serving.core.online_latency_model import (
    H100_QWEN3_TP4_KERNEL_CALIBRATED,
    OnlineBatchShape,
    OnlineLatencyModelError,
    resolve_online_latency_model,
    validate_online_latency_contract,
)
from serving.core.utils import get_config


class OnlineLatencyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        cls.config = get_config(cls.model_name)
        cls.provider = resolve_online_latency_model(
            name=H100_QWEN3_TP4_KERNEL_CALIBRATED,
            repo_root=cls.repo_root,
            hardware="H100",
            model=cls.model_name,
            config=cls.config,
            tp_size=4,
            pp_size=1,
            local_ep=4,
            ep_total=4,
            fp_bytes=2,
            dtype="bfloat16",
            kv_cache_dtype="auto",
            enable_attn_offloading=False,
        )

    def test_contract_is_exact_and_fails_loudly(self):
        for override, expected in (
                ({"hardware": "H200"}, "hardware"),
                ({"tp_size": 8}, "parallelism"),
                ({"kv_cache_dtype": "fp8"}, "kv_cache_dtype"),
                ({"enable_attn_offloading": True}, "offloading")):
            kwargs = {
                "name": H100_QWEN3_TP4_KERNEL_CALIBRATED,
                "hardware": "H100",
                "model": self.model_name,
                "config": self.config,
                "tp_size": 4,
                "pp_size": 1,
                "local_ep": 4,
                "ep_total": 4,
                "fp_bytes": 2,
                "dtype": "bfloat16",
                "kv_cache_dtype": "auto",
                "enable_attn_offloading": False,
            }
            kwargs.update(override)
            with self.assertRaisesRegex(
                    OnlineLatencyModelError, expected):
                validate_online_latency_contract(**kwargs)

    def test_decode_fit_has_contiguous_long_k_validation(self):
        fits = self.provider.decode_attention_fits
        self.assertEqual(fits[0].batch_min, 1)
        self.assertEqual(fits[-1].batch_max, 128)
        self.assertEqual(
            [fit.batch_min for fit in fits], list(range(1, 129))
        )
        for fit in fits:
            self.assertGreater(fit.training_rows, 0)
            self.assertGreater(fit.holdout_rows, 0)
            self.assertGreater(fit.source_k_max, 0)
            self.assertTrue(math.isfinite(fit.holdout_mape))
            self.assertTrue(math.isfinite(fit.holdout_p90_ape))
            self.assertLessEqual(fit.eta_fast, fit.eta_central)
            self.assertLessEqual(fit.eta_central, fit.eta_slow)

    def test_prefill_and_long_suffix_are_monotonic_and_finite(self):
        small = OnlineBatchShape(1_024, (1_024,), (0,), (), 1)
        medium = OnlineBatchShape(8_192, (8_192,), (0,), (), 1)
        long_chunk = OnlineBatchShape(
            131_072, (131_072,), (0,), (), 1
        )
        cached_short = OnlineBatchShape(4_096, (4_096,), (0,), (), 1)
        cached_long = OnlineBatchShape(
            4_096, (4_096,), (900_000,), (), 1
        )
        values = [
            self.provider.attention_latency_ns(shape)
            for shape in (small, medium, long_chunk)
        ]
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])
        self.assertGreater(
            self.provider.attention_latency_ns(cached_long),
            self.provider.attention_latency_ns(cached_short),
        )
        self.assertTrue(all(0 < value < 3_600e9 for value in values))

    def test_decode_and_mixed_batches_cover_one_million_tokens(self):
        decode_short = OnlineBatchShape(1, (), (), (4_096,), 1)
        decode_long = OnlineBatchShape(1, (), (), (1_000_000,), 1)
        decode_many = OnlineBatchShape(
            64, (), (), (1_000_000,) * 64, 64
        )
        mixed = OnlineBatchShape(
            4_160,
            (4_096,),
            (500_000,),
            (1_000_000,) * 64,
            65,
        )
        short_ns = self.provider.attention_latency_ns(decode_short)
        long_ns = self.provider.attention_latency_ns(decode_long)
        many_ns = self.provider.attention_latency_ns(decode_many)
        mixed_ns = self.provider.attention_latency_ns(mixed)
        self.assertLess(short_ns, long_ns)
        self.assertLess(long_ns, many_ns)
        self.assertGreater(
            mixed_ns,
            self.provider.attention_latency_ns(
                OnlineBatchShape(
                    4_096, (4_096,), (500_000,), (), 1
                )
            ),
        )
        self.assertGreater(mixed_ns, many_ns)
        for shape in (decode_long, decode_many, mixed):
            critical_ns = self.provider.batch_kernel_critical_path_ns(shape)
            self.assertGreater(critical_ns, 1_000)
            self.assertLess(critical_ns, 3_600 * 1_000_000_000)

    def test_decode_skew_is_not_free(self):
        uniform = OnlineBatchShape(
            8, (), (), (500_000,) * 8, 8
        )
        skewed = OnlineBatchShape(
            8, (), (), (1_000,) * 4 + (999_000,) * 4, 8
        )
        # Both batches have the same total K. The skew correction must make
        # the heterogeneous batch slower than the uniform-mean batch.
        self.assertGreater(
            self.provider.attention_latency_ns(skewed),
            self.provider.attention_latency_ns(uniform),
        )

    def test_decode_latency_is_monotonic_across_batch_boundaries_and_k(self):
        batches = (1, 8, 9, 32, 33, 128)
        contexts = (32, 128, 1_024, 8_192, 65_536, 1_000_000)
        by_context = {}
        for context in contexts:
            values = []
            for batch_size in batches:
                shape = OnlineBatchShape(
                    batch_size,
                    (),
                    (),
                    (context,) * batch_size,
                    batch_size,
                )
                values.append(
                    self.provider.attention_latency_ns(shape)
                )
            self.assertEqual(values, sorted(values), (context, values))
            by_context[context] = values
        for batch_index, batch_size in enumerate(batches):
            values = [
                by_context[context][batch_index]
                for context in contexts
            ]
            self.assertEqual(values, sorted(values), (batch_size, values))

    def test_long_prompt_is_chunked_and_has_broad_sanity_anchors(self):
        cold_chunk = OnlineBatchShape(
            131_072, (131_072,), (0,), (), 1
        )
        final_long_chunk = OnlineBatchShape(
            131_072, (131_072,), (878_928,), (), 1
        )
        decode_1m = OnlineBatchShape(1, (), (), (1_000_000,), 1)
        cold_seconds = (
            self.provider.batch_kernel_critical_path_ns(cold_chunk) / 1e9
        )
        final_seconds = (
            self.provider.batch_kernel_critical_path_ns(final_long_chunk)
            / 1e9
        )
        decode_seconds = (
            self.provider.batch_kernel_critical_path_ns(decode_1m) / 1e9
        )
        self.assertGreater(cold_seconds, 1.0)
        self.assertLess(cold_seconds, 60.0)
        self.assertGreater(final_seconds, cold_seconds)
        self.assertLess(final_seconds, 600.0)
        self.assertGreater(decode_seconds, 0.05)
        self.assertLess(decode_seconds, 5.0)
        with self.assertRaisesRegex(
                OnlineLatencyModelError, "scheduler-generated"):
            self.provider.attention_latency_ns(OnlineBatchShape(
                1_000_000, (1_000_000,), (0,), (), 1
            ))

    def test_recompute_attribution_uses_cache_hit_counterfactual(self):
        shape = OnlineBatchShape(
            8_192, (8_192,), (0,), (), 1, (6_144,)
        )
        attribution = self.provider.recompute_attribution(shape)
        counterfactual = shape.without_recompute_queries()
        self.assertEqual(counterfactual.prefill_q, (2_048,))
        self.assertEqual(counterfactual.prefill_k, (6_144,))
        self.assertEqual(attribution["recompute_query_tokens"], 6_144)
        self.assertGreater(attribution["recompute_marginal_comp_ns"], 0)
        self.assertEqual(
            attribution[
                "recompute_marginal_comp_critical_path_ns"],
            attribution["modeled_comp_critical_path_ns"]
            - attribution[
                "cache_hit_counterfactual_comp_critical_path_ns"
            ],
        )
        self.assertLessEqual(
            attribution["recompute_marginal_comp_ns"],
            attribution["modeled_comp_critical_path_ns"],
        )
        no_recompute = self.provider.recompute_attribution(
            OnlineBatchShape(8_192, (8_192,), (0,), (), 1, (0,))
        )
        self.assertEqual(no_recompute["recompute_marginal_comp_ns"], 0)

    def test_singleton_prefill_cost_uses_online_chunk_geometry(self):
        cold_ns = self.provider.singleton_prefill_comp_ns(
            input_tokens=300_000,
            hit_tokens=0,
            max_chunk_tokens=131_072,
        )
        cached_ns = self.provider.singleton_prefill_comp_ns(
            input_tokens=300_000,
            hit_tokens=200_000,
            max_chunk_tokens=131_072,
        )
        self.assertGreater(cold_ns, cached_ns)
        self.assertGreater(cold_ns - cached_ns, 0)
        self.assertEqual(
            cold_ns,
            self.provider.singleton_prefill_comp_ns(
                input_tokens=300_000,
                hit_tokens=0,
                max_chunk_tokens=131_072,
            ),
        )
        with self.assertRaisesRegex(
                OnlineLatencyModelError, "hit_tokens"):
            self.provider.singleton_prefill_comp_ns(
                input_tokens=300_000,
                hit_tokens=300_000,
                max_chunk_tokens=131_072,
            )

    def test_unsupported_context_and_decode_batch_are_rejected(self):
        with self.assertRaisesRegex(
                OnlineLatencyModelError, "context contract"):
            self.provider.attention_latency_ns(OnlineBatchShape(
                10_001, (10_001,), (1_000_000,), (), 1
            ))
        with self.assertRaisesRegex(
                OnlineLatencyModelError, "128"):
            self.provider.attention_latency_ns(OnlineBatchShape(
                129, (), (), (1_024,) * 129, 129
            ))

    def test_metadata_is_explicit_about_calibration_boundary(self):
        metadata = self.provider.metadata()
        self.assertEqual(
            metadata["scope"], "online_trace_comp_nodes_only"
        )
        self.assertFalse(metadata["collectives_included_in_comp_time"])
        self.assertTrue(metadata["astra_collectives_remain_authoritative"])
        self.assertTrue(
            metadata["limits"]["long_context_is_analytical_extrapolation"]
        )
        self.assertEqual(
            metadata["long_context_experiment"]["mode"],
            "dca_dense_full_attention_sensitivity",
        )
        self.assertEqual(
            metadata["long_context_experiment"]["paper_interpretation"][
                "absolute_1m_latency"
            ],
            "sensitivity_only",
        )
        self.assertFalse(
            metadata["limits"]["sparse_attention_enabled"]
        )
        self.assertFalse(
            metadata["limits"]["official_dca_kernel_reproduced"]
        )
        self.assertEqual(
            metadata["limits"]["native_max_position_embeddings"],
            262_144,
        )
        self.assertEqual(metadata["limits"]["max_context_tokens"], 1_010_000)
        self.assertTrue(
            metadata["limits"][
                "mixed_prefill_decode_is_analytical_extrapolation"
            ]
        )
        self.assertEqual(len(metadata["source_sha256"]), 4)


if __name__ == "__main__":
    unittest.main()

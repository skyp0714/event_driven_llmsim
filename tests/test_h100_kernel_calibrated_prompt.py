import math
import unittest
from pathlib import Path
from unittest import mock

import serving.core.h100_kernel_calibrated_prompt as calibrated_prompt
from serving.core.h100_kernel_calibrated_prompt import (
    BF16_BYTES,
    CALIBRATION_SOURCE_PATHS,
    H100KernelCalibratedPromptModel,
    LEGACY_PRODUCER_SOURCE_PATHS,
    QWEN_EP,
    QWEN_EXPERTS,
    QWEN_HIDDEN_SIZE,
    QWEN_LAYERS,
    QWEN_LONG_CONTEXT_MODE,
    TP_COLLECTIVE_BYTES_PER_SECOND,
    TP_COLLECTIVE_FIXED_SECONDS,
    _bottom_right_causal_pairs,
    fit_h100_tp4_calibration,
)


class H100KernelCalibratedPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.calibration = fit_h100_tp4_calibration(cls.repo_root)

    def _model(self, band="central", attention_multiplier=1.0):
        return H100KernelCalibratedPromptModel(
            calibration=self.calibration,
            band=band,
            attention_multiplier=attention_multiplier,
            prefill_chunk_size=131_072,
            target_config_sha256="a" * 64,
        )

    def test_fit_is_bound_to_four_legacy_h100_tp4_sources(self):
        long_context = self.calibration.metadata()[
            "target_long_context_experiment"
        ]
        self.assertEqual(long_context["mode"], QWEN_LONG_CONTEXT_MODE)
        self.assertFalse(
            long_context["experiment_override"][
                "sparse_attention_enabled"
            ]
        )
        self.assertEqual(
            set(self.calibration.source_sha256),
            set(CALIBRATION_SOURCE_PATHS),
        )
        self.assertEqual(len(self.calibration.source_sha256), 4)
        self.assertTrue(
            all(len(value) == 64 for value in self.calibration.source_sha256.values())
        )
        self.assertTrue(
            all(value > 0 for value in self.calibration.source_rows.values())
        )
        self.assertEqual(
            self.calibration.source_rows,
            {
                CALIBRATION_SOURCE_PATHS[0]: 30_720,
                CALIBRATION_SOURCE_PATHS[1]: 21_603,
                CALIBRATION_SOURCE_PATHS[2]: 30_720,
                CALIBRATION_SOURCE_PATHS[3]: 24_979,
            },
        )
        self.assertTrue(
            all("/tp4/" in path for path in CALIBRATION_SOURCE_PATHS)
        )
        self.assertTrue(
            all("predictions" not in path for path in CALIBRATION_SOURCE_PATHS)
        )
        for path in (CALIBRATION_SOURCE_PATHS[1], CALIBRATION_SOURCE_PATHS[3]):
            accounting = self.calibration.source_row_accounting[path]
            self.assertEqual(accounting["eligible_prefill_rows"], 232)
            self.assertEqual(accounting["unique_prefill_points"], 216)
            self.assertEqual(accounting["fit_training_points"], 16)
            self.assertEqual(accounting["fit_holdout_points"], 8)
        self.assertEqual(
            set(self.calibration.producer_source_sha256),
            set(LEGACY_PRODUCER_SOURCE_PATHS),
        )
        self.assertTrue(
            all(
                len(value) == 64
                for value in self.calibration.producer_source_sha256.values()
            )
        )

    def test_source_attention_uses_bottom_right_causal_pairs(self):
        self.assertEqual(_bottom_right_causal_pairs(4, 4), 10.0)
        self.assertEqual(_bottom_right_causal_pairs(4, 10), 34.0)
        with self.assertRaises(ValueError):
            _bottom_right_causal_pairs(8, 4)
        attention = self.calibration.fits["prefill_attention"]
        self.assertGreater(attention.eta_central, 7.0)
        self.assertLess(attention.eta_central, 8.0)

    def test_every_required_family_has_contiguous_holdout_metrics(self):
        required = {
            "embedding",
            "norm",
            "q_projection",
            "o_projection",
            "rope",
            "router",
            "expert_up",
            "expert_down",
            "expert_activation",
            "lm_head",
            "prefill_attention",
        }
        self.assertTrue(required.issubset(self.calibration.fits))
        for family in required:
            kernel_fit = self.calibration.fits[family]
            self.assertGreater(kernel_fit.training_rows, 0)
            self.assertGreater(kernel_fit.holdout_rows, 0)
            self.assertTrue(math.isfinite(kernel_fit.holdout_mape))
            self.assertGreaterEqual(kernel_fit.holdout_p90_ape, 0.0)
            self.assertLessEqual(
                kernel_fit.eta_fast,
                kernel_fit.eta_central,
            )
            self.assertLessEqual(
                kernel_fit.eta_central,
                kernel_fit.eta_slow,
            )
        split = self.calibration.validation["holdout"]["split"]
        self.assertIn("contiguous holdout", split)
        attention = self.calibration.fits["prefill_attention"]
        self.assertLess(attention.holdout_mape, 0.10)
        self.assertLess(attention.holdout_p90_ape, 0.15)

    def test_fast_central_slow_bands_are_ordered(self):
        fast = self._model("fast").recompute_seconds(8_192)
        central = self._model("central").recompute_seconds(8_192)
        slow = self._model("slow").recompute_seconds(8_192)
        self.assertLessEqual(fast, central)
        self.assertLessEqual(central, slow)
        self.assertGreater(fast, 0.0)

    def test_one_third_multiplier_changes_only_prefill_attention(self):
        full = self._model(attention_multiplier=1.0).estimate_breakdown(65_536)
        third = self._model(attention_multiplier=1 / 3).estimate_breakdown(
            65_536
        )
        self.assertAlmostEqual(
            third["attention"],
            full["attention"] / 3,
            places=12,
        )
        for component in set(full) - {"attention", "total"}:
            self.assertEqual(third[component], full[component])
        self.assertAlmostEqual(
            full["total"] - third["total"],
            full["attention"] - third["attention"],
            places=12,
        )

    def test_ep_collectives_match_default_allgather_reduce_scatter_contract(self):
        q_tokens = 1_025
        model = self._model()
        breakdown = model.estimate_breakdown(q_tokens)
        local_dispatch_bytes = (
            max(1, q_tokens // QWEN_EP)
            * (QWEN_HIDDEN_SIZE + QWEN_EXPERTS)
            * BF16_BYTES
        )
        total_combine_bytes = q_tokens * QWEN_HIDDEN_SIZE * BF16_BYTES
        expected_allgather = (
            TP_COLLECTIVE_FIXED_SECONDS
            + (QWEN_EP - 1)
            * local_dispatch_bytes
            / TP_COLLECTIVE_BYTES_PER_SECOND
        )
        expected_reduce_scatter = (
            TP_COLLECTIVE_FIXED_SECONDS
            + (QWEN_EP - 1)
            / QWEN_EP
            * total_combine_bytes
            / TP_COLLECTIVE_BYTES_PER_SECOND
        )
        self.assertAlmostEqual(
            breakdown["ep_allgather"],
            expected_allgather * QWEN_LAYERS,
            places=12,
        )
        self.assertAlmostEqual(
            breakdown["ep_reduce_scatter"],
            expected_reduce_scatter * QWEN_LAYERS,
            places=12,
        )
        self.assertNotIn("ep_alltoall", breakdown)

    def test_latency_cache_is_bounded_and_stores_scalars_only(self):
        model = self._model()
        with mock.patch.object(
            calibrated_prompt,
            "LATENCY_CACHE_MAX_ENTRIES",
            2,
        ):
            for tokens in (32, 64, 96):
                model.recompute_seconds(tokens)
        self.assertEqual(len(model._latency_cache), 2)
        self.assertTrue(
            all(isinstance(value, float) for value in model._latency_cache.values())
        )
        cache_before = dict(model._latency_cache)
        model.estimate_breakdown(128)
        self.assertEqual(dict(model._latency_cache), cache_before)

    def test_cached_suffix_retains_context_attention_but_avoids_full_recompute(self):
        model = self._model()
        full = model.recompute_seconds(131_072)
        cached = model.cached_prefill_seconds(131_072, 120_000)
        self.assertGreater(cached, 0.0)
        self.assertLess(cached, full)
        self.assertEqual(model.cached_prefill_seconds(100, 100), 0.0)
        full_context = model.recompute_seconds(1_010_000)
        long_suffix = model.cached_prefill_seconds(1_010_000, 900_000)
        self.assertTrue(math.isfinite(full_context))
        self.assertTrue(math.isfinite(long_suffix))
        self.assertGreater(full_context, long_suffix)

    def test_metadata_exposes_equation_validation_and_boundary(self):
        metadata = self._model().metadata()
        self.assertEqual(metadata["schema_version"], 4)
        self.assertEqual(
            metadata["long_context_experiment"]["paper_interpretation"][
                "absolute_1m_latency"
            ],
            "sensitivity_only",
        )
        self.assertIn("t_roof=max", metadata["analytical_model"]["equation"])
        self.assertIn("holdout", metadata["validation"])
        self.assertEqual(
            metadata["target_geometry"]["fused_qkv_output_per_rank"], 1_280
        )
        self.assertEqual(
            metadata["target_geometry"][
                "balanced_expert_pairs_per_rank_per_token"
            ],
            2.0,
        )
        self.assertFalse(metadata["limitations"]["measured_qwen3_h100"])
        self.assertTrue(
            metadata["limitations"][
                "one_million_token_attention_is_extrapolated"
            ]
        )
        self.assertEqual(
            metadata["limitations"]["attention_multiplier_scope"],
            "prefill attention only",
        )
        self.assertEqual(
            metadata["collective_model"]["backend"],
            "allgather_reducescatter",
        )
        self.assertFalse(
            metadata["collective_model"]["measured_or_fitted"]
        )
        self.assertEqual(
            metadata["limitations"]["source_measurement_dtype"],
            "float16",
        )
        self.assertEqual(
            metadata["limitations"]["target_model_dtype"],
            "bfloat16",
        )
        self.assertFalse(
            metadata["limitations"]["attention_holdout_validates_long_k"]
        )
        self.assertFalse(
            metadata["limitations"][
                "aggregate_holdout_is_target_end_to_end_accuracy"
            ]
        )
        self.assertEqual(
            metadata["target_geometry"]["config_sha256"], "a" * 64
        )
        self.assertEqual(
            set(metadata["producer_source_sha256"]),
            set(LEGACY_PRODUCER_SOURCE_PATHS),
        )
        self.assertEqual(
            metadata["source_identity"]["artifact_label"],
            "legacy-H100-labeled",
        )
        self.assertFalse(
            metadata["source_identity"]["exact_h100_sku_known"]
        )
        self.assertFalse(
            metadata["limitations"]["absolute_dgx_h100_accuracy_validated"]
        )
        self.assertTrue(
            metadata["limitations"][
                "absolute_1m_latency_sensitivity_only"
            ]
        )

    def test_target_config_hash_must_be_lowercase_sha256(self):
        with self.assertRaises(ValueError):
            H100KernelCalibratedPromptModel(
                calibration=self.calibration,
                target_config_sha256="not-a-digest",
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from serving.core.hbf_comparison_workload import load_comparison_workload
from serving.core.tracelab_comparison_scenarios import (
    BALANCED_SOURCE_INDICES,
    FULL_COHORT_ANCHOR_RATES,
    LONG_COLD_ANCHOR_RATES,
    LONG_COLD_BLOCK_ROUNDED_FINAL_KV_BYTES,
    LONG_COLD_COMBINED_D_HBM_AND_CPU_BYTES,
    LONG_COLD_END_CALL_INDICES,
    LONG_COLD_FINAL_KV_EXCESS_BYTES,
    LONG_COLD_SOURCE_INDICES,
    LONG_COLD_TARGET_CALL_INDICES,
    PINNED_RESUME_TO_FIRST_SERVICE_RATIO,
    ROLE_GUARD,
    ROLE_MEASUREMENT,
    ROLE_WARMUP,
    BalancedCausalPrefixManifest,
    FullCohortSensitivityManifest,
    LongColdContextStressManifest,
    build_balanced_causal_prefix_scenario,
    build_full_cohort_sensitivity_scenario,
    build_long_cold_context_stress_scenario,
    load_balanced_causal_prefix_scenario,
    load_full_cohort_sensitivity_scenario,
    load_long_cold_context_stress_scenario,
)


TRACE_PATH = (
    Path.home() / "llmsim-data/tracelab-schema3-sps0.2-final.jsonl"
)


def _call(
        input_tokens: int,
        output_tokens: int,
        prefix_tokens: int,
        *,
        tool_duration_ns: int,
) -> dict[str, object]:
    return {
        "input_toks": input_tokens,
        "output_toks": output_tokens,
        "tool_duration_ns": tool_duration_ns,
        "prefix_reuse_toks": prefix_tokens,
        "lineage_status": (
            "session_start" if prefix_tokens == 0 else "adjacent"
        ),
        "inter_turn_gap_type": "tool",
    }


def _row(session_id: str, calls: list[dict[str, object]]) -> dict[str, object]:
    return {
        "session_id": session_id,
        "arrival_time_ns": 0,
        "trace_metadata": {
            "source_session_identity_sha256": hashlib.sha256(
                session_id.encode("utf-8")
            ).hexdigest(),
        },
        "sub_requests": calls,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True))
            output.write("\n")


def _fixture_workload(path: Path):
    rows = [
        _row("three", [
            _call(100, 3, 0, tool_duration_ns=11),
            _call(125, 2, 90, tool_duration_ns=22),
            _call(150, 1, 120, tool_duration_ns=33),
        ]),
        _row("two", [
            _call(200, 3, 0, tool_duration_ns=44),
            _call(220, 2, 190, tool_duration_ns=55),
        ]),
        _row("four", [
            _call(300, 5, 0, tool_duration_ns=66),
            _call(340, 4, 290, tool_duration_ns=77),
            _call(360, 3, 330, tool_duration_ns=88),
            _call(390, 2, 350, tool_duration_ns=99),
        ]),
        _row("one", [
            _call(400, 2, 0, tool_duration_ns=111),
        ]),
    ]
    _write_jsonl(path, rows)
    return load_comparison_workload(
        path, source_indices=(0, 1, 2, 3)
    )


def _long_cold_fixture_workload(path: Path):
    rows = [
        _row("early-target", [
            _call(10_000, 3, 0, tool_duration_ns=11),
            _call(101_000, 2, 100_000, tool_duration_ns=22),
            _call(102_000, 4, 101_000, tool_duration_ns=33),
            _call(103_000, 5, 102_000, tool_duration_ns=44),
            _call(104_000, 6, 103_000, tool_duration_ns=55),
        ]),
        _row("later-target", [
            _call(20_000, 3, 0, tool_duration_ns=66),
            _call(99_000, 4, 98_000, tool_duration_ns=77),
            _call(102_000, 5, 101_000, tool_duration_ns=88),
            _call(103_000, 6, 102_000, tool_duration_ns=99),
            _call(104_000, 7, 103_000, tool_duration_ns=111),
            _call(105_000, 8, 104_000, tool_duration_ns=122),
        ]),
    ]
    _write_jsonl(path, rows)
    return load_comparison_workload(path, source_indices=(0, 1))


class TraceLabComparisonScenariosTest(unittest.TestCase):

    def test_balanced_builder_filters_truncates_repeats_and_maps_provenance(
            self):
        with tempfile.TemporaryDirectory() as directory:
            source = _fixture_workload(Path(directory) / "trace.jsonl")
            scenario = build_balanced_causal_prefix_scenario(
                source,
                epoch_count=4,
                warmup_epochs=(0,),
                measurement_epochs=(1, 2),
                guard_epochs=(3,),
                rates=(0.5, 2.0),
                maximum_rate=5.0,
            )

        manifest = scenario.manifest
        self.assertIsInstance(manifest, BalancedCausalPrefixManifest)
        self.assertEqual(manifest.selected_source_indices, (0, 2))
        self.assertEqual(
            manifest.selected_source_session_ids, ("three", "four")
        )
        self.assertEqual(len(scenario.workload.sessions), 8)
        self.assertEqual(scenario.workload.summary.call_count, 24)
        self.assertEqual(
            manifest.measurement_stats.cohort.session_count, 4
        )
        self.assertEqual(
            manifest.measurement_stats.cohort.first_turn_count, 4
        )
        self.assertEqual(
            manifest.measurement_stats.cohort.resume_count, 8
        )
        self.assertEqual(
            len(manifest.measurement_request_identities), 12
        )
        self.assertEqual(
            len(manifest.measurement_first_request_identities), 4
        )
        self.assertEqual(
            len(manifest.measurement_resume_request_identities), 8
        )

        mappings = manifest.epoch_mapping
        self.assertEqual(
            [mapping.role for mapping in mappings],
            [
                ROLE_WARMUP, ROLE_WARMUP,
                ROLE_MEASUREMENT, ROLE_MEASUREMENT,
                ROLE_MEASUREMENT, ROLE_MEASUREMENT,
                ROLE_GUARD, ROLE_GUARD,
            ],
        )
        self.assertEqual(
            len({mapping.synthetic_source_index for mapping in mappings}),
            8,
        )
        self.assertEqual(
            len({mapping.synthetic_session_id for mapping in mappings}),
            8,
        )
        self.assertEqual(
            len({
                mapping.synthetic_source_identity_sha256
                for mapping in mappings
            }),
            8,
        )
        self.assertEqual(
            [(mapping.source_index, mapping.source_session_id)
             for mapping in mappings[:2]],
            [(0, "three"), (2, "four")],
        )
        for session, mapping in zip(
                scenario.workload.sessions, mappings):
            self.assertEqual(
                session.source_index, mapping.synthetic_source_index
            )
            self.assertEqual(
                session.session_id, mapping.synthetic_session_id
            )
            self.assertEqual(len(session.calls), 3)
            self.assertTrue(all(
                call.session_id == session.session_id
                and call.source_index == session.source_index
                for call in session.calls
            ))

        # Truncation retains the predecessor tool gaps and only session starts
        # appear in the offered plan. Successors remain dynamically causal.
        first_synthetic = scenario.workload.sessions[0]
        self.assertEqual(
            [call.tool_duration_ns for call in first_synthetic.calls],
            [11, 22, 33],
        )
        plan = scenario.build_offered_plan(seed=101)
        self.assertEqual(len(plan.offers), 8)
        self.assertEqual(
            [offer.offer_index for offer in plan.offers],
            list(range(8)),
        )
        self.assertEqual(
            [offer.session.session_id for offer in plan.offers],
            [session.session_id for session in scenario.workload.sessions],
        )
        self.assertTrue(all(len(offer.session.calls) == 3
                            for offer in plan.offers))
        json.dumps(manifest.to_dict(), sort_keys=True)

    def test_balanced_plan_reuses_draws_across_rates_and_enforces_maximum(
            self):
        with tempfile.TemporaryDirectory() as directory:
            source = _fixture_workload(Path(directory) / "trace.jsonl")
            scenario = build_balanced_causal_prefix_scenario(
                source,
                epoch_count=3,
                warmup_epochs=(0,),
                measurement_epochs=(1,),
                guard_epochs=(2,),
                rates=(1.0, 5.0),
            )

        plan = scenario.build_offered_plan(seed=211)
        again = scenario.build_offered_plan(seed=211)
        self.assertEqual(plan.plan, again.plan)
        slow = plan.at_rate(1.0, start_time_ns=17)
        fast = plan.at_rate(5.0, start_time_ns=17)
        self.assertEqual(
            [row.session.session_id for row in slow],
            [row.session.session_id for row in fast],
        )
        self.assertEqual(
            [row.unit_interarrival for row in slow],
            [row.unit_interarrival for row in fast],
        )
        self.assertTrue(all(
            abs(
                (slow_row.arrival_time_ns - 17)
                - 5 * (fast_row.arrival_time_ns - 17)
            ) <= 2
            for slow_row, fast_row in zip(slow, fast)
        ))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            plan.at_rate(5.01)

    def test_epoch_partition_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _fixture_workload(Path(directory) / "trace.jsonl")
            with self.assertRaisesRegex(
                    ValueError, "multiple metric roles"):
                build_balanced_causal_prefix_scenario(
                    source,
                    epoch_count=3,
                    warmup_epochs=(0, 1),
                    measurement_epochs=(1,),
                    guard_epochs=(2,),
                )
            with self.assertRaisesRegex(ValueError, "missing"):
                build_balanced_causal_prefix_scenario(
                    source,
                    epoch_count=4,
                    warmup_epochs=(0,),
                    measurement_epochs=(1,),
                    guard_epochs=(3,),
                )

    def test_full_cohort_sensitivity_is_explicitly_nonsteady_and_anchored(
            self):
        with tempfile.TemporaryDirectory() as directory:
            source = _fixture_workload(Path(directory) / "trace.jsonl")
            scenario = build_full_cohort_sensitivity_scenario(source)

        manifest = scenario.manifest
        self.assertIsInstance(manifest, FullCohortSensitivityManifest)
        self.assertFalse(manifest.equilibrium_workload)
        self.assertIn("non_steady", manifest.workload_semantics)
        self.assertIn(
            "offered_load_normalized",
            manifest.offered_load_normalization,
        )
        self.assertEqual(
            manifest.arrival_contract.rates,
            FULL_COHORT_ANCHOR_RATES,
        )
        plan = scenario.build_offered_plan(seed=307)
        self.assertEqual(len(plan.offers), 4)
        plan.at_rate(3.0)
        with self.assertRaisesRegex(ValueError, "anchor rates"):
            plan.at_rate(2.0)

    def test_long_cold_builder_preserves_prefix_and_measures_only_window(
            self):
        with tempfile.TemporaryDirectory() as directory:
            source = _long_cold_fixture_workload(
                Path(directory) / "trace.jsonl"
            )
            scenario = build_long_cold_context_stress_scenario(
                source,
                source_indices=(0, 1),
                epoch_count=4,
                warmup_epochs=(0,),
                measurement_epochs=(1, 2),
                guard_epochs=(3,),
            )

        manifest = scenario.manifest
        self.assertIsInstance(manifest, LongColdContextStressManifest)
        self.assertEqual(
            tuple(
                window.target_call_index
                for window in manifest.selection_windows
            ),
            (1, 2),
        )
        self.assertEqual(
            tuple(
                window.end_call_index
                for window in manifest.selection_windows
            ),
            (3, 4),
        )
        self.assertEqual(
            [len(session.calls) for session in scenario.workload.sessions],
            [4, 5] * 4,
        )
        self.assertEqual(
            scenario.workload.summary.call_count, 36
        )
        self.assertEqual(
            manifest.full_replay_stats.cohort.first_turn_count, 8
        )
        self.assertEqual(
            manifest.full_replay_stats.cohort.resume_count, 28
        )
        self.assertEqual(
            manifest.measurement_stats.cohort.session_count, 4
        )
        self.assertEqual(
            manifest.measurement_stats.cohort.call_count, 12
        )
        self.assertEqual(
            manifest.measurement_stats.cohort.first_turn_count, 0
        )
        self.assertEqual(
            manifest.measurement_stats.cohort.resume_count, 12
        )
        self.assertEqual(
            len(manifest.measurement_first_request_identities), 0
        )
        self.assertEqual(
            len(manifest.measurement_resume_request_identities), 12
        )
        self.assertEqual(
            manifest.measurement_request_identities,
            manifest.measurement_resume_request_identities,
        )
        self.assertFalse(manifest.equilibrium_workload)
        self.assertIn(
            "must_not_be_labeled_maximum_sustainable_throughput",
            manifest.offered_load_normalization,
        )

        source_calls = {
            session.source_index: session.calls
            for session in source.sessions
        }
        for synthetic, mapping in zip(
                scenario.workload.sessions, manifest.epoch_mapping):
            original = source_calls[mapping.source_index]
            self.assertEqual(
                [
                    (
                        call.call_index,
                        call.input_tokens,
                        call.output_tokens,
                        call.cached_prefix_tokens,
                        call.fresh_input_tokens,
                        call.tool_duration_ns,
                        call.lineage_status,
                        call.inter_turn_gap_type,
                    )
                    for call in synthetic.calls
                ],
                [
                    (
                        call.call_index,
                        call.input_tokens,
                        call.output_tokens,
                        call.cached_prefix_tokens,
                        call.fresh_input_tokens,
                        call.tool_duration_ns,
                        call.lineage_status,
                        call.inter_turn_gap_type,
                    )
                    for call in original[:len(synthetic.calls)]
                ],
            )

        plan = scenario.build_offered_plan(seed=401)
        self.assertEqual(len(plan.offers), 8)
        self.assertEqual(
            manifest.arrival_contract.rates, LONG_COLD_ANCHOR_RATES
        )
        plan.at_rate(3.0)
        with self.assertRaisesRegex(ValueError, "anchor rates"):
            plan.at_rate(4.0)
        json.dumps(manifest.to_dict(), sort_keys=True)

    def test_long_cold_builder_fails_without_target_or_successors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            _write_jsonl(path, [
                _row("no-target", [
                    _call(10_000, 2, 0, tool_duration_ns=1),
                    _call(99_000, 2, 98_000, tool_duration_ns=2),
                ]),
                _row("no-successors", [
                    _call(10_000, 2, 0, tool_duration_ns=3),
                    _call(101_000, 2, 100_000, tool_duration_ns=4),
                ]),
                _row("shrinking-window", [
                    _call(10_000, 2, 0, tool_duration_ns=5),
                    _call(101_000, 2, 100_000, tool_duration_ns=6),
                    _call(99_500, 2, 99_000, tool_duration_ns=7),
                    _call(102_000, 2, 101_000, tool_duration_ns=8),
                ]),
            ])
            source = load_comparison_workload(
                path, source_indices=(0, 1, 2)
            )
            with self.assertRaisesRegex(
                    ValueError, "never reaches"):
                build_long_cold_context_stress_scenario(
                    source, source_indices=(0,)
                )
            with self.assertRaisesRegex(
                    ValueError, "successor calls"):
                build_long_cold_context_stress_scenario(
                    source, source_indices=(1,)
                )
            with self.assertRaisesRegex(
                    ValueError, "fell below"):
                build_long_cold_context_stress_scenario(
                    source, source_indices=(2,)
                )

    @unittest.skipUnless(TRACE_PATH.exists(), "TraceLab release not present")
    def test_pinned_balanced_contract(self):
        scenario = load_balanced_causal_prefix_scenario(TRACE_PATH)
        manifest = scenario.manifest
        self.assertIsInstance(manifest, BalancedCausalPrefixManifest)
        self.assertEqual(
            manifest.selected_source_indices, BALANCED_SOURCE_INDICES
        )
        self.assertEqual(len(manifest.selected_source_indices), 27)
        self.assertEqual(manifest.source_session_count, 4281)
        self.assertEqual(len(scenario.workload.sessions), 216)
        self.assertEqual(scenario.workload.summary.call_count, 648)
        self.assertEqual(
            manifest.measurement_stats.cohort.session_count, 108
        )
        self.assertEqual(
            manifest.measurement_stats.cohort.first_turn_count, 108
        )
        self.assertEqual(
            manifest.measurement_stats.cohort.resume_count, 216
        )
        self.assertEqual(
            len(manifest.measurement_request_identities), 324
        )
        self.assertEqual(
            manifest.base_stats.first_fresh_input_tokens, 529_059
        )
        self.assertEqual(
            manifest.base_stats.resume_fresh_input_tokens, 224_579
        )
        self.assertIsNotNone(manifest.isolated_prefill_service)
        self.assertAlmostEqual(
            manifest.isolated_prefill_service.resume_to_first_ratio,
            PINNED_RESUME_TO_FIRST_SERVICE_RATIO,
            places=12,
        )
        self.assertAlmostEqual(
            PINNED_RESUME_TO_FIRST_SERVICE_RATIO, 0.8885832292,
            places=9,
        )

    @unittest.skipUnless(TRACE_PATH.exists(), "TraceLab release not present")
    def test_pinned_full_cohort_sensitivity_contract(self):
        scenario = load_full_cohort_sensitivity_scenario(TRACE_PATH)
        manifest = scenario.manifest
        self.assertIsInstance(manifest, FullCohortSensitivityManifest)
        self.assertEqual(
            manifest.workload_stats.cohort.session_count, 32
        )
        self.assertEqual(
            manifest.workload_stats.cohort.call_count, 2680
        )
        self.assertEqual(
            manifest.arrival_contract.rates,
            FULL_COHORT_ANCHOR_RATES,
        )
        self.assertFalse(manifest.equilibrium_workload)

    @unittest.skipUnless(TRACE_PATH.exists(), "TraceLab release not present")
    def test_pinned_long_cold_context_contract(self):
        scenario = load_long_cold_context_stress_scenario(TRACE_PATH)
        manifest = scenario.manifest
        self.assertIsInstance(manifest, LongColdContextStressManifest)
        self.assertEqual(
            manifest.selected_source_indices, LONG_COLD_SOURCE_INDICES
        )
        self.assertEqual(
            tuple(
                window.target_call_index
                for window in manifest.selection_windows
            ),
            LONG_COLD_TARGET_CALL_INDICES,
        )
        self.assertEqual(
            tuple(
                window.end_call_index
                for window in manifest.selection_windows
            ),
            LONG_COLD_END_CALL_INDICES,
        )
        self.assertEqual(len(scenario.workload.sessions), 160)
        self.assertEqual(scenario.workload.summary.call_count, 3_168)
        self.assertEqual(
            manifest.base_prefix_stats.cohort.session_count, 5
        )
        self.assertEqual(
            manifest.base_prefix_stats.cohort.call_count, 99
        )
        self.assertEqual(
            manifest.base_prefix_stats.cohort.first_turn_count, 5
        )
        self.assertEqual(
            manifest.base_prefix_stats.cohort.resume_count, 94
        )
        self.assertEqual(
            manifest.measurement_stats.cohort.session_count, 80
        )
        self.assertEqual(
            manifest.measurement_stats.cohort.call_count, 240
        )
        self.assertEqual(
            manifest.measurement_stats.cohort.first_turn_count, 0
        )
        self.assertEqual(
            manifest.measurement_stats.cohort.resume_count, 240
        )
        self.assertEqual(
            manifest.measurement_stats.cohort.total_cached_prefix_tokens,
            25_116_752,
        )
        self.assertEqual(
            manifest.measurement_stats.cohort.total_fresh_input_tokens,
            428_208,
        )
        self.assertEqual(
            manifest.measurement_stats.max_cached_prefix_tokens,
            111_123,
        )
        self.assertEqual(
            len(manifest.measurement_request_identities), 240
        )
        self.assertEqual(
            manifest.arrival_contract.rates,
            LONG_COLD_ANCHOR_RATES,
        )
        self.assertEqual(
            LONG_COLD_BLOCK_ROUNDED_FINAL_KV_BYTES
            - LONG_COLD_COMBINED_D_HBM_AND_CPU_BYTES,
            LONG_COLD_FINAL_KV_EXCESS_BYTES,
        )
        self.assertGreater(LONG_COLD_FINAL_KV_EXCESS_BYTES, 0)
        self.assertFalse(manifest.equilibrium_workload)


if __name__ == "__main__":
    unittest.main()

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from serving.core.live_balanced_storage_scenario import (
    BASELINE_USABLE_D_HBM_AND_CPU_BYTES_PER_NODE,
    SCENARIO_ID as PUBLICATION_SCENARIO_ID,
    _load_templates,
)
from serving.core.live_balanced_stress_scenario import (
    EXPECTED_MEASUREMENT_REQUEST_COUNT,
    EXPECTED_MEASUREMENT_SESSION_COUNT,
    EXPECTED_SSD_CAPACITY_BYTES_PER_NODE,
    EXPECTED_STRESS_SCHEDULE_MATRIX_SHA256,
    MAXIMUM_AUDITED_RATE,
    MEASUREMENT_EPOCH_COUNT,
    REQUESTED_MINIMUM_WARMUP_GUARD_EPOCHS,
    REQUIRED_ARRIVAL_SPAN_NS,
    SCENARIO_ID,
    STRESS_RATES,
    STRESS_SEEDS,
    STRESS_WARMUP_GUARD_EPOCHS,
    StressEpoch,
    _ordered_sessions_for_seed,
    _scheduled_arrivals_sha256,
    _validated_ssd_capacity,
    build_high_rate_stress,
)


TRACE = Path(
    os.environ.get("LLMSIM_DATA", str(Path.home() / "llmsim-data"))
) / "tracelab-schema3-sps0.2-final.jsonl"


def _source_call_content(call):
    return (
        call.call_index,
        call.input_tokens,
        call.output_tokens,
        call.tool_duration_ns,
        call.cached_prefix_tokens,
        call.fresh_input_tokens,
        call.lineage_status,
        call.inter_turn_gap_type,
    )


@unittest.skipUnless(TRACE.is_file(), "pinned TraceLab release is unavailable")
class LiveBalancedStressScenarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates, _ = _load_templates(TRACE)
        cls.scenario = build_high_rate_stress(TRACE)

    def test_factory_is_distinct_and_publication_contract_is_unchanged(self):
        manifest = self.scenario.manifest

        self.assertNotEqual(SCENARIO_ID, PUBLICATION_SCENARIO_ID)
        self.assertEqual(manifest.scenario_id, SCENARIO_ID)
        self.assertEqual(manifest.epoch_profile, "high_rate_stress_v2")
        self.assertEqual(manifest.recommended_rates, STRESS_RATES)
        self.assertEqual(manifest.recommended_seeds, STRESS_SEEDS)
        self.assertEqual(manifest.maximum_audited_rate, 3.0)
        self.assertIn("finite stress", manifest.experiment_sequence_semantics)

    def test_rate_specific_counts_are_first_seeded_span_passes(self):
        profiles = {
            profile.offered_session_rate_per_second: profile
            for profile in self.scenario.manifest.rate_profiles
        }
        self.assertEqual(set(profiles), set(STRESS_RATES))

        for rate in STRESS_RATES:
            with self.subTest(rate=rate):
                profile = profiles[rate]
                self.assertEqual(
                    profile.requested_minimum_warmup_guard_epochs,
                    REQUESTED_MINIMUM_WARMUP_GUARD_EPOCHS[rate],
                )
                self.assertEqual(
                    profile.selected_warmup_guard_epochs,
                    STRESS_WARMUP_GUARD_EPOCHS[rate],
                )
                self.assertGreaterEqual(
                    profile.selected_warmup_guard_epochs,
                    profile.requested_minimum_warmup_guard_epochs,
                )
                self.assertEqual(
                    profile.measurement_epochs,
                    MEASUREMENT_EPOCH_COUNT,
                )
                self.assertEqual(
                    profile.offered_first_calls,
                    profile.offered_resume_calls,
                )
                self.assertGreaterEqual(
                    profile.minimum_warmup_arrival_span_ns,
                    REQUIRED_ARRIVAL_SPAN_NS,
                )
                self.assertGreaterEqual(
                    profile.minimum_guard_arrival_span_ns,
                    REQUIRED_ARRIVAL_SPAN_NS,
                )
                self.assertGreater(
                    profile.minimum_active_offered_tail_ns, 0)

    def test_measurement_roster_is_rate_invariant_and_exactly_balanced(self):
        manifest = self.scenario.manifest
        self.assertEqual(
            manifest.measurement_session_count,
            EXPECTED_MEASUREMENT_SESSION_COUNT,
        )
        self.assertEqual(
            manifest.measurement_request_count,
            EXPECTED_MEASUREMENT_REQUEST_COUNT,
        )
        self.assertEqual(manifest.measurement_first_call_count, 112)
        self.assertEqual(manifest.measurement_resume_call_count, 112)
        self.assertEqual(
            len(manifest.measurement_session_ids), 112)
        self.assertEqual(
            len(set(manifest.measurement_session_ids)), 112)

        expected_ids = None
        for rate in STRESS_RATES:
            ids = tuple(
                session.session_id
                for epoch in self.scenario.epochs_for_rate(rate)
                if epoch.role == "measurement"
                for session in epoch.sessions
            )
            if expected_ids is None:
                expected_ids = ids
            self.assertEqual(ids, expected_ids)
        self.assertEqual(expected_ids, manifest.measurement_session_ids)

    def test_every_cloned_epoch_preserves_complete_tracelab_calls(self):
        expected = tuple(
            tuple(_source_call_content(call) for call in source.calls)
            for source in self.templates
        )
        for rate in STRESS_RATES:
            with self.subTest(rate=rate):
                for epoch in self.scenario.epochs_for_rate(rate):
                    self.assertEqual(len(epoch.sessions), 7)
                    for offset, session in enumerate(epoch.sessions):
                        self.assertEqual(
                            tuple(
                                _source_call_content(call)
                                for call in session.calls
                            ),
                            expected[offset],
                        )

    def test_all_seed_rate_audits_are_pressured_but_fit_finite_storage(self):
        manifest = self.scenario.manifest
        self.assertEqual(
            manifest.ssd_capacity_bytes_per_node,
            EXPECTED_SSD_CAPACITY_BYTES_PER_NODE,
        )
        self.assertEqual(
            manifest.baseline_pre_ssd_capacity_bytes_per_node,
            BASELINE_USABLE_D_HBM_AND_CPU_BYTES_PER_NODE,
        )
        for profile in manifest.rate_profiles:
            self.assertEqual(
                tuple(audit.seed for audit in profile.seed_audits),
                STRESS_SEEDS,
            )
            for audit in profile.seed_audits:
                with self.subTest(
                        rate=profile.offered_session_rate_per_second,
                        seed=audit.seed):
                    self.assertGreaterEqual(
                        audit.warmup_arrival_span_ns,
                        REQUIRED_ARRIVAL_SPAN_NS,
                    )
                    self.assertGreaterEqual(
                        audit.guard_arrival_span_ns,
                        REQUIRED_ARRIVAL_SPAN_NS,
                    )
                    self.assertEqual(
                        audit.measurement_session_count, 112)
                    self.assertEqual(
                        audit.offered_first_call_count,
                        audit.offered_resume_call_count,
                    )
                    self.assertEqual(
                        audit.measurement_request_count, 224)
                    self.assertEqual(
                        audit.measurement_request_release_count, 224)
                    self.assertEqual(
                        audit
                        .measurement_request_releases_before_last_guard_offer,
                        224,
                    )
                    self.assertEqual(
                        audit.measurement_resume_return_count, 112)
                    self.assertEqual(
                        audit
                        .measurement_resume_returns_before_last_guard_offer,
                        112,
                    )
                    self.assertEqual(
                        audit
                        .measurement_resume_returns_under_pre_ssd_node_pressure,
                        112,
                    )
                    self.assertTrue(
                        audit.both_nodes_exceed_pre_ssd_capacity)
                    self.assertTrue(
                        audit.fits_ssd_capacity_on_each_node)
                    self.assertTrue(audit.fits_hbf_tp4_capacity)
                    self.assertTrue(audit.fits_hbf_tp8_capacity)
                    self.assertTrue(
                        audit.fits_hbf_tp8_context_capacity)
                    self.assertTrue(all(
                        peak
                        > BASELINE_USABLE_D_HBM_AND_CPU_BYTES_PER_NODE
                        for peak in (
                            audit
                            .per_node_peak_recorded_gap_logical_kv_bytes)
                    ))

    def test_schedule_matrix_and_a_dry_schedule_are_content_addressed(self):
        manifest = self.scenario.manifest
        self.assertEqual(
            manifest.schedule_matrix_sha256,
            EXPECTED_STRESS_SCHEDULE_MATRIX_SHA256,
        )
        profile = manifest.rate_profiles[0]
        audit = profile.seed_audits[0]
        plan = self.scenario.build_offered_plan(seed=101)
        schedule = plan.at_rate(1.4)

        self.assertEqual(len(schedule), profile.offered_sessions)
        self.assertEqual(
            _scheduled_arrivals_sha256(schedule),
            audit.scheduled_arrivals_sha256,
        )
        self.assertEqual(schedule[0].arrival_time_ns, 0)
        self.assertGreater(
            schedule[-1].arrival_time_ns,
            REQUIRED_ARRIVAL_SPAN_NS,
        )
        guard = self.scenario.runtime_guard_contract(
            seed=101,
            sessions_per_second=1.4,
        )
        self.assertTrue(
            manifest.runtime_guard_validation_required)
        self.assertEqual(
            manifest.runtime_guard_expected_measurement_resume_count,
            112,
        )
        self.assertEqual(
            guard["last_external_guard_offer_ns"],
            schedule[-1].arrival_time_ns,
        )
        self.assertEqual(
            guard["last_external_guard_offer_ns"],
            audit.last_external_guard_offer_ns,
        )
        self.assertEqual(
            guard["expected_measurement_resume_count"], 112)

    def test_unsupported_seed_rate_and_rate_cap_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.scenario.build_offered_plan(seed=100)
        with self.assertRaisesRegex(ValueError, "seed must be an integer"):
            self.scenario.build_offered_plan(seed=True)

        plan = self.scenario.build_offered_plan(seed=101)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            plan.at_rate(1.2)
        with self.assertRaisesRegex(ValueError, "audited maximum"):
            plan.at_rate(MAXIMUM_AUDITED_RATE + 0.01)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            plan.at_rate(float("nan"))

    def test_ssd_capacity_provenance_fails_closed_on_config_drift(self):
        config = (
            Path(__file__).resolve().parents[1]
            / "configs/agentic_kv/qwen3_1m_p4d4/tiered_fullprompt.json"
        )
        capacity, digest = _validated_ssd_capacity(config)
        self.assertEqual(capacity, EXPECTED_SSD_CAPACITY_BYTES_PER_NODE)
        self.assertEqual(
            digest, self.scenario.manifest.tiered_config_sha256)

        changed = json.loads(config.read_text(encoding="utf-8"))
        changed["ssd_num_devices"] = 7
        with tempfile.TemporaryDirectory() as directory:
            changed_path = Path(directory) / "tiered.json"
            changed_path.write_text(
                json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(
                    ValueError, "SSD capacity contract changed"):
                _validated_ssd_capacity(changed_path)

    def test_scheduling_order_does_not_read_future_call_fields(self):
        epoch = self.scenario.epochs_for_rate(1.4)[0]
        mutated_sessions = []
        for session in epoch.sessions:
            mutated_calls = tuple(
                replace(
                    call,
                    output_tokens=call.output_tokens + 10_000,
                    call_index=call.call_index + 10_000,
                )
                for call in session.calls
            )
            mutated_sessions.append(
                replace(session, calls=mutated_calls))
        mutated_epoch = StressEpoch(
            role=epoch.role,
            role_epoch_index=epoch.role_epoch_index,
            sessions=tuple(mutated_sessions),
        )

        original_order = tuple(
            session.session_id
            for session in _ordered_sessions_for_seed(
                (epoch,), seed=101)
        )
        mutated_order = tuple(
            session.session_id
            for session in _ordered_sessions_for_seed(
                (mutated_epoch,), seed=101)
        )
        self.assertEqual(original_order, mutated_order)
        self.assertIn(
            "Future output tokens and call indices",
            self.scenario.manifest.causal_policy_semantics,
        )


if __name__ == "__main__":
    unittest.main()

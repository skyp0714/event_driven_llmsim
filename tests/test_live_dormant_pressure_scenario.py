import os
from pathlib import Path
import unittest

from serving.core.hbf_comparison_workload import TRACELAB_SCHEMA3_SHA256
from serving.core.live_dormant_pressure_scenario import (
    BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES,
    EXPECTED_CALLS_PER_EPOCH,
    EXPECTED_CONTEXT_FACTOR_DENOMINATOR,
    EXPECTED_CONTEXT_FACTOR_NUMERATOR,
    EXPECTED_FIRST_CALLS_PER_EPOCH,
    EXPECTED_FIRST_PREFILL_SERVICE_NS,
    EXPECTED_PILOT_ZERO_SERVICE_PEAK_LOGICAL_KV_BYTES,
    EXPECTED_RECORDED_GAPS_SHA256,
    EXPECTED_RESUME_CALLS_PER_EPOCH,
    EXPECTED_RESUME_PREFILL_SERVICE_NS,
    EXPECTED_SESSIONS_PER_EPOCH,
    EXPECTED_SOURCE_IDENTITY_SHA256,
    EXPECTED_TRANSFORMED_COHORT_SHA256,
    RECOMMENDED_PILOT_RATES,
    SELECTED_SOURCE_INDICES,
    build_live_dormant_pressure_scenario,
    build_pilot,
    build_smoke,
)


TRACE = Path(
    os.environ.get("LLMSIM_DATA", str(Path.home() / "llmsim-data"))
) / "tracelab-schema3-sps0.2-final.jsonl"


@unittest.skipUnless(TRACE.is_file(), "pinned TraceLab release is unavailable")
class LiveDormantPressureScenarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = build_live_dormant_pressure_scenario(
            TRACE,
            warmup_epoch_count=1,
            measurement_epoch_count=2,
            guard_epoch_count=1,
        )

    def test_complete_cohort_and_source_lineage_are_pinned(self):
        manifest = self.scenario.manifest
        audit = manifest.cohort_audit

        self.assertEqual(manifest.source_sha256, TRACELAB_SCHEMA3_SHA256)
        self.assertEqual(
            manifest.transformed_cohort_sha256,
            EXPECTED_TRANSFORMED_COHORT_SHA256,
        )
        self.assertEqual(
            manifest.selected_source_indices, SELECTED_SOURCE_INDICES)
        self.assertEqual(
            manifest.context_factor_numerator,
            EXPECTED_CONTEXT_FACTOR_NUMERATOR,
        )
        self.assertEqual(
            manifest.context_factor_denominator,
            EXPECTED_CONTEXT_FACTOR_DENOMINATOR,
        )
        self.assertEqual(
            audit.source_identity_sha256,
            EXPECTED_SOURCE_IDENTITY_SHA256,
        )
        self.assertEqual(
            audit.recorded_gaps_sha256,
            EXPECTED_RECORDED_GAPS_SHA256,
        )
        self.assertIn("every source sub-request", audit.completeness_semantics)
        self.assertIn("tool durations", audit.transform_semantics)
        self.assertIn(
            "custom complete-session control",
            manifest.evaluation_role_semantics,
        )

    def test_configurable_epoch_roster_implements_runner_protocol(self):
        manifest = self.scenario.manifest

        self.assertEqual(manifest.epoch_profile, "custom")
        self.assertEqual(manifest.epoch_count, 4)
        self.assertEqual(manifest.warmup_epochs, (0,))
        self.assertEqual(manifest.measurement_epochs, (1, 2))
        self.assertEqual(manifest.guard_epochs, (3,))
        self.assertEqual(
            len(manifest.measurement_session_ids),
            EXPECTED_SESSIONS_PER_EPOCH * 2,
        )
        self.assertEqual(
            manifest.measurement_request_count,
            EXPECTED_CALLS_PER_EPOCH * 2,
        )
        self.assertEqual(
            manifest.measurement_first_call_count,
            EXPECTED_FIRST_CALLS_PER_EPOCH * 2,
        )
        self.assertEqual(
            manifest.measurement_resume_call_count,
            EXPECTED_RESUME_CALLS_PER_EPOCH * 2,
        )

        plan = self.scenario.build_offered_plan(seed=101)
        scheduled = plan.at_rate(RECOMMENDED_PILOT_RATES[1])
        self.assertEqual(
            len(scheduled), EXPECTED_SESSIONS_PER_EPOCH * 4)
        self.assertEqual(
            sum(len(item.session.calls) for item in scheduled),
            EXPECTED_CALLS_PER_EPOCH * 4,
        )
        self.assertEqual(
            tuple(sorted(item.arrival_time_ns for item in scheduled)),
            tuple(item.arrival_time_ns for item in scheduled),
        )

    def test_prefill_service_is_balanced_without_truncating_sessions(self):
        service = self.scenario.manifest.prefill_service

        self.assertEqual(
            service.first_prefill_service_ns_per_epoch,
            EXPECTED_FIRST_PREFILL_SERVICE_NS,
        )
        self.assertEqual(
            service.resume_prefill_service_ns_per_epoch,
            EXPECTED_RESUME_PREFILL_SERVICE_NS,
        )
        self.assertAlmostEqual(
            service.resume_to_first_service_ratio,
            1.0449759856186969,
        )
        for epoch in self.scenario.epoch_sessions:
            self.assertEqual(len(epoch), EXPECTED_SESSIONS_PER_EPOCH)
            self.assertEqual(
                sum(len(session.calls) for session in epoch),
                EXPECTED_CALLS_PER_EPOCH,
            )

    def test_analytical_pressure_is_explicitly_not_realized_pressure(self):
        pressure = self.scenario.manifest.kv_pressure
        estimates = {
            estimate.offered_session_rate_per_second:
                estimate.analytical_recorded_gap_live_kv_bytes_floor
            for estimate in pressure.analytical_steady_state_estimates
        }

        self.assertEqual(estimates[0.02], 285_143_277_231)
        self.assertEqual(estimates[0.10], 1_425_716_386_155)
        self.assertEqual(estimates[0.12], 1_710_859_663_386)
        self.assertGreater(
            estimates[0.10],
            BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES,
        )
        self.assertIn("Little's law", pressure.analytical_semantics)
        self.assertIn("not observed occupancy", pressure.finite_schedule_semantics)
        self.assertIn("runtime report", pressure.realized_runtime_semantics)

    def test_smoke_factory_and_seed_behavior_are_stable(self):
        smoke = build_smoke(TRACE)
        self.assertEqual(smoke.manifest.epoch_profile, "smoke")
        self.assertEqual(smoke.manifest.epoch_count, 3)
        self.assertEqual(
            smoke.manifest.measurement_request_count,
            EXPECTED_CALLS_PER_EPOCH,
        )

        pilot = build_pilot(TRACE)
        witness = pilot.manifest.kv_pressure.finite_schedule_witness
        self.assertEqual(pilot.manifest.epoch_profile, "pilot")
        self.assertEqual(pilot.manifest.epoch_count, 68)
        self.assertEqual(
            len(pilot.manifest.measurement_session_ids),
            EXPECTED_SESSIONS_PER_EPOCH * 4,
        )
        self.assertEqual(
            witness.zero_service_peak_recorded_gap_logical_kv_bytes,
            EXPECTED_PILOT_ZERO_SERVICE_PEAK_LOGICAL_KV_BYTES,
        )
        self.assertTrue(witness.exceeds_baseline_capacity)
        self.assertIn(
            "finite pressure pilot",
            pilot.manifest.evaluation_role_semantics,
        )
        self.assertIn(
            "not realized tier occupancy",
            witness.semantics,
        )

        left = self.scenario.build_offered_plan(seed=101)
        repeated = self.scenario.build_offered_plan(seed=101)
        right = self.scenario.build_offered_plan(seed=211)
        self.assertEqual(
            left.offered_session_ids_sha256,
            repeated.offered_session_ids_sha256,
        )
        self.assertEqual(left.unit_draws_sha256, repeated.unit_draws_sha256)
        self.assertNotEqual(
            left.offered_session_ids_sha256,
            right.offered_session_ids_sha256,
        )
        self.assertNotEqual(left.unit_draws_sha256, right.unit_draws_sha256)

    def test_epoch_count_validation_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "measurement_epoch_count"):
            build_live_dormant_pressure_scenario(
                TRACE,
                warmup_epoch_count=1,
                measurement_epoch_count=0,
                guard_epoch_count=1,
            )
        with self.assertRaisesRegex(ValueError, "warmup_epoch_count"):
            build_live_dormant_pressure_scenario(
                TRACE,
                warmup_epoch_count=True,
                measurement_epoch_count=1,
                guard_epoch_count=1,
            )


if __name__ == "__main__":
    unittest.main()

import os
from pathlib import Path
import unittest

from serving.core.hbf_comparison_workload import TRACELAB_SCHEMA3_SHA256
from serving.core.live_pressure_scenario import (
    EXPECTED_BALANCED_FIRST_SERVICE_NS,
    EXPECTED_CALLS_PER_EPOCH,
    EXPECTED_FIRST_CALLS_PER_EPOCH,
    EXPECTED_RESUME_CALLS_PER_EPOCH,
    EXPECTED_SESSIONS_PER_EPOCH,
    EXPECTED_TERMINAL_LOGICAL_KV_BYTES_ALL_EPOCHS,
    MEASUREMENT_EPOCHS,
    RECOMMENDED_PILOT_RATES,
    SELECTED_SOURCE_INDICES,
    build_live_pressure_scenario,
)


TRACE = Path(
    os.environ.get("LLMSIM_DATA", str(Path.home() / "llmsim-data"))
) / "tracelab-schema3-sps0.2-final.jsonl"


@unittest.skipUnless(TRACE.is_file(), "pinned TraceLab release is unavailable")
class LivePressureScenarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = build_live_pressure_scenario(TRACE)

    def test_manifest_pins_service_balance_and_pressure(self):
        manifest = self.scenario.manifest

        self.assertEqual(manifest.source_sha256, TRACELAB_SCHEMA3_SHA256)
        self.assertEqual(
            manifest.selected_source_indices, SELECTED_SOURCE_INDICES)
        self.assertEqual(
            manifest.sessions_per_epoch, EXPECTED_SESSIONS_PER_EPOCH)
        self.assertEqual(
            manifest.calls_per_epoch, EXPECTED_CALLS_PER_EPOCH)
        self.assertEqual(
            manifest.first_calls_per_epoch, EXPECTED_FIRST_CALLS_PER_EPOCH)
        self.assertEqual(
            manifest.resume_calls_per_epoch, EXPECTED_RESUME_CALLS_PER_EPOCH)
        self.assertEqual(
            manifest.prefill_service.balanced_first_service_ns,
            EXPECTED_BALANCED_FIRST_SERVICE_NS,
        )
        self.assertAlmostEqual(
            manifest.prefill_service.resume_to_first_service_ratio,
            1.132992266509715,
        )
        self.assertEqual(
            manifest.kv_pressure.terminal_logical_kv_bytes_all_epochs,
            EXPECTED_TERMINAL_LOGICAL_KV_BYTES_ALL_EPOCHS,
        )
        self.assertGreater(
            manifest.kv_pressure.terminal_excess_over_baseline_bytes, 0)

    def test_measurement_roster_and_schedule_are_exact(self):
        manifest = self.scenario.manifest
        expected_measured_sessions = (
            EXPECTED_SESSIONS_PER_EPOCH * len(MEASUREMENT_EPOCHS))
        expected_measured_requests = (
            EXPECTED_CALLS_PER_EPOCH * len(MEASUREMENT_EPOCHS))

        self.assertEqual(
            len(manifest.measurement_session_ids),
            expected_measured_sessions,
        )
        self.assertEqual(
            manifest.measurement_request_count,
            expected_measured_requests,
        )
        plan = self.scenario.build_offered_plan(seed=101)
        scheduled = plan.at_rate(RECOMMENDED_PILOT_RATES[1])
        self.assertEqual(
            len(scheduled),
            EXPECTED_SESSIONS_PER_EPOCH * manifest.epoch_count,
        )
        self.assertEqual(
            sum(len(item.session.calls) for item in scheduled),
            EXPECTED_CALLS_PER_EPOCH * manifest.epoch_count,
        )
        self.assertEqual(
            tuple(sorted(item.arrival_time_ns for item in scheduled)),
            tuple(item.arrival_time_ns for item in scheduled),
        )

    def test_seed_changes_order_and_draws_but_not_measurement_roster(self):
        left = self.scenario.build_offered_plan(seed=101)
        right = self.scenario.build_offered_plan(seed=211)

        self.assertNotEqual(
            left.offered_session_ids_sha256,
            right.offered_session_ids_sha256,
        )
        self.assertNotEqual(left.unit_draws_sha256, right.unit_draws_sha256)
        self.assertEqual(
            len(self.scenario.manifest.measurement_session_ids),
            len(set(self.scenario.manifest.measurement_session_ids)),
        )


if __name__ == "__main__":
    unittest.main()

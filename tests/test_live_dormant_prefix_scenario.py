from copy import deepcopy
import json
import os
from pathlib import Path
import unittest

from serving.core.live_dormant_prefix_scenario import (
    BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES,
    EXPECTED_CALLS_PER_EPOCH,
    EXPECTED_FIRST_PREFILL_SERVICE_NS_PER_EPOCH,
    EXPECTED_RECORDED_GAP_NS,
    EXPECTED_RETAINED_MAX_SEQUENCE_TOKENS,
    EXPECTED_RESUME_PREFILL_SERVICE_NS_PER_EPOCH,
    EXPECTED_SESSIONS_PER_EPOCH,
    EXPECTED_TRANSFORMED_COHORT_SHA256,
    HBF_TP8_USABLE_LOGICAL_KV_BYTES,
    SOURCE_INDEX,
    _validate_source_row,
    build_live_dormant_prefix_scenario,
    build_pressure,
    build_protocol_smoke,
    build_smoke,
)


TRACE = Path(
    os.environ.get("LLMSIM_DATA", str(Path.home() / "llmsim-data"))
) / "tracelab-schema3-sps0.2-final.jsonl"


def _source_row():
    with TRACE.open("r", encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index == SOURCE_INDEX:
                return json.loads(line)
    raise AssertionError("pinned source row is unavailable")


@unittest.skipUnless(TRACE.is_file(), "pinned TraceLab release is unavailable")
class LiveDormantPrefixScenarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = build_live_dormant_prefix_scenario(
            TRACE,
            warmup_epoch_count=1,
            measurement_epoch_count=2,
            guard_epoch_count=1,
        )

    def test_source_transform_and_synthetic_semantics_are_pinned(self):
        manifest = self.scenario.manifest
        audit = manifest.source_audit
        self.assertEqual(
            manifest.transformed_cohort_sha256,
            EXPECTED_TRANSFORMED_COHORT_SHA256,
        )
        self.assertEqual(audit.retained_call_indices, (0, 1))
        self.assertEqual(audit.discarded_call_indices, (2, 3, 4))
        self.assertEqual(
            audit.recorded_successor_gap_ns, EXPECTED_RECORDED_GAP_NS)
        self.assertIn("truncated", audit.truncation_semantics)
        self.assertIn("replicated", audit.replication_semantics)
        self.assertIn("explicitly synthetic", manifest.workload_semantics)
        self.assertIn("neither an empirical", manifest.workload_semantics)
        self.assertIn("complete-session", manifest.workload_semantics)
        self.assertEqual(
            manifest.retained_max_sequence_tokens,
            EXPECTED_RETAINED_MAX_SEQUENCE_TOKENS,
        )
        self.assertIn("capacity-onset control", manifest.workload_semantics)

    def test_epoch_counts_gap_and_prefill_balance_are_exact(self):
        manifest = self.scenario.manifest
        self.assertEqual(manifest.sessions_per_epoch, EXPECTED_SESSIONS_PER_EPOCH)
        self.assertEqual(manifest.calls_per_epoch, EXPECTED_CALLS_PER_EPOCH)
        self.assertEqual(manifest.output_tokens_per_epoch, 2_685)
        for epoch in self.scenario.epoch_sessions:
            complete = [session for session in epoch if len(session.calls) == 2]
            first_only = [
                session for session in epoch if len(session.calls) == 1]
            self.assertEqual((len(complete), len(first_only)), (5, 3))
            self.assertTrue(all(
                session.calls[0].tool_duration_ns
                == EXPECTED_RECORDED_GAP_NS
                for session in complete
            ))
        service = manifest.prefill_service
        self.assertEqual(
            service.first_prefill_service_ns_per_epoch,
            EXPECTED_FIRST_PREFILL_SERVICE_NS_PER_EPOCH,
        )
        self.assertEqual(
            service.resume_prefill_service_ns_per_epoch,
            EXPECTED_RESUME_PREFILL_SERVICE_NS_PER_EPOCH,
        )
        self.assertAlmostEqual(
            service.resume_to_first_service_ratio, 0.983919827261605)
        self.assertEqual(service.declared_resume_cached_tokens, 120_336)
        self.assertEqual(service.operational_resume_hit_tokens, 120_335)

    def test_runner_protocol_and_measurement_roster_are_configurable(self):
        manifest = self.scenario.manifest
        self.assertEqual(manifest.epoch_profile, "custom")
        self.assertEqual(manifest.epoch_count, 4)
        self.assertEqual(manifest.measurement_epochs, (1, 2))
        self.assertEqual(
            len(manifest.measurement_session_ids),
            EXPECTED_SESSIONS_PER_EPOCH * 2,
        )
        self.assertEqual(
            manifest.measurement_request_count,
            EXPECTED_CALLS_PER_EPOCH * 2,
        )
        left = self.scenario.build_offered_plan(seed=101)
        repeated = self.scenario.build_offered_plan(seed=101)
        right = self.scenario.build_offered_plan(seed=211)
        self.assertEqual(left.unit_draws_sha256, repeated.unit_draws_sha256)
        self.assertNotEqual(left.unit_draws_sha256, right.unit_draws_sha256)
        with self.assertRaisesRegex(ValueError, "audited maximum"):
            left.at_rate(0.020_001)

    def test_logical_kv_byte_gap_and_capacities_are_audited(self):
        pressure = self.scenario.manifest.kv_pressure
        estimates = {
            item.offered_session_rate_per_second:
                item.analytical_recorded_gap_live_kv_bytes_floor
            for item in pressure.analytical_steady_state_estimates
        }
        self.assertEqual(estimates[0.006], 625_526_160_673)
        self.assertEqual(estimates[0.012], 1_251_052_321_347)
        self.assertEqual(estimates[0.02], 2_085_087_202_246)
        self.assertLess(
            estimates[0.012],
            BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES,
        )
        self.assertGreater(
            estimates[0.02],
            BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES,
        )
        self.assertLess(estimates[0.02], HBF_TP8_USABLE_LOGICAL_KV_BYTES)
        self.assertIn(
            "conventional TP8 charges two physical KV copies",
            pressure.capacity_semantics,
        )
        self.assertIn("runtime reports", pressure.realized_runtime_semantics)
        self.assertIn(
            "capacity-onset control", pressure.realized_runtime_semantics)
        self.assertEqual(
            pressure.hbf_config_path,
            "configs/wakekv_hbf/full_model_8card_server.json",
        )
        self.assertEqual(len(pressure.hbf_config_sha256), 64)

    def test_pressure_and_smoke_factories_have_explicit_roles(self):
        pressure = build_pressure(TRACE)
        coverage = pressure.manifest.schedule_coverage
        witness = pressure.manifest.kv_pressure.finite_pressure_witness
        self.assertEqual(pressure.manifest.epoch_profile, "pressure")
        self.assertEqual(pressure.manifest.epoch_count, 88)
        self.assertEqual(coverage.minimum_mean_epochs_for_one_gap, 36)
        self.assertEqual(
            coverage.minimum_warmup_arrival_span_ns,
            15_152_009_401_306,
        )
        self.assertEqual(
            coverage.minimum_guard_arrival_span_ns,
            15_356_399_188_640,
        )
        self.assertTrue(coverage.warmup_covers_gap_for_all_audited_seeds)
        self.assertTrue(coverage.guard_covers_gap_for_all_audited_seeds)
        self.assertEqual(
            witness.zero_service_peak_recorded_gap_logical_kv_bytes,
            2_176_629_866_496,
        )
        self.assertTrue(witness.exceeds_baseline_capacity)
        self.assertTrue(witness.fits_smallest_hbf_capacity)

        smoke = build_smoke(TRACE)
        self.assertEqual(smoke.manifest.epoch_profile, "smoke")
        self.assertEqual(smoke.manifest.epoch_count, 3)
        self.assertFalse(
            smoke.manifest.schedule_coverage
            .warmup_covers_gap_for_all_audited_seeds
        )
        self.assertFalse(
            smoke.manifest.kv_pressure.finite_pressure_witness
            .exceeds_baseline_capacity
        )

        protocol_smoke = build_protocol_smoke(TRACE)
        self.assertEqual(
            protocol_smoke.manifest.epoch_profile, "protocol_smoke")
        self.assertEqual(protocol_smoke.manifest.epoch_count, 1)
        self.assertEqual(
            protocol_smoke.manifest.measurement_epochs, (0,))
        self.assertEqual(
            len(protocol_smoke.manifest.measurement_session_ids),
            EXPECTED_SESSIONS_PER_EPOCH,
        )
        self.assertEqual(
            protocol_smoke.manifest.measurement_request_count,
            EXPECTED_CALLS_PER_EPOCH,
        )

    def test_source_gap_and_call_count_fail_closed(self):
        row = _source_row()
        self.assertEqual(
            _validate_source_row(row).recorded_successor_gap_ns,
            EXPECTED_RECORDED_GAP_NS,
        )
        wrong_count = deepcopy(row)
        wrong_count["sub_requests"].pop()
        with self.assertRaisesRegex(ValueError, "source call count changed"):
            _validate_source_row(wrong_count)
        wrong_gap = deepcopy(row)
        wrong_gap["sub_requests"][0]["tool_duration_ns"] += 1
        with self.assertRaisesRegex(ValueError, "recorded gap changed"):
            _validate_source_row(wrong_gap)

    def test_epoch_count_validation_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "measurement_epoch_count"):
            build_live_dormant_prefix_scenario(
                TRACE,
                warmup_epoch_count=1,
                measurement_epoch_count=0,
                guard_epoch_count=1,
            )


if __name__ == "__main__":
    unittest.main()

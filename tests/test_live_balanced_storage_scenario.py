from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from serving.core.live_balanced_storage_scenario import (
    BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES,
    EXPECTED_CALLS_PER_EPOCH,
    EXPECTED_COMPLETE_COHORT_SHA256,
    EXPECTED_FIRST_PREFILL_SERVICE_NS,
    EXPECTED_MAX_RECORDED_GAP_NS,
    EXPECTED_PREFILL_SERVICE_DIFFERENCE_NS,
    EXPECTED_RECORDED_GAP_LOGICAL_KV_BYTE_NS_PER_EPOCH,
    EXPECTED_RESUME_PREFILL_SERVICE_NS,
    EXPECTED_SESSIONS_PER_EPOCH,
    EXPECTED_SOURCE_IDENTITY_SHA256,
    HBF_TP8_USABLE_LOGICAL_KV_BYTES,
    MAXIMUM_AUDITED_RATE,
    PRESSURE_AUDIT_SEEDS,
    RECOMMENDED_RATES,
    RECOMMENDED_SEEDS,
    SELECTED_SOURCE_INDICES,
    _load_templates,
    _validated_baseline_usable_bytes_per_node,
    _validate_templates,
    build_live_balanced_storage_scenario,
    build_protocol_smoke,
    build_publication,
)


TRACE = Path(
    os.environ.get("LLMSIM_DATA", str(Path.home() / "llmsim-data"))
) / "tracelab-schema3-sps0.2-final.jsonl"


def _call_content(call):
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
class LiveBalancedStorageScenarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates, cls.template_audit = _load_templates(TRACE)
        cls.custom = build_live_balanced_storage_scenario(
            TRACE,
            warmup_epoch_count=1,
            measurement_epoch_count=2,
            guard_epoch_count=1,
        )
        cls.publication = build_publication(TRACE)

    def test_complete_original_cohort_is_content_addressed(self):
        manifest = self.custom.manifest
        audit = manifest.cohort_audit

        self.assertEqual(
            manifest.selected_source_indices, SELECTED_SOURCE_INDICES)
        self.assertEqual(
            audit.complete_cohort_sha256,
            EXPECTED_COMPLETE_COHORT_SHA256,
        )
        self.assertEqual(
            audit.source_identity_sha256,
            EXPECTED_SOURCE_IDENTITY_SHA256,
        )
        self.assertEqual(
            tuple(len(session.calls) for session in self.templates),
            (8, 1, 1, 1, 1, 1, 1),
        )
        self.assertEqual(
            tuple(session.output_tokens for session in self.templates),
            (34, 1, 1, 1, 1, 1, 1),
        )
        self.assertIn(
            "original complete TraceLab sessions",
            manifest.workload_semantics,
        )
        self.assertIn("no context scaling", audit.transform_semantics)
        self.assertIn("no", audit.transform_semantics)

    def test_every_epoch_preserves_internal_call_content(self):
        expected = {
            source.source_index: tuple(
                _call_content(call) for call in source.calls)
            for source in self.templates
        }
        for epoch in self.custom.epoch_sessions:
            self.assertEqual(len(epoch), EXPECTED_SESSIONS_PER_EPOCH)
            for session, source_index in zip(
                    epoch, SELECTED_SOURCE_INDICES):
                self.assertEqual(
                    tuple(_call_content(call) for call in session.calls),
                    expected[source_index],
                )

    def test_first_resume_counts_and_prefill_service_are_exactly_balanced(self):
        manifest = self.custom.manifest
        service = manifest.prefill_service

        self.assertEqual(manifest.sessions_per_epoch, 7)
        self.assertEqual(manifest.calls_per_epoch, 14)
        self.assertEqual(manifest.first_calls_per_epoch, 7)
        self.assertEqual(manifest.resume_calls_per_epoch, 7)
        self.assertEqual(manifest.output_tokens_per_epoch, 40)
        self.assertEqual(
            service.first_prefill_service_ns_per_epoch,
            EXPECTED_FIRST_PREFILL_SERVICE_NS,
        )
        self.assertEqual(
            service.resume_prefill_service_ns_per_epoch,
            EXPECTED_RESUME_PREFILL_SERVICE_NS,
        )
        self.assertEqual(
            service.absolute_difference_ns,
            EXPECTED_PREFILL_SERVICE_DIFFERENCE_NS,
        )
        self.assertAlmostEqual(
            service.resume_to_first_service_ratio,
            1.0000000029127045,
        )

    def test_custom_measurement_roster_and_seeded_plan_are_stable(self):
        manifest = self.custom.manifest
        self.assertEqual(manifest.epoch_profile, "custom")
        self.assertEqual(manifest.epoch_count, 4)
        self.assertEqual(manifest.warmup_epochs, (0,))
        self.assertEqual(manifest.measurement_epochs, (1, 2))
        self.assertEqual(manifest.guard_epochs, (3,))
        self.assertEqual(
            len(manifest.measurement_session_ids), 14)
        self.assertEqual(
            manifest.measurement_request_count,
            EXPECTED_CALLS_PER_EPOCH * 2,
        )
        self.assertEqual(
            manifest.measurement_first_call_count,
            manifest.measurement_resume_call_count,
        )
        self.assertEqual(
            manifest.epoch_mapping_sha256,
            "bca1d89abeee7062e37d78d6697bde13c3f0695fa9079e0826c0b887507a06ca",
        )

        left = self.custom.build_offered_plan(seed=101)
        repeated = self.custom.build_offered_plan(seed=101)
        right = self.custom.build_offered_plan(seed=102)
        self.assertEqual(left.unit_draws_sha256, repeated.unit_draws_sha256)
        self.assertEqual(
            left.offered_session_ids_sha256,
            repeated.offered_session_ids_sha256,
        )
        self.assertNotEqual(left.unit_draws_sha256, right.unit_draws_sha256)
        self.assertNotEqual(
            left.offered_session_ids_sha256,
            right.offered_session_ids_sha256,
        )
        self.assertEqual(
            len(left.at_rate(MAXIMUM_AUDITED_RATE)), 28)
        with self.assertRaisesRegex(ValueError, "audited maximum"):
            left.at_rate(MAXIMUM_AUDITED_RATE + 0.001)

    def test_storage_knee_and_hbf_capacity_are_explicit(self):
        pressure = self.custom.manifest.kv_pressure
        estimates = {
            item.offered_session_rate_per_second:
                item.analytical_recorded_gap_live_kv_bytes_floor
            for item in pressure.analytical_steady_state_estimates
        }

        self.assertEqual(
            pressure.recorded_gap_logical_kv_byte_ns_per_epoch,
            EXPECTED_RECORDED_GAP_LOGICAL_KV_BYTE_NS_PER_EPOCH,
        )
        self.assertEqual(pressure.finite_schedule.recorded_max_gap_ns,
                         EXPECTED_MAX_RECORDED_GAP_NS)
        self.assertEqual(
            pressure.baseline_usable_bytes_per_node,
            691_286_941_696,
        )
        self.assertEqual(
            pressure.baseline_cluster_config_path,
            "configs/cluster/dual_node_qwen3_1m_pd_p4d4_h100.json",
        )
        self.assertEqual(
            pressure.baseline_cluster_config_sha256,
            "2c17e2a94d94e3fc635dca779b82d16013da0f42368c42b3d09a88441eea5d3d",
        )
        self.assertAlmostEqual(
            pressure.analytical_storage_knee_sessions_per_second,
            0.9790280741491629,
        )
        self.assertEqual(estimates[0.25], 353_047_558_057)
        self.assertEqual(estimates[0.80], 1_129_752_185_783)
        self.assertEqual(estimates[1.00], 1_412_190_232_229)
        self.assertEqual(estimates[1.20], 1_694_628_278_675)
        self.assertLess(
            estimates[0.80],
            BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES,
        )
        self.assertGreater(
            estimates[1.00],
            BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES,
        )
        self.assertLess(estimates[1.20], HBF_TP8_USABLE_LOGICAL_KV_BYTES)
        self.assertIn(
            "conventional TP8", pressure.capacity_semantics)
        self.assertIn(
            "TP4 has two weight replicas", pressure.capacity_semantics)

    def test_baseline_capacity_validation_fails_closed_on_config_drift(self):
        from serving.live_astra_comparison_sweep import DUAL_CLUSTER

        config = (
            Path(__file__).resolve().parents[1]
            / "configs/cluster/dual_node_qwen3_1m_pd_p4d4_h100.json"
        )
        self.assertEqual(
            self.custom.manifest.kv_pressure
            .baseline_cluster_config_path,
            DUAL_CLUSTER.as_posix(),
        )
        derived, digest = _validated_baseline_usable_bytes_per_node(config)
        self.assertEqual(derived, 691_286_941_696)
        self.assertEqual(
            digest,
            self.custom.manifest.kv_pressure
            .baseline_cluster_config_sha256,
        )

        changed = json.loads(config.read_text(encoding="utf-8"))
        changed["nodes"][0]["cpu_mem"]["mem_size"] += 1
        with tempfile.TemporaryDirectory() as directory:
            changed_path = Path(directory) / "cluster.json"
            changed_path.write_text(
                json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(
                    ValueError, "node capacities differ"):
                _validated_baseline_usable_bytes_per_node(changed_path)

    def test_publication_profile_has_robust_per_node_pressure(self):
        manifest = self.publication.manifest
        finite = manifest.kv_pressure.finite_schedule

        self.assertEqual(manifest.epoch_profile, "publication")
        self.assertEqual(manifest.epoch_count, 776)
        self.assertEqual(
            len(manifest.measurement_session_ids), 16 * 7)
        self.assertEqual(
            manifest.measurement_request_count, 16 * 14)
        self.assertEqual(
            manifest.epoch_mapping_sha256,
            "6520eb386441d06543d8c102e5964c1b3938f9c8b75e7176ad03abad2bc8f107",
        )
        self.assertEqual(
            manifest.recommended_rates, RECOMMENDED_RATES)
        self.assertEqual(
            manifest.recommended_seeds, RECOMMENDED_SEEDS)
        self.assertEqual(finite.audited_seeds, PRESSURE_AUDIT_SEEDS)
        self.assertEqual(
            finite.minimum_measurement_node_pressure_return_count, 112)
        self.assertEqual(
            finite.minimum_measurement_long_gap_node_pressure_return_count,
            16,
        )
        self.assertEqual(
            finite.minimum_aggregate_peak_bytes,
            1_672_335_065_088,
        )
        self.assertEqual(
            finite.maximum_aggregate_peak_bytes,
            1_803_886_264_320,
        )
        self.assertFalse(
            finite.warmup_span_covers_max_gap_for_all_seeds)
        self.assertFalse(
            finite.guard_span_covers_max_gap_for_all_seeds)
        self.assertEqual(
            manifest.epoch_count * manifest.output_tokens_per_epoch,
            31_040,
        )

        seed101 = finite.witnesses[0]
        self.assertEqual(seed101.seed, 101)
        self.assertEqual(
            seed101.per_node_peak_recorded_gap_logical_kv_bytes,
            (895_758_630_912, 895_758_630_912),
        )
        self.assertEqual(
            seed101.aggregate_peak_recorded_gap_logical_kv_bytes,
            1_748_719_632_384,
        )
        self.assertTrue(seed101.exceeds_baseline_capacity)
        self.assertTrue(seed101.fits_smallest_hbf_capacity)
        self.assertIn(
            "sticky round-robin", seed101.node_assignment_semantics)

    def test_rate_grid_brackets_finite_pressure_onset(self):
        low = self.publication.audit_zero_service_pressure(
            seed=101, sessions_per_second=0.80)
        onset = self.publication.audit_zero_service_pressure(
            seed=101, sessions_per_second=1.00)
        high = self.publication.audit_zero_service_pressure(
            seed=101, sessions_per_second=1.20)

        self.assertEqual(
            low.aggregate_peak_recorded_gap_logical_kv_bytes,
            1_184_322_551_808,
        )
        self.assertEqual(low.measurement_node_pressure_return_count, 0)
        self.assertEqual(
            onset.aggregate_peak_recorded_gap_logical_kv_bytes,
            1_464_399_298_560,
        )
        self.assertEqual(
            onset.measurement_node_pressure_return_count, 76)
        self.assertEqual(
            onset.measurement_long_gap_node_pressure_return_count, 7)
        self.assertEqual(
            high.measurement_node_pressure_return_count, 112)
        self.assertEqual(
            high.measurement_long_gap_node_pressure_return_count, 16)

    def test_protocol_smoke_is_one_complete_epoch(self):
        smoke = build_protocol_smoke(TRACE)
        self.assertEqual(smoke.manifest.epoch_profile, "protocol_smoke")
        self.assertEqual(smoke.manifest.epoch_count, 1)
        self.assertEqual(smoke.manifest.warmup_epochs, ())
        self.assertEqual(smoke.manifest.measurement_epochs, (0,))
        self.assertEqual(smoke.manifest.guard_epochs, ())
        self.assertEqual(
            smoke.manifest.measurement_request_count,
            EXPECTED_CALLS_PER_EPOCH,
        )
        self.assertEqual(
            smoke.manifest.epoch_mapping_sha256,
            "364b12ddbd1e22741767337a90b13ee2e1d9691aa013d9d324ab324e9c952316",
        )
        self.assertFalse(
            smoke.manifest.kv_pressure.finite_schedule.witnesses[0]
            .exceeds_baseline_capacity
        )

    def test_source_content_fingerprint_fails_closed(self):
        source = self.templates[0]
        first = source.calls[0]
        changed_first = replace(
            first,
            input_tokens=first.input_tokens + 1,
            fresh_input_tokens=first.fresh_input_tokens + 1,
        )
        changed_source = replace(
            source,
            calls=(changed_first,) + source.calls[1:],
        )
        changed = (changed_source,) + self.templates[1:]
        with self.assertRaisesRegex(
                ValueError, "complete TraceLab cohort changed"):
            _validate_templates(changed)

    def test_epoch_count_validation_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "measurement_epoch_count"):
            build_live_balanced_storage_scenario(
                TRACE,
                warmup_epoch_count=1,
                measurement_epoch_count=0,
                guard_epoch_count=1,
            )
        with self.assertRaisesRegex(ValueError, "warmup_epoch_count"):
            build_live_balanced_storage_scenario(
                TRACE,
                warmup_epoch_count=True,
                measurement_epoch_count=1,
                guard_epoch_count=1,
            )


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from serving.agentic_kv_capacity_replay import build_parser
from serving.core.agentic_kv_capacity_replay import (
    CapacityReplayConfig,
    DGX_H100_CM6_IDEAL_READ_GBPS,
    DGX_H100_CM6_IDEAL_WRITE_GBPS,
    DGX_H100_NVLINK_ONE_WAY_GBPS_PER_GPU,
    _Active,
    _CapacityReplay,
    _Entry,
    estimate_model_weight_bytes_per_rank,
    infinite_hbm_oracle_capacity,
    load_capacity_replay_workload,
    replay_capacity_aware,
    replay_capacity_aware_with_oracle,
)
from serving.core.agentic_kv_roofline import (
    DEFAULT_HARDWARE_SPECS,
    ModelShape,
    kv_layout,
    override_transfer_defaults,
)


def _tiny_model() -> ModelShape:
    return ModelShape(
        name="tiny",
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
    )


def _session(session_id, arrival_ns, tool_ns, tokens=100):
    return {
        "session_id": session_id,
        "arrival_time_ns": arrival_ns,
        "sub_requests": [
            {
                "input_toks": tokens,
                "output_toks": 1,
                "tool_duration_ns": tool_ns,
            },
            {
                "input_toks": tokens + 8,
                "output_toks": 1,
                "tool_duration_ns": 0,
                "prefix_reuse_toks": tokens,
                "prefix_reuse_source": "reported",
            },
        ],
    }


class _FixedPromptComputeModel:
    def recompute_seconds(self, tokens):
        return tokens * 1e-6

    def cached_prefill_seconds(self, total_tokens, cached_tokens):
        return (total_tokens - cached_tokens) * 1e-6

    def metadata(self):
        return {
            "model_kind": "test_kernel_calibrated",
            "calibrated_from_measurements": True,
            "description": "Deterministic test prompt model.",
        }


class AgenticKvCapacityReplayTest(unittest.TestCase):
    def _write(self, sessions):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "trace.jsonl"
        path.write_text(
            "".join(json.dumps(session) + "\n" for session in sessions),
            encoding="utf-8",
        )
        return directory, path

    def _finish_manual_lower_restore(self, replay, session_id):
        matches = [
            event for event in replay.events
            if event[3] == "restore_complete"
            and event[4][0] == session_id
        ]
        self.assertEqual(len(matches), 1)
        finish_ns, _, _, _, payload = matches[0]
        replay._finish_restore(
            finish_ns,
            str(payload[0]),
            int(payload[1]),
            str(payload[2]),
            int(payload[3]),
            int(payload[4]),
            int(payload[5]),
            int(payload[6]),
        )
        return finish_ns

    def test_cli_defaults_use_dgx_h100_cm6_eight_drive_upper_bound(self):
        args = build_parser().parse_args([
            "--workload", "trace.jsonl",
            "--model", "meta-llama/Llama-3.1-70B",
            "--hardware", "H100",
            "--output", "report.json",
        ])
        self.assertEqual(args.cpu_rank_gbps, 50.0)
        self.assertEqual(args.cpu_aggregate_gbps, 400.0)
        self.assertAlmostEqual(
            args.ssd_read_gbps, DGX_H100_CM6_IDEAL_READ_GBPS
        )
        self.assertAlmostEqual(
            args.ssd_write_gbps, DGX_H100_CM6_IDEAL_WRITE_GBPS
        )
        self.assertIsNone(
            args.prefill_hbm_static_reserve_gib_per_rank
        )
        self.assertIsNone(
            args.decode_hbm_static_reserve_gib_per_rank
        )
        self.assertEqual(args.prompt_compute_scale, 1.0)
        self.assertEqual(
            args.pd_link_gbps_per_rank,
            DGX_H100_NVLINK_ONE_WAY_GBPS_PER_GPU,
        )

    def test_pd_role_reserves_produce_independent_hbm_budgets(self):
        model = _tiny_model()
        config = CapacityReplayConfig(
            hbm_capacity_bytes_per_rank=1 << 30,
            prefill_hbm_static_reserve_bytes_per_rank=1000,
            decode_hbm_static_reserve_bytes_per_rank=2000,
            pd_disaggregated=True,
        )
        directory, path = self._write([_session("role-reserve", 0, 1)])
        try:
            replay = _CapacityReplay(
                load_capacity_replay_workload(path),
                model,
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                config,
            )
        finally:
            directory.cleanup()

        self.assertEqual(
            replay.prefill_hbm_kv_budget - replay.decode_hbm_kv_budget,
            1000,
        )
        self.assertEqual(
            config.effective_prefill_hbm_static_reserve_bytes_per_rank,
            1000,
        )
        self.assertEqual(
            config.effective_decode_hbm_static_reserve_bytes_per_rank,
            2000,
        )

    def test_prompt_compute_scale_is_explicit_and_multiplicative(self):
        directory, path = self._write([_session("compute-scale", 0, 1)])
        try:
            workload = load_capacity_replay_workload(path)
            identity = _CapacityReplay(
                workload,
                _tiny_model(),
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=1 << 30,
                ),
            )
            fast = _CapacityReplay(
                workload,
                _tiny_model(),
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=1 << 30,
                    prompt_compute_scale=1 / 3,
                    prompt_compute_scale_provenance=(
                        "Adversarial three-times-faster recompute endpoint."
                    ),
                ),
            )
        finally:
            directory.cleanup()

        self.assertAlmostEqual(
            fast._roofline_seconds(100),
            identity._roofline_seconds(100) / 3,
        )
        self.assertAlmostEqual(
            fast._cached_prefill_seconds(108, 100),
            identity._cached_prefill_seconds(108, 100) / 3,
        )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            CapacityReplayConfig(
                hbm_capacity_bytes_per_rank=1 << 30,
                prompt_compute_scale=0,
            ).validate()
        with self.assertRaisesRegex(ValueError, "requires provenance"):
            CapacityReplayConfig(
                hbm_capacity_bytes_per_rank=1 << 30,
                prompt_compute_scale=0.5,
            ).validate()
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            CapacityReplayConfig(
                hbm_capacity_bytes_per_rank=1 << 30,
                prompt_compute_scale_provenance=" ",
            ).validate()

    def test_external_prompt_model_is_used_by_finite_and_oracle(self):
        directory, path = self._write([
            _session("external-compute", 0, 1_000_000, tokens=100)
        ])
        try:
            report = replay_capacity_aware_with_oracle(
                load_capacity_replay_workload(path),
                _tiny_model(),
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=1 << 30,
                    policy="hbm_lru_recompute",
                    demotion_mode="capacity-only",
                ),
                _FixedPromptComputeModel(),
            )
        finally:
            directory.cleanup()

        scope = report["execution_scope"]
        self.assertEqual(
            scope["prompt_compute_model_kind"],
            "test_kernel_calibrated",
        )
        self.assertEqual(
            scope["prompt_compute_calibration"]["description"],
            "Deterministic test prompt model.",
        )
        comparison = report["infinite_hbm_oracle_comparison"]
        self.assertTrue(comparison["same_prompt_compute_model"])
        self.assertEqual(
            comparison["prompt_compute_model_kind"],
            "test_kernel_calibrated",
        )

    def test_role_reserves_are_rejected_for_single_hbm_pool(self):
        with self.assertRaisesRegex(ValueError, "require P/D"):
            CapacityReplayConfig(
                hbm_capacity_bytes_per_rank=1 << 30,
                pd_disaggregated=False,
                prefill_hbm_static_reserve_bytes_per_rank=1000,
            ).validate()

    def test_report_labels_ssd_default_as_manufacturer_upper_bound(self):
        directory, path = self._write([_session("ssd-spec", 0, 1_000_000)])
        try:
            hardware = override_transfer_defaults(
                DEFAULT_HARDWARE_SPECS["H100"],
                cpu_rank_gbps=50.0,
                cpu_aggregate_gbps=400.0,
                ssd_read_gbps=DGX_H100_CM6_IDEAL_READ_GBPS,
                ssd_write_gbps=DGX_H100_CM6_IDEAL_WRITE_GBPS,
            )
            report = replay_capacity_aware(
                load_capacity_replay_workload(path),
                _tiny_model(),
                hardware,
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=1 << 30,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    demotion_mode="capacity-only",
                ),
            )
        finally:
            directory.cleanup()

        sensitivity = report["hardware_sensitivity"]
        self.assertEqual(report["schema_version"], 11)
        self.assertEqual(sensitivity["ssd_reference_model"], "KCM6DRUL3T84")
        self.assertEqual(sensitivity["ssd_reference_interface"], "PCIe 4.0 x4")
        self.assertAlmostEqual(
            sensitivity["ssd_reference_ideal_read_gbps_aggregate"], 55.2
        )
        self.assertAlmostEqual(
            sensitivity["ssd_reference_ideal_write_gbps_aggregate"], 33.6
        )
        self.assertTrue(
            sensitivity["ssd_bandwidths_match_manufacturer_upper_bound"]
        )
        self.assertIn("upper bound", sensitivity["ssd_contract_provenance"])

    def test_loader_retains_arrivals_and_eligible_denominator(self):
        sessions = [
            _session("a", 0, 1_000_000),
            _session("b", 10, 1_000_000, tokens=200),
        ]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(
                path, max_context_tokens=128
            )
        finally:
            directory.cleanup()
        self.assertEqual(workload.sessions[1].arrival_time_ns, 10)
        self.assertEqual(workload.selected_positive_transitions, 1)
        self.assertEqual(workload.selected_reuse_eligible_transitions, 1)
        self.assertEqual(workload.transitions_excluded_context, 1)
        self.assertEqual(
            workload.metadata_dict()["reuse_source_counts"],
            {"explicit_reported": 1},
        )

    def test_context_limit_includes_prompt_and_requested_output(self):
        session = _session("total-context", 0, 1_000_000, tokens=120)
        session["sub_requests"][0]["output_toks"] = 9
        session["sub_requests"][1]["input_toks"] = 128
        session["sub_requests"][1]["output_toks"] = 1
        directory, path = self._write([session])
        try:
            workload = load_capacity_replay_workload(
                path, max_context_tokens=128
            )
        finally:
            directory.cleanup()

        first, second = workload.sessions[0].calls
        self.assertEqual(first.total_sequence_tokens, 129)
        self.assertFalse(first.context_eligible)
        self.assertFalse(first.cache_eligible)
        self.assertEqual(second.total_sequence_tokens, 129)
        self.assertFalse(second.context_eligible)
        self.assertEqual(workload.selected_positive_transitions, 0)
        self.assertEqual(workload.transitions_excluded_context, 1)
        self.assertIn(
            "input_toks + output_toks",
            workload.metadata_dict()["context_eligibility_semantics"],
        )

    def test_return_class_comes_from_preceding_call(self):
        session = _session("return-owner", 0, 1_000_000)
        session["sub_requests"][0].update({
            "inter_turn_gap_type": "human",
            "tool_wait_source": "request_ready_boundary",
        })
        session["sub_requests"][1]["inter_turn_gap_type"] = "tool"
        directory, path = self._write([session])
        try:
            workload = load_capacity_replay_workload(path)
            report = replay_capacity_aware(
                workload,
                _tiny_model(),
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=1 << 30,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    demotion_mode="capacity-only",
                    pd_disaggregated=True,
                ),
            )
        finally:
            directory.cleanup()
        first, second = workload.sessions[0].calls
        self.assertEqual(first.return_gap_type, "session_start")
        self.assertEqual(second.return_gap_type, "human")
        self.assertEqual(second.return_gap_source, "request_ready_boundary")
        self.assertEqual(second.return_gap_ns, 1_000_000)
        metadata = report["workload"]
        self.assertEqual(
            metadata["return_gap_type_counts"],
            {"human": 1, "session_start": 1},
        )
        human = report["resume"]["by_return_gap_type"]["human"]
        self.assertEqual(human["all_request_count"], 1)
        self.assertEqual(human["reuse_eligible_transition_count"], 1)
        self.assertEqual(human["source_counts"]["decode_hbm"], 1)
        start = report["resume"]["by_return_gap_type"]["session_start"]
        self.assertEqual(
            start["not_reuse_eligible_or_not_selected_count"], 1
        )

    def test_loader_preserves_fresh_prompt_tokens_with_fallback(self):
        session = _session("fresh", 0, 1_000_000)
        session["sub_requests"][0]["raw_newly_append_toks"] = 0
        session["sub_requests"][0]["newly_append_toks"] = 1
        session["sub_requests"][1]["newly_append_toks"] = 7
        session["sub_requests"].append({
            "input_toks": 120,
            "output_toks": 1,
            "tool_duration_ns": 0,
            "prefix_reuse_toks": 108,
        })
        directory, path = self._write([session])
        try:
            workload = load_capacity_replay_workload(path)
        finally:
            directory.cleanup()
        calls = workload.sessions[0].calls
        self.assertEqual(
            [call.fresh_prompt_tokens for call in calls], [100, 8, 12]
        )
        self.assertEqual(
            [call.declared_newly_append_tokens for call in calls],
            [0, 7, None],
        )
        self.assertEqual(
            workload.metadata_dict()["declared_zero_append_calls"], 1
        )

    def test_async_restore_joins_prefill_and_reports_exposed_barrier(self):
        session = _session("join", 0, 1_000_000)
        session["sub_requests"][0]["inter_turn_gap_type"] = "tool"
        session["sub_requests"][1]["newly_append_toks"] = 8
        directory, path = self._write([session])
        try:
            workload = load_capacity_replay_workload(path)
            base = dict(
                hbm_capacity_bytes_per_rank=1 << 30,
                cpu_capacity_bytes=1 << 30,
                ssd_capacity_bytes=1 << 30,
                hbm_ttl_ns=0,
                cpu_ttl_ns=10**12,
                ssd_ttl_ns=10**12,
                enable_transfer_queueing=False,
                cancel_migration_on_resume=True,
                pd_disaggregated=True,
            )
            async_report = replay_capacity_aware(
                workload,
                _tiny_model(),
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    **base,
                    restore_execution_mode="async-decode-join",
                ),
            )
            gated_report = replay_capacity_aware(
                workload,
                _tiny_model(),
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(**base),
            )
        finally:
            directory.cleanup()

        async_timing = async_report["resume"]["restore_timing"]
        gated_timing = gated_report["resume"]["restore_timing"]
        self.assertEqual(async_report["resume"]["source_counts"]["cpu"], 1)
        self.assertGreater(
            async_timing["request_summed_hidden_by_prefill_seconds"], 0
        )
        self.assertGreater(
            async_timing["request_summed_exposed_decode_barrier_seconds"], 0
        )
        self.assertAlmostEqual(
            async_timing["request_summed_raw_elapsed_seconds"],
            async_timing["request_summed_hidden_by_prefill_seconds"]
            + async_timing["request_summed_exposed_decode_barrier_seconds"]
            + async_timing[
                "request_summed_other_concurrent_or_admission_seconds"
            ],
        )
        self.assertEqual(
            gated_timing["request_summed_hidden_by_prefill_seconds"], 0
        )
        self.assertEqual(
            gated_timing[
                "request_summed_exposed_compute_admission_gate_seconds"
            ],
            gated_timing["request_summed_raw_elapsed_seconds"],
        )
        self.assertEqual(
            gated_timing["request_summed_raw_elapsed_seconds"],
            gated_timing["request_summed_exposed_decode_barrier_seconds"],
        )
        self.assertLess(
            async_report["request_makespan_seconds"],
            gated_report["request_makespan_seconds"],
        )
        tool_row = async_timing["by_return_gap_type"]["tool"]
        self.assertEqual(tool_row["event_count"], 1)
        self.assertGreater(
            tool_row["wall_clock_exposed_decode_barrier_union_seconds"], 0
        )

    def test_one_fresh_prompt_token_exposes_full_restore(self):
        session = _session("one-fresh", 0, 1_000_000)
        session["sub_requests"][0]["inter_turn_gap_type"] = "human"
        session["sub_requests"][1]["input_toks"] = 101
        session["sub_requests"][1]["raw_newly_append_toks"] = 0
        session["sub_requests"][1]["newly_append_toks"] = 1
        directory, path = self._write([session])
        try:
            workload = load_capacity_replay_workload(path)
            report = replay_capacity_aware(
                workload,
                _tiny_model(),
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=1 << 30,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    hbm_ttl_ns=0,
                    cpu_ttl_ns=10**12,
                    ssd_ttl_ns=10**12,
                    enable_transfer_queueing=False,
                    cancel_migration_on_resume=True,
                    pd_disaggregated=True,
                ),
            )
        finally:
            directory.cleanup()
        self.assertEqual(
            workload.sessions[0].calls[1].fresh_prompt_tokens, 1
        )
        timing = report["resume"]["restore_timing"]
        self.assertEqual(timing["request_summed_hidden_by_prefill_seconds"], 0)
        self.assertEqual(
            timing["request_summed_raw_elapsed_seconds"],
            timing["request_summed_exposed_decode_barrier_seconds"],
        )
        self.assertEqual(
            timing["by_return_gap_type"]["human"]["event_count"], 1
        )

    def test_restore_hiding_uses_actual_prefill_interval_intersection(self):
        session = _session("intersection", 0, 1_000_000)
        directory, path = self._write([session])
        try:
            workload = load_capacity_replay_workload(path)
            replay = _CapacityReplay(
                workload,
                _tiny_model(),
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=1 << 30,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    restore_execution_mode="async-decode-join",
                ),
            )
            replay._record_restore_join(
                workload.sessions[0].calls[1],
                "cpu",
                0,
                100,
                120,
                150,
            )
        finally:
            directory.cleanup()
        self.assertEqual(replay.raw_restore_elapsed_ns, 150)
        self.assertEqual(replay.restore_hidden_by_prefill_ns, 20)
        self.assertEqual(replay.exposed_restore_barrier_ns, 30)
        self.assertEqual(
            replay.restore_other_concurrent_or_admission_ns, 100
        )
        no_restore_completion_ns = 140
        async_completion_ns = replay._prompt_join_completion_ns(
            150, 100, 120, 40, 20
        )
        self.assertEqual(
            async_completion_ns - no_restore_completion_ns,
            replay.exposed_restore_barrier_ns,
        )

        serial = _CapacityReplay(
            workload,
            _tiny_model(),
            DEFAULT_HARDWARE_SPECS["H100"],
            1,
            2,
            CapacityReplayConfig(
                hbm_capacity_bytes_per_rank=1 << 30,
                cpu_capacity_bytes=1 << 30,
                ssd_capacity_bytes=1 << 30,
                restore_execution_mode="serial-before-prefill",
            ),
        )
        serial._record_restore_join(
            workload.sessions[0].calls[1],
            "cpu",
            0,
            100,
            120,
            150,
        )
        self.assertEqual(serial.restore_hidden_by_prefill_ns, 0)
        self.assertEqual(serial.exposed_restore_barrier_ns, 150)
        self.assertEqual(
            serial.restore_other_concurrent_or_admission_ns, 0
        )
        serial_completion_ns = serial._prompt_join_completion_ns(
            150, 100, 120, 40, 20
        )
        self.assertEqual(
            serial_completion_ns - 40,
            serial.exposed_restore_barrier_ns,
        )

    def test_offered_call_activity_idle_complement_is_not_utilization(self):
        sessions = [
            {
                "session_id": "early",
                "arrival_time_ns": 0,
                "sub_requests": [{
                    "input_toks": 8,
                    "output_toks": 1,
                    "tool_duration_ns": 0,
                }],
            },
            {
                "session_id": "late",
                "arrival_time_ns": 1_000_000_000,
                "sub_requests": [{
                    "input_toks": 8,
                    "output_toks": 1,
                    "tool_duration_ns": 0,
                }],
            },
        ]
        directory, path = self._write(sessions)
        try:
            report = replay_capacity_aware(
                load_capacity_replay_workload(path),
                _tiny_model(),
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=1 << 30,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    demotion_mode="capacity-only",
                ),
            )
        finally:
            directory.cleanup()
        activity = report["offered_load_call_activity"]
        self.assertFalse(activity["is_server_utilization"])
        self.assertGreater(
            activity["wall_clock_with_no_active_call_seconds"], 0.9
        )
        self.assertAlmostEqual(
            activity["wall_clock_with_at_least_one_active_call_seconds"]
            + activity["wall_clock_with_no_active_call_seconds"],
            activity["window_seconds"],
        )

    def test_paired_infinite_hbm_reference_preserves_pd_transfers(self):
        session = _session("oracle", 0, 100_000_000)
        session["sub_requests"][0]["inter_turn_gap_type"] = "tool"
        directory, path = self._write([session])
        try:
            workload = load_capacity_replay_workload(path)
            model = _tiny_model()
            config = CapacityReplayConfig(
                hbm_capacity_bytes_per_rank=1 << 30,
                cpu_capacity_bytes=1 << 30,
                ssd_capacity_bytes=1 << 30,
                hbm_ttl_ns=1,
                cpu_ttl_ns=10_000_000_000,
                ssd_ttl_ns=10_000_000_000,
                enable_transfer_queueing=False,
                cancel_migration_on_resume=True,
                pd_disaggregated=True,
                prefill_hbm_static_reserve_bytes_per_rank=1000,
                decode_hbm_static_reserve_bytes_per_rank=2000,
            )
            capacity = infinite_hbm_oracle_capacity(
                workload, model, 1, 2, config
            )
            report = replay_capacity_aware_with_oracle(
                workload,
                model,
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                config,
            )
        finally:
            directory.cleanup()

        self.assertGreaterEqual(
            capacity["decode_kv_bound_bytes_per_rank"],
            capacity["prefill_kv_bound_bytes_per_rank"],
        )
        self.assertEqual(report["schema_version"], 15)
        self.assertIn(
            "full-prompt prediction minus cached-prefix prediction",
            report["recompute"]["numerator_scope"],
        )
        self.assertEqual(
            capacity["total_hbm_capacity_bytes_per_rank"],
            max(
                capacity["prefill_total_hbm_capacity_bytes_per_rank"],
                capacity["decode_total_hbm_capacity_bytes_per_rank"],
            ),
        )
        self.assertEqual(
            report["capacity"][
                "prefill_hbm_static_reserve_bytes_per_rank"
            ],
            1000,
        )
        self.assertEqual(
            report["capacity"][
                "decode_hbm_static_reserve_bytes_per_rank"
            ],
            2000,
        )
        self.assertTrue(
            report["capacity"][
                "role_specific_hbm_reserve_overrides_present"
            ]
        )
        self.assertTrue(
            report["capacity"][
                "effective_role_hbm_reserves_differ"
            ]
        )
        self.assertNotIn(
            "hbm_kv_budget_bytes_per_rank", report["capacity"]
        )
        comparison = report["infinite_hbm_oracle_comparison"]
        validation = comparison["oracle_validation"]
        self.assertEqual(validation["eligible_source_counts"]["cpu"], 0)
        self.assertEqual(validation["eligible_source_counts"]["ssd"], 0)
        self.assertEqual(
            validation["eligible_source_counts"]["recompute"], 0
        )
        self.assertEqual(
            validation["eligible_source_counts"]["decode_hbm"], 1
        )
        self.assertEqual(validation["capacity_action_count"], 0)
        self.assertGreater(
            comparison["all_calls"][
                "delta_request_summed_ready_to_complete_seconds"
            ],
            0,
        )
        self.assertTrue(
            comparison["closed_loop_delay_conservation_checked"]
        )
        self.assertFalse(comparison["compute_queue_or_batching_modeled"])
        self.assertGreater(
            report["pd_transfer"]["decode_to_prefill_bytes"], 0
        )

    def test_capacity_cascade_produces_ssd_resume(self):
        model = _tiny_model()
        hardware = DEFAULT_HARDWARE_SPECS["H100"]
        layout = kv_layout(model, 1, 2)
        object_rank_bytes = 100 * layout.physical_bytes_per_token_per_rank
        weight_bytes = estimate_model_weight_bytes_per_rank(model, 1, 2, layout)
        sessions = [
            _session("a", 0, 1_000_000_000),
            _session("b", 100_000_000, 2_000_000_000),
            _session("c", 200_000_000, 2_000_000_000),
        ]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            report = replay_capacity_aware(
                workload,
                model,
                hardware,
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=weight_bytes + object_rank_bytes + 1024,
                    cpu_capacity_bytes=object_rank_bytes + 1024,
                    ssd_capacity_bytes=100 * object_rank_bytes,
                    hbm_ttl_ns=10_000_000_000,
                    cpu_ttl_ns=10_000_000_000,
                    ssd_ttl_ns=10_000_000_000,
                    enable_transfer_queueing=True,
                    pd_disaggregated=False,
                ),
            )
        finally:
            directory.cleanup()
        self.assertGreaterEqual(report["resume"]["source_counts"]["ssd"], 1)
        self.assertGreater(
            report["policy"]["actions_by_reason"].get("cpu_capacity", 0), 0
        )
        self.assertTrue(report["policy"]["cpu_cache_enabled"])
        self.assertIsNone(report["policy"]["ssd_direct_semantics"])
        self.assertTrue(report["capacity"]["capacity_invariant_checked"])
        for fraction in report["capacity"]["peak_fraction"].values():
            self.assertLessEqual(fraction, 1.0)

    def test_capacity_only_direct_ssd_has_no_cpu_cache_but_stages_reads(self):
        model = _tiny_model()
        hardware = DEFAULT_HARDWARE_SPECS["H100"]
        layout = kv_layout(model, 1, 2)
        object_rank_bytes = 100 * layout.physical_bytes_per_token_per_rank
        weight_bytes = estimate_model_weight_bytes_per_rank(model, 1, 2, layout)
        sessions = [
            _session("a", 0, 1_000_000_000),
            _session("b", 100_000_000, 2_000_000_000),
            _session("c", 200_000_000, 2_000_000_000),
        ]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            report = replay_capacity_aware(
                workload,
                model,
                hardware,
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=(
                        weight_bytes + object_rank_bytes + 1024
                    ),
                    cpu_capacity_bytes=object_rank_bytes + 1024,
                    ssd_capacity_bytes=100 * object_rank_bytes,
                    policy="hbm_ssd_direct",
                    demotion_mode="capacity-only",
                    pd_disaggregated=False,
                ),
            )
        finally:
            directory.cleanup()
        self.assertGreaterEqual(report["resume"]["source_counts"]["ssd"], 1)
        self.assertEqual(report["resume"]["source_counts"]["cpu"], 0)
        jobs = report["transfer_queue"]["jobs_by_kind"]
        self.assertGreater(jobs.get("hbm_to_ssd_direct", 0), 0)
        self.assertGreater(jobs.get("ssd_to_cpu_stage_for_hbm", 0), 0)
        self.assertGreater(jobs.get("cpu_stage_to_hbm", 0), 0)
        self.assertNotIn("hbm_to_cpu", jobs)
        self.assertNotIn("cpu_to_ssd", jobs)
        self.assertEqual(report["capacity"]["peak_occupancy"]["cpu_bytes"], 0)
        self.assertFalse(report["policy"]["cpu_cache_enabled"])
        self.assertIsNotNone(report["policy"]["ssd_direct_semantics"])
        self.assertFalse(
            report["ssd_io"][
                "transient_cpu_stage_counts_as_cpu_cache_occupancy"
            ]
        )

    def test_capacity_only_hbm_policy_recomputes_without_lower_tier(self):
        model = _tiny_model()
        hardware = DEFAULT_HARDWARE_SPECS["H100"]
        layout = kv_layout(model, 1, 2)
        object_rank_bytes = 100 * layout.physical_bytes_per_token_per_rank
        weight_bytes = estimate_model_weight_bytes_per_rank(model, 1, 2, layout)
        sessions = [
            _session("a", 0, 1_000_000_000),
            _session("b", 100_000_000, 2_000_000_000),
            _session("c", 200_000_000, 2_000_000_000),
        ]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            report = replay_capacity_aware(
                workload,
                model,
                hardware,
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=(
                        weight_bytes + object_rank_bytes + 1024
                    ),
                    cpu_capacity_bytes=object_rank_bytes + 1024,
                    ssd_capacity_bytes=100 * object_rank_bytes,
                    policy="hbm_lru_recompute",
                    demotion_mode="capacity-only",
                    pd_disaggregated=False,
                ),
            )
        finally:
            directory.cleanup()
        self.assertGreaterEqual(
            report["resume"]["source_counts"]["recompute"], 1
        )
        self.assertEqual(report["resume"]["source_counts"]["cpu"], 0)
        self.assertEqual(report["resume"]["source_counts"]["ssd"], 0)
        self.assertEqual(report["transfer_queue"]["jobs"], 0)
        self.assertEqual(report["capacity"]["peak_occupancy"]["cpu_bytes"], 0)
        self.assertEqual(report["capacity"]["peak_occupancy"]["ssd_bytes"], 0)

    def test_ssd_capacity_eviction_causes_recompute(self):
        model = _tiny_model()
        hardware = DEFAULT_HARDWARE_SPECS["H100"]
        layout = kv_layout(model, 1, 2)
        object_rank_bytes = 100 * layout.physical_bytes_per_token_per_rank
        weight_bytes = estimate_model_weight_bytes_per_rank(model, 1, 2, layout)
        sessions = [
            _session("a", 0, 1_000_000_000),
            _session("b", 100_000_000, 2_000_000_000),
            _session("c", 200_000_000, 2_000_000_000),
        ]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            report = replay_capacity_aware(
                workload,
                model,
                hardware,
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=weight_bytes + object_rank_bytes + 1024,
                    cpu_capacity_bytes=object_rank_bytes + 1024,
                    ssd_capacity_bytes=object_rank_bytes // 2,
                    hbm_ttl_ns=10_000_000_000,
                    cpu_ttl_ns=10_000_000_000,
                    ssd_ttl_ns=10_000_000_000,
                    pd_disaggregated=False,
                ),
            )
        finally:
            directory.cleanup()
        recompute = report["recompute"]
        self.assertGreaterEqual(recompute["event_count"], 1)
        self.assertGreater(recompute["tokens"], 0)
        self.assertGreater(
            recompute["analytical_time_fraction_of_executed_prompt_compute"], 0
        )
        self.assertIn(
            "seconds returned by the configured prompt-compute model",
            recompute["time_denominator_scope"],
        )
        self.assertIn(
            "kernel and collective terms",
            recompute["time_denominator_scope"],
        )
        self.assertIn("ssd_object_oversize", recompute["reasons"])

    def test_unbounded_capacity_reaches_all_ttl_sources(self):
        sessions = [
            _session("hbm", 0, 10_000_000),
            _session("cpu", 1, 100_000_000),
            _session("ssd", 2, 300_000_000),
            _session("drop", 3, 700_000_000),
        ]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            report = replay_capacity_aware(
                workload,
                _tiny_model(),
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=1 << 30,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    hbm_ttl_ns=50_000_000,
                    cpu_ttl_ns=100_000_000,
                    ssd_ttl_ns=200_000_000,
                    enable_transfer_queueing=False,
                    cancel_migration_on_resume=True,
                    pd_disaggregated=False,
                ),
            )
        finally:
            directory.cleanup()
        self.assertEqual(
            report["resume"]["source_counts"],
            {"hbm": 1, "cpu": 1, "ssd": 1, "recompute": 1},
        )
        self.assertEqual(report["resume"]["all_request_count"], 8)
        self.assertEqual(
            report["resume"]["cpu_or_ssd_resume_count"], 2
        )
        self.assertAlmostEqual(
            report["resume"]["source_fractions_of_all_requests"]["cpu"],
            1 / 8,
        )
        self.assertAlmostEqual(
            report["resume"]["source_fractions_of_all_requests"]["ssd"],
            1 / 8,
        )
        self.assertAlmostEqual(
            report["resume"]["cpu_or_ssd_resume_fraction_of_all_requests"],
            2 / 8,
        )
        self.assertAlmostEqual(
            sum(report["resume"]["source_fractions_of_reuse_eligible"].values()),
            1.0,
        )

    def test_capacity_only_disables_ttl_actions(self):
        sessions = [_session("capacity-only", 0, 10_000_000_000)]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            report = replay_capacity_aware(
                workload,
                _tiny_model(),
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=1 << 30,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    demotion_mode="capacity-only",
                    hbm_ttl_ns=0,
                    cpu_ttl_ns=0,
                    ssd_ttl_ns=0,
                    pd_disaggregated=False,
                ),
            )
        finally:
            directory.cleanup()
        self.assertEqual(report["policy"]["demotion_mode"], "capacity-only")
        self.assertEqual(report["resume"]["source_counts"]["hbm"], 1)
        self.assertFalse(
            any(reason.endswith("_ttl") for reason in report["policy"]["actions_by_reason"])
        )

    def test_default_waits_for_inflight_demotion_without_ghost_cancel(self):
        sessions = [_session("wait", 0, 1)]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            report = replay_capacity_aware(
                workload,
                _tiny_model(),
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=1 << 30,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    hbm_ttl_ns=0,
                    cpu_ttl_ns=1_000_000_000,
                    ssd_ttl_ns=1_000_000_000,
                    pd_disaggregated=False,
                ),
            )
        finally:
            directory.cleanup()
        self.assertEqual(report["resume"]["source_counts"]["cpu"], 1)
        self.assertGreater(
            report["resume"]["aggregate_inflight_demotion_wait_seconds"], 0
        )
        self.assertNotIn(
            "migration_cancel_on_resume",
            report["policy"]["actions_by_reason"],
        )

    def test_entry_generation_prevents_stale_ttl_from_demoting_replacement(self):
        session = {
            "session_id": "epoch",
            "arrival_time_ns": 0,
            "sub_requests": [
                {"input_toks": 100, "output_toks": 1,
                 "tool_duration_ns": 10_000_000},
                {"input_toks": 110, "output_toks": 1,
                 "tool_duration_ns": 45_000_000,
                 "prefix_reuse_toks": 100},
                {"input_toks": 120, "output_toks": 1,
                 "tool_duration_ns": 0, "prefix_reuse_toks": 110},
            ],
        }
        directory, path = self._write([session])
        try:
            workload = load_capacity_replay_workload(path)
            report = replay_capacity_aware(
                workload, _tiny_model(), DEFAULT_HARDWARE_SPECS["H100"],
                1, 2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=1 << 30,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    hbm_ttl_ns=50_000_000,
                    cpu_ttl_ns=1_000_000_000,
                    ssd_ttl_ns=1_000_000_000,
                    enable_transfer_queueing=False,
                    cancel_migration_on_resume=True,
                    pd_disaggregated=False,
                ),
            )
        finally:
            directory.cleanup()
        self.assertEqual(report["resume"]["source_counts"]["hbm"], 2)

    def test_pd_d_blocked_does_not_delay_prefill_branch(self):
        session = _session("d-blocked", 0, 1_000_000)
        directory, path = self._write([session])
        try:
            workload = load_capacity_replay_workload(path)
            model = _tiny_model()
            layout = kv_layout(model, 1, 2)
            weight = estimate_model_weight_bytes_per_rank(
                model, 1, 2, layout
            )
            object_bytes = (
                112 * layout.physical_bytes_per_token_per_rank
            )
            replay = _CapacityReplay(
                workload,
                model,
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=weight + object_bytes,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    enable_transfer_queueing=False,
                    pd_disaggregated=True,
                    restore_execution_mode="async-decode-join",
                ),
            )
            source = _Entry(
                session_id="d-blocked",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=0,
                generation=1,
            )
            blocker = _Entry(
                session_id="d-capacity-blocker",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=-1,
                generation=1,
            )
            for entry, tier in ((source, "cpu"), (blocker, "pinned_hbm")):
                replay.session_generations[entry.session_id] = 1
                replay.entries[entry.session_id] = entry
                replay._set_tier(entry, tier)

            replay._start_call(0, "d-blocked", 1)
            state = replay.pd_pending_compute["d-blocked"]
            self.assertTrue(state.prefill_admitted)
            self.assertEqual(state.prefill_start_ns, 0)
            self.assertFalse(state.decode_admitted)
            self.assertFalse(state.lower_restore_scheduled)
            self.assertFalse(state.join_scheduled)
            self.assertEqual(
                list(replay.decode_restore_waiters), [("d-blocked", 1)]
            )

            replay._remove_entry(blocker.session_id)
            replay._wake_decode_restore_head(10)
            replay._handle_decode_restore_capacity_wakeup(
                11, replay.decode_restore_wakeup_generation
            )
            self.assertTrue(state.decode_admitted)
            self.assertEqual(state.decode_admission_ns, 11)
            self.assertEqual(state.lower_restore_issue_ns, 11)
            restore_to_d_finish = self._finish_manual_lower_restore(
                replay, "d-blocked"
            )
            self.assertEqual(
                state.lower_restore_finish_ns, restore_to_d_finish
            )
            self.assertEqual(state.d2p_issue_ns, restore_to_d_finish)
            self.assertTrue(state.join_scheduled)
            self.assertGreaterEqual(
                state.join_completion_ns, state.restore_finish_ns
            )
            jobs = replay.queue.jobs
            timing_events = replay.restore_timing_by_source["cpu"][
                "event_count"
            ]
            replay._advance_pd_call(restore_to_d_finish, state)
            replay._advance_pd_call(restore_to_d_finish, state)
            self.assertEqual(replay.queue.jobs, jobs)
            self.assertEqual(
                replay.restore_timing_by_source["cpu"]["event_count"],
                timing_events,
            )
        finally:
            directory.cleanup()

    def test_default_cold_restore_gates_only_owner_compute_admission(self):
        sessions = [
            _session("cold-owner", 0, 1_000_000),
            _session("hbm-peer", 0, 1_000_000),
        ]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            model = _tiny_model()
            layout = kv_layout(model, 1, 2)
            weight = estimate_model_weight_bytes_per_rank(
                model, 1, 2, layout
            )
            object_bytes = (
                112 * layout.physical_bytes_per_token_per_rank
            )
            replay = _CapacityReplay(
                workload,
                model,
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=weight + 4 * object_bytes,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    enable_transfer_queueing=False,
                    pd_disaggregated=True,
                ),
            )
            for session_id, tier in (
                ("cold-owner", "cpu"),
                ("hbm-peer", "hbm"),
            ):
                entry = _Entry(
                    session_id=session_id,
                    cache_tokens=100,
                    cluster_bytes=object_bytes,
                    per_rank_bytes=object_bytes,
                    tier="none",
                    last_access_ns=0,
                    generation=1,
                )
                replay.session_generations[session_id] = 1
                replay.entries[session_id] = entry
                replay._set_tier(entry, tier)

            replay._start_call(0, "cold-owner", 1)
            cold = replay.pd_pending_compute["cold-owner"]
            self.assertTrue(cold.decode_admitted)
            self.assertGreater(cold.decode_reservation_bytes, 0)
            self.assertTrue(cold.lower_restore_scheduled)
            self.assertFalse(cold.decode_prefix_ready)
            self.assertFalse(cold.prefill_admitted)
            self.assertNotIn("cold-owner", replay.prefill_waiter_sessions)

            replay._start_call(0, "hbm-peer", 1)
            peer = replay.pd_pending_compute["hbm-peer"]
            self.assertTrue(peer.prefill_admitted)
            self.assertTrue(peer.decode_admitted)
            self.assertTrue(peer.d2p_scheduled)

            finish_ns = self._finish_manual_lower_restore(
                replay, "cold-owner"
            )
            self.assertTrue(cold.decode_prefix_ready)
            self.assertTrue(cold.prefill_admitted)
            self.assertTrue(cold.d2p_scheduled)
            self.assertGreaterEqual(cold.d2p_issue_ns, finish_ns)
        finally:
            directory.cleanup()

    def test_restore_ready_call_cannot_deadlock_behind_younger_p_waiter(self):
        sessions = [
            _session("older-cold", 0, 1_000_000),
            _session("younger-hbm", 0, 1_000_000),
        ]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            model = _tiny_model()
            layout = kv_layout(model, 1, 2)
            weight = estimate_model_weight_bytes_per_rank(
                model, 1, 2, layout
            )
            object_bytes = (
                112 * layout.physical_bytes_per_token_per_rank
            )
            replay = _CapacityReplay(
                workload,
                model,
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=weight + 2 * object_bytes,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    enable_transfer_queueing=False,
                    pd_disaggregated=True,
                ),
            )
            for session_id, tier in (
                ("older-cold", "cpu"),
                ("younger-hbm", "hbm"),
            ):
                entry = _Entry(
                    session_id=session_id,
                    cache_tokens=100,
                    cluster_bytes=object_bytes,
                    per_rank_bytes=object_bytes,
                    tier="none",
                    last_access_ns=0,
                    generation=1,
                )
                replay.session_generations[session_id] = 1
                replay.entries[session_id] = entry
                replay._set_tier(entry, tier)

            # Fill the independent P pool while the older CPU restore runs.
            # The younger HBM-ready call is allowed to reach the P FIFO, but
            # cannot be admitted yet.
            replay.active["p-blocker"] = _Active(
                "p-blocker", 2 * object_bytes, -1
            )
            replay.active_bytes_per_rank = 2 * object_bytes
            replay._record_active_peak()
            replay._start_call(0, "older-cold", 1)
            replay._start_call(0, "younger-hbm", 1)
            self.assertEqual(
                list(replay.prefill_waiters), [("younger-hbm", 1)]
            )

            finish_ns = self._finish_manual_lower_restore(
                replay, "older-cold"
            )
            self.assertEqual(
                list(replay.prefill_waiters),
                [("older-cold", 1), ("younger-hbm", 1)],
            )
            self.assertEqual(replay.prefill_wakeup_ns, finish_ns + 1)

            blocker = replay.active.pop("p-blocker")
            replay.active_bytes_per_rank -= blocker.per_rank_bytes
            replay._handle_prefill_capacity_wakeup(
                replay.prefill_wakeup_ns,
                replay.prefill_wakeup_generation,
            )
            self.assertTrue(
                replay.pd_pending_compute["older-cold"].prefill_admitted
            )
            self.assertEqual(
                list(replay.prefill_waiters), [("younger-hbm", 1)]
            )
        finally:
            directory.cleanup()

    def test_default_capacity_and_restore_metadata_match_dgx_sensitivity(self):
        config = CapacityReplayConfig(hbm_capacity_bytes_per_rank=1 << 30)
        self.assertEqual(config.cpu_capacity_bytes, 2_000_000_000_000)
        self.assertEqual(config.restore_execution_mode, "async-pre-admission")

    def test_pd_p_blocked_does_not_delay_lower_restore_branch(self):
        session = _session("p-blocked", 0, 1_000_000)
        directory, path = self._write([session])
        try:
            workload = load_capacity_replay_workload(path)
            model = _tiny_model()
            layout = kv_layout(model, 1, 2)
            weight = estimate_model_weight_bytes_per_rank(
                model, 1, 2, layout
            )
            object_bytes = (
                112 * layout.physical_bytes_per_token_per_rank
            )
            replay = _CapacityReplay(
                workload,
                model,
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=weight + object_bytes,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    enable_transfer_queueing=False,
                    pd_disaggregated=True,
                    restore_execution_mode="async-decode-join",
                ),
            )
            source = _Entry(
                session_id="p-blocked",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=0,
                generation=1,
            )
            replay.session_generations[source.session_id] = 1
            replay.entries[source.session_id] = source
            replay._set_tier(source, "cpu")
            replay.active["p-capacity-blocker"] = _Active(
                "p-capacity-blocker", object_bytes, -1
            )
            replay.active_bytes_per_rank = object_bytes
            replay._record_active_peak()

            replay._start_call(0, "p-blocked", 1)
            state = replay.pd_pending_compute["p-blocked"]
            self.assertTrue(state.decode_admitted)
            self.assertEqual(
                state.decode_reservation_bytes,
                state.full_decode_per_rank,
            )
            self.assertEqual(state.decode_admission_ns, 0)
            self.assertTrue(state.lower_restore_scheduled)
            self.assertEqual(state.lower_restore_issue_ns, 0)
            self.assertFalse(state.prefill_admitted)
            self.assertFalse(state.join_scheduled)
            self.assertEqual(
                list(replay.prefill_waiters), [("p-blocked", 1)]
            )

            restore_to_d_finish = self._finish_manual_lower_restore(
                replay, "p-blocked"
            )
            self.assertTrue(state.decode_prefix_ready)
            pinned = replay.entries["p-blocked"].per_rank_bytes
            self.assertGreaterEqual(
                pinned + state.decode_reservation_bytes,
                state.full_decode_per_rank,
            )
            self.assertFalse(state.d2p_scheduled)
            blocker = replay.active.pop("p-capacity-blocker")
            replay.active_bytes_per_rank -= blocker.per_rank_bytes
            replay._wake_prefill_head(restore_to_d_finish + 10)
            replay._handle_prefill_capacity_wakeup(
                restore_to_d_finish + 11,
                replay.prefill_wakeup_generation,
            )
            self.assertTrue(state.prefill_admitted)
            self.assertEqual(
                state.prefill_start_ns, restore_to_d_finish + 11
            )
            self.assertEqual(
                state.d2p_issue_ns, restore_to_d_finish + 11
            )
            self.assertTrue(state.join_scheduled)
            self.assertGreaterEqual(
                state.join_completion_ns,
                state.overlap_prefill_finish_ns,
            )
        finally:
            directory.cleanup()

    def test_pd_no_reuse_reserves_decode_before_prefill_admission(self):
        session = {
            "session_id": "no-reuse-guard",
            "arrival_time_ns": 0,
            "sub_requests": [{
                "input_toks": 100,
                "output_toks": 1,
                "tool_duration_ns": 0,
            }],
        }
        directory, path = self._write([session])
        try:
            workload = load_capacity_replay_workload(path)
            model = _tiny_model()
            layout = kv_layout(model, 1, 2)
            weight = estimate_model_weight_bytes_per_rank(
                model, 1, 2, layout
            )
            object_bytes = (
                112 * layout.physical_bytes_per_token_per_rank
            )
            replay = _CapacityReplay(
                workload,
                model,
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=weight + object_bytes,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    pd_disaggregated=True,
                ),
            )
            blocker = _Entry(
                session_id="no-reuse-d-blocker",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=-1,
                generation=1,
            )
            replay.session_generations[blocker.session_id] = 1
            replay.entries[blocker.session_id] = blocker
            replay._set_tier(blocker, "pinned_hbm")

            replay._start_call(0, "no-reuse-guard", 0)
            state = replay.pd_pending_compute["no-reuse-guard"]
            self.assertFalse(state.prefill_admitted)
            self.assertFalse(state.decode_admitted)
            self.assertFalse(state.join_scheduled)
            replay._remove_entry(blocker.session_id)
            late_decode_admission_ns = 100
            replay._wake_decode_restore_head(
                late_decode_admission_ns - 1
            )
            replay._handle_decode_restore_capacity_wakeup(
                late_decode_admission_ns,
                replay.decode_restore_wakeup_generation,
            )
            self.assertTrue(state.decode_admitted)
            self.assertEqual(
                state.decode_reservation_bytes,
                state.full_decode_per_rank,
            )
            self.assertTrue(state.prefill_admitted)
            self.assertEqual(
                state.prefill_start_ns, late_decode_admission_ns
            )
            self.assertTrue(state.join_scheduled)
            self.assertEqual(
                state.join_completion_ns,
                late_decode_admission_ns + state.compute_ns,
            )
        finally:
            directory.cleanup()

    def test_pd_pre_admission_avoids_cross_pool_hold_and_wait(self):
        model = _tiny_model()
        layout = kv_layout(model, 1, 2)
        weight = estimate_model_weight_bytes_per_rank(
            model, 1, 2, layout
        )
        object_bytes = 112 * layout.physical_bytes_per_token_per_rank
        sessions = [
            _session("older", 0, 1, tokens=100),
            _session("younger", 0, 1, tokens=100),
        ]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            for policy in ("hbm_ssd_direct", "tiered"):
                with self.subTest(policy=policy):
                    report = replay_capacity_aware(
                        workload,
                        model,
                        DEFAULT_HARDWARE_SPECS["H100"],
                        1,
                        2,
                        CapacityReplayConfig(
                            hbm_capacity_bytes_per_rank=(
                                weight + object_bytes
                            ),
                            cpu_capacity_bytes=1 << 30,
                            ssd_capacity_bytes=1 << 30,
                            policy=policy,
                            demotion_mode="capacity-only",
                            pd_disaggregated=True,
                            restore_execution_mode="async-pre-admission",
                        ),
                    )
                    self.assertEqual(
                        report["resume"][
                            "all_selected_positive_transition_count"
                        ],
                        2,
                    )
                    self.assertEqual(
                        sum(
                            report["resume"]["source_counts"][source]
                            for source in (
                                "decode_hbm", "cpu", "ssd", "recompute"
                            )
                        ),
                        2,
                    )
        finally:
            directory.cleanup()

    def test_pd_resident_hbm_backfills_behind_lower_restore_waiter(self):
        sessions = [
            _session("older", 0, 1_000_000),
            _session("younger", 0, 1_000_000),
        ]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            model = _tiny_model()
            layout = kv_layout(model, 1, 2)
            weight = estimate_model_weight_bytes_per_rank(
                model, 1, 2, layout
            )
            object_bytes = (
                112 * layout.physical_bytes_per_token_per_rank
            )
            replay = _CapacityReplay(
                workload,
                model,
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=weight + 2 * object_bytes,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    enable_transfer_queueing=False,
                    pd_disaggregated=True,
                    restore_execution_mode="async-decode-join",
                ),
            )
            older_entry = _Entry(
                session_id="older",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=0,
                generation=1,
            )
            replay.session_generations["older"] = 1
            replay.entries["older"] = older_entry
            replay._set_tier(older_entry, "cpu")
            younger_entry = _Entry(
                session_id="younger",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=0,
                generation=1,
            )
            replay.session_generations["younger"] = 1
            replay.entries["younger"] = younger_entry
            replay._set_tier(younger_entry, "hbm")
            blocker = _Entry(
                session_id="ordered-d-blocker",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=-1,
                generation=1,
            )
            replay.session_generations[blocker.session_id] = 1
            replay.entries[blocker.session_id] = blocker
            replay._set_tier(blocker, "pinned_hbm")

            replay._start_call(0, "older", 1)
            replay._start_call(0, "younger", 1)
            older = replay.pd_pending_compute["older"]
            younger = replay.pd_pending_compute["younger"]
            self.assertTrue(older.prefill_admitted)
            self.assertFalse(older.decode_admitted)
            self.assertTrue(younger.prefill_admitted)
            self.assertTrue(younger.decode_admitted)
            self.assertTrue(younger.d2p_scheduled)
            self.assertEqual(
                list(replay.decode_restore_waiters),
                [("older", 1)],
            )
        finally:
            directory.cleanup()

    def test_pd_safe_backfill_retires_existing_decode_waiter(self):
        sessions = [
            _session("stale-older", 0, 1_000_000),
            _session("stale-backfill", 0, 1_000_000),
        ]
        sessions[1]["sub_requests"].append({
            "input_toks": 116,
            "output_toks": 1,
            "tool_duration_ns": 0,
            "prefix_reuse_toks": 108,
            "prefix_reuse_source": "reported",
        })
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            model = _tiny_model()
            layout = kv_layout(model, 1, 2)
            weight = estimate_model_weight_bytes_per_rank(
                model, 1, 2, layout
            )
            object_bytes = (
                112 * layout.physical_bytes_per_token_per_rank
            )
            replay = _CapacityReplay(
                workload,
                model,
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=weight + 2 * object_bytes,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    enable_transfer_queueing=False,
                    pd_disaggregated=True,
                    restore_execution_mode="async-decode-join",
                ),
            )
            older_entry = _Entry(
                session_id="stale-older",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=0,
                generation=1,
            )
            replay.session_generations[older_entry.session_id] = 1
            replay.entries[older_entry.session_id] = older_entry
            replay._set_tier(older_entry, "cpu")
            backfill_entry = _Entry(
                session_id="stale-backfill",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=0,
                generation=1,
            )
            replay.session_generations[backfill_entry.session_id] = 1
            replay.entries[backfill_entry.session_id] = backfill_entry
            replay._set_tier(backfill_entry, "hbm")
            decode_blocker = _Entry(
                session_id="stale-d-blocker",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=-1,
                generation=1,
            )
            replay.session_generations[decode_blocker.session_id] = 1
            replay.entries[decode_blocker.session_id] = decode_blocker
            replay._set_tier(decode_blocker, "pinned_hbm")
            replay.active["stale-p-blocker"] = _Active(
                "stale-p-blocker", object_bytes, -1
            )
            replay.active_bytes_per_rank += object_bytes

            replay._start_call(0, "stale-older", 1)
            replay._start_call(0, "stale-backfill", 1)
            self.assertEqual(
                list(replay.decode_restore_waiters),
                [("stale-older", 1), ("stale-backfill", 1)],
            )
            self.assertEqual(
                list(replay.prefill_waiters), [("stale-backfill", 1)]
            )

            replay.active.pop("stale-p-blocker")
            replay.active_bytes_per_rank -= object_bytes
            replay._wake_prefill_head(0)
            replay._handle_prefill_capacity_wakeup(
                1, replay.prefill_wakeup_generation
            )

            state = replay.pd_pending_compute["stale-backfill"]
            self.assertTrue(state.decode_admitted)
            self.assertTrue(state.prefill_admitted)
            self.assertEqual(
                list(replay.decode_restore_waiters),
                [("stale-older", 1)],
            )
            self.assertNotIn(
                "stale-backfill", replay.decode_restore_waiter_sessions
            )
            self.assertNotIn(
                "stale-backfill", replay.decode_restore_waiter_since_ns
            )
            self.assertNotIn(
                "stale-backfill", replay.prefill_waiter_sessions
            )
            self.assertNotIn(
                "stale-backfill", replay.prefill_waiter_since_ns
            )

            replay._prompt_complete(
                state.join_completion_ns,
                "stale-backfill",
                1,
                state.active_per_rank,
            )
            completion_events = [
                event for event in replay.events
                if event[3] == "call_complete"
                and event[4][0] == "stale-backfill"
                and event[4][1] == 1
            ]
            self.assertEqual(len(completion_events), 1)
            completion_ns, _, _, _, payload = completion_events[0]
            replay._complete_call(
                completion_ns,
                str(payload[0]),
                int(payload[1]),
                int(payload[2]),
                int(payload[3]),
                int(payload[4]),
            )
            replay._start_call(completion_ns, "stale-backfill", 2)
            self.assertEqual(
                replay.pd_pending_compute["stale-backfill"].call_index, 2
            )
            replay._handle_decode_restore_capacity_wakeup(
                completion_ns + 1,
                replay.decode_restore_wakeup_generation,
            )
            self.assertNotIn(
                ("stale-backfill", 1), replay.decode_restore_waiters
            )
        finally:
            directory.cleanup()

    def test_pd_growing_hbm_prefill_runs_while_decode_branch_queues(self):
        sessions = [
            _session("older-growing", 0, 1_000_000),
            _session("younger-growing", 0, 1_000_000, tokens=105),
        ]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            model = _tiny_model()
            layout = kv_layout(model, 1, 2)
            weight = estimate_model_weight_bytes_per_rank(
                model, 1, 2, layout
            )
            object_bytes = (
                112 * layout.physical_bytes_per_token_per_rank
            )
            growing_bytes = (
                128 * layout.physical_bytes_per_token_per_rank
            )
            replay = _CapacityReplay(
                workload,
                model,
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=(
                        weight + object_bytes + growing_bytes
                    ),
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    enable_transfer_queueing=False,
                    pd_disaggregated=True,
                    restore_execution_mode="async-decode-join",
                ),
            )
            older_entry = _Entry(
                session_id="older-growing",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=0,
                generation=1,
            )
            replay.session_generations[older_entry.session_id] = 1
            replay.entries[older_entry.session_id] = older_entry
            replay._set_tier(older_entry, "cpu")
            younger_entry = _Entry(
                session_id="younger-growing",
                cache_tokens=105,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=0,
                generation=1,
            )
            replay.session_generations[younger_entry.session_id] = 1
            replay.entries[younger_entry.session_id] = younger_entry
            replay._set_tier(younger_entry, "hbm")
            blocker = _Entry(
                session_id="growing-d-blocker",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=-1,
                generation=1,
            )
            replay.session_generations[blocker.session_id] = 1
            replay.entries[blocker.session_id] = blocker
            replay._set_tier(blocker, "pinned_hbm")

            replay._start_call(0, "older-growing", 1)
            replay._start_call(0, "younger-growing", 1)

            older = replay.pd_pending_compute["older-growing"]
            younger = replay.pd_pending_compute["younger-growing"]
            self.assertTrue(older.prefill_admitted)
            self.assertFalse(older.decode_admitted)
            self.assertTrue(younger.prefill_admitted)
            self.assertFalse(younger.decode_admitted)
            self.assertNotIn(
                "younger-growing", replay.decode_restore_source_pins
            )
            self.assertGreater(
                younger.full_decode_per_rank,
                younger_entry.per_rank_bytes,
            )
            self.assertEqual(
                list(replay.decode_restore_waiters),
                [("older-growing", 1), ("younger-growing", 1)],
            )
        finally:
            directory.cleanup()

    def test_pd_serial_source_loss_restarts_compute_at_finalize(self):
        directory, path = self._write(
            [_session("serial-source-loss", 0, 1_000_000)]
        )
        try:
            workload = load_capacity_replay_workload(path)
            model = _tiny_model()
            layout = kv_layout(model, 1, 2)
            object_bytes = (
                112 * layout.physical_bytes_per_token_per_rank
            )
            replay = _CapacityReplay(
                workload,
                model,
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=1 << 30,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    enable_transfer_queueing=False,
                    pd_disaggregated=True,
                    restore_execution_mode="serial-before-prefill",
                ),
            )
            entry = _Entry(
                session_id="serial-source-loss",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=0,
                generation=1,
                move_reason="completion",
            )
            replay.session_generations[entry.session_id] = 1
            replay.entries[entry.session_id] = entry
            replay._set_tier(entry, "ssd")
            call_key = ("serial-source-loss", 1)
            replay.call_logical_ready_ns[call_key] = 0
            state = replay._initialize_pd_call_state(
                0,
                "serial-source-loss",
                1,
                "ssd",
                "completion",
            )
            replay._try_start_pd_prefill(0, state)
            self.assertFalse(state.prefill_admitted)

            entry.drop_reason = "ssd_ttl"
            replay._set_tier(entry, "dropped")
            replay._advance_pd_call(100, state)

            self.assertEqual(state.source, "recompute")
            self.assertEqual(state.prefill_start_ns, 100)
            self.assertEqual(state.speculative_compute_seconds, 0.0)
            self.assertTrue(state.join_scheduled)
            self.assertEqual(
                state.join_completion_ns, 100 + state.compute_ns
            )
        finally:
            directory.cleanup()

    def test_pd_uses_two_pools_and_separate_d_first_restore(self):
        sessions = [_session("pd", 0, 100_000_000)]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            model = _tiny_model()
            layout = kv_layout(model, 1, 2)
            report = replay_capacity_aware(
                workload, model, DEFAULT_HARDWARE_SPECS["H100"], 1, 2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=1 << 30,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    enable_transfer_queueing=False,
                    cancel_migration_on_resume=True,
                    pd_disaggregated=True,
                ),
            )
        finally:
            directory.cleanup()
        self.assertEqual(report["execution_scope"]["pd_hbm_pool_count"], 2)
        self.assertEqual(report["resume"]["source_counts"]["cpu"], 1)
        jobs = report["transfer_queue"]["jobs_by_kind"]
        self.assertEqual(jobs["cpu_to_decode"], 1)
        self.assertEqual(jobs["decode_hbm_to_prefill"], 1)
        # Exact D->P prefix bytes; P->D is first full prompt plus the suffix.
        per_token_cluster = layout.physical_bytes_per_token_cluster
        self.assertEqual(
            report["pd_transfer"]["decode_to_prefill_bytes"],
            100 * per_token_cluster,
        )
        self.assertEqual(
            report["pd_transfer"]["prefill_to_decode_bytes"],
            (100 + 8) * per_token_cluster,
        )
        peak = report["capacity"]["peak_occupancy"]
        self.assertGreater(peak["prefill_hbm_active_bytes_per_rank"], 0)
        self.assertGreater(
            peak["decode_hbm_active_plus_idle_bytes_per_rank"], 0
        )
        admission = report["admission_queues"]["decode_restore_hbm"]
        self.assertIn("completion-safe", admission["discipline"])
        self.assertEqual(
            admission["aggregate_admission_wait_seconds"],
            admission["aggregate_capacity_block_seconds"],
        )
        self.assertIn(
            "not only physical capacity shortage",
            admission["aggregate_capacity_block_seconds_semantics"],
        )

    def test_pd_decode_capacity_waits_for_pinned_call_completion(self):
        sessions = []
        for index in range(2):
            sessions.append({
                "session_id": f"overlap-{index}",
                # With the tiny layout, a 100-token P->D handoff at the
                # default 900 GB/s + 3 us takes ceil(3007.11)=3008 ns. This
                # makes B's prompt completion coincide with A's CALL_COMPLETE
                # and exercises event-priority-safe capacity retry.
                "arrival_time_ns": index * 3008,
                "sub_requests": [
                    {"input_toks": 100, "output_toks": 1000,
                     "tool_duration_ns": 1_000_000_000},
                    {"input_toks": 1100, "output_toks": 1,
                     "tool_duration_ns": 0, "prefix_reuse_toks": 1099},
                ],
            })
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            model = _tiny_model()
            layout = kv_layout(model, 1, 2)
            weight = estimate_model_weight_bytes_per_rank(
                model, 1, 2, layout
            )
            report = replay_capacity_aware(
                workload, model, DEFAULT_HARDWARE_SPECS["H100"], 1, 2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=(
                        weight
                        + 1200 * layout.physical_bytes_per_token_per_rank
                    ),
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    enable_transfer_queueing=False,
                    pd_disaggregated=True,
                ),
            )
        finally:
            directory.cleanup()
        self.assertEqual(
            report["resume"]["reuse_eligible_transition_count"], 2
        )
        self.assertGreater(
            report["resume"]["aggregate_hbm_capacity_block_seconds"], 0
        )
        self.assertTrue(report["capacity"]["capacity_invariant_checked"])

    def test_pd_prefill_capacity_serializes_concurrent_admission(self):
        sessions = [
            _session(f"prefill-wait-{index}", 0, 1_000_000)
            for index in range(8)
        ]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            model = _tiny_model()
            layout = kv_layout(model, 1, 2)
            weight = estimate_model_weight_bytes_per_rank(
                model, 1, 2, layout
            )
            one_object = (
                112 * layout.physical_bytes_per_token_per_rank
            )
            report = replay_capacity_aware(
                workload, model, DEFAULT_HARDWARE_SPECS["H100"], 1, 2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=weight + one_object + 1024,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    hbm_ttl_ns=10_000_000_000,
                    cpu_ttl_ns=10_000_000_000,
                    ssd_ttl_ns=10_000_000_000,
                    enable_transfer_queueing=True,
                    pd_disaggregated=True,
                ),
            )
        finally:
            directory.cleanup()
        self.assertEqual(
            report["resume"]["reuse_eligible_transition_count"], 8
        )
        self.assertGreater(
            report["resume"]["aggregate_hbm_capacity_block_seconds"], 0
        )
        self.assertTrue(report["capacity"]["capacity_invariant_checked"])

    def test_pd_decode_restore_admission_uses_one_fcfs_wakeup(self):
        sessions = [
            _session(f"restore-wait-{index}", 0, 1_000_000)
            for index in range(3)
        ]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            model = _tiny_model()
            hardware = DEFAULT_HARDWARE_SPECS["H100"]
            layout = kv_layout(model, 1, 2)
            weight = estimate_model_weight_bytes_per_rank(
                model, 1, 2, layout
            )
            object_bytes = (
                112 * layout.physical_bytes_per_token_per_rank
            )
            replay = _CapacityReplay(
                workload,
                model,
                hardware,
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=weight + 2 * object_bytes,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    enable_transfer_queueing=False,
                    pd_disaggregated=True,
                    restore_execution_mode="async-decode-join",
                ),
            )
            for index, session in enumerate(sessions):
                session_id = session["session_id"]
                entry = _Entry(
                    session_id=session_id,
                    cache_tokens=100,
                    cluster_bytes=object_bytes,
                    per_rank_bytes=object_bytes,
                    tier="none",
                    last_access_ns=0,
                    generation=1,
                )
                replay.session_generations[session_id] = 1
                replay.entries[session_id] = entry
                replay._set_tier(entry, "ssd" if index == 2 else "cpu")
                replay._push_lru(entry)
            blocker = _Entry(
                session_id="decode-blocker",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=-1,
                generation=1,
            )
            replay.session_generations[blocker.session_id] = 1
            replay.entries[blocker.session_id] = blocker
            replay._set_tier(blocker, "hbm")
            replay._push_lru(blocker)

            for index in range(3):
                replay._start_call(0, f"restore-wait-{index}", 1)

            self.assertEqual(
                list(replay.decode_restore_waiters),
                [
                    ("restore-wait-1", 1),
                    ("restore-wait-2", 1),
                ],
            )
            self.assertEqual(replay.decode_restore_max_depth, 2)
            self.assertEqual(replay.decode_restore_wakeup_event_count, 1)
            self.assertEqual(
                list(replay.prefill_waiters),
                [("restore-wait-2", 1)],
            )
            self.assertEqual(
                replay.decode_restore_source_pins,
                {"restore-wait-1"},
            )
            waiter_events = [
                event for event in replay.events
                if event[3] == "decode_restore_capacity_wakeup"
            ]
            direct_retries = [
                event for event in replay.events
                if event[3] == "call_ready"
                and event[4][0] in {"restore-wait-1", "restore-wait-2"}
            ]
            self.assertEqual(len(waiter_events), 1)
            self.assertFalse(direct_retries)
        finally:
            directory.cleanup()

    def test_pd_decode_restore_wait_pins_cpu_and_ssd_sources(self):
        sessions = [
            _session(f"pin-{index}", 0, 1_000_000)
            for index in range(3)
        ]
        directory, path = self._write(sessions)
        try:
            workload = load_capacity_replay_workload(path)
            model = _tiny_model()
            hardware = DEFAULT_HARDWARE_SPECS["H100"]
            layout = kv_layout(model, 1, 2)
            weight = estimate_model_weight_bytes_per_rank(
                model, 1, 2, layout
            )
            object_bytes = (
                112 * layout.physical_bytes_per_token_per_rank
            )
            replay = _CapacityReplay(
                workload,
                model,
                hardware,
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=weight + 2 * object_bytes,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    hbm_ttl_ns=1,
                    cpu_ttl_ns=1,
                    ssd_ttl_ns=1,
                    enable_transfer_queueing=False,
                    pd_disaggregated=True,
                ),
            )
            for index, session in enumerate(sessions):
                session_id = session["session_id"]
                entry = _Entry(
                    session_id=session_id,
                    cache_tokens=100,
                    cluster_bytes=object_bytes,
                    per_rank_bytes=object_bytes,
                    tier="none",
                    last_access_ns=index,
                    generation=1,
                )
                replay.session_generations[session_id] = 1
                replay.entries[session_id] = entry
                replay._set_tier(entry, "ssd" if index == 2 else "cpu")
                replay._push_lru(entry)
            blocker = _Entry(
                session_id="pin-blocker",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=-1,
                generation=1,
            )
            replay.session_generations[blocker.session_id] = 1
            replay.entries[blocker.session_id] = blocker
            replay._set_tier(blocker, "hbm")
            replay._push_lru(blocker)

            for index in range(3):
                replay._start_call(0, f"pin-{index}", 1)

            cpu_entry = replay.entries["pin-1"]
            ssd_entry = replay.entries["pin-2"]
            self.assertEqual(cpu_entry.tier, "cpu")
            self.assertEqual(ssd_entry.tier, "ssd")
            replay._handle_ttl(10, "pin-1", cpu_entry.generation, "cpu")
            replay._handle_ttl(10, "pin-2", ssd_entry.generation, "ssd")
            self.assertEqual(cpu_entry.tier, "cpu")
            self.assertEqual(ssd_entry.tier, "dropped")
            self.assertEqual(
                replay.decode_restore_source_ttl_deferral_count, 1
            )

            # Only the admitted-frontier source is pinned. The younger source
            # remains policy-managed until its source selection epoch.
            self.assertIsNone(replay._lru_entry("cpu"))
            cpu_retry = replay._ensure_cpu_space(
                replay.config.cpu_capacity_bytes - replay.used["cpu"] + 1,
                10,
            )
            self.assertIsNotNone(cpu_retry)
            self.assertEqual(cpu_entry.tier, "cpu")
            self.assertEqual(ssd_entry.tier, "dropped")
        finally:
            directory.cleanup()

    def test_pd_prefill_and_decode_restore_waiters_are_independent(self):
        directory, path = self._write([_session("independent", 0, 1)])
        try:
            workload = load_capacity_replay_workload(path)
            replay = _CapacityReplay(
                workload,
                _tiny_model(),
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=1 << 30,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    pd_disaggregated=True,
                ),
            )
            replay._queue_prefill_waiter(100, "prefill", 0, 0)
            replay._queue_decode_restore_waiter(50, "restore", 1, 0)
            self.assertEqual(list(replay.prefill_waiters), [("prefill", 0)])
            self.assertEqual(
                list(replay.decode_restore_waiters), [("restore", 1)]
            )
            self.assertEqual(replay.prefill_wakeup_ns, 100)
            self.assertEqual(replay.decode_restore_wakeup_ns, 50)
            self.assertEqual(
                {event[3] for event in replay.events},
                {
                    "prefill_capacity_wakeup",
                    "decode_restore_capacity_wakeup",
                },
            )
        finally:
            directory.cleanup()

    def test_pd_prefill_progresses_while_atomic_cpu_demotion_finishes(self):
        directory, path = self._write([_session("transit-pd", 0, 1)])
        try:
            workload = load_capacity_replay_workload(path)
            model = _tiny_model()
            layout = kv_layout(model, 1, 2)
            weight = estimate_model_weight_bytes_per_rank(
                model, 1, 2, layout
            )
            object_bytes = (
                112 * layout.physical_bytes_per_token_per_rank
            )
            replay = _CapacityReplay(
                workload,
                model,
                DEFAULT_HARDWARE_SPECS["H100"],
                1,
                2,
                CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=weight + object_bytes,
                    cpu_capacity_bytes=1 << 30,
                    ssd_capacity_bytes=1 << 30,
                    enable_transfer_queueing=False,
                    pd_disaggregated=True,
                    restore_execution_mode="async-decode-join",
                ),
            )
            entry = _Entry(
                session_id="transit-pd",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=0,
                generation=1,
                available_ns=100,
                move_reason="cpu_capacity",
                transit_source_tier="cpu",
            )
            replay.session_generations[entry.session_id] = 1
            replay.entries[entry.session_id] = entry
            replay._set_tier(entry, "transit_ssd")
            blocker = _Entry(
                session_id="transit-d-blocker",
                cache_tokens=100,
                cluster_bytes=object_bytes,
                per_rank_bytes=object_bytes,
                tier="none",
                last_access_ns=-1,
                generation=1,
            )
            replay.session_generations[blocker.session_id] = 1
            replay.entries[blocker.session_id] = blocker
            replay._set_tier(blocker, "pinned_hbm")

            replay._start_call(0, "transit-pd", 1)

            state = replay.pd_pending_compute["transit-pd"]
            self.assertEqual(state.source, "cpu")
            self.assertTrue(state.prefill_admitted)
            self.assertFalse(state.decode_admitted)
            self.assertEqual(
                list(replay.decode_restore_waiters),
                [("transit-pd", 1)],
            )
            self.assertEqual(replay.resume_inflight_migration_wait_ns, 100)
        finally:
            directory.cleanup()


if __name__ == "__main__":
    unittest.main()

import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serving.online_experiments import (
    ExperimentError,
    SCHEMA_VERSION,
    _agentic_config_fingerprints,
    _canonical_agentic_config_payload,
    _apply_dataset_path_overrides,
    _external_fabric_model_coexecution_audit,
    _expected_poisson_offered_arrivals,
    _derive_exact_rate_metrics,
    _normalize_allowed_timing_warning_codes,
    _normalize_plot_settings,
    _operational_metric_sources,
    _prepare_ssd_resume_opportunity_contract,
    _stable_json_hash,
    _validate_completed_report,
    _validate_measurement_cohort_contract,
    _validate_offered_arrival_trace,
    _validate_poisson_common_random_numbers,
    _validate_policy_invariants,
    _validate_session_queue_contract,
    _validated_resume_timing,
    _validate_summary_row,
    _validate_trace_identity,
    build_backlog_slowdown_audit,
    build_parser,
    build_run_descriptors,
    build_ssd_resume_opportunity_audit,
    collect_results,
    execute_run,
    materialize_session_cohort,
    plot_backlog_oracle_normalized_throughput,
    plot_grouped_throughput,
    plot_poisson_reference_normalized_jct,
    plot_poisson_session_jct,
    plot_poisson_session_jct_decomposition,
    plot_poisson_rate_metrics,
    run_suite,
    save_results,
)


def _session(
        session_id, context, gap, *, output_tokens=10, gap_ns=100,
        reuse_tokens=None):
    if reuse_tokens is None:
        reuse_tokens = context
    return {
        "session_id": session_id,
        "arrival_time_ns": 0,
        "sub_requests": [
            {
                "input_toks": context,
                "output_toks": output_tokens,
                "tool_duration_ns": gap_ns,
                "inter_turn_gap_type": gap,
                "prefix_reuse_toks": 0,
                "policy_independent_reuse_toks": 0,
                "newly_append_toks": context,
                "reported_input_toks": context,
                "observed_provider_hit_toks": 0,
                "lineage_status": "session_start",
            },
            {
                "input_toks": context + 10,
                "output_toks": output_tokens,
                "tool_duration_ns": 0,
                "inter_turn_gap_type": "none",
                "prefix_reuse_toks": reuse_tokens,
                "policy_independent_reuse_toks": reuse_tokens,
                "newly_append_toks": 10,
                "reported_input_toks": context + 10,
                "observed_provider_hit_toks": max(0, reuse_tokens - 1),
                "lineage_status": "adjacent_estimate",
            },
        ],
    }


def _distribution(value, count=1):
    return {
        "count": count,
        "sum": value * count,
        "mean": value,
        "p50": value,
        "p90": value,
        "p99": value,
        "min": value,
        "max": value,
    }


def _session_report(run_id, session_ids, *, source="cpu", oracle=False):
    requests = len(session_ids) * 2
    def request_group(count):
        return {
            "count": count,
            "latency_ns": _distribution(100, count),
            "release_to_first_schedule_ns": _distribution(10, count),
            "scheduler_queue_wait_ns": _distribution(2, count),
            "ttft_ns": _distribution(20, count),
            "tpot_ns": _distribution(5, count),
            "itl_ns": _distribution(5, count),
            "restore_gate_wait_ns": _distribution(3, count),
            "owner_ready_gate_ns": _distribution(3, count),
            "pd_pair_fifo_wait_ns": _distribution(0, count),
            "prepare_boundary_wait_ns": _distribution(0, count),
            "source_demotion_join_wait_ns": _distribution(0, count),
            "hbm_admission_wait_ns": _distribution(1, count),
            "transient_dram_capacity_wait_ns": _distribution(0, count),
            "restore_queue_wait_ns": _distribution(1, count),
            "restore_service_ns": _distribution(1, count),
            "pd_launch_admission_wait_ns": _distribution(1, count),
            "pd_launch_admission_critical_wait_ns": _distribution(1, count),
        }
    group = request_group(requests)
    initial = request_group(len(session_ids))
    resume = request_group(len(session_ids))
    offered_trace = [
        {"session_id": value, "offered_time_ns": index}
        for index, value in enumerate(session_ids)
    ]
    return {
        "run_id": run_id,
        "measurement_window": {
            "warmup_complete": True,
            "measurement_complete": True,
            "measure_completions_observed": len(session_ids),
            "measurement_start_ns": 0,
            "measurement_end_ns": 1_000,
            "measurement_duration_ns": 1_000,
            "simulated_duration_ns": 1_000,
        },
        "throughput": {
            "sessions_per_second_measurement_window": (
                len(session_ids) * 1_000_000),
            "requests_per_second_measurement_window": 2.0,
            "total_tokens_per_second_measurement_window": 3.0,
            "completed_sessions": len(session_ids),
            "completed_sessions_total": len(session_ids),
            "completed_requests": requests,
            "completed_requests_total": requests,
            "completed_requests_in_session_cohort": requests,
        },
        "active_session_population": {},
        "overhead_denominators": {
            "measured_session_cohort": {
                "request_count": requests,
                "request_latency_ns": requests * 100,
                "pd_pair_fifo_wait_ns": 0,
                "pd_pair_fifo_fraction_of_request_latency": 0.0,
                "prepare_boundary_wait_ns": 0,
                "prepare_boundary_fraction_of_request_latency": 0.0,
            },
            "strict_completion_window": {
                "request_count": requests,
                "request_latency_ns": requests * 100,
                "pd_pair_fifo_wait_ns": 0,
                "pd_pair_fifo_fraction_of_request_latency": 0.0,
                "prepare_boundary_wait_ns": 0,
                "prepare_boundary_fraction_of_request_latency": 0.0,
            },
        },
        "sessions": {
            "offered_arrival_trace_count": len(session_ids),
            "offered_arrival_trace_sha256": _stable_json_hash(
                {"fixture": "same-offered-arrival-trace"}),
            "admission_queue_wait_ns": _distribution(4, len(session_ids)),
            "e2e_from_admission_ns": _distribution(
                80, len(session_ids)),
            "e2e_from_offer_ns": _distribution(84, len(session_ids)),
            "records": [
                {
                    "session_id": value,
                    "offered_time_ns": index,
                    "measurement_included": True,
                }
                for index, value in enumerate(session_ids)
            ],
        },
        "requests": {
            "records": [],
            "all": group,
            "initial": initial,
            "resume": resume,
            "resume_by_source": {source: resume},
            "resume_by_return_gap_type": {"tool": resume},
            "resume_by_residency_at_return": {source: resume},
            "resume_by_return_gap_type_and_source": {
                "tool": {source: resume}},
            "resume_by_return_gap_type_and_residency_at_return": {
                "tool": {source: resume}},
        },
        "online_model_compute": {},
        "strict_infinite_hbm_oracle": (
            {"passed": True} if oracle else None),
        "validation": {"timing": {"passed": True, "warnings": []}},
        "censoring": {"censored_sessions": 0},
    }


def _agentic_report(run_id):
    return {
        "schema_version": 20,
        "run_id": run_id,
        "simulated_duration_ns": 1_000,
        "time_breakdown": {},
        "totals": {
            "pd_chunk_admissions": 5,
            "pd_chunk_waiting_admissions": 2,
            "pd_chunk_admitted_tokens": 80,
            "pd_chunk_prefill_reserved_bytes": 1000,
            "pd_chunk_decode_reserved_bytes": 2000,
            "pd_chunk_admission_wait_ns": 23,
            "pd_chunk_admission_critical_wait_ns": 19,
            "pd_chunk_snapshot_joined_admissions": 1,
            "pd_chunk_snapshot_feasible_admissions": 1,
            "pd_chunk_snapshot_feasible_waiting_admissions": 1,
            "pd_chunk_snapshot_feasible_wait_ns": 7,
            "pd_chunk_cancelled_admissions": 0,
            "pd_chunk_cancelled_waiting_admissions": 0,
            "pd_chunk_cancelled_admission_wait_ns": 0,
            "pd_chunk_cancelled_admission_critical_wait_ns": 0,
            "pd_active_prefill_recompute_preemptions": 0,
            "pd_active_prefill_recompute_tokens": 0,
            "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute": 0,
        },
        "asynchronous_restore": {},
        "synchronous_swap": {},
        "observed_load_activity": {},
        "queue_recompute_policy": {
            "full_restore_decisions": 3,
            "partial_restore_decisions": 2,
            "zero_restore_decisions": 1,
            "partial_cpu_decisions": 1,
            "partial_ssd_decisions": 1,
            "selected_restore_tokens": 48,
            "dropped_suffix_tokens": 32,
            "selected_restore_bytes": 4800,
            "dropped_suffix_bytes": 3200,
            "modified_full_projected_queue_wait_ns": 7,
            "modified_full_projected_hbm_admission_wait_ns": 11,
            "modified_full_projected_transient_dram_capacity_wait_ns": 5,
            "modified_full_projected_total_wait_ns": 18,
            "modified_full_projected_service_ns": 13,
            "partial_prefix_projected_queue_wait_ns": 2,
            "partial_prefix_projected_hbm_admission_wait_ns": 0,
            "partial_prefix_projected_transient_dram_capacity_wait_ns": 1,
            "partial_prefix_projected_service_ns": 5,
            "selected_estimated_suffix_recompute_comp_ns": 17,
            "accounting_invariants": {"passed": True},
            "selected_projected_queue_wait_ns": 7,
            "selected_projected_hbm_admission_wait_ns": 11,
            "selected_projected_transient_dram_capacity_wait_ns": 5,
            "selected_projected_total_wait_ns": 18,
            "selected_projected_service_ns": 13,
            "selected_estimated_incremental_recompute_comp_ns": 17,
        },
    }


def _canonical_policy_report(policy, *, ssd_hit=False):
    totals = {
        "cpu_hits": 0,
        "ssd_hits": int(ssd_hit),
        "hbm_to_cpu_bytes": 0,
        "cpu_to_hbm_bytes": 0,
        "cpu_to_ssd_bytes": 0,
        "hbm_to_ssd_bytes": 100 if ssd_hit else 0,
        "ssd_to_hbm_bytes": 100 if ssd_hit else 0,
        "ssd_to_cpu_stage_bytes": 100 if ssd_hit else 0,
        "cpu_stage_to_hbm_bytes": 100 if ssd_hit else 0,
        "ssd_host_write_bytes": 0,
        "ssd_host_read_bytes": 100 if ssd_hit else 0,
        "direct_ssd_write_bytes": 0,
        "direct_ssd_read_bytes": 0,
        "background_cancelled_jobs": 0,
        "background_cancelled_bytes": 0,
        "background_wasted_bytes": 0,
        "ssd_demotion_cancelled": 0,
        "ssd_cancelled_host_write_bytes": 0,
        "hbm_capacity_drops": 0,
        "ttl_drops": 0,
        "capacity_drops": 0,
        "ssd_capacity_evictions": 0,
        "ssd_capacity_admission_drops": 0,
        "dropped_misses": 0,
        "capacity_induced_recompute_tokens": 0,
        "policy_avoidable_recompute_tokens": 0,
        "hbf_dropped_recompute_tokens": 0,
        "transient_dram_capacity_oversize": 0,
        "queue_recompute_evaluation_attempts": 0,
        "queue_recompute_severe_gate_passes": 0,
        "queue_recompute_cost_gate_passes": 0,
        "queue_recompute_full_restore_decisions": 0,
        "queue_recompute_partial_restore_decisions": 0,
        "queue_recompute_zero_restore_decisions": 0,
        "queue_recompute_partial_cpu_decisions": 0,
        "queue_recompute_partial_ssd_decisions": 0,
        "queue_recompute_drop_decisions": 0,
        "queue_recompute_cpu_drop_decisions": 0,
        "queue_recompute_ssd_drop_decisions": 0,
        "queue_recompute_dropped_bytes": 0,
        "queue_recompute_avoided_restore_bytes": 0,
        "queue_recompute_physical_entry_dropped_bytes": 0,
        "queue_recompute_projected_queue_wait_ns": 0,
        "queue_recompute_projected_hbm_admission_wait_ns": 0,
        "queue_recompute_projected_transient_dram_capacity_wait_ns": 0,
        "queue_recompute_projected_service_ns": 0,
        "queue_recompute_prefix_projected_queue_wait_ns": 0,
        "queue_recompute_prefix_projected_hbm_admission_wait_ns": 0,
        "queue_recompute_prefix_projected_transient_dram_capacity_wait_ns": 0,
        "queue_recompute_prefix_projected_service_ns": 0,
        "queue_recompute_estimated_recompute_ns": 0,
        "queue_recompute_tokens": 0,
        "queue_recompute_policy_avoidable_tokens": 0,
        "queue_recompute_selected_restore_tokens": 0,
        "queue_recompute_dropped_suffix_tokens": 0,
        "queue_recompute_selected_restore_bytes": 0,
        "queue_recompute_dropped_suffix_bytes": 0,
        "pd_chunk_admissions": 0,
        "pd_chunk_waiting_admissions": 0,
        "pd_chunk_admitted_tokens": 0,
        "pd_chunk_prefill_reserved_bytes": 0,
        "pd_chunk_decode_reserved_bytes": 0,
        "pd_chunk_admission_wait_ns": 0,
        "pd_chunk_admission_critical_wait_ns": 0,
        "pd_chunk_snapshot_joined_admissions": 0,
        "pd_chunk_snapshot_feasible_admissions": 0,
        "pd_chunk_snapshot_feasible_waiting_admissions": 0,
        "pd_chunk_snapshot_feasible_wait_ns": 0,
        "pd_chunk_cancelled_admissions": 0,
        "pd_chunk_cancelled_waiting_admissions": 0,
        "pd_chunk_cancelled_admission_wait_ns": 0,
        "pd_chunk_cancelled_admission_critical_wait_ns": 0,
        "pd_active_prefill_recompute_preemptions": 0,
        "pd_active_prefill_recompute_tokens": 0,
        "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute": 0,
    }
    events = []
    if ssd_hit:
        events = [
            {
                "time_ns": 10,
                "session_id": "session",
                "event": "resume",
                "source": "ssd",
                "source_instance_id": 1,
                "target_instance_id": 0,
                "source_node_id": 0,
                "target_node_id": 0,
                "hit_tokens": 1,
                "bytes": 100,
                "restore_ns": 30,
                "owner_gate_ns": 35,
                "pd_pair_fifo_wait_ns": 2,
                "prepare_boundary_wait_ns": 3,
                "source_demotion_join_wait_ns": 0,
                "restore_issue_time_ns": 15,
                "target_hbm_ready_time_ns": 20,
                "restore_ready_time_ns": 45,
                "hbm_admission_wait_ns": 5,
                "queue_wait_ns": 2,
                "restore_service_ns": 23,
            },
            {
                "time_ns": 20,
                "session_id": "session",
                "event": "migration_reserve",
                "kind": "ssd_to_cpu_stage",
                "start_ns": 20,
                "complete_ns": 28,
                "service_ns": 8,
                "queue_wait_ns": 0,
                "bytes": 100,
                "foreground": True,
                "resources": ["node:0:dram", "ssd-pool:read"],
            },
            {
                "time_ns": 28,
                "session_id": "session",
                "event": "migration_reserve",
                "kind": "cpu_stage_to_hbm",
                "start_ns": 30,
                "complete_ns": 45,
                "service_ns": 15,
                "queue_wait_ns": 2,
                "bytes": 100,
                "foreground": True,
                "resources": [
                    "node:0:dram", "instance:0:pcie-copy:0"],
            },
        ]
    return {
        "schema_version": 20,
        "policy": policy,
        "config": {
            "policy": policy,
            "pressure_policy": "lru-drop",
            "demotion_mode": "capacity-only",
            "swap_execution_mode": "async-pre-admission",
            "active_preemption_mode": "recompute",
            "block_size": 16,
            "queue_recompute_wait_service_ratio": 1.0,
            "queue_recompute_min_wait_ms": 0.0,
            "queue_recompute_cost_guard_multiplier": 0.0,
            "queue_recompute_prefill_headroom_chunks": 1.0,
        },
        "totals": totals,
        "events": events,
        "measurement_cutoff_dma_tail": {
            "foreground_jobs": 0,
            "background_jobs": 0,
        },
        "pd_chunk_accounting": {
            "status": "ok",
            "chunk_admissions": 0,
            "first_chunk_admissions": 0,
            "snapshot_joined_first_chunks": 0,
            "snapshot_feasible_first_chunks": 0,
            "snapshot_feasible_waiting_first_chunks": 0,
            "cancelled_chunk_admissions": 0,
            "checks": {"fixture_reconciled": True},
        },
        "pd_active_prefill_recompute_accounting": {
            "status": "ok",
            "preemptions": 0,
            "discarded_tokens": 0,
            "restored_hit_tokens_discarded": 0,
            "checks": {"fixture_reconciled": True},
        },
        "queue_recompute_policy": {
            "enabled": policy == "tiered_queue_recompute",
            "configured_wait_service_ratio": 1.0,
            "configured_min_wait_ns": 0,
            "configured_cost_guard_multiplier": 0.0,
            "configured_prefill_headroom_chunks": 1.0,
            "headroom_semantics": (
                "causal_unreserved_P_and_D_snapshot_not_reservation"),
            "headroom_owner": "ordinary_atomic_pd_chunk_admission",
            "evaluation_attempts": 0,
            "severe_gate_passes": 0,
            "cost_gate_passes": 0,
            "full_restore_decisions": 0,
            "partial_restore_decisions": 0,
            "zero_restore_decisions": 0,
            "partial_cpu_decisions": 0,
            "partial_ssd_decisions": 0,
            "drop_decisions": 0,
            "cpu_drop_decisions": 0,
            "ssd_drop_decisions": 0,
            "dropped_bytes": 0,
            "avoided_restore_bytes": 0,
            "physical_entry_dropped_bytes": 0,
            "declared_recompute_tokens": 0,
            "policy_avoidable_recompute_tokens": 0,
            "selected_restore_tokens": 0,
            "dropped_suffix_tokens": 0,
            "selected_restore_bytes": 0,
            "dropped_suffix_bytes": 0,
            "modified_full_projected_queue_wait_ns": 0,
            "modified_full_projected_hbm_admission_wait_ns": 0,
            "modified_full_projected_transient_dram_capacity_wait_ns": 0,
            "modified_full_projected_total_wait_ns": 0,
            "modified_full_projected_service_ns": 0,
            "selected_projected_queue_wait_ns": 0,
            "selected_projected_hbm_admission_wait_ns": 0,
            "selected_projected_transient_dram_capacity_wait_ns": 0,
            "selected_projected_total_wait_ns": 0,
            "selected_projected_service_ns": 0,
            "partial_prefix_projected_queue_wait_ns": 0,
            "partial_prefix_projected_hbm_admission_wait_ns": 0,
            "partial_prefix_projected_transient_dram_capacity_wait_ns": 0,
            "partial_prefix_projected_service_ns": 0,
            "selected_estimated_suffix_recompute_comp_ns": 0,
            "accounting_invariants": {
                "passed": True,
                "errors": [],
                "evaluation_events": 0,
                "partial_events": 0,
                "zero_restore_events": 0,
                "block_size_tokens": 16,
                "logical_session_drop_count": 0,
                "headroom_semantics": "causal_snapshot_not_reservation",
            },
            "pending_restore_commitments": 0,
        },
    }


def _selected_queue_recompute_report(
        *, hbm_wait_ns=50, queue_wait_ns=51, service_ns=100,
        transient_dram_capacity_wait_ns=0, cost_multiplier=0.0,
        estimated_recompute_ns=None):
    """Return a schema-19 H=0 selection with all legacy aliases intact."""
    agentic = _canonical_policy_report("tiered_queue_recompute")
    total_wait_ns = hbm_wait_ns + queue_wait_ns
    restore_ns = total_wait_ns + service_ns
    ratio = 1.0
    ratio_threshold_ns = math.ceil(ratio * service_ns)
    threshold_ns = ratio_threshold_ns
    cost_threshold_ns = (
        None if estimated_recompute_ns is None else
        math.ceil(cost_multiplier * estimated_recompute_ns)
    )
    severe_gate_pass = total_wait_ns > threshold_ns
    if not severe_gate_pass:
        raise ValueError("Fixture must select queue-pressure recomputation")
    selected_path_ns = cost_threshold_ns or 0
    if selected_path_ns >= restore_ns:
        raise ValueError("Fixture H=0 path must strictly improve full restore")
    event_time_ns = 1_000
    common = {
        "time_ns": event_time_ns,
        "session_id": "s",
        "source": "cpu",
        "transfer_kinds": ["cpu_to_hbm"],
        "bytes": 1600,
        "reusable_tokens_R": 16,
        "selected_prefix_tokens_H": 0,
        "selected_prefix_block_tokens": 0,
        "dropped_suffix_tokens": 16,
        "selected_restore_bytes": 0,
        "dropped_suffix_bytes": 1600,
        "avoided_restore_bytes": 1600,
        "physical_entry_dropped_bytes": 1600,
        "projection_arrival_ns": event_time_ns + hbm_wait_ns,
        "projection_available": True,
        "projection_available_without_new_lru_work": False,
        "projection_includes_collateral_lru_work": True,
        "projected_hbm_victim_sessions": ["hbm-victim"],
        "projected_cpu_victim_sessions": [],
        "projection_precedes_destination_hbm_reservation": True,
        "projected_hbm_admission_wait_ns": hbm_wait_ns,
        "projected_transient_dram_capacity_wait_ns": (
            transient_dram_capacity_wait_ns),
        "projected_queue_wait_ns": queue_wait_ns,
        "projected_total_wait_ns": total_wait_ns,
        "projected_service_ns": service_ns,
        "projected_restore_ns": restore_ns,
        "estimated_incremental_recompute_comp_ns": (
            estimated_recompute_ns),
        "estimated_suffix_recompute_comp_ns": estimated_recompute_ns,
        "selected_predicted_resume_path_ns": selected_path_ns,
        "full_predicted_resume_path_ns": restore_ns,
        "candidate_prefix_tokens": [16, 0],
        "full_projection_status": "available_with_collateral_lru",
        "prefix_projection_available": False,
        "prefix_projected_hbm_admission_wait_ns": 0,
        "prefix_projected_transient_dram_capacity_wait_ns": 0,
        "prefix_projected_queue_wait_ns": 0,
        "prefix_projected_service_ns": 0,
        "capacity_headroom_snapshot": None,
        "capacity_headroom_snapshot_only": True,
        "capacity_headroom_claimed_by_policy": False,
        "pd_first_chunk_immediate_admission_guaranteed": False,
        "configured_wait_service_ratio": ratio,
        "configured_min_wait_ns": 0,
        "configured_cost_guard_multiplier": cost_multiplier,
        "ratio_threshold_ns": ratio_threshold_ns,
        "threshold_ns": threshold_ns,
        "cost_threshold_ns": cost_threshold_ns,
        "severe_gate_pass": severe_gate_pass,
        "cost_gate_pass": True,
    }
    evaluation = dict(common)
    evaluation.update({
        "event": "queue_recompute_evaluate",
        "decision": "drop_recompute",
    })
    decision = dict(common)
    decision.update({
        "event": "queue_recompute_drop",
        "declared_reuse_tokens": 16,
        "reusable_tokens": 16,
        "policy_avoidable_tokens": 16,
        "object_scope": "kv_cache_entry",
        "selection_scope": "whole_reusable_entry",
        "selection_reason": "zero_restore_lowest_predicted_path",
        "source_pin_scope": "not_applicable",
        "recompute_scope": "whole_reusable_prefix",
        "logical_session_effect": "none",
    })
    agentic["config"][
        "queue_recompute_cost_guard_multiplier"] = cost_multiplier
    agentic["totals"].update({
        "dropped_misses": 1,
        "policy_avoidable_recompute_tokens": 16,
        "hbf_dropped_recompute_tokens": 16,
        "queue_recompute_evaluation_attempts": 1,
        "queue_recompute_severe_gate_passes": 1,
        "queue_recompute_cost_gate_passes": 1,
        "queue_recompute_full_restore_decisions": 0,
        "queue_recompute_partial_restore_decisions": 0,
        "queue_recompute_zero_restore_decisions": 1,
        "queue_recompute_drop_decisions": 1,
        "queue_recompute_cpu_drop_decisions": 1,
        "queue_recompute_dropped_bytes": 1600,
        "queue_recompute_avoided_restore_bytes": 1600,
        "queue_recompute_physical_entry_dropped_bytes": 1600,
        "queue_recompute_projected_queue_wait_ns": queue_wait_ns,
        "queue_recompute_projected_hbm_admission_wait_ns": hbm_wait_ns,
        "queue_recompute_projected_transient_dram_capacity_wait_ns": (
            transient_dram_capacity_wait_ns),
        "queue_recompute_projected_service_ns": service_ns,
        "queue_recompute_estimated_recompute_ns": (
            estimated_recompute_ns or 0),
        "queue_recompute_tokens": 16,
        "queue_recompute_policy_avoidable_tokens": 16,
        "queue_recompute_selected_restore_tokens": 0,
        "queue_recompute_dropped_suffix_tokens": 16,
        "queue_recompute_selected_restore_bytes": 0,
        "queue_recompute_dropped_suffix_bytes": 1600,
    })
    agentic["queue_recompute_policy"].update({
        "configured_cost_guard_multiplier": cost_multiplier,
        "evaluation_attempts": 1,
        "severe_gate_passes": 1,
        "cost_gate_passes": 1,
        "full_restore_decisions": 0,
        "partial_restore_decisions": 0,
        "zero_restore_decisions": 1,
        "drop_decisions": 1,
        "cpu_drop_decisions": 1,
        "dropped_bytes": 1600,
        "avoided_restore_bytes": 1600,
        "physical_entry_dropped_bytes": 1600,
        "declared_recompute_tokens": 16,
        "policy_avoidable_recompute_tokens": 16,
        "selected_restore_tokens": 0,
        "dropped_suffix_tokens": 16,
        "selected_restore_bytes": 0,
        "dropped_suffix_bytes": 1600,
        "modified_full_projected_queue_wait_ns": queue_wait_ns,
        "modified_full_projected_hbm_admission_wait_ns": hbm_wait_ns,
        "modified_full_projected_transient_dram_capacity_wait_ns": (
            transient_dram_capacity_wait_ns),
        "modified_full_projected_total_wait_ns": total_wait_ns,
        "modified_full_projected_service_ns": service_ns,
        "selected_projected_queue_wait_ns": queue_wait_ns,
        "selected_projected_hbm_admission_wait_ns": hbm_wait_ns,
        "selected_projected_transient_dram_capacity_wait_ns": (
            transient_dram_capacity_wait_ns),
        "selected_projected_total_wait_ns": total_wait_ns,
        "selected_projected_service_ns": service_ns,
        "selected_estimated_suffix_recompute_comp_ns": (
            estimated_recompute_ns or 0),
        "accounting_invariants": {
            "passed": True,
            "errors": [],
            "evaluation_events": 1,
            "partial_events": 0,
            "zero_restore_events": 1,
            "block_size_tokens": 16,
            "logical_session_drop_count": 0,
            "headroom_semantics": "causal_snapshot_not_reservation",
        },
    })
    agentic["events"] = [
        evaluation,
        decision,
        {
            "time_ns": event_time_ns,
            "event": "drop",
            "session_id": "s",
            "reason": "queue_pressure",
            "drop_class": "policy_loss",
            "object_scope": "kv_cache_entry",
            "logical_session_effect": "none",
        },
    ]
    return agentic


def _selected_partial_queue_recompute_report(*, actual_wait_ns=7):
    """Return one schema-19 block-prefix choice and its first P/D claim."""
    agentic = _canonical_policy_report("tiered_queue_recompute")
    agentic["config"]["queue_recompute_cost_guard_multiplier"] = 1.0
    decision_time_ns = 1_000
    full_hbm_wait_ns = 50
    full_queue_wait_ns = 51
    full_service_ns = 100
    full_total_wait_ns = full_hbm_wait_ns + full_queue_wait_ns
    full_path_ns = full_total_wait_ns + full_service_ns
    prefix_queue_wait_ns = 10
    prefix_service_ns = 20
    suffix_recompute_ns = 80
    selected_path_ns = (
        prefix_queue_wait_ns + prefix_service_ns + suffix_recompute_ns)
    snapshot = {
        "time_ns": decision_time_ns,
        "prefix_tokens": 16,
        "prefix_block_tokens": 16,
        "next_chunk_tokens": 16,
        "through_next_chunk_block_tokens": 32,
        "prefill_instance_id": 0,
        "prefill_unreserved_per_rank_bytes": 4000,
        "prefill_prefix_per_rank_bytes": 1600,
        "prefill_growth_headroom_per_rank_bytes": 1600,
        "prefill_required_through_chunk_per_rank_bytes": 3200,
        "decode_instance_id": 1,
        "decode_unreserved_per_rank_bytes": 4000,
        "decode_required_through_chunk_per_rank_bytes": 3200,
        "feasible": True,
        "semantics": "causal_snapshot_not_reservation",
    }
    common = {
        "time_ns": decision_time_ns,
        "session_id": "partial-session",
        "source": "cpu",
        "transfer_kinds": ["cpu_to_hbm"],
        "bytes": 3200,
        "reusable_tokens_R": 32,
        "selected_prefix_tokens_H": 16,
        "selected_prefix_block_tokens": 16,
        "dropped_suffix_tokens": 16,
        "selected_restore_bytes": 1600,
        "dropped_suffix_bytes": 1600,
        "avoided_restore_bytes": 1600,
        "physical_entry_dropped_bytes": 0,
        "projection_arrival_ns": decision_time_ns + full_hbm_wait_ns,
        "projection_available": True,
        "projection_available_without_new_lru_work": False,
        "projection_includes_collateral_lru_work": True,
        "projected_hbm_victim_sessions": ["hbm-victim"],
        "projected_cpu_victim_sessions": [],
        "projection_precedes_destination_hbm_reservation": True,
        "projected_hbm_admission_wait_ns": full_hbm_wait_ns,
        "projected_transient_dram_capacity_wait_ns": 0,
        "projected_queue_wait_ns": full_queue_wait_ns,
        "projected_total_wait_ns": full_total_wait_ns,
        "projected_service_ns": full_service_ns,
        "projected_restore_ns": full_path_ns,
        "estimated_incremental_recompute_comp_ns": 160,
        "estimated_suffix_recompute_comp_ns": suffix_recompute_ns,
        "selected_predicted_resume_path_ns": selected_path_ns,
        "full_predicted_resume_path_ns": full_path_ns,
        "candidate_prefix_tokens": [32, 16, 0],
        "full_projection_status": "available_with_collateral_lru",
        "prefix_projection_available": True,
        "prefix_projected_hbm_admission_wait_ns": 0,
        "prefix_projected_transient_dram_capacity_wait_ns": 0,
        "prefix_projected_queue_wait_ns": prefix_queue_wait_ns,
        "prefix_projected_service_ns": prefix_service_ns,
        "capacity_headroom_snapshot": snapshot,
        "capacity_headroom_snapshot_only": True,
        "capacity_headroom_claimed_by_policy": False,
        "pd_first_chunk_immediate_admission_guaranteed": False,
        "configured_wait_service_ratio": 1.0,
        "configured_min_wait_ns": 0,
        "configured_cost_guard_multiplier": 1.0,
        "ratio_threshold_ns": full_service_ns,
        "threshold_ns": full_service_ns,
        "cost_threshold_ns": suffix_recompute_ns,
        "severe_gate_pass": True,
        "cost_gate_pass": True,
    }
    evaluation = {
        **common,
        "event": "queue_recompute_evaluate",
        "decision": "partial_restore_suffix_recompute",
    }
    selection = {
        **common,
        "event": "queue_recompute_partial",
        "physical_source_bytes_pinned_until_dma_complete": 4800,
        "declared_reuse_tokens": 32,
        "reusable_tokens": 32,
        "policy_avoidable_tokens": 16,
        "object_scope": "kv_cache_entry",
        "selection_scope": "contiguous_block_aligned_prefix",
        "selection_reason": "partial_prefix_lowest_predicted_path",
        "source_pin_scope": (
            "full_physical_source_until_prefix_dma_complete"),
        "recompute_scope": "contiguous_suffix_H_to_R",
        "logical_session_effect": "none",
    }
    admitted_ns = decision_time_ns + actual_wait_ns
    chunk = {
        "time_ns": admitted_ns,
        "event": "pd_chunk_admission",
        "admission_scope": "one_prefill_chunk_atomic_pd_claim",
        "admission_semantics": (
            "policy_independent_authoritative_dispatch_claim"),
        "session_id": "partial-session",
        "request_id": 9,
        "prefill_instance_id": 0,
        "decode_instance_id": 1,
        "computed_tokens": 16,
        "chunk_tokens": 16,
        "target_tokens": 32,
        "prefill_current_per_rank_bytes": 1600,
        "decode_current_per_rank_bytes": 0,
        "prefill_target_per_rank_bytes": 3200,
        "decode_target_per_rank_bytes": 3200,
        "prefill_delta_per_rank_bytes": 1600,
        "decode_delta_per_rank_bytes": 3200,
        "prefill_unreserved_per_rank_bytes": 3000,
        "decode_unreserved_per_rank_bytes": 3000,
        "enqueued_ns": decision_time_ns,
        "prefill_capacity_ready_ns": admitted_ns,
        "decode_capacity_ready_ns": admitted_ns,
        "admitted_ns": admitted_ns,
        "wait_ns": actual_wait_ns,
        "critical_wait_after_restore_ns": actual_wait_ns,
        "prefill_delta_bytes": 6400,
        "decode_delta_bytes": 12800,
        "restore_ready_ns": decision_time_ns,
        "chunk_index": 1,
        "first_chunk": True,
        "capacity_headroom_snapshot": snapshot,
        "capacity_headroom_snapshot_only": True,
        "capacity_headroom_claimed_by_policy": False,
        "capacity_snapshot_decision_time_ns": decision_time_ns,
        "capacity_snapshot_to_admission_ns": actual_wait_ns,
        "capacity_snapshot_feasible": True,
        "snapshot_feasible_but_actual_waited": actual_wait_ns > 0,
    }
    agentic["events"] = [evaluation, selection, chunk]
    agentic["pd_chunk_accounting"].update({
        "chunk_admissions": 1,
        "first_chunk_admissions": 1,
        "snapshot_joined_first_chunks": 1,
        "snapshot_feasible_first_chunks": 1,
        "snapshot_feasible_waiting_first_chunks": int(actual_wait_ns > 0),
    })
    agentic["totals"].update({
        "queue_recompute_evaluation_attempts": 1,
        "queue_recompute_severe_gate_passes": 1,
        "queue_recompute_cost_gate_passes": 1,
        "queue_recompute_full_restore_decisions": 0,
        "queue_recompute_partial_restore_decisions": 1,
        "queue_recompute_zero_restore_decisions": 0,
        "queue_recompute_partial_cpu_decisions": 1,
        "queue_recompute_dropped_bytes": 1600,
        "queue_recompute_avoided_restore_bytes": 1600,
        "queue_recompute_projected_queue_wait_ns": full_queue_wait_ns,
        "queue_recompute_projected_hbm_admission_wait_ns": full_hbm_wait_ns,
        "queue_recompute_projected_service_ns": full_service_ns,
        "queue_recompute_prefix_projected_queue_wait_ns": (
            prefix_queue_wait_ns),
        "queue_recompute_prefix_projected_service_ns": prefix_service_ns,
        "queue_recompute_estimated_recompute_ns": suffix_recompute_ns,
        "queue_recompute_tokens": 16,
        "queue_recompute_policy_avoidable_tokens": 16,
        "queue_recompute_selected_restore_tokens": 16,
        "queue_recompute_dropped_suffix_tokens": 16,
        "queue_recompute_selected_restore_bytes": 1600,
        "queue_recompute_dropped_suffix_bytes": 1600,
        "pd_chunk_admissions": 1,
        "pd_chunk_waiting_admissions": int(actual_wait_ns > 0),
        "pd_chunk_admitted_tokens": 16,
        "pd_chunk_prefill_reserved_bytes": 6400,
        "pd_chunk_decode_reserved_bytes": 12800,
        "pd_chunk_admission_wait_ns": actual_wait_ns,
        "pd_chunk_admission_critical_wait_ns": actual_wait_ns,
        "pd_chunk_snapshot_joined_admissions": 1,
        "pd_chunk_snapshot_feasible_admissions": 1,
        "pd_chunk_snapshot_feasible_waiting_admissions": int(
            actual_wait_ns > 0),
        "pd_chunk_snapshot_feasible_wait_ns": actual_wait_ns,
    })
    agentic["queue_recompute_policy"].update({
        "configured_cost_guard_multiplier": 1.0,
        "evaluation_attempts": 1,
        "severe_gate_passes": 1,
        "cost_gate_passes": 1,
        "full_restore_decisions": 0,
        "partial_restore_decisions": 1,
        "zero_restore_decisions": 0,
        "partial_cpu_decisions": 1,
        "partial_ssd_decisions": 0,
        "drop_decisions": 0,
        "selected_restore_tokens": 16,
        "dropped_suffix_tokens": 16,
        "selected_restore_bytes": 1600,
        "dropped_suffix_bytes": 1600,
        "dropped_bytes": 1600,
        "avoided_restore_bytes": 1600,
        "declared_recompute_tokens": 16,
        "policy_avoidable_recompute_tokens": 16,
        "modified_full_projected_queue_wait_ns": full_queue_wait_ns,
        "modified_full_projected_hbm_admission_wait_ns": full_hbm_wait_ns,
        "modified_full_projected_total_wait_ns": full_total_wait_ns,
        "modified_full_projected_service_ns": full_service_ns,
        "selected_projected_queue_wait_ns": full_queue_wait_ns,
        "selected_projected_hbm_admission_wait_ns": full_hbm_wait_ns,
        "selected_projected_total_wait_ns": full_total_wait_ns,
        "selected_projected_service_ns": full_service_ns,
        "partial_prefix_projected_queue_wait_ns": prefix_queue_wait_ns,
        "partial_prefix_projected_hbm_admission_wait_ns": 0,
        "partial_prefix_projected_transient_dram_capacity_wait_ns": 0,
        "partial_prefix_projected_service_ns": prefix_service_ns,
        "selected_estimated_suffix_recompute_comp_ns": suffix_recompute_ns,
        "accounting_invariants": {
            "passed": True,
            "errors": [],
            "evaluation_events": 1,
            "partial_events": 1,
            "zero_restore_events": 0,
            "block_size_tokens": 16,
            "logical_session_drop_count": 0,
            "headroom_semantics": "causal_snapshot_not_reservation",
        },
    })
    return agentic


def _full_restore_queue_recompute_report():
    agentic = _canonical_policy_report("tiered_queue_recompute")
    event_time_ns = 1_000
    event = {
        "time_ns": event_time_ns,
        "event": "queue_recompute_evaluate",
        "decision": "restore",
        "session_id": "full-session",
        "source": "cpu",
        "transfer_kinds": ["cpu_to_hbm"],
        "bytes": 1600,
        "reusable_tokens_R": 16,
        "selected_prefix_tokens_H": 16,
        "selected_prefix_block_tokens": 16,
        "dropped_suffix_tokens": 0,
        "selected_restore_bytes": 1600,
        "dropped_suffix_bytes": 0,
        "avoided_restore_bytes": 0,
        "physical_entry_dropped_bytes": 0,
        "projection_arrival_ns": event_time_ns + 20,
        "projection_available": True,
        "projection_available_without_new_lru_work": True,
        "projection_includes_collateral_lru_work": False,
        "projected_hbm_victim_sessions": [],
        "projected_cpu_victim_sessions": [],
        "projection_precedes_destination_hbm_reservation": True,
        "projected_hbm_admission_wait_ns": 20,
        "projected_transient_dram_capacity_wait_ns": 0,
        "projected_queue_wait_ns": 10,
        "projected_total_wait_ns": 30,
        "projected_service_ns": 100,
        "projected_restore_ns": 130,
        "estimated_incremental_recompute_comp_ns": None,
        "estimated_suffix_recompute_comp_ns": 0,
        "selected_predicted_resume_path_ns": 130,
        "full_predicted_resume_path_ns": 130,
        "candidate_prefix_tokens": [],
        "full_projection_status": "available_without_collateral_lru",
        "prefix_projection_available": True,
        "prefix_projected_hbm_admission_wait_ns": 20,
        "prefix_projected_transient_dram_capacity_wait_ns": 0,
        "prefix_projected_queue_wait_ns": 10,
        "prefix_projected_service_ns": 100,
        "capacity_headroom_snapshot": None,
        "capacity_headroom_snapshot_only": True,
        "capacity_headroom_claimed_by_policy": False,
        "pd_first_chunk_immediate_admission_guaranteed": False,
        "configured_wait_service_ratio": 1.0,
        "configured_min_wait_ns": 0,
        "configured_cost_guard_multiplier": 0.0,
        "ratio_threshold_ns": 100,
        "threshold_ns": 100,
        "cost_threshold_ns": 0,
        "severe_gate_pass": False,
        "cost_gate_pass": False,
    }
    agentic["events"] = [event]
    agentic["totals"].update({
        "queue_recompute_evaluation_attempts": 1,
        "queue_recompute_full_restore_decisions": 1,
    })
    agentic["queue_recompute_policy"].update({
        "evaluation_attempts": 1,
        "full_restore_decisions": 1,
        "accounting_invariants": {
            "passed": True,
            "errors": [],
            "evaluation_events": 1,
            "partial_events": 0,
            "zero_restore_events": 0,
            "block_size_tokens": 16,
            "logical_session_drop_count": 0,
            "headroom_semantics": "causal_snapshot_not_reservation",
        },
    })
    return agentic


class OnlineExperimentTests(unittest.TestCase):
    def test_dataset_path_overrides_preserve_the_immutable_contract(self):
        spec = {
            "dataset": "workloads/generated/trace.jsonl",
            "dataset_contract": {
                "expected_sha256": "a" * 64,
                "expected_manifest_sha256": "b" * 64,
                "manifest": "workloads/generated/trace.manifest.json",
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            effective, overrides = _apply_dataset_path_overrides(
                spec,
                root,
                dataset_path="artifacts/trace.jsonl",
                manifest_path="artifacts/trace.manifest.json",
            )

        self.assertEqual(
            effective["dataset"],
            str((root / "artifacts/trace.jsonl").resolve()),
        )
        self.assertEqual(
            effective["dataset_contract"]["manifest"],
            str((root / "artifacts/trace.manifest.json").resolve()),
        )
        self.assertEqual(
            effective["dataset_contract"]["expected_sha256"], "a" * 64)
        self.assertEqual(
            effective["dataset_contract"]["expected_manifest_sha256"],
            "b" * 64,
        )
        self.assertEqual(
            overrides["dataset"]["declared_path"],
            "workloads/generated/trace.jsonl",
        )
        self.assertEqual(
            spec["dataset_contract"]["manifest"],
            "workloads/generated/trace.manifest.json",
        )

    def test_manifest_override_requires_a_hash_contract(self):
        with self.assertRaisesRegex(
                ExperimentError, "requires a dataset_contract"):
            _apply_dataset_path_overrides(
                {"dataset": "trace.jsonl"},
                Path.cwd(),
                manifest_path="trace.manifest.json",
            )

    def test_cli_accepts_explicit_local_trace_artifacts(self):
        args = build_parser().parse_args([
            "--spec", "paper.json",
            "--mode", "poisson",
            "--dataset-override", "/data/trace.jsonl",
            "--dataset-manifest-override", "/data/trace.manifest.json",
        ])

        self.assertEqual(args.modes, ["poisson"])
        self.assertEqual(args.dataset_override, "/data/trace.jsonl")
        self.assertEqual(
            args.dataset_manifest_override,
            "/data/trace.manifest.json",
        )

    def test_dataset_contract_pins_manifest_source_and_selected_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            rows = [
                _session("first", 1_000, "tool"),
                _session("second", 2_000, "human"),
            ]
            source.write_text("".join(
                json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            identity_payload = [
                {"source_index": 0, "session_id": "first"},
                {"source_index": 1, "session_id": "second"},
            ]
            identity_hash = hashlib.sha256(json.dumps(
                identity_payload, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")).hexdigest()
            manifest = directory / "source.manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": 3,
                "source": {
                    "sha256": "1" * 64,
                    "revision": "immutable-revision",
                    "tracelab_reuse_mode": "eligible",
                },
                "summary": {"sessions_emitted": 2},
                "output": {"sha256": source_sha256},
            }), encoding="utf-8")
            manifest_sha256 = hashlib.sha256(
                manifest.read_bytes()).hexdigest()

            cohort = materialize_session_cohort(
                source,
                directory / "result",
                dataset_contract={
                    "expected_sha256": source_sha256,
                    "expected_source_session_count": 2,
                    "expected_schema_version": 3,
                    "manifest": "source.manifest.json",
                    "expected_manifest_sha256": manifest_sha256,
                    "source_sha256": "1" * 64,
                    "source_revision": "immutable-revision",
                    "tracelab_reuse_mode": "eligible",
                    "expected_selected_template_count": 2,
                    "expected_selected_request_count": 4,
                    "expected_selected_session_identity_hash": identity_hash,
                },
                repo_root=directory,
            )

            validation = cohort["dataset_contract_validation"]
            self.assertTrue(validation["passed"])
            self.assertEqual(
                validation["schema_verified_by"], "companion_manifest")
            self.assertEqual(validation["parsed_source_session_count"], 2)
            self.assertEqual(
                cohort["selected_session_identity_hash"], identity_hash)

    def test_dataset_contract_source_hash_fails_before_json_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "invalid.jsonl"
            source.write_text("not-json\n", encoding="utf-8")
            with self.assertRaisesRegex(
                    ExperimentError, "expected_sha256 mismatch"):
                materialize_session_cohort(
                    source,
                    directory / "result",
                    dataset_contract={"expected_sha256": "0" * 64},
                    repo_root=directory,
                )

    def test_dataset_contract_row_schema_fallback_and_selection_fail_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            row = _session("only", 1_000, "tool")
            row["schema_version"] = 3
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

            cohort = materialize_session_cohort(
                source,
                directory / "valid",
                dataset_contract={
                    "expected_sha256": source_sha256,
                    "expected_source_session_count": 1,
                    "expected_schema_version": 3,
                    "expected_selected_template_count": 1,
                    "expected_selected_request_count": 2,
                },
                repo_root=directory,
            )
            self.assertEqual(
                cohort["dataset_contract_validation"]["schema_verified_by"],
                "converted_rows",
            )

            with self.assertRaisesRegex(
                    ExperimentError, "expected_selected_request_count"):
                materialize_session_cohort(
                    source,
                    directory / "invalid-selection",
                    dataset_contract={
                        "expected_sha256": source_sha256,
                        "expected_source_session_count": 1,
                        "expected_schema_version": 3,
                        "expected_selected_request_count": 3,
                    },
                    repo_root=directory,
                )

    def test_dataset_contract_rejects_manifest_count_and_hash_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            source.write_text(
                json.dumps(_session("only", 1_000, "tool")) + "\n",
                encoding="utf-8",
            )
            source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = directory / "source.manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": 3,
                "source": {},
                "summary": {"sessions_emitted": 2},
                "output": {"sha256": source_sha256},
            }), encoding="utf-8")
            manifest_sha256 = hashlib.sha256(
                manifest.read_bytes()).hexdigest()

            with self.assertRaisesRegex(
                    ExperimentError, "session count conflicts"):
                materialize_session_cohort(
                    source,
                    directory / "bad-count",
                    dataset_contract={
                        "expected_sha256": source_sha256,
                        "expected_source_session_count": 1,
                        "expected_schema_version": 3,
                        "manifest": "source.manifest.json",
                        "expected_manifest_sha256": manifest_sha256,
                    },
                    repo_root=directory,
                )

            with self.assertRaisesRegex(
                    ExperimentError, "expected_manifest_sha256 mismatch"):
                materialize_session_cohort(
                    source,
                    directory / "bad-manifest-hash",
                    dataset_contract={
                        "expected_sha256": source_sha256,
                        "manifest": "source.manifest.json",
                        "expected_manifest_sha256": "0" * 64,
                    },
                    repo_root=directory,
                )

    def test_checked_in_pressure_pilot_keeps_balanced_fixed_cohort_pressure(self):
        repo_root = Path(__file__).resolve().parents[1]
        spec = json.loads((
            repo_root
            / "configs/experiments/online_tracelab_qwen3_1m_p4d4_pressure_pilot.json"
        ).read_text(encoding="utf-8"))
        selection = spec["workload_selection"]
        mode = spec["modes"]["backlog"]
        contract = spec["dataset_contract"]

        source_indices = selection["include_source_indices"]
        self.assertEqual(source_indices, [2113, 3726])
        self.assertEqual(selection["max_sessions"], 2)
        self.assertEqual(selection["target_max_sequence_tokens"], 1_000_000)
        self.assertEqual(contract["expected_selected_template_count"], 2)
        self.assertEqual(contract["expected_selected_request_count"], 4)
        identity_payload = [
            {
                "source_index": 2113,
                "session_id": (
                    "claude:b707f5ac-76ab-9a97-ac26-c45554e41a7d"),
            },
            {
                "source_index": 3726,
                "session_id": (
                    "claude:8c59de67-05f9-ae3d-0a63-d8e34988e84c"),
            },
        ]
        identity_hash = hashlib.sha256(json.dumps(
            identity_payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        self.assertEqual(
            contract["expected_selected_session_identity_hash"],
            identity_hash,
        )

        self.assertEqual(mode["k_values"], [12])
        self.assertEqual(mode["backlog_epochs"], 8)
        self.assertEqual(mode["measure_completions"], 4)
        epoch_major = source_indices * mode["backlog_epochs"]
        measured = epoch_major[:mode["measure_completions"]]
        self.assertEqual(measured, [2113, 3726, 2113, 3726])
        self.assertEqual(
            mode["measure_completions"]
            * contract["expected_selected_request_count"]
            // contract["expected_selected_template_count"],
            8,
        )
        gap_type = {2113: "tool", 3726: "human"}
        self.assertEqual(
            [gap_type[index] for index in measured].count("tool"), 2)
        self.assertEqual(
            [gap_type[index] for index in measured].count("human"), 2)

        # These dormant lengths are fixed by the content-addressed TraceLab
        # source and the globally pinned 1M-token transform.
        dormant_tokens = {2113: 426_734, 3726: 999_375}
        block_size = 16
        model = json.loads((
            repo_root
            / "configs/model/Qwen/Qwen3-30B-A3B-Instruct-2507.json"
        ).read_text(encoding="utf-8"))
        kv_bytes_per_token = (
            model["num_hidden_layers"]
            * 2
            * model["num_key_value_heads"]
            * model["head_dim"]
            * 2
        )
        initial_active = epoch_major[:mode["k_values"][0]]
        pressure_bytes = sum(
            ((dormant_tokens[index] + block_size - 1) // block_size)
            * block_size
            * kv_bytes_per_token
            for index in initial_active
        )
        previous_pressure_bytes = 828_929_212_416
        self.assertEqual(pressure_bytes, 841_155_084_288)
        self.assertLess(
            abs(pressure_bytes - previous_pressure_bytes)
            / previous_pressure_bytes,
            0.015,
        )
        measured_output_tokens = sum(
            {2113: 4, 3726: 31}[index] for index in measured)
        self.assertEqual(measured_output_tokens, 70)

    def test_grouped_plot_is_dependency_free_svg(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                {
                    "mode": "backlog",
                    "load_value": load,
                    "policy": policy,
                    "policy_order": order,
                    "sessions_per_second": value,
                }
                for load, policy, order, value in (
                    (1.0, "tiered<&", 0, 1.0),
                    (1.0, "oracle", 1, 1.5),
                    (2.0, "tiered<&", 0, 1.2),
                    (2.0, "oracle", 1, 1.8),
                )
            ]
            paths = plot_grouped_throughput(rows, directory)
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].suffix, ".svg")
            payload = paths[0].read_text(encoding="utf-8")
            self.assertIn("<svg", payload)
            self.assertIn("tiered&lt;&amp;", payload)
            self.assertNotIn("tiered<&", payload)

    def test_poisson_jct_plot_uses_seed_means_and_student_t_ci(self):
        rows = []
        policies = ("tiered<&", "oracle")
        values = {
            (0.1, "tiered<&"): (1.0, 2.0, 3.0),
            (0.1, "oracle"): (0.5, 1.0, 1.5),
            (0.2, "tiered<&"): (4.0, 5.0, 6.0),
            (0.2, "oracle"): (2.0, 2.5, 3.0),
        }
        for rate in (0.1, 0.2):
            for seed_index, seed in enumerate((41, 42, 43)):
                pair_key = f"poisson:{rate}:seed={seed}"
                for order, policy in enumerate(policies):
                    rows.append({
                        "mode": "poisson",
                        "load_value": rate,
                        "policy": policy,
                        "policy_order": order,
                        "pair_key": pair_key,
                        "arrival_seed": seed,
                        "session_jct_mean_ns": (
                            values[(rate, policy)][seed_index] * 1e9),
                    })

        with tempfile.TemporaryDirectory() as directory:
            plot_paths, source_path = plot_poisson_session_jct(
                rows, directory, oracle_label="oracle")
            self.assertTrue(source_path.is_file())
            self.assertIn(
                Path(directory) / "poisson_session_jct_grouped.svg",
                plot_paths,
            )
            self.assertTrue(all(path.is_file() for path in plot_paths))
            payload = (
                Path(directory) / "poisson_session_jct_grouped.svg"
            ).read_text(encoding="utf-8")
            self.assertIn("lower is better", payload)
            self.assertIn("tiered&lt;&amp;", payload)
            self.assertNotIn("tiered<&", payload)

            with open(source_path, newline="", encoding="utf-8") as source:
                source_rows = list(csv.DictReader(source))
            self.assertEqual(
                [
                    (float(row["offered_rate_sessions_per_second"]),
                     row["policy"])
                    for row in source_rows
                ],
                [
                    (0.1, "tiered<&"), (0.1, "oracle"),
                    (0.2, "tiered<&"), (0.2, "oracle"),
                ],
            )
            tiered = source_rows[0]
            self.assertEqual(int(tiered["seed_count"]), 3)
            self.assertEqual(
                json.loads(tiered["arrival_seeds_json"]), [41, 42, 43])
            self.assertEqual(
                json.loads(tiered[
                    "seed_level_mean_session_jct_seconds_json"]),
                [1.0, 2.0, 3.0],
            )
            self.assertAlmostEqual(
                float(tiered["mean_session_jct_seconds"]), 2.0)
            self.assertAlmostEqual(
                float(tiered["sample_stddev_seconds"]), 1.0)
            self.assertAlmostEqual(
                float(tiered["ci95_half_width_seconds"]),
                4.30265272975 / math.sqrt(3),
            )
            self.assertEqual(
                tiered["aggregation_unit"], "arrival_seed_level_mean")

    def test_poisson_jct_plot_rejects_unpaired_seed_grid(self):
        rows = [
            {
                "mode": "poisson",
                "load_value": 0.1,
                "policy": policy,
                "policy_order": order,
                "pair_key": f"poisson:0.1:seed={seed}",
                "arrival_seed": seed,
                "session_jct_mean_ns": 1e9,
            }
            for policy, order, seed in (
                ("tiered", 0, 41),
                ("oracle", 1, 42),
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                    ExperimentError, "unpaired seeds"):
                plot_poisson_session_jct(
                    rows, directory, oracle_label="oracle")

    def test_exact_rate_metrics_use_explicit_eligibility_and_server_gaps(self):
        chunk_waits = (10, 20, 30, 40)
        chunk_critical_waits = (1, 10, 15, 20)
        request_records = [
            {
                "request_id": request_id,
                "session_id": session_id,
                "sub_request_index": sub_index,
                "generated_tokens": generated,
                "ttft_ns": ttft,
                "tpot_ns": tpot,
                "return_gap_ns": gap,
                "agentic_kv_hbm_admission_wait_ns": restore_wait,
                "agentic_kv_hit_tokens": (
                    20 if request_id == 1 else 30 if request_id == 3 else 0),
                "agentic_kv_recompute_tokens": (
                    5 if request_id == 1 else 7 if request_id == 3 else 0),
                "agentic_kv_source": (
                    "cpu" if request_id == 1 else
                    "ssd" if request_id == 3 else None),
                "return_gap_type": (
                    "human" if request_id == 3 else
                    "tool" if sub_index > 0 else "session_start"),
                "pd_chunk_admission_count": 1,
                "pd_chunk_cancelled_admission_count": (
                    1 if request_id == 1 else 0),
                "pd_chunk_admission_wait_ns_total": chunk_waits[request_id],
                "pd_chunk_admission_critical_wait_ns_total": (
                    chunk_critical_waits[request_id]),
                "pd_chunk_successful_admission_wait_ns_total": (
                    15 if request_id == 1 else chunk_waits[request_id]),
                "pd_chunk_successful_admission_critical_wait_ns_total": (
                    7 if request_id == 1
                    else chunk_critical_waits[request_id]),
                "pd_chunk_cancelled_admission_wait_ns_total": (
                    5 if request_id == 1 else 0),
                "pd_chunk_cancelled_admission_critical_wait_ns_total": (
                    3 if request_id == 1 else 0),
                "active_prefill_recompute_preemptions": (
                    1 if request_id == 1 else 0),
                "active_prefill_recompute_tokens": (
                    25 if request_id == 1 else 0),
                "active_prefill_recompute_frontier_tokens": (
                    25 if request_id == 1 else 0),
                "pd_active_prefill_recompute_generation": (
                    1 if request_id == 1 else 0),
                "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute": (
                    20 if request_id == 1 else 0),
            }
            for request_id, session_id, sub_index, generated, ttft, tpot,
            gap, restore_wait in (
                (0, "s1", 0, 2, 10, 5, 0, 1),
                (1, "s1", 1, 3, 30, 10, 100, 2),
                (2, "s2", 0, 1, 20, 0, 0, 3),
                (3, "s2", 1, 2, 50, 20, 200, 4),
            )
        ]
        report = {
            "requests": {
                "records": request_records,
                "resume": {
                    "count": 2,
                    "ttft_ns": _distribution(40, 2),
                },
                "attempted_physical_resume_count": 2,
                "effective_surviving_resume_count": 1,
                "kv_state_unavailable_resume_count": 0,
                "zero_overlap_resume_count": 0,
                "attempted_physical_resume_counts_by_source": {
                    "hbm": 0, "cpu": 1, "ssd": 1},
                "effective_surviving_resume_counts_by_source": {
                    "hbm": 0, "cpu": 0, "ssd": 1},
                "attempted_physical_resume_fractions_of_all_requests": {
                    "hbm": 0.0, "cpu": 0.25, "ssd": 0.25},
                "effective_surviving_resume_fractions_of_all_requests": {
                    "hbm": 0.0, "cpu": 0.0, "ssd": 0.25},
                "attempted_physical_resume_by_return_gap_type_and_source": {
                    "human": {"ssd": {"count": 1}},
                    "tool": {"cpu": {"count": 1}},
                },
                "effective_surviving_resume_by_return_gap_type_and_source": {
                    "human": {"ssd": {"count": 1}},
                },
                "resume_reuse_token_accounting": {
                    "attempted_restored_hit_tokens": 50,
                    "restored_hit_tokens_discarded_by_active_prefill_recompute": 20,
                    "effective_surviving_hit_tokens": 30,
                    "conservation_passed": True,
                },
            },
            "sessions": {
                "records": [
                    {
                        "session_id": "s1",
                        "status": "completed",
                        "measurement_included": True,
                        "offered_time_ns": 0,
                        "completion_time_ns": 1000,
                    },
                    {
                        "session_id": "s2",
                        "status": "completed",
                        "measurement_included": True,
                        "offered_time_ns": 10,
                        "completion_time_ns": 1210,
                    },
                ],
            },
            "throughput": {
                "completed_requests_total": 4,
                "completed_sessions_total": 2,
            },
            "session_admission": {"cutoff_disposition": "drain"},
        }
        manager_events = [
            {
                "event": "pd_chunk_admission",
                "request_id": record["request_id"],
                "session_id": record["session_id"],
                "wait_ns": record[
                    "pd_chunk_successful_admission_wait_ns_total"],
                "critical_wait_after_restore_ns": record[
                    "pd_chunk_successful_admission_critical_wait_ns_total"],
            }
            for record in request_records
        ]
        manager_events.extend([
            {
                "event": (
                    "pd_chunk_admission_cancelled_for_active_prefill_"
                    "recompute"),
                "request_id": 1,
                "session_id": "s1",
                "wait_ns": 5,
                "critical_wait_after_restore_ns": 3,
            },
            {
                "event": "pd_active_prefill_recompute_preempt",
                "request_id": 1,
                "session_id": "s1",
                "discarded_tokens": 25,
                "restored_hit_tokens_discarded": 20,
                "cumulative_active_prefill_recompute_tokens": 25,
                "cumulative_restored_hit_tokens_discarded": 20,
            },
            {
                "event": "resume",
                "request_id": 1,
                "session_id": "s1",
                "sub_request_index": 1,
                "source": "cpu",
                "hit_tokens": 20,
                "recompute_tokens": 5,
                "hbm_admission_wait_ns": 2,
                "return_gap_type": "tool",
            },
            {
                "event": "resume",
                "request_id": 3,
                "session_id": "s2",
                "sub_request_index": 1,
                "source": "ssd",
                "hit_tokens": 30,
                "recompute_tokens": 7,
                "hbm_admission_wait_ns": 4,
                "return_gap_type": "human",
            },
        ])
        agentic_report = {
            "schema_version": 20,
            "events": manager_events,
            "totals": {
                "hbm_hits": 0,
                "cpu_hits": 1,
                "ssd_hits": 1,
                "cache_hit_tokens": 50,
                "recompute_tokens": 12,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            request_csv = Path(directory) / "requests.csv"
            with open(request_csv, "w", newline="", encoding="utf-8") as out:
                writer = csv.DictWriter(out, fieldnames=[
                    "request id", "session_id",
                    "agentic_kv_hit_tokens", "agentic_kv_recompute_tokens",
                    "agentic_kv_source",
                    "return_gap_type",
                    "agentic_kv_hbm_admission_wait_ns",
                    "pd_chunk_admission_count",
                    "pd_chunk_cancelled_admission_count",
                    "pd_chunk_admission_wait_ns_total",
                    "pd_chunk_admission_critical_wait_ns_total",
                    "pd_chunk_successful_admission_wait_ns_total",
                    "pd_chunk_successful_admission_critical_wait_ns_total",
                    "pd_chunk_cancelled_admission_wait_ns_total",
                    "pd_chunk_cancelled_admission_critical_wait_ns_total",
                    "active_prefill_recompute_preemptions",
                    "active_prefill_recompute_tokens",
                    "active_prefill_recompute_frontier_tokens",
                    "pd_active_prefill_recompute_generation",
                    "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute",
                ])
                writer.writeheader()
                for record in request_records:
                    row = dict(record)
                    row["request id"] = row.pop("request_id")
                    writer.writerow({
                        field: row.get(field)
                        for field in writer.fieldnames
                    })
            metrics = _derive_exact_rate_metrics(
                report,
                agentic_report,
                {
                    "schema_version": SCHEMA_VERSION,
                    "stop_after_measurement": False,
                    "run_id": "exact-metrics",
                    "request_csv": str(request_csv),
                },
            )
            self.assertTrue(
                metrics["cross_layer_request_accounting"][
                    "full_completed_cohort"])
            unknown_json_record = dict(request_records[0])
            unknown_json_record.update({
                "request_id": 99,
                "session_id": "unknown-json-session",
            })
            request_records.append(unknown_json_record)
            with self.assertRaisesRegex(
                    ExperimentError, "Full-cohort session JSON"):
                _derive_exact_rate_metrics(
                    report,
                    agentic_report,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "stop_after_measurement": False,
                        "run_id": "exact-metrics-full-json-corrupt",
                        "request_csv": str(request_csv),
                    },
                )
            request_records.pop()

            with open(request_csv, newline="", encoding="utf-8") as source:
                csv_reader = csv.DictReader(source)
                fieldnames = list(csv_reader.fieldnames)
                clean_csv_records = list(csv_reader)

            def write_csv_records(rows):
                with open(
                        request_csv, "w", newline="", encoding="utf-8") as out:
                    writer = csv.DictWriter(out, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

            unknown_csv_record = dict(clean_csv_records[0])
            unknown_csv_record.update({
                "request id": "99",
                "session_id": "unknown-csv-session",
            })
            write_csv_records(clean_csv_records + [unknown_csv_record])
            with self.assertRaisesRegex(
                    ExperimentError, "Full-cohort request CSV"):
                _derive_exact_rate_metrics(
                    report,
                    agentic_report,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "stop_after_measurement": False,
                        "run_id": "exact-metrics-full-csv-corrupt",
                        "request_csv": str(request_csv),
                    },
                )
            write_csv_records(clean_csv_records)

            fake_unmeasured_event = {
                "event": "pd_chunk_admission",
                "request_id": 99,
                "session_id": "warmup-only",
                "wait_ns": 0,
                "critical_wait_after_restore_ns": 0,
            }
            manager_events.append(fake_unmeasured_event)
            with self.assertRaisesRegex(
                    ExperimentError, "Full-cohort manager event"):
                _derive_exact_rate_metrics(
                    report,
                    agentic_report,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "stop_after_measurement": False,
                        "run_id": "exact-metrics-full-cohort-corrupt",
                        "request_csv": str(request_csv),
                    },
                )
            generic_report = json.loads(json.dumps(report))
            generic_report["sessions"]["records"].append({
                "session_id": "warmup-only",
                "status": "censored",
                "measurement_included": False,
                "offered_time_ns": 0,
                "completion_time_ns": 500,
            })
            generic_report["session_admission"][
                "cutoff_disposition"] = "right_censor"
            generic_metrics = _derive_exact_rate_metrics(
                generic_report,
                agentic_report,
                {
                    "schema_version": SCHEMA_VERSION,
                    "stop_after_measurement": True,
                    "run_id": "exact-metrics-generic-subset",
                    "request_csv": str(request_csv),
                },
            )
            self.assertFalse(
                generic_metrics["cross_layer_request_accounting"][
                    "full_completed_cohort"])
            self.assertEqual(
                generic_metrics["cross_layer_request_accounting"][
                    "unmatched_scoped_event_count"],
                1,
            )
            manager_events.pop()
            manager_events[-1]["hit_tokens"] = 29
            with self.assertRaisesRegex(
                    ExperimentError, "Resume event/request mismatch"):
                _derive_exact_rate_metrics(
                    report,
                    agentic_report,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": "exact-metrics-corrupt",
                        "request_csv": str(request_csv),
                    },
                )
            manager_events[-1]["hit_tokens"] = 30
            active_event = next(
                event for event in manager_events
                if event["event"] == "pd_active_prefill_recompute_preempt"
            )
            active_event["discarded_tokens"] = 24
            with self.assertRaisesRegex(
                    ExperimentError, "Active-prefill event/request mismatch"):
                _derive_exact_rate_metrics(
                    report,
                    agentic_report,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": "exact-metrics-active-corrupt",
                        "request_csv": str(request_csv),
                    },
                )
            active_event["discarded_tokens"] = 25
            request_one = next(
                record for record in request_records
                if record["request_id"] == 1
            )
            csv_request_one = next(
                row for row in clean_csv_records
                if row["request id"] == "1"
            )
            request_one["active_prefill_recompute_frontier_tokens"] = 24
            csv_request_one[
                "active_prefill_recompute_frontier_tokens"] = "24"
            write_csv_records(clean_csv_records)
            with self.assertRaisesRegex(
                    ExperimentError, "Active-prefill event/request mismatch"):
                _derive_exact_rate_metrics(
                    report,
                    agentic_report,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": "exact-metrics-frontier-corrupt",
                        "request_csv": str(request_csv),
                    },
                )
            request_one["active_prefill_recompute_frontier_tokens"] = 25
            csv_request_one[
                "active_prefill_recompute_frontier_tokens"] = "25"
            write_csv_records(clean_csv_records)

            request_one["agentic_kv_hbm_admission_wait_ns"] = 3
            csv_request_one["agentic_kv_hbm_admission_wait_ns"] = "3"
            write_csv_records(clean_csv_records)
            with self.assertRaisesRegex(
                    ExperimentError, "Resume event/request mismatch"):
                _derive_exact_rate_metrics(
                    report,
                    agentic_report,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": "exact-metrics-hbm-wait-corrupt",
                        "request_csv": str(request_csv),
                    },
                )
            request_one["agentic_kv_hbm_admission_wait_ns"] = 2
            csv_request_one["agentic_kv_hbm_admission_wait_ns"] = "2"
            write_csv_records(clean_csv_records)

            chunk_event = next(
                event for event in manager_events
                if (event["event"] == "pd_chunk_admission"
                    and event["request_id"] == 1)
            )
            chunk_event["wait_ns"] = 14
            with self.assertRaisesRegex(
                    ExperimentError, "P/D chunk event/request mismatch"):
                _derive_exact_rate_metrics(
                    report,
                    agentic_report,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": "exact-metrics-chunk-corrupt",
                        "request_csv": str(request_csv),
                    },
                )
            chunk_event["wait_ns"] = 15
            agentic_report["totals"]["cache_hit_tokens"] = 49
            with self.assertRaisesRegex(
                    ExperimentError, "Resume event/manager aggregate"):
                _derive_exact_rate_metrics(
                    report,
                    agentic_report,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": "exact-metrics-manager-corrupt",
                        "request_csv": str(request_csv),
                    },
                )
            agentic_report["totals"]["cache_hit_tokens"] = 50
            request_records[0]["active_prefill_recompute_tokens"] = True
            with self.assertRaisesRegex(
                    ExperimentError, "invalid exact integer fields"):
                _derive_exact_rate_metrics(
                    report,
                    agentic_report,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": "exact-metrics-type-corrupt",
                        "request_csv": str(request_csv),
                    },
                )
            request_records[0]["active_prefill_recompute_tokens"] = 0
            with open(request_csv, newline="", encoding="utf-8") as source:
                csv_reader = csv.DictReader(source)
                fieldnames = list(csv_reader.fieldnames)
                csv_records = list(csv_reader)
            next(
                row for row in csv_records if row["request id"] == "3"
            )["session_id"] = "s1"
            write_csv_records(csv_records)
            with self.assertRaisesRegex(
                    ExperimentError, "CSV/JSON session mismatch"):
                _derive_exact_rate_metrics(
                    report,
                    agentic_report,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": "exact-metrics-session-corrupt",
                        "request_csv": str(request_csv),
                    },
                )

        self.assertEqual(metrics["resume_ttft"]["count"], 2)
        self.assertEqual(metrics["resume_ttft"]["mean"], 40)
        self.assertAlmostEqual(metrics["resume_ttft"]["p95"], 49)
        self.assertEqual(metrics["tpot"]["count"], 3)
        self.assertAlmostEqual(metrics["tpot"]["mean"], 35 / 3)
        self.assertAlmostEqual(metrics["tpot"]["p95"], 19)
        self.assertEqual(metrics["server_added_jct"]["count"], 2)
        self.assertEqual(metrics["server_added_jct"]["mean"], 950)
        self.assertEqual(metrics["trace_idle_gaps"]["sum"], 300)
        self.assertEqual(metrics["restore_hbm_admission"]["mean"], 2.5)
        self.assertEqual(metrics["pd_chunk_attempt_admission"]["mean"], 25)
        self.assertEqual(metrics["pd_chunk_hbm_admission"]["mean"], 11.5)
        self.assertEqual(
            metrics["pd_chunk_successful_admission"]["sum"], 95)
        self.assertEqual(
            metrics["pd_chunk_cancelled_admission"]["sum"], 5)
        self.assertEqual(metrics["hbm_admission"]["mean"], 14)
        self.assertAlmostEqual(metrics["hbm_admission"]["p95"], 23.1)
        self.assertIn("excludes_pd_pair_fifo", metrics["hbm_admission_scope"])
        source = metrics["resume_source_accounting"]
        self.assertEqual(source["attempted_counts_by_source"]["cpu"], 1)
        self.assertEqual(
            source["effective_surviving_counts_by_source"]["cpu"], 0)
        self.assertEqual(source["attempted_restored_hit_tokens"], 50)
        self.assertEqual(source["effective_surviving_hit_tokens"], 30)

    def test_poisson_rate_metric_plots_use_seed_level_ci_and_declared_slo(self):
        rows = []
        for rate in (0.1, 0.2):
            for seed in (41, 42, 43):
                for order, policy in enumerate(("tiered<&", "oracle")):
                    base = rate * 100 + seed + order
                    rows.append({
                        "mode": "poisson",
                        "online_artifact_schema_version": SCHEMA_VERSION,
                        "session_report_schema_version": 11,
                        "operational_metric_source_status": (
                            "schema11_exact_measurement_window"),
                        "load_value": rate,
                        "policy": policy,
                        "policy_order": order,
                        "pair_key": f"poisson:{rate}:seed={seed}",
                        "arrival_seed": seed,
                        "resume_ttft_exact_mean_ns": base * 1e6,
                        "resume_ttft_p95_ns": base * 2e6,
                        "resume_ttft_denominator": "resume calls",
                        "tpot_exact_mean_ns": base * 1e5,
                        "tpot_p95_ns": base * 2e5,
                        "tpot_denominator": "calls with >=2 tokens",
                        "server_added_session_jct_mean_ns": base * 1e7,
                        "server_added_session_jct_p95_ns": base * 2e7,
                        "server_added_session_jct_denominator": "sessions",
                        "total_hbm_capacity_admission_wait_mean_ns": base,
                        "total_hbm_capacity_admission_wait_p95_ns": base * 2,
                        "restore_hbm_capacity_admission_wait_mean_ns": base,
                        "restore_hbm_capacity_admission_wait_p95_ns": base * 2,
                        "pd_chunk_hbm_capacity_admission_wait_mean_ns": base,
                        "pd_chunk_hbm_capacity_admission_wait_p95_ns": base * 2,
                        "total_hbm_capacity_admission_wait_scope": (
                            "request critical capacity gates"),
                        "average_active_batch_size": 3 + order,
                        "active_batch_size_scope": "non-dummy batches",
                        "hbm_kv_average_physical_idle_reusable_fraction": 0.2,
                        "hbm_kv_average_physical_non_idle_active_fraction": 0.6,
                        "hbm_kv_average_physical_free_fraction": 0.2,
                        "hbm_kv_average_logical_destination_reservation_fraction": 0.1,
                        "hbm_kv_average_reserved_free_slack_fraction": 0.05,
                        "hbm_kv_average_future_reclaim_backed_reservation_fraction": 0.05,
                        "hbm_kv_average_unclaimed_allocatable_slack_fraction": 0.15,
                        "hbm_kv_average_reservation_adjusted_claim_fraction": 0.85,
                        "hbm_kv_occupancy_scope": (
                            "physical stack; reservation overlay"),
                    })
        slo_settings = {
            "resume_ttft_slo": {
                "threshold_ms": 500,
                "basis": "5x zero-load reference p95",
                "provenance": {"artifact_sha256": "a" * 64},
            },
            "tpot_slo": {
                "threshold_ms": 100,
                "basis": "explicit raw TPOT SLO",
                "provenance": {"citation": "fixture"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            paths, source_path = plot_poisson_rate_metrics(
                rows,
                directory,
                reference_label="oracle",
                slo_settings=slo_settings,
            )
            self.assertEqual(len(paths), 8)
            self.assertTrue(all(path.is_file() for path in paths))
            resume_svg = (
                Path(directory) / "poisson_resume_ttft_by_rate.svg"
            ).read_text(encoding="utf-8")
            self.assertIn('class="slo"', resume_svg)
            self.assertIn("declared SLO", resume_svg)
            self.assertIn("tiered&lt;&amp;", resume_svg)
            with open(source_path, newline="", encoding="utf-8") as source:
                source_rows = list(csv.DictReader(source))
            resume_rows = [
                row for row in source_rows if row["metric"] == "resume_ttft"
            ]
            self.assertEqual(len(resume_rows), 8)
            self.assertEqual(
                json.loads(resume_rows[0]["arrival_seeds_json"]),
                [41, 42, 43],
            )
            self.assertEqual(
                resume_rows[0]["slo_basis"],
                "5x zero-load reference p95",
            )
            self.assertEqual(
                json.loads(resume_rows[0]["slo_provenance_json"]),
                {"artifact_sha256": "a" * 64},
            )
            self.assertEqual(
                resume_rows[0]["aggregation_unit"],
                "arrival_seed_level_run_statistic",
            )
            occupancy_svg = (
                Path(directory)
                / "poisson_hbm_kv_occupancy_breakdown_by_rate.svg"
            ).read_text(encoding="utf-8")
            self.assertIn("non-additive reservation-adjusted claim", occupancy_svg)
            occupancy_rows = [
                row for row in source_rows
                if row["metric"] == "hbm_kv_occupancy_breakdown"
            ]
            self.assertEqual(len(occupancy_rows), 32)
            self.assertIn(
                "non_additive_logical_overlay",
                {row["series_semantics"] for row in occupancy_rows},
            )

        legacy_rows = json.loads(json.dumps(rows))
        for row in legacy_rows:
            row["online_artifact_schema_version"] = 10
            row["session_report_schema_version"] = 9
            row["operational_metric_source_status"] = (
                "unavailable_legacy_session_report")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                    ExperimentError, "legacy/incompatible"):
                plot_poisson_rate_metrics(
                    legacy_rows,
                    directory,
                    reference_label="oracle",
                )

    def test_schema10_operational_sources_fail_closed_and_reconcile(self):
        capacity = 100
        averages = {
            "physical_idle_reusable": 20,
            "physical_non_idle_active": 60,
            "physical_free": 20,
            "logical_destination_admission_reservation": 10,
            "reserved_free_slack": 5,
            "future_reclaim_backed_reservation": 5,
            "unclaimed_allocatable_slack": 15,
        }
        categories = {
            name: {
                "byte_ns": value * 1000,
                "average_per_rank_bytes": value,
                "peak_per_rank_bytes": value,
                "average_fraction_of_capacity": value / capacity,
                "peak_fraction_of_capacity": value / capacity,
            }
            for name, value in averages.items()
        }
        batch = {
            "completed_batch_count": 4,
            "non_dummy_completed_batch_count": 3,
            "dp_dummy_completed_batch_count": 1,
            "total_real_request_memberships": 9,
            "mean_real_requests_per_non_dummy_batch": 3,
            "mean_real_requests_per_completed_batch_including_dummy": 2.25,
            "membership_semantics": "real requests in non-dummy batches",
            "by_pd_type": {
                "decode": {
                    "completed_batch_count": 4,
                    "non_dummy_completed_batch_count": 3,
                    "dp_dummy_completed_batch_count": 1,
                    "total_real_request_memberships": 9,
                },
            },
        }
        report = {
            "schema_version": 11,
            "measurement_window": {
                "measurement_start_ns": 10,
                "measurement_end_ns": 1010,
                "measurement_duration_ns": 1000,
            },
            "online_model_compute": {"real_batch_size": batch},
            "hbm_kv_occupancy": {
                "schema_version": 1,
                "units": "per_rank_bytes",
                "window_start_ns": 10,
                "window_end_ns": 1010,
                "window_duration_ns": 1000,
                "coverage": {"covers_window": True},
                "physical_capacity_breakdown": [
                    "physical_idle_reusable",
                    "physical_non_idle_active",
                    "physical_free",
                ],
                "logical_reservation_overlay": [
                    "logical_destination_admission_reservation",
                    "reserved_free_slack",
                    "future_reclaim_backed_reservation",
                    "unclaimed_allocatable_slack",
                ],
                "per_instance": {
                    "0": {
                        "capacity_per_rank_bytes": capacity,
                        "categories": categories,
                    },
                },
                "aggregate": {
                    "capacity_per_rank_bytes_sum": capacity,
                    "categories": categories,
                    "average_physical_occupied_per_rank_bytes": 80,
                    "average_physical_occupied_utilization_fraction": 0.8,
                    "average_reservation_adjusted_claim_per_rank_bytes": 85,
                    "average_reservation_adjusted_claim_fraction": 0.85,
                },
                "conservation": {"passed": True},
            },
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": "operational-source",
        }
        metrics = _operational_metric_sources(manifest, report)
        self.assertEqual(metrics["average_active_batch_size"], 3)
        self.assertEqual(
            metrics["hbm_occupancy"]["categories"]
            ["physical_non_idle_active"],
            60,
        )
        self.assertIn(
            "non_additive_overlay", metrics["hbm_occupancy"]["scope"])

        broken = json.loads(json.dumps(report))
        broken["hbm_kv_occupancy"]["aggregate"]["categories"][
            "physical_free"]["average_per_rank_bytes"] = 21
        with self.assertRaisesRegex(ExperimentError, "fraction does not"):
            _operational_metric_sources(manifest, broken)
        broken = json.loads(json.dumps(report))
        broken["online_model_compute"]["real_batch_size"][
            "dp_dummy_completed_batch_count"] = 2
        with self.assertRaisesRegex(ExperimentError, "do not reconcile"):
            _operational_metric_sources(manifest, broken)
        with self.assertRaisesRegex(
                ExperimentError, "requires session report schema 11"):
            _operational_metric_sources(
                manifest,
                {"schema_version": 9},
            )
        legacy = _operational_metric_sources(
            {"schema_version": 10, "run_id": "legacy"},
            {"schema_version": 9},
        )
        self.assertEqual(
            legacy["source_status"],
            "unavailable_legacy_session_report",
        )

    def test_poisson_paired_performance_and_jct_decomposition_artifacts(self):
        rows = []
        policies = ("tiered<&", "residency-reference")
        jct_seconds = {
            "tiered<&": (4, 5, 8),
            "residency-reference": (2, 3, 4),
        }
        admission_seconds = {
            "tiered<&": (1, 2, 3),
            "residency-reference": (0.5, 1, 1.5),
        }
        for seed_index, seed in enumerate((41, 42, 43)):
            pair_key = f"poisson:0.2:seed={seed}"
            for order, policy in enumerate(policies):
                total = jct_seconds[policy][seed_index]
                admission = admission_seconds[policy][seed_index]
                rows.append({
                    "mode": "poisson",
                    "load_value": 0.2,
                    "policy": policy,
                    "policy_order": order,
                    "pair_key": pair_key,
                    "arrival_seed": seed,
                    "offered_arrival_trace_sha256": f"arrival-{seed}",
                    "input_session_ids_hash": "same-cohort",
                    "measurement_target_session_ids_hash": (
                        f"policy-dependent-order-{policy}"),
                    "measurement_required_session_ids_hash": (
                        f"policy-dependent-required-order-{policy}"),
                    "measured_session_ids_hash": "same-sorted-measured-set",
                    "session_jct_mean_ns": total * 1e9,
                    "session_admission_queue_mean_ns": admission * 1e9,
                    "session_execution_mean_ns": (
                        (total - admission) * 1e9),
                    "session_jct_count": 3,
                    "session_jct_sum_ns": int(total * 3e9),
                    "session_admission_queue_count": 3,
                    "session_admission_queue_sum_ns": int(
                        admission * 3e9),
                    "session_execution_count": 3,
                    "session_execution_sum_ns": int(
                        (total - admission) * 3e9),
                })

        with tempfile.TemporaryDirectory() as directory:
            normalized_paths, normalized_source = (
                plot_poisson_reference_normalized_jct(
                    rows,
                    directory,
                    reference_label="residency-reference",
                )
            )
            decomposition_paths, decomposition_source = (
                plot_poisson_session_jct_decomposition(
                    rows,
                    directory,
                    reference_label="residency-reference",
                )
            )
            self.assertEqual(len(normalized_paths), 1)
            self.assertEqual(len(decomposition_paths), 1)
            normalized_svg = normalized_paths[0].read_text(encoding="utf-8")
            decomposition_svg = decomposition_paths[0].read_text(
                encoding="utf-8")
            self.assertIn("higher is better", normalized_svg)
            self.assertIn("residency reference", normalized_svg)
            self.assertNotIn("strict oracle", normalized_svg)
            self.assertIn("tiered&lt;&amp;", normalized_svg)
            self.assertNotIn("tiered<&", normalized_svg)
            self.assertIn("admission-segment", decomposition_svg)
            self.assertIn("tiered&lt;&amp;", decomposition_svg)
            self.assertNotIn("tiered<&", decomposition_svg)

            with open(
                    normalized_source, newline="", encoding="utf-8") as source:
                normalized_rows = list(csv.DictReader(source))
            tiered = normalized_rows[0]
            reference = normalized_rows[1]
            ratios = json.loads(
                tiered["seed_level_reference_jct_over_system_jct_json"])
            self.assertEqual(ratios, [0.5, 0.6, 0.5])
            self.assertAlmostEqual(
                float(tiered["mean_reference_jct_over_system_jct"]),
                sum(ratios) / 3,
            )
            ratio_stddev = math.sqrt(sum(
                (value - sum(ratios) / 3) ** 2 for value in ratios
            ) / 2)
            self.assertAlmostEqual(
                float(tiered[
                    "ci95_half_width_reference_jct_over_system_jct"]),
                4.30265272975 * ratio_stddev / math.sqrt(3),
            )
            self.assertEqual(
                float(reference["mean_reference_jct_over_system_jct"]), 1.0)
            self.assertEqual(
                float(reference[
                    "ci95_half_width_reference_jct_over_system_jct"]), 0.0)
            self.assertIn(
                "not a strict JCT oracle", reference["reference_semantics"])

            with open(
                    decomposition_source, newline="", encoding="utf-8") as source:
                decomposition_rows = list(csv.DictReader(source))
            tiered_decomposition = decomposition_rows[0]
            self.assertEqual(
                json.loads(tiered_decomposition[
                    "seed_level_admission_queue_seconds_json"]),
                [1.0, 2.0, 3.0],
            )
            self.assertEqual(
                float(tiered_decomposition[
                    "mean_total_session_jct_seconds"]),
                float(tiered_decomposition[
                    "mean_admission_queue_seconds"])
                + float(tiered_decomposition[
                    "mean_session_execution_seconds"]),
            )
            self.assertEqual(
                tiered_decomposition["aggregation_unit"],
                "arrival_seed_level_mean",
            )

    def test_poisson_paired_jct_plot_rejects_unpaired_and_bad_provenance(self):
        rows = []
        for seed in (41, 42, 43):
            for order, policy in enumerate(("tiered", "reference")):
                if seed == 43 and policy == "tiered":
                    continue
                rows.append({
                    "mode": "poisson",
                    "load_value": 0.2,
                    "policy": policy,
                    "policy_order": order,
                    "pair_key": f"poisson:0.2:seed={seed}",
                    "arrival_seed": seed,
                    "offered_arrival_trace_sha256": f"arrival-{seed}",
                    "session_jct_mean_ns": 1e9,
                })
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                    ExperimentError, "incomplete/unpaired grid"):
                plot_poisson_reference_normalized_jct(
                    rows, directory, reference_label="reference")
        rows.append({
            "mode": "poisson",
            "load_value": 0.2,
            "policy": "tiered",
            "policy_order": 0,
            "pair_key": "poisson:0.2:seed=43",
            "arrival_seed": 43,
            "offered_arrival_trace_sha256": "conflicting-arrival",
            "session_jct_mean_ns": 1e9,
        })
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                    ExperimentError, "conflicting provenance"):
                plot_poisson_reference_normalized_jct(
                    rows, directory, reference_label="reference")

    def test_poisson_paired_plot_uses_sorted_measured_set_not_completion_order(self):
        rows = [
            {
                "mode": "poisson",
                "load_value": 0.2,
                "policy": policy,
                "policy_order": order,
                "pair_key": "poisson:0.2:seed=41",
                "arrival_seed": 41,
                "offered_arrival_trace_sha256": "same-arrivals",
                "input_session_ids_hash": "same-input-set",
                "measurement_target_session_ids_hash": target_hash,
                "measurement_required_session_ids_hash": required_hash,
                "measured_session_ids_hash": "same-sorted-measured-set",
                "session_jct_mean_ns": 1e9,
            }
            for order, (policy, target_hash, required_hash) in enumerate((
                ("tiered", "completion-order-a", "required-order-a"),
                ("reference", "completion-order-b", "required-order-b"),
            ))
        ]
        with tempfile.TemporaryDirectory() as directory:
            paths, source = plot_poisson_reference_normalized_jct(
                rows, directory, reference_label="reference")
            self.assertEqual(len(paths), 1)
            self.assertTrue(source.is_file())

        rows[0]["measured_session_ids_hash"] = "different-measured-set"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                    ExperimentError, "conflicting provenance"):
                plot_poisson_reference_normalized_jct(
                    rows, directory, reference_label="reference")

    def test_poisson_jct_decomposition_rejects_nonadditive_row(self):
        rows = [
            {
                "mode": "poisson",
                "load_value": 0.2,
                "policy": policy,
                "policy_order": order,
                "pair_key": "poisson:0.2:seed=41",
                "arrival_seed": 41,
                "session_jct_mean_ns": 10,
                "session_admission_queue_mean_ns": 3,
                "session_execution_mean_ns": 6 if policy == "tiered" else 7,
                "session_jct_count": 1,
                "session_jct_sum_ns": 10,
                "session_admission_queue_count": 1,
                "session_admission_queue_sum_ns": 3,
                "session_execution_count": 1,
                "session_execution_sum_ns": (
                    6 if policy == "tiered" else 7),
            }
            for order, policy in enumerate(("tiered", "reference"))
        ]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ExperimentError, "nonadditive"):
                plot_poisson_session_jct_decomposition(
                    rows, directory, reference_label="reference")

    def test_poisson_jct_decomposition_uses_exact_sums_for_fractional_means(self):
        rows = []
        for order, policy in enumerate(("tiered", "reference")):
            rows.append({
                "mode": "poisson",
                "load_value": 0.2,
                "policy": policy,
                "policy_order": order,
                "pair_key": "poisson:0.2:seed=41",
                "arrival_seed": 41,
                "session_jct_mean_ns": 1.0,
                "session_admission_queue_mean_ns": 1 / 3,
                "session_execution_mean_ns": 2 / 3,
                "session_jct_count": 3,
                "session_jct_sum_ns": 3,
                "session_admission_queue_count": 3,
                "session_admission_queue_sum_ns": 1,
                "session_execution_count": 3,
                "session_execution_sum_ns": 2,
            })
        with tempfile.TemporaryDirectory() as directory:
            paths, source = plot_poisson_session_jct_decomposition(
                rows, directory, reference_label="reference")
            self.assertEqual(len(paths), 1)
            self.assertTrue(source.is_file())

    def test_agentic_config_fingerprints_normalize_defaults_and_policy_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tiered = directory / "tiered.json"
            queue = directory / "queue.json"
            tiered.write_text(
                json.dumps({"policy": "tiered"}), encoding="utf-8")
            queue.write_text(json.dumps({
                "policy": "tiered_queue_recompute",
                "queue_recompute_wait_service_ratio": 2,
            }), encoding="utf-8")
            queue_headroom = directory / "queue-headroom.json"
            queue_headroom.write_text(json.dumps({
                "policy": "tiered_queue_recompute",
                "queue_recompute_wait_service_ratio": 2,
                "queue_recompute_prefill_headroom_chunks": 2,
            }), encoding="utf-8")
            explicit = directory / "explicit.json"
            explicit.write_text(json.dumps({
                "policy": "tiered",
                "pcie_bandwidth_gbps": 50,
                "cpu_bandwidth_gbps": 200,
                "ssd_num_devices": 1,
            }), encoding="utf-8")
            tiered_hashes = _agentic_config_fingerprints(tiered)
            queue_hashes = _agentic_config_fingerprints(queue)
            queue_headroom_hashes = _agentic_config_fingerprints(
                queue_headroom)
            explicit_hashes = _agentic_config_fingerprints(explicit)
            self.assertEqual(tiered_hashes, explicit_hashes)
            self.assertEqual(
                tiered_hashes["agentic_hardware_config_hash"],
                queue_hashes["agentic_hardware_config_hash"],
            )
            self.assertEqual(
                tiered_hashes["agentic_shared_control_config_hash"],
                queue_hashes["agentic_shared_control_config_hash"],
            )
            self.assertNotEqual(
                tiered_hashes["agentic_effective_config_hash"],
                queue_hashes["agentic_effective_config_hash"],
            )
            self.assertEqual(
                queue_hashes["agentic_shared_control_config_hash"],
                queue_headroom_hashes["agentic_shared_control_config_hash"],
            )
            self.assertNotEqual(
                queue_hashes["agentic_effective_config_hash"],
                queue_headroom_hashes["agentic_effective_config_hash"],
            )
            recompute_hashes = _agentic_config_fingerprints(
                tiered, policy_override="recompute")
            preserve_hashes = _agentic_config_fingerprints(
                tiered, policy_override="preserve")
            self.assertEqual(
                recompute_hashes["agentic_hardware_config_hash"],
                preserve_hashes["agentic_hardware_config_hash"],
            )
            self.assertEqual(
                recompute_hashes["agentic_shared_control_config_hash"],
                preserve_hashes["agentic_shared_control_config_hash"],
            )
            self.assertNotEqual(
                recompute_hashes["agentic_effective_config_hash"],
                preserve_hashes["agentic_effective_config_hash"],
            )
            oracle_payload = _canonical_agentic_config_payload(
                {"policy": "tiered"},
                policy_override="preserve",
                strict_oracle=True,
            )
            self.assertEqual(oracle_payload["policy"], "preserve")
            self.assertEqual(
                oracle_payload["demotion_mode"], "capacity-only")

    def test_oracle_normalized_backlog_plot_pairs_rows_and_filters_k(self):
        policies = (
            "hbm_lru_recompute",
            "hbm_ssd_direct",
            "hbm_cpu_ssd",
            "hbm_cpu_ssd_queue_recompute<&",
            "infinite_hbm_oracle",
        )
        ratios = (0.5, 0.6, 0.7, 0.8, 1.0)
        rows = []
        for load in (8, 10, 12):
            for seed, oracle_value in ((41, 2.0), (42, 4.0)):
                pair_key = f"backlog:{load}:seed={seed}"
                for order, (policy, ratio) in enumerate(
                        zip(policies, ratios)):
                    if order == 0 and seed == 42:
                        ratio = 0.75
                    rows.append({
                        "mode": "backlog",
                        "load_value": load,
                        "policy": policy,
                        "policy_order": order,
                        "pair_key": pair_key,
                        "arrival_seed": seed,
                        "sessions_per_second": oracle_value * ratio,
                    })
        with tempfile.TemporaryDirectory() as directory:
            path = plot_backlog_oracle_normalized_throughput(
                rows,
                directory,
                minimum_k=10,
            )
            self.assertEqual(
                path.name,
                "backlog_throughput_oracle_normalized_k10plus.svg",
            )
            payload = path.read_text(encoding="utf-8")
            self.assertIn(">10<", payload)
            self.assertIn(">12<", payload)
            self.assertNotIn(">8<", payload)
            self.assertIn(">0.62<", payload)
            self.assertIn(">1.00<", payload)
            self.assertIn("hbm_cpu_ssd_queue_recompute&lt;&amp;", payload)
            self.assertNotIn("hbm_cpu_ssd_queue_recompute<&", payload)
            self.assertIn('width="', payload)
            sidecar = json.loads(
                path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(
                sidecar["formula"],
                "mean_over_pairs(policy_sessions_per_second / "
                "oracle_sessions_per_second)",
            )
            cell = next(
                cell for cell in sidecar["cells"]
                if (cell["load_k"] == 10
                    and cell["policy"] == "hbm_lru_recompute"))
            self.assertEqual(
                cell["mean_paired_throughput_ratio"], 0.625)
            self.assertEqual(
                [pair["throughput_ratio"] for pair in cell["pairs"]],
                [0.5, 0.75],
            )
            self.assertNotEqual(
                cell["mean_paired_throughput_ratio"],
                (1.0 + 3.0) / (2.0 + 4.0),
            )

    def test_oracle_normalized_backlog_plot_rejects_unpaired_rows(self):
        rows = [
            {
                "mode": "backlog",
                "load_value": 10,
                "policy": "baseline",
                "policy_order": 0,
                "pair_key": "pair-a",
                "sessions_per_second": 1.0,
            },
            {
                "mode": "backlog",
                "load_value": 10,
                "policy": "infinite_hbm_oracle",
                "policy_order": 1,
                "pair_key": "pair-b",
                "sessions_per_second": 2.0,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                    ExperimentError, "multiple pair keys|unpaired rows"):
                plot_backlog_oracle_normalized_throughput(
                    rows, directory, minimum_k=10)

    def test_oracle_normalized_plot_rejects_pair_key_permutation(self):
        rows = []
        for policy, order in (("baseline", 0),
                              ("infinite_hbm_oracle", 1)):
            for seed in (41, 42):
                pair_seed = 83 - seed if policy == "baseline" else seed
                rows.append({
                    "mode": "backlog",
                    "load_value": 10,
                    "policy": policy,
                    "policy_order": order,
                    "pair_key": f"backlog:10:seed={pair_seed}",
                    "arrival_seed": seed,
                    "sessions_per_second": 1.0,
                })
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                    ExperimentError, "conflicting provenance"):
                plot_backlog_oracle_normalized_throughput(
                    rows, directory, minimum_k=10)

    def test_grouped_plot_rejects_inconsistent_policy_order(self):
        rows = [
            {
                "mode": "backlog",
                "load_value": load,
                "policy": "baseline",
                "policy_order": order,
                "sessions_per_second": 1.0,
            }
            for load, order in ((10, 0), (12, 1))
        ]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                    ExperimentError, "inconsistent policy_order"):
                plot_grouped_throughput(rows, directory)

    def test_plot_settings_are_strictly_normalized(self):
        self.assertEqual(_normalize_plot_settings(None), {})
        self.assertEqual(
            _normalize_plot_settings({
                "backlog_oracle_normalized": {"minimum_k": 10},
            }),
            {"backlog_oracle_normalized": {"minimum_k": 10}},
        )
        declared_slos = {
            "poisson_rate_metrics": {
                "resume_ttft_slo": {
                    "threshold_ms": 500,
                    "basis": "5x zero-load reference p95",
                    "provenance": {"artifact_sha256": "a" * 64},
                },
                "tpot_slo": {
                    "threshold_ms": 100,
                    "basis": "explicit raw threshold",
                    "provenance": {"citation": "fixture"},
                },
            },
        }
        self.assertEqual(
            _normalize_plot_settings(declared_slos), declared_slos)
        for invalid in (
            [],
            {"unknown": True},
            {"backlog_oracle_normalized": "yes"},
            {"backlog_oracle_normalized": {"minimum_k": True}},
            {"backlog_oracle_normalized": {"minimum_k": -1}},
            {"backlog_oracle_normalized": {"minimum_k": 1.5}},
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ExperimentError):
                    _normalize_plot_settings(invalid)
        with self.assertRaisesRegex(ExperimentError, "requires backlog mode"):
            _normalize_plot_settings(
                {"backlog_oracle_normalized": {"minimum_k": 10}},
                {"poisson": {"rates_sps": [0.1]}},
            )
        for invalid_slo in (
            {"resume_ttft_slo": 500},
            {"resume_ttft_slo": {
                "threshold_ms": 0,
                "basis": "basis",
                "provenance": {"source": "fixture"},
            }},
            {"resume_ttft_slo": {
                "threshold_ms": 500,
                "basis": "",
                "provenance": {"source": "fixture"},
            }},
            {"resume_ttft_slo": {
                "threshold_ms": 500,
                "basis": "basis",
                "provenance": {},
            }},
        ):
            with self.subTest(invalid_slo=invalid_slo):
                with self.assertRaises(ExperimentError):
                    _normalize_plot_settings({
                        "poisson_rate_metrics": invalid_slo,
                    })
        with self.assertRaisesRegex(ExperimentError, "requires poisson mode"):
            _normalize_plot_settings(
                {"poisson_rate_metrics": True},
                {"backlog": {"k_values": [10]}},
            )
        with self.assertRaisesRegex(ExperimentError, "excludes the entire"):
            _normalize_plot_settings(
                {"backlog_oracle_normalized": {"minimum_k": 10}},
                {"backlog": {"k_values": [1, 8]}},
            )

    def test_long_run_preflight_rejects_invalid_timeout_without_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            output_dir = directory / "output"
            spec_path = directory / "spec.json"
            spec_path.write_text(json.dumps({
                "name": "preflight",
                "output_dir": str(output_dir),
                "modes": {"backlog": {"k_values": [10]}},
            }), encoding="utf-8")
            for timeout in (float("nan"), 3_601):
                with self.subTest(timeout=timeout):
                    with self.assertRaisesRegex(
                            ExperimentError,
                            "positive and finite|hard per-run wall cap"):
                        run_suite(spec_path, timeout_seconds=timeout)
                    self.assertFalse(output_dir.exists())
            spec_path.write_text(json.dumps({
                "name": "preflight",
                "output_dir": str(output_dir),
                "modes": {"backlog": {"k_values": [10]}},
                "plots": {
                    "backlog_oracle_normalized": {"minimum_k": 12},
                },
            }), encoding="utf-8")
            with self.assertRaisesRegex(
                    ExperimentError, "excludes the entire"):
                run_suite(spec_path)
            self.assertFalse(output_dir.exists())

    def test_stratified_selection_never_truncates_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            rows = [
                _session("short-tool", 100, "tool"),
                _session("long-human", 100_000, "human"),
                _session("medium-tool", 10_000, "tool"),
            ]
            source.write_text("".join(
                json.dumps(row) + "\n" for row in rows))
            cohort = materialize_session_cohort(
                source,
                directory / "result",
                {
                    "strategy": "stratified_context_gap",
                    "max_sessions": 2,
                    "seed": 7,
                },
            )
            selected = [
                json.loads(line)
                for line in Path(cohort["materialized_path"])
                .read_text().splitlines()
            ]
            self.assertEqual(len(selected), 2)
            self.assertTrue(all(len(row["sub_requests"]) == 2 for row in selected))
            self.assertFalse(
                cohort["selection_rule"]["sub_requests_truncated"])
            self.assertEqual(len(cohort["selected_session_ids_hash"]), 64)

    def test_long_low_output_filters_and_repetitions_preserve_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            rows = [
                _session("tool-long", 120_000, "tool", output_tokens=1),
                _session("too-short", 1_000, "tool", output_tokens=1),
                _session("too-much-output", 130_000, "tool", output_tokens=9),
                _session(
                    "too-much-gap", 140_000, "human", output_tokens=1,
                    gap_ns=10_000),
                _session("human-long", 110_000, "human", output_tokens=1),
            ]
            source.write_text("".join(
                json.dumps(row) + "\n" for row in rows))
            cohort = materialize_session_cohort(
                source,
                directory / "result",
                {
                    "strategy": "long_context_low_output",
                    "min_context_tokens": 50_000,
                    "max_output_tokens_per_session": 5,
                    "max_total_gap_ns": 1_000,
                    "min_reuse_eligible_transitions": 1,
                    "max_sessions": 2,
                    "repetitions": 2,
                },
            )
            selected = [
                json.loads(line)
                for line in Path(cohort["materialized_path"])
                .read_text().splitlines()
            ]
            self.assertEqual(cohort["selected_template_count"], 2)
            self.assertEqual(cohort["selected_session_count"], 4)
            self.assertEqual(cohort["selected_request_count"], 8)
            self.assertEqual(len({row["session_id"] for row in selected}), 4)
            self.assertEqual(
                [row["online_experiment_source"]["source_index"]
                 for row in selected],
                [0, 4, 0, 4],
            )
            self.assertTrue(all(len(row["sub_requests"]) == 2
                                for row in selected))
            rejection_counts = cohort["selection_rule"][
                "filter_rejection_counts"]
            self.assertEqual(sum(rejection_counts.values()), 3)

    def test_global_context_scaling_is_single_factor_and_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            rows = [
                _session("tool", 120_000, "tool", output_tokens=2),
                _session("human", 80_000, "human", output_tokens=1),
            ]
            source.write_text("".join(
                json.dumps(row) + "\n" for row in rows))
            cohort = materialize_session_cohort(
                source,
                directory / "result",
                {"target_max_sequence_tokens": 1_000_000},
            )
            selected = [
                json.loads(line)
                for line in Path(cohort["materialized_path"])
                .read_text().splitlines()
            ]
            transform = cohort["context_length_transform"]
            self.assertTrue(transform["enabled"])
            self.assertFalse(transform["empirical_length_distribution"])
            self.assertEqual(
                transform["realized_max_sequence_tokens"], 1_000_000)
            self.assertEqual(
                max(sub["input_toks"] + sub["output_toks"]
                    for row in selected for sub in row["sub_requests"]),
                1_000_000,
            )
            self.assertEqual(
                sum(sub["output_toks"]
                    for row in selected for sub in row["sub_requests"]),
                6,
            )
            for row in selected:
                self.assertEqual(
                    row["online_experiment_context_transform"][
                        "global_factor_numerator"],
                    transform["global_factor_numerator"],
                )
                for sub in row["sub_requests"]:
                    original = sub["online_length_scaling_original"]
                    self.assertEqual(
                        sub["reported_input_toks"],
                        original["reported_input_toks"],
                    )
                    self.assertEqual(
                        sub["observed_provider_hit_toks"],
                        original["observed_provider_hit_toks"],
                    )
                    self.assertLessEqual(
                        sub["prefix_reuse_toks"], sub["input_toks"])
                    self.assertEqual(
                        sub["prefix_reuse_toks"]
                        + sub["newly_append_toks"],
                        sub["input_toks"],
                    )

    def test_context_scaling_preserves_full_adjacent_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            row = _session(
                "full-adjacent", 100, "tool", output_tokens=5,
                reuse_tokens=105,
            )
            row["sub_requests"][0]["raw_newly_append_toks"] = 100
            row["sub_requests"][1]["raw_newly_append_toks"] = 5
            source.write_text(json.dumps(row) + "\n")

            unscaled = materialize_session_cohort(
                source, directory / "unscaled")
            scaled = materialize_session_cohort(
                source,
                directory / "scaled",
                {"target_max_sequence_tokens": 1_000},
            )
            selected = json.loads(
                Path(scaled["materialized_path"]).read_text().strip())
            first, second = selected["sub_requests"]
            transform = scaled["context_length_transform"]

            self.assertEqual(transform["global_factor_numerator"], 199)
            self.assertEqual(transform["global_factor_denominator"], 22)
            self.assertEqual(first["input_toks"], 904)
            self.assertEqual(first["input_toks"] + first["output_toks"], 909)
            self.assertEqual(second["prefix_reuse_toks"], 909)
            self.assertEqual(
                second["policy_independent_reuse_toks"], 909)
            self.assertEqual(second["newly_append_toks"], 86)
            self.assertEqual(
                second["prefix_reuse_toks"]
                + second["newly_append_toks"],
                second["input_toks"],
            )
            self.assertEqual(second["raw_newly_append_toks"], 5)
            self.assertEqual(
                second["online_length_scaling_original"][
                    "newly_append_toks"],
                10,
            )
            self.assertEqual(
                second["online_length_scaling_lineage"][
                    "predecessor_available_tokens"],
                909,
            )
            self.assertEqual(
                unscaled["selected_session_identity_hash"],
                scaled["selected_session_identity_hash"],
            )

    def test_context_scaling_maps_partial_output_prefix_without_scaling_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            row = {
                "session_id": "partial-output-prefix",
                "arrival_time_ns": 0,
                "sub_requests": [
                    {
                        "input_toks": 100,
                        "output_toks": 10,
                        "tool_duration_ns": 1,
                        "inter_turn_gap_type": "tool",
                        "prefix_reuse_toks": 0,
                        "policy_independent_reuse_toks": 0,
                        "newly_append_toks": 100,
                        "raw_newly_append_toks": 100,
                        "reported_input_toks": 100,
                        "observed_provider_hit_toks": 0,
                        "lineage_status": "session_start",
                    },
                    {
                        "input_toks": 130,
                        "output_toks": 1,
                        "tool_duration_ns": 0,
                        "inter_turn_gap_type": "none",
                        "prefix_reuse_toks": 105,
                        "policy_independent_reuse_toks": 108,
                        "newly_append_toks": 25,
                        "raw_newly_append_toks": 25,
                        "reported_input_toks": 130,
                        "observed_provider_hit_toks": 105,
                        "lineage_status": "adjacent_estimate",
                    },
                ],
            }
            source.write_text(json.dumps(row) + "\n")
            cohort = materialize_session_cohort(
                source,
                directory / "result",
                {"target_max_sequence_tokens": 1_000},
            )
            selected = json.loads(
                Path(cohort["materialized_path"]).read_text().strip())
            first, second = selected["sub_requests"]

            self.assertEqual(first["input_toks"], 768)
            self.assertEqual(first["input_toks"] + first["output_toks"], 778)
            self.assertEqual(second["input_toks"], 999)
            self.assertEqual(second["prefix_reuse_toks"], 773)
            self.assertEqual(
                second["policy_independent_reuse_toks"], 776)
            self.assertEqual(second["newly_append_toks"], 226)
            self.assertLessEqual(
                second["policy_independent_reuse_toks"],
                first["input_toks"] + first["output_toks"],
            )

    def test_context_scaling_zeroes_explicit_lineage_break_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            row = _session(
                "lineage-break", 100, "human", output_tokens=5,
                reuse_tokens=80,
            )
            second = row["sub_requests"][1]
            second["lineage_status"] = "explicit_compaction"
            second["raw_newly_append_toks"] = 30
            source.write_text(json.dumps(row) + "\n")
            cohort = materialize_session_cohort(
                source,
                directory / "result",
                {"target_max_sequence_tokens": 1_000},
            )
            selected = json.loads(
                Path(cohort["materialized_path"]).read_text().strip())
            transformed = selected["sub_requests"][1]

            self.assertEqual(transformed["prefix_reuse_toks"], 0)
            self.assertEqual(
                transformed["policy_independent_reuse_toks"], 0)
            self.assertEqual(
                transformed["newly_append_toks"],
                transformed["input_toks"],
            )
            self.assertEqual(
                transformed["online_length_scaling_original"][
                    "prefix_reuse_toks"],
                80,
            )
            self.assertTrue(
                transformed["online_length_scaling_lineage"][
                    "lineage_break"])
            self.assertEqual(
                cohort["context_length_transform"]["reuse_adjustments"][
                    "lineage_break_fields_zeroed"],
                2,
            )

    def test_checked_in_poisson_refinement_descriptor_is_frozen(self):
        repo_root = Path(__file__).resolve().parents[1]
        spec_path = (
            repo_root / "configs" / "experiments"
            / "online_tracelab_qwen3_1m_p4d4_poisson_backlog_refinement.json"
        )
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "selected.jsonl"
            source.write_text("".join(
                json.dumps(_session(
                    f"selected-{index}", 12_000 + index, "tool")) + "\n"
                for index in range(2)
            ), encoding="utf-8")
            cohort = materialize_session_cohort(
                source, directory / "cohort")
            runs = build_run_descriptors(
                spec,
                repo_root,
                directory / "results",
                cohort,
                selected_modes=["poisson"],
            )

        self.assertEqual(len(runs), 7)
        self.assertEqual({run["load_value"] for run in runs}, {0.006})
        self.assertEqual({run["arrival_seed"] for run in runs}, {17})
        self.assertEqual(
            {run["session_repetitions"] for run in runs}, {16})
        self.assertEqual(
            {run["max_active_sessions"] for run in runs}, {20})
        self.assertEqual(
            sum(run["strict_oracle"] for run in runs), 1)
        oracle = next(run for run in runs if run["strict_oracle"])
        self.assertEqual(
            oracle["policy"], "infinite_hbm_residency_reference")

    def test_all_subprocess_descriptors_use_online_serving(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            source.write_text("".join(
                json.dumps(_session(f"s-{index}", 1000 + index, "tool"))
                + "\n" for index in range(4)))
            cluster = directory / "cluster.json"
            cluster.write_text("{}\n")
            policy = directory / "policy.json"
            policy.write_text(json.dumps({
                "policy": "tiered",
                "pcie_bandwidth_gbps": 50,
            }))
            cohort = materialize_session_cohort(
                source, directory / "result")
            spec = {
                "name": "test",
                "cluster_config": str(cluster),
                "allowed_timing_warning_codes": [
                    "request_latency_over_one_hour",
                ],
                "policies": {
                    "tiered": {
                        "agentic_kv_config": str(policy),
                        "durable_capacity_contract": (
                            "lossless-working-set"),
                    },
                    "recompute": {
                        "agentic_kv_config": str(policy),
                        "agentic_kv_policy": "hbm_lru_recompute",
                    },
                },
                "modes": {
                    "backlog": {
                        "k_values": [1, 2],
                        "warmup_completions": 1,
                        "measure_completions": 2,
                        "backlog_epochs": 1,
                    },
                    "poisson": {
                        "rates_sps": [0.5],
                        "warmup_completions": 1,
                        "measure_completions": 2,
                        "arrival_seed": 99,
                    },
                },
                "common_serving_args": ["--no-enable-prefix-caching"],
            }
            repo_root = Path(__file__).resolve().parents[1]
            runs = build_run_descriptors(
                spec, repo_root, directory / "result", cohort)
            self.assertEqual(len(runs), 9)
            for run in runs:
                self.assertEqual(run["argv"][1:3], ["-m", "serving"])
                self.assertIn("--session-stop-after-measurement", run["argv"])
                self.assertIn("--output", run["argv"])
                self.assertTrue(run["request_csv"].endswith("requests.csv"))
                self.assertEqual(
                    run["allowed_timing_warning_codes"],
                    ["request_latency_over_one_hour"],
                )
                self.assertEqual(
                    run["selected_session_ids_hash"],
                    cohort["selected_session_ids_hash"],
                )
            self.assertEqual(sum(run["strict_oracle"] for run in runs), 3)
            self.assertTrue(all(
                run["durable_capacity_contract"]
                == ("lossless-working-set"
                    if run["policy"] == "tiered" else None)
                for run in runs
            ))

            backlog_runs = build_run_descriptors(
                spec, repo_root, directory / "backlog-result", cohort,
                selected_modes=["backlog"])
            poisson_runs = build_run_descriptors(
                spec, repo_root, directory / "poisson-result", cohort,
                selected_modes=["poisson"])
            self.assertEqual(len(backlog_runs), 6)
            self.assertEqual({run["mode"] for run in backlog_runs}, {"backlog"})
            self.assertEqual(len(poisson_runs), 3)
            self.assertEqual({run["mode"] for run in poisson_runs}, {"poisson"})

            with self.assertRaisesRegex(
                    ExperimentError, "absent from spec"):
                build_run_descriptors(
                    {**spec, "modes": {"backlog": spec["modes"]["backlog"]}},
                    repo_root,
                    directory / "missing-mode-result",
                    cohort,
                    selected_modes=["poisson"],
                )
            with self.assertRaisesRegex(
                    ExperimentError, "unsupported durable capacity contract"):
                build_run_descriptors(
                    {
                        **spec,
                        "policies": {"tiered": {
                            "agentic_kv_config": str(policy),
                            "durable_capacity_contract": "unbounded-magic",
                        }},
                    },
                    repo_root,
                    directory / "bad-contract-result",
                    cohort,
                )
            with self.assertRaisesRegex(
                    ExperimentError, "cannot set durable_capacity_contract"):
                build_run_descriptors(
                    {
                        **spec,
                        "policies": {"hbm": {
                            "agentic_kv_config": str(policy),
                            "agentic_kv_policy": "hbm_lru_recompute",
                            "durable_capacity_contract": (
                                "lossless-working-set"),
                        }},
                    },
                    repo_root,
                    directory / "non-durable-contract-result",
                    cohort,
                )

    def test_admission_order_descriptor_pins_epoch_major_target(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            source.write_text("".join(
                json.dumps(_session(f"s-{index}", 1000, "tool")) + "\n"
                for index in range(2)))
            cluster = directory / "cluster.json"
            cluster.write_text("{}\n")
            policy = directory / "policy.json"
            policy.write_text(json.dumps({"policy": "tiered"}))
            cohort = materialize_session_cohort(
                source, directory / "result")
            spec = {
                "name": "fixed-admission",
                "cluster_config": str(cluster),
                "policies": {"tiered": str(policy)},
                "modes": {"backlog": {
                    "k_values": [1],
                    "backlog_epochs": 2,
                    "warmup_completions": 1,
                    "measure_completions": 2,
                    "measurement_cohort_selection": "admission_order",
                }},
            }

            runs = build_run_descriptors(
                spec, Path(__file__).resolve().parents[1],
                directory / "result", cohort)

            expected_warmup_ids = [
                "s-0::template=0::epoch=0",
            ]
            expected_ids = [
                "s-1::template=1::epoch=0",
                "s-0::template=0::epoch=1",
            ]
            expected_required_ids = expected_warmup_ids + expected_ids
            expected_hash = hashlib.sha256(json.dumps(
                expected_ids,
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")).hexdigest()
            self.assertEqual(len(runs), 2)
            for run in runs:
                self.assertEqual(
                    run["measurement_cohort_selection"], "admission_order")
                self.assertEqual(
                    run["expected_measurement_warmup_session_ids"],
                    expected_warmup_ids,
                )
                self.assertEqual(
                    run["expected_measurement_target_session_ids"],
                    expected_ids,
                )
                self.assertEqual(
                    run["expected_measurement_required_session_ids"],
                    expected_required_ids,
                )
                self.assertEqual(
                    run["expected_measurement_target_session_ids_hash"],
                    expected_hash,
                )
                self.assertEqual(run["expected_runtime_session_count"], 4)
                self.assertEqual(
                    len(run["expected_runtime_session_ids_hash"]), 64)
                flag_index = run["argv"].index(
                    "--session-measurement-cohort-selection")
                self.assertEqual(
                    run["argv"][flag_index + 1], "admission_order")

    def test_admission_order_specs_fail_before_launch_when_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            source.write_text(json.dumps(
                _session("s", 1000, "tool")) + "\n")
            cluster = directory / "cluster.json"
            cluster.write_text("{}\n")
            policy = directory / "policy.json"
            policy.write_text(json.dumps({"policy": "tiered"}))
            cohort = materialize_session_cohort(
                source, directory / "result")
            base = {
                "cluster_config": str(cluster),
                "policies": {"tiered": str(policy)},
            }

            with self.assertRaisesRegex(
                    ExperimentError, "require backlog mode"):
                build_run_descriptors({
                    **base,
                    "modes": {"poisson": {
                        "rates_sps": [1.0],
                        "measure_completions": 1,
                        "measurement_cohort_selection": "admission_order",
                    }},
                }, Path(__file__).resolve().parents[1],
                    directory / "poisson", cohort)
            with self.assertRaisesRegex(
                    ExperimentError, "must be one of"):
                build_run_descriptors({
                    **base,
                    "modes": {"backlog": {
                        "k_values": [1],
                        "measure_completions": 1,
                        "measurement_cohort_selection": "random_order",
                    }},
                }, Path(__file__).resolve().parents[1],
                    directory / "invalid", cohort)
            with self.assertRaisesRegex(
                    ExperimentError, "managed flags"):
                build_run_descriptors({
                    **base,
                    "modes": {"backlog": {
                        "k_values": [1],
                        "measure_completions": 1,
                    }},
                    "common_serving_args": [
                        "--session-measurement-cohort-selection",
                        "admission_order",
                    ],
                }, Path(__file__).resolve().parents[1],
                    directory / "managed", cohort)

    def test_admission_order_report_must_match_exact_target_list_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            workload = directory / "sessions.jsonl"
            workload.write_text("".join(
                json.dumps(_session(f"s-{index}", 1000, "tool")) + "\n"
                for index in range(3)))
            runtime_ids = [
                "s-0::template=0::epoch=0",
                "s-1::template=1::epoch=0",
                "s-2::template=2::epoch=0",
            ]
            warmup_ids = runtime_ids[:1]
            expected_ids = runtime_ids[1:]
            required_ids = warmup_ids + expected_ids
            runtime_hash = hashlib.sha256(json.dumps(
                runtime_ids,
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")).hexdigest()
            warmup_hash = hashlib.sha256(json.dumps(
                warmup_ids,
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")).hexdigest()
            expected_hash = hashlib.sha256(json.dumps(
                expected_ids,
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")).hexdigest()
            required_hash = hashlib.sha256(json.dumps(
                required_ids,
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")).hexdigest()
            manifest = {
                "run_id": "fixed-report",
                "mode": "backlog",
                "workload_path": str(workload),
                "backlog_epochs": 1,
                "warmup_completions": 1,
                "measure_completions": 2,
                "measurement_cohort_selection": "admission_order",
                "expected_runtime_session_count": 3,
                "expected_runtime_session_ids_hash": runtime_hash,
                "expected_measurement_warmup_session_ids": warmup_ids,
                "expected_measurement_warmup_session_ids_hash": warmup_hash,
                "expected_measurement_target_session_ids": expected_ids,
                "expected_measurement_target_session_ids_hash": expected_hash,
                "expected_measurement_required_session_ids": required_ids,
                "expected_measurement_required_session_ids_hash": (
                    required_hash),
            }
            report = _session_report("fixed-report", expected_ids)
            report["session_admission"] = {
                "measurement_cohort_selection": "admission_order",
                "measurement_warmup_session_ids": warmup_ids,
                "measurement_warmup_session_count": 1,
                "measurement_warmup_completed_sessions": 1,
                "measurement_target_session_count": 2,
                "measurement_target_completed_sessions": 2,
                "measurement_required_session_ids": required_ids,
                "measurement_required_session_count": 3,
                "measurement_required_completed_sessions": 3,
                "measurement_prefix_id_overlap_count": 0,
            }
            report["measurement_window"].update({
                "measurement_cohort_selection": "admission_order",
                "measurement_warmup_session_ids": warmup_ids,
                "measurement_warmup_session_count": 1,
                "measurement_warmup_completed_sessions": 1,
                "measurement_warmup_session_ids_hash": warmup_hash,
                "measurement_target_session_ids": expected_ids,
                "measurement_target_session_count": 2,
                "measurement_target_completed_sessions": 2,
                "measurement_target_session_ids_hash": expected_hash,
                "measurement_required_session_ids": required_ids,
                "measurement_required_session_count": 3,
                "measurement_required_completed_sessions": 3,
                "measurement_required_session_ids_hash": required_hash,
                "measurement_prefix_id_overlap_count": 0,
                "measurement_boundary_complete": True,
                "warmup_completion_boundary_ns": 300,
                "target_admitted_before_warmup_complete_session_count": 2,
                "target_completed_before_warmup_complete_session_count": 2,
                "target_execution_overlapped_unfinished_warmup": True,
                "target_semantics": "fixed epoch-major admission order",
                "target_order_and_hash_semantics": "canonical ordered IDs",
                "start_semantics": "minimum target admission",
                "end_semantics": "maximum target completion",
                "measurement_start_ns": 20,
                "measurement_end_ns": 200,
                "measurement_duration_ns": 180,
            })
            report["sessions"]["records"] = [
                {
                    "session_id": warmup_ids[0],
                    "status": "completed",
                    "measurement_included": False,
                    "measurement_warmup": True,
                    "measurement_target": False,
                    "measurement_required": True,
                    "measurement_role": "fixed_admission_prefix_warmup",
                    "planned_admission_index": 0,
                    "admission_index": 0,
                    "admission_time_ns": 10,
                    "completion_time_ns": 300,
                },
                {
                    "session_id": expected_ids[0],
                    "status": "completed",
                    "measurement_included": True,
                    "measurement_warmup": False,
                    "measurement_target": True,
                    "measurement_required": True,
                    "measurement_role": "measurement_target",
                    "planned_admission_index": 1,
                    "admission_index": 1,
                    "admission_time_ns": 20,
                    "completion_time_ns": 100,
                },
                {
                    "session_id": expected_ids[1],
                    "status": "completed",
                    "measurement_included": True,
                    "measurement_warmup": False,
                    "measurement_target": True,
                    "measurement_required": True,
                    "measurement_role": "measurement_target",
                    "planned_admission_index": 2,
                    "admission_index": 2,
                    "admission_time_ns": 30,
                    "completion_time_ns": 200,
                },
            ]

            validation = _validate_measurement_cohort_contract(
                manifest, report)
            self.assertTrue(validation["performed"])
            self.assertEqual(validation["expected_session_ids"], expected_ids)
            self.assertEqual(
                validation["expected_warmup_session_ids"], warmup_ids)

            reordered = json.loads(json.dumps(report))
            reordered_ids = list(reversed(expected_ids))
            reordered["measurement_window"][
                "measurement_target_session_ids"] = reordered_ids
            reordered["measurement_window"][
                "measurement_target_session_ids_hash"] = hashlib.sha256(
                    json.dumps(
                        reordered_ids, sort_keys=True,
                        separators=(",", ":"), ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest()
            with self.assertRaisesRegex(
                    ExperimentError, "fixed epoch-major sessions after warmup"):
                _validate_measurement_cohort_contract(manifest, reordered)

            bad_hash = json.loads(json.dumps(report))
            bad_hash["measurement_window"][
                "measurement_target_session_ids_hash"] = "0" * 64
            with self.assertRaisesRegex(
                    ExperimentError, "hash mismatch"):
                _validate_measurement_cohort_contract(manifest, bad_hash)

    def test_backlog_load_overrides_drive_complete_epoch_descriptors(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            source.write_text("".join(
                json.dumps(_session(f"s-{index}", 1000, "tool")) + "\n"
                for index in range(2)))
            cluster = directory / "cluster.json"
            cluster.write_text("{}\n")
            policy = directory / "policy.json"
            policy.write_text(json.dumps({"policy": "tiered"}))
            cohort = materialize_session_cohort(
                source, directory / "result")
            spec = {
                "name": "per-load",
                "cluster_config": str(cluster),
                "policies": {"tiered": str(policy)},
                "modes": {
                    "backlog": {
                        "k_values": [1, 2, 4],
                        "backlog_epochs": 3,
                        "warmup_completions": 2,
                        "measure_completions": 3,
                        "stop_after_measurement": False,
                        "require_complete_session_cohort": True,
                        "min_fraction_at_configured_k": 0.9,
                        "load_overrides": {
                            "01": {
                                "backlog_epochs": 1,
                                "warmup_completions": 1,
                                "measure_completions": 1,
                                "min_fraction_at_configured_k": 0.7,
                            },
                            "2.0": {
                                "backlog_epochs": 2,
                                "warmup_completions": 1,
                                "measure_completions": "all",
                                "min_fraction_at_configured_k": 0.8,
                            },
                        },
                    },
                },
            }
            runs = build_run_descriptors(
                spec, Path(__file__).resolve().parents[1],
                directory / "result", cohort)

            self.assertEqual(len(runs), 6)
            self.assertEqual(len({run["workload_path"] for run in runs}), 1)
            workload_rows = [
                json.loads(line) for line in
                Path(runs[0]["workload_path"]).read_text().splitlines()
            ]
            self.assertEqual(len(workload_rows), 2)
            self.assertTrue(all(
                len(row["sub_requests"]) == 2 for row in workload_rows))

            expected = {
                1: (1, 1, 1, 0.7, 2, 4, True),
                2: (2, 1, 3, 0.8, 4, 8, True),
                4: (3, 2, 3, 0.9, 6, 12, False),
            }

            def argument_value(run, flag):
                index = run["argv"].index(flag)
                return run["argv"][index + 1]

            for run in runs:
                load = int(run["load_value"])
                (epochs, warmup, measure, min_fraction, available,
                 request_count, overridden) = expected[load]
                self.assertEqual(run["backlog_epochs"], epochs)
                self.assertEqual(run["warmup_completions"], warmup)
                self.assertEqual(run["measure_completions"], measure)
                self.assertEqual(
                    run["min_fraction_at_configured_k"], min_fraction)
                self.assertEqual(run["available_sessions"], available)
                self.assertEqual(
                    run["expected_request_count"], request_count)
                self.assertIs(run["load_override_applied"], overridden)
                self.assertEqual(run["effective_load_settings"], {
                    "backlog_epochs": epochs,
                    "warmup_completions": warmup,
                    "measure_completions": measure,
                    "min_fraction_at_configured_k": min_fraction,
                })
                self.assertEqual(
                    argument_value(run, "--session-backlog-epochs"),
                    str(epochs),
                )
                self.assertEqual(
                    argument_value(run, "--session-warmup-completions"),
                    str(warmup),
                )
                self.assertEqual(
                    argument_value(run, "--session-measure-completions"),
                    str(measure),
                )

            class FailedProcess:
                returncode = 1

                def wait(self, timeout):
                    return self.returncode

            selected_run = next(
                run for run in runs if run["load_value"] == 1.0)
            with patch(
                    "serving.online_experiments.subprocess.Popen",
                    return_value=FailedProcess()):
                manifest = execute_run(
                    selected_run,
                    Path(__file__).resolve().parents[1],
                    10,
                    {},
                )
            self.assertEqual(manifest["backlog_epochs"], 1)
            self.assertEqual(manifest["available_sessions"], 2)
            self.assertEqual(manifest["expected_request_count"], 4)
            self.assertEqual(
                manifest["effective_load_settings"],
                selected_run["effective_load_settings"],
            )

    def test_backlog_load_overrides_reject_ambiguous_or_invalid_specs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            source.write_text("".join(
                json.dumps(_session(f"s-{index}", 1000, "tool")) + "\n"
                for index in range(2)))
            cluster = directory / "cluster.json"
            cluster.write_text("{}\n")
            policy = directory / "policy.json"
            policy.write_text(json.dumps({"policy": "tiered"}))
            cohort = materialize_session_cohort(
                source, directory / "result")

            def build(backlog):
                return build_run_descriptors({
                    "name": "invalid-per-load",
                    "cluster_config": str(cluster),
                    "policies": {"tiered": str(policy)},
                    "modes": {"backlog": backlog},
                }, Path(__file__).resolve().parents[1],
                    directory / "result", cohort)

            base = {
                "k_values": [1],
                "backlog_epochs": 1,
                "warmup_completions": 0,
                "measure_completions": 2,
                "stop_after_measurement": False,
                "require_complete_session_cohort": True,
            }
            with self.assertRaisesRegex(
                    ExperimentError, "absent from k_values"):
                build({**base, "load_overrides": {"2": {}}})
            with self.assertRaisesRegex(
                    ExperimentError, "duplicate keys after integer"):
                build({
                    **base,
                    "load_overrides": {"1": {}, "01.0": {}},
                })
            with self.assertRaisesRegex(ExperimentError, "unsupported keys"):
                build({
                    **base,
                    "load_overrides": {
                        "1": {"stop_after_measurement": False},
                    },
                })
            with self.assertRaisesRegex(
                    ExperimentError, r"warmup \+ measurement requires 3"):
                build({
                    **base,
                    "load_overrides": {
                        "1": {
                            "warmup_completions": 2,
                            "measure_completions": 1,
                        },
                    },
                })
            with self.assertRaisesRegex(
                    ExperimentError, "complete generated cohort"):
                build({
                    **base,
                    "stop_after_measurement": True,
                    "require_complete_session_cohort": True,
                    "load_overrides": {
                        "1": {"measure_completions": 1},
                    },
                })
            with self.assertRaisesRegex(
                    ExperimentError, "duplicate values after integer"):
                build({
                    **base,
                    "k_values": [1, "1.0"],
                })
            with self.assertRaisesRegex(
                    ExperimentError, "backlog_epochs must be a positive"):
                build({
                    **base,
                    "load_overrides": {"1": {"backlog_epochs": 1.5}},
                })

    def test_poisson_rejects_backlog_load_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            source.write_text(json.dumps(_session("s", 1000, "tool")) + "\n")
            cluster = directory / "cluster.json"
            cluster.write_text("{}\n")
            policy = directory / "policy.json"
            policy.write_text(json.dumps({"policy": "tiered"}))
            cohort = materialize_session_cohort(
                source, directory / "result")
            with self.assertRaisesRegex(
                    ExperimentError, "only for backlog mode"):
                build_run_descriptors({
                    "cluster_config": str(cluster),
                    "policies": {"tiered": str(policy)},
                    "modes": {"poisson": {
                        "rates_sps": [1.0],
                        "measure_completions": 1,
                        "load_overrides": {},
                    }},
                }, Path(__file__).resolve().parents[1],
                    directory / "result", cohort)

    def test_poisson_repetitions_seeds_and_full_drain_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            source.write_text("".join(
                json.dumps(_session(f"s-{index}", 1000, "tool")) + "\n"
                for index in range(2)))
            cluster = directory / "cluster.json"
            cluster.write_text("{}\n")
            policy = directory / "policy.json"
            policy.write_text(json.dumps({"policy": "tiered"}))
            cohort = materialize_session_cohort(
                source, directory / "result")
            spec = {
                "name": "steady",
                "cluster_config": str(cluster),
                "policies": {"tiered": str(policy)},
                "modes": {
                    "poisson": {
                        "rates_sps": [1.0],
                        "max_active_sessions": 20,
                        "arrival_seeds": [7, 11],
                        "session_repetitions": 2,
                        "warmup_completions": 1,
                        "measure_completions": 2,
                        "stop_after_measurement": False,
                        "require_complete_session_cohort": True,
                    },
                },
            }
            runs = build_run_descriptors(
                spec, Path(__file__).resolve().parents[1],
                directory / "result", cohort)

            self.assertEqual(len(runs), 4)
            self.assertEqual({run["arrival_seed"] for run in runs}, {7, 11})
            self.assertTrue(all(run["max_active_sessions"] == 20
                                for run in runs))
            self.assertTrue(all(
                run["argv"][
                    run["argv"].index("--max-active-sessions") + 1
                ] == "20"
                for run in runs))
            self.assertTrue(all(run["available_sessions"] == 4
                                for run in runs))
            self.assertTrue(all(run["expected_request_count"] == 8
                                for run in runs))
            self.assertTrue(all(
                "--no-session-stop-after-measurement" in run["argv"]
                for run in runs))
            workload_paths = {run["workload_path"] for run in runs}
            self.assertEqual(len(workload_paths), 1)
            rows = [json.loads(line) for line in
                    Path(workload_paths.pop()).read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({row["session_id"] for row in rows}), 4)
            self.assertTrue(all(len(row["sub_requests"]) == 2
                                for row in rows))

    def test_poisson_offered_trace_reproduces_exact_seeded_process(self):
        times, unit_hash = _expected_poisson_offered_arrivals(
            4, 0.25, 7)
        trace = [
            {"session_id": f"session-{index}", "offered_time_ns": time_ns}
            for index, time_ns in enumerate(times)
        ]
        manifest = {
            "run_id": "poisson-arrival-proof",
            "mode": "poisson",
            "load_value": 0.25,
            "arrival_seed": 7,
            "available_sessions": 4,
        }
        report = {"sessions": {
            "offered_arrival_trace_count": 4,
            "offered_arrival_trace_sha256": _stable_json_hash(trace),
            "records": trace,
        }}

        validation = _validate_offered_arrival_trace(manifest, report)
        self.assertTrue(validation["poisson_reproduction_exact"])
        self.assertEqual(
            validation["poisson_unit_draw_trace_sha256"], unit_hash)

        report["sessions"].pop("offered_arrival_trace_sha256")
        with self.assertRaisesRegex(ExperimentError, "hash is missing"):
            _validate_offered_arrival_trace(manifest, report)
        trace[2]["offered_time_ns"] += 1
        report["sessions"]["offered_arrival_trace_sha256"] = (
            _stable_json_hash(trace))
        with self.assertRaisesRegex(
                ExperimentError, "does not reproduce"):
            _validate_offered_arrival_trace(manifest, report)

    def test_poisson_cross_rate_crn_requires_fixed_draw_and_seed_grid(self):
        rows = []
        manifests = []
        for rate in (0.25, 0.5):
            for seed in (7, 11):
                _, unit_hash = _expected_poisson_offered_arrivals(
                    4, rate, seed)
                for policy in ("tiered", "oracle"):
                    run_id = f"rate-{rate}-seed-{seed}-{policy}"
                    rows.append({
                        "run_id": run_id,
                        "mode": "poisson",
                        "load_value": rate,
                        "arrival_seed": seed,
                        "policy": policy,
                        "offered_arrival_trace_count": 4,
                        "poisson_unit_draw_trace_sha256": unit_hash,
                    })
                    manifests.append({
                        "run_id": run_id,
                        "status": "succeeded",
                        "schema_version": SCHEMA_VERSION,
                        "workload_sha256": "1" * 64,
                        "selected_session_identity_hash": "2" * 64,
                    })

        validation = _validate_poisson_common_random_numbers(
            rows, manifests)
        self.assertTrue(validation["passed"])
        self.assertEqual(validation["rates_sps"], [0.25, 0.5])
        self.assertEqual(validation["arrival_seeds"], [7, 11])

        rows[-1]["poisson_unit_draw_trace_sha256"] = "0" * 64
        with self.assertRaisesRegex(
                ExperimentError, "unit-rate exponential draw stream"):
            _validate_poisson_common_random_numbers(rows, manifests)

    def test_exact_trace_identity_checks_every_call_field_and_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            workload = Path(directory) / "sessions.jsonl"
            workload.write_text(json.dumps(
                _session("trace", 100, "tool")) + "\n")
            manifest = {
                "run_id": "identity",
                "mode": "poisson",
                "workload_path": str(workload),
                "backlog_epochs": 1,
            }
            report = {
                "sessions": {"records": [{
                    "session_id": "trace",
                    "admission_time_ns": 5,
                    "completion_time_ns": 225,
                    "measurement_included": True,
                }]},
                "requests": {"records": [
                    {
                        "request_id": 0,
                        "session_id": "trace",
                        "source_session_id": "trace",
                        "session_template_index": 0,
                        "session_epoch": 0,
                        "sub_request_index": 0,
                        "input_tokens": 100,
                        "requested_output_tokens": 10,
                        "generated_tokens": 10,
                        "prefix_reuse_tokens": 0,
                        "return_gap_type": "session_start",
                        "return_gap_ns": 0,
                        "arrival_time_ns": 5,
                        "end_time_ns": 100,
                    },
                    {
                        "request_id": 1,
                        "session_id": "trace",
                        "source_session_id": "trace",
                        "session_template_index": 0,
                        "session_epoch": 0,
                        "sub_request_index": 1,
                        "input_tokens": 110,
                        "requested_output_tokens": 10,
                        "generated_tokens": 10,
                        "prefix_reuse_tokens": 100,
                        "return_gap_type": "tool",
                        "return_gap_ns": 100,
                        "arrival_time_ns": 200,
                        "end_time_ns": 225,
                    },
                ]},
            }
            validation = _validate_trace_identity(manifest, report)
            self.assertTrue(validation["passed"])
            report["requests"]["records"][1]["input_tokens"] = 109
            with self.assertRaisesRegex(
                    ExperimentError, "field=input_tokens"):
                _validate_trace_identity(manifest, report)

    def test_policy_invariants_cover_all_baselines_and_strict_oracle(self):
        canonical = (
            ("hbm_lru_recompute", False),
            ("hbm_ssd_direct", False),
            ("tiered", False),
            ("tiered_queue_recompute", False),
            ("preserve", True),
        )
        for policy, strict in canonical:
            with self.subTest(policy=policy):
                report = {
                    "requests": {"records": [{
                        "sub_request_index": 1,
                        "agentic_kv_source": "hbm",
                    }]},
                    "strict_infinite_hbm_oracle": ({
                        "enabled": True,
                        "passed": True,
                        "violations": [],
                        "invalid_resume_sources": [],
                        "per_instance": {"0": {"nonbinding": True}},
                    } if strict else None),
                }
                validation = _validate_policy_invariants(
                    {
                        "run_id": f"policy-{policy}",
                        "expected_agentic_policy": policy,
                        "strict_oracle": strict,
                    },
                    report,
                    _canonical_policy_report(policy),
                )
                self.assertTrue(validation["passed"])

    def test_schema7_policy_invariants_pin_runtime_effective_config(self):
        agentic = _canonical_policy_report("tiered")
        expected_hash = _stable_json_hash(
            _canonical_agentic_config_payload(agentic["config"]))
        manifest = {
            "schema_version": 7,
            "run_id": "effective-config",
            "expected_agentic_policy": "tiered",
            "strict_oracle": False,
            "agentic_effective_config_hash": expected_hash,
        }
        report = {"requests": {"records": []}}
        self.assertTrue(_validate_policy_invariants(
            manifest, report, agentic)["passed"])
        manifest["agentic_effective_config_hash"] = "0" * 64
        with self.assertRaisesRegex(
                ExperimentError, "Runtime effective agentic config hash"):
            _validate_policy_invariants(manifest, report, agentic)
        manifest.pop("agentic_effective_config_hash")
        with self.assertRaisesRegex(
                ExperimentError, "Missing or malformed effective"):
            _validate_policy_invariants(manifest, report, agentic)
        manifest.pop("expected_agentic_policy")
        with self.assertRaisesRegex(
                ExperimentError, "missing expected_agentic_policy"):
            _validate_policy_invariants(manifest, report, agentic)

    def test_queue_recompute_policy_reconciles_online_decisions(self):
        self.assertEqual(SCHEMA_VERSION, 12)
        manifest = {
            "run_id": "queue-recompute",
            "expected_agentic_policy": "tiered_queue_recompute",
            "strict_oracle": False,
        }
        report = {"requests": {"records": [{
            "sub_request_index": 1,
            "agentic_kv_source": "dropped",
        }]}}
        agentic = _selected_queue_recompute_report()

        validation = _validate_policy_invariants(
            manifest, report, agentic)

        self.assertTrue(validation["passed"])
        self.assertEqual(
            agentic["events"][1]["projected_total_wait_ns"], 101)
        self.assertEqual(
            agentic["events"][1]["projected_queue_wait_ns"], 51)
        self.assertEqual(
            agentic["events"][1][
                "projected_transient_dram_capacity_wait_ns"], 0)
        self.assertFalse(agentic["events"][1][
            "projection_available_without_new_lru_work"])
        agentic["events"][1]["projected_total_wait_ns"] = 100
        with self.assertRaisesRegex(
                ExperimentError, "Queue-recompute accounting"):
            _validate_policy_invariants(manifest, report, agentic)

    def test_queue_recompute_partial_prefix_reconciles_snapshot_and_chunk(self):
        manifest = {
            "run_id": "queue-recompute-partial",
            "expected_agentic_policy": "tiered_queue_recompute",
            "strict_oracle": False,
        }
        report = {"requests": {"records": []}}
        agentic = _selected_partial_queue_recompute_report(
            actual_wait_ns=7)

        validation = _validate_policy_invariants(
            manifest, report, agentic)

        queue_audit = validation["queue_recompute"]
        self.assertTrue(queue_audit["passed"])
        self.assertEqual(queue_audit["partial_restore_decisions"], 1)
        first_chunk = queue_audit["snapshot_to_first_chunk"]
        self.assertTrue(first_chunk["passed"])
        self.assertEqual(first_chunk["joined_count"], 1)
        self.assertEqual(first_chunk["waiting_count"], 1)
        self.assertEqual(first_chunk["actual_wait_ns"], 7)
        self.assertTrue(validation["pd_chunk_admission"]["passed"])

    def test_queue_recompute_partial_prefix_fails_closed_on_key_invariants(self):
        manifest = {
            "run_id": "queue-recompute-partial-invalid",
            "expected_agentic_policy": "tiered_queue_recompute",
            "strict_oracle": False,
        }
        report = {"requests": {"records": []}}
        mutations = (
            lambda agentic: agentic["events"][0].__setitem__(
                "selected_prefix_tokens_H", 15),
            lambda agentic: [
                event.__setitem__(
                    "selected_predicted_resume_path_ns",
                    event["full_predicted_resume_path_ns"],
                )
                for event in agentic["events"][:2]
            ],
            lambda agentic: agentic["events"][2].__setitem__(
                "capacity_headroom_claimed_by_policy", True),
            lambda agentic: agentic["queue_recompute_policy"][
                "accounting_invariants"].__setitem__("passed", False),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                agentic = _selected_partial_queue_recompute_report()
                mutation(agentic)
                with self.assertRaisesRegex(
                        ExperimentError,
                        "Queue-recompute accounting|P/D chunk-admission"):
                    _validate_policy_invariants(manifest, report, agentic)

    def test_queue_recompute_h_zero_preserves_legacy_drop_partitions(self):
        manifest = {
            "run_id": "queue-recompute-zero-legacy",
            "expected_agentic_policy": "tiered_queue_recompute",
            "strict_oracle": False,
        }
        validation = _validate_policy_invariants(
            manifest,
            {"requests": {"records": []}},
            _selected_queue_recompute_report(),
        )
        queue_audit = validation["queue_recompute"]
        self.assertEqual(queue_audit["zero_restore_decisions"], 1)
        self.assertEqual(queue_audit["partial_restore_decisions"], 0)

    def test_queue_recompute_full_restore_is_explicit_partition(self):
        validation = _validate_policy_invariants(
            {
                "run_id": "queue-recompute-full",
                "expected_agentic_policy": "tiered_queue_recompute",
                "strict_oracle": False,
            },
            {"requests": {"records": []}},
            _full_restore_queue_recompute_report(),
        )
        queue_audit = validation["queue_recompute"]
        self.assertEqual(queue_audit["full_restore_decisions"], 1)
        self.assertEqual(queue_audit["partial_restore_decisions"], 0)
        self.assertEqual(queue_audit["zero_restore_decisions"], 0)

    def test_queue_recompute_transient_dram_wait_is_bounded_and_reconciled(self):
        manifest = {
            "run_id": "queue-recompute-transient-dram",
            "expected_agentic_policy": "tiered_queue_recompute",
            "strict_oracle": False,
        }
        report = {"requests": {"records": []}}
        agentic = _selected_queue_recompute_report(
            transient_dram_capacity_wait_ns=50)
        self.assertTrue(_validate_policy_invariants(
            manifest, report, agentic)["passed"])

        for event in agentic["events"][:2]:
            event["projected_transient_dram_capacity_wait_ns"] = 52
        agentic["totals"][
            "queue_recompute_projected_transient_dram_capacity_wait_ns"] = 52
        agentic["queue_recompute_policy"][
            "selected_projected_transient_dram_capacity_wait_ns"] = 52
        with self.assertRaisesRegex(
                ExperimentError, "Queue-recompute accounting"):
            _validate_policy_invariants(manifest, report, agentic)

    def test_current_online_schema_requires_agentic_report_schema20(self):
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": "agentic-schema",
            "expected_agentic_policy": "tiered",
            "strict_oracle": False,
        }
        agentic = _canonical_policy_report("tiered")
        manifest["agentic_effective_config_hash"] = _stable_json_hash(
            _canonical_agentic_config_payload(agentic["config"]))
        agentic["schema_version"] = 19
        with self.assertRaisesRegex(
                ExperimentError, "agentic KV report schema >= 20"):
            _validate_policy_invariants(
                manifest, {"requests": {"records": []}}, agentic)
        agentic["schema_version"] = 20
        self.assertTrue(_validate_policy_invariants(
            manifest, {"requests": {"records": []}}, agentic)["passed"])

    def test_schema20_reconciles_cancelled_chunk_and_active_prefill_events(self):
        agentic = _canonical_policy_report("tiered")
        agentic["events"].extend([
            {
                "event": (
                    "pd_chunk_admission_cancelled_for_active_prefill_"
                    "recompute"),
                "request_id": 7,
                "active_prefill_recompute_generation": 0,
                "enqueued_ns": 10,
                "cancelled_ns": 15,
                "wait_ns": 5,
                "critical_wait_after_restore_ns": 3,
                "time_ns": 15,
                "cancelled_by_active_prefill_recompute": True,
                "preempted_before_commit": True,
                "committed": False,
                "admission_semantics": "cancelled_before_graph_commit",
            },
            {
                "event": "pd_active_prefill_recompute_preempt",
                "request_id": 7,
                "discarded_tokens": 40,
                "restored_hit_tokens_discarded": 32,
                "cumulative_active_prefill_recompute_tokens": 40,
                "cumulative_restored_hit_tokens_discarded": 32,
                "old_active_prefill_recompute_generation": 0,
                "new_active_prefill_recompute_generation": 1,
            },
        ])
        agentic["totals"].update({
            "pd_chunk_cancelled_admissions": 1,
            "pd_chunk_cancelled_waiting_admissions": 1,
            "pd_chunk_cancelled_admission_wait_ns": 5,
            "pd_chunk_cancelled_admission_critical_wait_ns": 3,
            "pd_active_prefill_recompute_preemptions": 1,
            "pd_active_prefill_recompute_tokens": 40,
            "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute": 32,
        })
        agentic["pd_chunk_accounting"].update({
            "cancelled_chunk_admissions": 1,
        })
        agentic["pd_active_prefill_recompute_accounting"].update({
            "preemptions": 1,
            "discarded_tokens": 40,
            "restored_hit_tokens_discarded": 32,
        })

        validation = _validate_policy_invariants(
            {
                "run_id": "schema20-active-prefill",
                "expected_agentic_policy": "tiered",
                "strict_oracle": False,
            },
            {"requests": {"records": []}},
            agentic,
        )
        self.assertTrue(validation["pd_chunk_admission"]["passed"])
        self.assertTrue(
            validation["pd_active_prefill_recompute"]["passed"])

        agentic["events"][-1][
            "new_active_prefill_recompute_generation"] = 2
        with self.assertRaisesRegex(
                ExperimentError, "active-prefill accounting"):
            _validate_policy_invariants(
                {
                    "run_id": "schema20-active-prefill-bad",
                    "expected_agentic_policy": "tiered",
                    "strict_oracle": False,
                },
                {"requests": {"records": []}},
                agentic,
            )

    def test_queue_recompute_cost_gate_uses_total_wait(self):
        manifest = {
            "run_id": "queue-recompute-cost",
            "expected_agentic_policy": "tiered_queue_recompute",
            "strict_oracle": False,
        }
        report = {"requests": {"records": [{
            "sub_request_index": 1,
            "agentic_kv_source": "dropped",
        }]}}
        agentic = _selected_queue_recompute_report(
            hbm_wait_ns=100,
            queue_wait_ns=1,
            service_ns=100,
            cost_multiplier=1.25,
            estimated_recompute_ns=160,
        )

        self.assertLessEqual(1 + 100, 200)
        self.assertGreater(100 + 1 + 100, 200)
        self.assertTrue(_validate_policy_invariants(
            manifest, report, agentic)["passed"])

    def test_queue_recompute_requires_explicit_projection_availability(self):
        manifest = {
            "run_id": "queue-recompute-available",
            "expected_agentic_policy": "tiered_queue_recompute",
            "strict_oracle": False,
        }
        report = {"requests": {"records": []}}
        agentic = _selected_queue_recompute_report()
        for event in agentic["events"][:2]:
            event.pop("projection_available")

        with self.assertRaisesRegex(
                ExperimentError, "Queue-recompute accounting"):
            _validate_policy_invariants(manifest, report, agentic)

    def test_queue_recompute_rejects_victim_flag_inconsistency(self):
        manifest = {
            "run_id": "queue-recompute-victims",
            "expected_agentic_policy": "tiered_queue_recompute",
            "strict_oracle": False,
        }
        report = {"requests": {"records": []}}
        agentic = _selected_queue_recompute_report()
        for event in agentic["events"][:2]:
            event["projection_includes_collateral_lru_work"] = False
            event["projection_available_without_new_lru_work"] = True

        with self.assertRaisesRegex(
                ExperimentError, "Queue-recompute accounting"):
            _validate_policy_invariants(manifest, report, agentic)

    def test_nonqueue_policy_requires_zero_projected_hbm_wait(self):
        manifest = {
            "run_id": "tiered-nonqueue-counter",
            "expected_agentic_policy": "tiered",
            "strict_oracle": False,
        }
        agentic = _canonical_policy_report("tiered")
        agentic["totals"][
            "queue_recompute_projected_hbm_admission_wait_ns"] = 1

        with self.assertRaisesRegex(
                ExperimentError, "expected zero totals"):
            _validate_policy_invariants(
                manifest, {"requests": {"records": []}}, agentic)

    def test_durable_capacity_contract_rejects_cancel_or_hbm_drop(self):
        manifest = {
            "run_id": "durable-semantic-contract",
            "expected_agentic_policy": "tiered",
            "strict_oracle": False,
        }
        report = {"requests": {"records": []}}
        for field in (
                "background_cancelled_jobs", "background_cancelled_bytes",
                "background_wasted_bytes", "ssd_demotion_cancelled",
                "ssd_cancelled_host_write_bytes", "hbm_capacity_drops",
                "ttl_drops"):
            with self.subTest(field=field):
                agentic = _canonical_policy_report("tiered")
                agentic["totals"][field] = 1
                with self.assertRaisesRegex(
                        ExperimentError, "expected zero totals"):
                    _validate_policy_invariants(manifest, report, agentic)

        missing = _canonical_policy_report("tiered")
        del missing["totals"]["background_cancelled_jobs"]
        with self.assertRaisesRegex(
                ExperimentError, "missing required zero totals"):
            _validate_policy_invariants(manifest, report, missing)

        cancelled = _canonical_policy_report("tiered")
        cancelled["events"].append({
            "event": "migration_cancel",
            "session_id": "s",
        })
        with self.assertRaisesRegex(
                ExperimentError, "cancelled migration"):
            _validate_policy_invariants(manifest, report, cancelled)

    def test_lossless_working_set_rejects_terminal_prefix_loss(self):
        manifest = {
            "run_id": "lossless-working-set",
            "expected_agentic_policy": "tiered",
            "durable_capacity_contract": "lossless-working-set",
            "strict_oracle": False,
        }
        reusable_hbm = {"requests": {"records": [{
            "session_id": "s",
            "sub_request_index": 1,
            "prefix_reuse_tokens": 100,
            "agentic_kv_source": "hbm",
        }]}}
        self.assertTrue(_validate_policy_invariants(
            manifest, reusable_hbm,
            _canonical_policy_report("tiered"))["passed"])

        for field in (
                "capacity_drops", "ssd_capacity_evictions",
                "ssd_capacity_admission_drops", "dropped_misses",
                "capacity_induced_recompute_tokens",
                "policy_avoidable_recompute_tokens",
                "hbf_dropped_recompute_tokens",
                "transient_dram_capacity_oversize"):
            with self.subTest(field=field):
                agentic = _canonical_policy_report("tiered")
                agentic["totals"][field] = 1
                with self.assertRaisesRegex(
                        ExperimentError, "expected zero totals"):
                    _validate_policy_invariants(
                        manifest, reusable_hbm, agentic)

        dropped_reusable = {"requests": {"records": [{
            "session_id": "s",
            "sub_request_index": 1,
            "prefix_reuse_tokens": 100,
            "agentic_kv_source": "dropped",
        }]}}
        with self.assertRaisesRegex(
                ExperimentError, "dropped reusable prefixes"):
            _validate_policy_invariants(
                manifest, dropped_reusable,
                _canonical_policy_report("tiered"))

        lineage_break = {"requests": {"records": [{
            "session_id": "s",
            "sub_request_index": 1,
            "prefix_reuse_tokens": 0,
            "agentic_kv_source": "dropped",
        }]}}
        self.assertTrue(_validate_policy_invariants(
            manifest, lineage_break,
            _canonical_policy_report("tiered"))["passed"])

    def test_terminal_ssd_lru_contract_allows_terminal_overflow(self):
        manifest = {
            "run_id": "terminal-ssd-overflow",
            "expected_agentic_policy": "tiered",
            "durable_capacity_contract": "terminal-ssd-lru",
            "strict_oracle": False,
        }
        report = {"requests": {"records": [{
            "session_id": "s",
            "sub_request_index": 1,
            "prefix_reuse_tokens": 100,
            "agentic_kv_source": "dropped",
        }]}}
        agentic = _canonical_policy_report("tiered")
        agentic["totals"].update({
            "capacity_drops": 1,
            "ssd_capacity_evictions": 1,
            "dropped_misses": 1,
            "capacity_induced_recompute_tokens": 100,
            "policy_avoidable_recompute_tokens": 100,
            "hbf_dropped_recompute_tokens": 100,
        })

        validation = _validate_policy_invariants(
            manifest, report, agentic)

        self.assertTrue(validation["passed"])
        self.assertEqual(
            validation["durable_capacity_contract"], "terminal-ssd-lru")

    def test_resume_timing_includes_source_demotion_join(self):
        timing = _validated_resume_timing("join-timing", {
            "time_ns": 10,
            "pd_pair_fifo_wait_ns": 2,
            "prepare_boundary_wait_ns": 3,
            "source_demotion_join_wait_ns": 7,
            "hbm_admission_wait_ns": 1,
            "transient_dram_capacity_wait_ns": 2,
            "queue_wait_ns": 2,
            "restore_service_ns": 7,
            "restore_ns": 10,
            "owner_gate_ns": 22,
            "restore_issue_time_ns": 22,
            "target_hbm_ready_time_ns": 23,
            "restore_ready_time_ns": 32,
        })

        self.assertEqual(timing["source_demotion_join_ns"], 7)
        self.assertEqual(timing["transient_dram_capacity_ns"], 2)

        with self.assertRaisesRegex(ExperimentError, "does not reconcile"):
            _validated_resume_timing("join-timing", {
                "time_ns": 10,
                "pd_pair_fifo_wait_ns": 2,
                "prepare_boundary_wait_ns": 3,
                "source_demotion_join_wait_ns": 7,
                "hbm_admission_wait_ns": 1,
                "transient_dram_capacity_wait_ns": 3,
                "queue_wait_ns": 2,
                "restore_service_ns": 7,
                "restore_ns": 10,
                "owner_gate_ns": 22,
                "restore_issue_time_ns": 22,
                "target_hbm_ready_time_ns": 23,
                "restore_ready_time_ns": 32,
            })

    def test_direct_fabric_policy_requires_drained_causal_astra_evidence(self):
        agentic = _canonical_policy_report("hbm_lru_recompute")
        agentic["config"].update({
            "pd_peer_transfer_mode": "direct-fabric",
            "pd_peer_bandwidth_gbps": 450,
            "pd_peer_latency_us": 1,
        })
        agentic["totals"].update({
            "external_fabric_lane_bytes": 400,
            "external_fabric_censored_lane_bytes": 0,
            "external_fabric_jobs_censored": 0,
            "pd_hbm_to_hbm_bytes": 400,
            "hbm_hits": 1,
        })
        agentic["external_fabric"] = {
            "enabled": True,
            "authority": {
                "backend": "analytical-congestion-aware",
                "bandwidth_gbps": 450,
                "bandwidth_unit": "decimal_GBps",
                "latency_ns": 1_000,
            },
            "issued_jobs": 1,
            "completed_jobs": 1,
            "censored_jobs": 0,
            "censored_lane_bytes": 0,
            "pending_jobs": 0,
            "pending_sessions": [],
            "completed_intervals": [{
                "job_id": "coldkv.0",
                "session_id": "external-session",
                "source_instance_id": 1,
                "target_instance_id": 0,
                "arrival_ns": 10,
                "start_ns": 15,
                "complete_ns": 1_020,
                "queue_wait_ns": 5,
                "service_ns": 1_005,
                "bytes_per_lane": 100,
                "lane_count": 4,
                "bytes": 400,
            }],
        }
        agentic["events"] = [{
            "time_ns": 0,
            "session_id": "external-session",
            "event": "resume",
            "source": "hbm",
            "source_instance_id": 1,
            "target_instance_id": 0,
            "source_node_id": 0,
            "target_node_id": 0,
            "hit_tokens": 1,
            "bytes": 400,
            "pd_pair_fifo_wait_ns": 2,
            "prepare_boundary_wait_ns": 3,
            "source_demotion_join_wait_ns": 0,
            "hbm_admission_wait_ns": 5,
            "queue_wait_ns": 5,
            "restore_service_ns": 1_005,
            "restore_ns": 1_015,
            "owner_gate_ns": 1_020,
            "restore_issue_time_ns": 5,
            "target_hbm_ready_time_ns": 10,
            "restore_ready_time_ns": 1_020,
        }]
        manifest = {
            "run_id": "external-proof",
            "expected_agentic_policy": "hbm_lru_recompute",
            "strict_oracle": False,
            "require_complete_session_cohort": True,
        }
        report = {"requests": {"records": [{
            "sub_request_index": 1,
            "agentic_kv_source": "hbm",
        }]}}

        validation = _validate_policy_invariants(
            manifest, report, agentic)

        self.assertEqual(validation["external_fabric_jobs"], 1)
        interval = agentic["external_fabric"]["completed_intervals"][0]
        for key in ("arrival_ns", "start_ns", "complete_ns"):
            interval[key] += 1
        with self.assertRaisesRegex(
                ExperimentError, "lacks one exact external ASTRA job"):
            _validate_policy_invariants(manifest, report, agentic)
        for key in ("arrival_ns", "start_ns", "complete_ns"):
            interval[key] -= 1
        agentic["external_fabric"]["pending_jobs"] = 1
        with self.assertRaisesRegex(ExperimentError, "does not drain"):
            _validate_policy_invariants(manifest, report, agentic)

    def test_external_fabric_model_coexecution_uses_exact_endpoint_windows(self):
        agentic = {
            "external_fabric": {
                "enabled": True,
                "completed_intervals": [{
                    "job_id": "coldkv.overlap",
                    "session_id": "owner",
                    "source_instance_id": 1,
                    "target_instance_id": 0,
                    "start_ns": 15,
                    "complete_ns": 50,
                }],
            },
            "events": [
                {
                    "event": "astra_shared_fabric_window",
                    "instance_id": 0,
                    "batch_id": 10,
                    "resource": "node:0:pd-fabric",
                    "start_ns": 0,
                    "complete_ns": 30,
                },
                {
                    "event": "astra_shared_fabric_window",
                    "instance_id": 1,
                    "batch_id": 11,
                    "resource": "node:0:pd-fabric",
                    "start_ns": 40,
                    "complete_ns": 60,
                },
                {
                    "event": "astra_shared_fabric_window",
                    "instance_id": 2,
                    "batch_id": 12,
                    "resource": "node:0:pd-fabric",
                    "start_ns": 10,
                    "complete_ns": 50,
                },
                {
                    "event": "astra_shared_fabric_window",
                    "instance_id": 0,
                    "batch_id": 13,
                    "resource": "node:0:pd-fabric",
                    "start_ns": 50,
                    "complete_ns": 70,
                },
            ],
        }

        audit = _external_fabric_model_coexecution_audit(agentic)

        self.assertTrue(audit["performed"])
        self.assertEqual(audit["coexecution_pair_count"], 2)
        self.assertEqual(audit["overlapped_job_count"], 1)
        self.assertEqual(audit["overlapped_model_window_count"], 2)
        self.assertEqual(audit["coexecution_membership_ns"], 25)
        self.assertEqual(audit["coexecution_union_ns"], 25)
        self.assertEqual(
            {sample["model_instance_id"] for sample in audit["samples"]},
            {0, 1},
        )

    def test_direct_fabric_wire_bound_uses_decimal_gbps(self):
        agentic = _canonical_policy_report("hbm_lru_recompute")
        agentic["config"]["pd_peer_transfer_mode"] = "direct-fabric"
        bytes_per_lane = 450_000_000
        total_bytes = bytes_per_lane * 4
        agentic["totals"].update({
            "external_fabric_lane_bytes": total_bytes,
            "external_fabric_censored_lane_bytes": 0,
            "external_fabric_jobs_censored": 0,
            "pd_hbm_to_hbm_bytes": total_bytes,
            "hbm_hits": 1,
        })
        agentic["external_fabric"] = {
            "enabled": True,
            "authority": {
                "backend": "analytical-congestion-aware",
                "bandwidth_gbps": 450,
                "bandwidth_unit": "decimal_GBps",
                "latency_ns": 1_000,
            },
            "issued_jobs": 1,
            "completed_jobs": 1,
            "censored_jobs": 0,
            "censored_lane_bytes": 0,
            "pending_jobs": 0,
            "pending_sessions": [],
            "completed_intervals": [{
                "job_id": "coldkv.decimal",
                "session_id": "decimal-session",
                "source_instance_id": 1,
                "target_instance_id": 0,
                "arrival_ns": 0,
                "start_ns": 0,
                "complete_ns": 950_000,
                "queue_wait_ns": 0,
                "service_ns": 950_000,
                "bytes_per_lane": bytes_per_lane,
                "lane_count": 4,
                "bytes": total_bytes,
            }],
        }
        agentic["events"] = [{
            "time_ns": 0,
            "session_id": "decimal-session",
            "event": "resume",
            "source": "hbm",
            "source_instance_id": 1,
            "target_instance_id": 0,
            "source_node_id": 0,
            "target_node_id": 0,
            "hit_tokens": 1,
            "bytes": total_bytes,
            "pd_pair_fifo_wait_ns": 0,
            "prepare_boundary_wait_ns": 0,
            "source_demotion_join_wait_ns": 0,
            "hbm_admission_wait_ns": 0,
            "queue_wait_ns": 0,
            "restore_service_ns": 950_000,
            "restore_ns": 950_000,
            "owner_gate_ns": 950_000,
            "restore_issue_time_ns": 0,
            "target_hbm_ready_time_ns": 0,
            "restore_ready_time_ns": 950_000,
        }]
        manifest = {
            "run_id": "external-decimal-bound",
            "expected_agentic_policy": "hbm_lru_recompute",
            "strict_oracle": False,
            "require_complete_session_cohort": True,
        }
        report = {"requests": {"records": [{
            "sub_request_index": 1,
            "agentic_kv_source": "hbm",
        }]}}

        with self.assertRaisesRegex(ExperimentError, "wire lower bound"):
            _validate_policy_invariants(manifest, report, agentic)

    def test_direct_fabric_censored_tail_reconciles_separately(self):
        agentic = _canonical_policy_report("hbm_lru_recompute")
        agentic["config"]["pd_peer_transfer_mode"] = "direct-fabric"
        agentic["totals"].update({
            "external_fabric_lane_bytes": 400,
            "external_fabric_censored_lane_bytes": 400,
            "external_fabric_jobs_censored": 1,
            "pd_hbm_to_hbm_bytes": 0,
            "hbm_hits": 0,
        })
        agentic["external_fabric"] = {
            "enabled": True,
            "authority": {
                "backend": "analytical-congestion-aware",
                "bandwidth_gbps": 450,
                "bandwidth_unit": "decimal_GBps",
                "latency_ns": 1_000,
            },
            "issued_jobs": 1,
            "completed_jobs": 1,
            "censored_jobs": 1,
            "censored_lane_bytes": 400,
            "pending_jobs": 0,
            "pending_sessions": [],
            "completed_intervals": [{
                "job_id": "coldkv.censored",
                "arrival_ns": 10,
                "start_ns": 10,
                "complete_ns": 1_011,
                "queue_wait_ns": 0,
                "service_ns": 1_001,
                "bytes_per_lane": 100,
                "lane_count": 4,
                "bytes": 400,
            }],
        }

        validation = _validate_policy_invariants({
            "run_id": "external-censored-tail",
            "expected_agentic_policy": "hbm_lru_recompute",
            "strict_oracle": False,
            "require_complete_session_cohort": True,
        }, {"requests": {"records": []}}, agentic)

        self.assertEqual(validation["external_fabric_jobs"], 1)

    def test_ssd_restore_requires_exact_two_stage_serial_evidence(self):
        manifest = {
            "run_id": "ssd-chain",
            "expected_agentic_policy": "hbm_ssd_direct",
            "strict_oracle": False,
        }
        report = {"requests": {"records": [{
            "sub_request_index": 1,
            "agentic_kv_source": "ssd",
        }]}}
        agentic = _canonical_policy_report(
            "hbm_ssd_direct", ssd_hit=True)
        validation = _validate_policy_invariants(
            manifest, report, agentic)
        self.assertEqual(validation["ssd_two_stage_restore_count"], 1)
        agentic["events"][2]["resources"][1] = (
            "instance:99:pcie-copy:0")
        with self.assertRaisesRegex(
                ExperimentError, "DRAM-to-HBM stage has invalid resources"):
            _validate_policy_invariants(manifest, report, agentic)
        agentic["events"][2]["resources"][1] = (
            "instance:0:pcie-copy:0")
        agentic["events"][1]["resources"][0] = "node:123:dram"
        agentic["events"][2]["resources"][0] = "node:123:dram"
        with self.assertRaisesRegex(
                ExperimentError, "SSD media stage has invalid resources"):
            _validate_policy_invariants(manifest, report, agentic)
        agentic["events"][1]["resources"][0] = "node:0:dram"
        agentic["events"][2]["resources"][0] = "node:0:dram"
        agentic["events"][1]["start_ns"] = 999
        with self.assertRaisesRegex(
                ExperimentError, "reservation timing does not reconcile"):
            _validate_policy_invariants(manifest, report, agentic)
        agentic["events"][1]["start_ns"] = 20
        for event in agentic["events"][1:]:
            for key in ("time_ns", "start_ns", "complete_ns"):
                event[key] += 1
        with self.assertRaisesRegex(ExperimentError, "exact serial"):
            _validate_policy_invariants(manifest, report, agentic)
        for event in agentic["events"][1:]:
            for key in ("time_ns", "start_ns", "complete_ns"):
                event[key] -= 1
        agentic["events"].pop()
        with self.assertRaisesRegex(ExperimentError, "stage counts"):
            _validate_policy_invariants(manifest, report, agentic)

    def test_cpu_restore_requires_exact_request_and_resource_evidence(self):
        manifest = {
            "run_id": "cpu-chain",
            "expected_agentic_policy": "tiered",
            "strict_oracle": False,
        }
        report = {"requests": {"records": [{
            "sub_request_index": 1,
            "agentic_kv_source": "cpu",
        }]}}
        agentic = _canonical_policy_report("tiered")
        agentic["totals"].update({
            "cpu_hits": 1,
            "cpu_to_hbm_bytes": 100,
        })
        agentic["events"] = [
            {
                "time_ns": 10,
                "session_id": "cpu-session",
                "event": "resume",
                "source": "cpu",
                "source_instance_id": 1,
                "target_instance_id": 0,
                "source_node_id": 0,
                "target_node_id": 0,
                "hit_tokens": 1,
                "bytes": 100,
                "pd_pair_fifo_wait_ns": 2,
                "prepare_boundary_wait_ns": 3,
                "source_demotion_join_wait_ns": 0,
                "hbm_admission_wait_ns": 5,
                "queue_wait_ns": 2,
                "restore_service_ns": 8,
                "restore_ns": 15,
                "owner_gate_ns": 20,
                "restore_issue_time_ns": 15,
                "target_hbm_ready_time_ns": 20,
                "restore_ready_time_ns": 30,
            },
            {
                "time_ns": 20,
                "session_id": "cpu-session",
                "event": "migration_reserve",
                "kind": "cpu_to_hbm",
                "start_ns": 22,
                "complete_ns": 30,
                "service_ns": 8,
                "queue_wait_ns": 2,
                "bytes": 100,
                "foreground": True,
                "resources": [
                    "node:0:dram", "instance:0:pcie-copy:0"],
            },
        ]

        validation = _validate_policy_invariants(
            manifest, report, agentic)
        self.assertEqual(validation["cpu_restore_count"], 1)

        reservation = agentic["events"][1]
        reservation["resources"][1] = "instance:99:pcie-copy:0"
        with self.assertRaisesRegex(ExperimentError, "one exact CPU->HBM"):
            _validate_policy_invariants(manifest, report, agentic)
        reservation["resources"][1] = "instance:0:pcie-copy:0"
        reservation["start_ns"] = 999
        with self.assertRaisesRegex(
                ExperimentError, "reservation timing does not reconcile"):
            _validate_policy_invariants(manifest, report, agentic)
        reservation["start_ns"] = 22
        for key in ("time_ns", "start_ns", "complete_ns"):
            reservation[key] += 1
        with self.assertRaisesRegex(ExperimentError, "one exact CPU->HBM"):
            _validate_policy_invariants(manifest, report, agentic)

    def test_collect_results_rejects_policy_dependent_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifests = []
            for label, session_ids, oracle in (
                    ("tiered", ["a", "b"], False),
                    ("oracle", ["a", "c"], True)):
                run_id = f"run-{label}"
                session_path = directory / f"{label}-session.json"
                agentic_path = directory / f"{label}-agentic.json"
                session_path.write_text(json.dumps(_session_report(
                    run_id, session_ids, source=("hbm" if oracle else "cpu"),
                    oracle=oracle)))
                agentic_path.write_text(json.dumps(_agentic_report(run_id)))
                manifests.append({
                    "status": "succeeded",
                    "run_id": run_id,
                    "session_metrics": str(session_path),
                    "agentic_kv_metrics": str(agentic_path),
                    "strict_oracle": oracle,
                    "mode": "backlog",
                    "load_value": 2.0,
                    "policy": "oracle" if oracle else "tiered",
                    "measure_completions": 2,
                    "require_complete_session_cohort": True,
                    "available_sessions": 2,
                    "expected_request_count": 4,
                    "selected_session_ids_hash": "same",
                    "selected_session_identity_hash": "same-identity",
                    "workload_sha256": "workload",
                    "cluster_config_sha256": "cluster",
                    "arrival_seed": None,
                    "warmup_completions": 0,
                    "agentic_hardware_config_hash": "hardware",
                })
            with self.assertRaisesRegex(
                    ExperimentError, "cohort differs from oracle"):
                collect_results(manifests, oracle_label="oracle")

    def test_collect_results_allows_unpaired_completion_order_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifests = []
            for label, session_ids, oracle in (
                    ("tiered", ["a", "b"], False),
                    ("oracle", ["a", "c"], True)):
                run_id = f"completion-run-{label}"
                report = _session_report(
                    run_id, session_ids,
                    source=("hbm" if oracle else "cpu"),
                    oracle=oracle,
                )
                report["measurement_window"].update({
                    "measurement_cohort_selection": "completion_order",
                    "measurement_target_session_ids": session_ids,
                    "measurement_target_session_count": len(session_ids),
                    "measurement_target_completed_sessions": len(
                        session_ids),
                    "measurement_target_session_ids_hash": (
                        _stable_json_hash(session_ids)),
                    "target_semantics": "post-execution completion order",
                    "target_order_and_hash_semantics": "ordered IDs",
                    "start_semantics": "completion-count warmup boundary",
                    "end_semantics": "final measured completion",
                })
                report["session_admission"] = {
                    "measurement_cohort_selection": "completion_order",
                }
                session_path = directory / f"{label}-session.json"
                agentic_path = directory / f"{label}-agentic.json"
                session_path.write_text(json.dumps(report))
                agentic_path.write_text(json.dumps(_agentic_report(run_id)))
                manifests.append({
                    "status": "succeeded",
                    "run_id": run_id,
                    "session_metrics": str(session_path),
                    "agentic_kv_metrics": str(agentic_path),
                    "strict_oracle": oracle,
                    "mode": "backlog",
                    "load_value": 2.0,
                    "policy": "oracle" if oracle else "tiered",
                    "measure_completions": 2,
                    "measurement_cohort_selection": "completion_order",
                    "require_complete_session_cohort": False,
                    "available_sessions": 3,
                    "expected_request_count": 4,
                    "selected_session_ids_hash": "same",
                    "selected_session_identity_hash": "same-identity",
                    "workload_sha256": "workload",
                    "cluster_config_sha256": "cluster",
                    "arrival_seed": None,
                    "warmup_completions": 0,
                    "agentic_hardware_config_hash": "hardware",
                })

            rows = collect_results(manifests, oracle_label="oracle")
            tiered = next(row for row in rows if row["policy"] == "tiered")
            self.assertFalse(
                tiered["measured_completion_cohort_pairing_required"])
            self.assertFalse(
                tiered["measured_completion_cohort_matches_oracle"])
            self.assertIsNotNone(
                tiered["oracle_throughput_slowdown_fraction"])
            self.assertIsNone(tiered["oracle_ttft_mean_slowdown_fraction"])
            self.assertIsNone(
                tiered["oracle_session_jct_mean_slowdown_fraction"])
            self.assertEqual(
                tiered["oracle_latency_comparison_status"],
                "unpaired_policy_dependent_completion_order",
            )

    def test_collect_results_pairs_each_seed_and_reports_sample_count(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifests = []
            for seed in (7, 11):
                for label, oracle in (("tiered", False), ("oracle", True)):
                    run_id = f"run-{seed}-{label}"
                    report = _session_report(
                        run_id, ["a", "b"],
                        source=("hbm" if oracle else "cpu"),
                        oracle=oracle,
                    )
                    report["session_admission"] = {
                        "session_arrival_rate_sps": 1.0,
                        "session_arrival_seed": seed,
                        "queue_policy": "arrival_time_order",
                        "logical_session_drop_count": 0,
                        "cutoff_disposition": "right_censor",
                        "slot_release_event": (
                            "final_request_completion_on_decode_owner"),
                        "slot_release_event_legacy": (
                            "final_decode_completion"),
                    }
                    report["throughput"][
                        "realized_session_offer_rate_sps"] = 1.1
                    session_path = directory / f"{run_id}-session.json"
                    agentic_path = directory / f"{run_id}-agentic.json"
                    session_path.write_text(json.dumps(report))
                    agentic_path.write_text(json.dumps(
                        _agentic_report(run_id)))
                    manifests.append({
                        "status": "succeeded",
                        "run_id": run_id,
                        "session_metrics": str(session_path),
                        "agentic_kv_metrics": str(agentic_path),
                        "strict_oracle": oracle,
                        "mode": "poisson",
                        "load_value": 1.0,
                        "policy": label,
                        "policy_order": int(oracle),
                        "arrival_seed": seed,
                        "measure_completions": 2,
                        "require_complete_session_cohort": True,
                        "available_sessions": 2,
                        "expected_request_count": 4,
                        "selected_session_ids_hash": "same",
                        "selected_session_identity_hash": "same-identity",
                        "workload_sha256": "workload",
                        "cluster_config_sha256": "cluster",
                        "warmup_completions": 0,
                        "agentic_hardware_config_hash": "hardware",
                    })

            rows = collect_results(manifests, oracle_label="oracle")
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                {row["arrival_seed_sample_count"] for row in rows}, {2})
            self.assertEqual({row["arrival_seed"] for row in rows}, {7, 11})
            arrival_hashes = {
                row["offered_arrival_trace_sha256"] for row in rows}
            self.assertEqual(len(arrival_hashes), 1)
            self.assertTrue(all(
                len(value) == 64 for value in arrival_hashes))
            self.assertEqual(
                {row["session_jct_mean_ns"] for row in rows}, {84})
            self.assertEqual(
                {row["session_jct_count"] for row in rows}, {2})
            self.assertEqual(
                {row["session_jct_sum_ns"] for row in rows}, {168})
            self.assertEqual(
                {row["session_admission_queue_sum_ns"] for row in rows},
                {8},
            )
            self.assertEqual(
                {row["session_execution_sum_ns"] for row in rows}, {160})
            self.assertEqual(
                {row["source_demotion_join_mean_ns"] for row in rows}, {0})
            self.assertEqual(
                {row["resume_source_demotion_join_mean_ns"] for row in rows},
                {0},
            )
            self.assertEqual(
                {row["transient_dram_capacity_mean_ns"] for row in rows},
                {0},
            )
            self.assertEqual(
                {row["resume_transient_dram_capacity_mean_ns"] for row in rows},
                {0},
            )
            self.assertEqual(
                {row["configured_session_arrival_rate_sps"] for row in rows},
                {1.0},
            )
            self.assertEqual(
                {row["queue_policy"] for row in rows},
                {"arrival_time_order"},
            )
            self.assertEqual(
                {row["logical_session_drop_count"] for row in rows}, {0})
            self.assertEqual({
                row[
                    "queue_recompute_selected_projected_hbm_admission_wait_ns_full_simulation"]
                for row in rows
            }, {11})
            self.assertEqual({
                row[
                    "queue_recompute_selected_projected_total_wait_ns_full_simulation"]
                for row in rows
            }, {18})
            self.assertEqual({
                row[
                    "queue_recompute_selected_projected_transient_dram_capacity_wait_ns_full_simulation"]
                for row in rows
            }, {5})
            self.assertEqual({
                row[
                    "queue_recompute_partial_restore_decisions_full_simulation"]
                for row in rows
            }, {2})
            self.assertEqual({
                row[
                    "queue_recompute_zero_restore_decisions_full_simulation"]
                for row in rows
            }, {1})
            self.assertEqual({
                row[
                    "queue_recompute_selected_restore_tokens_full_simulation"]
                for row in rows
            }, {48})
            self.assertEqual({
                row[
                    "queue_recompute_selected_estimated_suffix_recompute_ns_full_simulation"]
                for row in rows
            }, {17})
            self.assertEqual({
                row[
                    "queue_recompute_selected_estimated_recompute_ns_full_simulation"]
                for row in rows
            }, {17})
            self.assertEqual({
                row["pd_chunk_admissions_full_simulation"]
                for row in rows
            }, {5})
            self.assertEqual({
                row["pd_chunk_snapshot_feasible_wait_ns_full_simulation"]
                for row in rows
            }, {7})
            self.assertEqual(
                {row["cutoff_disposition"] for row in rows},
                {"right_censor"},
            )
            self.assertEqual(
                {row["slot_release_event"] for row in rows},
                {"final_request_completion_on_decode_owner"},
            )
            self.assertEqual(
                {row["slot_release_event_legacy"] for row in rows},
                {"final_decode_completion"},
            )
            self.assertTrue(all(
                row["kv_state_unavailable_resume_count"]
                == row["dropped_resume_count"]
                for row in rows
            ))
            self.assertTrue(all(
                row[
                    "kv_state_unavailable_resume_fraction_of_all_requests"
                ] == row["dropped_resume_fraction_of_all_requests"]
                and row[
                    "kv_state_unavailable_resume_fraction_of_resume_requests"
                ] == row["dropped_resume_fraction_of_resume_requests"]
                for row in rows
            ))
            self.assertTrue(all(
                "not_a_logical_session_drop"
                in row["dropped_resume_columns_semantics"]
                for row in rows
            ))

            for manifest in manifests:
                report_path = Path(manifest["session_metrics"])
                report = json.loads(report_path.read_text())
                report["sessions"].pop("offered_arrival_trace_count")
                report["sessions"].pop("offered_arrival_trace_sha256")
                report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(
                    ExperimentError, "trace proof is missing"):
                collect_results(manifests, oracle_label="oracle")

    def test_save_results_writes_separate_mode_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                {"mode": "backlog", "value": 1},
                {"mode": "poisson", "value": 2},
            ]
            paths = save_results(rows, directory)
            self.assertTrue(paths["combined"].is_file())
            self.assertTrue(paths["backlog"].is_file())
            self.assertTrue(paths["poisson"].is_file())

    def test_timing_warnings_support_legacy_boolean_and_coded_allowlist(self):
        warning = "maximum request latency exceeds p50 by more than 1000x"
        report = _session_report("warning-run", ["a", "b"])
        report["validation"]["timing"]["warnings"] = [warning]
        manifest = {
            "run_id": "warning-run",
            "measure_completions": 2,
            "require_complete_session_cohort": True,
            "available_sessions": 2,
            "expected_request_count": 4,
            "allow_timing_warnings": False,
        }
        with self.assertRaisesRegex(ExperimentError, "warning"):
            _validate_completed_report(
                manifest, report, _agentic_report("warning-run"))
        manifest["allow_timing_warnings"] = True
        _validate_completed_report(
            manifest, report, _agentic_report("warning-run"))

        coded_report = _session_report("coded-warning-run", ["a", "b"])
        coded_report["validation"]["timing"].update({
            "warnings": [warning],
            "warning_codes": [
                "max_request_latency_over_p50_1000x",
            ],
        })
        coded_manifest = {
            **manifest,
            "run_id": "coded-warning-run",
            "allow_timing_warnings": False,
            "allowed_timing_warning_codes": [
                "max_request_latency_over_p50_1000x",
            ],
        }
        _validate_completed_report(
            coded_manifest,
            coded_report,
            _agentic_report("coded-warning-run"),
        )

        coded_manifest["allowed_timing_warning_codes"] = []
        with self.assertRaisesRegex(ExperimentError, "outside the allowlist"):
            _validate_completed_report(
                coded_manifest,
                coded_report,
                _agentic_report("coded-warning-run"),
            )

        uncoded_manifest = {
            **manifest,
            "allow_timing_warnings": True,
            "allowed_timing_warning_codes": [
                "max_request_latency_over_p50_1000x",
            ],
        }
        with self.assertRaisesRegex(ExperimentError, "legacy un-coded"):
            _validate_completed_report(
                uncoded_manifest,
                report,
                _agentic_report("warning-run"),
            )

    def test_timing_warning_allowlist_rejects_unknown_and_duplicate_codes(self):
        self.assertEqual(
            _normalize_allowed_timing_warning_codes(
                ["request_latency_over_one_hour"], "test-setting"),
            ["request_latency_over_one_hour"],
        )
        with self.assertRaisesRegex(ExperimentError, "unknown"):
            _normalize_allowed_timing_warning_codes(
                ["request_latency_over_two_hours"], "test-setting")
        with self.assertRaisesRegex(ExperimentError, "duplicate"):
            _normalize_allowed_timing_warning_codes(
                [
                    "request_latency_over_one_hour",
                    "request_latency_over_one_hour",
                ],
                "test-setting",
            )

    def test_timing_validation_rejects_unknown_emitted_warning_code(self):
        report = _session_report("unknown-warning-run", ["a", "b"])
        report["validation"]["timing"].update({
            "warnings": ["unknown warning"],
            "warning_codes": ["unknown_warning_code"],
        })
        manifest = {
            "run_id": "unknown-warning-run",
            "measure_completions": 2,
            "require_complete_session_cohort": True,
            "available_sessions": 2,
            "expected_request_count": 4,
            "allow_timing_warnings": True,
        }

        with self.assertRaisesRegex(ExperimentError, "unknown warning codes"):
            _validate_completed_report(
                manifest,
                report,
                _agentic_report("unknown-warning-run"),
            )

    def test_full_drain_counters_and_backlog_saturation_are_enforced(self):
        report = _session_report("drain-run", ["a", "b"])
        report["measurement_window"]["measurement_early_stopped"] = False
        report["session_admission"] = {
            "stop_after_measurement": False,
            "planned_sessions": 2,
            "offered_sessions": 2,
            "admitted_sessions": 2,
            "completed_sessions": 2,
            "active_sessions": 0,
            "remaining_unadmitted_sessions": 0,
            "remaining_backlog_sessions": 0,
            "admission_frozen": False,
        }
        report["active_session_population"] = {
            "fraction_at_configured_k": 0.95,
        }
        manifest = {
            "run_id": "drain-run",
            "mode": "backlog",
            "load_value": 2,
            "measure_completions": 2,
            "available_sessions": 2,
            "expected_request_count": 4,
            "require_complete_session_cohort": True,
            "stop_after_measurement": False,
            "min_fraction_at_configured_k": 0.9,
        }
        _validate_completed_report(
            manifest, report, _agentic_report("drain-run"))

        report["session_admission"]["admitted_sessions"] = 1
        with self.assertRaisesRegex(
                ExperimentError, "Planned/admitted/completed"):
            _validate_completed_report(
                manifest, report, _agentic_report("drain-run"))
        report["session_admission"]["admitted_sessions"] = 2
        report["active_session_population"][
            "fraction_at_configured_k"] = 0.89
        with self.assertRaisesRegex(ExperimentError, "not sustained"):
            _validate_completed_report(
                manifest, report, _agentic_report("drain-run"))

    def test_schema8_session_queue_contract_forbids_logical_drops(self):
        manifest = {
            "run_id": "queue-run",
            "mode": "backlog",
            "stop_after_measurement": True,
        }
        report = {
            "schema_version": 8,
            "session_admission": {
                "queue_policy": "fifo_wait_for_slot",
                "logical_session_drop_count": 0,
                "slot_release_event": "final_decode_completion",
                "cutoff_disposition": "right_censor",
            },
            "sessions": {
                "records": [
                    {"session_id": "done", "status": "completed"},
                    {"session_id": "waiting", "status": "censored"},
                ],
            },
        }
        validation = _validate_session_queue_contract(manifest, report)
        self.assertTrue(validation["performed"])
        self.assertEqual(validation["logical_session_drop_count"], 0)

        report["session_admission"]["logical_session_drop_count"] = 1
        with self.assertRaisesRegex(ExperimentError, "Logical session drop"):
            _validate_session_queue_contract(manifest, report)
        report["session_admission"]["logical_session_drop_count"] = 0
        report["sessions"]["records"][1]["status"] = "dropped"
        with self.assertRaisesRegex(ExperimentError, "does not reconcile"):
            _validate_session_queue_contract(manifest, report)

    def test_current_manifest_rejects_legacy_session_queue_schema(self):
        manifest = {
            "run_id": "current-manifest-legacy-report",
            "schema_version": SCHEMA_VERSION,
            "mode": "poisson",
            "max_active_sessions": 20,
        }
        with self.assertRaisesRegex(
                ExperimentError, "requires session report schema"):
            _validate_session_queue_contract(
                manifest, {"schema_version": 9})

    def test_checked_in_paper_backlog_sweeps_balanced_pressure_contract(self):
        repo_root = Path(__file__).resolve().parents[1]
        spec = json.loads((
            repo_root
            / "configs/experiments/online_tracelab_qwen3_1m_p4d4_paper_backlog.json"
        ).read_text(encoding="utf-8"))
        mode = spec["modes"]["backlog"]
        contract = spec["dataset_contract"]

        self.assertEqual(
            spec["workload_selection"]["include_source_indices"],
            [2113, 3726],
        )
        self.assertEqual(
            spec["workload_selection"]["target_max_sequence_tokens"],
            1_000_000,
        )
        self.assertEqual(contract["expected_selected_template_count"], 2)
        self.assertEqual(contract["expected_selected_request_count"], 4)
        self.assertEqual(
            set(spec["policies"]),
            {"hbm_lru_recompute", "hbm_ssd_direct", "hbm_cpu_ssd"},
        )
        self.assertEqual(mode["k_values"], [8, 10, 12, 14, 16])
        self.assertEqual(mode["backlog_epochs"], 16)
        self.assertEqual(mode["warmup_completions"], 0)
        self.assertEqual(mode["measure_completions"], 4)
        self.assertEqual(
            mode["measurement_cohort_selection"], "admission_order")
        self.assertTrue(mode["stop_after_measurement"])
        self.assertFalse(mode["require_complete_session_cohort"])
        self.assertEqual(mode["min_fraction_at_configured_k"], 0.95)
        self.assertEqual(spec["required_max_slowdown_fraction"], 0.5)
        self.assertEqual(spec["max_parallel"], 2)
        self.assertEqual(spec["timeout_seconds"], 600)
        self.assertEqual(spec["ssd_resume_opportunity_contract"], {
            "mode": "backlog",
            "policy": "hbm_cpu_ssd",
            "minimum_fraction_of_all_requests": 0.3,
        })

    def test_checked_in_paper_poisson_has_exact_full_drain_descriptor_grid(self):
        repo_root = Path(__file__).resolve().parents[1]
        spec = json.loads((
            repo_root
            / "configs/experiments/online_tracelab_qwen3_1m_p4d4_paper_poisson.json"
        ).read_text(encoding="utf-8"))
        mode = spec["modes"]["poisson"]

        self.assertEqual(
            spec["workload_selection"]["include_source_indices"],
            [2113, 3726],
        )
        self.assertEqual(
            spec["workload_selection"]["target_max_sequence_tokens"],
            1_000_000,
        )
        self.assertEqual(
            set(spec["policies"]),
            {"hbm_lru_recompute", "hbm_ssd_direct", "hbm_cpu_ssd"},
        )
        self.assertEqual(mode["rates_sps"], [0.00035, 0.00075, 0.006])
        self.assertEqual(mode["arrival_seeds"], [41, 42, 43])
        self.assertEqual(mode["session_repetitions"], 4)
        self.assertEqual(mode["warmup_completions"], 0)
        self.assertEqual(mode["measure_completions"], "all")
        self.assertEqual(
            mode["measurement_cohort_selection"], "completion_order")
        self.assertFalse(mode["stop_after_measurement"])
        self.assertTrue(mode["require_complete_session_cohort"])
        self.assertFalse(mode["allow_timing_warnings"])
        self.assertEqual(
            mode["allowed_timing_warning_codes"],
            ["request_latency_over_one_hour"],
        )
        self.assertNotIn("ssd_resume_opportunity_contract", spec)
        self.assertNotIn("required_max_slowdown_fraction", spec)
        self.assertEqual(spec["max_parallel"], 2)
        self.assertEqual(spec["timeout_seconds"], 600)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            source.write_text("".join([
                json.dumps(_session("tool", 426_734, "tool")) + "\n",
                json.dumps(_session("human", 999_375, "human")) + "\n",
            ]))
            cohort = materialize_session_cohort(
                source, directory / "cohort")
            runs = build_run_descriptors(
                spec, repo_root, directory / "result", cohort)

            self.assertEqual(len(runs), 36)
            self.assertEqual(
                {(run["load_value"], run["arrival_seed"])
                 for run in runs},
                {(rate, seed) for rate in mode["rates_sps"]
                 for seed in mode["arrival_seeds"]},
            )
            pair_sizes = {}
            for run in runs:
                pair_sizes[run["pair_key"]] = (
                    pair_sizes.get(run["pair_key"], 0) + 1)
                self.assertEqual(run["arrival_seed_count"], 3)
                self.assertEqual(run["session_repetitions"], 4)
                self.assertEqual(run["available_sessions"], 8)
                self.assertEqual(run["expected_request_count"], 16)
                self.assertEqual(run["warmup_completions"], 0)
                self.assertEqual(run["measure_completions"], 8)
                self.assertEqual(
                    run["measurement_cohort_selection"],
                    "completion_order",
                )
                self.assertFalse(run["stop_after_measurement"])
                self.assertTrue(run["require_complete_session_cohort"])
                self.assertFalse(run["allow_timing_warnings"])
                self.assertEqual(
                    run["allowed_timing_warning_codes"],
                    ["request_latency_over_one_hour"],
                )
                self.assertIn(
                    "--no-session-stop-after-measurement", run["argv"])
                self.assertNotIn(
                    "--session-stop-after-measurement", run["argv"])
                self.assertEqual(
                    run["argv"].count("--no-enable-prefix-caching"), 1)
                self.assertEqual(
                    run["argv"].count("--strict-infinite-hbm-oracle"),
                    int(run["policy"] == "infinite_hbm_oracle"),
                )
            self.assertEqual(len(pair_sizes), 9)
            self.assertEqual(set(pair_sizes.values()), {4})
            self.assertEqual(
                {run["policy"] for run in runs},
                {
                    "hbm_lru_recompute",
                    "hbm_ssd_direct",
                    "hbm_cpu_ssd",
                    "infinite_hbm_oracle",
                },
            )
            self.assertEqual(sum(run["strict_oracle"] for run in runs), 9)
            self.assertEqual(len({run["workload_path"] for run in runs}), 1)
            self.assertEqual(
                len({run["selected_session_identity_hash"] for run in runs}),
                1,
            )

            repeated_rows = [
                json.loads(line)
                for line in Path(runs[0]["workload_path"])
                .read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["session_id"] for row in repeated_rows],
                [
                    f"{source_id}::poisson-rep-{repetition}"
                    for repetition in range(4)
                    for source_id in ("tool", "human")
                ],
            )
            self.assertEqual(
                sum(len(row["sub_requests"]) for row in repeated_rows), 16)

    def test_checked_in_poisson_backlog_discovery_descriptor_contract(self):
        repo_root = Path(__file__).resolve().parents[1]
        spec = json.loads((
            repo_root
            / "configs/experiments/"
            "online_tracelab_qwen3_1m_p4d4_poisson_backlog_discovery.json"
        ).read_text(encoding="utf-8"))
        mode = spec["modes"]["poisson"]

        self.assertEqual(
            mode["rates_sps"], [
                0.002, 0.003, 0.0045, 0.006, 0.009,
                0.0135, 0.02025, 0.030375,
            ])
        self.assertEqual(mode["arrival_seeds"], [17])
        self.assertEqual(mode["session_repetitions"], 16)
        self.assertEqual(mode["max_active_sessions"], 20)
        self.assertEqual(len(spec["policies"]), 3)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"
            source.write_text("".join([
                json.dumps(_session("tool", 426_734, "tool")) + "\n",
                json.dumps(_session("human", 999_375, "human")) + "\n",
            ]))
            cohort = materialize_session_cohort(
                source, directory / "cohort")
            runs = build_run_descriptors(
                spec, repo_root, directory / "result", cohort)

            self.assertEqual(len(runs), 32)
            self.assertEqual(
                {(run["load_value"], run["arrival_seed"])
                 for run in runs},
                {(rate, 17) for rate in mode["rates_sps"]},
            )
            pair_sizes = {}
            for run in runs:
                pair_sizes[run["pair_key"]] = (
                    pair_sizes.get(run["pair_key"], 0) + 1)
                self.assertEqual(run["session_repetitions"], 16)
                self.assertEqual(run["max_active_sessions"], 20)
                self.assertEqual(run["available_sessions"], 32)
                self.assertEqual(run["expected_request_count"], 64)
                self.assertEqual(
                    run["argv"][
                        run["argv"].index("--max-active-sessions") + 1
                    ],
                    "20",
                )
            self.assertEqual(len(pair_sizes), 8)
            self.assertEqual(set(pair_sizes.values()), {4})
            self.assertEqual(
                len({run["workload_path"] for run in runs}), 1)
            self.assertEqual(
                {run["policy"] for run in runs},
                {*spec["policies"], "infinite_hbm_oracle"},
            )
            self.assertEqual(sum(run["strict_oracle"] for run in runs), 8)
            self.assertEqual(
                len({run["selected_session_identity_hash"] for run in runs}),
                1,
            )

            repeated_rows = [
                json.loads(line)
                for line in Path(runs[0]["workload_path"])
                .read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["session_id"] for row in repeated_rows],
                [
                    f"{source_id}::poisson-rep-{repetition}"
                    for repetition in range(16)
                    for source_id in ("tool", "human")
                ],
            )
            self.assertEqual(
                sum(len(row["sub_requests"]) for row in repeated_rows), 64)

    def test_schema9_session_queue_contract_reconciles_fifo_censoring_and_k(self):
        manifest = {
            "run_id": "strict-queue-run",
            "mode": "backlog",
            "stop_after_measurement": True,
        }
        report = {
            "schema_version": 9,
            "session_admission": {
                "queue_policy": "fifo_wait_for_slot",
                "logical_session_drop_count": 0,
                "slot_release_event": (
                    "final_request_completion_on_decode_owner"),
                "slot_release_event_legacy": "final_decode_completion",
                "cutoff_disposition": "right_censor",
                "max_active_sessions": 2,
                "planned_sessions": 3,
                "offered_sessions": 3,
                "admitted_sessions": 2,
                "completed_sessions": 1,
                "active_sessions": 0,
                "remaining_unadmitted_sessions": 1,
                "remaining_backlog_sessions": 1,
            },
            "active_session_population": {"peak_active_sessions": 2},
            "censoring": {"censored_sessions": 2},
            "sessions": {
                "records": [
                    {
                        "session_id": "done",
                        "planned_admission_index": 0,
                        "admission_index": 0,
                        "status": "completed",
                    },
                    {
                        "session_id": "active-at-cutoff",
                        "planned_admission_index": 1,
                        "admission_index": 1,
                        "status": "censored",
                        "status_before_censoring": "active",
                    },
                    {
                        "session_id": "fifo-waiter",
                        "planned_admission_index": 2,
                        "admission_index": None,
                        "status": "censored",
                        "status_before_censoring": "backlog",
                    },
                ],
            },
        }

        validation = _validate_session_queue_contract(manifest, report)
        self.assertTrue(validation["performed"])
        self.assertEqual(
            validation["slot_release_event"],
            "final_request_completion_on_decode_owner",
        )

        report["sessions"]["records"][1]["admission_index"] = 2
        with self.assertRaisesRegex(ExperimentError, "contiguous FIFO"):
            _validate_session_queue_contract(manifest, report)
        report["sessions"]["records"][1]["admission_index"] = 1

        report["sessions"]["records"][0]["admission_index"] = 1
        report["sessions"]["records"][1]["admission_index"] = 0
        with self.assertRaisesRegex(ExperimentError, "planned FIFO prefix"):
            _validate_session_queue_contract(manifest, report)
        report["sessions"]["records"][0]["admission_index"] = 0
        report["sessions"]["records"][1]["admission_index"] = 1

        report["session_admission"]["admitted_sessions"] = 1
        with self.assertRaisesRegex(ExperimentError, "counters"):
            _validate_session_queue_contract(manifest, report)
        report["session_admission"]["admitted_sessions"] = 2

        report["active_session_population"]["peak_active_sessions"] = 3
        with self.assertRaisesRegex(ExperimentError, "peak"):
            _validate_session_queue_contract(manifest, report)
        report["active_session_population"]["peak_active_sessions"] = 2

        report["sessions"]["records"][2]["status"] = "dropped"
        with self.assertRaisesRegex(ExperimentError, "does not reconcile"):
            _validate_session_queue_contract(manifest, report)

    def test_schema9_capped_poisson_queue_contract_reconciles_fifo_and_k(self):
        manifest = {
            "run_id": "capped-poisson-run",
            "mode": "poisson",
            "max_active_sessions": 2,
            "stop_after_measurement": False,
        }
        report = {
            "schema_version": 9,
            "session_admission": {
                "queue_policy": "poisson_fifo_wait_for_slot",
                "logical_session_drop_count": 0,
                "slot_release_event": (
                    "final_request_completion_on_decode_owner"),
                "slot_release_event_legacy": "final_decode_completion",
                "cutoff_disposition": "drain",
                "max_active_sessions": 2,
                "planned_sessions": 3,
                "offered_sessions": 3,
                "admitted_sessions": 3,
                "completed_sessions": 3,
                "active_sessions": 0,
                "remaining_unadmitted_sessions": 0,
                "remaining_backlog_sessions": 0,
            },
            "active_session_population": {"peak_active_sessions": 2},
            "censoring": {"censored_sessions": 0},
            "sessions": {
                "records": [
                    {
                        "session_id": f"session-{index}",
                        "planned_admission_index": index,
                        "admission_index": index,
                        "status": "completed",
                    }
                    for index in range(3)
                ],
            },
        }

        validation = _validate_session_queue_contract(manifest, report)
        self.assertEqual(
            validation["queue_policy"], "poisson_fifo_wait_for_slot")

        report["session_admission"]["max_active_sessions"] = 3
        with self.assertRaisesRegex(ExperimentError, "configured_K=2"):
            _validate_session_queue_contract(manifest, report)

    def test_legacy_session_report_skips_queue_contract(self):
        validation = _validate_session_queue_contract(
            {"run_id": "legacy", "mode": "backlog"},
            {"schema_version": 7},
        )
        self.assertFalse(validation["performed"])

    def test_cross_classification_marginals_are_enforced(self):
        report = _session_report("cross-run", ["a", "b"])
        report["requests"][
            "resume_by_return_gap_type_and_source"]["human"] = {
                "cpu": {"count": 1}}
        manifest = {
            "run_id": "cross-run",
            "measure_completions": 2,
            "require_complete_session_cohort": True,
            "available_sessions": 2,
            "expected_request_count": 4,
        }
        with self.assertRaisesRegex(
                ExperimentError, "cross-classification"):
            _validate_completed_report(
                manifest, report, _agentic_report("cross-run"))

    def test_execute_run_rejects_timeout_above_hard_cap(self):
        with self.assertRaisesRegex(ExperimentError, "no greater than 3600"):
            execute_run({}, Path.cwd(), 3601, {})

    def test_nonfinite_or_negative_summary_metrics_are_rejected(self):
        with self.assertRaisesRegex(ExperimentError, "Non-finite"):
            _validate_summary_row({"ttft_mean_ns": float("nan")})
        with self.assertRaisesRegex(ExperimentError, "Negative"):
            _validate_summary_row({"restore_service_mean_ns": -1})

    def test_backlog_slowdown_audit_finds_first_threshold_crossing(self):
        rows = [
            {
                "mode": "backlog", "policy": "tiered",
                "load_value": 4.0,
                "oracle_throughput_slowdown_fraction": 0.2,
            },
            {
                "mode": "backlog", "policy": "tiered",
                "load_value": 8.0,
                "oracle_throughput_slowdown_fraction": 0.7,
            },
            {
                "mode": "backlog", "policy": "tiered",
                "load_value": 16.0,
                "oracle_throughput_slowdown_fraction": 1.2,
            },
            {
                "mode": "backlog", "policy": "oracle",
                "load_value": 16.0,
                "oracle_throughput_slowdown_fraction": 0.0,
            },
        ]
        audit = build_backlog_slowdown_audit(
            rows,
            oracle_label="oracle",
            required_max_slowdown_fraction=1.0,
        )
        self.assertTrue(audit["passed"])
        crossings = audit["per_policy"]["tiered"][
            "threshold_crossings"]
        self.assertEqual(crossings["0.5"]["first_k"], 8)
        self.assertEqual(crossings["1.0"]["first_k"], 16)

    def test_ssd_opportunity_audit_targets_one_policy_and_pools_counts(self):
        rows = [
            {
                "run_id": "tiered-k8", "mode": "backlog",
                "policy": "hbm_cpu_ssd", "load_value": 8.0,
                "arrival_seed": None, "attempted_ssd_resume_count": 2,
                "session_cohort_request_count": 10,
                "attempted_ssd_resume_fraction_of_all_requests": 0.2,
            },
            {
                "run_id": "tiered-k16-seed1", "mode": "backlog",
                "policy": "hbm_cpu_ssd", "load_value": 16.0,
                "arrival_seed": 1, "attempted_ssd_resume_count": 3,
                "session_cohort_request_count": 5,
                "attempted_ssd_resume_fraction_of_all_requests": 0.6,
            },
            {
                "run_id": "tiered-k16-seed2", "mode": "backlog",
                "policy": "hbm_cpu_ssd", "load_value": 16.0,
                "arrival_seed": 2, "attempted_ssd_resume_count": 3,
                "session_cohort_request_count": 15,
                "attempted_ssd_resume_fraction_of_all_requests": 0.2,
            },
            {
                "run_id": "oracle-k4", "mode": "backlog",
                "policy": "infinite_hbm_oracle", "load_value": 4.0,
                "arrival_seed": None, "attempted_ssd_resume_count": 10,
                "session_cohort_request_count": 10,
                "attempted_ssd_resume_fraction_of_all_requests": 1.0,
            },
            {
                "run_id": "recompute-k4", "mode": "backlog",
                "policy": "hbm_lru_recompute", "load_value": 4.0,
                "arrival_seed": None, "attempted_ssd_resume_count": 10,
                "session_cohort_request_count": 10,
                "attempted_ssd_resume_fraction_of_all_requests": 1.0,
            },
        ]

        audit = build_ssd_resume_opportunity_audit(rows, {
            "mode": "backlog",
            "policy": "hbm_cpu_ssd",
            "minimum_fraction_of_all_requests": 0.3,
        })

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["first_reaching_load_value"], 16.0)
        self.assertEqual(
            audit["max_observed_fraction_of_all_requests"], 0.3)
        k16 = audit["per_load"][1]
        self.assertEqual(k16["attempted_ssd_resume_count"], 6)
        self.assertEqual(k16["all_request_count"], 20)
        self.assertEqual(k16["run_count"], 2)

        missed = build_ssd_resume_opportunity_audit(rows, {
            "policy": "hbm_cpu_ssd",
            "minimum_fraction_of_all_requests": 0.31,
        })
        self.assertFalse(missed["passed"])
        self.assertIsNone(missed["first_reaching_load_value"])

    def test_ssd_opportunity_audit_rejects_unauditable_inputs(self):
        row = {
            "run_id": "tiered-k16", "mode": "backlog",
            "policy": "hbm_cpu_ssd", "load_value": 16.0,
            "arrival_seed": None, "attempted_ssd_resume_count": 3,
            "session_cohort_request_count": 10,
            "attempted_ssd_resume_fraction_of_all_requests": 0.4,
        }
        contract = {
            "policy": "hbm_cpu_ssd",
            "minimum_fraction_of_all_requests": 0.3,
        }
        with self.assertRaisesRegex(
                ExperimentError, "does not reconcile"):
            build_ssd_resume_opportunity_audit([row], contract)
        with self.assertRaisesRegex(
                ExperimentError, "has no result rows"):
            build_ssd_resume_opportunity_audit([row], {
                **contract, "policy": "hbm_ssd_direct",
            })
        with self.assertRaisesRegex(ExperimentError, "finite and in"):
            build_ssd_resume_opportunity_audit([row], {
                **contract, "minimum_fraction_of_all_requests": 1.01,
            })

    def test_ssd_contract_skips_unselected_mode_but_validates_references(self):
        spec = {
            "modes": {
                "backlog": {"k_values": [16]},
                "poisson": {"rates_sps": [0.1]},
            },
            "policies": {"hbm_cpu_ssd": {}},
            "ssd_resume_opportunity_contract": {
                "mode": "backlog",
                "policy": "hbm_cpu_ssd",
                "minimum_fraction_of_all_requests": 0.3,
            },
        }

        declared, active = _prepare_ssd_resume_opportunity_contract(
            spec, ("poisson",))
        self.assertEqual(declared["mode"], "backlog")
        self.assertIsNone(active)
        _, active = _prepare_ssd_resume_opportunity_contract(
            spec, ("backlog",))
        self.assertEqual(active, declared)

        with self.assertRaisesRegex(ExperimentError, "unknown policy"):
            _prepare_ssd_resume_opportunity_contract({
                **spec,
                "ssd_resume_opportunity_contract": {
                    **spec["ssd_resume_opportunity_contract"],
                    "policy": "hbm_cpu_ssd_typo",
                },
            }, ("poisson",))
        with self.assertRaisesRegex(ExperimentError, "unconfigured mode"):
            _prepare_ssd_resume_opportunity_contract({
                **spec,
                "modes": {"poisson": {"rates_sps": [0.1]}},
            }, ("poisson",))

    def test_ssd_contract_policy_typo_fails_before_subprocess_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            output_dir = directory / "must-not-be-created"
            spec_path = directory / "spec.json"
            spec_path.write_text(json.dumps({
                "output_dir": str(output_dir),
                "modes": {
                    "backlog": {"k_values": [16]},
                    "poisson": {"rates_sps": [0.1]},
                },
                "policies": {"hbm_cpu_ssd": {}},
                "ssd_resume_opportunity_contract": {
                    "mode": "backlog",
                    "policy": "hbm_cpu_ssd_typo",
                    "minimum_fraction_of_all_requests": 0.3,
                },
            }))
            with patch(
                    "serving.online_experiments.execute_run") as execute:
                with self.assertRaisesRegex(
                        ExperimentError, "unknown policy"):
                    run_suite(spec_path, modes=["poisson"])
                execute.assert_not_called()
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()

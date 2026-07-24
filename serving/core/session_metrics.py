"""Summary metrics for trace, Poisson, and closed-backlog session runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict


TIMING_WARNING_CODES = frozenset({
    "zero_inter_token_latency",
    "zero_request_latency",
    "request_latency_over_one_hour",
    "max_request_latency_over_p50_1000x",
})


def _stable_json_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _percentile(sorted_values, percentile):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _distribution(values):
    raw = list(values)
    invalid = [
        value for value in raw
        if value is None or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value)) or value < 0
        or int(value) != value
    ]
    if invalid:
        raise ValueError(
            "Latency distributions cannot silently discard missing, "
            f"non-finite, or negative values: {invalid[:5]}")
    clean = sorted(int(value) for value in raw)
    if not clean:
        return {
            "count": 0,
            "sum": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(clean),
        "sum": sum(clean),
        "mean": sum(clean) / len(clean),
        "p50": _percentile(clean, 50),
        "p90": _percentile(clean, 90),
        "p99": _percentile(clean, 99),
        "min": clean[0],
        "max": clean[-1],
    }


def _rate(count, duration_ns):
    if duration_ns is None or duration_ns <= 0:
        return None
    return count * 1_000_000_000 / duration_ns


def _request_group(requests):
    return {
        "count": len(requests),
        "latency_ns": _distribution(request.latency for request in requests),
        "release_to_first_schedule_ns": _distribution(
            request.queuing_delay for request in requests
        ),
        "scheduler_queue_wait_ns": _distribution(
            request.scheduler_queue_wait_ns for request in requests
        ),
        "ttft_ns": _distribution(request.ttft for request in requests),
        "tpot_ns": _distribution(request.tpot for request in requests),
        "itl_ns": _distribution(
            value for request in requests for value in request.itl
        ),
        "restore_gate_wait_ns": _distribution(
            request.agentic_kv_restore_gate_wait_ns for request in requests
        ),
        "owner_ready_gate_ns": _distribution(
            request.agentic_kv_owner_gate_ns for request in requests
        ),
        "pd_pair_fifo_wait_ns": _distribution(
            request.pd_pair_fifo_wait_ns for request in requests
        ),
        "prepare_boundary_wait_ns": _distribution(
            request.agentic_kv_prepare_boundary_wait_ns
            for request in requests
        ),
        "source_demotion_join_wait_ns": _distribution(
            request.agentic_kv_source_demotion_join_wait_ns
            for request in requests
        ),
        "hbm_admission_wait_ns": _distribution(
            request.agentic_kv_hbm_admission_wait_ns for request in requests
        ),
        "transient_dram_capacity_wait_ns": _distribution(
            request.agentic_kv_transient_dram_capacity_wait_ns
            for request in requests
        ),
        "restore_queue_wait_ns": _distribution(
            request.agentic_kv_restore_queue_wait_ns for request in requests
        ),
        "restore_service_ns": _distribution(
            request.agentic_kv_restore_service_ns for request in requests
        ),
        "pd_launch_admission_wait_ns": _distribution(
            request.pd_launch_admission_wait_ns for request in requests
        ),
        "pd_launch_admission_critical_wait_ns": _distribution(
            request.pd_launch_admission_critical_wait_ns for request in requests
        ),
        "pd_chunk_attempt_admission_wait_ns": _distribution(
            request.pd_chunk_admission_wait_ns_total for request in requests
        ),
        "pd_chunk_attempt_admission_critical_wait_ns": _distribution(
            request.pd_chunk_admission_critical_wait_ns_total
            for request in requests
        ),
        "pd_chunk_successful_admission_wait_ns": _distribution(
            request.pd_chunk_successful_admission_wait_ns_total
            for request in requests
        ),
        "pd_chunk_successful_admission_critical_wait_ns": _distribution(
            request.pd_chunk_successful_admission_critical_wait_ns_total
            for request in requests
        ),
        "pd_chunk_cancelled_admission_wait_ns": _distribution(
            request.pd_chunk_cancelled_admission_wait_ns_total
            for request in requests
        ),
        "pd_chunk_cancelled_admission_critical_wait_ns": _distribution(
            request.pd_chunk_cancelled_admission_critical_wait_ns_total
            for request in requests
        ),
    }


def _cross_request_groups(requests, first_key, second_key):
    groups = defaultdict(lambda: defaultdict(list))
    for request in requests:
        first = str(first_key(request) or "unknown")
        second = str(second_key(request) or "unknown")
        groups[first][second].append(request)
    return {
        first: {
            second: _request_group(group)
            for second, group in sorted(second_groups.items())
        }
        for first, second_groups in sorted(groups.items())
    }


def _timing_validation(requests, simulated_duration_ns, lifecycle_by_session):
    violations = []
    warnings = []
    warning_codes = []

    def add_warning(code, message):
        if code not in TIMING_WARNING_CODES:
            raise RuntimeError(f"Unknown internal timing warning code: {code}")
        warning_codes.append(code)
        warnings.append(message)

    request_ids = [int(request.id) for request in requests]
    if len(request_ids) != len(set(request_ids)):
        violations.append("completed request IDs are not unique")
    for request in requests:
        prefix = f"request={request.id}, session={request.session_id}"
        if request.first_schedule_time_ns is None:
            violations.append(f"{prefix}: missing first schedule timestamp")
            continue
        if request.first_schedule_eligibility_time_ns is None:
            violations.append(f"{prefix}: missing scheduler eligibility")
            continue
        if request.first_schedule_request_ready_time_ns is None:
            violations.append(
                f"{prefix}: missing request-ready snapshot at first schedule")
            continue
        if request.first_schedule_resource_ready_time_ns is None:
            violations.append(
                f"{prefix}: missing resource-ready snapshot at first schedule")
            continue
        expected_eligibility = max(
            int(request.arrival),
            int(request.first_schedule_request_ready_time_ns),
            int(request.first_schedule_resource_ready_time_ns),
        )
        if int(request.first_schedule_eligibility_time_ns) != expected_eligibility:
            violations.append(
                f"{prefix}: scheduler eligibility does not reconcile "
                f"(recorded={request.first_schedule_eligibility_time_ns}, "
                f"expected={expected_eligibility})")
        if int(request.first_schedule_time_ns) < int(
                request.first_schedule_eligibility_time_ns):
            violations.append(f"{prefix}: scheduled before eligibility")
        if int(request.first_schedule_eligibility_time_ns) < int(
                request.arrival):
            violations.append(f"{prefix}: eligibility precedes release")
        if int(request.first_schedule_time_ns) > int(request.end_time):
            violations.append(f"{prefix}: first schedule follows completion")
        if request.scheduler_queue_wait_ns is None:
            violations.append(f"{prefix}: missing scheduler queue wait")
        elif int(request.scheduler_queue_wait_ns) < 0:
            violations.append(f"{prefix}: negative scheduler queue wait")
        elif int(request.scheduler_queue_wait_ns) != (
                int(request.first_schedule_time_ns)
                - int(request.first_schedule_eligibility_time_ns)):
            violations.append(
                f"{prefix}: scheduler queue wait does not reconcile")
        elif int(request.scheduler_queue_wait_ns) > int(request.queuing_delay):
            violations.append(
                f"{prefix}: pure scheduler wait exceeds release-to-schedule")
        if int(request.queuing_delay) != (
                int(request.first_schedule_time_ns) - int(request.arrival)):
            violations.append(
                f"{prefix}: release-to-schedule delay does not reconcile")
        if int(request.end_time) < int(request.arrival):
            violations.append(f"{prefix}: completion precedes release")
        if int(request.latency) != int(request.end_time) - int(request.arrival):
            violations.append(f"{prefix}: latency does not reconcile")
        if not 0 <= int(request.ttft) <= int(request.latency):
            violations.append(f"{prefix}: TTFT is outside request latency")
        if int(request.ttft) < int(request.queuing_delay):
            violations.append(f"{prefix}: TTFT precedes first schedule")
        if int(request.tpot) < 0:
            violations.append(f"{prefix}: negative TPOT")
        requested_output_tokens = int(request.output) - int(
            request.original_input)
        if requested_output_tokens <= 0:
            violations.append(f"{prefix}: non-positive requested output")
        if int(request.generated_tokens) != requested_output_tokens:
            violations.append(
                f"{prefix}: generated-token count does not reconcile "
                f"(generated={request.generated_tokens}, "
                f"requested={requested_output_tokens})")
        expected_itl_count = max(0, requested_output_tokens - 1)
        if len(request.itl) != expected_itl_count:
            violations.append(
                f"{prefix}: ITL count does not reconcile "
                f"(observed={len(request.itl)}, "
                f"expected={expected_itl_count})")
        if any(int(value) < 0 for value in request.itl):
            violations.append(f"{prefix}: negative inter-token latency")
        elif any(int(value) == 0 for value in request.itl):
            add_warning(
                "zero_inter_token_latency",
                f"{prefix}: zero inter-token latency",
            )
        if sum(int(value) for value in request.itl) != (
                int(request.latency) - int(request.ttft)):
            violations.append(
                f"{prefix}: ITL sum does not reconcile with latency and TTFT")
        expected_tpot = (
            0 if requested_output_tokens == 1 else
            (int(request.latency) - int(request.ttft))
            // (requested_output_tokens - 1)
        )
        if int(request.tpot) != expected_tpot:
            violations.append(
                f"{prefix}: TPOT does not reconcile with latency, TTFT, and "
                "output-token count")
        if (str(request.agentic_kv_source) in {"cpu", "ssd"}
                and int(request.prefix_reuse_tokens) > 0
                and int(request.agentic_kv_restore_service_ns) <= 0):
            violations.append(
                f"{prefix}: lower-tier hit has no restore service")
        source_demotion_join_wait_ns = int(
            request.agentic_kv_source_demotion_join_wait_ns)
        if (source_demotion_join_wait_ns > 0
                and str(request.agentic_kv_source) not in {
                    "cpu", "ssd", "dropped"
                }):
            violations.append(
                f"{prefix}: source-demotion join did not resolve to a lower "
                "or terminal tier")
        if (int(request.agentic_kv_restore_ready_time_ns)
                > int(request.end_time)):
            violations.append(
                f"{prefix}: request completed before restore dependency")
        transient_dram_wait_ns = int(
            request.agentic_kv_transient_dram_capacity_wait_ns)
        restore_queue_wait_ns = int(
            request.agentic_kv_restore_queue_wait_ns)
        if transient_dram_wait_ns > restore_queue_wait_ns:
            violations.append(
                f"{prefix}: transient DRAM capacity wait exceeds total "
                "lower-tier restore queue wait")
        if (transient_dram_wait_ns > 0
                and str(request.agentic_kv_source) != "ssd"):
            violations.append(
                f"{prefix}: non-SSD resume has transient DRAM capacity wait")
        restore_ready_ns = int(request.agentic_kv_restore_ready_time_ns)
        if not request.agentic_kv_async_decode_join:
            if int(request.first_schedule_time_ns) < restore_ready_ns:
                violations.append(
                    f"{prefix}: pre-admission request scheduled before "
                    "restore-ready dependency")
        else:
            cutoff = request.agentic_kv_overlap_cutoff_tokens
            if (int(request.first_schedule_time_ns) < restore_ready_ns
                    and cutoff is None):
                violations.append(
                    f"{prefix}: decode-join overlap started without an "
                    "explicit prompt cutoff")
            if cutoff is not None and not (
                    int(request.agentic_kv_hit_tokens)
                    <= int(cutoff)
                    <= int(request.original_input) - 1):
                violations.append(
                    f"{prefix}: decode-join prompt cutoff is invalid")
            if (int(request.agentic_kv_restore_gate_wait_ns) > 0
                    and int(request.agentic_kv_restore_gate_start_ns)
                    + int(request.agentic_kv_restore_gate_wait_ns)
                    != restore_ready_ns):
                violations.append(
                    f"{prefix}: decode-join barrier does not end at "
                    "restore-ready time")
        if int(request.arrival) + int(request.ttft) < restore_ready_ns:
            violations.append(
                f"{prefix}: first output precedes restore-ready dependency")
        restore_components = (
            int(request.agentic_kv_hbm_admission_wait_ns)
            + int(request.agentic_kv_restore_queue_wait_ns)
            + int(request.agentic_kv_restore_service_ns)
        )
        if int(request.agentic_kv_restore_ns) != restore_components:
            violations.append(
                f"{prefix}: physical restore total does not reconcile with "
                "HBM, queue, and service components")
        hit_tokens = int(request.agentic_kv_hit_tokens)
        restored_discarded = int(
            request
            .agentic_kv_restored_tokens_discarded_by_active_prefill_recompute)
        active_prefill_preemptions = int(
            request.active_prefill_recompute_preemptions)
        active_prefill_generation = int(
            request.pd_active_prefill_recompute_generation)
        active_prefill_tokens = int(request.active_prefill_recompute_tokens)
        active_prefill_frontier = int(
            request.active_prefill_recompute_frontier_tokens)
        if not 0 <= restored_discarded <= hit_tokens:
            violations.append(
                f"{prefix}: restored-hit discard is outside attempted hit "
                f"tokens ({restored_discarded}/{hit_tokens})")
        if (request.sub_request_index is not None
                and int(request.sub_request_index) > 0
                and hit_tokens > 0
                and str(request.agentic_kv_source) not in {
                    "hbm", "cpu", "ssd"
                }):
            violations.append(
                f"{prefix}: attempted restored hit lacks physical source "
                f"provenance ({request.agentic_kv_source!r})")
        if active_prefill_preemptions != active_prefill_generation:
            violations.append(
                f"{prefix}: active-prefill preemption count/generation "
                "diverge")
        if (active_prefill_tokens < active_prefill_frontier
                or active_prefill_frontier < 0):
            violations.append(
                f"{prefix}: cumulative active-prefill recompute tokens do "
                "not cover the replay frontier")
        if (active_prefill_preemptions == 0
                and (active_prefill_tokens != 0
                     or restored_discarded != 0)):
            violations.append(
                f"{prefix}: active-prefill work exists without a preemption")
        successful_chunk_wait = int(
            request.pd_chunk_successful_admission_wait_ns_total)
        cancelled_chunk_wait = int(
            request.pd_chunk_cancelled_admission_wait_ns_total)
        gross_chunk_wait = int(request.pd_chunk_admission_wait_ns_total)
        successful_chunk_critical = int(
            request.pd_chunk_successful_admission_critical_wait_ns_total)
        cancelled_chunk_critical = int(
            request.pd_chunk_cancelled_admission_critical_wait_ns_total)
        gross_chunk_critical = int(
            request.pd_chunk_admission_critical_wait_ns_total)
        if successful_chunk_wait + cancelled_chunk_wait != gross_chunk_wait:
            violations.append(
                f"{prefix}: successful/cancelled P/D wait does not "
                "reconcile with gross attempt wait")
        if (successful_chunk_critical + cancelled_chunk_critical
                != gross_chunk_critical):
            violations.append(
                f"{prefix}: successful/cancelled P/D critical wait does not "
                "reconcile with gross attempt critical wait")
        if int(request.pd_chunk_cancelled_admission_count) > (
                active_prefill_preemptions):
            violations.append(
                f"{prefix}: cancelled P/D claims exceed active-prefill "
                "preemptions")
        owner_gate_ns = (
            int(request.pd_pair_fifo_wait_ns)
            + int(request.agentic_kv_prepare_boundary_wait_ns)
            + int(request.agentic_kv_source_demotion_join_wait_ns)
            + int(request.agentic_kv_restore_ns)
        )
        if int(request.agentic_kv_owner_gate_ns) != owner_gate_ns:
            violations.append(
                f"{prefix}: owner-ready gate does not reconcile with P/D "
                "pair FIFO, prepare-boundary wait, source-demotion join, "
                "and physical restore")
        restore_issue_ns = int(request.agentic_kv_restore_issue_time_ns)
        target_hbm_ready_ns = int(
            request.agentic_kv_target_hbm_ready_time_ns)
        if restore_ready_ns != (
                restore_issue_ns + int(request.agentic_kv_restore_ns)):
            violations.append(
                f"{prefix}: restore-ready timestamp does not reconcile with "
                "issue time and restore duration")
        if restore_issue_ns != (
                int(request.arrival)
                + int(request.pd_pair_fifo_wait_ns)
                + int(request.agentic_kv_prepare_boundary_wait_ns)
                + int(request.agentic_kv_source_demotion_join_wait_ns)):
            violations.append(
                f"{prefix}: physical restore issue does not follow P/D pair "
                "FIFO, prepare-boundary admission, and source-demotion join")
        if restore_ready_ns != (
                int(request.arrival) + owner_gate_ns):
            violations.append(
                f"{prefix}: restore-ready timestamp does not reconcile with "
                "release-to-owner gate")
        if not restore_issue_ns <= target_hbm_ready_ns <= restore_ready_ns:
            violations.append(
                f"{prefix}: target-HBM-ready timestamp is outside the "
                "restore interval")
        if target_hbm_ready_ns != (
                restore_issue_ns
                + int(request.agentic_kv_hbm_admission_wait_ns)):
            violations.append(
                f"{prefix}: target-HBM-ready timestamp does not reconcile "
                "with issue time and HBM admission wait")
        if ((int(request.pd_pair_fifo_wait_ns)
                + int(request.agentic_kv_prepare_boundary_wait_ns)
                + int(request.agentic_kv_source_demotion_join_wait_ns)) > 0
                and int(request.first_schedule_eligibility_time_ns)
                < restore_issue_ns):
            violations.append(
                f"{prefix}: scheduler eligibility precedes pre-restore "
                "admission")
        if (int(request.pd_launch_admission_critical_wait_ns)
                > int(request.pd_launch_admission_wait_ns)):
            violations.append(
                f"{prefix}: critical P/D wait exceeds gross admission wait")
        if int(request.end_time) > int(simulated_duration_ns):
            violations.append(f"{prefix}: completion exceeds simulation time")
        if int(request.latency) == 0:
            add_warning(
                "zero_request_latency",
                f"{prefix}: zero request latency",
            )
        if int(request.latency) > 3_600_000_000_000:
            add_warning(
                "request_latency_over_one_hour",
                f"{prefix}: request latency exceeds one hour",
            )
    positive_latencies = sorted(
        int(request.latency) for request in requests
        if int(request.latency) > 0
    )
    if positive_latencies:
        median = _percentile(positive_latencies, 50)
        if median > 0 and positive_latencies[-1] / median > 1000:
            add_warning(
                "max_request_latency_over_p50_1000x",
                "maximum request latency exceeds p50 by more than 1000x",
            )

    by_session = defaultdict(list)
    for request in requests:
        by_session[str(request.session_id)].append(request)
    for session_id, session_requests in sorted(by_session.items()):
        ordered = sorted(
            session_requests,
            key=lambda request: int(request.sub_request_index),
        )
        indices = [int(request.sub_request_index) for request in ordered]
        expected_indices = list(range(len(ordered)))
        if indices != expected_indices:
            violations.append(
                f"session={session_id}: sub-request indices are not exactly "
                f"{expected_indices}; observed={indices}")
            continue
        lifecycle = lifecycle_by_session.get(session_id)
        if lifecycle is None:
            violations.append(
                f"session={session_id}: missing lifecycle record")
            continue
        offered_ns = int(lifecycle["offered_time_ns"])
        admission_ns = int(lifecycle["admission_time_ns"])
        completion_ns = int(lifecycle["completion_time_ns"])
        if not offered_ns <= admission_ns <= completion_ns:
            violations.append(
                f"session={session_id}: offered/admitted/completed "
                "timestamps are not monotonic")
        if int(lifecycle["admission_queue_wait_ns"]) != (
                admission_ns - offered_ns):
            violations.append(
                f"session={session_id}: admission queue wait does not "
                "reconcile")
        if int(lifecycle["e2e_ns"]) != completion_ns - admission_ns:
            violations.append(
                f"session={session_id}: admission-to-completion latency "
                "does not reconcile")
        if int(ordered[0].arrival) != int(lifecycle["admission_time_ns"]):
            violations.append(
                f"session={session_id}: initial request arrival does not "
                "match session admission")
        for previous, current in zip(ordered, ordered[1:]):
            expected_arrival = (
                int(previous.end_time) + int(current.return_gap_ns))
            if int(current.arrival) != expected_arrival:
                violations.append(
                    f"session={session_id}, request={current.id}: successor "
                    "arrival does not equal predecessor completion plus the "
                    f"incoming gap (arrival={current.arrival}, "
                    f"expected={expected_arrival})")
        if int(ordered[-1].end_time) != completion_ns:
            violations.append(
                f"session={session_id}: final request completion does not "
                "match session completion")
    return {
        "passed": not violations,
        "violations": violations,
        "warnings": warnings,
        "warning_codes": warning_codes,
        "checked_requests": len(requests),
    }


def _request_record(request):
    """Return the minimal exact record needed for offline cohort validation."""
    return {
        "request_id": int(request.id),
        "session_id": str(request.session_id),
        "sub_request_index": int(request.sub_request_index),
        "source_session_id": str(request.source_session_id),
        "session_template_index": int(request.session_template_index),
        "session_epoch": int(request.session_epoch),
        "input_tokens": int(request.original_input),
        "requested_output_tokens": int(request.requested_output_tokens),
        "generated_tokens": int(request.generated_tokens),
        "arrival_time_ns": int(request.arrival),
        "end_time_ns": int(request.end_time),
        "first_schedule_time_ns": int(request.first_schedule_time_ns),
        "ttft_ns": int(request.ttft),
        "tpot_ns": int(request.tpot),
        "itl_ns": [int(value) for value in request.itl],
        "prefix_reuse_tokens": int(request.prefix_reuse_tokens),
        "agentic_kv_hit_tokens": int(request.agentic_kv_hit_tokens),
        "agentic_kv_recompute_tokens": int(
            request.agentic_kv_recompute_tokens),
        "return_gap_type": str(request.return_gap_type),
        "return_gap_ns": int(request.return_gap_ns),
        "agentic_kv_residency_at_return": (
            None if request.agentic_kv_residency_at_return is None
            else str(request.agentic_kv_residency_at_return)
        ),
        "agentic_kv_source": (
            None if request.agentic_kv_source is None
            else str(request.agentic_kv_source)
        ),
        "agentic_kv_restore_issue_time_ns": int(
            request.agentic_kv_restore_issue_time_ns),
        "agentic_kv_target_hbm_ready_time_ns": int(
            request.agentic_kv_target_hbm_ready_time_ns),
        "agentic_kv_restore_ready_time_ns": int(
            request.agentic_kv_restore_ready_time_ns),
        "agentic_kv_restore_ns": int(request.agentic_kv_restore_ns),
        "agentic_kv_owner_gate_ns": int(
            request.agentic_kv_owner_gate_ns),
        "pd_pair_fifo_wait_ns": int(request.pd_pair_fifo_wait_ns),
        "agentic_kv_prepare_boundary_wait_ns": int(
            request.agentic_kv_prepare_boundary_wait_ns),
        "agentic_kv_source_demotion_join_wait_ns": int(
            request.agentic_kv_source_demotion_join_wait_ns),
        "agentic_kv_hbm_admission_wait_ns": int(
            request.agentic_kv_hbm_admission_wait_ns),
        "agentic_kv_transient_dram_capacity_wait_ns": int(
            request.agentic_kv_transient_dram_capacity_wait_ns),
        "agentic_kv_restore_queue_wait_ns": int(
            request.agentic_kv_restore_queue_wait_ns),
        "agentic_kv_restore_service_ns": int(
            request.agentic_kv_restore_service_ns),
        "pd_chunk_admission_count": int(
            request.pd_chunk_admission_count),
        "pd_chunk_cancelled_admission_count": int(
            request.pd_chunk_cancelled_admission_count),
        "pd_chunk_admission_wait_ns_total": int(
            request.pd_chunk_admission_wait_ns_total),
        "pd_chunk_admission_critical_wait_ns_total": int(
            request.pd_chunk_admission_critical_wait_ns_total),
        "pd_chunk_successful_admission_wait_ns_total": int(
            request.pd_chunk_successful_admission_wait_ns_total),
        "pd_chunk_successful_admission_critical_wait_ns_total": int(
            request.pd_chunk_successful_admission_critical_wait_ns_total),
        "pd_chunk_cancelled_admission_wait_ns_total": int(
            request.pd_chunk_cancelled_admission_wait_ns_total),
        "pd_chunk_cancelled_admission_critical_wait_ns_total": int(
            request.pd_chunk_cancelled_admission_critical_wait_ns_total),
        "active_prefill_recompute_preemptions": int(
            request.active_prefill_recompute_preemptions),
        "active_prefill_recompute_tokens": int(
            request.active_prefill_recompute_tokens),
        "active_prefill_recompute_frontier_tokens": int(
            request.active_prefill_recompute_frontier_tokens),
        "pd_active_prefill_recompute_generation": int(
            request.pd_active_prefill_recompute_generation),
        "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute": int(
            request
            .agentic_kv_restored_tokens_discarded_by_active_prefill_recompute),
    }


def _active_population(lifecycle, start_ns, end_ns, configured_k):
    if start_ns is None or end_ns is None or end_ns <= start_ns:
        return {
            "mean_active_sessions": None,
            "peak_active_sessions": None,
            "fraction_at_configured_k": None,
        }

    start_ns = int(start_ns)
    end_ns = int(end_ns)
    events = defaultdict(int)
    for row in lifecycle:
        admission = row.get("admission_time_ns")
        completion = row.get("completion_time_ns")
        censored = row.get("censored_time_ns")
        if admission is None:
            continue
        admission = int(admission)
        if completion is not None:
            terminal = int(completion)
        elif censored is not None:
            terminal = int(censored)
        else:
            # An admitted session that is still live at report time remains
            # active through the end of the measurement window.
            terminal = end_ns

        # Treat residency as a half-open [admission, terminal) interval and
        # clip it to the measurement window.  This includes pre-window
        # admissions and post-window completions without allowing invalid or
        # zero-length lifecycle rows to perturb the population integral.
        interval_start = max(start_ns, admission)
        interval_end = min(end_ns, terminal)
        if interval_end <= interval_start:
            continue
        events[interval_start] += 1
        events[interval_end] -= 1

    active = 0
    active_time = 0
    at_k_time = 0
    peak = 0
    previous = start_ns
    for timestamp in sorted(events):
        duration = timestamp - previous
        active_time += active * duration
        if configured_k > 0 and active == configured_k:
            at_k_time += duration
        active += events[timestamp]
        peak = max(peak, active)
        previous = timestamp
    window = end_ns - start_ns
    return {
        "mean_active_sessions": active_time / window,
        "peak_active_sessions": peak,
        "fraction_at_configured_k": (
            at_k_time / window if configured_k > 0 else None
        ),
    }


def _realized_offer_rate(lifecycle):
    offered = sorted(
        int(row["offered_time_ns"])
        for row in lifecycle
        if row.get("offered_time_ns") is not None
    )
    if len(offered) < 2 or offered[-1] <= offered[0]:
        return None
    return (len(offered) - 1) * 1_000_000_000 / (offered[-1] - offered[0])


def build_session_metrics(
        router, schedulers, simulated_duration_ns, *, dataset=None, run_id=None,
        measurement_early_stopped=False, online_compute=None,
        oracle_validation=None, censoring=None,
        hbm_occupancy_accounting=None):
    """Build one auditable report from completed online-simulator requests."""
    all_requests = [
        request
        for scheduler in schedulers
        if scheduler.pd_type != "prefill"
        for request in scheduler.done
        if request.session_id is not None
    ]
    all_requests.sort(key=lambda request: (request.end_time, request.id))
    lifecycle = router.session_lifecycle_records()
    lifecycle_by_session = {
        str(row["session_id"]): row for row in lifecycle
    }
    completed_sessions_all = sorted(
        (
            row for row in lifecycle
            if row.get("status") == "completed"
        ),
        key=lambda row: (row["completion_time_ns"], row["session_id"]),
    )
    admission = router.session_admission
    cohort_selection = str(getattr(
        admission, "measurement_cohort_selection", "completion_order",
    )).strip().lower()
    if cohort_selection == "admission_order":
        target_accessor = getattr(
            router, "measurement_target_session_ids", None)
        if target_accessor is None:
            raise RuntimeError(
                "admission_order metrics require Router fixed-target IDs")
        measurement_target_session_ids = [
            str(session_id) for session_id in target_accessor()
        ]
        warmup_accessor = getattr(
            router, "measurement_warmup_session_ids", None)
        required_accessor = getattr(
            router, "measurement_required_session_ids", None)
        if warmup_accessor is None:
            if int(admission.warmup_completions) != 0:
                raise RuntimeError(
                    "Admission-order warmup metrics require Router fixed "
                    "admission-prefix tracking")
            measurement_warmup_session_ids = []
        else:
            measurement_warmup_session_ids = [
                str(session_id) for session_id in warmup_accessor()
            ]
        if required_accessor is None:
            measurement_required_session_ids = (
                measurement_warmup_session_ids
                + measurement_target_session_ids
            )
        else:
            measurement_required_session_ids = [
                str(session_id) for session_id in required_accessor()
            ]
        if len(measurement_target_session_ids) != len(set(
                measurement_target_session_ids)):
            raise RuntimeError(
                "Admission-order measurement target IDs are not unique")
        if len(measurement_warmup_session_ids) != len(set(
                measurement_warmup_session_ids)):
            raise RuntimeError(
                "Admission-order warmup-prefix IDs are not unique")
        expected_required_ids = (
            measurement_warmup_session_ids
            + measurement_target_session_ids
        )
        if measurement_required_session_ids != expected_required_ids:
            raise RuntimeError(
                "Admission-order required IDs must be the ordered warmup "
                "prefix followed by the measurement target")
        if len(measurement_required_session_ids) != len(set(
                measurement_required_session_ids)):
            raise RuntimeError(
                "Admission-order warmup and target IDs overlap")
        if (len(measurement_warmup_session_ids)
                != int(admission.warmup_completions)):
            raise RuntimeError(
                "Admission-order warmup-prefix count does not match "
                "warmup_completions")
        if (admission.measure_completions > 0
                and len(measurement_target_session_ids)
                != int(admission.measure_completions)):
            raise RuntimeError(
                "Admission-order measurement target count does not match "
                "measure_completions")
        missing_required = [
            session_id for session_id in measurement_required_session_ids
            if session_id not in lifecycle_by_session
        ]
        if missing_required:
            raise RuntimeError(
                "Admission-order required-prefix sessions are absent from "
                f"the session lifecycle: {missing_required[:5]}")
        warmup_lifecycle = [
            lifecycle_by_session[session_id]
            for session_id in measurement_warmup_session_ids
        ]
        target_lifecycle = [
            lifecycle_by_session[session_id]
            for session_id in measurement_target_session_ids
        ]
        completed_sessions = [
            row for row in target_lifecycle
            if row.get("status") == "completed"
        ]
        completed_warmup_sessions = [
            row for row in warmup_lifecycle
            if row.get("status") == "completed"
        ]
        completed_required_sessions = [
            lifecycle_by_session[session_id]
            for session_id in measurement_required_session_ids
            if lifecycle_by_session[session_id].get("status") == "completed"
        ]
        warmup_count = len(completed_warmup_sessions)
        measurement_start_ns = min(
            (
                int(row["admission_time_ns"])
                for row in target_lifecycle
                if row.get("admission_time_ns") is not None
            ),
            default=None,
        )
        measurement_end_ns = max(
            (
                int(row["completion_time_ns"])
                for row in target_lifecycle
                if row.get("completion_time_ns") is not None
            ),
            default=None,
        )
        measurement_complete = (
            bool(measurement_target_session_ids)
            and len(completed_sessions)
            == len(measurement_target_session_ids)
        )
        target_semantics = (
            "Before execution, the first warmup_completions runtime "
            "sessions in deterministic epoch-major backlog admission order "
            "are fixed as an excluded admission prefix; the immediately "
            "following measure_completions sessions are the measured target. "
            "This is not a temporal warmup barrier."
        )
        start_semantics = (
            "minimum admission timestamp among fixed target sessions"
        )
        end_semantics = (
            "maximum completion timestamp among fixed target sessions"
        )
    elif cohort_selection == "completion_order":
        warmup_count = min(
            admission.warmup_completions, len(completed_sessions_all))
        remaining_sessions = completed_sessions_all[warmup_count:]
        if admission.measure_completions > 0:
            completed_sessions = remaining_sessions[
                :admission.measure_completions]
        else:
            completed_sessions = remaining_sessions
        measurement_target_session_ids = [
            str(row["session_id"]) for row in completed_sessions
        ]
        measurement_warmup_session_ids = [
            str(row["session_id"])
            for row in completed_sessions_all[:warmup_count]
        ]
        measurement_required_session_ids = (
            measurement_warmup_session_ids
            + measurement_target_session_ids
        )
        completed_warmup_sessions = completed_sessions_all[:warmup_count]
        completed_required_sessions = (
            completed_warmup_sessions + completed_sessions
        )
        if warmup_count > 0:
            measurement_start_ns = int(
                completed_sessions_all[
                    warmup_count - 1]["completion_time_ns"])
        else:
            measurement_start_ns = min(
                (
                    int(row["admission_time_ns"])
                    for row in completed_sessions
                ),
                default=None,
            )
        measurement_end_ns = max(
            (
                int(row["completion_time_ns"])
                for row in completed_sessions
            ),
            default=None,
        )
        measurement_complete = (
            admission.measure_completions == 0
            or len(completed_sessions) == admission.measure_completions
        )
        target_semantics = (
            "The measured cohort is selected after execution by completion "
            "order, after excluding the configured completion-count warmup."
        )
        start_semantics = (
            "completion of the final excluded warmup session"
            if warmup_count > 0
            else "first admission among measured sessions"
        )
        end_semantics = "completion of the final measured session"
    else:
        raise RuntimeError(
            "Unknown measurement cohort selection "
            f"{cohort_selection!r}")
    warmup_session_id_set = set(measurement_warmup_session_ids)
    target_session_id_set = set(measurement_target_session_ids)
    required_session_id_set = set(measurement_required_session_ids)
    prefix_id_overlap_count = len(
        warmup_session_id_set & target_session_id_set)
    warmup_completion_boundary_ns = max(
        (
            int(row["completion_time_ns"])
            for row in completed_warmup_sessions
            if row.get("completion_time_ns") is not None
        ),
        default=None,
    )
    target_admitted_before_warmup_complete = (
        sum(
            int(row["admission_time_ns"]) < warmup_completion_boundary_ns
            for row in (
                lifecycle_by_session[session_id]
                for session_id in measurement_target_session_ids
            )
            if row.get("admission_time_ns") is not None
        )
        if warmup_completion_boundary_ns is not None else 0
    )
    target_completed_before_warmup_complete = (
        sum(
            int(row["completion_time_ns"]) < warmup_completion_boundary_ns
            for row in completed_sessions
            if row.get("completion_time_ns") is not None
        )
        if warmup_completion_boundary_ns is not None else 0
    )
    measured_session_ids = {
        str(session_id) for session_id in measurement_target_session_ids
    }
    requests = [
        request for request in all_requests
        if str(request.session_id) in measured_session_ids
    ]
    initial_requests = [
        request for request in requests
        if request.sub_request_index is not None
        and int(request.sub_request_index) == 0
    ]
    resumes = [
        request for request in requests
        if request.sub_request_index is not None
        and int(request.sub_request_index) > 0
    ]

    by_gap = defaultdict(list)
    by_residency = defaultdict(list)
    by_source = defaultdict(list)
    attempted_by_source = defaultdict(list)
    effective_surviving_by_source = defaultdict(list)
    physical_resume_sources = ("hbm", "cpu", "ssd")
    for request in resumes:
        by_gap[str(request.return_gap_type or "unknown")].append(request)
        by_residency[str(
            request.agentic_kv_residency_at_return or "unknown"
        )].append(request)
        source = str(request.agentic_kv_source or "unknown")
        by_source[source].append(request)
        hit_tokens = int(request.agentic_kv_hit_tokens)
        restored_discarded = int(
            request
            .agentic_kv_restored_tokens_discarded_by_active_prefill_recompute)
        if source in physical_resume_sources and hit_tokens > 0:
            attempted_by_source[source].append(request)
            if hit_tokens - restored_discarded > 0:
                effective_surviving_by_source[source].append(request)

    measurement_duration_ns = (
        measurement_end_ns - measurement_start_ns
        if measurement_start_ns is not None and measurement_end_ns is not None
        else None
    )
    hbm_kv_occupancy = None
    if hbm_occupancy_accounting is not None:
        if measurement_start_ns is None or measurement_end_ns is None:
            raise RuntimeError(
                "HBM occupancy accounting requires a complete measurement "
                "window")
        hbm_kv_occupancy = hbm_occupancy_accounting.summary(
            measurement_start_ns, measurement_end_ns)
    temporal_window_requests = [
        request for request in all_requests
        if measurement_start_ns is not None
        and measurement_end_ns is not None
        and measurement_start_ns < int(request.end_time) <= measurement_end_ns
    ]
    # A fixed admission-order cohort is a set-valued experiment definition,
    # not a temporal slice. Keep all headline request/token metrics on that
    # target set even when its execution overlaps unfinished warmup sessions.
    window_requests = (
        requests
        if cohort_selection == "admission_order"
        else temporal_window_requests
    )
    lifecycle_records = [
        {
            **row,
            "measurement_warmup": (
                str(row["session_id"]) in warmup_session_id_set),
            "measurement_target": (
                str(row["session_id"]) in target_session_id_set),
            "measurement_required": (
                str(row["session_id"]) in required_session_id_set),
            "measurement_role": (
                "fixed_admission_prefix_warmup"
                if str(row["session_id"]) in warmup_session_id_set else
                "measurement_target"
                if str(row["session_id"]) in target_session_id_set else
                "outside_required_prefix"
                if cohort_selection == "admission_order" else
                "outside_measurement_cohort"
            ),
            "measurement_included": (
                str(row["session_id"]) in measured_session_ids),
        }
        for row in lifecycle
    ]
    offered_arrival_trace = [
        {
            "session_id": str(row["session_id"]),
            "offered_time_ns": int(row["offered_time_ns"]),
        }
        for row in lifecycle
    ]
    cohort_generated_tokens = sum(
        max(0, int(request.output) - int(request.input))
        for request in requests
    )
    cohort_prompt_tokens = sum(int(request.input) for request in requests)
    generated_tokens = sum(
        max(0, int(request.output) - int(request.input))
        for request in window_requests
    )
    prompt_tokens = sum(int(request.input) for request in window_requests)
    source_counts = Counter(
        str(request.agentic_kv_source or "initial_or_no_resume")
        for request in requests
    )
    resume_denominator = len(resumes)
    attempted_source_counts = {
        source: len(attempted_by_source[source])
        for source in physical_resume_sources
    }
    effective_source_counts = {
        source: len(effective_surviving_by_source[source])
        for source in physical_resume_sources
    }
    attempted_resume_count = sum(attempted_source_counts.values())
    effective_surviving_resume_count = sum(
        effective_source_counts.values())
    kv_state_unavailable_resumes = [
        request for request in resumes
        if (str(request.agentic_kv_source or "unknown") == "dropped"
            and int(request.agentic_kv_recompute_tokens) > 0)
    ]
    zero_overlap_resumes = [
        request for request in resumes
        if (int(request.agentic_kv_hit_tokens) == 0
            and int(request.agentic_kv_recompute_tokens) == 0)
    ]
    attempted_resume_requests = [
        request
        for source in physical_resume_sources
        for request in attempted_by_source[source]
    ]
    effective_surviving_resume_requests = [
        request
        for source in physical_resume_sources
        for request in effective_surviving_by_source[source]
    ]
    attempted_restored_hit_tokens = sum(
        int(request.agentic_kv_hit_tokens)
        for source in physical_resume_sources
        for request in attempted_by_source[source]
    )
    restored_hit_tokens_discarded = sum(
        int(request
            .agentic_kv_restored_tokens_discarded_by_active_prefill_recompute)
        for source in physical_resume_sources
        for request in attempted_by_source[source]
    )
    effective_surviving_hit_tokens = (
        attempted_restored_hit_tokens - restored_hit_tokens_discarded)
    cohort_pair_fifo_wait_ns = sum(
        int(request.pd_pair_fifo_wait_ns) for request in requests)
    cohort_prepare_boundary_wait_ns = sum(
        int(request.agentic_kv_prepare_boundary_wait_ns)
        for request in requests)
    cohort_source_demotion_join_wait_ns = sum(
        int(request.agentic_kv_source_demotion_join_wait_ns)
        for request in requests)
    cohort_request_latency_ns = sum(
        int(request.latency) for request in requests)
    window_pair_fifo_wait_ns = sum(
        int(request.pd_pair_fifo_wait_ns) for request in window_requests)
    window_prepare_boundary_wait_ns = sum(
        int(request.agentic_kv_prepare_boundary_wait_ns)
        for request in window_requests)
    window_source_demotion_join_wait_ns = sum(
        int(request.agentic_kv_source_demotion_join_wait_ns)
        for request in window_requests)
    window_request_latency_ns = sum(
        int(request.latency) for request in window_requests)
    timing_validation = _timing_validation(
        requests, simulated_duration_ns, lifecycle_by_session)
    if not timing_validation["passed"]:
        raise RuntimeError(
            "Online session timing validation failed: "
            + "; ".join(timing_validation["violations"][:10]))

    return {
        "schema_version": 11,
        "run_id": run_id,
        "dataset": dataset,
        "session_admission": router.session_admission_summary(),
        "measurement_window": {
            "simulated_duration_ns": int(simulated_duration_ns),
            "measurement_cohort_selection": cohort_selection,
            "measurement_warmup_session_ids": (
                measurement_warmup_session_ids),
            "measurement_warmup_session_count": len(
                measurement_warmup_session_ids),
            "measurement_warmup_completed_sessions": len(
                completed_warmup_sessions),
            "measurement_warmup_session_ids_hash": _stable_json_hash(
                measurement_warmup_session_ids),
            "measurement_target_session_ids": (
                measurement_target_session_ids),
            "measurement_target_session_count": len(
                measurement_target_session_ids),
            "measurement_target_completed_sessions": len(
                completed_sessions),
            "measurement_target_session_ids_hash": _stable_json_hash(
                measurement_target_session_ids),
            "measurement_required_session_ids": (
                measurement_required_session_ids),
            "measurement_required_session_count": len(
                measurement_required_session_ids),
            "measurement_required_completed_sessions": len(
                completed_required_sessions),
            "measurement_required_session_ids_hash": _stable_json_hash(
                measurement_required_session_ids),
            "measurement_prefix_id_overlap_count": (
                prefix_id_overlap_count),
            "warmup_completion_boundary_ns": (
                warmup_completion_boundary_ns),
            "target_admitted_before_warmup_complete_session_count": (
                target_admitted_before_warmup_complete),
            "target_completed_before_warmup_complete_session_count": (
                target_completed_before_warmup_complete),
            "target_execution_overlapped_unfinished_warmup": (
                target_admitted_before_warmup_complete > 0),
            "target_semantics": target_semantics,
            "target_order_and_hash_semantics": (
                "Session IDs retain selection order; the hash is SHA-256 "
                "over their canonical compact JSON list."
            ),
            "warmup_completions_requested": admission.warmup_completions,
            "warmup_completions_observed": warmup_count,
            "warmup_complete": (
                warmup_count == admission.warmup_completions
            ),
            "measure_completions_requested": admission.measure_completions,
            "measure_completions_observed": len(completed_sessions),
            "measurement_complete": measurement_complete,
            "measurement_boundary_complete": (
                len(completed_required_sessions)
                == len(measurement_required_session_ids)
                and measurement_complete
            ),
            "measurement_early_stopped": bool(measurement_early_stopped),
            "measurement_start_ns": measurement_start_ns,
            "measurement_end_ns": measurement_end_ns,
            "measurement_duration_ns": measurement_duration_ns,
            "start_semantics": start_semantics,
            "end_semantics": end_semantics,
        },
        "throughput": {
            "completed_sessions_total": len(completed_sessions_all),
            "completed_requests_total": len(all_requests),
            "completed_sessions": len(completed_sessions),
            "completed_requests": len(window_requests),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "completed_requests_in_session_cohort": len(requests),
            "prompt_tokens_in_session_cohort": cohort_prompt_tokens,
            "generated_tokens_in_session_cohort": cohort_generated_tokens,
            "sessions_per_second_measurement_window": _rate(
                len(completed_sessions), measurement_duration_ns
            ),
            "requests_per_second_measurement_window": _rate(
                len(window_requests), measurement_duration_ns
            ),
            "total_tokens_per_second_measurement_window": _rate(
                prompt_tokens + generated_tokens, measurement_duration_ns
            ),
            "sessions_per_second_full_simulation": _rate(
                len(completed_sessions_all), int(simulated_duration_ns)
            ),
            "requests_per_second_full_simulation": _rate(
                len(all_requests), int(simulated_duration_ns)
            ),
            "realized_session_offer_rate_sps": _realized_offer_rate(lifecycle),
        },
        "active_session_population": _active_population(
            lifecycle,
            measurement_start_ns,
            measurement_end_ns,
            admission.max_active_sessions,
        ),
        "overhead_denominators": {
            "measured_session_cohort": {
                "request_count": len(requests),
                "request_latency_ns": cohort_request_latency_ns,
                "pd_pair_fifo_wait_ns": cohort_pair_fifo_wait_ns,
                "pd_pair_fifo_fraction_of_request_latency": (
                    cohort_pair_fifo_wait_ns / cohort_request_latency_ns
                    if cohort_request_latency_ns > 0 else None
                ),
                "prepare_boundary_wait_ns": (
                    cohort_prepare_boundary_wait_ns),
                "prepare_boundary_fraction_of_request_latency": (
                    cohort_prepare_boundary_wait_ns
                    / cohort_request_latency_ns
                    if cohort_request_latency_ns > 0 else None
                ),
                "source_demotion_join_wait_ns": (
                    cohort_source_demotion_join_wait_ns),
                "source_demotion_join_fraction_of_request_latency": (
                    cohort_source_demotion_join_wait_ns
                    / cohort_request_latency_ns
                    if cohort_request_latency_ns > 0 else None
                ),
                "scope": (
                    "Every completed LLM request belonging to the completed "
                    "measured-session cohort. For fixed admission-order "
                    "selection, excluded warmup sessions may execute "
                    "concurrently but never enter this denominator; this is "
                    "the latency cohort used by requests.all and the paper "
                    "latency CDFs."
                ),
            },
            "strict_completion_window": {
                "request_count": len(window_requests),
                "request_latency_ns": window_request_latency_ns,
                "pd_pair_fifo_wait_ns": window_pair_fifo_wait_ns,
                "pd_pair_fifo_fraction_of_request_latency": (
                    window_pair_fifo_wait_ns / window_request_latency_ns
                    if window_request_latency_ns > 0 else None
                ),
                "prepare_boundary_wait_ns": (
                    window_prepare_boundary_wait_ns),
                "prepare_boundary_fraction_of_request_latency": (
                    window_prepare_boundary_wait_ns
                    / window_request_latency_ns
                    if window_request_latency_ns > 0 else None
                ),
                "source_demotion_join_wait_ns": (
                    window_source_demotion_join_wait_ns),
                "source_demotion_join_fraction_of_request_latency": (
                    window_source_demotion_join_wait_ns
                    / window_request_latency_ns
                    if window_request_latency_ns > 0 else None
                ),
                "scope": (
                    "For completion-order selection, all requests whose "
                    "completion timestamp lies in (measurement_start_ns, "
                    "measurement_end_ns]. For fixed admission-order "
                    "selection, exactly the target-session requests; the "
                    "headline request throughput never counts overlapping "
                    "warmup work."
                ),
            },
        },
        "sessions": {
            "offered_arrival_trace_count": len(offered_arrival_trace),
            "offered_arrival_trace_sha256": _stable_json_hash(
                offered_arrival_trace),
            "offered_arrival_trace_semantics": (
                "SHA-256 over the router's deterministic lifecycle order; "
                "each record contains runtime session identity and "
                "offered_time_ns. Paired policies at one load and seed "
                "must match exactly."
            ),
            "admission_queue_wait_ns": _distribution(
                row.get("admission_queue_wait_ns")
                for row in completed_sessions
            ),
            "e2e_from_admission_ns": _distribution(
                row.get("e2e_ns") for row in completed_sessions
            ),
            "e2e_from_offer_ns": _distribution(
                int(row["completion_time_ns"]) - int(row["offered_time_ns"])
                for row in completed_sessions
            ),
            "records": lifecycle_records,
        },
        "requests": {
            "records": [
                _request_record(request)
                for request in sorted(
                    requests,
                    key=lambda request: (
                        str(request.session_id),
                        int(request.sub_request_index),
                    ),
                )
            ],
            "all": _request_group(requests),
            "initial": _request_group(initial_requests),
            "resume": _request_group(resumes),
            "resume_by_return_gap_type": {
                key: _request_group(group)
                for key, group in sorted(by_gap.items())
            },
            "resume_by_residency_at_return": {
                key: _request_group(group)
                for key, group in sorted(by_residency.items())
            },
            "resume_by_source": {
                key: _request_group(group)
                for key, group in sorted(by_source.items())
            },
            "attempted_physical_resume_by_source": {
                source: _request_group(attempted_by_source[source])
                for source in physical_resume_sources
            },
            "effective_surviving_resume_by_source": {
                source: _request_group(
                    effective_surviving_by_source[source])
                for source in physical_resume_sources
            },
            "attempted_physical_resume_by_return_gap_type_and_source": (
                _cross_request_groups(
                    attempted_resume_requests,
                    lambda request: request.return_gap_type,
                    lambda request: request.agentic_kv_source,
                )
            ),
            "effective_surviving_resume_by_return_gap_type_and_source": (
                _cross_request_groups(
                    effective_surviving_resume_requests,
                    lambda request: request.return_gap_type,
                    lambda request: request.agentic_kv_source,
                )
            ),
            "resume_by_return_gap_type_and_residency_at_return": (
                _cross_request_groups(
                    resumes,
                    lambda request: request.return_gap_type,
                    lambda request: request.agentic_kv_residency_at_return,
                )
            ),
            "resume_by_return_gap_type_and_source": (
                _cross_request_groups(
                    resumes,
                    lambda request: request.return_gap_type,
                    lambda request: request.agentic_kv_source,
                )
            ),
            "source_counts_all_requests": dict(sorted(source_counts.items())),
            "source_fractions_of_all_requests": {
                key: value / len(requests) if requests else None
                for key, value in sorted(source_counts.items())
            },
            "resume_source_fractions_of_all_requests": {
                key: len(group) / len(requests) if requests else None
                for key, group in sorted(by_source.items())
            },
            "resume_source_fractions_of_resume_requests": {
                key: len(group) / resume_denominator
                if resume_denominator else None
                for key, group in sorted(by_source.items())
            },
            "attempted_physical_resume_count": attempted_resume_count,
            "effective_surviving_resume_count": (
                effective_surviving_resume_count),
            "attempted_physical_resume_counts_by_source": (
                attempted_source_counts),
            "effective_surviving_resume_counts_by_source": (
                effective_source_counts),
            "attempted_physical_resume_fractions_of_all_requests": {
                source: (
                    attempted_source_counts[source] / len(requests)
                    if requests else None)
                for source in physical_resume_sources
            },
            "effective_surviving_resume_fractions_of_all_requests": {
                source: (
                    effective_source_counts[source] / len(requests)
                    if requests else None)
                for source in physical_resume_sources
            },
            "attempted_physical_resume_fraction_of_all_requests": (
                attempted_resume_count / len(requests)
                if requests else None),
            "effective_surviving_resume_fraction_of_all_requests": (
                effective_surviving_resume_count / len(requests)
                if requests else None),
            "kv_state_unavailable_resume_count": len(
                kv_state_unavailable_resumes),
            "kv_state_unavailable_resume_fraction_of_all_requests": (
                len(kv_state_unavailable_resumes) / len(requests)
                if requests else None),
            "zero_overlap_resume_count": len(zero_overlap_resumes),
            "zero_overlap_resume_fraction_of_all_requests": (
                len(zero_overlap_resumes) / len(requests)
                if requests else None),
            "resume_reuse_token_accounting": {
                "attempted_restored_hit_tokens": (
                    attempted_restored_hit_tokens),
                "restored_hit_tokens_discarded_by_active_prefill_recompute": (
                    restored_hit_tokens_discarded),
                "effective_surviving_hit_tokens": (
                    effective_surviving_hit_tokens),
                "conservation_passed": (
                    attempted_restored_hit_tokens
                    == restored_hit_tokens_discarded
                    + effective_surviving_hit_tokens),
            },
        },
        "denominator_notes": {
            "all_requests": (
                "All completed non-prefill requests belonging to sessions in "
                "the selected measurement cohort."
            ),
            "resume_requests": (
                "Completed session requests with sub_request_index > 0."
            ),
            "resume_source_fractions_of_all_requests": (
                "The denominator is every completed session request, including "
                "initial calls and returns without reusable KV, within the "
                "selected measurement cohort."
            ),
            "attempted_vs_effective_resume": (
                "Attempted physical source counts retain the HBM/CPU/SSD "
                "I/O provenance selected at request return. Effective-"
                "surviving counts require at least one originally restored "
                "hit token to remain after any active-prefill preemption. "
                "Both reported fractions use every completed measured-"
                "cohort request as the denominator."
            ),
            "kv_state_unavailable_resume": (
                "A dropped manager-source label counts as unavailable only "
                "when the continuation has positive recompute tokens. "
                "Zero-overlap returns are excluded."
            ),
            "ttft": (
                "Request-ready to first output completion; it includes restore, "
                "HBM admission, scheduler queueing, and prompt compute on the "
                "request's critical path."
            ),
            "release_to_first_schedule": (
                "Request release to first model schedule. It intentionally "
                "includes restore, HBM admission, P/D admission, and scheduler "
                "waiting that occur before first compute eligibility. Component "
                "wait fields may overlap and must not be added together."
            ),
            "transient_dram_capacity_wait": (
                "A non-additive subset of restore_queue_wait_ns for SSD "
                "resumes. It is lower-tier admission time spent waiting for "
                "full-object node-DRAM bounce capacity and must not be added "
                "again to restore or TTFT components."
            ),
            "source_demotion_join_wait": (
                "The request-visible tail of an already-issued asynchronous "
                "capacity demotion when the owning session returns before its "
                "atomic lower-tier commit. It is additive in owner_ready_gate, "
                "but the full background swap-out service is not added again."
            ),
            "scheduler_queue_wait": (
                "Time from final request-specific scheduler eligibility to "
                "first model schedule. Restore, HBM reclaim, and atomic P/D "
                "launch admission finish before this boundary, so this field "
                "is the non-overlapping pure runnable-queue component."
            ),
            "measurement_cohort": (
                "Requests belong to the exact session-ID list recorded in "
                "measurement_window.measurement_target_session_ids. The "
                "selection rule is recorded by measurement_cohort_selection."
            ),
            "throughput_request_window": (
                "Request and token throughput counts completion events in "
                "the half-open measurement interval (start, end]. It is "
                "separate from the completion-selected session cohort so "
                "requests executed before warmup are not charged to the "
                "steady-state window."
            ),
        },
        "online_model_compute": online_compute,
        "hbm_kv_occupancy": hbm_kv_occupancy,
        "strict_infinite_hbm_oracle": oracle_validation,
        "censoring": censoring,
        "validation": {
            "timing": timing_validation,
            "dependency_semantics": (
                "Every completed lower-tier resume is checked for positive "
                "transfer service and every request must complete after its "
                "restore-ready dependency."
            ),
        },
    }


def save_session_metrics(report, path):
    """Write a session metrics report as stable, human-readable JSON."""
    if not os.path.isabs(path):
        path = os.path.join("..", path)
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")

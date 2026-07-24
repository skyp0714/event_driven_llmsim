import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from serving.live_astra_comparison_collect import (
    LiveAstraCollectError,
    _measurement_resume_arrival_guard,
    collect_campaign,
    write_compact_csv,
    write_compact_json,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _stable_json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def test_runtime_guard_rejects_one_late_measurement_resume():
    cutoff = 10_000
    resumes = tuple(
        {
            "session_id": f"measurement-{index:03d}",
            "call_index": 1,
            "arrival_ns": cutoff,
        }
        for index in range(112)
    )
    contract = {
        "last_external_guard_offer_ns": cutoff,
        "expected_measurement_resume_count": 112,
    }
    summary = _measurement_resume_arrival_guard(
        resumes, contract, cell_id="stress-cell")
    assert summary["observed_measurement_resume_count"] == 112
    assert summary["arrived_after_last_external_guard_offer_count"] == 0
    assert summary[
        "all_measurement_resumes_arrived_by_last_external_guard_offer"
    ] is True

    late = list(resumes)
    late[-1] = {**late[-1], "arrival_ns": cutoff + 1}
    with pytest.raises(
            LiveAstraCollectError,
            match="1/112 measurement resume arrivals after"):
        _measurement_resume_arrival_guard(
            tuple(late), contract, cell_id="stress-cell")


def _metrics() -> dict[str, object]:
    scale = 1_000_000_000 / 190
    distribution_ttft = {
        "count": 1,
        "mean_ns": 50.0,
        "minimum_ns": 50.0,
        "p50_ns": 50.0,
        "p95_ns": 50.0,
        "p99_ns": 50.0,
        "maximum_ns": 50.0,
        "percentile_method": "inclusive_nearest_rank",
    }
    distribution_tpot = {
        "count": 1,
        "mean_ns": 20.0,
        "minimum_ns": 20.0,
        "p50_ns": 20.0,
        "p95_ns": 20.0,
        "p99_ns": 20.0,
        "maximum_ns": 20.0,
        "percentile_method": "inclusive_nearest_rank",
    }
    all_tpot = {
        **distribution_tpot,
        "count": 2,
        "mean_ns": 20.0,
    }
    return {
        "measurement_session_ids": ["s1"],
        "measurement_request_count": 2,
        "resume_request_count": 1,
        "tpot_eligible_request_count": 2,
        "resume_tpot_eligible_request_count": 1,
        "ttft_slo_ns": 60,
        "tpot_slo_ns": 30,
        "resume_ttft_ns": distribution_ttft,
        "tpot_ns": all_tpot,
        "resume_tpot_ns": distribution_tpot,
        "joint_slo_pass_count": 2,
        "joint_slo_fail_count": 0,
        "resume_joint_slo_pass_count": 1,
        "resume_joint_slo_fail_count": 0,
        "joint_slo_pass_output_tokens": 5,
        "joint_slo_pass_session_count": 1,
        "joint_slo_fail_session_count": 0,
        "window_start_ns": 100,
        "window_end_ns": 290,
        "window_duration_ns": 190,
        "operational_request_goodput_per_second": 2 * scale,
        "operational_resume_goodput_per_second": scale,
        "operational_token_goodput_per_second": 5 * scale,
        "operational_session_goodput_per_second": scale,
    }


def _request_rows(*, include_optional: bool) -> tuple[list[str], list[list[str]]]:
    header = [
        "session_id",
        "sub_request_index",
        "arrival",
        "end_time",
        "latency",
        "TTFT",
        "TPOT",
        "output",
        "generated_tokens",
    ]
    rows = [
        ["warmup", "0", "0", "30", "30", "10", "20", "2", "2"],
        ["s1", "0", "100", "150", "50", "30", "20", "2", "2"],
        ["s1", "1", "200", "290", "90", "50", "20", "3", "3"],
    ]
    if include_optional:
        optional = [
            "agentic_kv_source",
            "agentic_kv_residency_at_return",
            "return_gap_type",
            "agentic_kv_hit_tokens",
            "agentic_kv_recompute_tokens",
            "scheduler_queue_wait_ns",
            "agentic_kv_owner_gate_ns",
        ]
        header.extend(optional)
        rows[0].extend(["ssd", "ssd", "tool", "10", "0", "999", "999"])
        rows[1].extend(["", "", "session_start", "0", "0", "3", "0"])
        rows[2].extend(["hbf", "hbf", "tool", "70", "5", "7", "11"])
    return header, rows


def _build_campaign(
    tmp_path: Path,
    *,
    runtime_kind: str = "full_model_hbf",
    include_optional: bool = True,
    runtime: dict[str, object] | None = None,
    runtime_guard: bool = False,
) -> Path:
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    requests_path = cell_dir / "requests.csv"
    header, rows = _request_rows(include_optional=include_optional)
    with requests_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)

    session = {
        "measurement_window": {
            "measurement_required_session_ids": ["s1"],
            "measurement_complete": True,
            "measurement_boundary_complete": True,
            "measurement_early_stopped": False,
        },
        "validation": {
            "timing": {
                "checked_requests": 3,
                "passed": True,
                "violations": [],
                "warnings": [],
            },
        },
    }
    session_path = cell_dir / "session.json"
    _write_json(session_path, session)
    if runtime is None:
        runtime = {
            "hardware": {
                "card_count": 8,
                "hbf_capacity_bytes_per_card": 1_280_000_000_000,
                "lpddr_capacity_bytes_per_card": 64_000,
                "pcie_root_bandwidth_gbps": 200.0,
            },
            "layout": {"key": "tp8_context", "tp_size": 8, "replicas": 1},
            "options": {
                "prefill_drain_tail_tokens": 2_048,
                "prefill_drain_min_tokens": 4_096,
            },
            "adapter": {
                "metrics": {
                    "offered_requests": 3,
                    "gpu_requests": 2,
                    "hbf_requests": 1,
                    "gpu_completions": 2,
                    "hbf_completions": 1,
                },
                "execution_counts": {
                    "gpu_first_turn": 2,
                    "hbf_ready": 1,
                },
                "pending_router_completion_count": 0,
                "pending_hbf_turn_finalization_count": 0,
                "pending_gpu_hbm_event_count": 0,
                "staged_hbf_admission_count": 0,
                "pending_prefill_drain_request_by_session": {},
                "active_prefill_drain_request_by_job": {},
                "waiting_prefill_drain_append_jobs_by_session": {},
                "multiplexer": {
                    "pending_job_count": 0,
                    "ready_job_count": 0,
                    "quarantined_dispatch_count": 0,
                    "completed_job_count": 4,
                },
                "lifecycle": {
                    "pending_job_count": 0,
                    "active_prefill_drain_pending_job_ids": [],
                    "kv_bytes_per_token": 1_024,
                    "metrics": {
                        "migrations_started": 1,
                        "migrations_committed": 1,
                        "migration_logical_bytes": 100,
                        "migration_physical_bytes": 200,
                        "hbf_reserved_bytes_peak": 300,
                        "capacity_evictions": 0,
                        "active_prefill_drain_candidates": 3,
                        "active_prefill_drain_started": 2,
                        "active_prefill_drain_satisfied": 0,
                        "active_prefill_drain_wait_existing_append": 1,
                        "active_prefill_drain_capacity_fallback": 0,
                        "active_prefill_drain_committed": 1,
                        "active_prefill_drain_stale": 1,
                        "astra_completed_jobs": 2,
                        "astra_completion_elapsed_ns": 95,
                        "astra_resource_delay_ns": 15,
                        "astra_dependency_critical_path_ns": 80,
                        "astra_solo_resource_serialized_completion_ns": 90,
                        "astra_actual_resource_serialized_completion_ns": 95,
                        "astra_internal_resource_serialization_wait_ns": 10,
                        "astra_signed_interference_delta_ns": 5,
                    },
                },
                "pool": {
                    "pending_batch_count": 0,
                    "pending_launch_count": 0,
                    "prefill_drain_tail_tokens": 2_048,
                    "prefill_drain_min_tokens": 4_096,
                    "lpddr_kv_capacity_bytes_per_card": 60_000,
                    "lpddr_peak_bytes_by_card": {
                        "0": {"0": 20, "1": 30},
                    },
                    "metrics": {
                        "submitted_requests": 1,
                        "completed_requests": 1,
                        "batches": 2,
                        "attention_compute_roof_ns": 10,
                        "attention_hbf_roof_ns": 20,
                        "attention_lpddr_roof_ns": 30,
                        "attention_hbf_dominant_batches": 2,
                        "lpddr_capacity_deferrals": 0,
                        "prefill_drain_candidates": 1,
                        "prefill_drain_claimed": 1,
                        "prefill_drain_started": 2,
                        "prefill_drain_completed": 1,
                        "prefill_drain_fallbacks": 0,
                        "prefill_drain_logical_tokens": 4_096,
                        "prefill_drain_wait_ns": 6_000_000,
                        "astra_completed_batches": 2,
                        "astra_completion_elapsed_ns": 105,
                        "astra_resource_delay_ns": 5,
                        "astra_dependency_critical_path_ns": 100,
                        "astra_solo_resource_serialized_completion_ns": 110,
                        "astra_actual_resource_serialized_completion_ns": 105,
                        "astra_internal_resource_serialization_wait_ns": 10,
                        "astra_signed_interference_delta_ns": -5,
                    },
                },
            },
            "gpu_hbm_bridge": {
                "metrics": {"rejected_events": 0},
                "pending_colocated_claims": [],
                "pending_pd_recompute_bindings": [],
                "pending_pd_decode_reservations": [],
            },
        }
    runtime_path = cell_dir / "runtime.json"
    _write_json(runtime_path, runtime)
    stdout_path = cell_dir / "stdout.log"
    stderr_path = cell_dir / "stderr.log"
    stdout_path.write_text("ok\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    result = {
        "schema_version": 2,
        "status": "completed",
        "cell_id": "cell-1",
        "system": "hbf_tp8_context" if runtime_kind == "full_model_hbf"
        else "ssd_tiering",
        "runtime_kind": runtime_kind,
        "layout": "tp8_context" if runtime_kind == "full_model_hbf" else None,
        "seed": 101,
        "offered_session_rate_per_second": 0.5,
        "runtime_guard_contract": (
            {
                "seed": 101,
                "offered_session_rate_per_second": 0.5,
                "last_external_guard_offer_ns": 200,
                "expected_measurement_resume_count": 1,
            }
            if runtime_guard else None
        ),
        "workload": {
            "sha256": "1" * 64,
            "request_count": 3,
            "session_count": 2,
        },
        "metrics": _metrics(),
        "artifacts": {
            "requests": _artifact(requests_path),
            "session_report": _artifact(session_path),
            "runtime_report": _artifact(runtime_path),
            "stdout": _artifact(stdout_path),
            "stderr": _artifact(stderr_path),
        },
    }
    result_path = cell_dir / "result.json"
    _write_json(result_path, result)
    campaign = {
        "measurement_session_ids_sha256": (
            _stable_json_sha256(["s1"])),
    }
    if runtime_guard:
        campaign.update({
            "runtime_guard_validation_required": True,
            "runtime_guard_contracts": [
                result["runtime_guard_contract"],
            ],
        })
    manifest = {
        "schema_version": 2,
        "campaign_sha256": _stable_json_sha256(campaign),
        "campaign": campaign,
        "status": "completed",
        "cells": {
            "cell-1": {
                "status": "completed",
                "system": result["system"],
                "seed": 101,
                "rate": 0.5,
                "workload_sha256": "1" * 64,
                "request_count": 3,
                "session_count": 2,
                "runtime_guard_contract": (
                    result["runtime_guard_contract"]),
                "result": str(result_path),
                "result_sha256": _sha256(result_path),
                "result_bytes": result_path.stat().st_size,
            },
        },
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def test_collect_rejects_actual_resume_arrival_after_guard(tmp_path):
    manifest_path = _build_campaign(tmp_path, runtime_guard=True)
    collected = collect_campaign(manifest_path)
    guard = collected["cells"][0]["validity"][
        "measurement_resume_arrival_guard"]
    assert guard["observed_measurement_resume_count"] == 1
    assert guard["latest_measurement_resume_arrival_ns"] == 200

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["cells"]["cell-1"]
    result_path = Path(entry["result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    request_record = result["artifacts"]["requests"]
    request_path = Path(request_record["path"])
    header, rows = _request_rows(include_optional=True)
    rows[2][2] = "201"
    rows[2][3] = "291"
    with request_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    result["artifacts"]["requests"] = _artifact(request_path)
    _write_json(result_path, result)
    entry["result_sha256"] = _sha256(result_path)
    entry["result_bytes"] = result_path.stat().st_size
    _write_json(manifest_path, manifest)

    with pytest.raises(
            LiveAstraCollectError,
            match="1/1 measurement resume arrivals after"):
        collect_campaign(manifest_path)


def test_collect_rejects_stress_campaign_without_runtime_guard(tmp_path):
    manifest_path = _build_campaign(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["campaign"]["scenario_id"] = (
        "tracelab-headline-1741-balanced-highrate-v2")
    manifest["campaign_sha256"] = _stable_json_sha256(
        manifest["campaign"])
    _write_json(manifest_path, manifest)

    with pytest.raises(
            LiveAstraCollectError,
            match="missing mandatory runtime guard"):
        collect_campaign(manifest_path)


def _mutate_runtime_and_rehash(
    manifest_path: Path,
    mutation,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["cells"]["cell-1"]
    result_path = Path(entry["result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    runtime_record = result["artifacts"]["runtime_report"]
    runtime_path = Path(runtime_record["path"])
    if not runtime_path.is_absolute():
        runtime_path = result_path.parent / runtime_path
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    mutation(runtime)
    _write_json(runtime_path, runtime)
    result["artifacts"]["runtime_report"] = _artifact(runtime_path)
    _write_json(result_path, result)
    entry["result_sha256"] = _sha256(result_path)
    entry["result_bytes"] = result_path.stat().st_size
    _write_json(manifest_path, manifest)


def _mutate_session_and_rehash(
    manifest_path: Path,
    mutation,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["cells"]["cell-1"]
    result_path = Path(entry["result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    session_record = result["artifacts"]["session_report"]
    session_path = Path(session_record["path"])
    if not session_path.is_absolute():
        session_path = result_path.parent / session_path
    session = json.loads(session_path.read_text(encoding="utf-8"))
    mutation(session)
    _write_json(session_path, session)
    result["artifacts"]["session_report"] = _artifact(session_path)
    _write_json(result_path, result)
    entry["result_sha256"] = _sha256(result_path)
    entry["result_bytes"] = result_path.stat().st_size
    _write_json(manifest_path, manifest)


def _enable_strict_full_drain_superset(manifest_path: Path) -> None:
    full_roster = ["warmup", "s1"]

    def mutate(session):
        session["measurement_window"] = {
            "measurement_cohort_selection": "completion_order",
            "warmup_completions_requested": 0,
            "measure_completions_requested": 0,
            "measurement_complete": True,
            "measurement_boundary_complete": True,
            "measurement_early_stopped": False,
            "warmup_complete": True,
            "measurement_warmup_session_ids": [],
            "measurement_warmup_session_count": 0,
            "measurement_warmup_completed_sessions": 0,
            "measurement_warmup_session_ids_hash": (
                _stable_json_sha256([])),
            "warmup_completions_observed": 0,
            "measurement_prefix_id_overlap_count": 0,
            "measurement_target_session_ids": full_roster,
            "measurement_target_session_count": 2,
            "measurement_target_completed_sessions": 2,
            "measurement_target_session_ids_hash": (
                _stable_json_sha256(full_roster)),
            "measurement_required_session_ids": full_roster,
            "measurement_required_session_count": 2,
            "measurement_required_completed_sessions": 2,
            "measurement_required_session_ids_hash": (
                _stable_json_sha256(full_roster)),
            "measure_completions_observed": 2,
            "measurement_start_ns": 0,
            "measurement_end_ns": 290,
            "measurement_duration_ns": 290,
        }
        session["throughput"] = {
            "completed_sessions_total": 2,
            "completed_sessions": 2,
            "completed_requests_total": 3,
            "completed_requests": 3,
            "completed_requests_in_session_cohort": 3,
            "generated_tokens": 7,
            "generated_tokens_in_session_cohort": 7,
        }

    _mutate_session_and_rehash(manifest_path, mutate)


def test_collects_measurement_roster_and_raw_bottlenecks(tmp_path):
    manifest = _build_campaign(tmp_path)

    collected = collect_campaign(manifest)
    assert collected["schema_version"] == 2
    assert collected["collected_cell_count"] == 1
    cell = collected["cells"][0]
    performance = cell["performance"]
    assert performance["resume_ttft_ns"]["p95_ns"] == 50.0
    assert performance["resume_tpot_ns"]["p95_ns"] == 20.0
    assert math.isclose(
        performance["raw_request_throughput_per_second"],
        2 * 1_000_000_000 / 190,
    )
    assert math.isclose(
        performance["operational_resume_goodput_per_second"],
        1_000_000_000 / 190,
    )
    assert performance["offered_normalized_request_load_per_second"] == 1.0
    assert performance["offered_normalized_resume_load_per_second"] == 0.5
    assert (
        performance["offered_normalized_request_slo_goodput_per_second"]
        == 1.0
    )
    assert (
        performance["offered_normalized_resume_slo_goodput_per_second"]
        == 0.5
    )
    assert "external-gap boundary effects" in (
        performance["offered_normalized_goodput_semantics"]
    )

    source = cell["sources"]["resume_source"]
    assert source["counts"] == {"hbf": 1}
    assert source["fraction_of_measurement_resumes"] == {"hbf": 1.0}
    assert "ssd" not in source["counts"]
    assert (
        cell["bottlenecks"]["hbf"]["attention"]
        ["astra_signed_interference_delta_ns"]
        == -5
    )
    assert (
        cell["bottlenecks"]["hbf"]["network"]
        ["lifecycle_astra_signed_interference_delta_ns"]
        == 5
    )
    assert cell["sources"]["prefix_token_accounting"] == {
        "reported_count": 1,
        "missing_count": 0,
        "hit_tokens": 70,
        "recompute_tokens": 5,
        "hit_fraction_of_hit_plus_recompute_tokens": 70 / 75,
    }
    assert cell["bottlenecks"]["measurement_resume_waits"][
        "scheduler_queue"]["sum_ns"] == 7
    hbf = cell["bottlenecks"]["hbf"]
    assert hbf["capacity"]["migration_physical_to_logical_ratio"] == 2.0
    assert hbf["capacity"]["lpddr_peak_bytes_by_card_summary"] == {
        "leaf_count": 2,
        "maximum_bytes": 30,
        "sum_of_reported_leaf_peaks_bytes": 50,
    }
    assert hbf["attention"]["attention_hbf_dominant_batches"] == 2
    drain = hbf["prefill_drain"]
    assert drain["policy"] == {
        "tail_tokens": 2_048,
        "min_tokens": 4_096,
    }
    assert drain["pool"] == {
        "candidates": 1,
        "claimed": 1,
        "started": 2,
        "completed": 1,
        "fallbacks": 0,
        "logical_tokens": 4_096,
        "wait_ns": 6_000_000,
    }
    assert drain["lifecycle"] == {
        "candidates": 3,
        "started": 2,
        "satisfied": 0,
        "wait_existing_append": 1,
        "capacity_fallback": 0,
        "committed": 1,
        "stale": 1,
    }
    assert drain["kv_bytes_per_token"] == 1_024
    assert drain["pending"] == {
        "adapter_request_by_session_count": 0,
        "adapter_active_request_by_job_count": 0,
        "adapter_waiting_append_by_session_count": 0,
        "lifecycle_active_job_count": 0,
    }
    assert drain["derived"] == {
        "candidate_fraction": 1.0,
        "mean_wait_ms": 6.0,
        "fallback_fraction": 0.0,
        "logical_traffic_gib": 1 / 256,
    }
    assert cell["validity"]["headline_metric_crosscheck_mismatch_count"] == 0
    assert cell["validity"]["measurement_roster_relation"] == "exact"
    assert (
        cell["validity"]["measurement_roster_authoritative_source"]
        == "result.metrics.measurement_session_ids"
    )
    assert (
        cell["validity"]["measurement_roster_ordered_hash_verified"]
        is True
    )
    assert cell["validity"]["verified_artifact_count"] == 5
    assert (
        cell["validity"]["adapter_pending_prefill_drain_session_count"]
        == 0
    )
    assert (
        cell["validity"][
            "lifecycle_active_prefill_drain_pending_job_count"]
        == 0
    )
    assert cell["validity"]["paired_workload_sha_verified"] is True

    output_json = tmp_path / "compact.json"
    output_csv = tmp_path / "compact.csv"
    write_compact_json(collected, output_json)
    write_compact_csv(collected, output_csv)
    assert json.loads(output_json.read_text())["collected_cell_count"] == 1
    rows = list(csv.DictReader(output_csv.open(newline="")))
    assert len(rows) == 1
    assert rows[0]["system"] == "hbf_tp8_context"
    assert rows[0]["performance.resume_ttft_ns.p95_ns"] == "50.0"
    assert (
        rows[0][
            "bottlenecks.hbf.prefill_drain.derived.mean_wait_ms"]
        == "6.0"
    )


def test_rejects_noninteger_hbf_astra_interference_after_rehash(tmp_path):
    manifest = _build_campaign(tmp_path)

    def mutate(runtime):
        runtime["adapter"]["pool"]["metrics"][
            "astra_signed_interference_delta_ns"] = -5.0

    _mutate_runtime_and_rehash(manifest, mutate)
    with pytest.raises(
            LiveAstraCollectError, match="finite integer"):
        collect_campaign(manifest)


def test_rejects_hbf_astra_timing_algebra_after_rehash(tmp_path):
    manifest = _build_campaign(tmp_path)

    def mutate(runtime):
        runtime["adapter"]["lifecycle"]["metrics"][
            "astra_resource_delay_ns"] += 1

    _mutate_runtime_and_rehash(manifest, mutate)
    with pytest.raises(
            LiveAstraCollectError, match="timing algebra"):
        collect_campaign(manifest)


def test_accepts_guarded_default_full_drain_roster_superset(tmp_path):
    manifest = _build_campaign(tmp_path)
    _enable_strict_full_drain_superset(manifest)

    validity = collect_campaign(manifest)["cells"][0]["validity"]
    assert (
        validity["measurement_roster_relation"]
        == "strict_full_drain_superset"
    )
    assert validity["session_report_full_drain_superset_verified"] is True
    assert validity["preregistered_measurement_session_count"] == 1
    assert validity["session_report_measurement_session_count"] == 2
    assert validity["session_report_full_drain_native_session_count"] == 2
    assert validity["session_report_full_drain_native_request_count"] == 3
    assert validity["session_report_full_drain_crosscheck_count"] > 20
    assert validity["headline_metric_crosscheck_count"] == 42


def test_rejects_manifest_campaign_digest_tampering(tmp_path):
    manifest_path = _build_campaign(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["campaign"]["tampered"] = True
    _write_json(manifest_path, manifest)

    with pytest.raises(
        LiveAstraCollectError,
        match="campaign stable digest",
    ):
        collect_campaign(manifest_path)


def test_rejects_campaign_measurement_roster_hash_tampering(tmp_path):
    manifest_path = _build_campaign(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["campaign"]["measurement_session_ids_sha256"] = "b" * 64
    manifest["campaign_sha256"] = _stable_json_sha256(manifest["campaign"])
    _write_json(manifest_path, manifest)

    with pytest.raises(
        LiveAstraCollectError,
        match="measurement roster disagrees with campaign identity",
    ):
        collect_campaign(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda session: session["measurement_window"].__setitem__(
                "measurement_cohort_selection", "admission_order"),
            "measurement_cohort_selection is inconsistent",
        ),
        (
            lambda session: session["measurement_window"].__setitem__(
                "warmup_completions_requested", 1),
            "warmup_completions_requested is inconsistent",
        ),
        (
            lambda session: session["measurement_window"].__setitem__(
                "measure_completions_requested", 1),
            "measure_completions_requested is inconsistent",
        ),
        (
            lambda session: session["measurement_window"].__setitem__(
                "measurement_complete", False),
            "measurement_complete is inconsistent",
        ),
        (
            lambda session: session["measurement_window"].__setitem__(
                "measurement_boundary_complete", False),
            "measurement_boundary_complete is inconsistent",
        ),
        (
            lambda session: session["measurement_window"].__setitem__(
                "measurement_early_stopped", True),
            "measurement_early_stopped is inconsistent",
        ),
        (
            lambda session: session["measurement_window"].__setitem__(
                "measurement_target_session_ids_hash", "b" * 64),
            "roster hash is inconsistent",
        ),
        (
            lambda session: session["measurement_window"].__setitem__(
                "measurement_target_session_count", 1),
            "measurement_target_session_count is inconsistent",
        ),
        (
            lambda session: session["validation"]["timing"].__setitem__(
                "checked_requests", 2),
            "validation.timing.checked_requests is inconsistent",
        ),
        (
            lambda session: session["throughput"].__setitem__(
                "generated_tokens", 6),
            "throughput.generated_tokens is inconsistent",
        ),
    ],
)
def test_rejects_tampered_full_drain_superset_fields(
    tmp_path,
    mutation,
    message,
):
    manifest = _build_campaign(tmp_path)
    _enable_strict_full_drain_superset(manifest)
    _mutate_session_and_rehash(manifest, mutation)

    with pytest.raises(LiveAstraCollectError, match=message):
        collect_campaign(manifest)


def test_rejects_full_drain_roster_that_disagrees_with_native_csv(tmp_path):
    manifest = _build_campaign(tmp_path)
    _enable_strict_full_drain_superset(manifest)

    def reverse_roster(session):
        window = session["measurement_window"]
        reversed_roster = ["s1", "warmup"]
        window["measurement_target_session_ids"] = reversed_roster
        window["measurement_required_session_ids"] = reversed_roster
        digest = _stable_json_sha256(reversed_roster)
        window["measurement_target_session_ids_hash"] = digest
        window["measurement_required_session_ids_hash"] = digest

    _mutate_session_and_rehash(manifest, reverse_roster)
    with pytest.raises(
        LiveAstraCollectError,
        match="completion order disagrees with requests.csv",
    ):
        collect_campaign(manifest)


def test_rejects_full_drain_roster_missing_preregistered_session(tmp_path):
    manifest = _build_campaign(tmp_path)
    _enable_strict_full_drain_superset(manifest)

    def replace_roster(session):
        window = session["measurement_window"]
        replacement = ["warmup", "other"]
        window["measurement_target_session_ids"] = replacement
        window["measurement_required_session_ids"] = replacement
        digest = _stable_json_sha256(replacement)
        window["measurement_target_session_ids_hash"] = digest
        window["measurement_required_session_ids_hash"] = digest

    _mutate_session_and_rehash(manifest, replace_roster)
    with pytest.raises(
        LiveAstraCollectError,
        match="not a strict full-drain superset",
    ):
        collect_campaign(manifest)


def test_rejects_hbf_drain_policy_mismatch_after_rehash(tmp_path):
    manifest = _build_campaign(tmp_path)

    def mutate(runtime):
        runtime["adapter"]["pool"][
            "prefill_drain_tail_tokens"] = 2_049

    _mutate_runtime_and_rehash(manifest, mutate)
    with pytest.raises(
        LiveAstraCollectError,
        match="policy options disagree",
    ):
        collect_campaign(manifest)


def test_rejects_nonquiescent_hbf_drain_after_rehash(tmp_path):
    manifest = _build_campaign(tmp_path)

    def mutate(runtime):
        runtime["adapter"][
            "pending_prefill_drain_request_by_session"] = {"s1": 2}

    _mutate_runtime_and_rehash(manifest, mutate)
    with pytest.raises(
        LiveAstraCollectError,
        match="prefill-drain report is not quiescent",
    ):
        collect_campaign(manifest)


def test_rejects_hbf_drain_pool_algebra_after_rehash(tmp_path):
    manifest = _build_campaign(tmp_path)

    def mutate(runtime):
        runtime["adapter"]["pool"]["metrics"][
            "prefill_drain_claimed"] = 0

    _mutate_runtime_and_rehash(manifest, mutate)
    with pytest.raises(
        LiveAstraCollectError,
        match="pool candidate/claim accounting mismatch",
    ):
        collect_campaign(manifest)


def test_rejects_hbf_drain_lifecycle_algebra_after_rehash(tmp_path):
    manifest = _build_campaign(tmp_path)

    def mutate(runtime):
        runtime["adapter"]["lifecycle"]["metrics"][
            "active_prefill_drain_candidates"] = 2

    _mutate_runtime_and_rehash(manifest, mutate)
    with pytest.raises(
        LiveAstraCollectError,
        match="lifecycle outcome accounting mismatch",
    ):
        collect_campaign(manifest)


def test_rejects_hbf_drain_zero_denominator_after_rehash(tmp_path):
    manifest = _build_campaign(tmp_path)

    def mutate(runtime):
        pool = runtime["adapter"]["pool"]["metrics"]
        pool.update({
            "prefill_drain_candidates": 0,
            "prefill_drain_claimed": 0,
            "prefill_drain_started": 0,
            "prefill_drain_completed": 0,
            "prefill_drain_fallbacks": 0,
            "prefill_drain_logical_tokens": 0,
            "prefill_drain_wait_ns": 1,
        })
        lifecycle = runtime["adapter"]["lifecycle"]["metrics"]
        lifecycle.update({
            "active_prefill_drain_candidates": 0,
            "active_prefill_drain_started": 0,
            "active_prefill_drain_satisfied": 0,
            "active_prefill_drain_wait_existing_append": 0,
            "active_prefill_drain_capacity_fallback": 0,
            "active_prefill_drain_committed": 0,
            "active_prefill_drain_stale": 0,
        })

    _mutate_runtime_and_rehash(manifest, mutate)
    with pytest.raises(
        LiveAstraCollectError,
        match="mean wait has a nonzero numerator with a zero denominator",
    ):
        collect_campaign(manifest)


def test_rejects_missing_hbf_drain_metric_after_rehash(tmp_path):
    manifest = _build_campaign(tmp_path)

    def mutate(runtime):
        del runtime["adapter"]["pool"]["metrics"][
            "prefill_drain_wait_ns"]

    _mutate_runtime_and_rehash(manifest, mutate)
    with pytest.raises(
        LiveAstraCollectError,
        match="missing required HBF prefill-drain field",
    ):
        collect_campaign(manifest)


def test_rejects_changed_raw_artifact(tmp_path):
    manifest = _build_campaign(tmp_path)
    runtime = tmp_path / "cell" / "runtime.json"
    runtime.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        LiveAstraCollectError,
        match="runtime_report artifact (byte count|digest) changed",
    ):
        collect_campaign(manifest)


def test_rejects_changed_result(tmp_path):
    manifest = _build_campaign(tmp_path)
    result = tmp_path / "cell" / "result.json"
    result.write_text(result.read_text() + "\n", encoding="utf-8")

    with pytest.raises(
        LiveAstraCollectError,
        match="result digest changed",
    ):
        collect_campaign(manifest)


def test_rejects_headline_metric_disagreement(tmp_path):
    manifest = _build_campaign(tmp_path)
    manifest_value = json.loads(manifest.read_text())
    result_path = tmp_path / "cell" / "result.json"
    result = json.loads(result_path.read_text())
    result["metrics"]["resume_ttft_ns"]["p95_ns"] = 51.0
    _write_json(result_path, result)
    entry = manifest_value["cells"]["cell-1"]
    entry["result_sha256"] = _sha256(result_path)
    entry["result_bytes"] = result_path.stat().st_size
    _write_json(manifest, manifest_value)

    with pytest.raises(
        LiveAstraCollectError,
        match="metrics.resume_ttft_ns.p95_ns mismatch",
    ):
        collect_campaign(manifest)


def test_omits_unreported_optional_fields(tmp_path):
    manifest = _build_campaign(
        tmp_path,
        runtime_kind="agentic_kv",
        include_optional=False,
        runtime={},
    )

    cell = collect_campaign(manifest)["cells"][0]
    assert cell["sources"] == {}
    assert cell["bottlenecks"] == {}
    assert "external_fabric_pending_jobs" not in cell["validity"]
    assert cell["validity"]["headline_metric_crosscheck_mismatch_count"] == 0


def test_collects_baseline_ssd_dram_and_pcie_counters(tmp_path):
    runtime = {
        "totals": {
            "hbm_hits": 1,
            "cpu_hits": 2,
            "ssd_hits": 3,
            "ssd_host_write_bytes": 400,
            "ssd_host_read_bytes": 500,
            "peak_ssd_used_bytes": 600,
            "cpu_byte_ns": 700,
            "peak_idle_cpu_bytes": 800,
            "transient_dram_capacity_deferrals": 9,
            "pd_hbm_to_hbm_bytes": 1_000,
            "migration_service_ns": 1_100,
            "migration_queue_wait_ns": 1_200,
            "external_fabric_jobs_issued": 4,
            "external_fabric_jobs_completed": 4,
            "external_fabric_jobs_censored": 0,
            "external_fabric_lane_bytes": 1_300,
        },
        "ssd": {
            "capacity_bytes": 10_000,
            "used_bytes": 600,
            "reserved_bytes": 0,
        },
        "storage": {
            "totals": {
                "aligned_host_write_bytes": 400,
                "host_read_bytes": 500,
            },
        },
        "host_dram_staging": {
            "reservation_count": 2,
            "aggregate_explicit_capacity_wait_ns": 50,
            "capacity_deferrals": 9,
            "peak_transient_bytes": 900,
        },
        "observed_load_activity": {
            "global_transfer_execution_ns": 2_000,
            "global_any_model_busy_fraction": 0.75,
        },
        "resource_queues": {
            "node:0:ssd-pool:read": {
                "busy_until_ns": 2_100,
                "service_demand_ns": 200,
                "jobs": 2,
            },
            "node:0:dram": {
                "busy_until_ns": 2_200,
                "service_demand_ns": 300,
                "jobs": 3,
            },
            "instance:0:pcie-copy:0": {
                "busy_until_ns": 2_300,
                "service_demand_ns": 400,
                "jobs": 4,
            },
        },
        "external_fabric": {
            "issued_jobs": 4,
            "completed_jobs": 4,
            "censored_jobs": 0,
            "pending_jobs": 0,
        },
    }
    manifest = _build_campaign(
        tmp_path,
        runtime_kind="agentic_kv",
        runtime=runtime,
    )

    cell = collect_campaign(manifest)["cells"][0]
    baseline = cell["bottlenecks"]["baseline"]
    assert baseline["ssd"]["ssd_hits"] == 3
    assert baseline["ssd"]["ssd_host_read_bytes"] == 500
    assert baseline["dram"]["transient_capacity_deferrals"] == 9
    assert baseline["pcie_and_fabric"]["migration_queue_wait_ns"] == 1_200
    assert baseline["resource_queues"]["ssd_read"] == {
        "resource_count": 1,
        "jobs": 2,
        "service_demand_ns": 200,
        "maximum_busy_until_ns": 2_100,
    }
    assert baseline["resource_queues"]["dram"]["service_demand_ns"] == 300
    assert baseline["resource_queues"]["pcie"]["jobs"] == 4
    assert cell["validity"]["external_fabric_pending_jobs"] == 0


def test_incomplete_cells_require_explicit_permission(tmp_path):
    manifest_path = _build_campaign(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["cells"]["pending"] = {
        "status": "pending",
        "system": "oracle",
    }
    _write_json(manifest_path, manifest)

    with pytest.raises(
        LiveAstraCollectError,
        match="manifest cell pending is not completed",
    ):
        collect_campaign(manifest_path)
    collected = collect_campaign(manifest_path, allow_incomplete=True)
    assert collected["collected_cell_count"] == 1
    assert collected["skipped_incomplete_cell_ids"] == ["pending"]

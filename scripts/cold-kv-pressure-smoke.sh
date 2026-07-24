#!/usr/bin/env bash

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(dirname -- "$script_dir")
cd "$repo_root"

smoke_dir=$(mktemp -d)
trap 'rm -r -- "$smoke_dir"' EXIT
export COLD_KV_SMOKE_DIR="$smoke_dir"

while read -r label config; do
    log_path="$smoke_dir/${label}.log"
    if ! python -m serving \
        --cluster-config configs/cluster/single_node_cold_kv_pressure_smoke.json \
        --dataset workloads/cold-kv-pressure-smoke.jsonl \
        --num-reqs 3 \
        --max-num-seqs 1 \
        --max-num-batched-tokens 512 \
        --no-enable-prefix-caching \
        --agentic-kv-config "$config" \
        --agentic-kv-metrics "$smoke_dir/${label}.json" \
        --output "$smoke_dir/${label}.csv" \
        --run-id "cold-kv-pressure-${label}-$$" \
        --log-level WARNING >"$log_path" 2>&1
    then
        sed -n '1,240p' "$log_path"
        exit 1
    fi
done <<'EOF'
hbm configs/agentic_kv/hbm_lru_recompute.json
direct configs/agentic_kv/hbm_ssd_direct_8ssd.json
tiered configs/agentic_kv/tiered_capacity_fullwrite_8ssd.json
EOF

pd_log_path="$smoke_dir/pd-tiered.log"
if ! python -m serving \
    --cluster-config configs/cluster/single_node_pd_cold_kv_pressure_smoke.json \
    --dataset workloads/cold-kv-pressure-smoke.jsonl \
    --num-reqs 3 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 2048 \
    --no-enable-prefix-caching \
    --agentic-kv-config configs/agentic_kv/tiered_capacity_fullwrite_8ssd.json \
    --agentic-kv-metrics "$smoke_dir/pd-tiered.json" \
    --output "$smoke_dir/pd-tiered.csv" \
    --network-backend analytical-congestion-aware \
    --run-id "cold-kv-pressure-pd-tiered-$$" \
    --log-level WARNING >"$pd_log_path" 2>&1
then
    sed -n '1,240p' "$pd_log_path"
    exit 1
fi

python - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["COLD_KV_SMOKE_DIR"])
objects = 3
object_bytes = 35_651_584

expected = {
    "hbm": {
        "hbm_hits": 0,
        "cpu_hits": 0,
        "ssd_hits": 0,
        "recompute_tokens": 816,
        "policy_avoidable_recompute_tokens": 813,
        "hbm_to_cpu_bytes": 0,
        "cpu_to_ssd_bytes": 0,
        "hbm_to_ssd_bytes": 0,
        "ssd_to_hbm_bytes": 0,
        "ssd_to_cpu_stage_bytes": 0,
        "cpu_stage_to_hbm_bytes": 0,
        "active_hbm_reclaim_wait_ns": 0,
    },
    "direct": {
        "hbm_hits": 0,
        "cpu_hits": 0,
        "ssd_hits": 3,
        "recompute_tokens": 3,
        "policy_avoidable_recompute_tokens": 0,
        "hbm_to_cpu_bytes": 0,
        "cpu_to_ssd_bytes": 0,
        "hbm_to_ssd_bytes": objects * object_bytes,
        "ssd_to_hbm_bytes": objects * object_bytes,
        "ssd_to_cpu_stage_bytes": objects * object_bytes,
        "cpu_stage_to_hbm_bytes": objects * object_bytes,
        "active_hbm_reclaim_wait_ns": 1_466_064,
    },
    "tiered": {
        "hbm_hits": 0,
        "cpu_hits": 1,
        "ssd_hits": 2,
        "recompute_tokens": 3,
        "policy_avoidable_recompute_tokens": 0,
        "hbm_to_cpu_bytes": objects * object_bytes,
        "cpu_to_ssd_bytes": 2 * object_bytes,
        "hbm_to_ssd_bytes": 0,
        "ssd_to_hbm_bytes": 2 * object_bytes,
        "ssd_to_cpu_stage_bytes": 2 * object_bytes,
        "cpu_stage_to_hbm_bytes": 2 * object_bytes,
        "active_hbm_reclaim_wait_ns": 1_545_193,
    },
}


def assert_staged_ssd_reads(metrics):
    totals = metrics["totals"]
    foreground = [
        event for event in metrics["events"]
        if event["event"] == "migration_reserve"
        and event["foreground"]
    ]
    media_events = [
        event for event in foreground
        if event["kind"] == "ssd_to_cpu_stage"
    ]
    h2d_events = [
        event for event in foreground
        if event["kind"] in {
            "cpu_stage_to_hbm", "cpu_stage_to_decode",
        }
    ]
    assert len(media_events) == totals["ssd_hits"], (
        len(media_events), totals["ssd_hits"])
    assert len(h2d_events) == len(media_events), (
        len(h2d_events), len(media_events))
    assert sum(event["bytes"] for event in media_events) == totals[
        "ssd_to_cpu_stage_bytes"]
    assert sum(event["bytes"] for event in h2d_events) == totals[
        "cpu_stage_to_hbm_bytes"]
    assert totals["ssd_to_cpu_stage_bytes"] == totals[
        "cpu_stage_to_hbm_bytes"]
    assert totals["ssd_to_cpu_stage_bytes"] == totals[
        "ssd_host_read_bytes"]

    for media in media_events:
        matches = [
            event for event in h2d_events
            if event["session_id"] == media["session_id"]
            and event["time_ns"] == media["complete_ns"]
            and event["bytes"] == media["bytes"]
        ]
        assert len(matches) == 1, (media, matches)
        h2d = matches[0]
        assert "ssd-pool:read" in media["resources"], media
        assert any(
            resource.endswith(":dram")
            for resource in media["resources"]
        ), media
        assert not any(
            "pcie-copy" in resource for resource in media["resources"]
        ), media
        assert "ssd-pool:read" not in h2d["resources"], h2d
        assert any(
            resource.endswith(":dram")
            for resource in h2d["resources"]
        ), h2d
        assert any(
            "pcie-copy" in resource for resource in h2d["resources"]
        ), h2d
        assert h2d["complete_ns"] > h2d["start_ns"], h2d

    return media_events


def assert_pre_admission_rows(metrics, csv_path):
    foreground_by_session = {}
    for event in metrics["events"]:
        if (event["event"] == "migration_reserve"
                and event["foreground"]):
            foreground_by_session.setdefault(
                event["session_id"], []).append(event)
    commits_by_session = {
        event["session_id"]: event
        for event in metrics["events"]
        if event["event"] == "hbm_capacity_reservation_commit"
    }

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        restore_ns = int(row["agentic_kv_restore_ns"])
        assert restore_ns == sum(
            int(row[key]) for key in (
                "agentic_kv_hbm_admission_wait_ns",
                "agentic_kv_restore_queue_wait_ns",
                "agentic_kv_restore_service_ns",
            )
        )
        if restore_ns == 0:
            continue

        issue_ns = int(row["agentic_kv_restore_issue_time_ns"])
        target_ready_ns = int(
            row["agentic_kv_target_hbm_ready_time_ns"])
        restore_ready_ns = int(
            row["agentic_kv_restore_ready_time_ns"])
        first_dispatch_ns = int(row["arrival"]) + int(
            row["queuing_delay"])
        session_events = sorted(
            foreground_by_session[row["session_id"]],
            key=lambda event: event["time_ns"],
        )

        assert target_ready_ns >= issue_ns, row
        assert session_events[0]["start_ns"] >= target_ready_ns, (
            row, session_events[0])
        assert restore_ready_ns == issue_ns + restore_ns, row
        assert first_dispatch_ns >= restore_ready_ns, row
        assert int(row["agentic_kv_restore_compute_overlap_ns"]) == 0, row
        commit = commits_by_session.get(row["session_id"])
        if commit is not None:
            assert commit["time_ns"] <= session_events[0]["start_ns"], (
                commit, session_events[0])

    return rows


for label, counters in expected.items():
    metrics = json.loads((root / f"{label}.json").read_text())
    totals = metrics["totals"]
    assert metrics["schema_version"] == 12
    synchronous_swap = metrics["synchronous_swap"]
    asynchronous_restore = metrics["asynchronous_restore"]
    assert synchronous_swap["mode"] == "async-pre-admission"
    assert not synchronous_swap["enabled"]
    assert not synchronous_swap[
        "same_batch_membership_frozen_before_restore"
    ]
    assert synchronous_swap["pending_prepare_locks"] == 0
    assert synchronous_swap["pending_prepare_pinned_sessions"] == 0
    assert synchronous_swap[
        "aggregate_reservation_barrier_union_ns"
    ] == 0
    assert synchronous_swap["aggregate_exposed_engine_wait_ns"] == 0
    assert synchronous_swap["blocked_iteration_count"] == 0
    assert not metrics["batch_composition"][
        "restore_barrier_inside_batch"
    ]
    assert not metrics["batch_composition"][
        "sync_swap_barrier_before_batch"
    ]
    assert metrics["batch_composition"][
        "restore_barrier_semantics"
    ] == "restore_completed_before_scheduler_visibility"
    assert all(
        not event["restore_barrier_inside_batch"]
        and not event["sync_swap_barrier_before_batch"]
        for event in metrics["events"]
        if event["event"] == "agentic_batch_schedule"
    )
    assert asynchronous_restore["mode"] == "async-pre-admission"
    assert not asynchronous_restore["swap_out_blocks_model"]
    assert not asynchronous_restore["swap_in_blocks_other_requests"]
    assert asynchronous_restore["decode_requires_restore_complete"]
    assert asynchronous_restore["overlap_model"] == "none"
    async_overlap_ns = asynchronous_restore[
        "aggregate_prefill_execution_overlap_ns"
    ]
    assert async_overlap_ns == 0
    for key, value in counters.items():
        assert totals[key] == value, (label, key, totals[key], value)

    assert totals["critical_restore_ns"] == (
        totals["critical_restore_hbm_admission_wait_ns"]
        + totals["critical_restore_queue_wait_ns"]
        + totals["critical_restore_service_ns"]
    )
    for event in metrics["events"]:
        if event["event"] == "active_hbm_reclaim_consume":
            assert event["time_ns"] == event["ready_ns"], event

    reservations = [
        event for event in metrics["events"]
        if event["event"] == "migration_reserve"
    ]
    resources = {
        resource
        for event in reservations
        for resource in event["resources"]
    }
    if label == "direct":
        assert resources == {
            "instance:0:pcie-copy:0",
            "node:0:dram",
            "ssd-pool:read",
            "ssd-pool:write",
        }, resources
        assert totals["cpu_hits"] == 0
        assert totals["hbm_to_cpu_bytes"] == 0
        assert totals["cpu_to_hbm_bytes"] == 0
        assert totals["cpu_to_ssd_bytes"] == 0
        assert totals["cpu_byte_ns"] == 0
        assert metrics["latency_model"]["storage_path"] == (
            "gpu_ssd_direct_write_host_dram_staged_read_analytical"
        )
    if label == "tiered":
        assert "node:0:dram" in resources, resources

    media_events = assert_staged_ssd_reads(metrics)
    rows = assert_pre_admission_rows(
        metrics, root / f"{label}.csv")
    if label == "hbm":
        assert not media_events
    else:
        assert media_events
        assert any(
            int(row["agentic_kv_restore_ns"]) > 0 for row in rows)

    print(
        f"{label}: duration={metrics['simulated_duration_ns']} ns, "
        f"recompute={totals['recompute_tokens']} tokens, "
        f"restore={totals['critical_restore_ns']} ns, "
        f"same-owner-prefill-overlap={async_overlap_ns} ns"
    )

pd_metrics = json.loads((root / "pd-tiered.json").read_text())
pd_totals = pd_metrics["totals"]
assert pd_metrics["schema_version"] == 12
assert pd_metrics["synchronous_swap"]["mode"] == "async-pre-admission"
assert not pd_metrics["synchronous_swap"]["enabled"]
assert pd_metrics["synchronous_swap"][
    "aggregate_reservation_barrier_union_ns"
] == 0
assert pd_metrics["synchronous_swap"][
    "aggregate_exposed_engine_wait_ns"
] == 0
assert not pd_metrics["synchronous_swap"][
    "same_batch_membership_frozen_before_restore"
]
assert pd_metrics["synchronous_swap"]["pending_prepare_locks"] == 0
assert pd_metrics["synchronous_swap"][
    "pending_prepare_pinned_sessions"
] == 0
assert pd_metrics["synchronous_swap"]["blocked_iteration_count"] == 0
assert not pd_metrics["batch_composition"][
    "sync_swap_barrier_before_batch"
]
pd_barriers = pd_metrics["synchronous_swap"][
    "reservation_barrier_union_ns_by_instance"
]
assert len(pd_barriers) == 2, pd_barriers
assert all(wait_ns == 0 for wait_ns in pd_barriers.values()), pd_barriers
pd_async = pd_metrics["asynchronous_restore"]
assert pd_async["mode"] == "async-pre-admission"
assert not pd_async["swap_out_blocks_model"]
assert not pd_async["swap_in_blocks_other_requests"]
assert pd_async["decode_requires_restore_complete"]
assert pd_async["overlap_model"] == "none"
assert pd_async["aggregate_prefill_execution_overlap_ns"] == 0
assert pd_totals["pd_prefill_admissions"] == 6
assert pd_totals["pd_decode_receive_admissions"] == 6
assert pd_totals["pd_launch_admissions"] == 6
assert not pd_metrics["active_hbm_reclaim"]["outstanding_claims"]
assert pd_metrics["request_classification"][
    "all_agentic_request_count"
] == 6
pd_media_events = assert_staged_ssd_reads(pd_metrics)
pd_rows = assert_pre_admission_rows(
    pd_metrics, root / "pd-tiered.csv")
assert len(pd_media_events) == 1

# The resumed SSD owner remains invisible to compute until its complete
# SSD -> CPU -> decode-HBM -> prefill-HBM chain is ready.  A peer decode batch
# still dispatches while that owner-local chain is in flight.
ssd_row = next(
    row for row in pd_rows if row["agentic_kv_source"] == "ssd")
ssd_issue_ns = int(ssd_row["agentic_kv_restore_issue_time_ns"])
ssd_ready_ns = int(ssd_row["agentic_kv_restore_ready_time_ns"])
assert any(
    ssd_issue_ns < event["time_ns"] < ssd_ready_ns
    and event["source_counts"].get("ssd", 0) == 0
    for event in pd_metrics["events"]
    if event["event"] == "agentic_batch_schedule"
)

prefill_admissions = [
    event for event in pd_metrics["events"]
    if event["event"] == "pd_prefill_active_admission"
]
decode_admissions = [
    event for event in pd_metrics["events"]
    if event["event"] == "pd_decode_receive_admission"
]
launch_admissions = [
    event for event in pd_metrics["events"]
    if event["event"] == "pd_launch_admission"
]
assert (
    len(prefill_admissions)
    == len(decode_admissions)
    == len(launch_admissions)
    == 6
)
for event in prefill_admissions:
    assert event["full_per_rank_bytes"] == (
        event["restored_prefix_per_rank_bytes"]
        + event["newly_reserved_per_rank_bytes"]
    ), event
for event in decode_admissions:
    assert event["full_per_rank_bytes"] == (
        event["retained_per_rank_bytes"]
        + event["newly_reserved_per_rank_bytes"]
    ), event

prefill_by_request = {
    event["request_id"]: event for event in prefill_admissions
}
decode_by_request = {
    event["request_id"]: event for event in decode_admissions
}
for launch in launch_admissions:
    prefill = prefill_by_request[launch["request_id"]]
    decode = decode_by_request[launch["request_id"]]
    admitted_ns = launch["admitted_ns"]
    assert prefill["admitted_ns"] == admitted_ns, (prefill, launch)
    assert decode["admitted_ns"] == admitted_ns, (decode, launch)
    assert admitted_ns >= max(
        prefill["capacity_ready_ns"], decode["capacity_ready_ns"]
    ), launch
    assert launch["wait_ns"] == (
        admitted_ns - launch["enqueued_ns"]
    ), launch
    assert launch["critical_wait_after_restore_ns"] == max(
        0,
        admitted_ns
        - max(launch["restore_ready_ns"], launch["enqueued_ns"]),
    ), launch

assert len(pd_rows) == 6
assert all(
    int(row["pd_prefill_capacity_wait_ns"])
    <= int(row["pd_prefill_admission_wait_ns"])
    and int(row["pd_decode_capacity_wait_ns"])
    <= int(row["pd_decode_admission_wait_ns"])
    and int(row["pd_launch_admission_wait_ns"])
    == int(row["pd_prefill_admission_wait_ns"])
    == int(row["pd_decode_admission_wait_ns"])
    and int(row["pd_launch_admission_critical_wait_ns"])
    >= max(
        int(row["pd_prefill_admission_critical_wait_ns"]),
        int(row["pd_decode_admission_critical_wait_ns"]),
    )
    for row in pd_rows
)
print(
    "pd-tiered: strict P/D full-context admissions="
    f"{pd_totals['pd_prefill_admissions']}, "
    "decode receive admissions="
    f"{pd_totals['pd_decode_receive_admissions']}, "
    "same-owner-prefill-overlap="
    f"{pd_async['aggregate_prefill_execution_overlap_ns']} ns, "
    "peer-progress-during-ssd-restore=yes"
)
PY

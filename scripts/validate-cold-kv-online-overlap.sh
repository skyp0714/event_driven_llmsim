#!/usr/bin/env bash

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(dirname -- "$script_dir")
cd "$repo_root"

validation_root=${1:-$(mktemp -d /tmp/cold-kv-online-overlap.XXXXXX)}
mkdir -p "$validation_root"
export COLD_KV_OVERLAP_VALIDATION_ROOT="$validation_root"

run_case() {
    label=$1
    dataset=$2
    num_reqs=$3
    case_dir="$validation_root/$label"
    mkdir -p "$case_dir"
    /usr/bin/time -f '%e' -o "$case_dir/wall_seconds.txt" \
        python -m serving \
        --cluster-config configs/cluster/single_node_qwen3_1m_pd_p4d4_h100.json \
        --dataset "$dataset" \
        --num-reqs "$num_reqs" \
        --max-num-seqs 128 \
        --max-num-batched-tokens 131072 \
        --long-prefill-token-threshold 131072 \
        --no-enable-prefix-caching \
        --agentic-kv-config configs/agentic_kv/qwen3_1m_p4d4/tiered.json \
        --agentic-kv-metrics "$case_dir/agentic.json" \
        --session-metrics "$case_dir/session.json" \
        --output "$case_dir/requests.csv" \
        --network-backend analytical-congestion-aware \
        --run-id "cold-kv-online-overlap-$label-$$" \
        --log-level WARNING >"$case_dir/run.log" 2>&1
}

# The first run contains an HBM-resident D->P return and an unrelated P batch.
# The next two runs isolate the same transfer and the same unrelated batch so
# ASTRA contention can be measured without changing either operation's shape.
run_case overlap workloads/cold-kv-online-overlap-validation.jsonl 2
run_case isolated_restore workloads/cold-kv-online-overlap-validation.jsonl 1
run_case isolated_model workloads/cold-kv-online-overlap-unrelated-control.jsonl 1

python - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

from serving.online_experiments import (
    _external_fabric_model_coexecution_audit,
)


root = Path(os.environ["COLD_KV_OVERLAP_VALIDATION_ROOT"])


def load_json(case, name):
    with (root / case / name).open() as source:
        return json.load(source)


def load_requests(case):
    with (root / case / "requests.csv").open(newline="") as source:
        return list(csv.DictReader(source))


overlap = load_json("overlap", "agentic.json")
isolated_restore = load_json("isolated_restore", "agentic.json")
isolated_model = load_json("isolated_model", "agentic.json")
requests = load_requests("overlap")

authority = overlap["external_fabric"]["authority"]
assert authority == {
    "backend": "analytical-congestion-aware",
    "bandwidth_gbps": 450.0,
    "bandwidth_unit": "decimal_GBps",
    "latency_ns": 1000,
    "completion_source": "astra_event_queue_callback",
}, authority

for report in (overlap, isolated_restore):
    external = report["external_fabric"]
    assert external["issued_jobs"] == 1, external
    assert external["completed_jobs"] == 1, external
    assert external["censored_jobs"] == 0, external
    assert external["pending_jobs"] == 0, external
    assert external["pending_sessions"] == [], external

cold = overlap["external_fabric"]["completed_intervals"][0]
isolated_cold = isolated_restore[
    "external_fabric"]["completed_intervals"][0]
assert cold["bytes"] == isolated_cold["bytes"] == 3_221_225_472
assert cold["bytes_per_lane"] == isolated_cold["bytes_per_lane"]
assert cold["lane_count"] == isolated_cold["lane_count"] == 4
assert cold["queue_wait_ns"] == isolated_cold["queue_wait_ns"] == 0
assert cold["arrival_ns"] <= cold["start_ns"] < cold["complete_ns"]

wire_lower_bound_ns = (
    math.ceil(
        cold["bytes_per_lane"] / authority["bandwidth_gbps"])
    + authority["latency_ns"]
)
assert isolated_cold["service_ns"] >= wire_lower_bound_ns
transfer_contention_ns = (
    cold["service_ns"] - isolated_cold["service_ns"])
assert transfer_contention_ns > 0, transfer_contention_ns

model_windows = [
    event for event in overlap["events"]
    if event.get("event") == "astra_shared_fabric_window"
]
overlapping_windows = [
    event for event in model_windows
    if event["start_ns"] < cold["complete_ns"]
    and cold["start_ns"] < event["complete_ns"]
]
assert overlapping_windows, (cold, model_windows)
report_layer_overlap = _external_fabric_model_coexecution_audit(overlap)
assert report_layer_overlap["coexecution_pair_count"] >= 1, (
    report_layer_overlap)
assert report_layer_overlap["overlapped_job_count"] == 1, (
    report_layer_overlap)

owner = next(
    row for row in requests
    if row["session_id"] == "cold-restore-owner"
    and int(row["sub_request_index"]) == 1
)
unrelated = next(
    row for row in requests
    if row["session_id"] == "unrelated-prefill"
)
assert owner["agentic_kv_source"] == "hbm", owner
assert owner["return_gap_type"] == "tool", owner
assert int(owner["arrival"]) == cold["arrival_ns"]
assert int(owner["agentic_kv_restore_issue_time_ns"]) == cold["arrival_ns"]
assert int(owner["agentic_kv_restore_ready_time_ns"]) == cold["complete_ns"]
assert int(owner["agentic_kv_restore_ns"]) == (
    cold["complete_ns"] - cold["arrival_ns"])
assert int(owner["first_schedule_eligibility_time_ns"]) == cold["complete_ns"]
assert int(owner["first_schedule_time_ns"]) >= cold["complete_ns"]
assert int(owner["TTFT"]) >= int(owner["agentic_kv_restore_ns"])

assert int(unrelated["first_schedule_time_ns"]) < cold["arrival_ns"]
assert int(unrelated["end_time"]) > cold["complete_ns"]
unrelated_window = next(
    event for event in model_windows
    if event["start_ns"] == int(unrelated["first_schedule_time_ns"])
)
isolated_model_windows = [
    event for event in isolated_model["events"]
    if event.get("event") == "astra_shared_fabric_window"
]
assert len(isolated_model_windows) == 1, isolated_model_windows
isolated_model_window = isolated_model_windows[0]
model_contention_ns = (
    unrelated_window["duration_ns"]
    - isolated_model_window["duration_ns"])
assert model_contention_ns > 0, model_contention_ns

totals = overlap["totals"]
assert totals["direct_fabric_dispatch_blocks"] == 0, totals
assert totals["direct_fabric_dispatch_wait_ns"] == 0, totals
assert totals["external_fabric_jobs_issued"] == 1, totals
assert totals["external_fabric_jobs_completed"] == 1, totals
assert totals["external_fabric_lane_bytes"] == cold["bytes"], totals
assert totals["pd_hbm_to_hbm_bytes"] == cold["bytes"], totals
assert totals["critical_restore_ns"] == (
    cold["complete_ns"] - cold["arrival_ns"]), totals

bridge = overlap["online_resource_bridge"]
assert bridge["mode"] == "astra_shared_fabric_owner_ready_barrier", bridge
assert bridge["forbidden_overlap_count"] == 0, bridge
assert bridge["open_astra_window_count"] == 0, bridge
assert bridge["pending_direct_fabric_prepare_locks"] == 0, bridge
assert bridge["future_fabric_dispatch_is_gated"] is False, bridge
assert bridge["shared_fabric_contention_may_extend_model_communication"] is True

summary = {
    "status": "pass",
    "execution_path": "python -m serving -> Scheduler/trace/Chakra/ASTRA",
    "external_fabric_authority": authority,
    "cold_restore": cold,
    "isolated_cold_restore_service_ns": isolated_cold["service_ns"],
    "wire_lower_bound_ns": wire_lower_bound_ns,
    "transfer_contention_ns": transfer_contention_ns,
    "transfer_contention_fraction": (
        transfer_contention_ns / isolated_cold["service_ns"]),
    "unrelated_model_window": unrelated_window,
    "isolated_model_window": isolated_model_window,
    "model_contention_ns": model_contention_ns,
    "model_contention_fraction": (
        model_contention_ns / isolated_model_window["duration_ns"]),
    "owner": {
        "request_ready_ns": int(owner["arrival"]),
        "restore_ready_ns": int(
            owner["agentic_kv_restore_ready_time_ns"]),
        "first_schedule_eligibility_ns": int(
            owner["first_schedule_eligibility_time_ns"]),
        "first_schedule_ns": int(owner["first_schedule_time_ns"]),
        "ttft_ns": int(owner["TTFT"]),
    },
    "unrelated": {
        "first_schedule_ns": int(unrelated["first_schedule_time_ns"]),
        "completion_ns": int(unrelated["end_time"]),
    },
    "raw_external_overlap_count": len(overlapping_windows),
    "report_layer_external_model_coexecution": report_layer_overlap,
    "reported_allowed_model_overlap_count": bridge[
        "allowed_model_overlap_count"],
    "wall_seconds": {
        case: float((root / case / "wall_seconds.txt").read_text())
        for case in ("overlap", "isolated_restore", "isolated_model")
    },
}
(root / "validation.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "Validation artifacts: $validation_root"

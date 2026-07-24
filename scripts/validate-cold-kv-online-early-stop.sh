#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
validation_root=$(mktemp -d "${TMPDIR:-/tmp}/cold-kv-online-early-stop.XXXXXX")

run_case() {
    local case_name=$1
    local dataset=$2
    local arrival_mode=$3
    local max_active_sessions=$4
    local case_root="${validation_root}/${case_name}"
    mkdir -p "${case_root}"

    local -a command=(
        python -m serving
        --cluster-config configs/cluster/single_node_qwen3_1m_pd_p4d4_h100.json
        --dataset "${dataset}"
        --agentic-kv-config configs/agentic_kv/qwen3_1m_p4d4/tiered.json
        --network-backend analytical-congestion-aware
        --run-id "cold-kv-early-stop-${case_name}"
        --inputs-root "${case_root}/inputs"
        --no-cleanup-inputs
        --session-arrival-mode "${arrival_mode}"
        --session-warmup-completions 0
        --session-measure-completions 1
        --session-stop-after-measurement
        --session-metrics "${case_root}/session.json"
        --agentic-kv-metrics "${case_root}/agentic.json"
        --output "${case_root}/requests.csv"
        --log-level WARNING
        --log-interval 1000000
    )
    if [[ "${arrival_mode}" == "backlog" ]]; then
        command+=(--max-active-sessions "${max_active_sessions}" --session-backlog-epochs 1)
    fi
    (
        cd "${repo_root}"
        "${command[@]}" >"${case_root}/run.log" 2>&1
    )
}

run_case \
    queued_decode \
    workloads/cold-kv-online-early-stop-validation.jsonl \
    backlog \
    2 &
queued_pid=$!
run_case \
    inflight_prefill \
    workloads/cold-kv-online-early-stop-inflight-p-validation.jsonl \
    trace \
    0 &
prefill_pid=$!
wait "${queued_pid}"
wait "${prefill_pid}"

python - "${validation_root}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expectations = {
    "queued_decode": (1, 0),
    "inflight_prefill": (0, 1),
}
summary = {"status": "PASS", "root": str(root), "cases": {}}
for name, (queued_decode, completed_prefill) in expectations.items():
    report = json.loads((root / name / "session.json").read_text())
    window = report["measurement_window"]
    censoring = report["censoring"]
    if window["measurement_early_stopped"] is not True:
        raise SystemExit(f"{name}: measurement did not early-stop")
    if censoring["censored_queued_active_requests"] != queued_decode:
        raise SystemExit(f"{name}: queued-D censor count mismatch")
    if censoring["censored_completed_pd_prefill_requests"] != completed_prefill:
        raise SystemExit(f"{name}: drained-P censor count mismatch")
    drain = censoring.get("manager_drain_audit") or {}
    if drain.get("passed") is not True or drain.get("live_state"):
        raise SystemExit(f"{name}: manager did not drain: {drain}")
    for instance_id, memory in censoring["memory_after_censoring"].items():
        if memory["npu_used"] != memory["npu_baseline"] or memory["cpu_used"] != 0:
            raise SystemExit(
                f"{name}: instance {instance_id} retained memory: {memory}"
            )
    summary["cases"][name] = {
        "censored_queued_active_requests": queued_decode,
        "censored_completed_pd_prefill_requests": completed_prefill,
        "memory_after_censoring": censoring["memory_after_censoring"],
        "manager_drain_audit": drain,
    }

(root / "validation.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(root / "validation.json")
PY

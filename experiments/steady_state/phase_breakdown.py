"""Split resume TTFT into the three phases that can each explain it.

A resume TTFT of tens of seconds has three candidate owners and they call for
different fixes, so the aggregate number cannot settle the question:

  release -> prepare_start        admission wait.  The call is eligible but
                                  the node will not start it: no D-HBM
                                  headroom, or a restore slot is busy.  A
                                  policy problem.
  prepare_start -> prepare_done   restore transfer.  KV is moving from CPU or
                                  SSD into D-HBM.  A bandwidth problem, and
                                  the one the tier hierarchy is supposed to
                                  own.
  prepare_done -> first_token     prefill queue + prefill compute.  The KV is
                                  resident and the request is waiting on, or
                                  running in, the P workers.  A compute
                                  problem, which HBM capacity cannot fix.

The systems already record every one of these timestamps on their runtime
call records and never delete them, so this is pure extraction: nothing in
serving/ has to change to get the decomposition.
"""

from __future__ import annotations

import statistics
from collections import Counter


def _percentile(values, q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _summary(values) -> dict:
    if not values:
        return {"n": 0, "mean_s": 0.0, "p50_s": 0.0,
                "p95_s": 0.0, "p99_s": 0.0, "share": 0.0}
    return {
        "n": len(values),
        "mean_s": statistics.fmean(values),
        "p50_s": _percentile(values, 0.50),
        "p95_s": _percentile(values, 0.95),
        "p99_s": _percentile(values, 0.99),
    }


def extract(system, completed) -> dict | None:
    """Return per-phase distributions over resume calls, or None.

    `completed` is the list of CompletedRequest snapshots, which carry the
    request key; the runtime call records carry the phase timestamps.  They
    join on request_id, which the snapshots do not expose, so we key the
    runtime side by (session_id, call_index) instead -- that pair is unique
    per call and is present on both sides.
    """

    runtime = getattr(system, "_runtime_calls", None)
    if not runtime:
        return None

    by_key = {}
    for call in runtime.values():
        session_id = getattr(call, "session_id", None)
        call_index = getattr(call, "call_index", None)
        if session_id is None or call_index is None:
            continue
        by_key[(session_id, call_index)] = call

    admission, restore, prefill, total = [], [], [], []
    sources = Counter()
    deferrals = 0
    unmatched = 0
    missing_phase = 0

    for snapshot in completed:
        if snapshot.key.sub_request_index == 0:
            continue  # first calls have no prior KV to restore
        call = by_key.get(
            (snapshot.key.session_id, snapshot.key.sub_request_index))
        if call is None:
            unmatched += 1
            continue
        start = getattr(call, "prepare_start_ns", None)
        done = getattr(call, "prepare_completion_ns", None)
        deferrals += int(getattr(call, "capacity_deferrals", 0) or 0)
        source = getattr(call, "prepare_source", None)
        if source is not None:
            sources[getattr(source, "value", str(source))] += 1
        if start is None or done is None:
            missing_phase += 1
            continue
        release = snapshot.release_ns
        first = snapshot.first_token_ns
        admission.append(max(0, start - release) / 1e9)
        restore.append(max(0, done - start) / 1e9)
        prefill.append(max(0, first - done) / 1e9)
        total.append(max(0, first - release) / 1e9)

    if not total:
        return {
            "resume_calls_matched": 0,
            "unmatched": unmatched,
            "missing_phase_timestamps": missing_phase,
            "capacity_deferrals": deferrals,
            "restore_sources": dict(sources),
        }

    mean_total = statistics.fmean(total)
    out = {
        "resume_calls_matched": len(total),
        "unmatched": unmatched,
        "missing_phase_timestamps": missing_phase,
        "capacity_deferrals": deferrals,
        "restore_sources": dict(sources),
        "admission_wait": _summary(admission),
        "restore_transfer": _summary(restore),
        "prefill_queue_and_compute": _summary(prefill),
        "resume_ttft_total": _summary(total),
    }
    # Shares of the mean, which is what "who owns the latency" means here.
    for name, series in (
        ("admission_wait", admission),
        ("restore_transfer", restore),
        ("prefill_queue_and_compute", prefill),
    ):
        out[name]["share_of_mean"] = (
            statistics.fmean(series) / mean_total if mean_total > 0 else 0.0)
    return out

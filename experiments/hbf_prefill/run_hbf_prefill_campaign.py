#!/usr/bin/env python3
"""One-server steady-state campaign: HBF-prefill P4D4 vs CPU/SSD tiering.

Both systems are a single eight-card server with SSD offloading:

* ``baseline_cpu_ssd`` -- eight H100s (4P + 4D) with D-HBM retention,
  CPU staging, and SSD spill.  Pure LRU under capacity pressure; no TTL.
* ``hbf_prefill_p4d4`` -- four HBF+LPDDR prefill cards and four H100
  decode GPUs.  Committed KV's durable home is the P-side HBF (one TP4
  replica, 5.12 TB); decode copies stream over NVLink via the layerwise
  handoff; the cold tail spills to the same SSDs.
* ``oracle_infinite_hbm`` -- the unbounded-HBM upper bound.

The population model reuses the steady-state campaign: Little's-Law
resident preload (length-biased over inter-turn gaps) plus Poisson
arrivals, measured on a call-count window.  This driver adds the
gap-duration bucketing (resume TTFT and turn latency conditioned on how
long the session had been idle) and SSD/HBF write accounting.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import math
import os
import random
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parent
STEADY_STATE_ROOT = SCRIPT_ROOT.parent / "steady_state"
SESSION_SCALING_ROOT = SCRIPT_ROOT.parent / "session_scaling"
for root in (STEADY_STATE_ROOT, SESSION_SCALING_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import phase_breakdown  # noqa: E402
import run_session_scaling_campaign as C  # noqa: E402
import run_steady_state_campaign as S  # noqa: E402

REPO_ROOT = C.REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 2
FAMILIES = S.FAMILIES
SYSTEMS = (
    "baseline_cpu_ssd",
    "oracle_infinite_hbm",
    "hbf_prefill_p4d4",
)
DEFAULT_RATES = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064)
DEFAULT_SEEDS = (101, 102, 103)
DEFAULT_MEASURED_CALLS = 6_000

GPU_CONFIG = REPO_ROOT / "configs/wakekv_hbf/p4d4_gpu_server.json"
HETERO_CONFIG = (
    REPO_ROOT / "configs/wakekv_hbf/hetero_p4hbf_d4hbm_server.json")

# Idle-gap buckets for conditioned resume metrics.  Upper edges in
# seconds; the last bucket is open.
GAP_BUCKETS = (
    (60.0, "lt_1m"),
    (300.0, "1m_5m"),
    (1_800.0, "5m_30m"),
    (3_600.0, "30m_1h"),
    (14_400.0, "1h_4h"),
    (43_200.0, "4h_12h"),
    (math.inf, "gt_12h"),
)

# Context-size buckets over the resume call's reused prefix, in tokens.
# Edges track the pool's per-event distribution (p50 ~120-135k for both
# families) and its gap-weighted resident tail (claude reaches ~1M).
CONTEXT_BUCKETS = (
    (16_384.0, "lt_16k"),
    (65_536.0, "16k_64k"),
    (131_072.0, "64k_128k"),
    (262_144.0, "128k_256k"),
    (524_288.0, "256k_512k"),
    (math.inf, "gt_512k"),
)

KV_AGGREGATE_BYTES_PER_TOKEN = 4 * S.KV_BYTES_PER_TOKEN_PER_RANK


def _bucket_of(value: float, buckets) -> str:
    for edge, label in buckets:
        if value < edge:
            return label
    return buckets[-1][1]


def gap_bucket(gap_s: float) -> str:
    return _bucket_of(gap_s, GAP_BUCKETS)


def context_bucket(tokens: float) -> str:
    return _bucket_of(tokens, CONTEXT_BUCKETS)


def build_residents_steady(sessions, target_l: int, rng):
    """Sample the resident snapshot from the true steady-state law.

    A steady-state snapshot of idle sessions samples (session, gap)
    pairs jointly with probability proportional to the gap duration.
    The earlier helper picked the template uniformly and only weighted
    gaps within it, which under-represents sessions that spend most of
    their lifetime idle -- exactly the long-idle, large-context
    population the capacity tiers exist for.  The revive time is the
    residual of the sampled gap (uniform within it), so sessions deep
    in multi-hour gaps preload with the correct remaining age.
    """

    points = []
    weights = []
    for template_index, template in enumerate(sessions):
        for call_index, call in enumerate(template.calls):
            if call_index == 0 or call.cached_prefix_tokens <= 0:
                continue
            gap_ns = max(1, template.calls[call_index - 1].tool_duration_ns)
            points.append((template_index, call_index, gap_ns))
            weights.append(gap_ns)
    if not points:
        raise RuntimeError("session pool has no resumable calls")

    residents = []
    for rank, (template_index, start, gap_ns) in enumerate(
            rng.choices(points, weights=weights, k=target_l)):
        template = sessions[template_index]
        clone = S.clone_session(template, S.RESIDENT_ID_BASE + rank)
        truncated = replace(clone, calls=clone.calls[start:])
        past = clone.calls[:start]
        arrival_ns = rng.randrange(gap_ns)
        mean_fresh_in = statistics.fmean(
            max(1, call.input_tokens - call.cached_prefix_tokens)
            for call in past)
        residents.append({
            "session": truncated,
            "context_tokens": clone.calls[start].cached_prefix_tokens,
            "arrival_ns": arrival_ns,
            "rank": rank,
            "past_calls": start,
            "future_calls": len(truncated.calls),
            "sitting_gap_ns": gap_ns,
            "idle_age_ns": gap_ns - arrival_ns,
            "mean_fresh_in": mean_fresh_in,
        })
    return residents


def build_system(system_key: str, hardware):
    from serving.core.gpu_pd_hbf_prefill import (
        build_hetero_system_from_config)
    from serving.core.gpu_pd_single_system import (
        SingleFiniteHBMTieredBaseline,
        SingleStrictInfiniteHBMOracle,
    )
    from serving.core.gpu_pd_tier_lifecycle import (
        RESTORE_EXECUTION_BULK,
        RESTORE_EXECUTION_LAYERWISE,
    )

    engine = dict(C.ENGINE)
    if system_key == "baseline_cpu_ssd":
        # Layerwise streaming is the generous default: SSD restores hide
        # under the suffix prefill's per-layer compute.  The bulk mode
        # (the repo-wide default elsewhere) exposes the whole transfer
        # and serves as the sensitivity's other pole.
        restore_mode = (
            RESTORE_EXECUTION_BULK
            if os.environ.get("LLMSIM_BASELINE_RESTORE_MODE") == "bulk"
            else RESTORE_EXECUTION_LAYERWISE)
        return SingleFiniteHBMTieredBaseline(
            repo_root=REPO_ROOT, hardware=hardware, policy="cpu_ssd",
            restore_execution_mode=restore_mode,
            **engine)
    if system_key == "oracle_infinite_hbm":
        return SingleStrictInfiniteHBMOracle(
            repo_root=REPO_ROOT, hardware=hardware, **engine)
    if system_key == "hbf_prefill_p4d4":
        return build_hetero_system_from_config(
            repo_root=REPO_ROOT, config_path=HETERO_CONFIG, **engine)
    raise ValueError(f"unknown system {system_key!r}")


def preload(system, system_key: str, residents, now_ns: int):
    # The hetero system's node shares the tiered baseline's structure --
    # its "CPU" ledger is the HBF home -- so the same LRU fill applies:
    # most recent into D, then HBF, then SSD.
    mapped = (
        "baseline_cpu_ssd"
        if system_key == "hbf_prefill_p4d4" else system_key
    )
    placed = S.preload_population(system, mapped, residents, now_ns=now_ns)
    if system_key == "hbf_prefill_p4d4" and "cpu" in placed:
        placed = dict(placed)
        placed["hbf"] = placed.pop("cpu")
    return placed


def _bucket_stats(
        values_by_bucket: dict[str, list[float]],
        buckets=GAP_BUCKETS) -> dict:
    out = {}
    for _, label in buckets:
        values = values_by_bucket.get(label, [])
        out[label] = {
            "count": len(values),
            "mean": C._mean(values),
            "p50": C._percentile(values, 0.50),
            "p90": C._percentile(values, 0.90),
            "p95": C._percentile(values, 0.95),
        }
    return out


def conditioned_metrics(completed, residents, scheduled) -> tuple[dict, dict]:
    """Bucket resume TTFT/turn latency by idle gap and by context size.

    A resume's gap is the trace's tool/think time (release minus the
    predecessor's user completion).  A preloaded resident's first
    in-window call resumes a session whose full pre-resume gap is the
    sampled sitting gap, which spans t=0.  The context axis buckets the
    same events by the resume call's reused-prefix size.
    """

    spec_by_key = {}
    for sched in scheduled:
        for call in sched.session.calls:
            spec_by_key[
                (sched.session.session_id, call.call_index)] = call
    completion_by_key = {
        (c.key.session_id, c.key.sub_request_index): c for c in completed}
    sitting_gap_by_session = {
        r["session"].session_id: r["sitting_gap_ns"] for r in residents}
    first_index_seen: dict[str, int] = {}
    for c in completed:
        sid = c.key.session_id
        idx = c.key.sub_request_index
        first_index_seen[sid] = min(idx, first_index_seen.get(sid, idx))

    gap_ttft: dict[str, list[float]] = {}
    gap_turn: dict[str, list[float]] = {}
    ctx_ttft: dict[str, list[float]] = {}
    ctx_turn: dict[str, list[float]] = {}
    unmatched = 0
    for c in completed:
        sid = c.key.session_id
        idx = c.key.sub_request_index
        if idx == 0:
            continue
        predecessor = completion_by_key.get((sid, idx - 1))
        if predecessor is not None:
            gap_ns = c.release_ns - predecessor.completion_ns
        elif (
            sid in sitting_gap_by_session
            and idx == first_index_seen[sid]
        ):
            gap_ns = sitting_gap_by_session[sid]
        else:
            unmatched += 1
            continue
        ttft_s = (c.first_token_ns - c.release_ns) / 1e9
        turn_s = (c.completion_ns - c.release_ns) / 1e9
        label = gap_bucket(max(0.0, gap_ns / 1e9))
        gap_ttft.setdefault(label, []).append(ttft_s)
        gap_turn.setdefault(label, []).append(turn_s)
        spec = spec_by_key.get((sid, idx))
        if spec is not None:
            ctx_label = context_bucket(float(spec.cached_prefix_tokens))
            ctx_ttft.setdefault(ctx_label, []).append(ttft_s)
            ctx_turn.setdefault(ctx_label, []).append(turn_s)
    gap_out = {
        "bucket_edges_s": [edge for edge, _ in GAP_BUCKETS[:-1]],
        "unmatched_resumes": unmatched,
        "resume_ttft_s": _bucket_stats(gap_ttft),
        "turn_latency_s": _bucket_stats(gap_turn),
    }
    ctx_out = {
        "bucket_edges_tokens": [
            edge for edge, _ in CONTEXT_BUCKETS[:-1]],
        "resume_ttft_s": _bucket_stats(ctx_ttft, CONTEXT_BUCKETS),
        "turn_latency_s": _bucket_stats(ctx_turn, CONTEXT_BUCKETS),
    }
    return gap_out, ctx_out


def write_accounting(system, system_key: str, completed, scheduled) -> dict:
    """Gross media traffic for the endurance analysis."""

    spec_by_key = {}
    for sched in scheduled:
        for call in sched.session.calls:
            spec_by_key[(sched.session.session_id, call.call_index)] = call

    produced_tokens = 0
    for c in completed:
        spec = spec_by_key.get(
            (c.key.session_id, c.key.sub_request_index))
        if spec is None:
            continue
        fresh = max(0, spec.input_tokens - spec.cached_prefix_tokens)
        produced_tokens += fresh + max(0, spec.output_tokens - 1)
    produced_kv_bytes = produced_tokens * KV_AGGREGATE_BYTES_PER_TOKEN

    lifecycle = getattr(getattr(system, "node", None), "lifecycle", None)
    ssd_write_bytes = ssd_read_bytes = 0
    if lifecycle is not None:
        ssd_write_bytes = int(lifecycle.metrics.ssd_write_bytes)
        ssd_read_bytes = int(lifecycle.metrics.ssd_read_bytes)

    pool = getattr(getattr(system, "node", None), "pool", None)
    handoff_bytes = (
        int(pool.metrics.handoff_aggregate_bytes)
        if pool is not None else 0)

    out = {
        "produced_kv_bytes": produced_kv_bytes,
        "ssd_write_bytes": ssd_write_bytes,
        "ssd_read_bytes": ssd_read_bytes,
        "nvlink_handoff_bytes": handoff_bytes,
    }
    if system_key == "hbf_prefill_p4d4":
        # Every produced token's KV commits once into the HBF home, and
        # SSD restores land back in HBF.  Demotions D-to-HBF are
        # ownership updates (the durable copy already exists).
        out["hbf_media_write_bytes"] = (
            produced_kv_bytes + ssd_read_bytes)
    return out


def run_cell(task: dict) -> dict:
    from serving.core.gpu_pd_latency import load_p4d4_gpu_config
    from serving.core.hbf_comparison_workload import ScheduledSession

    family = task["family"]
    rate = task["rate"]
    seed = task["seed"]
    system_key = task["system"]
    target_calls = task["measured_calls"]
    out_path = Path(task["out_path"])

    sessions, lifetimes = S.load_pool(family)
    mean_w = S.mean_lifetime_s(lifetimes)
    target_l = max(1, int(round(rate * mean_w)))

    hardware = load_p4d4_gpu_config(GPU_CONFIG)
    system = build_system(system_key, hardware)
    rng = random.Random(S.PRELOAD_SHUFFLE_SEED + seed)
    started = time.time()

    calls_per_session = statistics.fmean(len(s.calls) for s in sessions)
    mean_think_ns = statistics.fmean(
        call.tool_duration_ns for s in sessions for call in s.calls) or 1
    call_rate = max(
        1e-12, target_l * calls_per_session / max(1e-9, mean_w))
    horizon_ns = int((target_calls / call_rate) * 1e9)
    # Long-gap resumes are rare per unit time, so the >1h buckets need a
    # window that is long in *simulated* hours, not in call count.  The
    # floor mainly matters at low rates, where extra simulated hours are
    # cheap (wall cost scales with calls, not with the horizon).
    min_horizon_ns = int(task.get("min_horizon_s", 0) * 1e9)
    horizon_ns = max(horizon_ns, min_horizon_ns)

    residents = build_residents_steady(sessions, target_l, rng)
    arrivals = S.build_arrivals(sessions, rate, seed, horizon_ns)

    offers = [
        (resident["arrival_ns"], resident["session"],
         S.RESIDENT_ID_BASE + resident["rank"])
        for resident in residents
    ]
    offers.extend(arrivals)
    offers.sort(key=lambda item: (item[0], item[2]))
    scheduled = tuple(
        ScheduledSession(
            offer_index=order,
            session=template,
            arrival_time_ns=arrival_ns,
            unit_interarrival=0.0,
            unit_arrival_time=float(order),
        )
        for order, (arrival_ns, template, _instance) in enumerate(offers)
    )
    if not scheduled:
        raise RuntimeError(
            f"no sessions generated for rate={rate} horizon={horizon_ns}")

    system.load(scheduled)
    node = getattr(system, "node", None)
    if node is not None and hasattr(node, "set_spec_total_calls"):
        for sched in scheduled:
            node.set_spec_total_calls(
                sched.session.session_id, len(sched.session.calls))
    seeded_cursors = S.seed_call_cursors(system, residents)
    placed = preload(system, system_key, residents, now_ns=0)
    system.run_until(horizon_ns)
    completed = [
        system._completed_snapshots[request_id]
        for request_id in system._completed_ids
    ]
    wall_s = time.time() - started
    peak_rss_mb = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 1024.0

    first, resume, tpot, turn = [], [], [], []
    level_pass = {name: 0 for name in C.SLO_LEVELS}
    level_tok = {name: 0 for name in C.SLO_LEVELS}
    output_tokens = 0
    window_end = 0
    for call in completed:
        ttft = (call.first_token_ns - call.release_ns) / 1e9
        is_first = call.key.sub_request_index == 0
        (first if is_first else resume).append(ttft)
        per_token_ms = (
            (call.completion_ns - call.first_token_ns) / 1e6
            / (call.output_tokens - 1)
            if call.output_tokens > 1 else 0.0)
        tpot.append(per_token_ms)
        turn.append((call.completion_ns - call.release_ns) / 1e9)
        for name, (f_s, r_s, t_ms) in C.SLO_LEVELS.items():
            if ttft <= (f_s if is_first else r_s) and per_token_ms <= t_ms:
                level_pass[name] += 1
                level_tok[name] += call.output_tokens
        output_tokens += call.output_tokens
        window_end = max(window_end, call.completion_ns)
    scored = len(completed)
    window_s = max(1e-9, horizon_ns / 1e9)

    gap_conditioned, context_conditioned = conditioned_metrics(
        completed, residents, scheduled)

    counters: dict[str, int] = {}
    try:
        report = system.report()
    except Exception as error:
        report_error = repr(error)
    else:
        report_error = None
        C._walk_numbers(report, counters)

    row = {
        "schema_version": SCHEMA_VERSION,
        "family": family, "rate": rate, "seed": seed,
        "system": system_key,
        "topology": "one_server_8_cards",
        "mean_session_lifetime_s": mean_w,
        "target_concurrency": target_l,
        "preloaded": placed,
        "pool_sessions": len(sessions),
        "calls_per_session_mean": calls_per_session,
        "resident_sessions": len(residents),
        "seeded_cursors": seeded_cursors,
        "offered_arrivals": len(arrivals),
        "offered_sessions": len(scheduled),
        "horizon_s": horizon_ns / 1e9,
        "last_completion_s": window_end / 1e9,
        "scored_calls": scored,
        "measurement_window_s": window_s,
        "phase_breakdown": phase_breakdown.extract(system, completed),
        "gap_conditioned": gap_conditioned,
        "context_conditioned": context_conditioned,
        "write_accounting": write_accounting(
            system, system_key, completed, scheduled),
        "first_ttft_p50_s": C._percentile(first, 0.50),
        "first_ttft_p95_s": C._percentile(first, 0.95),
        "first_ttft_p99_s": C._percentile(first, 0.99),
        "resume_ttft_p50_s": C._percentile(resume, 0.50),
        "resume_ttft_p95_s": C._percentile(resume, 0.95),
        "resume_ttft_p99_s": C._percentile(resume, 0.99),
        "tpot_p50_ms": C._percentile(tpot, 0.50),
        "tpot_p95_ms": C._percentile(tpot, 0.95),
        "tpot_p99_ms": C._percentile(tpot, 0.99),
        "tpot_mean_ms": C._mean(tpot),
        "turn_latency_mean_s": C._mean(turn),
        "turn_latency_p50_s": C._percentile(turn, 0.50),
        "turn_latency_p99_s": C._percentile(turn, 0.99),
        "slo_levels": {
            name: {
                "pass_fraction": (
                    level_pass[name] / scored if scored else 0.0),
                "good_output_tokens_per_s": level_tok[name] / window_s,
            } for name in C.SLO_LEVELS},
        "output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / window_s,
        "wall_s": wall_s, "peak_rss_mb": peak_rss_mb,
        "report_error": report_error,
        "counters": {k: counters.get(k, 0) for k in C.COUNTER_KEYS},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_path)
    return row


def build_tasks(root, families, rates, seeds, systems, measured_calls,
                resume_existing, min_horizon_s=0.0):
    tasks = []
    skipped = 0
    for family in families:
        for rate in rates:
            for seed in seeds:
                for system_key in systems:
                    out_path = (
                        root / "cells" / family / f"rate-{rate:g}"
                        / f"seed-{seed}-{system_key}.json")
                    if resume_existing and out_path.is_file():
                        skipped += 1
                        continue
                    tasks.append({
                        "family": family, "rate": rate, "seed": seed,
                        "system": system_key,
                        "measured_calls": measured_calls,
                        "min_horizon_s": min_horizon_s,
                        "out_path": str(out_path),
                    })
    return tasks, skipped


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=SCRIPT_ROOT / "hbf_prefill_v1")
    parser.add_argument(
        "--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    parser.add_argument(
        "--rates", type=float, nargs="+", default=list(DEFAULT_RATES))
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    parser.add_argument(
        "--measured-calls", type=int, default=DEFAULT_MEASURED_CALLS)
    parser.add_argument(
        "--min-horizon-s", type=float, default=0.0,
        help="floor on the simulated window, in seconds; raises the "
             "horizon beyond the call-count target so multi-hour idle "
             "gaps can complete inside the window")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    root = args.output_root
    tasks, skipped = build_tasks(
        root, args.families, args.rates, args.seeds, args.systems,
        args.measured_calls, args.resume,
        min_horizon_s=args.min_horizon_s)
    total = len(tasks)
    print(
        f"hbf-prefill campaign: {total} cells "
        f"(skipped {skipped} existing)  workers={args.workers}",
        flush=True)
    started = time.time()
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_cell, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            label = (
                f"{task['family']} rate={task['rate']:<8g}"
                f"seed={task['seed']} {task['system']:<20s}")
            try:
                row = future.result()
            except Exception as error:
                failures += 1
                print(f"[{index}/{total}] {label} FAILED: {error!r}",
                      flush=True)
                continue
            print(
                f"[{index}/{total}] {label}"
                f" L={row['target_concurrency']:>5}"
                f" calls={row['scored_calls']:>6,}"
                f" resume_p95={row['resume_ttft_p95_s']:>8.3f}s"
                f" tpot_p95={row['tpot_p95_ms']:>7.1f}ms"
                f" turn_mean={row['turn_latency_mean_s']:>7.2f}s"
                f" wall={row['wall_s']:>6.0f}s",
                flush=True)
    print(
        f"complete in {(time.time() - started) / 60:.1f} min "
        f"({failures} failures)",
        flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Session-count scaling campaign for the three compared systems.

The arrival-rate axis is inert for this trace: above ~0.16 sessions/s the
arrival span is negligible against the session lifetime, so every session is
already co-resident and the knob is exhausted.  Concurrent KV residency is
therefore swept directly, by cohort size N.

Cohorts are nested (N=64 sessions are a subset of N=128) so the axis reads as
"add load", and each cohort is split 25/50/25 into disjoint warmup,
measurement, and guard partitions.  Only measurement-partition calls are
scored; warmup and guard exist to keep the measured window away from the
cold-start and drain transients.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
import random
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(os.environ.get(
    "LLMSIM_REPO", Path(__file__).resolve().parents[2]))
TRACE_PATH = Path(os.environ.get(
    "LLMSIM_TRACE",
    Path.home() / "llmsim-data"
    / "tracelab-schema3-sps0.2-final.jsonl"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_ROOT = Path(os.environ.get(
    "LLMSIM_RESULTS", REPO_ROOT / "results"))

SCHEMA_VERSION = 1
SOURCE_SESSION_COUNT = 4_281
COHORT_SHUFFLE_SEED = 20260727
ROLE_SHUFFLE_SEED = 20260728

DEFAULT_SESSION_COUNTS = (64, 128, 192, 256, 384, 512, 768, 1024)
DEFAULT_SEEDS = (101, 102, 103)
# Session starts are Poisson at this rate.  Above ~0.02 sessions/s the
# arrival span is negligible against session lifetime, so this does not
# change steady-state concurrency -- but it does set how tightly the
# cold-start first calls bunch together.  At rate 1.0 every first call of a
# 256-session cohort lands inside the first ~0.02% of the timeline, which
# makes first-call TTFT a thundering-herd measurement rather than a
# steady-state one.  Lower it to spread the cold start.
SESSION_RATE = float(os.environ.get("LLMSIM_SESSION_RATE", "1.0"))

SYSTEMS = (
    "baseline_cpu_ssd",
    "oracle_infinite_hbm",
    "hbf_tp8_context",
    "hbf_tp4x2",
)
# Relative single-threaded cost per call, measured on this host.  Used only
# to start the long poles first; it never affects results.
SYSTEM_COST_WEIGHT = {
    "baseline_cpu_ssd": 1.0,
    "oracle_infinite_hbm": 0.6,
    "hbf_tp8_context": 8.0,
    "hbf_tp4x2": 8.0,
}

FIRST_TTFT_SLO_SECONDS = 30.0
RESUME_TTFT_SLO_SECONDS = 30.0
TPOT_SLO_MILLISECONDS = 300.0

# The legacy 30s/30s/300ms thresholds saturate near 1.0 for every system on
# this trace, so they cannot rank the systems.  Each cell scores three
# preregistered threshold sets; the tighter ones keep resolution once the
# loose one saturates.  All are computed from the same per-call data.
SLO_LEVELS = {
    "loose": (10.0, 10.0, 150.0),
    "medium": (5.0, 5.0, 100.0),
    "tight": (3.0, 3.0, 50.0),
}

# Pinned engine configuration, identical for every system.
ENGINE = dict(
    max_num_batched_tokens=131_072,
    max_num_seqs=128,
    p_max_num_seqs=32,
    d_max_num_seqs=128,
    max_prefill_chunk_tokens=131_072,
    validate_every_event=False,
)
HBF_ACTIVE_MEMORY = dict(
    capacity_gib_per_card=16.0, bandwidth_gbps_per_card=409.6)


# --------------------------------------------------------------------------
# Cohort construction
# --------------------------------------------------------------------------

def master_ordering() -> tuple[int, ...]:
    """Return one fixed shuffle of every source index in the release."""

    order = list(range(SOURCE_SESSION_COUNT))
    random.Random(COHORT_SHUFFLE_SEED).shuffle(order)
    return tuple(order)


def cohort_indices(session_count: int) -> tuple[int, ...]:
    """Return the nested cohort of ``session_count`` source indices."""

    if session_count <= 0 or session_count > SOURCE_SESSION_COUNT:
        raise ValueError(
            f"session_count must be in 1..{SOURCE_SESSION_COUNT}")
    return tuple(master_ordering()[:session_count])


def role_partition(session_count: int) -> dict[str, tuple[int, ...]]:
    """Split one cohort into disjoint warmup/measurement/guard positions."""

    warmup = session_count // 4
    guard = session_count // 4
    measurement = session_count - warmup - guard
    positions = list(range(session_count))
    random.Random(ROLE_SHUFFLE_SEED + session_count).shuffle(positions)
    return {
        "warmup": tuple(sorted(positions[:warmup])),
        "measurement": tuple(
            sorted(positions[warmup:warmup + measurement])),
        "guard": tuple(sorted(positions[warmup + measurement:])),
    }


# --------------------------------------------------------------------------
# One cell
# --------------------------------------------------------------------------

def _build_system(system_key: str, hardware):
    from serving.core.gpu_pd_dual_oracle import (
        ROUTE_BALANCED_TRACE_WORK, DualStrictInfiniteHBMOracle)
    from serving.core.gpu_pd_dual_tiered import DualFiniteHBMTieredBaseline
    from serving.core.hbf_design_tco import lpddr_active_memory
    from serving.ssd_hbf_design_sweep import (
        make_design_spec, make_design_system)

    if system_key == "baseline_cpu_ssd":
        return DualFiniteHBMTieredBaseline(
            repo_root=REPO_ROOT, hardware=hardware, policy="cpu_ssd",
            route_policy=ROUTE_BALANCED_TRACE_WORK, **ENGINE)
    if system_key == "oracle_infinite_hbm":
        return DualStrictInfiniteHBMOracle(
            repo_root=REPO_ROOT, hardware=hardware,
            route_policy=ROUTE_BALANCED_TRACE_WORK, **ENGINE)
    if system_key == "hbf_tp8_context":
        spec = make_design_spec(
            hbf_layout="tp8_context",
            migration_policy=os.environ.get(
                "LLMSIM_MIGRATION_POLICY", "load_aware"),
            active_memory=lpddr_active_memory(**HBF_ACTIVE_MEMORY),
        )
        return make_design_system(repo_root=REPO_ROOT, spec=spec)
    if system_key == "hbf_tp4x2":
        spec = make_design_spec(
            hbf_layout="tp4x2",
            migration_policy=os.environ.get(
                "LLMSIM_MIGRATION_POLICY", "load_aware"),
            active_memory=lpddr_active_memory(**HBF_ACTIVE_MEMORY),
        )
        return make_design_system(repo_root=REPO_ROOT, spec=spec)
    raise ValueError(f"unknown system {system_key!r}")


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _walk_numbers(node: Any, out: dict[str, int], leaf: str = "") -> None:
    """Sum integer counters by leaf name across a nested report mapping.

    Reports nest the same counter under different parents (per node, per
    replica, aggregate), so keying by leaf name and summing gives one
    system-wide total without hard-coding each report's shape.
    """

    if isinstance(node, Mapping):
        for key, value in node.items():
            _walk_numbers(value, out, str(key))
    elif isinstance(node, (list, tuple)):
        for item in node:
            _walk_numbers(item, out, leaf)
    elif isinstance(node, int) and not isinstance(node, bool):
        if leaf:
            out[leaf] = out.get(leaf, 0) + node


def _find_write_accounting(node: Any) -> dict | None:
    """Return the HBF write-accounting block from anywhere in a report.

    The endurance projection divides cumulative writes by the *whole* run
    duration, which dilutes the steady-state rate with the arrival ramp and
    the long drain tail and therefore overstates HBF lifetime.  The cell
    keeps the raw cumulative bytes so post-processing can also divide by the
    measurement window and bracket the true rate.
    """

    if isinstance(node, Mapping):
        if "hbf_write_accounting" in node:
            block = node["hbf_write_accounting"]
            if isinstance(block, Mapping):
                return {
                    key: block.get(key)
                    for key in (
                        "total_physical_write_bytes",
                        "wasted_physical_write_bytes",
                        "complete_for_endurance_projection",
                    )
                }
        for value in node.values():
            found = _find_write_accounting(value)
            if found is not None:
                return found
    elif isinstance(node, (list, tuple)):
        for item in node:
            found = _find_write_accounting(item)
            if found is not None:
                return found
    return None


COUNTER_KEYS = (
    "cpu_prepare_hits", "ssd_prepare_hits", "d_prepare_hits",
    "d_drops", "d_to_cpu_started", "d_to_ssd_started",
    "capacity_evictions", "migrations_committed",
    "gpu_hbm_idle_release_events", "promotion_load_deferrals",
    "recompute_resumes", "lower_tier_hits", "stable_d_hits",
)


def run_cell(task: dict) -> dict:
    from serving.core.gpu_pd_latency import load_p4d4_gpu_config
    from serving.core.hbf_comparison_workload import (
        build_offered_plan, load_comparison_workload)

    n = task["session_count"]
    seed = task["seed"]
    system_key = task["system"]
    out_path = Path(task["out_path"])

    indices = sorted(cohort_indices(n))
    hardware = load_p4d4_gpu_config(
        REPO_ROOT / "configs/wakekv_hbf/p4d4_gpu_server.json")
    workload = load_comparison_workload(TRACE_PATH, source_indices=indices)
    plan = build_offered_plan(workload.sessions, seed=seed)
    scheduled = plan.at_rate(SESSION_RATE)

    # Roles are assigned over the frozen offered order so every system and
    # seed scores exactly the same session set.
    roles = role_partition(n)
    offer_role = {}
    for role, positions in roles.items():
        for position in positions:
            offer_role[position] = role
    scored_sessions = {
        item.session.session_id
        for item in scheduled
        if offer_role[item.offer_index] == "measurement"
    }

    system = _build_system(system_key, hardware)
    started = time.time()
    completed = system.run(scheduled)
    wall_s = time.time() - started
    peak_rss_mb = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 1024.0

    first, resume, tpot, turn_latency = [], [], [], []
    passes = scored = 0
    level_passes = {name: 0 for name in SLO_LEVELS}
    level_output_tokens = {name: 0 for name in SLO_LEVELS}
    window_start = None
    window_end = 0
    output_tokens = 0
    for call in completed:
        if call.key.session_id not in scored_sessions:
            continue
        scored += 1
        ttft = (call.first_token_ns - call.release_ns) / 1e9
        is_first = call.key.sub_request_index == 0
        (first if is_first else resume).append(ttft)
        per_token_ms = (
            (call.completion_ns - call.first_token_ns) / 1e6
            / (call.output_tokens - 1)
            if call.output_tokens > 1 else 0.0
        )
        tpot.append(per_token_ms)
        # Service time for one call: release -> completion, with the
        # tool/human wait that follows it excluded.
        turn_latency.append(
            (call.completion_ns - call.release_ns) / 1e9)
        slo = FIRST_TTFT_SLO_SECONDS if is_first else RESUME_TTFT_SLO_SECONDS
        if ttft <= slo and per_token_ms <= TPOT_SLO_MILLISECONDS:
            passes += 1
        for name, (first_s, resume_s, tpot_ms) in SLO_LEVELS.items():
            limit = first_s if is_first else resume_s
            if ttft <= limit and per_token_ms <= tpot_ms:
                level_passes[name] += 1
                level_output_tokens[name] += call.output_tokens
        output_tokens += call.output_tokens
        window_start = (
            call.release_ns if window_start is None
            else min(window_start, call.release_ns))
        window_end = max(window_end, call.completion_ns)

    window_s = max(1e-9, (window_end - (window_start or 0)) / 1e9)
    counters: dict[str, int] = {}
    write_accounting = None
    try:
        report = system.report()
    except Exception as error:  # report shape differs per system
        report_error = repr(error)
    else:
        report_error = None
        _walk_numbers(report, counters)
        write_accounting = _find_write_accounting(report)

    row = {
        "schema_version": SCHEMA_VERSION,
        "session_count": n,
        "seed": seed,
        "system": system_key,
        "session_rate": SESSION_RATE,
        "migration_policy": os.environ.get(
            "LLMSIM_MIGRATION_POLICY", "load_aware"),
        "kernel_semantics": os.environ.get(
            "LLMSIM_KERNEL_SEMANTICS", "v2_default"),
        "scored_calls": scored,
        "total_calls": len(completed),
        "measurement_sessions": len(scored_sessions),
        "first_ttft_p50_s": _percentile(first, 0.50),
        "first_ttft_p95_s": _percentile(first, 0.95),
        "first_ttft_p99_s": _percentile(first, 0.99),
        "first_ttft_mean_s": _mean(first),
        "resume_ttft_p50_s": _percentile(resume, 0.50),
        "resume_ttft_p95_s": _percentile(resume, 0.95),
        "resume_ttft_p99_s": _percentile(resume, 0.99),
        "resume_ttft_mean_s": _mean(resume),
        "tpot_p50_ms": _percentile(tpot, 0.50),
        "tpot_p95_ms": _percentile(tpot, 0.95),
        "tpot_p99_ms": _percentile(tpot, 0.99),
        "tpot_mean_ms": _mean(tpot),
        "turn_latency_mean_s": _mean(turn_latency),
        "turn_latency_p50_s": _percentile(turn_latency, 0.50),
        "turn_latency_p99_s": _percentile(turn_latency, 0.99),
        "joint_slo_pass_fraction": passes / scored if scored else 0.0,
        "slo_levels": {
            name: {
                "thresholds": {
                    "first_ttft_s": SLO_LEVELS[name][0],
                    "resume_ttft_s": SLO_LEVELS[name][1],
                    "tpot_ms": SLO_LEVELS[name][2],
                },
                "pass_fraction": (
                    level_passes[name] / scored if scored else 0.0),
                "good_output_tokens": level_output_tokens[name],
                "good_output_tokens_per_s": (
                    level_output_tokens[name] / max(
                        1e-9, (window_end - (window_start or 0)) / 1e9)),
            }
            for name in SLO_LEVELS
        },
        "output_tokens": output_tokens,
        "measurement_window_s": window_s,
        "output_tokens_per_s": output_tokens / window_s,
        "wall_s": wall_s,
        "peak_rss_mb": peak_rss_mb,
        "report_error": report_error,
        # Endurance denominators are bracketed rather than assumed: the
        # full horizon is the optimistic bound, the measurement window the
        # conservative one.
        "full_horizon_s": (window_end / 1e9) if window_end else 0.0,
        "hbf_write_accounting": write_accounting,
        "counters": {
            key: counters.get(key, 0) for key in COUNTER_KEYS},
        "peak_bytes": {
            key: counters.get(key, 0)
            for key in (
                "peak_used_bytes", "hbf_reserved_bytes_peak")},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_path)
    return row


# --------------------------------------------------------------------------
# Campaign driver
# --------------------------------------------------------------------------

def build_tasks(root: Path, counts, seeds, systems) -> list[dict]:
    tasks = []
    for n in counts:
        for seed in seeds:
            for system in systems:
                out_path = (
                    root / "cells" / f"n-{n}" / f"seed-{seed}-{system}.json")
                tasks.append({
                    "session_count": n, "seed": seed, "system": system,
                    "out_path": str(out_path),
                    "cost": n * SYSTEM_COST_WEIGHT[system],
                })
    # Longest first so the wall clock is bounded by the true long poles.
    tasks.sort(key=lambda item: -item["cost"])
    return tasks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=RESULTS_ROOT / "session_scaling_v2")
    parser.add_argument(
        "--session-counts", type=int, nargs="+",
        default=list(DEFAULT_SESSION_COUNTS))
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = args.output_root
    tasks = build_tasks(
        root, args.session_counts, args.seeds, args.systems)
    if args.resume:
        tasks = [
            task for task in tasks
            if not Path(task["out_path"]).is_file()]
    print(f"campaign root : {root}")
    print(f"session counts: {args.session_counts}")
    print(f"seeds         : {args.seeds}")
    print(f"systems       : {args.systems}")
    print(f"pending cells : {len(tasks)}  workers={args.workers}",
          flush=True)
    if args.dry_run:
        for task in tasks[:10]:
            print("  ", task["out_path"])
        return 0
    if not tasks:
        print("nothing to do")
        return 0

    root.mkdir(parents=True, exist_ok=True)
    (root / "campaign_metadata.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "load_axis": "cohort_session_count",
        "load_axis_rationale": (
            "arrival rate saturates once the arrival span is negligible "
            "against session lifetime; cohort size sweeps concurrent KV "
            "residency directly"),
        "session_rate_fixed": SESSION_RATE,
        "session_counts": list(args.session_counts),
        "seeds": list(args.seeds),
        "systems": list(args.systems),
        "cohort_shuffle_seed": COHORT_SHUFFLE_SEED,
        "role_shuffle_seed": ROLE_SHUFFLE_SEED,
        "partition_fractions": "25% warmup / 50% measurement / 25% guard",
        "cohorts_are_nested": True,
        "slo": {
            "first_ttft_seconds": FIRST_TTFT_SLO_SECONDS,
            "resume_ttft_seconds": RESUME_TTFT_SLO_SECONDS,
            "tpot_milliseconds": TPOT_SLO_MILLISECONDS,
        },
        "engine": ENGINE,
        "hbf_active_memory": HBF_ACTIVE_MEMORY,
        "migration_policy": os.environ.get(
            "LLMSIM_MIGRATION_POLICY", "load_aware"),
        "trace_path": str(TRACE_PATH),
        "repo_root": str(REPO_ROOT),
    }, indent=2, sort_keys=True) + "\n")

    done = 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_cell, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            done += 1
            try:
                row = future.result()
            except Exception as error:
                print(f"[{done}/{len(tasks)}] FAILED n={task['session_count']} "
                      f"seed={task['seed']} {task['system']}: {error!r}",
                      flush=True)
                continue
            elapsed = time.time() - started
            print(
                f"[{done}/{len(tasks)}] n={row['session_count']:5d} "
                f"seed={row['seed']} {row['system']:20} "
                f"wall={row['wall_s']:8.1f}s "
                f"jointSLO={row['joint_slo_pass_fraction']:.4f} "
                f"resume_p95={row['resume_ttft_p95_s']:8.2f} "
                f"tok/s={row['output_tokens_per_s']:8.1f} "
                f"rss={row['peak_rss_mb']:6.0f}MB "
                f"[{elapsed/60:.1f} min elapsed]",
                flush=True)
    print(f"campaign complete in {(time.time()-started)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

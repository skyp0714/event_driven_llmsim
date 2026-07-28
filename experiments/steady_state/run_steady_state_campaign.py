#!/usr/bin/env python3
"""Open-system steady-state campaign: arrival rate is the load knob.

The earlier closed-cohort design could not use arrival rate as a load knob.
All N sessions arrived inside N/lambda seconds and then never left, so above
~0.02 sessions/s the arrival span was negligible against the ~1.2e6 s session
lifetime and the rate stopped changing anything.

Here sessions arrive Poisson from a large pool, run their recorded call
sequence, and leave.  Little's Law then makes concurrency a function of the
knob: L = lambda * W, with W the mean session lifetime measured from the
pool.  Reaching that equilibrium by simulation would cost a full W of warmup
-- an order of magnitude more work than the measurement window it enables --
so the resident population is injected directly at t=0 via the lifecycles'
`preload_session`, at the placement the policy would have produced:
LRU-ordered fill of D-HBM, then CPU, then SSD for the tiering baseline; for
the HBF system a maturity-correlated split in which the most mature half of
the population (by context tokens) is HBF-resident and the remainder fills
the GPU ladder youngest-first (D-HBM, then CPU, then SSD).

The measurement ends on a call count, not a simulated duration, so cost is
bounded by construction.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from dataclasses import replace
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
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
SESSION_SCALING_ROOT = SCRIPT_ROOT.parent / "session_scaling"
if str(SESSION_SCALING_ROOT) not in sys.path:
    sys.path.insert(0, str(SESSION_SCALING_ROOT))

import phase_breakdown  # noqa: E402
import run_session_scaling_campaign as C  # noqa: E402

REPO_ROOT = C.REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 3
FAMILIES = ("claude", "codex")
SYSTEMS = (
    "baseline_cpu_ssd",
    "oracle_infinite_hbm",
    "hbf_tp8_context",
    "hbf_tp4x2",
)
DEFAULT_RATES = (0.0005, 0.001, 0.002, 0.004, 0.008, 0.016)
DEFAULT_SEEDS = (101, 102, 103)
DEFAULT_MEASURED_CALLS = 6_000
PRELOAD_SHUFFLE_SEED = 20260728
# Resident instances are numbered from here so their identities can never
# collide with the Poisson arrivals numbered from zero.
RESIDENT_ID_BASE = 10_000_000

KV_BYTES_PER_TOKEN_PER_RANK = 24_576


def family_of(session_id: str) -> str:
    return session_id.split(":", 1)[0]


def load_pool(family: str):
    """Return every session of one agent family, with its lifetime."""

    from serving.core.hbf_comparison_workload import load_comparison_workload

    indices = []
    with C.TRACE_PATH.open() as handle:
        for index, line in enumerate(handle):
            if line.startswith('{"session_id":"' + family + ':'):
                indices.append(index)
    if not indices:
        raise SystemExit(f"no sessions for family {family!r}")
    workload = load_comparison_workload(
        C.TRACE_PATH, source_indices=tuple(indices))
    sessions = list(workload.sessions)
    lifetimes = []
    for session in sessions:
        span = sum(call.tool_duration_ns for call in session.calls)
        lifetimes.append(span)
    return sessions, lifetimes


def mean_lifetime_s(lifetimes: Sequence[int]) -> float:
    return statistics.fmean(lifetimes) / 1e9 if lifetimes else 0.0


def clone_session(template, instance: int):
    """Return a distinct session instance of one pool template.

    Sampling the pool with replacement is what keeps the pool size from
    capping offered load, but every live session needs its own identity:
    the schedulers key placement and lineage on session_id.
    """

    session_id = f"{template.session_id}#{instance}"
    calls = tuple(
        replace(call, session_id=session_id, source_index=instance)
        for call in template.calls)
    return replace(
        template, session_id=session_id, source_index=instance, calls=calls)


def build_arrivals(sessions, rate: float, seed: int, horizon_ns: int):
    """Poisson arrivals drawn with replacement from the pool.

    Sampling with replacement removes the pool size as a ceiling on the
    offered load: a session may appear more than once as distinct instances.
    """

    from serving.core.hbf_comparison_workload import ScheduledSession

    rng = random.Random(seed)
    out = []
    now = 0.0
    index = 0
    while True:
        now += rng.expovariate(rate)
        if now * 1e9 > horizon_ns:
            break
        template = sessions[rng.randrange(len(sessions))]
        out.append((int(now * 1e9), clone_session(template, index), index))
        index += 1
    return out


def build_residents(sessions, target_l: int, rng, stagger_ns: int):
    """Build the resident population as live, mid-conversation sessions.

    A resident is not ballast.  It is a session that arrived some time ago,
    already holds a KV context, and will keep issuing calls throughout the
    measurement window -- which is the only way the population can be what
    generates the offered load.  Each resident is a pool template truncated
    at a random resume point, so its first scored call is a resume against
    a context that is pre-placed in the tiers.

    The stagger spreads their first calls uniformly over one mean think-time
    so the window opens with the population at random phase in its
    call/tool-call cycle rather than as a synchronised burst.
    """

    residents = []
    attempts = 0
    while len(residents) < target_l and attempts < target_l * 20:
        attempts += 1
        template = sessions[rng.randrange(len(sessions))]
        resume_points = [
            index for index, call in enumerate(template.calls)
            if index > 0 and call.cached_prefix_tokens > 0]
        if not resume_points:
            continue
        start = resume_points[rng.randrange(len(resume_points))]
        rank = len(residents)
        clone = clone_session(template, RESIDENT_ID_BASE + rank)
        truncated = replace(clone, calls=clone.calls[start:])
        residents.append({
            "session": truncated,
            "context_tokens": clone.calls[start].cached_prefix_tokens,
            "arrival_ns": rng.randrange(max(1, stagger_ns)),
            "rank": rank,
        })
    return residents


def _system_nodes(system):
    nodes = list(getattr(system, "nodes", ()))
    single = getattr(system, "node", None)
    if single is not None:
        nodes.append(single)
    return nodes


def _seed_lineage(node, session_id: str, last_index: int, tokens: int):
    """Create the node's session record as if earlier calls had run.

    Nodes track two cursors: a submission cursor that enforces contiguous
    call order, and a lineage record that enforces the same thing again at
    dispatch.  Both are seeded from -1, which is correct for a session that
    starts at call 0 and wrong for a resident that starts mid-conversation.
    """

    cursor = getattr(node, "_last_submitted_call_index", None)
    if cursor is not None:
        cursor[session_id] = last_index

    records = getattr(node, "sessions", None)
    if records is None or session_id in records:
        return
    factory = type(node).__init__  # placeholder to keep the import local
    del factory
    module = type(node).__module__
    if "tiered" in module:
        from serving.core.gpu_pd_tiered_node import TieredSessionLineage
        records[session_id] = TieredSessionLineage(
            session_id=session_id, last_call_index=last_index)
    elif "oracle" in module:
        from serving.core.gpu_pd_oracle_node import OracleSessionPlacement
        # Unbounded HBM: a resident's context is already materialised and
        # resident, so the oracle pays no restore for it.  That is the bound.
        records[session_id] = OracleSessionPlacement(
            session_id=session_id, last_call_index=last_index,
            materialized_tokens=tokens, d_resident=True)
    elif "hybrid" in module:
        from serving.core.gpu_hbf_hybrid import HybridSession
        records[session_id] = HybridSession(
            session_id=session_id, last_internal_call_index=last_index)


def seed_call_cursors(system, residents) -> int:
    """Tell the nodes that a resident's earlier calls already happened."""

    nodes = _system_nodes(system)
    route = getattr(system, "node_for_session", None)
    seeded = 0
    for resident in residents:
        first_index = resident["session"].calls[0].call_index
        if first_index == 0:
            continue
        session_id = resident["session"].session_id
        targets = nodes
        if route is not None:
            try:
                routed = route(session_id)
            except (KeyError, ValueError):
                routed = None
            if isinstance(routed, int):
                # node_for_session returns a node id, not the node itself.
                routed = nodes[routed] if 0 <= routed < len(nodes) else None
            if routed is not None:
                targets = [routed]
        for node in targets:
            _seed_lineage(
                node, session_id, first_index - 1,
                resident["context_tokens"])
            seeded += 1
    return seeded


def preload_population(system, system_key, residents, now_ns):
    """Place each resident's existing KV where the policy would have put it."""

    from serving.core.gpu_pd_tier_lifecycle import Tier

    # LRU order: the tier fill has to match what eviction would have
    # produced, so the most recently used residents land highest.
    order = sorted(residents, key=lambda r: r["arrival_ns"])

    placed = {"d": 0, "cpu": 0, "ssd": 0, "hbf": 0, "skipped": 0}
    if system_key == "oracle_infinite_hbm":
        # Unbounded HBM: the oracle has no tier to place into, and its
        # residents pay no restore, which is exactly the bound we want.
        placed["skipped"] = len(order)
        return placed
    if system_key == "baseline_cpu_ssd":
        nodes = list(system.nodes)
        for rank, resident in enumerate(order):
            tokens = resident["context_tokens"]
            node = nodes[rank % len(nodes)]
            lifecycle = node.lifecycle
            per_rank = lifecycle._per_rank_bytes(tokens)
            aggregate = lifecycle._aggregate_bytes(tokens)
            if lifecycle.d_ledger.free_bytes >= per_rank:
                tier = Tier.D
            elif lifecycle.cpu_ledger.free_bytes >= aggregate:
                tier = Tier.CPU
            elif lifecycle.ssd_ledger.free_bytes >= aggregate:
                tier = Tier.SSD
            else:
                placed["skipped"] += 1
                continue
            lifecycle.preload_session(
                resident["session"].session_id, tokens, tier=tier,
                now_ns=now_ns,
                last_access_ns=max(0, now_ns - rank * 1_000_000))
            placed[tier.value] += 1
        return placed
    # HBF hybrid: the equilibrium of a maturity-aware policy is a
    # correlated split, not full HBF residency.  The HBF host's bound is
    # decode occupancy rather than media capacity, so only half the
    # resident population lives there -- the most mature half by context.
    # The remainder mirrors the GPU-side ladder the staged policy leaves
    # behind: the youngest sessions are still hot in D-HBM, the next ones
    # sit on CPU, and the durable overflow is an SSD checkpoint.  Read
    # most-mature-first that is HBF -> SSD -> CPU -> D-HBM.
    node = getattr(system, "node", None)
    lifecycle = getattr(system, "hbf_lifecycle", None)
    if lifecycle is None:
        lifecycle = node.hbf_lifecycle
    gpu_lifecycle = node.gpu_lifecycle
    recency_rank = {
        resident["session"].session_id: rank
        for rank, resident in enumerate(order)
    }

    def access_ns(resident):
        return max(
            0,
            now_ns
            - recency_rank[resident["session"].session_id] * 1_000_000,
        )

    mature = sorted(order, key=lambda r: -r["context_tokens"])
    hbf_budget = (len(mature) + 1) // 2
    gpu_side = []
    for resident in mature[:hbf_budget]:
        try:
            lifecycle.preload_session(
                resident["session"].session_id, resident["context_tokens"],
                now_ns=now_ns,
                last_access_ns=access_ns(resident))
        except RuntimeError:
            gpu_side.append(resident)
            continue
        placed["hbf"] += 1
    gpu_side.extend(mature[hbf_budget:])
    # Youngest-first fill keeps the freshest sessions in D-HBM and pushes
    # the more mature remainder down to CPU and then SSD.
    gpu_side.sort(key=lambda r: r["context_tokens"])
    for resident in gpu_side:
        tokens = resident["context_tokens"]
        session_id = resident["session"].session_id
        per_rank = gpu_lifecycle._per_rank_bytes(tokens)
        aggregate = gpu_lifecycle._aggregate_bytes(tokens)
        if gpu_lifecycle.d_ledger.free_bytes >= per_rank:
            tier = Tier.D
        elif gpu_lifecycle.cpu_ledger.free_bytes >= aggregate:
            tier = Tier.CPU
        elif gpu_lifecycle.ssd_ledger.free_bytes >= aggregate:
            tier = Tier.SSD
        else:
            placed["skipped"] += 1
            continue
        gpu_lifecycle.preload_session(
            session_id, tokens, tier=tier,
            now_ns=now_ns,
            last_access_ns=access_ns(resident))
        lifecycle.preload_gpu_session(
            session_id, tokens,
            now_ns=now_ns,
            durable_ssd=(tier == Tier.SSD),
            last_access_ns=access_ns(resident))
        placed[tier.value] += 1
    return placed


def run_cell(task: dict) -> dict:
    from serving.core.gpu_pd_latency import load_p4d4_gpu_config

    family = task["family"]
    rate = task["rate"]
    seed = task["seed"]
    system_key = task["system"]
    target_calls = task["measured_calls"]
    out_path = Path(task["out_path"])

    sessions, lifetimes = load_pool(family)
    mean_w = mean_lifetime_s(lifetimes)
    target_l = max(1, int(round(rate * mean_w)))

    hardware = load_p4d4_gpu_config(
        REPO_ROOT / "configs/wakekv_hbf/p4d4_gpu_server.json")
    system = C._build_system(system_key, hardware)
    rng = random.Random(PRELOAD_SHUFFLE_SEED + seed)
    started = time.time()

    calls_per_session = statistics.fmean(len(s.calls) for s in sessions)
    mean_think_ns = statistics.fmean(
        call.tool_duration_ns for s in sessions for call in s.calls) or 1
    # The window has to hold `target_calls` at the population's call rate.
    # Residents issue calls_per_session calls over their lifetime W, so the
    # population emits target_l * calls_per_session / W calls per second.
    call_rate = max(1e-12, target_l * calls_per_session / max(1e-9, mean_w))
    horizon_ns = int((target_calls / call_rate) * 1e9)

    residents = build_residents(
        sessions, target_l, rng, stagger_ns=int(mean_think_ns))

    # Poisson arrivals replace the residents that depart during the window.
    arrivals = build_arrivals(sessions, rate, seed, horizon_ns)

    from serving.core.hbf_comparison_workload import ScheduledSession
    offers = [
        (resident["arrival_ns"], resident["session"],
         RESIDENT_ID_BASE + resident["rank"])
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
    # Load first so routing is decided, then install the residents' history
    # on the node each one actually landed on.
    system.load(scheduled)
    seeded_cursors = seed_call_cursors(system, residents)
    placed = preload_population(system, system_key, residents, now_ns=0)
    # Cut on the window rather than draining: the residents carry many more
    # calls than the window needs, and draining them would measure a
    # population that is emptying instead of one in steady state.
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
    # The window is the cutoff, not the last completion: throughput has to be
    # divided by the time the system was actually offered load.
    window_s = max(1e-9, horizon_ns / 1e9)

    counters: dict[str, int] = {}
    write_accounting = None
    try:
        report = system.report()
    except Exception as error:
        report_error = repr(error)
    else:
        report_error = None
        C._walk_numbers(report, counters)
        write_accounting = C._find_write_accounting(report)

    row = {
        "schema_version": SCHEMA_VERSION,
        "family": family, "rate": rate, "seed": seed,
        "system": system_key,
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
                "pass_fraction": level_pass[name] / scored if scored else 0.0,
                "good_output_tokens_per_s": level_tok[name] / window_s,
            } for name in C.SLO_LEVELS},
        "output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / window_s,
        "wall_s": wall_s, "peak_rss_mb": peak_rss_mb,
        "report_error": report_error,
        "counters": {k: counters.get(k, 0) for k in C.COUNTER_KEYS},
        "peak_bytes": {
            k: counters.get(k, 0)
            for k in ("peak_used_bytes", "hbf_reserved_bytes_peak")},
        "hbf_write_accounting": write_accounting,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_path)
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=SCRIPT_ROOT / "steady_state_v1")
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
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    tasks = []
    for family in args.families:
        for rate in args.rates:
            for seed in args.seeds:
                for system in args.systems:
                    out_path = (
                        args.output_root / "cells" / family
                        / f"rate-{rate:g}" / f"seed-{seed}-{system}.json")
                    tasks.append({
                        "family": family, "rate": rate, "seed": seed,
                        "system": system, "out_path": str(out_path),
                        "measured_calls": args.measured_calls,
                        "cost": rate * C.SYSTEM_COST_WEIGHT[system]})
    if args.resume:
        tasks = [t for t in tasks if not Path(t["out_path"]).is_file()]
    tasks.sort(key=lambda t: -t["cost"])
    print(f"steady-state campaign: {len(tasks)} cells  "
          f"workers={args.workers}  measured_calls={args.measured_calls}",
          flush=True)
    if not tasks:
        print("nothing to do")
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)

    done = 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_cell, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            done += 1
            try:
                row = future.result()
            except Exception as error:
                print(f"[{done}/{len(tasks)}] FAILED {task['family']} "
                      f"rate={task['rate']} {task['system']}: {error!r}",
                      flush=True)
                continue
            print(f"[{done}/{len(tasks)}] {row['family']:6} "
                  f"rate={row['rate']:<7g} seed={row['seed']} "
                  f"{row['system']:20} L={row['target_concurrency']:5d} "
                  f"calls={row['scored_calls']:6,} "
                  f"resume_p99={row['resume_ttft_p99_s']:8.2f} "
                  f"tpot_p99={row['tpot_p99_ms']:7.1f} "
                  f"turn_mean={row['turn_latency_mean_s']:7.2f} "
                  f"wall={row['wall_s']:6.0f}s", flush=True)
    print(f"complete in {(time.time()-started)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

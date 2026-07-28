#!/usr/bin/env python3
"""Policy and architecture variant sweep for the HBF system.

Run after the main session-scaling campaign, at the cohort size where the
HBF result needs explaining.  Every variant is scored with the same cohort,
seeds, partitions, and SLO levels as the main campaign, so variants are
directly comparable to the campaign's `hbf_tp8_context` cells.

Two families:

* policy    -- migration/promotion and HBF read mode.  No hardware change.
* hardware  -- LPDDR width/capacity, intra-server fabric, HBF read
               bandwidth.  These carry a TCO consequence and must be priced
               before any of them is claimed as a win.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import resource
import sys
import time
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_session_scaling_campaign as C  # noqa: E402

REPO_ROOT = C.REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# variant_key -> (migration_policy, read_mode, hw_overrides, lpddr_overrides)
VARIANTS: dict[str, tuple[str, str, dict, dict]] = {
    # --- reference (identical to the main campaign cell) ---
    "ref_load_aware": ("load_aware", "demand", {}, {}),

    # --- policy family: no hardware change ---
    "pol_composite_ready_adaptive": (
        "composite_ready_adaptive", "demand", {}, {}),
    "pol_composite_ready": ("composite_ready", "demand", {}, {}),
    "pol_eager": ("eager", "demand", {}, {}),
    "pol_tool_or_human_immediate": (
        "tool_or_human_immediate", "demand", {}, {}),
    "pol_prefetch_read": ("load_aware", "prefetch", {}, {}),

    # --- hardware family: priced in the TCO model ---
    "hw_lpddr_2x": (
        "load_aware", "demand", {},
        {"bandwidth_gbps_per_card": 819.2}),
    "hw_lpddr_4x": (
        "load_aware", "demand", {},
        {"bandwidth_gbps_per_card": 1638.4}),
    "hw_lpddr_cap_32g": (
        "load_aware", "demand", {},
        {"capacity_gib_per_card": 32.0}),
    # Raising only the per-card link is inert: collective_runtime_ns takes
    # max(per_card, cards_per_root * per_card / root), and at 50 GB/s card
    # with a 200 GB/s root shared by four cards the two terms are exactly
    # equal, so the root stays binding.  The root and inter-root links must
    # move together for the fabric to stop being the constraint.
    "hw_fabric_ualink": (
        "load_aware", "demand",
        {"intra_fabric_bandwidth_gbps_per_card": 200.0}, {}),
    "hw_fabric_full": (
        "load_aware", "demand",
        {"intra_fabric_bandwidth_gbps_per_card": 200.0,
         "pcie_root_bandwidth_gbps": 800.0,
         "pcie_inter_root_bandwidth_gbps": 400.0}, {}),
    "hw_lpddr_2x_fabric_full": (
        "load_aware", "demand",
        {"intra_fabric_bandwidth_gbps_per_card": 200.0,
         "pcie_root_bandwidth_gbps": 800.0,
         "pcie_inter_root_bandwidth_gbps": 400.0},
        {"bandwidth_gbps_per_card": 819.2}),
    "hw_lpddr_2x_fabric_ualink": (
        "load_aware", "demand",
        {"intra_fabric_bandwidth_gbps_per_card": 200.0},
        {"bandwidth_gbps_per_card": 819.2}),
    "hw_hbf_read_2x": (
        "load_aware", "demand",
        {"hbf_read_bandwidth_gbps_per_card": 6700.0}, {}),
}

BASE_LPDDR = dict(C.HBF_ACTIVE_MEMORY)


def build_variant_system(variant_key: str):
    from serving.core.gpu_pd_latency import load_p4d4_gpu_config
    from serving.core.gpu_ssd_hbf_hybrid import SSDStagedGPUHBFSystem
    from serving.core.hbf_full_model_latency import load_hbf_server_config

    policy, read_mode, hw_overrides, lpddr_overrides = VARIANTS[variant_key]
    lpddr = {**BASE_LPDDR, **lpddr_overrides}

    gpu_hardware = load_p4d4_gpu_config(
        REPO_ROOT / "configs/wakekv_hbf/p4d4_gpu_server.json")
    hbf_hardware, _ = load_hbf_server_config(
        REPO_ROOT / "configs/wakekv_hbf/full_model_8card_server.json")
    hbf_hardware = replace(
        hbf_hardware,
        lpddr_capacity_bytes_per_card=int(
            lpddr["capacity_gib_per_card"] * 1024 ** 3),
        lpddr_bandwidth_gbps_per_card=lpddr["bandwidth_gbps_per_card"],
        hbf_read_prefetch_enabled=(read_mode == "prefetch"),
        **hw_overrides,
    )
    hbf_hardware.validate()
    return SSDStagedGPUHBFSystem(
        repo_root=REPO_ROOT,
        gpu_hardware=gpu_hardware,
        hbf_hardware=hbf_hardware,
        hbf_layout="tp8_context",
        promotion_policy=policy,
        max_num_batched_tokens=C.ENGINE["max_num_batched_tokens"],
        max_num_seqs=C.ENGINE["max_num_seqs"],
        p_max_num_seqs=C.ENGINE["p_max_num_seqs"],
        d_max_num_seqs=C.ENGINE["d_max_num_seqs"],
        max_prefill_chunk_tokens=C.ENGINE["max_prefill_chunk_tokens"],
        hbf_mixed_batch_latency_limit_ns=None,
        restore_execution_mode="bulk",
        validate_every_event=False,
    )


def run_variant_cell(task: dict) -> dict:
    from serving.core.hbf_comparison_workload import (
        build_offered_plan, load_comparison_workload)

    n = task["session_count"]
    seed = task["seed"]
    variant_key = task["variant"]
    out_path = Path(task["out_path"])

    indices = sorted(C.cohort_indices(n))
    workload = load_comparison_workload(
        C.TRACE_PATH, source_indices=indices)
    plan = build_offered_plan(workload.sessions, seed=seed)
    scheduled = plan.at_rate(C.SESSION_RATE)

    roles = C.role_partition(n)
    offer_role = {}
    for role, positions in roles.items():
        for position in positions:
            offer_role[position] = role
    scored_sessions = {
        item.session.session_id for item in scheduled
        if offer_role[item.offer_index] == "measurement"}

    system = build_variant_system(variant_key)
    started = time.time()
    completed = system.run(scheduled)
    wall_s = time.time() - started
    peak_rss_mb = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 1024.0

    first, resume, tpot = [], [], []
    level_passes = {name: 0 for name in C.SLO_LEVELS}
    level_tokens = {name: 0 for name in C.SLO_LEVELS}
    scored = 0
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
            if call.output_tokens > 1 else 0.0)
        tpot.append(per_token_ms)
        for name, (f_s, r_s, t_ms) in C.SLO_LEVELS.items():
            limit = f_s if is_first else r_s
            if ttft <= limit and per_token_ms <= t_ms:
                level_passes[name] += 1
                level_tokens[name] += call.output_tokens
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
    except Exception as error:
        report_error = repr(error)
    else:
        report_error = None
        C._walk_numbers(report, counters)
        write_accounting = C._find_write_accounting(report)

    policy, read_mode, hw_overrides, lpddr_overrides = VARIANTS[variant_key]
    row = {
        "schema_version": C.SCHEMA_VERSION,
        "variant": variant_key,
        "variant_family": (
            "reference" if variant_key.startswith("ref_")
            else "policy" if variant_key.startswith("pol_")
            else "hardware"),
        "migration_policy": policy,
        "hbf_read_mode": read_mode,
        "hardware_overrides": hw_overrides,
        "lpddr": {**BASE_LPDDR, **lpddr_overrides},
        "session_count": n,
        "seed": seed,
        "system": "hbf_tp8_context",
        "scored_calls": scored,
        "total_calls": len(completed),
        "first_ttft_p50_s": C._percentile(first, 0.50),
        "first_ttft_p95_s": C._percentile(first, 0.95),
        "resume_ttft_p50_s": C._percentile(resume, 0.50),
        "resume_ttft_p95_s": C._percentile(resume, 0.95),
        "tpot_p50_ms": C._percentile(tpot, 0.50),
        "tpot_p95_ms": C._percentile(tpot, 0.95),
        "slo_levels": {
            name: {
                "pass_fraction": (
                    level_passes[name] / scored if scored else 0.0),
                "good_output_tokens_per_s": (
                    level_tokens[name] / window_s),
            }
            for name in C.SLO_LEVELS
        },
        "output_tokens_per_s": output_tokens / window_s,
        "measurement_window_s": window_s,
        "full_horizon_s": (window_end / 1e9) if window_end else 0.0,
        "wall_s": wall_s,
        "peak_rss_mb": peak_rss_mb,
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
        default=C.RESULTS_ROOT / "hbf_variant_sweep_v2")
    parser.add_argument("--session-count", type=int, default=512)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(C.DEFAULT_SEEDS))
    parser.add_argument(
        "--variants", nargs="+", default=list(VARIANTS),
        choices=list(VARIANTS))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    tasks = []
    for variant in args.variants:
        for seed in args.seeds:
            out_path = (
                args.output_root / "cells"
                / f"n-{args.session_count}"
                / f"seed-{seed}-{variant}.json")
            tasks.append({
                "variant": variant, "seed": seed,
                "session_count": args.session_count,
                "out_path": str(out_path)})
    if args.resume:
        tasks = [t for t in tasks if not Path(t["out_path"]).is_file()]
    print(f"variant sweep at N={args.session_count}: "
          f"{len(tasks)} cells, workers={args.workers}", flush=True)
    if not tasks:
        print("nothing to do")
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "variant_definitions.json").write_text(json.dumps(
        {k: {"migration_policy": v[0], "hbf_read_mode": v[1],
             "hardware_overrides": v[2], "lpddr_overrides": v[3]}
         for k, v in VARIANTS.items()}, indent=2, sort_keys=True) + "\n")

    done = 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_variant_cell, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            done += 1
            try:
                row = future.result()
            except Exception as error:
                print(f"[{done}/{len(tasks)}] FAILED {task['variant']} "
                      f"seed={task['seed']}: {error!r}", flush=True)
                continue
            tight = row["slo_levels"]["tight"]["pass_fraction"]
            print(f"[{done}/{len(tasks)}] {row['variant']:28} "
                  f"seed={row['seed']} wall={row['wall_s']:7.1f}s "
                  f"tightSLO={tight:.4f} "
                  f"resume_p95={row['resume_ttft_p95_s']:7.2f} "
                  f"tpot_p95={row['tpot_p95_ms']:7.1f} "
                  f"first_p95={row['first_ttft_p95_s']:7.2f} "
                  f"[{(time.time()-started)/60:.1f} min]", flush=True)
    print(f"variant sweep complete in {(time.time()-started)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Aggregate the session-scaling campaign and diagnose the HBF result.

Produces `aggregate.csv` (mean + bootstrap-free normal CI95 across seeds)
and prints the comparison table plus a bottleneck attribution for HBF.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import math
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_ROOT = Path(os.environ.get(
    "LLMSIM_RESULTS", Path(__file__).resolve().parents[2] / "results"))
DEFAULT_ROOT = RESULTS_ROOT / "session_scaling_v2"

SLO_LEVEL_NAMES = ("loose", "medium", "tight")

METRICS = (
    "joint_slo_pass_fraction",
    "slo_loose", "slo_medium", "slo_tight",
    "goodput_loose", "goodput_medium", "goodput_tight",
    "first_ttft_p50_s", "first_ttft_p95_s",
    "resume_ttft_p50_s", "resume_ttft_p95_s",
    "tpot_p50_ms", "tpot_p95_ms",
    "output_tokens_per_s",
    "measurement_window_s", "wall_s", "peak_rss_mb",
)
COUNTERS = (
    "cpu_prepare_hits", "ssd_prepare_hits", "d_prepare_hits",
    "d_drops", "d_to_cpu_started", "d_to_ssd_started",
    "capacity_evictions", "migrations_committed",
    "promotion_load_deferrals", "recompute_resumes",
    "lower_tier_hits", "stable_d_hits",
)
SYSTEM_ORDER = (
    "baseline_cpu_ssd", "oracle_infinite_hbm", "hbf_tp8_context")
LABEL = {
    "baseline_cpu_ssd": "2xHBM+SSD tiering",
    "oracle_infinite_hbm": "Infinite-HBM Oracle",
    "hbf_tp8_context": "HBM+HBF (tp8_context)",
}
# 8 cards x (1.28 TB - per-card weights) / replication, from the tp8_context
# layout: usable logical KV bytes for the whole HBF server.
HBF_LOGICAL_KV_BYTES = 10_179_000_000_000


def load_cells(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "cells").glob("*/*.json")):
        with path.open() as handle:
            row = json.load(handle)
        # Flatten the preregistered SLO threshold sets so they aggregate
        # through the same mean/CI path as every other metric.
        for name in SLO_LEVEL_NAMES:
            level = (row.get("slo_levels") or {}).get(name) or {}
            row[f"slo_{name}"] = level.get("pass_fraction")
            row[f"goodput_{name}"] = level.get("good_output_tokens_per_s")
        rows.append(row)
    return rows


def ci95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["session_count"], row["system"])].append(row)
    out = []
    for (count, system), cells in sorted(grouped.items()):
        record = {
            "session_count": count,
            "system": system,
            "label": LABEL[system],
            "seed_count": len(cells),
            "seeds": ";".join(str(c["seed"]) for c in sorted(
                cells, key=lambda c: c["seed"])),
        }
        for metric in METRICS:
            values = [
                float(c[metric]) for c in cells
                if c.get(metric) is not None
                and not math.isnan(float(c[metric]))
            ]
            if not values:
                continue
            mean, low, high = ci95(values)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci95_lower"] = low
            record[f"{metric}_ci95_upper"] = high
        for counter in COUNTERS:
            record[f"{counter}_mean"] = statistics.fmean(
                [float(c["counters"].get(counter, 0)) for c in cells])
        record["scored_calls_mean"] = statistics.fmean(
            [float(c["scored_calls"]) for c in cells])
        write_bytes = [
            float((c.get("hbf_write_accounting") or {}).get(
                "total_physical_write_bytes") or 0.0)
            for c in cells
        ]
        record["hbf_write_bytes_mean"] = statistics.fmean(write_bytes)
        horizon = statistics.fmean(
            [float(c["full_horizon_s"]) for c in cells])
        window = statistics.fmean(
            [float(c["measurement_window_s"]) for c in cells])
        record["full_horizon_s_mean"] = horizon
        # Endurance rate is bracketed: the full horizon dilutes the rate
        # with ramp and drain, the measurement window is the tighter
        # (pessimistic) denominator.
        record["hbf_write_tb_per_day_optimistic"] = (
            record["hbf_write_bytes_mean"] / max(horizon, 1e-9)
            * 86_400.0 / 1e12)
        record["hbf_write_tb_per_day_conservative"] = (
            record["hbf_write_bytes_mean"] / max(window, 1e-9)
            * 86_400.0 / 1e12)
        record["hbf_reserved_bytes_peak_mean"] = statistics.fmean(
            [float(c["peak_bytes"].get("hbf_reserved_bytes_peak", 0))
             for c in cells])
        record["hbf_fill_fraction"] = (
            record["hbf_reserved_bytes_peak_mean"] / HBF_LOGICAL_KV_BYTES)
        out.append(record)
    return out


def write_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def print_tables(records: list[dict]) -> None:
    by_count: dict[int, dict[str, dict]] = defaultdict(dict)
    for record in records:
        by_count[record["session_count"]][record["system"]] = record

    for level in SLO_LEVEL_NAMES:
        metric = f"slo_{level}_mean"
        if not any(record.get(metric) is not None for record in records):
            continue
        print(f"\n=== SLO pass fraction, {level} thresholds "
              f"(higher better) ===")
        head = f"{'N':>6}" + "".join(
            f"{LABEL[s][:20]:>22}" for s in SYSTEM_ORDER) + f"{'HBF/base':>10}"
        print(head)
        for count in sorted(by_count):
            line = f"{count:6d}"
            for system in SYSTEM_ORDER:
                rec = by_count[count].get(system)
                value = rec.get(metric) if rec else None
                line += (
                    f"{value:22.4f}" if value is not None else f"{'-':>22}")
            base = by_count[count].get("baseline_cpu_ssd")
            hbf = by_count[count].get("hbf_tp8_context")
            ratio = float("nan")
            if base and hbf and base.get(metric):
                ratio = hbf[metric] / base[metric]
            print(line + f"{ratio:10.3f}")

    print("\n=== joint SLO pass fraction (higher better) ===")
    header = f"{'N':>6}" + "".join(
        f"{LABEL[s][:20]:>22}" for s in SYSTEM_ORDER) + f"{'HBF/base':>10}"
    print(header)
    for count in sorted(by_count):
        line = f"{count:6d}"
        for system in SYSTEM_ORDER:
            rec = by_count[count].get(system)
            line += (
                f"{rec['joint_slo_pass_fraction_mean']:22.4f}"
                if rec else f"{'-':>22}")
        base = by_count[count].get("baseline_cpu_ssd")
        hbf = by_count[count].get("hbf_tp8_context")
        ratio = (
            hbf["joint_slo_pass_fraction_mean"]
            / base["joint_slo_pass_fraction_mean"]
            if base and hbf and base["joint_slo_pass_fraction_mean"] else
            float("nan"))
        print(line + f"{ratio:10.3f}")

    print("\n=== resume TTFT p95 seconds (lower better) ===")
    print(header)
    for count in sorted(by_count):
        line = f"{count:6d}"
        for system in SYSTEM_ORDER:
            rec = by_count[count].get(system)
            line += (
                f"{rec['resume_ttft_p95_s_mean']:22.3f}"
                if rec else f"{'-':>22}")
        base = by_count[count].get("baseline_cpu_ssd")
        hbf = by_count[count].get("hbf_tp8_context")
        ratio = (
            base["resume_ttft_p95_s_mean"] / hbf["resume_ttft_p95_s_mean"]
            if base and hbf and hbf["resume_ttft_p95_s_mean"] else
            float("nan"))
        print(line + f"{ratio:10.2f}x")

    print("\n=== output tokens/s (higher better) ===")
    print(header)
    for count in sorted(by_count):
        line = f"{count:6d}"
        for system in SYSTEM_ORDER:
            rec = by_count[count].get(system)
            line += (
                f"{rec['output_tokens_per_s_mean']:22.1f}"
                if rec else f"{'-':>22}")
        base = by_count[count].get("baseline_cpu_ssd")
        hbf = by_count[count].get("hbf_tp8_context")
        ratio = (
            hbf["output_tokens_per_s_mean"]
            / base["output_tokens_per_s_mean"]
            if base and hbf and base["output_tokens_per_s_mean"] else
            float("nan"))
        print(line + f"{ratio:10.3f}")

    print("\n=== capacity pressure / tier activity ===")
    print(f"{'N':>6} {'system':>22} {'lowerTier':>10} {'d_hits':>9} "
          f"{'evict':>8} {'migr':>7} {'loadDefer':>10} {'HBF fill':>9}")
    for count in sorted(by_count):
        for system in SYSTEM_ORDER:
            rec = by_count[count].get(system)
            if not rec:
                continue
            lower = (
                rec["cpu_prepare_hits_mean"] + rec["ssd_prepare_hits_mean"])
            print(f"{count:6d} {LABEL[system][:22]:>22} {lower:10.0f} "
                  f"{rec['d_prepare_hits_mean']:9.0f} "
                  f"{rec['capacity_evictions_mean']:8.0f} "
                  f"{rec['migrations_committed_mean']:7.0f} "
                  f"{rec['promotion_load_deferrals_mean']:10.0f} "
                  f"{rec['hbf_fill_fraction']*100:8.1f}%")

    print("\n=== HBF write endurance rate (bracketed) ===")
    print(f"{'N':>6} {'TB/day optimistic':>20} {'TB/day conservative':>21} "
          f"{'dilution':>10}")
    for count in sorted(by_count):
        rec = by_count[count].get("hbf_tp8_context")
        if not rec or not rec["hbf_write_bytes_mean"]:
            continue
        opt = rec["hbf_write_tb_per_day_optimistic"]
        con = rec["hbf_write_tb_per_day_conservative"]
        print(f"{count:6d} {opt:20.2f} {con:21.2f} "
              f"{(con / opt if opt else float('nan')):9.2f}x")


def diagnose(records: list[dict]) -> None:
    """Attribute the HBF result at the largest completed cohort."""

    by_count: dict[int, dict[str, dict]] = defaultdict(dict)
    for record in records:
        by_count[record["session_count"]][record["system"]] = record
    counts = sorted(by_count)
    if not counts:
        return
    top = counts[-1]
    base = by_count[top].get("baseline_cpu_ssd")
    hbf = by_count[top].get("hbf_tp8_context")
    oracle = by_count[top].get("oracle_infinite_hbm")
    if not (base and hbf):
        return
    print(f"\n=== diagnosis at N={top} ===")
    slo_ratio = (
        hbf["joint_slo_pass_fraction_mean"]
        / max(base["joint_slo_pass_fraction_mean"], 1e-9))
    print(f"  HBF/baseline joint SLO : {slo_ratio:.3f}")
    if oracle:
        print(f"  HBF/oracle   joint SLO : "
              f"{hbf['joint_slo_pass_fraction_mean'] / max(oracle['joint_slo_pass_fraction_mean'], 1e-9):.3f}")
    print(f"  HBF fill at peak       : {hbf['hbf_fill_fraction']*100:.1f}% "
          f"of {HBF_LOGICAL_KV_BYTES/1e12:.2f} TB logical KV")
    print(f"  HBF lower-tier hits    : "
          f"{hbf['cpu_prepare_hits_mean'] + hbf['ssd_prepare_hits_mean']:.0f}"
          f" of {hbf['scored_calls_mean']:.0f} scored calls")
    print(f"  HBF promotion deferrals: "
          f"{hbf['promotion_load_deferrals_mean']:.0f}")
    print(f"  HBF capacity evictions : "
          f"{hbf['capacity_evictions_mean']:.0f}")
    print(f"  HBF migrations         : "
          f"{hbf['migrations_committed_mean']:.0f}")
    verdict = []
    if slo_ratio >= 1.0:
        verdict.append("HBF >= baseline on joint SLO")
    else:
        verdict.append("HBF BELOW baseline -- investigate")
        if hbf["promotion_load_deferrals_mean"] > (
                hbf["migrations_committed_mean"]):
            verdict.append(
                "load_aware admission is deferring more promotions than it "
                "commits: policy-addressable (raise hysteresis / relax gate)")
        if hbf["capacity_evictions_mean"] > 0:
            verdict.append(
                "HBF capacity evictions are non-zero: eviction policy or "
                "cohort size is binding")
        if hbf["hbf_fill_fraction"] < 0.5 and (
                hbf["capacity_evictions_mean"] == 0):
            verdict.append(
                "HBF is not capacity-bound; the loss is compute or "
                "collective bound, NOT policy-addressable")
    for line in verdict:
        print(f"  -> {line}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    rows = load_cells(args.root)
    if not rows:
        print(f"no cells under {args.root}")
        return 1
    print(f"loaded {len(rows)} cells from {args.root}")
    records = aggregate(rows)
    write_csv(records, args.root / "aggregate.csv")
    (args.root / "aggregate.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n")
    print_tables(records)
    diagnose(records)
    print(f"\nwrote {args.root/'aggregate.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

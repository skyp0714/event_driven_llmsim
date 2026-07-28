#!/usr/bin/env python3
"""Aggregate the steady-state cells into one row per (family, system, rate).

Seeds are replicates of the same operating point, so every metric is reported
as its across-seed mean with the half-width of a 95% t-interval beside it.
Counters and the phase decomposition are carried through the same way, which
is what lets a later plot attribute a latency change to admission, restore, or
prefill without going back to the per-cell files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent

# 95% two-sided t critical values, indexed by degrees of freedom.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}

SCALAR_METRICS = (
    "first_ttft_p50_s", "first_ttft_p95_s", "first_ttft_p99_s",
    "resume_ttft_p50_s", "resume_ttft_p95_s", "resume_ttft_p99_s",
    "tpot_p50_ms", "tpot_p95_ms", "tpot_p99_ms", "tpot_mean_ms",
    "turn_latency_mean_s", "turn_latency_p50_s", "turn_latency_p99_s",
    "output_tokens_per_s", "scored_calls", "measurement_window_s",
    "horizon_s", "target_concurrency", "resident_sessions",
    "offered_arrivals", "output_tokens", "wall_s", "peak_rss_mb",
)

PHASE_KEYS = (
    "admission_wait", "restore_transfer", "prefill_queue_and_compute",
    "resume_ttft_total",
)


def load_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*.json")):
        if path.suffix != ".json" or path.name.endswith(".tmp"):
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        # aggregate.json / economics.json live beside the cells.
        if isinstance(record, dict) and "system" in record:
            rows.append(record)
    return rows


def mean_ci(values):
    clean = [v for v in values
             if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return None, None
    mean = statistics.fmean(clean)
    if len(clean) < 2:
        return mean, 0.0
    stdev = statistics.stdev(clean)
    half = T95.get(len(clean) - 1, 1.96) * stdev / math.sqrt(len(clean))
    return mean, half


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["system"], row["rate"])].append(row)

    slo_levels = sorted({
        name for row in rows for name in (row.get("slo_levels") or {})})
    counter_keys = sorted({
        key for row in rows for key in (row.get("counters") or {})})

    out = []
    for (family, system, rate), cells in sorted(groups.items()):
        record = {
            "family": family,
            "system": system,
            "rate": rate,
            "seeds": ";".join(str(c["seed"]) for c in sorted(
                cells, key=lambda c: c["seed"])),
            "n_seeds": len(cells),
        }
        for metric in SCALAR_METRICS:
            mean, half = mean_ci([c.get(metric) for c in cells])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci95"] = half

        for level in slo_levels:
            for field, alias in (
                ("pass_fraction", "slo_pass"),
                ("good_output_tokens_per_s", "goodput"),
            ):
                mean, half = mean_ci([
                    (c.get("slo_levels") or {}).get(level, {}).get(field)
                    for c in cells])
                record[f"{alias}_{level}_mean"] = mean
                record[f"{alias}_{level}_ci95"] = half

        for key in counter_keys:
            mean, _ = mean_ci([
                (c.get("counters") or {}).get(key) for c in cells])
            record[f"counter_{key}_mean"] = mean

        # Phase decomposition: mean and share, so a plot can stack them.
        for phase in PHASE_KEYS:
            for stat in ("mean_s", "p95_s", "share_of_mean"):
                mean, _ = mean_ci([
                    ((c.get("phase_breakdown") or {}).get(phase) or {}).get(
                        stat) for c in cells])
                record[f"phase_{phase}_{stat}_mean"] = mean
        mean, _ = mean_ci([
            (c.get("phase_breakdown") or {}).get("capacity_deferrals")
            for c in cells])
        record["phase_capacity_deferrals_mean"] = mean

        # Preload placement, to show where the resident population landed.
        for tier in ("d", "cpu", "ssd", "hbf", "skipped"):
            mean, _ = mean_ci([
                (c.get("preloaded") or {}).get(tier) for c in cells])
            record[f"preloaded_{tier}_mean"] = mean

        # Peak HBF occupancy drives the retention refresh term.
        for key in ("hbf_reserved_bytes_peak", "peak_used_bytes"):
            mean, _ = mean_ci([
                (c.get("peak_bytes") or {}).get(key) for c in cells])
            record[f"peak_{key}_mean"] = mean

        # HBF write accounting, carried through for the endurance panel.
        writes = [c.get("hbf_write_accounting") for c in cells]
        writes = [w for w in writes if isinstance(w, dict)]
        if writes:
            for key in sorted({k for w in writes for k in w}):
                values = [w.get(key) for w in writes]
                if all(isinstance(v, (int, float)) or v is None
                       for v in values):
                    mean, _ = mean_ci(values)
                    record[f"write_{key}_mean"] = mean
        out.append(record)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=SCRIPT_ROOT / "steady_state_v1")
    args = parser.parse_args()

    rows = load_rows(args.root)
    if not rows:
        raise SystemExit(f"no result rows under {args.root}")
    agg = aggregate(rows)

    fields = sorted({key for row in agg for key in row})
    lead = ["family", "system", "rate", "n_seeds", "seeds"]
    fields = lead + [f for f in fields if f not in lead]

    out_csv = args.root / "aggregate.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in agg:
            writer.writerow(row)
    (args.root / "aggregate.json").write_text(
        json.dumps(agg, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} cells -> {len(agg)} operating points")
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Aggregate hbf_prefill campaign cells into flat CSVs.

Produces, under the campaign root:

* ``aggregate.csv``   -- one row per family x rate x system (seed-meaned
  headline metrics: throughput, TTFT, TPOT, turn latency, SLO goodput).
* ``gap_buckets.csv`` -- one row per family x rate x system x gap bucket
  (seed-pooled counts, seed-meaned mean/p50/p90/p95 for resume TTFT and
  turn latency).
* ``writes.csv``      -- media write/read rates per cell group plus the
  SSD endurance-budget multiple (fleet TBW spread over the warranty).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[1]

SSD_PROFILE = (
    REPO_ROOT / "configs/storage/micron_9550_pro_3_84tb.json")
SSD_DEVICE_COUNT = 8
SSD_WARRANTY_YEARS = 5

HEADLINE_FIELDS = (
    "target_concurrency",
    "scored_calls",
    "output_tokens_per_s",
    "first_ttft_p50_s",
    "first_ttft_p95_s",
    "resume_ttft_p50_s",
    "resume_ttft_p95_s",
    "resume_ttft_p99_s",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "turn_latency_mean_s",
    "turn_latency_p50_s",
    "turn_latency_p99_s",
    "wall_s",
)


def ssd_endurance_budget_bytes_per_s() -> float:
    profile = json.loads(SSD_PROFILE.read_text())
    rated_tbw_tb = profile["ratings"]["conservative_4k_random"][
        "rated_tbw_tb"]
    total_bytes = rated_tbw_tb * 1e12 * SSD_DEVICE_COUNT
    return total_bytes / (SSD_WARRANTY_YEARS * 365.0 * 86_400.0)


def load_cells(root: Path):
    for path in sorted(root.glob("cells/*/rate-*/seed-*.json")):
        yield json.loads(path.read_text())


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else 0.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=SCRIPT_ROOT / "hbf_prefill_v1")
    args = parser.parse_args(argv)
    root = args.root

    groups = defaultdict(list)
    for row in load_cells(root):
        groups[(row["family"], row["rate"], row["system"])].append(row)
    if not groups:
        raise SystemExit(f"no cells under {root}")

    with (root / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("family", "rate", "system", "seeds")
            + tuple(f"{name}_mean" for name in HEADLINE_FIELDS)
            + ("slo_names", "slo_good_tokens_per_s"))
        for (family, rate, system), rows in sorted(groups.items()):
            slo_names = sorted(rows[0]["slo_levels"])
            slo_good = "|".join(
                f"{mean([r['slo_levels'][name]['good_output_tokens_per_s'] for r in rows]):.1f}"
                for name in slo_names)
            writer.writerow(
                (family, rate, system, len(rows))
                + tuple(
                    round(mean([r[name] for r in rows]), 6)
                    for name in HEADLINE_FIELDS)
                + ("|".join(slo_names), slo_good))

    def write_buckets(filename: str, cell_key: str):
        with (root / filename).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow((
                "family", "rate", "system", "bucket",
                "resume_count",
                "resume_ttft_mean_s", "resume_ttft_p50_s",
                "resume_ttft_p90_s", "resume_ttft_p95_s",
                "turn_latency_mean_s", "turn_latency_p90_s",
            ))
            for (family, rate, system), rows in sorted(groups.items()):
                rows = [r for r in rows if cell_key in r]
                if not rows:
                    continue
                buckets = rows[0][cell_key]["resume_ttft_s"].keys()
                for bucket in buckets:
                    ttft = [
                        r[cell_key]["resume_ttft_s"][bucket]
                        for r in rows]
                    turns = [
                        r[cell_key]["turn_latency_s"][bucket]
                        for r in rows]
                    count = sum(t["count"] for t in ttft)
                    populated_t = [t for t in ttft if t["count"]]
                    populated_u = [t for t in turns if t["count"]]
                    writer.writerow((
                        family, rate, system, bucket, count,
                        round(mean([t["mean"] for t in populated_t]), 6),
                        round(mean([t["p50"] for t in populated_t]), 6),
                        round(mean([t["p90"] for t in populated_t]), 6),
                        round(mean([t["p95"] for t in populated_t]), 6),
                        round(mean([t["mean"] for t in populated_u]), 6),
                        round(mean([t["p90"] for t in populated_u]), 6),
                    ))

    write_buckets("gap_buckets.csv", "gap_conditioned")
    write_buckets("context_buckets.csv", "context_conditioned")

    budget = ssd_endurance_budget_bytes_per_s()
    with (root / "writes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "family", "rate", "system",
            "window_s_mean",
            "produced_kv_gb_per_s",
            "ssd_write_gb_per_s", "ssd_read_gb_per_s",
            "hbf_media_write_gb_per_s",
            "nvlink_handoff_gb_per_s",
            "ssd_endurance_budget_gb_per_s",
            "ssd_endurance_multiple",
        ))
        for (family, rate, system), rows in sorted(groups.items()):
            window = mean([r["measurement_window_s"] for r in rows])

            def rate_of(key):
                return mean([
                    r["write_accounting"].get(key, 0)
                    / max(1e-9, r["measurement_window_s"])
                    for r in rows]) / 1e9

            ssd_write = rate_of("ssd_write_bytes")
            writer.writerow((
                family, rate, system,
                round(window, 1),
                round(rate_of("produced_kv_bytes"), 4),
                round(ssd_write, 4),
                round(rate_of("ssd_read_bytes"), 4),
                round(rate_of("hbf_media_write_bytes"), 4),
                round(rate_of("nvlink_handoff_bytes"), 4),
                round(budget / 1e9, 4),
                round(ssd_write / (budget / 1e9), 4)
                if budget else 0.0,
            ))

    print(
        "wrote aggregate.csv, gap_buckets.csv, context_buckets.csv, "
        f"writes.csv under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

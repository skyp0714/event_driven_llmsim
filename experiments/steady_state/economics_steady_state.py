#!/usr/bin/env python3
"""Attach TCO, goodput-per-dollar, and HBF write endurance to the sweep.

Consumes `aggregate.csv` from analyze_steady_state.py and emits
`economics.csv` / `economics.json` next to it, one row per (family, rate).

The endurance denominator is no longer ambiguous.  The earlier closed-cohort
runs had to choose between dividing cumulative writes by the whole run --
which diluted the steady-state write rate with the arrival ramp and the drain
tail, overstating card lifetime -- and by a narrower window.  The open-system
design has no ramp and no drain: the resident population is installed at t=0
and the run is cut at the horizon, so the entire window is steady state and
the write rate it yields is the one the cards would actually see.

The Oracle still has no TCO.  Infinite HBM is not a bill of materials, so it
stays a performance-only reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get(
    "LLMSIM_REPO", ".."))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.core.hbf_design_tco import (  # noqa: E402
    CENTRAL_SENSITIVITY_POINT, lpddr_active_memory)
from serving.core.ssd_hbf_tco import evaluate_ssd_hbf_tco  # noqa: E402
from serving.core.hbf_endurance import DAYS_PER_YEAR  # noqa: E402

HBF_CARDS = 8
# tp8_context stores one physical KV copy, so the writable KV region is the
# whole per-card HBF minus the per-card model weights.
HBF_KV_REGION_BYTES_PER_CARD = 1_280_000_000_000 - 7_680_585_728
SLO_LEVEL = "tight"

WRITE_BYTE_KEYS = (
    "write_total_physical_write_bytes_mean",
    "write_total_write_bytes_mean",
    "write_hbf_write_bytes_mean",
    "write_total_bytes_mean",
    "counter_hbf_write_bytes_mean",
    "counter_hbf_bytes_written_mean",
)


def load_aggregate(root: Path) -> list[dict]:
    with (root / "aggregate.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key in {"family", "system", "seeds"}:
                continue
            try:
                row[key] = float(value) if value not in ("", None) else None
            except (TypeError, ValueError):
                row[key] = None
    return rows


def hbf_write_bytes(row: dict) -> float:
    for key in WRITE_BYTE_KEYS:
        value = row.get(key)
        if value:
            return float(value)
    return 0.0


def endurance_years(write_bytes: float, seconds: float,
                    rated_full_writes: float, waf: float) -> float | None:
    """Years until the first card hits its rated full-region write budget."""

    if write_bytes <= 0 or seconds <= 0:
        return None
    per_card_per_s = write_bytes / HBF_CARDS / seconds
    cycles_per_day = (
        per_card_per_s * 86_400.0 * waf / HBF_KV_REGION_BYTES_PER_CARD)
    if cycles_per_day <= 0:
        return None
    return rated_full_writes / cycles_per_day / DAYS_PER_YEAR


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=SCRIPT_ROOT / "steady_state_v1")
    args = parser.parse_args()

    rows = load_aggregate(args.root)
    by_point: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_point[(row["family"], row["rate"])][row["system"]] = row

    active_memory = lpddr_active_memory(
        capacity_gib_per_card=16.0, bandwidth_gbps_per_card=409.6)
    out = []
    for (family, rate) in sorted(by_point):
        point = by_point[(family, rate)]
        base = point.get("baseline_cpu_ssd")
        hbf = point.get("hbf_tp8_context")
        oracle = point.get("oracle_infinite_hbm")
        if not (base and hbf):
            continue
        base_good = base.get(f"goodput_{SLO_LEVEL}_mean") or 0.0
        hbf_good = hbf.get(f"goodput_{SLO_LEVEL}_mean") or 0.0
        if base_good <= 0:
            continue
        report = evaluate_ssd_hbf_tco(
            hbf_layout="tp8",
            active_memory=active_memory,
            baseline_slo_good_output_tokens_per_second=base_good,
            proposed_slo_good_output_tokens_per_second=hbf_good,
            oracle_slo_good_output_tokens_per_second=(
                (oracle or {}).get(f"goodput_{SLO_LEVEL}_mean") or None),
            sensitivity_point=CENTRAL_SENSITIVITY_POINT,
        )
        base_cost = report.baseline_cost
        hbf_cost = report.proposed_cost
        base_tco = (
            base_cost.capex_usd + base_cost.five_year_electricity_opex_usd)
        hbf_tco = (
            hbf_cost.capex_usd + hbf_cost.five_year_electricity_opex_usd)

        write_bytes = hbf_write_bytes(hbf)
        # No ramp, no drain: the window is the steady state by construction.
        window = hbf.get("measurement_window_s_mean") or 0.0
        record = {
            "family": family,
            "rate": rate,
            "slo_level": SLO_LEVEL,
            "target_concurrency": base.get("target_concurrency_mean"),
            "baseline_goodput_tok_s": base_good,
            "hbf_goodput_tok_s": hbf_good,
            "oracle_goodput_tok_s": (
                (oracle or {}).get(f"goodput_{SLO_LEVEL}_mean")),
            "baseline_tco_usd": base_tco,
            "hbf_tco_usd": hbf_tco,
            "baseline_capex_usd": base_cost.capex_usd,
            "hbf_capex_usd": hbf_cost.capex_usd,
            "baseline_opex_usd": base_cost.five_year_electricity_opex_usd,
            "hbf_opex_usd": hbf_cost.five_year_electricity_opex_usd,
            "baseline_it_power_w": base_cost.it_power_w,
            "hbf_it_power_w": hbf_cost.it_power_w,
            "baseline_goodput_per_musd": base_good / (base_tco / 1e6),
            "hbf_goodput_per_musd": hbf_good / (hbf_tco / 1e6),
            "hbf_over_baseline_goodput": hbf_good / base_good,
            "hbf_over_baseline_tco": hbf_tco / base_tco,
            "hbf_over_baseline_goodput_per_dollar": (
                (hbf_good / hbf_tco) / (base_good / base_tco)),
            "hbf_write_bytes": write_bytes,
            "measurement_window_s": window,
            "hbf_write_tb_per_day": (
                write_bytes / window * 86_400 / 1e12 if window else None),
        }
        for label, rated, waf in (
            ("slc_100k_waf1", 100_000.0, 1.0),
            ("slc_100k_waf2", 100_000.0, 2.0),
            ("slc_100k_waf4", 100_000.0, 4.0),
        ):
            record[f"endurance_years_{label}"] = endurance_years(
                write_bytes, window, rated, waf)
        out.append(record)

    if not out:
        print("no comparable (baseline, hbf) pairs yet")
        return 1
    fields: list[str] = []
    for record in out:
        for key in record:
            if key not in fields:
                fields.append(key)
    with (args.root / "economics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    (args.root / "economics.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(f"{'family':8} {'rate':>7} {'base tok/s':>11} {'HBF tok/s':>11} "
          f"{'HBF/base':>9} {'base TCO $M':>12} {'HBF TCO $M':>11} "
          f"{'tok/$ ratio':>12} {'HBF TB/day':>11} {'life yr WAF2':>13}")
    for r in out:
        life = r["endurance_years_slc_100k_waf2"]
        print(f"{r['family']:8} {r['rate']:7.4f} "
              f"{r['baseline_goodput_tok_s']:11.1f} "
              f"{r['hbf_goodput_tok_s']:11.1f} "
              f"{r['hbf_over_baseline_goodput']:9.3f} "
              f"{r['baseline_tco_usd']/1e6:12.3f} "
              f"{r['hbf_tco_usd']/1e6:11.3f} "
              f"{r['hbf_over_baseline_goodput_per_dollar']:12.3f} "
              f"{(r['hbf_write_tb_per_day'] or 0):11.2f} "
              f"{(life if life else float('nan')):13.1f}")
    print(f"\nwrote {args.root/'economics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

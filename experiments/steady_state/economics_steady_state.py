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

Write volume carries both terms.  The simulator observes only the KV that
turns commit, but flash retention is finite: under a retention window of `w`,
every byte still resident after `w` must be rewritten to stay readable, and
that rewrite spends a program/erase cycle exactly like a workload write.  How
much it adds is a property of the trace -- bytes leave HBF when their session
ends -- so the refresh term is the share of resident byte-time older than the
window, taken from the measured session-lifetime distribution.

The Oracle still has no TCO.  Infinite HBM is not a bill of materials, so it
stays a performance-only reference.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
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
SLO_LEVEL = "ttft5_tpot100"
SECONDS_PER_DAY = 86_400.0
# Flash retention window assumed for the refresh term.  A KV cache only has to
# outlive its session, and the measured p90 session is under seven hours, so a
# day is already generous for this workload.
RETENTION_WINDOW_S = SECONDS_PER_DAY

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


@lru_cache(maxsize=64)
def stale_byte_share(family: str, window_s: float) -> float:
    """Fraction of resident HBF bytes older than the retention window.

    In steady state a session of lifetime T contributes T seconds of byte
    residency, of which max(0, T - w) sits past the window.  The instantaneous
    share of resident bytes needing refresh is therefore
    E[max(0, T - w)] / E[T] over the session-lifetime distribution.
    """

    import run_steady_state_campaign as steady

    _, lifetimes = steady.load_pool(family)
    seconds = [value / 1e9 for value in lifetimes]
    total = sum(seconds)
    if total <= 0:
        return 0.0
    return sum(max(0.0, value - window_s) for value in seconds) / total


def refresh_bytes_per_second(
        occupied_bytes: float, family: str, window_s: float) -> float:
    """Rewrite rate needed to keep past-window bytes readable."""

    if occupied_bytes <= 0 or window_s <= 0:
        return 0.0
    return occupied_bytes * stale_byte_share(family, window_s) / window_s


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
    # One record per HBF layout: the layouts' goodput can differ by
    # double digits (codex 0.008: tp4x2 544 vs tp8 465 tok/s), so a
    # single hardcoded layout would silently mismatch any table that
    # quotes the other one's performance.
    hbf_layout_systems = (
        ("hbf_tp8_context", "tp8"),
        ("hbf_tp4x2", "tp4x2"),
    )
    for (family, rate), (hbf_system, tco_layout) in (
            (point, layout)
            for point in sorted(by_point)
            for layout in hbf_layout_systems):
        point = by_point[(family, rate)]
        base = point.get("baseline_cpu_ssd")
        hbf = point.get(hbf_system)
        oracle = point.get("oracle_infinite_hbm")
        if not (base and hbf):
            continue
        base_good = base.get(f"goodput_{SLO_LEVEL}_mean") or 0.0
        hbf_good = hbf.get(f"goodput_{SLO_LEVEL}_mean") or 0.0
        if base_good <= 0:
            continue
        report = evaluate_ssd_hbf_tco(
            hbf_layout=tco_layout,
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

        workload_write_bytes = hbf_write_bytes(hbf)
        # No ramp, no drain: the window is the steady state by construction.
        window = hbf.get("measurement_window_s_mean") or 0.0
        occupied = hbf.get("peak_hbf_reserved_bytes_peak_mean") or 0.0
        refresh_write_bytes = refresh_bytes_per_second(
            occupied, family, RETENTION_WINDOW_S) * window
        write_bytes = workload_write_bytes + refresh_write_bytes
        record = {
            "family": family,
            "rate": rate,
            "hbf_system": hbf_system,
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
            "hbf_occupied_bytes": occupied,
            "retention_window_h": RETENTION_WINDOW_S / 3600.0,
            "stale_byte_share": stale_byte_share(family, RETENTION_WINDOW_S),
            "hbf_workload_write_bytes": workload_write_bytes,
            "hbf_refresh_write_bytes": refresh_write_bytes,
            "hbf_write_bytes": write_bytes,
            "refresh_share_of_writes": (
                refresh_write_bytes / write_bytes if write_bytes else None),
            "measurement_window_s": window,
            "hbf_workload_tb_per_day": (
                workload_write_bytes / window * SECONDS_PER_DAY / 1e12
                if window else None),
            "hbf_refresh_tb_per_day": (
                refresh_write_bytes / window * SECONDS_PER_DAY / 1e12
                if window else None),
            "hbf_write_tb_per_day": (
                write_bytes / window * SECONDS_PER_DAY / 1e12
                if window else None),
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

    print(f"\nretention window = {RETENTION_WINDOW_S / 3600:.0f} h;  "
          f"write volume = workload commits + retention refresh")
    print(f"{'':32}{'--------- TB/day ---------':^37}")
    print(f"{'family':8}{'rate':>7}{'HBF/base':>9}{'tok/$':>8}"
          f"{'occ TB':>9}{'workload':>10}{'refresh':>9}{'total':>9}"
          f"{'refresh%':>10}{'yr WAF1':>10}{'yr WAF2':>10}{'yr WAF4':>10}")
    for r in out:
        nan = float("nan")
        print(f"{r['family']:8}{r['rate']:7.4f} {r['hbf_system']:16}"
              f"{r['hbf_over_baseline_goodput']:9.3f}"
              f"{r['hbf_over_baseline_goodput_per_dollar']:8.3f}"
              f"{r['hbf_occupied_bytes']/1e12:9.2f}"
              f"{(r['hbf_workload_tb_per_day'] or 0):10.2f}"
              f"{(r['hbf_refresh_tb_per_day'] or 0):9.2f}"
              f"{(r['hbf_write_tb_per_day'] or 0):9.2f}"
              f"{(r['refresh_share_of_writes'] or 0):10.1%}"
              f"{(r['endurance_years_slc_100k_waf1'] or nan):10.1f}"
              f"{(r['endurance_years_slc_100k_waf2'] or nan):10.1f}"
              f"{(r['endurance_years_slc_100k_waf4'] or nan):10.1f}")
    print(f"\nwrote {args.root/'economics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

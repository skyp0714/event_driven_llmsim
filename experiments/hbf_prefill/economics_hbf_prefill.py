#!/usr/bin/env python3
"""TCO and endurance economics for the one-server hbf_prefill campaign.

Consumes ``aggregate.csv`` and ``writes.csv`` from analyze_hbf_prefill.py
and emits ``economics.csv`` next to them, one row per (family, rate).

Both bills of materials describe the same eight-card server chassis; they
differ only in the four prefill cards.  The baseline populates all eight
positions with full H100s (logic + HBM stack).  The heterogeneous server
keeps H100s in the four decode positions and, in the four prefill
positions, replaces each HBM stack with HBF media plus a 64 GiB LPDDR
tier -- the established anchor pricing (HBF media at 0.1x the HBM-cube
capex, 3.5x its power).

SSD endurance enters the bill directly: when a system's sustained SSD
write rate exceeds what the fleet's rated TBW can absorb over the
lifetime, the excess is priced as replacement drive sets.  The oracle
remains a performance-only reference with no bill.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.core.hbf_design_tco import HardwareAnchors  # noqa: E402
from serving.core.ssd_hbf_tco import EvaluationAssumptions  # noqa: E402

BYTES_PER_GIB = 1024 ** 3
H100_CARDS_BASELINE = 8
DECODE_CARDS = 4
PREFILL_CARDS = 4
LPDDR_GIB_PER_PREFILL_CARD = 64.0
HOST_DRAM_GIB = 512e9 / BYTES_PER_GIB
SSD_DEVICES = 8
SSD_PROFILE = (
    REPO_ROOT / "configs/storage/micron_9550_pro_3_84tb.json")
SLO_LEVEL_INDEX = None  # resolved from the aggregate's slo_names column


def _capex_power(anchors: HardwareAnchors, hetero: bool):
    """Return (capex_usd, power_w) for one server chassis."""

    lines = []

    def add(count, capex_each, power_each):
        lines.append((count * capex_each, count * power_each))

    add(1, anchors.cpu_host_base_capex_usd, anchors.cpu_host_base_power_w)
    add(HOST_DRAM_GIB, anchors.host_dram_capex_usd_per_gib,
        anchors.host_dram_power_w_per_gib)
    add(H100_CARDS_BASELINE, anchors.gpu_logic_capex_usd_per_card,
        anchors.gpu_logic_power_w_per_card)
    hbm_cards = DECODE_CARDS if hetero else H100_CARDS_BASELINE
    add(hbm_cards, anchors.hbm_stack_capex_usd_per_card,
        anchors.hbm_stack_power_w_per_card)
    if hetero:
        add(PREFILL_CARDS,
            anchors.hbf_media_controller_capex_usd_per_card,
            anchors.hbf_media_controller_power_w_per_card)
        add(PREFILL_CARDS * LPDDR_GIB_PER_PREFILL_CARD,
            anchors.lpddr_capex_usd_per_gib,
            anchors.lpddr_power_w_per_gib)
    add(1, anchors.gpu_intraserver_fabric_capex_usd_per_unit,
        anchors.gpu_intraserver_fabric_power_w_per_unit)
    add(SSD_DEVICES, anchors.nvme_ssd_capex_usd_per_device,
        anchors.nvme_ssd_power_w_per_device)
    add(1, anchors.baseline_nic_capex_usd, anchors.baseline_nic_power_w)
    add(1, anchors.baseline_fabric_capex_usd,
        anchors.baseline_fabric_power_w)
    return (sum(c for c, _ in lines), sum(p for _, p in lines))


def tco_usd(
        anchors: HardwareAnchors,
        evaluation: EvaluationAssumptions,
        *, hetero: bool,
        ssd_write_gb_per_s: float,
        ssd_budget_gb_per_s: float,
        ssd_rated_total_bytes: float) -> dict:
    capex, power_w = _capex_power(anchors, hetero)
    hours = evaluation.lifetime_years * 365.0 * 24.0
    energy_kwh = (
        power_w / 1e3
        * hours
        * evaluation.average_utilization
        * evaluation.pue
    )
    energy_usd = energy_kwh * evaluation.electricity_usd_per_kwh
    lifetime_writes = (
        ssd_write_gb_per_s * 1e9 * hours * 3_600.0
        * evaluation.average_utilization)
    sets_consumed = (
        lifetime_writes / ssd_rated_total_bytes
        if ssd_rated_total_bytes else 0.0)
    replacement_sets = max(0.0, sets_consumed - 1.0)
    replacement_usd = (
        replacement_sets
        * SSD_DEVICES
        * anchors.nvme_ssd_capex_usd_per_device)
    total = capex + energy_usd + replacement_usd
    return {
        "capex_usd": capex,
        "power_w": power_w,
        "energy_usd": energy_usd,
        "ssd_sets_consumed": sets_consumed,
        "ssd_replacement_usd": replacement_usd,
        "tco_usd": total,
        "tco_usd_per_hour": total / hours,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=SCRIPT_ROOT / "hbf_prefill_v1")
    parser.add_argument(
        "--slo-level", default="ttft5_tpot100",
        help="SLO level name used for goodput-per-dollar")
    args = parser.parse_args(argv)
    root = args.root

    profile = json.loads(SSD_PROFILE.read_text())
    rated_total_bytes = (
        profile["ratings"]["conservative_4k_random"]["rated_tbw_tb"]
        * 1e12 * SSD_DEVICES)
    anchors = HardwareAnchors()
    evaluation = EvaluationAssumptions()
    budget_gb_per_s = rated_total_bytes / (
        evaluation.lifetime_years * 365.0 * 86_400.0) / 1e9

    with (root / "aggregate.csv").open(newline="") as handle:
        aggregate = list(csv.DictReader(handle))
    with (root / "writes.csv").open(newline="") as handle:
        writes = {
            (r["family"], r["rate"], r["system"]): r
            for r in csv.DictReader(handle)}

    rows = []
    for row in aggregate:
        system = row["system"]
        if system == "oracle_infinite_hbm":
            continue
        key = (row["family"], row["rate"], system)
        write_row = writes.get(key)
        ssd_write = (
            float(write_row["ssd_write_gb_per_s"]) if write_row else 0.0)
        hetero = system == "hbf_prefill_p4d4"
        economics = tco_usd(
            anchors, evaluation, hetero=hetero,
            ssd_write_gb_per_s=ssd_write,
            ssd_budget_gb_per_s=budget_gb_per_s,
            ssd_rated_total_bytes=rated_total_bytes)
        slo_names = row["slo_names"].split("|")
        slo_values = [float(v) for v in row["slo_good_tokens_per_s"].split("|")]
        try:
            good = slo_values[slo_names.index(args.slo_level)]
        except ValueError:
            raise SystemExit(
                f"unknown SLO level {args.slo_level!r}; "
                f"available {slo_names}")
        rows.append({
            "family": row["family"],
            "rate": float(row["rate"]),
            "system": system,
            "output_tokens_per_s": float(
                row["output_tokens_per_s_mean"]),
            "slo_level": args.slo_level,
            "slo_good_tokens_per_s": good,
            **{k: round(v, 4) for k, v in economics.items()},
            "ssd_write_gb_per_s": ssd_write,
            "ssd_budget_gb_per_s": round(budget_gb_per_s, 4),
            "ssd_endurance_multiple": round(
                ssd_write / budget_gb_per_s, 3) if budget_gb_per_s else 0,
            "good_tokens_per_usd_hour": round(
                good / economics["tco_usd_per_hour"], 2)
            if economics["tco_usd_per_hour"] else 0.0,
        })

    out = root / "economics.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

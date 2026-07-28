#!/usr/bin/env python3
"""Render the session-scaling campaign panels.

The load axis is cohort session count, not offered rate, so this does not
reuse `plot_fair_rate_sweep`'s rate-axis loader.  It keeps that module's
conventions: one PNG per panel plus a combined dashboard, CI95 bands, and a
fixed colour per system.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

RESULTS_ROOT = Path(os.environ.get(
    "LLMSIM_RESULTS", Path(__file__).resolve().parents[2] / "results"))
DEFAULT_ROOT = RESULTS_ROOT / "session_scaling_v2"

SYSTEM_ORDER = (
    "baseline_cpu_ssd", "oracle_infinite_hbm", "hbf_tp8_context")
STYLE = {
    "baseline_cpu_ssd": ("#B3423B", "o", "2xHBM + CPU/SSD tiering"),
    "oracle_infinite_hbm": ("#6B6B6B", "s", "Infinite-HBM Oracle"),
    "hbf_tp8_context": ("#2E6DA4", "^", "HBM + HBF (tp8_context)"),
}
HBF_LOGICAL_KV_TB = 10.179


def load(root: Path) -> list[dict]:
    path = root / "aggregate.csv"
    if not path.is_file():
        raise SystemExit(f"aggregate.csv not found: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key in {"system", "label", "seeds"}:
                continue
            try:
                row[key] = float(value) if value not in ("", None) else None
            except (TypeError, ValueError):
                row[key] = None
    return rows


def series(rows, system, metric):
    points = [
        (row["session_count"], row.get(metric),
         row.get(f"{metric[:-5]}_ci95_lower") if metric.endswith("_mean")
         else None,
         row.get(f"{metric[:-5]}_ci95_upper") if metric.endswith("_mean")
         else None)
        for row in rows if row["system"] == system
        and row.get(metric) is not None
        and not (isinstance(row[metric], float) and math.isnan(row[metric]))
    ]
    points.sort(key=lambda item: item[0])
    return points


def draw(ax: Axes, rows, metric, *, ylabel, title, logy=False,
         systems=SYSTEM_ORDER, percent=False):
    for system in systems:
        points = series(rows, system, metric)
        if not points:
            continue
        colour, marker, label = STYLE[system]
        xs = [p[0] for p in points]
        ys = [p[1] * (100 if percent else 1) for p in points]
        ax.plot(xs, ys, marker=marker, color=colour, label=label,
                linewidth=1.8, markersize=5)
        lows = [p[2] for p in points]
        highs = [p[3] for p in points]
        if all(v is not None for v in lows + highs):
            ax.fill_between(
                xs,
                [v * (100 if percent else 1) for v in lows],
                [v * (100 if percent else 1) for v in highs],
                color=colour, alpha=0.15, linewidth=0)
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Cohort size (concurrent sessions, log2)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7)


def panel_slo(ax, rows):
    draw(ax, rows, "joint_slo_pass_fraction_mean",
         ylabel="Joint SLO pass (%)",
         title="Joint SLO attainment vs cohort size", percent=True)


def panel_resume(ax, rows):
    draw(ax, rows, "resume_ttft_p95_s_mean", ylabel="Resume TTFT p95 (s)",
         title="Resume TTFT p95 (restore path)", logy=True)


def panel_first(ax, rows):
    draw(ax, rows, "first_ttft_p95_s_mean", ylabel="First TTFT p95 (s)",
         title="First-call TTFT p95 (cold prefill)", logy=True)


def panel_tpot(ax, rows):
    draw(ax, rows, "tpot_p95_ms_mean", ylabel="TPOT p95 (ms)",
         title="Decode TPOT p95")


def panel_throughput(ax, rows):
    draw(ax, rows, "output_tokens_per_s_mean",
         ylabel="Output tokens/s",
         title="Measured output throughput")


def panel_tier(ax, rows):
    """Lower-tier restore fraction: the baseline's failure mechanism."""

    for system in SYSTEM_ORDER:
        points = []
        for row in rows:
            if row["system"] != system or not row.get("scored_calls_mean"):
                continue
            lower = ((row.get("cpu_prepare_hits_mean") or 0.0)
                     + (row.get("ssd_prepare_hits_mean") or 0.0))
            points.append(
                (row["session_count"], 100.0 * lower / row["scored_calls_mean"]))
        if not points:
            continue
        points.sort()
        colour, marker, label = STYLE[system]
        ax.plot([p[0] for p in points], [p[1] for p in points],
                marker=marker, color=colour, label=label,
                linewidth=1.8, markersize=5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Cohort size (concurrent sessions, log2)")
    ax.set_ylabel("Calls served from CPU/SSD (%)")
    ax.set_title("Lower-tier restore rate (eviction pressure)", fontsize=10)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7)


def panel_fill(ax, rows):
    points = sorted(
        (row["session_count"], 100.0 * (row.get("hbf_fill_fraction") or 0.0))
        for row in rows if row["system"] == "hbf_tp8_context")
    if points:
        ax.plot([p[0] for p in points], [p[1] for p in points],
                marker="^", color=STYLE["hbf_tp8_context"][0],
                linewidth=1.8, markersize=5, label="HBF peak occupancy")
    ax.axhline(100.0, color="#B3423B", linestyle="--", linewidth=1.0,
               label="HBF capacity")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Cohort size (concurrent sessions, log2)")
    ax.set_ylabel(f"% of {HBF_LOGICAL_KV_TB:.2f} TB logical KV")
    ax.set_title("HBF capacity runway", fontsize=10)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7)


def panel_endurance(ax, rows):
    """Write rate is bracketed: the true steady rate lies between."""

    opt = sorted(
        (row["session_count"],
         row.get("hbf_write_tb_per_day_optimistic") or 0.0)
        for row in rows if row["system"] == "hbf_tp8_context")
    con = sorted(
        (row["session_count"],
         row.get("hbf_write_tb_per_day_conservative") or 0.0)
        for row in rows if row["system"] == "hbf_tp8_context")
    if opt:
        xs = [p[0] for p in opt]
        ax.fill_between(xs, [p[1] for p in opt], [p[1] for p in con],
                        color="#2E6DA4", alpha=0.20, linewidth=0,
                        label="full-horizon .. measurement-window")
        ax.plot(xs, [p[1] for p in opt], marker="o", color="#2E6DA4",
                linewidth=1.6, markersize=4,
                label="full-horizon denominator (optimistic)")
        ax.plot(xs, [p[1] for p in con], marker="v", color="#14406B",
                linewidth=1.6, markersize=4, linestyle="--",
                label="measurement-window denominator")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Cohort size (concurrent sessions, log2)")
    ax.set_ylabel("HBF media writes (TB/day)")
    ax.set_title("HBF write rate for endurance (bracketed)", fontsize=10)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=6)


PANELS = (
    ("01_joint_slo", panel_slo),
    ("02_resume_ttft_p95", panel_resume),
    ("03_first_ttft_p95", panel_first),
    ("04_tpot_p95", panel_tpot),
    ("05_output_throughput", panel_throughput),
    ("06_lower_tier_restore_rate", panel_tier),
    ("07_hbf_capacity_runway", panel_fill),
    ("08_hbf_write_endurance", panel_endurance),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    rows = load(args.root)
    out_dir = args.root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, panel in PANELS:
        figure, ax = plt.subplots(figsize=(5.2, 3.6), dpi=160)
        panel(ax, rows)
        figure.tight_layout()
        path = out_dir / f"{stem}.png"
        figure.savefig(path)
        plt.close(figure)
        written.append(path)
    figure, axes = plt.subplots(4, 2, figsize=(11, 15), dpi=140)
    for ax, (_, panel) in zip(axes.flatten(), PANELS):
        panel(ax, rows)
    figure.suptitle(
        "TraceLab agentic sessions: capacity scaling of three KV tiers",
        fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    dashboard = out_dir / "00_dashboard.png"
    figure.savefig(dashboard)
    plt.close(figure)
    written.append(dashboard)
    (out_dir / "plot_manifest.json").write_text(json.dumps(
        {"source": str(args.root / "aggregate.csv"),
         "panels": [p.name for p in written]}, indent=2) + "\n")
    for path in written:
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

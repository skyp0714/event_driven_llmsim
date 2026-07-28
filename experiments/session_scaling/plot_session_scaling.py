#!/usr/bin/env python3
"""Render the session-scaling campaign figures.

Panels: resume TTFT, TPOT, SLO goodput, goodput per dollar, five-year TCO,
and HBF write endurance.  One PNG per panel plus a combined dashboard.

The load axis is cohort session count, not offered rate: above ~0.02
sessions/s the arrival span is negligible against session lifetime, so the
rate knob is exhausted while cohort size keeps driving concurrent KV
residency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

RESULTS_ROOT = Path(os.environ.get(
    "LLMSIM_RESULTS", str(Path(__file__).resolve().parents[2] / "results")))
DEFAULT_ROOT = RESULTS_ROOT / "session_scaling_v2"

SYSTEM_ORDER = (
    "baseline_cpu_ssd", "oracle_infinite_hbm", "hbf_tp8_context")
STYLE = {
    "baseline_cpu_ssd": ("#B3423B", "o", "2xHBM + CPU/SSD tiering"),
    "oracle_infinite_hbm": ("#6B6B6B", "s", "Infinite-HBM Oracle"),
    "hbf_tp8_context": ("#2E6DA4", "^", "HBM + HBF (tp8_context)"),
}
SLO_LEVEL = "tight"
SLO_CAPTION = "SLO: first 5 s / resume 2 s / TPOT 100 ms"


def _floatify(rows):
    for row in rows:
        for key, value in list(row.items()):
            if key in {"system", "label", "seeds", "slo_level"}:
                continue
            try:
                row[key] = float(value) if value not in ("", None) else None
            except (TypeError, ValueError):
                row[key] = None
    return rows


def load(root: Path):
    with (root / "aggregate.csv").open(newline="") as handle:
        agg = _floatify(list(csv.DictReader(handle)))
    econ = []
    econ_path = root / "economics.csv"
    if econ_path.is_file():
        with econ_path.open(newline="") as handle:
            econ = _floatify(list(csv.DictReader(handle)))
    return agg, econ


def _series(rows, system, metric):
    pts = [
        (r["session_count"], r[metric]) for r in rows
        if r["system"] == system and r.get(metric) is not None
        and not (isinstance(r[metric], float) and math.isnan(r[metric]))
    ]
    pts.sort()
    return pts


def _line(ax, rows, metric, *, systems=SYSTEM_ORDER, scale=1.0):
    drew = False
    for system in systems:
        pts = _series(rows, system, metric)
        if not pts:
            continue
        colour, marker, label = STYLE[system]
        ax.plot([p[0] for p in pts], [p[1] * scale for p in pts],
                marker=marker, color=colour, label=label,
                linewidth=1.9, markersize=5)
        drew = True
    return drew


def _finish(ax, *, ylabel, title, logy=False, note=None):
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Concurrent sessions in cohort (log2)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7)
    if note:
        ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=6,
                color="#555555")


def panel_resume_ttft(ax: Axes, agg, econ):
    """p50 solid, p99 dashed -- the restore path is the whole story."""
    for system in SYSTEM_ORDER:
        colour, marker, label = STYLE[system]
        p50 = _series(agg, system, "resume_ttft_p50_s_mean")
        tail_metric = (
            "resume_ttft_p99_s_mean"
            if _series(agg, system, "resume_ttft_p99_s_mean")
            else "resume_ttft_p95_s_mean")
        tail = _series(agg, system, tail_metric)
        if p50:
            ax.plot([p[0] for p in p50], [p[1] for p in p50], marker=marker,
                    color=colour, linewidth=1.9, markersize=5,
                    label=f"{label} p50")
        if tail:
            suffix = "p99" if tail_metric.endswith("p99_s_mean") else "p95"
            ax.plot([p[0] for p in tail], [p[1] for p in tail],
                    marker=marker, color=colour, linewidth=1.3,
                    markersize=4, linestyle="--", alpha=0.8,
                    label=f"{label} {suffix}")
    _finish(ax, ylabel="Resume TTFT (s)",
            title="Resume TTFT: p50 (solid) and tail (dashed)", logy=True)


def panel_tpot(ax: Axes, agg, econ):
    for system in SYSTEM_ORDER:
        colour, marker, label = STYLE[system]
        p50 = _series(agg, system, "tpot_p50_ms_mean")
        tail_metric = (
            "tpot_p99_ms_mean" if _series(agg, system, "tpot_p99_ms_mean")
            else "tpot_p95_ms_mean")
        tail = _series(agg, system, tail_metric)
        if p50:
            ax.plot([p[0] for p in p50], [p[1] for p in p50], marker=marker,
                    color=colour, linewidth=1.9, markersize=5,
                    label=f"{label} p50")
        if tail:
            suffix = "p99" if tail_metric.endswith("p99_ms_mean") else "p95"
            ax.plot([p[0] for p in tail], [p[1] for p in tail],
                    marker=marker, color=colour, linewidth=1.3,
                    markersize=4, linestyle="--", alpha=0.8,
                    label=f"{label} {suffix}")
    _finish(ax, ylabel="TPOT (ms)",
            title="Decode TPOT: p50 (solid) and tail (dashed)", logy=True)


def panel_goodput(ax: Axes, agg, econ):
    _line(ax, agg, f"goodput_{SLO_LEVEL}_mean")
    _finish(ax, ylabel="SLO-good output tokens/s",
            title="SLO goodput", note=SLO_CAPTION)


def panel_goodput_per_dollar(ax: Axes, agg, econ):
    if not econ:
        ax.text(0.5, 0.5, "economics.csv not generated yet", ha="center",
                transform=ax.transAxes, fontsize=8)
        return
    for key, colour, marker, label in (
        ("baseline_goodput_per_musd", *STYLE["baseline_cpu_ssd"][:2],
         STYLE["baseline_cpu_ssd"][2]),
        ("hbf_goodput_per_musd", *STYLE["hbf_tp8_context"][:2],
         STYLE["hbf_tp8_context"][2]),
    ):
        pts = sorted((r["session_count"], r[key]) for r in econ
                     if r.get(key) is not None)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker,
                color=colour, linewidth=1.9, markersize=5, label=label)
    _finish(ax, ylabel="SLO-good tokens/s per $M of 5-year TCO",
            title="Goodput per dollar",
            note="Oracle excluded: infinite HBM has no bill of materials")


def panel_tco(ax: Axes, agg, econ):
    if not econ:
        ax.text(0.5, 0.5, "economics.csv not generated yet", ha="center",
                transform=ax.transAxes, fontsize=8)
        return
    counts = [r["session_count"] for r in econ]
    width = 0.35
    xs = range(len(counts))
    ax.bar([x - width / 2 for x in xs],
           [r["baseline_capex_usd"] / 1e6 for r in econ], width,
           color=STYLE["baseline_cpu_ssd"][0], label="baseline CAPEX")
    ax.bar([x - width / 2 for x in xs],
           [r["baseline_opex_usd"] / 1e6 for r in econ], width,
           bottom=[r["baseline_capex_usd"] / 1e6 for r in econ],
           color=STYLE["baseline_cpu_ssd"][0], alpha=0.45,
           label="baseline 5-yr energy")
    ax.bar([x + width / 2 for x in xs],
           [r["hbf_capex_usd"] / 1e6 for r in econ], width,
           color=STYLE["hbf_tp8_context"][0], label="HBF CAPEX")
    ax.bar([x + width / 2 for x in xs],
           [r["hbf_opex_usd"] / 1e6 for r in econ], width,
           bottom=[r["hbf_capex_usd"] / 1e6 for r in econ],
           color=STYLE["hbf_tp8_context"][0], alpha=0.45,
           label="HBF 5-yr energy")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(int(c)) for c in counts])
    ax.set_xlabel("Concurrent sessions in cohort")
    ax.set_ylabel("Five-year TCO ($M)")
    ax.set_title("Five-year TCO: CAPEX + energy", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=6)
    ax.text(0.02, 0.95,
            "HBF media = 0.10x HBM-cube CAPEX, 3.5x power",
            transform=ax.transAxes, fontsize=6, color="#555555",
            va="top")


def panel_endurance(ax: Axes, agg, econ):
    if not econ:
        ax.text(0.5, 0.5, "economics.csv not generated yet", ha="center",
                transform=ax.transAxes, fontsize=8)
        return
    opt = sorted((r["session_count"], r["hbf_write_tb_per_day_optimistic"])
                 for r in econ
                 if r.get("hbf_write_tb_per_day_optimistic") is not None)
    con = sorted((r["session_count"], r["hbf_write_tb_per_day_conservative"])
                 for r in econ
                 if r.get("hbf_write_tb_per_day_conservative") is not None)
    if opt and con:
        xs = [p[0] for p in opt]
        ax.fill_between(xs, [p[1] for p in opt], [p[1] for p in con],
                        color="#2E6DA4", alpha=0.20, linewidth=0,
                        label="denominator bracket")
        ax.plot(xs, [p[1] for p in opt], marker="o", color="#2E6DA4",
                linewidth=1.7, markersize=4,
                label="full-horizon (optimistic)")
        ax.plot(xs, [p[1] for p in con], marker="v", color="#14406B",
                linewidth=1.7, markersize=4, linestyle="--",
                label="measurement window")
    twin = ax.twinx()
    life = sorted(
        (r["session_count"], r["endurance_years_slc_100k_waf2_conservative"])
        for r in econ
        if r.get("endurance_years_slc_100k_waf2_conservative") is not None)
    if life:
        twin.plot([p[0] for p in life], [p[1] for p in life], marker="D",
                  color="#2E8B57", linewidth=1.4, markersize=4,
                  label="life, SLC 100k P/E, WAF 2")
        twin.set_yscale("log")
        twin.set_ylabel("Years to first-card EOL", color="#2E8B57",
                        fontsize=8)
        twin.tick_params(axis="y", labelcolor="#2E8B57", labelsize=7)
        twin.axhline(5.0, color="#2E8B57", linestyle=":", linewidth=1.0)
    _finish(ax, ylabel="HBF media writes (TB/day)",
            title="HBF write rate and projected endurance",
            note="dotted green = 5-year service life")


PANELS = (
    ("01_resume_ttft", panel_resume_ttft),
    ("02_tpot", panel_tpot),
    ("03_slo_goodput", panel_goodput),
    ("04_goodput_per_dollar", panel_goodput_per_dollar),
    ("05_five_year_tco", panel_tco),
    ("06_hbf_write_endurance", panel_endurance),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    agg, econ = load(args.root)
    out_dir = args.output_dir or (args.root / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for stem, panel in PANELS:
        figure, ax = plt.subplots(figsize=(5.4, 3.8), dpi=170)
        panel(ax, agg, econ)
        figure.tight_layout()
        path = out_dir / f"{stem}.png"
        figure.savefig(path)
        plt.close(figure)
        written.append(path)

    figure, axes = plt.subplots(3, 2, figsize=(12, 13), dpi=150)
    for ax, (_, panel) in zip(axes.flatten(), PANELS):
        panel(ax, agg, econ)
    figure.suptitle(
        "TraceLab agentic sessions: KV-tier scaling "
        "(2xHBM+SSD vs infinite-HBM Oracle vs HBM+HBF)", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    dashboard = out_dir / "00_dashboard.png"
    figure.savefig(dashboard)
    plt.close(figure)
    written.append(dashboard)

    (out_dir / "plot_manifest.json").write_text(json.dumps({
        "aggregate": str(args.root / "aggregate.csv"),
        "economics": str(args.root / "economics.csv"),
        "slo_level": SLO_LEVEL,
        "panels": [p.name for p in written],
    }, indent=2) + "\n")
    for path in written:
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

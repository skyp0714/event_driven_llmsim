#!/usr/bin/env python3
"""Panels for the open-system steady-state sweep, one figure per family.

The x axis is the Poisson session arrival rate, which is the load knob: by
Little's Law the resident population is lambda * W, so moving right along the
axis is adding concurrent sessions, and the point where a curve bends is the
capacity of that system rather than an artefact of how long the run was.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

SCRIPT_ROOT = Path(__file__).resolve().parent

SYSTEM_ORDER = (
    "baseline_cpu_ssd", "oracle_infinite_hbm", "hbf_tp8_context")
STYLE = {
    "baseline_cpu_ssd": ("#B3423B", "o", "2xHBM + CPU/SSD tiering"),
    "oracle_infinite_hbm": ("#6B6B6B", "s", "Infinite-HBM Oracle"),
    "hbf_tp8_context": ("#2E6DA4", "^", "HBM + HBF (tp8_context)"),
}
FAMILY_LABEL = {
    "claude": "Claude Code sessions",
    "codex": "Codex sessions",
}
SLO_LEVEL = "tight"
SLO_CAPTION = "SLO: first 5 s / resume 2 s / TPOT 100 ms"


def _floatify(rows):
    for row in rows:
        for key, value in list(row.items()):
            if key in {"family", "system", "seeds", "slo_level"}:
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
        (r["rate"], r[metric]) for r in rows
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


def _finish(ax, *, ylabel, title, logy=False, note=None, legend=True):
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Session arrival rate $\\lambda$ (sessions/s, log2)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    if legend:
        ax.legend(fontsize=7)
    if note:
        ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=6,
                color="#555555")


def panel_resume_ttft(ax: Axes, agg, econ):
    """p50 solid, p99 dashed -- the restore path is the whole story."""
    for system in SYSTEM_ORDER:
        colour, marker, label = STYLE[system]
        p50 = _series(agg, system, "resume_ttft_p50_s_mean")
        p99 = _series(agg, system, "resume_ttft_p99_s_mean")
        if p50:
            ax.plot([p[0] for p in p50], [p[1] for p in p50], marker=marker,
                    color=colour, linewidth=1.9, markersize=5,
                    label=f"{label} p50")
        if p99:
            ax.plot([p[0] for p in p99], [p[1] for p in p99], marker=marker,
                    color=colour, linewidth=1.3, markersize=4,
                    linestyle="--", alpha=0.8, label=f"{label} p99")
    _finish(ax, ylabel="Resume TTFT (s)",
            title="Resume TTFT: p50 (solid), p99 (dashed)", logy=True)


def panel_tpot(ax: Axes, agg, econ):
    for system in SYSTEM_ORDER:
        colour, marker, label = STYLE[system]
        p50 = _series(agg, system, "tpot_p50_ms_mean")
        p99 = _series(agg, system, "tpot_p99_ms_mean")
        if p50:
            ax.plot([p[0] for p in p50], [p[1] for p in p50], marker=marker,
                    color=colour, linewidth=1.9, markersize=5,
                    label=f"{label} p50")
        if p99:
            ax.plot([p[0] for p in p99], [p[1] for p in p99], marker=marker,
                    color=colour, linewidth=1.3, markersize=4,
                    linestyle="--", alpha=0.8, label=f"{label} p99")
    _finish(ax, ylabel="TPOT (ms/token)",
            title="Inter-token latency: p50 (solid), p99 (dashed)", logy=True)


def panel_goodput(ax: Axes, agg, econ):
    _line(ax, agg, f"goodput_{SLO_LEVEL}_mean")
    _finish(ax, ylabel="SLO-good output tokens/s",
            title=f"Goodput ({SLO_LEVEL} SLO)", note=SLO_CAPTION)


def panel_goodput_per_dollar(ax: Axes, econ_rows):
    if not econ_rows:
        ax.text(0.5, 0.5, "economics.csv not built yet", ha="center",
                va="center", transform=ax.transAxes, fontsize=8)
        ax.set_axis_off()
        return
    econ_rows = sorted(econ_rows, key=lambda r: r["rate"])
    rates = [r["rate"] for r in econ_rows]
    for key, system in (
        ("baseline_goodput_per_musd", "baseline_cpu_ssd"),
        ("hbf_goodput_per_musd", "hbf_tp8_context"),
    ):
        colour, marker, label = STYLE[system]
        values = [r.get(key) for r in econ_rows]
        ax.plot(rates, values, marker=marker, color=colour, linewidth=1.9,
                markersize=5, label=label)
    _finish(ax, ylabel="SLO-good tokens/s per $M of 5-yr TCO",
            title="Goodput per dollar", note=SLO_CAPTION)


def panel_tco(ax: Axes, econ_rows):
    if not econ_rows:
        ax.text(0.5, 0.5, "economics.csv not built yet", ha="center",
                va="center", transform=ax.transAxes, fontsize=8)
        ax.set_axis_off()
        return
    row = sorted(econ_rows, key=lambda r: r["rate"])[-1]
    labels = ["2xHBM +\nCPU/SSD", "HBM + HBF"]
    capex = [row["baseline_capex_usd"] / 1e6, row["hbf_capex_usd"] / 1e6]
    opex = [row["baseline_opex_usd"] / 1e6, row["hbf_opex_usd"] / 1e6]
    x = range(len(labels))
    ax.bar(x, capex, 0.55, label="Capex", color="#2E6DA4")
    ax.bar(x, opex, 0.55, bottom=capex, label="5-yr electricity",
           color="#8FB8DE")
    for index, (c, o) in enumerate(zip(capex, opex)):
        ax.text(index, c + o, f"${c + o:,.2f}M", ha="center", va="bottom",
                fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("5-year TCO ($M)")
    ax.set_title("Cost of ownership", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7)
    ax.text(0.02, 0.92,
            "HBF cube: 0.10x HBM cube capex, 3.50x power",
            transform=ax.transAxes, fontsize=6, color="#555555")


def panel_endurance(ax: Axes, econ_rows):
    if not econ_rows:
        ax.text(0.5, 0.5, "economics.csv not built yet", ha="center",
                va="center", transform=ax.transAxes, fontsize=8)
        ax.set_axis_off()
        return
    econ_rows = sorted(econ_rows, key=lambda r: r["rate"])
    rates = [r["rate"] for r in econ_rows]
    for label, key, style in (
        ("WAF 1.0", "endurance_years_slc_100k_waf1", "-"),
        ("WAF 2.0", "endurance_years_slc_100k_waf2", "--"),
        ("WAF 4.0", "endurance_years_slc_100k_waf4", ":"),
    ):
        values = [r.get(key) for r in econ_rows]
        if not any(v for v in values):
            continue
        ax.plot(rates, values, style, marker="^", color="#2E6DA4",
                linewidth=1.7, markersize=4, label=label)
    ax.axhline(5.0, color="#B3423B", linewidth=1.2, linestyle="-.",
               label="5-year deployment")
    _finish(ax, ylabel="Years to rated write budget",
            title="HBF write endurance (SLC 100k P/E)", logy=True,
            note="Steady-state window: no ramp or drain to dilute the rate")


def panel_phases(ax: Axes, agg):
    """Where resume TTFT is actually spent, baseline vs HBF."""
    phases = (
        ("phase_admission_wait_mean_s_mean", "Admission wait", "#B3423B"),
        ("phase_restore_transfer_mean_s_mean", "KV restore", "#E0A458"),
        ("phase_prefill_queue_and_compute_mean_s_mean",
         "Prefill queue + compute", "#2E6DA4"),
    )
    drew = False
    for system, hatch in (("baseline_cpu_ssd", None),
                          ("hbf_tp8_context", "//")):
        rates = sorted({
            r["rate"] for r in agg if r["system"] == system})
        if not rates:
            continue
        bottom = [0.0] * len(rates)
        for key, label, colour in phases:
            values = []
            for rate in rates:
                match = [r for r in agg
                         if r["system"] == system and r["rate"] == rate]
                values.append((match[0].get(key) or 0.0) if match else 0.0)
            positions = [
                index + (0.0 if hatch is None else 0.38)
                for index in range(len(rates))]
            ax.bar(positions, values, 0.36, bottom=bottom, color=colour,
                   hatch=hatch, edgecolor="white", linewidth=0.4,
                   label=(f"{label}" if system == "baseline_cpu_ssd"
                          else None))
            bottom = [b + v for b, v in zip(bottom, values)]
            drew = True
        ax.set_xticks([index + 0.19 for index in range(len(rates))])
        ax.set_xticklabels([f"{r:g}" for r in rates], fontsize=7)
    if not drew:
        ax.set_axis_off()
        return
    ax.set_xlabel("Session arrival rate $\\lambda$ (sessions/s)")
    ax.set_ylabel("Mean resume TTFT (s)")
    ax.set_title("Resume TTFT decomposition\n"
                 "(left bar: tiering baseline, hatched: HBF)", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7)


def build_family_figure(family: str, agg, econ, out_dir: Path):
    rows = [r for r in agg if r["family"] == family]
    econ_rows = [r for r in econ if r["family"] == family]
    if not rows:
        return []

    written = []
    singles = (
        ("resume_ttft", panel_resume_ttft, (rows, econ_rows)),
        ("tpot", panel_tpot, (rows, econ_rows)),
        ("goodput", panel_goodput, (rows, econ_rows)),
    )
    for name, fn, payload in singles:
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        fn(ax, *payload)
        fig.tight_layout()
        path = out_dir / f"{family}_{name}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    for name, fn in (
        ("goodput_per_dollar", panel_goodput_per_dollar),
        ("tco", panel_tco),
        ("write_endurance", panel_endurance),
    ):
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        fn(ax, econ_rows)
        fig.tight_layout()
        path = out_dir / f"{family}_{name}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    panel_phases(ax, rows)
    fig.tight_layout()
    path = out_dir / f"{family}_phase_decomposition.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(2, 4, figsize=(22.0, 9.0))
    panel_resume_ttft(axes[0][0], rows, econ_rows)
    panel_tpot(axes[0][1], rows, econ_rows)
    panel_goodput(axes[0][2], rows, econ_rows)
    panel_goodput_per_dollar(axes[0][3], econ_rows)
    panel_tco(axes[1][0], econ_rows)
    panel_endurance(axes[1][1], econ_rows)
    panel_phases(axes[1][2], rows)
    axes[1][3].set_axis_off()
    fig.suptitle(
        f"Open-system steady state - {FAMILY_LABEL.get(family, family)}",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / f"{family}_dashboard.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=SCRIPT_ROOT / "steady_state_v1")
    parser.add_argument(
        "--out", type=Path, default=Path(os.environ.get(
            "LLMSIM_REPO", ".."))
        / "figures" / "steady_state_v1")
    args = parser.parse_args()

    agg, econ = load(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    families = sorted({r["family"] for r in agg})
    written = []
    for family in families:
        written.extend(build_family_figure(family, agg, econ, args.out))
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

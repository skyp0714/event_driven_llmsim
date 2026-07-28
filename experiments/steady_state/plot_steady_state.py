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
    "baseline_cpu_ssd", "oracle_infinite_hbm", "hbf_tp8_context",
    "hbf_tp4x2")
STYLE = {
    "baseline_cpu_ssd": ("#B3423B", "o", "2xHBM + CPU/SSD tiering"),
    "oracle_infinite_hbm": ("#6B6B6B", "s", "Infinite-HBM Oracle"),
    "hbf_tp8_context": ("#2E6DA4", "^", "HBM + HBF (tp8_context)"),
    "hbf_tp4x2": ("#3F8F5F", "v", "HBM + HBF (tp4x2)"),
}
FAMILY_LABEL = {
    "claude": "Claude Code sessions",
    "codex": "Codex sessions",
}
SLO_LEVEL = "ttft5_tpot100"
SLO_CAPTION = "SLO: TTFT 5 s / TPOT 100 ms (of 3x3 grid)"


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
                    label=label)
        if p99:
            ax.plot([p[0] for p in p99], [p[1] for p in p99], marker=marker,
                    color=colour, linewidth=1.3, markersize=4,
                    linestyle="--", alpha=0.8)
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
                    label=label)
        if p99:
            ax.plot([p[0] for p in p99], [p[1] for p in p99], marker=marker,
                    color=colour, linewidth=1.3, markersize=4,
                    linestyle="--", alpha=0.8)
    _finish(ax, ylabel="TPOT (ms/token)",
            title="Inter-token latency: p50 (solid), p99 (dashed)", logy=True)


def panel_turn_latency(ax: Axes, agg, econ):
    """Release-to-completion per call: mean solid, p99 dashed.

    The mean is the discriminating view: the median turn is short and
    near-identical across systems, while the mean carries the queueing
    and restore penalties that differ between them.
    """
    for system in SYSTEM_ORDER:
        colour, marker, label = STYLE[system]
        mean = _series(agg, system, "turn_latency_mean_s_mean")
        p99 = _series(agg, system, "turn_latency_p99_s_mean")
        if mean:
            ax.plot([p[0] for p in mean], [p[1] for p in mean],
                    marker=marker, color=colour, linewidth=1.9,
                    markersize=5, label=label)
        if p99:
            ax.plot([p[0] for p in p99], [p[1] for p in p99], marker=marker,
                    color=colour, linewidth=1.3, markersize=4,
                    linestyle="--", alpha=0.8)
    _finish(ax, ylabel="Turn latency (s)",
            title="Turn latency: mean (solid), p99 (dashed)", logy=True)


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
    by_system = defaultdict(list)
    for row in econ_rows:
        by_system[row.get("hbf_system", "hbf_tp8_context")].append(row)
    baseline_rows = sorted(
        next(iter(by_system.values())), key=lambda r: r["rate"])
    colour, marker, label = STYLE["baseline_cpu_ssd"]
    ax.plot([r["rate"] for r in baseline_rows],
            [r.get("baseline_goodput_per_musd") for r in baseline_rows],
            marker=marker, color=colour, linewidth=1.9, markersize=5,
            label=label)
    for system in ("hbf_tp8_context", "hbf_tp4x2"):
        rows_l = sorted(
            by_system.get(system, ()), key=lambda r: r["rate"])
        if not rows_l:
            continue
        colour, marker, label = STYLE[system]
        ax.plot([r["rate"] for r in rows_l],
                [r.get("hbf_goodput_per_musd") for r in rows_l],
                marker=marker, color=colour, linewidth=1.9,
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


def _saturation_cutoff(rows) -> float:
    """Return the largest rate still on the rising side of goodput.

    Past saturation every system sheds goodput and the figures only show
    collapse noise, so the plotted range ends at the last rate where the
    best system's goodput is still increasing.
    """

    best_by_rate = {}
    for row in rows:
        value = row.get(f"goodput_{SLO_LEVEL}_mean")
        if value is None:
            continue
        rate = row["rate"]
        best_by_rate[rate] = max(best_by_rate.get(rate, 0.0), value)
    rates = sorted(best_by_rate)
    cutoff = rates[-1] if rates else float("inf")
    for previous, current in zip(rates, rates[1:]):
        if best_by_rate[current] < best_by_rate[previous]:
            cutoff = previous
            break
    return cutoff


def build_family_figure(
        family: str, agg, econ, out_dir: Path, *,
        include_saturated: bool = False):
    rows = [r for r in agg if r["family"] == family]
    econ_rows = [r for r in econ if r["family"] == family]
    # The TCO and endurance panels describe one HBF build; the tp8 rows
    # are the canonical single-copy layout.  Goodput-per-dollar keeps
    # every layout's rows and draws one line per layout.
    econ_tp8 = [
        r for r in econ_rows
        if r.get("hbf_system", "hbf_tp8_context") == "hbf_tp8_context"
    ]
    if not rows:
        return []
    if not include_saturated:
        cutoff = _saturation_cutoff(rows)
        dropped = sorted({
            r["rate"] for r in rows if r["rate"] > cutoff})
        if dropped:
            print(f"{family}: dropping saturated rates "
                  f"{', '.join(f'{r:g}' for r in dropped)} "
                  f"(goodput peaks at {cutoff:g})")
        rows = [r for r in rows if r["rate"] <= cutoff]
        econ_rows = [r for r in econ_rows if r["rate"] <= cutoff]
        econ_tp8 = [r for r in econ_tp8 if r["rate"] <= cutoff]

    written = []
    singles = (
        ("resume_ttft", panel_resume_ttft, (rows, econ_rows)),
        ("tpot", panel_tpot, (rows, econ_rows)),
        ("turn_latency", panel_turn_latency, (rows, econ_rows)),
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

    for name, fn, econ_payload in (
        ("goodput_per_dollar", panel_goodput_per_dollar, econ_rows),
        ("tco", panel_tco, econ_tp8),
        ("write_endurance", panel_endurance, econ_tp8),
    ):
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        fn(ax, econ_payload)
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
    panel_tco(axes[1][0], econ_tp8)
    panel_endurance(axes[1][1], econ_tp8)
    panel_phases(axes[1][2], rows)
    panel_turn_latency(axes[1][3], rows, econ_rows)
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
    parser.add_argument(
        "--include-saturated", action="store_true",
        help="keep operating points past the goodput peak")
    args = parser.parse_args()

    agg, econ = load(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    families = sorted({r["family"] for r in agg})
    written = []
    for family in families:
        written.extend(build_family_figure(
            family, agg, econ, args.out,
            include_saturated=args.include_saturated))
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Four-panel head-to-head of the three systems, one figure per family.

Resume TTFT, TPOT, turn latency and SLO goodput on one sheet, because the
three systems fail in different places and no single panel ranks them: the
tiering baseline holds TPOT flat by refusing admission and pays in queueing,
HBF pays for its residency in per-token latency, and only goodput -- which
scores TTFT and TPOT together -- puts them on one axis.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


def load(root: Path):
    rows = []
    with (root / "aggregate.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            out = {}
            for key, value in row.items():
                if key in {"family", "system", "seeds"}:
                    out[key] = value
                    continue
                try:
                    out[key] = float(value) if value not in ("", None) else None
                except (TypeError, ValueError):
                    out[key] = None
            rows.append(out)
    return rows


def series(rows, system, metric):
    pts = [
        (r["rate"], r[metric]) for r in rows
        if r["system"] == system and r.get(metric) is not None
        and not (isinstance(r[metric], float) and math.isnan(r[metric]))
    ]
    pts.sort()
    return pts


def concurrency_ticks(ax, rows):
    """Label the load axis with what it actually means: resident sessions."""
    pairs = sorted({
        (r["rate"], r.get("target_concurrency_mean"))
        for r in rows if r.get("target_concurrency_mean")})
    if not pairs:
        return
    top = ax.secondary_xaxis("top")
    top.set_xticks([rate for rate, _ in pairs])
    top.set_xticklabels([f"{int(L)}" for _, L in pairs], fontsize=7)
    top.set_xlabel("Resident sessions $L = \\lambda W$", fontsize=8)


def finish(ax, rows, *, ylabel, title, logy=True, note=None):
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Session arrival rate $\\lambda$ (sessions/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=6.5)
    concurrency_ticks(ax, rows)
    if note:
        ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=6,
                color="#555555")


def pair_panel(ax, rows, p50_metric, tail_metric, *, ylabel, title, note=None):
    for system in SYSTEM_ORDER:
        colour, marker, label = STYLE[system]
        p50 = series(rows, system, p50_metric)
        tail = series(rows, system, tail_metric)
        if p50:
            ax.plot([p[0] for p in p50], [p[1] for p in p50], marker=marker,
                    color=colour, linewidth=1.9, markersize=5,
                    label=f"{label} p50")
        if tail:
            ax.plot([p[0] for p in tail], [p[1] for p in tail], marker=marker,
                    color=colour, linewidth=1.2, markersize=4,
                    linestyle="--", alpha=0.75, label=f"{label} p99")
    finish(ax, rows, ylabel=ylabel, title=title, note=note)


def panel_turn(ax, rows):
    for system in SYSTEM_ORDER:
        colour, marker, label = STYLE[system]
        pts = series(rows, system, "turn_latency_mean_s_mean")
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker,
                    color=colour, linewidth=1.9, markersize=5,
                    label=f"{label} mean")
        tail = series(rows, system, "turn_latency_p99_s_mean")
        if tail:
            ax.plot([p[0] for p in tail], [p[1] for p in tail], marker=marker,
                    color=colour, linewidth=1.2, markersize=4,
                    linestyle="--", alpha=0.75, label=f"{label} p99")
    finish(ax, rows, ylabel="Turn latency (s)",
           title="End-to-end turn: mean (solid), p99 (dashed)",
           note="Output length makes TPOT, not TTFT, dominate this")


def panel_goodput(ax, rows):
    for system in SYSTEM_ORDER:
        colour, marker, label = STYLE[system]
        pts = series(rows, system, f"goodput_{SLO_LEVEL}_mean")
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker,
                color=colour, linewidth=1.9, markersize=5, label=label)
    finish(ax, rows, ylabel="SLO-good output tokens/s", logy=False,
           title=f"Goodput ({SLO_LEVEL} SLO) -- the only axis that ranks them",
           note=SLO_CAPTION)


def build(family: str, rows, out_dir: Path):
    sub = [r for r in rows if r["family"] == family]
    if not sub:
        return []
    written = []

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.2))
    pair_panel(axes[0][0], sub,
               "resume_ttft_p50_s_mean", "resume_ttft_p99_s_mean",
               ylabel="Resume TTFT (s)",
               title="Resume TTFT: p50 (solid), p99 (dashed)",
               note="Baseline pays its capacity limit here")
    pair_panel(axes[0][1], sub, "tpot_p50_ms_mean", "tpot_p99_ms_mean",
               ylabel="TPOT (ms/token)",
               title="Inter-token latency: p50 (solid), p99 (dashed)",
               note="Flat baseline = admission throttling, not speed")
    panel_turn(axes[1][0], sub)
    panel_goodput(axes[1][1], sub)
    fig.suptitle(
        f"Three-system comparison at steady state - "
        f"{FAMILY_LABEL.get(family, family)}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    path = out_dir / f"{family}_system_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    for name, fn in (
        ("resume_ttft", lambda ax: pair_panel(
            ax, sub, "resume_ttft_p50_s_mean", "resume_ttft_p99_s_mean",
            ylabel="Resume TTFT (s)",
            title="Resume TTFT: p50 (solid), p99 (dashed)")),
        ("tpot", lambda ax: pair_panel(
            ax, sub, "tpot_p50_ms_mean", "tpot_p99_ms_mean",
            ylabel="TPOT (ms/token)",
            title="Inter-token latency: p50 (solid), p99 (dashed)")),
        ("turn_latency", lambda ax: panel_turn(ax, sub)),
        ("goodput", lambda ax: panel_goodput(ax, sub)),
    ):
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        fn(ax)
        fig.tight_layout()
        path = out_dir / f"{family}_{name}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=SCRIPT_ROOT / "steady_state_v1")
    parser.add_argument(
        "--out", type=Path,
        default=Path("..")
        / "figures" / "steady_state_v1")
    args = parser.parse_args()

    rows = load(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for family in sorted({r["family"] for r in rows}):
        written.extend(build(family, rows, args.out))
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

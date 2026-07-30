#!/usr/bin/env python3
"""Campaign figures for the one-server HBF-prefill comparison.

Reads aggregate.csv / gap_buckets.csv / writes.csv / economics.csv from
the campaign root and writes four figures per family under
``figures/<root name>/``:

* ``<family>_throughput_turn.png``   -- output tokens/s and mean turn
  latency versus offered session rate (one axis per panel).
* ``<family>_resume_ttft_by_gap.png`` -- resume TTFT by idle-gap bucket,
  grouped bars at a pre-knee and a saturated rate.
* ``<family>_tco_goodput.png``       -- TCO composition and SLO goodput
  per TCO dollar-hour.
* ``<family>_write_endurance.png``   -- sustained media write rates
  against the five-year SSD TBW budget.

Colors follow the repo-neutral validated palette: baseline blue,
HBF-prefill orange, oracle aqua (aqua carries direct labels for relief).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[1]

SYSTEMS = ("baseline_cpu_ssd", "hbf_prefill_p4d4", "oracle_infinite_hbm")
LABELS = {
    "baseline_cpu_ssd": "H100 + CPU/SSD tiering",
    "hbf_prefill_p4d4": "HBF-prefill P4D4 (ours)",
    "oracle_infinite_hbm": "Infinite-HBM oracle",
}
COLORS = {
    "baseline_cpu_ssd": "#2a78d6",
    "hbf_prefill_p4d4": "#eb6834",
    "oracle_infinite_hbm": "#1baf7a",
}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e7e6e3"

BUCKET_ORDER = (
    "lt_1m", "1m_5m", "5m_30m", "30m_1h", "1h_4h", "4h_12h", "gt_12h")
BUCKET_LABEL = {
    "lt_1m": "<1m", "1m_5m": "1–5m", "5m_30m": "5–30m",
    "30m_1h": "30m–1h", "1h_4h": "1–4h", "4h_12h": "4–12h",
    "gt_12h": ">12h",
}
CONTEXT_ORDER = (
    "lt_16k", "16k_64k", "64k_128k", "128k_256k",
    "256k_512k", "gt_512k")
CONTEXT_LABEL = {
    "lt_16k": "<16k", "16k_64k": "16–64k", "64k_128k": "64–128k",
    "128k_256k": "128–256k", "256k_512k": "256–512k",
    "gt_512k": ">512k",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def rate_axis(ax, rates):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(sorted(rates)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels(
        [f"{r:g}" for r in sorted(rates)], rotation=45, ha="right")
    ax.set_xlabel("offered session rate (sessions/s)", color=TEXT)


def by_system(rows, family, value_key):
    out = defaultdict(dict)
    for row in rows:
        if row["family"] != family:
            continue
        out[row["system"]][float(row["rate"])] = float(row[value_key])
    return out


def direct_label(ax, xs, ys, system, dy=0):
    ax.annotate(
        LABELS[system], (xs[-1], ys[-1]),
        xytext=(6, dy), textcoords="offset points",
        fontsize=8.5, color=COLORS[system], fontweight="bold",
        va="center")


def plot_throughput_turn(aggregate, family, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    specs = (
        (axes[0], "output_tokens_per_s_mean",
         "output tokens/s", "Raw generation throughput"),
        (axes[1], "turn_latency_mean_s_mean",
         "mean turn latency (s)", "Turn latency"),
    )
    rates = sorted({
        float(r["rate"]) for r in aggregate if r["family"] == family})
    for ax, key, ylabel, title in specs:
        series = by_system(aggregate, family, key)
        for offset, system in enumerate(SYSTEMS):
            points = sorted(series.get(system, {}).items())
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(
                xs, ys, color=COLORS[system], linewidth=2,
                marker="o", markersize=4.5,
                label=LABELS[system])
            direct_label(ax, xs, ys, system, dy=(offset - 1) * 11)
        style_axis(ax)
        rate_axis(ax, rates)
        ax.set_ylabel(ylabel, color=TEXT)
        ax.set_title(title, color=TEXT, fontsize=11, loc="left")
        if key == "turn_latency_mean_s_mean":
            ax.set_yscale("log")
    axes[0].legend(
        frameon=False, fontsize=8.5, loc="upper left",
        labelcolor=TEXT_2)
    fig.suptitle(
        f"{family}: one eight-card server, rate sweep",
        color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.99, 0.94))
    out = out_dir / f"{family}_throughput_turn.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_bucket_axis(
        rows, family, rates_shown, out_dir, *,
        order, labels, xlabel, stem, title):
    """Grouped mean bars with a p90 tick per bar, one panel per rate."""

    fig, axes = plt.subplots(
        1, len(rates_shown), figsize=(5.9 * len(rates_shown), 4.2),
        dpi=150, sharey=True)
    if len(rates_shown) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)
    width = 0.27
    for ax, rate in zip(axes, rates_shown):
        for index, system in enumerate(SYSTEMS):
            values, p90s = [], []
            for bucket in order:
                row = next(
                    (r for r in rows
                     if r["family"] == family
                     and float(r["rate"]) == rate
                     and r["system"] == system
                     and r["bucket"] == bucket), None)
                values.append(
                    float(row["resume_ttft_mean_s"]) if row else 0.0)
                p90s.append(
                    float(row["resume_ttft_p90_s"]) if row else 0.0)
            positions = [
                i + (index - 1) * width for i in range(len(order))]
            ax.bar(
                positions, values, width=width,
                color=COLORS[system], label=LABELS[system],
                edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.scatter(
                [p for p, v in zip(positions, p90s) if v > 0],
                [v for v in p90s if v > 0],
                marker="_", s=90, color=COLORS[system],
                linewidth=1.6, zorder=4)
            if system == "oracle_infinite_hbm":
                for pos, val in zip(positions, values):
                    if val > 0:
                        ax.annotate(
                            f"{val:.2g}", (pos, val),
                            xytext=(0, 3), textcoords="offset points",
                            fontsize=6.5, color=TEXT_2, ha="center")
        style_axis(ax)
        ax.set_yscale("log")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(
            [labels[b] for b in order], fontsize=8.5)
        ax.set_xlabel(xlabel, color=TEXT)
        counts_row = [
            sum(
                int(r["resume_count"])
                for r in rows
                if r["family"] == family
                and float(r["rate"]) == rate
                and r["bucket"] == bucket
                and r["system"] == "baseline_cpu_ssd")
            for bucket in order]
        ax.set_title(
            f"rate {rate:g}/s   (events/bucket: "
            + ", ".join(str(c) for c in counts_row) + ")",
            color=TEXT, fontsize=9.5, loc="left")
    axes[0].set_ylabel(
        "resume TTFT (s)  — bar: mean, tick: p90", color=TEXT)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=TEXT_2)
    fig.suptitle(
        f"{family}: {title}", color=TEXT, fontsize=12, x=0.01,
        ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = out_dir / f"{family}_{stem}.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_tco_goodput(economics, family, out_dir):
    rows = [r for r in economics if r["family"] == family]
    rates = sorted({float(r["rate"]) for r in rows})
    fig, axes = plt.subplots(
        1, 2, figsize=(10.6, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    width = 0.38
    shown = [rate for rate in rates]
    for index, system in enumerate(
            ("baseline_cpu_ssd", "hbf_prefill_p4d4")):
        capex, energy, replacement = [], [], []
        for rate in shown:
            row = next(
                (r for r in rows
                 if float(r["rate"]) == rate and r["system"] == system),
                None)
            capex.append(float(row["capex_usd"]) / 1e3 if row else 0)
            energy.append(float(row["energy_usd"]) / 1e3 if row else 0)
            replacement.append(
                float(row["ssd_replacement_usd"]) / 1e3 if row else 0)
        positions = [
            i + (index - 0.5) * width for i in range(len(shown))]
        base = COLORS[system]
        ax.bar(positions, capex, width=width, color=base,
               edgecolor=SURFACE, linewidth=1.5,
               label=f"{LABELS[system]} capex", zorder=3)
        ax.bar(positions, energy, width=width, bottom=capex,
               color=base, alpha=0.45, edgecolor=SURFACE,
               linewidth=1.5, label=f"{LABELS[system]} energy", zorder=3)
        tops = [c + e for c, e in zip(capex, energy)]
        ax.bar(positions, replacement, width=width, bottom=tops,
               color="#52514e", edgecolor=SURFACE, linewidth=1.5,
               hatch="//",
               label=(
                   "SSD replacement (endurance)"
                   if index == 0 else None),
               zorder=3)
    style_axis(ax)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([f"{r:g}" for r in shown], rotation=45,
                       ha="right", fontsize=8.5)
    ax.set_xlabel("offered session rate (sessions/s)", color=TEXT)
    ax.set_ylabel("5-year TCO (k$)", color=TEXT)
    ax.set_title("TCO composition", color=TEXT, fontsize=11, loc="left")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=7.5,
              labelcolor=TEXT_2, ncol=1, loc="upper left")

    ax = axes[1]
    for system in ("baseline_cpu_ssd", "hbf_prefill_p4d4"):
        points = sorted(
            (float(r["rate"]), float(r["good_tokens_per_usd_hour"]))
            for r in rows if r["system"] == system)
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=COLORS[system], linewidth=2, marker="o",
                markersize=4.5, label=LABELS[system])
        direct_label(ax, xs, ys, system)
    style_axis(ax)
    rate_axis(ax, rates)
    ax.set_ylabel(
        "SLO-good tokens/s per TCO $/hour", color=TEXT)
    ax.set_title(
        f"Goodput per dollar ({rows[0]['slo_level']})",
        color=TEXT, fontsize=11, loc="left")
    fig.suptitle(
        f"{family}: economics", color=TEXT, fontsize=12, x=0.01,
        ha="left")
    fig.tight_layout(rect=(0, 0, 0.99, 0.93))
    out = out_dir / f"{family}_tco_goodput.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_write_endurance(writes, family, out_dir):
    rows = [r for r in writes if r["family"] == family]
    rates = sorted({float(r["rate"]) for r in rows})
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    budget = float(rows[0]["ssd_endurance_budget_gb_per_s"])

    series = (
        ("baseline_cpu_ssd", "ssd_write_gb_per_s", "-",
         "baseline SSD writes"),
        ("hbf_prefill_p4d4", "ssd_write_gb_per_s", "-",
         "ours: SSD writes (≈0)"),
        ("hbf_prefill_p4d4", "hbf_media_write_gb_per_s", "--",
         "ours: HBF media writes"),
    )
    floor = budget / 300.0
    for system, key, style, label in series:
        points = sorted(
            (float(r["rate"]), max(floor, float(r[key])))
            for r in rows if r["system"] == system)
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=COLORS[system], linewidth=2,
                linestyle=style, marker="o", markersize=4,
                label=label)
        ax.annotate(
            label, (xs[-1], ys[-1]), xytext=(6, 0),
            textcoords="offset points", fontsize=8,
            color=COLORS[system], fontweight="bold", va="center")
    ax.axhline(budget, color=TEXT_2, linewidth=1.4, linestyle=":")
    ax.annotate(
        f"5-year SSD TBW budget ({budget:.2f} GB/s)",
        (min(rates), budget), xytext=(0, 5),
        textcoords="offset points", fontsize=8.5, color=TEXT_2)
    style_axis(ax)
    rate_axis(ax, rates)
    ax.set_yscale("log")
    ax.set_ylabel("sustained media writes (GB/s)", color=TEXT)
    ax.set_title(
        f"{family}: write endurance", color=TEXT, fontsize=12,
        loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2,
              loc="upper left")
    fig.tight_layout()
    out = out_dir / f"{family}_write_endurance.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def pick_gap_rates(gap_rows, family) -> list[float]:
    """One pre-knee rate and one saturated rate with populated buckets."""

    rates = sorted({
        float(r["rate"]) for r in gap_rows if r["family"] == family})
    if not rates:
        return []

    def long_gap_events(rate):
        return sum(
            int(r["resume_count"]) for r in gap_rows
            if r["family"] == family and float(r["rate"]) == rate
            and r["system"] == "baseline_cpu_ssd"
            and r["bucket"] in ("30m_1h", "1h_4h", "4h_12h", "gt_12h"))

    populated = [r for r in rates if long_gap_events(r) >= 10]
    if not populated:
        populated = rates
    low = populated[0]
    high = populated[-1]
    return [low] if high == low else [low, high]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=SCRIPT_ROOT / "hbf_prefill_v1")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root
    out_dir = args.out_dir or (REPO_ROOT / "figures" / root.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate = read_csv(root / "aggregate.csv")
    gap_rows = read_csv(root / "gap_buckets.csv")
    context_path = root / "context_buckets.csv"
    context_rows = (
        read_csv(context_path) if context_path.is_file() else [])
    writes = read_csv(root / "writes.csv")
    economics_path = root / "economics.csv"
    economics = (
        read_csv(economics_path) if economics_path.is_file() else [])

    families = sorted({r["family"] for r in aggregate})
    written = []
    for family in families:
        written.append(plot_throughput_turn(aggregate, family, out_dir))
        rates_shown = pick_gap_rates(gap_rows, family)
        if rates_shown:
            written.append(plot_bucket_axis(
                gap_rows, family, rates_shown, out_dir,
                order=BUCKET_ORDER, labels=BUCKET_LABEL,
                xlabel="idle gap before resume",
                stem="resume_ttft_by_gap",
                title="resume TTFT conditioned on idle gap"))
        if context_rows and rates_shown:
            written.append(plot_bucket_axis(
                context_rows, family, rates_shown, out_dir,
                order=CONTEXT_ORDER, labels=CONTEXT_LABEL,
                xlabel="reused context at resume (tokens)",
                stem="resume_ttft_by_context",
                title="resume TTFT conditioned on context size"))
        if economics:
            written.append(plot_tco_goodput(economics, family, out_dir))
        written.append(plot_write_endurance(writes, family, out_dir))
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

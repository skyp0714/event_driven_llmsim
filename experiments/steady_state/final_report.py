"""Final report: CSVs and figures for the v8/v9 server-count study.

claude is the 2-server comparison (2xGPU baseline vs GPU+HBF hybrid).
codex is the 3-server comparison: the 1:2 hybrid (one GPU host plus two
HBF hosts, simulated as one fused host via LLMSIM_HBF_HW_SCALE=2) runs
at the nominal rate, while the 3xGPU baseline and oracle were simulated
as the 2-node systems at 2/3 of the nominal rate — share-nothing fleet
scaling — so their throughput-like metrics are multiplied by 1.5 here
and their cells are re-keyed to the nominal rate.

v8 uses the uniform-template resident population, v9 the
lifetime-weighted (renewal steady state) population.

Per version root:
  steady_state_v*/final_results.csv     per family/rate/system metrics
  steady_state_v*/final_attainment.csv  90%-attainment max rates
  steady_state_v*/final_jct.csv         matched-session JCT by rate
  ../../figures/steady_state_v*/*.png

JCT inputs are `jct/jr_{cl|cx}_{base|hyb}_{nominal_rate}.json` dumps
from the young_sessions instrument harness (codex base files were
simulated at 2/3 of the nominal rate encoded in the name).
"""

from collections import defaultdict
from pathlib import Path
import csv
import json
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_ROOT = Path(__file__).resolve().parent
VERSIONS = ("steady_state_v8", "steady_state_v9")
POPULATION = {
    "steady_state_v8": "uniform population",
    "steady_state_v9": "lifetime-weighted population",
}

CODEX_FLEET_SYSTEMS = {"baseline_cpu_ssd", "oracle_infinite_hbm"}
CODEX_FLEET_SCALE = 1.5
SCALED_COLUMNS = ("output_tokens_per_s",)

STYLE = {
    "baseline_cpu_ssd": ("#B3423B", "o"),
    "oracle_infinite_hbm": ("#6B6B6B", "s"),
    "hbf_tp8_context": ("#2E6DA4", "^"),
    "hbf_tp4x2": ("#3F8F5F", "v"),
}
LABELS = {
    ("claude", "baseline_cpu_ssd"): "2xGPU + CPU/SSD tiering",
    ("claude", "oracle_infinite_hbm"): "Infinite-HBM oracle",
    ("claude", "hbf_tp8_context"): "GPU+HBF (tp8_context)",
    ("claude", "hbf_tp4x2"): "GPU+HBF (tp4x2)",
    ("codex", "baseline_cpu_ssd"): "3xGPU + CPU/SSD tiering",
    ("codex", "oracle_infinite_hbm"): "Infinite-HBM oracle (3 hosts)",
    ("codex", "hbf_tp4x2"): "GPU+2xHBF hybrid (1:2)",
}
FAMILY_TITLE = {
    "claude": "Claude Code sessions (2-server)",
    "codex": "Codex sessions (3-server)",
}
JCT_FAMILY = {"cl": "claude", "cx": "codex"}
SYSTEM_ORDER = (
    "baseline_cpu_ssd", "oracle_infinite_hbm", "hbf_tp8_context",
    "hbf_tp4x2")
TURN_LEVELS = ("turn10", "turn30", "turn60")
MED_LEVEL = "ttft5_tpot100"
ATTAIN_FRACTION = 0.90


def load_cells(version):
    """Aggregate cells to (family, nominal rate, system) seed means."""
    rows = defaultdict(lambda: defaultdict(list))
    for path in (SCRIPT_ROOT / version / "cells").rglob("*.json"):
        cell = json.loads(path.read_text())
        family, system = cell["family"], cell["system"]
        scale = 1.0
        rate = cell["rate"]
        if family == "codex" and system in CODEX_FLEET_SYSTEMS:
            scale = CODEX_FLEET_SCALE
            rate = round(rate * CODEX_FLEET_SCALE, 6)
        bucket = rows[(family, rate, system)]
        for column in SCALED_COLUMNS:
            bucket[column].append(cell[column] * scale)
        for column in (
                "resume_ttft_p50_s", "resume_ttft_p99_s",
                "tpot_p50_ms", "tpot_p99_ms",
                "turn_latency_mean_s", "turn_latency_p50_s",
                "turn_latency_p99_s"):
            bucket[column].append(cell[column])
        for level, values in cell["slo_levels"].items():
            bucket[f"pass_{level}"].append(values["pass_fraction"])
            bucket[f"goodput_{level}"].append(
                values["good_output_tokens_per_s"] * scale)
        bucket["_seeds"].append(cell["seed"])
    aggregated = {}
    for key, bucket in rows.items():
        aggregated[key] = {
            column: statistics.fmean(values)
            for column, values in bucket.items()
            if column != "_seeds"
        }
        aggregated[key]["n_seeds"] = len(bucket["_seeds"])
    return aggregated


def load_jct(version):
    """Matched-session JCT quantiles keyed by family and nominal rate."""
    pending = defaultdict(dict)
    for path in (SCRIPT_ROOT / version / "jct").glob("jr_*.json"):
        match = re.match(
            r"jr_(cl|cx)_(base|hyb)_([0-9.]+)\.json", path.name)
        if not match:
            continue
        family = JCT_FAMILY[match.group(1)]
        rate = float(match.group(3))
        data = json.loads(path.read_text())["jct_by_session"]
        pending[(family, rate)][match.group(2)] = data
    quantile = lambda v, p: statistics.quantiles(v, n=100)[p - 1]
    table = defaultdict(dict)
    for (family, rate), sides in sorted(pending.items()):
        if set(sides) != {"base", "hyb"}:
            continue
        ids = sorted(set(sides["base"]) & set(sides["hyb"]))
        if len(ids) < 5:
            continue
        base = [sides["base"][i] for i in ids]
        hyb = [sides["hyb"][i] for i in ids]
        table[family][rate] = {
            "n": len(ids),
            "base_p50": quantile(base, 50),
            "base_p99": quantile(base, 99),
            "hyb_p50": quantile(hyb, 50),
            "hyb_p99": quantile(hyb, 99),
        }
    return table


def write_results_csv(cells, out_root):
    columns = sorted({c for v in cells.values() for c in v})
    out = out_root / "final_results.csv"
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "rate", "system", "label"] + columns)
        for (family, rate, system) in sorted(cells):
            row = cells[(family, rate, system)]
            writer.writerow(
                [family, rate, system, LABELS[(family, system)]]
                + [f"{row.get(c, ''):.6g}" if c in row else ""
                   for c in columns])
    return out


def write_attainment_csv(cells, out_root):
    out = out_root / "final_attainment.csv"
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "family", "system", "slo_level",
            f"max_rate_pass{int(ATTAIN_FRACTION * 100)}",
            "goodput_at_max"])
        families = sorted({k[0] for k in cells})
        for family in families:
            systems = sorted({k[2] for k in cells if k[0] == family})
            for system in systems:
                rates = sorted(
                    k[1] for k in cells
                    if k[0] == family and k[2] == system)
                for level in TURN_LEVELS + (MED_LEVEL,):
                    best = None
                    for rate in rates:
                        row = cells[(family, rate, system)]
                        if row.get(
                                f"pass_{level}", 0) >= ATTAIN_FRACTION:
                            best = rate
                    goodput = (
                        "" if best is None else
                        f"{cells[(family, best, system)][f'goodput_{level}']:.1f}")
                    writer.writerow([
                        family, system, level,
                        "" if best is None else f"{best:g}", goodput])
    return out


def write_jct_csv(jct, out_root):
    out = out_root / "final_jct.csv"
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "family", "rate", "matched_sessions",
            "base_p50_s", "hybrid_p50_s", "p50_ratio",
            "base_p99_s", "hybrid_p99_s", "p99_ratio"])
        for family in sorted(jct):
            for rate in sorted(jct[family]):
                row = jct[family][rate]
                writer.writerow([
                    family, rate, row["n"],
                    f"{row['base_p50']:.1f}", f"{row['hyb_p50']:.1f}",
                    f"{row['base_p50'] / row['hyb_p50']:.2f}",
                    f"{row['base_p99']:.1f}", f"{row['hyb_p99']:.1f}",
                    f"{row['base_p99'] / row['hyb_p99']:.2f}"])
    return out


def _series(cells, family, system, column):
    points = sorted(
        (rate, cells[(family, rate, system)][column])
        for (fam, rate, sysk) in cells
        if fam == family and sysk == system
        and column in cells[(family, rate, sysk)])
    return [p[0] for p in points], [p[1] for p in points]


def _systems(cells, family):
    present = {k[2] for k in cells if k[0] == family}
    return [s for s in SYSTEM_ORDER if s in present]


def _finish(ax, *, ylabel, title):
    ax.set_xlabel("Offered session rate (sessions/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


def fig_goodput(cells, family, fig_root, population):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = list(TURN_LEVELS) + [MED_LEVEL]
    captions = {
        "turn10": "Turn SLO 10 s", "turn30": "Turn SLO 30 s",
        "turn60": "Turn SLO 60 s",
        MED_LEVEL: "TTFT 5 s / TPOT 100 ms"}
    for ax, level in zip(axes.flat, panels):
        for system in _systems(cells, family):
            colour, marker = STYLE[system]
            xs, ys = _series(cells, family, system,
                             f"goodput_{level}")
            ax.plot(xs, ys, marker=marker, color=colour,
                    linewidth=1.9, markersize=5,
                    label=LABELS[(family, system)])
        _finish(ax, ylabel="SLO-good output tokens/s",
                title=captions[level])
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        f"{FAMILY_TITLE[family]} - goodput by SLO level "
        f"({population})", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_root / f"{family}_goodput.png", dpi=150)
    plt.close(fig)


def fig_attainment(cells, family, fig_root, population):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, level in zip(axes, TURN_LEVELS):
        for system in _systems(cells, family):
            colour, marker = STYLE[system]
            xs, ys = _series(cells, family, system, f"pass_{level}")
            ax.plot(xs, [y * 100 for y in ys], marker=marker,
                    color=colour, linewidth=1.9, markersize=5,
                    label=LABELS[(family, system)])
        ax.axhline(ATTAIN_FRACTION * 100, color="#999999",
                   linestyle=":", linewidth=1.2)
        _finish(ax, ylabel="Calls within SLO (%)",
                title=f"Turn SLO {level[4:]} s")
        ax.set_ylim(0, 101)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"{FAMILY_TITLE[family]} - turn-SLO attainment, dotted: 90% "
        f"target ({population})", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_root / f"{family}_attainment.png", dpi=150)
    plt.close(fig)


def fig_latency(cells, family, fig_root, population):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = (
        ("resume_ttft_p50_s", "resume_ttft_p99_s",
         "Resume TTFT (s)"),
        ("tpot_p50_ms", "tpot_p99_ms", "TPOT (ms/token)"),
        ("turn_latency_mean_s", "turn_latency_p99_s",
         "Turn latency (s)"),
    )
    for ax, (solid, dashed, ylabel) in zip(axes, panels):
        for system in _systems(cells, family):
            colour, marker = STYLE[system]
            xs, ys = _series(cells, family, system, solid)
            ax.plot(xs, ys, marker=marker, color=colour,
                    linewidth=1.9, markersize=5,
                    label=LABELS[(family, system)])
            xs, ys = _series(cells, family, system, dashed)
            ax.plot(xs, ys, marker=marker, color=colour,
                    linewidth=1.2, markersize=4, linestyle="--")
        ax.set_yscale("log")
        _finish(ax, ylabel=ylabel,
                title=f"{ylabel} (solid p50/mean, dashed p99)")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"{FAMILY_TITLE[family]} - latency ({population})",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_root / f"{family}_latency.png", dpi=150)
    plt.close(fig)


def fig_throughput(cells, family, fig_root, population):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for system in _systems(cells, family):
        colour, marker = STYLE[system]
        xs, ys = _series(cells, family, system, "output_tokens_per_s")
        ax.plot(xs, ys, marker=marker, color=colour, linewidth=1.9,
                markersize=5, label=LABELS[(family, system)])
    _finish(ax, ylabel="Output tokens/s (fleet)",
            title=f"{FAMILY_TITLE[family]} - raw throughput "
                  f"({population})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_root / f"{family}_throughput.png", dpi=150)
    plt.close(fig)


def fig_jct(jct, family, fig_root, population):
    if family not in jct or not jct[family]:
        return
    rates = sorted(jct[family])
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for side, system in (("base", "baseline_cpu_ssd"),
                         ("hyb", "hbf_tp4x2")):
        colour, marker = STYLE[system]
        ax.plot(rates, [jct[family][r][f"{side}_p50"] for r in rates],
                marker=marker, color=colour, linewidth=1.9,
                markersize=5, label=LABELS[(family, system)])
        ax.plot(rates, [jct[family][r][f"{side}_p99"] for r in rates],
                marker=marker, color=colour, linewidth=1.2,
                markersize=4, linestyle="--")
    ax.set_yscale("log")
    counts = [jct[family][r]["n"] for r in rates]
    _finish(ax, ylabel="Session completion time (s)",
            title=f"matched-session JCT (solid p50, dashed p99; "
                  f"n={min(counts)}-{max(counts)} per rate)")
    ax.legend(fontsize=8)
    fig.suptitle(
        f"{FAMILY_TITLE[family]} - JCT vs load ({population})",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_root / f"{family}_jct.png", dpi=150)
    plt.close(fig)


def main():
    for version in VERSIONS:
        out_root = SCRIPT_ROOT / version
        fig_root = (
            SCRIPT_ROOT.parent.parent / "figures" / version)
        fig_root.mkdir(parents=True, exist_ok=True)
        population = POPULATION[version]
        cells = load_cells(version)
        jct = load_jct(version)
        print(version)
        print("  results:", write_results_csv(cells, out_root))
        print("  attainment:", write_attainment_csv(cells, out_root))
        print("  jct:", write_jct_csv(jct, out_root))
        for family in ("claude", "codex"):
            fig_goodput(cells, family, fig_root, population)
            fig_attainment(cells, family, fig_root, population)
            fig_latency(cells, family, fig_root, population)
            fig_throughput(cells, family, fig_root, population)
            fig_jct(jct, family, fig_root, population)
        print("  figures:", fig_root)


if __name__ == "__main__":
    main()

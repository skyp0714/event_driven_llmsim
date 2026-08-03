"""Final v8 report: CSVs and figures for the fixed server-count study.

claude is the 2-server comparison (2xGPU baseline vs GPU+HBF hybrid).
codex is the 3-server comparison: the 1:2 hybrid (one GPU host plus two
HBF hosts, simulated as one fused host via LLMSIM_HBF_HW_SCALE=2) runs
at the nominal rate, while the 3xGPU baseline and oracle were simulated
as the 2-node systems at 2/3 of the nominal rate — share-nothing fleet
scaling — so their throughput-like metrics are multiplied by 1.5 here
and their cells are re-keyed to the nominal rate.

Outputs:
  steady_state_v8/final_results.csv     per family/rate/system metrics
  steady_state_v8/final_attainment.csv  90%-attainment max rates
  steady_state_v8/final_jct.csv         knee-rate matched-session JCT
  ../../figures/steady_state_v8/*.png
"""

from collections import defaultdict
from pathlib import Path
import csv
import json
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_ROOT = Path(__file__).resolve().parent
CELLS_ROOT = SCRIPT_ROOT / "steady_state_v8" / "cells"
OUT_ROOT = SCRIPT_ROOT / "steady_state_v8"
JCT_ROOT = OUT_ROOT / "jct"
FIG_ROOT = SCRIPT_ROOT.parent.parent / "figures" / "steady_state_v8"

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
SYSTEM_ORDER = (
    "baseline_cpu_ssd", "oracle_infinite_hbm", "hbf_tp8_context",
    "hbf_tp4x2")
TURN_LEVELS = ("turn10", "turn30", "turn60")
MED_LEVEL = "ttft5_tpot100"
ATTAIN_FRACTION = 0.90
JCT_KNEE = {"claude": 0.024, "codex": 0.012}
JCT_FILES = {
    ("claude", "baseline_cpu_ssd"): "fin_v8_cl_base.json",
    ("claude", "hbf_tp4x2"): "fin_v8_cl_hyb.json",
    ("codex", "baseline_cpu_ssd"): "fin_v8_cx_base.json",
    ("codex", "hbf_tp4x2"): "fin_v8_cx_hyb.json",
}


def load_cells():
    """Aggregate cells to (family, nominal rate, system) seed means."""
    rows = defaultdict(lambda: defaultdict(list))
    for path in CELLS_ROOT.rglob("*.json"):
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


def write_results_csv(cells):
    columns = sorted({c for v in cells.values() for c in v})
    out = OUT_ROOT / "final_results.csv"
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


def write_attainment_csv(cells):
    out = OUT_ROOT / "final_attainment.csv"
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


def write_jct_csv():
    out = OUT_ROOT / "final_jct.csv"
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "family", "knee_rate", "matched_sessions",
            "base_p50_s", "hybrid_p50_s", "p50_ratio",
            "base_p90_s", "hybrid_p90_s", "p90_ratio"])
        for family in ("claude", "codex"):
            base = json.loads(
                (JCT_ROOT / JCT_FILES[(family, "baseline_cpu_ssd")])
                .read_text())["jct_by_session"]
            hyb = json.loads(
                (JCT_ROOT / JCT_FILES[(family, "hbf_tp4x2")])
                .read_text())["jct_by_session"]
            ids = sorted(set(base) & set(hyb))
            bv = [base[i] for i in ids]
            hv = [hyb[i] for i in ids]
            q = lambda v, p: statistics.quantiles(v, n=100)[p - 1]
            writer.writerow([
                family, JCT_KNEE[family], len(ids),
                f"{q(bv, 50):.1f}", f"{q(hv, 50):.1f}",
                f"{q(bv, 50) / q(hv, 50):.2f}",
                f"{q(bv, 90):.1f}", f"{q(hv, 90):.1f}",
                f"{q(bv, 90) / q(hv, 90):.2f}"])
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


def _finish(ax, *, ylabel, title, note=None):
    ax.set_xlabel("Offered session rate (sessions/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    if note:
        ax.text(
            0.02, 0.02, note, transform=ax.transAxes, fontsize=7,
            color="#555555")


def fig_goodput(cells, family):
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
        f"{FAMILY_TITLE[family]} - goodput by SLO level", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / f"{family}_goodput.png", dpi=150)
    plt.close(fig)


def fig_attainment(cells, family):
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
        f"{FAMILY_TITLE[family]} - turn-SLO attainment "
        "(dotted: 90% target)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / f"{family}_attainment.png", dpi=150)
    plt.close(fig)


def fig_latency(cells, family):
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
        f"{FAMILY_TITLE[family]} - latency", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / f"{family}_latency.png", dpi=150)
    plt.close(fig)


def fig_throughput(cells, family):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for system in _systems(cells, family):
        colour, marker = STYLE[system]
        xs, ys = _series(cells, family, system, "output_tokens_per_s")
        ax.plot(xs, ys, marker=marker, color=colour, linewidth=1.9,
                markersize=5, label=LABELS[(family, system)])
    _finish(ax, ylabel="Output tokens/s (fleet)",
            title=f"{FAMILY_TITLE[family]} - raw throughput")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / f"{family}_throughput.png", dpi=150)
    plt.close(fig)


def fig_jct(family):
    base = json.loads(
        (JCT_ROOT / JCT_FILES[(family, "baseline_cpu_ssd")])
        .read_text())["jct_by_session"]
    hyb = json.loads(
        (JCT_ROOT / JCT_FILES[(family, "hbf_tp4x2")])
        .read_text())["jct_by_session"]
    ids = sorted(set(base) & set(hyb))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, data, system in (
            ("baseline", [base[i] for i in ids], "baseline_cpu_ssd"),
            ("hybrid", [hyb[i] for i in ids], "hbf_tp4x2")):
        values = sorted(data)
        cdf = [(i + 1) / len(values) for i in range(len(values))]
        colour, _ = STYLE[system]
        ax.plot(values, cdf, color=colour, linewidth=1.9,
                label=LABELS[(family, system)])
    ax.set_xscale("log")
    ax.set_xlabel("Session completion time (s)")
    ax.set_ylabel("CDF over matched sessions")
    ax.set_title(
        f"{FAMILY_TITLE[family]} - matched-session JCT at knee "
        f"rate {JCT_KNEE[family]:g} (n={len(ids)})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / f"{family}_jct_cdf.png", dpi=150)
    plt.close(fig)


def main():
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    print("results:", write_results_csv(cells))
    print("attainment:", write_attainment_csv(cells))
    print("jct:", write_jct_csv())
    for family in ("claude", "codex"):
        fig_goodput(cells, family)
        fig_attainment(cells, family)
        fig_latency(cells, family)
        fig_throughput(cells, family)
        fig_jct(family)
    print("figures:", FIG_ROOT)


if __name__ == "__main__":
    main()

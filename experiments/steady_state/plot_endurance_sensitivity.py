#!/usr/bin/env python3
"""How much of the HBF endurance answer is measurement, and how much is assumption.

The simulator measures exactly one quantity here: the payload bytes written to
HBF per second of steady state.  Everything that turns that into a lifetime in
years is an assumption -- how many program/erase cycles the media is rated for,
how much erase and garbage-collection overhead sits on top of the payload
(WAF), and how much of each card is actually a writable KV region.  This figure
sweeps each of those so the reader can see which conclusions survive and which
depend on a number nobody has measured for HBF yet.
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

HBF_CARDS = 8
HBF_CAPACITY_BYTES_PER_CARD = 1_280_000_000_000
MODEL_WEIGHT_BYTES_PER_CARD = 7_680_585_728
KV_REGION_BYTES_PER_CARD = (
    HBF_CAPACITY_BYTES_PER_CARD - MODEL_WEIGHT_BYTES_PER_CARD)
DAYS_PER_YEAR = 365.0
DEPLOYMENT_YEARS = 5.0

FAMILY_STYLE = {
    "claude": ("#2E6DA4", "o", "Claude Code sessions"),
    "codex": ("#B3423B", "^", "Codex sessions"),
}
WAF_STYLE = {1.0: "-", 1.3: "--", 2.0: "-.", 4.0: ":"}


def lifetime_years(write_bytes_per_s: float, *, pe_cycles: float, waf: float,
                   kv_region_bytes: float = KV_REGION_BYTES_PER_CARD,
                   cards: int = HBF_CARDS) -> float | None:
    """Years until one card exhausts its rated full-region write budget."""

    if write_bytes_per_s <= 0:
        return None
    per_card_per_s = write_bytes_per_s / cards
    cycles_per_day = per_card_per_s * 86_400.0 * waf / kv_region_bytes
    if cycles_per_day <= 0:
        return None
    return pe_cycles / cycles_per_day / DAYS_PER_YEAR


def load(root: Path):
    rows = []
    with (root / "economics.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            window = float(row["measurement_window_s"] or 0)
            written = float(row["hbf_write_bytes"] or 0)
            if window <= 0 or written <= 0:
                continue
            rows.append({
                "family": row["family"],
                "rate": float(row["rate"]),
                "concurrency": float(row["target_concurrency"] or 0),
                "bytes_per_s": written / window,
            })
    return rows


def panel_vs_load(ax, rows):
    for family, (colour, marker, label) in FAMILY_STYLE.items():
        series = sorted(
            (r for r in rows if r["family"] == family),
            key=lambda r: r["rate"])
        if not series:
            continue
        rates = [r["rate"] for r in series]
        for waf, style in WAF_STYLE.items():
            years = [
                lifetime_years(r["bytes_per_s"], pe_cycles=100_000.0, waf=waf)
                for r in series]
            ax.plot(rates, years, style, color=colour, marker=marker,
                    markersize=3.5, linewidth=1.5,
                    label=f"{label}, WAF {waf:g}")
    ax.axhline(DEPLOYMENT_YEARS, color="#333333", linewidth=1.3,
               linestyle="-", alpha=0.8)
    ax.text(0.02, 0.06, "5-year deployment", transform=ax.transAxes,
            fontsize=7, color="#333333")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Session arrival rate $\\lambda$ (sessions/s)")
    ax.set_ylabel("Years to rated write budget")
    ax.set_title("Lifetime falls with offered load\n(SLC 100k P/E)",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=5.5, ncol=2)


def panel_vs_pe(ax, rows):
    """The rating is the single largest unknown, so sweep it."""
    pe_grid = [10 ** (3 + 0.1 * i) for i in range(41)]  # 1e3 .. 1e7
    for family, (colour, marker, label) in FAMILY_STYLE.items():
        series = sorted(
            (r for r in rows if r["family"] == family),
            key=lambda r: r["rate"])
        if not series:
            continue
        worst = series[-1]
        for waf, style in ((1.0, "-"), (2.0, "--"), (4.0, ":")):
            years = [
                lifetime_years(worst["bytes_per_s"], pe_cycles=pe, waf=waf)
                for pe in pe_grid]
            ax.plot(pe_grid, years, style, color=colour, linewidth=1.5,
                    label=f"{label} @ $\\lambda$={worst['rate']:g}, "
                          f"WAF {waf:g}")
    ax.axhline(DEPLOYMENT_YEARS, color="#333333", linewidth=1.3, alpha=0.8)
    for pe, name in ((3_000.0, "TLC"), (30_000.0, "MLC"),
                     (100_000.0, "SLC"), (1_000_000.0, "relaxed\nretention")):
        ax.axvline(pe, color="#999999", linewidth=0.7, linestyle=":")
        ax.text(pe, ax.get_ylim()[0], f" {name}", fontsize=6,
                color="#666666", rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rated full-region writes (P/E cycles)")
    ax.set_ylabel("Years to rated write budget")
    ax.set_title("Media rating is the dominant assumption\n"
                 "(at each family's highest measured load)", fontsize=10)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=5.5)


def panel_headroom(ax, rows):
    """How much more write traffic the design absorbs before 5 years."""
    for family, (colour, marker, label) in FAMILY_STYLE.items():
        series = sorted(
            (r for r in rows if r["family"] == family),
            key=lambda r: r["rate"])
        if not series:
            continue
        rates = [r["rate"] for r in series]
        for waf, style in ((1.0, "-"), (2.0, "--"), (4.0, ":")):
            factors = []
            for r in series:
                years = lifetime_years(
                    r["bytes_per_s"], pe_cycles=100_000.0, waf=waf)
                factors.append(
                    years / DEPLOYMENT_YEARS if years else float("nan"))
            ax.plot(rates, factors, style, color=colour, marker=marker,
                    markersize=3.5, linewidth=1.5,
                    label=f"{label}, WAF {waf:g}")
    ax.axhline(1.0, color="#333333", linewidth=1.3, alpha=0.8)
    ax.text(0.02, 0.06, "break-even at 5 years", transform=ax.transAxes,
            fontsize=7, color="#333333")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Session arrival rate $\\lambda$ (sessions/s)")
    ax.set_ylabel("Write-rate headroom ($\\times$ measured)")
    ax.set_title("Headroom before endurance binds\n"
                 "(multiple of the measured write rate)", fontsize=10)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=5.5, ncol=2)


def panel_kv_region(ax, rows):
    """A smaller writable region wears out proportionally faster."""
    fractions = [0.1 * i for i in range(1, 11)]
    for family, (colour, marker, label) in FAMILY_STYLE.items():
        series = sorted(
            (r for r in rows if r["family"] == family),
            key=lambda r: r["rate"])
        if not series:
            continue
        worst = series[-1]
        years = [
            lifetime_years(
                worst["bytes_per_s"], pe_cycles=100_000.0, waf=2.0,
                kv_region_bytes=KV_REGION_BYTES_PER_CARD * fraction)
            for fraction in fractions]
        ax.plot([100 * f for f in fractions], years, "-", color=colour,
                marker=marker, markersize=4, linewidth=1.7,
                label=f"{label} @ $\\lambda$={worst['rate']:g}")
    ax.axhline(DEPLOYMENT_YEARS, color="#333333", linewidth=1.3, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Share of the card usable as KV region (%)")
    ax.set_ylabel("Years to rated write budget")
    ax.set_title("Wear levels over the writable region\n"
                 "(SLC 100k P/E, WAF 2.0)", fontsize=10)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=6)


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
    if not rows:
        raise SystemExit("no HBF write samples in economics.csv")
    args.out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    panel_vs_load(axes[0][0], rows)
    panel_vs_pe(axes[0][1], rows)
    panel_headroom(axes[1][0], rows)
    panel_kv_region(axes[1][1], rows)
    fig.suptitle(
        "HBF write endurance: one measured quantity, three assumptions",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = args.out / "write_endurance_sensitivity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"wrote {path}")

    print(f"\n{'family':8}{'lambda':>8}{'L':>6}{'TB/day':>9}"
          f"{'100k/WAF1':>11}{'100k/WAF2':>11}{'100k/WAF4':>11}"
          f"{'1M/WAF2':>11}")
    for row in sorted(rows, key=lambda r: (r["family"], r["rate"])):
        per_day_tb = row["bytes_per_s"] * 86_400 / 1e12
        cells = [
            lifetime_years(row["bytes_per_s"], pe_cycles=pe, waf=waf)
            for pe, waf in ((100_000.0, 1.0), (100_000.0, 2.0),
                            (100_000.0, 4.0), (1_000_000.0, 2.0))]
        print(f"{row['family']:8}{row['rate']:8.4f}{row['concurrency']:6.0f}"
              f"{per_day_tb:9.2f}" + "".join(f"{c:11.1f}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

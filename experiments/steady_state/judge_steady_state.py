"""Sanity-judge the open-system steady-state sweep.

The sweep is only worth plotting if the harness itself is sound.  This script
answers four questions that would each invalidate the campaign:

1. Did the population actually load?  Pre-population replaces a warmup ramp,
   so a cell whose preload was mostly skipped never reached the intended L.
2. Is the offered load the knob we think it is?  Little's Law says the
   resident population should track lambda * W; if measured concurrency is
   flat in lambda, the knob is not connected.
3. Do latencies move monotonically with load?  A queueing system that does
   not degrade under rising load is not being loaded.
4. Does the oracle bound the baseline?  Infinite HBM removes eviction, so it
   can never be slower than the tiered baseline by more than noise.  A
   violation means the two systems differ in something other than capacity.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
NOISE_TOLERANCE = 0.05


def load_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith(".tmp"):
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        # aggregate.json / economics.json live beside the cells.
        if isinstance(record, dict) and "system" in record:
            rows.append(record)
    return rows


def group(rows, *keys):
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in keys)].append(row)
    return out


def fmean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else float("nan")


def check_preload(rows) -> list[str]:
    problems = []
    for row in rows:
        placed = row.get("preloaded") or {}
        if row["system"] == "oracle_infinite_hbm":
            continue  # unbounded HBM: residency is implicit, nothing to place
        total = sum(v for k, v in placed.items() if k != "skipped")
        skipped = placed.get("skipped", 0)
        want = row["target_concurrency"]
        if want and skipped / max(1, want) > 0.10:
            problems.append(
                f"{row['family']} rate={row['rate']} seed={row['seed']} "
                f"{row['system']}: preload skipped {skipped}/{want} "
                f"({skipped / want:.0%}) -> population never reached target")
        elif want and total < want * 0.90:
            problems.append(
                f"{row['family']} rate={row['rate']} {row['system']}: "
                f"placed {total} of {want}")
    return problems


def report(rows) -> int:
    failures = 0
    print(f"{len(rows)} cells loaded\n")

    print("=" * 78)
    print("1. PRE-POPULATION")
    print("=" * 78)
    problems = check_preload(rows)
    if problems:
        failures += 1
        for line in problems[:12]:
            print(f"  FAIL {line}")
        if len(problems) > 12:
            print(f"  ... {len(problems) - 12} more")
    else:
        print("  OK  every tiered cell placed >=90% of its target population")

    print()
    print("=" * 78)
    print("2. LOAD KNOB (Little's Law: L = lambda * W)")
    print("=" * 78)
    print(f"  {'family':7} {'system':20} {'rate':>8} {'target_L':>9} "
          f"{'calls':>8} {'window_s':>10} {'tok/s':>9}")
    for key in sorted(group(rows, "family", "system", "rate")):
        family, system, rate = key
        cells = group(rows, "family", "system", "rate")[key]
        print(f"  {family:7} {system:20} {rate:8.4f} "
              f"{fmean(c['target_concurrency'] for c in cells):9.0f} "
              f"{fmean(c['scored_calls'] for c in cells):8.0f} "
              f"{fmean(c['measurement_window_s'] for c in cells):10.0f} "
              f"{fmean(c['output_tokens_per_s'] for c in cells):9.1f}")

    print()
    print("=" * 78)
    print("3. MONOTONICITY IN LOAD")
    print("=" * 78)
    for key in sorted(group(rows, "family", "system")):
        family, system = key
        cells = group(rows, "family", "system")[key]
        by_rate = {key[0]: value for key, value in group(cells, "rate").items()}
        rates = sorted(by_rate)
        series = [fmean(c["resume_ttft_p95_s"] for c in by_rate[r])
                  for r in rates]
        tpots = [fmean(c["tpot_p95_ms"] for c in by_rate[r]) for r in rates]
        inversions = sum(
            1 for a, b in zip(series, series[1:])
            if b < a * (1.0 - NOISE_TOLERANCE))
        flag = "OK " if inversions == 0 else "WARN"
        if inversions:
            failures += 1
        print(f"  {flag} {family:7} {system:20}")
        print(f"       rate       " + " ".join(f"{r:>8.4f}" for r in rates))
        print(f"       resume p95 " + " ".join(f"{v:>8.2f}" for v in series))
        print(f"       tpot p95   " + " ".join(f"{v:>8.2f}" for v in tpots))

    print()
    print("=" * 78)
    print("4. ORACLE BOUNDS BASELINE")
    print("=" * 78)
    violations = []
    by_cell = group(rows, "family", "rate", "seed")
    for key, cells in sorted(by_cell.items()):
        systems = {c["system"]: c for c in cells}
        base = systems.get("baseline_cpu_ssd")
        orac = systems.get("oracle_infinite_hbm")
        if not base or not orac:
            continue
        for metric in ("resume_ttft_p95_s", "tpot_p95_ms",
                       "turn_latency_mean_s"):
            b, o = base[metric], orac[metric]
            if b <= 0:
                continue
            if o > b * (1.0 + NOISE_TOLERANCE):
                violations.append(
                    f"{key[0]} rate={key[1]} seed={key[2]} {metric}: "
                    f"oracle {o:.3f} > baseline {b:.3f} "
                    f"(+{100 * (o / b - 1):.1f}%)")
    if violations:
        failures += 1
        for line in violations[:15]:
            print(f"  FAIL {line}")
        if len(violations) > 15:
            print(f"  ... {len(violations) - 15} more")
    else:
        print("  OK  oracle never exceeds baseline beyond noise")

    print()
    print("=" * 78)
    print("5. SEPARATION (does tiering ever cost anything?)")
    print("=" * 78)
    print(f"  {'family':7} {'rate':>8} {'base p95':>10} {'orac p95':>10} "
          f"{'gap':>8}   {'base tpot':>10} {'orac tpot':>10}")
    for key in sorted(group(rows, "family", "rate")):
        cells = group(rows, "family", "rate")[key]
        base = [c for c in cells if c["system"] == "baseline_cpu_ssd"]
        orac = [c for c in cells if c["system"] == "oracle_infinite_hbm"]
        if not base or not orac:
            continue
        b = fmean(c["resume_ttft_p95_s"] for c in base)
        o = fmean(c["resume_ttft_p95_s"] for c in orac)
        gap = (b / o - 1) * 100 if o > 0 else float("nan")
        print(f"  {key[0]:7} {key[1]:8.4f} {b:10.2f} {o:10.2f} "
              f"{gap:7.1f}%   "
              f"{fmean(c['tpot_p95_ms'] for c in base):10.2f} "
              f"{fmean(c['tpot_p95_ms'] for c in orac):10.2f}")

    print()
    print("=" * 78)
    print("6. TIER TRAFFIC (is the baseline actually evicting?)")
    print("=" * 78)
    tier_keys = [
        k for k in (rows[0].get("counters") or {})
        if any(t in k for t in ("cpu", "ssd", "evict", "restore", "demot"))]
    for key in sorted(group(rows, "family", "rate")):
        cells = [c for c in group(rows, "family", "rate")[key]
                 if c["system"] == "baseline_cpu_ssd"]
        if not cells:
            continue
        totals = {
            k: fmean(c["counters"].get(k, 0) for c in cells)
            for k in tier_keys}
        active = {k: v for k, v in totals.items() if v > 0}
        summary = "  ".join(f"{k}={v:,.0f}" for k, v in
                            sorted(active.items(), key=lambda kv: -kv[1])[:6])
        print(f"  {key[0]:7} rate={key[1]:<8.4f} "
              f"{summary if summary else '(no tier traffic)'}")

    print()
    print("=" * 78)
    print(f"VERDICT: {'PASS' if failures == 0 else f'{failures} check(s) failed'}")
    print("=" * 78)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=SCRIPT_ROOT / "steady_state_v1")
    args = parser.parse_args()
    rows = load_rows(args.root)
    if not rows:
        raise SystemExit(f"no result rows under {args.root}")
    return report(rows)


if __name__ == "__main__":
    raise SystemExit(main())

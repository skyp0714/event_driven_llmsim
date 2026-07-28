#!/usr/bin/env python3
"""Ablate the HBF migration policy at the loaded end of the sweep.

The main campaign runs one policy (`load_aware`).  Under load that policy
almost never promotes: it compares HBF-side queue pressure against GPU-side
queue pressure and only migrates when HBF is the quieter of the two, so once
the HBF host is busy serving the resident population it stops accepting new
sessions -- which is precisely when a session would most benefit from moving
there.  Whether that is the right call is an empirical question, so this
script re-runs the loaded operating points under each candidate policy and
reports goodput, resume tail, and how much migration each one actually did.

Only the top rates are swept: at low load nothing is evicting, so no policy
can distinguish itself there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

POLICIES = (
    "load_aware",
    "eager",
    "composite_ready_adaptive",
    "composite",
)
# Each policy preloads at the split its own ranking would converge to.
PRELOAD_SPLIT_BY_POLICY = {
    "load_aware_density": "request_density",
    "load_aware_density_oracle": "request_density_oracle",
    "load_aware_cost": "cost_dcap",
    "load_aware_calls": "hot_dcap",
    "load_aware_calls_oracle": "hot_dcap_oracle",
}
DEFAULT_RATES = (0.004, 0.008, 0.016)
DEFAULT_SEEDS = (101, 102, 103)


def run_one(task: dict) -> dict:
    os.environ["LLMSIM_MIGRATION_POLICY"] = task["policy"]
    os.environ["LLMSIM_PRELOAD_SPLIT"] = PRELOAD_SPLIT_BY_POLICY.get(
        task["policy"], "context_half")
    import run_steady_state_campaign as S

    row = S.run_cell({
        "family": task["family"],
        "rate": task["rate"],
        "seed": task["seed"],
        "system": "hbf_tp8_context",
        "measured_calls": task["measured_calls"],
        "out_path": task["out_path"],
    })
    row["migration_policy"] = task["policy"]
    Path(task["out_path"]).write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=SCRIPT_ROOT / "policy_ablation_v1")
    parser.add_argument("--families", nargs="+", default=["claude", "codex"])
    parser.add_argument(
        "--rates", type=float, nargs="+", default=list(DEFAULT_RATES))
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--measured-calls", type=int, default=4_000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tasks = []
    for policy in args.policies:
        for family in args.families:
            for rate in args.rates:
                for seed in args.seeds:
                    out_path = (
                        args.output_root / policy / family
                        / f"rate{rate:g}_seed{seed}.json")
                    if args.resume and out_path.is_file():
                        continue
                    tasks.append({
                        "policy": policy, "family": family, "rate": rate,
                        "seed": seed, "measured_calls": args.measured_calls,
                        "out_path": str(out_path),
                    })
    if not tasks:
        print("nothing to run")
        return 0

    print(f"policy ablation: {len(tasks)} cells  workers={args.workers}",
          flush=True)
    started = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            done += 1
            try:
                row = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"[{done}/{len(tasks)}] FAILED {task['policy']} "
                      f"{task['family']} rate={task['rate']}: {error!r}",
                      flush=True)
                continue
            counters = row.get("counters") or {}
            print(f"[{done}/{len(tasks)}] {task['policy']:26} "
                  f"{task['family']:7} rate={task['rate']:<7} "
                  f"seed={task['seed']} "
                  f"resume_p99={row['resume_ttft_p99_s']:8.2f} "
                  f"tpot_p99={row['tpot_p99_ms']:7.1f} "
                  f"migr={counters.get('migrations_committed', 0):6,} "
                  f"defer={counters.get('promotion_load_deferrals', 0):8,} "
                  f"wall={row['wall_s']:5.0f}s", flush=True)
    print(f"complete in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

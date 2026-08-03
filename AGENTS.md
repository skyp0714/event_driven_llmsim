# AGENTS.md

Guidelines for AI coding agents working in this repository.

## Project context

This is `event_driven_llmsim`, a research fork of LLMServingSim. The
active work is the **event-driven, kernel-calibrated KV-tier
simulator** in `serving/core/` and the **GPU+HBF hybrid study** in
`experiments/` (see `README.md` and `experiments/README.md`). The
upstream LLMServingSim components (`profiler/`, `bench/`, `docs/`,
`astra-sim/`, `serving/__main__.py` flow) remain in-tree but are not
the focus; follow upstream conventions only when editing them.

### Key modules (`serving/core/`)

- `online_latency_model.py` — calibrated analytical kernel latencies.
  Calibration sources live under `profiler/v0/` and `results/`; they
  look like legacy data but are **load-bearing — never delete**.
- `gpu_pd_*.py` — P4D4 GPU host: pool, tiered node, KV tier lifecycle
  (D-HBM/CPU/SSD ledgers, prepare/handoff/demotion). The tier policies
  are `hbm_lru_recompute`, `cpu_ssd`, `ssd_direct`; every eviction
  path must terminate — a full tier with no drain is a livelock (see
  the `ssd_direct` CPU-spill regression test).
- `hbf_full_model_*.py` — HBF host: flash/LPDDR rooflines, worker
  queues (waiting/prefill_drain/active_decode), lifecycle ledgers.
- `gpu_ssd_hbf_hybrid.py` — hybrid node: routing, SSD-staged
  promotion/demotion, big-prefill mirror gate.
- `ssd_hbf_design_sweep.py` — design specs and system construction;
  env knobs `LLMSIM_MIGRATION_POLICY`, `LLMSIM_HBF_READ_MODE`,
  `LLMSIM_HBF_HW_SCALE` (host-ratio studies).
- `hbf_comparison_workload.py` / `hbf_comparison_metrics.py` —
  session traces, `CompletedRequest` snapshots.

### Experiments

- `experiments/steady_state/run_steady_state_campaign.py` runs cells;
  population via `LLMSIM_RESIDENT_SAMPLING=uniform|lifetime`;
  `experiments/steady_state/final_report.py` writes CSVs + figures.
  `run_final_campaign.sh` records the exact v8/v9 launch.
- codex 3-server arms use share-nothing fleet scaling (baseline and
  oracle at 2/3 rate, metrics x1.5 at analysis) and the fused-host
  `LLMSIM_HBF_HW_SCALE=2` hybrid; claude is the plain 2-server
  comparison.
- Verify policy changes at full campaign scale before concluding —
  short-window wins have repeatedly evaporated at scale. When one
  seed of a cell diverges wildly from another, suspect a stuck-cohort
  livelock (survivorship makes the sick cell's per-call metrics look
  good) before believing either number.

## Code style

- Python, 4-space indent, snake_case; match surrounding style.
- Comments and log messages in English only.
- No `getattr` fallbacks for attributes you can initialize in
  `__init__`; initialize and access directly.
- Keep commits focused; short imperative messages.

## Testing & validation

- `python -m pytest tests/ -q` — the system models have a real suite
  (`test_gpu_pd_tier_lifecycle.py`, `test_gpu_ssd_hbf_hybrid.py`,
  `test_ssd_hbf_design_sweep.py`, ...). Add a regression test when
  fixing a system-model bug.
- For behavioral changes, run the smallest relevant campaign cell and
  inspect the cell JSON before a full sweep.

## Common pitfalls

- **Don't delete `profiler/v0/` or `results/`** — live calibration
  inputs for the latency model.
- **Don't edit `astra-sim/`** unless the change targets upstream
  simulator integration.
- **Don't commit large generated artifacts** (traces, logs, `.et`
  files); campaign cell JSONs and final CSVs/figures are the
  exception and are versioned deliberately.
- Simulator scripts assume the repo venv
  (`~/.venvs/llmsim/bin/python`).
- `Date`-like nondeterminism does not exist in the simulator: cells
  are deterministic given (config, seed) — identical outputs across
  reruns are expected, not suspicious.

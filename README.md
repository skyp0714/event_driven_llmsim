# event_driven_llmsim

An event-driven, kernel-calibrated simulator for studying KV-cache
tier designs in agentic LLM serving — in particular a **GPU + HBF
(High-Bandwidth Flash) hybrid** fleet against an all-GPU CPU/SSD
tiering baseline on real Claude Code and Codex session traces.

This repository is a research fork of
[LLMServingSim](https://github.com/casys-kaist/LLMServingSim); the
upstream cycle-level simulator, profiler, and bench live on unchanged
(docs at [llmservingsim.ai](https://llmservingsim.ai)), while the work
here is the analytical event-driven model under `serving/core/` and
the campaign in `experiments/`.

## What it models

- **P4D4 GPU host**: 4 prefill + 4 decode H100s, chunked prefill,
  continuous batching, and a finite-HBM KV lifecycle with
  D-HBM -> CPU DRAM -> SSD tiering (`gpu_pd_*.py`).
- **HBF host**: eight flash-based cards (weights + committed KV read
  from flash at HBM-class bandwidth, LPDDR for activations) serving
  resumed long-context sessions in place (`hbf_full_model_*.py`).
- **Hybrid node**: SSD-staged migration between the two hosts with a
  load-aware promotion/demotion policy and a big-prefill mirror gate
  (`gpu_ssd_hbf_hybrid.py`, `ssd_hbf_design_sweep.py`).
- **Latency**: analytical roofline models calibrated against kernel
  measurements (`online_latency_model.py`; calibration sources under
  `profiler/v0/` and `results/` are load-bearing — do not delete).
- **Workload**: closed-loop agentic sessions (tool-call chains with
  think times) replayed from traces, with an open-system steady-state
  population installed by Little's law.

## Reproducing the study

```bash
# all campaign cells (v8 uniform + v9 lifetime populations)
./run_final_campaign.sh
# CSVs + figures
python experiments/steady_state/final_report.py
```

Design, population models, policy configuration, and headline results
are documented in [`experiments/README.md`](experiments/README.md).
Summary: at equal server count the hybrid raises the 90%-attainment
turn-SLO capacity 20-40% over the tiering baseline, reaching the
infinite-HBM oracle at several operating points, with matched-session
JCT up to 3-4x faster at the knee (10 seeds, error-barred).

## Layout

```
serving/core/          event-driven system models (this fork's work)
experiments/           final v8/v9 campaign: runner, data, report
figures/               generated figures (steady_state_v8 / _v9)
tests/                 pytest suite for the system models
profiler/, bench/,     upstream LLMServingSim components (see their
docs/, astra-sim/      own READMEs; docs at llmservingsim.ai)
```

## Upstream

LLMServingSim 2.0 (ISPASS 2026): *A Unified Simulator for
Heterogeneous and Disaggregated LLM Serving Infrastructure* — Cho,
Choi, Heo, Park (KAIST). If you build on the upstream simulator,
cite their papers (see the upstream repository).

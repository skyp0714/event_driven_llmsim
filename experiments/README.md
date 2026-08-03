# GPU+HBF hybrid serving for agentic workloads — final study (v8/v9)

One consolidated record of the steady-state campaign comparing a
GPU+HBF (High-Bandwidth Flash) hybrid serving fleet against an
all-GPU CPU/SSD-tiering baseline on real agentic traces (Claude Code
and Codex session families), with an infinite-HBM oracle as the
performance reference. Everything below is reproducible from
`steady_state/run_steady_state_campaign.py` (cells) and
`steady_state/final_report.py` (CSVs + figures).

## Systems and server-count parity

| family | comparison | baseline | hybrid |
|---|---|---|---|
| claude | 2 servers | 2x P4D4 GPU hosts, D-HBM -> CPU -> SSD tiering | 1 GPU host + 1 eight-card HBF host |
| codex | 3 servers | 3x GPU hosts | 1 GPU host + 2 HBF hosts (1:2) |

The HBF host holds idle-session KV on flash (3,350 Gbps/card reads,
prefetch hides the 5 us first-access latency) and serves resumed
sessions in place, avoiding the baseline's SSD-restore storm. The
codex 3-server arms are simulated with share-nothing fleet scaling:
the 3xGPU baseline and oracle run the 2-node systems at 2/3 of the
nominal rate (metrics x1.5 at analysis time), and the 1:2 hybrid runs
as one fused host with doubled per-card rooflines
(`LLMSIM_HBF_HW_SCALE=2`). The ratio study that picked 1:2 over 2:1
showed more HBF is what absorbs the standing KV mass; halving HBF per
GPU collapses attainment at high load.

## Population models (v8 vs v9)

Residents are preloaded at t=0 to the Little's-law standing population
`L = rate x mean lifetime`, with resume points sampled length-biased
within each session. Between sessions:

- **v8 (uniform)**: session templates drawn uniformly — a young
  deployment whose history is shorter than the trace's long tail.
- **v9 (lifetime)**: templates drawn proportional to session lifetime —
  the renewal-theory steady state, dominated by heavy long-lived
  sessions (`LLMSIM_RESIDENT_SAMPLING=lifetime`).

Both are committed so either frame can be cited. v9 stresses tier
capacity ~10x harder; it is what exposed (and now regression-tests)
the `ssd_direct` CPU-tier eviction livelock fixed in
`serving/core/gpu_pd_tier_lifecycle.py`.

## Final policy configuration

`LLMSIM_MIGRATION_POLICY=load_aware_demote_h2_bigp` +
`LLMSIM_HBF_READ_MODE=prefetch`:

- context-ranked promotion of mature idle sessions to HBF with a
  load-aware gate;
- demotion on persistent imbalance (dynamic-score floor 4.0,
  hysteresis 2.0, D-fit guard);
- big-prefill mirror gate: sessions whose fresh-input EWMA exceeds 4k
  tokens are mirrored to SSD and served by the GPU side, keeping
  input-heavy prefills from stalling co-batched HBF decodes;
- prefetch read mode on the HBF card.

Eleven other levers (density/calls oracles, decode-KV guard, SJF,
mixed guard, PD slots, chunk caps, utilization balancing, big-output
gates, preload fractions) were tested and refuted at campaign scale.

## Campaign spec

- Rates (dense past the knee): claude 0.016-0.036 (8 points), codex
  nominal 0.0075-0.0165 (7 points); seeds 101/102; 6,000 measured
  calls per cell; 212 cells, 0 failures. Launch script:
  `../run_final_campaign.sh`.
- Metrics per cell: 3x3 TTFT/TPOT SLO grid + turn-SLO (10/30/60 s)
  goodput and pass fractions, resume TTFT, TPOT, turn latency,
  raw throughput, write accounting.
- Matched-session JCT from the instrument harness at three rates per
  family (`steady_state_v*/jct/jr_*.json`), intersection-matched
  between baseline and hybrid.

## Headline results

90%-attainment max rate (turn-SLO):

| | baseline | hybrid | oracle |
|---|---|---|---|
| claude v8, turn60 | 0.020 | **0.026 (+30%)** | 0.026 |
| claude v8, turn30 | below grid | **0.016** | below grid |
| codex v8, turn30 | 0.0075 | **0.0105 (+40%)** | 0.0105 |
| codex v8, turn60 | 0.0135 | **0.015 (+11%)** | 0.0165 |
| claude v9, turn60 | below grid | **0.020** | 0.020 |
| codex v9, turn30/60 | 0.015 / 0.0165 | parity | 0.0165 / 0.0165 |

The hybrid reaches the infinite-HBM oracle's attainment capacity in
both families under v8. Goodput ratios grow with load: claude
turn30-goodput 1.09->1.74x (v8) with medium-SLO goodput up to 4.5x;
codex 1.15-1.29x (v8). Under v9 the claude advantage amplifies
(turn30-goodput peak 1.69x, turn latency up to 2.1x, resume TTFT p99
up to 11.6x) while codex holds parity at equal server count with a
cheaper bill of materials. Matched JCT at the knee: claude p50 3.2x /
p90 1.6x faster (v8) and p50 2.7x / p90 4.5x (v9); codex p50 parity /
p90 1.2x faster (v8). turn10 at 90% is unattainable for every system
including the unloaded oracle — a workload-shape bound, not a design
gap.

Mechanisms, in one paragraph: the baseline's failure mode is the
restore storm — resumed long-context sessions pay SSD->HBM restores
that queue behind each other and poison TTFT tails, while its decode
batches stay HBM-fast. The hybrid keeps mature KV resident on flash
and pays a small per-token read cost instead, so it wins exactly when
the standing KV mass exceeds fast-tier capacity (late-session-heavy
populations, high rates) and ties otherwise. Input-heavy sessions are
the externality — their prefills stall co-batched HBF decodes — so
the big-prefill gate routes them to the GPU side (this is why the
equivalent big-output gate failed: output-heavy sessions pay their own
cost). Capacity conservation holds at saturation: no scheduling trick
moves failure mass off both SLO axes at once, which is why the
attainment frontier, not any single latency metric, is the honest
summary.

## Files

```
experiments/
  README.md                        this document
  steady_state/
    run_steady_state_campaign.py   cell runner (population via
                                   LLMSIM_RESIDENT_SAMPLING)
    phase_breakdown.py             per-cell phase decomposition
    final_report.py                CSVs + figures for v8/v9
    steady_state_v8/               uniform-population cells, CSVs, jct/
    steady_state_v9/               lifetime-population cells, CSVs, jct/
  session_scaling/
    run_session_scaling_campaign.py  system builders (_build_system),
                                     shared by the steady-state runner
figures/steady_state_v8/           goodput, attainment, latency,
figures/steady_state_v9/           throughput, JCT-vs-rate per family
run_final_campaign.sh              exact launch commands for v8/v9
```

Earlier exploratory campaigns (session-count scaling, the one-server
HBF-prefill design, steady-state v1-v7) and their per-version
analysis scripts were removed once v8/v9 superseded them; they remain
in git history before commit `"Add v8 final report"`.

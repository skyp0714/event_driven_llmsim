# hbf_prefill: one-server HBF-prefill P4D4 vs CPU/SSD tiering

Two eight-card single-server systems under the same open-system
steady-state workload, plus an infinite-HBM reference:

| system | prefill role | decode role | idle-KV home |
|---|---|---|---|
| `baseline_cpu_ssd` | 4x H100 | 4x H100 | D-HBM -> CPU (512 GB) -> SSD, pure LRU |
| `hbf_prefill_p4d4` | 4x HBF+LPDDR cards | 4x H100 | D-HBM cache over P-side HBF (5.12 TB) -> SSD |
| `oracle_infinite_hbm` | 4x H100 | 4x H100 | unbounded HBM (memory bound only; admits everything, so it is not a scheduling bound past saturation) |

Both real systems share the same SSDs (8x 3.84 TB), NVLink handoff
fabric, decode hardware, and engine limits.  The prefill cards are
H100-class dies with HBF (weights + committed KV, HBM-class reads) and
wide LPDDR (activations) in place of HBM, on the same NVLink fabric.
System model: `serving/core/gpu_pd_hbf_prefill.py`.

## Population model

Sessions arrive Poisson from the family pool and leave; Little's Law
sets the resident population `L = rate x W`, which is installed at t=0
via `preload_session` in LRU order (D, then HBF/CPU, then SSD).  The
resident snapshot samples `(session, gap)` pairs jointly with
probability proportional to gap duration -- the steady-state law -- and
revives each resident at a uniform residual of its sampled gap.  The
gap-weighted resident context median is ~183k tokens (18 GB) for the
claude family with a tail to ~1M tokens; a uniform-template sampler
would understate this threefold (see `build_residents_steady`).

## Workload fidelity

Measured resume events reproduce the trace's per-call distributions:
gap buckets match ground truth within ±0.5 %p in every bucket for both
families and both passes, and context buckets within ±1 %p except the
claude long-window pass, which over-weights the >512k bucket (14.0%
vs 8.9%): inside a finite 8h window, arrivals only reach their
early-session (small-context) calls while gap-weighted residents start
late-session, an imbalance that vanishes in the infinite-window limit.
This does not bias the products: bucket-conditioned metrics are
invariant to the event mix, and the aggregate metrics are taken from
the rate-sweep pass, whose mix matches the trace within ±1 %p.

## Passes

* `hbf_prefill_v3/` -- rate sweep, call-count windows (~6k measured
  calls/cell): throughput, turn latency, SLO goodput, TCO, endurance.
* `hbf_prefill_v3_longwin/` -- 8-simulated-hour windows at
  sub-saturation rates (`--min-horizon-s 28800`): the gap- and
  context-conditioned resume TTFT product.  Long windows are what let
  multi-hour idle gaps complete inside the measurement; call-count
  windows are structurally blind to them.

Cells: `run_hbf_prefill_campaign.py`; aggregation:
`analyze_hbf_prefill.py` (aggregate / gap_buckets / context_buckets /
writes CSVs); economics: `economics_hbf_prefill.py` (per-server BOM
from `HardwareAnchors`, SSD endurance priced as replacement sets);
figures: `plot_hbf_prefill.py` -> `figures/<root>/`.

## Headline results (claude family, seed-meaned)

* Resume TTFT, 8h windows, rate 0.006: baseline pays SSD restores for
  every resume that fell out of D+CPU; ours reads the HBF home.
  `>12h`-idle bucket mean 4.0 s -> 1.75 s (2.3x), `4-12h` 5.9 -> 4.4 s,
  `1-4h` 9.4 -> 7.4 s.
* SLO goodput (ttft5/tpot100): +7 to +15% from rate 0.006 upward;
  goodput per TCO dollar-hour +15 to +32% (the hetero server is also
  ~12% cheaper: four HBM stacks replaced by HBF media + LPDDR).
* SSD endurance: baseline exceeds the 5-year fleet TBW budget from
  rate 0.006 (3.0x, reaching 9.3x at 0.016 -- a ~6.5-month fleet
  life); ours stays at 0 until its own HBF capacity spills (~0.012,
  1.8x at 0.016).
* codex family: contexts cap at ~230k tokens, so per-resume restore
  savings are smaller than the HBF-card prefill tax except in the
  deepest idle buckets; the economics still favor the hetero server
  (~+11% goodput/$) on capex and endurance alone.

## Why aggregate resume TTFT moves less than the capacity ratio

The phase decomposition (claude, 8h windows) shows what resume TTFT is
made of: at sub-saturation rates it is 73-96% suffix-prefill queue and
compute (large fresh inputs over 100k+ contexts, paid identically by
both systems), and at the knee (0.006) it splits ~49% D-slot admission
wait / ~47% prefill / ~4% restore transfer.  The two dominant terms
live on the decode slots and the prefill compute, which the systems
share; the restore term the HBF capacity eliminates is small in the
aggregate because CPU and retained-D absorb ~92% of baseline resumes
at these rates.  The capacity payoff is therefore concentrated exactly
where the SSD path is touched -- the deep-idle buckets (2.3-3.1x at
rate 0.004-0.006) -- plus the endurance and TCO columns, where the
writes happen regardless of how well reads are hidden.

A restore-execution sensitivity (`LLMSIM_BASELINE_RESTORE_MODE=bulk`,
`hbf_prefill_v3_bulkrestore/`) replaces the baseline's layerwise
restore streaming with fully exposed bulk transfers: the aggregate
moves only 2.85 -> 2.92 s and the deep buckets stay within noise, so
the headline comparison does not hinge on granting the baseline
perfect streaming.

## Handoff-deferred D reservation (LLMSIM_D_RESERVATION=handoff)

`D_RESERVATION_HANDOFF_DEFERRED` admits a resume on P capacity alone
and claims the decode-side reservation at the P-to-D handoff gate
(first-fit release with a 5 s head-aging bound, plus an admission
backstop while 16 finished prefills are parked).  Verdict from the
claude A/B (`hbf_prefill_v3_dhandoff*/`):

* At the knee (0.006) it is a strict win for the hetero server:
  resume p95 9.6 -> 6.8 s, every idle-gap bucket improves 20-40%
  (`4-12h` 4.3 -> 2.1 s, `>12h` 1.75 -> 1.26 s), throughput and turn
  par.  The baseline also improves, but less -- its 60 GB/rank P-HBM
  is the very backpressure the policy relaxes.
* At deep saturation the asymmetry inverts: the baseline keeps a
  strict win (0.032: resume p50 8.7 -> 2.6 s at full throughput)
  while the capacity-rich P role floods its own prefill worker and
  the fixed park cap oscillates (0.032: turn 46 -> 90 s, throughput
  -42%).  A drain-rate-aware admission controller, not a fixed cap,
  is the follow-up lever; until then the deferred policy is
  recommended only up to the goodput-optimal band (<= 0.016), which
  is where a provider would operate anyway.

Best-policy-per-system, this scheduling lever rewrites the headline
economics: the upfront-vs-upfront +32% goodput-per-dollar at
0.012-0.016 becomes ~+8% (ours 253.3 good tok/s at $282k vs baseline
278.7 at ~$334k with SSD replacements), and at 0.032 the baseline
under deferred-D wins goodput per dollar outright (315.4 vs 163.1) --
though ours' number there carries the unfinished-controller artifact,
and the deep-idle buckets (>12h 1.26 s vs 3.85 s), the knee band, the
endurance envelope (0 vs 9x budget), and the -12% capex remain
policy-independent hetero advantages.

Raw generation throughput is decode-bound and essentially identical
across the two real systems at every measured rate (claude 0.032:
575 vs 568 tok/s; codex 0.016: 497 vs 517 tok/s).  At deep saturation
both spill to the shared SSDs -- the resident population reaches
~12.6 TB at claude 0.032 against a 5.36 TB HBF home -- so the hetero
server's advantage there is not throughput but the SLO-goodput,
endurance, and TCO columns.  The oracle stops being a meaningful
reference past the knee: it admits everything, floods decode with
whale contexts, and loses throughput to TPOT inflation.

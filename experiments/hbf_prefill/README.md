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

Raw generation throughput is decode-bound and identical across systems
until deep saturation, where the baseline's restore gating on the
shared SSDs starts to cost it throughput as well.

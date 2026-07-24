# TraceLab schema-3 P/D capacity-only replay

> Historical artifact: this directory records the former
> `async-decode-join`, 512-GiB, no-DRAM-bounce direct-restore interpretation.
> It is not the current headline contract. See
> `../tracelab-schema3-pd-pre-admission-dgx-sensitivity/` for the corrected
> request-local pre-admission and staged-SSD result.

This directory is the compact, reproducible summary of the TraceLab
capacity-pressure replay for three cold-session KV baselines:

1. **HBM-LRU-Recompute:** decode-HBM pressure drops the LRU idle object; a
   later miss recomputes the reusable prefix.
2. **HBM-SSD-Direct:** decode-HBM pressure atomically demotes the LRU object
   directly to SSD; restore is SSD to decode HBM without a CPU-DRAM cache or
   bounce path.
3. **HBM-CPU-SSD:** decode-HBM LRU demotes to CPU DRAM, CPU LRU demotes to SSD,
   and SSD LRU drops to recomputation.

Every result in this directory uses capacity-only demotion. TTL actions are
disabled. Swap-out consumes transfer resources but never blocks a model
engine. Swap-in starts at the returning request's ready event, may overlap
analytical fresh-suffix prefill, and forms a request-local barrier before the
final prompt token/decode boundary.

## Primary placement result

The primary point reserves 8,000,000,000 bytes, or 10% of each marketed 80 GB
H100, for non-weight runtime state. The denominator is every TraceLab LLM
invocation, including first calls, zero-reuse calls, and calls above the target
model's context limit.

| Baseline | D-HBM | CPU | SSD | Recompute |
|---|---:|---:|---:|---:|
| HBM-LRU-Recompute | 69,114 (19.3509%) | 0 | 0 | 107,269 (30.0338%) |
| HBM-SSD-Direct | 56,449 (15.8049%) | 0 | 119,934 (33.5798%) | 0 |
| HBM-CPU-SSD | 53,948 (15.1047%) | 1,750 (0.4900%) | 120,685 (33.7901%) | 0 |

The literal all-request denominator is 357,161. The three-tier lower-tier
total is 122,435 resumes, or 34.2801% of all calls. CPU alone is only 0.4900%:
the 512 GiB shared CPU budget holds few objects at this trace's context sizes,
so the three-tier result is primarily an SSD result. Do not tune CPU capacity
merely to force a desired percentage; report a measured capacity or a labeled
sensitivity.

There are 176,383 context-admissible reuse-eligible transitions. On that
denominator, HBM-only recomputes 60.8160%, direct SSD restores 67.9956%, and
three-tier restores 0.9922% from CPU plus 68.4216% from SSD.

## Context and denominator audit

Of 357,161 source calls, 167,718 exceed Llama-3.1-70B's 131,072-token context
limit, leaving 189,443 context-admissible calls. The unfiltered source token
counts have input p50/p90/p99 of 124,018/256,767/822,888 and a maximum of
999,888 tokens. Only 53.0413% fit 128K. Within the admissible subset,
p50/p90/p99 are 74,166/118,842/129,829.

The primary all-call rate and the context-admissible secondary rate answer
different questions:

| Baseline | CPU / admissible | SSD / admissible | Recompute / admissible |
|---|---:|---:|---:|
| HBM-LRU-Recompute | 0% | 0% | 56.6234% |
| HBM-SSD-Direct | 0% | 63.3088% | 0% |
| HBM-CPU-SSD | 0.9238% | 63.7052% | 0% |

TraceLab is therefore not a short-context workload. Its prompts are sanitized,
so these source-token counts cannot be retokenized for Llama; reuse remains an
explicitly labeled adjacent-round estimate rather than a target-tokenizer-exact
LCP.

## Human and tool returns

TraceLab stores the pause from call *N* to call *N+1* on call *N*. Conversion
copies that label forward so the source is assigned to the request that is
actually returning. `tool_duration_ns` is a legacy field name and can encode a
`tool`, `human`, `mixed`, or `unknown` completion-to-request-ready gap.

| Return class | All / reuse-eligible | HBM-only: HBM / recompute | Direct: HBM / SSD | Tiered: HBM / CPU / SSD |
|---|---:|---:|---:|---:|
| Human | 31,997 / 10,468 | 2,516 / 7,952 | 2,255 / 8,213 | 2,010 / 402 / 8,056 |
| Mixed | 3,097 / 1,165 | 357 / 808 | 306 / 859 | 283 / 51 / 831 |
| Tool | 313,179 / 164,750 | 66,241 / 98,509 | 53,888 / 110,862 | 51,655 / 1,297 / 111,798 |

Among reuse-eligible returns, the cold fraction is 75.96% for human versus
59.79% for tool under HBM-only, 78.46% versus 67.29% under direct SSD, and
80.80% versus 68.65% under three-tier. Exact all-class and eligible-class
fractions are in `primary_resume_by_return.csv`.

## Capacity sensitivity

Reserve is a physical sensitivity, not a knob selected to reach a target
resume rate.

| Reserve | HBM-only recompute / all | Direct SSD / all | Tiered CPU / all | Tiered SSD / all | Tiered CPU+SSD / all |
|---:|---:|---:|---:|---:|---:|
| 0% | 22.3356% | 33.3399% | 0.5017% | 33.3575% | 33.8592% |
| 10% primary | 30.0338% | 33.5798% | 0.4900% | 33.7901% | 34.2801% |
| 20% | 31.7224% | 33.9626% | 0.4793% | 33.9880% | 34.4674% |
| 30% | 34.4338% | 34.5351% | 0.5969% | 34.3128% | 34.9097% |

At 30%, three-tier also drops 21 transitions, or 0.0059% of all requests,
after exhausting the configured SSD capacity. `summary.csv` contains exact
counts, fractions, capacities, and report hashes for all 12 runs.

## Asynchronous restore overhead

The default execution mode is `async-decode-join`. Restore starts at the
request-ready event: the final tool-result boundary for a tool return, the
user-message boundary for a human return, and the latest required input for a
mixed return. It is never backdated into an elapsed tool or human gap.

The timing identity is exact:

```text
raw restore = fresh-prefill hidden + owner decode barrier
              + other concurrent/admission
```

The final component is mainly logical-ready-to-P-admission time; it is causal
request service delay but is not mislabeled as a decode-join barrier.

| Baseline | Raw restore sum | Hidden by suffix prefill | Owner decode barrier | Other concurrent/admission | Recompute / executed prompt compute |
|---|---:|---:|---:|---:|---:|
| HBM-LRU-Recompute | 4,934.614 s | 2,226.916 s | 2,707.610 s | 0.089 s | 89.6721% |
| HBM-SSD-Direct | 28,771,360.638 s | 15,160.083 s | 1,022,500.116 s | 27,733,700.440 s | 0% |
| HBM-CPU-SSD | 77,408,198.506 s | 15,289.298 s | 1,972,149.784 s | 75,420,759.423 s | 0% |

These are request sums and may exceed wall-clock trace length. The source-level
mean makes the cold paths easier to read:

| Baseline/source | Timing events | Mean raw | Mean hidden | Mean owner barrier | Mean other/admission |
|---|---:|---:|---:|---:|---:|
| Direct SSD | 123,241 | 233.410 s | 0.107 s | 8.267 s | 225.036 s |
| Tiered CPU | 1,782 | 5.521 s | 0.077 s | 3.499 s | 1.945 s |
| Tiered SSD | 123,782 | 625.240 s | 0.108 s | 15.857 s | 609.275 s |

Fresh-prefill overlap is implemented for both human and tool returns, but it
hides little of the cold SSD raw interval at this load: full-object demotions
occupy transfer resources and D admission queues much longer than the fresh
suffix computation. `primary_restore_by_return_source.csv` preserves the
human/tool/mixed and HBM/CPU/SSD cross-tab, including sums and means.

The direct run issues 2.860 PB of SSD writes and 2.812 PB of reads; three-tier
issues 2.875 PB and 2.840 PB. Direct transfer jobs sum to 96,089.807 s of
service and 840,142.086 s of queue wait. Three-tier sums to 125,676.931 s and
741,308.823 s. The canonical D-admission wait additionally includes branch
FCFS and source-transit time, not just physical HBM shortage; it is exposed as
`aggregate_admission_wait_seconds` and must not be called pure capacity wait.

## Paired infinite-HBM residency reference

The paired reference gives P and D nonbinding HBM capacity while preserving
the same workload, analytical prompt model, P/D link, mandatory D-to-P hits,
P-to-D handoffs, return gaps, and restore execution mode. The primary service
denominator sums each call's logical-ready-to-completion interval, excluding
the fixed gap itself while preserving closed-loop propagation.

| Baseline | Finite / oracle service sum | Slowdown | Delta p90 / p99 | Session-E2E slowdown | Full-trace makespan slowdown |
|---|---:|---:|---:|---:|---:|
| HBM-LRU-Recompute | 6,220,667.407 / 53,626.527 s | +11,499.982% | 59.551 / 119.372 s | +5.3847% | +0.05379% |
| HBM-SSD-Direct | 30,170,177.249 / 53,626.527 s | +56,159.801% | 349.658 / 447.367 s | +26.2959% | +0.44261% |
| HBM-CPU-SSD | 81,038,371.340 / 53,626.527 s | +151,016.203% | 911.707 / 1,122.640 s | +70.7107% | +1.16447% |

The old approximately 0.23% number was a wall-clock interval-union occupancy,
not a request latency slowdown. The paired service result shows that waits are
large under this pressure point. Conversely, the full-trace makespan is
diluted by multi-day gaps and cross-session overlap.

The residency reference is not a strict per-call lower bound: preserving a hit
can send more D-to-P traffic than a finite run that recomputes, changing shared
fabric ordering. Aggregate service sums are the authoritative comparison;
per-call signs may differ. Exact hashes and source/return cross-tabs are in the
three `primary_infinite_hbm_*.csv` files.
The finite-service source cross-tab classifies every call, whereas
`primary_resume_by_return.csv` counts only reuse-eligible transition decisions;
their source counts are intentionally not interchangeable.

## The TraceLab replay can be idle

This is not an always-backlogged workload. Only each session's first call is
injected by a synthetic Poisson process at 0.2 sessions/s; subsequent calls are
released closed-loop after the preceding completion plus the preserved
human/tool gap. The 4,281 first arrivals span about 21,624 s, while the longest
dependency chain produces an approximately 70.6-day drain tail.

The full-window complement with no active analytical call is 99.0942% for
HBM-only, 98.7578% for direct SSD, and 97.8990% for three-tier. This field is
explicitly `is_server_utilization=false`: calls overlap without a shared
compute queue or batcher, so it is offered-call activity, not GPU utilization.

A paper should therefore report two separate panels:

1. Trace-faithful dormant-cache replay with original gaps and request-summed
   service metrics.
2. A controlled-load or saturated sweep, with warm-up/drain excluded, for
   queueing, batching, and throughput claims.

## Serial sensitivity

`serial-before-prefill` disables suffix overlap for the returning request. It
is a sensitivity, not the headline mode.

| Baseline | Async exposed | Serial exposed | Serial lower-tier/recompute rate |
|---|---:|---:|---:|
| HBM-LRU-Recompute | 2,707.610 s | 4,949.424 s | recompute 29.9646% / all |
| HBM-SSD-Direct | 1,022,500.116 s | 1,023,865.852 s | SSD 33.5728% / all |
| HBM-CPU-SSD | 1,972,149.784 s | 1,977,746.449 s | CPU+SSD 34.3254% / all |

The finite and oracle runs are closed-loop, so changing completion epochs also
changes later placement and queue order. The serial aggregate is consequently
not guaranteed to be a monotone end-to-end upper bound. Use
`serial_sensitivity.csv` for provenance and use a fixed-placement paired run
when isolating overlap alone.

## Relation to InferCept

InferCept's often-cited 37--40% recomputation value is a fraction of total
model forwarding time, while its greater-than-25% naive-swap wait is a fraction
of total workload time under a synchronous foreground swap. This artifact's
89.6721% is an analytical prompt-compute-only fraction; decode and calibrated
kernels are absent. Its async raw/hidden/exposed decomposition is also not the
same denominator as naive synchronous swap. These values should not be forced
to match.

TraceLab is much longer-context than InferCept's reported 753--2,185-token
mean contexts. Workload, denominator, capacity, and execution semantics must be
matched before using the older percentages as validation targets.

## Trace provenance

- Source: TraceLab v0.0.1 `syfi_coding_trace.jsonl.gz`
- Release: https://github.com/uw-syfi/TraceLab/releases/tag/v0.0.1
- Raw gzip SHA-256:
  `9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b`
- Converted schema: 3
- Converted JSONL SHA-256:
  `6721238648d49236d692d70ad72ee3666e829ee5530be175644041fe016176dc`
- Conversion manifest SHA-256:
  `f1069ffa477b6c56395c1934ace50d354fcaff91907e48a23123689688294d69`
- Population: 4,281 sessions and 357,161 LLM calls
- First-session arrivals: Poisson 0.2 sessions/s, seed 42
- Gap boundary: previous completion to next request-ready event; tool-union is
  a labeled fallback only
- Reuse: policy-independent eligible estimate

The conversion audit has 316,371 exact event-boundary gaps, 31,885 tool-union
fallbacks, 4,607 unmeasurable zero gaps, 17 negative gaps clamped to zero,
7,867 context-shrink lineage breaks, 32 disambiguated source-ID collisions,
and zero validation errors. Observed provider hits are preserved separately
from policy-independent reuse. Ordering uses provider/project/file/session,
round index, and ingest sequence rather than `session_id` alone. Context shrink
and explicit compaction/reset markers break lineage.

Recreate the workload:

```bash
python -m workloads.generators agent-traces \
  --format tracelab \
  --source /tmp/syfi_coding_trace.jsonl.gz \
  --source-revision v0.0.1 \
  --output /tmp/tracelab-schema3-sps0.2.jsonl \
  --sps 0.2 --seed 42 \
  --tool-wait-mode union \
  --tracelab-reuse-mode eligible
```

## Replay assumptions and command

- Model: `meta-llama/Llama-3.1-70B`
- Hardware: H100, 80 decimal GB per rank
- Topology: one TP8 P instance plus one TP8 D instance
- BF16 KV, 16-token blocks, 2,048-token analytical prefill chunks
- Architecture-derived weight estimate: 17,640,734,720 bytes/rank
- Primary KV budget: 54,359,265,280 bytes/rank after weights and reserve
- Shared CPU KV budget: 512 GiB
- SSD KV budget: 30.72 decimal TB
- PCIe: 50 GB/s/rank; aggregate CPU DRAM: 200 GB/s
- SSD: 112 GB/s read and 48 GB/s write aggregate
- P/D: 50 GB/s/rank plus 10 us fixed latency
- Transfer scheduling: non-preemptive gang FCFS
- P and D admission: branch-local common-age FCFS, complete final D footprint
  pre-reservation, source pinning only at the D head, and a completion-safe
  zero-D-growth HBM backfill exception

The P branch may admit and execute analytical suffix work while the D branch
waits. Lower-tier restore first reaches D; D-to-P begins only after the prefix
and P branch are ready. Prompt completion never makes a late D capacity request.

```bash
python -m serving.agentic_kv_capacity_replay \
  --workload /tmp/tracelab-schema3-sps0.2.jsonl \
  --model meta-llama/Llama-3.1-70B \
  --hardware H100 --tp-size 8 --block-size 16 \
  --max-context-tokens 131072 \
  --hbm-capacity-gb-per-rank 80 \
  --hbm-static-reserve-gib-per-rank 7.450580596923828 \
  --cpu-kv-budget-gib 512 --ssd-kv-budget-tb 30.72 \
  --policy POLICY --demotion-mode capacity-only \
  --cpu-rank-gbps 50 --cpu-aggregate-gbps 200 \
  --ssd-read-gbps 112 --ssd-write-gbps 48 \
  --pd-link-gbps-per-rank 50 --pd-fixed-latency-us 10 \
  --restore-execution-mode async-decode-join \
  --compare-infinite-hbm-oracle \
  --output /tmp/tracelab-capacity-POLICY.json
```

`POLICY` is `hbm_lru_recompute`, `hbm_ssd_direct`, or `tiered`.

## Code and validation provenance

- Request-local asynchronous restore join: `296ea19`
- Independent P/D branch admission and full D reservation: `a073ca9`
- Cross-branch waiter lifecycle fix and admission metric correction: `48096bf`
- Capacity report schema: 8
- Online metrics schema: 9

Validation:

```text
python -m py_compile serving/core/agentic_kv_capacity_replay.py \
  serving/agentic_kv_capacity_replay.py tests/test_agentic_kv_capacity_replay.py
PYTHONPATH=. pytest -q tests/test_agentic_kv_capacity_replay.py  # 30 passed
PYTHONPATH=. pytest -q tests                                    # 235 passed
./scripts/cold-kv-pressure-smoke.sh                              # PASS
```

## Limitations

This is a global analytical capacity and transfer-queue replay, not a
cycle-level throughput, TTFT, or TPOT result. Prompt compute uses an H100
roofline. Decode compute, decode-active residence, batching, and the LLM
compute-server queue are absent. It cannot measure a mixed-batch compute-shape
penalty, and the repository has no measured current-profiler H100 TP8 profile
for this run.

The online cycle path defaults to request-local asynchronous restore. It does
not freeze a selected batch before DMA or put a restore barrier inside a batch;
unrelated HBM work remains dispatchable. Its migration queue and ASTRA-Sim's
normal P-to-D graph traffic are still separate contention domains rather than
one unified fabric/copy-engine scheduler.

The whole-suffix overlap is a coarse optimistic bound. A physical
implementation needs layerwise KV streaming and calibrated kernel dependencies.
`serial-before-prefill` is the analytical no-overlap sensitivity.

The direct SSD path is an analytical no-DRAM path, not a measured GDS or Weka
deployment. Bandwidth, fixed latency, weight residency, runtime reserve,
filesystem, queue depth, and object-size service curves require calibration on
the paper's target system before using absolute latency as a hardware claim.

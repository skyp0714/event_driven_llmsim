---
sidebar_position: 10
title: Cold-session KV online evaluation handoff
---

# Cold-session KV online evaluation handoff

This page is the engineering and experimental handoff for the cold-session KV
cache study. It is intentionally more explicit than a normal user guide: a new
machine and a new Codex session should be able to reconstruct the design,
verify the completed debug work, and run the paper suite on a suitably
provisioned host without relying on chat history.

## Status at handoff

The active branch is `codex/cold-kv-hardening`. The finite-HBM active-prefill
P/D reclaim path is `565defe`; the graph-commit retry dependency that closes
the remaining online liveness hole is `3f6e79e`; reporting-schema hardening is
`b60f306`; and the disk-bounded high-rate smoke contract is `708f107`. The
expanded pressure grids are `e2d53e0` and `9ff11fb`, while the host-concurrency
pilot is `8639826`. These are separate commits so scheduler policy, reporting,
debug workload size, and launcher isolation remain independently reviewable.

The following status distinction is mandatory:

- The trace conversion, four policy definitions, strict residency oracle,
  H100 analytical latency provider, P4+D4 online path, fixed-K Poisson runner,
  common-random-number validation, operational metrics, SLO calibration, and
  expanded rate grids are committed.
- Real online runs exposed two related finite-HBM liveness defects: missing
  active-partial-prefill reclamation and a stale retry key after a frozen graph
  commits. Both are fixed, independently reviewed, and exercised by the
  bounded high-rate online pair. The reporting commit adds request-level
  event-to-JSON-to-CSV conservation without changing victim selection.
- The complete test suite, exact bounded high-rate smoke, and 4-versus-8
  host-concurrency pilot all pass. Their commands, execution IDs, wall times,
  hashes, and accounting checks are recorded below.
- The user explicitly requested debugging only on the current machine. The
  32-cell discovery and 90-cell main suite have **not** been run on the final
  code. They are for the destination machine; none of the debug outputs below
  is a paper result.

The worktree may contain unrelated untracked profiler/calibration files and
historical results. They belong to other work. Do not stage, delete, or rewrite
them while finishing this study.

The source machine has Node 22, `/usr/bin/corepack`, and a populated
`docs/node_modules`. The final documentation validation command is
`(cd docs && ./node_modules/.bin/docusaurus build)`; a clean destination can
instead run `(cd docs && corepack pnpm install --frozen-lockfile && corepack
pnpm build)`.

## Research question and claim boundary

The primary question is:

> For long, temporarily dormant agent sessions, how much online serving
> performance is lost when reusable KV does not remain in HBM, and how do
> recomputation, direct SSD persistence, conventional CPU/SSD tiering, and a
> severe-queue partial-recompute fallback compare when all capacity,
> communication, admission, queueing, batching, and P/D dependencies are
> causal?

The desired qualitative pressure-region ordering is, for higher-is-better
throughput,

```text
HBM-LRU-Recompute < HBM-CPU-SSD < HBM-CPU-SSD-Partial < Infinite-HBM
```

or the reverse for server-added JCT. This is a hypothesis and a discovery-rate
selection criterion, not a license to tune hardware parameters or discard
contrary data. HBM-SSD-Direct is also a headline baseline and must appear in
the main panel even though it is not part of that particular four-way ordering.

The central paper claim must remain a **matched-workload relative policy
comparison under one fixed analytical calibration band**. It is not an
absolute measured Qwen3-on-H100 serving result. The one-million-token workload
is a synthetic length sensitivity, and the implemented attention path is a
dense/full-attention analytical extrapolation rather than Qwen's official
sparse DCA kernel.

## Headline systems

All finite systems use capacity-only demotion, whole-session LRU ordering with
`(last_access_ns, session_id)` as the deterministic key, asynchronous swap-out,
request-local asynchronous swap-in, the same active-preemption mode, the same
hardware inputs, and the same online compute provider.

| Label | Config | Exact behavior |
| --- | --- | --- |
| `hbm_lru_recompute` | `configs/agentic_kv/qwen3_1m_p4d4/hbm_lru_recompute.json` | Keep dormant KV in HBM. Under HBM pressure, discard least-recently-used idle KV objects. A later reusable-prefix miss recomputes the lost prefix. There is no CPU or SSD cache. The logical session and its active-session slot are never dropped. |
| `hbm_ssd_direct` | `configs/agentic_kv/qwen3_1m_p4d4/hbm_ssd_direct.json` | Apply the same HBM LRU pressure rule, demote directly to SSD, and retain no persistent CPU cache object. Restore is still physically SSD to transient DRAM followed by DRAM to P-HBM; “direct” means no CPU cache tier, not that reads bypass host memory. |
| `hbm_cpu_ssd` | `configs/agentic_kv/qwen3_1m_p4d4/tiered.json` | Demote HBM LRU to CPU DRAM, CPU LRU to SSD, and terminal SSD LRU to loss/recomputation if SSD is full. CPU/SSD hits restore to P-HBM before the request becomes compute-eligible. |
| `hbm_cpu_ssd_queue_recompute` | `configs/agentic_kv/qwen3_1m_p4d4/tiered_queue_recompute.json` | Use the same hierarchy as `hbm_cpu_ssd`. Only when the immutable projected admission/transfer queue wait of a full CPU/SSD restore exceeds four times isolated service, compare full, zero, and feasible block-aligned prefix restores. Cost is projected prefix restore plus `1.25 *` the same online provider's singleton suffix-recompute COMP estimate. Ties retain more KV, and a modified choice must strictly improve the projected path. |
| `infinite_hbm_oracle` | Runtime flag `--strict-infinite-hbm-oracle` with the tiered config | Replace finite HBM with a proven nonbinding per-rank bound: weights plus two block-rounded copies of every selected call plus one safety block. Every reusable continuation must be an HBM hit and all demotion, lower-tier-hit, capacity-loss, and avoidable-recompute counters must remain zero. |

The oracle is strict about **residency**, not an impossible zero-latency JCT
oracle. It retains the same session arrivals, fixed active-session cap, P/D
handoffs, compute, collectives, scheduler queueing, continuous batching, and
closed-loop dependencies. Therefore it is the matched residency-nonbinding
reference for the incremental cost of finite KV capacity, but it can still
have substantial TTFT and JCT. It is an expected best case under the fixed
scheduler, not a mathematical throughput upper bound: finite-cache timing can
also alter batch composition and nonlinear kernel efficiency.

The partial policy never drops a logical session. `H=0` means that the reusable
KV object is released and its prefix is recomputed when the already-admitted
session resumes. Every raw `event="drop"` must be interpreted through
`object_scope`, `drop_class`, and `logical_session_effect`; it is not a session
admission drop.

## Why TraceLab is the primary trace

TraceLab v0.0.1 is the primary source because it supplies the combination this
study needs: real multi-call coding-agent sessions, source token accounting,
long contexts, model-output and next-input event timestamps, and enough event
types to distinguish tool, human, and mixed returns. Those fields let the
runtime preserve dormant intervals and release continuations causally.

LMCache is still useful as a sensitivity when cumulative messages can be
retokenized and exact longest-common-prefix reuse is the main concern. Its
`pre_gap` convention is shifted onto the preceding call by the converter, but
it does not provide the same TraceLab event-boundary and return-class contract.
Do not merge TraceLab and LMCache rows into a headline cohort: their reuse and
timing provenance differ. A Weka or storage trace is not a substitute unless it
also exposes request dependency, reusable-prefix lineage, prompt/output token
counts, and request-ready boundaries. A later trace survey should evaluate
those fields explicitly rather than choosing a dataset because it contains
storage activity.

### Content-addressed TraceLab artifacts

The source and converted files are external because the converted JSONL is
about 230 MB. The experiment specifications bind them by content:

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| Raw TraceLab v0.0.1 `syfi_coding_trace.jsonl.gz` | 53,601,226 bytes | `9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b` |
| Converted schema-3 `tracelab-schema3-sps0.2-final.jsonl` | 230,503,348 bytes | `b6188582aac9467cee8c73e4275f9a9606b359f8c2fa000d9f49a9ca3bde02f0` |
| Companion manifest | 4,875 bytes | `e45575b40c204df259ad8535a2022ccce7d5c09f9265303d14040d47e0cc1e3e` |

The source contains 357,161 LLM rounds in 4,281 sessions. The converter was run
with TraceLab `eligible` reuse mode, source revision `v0.0.1`, Poisson metadata
rate `0.2`, seed `42`, union fallback for tool intervals, and no target
tokenizer because sanitized prompt text is unavailable. Its manifest records
converter commit `42a84b9ccff12a52ebfafa5f900e0d28e3015a8f` and generator
module SHA-256
`679427d03da64ac69de20a3fae6c6fcfe78a6b228a76d9406880a4306e98be26`.
The manifest does not record an authoritative public download URL or license;
copy the bound raw or converted artifacts to the destination instead of
inventing acquisition provenance.

A reproduction with the restored raw file is:

```bash
python -m workloads.generators agent-traces \
  --format tracelab \
  --source /data/syfi_coding_trace.jsonl.gz \
  --source-revision v0.0.1 \
  --output /data/tracelab-schema3-sps0.2-final.jsonl \
  --manifest-output /data/tracelab-schema3-sps0.2-final.jsonl.manifest.json \
  --sps 0.2 \
  --seed 42 \
  --tool-wait-mode union \
  --tracelab-reuse-mode eligible
```

The regenerated JSONL should match its bound hash when the source bytes,
converter bytes, and arguments match. The manifest embeds absolute source and
output paths (`/tmp/...` in the bound sidecar), so regeneration under `/data`
cannot be byte-identical to the original sidecar even when its semantic content
is otherwise identical. Preserve and verify the copied original sidecar by its
published hash; for a newly generated sidecar compare all semantic fields while
explicitly allowing only the two path fields to differ. Never change the
checked-in expected hashes merely to make a run pass.

### Conversion semantics that must not regress

1. Rows are grouped by the public surrogate identity
   `(provider, project, session_file, session_id)`, not by `session_id` alone.
   SQLite spooling reunites non-contiguous rows while preserving source ingest
   order and disambiguates raw-ID collisions.
2. Rounds are ordered by `(round_index, source_ingest_seq)`. Selection later
   restores source surrogate order rather than ranking traversal order.
3. The outgoing gap on call `N` is the entire interval from the last LLM model
   output event of `N` to the latest request-ready input event of `N+1`.
4. A next input composed only of `tool_result` is labeled `tool`; only
   `user_message` is `human`; both are `mixed`. Missing boundaries fall back
   to the previous round's union/max tool interval with explicit provenance.
5. `observed_provider_hit_toks` is the incumbent provider's observed outcome.
   It is retained for sensitivity only. `policy_independent_reuse_toks` is the
   adjacent-lineage counterfactual and drives the `eligible` workload. Feeding
   observed hits into the proposed cache policy would make the evaluation
   circular.
6. A nonadjacent round, context shrink, explicit compaction, or explicit
   context reset breaks lineage and makes reusable tokens zero.
7. Zero append/output source values are preserved in raw provenance but are
   promoted to one only where the executable request schema requires a
   positive value; every promotion is counted in the manifest.

The current raw TraceLab manifest has 31,997 human, 313,179 tool, 3,097 mixed,
and 4,607 unknown outgoing transitions, plus 4,281 session ends. Event-boundary
timing is available for 316,371 transitions; 31,885 use tool-union fallback,
17 negative event-boundary gaps are zeroed, and 4,607 unmeasurable gaps are
explicitly zeroed. Additional audited warnings are 12 missing tool latencies,
32 raw session-ID collision disambiguations, 12 zero-append promotions, 32
zero-output promotions, and 7,867 context-shrink lineage resets. These warnings
are part of the dataset limitation and must be reported, not hidden.

## One-million-token workload transform

The source distribution itself is not relabeled as a one-million-token trace.
For each selected cohort, the online runner derives one cohort-global rational
scale factor from that cohort's maximum sequence and applies it to
prompt-prefix coordinates. It floors scaled prompt coordinates, keeps
generated output suffixes unchanged, maps reuse through the transformed
predecessor prompt plus its unchanged output, and caps reuse by both
predecessor KV and current input. It then derives
`newly_append_toks = input_toks - prefix_reuse_toks`.

The main-long cohort uses source surrogate indices
`[487, 488, 1759, 1836, 1902, 2021, 2047, 3791]`. It contains eight complete
templates and 24 calls per repetition and uses factor `999999 / 98746`. The
realized maximum sequence length is exactly 1,000,000 tokens; session ordering,
gap durations/classes, provider observations, and output counts remain
unchanged. The selected identity hash is
`4dded4375d266cda87c46a7e9c10633e1f7f60ff88c431d10d65c7c67677be58`.

The discovery cohort is disjoint and uses indices `[2113, 3726]`, two complete
templates, and four calls per repetition. It uses factor `999980 / 27607` and
also realizes a maximum sequence length of exactly 1,000,000 tokens. Its
selected identity hash is
`c458507d7c117a0741630783eeb5dcbdd6ab1cc57b56c78dc0ed94b1f06f8fbf`.

Never slice a session or truncate its call chain to reduce runtime. Reduce the
number of complete templates or repetitions in a separately labeled debug
spec instead.

## Model, calibration, and hardware contract

### P4+D4 Qwen3 configuration

The online cluster is
`configs/cluster/single_node_qwen3_1m_pd_p4d4_h100.json`:

- one prefill instance with TP4/EP4 and one decode instance with TP4/EP4;
  there are eight H100 ranks total, not a single decode TP8 instance;
- Qwen3-30B-A3B-Instruct-2507, BF16 weights and BF16 KV, 48 layers, four KV
  heads, 128-dimensional heads, and 16-token KV blocks;
- runtime `max_model_len=1,010,000`, 131,072-token chunked-prefill budget,
  `max_num_seqs=32` on P and `128` on D;
- 80 GB SI HBM per rank, represented as `74.50580596923828` GiB, with
  exactly 19,893,012,480 bytes per rank reserved for non-weight runtime/static
  state before weight and KV admission;
- one node-shared 512 GB SI host-DRAM capacity, represented as
  `476.837158203125` GiB, at an effective aggregate 200 GB/s. This is shared by
  P and D rather than duplicated per instance;
- 50 GB/s effective CPU-to-GPU PCIe service per participating GPU/rank;
- eight NVMe devices with aggregate 55.2 GB/s read and 33.6 GB/s write service,
  derived from eight 6.9/4.2 GB/s device limits, plus 20 microseconds fixed
  read/write latency;
- 450 decimal GB/s per-rank direct P/D fabric and one microsecond fixed peer
  latency on the congestion-aware analytical ASTRA topology.

The policy JSON sets `ssd_num_devices=8` and `ssd_capacity_gb=3840`.
`AgenticKVConfig.ssd_capacity_bytes` treats that field as **per-device** and
multiplies it by the device count, so modeled aggregate capacity is 30.72 TB
SI. In contrast, the configured 55.2/33.6 GB/s rates are already aggregate
eight-device bandwidths and are not multiplied again. This matches eight
3.84-TB drives. Any experiment intending 3.84 TB total must set 480 GB per
device (or define a new explicit total-capacity contract) and rerun; it must
not reinterpret existing output after the fact. The 30.72-TB capacity also
means SSD-terminal eviction may be rare, so opportunity-region claims must
report observed SSD occupancy/evictions rather than assume the tier filled.

The 512 GB host setting is an intentional capacity-pressure sensitivity, not
the physical 2 TB host capacity of a stock DGX H100. The SSD rates are sums of
manufacturer sequential limits, not measured RAID or filesystem throughput.
Both facts belong in the paper limitations.

### H100 analytical calibration connected to online simulation

The cluster selects `h100-qwen3-tp4-kernel-calibrated`. This provider is wired
directly into online trace generation; the experiments do not use standalone
capacity replay and do not post-process compute time onto a finished trace.

The provider fits kernel-family residuals to legacy H100-labeled TP4 component
measurements for Llama-3.1-70B and Mixtral-8x7B. For each family it evaluates

```text
t_roof = max((F / P_peak) * u, bytes / BW_peak)
t = max(t_launch, eta * t_roof)
```

using 989.5 BF16 TFLOP/s, 3.35 TB/s HBM, and 132 SMs. The central band is the
median fitted `eta`; fast and slow sensitivities are p10 and p90. GEMM
occupancy is folded into `eta` because the legacy traces do not preserve the
selected cuBLAS tile/block geometry. Prefill attention uses a documented
`ceil(q/128) * local_q_heads` wave proxy and bottom-right causal pair count.

Layer families train on token shapes 1,025--1,536 with a contiguous
1,537--2,048 holdout. Prefill attention trains on query sizes 512--768 and
holds out 800--1,024. These are component interpolation checks. They do not
validate long-K attention, Qwen3 end to end, one-million-token attention,
serving overhead, or the exact H100 SKU/software environment.

The online provider evaluates the actual Qwen TP4/EP4 shapes and mixed online
batch geometry, including decode-attention shape fits, but supplies **COMP node
durations only**. It never includes TP/EP/P-to-D communication in COMP time.
Trace generation emits those communication nodes and ASTRA-Sim is the sole
communication authority. Recompute-time attribution uses the provider's exact
batch-level cache-hit counterfactual and critical path, not a rank-summed MoE
COMP value.

The official Qwen 1M config enables sparse DCA. The experiment explicitly
disables that sparse execution and labels the result
`dca_dense_full_attention_sensitivity`. Absolute TTFT, TPOT, and throughput are
therefore analytical sensitivities. A paper-quality absolute claim requires a
small serving-matched TP4 validation set on the target H100 stack, separated by
GEMM, attention, MoE, and collective family.

## Online execution semantics

Every experiment cell is a normal `python -m serving` execution driven by
Router, continuous-batching Scheduler, online trace generation, Chakra graph
generation, and ASTRA-Sim. `serving.online_experiments` only materializes
paired workloads, launches cells, validates reports, and plots them. It must
never call the standalone capacity replay for a paper result.

### Implementation ownership map

| Surface | Responsibility |
| --- | --- |
| `workloads/generators/agent_traces.py` | TraceLab/LMCache conversion, surrogate session identity and ordering, gap/return labels, policy-independent reuse, lineage breaks, and conversion audit manifest. |
| `serving/online_experiments.py` | Complete-session cohort selection and 1M scaling, fixed common-random-number workload generation, independent cell launch/isolation, fail-closed report validation, source CSVs, and plots. |
| `serving/core/session_admission.py` | Backlog and Poisson session-arrival configuration, fixed-K active-session admission, warmup/measurement scope, and full-drain controls. |
| `serving/core/router.py` | Dependency release, sorted request-ready queue, fixed P/D pair FIFO, foreground preparation, atomic paired chunk claims, and retry-state dependency tracking. |
| `serving/core/agentic_kv.py` | Authoritative HBM/CPU/SSD objects, whole-session LRU, async demotion/restore reservations, DRAM/PCIe/NVMe calendars, queue-recompute decisions, capacity conservation, and raw event metrics. |
| `serving/core/scheduler.py` | Continuous batching, incremental chunked prefill, P/D KV ownership and handoff, frozen graph boundaries, active-prefill liveness preemption, replay attribution, and reclaimability generation. |
| `serving/__main__.py` | Causal online event loop, Router/Scheduler/ASTRA integration, exact idle fast-forward, external-fabric callbacks, drain termination, and run-specific scratch cleanup. |
| `serving/core/h100_kernel_calibrated_prompt.py` | Legacy-H100 analytical family fits and online Qwen TP4/EP4 COMP-node prediction. |
| `serving/core/online_measurement.py` | Strict infinite-HBM oracle sizing/invariants, online compute accounting, and time-weighted HBM occupancy. |
| `serving/core/session_metrics.py` | Request/session lifecycle reconciliation, TTFT/TPOT/JCT/admission distributions, timing validation, and schemas 11/12 inputs. |
| `configs/agentic_kv/qwen3_1m_p4d4/` | The four finite policy contracts and all bandwidth/capacity/policy constants. |
| `configs/experiments/online_tracelab_qwen3_1m_p4d4_*.json` | Immutable cohort, rate, seed, K, SLO, policy-set, and opportunity contracts for debug, discovery, and main suites. |

The corresponding regression surfaces are
`tests/test_agentic_kv.py`, `tests/test_agentic_router.py`,
`tests/test_active_kv_preemption.py`,
`tests/test_pd_pending_launch_admission.py`,
`tests/test_online_experiments.py`, `tests/test_online_measurement.py`, and
`tests/test_session_metrics.py`. Start diagnosis at the owner in this table;
do not patch the standalone capacity replay to change online behavior.

### Continuous batching and request-local restore gates

Swap-out is background and asynchronous. It consumes the relevant PCIe, DRAM,
SSD, or fabric resource but does not install a global model-engine barrier.
The source remains authoritative until demotion commit; a return that races the
copy may cancel it, while already consumed queue/service work remains charged.

Swap-in follows these rules:

1. The continuation's observed request-ready event is the earliest issue
   epoch. A tool return starts preparation at its tool-result event. A human
   return starts at the observed user-message event and is never predicted or
   backdated into human think time. Mixed returns wait for their latest input.
2. Per-pair FIFO and a safe preparation boundary may delay issue. These are
   admission delays, not transfer service.
3. Destination P-HBM is logically reserved before foreground transfer. This
   reduces admission capacity while the bytes are in flight.
4. A CPU hit uses DRAM plus the P-side per-rank PCIe/copy engines. An SSD hit
   is strictly serial SSD-to-DRAM followed by DRAM-to-P-HBM; SSD reads never
   bypass DRAM in the headline model.
5. The returning request remains outside every compute batch until the whole
   selected prefix is ready. There is no same-owner prompt overlap in
   `async-pre-admission`.
6. Unrelated ready requests and already running batches continue. A restore is
   not inserted into an already formed batch. After readiness, the returning
   call can affect peers only through ordinary batching, token budgets,
   attention-shape effects, HBM occupancy, and shared-resource contention.

Host/storage migrations use deterministic gang FCFS. A CPU transfer's isolated
service is

```text
fixed_cpu + max(bytes_per_rank / BW_PCIe,
                bytes_cluster / BW_DRAM)
```

and an SSD restore is

```text
[fixed_ssd + max(bytes_cluster / BW_SSD_read,
                 bytes_cluster / BW_DRAM)]
+ [fixed_cpu + max(bytes_per_rank / BW_PCIe,
                   bytes_cluster / BW_DRAM)]
```

HBM-resident D-to-P cold restores use four exact rank lanes in the shared
congestion-aware ASTRA event queue. They contend with ordinary P-to-D and
TP/EP communication and with physical endpoints, but completion gates only
the returning owner. CPU/SSD calendars and ASTRA are complementary first-order
models; they are not one unified GPU-kernel/PCIe/NVMe resource simulator. This
separation is a known limitation for deployment-level tail-latency claims.

### P/D allocation and handoff

Prompt KV is admitted incrementally by the next prefill chunk, not
preallocated for the full long prompt. The fixed P/D pair must atomically own
enough P and D capacity for the chunk before it can launch. A failed two-sided
claim rolls back without leaving one-sided capacity. The normal P-to-D handoff
then transfers newly computed KV, and decode starts only after D ownership is
ready.

For a lower-tier resume, CPU/SSD data restores to P-HBM first. P processes the
fresh or recomputed suffix in chunks while D admits the corresponding lineage.
For an HBM-resident return, the prior D-HBM object is copied back to P through
the direct fabric before resumed prefill. All paths obey the same fixed-pair
FIFO and exact owner barriers.

## Fixed-K Poisson closed-loop design

The headline load design is an open Poisson stream of **sessions** feeding a
finite FIFO admission backlog, combined with closed-loop dependencies inside
each admitted session:

- `max_active_sessions=20` is fixed while offered session rate is swept;
- a session that arrives when all 20 slots are occupied waits in the session
  admission queue; it is never dropped;
- after admission, only the first call is ready. Call `N+1` is released at the
  actual completion of `N` plus the trace's human/tool/mixed gap;
- a slot is released only by final-call completion on the owning decode path;
- every finite cell drains the complete planned cohort with zero censoring;
- every policy at one `(rate, seed)` receives identical Poisson unit draws,
  offered timestamps, runtime session identities, and measurement target set.
  Hash mismatch is fatal.

This answers a different question from a closed backlog K sweep. It exposes
session admission queueing as rate rises while retaining realistic internal
think/tool time. Do not vary K in the headline Poisson panel.

### Discovery suite

`configs/experiments/online_tracelab_qwen3_1m_p4d4_poisson_backlog_discovery.json`
uses:

- rates `[0.002, 0.003, 0.0045, 0.006, 0.009, 0.0135, 0.02025,
  0.030375]` sessions/s;
- fixed seed `17`, K=20, 16 complete repetitions of the two-template cohort;
- 32 sessions and 64 LLM calls per cell;
- `full_recompute`, `tiering`, `tiering_partial_recompute`, and the strict
  oracle, for 32 total cells.

This suite is for disjoint rate discovery and debugging. Use it to find where
SSD resumes are material, admission/transfer queueing is visible, the partial
policy actually fires, and the qualitative ordering is informative. Do not
treat its single seed or two-template cohort as the main result.

### Main suite

`configs/experiments/online_tracelab_qwen3_1m_p4d4_main_long_poisson.json`
uses:

- rates `[0.003, 0.006, 0.009, 0.0135, 0.02025, 0.030375]` sessions/s;
- seeds `[101, 211, 307]`, K=20, and eight complete repetitions of the
  eight-template main cohort;
- 64 sessions and 192 LLM calls per cell;
- all four finite baselines plus the strict oracle, for 90 cells;
- a preregistered requirement that SSD resumes under `hbm_cpu_ssd` are at
  least 30% of **all measured requests** in the accepted opportunity region.

The three seed values and all unit-exponential draws are fixed across systems.
Do not resample a favorable seed per policy. If discovery changes the final
reported rate subset, commit that selection and rationale before running the
main seeds. Do not tune the partial-policy ratio or cost multiplier on the main
cohort.

## Metrics and denominators

The final reporting commit uses online artifact schema 12, session report
schema 11, agentic-KV report schema 20, and the independent HBM occupancy
schema 1. Consumers fail closed on older versions rather than silently
interpreting shifted columns or changed denominators.

Report at least the following per rate and policy:

| Metric | Definition and denominator |
| --- | --- |
| Session throughput | Completed measured sessions divided by the exact measurement duration. Show absolute sessions/s; oracle-normalized values are paired by the same rate, seed, and measured session set. |
| Session admission wait | Offer-to-active-slot admission per session, mean and tail. This is where fixed-K Poisson backlog pressure appears. |
| Server-added JCT | Session offer-to-final completion minus the sum of trace-provided human/tool/mixed gaps. It retains admission, scheduler, compute, communication, restore, and recomputation time without letting idle think time dominate. Report mean and p95/p99 as configured. |
| TTFT | Request ready/admission to first generated token for all measured calls. Keep initial and resumed calls distinguishable. |
| Resume TTFT | The same request-local first-token path for every non-initial measured call, including zero-reuse returns. Denominator is all non-initial measured calls, not only CPU/SSD hits. Report mean and p95. |
| TPOT | Mean inter-token interval over calls that generated at least two tokens. Report mean and p95 and preserve the eligible-call count. |
| Resume source fractions | Report both **attempted physical source** (the HBM/CPU/SSD object from which I/O was initiated) and **effective surviving source** after a later active-prefill discard. Each HBM/CPU/SSD fraction uses all measured requests as its denominator, including initial and non-reuse calls, and is split by human/tool return class. Report recomputation/all requests separately. |
| Swap overhead | Separate owner HBM-admission wait, transfer queue wait, transfer service, and pair/preparation admission. Request-summed stall divided by request latency is request-centric; restore-interval union divided by makespan is occupancy/exposure. Never sum overlapping requests and call that wall-clock penalty. |
| Recomputation overhead | Exact provider marginal recompute critical-path COMP divided by total modeled COMP critical path, plus recomputed-prefix tokens divided by executed prefill tokens. Do not substitute a token fraction for a time fraction. |
| HBM admission wait | Canonical total is destination-restore admission critical wait plus the post-restore critical wait of **every gross P/D chunk attempt**, successful or cancelled. Keep successful and cancelled attempt count/wait, gross enqueue-to-admit wall wait, and restore-destination wait separately. Gross wall waits may overlap pre-restore intervals and must not be blindly added to the canonical critical path. Pair FIFO, transfer queue, and transfer service remain separate. |
| Average active batch size | Non-dummy `len(batch.requests)` for each completed online model iteration, with P/D role breakdown. |
| HBM occupancy | Time-weighted physical stack and non-additive logical-reservation overlay described below. Report average and peak per instance. |

The runner writes `poisson_rate_metrics_source.csv` and separate rate plots for
resume TTFT, TPOT, server-added JCT, total HBM admission wait,
restore-destination HBM admission wait, P/D chunk admission wait, active batch
size, and HBM occupancy. Curves show mean and p95 where applicable, fixed-seed
means, and 95% Student-t confidence intervals across the three main seeds.

For schema 12, the `attempted_{hbm,cpu,ssd}_resume_*` columns are the
authoritative physical-resume numerators and require positive restored hit
tokens. The older `hbm/cpu/ssd_resume_*` columns retain raw source-label
compatibility: a zero-overlap continuation can still name the tier where its
now-irrelevant old object resided even though it performs no I/O. Do not use
those legacy columns for the 30% SSD opportunity contract. A zero-overlap
return contributes neither attempted resume, dropped/KV-unavailable miss,
recompute tokens, nor transfer bytes.

### SLO lines

Resume-TTFT uses the preregistered rule `5 *` strict-infinite-HBM zero-load
p95. The separate K=1 calibration completed over the same eight templates and
24 calls. Its 16 non-initial strict-oracle calls have p95
`363,169,188,605.5 ns`, so the frozen line is:

```text
resume TTFT SLO = 1,815,845.9430275 ms
```

The calibration execution ID is
`online-tracelab-qwen3-1m-p4d4-main-long-zero-load-slo-calibration-20260721T134416521822Z-3d28e6d6`
and its session-report SHA-256 is
`d0c0832b297486bd6041f9dc00d1b1bea7e5e4ff10f6ac9c5b4b293f9c3f2f13`.
The TPOT line is the preregistered loose interactive-agent threshold of
100 ms/token. Both thresholds were frozen in `21bb8d0` before the expanded
main sweep. They must be dotted reference lines, not refit after observing the
main data.

### HBM occupancy semantics

The physical categories form a true capacity stack:

```text
physical_idle_reusable
+ physical_non_idle_active
+ physical_free
= allocatable HBM capacity
```

`logical_destination_admission_reservation` is not another physical stack
segment. A future destination reservation can overlap the still-physical
source/victim of an asynchronous demotion. The overlay is decomposed as:

```text
logical reservation
= reserved_free_slack
+ future_reclaim_backed_reservation
```

Only `reserved_free_slack` can be added to physical occupancy for a
reservation-adjusted claim. `future_reclaim_backed_reservation` is displayed as
a non-additive overlay. Stacking it on top of physical occupancy double-counts
the same bytes.

## Safety and causality invariants

The collector should fail closed on every violation below:

1. Logical sessions are never dropped because K is full or KV is evicted.
   Offered, admitted, completed, remaining, and censored session counts must
   reconcile, with zero censoring in headline full-drain runs.
2. A continuation is released only after its predecessor completes plus its
   exact trace gap. Restore never starts before request readiness.
3. Every same-pair preparation retains FIFO order. A later request cannot
   silently bypass the head to gain capacity.
4. P/D chunk capacity is atomic: both P and D claims commit or neither does.
   One-sided rollback, claim generations, and histories must reconcile.
5. A foreground restore reserves destination HBM before I/O and the owner
   cannot enter a compute batch before complete readiness. It never blocks an
   unrelated current batch.
6. Every SSD hit has exactly ordered SSD-to-DRAM and DRAM-to-P-HBM stages. A
   direct-SSD policy has no persistent CPU object.
7. Background demotion retains an authoritative source until commit. Race
   cancellation cannot lose the only valid copy, and consumed service/wasted
   bytes remain accounted.
8. Idle LRU victims are reclaimed before active work. Active preemption is a
   common fallback across finite baselines, not a policy-specific advantage.
9. In-flight ASTRA graphs, frozen/current-pass chunks, active foreground
   restores, pinned SSD records, and one-sided P/D claims are never selected as
   unsafe victims.
10. Recompute attribution covers every replayed prefix token exactly once.
    Prompt throughput must not double-count a discarded/replayed active
    prefix.
11. HBM physical capacity, logical reservations, CPU/SSD occupancy, transfer
    calendars, durable-record pins, and source/destination ownership conserve
    at every report boundary.
12. The strict oracle must remain nonbinding and observe only HBM sources for
    reusable returns, with all forbidden finite-capacity counters equal to
    zero.
13. All paired policies at one rate/seed must have identical selected-session,
    offered-arrival, Poisson-unit-draw, and measured-session hashes.
14. Every reported latency is nonnegative, every dependency timestamp is
    monotonic, and timing warnings outside the explicitly permitted one-hour
    long-request warning are fatal.

## Active-prefill P/D deadlock: reproducer and fix

### Failure discovered online

The first committed discovery grid was launched through the real online path,
not capacity replay. At rate `0.002`, every finite policy eventually stopped
making progress while the strict oracle continued. The watchdog found no
future exact wakeup after 192 polls.

In the full-recompute cell, simulated time was about 17,101.6 seconds. The
FIFO-head request had computed 802,129 of 999,364 prompt tokens, and P used
59,559,383,040 of 60,106,987,520 allocatable bytes per rank. In tiered/partial
cells, simulated time was about 18,560.3 seconds, the head had computed 933,201
of 999,364 tokens, and P/D usage was 57,562,632,192 bytes per rank. Pending
handoffs remained live.

The root cause is structural. P/D next-chunk admission occurs before Scheduler
step 3. Twenty active partial prefills can fill both HBM instances. Idle LRU
has no victim, and the older active-preemption path handled decode generation
but not active partial prefills. Every request then waits for a next chunk,
none can finish and become idle, and no event can free capacity.

### Committed fix design

When the same-pair FIFO head cannot atomically acquire its next P+D chunk and
ordinary reclaim cannot make progress, the new path:

1. keeps the FIFO head immutable;
2. considers only restore-ready, non-inflight, non-frozen queued P-prefill
   peers, preferring least-progressed and newest victims and using a
   higher-progress fallback only when FIFO liveness requires it;
3. performs a complete preflight of P/D ownership, weight floors, active
   reclaim claims, pending chunk identities, and generations before mutation;
4. cancels exact pending chunk claims and rolls back any safe one-sided state;
5. atomically releases the victim's P and D owned KV;
6. resets normal original-input prefill to token zero while preserving the
   logical session, active slot, queue order, arrival/admission timestamps,
   first-schedule timestamp, TTFT identity, and normal later P-to-D handoff;
7. records every discarded frontier and generation so trace generation and
   online metrics attribute each replayed interval as real recomputation
   without double-counting prompt throughput; repeated preemption remains
   legal even before replay reaches the original restored-hit frontier; and
8. retries the unchanged FIFO head exactly once.

Frozen graph work and current-pass admitted IDs remain protected. P/D ownership
must be injective and the paired schedulers must be distinct; unsupported
many-P-to-one-D mappings and P/D aliasing fail fast. The final preflight proves
weight-floor usable ownership and allocator headroom on both schedulers before
either is mutated. Cancelled claims retain exact count, wait, critical-wait,
and generation provenance.

The structural reclaim change is commit `565defe`. It closes the state in which
all finite HBM is owned by active partial prefills and no idle LRU victim can
exist.

### Retry-cache dependency exposed after graph commit

The first full high-rate smoke after `565defe` passed the original discovery
deadlock but exposed a second liveness dependency. At simulated time
`3,405,582,422,130 ns`, request 36 needed `3,221,225,472` bytes per rank on
both P and D while each side had only `754,042,880` bytes of slack. There was
no active reclaim claim, pending HBM allocation, or protected victim. Requests
14 through 34 held enough P+D KV to make progress, but at the first failed
admission attempt they were frozen at a `131,072`-token graph frontier and
were correctly ineligible for preemption.

The Scheduler request-deque head in that snapshot was request 14, at
482,399 of 999,364 prompt tokens; request 36 was the distinct Router P/D
pair-admission FIFO head. Keeping those two queue heads separate is essential
when reproducing the snapshot.

After those graphs committed, the requests became safe victims. HBM used/free
bytes and the manager capacity generation did not change, however, so the
Router's cached two-sided admission state still looked identical and victim
selection was never retried. Invoking the same progress operation manually at
the failure snapshot immediately reclaimed a victim and admitted the FIFO
head. This proved that the missing dependency was graph-commit
reclaimability, not capacity, protected-set, or claim cleanup.

Commit `3f6e79e` adds a scheduler-local monotonic
`pd_prefill_reclaimability_generation`. A P scheduler advances it once per
completed batch when frozen P/D chunks actually commit and thaw. The Router
includes that generation only in the P/D pair retry key; it does not perturb
unrelated global memory-manager retries. Tests prove that an unchanged
capacity tuple is retried after graph commit, that multiple victims in one
batch advance the generation once, and that non-P/D paths do not advance it.

The reporting change `b60f306` does not alter victim selection or admission.
It adds per-request cross-layer conservation for active-prefill events,
successful/cancelled P/D attempts, attempted-versus-surviving resume
provenance, exact CSV session identity, and strict numeric types. The bounded
debug cohort in `708f107` reduces only the number of complete repetitions; it
does not slice sessions or change the 1M-token transform. Final test and online
records are below.

## Validation sequence

Run validation in this order. Do not launch the 90-cell main suite as a test.

### 1. Inspect provenance and preserve unrelated work

```bash
git branch --show-current
git log -5 --oneline
git status --short
git submodule status astra-sim
```

Confirm `565defe`, `3f6e79e`, `b60f306`, `708f107`, and ASTRA
submodule `63e91acf0b48471d33c7986e36e815dbb07f9bb3`. Do not stage unrelated
untracked files or historical result trees.

### 2. Run focused regression tests

```bash
python -m pytest -q \
  tests/test_agentic_kv.py \
  tests/test_agentic_router.py \
  tests/test_active_kv_preemption.py \
  tests/test_pd_pending_launch_admission.py \
  tests/test_agentic_scheduler_metadata.py \
  tests/test_online_experiments.py \
  tests/test_online_measurement.py \
  tests/test_online_trace_calibration.py \
  tests/test_session_metrics.py \
  --tb=short
```

Required cases include 20 partial prefills, FIFO priority inversion, frozen and
current-pass protection, one-sided rollback, first-attempt cancellation,
non-default chunk lengths, P/D generation reuse, prompt/recompute attribution,
manager aggregate conservation, and report schema validation.

### 3. Run the complete repository tests

```bash
python -m pytest -q tests --tb=short
```

The final debug code passed 664 tests. Treat that exact count as a checkpoint,
not as a reason to ignore new tests added after handoff.

### 4. Run the exact high-pressure online debug pair

The `9fd4f0b` spec established the portable high-rate pair. Commit `708f107`
then bounds this **debug** spec to four complete repetitions of the two
discovery templates: eight sessions and 16 calls per cell. It launches exactly
the normal finite tiered cell and its paired strict residency oracle at rate
`0.030375`, K=20, and seed 17. Both are normal online `python -m serving`
subprocesses; neither is capacity replay. The 32-session discovery and the
64-session main cohorts remain unchanged.

The command below is the portable destination form. The recorded source-host
run used the same command with both `/data` TraceLab paths replaced by their
content-identical `/tmp` paths.

```bash
python -m serving.online_experiments \
  --spec configs/experiments/online_tracelab_qwen3_1m_p4d4_high_rate_smoke.json \
  --mode poisson \
  --dataset-override /data/tracelab-schema3-sps0.2-final.jsonl \
  --dataset-manifest-override /data/tracelab-schema3-sps0.2-final.jsonl.manifest.json \
  --max-parallel 2 \
  --timeout-seconds 1800
```

Acceptance requires:

- completion without watchdog liveness failure or timeout;
- exact planned/offered/admitted/completed reconciliation and zero logical
  session drops/censoring;
- at least one active-prefill liveness preemption in the finite cell, or a
  documented proof that the higher-rate arrival draw no longer reaches the
  original state and an additional deterministic reproducer that does;
- nonnegative monotonic timing, exact P/D claim history, recompute frontier and
  prompt-throughput reconciliation, and weight-only final scheduler baseline;
- no live tier entry, claim, preparation lock, restore wait, transfer, ASTRA
  window, or external fabric job at cutoff.

The exact command, branch commit, wall duration, output path, and report hashes
are recorded in the final validation record below.

### 5. Optional discovery-only debugging

After the debug pair, the expanded 32-cell discovery remains optional. The user
asked not to run either discovery or main on the current machine; this command
is retained only for the destination workflow.

```bash
python -m serving.online_experiments \
  --spec configs/experiments/online_tracelab_qwen3_1m_p4d4_poisson_backlog_discovery.json \
  --mode poisson \
  --dataset-override /data/tracelab-schema3-sps0.2-final.jsonl \
  --dataset-manifest-override /data/tracelab-schema3-sps0.2-final.jsonl.manifest.json \
  --max-parallel 4 \
  --timeout-seconds 3600
```

The runner must produce source CSVs and plots only after every cell validates.
Never merge a partially successful execution with a later rerun by hand.

## Parallelism, runtime, and destination-server guidance

`max_parallel=4` is a conservative high-pressure operating point, not a simulator
correctness limit. Every cell has a run-specific ASTRA input root, independent
process group, workload copy, and result directory, so independent cells may
run concurrently. Increasing host parallelism does not represent more serving
hardware inside a cell; it only reduces experiment wall time.

There is no software cap at four. `serving.online_experiments` accepts every
positive `max_parallel` and uses one worker per independent cell. Each worker
owns a unique run ID, result directory, process group, and
`astra-sim/inputs/runs/<run-id>` tree; shared model/profile/config inputs are
read-only. A cell launches a Python frontend and one ASTRA C++ process, with
bursty Chakra conversion and trace/graph I/O.

The source machine has 24 physical cores (two 12-core Xeon E5-2650 v4 sockets,
no SMT), 62 GiB RAM with about 51 GiB available at inspection, and a root
filesystem that reached 100% displayed use with less than 1 GiB free and was
still declining at final inspection.
Historical online results occupy about 4.8 GiB and retained ASTRA input trees
about 7.4 GiB across 88 directories. Successful cells clean their ASTRA
scratch by default, but failed or killed cells retain roughly 140--500 MiB
each for diagnosis. One interrupted full high-pressure cell grew about 2.8 GiB
before it was stopped. Do not launch another simulation, full test, or docs
build on this host without first moving the work or explicitly cleaning
reviewed artifacts. Disk headroom and transient Chakra/ASTRA graph I/O are the
current hard constraints; RAM and a software parallelism cap are not.

Observed wall times **before** the active-prefill fix were:

- zero-load calibration: about 21.4--22.3 seconds per cell with two cells in
  parallel;
- an older 60-cell long suite: 54 minutes 14 seconds observed at
  `max_parallel=4`; individual cells were 162.6--314.8 seconds (about 212
  seconds mean), with 12,718.9 serial seconds in total.

Those old runs are useful only for rough wall-time planning. Their performance
outputs are stale and the estimates are extrapolations, not measurements of the
fixed code. Linear scaling from the observed 60-cell run puts the expanded
90-cell main near 81 minutes at four-way concurrency; use 75--95 minutes as a
planning range. The two-template 32-cell discovery is expected to take roughly
10--20 minutes once liveness is fixed. The runner has a hard 3,600-second
per-cell cap.

The committed debug-only isolation pilot contains eight fixed cells. The
source-host validation ran it twice with identical work. These are portable
destination commands; the recorded runs substituted the content-identical
`/tmp` TraceLab paths for `/data`:

```bash
python -m serving.online_experiments \
  --spec configs/experiments/online_tracelab_qwen3_1m_p4d4_parallelism_smoke.json \
  --mode poisson \
  --dataset-override /data/tracelab-schema3-sps0.2-final.jsonl \
  --dataset-manifest-override /data/tracelab-schema3-sps0.2-final.jsonl.manifest.json \
  --max-parallel 4 --timeout-seconds 3600

python -m serving.online_experiments \
  --spec configs/experiments/online_tracelab_qwen3_1m_p4d4_parallelism_smoke.json \
  --mode poisson \
  --dataset-override /data/tracelab-schema3-sps0.2-final.jsonl \
  --dataset-manifest-override /data/tracelab-schema3-sps0.2-final.jsonl.manifest.json \
  --max-parallel 8 --timeout-seconds 3600
```

This K=1 pilot tests launcher isolation and host scaling only. It is not proof
that eight high-pressure 1M-token cells fit the same RSS/scratch envelope. All
eight logical rows were equal between the runs after excluding only run IDs,
artifact paths/hashes that embed those paths, and launcher wall duration. The
paired workload SHA-256 was identical. The complete rate-source CSV was
byte-identical. External `/usr/bin/time -v` command wall fell from about 49.45
seconds at four-way concurrency to 28.44 seconds at eight-way concurrency, a
1.74x speedup and 42.5% reduction. The manifest's own
`provenance.created_at`-to-`finished_at` span fell from 44.407 to 23.422
seconds, or 1.90x and 47.3%. Mean per-cell time rose from 21.59 to 22.86
seconds because of host contention. Thus eight workers are supported and
isolated for this light pilot. More than eight was not tested because the
pilot has only eight cells and the nearly full disk makes a heavier pilot
unsafe. Eight cells already imply roughly 16 persistent frontend/backend
processes, plus bursty converter work and concurrent trace/graph I/O.

On a destination server, prefer at least 32 physical/logical cores, 128 GiB
RAM, and at least 100 GiB of free local NVMe scratch, with the repository and
default ASTRA run roots placed on that filesystem. Begin with the light
eight-worker isolation pilot, then compare representative high-pressure cells
at four and eight workers before launching discovery. Do not try 12 until that
representative pilot establishes CPU, RSS, scratch, and I/O headroom. Compare
semantic metrics, selected/offered hashes, invariants, and ASTRA logical
timestamps. Also record aggregate RSS, disk high-water mark, I/O wait, and
per-cell wall-time inflation. Whole manifest hashes will differ because paths
and wall-clock metadata differ. If logical outputs change, treat it as an
isolation bug and return to the last validated concurrency; do not average the
discrepancy away.

## Final debug validation record

All records in this section are debugging evidence, not paper measurements.
The successful bounded smoke and both parallelism pilots report commit
`708f107b00d1fe9530a34be24aec66b7fa2e19da`. Their Git provenance says
`dirty=true` because the shared worktree contained unrelated untracked
calibration and historical-result files; the tracked cold-KV implementation
was clean.

### Tests and documentation

| Check | Result |
| --- | --- |
| Focused command shown above | 402 passed in 3.27 seconds |
| `python -m pytest -q tests --tb=short` | 664 passed in 21.25 seconds on the final rerun |
| `(cd docs && ./node_modules/.bin/docusaurus build)` | succeeded before the final prose audit; only the pre-existing `vscode-languageserver` critical-dependency warning appeared. It was not rerun after the last Markdown-only corrections because the root filesystem reached 100% displayed use. |

### Bounded high-rate online pair

The source-host command was the validation command above with the two `/data`
paths replaced by
`/tmp/tracelab-schema3-sps0.2-final.jsonl` and its adjacent
`.manifest.json`; the bound dataset hashes matched. `/data` is the portable
destination convention, not a different workload.

Execution
`online-tracelab-qwen3-1m-p4d4-high-rate-smoke-20260721T155757406017Z-84952655`
completed both cells with return code zero and no timeout. The finite cell took
47.103 seconds and the oracle 30.963 seconds; the manifest suite span was
47.305 seconds. Workload SHA-256 was
`c607f772cf9dcf9552dd18af66a39e2e5f8834a92fd69817dfa1f6991ebfd19c`.
Both cells reconciled eight planned/offered/admitted/completed sessions and 16
requests, with zero logical drops, censoring, or remaining sessions.

The finite run exercised nine active-prefill liveness preemptions and replayed
5,711,751 tokens. It recorded 103 successful P/D chunk admissions plus one
cancelled attempt, all reconciled; the P/D and cross-layer accounting audits
passed. It ended with no
active reclaim claim, pending external-fabric job, or foreground/background
DMA tail. Seven external-fabric jobs were issued and completed; the oracle
issued and completed eight. Capacity drops and CPU/SSD capacity evictions were
zero. The finite run emitted five permitted
`request_latency_over_one_hour` warnings and no unpermitted timing warning;
the oracle emitted none.

The physical resume attempts in this deliberately small liveness cohort were
seven HBM, one CPU, and zero SSD over all 16 requests. Active-prefill replay
later discarded all 5,704,428 restored-hit tokens, so no attempted resume
survived to effective execution in the finite cell. This smoke can neither
satisfy nor refute the preregistered main-region requirement of at least 30%
SSD resumes over all requests.

Content hashes are:

| Artifact | SHA-256 |
| --- | --- |
| Suite manifest | `11f1ac658744eef9f6b62e5b71808ace3accf9dbf86c4712723711e0564d6bcc` |
| Summary JSON | `b407ba500ff149c96b5a7d2da0257356788704318bc6ec39d0f54c871455db18` |
| Summary CSV | `65b1317014cd6919d7e0c2c57f946cf0125281f6016ff7e7a7473ab456c2993e` |
| Rate-source CSV | `7b7fc9b41b0143a41704713c7f47902b8649596dd668875399a1b2030427002c` |
| Finite requests / session / agentic reports | `7d456f52c98514fd7ebc7ff85e3f160f671633d8b8d135e211443802288ad145` / `d20a9e976ec39c789149afec512c1a716cd13216df2fc27f586928d445de4fbb` / `8d46c495190f85d298ae99c5528558186c506f175e4ab8b9acb39e12cfe1e44f` |
| Oracle requests / session / agentic reports | `4fb8abf5078e299dd91182bdcb2017600facdf66dd12940d03f12d457f30044f` / `c69de3b847cc97201c56950695044d3fb43ab324e2774b7ea6ceb494d647ff1d` / `c86e22dbe6924b2d8e8cb7f993c2efa9239a7cf4ba838d09a2475eb8f4b548c9` |

### Four-versus-eight worker pilot

The four-worker execution is
`online-tracelab-qwen3-1m-p4d4-parallelism-smoke-20260721T160022163341Z-e873d63c`;
the eight-worker execution is
`online-tracelab-qwen3-1m-p4d4-parallelism-smoke-20260721T160125899372Z-97c0b82a`.
Both completed all eight cells with return code zero and no timeout. The
workload hash in both was
`12b6f66409f9f1ae23565432c235f7c11180702c4c6ac33f520ce6d281aa0761`.
The four-worker cells ranged from 21.056 to 22.324 seconds, while eight-worker
cells ranged from 22.772 to 23.064 seconds. The rate-source CSV from each run
is byte-identical with SHA-256
`15a0180acd0fd56186c3a2de022e14a36e9b28a15cdb7aa50b9288994cc4cf8c`.
The outer `/usr/bin/time -v` observations were 49.45 versus 28.44 seconds
wall, 452% versus 778% CPU, and 587,556 KiB maximum single-process RSS in both
runs. These are not aggregate memory high-water marks, so destination
validation must measure the whole process group as well.
Suite-manifest hashes are respectively
`8cc279100a0508d17b25b12ffb74f49fb4173aa60359b1bc1ce239dbe7a81ddd`
and
`7f0f59d8639817ea0a018d523f86e2a6bbd9b57760229d4e1c9e6e8d7574dee9`.

An earlier 16-repetition high-rate diagnostic
`online-tracelab-qwen3-1m-p4d4-high-rate-smoke-20260721T154631194045Z-a4036206`
ran commit `3f6e79e` over 32 sessions and 64 requests per cell. Its oracle
completed in 98.695 seconds, while the finite cell passed the old failure point
and continued until the operator interrupted the suite after about 8.5 minutes
because retained ASTRA scratch approached 2.8 GiB. The finite run and suite
manifest remain `running`, and no summary exists: this is neither a success
nor another liveness failure. Only that run's regenerable ASTRA scratch was
removed for disk safety; its partial output/log directory remains.

## Moving to another server

### Code and submodule

Prefer pushing the branch and the ASTRA submodule commit to reachable remotes.
The superproject currently records ASTRA commit
`63e91acf0b48471d33c7986e36e815dbb07f9bb3`, published on branch
`codex/cold-kv-handoff` at `https://github.com/skyp0714/astra-sim.git` and
selected by `.gitmodules`. If either commit is unavailable from its remote,
create bundles for both repositories **after every final commit in this
handoff**; the main bundle does not contain an unavailable submodule object.

```bash
git bundle create /data/cold-kv-hardening.bundle codex/cold-kv-hardening
git -C astra-sim bundle create /data/cold-kv-astra.bundle HEAD
```

On the destination:

```bash
git clone --recurse-submodules <repository-url> LLMServingSim
cd LLMServingSim
git fetch /data/cold-kv-hardening.bundle \
  codex/cold-kv-hardening:codex/cold-kv-hardening
git checkout codex/cold-kv-hardening
git submodule update --init --recursive
```

If the recorded ASTRA object is not on its remote, first clone/initialize a
normal `astra-sim` worktree, fetch `/data/cold-kv-astra.bundle` from inside that
worktree, fetch the bundle's `HEAD`, and checkout the exact
superproject-recorded object:

```bash
git submodule update --init --recursive astra-sim
git -C astra-sim fetch /data/cold-kv-astra.bundle HEAD
git -C astra-sim checkout --detach \
  63e91acf0b48471d33c7986e36e815dbb07f9bb3
git rev-parse HEAD
git -C astra-sim rev-parse HEAD
git submodule status astra-sim
```

The first `git rev-parse` should identify the final handoff commit on
`codex/cold-kv-hardening`; the second and `git submodule status` must show the
exact ASTRA hash above before building.

### External data and calibration artifact

The validated converted trace and sidecar currently live under `/tmp` on the
source host. Treat that location as ephemeral: copy both to durable destination
storage before rebooting or retiring the source host, then verify them:

```bash
sha256sum \
  /data/tracelab-schema3-sps0.2-final.jsonl \
  /data/tracelab-schema3-sps0.2-final.jsonl.manifest.json
```

Also copy the zero-load SLO calibration execution directory if long-term
artifact provenance is required. The numeric SLO is frozen in the specs, but
the bound report should remain available for audit. Historical main/discovery
result trees are not required to continue the experiment.

### Build and smoke

Use the simulator environment, initialize Chakra/ASTRA, and build according to
the normal contributor guide. `scripts/docker-sim.sh` starts an interactive
container; run the following compile, test, and experiment commands **inside**
that container rather than expecting the wrapper to execute the next shell
line automatically:

```bash
./scripts/docker-sim.sh
./scripts/compile.sh
python -m pytest -q tests --tb=short
```

Then perform the exact high-pressure smoke from the validation section. Only
after it passes should the destination run discovery or main.

### Main command on the destination only

```bash
python -m serving.online_experiments \
  --spec configs/experiments/online_tracelab_qwen3_1m_p4d4_main_long_poisson.json \
  --mode poisson \
  --dataset-override /data/tracelab-schema3-sps0.2-final.jsonl \
  --dataset-manifest-override /data/tracelab-schema3-sps0.2-final.jsonl.manifest.json \
  --max-parallel 8 \
  --timeout-seconds 3600
```

Choose `8` only after the destination concurrency pilot. Preserve the execution
directory exactly; it contains the spec snapshot, cohort manifest, per-cell
argv and hashes, reports, source CSVs, and plots.

## Commit timeline

The following milestones explain the design evolution and which older behavior
was superseded:

| Commit(s) | Milestone |
| --- | --- |
| `87bb0ff`, `1b89385` | Add agentic idle-KV tiering and fix source session identity. |
| `a958ac7`, `a82e651`, `85e6a31` | Preserve full TraceLab inter-turn gaps, separate policy-independent reuse from provider hits, and complete lineage/reset semantics. |
| `69b2422`, `6c1dfb3`, `ff0df17` | Establish HBM recompute, direct SSD, and capacity-only HBM/CPU/SSD baselines. |
| `36bfaae`, `c53e46c`, `c50c00b`, `fb254fd` | Add direct SSD P/D restore, all-request resume denominators, and two-sided P/D HBM admission. |
| `8968e17`, `0a02fdf` | Put P/D traffic, including direct cold HBM returns, on congestion-aware ASTRA paths. |
| `b3ec70d`, `c9fb53d`, `296ea19`, `a073ca9` | Replace synchronous/global swap behavior with request-local asynchronous restore semantics. Earlier synchronous commits are adverse sensitivities, not headline policy. |
| `edef6bd`, `785995d` | Enforce serial SSD-to-DRAM-to-HBM restores and realistic aggregate eight-drive throughput. |
| `052becb`, `974e009`, `e4f4cab`, `32c11d3` | Add bounded session-load modes and enforce FIFO waiting instead of logical session drops. |
| `b8479bd`, `be4426d`, `59543c5` | Add Qwen3 1M P4+D4 runtime contract, dormant cohorts, and lineage-safe global context scaling. |
| `87b9804`, `45d5042` | Fit H100 kernel/roofline models and connect their COMP predictions to the online trace path. |
| `644d12f`, `c7e0cd1`, `9bffb2d` | Add online tier scheduling, bounded experiment runner, and fail-closed report validation. |
| `6134c22`, `068be6a`, `50f5363` | Split backlog/Poisson suites and define balanced quick versus disjoint main-long cohorts. |
| `7cb8829`, `fea4286`, `328e823`, `0fbdc3e` | Add paper Poisson load, server-added JCT/admission metrics, capped discovery, and exact paired arrival proofs. |
| `3e0ff2d`, `aa984c5`, `7d4aea1`, `a464520`, `7793ac6` | Add causal queue-aware recomputation, shadow HBM/transfer projections, and block-aligned partial-prefix choice. |
| `7e00988` | Replace full-prompt P/D KV preallocation with incremental chunk admission. This exposed the active-partial-prefill liveness hole described above. |
| `1233f04`, `e114a59` | Validate partial-prefix online accounting and fix seed/K/cohort contracts for Poisson sweeps. |
| `399155d`, `d8be86a`, `075e48e` | Add online HBM occupancy, batch-size metrics, per-instance peaks, rate plots, and confidence intervals. |
| `041f0ca`, `21bb8d0` | Preregister zero-load SLO calibration and freeze the resulting resume-TTFT/TPOT thresholds. |
| `e2d53e0`, `9ff11fb` | Extend discovery to eight rates and main to six rates—32 and 90 cells respectively—and freeze the expanded grid test. |
| `8639826` | Add the fixed eight-cell 4-versus-8 host-concurrency pilot. |
| `565defe` | Guarantee finite-HBM P/D prefill progress through atomic peer replay with strict preflight and repeated-preemption support. |
| `9fd4f0b` | Add the portable two-cell rate-0.030375 online liveness smoke. |
| `b60f306` | Add schemas 12/11/20, request-level event/JSON/CSV conservation, attempted-versus-surviving resume provenance, and gross P/D attempt wait. |
| `3f6e79e` | Retry P/D admission when graph commit changes active-prefill reclaimability without changing the HBM capacity tuple. |
| `708f107` | Bound the portable high-rate debug pair to four complete repetitions for disk-safe liveness validation. |
| This documentation commit | Record the post-fix tests, online debug artifacts, host-concurrency result, and destination-server procedure. |

## Results that must not be used

Do not use any of the following as a paper result:

- `results/online/online-tracelab-qwen3-1m-p4d4-main-long-poisson/executions/online-tracelab-qwen3-1m-p4d4-main-long-poisson-20260721T084042016535Z-9486be38`;
- the older Poisson refinement/discovery graphs that showed approximately
  0.23% slowdown or an apparent full-recompute/tiering/oracle ordering;
- `/tmp/llmsim-discovery-e114a59` or the corresponding failed finite-HBM
  discovery execution;
- high-rate execution
  `online-tracelab-qwen3-1m-p4d4-high-rate-smoke-20260721T153339756931Z-8d1704a9`,
  whose finite cell exposed the graph-commit retry defect while its oracle
  completed;
- interrupted high-rate execution
  `online-tracelab-qwen3-1m-p4d4-high-rate-smoke-20260721T154631194045Z-a4036206`,
  whose suite manifest remains `running` because the operator stopped it for
  disk safety;
- successful bounded smoke `...20260721T155757406017Z-84952655` and the two
  parallelism pilots as policy-performance or SSD-opportunity evidence; they
  validate liveness, accounting, and launcher isolation only;
- quick/backlog outputs created before incremental P/D chunk admission and the
  active-prefill liveness fix;
- any `results/agentic-kv/**` standalone capacity-replay output as online TTFT,
  TPOT, queueing, JCT, or throughput evidence; or
- a partially completed suite assembled manually from cells with different
  commits, cohorts, arrival hashes, schema versions, or policy config hashes.

The historical artifacts in that list predate one or more of incremental P/D
admission, the liveness fixes, the current operational metric schema, the
fixed SLO, or the expanded rate contract. The `a4036206`, `84952655`, and
parallelism artifacts are current enough for their explicitly bounded debug
purposes but are incomplete, too small, or too lightly loaded for policy
claims. All are useful only within the scope stated above.

The one historical numerical artifact retained as a paper input is the
separately labeled K=1 strict-oracle SLO calibration identified above. It
establishes threshold provenance only; because it predates schemas 12/11/20,
it is not operational validation of the current reports and is not a main
policy comparison.

## Remaining work and known limitations

Before the main run:

1. move the branch, ASTRA object, TraceLab JSONL/sidecar, and any retained SLO
   artifact to the destination and verify every recorded hash;
2. start from a clean tracked worktree, build ASTRA/Chakra, run the focused and
   complete test suites, and repeat the bounded high-rate pair;
3. run a representative high-pressure four-versus-eight worker pilot and pick
   the highest concurrency whose logical output is identical and whose CPU,
   RSS, disk, and I/O headroom is safe;
4. run the expanded 32-cell discovery, select the informative rate region
   using only the disjoint cohort, and commit any main-rate subset before
   launching seeds 101/211/307;
5. verify `hbm_cpu_ssd` reaches at least 30% SSD resumes over all requests and
   that partial decisions occur only behind the severe gate;
6. verify every plot is regenerated from schema-12 suite source CSVs backed by
   session schema 11 and agentic schema 20, and render SVGs to PNG for remote
   review without editing plotted values; and
7. run the 90-cell main only after the discovery-rate decision and
   destination-host checks are committed.

Limit every paper claim by the following facts:

- TraceLab prompt text is sanitized, so reuse is a policy-independent adjacent
  lineage estimate rather than target-tokenizer exact LCP.
- Context scaling is synthetic and does not preserve TraceLab's empirical
  length distribution.
- Dense/full attention at one million tokens is extrapolated far beyond the
  measured legacy attention K range and is not sparse DCA.
- The exact H100 SKU, measurement software versions, and measurement command
  for legacy calibration sources are unknown.
- Host/storage gang FCFS and congestion-aware ASTRA are separate resource
  models, not a unified PCIe/NVMe/GPU contention simulator.
- SSD bandwidth uses manufacturer-limit sums; queue-depth, filesystem, RAID,
  DMA setup, and NUMA effects are not calibrated.
- The 512 GB host capacity is a deliberate pressure sensitivity.
- Generic radix prefix caching is disabled because it does not yet share a
  physical block directory with session-tier KV. Enabling both would
  double-count bytes and hits.
- The oracle removes finite residency pressure only; it does not remove
  compute, P/D transfer, batching, or session admission queueing.

## Starter brief for a new Codex session

Give a new session the repository and this instruction:

```text
Read docs/docs/contributor/cold-session-kv-online-evaluation-handoff.md in
full and continue from its status section. Preserve all unrelated dirty and
untracked files. The two active-prefill P/D liveness fixes and schemas
12/11/20 are already implemented and debug-validated; do not redesign or
silently retune them. Do not use any historical or bounded-debug output as a
paper result. First verify the branch, superproject and ASTRA commits, external
TraceLab hashes, clean tracked status, the documented focused tests, the full
test suite, and the exact rate=0.030375 K=20 seed=17 bounded pair through
python -m serving.online_experiments. The source artifacts currently live
under /tmp and must be copied to durable destination storage before this host
is retired; path-only sidecar differences are not hash-equivalent. Then run a
representative high-pressure 4-vs-8 concurrency pilot, followed by the
32-cell disjoint discovery. Freeze any informative main-rate subset in a
focused commit before launching seeds 101/211/307. The 90-cell main belongs on
the larger destination server and must use serving.online_experiments, never
standalone capacity replay. Preserve all execution directories and report
hashes; fail closed on cohort, arrival, schema, liveness, or accounting
mismatch.
```

---
sidebar_position: 12
title: Full-model HBF and ASTRA-Sim integration
---

# Full-model HBF and ASTRA-Sim integration

This page defines the implemented online boundary between the full-model
eight-card HBF server and ASTRA-Sim. It is both an operator guide and a claim
boundary: it explains which events ASTRA-Sim owns, which latencies remain
analytical inputs, and what must be checked before using a run in a comparison.

## Supported system

The strict online path models two physical servers:

| Server | Modeled resources | Role |
| --- | --- | --- |
| GPU server | one TP4 prefill instance and one TP4 decode instance on eight H100 GPUs | First-turn execution and explicit fallback recomputation |
| HBF server | eight HBF-NPU cards with card-local HBF and LPDDR | Full resume-prefill and decode after a session has a committed HBF record |

The GPU cluster must use
`Qwen/Qwen3-30B-A3B-Instruct-2507`, BF16 weights and KV, 16-token KV blocks,
PP1, and the `h100-qwen3-tp4-kernel-calibrated` GPU latency model. The P4 and
D4 instances must share one GPU server. Generic prefix caching, generic
agentic KV tiering, the infinite-HBM oracle, skipped prefill, and the existing
GPU-only power model are rejected rather than silently combined with this
path.

The HBF hardware file exposes the following parameters independently:

- card count and per-card HBF capacity;
- HBF read/write bandwidth and fixed latency;
- NPU peak throughput;
- LPDDR capacity and bandwidth;
- card-link, PCIe-root, inter-root, and RDMA bandwidth and latency;
- the explicit card-to-root and RDMA-NIC-to-root mappings;
- the PCIe peer-to-peer/cross-root mode; and
- the named-resource mode used by foreground collectives and lifecycle
  traffic.

The reference configuration is
`configs/wakekv_hbf/full_model_8card_server.json`. A different file may change
hardware values, but a layout key cannot change its canonical TP and replica
meaning. The reference file explicitly selects `pcie_resource_mode: shared`.
`legacy_isolated` is accepted only as a declared sensitivity mode; it restores
the old disjoint collective namespace and must not be used for publication
results.

## Request and KV lifecycle

Only agentic session rows with `sub_requests` are accepted. Flat or mixed
workload rows are rejected before placement state is mutated. Every call must
fit the current 1,010,000-token model contract, including its requested
output.

The implemented lifecycle is:

1. A first call runs through the GPU P4/D4 path.
2. At its turn boundary, the completed GPU KV becomes an HBF migration
   candidate. The migration and its publication are asynchronous ASTRA-owned
   work.
3. A continuation that finds a committed HBF version runs both its
   resume-prefill and decode on the HBF server. Committed prefix KV is read
   from HBF; current-turn KV and kernel workspace initially use bounded
   LPDDR.
4. The first output token fixes TTFT. Before the second token can run, a
   sufficiently large materialized prefill prefix is drained from LPDDR to
   HBF. A configurable recent tail remains in LPDDR. Decode is gated until
   the ASTRA append callback publishes the new placement; therefore the
   transfer affects the first inter-token interval, not TTFT.
5. At the next turn boundary, the remaining decode KV is appended to the
   durable HBF record. The successor is visible only after the append
   completes.
6. A continuation that returns while migration is in flight, or when HBF or
   LPDDR capacity cannot admit it, follows the explicit GPU recomputation
   route.
7. The final call releases all retained GPU, HBF, and LPDDR ownership.

Admission-time cache-hit accounting is immutable. The runtime separately
tracks the live HBF/LPDDR split after prefill and append publication, so a
drained fresh prefix cannot be misreported as an admission hit.

Active prefill drains reuse the ordinary LPDDR-to-HBF append DAG. ASTRA-Sim
models the exact card-local LPDDR read, HBF write bandwidth and fixed
latency, and contention with foreground attention. Appends are serialized
per session. A request waits for an older append before issuing the next
contiguous range. If target HBF capacity is unavailable, the gate is released
through an explicit LPDDR fallback rather than deadlocking.

GPU fallback deliberately does not restore decode-owned KV from D to P.
Modeling such reuse would require a D-to-P transfer, P-side HBM admission, and
their ordering against the next P graph. Until that path exists, fallback
recomputes the prefix instead of borrowing the SSD-tiering transport.

### Finite GPU HBM ownership

The HBF adapter does not maintain a shadow HBM counter. Its ownership events
are applied to the real scheduler `MemoryModel`. For P/D calls, the decode
allocation is block-rounded and reserved before the prefill request is made
runnable. A successful P-to-D handoff transfers that reservation to the
decode scheduler; cancellation releases it. This prevents a P graph from
finishing only to discover that D has no space.

Load-time preflight also checks the block-rounded terminal context
(`input + requested output - 1`) against the isolated D-HBM ceiling. A
request that could admit its prompt but can never finish decode is rejected
before adapter or scheduler state is created.

If the FIFO reservation head is blocked only by an idle `GPU_READY` copy
whose HBF migration could not be admitted, the adapter reclaims the oldest
safe copy on that same D instance. Selection is deterministic by
`(last_access_ns, session_id)`. The reclaim emits an `IDLE_RELEASE` into the
real scheduler memory model, moves the lifecycle record to `EVICTED`, and
retries the exact reservation. Active calls and records with a pending
migration or append callback are never eligible. A later resume of the
evicted lineage uses the explicit GPU recomputation path.

The simulation may terminate only when the HBF scheduler, ASTRA dispatches,
lifecycle jobs, GPU ownership events, P-to-D reservations, and native GPU
queues are all quiescent.

## HBF layouts

All layouts consume exactly eight cards. The online CLI accepts `dp8`, `tp4`,
`tp8`, and `tp8_context`; the two primary alternatives are:

| Layout | Execution shape | KV placement | Main tradeoff |
| --- | --- | --- | --- |
| `dp8` | eight independent TP1 replicas | one complete, unreplicated KV record per selected card | Highest request-level concurrency and no collectives, but no single-request model parallelism |
| `tp8_context` | one TP8 replica | each of four KV heads is mapped to a two-card pair; even/odd context tokens are stored uniquely across the pair | Full-card single-request parallelism and no TP8 KV duplication, but one scheduling group plus pair merge and TP collective costs |

Conventional `tp8` also has one TP8 replica, but Qwen's four KV heads are each
replicated across two query ranks. Its physical KV replication factor is
therefore two. `tp8_context` avoids that duplication: both cards in a pair
evaluate the eight query heads associated with their KV head over disjoint
context halves, then merge online-softmax partials. `tp4` provides two TP4
replicas and one unique KV head per rank. The TP4 model weights exist once in
each replica, but a session is assigned to only one replica; its KV is not
mirrored into the other TP4 replica. TP4 weight replication and conventional
TP8 GQA KV-head replication are separate costs.

Compare layouts with the same workload arrivals, measurement cohort, hardware
file, and latency uncertainty band. `dp8` and `tp8_context` have different
batching and concurrency, so an isolated single-call latency comparison is
not a throughput comparison.

## What ASTRA-Sim executes

GPU work continues to use the normal LLMServingSim path: Python emits a text
trace, the Chakra converter creates an execution graph, and ASTRA-Sim returns
the model-completion time.

Full-model HBF work uses a second protocol in the same congestion-aware
ASTRA-Sim process:

1. The HBF serving pool forms a continuous batch under its token, sequence,
   and LPDDR-capacity limits.
2. The H100-derived full-model analytical provider creates an ordered
   per-operation execution plan.
3. The plan is expanded into an `ordered-v2` dependency DAG. Card-local
   kernels fork onto exact physical-card resources; collective and
   online-softmax merge phases join through explicit shared resources.
4. Python sends the DAG with the `hbf-background-v1` interactive command.
5. ASTRA-Sim owns dependency release, named-resource queueing, stage
   completion, and the final callback timestamp.
6. The callback advances the HBF pool and lifecycle; Python cannot substitute
   its own projected completion time.

Migration and append jobs use the same asynchronous control protocol and
resource namespace as HBF model batches. Resource identifiers include the
server, replica, physical card, PCIe root, and route where applicable. This
prevents a request assigned to one replica from accidentally pooling all
eight card interfaces.

### Shared PCIe resource bridge

The foreground HBF collective projector and the lifecycle migration projector
resolve their routes through one validated PCIe topology. In the reference
two-root server:

- TP4 replica 0 owns cards 0–3 on root 0 and replica 1 owns cards 4–7 on
  root 1. Their collectives have disjoint root/card resources.
- Conventional TP8 dense and MoE collectives reserve both roots, all eight
  card interfaces, and the inter-root resource.
- `tp8_context` query and softmax-partial exchanges remain within each
  root-local two-card KV-head pair. Its ordinary TP/EP collectives still
  reserve the inter-root resource.
- An incoming migration resolves the destination card through the same
  card-to-root map. Its destination-root and card-link stages therefore
  contend with a same-root foreground collective.
- A migration whose destination differs from the configured NIC root also
  reserves the NIC root, destination root, and inter-root resource.

The analytical collective duration takes the maximum of its per-card,
aggregate-root, and (for cross-root routes) inter-root bandwidth floors. Fixed
latency is likewise the maximum latency on the collapsed route. Lifecycle
root service uses the slowest configured root/inter-root bandwidth and charges
the collapsed fixed latency once per non-empty destination-root stream. These
are analytical route stages: bytes and dependency edges are conserved, but the
bridge does not synthesize native Chakra PCIe nodes.

Dependency critical path and ASTRA completion are intentionally distinct.
`dependency_critical_path_ns` ignores named-resource calendars. The isolated
ASTRA expectation, `solo_resource_serialized_completion_ns`, applies FIFO
reservations on an initially idle calendar and includes conflicts among
branches of the same job. For example, root-0 and cross-root branches of one
TP8 migration both consume the NIC-root resource and intentionally serialize,
so isolated shared-mode completion can exceed the dependency-only path. This
is physical resource contention in the isolated descriptor order.

The isolated value is a diagnostic counterfactual, not a universal callback
lower bound. Another job can delay one root and thereby change which dependent
stage acquires a shared resource first. That interleaving can reduce internal
serialization in the affected job, making its actual elapsed time smaller than
its isolated descriptor-order completion. ASTRA callbacks are therefore
accepted when they are at least `arrival + dependency_critical_path_ns`;
dependency causality is the universal lower bound.

Runtime metrics split the timing as follows:

- `astra_internal_resource_serialization_wait_ns` is isolated
  resource-serialized completion minus dependency critical path;
- `astra_signed_interference_delta_ns` is actual callback elapsed time minus
  isolated resource-serialized completion. It may be positive or negative;
  a negative value records beneficial event reordering, not negative service;
- the compatibility field `astra_resource_delay_ns` is actual callback time
  minus dependency critical path and remains non-negative.

The exact identity is
`astra_resource_delay_ns = astra_internal_resource_serialization_wait_ns +
astra_signed_interference_delta_ns`. Runtime invariants require exact integer
fields and this algebra while allowing the signed delta to be negative.

Foreground projections whose stages do not conflict internally retain
dependency/isolated parity. Lifecycle projections are not required to retain
that parity. The actual no-contention ASTRA smoke is checked against the
isolated named-resource calculation, not against the dependency-only path.

The four root-local `tp8_context` pair exchanges are currently represented by
one synchronized analytical stage. It reserves both roots but not the
inter-root link. This preserves the collective barrier and critical path, but
is conservative when an unrelated migration could otherwise overlap one
idle root while a pair on the other root is active.

Topology validation fails before ASTRA starts if a card or NIC references an
unknown root, if the current server is not exactly two four-card roots, a TP4
replica spans roots, the two TP4 replicas alias one root, a `tp8_context`
KV-head pair spans roots, or a TP8 layout is selected while cross-root P2P is
disabled.

### Causal ordering at equal timestamps

GPU model completions, HBF callbacks, and P-to-D handoffs can occur at the
same analytical timestamp. The full-model path requires the
`post-endpoint-barrier-v1` capability and fails closed when the selected
binary does not advertise it. Their externally visible effects are deferred
through `control-after-endpoints`:

1. ASTRA-Sim drains the complete `EventQueue` list at the timestamp;
2. the congestion-aware frontend reports every ready end and controller
   endpoint before delivering the post-endpoint control callback;
3. Python collects every tied GPU, P, D, and HBF completion;
4. the complete tied set is tested against the measurement cutoff;
5. ownership effects are committed in deterministic request-ID order;
6. nonterminal P-to-D handoffs and successor releases occur only if the
   source remains open; and
7. no new GPU or HBF graph is dispatched while the barrier is pending.

This ordering is required for reproducible SLO-goodput runs. Without it,
ASTRA callback order could decide whether a tied continuation enters the
measured cohort or obtains newly freed HBM.

## Running the integrated smoke

Build ASTRA-Sim and Chakra first. From the repository root, activate the
simulator environment, then run:

```bash
export PYTHONPATH="$PWD/astra-sim/extern/graph_frontend/chakra/build/lib:$PWD/astra-sim/extern/graph_frontend/chakra${PYTHONPATH:+:$PYTHONPATH}"

python -m serving \
    --cluster-config configs/cluster/single_node_qwen3_1m_pd_p4d4_h100.json \
    --dataset workloads/full-model-hbf-lifecycle-smoke.jsonl \
    --num-reqs 1 \
    --network-backend analytical-congestion-aware \
    --latency-model h100-qwen3-tp4-kernel-calibrated \
    --no-enable-prefix-caching \
    --full-model-hbf-config configs/wakekv_hbf/full_model_8card_server.json \
    --full-model-hbf-layout tp8_context \
    --run-id hbf-tp8-context-smoke \
    --output outputs/hbf-tp8-context-smoke.csv \
    --session-metrics outputs/hbf-tp8-context-smoke-session.json \
    --full-model-hbf-metrics outputs/hbf-tp8-context-smoke-runtime.json \
    --log-level WARNING
```

The bundled lifecycle smoke has one first call and two continuations. It
exercises GPU execution, turn migration, an HBF resume, a committed HBF
append, a second HBF resume, and final cleanup. Repeat with
`--full-model-hbf-layout dp8` before comparing the two layouts.

The main HBF batching knobs are:

| Flag | Meaning |
| --- | --- |
| `--full-model-hbf-max-num-batched-tokens` | Total HBF continuous-batch token budget |
| `--full-model-hbf-max-num-seqs` | Maximum live sequences per HBF replica |
| `--full-model-hbf-max-prefill-chunk-tokens` | Maximum fresh prefill tokens contributed by one request and batch |
| `--full-model-hbf-prefill-drain-tail-tokens` | Recent materialized prefix tokens retained in LPDDR after the first token; default 2,048 |
| `--full-model-hbf-prefill-drain-min-tokens` | Minimum contiguous LPDDR range that triggers an active drain; default 4,096 |
| `--full-model-hbf-astra-chunk-bytes` | Maximum transfer chunk used when lifecycle traffic is expanded into ASTRA stages |
| `--latency-model-band` | `fast`, `central`, or `slow` analytical latency uncertainty band |

For throughput experiments, use the session admission flags to define a
Poisson or closed-backlog load and a fixed measurement cohort. Report the
exact mode, seed, active-session limit, warmup, measured completion count,
and whether early stopping was enabled.

## Outputs and required checks

The three output files answer different questions.

### Per-request CSV

`--output` contains ordinary request latency fields, including TTFT, TPOT,
ITL, queue delay, session identity, return type, prefix reuse, and KV-source
accounting. HBF-executed calls use an `instance id` such as `hbf:0`; GPU
fallback calls retain their GPU instance ID.

### Session metrics JSON

`--session-metrics` is the source for comparison-level latency and throughput:

- exact measurement-session IDs and their stable hash;
- initial and resume request groups;
- TTFT, TPOT, ITL, scheduler-wait, and end-to-end distributions;
- measured sessions, requests, prompt tokens, generated tokens, and rates;
- resume groups by return type, residency, and physical KV source; and
- timing, dependency, and reuse-token conservation checks.

Compute SLO-goodput from the measured completion window and the exact request
records using one declared resume-TTFT and TPOT threshold pair. Do not multiply
an isolated latency attainment fraction by an unrelated capacity ceiling.

### Full-model HBF runtime JSON

`--full-model-hbf-metrics` records the mechanism:

- hardware, layout, batching options, model, and P/D instance IDs;
- per-call `execution`, `route_reason`, residency, KV source, and state;
- HBF-pool batch mix, token counts, modeled component time, ASTRA delay,
  attention roof dominance, and LPDDR peaks/deferrals;
- migration, append, active-prefill-drain outcomes and wait time, capacity,
  and ASTRA lifecycle metrics;
- dependency-only, isolated resource-serialized, actual callback, internal
  serialization, and signed interference timing with their definitions
  embedded in `astra_timing_semantics`;
- GPU-HBM allocations, releases, P/D reservations, transfers, cancellations,
  pressure reclaims, and remaining ownership; and
- pending multiplexer jobs and callback counts.

A successful run must finish with no pending adapter work, no pending HBF
dispatch, no retained allocation for an ended session, no P/D reservation,
and zero live LPDDR ownership. Treat any failed invariant as a simulation
failure, even if a CSV was written.

### Live comparison TCO adapter

After a complete live-ASTRA comparison has been collected to compact schema
version 2, adapt one explicit rate and one explicit HBF layout to the physical
TCO sensitivity model:

```bash
python -m serving.live_astra_comparison_tco \
  results/live_astra_hbf_comparison/manifest.json \
  results/live_astra_hbf_comparison/compact_results.json \
  --rate 1.2 \
  --hbf-system hbf_tp8_context
```

The adapter reruns the canonical collector and rejects a compact artifact that
does not exactly match the fresh result. It reads only
`cells[].performance.offered_normalized_output_token_slo_goodput_per_second`
for token economics; request goodput is never substituted. Seeds must be
paired across SSD tiering, the selected HBF layout, and the infinite-HBM
Oracle. Multiple seeds receive a two-sided Student-t 95% interval for each
system's marginal seed sample. The cells are seed-aligned, but these are not
paired-difference or goodput-ratio intervals. A single-seed result explicitly
reports that no interval is available.

The JSON and CSV pin the campaign, manifest, compact result, paired workload
schedule, TraceLab source, simulator implementation, ASTRA binary, canonical
collector, TCO adapter, baseline dual-node cluster and policy, proposed
single-node GPU cluster and HBF config, Oracle cluster and policy, SLOs, rate,
layout, and active-prefill-drain-v2 contract hashes. The capacity disclosure
subtracts the exact per-card model-weight shard and applies the layout's
physical KV replication factor. Thus TP4 has two weight copies but one
physical KV copy per session, conventional TP8 has one weight copy and 2x GQA
KV replication, and TP8-context has one weight copy and one striped KV copy.
The adapter also parses the cluster, SSD-policy, and HBF configs and
cross-checks a semantic deployment snapshot against the generated BOM: two
hosts, sixteen H100s, and sixteen SSDs for the baseline; two hosts, eight
H100s, eight HBF-NPUs, 512 GiB LPDDR, and no SSD for the proposal. Every GPU
server is required to remain exactly 4P+4D (four H100s per instance,
TP4/PP1), Qwen3-30B-A3B-Instruct-2507, bfloat16 weights, automatic KV dtype,
512,000,000,000 bytes of CPU DRAM per host, and 80,000,000,000 bytes of HBM
per H100. The baseline must expose eight 3,840-GB SSDs per host. Because the
HBF config describes accelerator cards rather than its CPU server, the
proposal records a separate explicit BOM assumption of 512,000,000,000 bytes
of host DRAM for the HBF-NPU server. These values and the assumption semantics
are serialized in the report, covered by the deployment snapshot digest, and
embedded in the CSV export. Boolean values are never accepted as integer
counts, sizes, token quantities, TP degree, or replica count.
Every selected cell must report complete uncensored measurement and timing
validation, five verified artifacts, and zero metric cross-check mismatches.
Oracle invariants must pass. HBF pending queues, quarantines, lifecycle jobs,
LPDDR ledgers, HBF reservations, and GPU-HBM bridge ownership must all be
fully drained in both compact validity and the hash-verified raw runtime
report. The Oracle remains performance-only and never receives a BOM or
tokens/dollar value.

## Validation

Run the focused semantic tests before an end-to-end simulation:

```bash
python -m unittest -v \
    tests.test_hbf_full_model_latency \
    tests.test_hbf_full_model_astra \
    tests.test_hbf_full_model_astra_ordered \
    tests.test_hbf_full_model_lifecycle \
    tests.test_hbf_full_model_lifecycle_astra \
    tests.test_hbf_full_model_pool \
    tests.test_hbf_online_adapter \
    tests.test_hbf_online_runtime \
    tests.test_hbf_gpu_hbm_bridge \
    tests.test_router_full_model_hbf \
    tests.test_hbf_main_loop_contract \
    tests.test_controller_protocol
```

Then validate both layouts at three scales:

1. **One-session lifecycle smoke:** confirm two actual HBF resumes, one
   committed append, correct TTFT/TPOT, and clean shutdown.
2. **Contention run:** use enough active sessions to queue HBF batches and
   verify nonzero ASTRA resource delay without capacity-accounting failure.
3. **Measurement-cutoff run:** use a fixed admission-order cohort and early
   stopping; verify no post-cutoff successor is launched and every
   pre-cutoff ASTRA obligation drains.

Also check conservation rather than only headline latency:

- offered calls equal GPU calls plus HBF calls;
- HBF requests equal HBF completions after drain;
- ASTRA dispatches equal callbacks;
- migration and append reservations are either committed or cancelled;
- GPU-HBM bytes allocated equal bytes released plus explicitly live
  ownership; and
- per-card HBF and LPDDR peaks never exceed configured usable capacity.

## Balanced high-rate stress profile

The original `tracelab-headline-1741-balanced-v1` publication profile remains
the finite 0.25--1.2 sessions/s storage-onset experiment. Do not extend its
rate cap or reinterpret its results as a compute-saturation sweep. The
separate factory
`serving.core.live_balanced_stress_scenario:build` defines the high-rate
stress matrix:

| External sessions/s | Warmup epochs | Measurement epochs | Guard epochs |
| ---: | ---: | ---: | ---: |
| 1.4 | 497 | 16 | 497 |
| 1.6 | 573 | 16 | 573 |
| 2.2 | 784 | 16 | 784 |
| 2.8 | 1,008 | 16 | 1,008 |
| 3.0 | 1,078 | 16 | 1,078 |

The initially proposed warmup/guard lower bounds were
490/560/769/979/1,049 epochs. They are not sufficient for every exact
seed-101--105 Poisson draw. The selected counts above are the first counts at
or above those bounds for which both arrival spans cover 105% of the
2,329.224-second maximum recorded tool gap under a zero-service schedule.
This is a screening bound, not proof about a loaded system: a live successor
is released only after the preceding request completes and its tool duration
elapses.

Each epoch repeats the same seven complete, content-addressed TraceLab
sessions as the publication profile. It contains seven first calls and seven
resumes. The 16 rate-invariant measurement epochs therefore contain exactly
112 complete sessions and 224 requests, split 112:112 between first and
resume calls. No input, output, prefix coordinate, call, or tool gap is
rewritten.

The factory fails closed unless, for every rate and seed, the zero-service
screen satisfies:

- warmup and guard meet the 5% arrival-span margin;
- all 224 statically projected request releases occur before the last guard
  offer;
- all 112 statically projected resume returns observe their sticky baseline node above
  usable D-HBM plus host-DRAM capacity;
- both node-local finite peaks fit the configured SSD capacity; and
- the aggregate finite peak fits TP4, conventional TP8, and TP8-context HBF
  capacity.

The runner records the last external guard offer for every seed/rate in the
content-addressed campaign identity and checks that it equals the final
scheduled session arrival. The strict collector then reads the native
`requests.csv` arrival timestamp, which includes preceding live completion
and tool delay, for all 112 measured resumes. A stress cell is rejected unless
all 112 arrivals are no later than that pinned guard offer. The compact result
records the cutoff, latest resume arrival, remaining margin, and the exact
112/112 count.

Runtime SSD traffic, queueing, TTFT, TPOT, goodput, and exact full drain still
come from the live LLMServingSim/Chakra/ASTRA cell. Offered order uses only
scenario identity, seed, role, role-local epoch, and complete-session identity;
it never uses a future output, call index, runtime completion, queue, or
placement decision.

First materialize and inspect the seed-101 screen without launching a
simulation:

```bash
python -m serving.live_astra_comparison_sweep \
    --scenario-factory serving.core.live_balanced_stress_scenario:build \
    --rates 1.4,1.6,2.2,2.8,3.0 \
    --seeds 101 \
    --systems ssd_tiering,oracle,hbf_tp4,hbf_tp8,hbf_tp8_context \
    --output-root results/hbf-balanced-stress-screen \
    --inputs-root /dev/shm/llmsim-balanced-stress-screen \
    --dry-run
```

After the dry-run manifest and workload hashes pass review, run seed 101
through all systems as a cost and mechanism screen. Only then confirm retained
rates with `--seeds 101,102,103,104,105`. Report the result as the maximum
observed on this finite grid unless an independently defined stability test
establishes sustainable throughput. If seed 101 fails the live-arrival guard,
first report the measured late-arrival count and deficit; do not enlarge the
guard until that evidence supports a new preregistered schedule.

The sweep passes `--log-interval 60.0` to every serving process by default.
`--log-interval` accepts only a positive finite number and is pinned in the
campaign identity, per-cell manifest, command record, and result provenance.
Changing it changes the progress-log and stdout throughput-bucket frequency;
comparison request/session metrics continue to come from the native request
CSV and strict session/runtime reports.

## Structural limitations

The integration is causal and executable, but it is not a physical HBF
endpoint implementation inside Chakra.

### Interactive named-resource DAG, not native Chakra HBF nodes

`hbf-background-v1` is an `InteractiveControl` DAG with named resources. It
shares ASTRA-Sim's event timeline and ASTRA owns its dependencies, queues, and
callbacks. HBF stages are not Chakra `COMP_NODE`, `COMM_COLL_NODE`, or memory
endpoint nodes, and the HBF server is not yet instantiated as a native
ASTRA-Sim PCIe/RDMA topology.

Only stages that name the same resource contend with one another. The shared
bridge above covers HBF foreground collectives and HBF lifecycle migrations,
including their card, root, and inter-root PCIe contention. A native GPU
Chakra communication and an HBF interactive stage still do not automatically
share bandwidth merely because both are in one ASTRA process. Claims about
GPU-Chakra-to-HBF PCIe or RDMA contention still require a future common
native-topology implementation.

### Stage-count growth

`ordered-v2` preserves per-operation order and forks work across physical
cards. A TP8 context-striped batch can expand to thousands of ASTRA stages.
This fidelity is useful for smoke and stress validation but makes long
TraceLab rate sweeps expensive. Any future fusion or coarsening must preserve
dependencies, critical paths, byte totals, resource occupancy, and completion
ordering, and must be validated against unfused runs before it is used for
results.

### Analytical device latency

The NPU is modeled with an H100-derived analytical kernel model and
parameterized NPU compute, HBF bandwidth, and LPDDR bandwidth. ASTRA-Sim
schedules those stage runtimes; it does not derive them from a cycle-level HBF
device model. The result is suitable for controlled sensitivity studies, not
a claim of measured HBF-NPU silicon performance. New hardware claims require
calibration data and uncertainty reporting.

### Missing D-to-P restore

When a cold record is not committed or cannot be admitted to HBF, the current
fallback recomputes on the GPU P path. There is no modeled D-to-P KV restore.
Adding it requires a finite P-HBM destination reservation, an explicit
transfer route, overlap policy, and same-time ordering with P dispatch.

### Incomplete power and TCO coupling

The existing online power model covers GPU instances, not the HBF NPU, HBF
media, LPDDR, PCIe, or RDMA components. Full-model HBF therefore rejects
cluster power modeling. TCO and energy analysis must remain a separately
declared sensitivity until those components and their utilization-dependent
power are integrated.

These limits are fundamental modeling boundaries, not liveness exceptions.
The current implementation has one causal online ASTRA process for GPU
graphs and HBF jobs, finite-capacity ownership, ordered completion, and
metrics conservation within the stated named-resource model. It should not
be described as a native heterogeneous Chakra topology until the named HBF,
PCIe, and RDMA resources are replaced by physical ASTRA endpoints and links.

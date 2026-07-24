# TraceLab P8+D8 pre-admission restore sensitivity

This directory is the compact result artifact for the request-local cold-KV
capacity replay at simulator commit
`1de9618e8fbcc215713cf1a267b7c6df286005a1`. It uses the corrected
55.2/33.6 GB/s aggregate SSD read/write point and the prompt-plus-output
context-admissibility contract. The older 400/400 GB/s and input-only context
results remain recoverable from Git history and must not be mixed with these
tables.

## Scope and contract

- Swap-out is asynchronous, background, and nonpreemptive. It never creates a
  model-engine barrier, but it consumes transfer queues and retains the source
  allocation until commit.
- A cold continuation reserves its complete destination HBM footprint before
  loading. That owner cannot enter analytical prompt compute until every KV
  byte arrives; unrelated requests remain runnable.
- Every SSD restore is two serial stages: SSD to transient CPU DRAM, then CPU
  DRAM to decode HBM over the per-GPU PCIe links. `hbm_ssd_direct` has no
  persistent CPU cache or bounce-free/GDS path.
- The replay has independent P8 and D8 HBM pools, a global first-arrival clock,
  closed-loop session dependencies, recorded internal gaps, and transfer
  queues. It has no decode compute server, shared compute queue, or continuous
  batch formation. Its request sums are not cycle-level TTFT, TPOT, throughput,
  or measured utilization.

## Inputs

- Workload: TraceLab converter schema 3, 4,281 sessions and 357,161 calls;
  SHA-256
  `6721238648d49236d692d70ad72ee3666e829ee5530be175644041fe016176dc`.
- Model sensitivity: Llama-3.1-70B, BF16 KV, TP8, 16-token KV blocks, and a
  131,072-token prompt-plus-output limit.
- HBM: 80 GB SI/rank, 8 GB SI/rank static reserve, and a
  17,640,734,720-byte/rank architecture-derived weight estimate. The KV budget
  is 54,359,265,280 bytes/rank.
- CPU: exactly 2 TB SI persistent KV capacity, 50 GB/s effective PCIe per GPU,
  and one 400 GB/s node-shared half-duplex DRAM queue.
- SSD: eight 3.84-TB devices, 30.72 TB SI total. The configured 55.2/33.6 GB/s
  read/write values are the ideal sum of eight KIOXIA CM6 3.84-TB
  manufacturer ratings (6.9/4.2 GB/s each), not measured RAID 0 throughput.
  Replace them with target-host `fio` results for a hardware claim.
- P/D link: 50 GB/s/rank plus 10 microseconds; capacity-only whole-session LRU,
  full-object SSD writes, and `async-pre-admission` restore.

## Context coverage

Context admissibility is `input_toks + output_toks <= 131072`. Exactly 168,520
calls (47.183203% of all calls) are infeasible for Llama-3.1-70B, leaving
188,641 executable calls, 178,483 selected positive-gap transitions, and
175,626 reuse-eligible transitions. Reuse is `explicit_estimated`, not a
target-tokenizer-exact LCP.

Over-context calls remain in the literal all-request denominator, clear cache
lineage, and have zero target-model service time. Therefore this artifact is a
**censored Llama-128K placement/capacity sensitivity**, not a full-TraceLab
latency or throughput result. A paper headline requires a validated model path
that executes the complete population, or an explicit compaction policy.

## Placement result

The requested primary denominator is every one of the 357,161 calls, including
initial calls, no-reuse calls, and context-infeasible calls.

| Baseline | Decode HBM | CPU | SSD | Recompute | CPU+SSD / all |
|---|---:|---:|---:|---:|---:|
| HBM-LRU-Recompute | 69,723 | 0 | 0 | 105,903 (29.6513%) | 0% |
| HBM-SSD-Direct | 52,403 | 0 | 123,223 (34.5007%) | 0 | 34.5007% |
| HBM-CPU-SSD | 73,360 | 9,652 (2.7024%) | 92,614 (25.9306%) | 0 | 28.6330% |

On the context-admissible denominator, the corresponding opportunity is
56.1400% recomputation, 65.3214% direct SSD restore, and 54.2120% tiered
CPU-or-SSD restore. The HBM-only baseline spends 89.5821% of executed
analytical prompt compute on recomputed prefixes. That denominator excludes
decode and calibrated kernel/collective time.

At the tiered point, 77.86% of reuse-eligible human returns and 56.91% of
reuse-eligible tool returns come from CPU or SSD. `return_sources.csv`
preserves exact human, mixed, and tool counts with both the literal class and
reuse-eligible denominators.

## Request-local overhead and oracle comparison

All values below are request sums. Restore gate, P/D branch admission, queue,
and service intervals overlap and must not be added. The paired infinite-HBM
oracle preserves mandatory P/D traffic and the same analytical compute model.

| Baseline | Restore gate | Aggregate P/D HBM admission | Foreground restore queue | All transfer queue / service | Finite / oracle service | Service slowdown | Session-E2E slowdown | Trace makespan slowdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HBM-LRU-Recompute | 4,966 s | 10,818,308 s | 241 s | 1,552 / 11,306 s | 5,879,168 / 58,528 s | +9,945.0% | +5.082% | +0.050985% |
| HBM-SSD-Direct | 55,307,814 s | 104,895,248 s | 219,805 s | 1,244,217 / 157,343 s | 57,991,282 / 58,528 s | +98,982.5% | +50.581% | +0.858167% |
| HBM-CPU-SSD | 71,839,480 s | 137,975,288 s | 232,855 s | 1,038,446 / 127,605 s | 74,823,592 / 58,528 s | +127,741.7% | +65.277% | +0.030010% |

The exposed restore gate is 95.37% of direct finite request service and 96.01%
of tiered finite request service. This is much larger than the 25% swap-wait
opportunity cited by synchronous-swap studies, but it is not a matching
end-to-end denominator: this replay is saturated by background full-object
writes and lacks decode/continuous-batching compute.

The corrected SSD point materially changes the endogenous placement. Direct
SSD writes accumulate 1,023,946.717 seconds of queue wait, while tiering shifts
large background waits to CPU-to-SSD (495,618.354 seconds) and HBM-to-CPU
(308,655.449 seconds). Slow demotion keeps source HBM authoritative longer,
which changes later admissions and resume sources rather than merely scaling
an isolated copy latency.

Full-trace makespan remains heavily diluted because the offered-call activity
complement is 99.08%, 98.31%, and 97.77% for the three rows. These are gaps in
the analytical open/closed-loop trace, not GPU idle fractions. Session-E2E and
request-summed service expose the pressure much more clearly, but neither is a
substitute for the main online simulator.

`primary_summary.csv`, `transfer_stages.csv`, and `return_sources.csv` contain
the exact values and report hashes. The superseded legacy-bandwidth CSV was
removed from this current-contract artifact; it is recoverable from commit
`02c9414`.

## Reproduction

Run once for each `POLICY` in `hbm_lru_recompute`, `hbm_ssd_direct`, and
`tiered`:

```bash
python -m serving.agentic_kv_capacity_replay \
  --workload /tmp/tracelab-schema3-sps0.2.jsonl \
  --model meta-llama/Llama-3.1-70B --hardware H100 --tp-size 8 \
  --kv-dtype-bytes 2 --block-size 16 --max-context-tokens 131072 \
  --hbm-capacity-gb-per-rank 80 \
  --hbm-static-reserve-gib-per-rank 7.450580596923828 \
  --cpu-kv-budget-gib 1862.645149230957 --ssd-kv-budget-tb 30.72 \
  --policy POLICY --demotion-mode capacity-only \
  --cpu-rank-gbps 50 --cpu-aggregate-gbps 400 \
  --ssd-read-gbps 55.2 --ssd-write-gbps 33.6 \
  --pd-link-gbps-per-rank 50 --pd-fixed-latency-us 10 \
  --restore-execution-mode async-pre-admission \
  --compare-infinite-hbm-oracle --output /tmp/POLICY.json
```

The paired oracle is authoritative only in aggregate. Changed shared-fabric
ordering means it is not a strict per-call lower bound.

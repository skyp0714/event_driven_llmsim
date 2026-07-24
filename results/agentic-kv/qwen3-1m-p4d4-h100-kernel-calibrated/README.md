# Qwen3 1M P4+D4 legacy-H100-labeled sensitivity

This artifact evaluates cold-session KV placement with Qwen3-30B-A3B,
TP4/EP4 prefill plus TP4/EP4 decode, a 1,010,000-token context limit, and
three capacity-only baselines:

1. `hbm_lru_recompute`: HBM LRU drop, then recompute on a miss.
2. `hbm_ssd_direct`: HBM LRU demotion to SSD, with no CPU cache tier; SSD
   restore is physically staged through CPU DRAM before CPU-to-GPU PCIe.
3. `tiered`: HBM LRU to CPU, CPU LRU to SSD, then drop/recompute after SSD.

The result is a **legacy-H100-labeled analytical sensitivity**, not an
absolute DGX-H100 or measured Qwen result. The authoritative root
[`manifest.json`](manifest.json) verifies 12 shard manifests, 36 schema-15
reports, the report-derived combined tables, workload provenance, source
hashes, calibration metadata, resolved configs, and paired infinite-HBM
oracles.

## Immutable bindings

- Workload: 4,281 sessions and 357,161 calls, SHA-256
  `b6188582aac9467cee8c73e4275f9a9606b359f8c2fa000d9f49a9ca3bde02f0`.
- Local Qwen simulator config: SHA-256
  `0736f3bafb95cc658776c92ab34f5f98d8169634a3a36b84892d71448a1f19d3`.
- Base calibration metadata: SHA-256
  `0097dfd92fa7e1d4de9a357d507fd6a019c40a86dba8906d7abc6c4b909a4d3b`.
- Shard source commit: `5bcc8740ab1e88e017d57745d85cfb922517a824`.
  The runs used a dirty worktree because unrelated pre-existing files were
  present. The manifest's per-file SHA-256 bindings, rather than the commit
  alone, identify the code and inputs used by the replay.
- All-request denominator: all 357,161 calls, including first calls and
  calls without reusable KV. No call is excluded by the 1.01M context limit.

The four raw fit sources are the TP4 `layers.csv` and `attention.csv` files
for legacy Llama-3.1-70B and Mixtral-8x7B under
`profiler/v0/perf_models/H100/`. Their hashes are recorded in the root and
shard manifests. The manifests also hash the current legacy producer source,
but the CSVs do not identify the exact H100 SKU, form factor, clock state,
CUDA/PyTorch/FlashAttention versions, command, or producer commit. The
current producer hashes therefore do not prove the revision that generated
the measurements.

## Kernel calibration

The adapted KernelSight-LM structure is evaluated per kernel family:

```text
t = max(t0, t_roof * eta)
t_roof = max((F / P_peak) * u, bytes / BW_peak)
u = ceil(thread_blocks / num_sms) * num_sms / thread_blocks
```

The model uses Qwen TP4/EP4 FLOPs, tensor bytes, kernel identity, launch
shape, attention causal-pair geometry, expert-token pairs, and an attention
SM-wave proxy. It does not predict from FLOPs alone. Each compatible legacy
family fits a launch floor and p10/p50/p90 `eta`; p50 is the central endpoint
and p10/p90 are labeled sensitivities. TP/EP collective terms use an
unmeasured analytical 450 GB/s plus 3 microseconds per collective.

The component holdout has row-weighted MAPE 8.231%. Standard prefill
attention has 4.909% MAPE and 9.217% p90 absolute percentage error over 16
held-out short-range points. These are contiguous short-range extrapolation
and source-domain holdout checks, not Qwen end-to-end or 1M validation.
Examples of weaker transferred families are router MAPE 32.055%, RoPE
20.111%, and Q projection 12.249%.

The analytical full-prompt endpoints at 1,010,000 tokens are:

| Endpoint | Total | Attention | Non-attention | TP all-reduce | EP AG + RS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Central full causal | 795.829 s | 785.401 s | 10.428 s | 0.663 s | 0.685 s |
| Central attention / 3 | 272.228 s | 261.800 s | 10.428 s | 0.663 s | 0.685 s |
| Fast full causal | 485.122 s | 475.533 s | 9.590 s | 0.663 s | 0.685 s |
| Slow full causal | 934.406 s | 922.993 s | 11.413 s | 0.663 s | 0.685 s |

`Non-attention` already includes the TP and EP collective terms. The final two
columns are subsets provided for diagnosis, so the table columns are not all
additive.

Every point extrapolates beyond legacy attention `q <= 1,024`, `k <= 2,016`
and dense `M <= 2,048`. The attention `/3` point changes only prefill
attention and is a DCA/MInference sensitivity, not a measured optimization.

## Primary full-reserve result

The primary endpoint is central full-causal attention with the full inferred
runtime/activation reserve. Source fractions below use **all 357,161 calls**.
Slowdown is `(finite - paired infinite-HBM) / paired infinite-HBM`.

| Baseline | D-HBM / all | CPU / all | SSD / all | Recompute / all | Recompute / prompt compute | Request-service slowdown | Session-E2E slowdown | Trace-makespan slowdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HBM-LRU-Recompute | 11.846% | 0% | 0% | 82.899% | 96.460% | +146,837.7% | +377.398% | +36.714% |
| HBM-SSD-Direct | 31.321% | 0% | 63.424% | 0% | 0% | +8,846.9% | +22.738% | +0.437% |
| HBM-CPU-SSD | 49.680% | 24.955% | 20.111% | 0% | 0% | +5,467.6% | +14.053% | +0.192% |

Thus direct SSD resumes 63.424% of all requests. Tiering resumes 24.955%
from CPU and 20.111% from SSD, or 45.065% combined. For tiering, the
all-request-class CPU/SSD fractions are 38.625%/21.921% for human returns and
24.106%/20.536% for tool returns.

The recomputation numerator is the incremental counterfactual lost-prefix
cost: full-prompt prediction minus cached-prefix prediction, bounded to
`[0, full_prompt]`. Its 96.460% denominator is all configured analytical
prompt-compute seconds executed and includes the predictor's kernel and
collective terms. It excludes KV transfer, decode, host, scheduler, and batch
formation.

For direct SSD, the request-summed exposed restore/admission gate is
25,226,110 seconds, 95.583% of finite request-summed ready-to-complete time.
For tiering it is 15,794,368 seconds, or 96.170%. These are overlapping
request-local sums that include HBM admission, not global batch-blocking swap
fractions. Their wall-clock exposed unions are only 1.720% and 1.248% of the
respective trace makespans.

The offered-load replay has long TraceLab gaps: the no-active-call fractions
are 64.332% for recompute, 96.444% for direct SSD, and 96.670% for tiering.
They are not GPU utilization. This explains why request-local/service and
session overhead can be large while full-trace makespan overhead is small.

## Capacity sensitivity

Central full-causal results across the runtime-reserve sweep are:

| Reserve | Recompute / all | Direct SSD / all | Tiered CPU / all | Tiered SSD / all | Recompute session slowdown | Direct session slowdown | Tiered session slowdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero | 81.850% | 59.829% | 17.602% | 17.133% | +227.244% | +21.837% | +11.619% |
| Half | 82.296% | 61.056% | 19.875% | 20.026% | +289.154% | +22.112% | +13.875% |
| Full | 82.899% | 63.424% | 24.955% | 20.111% | +377.398% | +22.738% | +14.053% |

More non-KV reserve leaves less HBM for KV, so the full-reserve case is the
highest-pressure case.

## Interpretation boundary

- Swap-out is asynchronous and retains the source until copy commit.
- Swap-in reserves destination HBM and gates only the returning request until
  the complete restore finishes; unrelated calls remain runnable.
- SSD restore is `SSD -> CPU staging -> GPU`. The direct-SSD baseline has no
  reusable CPU cache tier, but it still uses the physical staging path.
- The replay has no shared LLM compute queue, continuous-batch formation,
  decode compute, host launch/sampling, measured DCA kernel, or online TTFT,
  TPOT, and throughput. Its service and session metrics must not be relabeled
  as those online metrics.
- The recompute and restore percentages use different denominators from
  CacheTTL and InferCept. They demonstrate a large cold-KV opportunity under
  this 1M analytical sensitivity, not replication of those papers.

## Reproduction

Each shard was run independently, then collected only after every report was
complete. The 230 MB converted workload is intentionally not checked into the
repository; reproduce or supply the exact bytes with workload SHA-256
`b6188582aac9467cee8c73e4275f9a9606b359f8c2fa000d9f49a9ca3bde02f0` and
the schema-3 generation manifest described by the root manifest. The command
below is therefore an execution recipe after that external input is restored,
not a self-contained data-fetch recipe:

```bash
root=results/agentic-kv/qwen3-1m-p4d4-h100-kernel-calibrated
for reserve in zero half full; do
  for endpoint in \
    central_full_attention central_attention_one_third \
    fast_full_attention slow_full_attention; do
    python -m serving.agentic_kv_qwen3_1m_p4d4 \
      --workload /path/to/tracelab-schema3-sps0.2-final.jsonl \
      --workload-manifest /path/to/tracelab-schema3-sps0.2-final.jsonl.manifest.json \
      --output-dir "$root/parts/$reserve-$endpoint" \
      --reserve-sweep "$reserve" \
      --compute-endpoint "$endpoint"
  done
done

python -m serving.agentic_kv_qwen3_1m_collect --root "$root"
```

Use [`summary.csv`](summary.csv) for the 36 headline rows,
[`return_sources.csv`](return_sources.csv) for human/tool source incidence,
and [`transfer_stages.csv`](transfer_stages.csv) for stage-level bytes and
queue/service time. The full report JSONs remain authoritative.

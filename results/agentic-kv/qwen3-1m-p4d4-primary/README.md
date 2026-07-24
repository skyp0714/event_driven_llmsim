# Qwen3 1M P4+D4 primary capacity replay

This artifact replays the unfiltered TraceLab schema-v3 workload with
Qwen3-30B-A3B-Instruct-2507, one TP4 prefill role and one TP4 decode role on a
single DGX H100. It is a capacity, communication, and transfer-queue
sensitivity. It is not a measured H100 DCA/MInference latency result and does
not model a shared compute queue, continuous batch formation, decode compute,
TTFT, or TPOT.

The run used Git commit `42a84b9ccff12a52ebfafa5f900e0d28e3015a8f`
from a clean detached worktree. `manifest.json` records the exact command,
raw-source/converter/workload hashes, conversion arguments and warnings,
official Qwen artifact hashes, excluded repository profiles, model geometry,
reserve derivation, hardware inputs, and every report/table hash.

## Workload and capacity contract

- 4,281 sessions and 357,161 LLM calls; all calls are in the denominator.
- No request is padded to 1M and no request is context-filtered under the
  source-reported token counts. These are provider-tokenizer surrogates, not
  exact Qwen token IDs.
- Workload SHA-256 is
  `b6188582aac9467cee8c73e4275f9a9606b359f8c2fa000d9f49a9ca3bde02f0`.
  The embedded schema-3 sidecar binds raw TraceLab v0.0.1 SHA-256
  `9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b`
  and records `passed_with_warnings`: 44,464 warnings in eight explicit
  categories, including fallback timing and lineage breaks.
- Input tokens: p50 124,018; p90 256,767; p99 822,895; max 999,888.
- Prompt plus output: p50 124,536; p90 257,628; p99 823,622; max 1,006,146.
- 34,954 calls (9.7866%) exceed the model's native 262,144-token window and
  exercise the declared 1,010,000-token path.
- BF16 KV at TP4 is exactly 24,576 bytes/token/rank, or 24,821,760,000 bytes
  per rank at 1.01M tokens.
- Per-rank HBM is 80,000,000,000 bytes. The core static reserve is
  19,893,012,480 bytes/rank, derived from the official approximate 240 GB
  whole-engine statement after reconciling the checkpoint and simulator weight
  estimates. The same conservative residual is applied independently to P and
  D. It is a labeled sensitivity, not a measured role-specific workspace size.
- CPU DRAM is 2,000,000,000,000 bytes; SSD is 30,720,000,000,000 bytes.
- CPU/GPU PCIe is 50 GB/s/rank, shared DRAM is 400 GB/s, direct P/D peer
  transfer is 450 GB/s/rank one-way plus 3 microseconds, and eight-drive SSD
  read/write is 55.2/33.6 GB/s. These are nominal or
  manufacturer-upper-bound inputs, not platform measurements.

Every row uses capacity-only LRU, asynchronous swap-out, request-local
pre-admission restore, D-HBM reservation before load, and an SSD-to-CPU-to-GPU
read path. The direct-SSD row has no persistent CPU cache tier.

## Headline results

All source fractions use all 357,161 LLM calls as the denominator. The residual
to 100% is session starts or a non-reuse-eligible/non-selected continuation.
Slowdown is relative to the paired infinite-HBM-residency replay with the same
P/D transfers and compute endpoint.

| Compute endpoint | Baseline | HBM | CPU | SSD | Recompute | Recompute / executed prompt compute | Session E2E slowdown | Trace-makespan slowdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-causal roofline | HBM-LRU-Recompute | 17.78% | 0% | 0% | 76.97% | 97.08% | 50.99% | 3.196% |
| Full-causal roofline | HBM-SSD-Direct | 33.26% | 0% | 61.48% | 0% | 0% | 22.48% | 0.430% |
| Full-causal roofline | HBM-CPU-SSD | 57.77% | 18.42% | 18.56% | 0% | 0% | 12.69% | 0.179% |
| Whole-prompt roofline / 3 | HBM-LRU-Recompute | 50.23% | 0% | 0% | 44.51% | 95.62% | 3.62% | 0.025% |
| Whole-prompt roofline / 3 | HBM-SSD-Direct | 33.15% | 0% | 61.59% | 0% | 0% | 22.62% | 0.438% |
| Whole-prompt roofline / 3 | HBM-CPU-SSD | 58.35% | 17.79% | 18.60% | 0% | 0% | 12.76% | 0.179% |

The full-causal tiered row splits by return class as follows:

| Return class | Class requests | HBM / all class requests | CPU / all class requests | SSD / all class requests |
| --- | ---: | ---: | ---: | ---: |
| Tool return | 313,179 | 62.96% | 16.79% | 18.88% |
| Human return | 31,997 | 25.74% | 37.60% | 20.89% |
| Mixed return | 3,097 | 30.13% | 37.88% | 14.56% |

The small trace-makespan percentages must not be interpreted as negligible
resume cost. TraceLab contains long closed-loop tool and human waits, so the
wall-clock makespan denominator includes long periods with no runnable call.
The session-E2E and request-local admission metrics preserve the waiting seen
by returning sessions, while request-summed exposure may overlap across
sessions and must not be treated as wall time.

## Reproduction and validation

```bash
python -m serving.agentic_kv_qwen3_1m_p4d4 \
  --workload /path/to/tracelab-schema3-sps0.2.jsonl \
  --workload-manifest /path/to/tracelab-schema3-sps0.2.jsonl.manifest.json \
  --output-dir results/agentic-kv/qwen3-1m-p4d4-primary

PYTHONPATH=. pytest -q \
  tests/test_agentic_kv_qwen3_1m_p4d4.py \
  tests/test_agentic_kv_capacity_replay.py \
  tests/test_runtime_max_model_len.py
```

The completed reports use schema 12. All six reports passed transition
conservation, finite-tier capacity invariants, and paired-oracle validation;
the oracle has no CPU/SSD/recompute source and no capacity action.

The matching future online configs are
`configs/cluster/single_node_qwen3_1m_pd_p4d4_h100.json` and the three files
under `configs/agentic_kv/qwen3_1m_p4d4/`. They use direct-fabric D-to-P
restore. The required measured H100 TP4 DCA profile is not checked in, so this
artifact must not be relabeled as online TTFT, TPOT, or throughput.

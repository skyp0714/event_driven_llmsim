# Qwen3 1M P4+D4 HBM-reserve sensitivity

This artifact varies only the unmeasured static HBM reserve in the Qwen3 1M
P4+D4 capacity replay. It exists to expose how strongly the placement result
depends on deriving one role's runtime residual from the model card's
approximate 240-GB whole-engine statement. It is not a confidence interval or
a measured P/D workspace calibration.

All runs use clean experiment commit
`42a84b9ccff12a52ebfafa5f900e0d28e3015a8f`, workload SHA-256
`b6188582aac9467cee8c73e4275f9a9606b359f8c2fa000d9f49a9ca3bde02f0`,
and the same all-357,161-request denominator. The zero and half directories
contain complete schema-2 manifests and six schema-12 paired reports each.
The full case is the separately checked-in primary artifact.

## Capacity cases

| Case | Static reserve/rank | KV budget/rank | Maximum-length KV objects/role |
| --- | ---: | ---: | ---: |
| Zero residual | 0 B | 64,714,772,480 B | 2.607 |
| Half residual | 9,936,923,136 B | 54,777,849,344 B | 2.207 |
| Full residual (primary) | 19,893,012,480 B | 44,821,760,000 B | 1.806 |

The object count is a capacity ratio, not a scheduler batch-size claim. The
same case is applied independently to P and D; role-specific measured values
remain future calibration work.

## Full-causal analytical endpoint

All percentages except slowdown use all requests as the denominator.

| Reserve | Baseline | HBM | CPU | SSD | Recompute | Session-E2E slowdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Zero | HBM-LRU-Recompute | 23.65% | 0% | 0% | 71.09% | 21.59% |
| Zero | HBM-SSD-Direct | 37.08% | 0% | 57.66% | 0% | 20.17% |
| Zero | HBM-CPU-SSD | 63.77% | 13.94% | 17.03% | 0% | 11.68% |
| Half | HBM-LRU-Recompute | 21.23% | 0% | 0% | 73.52% | 32.11% |
| Half | HBM-SSD-Direct | 34.74% | 0% | 60.00% | 0% | 22.05% |
| Half | HBM-CPU-SSD | 60.68% | 16.51% | 17.55% | 0% | 12.08% |
| Full | HBM-LRU-Recompute | 17.78% | 0% | 0% | 76.97% | 50.99% |
| Full | HBM-SSD-Direct | 33.26% | 0% | 61.48% | 0% | 22.48% |
| Full | HBM-CPU-SSD | 57.77% | 18.42% | 18.56% | 0% | 12.69% |

## Whole-prompt roofline / 3 endpoint

| Reserve | Baseline | HBM | CPU | SSD | Recompute | Session-E2E slowdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Zero | HBM-LRU-Recompute | 76.00% | 0% | 0% | 18.75% | 0.27% |
| Zero | HBM-SSD-Direct | 35.35% | 0% | 59.40% | 0% | 21.80% |
| Zero | HBM-CPU-SSD | 62.13% | 15.03% | 17.59% | 0% | 12.13% |
| Half | HBM-LRU-Recompute | 61.61% | 0% | 0% | 33.13% | 1.41% |
| Half | HBM-SSD-Direct | 34.75% | 0% | 59.99% | 0% | 22.02% |
| Half | HBM-CPU-SSD | 60.88% | 15.67% | 18.20% | 0% | 12.46% |
| Full | HBM-LRU-Recompute | 50.23% | 0% | 0% | 44.51% | 3.62% |
| Full | HBM-SSD-Direct | 33.15% | 0% | 61.59% | 0% | 22.62% |
| Full | HBM-CPU-SSD | 58.35% | 17.79% | 18.60% | 0% | 12.76% |

The recomputation row is reserve-sensitive because compute changes subsequent
closed-loop release times and therefore placement. The storage conclusion is
more stable: direct SSD remains at 20.17--22.62% session slowdown, while
tiering remains at 11.68--12.76%. CPU+SSD serves 30.97--36.97% of all calls
under the full-causal endpoint and 32.62--36.39% under the `/3` endpoint.

These are analytical request/session timing sensitivities. The replay has no
shared compute queue, continuous batch formation, or decode compute; do not
report the values as measured TTFT, TPOT, utilization, or throughput.

## Files and reproduction

- `combined_summary.csv`: all 18 zero/half/full rows, SHA-256
  `d46eb44262b32f79a2056cce8fd5eaf7285cb60c0fd1d45a45307e53aafb0b35`.
- `zero/manifest.json`: SHA-256
  `c3af4ec8d4b232e94e6d4e399008010ad9c9c4d9a4c24a4d8ea012bd4dac6ae0`.
- `half/manifest.json`: SHA-256
  `319a2b795699285162edd96b4181467899859ea3c7a8482199666210f9a7f03d`.
- The full-residual reports and manifest are under
  `../qwen3-1m-p4d4-primary/`.

```bash
python -m serving.agentic_kv_qwen3_1m_p4d4 \
  --workload /path/to/tracelab-schema3-sps0.2.jsonl \
  --workload-manifest /path/to/tracelab-schema3-sps0.2.jsonl.manifest.json \
  --reserve-sweep zero \
  --output-dir results/agentic-kv/qwen3-1m-p4d4-reserve-sensitivity/zero

python -m serving.agentic_kv_qwen3_1m_p4d4 \
  --workload /path/to/tracelab-schema3-sps0.2.jsonl \
  --workload-manifest /path/to/tracelab-schema3-sps0.2.jsonl.manifest.json \
  --reserve-sweep half \
  --output-dir results/agentic-kv/qwen3-1m-p4d4-reserve-sensitivity/half
```

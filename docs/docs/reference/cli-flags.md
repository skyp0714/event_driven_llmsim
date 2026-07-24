---
sidebar_position: 1
title: CLI flags
---

# `python -m serving` CLI flags

Complete reference for every command-line flag accepted by
`python -m serving`. For the conceptual side of each flag (what it
*does* internally), see **[Simulator](/docs/simulator/architecture)**.

## Cluster topology

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--cluster-config` | path | `configs/cluster/single_node_pd_instance.json` | Path to a cluster-config JSON. The default is a same-node prefill/decode pair. See **[Cluster config](./cluster-config)** |
| `--network-backend` | choice | `analytical` | Network simulation backend. `analytical` (fast, no overlap contention), `analytical-congestion-aware` (analytical link contention), or `ns3` (detailed, WIP) |

## Batching and scheduling

These flags are deployment defaults. A cluster config can override the
matching runtime knobs per `instances[i]`; see
**[Cluster config](./cluster-config#runtime-overrides-optional)**.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--max-model-len` | int | model `max_position_embeddings` | Semantic prompt-plus-output context limit; cluster instances may override it independently |
| `--max-num-seqs` | int | `128` | Max sequences in a batch. `0` = unlimited |
| `--max-num-batched-tokens` | int | `2048` | Max tokens per iteration across all requests (token budget) |
| `--long-prefill-token-threshold` | int | `0` | Per-request token cap per step for chunked prefill. `0` = disabled |
| `--enable-chunked-prefill` | bool | `True` | Split long prefill across iterations. Use `--no-enable-chunked-prefill` to disable |
| `--prioritize-prefill` | flag | off | Run prefill before decode in the same iteration |
| `--block-size` | int | `16` | KV cache block size in tokens |
| `--skip-prefill` | flag | off | Skip prefill, run decode only |

## Routing

| Flag | Choices | Default | Description |
| --- | --- | --- | --- |
| `--request-routing-policy` | `LOAD` / `RR` / `RAND` / `CUSTOM` | `LOAD` | Cross-instance request routing |
| `--expert-routing-policy` | `BALANCED` / `RR` / `RAND` / `CUSTOM` | `BALANCED` | MoE expert token routing |
| `--enable-block-copy` | bool | `True` | Replay one block's trace across layers (set False for per-layer EP variance) |

## Precision

| Flag | Choices | Default | Description |
| --- | --- | --- | --- |
| `--dtype` | `float16` / `bfloat16` / `float32` / `fp8` / `int8` | model's `torch_dtype`, fallback `bfloat16` | Model weight dtype |
| `--kv-cache-dtype` | `auto` / `fp8` | `auto` (inherits dtype) | KV cache dtype. `fp8` halves KV memory and selects a `*-kvfp8` profile variant |

## Prefix caching and offloading

| Flag | Default | Description |
| --- | --- | --- |
| `--enable-prefix-caching` | `True` | RadixAttention prefix caching. Use `--no-enable-prefix-caching` to disable |
| `--enable-prefix-sharing` | off | Second-tier prefix pool shared across instances within a node |
| `--prefix-storage` | `None` | Where the second-tier pool lives. `None` / `CPU` / `CXL` |
| `--agentic-kv-policy` | omitted | Idle session-KV policy: `off` / `preserve` / `recompute` / `hbm_lru_recompute` / `hbm_ssd_direct` / `cpu` / `tiered`. When omitted, use the config policy or disable the feature |
| `--agentic-kv-config` | `None` | JSON containing capacity-only vs. TTL demotion mode, request-local `async-pre-admission` (default), optimistic `async-decode-join`, or adverse `sync-engine-barrier` swap execution, active recomputation vs. CPU-swap preemption, link/tier service inputs and capacities, and SSD write mode |
| `--agentic-kv-metrics` | `None` | Write migration, residency, restore, and per-device SSD traffic to JSON |
| `--enable-local-offloading` | off | Weight offloading to NPU (counts weight reads in profiling) |
| `--enable-attn-offloading` | off | Attention computation offloading to PIM |
| `--enable-sub-batch-interleaving` | off | Overlap GPU compute with PIM attention. Requires `--enable-attn-offloading` |

Agentic session-KV tiering requires `--no-enable-prefix-caching`. The simulator
rejects generic Radix caching and session tiering together to prevent double
ownership of the same physical KV blocks. The three capacity-pressure paper
baselines (`hbm_lru_recompute`, `hbm_ssd_direct`, and capacity-only `tiered`)
support layout-compatible same-node P/D. A completed prefill waits in a decode
handoff queue until the same active-HBM admission contract has dropped or
demoted enough decode-side idle LRU state; asynchronous reclaim must complete
before decode allocation. With `pd_peer_transfer_mode="direct-fabric"` and
`--network-backend analytical-congestion-aware`, cold HBM-resident D→P copies
share ASTRA topology links and physical endpoint arbiters with ordinary P→D
and TP/EP traffic. Only the returning owner waits for the completion callback;
unrelated graphs can still dispatch and contend. CPU/SSD/PCIe/DRAM stages stay
on analytical manager calendars. Cross-node D→P remains rejected until
NIC/fabric queue resources are modeled. See the
[agentic idle-KV example](/docs/examples/memory-tiers/agentic-idle-kv-tiering).

## Dataset and output

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset` | path | `None` | JSONL workload file. See **[Workloads → JSONL format](/docs/workloads/jsonl-format)** |
| `--num-reqs` | int | `0` | Entries to load from the dataset (`0` = all). For agentic, each entry is a session |
| `--output` | path | `None` | Per-request CSV output path. Stdout only if `None`. The literal `{run_id}` is replaced with the active run id |
| `--session-arrival-mode` | choice | `trace` | `trace` preserves first-session timestamps; `poisson` regenerates them; `backlog` maintains a closed active population |
| `--session-arrival-rate-sps` | float | `0` | Poisson session rate in sessions/s. Required and positive in `poisson` mode |
| `--session-arrival-seed` | int | `42` | Deterministic Poisson random seed |
| `--max-active-sessions` | int | `0` | Active-session limit `K`. Required and positive for closed `backlog`; optional in `poisson`, where arrivals beyond `K` wait in a FIFO admission backlog |
| `--session-backlog-epochs` | int | `1` | Deterministic passes over session templates in `backlog` mode |
| `--session-warmup-completions` | int | `0` | With `completion_order`, completed sessions excluded before the metrics window. With backlog `admission_order`, fixed epoch-major admission-prefix size `W`, excluded from the measured target |
| `--session-measure-completions` | int | `0` | With `completion_order`, completed sessions included after warmup. With `admission_order`, target size `M` immediately after the fixed `W` prefix. `0` includes every remaining eligible session |
| `--session-measurement-cohort-selection` | choice | `completion_order` | `completion_order` selects sessions after execution by completion order. Backlog-only `admission_order` fixes an excluded `W`-session warmup prefix followed by an `M`-session measured target in deterministic epoch-major admission order; `W + M` must fit the materialized backlog |
| `--session-stop-after-measurement` | bool | `False` | Freeze new session/turn admission at the configured measurement boundary, drain already-dispatched ASTRA work, then censor and release queued/pending P/D and tier state. For `admission_order`, cutoff waits for the complete required `W + M` prefix; use `--no-session-stop-after-measurement` to finish the full finite workload |
| `--session-metrics` | path | `None` | Session admission/E2E, throughput, resume TTFT, TTFT, TPOT, and restore/queue summary JSON. Supports `{run_id}` |

Early-stop cleanup is an asserted ownership transition, not silent process
termination. The final session report contains a `censoring` block with
lifecycle counts, queued/handoff/preallocation audits, scheduler memory before
and after, and the tier-manager drain audit. The run fails if any non-weight
NPU/CPU allocation, idle tier entry, durable SSD record, HBM claim, preparation
lock, restore wait, ASTRA window, or external cold-fabric job remains live.

## Run isolation

Each invocation writes ASTRA-Sim intermediates under a run-specific input
root so parallel simulations do not overwrite each other's generated
configs, traces, or Chakra workloads. Generated text traces are removed
after Chakra conversion by default, and the run-specific input root is
removed after a successful simulation by default.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--run-id` | string | auto-generated | Path-safe id for this simulation run. Used in `astra-sim/inputs/runs/<run-id>` and the `{run_id}` output placeholder |
| `--inputs-root` | path | `astra-sim/inputs/runs/<run-id>` | Override the generated ASTRA-Sim input root, for example to place intermediates on local SSD or tmpfs |
| `--cleanup-inputs` / `--no-cleanup-inputs` | bool | `true` | Remove generated trace files after Chakra conversion and remove the generated run directory after a successful simulation. Use `--no-cleanup-inputs` to preserve traces, Chakra workloads, and input configs for debugging |

## Logging

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--log-interval` | float | `1.0` | Seconds between throughput / memory log lines |
| `--log-level` | choice | `WARNING` | `WARNING` (default) / `INFO` / `DEBUG` |

## Quick reference: which flag for which feature

| Feature | Flag(s) |
| --- | --- |
| Multi-instance (parallelism via cluster config) | (cluster config `num_instances`) |
| Tensor parallel | (cluster config `tp_size`) |
| MoE expert parallel | (cluster config `ep_size`) |
| DP+EP MoE | (cluster config `dp_group`) |
| Prefix caching | `--enable-prefix-caching` (default on), `--enable-prefix-sharing`, `--prefix-storage` |
| Agentic idle-KV tiering | `--no-enable-prefix-caching`, `--agentic-kv-config`, optionally `--agentic-kv-policy` and `--agentic-kv-metrics` |
| Agentic session load | `--session-arrival-mode`, Poisson rate/seed or backlog `--max-active-sessions`, optionally `--session-metrics` |
| Chunked prefill | `--enable-chunked-prefill` (default on), `--long-prefill-token-threshold` |
| PIM attention offload | `--enable-attn-offloading` (cluster config sets `pim_config`) |
| FP8 KV cache | `--kv-cache-dtype fp8` |
| ns3 backend | `--network-backend ns3` |
| Congestion-aware analytical backend | `--network-backend analytical-congestion-aware` |

For the full conceptual treatment of each feature, browse the
**[Simulator](/docs/simulator/architecture)** section. For runnable
examples, see **[Examples](/docs/examples)**.

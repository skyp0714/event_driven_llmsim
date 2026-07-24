---
sidebar_position: 1
title: Cluster config
---

# Cluster config schema

Formal field-by-field schema for the JSON file passed via
`--cluster-config`. For a guided walkthrough with examples, see
**[Examples → Cluster config explained](/docs/examples/cluster-config-explained)**.
This page is the **lookup reference**: every field, every type,
every default.

## File location

Configs live at `configs/cluster/<name>.json`. The simulator reads
the file once at startup and `serving/core/config_builder.py`
generates derived ASTRA-Sim input files (`network.yml`,
`system.json`, `memory_expansion.json`).

## Top-level

```json
{
  "num_nodes": 1,
  "link_bw": 16,
  "link_latency": 20000,
  "nodes": [...],
  "cxl_mem": {...}
}
```

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `num_nodes` | int | ✓ |  | Number of physical nodes in the cluster |
| `link_bw` | float or float[] | ✓ |  | ASTRA-Sim topology link bandwidth in **GB/s**. Scalars apply to every topology dimension; arrays must match the final `network.yml::npus_count` rank |
| `link_bw_unit` | string | optional | legacy `astra_GBps` interpretation | `astra_GBps` preserves historical ASTRA units; `decimal_GBps` makes each declared GB/s exactly one decimal byte/ns and is required by the Qwen3 paper contract |
| `link_latency` | float or float[] | ✓ |  | ASTRA-Sim topology link latency in **ns**. Scalars apply to every topology dimension; arrays must match the final `network.yml::npus_count` rank |
| `nodes` | array | ✓ |  | Length must equal `num_nodes` |
| `cxl_mem` | object | optional | absent | CXL memory expansion (see below) |

Example: if `network.yml` will end up with `npus_count: [4, 2]`, you may set
`link_bw: [900, 100]` and `link_latency: [0, 20000]` to assign different
bandwidth/latency per topology dimension.

With agentic `pd_peer_transfer_mode="direct-fabric"`, these same effective
link-bandwidth and latency values also govern cold HBM-resident D→P transfers.
The mode requires `analytical-congestion-aware`: cold transfers enter ASTRA's
event queue on the physical P/D endpoints and contend with ordinary P→D and
TP/EP communication. CPU/SSD restores do not use these fields; their
PCIe/DRAM/media service remains in the agentic-KV config's analytical resource
calendars.

## `cxl_mem` (top-level, optional)

```json
"cxl_mem": {
  "mem_size": 1024,
  "mem_bw": 60,
  "mem_latency": 250,
  "num_devices": 4
}
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `mem_size` | float | ✓ | Capacity per device in **GiB** (the simulator multiplies by 2^30 bytes) |
| `mem_bw` | float | ✓ | Bandwidth per device in **GB/s** |
| `mem_latency` | float | ✓ | Access latency in **ns** |
| `num_devices` | int | ✓ | Number of CXL devices (`cxl:0` through `cxl:N-1`) |

When present, instances can reference `cxl:N` in their `placement`
field.

## Per-node (`nodes[i]`)

```json
{
  "num_instances": 2,
  "cpu_mem": {"mem_size": 512, "mem_bw": 256, "mem_latency": 0},
  "instances": [...],
  "power": {...},
  "cpu_mem.pim_config": "DDR4_8GB_3200_pim"
}
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `num_instances` | int | ✓ | Number of serving instances on this node |
| `cpu_mem` | object | ✓ | Host CPU memory config (see below) |
| `instances` | array | ✓ | Length must equal `num_instances` |
| `power` | object | optional | Power model config (see below) |

### `cpu_mem`

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `mem_size` | float | ✓ | Host CPU memory capacity in **GiB** (the simulator multiplies by 2^30 bytes) |
| `mem_bw` | float | ✓ | CPU memory bandwidth in **GB/s** |
| `mem_latency` | float | ✓ | CPU memory latency in **ns** |
| `pim_config` | string | optional | Name of a PIM device config in `configs/pim/`. See **[PIM config](./pim-config)** |

### `power` (optional)

Enables the power model on this node. See **[Examples → Power
modeling](/docs/examples/advanced/power-modeling)** for the full
schema. Top-level structure:

```json
"power": {
  "base_node_power": 60,
  "npu": {"<hardware>": {...}},
  "cpu": {...},
  "dram": {...},
  "link": {...},
  "nic": {...},
  "storage": {...}
}
```

| Sub-field | Required | Description |
| --- | --- | --- |
| `base_node_power` | ✓ | Always-on host platform power in **W** |
| `npu.<hardware>.idle_power` | ✓ | NPU idle wattage |
| `npu.<hardware>.standby_power` | ✓ | NPU post-compute standby wattage |
| `npu.<hardware>.active_power` | ✓ | NPU active compute wattage |
| `npu.<hardware>.standby_duration` | ✓ | Time to stay in standby after compute, in **ns** |
| `cpu.idle_power`, `cpu.active_power`, `cpu.util` | ✓ | CPU baseline + utilization fraction |
| `dram.dimm_size`, `dram.idle_power`, `dram.energy_per_bit` | ✓ | DIMM size, idle power, per-bit energy |
| `link.num_links`, `link.idle_power`, `link.energy_per_bit` | ✓ | Network link power |
| `nic.num_nics`, `nic.idle_power` | ✓ | NIC count and baseline |
| `storage.num_devices`, `storage.idle_power` | ✓ | Storage devices |

## Per-instance (`instances[i]`)

```json
{
  "model_name": "Qwen/Qwen3-32B",
  "hardware": "RTXPRO6000",
  "npu_mem": {
    "mem_size": 96,
    "mem_bw": 1597,
    "mem_latency": 0,
    "runtime_reserve_bytes": 0
  },
  "num_npus": 2,
  "tp_size": 2,
  "pp_size": 1,
  "ep_size": 1,
  "dp_group": null,
  "pd_type": null,
  "max_model_len": 262144,
  "max_num_seqs": 128,
  "max_num_batched_tokens": 2048,
  "placement": {...}
}
```

### Required fields

| Field | Type | Description |
| --- | --- | --- |
| `model_name` | string | HF id. Must match a config at `configs/model/<model_name>.json` (see **[Model config](./model-config)**) |
| `hardware` | string | Hardware label. Must match `profiler/perf/<hardware>/` |
| `npu_mem.mem_size` | float | Per-GPU NPU memory in **GiB** (the simulator multiplies by 2^30 bytes) |
| `npu_mem.mem_bw` | float | Per-GPU NPU memory bandwidth in **GB/s** |
| `npu_mem.mem_latency` | float | Per-GPU NPU memory latency in **ns** |
| `pd_type` | string \| null | `"prefill"`, `"decode"`, or `null` (combined) |

`npu_mem.runtime_reserve_bytes` is an optional, non-negative integer byte
count per GPU. It represents serving-runtime and static allocations that are
not model weights or KV blocks. `MemoryModel` preserves the configured
physical total and subtracts this reserve once from allocatable capacity;
the reserve is not inserted into `npu_used` and cannot be evicted or freed.
It must be smaller than `npu_mem.mem_size * 2^30`.

### Parallelism (at least one of `num_npus` / `tp_size`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `num_npus` | int | inferred from `tp_size * pp_size` | Total GPUs for this instance |
| `tp_size` | int | inferred from `num_npus // pp_size` | Tensor-parallel degree |
| `pp_size` | int | `1` | Pipeline-parallel degree |
| `ep_size` | int | `tp_size` (MoE) / `1` (dense) | Expert-parallel degree |
| `dp_group` | string \| null | `null` | Group ID. Instances with the same string share experts via cross-instance ALLTOALL |

**Constraints:**

- `num_npus == tp_size * pp_size` (always)
- Without `dp_group`: `ep_size <= tp_size`
- For MoE: `ep_size` must divide `num_local_experts`

### Runtime overrides (optional)

These fields override the matching `python -m serving` CLI flag for this
instance only. Omitted fields keep the CLI value; for `dtype`, an omitted CLI
value still falls back to the model config's `torch_dtype`.

| Field | Type | CLI fallback | Description |
| --- | --- | --- | --- |
| `max_model_len` | int | `--max-model-len`, then model `max_position_embeddings` | Semantic prompt-plus-output context limit. Must be positive |
| `max_num_seqs` | int | `--max-num-seqs` | Max active sequences for this instance. `0` means unlimited |
| `max_num_batched_tokens` | int | `--max-num-batched-tokens` | Per-iteration token budget for this instance. `0` means unlimited |
| `long_prefill_token_threshold` | int | `--long-prefill-token-threshold` | Per-request chunk cap for chunked prefill |
| `block_size` | int | `--block-size` | KV-cache block size in tokens |
| `dtype` | string | `--dtype` | Weight/profile dtype for this instance |
| `kv_cache_dtype` | string | `--kv-cache-dtype` | KV-cache dtype for memory accounting and profile variant selection |
| `latency_model` | string or null | `--latency-model` | Optional online analytical COMP-node provider. It does not replace ASTRA-Sim communication |
| `latency_model_band` | string | `--latency-model-band` | Analytical provider sensitivity band: `fast`, `central`, or `slow` |
| `enable_chunked_prefill` | bool | `--enable-chunked-prefill` | Enable chunked prefill in this instance's scheduler |
| `enable_prefix_caching` | bool | `--enable-prefix-caching` | Enable this instance's local prefix cache |
| `prioritize_prefill` | bool | `--prioritize-prefill` | Prefer prefill requests when forming batches |
| `enable_local_offloading` | bool | `--enable-local-offloading` | Emit graph conversion with local offloading for this instance |
| `enable_attn_offloading` | bool | `--enable-attn-offloading` | Emit PIM attention offload for this instance |
| `enable_sub_batch_interleaving` | bool | `--enable-sub-batch-interleaving` | Enable sub-batch interleaving for this instance |
| `enable_block_copy` | bool | `--enable-block-copy` | Reuse one block trace across repeated transformer blocks |

`max_model_len` is an admission limit, not a kernel calibration switch. A
value above the model config is scientifically valid only when the selected
model runtime, positional encoding, attention implementation, and profiler
bundle all support that length. The simulator rejects such an override unless
the model config declares an explicit `long_context_experiment` mode and
rejects values above its `validated_runtime_max_model_len`. P/D peers must use
the same value because a session KV lineage moves between them.

The checked-in
`single_node_qwen3_1m_pd_p4d4_h100.json` records the intended one-DGX H100
P4+D4 topology for the Qwen3 1,010,000-token path. Each 80-GB-SI H100 rank
reserves 19,893,012,480 bytes for non-weight runtime/static state. The node
has exactly 512 GB SI of host DRAM (476.837158203125 GiB in the schema) at
200 GB/s. Prefill and decode admit at most 32 and 128 sequences,
respectively. Its fabric input is 450 decimal GB/s/rank plus 1 µs, its
agentic-KV policies use 50 GB/s CPU↔GPU PCIe per rank, and its eight-drive SSD
sensitivity uses 55.2/33.6 GB/s aggregate read/write. The 512-GB host setting
is deliberate capacity pressure, not a claim about the physical DGX H100 host
capacity; the SSD values are manufacturer-limit sums, not measured RAID
throughput.

That config selects `h100-qwen3-tp4-kernel-calibrated`, which supplies GPU
COMP-node times directly to the online trace path while ASTRA-Sim remains the
only authority for TP/EP/P-to-D and direct cold HBM D-to-P communication. The provider fits analytical
kernel/roofline models to legacy H100-labeled measurements and extrapolates
the Qwen TP4/EP4 long-context shapes. It is runnable and provenance-recorded,
but it models standard dense/full attention rather than the official sparse
DCA kernel. It is not an absolute measured Qwen/H100 profile; use the
fast/central/slow bands as sensitivity bounds and retain the machine-readable
`dca_dense_full_attention_sensitivity` limitation in paper claims.

### `placement` (optional)

Per-layer / per-block weight + KV-cache placement rules. See
**[Examples → CXL extended memory](/docs/examples/memory-tiers/cxl-memory)**
for a worked example.

```json
"placement": {
  "default": {"weights": "npu", "kv_loc": "npu", "kv_evict_loc": "cpu"},
  "blocks": [
    {"blocks": "0-3", "weights": "cxl:0", "kv_loc": "npu", "kv_evict_loc": "cpu"}
  ],
  "layers": {
    "embedding": {"weights": "cxl:1", "kv_loc": "npu", "kv_evict_loc": "cpu"}
  }
}
```

| Sub-field | Type | Required | Description |
| --- | --- | --- | --- |
| `default` | object | ✓ | Catch-all rule for layers / blocks not in `blocks` or `layers` |
| `blocks` | array | optional | Per-decoder-block-range overrides |
| `layers` | object | optional | Per-named-layer overrides |

Each rule object has three string fields:

| Field | Allowed values | Description |
| --- | --- | --- |
| `weights` | `npu` / `cpu` / `cxl:<id>` | Where this layer's weights live |
| `kv_loc` | `npu` / `cpu` / `cxl:<id>` | Where active KV blocks live (attention layers only) |
| `kv_evict_loc` | `npu` / `cpu` / `cxl:<id>` | Where evicted KV blocks spill |

`blocks` strings are dash-and-comma-separated ranges:
`"0-3"`, `"4-7"`, `"8,9,10"`, `"11-23"`. Layer-name keys must match
canonical layer names from the architecture YAML.

## Validation rules

- `num_nodes == len(nodes)` and per-node `num_instances == len(instances)`.
- Per-instance `weight_per_gpu * num_npus <= npu_mem.mem_size *
  num_npus` (otherwise startup OOM).
- Hardware folder must exist at `profiler/perf/<hardware>/<model_name>/<variant>/tp<tp_size>/`.
- `dp_group` must be a valid string or `null`.
- All instances within the same `dp_group` must share the same
  `ep_size` and `tp_size`.

## What's next

- **[Model config](./model-config)**: schema for the file
  `model_name` resolves to.
- **[PIM config](./pim-config)**: schema for the file
  `cpu_mem.pim_config` resolves to.

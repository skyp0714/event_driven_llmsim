---
sidebar_position: 3
title: PIM config
---

# PIM config schema

PIM (Processing-In-Memory) device configs live at
`configs/pim/<name>.ini` in **DRAMSim3 INI format**. The
simulator's `pim_model.py` reads these to compute PIM-side attention
latency when `--enable-attn-offloading` is on.

## File location

```
configs/pim/
├── DDR4_8GB_3200_pim.ini
├── HBM2_1GB_2000_pim.ini
├── LPDDR4X_2GB_4266_pim.ini
├── LPDDR5_2GB_6400_pim.ini
└── README.md
```

The cluster config references one of these via the node's
`cpu_mem.pim_config` field (without the `.ini` extension):

```json
"cpu_mem": {
  "mem_size": 512,
  "mem_bw": 256,
  "mem_latency": 0,
  "pim_config": "DDR4_8GB_3200_pim"
}
```

## Bundled configs

| File | Protocol | Capacity | Speed | Notes |
| --- | --- | --- | --- | --- |
| `DDR4_8GB_3200_pim.ini` | DDR4 | 8 GB | 3200 MT/s | Standard DDR4 PIM module |
| `HBM2_1GB_2000_pim.ini` | HBM2 | 1 GB | 2000 MT/s | HBM2 PIM (high-bandwidth) |
| `LPDDR4X_2GB_4266_pim.ini` | LPDDR4X | 2 GB | 4266 MT/s | Mobile-class PIM |
| `LPDDR5_2GB_6400_pim.ini` | LPDDR5 | 2 GB | 6400 MT/s | Mobile-class PIM, faster |

## INI structure

Each PIM config has three sections.

### `[dram_structure]`

```ini
[dram_structure]
protocol = DDR4
bankgroups = 2
banks_per_group = 4
rows = 65536
columns = 1024
device_width = 16
BL = 8
pim_type = SINGLE
```

| Field | Type | Description |
| --- | --- | --- |
| `protocol` | string | DRAM standard. `DDR4`, `DDR5`, `HBM2`, `HBM3`, `LPDDR4`, `LPDDR4X`, `LPDDR5` |
| `bankgroups` | int | Bank groups per device |
| `banks_per_group` | int | Banks per bank group |
| `rows` | int | Rows per bank |
| `columns` | int | Columns per row |
| `device_width` | int | Device data width in bits (typically 4 / 8 / 16) |
| `BL` | int | Burst length |
| `pim_type` | enum | `SINGLE` (one PIM unit per channel) or `DUAL` (two units per channel) |

The simulator computes:

- **Bandwidth** from `device_width × BL × tCK × channel_count`.
- **Capacity** from `rows × columns × device_width × banks × bankgroups`.

### `[timing]`

```ini
[timing]
tCK = 0.63          # clock period in ns
CL = 22             # CAS latency
CWL = 16            # CAS write latency
tRCD = 22           # RAS-to-CAS delay
tRP = 22            # row precharge time
tRAS = 52           # row active time
tRFC = 560          # refresh cycle
tREFI = 12480       # refresh interval
tRRD_S = 9          # row-to-row delay (different bank groups)
tRRD_L = 11         # row-to-row delay (same bank group)
tWTR_S = 4          # write-to-read delay (different bank groups)
tWTR_L = 12         # write-to-read delay (same bank group)
tFAW = 48           # four-activate window
tWR = 24            # write recovery
tRTP = 12           # read-to-precharge delay
tCCD_S = 4          # CAS-to-CAS (different bank groups)
tCCD_L = 8          # CAS-to-CAS (same bank group)
```

All timing parameters are in **clock cycles** unless explicitly named
otherwise (`tCK` is in ns). The full list mirrors DRAMSim3's spec.
The simulator extracts the latency-relevant subset for PIM access
modeling.

For full DRAMSim3 timing semantics, see the [DRAMSim3 docs](https://github.com/umd-memsys/DRAMsim3).

### `[system]`

```ini
[system]
channel_size = 8192
channels = 1
bus_width = 64
address_mapping = rorabgbachco
queue_structure = PER_BANK
row_buf_policy = OPEN_PAGE
```

| Field | Type | Description |
| --- | --- | --- |
| `channel_size` | int | Per-channel capacity in MB |
| `channels` | int | Number of memory channels (PIM compute happens per-channel) |
| `bus_width` | int | Memory bus width in bits |
| `address_mapping` | string | DRAMSim3 address-mapping scheme |
| `queue_structure` | enum | Queueing policy (`PER_BANK`, `PER_CHANNEL`, etc.) |
| `row_buf_policy` | enum | Row buffer policy (`OPEN_PAGE`, `CLOSE_PAGE`) |

`channels` is the most simulator-relevant field: more channels =
more parallel PIM compute per attention step. The trace generator
distributes attention heads across channels for parallel execution.

## Adding a new PIM config

1. Drop a new `.ini` file at `configs/pim/<name>.ini`.
2. Fill the three sections above. Reference the bundled configs for
   the right shape.
3. Reference it from your cluster config:
   `"cpu_mem": {"pim_config": "<name>"}`.
4. Run with `--enable-attn-offloading`.

The DRAMSim3 timing parameters can be sourced from a JEDEC datasheet
or vendor spec for the specific DRAM part you're modeling.

## Where this is used

- **`serving/core/pim_model.py`**: loads the INI and exposes timing
  parameters to the trace generator.
- **`serving/core/trace_generator.py`**: when
  `--enable-attn-offloading` is on, swaps NPU attention for
  PIM attention computed using the loaded model.
- **Power model**: if the cluster config has a `power:` block, PIM
  energy is accounted for via the channel count (one PIM unit per
  channel × per-channel power).

For the full PIM offload mechanics, see
**[Simulator → PIM offload](/docs/simulator/specialized/pim-offload)**.
For a worked example, see
**[Examples → PIM attention offload](/docs/examples/disaggregated/pim-attention-offload)**.

## Gotchas

1. **All four bundled INI files use `pim_type = SINGLE`.** Switching
   to `DUAL` doubles the per-channel PIM compute capacity but also
   needs `pim_type = DUAL` to be supported by the cluster config's
   power model entry.
2. **`channels = N` doesn't mean N independent PIM devices.** The
   simulator models per-channel parallelism within one PIM device.
   For multiple PIM devices, you'd configure multiple nodes, but
   that's a different topology.
3. **The INI is parsed as DRAMSim3 standard.** Don't add custom
   fields the simulator's loader doesn't know about; they'll be
   ignored.

## What's next

- **[Cluster config → `cpu_mem.pim_config`](./cluster-config#cpu_mem)**
  how to wire this file into a cluster.
- **[Simulator → PIM offload](/docs/simulator/specialized/pim-offload)**
  what happens at simulation time.

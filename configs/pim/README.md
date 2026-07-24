# configs/pim

PIM (Processing-In-Memory) device configuration files in DRAMSim3 INI format.
Used by `pim_model.py` to compute PIM attention latency and power parameters.

Enable PIM by setting `pim_config` in the cluster config's `cpu_mem` section and
passing `--enable-attn-offloading` to `python -m serving`.

## Provided configs

| Config | Protocol | Capacity | Speed | Description |
| --- | --- | --- | --- | --- |
| `DDR4_8GB_3200_pim.ini` | DDR4 | 8 GB | 3200 MT/s | DDR4 PIM module |
| `HBM2_1GB_2000_pim.ini` | HBM2 | 1 GB | 2000 MT/s | HBM2 PIM module |
| `LPDDR4X_2GB_4266_pim.ini` | LPDDR4X | 2 GB | 4266 MT/s | LPDDR4X PIM module |
| `LPDDR5_2GB_6400_pim.ini` | LPDDR5 | 2 GB | 6400 MT/s | LPDDR5 PIM module |

## Key parameters

The simulator extracts these from the INI files:

- **Bandwidth**: derived from `device_width`, `BL`, `tCK`, and channel count
- **Latency**: derived from timing parameters (`tRCD`, `CL`, etc.)
- **Capacity**: `rows * columns * device_width * banks * bankgroups`
- **PIM type**: `pim_type` in `[dram_structure]` section (`SINGLE` or `DUAL`)

# Scope: DeepSeek V4 (Sparse MLA) Support in the LLMServingSim Memory Model

> Companion to `deepseek_v4_profiling_guide.md`. That guide covers **data
> collection** (profiling) on a rented GPU. **This** document scopes the
> **code change in the simulator itself** (main dev machine, `serving/`) that
> DeepSeek needs to produce correct results. The two are independent: profiles
> can be collected first; this fix is required before the simulation numbers
> are trustworthy — and, per Tier 2 below, before DeepSeek's trace runs at all.
>
> All paths are relative to the repository root. Line numbers approximate.
>
> **⚠️ Verified against vLLM `main` source — DeepSeek V4 ≠ V3.** The numbers
> and layer names below were read from `vllm/models/deepseek_v4/` (a new
> package, distinct from the legacy `model_executor/models/deepseek_v2.py`
> that covers V2/V3). V4 uses **Sparse MLA** (an indexer selects top-k tokens)
> with a **compressed, FP8-quantized KV cache** and a separate **compressor
> state cache**, plus **MegaMoE** (FP4 experts, EP-only, Blackwell-optimized).
> Treat every figure here as "from a specific vLLM commit" — reconfirm against
> the exact vLLM version used to profile, since V4 is evolving quickly.

---

## 1. The problem

DeepSeek attention caches a **single compressed latent per token, shared across
all heads** — not per-head K and V. The simulator's KV-cache math is hard-wired
to **GQA/MHA** (`2 × kv_head × head_dim` per token), so it computes the KV
footprint wildly wrong.

**V4 is even more compressed than V3.** From vLLM `sparse_mla.py`, the V4 main
MLA cache in `fp8_ds_mla` mode is a **fixed 584 bytes per token per layer**:

```
584 B/token/layer  =  448 B (NoPE compressed latent, fp8)
                    + 128 B (RoPE key)
                    +   8 B (fp8 scale)
```

Note this is a **constant** — it does not depend on `num_heads` at all (the
latent is shared). Contrast the current GQA formula for the same model:

| | Bytes per token per layer |
|---|---|
| V4 Sparse MLA (correct, fp8_ds_mla) | **584** |
| Current GQA formula (`2 × kv_head × head_dim × 2B`) | `2 × 128 × 192 × 2` ≈ **98,304** |

That is a **~168× overestimate**. Since KV footprint drives max batch size,
eviction, and prefix-cache capacity, every memory-dependent DeepSeek result is
wrong (and absurdly pessimistic) — and this directly breaks the small-HBM-vs-
big-HBM offloading study, which is meaningless without correct KV sizing.

**Two additional V4-only caches the model must account for:**
1. **Compressor state cache** (`compressor.py`): a separate auxiliary cache
   with `compress_ratio ∈ {4, 128}` — stores ~1 compressed state per
   `compress_ratio` tokens, so it is small (≈ main-cache-bytes / compress_ratio
   scale). Used by the sparse indexer.
2. **Sparsity affects compute, not storage.** The indexer (`index_topk`) makes
   attention touch only top-k tokens — that changes attention *latency* (which
   comes from the profiled CSVs, not this code), but the cache still stores
   *all* tokens' compressed latent. So for the memory model, sparsity does not
   shrink capacity; the 584 B/token/layer stands.

Also note the KV **dtype** is not the model dtype: the cache is fp8
(`fp8_ds_mla`), so the simulator's `kv_fp` handling must reflect that the V4 KV
byte count is already fp8-baked (don't multiply by 2 for bf16).

## 2. Root cause (where GQA is baked in)

Two computation sites, both in `serving/core/memory_model.py`, and one
downstream consumer:

| Site | What it does | Why it breaks on MLA |
|---|---|---|
| `MemoryModel.get_kv(seq)` (~line 146) | `2 * kv_dim * seq * n_layer * kv_fp // num_npus`; `kv_dim = kv_head*head_dim` | Factor 2 (K+V) and per-head `kv_dim` are both wrong for a shared latent |
| `full_cluster_kv_bytes_per_token()` (~line 658) | Standalone mirror of `get_kv(1)*num_npus` | Same formula, must stay consistent |
| `calculate_sizes()` (~line 683–834) | Per-op activation/weight sizes for the trace; big `if layer_name == …` chain ending in **`else: raise ValueError`** (line ~833) | MLA has different projection layers (`q_a_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, …). Unknown names **raise** → trace generation crashes |

`get_config()` (`serving/core/utils.py`) just loads the raw JSON, so **all the
fields are already available** (`kv_lora_rank`, `q_lora_rank`,
`qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim`, `compress_ratio`,
`index_topk`) — they are simply never read. Detection signal is clean:
**`"kv_lora_rank" in config` ⟺ MLA** (V2/V3/V4).

## 3. The fix, in tiers

### Tier 1 — KV-cache capacity (essential; the "wrong numbers" fix)

Contained to `memory_model.py`. ~40–60 lines.

1. In `MemoryModel.__init__` (~line 39–42), after the existing head parsing,
   detect MLA and set the per-token-per-layer KV **byte** count directly. For
   **V4 (Sparse MLA, fp8_ds_mla)** the source gives a constant:
   ```python
   self.is_mla = 'kv_lora_rank' in self.config
   if self.is_mla:
       # V4 Sparse MLA main cache: 448 NoPE(fp8) + 128 RoPE + 8 fp8-scale.
       # Constant per token per layer — does NOT scale with num_heads.
       # (V2/V3 non-fp8 MLA: (kv_lora_rank + qk_rope_head_dim) * bytes.)
       self.mla_kv_bytes = 584
       # optional: add the compressor state cache, ~main/compress_ratio
       cr = self.config.get('compress_ratio', 0)
       if cr:
           self.mla_kv_bytes += 584 // cr   # small auxiliary cache
   ```
   Confirm `mla_kv_bytes` against the exact vLLM build: read the
   `get_kv_cache_shape` / bytes-per-token in `sparse_mla.py` for the cache
   dtype the profiling used (`fp8_ds_mla` gives 584). If a non-fp8 mode is
   used, compute `(kv_lora_rank + qk_rope_head_dim) * element_bytes` instead.
2. Branch `get_kv(seq)` to a **byte** count (not element count):
   ```python
   if self.is_mla:
       # already in bytes, fp8-baked; no ×2, no per-head; latent is
       # REPLICATED across TP ranks, not sharded → no //num_npus
       return self.mla_kv_bytes * seq * self.n_layer
   return 2 * self.kv_dim * seq * self.n_layer * self.kv_fp // self.num_npus
   ```
3. Apply the identical branch in `full_cluster_kv_bytes_per_token()`.

**Modeling decisions to confirm (flagged, not assumed):**
- **TP replication:** in GQA the code divides KV by `num_npus` (KV-heads shard
  across TP). The MLA latent is **replicated** on every TP rank, so per-GPU KV
  should **not** be divided. This directly sets per-GPU capacity → max batch
  size → the whole HBM-offloading study.
- **fp8 KV dtype:** the 584 figure is already fp8. Do **not** additionally
  apply the simulator's `kv_fp`/`--kv-cache-dtype` scaling on top, or you will
  double-count. Wire V4 to a fixed byte count, bypassing `kv_fp`.

### Tier 2 — trace can run at all (required, couples to the profiling YAML)

`calculate_sizes()` raises on unknown layer names. The architecture YAML the
profiling agent authors (`profiler/models/<model_type>.yaml`) will name V4's
projections. **The YAML's canonical layer names and `calculate_sizes()`'s
branches must be the same set** — this is the one hard coupling between the two
work streams.

**V4's attention is heavily FUSED** (from `vllm/models/deepseek_v4/attention.py`)
— it is *not* the V3 `q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`
layout. The real V4 submodules are approximately:

| V4 canonical layer (vLLM attr) | role |
|---|---|
| `fused_wqa_wkv` (MergedColumnParallelLinear) | fused Q-down + KV-down projection |
| `q_norm`, `kv_norm` (RMSNorm) | latent norms |
| `wq_b` (ColumnParallelLinear) | Q up-projection to heads |
| `indexer` (`DeepseekV4Indexer`, optional) | sparse top-k token selector (has `weights_proj`, `main_proj`, `main_norm`) |
| `compressor` (`DeepseekCompressor`, optional) | KV compression for the indexer |
| `rotary_emb` | RoPE (+ `indexer_rotary_emb`) |
| `attention` | sparse-MLA core (FlashMLA/flashinfer_sparse backend) |
| `wo_a`, `wo_b` (Column/RowParallelLinear) | low-rank output projection (split) |

For each, add a `calculate_sizes()` branch with its `n_embd`/`q_lora_rank`/
`kv_lora_rank`/head-dim shapes (read the exact dims from `attention.py`
`__init__`). **These names differ by platform** (`nvidia/`, `amd/`, `xpu/`) and
vLLM version — lock them against the build the agent profiles with, and keep the
YAML and these branches identical. The legacy V3 table below is kept for
reference only:

<details><summary>V3 layer table (reference — NOT V4)</summary>

| Canonical layer | in → out | weight |
|---|---|---|
| `q_a_proj` | `n_embd → q_lora_rank` | `n_embd × q_lora_rank` |
| `q_a_layernorm` | on `q_lora_rank` | scale |
| `q_b_proj` | `q_lora_rank → n_head×(qk_nope+qk_rope)` | `q_lora_rank × n_head×qk_head` |
| `kv_a_proj_with_mqa` | `n_embd → kv_lora_rank+qk_rope` | `n_embd × (kv_lora_rank+qk_rope)` |
| `kv_a_layernorm` | on `kv_lora_rank` | scale |
| `kv_b_proj` | `kv_lora_rank → n_head×(qk_nope+v_head_dim)` | `kv_lora_rank × n_head×(qk_nope+v)` |
| `attention` (MLA) | Q `n_head×qk_head` + KV-read `kv_len×(kv_lora_rank+qk_rope)` → `n_head×v_head_dim` | 0 |
| `o_proj` (MLA) | `n_head×v_head_dim → n_embd` | `n_head×v_head_dim × n_embd` |

(V2/V3 with `q_lora_rank=None` — e.g. V2-Lite — use a single `q_proj` instead
of `q_a`/`q_b`.)

</details>

TP sharding per layer mirrors vLLM: up-projections to heads (`wq_b`; V3's
`q_b`/`kv_b`) are column-parallel, the output projection (`wo_b`/`o_proj`) is
row-parallel, and the `_a` down-projections + latent norms are replicated.

### Tier 2.5 — MegaMoE routing/experts (V4-specific)

V4's MoE is **not** a plain `FusedMoE`. From `vllm/models/deepseek_v4/nvidia/
model.py`: `DeepseekV4MoE` uses `DeepseekV4MegaMoEExperts` which **requires
expert parallelism (EP), fp4 experts, and sqrtsoftplus routing**, and the fast
DeepGEMM path **requires SM100 (Blackwell)**. For `calculate_sizes()` the `moe`
branch still keys on `(tokens, activated_experts)` like the existing code, but:
- the **shared expert(s)** (`n_shared_experts`) run every token — model them as
  a dense MLP contribution alongside the routed experts;
- expert weights are **fp4** (0.5 byte/element), not `fp`, if you compute MoE
  weight sizes;
- the first `first_k_dense_replace` layers are **dense** (`DeepseekV4MLP`), the
  rest MoE — the existing dense/moe split handles this once the layer catalog
  is right.

### Tier 3 — fidelity refinements (optional, do after Tier 1–2 validate)

- **Attention KV-read volume:** the MLA `attention` `input_size` above models
  the latent read (not per-head ×2). This feeds decode memory-bandwidth /
  offload-transfer volume. Match it to vLLM's MLA mode (**"naive"** materializes
  full K/V from the latent; **"absorbed"/matabsorb** attends in latent space) —
  the two read different volumes. Compute *latency* itself comes from the
  profiled CSVs, so this only affects modeled data-movement, not the attention
  op's time.
- **`_build_trace_ctx` (`trace_generator.py` ~line 847):** carries
  `kv_head`/`head_dim` into `BatchCtx`. Confirm nothing downstream (PIM
  attention `channel_split = min(pim_channels, kv_head)`, skew lookup) misuses
  them under MLA. Likely minor; audit, don't assume.
- **Prefix-cache accounting:** `_bytes_per_token = get_kv(1)` (line 95) flows
  into the radix prefix-cache capacity — fixed automatically once `get_kv` is
  correct, but re-verify the multi-tier paths (CPU/CXL) with the new size.

## 4. Coordination & sequencing

- **YAML ↔ calculate_sizes contract:** agree the MLA canonical layer-name set
  once, use it in *both* the profiling YAML and the Tier-2 branches. Recommend
  fixing the names in this doc before either side is written.
- **Independent of profiling:** Tier 1 can be written and unit-tested now with
  just a DeepSeek `config.json` (no profiles needed). Tier 2 needs the YAML's
  names finalized. Profiles are only needed to run an end-to-end simulation.
- **`.FULL.json` interplay:** if profiling used reduced `n_routed_experts`
  (guide §4.2), the simulator must load the full-expert config — unrelated to
  MLA but same config file; keep them straight.

## 5. Validation

1. **Unit:** with a real DeepSeek `config.json`, assert `get_kv(1)` ≈
   `(kv_lora_rank + qk_rope_head_dim) × n_layer × kv_fp` and that it is ~50–85×
   smaller than the pre-fix value. Cross-check against DeepSeek's published
   per-token KV size (order ~70 KB/token for V3 at full precision).
2. **Behavioral:** run a single-instance sim (`python -m serving`, MoE cluster
   config) and confirm max batch size / KV memory is now realistic — the broken
   GQA estimate caps batch far too low; MLA should permit much larger batches.
3. **Ground truth (optional):** compare KV-cache blocks / max-concurrency
   against a real vLLM run of DeepSeek via `bench/` if a GPU is available.
4. **No-regression:** run the existing Qwen3-MoE / Llama examples and confirm
   identical output (the `is_mla` branch must be inert for non-MLA models).

## 6. Effort estimate

| Tier | Scope | Files | Rough size |
|---|---|---|---|
| 1 | KV capacity — fixed V4 byte count (the headline fix) | `memory_model.py` | ~40–60 lines, ~½ day |
| 2 | V4 fused-MLA layer-size branches (run at all) | `memory_model.py` (+ YAML agreement) | ~1–2 days |
| 2.5 | MegaMoE routing / shared-expert / fp4 handling | `memory_model.py` | ~½–1 day |
| 3 | Sparse-attention read-volume + audits | `memory_model.py`, `trace_generator.py` | ~1–2 days, iterative |

Tiers 1–2.5 are the real deliverable; Tier 3 is polish once results validate.
V4 is materially more work than a V3 MLA fix — sparse MLA, compressor cache,
and MegaMoE each add surface area. All of it is **CPU-only, main-machine
work** requiring no GPU.

## 7. Out of scope (captured elsewhere / not memory-model)

- **Attention/MoE *latency*** — comes from the profiled CSVs, not this code.
- **Sparse-attention selection itself** (`index_topk`) — affects compute
  (profiled), not stored capacity; Tier 3 only touches modeled read-volume.
- **The architecture YAML authoring** — the profiling agent's task (guide §3),
  coordinated here only via the shared layer-name set.
- **Confirm every number against the exact vLLM build** — the 584-byte cache,
  `compress_ratio ∈ {4,128}`, fp4 experts, and fused layer names were read
  from vLLM `main` at one commit; V4 is evolving. Reconfirm before coding.

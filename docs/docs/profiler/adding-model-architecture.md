---
sidebar_position: 6
title: Adding a model architecture
---

# Adding a model architecture

The profiler dispatches on the HF config's `model_type` field. If
your model's `model_type` already maps to a YAML under
`profiler/models/`, you're done, just run `profile.sh`. If not, you
need to add a YAML.

This page is about that case.

## When you need a new YAML

Run `cat configs/model/<your-org>/<your-model>.json | jq .model_type`
and compare against the bundled architectures:

| `model_type` | YAML | Covers |
| --- | --- | --- |
| `llama` | `llama.yaml` | Llama 3.x dense (8B / 70B / 405B / custom shapes), Mistral 7B, derivatives with the same block structure |
| `qwen3` | `qwen3.yaml` | Qwen3 dense (0.6B / 4B / 7B / 14B / 32B), with per-head `qk_norm` |
| `qwen3_moe` | `qwen3_moe.yaml` | Qwen3 MoE (30B-A3B, 235B-A22B) |
| `mixtral` | `mixtral.yaml` | `MixtralForCausalLM` (8x7B, 8x22B) |
| `phimoe` | `phimoe.yaml` | `PhiMoEForCausalLM` (Phi-3.5-MoE) |

If your `model_type` is one of these, you don't need to do anything
- the existing YAML handles it.

If it's a *new* `model_type` (e.g., `gemma2`, `deepseek_v3`,
`gpt_oss`), you need a new YAML. Read on.

## When you also need simulator code changes

Just adding a YAML is enough when the new model's per-iteration
flow fits the standard pattern:

```
prologue → pre_attn → post_attn → (mlp_dense | mlp_moe) → head
```

If the new model has a genuinely novel block structure, sliding
window attention, multi-latent attention (MLA, like DeepSeek V3),
dual MLP decoders, you'll also need to extend
`serving/core/trace_generator.py` to walk the new sequence and
attach the right collectives. We'll cover that at the end of this
page.

## YAML structure

Each architecture YAML has two top-level sections:

- `catalog:`: maps canonical layer names to vLLM internal class
  names. The profiler uses this to find the right module objects to
  time.
- `sequence:`: declares the order layers run in per iteration. The
  profiler emits one shot per sequence layer; the simulator's
  `trace_generator` walks the same list at trace time.

### Minimal example: `llama.yaml`

```yaml
catalog:
  embedding:
    cls: VocabParallelEmbedding
    category: dense
  layernorm:
    cls: RMSNorm
    category: dense
    tp_stable: true
  qkv_proj:
    cls: QKVParallelLinear
    category: dense
  rotary_emb:
    cls: RotaryEmbedding
    category: dense
  attention:
    cls: Attention
    category: attention
  o_proj:
    cls: RowParallelLinear
    category: dense
    tp_collective: ALLREDUCE
  gate_up_proj:
    cls: MergedColumnParallelLinear
    category: dense
  act_fn:
    cls: SiluAndMul
    category: dense
  down_proj:
    cls: RowParallelLinear
    category: dense
    tp_collective: ALLREDUCE
  final_layernorm:
    cls: RMSNorm
    category: dense
    tp_stable: true
  lm_head:
    cls: ParallelLMHead
    category: per_sequence
  sampler:
    cls: Sampler
    category: per_sequence
    tp_stable: true

sequence:
  prologue:
    - embedding
    - layernorm                   # input rms_norm before block 0
  pre_attn:
    - layernorm
    - qkv_proj
    - rotary_emb
  post_attn:
    - o_proj
    - layernorm                   # post_attention_layernorm
  mlp_dense:
    - gate_up_proj
    - act_fn
    - down_proj
  head:
    - final_layernorm
    - lm_head
    - sampler
```

### `catalog` field reference

| Field | Required | Meaning |
| --- | --- | --- |
| `cls` | ✓ | vLLM class name (used to resolve the module object via attribute lookup) |
| `category` | ✓ | One of `dense` / `per_sequence` / `attention` / `moe` |
| `tp_stable` | optional | `true` if the layer's latency doesn't depend on TP degree (e.g., layernorms, sampler). The writer profiles once at TP=1 and replicates to other `tp<N>/` folders |
| `tp_collective` | optional | If TP > 1, what collective fires after this layer: `ALLREDUCE` for `o_proj` and `down_proj`. Other layers don't need this |

### `sequence` section reference

| Group | Runs | Notes |
| --- | --- | --- |
| `prologue` | Once at the start of each iteration | Embedding lookup + initial input layernorm |
| `pre_attn` | Once per decoder block | qkv_proj + rotary_emb + (qk_norm if Qwen3) |
| `post_attn` | Once per decoder block | o_proj + post_attention_layernorm |
| `mlp_dense` | Once per decoder block (dense models) | gate_up_proj + act_fn + down_proj |
| `mlp_moe` | Once per decoder block (MoE models) | moe (with EP-ALLTOALL surround) |
| `head` | Once at the end of each iteration | final_layernorm + lm_head + sampler |

The `attention` layer always runs between `pre_attn` and `post_attn`
- it's not in `sequence`, it's implicit.

## MoE-specific YAML

MoE architectures add a `moe` entry in the catalog:

```yaml
catalog:
  # ... dense entries ...
  moe:
    cls: FusedMoE
    category: moe
    ep_collective: ALLTOALL    # always ALLTOALL for EP
```

And in `sequence`:

```yaml
sequence:
  # ... same as dense ...
  mlp_moe:
    - moe
  # don't include mlp_dense in MoE models
```

The simulator looks for `mlp_moe` in the YAML and, if present, runs
the EP-ALLTOALL dispatch + combine surround automatically.

See `qwen3_moe.yaml` and `mixtral.yaml` for full MoE YAMLs.

## Step-by-step: adding a new `model_type`

Suppose you want to support `gemma2` (the Google Gemma 2 series).
HF config has `model_type: "gemma2"`. Workflow:

### 1. Inspect the model's vLLM source

Look at `vllm/model_executor/models/<model>.py`. Identify:

- The decoder block class.
- Each layer attribute name (`self.qkv_proj`, `self.attention`, …).
- Whether layernorms are pre-attn / post-attn / both.
- Whether there are any extra layers (some models have post-MLP
  layernorms, etc.).
- For MoE: how experts are arranged.

### 2. Write `profiler/models/gemma2.yaml`

Start from the closest existing YAML (e.g., `llama.yaml` for a
Gemma-style dense model) and adjust:

- Update `cls` names to match the model's vLLM class names.
- Add any extra layers (e.g., Gemma 2's post-MLP layernorm) to the
  catalog and `sequence`.
- Set `tp_stable: true` on layers whose latency doesn't depend on
  TP.

### 3. Try profiling

```bash
MODEL="google/gemma-2-9b" \
HARDWARE="<your-hw>" \
TP_DEGREES=1 \
SKIP_SKEW=1 \
./profiler/profile.sh
```

Start with TP=1 and `SKIP_SKEW=1` for the fastest feedback. The
profiler will:

- Warn loudly if any layer in `sequence` isn't found on the model
  via the `cls` you specified.
- Skip layers it can't find (with a warning), so you can iterate.

If the YAML is right, you'll get clean CSVs. Run a tiny simulation
to confirm.

### 4. Try simulating

In your `cluster_config.json`:

```json
{
  "model_name": "google/gemma-2-9b",
  "hardware": "<your-hw>",
  "tp_size": 1,
  ...
}
```

Run `python -m serving --cluster-config ... --dataset workloads/example_trace.jsonl ...`.

If anything's off (layer not found, infinite loop, missing collective),
the simulator will tell you which layer in your YAML it doesn't know
how to handle. Fix and retry.

### 5. Commit + open a PR

Once it works, send a PR adding `profiler/models/gemma2.yaml`. Make
the PR title `Add gemma2 architecture support` and include:

- The HF model id you used to validate.
- Output of a smoke-test simulation (TTFT / TPOT for a small
  workload).
- Whether MoE was tested (or not, Gemma 2 isn't MoE, but other
  additions might be).

## When you also need to touch `serving/core/trace_generator.py`

Three flags that the YAML alone can't express. Each requires a small
Python addition:

### Sliding-window attention

Some models (Mistral, Llama 3.1 with sliding) limit attention to a
fixed-size window. The simulator's KV-cache budget needs to account
for this, total KV doesn't grow past the window size.

Where: extend the attention category lookup in `trace_generator.py`
to clip `kv_decode` at the window size, and update
`memory_model.py::get_kv` to cap KV blocks per request.

### MLA (Multi-Latent Attention, DeepSeek V3)

DeepSeek V3 compresses KV into a small latent and decompresses on
attention. KV size is much smaller than `num_heads * head_dim *
seq_len` would suggest.

Where: extend `memory_model.py::calculate_sizes` with an MLA case
that uses the latent dim (`kv_lora_rank`) instead of
`num_kv_heads * head_dim`.

### Dual MLP decoders

Some models (e.g., experimental architectures) have two MLPs per
block instead of one. Trace generation needs to know to emit two
`mlp_dense` runs per block.

Where: add a new `sequence` group (e.g., `mlp_dense_2`) and have
`trace_generator._emit_sequence` walk both.

These are all relatively small changes (~30–60 LOC each). The YAML
+ the existing trace generator handles 95% of new architectures
without touching Python.

## Where this gets validated

Once your YAML is in, the bundled `bench/` validation suite is the
sanity check: run vLLM end-to-end on the new model + run the same
workload through the simulator + see how close they match. If
TTFT / TPOT / throughput are all within ~5%, your YAML + (optional)
trace_generator changes are good.

See [`bench/README.md`](https://github.com/casys-kaist/LLMServingSim/tree/main/bench) on
GitHub for the validation methodology and per-model results.

## What's next

- **[Output bundle](./output-bundle)**: what CSVs the profiler
  produces given a working YAML.
- **[Simulator → Trace generation](/docs/simulator/trace-generation)** -
  what trace_generator does at runtime walking your `sequence:`.

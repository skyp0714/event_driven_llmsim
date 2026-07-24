# Collecting LLMServingSim Profiling Data for DeepSeek V4 — Agent Handoff Guide

> **Audience:** an autonomous agent running on a **rented GPU box** with no
> prior context on this project. Read this whole file first, then execute.
> **Deliverable:** a directory of per-operator latency CSVs that
> LLMServingSim consumes to simulate DeepSeek V4, **committed and pushed to
> this repo on a branch** (see §7).
>
> **You are collecting data, not running the full model.** The profiler
> builds a **1-layer model with dummy (random) weights** — it never
> downloads or loads the 671B-class checkpoint. Do not try to load the real
> weights.
>
> **⚠️ This is a ONE-SHOT GPU.** The box is rented and will be **returned
> permanently** after this run. Everything the simulator ever needs from a GPU
> must be collected now — re-collecting anything means renting another H100.
> Before you finish, clear the completeness gate in **§8**.

---

## 0. What "done" looks like

Success = this directory tree exists, populated and non-empty, **committed and
pushed to a branch of this repo** (see §7):

```
profiler/perf/<HARDWARE>/<org>/<DeepSeek-V4-name>/<variant>/
├── meta.yaml
├── tp1/
│   ├── dense.csv          # dense-layer op latencies
│   ├── attention.csv      # attention latency over (kv, chunk) grid
│   ├── moe.csv            # MoE latency over (tokens, activated_experts)
│   ├── per_sequence.csv   # lm_head / sampler
│   ├── skew.csv           # heterogeneous-batch attention samples
│   └── skew_fit.csv       # fitted alpha coefficients (from skew.csv)
└── tp<N>/ ...             # one folder per TP degree profiled
```

Plus the two input artifacts you will author on the way:
`configs/model/<org>/<DeepSeek-V4-name>.json` and
`profiler/models/<model_type>.yaml`.

---

## 1. Background: how this profiler works (read before doing anything)

LLMServingSim is an analytical LLM-serving simulator. It does **not** run the
model; it replays per-operator latencies measured once by this profiler.
Three facts govern everything you do:

1. **Dummy weights, one layer.** The profiler launches vLLM with
   `load_format="dummy"` and `hf_overrides={"num_hidden_layers": 1}`
   (defined in `profiler/core/config.py:HOST_ENGINE_DEFAULTS`). So it
   allocates **one decoder layer of random weights** — no checkpoint
   download. Only the model's `config.json` (a few KB) is needed.

   **These are two SEPARATE savings — don't conflate them:**
   - `load_format="dummy"` saves **download/disk** (0 bytes of checkpoint;
     latency depends on tensor shapes, not values, so random weights give
     correct timings). You **cannot** "download just one layer's real
     weights" — vLLM downloads the *whole* checkpoint if `load_format` is
     not dummy, then loads only what it needs. Dummy removes that entirely.
   - `num_hidden_layers=1` saves **GPU memory** (1 layer instead of ~61).
   - **Dummy weights still occupy full GPU memory** for that one layer —
     same shapes/dtypes as real weights. So GPU memory is still needed
     (≈ one decoder block; see §2/§4). Dummy is *not* why it fits on the
     GPU — the 1-layer override is. On a B200 (180 GB) one block fits easily.
   - **V4 fp4 caveat:** fp4 experts may need quantization scales that a plain
     dummy load doesn't populate, so building the 1-layer model can fail. Fix
     by supplying dummy scales — **not** by downloading real weights. If stuck
     here, report it (it's part of the §0 profiler-adaptation work).

2. **The architecture is resolved from a YAML catalog.** The profiler reads
   `model_type` from the model config, then loads
   `profiler/models/<model_type>.yaml`, which maps *canonical layer names*
   (what the simulator expects) to *vLLM class names* (what the CUDA
   profiler reports). **There is no DeepSeek YAML in the repo — you must
   author it** (§3). This is the main skilled task.

3. **Output is per-(TP, operator) CSVs.** For each tensor-parallel degree
   you profile, it sweeps operator input shapes and records latencies into
   the CSVs listed in §0.

Known-good examples already in the repo — study them, they are your
templates:
- MoE architecture YAML: `profiler/models/qwen3_moe.yaml`
- Dense YAML: `profiler/models/llama.yaml`
- A finished MoE profile:
  `profiler/perf/RTXPRO6000/Qwen/Qwen3-30B-A3B-Instruct-2507/`

---

## 2. Prerequisites & GPU sizing (decide before renting/starting)

### ⚠️ HARDWARE: V4 targets Blackwell (SM100), not Hopper (H100)

Verified in vLLM source (`vllm/models/deepseek_v4/`): V4's **MegaMoE** fast
path hard-requires **SM100 (Blackwell / B200)** —
`if device_capability != 10: raise NotImplementedError("DeepGEMM MegaMoE
requires SM100 GPUs")` — and its experts are **fp4**, EP-only. Sparse-MLA
attention supports SM90 (H100) *and* SM100, but the MoE does not.

**Consequence:** on an **H100 (SM90)** you can profile the attention side but
will likely hit `NotImplementedError` on the MoE (or a non-representative
fallback). To profile the *intended* V4 execution, rent a **B200 (SM100)**.
Before committing to a long run, verify on a cheap instance whether your GPU
+ vLLM build runs V4's MoE at all. Record the GPU compute capability in the
report.

### Memory sizing

**Why memory matters even for one layer.** A single DeepSeek decoder block
contains *all* of that layer's routed experts (V3: ~256; V4 "MegaMoE" may
differ — read the config). One dummy block is dominated by the experts; a
V3-class block is ~27 GB in bf16, but **V4 experts are fp4** (~4× smaller per
expert), so V4's one-block footprint can be lower for the same expert count.
Confirm from the V4 `config.json` (`n_routed_experts`, `moe_intermediate_size`,
`hidden_size`) — see §4 for the estimate.

**Recommendation: B200 (SM100).** A Blackwell GPU (180 GB) profiles V4 at full
width with no compromise and covers the MoE fast path. (If you must use H100,
expect to profile attention only, or fall back — and know the MoE numbers
won't represent the real V4.)

**How many B200s — 1 for most ops, but MegaMoE likely needs 2 (EP):**
- The profiler runs on a **single GPU** by default (`tensor_parallel_size=1`;
  higher `--tp` is *emulated* by sharding `hidden`/`heads`/`vocab` on one GPU).
  So **attention / dense / embedding** profiling needs only **one** GPU —
  restrict with `CUDA_VISIBLE_DEVICES=0` if you like.
- **But V4 MegaMoE requires expert parallelism** (source:
  `"DeepSeek V4 MegaMoE currently requires expert parallel"` → raises
  otherwise). A single GPU (EP=1) will either **error** or fall back to a
  **non-representative** `FusedMoE` path. So to profile the *real* MegaMoE you
  likely need **EP≥2, i.e. both B200s.**
- Therefore: **keep both rented B200s available.** Have the agent first verify,
  on a cheap check, whether the MoE path runs at EP=1 or requires EP=2 —
  the existing profiler assumes MoE at tp=1/one GPU (`moe.csv` profiled at
  tp=1), so supporting EP for MegaMoE is part of the §0 profiler adaptation.

Also required:
- **Docker with `--gpus all`** (NVIDIA Container Toolkit installed).
- **`HF_TOKEN`** env var with access to the DeepSeek V4 repo (needed only to
  fetch `config.json`; gated repos require it). Export it before starting.
- Outbound network to HuggingFace Hub (for the tiny config fetch) and to
  GitHub (to clone the repo).

---

## 3. Step-by-step

### Step 0 — GATE: you need a RECENT vLLM (v0.19.0 will NOT work)

Verified against vLLM `main`: **DeepSeek V4 lives in a new package**
`vllm/models/deepseek_v4/` (registered as `DeepseekV4ForCausalLM` →
`vllm.models.deepseek_v4`), **separate** from the legacy
`vllm/model_executor/models/deepseek_v2.py` that covers V2/V3. The repo's
pinned `vllm/vllm-openai:v0.19.0` predates this and **does not contain V4** —
you must use a recent vLLM image that has the `vllm/models/deepseek_v4`
package. Verify before renting time:

```bash
# check whatever image you plan to use (replace TAG):
docker run --rm --gpus all --entrypoint python3 vllm/vllm-openai:TAG \
  -c "import vllm,importlib.util as u; \
      print('V4:', u.find_spec('vllm.models.deepseek_v4') is not None); \
      print('vLLM', vllm.__version__)"
```

- `V4: True` → this image supports V4; use it. Record the vLLM version.
- `V4: False` → find a newer image tag (check vLLM releases for when the
  `deepseek_v4` package landed). Do **not** use the moving `latest` tag if you
  can pin a specific version — but a specific recent tag is required here.
- No released vLLM has it yet → **stop and report**; profiling is impossible.

> **Two version-specific breakages to expect on a recent vLLM (V4 is new and
> the profiler was written for v0.19.0):**
> 1. **MoE hook:** the profiler patches `FusedMoE.forward_native`, but V4 uses
>    **MegaMoE** (`DeepseekV4MegaMoEExperts`), a different path. `single_moe_
>    layer()` may find zero `FusedMoE` layers, or the fp4/EP MegaMoE path may
>    not run on non-Blackwell. Inspect `profiler/core/hooks/moe_hook.py` and
>    adapt it to V4's MoE class. This is real work, not a one-liner.
> 2. **Other internal APIs** (`layerwise_profile`, worker extension, dummy
>    load format) may have shifted across the version jump. Expect to debug.
>
> Document every profiler change you make in the report (§7) so it can be
> upstreamed into the simulator repo.

### Step 1 — Clone THIS repo and launch the profiling container

Clone the repo you were given (this one — it contains `agent/` and is where you
push results). Submodules are **not** needed for profiling (they are the C++
simulation backend); skip `--recurse-submodules` for a faster clone.

```bash
git clone git@github.com:yanggon-kim/LLMServingSim-repo.git
cd LLMServingSim-repo

# Launch the vLLM container with the repo mounted at /workspace.
# (This is scripts/docker-vllm.sh; run it directly, or inline as below.)
export HF_TOKEN=<your token>
docker run --name vllm_docker --gpus all -it \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$PWD":/workspace \
  --volume "$HOME/.cache/huggingface":/root/.cache/huggingface \
  --shm-size=16g -w /workspace \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.19.0 \
  -c "pip install pyyaml rich huggingface_hub && exec bash"
```

Everything below runs **inside** this container, from `/workspace`.

### Step 2 — Get the model config

Pick the exact HF id, e.g. `deepseek-ai/DeepSeek-V4` (confirm the real id).
The profiler auto-downloads `config.json` on first run, but fetch it now so
you can inspect it and (if needed) edit it:

```bash
python3 - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os, json, pathlib
hf_id = "deepseek-ai/DeepSeek-V4"          # <-- set the real id
src = hf_hub_download(repo_id=hf_id, filename="config.json",
                      token=os.environ.get("HF_TOKEN"))
dst = pathlib.Path("configs/model")/f"{hf_id}.json"
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(src, dst)
cfg = json.load(open(dst))
print("model_type      :", cfg.get("model_type"))
print("architectures   :", cfg.get("architectures"))
print("hidden_size     :", cfg.get("hidden_size"))
print("num_hidden_layers:", cfg.get("num_hidden_layers"))
print("n_routed_experts:", cfg.get("n_routed_experts"))
print("num_experts_per_tok:", cfg.get("num_experts_per_tok"))
print("n_shared_experts:", cfg.get("n_shared_experts"))
print("moe_intermediate_size:", cfg.get("moe_intermediate_size"))
print("kv_lora_rank    :", cfg.get("kv_lora_rank"))
print("q_lora_rank     :", cfg.get("q_lora_rank"))
print("vocab_size      :", cfg.get("vocab_size"))
PY
```

Record the printed values — you need `model_type` for the YAML filename in
Step 3 and the dimensions for the memory estimate in Step 4.

### Step 2b — REQUIRED config edit: force the single layer to be MoE

**This step is mandatory for DeepSeek and the run will crash without it.**

The profiler builds a **1-layer** model (`num_hidden_layers=1`, set
automatically). But DeepSeek interleaves layer types: the first
`first_k_dense_replace` layers are **dense MLPs**, and only later layers are
MoE (V3 uses `first_k_dense_replace=3`, i.e. layers 0–2 dense, 3+ MoE). So
the auto-built layer 0 is a **dense** layer, and the MoE profiler asserts it
finds exactly one MoE layer:

```
RuntimeError: Expected exactly one FusedMoE layer in the test model, got 0
```

The profiler has **no handling for `first_k_dense_replace`** (it was written
for Qwen/Mixtral where every layer is MoE). You must force the single layer
to be the MoE variant by editing the config:

```bash
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("configs/model/deepseek-ai/DeepSeek-V4.json")   # <-- your path
c = json.load(open(p))
if c.get("first_k_dense_replace", 0):
    print("was first_k_dense_replace =", c["first_k_dense_replace"])
c["first_k_dense_replace"] = 0        # make layer 0 an MoE layer
json.dump(c, open(p, "w"), indent=2)
print("set first_k_dense_replace = 0  → single profiled layer is now MoE")
PY
```

Check the printed original value first: if it was already `0`, no MoE trap
exists (but keep the edit — it's harmless). If it was `> 0`, this edit is
what prevents the crash.

**Consequence to record:** with all layers forced MoE, DeepSeek's few real
dense layers (`first_k_dense_replace` of ~61) are not separately profiled.
That's ~5% of layers and the dense-MLP path is minor, so it's normally an
acceptable approximation. If exact dense-layer timing is required, do a
**second** profiling pass into a different `--variant` with the config's MoE
disabled (e.g. set `n_routed_experts` such that the layer builds as dense, or
raise `first_k_dense_replace` above the single layer) — but this is optional
and usually skipped.

> If you also reduce experts for a small GPU (Step 4.2), make **both** edits
> to the same config file before running.

### Step 3 — Author the architecture YAML (the main task)

Create `profiler/models/<model_type>.yaml` (e.g. `deepseek_v3.yaml` — use the
**actual** `model_type` string from Step 2). It tells the profiler how to map
the layers vLLM instantiates to the canonical operator names the simulator
walks.

**Method (do this against the real V4 code, not from memory):**

1. Open the **V4 package** `vllm/models/deepseek_v4/` — specifically
   `nvidia/model.py` (`DeepseekV4DecoderLayer`, `DeepseekV4MoE`,
   `DeepseekV4MLP`) and `attention.py` (the fused MLA attention). Read each
   `__init__` and `forward` to list, **in execution order**, every sub-module
   and its class. Pick the subdir (`nvidia/`, `amd/`, `xpu/`) matching your GPU.
2. Copy `profiler/models/qwen3_moe.yaml` as a structural template (MoE +
   attention + shared), but expect to rewrite the attention block substantially.
3. Rewrite `sequence:`/`catalog:` to V4's **fused Sparse-MLA** layout. It is
   **not** the V3 `q_a_proj/q_b_proj/kv_a_proj_with_mqa/kv_b_proj` layout.
   The real V4 attention submodules (from `attention.py`, confirm exact names
   in your build):

   ```yaml
   # V4 fused MLA attention (attn_norm → ... → wo):
   #   fused_wqa_wkv   (MergedColumnParallelLinear)  fused Q-down + KV-down
   #   q_norm, kv_norm (RMSNorm)                     latent norms
   #   wq_b            (ColumnParallelLinear)         Q up-projection to heads
   #   indexer         (DeepseekV4Indexer, optional)  sparse top-k selector
   #   compressor      (DeepseekCompressor, optional) KV compression
   #   rotary_emb (+ indexer_rotary_emb)
   #   attention       (FlashMLA / flashinfer_sparse backend)
   #   wo_a, wo_b      (Column/RowParallelLinear)     low-rank output proj
   # MoE block (DeepseekV4MoE):
   #   experts = FusedMoE  OR  DeepseekV4MegaMoEExperts (fp4, EP-only, SM100)
   #   + shared expert(s) (n_shared_experts) run every token
   #   first `first_k_dense_replace` layers are DENSE (DeepseekV4MLP)
   ```

   These canonical names must **exactly match** the Tier-2 branches in the
   simulator fix (`agent/mla_memory_model_scope.md` §3) — agree the name set
   once and use it in both places.
4. Each `catalog` entry needs the `vllm:` class name exactly as reported; add
   `within:` to disambiguate repeated classes (`RMSNorm`) and `tp_stable: true`
   for TP-invariant ops (norms, samplers) — mirror `qwen3_moe.yaml`.
5. **Validation:** the first profiler run errors clearly if a `sequence:` layer
   is missing from `catalog:`, or a declared `vllm:` class never appears in the
   built model. Iterate with a short dry run (Step 5, `--skip-skew`, small grids)
   until mapping is clean.

> This YAML is the single highest-risk artifact and V4's fused/sparse/MegaMoE
> structure makes it **substantially harder than a V3 MLA rename**. Budget real
> time; verify every class name against the *actual* vLLM V4 source for your
> exact version and platform subdir.

### Step 4 — GPU memory sizing decision

**4.1 (preferred) Full-width profile on a ≥48 GB GPU.** Estimate one block:

```
experts_bytes = n_routed_experts * 3 * hidden_size * moe_intermediate_size * 2
embed_bytes   = 2 * vocab_size * hidden_size * 2          # embed + lm_head
one_block     ≈ experts_bytes + embed_bytes + ~1 GB (MLA + activations)
```

If `one_block` (plus vLLM's overhead; it targets
`gpu_memory_utilization=0.9`) fits your GPU, do nothing special — profile at
full width. **This is the accurate path; use it if you can.**

**4.2 (fallback) Small-GPU workaround — reduce experts.** Only if the GPU
can't hold all experts. There is **no CLI flag** for this; you edit the
config directly. The per-expert GEMM latency is independent of how many
experts exist, so a reduced-expert profile still measures the right
per-expert cost — but the `activated_experts` sweep is capped at the reduced
count and the simulator extrapolates beyond it (accuracy loss for
large-batch MoE).

```bash
# Keep a full-width copy for the simulator, profile with a reduced copy.
# NOTE: make the FULL copy BEFORE the Step 2b first_k_dense_replace edit if
# you want the simulator's config to keep the real dense-layer layout;
# otherwise copy the already-edited file and restore first_k_dense_replace
# in the FULL copy afterwards.
cp configs/model/<org>/<name>.json configs/model/<org>/<name>.FULL.json
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("configs/model/<org>/<name>.json")
c = json.load(open(p))
c["n_routed_experts"] = 32      # shrink to fit; keep >= a few * top_k
c["first_k_dense_replace"] = 0  # Step 2b — keep it set on the profiling copy
json.dump(c, open(p,"w"), indent=2)
print("profiling config:", c["n_routed_experts"], "experts, dense_replace=0")
PY
```

**Critical:** if you use 4.2, **note it in the returned report and ship the
`.FULL.json` too** — the simulator must run with the *full* expert count,
and whoever runs the simulation needs to restore it. Also record the reduced
number so the accuracy caveat is known.

### Step 5 — Run the profiler

Edit the knobs at the top of `profiler/profile.sh` (`MODEL`, `HARDWARE`,
`TP_DEGREES`), then run it — or call the module directly:

```bash
python3 -m profiler profile "deepseek-ai/DeepSeek-V4" \
    --hardware "<HARDWARE_LABEL>" \
    --tp "1"                       # add ",2,4,8" for the TP degrees you need
# useful during bring-up:
#   --skip-skew                    # skip the 1-2h/TP skew pass while debugging the YAML
#   --measurement-iterations 1     # faster, noisier; use 3 (default) for final
#   --force                        # wipe and re-profile (default is resume)
```

Notes:
- `<HARDWARE_LABEL>` is just an output folder name — use the real GPU name
  (e.g. `A100-80G`, `H100-SXM`), not `RTXPRO6000`.
- **TP degrees:** profile the TP degrees the simulation will actually use.
  DeepSeek is served at high TP/EP; if unsure, do `--tp 1,8` at minimum. Each
  TP degree emulates one rank by sharding `hidden`/`heads`/`vocab` in the
  config (experts are **not** TP-sharded, so higher TP does *not* reduce
  expert memory — see §4).
- **Do a dry run first:** `--tp 1 --skip-skew --measurement-iterations 1` to
  shake out YAML mapping errors in minutes before the multi-hour full run.
  If it crashes with **`Expected exactly one FusedMoE layer ... got 0`**, you
  skipped Step 2b — set `first_k_dense_replace=0` in the config and rerun.
- **Cover the target context lengths** with `--attention-max-kv` (default
  16384). The simulator *extrapolates* beyond the profiled grid, so if the
  simulation will study long-context serving (e.g. 128k), raise this to that
  range. You cannot extend the grid later without a GPU (§8).
- Expect the full run (with skew, 3 iterations) to take a few hours per TP
  degree.

### Step 6 — Verify the outputs

```bash
find profiler/perf/<HARDWARE>/<org>/<name> -type f | sort
# For every tp<N> folder, confirm these are present and NON-EMPTY:
for f in dense attention moe per_sequence skew skew_fit; do
  wc -l profiler/perf/<HARDWARE>/<org>/<name>/*/tp*/$f.csv
done
```

Sanity checks:
- `moe.csv` must have rows spanning a range of `activated_experts` (this is
  the MoE-specific data; if it's empty or single-valued, the YAML `moe`
  mapping is wrong).
- `attention.csv` latencies should grow with kv length.
- `skew_fit.csv` exists (the alpha fit succeeded) unless you used
  `--skip-skew` — for a **final** deliverable, do **not** skip skew.
- `meta.yaml` records dtype/variant/config — check it names DeepSeek V4.

### Step 7 — Commit and push to this repo

Push everything you produced to a branch. **Note the profiler's output dir
`profiler/perf/` is normally git-ignored** (it holds generated CSVs), so
force-add your profile bundle.

```bash
git checkout -b deepseek-v4-profiles

# 1) inputs you authored/edited
git add profiler/models/<model_type>.yaml
git add configs/model/<org>/<name>.json
git add configs/model/<org>/<name>.FULL.json      # only if §4.2 was used

# 2) the profile output (force past .gitignore)
git add -f profiler/perf/<HARDWARE>/<org>/<name>

# 3) your run report (write it as agent/REPORT-deepseek-v4.md — see below)
git add agent/REPORT-deepseek-v4.md

git commit -m "DeepSeek V4 profiles on <HARDWARE_LABEL> (tp <list>)"
git push -u origin deepseek-v4-profiles
```

(Pushing requires this box's git credentials to be configured — a deploy key
or PAT with write access to the repo. If `git push` is rejected for auth, stop
and report; do **not** discard the results.)

Write **`agent/REPORT-deepseek-v4.md`** stating:
- vLLM image tag used (from Step 0), and any `moe_hook.py` change you had to
  make for that version,
- `HARDWARE` label and actual GPU model/memory,
- TP degrees profiled,
- the original `first_k_dense_replace` value and confirmation it was set to 0
  for profiling (Step 2b) — so the dense-layer approximation is on record,
- whether §4.2 expert-reduction was used and to what count (accuracy caveat),
- whether skew was profiled,
- any YAML class-name deviations from V3 you had to make,
- any errors or shapes you could not profile,
- the exact `python3 -m profiler profile …` command(s) you ran.

### Step 8 — Completeness gate (DO NOT skip; the GPU is returned after this)

The rented GPU is your **only** shot at any GPU-dependent data. After it is
returned, the simulation runs CPU-only forever, but nothing here can be
re-collected without renting another H100. Confirm **all** of the following
before you consider the job done and the box is shut down:

- [ ] **Every TP degree that will be simulated is profiled.** You cannot add
      `tp16` later. If unsure which degrees, profile `1,8` at minimum (DeepSeek
      is served at high TP/EP).
- [ ] **The full skew pass ran** (you did **not** use `--skip-skew`), so every
      `tp<N>/` has a `skew.csv` and `skew_fit.csv`. Skew is required for
      accurate skewed-batch attention.
- [ ] **`--attention-max-kv` covers the intended context lengths** (Step 5).
- [ ] **The committed config has the FULL expert count** (`n_routed_experts`).
      If you used the §4.2 small-GPU reduction, the `.FULL.json` is committed
      and the report says so. (On an 80 GB H100 you should not have needed §4.2.)
- [ ] **All output CSVs verified** per Step 6 (present, non-empty, `moe.csv`
      spans a range of `activated_experts`, `attention.csv` grows with kv).
- [ ] **`agent/REPORT-deepseek-v4.md` is written** (Step 7).
- [ ] **The push actually landed on the remote.** After `git push`, re-verify:
      ```bash
      git ls-remote --heads origin | grep deepseek-v4-profiles
      ```
      and confirm the branch shows your commit on GitHub. **Do not shut down
      the box until you have seen the push succeed.** If auth fails, stop and
      report — never let the GPU be returned with un-pushed data.

**Power/energy data — decide now (optional, also un-re-collectable):** if the
simulation needs LLMServingSim's power/energy outputs, that requires GPU power
measurement (`profiler/power/`) and must be captured on this box. If only
throughput / latency / memory results are needed, skip it. **If you are unsure
whether power is in scope, ask the requester before releasing the GPU** — it
cannot be added afterward.

> **Not collectable on this single box (for awareness, not action):**
> end-to-end `bench/` validation against real vLLM needs the full model served
> across many GPUs, which one H100 cannot do for a 671B-class model. The
> profile-based estimate is the expected mode here; there will be no
> ground-truth cross-check, and that is fine.

---

## 4. Known limitations — set expectations (NOT the agent's job to fix)

These are for the requester's awareness; the agent should not attempt them:

1. **The simulator's memory model does not understand MLA.** LLMServingSim's
   `serving/core/memory_model.py` computes KV-cache size from
   `num_key_value_heads × head_dim` (standard GQA/MHA). DeepSeek's MLA uses a
   compressed latent KV (`kv_lora_rank`), which is much smaller. So even with
   perfect profiles, **KV-cache/memory results for DeepSeek will be wrong**
   until MLA is added to the simulator — a separate code change on the main
   development machine, not part of data collection.
2. **Expert-reduction (if §4.2 used) trades accuracy** on large-batch MoE, as
   noted there.
3. **DeepSeek V4 is newer than this guide's reference point.** All specific
   layer/class names here are from DeepSeek V3. **Trust the actual V4 config
   and the actual vLLM V4 model source over this document** wherever they
   disagree.

---

## 5. Quick reference

| Item | Value |
|---|---|
| Profiler entry | `python3 -m profiler profile <hf_id> --hardware <label> --tp <list>` |
| Container | `vllm/vllm-openai:v0.19.0` (or newer if V4 needs it) |
| Model config | `configs/model/<org>/<name>.json` (auto-fetched; editable) |
| **Required config edit** | `first_k_dense_replace: 0` (Step 2b) — else MoE profiler crashes with "got 0 FusedMoE" |
| Arch YAML (author this) | `profiler/models/<model_type>.yaml` |
| Output | `profiler/perf/<hardware>/<org>/<name>/<variant>/tp<N>/*.csv` |
| No-download mechanism | `load_format=dummy` + `num_hidden_layers=1` (built-in) |
| Templates to copy | `profiler/models/qwen3_moe.yaml`, `.../llama.yaml` |
| Reference profile | `profiler/perf/RTXPRO6000/Qwen/Qwen3-30B-A3B-Instruct-2507/` |
| Dry-run flags | `--tp 1 --skip-skew --measurement-iterations 1` |
| Final-run flags | `--tp <real degrees>` (skew ON, iterations 3) |
| Return method | `git push` a `deepseek-v4-profiles` branch (`git add -f profiler/perf/...`) + `agent/REPORT-deepseek-v4.md` (§7) |

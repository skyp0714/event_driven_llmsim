# START HERE — Agent Orientation

You are an autonomous agent on a **rented GPU box**. This repo is a copy of
**LLMServingSim** (a cycle-level LLM-serving simulator) plus this `agent/`
directory of instructions. Read this file first.

## Your task

**Collect operator-level profiling data for DeepSeek V4** so that
LLMServingSim can simulate DeepSeek V4 serving. You do this by running the
repo's profiler, which builds a **1-layer, dummy-weight** model in vLLM and
measures per-operator latencies — it never downloads the full checkpoint.

→ **Follow [`deepseek_v4_profiling_guide.md`](deepseek_v4_profiling_guide.md)
end to end.** It is the authoritative, step-by-step task. Do not deviate from
its memory/expert/first_k_dense_replace guidance.

## What you deliver

A branch pushed to this repo (`git push`, see guide §7) containing:
- `profiler/perf/<HW>/<org>/<name>/…` — the profile CSVs (the actual data)
- `profiler/models/<model_type>.yaml` — the architecture catalog you author
- `configs/model/<org>/<name>.json` (+ `.FULL.json` if you reduced experts)
- `agent/REPORT-deepseek-v4.md` — your run report (contents listed in guide §7)

## Scope — read this so you don't over-reach

- **In scope:** everything in the profiling guide, including **authoring the
  architecture YAML** (guide §3) and the required config edit (guide §2b).
  These are code/config additions to this repo — that is expected and correct.
- **Reference only, NOT your task:** [`mla_memory_model_scope.md`](mla_memory_model_scope.md)
  scopes a *simulator* code change (MLA support in `serving/`). You are **not**
  implementing that. It is included because it defines the **canonical MLA
  layer names** your YAML (guide §3) should use — read its Tier-2 table so your
  YAML and the future simulator fix agree on names. Do not edit `serving/`.

## ⚠️ This is a one-shot GPU — make it count

The GPU box is rented and **returned permanently** after this run. Everything
the simulator needs from a GPU must be collected now; re-collecting anything
means renting another H100. The simulation itself runs CPU-only afterward, so
the GPU is genuinely needed only this once — which is exactly why the profile
must be **complete and verified** before you finish. Clear the completeness
gate in **guide §8** before the box is shut down:
- profile every TP degree that will be simulated (can't add more later),
- run the **full** skew pass (no `--skip-skew`),
- set `--attention-max-kv` to cover the target context lengths,
- commit the **full-expert** config,
- decide whether power/energy data is needed (also un-re-collectable),
- and **confirm your push actually landed on the remote** before shutdown.

## ⚠️ DeepSeek V4 is a NEW architecture (verified in vLLM source)

V4 is **not** V3-with-more-layers. From `vllm/models/deepseek_v4/` (a new
package, separate from the V2/V3 file): it uses **Sparse MLA** (a top-k token
indexer) with a **compressed, fp8-quantized KV cache** (a fixed ~584
bytes/token/layer), plus **MegaMoE** (fp4 experts, expert-parallel-only,
**Blackwell/SM100-optimized**). This changes hardware, profiler, and the
simulator fix — read guide §0 and §2 before renting anything.

## The four things most likely to trip you up

1. **Hardware — V4 targets Blackwell (B200/SM100), not H100.** V4's MegaMoE
   fast path raises `NotImplementedError` on H100 (SM90). Prefer a **B200**;
   on H100 you can only profile attention reliably (guide §2). **GPU count:**
   the profiler uses **one** GPU for attention/dense, but **MegaMoE requires
   expert parallelism (EP)** — profiling the real MoE likely needs **two**
   GPUs (EP=2). Verify EP=1 vs EP=2 for the MoE before the long run (guide §2).
2. **vLLM must be recent** — V4 is in the new `vllm/models/deepseek_v4`
   package; the repo's pinned `v0.19.0` does NOT have it (guide §0). Expect the
   profiler's MoE hook (written for `FusedMoE`) to need adapting for **MegaMoE**.
3. **The architecture YAML is a real reverse-engineering job** (guide §3) —
   V4's fused MLA + indexer + compressor + MegaMoE, not a V3 rename. Main task.
4. **DeepSeek layer 0 is dense**, not MoE. Without the guide §2b config edit
   (`first_k_dense_replace=0`) MoE profiling crashes with `… got 0 FusedMoE`.

## Working conventions

- Work on a branch (`deepseek-v4-profiles`); commit your outputs and push.
- Pushing needs this box's configured git credentials (deploy key / PAT). If
  auth fails, **stop and report — never discard collected profiles.**
- The repo's own agent conventions live in `AGENTS.md` at the repo root (code
  style, canonical layer names, pitfalls) — consult it if you edit repo code.
- Submodules (`astra-sim/`) are the C++ backend; **not needed for profiling.**

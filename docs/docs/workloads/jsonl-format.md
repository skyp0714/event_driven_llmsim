---
sidebar_position: 2
title: JSONL format
---

# JSONL format

Workload files are line-delimited JSON (`.jsonl`). Each line is a
JSON object representing **either** an independent request (flat
format) or a session with chained LLM calls (agentic format). The
two formats can coexist in the same file, the loader auto-detects
per line.

## Flat format

Every line is one independent request:

```json
{"input_toks": 1472, "output_toks": 133, "arrival_time_ns": 4059740, "input_tok_ids": [1, 2, 3, ...], "output_tok_ids": [4, 5, 6, ...]}
```

### Fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `input_toks` | int | ✓ | Number of prompt tokens |
| `output_toks` | int | ✓ | Number of tokens to generate |
| `arrival_time_ns` | int | ✓ | When the request arrives in nanoseconds (relative to start of simulation) |
| `input_tok_ids` | list&lt;int&gt; | optional | Pre-tokenized prompt IDs (enables prefix-cache hashing, see [below](#why-token-ids-matter)) |
| `output_tok_ids` | list&lt;int&gt; | optional | Pre-tokenized output IDs (used internally for output-side analysis; usually fine to omit) |

If `input_tok_ids` is provided, `len(input_tok_ids)` must equal
`input_toks` (same for output).

### When to use flat

- ShareGPT-style benchmarks (independent prompts).
- Production trace replay (each prompt is its own request).
- Stress tests with a fixed Poisson arrival pattern.

## Agentic format

Every line is one **session** with multiple chained LLM calls. Each
call's arrival time is determined by the previous call's completion
plus the `tool_duration_ns` between them, the simulator respects
this dependency chain:

```json
{
  "session_id": "session_0",
  "arrival_time_ns": 4059740,
  "sub_requests": [
    {"input_toks": 1472, "output_toks": 133, "tool_duration_ns": 127348767},
    {"input_toks": 1582, "output_toks": 125, "tool_duration_ns": 197295027},
    {"input_toks": 1734, "output_toks": 77,  "tool_duration_ns": 0}
  ]
}
```

### Top-level fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `session_id` | string | ✓ | Unique identifier for the session |
| `arrival_time_ns` | int | ✓ | When the **first** sub-request arrives |
| `sub_requests` | list&lt;object&gt; | ✓ | Ordered chain of LLM calls. Length ≥ 1 |

### Sub-request fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `input_toks` | int | ✓ | Prompt tokens for this LLM call |
| `output_toks` | int | ✓ | Generated tokens |
| `tool_duration_ns` | int | ✓ | Completion-to-next-request-ready gap **after** this call; the legacy name can include human or mixed waits |
| `inter_turn_gap_type` | string | optional | Outgoing N→N+1 class: `human`, `tool`, `mixed`, or `unknown` |
| `tool_wait_source` | string | optional | Provenance for the measured outgoing gap boundary |
| `input_tok_ids` | list&lt;int&gt; | optional | Same as flat format |
| `output_tok_ids` | list&lt;int&gt; | optional | Same as flat format |
| `prefix_reuse_toks` | int | optional | Explicit logical prefix reusable from the preceding turn; used when exact token IDs are unavailable |
| `prefix_reuse_source` | string | optional | Provenance for the explicit count: normally `reported`, `exact`, or `estimated` |
| `raw_newly_append_toks` | int | optional | Source-derived appended-token count before compatibility normalization; zero is preserved |
| `newly_append_toks` | int | optional | Legacy normalized append count; some converters enforce a minimum of one, so do not use it as the execution-overlap authority |
| `observed_provider_hit_toks` | int | optional | Incumbent provider's observed cache-hit outcome, retained for sensitivity rather than policy-independent opportunity |

The last sub-request typically has `tool_duration_ns: 0` (nothing to
wait for after the session ends).

These three outgoing fields live on sub-request *N*. When sub-request *N+1*
is released, the runtime copies them forward as its incoming
`return_gap_ns`, `return_gap_type`, and `return_gap_source`. The first request
is labeled `session_start`. Keeping ownership explicit prevents a human return
from being mislabeled with the following call's outgoing tool gap.

The incoming request-ready epoch is the earliest legal restore issue boundary.
For a tool return it is the tool-result event; for a human return it is the
observed user-message event, which cannot be prefetched. A mixed return waits
for the latest required input. Strict P/D may then add a separately reported
per-pair FIFO and safe preparation-boundary wait before physical restore
issues; neither delay is backdated into the human/tool gap or counted as
transfer service. After issue, `async-decode-join` may overlap restore
with `input_toks - effective_reuse_toks - 1` fresh prompt tokens, but the final
prompt token/first-output boundary always waits for complete KV restore.

### When to use agentic

- **Tool-using agents** (browser agents, code agents, RAG with retrieval steps).
- **SWE-bench-style benchmarks** where each session involves multiple
  edits + tests + retries.
- **Multi-turn dialog** with simulated user think time between turns.

The simulator handles the chain via `Router._deferred_sessions` -
only the first sub-request is queued initially; the rest are released
as their predecessors complete. See
**[Simulator → Request lifecycle](/docs/simulator/request-lifecycle#agentic-sessions-when-stage-10-is-not-the-end)**
for the runtime mechanics.

## Mixing formats

A single `.jsonl` file can contain both flat and agentic entries.
The loader inspects each line:

- Has `sub_requests` key? → agentic.
- Otherwise → flat.

This is occasionally useful: an agentic SWE-bench workload can
include a few flat "baseline" requests for sanity-checking.

## Why token IDs matter

The optional `input_tok_ids` field is what makes prefix caching
work end-to-end:

- Without it, the simulator just knows "prompt has N tokens" but
  can't recognize when two prompts share a prefix.
- With it, the router computes a per-block hash of the token IDs at
  load time. The scheduler then matches requests against the
  RadixCache at run time using those hashes.

For ShareGPT-style traces where many requests share a system prompt,
having token IDs makes prefix-cache hit rates 5-10× higher than
without. **Pre-tokenize when you can.** The bundled generator does
this for you.

For agentic session tiering, a sanitized trace may instead provide
`prefix_reuse_toks` plus `prefix_reuse_source: "estimated"`. Provider-observed
cache accounting belongs in a separate field such as
`observed_provider_hit_toks`; feeding that outcome back as the proposed
policy's opportunity creates circular evaluation. Keep estimated, observed,
and target-tokenizer-exact classes separate in results.

If your dataset only has raw text, you have two options:

1. Run a tokenizer at workload-generation time to populate
   `input_tok_ids`. The ShareGPT generator does this.
2. Skip token IDs entirely. Prefix caching still works for *exact*
   prefix matches based on `input_toks` alone, but matches are much
   coarser and miss most opportunities.

**Tokenize with the same model the simulator runs.** A workload
generated with the Llama tokenizer won't produce useful prefix hits
in a Qwen3 simulation, the token streams are entirely different.

## Validation

The loader (`router.load_requests`) checks at startup:

- All required fields present.
- `len(input_tok_ids) == input_toks` if provided (same for output).
- `arrival_time_ns >= 0` and any order is fine, the loader sorts
  by arrival time anyway.
- Agentic: at least one sub-request, `tool_duration_ns >= 0`.

Validation errors are printed with the offending line number and
field name; the loader exits before any simulation work starts.

## Gotchas

1. **`arrival_time_ns` is the simulator clock**, not wall-clock. A
   workload generated at 10 sessions/s has arrival times spanning
   30 seconds for 300 sessions, that's 30 simulator-seconds, not 30
   real seconds.
2. **Token IDs are integers, not strings.** Whatever your tokenizer
   outputs (`tokenizer.encode(...).ids`) goes here directly.
3. **Output token IDs are usually unused at runtime**: the simulator
   doesn't need them to compute decode timing. Provided generators
   include them for downstream analysis tools.
4. **Mixing tokenizers across workloads is fine, but mixing inside
   one file is not.** All `input_tok_ids` should come from the same
   tokenizer.

## What's next

- **[ShareGPT generator](./sharegpt-generators)**: produce flat
  workloads from real ShareGPT traces with proper tokenization.
- **[Agentic sessions](./agentic-sessions)**: deeper dive on the
  agentic format and how to build your own chains.

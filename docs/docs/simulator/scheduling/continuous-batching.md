---
title: Continuous batching
sidebar_position: 1
---

# Continuous batching

The scheduler is the heart of each serving instance. Every iteration
of the main loop calls `scheduler.schedule(current, sys)` and gets
back a `Batch` (or `None`). The scheduler enforces the same
constraints vLLM does: token budget, sequence count cap, and
optionally chunked prefill. This page walks through the rules.

> Need the configuration knobs? See
> **[Reference → CLI flags](/docs/reference/cli-flags)** for the flag
> list. This page explains *what each flag does internally*.

## Two scheduling paths

Depending on whether `--enable-prefix-caching` is on (default), the
scheduler takes one of two code paths inside `serving/core/scheduler.py`:

| Flag | Method | What changes |
| --- | --- | --- |
| `--no-enable-prefix-caching` | `schedule_base` | Pure token-budget scheduler. Each request always runs from token 0. |
| `--enable-prefix-caching` (default) | `schedule_with_prefix` | Same plus a RadixCache lookup that returns `hit_len` per request. |

In both paths, the constraints are the same:

- **Sequence cap:** `len(batch) <= --max-num-seqs`. Default `128`.
  Set to `0` for unbounded.
- **Token budget:** `sum(tokens_to_run_this_step) <=
  --max-num-batched-tokens`. Default `2048`.
- **Per-request cap (chunked prefill):**
  `tokens_for_this_request_this_step <= --long-prefill-token-threshold`.
  Default `0` = disabled.

`schedule_with_prefix` additionally maintains the per-instance
`MemoryModel.npu_prefix_cache` (RadixCache). Details on
**[Prefix caching](./prefix-caching)**.

## What the scheduler picks each step

```mermaid
flowchart TD
    START([Iteration start]) --> INIT[remaining_budget = max_num_batched_tokens<br/>batch = []]
    INIT --> NEXT{More requests<br/>in queue?}
    NEXT -->|No| RETURN[Return Batch or None]
    NEXT -->|Yes| CAP{batch size<br/>>= max_num_seqs?}
    CAP -->|Yes| RETURN
    CAP -->|No| NEED[Compute needs:<br/>prefill chunk OR decode 1 token]
    NEED --> MIN[cap = min remaining_budget,<br/>long_prefill_threshold,<br/>tokens_needed]
    MIN --> CHECKCAP{cap > 0?}
    CHECKCAP -->|No| NEXT
    CHECKCAP -->|Yes| MEM{Memory fits<br/>after eviction?}
    MEM -->|No| NEXT
    MEM -->|Yes| ADD[Add to batch<br/>budget -= cap]
    ADD --> NEXT
```

Conceptually, the loop is:

```
remaining_token_budget = max_num_batched_tokens
batch = []
for request in queue (FIFO, prefill-first if prioritized):
    if len(batch) >= max_num_seqs: break

    needs_to_run = how_many_tokens_this_request_needs(request)
    cap = min(remaining_token_budget,
              long_prefill_token_threshold or remaining_token_budget,
              needs_to_run)
    if cap <= 0:
        continue       # try the next request

    schedule(request, tokens=cap)
    remaining_token_budget -= cap
    batch.append(request)

return Batch(batch) if batch else None
```

The "how many tokens this request needs" function differs by request
state:

- **Prefill, no chunk yet:** input length minus any prefix cache hit.
- **Prefill, mid-chunk:** remaining prompt tokens.
- **Decode:** always 1.

## Chunked prefill

`--long-prefill-token-threshold N` (or `--enable-chunked-prefill`
which sets a sensible default) lets the scheduler split a long prefill
across multiple iterations. Without it, a single 32k-token request
hogs the whole budget and TPOT for other in-flight requests
collapses.

Concretely, a request whose remaining prefill is 8000 tokens with
`--long-prefill-token-threshold 1024` runs as eight separate
8x1024-token chunks across eight scheduler iterations. The
`Request.num_computed_tokens` field tracks progress; on each
iteration the scheduler bumps it by however many tokens were just
processed.

Decode steps continue to run *concurrently* in the same batch, the
chunked prefill just keeps long prompts from monopolizing.

## Prefill priority

By default, prefill and decode requests share the same FIFO queue.
With `--prioritize-prefill`, the scheduler reorders the batch so all
prefill requests land first, then decodes fill the remaining budget.

This trades some TPOT for lower TTFT under bursty arrivals, useful
in deployments where users care more about "first token latency"
than steady-state generation rate.

## Pipeline depth (PP)

For `pp_size > 1` instances, the scheduler also keeps an `inflight`
list of batches currently traversing the pipeline. Its length is
capped at `pp_size`: when the pipeline is full, the scheduler
returns `None` until ASTRA-Sim drains a stage.

This makes the simulator's PP behavior match production training
frameworks (e.g., Megatron) where micro-batches stream through the
pipeline.

## Where the scheduler stops

The simulator exits when, simultaneously:

- Every scheduler returns `None` (no eligible requests).
- `Router.has_pending_requests()` returns `False` (no future arrivals).
- `Router.has_deferred_sessions()` returns `False` (no agentic sessions
  waiting on tool calls).

If only the third is non-empty, the main loop fast-forwards `current`
to the next pending arrival time and resumes.

## What the scheduler hands back

`scheduler.add_done(npu_id, sys, current)` is called once per
iteration when ASTRA-Sim reports completion. It returns:

```python
(prompt_throughput, decode_throughput, finished_requests)
```

- `prompt_throughput` counts **all input tokens including prefix
  cache hits**, matching vLLM's reporting (which also counts cached
  tokens). `decode_throughput` counts only newly generated tokens.
- `finished_requests` is the list of requests that completed during
  this iteration.

For prefill instances under P/D disaggregation, the main loop hands
`finished_requests` to `router.transfer_prefill_request` so the
decode instance picks them up.

## Gotchas

1. **Prefill plus prefix caching** doesn't double-count: `hit_len` is
   subtracted from the tokens the scheduler actually runs, but
   *added* to `prompt_throughput`. So a 1000-token request with 600
   tokens of prefix hit consumes 400 tokens of budget and reports
   1000 tokens of prompt throughput.

2. **`--max-num-seqs 0` means unlimited**, not zero. Useful when you
   want pure token-budget gating, but watch memory.

3. **The token budget is shared across prefill + decode.** A batch
   with 64 in-progress decodes and a 1500-token prefill chunk runs
   1564 tokens this step. Decode contributions count.

4. **Pipeline parallelism caps `inflight` at `pp_size`.** Chakra splits
   each iteration's layers across stages with send/recv between them,
   so inter-stage P2P latency *is* modeled.

## What's next

- **[Prefix caching](./prefix-caching)**: what `hit_len` means and
  how the RadixCache decides it.
- **[KV cache & memory](./kv-cache-and-memory)**: how the scheduler
  knows when memory is full.

---
title: Reading the output
sidebar_position: 8
---

# Reading the output

The simulator produces five kinds of output:

1. **Per-request CSV** at the path passed via `--output`.
2. **Throughput log line** printed every `--log-interval` seconds.
3. **Final power summary** (only if the cluster config has a
   `power:` block).
4. **Session metrics JSON** at `--session-metrics` for agentic load and
   fixed-cohort measurements.
5. **Agentic KV metrics JSON** at `--agentic-kv-metrics` (only when
   idle session-KV management is enabled).

This page covers what each one means and how to read them.

## Per-request CSV

When you pass `--output outputs/foo.csv`, the simulator writes one
row per finished request:

```csv
instance id,request id,model,input,output,generated_tokens,arrival,end_time,latency,queuing_delay,first_schedule_time_ns,first_schedule_eligibility_time_ns,scheduler_queue_wait_ns,TTFT,TPOT,ITL,session_id,sub_request_index,source_session_id,session_template_index,session_epoch,session_offered_time_ns,session_admission_time_ns,session_admission_queue_wait_ns,prefix_reuse_tokens,prefix_reuse_source,return_gap_type,return_gap_source,return_gap_ns,agentic_kv_hit_tokens,agentic_kv_recompute_tokens,agentic_kv_residency_at_return,agentic_kv_source,agentic_kv_restore_ns,agentic_kv_owner_gate_ns,agentic_kv_restore_issue_time_ns,agentic_kv_target_hbm_ready_time_ns,agentic_kv_restore_ready_time_ns,agentic_kv_fresh_prompt_tokens,agentic_kv_overlap_cutoff_tokens,agentic_kv_restore_compute_overlap_ns,agentic_kv_restore_gate_wait_ns,pd_pair_fifo_wait_ns,agentic_kv_prepare_boundary_wait_ns,agentic_kv_source_demotion_join_wait_ns,agentic_kv_hbm_admission_wait_ns,agentic_kv_transient_dram_capacity_wait_ns,agentic_kv_restore_queue_wait_ns,agentic_kv_restore_service_ns,pd_decode_capacity_wait_ns,pd_decode_admission_wait_ns,pd_decode_admission_critical_wait_ns,pd_prefill_capacity_wait_ns,pd_prefill_admission_wait_ns,pd_prefill_admission_critical_wait_ns,pd_launch_admission_wait_ns,pd_launch_admission_critical_wait_ns,pd_chunk_admission_count,pd_chunk_cancelled_admission_count,pd_chunk_admitted_tokens_total,pd_chunk_prefill_admitted_per_rank_bytes,pd_chunk_decode_admitted_per_rank_bytes,pd_chunk_admission_wait_ns_total,pd_chunk_admission_critical_wait_ns_total,pd_chunk_successful_admission_wait_ns_total,pd_chunk_successful_admission_critical_wait_ns_total,pd_chunk_cancelled_admission_wait_ns_total,pd_chunk_cancelled_admission_critical_wait_ns_total,pd_chunk_prefill_peak_hbm_used_per_rank_bytes,pd_chunk_decode_peak_hbm_used_per_rank_bytes,pd_prefill_initial_restored_per_rank_bytes,pd_prefill_handoff_released_per_rank_bytes,pd_decode_handoff_owned_per_rank_bytes,active_prefill_recompute_preemptions,active_prefill_recompute_tokens,active_prefill_recompute_frontier_tokens,pd_active_prefill_recompute_generation,agentic_kv_restored_tokens_discarded_by_active_prefill_recompute,pd_kv_ownership_state
0,0,Qwen/Qwen3-30B-A3B-Instruct-2507,1472,133,4059740,1082836204,1078776464,0,51162321,7784955,"[7780422, 7779379, 7779523, ...]"
0,3,meta-llama/Llama-3.1-8B,4,16,570907776,711600111,140692335,3739551,15137413,11414083,"[11043655, 11381158, ...]"
...
```

The bundled `outputs/example_*_run.csv` files (one per scenario in
`serving/run.sh`) are good examples to skim.

### Column reference

| Column | Type | Meaning |
| --- | --- | --- |
| `instance id` | int | Which serving instance ran this request |
| `request id` | int | Monotonic id assigned by the router |
| `model` | string | Model name (e.g., `meta-llama/Llama-3.1-8B`) |
| `input` | int | Prompt tokens (full input length, including any prefix-cache hits) |
| `output` | int | Requested decode-token count |
| `generated_tokens` | int | Decode tokens actually completed; validated paper runs require this to equal `output` |
| `arrival` | int (ns) | When the request arrived (simulator clock) |
| `end_time` | int (ns) | When the last generated token completed |
| `latency` | int (ns) | End-to-end latency: `end_time - arrival` |
| `queuing_delay` | int (ns) | From arrival to first scheduling step |
| `first_schedule_time_ns` | int (ns) | First batch-dispatch epoch for this request |
| `first_schedule_eligibility_time_ns` | int (ns) | Earliest epoch at which all owner-ready dependencies permit scheduler admission |
| `scheduler_queue_wait_ns` | int (ns) | `first_schedule_time_ns - first_schedule_eligibility_time_ns`; excludes the owner-ready gate |
| `TTFT` | int (ns) | Time-to-first-token: first-token-completion minus `arrival` |
| `TPOT` | int (ns) | Mean time-per-output-token: `(latency - TTFT) // (output - 1)` (or `0` when `output == 1`) |
| `ITL` | string | Inter-token latencies, ns. Serialized Python list, e.g. `"[7780422, 7779379, ...]"` |
| `session_id` | string or empty | Agentic dependency-chain identity |
| `sub_request_index` | int or empty | Zero-based turn index inside the session |
| `source_session_id` | string or empty | Identity in the source trace before backlog/Poisson replication |
| `session_template_index` | int or empty | Selected complete-session template index |
| `session_epoch` | int or empty | Backlog/Poisson repetition epoch |
| `session_offered_time_ns` | int | When the session entered the offered population |
| `session_admission_time_ns` | int | When the session entered the active population |
| `session_admission_queue_wait_ns` | int | Session-level admission queue wait |
| `prefix_reuse_tokens` | int | Reusable prefix declared or derived for this continuation |
| `prefix_reuse_source` | string | `exact`, `reported`, or `estimated` reuse provenance |
| `return_gap_type` | string | Incoming `session_start`, `human`, `tool`, `mixed`, or `unknown` class copied from the previous call's outgoing gap |
| `return_gap_source` | string | Provenance for the incoming completion-to-ready interval |
| `return_gap_ns` | int | Incoming completion-to-ready interval; excludes the preceding LLM execution |
| `agentic_kv_hit_tokens` | int | Logical reusable tokens restored or retained by the session tier |
| `agentic_kv_recompute_tokens` | int | Reusable-prefix tokens that must be recomputed after a miss or partial hit |
| `agentic_kv_residency_at_return` | string or empty | HBM/CPU/SSD/dropped state observed before restore; preserved if restore admission later fails |
| `agentic_kv_source` | string or empty | Resume source: HBM, CPU, SSD, or dropped |
| `agentic_kv_restore_ns` | int | Physical restore only: destination-HBM admission + transfer-resource queue + service. It excludes pair-FIFO and prepare-boundary admission |
| `agentic_kv_owner_gate_ns` | int | Complete request-ready-to-restore-ready gate: pair FIFO + prepare boundary + physical restore |
| `agentic_kv_restore_issue_time_ns` | int | Physical-restore issue epoch, after pair-FIFO and prepare-boundary admission |
| `agentic_kv_target_hbm_ready_time_ns` | int | Epoch at which destination-HBM capacity is ready; canonical pre-admission still withholds the owner until the full restore finishes |
| `agentic_kv_restore_ready_time_ns` | int | Epoch at which every required KV byte is available to the request |
| `agentic_kv_fresh_prompt_tokens` | int | Prompt tokens not covered by the restored reusable prefix |
| `agentic_kv_overlap_cutoff_tokens` | int or empty | Furthest cumulative prompt position allowed before restore completes; empty outside `async-decode-join` overlap |
| `agentic_kv_restore_compute_overlap_ns` | int | Observed execution-time intersection between this request's fresh prefill and its restore interval |
| `agentic_kv_restore_gate_wait_ns` | int | In-scheduler join wait for `async-decode-join`; zero in canonical `async-pre-admission`, whose delay appears in `agentic_kv_owner_gate_ns` |
| `pd_pair_fifo_wait_ns` | int | Admission wait behind an earlier preparation or pending P→D handoff on the same fixed P/D pair; not transfer time |
| `agentic_kv_prepare_boundary_wait_ns` | int | Wait for relevant P/D engines to reach a safe preparation boundary; not transfer time |
| `agentic_kv_source_demotion_join_wait_ns` | int | Request-visible tail of a source demotion already in flight at return; separate from restore service |
| `agentic_kv_hbm_admission_wait_ns` | int | Destination-HBM capacity wait inside the physical restore interval |
| `agentic_kv_transient_dram_capacity_wait_ns` | int | SSD bounce-buffer capacity-pressure diagnostic; do not add it again to the owner gate |
| `agentic_kv_restore_queue_wait_ns` | int | Transfer-resource queue wait inside the physical restore interval |
| `agentic_kv_restore_service_ns` | int | Isolated transfer service inside the physical restore interval |
| `pd_decode_capacity_wait_ns` | int | Time until the D-side full-prompt HBM claim becomes capacity-ready |
| `pd_decode_admission_wait_ns` | int | Time until the paired P/D gate physically admits both sides |
| `pd_decode_admission_critical_wait_ns` | int | D capacity-ready delay not hidden by restore; side diagnostic, not additive |
| `pd_prefill_capacity_wait_ns` | int | Time until the P-side full-prompt HBM claim becomes capacity-ready |
| `pd_prefill_admission_wait_ns` | int | Time until the paired P/D gate physically admits both sides |
| `pd_prefill_admission_critical_wait_ns` | int | P capacity-ready delay not hidden by restore; side diagnostic, not additive |
| `pd_launch_admission_wait_ns` | int | Canonical one-per-request wait until the atomic P/D launch gate admits both sides |
| `pd_launch_admission_critical_wait_ns` | int | Canonical P/D admission delay not hidden by restore; use this causal field instead of summing P and D |
| `pd_chunk_admission_count` | int | Successful incremental atomic P/D chunk claims |
| `pd_chunk_cancelled_admission_count` | int | Pre-commit chunk claims cancelled by active-prefill recomputation |
| `pd_chunk_admission_wait_ns_total` | int | Gross request-local wait over successful and cancelled chunk attempts |
| `pd_chunk_admission_critical_wait_ns_total` | int | Gross post-restore critical wait over successful and cancelled chunk attempts |
| `pd_chunk_successful_admission_wait_ns_total` | int | Wait charged only to successful chunk claims |
| `pd_chunk_successful_admission_critical_wait_ns_total` | int | Post-restore critical wait charged only to successful chunk claims |
| `pd_chunk_cancelled_admission_wait_ns_total` | int | Wait already spent by chunk claims cancelled before commit |
| `pd_chunk_cancelled_admission_critical_wait_ns_total` | int | Post-restore part of cancelled-claim wait |
| `active_prefill_recompute_preemptions` | int | Number of finite-HBM P/D prefill preemptions for this request |
| `active_prefill_recompute_tokens` | int | Cumulative discarded partial-prefill tokens over all such preemptions |
| `active_prefill_recompute_frontier_tokens` | int | Largest discarded prefix frontier; this is not cumulative work |
| `pd_active_prefill_recompute_generation` | int | Monotonic replay generation; equal to the request's active-prefill preemption count |
| `agentic_kv_restored_tokens_discarded_by_active_prefill_recompute` | int | Original restored hit tokens discarded exactly once at the first active-prefill preemption |
| `pd_kv_ownership_state` | string | Current P/D KV lifecycle state at request completion |

All times are in **nanoseconds**. Divide by `1e9` for seconds, `1e6`
for milliseconds. Column names use spaces, not underscores; quote
them in pandas (`df["instance id"]`).

The agentic columns are present for every run and remain empty or zero for
flat requests. Logical token counters are not block-rounded; physical traffic
in the metrics JSON is block-rounded.

Every successfully prepared continuation must satisfy:

```text
restore_issue_time = arrival + pd_pair_fifo_wait
                              + prepare_boundary_wait
restore_ns = hbm_admission_wait + restore_queue_wait + restore_service
owner_gate_ns = pd_pair_fifo_wait + prepare_boundary_wait + restore_ns
restore_ready_time = arrival + owner_gate_ns
pd_chunk_attempt_wait = pd_chunk_successful_wait
                      + pd_chunk_cancelled_wait
pd_chunk_attempt_critical_wait = pd_chunk_successful_critical_wait
                               + pd_chunk_cancelled_critical_wait
canonical_hbm_capacity_wait = hbm_admission_wait
                            + pd_chunk_attempt_critical_wait
```

The online experiment collector fails a run if any component is negative or
these timestamps do not reconcile. In particular, HBM-local or dropped
requests may have `restore_ns=0` while retaining a nonzero admission wait; do
not relabel that delay as swap service.

### Common derived metrics

```python
import pandas as pd
df = pd.read_csv("outputs/foo.csv")

# Wall-clock TTFT in milliseconds
df["TTFT_ms"] = df["TTFT"] / 1e6

# TPOT in milliseconds (already a per-token mean; divide for ms)
df["TPOT_ms"] = df["TPOT"] / 1e6

# End-to-end latency in seconds
df["latency_s"] = df["latency"] / 1e9

# Throughput across the whole run (tokens / second)
total_tokens = (df["input"] + df["output"]).sum()
sim_duration_s = (df["end_time"].max() - df["arrival"].min()) / 1e9
throughput = total_tokens / sim_duration_s

# Per-instance distribution
per_inst = df.groupby("instance id").agg(
    requests=("request id", "count"),
    p50_TTFT_ms=("TTFT", lambda x: x.quantile(0.5) / 1e6),
    p99_TTFT_ms=("TTFT", lambda x: x.quantile(0.99) / 1e6),
)

# Inter-token latency: parse the ITL string back into a list per row
import ast
df["ITL_list"] = df["ITL"].apply(ast.literal_eval)
df["ITL_p50_ms"] = df["ITL_list"].apply(lambda xs: pd.Series(xs).quantile(0.5) / 1e6)
```

## Session metrics JSON

Pass `--session-metrics outputs/run.session.json` to retain session admission,
completion, measurement-window, request, and timing-validation records. The
most important measurement distinction is between an exact session cohort and
a completion-time window:

- `requests.all` and `throughput.completed_requests_in_session_cohort` cover
  every request belonging to the sessions whose lifecycle row has
  `measurement_included: true`. Resume-source fractions, including SSD resume
  as a fraction of all requests, use this exact session-cohort denominator.
- `throughput.completed_requests`, prompt/generated tokens, and request/token
  throughput cover every request completion in the strict interval
  `measurement_start_ns < end_time <= measurement_end_ns`. Concurrent
  non-target sessions may contribute to this time-window throughput.
- `throughput.completed_sessions` and session throughput use the measured
  session count divided by that same interval duration.

Session-metrics schema 9 makes the admission and cutoff contract explicit
under `session_admission`:

| Field | Meaning |
| --- | --- |
| `queue_policy` | `fifo_wait_for_slot` for backlog mode; `arrival_time_order` for trace/Poisson admission |
| `logical_session_drop_count` | Count of lifecycle rows whose status is literally `dropped`; validated bounded runs require zero |
| `slot_release_event` | `final_request_completion_on_decode_owner` for P/D execution, or `final_request_completion_on_colocated_owner` without a decode tier |
| `slot_release_event_legacy` | Compatibility-only alias: `final_decode_completion` or `final_llm_request_completion`; do not infer that every final request launches a decode graph |
| `cutoff_disposition` | `drain` for a full run or `right_censor` for an explicit measurement cutoff |

In backlog mode, an offered session that is beyond the active limit is kept in
the FIFO backlog until a slot becomes available. An admitted session owns its
slot through all LLM turns, human/tool waits, prefill, and P→D handoffs; prefill
completion does not release it. With P/D disaggregation, the final request's
completion on the decode owner releases the slot and admits the next waiting
session at that logical timestamp. If that final request has `output_toks=1`,
its only token is produced by prefill and ownership completes after the P→D
handoff; no D-side model graph is launched. A measurement cutoff changes
unfinished lifecycle rows to `censored` and releases their resources during
audited cleanup. It is not a logical-session drop.

For `measurement_cohort_selection: admission_order`, let
`W = warmup_completions` and `M = measure_completions`. The first `W` sessions
in deterministic epoch-major backlog admission order are a fixed, excluded
warmup prefix; the next `M` sessions are the measured target. The early-stop
boundary requires the complete `W + M` prefix, not merely whichever target
sessions happen to complete first. This fixed warmup is not a temporal barrier:
target execution may overlap unfinished warmup sessions.
`measurement_window` records:

| Field | Meaning |
| --- | --- |
| `measurement_warmup_session_ids` | Ordered fixed admission-prefix IDs excluded from the measured target |
| `measurement_warmup_session_count`, `measurement_warmup_completed_sessions` | Requested fixed-prefix warmup size and its completed count |
| `measurement_target_session_ids` | Ordered runtime-session IDs in the fixed cohort |
| `measurement_target_session_count` | Number of ordered targets |
| `measurement_target_completed_sessions` | Targets completed at report time; a validated run requires the full target count |
| `measurement_target_session_ids_hash` | SHA-256 of the ordered ID list encoded as canonical compact JSON |
| `measurement_required_session_ids` | Ordered warmup IDs followed by target IDs; all must complete before an early cutoff |
| `measurement_required_session_count`, `measurement_required_completed_sessions` | Size and completed count of the `W + M` required prefix |
| `measurement_start_ns` | Minimum admission timestamp among target sessions |
| `measurement_end_ns` | Maximum completion timestamp among target sessions |
| `measurement_duration_ns` | End minus start |
| `target_semantics`, `target_order_and_hash_semantics` | Self-describing selection and digest contract |
| `start_semantics`, `end_semantics` | Self-describing boundary contract |

Each `sessions.records` lifecycle row also carries
`planned_admission_index`, `admission_index`, `measurement_warmup`,
`measurement_target`, `measurement_required`, and `measurement_included`. The
bounded online runner reconstructs the expected runtime IDs from the
materialized workload, verifies the ordered FIFO prefix, indices, bounds, and
digests, and compares exact measured membership across every policy and its
strict oracle. Its `summary.csv`/`summary.json` rows copy all three prefix
definitions. The separate
`measured_session_ids_hash` is a membership digest used by
`measured_completion_cohort_matches_oracle`; despite that legacy column name,
the check applies to either cohort-selection mode.

The bounded-run summary also copies the admission contract and disambiguates
the historical `dropped` resume label:

| Summary field | Meaning |
| --- | --- |
| `queue_policy` | Session-report queue policy; backlog runs require `fifo_wait_for_slot` |
| `logical_session_drop_count` | Literal logical-session drops; validated bounded runs require zero |
| `cutoff_disposition` | `drain` or explicit `right_censor` |
| `slot_release_event` | Precise owner-side final-request completion event |
| `slot_release_event_legacy` | Compatibility-only historical slot-release label |
| `kv_state_unavailable_resume_count` | Continuations with `agentic_kv_source=dropped` and positive recompute tokens; zero-overlap returns are excluded and the logical session still exists |
| `kv_state_unavailable_resume_fraction_of_all_requests` | Same count divided by every request in the fixed session cohort |
| `kv_state_unavailable_resume_fraction_of_resume_requests` | Same count divided by continuation requests only |
| `zero_overlap_resume_count` | Continuations with no reusable prefix overlap (`hit_tokens=0` and `recompute_tokens=0`); these are neither physical restores nor KV-unavailable misses |
| `zero_overlap_resume_fraction_of_all_requests` | Zero-overlap continuations divided by every request in the fixed session cohort |
| `dropped_resume_columns_semantics` | Declares `dropped_resume_*` to be legacy numeric aliases of the `kv_state_unavailable_*` fields, never logical-session-drop metrics |
| `raw_dropped_resume_source_count` | Every raw `source=dropped` continuation event, including zero-overlap object release; diagnostic only |
| `attempted_{hbm,cpu,ssd}_resume_count` | Requests that physically selected that resume source and restored or retained at least one hit token |
| `effective_surviving_{hbm,cpu,ssd}_resume_count` | Attempted source requests with at least one original restored hit token still surviving after active-prefill replay |
| `attempted_{hbm,cpu,ssd}_resume_fraction_of_all_requests` | Attempted count divided by every completed request in the measured-session cohort |
| `effective_surviving_{hbm,cpu,ssd}_resume_fraction_of_all_requests` | Effective-surviving count over the same all-request denominator |
| `attempted_restored_hit_tokens` | Original hit tokens physically restored or retained in the measured cohort |
| `restored_hit_tokens_discarded_by_active_prefill_recompute` | Attempted hit tokens later invalidated; charged once even if the request is preempted repeatedly |
| `effective_surviving_hit_tokens` | Attempted hit tokens minus the active-prefill discard total |
| `attempted_resume_by_return_gap_type_and_source_json` | Human/tool × physical-source request counts before active-prefill survival filtering |
| `effective_surviving_resume_by_return_gap_type_and_source_json` | The same cross-tab after survival filtering |

The legacy `dropped_resume_count`,
`dropped_resume_fraction_of_all_requests`, and
`dropped_resume_fraction_of_resume_requests` remain for downstream CSV
compatibility. Prefer the `kv_state_unavailable_*` names in new analysis.

The manager `agentic_kv_source` label is intentionally immutable. Legacy
HBM/CPU/SSD source-count columns can include a zero-hit object release and are
not authoritative physical-resume counters. Attempted counts additionally
require `agentic_kv_hit_tokens > 0`. A CPU or SSD restore that is later
discarded by finite-HBM active-prefill preemption still consumed that I/O
path, so it remains an attempted CPU or SSD resume.
Use the `effective_surviving_*` fields for cache reuse that actually survives
to the final prompt handoff. Both fraction families use all completed measured
requests, including initial calls and non-reuse returns, as the denominator.

Agentic-KV report schema 20 adds a fail-closed
`pd_active_prefill_recompute_accounting` audit and separates successful,
cancelled, and gross P/D chunk-attempt wait. Session report schema 11 and
online artifact schema 12 carry the corresponding exact per-request fields.
The collector compares CSV and session JSON for request/session identity,
source and gap class, hit/recompute tokens, P/D counts and waits, and the
active-prefill fields before producing summary rows. It also
joins manager `resume`, successful/cancelled P/D chunk, and active-prefill
preemption events to those records by `request_id`. Resume events preserve the
sub-request index, physical source, hit/recompute tokens, and return-gap class;
their global source counts and token sums must reconcile with manager totals.
The summary exposes this fail-closed result as
`cross_layer_request_accounting_audit_passed`.

The SSD opportunity contract uses
`attempted_ssd_resume_count / session_cohort_request_count`; the raw legacy
SSD source-label fraction is not a physical-I/O opportunity metric.

The online summary's canonical
`total_hbm_capacity_admission_wait_{sum,mean,p95}_ns` is the per-request sum
of destination-HBM restore admission and the gross post-restore P/D chunk
critical tail, including both successful and pre-commit-cancelled attempts.
It deliberately does not add gross enqueue-to-admission chunk wall time,
because that interval can overlap the restore destination gate. The separate
`pd_chunk_attempt_admission_wait_*` columns retain that gross wall-time view;
`pd_chunk_hbm_capacity_admission_wait_*` is the additive critical component.

For `tiered_queue_recompute`, agentic-KV report schema 19 changes the policy
unit from a whole entry to a contiguous block-aligned prefix. The
`queue_recompute_policy` object exposes:

| Field | Meaning |
| --- | --- |
| `full_restore_decisions`, `partial_restore_decisions`, `zero_restore_decisions` | Disjoint partition of evaluated CPU/SSD resumes. Legacy `drop_decisions` is exactly the `H=0` subset |
| `selected_restore_tokens`, `selected_restore_bytes` | Aggregate retained prefix `H` and its physical communication bytes over modified decisions |
| `dropped_suffix_tokens`, `dropped_suffix_bytes` | Aggregate `[H,R)` recomputation and avoided foreground communication |
| `modified_full_projected_*` | Full-`R` shadow components which triggered the severe gate |
| `partial_prefix_projected_*` | Components of the selected nonzero prefix restore |
| `configured_prefill_headroom_chunks` | Causal P/D capacity-snapshot horizon; `1.0` means the next runtime prefill chunk |
| `accounting_invariants` | Fail-closed checks for decision partitioning, block alignment, token/byte conservation, both policy gates, and zero logical-session drops |

The corresponding `queue_recompute_evaluate` event records
`reusable_tokens_R`, `selected_prefix_tokens_H`, the candidate set, full and
prefix projections, predicted path costs, selection reason, and a timestamped
`capacity_headroom_snapshot`. A modified decision emits
`queue_recompute_partial` for `0 < H < R` or the compatibility
`queue_recompute_drop` event for `H=0`. Snapshot capacity is not reserved by
the policy and does not guarantee zero later P/D chunk-admission wait. For a
partial restore, the full authoritative CPU/SSD source (and any durable SSD
duplicate) stays pinned until the `H`-byte DMA completes; the source is then
released and only `[H,R)` enters normal prefill recomputation.

Timing warnings are machine-readable under
`validation.timing.warning_codes`, aligned one-to-one with
`validation.timing.warnings`. A bounded spec's
`allowed_timing_warning_codes` accepts only the named codes. The older
`allow_timing_warnings: true` accepts all recognized warnings only when no
allowlist is present; invariant violations are never converted into warnings.
See [Agentic sessions → Timing-warning contracts](/docs/workloads/agentic-sessions#timing-warning-contracts)
for the supported codes and precedence rules.

## Agentic KV metrics JSON

Pass `--agentic-kv-metrics outputs/run.storage.json` to retain migration
events, resource-queue utilization, residency, SSD bytes, and denominated
overheads. The two primary migration ratios answer different questions:

- `migration_restore_exposure_fraction_of_makespan` is the union of
  overlapping request-blocking migration intervals divided by simulated
  makespan. This includes the visible tail of an already-running source
  demotion and the foreground restore. It answers “for what fraction of wall
  time was at least one request waiting on migration?”, not “what fraction of
  makespan was wasted?”
- `migration_stall_fraction_of_total_request_latency` is summed migration
  restore divided by summed request latency. Pair-FIFO and prepare-boundary
  admission are separate. Simultaneous requests contribute separately, so do
  not call this a wall-time fraction.

Use `time_breakdown.aggregate_pd_pair_fifo_wait_ns`,
`aggregate_prepare_boundary_wait_ns`,
`aggregate_source_demotion_join_wait_ns`, and
`aggregate_owner_ready_gate_ns` for the complete preparation decomposition.
The last field equals the first three plus the physical swap-in time. The
source-demotion field is only the tail exposed after the session returned; do
not add the full background swap-out service again. These agentic-KV aggregates
cover continuation preparations; the session-metrics request distributions
also retain an initial request that waited for its fixed pair. Do not add the
P-side and D-side admission audits because they observe the same atomic launch
gate.

`migration_makespan_penalty_fraction` is reserved for the causal paired-run
result and remains `null` in a single run.

`recompute_token_fraction` is always a token-domain result when a prompt-token
denominator exists. `recompute_fraction_of_total_model_compute` is separate and
may be `null` unless the execution backend can isolate recompute kernels and
all model compute. Exact wall-time penalty needs a paired zero-migration run
with identical placement/hits, or a measurement of restore intervals that
leave no useful model work runnable. See the
[agentic idle-KV example](/docs/examples/memory-tiers/agentic-idle-kv-tiering#reading-overhead-without-ambiguous-denominators)
for the queue model and interpretation.

For policy comparison, prefer
`policy_avoidable_recompute_fraction_of_executed_prefill`. It subtracts the
mandatory one-token execution cap from full-prefix hits before dividing by
the prefill tokens the model actually executes. The raw `recompute_tokens`
counter retains that cap because it describes executed work, not solely work
caused by an eviction decision.

Agentic-KV schema 14 includes literal all-agentic-request cross-tabs under
`request_classification` and batch-membership cross-tabs under
`batch_composition`. The latter can count one request repeatedly across
chunked-prefill or decode iterations. It also reports source-demotion join
wait separately from physical swap-in service. Schema 14 also makes every raw
`events[].event == "drop"` record self-classifying:

| Field | Value or interpretation |
| --- | --- |
| `object_scope` | Always `kv_cache_entry` |
| `logical_session_effect` | Always `none`; this event neither removes the session nor releases its admission slot |
| `drop_class` | Stable semantic class; use this rather than counting the legacy event name |
| `reason` | Concrete trigger within the class |

The class mapping is:

| `drop_class` | `reason` values | Meaning |
| --- | --- | --- |
| `capacity_loss` | `hbm_capacity`, `cpu_capacity`, `ssd_capacity` | Reusable KV was discarded because a tier was full |
| `ttl_loss` | `ssd_ttl` | Reusable KV was discarded by an age/TTL sensitivity policy |
| `resume_recompute_cleanup` | `resume_miss` | An obsolete KV copy was released after recomputation fallback |
| `normal_session_cleanup` | `session_end` | KV was released after normal logical-session completion |
| `measurement_cleanup` | `measurement_censor` | KV was released while an unfinished session was right-censored |

The same definitions are embedded in `event_semantics.drop.classes`. Raw
`drop` counts mix loss, normal cleanup, and cutoff cleanup, so they are not a
capacity-miss metric or a logical-session-drop metric.

The three capacity-only headline configs use `async-pre-admission`. Swap-out is
a background copy and never installs a model-engine barrier, although it still
occupies the configured PCIe, DRAM, or storage calendars. HBM-resident D→P
restore is submitted to the shared congestion-aware ASTRA topology; it
contends on topology links and physical endpoints with ordinary P→D and TP/EP
communication without blocking unrelated graph dispatch.
Swap-in is issued when the dependent request becomes ready, after destination
HBM has been reserved. The owner request is withheld from compute batches until
the complete reusable prefix arrives; unrelated HBM-resident requests remain
runnable. Tool and human return classes stay separate in the report, but the
capacity-only policy does not predict an unknown return or inspect a known gap
duration.

Read `asynchronous_restore.aggregate_swap_in_gross_ns` as raw restore work,
`aggregate_prefill_execution_overlap_ns` as observed overlap, and
`aggregate_owner_decode_barrier_ns` as the request-local exposed wait.
`aggregate_other_hidden_ns` is the remaining raw interval hidden by admission
or other concurrency. Source × return-class cells are available under
`asynchronous_restore.by_source_and_return_gap_type`. `async-decode-join`
remains an explicitly labeled idealized sensitivity: it lets the owner execute
a fresh-prompt region before the final prompt token joins the restore, which
would require a layerwise streaming implementation on real hardware.

`observed_load_activity` reports model-execution and transfer interval unions
under the selected session-load mode and closed-loop dependency timing. In
trace or Poisson mode, `global_all_model_engines_idle_ns` and
`fully_quiescent_ns` may be nonzero because no session may be ready. Backlog
mode immediately replaces a completed session while work remains, but internal
human/tool waits can still leave every active session temporarily unready.

When `--session-stop-after-measurement` is enabled, the session report's
`censoring` block records queued, pending-handoff, preallocated, and active
requests removed after already dispatched ASTRA work drains. It includes
memory before/after cleanup and the tier-manager drain audit. Reporting fails
if scheduler NPU ownership does not return to the weight-only baseline, CPU
active ownership is nonzero, or any tier entry, SSD record, HBM claim,
preparation lock, restore wait, ASTRA window, or external cold-fabric job
remains live. `measurement_cutoff_dma_tail.foreground_jobs` must be zero.

`sync-engine-barrier` remains available as an adverse sensitivity. In that
mode a GPU-facing swap drains the affected current iteration and gates the next
dispatch through commit. Raw reservation unions and causally exposed
engine-wait unions remain under `synchronous_swap`; they are zero in both
asynchronous modes. `async-pre-admission` withholds only the returning request
until the complete restore and performs no fresh-prefill overlap.

## Standard output (log levels)

The simulator's `--log-level` flag controls how much detail lands on
stdout while a run is in progress:

| Level | What you see |
| --- | --- |
| `WARNING` (default) | The throughput log line every `--log-interval` seconds, plus warnings (variant fallback, runtime exceeds profiler sweep, MoE config mismatch, etc.) |
| `INFO` | Adds per-iteration scheduler decisions (which requests entered the batch, prefix-cache hits per request) and the request lifecycle (arrival / first token / completion). Useful for debugging routing and scheduling. |
| `DEBUG` | Adds per-layer memory load / store activity, full `Batch` / `Request` dumps, and `npu_prefix_cache.format_prefix_info()` snapshots. Generates a lot of output; pipe to a file. |

Independently of the level, the simulator always emits:

- A startup banner with the resolved `(hardware, model, variant)`
  and the engine_effective comparison vs. `meta.yaml`.
- The final summary on shutdown (Total requests, mean TTFT / TPOT,
  throughput, plus the **power summary** below if `power:` is
  configured).

The throughput log line itself is identical regardless of level,
the only difference is what surrounds it.

## Throughput log line

Every `--log-interval` seconds the simulator prints a one-line
status update. The format adapts to which features are enabled:

### Single-instance baseline

```text
[INFO] step=42 batch=8 prompt_t=1.2k tok/s decode_t=420 tok/s npu_mem=88.4 GB
```

| Field | Meaning |
| --- | --- |
| `step` | Iteration number this interval ended on |
| `batch` | Batch size in requests |
| `prompt_t` | Prompt-side throughput (input tokens/sec, includes prefix hits) |
| `decode_t` | Decode-side throughput (generated tokens/sec) |
| `npu_mem` | NPU memory footprint at this moment |

### Multi-instance

```text
[INFO] step=21 inst0_batch=6 inst1_batch=4 prompt_t=2.5k tok/s decode_t=860 tok/s
       npu_mem=[63.2 GB, 63.2 GB]
```

`inst0_batch` / `inst1_batch` are per-instance batch sizes; `npu_mem`
is per-instance.

### Prefill / decode split

```text
[INFO] step=15 P=8 D=12 prompt_t=3.1k tok/s decode_t=620 tok/s
       npu_mem=[55.4 GB, 71.2 GB]
```

`P=` and `D=` are batch sizes on the prefill and decode instances.

### With prefix sharing

```text
[INFO] step=20 inst0_batch=6 inst1_batch=4 prompt_t=2.4k tok/s decode_t=820 tok/s
       prefix_hit=78% (npu=42%, cpu=36%)
```

The `prefix_hit` field shows the cache hit rate across the interval,
broken down by tier.

### With DP+EP MoE

```text
[INFO] step=8 batch=4+4 prompt_t=1.4k tok/s decode_t=520 tok/s
       npu_mem=[81.2 GB, 81.2 GB] alltoall=512 KB
```

`batch=4+4` shows per-DP-member batches. `alltoall` is the
wave-synchronized ALLTOALL message size.

### With PIM offload

```text
[INFO] step=10 batch=8 prompt_t=1.1k tok/s decode_t=520 tok/s
       npu_mem=63.4 GB pim_busy=72%
```

`pim_busy` is the fraction of the interval the PIM device was active.
At ~100% PIM is your bottleneck.

### With CXL memory

```text
[INFO] step=10 batch=4 prompt_t=620 tok/s decode_t=180 tok/s
       npu_mem=12.4 GB cxl_mem=[3.2 GB, 3.1 GB, 3.1 GB, 3.2 GB]
```

`cxl_mem` is per-device usage; `npu_mem` drops because weights are
on CXL.

### With power model

```text
[INFO] step=42 batch=8 prompt_t=1.2k tok/s decode_t=420 tok/s
       npu_mem=88.4 GB power=712 W
```

`power` is the **current** total system power.

## Final power summary

When `--output` is set and the cluster config has a `power:` block,
the simulator emits a per-node energy breakdown at the end:

```text
─────── Power summary (node 0) ───────
   NPU active     :   12,453 J  (78%)
   NPU standby    :    1,012 J   (6%)
   NPU idle       :       89 J   (1%)
   CPU            :    1,233 J   (8%)
   DRAM           :      442 J   (3%)
   Link           :      388 J   (2%)
   Base + NIC + storage : 332 J  (2%)
   ─────────────────────────────────
   Total energy   :   15,949 J
```

For multi-node runs you get one block per node plus a cluster total.
The breakdown is what makes power numbers actionable for
energy-efficiency research, you can see which component dominates.

## Common patterns to look for

### High waiting count, low NPU memory

The throughput log shows large `batch` counts but `npu_mem` is far
below the cluster config's `npu_mem.mem_size`. Likely cause: the
token budget (`--max-num-batched-tokens`) is the bottleneck, not
memory. Bump it.

### Decode TPOT spikes during prefill bursts

A prefill-heavy moment lands in the same batch as ongoing decodes,
the budget gets eaten by prefill, and decode latency stretches.

Mitigations:
- `--enable-chunked-prefill` (default) splits long prefills.
- `--long-prefill-token-threshold N` caps prefill tokens per
  step.
- `--prioritize-prefill` runs prefill first within a budget, trades
  TPOT for TTFT.

### Prefix hit rate near 0%

Either the workload genuinely has no shared prefixes, or you forgot
to pre-tokenize. Check that `input_tok_ids` is populated in the
JSONL (see [Workloads → JSONL format](/docs/workloads/jsonl-format)).

### MoE per-rank latency varies wildly

Set `--expert-routing-policy BALANCED` (default). RR or RAND can
produce uneven loads on small batches. With BALANCED, per-rank
latency should be uniform within ~1%.

### CXL latency dominates TPOT

Weights placed on CXL pay the round-trip on every decode step. If
TPOT looks far worse than expected, check the `placement` block -
moving cold layers (embedding, lm_head) to CXL helps; moving every
decoder block hurts.

## Validation against known references

LLMServingSim is validated end-to-end against real vLLM with sub-3%
error on TTFT / TPOT / throughput on the bundled hardware × model
combos. The validation methodology and per-model results live in
**[bench/](https://github.com/casys-kaist/LLMServingSim/tree/main/bench)**
on GitHub.

## What's next

- **[Reference → CLI flags](/docs/reference/cli-flags)**: every
  flag that affects the output.
- **[Examples](/docs/examples)**: worked configurations to compare
  your output against.

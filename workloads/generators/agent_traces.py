"""Convert public agent traces to LLMServingSim agentic JSONL.

Supported sources:

* ``exgentic``: one OpenTelemetry trace per row. LLM spans are ordered by
  timestamp and the tool/agent wait after call N is measured as
  ``start(N + 1) - end(N)``.
* ``tracelab``: one normalized TraceLab LLM round per row. Reported prompt
  cache accounting is preserved and parallel tool intervals are combined by
  interval union (or by their maximum latency).
* ``lmcache``: one LLM iteration per row with cumulative OpenAI messages.
  ``pre_gap`` belongs to the *current* request, so it is shifted to become the
  preceding request's ``tool_duration_ns``.

The simulator may replay a trace with a model other than the model that
created it. Token-count provenance is therefore explicit on every
sub-request. In particular, ``prefix_reuse_source`` is one of ``exact``
(longest common prefix under ``--tokenizer``), ``reported`` (provider prompt
cache accounting), or ``estimated`` (content/length estimate). Source token
counts are retained whenever they exist.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import subprocess
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence


_NS_PER_SECOND = 1_000_000_000
_NS_PER_MILLISECOND = 1_000_000
_PREFIX_SOURCES = {"exact", "reported", "estimated"}
_TRACELAB_INPUT_EVENT_TYPES = {"user_message", "tool_result"}
_TRACELAB_MODEL_OUTPUT_EVENT_TYPES = {"reasoning", "text", "tool_call"}


class TraceConversionError(ValueError):
    """Raised when a source session cannot be converted safely."""


@dataclass
class ConversionAudit:
    """Running validation and summary counters used by the manifest."""

    source_rows: int = 0
    sessions_seen: int = 0
    sessions_emitted: int = 0
    sessions_skipped: int = 0
    sub_requests_emitted: int = 0
    validation_errors: int = 0
    warning_counts: Counter[str] = field(default_factory=Counter)
    prefix_source_counts: Counter[str] = field(default_factory=Counter)
    token_count_source_counts: Counter[str] = field(default_factory=Counter)
    tool_wait_source_counts: Counter[str] = field(default_factory=Counter)
    inter_turn_gap_type_counts: Counter[str] = field(default_factory=Counter)
    lineage_status_counts: Counter[str] = field(default_factory=Counter)
    observed_provider_hit_tokens: list[int] = field(default_factory=list)
    policy_independent_reuse_tokens: list[int] = field(default_factory=list)
    source_models: set[str] = field(default_factory=set)
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    tool_wait_ns: list[int] = field(default_factory=list)
    turns_per_session: list[int] = field(default_factory=list)

    def warn(self, code: str) -> None:
        self.warning_counts[code] += 1

    def observe(self, session: Mapping[str, Any]) -> None:
        sub_requests = session["sub_requests"]
        self.sessions_emitted += 1
        self.sub_requests_emitted += len(sub_requests)
        self.turns_per_session.append(len(sub_requests))
        metadata = session.get("trace_metadata", {})
        for model in metadata.get("source_models", []):
            if model:
                self.source_models.add(str(model))
        for sub_request in sub_requests:
            input_toks = int(sub_request["input_toks"])
            output_toks = int(sub_request["output_toks"])
            wait_ns = int(sub_request["tool_duration_ns"])
            self.input_tokens.append(input_toks)
            self.output_tokens.append(output_toks)
            self.tool_wait_ns.append(wait_ns)
            self.prefix_source_counts[sub_request["prefix_reuse_source"]] += 1
            self.token_count_source_counts[
                f"input:{sub_request.get('input_token_count_source', 'unknown')}"
            ] += 1
            self.token_count_source_counts[
                f"output:{sub_request.get('output_token_count_source', 'unknown')}"
            ] += 1
            self.tool_wait_source_counts[
                sub_request.get("tool_wait_source", "unknown")
            ] += 1
            self.inter_turn_gap_type_counts[
                sub_request.get("inter_turn_gap_type", "unknown")
            ] += 1
            self.lineage_status_counts[
                sub_request.get("lineage_status", "unknown")
            ] += 1
            if "observed_provider_hit_toks" in sub_request:
                self.observed_provider_hit_tokens.append(
                    int(sub_request["observed_provider_hit_toks"]))
            if "policy_independent_reuse_toks" in sub_request:
                self.policy_independent_reuse_tokens.append(
                    int(sub_request["policy_independent_reuse_toks"]))


def register_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        required=True,
        choices=("exgentic", "tracelab", "lmcache"),
        dest="source_format",
        help="Input dataset schema.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Local .json/.jsonl[.gz] path or HuggingFace dataset id.",
    )
    parser.add_argument(
        "--source-revision",
        default=None,
        dest="source_revision",
        help=(
            "Optional immutable HuggingFace dataset revision (commit SHA or "
            "tag). Recorded in the manifest."
        ),
    )
    parser.add_argument("--output", required=True, help="Output agentic JSONL path.")
    parser.add_argument(
        "--sps",
        required=True,
        type=float,
        help="Synthetic Poisson session arrival rate in sessions/second.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Poisson RNG seed.")
    parser.add_argument(
        "--first-arrival-sec",
        type=float,
        default=0.0,
        dest="first_arrival_sec",
        help="Timestamp assigned to the first session.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        dest="max_sessions",
        help="Maximum emitted sessions (0 means all).",
    )
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=0,
        dest="max_source_rows",
        help="Maximum input rows read (0 means all). May truncate a session.",
    )
    parser.add_argument(
        "--split", default="train", help="HuggingFace split. Default: train."
    )
    parser.add_argument(
        "--hf-config",
        default=None,
        dest="hf_config",
        help="Optional HuggingFace dataset configuration/subset name.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Optional target HF tokenizer. Full-message inputs are retokenized; "
            "reported source output lengths remain unchanged."
        ),
    )
    parser.add_argument(
        "--tokenizer-revision",
        default=None,
        dest="tokenizer_revision",
        help=(
            "Optional immutable HuggingFace tokenizer revision. Recorded in "
            "the manifest."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="trust_remote_code",
        help="Allow tokenizer remote code when --tokenizer is used.",
    )
    parser.add_argument(
        "--tool-wait-mode",
        choices=("union", "max"),
        default="union",
        dest="tool_wait_mode",
        help=(
            "TraceLab fallback aggregation when event-boundary timestamps are "
            "unavailable. Normal conversion uses completion-to-next-input gaps."
        ),
    )
    parser.add_argument(
        "--tracelab-reuse-mode",
        choices=("eligible", "observed"),
        default="eligible",
        dest="tracelab_reuse_mode",
        help=(
            "TraceLab reuse semantics. 'eligible' uses the policy-independent "
            "adjacent-round estimate; 'observed' replays incumbent provider hits."
        ),
    )
    parser.add_argument(
        "--manifest-output",
        default=None,
        dest="manifest_output",
        help="Manifest JSON path. Default: <output>.manifest.json.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on the first invalid session instead of recording/skipping it.",
    )


def run(args: argparse.Namespace) -> int:
    if not math.isfinite(args.sps) or args.sps <= 0:
        raise SystemExit(f"--sps must be positive, got {args.sps!r}")
    if args.max_sessions < 0 or args.max_source_rows < 0:
        raise SystemExit("--max-sessions and --max-source-rows must be non-negative")
    if not math.isfinite(args.first_arrival_sec) or args.first_arrival_sec < 0:
        raise SystemExit("--first-arrival-sec must be finite and non-negative")

    tokenizer = None
    if args.tokenizer:
        tokenizer = _load_tokenizer(
            args.tokenizer,
            args.trust_remote_code,
            revision=args.tokenizer_revision,
        )

    audit = ConversionAudit()
    rows = _load_source_rows(
        args.source,
        split=args.split,
        hf_config=args.hf_config,
        revision=args.source_revision,
        max_rows=args.max_source_rows,
    )
    sessions = convert_rows(
        rows,
        args.source_format,
        tokenizer=tokenizer,
        target_tokenizer=args.tokenizer,
        tool_wait_mode=args.tool_wait_mode,
        tracelab_reuse_mode=args.tracelab_reuse_mode,
        audit=audit,
        strict=args.strict,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    arrival_ns = int(args.first_arrival_sec * _NS_PER_SECOND)
    previous_arrival_ns = -1
    output_hash = hashlib.sha256()
    seen_session_ids: set[str] = set()

    with output_path.open("w", encoding="utf-8") as output_file:
        for session in sessions:
            if args.max_sessions and audit.sessions_emitted >= args.max_sessions:
                break
            if audit.sessions_emitted:
                arrival_ns += int(rng.expovariate(args.sps) * _NS_PER_SECOND)
            session["arrival_time_ns"] = arrival_ns

            errors = validate_session(session)
            session_id = session.get("session_id")
            if session_id in seen_session_ids:
                errors.append(f"duplicate session_id: {session_id!r}")
            if arrival_ns < previous_arrival_ns:
                errors.append("session arrivals are not monotonic")
            if errors:
                audit.validation_errors += len(errors)
                audit.sessions_skipped += 1
                if args.strict:
                    raise TraceConversionError("; ".join(errors))
                audit.warn("invalid_session_skipped")
                continue

            seen_session_ids.add(str(session_id))
            previous_arrival_ns = arrival_ns
            encoded = (
                json.dumps(session, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            output_file.write(encoded.decode("utf-8"))
            output_hash.update(encoded)
            audit.observe(session)

    if audit.sessions_emitted == 0:
        raise TraceConversionError("No valid sessions were emitted")

    manifest_path = (
        Path(args.manifest_output)
        if args.manifest_output
        else Path(str(output_path) + ".manifest.json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        args=args,
        audit=audit,
        output_path=output_path,
        output_sha256=output_hash.hexdigest(),
    )
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2, sort_keys=True)
        manifest_file.write("\n")

    print(
        f"Wrote {audit.sessions_emitted} sessions / "
        f"{audit.sub_requests_emitted} LLM calls -> {output_path}"
    )
    print(
        f"Validation: {manifest['validation']['status']}; "
        f"manifest -> {manifest_path}"
    )
    return 0


def convert_rows(
    rows: Iterable[Mapping[str, Any]],
    source_format: str,
    *,
    tokenizer: Any = None,
    target_tokenizer: str | None = None,
    tool_wait_mode: str = "union",
    tracelab_reuse_mode: str = "eligible",
    audit: ConversionAudit | None = None,
    strict: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield normalized sessions from a supported row iterable."""

    audit = audit or ConversionAudit()
    if source_format == "exgentic":
        for row_index, row in enumerate(rows):
            audit.source_rows += 1
            audit.sessions_seen += 1
            try:
                session = convert_exgentic_session(
                    row,
                    tokenizer=tokenizer,
                    target_tokenizer=target_tokenizer,
                    fallback_index=row_index,
                    audit=audit,
                )
            except (TraceConversionError, TypeError, ValueError) as error:
                audit.sessions_skipped += 1
                audit.warn("exgentic_session_conversion_failed")
                if strict:
                    raise TraceConversionError(
                        f"Exgentic row {row_index}: {error}"
                    ) from error
                continue
            if session is not None:
                yield session
        return

    if source_format not in {"tracelab", "lmcache"}:
        raise TraceConversionError(f"Unsupported source format: {source_format!r}")

    grouped_rows = _group_contiguous_sessions(
        rows,
        source_format=source_format,
        audit=audit,
        strict=strict,
    )
    for fallback_index, (session_id, session_rows) in enumerate(grouped_rows):
        audit.sessions_seen += 1
        try:
            if source_format == "tracelab":
                session = convert_tracelab_session(
                    session_rows,
                    tool_wait_mode=tool_wait_mode,
                    reuse_mode=tracelab_reuse_mode,
                    target_tokenizer=target_tokenizer,
                    session_id_override=session_id,
                    fallback_index=fallback_index,
                    audit=audit,
                )
            else:
                session = convert_lmcache_session(
                    session_rows,
                    tokenizer=tokenizer,
                    target_tokenizer=target_tokenizer,
                    session_id_override=session_id,
                    fallback_index=fallback_index,
                    audit=audit,
                )
        except (TraceConversionError, TypeError, ValueError) as error:
            audit.sessions_skipped += 1
            audit.warn(f"{source_format}_session_conversion_failed")
            if strict:
                raise TraceConversionError(f"Session {session_id!r}: {error}") from error
            continue
        if session is not None:
            yield session


def convert_tracelab_session(
    rows: Sequence[Mapping[str, Any]],
    *,
    tool_wait_mode: str = "union",
    reuse_mode: str = "eligible",
    target_tokenizer: str | None = None,
    session_id_override: str | None = None,
    fallback_index: int = 0,
    audit: ConversionAudit | None = None,
) -> dict[str, Any]:
    """Convert normalized TraceLab rounds for one session."""

    if tool_wait_mode not in {"union", "max"}:
        raise TraceConversionError(f"invalid tool wait mode: {tool_wait_mode!r}")
    if reuse_mode not in {"eligible", "observed"}:
        raise TraceConversionError(f"invalid TraceLab reuse mode: {reuse_mode!r}")
    audit = audit or ConversionAudit()
    ordered = sorted(enumerate(rows), key=lambda pair: _round_sort_key(pair[1], pair[0]))
    if not ordered:
        raise TraceConversionError("empty TraceLab session")

    first = ordered[0][1]
    source_session_id = str(
        first.get("session_id") or f"tracelab-{fallback_index}")
    session_id = session_id_override or source_session_id
    models = _unique_strings(row.get("model") for _, row in ordered)
    providers = _unique_strings(row.get("provider") for _, row in ordered)
    sub_requests: list[dict[str, Any]] = []

    previous_input_toks: int | None = None
    previous_output_toks: int | None = None
    previous_round_index: int | None = None
    for position, (_, row) in enumerate(ordered):
        prefix_toks = _nonnegative_int(row.get("prefix_tokens"), "prefix_tokens")
        raw_append_toks = _nonnegative_int(
            row.get("newly_append_tokens"), "newly_append_tokens"
        )
        append_toks = raw_append_toks
        if append_toks == 0:
            # Match TraceLab's official CSV export: a zero-sized append/output
            # cannot be replayed as an LLM request, so retain it as one token
            # and make the normalization visible in the manifest.
            append_toks = 1
            audit.warn("tracelab_zero_append_promoted_to_one")
        reported_total = _optional_nonnegative_int(row.get("input_tokens_total"))
        derived_total = prefix_toks + append_toks
        if reported_total is None or reported_total <= 0:
            input_toks = derived_total
        else:
            input_toks = reported_total
            if reported_total != derived_total:
                audit.warn("tracelab_input_accounting_mismatch")
        if prefix_toks > input_toks:
            audit.warn("tracelab_prefix_clamped_to_input")
            prefix_toks = input_toks
        output_toks = _nonnegative_int(row.get("output_tokens"), "output_tokens")
        if output_toks == 0:
            output_toks = 1
            audit.warn("tracelab_zero_output_promoted_to_one")
        current_round_index = _optional_nonnegative_int(row.get("round_index"))
        eligible_reuse, lineage_status = _tracelab_policy_independent_reuse(
            row=row,
            input_toks=input_toks,
            append_toks=append_toks,
            previous_input_toks=previous_input_toks,
            previous_output_toks=previous_output_toks,
            previous_round_index=previous_round_index,
            current_round_index=current_round_index,
            audit=audit,
        )
        if reuse_mode == "observed" and position > 0:
            replay_reuse = prefix_toks
            prefix_source = "reported"
        else:
            replay_reuse = eligible_reuse
            prefix_source = "estimated"
        is_last = position == len(ordered) - 1
        if is_last:
            tool_wait_ns = 0
            tool_wait_source = "session_end"
            gap_type = "none"
        else:
            next_row = ordered[position + 1][1]
            tool_wait_ns, tool_wait_source, gap_type = _tracelab_transition_wait_ns(
                row, next_row, tool_wait_mode, audit)

        sub_request = {
            "input_toks": input_toks,
            "output_toks": output_toks,
            "tool_duration_ns": max(0, tool_wait_ns),
            "prefix_reuse_toks": replay_reuse,
            "prefix_reuse_source": prefix_source,
            "observed_provider_hit_toks": prefix_toks,
            "policy_independent_reuse_toks": eligible_reuse,
            "reuse_estimator": "tracelab_eviction_tradeoff_v0.0.1",
            "lineage_status": lineage_status,
            "input_token_count_source": "reported_source_tokenizer",
            "output_token_count_source": "reported_source_tokenizer",
            "reported_input_toks": input_toks,
            "reported_output_toks": output_toks,
            "raw_newly_append_toks": raw_append_toks,
            "newly_append_toks": append_toks,
            "source_round_index": _round_sort_key(row, position)[0],
            "source_ingest_seq": _round_sort_key(row, position)[1],
            "tool_wait_source": tool_wait_source,
            "inter_turn_gap_type": gap_type,
        }
        sub_requests.append(sub_request)
        previous_input_toks = input_toks
        previous_output_toks = output_toks
        previous_round_index = current_round_index

    if target_tokenizer:
        audit.warn("tracelab_target_tokenizer_unavailable_sanitized_content")
    return _session_row(
        session_id,
        sub_requests,
        source_format="tracelab",
        source_models=models,
        target_tokenizer=target_tokenizer,
        extra_metadata={
            "providers": providers,
            "tool_wait_mode": tool_wait_mode,
            "tracelab_reuse_mode": reuse_mode,
            "source_session_id": source_session_id,
            "source_session_identity_sha256": _source_session_identity_sha256(
                first, "tracelab"),
        },
    )


def convert_lmcache_session(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any = None,
    target_tokenizer: str | None = None,
    session_id_override: str | None = None,
    fallback_index: int = 0,
    audit: ConversionAudit | None = None,
) -> dict[str, Any]:
    """Convert LMCache cumulative-message rows for one session."""

    audit = audit or ConversionAudit()
    if not rows:
        raise TraceConversionError("empty LMCache session")
    source_session_id = str(
        rows[0].get("session_id") or f"lmcache-{fallback_index}")
    session_id = session_id_override or source_session_id
    models = _unique_strings(row.get("model") for row in rows)
    prepared: list[dict[str, Any]] = []
    previous_ids: list[int] | None = None
    previous_serialized = ""
    previous_messages: list[Mapping[str, Any]] | None = None
    previous_input_toks = 0
    previous_output_toks = 0

    for row_index, row in enumerate(rows):
        messages = _message_list(row.get("input") or row.get("messages"))
        if not messages:
            audit.warn("lmcache_missing_messages")
            raise TraceConversionError(
                f"LMCache row {row_index} has no cumulative message list"
            )
        serialized = _serialize_messages(messages)
        input_ids: list[int] | None = None
        if tokenizer is not None:
            input_ids = _tokenize_messages(tokenizer, messages)
            if not input_ids:
                audit.warn("lmcache_empty_target_tokenization")
                input_ids = None

        reported_input = _first_optional_nonnegative_int(
            row,
            ("input_tokens", "input_length", "prompt_tokens", "input_toks"),
        )
        if input_ids is not None:
            input_toks = len(input_ids)
            input_source = "target_tokenizer"
            input_only_reuse = _longest_common_prefix(previous_ids or [], input_ids)
            prefix_reuse = input_only_reuse
            prefix_source = "exact"
            lineage_scope = "session_start"
            completed_messages, lineage_scope = _lmcache_completed_messages(
                previous_messages, messages)
            if lineage_scope == "input_only_message_prefix_diverged":
                audit.warn("lmcache_message_prefix_diverged")
            if completed_messages is not None and previous_ids is not None:
                completed_ids = _tokenize_messages(
                    tokenizer,
                    completed_messages,
                    add_generation_prompt=False,
                )
                completed_reuse = _longest_common_prefix(completed_ids, input_ids)
                if completed_reuse < input_only_reuse:
                    audit.warn("lmcache_completed_context_prefix_mismatch")
                    lineage_scope = "input_only_completed_context_mismatch"
                else:
                    prefix_reuse = min(
                        completed_reuse,
                        len(input_ids),
                        len(previous_ids) + previous_output_toks,
                    )
        else:
            input_toks = reported_input or _estimate_tokens_from_text(serialized)
            input_source = (
                "reported_source_tokenizer" if reported_input else "estimated_chars"
            )
            common_chars = _longest_common_prefix(previous_serialized, serialized)
            prefix_reuse = min(
                input_toks,
                int(round(input_toks * common_chars / max(len(serialized), 1))),
            )
            prefix_source = "estimated"
            lineage_scope = "session_start"
            completed_messages, lineage_scope = _lmcache_completed_messages(
                previous_messages, messages)
            if lineage_scope == "input_only_message_prefix_diverged":
                audit.warn("lmcache_message_prefix_diverged")
            if completed_messages is not None and previous_messages is not None:
                completed_serialized = _serialize_messages(completed_messages)
                completed_chars = _longest_common_prefix(
                    completed_serialized, serialized)
                completed_reuse = int(round(
                    input_toks * completed_chars / max(len(serialized), 1)))
                prefix_reuse = min(
                    input_toks,
                    max(prefix_reuse, completed_reuse),
                    previous_input_toks + previous_output_toks,
                )

        output_toks = _positive_int(row.get("output_length"), "output_length")
        next_gap_seconds = 0.0
        if row_index + 1 < len(rows):
            next_gap_seconds = _nonnegative_float(
                rows[row_index + 1].get("pre_gap", 0.0), "pre_gap"
            )
        sub_request = {
            "input_toks": max(1, input_toks),
            "output_toks": output_toks,
            "tool_duration_ns": int(round(next_gap_seconds * _NS_PER_SECOND)),
            "prefix_reuse_toks": prefix_reuse,
            "prefix_reuse_source": prefix_source,
            "prefix_lineage_scope": lineage_scope,
            "lineage_status": lineage_scope,
            "input_token_count_source": input_source,
            "output_token_count_source": "reported_source_tokenizer",
            "reported_output_toks": output_toks,
            "source_round_index": row_index,
            "tool_wait_source": "lmcache_next_row_pre_gap",
        }
        if reported_input is not None:
            sub_request["reported_input_toks"] = reported_input
        if input_ids is not None:
            sub_request["input_tok_ids"] = input_ids
        prepared.append(sub_request)
        previous_ids = input_ids
        previous_serialized = serialized
        previous_messages = list(messages)
        previous_input_toks = input_toks
        previous_output_toks = output_toks

    if not prepared:
        raise TraceConversionError("LMCache session has no valid message rows")
    prepared[-1]["tool_duration_ns"] = 0
    return _session_row(
        session_id,
        prepared,
        source_format="lmcache",
        source_models=models,
        target_tokenizer=target_tokenizer,
        extra_metadata={
            "source_output_tokenizer": "source model",
            "mixed_tokenizers": bool(target_tokenizer),
            "source_session_id": source_session_id,
        },
    )


def convert_exgentic_session(
    row: Mapping[str, Any],
    *,
    tokenizer: Any = None,
    target_tokenizer: str | None = None,
    fallback_index: int = 0,
    audit: ConversionAudit | None = None,
) -> dict[str, Any]:
    """Convert one Exgentic OpenTelemetry trace row."""

    audit = audit or ConversionAudit()
    spans_value = _json_value(row.get("spans"))
    if not isinstance(spans_value, list):
        raise TraceConversionError("Exgentic row has no span list")

    llm_spans: list[dict[str, Any]] = []
    seen_calls: set[tuple[Any, ...]] = set()
    for source_index, raw_span in enumerate(spans_value):
        if not isinstance(raw_span, Mapping):
            continue
        attrs = _attribute_mapping(raw_span.get("attributes"))
        input_toks = _first_optional_nonnegative_int(
            attrs,
            (
                "gen_ai.usage.input_tokens",
                "gen_ai.usage.prompt_tokens",
                "llm.token_count.prompt",
                "input_tokens",
                "prompt_tokens",
            ),
        )
        output_toks = _first_optional_nonnegative_int(
            attrs,
            (
                "gen_ai.usage.output_tokens",
                "gen_ai.usage.completion_tokens",
                "llm.token_count.completion",
                "output_tokens",
                "completion_tokens",
            ),
        )
        operation = str(attrs.get("gen_ai.operation.name") or "").lower()
        span_name = str(raw_span.get("name") or "").lower()
        if operation not in {"chat", "completion", "generate"} and not (
            input_toks is not None and (span_name.startswith("chat") or "llm" in span_name)
        ):
            continue
        if not input_toks or not output_toks:
            audit.warn("exgentic_llm_span_missing_positive_usage")
            continue
        start_ns = _timestamp_ns(raw_span.get("start_time"))
        end_ns = _timestamp_ns(raw_span.get("end_time"))
        if start_ns is None or end_ns is None:
            audit.warn("exgentic_llm_span_missing_timestamp")
            continue
        if end_ns < start_ns:
            audit.warn("exgentic_llm_span_negative_duration")
            continue
        response_id = attrs.get("gen_ai.response.id")
        dedup_key = (
            response_id or raw_span.get("span_id"),
            start_ns,
            end_ns,
            input_toks,
            output_toks,
        )
        if dedup_key in seen_calls:
            audit.warn("exgentic_duplicate_llm_span")
            continue
        seen_calls.add(dedup_key)
        messages = _message_list(
            attrs.get("gen_ai.input.messages")
            or attrs.get("llm.prompts")
            or raw_span.get("messages")
        )
        llm_spans.append(
            {
                "source_index": source_index,
                "span_id": raw_span.get("span_id"),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "reported_input_toks": input_toks,
                "reported_output_toks": output_toks,
                "messages": messages,
                "model": attrs.get("gen_ai.response.model")
                or attrs.get("gen_ai.request.model"),
            }
        )

    llm_spans.sort(key=lambda span: (span["start_ns"], span["source_index"]))
    if not llm_spans:
        raise TraceConversionError("no valid LLM spans with usage and timestamps")

    previous_ids: list[int] | None = None
    previous_serialized = ""
    previous_input_toks = 0
    previous_output_toks = 0
    sub_requests: list[dict[str, Any]] = []
    span_models = _unique_strings(span.get("model") for span in llm_spans)

    for index, span in enumerate(llm_spans):
        messages = span["messages"]
        serialized = _serialize_messages(messages) if messages else ""
        input_ids: list[int] | None = None
        if tokenizer is not None and messages:
            input_ids = _tokenize_messages(tokenizer, messages)
            if not input_ids:
                input_ids = None
                audit.warn("exgentic_empty_target_tokenization")

        if input_ids is not None:
            input_toks = len(input_ids)
            input_source = "target_tokenizer"
            prefix_reuse = _longest_common_prefix(previous_ids or [], input_ids)
            prefix_source = "exact"
        else:
            input_toks = span["reported_input_toks"]
            input_source = "reported_source_tokenizer"
            if serialized and previous_serialized:
                common_chars = _longest_common_prefix(previous_serialized, serialized)
                prefix_reuse = min(
                    input_toks,
                    int(round(input_toks * common_chars / max(len(serialized), 1))),
                )
            else:
                prefix_reuse = min(
                    input_toks, previous_input_toks + previous_output_toks
                )
            prefix_source = "estimated"

        wait_ns = 0
        if index + 1 < len(llm_spans):
            wait_ns = llm_spans[index + 1]["start_ns"] - span["end_ns"]
            if wait_ns < 0:
                audit.warn("exgentic_overlapping_llm_spans_wait_clamped")
                wait_ns = 0
        sub_request = {
            "input_toks": input_toks,
            "output_toks": span["reported_output_toks"],
            "tool_duration_ns": int(wait_ns),
            "prefix_reuse_toks": prefix_reuse,
            "prefix_reuse_source": prefix_source,
            "input_token_count_source": input_source,
            "output_token_count_source": "reported_source_tokenizer",
            "reported_input_toks": span["reported_input_toks"],
            "reported_output_toks": span["reported_output_toks"],
            "source_round_index": index,
            "source_span_id": span["span_id"],
            "tool_wait_source": "otel_next_llm_start_minus_current_llm_end",
        }
        if input_ids is not None:
            sub_request["input_tok_ids"] = input_ids
        sub_requests.append(sub_request)
        previous_ids = input_ids
        previous_serialized = serialized
        previous_input_toks = input_toks
        previous_output_toks = span["reported_output_toks"]

    session_id = str(
        row.get("session_id")
        or row.get("trace_id")
        or f"exgentic-{fallback_index}"
    )
    row_models = row.get("models")
    if isinstance(row_models, str):
        row_models = _json_value(row_models)
    models = _unique_strings(
        (list(row_models) if isinstance(row_models, list) else []) + span_models
    )
    return _session_row(
        session_id,
        sub_requests,
        source_format="exgentic",
        source_models=models,
        target_tokenizer=target_tokenizer,
        extra_metadata={
            "benchmark": row.get("benchmark"),
            "harness": row.get("harness"),
            "source_output_tokenizer": "source model",
            "mixed_tokenizers": bool(target_tokenizer),
        },
    )


def validate_session(session: Mapping[str, Any]) -> list[str]:
    """Return schema/semantic errors without mutating the session."""

    errors: list[str] = []
    session_id = session.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        errors.append("session_id must be a non-empty string")
    arrival = session.get("arrival_time_ns")
    if not _is_integer(arrival) or int(arrival) < 0:
        errors.append("arrival_time_ns must be a non-negative integer")
    sub_requests = session.get("sub_requests")
    if not isinstance(sub_requests, list) or not sub_requests:
        errors.append("sub_requests must be a non-empty list")
        return errors

    for index, sub_request in enumerate(sub_requests):
        if not isinstance(sub_request, Mapping):
            errors.append(f"sub_requests[{index}] is not an object")
            continue
        for field_name in ("input_toks", "output_toks"):
            value = sub_request.get(field_name)
            if not _is_integer(value) or int(value) <= 0:
                errors.append(f"sub_requests[{index}].{field_name} must be positive")
        wait = sub_request.get("tool_duration_ns")
        if not _is_integer(wait) or int(wait) < 0:
            errors.append(
                f"sub_requests[{index}].tool_duration_ns must be non-negative"
            )
        prefix = sub_request.get("prefix_reuse_toks")
        if not _is_integer(prefix) or int(prefix) < 0:
            errors.append(
                f"sub_requests[{index}].prefix_reuse_toks must be non-negative"
            )
        elif _is_integer(sub_request.get("input_toks")) and int(prefix) > int(
            sub_request["input_toks"]
        ):
            errors.append(
                f"sub_requests[{index}].prefix_reuse_toks exceeds input_toks"
            )
        if sub_request.get("prefix_reuse_source") not in _PREFIX_SOURCES:
            errors.append(
                f"sub_requests[{index}].prefix_reuse_source must be one of "
                f"{sorted(_PREFIX_SOURCES)}"
            )
        input_ids = sub_request.get("input_tok_ids")
        if input_ids is not None:
            if not isinstance(input_ids, list):
                errors.append(f"sub_requests[{index}].input_tok_ids must be a list")
            elif _is_integer(sub_request.get("input_toks")) and len(input_ids) != int(
                sub_request["input_toks"]
            ):
                errors.append(
                    f"sub_requests[{index}].input_tok_ids length does not match input_toks"
                )
    if sub_requests and sub_requests[-1].get("tool_duration_ns") != 0:
        errors.append("the final sub-request must have tool_duration_ns=0")
    return errors


def build_manifest(
    *,
    args: argparse.Namespace,
    audit: ConversionAudit,
    output_path: Path,
    output_sha256: str,
) -> dict[str, Any]:
    warning_total = sum(audit.warning_counts.values())
    source_path = Path(args.source).expanduser()
    source_is_local_file = source_path.is_file()
    module_path = Path(__file__).resolve()
    repository_root = module_path.parents[2]
    git_commit, git_dirty = _git_provenance(repository_root)
    if audit.validation_errors:
        validation_status = "failed"
    elif warning_total:
        validation_status = "passed_with_warnings"
    else:
        validation_status = "passed"
    return {
        "schema_version": 3,
        "generator": "workloads.generators.agent_traces",
        "converter": {
            "module": "workloads.generators.agent_traces",
            "module_sha256": _file_sha256(module_path),
            "git_commit": git_commit,
            "git_dirty_tracked_files": git_dirty,
            "arguments": {
                "source_format": args.source_format,
                "source_revision": args.source_revision,
                "split": args.split,
                "hf_config": args.hf_config,
                "sessions_per_second": args.sps,
                "seed": args.seed,
                "first_arrival_sec": args.first_arrival_sec,
                "max_sessions": args.max_sessions,
                "max_source_rows": args.max_source_rows,
                "target_tokenizer": args.tokenizer,
                "target_tokenizer_revision": args.tokenizer_revision,
                "trust_remote_code": args.trust_remote_code,
                "tool_wait_mode": args.tool_wait_mode,
                "tracelab_reuse_mode": args.tracelab_reuse_mode,
                "strict": args.strict,
            },
        },
        "source": {
            "location": str(args.source),
            "sha256": (
                _file_sha256(source_path) if source_is_local_file else None
            ),
            "size_bytes": (
                source_path.stat().st_size if source_is_local_file else None
            ),
            "revision": args.source_revision,
            "format": args.source_format,
            "split": args.split,
            "hf_config": args.hf_config,
            "rows_read": audit.source_rows,
            "models": sorted(audit.source_models),
            "tracelab_reuse_mode": (
                args.tracelab_reuse_mode
                if args.source_format == "tracelab" else None
            ),
        },
        "tokenizers": {
            "source": "reported per source model when available",
            "target": args.tokenizer,
            "target_revision": args.tokenizer_revision,
            "mixed_source_and_target_counts": bool(args.tokenizer),
            "note": (
                "TraceLab has no sanitized prompt text. LMCache has no standalone "
                "output field, but a normal continuation carries the previous "
                "assistant message in its next cumulative prompt. Source-reported "
                "counts are retained where target retokenization is impossible."
            ),
        },
        "arrivals": {
            "distribution": "poisson",
            "sessions_per_second": args.sps,
            "seed": args.seed,
            "first_arrival_sec": args.first_arrival_sec,
        },
        "summary": {
            "sessions_seen": audit.sessions_seen,
            "sessions_emitted": audit.sessions_emitted,
            "sessions_skipped": audit.sessions_skipped,
            "sub_requests_emitted": audit.sub_requests_emitted,
            "turns_per_session": _number_summary(audit.turns_per_session),
            "input_tokens": _number_summary(audit.input_tokens),
            "output_tokens": _number_summary(audit.output_tokens),
            "tool_wait_ns": _number_summary(audit.tool_wait_ns),
            "total_tool_wait_ns": sum(audit.tool_wait_ns),
            "prefix_reuse_source_counts": dict(
                sorted(audit.prefix_source_counts.items())
            ),
            "token_count_source_counts": dict(
                sorted(audit.token_count_source_counts.items())
            ),
            "tool_wait_source_counts": dict(
                sorted(audit.tool_wait_source_counts.items())
            ),
            "inter_turn_gap_type_counts": dict(
                sorted(audit.inter_turn_gap_type_counts.items())
            ),
            "lineage_status_counts": dict(
                sorted(audit.lineage_status_counts.items())
            ),
            "observed_provider_hit_tokens": _number_summary(
                audit.observed_provider_hit_tokens),
            "policy_independent_reuse_tokens": _number_summary(
                audit.policy_independent_reuse_tokens),
        },
        "validation": {
            "status": validation_status,
            "error_count": audit.validation_errors,
            "warning_count": warning_total,
            "warning_counts": dict(sorted(audit.warning_counts.items())),
        },
        "output": {
            "path": str(output_path),
            "sha256": output_sha256,
        },
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(repository_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit or None, dirty


def _load_tokenizer(
        name_or_path: str, trust_remote_code: bool,
        *, revision: str | None = None) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "transformers is required with --tokenizer. Run inside the vLLM "
            "container or install transformers."
        ) from error
    return AutoTokenizer.from_pretrained(
        name_or_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )


def _load_source_rows(
    source: str,
    *,
    split: str,
    hf_config: str | None,
    revision: str | None,
    max_rows: int,
) -> Iterator[Mapping[str, Any]]:
    path = Path(source)
    if path.exists():
        yield from _load_local_rows(path, max_rows=max_rows)
        return

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "huggingface datasets is required for a remote --source. Run "
            "inside the vLLM container or install datasets."
        ) from error
    dataset_args: list[str] = [source]
    if hf_config:
        dataset_args.append(hf_config)
    dataset = load_dataset(
        *dataset_args,
        split=split,
        streaming=True,
        revision=revision,
    )
    for index, row in enumerate(dataset):
        if max_rows and index >= max_rows:
            break
        yield row


def _load_local_rows(path: Path, *, max_rows: int = 0) -> Iterator[Mapping[str, Any]]:
    suffixes = path.suffixes
    compressed = bool(suffixes and suffixes[-1] == ".gz")
    logical_suffix = suffixes[-2] if compressed and len(suffixes) >= 2 else path.suffix
    opener = gzip.open if compressed else open
    if logical_suffix == ".jsonl":
        with opener(path, "rt", encoding="utf-8") as input_file:
            emitted = 0
            for line_number, line in enumerate(input_file, start=1):
                if max_rows and emitted >= max_rows:
                    break
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise TraceConversionError(
                        f"{path}:{line_number}: JSONL row is not an object"
                    )
                yield value
                emitted += 1
        return
    if logical_suffix != ".json":
        raise TraceConversionError(
            f"Local source must end in .json[.gz] or .jsonl[.gz]: {path}"
        )
    with opener(path, "rt", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if isinstance(data, Mapping):
        for key in ("rows", "data", "traces"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise TraceConversionError(f"{path}: JSON root is not an object or array")
    limit = len(data) if not max_rows else min(len(data), max_rows)
    for index in range(limit):
        row = data[index]
        if not isinstance(row, Mapping):
            raise TraceConversionError(f"{path}: row {index} is not an object")
        yield row


def _group_contiguous_sessions(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_format: str,
    audit: ConversionAudit,
    strict: bool,
) -> Iterator[tuple[str, list[Mapping[str, Any]]]]:
    """Disk-spool rows so non-contiguous official sessions remain correct.

    TraceLab's full v0.0.1 release contains sessions that reappear later in
    the file.  Yielding a session on the first ID change silently splits its
    dependency chain.  SQLite keeps the conversion bounded by the largest
    session rather than the multi-GB decompressed corpus, while preserving
    first-seen session order and source order within each session.
    """

    with tempfile.TemporaryDirectory(prefix="llmservingsim-agent-trace-") as tmp:
        database = Path(tmp) / "sessions.sqlite3"
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE rows ("
                "group_key TEXT NOT NULL, raw_session_id TEXT NOT NULL, "
                "source_order INTEGER NOT NULL, payload TEXT NOT NULL)"
            )
            for row_index, row in enumerate(rows):
                audit.source_rows += 1
                raw_id = row.get("session_id")
                if raw_id is None or str(raw_id) == "":
                    audit.warn("source_row_missing_session_id")
                    if strict:
                        raise TraceConversionError(
                            f"source row {row_index} has no session_id")
                    continue
                group_key = _source_session_group_key(row, source_format)
                connection.execute(
                    "INSERT INTO rows VALUES (?, ?, ?, ?)",
                    (
                        group_key,
                        str(raw_id),
                        row_index,
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                    ),
                )
            connection.commit()
            connection.execute(
                "CREATE INDEX rows_session_order "
                "ON rows(group_key, source_order)"
            )
            collision_counts = dict(connection.execute(
                "SELECT raw_session_id, COUNT(DISTINCT group_key) "
                "FROM rows GROUP BY raw_session_id"
            ))
            session_groups = connection.execute(
                "SELECT group_key, raw_session_id FROM rows GROUP BY group_key, raw_session_id "
                "ORDER BY MIN(source_order)"
            )
            for group_key, raw_session_id in session_groups:
                session_rows = []
                for source_order, payload in connection.execute(
                        "SELECT source_order, payload FROM rows WHERE group_key = ? "
                        "ORDER BY source_order",
                        (group_key,)):
                    decoded = json.loads(payload)
                    decoded["_source_ingest_seq"] = source_order
                    session_rows.append(decoded)
                output_session_id = raw_session_id
                if collision_counts[raw_session_id] > 1:
                    digest = hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:12]
                    output_session_id = f"{raw_session_id}@{digest}"
                    audit.warn("source_session_id_collision_disambiguated")
                yield output_session_id, session_rows
        finally:
            connection.close()


def _session_row(
    session_id: str,
    sub_requests: list[dict[str, Any]],
    *,
    source_format: str,
    source_models: Sequence[str],
    target_tokenizer: str | None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "source_format": source_format,
        "source_models": list(source_models),
        "source_tokenizer": "source model/provider tokenizer",
        "target_tokenizer": target_tokenizer,
    }
    if extra_metadata:
        metadata.update(
            {key: value for key, value in extra_metadata.items() if value is not None}
        )
    return {
        "session_id": session_id,
        "arrival_time_ns": 0,
        "trace_metadata": metadata,
        "sub_requests": sub_requests,
    }


def _tracelab_tool_wait_ms(
    row: Mapping[str, Any], mode: str, audit: ConversionAudit
) -> float:
    tools = _json_value(row.get("tools"))
    if not isinstance(tools, list) or not tools:
        return 0.0
    intervals: list[tuple[int, int]] = []
    fallback_latencies: list[float] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            audit.warn("tracelab_invalid_tool_record")
            continue
        start_ns = _timestamp_ns(tool.get("emitted_at"))
        end_ns = _timestamp_ns(tool.get("result_at"))
        if start_ns is not None and end_ns is not None and end_ns >= start_ns:
            intervals.append((start_ns, end_ns))
            continue
        latency_ms = _optional_nonnegative_float(tool.get("tool_wall_latency_ms"))
        if latency_ms is not None:
            fallback_latencies.append(latency_ms)
            audit.warn("tracelab_tool_interval_missing_used_wall_latency")
        else:
            audit.warn("tracelab_tool_latency_missing")
    interval_durations_ms = [
        (end_ns - start_ns) / _NS_PER_MILLISECOND for start_ns, end_ns in intervals
    ]
    if mode == "max":
        return max(interval_durations_ms + fallback_latencies, default=0.0)
    union_ms = _interval_union_ns(intervals) / _NS_PER_MILLISECOND
    if fallback_latencies:
        # With no timestamps, overlap is unknowable. Treat missing intervals as
        # parallel and take the larger value instead of over-counting by sum.
        return max(union_ms, max(fallback_latencies))
    return union_ms


def _tracelab_transition_wait_ns(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    fallback_mode: str,
    audit: ConversionAudit,
) -> tuple[int, str, str]:
    """Measure completion-to-next-input delay for one TraceLab transition."""

    previous_completion_ns = _tracelab_last_model_output_ns(previous)
    current_ready_ns, gap_type = _tracelab_request_ready_ns(current)
    if previous_completion_ns is not None and current_ready_ns is not None:
        gap_ns = current_ready_ns - previous_completion_ns
        if gap_ns >= 0:
            return gap_ns, "tracelab_event_boundary", gap_type
        audit.warn("tracelab_negative_event_boundary_gap_zeroed")
        return 0, "tracelab_negative_event_boundary_zero", gap_type

    fallback_ms = _tracelab_tool_wait_ms(previous, fallback_mode, audit)
    if fallback_ms > 0:
        audit.warn("tracelab_event_boundary_missing_used_tool_fallback")
        return (
            int(round(fallback_ms * _NS_PER_MILLISECOND)),
            f"tracelab_tool_{fallback_mode}_fallback",
            "tool",
        )
    audit.warn("tracelab_transition_gap_unmeasurable_zeroed")
    return 0, "tracelab_unmeasurable_zero", "unknown"


def _tracelab_policy_independent_reuse(
    *,
    row: Mapping[str, Any],
    input_toks: int,
    append_toks: int,
    previous_input_toks: int | None,
    previous_output_toks: int | None,
    previous_round_index: int | None,
    current_round_index: int | None,
    audit: ConversionAudit,
) -> tuple[int, str]:
    """Return TraceLab's ideal-cache reusable-token upper bound.

    Provider ``prefix_tokens`` is an observed outcome of the incumbent cache.
    The eviction counterfactual instead subtracts irreducibly fresh external
    input from the current context so a different retention policy can recover
    a prefix that the incumbent provider evicted.
    """

    if previous_input_toks is None or previous_output_toks is None:
        return 0, "session_start"
    if (previous_round_index is None or current_round_index is None
            or current_round_index != previous_round_index + 1):
        audit.warn("tracelab_nonadjacent_round_lineage_reset")
        return 0, "round_gap"
    if input_toks < previous_input_toks:
        audit.warn("tracelab_context_shrink_lineage_reset")
        return 0, "context_shrink"
    lineage_break = _tracelab_lineage_break_marker(row)
    if lineage_break is not None:
        audit.warn(f"tracelab_{lineage_break}_lineage_reset")
        return 0, lineage_break

    context_growth = max(0, input_toks - previous_input_toks)
    fresh = max(0, context_growth - previous_output_toks)
    fresh = min(append_toks, fresh)
    reusable = max(0, min(input_toks, input_toks - fresh))
    return reusable, "adjacent_estimate"


def _tracelab_lineage_break_marker(row: Mapping[str, Any]) -> str | None:
    for event in _tracelab_timing_events(row):
        markers = (
            str(event.get("event_type") or "").lower(),
            str(event.get("source") or "").lower(),
        )
        if any("compact" in marker for marker in markers):
            return "explicit_compaction"
        normalized = tuple(
            marker.replace(".", "_").replace("-", "_").replace(" ", "_")
            for marker in markers
        )
        reset_aliases = (
            "context_reset",
            "reset_context",
            "context_clear",
            "clear_context",
        )
        if any(alias in marker for marker in normalized for alias in reset_aliases):
            return "explicit_context_reset"
    return None


def _tracelab_timing_events(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = _json_value(row.get("timing_events"))
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, Mapping)]


def _tracelab_last_model_output_ns(row: Mapping[str, Any]) -> int | None:
    timestamps = [
        timestamp
        for event in _tracelab_timing_events(row)
        if event.get("event_type") in _TRACELAB_MODEL_OUTPUT_EVENT_TYPES
        if (timestamp := _timestamp_ns(event.get("timestamp"))) is not None
    ]
    return max(timestamps) if timestamps else None


def _tracelab_request_ready_ns(
    row: Mapping[str, Any],
) -> tuple[int | None, str]:
    events = _tracelab_timing_events(row)
    output_timestamps = [
        timestamp
        for event in events
        if event.get("event_type") in _TRACELAB_MODEL_OUTPUT_EVENT_TYPES
        if (timestamp := _timestamp_ns(event.get("timestamp"))) is not None
    ]
    first_output_ns = min(output_timestamps) if output_timestamps else None
    candidates: list[tuple[int, str]] = []
    for event in events:
        event_type = event.get("event_type")
        if event_type not in _TRACELAB_INPUT_EVENT_TYPES:
            continue
        timestamp = _timestamp_ns(event.get("timestamp"))
        if timestamp is None:
            continue
        if first_output_ns is None or timestamp <= first_output_ns:
            candidates.append((timestamp, str(event_type)))
    if not candidates:
        return None, "unknown"
    ready_ns = max(timestamp for timestamp, _ in candidates)
    input_types = {event_type for _, event_type in candidates}
    if input_types == {"user_message"}:
        gap_type = "human"
    elif input_types == {"tool_result"}:
        gap_type = "tool"
    else:
        gap_type = "mixed"
    return ready_ns, gap_type


def _interval_union_ns(intervals: Sequence[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _round_sort_key(row: Mapping[str, Any], fallback: int) -> tuple[int, int]:
    value = _optional_nonnegative_int(row.get("round_index"))
    ingest_seq = _optional_nonnegative_int(row.get("_source_ingest_seq"))
    return (
        value if value is not None else fallback,
        ingest_seq if ingest_seq is not None else fallback,
    )


def _source_session_group_key(
    row: Mapping[str, Any], source_format: str
) -> str:
    raw_session_id = str(row.get("session_id") or "")
    if source_format == "tracelab":
        identity = [
            str(row.get("provider") or ""),
            str(row.get("project") or ""),
            str(row.get("session_file") or ""),
            raw_session_id,
        ]
    else:
        identity = [raw_session_id]
    return json.dumps(identity, ensure_ascii=False, separators=(",", ":"))


def _source_session_identity_sha256(
    row: Mapping[str, Any], source_format: str
) -> str:
    return hashlib.sha256(
        _source_session_group_key(row, source_format).encode("utf-8")
    ).hexdigest()


def _attribute_mapping(value: Any) -> dict[str, Any]:
    value = _json_value(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list):
        result: dict[str, Any] = {}
        for item in value:
            if isinstance(item, Mapping) and "key" in item:
                result[str(item["key"])] = item.get("value")
        return result
    return {}


def _message_list(value: Any) -> list[Mapping[str, Any]]:
    value = _json_value(value)
    if isinstance(value, Mapping):
        value = value.get("messages") or value.get("input")
    if not isinstance(value, list):
        return []
    return [message for message in value if isinstance(message, Mapping)]


def _lmcache_completed_messages(
    previous: Sequence[Mapping[str, Any]] | None,
    current: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]] | None, str]:
    """Recover the prior completed context from a cumulative LMCache prompt."""

    if previous is None:
        return None, "session_start"
    if len(current) <= len(previous) or list(current[:len(previous)]) != list(previous):
        return None, "input_only_message_prefix_diverged"
    first_appended = current[len(previous)]
    if str(first_appended.get("role") or "").lower() != "assistant":
        return None, "input_only_no_replayed_output"
    return list(current[:len(previous) + 1]), "completed_context"


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _serialize_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    # One independently serialized object per line makes an appended message a
    # true string prefix, unlike serializing a JSON array with a closing ']'.
    return "".join(
        json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for message in messages
    )


def _tokenize_messages(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool = True,
) -> list[int]:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            result = apply_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
            ids = _token_ids(result)
            if ids:
                return ids
        except (KeyError, TypeError, ValueError):
            pass
    serialized = _serialize_messages(messages)
    result = tokenizer(serialized, add_special_tokens=False)
    ids = _token_ids(result)
    if not ids:
        raise TraceConversionError("target tokenizer produced no input ids")
    return ids


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        return []
    ids: list[int] = []
    for token_id in value:
        if not _is_integer(token_id):
            return []
        ids.append(int(token_id))
    return ids


def _timestamp_ns(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        magnitude = abs(number)
        if magnitude >= 1e17:
            return int(number)  # nanoseconds since epoch
        if magnitude >= 1e14:
            return int(number * 1_000)  # microseconds since epoch
        if magnitude >= 1e11:
            return int(number * 1_000_000)  # milliseconds since epoch
        return int(number * _NS_PER_SECOND)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = parsed - epoch
    return (
        (delta.days * 86_400 + delta.seconds) * _NS_PER_SECOND
        + delta.microseconds * 1_000
    )


def _longest_common_prefix(left: Sequence[Any], right: Sequence[Any]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _estimate_tokens_from_text(text: str) -> int:
    # Explicitly marked estimated in the output. Four UTF-8 characters/token
    # is a conservative, transparent fallback when no tokenizer is installed.
    return max(1, int(math.ceil(len(text) / 4.0)))


def _unique_strings(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if value not in (None, "")})


def _number_summary(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "max": None, "mean": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": _nearest_rank(ordered, 0.50),
        "p90": _nearest_rank(ordered, 0.90),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _nearest_rank(ordered: Sequence[int], quantile: float) -> int:
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def _first_optional_nonnegative_int(
    mapping: Mapping[str, Any], keys: Sequence[str]
) -> int | None:
    for key in keys:
        value = _optional_nonnegative_int(mapping.get(key))
        if value is not None:
            return value
    return None


def _positive_int(value: Any, field_name: str) -> int:
    parsed = _optional_nonnegative_int(value)
    if parsed is None or parsed <= 0:
        raise TraceConversionError(f"{field_name} must be positive, got {value!r}")
    return parsed


def _nonnegative_int(value: Any, field_name: str) -> int:
    parsed = _optional_nonnegative_int(value)
    if parsed is None:
        raise TraceConversionError(
            f"{field_name} must be a non-negative integer, got {value!r}"
        )
    return parsed


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def _nonnegative_float(value: Any, field_name: str) -> float:
    parsed = _optional_nonnegative_float(value)
    if parsed is None:
        raise TraceConversionError(
            f"{field_name} must be finite and non-negative, got {value!r}"
        )
    return parsed


def _optional_nonnegative_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)

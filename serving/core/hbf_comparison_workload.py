"""Fixed TraceLab workload contract for fair HBF system comparisons.

The comparison runner must offer exactly the same complete sessions to every
system.  This module deliberately keeps workload selection, Poisson draws, and
full-drain identity checks independent of any simulator policy.

TraceLab's operational cached prefix is ``prefix_reuse_toks``.  In particular,
``newly_append_toks`` is provenance from the source conversion and is not used
to derive service demand.  Fresh prompt work is always:

``input_toks - prefix_reuse_toks``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


TRACELAB_SCHEMA3_SHA256 = (
    "b6188582aac9467cee8c73e4275f9a9606b359f8c2fa000d9f49a9ca3bde02f0"
)

FIXED_SOURCE_INDICES = (
    25, 71, 165, 389, 442, 447, 479, 864,
    1395, 1472, 1490, 1531, 1554, 1853,
    2266, 2270, 2276, 2399, 2402, 2471,
    2850, 3047, 3131, 3277, 3437, 3722,
    3813, 3961, 4050, 4055, 4090, 4177,
)


class WorkloadValidationError(ValueError):
    """Raised when the fixed workload contract is not satisfied."""


@dataclass(frozen=True)
class CallSpec:
    """One immutable LLM call in an agentic session."""

    session_id: str
    source_index: int
    call_index: int
    input_tokens: int
    output_tokens: int
    tool_duration_ns: int
    cached_prefix_tokens: int
    fresh_input_tokens: int
    lineage_status: str | None
    inter_turn_gap_type: str | None

    @property
    def is_first_turn(self) -> bool:
        return self.call_index == 0

    @property
    def is_resume(self) -> bool:
        return self.call_index > 0

    @property
    def has_cached_prefix(self) -> bool:
        return self.cached_prefix_tokens > 0

    @property
    def is_context_shrink(self) -> bool:
        return self.is_resume and self.cached_prefix_tokens == 0

    @property
    def tpot_eligible(self) -> bool:
        """Whether at least one inter-token interval exists."""

        return self.output_tokens > 1

    @property
    def completion_identity(self) -> str:
        return f"{self.session_id}::call-{self.call_index}"


@dataclass(frozen=True)
class SessionSpec:
    """One immutable complete TraceLab session."""

    source_index: int
    session_id: str
    source_arrival_time_ns: int
    source_session_identity_sha256: str | None
    calls: tuple[CallSpec, ...]

    @property
    def output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.calls)

    @property
    def completion_identities(self) -> tuple[str, ...]:
        return tuple(call.completion_identity for call in self.calls)


@dataclass(frozen=True)
class CohortSummary:
    """Content and metric denominators pinned by the fixed cohort."""

    session_count: int
    call_count: int
    first_turn_count: int
    resume_count: int
    adjacent_cached_resume_count: int
    context_shrink_resume_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cached_prefix_tokens: int
    total_fresh_input_tokens: int
    resume_fresh_input_tokens: int
    single_output_call_count: int
    tpot_eligible_call_count: int
    max_input_context_tokens: int
    max_sequence_tokens: int
    selected_session_ids_sha256: str
    source_index_session_id_sha256: str


FIXED_COHORT_SUMMARY = CohortSummary(
    session_count=32,
    call_count=2680,
    first_turn_count=32,
    resume_count=2648,
    adjacent_cached_resume_count=2597,
    context_shrink_resume_count=51,
    total_input_tokens=409_094_011,
    total_output_tokens=1_396_785,
    total_cached_prefix_tokens=398_757_236,
    total_fresh_input_tokens=10_336_775,
    resume_fresh_input_tokens=9_582_453,
    single_output_call_count=29,
    tpot_eligible_call_count=2651,
    max_input_context_tokens=415_963,
    max_sequence_tokens=420_339,
    selected_session_ids_sha256=(
        "985d7fff295973f3a1a6d15f7c847455ddd54585f28a8656c904e9749f1b6eca"
    ),
    source_index_session_id_sha256=(
        "b1f47d17a50d2a68008a67bd1c14e797b0a1bd6105c913701eeda0068eab9573"
    ),
)


@dataclass(frozen=True)
class ComparisonWorkload:
    """A content-addressed, fixed-session workload."""

    source_path: Path
    source_sha256: str
    source_session_count: int
    sessions: tuple[SessionSpec, ...]
    summary: CohortSummary

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(session.session_id for session in self.sessions)

    @property
    def call_completion_identities(self) -> tuple[str, ...]:
        return tuple(
            identity
            for session in self.sessions
            for identity in session.completion_identities
        )


@dataclass(frozen=True)
class OfferedSession:
    """A session and its policy-independent unit-rate arrival coordinate."""

    offer_index: int
    session: SessionSpec
    unit_interarrival: float
    unit_arrival_time: float


@dataclass(frozen=True)
class ScheduledSession:
    """An offered session scaled to one rate without redrawing arrivals."""

    offer_index: int
    session: SessionSpec
    arrival_time_ns: int
    unit_interarrival: float
    unit_arrival_time: float


@dataclass(frozen=True)
class OfferedPlan:
    """One deterministic permutation and unit-rate exponential draw vector."""

    seed: int
    offers: tuple[OfferedSession, ...]
    offered_session_ids_sha256: str
    unit_draws_sha256: str

    def at_rate(
            self,
            sessions_per_second: float,
            *,
            start_time_ns: int = 0,
    ) -> tuple[ScheduledSession, ...]:
        """Scale the same unit-rate process to ``sessions_per_second``."""

        if (
            isinstance(sessions_per_second, bool)
            or not isinstance(sessions_per_second, (int, float))
            or not math.isfinite(float(sessions_per_second))
            or float(sessions_per_second) <= 0
        ):
            raise ValueError("sessions_per_second must be positive and finite")
        if (
            isinstance(start_time_ns, bool)
            or not isinstance(start_time_ns, int)
            or start_time_ns < 0
        ):
            raise ValueError("start_time_ns must be a non-negative integer")

        rate = float(sessions_per_second)
        return tuple(
            ScheduledSession(
                offer_index=offer.offer_index,
                session=offer.session,
                arrival_time_ns=start_time_ns + int(round(
                    offer.unit_arrival_time * 1_000_000_000 / rate
                )),
                unit_interarrival=offer.unit_interarrival,
                unit_arrival_time=offer.unit_arrival_time,
            )
            for offer in self.offers
        )


@dataclass(frozen=True)
class FullDrainHashes:
    """Identity hashes proving that a comparison completed the full cohort."""

    identity_count: int
    offered_order_sha256: str
    expected_set_sha256: str
    completion_order_sha256: str
    completion_set_sha256: str


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_int(
        raw: dict[str, object],
        field: str,
        *,
        context: str,
        minimum: int,
) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WorkloadValidationError(
            f"{context}.{field} must be an integer >= {minimum}"
        )
    return value


def _optional_string(
        raw: dict[str, object],
        field: str,
        *,
        context: str,
) -> str | None:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise WorkloadValidationError(f"{context}.{field} must be a string")
    return value


def _parse_session(raw: object, source_index: int) -> SessionSpec:
    context = f"source row {source_index}"
    if not isinstance(raw, dict):
        raise WorkloadValidationError(f"{context} must be a JSON object")

    session_id = raw.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise WorkloadValidationError(
            f"{context}.session_id must be a non-empty string"
        )
    source_arrival_time_ns = _require_int(
        raw, "arrival_time_ns", context=context, minimum=0
    )
    sub_requests = raw.get("sub_requests")
    if not isinstance(sub_requests, list) or not sub_requests:
        raise WorkloadValidationError(
            f"{context}.sub_requests must be a non-empty list"
        )

    metadata = raw.get("trace_metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise WorkloadValidationError(
            f"{context}.trace_metadata must be an object"
        )
    source_identity = metadata.get("source_session_identity_sha256")
    if source_identity is not None and not _is_sha256(source_identity):
        raise WorkloadValidationError(
            f"{context} has an invalid source_session_identity_sha256"
        )

    calls = []
    for call_index, call_raw in enumerate(sub_requests):
        call_context = f"{context}.sub_requests[{call_index}]"
        if not isinstance(call_raw, dict):
            raise WorkloadValidationError(
                f"{call_context} must be a JSON object"
            )
        input_tokens = _require_int(
            call_raw, "input_toks", context=call_context, minimum=1
        )
        output_tokens = _require_int(
            call_raw, "output_toks", context=call_context, minimum=1
        )
        tool_duration_ns = _require_int(
            call_raw, "tool_duration_ns", context=call_context, minimum=0
        )
        cached_prefix_tokens = _require_int(
            call_raw, "prefix_reuse_toks", context=call_context, minimum=0
        )
        if cached_prefix_tokens > input_tokens:
            raise WorkloadValidationError(
                f"{call_context}.prefix_reuse_toks exceeds input_toks"
            )
        if call_index == 0 and cached_prefix_tokens != 0:
            raise WorkloadValidationError(
                f"{call_context} first turn cannot reuse an earlier prefix"
            )

        # Do not use newly_append_toks here.  That field is source-converter
        # provenance and can differ from the operational prefix definition.
        fresh_input_tokens = input_tokens - cached_prefix_tokens
        calls.append(CallSpec(
            session_id=session_id,
            source_index=source_index,
            call_index=call_index,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_duration_ns=tool_duration_ns,
            cached_prefix_tokens=cached_prefix_tokens,
            fresh_input_tokens=fresh_input_tokens,
            lineage_status=_optional_string(
                call_raw, "lineage_status", context=call_context
            ),
            inter_turn_gap_type=_optional_string(
                call_raw, "inter_turn_gap_type", context=call_context
            ),
        ))

    return SessionSpec(
        source_index=source_index,
        session_id=session_id,
        source_arrival_time_ns=source_arrival_time_ns,
        source_session_identity_sha256=source_identity,
        calls=tuple(calls),
    )


def summarize_sessions(sessions: Sequence[SessionSpec]) -> CohortSummary:
    if not sessions:
        raise WorkloadValidationError("comparison cohort cannot be empty")
    session_ids = [session.session_id for session in sessions]
    if len(session_ids) != len(set(session_ids)):
        raise WorkloadValidationError(
            "comparison cohort contains duplicate session_id values"
        )
    source_indices = [session.source_index for session in sessions]
    if len(source_indices) != len(set(source_indices)):
        raise WorkloadValidationError(
            "comparison cohort contains duplicate source indices"
        )

    calls = [call for session in sessions for call in session.calls]
    resumes = [call for call in calls if call.is_resume]
    return CohortSummary(
        session_count=len(sessions),
        call_count=len(calls),
        first_turn_count=sum(call.is_first_turn for call in calls),
        resume_count=len(resumes),
        adjacent_cached_resume_count=sum(
            call.has_cached_prefix for call in resumes
        ),
        context_shrink_resume_count=sum(
            call.is_context_shrink for call in resumes
        ),
        total_input_tokens=sum(call.input_tokens for call in calls),
        total_output_tokens=sum(call.output_tokens for call in calls),
        total_cached_prefix_tokens=sum(
            call.cached_prefix_tokens for call in calls
        ),
        total_fresh_input_tokens=sum(
            call.fresh_input_tokens for call in calls
        ),
        resume_fresh_input_tokens=sum(
            call.fresh_input_tokens for call in resumes
        ),
        single_output_call_count=sum(
            call.output_tokens == 1 for call in calls
        ),
        tpot_eligible_call_count=sum(call.tpot_eligible for call in calls),
        max_input_context_tokens=max(call.input_tokens for call in calls),
        max_sequence_tokens=max(
            call.input_tokens + call.output_tokens for call in calls
        ),
        selected_session_ids_sha256=stable_json_sha256(session_ids),
        source_index_session_id_sha256=stable_json_sha256([
            {
                "source_index": session.source_index,
                "session_id": session.session_id,
            }
            for session in sessions
        ]),
    )


def _validate_summary(
        observed: CohortSummary,
        expected: CohortSummary,
) -> None:
    mismatches = {
        field: (getattr(observed, field), getattr(expected, field))
        for field in CohortSummary.__dataclass_fields__
        if getattr(observed, field) != getattr(expected, field)
    }
    if mismatches:
        details = ", ".join(
            f"{field}={observed_value!r} (expected {expected_value!r})"
            for field, (observed_value, expected_value)
            in sorted(mismatches.items())
        )
        raise WorkloadValidationError(
            f"comparison cohort contract mismatch: {details}"
        )


def load_comparison_workload(
        path: str | Path,
        *,
        source_indices: Sequence[int],
        expected_summary: CohortSummary | None = None,
        expected_source_sha256: str | None = None,
        expected_source_session_count: int | None = None,
) -> ComparisonWorkload:
    """Stream a schema-3 JSONL and materialize only selected complete rows."""

    source_path = Path(path).expanduser().resolve()
    if not source_indices:
        raise ValueError("source_indices cannot be empty")
    indices = tuple(source_indices)
    if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in indices):
        raise ValueError("source_indices must be non-negative integers")
    if len(indices) != len(set(indices)):
        raise ValueError("source_indices cannot contain duplicates")
    if tuple(sorted(indices)) != indices:
        raise ValueError("source_indices must be strictly increasing")
    if expected_source_sha256 is not None and not _is_sha256(
            expected_source_sha256):
        raise ValueError("expected_source_sha256 must be a lowercase SHA-256")
    if (
            expected_source_session_count is not None
            and (
                isinstance(expected_source_session_count, bool)
                or not isinstance(expected_source_session_count, int)
                or expected_source_session_count <= 0
            )
    ):
        raise ValueError("expected_source_session_count must be positive")

    selected = {}
    digest = hashlib.sha256()
    source_session_count = 0
    wanted = set(indices)
    try:
        with source_path.open("rb") as source:
            for source_index, raw_line in enumerate(source):
                digest.update(raw_line)
                source_session_count = source_index + 1
                if source_index not in wanted:
                    continue
                try:
                    raw = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise WorkloadValidationError(
                        f"invalid JSON at selected source row {source_index}: "
                        f"{exc}"
                    ) from exc
                selected[source_index] = _parse_session(raw, source_index)
    except OSError as exc:
        raise WorkloadValidationError(
            f"unable to read comparison workload {source_path}: {exc}"
        ) from exc

    missing = [index for index in indices if index not in selected]
    if missing:
        raise WorkloadValidationError(
            f"comparison workload is missing source indices {missing}"
        )
    source_sha256 = digest.hexdigest()
    if (
            expected_source_sha256 is not None
            and source_sha256 != expected_source_sha256
    ):
        raise WorkloadValidationError(
            "comparison source SHA-256 mismatch: "
            f"observed={source_sha256}, expected={expected_source_sha256}"
        )
    if (
            expected_source_session_count is not None
            and source_session_count != expected_source_session_count
    ):
        raise WorkloadValidationError(
            "comparison source session count mismatch: "
            f"observed={source_session_count}, "
            f"expected={expected_source_session_count}"
        )

    sessions = tuple(selected[index] for index in indices)
    summary = summarize_sessions(sessions)
    if expected_summary is not None:
        _validate_summary(summary, expected_summary)
    return ComparisonWorkload(
        source_path=source_path,
        source_sha256=source_sha256,
        source_session_count=source_session_count,
        sessions=sessions,
        summary=summary,
    )


def load_fixed_comparison_workload(path: str | Path) -> ComparisonWorkload:
    """Load and strictly validate the preregistered 32-session TraceLab set."""

    return load_comparison_workload(
        path,
        source_indices=FIXED_SOURCE_INDICES,
        expected_summary=FIXED_COHORT_SUMMARY,
        expected_source_sha256=TRACELAB_SCHEMA3_SHA256,
        expected_source_session_count=4281,
    )


def build_offered_plan(
        sessions: Sequence[SessionSpec],
        *,
        seed: int,
        shuffle: bool = True,
) -> OfferedPlan:
    """Freeze offered order and unit exponential draws for paired runs.

    The first session arrives at unit time zero.  Every subsequent gap is a
    unit-rate exponential draw.  A rate sweep only rescales the accumulated
    unit coordinates; it never redraws or reorders them.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(shuffle, bool):
        raise ValueError("shuffle must be a boolean")
    if not sessions:
        raise ValueError("sessions cannot be empty")
    session_ids = [session.session_id for session in sessions]
    if len(session_ids) != len(set(session_ids)):
        raise ValueError("sessions contain duplicate session_id values")

    rng = random.Random(seed)
    ordered = list(sessions)
    if shuffle:
        rng.shuffle(ordered)

    offers = []
    unit_arrival_time = 0.0
    unit_draws = []
    for offer_index, session in enumerate(ordered):
        if offer_index == 0:
            unit_interarrival = 0.0
        else:
            # Explicit inverse CDF keeps the unit draw definition visible and
            # independent from the rate used by a particular experiment.
            unit_interarrival = -math.log1p(-rng.random())
        unit_arrival_time += unit_interarrival
        unit_draws.append(unit_interarrival)
        offers.append(OfferedSession(
            offer_index=offer_index,
            session=session,
            unit_interarrival=unit_interarrival,
            unit_arrival_time=unit_arrival_time,
        ))

    return OfferedPlan(
        seed=seed,
        offers=tuple(offers),
        offered_session_ids_sha256=stable_json_sha256([
            session.session_id for session in ordered
        ]),
        unit_draws_sha256=stable_json_sha256([
            draw.hex() for draw in unit_draws
        ]),
    )


def full_drain_hashes(
        expected_identities: Iterable[str],
        completed_identities: Iterable[str],
) -> FullDrainHashes:
    """Validate an exact, duplicate-free full drain and return audit hashes."""

    expected = tuple(expected_identities)
    completed = tuple(completed_identities)
    for label, identities in (
            ("expected", expected), ("completed", completed)):
        if any(
                not isinstance(identity, str) or not identity
                for identity in identities):
            raise WorkloadValidationError(
                f"{label} identities must be non-empty strings"
            )
        if len(identities) != len(set(identities)):
            raise WorkloadValidationError(
                f"{label} identities contain duplicates"
            )

    expected_set = set(expected)
    completed_set = set(completed)
    if expected_set != completed_set:
        missing = sorted(expected_set - completed_set)
        unexpected = sorted(completed_set - expected_set)
        raise WorkloadValidationError(
            "comparison did not fully drain the fixed identity set: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    return FullDrainHashes(
        identity_count=len(expected),
        offered_order_sha256=stable_json_sha256(list(expected)),
        expected_set_sha256=stable_json_sha256(sorted(expected_set)),
        completion_order_sha256=stable_json_sha256(list(completed)),
        completion_set_sha256=stable_json_sha256(sorted(completed_set)),
    )


def session_full_drain_hashes(
        plan: OfferedPlan,
        completed_session_ids: Iterable[str],
) -> FullDrainHashes:
    return full_drain_hashes(
        (offer.session.session_id for offer in plan.offers),
        completed_session_ids,
    )


def call_full_drain_hashes(
        workload: ComparisonWorkload,
        completed_call_identities: Iterable[str],
) -> FullDrainHashes:
    return full_drain_hashes(
        workload.call_completion_identities,
        completed_call_identities,
    )

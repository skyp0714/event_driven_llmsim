"""Fail-closed workload and metric plumbing for live simulator comparisons.

This module bridges the immutable comparison schedule to LLMServingSim's
agentic JSONL input and parses the simulator's native ``requests.csv`` output.
It deliberately recomputes latency and SLO results from request timestamps
instead of trusting aggregate log output.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

from .hbf_comparison_workload import CallSpec, ScheduledSession


NANOSECONDS_PER_SECOND = 1_000_000_000
DEFAULT_TTFT_SLO_NS = 30 * NANOSECONDS_PER_SECOND
DEFAULT_TPOT_SLO_NS = 300_000_000

RequestIdentity = tuple[str, int]

_REQUIRED_REQUEST_CSV_FIELDS = frozenset({
    "session_id",
    "sub_request_index",
    "arrival",
    "end_time",
    "latency",
    "TTFT",
    "TPOT",
    "output",
    "generated_tokens",
})


class LiveComparisonMetricsError(ValueError):
    """Raised when a live comparison artifact violates its frozen contract."""


@dataclass(frozen=True)
class MaterializedWorkload:
    """Content address and roster for one atomically written workload."""

    path: Path
    sha256: str
    byte_count: int
    session_count: int
    request_count: int
    request_identities: tuple[RequestIdentity, ...]


@dataclass(frozen=True)
class LiveServingRequest:
    """One fully validated row from LLMServingSim's native request CSV."""

    session_id: str
    call_index: int
    arrival_ns: int
    first_token_ns: int
    completion_ns: int
    output_tokens: int
    csv_tpot_ns: int

    @property
    def identity(self) -> RequestIdentity:
        return (self.session_id, self.call_index)

    @property
    def is_resume(self) -> bool:
        return self.call_index > 0

    @property
    def ttft_ns(self) -> int:
        return self.first_token_ns - self.arrival_ns

    @property
    def tpot_ns(self) -> Fraction | None:
        if self.output_tokens < 2:
            return None
        return Fraction(
            self.completion_ns - self.first_token_ns,
            self.output_tokens - 1,
        )


@dataclass(frozen=True)
class ExactDistribution:
    """A nearest-rank distribution derived from exact request values."""

    count: int
    mean_ns: float
    minimum_ns: float
    p50_ns: float
    p95_ns: float
    p99_ns: float
    maximum_ns: float
    percentile_method: str = "inclusive_nearest_rank"


@dataclass(frozen=True)
class LiveComparisonMetrics:
    """Exact live-request SLO metrics over a fixed measurement roster."""

    measurement_session_ids: tuple[str, ...]
    measurement_request_count: int
    resume_request_count: int
    tpot_eligible_request_count: int
    resume_tpot_eligible_request_count: int
    ttft_slo_ns: int
    tpot_slo_ns: int
    resume_ttft_ns: ExactDistribution
    tpot_ns: ExactDistribution
    resume_tpot_ns: ExactDistribution
    joint_slo_pass_count: int
    joint_slo_fail_count: int
    resume_joint_slo_pass_count: int
    resume_joint_slo_fail_count: int
    joint_slo_pass_output_tokens: int
    joint_slo_pass_session_count: int
    joint_slo_fail_session_count: int
    window_start_ns: int
    window_end_ns: int
    window_duration_ns: int
    operational_request_goodput_per_second: float
    operational_resume_goodput_per_second: float
    operational_token_goodput_per_second: float
    operational_session_goodput_per_second: float


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LiveComparisonMetricsError(
            f"{name} must be a non-negative integer")
    return value


def _require_positive_int(name: str, value: object) -> int:
    integer = _require_nonnegative_int(name, value)
    if integer == 0:
        raise LiveComparisonMetricsError(
            f"{name} must be a positive integer")
    return integer


def _require_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise LiveComparisonMetricsError(
            f"{name} must be a non-empty string")
    return value


def _require_optional_string(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(name, value)


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LiveComparisonMetricsError(
            f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_coordinate(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiveComparisonMetricsError(
            f"{name} must be a non-negative finite number")
    coordinate = float(value)
    if not math.isfinite(coordinate) or coordinate < 0.0:
        raise LiveComparisonMetricsError(
            f"{name} must be a non-negative finite number")
    return coordinate


def _validate_call(
        scheduled: ScheduledSession,
        call: CallSpec,
        expected_call_index: int,
) -> None:
    if not isinstance(call, CallSpec):
        raise LiveComparisonMetricsError(
            "scheduled session contains a non-CallSpec value")
    if call.session_id != scheduled.session.session_id:
        raise LiveComparisonMetricsError(
            "call session_id disagrees with its containing session")
    if call.source_index != scheduled.session.source_index:
        raise LiveComparisonMetricsError(
            "call source_index disagrees with its containing session")
    if call.call_index != expected_call_index:
        raise LiveComparisonMetricsError(
            "call indices must be contiguous and start at zero")
    _require_positive_int("input_tokens", call.input_tokens)
    _require_positive_int("output_tokens", call.output_tokens)
    _require_nonnegative_int("tool_duration_ns", call.tool_duration_ns)
    cached = _require_nonnegative_int(
        "cached_prefix_tokens", call.cached_prefix_tokens)
    if cached > call.input_tokens:
        raise LiveComparisonMetricsError(
            "cached_prefix_tokens exceeds input_tokens")
    if call.call_index == 0 and cached:
        raise LiveComparisonMetricsError(
            "a first call cannot reuse an earlier session prefix")
    if call.fresh_input_tokens != call.input_tokens - cached:
        raise LiveComparisonMetricsError(
            "fresh_input_tokens disagrees with input minus cached prefix")
    _require_optional_string("lineage_status", call.lineage_status)
    _require_optional_string(
        "inter_turn_gap_type", call.inter_turn_gap_type)


def _schedule_contract(
        scheduled_sessions: tuple[ScheduledSession, ...],
) -> tuple[
        tuple[RequestIdentity, ...],
        dict[RequestIdentity, int],
        dict[str, int],
]:
    if not isinstance(scheduled_sessions, tuple):
        raise LiveComparisonMetricsError(
            "scheduled_sessions must be an immutable tuple")
    if not scheduled_sessions:
        raise LiveComparisonMetricsError(
            "scheduled_sessions cannot be empty")

    identities: list[RequestIdentity] = []
    output_tokens: dict[RequestIdentity, int] = {}
    arrivals: dict[str, int] = {}
    offer_indices: set[int] = set()
    previous_arrival = -1
    for scheduled in scheduled_sessions:
        if not isinstance(scheduled, ScheduledSession):
            raise LiveComparisonMetricsError(
                "scheduled_sessions contains a non-ScheduledSession value")
        session = scheduled.session
        session_id = _require_nonempty_string(
            "session_id", session.session_id)
        if session_id in arrivals:
            raise LiveComparisonMetricsError(
                f"duplicate scheduled session_id {session_id!r}")
        _require_nonnegative_int(
            "source_index", session.source_index)
        _require_nonnegative_int(
            "source_arrival_time_ns", session.source_arrival_time_ns)
        if session.source_session_identity_sha256 is not None:
            _require_sha256(
                "source_session_identity_sha256",
                session.source_session_identity_sha256,
            )
        offer_index = _require_nonnegative_int(
            "offer_index", scheduled.offer_index)
        if offer_index in offer_indices:
            raise LiveComparisonMetricsError(
                f"duplicate scheduled offer_index {offer_index}")
        offer_indices.add(offer_index)
        arrival = _require_nonnegative_int(
            "arrival_time_ns", scheduled.arrival_time_ns)
        if arrival < previous_arrival:
            raise LiveComparisonMetricsError(
                "scheduled arrivals must be nondecreasing")
        previous_arrival = arrival
        _require_coordinate(
            "unit_interarrival", scheduled.unit_interarrival)
        _require_coordinate(
            "unit_arrival_time", scheduled.unit_arrival_time)
        if not isinstance(session.calls, tuple) or not session.calls:
            raise LiveComparisonMetricsError(
                f"scheduled session {session_id!r} must contain calls")
        arrivals[session_id] = arrival
        for call_index, call in enumerate(session.calls):
            _validate_call(scheduled, call, call_index)
            identity = (session_id, call_index)
            if identity in output_tokens:
                raise LiveComparisonMetricsError(
                    f"duplicate scheduled request identity {identity!r}")
            identities.append(identity)
            output_tokens[identity] = call.output_tokens
    return tuple(identities), output_tokens, arrivals


def expected_request_identities(
        scheduled_sessions: tuple[ScheduledSession, ...],
) -> tuple[RequestIdentity, ...]:
    """Return the exact scheduled request roster in offer and call order."""

    identities, _, _ = _schedule_contract(scheduled_sessions)
    return identities


def _session_json_row(
        scheduled: ScheduledSession,
        source_sha256: str,
) -> dict[str, object]:
    session = scheduled.session
    metadata: dict[str, object] = {
        "source_sha256": source_sha256,
        "source_index": session.source_index,
        "source_arrival_time_ns": session.source_arrival_time_ns,
        "source_session_identity_sha256": (
            session.source_session_identity_sha256),
        "offer_index": scheduled.offer_index,
        "unit_interarrival": float(scheduled.unit_interarrival),
        "unit_interarrival_hex": float(
            scheduled.unit_interarrival).hex(),
        "unit_arrival_time": float(scheduled.unit_arrival_time),
        "unit_arrival_time_hex": float(
            scheduled.unit_arrival_time).hex(),
    }
    sub_requests = []
    for call in session.calls:
        row: dict[str, object] = {
            "input_toks": call.input_tokens,
            "output_toks": call.output_tokens,
            "tool_duration_ns": call.tool_duration_ns,
            "prefix_reuse_toks": call.cached_prefix_tokens,
        }
        if call.lineage_status is not None:
            row["lineage_status"] = call.lineage_status
        if call.inter_turn_gap_type is not None:
            row["inter_turn_gap_type"] = call.inter_turn_gap_type
        sub_requests.append(row)
    return {
        "session_id": session.session_id,
        "arrival_time_ns": scheduled.arrival_time_ns,
        "trace_metadata": metadata,
        "sub_requests": sub_requests,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_descriptor = os.open(
                path.parent, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def materialize_scheduled_sessions(
        scheduled_sessions: tuple[ScheduledSession, ...],
        output_path: str | Path,
        *,
        source_sha256: str,
) -> MaterializedWorkload:
    """Atomically write a lossless agentic JSONL workload and return its SHA."""

    identities, _, _ = _schedule_contract(scheduled_sessions)
    digest = _require_sha256("source_sha256", source_sha256)
    path = Path(output_path)
    if not path.name:
        raise LiveComparisonMetricsError(
            "output_path must name a workload file")
    lines = [
        json.dumps(
            _session_json_row(scheduled, digest),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        for scheduled in scheduled_sessions
    ]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    _atomic_write(path, payload)
    observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_sha256 != payload_sha256:
        raise RuntimeError(
            "atomically published workload failed its SHA-256 verification")
    return MaterializedWorkload(
        path=path,
        sha256=payload_sha256,
        byte_count=len(payload),
        session_count=len(scheduled_sessions),
        request_count=len(identities),
        request_identities=identities,
    )


def _normalize_expected_identities(
        values: Iterable[RequestIdentity],
) -> tuple[RequestIdentity, ...]:
    try:
        identities = tuple(values)
    except TypeError as exc:
        raise LiveComparisonMetricsError(
            "expected_identities must be iterable") from exc
    if not identities:
        raise LiveComparisonMetricsError(
            "expected_identities cannot be empty")
    normalized = []
    for identity in identities:
        if not isinstance(identity, tuple) or len(identity) != 2:
            raise LiveComparisonMetricsError(
                "expected identities must be (session_id, call_index) tuples")
        session_id = _require_nonempty_string(
            "expected session_id", identity[0])
        call_index = _require_nonnegative_int(
            "expected call_index", identity[1])
        normalized.append((session_id, call_index))
    if len(normalized) != len(set(normalized)):
        raise LiveComparisonMetricsError(
            "expected_identities contains duplicates")
    return tuple(normalized)


def _parse_csv_integer(
        row: Mapping[str, str],
        field: str,
        *,
        line_number: int,
) -> int:
    raw = row.get(field)
    if raw is None or not raw or raw.strip() != raw:
        raise LiveComparisonMetricsError(
            f"requests.csv line {line_number} has invalid {field!r}")
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise LiveComparisonMetricsError(
            f"requests.csv line {line_number} has invalid {field!r}") from exc
    if str(value) != raw:
        raise LiveComparisonMetricsError(
            f"requests.csv line {line_number} has non-canonical {field!r}")
    return value


def parse_serving_requests_csv(
        requests_csv: str | Path,
        *,
        expected_identities: Iterable[RequestIdentity],
) -> tuple[LiveServingRequest, ...]:
    """Parse native simulator output and require exact request-set equality."""

    expected = _normalize_expected_identities(expected_identities)
    expected_set = set(expected)
    path = Path(requests_csv)
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise LiveComparisonMetricsError(
            f"cannot read requests.csv at {path}") from exc
    parsed: list[LiveServingRequest] = []
    seen: set[RequestIdentity] = set()
    with handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if header is None:
            raise LiveComparisonMetricsError(
                "requests.csv is missing its header")
        if len(header) != len(set(header)) or any(not field for field in header):
            raise LiveComparisonMetricsError(
                "requests.csv has duplicate or empty header fields")
        missing_fields = _REQUIRED_REQUEST_CSV_FIELDS - set(header)
        if missing_fields:
            raise LiveComparisonMetricsError(
                "requests.csv is missing required fields: "
                + ", ".join(sorted(missing_fields))
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise LiveComparisonMetricsError(
                    f"requests.csv line {line_number} has extra columns")
            session_id = _require_nonempty_string(
                f"requests.csv line {line_number} session_id",
                row["session_id"],
            )
            call_index = _parse_csv_integer(
                row, "sub_request_index", line_number=line_number)
            if call_index < 0:
                raise LiveComparisonMetricsError(
                    f"requests.csv line {line_number} has negative call index")
            identity = (session_id, call_index)
            if identity in seen:
                raise LiveComparisonMetricsError(
                    f"requests.csv contains duplicate identity {identity!r}")
            if identity not in expected_set:
                raise LiveComparisonMetricsError(
                    f"requests.csv contains unexpected identity {identity!r}")
            seen.add(identity)

            arrival = _parse_csv_integer(
                row, "arrival", line_number=line_number)
            completion = _parse_csv_integer(
                row, "end_time", line_number=line_number)
            latency = _parse_csv_integer(
                row, "latency", line_number=line_number)
            ttft = _parse_csv_integer(
                row, "TTFT", line_number=line_number)
            csv_tpot = _parse_csv_integer(
                row, "TPOT", line_number=line_number)
            output_tokens = _parse_csv_integer(
                row, "output", line_number=line_number)
            generated_tokens = _parse_csv_integer(
                row, "generated_tokens", line_number=line_number)
            for field, value in (
                ("arrival", arrival),
                ("end_time", completion),
                ("latency", latency),
                ("TTFT", ttft),
                ("TPOT", csv_tpot),
            ):
                if value < 0:
                    raise LiveComparisonMetricsError(
                        f"requests.csv line {line_number} has negative {field}")
            if output_tokens <= 0 or generated_tokens <= 0:
                raise LiveComparisonMetricsError(
                    f"requests.csv line {line_number} has non-positive output")
            if generated_tokens != output_tokens:
                raise LiveComparisonMetricsError(
                    f"requests.csv line {line_number} did not generate "
                    "the requested output token count")
            if completion < arrival or latency != completion - arrival:
                raise LiveComparisonMetricsError(
                    f"requests.csv line {line_number} has inconsistent "
                    "arrival, completion, and latency")
            if ttft > latency:
                raise LiveComparisonMetricsError(
                    f"requests.csv line {line_number} has TTFT after completion")
            post_first_token_ns = latency - ttft
            if output_tokens == 1:
                if post_first_token_ns != 0 or csv_tpot != 0:
                    raise LiveComparisonMetricsError(
                        f"requests.csv line {line_number} has invalid "
                        "one-token timing")
            elif csv_tpot != post_first_token_ns // (output_tokens - 1):
                raise LiveComparisonMetricsError(
                    f"requests.csv line {line_number} has inconsistent TPOT")
            parsed.append(LiveServingRequest(
                session_id=session_id,
                call_index=call_index,
                arrival_ns=arrival,
                first_token_ns=arrival + ttft,
                completion_ns=completion,
                output_tokens=output_tokens,
                csv_tpot_ns=csv_tpot,
            ))
    missing = expected_set - seen
    if missing:
        rendered = ", ".join(
            f"{session_id}:{call_index}"
            for session_id, call_index in sorted(missing)[:5]
        )
        raise LiveComparisonMetricsError(
            "requests.csv is missing expected identities: "
            f"{rendered}"
        )
    return tuple(parsed)


def _nearest_rank(
        ordered: Sequence[Fraction],
        percentile: float,
) -> Fraction:
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _distribution(
        values: Sequence[int | Fraction],
        *,
        name: str,
) -> ExactDistribution:
    if not values:
        raise LiveComparisonMetricsError(
            f"{name} distribution cannot be empty")
    exact = tuple(
        value if isinstance(value, Fraction) else Fraction(value)
        for value in values
    )
    ordered = tuple(sorted(exact))
    return ExactDistribution(
        count=len(ordered),
        mean_ns=float(sum(ordered, Fraction()) / len(ordered)),
        minimum_ns=float(ordered[0]),
        p50_ns=float(_nearest_rank(ordered, 0.50)),
        p95_ns=float(_nearest_rank(ordered, 0.95)),
        p99_ns=float(_nearest_rank(ordered, 0.99)),
        maximum_ns=float(ordered[-1]),
    )


def _request_joint_slo_pass(
        request: LiveServingRequest,
        *,
        ttft_slo_ns: int,
        tpot_slo_ns: int,
) -> bool:
    if request.ttft_ns > ttft_slo_ns:
        return False
    tpot = request.tpot_ns
    return tpot is None or tpot <= tpot_slo_ns


def compute_live_comparison_metrics(
        scheduled_sessions: tuple[ScheduledSession, ...],
        requests: Sequence[LiveServingRequest],
        *,
        measurement_session_ids: Sequence[str],
        ttft_slo_ns: int = DEFAULT_TTFT_SLO_NS,
        tpot_slo_ns: int = DEFAULT_TPOT_SLO_NS,
) -> LiveComparisonMetrics:
    """Compute operational goodput from one exact live request cohort.

    The operational window starts at the first measured session's offered
    arrival and ends at the final completion among all of its requests.
    Request and token goodput count only joint-SLO-passing calls.  A session
    contributes to session goodput only when every one of its calls passes.
    """

    identities, expected_outputs, session_arrivals = _schedule_contract(
        scheduled_sessions)
    ttft_limit = _require_positive_int("ttft_slo_ns", ttft_slo_ns)
    tpot_limit = _require_positive_int("tpot_slo_ns", tpot_slo_ns)
    measured_ids = tuple(measurement_session_ids)
    if not measured_ids:
        raise LiveComparisonMetricsError(
            "measurement_session_ids cannot be empty")
    if len(measured_ids) != len(set(measured_ids)):
        raise LiveComparisonMetricsError(
            "measurement_session_ids contains duplicates")
    for session_id in measured_ids:
        _require_nonempty_string("measurement session_id", session_id)
        if session_id not in session_arrivals:
            raise LiveComparisonMetricsError(
                f"unknown measurement session_id {session_id!r}")

    expected_set = set(identities)
    indexed: dict[RequestIdentity, LiveServingRequest] = {}
    for request in requests:
        if not isinstance(request, LiveServingRequest):
            raise LiveComparisonMetricsError(
                "requests contains a non-LiveServingRequest value")
        identity = request.identity
        if identity in indexed:
            raise LiveComparisonMetricsError(
                f"requests contains duplicate identity {identity!r}")
        if identity not in expected_set:
            raise LiveComparisonMetricsError(
                f"requests contains unexpected identity {identity!r}")
        if request.output_tokens != expected_outputs[identity]:
            raise LiveComparisonMetricsError(
                f"output-token mismatch for identity {identity!r}")
        indexed[identity] = request
    missing = expected_set - set(indexed)
    if missing:
        raise LiveComparisonMetricsError(
            "requests does not contain the complete scheduled roster")

    measured_set = set(measured_ids)
    measured = tuple(
        indexed[identity]
        for identity in identities
        if identity[0] in measured_set
    )
    if not measured:
        raise LiveComparisonMetricsError(
            "measurement request cohort cannot be empty")
    resumes = tuple(request for request in measured if request.is_resume)
    if not resumes:
        raise LiveComparisonMetricsError(
            "measurement request cohort has no resume requests")
    tpot_eligible = tuple(
        request for request in measured if request.tpot_ns is not None)
    if not tpot_eligible:
        raise LiveComparisonMetricsError(
            "measurement request cohort has no TPOT-eligible requests")
    resume_tpot_eligible = tuple(
        request for request in resumes if request.tpot_ns is not None)
    if not resume_tpot_eligible:
        raise LiveComparisonMetricsError(
            "measurement resume cohort has no TPOT-eligible requests")

    pass_by_identity = {
        request.identity: _request_joint_slo_pass(
            request,
            ttft_slo_ns=ttft_limit,
            tpot_slo_ns=tpot_limit,
        )
        for request in measured
    }
    passed = tuple(
        request for request in measured
        if pass_by_identity[request.identity]
    )
    passed_resumes = tuple(
        request for request in resumes
        if pass_by_identity[request.identity]
    )
    passed_sessions = tuple(
        session_id
        for session_id in measured_ids
        if all(
            pass_by_identity[request.identity]
            for request in measured
            if request.session_id == session_id
        )
    )
    window_start = min(
        session_arrivals[session_id] for session_id in measured_ids)
    window_end = max(request.completion_ns for request in measured)
    window_duration = window_end - window_start
    if window_duration <= 0:
        raise LiveComparisonMetricsError(
            "operational measurement window must have positive duration")

    scale = NANOSECONDS_PER_SECOND / window_duration
    return LiveComparisonMetrics(
        measurement_session_ids=measured_ids,
        measurement_request_count=len(measured),
        resume_request_count=len(resumes),
        tpot_eligible_request_count=len(tpot_eligible),
        resume_tpot_eligible_request_count=len(resume_tpot_eligible),
        ttft_slo_ns=ttft_limit,
        tpot_slo_ns=tpot_limit,
        resume_ttft_ns=_distribution(
            [request.ttft_ns for request in resumes],
            name="resume TTFT",
        ),
        tpot_ns=_distribution(
            [
                request.tpot_ns
                for request in tpot_eligible
                if request.tpot_ns is not None
            ],
            name="TPOT",
        ),
        resume_tpot_ns=_distribution(
            [
                request.tpot_ns
                for request in resume_tpot_eligible
                if request.tpot_ns is not None
            ],
            name="resume TPOT",
        ),
        joint_slo_pass_count=len(passed),
        joint_slo_fail_count=len(measured) - len(passed),
        resume_joint_slo_pass_count=len(passed_resumes),
        resume_joint_slo_fail_count=len(resumes) - len(passed_resumes),
        joint_slo_pass_output_tokens=sum(
            request.output_tokens for request in passed),
        joint_slo_pass_session_count=len(passed_sessions),
        joint_slo_fail_session_count=(
            len(measured_ids) - len(passed_sessions)),
        window_start_ns=window_start,
        window_end_ns=window_end,
        window_duration_ns=window_duration,
        operational_request_goodput_per_second=len(passed) * scale,
        operational_resume_goodput_per_second=(
            len(passed_resumes) * scale),
        operational_token_goodput_per_second=sum(
            request.output_tokens for request in passed) * scale,
        operational_session_goodput_per_second=(
            len(passed_sessions) * scale),
    )

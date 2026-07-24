"""Request-level latency, SLO, and paired-comparison metrics.

The functions in this module operate on fully drained request cohorts.  A
request is identified by ``(session_id, sub_request_index)`` so that systems
with different completion orders can still be compared on exactly the same
work.  Goodput is derived from the offered *session* rate; this is important
for agentic traces where one admitted session releases many requests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Optional, Sequence


NANOSECONDS_PER_SECOND = 1_000_000_000
NANOSECONDS_PER_MILLISECOND = 1_000_000

DEFAULT_FIRST_TTFT_SLO_NS = 30 * NANOSECONDS_PER_SECOND
DEFAULT_RESUME_TTFT_SLO_NS = 30 * NANOSECONDS_PER_SECOND
DEFAULT_TPOT_SLO_NS = 300 * NANOSECONDS_PER_MILLISECOND


class ComparisonMetricError(ValueError):
    """Raised when a metric cohort violates the comparison contract."""


def _require_nonnegative_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ComparisonMetricError(
            f"{name} must be a non-negative integer, got {value!r}")


def _require_positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ComparisonMetricError(
            f"{name} must be a positive integer, got {value!r}")


def _require_finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ComparisonMetricError(
            f"{name} must be a finite number, got {value!r}")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonMetricError(
            f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(converted):
        raise ComparisonMetricError(
            f"{name} must be a finite number, got {value!r}")
    return converted


@dataclass(frozen=True, order=True)
class RequestKey:
    """Stable request identity shared by every compared system."""

    session_id: str
    sub_request_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ComparisonMetricError(
                "session_id must be a non-empty string")
        _require_nonnegative_integer(
            "sub_request_index", self.sub_request_index)

    @property
    def is_resume(self) -> bool:
        return self.sub_request_index > 0

    @property
    def kind(self) -> str:
        return "resume" if self.is_resume else "first"


@dataclass(frozen=True)
class CompletedRequest:
    """The timestamps needed for exact request latency accounting."""

    key: RequestKey
    release_ns: int
    first_token_ns: int
    completion_ns: int
    output_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, RequestKey):
            raise ComparisonMetricError("key must be a RequestKey")
        _require_nonnegative_integer("release_ns", self.release_ns)
        _require_nonnegative_integer("first_token_ns", self.first_token_ns)
        _require_nonnegative_integer("completion_ns", self.completion_ns)
        _require_positive_integer("output_tokens", self.output_tokens)
        if self.first_token_ns < self.release_ns:
            raise ComparisonMetricError(
                f"{self.key}: first token precedes request release")
        if self.completion_ns < self.first_token_ns:
            raise ComparisonMetricError(
                f"{self.key}: completion precedes first token")
        if (
            self.output_tokens == 1
            and self.completion_ns != self.first_token_ns
        ):
            raise ComparisonMetricError(
                f"{self.key}: one-token completion must equal first token")

    @property
    def ttft_ns(self) -> int:
        """Time from request release until the first output token."""

        return self.first_token_ns - self.release_ns

    @property
    def tpot_ns(self) -> Optional[float]:
        """Mean inter-token time after the first output token."""

        if self.output_tokens == 1:
            return None
        return (
            (self.completion_ns - self.first_token_ns)
            / (self.output_tokens - 1)
        )

    @property
    def is_resume(self) -> bool:
        return self.key.is_resume


@dataclass(frozen=True)
class SLOThresholds:
    first_ttft_ns: int = DEFAULT_FIRST_TTFT_SLO_NS
    resume_ttft_ns: int = DEFAULT_RESUME_TTFT_SLO_NS
    tpot_ns: int = DEFAULT_TPOT_SLO_NS

    def __post_init__(self) -> None:
        _require_positive_integer("first_ttft_ns", self.first_ttft_ns)
        _require_positive_integer("resume_ttft_ns", self.resume_ttft_ns)
        _require_positive_integer("tpot_ns", self.tpot_ns)

    def ttft_ns_for(self, request: CompletedRequest) -> int:
        return self.resume_ttft_ns if request.is_resume else self.first_ttft_ns


DEFAULT_SLO_THRESHOLDS = SLOThresholds()


def slo_sensitivity_grid(
        *, first_ttft_seconds: int = 30,
        resume_ttft_seconds: Sequence[int] = (30, 60, 120),
        tpot_milliseconds: Sequence[int] = (100, 300, 600),
) -> tuple[SLOThresholds, ...]:
    """Return the preregistered resume-TTFT by TPOT sensitivity grid."""

    _require_positive_integer("first_ttft_seconds", first_ttft_seconds)
    if not resume_ttft_seconds or not tpot_milliseconds:
        raise ComparisonMetricError(
            "SLO sensitivity axes must both be non-empty")
    grid = []
    for ttft_seconds in resume_ttft_seconds:
        _require_positive_integer("resume_ttft_seconds", ttft_seconds)
        for tpot_ms in tpot_milliseconds:
            _require_positive_integer("tpot_milliseconds", tpot_ms)
            grid.append(SLOThresholds(
                first_ttft_ns=(
                    first_ttft_seconds * NANOSECONDS_PER_SECOND),
                resume_ttft_ns=ttft_seconds * NANOSECONDS_PER_SECOND,
                tpot_ns=tpot_ms * NANOSECONDS_PER_MILLISECOND,
            ))
    return tuple(grid)


def ttft_slo_pass(
        request: CompletedRequest,
        thresholds: SLOThresholds = DEFAULT_SLO_THRESHOLDS,
) -> bool:
    return request.ttft_ns <= thresholds.ttft_ns_for(request)


def tpot_slo_pass(
        request: CompletedRequest,
        thresholds: SLOThresholds = DEFAULT_SLO_THRESHOLDS,
) -> Optional[bool]:
    tpot_ns = request.tpot_ns
    if tpot_ns is None:
        return None
    return tpot_ns <= thresholds.tpot_ns


def joint_slo_pass(
        request: CompletedRequest,
        thresholds: SLOThresholds = DEFAULT_SLO_THRESHOLDS,
) -> bool:
    """Return the request-level joint SLO result.

    A one-token request has no inter-token interval, so its joint result is
    determined by TTFT alone.  Requests with two or more output tokens must
    satisfy both TTFT and TPOT.
    """

    if not ttft_slo_pass(request, thresholds):
        return False
    tpot_pass = tpot_slo_pass(request, thresholds)
    return True if tpot_pass is None else tpot_pass


def nearest_rank_percentile(
        values: Sequence[float], percentile: float) -> float:
    """Compute an inclusive nearest-rank percentile without third parties."""

    if not values:
        raise ComparisonMetricError(
            "percentile requires at least one value")
    percentile = _require_finite("percentile", percentile)
    if percentile <= 0.0 or percentile > 1.0:
        raise ComparisonMetricError(
            f"percentile must be in (0, 1], got {percentile!r}")
    ordered = sorted(
        _require_finite("percentile value", value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    mean: Optional[float]
    minimum: Optional[float]
    p50: Optional[float]
    p95: Optional[float]
    p99: Optional[float]
    maximum: Optional[float]
    percentile_method: str = "nearest_rank"


def summarize_distribution(
        values: Sequence[float]) -> DistributionSummary:
    converted = tuple(
        _require_finite("distribution value", value) for value in values)
    if not converted:
        return DistributionSummary(
            count=0,
            mean=None,
            minimum=None,
            p50=None,
            p95=None,
            p99=None,
            maximum=None,
        )
    ordered = tuple(sorted(converted))
    return DistributionSummary(
        count=len(ordered),
        mean=math.fsum(ordered) / len(ordered),
        minimum=ordered[0],
        p50=nearest_rank_percentile(ordered, 0.50),
        p95=nearest_rank_percentile(ordered, 0.95),
        p99=nearest_rank_percentile(ordered, 0.99),
        maximum=ordered[-1],
    )


@dataclass(frozen=True)
class SLOGoodput:
    pass_requests_per_second: float
    pass_output_tokens_per_second: float


def goodput_from_pass_counts(
        *, offered_session_rate: float, offered_session_count: int,
        pass_request_count: int,
        pass_output_tokens: int) -> SLOGoodput:
    """Scale pass counts per offered session by the offered session rate."""

    rate = _require_finite(
        "offered_session_rate", offered_session_rate)
    if rate < 0.0:
        raise ComparisonMetricError(
            "offered_session_rate must be non-negative")
    _require_positive_integer(
        "offered_session_count", offered_session_count)
    _require_nonnegative_integer(
        "pass_request_count", pass_request_count)
    _require_nonnegative_integer(
        "pass_output_tokens", pass_output_tokens)
    return SLOGoodput(
        pass_requests_per_second=(
            rate * pass_request_count / offered_session_count),
        pass_output_tokens_per_second=(
            rate * pass_output_tokens / offered_session_count),
    )


@dataclass(frozen=True)
class RequestKindSummary:
    kind: str
    request_count: int
    output_tokens: int
    ttft_ns: DistributionSummary
    tpot_ns: DistributionSummary
    tpot_eligible_count: int
    ttft_slo_pass_count: int
    ttft_slo_attainment: Optional[float]
    tpot_slo_pass_count: int
    tpot_slo_attainment: Optional[float]
    joint_slo_pass_count: int
    joint_slo_attainment: Optional[float]
    joint_slo_pass_output_tokens: int
    joint_slo_goodput: SLOGoodput


@dataclass(frozen=True)
class RequestCohortSummary:
    offered_session_rate: float
    offered_session_count: int
    request_count: int
    output_tokens: int
    thresholds: SLOThresholds
    all_requests: RequestKindSummary
    first: RequestKindSummary
    resume: RequestKindSummary

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _summarize_kind(
        kind: str, requests: Sequence[CompletedRequest], *,
        offered_session_rate: float, offered_session_count: int,
        thresholds: SLOThresholds) -> RequestKindSummary:
    ttft_values = tuple(float(request.ttft_ns) for request in requests)
    tpot_requests = tuple(
        request for request in requests if request.tpot_ns is not None)
    tpot_values = tuple(
        float(request.tpot_ns) for request in tpot_requests
        if request.tpot_ns is not None)
    ttft_passes = sum(
        ttft_slo_pass(request, thresholds) for request in requests)
    tpot_passes = sum(
        bool(tpot_slo_pass(request, thresholds))
        for request in tpot_requests)
    joint_pass_requests = tuple(
        request for request in requests
        if joint_slo_pass(request, thresholds))
    joint_output_tokens = sum(
        request.output_tokens for request in joint_pass_requests)
    request_count = len(requests)
    tpot_eligible_count = len(tpot_requests)
    goodput = goodput_from_pass_counts(
        offered_session_rate=offered_session_rate,
        offered_session_count=offered_session_count,
        pass_request_count=len(joint_pass_requests),
        pass_output_tokens=joint_output_tokens,
    )
    return RequestKindSummary(
        kind=kind,
        request_count=request_count,
        output_tokens=sum(request.output_tokens for request in requests),
        ttft_ns=summarize_distribution(ttft_values),
        tpot_ns=summarize_distribution(tpot_values),
        tpot_eligible_count=tpot_eligible_count,
        ttft_slo_pass_count=ttft_passes,
        ttft_slo_attainment=(
            ttft_passes / request_count if request_count else None),
        tpot_slo_pass_count=tpot_passes,
        tpot_slo_attainment=(
            tpot_passes / tpot_eligible_count
            if tpot_eligible_count else None),
        joint_slo_pass_count=len(joint_pass_requests),
        joint_slo_attainment=(
            len(joint_pass_requests) / request_count
            if request_count else None),
        joint_slo_pass_output_tokens=joint_output_tokens,
        joint_slo_goodput=goodput,
    )


def index_completed_requests(
        requests: Sequence[CompletedRequest], *,
        system_name: str = "system",
) -> dict[RequestKey, CompletedRequest]:
    """Build a unique request index and fail on duplicate IDs."""

    if not isinstance(system_name, str) or not system_name:
        raise ComparisonMetricError(
            "system_name must be a non-empty string")
    indexed: dict[RequestKey, CompletedRequest] = {}
    for request in requests:
        if not isinstance(request, CompletedRequest):
            raise ComparisonMetricError(
                f"{system_name} contains a non-CompletedRequest value")
        if request.key in indexed:
            raise ComparisonMetricError(
                f"{system_name} contains duplicate request ID {request.key}")
        indexed[request.key] = request
    return indexed


def summarize_completed_requests(
        requests: Sequence[CompletedRequest], *,
        offered_session_rate: float,
        thresholds: SLOThresholds = DEFAULT_SLO_THRESHOLDS,
) -> RequestCohortSummary:
    """Summarize a fully drained measurement cohort."""

    indexed = index_completed_requests(requests)
    if not indexed:
        raise ComparisonMetricError(
            "request cohort must contain at least one completed request")
    ordered = tuple(indexed[key] for key in sorted(indexed))
    session_count = len({request.key.session_id for request in ordered})
    rate = _require_finite(
        "offered_session_rate", offered_session_rate)
    if rate < 0.0:
        raise ComparisonMetricError(
            "offered_session_rate must be non-negative")
    first = tuple(request for request in ordered if not request.is_resume)
    resume = tuple(request for request in ordered if request.is_resume)
    return RequestCohortSummary(
        offered_session_rate=rate,
        offered_session_count=session_count,
        request_count=len(ordered),
        output_tokens=sum(request.output_tokens for request in ordered),
        thresholds=thresholds,
        all_requests=_summarize_kind(
            "all", ordered,
            offered_session_rate=rate,
            offered_session_count=session_count,
            thresholds=thresholds,
        ),
        first=_summarize_kind(
            "first", first,
            offered_session_rate=rate,
            offered_session_count=session_count,
            thresholds=thresholds,
        ),
        resume=_summarize_kind(
            "resume", resume,
            offered_session_rate=rate,
            offered_session_count=session_count,
            thresholds=thresholds,
        ),
    )


def _render_keys(keys: set[RequestKey], limit: int = 5) -> str:
    ordered = sorted(keys)
    rendered = ", ".join(
        f"{key.session_id}:{key.sub_request_index}"
        for key in ordered[:limit])
    if len(ordered) > limit:
        rendered += f", ... ({len(ordered)} total)"
    return rendered or "none"


def validate_full_drain_same_request_ids(
        requests_by_system: Mapping[str, Sequence[CompletedRequest]],
        *,
        expected_request_ids: Optional[Sequence[RequestKey]] = None,
) -> tuple[RequestKey, ...]:
    """Validate exact full-drain ID equality across compared systems.

    Completion order and release timestamps may differ between systems.  The
    latter is expected for closed-loop sessions.  Output-token counts are
    immutable workload data and therefore must match for every paired ID.
    """

    if not requests_by_system:
        raise ComparisonMetricError(
            "full-drain validation requires at least one system")
    names = sorted(requests_by_system)
    indexes = {
        name: index_completed_requests(
            requests_by_system[name], system_name=name)
        for name in names
    }
    reference_name = names[0]
    reference = indexes[reference_name]
    if expected_request_ids is None:
        if not reference:
            raise ComparisonMetricError(
                f"{reference_name} has an empty completed-request cohort")
        reference_ids = set(reference)
        systems_to_check = names[1:]
        comparison_name = reference_name
    else:
        expected = tuple(expected_request_ids)
        if not expected:
            raise ComparisonMetricError(
                "expected_request_ids must not be empty")
        if any(not isinstance(key, RequestKey) for key in expected):
            raise ComparisonMetricError(
                "expected_request_ids must contain only RequestKey values")
        reference_ids = set(expected)
        if len(reference_ids) != len(expected):
            raise ComparisonMetricError(
                "expected_request_ids contains duplicate IDs")
        systems_to_check = names
        comparison_name = "expected request roster"
    for name in systems_to_check:
        candidate = indexes[name]
        candidate_ids = set(candidate)
        missing = reference_ids - candidate_ids
        extra = candidate_ids - reference_ids
        if missing or extra:
            raise ComparisonMetricError(
                f"{name} is not a full-drain match for {comparison_name}: "
                f"missing={_render_keys(missing)}; "
                f"extra={_render_keys(extra)}")
    for name in names[1:]:
        candidate = indexes[name]
        for key in sorted(reference_ids):
            if candidate[key].output_tokens != reference[key].output_tokens:
                raise ComparisonMetricError(
                    f"{name} output_tokens mismatch for {key}: "
                    f"{candidate[key].output_tokens} != "
                    f"{reference[key].output_tokens}")
    return tuple(sorted(reference_ids))


@dataclass(frozen=True)
class SeedAggregate:
    seed_ids: tuple[int | str, ...]
    values: tuple[float, ...]
    mean: float
    sample_stddev: Optional[float]
    ci95_half_width: Optional[float]
    ci95_lower: Optional[float]
    ci95_upper: Optional[float]
    ci_method: str


_STUDENT_T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _ordered_seed_ids(
        values_by_seed: Mapping[int | str, float]) -> tuple[int | str, ...]:
    if not values_by_seed:
        raise ComparisonMetricError(
            "seed aggregation requires at least one seed")
    for seed in values_by_seed:
        if (
            not isinstance(seed, (int, str))
            or isinstance(seed, bool)
            or isinstance(seed, str) and not seed
        ):
            raise ComparisonMetricError(
                f"seed IDs must be integers or non-empty strings: {seed!r}")
    return tuple(sorted(
        values_by_seed, key=lambda seed: (type(seed).__name__, repr(seed))))


def aggregate_seed_values(
        values_by_seed: Mapping[int | str, float]) -> SeedAggregate:
    """Aggregate independent seed-level values with a Student-t 95% CI."""

    seed_ids = _ordered_seed_ids(values_by_seed)
    values = tuple(
        _require_finite(
            f"value for seed {seed!r}", values_by_seed[seed])
        for seed in seed_ids)
    mean = math.fsum(values) / len(values)
    if len(values) == 1:
        return SeedAggregate(
            seed_ids=seed_ids,
            values=values,
            mean=mean,
            sample_stddev=None,
            ci95_half_width=None,
            ci95_lower=None,
            ci95_upper=None,
            ci_method="unavailable_single_seed",
        )
    variance = math.fsum(
        (value - mean) ** 2 for value in values) / (len(values) - 1)
    sample_stddev = math.sqrt(variance)
    degrees_of_freedom = len(values) - 1
    critical = _STUDENT_T_975.get(degrees_of_freedom, 1.96)
    half_width = critical * sample_stddev / math.sqrt(len(values))
    return SeedAggregate(
        seed_ids=seed_ids,
        values=values,
        mean=mean,
        sample_stddev=sample_stddev,
        ci95_half_width=half_width,
        ci95_lower=mean - half_width,
        ci95_upper=mean + half_width,
        ci_method="student_t_95",
    )


@dataclass(frozen=True)
class PairedSeedAggregate:
    seed_ids: tuple[int | str, ...]
    reference: SeedAggregate
    candidate: SeedAggregate
    candidate_minus_reference: SeedAggregate
    candidate_over_reference: Optional[SeedAggregate]
    ratio_unavailable_reason: Optional[str]


def aggregate_paired_seed_values(
        reference_by_seed: Mapping[int | str, float],
        candidate_by_seed: Mapping[int | str, float],
) -> PairedSeedAggregate:
    """Pair by seed before aggregating differences and ratios."""

    reference_seeds = set(reference_by_seed)
    candidate_seeds = set(candidate_by_seed)
    if reference_seeds != candidate_seeds:
        missing = reference_seeds - candidate_seeds
        extra = candidate_seeds - reference_seeds
        raise ComparisonMetricError(
            "paired seed sets differ: "
            f"candidate_missing={sorted(missing, key=repr)!r}; "
            f"candidate_extra={sorted(extra, key=repr)!r}")
    seed_ids = _ordered_seed_ids(reference_by_seed)
    reference = {
        seed: _require_finite(
            f"reference value for seed {seed!r}",
            reference_by_seed[seed])
        for seed in seed_ids
    }
    candidate = {
        seed: _require_finite(
            f"candidate value for seed {seed!r}",
            candidate_by_seed[seed])
        for seed in seed_ids
    }
    differences = {
        seed: candidate[seed] - reference[seed] for seed in seed_ids}
    ratio: Optional[SeedAggregate]
    reason: Optional[str]
    zero_denominators = tuple(
        seed for seed in seed_ids if reference[seed] == 0.0)
    if zero_denominators:
        ratio = None
        reason = (
            "reference is zero for seeds "
            + ", ".join(repr(seed) for seed in zero_denominators)
        )
    else:
        ratio = aggregate_seed_values({
            seed: candidate[seed] / reference[seed] for seed in seed_ids
        })
        reason = None
    return PairedSeedAggregate(
        seed_ids=seed_ids,
        reference=aggregate_seed_values(reference),
        candidate=aggregate_seed_values(candidate),
        candidate_minus_reference=aggregate_seed_values(differences),
        candidate_over_reference=ratio,
        ratio_unavailable_reason=reason,
    )

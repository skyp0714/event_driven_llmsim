"""Fixed-window metrics for stationary TraceLab comparison cells.

The finite comparison path measures a preregistered identity roster after a
full drain.  A steady-state experiment needs different semantics: membership
is determined by the request's *causal release time*, successors may therefore
differ across systems, and an overloaded cell must stop at a fixed loaded
cutoff instead of draining indefinitely.

This module is deliberately pure.  It consumes an immutable cutoff snapshot
and does not know which serving system produced it.  The same function can be
run on a later full-drain snapshot because all timestamps after the cutoff are
masked.  That property is used to prove cutoff/full-drain metric equivalence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Optional, Sequence

from .hbf_comparison_metrics import (
    CompletedRequest,
    DEFAULT_SLO_THRESHOLDS,
    RequestKey,
    SLOThresholds,
    joint_slo_pass,
    summarize_distribution,
)


NANOSECONDS_PER_SECOND = 1_000_000_000

DEFAULT_WARMUP_U1_START_NS = 900 * NANOSECONDS_PER_SECOND
DEFAULT_WARMUP_U1_END_NS = 1_200 * NANOSECONDS_PER_SECOND
DEFAULT_MEASUREMENT_START_NS = 1_500 * NANOSECONDS_PER_SECOND
DEFAULT_MEASUREMENT_MID_NS = 1_800 * NANOSECONDS_PER_SECOND
DEFAULT_MEASUREMENT_END_NS = 2_100 * NANOSECONDS_PER_SECOND
DEFAULT_CUTOFF_NS = 2_811 * NANOSECONDS_PER_SECOND
DEFAULT_MAX_JOINT_PASS_LATENCY_NS = 710_400_000_000

MAX_COMPLETE_CASE_CENSOR_FRACTION = 0.05
MIN_MEASUREMENT_FIRST_RELEASES = 200
MIN_MEASUREMENT_RELEASES_PER_CALL_INDEX = 150
MIN_MEASUREMENT_TOTAL_RELEASES = 500
MIN_SUBWINDOW_RELEASES = 200
MIN_SUBWINDOW_COMPLETIONS = 200


class StationaryMetricError(ValueError):
    """Raised when a cutoff snapshot or window contract is inconsistent."""


def _integer(
        name: str,
        value: object,
        *,
        minimum: int = 0,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise StationaryMetricError(
            f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _closed_unit_interval(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise StationaryMetricError(
            f"{name} must be finite and in [0, 1], got {value!r}")
    return float(value)


@dataclass(frozen=True)
class StationaryWindowContract:
    """Preregistered loaded time windows and the exact censoring bound."""

    warmup_u1_start_ns: int = DEFAULT_WARMUP_U1_START_NS
    warmup_u1_end_ns: int = DEFAULT_WARMUP_U1_END_NS
    measurement_start_ns: int = DEFAULT_MEASUREMENT_START_NS
    measurement_mid_ns: int = DEFAULT_MEASUREMENT_MID_NS
    measurement_end_ns: int = DEFAULT_MEASUREMENT_END_NS
    cutoff_ns: int = DEFAULT_CUTOFF_NS
    max_joint_pass_latency_ns: int = DEFAULT_MAX_JOINT_PASS_LATENCY_NS

    def __post_init__(self) -> None:
        for name in (
            "warmup_u1_start_ns",
            "warmup_u1_end_ns",
            "measurement_start_ns",
            "measurement_mid_ns",
            "measurement_end_ns",
            "cutoff_ns",
            "max_joint_pass_latency_ns",
        ):
            _integer(name, getattr(self, name), minimum=1)
        boundaries = (
            self.warmup_u1_start_ns,
            self.warmup_u1_end_ns,
            self.measurement_start_ns,
            self.measurement_mid_ns,
            self.measurement_end_ns,
            self.cutoff_ns,
        )
        if any(
                left >= right
                for left, right in zip(boundaries, boundaries[1:])
        ):
            raise StationaryMetricError(
                "stationary window boundaries must be strictly increasing")
        warmup_width = (
            self.warmup_u1_end_ns - self.warmup_u1_start_ns)
        warmup_u2_width = (
            self.measurement_start_ns - self.warmup_u1_end_ns)
        measurement_m1_width = (
            self.measurement_mid_ns - self.measurement_start_ns)
        measurement_m2_width = (
            self.measurement_end_ns - self.measurement_mid_ns)
        if len({
            warmup_width,
            warmup_u2_width,
            measurement_m1_width,
            measurement_m2_width,
        }) != 1:
            raise StationaryMetricError(
                "U1, U2, M1, and M2 must have the same duration")
        if self.guard_duration_ns <= self.max_joint_pass_latency_ns:
            raise StationaryMetricError(
                "loaded guard must be strictly longer than the maximum "
                "joint-SLO passing latency")

    @property
    def measurement_duration_ns(self) -> int:
        return self.measurement_end_ns - self.measurement_start_ns

    @property
    def measurement_duration_seconds(self) -> float:
        return self.measurement_duration_ns / NANOSECONDS_PER_SECOND

    @property
    def guard_duration_ns(self) -> int:
        return self.cutoff_ns - self.measurement_end_ns

    @property
    def intervals(self) -> Mapping[str, tuple[int, int]]:
        return {
            "U1": (
                self.warmup_u1_start_ns,
                self.warmup_u1_end_ns,
            ),
            "U2": (
                self.warmup_u1_end_ns,
                self.measurement_start_ns,
            ),
            "M1": (
                self.measurement_start_ns,
                self.measurement_mid_ns,
            ),
            "M2": (
                self.measurement_mid_ns,
                self.measurement_end_ns,
            ),
        }


@dataclass(frozen=True)
class CutoffRequestObservation:
    """One scheduled call as observed at, or after, the loaded cutoff."""

    key: RequestKey
    output_tokens: int
    release_ns: Optional[int] = None
    first_token_ns: Optional[int] = None
    completion_ns: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, RequestKey):
            raise StationaryMetricError("key must be a RequestKey")
        _integer("output_tokens", self.output_tokens, minimum=1)
        for name in ("release_ns", "first_token_ns", "completion_ns"):
            value = getattr(self, name)
            if value is not None:
                _integer(name, value)
        if self.release_ns is None:
            if self.first_token_ns is not None or self.completion_ns is not None:
                raise StationaryMetricError(
                    "an unreleased request cannot have token timestamps")
            return
        if (
            self.first_token_ns is not None
            and self.first_token_ns < self.release_ns
        ):
            raise StationaryMetricError(
                "first token precedes request release")
        if self.completion_ns is not None:
            if self.first_token_ns is None:
                raise StationaryMetricError(
                    "completion requires a first-token timestamp")
            if self.completion_ns < self.first_token_ns:
                raise StationaryMetricError(
                    "completion precedes the first token")
            if (
                self.output_tokens == 1
                and self.completion_ns != self.first_token_ns
            ):
                raise StationaryMetricError(
                    "one-token completion must equal its first token")

    def released_in(self, start_ns: int, end_ns: int) -> bool:
        return (
            self.release_ns is not None
            and start_ns <= self.release_ns < end_ns
        )

    def completed_in(self, start_ns: int, end_ns: int) -> bool:
        return (
            self.completion_ns is not None
            and start_ns <= self.completion_ns < end_ns
        )

    def completed_by(self, cutoff_ns: int) -> bool:
        return (
            self.completion_ns is not None
            and self.completion_ns <= cutoff_ns
        )

    def first_token_by(self, cutoff_ns: int) -> bool:
        return (
            self.first_token_ns is not None
            and self.first_token_ns <= cutoff_ns
        )

    def as_completed_request(self) -> CompletedRequest:
        if (
            self.release_ns is None
            or self.first_token_ns is None
            or self.completion_ns is None
        ):
            raise StationaryMetricError(
                "request is not a completed observation")
        return CompletedRequest(
            key=self.key,
            release_ns=self.release_ns,
            first_token_ns=self.first_token_ns,
            completion_ns=self.completion_ns,
            output_tokens=self.output_tokens,
        )


def _index_observations(
        observations: Sequence[CutoffRequestObservation],
        *,
        expected_calls_per_session: int,
        cutoff_ns: int,
) -> tuple[
    tuple[CutoffRequestObservation, ...],
    Mapping[str, tuple[CutoffRequestObservation, ...]],
]:
    if isinstance(observations, (str, bytes)):
        raise StationaryMetricError(
            "observations must be a request-observation sequence")
    _integer(
        "expected_calls_per_session",
        expected_calls_per_session,
        minimum=1,
    )
    by_key = {}
    by_session: dict[str, list[CutoffRequestObservation]] = {}
    for index, observation in enumerate(observations):
        if not isinstance(observation, CutoffRequestObservation):
            raise StationaryMetricError(
                f"observations[{index}] is not a CutoffRequestObservation")
        if observation.key in by_key:
            raise StationaryMetricError(
                f"duplicate request observation {observation.key!r}")
        by_key[observation.key] = observation
        by_session.setdefault(
            observation.key.session_id, []).append(observation)
    if not by_key:
        raise StationaryMetricError("request observations cannot be empty")

    normalized_sessions = {}
    for session_id, values in by_session.items():
        ordered = tuple(sorted(
            values, key=lambda item: item.key.sub_request_index))
        indices = tuple(
            item.key.sub_request_index for item in ordered)
        if indices != tuple(range(expected_calls_per_session)):
            raise StationaryMetricError(
                f"session {session_id!r} does not contain exactly "
                f"{expected_calls_per_session} contiguous calls")
        first = ordered[0]
        if first.release_ns is None or first.release_ns >= cutoff_ns:
            raise StationaryMetricError(
                f"session {session_id!r} lacks an included first release")
        normalized_sessions[session_id] = ordered
    return tuple(by_key[key] for key in sorted(by_key)), normalized_sessions


def _count_released(
        observations: Sequence[CutoffRequestObservation],
        start_ns: int,
        end_ns: int,
) -> int:
    return sum(
        observation.released_in(start_ns, end_ns)
        for observation in observations
    )


def _count_completed(
        observations: Sequence[CutoffRequestObservation],
        start_ns: int,
        end_ns: int,
) -> int:
    return sum(
        observation.completed_in(start_ns, end_ns)
        for observation in observations
    )


def _backlog_at(
        observations: Sequence[CutoffRequestObservation],
        timestamp_ns: int,
) -> int:
    return sum(
        observation.release_ns is not None
        and observation.release_ns <= timestamp_ns
        and (
            observation.completion_ns is None
            or observation.completion_ns > timestamp_ns
        )
        for observation in observations
    )


def _active_sessions_at(
        sessions: Mapping[str, Sequence[CutoffRequestObservation]],
        timestamp_ns: int,
) -> int:
    return sum(
        values[0].release_ns is not None
        and values[0].release_ns <= timestamp_ns
        and (
            values[-1].completion_ns is None
            or values[-1].completion_ns > timestamp_ns
        )
        for values in sessions.values()
    )


def _relative_difference(left: int, right: int) -> Optional[float]:
    total = left + right
    if total == 0:
        return None
    return 2.0 * abs(left - right) / total


def _latency_summary(
        observations: Sequence[CutoffRequestObservation],
        *,
        cutoff_ns: int,
        max_censor_fraction: float,
) -> Mapping[str, object]:
    released = tuple(
        observation for observation in observations
        if observation.release_ns is not None
    )
    ttft_observed = tuple(
        observation for observation in released
        if observation.first_token_by(cutoff_ns)
    )
    tpot_eligible = tuple(
        observation for observation in released
        if observation.output_tokens > 1
    )
    tpot_observed = tuple(
        observation for observation in tpot_eligible
        if observation.completed_by(cutoff_ns)
    )
    ttft_values = tuple(
        float(observation.first_token_ns - observation.release_ns)
        for observation in ttft_observed
    )
    tpot_values = tuple(
        float(
            (observation.completion_ns - observation.first_token_ns)
            / (observation.output_tokens - 1)
        )
        for observation in tpot_observed
    )
    ttft_censored = len(released) - len(ttft_observed)
    tpot_censored = len(tpot_eligible) - len(tpot_observed)
    ttft_fraction = (
        ttft_censored / len(released) if released else 0.0)
    tpot_fraction = (
        tpot_censored / len(tpot_eligible) if tpot_eligible else 0.0)
    return {
        "released_count": len(released),
        "ttft": {
            "observed_count": len(ttft_observed),
            "censored_count": ttft_censored,
            "censor_fraction": ttft_fraction,
            "complete_case_distribution_ns": asdict(
                summarize_distribution(ttft_values)),
            "p95_publishable": ttft_fraction <= max_censor_fraction,
        },
        "tpot": {
            "eligible_count": len(tpot_eligible),
            "observed_count": len(tpot_observed),
            "censored_count": tpot_censored,
            "censor_fraction": tpot_fraction,
            "complete_case_distribution_ns": asdict(
                summarize_distribution(tpot_values)),
            "p95_publishable": tpot_fraction <= max_censor_fraction,
        },
        "publication_semantics": (
            "complete-case latency distribution with explicit censor "
            "coverage; p95 is not publishable when censor_fraction > "
            f"{max_censor_fraction}"
        ),
    }


def summarize_stationary_cutoff(
        observations: Sequence[CutoffRequestObservation],
        *,
        window: StationaryWindowContract = StationaryWindowContract(),
        thresholds: SLOThresholds = DEFAULT_SLO_THRESHOLDS,
        expected_calls_per_session: int = 3,
        max_complete_case_censor_fraction: float = (
            MAX_COMPLETE_CASE_CENSOR_FRACTION),
) -> Mapping[str, object]:
    """Return release-window SLO goodput and stationarity seed statistics."""

    if not isinstance(window, StationaryWindowContract):
        raise StationaryMetricError(
            "window must be a StationaryWindowContract")
    if not isinstance(thresholds, SLOThresholds):
        raise StationaryMetricError("thresholds must be SLOThresholds")
    censor_limit = _closed_unit_interval(
        "max_complete_case_censor_fraction",
        max_complete_case_censor_fraction,
    )
    ordered, sessions = _index_observations(
        observations,
        expected_calls_per_session=expected_calls_per_session,
        cutoff_ns=window.cutoff_ns,
    )
    measurement = tuple(
        observation for observation in ordered
        if observation.released_in(
            window.measurement_start_ns,
            window.measurement_end_ns,
        )
    )
    if not measurement:
        raise StationaryMetricError(
            "measurement release window contains no requests")

    completed_by_cutoff = tuple(
        observation for observation in measurement
        if observation.completed_by(window.cutoff_ns)
    )
    passed = tuple(
        observation
        for observation in completed_by_cutoff
        if joint_slo_pass(
            observation.as_completed_request(), thresholds)
    )
    incomplete = tuple(
        observation for observation in measurement
        if not observation.completed_by(window.cutoff_ns)
    )
    exact_failed_censors = []
    ambiguous_censors = []
    for observation in incomplete:
        assert observation.release_ns is not None
        ttft_limit = (
            thresholds.resume_ttft_ns
            if observation.key.is_resume
            else thresholds.first_ttft_ns
        )
        latest_passing_completion_ns = (
            observation.release_ns
            + ttft_limit
            + (observation.output_tokens - 1) * thresholds.tpot_ns
        )
        if latest_passing_completion_ns <= window.cutoff_ns:
            exact_failed_censors.append(observation)
        else:
            ambiguous_censors.append(observation)
    if ambiguous_censors:
        raise StationaryMetricError(
            "loaded guard does not decide every measurement request's SLO")

    duration_seconds = window.measurement_duration_seconds
    pass_output_tokens = sum(
        observation.output_tokens for observation in passed)
    completion_window = tuple(
        observation for observation in ordered
        if observation.completed_in(
            window.measurement_start_ns,
            window.measurement_end_ns,
        )
    )
    subwindows = {}
    for name, (start_ns, end_ns) in window.intervals.items():
        subwindows[name] = {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "released_calls": _count_released(
                ordered, start_ns, end_ns),
            "completed_calls": _count_completed(
                ordered, start_ns, end_ns),
        }

    backlog_start = _backlog_at(
        ordered, window.measurement_start_ns)
    backlog_end = _backlog_at(
        ordered, window.measurement_end_ns)
    active_start = _active_sessions_at(
        sessions, window.measurement_start_ns)
    active_end = _active_sessions_at(
        sessions, window.measurement_end_ns)
    release_by_call_index = {
        str(call_index): sum(
            observation.key.sub_request_index == call_index
            for observation in measurement
        )
        for call_index in range(expected_calls_per_session)
    }
    first_releases = release_by_call_index["0"]

    minimum_sample_violations = []
    if first_releases < MIN_MEASUREMENT_FIRST_RELEASES:
        minimum_sample_violations.append(
            "measurement_first_releases")
    for call_index in range(expected_calls_per_session):
        if release_by_call_index[str(call_index)] < (
                MIN_MEASUREMENT_RELEASES_PER_CALL_INDEX):
            minimum_sample_violations.append(
                f"measurement_call_{call_index}_releases")
    if len(measurement) < MIN_MEASUREMENT_TOTAL_RELEASES:
        minimum_sample_violations.append(
            "measurement_total_releases")
    for name, counts in subwindows.items():
        if counts["released_calls"] < MIN_SUBWINDOW_RELEASES:
            minimum_sample_violations.append(
                f"{name}_released_calls")
        if counts["completed_calls"] < MIN_SUBWINDOW_COMPLETIONS:
            minimum_sample_violations.append(
                f"{name}_completed_calls")

    first_measurement = tuple(
        observation for observation in measurement
        if not observation.key.is_resume
    )
    resume_measurement = tuple(
        observation for observation in measurement
        if observation.key.is_resume
    )
    result = {
        "schema_version": 1,
        "window": {
            **asdict(window),
            "measurement_duration_seconds": duration_seconds,
            "loaded_guard": True,
            "membership": (
                "measurement_start_ns <= request.release_ns "
                "< measurement_end_ns"
            ),
        },
        "measurement": {
            "released_calls": len(measurement),
            "released_first_calls": len(first_measurement),
            "released_resume_calls": len(resume_measurement),
            "completed_by_cutoff": len(completed_by_cutoff),
            "incomplete_at_cutoff": len(incomplete),
            "exact_failed_censor_count": len(exact_failed_censors),
            "ambiguous_measurement_censor_count": 0,
            "joint_slo_pass_count": len(passed),
            "joint_slo_pass_output_tokens": pass_output_tokens,
            "joint_slo_pass_fraction": len(passed) / len(measurement),
            "joint_slo_request_goodput_per_second": (
                len(passed) / duration_seconds),
            "joint_slo_output_token_goodput_per_second": (
                pass_output_tokens / duration_seconds),
            "request_goodput_formula": (
                "joint_slo_pass_count / measurement_duration_seconds"
            ),
            "output_goodput_formula": (
                "joint_slo_pass_output_tokens "
                "/ measurement_duration_seconds"
            ),
        },
        "latency": {
            "first": _latency_summary(
                first_measurement,
                cutoff_ns=window.cutoff_ns,
                max_censor_fraction=censor_limit,
            ),
            "resume": _latency_summary(
                resume_measurement,
                cutoff_ns=window.cutoff_ns,
                max_censor_fraction=censor_limit,
            ),
        },
        "completion_window_throughput": {
            "completed_calls": len(completion_window),
            "completed_output_tokens": sum(
                observation.output_tokens
                for observation in completion_window
            ),
            "requests_per_second": (
                len(completion_window) / duration_seconds),
            "output_tokens_per_second": (
                sum(
                    observation.output_tokens
                    for observation in completion_window
                )
                / duration_seconds
            ),
            "membership": (
                "measurement_start_ns <= request.completion_ns "
                "< measurement_end_ns"
            ),
        },
        "stationarity_seed_statistics": {
            "subwindows": subwindows,
            "warmup_release_relative_difference": _relative_difference(
                subwindows["U1"]["released_calls"],
                subwindows["U2"]["released_calls"],
            ),
            "warmup_completion_relative_difference": _relative_difference(
                subwindows["U1"]["completed_calls"],
                subwindows["U2"]["completed_calls"],
            ),
            "measurement_release_relative_difference": (
                _relative_difference(
                    subwindows["M1"]["released_calls"],
                    subwindows["M2"]["released_calls"],
                )
            ),
            "measurement_completion_relative_difference": (
                _relative_difference(
                    subwindows["M1"]["completed_calls"],
                    subwindows["M2"]["completed_calls"],
                )
            ),
            "backlog_at_measurement_start": backlog_start,
            "backlog_at_measurement_end": backlog_end,
            "backlog_growth_fraction": (
                (backlog_end - backlog_start) / len(measurement)
            ),
            "active_sessions_at_measurement_start": active_start,
            "active_sessions_at_measurement_end": active_end,
            "active_session_growth_fraction": (
                (active_end - active_start) / first_releases
                if first_releases else None
            ),
            "measurement_release_count_by_call_index": (
                release_by_call_index),
            "call_1_to_call_0_release_ratio": (
                release_by_call_index.get("1", 0) / first_releases
                if first_releases and expected_calls_per_session > 1
                else None
            ),
            "call_2_to_call_0_release_ratio": (
                release_by_call_index.get("2", 0) / first_releases
                if first_releases and expected_calls_per_session > 2
                else None
            ),
            "minimum_sample_gate_pass": (
                not minimum_sample_violations),
            "minimum_sample_violations": (
                minimum_sample_violations),
        },
        "cutoff_audit": {
            "scheduled_calls": len(ordered),
            "unreleased_calls": sum(
                observation.release_ns is None
                or observation.release_ns > window.cutoff_ns
                for observation in ordered
            ),
            "released_live_calls": sum(
                observation.release_ns is not None
                and observation.release_ns <= window.cutoff_ns
                and not observation.completed_by(window.cutoff_ns)
                for observation in ordered
            ),
            "completed_calls": sum(
                observation.completed_by(window.cutoff_ns)
                for observation in ordered
            ),
            "post_cutoff_timestamps_are_masked": True,
        },
    }
    return result


__all__ = [
    "CutoffRequestObservation",
    "DEFAULT_CUTOFF_NS",
    "DEFAULT_MAX_JOINT_PASS_LATENCY_NS",
    "DEFAULT_MEASUREMENT_END_NS",
    "DEFAULT_MEASUREMENT_MID_NS",
    "DEFAULT_MEASUREMENT_START_NS",
    "DEFAULT_WARMUP_U1_END_NS",
    "DEFAULT_WARMUP_U1_START_NS",
    "MAX_COMPLETE_CASE_CENSOR_FRACTION",
    "StationaryMetricError",
    "StationaryWindowContract",
    "summarize_stationary_cutoff",
]

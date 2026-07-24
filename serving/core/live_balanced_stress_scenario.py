"""Fail-closed high-rate stress extension of the balanced TraceLab cohort.

This module deliberately does not change the finite 0.25--1.2 publication
profile in :mod:`live_balanced_storage_scenario`.  It reuses the same seven
complete, content-addressed TraceLab sessions while giving each stress rate
its own sufficiently long warmup and guard.  The static schedule gives each
measured successor a candidate active-offer window.  The strict live collector,
not this zero-service construction, decides whether the actual completion-plus-
tool release stayed inside that window.

The schedule is causal.  Offered order depends only on the scenario ID, seed,
role, role-local epoch index, and complete-session identity.  Future output
lengths, call indices, runtime completions, queue state, and placement state
are audit inputs or runtime outcomes, never scheduling inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

from .hbf_comparison_workload import (
    OfferedPlan,
    ScheduledSession,
    SessionSpec,
    TRACELAB_SCHEMA3_SHA256,
    build_offered_plan,
    stable_json_sha256,
)
from .live_balanced_storage_scenario import (
    BASELINE_USABLE_D_HBM_AND_CPU_BYTES_PER_NODE,
    EXPECTED_CALLS_PER_EPOCH,
    EXPECTED_FIRST_CALLS_PER_EPOCH,
    EXPECTED_MAX_RECORDED_GAP_NS,
    EXPECTED_RESUME_CALLS_PER_EPOCH,
    EXPECTED_SESSIONS_PER_EPOCH,
    EXPECTED_SOURCE_SESSION_IDS,
    HBF_TP4_USABLE_LOGICAL_KV_BYTES,
    HBF_TP8_CONTEXT_USABLE_LOGICAL_KV_BYTES,
    HBF_TP8_USABLE_LOGICAL_KV_BYTES,
    SELECTED_SOURCE_INDICES,
    _block_rounded_kv_bytes,
    _load_templates,
    _prefill_service_audit,
    _sha256_file,
    _validated_baseline_usable_bytes_per_node,
)


SCENARIO_SCHEMA_VERSION = 1
SCENARIO_ID = "tracelab-headline-1741-balanced-highrate-v2"

STRESS_RATES = (1.4, 1.6, 2.2, 2.8, 3.0)
STRESS_SEEDS = (101, 102, 103, 104, 105)
MAXIMUM_AUDITED_RATE = max(STRESS_RATES)
MEASUREMENT_EPOCH_COUNT = 16

# These are the requested lower bounds.  Exact seeded Poisson draws did not
# give both sides a full 5%-margin span at these counts.
REQUESTED_MINIMUM_WARMUP_GUARD_EPOCHS = {
    1.4: 490,
    1.6: 560,
    2.2: 769,
    2.8: 979,
    3.0: 1049,
}

# First counts at or above the requested lower bounds for which every seed
# has both a warmup and guard arrival span of at least max_gap * 1.05.
STRESS_WARMUP_GUARD_EPOCHS = {
    1.4: 497,
    1.6: 573,
    2.2: 784,
    2.8: 1008,
    3.0: 1078,
}

TOOL_GAP_MARGIN_NUMERATOR = 105
TOOL_GAP_MARGIN_DENOMINATOR = 100
REQUIRED_ARRIVAL_SPAN_NS = (
    EXPECTED_MAX_RECORDED_GAP_NS
    * TOOL_GAP_MARGIN_NUMERATOR
    // TOOL_GAP_MARGIN_DENOMINATOR
)

EXPECTED_MEASUREMENT_SESSION_COUNT = (
    MEASUREMENT_EPOCH_COUNT * EXPECTED_SESSIONS_PER_EPOCH
)
EXPECTED_MEASUREMENT_REQUEST_COUNT = (
    MEASUREMENT_EPOCH_COUNT * EXPECTED_CALLS_PER_EPOCH
)
EXPECTED_MEASUREMENT_FIRST_CALL_COUNT = (
    MEASUREMENT_EPOCH_COUNT * EXPECTED_FIRST_CALLS_PER_EPOCH
)
EXPECTED_MEASUREMENT_RESUME_CALL_COUNT = (
    MEASUREMENT_EPOCH_COUNT * EXPECTED_RESUME_CALLS_PER_EPOCH
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TIERED_CONFIG_RELATIVE = Path(
    "configs/agentic_kv/qwen3_1m_p4d4/tiered_fullprompt.json"
)
_TIERED_CONFIG_PATH = _REPO_ROOT / _TIERED_CONFIG_RELATIVE
_HBF_CONFIG_RELATIVE = Path(
    "configs/wakekv_hbf/full_model_8card_server.json"
)
_HBF_CONFIG_PATH = _REPO_ROOT / _HBF_CONFIG_RELATIVE
EXPECTED_SSD_CAPACITY_BYTES_PER_NODE = 30_720_000_000_000

# Canonical digest after all 25 rate/seed schedules are normalized.  This
# pins the full matrix, while source and configuration hashes pin its inputs.
EXPECTED_STRESS_SCHEDULE_MATRIX_SHA256 = (
    "aafc56f92e1c09ac3ad4e801963ed541e4c38044bd48f1acd35b4e0cefd1de78"
)

_ROLE_SOURCE_BASE = {
    "warmup": 0,
    "measurement": 1_000_000_000,
    "guard": 2_000_000_000,
}


class LiveBalancedStressScenarioError(ValueError):
    """Raised when the preregistered stress contract drifts."""


@dataclass(frozen=True)
class StressEpoch:
    role: str
    role_epoch_index: int
    sessions: tuple[SessionSpec, ...]


@dataclass(frozen=True)
class StressSeedAudit:
    seed: int
    offered_session_rate_per_second: float
    offered_session_count: int
    offered_request_count: int
    offered_first_call_count: int
    offered_resume_call_count: int
    offered_session_ids_sha256: str
    unit_draws_sha256: str
    scheduled_arrivals_sha256: str
    warmup_arrival_span_ns: int
    guard_arrival_span_ns: int
    required_arrival_span_ns: int
    last_external_guard_offer_ns: int
    latest_measurement_request_release_ns: int
    active_offered_tail_after_latest_measurement_release_ns: int
    measurement_session_count: int
    measurement_request_count: int
    measurement_first_call_count: int
    measurement_resume_call_count: int
    measurement_request_release_count: int
    measurement_request_releases_before_last_guard_offer: int
    measurement_resume_return_count: int
    measurement_resume_returns_before_last_guard_offer: int
    measurement_resume_returns_under_pre_ssd_node_pressure: int
    per_node_peak_recorded_gap_logical_kv_bytes: tuple[int, int]
    aggregate_peak_recorded_gap_logical_kv_bytes: int
    both_nodes_exceed_pre_ssd_capacity: bool
    fits_ssd_capacity_on_each_node: bool
    fits_hbf_tp4_capacity: bool
    fits_hbf_tp8_capacity: bool
    fits_hbf_tp8_context_capacity: bool


@dataclass(frozen=True)
class StressRateProfile:
    offered_session_rate_per_second: float
    requested_minimum_warmup_guard_epochs: int
    selected_warmup_guard_epochs: int
    measurement_epochs: int
    total_epochs: int
    offered_sessions: int
    offered_requests: int
    offered_first_calls: int
    offered_resume_calls: int
    minimum_warmup_arrival_span_ns: int
    minimum_guard_arrival_span_ns: int
    minimum_active_offered_tail_ns: int
    minimum_per_node_peak_recorded_gap_logical_kv_bytes: int
    maximum_aggregate_peak_recorded_gap_logical_kv_bytes: int
    seed_audits: tuple[StressSeedAudit, ...]


@dataclass(frozen=True)
class LiveBalancedStressManifest:
    schema_version: int
    scenario_id: str
    epoch_profile: str
    source_sha256: str
    selected_source_indices: tuple[int, ...]
    selected_source_session_ids: tuple[str, ...]
    complete_cohort_sha256: str
    source_identity_sha256: str
    measurement_epochs: int
    measurement_session_ids: tuple[str, ...]
    measurement_session_count: int
    measurement_request_count: int
    measurement_first_call_count: int
    measurement_resume_call_count: int
    prefill_service: object
    recommended_rates: tuple[float, ...]
    recommended_seeds: tuple[int, ...]
    maximum_audited_rate: float
    requested_minimum_warmup_guard_epochs: tuple[tuple[float, int], ...]
    selected_warmup_guard_epochs: tuple[tuple[float, int], ...]
    tool_gap_margin_numerator: int
    tool_gap_margin_denominator: int
    recorded_max_tool_gap_ns: int
    required_arrival_span_ns: int
    baseline_pre_ssd_capacity_bytes_per_node: int
    baseline_cluster_config_path: str
    baseline_cluster_config_sha256: str
    ssd_capacity_bytes_per_node: int
    tiered_config_path: str
    tiered_config_sha256: str
    hbf_tp4_usable_logical_kv_bytes: int
    hbf_tp8_usable_logical_kv_bytes: int
    hbf_tp8_context_usable_logical_kv_bytes: int
    hbf_config_path: str
    hbf_config_sha256: str
    schedule_matrix_sha256: str
    rate_profiles: tuple[StressRateProfile, ...]
    runtime_guard_validation_required: bool
    runtime_guard_expected_measurement_resume_count: int
    workload_semantics: str
    arrival_window_semantics: str
    capacity_semantics: str
    measurement_semantics: str
    causal_policy_semantics: str
    experiment_sequence_semantics: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StressRateBoundedOfferedPlan:
    """Choose the preregistered rate-specific finite schedule for one seed."""

    seed: int
    epochs_by_rate: Mapping[float, tuple[StressEpoch, ...]]

    def at_rate(
            self,
            sessions_per_second: float,
            *,
            start_time_ns: int = 0,
    ) -> tuple[ScheduledSession, ...]:
        rate = _validate_supported_rate(sessions_per_second)
        plan = _build_rate_offered_plan(
            self.epochs_by_rate[rate],
            seed=self.seed,
        )
        return plan.at_rate(rate, start_time_ns=start_time_ns)


@dataclass(frozen=True)
class LiveBalancedStressScenario:
    manifest: LiveBalancedStressManifest
    epochs_by_rate: Mapping[float, tuple[StressEpoch, ...]]

    def build_offered_plan(self, *, seed: int) -> StressRateBoundedOfferedPlan:
        seed = _validate_supported_seed(seed)
        return StressRateBoundedOfferedPlan(
            seed=seed,
            epochs_by_rate=self.epochs_by_rate,
        )

    def epochs_for_rate(
            self, sessions_per_second: float,
    ) -> tuple[StressEpoch, ...]:
        rate = _validate_supported_rate(sessions_per_second)
        return self.epochs_by_rate[rate]

    def runtime_guard_contract(
            self,
            *,
            seed: int,
            sessions_per_second: float,
    ) -> dict[str, object]:
        """Return the content-addressed live-arrival guard for one schedule."""

        seed = _validate_supported_seed(seed)
        rate = _validate_supported_rate(sessions_per_second)
        profile = next(
            profile for profile in self.manifest.rate_profiles
            if profile.offered_session_rate_per_second == rate
        )
        audit = next(
            audit for audit in profile.seed_audits
            if audit.seed == seed
        )
        return {
            "seed": seed,
            "offered_session_rate_per_second": rate,
            "last_external_guard_offer_ns": (
                audit.last_external_guard_offer_ns),
            "expected_measurement_resume_count": (
                self.manifest
                .runtime_guard_expected_measurement_resume_count),
        }


def _validate_supported_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if seed not in STRESS_SEEDS:
        raise ValueError(
            f"seed={seed} is unsupported; expected one of {STRESS_SEEDS}")
    return seed


def _validate_supported_rate(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError("sessions_per_second must be positive and finite")
    rate = float(value)
    if rate > MAXIMUM_AUDITED_RATE:
        raise ValueError(
            f"sessions_per_second={rate} exceeds the audited maximum "
            f"{MAXIMUM_AUDITED_RATE}")
    if rate not in STRESS_RATES:
        raise ValueError(
            f"sessions_per_second={rate} is unsupported; "
            f"expected one of {STRESS_RATES}")
    return rate


def _validated_ssd_capacity(
        path: Path = _TIERED_CONFIG_PATH,
) -> tuple[int, str]:
    try:
        payload = path.read_bytes()
        raw = json.loads(payload)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LiveBalancedStressScenarioError(
            f"unable to load tiered config {path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise LiveBalancedStressScenarioError(
            "tiered config must be a JSON object")
    expected = {
        "policy": "tiered",
        "ssd_capacity_gb": 3840,
        "ssd_num_devices": 8,
        "block_size": 16,
        "demotion_mode": "capacity-only",
        "swap_execution_mode": "async-pre-admission",
    }
    mismatches = {
        key: (raw.get(key), expected_value)
        for key, expected_value in expected.items()
        if raw.get(key) != expected_value
    }
    if mismatches:
        raise LiveBalancedStressScenarioError(
            f"tiered SSD capacity contract changed: {mismatches}")
    capacity = int(
        float(raw["ssd_capacity_gb"])
        * 1_000_000_000
        * int(raw["ssd_num_devices"])
    )
    if capacity != EXPECTED_SSD_CAPACITY_BYTES_PER_NODE:
        raise LiveBalancedStressScenarioError(
            "tiered SSD capacity changed: "
            f"observed={capacity}, "
            f"expected={EXPECTED_SSD_CAPACITY_BYTES_PER_NODE}")
    return capacity, hashlib.sha256(payload).hexdigest()


def _clone_session(
        source: SessionSpec,
        *,
        role: str,
        role_epoch_index: int,
        template_offset: int,
) -> SessionSpec:
    if role not in _ROLE_SOURCE_BASE:
        raise LiveBalancedStressScenarioError(
            f"unsupported stress role {role!r}")
    synthetic_source_index = (
        _ROLE_SOURCE_BASE[role]
        + role_epoch_index * EXPECTED_SESSIONS_PER_EPOCH
        + template_offset
    )
    session_id = (
        f"{SCENARIO_ID}::{role}-{role_epoch_index:04d}"
        f"::source-{source.source_index:04d}::{source.session_id}"
    )
    calls = tuple(
        replace(
            call,
            session_id=session_id,
            source_index=synthetic_source_index,
            call_index=call_index,
        )
        for call_index, call in enumerate(source.calls)
    )
    identity = stable_json_sha256({
        "scenario_id": SCENARIO_ID,
        "role": role,
        "role_epoch_index": role_epoch_index,
        "source_index": source.source_index,
        "source_session_id": source.session_id,
        "source_session_identity_sha256": (
            source.source_session_identity_sha256),
        "complete_session": True,
        "unmodified_call_count": len(calls),
    })
    return SessionSpec(
        source_index=synthetic_source_index,
        session_id=session_id,
        source_arrival_time_ns=source.source_arrival_time_ns,
        source_session_identity_sha256=identity,
        calls=calls,
    )


def _build_rate_epochs(
        templates: Sequence[SessionSpec],
        *,
        warmup_guard_epoch_count: int,
) -> tuple[StressEpoch, ...]:
    epochs = []
    for role, count in (
        ("warmup", warmup_guard_epoch_count),
        ("measurement", MEASUREMENT_EPOCH_COUNT),
        ("guard", warmup_guard_epoch_count),
    ):
        for role_epoch_index in range(count):
            sessions = tuple(
                _clone_session(
                    source,
                    role=role,
                    role_epoch_index=role_epoch_index,
                    template_offset=template_offset,
                )
                for template_offset, source in enumerate(templates)
            )
            epochs.append(StressEpoch(
                role=role,
                role_epoch_index=role_epoch_index,
                sessions=sessions,
            ))
    return tuple(epochs)


def _ordered_sessions_for_seed(
        epochs: Sequence[StressEpoch],
        *,
        seed: int,
) -> tuple[SessionSpec, ...]:
    ordered = []
    for epoch in epochs:
        shuffled = list(epoch.sessions)
        epoch_seed = int.from_bytes(
            hashlib.sha256(
                (
                    f"{SCENARIO_ID}:{seed}:{epoch.role}:"
                    f"{epoch.role_epoch_index}"
                ).encode("utf-8")
            ).digest()[:8],
            byteorder="big",
        )
        random.Random(epoch_seed).shuffle(shuffled)
        ordered.extend(shuffled)
    return tuple(ordered)


def _build_rate_offered_plan(
        epochs: Sequence[StressEpoch],
        *,
        seed: int,
) -> OfferedPlan:
    return build_offered_plan(
        _ordered_sessions_for_seed(epochs, seed=seed),
        seed=seed,
        shuffle=False,
    )


def _arrival_spans_for_count(
        *,
        rate: float,
        seed: int,
        warmup_guard_epoch_count: int,
) -> tuple[int, int]:
    session_count = (
        (2 * warmup_guard_epoch_count + MEASUREMENT_EPOCH_COUNT)
        * EXPECTED_SESSIONS_PER_EPOCH
    )
    rng = random.Random(seed)
    unit_arrivals = [0.0]
    for _ in range(1, session_count):
        unit_arrivals.append(
            unit_arrivals[-1] - math.log1p(-rng.random()))
    arrivals_ns = tuple(
        int(round(value * 1_000_000_000 / rate))
        for value in unit_arrivals
    )
    first_measurement_index = (
        warmup_guard_epoch_count * EXPECTED_SESSIONS_PER_EPOCH
    )
    last_measurement_index = (
        (
            warmup_guard_epoch_count + MEASUREMENT_EPOCH_COUNT
        )
        * EXPECTED_SESSIONS_PER_EPOCH
        - 1
    )
    return (
        arrivals_ns[first_measurement_index] - arrivals_ns[0],
        arrivals_ns[-1] - arrivals_ns[last_measurement_index],
    )


def _first_passing_epoch_count(rate: float) -> int:
    requested = REQUESTED_MINIMUM_WARMUP_GUARD_EPOCHS[rate]
    selected = STRESS_WARMUP_GUARD_EPOCHS[rate]
    for count in range(requested, selected + 1):
        if all(
            min(_arrival_spans_for_count(
                rate=rate,
                seed=seed,
                warmup_guard_epoch_count=count,
            )) >= REQUIRED_ARRIVAL_SPAN_NS
            for seed in STRESS_SEEDS
        ):
            return count
    raise LiveBalancedStressScenarioError(
        f"no passing warmup/guard count found for rate={rate}")


def _scheduled_arrivals_sha256(
        schedule: Sequence[ScheduledSession],
) -> str:
    return stable_json_sha256([
        {
            "offer_index": item.offer_index,
            "session_id": item.session.session_id,
            "arrival_time_ns": item.arrival_time_ns,
            "unit_interarrival": item.unit_interarrival.hex(),
            "unit_arrival_time": item.unit_arrival_time.hex(),
        }
        for item in schedule
    ])


def _seed_schedule_audit(
        *,
        rate: float,
        seed: int,
        epochs: Sequence[StressEpoch],
        measurement_session_ids: set[str],
        ssd_capacity_bytes_per_node: int,
) -> StressSeedAudit:
    plan = _build_rate_offered_plan(epochs, seed=seed)
    schedule = plan.at_rate(rate)
    warmup_count = STRESS_WARMUP_GUARD_EPOCHS[rate]
    first_measurement_index = (
        warmup_count * EXPECTED_SESSIONS_PER_EPOCH)
    last_measurement_index = (
        (warmup_count + MEASUREMENT_EPOCH_COUNT)
        * EXPECTED_SESSIONS_PER_EPOCH
        - 1
    )
    warmup_span = (
        schedule[first_measurement_index].arrival_time_ns
        - schedule[0].arrival_time_ns
    )
    guard_span = (
        schedule[-1].arrival_time_ns
        - schedule[last_measurement_index].arrival_time_ns
    )
    last_external_guard_offer_ns = schedule[-1].arrival_time_ns

    measurement_sessions = [
        item for item in schedule
        if item.session.session_id in measurement_session_ids
    ]
    measurement_calls = [
        call
        for item in measurement_sessions
        for call in item.session.calls
    ]
    offered_calls = [
        call
        for item in schedule
        for call in item.session.calls
    ]
    request_release_times = []
    resume_return_times = []
    events = []
    for item in schedule:
        node_id = item.offer_index % 2
        boundary_ns = item.arrival_time_ns
        is_measurement = (
            item.session.session_id in measurement_session_ids)
        if is_measurement:
            request_release_times.append(boundary_ns)
        for call in item.session.calls[:-1]:
            next_boundary_ns = boundary_ns + call.tool_duration_ns
            logical_bytes = _block_rounded_kv_bytes(
                call.input_tokens + call.output_tokens)
            if next_boundary_ns <= boundary_ns:
                raise LiveBalancedStressScenarioError(
                    "pinned stress cohort gained a non-positive "
                    "inter-turn gap")
            events.append((
                boundary_ns,
                1,
                logical_bytes,
                node_id,
                False,
            ))
            events.append((
                next_boundary_ns,
                0,
                -logical_bytes,
                node_id,
                is_measurement,
            ))
            boundary_ns = next_boundary_ns
            if is_measurement:
                request_release_times.append(boundary_ns)
                resume_return_times.append(boundary_ns)

    events.sort(key=lambda event: (event[0], event[1]))
    live = [0, 0]
    node_peak = [0, 0]
    aggregate_peak = 0
    measurement_pressure_returns = 0
    event_index = 0
    while event_index < len(events):
        next_index = event_index + 1
        while (
            next_index < len(events)
            and events[next_index][0] == events[event_index][0]
        ):
            next_index += 1
        same_time = events[event_index:next_index]

        # A return observes its own still-live dormant interval.  All ends at
        # this timestamp are then applied before starts at the same timestamp.
        for _, kind, _, node_id, is_measurement in same_time:
            if kind == 0 and is_measurement:
                measurement_pressure_returns += int(
                    live[node_id]
                    > BASELINE_USABLE_D_HBM_AND_CPU_BYTES_PER_NODE
                )
        for _, _, delta, node_id, _ in same_time:
            live[node_id] += delta
            if live[node_id] < 0:
                raise LiveBalancedStressScenarioError(
                    "finite dormant-KV accounting became negative")
            node_peak[node_id] = max(node_peak[node_id], live[node_id])
            aggregate_peak = max(aggregate_peak, sum(live))
        event_index = next_index
    if live != [0, 0]:
        raise LiveBalancedStressScenarioError(
            "finite dormant-KV accounting did not drain")

    latest_measurement_release = max(request_release_times)
    audit = StressSeedAudit(
        seed=seed,
        offered_session_rate_per_second=rate,
        offered_session_count=len(schedule),
        offered_request_count=len(offered_calls),
        offered_first_call_count=sum(
            call.is_first_turn for call in offered_calls),
        offered_resume_call_count=sum(
            call.is_resume for call in offered_calls),
        offered_session_ids_sha256=(
            plan.offered_session_ids_sha256),
        unit_draws_sha256=plan.unit_draws_sha256,
        scheduled_arrivals_sha256=(
            _scheduled_arrivals_sha256(schedule)),
        warmup_arrival_span_ns=warmup_span,
        guard_arrival_span_ns=guard_span,
        required_arrival_span_ns=REQUIRED_ARRIVAL_SPAN_NS,
        last_external_guard_offer_ns=last_external_guard_offer_ns,
        latest_measurement_request_release_ns=(
            latest_measurement_release),
        active_offered_tail_after_latest_measurement_release_ns=(
            last_external_guard_offer_ns - latest_measurement_release),
        measurement_session_count=len(measurement_sessions),
        measurement_request_count=len(measurement_calls),
        measurement_first_call_count=sum(
            call.is_first_turn for call in measurement_calls),
        measurement_resume_call_count=sum(
            call.is_resume for call in measurement_calls),
        measurement_request_release_count=len(request_release_times),
        measurement_request_releases_before_last_guard_offer=sum(
            timestamp <= last_external_guard_offer_ns
            for timestamp in request_release_times
        ),
        measurement_resume_return_count=len(resume_return_times),
        measurement_resume_returns_before_last_guard_offer=sum(
            timestamp <= last_external_guard_offer_ns
            for timestamp in resume_return_times
        ),
        measurement_resume_returns_under_pre_ssd_node_pressure=(
            measurement_pressure_returns),
        per_node_peak_recorded_gap_logical_kv_bytes=(
            node_peak[0], node_peak[1]),
        aggregate_peak_recorded_gap_logical_kv_bytes=aggregate_peak,
        both_nodes_exceed_pre_ssd_capacity=all(
            value > BASELINE_USABLE_D_HBM_AND_CPU_BYTES_PER_NODE
            for value in node_peak
        ),
        fits_ssd_capacity_on_each_node=all(
            value < ssd_capacity_bytes_per_node
            for value in node_peak
        ),
        fits_hbf_tp4_capacity=(
            aggregate_peak < HBF_TP4_USABLE_LOGICAL_KV_BYTES),
        fits_hbf_tp8_capacity=(
            aggregate_peak < HBF_TP8_USABLE_LOGICAL_KV_BYTES),
        fits_hbf_tp8_context_capacity=(
            aggregate_peak
            < HBF_TP8_CONTEXT_USABLE_LOGICAL_KV_BYTES),
    )
    expected_total_epochs = 2 * warmup_count + MEASUREMENT_EPOCH_COUNT
    expected_offered_sessions = (
        expected_total_epochs * EXPECTED_SESSIONS_PER_EPOCH)
    expected_offered_requests = (
        expected_total_epochs * EXPECTED_CALLS_PER_EPOCH)
    observed_counts = (
        audit.offered_session_count,
        audit.offered_request_count,
        audit.offered_first_call_count,
        audit.offered_resume_call_count,
        audit.measurement_session_count,
        audit.measurement_request_count,
        audit.measurement_first_call_count,
        audit.measurement_resume_call_count,
        audit.measurement_request_release_count,
        audit.measurement_request_releases_before_last_guard_offer,
        audit.measurement_resume_return_count,
        audit.measurement_resume_returns_before_last_guard_offer,
        audit.measurement_resume_returns_under_pre_ssd_node_pressure,
    )
    expected_counts = (
        expected_offered_sessions,
        expected_offered_requests,
        expected_offered_sessions,
        expected_offered_sessions,
        EXPECTED_MEASUREMENT_SESSION_COUNT,
        EXPECTED_MEASUREMENT_REQUEST_COUNT,
        EXPECTED_MEASUREMENT_FIRST_CALL_COUNT,
        EXPECTED_MEASUREMENT_RESUME_CALL_COUNT,
        EXPECTED_MEASUREMENT_REQUEST_COUNT,
        EXPECTED_MEASUREMENT_REQUEST_COUNT,
        EXPECTED_MEASUREMENT_RESUME_CALL_COUNT,
        EXPECTED_MEASUREMENT_RESUME_CALL_COUNT,
        EXPECTED_MEASUREMENT_RESUME_CALL_COUNT,
    )
    if observed_counts != expected_counts:
        raise LiveBalancedStressScenarioError(
            "stress measurement/full-release contract changed: "
            f"observed={observed_counts}, expected={expected_counts}")
    if (
        audit.warmup_arrival_span_ns < REQUIRED_ARRIVAL_SPAN_NS
        or audit.guard_arrival_span_ns < REQUIRED_ARRIVAL_SPAN_NS
        or audit.active_offered_tail_after_latest_measurement_release_ns <= 0
    ):
        raise LiveBalancedStressScenarioError(
            f"rate={rate}, seed={seed} no longer keeps measurement "
            "returns inside the guarded offered-arrival window")
    if not (
        audit.both_nodes_exceed_pre_ssd_capacity
        and audit.fits_ssd_capacity_on_each_node
        and audit.fits_hbf_tp4_capacity
        and audit.fits_hbf_tp8_capacity
        and audit.fits_hbf_tp8_context_capacity
    ):
        raise LiveBalancedStressScenarioError(
            f"rate={rate}, seed={seed} no longer separates pre-SSD "
            "pressure from finite SSD/HBF capacity")
    return audit


def build_high_rate_stress(
        trace_path: str | Path,
) -> LiveBalancedStressScenario:
    """Build and validate the 1.4--3.0 balanced high-rate stress matrix."""

    path = Path(trace_path).expanduser().resolve()
    if not path.is_file():
        raise LiveBalancedStressScenarioError(
            f"TraceLab source does not exist: {path}")
    templates, cohort_audit = _load_templates(path)
    prefill_service = _prefill_service_audit(templates)
    baseline_capacity, baseline_config_sha256 = (
        _validated_baseline_usable_bytes_per_node())
    ssd_capacity, tiered_config_sha256 = _validated_ssd_capacity()

    selected_counts = {
        rate: _first_passing_epoch_count(rate)
        for rate in STRESS_RATES
    }
    if selected_counts != STRESS_WARMUP_GUARD_EPOCHS:
        raise LiveBalancedStressScenarioError(
            "first passing warmup/guard counts changed: "
            f"observed={selected_counts}, "
            f"expected={STRESS_WARMUP_GUARD_EPOCHS}")

    epochs_by_rate = {
        rate: _build_rate_epochs(
            templates,
            warmup_guard_epoch_count=(
                STRESS_WARMUP_GUARD_EPOCHS[rate]),
        )
        for rate in STRESS_RATES
    }
    canonical_measurement_ids = tuple(
        session.session_id
        for epoch in epochs_by_rate[STRESS_RATES[0]]
        if epoch.role == "measurement"
        for session in epoch.sessions
    )
    if (
        len(canonical_measurement_ids)
        != EXPECTED_MEASUREMENT_SESSION_COUNT
        or len(canonical_measurement_ids)
        != len(set(canonical_measurement_ids))
    ):
        raise LiveBalancedStressScenarioError(
            "measurement session roster is invalid")
    for rate, epochs in epochs_by_rate.items():
        observed_ids = tuple(
            session.session_id
            for epoch in epochs
            if epoch.role == "measurement"
            for session in epoch.sessions
        )
        if observed_ids != canonical_measurement_ids:
            raise LiveBalancedStressScenarioError(
                f"measurement roster changed across rate={rate}")

    rate_profiles = []
    for rate in STRESS_RATES:
        epochs = epochs_by_rate[rate]
        seed_audits = tuple(
            _seed_schedule_audit(
                rate=rate,
                seed=seed,
                epochs=epochs,
                measurement_session_ids=set(
                    canonical_measurement_ids),
                ssd_capacity_bytes_per_node=ssd_capacity,
            )
            for seed in STRESS_SEEDS
        )
        count = STRESS_WARMUP_GUARD_EPOCHS[rate]
        total_epochs = 2 * count + MEASUREMENT_EPOCH_COUNT
        rate_profiles.append(StressRateProfile(
            offered_session_rate_per_second=rate,
            requested_minimum_warmup_guard_epochs=(
                REQUESTED_MINIMUM_WARMUP_GUARD_EPOCHS[rate]),
            selected_warmup_guard_epochs=count,
            measurement_epochs=MEASUREMENT_EPOCH_COUNT,
            total_epochs=total_epochs,
            offered_sessions=(
                total_epochs * EXPECTED_SESSIONS_PER_EPOCH),
            offered_requests=(
                total_epochs * EXPECTED_CALLS_PER_EPOCH),
            offered_first_calls=(
                total_epochs * EXPECTED_FIRST_CALLS_PER_EPOCH),
            offered_resume_calls=(
                total_epochs * EXPECTED_RESUME_CALLS_PER_EPOCH),
            minimum_warmup_arrival_span_ns=min(
                audit.warmup_arrival_span_ns
                for audit in seed_audits
            ),
            minimum_guard_arrival_span_ns=min(
                audit.guard_arrival_span_ns
                for audit in seed_audits
            ),
            minimum_active_offered_tail_ns=min(
                audit.active_offered_tail_after_latest_measurement_release_ns
                for audit in seed_audits
            ),
            minimum_per_node_peak_recorded_gap_logical_kv_bytes=min(
                min(audit.per_node_peak_recorded_gap_logical_kv_bytes)
                for audit in seed_audits
            ),
            maximum_aggregate_peak_recorded_gap_logical_kv_bytes=max(
                audit.aggregate_peak_recorded_gap_logical_kv_bytes
                for audit in seed_audits
            ),
            seed_audits=seed_audits,
        ))
    rate_profiles_tuple = tuple(rate_profiles)
    schedule_matrix_sha256 = stable_json_sha256([
        {
            "rate": profile.offered_session_rate_per_second,
            "warmup_guard_epochs": (
                profile.selected_warmup_guard_epochs),
            "seed": audit.seed,
            "offered_session_ids_sha256": (
                audit.offered_session_ids_sha256),
            "unit_draws_sha256": audit.unit_draws_sha256,
            "scheduled_arrivals_sha256": (
                audit.scheduled_arrivals_sha256),
        }
        for profile in rate_profiles_tuple
        for audit in profile.seed_audits
    ])
    if (
        EXPECTED_STRESS_SCHEDULE_MATRIX_SHA256
        and schedule_matrix_sha256
        != EXPECTED_STRESS_SCHEDULE_MATRIX_SHA256
    ):
        raise LiveBalancedStressScenarioError(
            "stress schedule matrix changed: "
            f"observed={schedule_matrix_sha256}, "
            f"expected={EXPECTED_STRESS_SCHEDULE_MATRIX_SHA256}")

    manifest = LiveBalancedStressManifest(
        schema_version=SCENARIO_SCHEMA_VERSION,
        scenario_id=SCENARIO_ID,
        epoch_profile="high_rate_stress_v2",
        source_sha256=TRACELAB_SCHEMA3_SHA256,
        selected_source_indices=SELECTED_SOURCE_INDICES,
        selected_source_session_ids=EXPECTED_SOURCE_SESSION_IDS,
        complete_cohort_sha256=cohort_audit.complete_cohort_sha256,
        source_identity_sha256=cohort_audit.source_identity_sha256,
        measurement_epochs=MEASUREMENT_EPOCH_COUNT,
        measurement_session_ids=canonical_measurement_ids,
        measurement_session_count=EXPECTED_MEASUREMENT_SESSION_COUNT,
        measurement_request_count=EXPECTED_MEASUREMENT_REQUEST_COUNT,
        measurement_first_call_count=(
            EXPECTED_MEASUREMENT_FIRST_CALL_COUNT),
        measurement_resume_call_count=(
            EXPECTED_MEASUREMENT_RESUME_CALL_COUNT),
        prefill_service=prefill_service,
        recommended_rates=STRESS_RATES,
        recommended_seeds=STRESS_SEEDS,
        maximum_audited_rate=MAXIMUM_AUDITED_RATE,
        requested_minimum_warmup_guard_epochs=tuple(
            (rate, REQUESTED_MINIMUM_WARMUP_GUARD_EPOCHS[rate])
            for rate in STRESS_RATES
        ),
        selected_warmup_guard_epochs=tuple(
            (rate, STRESS_WARMUP_GUARD_EPOCHS[rate])
            for rate in STRESS_RATES
        ),
        tool_gap_margin_numerator=TOOL_GAP_MARGIN_NUMERATOR,
        tool_gap_margin_denominator=TOOL_GAP_MARGIN_DENOMINATOR,
        recorded_max_tool_gap_ns=EXPECTED_MAX_RECORDED_GAP_NS,
        required_arrival_span_ns=REQUIRED_ARRIVAL_SPAN_NS,
        baseline_pre_ssd_capacity_bytes_per_node=baseline_capacity,
        baseline_cluster_config_path=(
            "configs/cluster/dual_node_qwen3_1m_pd_p4d4_h100.json"),
        baseline_cluster_config_sha256=baseline_config_sha256,
        ssd_capacity_bytes_per_node=ssd_capacity,
        tiered_config_path=_TIERED_CONFIG_RELATIVE.as_posix(),
        tiered_config_sha256=tiered_config_sha256,
        hbf_tp4_usable_logical_kv_bytes=(
            HBF_TP4_USABLE_LOGICAL_KV_BYTES),
        hbf_tp8_usable_logical_kv_bytes=(
            HBF_TP8_USABLE_LOGICAL_KV_BYTES),
        hbf_tp8_context_usable_logical_kv_bytes=(
            HBF_TP8_CONTEXT_USABLE_LOGICAL_KV_BYTES),
        hbf_config_path=_HBF_CONFIG_RELATIVE.as_posix(),
        hbf_config_sha256=_sha256_file(_HBF_CONFIG_PATH),
        schedule_matrix_sha256=schedule_matrix_sha256,
        rate_profiles=rate_profiles_tuple,
        runtime_guard_validation_required=True,
        runtime_guard_expected_measurement_resume_count=(
            EXPECTED_MEASUREMENT_RESUME_CALL_COUNT),
        workload_semantics=(
            "the unchanged seven-session publication roster is repeated as "
            "complete epochs: source 1741 retains all eight calls and six "
            "natural one-call sessions remain complete. Inputs, outputs, "
            "cached-prefix coordinates, and tool gaps are never rewritten"),
        arrival_window_semantics=(
            "each rate has the first warmup/guard epoch count at or above "
            "its requested lower bound whose exact seed-101--105 Poisson "
            "arrival spans both cover 105% of the 2329.224-second maximum "
            "tool gap. This is only a zero-service schedule witness. The "
            "runner pins the final external guard offer for each seed/rate, "
            "and the strict collector must verify all 112 native resume "
            "arrival timestamps against it"),
        capacity_semantics=(
            "an exact zero-service block-rounded recorded-gap sweep uses "
            "sticky offered-index round robin across two baseline nodes. "
            "Every measured resume observes its own node above usable "
            "D-HBM plus host DRAM, while each finite peak fits the configured "
            "node-local SSD and all three configured HBF logical capacities"),
        measurement_semantics=(
            "the rate-invariant measurement roster contains exactly 112 "
            "complete sessions and 224 requests: 112 first calls and 112 "
            "resumes. Static audits only screen the proposed guard. A valid "
            "live stress cell must observe every request exactly once, fully "
            "drain, and have the strict collector confirm that every actual "
            "resume arrival is no later than the pinned final guard offer"),
        causal_policy_semantics=(
            "session ordering uses only scenario ID, seed, role, role-local "
            "epoch index, and complete-session identity. Future output "
            "tokens and call indices are used only for post-schedule audit "
            "classification; runtime completion, queues, and placement never "
            "change offered order or routing"),
        experiment_sequence_semantics=(
            "screen all systems at seed 101 before spending the full matrix, "
            "then confirm every retained rate with seeds 101--105. This is a "
            "finite stress/observed-grid experiment, not an unbounded "
            "sustainable-throughput claim"),
    )
    return LiveBalancedStressScenario(
        manifest=manifest,
        epochs_by_rate=epochs_by_rate,
    )


def build(trace_path: str | Path) -> LiveBalancedStressScenario:
    """Custom scenario-factory entry point for the live ASTRA runner."""

    return build_high_rate_stress(trace_path)


__all__ = [
    "EXPECTED_MEASUREMENT_REQUEST_COUNT",
    "EXPECTED_MEASUREMENT_SESSION_COUNT",
    "LiveBalancedStressManifest",
    "LiveBalancedStressScenario",
    "LiveBalancedStressScenarioError",
    "MAXIMUM_AUDITED_RATE",
    "MEASUREMENT_EPOCH_COUNT",
    "REQUESTED_MINIMUM_WARMUP_GUARD_EPOCHS",
    "STRESS_RATES",
    "STRESS_SEEDS",
    "STRESS_WARMUP_GUARD_EPOCHS",
    "build",
    "build_high_rate_stress",
]

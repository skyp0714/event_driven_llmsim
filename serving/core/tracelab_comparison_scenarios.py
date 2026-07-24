"""Preregistered TraceLab scenarios for the GPU/HBF comparison.

The native 32-session TraceLab cohort is useful for lifecycle and long-context
stress, but it is not a steady-state workload: one finite burst contains long
tool gaps and very unequal numbers of calls per session.  This module exposes
three explicit, policy-independent scenarios:

* a finite, three-call causal-prefix replay whose first- and resume-prefill
  service are approximately balanced;
* a finite, trace-derived long-cold-context stress whose measured calls all
  reuse at least 100k tokens after their complete native causal prefixes; and
* the original complete 32-session cohort as a non-steady lifecycle
  sensitivity.

Only session starts are placed on the Poisson arrival process.  Successor calls
remain inside :class:`SessionSpec`, so a simulator must release call ``N + 1``
after call ``N`` completes and its recorded tool gap elapses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from typing import Mapping, Sequence

from .hbf_comparison_workload import (
    CohortSummary,
    ComparisonWorkload,
    FIXED_SOURCE_INDICES,
    OfferedPlan,
    ScheduledSession,
    SessionSpec,
    TRACELAB_SCHEMA3_SHA256,
    WorkloadValidationError,
    build_offered_plan,
    load_fixed_comparison_workload,
    stable_json_sha256,
    summarize_sessions,
)


SCENARIO_SCHEMA_VERSION = 1

BALANCED_CALLS_PER_SESSION = 3
BALANCED_EPOCH_COUNT = 8
BALANCED_WARMUP_EPOCHS = (0, 1)
BALANCED_MEASUREMENT_EPOCHS = (2, 3, 4, 5)
BALANCED_GUARD_EPOCHS = (6, 7)
BALANCED_DEFAULT_RATES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
BALANCED_MAX_RATE = 5.0

FULL_COHORT_ANCHOR_RATES = (1.0, 3.0, 5.0)

LONG_COLD_CACHED_PREFIX_THRESHOLD = 100_000
LONG_COLD_SUCCESSOR_CALLS = 2
LONG_COLD_EPOCH_COUNT = 32
LONG_COLD_WARMUP_EPOCHS = tuple(range(0, 8))
LONG_COLD_MEASUREMENT_EPOCHS = tuple(range(8, 24))
LONG_COLD_GUARD_EPOCHS = tuple(range(24, 32))
LONG_COLD_ANCHOR_RATES = (3.0, 5.0)
LONG_COLD_MAX_RATE = 5.0
LONG_COLD_SOURCE_INDICES = (1531, 1554, 2850, 3813, 4055)
LONG_COLD_TARGET_CALL_INDICES = (19, 14, 13, 19, 19)
LONG_COLD_END_CALL_INDICES = (21, 16, 15, 21, 21)

# Qwen3's logical BF16 KV and the finite P4D4 block/capacity values are pinned
# by the comparison configs.  The 32 epochs retain 1,707,048,173,568
# block-rounded logical KV bytes if
# every replica reaches its final retained context concurrently.  That bound
# exceeds the pinned two-server usable D-HBM plus CPU capacity
# (1,382,573,883,392 bytes) by 324,474,290,176 bytes.  Causal completion can
# reduce live residency, so this is a tier-pressure sensitivity rather than a
# guarantee that every rate and seed reaches SSD.
LONG_COLD_LOGICAL_KV_BYTES_PER_TOKEN = 98_304
LONG_COLD_KV_BLOCK_SIZE_TOKENS = 16
LONG_COLD_COMBINED_USABLE_D_HBM_BYTES = 358_573_883_392
LONG_COLD_COMBINED_CPU_BYTES = 1_024_000_000_000
LONG_COLD_BLOCK_ROUNDED_FINAL_KV_BYTES = 1_707_048_173_568
LONG_COLD_COMBINED_D_HBM_AND_CPU_BYTES = (
    LONG_COLD_COMBINED_USABLE_D_HBM_BYTES
    + LONG_COLD_COMBINED_CPU_BYTES
)
LONG_COLD_FINAL_KV_EXCESS_BYTES = 324_474_290_176

# Filled from the strictly validated schema-3 TraceLab release.  The hashes
# cover source selection/endpoints, every retained native call field, every
# epoch identity mapping, the measured identity roster, and the three token
# summaries.  Keeping each contract separate makes drift diagnosable.
PINNED_LONG_COLD_SELECTION_WINDOWS_SHA256 = (
    "de51d608038356247cf86bd0980d766c7aa1d2cee7f718da5be3a15629d8992c"
)
PINNED_LONG_COLD_RETAINED_SOURCE_CALLS_SHA256 = (
    "e207f550349c3d86b4f74711f7b007a7630bc8e21667abae00c2ef106a22804f"
)
PINNED_LONG_COLD_EPOCH_MAPPING_SHA256 = (
    "b34aaa1c07fc36094be598d1ffbbf7e9df22988b2293e43b4e26301d1d8dbcfd"
)
PINNED_LONG_COLD_MEASUREMENT_IDENTITIES_SHA256 = (
    "760bc2821b96d0a592846caf8f5257f218fe699b3d16941cdbb91c167c7bb9a4"
)
PINNED_LONG_COLD_BASE_STATS_SHA256 = (
    "6bf27ef1a473caa7010ef2f6187c26d364470678e694c5418933620f4316f10f"
)
PINNED_LONG_COLD_FULL_REPLAY_STATS_SHA256 = (
    "cd1e66a361e88e670ad56bc176ce6b7c3f5104e96b7eebe820150f9bfbddd17b"
)
PINNED_LONG_COLD_MEASUREMENT_STATS_SHA256 = (
    "3097097d4a73c390d49c210120ebbc7674879d434950b6d4f2891dd14a6472fa"
)

# The five fixed-cohort sessions not listed here have fewer than three calls.
# Keeping this tuple explicit makes a change in the pinned cohort fail closed.
BALANCED_SOURCE_INDICES = (
    25, 71, 165, 389, 442, 447, 864, 1395, 1472, 1490, 1531, 1554,
    1853, 2266, 2270, 2276, 2471, 2850, 3047, 3131, 3277, 3437, 3813,
    3961, 4055, 4090, 4177,
)

# Audited with P4D4LatencyModel's central band, one request per batch, no
# queueing, and the raw TraceLab prefix on the pinned schema-3 release.
PINNED_FIRST_PREFILL_SERVICE_NS = 15_726_011_216
PINNED_RESUME_PREFILL_SERVICE_NS = 13_973_869_829
PINNED_RESUME_TO_FIRST_SERVICE_RATIO = (
    PINNED_RESUME_PREFILL_SERVICE_NS
    / PINNED_FIRST_PREFILL_SERVICE_NS
)

ROLE_WARMUP = "warmup"
ROLE_MEASUREMENT = "measurement"
ROLE_GUARD = "guard"


@dataclass(frozen=True)
class TokenContextStats:
    """Turn-split token totals plus the existing cohort summary."""

    cohort: CohortSummary
    first_input_tokens: int
    resume_input_tokens: int
    first_output_tokens: int
    resume_output_tokens: int
    first_fresh_input_tokens: int
    resume_fresh_input_tokens: int
    max_cached_prefix_tokens: int
    max_fresh_input_tokens: int


@dataclass(frozen=True)
class EpochSessionMapping:
    """Lossless provenance for one synthetic epoch session."""

    epoch_index: int
    role: str
    base_ordinal: int
    synthetic_source_index: int
    synthetic_session_id: str
    synthetic_source_identity_sha256: str
    source_index: int
    source_session_id: str
    source_session_identity_sha256: str | None
    source_arrival_time_ns: int


@dataclass(frozen=True)
class LongColdSelectionWindow:
    """Pinned native causal prefix and measured long-cold call window."""

    source_index: int
    source_session_id: str
    source_session_identity_sha256: str | None
    source_arrival_time_ns: int
    source_call_count: int
    target_call_index: int
    end_call_index: int
    retained_call_count: int
    target_input_tokens: int
    target_cached_prefix_tokens: int
    target_output_tokens: int
    target_tool_duration_ns: int


@dataclass(frozen=True)
class IsolatedPrefillServiceAudit:
    """Pinned evidence for the approximately service-balanced selection."""

    first_service_ns: int
    resume_service_ns: int
    resume_to_first_ratio: float
    model: str
    topology: str
    latency_band: str
    method: str


@dataclass(frozen=True)
class ArrivalRateContract:
    """Arrival grid and interpretation shared by all compared systems."""

    rates: tuple[float, ...]
    maximum_rate: float
    enumerated_only: bool
    rate_unit: str
    process: str
    first_arrival_semantics: str
    offer_order_semantics: str

    def __post_init__(self) -> None:
        if not self.rates:
            raise ValueError("arrival rates cannot be empty")
        if (
            isinstance(self.maximum_rate, bool)
            or not isinstance(self.maximum_rate, (int, float))
            or not math.isfinite(float(self.maximum_rate))
            or self.maximum_rate <= 0
        ):
            raise ValueError("maximum_rate must be positive and finite")
        if not isinstance(self.enumerated_only, bool):
            raise ValueError("enumerated_only must be a boolean")
        previous = 0.0
        for rate in self.rates:
            if (
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate))
                or rate <= previous
                or rate > self.maximum_rate
            ):
                raise ValueError(
                    "arrival rates must be strictly increasing, positive, "
                    "finite, and no greater than maximum_rate"
                )
            previous = float(rate)
        for name in (
            "rate_unit",
            "process",
            "first_arrival_semantics",
            "offer_order_semantics",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(
                    self, name):
                raise ValueError(f"{name} must be a non-empty string")

    def validate_rate(self, sessions_per_second: float) -> float:
        if (
            isinstance(sessions_per_second, bool)
            or not isinstance(sessions_per_second, (int, float))
            or not math.isfinite(float(sessions_per_second))
            or float(sessions_per_second) <= 0
        ):
            raise ValueError("sessions_per_second must be positive and finite")
        rate = float(sessions_per_second)
        if rate > self.maximum_rate:
            raise ValueError(
                f"sessions_per_second={rate} exceeds the preregistered "
                f"maximum {self.maximum_rate}"
            )
        if self.enumerated_only and rate not in self.rates:
            raise ValueError(
                f"sessions_per_second={rate} is not one of the "
                f"preregistered anchor rates {self.rates}"
            )
        return rate


@dataclass(frozen=True)
class BalancedCausalPrefixManifest:
    """Content and metric-window contract for the eight-epoch replay."""

    schema_version: int
    scenario_id: str
    source_sha256: str
    source_session_count: int
    selected_source_indices: tuple[int, ...]
    selected_source_session_ids: tuple[str, ...]
    calls_per_session: int
    epoch_count: int
    warmup_epochs: tuple[int, ...]
    measurement_epochs: tuple[int, ...]
    guard_epochs: tuple[int, ...]
    epoch_mapping: tuple[EpochSessionMapping, ...]
    base_stats: TokenContextStats
    full_replay_stats: TokenContextStats
    measurement_stats: TokenContextStats
    measurement_session_ids: tuple[str, ...]
    measurement_request_identities: tuple[str, ...]
    measurement_first_request_identities: tuple[str, ...]
    measurement_resume_request_identities: tuple[str, ...]
    isolated_prefill_service: IsolatedPrefillServiceAudit | None
    arrival_contract: ArrivalRateContract
    selection_semantics: str
    workload_semantics: str
    metric_window_semantics: str
    successor_release_semantics: str
    equilibrium_workload: bool
    offered_load_normalization: str
    epoch_mapping_sha256: str
    measurement_request_identities_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FullCohortSensitivityManifest:
    """Explicit non-steady interpretation of the original 32 sessions."""

    schema_version: int
    scenario_id: str
    source_sha256: str
    source_session_count: int
    selected_source_indices: tuple[int, ...]
    selected_source_session_ids: tuple[str, ...]
    workload_stats: TokenContextStats
    measurement_session_ids: tuple[str, ...]
    measurement_request_identities: tuple[str, ...]
    arrival_contract: ArrivalRateContract
    workload_semantics: str
    metric_window_semantics: str
    successor_release_semantics: str
    equilibrium_workload: bool
    offered_load_normalization: str
    measurement_request_identities_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LongColdContextStressManifest:
    """Trace-derived, non-equilibrium long-cold-context sensitivity."""

    schema_version: int
    scenario_id: str
    source_sha256: str
    source_session_count: int
    cached_prefix_threshold: int
    successor_call_count: int
    selected_source_indices: tuple[int, ...]
    selected_source_session_ids: tuple[str, ...]
    selection_windows: tuple[LongColdSelectionWindow, ...]
    epoch_count: int
    warmup_epochs: tuple[int, ...]
    measurement_epochs: tuple[int, ...]
    guard_epochs: tuple[int, ...]
    epoch_mapping: tuple[EpochSessionMapping, ...]
    base_prefix_stats: TokenContextStats
    full_replay_stats: TokenContextStats
    measurement_stats: TokenContextStats
    measurement_session_ids: tuple[str, ...]
    measurement_request_identities: tuple[str, ...]
    measurement_first_request_identities: tuple[str, ...]
    measurement_resume_request_identities: tuple[str, ...]
    arrival_contract: ArrivalRateContract
    selection_semantics: str
    workload_semantics: str
    metric_window_semantics: str
    successor_release_semantics: str
    equilibrium_workload: bool
    offered_load_normalization: str
    selection_windows_sha256: str
    retained_source_calls_sha256: str
    epoch_mapping_sha256: str
    measurement_request_identities_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ScenarioManifest = (
    BalancedCausalPrefixManifest
    | LongColdContextStressManifest
    | FullCohortSensitivityManifest
)


@dataclass(frozen=True)
class ScenarioOfferedPlan:
    """Rate-bounded wrapper around the shared unit-rate Poisson plan."""

    scenario_id: str
    arrival_contract: ArrivalRateContract
    plan: OfferedPlan

    @property
    def seed(self) -> int:
        return self.plan.seed

    @property
    def offers(self):
        return self.plan.offers

    @property
    def offered_session_ids_sha256(self) -> str:
        return self.plan.offered_session_ids_sha256

    @property
    def unit_draws_sha256(self) -> str:
        return self.plan.unit_draws_sha256

    def at_rate(
            self,
            sessions_per_second: float,
            *,
            start_time_ns: int = 0,
    ) -> tuple[ScheduledSession, ...]:
        rate = self.arrival_contract.validate_rate(sessions_per_second)
        return self.plan.at_rate(rate, start_time_ns=start_time_ns)


@dataclass(frozen=True)
class TraceLabComparisonScenario:
    """A workload, its interpretation, and a paired-arrival constructor."""

    workload: ComparisonWorkload
    manifest: ScenarioManifest
    shuffle_session_starts: bool

    def build_offered_plan(self, *, seed: int) -> ScenarioOfferedPlan:
        return ScenarioOfferedPlan(
            scenario_id=self.manifest.scenario_id,
            arrival_contract=self.manifest.arrival_contract,
            plan=build_offered_plan(
                self.workload.sessions,
                seed=seed,
                shuffle=self.shuffle_session_starts,
            ),
        )


def _token_context_stats(
        sessions: Sequence[SessionSpec],
) -> TokenContextStats:
    calls = tuple(call for session in sessions for call in session.calls)
    first = tuple(call for call in calls if call.is_first_turn)
    resume = tuple(call for call in calls if call.is_resume)
    return TokenContextStats(
        cohort=summarize_sessions(sessions),
        first_input_tokens=sum(call.input_tokens for call in first),
        resume_input_tokens=sum(call.input_tokens for call in resume),
        first_output_tokens=sum(call.output_tokens for call in first),
        resume_output_tokens=sum(call.output_tokens for call in resume),
        first_fresh_input_tokens=sum(
            call.fresh_input_tokens for call in first
        ),
        resume_fresh_input_tokens=sum(
            call.fresh_input_tokens for call in resume
        ),
        max_cached_prefix_tokens=max(
            call.cached_prefix_tokens for call in calls
        ),
        max_fresh_input_tokens=max(
            call.fresh_input_tokens for call in calls
        ),
    )


def _validate_epoch_partition(
        *,
        epoch_count: int,
        warmup_epochs: Sequence[int],
        measurement_epochs: Sequence[int],
        guard_epochs: Sequence[int],
) -> Mapping[int, str]:
    if (
        isinstance(epoch_count, bool)
        or not isinstance(epoch_count, int)
        or epoch_count <= 0
    ):
        raise ValueError("epoch_count must be a positive integer")
    groups = (
        (ROLE_WARMUP, tuple(warmup_epochs)),
        (ROLE_MEASUREMENT, tuple(measurement_epochs)),
        (ROLE_GUARD, tuple(guard_epochs)),
    )
    role_by_epoch: dict[int, str] = {}
    for role, epochs in groups:
        if not epochs:
            raise ValueError(f"{role}_epochs cannot be empty")
        for epoch in epochs:
            if (
                isinstance(epoch, bool)
                or not isinstance(epoch, int)
                or epoch < 0
                or epoch >= epoch_count
            ):
                raise ValueError(
                    f"{role} epoch {epoch!r} is outside [0, {epoch_count})"
                )
            if epoch in role_by_epoch:
                raise ValueError(
                    f"epoch {epoch} appears in multiple metric roles"
                )
            role_by_epoch[epoch] = role
    if set(role_by_epoch) != set(range(epoch_count)):
        missing = sorted(set(range(epoch_count)) - set(role_by_epoch))
        raise ValueError(
            f"epoch roles must cover every epoch; missing={missing}"
        )
    return role_by_epoch


def _truncate_session(
        session: SessionSpec,
        *,
        calls_per_session: int,
) -> SessionSpec:
    return replace(session, calls=session.calls[:calls_per_session])


def _synthetic_session(
        source: SessionSpec,
        *,
        scenario_id: str,
        epoch_index: int,
        role: str,
        base_ordinal: int,
        base_session_count: int,
) -> tuple[SessionSpec, EpochSessionMapping]:
    synthetic_source_index = (
        epoch_index * base_session_count + base_ordinal
    )
    synthetic_session_id = (
        f"{scenario_id}::epoch-{epoch_index:02d}"
        f"::source-{source.source_index:04d}::{source.session_id}"
    )
    synthetic_identity = stable_json_sha256({
        "scenario_id": scenario_id,
        "epoch_index": epoch_index,
        "source_index": source.source_index,
        "source_session_id": source.session_id,
        "source_session_identity_sha256": (
            source.source_session_identity_sha256
        ),
    })
    calls = tuple(
        replace(
            call,
            session_id=synthetic_session_id,
            source_index=synthetic_source_index,
        )
        for call in source.calls
    )
    synthetic = SessionSpec(
        source_index=synthetic_source_index,
        session_id=synthetic_session_id,
        source_arrival_time_ns=source.source_arrival_time_ns,
        source_session_identity_sha256=synthetic_identity,
        calls=calls,
    )
    mapping = EpochSessionMapping(
        epoch_index=epoch_index,
        role=role,
        base_ordinal=base_ordinal,
        synthetic_source_index=synthetic_source_index,
        synthetic_session_id=synthetic_session_id,
        synthetic_source_identity_sha256=synthetic_identity,
        source_index=source.source_index,
        source_session_id=source.session_id,
        source_session_identity_sha256=(
            source.source_session_identity_sha256
        ),
        source_arrival_time_ns=source.source_arrival_time_ns,
    )
    return synthetic, mapping


def build_balanced_causal_prefix_scenario(
        source_workload: ComparisonWorkload,
        *,
        calls_per_session: int = BALANCED_CALLS_PER_SESSION,
        epoch_count: int = BALANCED_EPOCH_COUNT,
        warmup_epochs: Sequence[int] = BALANCED_WARMUP_EPOCHS,
        measurement_epochs: Sequence[int] = (
            BALANCED_MEASUREMENT_EPOCHS),
        guard_epochs: Sequence[int] = BALANCED_GUARD_EPOCHS,
        rates: Sequence[float] = BALANCED_DEFAULT_RATES,
        maximum_rate: float = BALANCED_MAX_RATE,
        isolated_prefill_service: (
            IsolatedPrefillServiceAudit | None
        ) = None,
        expected_base_session_count: int | None = None,
) -> TraceLabComparisonScenario:
    """Build an epoch-repeated causal prefix from eligible source sessions.

    Selection is deterministic: retain source order, require at least
    ``calls_per_session`` calls, and truncate every selected session to that
    many calls.  Epoch order is retained in the offered plan so warmup,
    measurement, and guard membership cannot be shuffled across boundaries.
    """

    if not isinstance(source_workload, ComparisonWorkload):
        raise TypeError("source_workload must be a ComparisonWorkload")
    if (
        isinstance(calls_per_session, bool)
        or not isinstance(calls_per_session, int)
        or calls_per_session <= 1
    ):
        raise ValueError("calls_per_session must be an integer >= 2")
    role_by_epoch = _validate_epoch_partition(
        epoch_count=epoch_count,
        warmup_epochs=warmup_epochs,
        measurement_epochs=measurement_epochs,
        guard_epochs=guard_epochs,
    )
    base_sessions = tuple(
        _truncate_session(
            session, calls_per_session=calls_per_session
        )
        for session in source_workload.sessions
        if len(session.calls) >= calls_per_session
    )
    if not base_sessions:
        raise WorkloadValidationError(
            "no source session has enough calls for the causal prefix"
        )
    if (
        expected_base_session_count is not None
        and len(base_sessions) != expected_base_session_count
    ):
        raise WorkloadValidationError(
            "eligible causal-prefix session count mismatch: "
            f"observed={len(base_sessions)}, "
            f"expected={expected_base_session_count}"
        )

    scenario_id = (
        f"tracelab-balanced-{calls_per_session}-call-causal-prefix-v1"
    )
    synthetic_sessions = []
    mappings = []
    for epoch_index in range(epoch_count):
        role = role_by_epoch[epoch_index]
        for base_ordinal, source in enumerate(base_sessions):
            synthetic, mapping = _synthetic_session(
                source,
                scenario_id=scenario_id,
                epoch_index=epoch_index,
                role=role,
                base_ordinal=base_ordinal,
                base_session_count=len(base_sessions),
            )
            synthetic_sessions.append(synthetic)
            mappings.append(mapping)

    full_sessions = tuple(synthetic_sessions)
    mapping_values = tuple(mappings)
    measurement_id_set = {
        mapping.synthetic_session_id
        for mapping in mapping_values
        if mapping.role == ROLE_MEASUREMENT
    }
    measurement_sessions = tuple(
        session
        for session in full_sessions
        if session.session_id in measurement_id_set
    )
    measurement_calls = tuple(
        call
        for session in measurement_sessions
        for call in session.calls
    )
    measurement_session_ids = tuple(
        session.session_id for session in measurement_sessions
    )
    measurement_request_identities = tuple(
        call.completion_identity for call in measurement_calls
    )
    measurement_first_identities = tuple(
        call.completion_identity
        for call in measurement_calls
        if call.is_first_turn
    )
    measurement_resume_identities = tuple(
        call.completion_identity
        for call in measurement_calls
        if call.is_resume
    )

    rates_tuple = tuple(float(rate) for rate in rates)
    arrival_contract = ArrivalRateContract(
        rates=rates_tuple,
        maximum_rate=float(maximum_rate),
        enumerated_only=False,
        rate_unit="system_wide_causal_session_starts_per_second",
        process="seeded_poisson_exponential_interarrivals",
        first_arrival_semantics="first_session_arrives_at_start_time",
        offer_order_semantics=(
            "epoch_major_then_pinned_source_order_no_cross_epoch_shuffle"
        ),
    )
    for rate in rates_tuple:
        arrival_contract.validate_rate(rate)

    replay_workload = ComparisonWorkload(
        source_path=source_workload.source_path,
        source_sha256=source_workload.source_sha256,
        source_session_count=source_workload.source_session_count,
        sessions=full_sessions,
        summary=summarize_sessions(full_sessions),
    )
    mapping_payload = [
        asdict(mapping) for mapping in mapping_values
    ]
    manifest = BalancedCausalPrefixManifest(
        schema_version=SCENARIO_SCHEMA_VERSION,
        scenario_id=scenario_id,
        source_sha256=source_workload.source_sha256,
        source_session_count=source_workload.source_session_count,
        selected_source_indices=tuple(
            session.source_index for session in base_sessions
        ),
        selected_source_session_ids=tuple(
            session.session_id for session in base_sessions
        ),
        calls_per_session=calls_per_session,
        epoch_count=epoch_count,
        warmup_epochs=tuple(warmup_epochs),
        measurement_epochs=tuple(measurement_epochs),
        guard_epochs=tuple(guard_epochs),
        epoch_mapping=mapping_values,
        base_stats=_token_context_stats(base_sessions),
        full_replay_stats=_token_context_stats(full_sessions),
        measurement_stats=_token_context_stats(
            measurement_sessions
        ),
        measurement_session_ids=measurement_session_ids,
        measurement_request_identities=(
            measurement_request_identities
        ),
        measurement_first_request_identities=(
            measurement_first_identities
        ),
        measurement_resume_request_identities=(
            measurement_resume_identities
        ),
        isolated_prefill_service=isolated_prefill_service,
        arrival_contract=arrival_contract,
        selection_semantics=(
            "retain_pinned_source_order_filter_sessions_with_at_least_"
            f"{calls_per_session}_calls_then_keep_calls_0_through_"
            f"{calls_per_session - 1}"
        ),
        workload_semantics=(
            f"finite_epoch_repeated_{calls_per_session}_call_"
            "causal_prefix_not_equilibrium"
        ),
        metric_window_semantics=(
            "fixed_epoch_membership_"
            f"warmup_{'_'.join(map(str, warmup_epochs))}_"
            f"measure_{'_'.join(map(str, measurement_epochs))}_"
            f"guard_{'_'.join(map(str, guard_epochs))}_with_full_drain"
        ),
        successor_release_semantics=(
            "call_n_plus_1_released_only_after_call_n_completion_plus_"
            "recorded_tool_duration"
        ),
        equilibrium_workload=False,
        offered_load_normalization=(
            "measurement_goodput_is_normalized_to_the_system_wide_"
            "session_start_rate"
        ),
        epoch_mapping_sha256=stable_json_sha256(mapping_payload),
        measurement_request_identities_sha256=stable_json_sha256(
            list(measurement_request_identities)
        ),
    )
    return TraceLabComparisonScenario(
        workload=replay_workload,
        manifest=manifest,
        shuffle_session_starts=False,
    )


def load_balanced_causal_prefix_scenario(
        path: str | Path,
) -> TraceLabComparisonScenario:
    """Load and strictly validate the pinned 27-session, eight-epoch replay."""

    source = load_fixed_comparison_workload(path)
    observed_indices = tuple(
        session.source_index
        for session in source.sessions
        if len(session.calls) >= BALANCED_CALLS_PER_SESSION
    )
    if observed_indices != BALANCED_SOURCE_INDICES:
        raise WorkloadValidationError(
            "pinned balanced source selection changed: "
            f"observed={observed_indices}, "
            f"expected={BALANCED_SOURCE_INDICES}"
        )
    audit = IsolatedPrefillServiceAudit(
        first_service_ns=PINNED_FIRST_PREFILL_SERVICE_NS,
        resume_service_ns=PINNED_RESUME_PREFILL_SERVICE_NS,
        resume_to_first_ratio=(
            PINNED_RESUME_TO_FIRST_SERVICE_RATIO
        ),
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        topology="one_tp4_h100_prefill_partition",
        latency_band="central",
        method=(
            "sum_of_isolated_single_request_P4D4LatencyModel_batches_"
            "using_raw_tracelab_prefix_without_queueing_or_cross_request_"
            "batching"
        ),
    )
    scenario = build_balanced_causal_prefix_scenario(
        source,
        isolated_prefill_service=audit,
        expected_base_session_count=len(BALANCED_SOURCE_INDICES),
    )
    manifest = scenario.manifest
    if not isinstance(manifest, BalancedCausalPrefixManifest):
        raise RuntimeError("balanced scenario constructed wrong manifest")
    if (
        manifest.source_sha256 != TRACELAB_SCHEMA3_SHA256
        or manifest.measurement_stats.cohort.first_turn_count != 108
        or manifest.measurement_stats.cohort.resume_count != 216
    ):
        raise WorkloadValidationError(
            "pinned balanced causal-prefix contract mismatch"
        )
    return scenario


def _long_cold_source_call_payload(
        sessions: Sequence[SessionSpec],
) -> list[dict[str, object]]:
    return [
        {
            "source_index": session.source_index,
            "session_id": session.session_id,
            "source_arrival_time_ns": session.source_arrival_time_ns,
            "source_session_identity_sha256": (
                session.source_session_identity_sha256
            ),
            "call_index": call.call_index,
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "tool_duration_ns": call.tool_duration_ns,
            "cached_prefix_tokens": call.cached_prefix_tokens,
            "fresh_input_tokens": call.fresh_input_tokens,
            "lineage_status": call.lineage_status,
            "inter_turn_gap_type": call.inter_turn_gap_type,
        }
        for session in sessions
        for call in session.calls
    ]


def _long_cold_final_kv_bytes(
        sessions: Sequence[SessionSpec],
        *,
        epoch_count: int,
) -> int:
    block = LONG_COLD_KV_BLOCK_SIZE_TOKENS
    bytes_per_token = LONG_COLD_LOGICAL_KV_BYTES_PER_TOKEN
    per_epoch = sum(
        (
            (
                call.input_tokens + call.output_tokens - 1
                + block - 1
            )
            // block
            * block
            * bytes_per_token
        )
        for session in sessions
        for call in (session.calls[-1],)
    )
    return per_epoch * epoch_count


def _select_long_cold_prefixes(
        source_workload: ComparisonWorkload,
        *,
        source_indices: Sequence[int],
        cached_prefix_threshold: int,
        successor_call_count: int,
) -> tuple[
        tuple[SessionSpec, ...],
        tuple[LongColdSelectionWindow, ...],
]:
    if (
        isinstance(cached_prefix_threshold, bool)
        or not isinstance(cached_prefix_threshold, int)
        or cached_prefix_threshold <= 0
    ):
        raise ValueError(
            "cached_prefix_threshold must be a positive integer"
        )
    if (
        isinstance(successor_call_count, bool)
        or not isinstance(successor_call_count, int)
        or successor_call_count < 0
    ):
        raise ValueError(
            "successor_call_count must be a non-negative integer"
        )
    selected_indices = tuple(source_indices)
    if not selected_indices:
        raise ValueError("source_indices cannot be empty")
    if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in selected_indices
    ):
        raise ValueError(
            "source_indices must contain non-negative integers"
        )
    if len(selected_indices) != len(set(selected_indices)):
        raise ValueError("source_indices cannot contain duplicates")

    session_by_source_index = {
        session.source_index: session
        for session in source_workload.sessions
    }
    missing = [
        index
        for index in selected_indices
        if index not in session_by_source_index
    ]
    if missing:
        raise WorkloadValidationError(
            f"long-cold source selection is missing indices {missing}"
        )

    retained = []
    windows = []
    for source_index in selected_indices:
        session = session_by_source_index[source_index]
        target_call_index = next(
            (
                call.call_index
                for call in session.calls
                if (
                    call.is_resume
                    and call.cached_prefix_tokens
                    >= cached_prefix_threshold
                )
            ),
            None,
        )
        if target_call_index is None:
            raise WorkloadValidationError(
                f"source index {source_index} never reaches "
                f"{cached_prefix_threshold} cached prefix tokens"
            )
        end_call_index = target_call_index + successor_call_count
        if end_call_index >= len(session.calls):
            available = len(session.calls) - target_call_index - 1
            raise WorkloadValidationError(
                f"source index {source_index} has only {available} "
                f"successor calls after target {target_call_index}; "
                f"{successor_call_count} required"
            )
        measured_window = session.calls[
            target_call_index:end_call_index + 1
        ]
        below_threshold = [
            call.call_index
            for call in measured_window
            if call.cached_prefix_tokens < cached_prefix_threshold
        ]
        if below_threshold:
            raise WorkloadValidationError(
                f"source index {source_index} measured target/successor "
                f"calls fell below {cached_prefix_threshold} cached "
                f"prefix tokens: call_indices={below_threshold}"
            )
        target = session.calls[target_call_index]
        retained_session = replace(
            session,
            calls=session.calls[:end_call_index + 1],
        )
        retained.append(retained_session)
        windows.append(LongColdSelectionWindow(
            source_index=source_index,
            source_session_id=session.session_id,
            source_session_identity_sha256=(
                session.source_session_identity_sha256
            ),
            source_arrival_time_ns=session.source_arrival_time_ns,
            source_call_count=len(session.calls),
            target_call_index=target_call_index,
            end_call_index=end_call_index,
            retained_call_count=end_call_index + 1,
            target_input_tokens=target.input_tokens,
            target_cached_prefix_tokens=target.cached_prefix_tokens,
            target_output_tokens=target.output_tokens,
            target_tool_duration_ns=target.tool_duration_ns,
        ))
    return tuple(retained), tuple(windows)


def build_long_cold_context_stress_scenario(
        source_workload: ComparisonWorkload,
        *,
        source_indices: Sequence[int] = LONG_COLD_SOURCE_INDICES,
        cached_prefix_threshold: int = LONG_COLD_CACHED_PREFIX_THRESHOLD,
        successor_call_count: int = LONG_COLD_SUCCESSOR_CALLS,
        epoch_count: int = LONG_COLD_EPOCH_COUNT,
        warmup_epochs: Sequence[int] = LONG_COLD_WARMUP_EPOCHS,
        measurement_epochs: Sequence[int] = (
            LONG_COLD_MEASUREMENT_EPOCHS),
        guard_epochs: Sequence[int] = LONG_COLD_GUARD_EPOCHS,
        anchor_rates: Sequence[float] = LONG_COLD_ANCHOR_RATES,
        maximum_rate: float = LONG_COLD_MAX_RATE,
) -> TraceLabComparisonScenario:
    """Build a native-prefix long-cold-context sensitivity.

    Each selected session is preserved from call zero through the first call
    whose cached prefix reaches the threshold, plus the exact requested number
    of native successors.  Only that target-through-successor window belongs
    to the measurement roster; its preceding calls remain mandatory causal
    warmup inside every independently renamed epoch replica.
    """

    if not isinstance(source_workload, ComparisonWorkload):
        raise TypeError("source_workload must be a ComparisonWorkload")
    role_by_epoch = _validate_epoch_partition(
        epoch_count=epoch_count,
        warmup_epochs=warmup_epochs,
        measurement_epochs=measurement_epochs,
        guard_epochs=guard_epochs,
    )
    base_sessions, selection_windows = _select_long_cold_prefixes(
        source_workload,
        source_indices=source_indices,
        cached_prefix_threshold=cached_prefix_threshold,
        successor_call_count=successor_call_count,
    )
    scenario_id = (
        f"tracelab-long-cold-{cached_prefix_threshold}-cached-"
        "native-prefix-v1"
    )

    synthetic_sessions = []
    mappings = []
    measurement_sessions = []
    for epoch_index in range(epoch_count):
        role = role_by_epoch[epoch_index]
        for base_ordinal, source in enumerate(base_sessions):
            synthetic, mapping = _synthetic_session(
                source,
                scenario_id=scenario_id,
                epoch_index=epoch_index,
                role=role,
                base_ordinal=base_ordinal,
                base_session_count=len(base_sessions),
            )
            synthetic_sessions.append(synthetic)
            mappings.append(mapping)
            if role == ROLE_MEASUREMENT:
                window = selection_windows[base_ordinal]
                measurement_sessions.append(replace(
                    synthetic,
                    calls=synthetic.calls[
                        window.target_call_index:
                        window.end_call_index + 1
                    ],
                ))

    full_sessions = tuple(synthetic_sessions)
    mapping_values = tuple(mappings)
    measured_sessions = tuple(measurement_sessions)
    measurement_calls = tuple(
        call
        for session in measured_sessions
        for call in session.calls
    )
    measurement_session_ids = tuple(
        session.session_id for session in measured_sessions
    )
    measurement_request_identities = tuple(
        call.completion_identity for call in measurement_calls
    )
    measurement_first_identities = tuple(
        call.completion_identity
        for call in measurement_calls
        if call.is_first_turn
    )
    measurement_resume_identities = tuple(
        call.completion_identity
        for call in measurement_calls
        if call.is_resume
    )

    rates = tuple(float(rate) for rate in anchor_rates)
    arrival_contract = ArrivalRateContract(
        rates=rates,
        maximum_rate=float(maximum_rate),
        enumerated_only=True,
        rate_unit="system_wide_causal_session_starts_per_second",
        process="seeded_poisson_exponential_interarrivals",
        first_arrival_semantics="first_session_arrives_at_start_time",
        offer_order_semantics=(
            "epoch_major_then_pinned_source_order_no_cross_epoch_shuffle"
        ),
    )
    for rate in rates:
        arrival_contract.validate_rate(rate)

    replay_workload = ComparisonWorkload(
        source_path=source_workload.source_path,
        source_sha256=source_workload.source_sha256,
        source_session_count=source_workload.source_session_count,
        sessions=full_sessions,
        summary=summarize_sessions(full_sessions),
    )
    mapping_payload = [
        asdict(mapping) for mapping in mapping_values
    ]
    selection_payload = [
        asdict(window) for window in selection_windows
    ]
    manifest = LongColdContextStressManifest(
        schema_version=SCENARIO_SCHEMA_VERSION,
        scenario_id=scenario_id,
        source_sha256=source_workload.source_sha256,
        source_session_count=source_workload.source_session_count,
        cached_prefix_threshold=cached_prefix_threshold,
        successor_call_count=successor_call_count,
        selected_source_indices=tuple(
            session.source_index for session in base_sessions
        ),
        selected_source_session_ids=tuple(
            session.session_id for session in base_sessions
        ),
        selection_windows=selection_windows,
        epoch_count=epoch_count,
        warmup_epochs=tuple(warmup_epochs),
        measurement_epochs=tuple(measurement_epochs),
        guard_epochs=tuple(guard_epochs),
        epoch_mapping=mapping_values,
        base_prefix_stats=_token_context_stats(base_sessions),
        full_replay_stats=_token_context_stats(full_sessions),
        measurement_stats=_token_context_stats(measured_sessions),
        measurement_session_ids=measurement_session_ids,
        measurement_request_identities=(
            measurement_request_identities
        ),
        measurement_first_request_identities=(
            measurement_first_identities
        ),
        measurement_resume_request_identities=(
            measurement_resume_identities
        ),
        arrival_contract=arrival_contract,
        selection_semantics=(
            "for_each_explicit_pinned_source_keep_complete_native_calls_"
            f"0_through_first_cached_prefix_ge_{cached_prefix_threshold}_"
            f"plus_exactly_{successor_call_count}_native_successors"
        ),
        workload_semantics=(
            "finite_epoch_repeated_trace_derived_long_cold_context_D_HBM_"
            "CPU_SSD_tier_queue_sensitivity_not_equilibrium"
        ),
        metric_window_semantics=(
            f"only_target_ge_{cached_prefix_threshold}_cached_resume_and_"
            f"{successor_call_count}_native_successors_"
            f"in_measurement_epochs_{'_'.join(map(str, measurement_epochs))}"
            "_with_complete_preceding_causal_prefix_and_full_drain"
        ),
        successor_release_semantics=(
            "call_n_plus_1_released_only_after_call_n_completion_plus_"
            "unchanged_recorded_tool_duration"
        ),
        equilibrium_workload=False,
        offered_load_normalization=(
            "finite_stress_goodput_is_offered_load_normalized_sensitivity_"
            "and_must_not_be_labeled_maximum_sustainable_throughput"
        ),
        selection_windows_sha256=stable_json_sha256(
            selection_payload
        ),
        retained_source_calls_sha256=stable_json_sha256(
            _long_cold_source_call_payload(base_sessions)
        ),
        epoch_mapping_sha256=stable_json_sha256(mapping_payload),
        measurement_request_identities_sha256=stable_json_sha256(
            list(measurement_request_identities)
        ),
    )
    return TraceLabComparisonScenario(
        workload=replay_workload,
        manifest=manifest,
        shuffle_session_starts=False,
    )


def load_long_cold_context_stress_scenario(
        path: str | Path,
) -> TraceLabComparisonScenario:
    """Load and fail-closed validate the pinned long-cold sensitivity."""

    source = load_fixed_comparison_workload(path)
    scenario = build_long_cold_context_stress_scenario(source)
    manifest = scenario.manifest
    if not isinstance(manifest, LongColdContextStressManifest):
        raise RuntimeError("long-cold scenario constructed wrong manifest")

    observed_contract = {
        "selected_source_indices": manifest.selected_source_indices,
        "target_call_indices": tuple(
            window.target_call_index
            for window in manifest.selection_windows
        ),
        "end_call_indices": tuple(
            window.end_call_index
            for window in manifest.selection_windows
        ),
        "selection_windows_sha256": (
            manifest.selection_windows_sha256
        ),
        "retained_source_calls_sha256": (
            manifest.retained_source_calls_sha256
        ),
        "epoch_mapping_sha256": manifest.epoch_mapping_sha256,
        "measurement_request_identities_sha256": (
            manifest.measurement_request_identities_sha256
        ),
        "base_stats_sha256": stable_json_sha256(
            asdict(manifest.base_prefix_stats)
        ),
        "full_replay_stats_sha256": stable_json_sha256(
            asdict(manifest.full_replay_stats)
        ),
        "measurement_stats_sha256": stable_json_sha256(
            asdict(manifest.measurement_stats)
        ),
        "block_rounded_final_kv_bytes": _long_cold_final_kv_bytes(
            scenario.workload.sessions[:len(
                manifest.selection_windows
            )],
            epoch_count=manifest.epoch_count,
        ),
    }
    expected_contract = {
        "selected_source_indices": LONG_COLD_SOURCE_INDICES,
        "target_call_indices": LONG_COLD_TARGET_CALL_INDICES,
        "end_call_indices": LONG_COLD_END_CALL_INDICES,
        "selection_windows_sha256": (
            PINNED_LONG_COLD_SELECTION_WINDOWS_SHA256
        ),
        "retained_source_calls_sha256": (
            PINNED_LONG_COLD_RETAINED_SOURCE_CALLS_SHA256
        ),
        "epoch_mapping_sha256": (
            PINNED_LONG_COLD_EPOCH_MAPPING_SHA256
        ),
        "measurement_request_identities_sha256": (
            PINNED_LONG_COLD_MEASUREMENT_IDENTITIES_SHA256
        ),
        "base_stats_sha256": PINNED_LONG_COLD_BASE_STATS_SHA256,
        "full_replay_stats_sha256": (
            PINNED_LONG_COLD_FULL_REPLAY_STATS_SHA256
        ),
        "measurement_stats_sha256": (
            PINNED_LONG_COLD_MEASUREMENT_STATS_SHA256
        ),
        "block_rounded_final_kv_bytes": (
            LONG_COLD_BLOCK_ROUNDED_FINAL_KV_BYTES
        ),
    }
    mismatches = {
        name: (observed_contract[name], expected)
        for name, expected in expected_contract.items()
        if observed_contract[name] != expected
    }
    if mismatches:
        details = ", ".join(
            f"{name}={observed!r} (expected {expected!r})"
            for name, (observed, expected) in sorted(mismatches.items())
        )
        raise WorkloadValidationError(
            f"pinned long-cold-context contract mismatch: {details}"
        )
    return scenario


def build_full_cohort_sensitivity_scenario(
        source_workload: ComparisonWorkload,
        *,
        anchor_rates: Sequence[float] = FULL_COHORT_ANCHOR_RATES,
) -> TraceLabComparisonScenario:
    """Label a complete cohort as non-steady lifecycle sensitivity only."""

    if not isinstance(source_workload, ComparisonWorkload):
        raise TypeError("source_workload must be a ComparisonWorkload")
    rates = tuple(float(rate) for rate in anchor_rates)
    if not rates:
        raise ValueError("anchor_rates cannot be empty")
    maximum_rate = max(rates)
    arrival_contract = ArrivalRateContract(
        rates=rates,
        maximum_rate=maximum_rate,
        enumerated_only=True,
        rate_unit="system_wide_causal_session_starts_per_second",
        process="seeded_poisson_exponential_interarrivals",
        first_arrival_semantics="first_session_arrives_at_start_time",
        offer_order_semantics="one_seeded_permutation_of_complete_sessions",
    )
    for rate in rates:
        arrival_contract.validate_rate(rate)
    request_identities = (
        source_workload.call_completion_identities
    )
    manifest = FullCohortSensitivityManifest(
        schema_version=SCENARIO_SCHEMA_VERSION,
        scenario_id="tracelab-full-native-cohort-lifecycle-stress-v1",
        source_sha256=source_workload.source_sha256,
        source_session_count=source_workload.source_session_count,
        selected_source_indices=tuple(
            session.source_index for session in source_workload.sessions
        ),
        selected_source_session_ids=source_workload.session_ids,
        workload_stats=_token_context_stats(
            source_workload.sessions
        ),
        measurement_session_ids=source_workload.session_ids,
        measurement_request_identities=request_identities,
        arrival_contract=arrival_contract,
        workload_semantics=(
            "complete_native_trace_finite_burst_non_steady_lifecycle_stress"
        ),
        metric_window_semantics=(
            "all_native_calls_with_exact_full_drain"
        ),
        successor_release_semantics=(
            "call_n_plus_1_released_only_after_call_n_completion_plus_"
            "recorded_tool_duration"
        ),
        equilibrium_workload=False,
        offered_load_normalization=(
            "reported_throughput_is_offered_load_normalized_and_must_not_"
            "be_labeled_observed_steady_state_throughput"
        ),
        measurement_request_identities_sha256=stable_json_sha256(
            list(request_identities)
        ),
    )
    return TraceLabComparisonScenario(
        workload=source_workload,
        manifest=manifest,
        shuffle_session_starts=True,
    )


def load_full_cohort_sensitivity_scenario(
        path: str | Path,
) -> TraceLabComparisonScenario:
    """Load the pinned full 32-session lifecycle-stress sensitivity."""

    source = load_fixed_comparison_workload(path)
    if tuple(
            session.source_index for session in source.sessions
    ) != FIXED_SOURCE_INDICES:
        raise WorkloadValidationError(
            "pinned full-cohort source selection changed"
        )
    return build_full_cohort_sensitivity_scenario(source)


__all__ = [
    "ArrivalRateContract",
    "BALANCED_CALLS_PER_SESSION",
    "BALANCED_DEFAULT_RATES",
    "BALANCED_EPOCH_COUNT",
    "BALANCED_GUARD_EPOCHS",
    "BALANCED_MAX_RATE",
    "BALANCED_MEASUREMENT_EPOCHS",
    "BALANCED_SOURCE_INDICES",
    "BALANCED_WARMUP_EPOCHS",
    "BalancedCausalPrefixManifest",
    "EpochSessionMapping",
    "FULL_COHORT_ANCHOR_RATES",
    "FullCohortSensitivityManifest",
    "IsolatedPrefillServiceAudit",
    "LONG_COLD_ANCHOR_RATES",
    "LONG_COLD_BLOCK_ROUNDED_FINAL_KV_BYTES",
    "LONG_COLD_CACHED_PREFIX_THRESHOLD",
    "LONG_COLD_COMBINED_CPU_BYTES",
    "LONG_COLD_COMBINED_D_HBM_AND_CPU_BYTES",
    "LONG_COLD_COMBINED_USABLE_D_HBM_BYTES",
    "LONG_COLD_END_CALL_INDICES",
    "LONG_COLD_EPOCH_COUNT",
    "LONG_COLD_FINAL_KV_EXCESS_BYTES",
    "LONG_COLD_GUARD_EPOCHS",
    "LONG_COLD_KV_BLOCK_SIZE_TOKENS",
    "LONG_COLD_LOGICAL_KV_BYTES_PER_TOKEN",
    "LONG_COLD_MAX_RATE",
    "LONG_COLD_MEASUREMENT_EPOCHS",
    "LONG_COLD_SOURCE_INDICES",
    "LONG_COLD_SUCCESSOR_CALLS",
    "LONG_COLD_TARGET_CALL_INDICES",
    "LONG_COLD_WARMUP_EPOCHS",
    "LongColdContextStressManifest",
    "LongColdSelectionWindow",
    "PINNED_FIRST_PREFILL_SERVICE_NS",
    "PINNED_LONG_COLD_BASE_STATS_SHA256",
    "PINNED_LONG_COLD_EPOCH_MAPPING_SHA256",
    "PINNED_LONG_COLD_FULL_REPLAY_STATS_SHA256",
    "PINNED_LONG_COLD_MEASUREMENT_IDENTITIES_SHA256",
    "PINNED_LONG_COLD_MEASUREMENT_STATS_SHA256",
    "PINNED_LONG_COLD_RETAINED_SOURCE_CALLS_SHA256",
    "PINNED_LONG_COLD_SELECTION_WINDOWS_SHA256",
    "PINNED_RESUME_PREFILL_SERVICE_NS",
    "PINNED_RESUME_TO_FIRST_SERVICE_RATIO",
    "ROLE_GUARD",
    "ROLE_MEASUREMENT",
    "ROLE_WARMUP",
    "SCENARIO_SCHEMA_VERSION",
    "ScenarioOfferedPlan",
    "TokenContextStats",
    "TraceLabComparisonScenario",
    "build_balanced_causal_prefix_scenario",
    "build_full_cohort_sensitivity_scenario",
    "build_long_cold_context_stress_scenario",
    "load_balanced_causal_prefix_scenario",
    "load_full_cohort_sensitivity_scenario",
    "load_long_cold_context_stress_scenario",
]

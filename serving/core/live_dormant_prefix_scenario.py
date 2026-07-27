"""Runtime-efficient synthetic two-turn dormant-prefix sensitivity.

This scenario starts from one pinned five-call TraceLab session, applies the
repository's audited global transform whose discarded suffix reaches 250k,
and retains only calls zero and one (maximum retained sequence: 195,151
tokens).  Each epoch then replicates that two-call prefix five times and adds
three first-call-only replicas.  The replication balances singleton TP4
first-prefill and resume-prefill service while retaining the source's
14,100.92-second external gap.

The truncation, replication, and global context scaling make this an
explicitly synthetic prefix sensitivity.  It is neither an empirical TraceLab
distribution nor a complete-session evaluation.  It is a capacity-onset
control, not a bandwidth-queue or headline-goodput workload.  The
finite-schedule and Little's-law storage estimates are analytical audits; tier
occupancy and SSD traffic must still be established from each simulator
runtime report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
from typing import Mapping, Sequence

from .hbf_comparison_workload import (
    CallSpec,
    SessionSpec,
    TRACELAB_SCHEMA3_SHA256,
    build_offered_plan,
    stable_json_sha256,
)
from .hbf_full_model_latency import (
    load_hbf_server_config,
    qwen_model_weight_bytes_per_rank,
)
from .online_latency_model import (
    H100_QWEN3_TP4_KERNEL_CALIBRATED,
    resolve_online_latency_model,
)
from .utils import get_config


SCENARIO_SCHEMA_VERSION = 1
SCENARIO_ID = (
    "tracelab-globally-250k-scaled-two-turn-dormant-prefix-sensitivity-v1"
)
TARGET_MAX_SEQUENCE_TOKENS = 250_000
EXPECTED_RETAINED_MAX_SEQUENCE_TOKENS = 195_151
SOURCE_INDEX = 1_426
SOURCE_SESSION_ID = "codex:55c85201-ad3a-4785-c8dc-677b6572eea1"
SOURCE_SESSION_IDENTITY_SHA256 = (
    "35f73dc01b29f18a37859426239ce6a51419753e961546730227de4fb74498d2"
)
EXPECTED_SOURCE_CALL_COUNT = 5
EXPECTED_TRANSFORMED_COHORT_SHA256 = (
    "2ffda62d11ddd572883c90fcd03475ef32318b56f4155b4901ceb0de80b6cd31"
)
EXPECTED_CONTEXT_FACTOR_NUMERATOR = 249_895
EXPECTED_CONTEXT_FACTOR_DENOMINATOR = 50_937

EXPECTED_SOURCE_FIRST_INPUT_TOKENS = 24_493
EXPECTED_SOURCE_RESUME_INPUT_TOKENS = 39_726
EXPECTED_SOURCE_FIRST_OUTPUT_TOKENS = 175
EXPECTED_SOURCE_RESUME_OUTPUT_TOKENS = 257
EXPECTED_SOURCE_RESUME_CACHED_TOKENS = 24_668
EXPECTED_RECORDED_GAP_NS = 14_100_920_000_000
EXPECTED_TRUNCATED_TERMINAL_GAP_NS = 51_704_000_000

EXPECTED_FIRST_INPUT_TOKENS = 120_161
EXPECTED_RESUME_INPUT_TOKENS = 194_894
EXPECTED_RESUME_CACHED_TOKENS = 120_336
EXPECTED_RESUME_FRESH_TOKENS = 74_558

COMPLETE_PREFIX_CLONES_PER_EPOCH = 5
FIRST_ONLY_PREFIX_CLONES_PER_EPOCH = 3
EXPECTED_SESSIONS_PER_EPOCH = 8
EXPECTED_CALLS_PER_EPOCH = 13
EXPECTED_FIRST_CALLS_PER_EPOCH = 8
EXPECTED_RESUME_CALLS_PER_EPOCH = 5
EXPECTED_OUTPUT_TOKENS_PER_EPOCH = 2_685
EXPECTED_INPUT_TOKENS_PER_EPOCH = 1_935_758
EXPECTED_CACHED_PREFIX_TOKENS_PER_EPOCH = 601_680
EXPECTED_FRESH_INPUT_TOKENS_PER_EPOCH = 1_334_078

EXPECTED_SINGLETON_FIRST_PREFILL_SERVICE_NS = 11_298_858_584
EXPECTED_SINGLETON_RESUME_PREFILL_SERVICE_NS = 18_388_755_540
EXPECTED_FIRST_PREFILL_SERVICE_NS_PER_EPOCH = 90_390_868_672
EXPECTED_RESUME_PREFILL_SERVICE_NS_PER_EPOCH = 91_943_777_700
EXPECTED_OPERATIONAL_RESUME_HIT_TOKENS = 120_335

LOGICAL_KV_BYTES_PER_TOKEN = 98_304
KV_BLOCK_TOKENS = 16
EXPECTED_DORMANT_LOGICAL_KV_TOKENS_PER_COMPLETE_CLONE = 120_336
EXPECTED_DORMANT_LOGICAL_KV_BYTES_PER_COMPLETE_CLONE = 11_829_510_144
EXPECTED_RECORDED_GAP_LOGICAL_KV_BYTE_NS_PER_EPOCH = (
    834_034_880_898_662_400_000_000
)


class LiveDormantPrefixScenarioError(ValueError):
    """Raised when the pinned synthetic prefix sensitivity drifts."""


# Two baseline servers: usable D-side HBM plus usable CPU DRAM.
BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES = 1_382_573_883_392
_HBF_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/wakekv_hbf/full_model_8card_server.json"
)


def _configured_hbf_logical_capacities() -> dict[str, int]:
    """Derive usable logical KV capacity from the runtime HBF config."""

    hardware, layouts = load_hbf_server_config(_HBF_CONFIG_PATH)
    required = ("tp4", "tp8", "tp8_context")
    if any(key not in layouts for key in required):
        raise LiveDormantPrefixScenarioError(
            "runtime HBF config omits a required pressure layout")
    capacities = {}
    for key in required:
        layout = layouts[key]
        weight_bytes_per_rank = qwen_model_weight_bytes_per_rank(
            layout.tp_size)
        free_bytes_per_card = (
            hardware.hbf_capacity_bytes_per_card
            - weight_bytes_per_rank
        )
        if free_bytes_per_card <= 0:
            raise LiveDormantPrefixScenarioError(
                f"modeled Qwen weights do not fit the {key} HBF layout")
        physical_free_bytes = (
            free_bytes_per_card
            * layout.tp_size
            * layout.replicas
        )
        capacities[key] = (
            physical_free_bytes
            // layout.physical_kv_replication_factor
        )
    return capacities


_HBF_LOGICAL_CAPACITIES = _configured_hbf_logical_capacities()
HBF_TP4_USABLE_LOGICAL_KV_BYTES = _HBF_LOGICAL_CAPACITIES["tp4"]
HBF_TP8_USABLE_LOGICAL_KV_BYTES = _HBF_LOGICAL_CAPACITIES["tp8"]
HBF_TP8_CONTEXT_USABLE_LOGICAL_KV_BYTES = (
    _HBF_LOGICAL_CAPACITIES["tp8_context"]
)

RECOMMENDED_RATES = (0.006, 0.012, 0.02)
RECOMMENDED_PILOT_RATES = RECOMMENDED_RATES
RECOMMENDED_SEEDS = tuple(range(101, 113))
PRESSURE_WITNESS_SEED = 101
PRESSURE_WITNESS_RATE = 0.02

PRESSURE_WARMUP_EPOCH_COUNT = 40
PRESSURE_MEASUREMENT_EPOCH_COUNT = 8
PRESSURE_GUARD_EPOCH_COUNT = 40
SMOKE_WARMUP_EPOCH_COUNT = 1
SMOKE_MEASUREMENT_EPOCH_COUNT = 1
SMOKE_GUARD_EPOCH_COUNT = 1
PROTOCOL_SMOKE_WARMUP_EPOCH_COUNT = 0
PROTOCOL_SMOKE_MEASUREMENT_EPOCH_COUNT = 1
PROTOCOL_SMOKE_GUARD_EPOCH_COUNT = 0

_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"


@dataclass(frozen=True)
class PrefixEpochMapping:
    epoch_index: int
    role: str
    role_epoch_index: int
    session_id: str
    synthetic_source_index: int
    source_index: int
    source_session_id: str
    clone_kind: str
    clone_index: int


@dataclass(frozen=True)
class TwoTurnSourceAudit:
    source_call_count: int
    retained_call_indices: tuple[int, ...]
    discarded_call_indices: tuple[int, ...]
    recorded_successor_gap_ns: int
    retained_first_input_tokens: int
    retained_resume_input_tokens: int
    retained_resume_cached_tokens: int
    truncation_semantics: str
    replication_semantics: str


@dataclass(frozen=True)
class PrefillServiceBalanceAudit:
    singleton_first_prefill_service_ns: int
    singleton_resume_prefill_service_ns: int
    first_prefill_service_ns_per_epoch: int
    resume_prefill_service_ns_per_epoch: int
    resume_to_first_service_ratio: float
    declared_resume_cached_tokens: int
    operational_resume_hit_tokens: int
    model: str
    topology: str
    latency_band: str
    method: str


@dataclass(frozen=True)
class SteadyStateKVEstimate:
    offered_session_rate_per_second: float
    analytical_recorded_gap_live_kv_bytes_floor: int
    exceeds_baseline_capacity: bool
    fits_hbf_tp4_capacity: bool
    fits_hbf_tp8_capacity: bool
    fits_hbf_tp8_context_capacity: bool


@dataclass(frozen=True)
class FinitePressureWitness:
    seed: int
    offered_session_rate_per_second: float
    zero_service_peak_recorded_gap_logical_kv_bytes: int
    exceeds_baseline_capacity: bool
    fits_smallest_hbf_capacity: bool
    semantics: str


@dataclass(frozen=True)
class KVByteGapCapacityAudit:
    logical_kv_bytes_per_token: int
    block_tokens: int
    dormant_logical_kv_tokens_per_complete_clone: int
    dormant_logical_kv_bytes_per_complete_clone: int
    recorded_gap_ns: int
    complete_prefix_clones_per_epoch: int
    recorded_gap_logical_kv_byte_ns_per_epoch: int
    baseline_combined_usable_d_hbm_and_cpu_bytes: int
    hbf_tp4_usable_logical_kv_bytes: int
    hbf_tp8_usable_logical_kv_bytes: int
    hbf_tp8_context_usable_logical_kv_bytes: int
    hbf_config_path: str
    hbf_config_sha256: str
    analytical_steady_state_estimates: tuple[SteadyStateKVEstimate, ...]
    finite_pressure_witness: FinitePressureWitness
    analytical_semantics: str
    capacity_semantics: str
    realized_runtime_semantics: str


@dataclass(frozen=True)
class FiniteScheduleCoverageAudit:
    audited_seeds: tuple[int, ...]
    audited_max_rate_per_second: float
    minimum_mean_epochs_for_one_gap: int
    warmup_epoch_count: int
    guard_epoch_count: int
    minimum_warmup_arrival_span_ns: int
    minimum_guard_arrival_span_ns: int
    warmup_covers_gap_for_all_audited_seeds: bool
    guard_covers_gap_for_all_audited_seeds: bool
    semantics: str


@dataclass(frozen=True)
class LiveDormantPrefixManifest:
    schema_version: int
    scenario_id: str
    epoch_profile: str
    source_sha256: str
    transformed_cohort_sha256: str
    selected_source_indices: tuple[int, ...]
    selected_source_session_ids: tuple[str, ...]
    target_max_sequence_tokens: int
    retained_max_sequence_tokens: int
    context_factor_numerator: int
    context_factor_denominator: int
    epoch_count: int
    warmup_epochs: tuple[int, ...]
    measurement_epochs: tuple[int, ...]
    guard_epochs: tuple[int, ...]
    sessions_per_epoch: int
    calls_per_epoch: int
    first_calls_per_epoch: int
    resume_calls_per_epoch: int
    output_tokens_per_epoch: int
    measurement_session_ids: tuple[str, ...]
    measurement_request_count: int
    measurement_first_call_count: int
    measurement_resume_call_count: int
    epoch_mapping_sha256: str
    source_audit: TwoTurnSourceAudit
    prefill_service: PrefillServiceBalanceAudit
    kv_pressure: KVByteGapCapacityAudit
    schedule_coverage: FiniteScheduleCoverageAudit
    recommended_rates: tuple[float, ...]
    recommended_seeds: tuple[int, ...]
    rate_semantics: str
    workload_semantics: str
    successor_release_semantics: str
    measurement_semantics: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LiveDormantPrefixScenario:
    manifest: LiveDormantPrefixManifest
    epoch_sessions: tuple[tuple[SessionSpec, ...], ...]

    def build_offered_plan(self, *, seed: int):
        return RateBoundedOfferedPlan(
            plan=_build_epoch_offered_plan(
                self.epoch_sessions, seed=seed),
            maximum_rate=max(self.manifest.recommended_rates),
        )


@dataclass(frozen=True)
class RateBoundedOfferedPlan:
    """Forward an offered plan while enforcing the audited rate ceiling."""

    plan: object
    maximum_rate: float

    def __getattr__(self, name: str):
        return getattr(self.plan, name)

    def at_rate(self, sessions_per_second: float, **kwargs):
        if (
            isinstance(sessions_per_second, bool)
            or not isinstance(sessions_per_second, (int, float))
            or not math.isfinite(float(sessions_per_second))
            or float(sessions_per_second) <= 0
        ):
            raise ValueError(
                "sessions_per_second must be positive and finite")
        rate = float(sessions_per_second)
        if rate > self.maximum_rate:
            raise ValueError(
                f"sessions_per_second={rate} exceeds the audited maximum "
                f"{self.maximum_rate}")
        return self.plan.at_rate(rate, **kwargs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_epoch_offered_plan(
        epoch_sessions: Sequence[Sequence[SessionSpec]],
        *,
        seed: int,
):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    ordered = []
    for epoch_index, group in enumerate(epoch_sessions):
        shuffled = list(group)
        epoch_seed = int.from_bytes(
            hashlib.sha256(
                f"{SCENARIO_ID}:{seed}:{epoch_index}".encode("utf-8")
            ).digest()[:8],
            byteorder="big",
        )
        random.Random(epoch_seed).shuffle(shuffled)
        ordered.extend(shuffled)
    return build_offered_plan(tuple(ordered), seed=seed, shuffle=False)


def _selected_source_row(trace_path: Path) -> Mapping[str, object]:
    with trace_path.open("r", encoding="utf-8") as source:
        for source_index, line in enumerate(source):
            if source_index == SOURCE_INDEX:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise LiveDormantPrefixScenarioError(
                        "pinned TraceLab source row is not an object")
                return row
    raise LiveDormantPrefixScenarioError(
        f"TraceLab source row {SOURCE_INDEX} is missing")


def _validate_source_row(row: Mapping[str, object]) -> TwoTurnSourceAudit:
    """Fail closed on the source call count and the retained boundary."""

    calls = row.get("sub_requests")
    if not isinstance(calls, list):
        raise LiveDormantPrefixScenarioError(
            "pinned TraceLab source has no sub_requests list")
    if len(calls) != EXPECTED_SOURCE_CALL_COUNT:
        raise LiveDormantPrefixScenarioError(
            "pinned TraceLab source call count changed: "
            f"observed={len(calls)}, expected={EXPECTED_SOURCE_CALL_COUNT}")
    trace_metadata = row.get("trace_metadata")
    if not isinstance(trace_metadata, dict):
        raise LiveDormantPrefixScenarioError(
            "pinned TraceLab source has no trace metadata")
    observed_identity = (
        row.get("session_id"),
        trace_metadata.get("source_session_identity_sha256"),
    )
    expected_identity = (
        SOURCE_SESSION_ID,
        SOURCE_SESSION_IDENTITY_SHA256,
    )
    if observed_identity != expected_identity:
        raise LiveDormantPrefixScenarioError(
            "pinned TraceLab source identity changed")

    first = calls[0]
    resume = calls[1]
    if not isinstance(first, dict) or not isinstance(resume, dict):
        raise LiveDormantPrefixScenarioError(
            "retained TraceLab calls are not objects")
    observed = (
        int(first.get("input_toks", -1)),
        int(first.get("output_toks", -1)),
        int(first.get("prefix_reuse_toks", -1)),
        int(first.get("tool_duration_ns", -1)),
        first.get("lineage_status"),
        first.get("inter_turn_gap_type"),
        int(resume.get("input_toks", -1)),
        int(resume.get("output_toks", -1)),
        int(resume.get("prefix_reuse_toks", -1)),
        int(resume.get("tool_duration_ns", -1)),
        resume.get("lineage_status"),
        resume.get("inter_turn_gap_type"),
    )
    expected = (
        EXPECTED_SOURCE_FIRST_INPUT_TOKENS,
        EXPECTED_SOURCE_FIRST_OUTPUT_TOKENS,
        0,
        EXPECTED_RECORDED_GAP_NS,
        "session_start",
        "human",
        EXPECTED_SOURCE_RESUME_INPUT_TOKENS,
        EXPECTED_SOURCE_RESUME_OUTPUT_TOKENS,
        EXPECTED_SOURCE_RESUME_CACHED_TOKENS,
        EXPECTED_TRUNCATED_TERMINAL_GAP_NS,
        "adjacent_estimate",
        "human",
    )
    if observed != expected:
        raise LiveDormantPrefixScenarioError(
            "pinned TraceLab two-turn prefix or recorded gap changed: "
            f"observed={observed}, expected={expected}")
    return TwoTurnSourceAudit(
        source_call_count=len(calls),
        retained_call_indices=(0, 1),
        discarded_call_indices=(2, 3, 4),
        recorded_successor_gap_ns=EXPECTED_RECORDED_GAP_NS,
        retained_first_input_tokens=EXPECTED_SOURCE_FIRST_INPUT_TOKENS,
        retained_resume_input_tokens=EXPECTED_SOURCE_RESUME_INPUT_TOKENS,
        retained_resume_cached_tokens=EXPECTED_SOURCE_RESUME_CACHED_TOKENS,
        truncation_semantics=(
            "the five-call source session is intentionally truncated after "
            "call 1; calls 2-4 are discarded and call 1's post-call gap is "
            "terminal and therefore does not release another request"),
        replication_semantics=(
            "the retained prefix is replicated into five two-call clones "
            "plus three first-call-only clones per epoch; this is synthetic "
            "service balancing, not an empirical frequency estimate"),
    )


def _selection() -> dict[str, object]:
    return {
        "strategy": "all",
        "include_source_indices": [SOURCE_INDEX],
        "max_sessions": 1,
        "target_max_sequence_tokens": TARGET_MAX_SEQUENCE_TOKENS,
    }


def _materialized_template(
        trace_path: Path,
) -> tuple[SessionSpec, str, dict[str, object], TwoTurnSourceAudit]:
    if _sha256_file(trace_path) != TRACELAB_SCHEMA3_SHA256:
        raise LiveDormantPrefixScenarioError(
            "TraceLab source SHA-256 does not match the pinned schema-3 trace")
    source_audit = _validate_source_row(_selected_source_row(trace_path))

    from serving.online_experiments import materialize_session_cohort

    with tempfile.TemporaryDirectory(
            prefix="llmsim-live-dormant-prefix-") as temporary:
        descriptor = materialize_session_cohort(
            trace_path,
            Path(temporary),
            _selection(),
        )
        transformed_path = Path(descriptor["materialized_path"])
        transformed_sha = _sha256_file(transformed_path)
        if transformed_sha != EXPECTED_TRANSFORMED_COHORT_SHA256:
            raise LiveDormantPrefixScenarioError(
                "250k transformed TraceLab prefix source changed: "
                f"observed={transformed_sha}, "
                f"expected={EXPECTED_TRANSFORMED_COHORT_SHA256}")
        rows = tuple(
            json.loads(line)
            for line in transformed_path.read_text(
                encoding="utf-8").splitlines()
            if line
        )
    if len(rows) != 1:
        raise LiveDormantPrefixScenarioError(
            "materialized prefix source must contain exactly one session")
    row = rows[0]
    source = row.get("online_experiment_source")
    if not isinstance(source, dict) or (
        int(source.get("source_index", -1)) != SOURCE_INDEX
        or source.get("source_session_id") != SOURCE_SESSION_ID
    ):
        raise LiveDormantPrefixScenarioError(
            "materialized prefix source lineage changed")

    transform = dict(descriptor["context_length_transform"])
    if (
        transform.get("global_factor_numerator")
        != EXPECTED_CONTEXT_FACTOR_NUMERATOR
        or transform.get("global_factor_denominator")
        != EXPECTED_CONTEXT_FACTOR_DENOMINATOR
        or transform.get("realized_max_sequence_tokens")
        != TARGET_MAX_SEQUENCE_TOKENS
    ):
        raise LiveDormantPrefixScenarioError(
            "pinned 250k context transform changed")

    transformed_calls = row.get("sub_requests")
    if (
        not isinstance(transformed_calls, list)
        or len(transformed_calls) != EXPECTED_SOURCE_CALL_COUNT
    ):
        raise LiveDormantPrefixScenarioError(
            "materialized source call count changed")
    retained = transformed_calls[:2]
    observed = tuple(
        (
            int(call["input_toks"]),
            int(call["output_toks"]),
            int(call["prefix_reuse_toks"]),
            int(call["tool_duration_ns"]),
            call.get("lineage_status"),
            call.get("inter_turn_gap_type"),
        )
        for call in retained
    )
    expected = (
        (
            EXPECTED_FIRST_INPUT_TOKENS,
            EXPECTED_SOURCE_FIRST_OUTPUT_TOKENS,
            0,
            EXPECTED_RECORDED_GAP_NS,
            "session_start",
            "human",
        ),
        (
            EXPECTED_RESUME_INPUT_TOKENS,
            EXPECTED_SOURCE_RESUME_OUTPUT_TOKENS,
            EXPECTED_RESUME_CACHED_TOKENS,
            EXPECTED_TRUNCATED_TERMINAL_GAP_NS,
            "adjacent_estimate",
            "human",
        ),
    )
    if observed != expected:
        raise LiveDormantPrefixScenarioError(
            "materialized two-turn prefix coordinates changed: "
            f"observed={observed}, expected={expected}")

    calls = []
    for call_index, call in enumerate(retained):
        input_tokens = int(call["input_toks"])
        cached_tokens = int(call["prefix_reuse_toks"])
        if not 0 <= cached_tokens <= input_tokens:
            raise LiveDormantPrefixScenarioError(
                "materialized cached prefix is outside its prompt")
        calls.append(CallSpec(
            session_id=SOURCE_SESSION_ID,
            source_index=SOURCE_INDEX,
            call_index=call_index,
            input_tokens=input_tokens,
            output_tokens=int(call["output_toks"]),
            tool_duration_ns=int(call["tool_duration_ns"]),
            cached_prefix_tokens=cached_tokens,
            fresh_input_tokens=input_tokens - cached_tokens,
            lineage_status=call.get("lineage_status"),
            inter_turn_gap_type=call.get("inter_turn_gap_type"),
        ))
    template = SessionSpec(
        source_index=SOURCE_INDEX,
        session_id=SOURCE_SESSION_ID,
        source_arrival_time_ns=int(row.get("arrival_time_ns", 0)),
        source_session_identity_sha256=SOURCE_SESSION_IDENTITY_SHA256,
        calls=tuple(calls),
    )
    return template, transformed_sha, transform, source_audit


def _latency_provider():
    repo_root = Path(__file__).resolve().parents[2]
    return resolve_online_latency_model(
        name=H100_QWEN3_TP4_KERNEL_CALIBRATED,
        repo_root=repo_root,
        hardware="H100",
        model=_MODEL_NAME,
        config=get_config(_MODEL_NAME),
        tp_size=4,
        pp_size=1,
        local_ep=4,
        ep_total=4,
        fp_bytes=2,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        enable_attn_offloading=False,
    )


def _prefill_service_audit(
        template: SessionSpec,
) -> PrefillServiceBalanceAudit:
    provider = _latency_provider()
    first, resume = template.calls
    first_ns = provider.singleton_prefill_comp_ns(
        input_tokens=first.input_tokens,
        hit_tokens=0,
        max_chunk_tokens=131_072,
    )
    operational_hit_tokens = min(
        resume.cached_prefix_tokens,
        first.input_tokens + first.output_tokens - 1,
        resume.input_tokens - 1,
    )
    if operational_hit_tokens != EXPECTED_OPERATIONAL_RESUME_HIT_TOKENS:
        raise LiveDormantPrefixScenarioError(
            "operational predecessor reuse boundary changed")
    resume_ns = provider.singleton_prefill_comp_ns(
        input_tokens=resume.input_tokens,
        hit_tokens=operational_hit_tokens,
        max_chunk_tokens=131_072,
    )
    aggregate_first_ns = first_ns * EXPECTED_FIRST_CALLS_PER_EPOCH
    aggregate_resume_ns = resume_ns * EXPECTED_RESUME_CALLS_PER_EPOCH
    observed = (
        first_ns,
        resume_ns,
        aggregate_first_ns,
        aggregate_resume_ns,
    )
    expected = (
        EXPECTED_SINGLETON_FIRST_PREFILL_SERVICE_NS,
        EXPECTED_SINGLETON_RESUME_PREFILL_SERVICE_NS,
        EXPECTED_FIRST_PREFILL_SERVICE_NS_PER_EPOCH,
        EXPECTED_RESUME_PREFILL_SERVICE_NS_PER_EPOCH,
    )
    if observed != expected:
        raise LiveDormantPrefixScenarioError(
            "pinned first/resume prefill service audit changed: "
            f"observed={observed}, expected={expected}")
    return PrefillServiceBalanceAudit(
        singleton_first_prefill_service_ns=first_ns,
        singleton_resume_prefill_service_ns=resume_ns,
        first_prefill_service_ns_per_epoch=aggregate_first_ns,
        resume_prefill_service_ns_per_epoch=aggregate_resume_ns,
        resume_to_first_service_ratio=(
            aggregate_resume_ns / aggregate_first_ns),
        declared_resume_cached_tokens=resume.cached_prefix_tokens,
        operational_resume_hit_tokens=operational_hit_tokens,
        model=_MODEL_NAME,
        topology="one_tp4_h100_model_partition",
        latency_band="central",
        method=(
            "sum_of_singleton_prefill_COMP_critical_paths_with_131072_"
            "token_chunks_no_collectives_no_queueing; operational hit is "
            "capped by predecessor input+output-1"),
    )


def _block_rounded_kv_bytes(token_count: int) -> int:
    blocks = (token_count + KV_BLOCK_TOKENS - 1) // KV_BLOCK_TOKENS
    return blocks * KV_BLOCK_TOKENS * LOGICAL_KV_BYTES_PER_TOKEN


def _steady_state_estimates(
        byte_ns_per_epoch: int,
) -> tuple[SteadyStateKVEstimate, ...]:
    estimates = []
    for rate in RECOMMENDED_RATES:
        rate_fraction = Fraction(str(rate))
        numerator = byte_ns_per_epoch * rate_fraction.numerator
        denominator = (
            EXPECTED_SESSIONS_PER_EPOCH
            * 1_000_000_000
            * rate_fraction.denominator
        )
        live_bytes = numerator // denominator
        estimates.append(SteadyStateKVEstimate(
            offered_session_rate_per_second=rate,
            analytical_recorded_gap_live_kv_bytes_floor=live_bytes,
            exceeds_baseline_capacity=(
                live_bytes
                > BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES),
            fits_hbf_tp4_capacity=(
                live_bytes < HBF_TP4_USABLE_LOGICAL_KV_BYTES),
            fits_hbf_tp8_capacity=(
                live_bytes < HBF_TP8_USABLE_LOGICAL_KV_BYTES),
            fits_hbf_tp8_context_capacity=(
                live_bytes < HBF_TP8_CONTEXT_USABLE_LOGICAL_KV_BYTES),
        ))
    return tuple(estimates)


def _validate_epoch_count(value: int, name: str, *, positive: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        bound = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {bound}")
    return value


def _epoch_profile(
        warmup_epoch_count: int,
        measurement_epoch_count: int,
        guard_epoch_count: int,
) -> str:
    counts = (
        warmup_epoch_count,
        measurement_epoch_count,
        guard_epoch_count,
    )
    if counts == (
        PRESSURE_WARMUP_EPOCH_COUNT,
        PRESSURE_MEASUREMENT_EPOCH_COUNT,
        PRESSURE_GUARD_EPOCH_COUNT,
    ):
        return "pressure"
    if counts == (
        SMOKE_WARMUP_EPOCH_COUNT,
        SMOKE_MEASUREMENT_EPOCH_COUNT,
        SMOKE_GUARD_EPOCH_COUNT,
    ):
        return "smoke"
    if counts == (
        PROTOCOL_SMOKE_WARMUP_EPOCH_COUNT,
        PROTOCOL_SMOKE_MEASUREMENT_EPOCH_COUNT,
        PROTOCOL_SMOKE_GUARD_EPOCH_COUNT,
    ):
        return "protocol_smoke"
    return "custom"


def _clone_session(
        source: SessionSpec,
        *,
        epoch_index: int,
        role: str,
        role_epoch_index: int,
        clone_kind: str,
        clone_index: int,
        synthetic_source_index: int,
) -> tuple[SessionSpec, PrefixEpochMapping]:
    if clone_kind == "complete_two_turn_prefix":
        source_calls: Sequence[CallSpec] = source.calls
    elif clone_kind == "first_turn_only_prefix":
        source_calls = source.calls[:1]
    else:
        raise ValueError(f"unsupported clone_kind {clone_kind!r}")
    session_id = (
        f"{SCENARIO_ID}::{role}-{role_epoch_index:03d}"
        f"::epoch-{epoch_index:03d}::{clone_kind}-{clone_index:02d}"
        f"::source-{source.source_index:04d}"
    )
    calls = tuple(
        replace(
            call,
            session_id=session_id,
            source_index=synthetic_source_index,
            call_index=call_index,
        )
        for call_index, call in enumerate(source_calls)
    )
    identity = stable_json_sha256({
        "scenario_id": SCENARIO_ID,
        "epoch_index": epoch_index,
        "role": role,
        "role_epoch_index": role_epoch_index,
        "clone_kind": clone_kind,
        "clone_index": clone_index,
        "source_index": source.source_index,
        "source_session_id": source.session_id,
        "source_session_identity_sha256": (
            source.source_session_identity_sha256),
        "retained_call_count": len(calls),
    })
    session = SessionSpec(
        source_index=synthetic_source_index,
        session_id=session_id,
        source_arrival_time_ns=source.source_arrival_time_ns,
        source_session_identity_sha256=identity,
        calls=calls,
    )
    return session, PrefixEpochMapping(
        epoch_index=epoch_index,
        role=role,
        role_epoch_index=role_epoch_index,
        session_id=session_id,
        synthetic_source_index=synthetic_source_index,
        source_index=source.source_index,
        source_session_id=source.session_id,
        clone_kind=clone_kind,
        clone_index=clone_index,
    )


def _schedule_coverage_audit(
        epoch_sessions: Sequence[Sequence[SessionSpec]],
        *,
        warmup_epoch_count: int,
        measurement_epoch_count: int,
        guard_epoch_count: int,
        profile: str,
) -> FiniteScheduleCoverageAudit:
    minimum_mean_epochs = math.ceil(
        max(RECOMMENDED_RATES)
        * EXPECTED_RECORDED_GAP_NS
        / (EXPECTED_SESSIONS_PER_EPOCH * 1_000_000_000)
    )
    warmup_spans = []
    guard_spans = []
    warmup_sessions = (
        warmup_epoch_count * EXPECTED_SESSIONS_PER_EPOCH)
    measurement_sessions = (
        measurement_epoch_count * EXPECTED_SESSIONS_PER_EPOCH)
    for seed in RECOMMENDED_SEEDS:
        scheduled = _build_epoch_offered_plan(
            epoch_sessions, seed=seed).at_rate(max(RECOMMENDED_RATES))
        if warmup_sessions:
            first_measurement = warmup_sessions
            warmup_span = (
                scheduled[first_measurement].arrival_time_ns
                - scheduled[0].arrival_time_ns
            )
        else:
            warmup_span = 0
        if guard_epoch_count:
            last_measurement = (
                warmup_sessions + measurement_sessions - 1)
            guard_span = (
                scheduled[-1].arrival_time_ns
                - scheduled[last_measurement].arrival_time_ns
            )
        else:
            guard_span = 0
        warmup_spans.append(warmup_span)
        guard_spans.append(guard_span)
    minimum_warmup_span = min(warmup_spans)
    minimum_guard_span = min(guard_spans)
    warmup_covers = minimum_warmup_span >= EXPECTED_RECORDED_GAP_NS
    guard_covers = minimum_guard_span >= EXPECTED_RECORDED_GAP_NS
    if profile == "pressure" and not (warmup_covers and guard_covers):
        raise LiveDormantPrefixScenarioError(
            "pressure profile no longer covers one recorded gap on both "
            "sides of the measurement roster")
    return FiniteScheduleCoverageAudit(
        audited_seeds=RECOMMENDED_SEEDS,
        audited_max_rate_per_second=max(RECOMMENDED_RATES),
        minimum_mean_epochs_for_one_gap=minimum_mean_epochs,
        warmup_epoch_count=warmup_epoch_count,
        guard_epoch_count=guard_epoch_count,
        minimum_warmup_arrival_span_ns=minimum_warmup_span,
        minimum_guard_arrival_span_ns=minimum_guard_span,
        warmup_covers_gap_for_all_audited_seeds=warmup_covers,
        guard_covers_gap_for_all_audited_seeds=guard_covers,
        semantics=(
            "exact Poisson arrival spans for seeds 101-112 at the highest "
            "selected rate; lower selected rates only lengthen these spans. "
            "Coverage maintains offered load around delayed measurement "
            "resumes but does not assert realized tier occupancy"),
    )


def _finite_pressure_witness(
        epoch_sessions: Sequence[Sequence[SessionSpec]],
) -> FinitePressureWitness:
    scheduled = _build_epoch_offered_plan(
        epoch_sessions,
        seed=PRESSURE_WITNESS_SEED,
    ).at_rate(PRESSURE_WITNESS_RATE)
    events = []
    for item in scheduled:
        if len(item.session.calls) != 2:
            continue
        logical_bytes = _block_rounded_kv_bytes(
            item.session.calls[0].input_tokens
            + item.session.calls[0].output_tokens
        )
        start_ns = item.arrival_time_ns
        end_ns = start_ns + EXPECTED_RECORDED_GAP_NS
        # End events sort before starts when timestamps are equal.
        events.append((start_ns, 1, logical_bytes))
        events.append((end_ns, 0, -logical_bytes))
    live_bytes = 0
    peak_bytes = 0
    for _, _, delta_bytes in sorted(events):
        live_bytes += delta_bytes
        if live_bytes < 0:
            raise LiveDormantPrefixScenarioError(
                "finite dormant-prefix accounting became negative")
        peak_bytes = max(peak_bytes, live_bytes)
    if live_bytes != 0:
        raise LiveDormantPrefixScenarioError(
            "finite dormant-prefix accounting did not drain")
    return FinitePressureWitness(
        seed=PRESSURE_WITNESS_SEED,
        offered_session_rate_per_second=PRESSURE_WITNESS_RATE,
        zero_service_peak_recorded_gap_logical_kv_bytes=peak_bytes,
        exceeds_baseline_capacity=(
            peak_bytes
            > BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES),
        fits_smallest_hbf_capacity=(
            peak_bytes < HBF_TP8_USABLE_LOGICAL_KV_BYTES),
        semantics=(
            "exact interval sweep over the finite offered schedule with LLM "
            "service, queueing, migration, and transfer time set to zero; "
            "the runtime report remains authoritative for realized pressure"),
    )


def build_live_dormant_prefix_scenario(
        trace_path: str | Path,
        *,
        warmup_epoch_count: int = PRESSURE_WARMUP_EPOCH_COUNT,
        measurement_epoch_count: int = PRESSURE_MEASUREMENT_EPOCH_COUNT,
        guard_epoch_count: int = PRESSURE_GUARD_EPOCH_COUNT,
) -> LiveDormantPrefixScenario:
    """Build and fail-closed validate the synthetic two-turn sensitivity."""

    warmup_epoch_count = _validate_epoch_count(
        warmup_epoch_count, "warmup_epoch_count", positive=False)
    measurement_epoch_count = _validate_epoch_count(
        measurement_epoch_count, "measurement_epoch_count", positive=True)
    guard_epoch_count = _validate_epoch_count(
        guard_epoch_count, "guard_epoch_count", positive=False)
    path = Path(trace_path).resolve()
    if not path.is_file():
        raise LiveDormantPrefixScenarioError(
            f"TraceLab source does not exist: {path}")
    template, transformed_sha, transform, source_audit = (
        _materialized_template(path))
    service_audit = _prefill_service_audit(template)
    retained_max_sequence_tokens = max(
        call.input_tokens + call.output_tokens
        for call in template.calls
    )
    if (
        retained_max_sequence_tokens
        != EXPECTED_RETAINED_MAX_SEQUENCE_TOKENS
    ):
        raise LiveDormantPrefixScenarioError(
            "retained two-turn maximum sequence length changed")

    roles = (
        ("warmup", warmup_epoch_count),
        ("measurement", measurement_epoch_count),
        ("guard", guard_epoch_count),
    )
    epoch_groups = []
    mappings = []
    role_epoch_indices: dict[str, list[int]] = {
        role: [] for role, _ in roles
    }
    epoch_index = 0
    synthetic_source_index = 0
    for role, role_count in roles:
        for role_epoch_index in range(role_count):
            role_epoch_indices[role].append(epoch_index)
            group = []
            for clone_index in range(
                    COMPLETE_PREFIX_CLONES_PER_EPOCH):
                session, mapping = _clone_session(
                    template,
                    epoch_index=epoch_index,
                    role=role,
                    role_epoch_index=role_epoch_index,
                    clone_kind="complete_two_turn_prefix",
                    clone_index=clone_index,
                    synthetic_source_index=synthetic_source_index,
                )
                synthetic_source_index += 1
                group.append(session)
                mappings.append(mapping)
            for clone_index in range(
                    FIRST_ONLY_PREFIX_CLONES_PER_EPOCH):
                session, mapping = _clone_session(
                    template,
                    epoch_index=epoch_index,
                    role=role,
                    role_epoch_index=role_epoch_index,
                    clone_kind="first_turn_only_prefix",
                    clone_index=clone_index,
                    synthetic_source_index=synthetic_source_index,
                )
                synthetic_source_index += 1
                group.append(session)
                mappings.append(mapping)
            epoch_groups.append(tuple(group))
            epoch_index += 1

    first_group = epoch_groups[0]
    all_calls = tuple(
        call for session in first_group for call in session.calls)
    observed_counts = (
        len(first_group),
        len(all_calls),
        len(first_group),
        len(all_calls) - len(first_group),
        sum(call.output_tokens for call in all_calls),
        sum(call.input_tokens for call in all_calls),
        sum(call.cached_prefix_tokens for call in all_calls),
        sum(call.fresh_input_tokens for call in all_calls),
    )
    expected_counts = (
        EXPECTED_SESSIONS_PER_EPOCH,
        EXPECTED_CALLS_PER_EPOCH,
        EXPECTED_FIRST_CALLS_PER_EPOCH,
        EXPECTED_RESUME_CALLS_PER_EPOCH,
        EXPECTED_OUTPUT_TOKENS_PER_EPOCH,
        EXPECTED_INPUT_TOKENS_PER_EPOCH,
        EXPECTED_CACHED_PREFIX_TOKENS_PER_EPOCH,
        EXPECTED_FRESH_INPUT_TOKENS_PER_EPOCH,
    )
    if observed_counts != expected_counts:
        raise LiveDormantPrefixScenarioError(
            "pinned synthetic epoch counts changed: "
            f"observed={observed_counts}, expected={expected_counts}")

    dormant_tokens = (
        template.calls[0].input_tokens
        + template.calls[0].output_tokens
    )
    dormant_bytes = _block_rounded_kv_bytes(dormant_tokens)
    byte_ns_per_epoch = (
        dormant_bytes
        * EXPECTED_RECORDED_GAP_NS
        * COMPLETE_PREFIX_CLONES_PER_EPOCH
    )
    if (
        dormant_tokens
        != EXPECTED_DORMANT_LOGICAL_KV_TOKENS_PER_COMPLETE_CLONE
        or dormant_bytes
        != EXPECTED_DORMANT_LOGICAL_KV_BYTES_PER_COMPLETE_CLONE
        or byte_ns_per_epoch
        != EXPECTED_RECORDED_GAP_LOGICAL_KV_BYTE_NS_PER_EPOCH
    ):
        raise LiveDormantPrefixScenarioError(
            "pinned dormant logical-KV byte-gap audit changed")

    profile = _epoch_profile(
        warmup_epoch_count,
        measurement_epoch_count,
        guard_epoch_count,
    )
    schedule_coverage = _schedule_coverage_audit(
        epoch_groups,
        warmup_epoch_count=warmup_epoch_count,
        measurement_epoch_count=measurement_epoch_count,
        guard_epoch_count=guard_epoch_count,
        profile=profile,
    )
    finite_witness = _finite_pressure_witness(epoch_groups)
    if profile == "pressure" and not (
        finite_witness.exceeds_baseline_capacity
        and finite_witness.fits_smallest_hbf_capacity
    ):
        raise LiveDormantPrefixScenarioError(
            "finite pressure witness no longer separates baseline and HBF "
            "storage capacities")

    measurement_epoch_set = set(role_epoch_indices["measurement"])
    measurement_ids = tuple(
        mapping.session_id
        for mapping in mappings
        if mapping.epoch_index in measurement_epoch_set
    )
    measurement_request_count = sum(
        len(session.calls)
        for index, group in enumerate(epoch_groups)
        if index in measurement_epoch_set
        for session in group
    )
    measurement_first_count = (
        EXPECTED_SESSIONS_PER_EPOCH * measurement_epoch_count)
    measurement_resume_count = (
        measurement_request_count - measurement_first_count)
    mapping_payload = [asdict(mapping) for mapping in mappings]
    manifest = LiveDormantPrefixManifest(
        schema_version=SCENARIO_SCHEMA_VERSION,
        scenario_id=SCENARIO_ID,
        epoch_profile=profile,
        source_sha256=TRACELAB_SCHEMA3_SHA256,
        transformed_cohort_sha256=transformed_sha,
        selected_source_indices=(SOURCE_INDEX,),
        selected_source_session_ids=(SOURCE_SESSION_ID,),
        target_max_sequence_tokens=TARGET_MAX_SEQUENCE_TOKENS,
        retained_max_sequence_tokens=retained_max_sequence_tokens,
        context_factor_numerator=int(
            transform["global_factor_numerator"]),
        context_factor_denominator=int(
            transform["global_factor_denominator"]),
        epoch_count=len(epoch_groups),
        warmup_epochs=tuple(role_epoch_indices["warmup"]),
        measurement_epochs=tuple(role_epoch_indices["measurement"]),
        guard_epochs=tuple(role_epoch_indices["guard"]),
        sessions_per_epoch=EXPECTED_SESSIONS_PER_EPOCH,
        calls_per_epoch=EXPECTED_CALLS_PER_EPOCH,
        first_calls_per_epoch=EXPECTED_FIRST_CALLS_PER_EPOCH,
        resume_calls_per_epoch=EXPECTED_RESUME_CALLS_PER_EPOCH,
        output_tokens_per_epoch=EXPECTED_OUTPUT_TOKENS_PER_EPOCH,
        measurement_session_ids=measurement_ids,
        measurement_request_count=measurement_request_count,
        measurement_first_call_count=measurement_first_count,
        measurement_resume_call_count=measurement_resume_count,
        epoch_mapping_sha256=stable_json_sha256(mapping_payload),
        source_audit=source_audit,
        prefill_service=service_audit,
        kv_pressure=KVByteGapCapacityAudit(
            logical_kv_bytes_per_token=LOGICAL_KV_BYTES_PER_TOKEN,
            block_tokens=KV_BLOCK_TOKENS,
            dormant_logical_kv_tokens_per_complete_clone=dormant_tokens,
            dormant_logical_kv_bytes_per_complete_clone=dormant_bytes,
            recorded_gap_ns=EXPECTED_RECORDED_GAP_NS,
            complete_prefix_clones_per_epoch=(
                COMPLETE_PREFIX_CLONES_PER_EPOCH),
            recorded_gap_logical_kv_byte_ns_per_epoch=byte_ns_per_epoch,
            baseline_combined_usable_d_hbm_and_cpu_bytes=(
                BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES),
            hbf_tp4_usable_logical_kv_bytes=(
                HBF_TP4_USABLE_LOGICAL_KV_BYTES),
            hbf_tp8_usable_logical_kv_bytes=(
                HBF_TP8_USABLE_LOGICAL_KV_BYTES),
            hbf_tp8_context_usable_logical_kv_bytes=(
                HBF_TP8_CONTEXT_USABLE_LOGICAL_KV_BYTES),
            hbf_config_path=str(_HBF_CONFIG_PATH.relative_to(
                Path(__file__).resolve().parents[2])),
            hbf_config_sha256=_sha256_file(_HBF_CONFIG_PATH),
            analytical_steady_state_estimates=(
                _steady_state_estimates(byte_ns_per_epoch)),
            finite_pressure_witness=finite_witness,
            analytical_semantics=(
                "Little's law for the synthetic uniform epoch mix: offered "
                "session rate / 8 multiplied by five clones' block-rounded "
                "logical-KV byte-nanoseconds over the unchanged recorded "
                "gap; active service, queues, and transfers are excluded"),
            capacity_semantics=(
                "baseline is combined usable D-HBM plus host DRAM for two "
                "servers; HBF capacities reserve exact modeled Qwen weights, "
                "and conventional TP8 charges two physical KV copies while "
                "TP4 duplicates only weights across its two replicas"),
            realized_runtime_semantics=(
                "this low-churn sensitivity is a capacity-onset control, "
                "not evidence of SSD/PCIe bandwidth collapse or headline "
                "goodput separation; analytical estimates do not prove SSD "
                "traffic or tier occupancy, so accept those claims only "
                "from per-cell runtime reports and physical/logical HBF "
                "accounting"),
        ),
        schedule_coverage=schedule_coverage,
        recommended_rates=RECOMMENDED_RATES,
        recommended_seeds=RECOMMENDED_SEEDS,
        rate_semantics=(
            "system-wide external session starts per second; five of every "
            "eight sessions have a delayed resume, so complete-prefix start "
            "rate is five eighths of the offered session rate"),
        workload_semantics=(
            "explicitly synthetic two-turn prefix sensitivity: one five-call "
            "TraceLab session is globally scaled until its discarded suffix "
            "reaches 250k, while the retained prefix reaches 195151 tokens; "
            "it is truncated after call 1 and replicated 5:3 for service "
            "balance. It is a capacity-onset control, not a bandwidth-queue "
            "or headline-goodput workload, and is neither an empirical "
            "context distribution nor a complete-session TraceLab "
            "evaluation"),
        successor_release_semantics=(
            "each complete clone's call 1 is released only after call 0 "
            "completion plus the unchanged recorded 14,100.92-second human "
            "gap; first-only and truncated terminal calls release nothing"),
        measurement_semantics=(
            "only sessions in the explicit measurement epochs contribute to "
            "headline metrics; warmup and guard sessions maintain the finite "
            "arrival history, and every system fully drains"),
    )
    return LiveDormantPrefixScenario(
        manifest=manifest,
        epoch_sessions=tuple(epoch_groups),
    )


def build_pressure(
        trace_path: str | Path,
) -> LiveDormantPrefixScenario:
    """Pressure-profile factory for the live ASTRA comparison runner."""

    return build_live_dormant_prefix_scenario(
        trace_path,
        warmup_epoch_count=PRESSURE_WARMUP_EPOCH_COUNT,
        measurement_epoch_count=PRESSURE_MEASUREMENT_EPOCH_COUNT,
        guard_epoch_count=PRESSURE_GUARD_EPOCH_COUNT,
    )


def build_smoke(
        trace_path: str | Path,
) -> LiveDormantPrefixScenario:
    """Minimal integration smoke; it is not a storage-pressure claim."""

    return build_live_dormant_prefix_scenario(
        trace_path,
        warmup_epoch_count=SMOKE_WARMUP_EPOCH_COUNT,
        measurement_epoch_count=SMOKE_MEASUREMENT_EPOCH_COUNT,
        guard_epoch_count=SMOKE_GUARD_EPOCH_COUNT,
    )


def build_protocol_smoke(
        trace_path: str | Path,
) -> LiveDormantPrefixScenario:
    """Smallest full first/resume protocol smoke; not a pressure claim."""

    return build_live_dormant_prefix_scenario(
        trace_path,
        warmup_epoch_count=PROTOCOL_SMOKE_WARMUP_EPOCH_COUNT,
        measurement_epoch_count=PROTOCOL_SMOKE_MEASUREMENT_EPOCH_COUNT,
        guard_epoch_count=PROTOCOL_SMOKE_GUARD_EPOCH_COUNT,
    )


def build(trace_path: str | Path) -> LiveDormantPrefixScenario:
    """Default scenario-factory entry point (the pressure profile)."""

    return build_pressure(trace_path)


__all__ = [
    "LiveDormantPrefixManifest",
    "LiveDormantPrefixScenario",
    "LiveDormantPrefixScenarioError",
    "RECOMMENDED_PILOT_RATES",
    "RECOMMENDED_RATES",
    "RECOMMENDED_SEEDS",
    "build",
    "build_live_dormant_prefix_scenario",
    "build_pressure",
    "build_protocol_smoke",
    "build_smoke",
]

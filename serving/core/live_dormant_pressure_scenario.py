"""Trace-faithful dormant-session pressure scenario for live ASTRA sweeps.

This scenario keeps eight complete TraceLab sessions, including every recorded
turn and external gap.  The audited context transform scales prompt and prefix
coordinates to a 250k-token maximum while leaving outputs, call ordering, and
tool durations unchanged.  The selected mix has nearly equal aggregate
first-prefill and resume-prefill service in the central H100 TP4 model.

The manifest reports a Little's-law estimate of logical KV held specifically
during recorded external gaps.  That estimate is not a claim about a finite
simulation's realized tier occupancy.  Realized HBM, host-DRAM, SSD, HBF, and
LPDDR pressure must be read from the runtime report of each simulated system.
Because retaining all 28 calls also retains 4,611 generated tokens per epoch,
the publication factory is an expensive trace-faithful control.  A separately
preregistered shorter sensitivity should be used for the primary broad sweep.
The 32/4/32 pilot is still large enough that its pinned seed-101, rate-0.12
zero-service interval sweep crosses baseline HBM-plus-host capacity; the
1/1/1 smoke is only a protocol check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json
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
from .online_latency_model import (
    H100_QWEN3_TP4_KERNEL_CALIBRATED,
    resolve_online_latency_model,
)
from .utils import get_config


SCENARIO_SCHEMA_VERSION = 1
SCENARIO_ID = "tracelab-250k-complete-dormant-pressure-v1"
TARGET_MAX_SEQUENCE_TOKENS = 250_000
SELECTED_SOURCE_INDICES = (
    523, 610, 691, 1500, 2257, 3320, 3321, 3809,
)
EXPECTED_TRANSFORMED_COHORT_SHA256 = (
    "d6accbe259424a595c6b778575d055f37802ece47dcb8f30e0d25d9d397a6781"
)
EXPECTED_CONTEXT_FACTOR_NUMERATOR = 49_960
EXPECTED_CONTEXT_FACTOR_DENOMINATOR = 6_151
EXPECTED_SOURCE_IDENTITY_SHA256 = (
    "b8ff70723607b3d28523d124f0b21850265e729affee8a71ff1b0ed2a9ea92e1"
)
EXPECTED_RECORDED_GAPS_SHA256 = (
    "bf5db34964bd6f7725355545774e7b1213caff08a3950190a7a1cc0888ca5fb0"
)

DEFAULT_WARMUP_EPOCH_COUNT = 72
DEFAULT_MEASUREMENT_EPOCH_COUNT = 8
DEFAULT_GUARD_EPOCH_COUNT = 72
PILOT_WARMUP_EPOCH_COUNT = 32
PILOT_MEASUREMENT_EPOCH_COUNT = 4
PILOT_GUARD_EPOCH_COUNT = 32
SMOKE_WARMUP_EPOCH_COUNT = 1
SMOKE_MEASUREMENT_EPOCH_COUNT = 1
SMOKE_GUARD_EPOCH_COUNT = 1
FINITE_PRESSURE_WITNESS_SEED = 101
FINITE_PRESSURE_WITNESS_RATE = 0.12
EXPECTED_PILOT_ZERO_SERVICE_PEAK_LOGICAL_KV_BYTES = 1_739_935_186_944

EXPECTED_SESSIONS_PER_EPOCH = 8
EXPECTED_CALLS_PER_EPOCH = 28
EXPECTED_FIRST_CALLS_PER_EPOCH = 8
EXPECTED_RESUME_CALLS_PER_EPOCH = 20
EXPECTED_OUTPUT_TOKENS_PER_EPOCH = 4_611
EXPECTED_INPUT_TOKENS_PER_EPOCH = 3_980_420
EXPECTED_CACHED_PREFIX_TOKENS_PER_EPOCH = 3_293_895
EXPECTED_FRESH_INPUT_TOKENS_PER_EPOCH = 686_525
EXPECTED_POSITIVE_RECORDED_GAPS_PER_EPOCH = 20
EXPECTED_RECORDED_GAP_NS_PER_EPOCH = 6_187_282_000_000
EXPECTED_MAX_RECORDED_GAP_NS = 3_898_298_000_000

EXPECTED_FIRST_PREFILL_SERVICE_NS = 50_568_365_667
EXPECTED_RESUME_PREFILL_SERVICE_NS = 52_842_727_754

LOGICAL_KV_BYTES_PER_TOKEN = 98_304
KV_BLOCK_TOKENS = 16
EXPECTED_RECORDED_GAP_LOGICAL_KV_BYTE_NS_PER_EPOCH = (
    114_057_310_892_457_984_000_000
)
EXPECTED_TERMINAL_LOGICAL_KV_BYTES_PER_EPOCH = 67_947_724_800
BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES = 1_382_573_883_392

RECOMMENDED_RATES = (0.02, 0.05, 0.08, 0.10, 0.12)
RECOMMENDED_PILOT_RATES = (0.02, 0.08, 0.12)

_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class LiveDormantPressureScenarioError(ValueError):
    """Raised when the pinned complete-session scenario drifts."""


@dataclass(frozen=True)
class DormantEpochMapping:
    epoch_index: int
    role: str
    role_epoch_index: int
    session_id: str
    synthetic_source_index: int
    source_index: int
    source_session_id: str


@dataclass(frozen=True)
class CompleteCohortAudit:
    sessions_per_epoch: int
    calls_per_epoch: int
    first_calls_per_epoch: int
    resume_calls_per_epoch: int
    output_tokens_per_epoch: int
    input_tokens_per_epoch: int
    cached_prefix_tokens_per_epoch: int
    fresh_input_tokens_per_epoch: int
    positive_recorded_gaps_per_epoch: int
    recorded_gap_ns_per_epoch: int
    max_recorded_gap_ns: int
    source_identity_sha256: str
    recorded_gaps_sha256: str
    completeness_semantics: str
    transform_semantics: str


@dataclass(frozen=True)
class PrefillServiceBalanceAudit:
    first_prefill_service_ns_per_epoch: int
    resume_prefill_service_ns_per_epoch: int
    resume_to_first_service_ratio: float
    model: str
    topology: str
    latency_band: str
    method: str


@dataclass(frozen=True)
class SteadyStateKVEstimate:
    offered_session_rate_per_second: float
    analytical_recorded_gap_live_kv_bytes_floor: int


@dataclass(frozen=True)
class FiniteSchedulePressureWitness:
    seed: int
    offered_session_rate_per_second: float
    epoch_profile: str
    zero_service_peak_recorded_gap_logical_kv_bytes: int
    baseline_combined_usable_d_hbm_and_cpu_bytes: int
    exceeds_baseline_capacity: bool
    semantics: str


@dataclass(frozen=True)
class DormantKVPressureAudit:
    logical_kv_bytes_per_token: int
    block_tokens: int
    recorded_gap_logical_kv_byte_ns_per_epoch: int
    terminal_logical_kv_bytes_per_epoch: int
    baseline_combined_usable_d_hbm_and_cpu_bytes: int
    analytical_steady_state_estimates: tuple[SteadyStateKVEstimate, ...]
    analytical_semantics: str
    finite_schedule_semantics: str
    realized_runtime_semantics: str
    finite_schedule_witness: FiniteSchedulePressureWitness | None = None


@dataclass(frozen=True)
class LiveDormantPressureManifest:
    schema_version: int
    scenario_id: str
    epoch_profile: str
    source_sha256: str
    transformed_cohort_sha256: str
    selected_source_indices: tuple[int, ...]
    selected_source_session_ids: tuple[str, ...]
    target_max_sequence_tokens: int
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
    cohort_audit: CompleteCohortAudit
    prefill_service: PrefillServiceBalanceAudit
    kv_pressure: DormantKVPressureAudit
    recommended_rates: tuple[float, ...]
    rate_semantics: str
    workload_semantics: str
    evaluation_role_semantics: str
    successor_release_semantics: str
    measurement_semantics: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LiveDormantPressureScenario:
    manifest: LiveDormantPressureManifest
    epoch_sessions: tuple[tuple[SessionSpec, ...], ...]

    def build_offered_plan(self, *, seed: int):
        return _build_epoch_offered_plan(self.epoch_sessions, seed=seed)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection() -> dict[str, object]:
    return {
        "strategy": "all",
        "include_source_indices": list(SELECTED_SOURCE_INDICES),
        "max_sessions": len(SELECTED_SOURCE_INDICES),
        "target_max_sequence_tokens": TARGET_MAX_SEQUENCE_TOKENS,
    }


def _gap_payload(
        rows: Sequence[Mapping[str, object]],
        *,
        materialized: bool,
) -> list[dict[str, object]]:
    payload = []
    for fallback_index, row in enumerate(rows):
        if materialized:
            source = row["online_experiment_source"]
            source_index = int(source["source_index"])
        else:
            source_index = SELECTED_SOURCE_INDICES[fallback_index]
        for call_index, call in enumerate(row["sub_requests"]):
            payload.append({
                "source_index": source_index,
                "call_index": call_index,
                "tool_duration_ns": int(call["tool_duration_ns"]),
                "inter_turn_gap_type": call.get("inter_turn_gap_type"),
            })
    return payload


def _selected_source_rows(trace_path: Path) -> tuple[dict[str, object], ...]:
    selected = set(SELECTED_SOURCE_INDICES)
    rows = []
    with trace_path.open("r", encoding="utf-8") as source:
        for source_index, line in enumerate(source):
            if source_index not in selected:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise LiveDormantPressureScenarioError(
                    f"TraceLab source row {source_index} is not an object")
            rows.append(row)
    if len(rows) != len(SELECTED_SOURCE_INDICES):
        raise LiveDormantPressureScenarioError(
            "pinned source rows are missing from TraceLab")
    return tuple(rows)


def _materialized_templates(
        trace_path: Path,
) -> tuple[tuple[SessionSpec, ...], str, dict[str, object], CompleteCohortAudit]:
    if _sha256_file(trace_path) != TRACELAB_SCHEMA3_SHA256:
        raise LiveDormantPressureScenarioError(
            "TraceLab source SHA-256 does not match the pinned schema-3 trace")
    source_rows = _selected_source_rows(trace_path)
    source_gap_payload = _gap_payload(source_rows, materialized=False)

    from serving.online_experiments import materialize_session_cohort

    with tempfile.TemporaryDirectory(
            prefix="llmsim-live-dormant-cohort-") as temporary:
        descriptor = materialize_session_cohort(
            trace_path,
            Path(temporary),
            _selection(),
        )
        transformed_path = Path(descriptor["materialized_path"])
        transformed_sha = _sha256_file(transformed_path)
        if transformed_sha != EXPECTED_TRANSFORMED_COHORT_SHA256:
            raise LiveDormantPressureScenarioError(
                "250k transformed complete TraceLab cohort changed: "
                f"observed={transformed_sha}, "
                f"expected={EXPECTED_TRANSFORMED_COHORT_SHA256}")
        rows = tuple(
            json.loads(line)
            for line in transformed_path.read_text(
                encoding="utf-8").splitlines()
            if line
        )

    materialized_gap_payload = _gap_payload(rows, materialized=True)
    if materialized_gap_payload != source_gap_payload:
        raise LiveDormantPressureScenarioError(
            "context materialization changed a recorded TraceLab gap")
    recorded_gaps_sha = stable_json_sha256(materialized_gap_payload)
    if recorded_gaps_sha != EXPECTED_RECORDED_GAPS_SHA256:
        raise LiveDormantPressureScenarioError(
            "pinned TraceLab recorded-gap audit changed")

    templates = []
    for row in rows:
        source = row["online_experiment_source"]
        source_index = int(source["source_index"])
        session_id = str(row["session_id"])
        trace_metadata = row.get("trace_metadata") or {}
        calls = []
        for call_index, call in enumerate(row["sub_requests"]):
            input_tokens = int(call["input_toks"])
            cached_tokens = int(call["prefix_reuse_toks"])
            if not 0 <= cached_tokens <= input_tokens:
                raise LiveDormantPressureScenarioError(
                    "materialized cached prefix is outside its prompt")
            calls.append(CallSpec(
                session_id=session_id,
                source_index=source_index,
                call_index=call_index,
                input_tokens=input_tokens,
                output_tokens=int(call["output_toks"]),
                tool_duration_ns=int(call["tool_duration_ns"]),
                cached_prefix_tokens=cached_tokens,
                fresh_input_tokens=input_tokens - cached_tokens,
                lineage_status=call.get("lineage_status"),
                inter_turn_gap_type=call.get("inter_turn_gap_type"),
            ))
        if not calls:
            raise LiveDormantPressureScenarioError(
                "selected complete session has no calls")
        templates.append(SessionSpec(
            source_index=source_index,
            session_id=session_id,
            source_arrival_time_ns=int(row.get("arrival_time_ns", 0)),
            source_session_identity_sha256=(
                trace_metadata.get("source_session_identity_sha256")),
            calls=tuple(calls),
        ))

    observed_indices = tuple(template.source_index for template in templates)
    if observed_indices != SELECTED_SOURCE_INDICES:
        raise LiveDormantPressureScenarioError(
            "selected TraceLab source order changed: "
            f"observed={observed_indices}, "
            f"expected={SELECTED_SOURCE_INDICES}")
    source_identity_sha = stable_json_sha256([
        (template.source_index, template.session_id)
        for template in templates
    ])
    if source_identity_sha != EXPECTED_SOURCE_IDENTITY_SHA256:
        raise LiveDormantPressureScenarioError(
            "pinned TraceLab source identities changed")

    transform = dict(descriptor["context_length_transform"])
    if (
        transform.get("global_factor_numerator")
        != EXPECTED_CONTEXT_FACTOR_NUMERATOR
        or transform.get("global_factor_denominator")
        != EXPECTED_CONTEXT_FACTOR_DENOMINATOR
        or transform.get("realized_max_sequence_tokens")
        != TARGET_MAX_SEQUENCE_TOKENS
    ):
        raise LiveDormantPressureScenarioError(
            "pinned 250k context transform changed")

    calls = tuple(call for template in templates for call in template.calls)
    counts = (
        len(templates),
        len(calls),
        len(templates),
        len(calls) - len(templates),
        sum(call.output_tokens for call in calls),
        sum(call.input_tokens for call in calls),
        sum(call.cached_prefix_tokens for call in calls),
        sum(call.fresh_input_tokens for call in calls),
        sum(call.tool_duration_ns > 0 for call in calls),
        sum(call.tool_duration_ns for call in calls),
        max(call.tool_duration_ns for call in calls),
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
        EXPECTED_POSITIVE_RECORDED_GAPS_PER_EPOCH,
        EXPECTED_RECORDED_GAP_NS_PER_EPOCH,
        EXPECTED_MAX_RECORDED_GAP_NS,
    )
    if counts != expected_counts:
        raise LiveDormantPressureScenarioError(
            "pinned complete TraceLab cohort changed: "
            f"observed={counts}, expected={expected_counts}")
    cohort_audit = CompleteCohortAudit(
        sessions_per_epoch=len(templates),
        calls_per_epoch=len(calls),
        first_calls_per_epoch=len(templates),
        resume_calls_per_epoch=len(calls) - len(templates),
        output_tokens_per_epoch=sum(call.output_tokens for call in calls),
        input_tokens_per_epoch=sum(call.input_tokens for call in calls),
        cached_prefix_tokens_per_epoch=sum(
            call.cached_prefix_tokens for call in calls),
        fresh_input_tokens_per_epoch=sum(
            call.fresh_input_tokens for call in calls),
        positive_recorded_gaps_per_epoch=sum(
            call.tool_duration_ns > 0 for call in calls),
        recorded_gap_ns_per_epoch=sum(
            call.tool_duration_ns for call in calls),
        max_recorded_gap_ns=max(call.tool_duration_ns for call in calls),
        source_identity_sha256=source_identity_sha,
        recorded_gaps_sha256=recorded_gaps_sha,
        completeness_semantics=(
            "all source sessions and every source sub-request are retained; "
            "natural one-call sessions remain complete one-call sessions"),
        transform_semantics=(
            "one global rational factor scales prompt and prefix coordinates; "
            "outputs, recorded tool durations, lineage labels, and call order "
            "are byte-for-byte source values"),
    )
    return tuple(templates), transformed_sha, transform, cohort_audit


def _latency_provider():
    repo_root = Path(__file__).resolve().parents[2]
    config = get_config(_MODEL_NAME)
    return resolve_online_latency_model(
        name=H100_QWEN3_TP4_KERNEL_CALIBRATED,
        repo_root=repo_root,
        hardware="H100",
        model=_MODEL_NAME,
        config=config,
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
        templates: Sequence[SessionSpec],
) -> PrefillServiceBalanceAudit:
    provider = _latency_provider()
    first_ns = 0
    resume_ns = 0
    for template in templates:
        for call in template.calls:
            hit_tokens = (
                0 if call.is_first_turn
                else min(call.cached_prefix_tokens, call.input_tokens - 1)
            )
            service_ns = provider.singleton_prefill_comp_ns(
                input_tokens=call.input_tokens,
                hit_tokens=hit_tokens,
                max_chunk_tokens=131_072,
            )
            if call.is_first_turn:
                first_ns += service_ns
            else:
                resume_ns += service_ns
    observed = (first_ns, resume_ns)
    expected = (
        EXPECTED_FIRST_PREFILL_SERVICE_NS,
        EXPECTED_RESUME_PREFILL_SERVICE_NS,
    )
    if observed != expected:
        raise LiveDormantPressureScenarioError(
            "pinned first/resume prefill service audit changed: "
            f"observed={observed}, expected={expected}")
    return PrefillServiceBalanceAudit(
        first_prefill_service_ns_per_epoch=first_ns,
        resume_prefill_service_ns_per_epoch=resume_ns,
        resume_to_first_service_ratio=resume_ns / first_ns,
        model=_MODEL_NAME,
        topology="one_tp4_h100_model_partition",
        latency_band="central",
        method=(
            "sum_of_singleton_prefill_COMP_critical_paths_with_131072_"
            "token_chunks_no_collectives_no_queueing"),
    )


def _block_rounded_kv_bytes(token_count: int) -> int:
    blocks = (
        token_count + KV_BLOCK_TOKENS - 1
    ) // KV_BLOCK_TOKENS
    return blocks * KV_BLOCK_TOKENS * LOGICAL_KV_BYTES_PER_TOKEN


def _kv_pressure_audit(
        templates: Sequence[SessionSpec],
) -> DormantKVPressureAudit:
    byte_ns = 0
    terminal_bytes = 0
    for template in templates:
        for call in template.calls[:-1]:
            byte_ns += (
                _block_rounded_kv_bytes(
                    call.input_tokens + call.output_tokens)
                * call.tool_duration_ns
            )
        final = template.calls[-1]
        terminal_bytes += _block_rounded_kv_bytes(
            final.input_tokens + final.output_tokens)
    if (
        byte_ns
        != EXPECTED_RECORDED_GAP_LOGICAL_KV_BYTE_NS_PER_EPOCH
        or terminal_bytes
        != EXPECTED_TERMINAL_LOGICAL_KV_BYTES_PER_EPOCH
    ):
        raise LiveDormantPressureScenarioError(
            "pinned dormant logical-KV audit changed")

    estimates = []
    for rate in RECOMMENDED_RATES:
        rate_fraction = Fraction(str(rate))
        numerator = byte_ns * rate_fraction.numerator
        denominator = (
            EXPECTED_SESSIONS_PER_EPOCH
            * 1_000_000_000
            * rate_fraction.denominator
        )
        estimates.append(SteadyStateKVEstimate(
            offered_session_rate_per_second=rate,
            analytical_recorded_gap_live_kv_bytes_floor=(
                numerator // denominator),
        ))
    return DormantKVPressureAudit(
        logical_kv_bytes_per_token=LOGICAL_KV_BYTES_PER_TOKEN,
        block_tokens=KV_BLOCK_TOKENS,
        recorded_gap_logical_kv_byte_ns_per_epoch=byte_ns,
        terminal_logical_kv_bytes_per_epoch=terminal_bytes,
        baseline_combined_usable_d_hbm_and_cpu_bytes=(
            BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES),
        analytical_steady_state_estimates=tuple(estimates),
        analytical_semantics=(
            "Little's law for an unbounded stationary uniform mix: offered "
            "session rate / 8 multiplied by block-rounded logical-KV "
            "byte-nanoseconds held during unchanged recorded external gaps; "
            "it excludes active service, queueing, migrations, and terminal "
            "KV after the final call"),
        finite_schedule_semantics=(
            "the finite Poisson schedule, dependency completion times, and "
            "warmup length can differ from the stationary mean; this field is "
            "an analytical pressure target, not observed occupancy"),
        realized_runtime_semantics=(
            "realized tier pressure is accepted only from each cell's "
            "runtime report, including peak tier bytes, SSD reads/writes, "
            "resource queues, HBF physical/logical bytes, and LPDDR peaks"),
    )


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
        DEFAULT_WARMUP_EPOCH_COUNT,
        DEFAULT_MEASUREMENT_EPOCH_COUNT,
        DEFAULT_GUARD_EPOCH_COUNT,
    ):
        return "publication"
    if counts == (
        PILOT_WARMUP_EPOCH_COUNT,
        PILOT_MEASUREMENT_EPOCH_COUNT,
        PILOT_GUARD_EPOCH_COUNT,
    ):
        return "pilot"
    if counts == (
        SMOKE_WARMUP_EPOCH_COUNT,
        SMOKE_MEASUREMENT_EPOCH_COUNT,
        SMOKE_GUARD_EPOCH_COUNT,
    ):
        return "smoke"
    return "custom"


def _evaluation_role_semantics(profile: str) -> str:
    if profile == "publication":
        return (
            "expensive trace-faithful validation/control with 4611 generated "
            "tokens per epoch; not the primary runtime-efficient broad-sweep "
            "sensitivity")
    if profile == "pilot":
        return (
            "finite pressure pilot whose pinned seed-101 rate-0.12 "
            "zero-service schedule exceeds baseline HBM-plus-host capacity; "
            "realized SSD pressure still requires runtime-report validation")
    if profile == "smoke":
        return (
            "minimal protocol smoke only; its finite schedule is not expected "
            "to create baseline SSD pressure")
    return (
        "custom complete-session control; its finite schedule must be audited "
        "before making a tier-pressure claim")


def _finite_schedule_pressure_witness(
        epoch_sessions: Sequence[Sequence[SessionSpec]],
        *,
        profile: str,
) -> FiniteSchedulePressureWitness:
    scheduled = _build_epoch_offered_plan(
        epoch_sessions,
        seed=FINITE_PRESSURE_WITNESS_SEED,
    ).at_rate(FINITE_PRESSURE_WITNESS_RATE)
    events = []
    for item in scheduled:
        boundary_time_ns = item.arrival_time_ns
        for call in item.session.calls[:-1]:
            next_boundary_time_ns = (
                boundary_time_ns + call.tool_duration_ns)
            if next_boundary_time_ns > boundary_time_ns:
                logical_kv_bytes = _block_rounded_kv_bytes(
                    call.input_tokens + call.output_tokens)
                # End events sort before start events at a shared timestamp.
                events.append((
                    boundary_time_ns, 1, logical_kv_bytes))
                events.append((
                    next_boundary_time_ns, 0, -logical_kv_bytes))
            boundary_time_ns = next_boundary_time_ns

    live_bytes = 0
    peak_bytes = 0
    for _, _, delta_bytes in sorted(events):
        live_bytes += delta_bytes
        if live_bytes < 0:
            raise LiveDormantPressureScenarioError(
                "finite dormant-KV interval accounting became negative")
        peak_bytes = max(peak_bytes, live_bytes)
    if live_bytes != 0:
        raise LiveDormantPressureScenarioError(
            "finite dormant-KV interval accounting did not drain")
    if (
        profile == "pilot"
        and peak_bytes
        != EXPECTED_PILOT_ZERO_SERVICE_PEAK_LOGICAL_KV_BYTES
    ):
        raise LiveDormantPressureScenarioError(
            "pinned finite pressure-pilot witness changed: "
            f"observed={peak_bytes}, "
            "expected="
            f"{EXPECTED_PILOT_ZERO_SERVICE_PEAK_LOGICAL_KV_BYTES}")
    return FiniteSchedulePressureWitness(
        seed=FINITE_PRESSURE_WITNESS_SEED,
        offered_session_rate_per_second=FINITE_PRESSURE_WITNESS_RATE,
        epoch_profile=profile,
        zero_service_peak_recorded_gap_logical_kv_bytes=peak_bytes,
        baseline_combined_usable_d_hbm_and_cpu_bytes=(
            BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES),
        exceeds_baseline_capacity=(
            peak_bytes
            > BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES),
        semantics=(
            "exact interval sweep of the finite offered schedule with LLM "
            "service, queueing, migration, and transfer time set to zero; "
            "this is a deterministic storage-pressure witness, not realized "
            "tier occupancy"),
    )


def _clone_session(
        source: SessionSpec,
        *,
        epoch_index: int,
        role: str,
        role_epoch_index: int,
        synthetic_source_index: int,
) -> tuple[SessionSpec, DormantEpochMapping]:
    session_id = (
        f"{SCENARIO_ID}::{role}-{role_epoch_index:03d}"
        f"::epoch-{epoch_index:03d}::source-{source.source_index:04d}"
        f"::{source.session_id}"
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
        "epoch_index": epoch_index,
        "role": role,
        "role_epoch_index": role_epoch_index,
        "source_index": source.source_index,
        "source_session_id": source.session_id,
        "source_session_identity_sha256": (
            source.source_session_identity_sha256),
        "complete_session": True,
    })
    session = SessionSpec(
        source_index=synthetic_source_index,
        session_id=session_id,
        source_arrival_time_ns=source.source_arrival_time_ns,
        source_session_identity_sha256=identity,
        calls=calls,
    )
    return session, DormantEpochMapping(
        epoch_index=epoch_index,
        role=role,
        role_epoch_index=role_epoch_index,
        session_id=session_id,
        synthetic_source_index=synthetic_source_index,
        source_index=source.source_index,
        source_session_id=source.session_id,
    )


def build_live_dormant_pressure_scenario(
        trace_path: str | Path,
        *,
        warmup_epoch_count: int = DEFAULT_WARMUP_EPOCH_COUNT,
        measurement_epoch_count: int = DEFAULT_MEASUREMENT_EPOCH_COUNT,
        guard_epoch_count: int = DEFAULT_GUARD_EPOCH_COUNT,
) -> LiveDormantPressureScenario:
    """Build and fail-closed validate the complete dormant-session scenario."""

    warmup_epoch_count = _validate_epoch_count(
        warmup_epoch_count, "warmup_epoch_count", positive=False)
    measurement_epoch_count = _validate_epoch_count(
        measurement_epoch_count, "measurement_epoch_count", positive=True)
    guard_epoch_count = _validate_epoch_count(
        guard_epoch_count, "guard_epoch_count", positive=False)
    path = Path(trace_path).resolve()
    if not path.is_file():
        raise LiveDormantPressureScenarioError(
            f"TraceLab source does not exist: {path}")
    templates, transformed_sha, transform, cohort_audit = (
        _materialized_templates(path))
    service_audit = _prefill_service_audit(templates)
    kv_pressure = _kv_pressure_audit(templates)

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
            for source in templates:
                session, mapping = _clone_session(
                    source,
                    epoch_index=epoch_index,
                    role=role,
                    role_epoch_index=role_epoch_index,
                    synthetic_source_index=synthetic_source_index,
                )
                synthetic_source_index += 1
                group.append(session)
                mappings.append(mapping)
            epoch_groups.append(tuple(group))
            epoch_index += 1

    measurement_ids = tuple(
        mapping.session_id
        for mapping in mappings
        if mapping.role == "measurement"
    )
    measurement_request_count = (
        EXPECTED_CALLS_PER_EPOCH * measurement_epoch_count)
    measurement_first_count = (
        EXPECTED_FIRST_CALLS_PER_EPOCH * measurement_epoch_count)
    measurement_resume_count = (
        EXPECTED_RESUME_CALLS_PER_EPOCH * measurement_epoch_count)
    mapping_payload = [asdict(mapping) for mapping in mappings]
    profile = _epoch_profile(
        warmup_epoch_count,
        measurement_epoch_count,
        guard_epoch_count,
    )
    kv_pressure = replace(
        kv_pressure,
        finite_schedule_witness=_finite_schedule_pressure_witness(
            epoch_groups,
            profile=profile,
        ),
    )
    manifest = LiveDormantPressureManifest(
        schema_version=SCENARIO_SCHEMA_VERSION,
        scenario_id=SCENARIO_ID,
        epoch_profile=profile,
        source_sha256=TRACELAB_SCHEMA3_SHA256,
        transformed_cohort_sha256=transformed_sha,
        selected_source_indices=SELECTED_SOURCE_INDICES,
        selected_source_session_ids=tuple(
            template.session_id for template in templates),
        target_max_sequence_tokens=TARGET_MAX_SEQUENCE_TOKENS,
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
        cohort_audit=cohort_audit,
        prefill_service=service_audit,
        kv_pressure=kv_pressure,
        recommended_rates=RECOMMENDED_RATES,
        rate_semantics=(
            "system-wide external complete-session starts per second; each "
            "eight-session epoch contains the same pinned TraceLab mix"),
        workload_semantics=(
            "eight complete TraceLab sessions with all 28 calls and all "
            "recorded gaps retained, under one audited global 250k prompt/"
            "prefix sensitivity transform; not an empirical context-length "
            "distribution"),
        evaluation_role_semantics=_evaluation_role_semantics(profile),
        successor_release_semantics=(
            "successor call N+1 is released only after call N completion plus "
            "the unchanged recorded TraceLab tool duration"),
        measurement_semantics=(
            f"{warmup_epoch_count} complete epochs warm up, "
            f"{measurement_epoch_count} complete epochs form the exact fixed "
            f"measurement roster, {guard_epoch_count} complete epochs guard "
            "the tail, and every system fully drains"),
    )
    return LiveDormantPressureScenario(
        manifest=manifest,
        epoch_sessions=tuple(epoch_groups),
    )


def build(trace_path: str | Path) -> LiveDormantPressureScenario:
    """Publication-size scenario-factory entry point."""

    return build_live_dormant_pressure_scenario(trace_path)


def build_pilot(trace_path: str | Path) -> LiveDormantPressureScenario:
    """Finite pressure pilot with a pinned zero-service SSD-onset witness."""

    return build_live_dormant_pressure_scenario(
        trace_path,
        warmup_epoch_count=PILOT_WARMUP_EPOCH_COUNT,
        measurement_epoch_count=PILOT_MEASUREMENT_EPOCH_COUNT,
        guard_epoch_count=PILOT_GUARD_EPOCH_COUNT,
    )


def build_smoke(trace_path: str | Path) -> LiveDormantPressureScenario:
    """Minimal plumbing smoke with one epoch in each roster role."""

    return build_live_dormant_pressure_scenario(
        trace_path,
        warmup_epoch_count=SMOKE_WARMUP_EPOCH_COUNT,
        measurement_epoch_count=SMOKE_MEASUREMENT_EPOCH_COUNT,
        guard_epoch_count=SMOKE_GUARD_EPOCH_COUNT,
    )


__all__ = [
    "LiveDormantPressureManifest",
    "LiveDormantPressureScenario",
    "LiveDormantPressureScenarioError",
    "RECOMMENDED_PILOT_RATES",
    "RECOMMENDED_RATES",
    "build",
    "build_live_dormant_pressure_scenario",
    "build_pilot",
    "build_smoke",
]

"""Trace-faithful headline cohort for storage pressure before compute saturation.

Each epoch retains seven complete, unmodified TraceLab sessions.  Source 1741
contributes one first turn and seven resumes; six natural one-call sessions
make the epoch exactly 7:7 first/resume requests.  Those six sessions were
selected so the aggregate H100 TP4 singleton first-prefill service differs
from aggregate resume-prefill service by only seven nanoseconds.

The publication profile is a finite storage-pressure experiment, not a claim
that its Poisson warmup spans the complete 2,329.224-second gap for every
seed.  Its exact zero-service interval audit instead verifies that every
measurement resume returns while its sticky baseline node exceeds usable
D-HBM plus host-DRAM capacity at the highest selected rate.  Runtime SSD
traffic and queues remain authoritative.

The conventional ``tp8`` HBF layout is deliberately retained as a comparison
point, but its singleton service knee is slightly below the baseline storage
knee for this cohort because Qwen has four KV heads and conventional TP8
replicates each head across two ranks.  ``tp8_context`` removes that KV
replication, while TP4 uses two model-weight replicas and assigns each
session's single KV copy to only one replica.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Sequence

from .hbf_comparison_workload import (
    CallSpec,
    SessionSpec,
    TRACELAB_SCHEMA3_SHA256,
    build_offered_plan,
    load_comparison_workload,
    stable_json_sha256,
    summarize_sessions,
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
SCENARIO_ID = "tracelab-headline-1741-balanced-v1"

# The order is part of the scenario contract.  The first source is the
# complete multi-turn session; the remainder are complete natural one-call
# sessions, not truncated replicas.
SELECTED_SOURCE_INDICES = (
    1741,
    3320,
    1976,
    1964,
    592,
    1771,
    1791,
)
EXPECTED_SOURCE_SESSION_IDS = (
    "claude:1ad80c05-e6f6-90bc-5562-e38a80cbff17",
    "claude:2fd22673-a7bb-53a1-07eb-62687d2e7ff0",
    "claude:b893d9f3-aa2e-3de8-9e8c-72c8790d9e2a",
    "claude:607f5a07-d883-a426-22cb-6f3b617f5ddf",
    "claude:006bdbec-9a1e-8f89-340f-e5c2e5385e40",
    "claude:241eeedd-4dae-aede-da74-611361d8ddcb",
    "claude:0ffa18da-f080-9abc-f498-a4b96ee846d8",
)
EXPECTED_COMPLETE_COHORT_SHA256 = (
    "0fe2d41676e86793cdb45534ea01197e2267aa8df0501fbafd8d3e04ae492ded"
)
EXPECTED_SOURCE_IDENTITY_SHA256 = (
    "3ba8bce9ca4ebc941a95cc3dc48f08844cc23b5f2dc54431525e33acd7f9c0c0"
)
EXPECTED_RECORDED_GAPS_SHA256 = (
    "9229ac0d877e4fee71092d068d66bb7e31c608fd4eb6c45675a63f30ddf861df"
)

EXPECTED_SESSIONS_PER_EPOCH = 7
EXPECTED_CALLS_PER_EPOCH = 14
EXPECTED_FIRST_CALLS_PER_EPOCH = 7
EXPECTED_RESUME_CALLS_PER_EPOCH = 7
EXPECTED_OUTPUT_TOKENS_PER_EPOCH = 40
EXPECTED_INPUT_TOKENS_PER_EPOCH = 393_011
EXPECTED_CACHED_PREFIX_TOKENS_PER_EPOCH = 259_418
EXPECTED_FRESH_INPUT_TOKENS_PER_EPOCH = 133_593
EXPECTED_RESUME_FRESH_INPUT_TOKENS_PER_EPOCH = 36_279
EXPECTED_POSITIVE_RECORDED_GAPS_PER_EPOCH = 7
EXPECTED_RECORDED_GAP_NS_PER_EPOCH = 2_329_585_000_000
EXPECTED_MAX_RECORDED_GAP_NS = 2_329_224_000_000

EXPECTED_FIRST_PREFILL_SERVICE_NS = 2_403_264_714
EXPECTED_RESUME_PREFILL_SERVICE_NS = 2_403_264_721
EXPECTED_PREFILL_SERVICE_DIFFERENCE_NS = 7

LOGICAL_KV_BYTES_PER_TOKEN = 98_304
KV_BLOCK_TOKENS = 16
EXPECTED_RECORDED_GAP_LOGICAL_KV_BYTE_NS_PER_EPOCH = (
    9_885_331_625_607_168_000_000
)
EXPECTED_TERMINAL_LOGICAL_KV_BYTES_PER_EPOCH = 13_142_851_584

# Two baseline servers, each with one D-side usable HBM pool and host DRAM.
BASELINE_USABLE_D_HBM_AND_CPU_BYTES_PER_NODE = 691_286_941_696
BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES = (
    2 * BASELINE_USABLE_D_HBM_AND_CPU_BYTES_PER_NODE
)

RECOMMENDED_RATES = (0.25, 0.50, 0.80, 1.00, 1.20)
RECOMMENDED_SEEDS = (101, 102, 103, 104, 105)
PRESSURE_AUDIT_SEEDS = tuple(range(101, 113))
MAXIMUM_AUDITED_RATE = max(RECOMMENDED_RATES)

PUBLICATION_WARMUP_EPOCH_COUNT = 380
PUBLICATION_MEASUREMENT_EPOCH_COUNT = 16
PUBLICATION_GUARD_EPOCH_COUNT = 380
PROTOCOL_SMOKE_WARMUP_EPOCH_COUNT = 0
PROTOCOL_SMOKE_MEASUREMENT_EPOCH_COUNT = 1
PROTOCOL_SMOKE_GUARD_EPOCH_COUNT = 0

EXPECTED_PUBLICATION_SEED101_NODE_PEAK_BYTES = (
    895_758_630_912,
    895_758_630_912,
)
EXPECTED_PUBLICATION_SEED101_AGGREGATE_PEAK_BYTES = 1_748_719_632_384
EXPECTED_PUBLICATION_MEASUREMENT_RESUME_RETURNS = 112
EXPECTED_PUBLICATION_MEASUREMENT_LONG_GAP_RETURNS = 16

_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_CLUSTER_CONFIG_RELATIVE = Path(
    "configs/cluster/dual_node_qwen3_1m_pd_p4d4_h100.json"
)
_BASELINE_CLUSTER_CONFIG_PATH = (
    _REPO_ROOT / _BASELINE_CLUSTER_CONFIG_RELATIVE
)
_HBF_CONFIG_PATH = (
    _REPO_ROOT / "configs/wakekv_hbf/full_model_8card_server.json"
)


class LiveBalancedStorageScenarioError(ValueError):
    """Raised when the pinned complete-session headline cohort drifts."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_baseline_usable_bytes_per_node(
        path: Path = _BASELINE_CLUSTER_CONFIG_PATH,
) -> tuple[int, str]:
    """Derive and pin the baseline's pre-SSD KV capacity per physical node."""

    try:
        payload = path.read_bytes()
        raw = json.loads(payload)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LiveBalancedStorageScenarioError(
            f"unable to load baseline cluster config {path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise LiveBalancedStorageScenarioError(
            "baseline cluster config must be a JSON object")
    nodes = raw.get("nodes")
    if raw.get("num_nodes") != 2 or not isinstance(nodes, list) or len(nodes) != 2:
        raise LiveBalancedStorageScenarioError(
            "baseline cluster config must describe exactly two nodes")

    capacities = []
    for node_index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise LiveBalancedStorageScenarioError(
                f"baseline node {node_index} must be an object")
        instances = node.get("instances")
        if (
            not isinstance(instances, list)
            or node.get("num_instances") != len(instances)
        ):
            raise LiveBalancedStorageScenarioError(
                f"baseline node {node_index} has invalid instances")
        decode_instances = [
            instance
            for instance in instances
            if isinstance(instance, dict)
            and instance.get("pd_type") == "decode"
        ]
        if len(decode_instances) != 1:
            raise LiveBalancedStorageScenarioError(
                f"baseline node {node_index} must have one decode instance")
        decode = decode_instances[0]
        expected_decode = {
            "model_name": _MODEL_NAME,
            "hardware": "H100",
            "num_npus": 4,
            "tp_size": 4,
            "pp_size": 1,
            "ep_size": 4,
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
        }
        mismatches = {
            key: (decode.get(key), expected)
            for key, expected in expected_decode.items()
            if decode.get(key) != expected
        }
        if mismatches:
            raise LiveBalancedStorageScenarioError(
                f"baseline node {node_index} decode geometry changed: "
                f"{mismatches}")

        cpu_mem = node.get("cpu_mem")
        npu_mem = decode.get("npu_mem")
        if not isinstance(cpu_mem, dict) or not isinstance(npu_mem, dict):
            raise LiveBalancedStorageScenarioError(
                f"baseline node {node_index} has invalid memory config")
        cpu_gib = cpu_mem.get("mem_size")
        npu_gib = npu_mem.get("mem_size")
        reserve_bytes = npu_mem.get("runtime_reserve_bytes")
        for name, value in (
            ("cpu_mem.mem_size", cpu_gib),
            ("npu_mem.mem_size", npu_gib),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise LiveBalancedStorageScenarioError(
                    f"baseline node {node_index} {name} must be positive "
                    "and finite")
        if (
            isinstance(reserve_bytes, bool)
            or not isinstance(reserve_bytes, int)
            or reserve_bytes < 0
        ):
            raise LiveBalancedStorageScenarioError(
                f"baseline node {node_index} runtime reserve is invalid")

        cpu_bytes = int(round(float(cpu_gib) * 1024 ** 3))
        npu_physical_bytes = int(round(float(npu_gib) * 1024 ** 3))
        weight_bytes_per_rank = qwen_model_weight_bytes_per_rank(
            int(decode["tp_size"]))
        usable_npu_bytes_per_rank = (
            npu_physical_bytes - reserve_bytes - weight_bytes_per_rank
        )
        if usable_npu_bytes_per_rank <= 0:
            raise LiveBalancedStorageScenarioError(
                f"baseline node {node_index} has no usable decode HBM")
        capacities.append(
            cpu_bytes
            + int(decode["num_npus"]) * usable_npu_bytes_per_rank
        )

    if len(set(capacities)) != 1:
        raise LiveBalancedStorageScenarioError(
            f"baseline node capacities differ: {capacities}")
    derived = capacities[0]
    if derived != BASELINE_USABLE_D_HBM_AND_CPU_BYTES_PER_NODE:
        raise LiveBalancedStorageScenarioError(
            "baseline usable D-HBM plus host-DRAM capacity changed: "
            f"derived={derived}, "
            f"expected={BASELINE_USABLE_D_HBM_AND_CPU_BYTES_PER_NODE}")
    return derived, hashlib.sha256(payload).hexdigest()


def _configured_hbf_logical_capacities() -> dict[str, int]:
    hardware, layouts = load_hbf_server_config(_HBF_CONFIG_PATH)
    required = ("tp4", "tp8", "tp8_context")
    if any(key not in layouts for key in required):
        raise LiveBalancedStorageScenarioError(
            "runtime HBF config omits a required layout")
    capacities = {}
    for key in required:
        layout = layouts[key]
        free_bytes_per_card = (
            hardware.hbf_capacity_bytes_per_card
            - qwen_model_weight_bytes_per_rank(layout.tp_size)
        )
        if free_bytes_per_card <= 0:
            raise LiveBalancedStorageScenarioError(
                f"modeled Qwen weights do not fit the {key} HBF layout")
        physical_free = (
            free_bytes_per_card
            * layout.tp_size
            * layout.replicas
        )
        capacities[key] = (
            physical_free
            // layout.physical_kv_replication_factor
        )
    return capacities


_HBF_LOGICAL_CAPACITIES = _configured_hbf_logical_capacities()
HBF_TP4_USABLE_LOGICAL_KV_BYTES = _HBF_LOGICAL_CAPACITIES["tp4"]
HBF_TP8_USABLE_LOGICAL_KV_BYTES = _HBF_LOGICAL_CAPACITIES["tp8"]
HBF_TP8_CONTEXT_USABLE_LOGICAL_KV_BYTES = (
    _HBF_LOGICAL_CAPACITIES["tp8_context"]
)


@dataclass(frozen=True)
class BalancedEpochMapping:
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
    resume_fresh_input_tokens_per_epoch: int
    positive_recorded_gaps_per_epoch: int
    recorded_gap_ns_per_epoch: int
    max_recorded_gap_ns: int
    complete_cohort_sha256: str
    source_identity_sha256: str
    recorded_gaps_sha256: str
    completeness_semantics: str
    transform_semantics: str


@dataclass(frozen=True)
class PrefillServiceBalanceAudit:
    first_prefill_service_ns_per_epoch: int
    resume_prefill_service_ns_per_epoch: int
    absolute_difference_ns: int
    resume_to_first_service_ratio: float
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
class FiniteNodePressureWitness:
    seed: int
    offered_session_rate_per_second: float
    per_node_peak_recorded_gap_logical_kv_bytes: tuple[int, int]
    aggregate_peak_recorded_gap_logical_kv_bytes: int
    measurement_resume_return_count: int
    measurement_node_pressure_return_count: int
    measurement_aggregate_pressure_return_count: int
    measurement_long_gap_return_count: int
    measurement_long_gap_node_pressure_return_count: int
    exceeds_baseline_capacity: bool
    fits_smallest_hbf_capacity: bool
    node_assignment_semantics: str
    semantics: str


@dataclass(frozen=True)
class FiniteSchedulePressureAudit:
    audited_seeds: tuple[int, ...]
    audited_max_rate_per_second: float
    minimum_warmup_arrival_span_ns: int
    minimum_guard_arrival_span_ns: int
    recorded_max_gap_ns: int
    warmup_span_covers_max_gap_for_all_seeds: bool
    guard_span_covers_max_gap_for_all_seeds: bool
    minimum_measurement_node_pressure_return_count: int
    minimum_measurement_long_gap_node_pressure_return_count: int
    minimum_aggregate_peak_bytes: int
    maximum_aggregate_peak_bytes: int
    witnesses: tuple[FiniteNodePressureWitness, ...]
    semantics: str


@dataclass(frozen=True)
class KVPressureAudit:
    logical_kv_bytes_per_token: int
    block_tokens: int
    recorded_gap_logical_kv_byte_ns_per_epoch: int
    terminal_logical_kv_bytes_per_epoch: int
    baseline_usable_bytes_per_node: int
    baseline_combined_usable_bytes: int
    baseline_cluster_config_path: str
    baseline_cluster_config_sha256: str
    hbf_tp4_usable_logical_kv_bytes: int
    hbf_tp8_usable_logical_kv_bytes: int
    hbf_tp8_context_usable_logical_kv_bytes: int
    hbf_config_path: str
    hbf_config_sha256: str
    analytical_storage_knee_sessions_per_second: float
    analytical_steady_state_estimates: tuple[SteadyStateKVEstimate, ...]
    finite_schedule: FiniteSchedulePressureAudit
    analytical_semantics: str
    capacity_semantics: str
    realized_runtime_semantics: str


@dataclass(frozen=True)
class LiveBalancedStorageManifest:
    schema_version: int
    scenario_id: str
    epoch_profile: str
    source_sha256: str
    selected_source_indices: tuple[int, ...]
    selected_source_session_ids: tuple[str, ...]
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
    kv_pressure: KVPressureAudit
    recommended_rates: tuple[float, ...]
    recommended_seeds: tuple[int, ...]
    maximum_audited_rate: float
    rate_semantics: str
    workload_semantics: str
    successor_release_semantics: str
    measurement_semantics: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RateBoundedOfferedPlan:
    """Forward an offered plan while enforcing the preregistered rate cap."""

    plan: object
    maximum_rate: float

    def __getattr__(self, name: str):
        return getattr(self.plan, name)

    def at_rate(self, sessions_per_second: float, **kwargs):
        rate = _validate_rate(sessions_per_second)
        if rate > self.maximum_rate:
            raise ValueError(
                f"sessions_per_second={rate} exceeds the audited maximum "
                f"{self.maximum_rate}")
        return self.plan.at_rate(rate, **kwargs)


@dataclass(frozen=True)
class LiveBalancedStorageScenario:
    manifest: LiveBalancedStorageManifest
    epoch_sessions: tuple[tuple[SessionSpec, ...], ...]

    def build_offered_plan(self, *, seed: int):
        return RateBoundedOfferedPlan(
            plan=_build_epoch_offered_plan(
                self.epoch_sessions, seed=seed),
            maximum_rate=self.manifest.maximum_audited_rate,
        )

    def audit_zero_service_pressure(
            self, *, seed: int,
            sessions_per_second: float) -> FiniteNodePressureWitness:
        """Audit finite dormant occupancy under sticky two-node RR."""

        rate = _validate_rate(sessions_per_second)
        if rate > self.manifest.maximum_audited_rate:
            raise ValueError(
                f"sessions_per_second={rate} exceeds the audited maximum "
                f"{self.manifest.maximum_audited_rate}")
        return _zero_service_pressure_witness(
            self.epoch_sessions,
            measurement_session_ids=set(
                self.manifest.measurement_session_ids),
            seed=seed,
            rate=rate,
        )


def _validate_rate(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError("sessions_per_second must be positive and finite")
    return float(value)


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


def _canonical_cohort_payload(
        sessions: Sequence[SessionSpec],
) -> list[dict[str, object]]:
    return [
        {
            "source_index": session.source_index,
            "session_id": session.session_id,
            "source_arrival_time_ns": session.source_arrival_time_ns,
            "source_session_identity_sha256": (
                session.source_session_identity_sha256),
            "calls": [asdict(call) for call in session.calls],
        }
        for session in sessions
    ]


def _validate_templates(
        sessions: Sequence[SessionSpec],
) -> tuple[tuple[SessionSpec, ...], CompleteCohortAudit]:
    templates = tuple(sessions)
    observed_indices = tuple(
        session.source_index for session in templates)
    if observed_indices != SELECTED_SOURCE_INDICES:
        raise LiveBalancedStorageScenarioError(
            "selected TraceLab source order changed: "
            f"observed={observed_indices}, "
            f"expected={SELECTED_SOURCE_INDICES}")
    observed_ids = tuple(session.session_id for session in templates)
    if observed_ids != EXPECTED_SOURCE_SESSION_IDS:
        raise LiveBalancedStorageScenarioError(
            "selected TraceLab source identities changed")

    complete_sha = stable_json_sha256(
        _canonical_cohort_payload(templates))
    if complete_sha != EXPECTED_COMPLETE_COHORT_SHA256:
        raise LiveBalancedStorageScenarioError(
            "pinned complete TraceLab cohort changed: "
            f"observed={complete_sha}, "
            f"expected={EXPECTED_COMPLETE_COHORT_SHA256}")
    identity_sha = stable_json_sha256([
        (
            session.source_index,
            session.session_id,
            session.source_session_identity_sha256,
        )
        for session in templates
    ])
    if identity_sha != EXPECTED_SOURCE_IDENTITY_SHA256:
        raise LiveBalancedStorageScenarioError(
            "pinned TraceLab source identity fingerprint changed")
    gaps_sha = stable_json_sha256([
        {
            "source_index": session.source_index,
            "call_index": call.call_index,
            "tool_duration_ns": call.tool_duration_ns,
            "inter_turn_gap_type": call.inter_turn_gap_type,
        }
        for session in templates
        for call in session.calls
    ])
    if gaps_sha != EXPECTED_RECORDED_GAPS_SHA256:
        raise LiveBalancedStorageScenarioError(
            "pinned TraceLab recorded-gap fingerprint changed")

    summary = summarize_sessions(templates)
    observed = (
        summary.session_count,
        summary.call_count,
        summary.first_turn_count,
        summary.resume_count,
        summary.total_output_tokens,
        summary.total_input_tokens,
        summary.total_cached_prefix_tokens,
        summary.total_fresh_input_tokens,
        summary.resume_fresh_input_tokens,
        sum(
            call.tool_duration_ns > 0
            for session in templates
            for call in session.calls
        ),
        sum(
            call.tool_duration_ns
            for session in templates
            for call in session.calls
        ),
        max(
            call.tool_duration_ns
            for session in templates
            for call in session.calls
        ),
        tuple(len(session.calls) for session in templates),
        tuple(session.output_tokens for session in templates),
    )
    expected = (
        EXPECTED_SESSIONS_PER_EPOCH,
        EXPECTED_CALLS_PER_EPOCH,
        EXPECTED_FIRST_CALLS_PER_EPOCH,
        EXPECTED_RESUME_CALLS_PER_EPOCH,
        EXPECTED_OUTPUT_TOKENS_PER_EPOCH,
        EXPECTED_INPUT_TOKENS_PER_EPOCH,
        EXPECTED_CACHED_PREFIX_TOKENS_PER_EPOCH,
        EXPECTED_FRESH_INPUT_TOKENS_PER_EPOCH,
        EXPECTED_RESUME_FRESH_INPUT_TOKENS_PER_EPOCH,
        EXPECTED_POSITIVE_RECORDED_GAPS_PER_EPOCH,
        EXPECTED_RECORDED_GAP_NS_PER_EPOCH,
        EXPECTED_MAX_RECORDED_GAP_NS,
        (8, 1, 1, 1, 1, 1, 1),
        (34, 1, 1, 1, 1, 1, 1),
    )
    if observed != expected:
        raise LiveBalancedStorageScenarioError(
            "pinned balanced epoch counts changed: "
            f"observed={observed}, expected={expected}")
    if any(
            call.output_tokens != 1
            for session in templates[1:]
            for call in session.calls
    ):
        raise LiveBalancedStorageScenarioError(
            "one-call balancing sessions no longer emit one token")

    audit = CompleteCohortAudit(
        sessions_per_epoch=summary.session_count,
        calls_per_epoch=summary.call_count,
        first_calls_per_epoch=summary.first_turn_count,
        resume_calls_per_epoch=summary.resume_count,
        output_tokens_per_epoch=summary.total_output_tokens,
        input_tokens_per_epoch=summary.total_input_tokens,
        cached_prefix_tokens_per_epoch=(
            summary.total_cached_prefix_tokens),
        fresh_input_tokens_per_epoch=summary.total_fresh_input_tokens,
        resume_fresh_input_tokens_per_epoch=(
            summary.resume_fresh_input_tokens),
        positive_recorded_gaps_per_epoch=(
            EXPECTED_POSITIVE_RECORDED_GAPS_PER_EPOCH),
        recorded_gap_ns_per_epoch=EXPECTED_RECORDED_GAP_NS_PER_EPOCH,
        max_recorded_gap_ns=EXPECTED_MAX_RECORDED_GAP_NS,
        complete_cohort_sha256=complete_sha,
        source_identity_sha256=identity_sha,
        recorded_gaps_sha256=gaps_sha,
        completeness_semantics=(
            "seven original complete TraceLab sessions are retained; source "
            "1741 keeps all eight calls and every one-call source remains a "
            "natural complete one-call session"),
        transform_semantics=(
            "no context scaling, call truncation, call replication inside "
            "an epoch, or token/gap rewrite is applied; only complete-epoch "
            "repetition, deterministic within-epoch interleaving, and "
            "synthetic lineage renaming are used"),
    )
    return templates, audit


def _load_templates(
        trace_path: Path,
) -> tuple[tuple[SessionSpec, ...], CompleteCohortAudit]:
    workload = load_comparison_workload(
        trace_path,
        source_indices=tuple(sorted(SELECTED_SOURCE_INDICES)),
        expected_source_sha256=TRACELAB_SCHEMA3_SHA256,
        expected_source_session_count=4281,
    )
    by_index = {
        session.source_index: session
        for session in workload.sessions
    }
    templates = tuple(
        by_index[index] for index in SELECTED_SOURCE_INDICES)
    return _validate_templates(templates)


def _latency_provider():
    return resolve_online_latency_model(
        name=H100_QWEN3_TP4_KERNEL_CALIBRATED,
        repo_root=_REPO_ROOT,
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
    observed = (
        first_ns,
        resume_ns,
        resume_ns - first_ns,
    )
    expected = (
        EXPECTED_FIRST_PREFILL_SERVICE_NS,
        EXPECTED_RESUME_PREFILL_SERVICE_NS,
        EXPECTED_PREFILL_SERVICE_DIFFERENCE_NS,
    )
    if observed != expected:
        raise LiveBalancedStorageScenarioError(
            "pinned first/resume prefill service audit changed: "
            f"observed={observed}, expected={expected}")
    return PrefillServiceBalanceAudit(
        first_prefill_service_ns_per_epoch=first_ns,
        resume_prefill_service_ns_per_epoch=resume_ns,
        absolute_difference_ns=resume_ns - first_ns,
        resume_to_first_service_ratio=resume_ns / first_ns,
        model=_MODEL_NAME,
        topology="one_tp4_h100_model_partition",
        latency_band="central",
        method=(
            "sum of singleton prefill COMP critical paths with 131072-token "
            "chunks, no collectives, and no queueing; resume hits use the "
            "trace-declared prefix capped to input_tokens-1"),
    )


def _block_rounded_kv_bytes(token_count: int) -> int:
    blocks = (
        token_count + KV_BLOCK_TOKENS - 1
    ) // KV_BLOCK_TOKENS
    return blocks * KV_BLOCK_TOKENS * LOGICAL_KV_BYTES_PER_TOKEN


def _kv_byte_gap_audit(
        templates: Sequence[SessionSpec],
) -> tuple[int, int]:
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
    observed = (byte_ns, terminal_bytes)
    expected = (
        EXPECTED_RECORDED_GAP_LOGICAL_KV_BYTE_NS_PER_EPOCH,
        EXPECTED_TERMINAL_LOGICAL_KV_BYTES_PER_EPOCH,
    )
    if observed != expected:
        raise LiveBalancedStorageScenarioError(
            "pinned dormant logical-KV byte-gap audit changed: "
            f"observed={observed}, expected={expected}")
    return observed


def _steady_state_estimates(
        byte_ns_per_epoch: int,
) -> tuple[SteadyStateKVEstimate, ...]:
    estimates = []
    for rate in RECOMMENDED_RATES:
        rate_fraction = Fraction(str(rate))
        live_bytes = (
            byte_ns_per_epoch
            * rate_fraction.numerator
            // (
                EXPECTED_SESSIONS_PER_EPOCH
                * 1_000_000_000
                * rate_fraction.denominator
            )
        )
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


def _clone_session(
        source: SessionSpec,
        *,
        epoch_index: int,
        role: str,
        role_epoch_index: int,
        synthetic_source_index: int,
) -> tuple[SessionSpec, BalancedEpochMapping]:
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
        "unmodified_call_count": len(calls),
    })
    session = SessionSpec(
        source_index=synthetic_source_index,
        session_id=session_id,
        source_arrival_time_ns=source.source_arrival_time_ns,
        source_session_identity_sha256=identity,
        calls=calls,
    )
    mapping = BalancedEpochMapping(
        epoch_index=epoch_index,
        role=role,
        role_epoch_index=role_epoch_index,
        session_id=session_id,
        synthetic_source_index=synthetic_source_index,
        source_index=source.source_index,
        source_session_id=source.session_id,
    )
    return session, mapping


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
        PUBLICATION_WARMUP_EPOCH_COUNT,
        PUBLICATION_MEASUREMENT_EPOCH_COUNT,
        PUBLICATION_GUARD_EPOCH_COUNT,
    ):
        return "publication"
    if counts == (
        PROTOCOL_SMOKE_WARMUP_EPOCH_COUNT,
        PROTOCOL_SMOKE_MEASUREMENT_EPOCH_COUNT,
        PROTOCOL_SMOKE_GUARD_EPOCH_COUNT,
    ):
        return "protocol_smoke"
    return "custom"


def _zero_service_pressure_witness(
        epoch_sessions: Sequence[Sequence[SessionSpec]],
        *,
        measurement_session_ids: set[str],
        seed: int,
        rate: float,
) -> FiniteNodePressureWitness:
    scheduled = _build_epoch_offered_plan(
        epoch_sessions, seed=seed).at_rate(rate)
    events = []
    for item in scheduled:
        node_id = item.offer_index % 2
        boundary_ns = item.arrival_time_ns
        is_measurement = item.session.session_id in measurement_session_ids
        for call in item.session.calls[:-1]:
            next_boundary_ns = boundary_ns + call.tool_duration_ns
            if next_boundary_ns > boundary_ns:
                logical_bytes = _block_rounded_kv_bytes(
                    call.input_tokens + call.output_tokens)
                events.append((
                    boundary_ns,
                    1,
                    logical_bytes,
                    node_id,
                    False,
                    call.call_index,
                    call.tool_duration_ns,
                ))
                events.append((
                    next_boundary_ns,
                    0,
                    -logical_bytes,
                    node_id,
                    is_measurement,
                    call.call_index,
                    call.tool_duration_ns,
                ))
            boundary_ns = next_boundary_ns

    events.sort(key=lambda event: (event[0], event[1]))
    live = [0, 0]
    node_peak = [0, 0]
    aggregate_peak = 0
    measurement_returns = 0
    measurement_node_pressure = 0
    measurement_aggregate_pressure = 0
    measurement_long_returns = 0
    measurement_long_node_pressure = 0
    event_index = 0
    while event_index < len(events):
        timestamp_ns = events[event_index][0]
        next_index = event_index + 1
        while (
            next_index < len(events)
            and events[next_index][0] == timestamp_ns
        ):
            next_index += 1
        same_time = events[event_index:next_index]

        # Every return at the same timestamp observes the same pre-boundary
        # occupancy.  End events are then applied before new starts.
        for (
            _,
            kind,
            _,
            node_id,
            is_measurement,
            _,
            gap_ns,
        ) in same_time:
            if kind != 0 or not is_measurement:
                continue
            node_pressured = (
                live[node_id]
                > BASELINE_USABLE_D_HBM_AND_CPU_BYTES_PER_NODE
            )
            aggregate_pressured = (
                sum(live)
                > BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES
            )
            measurement_returns += 1
            measurement_node_pressure += int(node_pressured)
            measurement_aggregate_pressure += int(aggregate_pressured)
            if gap_ns == EXPECTED_MAX_RECORDED_GAP_NS:
                measurement_long_returns += 1
                measurement_long_node_pressure += int(node_pressured)

        for _, _, delta, node_id, _, _, _ in same_time:
            live[node_id] += delta
            if live[node_id] < 0:
                raise LiveBalancedStorageScenarioError(
                    "finite per-node dormant-KV accounting became negative")
            node_peak[node_id] = max(
                node_peak[node_id], live[node_id])
            aggregate_peak = max(aggregate_peak, sum(live))
        event_index = next_index

    if live != [0, 0]:
        raise LiveBalancedStorageScenarioError(
            "finite per-node dormant-KV accounting did not drain")
    return FiniteNodePressureWitness(
        seed=seed,
        offered_session_rate_per_second=rate,
        per_node_peak_recorded_gap_logical_kv_bytes=(
            node_peak[0], node_peak[1]),
        aggregate_peak_recorded_gap_logical_kv_bytes=aggregate_peak,
        measurement_resume_return_count=measurement_returns,
        measurement_node_pressure_return_count=(
            measurement_node_pressure),
        measurement_aggregate_pressure_return_count=(
            measurement_aggregate_pressure),
        measurement_long_gap_return_count=measurement_long_returns,
        measurement_long_gap_node_pressure_return_count=(
            measurement_long_node_pressure),
        exceeds_baseline_capacity=(
            aggregate_peak
            > BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES),
        fits_smallest_hbf_capacity=(
            aggregate_peak < HBF_TP8_USABLE_LOGICAL_KV_BYTES),
        node_assignment_semantics=(
            "two-node sticky round-robin by offered-session index; all calls "
            "of a complete session remain on that session's assigned node"),
        semantics=(
            "exact block-rounded recorded-gap interval sweep with LLM "
            "service, queueing, migration, and transfer time set to zero; "
            "a pressured return is evaluated before its own dormant interval "
            "ends, and runtime reports remain authoritative for SSD hits"),
    )


def _finite_schedule_pressure_audit(
        epoch_sessions: Sequence[Sequence[SessionSpec]],
        *,
        measurement_session_ids: set[str],
        warmup_epoch_count: int,
        measurement_epoch_count: int,
        guard_epoch_count: int,
        profile: str,
) -> FiniteSchedulePressureAudit:
    witnesses = tuple(
        _zero_service_pressure_witness(
            epoch_sessions,
            measurement_session_ids=measurement_session_ids,
            seed=seed,
            rate=MAXIMUM_AUDITED_RATE,
        )
        for seed in PRESSURE_AUDIT_SEEDS
    )
    warmup_session_count = (
        warmup_epoch_count * EXPECTED_SESSIONS_PER_EPOCH)
    measurement_session_count = (
        measurement_epoch_count * EXPECTED_SESSIONS_PER_EPOCH)
    warmup_spans = []
    guard_spans = []
    for seed in PRESSURE_AUDIT_SEEDS:
        scheduled = _build_epoch_offered_plan(
            epoch_sessions, seed=seed).at_rate(MAXIMUM_AUDITED_RATE)
        if warmup_session_count:
            warmup_spans.append(
                scheduled[warmup_session_count].arrival_time_ns
                - scheduled[0].arrival_time_ns
            )
        else:
            warmup_spans.append(0)
        if guard_epoch_count:
            last_measurement_index = (
                warmup_session_count + measurement_session_count - 1)
            guard_spans.append(
                scheduled[-1].arrival_time_ns
                - scheduled[last_measurement_index].arrival_time_ns
            )
        else:
            guard_spans.append(0)
    minimum_warmup_span = min(warmup_spans)
    minimum_guard_span = min(guard_spans)
    audit = FiniteSchedulePressureAudit(
        audited_seeds=PRESSURE_AUDIT_SEEDS,
        audited_max_rate_per_second=MAXIMUM_AUDITED_RATE,
        minimum_warmup_arrival_span_ns=minimum_warmup_span,
        minimum_guard_arrival_span_ns=minimum_guard_span,
        recorded_max_gap_ns=EXPECTED_MAX_RECORDED_GAP_NS,
        warmup_span_covers_max_gap_for_all_seeds=(
            minimum_warmup_span >= EXPECTED_MAX_RECORDED_GAP_NS),
        guard_span_covers_max_gap_for_all_seeds=(
            minimum_guard_span >= EXPECTED_MAX_RECORDED_GAP_NS),
        minimum_measurement_node_pressure_return_count=min(
            witness.measurement_node_pressure_return_count
            for witness in witnesses
        ),
        minimum_measurement_long_gap_node_pressure_return_count=min(
            witness.measurement_long_gap_node_pressure_return_count
            for witness in witnesses
        ),
        minimum_aggregate_peak_bytes=min(
            witness.aggregate_peak_recorded_gap_logical_kv_bytes
            for witness in witnesses
        ),
        maximum_aggregate_peak_bytes=max(
            witness.aggregate_peak_recorded_gap_logical_kv_bytes
            for witness in witnesses
        ),
        witnesses=witnesses,
        semantics=(
            "arrival spans and zero-service occupancy are evaluated for "
            "seeds 101-112 at 1.2 external sessions/s. Full max-gap span "
            "coverage is reported separately from the stronger observed "
            "measurement-return pressure invariant; neither substitutes "
            "for runtime SSD traffic and queue counters"),
    )
    if profile == "publication":
        seed101 = next(
            witness for witness in witnesses if witness.seed == 101)
        observed_seed101 = (
            seed101.per_node_peak_recorded_gap_logical_kv_bytes,
            seed101.aggregate_peak_recorded_gap_logical_kv_bytes,
        )
        expected_seed101 = (
            EXPECTED_PUBLICATION_SEED101_NODE_PEAK_BYTES,
            EXPECTED_PUBLICATION_SEED101_AGGREGATE_PEAK_BYTES,
        )
        if observed_seed101 != expected_seed101:
            raise LiveBalancedStorageScenarioError(
                "publication seed-101 finite occupancy changed: "
                f"observed={observed_seed101}, "
                f"expected={expected_seed101}")
        if (
            audit.minimum_measurement_node_pressure_return_count
            != EXPECTED_PUBLICATION_MEASUREMENT_RESUME_RETURNS
            or audit.minimum_measurement_long_gap_node_pressure_return_count
            != EXPECTED_PUBLICATION_MEASUREMENT_LONG_GAP_RETURNS
            or not all(
                witness.exceeds_baseline_capacity
                and witness.fits_smallest_hbf_capacity
                for witness in witnesses
            )
        ):
            raise LiveBalancedStorageScenarioError(
                "publication finite pressure no longer separates baseline "
                "and HBF storage for every audited seed")
    return audit


def build_live_balanced_storage_scenario(
        trace_path: str | Path,
        *,
        warmup_epoch_count: int = PUBLICATION_WARMUP_EPOCH_COUNT,
        measurement_epoch_count: int = PUBLICATION_MEASUREMENT_EPOCH_COUNT,
        guard_epoch_count: int = PUBLICATION_GUARD_EPOCH_COUNT,
) -> LiveBalancedStorageScenario:
    """Build and fail-closed validate the complete balanced headline cohort."""

    warmup_epoch_count = _validate_epoch_count(
        warmup_epoch_count, "warmup_epoch_count", positive=False)
    measurement_epoch_count = _validate_epoch_count(
        measurement_epoch_count, "measurement_epoch_count", positive=True)
    guard_epoch_count = _validate_epoch_count(
        guard_epoch_count, "guard_epoch_count", positive=False)
    path = Path(trace_path).expanduser().resolve()
    if not path.is_file():
        raise LiveBalancedStorageScenarioError(
            f"TraceLab source does not exist: {path}")

    templates, cohort_audit = _load_templates(path)
    prefill_service = _prefill_service_audit(templates)
    byte_ns_per_epoch, terminal_bytes = _kv_byte_gap_audit(templates)
    baseline_usable_bytes, baseline_config_sha256 = (
        _validated_baseline_usable_bytes_per_node())

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
            for template in templates:
                session, mapping = _clone_session(
                    template,
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

    measurement_epoch_set = set(role_epoch_indices["measurement"])
    measurement_ids = tuple(
        mapping.session_id
        for mapping in mappings
        if mapping.epoch_index in measurement_epoch_set
    )
    measurement_request_count = (
        EXPECTED_CALLS_PER_EPOCH * measurement_epoch_count)
    measurement_first_count = (
        EXPECTED_FIRST_CALLS_PER_EPOCH * measurement_epoch_count)
    measurement_resume_count = (
        EXPECTED_RESUME_CALLS_PER_EPOCH * measurement_epoch_count)
    profile = _epoch_profile(
        warmup_epoch_count,
        measurement_epoch_count,
        guard_epoch_count,
    )
    finite_schedule = _finite_schedule_pressure_audit(
        epoch_groups,
        measurement_session_ids=set(measurement_ids),
        warmup_epoch_count=warmup_epoch_count,
        measurement_epoch_count=measurement_epoch_count,
        guard_epoch_count=guard_epoch_count,
        profile=profile,
    )
    storage_knee = (
        EXPECTED_SESSIONS_PER_EPOCH
        * BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES
        * 1_000_000_000
        / byte_ns_per_epoch
    )
    mapping_sha = stable_json_sha256([
        asdict(mapping) for mapping in mappings
    ])
    manifest = LiveBalancedStorageManifest(
        schema_version=SCENARIO_SCHEMA_VERSION,
        scenario_id=SCENARIO_ID,
        epoch_profile=profile,
        source_sha256=TRACELAB_SCHEMA3_SHA256,
        selected_source_indices=SELECTED_SOURCE_INDICES,
        selected_source_session_ids=EXPECTED_SOURCE_SESSION_IDS,
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
        epoch_mapping_sha256=mapping_sha,
        cohort_audit=cohort_audit,
        prefill_service=prefill_service,
        kv_pressure=KVPressureAudit(
            logical_kv_bytes_per_token=LOGICAL_KV_BYTES_PER_TOKEN,
            block_tokens=KV_BLOCK_TOKENS,
            recorded_gap_logical_kv_byte_ns_per_epoch=byte_ns_per_epoch,
            terminal_logical_kv_bytes_per_epoch=terminal_bytes,
            baseline_usable_bytes_per_node=(
                baseline_usable_bytes),
            baseline_combined_usable_bytes=(
                2 * baseline_usable_bytes),
            baseline_cluster_config_path=(
                _BASELINE_CLUSTER_CONFIG_RELATIVE.as_posix()),
            baseline_cluster_config_sha256=baseline_config_sha256,
            hbf_tp4_usable_logical_kv_bytes=(
                HBF_TP4_USABLE_LOGICAL_KV_BYTES),
            hbf_tp8_usable_logical_kv_bytes=(
                HBF_TP8_USABLE_LOGICAL_KV_BYTES),
            hbf_tp8_context_usable_logical_kv_bytes=(
                HBF_TP8_CONTEXT_USABLE_LOGICAL_KV_BYTES),
            hbf_config_path=str(
                _HBF_CONFIG_PATH.relative_to(_REPO_ROOT)),
            hbf_config_sha256=_sha256_file(_HBF_CONFIG_PATH),
            analytical_storage_knee_sessions_per_second=storage_knee,
            analytical_steady_state_estimates=(
                _steady_state_estimates(byte_ns_per_epoch)),
            finite_schedule=finite_schedule,
            analytical_semantics=(
                "Little's law for the uniform seven-session epoch mix: "
                "offered external session rate / 7 multiplied by source "
                "1741's block-rounded logical-KV byte-nanoseconds over every "
                "unchanged recorded external gap; active service, queues, "
                "migrations, and terminal KV are excluded"),
            capacity_semantics=(
                "baseline pressure is per sticky node against usable D-HBM "
                "plus host DRAM, with the aggregate shown separately. HBF "
                "capacities reserve exact modeled Qwen weights; TP4 has two "
                "weight replicas but one session-KV copy, conventional TP8 "
                "has one weight replica and two physical GQA KV copies, and "
                "TP8-context stores one context-striped KV copy"),
            realized_runtime_semantics=(
                "finite and stationary occupancy are preregistered pressure "
                "witnesses only. Headline SSD reads, writes, PCIe queueing, "
                "TTFT, TPOT, and goodput must come from each live ASTRA cell "
                "and its runtime report"),
        ),
        recommended_rates=RECOMMENDED_RATES,
        recommended_seeds=RECOMMENDED_SEEDS,
        maximum_audited_rate=MAXIMUM_AUDITED_RATE,
        rate_semantics=(
            "system-wide external complete-session starts per second; each "
            "seven-session epoch contains exactly one eight-call source and "
            "six complete one-call sources. Rates above 1.2 are rejected"),
        workload_semantics=(
            "headline storage-before-compute cohort built only from seven "
            "original complete TraceLab sessions. Internal calls, inputs, "
            "outputs, cached-prefix coordinates, and tool gaps are unchanged; "
            "only complete-epoch repetition, deterministic interleaving, and "
            "lineage renaming are applied. First/resume request counts are "
            "7:7 and singleton TP4 prefill service differs by seven ns"),
        successor_release_semantics=(
            "each successor call is released only after its predecessor "
            "completes plus the unchanged recorded TraceLab tool duration; "
            "complete one-call sessions release no successor"),
        measurement_semantics=(
            "only the exact measurement-session roster contributes headline "
            "metrics. Warmup and guard epochs create the finite arrival "
            "history, all systems receive the identical seed/rate-scaled "
            "offered plan, and every system fully drains"),
    )
    return LiveBalancedStorageScenario(
        manifest=manifest,
        epoch_sessions=tuple(epoch_groups),
    )


def build_publication(
        trace_path: str | Path,
) -> LiveBalancedStorageScenario:
    """Build the 380/16/380 headline publication profile."""

    return build_live_balanced_storage_scenario(
        trace_path,
        warmup_epoch_count=PUBLICATION_WARMUP_EPOCH_COUNT,
        measurement_epoch_count=PUBLICATION_MEASUREMENT_EPOCH_COUNT,
        guard_epoch_count=PUBLICATION_GUARD_EPOCH_COUNT,
    )


def build_protocol_smoke(
        trace_path: str | Path,
) -> LiveBalancedStorageScenario:
    """Build the 0/1/0 plumbing smoke; it makes no pressure claim."""

    return build_live_balanced_storage_scenario(
        trace_path,
        warmup_epoch_count=PROTOCOL_SMOKE_WARMUP_EPOCH_COUNT,
        measurement_epoch_count=PROTOCOL_SMOKE_MEASUREMENT_EPOCH_COUNT,
        guard_epoch_count=PROTOCOL_SMOKE_GUARD_EPOCH_COUNT,
    )


def build(trace_path: str | Path) -> LiveBalancedStorageScenario:
    """Default scenario-factory entry point (the publication profile)."""

    return build_publication(trace_path)


__all__ = [
    "LiveBalancedStorageManifest",
    "LiveBalancedStorageScenario",
    "LiveBalancedStorageScenarioError",
    "MAXIMUM_AUDITED_RATE",
    "PRESSURE_AUDIT_SEEDS",
    "RECOMMENDED_RATES",
    "RECOMMENDED_SEEDS",
    "SELECTED_SOURCE_INDICES",
    "build",
    "build_live_balanced_storage_scenario",
    "build_protocol_smoke",
    "build_publication",
]

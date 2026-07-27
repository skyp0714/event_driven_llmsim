"""TraceLab-derived, service-balanced cold-KV pressure scenario.

The ordinary three-call balanced cohort equalizes first- and resume-prefill
service, but its working set fits in host DRAM and therefore cannot evaluate
SSD tiering.  The existing long-context sensitivity creates SSD pressure, but
contains roughly 22 times more resume-prefill service than first-prefill
service.  This scenario combines two policy-independent TraceLab populations:

* eight complete, low-output sessions whose prompt/prefix coordinates are
  globally scaled to a 250k-token maximum; and
* twenty additional first-turn-only copies of each selected session.

Including the first turn of each complete session gives 21 first-turn sets.
The H100/Qwen3 central analytical model estimates 274.4 seconds of aggregate
first-prefill service and 276.6 seconds of aggregate resume-prefill service per
epoch.  Twelve epochs create a 1.878 TB terminal logical-KV upper bound, above
the two-server baseline's combined usable decode HBM plus host DRAM, while
remaining below every proposed HBF layout's logical capacity.

This is an explicitly scaled pressure sensitivity, not an empirical
context-length distribution or an equilibrium workload.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import tempfile
from typing import Sequence

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
SCENARIO_ID = "tracelab-250k-service-balanced-kv-pressure-v1"
TARGET_MAX_SEQUENCE_TOKENS = 250_000
SELECTED_SOURCE_INDICES = (
    1858, 1879, 1904, 1905, 1911, 1933, 3780, 3782,
)
EXPECTED_TRANSFORMED_COHORT_SHA256 = (
    "86bc409d81a331ac0419c7c75e07b0136fa78903ba7706fad44f11b723e04b1b"
)
EXPECTED_CONTEXT_FACTOR_NUMERATOR = 249_999
EXPECTED_CONTEXT_FACTOR_DENOMINATOR = 80_752

EPOCH_COUNT = 12
WARMUP_EPOCHS = (0, 1, 2)
MEASUREMENT_EPOCHS = (3, 4, 5, 6, 7, 8)
GUARD_EPOCHS = (9, 10, 11)
FIRST_ONLY_REPLICA_SETS = 20

EXPECTED_TEMPLATE_FIRST_SERVICE_NS = 11_345_501_006
EXPECTED_TEMPLATE_RESUME_SERVICE_NS = 269_941_662_889
EXPECTED_BALANCED_FIRST_SERVICE_NS = 238_255_521_126
EXPECTED_RESUME_TO_FIRST_SERVICE_RATIO = (
    EXPECTED_TEMPLATE_RESUME_SERVICE_NS
    / EXPECTED_BALANCED_FIRST_SERVICE_NS
)

LOGICAL_KV_BYTES_PER_TOKEN = 98_304
KV_BLOCK_TOKENS = 16
EXPECTED_TERMINAL_LOGICAL_KV_BYTES_PER_EPOCH = 156_517_269_504
EXPECTED_TERMINAL_LOGICAL_KV_BYTES_ALL_EPOCHS = 1_878_207_234_048
BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES = 1_382_573_883_392

EXPECTED_SESSIONS_PER_EPOCH = 168
EXPECTED_CALLS_PER_EPOCH = 322
EXPECTED_FIRST_CALLS_PER_EPOCH = 168
EXPECTED_RESUME_CALLS_PER_EPOCH = 154
EXPECTED_OUTPUT_TOKENS_PER_EPOCH = 923

RECOMMENDED_RATES = (0.1, 0.25, 0.4, 0.55, 0.7)
RECOMMENDED_PILOT_RATES = (0.1, 0.4, 0.7)

_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class LivePressureScenarioError(ValueError):
    """Raised when the pinned pressure workload drifts."""


@dataclass(frozen=True)
class PressureEpochMapping:
    epoch_index: int
    role: str
    session_id: str
    synthetic_source_index: int
    source_index: int
    source_session_id: str
    session_kind: str
    first_only_replica_index: int | None


@dataclass(frozen=True)
class PrefillServiceAudit:
    template_first_service_ns: int
    template_resume_service_ns: int
    first_only_replica_sets: int
    balanced_first_service_ns: int
    resume_to_first_service_ratio: float
    model: str
    topology: str
    latency_band: str
    method: str


@dataclass(frozen=True)
class KVPressureAudit:
    logical_kv_bytes_per_token: int
    block_tokens: int
    terminal_logical_kv_bytes_per_epoch: int
    terminal_logical_kv_bytes_all_epochs: int
    baseline_combined_usable_d_hbm_and_cpu_bytes: int
    terminal_excess_over_baseline_bytes: int
    semantics: str


@dataclass(frozen=True)
class LivePressureManifest:
    schema_version: int
    scenario_id: str
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
    first_only_replica_sets: int
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
    prefill_service: PrefillServiceAudit
    kv_pressure: KVPressureAudit
    recommended_rates: tuple[float, ...]
    rate_semantics: str
    workload_semantics: str
    successor_release_semantics: str
    measurement_semantics: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LivePressureScenario:
    manifest: LivePressureManifest
    epoch_sessions: tuple[tuple[SessionSpec, ...], ...]

    def build_offered_plan(self, *, seed: int):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        ordered = []
        for epoch_index, group in enumerate(self.epoch_sessions):
            shuffled = list(group)
            epoch_seed = int.from_bytes(
                hashlib.sha256(
                    f"{SCENARIO_ID}:{seed}:{epoch_index}".encode("utf-8")
                ).digest()[:8],
                byteorder="big",
            )
            random.Random(epoch_seed).shuffle(shuffled)
            ordered.extend(shuffled)
        return build_offered_plan(
            tuple(ordered),
            seed=seed,
            shuffle=False,
        )


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
        "min_context_tokens": 25_000,
        "max_context_tokens": 100_000,
        "max_output_tokens_per_session": 93,
        "min_requests_per_session": 16,
        "max_requests_per_session": 24,
        "max_total_gap_ns": 5_000_000_000,
        "min_reuse_eligible_transitions": 15,
        "allowed_gap_types": ["tool"],
        "max_sessions": len(SELECTED_SOURCE_INDICES),
        "target_max_sequence_tokens": TARGET_MAX_SEQUENCE_TOKENS,
    }


def _materialized_templates(trace_path: Path) -> tuple[
        tuple[SessionSpec, ...], str, dict[str, object]]:
    if _sha256_file(trace_path) != TRACELAB_SCHEMA3_SHA256:
        raise LivePressureScenarioError(
            "TraceLab source SHA-256 does not match the pinned schema-3 trace")

    # Reuse the repository's audited lineage-preserving global context
    # transform. Import lazily so users of unrelated simulator modes do not
    # pay the online-experiment module import cost.
    from serving.online_experiments import materialize_session_cohort

    with tempfile.TemporaryDirectory(
            prefix="llmsim-live-pressure-cohort-") as temporary:
        descriptor = materialize_session_cohort(
            trace_path,
            Path(temporary),
            _selection(),
        )
        transformed_path = Path(descriptor["materialized_path"])
        transformed_sha = _sha256_file(transformed_path)
        if transformed_sha != EXPECTED_TRANSFORMED_COHORT_SHA256:
            raise LivePressureScenarioError(
                "250k transformed TraceLab cohort changed: "
                f"observed={transformed_sha}, "
                f"expected={EXPECTED_TRANSFORMED_COHORT_SHA256}")
        rows = [
            json.loads(line)
            for line in transformed_path.read_text(
                encoding="utf-8").splitlines()
            if line
        ]

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
        templates.append(SessionSpec(
            source_index=source_index,
            session_id=session_id,
            source_arrival_time_ns=int(row.get("arrival_time_ns", 0)),
            source_session_identity_sha256=(
                trace_metadata.get("source_session_identity_sha256")),
            calls=tuple(calls),
        ))

    observed_indices = tuple(
        template.source_index for template in templates)
    if observed_indices != SELECTED_SOURCE_INDICES:
        raise LivePressureScenarioError(
            "selected TraceLab source order changed: "
            f"observed={observed_indices}, "
            f"expected={SELECTED_SOURCE_INDICES}")
    transform = dict(descriptor["context_length_transform"])
    if (
        transform.get("global_factor_numerator")
        != EXPECTED_CONTEXT_FACTOR_NUMERATOR
        or transform.get("global_factor_denominator")
        != EXPECTED_CONTEXT_FACTOR_DENOMINATOR
        or transform.get("realized_max_sequence_tokens")
        != TARGET_MAX_SEQUENCE_TOKENS
    ):
        raise LivePressureScenarioError(
            "pinned 250k context transform changed")
    return tuple(templates), transformed_sha, transform


def _clone_session(
        source: SessionSpec,
        *,
        epoch_index: int,
        synthetic_source_index: int,
        first_only_replica_index: int | None,
) -> tuple[SessionSpec, PressureEpochMapping]:
    if first_only_replica_index is None:
        kind = "complete_long_session"
        suffix = "long"
        source_calls: Sequence[CallSpec] = source.calls
    else:
        kind = "first_turn_only"
        suffix = f"first-only-{first_only_replica_index:02d}"
        source_calls = source.calls[:1]
    session_id = (
        f"{SCENARIO_ID}::epoch-{epoch_index:02d}::{suffix}"
        f"::source-{source.source_index:04d}::{source.session_id}"
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
        "source_index": source.source_index,
        "source_session_id": source.session_id,
        "source_session_identity_sha256": (
            source.source_session_identity_sha256),
        "session_kind": kind,
        "first_only_replica_index": first_only_replica_index,
    })
    session = SessionSpec(
        source_index=synthetic_source_index,
        session_id=session_id,
        source_arrival_time_ns=source.source_arrival_time_ns,
        source_session_identity_sha256=identity,
        calls=calls,
    )
    role = (
        "warmup" if epoch_index in WARMUP_EPOCHS
        else "measurement" if epoch_index in MEASUREMENT_EPOCHS
        else "guard"
    )
    return session, PressureEpochMapping(
        epoch_index=epoch_index,
        role=role,
        session_id=session_id,
        synthetic_source_index=synthetic_source_index,
        source_index=source.source_index,
        source_session_id=source.session_id,
        session_kind=kind,
        first_only_replica_index=first_only_replica_index,
    )


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
) -> PrefillServiceAudit:
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
    balanced_first_ns = first_ns * (FIRST_ONLY_REPLICA_SETS + 1)
    observed = (first_ns, resume_ns, balanced_first_ns)
    expected = (
        EXPECTED_TEMPLATE_FIRST_SERVICE_NS,
        EXPECTED_TEMPLATE_RESUME_SERVICE_NS,
        EXPECTED_BALANCED_FIRST_SERVICE_NS,
    )
    if observed != expected:
        raise LivePressureScenarioError(
            "pinned first/resume prefill service audit changed: "
            f"observed={observed}, expected={expected}")
    return PrefillServiceAudit(
        template_first_service_ns=first_ns,
        template_resume_service_ns=resume_ns,
        first_only_replica_sets=FIRST_ONLY_REPLICA_SETS,
        balanced_first_service_ns=balanced_first_ns,
        resume_to_first_service_ratio=resume_ns / balanced_first_ns,
        model=_MODEL_NAME,
        topology="one_tp4_h100_prefill_partition",
        latency_band="central",
        method=(
            "sum_of_singleton_prefill_COMP_critical_paths_with_131072_"
            "token_chunks_no_collectives_no_queueing"),
    )


def _terminal_kv_bytes(templates: Sequence[SessionSpec]) -> int:
    total = 0
    for template in templates:
        final = template.calls[-1]
        tokens = final.input_tokens + final.output_tokens
        block_tokens = (
            (tokens + KV_BLOCK_TOKENS - 1)
            // KV_BLOCK_TOKENS
            * KV_BLOCK_TOKENS
        )
        total += block_tokens * LOGICAL_KV_BYTES_PER_TOKEN
    return total


def build_live_pressure_scenario(
        trace_path: str | Path,
) -> LivePressureScenario:
    """Build and fail-closed validate the pressure-balanced scenario."""

    path = Path(trace_path).resolve()
    if not path.is_file():
        raise LivePressureScenarioError(
            f"TraceLab source does not exist: {path}")
    templates, transformed_sha, transform = _materialized_templates(path)
    service_audit = _prefill_service_audit(templates)
    terminal_per_epoch = _terminal_kv_bytes(templates)
    if terminal_per_epoch != EXPECTED_TERMINAL_LOGICAL_KV_BYTES_PER_EPOCH:
        raise LivePressureScenarioError(
            "pinned terminal logical-KV working set changed")

    epoch_groups = []
    mappings = []
    synthetic_source_index = 0
    for epoch_index in range(EPOCH_COUNT):
        group = []
        for source in templates:
            session, mapping = _clone_session(
                source,
                epoch_index=epoch_index,
                synthetic_source_index=synthetic_source_index,
                first_only_replica_index=None,
            )
            synthetic_source_index += 1
            group.append(session)
            mappings.append(mapping)
        for replica_index in range(FIRST_ONLY_REPLICA_SETS):
            for source in templates:
                session, mapping = _clone_session(
                    source,
                    epoch_index=epoch_index,
                    synthetic_source_index=synthetic_source_index,
                    first_only_replica_index=replica_index,
                )
                synthetic_source_index += 1
                group.append(session)
                mappings.append(mapping)
        epoch_groups.append(tuple(group))

    sessions_per_epoch = len(epoch_groups[0])
    calls_per_epoch = sum(
        len(session.calls) for session in epoch_groups[0])
    first_calls_per_epoch = sessions_per_epoch
    resume_calls_per_epoch = calls_per_epoch - first_calls_per_epoch
    output_tokens_per_epoch = sum(
        call.output_tokens
        for session in epoch_groups[0]
        for call in session.calls
    )
    observed_counts = (
        sessions_per_epoch,
        calls_per_epoch,
        first_calls_per_epoch,
        resume_calls_per_epoch,
        output_tokens_per_epoch,
    )
    expected_counts = (
        EXPECTED_SESSIONS_PER_EPOCH,
        EXPECTED_CALLS_PER_EPOCH,
        EXPECTED_FIRST_CALLS_PER_EPOCH,
        EXPECTED_RESUME_CALLS_PER_EPOCH,
        EXPECTED_OUTPUT_TOKENS_PER_EPOCH,
    )
    if observed_counts != expected_counts:
        raise LivePressureScenarioError(
            "pinned pressure epoch counts changed: "
            f"observed={observed_counts}, expected={expected_counts}")

    measurement_ids = tuple(
        mapping.session_id
        for mapping in mappings
        if mapping.role == "measurement"
    )
    measurement_request_count = sum(
        len(session.calls)
        for epoch_index, group in enumerate(epoch_groups)
        if epoch_index in MEASUREMENT_EPOCHS
        for session in group
    )
    measurement_first_count = (
        sessions_per_epoch * len(MEASUREMENT_EPOCHS))
    measurement_resume_count = (
        measurement_request_count - measurement_first_count)
    all_epoch_kv = terminal_per_epoch * EPOCH_COUNT
    if all_epoch_kv != EXPECTED_TERMINAL_LOGICAL_KV_BYTES_ALL_EPOCHS:
        raise LivePressureScenarioError(
            "pinned all-epoch logical-KV working set changed")
    mapping_payload = [asdict(mapping) for mapping in mappings]
    manifest = LivePressureManifest(
        schema_version=SCENARIO_SCHEMA_VERSION,
        scenario_id=SCENARIO_ID,
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
        epoch_count=EPOCH_COUNT,
        warmup_epochs=WARMUP_EPOCHS,
        measurement_epochs=MEASUREMENT_EPOCHS,
        guard_epochs=GUARD_EPOCHS,
        first_only_replica_sets=FIRST_ONLY_REPLICA_SETS,
        sessions_per_epoch=sessions_per_epoch,
        calls_per_epoch=calls_per_epoch,
        first_calls_per_epoch=first_calls_per_epoch,
        resume_calls_per_epoch=resume_calls_per_epoch,
        output_tokens_per_epoch=output_tokens_per_epoch,
        measurement_session_ids=measurement_ids,
        measurement_request_count=measurement_request_count,
        measurement_first_call_count=measurement_first_count,
        measurement_resume_call_count=measurement_resume_count,
        epoch_mapping_sha256=stable_json_sha256(mapping_payload),
        prefill_service=service_audit,
        kv_pressure=KVPressureAudit(
            logical_kv_bytes_per_token=LOGICAL_KV_BYTES_PER_TOKEN,
            block_tokens=KV_BLOCK_TOKENS,
            terminal_logical_kv_bytes_per_epoch=terminal_per_epoch,
            terminal_logical_kv_bytes_all_epochs=all_epoch_kv,
            baseline_combined_usable_d_hbm_and_cpu_bytes=(
                BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES),
            terminal_excess_over_baseline_bytes=(
                all_epoch_kv
                - BASELINE_COMBINED_USABLE_D_HBM_AND_CPU_BYTES),
            semantics=(
                "terminal block-rounded logical-KV upper bound if every "
                "complete session reaches and retains its final turn "
                "concurrently; causal completion may reduce realized peak"),
        ),
        recommended_rates=RECOMMENDED_RATES,
        rate_semantics=(
            "system-wide external session starts per second; one of every "
            "21 sessions is a complete multi-turn session, so the long-session "
            "start rate is system rate divided by 21"),
        workload_semantics=(
            "TraceLab-selected low-output sessions with lineage-preserving "
            "global 250k context scaling plus TraceLab first-turn-only load; "
            "explicit pressure sensitivity, not empirical context lengths or "
            "equilibrium"),
        successor_release_semantics=(
            "successor call N+1 is released only after call N completion plus "
            "the unchanged recorded tool duration"),
        measurement_semantics=(
            "epochs 0-2 warm up, epochs 3-8 form the exact fixed measurement "
            "roster, epochs 9-11 guard, and every system fully drains"),
    )
    return LivePressureScenario(
        manifest=manifest,
        epoch_sessions=tuple(epoch_groups),
    )


def build(trace_path: str | Path) -> LivePressureScenario:
    """Scenario-factory entry point for the live ASTRA sweep runner."""

    return build_live_pressure_scenario(trace_path)


__all__ = [
    "LivePressureManifest",
    "LivePressureScenario",
    "LivePressureScenarioError",
    "RECOMMENDED_PILOT_RATES",
    "RECOMMENDED_RATES",
    "build",
    "build_live_pressure_scenario",
]

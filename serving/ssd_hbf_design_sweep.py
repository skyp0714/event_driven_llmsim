"""Reproducible sweep for the SSD-staged one-GPU/one-HBF design.

This campaign is intentionally separate from ``hbf_design_space_sweep``.
The historical runner models direct GPU-HBM-to-HBF migration and compares
against two GPU servers.  This runner freezes the corrected physical and
causal contract:

* baseline: one finite-HBM P4D4 GPU server with eight local SSDs;
* Oracle: one strict infinite-HBM P4D4 GPU server;
* design: that one GPU+SSD server plus one eight-card HBF server.

Every seed builds one immutable long-cold-context schedule at exactly
3 sessions/s.  The same tuple and exact measurement roster are passed to
the baseline, Oracle, and every design cell.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import time
from typing import Callable, Mapping, Optional, Sequence

from .core.gpu_pd_latency import load_p4d4_gpu_config
from .core.gpu_pd_single_system import (
    SingleFiniteHBMTieredBaseline,
    SingleStrictInfiniteHBMOracle,
)
from .core.hbf_comparison_cell import (
    DEFAULT_FIRST_TTFT_SECONDS,
    DEFAULT_RESUME_TTFT_SECONDS,
    DEFAULT_TPOT_MILLISECONDS,
    D_MAX_NUM_SEQS,
    MAX_NUM_BATCHED_TOKENS,
    MAX_PREFILL_CHUNK_TOKENS,
    P_MAX_NUM_SEQS,
    PINNED_GPU_CONFIG,
    PINNED_HBF_CONFIG,
    SHARED_MAX_NUM_SEQS,
    build_slo_thresholds,
    json_safe,
    summarize_measurement_requests,
    validate_causal_release_contract,
    validate_system_call_projection,
    write_json_atomic,
)
from .core.hbf_comparison_metrics import (
    CompletedRequest,
    aggregate_paired_seed_values,
    aggregate_seed_values,
)
from .core.hbf_comparison_workload import (
    ScheduledSession,
    stable_json_sha256,
)
from .core.hbf_design_tco import ActiveMemorySpec, lpddr_active_memory
from .core.hbf_full_model_latency import (
    HBFParallelLayout,
    load_hbf_server_config,
)
from .core.hbf_full_model_pool import derive_lpddr_workspace_bytes
from .core.ssd_hbf_tco import (
    SSDHBFTCOError,
    evaluate_ssd_hbf_tco,
)
from .core.tracelab_comparison_scenarios import (
    LongColdContextStressManifest,
    TraceLabComparisonScenario,
    load_long_cold_context_stress_scenario,
)
from .hbf_comparison_sweep import (
    default_trace_path,
    default_worker_count,
)


SSD_HBF_SWEEP_SCHEMA_VERSION = 3
SSD_HBF_CELL_SCHEMA_VERSION = 2
SSD_HBF_CONTRACT_KEY = "one-gpu-local-ssd-plus-one-hbf-staged-v1"
REQUIRED_SESSION_RATE = 3.0

BASELINE_CANDIDATE_KEY = "baseline_one_gpu_local_ssd"
ORACLE_CANDIDATE_KEY = "oracle_one_gpu_infinite_hbm"
BASELINE_SYSTEM_KEY = "one_gpu_local_ssd_baseline"
ORACLE_SYSTEM_KEY = "one_gpu_infinite_hbm_oracle"

SUPPORTED_LAYOUTS = ("tp4x2", "tp8_context")
LAYOUT_TO_SIMULATOR = {
    "tp4x2": "tp4",
    "tp8_context": "tp8_context",
}
LAYOUT_TO_TCO = {
    "tp4x2": "tp4x2",
    "tp8_context": "tp8",
}
SUPPORTED_MIGRATION_POLICIES = (
    "eager",
    "tool_immediate",
    "human_immediate",
    "tool_or_human_immediate",
    "delay_25ms",
    "delay_50ms",
    "delay_100ms",
    "delay_200ms",
    "delay_500ms",
    "delay_1000ms",
    "delay_1s",
    "delay_5s",
    "delay_30s",
    "delay_300s",
    "load_aware",
    "never",
)
DEFAULT_MIGRATION_POLICIES = (
    "tool_or_human_immediate",
    "delay_1s",
    "delay_5s",
    "delay_30s",
    "delay_300s",
    "load_aware",
    "never",
)
SMOKE_MIGRATION_POLICIES = (
    "tool_or_human_immediate",
    "delay_1s",
)
SUPPORTED_HBF_READ_MODES = ("demand", "prefetch")
DEFAULT_HBF_READ_MODES = SUPPORTED_HBF_READ_MODES
DEFAULT_SSD_HBF_SEEDS = (101, 102, 103)
SMOKE_SEEDS = (101, 102)

ORACLE_MEAN_JOINT_MIN = 0.95
ORACLE_EVERY_SEED_JOINT_MIN = 0.90
BASELINE_OVER_ORACLE_GOODPUT_CI95_UPPER_MAX = 0.10

BYTES_PER_GIB = 1024 ** 3
_EXECUTION_INPUTS = (
    Path("serving/ssd_hbf_design_sweep.py"),
    Path("serving/core/gpu_pd_single_system.py"),
    Path("serving/core/gpu_ssd_hbf_hybrid.py"),
    Path("serving/core/ssd_hbf_tco.py"),
    Path("serving/core/gpu_pd_tiered_node.py"),
    Path("serving/core/gpu_pd_oracle_node.py"),
    Path("serving/core/hbf_full_model_latency.py"),
    Path("serving/core/hbf_comparison_cell.py"),
    Path("serving/core/hbf_comparison_metrics.py"),
    Path("serving/core/hbf_comparison_workload.py"),
    Path("serving/core/tracelab_comparison_scenarios.py"),
    PINNED_GPU_CONFIG,
    PINNED_HBF_CONFIG,
)


class SSDHBFDesignSweepError(ValueError):
    """Raised when the corrected comparison contract is inconsistent."""


def _finite_positive(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise SSDHBFDesignSweepError(
            f"{name} must be positive and finite")
    return float(value)


def _slug(value: str) -> str:
    normalized = re.sub(
        r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    if not normalized:
        raise SSDHBFDesignSweepError(
            "design key component cannot be empty")
    return normalized


def parse_active_memory_spec(value: str) -> ActiveMemorySpec:
    """Parse ``kind:GiB:GB/s[:USD/GiB:W/GiB]``."""

    if not isinstance(value, str):
        raise SSDHBFDesignSweepError(
            "active-memory spec must be a string")
    fields = tuple(part.strip() for part in value.split(":"))
    if len(fields) not in (3, 5):
        raise SSDHBFDesignSweepError(
            "active-memory spec must be kind:GiB:GB/s or "
            "kind:GiB:GB/s:USD/GiB:W/GiB")
    kind = fields[0]
    try:
        capacity = float(fields[1])
        bandwidth = float(fields[2])
        capex = float(fields[3]) if len(fields) == 5 else None
        power = float(fields[4]) if len(fields) == 5 else None
    except ValueError as exc:
        raise SSDHBFDesignSweepError(
            f"invalid active-memory numeric field in {value!r}") from exc
    if kind == "lpddr" and len(fields) == 3:
        return lpddr_active_memory(
            capacity_gib_per_card=capacity,
            bandwidth_gbps_per_card=bandwidth,
        )
    if len(fields) != 5:
        raise SSDHBFDesignSweepError(
            f"{kind!r} requires explicit USD/GiB and W/GiB assumptions")
    return ActiveMemorySpec(
        kind=kind,
        capacity_gib_per_card=capacity,
        bandwidth_gbps_per_card=bandwidth,
        capex_usd_per_gib=capex,
        power_w_per_gib=power,
        assumption=(
            "Explicit SSD-staged design-sweep assumption; not a vendor "
            "quote or measured product specification."
        ),
    )


@dataclass(frozen=True)
class SSDHBFDesignSpec:
    """One logical layout/policy/memory point on one physical HBF host."""

    key: str
    hbf_layout: str
    migration_policy: str
    active_memory: ActiveMemorySpec
    hbf_read_mode: str = "demand"
    gpu_host_count: int = 1
    hbf_host_count: int = 1
    hbf_card_count: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or _slug(self.key) != self.key:
            raise SSDHBFDesignSweepError(
                "design key must be a lowercase filesystem slug")
        if self.hbf_layout not in SUPPORTED_LAYOUTS:
            raise SSDHBFDesignSweepError(
                f"hbf_layout must be one of {SUPPORTED_LAYOUTS!r}")
        if self.migration_policy not in SUPPORTED_MIGRATION_POLICIES:
            raise SSDHBFDesignSweepError(
                "migration_policy must be one of "
                f"{SUPPORTED_MIGRATION_POLICIES!r}")
        if not isinstance(self.active_memory, ActiveMemorySpec):
            raise SSDHBFDesignSweepError(
                "active_memory must be ActiveMemorySpec")
        if self.hbf_read_mode not in SUPPORTED_HBF_READ_MODES:
            raise SSDHBFDesignSweepError(
                "hbf_read_mode must be one of "
                f"{SUPPORTED_HBF_READ_MODES!r}")
        if (
            self.gpu_host_count != 1
            or self.hbf_host_count != 1
            or self.hbf_card_count != 8
        ):
            raise SSDHBFDesignSweepError(
                "every design must use one GPU and one eight-card HBF host")

    @property
    def simulator_layout(self) -> str:
        return LAYOUT_TO_SIMULATOR[self.hbf_layout]

    @property
    def tco_layout(self) -> str:
        return LAYOUT_TO_TCO[self.hbf_layout]

    def to_json_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["simulator_layout"] = self.simulator_layout
        result["tco_layout"] = self.tco_layout
        return result


def make_design_spec(
        *,
        hbf_layout: str,
        migration_policy: str,
        active_memory: ActiveMemorySpec,
        hbf_read_mode: str = "demand",
) -> SSDHBFDesignSpec:
    if hbf_layout not in SUPPORTED_LAYOUTS:
        raise SSDHBFDesignSweepError(
            f"hbf_layout must be one of {SUPPORTED_LAYOUTS!r}")
    if migration_policy not in SUPPORTED_MIGRATION_POLICIES:
        raise SSDHBFDesignSweepError(
            f"unsupported migration policy {migration_policy!r}")
    if hbf_read_mode not in SUPPORTED_HBF_READ_MODES:
        raise SSDHBFDesignSweepError(
            f"unsupported HBF read mode {hbf_read_mode!r}")
    memory_key = (
        f"{active_memory.kind}-"
        f"{active_memory.capacity_gib_per_card:g}gib-"
        f"{active_memory.bandwidth_gbps_per_card:g}gbps-"
        f"{active_memory.capex_usd_per_gib:g}usdpgib-"
        f"{active_memory.power_w_per_gib:g}wpgib"
    )
    return SSDHBFDesignSpec(
        key=_slug(
            f"ssd-hbf-{hbf_layout}-{migration_policy}-"
            f"{hbf_read_mode}-{memory_key}"),
        hbf_layout=hbf_layout,
        migration_policy=migration_policy,
        active_memory=active_memory,
        hbf_read_mode=hbf_read_mode,
    )


def build_design_grid(
        *,
        layouts: Sequence[str],
        migration_policies: Sequence[str],
        active_memories: Sequence[ActiveMemorySpec],
        hbf_read_modes: Sequence[str] = ("demand",),
) -> tuple[SSDHBFDesignSpec, ...]:
    if (
        not layouts
        or not migration_policies
        or not active_memories
        or not hbf_read_modes
    ):
        raise SSDHBFDesignSweepError(
            "layout, policy, and active-memory axes must be non-empty")
    if len(layouts) != len(set(layouts)):
        raise SSDHBFDesignSweepError("layout axis contains duplicates")
    if len(migration_policies) != len(set(migration_policies)):
        raise SSDHBFDesignSweepError(
            "migration-policy axis contains duplicates")
    if (
        len(hbf_read_modes) != len(set(hbf_read_modes))
        or any(
            mode not in SUPPORTED_HBF_READ_MODES
            for mode in hbf_read_modes
        )
    ):
        raise SSDHBFDesignSweepError(
            "HBF read modes must be unique members of "
            f"{SUPPORTED_HBF_READ_MODES!r}")
    specs = tuple(
        make_design_spec(
            hbf_layout=layout,
            migration_policy=policy,
            active_memory=memory,
            hbf_read_mode=read_mode,
        )
        for layout in layouts
        for policy in migration_policies
        for memory in active_memories
        for read_mode in hbf_read_modes
    )
    keys = [spec.key for spec in specs]
    if len(keys) != len(set(keys)):
        raise SSDHBFDesignSweepError(
            "grid contains duplicate design keys")
    return specs


def validate_design_workspace(
        spec: SSDHBFDesignSpec,
        *,
        max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS,
        max_num_seqs: int = SHARED_MAX_NUM_SEQS,
) -> dict[str, object]:
    layout = HBFParallelLayout.for_key(spec.simulator_layout)
    required = derive_lpddr_workspace_bytes(
        layout,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )
    capacity = int(round(
        spec.active_memory.capacity_gib_per_card * BYTES_PER_GIB))
    if capacity <= required:
        raise SSDHBFDesignSweepError(
            f"design {spec.key!r} active memory cannot hold workspace: "
            f"capacity={capacity}, required>{required}")
    return {
        "capacity_bytes_per_card": capacity,
        "workspace_bytes_per_card": required,
        "minimum_free_bytes_per_card": capacity - required,
        "simulator_layout": spec.simulator_layout,
    }


def validate_scenario_contract(
        scenario: TraceLabComparisonScenario,
) -> dict[str, object]:
    if not isinstance(scenario, TraceLabComparisonScenario):
        raise SSDHBFDesignSweepError(
            "scenario must be TraceLabComparisonScenario")
    manifest = scenario.manifest
    if not isinstance(manifest, LongColdContextStressManifest):
        raise SSDHBFDesignSweepError(
            "SSD-HBF sweep requires the pinned long-cold-context scenario")
    if manifest.equilibrium_workload:
        raise SSDHBFDesignSweepError(
            "long-cold comparison must remain non-equilibrium")
    roster = tuple(manifest.measurement_request_identities)
    if not roster or len(roster) != len(set(roster)):
        raise SSDHBFDesignSweepError(
            "measurement roster must be non-empty and unique")
    roster_hash = stable_json_sha256(list(roster))
    if roster_hash != manifest.measurement_request_identities_sha256:
        raise SSDHBFDesignSweepError(
            "measurement roster hash disagrees with scenario manifest")
    manifest.arrival_contract.validate_rate(REQUIRED_SESSION_RATE)
    return {
        "scenario_id": manifest.scenario_id,
        "manifest_sha256": stable_json_sha256(manifest.to_dict()),
        "measurement_roster_sha256": roster_hash,
        "measurement_identity_count": len(roster),
        "required_session_rate": REQUIRED_SESSION_RATE,
    }


def make_design_system(
        *,
        repo_root: Path,
        spec: SSDHBFDesignSpec,
):
    """Construct exactly one GPU+SSD host and one eight-card HBF host."""

    from .core.gpu_ssd_hbf_hybrid import SSDStagedGPUHBFSystem

    workspace = validate_design_workspace(spec)
    root = Path(repo_root)
    gpu_hardware = load_p4d4_gpu_config(root / PINNED_GPU_CONFIG)
    hbf_hardware, _ = load_hbf_server_config(root / PINNED_HBF_CONFIG)
    hbf_hardware = replace(
        hbf_hardware,
        lpddr_capacity_bytes_per_card=int(
            workspace["capacity_bytes_per_card"]),
        lpddr_bandwidth_gbps_per_card=(
            spec.active_memory.bandwidth_gbps_per_card),
        hbf_read_prefetch_enabled=(
            spec.hbf_read_mode == "prefetch"),
    )
    hbf_hardware.validate()
    return SSDStagedGPUHBFSystem(
        repo_root=root,
        gpu_hardware=gpu_hardware,
        hbf_hardware=hbf_hardware,
        hbf_layout=spec.simulator_layout,
        promotion_policy=spec.migration_policy,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=SHARED_MAX_NUM_SEQS,
        p_max_num_seqs=P_MAX_NUM_SEQS,
        d_max_num_seqs=D_MAX_NUM_SEQS,
        max_prefill_chunk_tokens=MAX_PREFILL_CHUNK_TOKENS,
        validate_every_event=False,
    )


def make_reference_system(
        *,
        repo_root: Path,
        candidate_kind: str,
):
    root = Path(repo_root)
    hardware = load_p4d4_gpu_config(root / PINNED_GPU_CONFIG)
    common = {
        "repo_root": root,
        "hardware": hardware,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": SHARED_MAX_NUM_SEQS,
        "p_max_num_seqs": P_MAX_NUM_SEQS,
        "d_max_num_seqs": D_MAX_NUM_SEQS,
        "max_prefill_chunk_tokens": MAX_PREFILL_CHUNK_TOKENS,
        "validate_every_event": False,
    }
    if candidate_kind == "baseline":
        return SingleFiniteHBMTieredBaseline(
            policy="ssd_direct", **common)
    if candidate_kind == "oracle":
        return SingleStrictInfiniteHBMOracle(**common)
    raise SSDHBFDesignSweepError(
        f"unknown reference candidate kind {candidate_kind!r}")


def _identity(request: CompletedRequest) -> str:
    return (
        f"{request.key.session_id}::call-"
        f"{request.key.sub_request_index}"
    )


def _measurement_requests(
        completed: Sequence[CompletedRequest],
        measurement_identities: Sequence[str],
) -> tuple[CompletedRequest, ...]:
    roster = tuple(measurement_identities)
    if not roster or len(roster) != len(set(roster)):
        raise SSDHBFDesignSweepError(
            "measurement roster must be non-empty and unique")
    indexed = {_identity(request): request for request in completed}
    if len(indexed) != len(completed):
        raise SSDHBFDesignSweepError(
            "completed requests contain duplicate identities")
    missing = tuple(identity for identity in roster if identity not in indexed)
    if missing:
        raise SSDHBFDesignSweepError(
            f"measurement roster has missing completions: {missing[:5]}")
    return tuple(indexed[identity] for identity in roster)


def _run_cell_system(
        *,
        system: object,
        candidate_kind: str,
        candidate_key: str,
        scheduled_sessions: tuple[ScheduledSession, ...],
        session_rate: float,
        seed: int,
        measurement_identities: Sequence[str],
        thresholds,
        design: Optional[SSDHBFDesignSpec],
) -> dict[str, object]:
    start = time.perf_counter_ns()
    completed = tuple(system.run(scheduled_sessions))
    wall_ns = time.perf_counter_ns() - start
    projection_sha256 = validate_system_call_projection(
        scheduled_sessions, system.call_specs)
    full_drain = validate_causal_release_contract(
        scheduled_sessions, completed)
    measured = _measurement_requests(
        completed, measurement_identities)
    summary = summarize_measurement_requests(
        measured,
        session_rate=session_rate,
        thresholds=thresholds,
    )
    roster_hash = stable_json_sha256(
        list(measurement_identities))
    if design is None:
        topology = {
            "gpu_host_count": 1,
            "hbf_host_count": 0,
            "h100_card_count": 8,
            "hbf_card_count": 0,
            "local_ssd_device_count": 8,
        }
    else:
        topology = {
            "gpu_host_count": 1,
            "hbf_host_count": 1,
            "h100_card_count": 8,
            "hbf_card_count": 8,
            "local_ssd_device_count": 8,
        }
    result = {
        "schema_version": SSD_HBF_CELL_SCHEMA_VERSION,
        "comparison_contract": SSD_HBF_CONTRACT_KEY,
        "candidate_kind": candidate_kind,
        "candidate_key": candidate_key,
        "system_key": (
            BASELINE_SYSTEM_KEY
            if candidate_kind == "baseline"
            else ORACLE_SYSTEM_KEY
            if candidate_kind == "oracle"
            else candidate_key
        ),
        "seed": seed,
        "session_rate": session_rate,
        "design": (
            None if design is None else design.to_json_dict()),
        "physical_topology": topology,
        "measurement_roster": {
            "identity_count": len(measurement_identities),
            "ordered_identities_sha256": roster_hash,
        },
        "normalized_system_call_projection_sha256": projection_sha256,
        "full_drain": full_drain,
        "summary": summary,
        "execution_observation": {
            "simulated_horizon_ns": system.current_ns,
            "elapsed_wall_time_ns": wall_ns,
        },
        "system_report": system.report(),
    }
    safe = json_safe(result)
    json.dumps(safe, allow_nan=False, sort_keys=True)
    return safe


def run_reference_cell(
        *,
        repo_root: Path,
        candidate_kind: str,
        scheduled_sessions: tuple[ScheduledSession, ...],
        session_rate: float,
        seed: int,
        measurement_identities: Sequence[str],
        first_ttft_seconds: float = DEFAULT_FIRST_TTFT_SECONDS,
        resume_ttft_seconds: float = DEFAULT_RESUME_TTFT_SECONDS,
        tpot_milliseconds: float = DEFAULT_TPOT_MILLISECONDS,
) -> dict[str, object]:
    if candidate_kind not in {"baseline", "oracle"}:
        raise SSDHBFDesignSweepError(
            "reference kind must be baseline or oracle")
    rate = _finite_positive("session_rate", session_rate)
    if rate != REQUIRED_SESSION_RATE:
        raise SSDHBFDesignSweepError(
            f"corrected comparison requires rate={REQUIRED_SESSION_RATE}")
    thresholds = build_slo_thresholds(
        first_ttft_seconds=first_ttft_seconds,
        resume_ttft_seconds=resume_ttft_seconds,
        tpot_milliseconds=tpot_milliseconds,
    )
    return _run_cell_system(
        system=make_reference_system(
            repo_root=repo_root,
            candidate_kind=candidate_kind,
        ),
        candidate_kind=candidate_kind,
        candidate_key=(
            BASELINE_CANDIDATE_KEY
            if candidate_kind == "baseline"
            else ORACLE_CANDIDATE_KEY
        ),
        scheduled_sessions=scheduled_sessions,
        session_rate=rate,
        seed=seed,
        measurement_identities=measurement_identities,
        thresholds=thresholds,
        design=None,
    )


def run_design_cell(
        *,
        repo_root: Path,
        spec: SSDHBFDesignSpec,
        scheduled_sessions: tuple[ScheduledSession, ...],
        session_rate: float,
        seed: int,
        measurement_identities: Sequence[str],
        first_ttft_seconds: float = DEFAULT_FIRST_TTFT_SECONDS,
        resume_ttft_seconds: float = DEFAULT_RESUME_TTFT_SECONDS,
        tpot_milliseconds: float = DEFAULT_TPOT_MILLISECONDS,
) -> dict[str, object]:
    rate = _finite_positive("session_rate", session_rate)
    if rate != REQUIRED_SESSION_RATE:
        raise SSDHBFDesignSweepError(
            f"corrected comparison requires rate={REQUIRED_SESSION_RATE}")
    thresholds = build_slo_thresholds(
        first_ttft_seconds=first_ttft_seconds,
        resume_ttft_seconds=resume_ttft_seconds,
        tpot_milliseconds=tpot_milliseconds,
    )
    return _run_cell_system(
        system=make_design_system(repo_root=repo_root, spec=spec),
        candidate_kind="design",
        candidate_key=spec.key,
        scheduled_sessions=scheduled_sessions,
        session_rate=rate,
        seed=seed,
        measurement_identities=measurement_identities,
        thresholds=thresholds,
        design=spec,
    )


@dataclass(frozen=True)
class _CellTask:
    repo_root: Path
    candidate_kind: str
    candidate_key: str
    seed: int
    session_rate: float
    scheduled_sessions: tuple[ScheduledSession, ...]
    measurement_identities: tuple[str, ...]
    design: Optional[SSDHBFDesignSpec]
    first_ttft_seconds: float
    resume_ttft_seconds: float
    tpot_milliseconds: float
    scenario_contract_sha256: str
    execution_inputs_sha256: str


def _execution_inputs_sha256(repo_root: Path) -> str:
    digest = hashlib.sha256()
    root = Path(repo_root)
    for relative in _EXECUTION_INPUTS:
        path = root / relative
        if not path.is_file():
            raise SSDHBFDesignSweepError(
                f"missing execution input {path}")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_tasks(
        *,
        repo_root: Path,
        scenario: TraceLabComparisonScenario,
        designs: Sequence[SSDHBFDesignSpec],
        seeds: Sequence[int],
        session_rate: float = REQUIRED_SESSION_RATE,
        first_ttft_seconds: float = DEFAULT_FIRST_TTFT_SECONDS,
        resume_ttft_seconds: float = DEFAULT_RESUME_TTFT_SECONDS,
        tpot_milliseconds: float = DEFAULT_TPOT_MILLISECONDS,
) -> tuple[_CellTask, ...]:
    contract = validate_scenario_contract(scenario)
    rate = _finite_positive("session_rate", session_rate)
    if rate != REQUIRED_SESSION_RATE:
        raise SSDHBFDesignSweepError(
            f"corrected comparison requires rate={REQUIRED_SESSION_RATE}")
    design_values = tuple(designs)
    if not design_values:
        raise SSDHBFDesignSweepError("designs cannot be empty")
    if len({design.key for design in design_values}) != len(design_values):
        raise SSDHBFDesignSweepError("design keys contain duplicates")
    for design in design_values:
        validate_design_workspace(design)
    seed_values = tuple(seeds)
    if len(seed_values) < 2:
        raise SSDHBFDesignSweepError(
            "eligibility CI requires at least two seeds")
    if len(seed_values) != len(set(seed_values)):
        raise SSDHBFDesignSweepError("seeds contain duplicates")
    if any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in seed_values):
        raise SSDHBFDesignSweepError("seeds must be integers")

    execution_hash = _execution_inputs_sha256(repo_root)
    roster = tuple(
        scenario.manifest.measurement_request_identities)
    scenario_hash = str(contract["manifest_sha256"])
    tasks = []
    for seed in seed_values:
        schedule = scenario.build_offered_plan(
            seed=seed).at_rate(rate)
        if not isinstance(schedule, tuple):
            raise SSDHBFDesignSweepError(
                "scenario must freeze schedules as tuples")
        common = {
            "repo_root": Path(repo_root),
            "seed": seed,
            "session_rate": rate,
            "scheduled_sessions": schedule,
            "measurement_identities": roster,
            "first_ttft_seconds": first_ttft_seconds,
            "resume_ttft_seconds": resume_ttft_seconds,
            "tpot_milliseconds": tpot_milliseconds,
            "scenario_contract_sha256": scenario_hash,
            "execution_inputs_sha256": execution_hash,
        }
        tasks.extend((
            _CellTask(
                candidate_kind="baseline",
                candidate_key=BASELINE_CANDIDATE_KEY,
                design=None,
                **common,
            ),
            _CellTask(
                candidate_kind="oracle",
                candidate_key=ORACLE_CANDIDATE_KEY,
                design=None,
                **common,
            ),
        ))
        tasks.extend(
            _CellTask(
                candidate_kind="design",
                candidate_key=design.key,
                design=design,
                **common,
            )
            for design in design_values
        )
    return tuple(tasks)


def _task_contract(task: _CellTask) -> dict[str, object]:
    normalized = json_safe({
        "schema_version": SSD_HBF_CELL_SCHEMA_VERSION,
        "comparison_contract": SSD_HBF_CONTRACT_KEY,
        "candidate_kind": task.candidate_kind,
        "candidate_key": task.candidate_key,
        "seed": task.seed,
        "session_rate": task.session_rate,
        "schedule_sha256": stable_json_sha256([
            asdict(session) for session in task.scheduled_sessions
        ]),
        "measurement_identities_sha256": stable_json_sha256(
            list(task.measurement_identities)),
        "design": (
            None if task.design is None
            else task.design.to_json_dict()),
        "thresholds": {
            "first_ttft_seconds": task.first_ttft_seconds,
            "resume_ttft_seconds": task.resume_ttft_seconds,
            "tpot_milliseconds": task.tpot_milliseconds,
        },
        "scenario_contract_sha256": task.scenario_contract_sha256,
        "execution_inputs_sha256": task.execution_inputs_sha256,
    })
    if not isinstance(normalized, dict):
        raise AssertionError("task contract did not normalize to an object")
    return normalized


def _seal_record(
        task: _CellTask,
        record: Mapping[str, object],
) -> dict[str, object]:
    contract = _task_contract(task)
    sealed = {
        **dict(record),
        "cell_contract": contract,
        "cell_contract_sha256": stable_json_sha256(contract),
    }
    sealed["result_payload_sha256"] = stable_json_sha256(sealed)
    return sealed


def _execute_task(task: _CellTask) -> dict[str, object]:
    common = {
        "repo_root": task.repo_root,
        "scheduled_sessions": task.scheduled_sessions,
        "session_rate": task.session_rate,
        "seed": task.seed,
        "measurement_identities": task.measurement_identities,
        "first_ttft_seconds": task.first_ttft_seconds,
        "resume_ttft_seconds": task.resume_ttft_seconds,
        "tpot_milliseconds": task.tpot_milliseconds,
    }
    if task.candidate_kind == "design":
        if task.design is None:
            raise AssertionError("design task lacks a design spec")
        result = run_design_cell(spec=task.design, **common)
    else:
        result = run_reference_cell(
            candidate_kind=task.candidate_kind, **common)
    return _seal_record(task, result)


_AGGREGATE_METRICS = {
    "slo_good_output_tokens_per_second": (
        "offered_load_normalized_output_token_goodput", "value"),
    "joint_slo_pass_fraction": (
        "slo", "all_slo_pass_fraction"),
    "first_ttft_p95_ns": (
        "latency_distributions_ns", "first_ttft", "p95_ns"),
    "resume_ttft_p95_ns": (
        "latency_distributions_ns", "resume_ttft", "p95_ns"),
    "tpot_p95_ns": (
        "latency_distributions_ns", "tpot_eligible", "p95_ns"),
}


def _summary_metric(
        record: Mapping[str, object],
        path: Sequence[str],
) -> Optional[float]:
    value: object = record["summary"]
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise SSDHBFDesignSweepError(
                f"cell lacks summary metric path {tuple(path)!r}")
        value = value[key]
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SSDHBFDesignSweepError(
            f"cell metric {tuple(path)!r} is not finite")
    return float(value)


def _aggregate_candidate(
        records_by_seed: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    metrics = {}
    for name, path in _AGGREGATE_METRICS.items():
        values = {
            seed: value
            for seed, record in records_by_seed.items()
            if (value := _summary_metric(record, path)) is not None
        }
        metrics[name] = (
            None if not values
            else asdict(aggregate_seed_values(values))
        )
    return metrics


def evaluate_reference_eligibility(
        *,
        baseline_goodput_by_seed: Mapping[int, float],
        oracle_goodput_by_seed: Mapping[int, float],
        oracle_joint_slo_by_seed: Mapping[int, float],
) -> dict[str, object]:
    """Apply the fail-closed baseline-much-less-than-Oracle audit gate."""

    if (
        set(baseline_goodput_by_seed)
        != set(oracle_goodput_by_seed)
        or set(oracle_goodput_by_seed)
        != set(oracle_joint_slo_by_seed)
    ):
        raise SSDHBFDesignSweepError(
            "reference eligibility inputs have unpaired seeds")
    oracle_joint = asdict(aggregate_seed_values(
        oracle_joint_slo_by_seed))
    paired = aggregate_paired_seed_values(
        oracle_goodput_by_seed,
        baseline_goodput_by_seed,
    )
    paired_json = asdict(paired)
    ratio = paired.candidate_over_reference
    failures = []
    if oracle_joint["mean"] < ORACLE_MEAN_JOINT_MIN:
        failures.append("oracle_mean_joint_slo_below_0.95")
    below_seed = tuple(
        seed for seed, value in sorted(
            oracle_joint_slo_by_seed.items())
        if value < ORACLE_EVERY_SEED_JOINT_MIN
    )
    if below_seed:
        failures.append("oracle_seed_joint_slo_below_0.90")
    if ratio is None:
        failures.append("baseline_over_oracle_ratio_unavailable")
        ratio_upper = None
    else:
        ratio_upper = ratio.ci95_upper
        if ratio_upper is None:
            failures.append(
                "baseline_over_oracle_ci95_unavailable")
        elif ratio_upper > (
                BASELINE_OVER_ORACLE_GOODPUT_CI95_UPPER_MAX):
            failures.append(
                "baseline_over_oracle_ci95_upper_above_0.10")
    return {
        "eligible": not failures,
        "failures": failures,
        "thresholds": {
            "oracle_mean_joint_slo_min": ORACLE_MEAN_JOINT_MIN,
            "oracle_every_seed_joint_slo_min": (
                ORACLE_EVERY_SEED_JOINT_MIN),
            "baseline_over_oracle_goodput_ci95_upper_max": (
                BASELINE_OVER_ORACLE_GOODPUT_CI95_UPPER_MAX),
        },
        "oracle_joint_slo": oracle_joint,
        "oracle_joint_slo_below_threshold_seeds": list(below_seed),
        "paired_baseline_over_oracle_goodput": paired_json,
        "observed_baseline_over_oracle_ci95_upper": ratio_upper,
        "semantics": (
            "candidate_over_reference is baseline/Oracle paired by seed; "
            "the two-sided Student-t 95% CI upper bound must be <=0.10"
        ),
    }


def pareto_frontier(
        points: Mapping[str, tuple[float, float]],
) -> tuple[str, ...]:
    """Return keys not dominated in goodput-max/TCO-min space."""

    frontier = []
    for key, (goodput, tco) in points.items():
        dominated = any(
            (
                other_goodput >= goodput
                and other_tco <= tco
                and (
                    other_goodput > goodput
                    or other_tco < tco
                )
            )
            for other_key, (other_goodput, other_tco)
            in points.items()
            if other_key != key
        )
        if not dominated:
            frontier.append(key)
    return tuple(sorted(frontier))


def aggregate_cell_records(
        records: Sequence[Mapping[str, object]],
        designs: Sequence[SSDHBFDesignSpec],
        *,
        require_eligibility: bool = True,
) -> dict[str, object]:
    design_by_key = {design.key: design for design in designs}
    if not design_by_key:
        raise SSDHBFDesignSweepError("designs cannot be empty")
    expected_candidates = {
        BASELINE_CANDIDATE_KEY,
        ORACLE_CANDIDATE_KEY,
        *design_by_key,
    }
    grouped: dict[
        float, dict[str, dict[int, Mapping[str, object]]]
    ] = {}
    roster_hash: Optional[str] = None
    for record in records:
        rate = float(record["session_rate"])
        if rate != REQUIRED_SESSION_RATE:
            raise SSDHBFDesignSweepError(
                f"unexpected session rate {rate}")
        key = str(record["candidate_key"])
        seed = int(record["seed"])
        by_seed = grouped.setdefault(
            rate, {}).setdefault(key, {})
        if seed in by_seed:
            raise SSDHBFDesignSweepError(
                f"duplicate cell for rate={rate}, key={key}, seed={seed}")
        by_seed[seed] = record
        observed_roster = record.get("measurement_roster")
        if observed_roster is not None:
            if not isinstance(observed_roster, Mapping):
                raise SSDHBFDesignSweepError(
                    "measurement_roster must be a mapping")
            observed_hash = observed_roster.get(
                "ordered_identities_sha256")
            if (
                not isinstance(observed_hash, str)
                or len(observed_hash) != 64
            ):
                raise SSDHBFDesignSweepError(
                    "cell has invalid measurement roster hash")
            if roster_hash is None:
                roster_hash = observed_hash
            elif roster_hash != observed_hash:
                raise SSDHBFDesignSweepError(
                    "cells use different measurement rosters")
    if not grouped:
        raise SSDHBFDesignSweepError("cell records cannot be empty")

    rate_rows = []
    for rate, candidates in sorted(grouped.items()):
        if set(candidates) != expected_candidates:
            raise SSDHBFDesignSweepError(
                f"incomplete candidate cohort for rate={rate}: "
                f"missing={sorted(expected_candidates - set(candidates))}, "
                f"extra={sorted(set(candidates) - expected_candidates)}")
        seeds = set(candidates[BASELINE_CANDIDATE_KEY])
        if len(seeds) < 2:
            raise SSDHBFDesignSweepError(
                "eligibility CI requires at least two paired seeds")
        for key, by_seed in candidates.items():
            if set(by_seed) != seeds:
                raise SSDHBFDesignSweepError(
                    f"unpaired seeds for candidate={key}")

        aggregates = {
            key: _aggregate_candidate(by_seed)
            for key, by_seed in candidates.items()
        }

        def seed_values(candidate: str, metric: str) -> dict[int, float]:
            path = _AGGREGATE_METRICS[metric]
            result = {}
            for seed, record in candidates[candidate].items():
                value = _summary_metric(record, path)
                if value is None:
                    raise SSDHBFDesignSweepError(
                        f"{candidate} metric {metric} cannot be null")
                result[seed] = value
            return result

        baseline_goodput = seed_values(
            BASELINE_CANDIDATE_KEY,
            "slo_good_output_tokens_per_second",
        )
        oracle_goodput = seed_values(
            ORACLE_CANDIDATE_KEY,
            "slo_good_output_tokens_per_second",
        )
        oracle_joint = seed_values(
            ORACLE_CANDIDATE_KEY,
            "joint_slo_pass_fraction",
        )
        eligibility = evaluate_reference_eligibility(
            baseline_goodput_by_seed=baseline_goodput,
            oracle_goodput_by_seed=oracle_goodput,
            oracle_joint_slo_by_seed=oracle_joint,
        )
        if require_eligibility and not eligibility["eligible"]:
            raise SSDHBFDesignSweepError(
                "reference eligibility gate failed: "
                + ", ".join(eligibility["failures"]))

        baseline_mean = aggregates[BASELINE_CANDIDATE_KEY][
            "slo_good_output_tokens_per_second"]["mean"]
        oracle_mean = aggregates[ORACLE_CANDIDATE_KEY][
            "slo_good_output_tokens_per_second"]["mean"]
        design_rows = []
        pareto_points = {}
        for key in sorted(design_by_key):
            spec = design_by_key[key]
            values = seed_values(
                key, "slo_good_output_tokens_per_second")
            paired_baseline = asdict(aggregate_paired_seed_values(
                baseline_goodput, values))
            paired_oracle = asdict(aggregate_paired_seed_values(
                oracle_goodput, values))
            if eligibility["eligible"] and baseline_mean > 0.0:
                try:
                    tco = evaluate_ssd_hbf_tco(
                        hbf_layout=spec.tco_layout,
                        active_memory=spec.active_memory,
                        baseline_slo_good_output_tokens_per_second=(
                            baseline_mean),
                        proposed_slo_good_output_tokens_per_second=(
                            aggregates[key][
                                "slo_good_output_tokens_per_second"]["mean"]),
                        oracle_slo_good_output_tokens_per_second=oracle_mean,
                    ).to_json_dict()
                    tco_unavailable_reason = None
                except SSDHBFTCOError as exc:
                    tco = None
                    tco_unavailable_reason = str(exc)
            else:
                tco = None
                tco_unavailable_reason = (
                    "reference eligibility failed or baseline goodput is zero")
            row = {
                "design": spec.to_json_dict(),
                "metrics": aggregates[key],
                "paired_vs_baseline_goodput": paired_baseline,
                "paired_vs_oracle_goodput": paired_oracle,
                "tco": tco,
                "tco_unavailable_reason": tco_unavailable_reason,
            }
            design_rows.append(row)
            mean = aggregates[key][
                "slo_good_output_tokens_per_second"]["mean"]
            if tco is not None and mean > 0.0:
                pareto_points[key] = (
                    mean,
                    tco["proposed_cost"]["five_year_tco_usd"],
                )
        frontier = pareto_frontier(pareto_points)
        for row in design_rows:
            row["performance_tco_pareto"] = (
                row["design"]["key"] in frontier)
        rate_rows.append({
            "session_rate": rate,
            "reference_eligibility": eligibility,
            "references": {
                BASELINE_CANDIDATE_KEY: aggregates[
                    BASELINE_CANDIDATE_KEY],
                ORACLE_CANDIDATE_KEY: aggregates[
                    ORACLE_CANDIDATE_KEY],
            },
            "designs": design_rows,
            "performance_tco_pareto_design_keys": list(frontier),
            "performance_ranking": [
                key for key, _ in sorted(
                    (
                        (
                            row["design"]["key"],
                            row["metrics"][
                                "slo_good_output_tokens_per_second"]["mean"],
                        )
                        for row in design_rows
                    ),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
        })
    return {
        "schema_version": SSD_HBF_SWEEP_SCHEMA_VERSION,
        "comparison_contract": SSD_HBF_CONTRACT_KEY,
        "measurement_roster_sha256": roster_hash,
        "rates": rate_rows,
    }


def _strict_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SSDHBFDesignSweepError(
            f"cannot resume invalid cell {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SSDHBFDesignSweepError(
            f"resumable cell is not a JSON object: {path}")
    return value


def _load_resumable_cell(
        path: Path,
        task: _CellTask,
) -> dict[str, object]:
    record = _strict_json_object(path)
    contract = _task_contract(task)
    if (
        record.get("cell_contract") != contract
        or record.get("cell_contract_sha256")
        != stable_json_sha256(contract)
    ):
        raise SSDHBFDesignSweepError(
            f"resumable cell contract mismatch: {path}")
    observed_payload_hash = record.get("result_payload_sha256")
    unsealed = dict(record)
    unsealed.pop("result_payload_sha256", None)
    if (
        not isinstance(observed_payload_hash, str)
        or observed_payload_hash != stable_json_sha256(unsealed)
    ):
        raise SSDHBFDesignSweepError(
            f"resumable cell payload hash mismatch: {path}")
    for field, expected in (
        ("candidate_kind", task.candidate_kind),
        ("candidate_key", task.candidate_key),
        ("seed", task.seed),
        ("session_rate", task.session_rate),
    ):
        if record.get(field) != expected:
            raise SSDHBFDesignSweepError(
                f"resumable cell coordinate mismatch for {field}: {path}")
    roster = record.get("measurement_roster")
    if (
        not isinstance(roster, Mapping)
        or roster.get("ordered_identities_sha256")
        != stable_json_sha256(list(task.measurement_identities))
    ):
        raise SSDHBFDesignSweepError(
            f"resumable cell measurement roster mismatch: {path}")
    _summary_metric(
        record,
        _AGGREGATE_METRICS[
            "slo_good_output_tokens_per_second"],
    )
    return record


def _task_path(root: Path, task: _CellTask) -> Path:
    return (
        root / "cells"
        / f"rate-{task.session_rate:g}"
        / f"seed-{task.seed}"
        / f"{task.candidate_key}.json"
    )


def _write_summary_csv(
        path: Path,
        aggregate: Mapping[str, object],
) -> None:
    fields = (
        "session_rate",
        "design_key",
        "hbf_layout",
        "gpu_host_count",
        "hbf_host_count",
        "migration_policy",
        "hbf_read_mode",
        "active_memory_kind",
        "active_memory_gib_per_card",
        "active_memory_gbps_per_card",
        "goodput_mean",
        "goodput_ci95_lower",
        "goodput_ci95_upper",
        "baseline_goodput_mean",
        "oracle_goodput_mean",
        "goodput_ratio_to_baseline_mean",
        "goodput_ratio_to_oracle_mean",
        "tco_lifetime_years",
        "five_year_tco_usd",
        "tco_usd_per_million_slo_good_tokens",
        "baseline_it_power_w",
        "proposed_it_power_w",
        "incremental_it_power_w",
        "proposed_it_power_ratio_to_baseline",
        "baseline_five_year_facility_energy_kwh",
        "proposed_five_year_facility_energy_kwh",
        "incremental_five_year_facility_energy_kwh",
        "proposed_facility_energy_ratio_to_baseline",
        "meets_token_value_break_even",
        "performance_tco_pareto",
    )
    rows = []
    for rate_row in aggregate["rates"]:
        baseline = rate_row["references"][BASELINE_CANDIDATE_KEY][
            "slo_good_output_tokens_per_second"]["mean"]
        oracle = rate_row["references"][ORACLE_CANDIDATE_KEY][
            "slo_good_output_tokens_per_second"]["mean"]
        for row in rate_row["designs"]:
            design = row["design"]
            memory = design["active_memory"]
            goodput = row["metrics"][
                "slo_good_output_tokens_per_second"]
            tco = row["tco"]
            baseline_ratio = row["paired_vs_baseline_goodput"][
                "candidate_over_reference"]
            oracle_ratio = row["paired_vs_oracle_goodput"][
                "candidate_over_reference"]
            rows.append({
                "session_rate": rate_row["session_rate"],
                "design_key": design["key"],
                "hbf_layout": design["hbf_layout"],
                "gpu_host_count": design["gpu_host_count"],
                "hbf_host_count": design["hbf_host_count"],
                "migration_policy": design["migration_policy"],
                "hbf_read_mode": design["hbf_read_mode"],
                "active_memory_kind": memory["kind"],
                "active_memory_gib_per_card": (
                    memory["capacity_gib_per_card"]),
                "active_memory_gbps_per_card": (
                    memory["bandwidth_gbps_per_card"]),
                "goodput_mean": goodput["mean"],
                "goodput_ci95_lower": goodput["ci95_lower"],
                "goodput_ci95_upper": goodput["ci95_upper"],
                "baseline_goodput_mean": baseline,
                "oracle_goodput_mean": oracle,
                "goodput_ratio_to_baseline_mean": (
                    None if baseline_ratio is None
                    else baseline_ratio["mean"]),
                "goodput_ratio_to_oracle_mean": (
                    None if oracle_ratio is None
                    else oracle_ratio["mean"]),
                "tco_lifetime_years": (
                    None if tco is None else
                    tco["power_energy_comparison"]["lifetime_years"]),
                "five_year_tco_usd": (
                    None if tco is None else
                    tco["proposed_cost"]["five_year_tco_usd"]),
                "tco_usd_per_million_slo_good_tokens": (
                    None if tco is None else
                    tco["proposed_token_economics"][
                        "tco_usd_per_million_slo_good_output_tokens"]),
                "baseline_it_power_w": (
                    None if tco is None else
                    tco["power_energy_comparison"][
                        "baseline_it_power_w"]),
                "proposed_it_power_w": (
                    None if tco is None else
                    tco["power_energy_comparison"][
                        "proposed_it_power_w"]),
                "incremental_it_power_w": (
                    None if tco is None else
                    tco["power_energy_comparison"][
                        "incremental_it_power_w"]),
                "proposed_it_power_ratio_to_baseline": (
                    None if tco is None else
                    tco["power_energy_comparison"][
                        "proposed_it_power_ratio_to_baseline"]),
                "baseline_five_year_facility_energy_kwh": (
                    None if tco is None else
                    tco["power_energy_comparison"][
                        "baseline_five_year_facility_energy_kwh"]),
                "proposed_five_year_facility_energy_kwh": (
                    None if tco is None else
                    tco["power_energy_comparison"][
                        "proposed_five_year_facility_energy_kwh"]),
                "incremental_five_year_facility_energy_kwh": (
                    None if tco is None else
                    tco["power_energy_comparison"][
                        "incremental_five_year_facility_energy_kwh"]),
                "proposed_facility_energy_ratio_to_baseline": (
                    None if tco is None else
                    tco["power_energy_comparison"][
                        "proposed_facility_energy_ratio_to_baseline"]),
                "meets_token_value_break_even": (
                    None if tco is None else
                    tco[
                        "proposed_meets_or_exceeds_goodput_break_even"]),
                "performance_tco_pareto": (
                    row["performance_tco_pareto"]),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open(
            "w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_design_space(
        *,
        repo_root: Path,
        output_root: Path,
        scenario: TraceLabComparisonScenario,
        designs: Sequence[SSDHBFDesignSpec],
        seeds: Sequence[int],
        workers: int = 1,
        session_rate: float = REQUIRED_SESSION_RATE,
        first_ttft_seconds: float = DEFAULT_FIRST_TTFT_SECONDS,
        resume_ttft_seconds: float = DEFAULT_RESUME_TTFT_SECONDS,
        tpot_milliseconds: float = DEFAULT_TPOT_MILLISECONDS,
        resume: bool = False,
        progress: Optional[Callable[[Mapping[str, object]], None]] = None,
) -> tuple[dict[str, object], Path]:
    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not resume:
        raise SSDHBFDesignSweepError(
            f"output directory is not empty: {root}")
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers <= 0
    ):
        raise SSDHBFDesignSweepError(
            "workers must be a positive integer")
    scenario_contract = validate_scenario_contract(scenario)
    tasks = build_tasks(
        repo_root=Path(repo_root).resolve(),
        scenario=scenario,
        designs=designs,
        seeds=seeds,
        session_rate=session_rate,
        first_ttft_seconds=first_ttft_seconds,
        resume_ttft_seconds=resume_ttft_seconds,
        tpot_milliseconds=tpot_milliseconds,
    )
    root.mkdir(parents=True, exist_ok=True)
    records = []
    pending = []

    def accept(
            record: Mapping[str, object], *,
            reused: bool = False) -> None:
        records.append(dict(record))
        if not reused:
            write_json_atomic(
                _task_path(root, task_by_coordinate[(
                    record["candidate_key"],
                    record["seed"],
                )]),
                record,
            )
        if progress is not None:
            progress({
                "completed": len(records),
                "total": len(tasks),
                "candidate_key": record["candidate_key"],
                "seed": record["seed"],
                "reused": reused,
            })

    task_by_coordinate = {
        (task.candidate_key, task.seed): task for task in tasks
    }
    expected_paths = {_task_path(root, task) for task in tasks}
    if resume:
        cell_root = root / "cells"
        observed = (
            set(cell_root.rglob("*.json"))
            if cell_root.exists() else set()
        )
        unexpected = sorted(observed - expected_paths)
        if unexpected:
            raise SSDHBFDesignSweepError(
                "resume output contains cells outside the current grid: "
                f"{unexpected[:5]}")
    for task in tasks:
        path = _task_path(root, task)
        if resume and os.path.lexists(path):
            if path.is_symlink() or not path.is_file():
                raise SSDHBFDesignSweepError(
                    f"resumable cell must be a regular file: {path}")
            accept(_load_resumable_cell(path, task), reused=True)
        else:
            pending.append(task)

    if workers == 1:
        for task in pending:
            accept(_execute_task(task))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=1,
        ) as executor:
            futures = {
                executor.submit(_execute_task, task): task
                for task in pending
            }
            for future in as_completed(futures):
                try:
                    accept(future.result())
                except BaseException:
                    for other in futures:
                        other.cancel()
                    raise

    current_hash = _execution_inputs_sha256(repo_root)
    expected_hash = tasks[0].execution_inputs_sha256
    if current_hash != expected_hash:
        raise SSDHBFDesignSweepError(
            "execution source or hardware config changed during sweep")
    aggregate = aggregate_cell_records(records, designs)
    manifest = {
        **aggregate,
        "scenario": scenario_contract,
        "grid": {
            "session_rate": session_rate,
            "seeds": list(seeds),
            "design_count": len(designs),
            "reference_count": 2,
            "cell_count": len(tasks),
            "resumed_cell_count": len(tasks) - len(pending),
            "executed_cell_count": len(pending),
            "designs": [
                design.to_json_dict() for design in designs],
        },
        "execution_inputs_sha256": expected_hash,
    }
    aggregate_path = root / "aggregate.json"
    write_json_atomic(aggregate_path, manifest)
    _write_summary_csv(root / "summary.csv", manifest)
    return manifest, aggregate_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the one-GPU local-SSD plus one-HBF staged-promotion "
            "design on the pinned long-cold scenario at rate 3.0."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--trace", type=Path, default=default_trace_path())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rate", type=float, default=REQUIRED_SESSION_RATE)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--layouts", nargs="+",
        choices=SUPPORTED_LAYOUTS,
        default=list(SUPPORTED_LAYOUTS),
    )
    parser.add_argument(
        "--policies", nargs="+",
        choices=SUPPORTED_MIGRATION_POLICIES,
        default=list(DEFAULT_MIGRATION_POLICIES),
    )
    parser.add_argument(
        "--hbf-read-modes",
        nargs="+",
        choices=SUPPORTED_HBF_READ_MODES,
        default=list(DEFAULT_HBF_READ_MODES),
        help=(
            "demand exposes one configured fixed latency per HBF-touching "
            "kernel; prefetch hides only that fixed latency"
        ),
    )
    parser.add_argument(
        "--memory",
        action="append",
        default=None,
        help=(
            "kind:GiB:GB/s[:USD/GiB:W/GiB]; repeat for multiple points"
        ),
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--first-ttft-seconds",
        type=float,
        default=DEFAULT_FIRST_TTFT_SECONDS,
    )
    parser.add_argument(
        "--resume-ttft-seconds",
        type=float,
        default=DEFAULT_RESUME_TTFT_SECONDS,
    )
    parser.add_argument(
        "--tpot-milliseconds",
        type=float,
        default=DEFAULT_TPOT_MILLISECONDS,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "use two layouts, two representative policies, one memory "
            "point, and two seeds"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.rate != REQUIRED_SESSION_RATE:
        raise SSDHBFDesignSweepError(
            f"--rate must be exactly {REQUIRED_SESSION_RATE}")
    memories = tuple(parse_active_memory_spec(value) for value in (
        args.memory or ("lpddr:16:409.6",)
    ))
    policies = (
        SMOKE_MIGRATION_POLICIES
        if args.smoke else tuple(args.policies)
    )
    designs = build_design_grid(
        layouts=tuple(args.layouts),
        migration_policies=policies,
        active_memories=memories,
        hbf_read_modes=tuple(args.hbf_read_modes),
    )
    for design in designs:
        validate_design_workspace(design)
    scenario = load_long_cold_context_stress_scenario(args.trace)
    seeds = tuple(
        SMOKE_SEEDS
        if args.smoke and args.seeds is None
        else DEFAULT_SSD_HBF_SEEDS if args.seeds is None
        else args.seeds
    )
    workers = (
        default_worker_count()
        if args.workers is None else args.workers
    )
    tasks = build_tasks(
        repo_root=args.repo_root,
        scenario=scenario,
        designs=designs,
        seeds=seeds,
        session_rate=args.rate,
        first_ttft_seconds=args.first_ttft_seconds,
        resume_ttft_seconds=args.resume_ttft_seconds,
        tpot_milliseconds=args.tpot_milliseconds,
    )
    if args.dry_run:
        print(json.dumps({
            "comparison_contract": SSD_HBF_CONTRACT_KEY,
            "session_rate": args.rate,
            "seeds": list(seeds),
            "design_count": len(designs),
            "cell_count": len(tasks),
            "designs": [
                design.to_json_dict() for design in designs],
        }, indent=2, sort_keys=True))
        return 0

    def progress(event: Mapping[str, object]) -> None:
        print(
            f"[{event['completed']}/{event['total']}] "
            f"seed={event['seed']} "
            f"candidate={event['candidate_key']} "
            f"reused={event['reused']}",
            flush=True,
        )

    manifest, path = run_design_space(
        repo_root=args.repo_root,
        output_root=args.output,
        scenario=scenario,
        designs=designs,
        seeds=seeds,
        workers=workers,
        session_rate=args.rate,
        first_ttft_seconds=args.first_ttft_seconds,
        resume_ttft_seconds=args.resume_ttft_seconds,
        tpot_milliseconds=args.tpot_milliseconds,
        resume=args.resume,
        progress=progress,
    )
    print(json.dumps({
        "aggregate": str(path),
        "summary_csv": str(path.parent / "summary.csv"),
        "reference_eligible": manifest["rates"][0][
            "reference_eligibility"]["eligible"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_CANDIDATE_KEY",
    "BASELINE_OVER_ORACLE_GOODPUT_CI95_UPPER_MAX",
    "DEFAULT_SSD_HBF_SEEDS",
    "ORACLE_CANDIDATE_KEY",
    "ORACLE_EVERY_SEED_JOINT_MIN",
    "ORACLE_MEAN_JOINT_MIN",
    "REQUIRED_SESSION_RATE",
    "SMOKE_SEEDS",
    "SSDHBFDesignSpec",
    "SSDHBFDesignSweepError",
    "SSD_HBF_CELL_SCHEMA_VERSION",
    "SSD_HBF_CONTRACT_KEY",
    "SSD_HBF_SWEEP_SCHEMA_VERSION",
    "SUPPORTED_LAYOUTS",
    "SUPPORTED_MIGRATION_POLICIES",
    "_CellTask",
    "_load_resumable_cell",
    "_seal_record",
    "aggregate_cell_records",
    "build_design_grid",
    "build_tasks",
    "evaluate_reference_eligibility",
    "make_design_system",
    "make_design_spec",
    "make_reference_system",
    "pareto_frontier",
    "parse_active_memory_spec",
    "run_design_cell",
    "run_design_space",
    "run_reference_cell",
    "validate_design_workspace",
    "validate_scenario_contract",
]

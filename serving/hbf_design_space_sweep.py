"""Exploratory multi-HBF performance and TCO sweep.

This runner is intentionally separate from :mod:`hbf_comparison_sweep`.
The comparison sweep freezes one publication contract, whereas this module
varies the number and layout of independent eight-card HBF servers, migration
policy, and per-card active-memory assumptions.

Every ``(rate, seed)`` pair consumes one immutable schedule.  The CPU+SSD
baseline, infinite-HBM Oracle, and every HBF design therefore see identical
session starts and causal tool gaps.  TCO is evaluated only after seed-level
performance aggregation and uses the explicit design-space model in
``core.hbf_design_tco``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import time
from typing import Callable, Mapping, Optional, Sequence

from .core.gpu_hbf_hybrid import GPUHBFHybridSystem, MigrationPolicy
from .core.gpu_pd_latency import load_p4d4_gpu_config
from .core.hbf_comparison_cell import (
    DEFAULT_FIRST_TTFT_SECONDS,
    DEFAULT_RESUME_TTFT_SECONDS,
    DEFAULT_TPOT_MILLISECONDS,
    MAX_NUM_BATCHED_TOKENS,
    MAX_PREFILL_CHUNK_TOKENS,
    PINNED_GPU_CONFIG,
    PINNED_HBF_CONFIG,
    P_MAX_NUM_SEQS,
    D_MAX_NUM_SEQS,
    SHARED_MAX_NUM_SEQS,
    build_slo_thresholds,
    json_safe,
    run_comparison_cell,
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
from .core.hbf_design_tco import (
    ActiveMemorySpec,
    HBFDesignTCOError,
    evaluate_hbf_design_tco,
    lpddr_active_memory,
)
from .core.hbf_full_model_latency import (
    HBFParallelLayout,
    load_hbf_server_config,
)
from .core.hbf_full_model_pool import derive_lpddr_workspace_bytes
from .core.tracelab_comparison_scenarios import (
    BALANCED_DEFAULT_RATES,
    TraceLabComparisonScenario,
    load_balanced_causal_prefix_scenario,
)
from .hbf_comparison_sweep import (
    DEFAULT_SEEDS,
    default_trace_path,
    default_worker_count,
)


DESIGN_SPACE_SCHEMA_VERSION = 2
DESIGN_CELL_SCHEMA_VERSION = 2
BASELINE_SYSTEM_KEY = "cpu_ssd"
BASELINE_CANDIDATE_KEY = "baseline_cpu_ssd"
ORACLE_CANDIDATE_KEY = "oracle"
SUPPORTED_DESIGN_LAYOUTS = ("tp4", "tp8_context")
DEFAULT_MIGRATION_POLICIES = (
    "eager",
    "delay_100ms",
    "delay_200ms",
    "delay_500ms",
    "after_first_tool",
    "load_aware",
)
SUPPORTED_HBF_READ_MODES = ("demand", "prefetch")
DEFAULT_HBF_READ_MODES = SUPPORTED_HBF_READ_MODES
BYTES_PER_GIB = 1024 ** 3
_EXECUTION_INPUTS = (
    Path("serving/hbf_design_space_sweep.py"),
    Path("serving/core/gpu_hbf_hybrid.py"),
    Path("serving/core/multi_hbf_cluster.py"),
    Path("serving/core/hbf_full_model_latency.py"),
    Path("serving/core/hbf_full_model_lifecycle.py"),
    Path("serving/core/hbf_full_model_pool.py"),
    Path("serving/core/hbf_comparison_cell.py"),
    Path("serving/core/hbf_comparison_metrics.py"),
    Path("serving/core/hbf_comparison_workload.py"),
    Path("serving/core/hbf_design_tco.py"),
    PINNED_GPU_CONFIG,
    PINNED_HBF_CONFIG,
)


class HBFDesignSpaceError(ValueError):
    """Raised when an exploratory grid or completed cohort is inconsistent."""


def _finite_positive(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise HBFDesignSpaceError(f"{name} must be positive and finite")
    return float(value)


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    if not normalized:
        raise HBFDesignSpaceError("design key component cannot be empty")
    return normalized


def parse_layout_set(value: str) -> tuple[str, ...]:
    """Parse ``tp4,tp8_context`` into one canonical server-layout tuple."""

    if not isinstance(value, str) or not value.strip():
        raise HBFDesignSpaceError("layout set must be a non-empty string")
    layouts = tuple(part.strip() for part in value.split(","))
    if any(not part for part in layouts):
        raise HBFDesignSpaceError(
            f"layout set contains an empty item: {value!r}")
    unsupported = sorted(set(layouts) - set(SUPPORTED_DESIGN_LAYOUTS))
    if unsupported:
        raise HBFDesignSpaceError(
            f"unsupported design layouts {unsupported}; "
            f"supported={list(SUPPORTED_DESIGN_LAYOUTS)}")
    # HBF hosts are symmetric.  Canonical ordering prevents duplicate points
    # such as tp4+tp8_context and tp8_context+tp4.
    return tuple(sorted(layouts))


def parse_active_memory_spec(value: str) -> ActiveMemorySpec:
    """Parse an explicit active-memory point.

    Accepted forms are ``lpddr:GiB:GB/s`` and
    ``kind:GiB:GB/s:USD/GiB:W/GiB``.  ``sram_like`` deliberately requires
    explicit cost and power assumptions because no LPDDR economics should be
    inherited silently.
    """

    if not isinstance(value, str):
        raise HBFDesignSpaceError("active-memory spec must be a string")
    fields = tuple(part.strip() for part in value.split(":"))
    if len(fields) not in (3, 5):
        raise HBFDesignSpaceError(
            "active-memory spec must be kind:GiB:GB/s or "
            "kind:GiB:GB/s:USD/GiB:W/GiB")
    kind = fields[0]
    try:
        capacity = float(fields[1])
        bandwidth = float(fields[2])
        capex = float(fields[3]) if len(fields) == 5 else None
        power = float(fields[4]) if len(fields) == 5 else None
    except ValueError as exc:
        raise HBFDesignSpaceError(
            f"invalid numeric active-memory field in {value!r}") from exc
    if kind == "lpddr" and len(fields) == 3:
        return lpddr_active_memory(
            capacity_gib_per_card=capacity,
            bandwidth_gbps_per_card=bandwidth,
        )
    if kind == "sram_like" and len(fields) != 5:
        raise HBFDesignSpaceError(
            "sram_like requires explicit USD/GiB and W/GiB assumptions")
    if len(fields) != 5:
        raise HBFDesignSpaceError(
            f"{kind!r} requires explicit USD/GiB and W/GiB assumptions")
    return ActiveMemorySpec(
        kind=kind,
        capacity_gib_per_card=capacity,
        bandwidth_gbps_per_card=bandwidth,
        capex_usd_per_gib=capex,
        power_w_per_gib=power,
        assumption=(
            "Explicit command-line exploratory assumption; not a vendor "
            "price or measured product specification."
        ),
    )


@dataclass(frozen=True)
class HBFDesignSpec:
    """One performance/TCO design point."""

    key: str
    hbf_server_layouts: tuple[str, ...]
    migration_policy: str
    active_memory: ActiveMemorySpec
    hbf_read_mode: str = "demand"

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise HBFDesignSpaceError("design key must be non-empty")
        if _slug(self.key) != self.key:
            raise HBFDesignSpaceError(
                "design key must already be a lowercase filesystem slug")
        if not isinstance(self.hbf_server_layouts, tuple):
            raise HBFDesignSpaceError(
                "hbf_server_layouts must be an immutable tuple")
        canonical = parse_layout_set(",".join(self.hbf_server_layouts))
        if canonical != self.hbf_server_layouts:
            raise HBFDesignSpaceError(
                "hbf_server_layouts must use canonical sorted order")
        if not isinstance(self.active_memory, ActiveMemorySpec):
            raise HBFDesignSpaceError(
                "active_memory must be ActiveMemorySpec")
        if self.hbf_read_mode not in SUPPORTED_HBF_READ_MODES:
            raise HBFDesignSpaceError(
                "hbf_read_mode must be one of "
                f"{SUPPORTED_HBF_READ_MODES!r}")
        try:
            MigrationPolicy.for_key(self.migration_policy)
        except (TypeError, ValueError) as exc:
            raise HBFDesignSpaceError(
                f"invalid migration policy {self.migration_policy!r}"
            ) from exc

    @property
    def hbf_host_count(self) -> int:
        return len(self.hbf_server_layouts)

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def design_key(
        *,
        hbf_server_layouts: Sequence[str],
        migration_policy: str,
        active_memory: ActiveMemorySpec,
        hbf_read_mode: str = "demand",
) -> str:
    layouts = parse_layout_set(",".join(hbf_server_layouts))
    memory = (
        f"{active_memory.kind}-"
        f"{active_memory.capacity_gib_per_card:g}gib-"
        f"{active_memory.bandwidth_gbps_per_card:g}gbps-"
        f"{active_memory.capex_usd_per_gib:g}usdpgib-"
        f"{active_memory.power_w_per_gib:g}wpgib"
    )
    return _slug(
        f"hbf{len(layouts)}-"
        f"{'+'.join(layouts)}-"
        f"{migration_policy}-{hbf_read_mode}-{memory}"
    )


def make_design_spec(
        *,
        hbf_server_layouts: Sequence[str],
        migration_policy: str,
        active_memory: ActiveMemorySpec,
        hbf_read_mode: str = "demand",
) -> HBFDesignSpec:
    layouts = parse_layout_set(",".join(hbf_server_layouts))
    return HBFDesignSpec(
        key=design_key(
            hbf_server_layouts=layouts,
            migration_policy=migration_policy,
            active_memory=active_memory,
            hbf_read_mode=hbf_read_mode,
        ),
        hbf_server_layouts=layouts,
        migration_policy=migration_policy,
        active_memory=active_memory,
        hbf_read_mode=hbf_read_mode,
    )


def build_design_grid(
        *,
        hbf_host_counts: Sequence[int],
        layouts: Sequence[str],
        migration_policies: Sequence[str],
        active_memories: Sequence[ActiveMemorySpec],
        hbf_read_modes: Sequence[str] = ("demand",),
        include_mixed_layouts: bool = False,
) -> tuple[HBFDesignSpec, ...]:
    """Build deterministic homogeneous or symmetry-reduced mixed layouts."""

    if not hbf_host_counts:
        raise HBFDesignSpaceError("hbf_host_counts cannot be empty")
    counts = []
    for count in hbf_host_counts:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise HBFDesignSpaceError(
                "HBF host counts must be positive integers")
        counts.append(count)
    if len(counts) != len(set(counts)):
        raise HBFDesignSpaceError("HBF host counts contain duplicates")
    layout_axis = parse_layout_set(",".join(layouts))
    policies = tuple(migration_policies)
    if not policies:
        raise HBFDesignSpaceError("migration_policies cannot be empty")
    for policy in policies:
        try:
            MigrationPolicy.for_key(policy)
        except (TypeError, ValueError) as exc:
            raise HBFDesignSpaceError(
                f"invalid migration policy {policy!r}") from exc
    if len(policies) != len(set(policies)):
        raise HBFDesignSpaceError(
            "migration_policies contain duplicates")
    memories = tuple(active_memories)
    if not memories:
        raise HBFDesignSpaceError("active_memories cannot be empty")
    if any(not isinstance(item, ActiveMemorySpec) for item in memories):
        raise HBFDesignSpaceError(
            "active_memories must contain ActiveMemorySpec values")
    read_modes = tuple(hbf_read_modes)
    if (
        not read_modes
        or len(read_modes) != len(set(read_modes))
        or any(
            mode not in SUPPORTED_HBF_READ_MODES
            for mode in read_modes
        )
    ):
        raise HBFDesignSpaceError(
            "hbf_read_modes must be unique members of "
            f"{SUPPORTED_HBF_READ_MODES!r}")

    layout_sets = []
    for count in sorted(counts):
        if include_mixed_layouts:
            layout_sets.extend(
                itertools.combinations_with_replacement(
                    layout_axis, count))
        else:
            layout_sets.extend((layout,) * count for layout in layout_axis)
    specs = tuple(
        make_design_spec(
            hbf_server_layouts=layout_set,
            migration_policy=policy,
            active_memory=memory,
            hbf_read_mode=read_mode,
        )
        for layout_set in layout_sets
        for policy in policies
        for memory in memories
        for read_mode in read_modes
    )
    keys = [spec.key for spec in specs]
    if len(keys) != len(set(keys)):
        raise HBFDesignSpaceError(
            "design grid produces duplicate keys; active-memory points "
            "must differ in kind, capacity, bandwidth, cost, or power")
    return specs


def validate_design_workspace(
        spec: HBFDesignSpec,
        *,
        max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS,
        max_num_seqs: int = SHARED_MAX_NUM_SEQS,
) -> dict[str, object]:
    """Fail early when per-card active memory cannot hold fixed workspace."""

    capacity_bytes = int(round(
        spec.active_memory.capacity_gib_per_card * BYTES_PER_GIB))
    requirements = {
        layout_key: derive_lpddr_workspace_bytes(
            HBFParallelLayout.for_key(layout_key),
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        )
        for layout_key in sorted(set(spec.hbf_server_layouts))
    }
    maximum = max(requirements.values())
    if capacity_bytes <= maximum:
        raise HBFDesignSpaceError(
            f"design {spec.key!r} active memory cannot hold workspace: "
            f"capacity={capacity_bytes}, required>{maximum}")
    return {
        "capacity_bytes_per_card": capacity_bytes,
        "workspace_bytes_per_card_by_layout": requirements,
        "minimum_free_bytes_per_card": capacity_bytes - maximum,
    }


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
        raise HBFDesignSpaceError(
            "measurement roster must be non-empty and unique")
    indexed = {_identity(request): request for request in completed}
    if len(indexed) != len(completed):
        raise HBFDesignSpaceError("completed requests contain duplicates")
    missing = tuple(identity for identity in roster if identity not in indexed)
    if missing:
        raise HBFDesignSpaceError(
            f"measurement roster has missing completions: {missing[:5]}")
    return tuple(indexed[identity] for identity in roster)


def make_design_system(
        *,
        repo_root: Path,
        spec: HBFDesignSpec,
) -> GPUHBFHybridSystem:
    """Construct the requested independent-HBF analytical system."""

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
    layout_values = tuple(
        HBFParallelLayout.for_key(key)
        for key in spec.hbf_server_layouts
    )
    return GPUHBFHybridSystem(
        repo_root=root,
        gpu_hardware=gpu_hardware,
        hbf_hardware=hbf_hardware,
        hbf_layout=layout_values[0],
        hbf_server_layouts=layout_values,
        migration_policy=spec.migration_policy,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=SHARED_MAX_NUM_SEQS,
        p_max_num_seqs=P_MAX_NUM_SEQS,
        d_max_num_seqs=D_MAX_NUM_SEQS,
        max_prefill_chunk_tokens=MAX_PREFILL_CHUNK_TOKENS,
        validate_every_event=False,
    )


def run_design_cell(
        *,
        repo_root: Path,
        spec: HBFDesignSpec,
        scheduled_sessions: tuple[ScheduledSession, ...],
        session_rate: float,
        seed: int,
        measurement_identities: Sequence[str],
        first_ttft_seconds: float = DEFAULT_FIRST_TTFT_SECONDS,
        resume_ttft_seconds: float = DEFAULT_RESUME_TTFT_SECONDS,
        tpot_milliseconds: float = DEFAULT_TPOT_MILLISECONDS,
) -> dict[str, object]:
    """Run and validate one custom HBF cell."""

    rate = _finite_positive("session_rate", session_rate)
    thresholds = build_slo_thresholds(
        first_ttft_seconds=first_ttft_seconds,
        resume_ttft_seconds=resume_ttft_seconds,
        tpot_milliseconds=tpot_milliseconds,
    )
    system = make_design_system(repo_root=repo_root, spec=spec)
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
        session_rate=rate,
        thresholds=thresholds,
    )
    execution_mix = Counter()
    hbf_server_mix = Counter()
    for call in system.node.calls.values():
        if call.execution is not None:
            execution_mix[call.execution.value] += 1
        if call.hbf_server_index is not None:
            hbf_server_mix[str(call.hbf_server_index)] += 1
    result = {
        "schema_version": DESIGN_CELL_SCHEMA_VERSION,
        "candidate_kind": "design",
        "candidate_key": spec.key,
        "system_key": spec.key,
        "seed": seed,
        "session_rate": rate,
        "design": spec.to_json_dict(),
        "workspace": validate_design_workspace(spec),
        "measurement_roster": {
            "identity_count": len(measurement_identities),
            "ordered_identities_sha256": stable_json_sha256(
                list(measurement_identities)),
        },
        "normalized_system_call_projection_sha256": projection_sha256,
        "full_drain": full_drain,
        "summary": summary,
        "execution_mix": dict(sorted(execution_mix.items())),
        "hbf_server_assignment_mix": dict(
            sorted(hbf_server_mix.items())),
        "execution_observation": {
            "simulated_horizon_ns": system.current_ns,
            "elapsed_wall_time_ns": wall_ns,
        },
        "metrics": {
            "system": asdict(system.metrics),
            "node": asdict(system.node.metrics),
            "lifecycle": asdict(system.node.hbf_lifecycle.metrics),
            "hbf_pool": asdict(system.node.hbf_pool.metrics),
        },
    }
    safe = json_safe(result)
    json.dumps(safe, allow_nan=False, sort_keys=True)
    return safe


def _compact_reference_cell(
        result: Mapping[str, object],
        *,
        seed: int,
        candidate_kind: str,
        candidate_key: str,
) -> dict[str, object]:
    return {
        "schema_version": DESIGN_CELL_SCHEMA_VERSION,
        "candidate_kind": candidate_kind,
        "candidate_key": candidate_key,
        "system_key": result["system_key"],
        "seed": seed,
        "session_rate": result["session_rate"],
        "measurement_roster": result["measurement_roster"],
        "full_drain": result["full_drain"],
        "summary": result["summary"],
        "bottleneck_report": result["bottleneck_report"],
        "execution_observation": result["execution_observation"],
    }


@dataclass(frozen=True)
class _CellTask:
    repo_root: Path
    candidate_kind: str
    candidate_key: str
    seed: int
    session_rate: float
    scheduled_sessions: tuple[ScheduledSession, ...]
    measurement_identities: tuple[str, ...]
    design: Optional[HBFDesignSpec]
    first_ttft_seconds: float
    resume_ttft_seconds: float
    tpot_milliseconds: float
    execution_inputs_sha256: str


def _execution_inputs_sha256(repo_root: Path) -> str:
    digest = hashlib.sha256()
    root = Path(repo_root)
    for relative in _EXECUTION_INPUTS:
        path = root / relative
        if not path.is_file():
            raise HBFDesignSpaceError(
                f"missing execution input {path}")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _task_contract(task: _CellTask) -> dict[str, object]:
    raw = {
        "schema_version": DESIGN_CELL_SCHEMA_VERSION,
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
            None if task.design is None else task.design.to_json_dict()),
        "thresholds": {
            "first_ttft_seconds": task.first_ttft_seconds,
            "resume_ttft_seconds": task.resume_ttft_seconds,
            "tpot_milliseconds": task.tpot_milliseconds,
        },
        "execution_inputs_sha256": task.execution_inputs_sha256,
    }
    normalized = json_safe(raw)
    if not isinstance(normalized, dict):
        raise AssertionError("cell contract did not normalize to an object")
    return normalized


def _seal_record(
        task: _CellTask,
        record: Mapping[str, object],
) -> dict[str, object]:
    contract = _task_contract(task)
    value = {
        **dict(record),
        "cell_contract": contract,
        "cell_contract_sha256": stable_json_sha256(contract),
    }
    value["result_payload_sha256"] = stable_json_sha256(value)
    return value


def _execute_task(task: _CellTask) -> dict[str, object]:
    if task.candidate_kind == "design":
        if task.design is None:
            raise AssertionError("design task has no design")
        result = run_design_cell(
            repo_root=task.repo_root,
            spec=task.design,
            scheduled_sessions=task.scheduled_sessions,
            session_rate=task.session_rate,
            seed=task.seed,
            measurement_identities=task.measurement_identities,
            first_ttft_seconds=task.first_ttft_seconds,
            resume_ttft_seconds=task.resume_ttft_seconds,
            tpot_milliseconds=task.tpot_milliseconds,
        )
    else:
        system_key = (
            BASELINE_SYSTEM_KEY
            if task.candidate_kind == "baseline"
            else "oracle"
        )
        full_result = run_comparison_cell(
            repo_root=task.repo_root,
            system_key=system_key,
            scheduled_sessions=task.scheduled_sessions,
            session_rate=task.session_rate,
            measurement_identities=task.measurement_identities,
            first_ttft_seconds=task.first_ttft_seconds,
            resume_ttft_seconds=task.resume_ttft_seconds,
            tpot_milliseconds=task.tpot_milliseconds,
        )
        result = _compact_reference_cell(
            full_result,
            seed=task.seed,
            candidate_kind=task.candidate_kind,
            candidate_key=task.candidate_key,
        )
    return _seal_record(task, result)


def _summary_metric(
        record: Mapping[str, object],
        path: Sequence[str],
) -> Optional[float]:
    value: object = record["summary"]
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise HBFDesignSpaceError(
                f"cell lacks summary metric path {tuple(path)!r}")
        value = value[key]
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise HBFDesignSpaceError(
            f"cell metric {tuple(path)!r} is not finite")
    return float(value)


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
            None
            if not values
            else asdict(aggregate_seed_values(values))
        )
    return metrics


def pareto_frontier(
        points: Mapping[str, tuple[float, float]],
) -> tuple[str, ...]:
    """Return nondominated keys for ``(goodput, lifetime TCO)`` points."""

    normalized = {}
    for key, pair in points.items():
        if not isinstance(key, str) or not key:
            raise HBFDesignSpaceError(
                "Pareto point keys must be non-empty strings")
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise HBFDesignSpaceError(
                "Pareto points must be (goodput, TCO) tuples")
        goodput = _finite_positive("Pareto goodput", pair[0])
        tco = _finite_positive("Pareto TCO", pair[1])
        normalized[key] = (goodput, tco)
    frontier = []
    for key, (goodput, tco) in normalized.items():
        dominated = any(
            other_key != key
            and other_goodput >= goodput
            and other_tco <= tco
            and (other_goodput > goodput or other_tco < tco)
            for other_key, (other_goodput, other_tco)
            in normalized.items()
        )
        if not dominated:
            frontier.append(key)
    return tuple(sorted(
        frontier,
        key=lambda key: (
            normalized[key][1],
            -normalized[key][0],
            key,
        ),
    ))


def aggregate_cell_records(
        records: Sequence[Mapping[str, object]],
        designs: Sequence[HBFDesignSpec],
) -> dict[str, object]:
    """Aggregate paired seeds, add TCO, and identify each rate's frontier."""

    if not records:
        raise HBFDesignSpaceError("cannot aggregate an empty cell set")
    design_by_key = {design.key: design for design in designs}
    if len(design_by_key) != len(designs):
        raise HBFDesignSpaceError("design keys are not unique")
    expected_candidates = {
        BASELINE_CANDIDATE_KEY,
        ORACLE_CANDIDATE_KEY,
        *design_by_key,
    }
    grouped: dict[float, dict[str, dict[int, Mapping[str, object]]]] = {}
    for record in records:
        rate = _finite_positive("cell session_rate", record["session_rate"])
        candidate = record.get("candidate_key")
        seed = record.get("seed")
        if candidate not in expected_candidates:
            raise HBFDesignSpaceError(
                f"unexpected cell candidate {candidate!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise HBFDesignSpaceError("cell seed must be an integer")
        slots = grouped.setdefault(rate, {}).setdefault(candidate, {})
        if seed in slots:
            raise HBFDesignSpaceError(
                f"duplicate cell for rate={rate}, seed={seed}, "
                f"candidate={candidate}")
        slots[seed] = record

    rate_rows = []
    for rate in sorted(grouped):
        candidates = grouped[rate]
        if set(candidates) != expected_candidates:
            raise HBFDesignSpaceError(
                f"incomplete candidate set at rate={rate}: "
                f"missing={sorted(expected_candidates - set(candidates))}, "
                f"extra={sorted(set(candidates) - expected_candidates)}")
        reference_seeds = set(candidates[BASELINE_CANDIDATE_KEY])
        if not reference_seeds:
            raise HBFDesignSpaceError("seed set cannot be empty")
        for key, by_seed in candidates.items():
            if set(by_seed) != reference_seeds:
                raise HBFDesignSpaceError(
                    f"unpaired seeds for rate={rate}, candidate={key}")

        aggregates = {
            key: _aggregate_candidate(by_seed)
            for key, by_seed in candidates.items()
        }
        baseline_values = {
            seed: _summary_metric(record, _AGGREGATE_METRICS[
                "slo_good_output_tokens_per_second"])
            for seed, record
            in candidates[BASELINE_CANDIDATE_KEY].items()
        }
        oracle_values = {
            seed: _summary_metric(record, _AGGREGATE_METRICS[
                "slo_good_output_tokens_per_second"])
            for seed, record
            in candidates[ORACLE_CANDIDATE_KEY].items()
        }
        if any(value is None for value in baseline_values.values()):
            raise HBFDesignSpaceError("baseline goodput cannot be null")
        if any(value is None for value in oracle_values.values()):
            raise HBFDesignSpaceError("Oracle goodput cannot be null")
        baseline_numeric = {
            seed: float(value)
            for seed, value in baseline_values.items()
        }
        oracle_numeric = {
            seed: float(value)
            for seed, value in oracle_values.items()
        }
        baseline_mean = aggregates[BASELINE_CANDIDATE_KEY][
            "slo_good_output_tokens_per_second"]["mean"]
        oracle_mean = aggregates[ORACLE_CANDIDATE_KEY][
            "slo_good_output_tokens_per_second"]["mean"]

        design_rows = []
        pareto_points = {}
        for key in sorted(design_by_key):
            spec = design_by_key[key]
            values = {
                seed: float(_summary_metric(
                    record,
                    _AGGREGATE_METRICS[
                        "slo_good_output_tokens_per_second"],
                ))
                for seed, record in candidates[key].items()
            }
            paired_baseline = asdict(aggregate_paired_seed_values(
                baseline_numeric, values))
            paired_oracle = asdict(aggregate_paired_seed_values(
                oracle_numeric, values))
            if baseline_mean > 0.0:
                try:
                    tco = evaluate_hbf_design_tco(
                        hbf_host_count=spec.hbf_host_count,
                        active_memory=spec.active_memory,
                        baseline_slo_good_output_tokens_per_second=(
                            baseline_mean),
                        proposed_slo_good_output_tokens_per_second=(
                            aggregates[key][
                                "slo_good_output_tokens_per_second"]["mean"]),
                        oracle_slo_good_output_tokens_per_second=oracle_mean,
                    ).to_json_dict()
                    tco_unavailable_reason = None
                except HBFDesignTCOError as exc:
                    tco = None
                    tco_unavailable_reason = str(exc)
            else:
                tco = None
                tco_unavailable_reason = (
                    "baseline SLO-good output-token goodput is zero")
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
                    tco["proposed_cost"]["lifetime_tco_usd"],
                )
        frontier = pareto_frontier(pareto_points) if pareto_points else ()
        for row in design_rows:
            row["performance_tco_pareto"] = (
                row["design"]["key"] in frontier)
        performance_ranking = sorted(
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
        token_value_ranking = sorted(
            (
                (
                    row["design"]["key"],
                    row["tco"]["proposed_token_cost"][
                        "dollars_per_million_slo_good_output_tokens"],
                    row["metrics"][
                        "slo_good_output_tokens_per_second"]["mean"],
                )
                for row in design_rows
                if (
                    row["tco"] is not None
                    and row["tco"]["proposed_token_cost"][
                        "dollars_per_million_slo_good_output_tokens"]
                        is not None
                )
            ),
            key=lambda item: (item[1], -item[2], item[0]),
        )
        rate_rows.append({
            "session_rate": rate,
            "seed_ids": sorted(reference_seeds),
            "references": {
                BASELINE_CANDIDATE_KEY: aggregates[
                    BASELINE_CANDIDATE_KEY],
                ORACLE_CANDIDATE_KEY: aggregates[
                    ORACLE_CANDIDATE_KEY],
            },
            "designs": design_rows,
            "performance_tco_pareto_design_keys": list(frontier),
            "best_performance_design_key": (
                performance_ranking[0][0]
                if performance_ranking else None),
            "best_token_value_design_key": (
                token_value_ranking[0][0]
                if token_value_ranking else None),
            "performance_ranking": [
                {"design_key": key, "goodput_mean": goodput}
                for key, goodput in performance_ranking
            ],
            "token_value_ranking": [
                {
                    "design_key": key,
                    "dollars_per_million_slo_good_output_tokens": cost,
                    "goodput_mean": goodput,
                }
                for key, cost, goodput in token_value_ranking
            ],
        })
    return {
        "schema_version": DESIGN_SPACE_SCHEMA_VERSION,
        "aggregation": (
            "independent seed values with Student-t 95% CI; pairwise "
            "differences and ratios are formed per seed before aggregation"
        ),
        "performance_metric": (
            "offered-load-normalized joint-SLO-good output tokens/s"),
        "tco_semantics": (
            "central design-space assumptions at the seed-mean matched-rate "
            "goodput; Oracle excluded from TCO"),
        "rates": rate_rows,
    }


def _rate_text(rate: float) -> str:
    return format(rate, ".12g").replace("-", "m").replace(".", "p")


def _cell_path(
        output_root: Path, record: Mapping[str, object]) -> Path:
    return (
        output_root
        / "cells"
        / f"rate_{_rate_text(float(record['session_rate']))}"
        / f"seed_{record['seed']}"
        / f"{record['candidate_key']}.json"
    )


def _task_path(output_root: Path, task: _CellTask) -> Path:
    return (
        output_root
        / "cells"
        / f"rate_{_rate_text(task.session_rate)}"
        / f"seed_{task.seed}"
        / f"{task.candidate_key}.json"
    )


def _strict_json_object(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicates(
            pairs: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(
                source,
                parse_constant=reject_constant,
                object_pairs_hook=reject_duplicates,
            )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise HBFDesignSpaceError(
            f"cannot resume invalid cell {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HBFDesignSpaceError(
            f"resumable cell is not a JSON object: {path}")
    return value


def _load_resumable_cell(
        path: Path,
        task: _CellTask,
) -> dict[str, object]:
    record = _strict_json_object(path)
    expected_contract = _task_contract(task)
    expected_contract_sha256 = stable_json_sha256(expected_contract)
    if (
        record.get("cell_contract") != expected_contract
        or record.get("cell_contract_sha256")
        != expected_contract_sha256
    ):
        raise HBFDesignSpaceError(
            f"resumable cell contract mismatch: {path}")
    observed_payload_sha256 = record.get("result_payload_sha256")
    unsealed = dict(record)
    unsealed.pop("result_payload_sha256", None)
    if (
        not isinstance(observed_payload_sha256, str)
        or observed_payload_sha256 != stable_json_sha256(unsealed)
    ):
        raise HBFDesignSpaceError(
            f"resumable cell payload hash mismatch: {path}")
    expected_coordinates = {
        "candidate_kind": task.candidate_kind,
        "candidate_key": task.candidate_key,
        "seed": task.seed,
        "session_rate": task.session_rate,
    }
    for key, expected in expected_coordinates.items():
        if record.get(key) != expected:
            raise HBFDesignSpaceError(
                f"resumable cell coordinate mismatch for {key}: {path}")
    # Exercise the metric parser now rather than discovering a malformed
    # artifact only after all remaining 56-second cells finish.
    _summary_metric(
        record,
        _AGGREGATE_METRICS[
            "slo_good_output_tokens_per_second"],
    )
    return record


def _write_summary_csv(
        path: Path,
        aggregate: Mapping[str, object],
) -> None:
    fields = (
        "session_rate",
        "design_key",
        "hbf_host_count",
        "hbf_server_layouts",
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
        "lifetime_tco_usd",
        "dollars_per_million_slo_good_tokens",
        "baseline_it_power_w",
        "proposed_it_power_w",
        "incremental_it_power_w",
        "proposed_it_power_ratio_to_baseline",
        "baseline_lifetime_facility_energy_kwh",
        "proposed_lifetime_facility_energy_kwh",
        "incremental_lifetime_facility_energy_kwh",
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
            rows.append({
                "session_rate": rate_row["session_rate"],
                "design_key": design["key"],
                "hbf_host_count": len(design["hbf_server_layouts"]),
                "hbf_server_layouts": "+".join(
                    design["hbf_server_layouts"]),
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
                    row["paired_vs_baseline_goodput"][
                        "candidate_over_reference"]["mean"]
                    if row["paired_vs_baseline_goodput"][
                        "candidate_over_reference"] is not None else None
                ),
                "goodput_ratio_to_oracle_mean": (
                    row["paired_vs_oracle_goodput"][
                        "candidate_over_reference"]["mean"]
                    if row["paired_vs_oracle_goodput"][
                        "candidate_over_reference"] is not None else None
                ),
                "tco_lifetime_years": (
                    None if tco is None else
                    tco["proposed_cost"]["evaluation"]["lifetime_years"]),
                "lifetime_tco_usd": (
                    None if tco is None else
                    tco["proposed_cost"]["lifetime_tco_usd"]),
                "dollars_per_million_slo_good_tokens": (
                    None if tco is None else
                    tco["proposed_token_cost"][
                        "dollars_per_million_slo_good_output_tokens"]),
                "baseline_it_power_w": (
                    None if tco is None else
                    tco["baseline_cost"]["it_power_w"]),
                "proposed_it_power_w": (
                    None if tco is None else
                    tco["proposed_cost"]["it_power_w"]),
                "incremental_it_power_w": (
                    None if tco is None else
                    tco["proposed_cost"]["it_power_w"]
                    - tco["baseline_cost"]["it_power_w"]),
                "proposed_it_power_ratio_to_baseline": (
                    None if tco is None else
                    tco["proposed_cost"]["it_power_w"]
                    / tco["baseline_cost"]["it_power_w"]),
                "baseline_lifetime_facility_energy_kwh": (
                    None if tco is None else
                    tco["baseline_cost"][
                        "lifetime_facility_energy_kwh"]),
                "proposed_lifetime_facility_energy_kwh": (
                    None if tco is None else
                    tco["proposed_cost"][
                        "lifetime_facility_energy_kwh"]),
                "incremental_lifetime_facility_energy_kwh": (
                    None if tco is None else
                    tco["proposed_cost"][
                        "lifetime_facility_energy_kwh"]
                    - tco["baseline_cost"][
                        "lifetime_facility_energy_kwh"]),
                "proposed_facility_energy_ratio_to_baseline": (
                    None if tco is None else
                    tco["proposed_cost"][
                        "lifetime_facility_energy_kwh"]
                    / tco["baseline_cost"][
                        "lifetime_facility_energy_kwh"]),
                "meets_token_value_break_even": (
                    None if tco is None else
                    tco[
                        "proposed_meets_or_exceeds_token_value_break_even"]),
                "performance_tco_pareto": (
                    row["performance_tco_pareto"]),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_tasks(
        *,
        repo_root: Path,
        scenario: TraceLabComparisonScenario,
        designs: Sequence[HBFDesignSpec],
        rates: Sequence[float],
        seeds: Sequence[int],
        first_ttft_seconds: float = DEFAULT_FIRST_TTFT_SECONDS,
        resume_ttft_seconds: float = DEFAULT_RESUME_TTFT_SECONDS,
        tpot_milliseconds: float = DEFAULT_TPOT_MILLISECONDS,
) -> tuple[_CellTask, ...]:
    """Freeze all paired schedules without running a simulator."""

    if not designs:
        raise HBFDesignSpaceError("designs cannot be empty")
    if len({design.key for design in designs}) != len(designs):
        raise HBFDesignSpaceError("design keys contain duplicates")
    execution_inputs_sha256 = _execution_inputs_sha256(repo_root)
    measurement = tuple(
        scenario.manifest.measurement_request_identities)
    tasks = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise HBFDesignSpaceError("seeds must be integers")
        offered = scenario.build_offered_plan(seed=seed)
        for raw_rate in rates:
            rate = scenario.manifest.arrival_contract.validate_rate(
                raw_rate)
            schedule = offered.at_rate(rate)
            common = {
                "repo_root": Path(repo_root),
                "seed": seed,
                "session_rate": rate,
                "scheduled_sessions": schedule,
                "measurement_identities": measurement,
                "first_ttft_seconds": first_ttft_seconds,
                "resume_ttft_seconds": resume_ttft_seconds,
                "tpot_milliseconds": tpot_milliseconds,
                "execution_inputs_sha256": execution_inputs_sha256,
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
                for design in designs
            )
    return tuple(tasks)


def run_design_space(
        *,
        repo_root: Path,
        output_root: Path,
        scenario: TraceLabComparisonScenario,
        designs: Sequence[HBFDesignSpec],
        rates: Sequence[float],
        seeds: Sequence[int],
        workers: int = 1,
        first_ttft_seconds: float = DEFAULT_FIRST_TTFT_SECONDS,
        resume_ttft_seconds: float = DEFAULT_RESUME_TTFT_SECONDS,
        tpot_milliseconds: float = DEFAULT_TPOT_MILLISECONDS,
        resume: bool = False,
        progress: Optional[Callable[[Mapping[str, object]], None]] = None,
) -> tuple[dict[str, object], Path]:
    """Run references and designs, then publish JSON and a flat CSV."""

    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not resume:
        raise HBFDesignSpaceError(
            f"output directory is not empty: {root}")
    if not isinstance(resume, bool):
        raise HBFDesignSpaceError("resume must be a boolean")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise HBFDesignSpaceError("workers must be a positive integer")
    tasks = build_tasks(
        repo_root=Path(repo_root).resolve(),
        scenario=scenario,
        designs=designs,
        rates=rates,
        seeds=seeds,
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
            write_json_atomic(_cell_path(root, record), record)
        if progress is not None:
            progress({
                "completed": len(records),
                "total": len(tasks),
                "session_rate": record["session_rate"],
                "seed": record["seed"],
                "candidate_key": record["candidate_key"],
                "reused": reused,
            })

    expected_paths = {_task_path(root, task) for task in tasks}
    if resume:
        cell_root = root / "cells"
        observed_paths = (
            set(cell_root.rglob("*.json"))
            if cell_root.exists() else set()
        )
        unexpected = sorted(observed_paths - expected_paths)
        if unexpected:
            raise HBFDesignSpaceError(
                "resume output contains cells outside the current grid: "
                f"{unexpected[:5]}")
    for task in tasks:
        path = _task_path(root, task)
        if resume and os.path.lexists(path):
            if path.is_symlink() or not path.is_file():
                raise HBFDesignSpaceError(
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

    current_execution_inputs_sha256 = _execution_inputs_sha256(repo_root)
    expected_execution_inputs_sha256 = tasks[
        0].execution_inputs_sha256
    if current_execution_inputs_sha256 != expected_execution_inputs_sha256:
        raise HBFDesignSpaceError(
            "execution source or hardware config changed during the sweep")
    aggregate = aggregate_cell_records(records, designs)
    manifest = {
        **aggregate,
        "scenario": {
            "scenario_id": scenario.manifest.scenario_id,
            "manifest_sha256": stable_json_sha256(
                scenario.manifest.to_dict()),
        },
        "grid": {
            "rates": list(rates),
            "seeds": list(seeds),
            "design_count": len(designs),
            "reference_count": 2,
            "cell_count": len(tasks),
            "resumed_cell_count": len(tasks) - len(pending),
            "executed_cell_count": len(pending),
            "designs": [design.to_json_dict() for design in designs],
        },
        "execution_inputs_sha256": expected_execution_inputs_sha256,
    }
    path = root / "aggregate.json"
    write_json_atomic(path, manifest)
    _write_summary_csv(root / "summary.csv", manifest)
    return manifest, path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep multi-HBF layout, migration-policy, and active-memory "
            "designs against the CPU+SSD baseline and infinite-HBM Oracle."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--trace", type=Path, default=default_trace_path())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rates", type=float, nargs="+",
        default=list(BALANCED_DEFAULT_RATES),
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--hbf-counts", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument(
        "--layouts", nargs="+",
        choices=SUPPORTED_DESIGN_LAYOUTS,
        default=list(SUPPORTED_DESIGN_LAYOUTS),
    )
    parser.add_argument(
        "--layout-set",
        action="append",
        default=None,
        help=(
            "explicit comma-separated per-server layout tuple; repeat for "
            "multiple tuples and use instead of --hbf-counts/--layouts"
        ),
    )
    parser.add_argument(
        "--include-mixed-layouts",
        action="store_true",
        help=(
            "include symmetry-reduced heterogeneous tuples in addition to "
            "homogeneous server tuples"
        ),
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(DEFAULT_MIGRATION_POLICIES),
    )
    parser.add_argument(
        "--hbf-read-modes",
        nargs="+",
        choices=SUPPORTED_HBF_READ_MODES,
        default=list(DEFAULT_HBF_READ_MODES),
        help=(
            "demand exposes the configured fixed HBF read latency once per "
            "touching kernel; prefetch hides only that fixed latency"
        ),
    )
    parser.add_argument(
        "--memory",
        action="append",
        default=None,
        help=(
            "kind:GiB:GB/s[:USD/GiB:W/GiB]; repeat for multiple points. "
            "sram_like requires all five fields"
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
        "--dry-run",
        action="store_true",
        help="validate and print the grid without running simulations",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse only per-cell JSON whose input contract and payload "
            "hash both validate"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    memories = tuple(parse_active_memory_spec(value) for value in (
        args.memory or ("lpddr:16:204.8",)
    ))
    if args.layout_set:
        layout_sets = tuple(
            parse_layout_set(value) for value in args.layout_set)
        designs = tuple(
            make_design_spec(
                hbf_server_layouts=layout_set,
                migration_policy=policy,
                active_memory=memory,
                hbf_read_mode=read_mode,
            )
            for layout_set in layout_sets
            for policy in args.policies
            for memory in memories
            for read_mode in args.hbf_read_modes
        )
    else:
        designs = build_design_grid(
            hbf_host_counts=args.hbf_counts,
            layouts=args.layouts,
            migration_policies=args.policies,
            active_memories=memories,
            hbf_read_modes=args.hbf_read_modes,
            include_mixed_layouts=args.include_mixed_layouts,
        )
    for design in designs:
        validate_design_workspace(design)
    scenario = load_balanced_causal_prefix_scenario(args.trace)
    workers = (
        default_worker_count()
        if args.workers is None
        else args.workers
    )
    tasks = build_tasks(
        repo_root=args.repo_root,
        scenario=scenario,
        designs=designs,
        rates=args.rates,
        seeds=args.seeds,
        first_ttft_seconds=args.first_ttft_seconds,
        resume_ttft_seconds=args.resume_ttft_seconds,
        tpot_milliseconds=args.tpot_milliseconds,
    )
    if args.dry_run:
        print(json.dumps({
            "schema_version": DESIGN_SPACE_SCHEMA_VERSION,
            "scenario_id": scenario.manifest.scenario_id,
            "rates": args.rates,
            "seeds": args.seeds,
            "design_count": len(designs),
            "cell_count": len(tasks),
            "workers": workers,
            "designs": [
                {
                    **design.to_json_dict(),
                    "workspace": validate_design_workspace(design),
                }
                for design in designs
            ],
        }, indent=2, sort_keys=True, allow_nan=False))
        return 0

    def report(event: Mapping[str, object]) -> None:
        print(
            f"completed {event['completed']}/{event['total']} "
            f"rate={event['session_rate']} seed={event['seed']} "
            f"candidate={event['candidate_key']} "
            f"reused={event['reused']}",
            flush=True,
        )

    manifest, path = run_design_space(
        repo_root=args.repo_root,
        output_root=args.output,
        scenario=scenario,
        designs=designs,
        rates=args.rates,
        seeds=args.seeds,
        workers=workers,
        first_ttft_seconds=args.first_ttft_seconds,
        resume_ttft_seconds=args.resume_ttft_seconds,
        tpot_milliseconds=args.tpot_milliseconds,
        resume=args.resume,
        progress=report,
    )
    print(json.dumps({
        "aggregate": str(path),
        "summary_csv": str(path.parent / "summary.csv"),
        "cell_count": manifest["grid"]["cell_count"],
        "design_count": manifest["grid"]["design_count"],
        "resumed_cell_count": manifest["grid"]["resumed_cell_count"],
        "executed_cell_count": manifest["grid"]["executed_cell_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_CANDIDATE_KEY",
    "BASELINE_SYSTEM_KEY",
    "DEFAULT_MIGRATION_POLICIES",
    "DESIGN_SPACE_SCHEMA_VERSION",
    "HBFDesignSpaceError",
    "HBFDesignSpec",
    "ORACLE_CANDIDATE_KEY",
    "SUPPORTED_DESIGN_LAYOUTS",
    "aggregate_cell_records",
    "build_design_grid",
    "build_tasks",
    "design_key",
    "make_design_spec",
    "make_design_system",
    "pareto_frontier",
    "parse_active_memory_spec",
    "parse_layout_set",
    "run_design_cell",
    "run_design_space",
    "validate_design_workspace",
]

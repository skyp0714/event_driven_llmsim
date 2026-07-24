"""One-cell runner for the frozen GPU-tiering/HBF comparison.

A cell is one system, one offered session rate, and one immutable
``tuple[ScheduledSession, ...]``.  Workload construction and rate sweeps live
outside this module so workers can execute cells independently without
redrawing arrivals or silently changing the metric cohort.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time
from collections import Counter
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence

from .gpu_hbf_hybrid import GPUHBFHybridSystem
from .gpu_pd_dual_oracle import (
    ROUTE_BALANCED_TRACE_WORK,
    DualStrictInfiniteHBMOracle,
)
from .gpu_pd_dual_tiered import DualFiniteHBMTieredBaseline
from .gpu_pd_latency import load_p4d4_gpu_config
from .hbf_comparison_metrics import (
    CompletedRequest,
    RequestKey,
    SLOThresholds,
)
from .hbf_comparison_workload import (
    CallSpec,
    ScheduledSession,
    WorkloadValidationError,
    full_drain_hashes,
    stable_json_sha256,
)
from .hbf_full_model_latency import load_hbf_server_config


CELL_SCHEMA_VERSION = 1
SIMULATION_BACKEND = "python_analytical_discrete_event"
ASTRA_CYCLES_USED = False

MAX_NUM_BATCHED_TOKENS = 131_072
MAX_PREFILL_CHUNK_TOKENS = 131_072
P_MAX_NUM_SEQS = 32
D_MAX_NUM_SEQS = 128
SHARED_MAX_NUM_SEQS = 128

DEFAULT_FIRST_TTFT_SECONDS = 30.0
DEFAULT_RESUME_TTFT_SECONDS = 30.0
DEFAULT_TPOT_MILLISECONDS = 300.0

PINNED_GPU_CONFIG = Path("configs/wakekv_hbf/p4d4_gpu_server.json")
PINNED_HBF_CONFIG = Path("configs/wakekv_hbf/full_model_8card_server.json")
PINNED_HBF_WIDE_LPDDR_CONFIG = Path(
    "configs/wakekv_hbf/full_model_8card_server_wide_lpddr.json")

BASELINE_POLICIES = {
    "recompute": "hbm_lru_recompute",
    "ssd_direct": "ssd_direct",
    "cpu_ssd": "cpu_ssd",
}
HBF_LAYOUTS = {
    "hbf_dp8": "dp8",
    "hbf_tp4": "tp4",
    "hbf_tp8": "tp8",
    "hbf_tp4_wide": "tp4",
}
HBF_CONFIG_PATHS = {
    "hbf_dp8": PINNED_HBF_CONFIG,
    "hbf_tp4": PINNED_HBF_CONFIG,
    "hbf_tp8": PINNED_HBF_CONFIG,
    "hbf_tp4_wide": PINNED_HBF_WIDE_LPDDR_CONFIG,
}
SYSTEM_KEYS = (
    "recompute",
    "ssd_direct",
    "cpu_ssd",
    "oracle",
    "hbf_dp8",
    "hbf_tp4",
    "hbf_tp8",
    "hbf_tp4_wide",
)

REQUEST_CSV_FIELDS = (
    "system_key",
    "completion_identity",
    "session_id",
    "source_index",
    "offer_index",
    "call_index",
    "request_kind",
    "is_first_turn",
    "is_resume",
    "is_measurement",
    "expected_order_rank",
    "completion_order_rank",
    "input_tokens",
    "cached_prefix_tokens",
    "fresh_input_tokens",
    "tool_duration_ns",
    "execution_target",
    "execution_node_id",
    "execution_instance_id",
    "execution_group_id",
    "execution_policy",
    "route_reason",
    "release_ns",
    "first_token_ns",
    "completion_ns",
    "ttft_ns",
    "tpot_ns",
    "output_tokens",
    "ttft_slo_pass",
    "tpot_slo_eligible",
    "tpot_slo_pass",
    "all_slo_pass",
)


class ComparisonCellError(ValueError):
    """Raised when a cell violates the frozen comparison contract."""


def _positive_finite(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ComparisonCellError(
            f"{name} must be positive and finite")
    return float(value)


def _seconds_to_ns(name: str, value: object) -> int:
    seconds = _positive_finite(name, value)
    return int(round(seconds * 1_000_000_000))


def _milliseconds_to_ns(name: str, value: object) -> int:
    milliseconds = _positive_finite(name, value)
    return int(round(milliseconds * 1_000_000))


def build_slo_thresholds(
        *,
        first_ttft_seconds: float = DEFAULT_FIRST_TTFT_SECONDS,
        resume_ttft_seconds: float = DEFAULT_RESUME_TTFT_SECONDS,
        tpot_milliseconds: float = DEFAULT_TPOT_MILLISECONDS,
) -> SLOThresholds:
    """Convert the publication-facing SLO units to exact nanoseconds."""

    return SLOThresholds(
        first_ttft_ns=_seconds_to_ns(
            "first_ttft_seconds", first_ttft_seconds),
        resume_ttft_ns=_seconds_to_ns(
            "resume_ttft_seconds", resume_ttft_seconds),
        tpot_ns=_milliseconds_to_ns(
            "tpot_milliseconds", tpot_milliseconds),
    )


def _canonical_call_row(
        scheduled: ScheduledSession,
        call: CallSpec,
) -> dict[str, object]:
    return {
        "offer_index": scheduled.offer_index,
        "source_index": scheduled.session.source_index,
        "session_id": scheduled.session.session_id,
        "source_session_identity_sha256": (
            scheduled.session.source_session_identity_sha256
        ),
        "call_index": call.call_index,
        "completion_identity": call.completion_identity,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "tool_duration_ns": call.tool_duration_ns,
        "cached_prefix_tokens": call.cached_prefix_tokens,
        "fresh_input_tokens": call.fresh_input_tokens,
        "lineage_status": call.lineage_status,
        "inter_turn_gap_type": call.inter_turn_gap_type,
    }


def _validate_and_freeze_schedule(
        scheduled_sessions: tuple[ScheduledSession, ...],
) -> dict[str, object]:
    if not isinstance(scheduled_sessions, tuple):
        raise TypeError(
            "scheduled_sessions must be a frozen tuple")
    if not scheduled_sessions:
        raise ComparisonCellError(
            "scheduled_sessions cannot be empty")

    offer_indices = set()
    source_indices = set()
    session_ids = set()
    identities = set()
    call_rows = []
    schedule_rows = []
    descriptor_by_identity: dict[str, dict[str, object]] = {}
    session_call_identities: dict[str, list[str]] = {}
    previous_unit_arrival = 0.0
    previous_arrival_ns = -1

    for scheduled_rank, scheduled in enumerate(scheduled_sessions):
        if not isinstance(scheduled, ScheduledSession):
            raise TypeError(
                "scheduled_sessions contains a non-ScheduledSession value")
        session = scheduled.session
        if (
            isinstance(scheduled.offer_index, bool)
            or not isinstance(scheduled.offer_index, int)
            or scheduled.offer_index < 0
        ):
            raise ComparisonCellError(
                "offer_index must be a non-negative integer")
        if (
            isinstance(scheduled.arrival_time_ns, bool)
            or not isinstance(scheduled.arrival_time_ns, int)
            or scheduled.arrival_time_ns < 0
        ):
            raise ComparisonCellError(
                "arrival_time_ns must be a non-negative integer")
        for name in ("unit_interarrival", "unit_arrival_time"):
            coordinate = getattr(scheduled, name)
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not math.isfinite(float(coordinate))
                or float(coordinate) < 0.0
            ):
                raise ComparisonCellError(
                    f"{name} must be non-negative and finite")
        unit_interarrival = float(scheduled.unit_interarrival)
        unit_arrival_time = float(scheduled.unit_arrival_time)
        if scheduled.offer_index != scheduled_rank:
            raise ComparisonCellError(
                "scheduled tuple must be in contiguous offer_index order")
        if scheduled_rank == 0:
            if unit_interarrival != 0.0 or unit_arrival_time != 0.0:
                raise ComparisonCellError(
                    "the first offer must have zero unit interarrival "
                    "and zero unit arrival time")
        else:
            if unit_interarrival <= 0.0:
                raise ComparisonCellError(
                    "every offer after the first must have a positive "
                    "unit interarrival")
            if unit_arrival_time != (
                    previous_unit_arrival + unit_interarrival):
                raise ComparisonCellError(
                    "unit_arrival_time must exactly equal the cumulative "
                    "unit interarrival coordinates")
            if scheduled.arrival_time_ns < previous_arrival_ns:
                raise ComparisonCellError(
                    "scheduled arrivals must be nondecreasing")
        if scheduled.offer_index in offer_indices:
            raise ComparisonCellError("duplicate offer_index")
        if session.source_index in source_indices:
            raise ComparisonCellError("duplicate source_index")
        if session.session_id in session_ids:
            raise ComparisonCellError("duplicate session_id")
        if not session.calls:
            raise ComparisonCellError("scheduled session has no calls")
        offer_indices.add(scheduled.offer_index)
        source_indices.add(session.source_index)
        session_ids.add(session.session_id)

        session_identities = []
        for call_index, call in enumerate(session.calls):
            if not isinstance(call, CallSpec):
                raise TypeError(
                    "scheduled session contains a non-CallSpec value")
            if (
                call.session_id != session.session_id
                or call.source_index != session.source_index
                or call.call_index != call_index
            ):
                raise ComparisonCellError(
                    "call identity disagrees with its scheduled session")
            if call.fresh_input_tokens != (
                    call.input_tokens - call.cached_prefix_tokens):
                raise ComparisonCellError(
                    "fresh_input_tokens does not equal input minus prefix")
            if call_index == 0 and call.cached_prefix_tokens != 0:
                raise ComparisonCellError(
                    "first call cannot have a cached prefix")
            identity = call.completion_identity
            if identity in identities:
                raise ComparisonCellError(
                    f"duplicate completion identity {identity!r}")
            identities.add(identity)
            row = _canonical_call_row(scheduled, call)
            call_rows.append(row)
            descriptor_by_identity[identity] = {
                **row,
                "expected_order_rank": len(call_rows) - 1,
                "scheduled_arrival_time_ns": (
                    scheduled.arrival_time_ns),
            }
            session_identities.append(identity)
        session_call_identities[session.session_id] = session_identities
        schedule_rows.append({
            "scheduled_rank": scheduled_rank,
            "offer_index": scheduled.offer_index,
            "source_index": session.source_index,
            "session_id": session.session_id,
            "arrival_time_ns": scheduled.arrival_time_ns,
            # ``float.hex`` preserves the exact frozen Poisson coordinates.
            "unit_interarrival_hex": float(
                scheduled.unit_interarrival).hex(),
            "unit_arrival_time_hex": float(
                scheduled.unit_arrival_time).hex(),
            "completion_identities": session_identities,
        })
        previous_unit_arrival = unit_arrival_time
        previous_arrival_ns = scheduled.arrival_time_ns

    return {
        "session_count": len(scheduled_sessions),
        "call_count": len(call_rows),
        "session_ids": tuple(
            scheduled.session.session_id
            for scheduled in scheduled_sessions
        ),
        "completion_identities": tuple(
            row["completion_identity"] for row in call_rows
        ),
        "call_rows": tuple(call_rows),
        "schedule_rows": tuple(schedule_rows),
        "descriptor_by_identity": descriptor_by_identity,
        "session_call_identities": session_call_identities,
        "call_specs_sha256": stable_json_sha256(call_rows),
        "schedule_sha256": stable_json_sha256(schedule_rows),
    }


def _validate_rate_scaled_arrivals(
        scheduled_sessions: tuple[ScheduledSession, ...],
        session_rate: float,
) -> int:
    """Validate that one frozen unit-rate plan was scaled at ``session_rate``.

    ``OfferedPlan.at_rate`` defines each arrival as:

    ``start_time_ns + round(unit_arrival_time * 1e9 / session_rate)``.

    The first offer has unit coordinate zero, so its scheduled arrival
    identifies ``start_time_ns`` without accepting a caller-supplied offset.
    """

    first = scheduled_sessions[0]
    first_offset = int(round(
        float(first.unit_arrival_time)
        * 1_000_000_000
        / session_rate
    ))
    start_time_ns = first.arrival_time_ns - first_offset
    if start_time_ns < 0:
        raise ComparisonCellError(
            "inferred schedule start_time_ns is negative")
    for scheduled in scheduled_sessions:
        expected_arrival_ns = start_time_ns + int(round(
            float(scheduled.unit_arrival_time)
            * 1_000_000_000
            / session_rate
        ))
        if scheduled.arrival_time_ns != expected_arrival_ns:
            raise ComparisonCellError(
                "scheduled arrival is inconsistent with session_rate: "
                f"offer_index={scheduled.offer_index}, "
                f"observed_arrival_ns={scheduled.arrival_time_ns}, "
                f"expected_arrival_ns={expected_arrival_ns}, "
                f"session_rate={session_rate}")
    return start_time_ns


def _expected_system_call_projection(
        frozen: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    descriptor_by_identity = frozen["descriptor_by_identity"]
    return {
        identity: {
            "completion_identity": identity,
            "source_index": descriptor["source_index"],
            "offer_index": descriptor["offer_index"],
            "session_id": descriptor["session_id"],
            "call_index": descriptor["call_index"],
            "input_tokens": descriptor["input_tokens"],
            "output_tokens": descriptor["output_tokens"],
            "cached_prefix_tokens": descriptor[
                "cached_prefix_tokens"],
            "tool_duration_ns": descriptor["tool_duration_ns"],
        }
        for identity, descriptor
        in descriptor_by_identity.items()
    }


def _normalize_system_call_specs(
        system_call_specs: Sequence[object],
) -> dict[str, dict[str, object]]:
    if isinstance(system_call_specs, (str, bytes)):
        raise TypeError(
            "system_call_specs must be a sequence of call specs")
    normalized = {}
    required_fields = (
        "session_id",
        "call_index",
        "input_tokens",
        "output_tokens",
        "cached_prefix_tokens",
        "tool_duration_ns",
    )
    for spec_index, spec in enumerate(system_call_specs):
        missing = [
            field for field in required_fields
            if not hasattr(spec, field)
        ]
        if missing:
            raise ComparisonCellError(
                "system call spec lacks canonical fields: "
                f"spec_index={spec_index}, missing={missing}")
        identity = getattr(spec, "completion_identity", None)
        if not isinstance(identity, str) or not identity:
            identity = (
                f"{spec.session_id}::call-{spec.call_index}")
        if identity in normalized:
            raise ComparisonCellError(
                "system call projection contains duplicate identity "
                f"{identity!r}")
        normalized[identity] = {
            "completion_identity": identity,
            # Source/offer metadata is compared whenever the system model
            # carries it, but older projections without those fields remain
            # admissible.
            "source_index": getattr(spec, "source_index", None),
            "offer_index": getattr(spec, "offer_index", None),
            "session_id": spec.session_id,
            "call_index": spec.call_index,
            "input_tokens": spec.input_tokens,
            "output_tokens": spec.output_tokens,
            "cached_prefix_tokens": spec.cached_prefix_tokens,
            "tool_duration_ns": spec.tool_duration_ns,
        }
    return normalized


def _validate_system_call_projection(
        *,
        frozen: Mapping[str, object],
        system_call_specs: Sequence[object],
) -> str:
    expected = _expected_system_call_projection(frozen)
    observed = _normalize_system_call_specs(system_call_specs)
    expected_ids = set(expected)
    observed_ids = set(observed)
    if expected_ids != observed_ids:
        raise ComparisonCellError(
            "system call projection identity mismatch: "
            f"missing={sorted(expected_ids - observed_ids)[:5]}, "
            f"unexpected={sorted(observed_ids - expected_ids)[:5]}")
    comparison_fields = (
        "source_index",
        "offer_index",
        "session_id",
        "call_index",
        "input_tokens",
        "output_tokens",
        "cached_prefix_tokens",
        "tool_duration_ns",
    )
    for identity in sorted(expected):
        expected_row = expected[identity]
        observed_row = observed[identity]
        mismatches = {}
        for field in comparison_fields:
            observed_value = observed_row[field]
            if (
                field in {"source_index", "offer_index"}
                and observed_value is None
            ):
                continue
            if observed_value != expected_row[field]:
                mismatches[field] = {
                    "expected": expected_row[field],
                    "observed": observed_value,
                }
        if mismatches:
            raise ComparisonCellError(
                "system call projection disagrees with frozen workload: "
                f"identity={identity!r}, mismatches={mismatches}")
    rows = [observed[identity] for identity in sorted(observed)]
    return stable_json_sha256(rows)


def validate_system_call_projection(
        scheduled_sessions: tuple[ScheduledSession, ...],
        system_call_specs: Sequence[object],
) -> str:
    """Validate the exact workload fields consumed by a system runner."""

    frozen = _validate_and_freeze_schedule(scheduled_sessions)
    return _validate_system_call_projection(
        frozen=frozen,
        system_call_specs=system_call_specs,
    )


def _measurement_roster(
        frozen: Mapping[str, object],
        measurement_identities: Optional[Sequence[str]],
) -> tuple[str, ...]:
    expected = tuple(frozen["completion_identities"])
    if measurement_identities is None:
        return expected
    if isinstance(measurement_identities, (str, bytes)):
        raise TypeError(
            "measurement_identities must be a sequence of identities")
    roster = tuple(measurement_identities)
    if not roster:
        raise ComparisonCellError(
            "measurement identity roster cannot be empty")
    if any(not isinstance(identity, str) or not identity for identity in roster):
        raise ComparisonCellError(
            "measurement identities must be non-empty strings")
    if len(roster) != len(set(roster)):
        raise ComparisonCellError(
            "measurement identity roster contains duplicates")
    unexpected = sorted(set(roster) - set(expected))
    if unexpected:
        raise ComparisonCellError(
            "measurement identity roster is not a subset of the frozen "
            f"schedule: unexpected={unexpected[:5]}")
    return roster


def make_comparison_system(
        *,
        repo_root: Path,
        system_key: str,
):
    """Construct one comparison system with the documented P4D4 limits."""

    if system_key not in SYSTEM_KEYS:
        raise ComparisonCellError(
            f"unsupported system_key={system_key!r}; "
            f"supported={list(SYSTEM_KEYS)}")
    root = Path(repo_root)
    gpu_hardware = load_p4d4_gpu_config(root / PINNED_GPU_CONFIG)
    common = {
        "repo_root": root,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": SHARED_MAX_NUM_SEQS,
        "p_max_num_seqs": P_MAX_NUM_SEQS,
        "d_max_num_seqs": D_MAX_NUM_SEQS,
        "max_prefill_chunk_tokens": MAX_PREFILL_CHUNK_TOKENS,
        "validate_every_event": False,
    }
    if system_key in BASELINE_POLICIES:
        system = DualFiniteHBMTieredBaseline(
            hardware=gpu_hardware,
            policy=BASELINE_POLICIES[system_key],
            route_policy=ROUTE_BALANCED_TRACE_WORK,
            **common,
        )
    elif system_key == "oracle":
        system = DualStrictInfiniteHBMOracle(
            hardware=gpu_hardware,
            route_policy=ROUTE_BALANCED_TRACE_WORK,
            **common,
        )
    else:
        hbf_config_path = HBF_CONFIG_PATHS[system_key]
        hbf_hardware, layouts = load_hbf_server_config(
            root / hbf_config_path)
        layout_key = HBF_LAYOUTS[system_key]
        if layout_key not in layouts:
            raise ComparisonCellError(
                f"pinned HBF config lacks layout={layout_key!r}")
        system = GPUHBFHybridSystem(
            gpu_hardware=gpu_hardware,
            hbf_hardware=hbf_hardware,
            hbf_layout=layouts[layout_key],
            **common,
        )
    _assert_documented_factory_contract(system_key, system)
    return system


def _pool_contract(pool: object) -> dict[str, object]:
    return {
        "max_num_batched_tokens": pool.max_num_batched_tokens,
        "max_num_seqs": pool.max_num_seqs,
        "p_max_num_seqs": getattr(pool, "p_max_num_seqs", None),
        "d_max_num_seqs": getattr(pool, "d_max_num_seqs", None),
        "max_prefill_chunk_tokens": pool.max_prefill_chunk_tokens,
        "validate_every_event": pool.validate_every_event,
    }


def _assert_gpu_pool_contract(pool: object) -> None:
    observed = _pool_contract(pool)
    expected = {
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": SHARED_MAX_NUM_SEQS,
        "p_max_num_seqs": P_MAX_NUM_SEQS,
        "d_max_num_seqs": D_MAX_NUM_SEQS,
        "max_prefill_chunk_tokens": MAX_PREFILL_CHUNK_TOKENS,
        "validate_every_event": False,
    }
    if observed != expected:
        raise AssertionError(
            f"GPU pool contract mismatch: {observed!r}")


def _assert_documented_factory_contract(
        system_key: str,
        system: object,
) -> None:
    if system.validate_every_event:
        raise AssertionError(
            "comparison factories must disable per-event validation")
    if isinstance(system, (
            DualFiniteHBMTieredBaseline,
            DualStrictInfiniteHBMOracle)):
        if len(system.nodes) != 2:
            raise AssertionError(
                "baseline/oracle must contain two P4D4 servers")
        if system.route_policy != ROUTE_BALANCED_TRACE_WORK:
            raise AssertionError(
                "dual systems must use frozen balanced routing")
        for node in system.nodes:
            _assert_gpu_pool_contract(node.pool)
    elif isinstance(system, GPUHBFHybridSystem):
        _assert_gpu_pool_contract(system.node.gpu_pool)
        hbf_pool = system.node.hbf_pool
        if {
            "max_num_batched_tokens": hbf_pool.max_num_batched_tokens,
            "max_num_seqs": hbf_pool.max_num_seqs,
            "max_prefill_chunk_tokens": (
                hbf_pool.max_prefill_chunk_tokens),
            "validate_every_event": hbf_pool.validate_every_event,
        } != {
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "max_num_seqs": SHARED_MAX_NUM_SEQS,
            "max_prefill_chunk_tokens": MAX_PREFILL_CHUNK_TOKENS,
            "validate_every_event": False,
        }:
            raise AssertionError("HBF pool contract mismatch")
        if system.node.hbf_layout.key != HBF_LAYOUTS[system_key]:
            raise AssertionError("HBF layout factory mismatch")
    else:
        raise TypeError("unknown comparison system type")


def _config_file_contract(
        *,
        repo_root: Path,
        relative_path: Path,
        effective_values: Mapping[str, object],
) -> dict[str, object]:
    resolved = (repo_root / relative_path).resolve()
    try:
        content = resolved.read_bytes()
    except OSError as exc:
        raise ComparisonCellError(
            f"cannot read pinned hardware config {resolved}") from exc
    return {
        "repo_relative_path": str(relative_path),
        "resolved_path": str(resolved),
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "effective_values": dict(effective_values),
    }


def _documented_contract(
        system_key: str,
        system: object,
) -> dict[str, object]:
    if isinstance(system, GPUHBFHybridSystem):
        repo_root = Path(system.node.repo_root).resolve()
        gpu_pool_count = 1
        hbf_pool_contract = {
            **_pool_contract(system.node.hbf_pool),
            "layout": system.node.hbf_layout.key,
        }
        gpu_hardware = system.node.gpu_hardware
        hbf_hardware = system.node.hbf_hardware
        routing = (
            "first GPU; migration-inflight resume GPU; "
            "HBF-ready resume HBF end-to-end"
        )
    else:
        repo_root = Path(system.repo_root).resolve()
        gpu_pool_count = 2
        hbf_pool_contract = None
        gpu_hardware = system.hardware
        hbf_hardware = None
        routing = ROUTE_BALANCED_TRACE_WORK
    gpu_hardware_contract = _config_file_contract(
        repo_root=repo_root,
        relative_path=PINNED_GPU_CONFIG,
        effective_values=asdict(gpu_hardware),
    )
    hbf_hardware_contract = (
        None
        if hbf_hardware is None
        else _config_file_contract(
            repo_root=repo_root,
            relative_path=HBF_CONFIG_PATHS[system_key],
            effective_values=asdict(hbf_hardware),
        )
    )
    return {
        "system_key": system_key,
        "repo_root": str(repo_root),
        "execution_backend": {
            "name": SIMULATION_BACKEND,
            "astra_cycles_used": ASTRA_CYCLES_USED,
            "latency_source": (
                "H100-kernel-calibrated analytical models plus explicit "
                "bandwidth, latency, capacity, and resource calendars"
            ),
            "astra_conformance_scope": (
                "operation structure, collective dimensions, dependencies, "
                "and byte counts only; cycle equality is not claimed"
            ),
        },
        "gpu_server_count": (
            1 if isinstance(system, GPUHBFHybridSystem) else 2),
        "gpu_pool_count": gpu_pool_count,
        "hbf_server_count": (
            1 if isinstance(system, GPUHBFHybridSystem) else 0),
        "gpu_pool": {
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "max_num_seqs": SHARED_MAX_NUM_SEQS,
            "p_max_num_seqs": P_MAX_NUM_SEQS,
            "d_max_num_seqs": D_MAX_NUM_SEQS,
            "max_prefill_chunk_tokens": MAX_PREFILL_CHUNK_TOKENS,
        },
        "hbf_pool": hbf_pool_contract,
        "validate_every_event": False,
        "routing": routing,
        "hardware": {
            "gpu": gpu_hardware_contract,
            "hbf": hbf_hardware_contract,
        },
    }


def _request_identity(request: CompletedRequest) -> str:
    return (
        f"{request.key.session_id}::call-"
        f"{request.key.sub_request_index}"
    )


def _validate_completed_requests(
        completed: Sequence[CompletedRequest],
        frozen: Mapping[str, object],
) -> tuple[dict[str, CompletedRequest], tuple[str, ...]]:
    expected = tuple(frozen["completion_identities"])
    descriptor_by_identity = frozen["descriptor_by_identity"]
    completed_by_identity: dict[str, CompletedRequest] = {}
    completion_order = []
    for request in completed:
        if not isinstance(request, CompletedRequest):
            raise ComparisonCellError(
                "system returned a non-CompletedRequest value")
        identity = _request_identity(request)
        if identity in completed_by_identity:
            raise ComparisonCellError(
                f"duplicate completed identity {identity!r}")
        completed_by_identity[identity] = request
        completion_order.append(identity)
    # This call provides both the audit hashes and a fail-closed set check.
    _cell_full_drain_hashes(expected, completion_order)
    for identity, request in completed_by_identity.items():
        descriptor = descriptor_by_identity[identity]
        if request.output_tokens != descriptor["output_tokens"]:
            raise ComparisonCellError(
                f"output-token mismatch for {identity!r}")
        if (
            request.key.session_id != descriptor["session_id"]
            or request.key.sub_request_index != descriptor["call_index"]
        ):
            raise ComparisonCellError(
                f"request-key mismatch for {identity!r}")
    return completed_by_identity, tuple(completion_order)


def _cell_full_drain_hashes(
        expected_identities: Iterable[str],
        completed_identities: Iterable[str],
):
    try:
        return full_drain_hashes(
            expected_identities, completed_identities)
    except WorkloadValidationError as exc:
        raise ComparisonCellError(str(exc)) from exc


def _validate_causal_release_timestamps(
        completed_by_identity: Mapping[str, CompletedRequest],
        frozen: Mapping[str, object],
) -> None:
    """Prove that every runtime release obeys the closed-loop trace."""

    descriptor_by_identity = frozen["descriptor_by_identity"]
    for session_id, identities in frozen[
            "session_call_identities"].items():
        if not identities:
            raise ComparisonCellError(
                f"frozen session {session_id!r} has no calls")
        first_identity = identities[0]
        first = completed_by_identity[first_identity]
        scheduled_arrival_ns = descriptor_by_identity[
            first_identity]["scheduled_arrival_time_ns"]
        if first.release_ns != scheduled_arrival_ns:
            raise ComparisonCellError(
                "first-call release disagrees with scheduled arrival: "
                f"identity={first_identity!r}, "
                f"release_ns={first.release_ns}, "
                f"scheduled_arrival_ns={scheduled_arrival_ns}")
        for previous_identity, identity in zip(
                identities, identities[1:]):
            previous = completed_by_identity[previous_identity]
            request = completed_by_identity[identity]
            tool_duration_ns = descriptor_by_identity[
                previous_identity]["tool_duration_ns"]
            expected_release_ns = (
                previous.completion_ns + tool_duration_ns)
            if request.release_ns != expected_release_ns:
                raise ComparisonCellError(
                    "resume release violates causal successor timing: "
                    f"identity={identity!r}, "
                    f"release_ns={request.release_ns}, "
                    f"expected_release_ns={expected_release_ns}, "
                    f"predecessor={previous_identity!r}")


def validate_causal_release_contract(
        scheduled_sessions: tuple[ScheduledSession, ...],
        completed: Sequence[CompletedRequest],
) -> dict[str, object]:
    """Validate identity, output-token, and closed-loop release equality."""

    frozen = _validate_and_freeze_schedule(scheduled_sessions)
    completed_by_identity, completion_order = (
        _validate_completed_requests(completed, frozen)
    )
    _validate_causal_release_timestamps(
        completed_by_identity, frozen)
    return asdict(_cell_full_drain_hashes(
        frozen["completion_identities"],
        completion_order,
    ))


def _request_slo_values(
        request: CompletedRequest,
        thresholds: SLOThresholds,
) -> tuple[bool, Optional[bool], bool]:
    ttft_limit = (
        thresholds.resume_ttft_ns
        if request.is_resume else thresholds.first_ttft_ns
    )
    ttft_pass = request.ttft_ns <= ttft_limit
    tpot = request.tpot_ns
    tpot_pass = None if tpot is None else tpot <= thresholds.tpot_ns
    all_pass = ttft_pass and (
        True if tpot_pass is None else tpot_pass
    )
    return ttft_pass, tpot_pass, all_pass


def _nearest_rank(
        values: Sequence[float],
        percentile: float,
) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _distribution(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {
            "count": 0,
            "mean_ns": None,
            "p50_ns": None,
            "p90_ns": None,
            "p95_ns": None,
            "p99_ns": None,
            "percentile_method": "inclusive_nearest_rank",
        }
    converted = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in converted):
        raise ComparisonCellError(
            "latency distribution contains a non-finite value")
    return {
        "count": len(converted),
        "mean_ns": math.fsum(converted) / len(converted),
        "p50_ns": _nearest_rank(converted, 0.50),
        "p90_ns": _nearest_rank(converted, 0.90),
        "p95_ns": _nearest_rank(converted, 0.95),
        "p99_ns": _nearest_rank(converted, 0.99),
        "percentile_method": "inclusive_nearest_rank",
    }


def summarize_measurement_requests(
        requests: Sequence[CompletedRequest],
        *,
        session_rate: float,
        thresholds: SLOThresholds,
) -> dict[str, object]:
    """Summarize an already-selected measurement identity roster."""

    rate = _positive_finite("session_rate", session_rate)
    if not isinstance(thresholds, SLOThresholds):
        raise TypeError("thresholds must be SLOThresholds")
    if not requests:
        raise ComparisonCellError(
            "measurement request cohort cannot be empty")
    keys = [request.key for request in requests]
    if len(keys) != len(set(keys)):
        raise ComparisonCellError(
            "measurement request cohort contains duplicate identities")

    first = tuple(request for request in requests if not request.is_resume)
    resume = tuple(request for request in requests if request.is_resume)
    eligible = tuple(
        request for request in requests if request.tpot_ns is not None
    )
    first_eligible = tuple(
        request for request in first if request.tpot_ns is not None
    )
    resume_eligible = tuple(
        request for request in resume if request.tpot_ns is not None
    )
    evaluations = [
        (request, *_request_slo_values(request, thresholds))
        for request in requests
    ]
    ttft_pass = sum(row[1] for row in evaluations)
    tpot_pass = sum(
        row[2] is True for row in evaluations if row[2] is not None)
    all_pass_requests = tuple(
        row[0] for row in evaluations if row[3])
    first_evaluations = tuple(
        row for row in evaluations if not row[0].is_resume)
    resume_evaluations = tuple(
        row for row in evaluations if row[0].is_resume)
    measured_sessions = {
        request.key.session_id for request in requests}
    measured_session_count = len(measured_sessions)
    request_count = len(requests)
    output_tokens = sum(request.output_tokens for request in requests)
    all_pass_count = len(all_pass_requests)
    all_pass_output_tokens = sum(
        request.output_tokens for request in all_pass_requests)

    completion_order = tuple(sorted(
        requests,
        key=lambda request: (
            request.completion_ns,
            request.key.session_id,
            request.key.sub_request_index,
        ),
    ))
    completion_start = completion_order[0].completion_ns
    completion_end = completion_order[-1].completion_ns
    completion_span = completion_end - completion_start
    observed_request_rate = (
        (request_count - 1) * 1_000_000_000 / completion_span
        if completion_span and request_count > 1 else None
    )
    interval_output_tokens = sum(
        request.output_tokens for request in completion_order[1:])
    observed_output_rate = (
        interval_output_tokens * 1_000_000_000 / completion_span
        if completion_span and request_count > 1 else None
    )

    def _fraction(numerator: int, denominator: int) -> Optional[float]:
        return numerator / denominator if denominator else None

    first_ttft_pass = sum(row[1] for row in first_evaluations)
    resume_ttft_pass = sum(row[1] for row in resume_evaluations)

    def _kind_summary(
            kind: str,
            kind_requests: Sequence[CompletedRequest],
    ) -> dict[str, object]:
        kind_evaluations = tuple(
            row for row in evaluations
            if (
                kind == "all"
                or (kind == "resume") == row[0].is_resume
            )
        )
        kind_eligible = tuple(
            row for row in kind_evaluations
            if row[0].tpot_ns is not None
        )
        kind_ttft_pass = sum(row[1] for row in kind_evaluations)
        kind_tpot_pass = sum(
            row[2] is True for row in kind_eligible)
        kind_joint_pass = tuple(
            row[0] for row in kind_evaluations if row[3])
        kind_joint_tokens = sum(
            request.output_tokens for request in kind_joint_pass)
        return {
            "kind": kind,
            "request_count": len(kind_requests),
            "output_tokens": sum(
                request.output_tokens for request in kind_requests),
            "tpot_eligible_count": len(kind_eligible),
            "ttft_ns": _distribution([
                request.ttft_ns for request in kind_requests
            ]),
            "tpot_ns": _distribution([
                request.tpot_ns for request in kind_requests
                if request.tpot_ns is not None
            ]),
            "slo": {
                "ttft_pass_count": kind_ttft_pass,
                "ttft_pass_fraction": _fraction(
                    kind_ttft_pass, len(kind_requests)),
                "tpot_pass_count": kind_tpot_pass,
                "tpot_pass_fraction_of_eligible": _fraction(
                    kind_tpot_pass, len(kind_eligible)),
                "joint_pass_count": len(kind_joint_pass),
                "joint_pass_fraction": _fraction(
                    len(kind_joint_pass), len(kind_requests)),
                "joint_pass_output_tokens": kind_joint_tokens,
            },
            "offered_load_normalized_request_goodput": {
                "label": (
                    f"offered-load-normalized {kind} request goodput"),
                "unit": "requests/s",
                "value": (
                    rate * len(kind_joint_pass)
                    / measured_session_count
                ),
            },
            "offered_load_normalized_output_token_goodput": {
                "label": (
                    "offered-load-normalized "
                    f"{kind} output-token goodput"),
                "unit": "output tokens/s",
                "value": (
                    rate * kind_joint_tokens
                    / measured_session_count
                ),
            },
        }

    return {
        "counts": {
            "measurement_sessions": measured_session_count,
            "measurement_calls": request_count,
            "first_calls": len(first),
            "resume_calls": len(resume),
            "tpot_eligible_calls": len(eligible),
            "output_tokens": output_tokens,
        },
        "latency_distributions_ns": {
            "first_ttft": _distribution([
                request.ttft_ns for request in first
            ]),
            "resume_ttft": _distribution([
                request.ttft_ns for request in resume
            ]),
            "tpot_eligible": _distribution([
                request.tpot_ns for request in eligible
                if request.tpot_ns is not None
            ]),
            "first_tpot_eligible": _distribution([
                request.tpot_ns for request in first_eligible
                if request.tpot_ns is not None
            ]),
            "resume_tpot_eligible": _distribution([
                request.tpot_ns for request in resume_eligible
                if request.tpot_ns is not None
            ]),
        },
        "request_kind_summaries": {
            "all": _kind_summary("all", requests),
            "first": _kind_summary("first", first),
            "resume": _kind_summary("resume", resume),
        },
        "slo": {
            "thresholds_ns": asdict(thresholds),
            "first_ttft_pass_count": first_ttft_pass,
            "first_ttft_pass_fraction": _fraction(
                first_ttft_pass, len(first)),
            "resume_ttft_pass_count": resume_ttft_pass,
            "resume_ttft_pass_fraction": _fraction(
                resume_ttft_pass, len(resume)),
            "ttft_pass_count": ttft_pass,
            "ttft_pass_fraction": _fraction(
                ttft_pass, request_count),
            "tpot_pass_count": tpot_pass,
            "tpot_pass_fraction_of_eligible": _fraction(
                tpot_pass, len(eligible)),
            "all_slo_pass_count": all_pass_count,
            "all_slo_pass_fraction": _fraction(
                all_pass_count, request_count),
            "all_slo_pass_output_tokens": all_pass_output_tokens,
        },
        "offered_load_normalized_request_goodput": {
            "label": "offered-load-normalized request goodput",
            "unit": "requests/s",
            "value": (
                rate
                * request_count
                / measured_session_count
                * all_pass_count
                / request_count
            ),
            "formula": (
                "session_rate * measured_calls / measured_sessions "
                "* all_SLO_pass_fraction"
            ),
        },
        "offered_load_normalized_output_token_goodput": {
            "label": "offered-load-normalized output-token goodput",
            "unit": "output tokens/s",
            "value": (
                rate
                * all_pass_output_tokens
                / measured_session_count
            ),
            "formula": (
                "session_rate * all_SLO_pass_output_tokens "
                "/ measured_sessions"
            ),
        },
        "observed_completion_span_throughput": {
            "label": "observed inter-completion rate",
            "semantics": (
                "(N-1) completed-request intervals divided by the span "
                "from the first measured completion to the last; this is "
                "not offered-load-normalized goodput"
            ),
            "completion_start_ns": completion_start,
            "completion_end_ns": completion_end,
            "completion_span_ns": completion_span,
            "completion_event_count": request_count,
            "inter_completion_interval_count": max(
                0, request_count - 1),
            "interval_output_tokens": interval_output_tokens,
            "requests_per_second": observed_request_rate,
            "output_tokens_per_second": observed_output_rate,
            "zero_span_value": None,
        },
    }


def _build_request_rows(
        *,
        system_key: str,
        system: object,
        completed_by_identity: Mapping[str, CompletedRequest],
        completion_order: Sequence[str],
        frozen: Mapping[str, object],
        measurement_roster: Sequence[str],
        thresholds: SLOThresholds,
) -> list[dict[str, object]]:
    descriptor_by_identity = frozen["descriptor_by_identity"]
    measurement_set = set(measurement_roster)
    completion_rank = {
        identity: rank
        for rank, identity in enumerate(completion_order)
    }
    system_specs = {
        spec.completion_identity: spec
        for spec in system.call_specs
    }
    rows = []
    for identity in frozen["completion_identities"]:
        request = completed_by_identity[identity]
        descriptor = descriptor_by_identity[identity]
        system_spec = system_specs[identity]
        if isinstance(system, GPUHBFHybridSystem):
            runtime = system.node.calls[system_spec.request_id]
            if runtime.execution is None:
                raise ComparisonCellError(
                    f"hybrid call {identity!r} lacks an execution route")
            hbf_execution = runtime.hbf_request is not None
            execution_target = "hbf" if hbf_execution else "gpu"
            execution_group_id = (
                runtime.hbf_request.group_id
                if hbf_execution else None
            )
            execution_node_id = (
                1 if hbf_execution else system.node.gpu_node_id
            )
            execution_instance_id = (
                f"hbf-group-{execution_group_id}"
                if hbf_execution
                else f"gpu-node-{system.node.gpu_node_id}"
            )
            execution_policy = runtime.execution.value
            route_reason = runtime.route_reason
        else:
            execution_target = "gpu"
            execution_node_id = system_spec.node_id
            execution_instance_id = (
                f"gpu-node-{system_spec.node_id}")
            execution_group_id = None
            execution_policy = (
                BASELINE_POLICIES[system_key]
                if system_key in BASELINE_POLICIES
                else "strict_infinite_hbm_residency_oracle"
            )
            route_reason = system.route_policy
        ttft_pass, tpot_pass, all_pass = _request_slo_values(
            request, thresholds)
        rows.append({
            "system_key": system_key,
            "completion_identity": identity,
            "session_id": request.key.session_id,
            "source_index": descriptor["source_index"],
            "offer_index": descriptor["offer_index"],
            "call_index": request.key.sub_request_index,
            "request_kind": (
                "resume" if request.is_resume else "first"),
            "is_first_turn": not request.is_resume,
            "is_resume": request.is_resume,
            "is_measurement": identity in measurement_set,
            "expected_order_rank": descriptor["expected_order_rank"],
            "completion_order_rank": completion_rank[identity],
            "input_tokens": descriptor["input_tokens"],
            "cached_prefix_tokens": (
                descriptor["cached_prefix_tokens"]),
            "fresh_input_tokens": descriptor["fresh_input_tokens"],
            "tool_duration_ns": descriptor["tool_duration_ns"],
            "execution_target": execution_target,
            "execution_node_id": execution_node_id,
            "execution_instance_id": execution_instance_id,
            "execution_group_id": execution_group_id,
            "execution_policy": execution_policy,
            "route_reason": route_reason,
            "release_ns": request.release_ns,
            "first_token_ns": request.first_token_ns,
            "completion_ns": request.completion_ns,
            "ttft_ns": request.ttft_ns,
            "tpot_ns": request.tpot_ns,
            "output_tokens": request.output_tokens,
            "ttft_slo_pass": ttft_pass,
            "tpot_slo_eligible": request.tpot_ns is not None,
            "tpot_slo_pass": tpot_pass,
            "all_slo_pass": all_pass,
        })
    return rows


def _calendar_report(calendar: object, horizon_ns: int) -> dict[str, object]:
    resources = sorted(
        set(calendar.available_ns)
        | set(calendar.busy_ns)
        | set(calendar.reservation_count_by_resource)
        | set(calendar.reservation_bytes_by_resource)
    )
    rows = {
        resource_name: {
            "available_ns": calendar.available_ns.get(resource_name, 0),
            "busy_ns": calendar.busy_ns.get(resource_name, 0),
            "utilization": (
                calendar.busy_ns.get(resource_name, 0) / horizon_ns
                if horizon_ns else 0.0
            ),
            "reservation_count": (
                calendar.reservation_count_by_resource.get(
                    resource_name, 0)
            ),
            "reservation_bytes": (
                calendar.reservation_bytes_by_resource.get(
                    resource_name, 0)
            ),
        }
        for resource_name in resources
    }
    ranked = sorted(
        (
            (row["utilization"], name)
            for name, row in rows.items()
        ),
        reverse=True,
    )
    return {
        "horizon_ns": horizon_ns,
        "resources": rows,
        "highest_utilization_resources": [
            {
                "resource": name,
                "utilization": utilization,
            }
            for utilization, name in ranked[:8]
        ],
    }


def _hbm_report(hbm: object) -> dict[str, object]:
    return {
        "p_capacity_bytes_per_rank": hbm.p_capacity_bytes_per_rank,
        "d_capacity_bytes_per_rank": hbm.d_capacity_bytes_per_rank,
        "p_used_bytes_per_rank": hbm.p_used_bytes_per_rank,
        "d_used_bytes_per_rank": hbm.d_used_bytes_per_rank,
        "metrics": asdict(hbm.metrics),
    }


def _ledger_report(ledger: object) -> dict[str, object]:
    return {
        "capacity_bytes": ledger.capacity_bytes,
        "used_bytes": ledger.used_bytes,
        "peak_used_bytes": ledger.peak_used_bytes,
        "peak_fraction": (
            ledger.peak_used_bytes / ledger.capacity_bytes
        ),
    }


def _compact_bottleneck_report(
        system_key: str,
        system: object,
) -> dict[str, object]:
    horizon_ns = system.current_ns
    if isinstance(system, DualFiniteHBMTieredBaseline):
        nodes = []
        for node in system.nodes:
            nodes.append({
                "node_id": node.node_id,
                "node_metrics": asdict(node.metrics),
                "pool_metrics": asdict(node.pool.metrics),
                "lifecycle_metrics": asdict(node.lifecycle.metrics),
                "tier_ledgers": {
                    "p": _ledger_report(node.lifecycle.p_ledger),
                    "d": _ledger_report(node.lifecycle.d_ledger),
                    "cpu": _ledger_report(node.lifecycle.cpu_ledger),
                    "ssd": _ledger_report(node.lifecycle.ssd_ledger),
                },
                "resource_utilization": _calendar_report(
                    node.calendar, horizon_ns),
            })
        return {
            "system_metrics": asdict(system.metrics),
            "policy": BASELINE_POLICIES[system_key],
            "nodes": nodes,
        }
    if isinstance(system, DualStrictInfiniteHBMOracle):
        return {
            "system_metrics": asdict(system.metrics),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_metrics": asdict(node.metrics),
                    "pool_metrics": asdict(node.pool.metrics),
                    "hbm": _hbm_report(node.hbm),
                    "resource_utilization": _calendar_report(
                        node.calendar, horizon_ns),
                }
                for node in system.nodes
            ],
        }
    if isinstance(system, GPUHBFHybridSystem):
        node = system.node
        hbf_ledger = node.hbf_pool.lpddr_ledger
        return {
            "system_metrics": asdict(system.metrics),
            "node_metrics": asdict(node.metrics),
            "execution_counts": dict(sorted(Counter(
                call.execution.value
                for call in node.calls.values()
                if call.execution is not None
            ).items())),
            "gpu_pool_metrics": asdict(node.gpu_pool.metrics),
            "hbf_pool_metrics": asdict(node.hbf_pool.metrics),
            "hbf_lifecycle_metrics": asdict(node.hbf_lifecycle.metrics),
            "gpu_hbm": _hbm_report(node.gpu_hbm),
            "hbf_storage": {
                "layout": node.hbf_layout.key,
                "usable_bytes_per_card": (
                    node.hbf_lifecycle.usable_bytes_per_card),
                "reserved_per_card_bytes_by_group": dict(
                    node.hbf_lifecycle._reserved_per_card_by_group),
                "lpddr_capacity_bytes_per_group": (
                    hbf_ledger.capacity_bytes),
                "lpddr_used_bytes_per_group": {
                    str(group_id): hbf_ledger.used_bytes(group_id)
                    for group_id in range(hbf_ledger.group_count)
                },
                "lpddr_peak_bytes_per_group": {
                    str(group_id): hbf_ledger.peak_used_bytes[group_id]
                    for group_id in range(hbf_ledger.group_count)
                },
            },
            "gpu_resource_utilization": _calendar_report(
                node.gpu_calendar, horizon_ns),
            "hbf_resource_utilization": _calendar_report(
                node.hbf_calendar, horizon_ns),
        }
    raise TypeError("unknown comparison system type")


def _proc_memory_bytes() -> tuple[Optional[int], Optional[int], str]:
    """Return current RSS and process high-water RSS when available."""

    status = Path("/proc/self/status")
    if status.is_file():
        values = {}
        try:
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith(("VmRSS:", "VmHWM:")):
                    name, raw, unit = line.split()
                    if unit != "kB":
                        continue
                    values[name.rstrip(":")] = int(raw) * 1024
        except (OSError, ValueError):
            values = {}
        if values:
            return (
                values.get("VmRSS"),
                values.get("VmHWM"),
                "linux_proc_status",
            )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    multiplier = 1 if sys.platform == "darwin" else 1024
    return None, int(peak * multiplier), "resource_getrusage"


def run_comparison_cell(
        *,
        repo_root: Path,
        system_key: str,
        scheduled_sessions: tuple[ScheduledSession, ...],
        session_rate: float,
        measurement_identities: Optional[Sequence[str]] = None,
        first_ttft_seconds: float = DEFAULT_FIRST_TTFT_SECONDS,
        resume_ttft_seconds: float = DEFAULT_RESUME_TTFT_SECONDS,
        tpot_milliseconds: float = DEFAULT_TPOT_MILLISECONDS,
) -> dict[str, object]:
    """Run exactly one fully drained comparison cell."""

    rate = _positive_finite("session_rate", session_rate)
    frozen = _validate_and_freeze_schedule(scheduled_sessions)
    inferred_start_time_ns = _validate_rate_scaled_arrivals(
        scheduled_sessions, rate)
    roster = _measurement_roster(frozen, measurement_identities)
    thresholds = build_slo_thresholds(
        first_ttft_seconds=first_ttft_seconds,
        resume_ttft_seconds=resume_ttft_seconds,
        tpot_milliseconds=tpot_milliseconds,
    )
    system = make_comparison_system(
        repo_root=Path(repo_root),
        system_key=system_key,
    )

    rss_before, hwm_before, memory_source_before = _proc_memory_bytes()
    wall_start_ns = time.perf_counter_ns()
    completed = tuple(system.run(scheduled_sessions))
    elapsed_wall_ns = time.perf_counter_ns() - wall_start_ns
    rss_after, hwm_after, memory_source_after = _proc_memory_bytes()

    normalized_system_call_projection_sha256 = (
        _validate_system_call_projection(
            frozen=frozen,
            system_call_specs=system.call_specs,
        )
    )
    completed_by_identity, completion_order = (
        _validate_completed_requests(completed, frozen)
    )
    _validate_causal_release_timestamps(
        completed_by_identity, frozen)
    rows = _build_request_rows(
        system_key=system_key,
        system=system,
        completed_by_identity=completed_by_identity,
        completion_order=completion_order,
        frozen=frozen,
        measurement_roster=roster,
        thresholds=thresholds,
    )
    measured_requests = tuple(
        completed_by_identity[identity] for identity in roster)
    summary = summarize_measurement_requests(
        measured_requests,
        session_rate=rate,
        thresholds=thresholds,
    )

    call_drain = asdict(_cell_full_drain_hashes(
        frozen["completion_identities"],
        completion_order,
    ))
    last_identity_by_session = {
        session_id: identities[-1]
        for session_id, identities
        in frozen["session_call_identities"].items()
    }
    expected_session_order = tuple(frozen["session_ids"])
    expected_session_rank = {
        session_id: rank
        for rank, session_id in enumerate(expected_session_order)
    }
    completed_session_order = tuple(sorted(
        expected_session_order,
        key=lambda session_id: (
            completed_by_identity[
                last_identity_by_session[session_id]
            ].completion_ns,
            expected_session_rank[session_id],
        ),
    ))
    session_drain = asdict(_cell_full_drain_hashes(
        expected_session_order,
        completed_session_order,
    ))
    system_spec_rows = [
        asdict(spec) for spec in system.call_specs
    ]

    result = {
        "schema_version": CELL_SCHEMA_VERSION,
        "system_key": system_key,
        "session_rate": rate,
        "simulation_contract": _documented_contract(
            system_key, system),
        "frozen_workload": {
            "session_count": frozen["session_count"],
            "call_count": frozen["call_count"],
            "inferred_start_time_ns": inferred_start_time_ns,
            "call_specs_sha256": frozen["call_specs_sha256"],
            "schedule_sha256": frozen["schedule_sha256"],
            "expected_system_call_projection_sha256": (
                stable_json_sha256([
                    _expected_system_call_projection(frozen)[identity]
                    for identity in sorted(
                        _expected_system_call_projection(frozen))
                ])
            ),
            "normalized_system_call_projection_sha256": (
                normalized_system_call_projection_sha256),
            "expected_call_identities_sha256": stable_json_sha256(
                list(frozen["completion_identities"])),
            "expected_session_ids_sha256": stable_json_sha256(
                list(frozen["session_ids"])),
            "system_call_specs_sha256": stable_json_sha256(
                system_spec_rows),
        },
        "measurement_roster": {
            "identity_count": len(roster),
            "session_count": len({
                completed_by_identity[identity].key.session_id
                for identity in roster
            }),
            "ordered_identities_sha256": stable_json_sha256(
                list(roster)),
            "identity_set_sha256": stable_json_sha256(
                sorted(roster)),
            "excludes_non_roster_warmup_guard": (
                measurement_identities is not None),
        },
        "summary": summary,
        "full_drain": {
            "calls": call_drain,
            "sessions": session_drain,
        },
        "bottleneck_report": _compact_bottleneck_report(
            system_key, system),
        "execution_observation": {
            "simulated_horizon_ns": system.current_ns,
            "elapsed_wall_time_ns": elapsed_wall_ns,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "process_peak_rss_before_bytes": hwm_before,
            "process_peak_rss_after_bytes": hwm_after,
            "process_peak_rss_delta_bytes": (
                None
                if hwm_before is None or hwm_after is None
                else max(0, hwm_after - hwm_before)
            ),
            "memory_measurement_source_before": memory_source_before,
            "memory_measurement_source_after": memory_source_after,
            "peak_rss_scope": (
                "process high-water mark; the delta can be zero when an "
                "earlier cell established a higher process peak"
            ),
        },
        "requests": rows,
    }
    safe = json_safe(result)
    # Refuse NaN/Infinity and any accidental non-JSON object before returning.
    json.dumps(safe, allow_nan=False, sort_keys=True)
    return safe


def json_safe(value: object) -> object:
    """Convert supported report values to strict JSON primitives."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ComparisonCellError(
                "cannot serialize non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return json_safe(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                key = str(key)
            result[str(key)] = json_safe(item)
        return result
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value, key=str)]
    raise ComparisonCellError(
        f"unsupported JSON value type {type(value).__name__}")


def _atomic_text_writer(path: Path, write) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False) as temporary:
            temporary_name = temporary.name
            write(temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def write_json_atomic(path: Path, value: object) -> Path:
    """Write strict JSON through a same-directory atomic replacement."""

    safe = json_safe(value)

    def _write(handle) -> None:
        json.dump(
            safe,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")

    target = Path(path)
    _atomic_text_writer(target, _write)
    return target


def _validate_request_csv_rows(
        rows: Iterable[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    values = list(rows)
    for row_index, row in enumerate(values):
        if not isinstance(row, Mapping):
            raise ComparisonCellError(
                f"request CSV row {row_index} is not a mapping")
        unexpected = set(row) - set(REQUEST_CSV_FIELDS)
        missing = set(REQUEST_CSV_FIELDS) - set(row)
        if unexpected or missing:
            raise ComparisonCellError(
                "request CSV schema mismatch: "
                f"row={row_index}, missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}")
        for field in REQUEST_CSV_FIELDS:
            value = row[field]
            if isinstance(value, (Mapping, list, tuple, set, frozenset)):
                raise ComparisonCellError(
                    "request CSV values must be scalar: "
                    f"row={row_index}, field={field!r}")
            json_safe(value)
    return values


def write_request_csv_atomic(
        path: Path,
        rows: Iterable[Mapping[str, object]],
) -> Path:
    """Write flat per-request rows through an atomic replacement."""

    values = _validate_request_csv_rows(rows)

    def _write(handle) -> None:
        writer = csv.DictWriter(
            handle, fieldnames=REQUEST_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(values)

    target = Path(path)
    _atomic_text_writer(target, _write)
    return target


def write_cell_outputs_atomic(
        *,
        json_path: Path,
        csv_path: Path,
        result: Mapping[str, object],
) -> tuple[Path, Path]:
    """Write two caller-selected paths with JSON as the commit marker.

    This compatibility helper prevalidates both payloads, publishes CSV
    first, and atomically replaces JSON last.  New sweep code should use
    :func:`write_cell_output_bundle_atomic`, whose directory rename exposes
    both files together.
    """

    resolved_json = Path(json_path).resolve()
    resolved_csv = Path(csv_path).resolve()
    if resolved_json == resolved_csv:
        raise ComparisonCellError(
            "json_path and csv_path must resolve to distinct files")
    rows = result.get("requests")
    if not isinstance(rows, list):
        raise ComparisonCellError(
            "cell result lacks a request-row list")
    safe = json_safe(result)
    json.dumps(safe, allow_nan=False, sort_keys=True)
    validated_rows = _validate_request_csv_rows(rows)
    written_csv = write_request_csv_atomic(csv_path, validated_rows)
    written_json = write_json_atomic(json_path, safe)
    return written_json, written_csv


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_cell_output_bundle_atomic(
        output_dir: Path,
        result: Mapping[str, object],
) -> Path:
    """Publish ``cell.json`` and ``requests.csv`` as one new directory.

    The two complete files are first written into a hidden sibling directory.
    A same-filesystem directory rename is the only publication step.  Cell
    directories are immutable: an existing final path is never replaced.
    """

    target = Path(output_dir)
    rows = result.get("requests")
    if not isinstance(rows, list):
        raise ComparisonCellError(
            "cell result lacks a request-row list")
    safe = json_safe(result)
    json.dumps(safe, allow_nan=False, sort_keys=True)
    validated_rows = _validate_request_csv_rows(rows)

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        raise FileExistsError(
            f"cell output directory already exists: {target}")
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=parent,
    ))
    published = False
    try:
        write_json_atomic(temporary / "cell.json", safe)
        write_request_csv_atomic(
            temporary / "requests.csv", validated_rows)
        _fsync_directory(temporary)
        if os.path.lexists(target):
            raise FileExistsError(
                f"cell output directory already exists: {target}")
        os.rename(temporary, target)
        published = True
        _fsync_directory(parent)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)
    return target


__all__ = [
    "ASTRA_CYCLES_USED",
    "BASELINE_POLICIES",
    "CELL_SCHEMA_VERSION",
    "ComparisonCellError",
    "DEFAULT_FIRST_TTFT_SECONDS",
    "DEFAULT_RESUME_TTFT_SECONDS",
    "DEFAULT_TPOT_MILLISECONDS",
    "D_MAX_NUM_SEQS",
    "HBF_CONFIG_PATHS",
    "HBF_LAYOUTS",
    "MAX_NUM_BATCHED_TOKENS",
    "MAX_PREFILL_CHUNK_TOKENS",
    "P_MAX_NUM_SEQS",
    "PINNED_GPU_CONFIG",
    "PINNED_HBF_CONFIG",
    "PINNED_HBF_WIDE_LPDDR_CONFIG",
    "REQUEST_CSV_FIELDS",
    "SHARED_MAX_NUM_SEQS",
    "SIMULATION_BACKEND",
    "SYSTEM_KEYS",
    "build_slo_thresholds",
    "json_safe",
    "make_comparison_system",
    "run_comparison_cell",
    "summarize_measurement_requests",
    "validate_causal_release_contract",
    "validate_system_call_projection",
    "write_cell_outputs_atomic",
    "write_cell_output_bundle_atomic",
    "write_json_atomic",
    "write_request_csv_atomic",
]

"""Capacity-aware global replay for agentic idle KV tiering.

This module complements :mod:`agentic_kv_roofline`.  The older standalone
analysis classifies each tool gap independently and therefore cannot model
capacity pressure between concurrent sessions.  This replay retains session
arrival times, advances every dependency chain on one global clock, accounts
for analytical prompt execution, and enforces finite HBM, CPU DRAM, and SSD
budgets with an HBM -> CPU -> SSD -> recompute cascade.

The target-model execution time is an analytical prompt-only roofline by
default, or an explicitly supplied calibrated analytical prompt model. Calls
may overlap (there is no batching/compute-server queue) and decode execution is
not modeled. Active call KV is nevertheless reserved in HBM for the entire
analytical call. Consequently this is a capacity/transfer-queue sensitivity,
not a cycle-level serving result.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol

from .agentic_kv_roofline import (
    AnalysisConfigError,
    HardwareSpec,
    KvLayout,
    ModelShape,
    cpu_transfer_seconds,
    kv_layout,
    roofline_cached_prefill_seconds,
    roofline_recompute_seconds,
    ssd_media_seconds,
    ssd_transfer_seconds,
)


GIB = 1 << 30
SI_TB = 1_000_000_000_000

# The DGX H100 guide specifies eight 3.84-TB U.2 RAID 0 cache drives, and the
# firmware-guide example enumerates eight KIOXIA KCM6DRUL3T84 devices. KIOXIA
# rates that 3.84-TB PCIe 4.0 x4 model at up to 6.9/4.2 GB/s sequential
# read/write. The aggregate below infers an all-CM6 eight-drive manufacturer
# upper bound; end-to-end RAID 0 efficiency must be calibrated with fio.
DGX_H100_CM6_DEVICE_COUNT = 8
DGX_H100_CM6_READ_GBPS_PER_DEVICE = 6.9
DGX_H100_CM6_WRITE_GBPS_PER_DEVICE = 4.2
DGX_H100_CM6_IDEAL_READ_GBPS = (
    DGX_H100_CM6_DEVICE_COUNT * DGX_H100_CM6_READ_GBPS_PER_DEVICE
)
DGX_H100_CM6_IDEAL_WRITE_GBPS = (
    DGX_H100_CM6_DEVICE_COUNT * DGX_H100_CM6_WRITE_GBPS_PER_DEVICE
)
DGX_H100_NVLINK_BIDIRECTIONAL_GBPS_PER_GPU = 900.0
DGX_H100_NVLINK_ONE_WAY_GBPS_PER_GPU = (
    DGX_H100_NVLINK_BIDIRECTIONAL_GBPS_PER_GPU / 2
)
DGX_H100_SYSTEM_SPEC_URL = (
    "https://docs.nvidia.com/dgx/dgxh100-user-guide/"
    "introduction-to-dgxh100.html"
)
DGX_H100_NVME_SUPPORT_URL = (
    "https://docs.nvidia.com/dgx/dgxh100-fw-update-guide/"
    "nvme-fw-update.html"
)
KIOXIA_CM6_R_PRODUCT_BRIEF_URL = (
    "https://americas.kioxia.com/content/dam/kioxia/shared/business/ssd/"
    "enterprise-ssd/asset/productbrief/eSSD-CM6-R-product-brief.pdf"
)


class PromptComputeModel(Protocol):
    """Optional calibrated prompt-latency provider for analytical replay.

    Implementations replace only the prompt compute estimator. Capacity,
    transfer queues, and cache policy remain owned by this module. Keeping the
    interface at full/cached prompt granularity also ensures that the finite
    replay and its paired residency reference use exactly the same predictor.
    """

    def recompute_seconds(self, tokens: int) -> float:
        """Return full-prompt prefill time for ``tokens`` input tokens."""

    def cached_prefill_seconds(
        self, total_tokens: int, cached_tokens: int
    ) -> float:
        """Return suffix-prefill time with ``cached_tokens`` already present."""

    def metadata(self) -> Mapping[str, Any]:
        """Return JSON-serializable calibration and provenance metadata."""


@dataclass(frozen=True)
class ReplayCall:
    input_tokens: int
    output_tokens: int
    total_sequence_tokens: int
    tool_duration_ns: int
    cache_tokens: int
    effective_reuse_tokens: int
    reusable_allocation_tokens: int
    context_eligible: bool
    cache_eligible: bool
    selected_positive_transition: bool
    reuse_source: str
    return_gap_type: str
    return_gap_source: str
    return_gap_ns: int
    fresh_prompt_tokens: int = 0
    declared_newly_append_tokens: int | None = None


@dataclass(frozen=True)
class ReplaySession:
    session_id: str
    arrival_time_ns: int
    calls: tuple[ReplayCall, ...]


@dataclass(frozen=True)
class CapacityReplayWorkload:
    path: str
    sha256: str
    sessions: tuple[ReplaySession, ...]
    calls: int
    selected_positive_transitions: int
    selected_reuse_eligible_transitions: int
    transitions_excluded_context: int
    max_context_tokens: int | None
    block_size: int

    def metadata_dict(self) -> dict[str, Any]:
        reuse_source_counts: dict[str, int] = {}
        return_gap_type_counts: dict[str, int] = {}
        selected_return_gap_type_counts: dict[str, int] = {}
        eligible_return_gap_type_counts: dict[str, int] = {}
        declared_zero_append_calls = 0
        for session in self.sessions:
            for call in session.calls:
                if call.declared_newly_append_tokens == 0:
                    declared_zero_append_calls += 1
                return_gap_type_counts[call.return_gap_type] = (
                    return_gap_type_counts.get(call.return_gap_type, 0) + 1
                )
                if not call.selected_positive_transition:
                    continue
                selected_return_gap_type_counts[call.return_gap_type] = (
                    selected_return_gap_type_counts.get(
                        call.return_gap_type, 0
                    ) + 1
                )
                if call.effective_reuse_tokens > 0:
                    eligible_return_gap_type_counts[call.return_gap_type] = (
                        eligible_return_gap_type_counts.get(
                            call.return_gap_type, 0
                        ) + 1
                    )
                reuse_source_counts[call.reuse_source] = (
                    reuse_source_counts.get(call.reuse_source, 0) + 1
                )
        return {
            "path": self.path,
            "sha256": self.sha256,
            "sessions": len(self.sessions),
            "calls": self.calls,
            "selected_positive_transitions": (
                self.selected_positive_transitions
            ),
            "selected_reuse_eligible_transitions": (
                self.selected_reuse_eligible_transitions
            ),
            "transitions_excluded_context": (
                self.transitions_excluded_context
            ),
            "max_context_tokens": self.max_context_tokens,
            "context_eligibility_semantics": (
                "input_toks + output_toks <= max_context_tokens; the model "
                "context window includes both prompt and generated tokens"
            ),
            "block_size": self.block_size,
            "declared_zero_append_calls": declared_zero_append_calls,
            "fresh_prompt_token_semantics": (
                "Execution fresh tokens are always max(0, input_toks - "
                "effective_reuse_toks). raw_newly_append_toks is preferred "
                "over newly_append_toks and preserved separately as "
                "declared_newly_append_tokens."
            ),
            "reuse_source_counts": dict(sorted(reuse_source_counts.items())),
            "return_gap_type_counts": dict(
                sorted(return_gap_type_counts.items())
            ),
            "selected_return_gap_type_counts": dict(
                sorted(selected_return_gap_type_counts.items())
            ),
            "eligible_return_gap_type_counts": dict(
                sorted(eligible_return_gap_type_counts.items())
            ),
        }


@dataclass(frozen=True)
class CapacityReplayConfig:
    """Physical capacity and policy inputs.

    Capacity fields are exact bytes. ``hbm_capacity_bytes_per_rank`` is total
    device memory. Model weights and the effective role-specific static reserve
    are subtracted before active and idle KV admission. When a role-specific
    reserve is omitted, ``hbm_static_reserve_bytes_per_rank`` is its fallback.
    CPU capacity is a KV budget, not necessarily the host's physical DRAM total.
    """

    hbm_capacity_bytes_per_rank: int
    cpu_capacity_bytes: int = 2_000_000_000_000
    ssd_capacity_bytes: int = 30_720_000_000_000
    hbm_static_reserve_bytes_per_rank: int = 0
    prefill_hbm_static_reserve_bytes_per_rank: int | None = None
    decode_hbm_static_reserve_bytes_per_rank: int | None = None
    policy: str = "tiered"
    demotion_mode: str = "ttl-and-capacity"
    hbm_ttl_ns: int = 50_000_000
    cpu_ttl_ns: int = 30_000_000_000
    ssd_ttl_ns: int = 3_600_000_000_000
    block_size: int = 16
    prefill_chunk_size: int = 2048
    enable_transfer_queueing: bool = True
    cancel_migration_on_resume: bool = False
    weight_dtype_bytes: int = 2
    pd_disaggregated: bool = True
    pd_link_gbps_per_rank: float = DGX_H100_NVLINK_ONE_WAY_GBPS_PER_GPU
    pd_fixed_latency_us: float = 3.0
    restore_execution_mode: str = "async-pre-admission"
    prompt_compute_scale: float = 1.0
    prompt_compute_scale_provenance: str | None = None

    @property
    def effective_prefill_hbm_static_reserve_bytes_per_rank(self) -> int:
        value = self.prefill_hbm_static_reserve_bytes_per_rank
        return (
            self.hbm_static_reserve_bytes_per_rank
            if value is None
            else value
        )

    @property
    def effective_decode_hbm_static_reserve_bytes_per_rank(self) -> int:
        value = self.decode_hbm_static_reserve_bytes_per_rank
        return (
            self.hbm_static_reserve_bytes_per_rank
            if value is None
            else value
        )

    @property
    def effective_prompt_compute_scale_provenance(self) -> str:
        if self.prompt_compute_scale_provenance is not None:
            return self.prompt_compute_scale_provenance
        if self.prompt_compute_scale == 1.0:
            return (
                "Identity scale on the analytical full-causal prompt "
                "roofline."
            )
        return ""

    def validate(self) -> None:
        capacities = {
            "hbm_capacity_bytes_per_rank": self.hbm_capacity_bytes_per_rank,
            "cpu_capacity_bytes": self.cpu_capacity_bytes,
            "ssd_capacity_bytes": self.ssd_capacity_bytes,
        }
        for name, value in capacities.items():
            if value <= 0:
                raise AnalysisConfigError(f"{name} must be positive")
        if self.policy not in {
            "hbm_lru_recompute", "hbm_ssd_direct", "tiered"
        }:
            raise AnalysisConfigError(
                "policy must be 'hbm_lru_recompute', 'hbm_ssd_direct', or "
                "'tiered'"
            )
        reserve_values = {
            "hbm_static_reserve_bytes_per_rank": (
                self.hbm_static_reserve_bytes_per_rank
            ),
            "prefill_hbm_static_reserve_bytes_per_rank": (
                self.effective_prefill_hbm_static_reserve_bytes_per_rank
            ),
            "decode_hbm_static_reserve_bytes_per_rank": (
                self.effective_decode_hbm_static_reserve_bytes_per_rank
            ),
        }
        if any(value < 0 for value in reserve_values.values()):
            raise AnalysisConfigError(
                "HBM static reserve values cannot be negative"
            )
        if (
            not self.pd_disaggregated
            and (
                self.prefill_hbm_static_reserve_bytes_per_rank is not None
                or self.decode_hbm_static_reserve_bytes_per_rank is not None
            )
        ):
            raise AnalysisConfigError(
                "role-specific HBM reserves require P/D disaggregation"
            )
        if min(self.hbm_ttl_ns, self.cpu_ttl_ns, self.ssd_ttl_ns) < 0:
            raise AnalysisConfigError("tier TTLs cannot be negative")
        if self.demotion_mode not in {"ttl-and-capacity", "capacity-only"}:
            raise AnalysisConfigError(
                "demotion_mode must be 'ttl-and-capacity' or 'capacity-only'"
            )
        if self.block_size <= 0 or self.prefill_chunk_size <= 0:
            raise AnalysisConfigError(
                "block_size and prefill_chunk_size must be positive"
            )
        if self.weight_dtype_bytes <= 0:
            raise AnalysisConfigError("weight_dtype_bytes must be positive")
        if self.pd_link_gbps_per_rank <= 0 or self.pd_fixed_latency_us < 0:
            raise AnalysisConfigError("invalid P/D link specification")
        if (
            not math.isfinite(self.prompt_compute_scale)
            or self.prompt_compute_scale <= 0
        ):
            raise AnalysisConfigError(
                "prompt_compute_scale must be finite and positive"
            )
        if (
            self.prompt_compute_scale_provenance is not None
            and not self.prompt_compute_scale_provenance.strip()
        ):
            raise AnalysisConfigError(
                "prompt_compute_scale_provenance cannot be empty"
            )
        if (
            self.prompt_compute_scale != 1.0
            and not self.effective_prompt_compute_scale_provenance
        ):
            raise AnalysisConfigError(
                "a non-identity prompt_compute_scale requires provenance"
            )
        if self.restore_execution_mode not in {
            "async-pre-admission",
            "async-decode-join",
            "serial-before-prefill",
        }:
            raise AnalysisConfigError(
                "restore_execution_mode must be 'async-pre-admission', "
                "'async-decode-join', or 'serial-before-prefill'"
            )
        if self.enable_transfer_queueing and self.cancel_migration_on_resume:
            raise AnalysisConfigError(
                "cancel_migration_on_resume cannot be combined with scalar "
                "FCFS queueing because queued intervals cannot be reflowed"
            )


@dataclass
class _Entry:
    session_id: str
    cache_tokens: int
    cluster_bytes: int
    per_rank_bytes: int
    tier: str
    last_access_ns: int
    generation: int = 0
    available_ns: int = 0
    move_reason: str = "completion"
    drop_reason: str = ""
    transit_source_tier: str = ""


@dataclass
class _Active:
    session_id: str
    per_rank_bytes: int
    completion_ns: int


@dataclass
class _PdCallJoinState:
    session_id: str
    call_index: int
    admission_sequence: int
    source: str
    source_reason: str
    logical_ready_ns: int
    active_per_rank: int
    full_decode_per_rank: int
    restore_cluster_bytes: int
    restore_rank_bytes: int
    full_compute_seconds: float
    cached_compute_seconds: float
    selected_compute_seconds: float
    compute_ns: int
    overlap_compute_ns: int
    post_restore_compute_ns: int
    prefill_admitted: bool = False
    prefill_start_ns: int = 0
    overlap_prefill_finish_ns: int = 0
    decode_admitted: bool = False
    decode_admission_ns: int = 0
    decode_reservation_bytes: int = 0
    decode_prefix_ready: bool = False
    lower_restore_scheduled: bool = False
    lower_restore_issue_ns: int = 0
    lower_restore_finish_ns: int = 0
    d2p_scheduled: bool = False
    d2p_issue_ns: int = 0
    restore_finish_ns: int = 0
    join_scheduled: bool = False
    join_completion_ns: int = 0
    speculative_compute_seconds: float = 0.0
    source_wait_accounted_until_ns: int = 0
    accounting_recorded: bool = False


class _TransferQueue:
    """Deterministic non-preemptive gang-FCFS resource queue."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.busy_until: dict[str, int] = {}
        self.total_queue_wait_ns = 0
        self.total_service_ns = 0
        self.jobs = 0
        self.jobs_by_kind: dict[str, int] = {}
        self.queue_wait_by_kind: dict[str, int] = {}
        self.service_by_kind: dict[str, int] = {}
        self.bytes_by_kind: dict[str, int] = {}

    def schedule(
        self,
        arrival_ns: int,
        service_seconds: float,
        resources: tuple[str, ...],
        kind: str,
        byte_count: int = 0,
    ) -> tuple[int, int]:
        service_ns = max(0, int(math.ceil(service_seconds * 1e9)))
        if self.enabled:
            start_ns = max(
                [arrival_ns]
                + [self.busy_until.get(resource, 0) for resource in resources]
            )
        else:
            start_ns = arrival_ns
        finish_ns = start_ns + service_ns
        if self.enabled:
            for resource in resources:
                self.busy_until[resource] = finish_ns
        wait_ns = start_ns - arrival_ns
        self.total_queue_wait_ns += wait_ns
        self.total_service_ns += service_ns
        self.jobs += 1
        self.jobs_by_kind[kind] = self.jobs_by_kind.get(kind, 0) + 1
        self.queue_wait_by_kind[kind] = (
            self.queue_wait_by_kind.get(kind, 0) + wait_ns
        )
        self.service_by_kind[kind] = (
            self.service_by_kind.get(kind, 0) + service_ns
        )
        self.bytes_by_kind[kind] = (
            self.bytes_by_kind.get(kind, 0) + byte_count
        )
        return start_ns, finish_ns

    def report(self) -> dict[str, Any]:
        return {
            "discipline": "nonpreemptive_gang_fcfs",
            "enabled": self.enabled,
            "jobs": self.jobs,
            "jobs_by_kind": dict(sorted(self.jobs_by_kind.items())),
            "aggregate_queue_wait_seconds": self.total_queue_wait_ns / 1e9,
            "aggregate_service_seconds": self.total_service_ns / 1e9,
            "queue_wait_seconds_by_kind": {
                key: value / 1e9
                for key, value in sorted(self.queue_wait_by_kind.items())
            },
            "service_seconds_by_kind": {
                key: value / 1e9
                for key, value in sorted(self.service_by_kind.items())
            },
            "bytes_by_kind": dict(sorted(self.bytes_by_kind.items())),
            "resources": sorted(self.busy_until),
        }


def _longest_common_prefix(left: list[Any], right: list[Any]) -> int:
    count = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        count += 1
    return count


def _interval_union_ns(intervals: list[tuple[int, int]]) -> int:
    valid = sorted((start, end) for start, end in intervals if end > start)
    if not valid:
        return 0
    total = 0
    current_start, current_end = valid[0]
    for start, end in valid[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _quantile(values: list[float], fraction: float) -> float | None:
    """Return a deterministic linearly interpolated sample quantile."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(1.0, max(0.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _call_reuse(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    block_size: int,
) -> tuple[int, int, str]:
    previous_input = int(previous.get("input_toks", 0) or 0)
    previous_output = int(previous.get("output_toks", 0) or 0)
    current_input = int(current.get("input_toks", 0) or 0)
    previous_cache = max(0, previous_input + previous_output - 1)
    input_ids = previous.get("input_tok_ids")
    output_ids = previous.get("output_tok_ids")
    current_ids = current.get("input_tok_ids")
    if (
        isinstance(input_ids, list)
        and isinstance(output_ids, list)
        and isinstance(current_ids, list)
        and input_ids
        and current_ids
    ):
        previous_ids = list(input_ids)[:previous_input]
        if len(previous_ids) == previous_input:
            previous_ids += list(output_ids)[:max(0, previous_output - 1)]
        reuse = min(
            _longest_common_prefix(previous_ids, list(current_ids)),
            previous_cache,
            max(0, current_input - 1),
        )
        source = "token_ids_exact"
    else:
        try:
            declared = int(current.get("prefix_reuse_toks", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise AnalysisConfigError("invalid prefix_reuse_toks") from exc
        if declared < 0:
            raise AnalysisConfigError("prefix_reuse_toks cannot be negative")
        reuse = min(declared, previous_cache, max(0, current_input - 1))
        source = "explicit_" + str(
            current.get("prefix_reuse_source") or "reported"
        )
    allocation = (
        math.ceil(reuse / block_size) * block_size if reuse else 0
    )
    return reuse, allocation, source


def load_capacity_replay_workload(
    path: Path,
    block_size: int = 16,
    max_context_tokens: int | None = None,
) -> CapacityReplayWorkload:
    """Load complete sessions while preserving arrival time and excluded gaps."""

    if block_size <= 0:
        raise AnalysisConfigError("block_size must be positive")
    if max_context_tokens is not None and max_context_tokens <= 0:
        raise AnalysisConfigError("max_context_tokens must be positive")
    raw = path.read_bytes()
    sessions: list[ReplaySession] = []
    call_count = 0
    selected = 0
    eligible = 0
    context_excluded = 0
    previous_arrival = -1
    seen_ids: set[str] = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalysisConfigError(
                f"{path}:{line_number}: invalid JSON: {exc}"
            ) from exc
        raw_calls = record.get("sub_requests")
        if not isinstance(raw_calls, list) or not raw_calls:
            continue
        session_id = str(record.get("session_id", f"line-{line_number}"))
        if session_id in seen_ids:
            raise AnalysisConfigError(f"duplicate session_id {session_id!r}")
        seen_ids.add(session_id)
        try:
            arrival_ns = int(record.get("arrival_time_ns", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise AnalysisConfigError(
                f"{path}:{line_number}: invalid arrival_time_ns"
            ) from exc
        if arrival_ns < 0 or arrival_ns < previous_arrival:
            raise AnalysisConfigError(
                f"{path}:{line_number}: arrivals must be nonnegative and sorted"
            )
        previous_arrival = arrival_ns
        calls: list[ReplayCall] = []
        context_eligibility: list[bool] = []
        cache_eligibility: list[bool] = []
        parsed_counts: list[
            tuple[int, int, int, int, int, int | None]
        ] = []
        for index, raw_call in enumerate(raw_calls):
            try:
                input_tokens = int(raw_call.get("input_toks", 0) or 0)
                output_tokens = int(raw_call.get("output_toks", 0) or 0)
                tool_ns = int(raw_call.get("tool_duration_ns", 0) or 0)
                raw_fresh = raw_call.get(
                    "raw_newly_append_toks",
                    raw_call.get("newly_append_toks"),
                )
                declared_fresh = (
                    None if raw_fresh is None else int(raw_fresh)
                )
            except (TypeError, ValueError) as exc:
                raise AnalysisConfigError(
                    f"{path}:{line_number}: invalid call {index}"
                ) from exc
            if (
                input_tokens <= 0
                or output_tokens < 0
                or tool_ns < 0
                or (declared_fresh is not None and declared_fresh < 0)
            ):
                raise AnalysisConfigError(
                    f"{path}:{line_number}: invalid call {index}"
                )
            cache_tokens = max(0, input_tokens + output_tokens - 1)
            total_sequence_tokens = input_tokens + output_tokens
            context_eligible = (
                max_context_tokens is None
                or total_sequence_tokens <= max_context_tokens
            )
            cache_eligible = (
                context_eligible
                and (
                    max_context_tokens is None
                    or cache_tokens <= max_context_tokens
                )
            )
            context_eligibility.append(context_eligible)
            cache_eligibility.append(cache_eligible)
            parsed_counts.append(
                (
                    input_tokens,
                    output_tokens,
                    tool_ns,
                    cache_tokens,
                    total_sequence_tokens,
                    declared_fresh,
                )
            )
        for index, raw_call in enumerate(raw_calls):
            (
                input_tokens,
                output_tokens,
                tool_ns,
                cache_tokens,
                total_sequence_tokens,
                declared_fresh,
            ) = parsed_counts[index]
            reuse = 0
            allocation = 0
            reuse_source = "first_call"
            positive_transition = False
            return_gap_type = "session_start"
            return_gap_source = "session_start"
            return_gap_ns = 0
            if index:
                reuse, allocation, reuse_source = _call_reuse(
                    raw_calls[index - 1], raw_call, block_size
                )
                previous_tool = parsed_counts[index - 1][2]
                return_gap_ns = previous_tool
                previous_call = raw_calls[index - 1]
                raw_gap_type = str(
                    previous_call.get("inter_turn_gap_type") or "unknown"
                ).strip().lower()
                return_gap_type = (
                    raw_gap_type
                    if raw_gap_type in {"human", "tool", "mixed", "unknown"}
                    else "unknown"
                )
                return_gap_source = str(
                    previous_call.get("tool_wait_source") or "unknown"
                )
                positive_transition = (
                    previous_tool > 0
                    and cache_eligibility[index - 1]
                    and context_eligibility[index]
                )
                if (
                    not cache_eligibility[index - 1]
                    or not context_eligibility[index]
                ):
                    reuse = 0
                    allocation = 0
                if previous_tool > 0 and not positive_transition:
                    context_excluded += 1
                if positive_transition:
                    selected += 1
                    if reuse > 0:
                        eligible += 1
            fresh_prompt_tokens = max(0, input_tokens - reuse)
            calls.append(
                ReplayCall(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_sequence_tokens=total_sequence_tokens,
                    tool_duration_ns=tool_ns,
                    cache_tokens=cache_tokens,
                    effective_reuse_tokens=reuse,
                    fresh_prompt_tokens=fresh_prompt_tokens,
                    reusable_allocation_tokens=allocation,
                    context_eligible=context_eligibility[index],
                    cache_eligible=cache_eligibility[index],
                    selected_positive_transition=positive_transition,
                    reuse_source=reuse_source,
                    return_gap_type=return_gap_type,
                    return_gap_source=return_gap_source,
                    return_gap_ns=return_gap_ns,
                    declared_newly_append_tokens=declared_fresh,
                )
            )
        call_count += len(calls)
        sessions.append(
            ReplaySession(session_id, arrival_ns, tuple(calls))
        )
    if not sessions:
        raise AnalysisConfigError(f"{path} contains no agentic sessions")
    return CapacityReplayWorkload(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        sessions=tuple(sessions),
        calls=call_count,
        selected_positive_transitions=selected,
        selected_reuse_eligible_transitions=eligible,
        transitions_excluded_context=context_excluded,
        max_context_tokens=max_context_tokens,
        block_size=block_size,
    )


def estimate_model_weight_bytes_per_rank(
    model: ModelShape,
    tp_size: int,
    dtype_bytes: int = 2,
    kv: KvLayout | None = None,
) -> int:
    """Estimate resident BF16/FP16 model bytes for one tensor-parallel rank."""

    if tp_size <= 0 or dtype_bytes <= 0:
        raise AnalysisConfigError("tp_size and dtype_bytes must be positive")
    layout = kv or kv_layout(model, tp_size, dtype_bytes)
    config: dict[str, Any] = {}
    if model.config_path:
        with Path(model.config_path).open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    vocab_size = int(config.get("vocab_size", 0) or 0)
    tied = bool(config.get("tie_word_embeddings", False))
    embedding_elements = vocab_size * model.hidden_size * (1 if tied else 2)
    q_per_rank = model.q_dim / tp_size
    physical_kv_per_rank = layout.kv_heads_per_rank * model.head_dim
    attention_elements_per_layer_rank = (
        model.hidden_size * (q_per_rank + 2 * physical_kv_per_rank)
        + model.q_dim * model.hidden_size / tp_size
    )
    if model.is_moe:
        ffn_elements_per_layer_rank = (
            3
            * model.hidden_size
            * model.moe_intermediate_size
            * model.num_experts
            / tp_size
            + model.hidden_size * model.num_experts
        )
    else:
        ffn_elements_per_layer_rank = (
            3 * model.hidden_size * model.intermediate_size / tp_size
        )
    norm_elements_per_rank = (
        (2 * model.num_hidden_layers + 1) * model.hidden_size
    )
    total_elements_rank = (
        embedding_elements / tp_size
        + model.num_hidden_layers
        * (attention_elements_per_layer_rank + ffn_elements_per_layer_rank)
        + norm_elements_per_rank
    )
    return int(math.ceil(total_elements_rank * dtype_bytes))


class _CapacityReplay:
    _MIGRATION_COMPLETE = 0
    _PROMPT_COMPLETE = 1
    _CALL_COMPLETE = 2
    _TTL = 3
    _CALL_READY = 4

    def __init__(
        self,
        workload: CapacityReplayWorkload,
        model: ModelShape,
        hardware: HardwareSpec,
        tp_size: int,
        kv_dtype_bytes: int,
        config: CapacityReplayConfig,
        prompt_compute_model: PromptComputeModel | None = None,
    ) -> None:
        config.validate()
        hardware.validate()
        self.workload = workload
        self.model = model
        self.hardware = hardware
        self.tp_size = tp_size
        self.kv_dtype_bytes = kv_dtype_bytes
        self.config = config
        self.prompt_compute_model = prompt_compute_model
        if workload.block_size != config.block_size:
            raise AnalysisConfigError(
                "workload and replay block_size must match: "
                f"{workload.block_size} != {config.block_size}"
            )
        self.layout = kv_layout(model, tp_size, kv_dtype_bytes)
        self.weight_bytes_per_rank = estimate_model_weight_bytes_per_rank(
            model, tp_size, config.weight_dtype_bytes, self.layout
        )
        self.prefill_hbm_kv_budget = (
            config.hbm_capacity_bytes_per_rank
            - self.weight_bytes_per_rank
            - config.effective_prefill_hbm_static_reserve_bytes_per_rank
        )
        self.decode_hbm_kv_budget = (
            config.hbm_capacity_bytes_per_rank
            - self.weight_bytes_per_rank
            - config.effective_decode_hbm_static_reserve_bytes_per_rank
        )
        # Colocated runs have one pool. Role-specific reserves are meaningful
        # only with P/D disaggregation, so retain the legacy common reserve.
        self.hbm_kv_budget = (
            config.hbm_capacity_bytes_per_rank
            - self.weight_bytes_per_rank
            - config.hbm_static_reserve_bytes_per_rank
        )
        required_budgets = (
            (self.prefill_hbm_kv_budget, self.decode_hbm_kv_budget)
            if config.pd_disaggregated
            else (self.hbm_kv_budget,)
        )
        if min(required_budgets) <= 0:
            raise AnalysisConfigError(
                "model weights plus static reserve exhaust per-rank HBM"
            )
        self.sessions = {
            session.session_id: session for session in workload.sessions
        }
        self.events: list[tuple[int, int, int, str, tuple[Any, ...]]] = []
        self.event_sequence = 0
        self.lru_sequence = 0
        self.entries: dict[str, _Entry] = {}
        self.session_generations: dict[str, int] = {}
        self.active: dict[str, _Active] = {}
        self.active_bytes_per_rank = 0
        self.decode_reserved_bytes_per_rank = 0
        self.active_heap: list[tuple[int, str]] = []
        self.lru: dict[str, list[tuple[int, int, str, int]]] = {
            "hbm": [], "cpu": [], "ssd": []
        }
        self.transit_destination_heaps: dict[
            str, list[tuple[int, str, int]]
        ] = {"cpu": [], "ssd": []}
        self.source_release_heaps: dict[
            str, list[tuple[int, str, int]]
        ] = {"hbm": [], "cpu": [], "ssd": []}
        self.used = {"hbm": 0, "cpu": 0, "ssd": 0}
        self.peaks = {"hbm": 0, "cpu": 0, "ssd": 0, "active_hbm": 0}
        self.peak_decode_reserved_bytes_per_rank = 0
        self.queue = _TransferQueue(config.enable_transfer_queueing)
        self.source_counts = {
            "hbm": 0, "cpu": 0, "ssd": 0, "recompute": 0,
            "no_reuse": 0,
        }
        self.source_tokens = {
            "hbm": 0, "cpu": 0, "ssd": 0, "recompute": 0,
        }
        self.source_counts_by_return_gap_type: dict[
            str, dict[str, int]
        ] = {}
        self.source_tokens_by_return_gap_type: dict[
            str, dict[str, int]
        ] = {}
        self.ssd_source_reasons: dict[str, int] = {}
        self.recompute_reasons: dict[str, int] = {}
        self.policy_actions: dict[str, int] = {}
        self.selected_seen = 0
        self.eligible_seen = 0
        self.total_reusable_tokens = 0
        self.recompute_tokens = 0
        self.recompute_seconds = 0.0
        self.prompt_compute_seconds = 0.0
        self.full_prompt_reference_seconds = 0.0
        self.restore_stall_ns = 0
        self.restore_queue_wait_ns = 0
        self.resume_inflight_migration_wait_ns = 0
        self.no_reuse_inflight_migration_wait_ns = 0
        self.pd_d2p_stall_ns = 0
        self.pd_p2d_handoff_ns = 0
        self.pd_p2d_queue_wait_ns = 0
        self.pd_d2p_bytes = 0
        self.pd_p2d_bytes = 0
        self.pd_call_sources: dict[str, str] = {}
        self.pd_handoff_reservations: dict[str, tuple[int, int]] = {}
        self.pd_pending_compute: dict[str, _PdCallJoinState] = {}
        self.pd_call_admission_sequence = 0
        self.pd_speculative_prefill_wasted_seconds = 0.0
        self.foreground_migration_intervals: list[tuple[int, int]] = []
        self.foreground_kv_transfer_intervals: list[tuple[int, int]] = []
        self.decode_waiters: deque[tuple[str, int, int]] = deque()
        self.decode_waiter_sessions: set[str] = set()
        self.decode_wakeup_generation = 0
        self.decode_wakeup_ns: int | None = None
        self.decode_retrying_session: str | None = None
        self.prefill_waiters: deque[tuple[str, int]] = deque()
        self.prefill_waiter_sessions: set[str] = set()
        self.prefill_waiter_since_ns: dict[str, int] = {}
        self.prefill_wakeup_generation = 0
        self.prefill_wakeup_ns: int | None = None
        self.prefill_retrying_session: str | None = None
        # Lower-tier resume needs free space in the decode HBM pool before
        # its CPU/SSD -> D copy can be issued.  Keep this admission queue
        # separate from ``prefill_waiters``: under P/D disaggregation the two
        # HBM pools are independent, so sharing a queue would introduce
        # artificial cross-pool head-of-line blocking.
        self.decode_restore_waiters: deque[tuple[str, int]] = deque()
        self.decode_restore_waiter_sessions: set[str] = set()
        self.decode_restore_waiter_since_ns: dict[str, int] = {}
        self.decode_restore_wakeup_generation = 0
        self.decode_restore_wakeup_ns: int | None = None
        self.decode_restore_retrying_session: str | None = None
        self.decode_restore_source_pins: set[str] = set()
        self.decode_restore_enqueue_count = 0
        self.decode_restore_capacity_retry_count = 0
        self.decode_restore_fcfs_deferral_count = 0
        self.decode_restore_wakeup_count = 0
        self.decode_restore_wakeup_event_count = 0
        self.decode_restore_max_depth = 0
        self.decode_restore_max_source_pins = 0
        self.decode_restore_source_ttl_deferral_count = 0
        self.decode_restore_capacity_block_ns = 0
        self.context_infeasible_calls = 0
        self.hbm_capacity_block_ns = 0
        self.event_horizon_ns = 0
        self.request_trace_end_ns = 0
        # Kept in memory for paired finite-vs-oracle analysis. These maps are
        # deliberately not serialized in the ordinary report, which keeps a
        # full TraceLab result compact.
        self.call_logical_ready_ns: dict[tuple[str, int], int] = {}
        self.call_completion_ns: dict[tuple[str, int], int] = {}
        self.call_sources: dict[tuple[str, int], str] = {}
        self.raw_restore_elapsed_ns = 0
        self.restore_hidden_by_prefill_ns = 0
        self.exposed_restore_barrier_ns = 0
        self.restore_other_concurrent_or_admission_ns = 0
        self.restore_timing_by_source: dict[str, dict[str, int]] = {}
        self.restore_timing_by_return_gap_type: dict[str, dict[str, int]] = {}
        self.restore_timing_by_gap_and_source: dict[
            str, dict[str, dict[str, int]]
        ] = {}
        self.raw_restore_intervals: list[tuple[int, int]] = []
        self.exposed_restore_barrier_intervals: list[tuple[int, int]] = []
        self.raw_restore_intervals_by_gap: dict[
            str, list[tuple[int, int]]
        ] = {}
        self.exposed_restore_intervals_by_gap: dict[
            str, list[tuple[int, int]]
        ] = {}

    def _push(
        self, time_ns: int, priority: int, kind: str, *payload: Any
    ) -> None:
        self.event_sequence += 1
        heapq.heappush(
            self.events,
            (time_ns, priority, self.event_sequence, kind, payload),
        )

    def _usage_contributions(self, entry: _Entry, tier: str) -> dict[str, int]:
        def tier_bytes(name: str) -> int:
            return entry.per_rank_bytes if name == "hbm" else entry.cluster_bytes

        if tier in {"hbm", "cpu", "ssd"}:
            return {tier: tier_bytes(tier)}
        if tier == "pinned_hbm":
            return {"hbm": tier_bytes("hbm")}
        if tier == "restore_cpu":
            return {"cpu": tier_bytes("cpu")}
        if tier == "restore_ssd":
            return {"ssd": tier_bytes("ssd")}
        if tier == "transit_cpu":
            # HBM -> CPU copies retain the source until atomic commit while
            # reserving destination capacity.
            return {"hbm": tier_bytes("hbm"), "cpu": tier_bytes("cpu")}
        if tier == "transit_ssd":
            source = entry.transit_source_tier
            if source not in {"hbm", "cpu"}:
                raise AssertionError("SSD transit is missing its source tier")
            return {
                source: tier_bytes(source),
                "ssd": tier_bytes("ssd"),
            }
        return {}

    def _set_tier(self, entry: _Entry, tier: str) -> None:
        old_contributions = self._usage_contributions(entry, entry.tier)
        for key, byte_count in old_contributions.items():
            self.used[key] -= byte_count
        entry.tier = tier
        new_contributions = self._usage_contributions(entry, tier)
        for key, byte_count in new_contributions.items():
            self.used[key] += byte_count
            self.peaks[key] = max(self.peaks[key], self.used[key])
        self._check_capacity()

    def _check_capacity(self) -> None:
        if self.config.pd_disaggregated:
            if (
                self.used["hbm"] + self.decode_reserved_bytes_per_rank
                > self.decode_hbm_kv_budget
            ):
                raise AssertionError("decode HBM capacity violation")
            if self._active_bytes() > self.prefill_hbm_kv_budget:
                raise AssertionError("prefill HBM capacity violation")
        elif self.used["hbm"] + self._active_bytes() > self.hbm_kv_budget:
            raise AssertionError("single-pool HBM capacity violation")
        if self.used["cpu"] > self.config.cpu_capacity_bytes:
            raise AssertionError("CPU byte conservation/capacity violation")
        if self.used["ssd"] > self.config.ssd_capacity_bytes:
            raise AssertionError("SSD byte conservation/capacity violation")
        if min(self.used.values()) < 0:
            raise AssertionError("negative tier occupancy")

    def _active_bytes(self) -> int:
        return self.active_bytes_per_rank

    def _reserve_decode_hbm(self, byte_count: int) -> None:
        self.decode_reserved_bytes_per_rank += byte_count
        self.peak_decode_reserved_bytes_per_rank = max(
            self.peak_decode_reserved_bytes_per_rank,
            self.decode_reserved_bytes_per_rank,
        )
        self.peaks["hbm"] = max(
            self.peaks["hbm"],
            self.used["hbm"] + self.decode_reserved_bytes_per_rank,
        )
        self._check_capacity()

    def _release_decode_hbm(self, byte_count: int) -> None:
        self.decode_reserved_bytes_per_rank -= byte_count
        if self.decode_reserved_bytes_per_rank < 0:
            raise AssertionError("negative decode HBM reservation")
        self._check_capacity()

    def _record_active_peak(self) -> None:
        active_bytes = self._active_bytes()
        self.peaks["active_hbm"] = max(
            self.peaks["active_hbm"], active_bytes
        )
        if self.config.pd_disaggregated:
            self.peaks["hbm"] = max(
                self.peaks["hbm"],
                self.used["hbm"] + self.decode_reserved_bytes_per_rank,
            )
        else:
            self.peaks["hbm"] = max(
                self.peaks["hbm"], self.used["hbm"] + active_bytes
            )
        self._check_capacity()

    def _push_lru(self, entry: _Entry) -> None:
        if entry.tier not in self.lru:
            return
        self.lru_sequence += 1
        heapq.heappush(
            self.lru[entry.tier],
            (
                entry.last_access_ns,
                self.lru_sequence,
                entry.session_id,
                entry.generation,
            ),
        )

    def _bump_generation(self, entry: _Entry) -> int:
        generation = max(
            entry.generation,
            self.session_generations.get(entry.session_id, 0),
        ) + 1
        entry.generation = generation
        self.session_generations[entry.session_id] = generation
        return generation

    def _lru_entry(self, tier: str, exclude: str | None = None) -> _Entry | None:
        skipped: list[tuple[int, int, str, int]] = []
        heap = self.lru[tier]
        result = None
        while heap:
            item = heapq.heappop(heap)
            _, _, session_id, generation = item
            entry = self.entries.get(session_id)
            if (
                entry is None
                or entry.tier != tier
                or entry.generation != generation
            ):
                continue
            if (
                session_id == exclude
                or session_id in self.decode_restore_source_pins
                or session_id == self.decode_restore_retrying_session
            ):
                skipped.append(item)
                continue
            result = entry
            break
        for item in skipped:
            heapq.heappush(heap, item)
        return result

    def _earliest_active_completion(
        self,
        exclude: str | None = None,
        after_ns: int | None = None,
    ) -> int | None:
        skipped: list[tuple[int, str]] = []
        result = None
        while self.active_heap:
            completion_ns, session_id = heapq.heappop(self.active_heap)
            active = self.active.get(session_id)
            if active is None or active.completion_ns != completion_ns:
                continue
            if after_ns is not None and completion_ns <= after_ns:
                # A decode-capacity waiter can remain active after its
                # advertised completion time while the serialized wake-up
                # queue services older waiters. Its next timestamp will be
                # pushed when that waiter is retried, so this heap item no
                # longer predicts capacity progress.
                continue
            if session_id == exclude:
                skipped.append((completion_ns, session_id))
                continue
            result = completion_ns
            skipped.append((completion_ns, session_id))
            break
        for item in skipped:
            heapq.heappush(self.active_heap, item)
        return result

    def _next_active_progress_ns(
        self, now_ns: int, exclude: str | None = None
    ) -> int | None:
        skipped: list[tuple[int, str]] = []
        active_progress = None
        while self.active_heap:
            completion_ns, session_id = heapq.heappop(self.active_heap)
            active = self.active.get(session_id)
            if active is None or active.completion_ns != completion_ns:
                continue
            if session_id == exclude:
                skipped.append((completion_ns, session_id))
                continue
            if completion_ns < now_ns:
                # Serialized decode waiters can remain active past their old
                # advertised epoch. Their replacement epoch is pushed when
                # the waiter is serviced, so this record is stale.
                continue
            skipped.append((completion_ns, session_id))
            active_progress = (
                now_ns + 1 if completion_ns == now_ns else completion_ns
            )
            break
        for item in skipped:
            heapq.heappush(self.active_heap, item)

        wakeup_progress = (
            self.decode_wakeup_ns
            if self.decode_wakeup_ns is not None
            and self.decode_wakeup_ns > now_ns
            else None
        )
        candidates = [
            candidate
            for candidate in (active_progress, wakeup_progress)
            if candidate is not None
        ]
        if candidates:
            return min(candidates)
        # Completions strictly before ``now_ns`` belong to serialized
        # decode-capacity waiters. They do not predict progress and must not
        # cause one-nanosecond polling while a real DMA completion is pending.
        return None

    def _earliest_transit_completion(self, tier: str) -> int | None:
        heap = self.transit_destination_heaps[tier]
        transit = "transit_" + tier
        while heap:
            available_ns, session_id, generation = heap[0]
            entry = self.entries.get(session_id)
            if (
                entry is not None
                and entry.generation == generation
                and entry.tier == transit
                and entry.available_ns == available_ns
            ):
                return available_ns
            heapq.heappop(heap)
        return None

    def _earliest_source_release(self, tier: str) -> int | None:
        heap = self.source_release_heaps[tier]
        while heap:
            available_ns, session_id, generation = heap[0]
            entry = self.entries.get(session_id)
            valid = (
                entry is not None
                and entry.generation == generation
                and entry.available_ns == available_ns
                and (
                    (tier == "hbm" and entry.tier == "transit_cpu")
                    or (
                        entry.tier == "transit_ssd"
                        and entry.transit_source_tier == tier
                    )
                    or entry.tier == "restore_" + tier
                )
            )
            if valid:
                return available_ns
            heapq.heappop(heap)
        return None

    def _track_transit(self, entry: _Entry, destination: str) -> None:
        heapq.heappush(
            self.transit_destination_heaps[destination],
            (entry.available_ns, entry.session_id, entry.generation),
        )
        source = entry.transit_source_tier
        heapq.heappush(
            self.source_release_heaps[source],
            (entry.available_ns, entry.session_id, entry.generation),
        )

    def _track_restore(self, entry: _Entry, source: str) -> None:
        heapq.heappush(
            self.source_release_heaps[source],
            (entry.available_ns, entry.session_id, entry.generation),
        )

    def _drop(self, entry: _Entry, reason: str) -> None:
        self._bump_generation(entry)
        self._set_tier(entry, "dropped")
        entry.drop_reason = reason
        entry.move_reason = reason
        self.policy_actions[reason] = self.policy_actions.get(reason, 0) + 1

    def _ensure_ssd_space(
        self, byte_count: int, now_ns: int, exclude: str | None = None
    ) -> int | None:
        if byte_count > self.config.ssd_capacity_bytes:
            return -1
        while self.used["ssd"] + byte_count > self.config.ssd_capacity_bytes:
            victim = self._lru_entry("ssd", exclude)
            if victim is None:
                candidates = [
                    candidate
                    for candidate in (
                        self._earliest_source_release("ssd"),
                        # A destination commit does not free bytes, but it
                        # makes the object resident and therefore evictable on
                        # the retry.
                        self._earliest_transit_completion("ssd"),
                    )
                    if candidate is not None and candidate >= now_ns
                ]
                retry = min(candidates) if candidates else None
                return retry if retry is not None and retry > now_ns else -1
            self._drop(victim, "ssd_capacity")
        return None

    def _cpu_to_ssd(self, entry: _Entry, now_ns: int, reason: str) -> int | None:
        retry = self._ensure_ssd_space(
            entry.cluster_bytes, now_ns, entry.session_id
        )
        if retry is not None:
            if retry == -1:
                self._drop(entry, "ssd_object_oversize")
                return None
            return retry
        service = self.hardware.ssd.fixed_latency_us * 1e-6 + max(
            entry.cluster_bytes
            / (self.hardware.ssd.write_gbps_aggregate * 1e9),
            entry.cluster_bytes
            / (self.hardware.cpu.dram_read_gbps_aggregate * 1e9),
        )
        _, finish = self.queue.schedule(
            now_ns,
            service,
            ("cpu_dram", "ssd_write"),
            "cpu_to_ssd",
            entry.cluster_bytes,
        )
        generation = self._bump_generation(entry)
        entry.available_ns = finish
        entry.move_reason = reason
        entry.transit_source_tier = "cpu"
        self._set_tier(entry, "transit_ssd")
        self._track_transit(entry, "ssd")
        self.policy_actions[reason] = self.policy_actions.get(reason, 0) + 1
        self._push(
            finish,
            self._MIGRATION_COMPLETE,
            "migration_complete",
            entry.session_id,
            generation,
            "ssd",
        )
        return None

    def _ensure_cpu_space(
        self, byte_count: int, now_ns: int, exclude: str | None = None
    ) -> int | None:
        if byte_count > self.config.cpu_capacity_bytes:
            return -1
        while self.used["cpu"] + byte_count > self.config.cpu_capacity_bytes:
            victim = self._lru_entry("cpu", exclude)
            if victim is None:
                candidates = [
                    candidate
                    for candidate in (
                        self._earliest_source_release("cpu"),
                        self._earliest_transit_completion("cpu"),
                    )
                    if candidate is not None and candidate > now_ns
                ]
                retry = min(candidates) if candidates else None
                return retry if retry is not None and retry > now_ns else -1
            retry = self._cpu_to_ssd(victim, now_ns, "cpu_capacity")
            if retry is not None:
                self._push_lru(victim)
                return retry
            if victim.tier == "dropped":
                # An object larger than the SSD pool was discarded
                # immediately; CPU capacity is already released.
                continue
            if victim.tier == "transit_ssd":
                # Source-retained atomic commit does not release CPU capacity
                # until this transfer commits.
                return victim.available_ns
            raise AssertionError("CPU victim neither moved nor dropped")
        return None

    def _hbm_to_ssd_oversize(
        self, entry: _Entry, now_ns: int, reason: str
    ) -> int | None:
        retry = self._ensure_ssd_space(
            entry.cluster_bytes, now_ns, entry.session_id
        )
        if retry is not None:
            if retry == -1:
                self._drop(entry, "ssd_object_oversize")
                return None
            return retry
        service = ssd_transfer_seconds(
            entry.cluster_bytes,
            entry.per_rank_bytes,
            self.hardware,
            "out",
        )
        _, finish = self.queue.schedule(
            now_ns,
            service,
            ("pcie_out", "cpu_dram", "ssd_write"),
            "hbm_to_ssd_oversize",
            entry.cluster_bytes,
        )
        generation = self._bump_generation(entry)
        entry.available_ns = finish
        entry.move_reason = reason
        entry.transit_source_tier = "hbm"
        self._set_tier(entry, "transit_ssd")
        self._track_transit(entry, "ssd")
        self.policy_actions[reason] = self.policy_actions.get(reason, 0) + 1
        self._push(
            finish,
            self._MIGRATION_COMPLETE,
            "migration_complete",
            entry.session_id,
            generation,
            "ssd",
        )
        return None

    def _direct_ssd_seconds(
        self, cluster_bytes: int, per_rank_bytes: int, direction: str
    ) -> float:
        """Return a no-DRAM GPU/SSD demotion service time.

        This helper remains for the direct baseline's HBM-to-SSD swap-out.
        Every SSD restore, including that baseline, uses
        :meth:`_schedule_staged_ssd_restore` so the read lands in transient
        CPU DRAM before traversing the CPU-to-GPU PCIe link.
        """

        if direction != "out":
            raise AnalysisConfigError(
                "direct SSD helper is swap-out only; every SSD restore must "
                "use CPU-DRAM staging"
            )
        rank_gbps = self.hardware.cpu.gpu_to_host_gbps_per_rank
        ssd_gbps = self.hardware.ssd.write_gbps_aggregate
        return self.hardware.ssd.fixed_latency_us * 1e-6 + max(
            per_rank_bytes / (rank_gbps * 1e9),
            cluster_bytes / (ssd_gbps * 1e9),
        )

    def _schedule_staged_ssd_restore(
        self,
        now_ns: int,
        cluster_bytes: int,
        per_rank_bytes: int,
        destination: str,
    ) -> tuple[int, int, int]:
        """Schedule SSD -> transient CPU DRAM -> GPU as two serial stages.

        ``destination`` is ``hbm`` for a single-pool replay and ``decode``
        for the D side of a P/D replay.  The transient bounce buffer is not a
        persistent CPU-cache entry, so it does not contribute to CPU cache
        occupancy.  It does consume the shared CPU-DRAM queue resource.  The
        caller has already reserved destination HBM before this method runs.

        The return value is ``(first_start, final_finish, total_queue_wait)``.
        Queue wait includes both the media-stage wait and any wait between the
        completed DRAM stage and the PCIe stage.
        """

        if destination not in {"hbm", "decode"}:
            raise AnalysisConfigError(
                "SSD restore destination must be 'hbm' or 'decode'"
            )
        if cluster_bytes <= 0 or per_rank_bytes <= 0:
            raise AnalysisConfigError("SSD restore byte counts must be positive")
        media_service = self.hardware.ssd.fixed_latency_us * 1e-6 + max(
            cluster_bytes
            / (self.hardware.ssd.read_gbps_aggregate * 1e9),
            cluster_bytes
            / (self.hardware.cpu.dram_write_gbps_aggregate * 1e9),
        )
        media_start, cpu_ready = self.queue.schedule(
            now_ns,
            media_service,
            ("ssd_read", "cpu_dram"),
            f"ssd_to_cpu_stage_for_{destination}",
            cluster_bytes,
        )
        pcie_service = cpu_transfer_seconds(
            cluster_bytes,
            per_rank_bytes,
            self.hardware.cpu,
            "in",
        )
        pcie_start, finish = self.queue.schedule(
            cpu_ready,
            pcie_service,
            ("cpu_dram", "pcie_in"),
            f"cpu_stage_to_{destination}",
            cluster_bytes,
        )
        total_wait = (media_start - now_ns) + (pcie_start - cpu_ready)
        return media_start, finish, total_wait

    def _hbm_to_ssd_direct(
        self, entry: _Entry, now_ns: int, reason: str
    ) -> int | None:
        retry = self._ensure_ssd_space(
            entry.cluster_bytes, now_ns, entry.session_id
        )
        if retry is not None:
            if retry == -1:
                self._drop(entry, "ssd_object_oversize")
                return None
            return retry
        service = self._direct_ssd_seconds(
            entry.cluster_bytes, entry.per_rank_bytes, "out"
        )
        _, finish = self.queue.schedule(
            now_ns,
            service,
            ("pcie_out", "ssd_write"),
            "hbm_to_ssd_direct",
            entry.cluster_bytes,
        )
        generation = self._bump_generation(entry)
        entry.available_ns = finish
        entry.move_reason = reason
        entry.transit_source_tier = "hbm"
        self._set_tier(entry, "transit_ssd")
        self._track_transit(entry, "ssd")
        self.policy_actions[reason] = self.policy_actions.get(reason, 0) + 1
        self._push(
            finish,
            self._MIGRATION_COMPLETE,
            "migration_complete",
            entry.session_id,
            generation,
            "ssd",
        )
        return None

    def _demote_hbm_victim(
        self, entry: _Entry, now_ns: int, reason: str
    ) -> int | None:
        if self.config.policy == "hbm_lru_recompute":
            self._drop(entry, reason)
            return None
        if self.config.policy == "hbm_ssd_direct":
            return self._hbm_to_ssd_direct(entry, now_ns, reason)
        return self._hbm_to_cpu(entry, now_ns, reason)

    def _hbm_to_cpu(self, entry: _Entry, now_ns: int, reason: str) -> int | None:
        retry = self._ensure_cpu_space(
            entry.cluster_bytes, now_ns, entry.session_id
        )
        if retry == -1:
            return self._hbm_to_ssd_oversize(entry, now_ns, reason + "_cpu_oversize")
        if retry is not None:
            return retry
        service = cpu_transfer_seconds(
            entry.cluster_bytes,
            entry.per_rank_bytes,
            self.hardware.cpu,
            "out",
        )
        _, finish = self.queue.schedule(
            now_ns,
            service,
            ("pcie_out", "cpu_dram"),
            "hbm_to_cpu",
            entry.cluster_bytes,
        )
        generation = self._bump_generation(entry)
        entry.available_ns = finish
        entry.move_reason = reason
        entry.transit_source_tier = "hbm"
        self._set_tier(entry, "transit_cpu")
        self._track_transit(entry, "cpu")
        self.policy_actions[reason] = self.policy_actions.get(reason, 0) + 1
        self._push(
            finish,
            self._MIGRATION_COMPLETE,
            "migration_complete",
            entry.session_id,
            generation,
            "cpu",
        )
        return None

    def _ensure_hbm_space(
        self, per_rank_bytes: int, now_ns: int, exclude: str | None = None
    ) -> int | None:
        budget = (
            self.prefill_hbm_kv_budget
            if self.config.pd_disaggregated
            else self.hbm_kv_budget
        )
        if per_rank_bytes > budget:
            return -1
        if self.config.pd_disaggregated:
            if self._active_bytes() + per_rank_bytes <= budget:
                return None
            retry = self._next_active_progress_ns(now_ns, exclude)
            if retry is not None:
                return retry
            # An independently admitted P branch can be waiting for its D
            # branch and therefore have no final completion epoch yet.  It is
            # still a legitimate future capacity release, woken explicitly by
            # _complete_call; distinguish this from an oversized object.
            return -2 if self.active else -1
        while (
            self.used["hbm"] + self._active_bytes() + per_rank_bytes
            > self.hbm_kv_budget
        ):
            victim = self._lru_entry("hbm", exclude)
            if victim is None:
                retry_candidates = [
                    candidate
                    for candidate in (
                        self._earliest_active_completion(
                            exclude, after_ns=now_ns
                        ),
                        self._earliest_source_release("hbm"),
                    )
                    if candidate is not None and candidate > now_ns
                ]
                return min(retry_candidates) if retry_candidates else -1
            retry = self._demote_hbm_victim(
                victim, now_ns, "hbm_capacity"
            )
            if retry is not None:
                self._push_lru(victim)
                return retry
            if victim.tier == "dropped":
                continue
            if victim.tier in {"transit_cpu", "transit_ssd"}:
                # Atomic demotion still occupies HBM until its completion.
                return victim.available_ns
            raise AssertionError("HBM victim neither moved nor dropped")
        return None

    def _ensure_decode_hbm_space(
        self, per_rank_bytes: int, now_ns: int, exclude: str | None = None
    ) -> int | None:
        if per_rank_bytes > self.decode_hbm_kv_budget:
            return -1
        while (
            self.used["hbm"]
            + self.decode_reserved_bytes_per_rank
            + per_rank_bytes
            > self.decode_hbm_kv_budget
        ):
            victim = self._lru_entry("hbm", exclude)
            if victim is None:
                candidates = [
                    candidate
                    for candidate in (
                        self._earliest_source_release("hbm"),
                        self._next_active_progress_ns(now_ns, exclude),
                    )
                    if candidate is not None and candidate >= now_ns
                ]
                if candidates:
                    return min(candidates)
                return (
                    -2
                    if self.used["hbm"]
                    + self.decode_reserved_bytes_per_rank
                    else -1
                )
            retry = self._demote_hbm_victim(
                victim, now_ns, "hbm_capacity"
            )
            if retry is not None:
                self._push_lru(victim)
                return retry
            if victim.tier == "dropped":
                continue
            if victim.tier in {"transit_cpu", "transit_ssd"}:
                return victim.available_ns
            raise AssertionError("decode HBM victim neither moved nor dropped")
        return None

    def _remove_entry(self, session_id: str) -> _Entry | None:
        entry = self.entries.pop(session_id, None)
        if entry is not None:
            self._bump_generation(entry)
            self._set_tier(entry, "consumed")
        return entry

    def _cancel_background_migration(self, entry: _Entry) -> None:
        """Cancel an uncommitted demotion while retaining its upper tier.

        The already-issued queue job remains busy because the scalar FCFS
        model cannot reclaim an in-flight DMA interval. Destination capacity is
        released immediately and the stale completion event is invalidated.
        """

        if entry.tier == "transit_cpu":
            source = "hbm"
        elif entry.tier == "transit_ssd":
            source = entry.transit_source_tier
        else:
            return
        self._bump_generation(entry)
        self._set_tier(entry, source)
        entry.transit_source_tier = ""
        entry.available_ns = 0
        self.policy_actions["migration_cancel_on_resume"] = (
            self.policy_actions.get("migration_cancel_on_resume", 0) + 1
        )
        self._push_lru(entry)

    def _schedule_ttl(self, entry: _Entry, now_ns: int) -> None:
        if self.config.demotion_mode == "capacity-only":
            return
        ttl = {
            "hbm": self.config.hbm_ttl_ns,
            "cpu": self.config.cpu_ttl_ns,
            "ssd": self.config.ssd_ttl_ns,
        }[entry.tier]
        self._push(
            now_ns + ttl,
            self._TTL,
            "ttl",
            entry.session_id,
            entry.generation,
            entry.tier,
        )

    def _finish_migration(
        self, now_ns: int, session_id: str, generation: int, tier: str
    ) -> None:
        entry = self.entries.get(session_id)
        if entry is None or entry.generation != generation:
            return
        if entry.tier != "transit_" + tier:
            return
        self._set_tier(entry, tier)
        entry.transit_source_tier = ""
        entry.available_ns = now_ns
        self._push_lru(entry)
        self._schedule_ttl(entry, now_ns)

    def _finish_restore(
        self,
        now_ns: int,
        session_id: str,
        generation: int,
        source: str,
        decode_reservation_bytes: int = 0,
        restored_cluster_bytes: int = 0,
        restored_cache_tokens: int = 0,
        exact_reuse_tokens: int = 0,
    ) -> None:
        entry = self.entries.get(session_id)
        if (
            entry is None
            or entry.generation != generation
            or entry.tier != "restore_" + source
        ):
            return
        self._remove_entry(session_id)
        if decode_reservation_bytes:
            state = self.pd_pending_compute.get(session_id)
            if state is None or state.source != source:
                raise AssertionError("lower restore completed without join state")
            if state.decode_reservation_bytes < decode_reservation_bytes:
                raise AssertionError("restore consumed excess decode reservation")
            state.decode_reservation_bytes -= decode_reservation_bytes
            self._release_decode_hbm(decode_reservation_bytes)
            entry = _Entry(
                session_id=session_id,
                cache_tokens=restored_cache_tokens,
                cluster_bytes=restored_cluster_bytes,
                per_rank_bytes=decode_reservation_bytes,
                tier="none",
                last_access_ns=now_ns,
                available_ns=now_ns,
                generation=self.session_generations.get(session_id, 0) + 1,
            )
            self.session_generations[session_id] = entry.generation
            self.entries[session_id] = entry
            self._set_tier(entry, "pinned_hbm")
            state.decode_prefix_ready = True
            state.lower_restore_finish_ns = now_ns
            self._advance_pd_call(now_ns, state)
            # A cold call is intentionally absent from the P admission FIFO
            # until its lower-tier restore completes.  If the P pool became
            # idle while it was loading, there may be no later completion to
            # wake the FIFO.  Make KV readiness an admission wake-up without
            # blocking any already-running compute batch.
            self._wake_prefill_head(now_ns)

    def _pd_d2p_complete(self, now_ns: int, session_id: str) -> None:
        # Compatibility for stale externally constructed event streams. New
        # replays schedule the join as soon as D->P finish is known.
        state = self.pd_pending_compute.get(session_id)
        if state is not None:
            self._advance_pd_call(now_ns, state)

    def _handle_ttl(
        self, now_ns: int, session_id: str, generation: int, tier: str
    ) -> None:
        entry = self.entries.get(session_id)
        if (
            entry is None
            or entry.generation != generation
            or entry.tier != tier
        ):
            return
        if (
            session_id in self.decode_restore_source_pins
            or session_id == self.decode_restore_retrying_session
        ):
            # Only the D-admission head owns a stable foreground source. Other
            # waiters remain policy-managed and re-resolve their actual source
            # when they reach the head.
            self.decode_restore_source_ttl_deferral_count += 1
            return
        if tier == "hbm":
            retry = self._demote_hbm_victim(entry, now_ns, "hbm_ttl")
        elif tier == "cpu":
            if self.config.policy != "tiered":
                raise AssertionError(
                    "non-tiered policy unexpectedly owns a CPU entry"
                )
            retry = self._cpu_to_ssd(entry, now_ns, "cpu_ttl")
        else:
            self._drop(entry, "ssd_ttl")
            retry = None
        if retry is not None and retry > now_ns:
            self._push(
                retry,
                self._TTL,
                "ttl",
                session_id,
                generation,
                tier,
            )

    def _roofline_seconds(self, tokens: int) -> float:
        if self.prompt_compute_model is not None:
            seconds = self.prompt_compute_model.recompute_seconds(tokens)
        else:
            seconds = roofline_recompute_seconds(
                self.model,
                self.hardware,
                tokens,
                self.tp_size,
                self.kv_dtype_bytes,
                self.layout,
                self.config.prefill_chunk_size,
            ).total_seconds
        return seconds * self.config.prompt_compute_scale

    def _cached_prefill_seconds(
        self, total_tokens: int, cached_tokens: int
    ) -> float:
        bounded_cached_tokens = min(max(0, cached_tokens), total_tokens)
        if self.prompt_compute_model is not None:
            seconds = self.prompt_compute_model.cached_prefill_seconds(
                total_tokens, bounded_cached_tokens
            )
        else:
            seconds = roofline_cached_prefill_seconds(
                self.model,
                self.hardware,
                total_tokens,
                bounded_cached_tokens,
                self.tp_size,
                self.kv_dtype_bytes,
                self.layout,
                self.config.prefill_chunk_size,
            ).total_seconds
        return seconds * self.config.prompt_compute_scale

    def _allocation_tokens(self, tokens: int) -> int:
        if tokens <= 0:
            return 0
        return math.ceil(tokens / self.config.block_size) * self.config.block_size

    @staticmethod
    def _add_restore_timing(
        rows: dict[str, dict[str, int]],
        key: str,
        raw_ns: int,
        hidden_ns: int,
        exposed_ns: int,
    ) -> None:
        row = rows.setdefault(
            key,
            {
                "event_count": 0,
                "raw_elapsed_ns": 0,
                "hidden_by_prefill_ns": 0,
                "exposed_decode_barrier_ns": 0,
                "other_concurrent_or_admission_ns": 0,
            },
        )
        row["event_count"] += 1
        row["raw_elapsed_ns"] += raw_ns
        row["hidden_by_prefill_ns"] += hidden_ns
        row["exposed_decode_barrier_ns"] += exposed_ns
        row["other_concurrent_or_admission_ns"] += (
            raw_ns - hidden_ns - exposed_ns
        )

    def _record_restore_join(
        self,
        call: ReplayCall,
        source: str,
        logical_ready_ns: int,
        prefill_start_ns: int,
        overlap_prefill_finish_ns: int,
        restore_finish_ns: int,
    ) -> None:
        """Record request-local restore time at the prefill/decode join.

        Raw elapsed time begins at the trace-visible request-ready epoch and
        includes source availability, HBM admission, transfer queueing, and
        every transfer in the restore chain.  It is not itself causal stall.
        In the default mode, analytical suffix prefill except its final token
        is a perfect-overlap upper bound and decode waits for both branches.
        A one-fresh-token execution has no pre-restore overlap budget.  Owner
        barrier time begins at the admitted prefill cutoff, while time between
        logical ready and prefill admission is reported separately rather than
        misattributed to the restore barrier.
        """

        raw_ns = max(0, restore_finish_ns - logical_ready_ns)
        if raw_ns <= 0:
            return
        if (
            self.config.restore_execution_mode == "async-decode-join"
            and call.fresh_prompt_tokens > 1
        ):
            hidden_ns = max(
                0,
                min(restore_finish_ns, overlap_prefill_finish_ns)
                - max(logical_ready_ns, prefill_start_ns),
            )
        else:
            hidden_ns = 0
        if self.config.restore_execution_mode == "async-decode-join":
            barrier_start_ns = overlap_prefill_finish_ns
        else:
            # Pre-admission restore is a request-local readiness gate. The
            # returning request cannot enter analytical prompt compute while
            # its KV is loading, so the entire ready-to-KV-ready interval is
            # causally exposed even though unrelated calls keep running.
            barrier_start_ns = logical_ready_ns
        exposed_ns = max(0, restore_finish_ns - barrier_start_ns)
        other_ns = max(0, raw_ns - hidden_ns - exposed_ns)
        if raw_ns != hidden_ns + exposed_ns + other_ns:
            raise AssertionError("restore timing decomposition failed")
        self.raw_restore_elapsed_ns += raw_ns
        self.restore_hidden_by_prefill_ns += hidden_ns
        self.exposed_restore_barrier_ns += exposed_ns
        self.restore_other_concurrent_or_admission_ns += other_ns
        self.raw_restore_intervals.append(
            (logical_ready_ns, restore_finish_ns)
        )
        self.raw_restore_intervals_by_gap.setdefault(
            call.return_gap_type, []
        ).append((logical_ready_ns, restore_finish_ns))
        if exposed_ns:
            exposed_interval = (
                barrier_start_ns,
                restore_finish_ns,
            )
            self.exposed_restore_barrier_intervals.append(exposed_interval)
            self.exposed_restore_intervals_by_gap.setdefault(
                call.return_gap_type, []
            ).append(exposed_interval)
        self._add_restore_timing(
            self.restore_timing_by_source,
            source,
            raw_ns,
            hidden_ns,
            exposed_ns,
        )
        self._add_restore_timing(
            self.restore_timing_by_return_gap_type,
            call.return_gap_type,
            raw_ns,
            hidden_ns,
            exposed_ns,
        )
        cross = self.restore_timing_by_gap_and_source.setdefault(
            call.return_gap_type, {}
        )
        self._add_restore_timing(
            cross, source, raw_ns, hidden_ns, exposed_ns
        )

    def _prompt_join_completion_ns(
        self,
        restore_finish_ns: int,
        prefill_start_ns: int,
        overlap_prefill_finish_ns: int,
        total_compute_ns: int,
        post_restore_compute_ns: int,
    ) -> int:
        if self.config.restore_execution_mode in {
            "async-pre-admission",
            "serial-before-prefill",
        }:
            return (
                max(restore_finish_ns, prefill_start_ns)
                + total_compute_ns
            )
        return (
            max(restore_finish_ns, overlap_prefill_finish_ns)
            + post_restore_compute_ns
        )

    def _initialize_pd_call_state(
        self,
        now_ns: int,
        session_id: str,
        call_index: int,
        source: str,
        source_reason: str,
    ) -> _PdCallJoinState:
        session = self.sessions[session_id]
        call = session.calls[call_index]
        active_allocation_tokens = self._allocation_tokens(call.input_tokens)
        active_per_rank = (
            active_allocation_tokens
            * self.layout.physical_bytes_per_token_per_rank
        )
        full_decode_per_rank = (
            self._allocation_tokens(call.cache_tokens)
            * self.layout.physical_bytes_per_token_per_rank
        )
        restore_cluster = (
            call.reusable_allocation_tokens
            * self.layout.physical_bytes_per_token_cluster
        )
        restore_rank = (
            call.reusable_allocation_tokens
            * self.layout.physical_bytes_per_token_per_rank
        )
        full_seconds = self._roofline_seconds(call.input_tokens)
        cached_seconds = self._cached_prefill_seconds(
            call.input_tokens, call.effective_reuse_tokens
        )
        hit = (
            source in {"hbm", "cpu", "ssd"}
            and call.effective_reuse_tokens > 0
        )
        compute_seconds = cached_seconds if hit else full_seconds
        compute_ns = int(math.ceil(compute_seconds * 1e9))
        overlap_compute_ns = 0
        if hit and call.fresh_prompt_tokens > 1:
            overlap_compute_ns = int(math.ceil(
                self._cached_prefill_seconds(
                    call.input_tokens - 1,
                    call.effective_reuse_tokens,
                ) * 1e9
            ))
        overlap_compute_ns = min(compute_ns, overlap_compute_ns)
        call_key = (session_id, call_index)
        self.pd_call_admission_sequence += 1
        state = _PdCallJoinState(
            session_id=session_id,
            call_index=call_index,
            admission_sequence=self.pd_call_admission_sequence,
            source=source,
            source_reason=source_reason,
            logical_ready_ns=self.call_logical_ready_ns[call_key],
            active_per_rank=active_per_rank,
            full_decode_per_rank=full_decode_per_rank,
            restore_cluster_bytes=restore_cluster,
            restore_rank_bytes=restore_rank,
            full_compute_seconds=full_seconds,
            cached_compute_seconds=cached_seconds,
            selected_compute_seconds=compute_seconds,
            compute_ns=compute_ns,
            overlap_compute_ns=overlap_compute_ns,
            post_restore_compute_ns=compute_ns - overlap_compute_ns,
            decode_prefix_ready=False,
        )
        self.pd_pending_compute[session_id] = state
        entry = self.entries.get(session_id)
        if hit:
            source_is_atomic_transit_owner = (
                entry is not None
                and entry.tier in {"transit_cpu", "transit_ssd"}
                and entry.transit_source_tier == source
            )
            if entry is None or (
                entry.tier != source and not source_is_atomic_transit_owner
            ):
                raise AssertionError("selected P/D reuse source disappeared")
        else:
            self.decode_restore_source_pins.discard(session_id)
            self._remove_entry(session_id)
        return state

    def _refresh_pd_state_compute(
        self,
        now_ns: int,
        state: _PdCallJoinState,
        source: str,
        source_reason: str,
    ) -> None:
        if state.accounting_recorded and (
            source != state.source or source_reason != state.source_reason
        ):
            raise AssertionError("accounted P/D source changed")
        call = self.sessions[state.session_id].calls[state.call_index]
        old_hit = (
            state.source in {"hbm", "cpu", "ssd"}
            and call.effective_reuse_tokens > 0
        )
        new_hit = (
            source in {"hbm", "cpu", "ssd"}
            and call.effective_reuse_tokens > 0
        )
        if state.prefill_admitted and old_hit and not new_hit:
            if self.config.restore_execution_mode == "async-decode-join":
                wasted_ns = min(
                    state.overlap_compute_ns,
                    max(0, now_ns - state.prefill_start_ns),
                )
                wasted_seconds = wasted_ns / 1e9
                state.speculative_compute_seconds += wasted_seconds
                self.pd_speculative_prefill_wasted_seconds += wasted_seconds
            # Cached suffix work cannot be retained after the prefix source is
            # lost. Conservatively restart the full prompt at source-finalize.
            state.prefill_start_ns = now_ns

        state.source = source
        state.source_reason = source_reason
        selected_seconds = (
            state.cached_compute_seconds
            if new_hit
            else state.full_compute_seconds
        )
        state.selected_compute_seconds = selected_seconds
        state.compute_ns = int(math.ceil(selected_seconds * 1e9))
        overlap_compute_ns = 0
        if new_hit and call.fresh_prompt_tokens > 1:
            overlap_compute_ns = int(math.ceil(
                self._cached_prefill_seconds(
                    call.input_tokens - 1,
                    call.effective_reuse_tokens,
                ) * 1e9
            ))
        state.overlap_compute_ns = min(
            state.compute_ns, overlap_compute_ns
        )
        state.post_restore_compute_ns = (
            state.compute_ns - state.overlap_compute_ns
        )
        if state.prefill_admitted:
            state.overlap_prefill_finish_ns = (
                state.prefill_start_ns + state.overlap_compute_ns
            )

    def _record_pd_state_accounting(
        self, state: _PdCallJoinState
    ) -> None:
        if state.accounting_recorded:
            return
        call = self.sessions[state.session_id].calls[state.call_index]
        self.full_prompt_reference_seconds += state.full_compute_seconds
        self.prompt_compute_seconds += (
            state.selected_compute_seconds
            + state.speculative_compute_seconds
        )
        self._record_source(
            call,
            state.source,
            state.source_reason,
            state.full_compute_seconds,
            max(
                0.0,
                state.full_compute_seconds - state.cached_compute_seconds,
            ),
        )
        call_key = (state.session_id, state.call_index)
        self.call_sources[call_key] = state.source
        self.pd_call_sources[state.session_id] = state.source
        state.accounting_recorded = True

    def _resolve_pd_source_at_decode_head(
        self, now_ns: int, state: _PdCallJoinState
    ) -> tuple[_Entry | None, int | None]:
        call = self.sessions[state.session_id].calls[state.call_index]
        if call.effective_reuse_tokens <= 0:
            self._refresh_pd_state_compute(
                now_ns, state, "no_reuse", "no_reusable_prefix"
            )
            self._remove_entry(state.session_id)
            return None, None

        entry = self.entries.get(state.session_id)
        if entry is not None and entry.tier in {"transit_cpu", "transit_ssd"}:
            if self.config.cancel_migration_on_resume:
                self._cancel_background_migration(entry)
            else:
                wait_finish_ns = max(now_ns, entry.available_ns)
                unaccounted_start_ns = max(
                    now_ns, state.source_wait_accounted_until_ns
                )
                if wait_finish_ns > unaccounted_start_ns:
                    wait_ns = wait_finish_ns - unaccounted_start_ns
                    self.resume_inflight_migration_wait_ns += wait_ns
                    self.foreground_migration_intervals.append(
                        (unaccounted_start_ns, wait_finish_ns)
                    )
                    state.source_wait_accounted_until_ns = wait_finish_ns
                return entry, max(now_ns + 1, entry.available_ns + 1)

        entry = self.entries.get(state.session_id)
        if entry is not None and entry.tier in {"hbm", "cpu", "ssd"}:
            self._refresh_pd_state_compute(
                now_ns,
                state,
                entry.tier,
                entry.move_reason,
            )
            return entry, None

        source_reason = "missing_lineage"
        if entry is not None and entry.tier == "dropped":
            source_reason = entry.drop_reason or entry.move_reason
        self._refresh_pd_state_compute(
            now_ns, state, "recompute", source_reason
        )
        self._remove_entry(state.session_id)
        return None, None

    def _older_pd_state_waits(
        self, state: _PdCallJoinState, branch: str
    ) -> bool:
        for older in self.pd_pending_compute.values():
            if older.admission_sequence >= state.admission_sequence:
                continue
            if (
                branch == "prefill"
                and not older.prefill_admitted
                and self._pd_state_compute_ready(older)
            ):
                return True
            if (
                branch == "decode_restore"
                and not older.decode_admitted
            ):
                return True
        return False

    def _pd_state_compute_ready(self, state: _PdCallJoinState) -> bool:
        """Return whether a P/D call may enter analytical prompt compute."""

        call = self.sessions[state.session_id].calls[state.call_index]
        cold_hit = (
            state.source in {"cpu", "ssd"}
            and call.effective_reuse_tokens > 0
        )
        if self.config.restore_execution_mode in {
            "async-pre-admission",
            "serial-before-prefill",
        }:
            # Establish one acquisition order for the two finite HBM pools:
            # D admission always precedes P admission. Otherwise a younger
            # call can hold its P allocation while waiting for D, as an older
            # restored call holds D while waiting for P. Neither allocation
            # can then reach its join and the event queue drains in a classic
            # hold-and-wait deadlock. Cold restores additionally keep the
            # owner out of compute until the reserved D prefix is fully ready.
            return (
                state.decode_admitted
                and (not cold_hit or state.decode_prefix_ready)
            )
        return True

    def _wake_prefill_head(self, now_ns: int) -> None:
        retry_ns = now_ns + 1
        if self.prefill_waiters and (
            self.prefill_wakeup_ns is None
            or retry_ns < self.prefill_wakeup_ns
        ):
            self.prefill_wakeup_generation += 1
            self.prefill_wakeup_ns = retry_ns
            self._push(
                retry_ns,
                self._CALL_READY,
                "prefill_capacity_wakeup",
                self.prefill_wakeup_generation,
            )

    def _wake_decode_restore_head(self, now_ns: int) -> None:
        retry_ns = now_ns + 1
        if (
            self.decode_restore_waiters
            and (
                self.decode_restore_wakeup_ns is None
                or retry_ns < self.decode_restore_wakeup_ns
            )
        ):
            self.decode_restore_wakeup_generation += 1
            self.decode_restore_wakeup_ns = retry_ns
            self.decode_restore_wakeup_event_count += 1
            self._push(
                retry_ns,
                self._CALL_READY,
                "decode_restore_capacity_wakeup",
                self.decode_restore_wakeup_generation,
            )

    def _pd_prefill_can_admit_now(
        self, state: _PdCallJoinState
    ) -> bool:
        if state.prefill_admitted:
            return True
        if self._older_pd_state_waits(state, "prefill"):
            return False
        if (
            self.prefill_waiters
            and self.prefill_retrying_session != state.session_id
        ):
            return False
        return (
            self._active_bytes() + state.active_per_rank
            <= self.prefill_hbm_kv_budget
        )

    def _try_start_pd_prefill(
        self, now_ns: int, state: _PdCallJoinState
    ) -> None:
        if state.prefill_admitted:
            return
        if not self._pd_state_compute_ready(state):
            # Destination D-HBM has already been reserved before the lower
            # tier copy. Keep only this returning request out of compute; do
            # not enqueue it on the P admission FIFO until its prefix is ready.
            return
        if (
            self._older_pd_state_waits(state, "prefill")
            or (
                self.prefill_waiters
                and self.prefill_retrying_session != state.session_id
            )
        ):
            self._queue_prefill_waiter(
                None,
                state.session_id,
                state.call_index,
                now_ns,
            )
            return
        retry = self._ensure_hbm_space(
            state.active_per_rank, now_ns, state.session_id
        )
        if retry == -1:
            raise AnalysisConfigError(
                f"session {state.session_id} call {state.call_index}: "
                "active KV object does not fit the prefill HBM KV budget"
            )
        if retry is not None:
            self._queue_prefill_waiter(
                None if retry == -2 else retry,
                state.session_id,
                state.call_index,
                now_ns,
            )
            return
        self._dequeue_prefill_waiter(
            now_ns, state.session_id, state.call_index
        )
        state.prefill_admitted = True
        state.prefill_start_ns = now_ns
        state.overlap_prefill_finish_ns = (
            now_ns + state.overlap_compute_ns
        )
        self.active[state.session_id] = _Active(
            state.session_id,
            state.active_per_rank,
            -1,
        )
        self.active_bytes_per_rank += state.active_per_rank
        self._record_active_peak()
        self._wake_prefill_head(now_ns)

    def _try_start_pd_lower_restore(
        self, now_ns: int, state: _PdCallJoinState
    ) -> None:
        if state.decode_admitted:
            return
        call = self.sessions[state.session_id].calls[state.call_index]
        candidate_entry = self.entries.get(state.session_id)
        retained_hbm = (
            candidate_entry.per_rank_bytes
            if (
                call.effective_reuse_tokens > 0
                and candidate_entry is not None
                and candidate_entry.tier == "hbm"
            )
            else 0
        )
        candidate_admission_bytes = max(
            0, state.full_decode_per_rank - retained_hbm
        )
        order_blocked = (
            self._older_pd_state_waits(state, "decode_restore")
            or (
                self.decode_restore_waiters
                and self.decode_restore_retrying_session != state.session_id
            )
        )
        safe_hbm_backfill = (
            order_blocked
            and candidate_entry is not None
            and candidate_entry.tier == "hbm"
            and candidate_admission_bytes == 0
            and self._pd_prefill_can_admit_now(state)
        )
        if order_blocked and not safe_hbm_backfill:
            self.decode_restore_fcfs_deferral_count += 1
            self._queue_decode_restore_waiter(
                None,
                state.session_id,
                state.call_index,
                now_ns,
            )
            return

        entry, source_retry_ns = self._resolve_pd_source_at_decode_head(
            now_ns, state
        )
        if source_retry_ns is not None:
            self._queue_decode_restore_waiter(
                source_retry_ns,
                state.session_id,
                state.call_index,
                now_ns,
            )
            return
        retained_hbm = (
            entry.per_rank_bytes
            if (
                state.source == "hbm"
                and entry is not None
                and entry.tier == "hbm"
            )
            else 0
        )
        admission_bytes = max(
            0, state.full_decode_per_rank - retained_hbm
        )
        hit = (
            state.source in {"hbm", "cpu", "ssd"}
            and call.effective_reuse_tokens > 0
        )
        if hit:
            if entry is None or entry.tier != state.source:
                raise AssertionError("P/D source changed before pin")
            self.decode_restore_source_pins.add(state.session_id)
            self.decode_restore_max_source_pins = max(
                self.decode_restore_max_source_pins,
                len(self.decode_restore_source_pins),
            )
        else:
            self.decode_restore_source_pins.discard(state.session_id)
        retry = self._ensure_decode_hbm_space(
            admission_bytes, now_ns, state.session_id
        )
        if retry == -1:
            raise AnalysisConfigError(
                f"session {state.session_id} call {state.call_index}: "
                "final decode KV footprint does not fit decode HBM"
            )
        if retry is not None:
            self.decode_restore_capacity_retry_count += 1
            self._queue_decode_restore_waiter(
                None if retry == -2 else max(now_ns + 1, retry + 1),
                state.session_id,
                state.call_index,
                now_ns,
            )
            return
        self._reserve_decode_hbm(admission_bytes)
        state.decode_admitted = True
        state.decode_admission_ns = now_ns
        state.decode_reservation_bytes = admission_bytes
        # A P-capacity wakeup can make a queued resident-HBM call eligible for
        # completion-safe D backfill. In that path the D wakeup handler did not
        # pop this call, so admission itself must retire the exact waiter.
        self._dequeue_decode_restore_waiter(
            now_ns, state.session_id, state.call_index
        )
        self._record_pd_state_accounting(state)
        if state.source == "hbm":
            if entry is None or entry.tier != "hbm":
                raise AssertionError("decode-HBM source disappeared at admission")
            state.decode_prefix_ready = True
            self._wake_decode_restore_head(now_ns)
            return
        if state.source not in {"cpu", "ssd"}:
            state.decode_prefix_ready = True
            self._wake_decode_restore_head(now_ns)
            return
        if state.restore_rank_bytes > admission_bytes:
            raise AssertionError("restored prefix exceeds final decode footprint")
        if state.source == "cpu":
            service = cpu_transfer_seconds(
                state.restore_cluster_bytes,
                state.restore_rank_bytes,
                self.hardware.cpu,
                "in",
            )
            resources = ("pcie_in", "cpu_dram")
            kind = "cpu_to_decode"
            start, finish = self.queue.schedule(
                now_ns,
                service,
                resources,
                kind,
                state.restore_cluster_bytes,
            )
            queue_wait_ns = start - now_ns
        else:
            start, finish, queue_wait_ns = self._schedule_staged_ssd_restore(
                now_ns,
                state.restore_cluster_bytes,
                state.restore_rank_bytes,
                "decode",
            )
        self.restore_queue_wait_ns += queue_wait_ns
        self.restore_stall_ns += finish - now_ns
        self.foreground_migration_intervals.append((now_ns, finish))
        entry = self.entries.get(state.session_id)
        if entry is None or entry.tier != state.source:
            raise AssertionError("restore source disappeared before issue")
        restore_generation = self._bump_generation(entry)
        entry.available_ns = finish
        self._set_tier(entry, "restore_" + state.source)
        self.decode_restore_source_pins.discard(state.session_id)
        self._track_restore(entry, state.source)
        state.lower_restore_scheduled = True
        state.lower_restore_issue_ns = now_ns
        state.lower_restore_finish_ns = finish
        self._push(
            finish,
            self._MIGRATION_COMPLETE,
            "restore_complete",
            state.session_id,
            restore_generation,
            state.source,
            state.restore_rank_bytes,
            state.restore_cluster_bytes,
            self.sessions[state.session_id].calls[
                state.call_index
            ].reusable_allocation_tokens,
            self.sessions[state.session_id].calls[
                state.call_index
            ].effective_reuse_tokens,
        )
        self._wake_decode_restore_head(now_ns)

    def _schedule_pd_join(
        self, now_ns: int, state: _PdCallJoinState
    ) -> None:
        if (
            state.join_scheduled
            or not state.prefill_admitted
            or not state.decode_admitted
        ):
            return
        call = self.sessions[state.session_id].calls[state.call_index]
        hit = (
            state.source in {"hbm", "cpu", "ssd"}
            and call.effective_reuse_tokens > 0
        )
        if not hit:
            completion_ns = max(
                state.prefill_start_ns + state.compute_ns,
                state.decode_admission_ns,
            )
        else:
            if not state.d2p_scheduled:
                return
            completion_ns = self._prompt_join_completion_ns(
                state.restore_finish_ns,
                state.prefill_start_ns,
                state.overlap_prefill_finish_ns,
                state.compute_ns,
                state.post_restore_compute_ns,
            )
        state.join_scheduled = True
        state.join_completion_ns = completion_ns
        active = self.active.get(state.session_id)
        if active is None:
            raise AssertionError("P/D join lost its prefill reservation")
        active.completion_ns = completion_ns
        heapq.heappush(
            self.active_heap, (active.completion_ns, state.session_id)
        )
        self._push(
            completion_ns,
            self._PROMPT_COMPLETE,
            "prompt_complete",
            state.session_id,
            state.call_index,
            state.active_per_rank,
        )

    def _maybe_schedule_pd_d2p(
        self, now_ns: int, state: _PdCallJoinState
    ) -> None:
        call = self.sessions[state.session_id].calls[state.call_index]
        hit = (
            state.source in {"hbm", "cpu", "ssd"}
            and call.effective_reuse_tokens > 0
        )
        if not hit:
            self._schedule_pd_join(now_ns, state)
            return
        if (
            state.d2p_scheduled
            or not state.prefill_admitted
            or not state.decode_prefix_ready
        ):
            return
        entry = self.entries.get(state.session_id)
        if entry is None:
            raise AssertionError("decode-owned prefix disappeared before D->P")
        if entry.tier == "hbm":
            self._bump_generation(entry)
            self._set_tier(entry, "pinned_hbm")
        elif entry.tier != "pinned_hbm":
            raise AssertionError("D->P source is not decode-HBM resident")
        self.decode_restore_source_pins.discard(state.session_id)
        wire_rank = (
            call.effective_reuse_tokens
            * self.layout.physical_bytes_per_token_per_rank
        )
        wire_cluster = (
            call.effective_reuse_tokens
            * self.layout.physical_bytes_per_token_cluster
        )
        service = (
            self.config.pd_fixed_latency_us * 1e-6
            + wire_rank / (self.config.pd_link_gbps_per_rank * 1e9)
        )
        start, finish = self.queue.schedule(
            now_ns,
            service,
            ("pd_fabric", "decode_copy", "prefill_copy"),
            "decode_hbm_to_prefill",
            wire_cluster,
        )
        self.restore_queue_wait_ns += start - now_ns
        self.restore_stall_ns += finish - now_ns
        self.pd_d2p_stall_ns += finish - now_ns
        self.pd_d2p_bytes += wire_cluster
        self.foreground_migration_intervals.append((now_ns, finish))
        state.d2p_scheduled = True
        state.d2p_issue_ns = now_ns
        state.restore_finish_ns = finish
        self._record_restore_join(
            call,
            state.source,
            state.logical_ready_ns,
            state.prefill_start_ns,
            state.overlap_prefill_finish_ns,
            finish,
        )
        self._schedule_pd_join(now_ns, state)

    def _advance_pd_call(
        self, now_ns: int, state: _PdCallJoinState
    ) -> None:
        # The two branches intentionally make progress independently.  Strict
        # FCFS checks use the same call sequence in both pools. A zero-growth
        # HBM-resident D branch may completion-safely backfill only when its P
        # branch also fits immediately.
        self._try_start_pd_lower_restore(now_ns, state)
        self._try_start_pd_prefill(now_ns, state)
        self._maybe_schedule_pd_d2p(now_ns, state)

    def _record_source(
        self,
        call: ReplayCall,
        source: str,
        source_reason: str,
        full_seconds: float,
        recompute_extra_seconds: float,
    ) -> None:
        if not call.selected_positive_transition:
            return
        self.selected_seen += 1
        if call.effective_reuse_tokens <= 0:
            self.source_counts["no_reuse"] += 1
            return
        self.eligible_seen += 1
        self.total_reusable_tokens += call.effective_reuse_tokens
        self.source_counts[source] += 1
        self.source_tokens[source] += call.effective_reuse_tokens
        gap_counts = self.source_counts_by_return_gap_type.setdefault(
            call.return_gap_type, {}
        )
        gap_counts[source] = gap_counts.get(source, 0) + 1
        gap_tokens = self.source_tokens_by_return_gap_type.setdefault(
            call.return_gap_type, {}
        )
        gap_tokens[source] = (
            gap_tokens.get(source, 0) + call.effective_reuse_tokens
        )
        if source == "ssd":
            self.ssd_source_reasons[source_reason] = (
                self.ssd_source_reasons.get(source_reason, 0) + 1
            )
        if source == "recompute":
            reason = source_reason or "missing_lineage"
            self.recompute_reasons[reason] = (
                self.recompute_reasons.get(reason, 0) + 1
            )
            self.recompute_tokens += call.effective_reuse_tokens
            self.recompute_seconds += min(
                recompute_extra_seconds, full_seconds
            )

    def _start_call(self, now_ns: int, session_id: str, call_index: int) -> None:
        call_key = (session_id, call_index)
        self.call_logical_ready_ns.setdefault(call_key, now_ns)
        existing_pd_state = self.pd_pending_compute.get(session_id)
        if self.config.pd_disaggregated and existing_pd_state is not None:
            if existing_pd_state.call_index != call_index:
                raise AssertionError("session has two concurrent P/D calls")
            self._advance_pd_call(now_ns, existing_pd_state)
            return
        session = self.sessions[session_id]
        call = session.calls[call_index]
        if not call.context_eligible:
            self.call_sources[call_key] = "context_infeasible"
            self.context_infeasible_calls += 1
            self._remove_entry(session_id)
            self._push(
                now_ns,
                self._CALL_COMPLETE,
                "call_complete",
                session_id,
                call_index,
                0,
            )
            return
        entry = self.entries.get(session_id)
        if (
            not self.config.pd_disaggregated
            and entry is not None
            and entry.tier in {"transit_cpu", "transit_ssd"}
        ):
            if self.config.cancel_migration_on_resume:
                self._cancel_background_migration(entry)
            else:
                wait_ns = max(0, entry.available_ns - now_ns)
                self.resume_inflight_migration_wait_ns += wait_ns
                if call.effective_reuse_tokens <= 0:
                    self.no_reuse_inflight_migration_wait_ns += wait_ns
                if wait_ns:
                    self.foreground_migration_intervals.append(
                        (now_ns, entry.available_ns)
                    )
                self._push(
                    max(now_ns + 1, entry.available_ns),
                    self._CALL_READY,
                    "call_ready",
                    session_id,
                    call_index,
                )
                return

        source = "recompute"
        source_reason = "missing_lineage"
        if call.effective_reuse_tokens <= 0:
            source = "no_reuse"
            source_reason = "no_reusable_prefix"
        elif entry is not None and entry.tier in {"hbm", "cpu", "ssd"}:
            source = entry.tier
            source_reason = entry.move_reason
        elif (
            self.config.pd_disaggregated
            and entry is not None
            and entry.tier in {"transit_cpu", "transit_ssd"}
            and entry.transit_source_tier in {"hbm", "cpu"}
        ):
            # Atomic demotion still owns a readable upper-tier copy. The D
            # branch decides whether to cancel or await that copy operation;
            # the independent P branch may meanwhile execute fresh suffix work.
            source = entry.transit_source_tier
            source_reason = entry.move_reason
        elif entry is not None and entry.tier == "dropped":
            source_reason = entry.drop_reason or entry.move_reason

        if self.config.pd_disaggregated:
            state = self._initialize_pd_call_state(
                now_ns,
                session_id,
                call_index,
                source,
                source_reason,
            )
            self._advance_pd_call(now_ns, state)
            return

        foreground_lower_restore = (
            self.config.pd_disaggregated
            and source in {"cpu", "ssd"}
            and call.effective_reuse_tokens > 0
        )
        if foreground_lower_restore:
            self.decode_restore_source_pins.add(session_id)
            self.decode_restore_max_source_pins = max(
                self.decode_restore_max_source_pins,
                len(self.decode_restore_source_pins),
            )
        else:
            self.decode_restore_source_pins.discard(session_id)

        active_allocation_tokens = self._allocation_tokens(
            call.input_tokens
            if self.config.pd_disaggregated
            else call.cache_tokens
        )
        active_per_rank = (
            active_allocation_tokens
            * self.layout.physical_bytes_per_token_per_rank
        )
        hbm_credit = (
            entry.per_rank_bytes
            if (
                not self.config.pd_disaggregated
                and entry is not None
                and entry.tier == "hbm"
            )
            else 0
        )
        needed = max(0, active_per_rank - hbm_credit)
        retry = self._ensure_hbm_space(needed, now_ns, session_id)
        if retry == -1:
            raise AnalysisConfigError(
                f"session {session_id} call {call_index}: active KV object "
                "does not fit the HBM KV budget"
            )
        if retry is not None:
            if self.config.pd_disaggregated:
                self._queue_prefill_waiter(
                    retry, session_id, call_index, now_ns
                )
            else:
                self.hbm_capacity_block_ns += max(0, retry - now_ns)
                self._push(
                    retry,
                    self._CALL_READY,
                    "call_ready",
                    session_id,
                    call_index,
                )
            return
        blocked_since_ns = self.prefill_waiter_since_ns.pop(session_id, None)
        if blocked_since_ns is not None:
            self.hbm_capacity_block_ns += max(0, now_ns - blocked_since_ns)

        restore_finish = now_ns
        restore_cluster = (
            call.reusable_allocation_tokens
            * self.layout.physical_bytes_per_token_cluster
        )
        restore_rank = (
            call.reusable_allocation_tokens
            * self.layout.physical_bytes_per_token_per_rank
        )
        if (
            self.config.pd_disaggregated
            and source in {"cpu", "ssd"}
            and restore_rank > 0
        ):
            # Do not let a newly ready restore bypass an older decode-HBM
            # admission waiter.  The current head is exempt while its wake-up
            # handler calls back into this method.
            if (
                self.decode_restore_waiters
                and self.decode_restore_retrying_session != session_id
            ):
                self.decode_restore_fcfs_deferral_count += 1
                self._queue_decode_restore_waiter(
                    self.decode_restore_wakeup_ns or now_ns + 1,
                    session_id,
                    call_index,
                    now_ns,
                )
                return
            decode_retry = self._ensure_decode_hbm_space(
                restore_rank, now_ns, session_id
            )
            if decode_retry == -1:
                raise AnalysisConfigError(
                    f"session {session_id} call {call_index}: restored prefix "
                    "does not fit decode HBM"
                )
            if decode_retry is not None:
                retry_ns = max(now_ns + 1, decode_retry + 1)
                self.decode_restore_capacity_retry_count += 1
                self._queue_decode_restore_waiter(
                    retry_ns,
                    session_id,
                    call_index,
                    now_ns,
                )
                return
            self._reserve_decode_hbm(restore_rank)

        if (
            self.config.pd_disaggregated
            and source == "hbm"
            and restore_rank > 0
        ):
            exact_wire_rank = (
                call.effective_reuse_tokens
                * self.layout.physical_bytes_per_token_per_rank
            )
            exact_wire_cluster = (
                call.effective_reuse_tokens
                * self.layout.physical_bytes_per_token_cluster
            )
            pd_service = (
                self.config.pd_fixed_latency_us * 1e-6
                + exact_wire_rank
                / (self.config.pd_link_gbps_per_rank * 1e9)
            )
            start, restore_finish = self.queue.schedule(
                now_ns,
                pd_service,
                ("pd_fabric", "decode_copy", "prefill_copy"),
                "decode_hbm_to_prefill",
                exact_wire_cluster,
            )
            self.restore_queue_wait_ns += start - now_ns
            self.restore_stall_ns += restore_finish - now_ns
            self.foreground_migration_intervals.append(
                (now_ns, restore_finish)
            )
            self.pd_d2p_stall_ns += restore_finish - now_ns
            self.pd_d2p_bytes += exact_wire_cluster
            if entry is None:
                raise AssertionError("decode-HBM source disappeared")
            self._bump_generation(entry)
            self._set_tier(entry, "pinned_hbm")
        elif source in {"cpu", "ssd"} and restore_rank > 0:
            if source == "cpu":
                service = cpu_transfer_seconds(
                    restore_cluster, restore_rank, self.hardware.cpu, "in"
                )
                resources = ("pcie_in", "cpu_dram")
                kind = "cpu_to_hbm"
                start, restore_finish = self.queue.schedule(
                    now_ns, service, resources, kind, restore_cluster
                )
                queue_wait_ns = start - now_ns
            else:
                start, restore_finish, queue_wait_ns = (
                    self._schedule_staged_ssd_restore(
                        now_ns, restore_cluster, restore_rank, "hbm"
                    )
                )
            decode_reservation = 0
            if self.config.pd_disaggregated:
                decode_reservation = restore_rank
            self.restore_queue_wait_ns += queue_wait_ns
            self.restore_stall_ns += restore_finish - now_ns
            self.foreground_migration_intervals.append(
                (now_ns, restore_finish)
            )
            if entry is None:
                raise AssertionError("restore source disappeared")
            restore_generation = self._bump_generation(entry)
            entry.available_ns = restore_finish
            self._set_tier(entry, "restore_" + source)
            self.decode_restore_source_pins.discard(session_id)
            self._track_restore(entry, source)
            self._push(
                restore_finish,
                self._MIGRATION_COMPLETE,
                "restore_complete",
                session_id,
                restore_generation,
                source,
                decode_reservation,
                restore_cluster if decode_reservation else 0,
                call.reusable_allocation_tokens if decode_reservation else 0,
                call.effective_reuse_tokens if decode_reservation else 0,
            )
        else:
            self._remove_entry(session_id)

        full_seconds = self._roofline_seconds(call.input_tokens)
        cached_seconds = self._cached_prefill_seconds(
            call.input_tokens, call.effective_reuse_tokens
        )
        self.full_prompt_reference_seconds += full_seconds
        if source in {"hbm", "cpu", "ssd"} and call.effective_reuse_tokens > 0:
            compute_seconds = cached_seconds
        else:
            compute_seconds = full_seconds
        self.prompt_compute_seconds += compute_seconds
        self._record_source(
            call,
            source,
            source_reason,
            full_seconds,
            max(0.0, full_seconds - cached_seconds),
        )
        self.call_sources[call_key] = source
        self.pd_call_sources[session_id] = source
        compute_ns = int(math.ceil(compute_seconds * 1e9))
        logical_ready_ns = self.call_logical_ready_ns[call_key]
        has_restore_chain = (
            call.effective_reuse_tokens > 0
            and source in {"cpu", "ssd"}
        ) or (
            self.config.pd_disaggregated
            and call.effective_reuse_tokens > 0
            and source == "hbm"
        )
        prefill_start_ns = now_ns
        overlap_compute_ns = 0
        if has_restore_chain and call.fresh_prompt_tokens > 1:
            overlap_compute_ns = int(math.ceil(
                self._cached_prefill_seconds(
                    call.input_tokens - 1,
                    call.effective_reuse_tokens,
                ) * 1e9
            ))
        overlap_compute_ns = min(compute_ns, overlap_compute_ns)
        post_restore_compute_ns = compute_ns - overlap_compute_ns
        overlap_prefill_finish_ns = (
            prefill_start_ns + overlap_compute_ns
        )
        if has_restore_chain:
            self._record_restore_join(
                call,
                source,
                logical_ready_ns,
                prefill_start_ns,
                overlap_prefill_finish_ns,
                restore_finish,
            )
            completion_ns = self._prompt_join_completion_ns(
                restore_finish,
                prefill_start_ns,
                overlap_prefill_finish_ns,
                compute_ns,
                post_restore_compute_ns,
            )
        else:
            completion_ns = now_ns + compute_ns
        self.active[session_id] = _Active(
            session_id,
            active_per_rank,
            completion_ns,
        )
        self.active_bytes_per_rank += active_per_rank
        heapq.heappush(
            self.active_heap,
            (self.active[session_id].completion_ns, session_id),
        )
        self._record_active_peak()
        self._push(
            completion_ns,
            self._CALL_COMPLETE,
            "call_complete",
            session_id,
            call_index,
            active_per_rank,
            0,
            0,
        )

    def _queue_prefill_waiter(
        self,
        retry_ns: int | None,
        session_id: str,
        call_index: int,
        blocked_since_ns: int,
    ) -> None:
        self.prefill_waiter_since_ns.setdefault(
            session_id, blocked_since_ns
        )
        if session_id not in self.prefill_waiter_sessions:
            item = (session_id, call_index)
            if self.prefill_retrying_session == session_id:
                self.prefill_waiters.appendleft(item)
            else:
                # Cold calls join this FIFO only after their KV is ready.  A
                # younger ready call can already be queued by then.  Keep the
                # queue consistent with the admission-sequence ordering used
                # by _older_pd_state_waits; otherwise the younger head waits
                # for an older call sitting behind it and neither can make
                # progress.
                state = self.pd_pending_compute.get(session_id)
                insert_at = len(self.prefill_waiters)
                if state is not None:
                    for index, (queued_id, _) in enumerate(
                        self.prefill_waiters
                    ):
                        queued = self.pd_pending_compute.get(queued_id)
                        if (
                            queued is not None
                            and queued.admission_sequence
                            > state.admission_sequence
                        ):
                            insert_at = index
                            break
                self.prefill_waiters.insert(insert_at, item)
            self.prefill_waiter_sessions.add(session_id)
        if retry_ns is None:
            return
        retry_ns = max(retry_ns, blocked_since_ns + 1)
        if self.prefill_wakeup_ns is None or retry_ns < self.prefill_wakeup_ns:
            self.prefill_wakeup_generation += 1
            self.prefill_wakeup_ns = retry_ns
            self._push(
                retry_ns,
                self._CALL_READY,
                "prefill_capacity_wakeup",
                self.prefill_wakeup_generation,
            )

    def _dequeue_prefill_waiter(
        self, now_ns: int, session_id: str, call_index: int
    ) -> None:
        """Retire and account an exact P waiter at successful admission."""

        if session_id in self.prefill_waiter_sessions:
            item = (session_id, call_index)
            try:
                self.prefill_waiters.remove(item)
            except ValueError as exc:
                raise AssertionError(
                    "prefill waiter call index mismatch"
                ) from exc
            self.prefill_waiter_sessions.remove(session_id)
        blocked_since_ns = self.prefill_waiter_since_ns.pop(
            session_id, None
        )
        if blocked_since_ns is not None:
            self.hbm_capacity_block_ns += max(
                0, now_ns - blocked_since_ns
            )

    def _handle_prefill_capacity_wakeup(
        self, now_ns: int, generation: int
    ) -> None:
        if generation != self.prefill_wakeup_generation:
            return
        self.prefill_wakeup_ns = None
        if not self.prefill_waiters:
            return
        session_id, call_index = self.prefill_waiters.popleft()
        self.prefill_waiter_sessions.remove(session_id)
        self.prefill_retrying_session = session_id
        try:
            self._start_call(now_ns, session_id, call_index)
        finally:
            self.prefill_retrying_session = None

    def _queue_decode_restore_waiter(
        self,
        retry_ns: int | None,
        session_id: str,
        call_index: int,
        blocked_since_ns: int,
    ) -> None:
        """Queue one CPU/SSD resume for decode-HBM admission.

        Only the FIFO head is retried at a wake-up.  If it remains blocked,
        ``decode_restore_retrying_session`` puts it back at the head.  This
        avoids the quadratic retry storm caused by broadcasting one
        ``CALL_READY`` event per blocked request at every capacity release.
        """

        self.decode_restore_waiter_since_ns.setdefault(
            session_id, blocked_since_ns
        )
        if session_id not in self.decode_restore_waiter_sessions:
            item = (session_id, call_index)
            if self.decode_restore_retrying_session == session_id:
                self.decode_restore_waiters.appendleft(item)
            else:
                self.decode_restore_waiters.append(item)
            self.decode_restore_waiter_sessions.add(session_id)
            self.decode_restore_enqueue_count += 1
            self.decode_restore_max_depth = max(
                self.decode_restore_max_depth,
                len(self.decode_restore_waiters),
            )
        if retry_ns is None:
            return
        retry_ns = max(retry_ns, blocked_since_ns + 1)
        if (
            self.decode_restore_wakeup_ns is None
            or retry_ns < self.decode_restore_wakeup_ns
        ):
            self.decode_restore_wakeup_generation += 1
            self.decode_restore_wakeup_ns = retry_ns
            self.decode_restore_wakeup_event_count += 1
            self._push(
                retry_ns,
                self._CALL_READY,
                "decode_restore_capacity_wakeup",
                self.decode_restore_wakeup_generation,
            )

    def _dequeue_decode_restore_waiter(
        self, now_ns: int, session_id: str, call_index: int
    ) -> None:
        """Retire an exact waiter admitted outside the D-head wakeup path."""

        if session_id not in self.decode_restore_waiter_sessions:
            return
        item = (session_id, call_index)
        try:
            self.decode_restore_waiters.remove(item)
        except ValueError as exc:
            raise AssertionError(
                "decode-restore waiter call index mismatch"
            ) from exc
        self.decode_restore_waiter_sessions.remove(session_id)
        blocked_since_ns = self.decode_restore_waiter_since_ns.pop(
            session_id, now_ns
        )
        block_ns = max(0, now_ns - blocked_since_ns)
        self.hbm_capacity_block_ns += block_ns
        self.decode_restore_capacity_block_ns += block_ns

    def _handle_decode_restore_capacity_wakeup(
        self, now_ns: int, generation: int
    ) -> None:
        if generation != self.decode_restore_wakeup_generation:
            return
        self.decode_restore_wakeup_ns = None
        if not self.decode_restore_waiters:
            return
        session_id, call_index = self.decode_restore_waiters.popleft()
        self.decode_restore_waiter_sessions.remove(session_id)
        blocked_since_ns = self.decode_restore_waiter_since_ns.pop(
            session_id, now_ns
        )
        block_ns = max(0, now_ns - blocked_since_ns)
        self.hbm_capacity_block_ns += block_ns
        self.decode_restore_capacity_block_ns += block_ns
        self.decode_restore_wakeup_count += 1
        self.decode_restore_retrying_session = session_id
        try:
            self._start_call(now_ns, session_id, call_index)
        finally:
            self.decode_restore_retrying_session = None

    def _queue_decode_waiter(
        self,
        retry_ns: int,
        session_id: str,
        call_index: int,
        active_per_rank: int,
    ) -> None:
        if session_id not in self.decode_waiter_sessions:
            item = (session_id, call_index, active_per_rank)
            if self.decode_retrying_session == session_id:
                self.decode_waiters.appendleft(item)
            else:
                self.decode_waiters.append(item)
            self.decode_waiter_sessions.add(session_id)
        if self.decode_wakeup_ns is None or retry_ns < self.decode_wakeup_ns:
            self.decode_wakeup_generation += 1
            self.decode_wakeup_ns = retry_ns
            self._push(
                retry_ns,
                self._PROMPT_COMPLETE,
                "decode_capacity_wakeup",
                self.decode_wakeup_generation,
            )

    def _handle_decode_capacity_wakeup(
        self, now_ns: int, generation: int
    ) -> None:
        if generation != self.decode_wakeup_generation:
            return
        self.decode_wakeup_ns = None
        if not self.decode_waiters:
            return
        session_id, call_index, active_per_rank = self.decode_waiters.popleft()
        self.decode_waiter_sessions.remove(session_id)
        self.decode_retrying_session = session_id
        try:
            self._prompt_complete(
                now_ns, session_id, call_index, active_per_rank
            )
        finally:
            self.decode_retrying_session = None
        if self.decode_waiters and self.decode_wakeup_ns is None:
            self.decode_wakeup_generation += 1
            self.decode_wakeup_ns = now_ns + 1
            self._push(
                now_ns + 1,
                self._PROMPT_COMPLETE,
                "decode_capacity_wakeup",
                self.decode_wakeup_generation,
            )

    def _prompt_complete(
        self,
        now_ns: int,
        session_id: str,
        call_index: int,
        active_per_rank: int,
    ) -> None:
        session = self.sessions[session_id]
        call = session.calls[call_index]
        source = self.pd_call_sources.get(session_id, "recompute")
        entry = self.entries.get(session_id)
        retained_bytes = (
            entry.per_rank_bytes
            if entry is not None and entry.tier == "pinned_hbm"
            else 0
        )
        state = self.pd_pending_compute.get(session_id)
        if state is None or state.call_index != call_index:
            raise AssertionError("prompt completion lost P/D join state")
        if not state.decode_admitted:
            raise AssertionError("prompt completed before decode admission")
        full_per_rank = state.full_decode_per_rank
        needed = state.decode_reservation_bytes
        if retained_bytes + needed < full_per_rank:
            raise AssertionError(
                "pre-reserved decode footprint is incomplete"
            )
        if source in {"hbm", "cpu", "ssd"}:
            wire_tokens = max(
                0, call.input_tokens - call.effective_reuse_tokens
            )
        else:
            wire_tokens = call.input_tokens
        wire_rank = (
            wire_tokens * self.layout.physical_bytes_per_token_per_rank
        )
        wire_cluster = (
            wire_tokens * self.layout.physical_bytes_per_token_cluster
        )
        service = (
            self.config.pd_fixed_latency_us * 1e-6
            + wire_rank / (self.config.pd_link_gbps_per_rank * 1e9)
        )
        start, finish = self.queue.schedule(
            now_ns,
            service,
            ("pd_fabric", "prefill_copy", "decode_copy"),
            "prefill_to_decode",
            wire_cluster,
        )
        self.pd_p2d_queue_wait_ns += start - now_ns
        self.pd_p2d_handoff_ns += finish - now_ns
        self.pd_p2d_bytes += wire_cluster
        self.foreground_kv_transfer_intervals.append((now_ns, finish))
        active = self.active.get(session_id)
        if active is not None:
            active.completion_ns = finish
            heapq.heappush(
                self.active_heap, (active.completion_ns, session_id)
            )
        self.pd_handoff_reservations[session_id] = (needed, full_per_rank)
        self._push(
            finish,
            self._CALL_COMPLETE,
            "call_complete",
            session_id,
            call_index,
            active_per_rank,
            needed,
            full_per_rank,
        )

    def _complete_call(
        self,
        now_ns: int,
        session_id: str,
        call_index: int,
        active_per_rank: int,
        decode_reserved_bytes: int = 0,
        full_decode_bytes: int = 0,
    ) -> None:
        session = self.sessions[session_id]
        call = session.calls[call_index]
        active = self.active.pop(session_id, None)
        if active is not None:
            self.active_bytes_per_rank -= active.per_rank_bytes
        if self.config.pd_disaggregated:
            state = self.pd_pending_compute.pop(session_id, None)
            if state is None and active_per_rank > 0:
                raise AssertionError("completed P/D call lost its join state")
            if session_id in self.decode_restore_waiter_sessions:
                raise AssertionError(
                    "completed P/D call retained a decode-restore waiter"
                )
            if (
                session_id in self.prefill_waiter_sessions
                or session_id in self.prefill_waiter_since_ns
            ):
                raise AssertionError(
                    "completed P/D call retained a prefill waiter"
                )
            self.decode_restore_source_pins.discard(session_id)
            retained = self.entries.get(session_id)
            if retained is not None and retained.tier == "pinned_hbm":
                self._remove_entry(session_id)
            if decode_reserved_bytes:
                self._release_decode_hbm(decode_reserved_bytes)
            self.pd_handoff_reservations.pop(session_id, None)
            self.pd_call_sources.pop(session_id, None)
        if (
            active_per_rank > 0
            and call.cache_eligible
            and call_index + 1 < len(session.calls)
            and session.calls[call_index + 1].context_eligible
        ):
            cluster_bytes = (
                self._allocation_tokens(call.cache_tokens)
                * self.layout.physical_bytes_per_token_cluster
            )
            entry = _Entry(
                session_id=session_id,
                cache_tokens=call.cache_tokens,
                cluster_bytes=cluster_bytes,
                per_rank_bytes=(
                    full_decode_bytes
                    if self.config.pd_disaggregated
                    else active_per_rank
                ),
                tier="none",
                last_access_ns=now_ns,
                available_ns=now_ns,
                generation=self.session_generations.get(session_id, 0) + 1,
            )
            self.session_generations[session_id] = entry.generation
            self.entries[session_id] = entry
            self._set_tier(entry, "hbm")
            self._push_lru(entry)
            self._schedule_ttl(entry, now_ns)
        self._check_capacity()
        if self.config.pd_disaggregated:
            self._wake_prefill_head(now_ns)
            self._wake_decode_restore_head(now_ns)
        self.call_completion_ns[(session_id, call_index)] = now_ns
        self.request_trace_end_ns = max(self.request_trace_end_ns, now_ns)
        if call_index + 1 < len(session.calls):
            self._push(
                now_ns + call.tool_duration_ns,
                self._CALL_READY,
                "call_ready",
                session_id,
                call_index + 1,
            )

    def run(self) -> dict[str, Any]:
        for session in self.workload.sessions:
            self._push(
                session.arrival_time_ns,
                self._CALL_READY,
                "call_ready",
                session.session_id,
                0,
            )
        while self.events:
            now_ns, _, _, kind, payload = heapq.heappop(self.events)
            self.event_horizon_ns = max(self.event_horizon_ns, now_ns)
            if kind == "call_ready":
                self._start_call(now_ns, str(payload[0]), int(payload[1]))
            elif kind == "prefill_capacity_wakeup":
                self._handle_prefill_capacity_wakeup(
                    now_ns, int(payload[0])
                )
            elif kind == "decode_restore_capacity_wakeup":
                self._handle_decode_restore_capacity_wakeup(
                    now_ns, int(payload[0])
                )
            elif kind == "prompt_complete":
                self._prompt_complete(
                    now_ns,
                    str(payload[0]),
                    int(payload[1]),
                    int(payload[2]),
                )
            elif kind == "pd_d2p_complete":
                self._pd_d2p_complete(now_ns, str(payload[0]))
            elif kind == "decode_capacity_wakeup":
                self._handle_decode_capacity_wakeup(
                    now_ns, int(payload[0])
                )
            elif kind == "call_complete":
                self._complete_call(
                    now_ns,
                    str(payload[0]),
                    int(payload[1]),
                    int(payload[2]),
                    int(payload[3]) if len(payload) > 3 else 0,
                    int(payload[4]) if len(payload) > 4 else 0,
                )
            elif kind == "migration_complete":
                self._finish_migration(
                    now_ns, str(payload[0]), int(payload[1]), str(payload[2])
                )
            elif kind == "restore_complete":
                self._finish_restore(
                    now_ns,
                    str(payload[0]),
                    int(payload[1]),
                    str(payload[2]),
                    int(payload[3]),
                    int(payload[4]),
                    int(payload[5]),
                    int(payload[6]),
                )
            elif kind == "ttl":
                self._handle_ttl(
                    now_ns, str(payload[0]), int(payload[1]), str(payload[2])
                )
            else:
                raise AssertionError(f"unknown event kind {kind}")
        if self.selected_seen != self.workload.selected_positive_transitions:
            raise AssertionError(
                "selected transition conservation failed: "
                f"{self.selected_seen} != "
                f"{self.workload.selected_positive_transitions}"
            )
        if self.eligible_seen != self.workload.selected_reuse_eligible_transitions:
            raise AssertionError(
                "reuse-eligible transition conservation failed"
            )
        eligible_sources = {
            key: self.source_counts[key]
            for key in ("hbm", "cpu", "ssd", "recompute")
        }
        if sum(eligible_sources.values()) != self.eligible_seen:
            raise AssertionError("resume source counts do not sum to denominator")
        if sum(self.source_tokens.values()) != self.total_reusable_tokens:
            raise AssertionError("resume source token conservation failed")
        if self.active or self.active_bytes_per_rank != 0:
            raise AssertionError("active prefill KV leaked at terminal state")
        if self.decode_reserved_bytes_per_rank != 0:
            raise AssertionError("decode HBM reservation leaked at terminal state")
        if self.pd_pending_compute or self.pd_handoff_reservations:
            raise AssertionError("P/D in-flight state leaked at terminal state")
        if self.decode_waiters or self.decode_waiter_sessions:
            raise AssertionError("decode-capacity waiters leaked at terminal state")
        if (
            self.prefill_waiters
            or self.prefill_waiter_sessions
            or self.prefill_waiter_since_ns
        ):
            raise AssertionError("prefill-capacity waiters leaked at terminal state")
        if (
            self.decode_restore_waiters
            or self.decode_restore_waiter_sessions
            or self.decode_restore_waiter_since_ns
            or self.decode_restore_wakeup_ns is not None
            or self.decode_restore_retrying_session is not None
            or self.decode_restore_source_pins
        ):
            raise AssertionError(
                "decode-restore-capacity waiters leaked at terminal state"
            )
        if any(self.used.values()):
            raise AssertionError(f"tier bytes leaked at terminal state: {self.used}")
        if any(
            entry.tier not in {"dropped", "consumed"}
            for entry in self.entries.values()
        ):
            raise AssertionError("resident or in-flight KV leaked at terminal state")
        source_key = "decode_hbm" if self.config.pd_disaggregated else "hbm"
        reported_sources = {
            source_key: eligible_sources["hbm"],
            "cpu": eligible_sources["cpu"],
            "ssd": eligible_sources["ssd"],
            "recompute": eligible_sources["recompute"],
        }
        reported_source_tokens = {
            source_key: self.source_tokens["hbm"],
            "cpu": self.source_tokens["cpu"],
            "ssd": self.source_tokens["ssd"],
            "recompute": self.source_tokens["recompute"],
        }
        workload_metadata = self.workload.metadata_dict()
        all_calls_by_gap = workload_metadata["return_gap_type_counts"]
        selected_by_gap = workload_metadata[
            "selected_return_gap_type_counts"
        ]
        eligible_by_gap = workload_metadata[
            "eligible_return_gap_type_counts"
        ]
        source_counts_by_gap: dict[str, dict[str, Any]] = {}
        for gap_type in sorted(all_calls_by_gap):
            internal_counts = self.source_counts_by_return_gap_type.get(
                gap_type, {}
            )
            internal_tokens = self.source_tokens_by_return_gap_type.get(
                gap_type, {}
            )
            row_sources = {
                source_key: internal_counts.get("hbm", 0),
                "cpu": internal_counts.get("cpu", 0),
                "ssd": internal_counts.get("ssd", 0),
                "recompute": internal_counts.get("recompute", 0),
            }
            row_tokens = {
                source_key: internal_tokens.get("hbm", 0),
                "cpu": internal_tokens.get("cpu", 0),
                "ssd": internal_tokens.get("ssd", 0),
                "recompute": internal_tokens.get("recompute", 0),
            }
            row_all = all_calls_by_gap[gap_type]
            row_eligible = eligible_by_gap.get(gap_type, 0)
            if sum(row_sources.values()) != row_eligible:
                raise AssertionError(
                    "return-gap source row does not sum to its eligible "
                    f"denominator: {gap_type}"
                )
            source_counts_by_gap[gap_type] = {
                "all_request_count": row_all,
                "selected_positive_transition_count": selected_by_gap.get(
                    gap_type, 0
                ),
                "reuse_eligible_transition_count": row_eligible,
                "not_reuse_eligible_or_not_selected_count": (
                    row_all - row_eligible
                ),
                "source_counts": row_sources,
                "source_fractions_of_all_requests_in_return_class": {
                    key: value / row_all if row_all else 0.0
                    for key, value in row_sources.items()
                },
                "source_fractions_of_reuse_eligible_in_return_class": {
                    key: value / row_eligible if row_eligible else 0.0
                    for key, value in row_sources.items()
                },
                "source_reusable_tokens": row_tokens,
            }
        for tier, expected in reported_sources.items():
            observed = sum(
                row["source_counts"][tier]
                for row in source_counts_by_gap.values()
            )
            if observed != expected:
                raise AssertionError(
                    "return-gap source columns do not conserve global "
                    f"counts: {tier} {observed} != {expected}"
                )
        capacities = {
            "prefill_hbm_kv_budget_bytes_per_rank": (
                self.prefill_hbm_kv_budget
                if self.config.pd_disaggregated
                else self.hbm_kv_budget
            ),
            "decode_hbm_kv_budget_bytes_per_rank": (
                self.decode_hbm_kv_budget
                if self.config.pd_disaggregated
                else self.hbm_kv_budget
            ),
            "cpu_kv_budget_bytes": self.config.cpu_capacity_bytes,
            "ssd_kv_budget_bytes": self.config.ssd_capacity_bytes,
        }
        if not self.config.pd_disaggregated:
            capacities["hbm_kv_budget_bytes_per_rank"] = self.hbm_kv_budget
        if self.config.pd_disaggregated:
            peak = {
                "prefill_hbm_active_bytes_per_rank": self.peaks["active_hbm"],
                "decode_hbm_active_plus_idle_bytes_per_rank": self.peaks["hbm"],
                "decode_hbm_reserved_bytes_per_rank": (
                    self.peak_decode_reserved_bytes_per_rank
                ),
                "cpu_bytes": self.peaks["cpu"],
                "ssd_bytes": self.peaks["ssd"],
            }
            hbm_peak_fractions = {
                "prefill_hbm": peak["prefill_hbm_active_bytes_per_rank"]
                / self.prefill_hbm_kv_budget,
                "decode_hbm": peak[
                    "decode_hbm_active_plus_idle_bytes_per_rank"
                ] / self.decode_hbm_kv_budget,
            }
        else:
            peak = {
                "hbm_active_plus_idle_bytes_per_rank": self.peaks["hbm"],
                "hbm_active_bytes_per_rank": self.peaks["active_hbm"],
                "cpu_bytes": self.peaks["cpu"],
                "ssd_bytes": self.peaks["ssd"],
            }
            hbm_peak_fractions = {
                "hbm": peak["hbm_active_plus_idle_bytes_per_rank"]
                / self.hbm_kv_budget,
            }
        aggregate_migration_stall_ns = (
            self.restore_stall_ns + self.resume_inflight_migration_wait_ns
        )
        migration_exposure_union_ns = _interval_union_ns(
            self.foreground_migration_intervals
        )
        all_kv_exposure_union_ns = _interval_union_ns(
            self.foreground_migration_intervals
            + self.foreground_kv_transfer_intervals
        )
        first_arrival_ns = min(
            session.arrival_time_ns for session in self.workload.sessions
        )
        request_makespan_ns = max(
            0, self.request_trace_end_ns - first_arrival_ns
        )
        call_active_intervals = [
            (self.call_logical_ready_ns[key], completion_ns)
            for key, completion_ns in self.call_completion_ns.items()
        ]
        offered_call_active_union_ns = _interval_union_ns(
            call_active_intervals
        )
        offered_call_idle_complement_ns = max(
            0, request_makespan_ns - offered_call_active_union_ns
        )
        raw_restore_union_ns = _interval_union_ns(
            self.raw_restore_intervals
        )
        exposed_restore_union_ns = _interval_union_ns(
            self.exposed_restore_barrier_intervals
        )

        def timing_row(row: Mapping[str, int]) -> dict[str, Any]:
            return {
                "event_count": row["event_count"],
                "request_summed_raw_elapsed_seconds": (
                    row["raw_elapsed_ns"] / 1e9
                ),
                "request_summed_hidden_by_prefill_seconds": (
                    row["hidden_by_prefill_ns"] / 1e9
                ),
                "request_summed_exposed_decode_barrier_seconds": (
                    row["exposed_decode_barrier_ns"] / 1e9
                ),
                "request_summed_exposed_compute_admission_gate_seconds": (
                    row["exposed_decode_barrier_ns"] / 1e9
                ),
                "request_summed_other_concurrent_or_admission_seconds": (
                    row["other_concurrent_or_admission_ns"] / 1e9
                ),
                "hidden_fraction_of_raw_elapsed": (
                    row["hidden_by_prefill_ns"] / row["raw_elapsed_ns"]
                    if row["raw_elapsed_ns"] else 0.0
                ),
            }

        def public_source_name(source: str) -> str:
            return source_key if source == "hbm" else source

        restore_timing_by_source = {
            public_source_name(source): timing_row(row)
            for source, row in sorted(self.restore_timing_by_source.items())
        }
        restore_timing_by_gap = {}
        for gap_type, row in sorted(
            self.restore_timing_by_return_gap_type.items()
        ):
            restore_timing_by_gap[gap_type] = {
                **timing_row(row),
                "wall_clock_raw_elapsed_union_seconds": (
                    _interval_union_ns(
                        self.raw_restore_intervals_by_gap.get(gap_type, [])
                    ) / 1e9
                ),
                "wall_clock_exposed_decode_barrier_union_seconds": (
                    _interval_union_ns(
                        self.exposed_restore_intervals_by_gap.get(
                            gap_type, []
                        )
                    ) / 1e9
                ),
            }
        restore_timing_cross = {
            gap_type: {
                public_source_name(source): timing_row(row)
                for source, row in sorted(sources.items())
            }
            for gap_type, sources in sorted(
                self.restore_timing_by_gap_and_source.items()
            )
        }
        queue_report = self.queue.report()
        transfer_bytes = queue_report["bytes_by_kind"]
        ssd_write_bytes = (
            transfer_bytes.get("cpu_to_ssd", 0)
            + transfer_bytes.get("hbm_to_ssd_oversize", 0)
            + transfer_bytes.get("hbm_to_ssd_direct", 0)
        )
        ssd_read_bytes = (
            transfer_bytes.get("ssd_to_cpu_stage_for_hbm", 0)
            + transfer_bytes.get("ssd_to_cpu_stage_for_decode", 0)
            + transfer_bytes.get("ssd_to_hbm", 0)
            + transfer_bytes.get("ssd_to_decode", 0)
            + transfer_bytes.get("ssd_to_hbm_direct", 0)
            + transfer_bytes.get("ssd_to_decode_direct", 0)
        )
        cascade = {
            "hbm_lru_recompute": "HBM -> recompute",
            "hbm_ssd_direct": "HBM -> SSD -> recompute",
            "tiered": "HBM -> CPU DRAM -> SSD -> recompute",
        }[self.config.policy]
        if self.config.pd_disaggregated:
            effective_reserves = (
                self.config
                .effective_prefill_hbm_static_reserve_bytes_per_rank,
                self.config
                .effective_decode_hbm_static_reserve_bytes_per_rank,
            )
            zero_reserve_warnings = (
                [
                    "One or more effective static HBM reserves are zero; "
                    "this is optimistic for engine workspaces. Sweep or "
                    "measure each active role's reserve."
                ]
                if 0 in effective_reserves
                else []
            )
        else:
            zero_reserve_warnings = (
                [
                    "A zero static HBM reserve is optimistic for engine "
                    "workspaces; sweep or measure this reserve."
                ]
                if self.config.hbm_static_reserve_bytes_per_rank == 0
                else []
            )
        if self.prompt_compute_model is None:
            prompt_compute_metadata: dict[str, Any] = {
                "model_kind": "aggregate_nominal_roofline",
                "calibrated_from_measurements": False,
                "description": (
                    "Full-causal aggregate roofline multiplied by the "
                    "declared sensitivity scale."
                ),
            }
        else:
            raw_prompt_compute_metadata = self.prompt_compute_model.metadata()
            if not isinstance(raw_prompt_compute_metadata, Mapping):
                raise AnalysisConfigError(
                    "prompt compute model metadata must be a mapping"
                )
            prompt_compute_metadata = dict(raw_prompt_compute_metadata)
            prompt_compute_metadata.setdefault(
                "model_kind", "external_calibrated_prompt_model"
            )
            prompt_compute_metadata.setdefault(
                "calibrated_from_measurements", True
            )
            prompt_compute_metadata.setdefault(
                "description",
                "External calibrated full/cached prompt latency model.",
            )
        return {
            "schema_version": 11,
            "model": self.model.name,
            "hardware": self.hardware.name,
            "tp_size": self.tp_size,
            "kv_dtype_bytes": self.kv_dtype_bytes,
            "kv_layout": self.layout.to_dict(),
            "hardware_spec": self.hardware.to_dict(),
            "hardware_sensitivity": {
                "cpu_system_memory_bytes": self.config.cpu_capacity_bytes,
                "cpu_system_memory_reference_contract": (
                    "DGX H100 marketed 2 TB SI = 2,000,000,000,000 bytes"
                ),
                "cpu_capacity_matches_reference_contract": (
                    self.config.cpu_capacity_bytes == 2_000_000_000_000
                ),
                "pcie_effective_gbps_per_gpu": (
                    self.hardware.cpu.host_to_gpu_gbps_per_rank
                ),
                "pd_link_gbps_per_gpu_one_way": (
                    self.config.pd_link_gbps_per_rank
                ),
                "pd_link_reference_gbps_per_gpu_bidirectional": (
                    DGX_H100_NVLINK_BIDIRECTIONAL_GBPS_PER_GPU
                ),
                "pd_link_matches_half_bidirectional_reference": (
                    self.config.pd_link_gbps_per_rank
                    == DGX_H100_NVLINK_ONE_WAY_GBPS_PER_GPU
                ),
                "pd_link_contract_provenance": (
                    "The DGX H100 guide publishes 900 GB/s GPU-to-GPU "
                    "bandwidth. The default uses 450 GB/s per direction as "
                    "a one-way nominal sensitivity; replace it with a "
                    "measured peer-copy curve for headline latency."
                ),
                "cpu_dram_queue_semantics": (
                    "One node-shared half-duplex cpu_dram queue resource. "
                    "SSD staging writes, CPU-cache reads, and CPU<->GPU/SSD "
                    "paths serialize against the configured aggregate DRAM "
                    "bandwidth instead of assuming independent full-rate "
                    "read and write channels."
                ),
                "ssd_reference_platform": "DGX H100 KIOXIA CM6 example",
                "ssd_reference_model": "KCM6DRUL3T84",
                "ssd_reference_interface": "PCIe 4.0 x4",
                "ssd_reference_device_count": DGX_H100_CM6_DEVICE_COUNT,
                "ssd_reference_read_gbps_per_device": (
                    DGX_H100_CM6_READ_GBPS_PER_DEVICE
                ),
                "ssd_reference_write_gbps_per_device": (
                    DGX_H100_CM6_WRITE_GBPS_PER_DEVICE
                ),
                "ssd_reference_ideal_read_gbps_aggregate": (
                    DGX_H100_CM6_IDEAL_READ_GBPS
                ),
                "ssd_reference_ideal_write_gbps_aggregate": (
                    DGX_H100_CM6_IDEAL_WRITE_GBPS
                ),
                "ssd_configured_read_gbps_aggregate": (
                    self.hardware.ssd.read_gbps_aggregate
                ),
                "ssd_configured_write_gbps_aggregate": (
                    self.hardware.ssd.write_gbps_aggregate
                ),
                "ssd_bandwidths_match_manufacturer_upper_bound": (
                    math.isclose(
                        self.hardware.ssd.read_gbps_aggregate,
                        DGX_H100_CM6_IDEAL_READ_GBPS,
                    )
                    and math.isclose(
                        self.hardware.ssd.write_gbps_aggregate,
                        DGX_H100_CM6_IDEAL_WRITE_GBPS,
                    )
                ),
                "ssd_contract_provenance": (
                    "The NVIDIA DGX H100 user guide specifies an 8 x 3.84-TB "
                    "U.2 RAID 0 data-cache pool, and the firmware-guide "
                    "example enumerates eight KIOXIA KCM6DRUL3T84 devices. "
                    "KIOXIA rates that 3.84-TB PCIe 4.0 x4 model at up to "
                    "6.9/4.2 GB/s sequential read/write. Assuming an all-CM6 "
                    "pool, 55.2/33.6 GB/s is the inferred eight-drive "
                    "manufacturer upper bound, not measured end-to-end RAID "
                    "0 throughput. Replace it with fio measurements on the "
                    "target system."
                ),
                "ssd_reference_sources": [
                    DGX_H100_SYSTEM_SPEC_URL,
                    DGX_H100_NVME_SUPPORT_URL,
                    KIOXIA_CM6_R_PRODUCT_BRIEF_URL,
                ],
            },
            "replay_config": asdict(self.config),
            "workload": workload_metadata,
            "execution_scope": {
                "global_arrival_clock": True,
                "session_dependency_chains": True,
                "analytical_prompt_compute": True,
                "prompt_compute_scale": self.config.prompt_compute_scale,
                "prompt_compute_scale_provenance": (
                    self.config.effective_prompt_compute_scale_provenance
                ),
                "prompt_compute_model": (
                    prompt_compute_metadata["description"]
                ),
                "prompt_compute_model_kind": (
                    prompt_compute_metadata["model_kind"]
                ),
                "prompt_compute_calibration": prompt_compute_metadata,
                "active_request_kv_reserved": True,
                "pd_disaggregated": self.config.pd_disaggregated,
                "pd_hbm_pool_count": 2 if self.config.pd_disaggregated else 1,
                "decode_compute_modeled": False,
                "decode_active_kv_time_modeled": False,
                "llm_compute_queue_or_batching_modeled": False,
                "restore_execution_mode": (
                    self.config.restore_execution_mode
                ),
                "canonical_restore_execution_mode": (
                    "async-pre-admission"
                    if self.config.restore_execution_mode
                    == "serial-before-prefill"
                    else self.config.restore_execution_mode
                ),
                "swap_out_blocks_other_calls": False,
                "restore_barrier_scope": (
                    "returning_request_compute_admission_only"
                    if self.config.restore_execution_mode
                    in {"async-pre-admission", "serial-before-prefill"}
                    else "returning_request_decode_only"
                ),
                "restore_prefill_overlap_model": (
                    "perfect analytical overlap through fresh-1 suffix tokens, "
                    "then a restore join before the final prompt token"
                    if self.config.restore_execution_mode
                    == "async-decode-join"
                    else "no same-request analytical compute before complete "
                    "KV restore; unrelated calls remain runnable"
                ),
                "pd_branch_admission": (
                    "Final-footprint D-HBM admission reserves capacity before "
                    "lower-tier load. In async-pre-admission mode, P-HBM "
                    "admission waits until lower->D completes; D->P starts "
                    "after both the restored prefix and P capacity are ready."
                ),
                "pd_deadlock_prevention": (
                    "Both branches use the same admission sequence; only the "
                    "D-head pins a movable source. A zero-growth resident-HBM "
                    "call may backfill only when its P branch fits immediately. "
                    "The complete final D footprint is pre-reserved before "
                    "prompt completion."
                ),
                "interpretation": (
                    "Capacity/transfer-queue sensitivity. Calls overlap and "
                    "use a prompt-only roofline; this is not cycle-level serving."
                ),
            },
            "capacity": {
                "units": "exact bytes; GiB displays use 2^30 bytes",
                "hbm_total_bytes_per_rank": (
                    self.config.hbm_capacity_bytes_per_rank
                ),
                "model_weight_bytes_per_rank_estimate": (
                    self.weight_bytes_per_rank
                ),
                "weight_dtype_bytes": self.config.weight_dtype_bytes,
                "hbm_static_reserve_bytes_per_rank": (
                    self.config.hbm_static_reserve_bytes_per_rank
                ),
                "prefill_hbm_static_reserve_bytes_per_rank": (
                    self.config
                    .effective_prefill_hbm_static_reserve_bytes_per_rank
                ),
                "decode_hbm_static_reserve_bytes_per_rank": (
                    self.config
                    .effective_decode_hbm_static_reserve_bytes_per_rank
                ),
                "role_specific_hbm_reserve_overrides_present": (
                    self.config.pd_disaggregated
                    and (
                        self.config
                        .prefill_hbm_static_reserve_bytes_per_rank is not None
                        or self.config
                        .decode_hbm_static_reserve_bytes_per_rank is not None
                    )
                ),
                "effective_role_hbm_reserves_differ": (
                    self.config.pd_disaggregated
                    and self.config
                    .effective_prefill_hbm_static_reserve_bytes_per_rank
                    != self.config
                    .effective_decode_hbm_static_reserve_bytes_per_rank
                ),
                **capacities,
                "peak_occupancy": peak,
                "peak_fraction": {
                    **hbm_peak_fractions,
                    "cpu": peak["cpu_bytes"] / self.config.cpu_capacity_bytes,
                    "ssd": peak["ssd_bytes"] / self.config.ssd_capacity_bytes,
                },
                "capacity_invariant_checked": True,
            },
            "policy": {
                "name": self.config.policy,
                "cascade": cascade,
                "replacement": "LRU within each idle tier",
                "demotion_mode": self.config.demotion_mode,
                "cpu_cache_enabled": self.config.policy == "tiered",
                "ssd_direct": self.config.policy == "hbm_ssd_direct",
                "ssd_direct_semantics": (
                    "No persistent CPU cache tier; SSD restore still uses a "
                    "transient CPU-DRAM bounce stage before CPU->GPU PCIe."
                    if self.config.policy == "hbm_ssd_direct"
                    else None
                ),
                "hbm_ttl_ns": self.config.hbm_ttl_ns,
                "cpu_ttl_ns": self.config.cpu_ttl_ns,
                "ssd_ttl_ns": self.config.ssd_ttl_ns,
                "migration_commit": (
                    "atomic source-retained background copy with destination "
                    "reservation"
                ),
                "swap_out_execution": (
                    "asynchronous background transfer; never registers a "
                    "model-engine or global execution barrier"
                ),
                "swap_out_contention": (
                    "Background copies still occupy modeled DMA/media/DRAM "
                    "queue resources and retain source capacity until commit."
                ),
                "resume_during_migration": (
                    "cancel_and_use_upper_tier"
                    if self.config.cancel_migration_on_resume
                    else "wait_for_nonpreemptive_commit"
                ),
                "actions_by_reason": dict(sorted(self.policy_actions.items())),
            },
            "resume": {
                "all_request_count": self.workload.calls,
                "all_request_denominator_scope": (
                    "Every LLM invocation in workload sub_requests[], "
                    "including first calls, zero-gap or no-reuse calls, and "
                    "calls excluded by the target-model context limit."
                ),
                "all_selected_positive_transition_count": self.selected_seen,
                "reuse_eligible_transition_count": self.eligible_seen,
                "no_reuse_transition_count": self.source_counts["no_reuse"],
                "source_counts": reported_sources,
                "source_fractions_of_all_requests": {
                    key: value / self.workload.calls
                    if self.workload.calls else 0.0
                    for key, value in reported_sources.items()
                },
                "cpu_or_ssd_resume_count": (
                    reported_sources["cpu"] + reported_sources["ssd"]
                ),
                "cpu_or_ssd_resume_fraction_of_all_requests": (
                    (reported_sources["cpu"] + reported_sources["ssd"])
                    / self.workload.calls
                    if self.workload.calls else 0.0
                ),
                "source_fractions_of_reuse_eligible": {
                    key: value / self.eligible_seen if self.eligible_seen else 0.0
                    for key, value in reported_sources.items()
                },
                "source_fractions_of_all_selected_positive_transitions": {
                    key: value / self.selected_seen if self.selected_seen else 0.0
                    for key, value in reported_sources.items()
                },
                "source_reusable_tokens": reported_source_tokens,
                "source_token_fractions": {
                    key: value / self.total_reusable_tokens
                    if self.total_reusable_tokens else 0.0
                    for key, value in reported_source_tokens.items()
                },
                "by_return_gap_type": source_counts_by_gap,
                "return_gap_semantics": (
                    "The current call inherits return_gap_type/source/ns "
                    "from the preceding sub-request's outgoing inter-turn "
                    "gap. session_start marks first calls; mixed and unknown "
                    "remain separate. Source cells include reuse-eligible "
                    "selected transitions only, and the residual keeps the "
                    "all-request denominator explicit."
                ),
                "ssd_source_reasons": dict(sorted(self.ssd_source_reasons.items())),
                "restore_timing": {
                    "execution_mode": self.config.restore_execution_mode,
                    "canonical_execution_mode": (
                        "async-pre-admission"
                        if self.config.restore_execution_mode
                        == "serial-before-prefill"
                        else self.config.restore_execution_mode
                    ),
                    "issue_epoch": (
                        "The current call's logical request-ready epoch. "
                        "For TraceLab this is tool_result for tool returns "
                        "and user_message arrival for human returns; mixed "
                        "returns remain a separate conservative class. Raw "
                        "elapsed includes later source/HBM admission; DMA "
                        "begins only when those resources are available."
                    ),
                    "join_semantics": (
                        (
                            "Decode is request-locally gated by both complete "
                            "KV restore and analytical pre-restore suffix "
                            "prefill. The final prompt token remains after the "
                            "join. Other calls are not gated. The overlap is "
                            "a perfect-overlap bound, not a kernel-streamed "
                            "dependency model."
                            if self.config.restore_execution_mode
                            == "async-decode-join"
                            else "Destination HBM is reserved before load, but "
                            "the returning request cannot enter analytical "
                            "prompt compute until its complete KV restore "
                            "chain is ready. This is a request-local gate: "
                            "other calls and continuous batches are not gated."
                        )
                    ),
                    "prefill_cutoff_semantics": (
                        (
                            "Execution fresh tokens are input_tokens minus "
                            "exact effective reuse. Only max(0, fresh-1) "
                            "tokens may run before restore; the final prompt "
                            "token remains behind the join."
                            if self.config.restore_execution_mode
                            == "async-decode-join"
                            else "No fresh prompt token from a cold returning "
                            "request executes before the complete restore "
                            "chain. raw_newly_append_toks remains provenance "
                            "only."
                        )
                    ),
                    "request_summed_raw_elapsed_seconds": (
                        self.raw_restore_elapsed_ns / 1e9
                    ),
                    "request_summed_hidden_by_prefill_seconds": (
                        self.restore_hidden_by_prefill_ns / 1e9
                    ),
                    "request_summed_exposed_decode_barrier_seconds": (
                        self.exposed_restore_barrier_ns / 1e9
                    ),
                    "request_summed_exposed_compute_admission_gate_seconds": (
                        self.exposed_restore_barrier_ns / 1e9
                    ),
                    "request_summed_other_concurrent_or_admission_seconds": (
                        self.restore_other_concurrent_or_admission_ns / 1e9
                    ),
                    "raw_decomposition": (
                        "raw_elapsed = prefill_execution_overlap + exposed "
                        "owner gate + other_concurrent_or_admission. In "
                        "async-pre-admission mode the request-local owner gate "
                        "covers the full logical-ready-to-KV-ready interval; "
                        "in async-decode-join it starts at the prefill cutoff."
                    ),
                    "hidden_fraction_of_raw_elapsed": (
                        self.restore_hidden_by_prefill_ns
                        / self.raw_restore_elapsed_ns
                        if self.raw_restore_elapsed_ns else 0.0
                    ),
                    "wall_clock_raw_elapsed_union_seconds": (
                        raw_restore_union_ns / 1e9
                    ),
                    "wall_clock_exposed_decode_barrier_union_seconds": (
                        exposed_restore_union_ns / 1e9
                    ),
                    "by_source": restore_timing_by_source,
                    "by_return_gap_type": restore_timing_by_gap,
                    "by_return_gap_type_and_source": restore_timing_cross,
                },
                "aggregate_restore_stall_seconds": self.restore_stall_ns / 1e9,
                "aggregate_restore_stall_semantics": (
                    "Deprecated compatibility field: request-summed raw "
                    "transfer-stage elapsed time, including queue wait. It "
                    "is fully exposed by the default pre-admission gate but "
                    "may be partly hidden in async-decode-join sensitivity. "
                    "Use restore_timing.request_summed_exposed_compute_"
                    "admission_gate_seconds."
                ),
                "aggregate_inflight_demotion_wait_seconds": (
                    self.resume_inflight_migration_wait_ns / 1e9
                ),
                "no_reuse_inflight_demotion_wait_seconds": (
                    self.no_reuse_inflight_migration_wait_ns / 1e9
                ),
                "aggregate_migration_stall_seconds": (
                    aggregate_migration_stall_ns / 1e9
                ),
                "aggregate_migration_stall_semantics": (
                    "Deprecated compatibility sum of raw transfer-stage time "
                    "and same-object in-flight demotion wait; not causal stall."
                ),
                "migration_stall_fraction_of_request_summed_prompt_active_time": (
                    aggregate_migration_stall_ns
                    / (
                        aggregate_migration_stall_ns
                        + self.prompt_compute_seconds * 1e9
                    )
                    if aggregate_migration_stall_ns
                    + self.prompt_compute_seconds * 1e9
                    else 0.0
                ),
                "migration_stall_fraction_semantics": (
                    "Deprecated compatibility fraction formed from raw "
                    "transfer-stage elapsed plus same-object demotion wait; "
                    "it is not an exposed-delay fraction."
                ),
                "foreground_migration_exposure_union_seconds": (
                    migration_exposure_union_ns / 1e9
                ),
                "foreground_migration_exposure_fraction_of_request_makespan": (
                    migration_exposure_union_ns / request_makespan_ns
                    if request_makespan_ns else 0.0
                ),
                "all_foreground_kv_transfer_exposure_union_seconds": (
                    all_kv_exposure_union_ns / 1e9
                ),
                "all_foreground_kv_transfer_exposure_fraction_of_request_makespan": (
                    all_kv_exposure_union_ns / request_makespan_ns
                    if request_makespan_ns else 0.0
                ),
                "exposure_note": (
                    "Compatibility unions cover raw foreground transfer-chain "
                    "intervals, not necessarily blocked time. Use the exposed "
                    "decode-barrier union under restore_timing for causal "
                    "request-local waiting. Request sums are never divided "
                    "directly by wall-clock makespan."
                ),
                "aggregate_restore_queue_wait_seconds": (
                    self.restore_queue_wait_ns / 1e9
                ),
                "aggregate_hbm_capacity_block_seconds": (
                    self.hbm_capacity_block_ns / 1e9
                ),
                "pd_decode_to_prefill_stall_seconds": self.pd_d2p_stall_ns / 1e9,
                "pd_decode_to_prefill_stall_semantics": (
                    "Deprecated raw D->P stage elapsed, not necessarily "
                    "causal because it may overlap analytical prefill."
                ),
            },
            "recompute": {
                "event_count": self.source_counts["recompute"],
                "event_fraction_of_reuse_eligible_transitions": (
                    self.source_counts["recompute"] / self.eligible_seen
                    if self.eligible_seen else 0.0
                ),
                "event_fraction_of_all_selected_positive_transitions": (
                    self.source_counts["recompute"] / self.selected_seen
                    if self.selected_seen else 0.0
                ),
                "tokens": self.recompute_tokens,
                "total_reusable_tokens_requested": self.total_reusable_tokens,
                "token_fraction_of_reusable_tokens_requested": (
                    self.recompute_tokens / self.total_reusable_tokens
                    if self.total_reusable_tokens else 0.0
                ),
                "analytical_seconds": self.recompute_seconds,
                "numerator_scope": (
                    "Counterfactual incremental prompt-compute seconds "
                    "attributable to the lost reusable prefix: max(0, "
                    "full-prompt prediction minus cached-prefix prediction), "
                    "capped at the full-prompt prediction. This is not the "
                    "entire full-prompt compute time."
                ),
                "analytical_prompt_compute_seconds_executed": (
                    self.prompt_compute_seconds
                ),
                "analytical_time_fraction_of_executed_prompt_compute": (
                    self.recompute_seconds / self.prompt_compute_seconds
                    if self.prompt_compute_seconds else 0.0
                ),
                "full_prompt_reference_seconds": (
                    self.full_prompt_reference_seconds
                ),
                "reasons": dict(sorted(self.recompute_reasons.items())),
                "time_denominator_scope": (
                    "All seconds returned by the configured prompt-compute "
                    "model for context-eligible calls, including recomputed "
                    "prefix. This includes any kernel and collective terms "
                    "returned by that model; KV transfer, decode, host, "
                    "scheduler, and batch-formation time are excluded."
                ),
            },
            "transfer_queue": queue_report,
            "admission_queues": {
                "decode_restore_hbm": {
                    "discipline": (
                        "FCFS with completion-safe zero-growth resident-HBM "
                        "backfill exception"
                    ),
                    "resource_scope": (
                        "decode_hbm final-footprint reservation plus optional "
                        "lower-tier prefix restore"
                    ),
                    "enqueue_count": self.decode_restore_enqueue_count,
                    "capacity_retry_count": (
                        self.decode_restore_capacity_retry_count
                    ),
                    "fcfs_bypass_deferral_count": (
                        self.decode_restore_fcfs_deferral_count
                    ),
                    "wakeup_count": self.decode_restore_wakeup_count,
                    "wakeup_event_count": (
                        self.decode_restore_wakeup_event_count
                    ),
                    "max_depth": self.decode_restore_max_depth,
                    "max_foreground_source_pins": (
                        self.decode_restore_max_source_pins
                    ),
                    "source_ttl_deferral_count": (
                        self.decode_restore_source_ttl_deferral_count
                    ),
                    "aggregate_admission_wait_seconds": (
                        self.decode_restore_capacity_block_ns / 1e9
                    ),
                    "aggregate_capacity_block_seconds": (
                        self.decode_restore_capacity_block_ns / 1e9
                    ),
                    "aggregate_capacity_block_seconds_semantics": (
                        "Deprecated compatibility alias for aggregate admission "
                        "wait; includes FCFS, source-transit, and admission time, "
                        "not only physical capacity shortage."
                    ),
                    "independent_from_prefill_hbm_queue": True,
                },
                "pd_branch_admission_safety": {
                    "discipline": (
                        "branch-local logical-ready FCFS with "
                        "completion-safe HBM backfill"
                    ),
                    "purpose": (
                        "Avoid cross-pool hold-and-wait without globally "
                        "blocking independent younger P work."
                    ),
                    "source_pin_epoch": "decode-admission head",
                    "resident_hbm_backfill": (
                        "zero D growth and immediately admissible P branch"
                    ),
                    "final_decode_footprint_pre_reserved": True,
                    "speculative_prefill_wasted_seconds": (
                        self.pd_speculative_prefill_wasted_seconds
                    ),
                },
            },
            "ssd_io": {
                "issued_full_object_write_bytes": ssd_write_bytes,
                "issued_read_bytes": ssd_read_bytes,
                "restore_path": (
                    "SSD -> transient CPU DRAM -> destination GPU HBM; "
                    "P/D then performs decode-HBM -> prefill-HBM"
                ),
                "restore_stage_kinds": {
                    "ssd_to_cpu": [
                        "ssd_to_cpu_stage_for_hbm",
                        "ssd_to_cpu_stage_for_decode",
                    ],
                    "cpu_to_gpu": [
                        "cpu_stage_to_hbm",
                        "cpu_stage_to_decode",
                    ],
                },
                "restore_stage_metrics_location": (
                    "transfer_queue.queue_wait_seconds_by_kind, "
                    "transfer_queue.service_seconds_by_kind, and "
                    "transfer_queue.bytes_by_kind"
                ),
                "transient_cpu_stage_counts_as_cpu_cache_occupancy": False,
                "read_semantics": (
                    "Every SSD restore is two serial queue jobs. The first "
                    "lands the full block-rounded object in transient CPU "
                    "DRAM; the second reads that staging buffer and traverses "
                    "the per-GPU CPU-to-GPU PCIe links. Destination HBM is "
                    "reserved before the first stage."
                ),
                "completed_full_object_write_bytes": (
                    ssd_write_bytes
                    if not self.config.cancel_migration_on_resume
                    else None
                ),
                "write_semantics": (
                    (
                        "Every direct HBM->SSD job writes one full "
                        "block-rounded object without CPU DRAM staging."
                        if self.config.policy == "hbm_ssd_direct"
                        else "Every default noncancellable CPU->SSD or "
                        "oversize HBM->SSD job writes a full block-rounded "
                        "object."
                    )
                    + " Cancellable sensitivity does not attribute partial "
                    "media progress."
                ),
            },
            "pd_transfer": {
                "enabled": self.config.pd_disaggregated,
                "link_gbps_per_rank": self.config.pd_link_gbps_per_rank,
                "fixed_latency_us": self.config.pd_fixed_latency_us,
                "decode_to_prefill_bytes": self.pd_d2p_bytes,
                "decode_to_prefill_raw_elapsed_seconds": (
                    self.pd_d2p_stall_ns / 1e9
                ),
                "decode_to_prefill_stall_seconds": self.pd_d2p_stall_ns / 1e9,
                "decode_to_prefill_stall_semantics": (
                    "Deprecated compatibility alias for raw stage elapsed; "
                    "consult resume.restore_timing for exposed delay."
                ),
                "prefill_to_decode_bytes": self.pd_p2d_bytes,
                "prefill_to_decode_handoff_seconds": self.pd_p2d_handoff_ns / 1e9,
                "prefill_to_decode_queue_wait_seconds": (
                    self.pd_p2d_queue_wait_ns / 1e9
                ),
                "handoff_scope": (
                    "Successful reuse keeps a decode-owned prefix and sends "
                    "only the exact suffix P->D; recompute sends the full prompt."
                ),
            },
            "offered_load_call_activity": {
                "window_start_seconds": first_arrival_ns / 1e9,
                "window_end_seconds": self.request_trace_end_ns / 1e9,
                "window_seconds": request_makespan_ns / 1e9,
                "wall_clock_with_at_least_one_active_call_seconds": (
                    offered_call_active_union_ns / 1e9
                ),
                "wall_clock_with_no_active_call_seconds": (
                    offered_call_idle_complement_ns / 1e9
                ),
                "no_active_call_fraction": (
                    offered_call_idle_complement_ns / request_makespan_ns
                    if request_makespan_ns else 0.0
                ),
                "is_server_utilization": False,
                "interpretation": (
                    "Union/complement of analytical call ready-to-complete "
                    "intervals under the trace's open-loop first-session "
                    "arrivals and closed-loop inter-turn gaps. Because calls "
                    "overlap without a shared compute queue or batching, this "
                    "is offered call activity, not GPU/server utilization."
                ),
            },
            "trace_end_seconds": self.request_trace_end_ns / 1e9,
            "request_makespan_seconds": request_makespan_ns / 1e9,
            "background_event_horizon_seconds": self.event_horizon_ns / 1e9,
            "context_infeasible_calls": self.context_infeasible_calls,
            "warnings": [
                "Reuse provenance is reported in workload.reuse_source_counts; "
                "only token_ids_exact is a target-tokenizer-exact LCP, while "
                "explicit_estimated and explicit_reported retain their declared "
                "semantics.",
                "A call is over context when input_toks + output_toks exceeds "
                "max_context_tokens. Such calls advance their tool gaps and "
                "clear cache lineage, but their target-model execution time "
                "is unavailable and set to zero.",
                "Model weights are architecture-derived estimates; replace them with measured per-rank engine residency for headline results.",
                *zero_reserve_warnings,
                "All SSD restores use explicit serial SSD->CPU-DRAM and "
                "CPU-DRAM->GPU-PCIe queue stages. hbm_ssd_direct means no "
                "persistent CPU cache tier, not a GDS-style read path.",
                (
                    "Cache-hit compute uses the same calibrated analytical "
                    "kernel model as full recomputation, evaluated only on "
                    "the uncached suffix and its cached attention prefix."
                    if self.prompt_compute_model is not None
                    else "Cache-hit compute uses an analytical suffix-prefill "
                    "roofline with causal attention delta and retained "
                    "weight/launch costs; it is not a measured kernel replay."
                ),
                *(
                    [
                        "The prompt predictor is calibrated from component "
                        "measurements but remains an analytical target-model "
                        "prediction; it is not a measured end-to-end Qwen run."
                    ]
                    if self.prompt_compute_model is not None
                    else []
                ),
                *(
                    [
                        "Prompt compute is multiplied by a non-identity "
                        "sensitivity scale. It is not a measured DCA kernel "
                        "profile or a statistical confidence bound."
                    ]
                    if self.config.prompt_compute_scale != 1.0
                    else []
                ),
                *(
                    [
                        "Async restore/prefill overlap is an optimistic "
                        "sensitivity. Real suffix attention depends on prefix "
                        "KV unless a measured layer-streamed implementation "
                        "establishes that overlap."
                    ]
                    if self.config.restore_execution_mode
                    == "async-decode-join"
                    else []
                ),
                "Background swap-out never creates a global engine barrier, "
                "but a resume of that same source-retained object waits for "
                "its nonpreemptive demotion unless cancellation is enabled.",
                "The conservative noncancellable baseline waits for an in-flight old-object demotion even when the next call reports zero reusable prefix; this wait is reported separately.",
            ],
        }


def replay_capacity_aware(
    workload: CapacityReplayWorkload,
    model: ModelShape,
    hardware: HardwareSpec,
    tp_size: int,
    kv_dtype_bytes: int,
    config: CapacityReplayConfig,
    prompt_compute_model: PromptComputeModel | None = None,
) -> dict[str, Any]:
    """Run one capacity-aware global replay and return a JSON-ready report."""

    return _CapacityReplay(
        workload,
        model,
        hardware,
        tp_size,
        kv_dtype_bytes,
        config,
        prompt_compute_model,
    ).run()


def infinite_hbm_oracle_capacity(
    workload: CapacityReplayWorkload,
    model: ModelShape,
    tp_size: int,
    kv_dtype_bytes: int,
    config: CapacityReplayConfig,
) -> dict[str, int | str]:
    """Construct an auditable, nonbinding HBM residency capacity.

    Dependency chains permit at most one active call and one decode-owned KV
    object per session. Summing each session's largest possible object is
    therefore a safe simultaneous-residency bound. P/D uses independent HBM
    pools; a colocated run uses the larger per-session object because active
    and idle states are mutually exclusive within one session.
    """

    layout = kv_layout(model, tp_size, kv_dtype_bytes)
    bytes_per_token = layout.physical_bytes_per_token_per_rank

    def allocation_bytes(tokens: int) -> int:
        if tokens <= 0:
            return 0
        rounded = math.ceil(tokens / config.block_size) * config.block_size
        return rounded * bytes_per_token

    prefill_bound = 0
    decode_bound = 0
    colocated_bound = 0
    for session in workload.sessions:
        prefill_max = max(
            (
                allocation_bytes(call.input_tokens)
                for call in session.calls
                if call.context_eligible
            ),
            default=0,
        )
        # Even the final call performs the modeled P->D handoff before call
        # completion. Include every executable call, not only cacheable idle
        # objects, so the receive reservation is also provably nonbinding.
        decode_max = max(
            (
                allocation_bytes(call.cache_tokens)
                for call in session.calls
                if call.context_eligible
            ),
            default=0,
        )
        prefill_bound += prefill_max
        decode_bound += decode_max
        colocated_bound += max(prefill_max, decode_max)

    kv_budget = (
        max(prefill_bound, decode_bound)
        if config.pd_disaggregated
        else colocated_bound
    )
    weight_bytes = estimate_model_weight_bytes_per_rank(
        model,
        tp_size,
        config.weight_dtype_bytes,
        layout,
    )
    prefill_total_capacity = (
        weight_bytes
        + config.effective_prefill_hbm_static_reserve_bytes_per_rank
        + prefill_bound
    )
    decode_total_capacity = (
        weight_bytes
        + config.effective_decode_hbm_static_reserve_bytes_per_rank
        + decode_bound
    )
    total_capacity = (
        max(prefill_total_capacity, decode_total_capacity)
        if config.pd_disaggregated
        else (
            weight_bytes
            + config.hbm_static_reserve_bytes_per_rank
            + colocated_bound
        )
    )
    return {
        "construction": (
            "sum_of_per_session_maxima_for_independent_prefill_and_decode_"
            "pools" if config.pd_disaggregated
            else "sum_of_per_session_max_prefill_or_decode_object"
        ),
        "physical_bytes_per_token_per_rank": bytes_per_token,
        "prefill_kv_bound_bytes_per_rank": prefill_bound,
        "decode_kv_bound_bytes_per_rank": decode_bound,
        "colocated_kv_bound_bytes_per_rank": colocated_bound,
        "selected_kv_budget_bytes_per_rank": kv_budget,
        "model_weight_bytes_per_rank_estimate": weight_bytes,
        "hbm_static_reserve_bytes_per_rank": (
            config.hbm_static_reserve_bytes_per_rank
        ),
        "prefill_hbm_static_reserve_bytes_per_rank": (
            config.effective_prefill_hbm_static_reserve_bytes_per_rank
        ),
        "decode_hbm_static_reserve_bytes_per_rank": (
            config.effective_decode_hbm_static_reserve_bytes_per_rank
        ),
        "prefill_total_hbm_capacity_bytes_per_rank": (
            prefill_total_capacity
        ),
        "decode_total_hbm_capacity_bytes_per_rank": decode_total_capacity,
        "total_hbm_capacity_bytes_per_rank": total_capacity,
    }


def _paired_group_summary(
    keys: list[tuple[str, int]],
    finite: _CapacityReplay,
    oracle: _CapacityReplay,
) -> dict[str, Any]:
    finite_latencies = [
        finite.call_completion_ns[key] - finite.call_logical_ready_ns[key]
        for key in keys
    ]
    oracle_latencies = [
        oracle.call_completion_ns[key] - oracle.call_logical_ready_ns[key]
        for key in keys
    ]
    deltas = [
        finite_ns - oracle_ns
        for finite_ns, oracle_ns in zip(finite_latencies, oracle_latencies)
    ]
    ratios = [
        finite_ns / oracle_ns - 1.0
        for finite_ns, oracle_ns in zip(finite_latencies, oracle_latencies)
        if oracle_ns > 0
    ]
    finite_sum = sum(finite_latencies)
    oracle_sum = sum(oracle_latencies)
    completion_deltas = [
        finite.call_completion_ns[key] - oracle.call_completion_ns[key]
        for key in keys
    ]
    return {
        "call_count": len(keys),
        "positive_oracle_service_call_count": len(ratios),
        "finite_request_summed_ready_to_complete_seconds": finite_sum / 1e9,
        "oracle_request_summed_ready_to_complete_seconds": oracle_sum / 1e9,
        "delta_request_summed_ready_to_complete_seconds": (
            finite_sum - oracle_sum
        ) / 1e9,
        "slowdown_fraction_of_oracle_request_summed_service": (
            finite_sum / oracle_sum - 1.0 if oracle_sum else None
        ),
        "finite_mean_ready_to_complete_ms": (
            finite_sum / len(keys) / 1e6 if keys else None
        ),
        "oracle_mean_ready_to_complete_ms": (
            oracle_sum / len(keys) / 1e6 if keys else None
        ),
        "paired_service_delta_ms": {
            name: (
                None if value is None else value / 1e6
            )
            for name, value in (
                ("p50", _quantile(deltas, 0.50)),
                ("p90", _quantile(deltas, 0.90)),
                ("p95", _quantile(deltas, 0.95)),
                ("p99", _quantile(deltas, 0.99)),
            )
        },
        "paired_service_slowdown_fraction": {
            name: value
            for name, value in (
                ("p50", _quantile(ratios, 0.50)),
                ("p90", _quantile(ratios, 0.90)),
                ("p95", _quantile(ratios, 0.95)),
                ("p99", _quantile(ratios, 0.99)),
            )
        },
        "paired_absolute_completion_delta_ms": {
            name: (
                None if value is None else value / 1e6
            )
            for name, value in (
                ("p50", _quantile(completion_deltas, 0.50)),
                ("p90", _quantile(completion_deltas, 0.90)),
                ("p95", _quantile(completion_deltas, 0.95)),
                ("p99", _quantile(completion_deltas, 0.99)),
            )
        },
        "finite_faster_call_count": sum(delta < 0 for delta in deltas),
        "equal_call_count": sum(delta == 0 for delta in deltas),
        "finite_slower_call_count": sum(delta > 0 for delta in deltas),
    }


def _paired_oracle_comparison(
    finite: _CapacityReplay,
    finite_report: Mapping[str, Any],
    oracle: _CapacityReplay,
    oracle_report: Mapping[str, Any],
    capacity: Mapping[str, Any],
) -> dict[str, Any]:
    finite_keys = set(finite.call_completion_ns)
    oracle_keys = set(oracle.call_completion_ns)
    expected_keys = {
        (session.session_id, index)
        for session in finite.workload.sessions
        for index in range(len(session.calls))
    }
    if finite_keys != expected_keys or oracle_keys != expected_keys:
        raise AssertionError("paired oracle did not complete every workload call")
    if set(finite.call_logical_ready_ns) != expected_keys:
        raise AssertionError("finite replay is missing logical-ready epochs")
    if set(oracle.call_logical_ready_ns) != expected_keys:
        raise AssertionError("oracle replay is missing logical-ready epochs")

    ordered_keys = sorted(expected_keys)
    all_calls = _paired_group_summary(ordered_keys, finite, oracle)

    source_label = (
        "decode_hbm" if finite.config.pd_disaggregated else "hbm"
    )

    def public_source(source: str) -> str:
        return source_label if source == "hbm" else source

    by_source: dict[str, Any] = {}
    by_gap: dict[str, Any] = {}
    cross: dict[str, dict[str, Any]] = {}
    for key in ordered_keys:
        session_id, call_index = key
        call = finite.sessions[session_id].calls[call_index]
        source = public_source(finite.call_sources[key])
        by_source.setdefault(source, []).append(key)
        by_gap.setdefault(call.return_gap_type, []).append(key)
        cross.setdefault(call.return_gap_type, {}).setdefault(
            source, []
        ).append(key)
    by_source_report = {
        source: _paired_group_summary(keys, finite, oracle)
        for source, keys in sorted(by_source.items())
    }
    by_gap_report = {
        gap: _paired_group_summary(keys, finite, oracle)
        for gap, keys in sorted(by_gap.items())
    }
    cross_report = {
        gap: {
            source: _paired_group_summary(keys, finite, oracle)
            for source, keys in sorted(sources.items())
        }
        for gap, sources in sorted(cross.items())
    }

    finite_session_e2e_ns = 0
    oracle_session_e2e_ns = 0
    finite_final_ns: list[int] = []
    oracle_final_ns: list[int] = []
    for session in finite.workload.sessions:
        last_key = (session.session_id, len(session.calls) - 1)
        finite_final = finite.call_completion_ns[last_key]
        oracle_final = oracle.call_completion_ns[last_key]
        finite_final_ns.append(finite_final)
        oracle_final_ns.append(oracle_final)
        finite_session_e2e_ns += finite_final - session.arrival_time_ns
        oracle_session_e2e_ns += oracle_final - session.arrival_time_ns

    first_arrival = min(
        session.arrival_time_ns for session in finite.workload.sessions
    )
    finite_makespan = max(finite_final_ns) - first_arrival
    oracle_makespan = max(oracle_final_ns) - first_arrival
    session_delta = finite_session_e2e_ns - oracle_session_e2e_ns
    call_service_delta = sum(
        (
            finite.call_completion_ns[key]
            - finite.call_logical_ready_ns[key]
        )
        - (
            oracle.call_completion_ns[key]
            - oracle.call_logical_ready_ns[key]
        )
        for key in ordered_keys
    )
    if session_delta != call_service_delta:
        raise AssertionError(
            "closed-loop delay conservation failed: fixed gaps should make "
            "session-E2E delta equal call-service delta"
        )

    return {
        "reference": "paired_infinite_hbm_residency",
        "reference_is_strict_per_call_lower_bound": False,
        "same_workload_and_first_call_arrivals": True,
        "later_call_ready_times_are_endogenous": True,
        "same_gap_durations": True,
        "same_pd_topology_and_mandatory_transfers": True,
        "same_roofline_compute_model": True,
        "same_prompt_compute_model": True,
        "prompt_compute_model_kind": finite_report["execution_scope"][
            "prompt_compute_model_kind"
        ],
        "same_restore_execution_mode": True,
        "restore_execution_mode": finite.config.restore_execution_mode,
        "same_independent_pd_branch_admission": True,
        "same_final_decode_footprint_prereservation": True,
        "compute_queue_or_batching_modeled": False,
        "mixed_batch_restore_barrier_modeled": False,
        "oracle_capacity_construction": dict(capacity),
        "all_calls": all_calls,
        "by_finite_source": by_source_report,
        "by_return_gap_type": by_gap_report,
        "by_return_gap_type_and_finite_source": cross_report,
        "session_end_to_end": {
            "session_count": len(finite.workload.sessions),
            "finite_request_summed_arrival_to_final_completion_seconds": (
                finite_session_e2e_ns / 1e9
            ),
            "oracle_request_summed_arrival_to_final_completion_seconds": (
                oracle_session_e2e_ns / 1e9
            ),
            "delta_seconds": session_delta / 1e9,
            "slowdown_fraction_of_oracle": (
                finite_session_e2e_ns / oracle_session_e2e_ns - 1.0
                if oracle_session_e2e_ns else None
            ),
        },
        "trace_makespan": {
            "finite_seconds": finite_makespan / 1e9,
            "oracle_seconds": oracle_makespan / 1e9,
            "delta_seconds": (finite_makespan - oracle_makespan) / 1e9,
            "slowdown_fraction_of_oracle": (
                finite_makespan / oracle_makespan - 1.0
                if oracle_makespan else None
            ),
        },
        "closed_loop_delay_conservation_checked": True,
        "oracle_validation": {
            "eligible_source_counts": oracle_report["resume"][
                "source_counts"
            ],
            "capacity_action_count": sum(
                oracle_report["policy"]["actions_by_reason"].values()
            ),
            "aggregate_hbm_capacity_block_seconds": oracle_report["resume"][
                "aggregate_hbm_capacity_block_seconds"
            ],
            "capacity_invariant_checked": oracle_report["capacity"][
                "capacity_invariant_checked"
            ],
        },
        "interpretation": (
            "Primary service slowdown compares each call's first logical-ready "
            "epoch with its completion and therefore includes restore, HBM "
            "admission, P/D transfer, recomputation, and their closed-loop "
            "propagation without diluting them by fixed human/tool gaps. The "
            "trace-makespan and session-E2E views intentionally include those "
            "gaps. A finite call can beat the residency reference because the "
            "reference may send more D->P hit traffic and alter shared-fabric "
            "ordering; it is not a strict per-call lower bound."
        ),
        "finite_report_scope": finite_report["execution_scope"],
    }


def replay_capacity_aware_with_oracle(
    workload: CapacityReplayWorkload,
    model: ModelShape,
    hardware: HardwareSpec,
    tp_size: int,
    kv_dtype_bytes: int,
    config: CapacityReplayConfig,
    prompt_compute_model: PromptComputeModel | None = None,
) -> dict[str, Any]:
    """Run a finite replay and its paired nonbinding-HBM residency reference."""

    finite = _CapacityReplay(
        workload,
        model,
        hardware,
        tp_size,
        kv_dtype_bytes,
        config,
        prompt_compute_model,
    )
    finite_report = finite.run()
    capacity = infinite_hbm_oracle_capacity(
        workload, model, tp_size, kv_dtype_bytes, config
    )
    oracle_config = replace(
        config,
        hbm_capacity_bytes_per_rank=int(
            capacity["total_hbm_capacity_bytes_per_rank"]
        ),
        policy="hbm_lru_recompute",
        demotion_mode="capacity-only",
    )
    oracle = _CapacityReplay(
        workload,
        model,
        hardware,
        tp_size,
        kv_dtype_bytes,
        oracle_config,
        prompt_compute_model,
    )
    oracle_report = oracle.run()
    oracle_hbm_key = "decode_hbm" if config.pd_disaggregated else "hbm"
    oracle_sources = oracle_report["resume"]["source_counts"]
    if (
        oracle_sources["cpu"]
        or oracle_sources["ssd"]
        or oracle_sources["recompute"]
        or oracle_sources[oracle_hbm_key]
        != oracle_report["resume"]["reuse_eligible_transition_count"]
    ):
        raise AssertionError("infinite-HBM reference had a non-HBM eligible source")
    if oracle_report["policy"]["actions_by_reason"]:
        raise AssertionError("infinite-HBM reference triggered a capacity action")
    if oracle_report["resume"]["aggregate_hbm_capacity_block_seconds"]:
        raise AssertionError("infinite-HBM reference blocked on HBM capacity")

    finite_report["schema_version"] = 15
    finite_report["infinite_hbm_oracle_comparison"] = (
        _paired_oracle_comparison(
            finite, finite_report, oracle, oracle_report, capacity
        )
    )
    return finite_report


def write_capacity_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

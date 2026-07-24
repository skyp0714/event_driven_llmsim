"""Agent-aware KV-cache tiering for closed-loop tool-calling workloads.

This module deliberately models *idle session state*, not the generic radix
prefix cache.  A completed agent turn transfers ownership of its KV blocks
from the scheduler to :class:`AgenticKVManager`; the next turn transfers the
reusable prefix back.  Keeping the two ownership domains separate prevents
the same physical KV blocks from being counted twice.

The online policies never inspect the future tool duration when deciding
whether to demote a cache.  The known release timestamp is used only by the
event simulator to stop background work when the tool actually completes.
"""

from __future__ import annotations

import copy
import json
import heapq
import math
import os
import bisect
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Dict, Iterable, Optional, Sequence

from .memory_model import Device


NS_PER_SECOND = 1_000_000_000
DECIMAL_GB = 1_000_000_000

_KV_DROP_CLASS_BY_REASON = {
    "hbm_capacity": "capacity_loss",
    "cpu_capacity": "capacity_loss",
    "ssd_capacity": "capacity_loss",
    "ssd_ttl": "ttl_loss",
    "resume_miss": "resume_recompute_cleanup",
    "queue_pressure": "policy_loss",
    "session_end": "normal_session_cleanup",
    "measurement_censor": "measurement_cleanup",
}

_KV_DROP_CLASS_SEMANTICS = {
    "capacity_loss": "Reusable KV was discarded because a tier was full.",
    "ttl_loss": "Reusable KV was discarded by an age/TTL sensitivity policy.",
    "resume_recompute_cleanup": (
        "An obsolete KV copy was released when the returning request used "
        "recomputation fallback."
    ),
    "policy_loss": (
        "Reusable KV was deliberately discarded by an explicit online "
        "policy decision."
    ),
    "normal_session_cleanup": (
        "KV was released after the logical session had completed normally."
    ),
    "measurement_cleanup": (
        "KV was released while an unfinished logical session was explicitly "
        "right-censored at the measurement cutoff."
    ),
}


class KVLocation(str, Enum):
    HBM = "hbm"
    CPU = "cpu"
    SSD = "ssd"
    DROPPED = "dropped"


@dataclass
class AgenticKVConfig:
    """Configuration for online session-KV management.

    Bandwidths use decimal GB/s, matching hardware data sheets.  Capacity is
    expressed in decimal GB for SSDs; HBM and CPU capacities come from the
    cluster configuration and remain in the repository's existing GiB units.
    """

    policy: str = "off"
    hbm_ttl_ms: float = 50.0
    cpu_ttl_ms: float = 30_000.0
    ssd_ttl_ms: float = 3_600_000.0
    # PCIe bandwidth is per accelerator. CPU bandwidth is the aggregate
    # bandwidth of the shared host-memory path; SSD bandwidth is aggregate
    # across one node-local configured pool. ``ssd_capacity_gb`` is per
    # device and ``ssd_num_devices`` is the device count per host node.
    pcie_bandwidth_gbps: float = 50.0
    cpu_bandwidth_gbps: float = 200.0
    cpu_transfer_latency_us: float = 5.0
    # An HBM-resident D->P resume can either stage through host memory or use
    # an explicit same-node accelerator fabric. Lower-tier hits restore into P
    # directly. Peer bandwidth is per accelerator because TP ranks transfer
    # concurrently; the node fabric remains a shared queue resource so
    # independent cold P/D copies still contend.
    pd_peer_transfer_mode: str = "cpu-staged"
    pd_peer_bandwidth_gbps: float = 450.0
    pd_peer_latency_us: float = 3.0
    ssd_read_bandwidth_gbps: float = 13.0
    ssd_write_bandwidth_gbps: float = 6.6
    ssd_read_latency_us: float = 20.0
    ssd_write_latency_us: float = 20.0
    ssd_capacity_gb: float = 15_360.0
    ssd_num_devices: int = 1
    ssd_write_mode: str = "incremental"
    block_size: int = 16
    pressure_policy: str = "lru-drop"
    keep_ssd_copy_on_read: bool = True
    io_queue_policy: str = "gang-fcfs"
    # ``capacity-only`` disables age/TTL-triggered movement. Entries move only
    # when a higher tier must free space, which makes the three paper
    # baselines differ by storage hierarchy rather than an arbitrary timeout.
    demotion_mode: str = "ttl-and-capacity"
    # Active-request preemption is held constant across paper baselines so the
    # experiment isolates idle/cold session-KV placement. Non-agentic runs
    # retain Scheduler's legacy cpu-swap default.
    active_preemption_mode: str = "recompute"
    # ``async-pre-admission`` reserves the destination HBM before issuing a
    # restore and keeps the returning request out of compute batches until all
    # reusable KV has arrived. Unrelated continuous batches remain runnable.
    # The overlap and synchronous modes are retained as sensitivity baselines.
    swap_execution_mode: str = "async-pre-admission"
    # The queue-aware paper baseline retains ordinary HBM->CPU->SSD placement
    # but may recompute a returning lower-tier prefix when immutable queued
    # transfer work exceeds both this multiple of the restore's own service
    # time and an absolute minimum. The optional cost guard compares that
    # restore projection with a causal singleton-prefill estimate from the
    # same online COMP-node provider used by the simulation.
    queue_recompute_wait_service_ratio: float = 1.0
    queue_recompute_min_wait_ms: float = 0.0
    queue_recompute_cost_guard_multiplier: float = 0.0
    # A partial-prefix decision is useful only if the first normal prefill
    # chunk can plausibly follow it.  The policy therefore snapshots enough
    # unreserved P and D HBM for this many next-chunk equivalents.  This is a
    # causal observation, not a reservation; the ordinary P/D admission path
    # remains the sole owner of active-request HBM claims.
    queue_recompute_prefill_headroom_chunks: float = 1.0

    @classmethod
    def from_json(cls, path: str, policy: Optional[str] = None) -> "AgenticKVConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown agentic-KV config field(s): {unknown}")
        cfg = cls(**raw)
        if policy is not None:
            cfg.policy = policy
        cfg.validate()
        return cfg

    def validate(self) -> None:
        policies = {
            "off", "preserve", "recompute", "hbm_lru_recompute",
            "hbm_ssd_direct", "cpu", "tiered",
            "tiered_queue_recompute",
        }
        if self.policy not in policies:
            raise ValueError(
                f"agentic KV policy must be one of {sorted(policies)}, got {self.policy!r}"
            )
        if self.ssd_write_mode not in {"full", "incremental"}:
            raise ValueError("ssd_write_mode must be 'full' or 'incremental'")
        if self.pressure_policy != "lru-drop":
            raise ValueError("Only the reproducible lru-drop pressure policy is supported")
        if self.io_queue_policy != "gang-fcfs":
            raise ValueError(
                "Only the deterministic gang-fcfs I/O queue policy is supported")
        if self.demotion_mode not in {"ttl-and-capacity", "capacity-only"}:
            raise ValueError(
                "demotion_mode must be 'ttl-and-capacity' or 'capacity-only'")
        if self.active_preemption_mode not in {"recompute", "cpu-swap"}:
            raise ValueError(
                "active_preemption_mode must be 'recompute' or 'cpu-swap'")
        if self.swap_execution_mode not in {
                "async-decode-join", "async-pre-admission",
                "sync-engine-barrier"}:
            raise ValueError(
                "swap_execution_mode must be 'async-decode-join', "
                "'async-pre-admission', or 'sync-engine-barrier'")
        if self.pd_peer_transfer_mode not in {
                "cpu-staged", "direct-fabric"}:
            raise ValueError(
                "pd_peer_transfer_mode must be 'cpu-staged' or "
                "'direct-fabric'")
        positive_reals = {
            "pcie_bandwidth_gbps": self.pcie_bandwidth_gbps,
            "cpu_bandwidth_gbps": self.cpu_bandwidth_gbps,
            "pd_peer_bandwidth_gbps": self.pd_peer_bandwidth_gbps,
            "ssd_read_bandwidth_gbps": self.ssd_read_bandwidth_gbps,
            "ssd_write_bandwidth_gbps": self.ssd_write_bandwidth_gbps,
            "ssd_capacity_gb": self.ssd_capacity_gb,
        }
        invalid = [
            name for name, value in positive_reals.items()
            if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0)
        ]
        if invalid:
            raise ValueError(
                "Agentic-KV real values must be positive and finite: "
                f"{invalid}")
        positive_integers = {
            "ssd_num_devices": self.ssd_num_devices,
            "block_size": self.block_size,
        }
        invalid = [
            name for name, value in positive_integers.items()
            if (isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0)
        ]
        if invalid:
            raise ValueError(
                "Agentic-KV integer values must be positive integers: "
                f"{invalid}")
        nonnegative = {
            "hbm_ttl_ms": self.hbm_ttl_ms,
            "cpu_ttl_ms": self.cpu_ttl_ms,
            "ssd_ttl_ms": self.ssd_ttl_ms,
            "cpu_transfer_latency_us": self.cpu_transfer_latency_us,
            "pd_peer_latency_us": self.pd_peer_latency_us,
            "ssd_read_latency_us": self.ssd_read_latency_us,
            "ssd_write_latency_us": self.ssd_write_latency_us,
            "queue_recompute_min_wait_ms": (
                self.queue_recompute_min_wait_ms),
            "queue_recompute_cost_guard_multiplier": (
                self.queue_recompute_cost_guard_multiplier),
            "queue_recompute_prefill_headroom_chunks": (
                self.queue_recompute_prefill_headroom_chunks),
        }
        invalid = [
            name for name, value in nonnegative.items()
            if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0)
        ]
        if invalid:
            raise ValueError(
                "Agentic-KV real values must be non-negative and finite: "
                f"{invalid}")
        if (0 < self.queue_recompute_cost_guard_multiplier < 1):
            raise ValueError(
                "queue_recompute_cost_guard_multiplier must be 0 "
                "(disabled) or at least 1")
        if self.queue_recompute_prefill_headroom_chunks < 1:
            raise ValueError(
                "queue_recompute_prefill_headroom_chunks must be at least 1")
        if not isinstance(self.keep_ssd_copy_on_read, bool):
            raise ValueError("keep_ssd_copy_on_read must be boolean")
        ratio = self.queue_recompute_wait_service_ratio
        if (isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
                or not math.isfinite(ratio)
                or ratio < 0):
            raise ValueError(
                "queue_recompute_wait_service_ratio must be finite and "
                "non-negative")

    @property
    def enabled(self) -> bool:
        return self.policy != "off"

    @property
    def age_demotion_enabled(self) -> bool:
        return self.demotion_mode == "ttl-and-capacity"

    @property
    def tiered_family(self) -> bool:
        return self.policy in {"tiered", "tiered_queue_recompute"}

    @property
    def queue_recompute_enabled(self) -> bool:
        return self.policy == "tiered_queue_recompute"

    @property
    def queue_recompute_min_wait_ns(self) -> int:
        return int(math.ceil(self.queue_recompute_min_wait_ms * 1_000_000))

    @property
    def hbm_ttl_ns(self) -> int:
        return int(self.hbm_ttl_ms * 1_000_000)

    @property
    def cpu_ttl_ns(self) -> int:
        return int(self.cpu_ttl_ms * 1_000_000)

    @property
    def ssd_ttl_ns(self) -> int:
        return int(self.ssd_ttl_ms * 1_000_000)

    @property
    def ssd_capacity_bytes(self) -> int:
        return int(self.ssd_capacity_gb * DECIMAL_GB * self.ssd_num_devices)


@dataclass
class IdleKVEntry:
    session_id: str
    instance_id: int
    tokens: int
    block_tokens: int
    per_rank_bytes: int
    total_bytes: int
    location: KVLocation
    tier_since_ns: int
    last_access_ns: int
    incremental_base_tokens: int = 0
    next_use_ns: Optional[int] = None
    hbm_ssd_start_ns: Optional[int] = None
    migration_kind: Optional[str] = None
    migration_start_ns: Optional[int] = None
    migration_complete_ns: Optional[int] = None
    migration_service_ns: int = 0
    migration_queue_wait_ns: int = 0
    migration_reason: Optional[str] = None
    drop_reason: Optional[str] = None


@dataclass
class SSDRecord:
    tokens: int
    block_tokens: int
    bytes: int
    last_access_ns: int
    accounted_until_ns: int
    # A completed/retained read may outlive the nominal TTL. Keep a durable
    # not-before watermark so removing the pending read event cannot make a
    # later fast-forward reinsert an expiry in the past.
    pinned_until_ns: int = 0
    # SSD objects stay with the host node that owns the session. This prevents
    # independent serving replicas from sharing capacity or media queues.
    node_id: int = 0


@dataclass
class PendingSourceRelease:
    entry: IdleKVEntry
    ready_ns: int
    remove_ssd_record: bool = False


@dataclass
class PendingHBMAllocation:
    """HBM reservation that becomes physical after pressure eviction.

    A pressure victim remains authoritative in HBM until its demotion commits.
    Consequently, a foreground restore cannot consume the victim's bytes at
    eviction start.  This record reserves those future bytes and lets the
    request's ready time include the capacity wait without oversubscribing the
    physical HBM counter in the meantime.
    """

    entry: IdleKVEntry
    ready_ns: int


@dataclass(frozen=True)
class ActiveHBMReclaimClaim:
    """Logical HBM reservation for scheduler-owned active KV admission.

    The manager owns idle-session KV while the scheduler owns active KV.  A
    claim closes the gap between those ownership domains: idle demotions may
    make the bytes available only in the future, but no other manager-side
    admission may consume them after this record is created.
    """

    instance_id: int
    per_rank_bytes: int
    total_bytes: int
    admitted_ns: int
    ready_ns: int
    owner_kind: str = "legacy"
    owner_id: Optional[int] = None


@dataclass
class KVPreparation:
    hit_tokens: int
    recompute_tokens: int
    source: KVLocation
    # Physical restore only: destination-HBM admission plus transfer queue and
    # service after strict P/D-pair and prepare-boundary admission. Both
    # non-I/O admission components are separate below.
    restore_ns: int
    ready_time_ns: int
    restored_bytes: int
    owner_gate_ns: int = 0
    restore_issue_time_ns: int = 0
    target_hbm_ready_time_ns: int = 0
    restore_ready_time_ns: int = 0
    pd_pair_fifo_wait_ns: int = 0
    prepare_boundary_wait_ns: int = 0
    source_demotion_join_wait_ns: int = 0
    hbm_admission_wait_ns: int = 0
    transient_dram_capacity_wait_ns: int = 0
    queue_wait_ns: int = 0
    service_ns: int = 0
    retained_instance_id: Optional[int] = None
    retained_per_rank_bytes: int = 0
    residency_at_return: KVLocation = KVLocation.DROPPED


@dataclass
class ExternalFabricRestore:
    """One exact cold-HBM restore owned by the ASTRA event queue.

    The Python tier manager reserves destination HBM before issuing the job,
    but does not predict its completion.  The source entry remains authoritative
    and pinned until ASTRA reports the shared-fabric completion callback.
    """

    job_id: str
    session_id: str
    source_entry: IdleKVEntry
    target_entry: IdleKVEntry
    source_instance_id: int
    target_instance_id: int
    release_time_ns: int
    arrival_time_ns: int
    declared_reuse_tokens: int
    requested_reuse_tokens: int
    reusable_tokens: int
    input_tokens: int
    block_tokens: int
    bytes_per_lane: int
    lane_count: int
    total_bytes: int
    return_gap_type: str
    return_gap_source: str
    return_gap_ns: int
    residency_at_return: KVLocation
    pd_pair_fifo_wait_ns: int
    prepare_boundary_wait_ns: int
    status: str = "queued"
    completion_time_ns: Optional[int] = None
    critical_lane_start_ns: Optional[int] = None


@dataclass(frozen=True)
class TransferReservation:
    """One transfer scheduled on all of its bottleneck resources at once."""

    kind: str
    arrival_ns: int
    start_ns: int
    complete_ns: int
    service_ns: int
    queue_wait_ns: int
    resources: tuple[str, ...]
    completed: bool = True
    active_ns_before_cancel: int = 0
    reservation_sequence: int = 0
    parent_sequence: int = 0
    job_arrival_ns: int = 0
    transient_dram_capacity_wait_ns: int = 0


@dataclass(frozen=True)
class HBMRestoreProjection:
    """Pure destination-admission and full lower-tier restore forecast.

    ``post_reservation_fingerprint`` captures the completed shadow plan. It
    lets the live commit prove that ordinary HBM admission, DRAM capacity
    cascades, and foreground I/O took the same victims and calendar slots.
    """

    hbm_ready_ns: Optional[int]
    foreground_arrival_ns: Optional[int]
    restore_ready_ns: Optional[int]
    hbm_admission_wait_ns: int
    queue_wait_ns: int
    service_ns: int
    transfer_kinds: tuple[str, ...]
    transient_dram_capacity_wait_ns: int = 0
    hbm_victim_sessions: tuple[str, ...] = ()
    cpu_victim_sessions: tuple[str, ...] = ()
    foreground_reservation_signature: tuple[tuple, ...] = ()
    post_reservation_fingerprint: Optional[tuple] = None

    @property
    def available(self) -> bool:
        return (
            self.hbm_ready_ns is not None
            and self.foreground_arrival_ns is not None
            and self.restore_ready_ns is not None
        )

    @property
    def includes_new_lru_work(self) -> bool:
        return bool(
            self.hbm_victim_sessions or self.cpu_victim_sessions)

    @property
    def total_wait_ns(self) -> int:
        return self.hbm_admission_wait_ns + self.queue_wait_ns


@dataclass(frozen=True)
class QueueRecomputeCapacitySnapshot:
    """Causal P/D slack observation for one block-prefix candidate.

    The manager does not claim the recorded headroom.  It only rejects a
    partial prefix which could not fit through the next prefill chunk even at
    the decision timestamp.  The normal atomic P/D chunk-admission path must
    still acquire the bytes after the restore completes.
    """

    time_ns: int
    prefix_tokens: int
    prefix_block_tokens: int
    next_chunk_tokens: int
    through_next_chunk_block_tokens: int
    prefill_instance_id: int
    prefill_unreserved_per_rank_bytes: int
    prefill_prefix_per_rank_bytes: int
    prefill_growth_headroom_per_rank_bytes: int
    prefill_required_through_chunk_per_rank_bytes: int
    decode_instance_id: Optional[int] = None
    decode_unreserved_per_rank_bytes: Optional[int] = None
    decode_required_through_chunk_per_rank_bytes: int = 0

    @property
    def feasible(self) -> bool:
        if (self.prefill_required_through_chunk_per_rank_bytes
                > self.prefill_unreserved_per_rank_bytes):
            return False
        if self.decode_instance_id is None:
            return True
        return bool(
            self.decode_unreserved_per_rank_bytes is not None
            and self.decode_required_through_chunk_per_rank_bytes
            <= self.decode_unreserved_per_rank_bytes
        )


@dataclass(frozen=True)
class QueueRecomputeSelection:
    """One immutable full/partial/zero restore choice."""

    invocation: tuple
    reusable_tokens: int
    selected_tokens: int
    selected_block_tokens: int
    selected_per_rank_bytes: int
    selected_total_bytes: int
    full_total_bytes: int
    full_projection: Optional[HBMRestoreProjection]
    selected_projection: Optional[HBMRestoreProjection]
    capacity_snapshot: Optional[QueueRecomputeCapacitySnapshot]
    estimated_suffix_recompute_ns: Optional[int]
    estimated_full_recompute_ns: Optional[int]
    predicted_resume_path_ns: Optional[int]
    full_predicted_resume_path_ns: Optional[int]
    selection_reason: str
    candidate_tokens: tuple[int, ...] = ()

    @property
    def modified(self) -> bool:
        return self.selected_tokens < self.reusable_tokens

    @property
    def partial(self) -> bool:
        return 0 < self.selected_tokens < self.reusable_tokens

    @property
    def zero_restore(self) -> bool:
        return self.selected_tokens == 0

    @property
    def dropped_suffix_tokens(self) -> int:
        return max(0, self.reusable_tokens - self.selected_tokens)

    @property
    def dropped_suffix_bytes(self) -> int:
        return max(0, self.full_total_bytes - self.selected_total_bytes)


@dataclass(frozen=True)
class TransientDRAMReservation:
    """Full-object SSD bounce buffer held until its H2D DMA completes."""

    node_id: int
    session_id: Optional[str]
    start_ns: int
    complete_ns: int
    bytes: int
    reservation_sequence: int
    peak_node_committed_bytes: int = 0


@dataclass
class AgenticKVMetrics:
    # ``tool_pauses`` is retained as a backward-compatible total for every
    # inter-turn idle pause. The explicit counters below distinguish the
    # actual return class.
    tool_pauses: int = 0
    idle_pauses: int = 0
    tool_return_pauses: int = 0
    human_return_pauses: int = 0
    mixed_return_pauses: int = 0
    unknown_return_pauses: int = 0
    hbm_hits: int = 0
    cpu_hits: int = 0
    ssd_hits: int = 0
    dropped_misses: int = 0
    cache_hit_tokens: int = 0
    recompute_tokens: int = 0
    policy_avoidable_recompute_tokens: int = 0
    critical_restore_ns: int = 0
    pd_pair_fifo_admissions: int = 0
    pd_pair_fifo_waiting_admissions: int = 0
    pd_pair_fifo_wait_ns: int = 0
    prepare_boundary_admissions: int = 0
    prepare_boundary_waiting_admissions: int = 0
    prepare_boundary_wait_ns: int = 0
    source_demotion_join_admissions: int = 0
    source_demotion_join_waiting_admissions: int = 0
    source_demotion_join_wait_ns: int = 0
    pd_cross_instance_restore_ns: int = 0
    pd_decode_receive_admissions: int = 0
    pd_decode_receive_reserved_bytes: int = 0
    pd_decode_receive_capacity_wait_ns: int = 0
    pd_decode_receive_admission_wait_ns: int = 0
    pd_decode_receive_critical_wait_ns: int = 0
    pd_prefill_admissions: int = 0
    pd_prefill_reserved_bytes: int = 0
    pd_prefill_capacity_wait_ns: int = 0
    pd_prefill_admission_wait_ns: int = 0
    pd_prefill_admission_critical_wait_ns: int = 0
    pd_launch_admissions: int = 0
    pd_launch_admission_wait_ns: int = 0
    pd_launch_admission_critical_wait_ns: int = 0
    pd_chunk_admissions: int = 0
    pd_chunk_waiting_admissions: int = 0
    pd_chunk_admitted_tokens: int = 0
    pd_chunk_prefill_reserved_bytes: int = 0
    pd_chunk_decode_reserved_bytes: int = 0
    pd_chunk_admission_wait_ns: int = 0
    pd_chunk_admission_critical_wait_ns: int = 0
    pd_chunk_cancelled_admissions: int = 0
    pd_chunk_cancelled_waiting_admissions: int = 0
    pd_chunk_cancelled_admission_wait_ns: int = 0
    pd_chunk_cancelled_admission_critical_wait_ns: int = 0
    pd_active_prefill_recompute_preemptions: int = 0
    pd_active_prefill_recompute_tokens: int = 0
    agentic_kv_restored_tokens_discarded_by_active_prefill_recompute: int = 0
    pd_chunk_snapshot_joined_admissions: int = 0
    pd_chunk_snapshot_feasible_admissions: int = 0
    pd_chunk_snapshot_feasible_waiting_admissions: int = 0
    pd_chunk_snapshot_feasible_wait_ns: int = 0
    agentic_prefill_batches: int = 0
    agentic_mixed_hbm_lower_tier_prefill_batches: int = 0
    agentic_prefill_batch_execution_ns: int = 0
    agentic_mixed_hbm_lower_tier_batch_execution_ns: int = 0
    agentic_model_iteration_batches: int = 0
    agentic_model_iteration_execution_ns: int = 0
    astra_shared_fabric_windows: int = 0
    astra_shared_fabric_window_ns: int = 0
    direct_fabric_dispatch_blocks: int = 0
    direct_fabric_dispatch_wait_ns: int = 0
    external_fabric_jobs_issued: int = 0
    external_fabric_jobs_completed: int = 0
    external_fabric_jobs_censored: int = 0
    external_fabric_lane_bytes: int = 0
    external_fabric_censored_lane_bytes: int = 0
    sync_swap_barrier_jobs: int = 0
    sync_swap_in_barrier_jobs: int = 0
    sync_swap_out_barrier_jobs: int = 0
    sync_swap_engine_barrier_memberships: int = 0
    sync_swap_blocked_iterations: int = 0
    sync_swap_blocked_batch_memberships: int = 0
    sync_swap_ready_victim_memberships: int = 0
    sync_swap_max_ready_victims: int = 0
    hbm_to_cpu_bytes: int = 0
    pd_hbm_to_hbm_bytes: int = 0
    cpu_to_hbm_bytes: int = 0
    cpu_to_ssd_bytes: int = 0
    hbm_to_ssd_bytes: int = 0
    ssd_to_hbm_bytes: int = 0
    ssd_to_cpu_stage_bytes: int = 0
    cpu_stage_to_hbm_bytes: int = 0
    transient_dram_reservations: int = 0
    transient_dram_reserved_bytes: int = 0
    transient_dram_byte_ns: int = 0
    transient_dram_capacity_wait_ns: int = 0
    transient_dram_pressure_stall_ns: int = 0
    transient_dram_capacity_deferrals: int = 0
    transient_dram_capacity_oversize: int = 0
    transient_dram_cpu_lru_evictions: int = 0
    peak_transient_dram_bytes: int = 0
    peak_cpu_committed_plus_transient_bytes: int = 0
    ssd_host_write_bytes: int = 0
    ssd_cancelled_host_write_bytes: int = 0
    ssd_host_read_bytes: int = 0
    direct_ssd_write_bytes: int = 0
    # Schema compatibility counter. SSD swap-in is now always host staged.
    direct_ssd_read_bytes: int = 0
    capacity_drops: int = 0
    hbm_capacity_demotions: int = 0
    hbm_capacity_drops: int = 0
    cpu_capacity_evictions: int = 0
    cpu_capacity_bypasses: int = 0
    ssd_capacity_evictions: int = 0
    ssd_capacity_admission_drops: int = 0
    capacity_induced_recompute_tokens: int = 0
    queue_recompute_evaluation_attempts: int = 0
    queue_recompute_severe_gate_passes: int = 0
    queue_recompute_cost_gate_passes: int = 0
    queue_recompute_full_restore_decisions: int = 0
    queue_recompute_partial_restore_decisions: int = 0
    queue_recompute_zero_restore_decisions: int = 0
    queue_recompute_partial_cpu_decisions: int = 0
    queue_recompute_partial_ssd_decisions: int = 0
    queue_recompute_drop_decisions: int = 0
    queue_recompute_cpu_drop_decisions: int = 0
    queue_recompute_ssd_drop_decisions: int = 0
    # Compatibility alias: bytes of the foreground restore avoided by the
    # selected decision, not necessarily every physical copy discarded.
    queue_recompute_dropped_bytes: int = 0
    queue_recompute_avoided_restore_bytes: int = 0
    queue_recompute_physical_entry_dropped_bytes: int = 0
    queue_recompute_projected_queue_wait_ns: int = 0
    queue_recompute_projected_hbm_admission_wait_ns: int = 0
    queue_recompute_projected_transient_dram_capacity_wait_ns: int = 0
    queue_recompute_projected_service_ns: int = 0
    queue_recompute_prefix_projected_queue_wait_ns: int = 0
    queue_recompute_prefix_projected_hbm_admission_wait_ns: int = 0
    queue_recompute_prefix_projected_transient_dram_capacity_wait_ns: int = 0
    queue_recompute_prefix_projected_service_ns: int = 0
    queue_recompute_estimated_recompute_ns: int = 0
    queue_recompute_tokens: int = 0
    queue_recompute_policy_avoidable_tokens: int = 0
    queue_recompute_selected_restore_tokens: int = 0
    queue_recompute_dropped_suffix_tokens: int = 0
    queue_recompute_selected_restore_bytes: int = 0
    queue_recompute_dropped_suffix_bytes: int = 0
    ttl_drops: int = 0
    hbm_byte_ns: int = 0
    cpu_byte_ns: int = 0
    ssd_byte_ns: int = 0
    peak_ssd_used_bytes: int = 0
    peak_ssd_committed_reserved_bytes: int = 0
    transfer_jobs: int = 0
    queued_transfer_jobs: int = 0
    migration_service_ns: int = 0
    migration_queue_wait_ns: int = 0
    background_service_ns: int = 0
    background_queue_wait_ns: int = 0
    critical_restore_hbm_admission_wait_ns: int = 0
    critical_restore_service_ns: int = 0
    critical_restore_queue_wait_ns: int = 0
    async_restore_gross_ns: int = 0
    async_restore_compute_overlap_ns: int = 0
    async_restore_owner_barrier_ns: int = 0
    background_cancelled_jobs: int = 0
    background_cancelled_bytes: int = 0
    background_wasted_bytes: int = 0
    ssd_demotion_attempts: int = 0
    ssd_demotion_completions: int = 0
    ssd_demotion_cancelled: int = 0
    hbf_eligible_resumes: int = 0
    hbf_eligible_restore_bytes: int = 0
    hbf_gross_stall_upper_bound_ns: int = 0
    hbf_dropped_recompute_tokens: int = 0
    total_model_compute_ns: int = 0
    recompute_model_compute_ns: int = 0
    total_request_latency_ns: int = 0
    resumed_prompt_tokens: int = 0
    total_prompt_tokens: int = 0
    active_recompute_preemptions: int = 0
    active_recompute_tokens: int = 0
    active_cpu_swap_preemptions: int = 0
    active_cpu_swap_write_bytes: int = 0
    active_cpu_swap_read_bytes: int = 0
    active_cpu_swap_capacity_fallbacks: int = 0
    active_hbm_reclaim_admissions: int = 0
    active_hbm_reclaim_bytes: int = 0
    active_hbm_reclaim_per_rank_bytes: int = 0
    active_hbm_reclaim_wait_ns: int = 0
    peak_idle_hbm_bytes: int = 0
    peak_idle_cpu_bytes: int = 0


class AgenticKVManager:
    """Event-driven online tier manager shared by the request router."""

    def __init__(
            self, schedulers: Iterable[object], config: AgenticKVConfig,
            queue_recompute_latency_providers: Optional[Dict[int, object]] = None,
            ):
        config.validate()
        self.config = config
        self.schedulers = {s.instance_id: s for s in schedulers}
        self._queue_recompute_latency_providers = dict(
            queue_recompute_latency_providers or {})
        unknown_provider_instances = sorted(
            set(self._queue_recompute_latency_providers)
            - set(self.schedulers))
        if unknown_provider_instances:
            raise ValueError(
                "Queue-recompute latency providers reference unknown "
                f"scheduler instances {unknown_provider_instances}")
        if (config.queue_recompute_enabled
                and config.queue_recompute_cost_guard_multiplier > 0):
            missing = sorted(
                instance_id for instance_id, scheduler
                in self.schedulers.items()
                if getattr(scheduler, "pd_type", None) != "decode"
                and instance_id not in self._queue_recompute_latency_providers
            )
            if missing:
                raise ValueError(
                    "Cost-aware queue recomputation requires an online "
                    "latency provider for every eligible prefill scheduler; "
                    f"missing instances {missing}")
        conflicting = sorted(
            scheduler.instance_id
            for scheduler in self.schedulers.values()
            if getattr(scheduler, "enable_prefix_caching", False)
        )
        if config.enabled and conflicting:
            raise ValueError(
                "Agentic session-KV tiering and generic Radix prefix "
                "caching cannot share physical KV accounting; disable "
                "prefix caching on scheduler instance(s) "
                f"{conflicting}")
        for scheduler in self.schedulers.values():
            scheduler.agentic_kv_manager = self
        self.entries: Dict[str, IdleKVEntry] = {}
        self.ssd_records: Dict[str, SSDRecord] = {}
        self.pending_source_releases = []
        self.pending_hbm_allocations = []
        self._active_hbm_reclaim_claims: Dict[
            int, ActiveHBMReclaimClaim] = {}
        # Physical HBM counters are insufficient to identify every change in
        # reclaimability.  For example, an active request can free one block
        # at completion and immediately publish the same block as idle KV:
        # npu_used is unchanged, but the block has become an LRU victim.  This
        # monotonic signal makes those manager-owned admission dependencies
        # visible to coalesced router retries.
        self._hbm_admission_state_generation = 0
        self._direct_ssd_capacity_reservations: Dict[str, int] = {}
        self._direct_ssd_capacity_reservation_nodes: Dict[str, int] = {}
        self._ssd_node_ids = tuple(sorted({
            int(self._node_id(scheduler))
            for scheduler in self.schedulers.values()
        }))
        if not self._ssd_node_ids:
            raise ValueError("Agentic-KV requires at least one scheduler node")
        self.ssd_used_bytes = 0
        self.metrics = AgenticKVMetrics()
        self.events = []
        self._resource_busy_until: Dict[str, int] = {}
        self._resource_busy_ns: Dict[str, int] = {}
        self._resource_jobs: Dict[str, int] = {}
        # A scalar busy-until watermark cannot represent a prebooked future
        # stage followed by an independent short transfer that fits before
        # it.  Keep an immutable interval calendar per migration resource.
        # Multi-stage jobs reserve every child stage under one parent
        # sequence, so published owner-ready timestamps never need to move.
        self._resource_intervals: Dict[
            str, list[tuple[int, int, int, str]]
        ] = {}
        self._transient_dram_reservations: Dict[
            int, list[TransientDRAMReservation]
        ] = {}
        self._transient_dram_history: Dict[
            int, list[TransientDRAMReservation]
        ] = {}
        self._transfer_sequence = 0
        self._last_transfer_job_arrival_ns = -1
        self._logical_frontier_ns = 0
        # A returning lower-tier owner whose destination HBM is temporarily
        # full keeps its source pinned while the router retries admission.
        self._pending_restore_sessions: set[str] = set()
        # A returning continuation may observe its authoritative HBM/CPU
        # source while a capacity-triggered demotion is still in flight.  The
        # source remains pinned and only that continuation waits for the
        # already-published atomic commit; unrelated model work remains
        # runnable.  Router retries use the entry's immutable completion time.
        self._pending_demotion_join_sessions: set[str] = set()
        self._pending_demotion_join_windows: Dict[
            str, tuple[int, int, str]
        ] = {}
        self._pending_transient_restore_since: Dict[str, int] = {}
        self._pending_transient_restore_wait_ns: Dict[str, int] = {}
        # Destination admission starts only after routing/boundary waits and
        # any source-demotion join have elapsed.  Successful preparations are
        # accounted by the ordinary foreground restore interval; this state
        # exists so an early measurement cutoff can retain the open prefix.
        self._pending_destination_admission_since: Dict[str, int] = {}
        # A restore decision is made before any destination reservation can
        # enqueue LRU demotions. If HBM admission later retries, reuse this
        # immutable first decision so policy-created work cannot self-trigger
        # a different outcome.
        self._queue_recompute_restore_commitments: Dict[
            str, QueueRecomputeSelection] = {}
        self._active_reclaim_rejection_counts: Dict[str, int] = {}
        self._active_reclaim_rejection_samples = []
        self._active_reclaim_rejection_sample_limit = 128
        self._critical_restore_intervals = []
        self._source_demotion_join_intervals = []
        self._destination_admission_intervals = []
        # A measurement cutoff may terminate a continuation while it is
        # waiting for an already-issued source demotion.  Keep that clipped
        # exposure separate from completed-request accounting: it belongs in
        # the makespan interval union, but not in completed latency sums.
        self._censored_source_demotion_join_intervals = []
        self._censored_source_demotion_join_audits = []
        self._censored_destination_admission_intervals = []
        self._censored_destination_admission_audits = []
        self._censored_transient_restore_audits = []
        self._async_owner_barrier_intervals = []
        self._classified_request_ids = set()
        self._request_counts_by_residency_and_return: Dict[
            str, Dict[str, int]
        ] = {}
        self._request_counts_by_source_and_return: Dict[
            str, Dict[str, int]
        ] = {}
        self._batch_memberships_by_source_and_return: Dict[
            str, Dict[str, int]
        ] = {}
        self._async_restore_by_source_and_return: Dict[
            str, Dict[str, Dict[str, int]]
        ] = {}
        # GPU-facing synchronous swaps can stop P and D engines
        # independently. Keep these barriers separate from migration-resource
        # reservations so request-local restore accounting remains unchanged.
        self._sync_engine_barriers: Dict[
            int, list[tuple[int, int, str, str, Optional[str]]]
        ] = {instance_id: [] for instance_id in self.schedulers}
        self._sync_barrier_upcoming: Dict[int, list[tuple]] = {
            instance_id: [] for instance_id in self.schedulers
        }
        self._sync_barrier_active: Dict[int, Dict[int, tuple]] = {
            instance_id: {} for instance_id in self.schedulers
        }
        self._sync_barrier_end_heap: Dict[int, list[tuple[int, int]]] = {
            instance_id: [] for instance_id in self.schedulers
        }
        self._sync_barrier_sequence = 0
        self._sync_prepare_locks: Dict[int, tuple[int, ...]] = {}
        self._sync_prepare_sessions: Dict[int, str] = {}
        self._sync_deferred_hbm_demotions: set[str] = set()
        # Raw reservations and causally exposed scheduler stalls are distinct:
        # an I/O reservation can overlap an already-running iteration or an
        # otherwise idle engine. Only a failed dispatch attempt creates an
        # exposed interval and contributes to batch-blocking wait.
        self._sync_exposed_barriers: Dict[
            int, list[tuple[int, int, tuple[str, ...]]]
        ] = {instance_id: [] for instance_id in self.schedulers}
        self._sync_exposed_cursor: Dict[int, int] = {
            instance_id: 0 for instance_id in self.schedulers
        }
        self._sync_ready_victim_intervals: Dict[
            tuple[int, int], list[tuple[int, int]]
        ] = {}
        self._sync_unique_ready_victims: set[tuple[int, int]] = set()
        self._model_iteration_intervals: Dict[
            int, list[tuple[int, int]]
        ] = {instance_id: [] for instance_id in self.schedulers}
        # ASTRA-Sim owns TP/EP and ordinary P->D timing.  Cold HBM-resident
        # D->P copies are intentionally a separate asynchronous contention
        # domain: they serialize with one another on ``pd-fabric`` but never
        # gate unrelated model dispatch.  We still record ASTRA windows for an
        # explicit overlap audit; they do not reserve migration resources.
        self._astra_fabric_inflight: Dict[
            str, Dict[tuple[int, int], int]
        ] = {}
        self._astra_fabric_intervals: Dict[
            str, list[tuple[int, int, int, int]]
        ] = {}
        # Coalesced, start-sorted windows make retrospective event placement
        # O(log B + overlaps) instead of sorting/scanning all B batches for
        # every one of T transfers in a long online run.
        self._astra_fabric_calendar: Dict[
            str, list[tuple[int, int]]
        ] = {}
        # These compatibility counters remain empty in asynchronous mode.
        # Only sync-engine-barrier may gate an unrelated model dispatch.
        self._direct_fabric_dispatch_block_intervals: Dict[
            int, list[tuple[int, int]]
        ] = {}
        # Retained for report-schema compatibility.  Async direct-fabric
        # copies do not acquire model-boundary locks.
        self._fabric_prepare_locks: Dict[int, tuple[int, ...]] = {}
        self._fabric_prepare_sessions: Dict[int, str] = {}
        # Online congestion-aware runs hand direct D->P HBM copies to the
        # ASTRA event queue. Unit-level analytical helpers remain available
        # until this mode is explicitly enabled by the serving entry point.
        self._external_fabric_enabled = False
        self._external_fabric_authority = None
        self._external_fabric_sequence = 0
        self._external_fabric_by_session: Dict[
            str, ExternalFabricRestore] = {}
        self._external_fabric_by_job: Dict[
            str, ExternalFabricRestore] = {}
        self._external_fabric_outgoing: list[str] = []
        self._external_fabric_tombstones: Dict[
            str, tuple[int, int, int, int, int]
        ] = {}
        self._external_fabric_completion_generation: Dict[int, int] = {
            instance_id: 0 for instance_id in self.schedulers
        }
        self._external_fabric_history = []

    # ------------------------------------------------------------------
    # Latency helpers (streamed transfers use the slowest link).
    # ------------------------------------------------------------------

    @staticmethod
    def _parallel_wire_ns(paths, latency_us: float = 0.0) -> int:
        """Return a pipelined transfer bound across simultaneous bottlenecks.

        ``paths`` contains ``(bytes, decimal_GB/s)`` pairs.  For a TP cache,
        every GPU traverses its own PCIe link concurrently while the aggregate
        traffic shares host DRAM or the SSD pool.  Taking the maximum path time
        captures that distinction; dividing full-cluster bytes by one GPU's
        PCIe bandwidth would overstate TP8 latency by up to 8x.
        """
        active = [
            num_bytes / (bandwidth_gbps * DECIMAL_GB) * NS_PER_SECOND
            for num_bytes, bandwidth_gbps in paths
            if num_bytes > 0
        ]
        if not active:
            return 0
        return int(math.ceil(max(active) + latency_us * 1_000))

    def _cpu_transfer_ns(self, per_rank_bytes: int, total_bytes: int) -> int:
        return self._parallel_wire_ns(
            (
                (per_rank_bytes, self.config.pcie_bandwidth_gbps),
                (total_bytes, self.config.cpu_bandwidth_gbps),
            ),
            self.config.cpu_transfer_latency_us,
        )

    def _hbm_peer_transfer_ns(
            self, source_scheduler, target_scheduler,
            source_per_rank_bytes: int, target_per_rank_bytes: int,
            total_bytes: int) -> int:
        """Return the configured node-local D->P copy lower bound.

        In the default CPU-staged mode, source D2H and destination H2D can
        stream concurrently on distinct GPU links, while every byte is both
        written to and read from host DRAM. Direct-fabric mode instead sends
        each TP rank over its accelerator peer link. Cross-node relocation
        needs a separate NIC model and is rejected in both modes.
        """
        source_node = getattr(
            source_scheduler, "node_id", source_scheduler.instance_id)
        target_node = getattr(
            target_scheduler, "node_id", target_scheduler.instance_id)
        if source_node != target_node:
            raise RuntimeError(
                "Cross-node agentic D->P HBM relocation is not modeled; "
                "configure a same-node P/D pair")
        if self.config.pd_peer_transfer_mode == "direct-fabric":
            return self._parallel_wire_ns(
                ((
                    max(source_per_rank_bytes, target_per_rank_bytes),
                    self.config.pd_peer_bandwidth_gbps,
                ),),
                self.config.pd_peer_latency_us,
            )
        return self._parallel_wire_ns(
            (
                (source_per_rank_bytes, self.config.pcie_bandwidth_gbps),
                (target_per_rank_bytes, self.config.pcie_bandwidth_gbps),
                (2 * total_bytes, self.config.cpu_bandwidth_gbps),
            ),
            2 * self.config.cpu_transfer_latency_us,
        )

    def _validate_node_shared_restore_layout(
            self, entry: IdleKVEntry, target_scheduler,
            block_tokens: int, target_per_rank_bytes: int,
            target_total_bytes: int) -> None:
        """Validate a lower-tier object before restoring it directly to P.

        CPU DRAM and SSD are node-shared, so a record last owned by D does not
        need a D-HBM staging allocation.  It does still need a same-node,
        layout-compatible P target: no inter-node transport or KV reformatting
        is hidden in this analytical path.
        """
        source_scheduler = self._scheduler(entry.instance_id)
        if self._node_id(source_scheduler) != self._node_id(target_scheduler):
            raise RuntimeError(
                "Cross-node lower-tier KV restore is not modeled; configure "
                "the P target on the CPU/SSD cache's node")
        source_per_rank_bytes = int(
            source_scheduler.memory.get_kv(int(block_tokens)))
        source_total_bytes = (
            source_per_rank_bytes * int(source_scheduler.num_npus))
        if source_per_rank_bytes > int(entry.per_rank_bytes):
            raise RuntimeError(
                "Lower-tier source KV object is smaller than the reusable "
                "prefix")
        if (source_per_rank_bytes != int(target_per_rank_bytes)
                or source_total_bytes != int(target_total_bytes)):
            raise RuntimeError(
                "P/D lower-tier KV layouts differ; model, TP, PP, block size, "
                "and KV dtype must match before a node-shared restore")

    def _ssd_write_ns(self, total_bytes: int, per_rank_bytes: int = 0) -> int:
        media_ns = self._parallel_wire_ns(
            ((total_bytes, self.config.ssd_write_bandwidth_gbps),),
            self.config.ssd_write_latency_us,
        )
        if not per_rank_bytes:
            return media_ns
        # The baseline stages through host DRAM. Keep the two legs serial so
        # online service time matches the standalone analyzer; a future
        # direct-storage/pipelined backend should be a separate policy.
        return media_ns + self._cpu_transfer_ns(per_rank_bytes, total_bytes)

    def _ssd_to_cpu_stage_ns(self, total_bytes: int) -> int:
        """Return SSD-media plus host-DRAM-write pipeline service."""
        return self._parallel_wire_ns(
            (
                (total_bytes, self.config.ssd_read_bandwidth_gbps),
                (total_bytes, self.config.cpu_bandwidth_gbps),
            ),
            self.config.ssd_read_latency_us,
        )

    def _ssd_read_ns(self, total_bytes: int, per_rank_bytes: int = 0) -> int:
        media_ns = self._ssd_to_cpu_stage_ns(total_bytes)
        if not per_rank_bytes:
            return media_ns
        return media_ns + self._cpu_transfer_ns(per_rank_bytes, total_bytes)

    def _direct_ssd_write_ns(
            self, per_rank_bytes: int, total_bytes: int) -> int:
        """GPU-to-SSD service time without a host-DRAM staging leg."""
        return self._parallel_wire_ns(
            (
                (per_rank_bytes, self.config.pcie_bandwidth_gbps),
                (total_bytes, self.config.ssd_write_bandwidth_gbps),
            ),
            self.config.ssd_write_latency_us,
        )

    def _direct_ssd_write_shape(
            self, entry: IdleKVEntry) -> tuple[int, int, int]:
        """Return issued bytes, per-rank bytes, and direct write service."""
        num_bytes = self._ssd_write_bytes(entry)
        num_ranks = self._scheduler(entry.instance_id).num_npus
        per_rank_bytes = (
            (num_bytes + num_ranks - 1) // num_ranks if num_bytes else 0)
        return (
            num_bytes,
            per_rank_bytes,
            self._direct_ssd_write_ns(per_rank_bytes, num_bytes),
        )

    @staticmethod
    def _node_id(scheduler) -> int:
        return getattr(scheduler, "node_id", scheduler.instance_id)

    @staticmethod
    def _pcie_resources(scheduler) -> list[str]:
        return [
            f"instance:{scheduler.instance_id}:pcie-copy:{rank}"
            for rank in range(scheduler.num_npus)
        ]

    @staticmethod
    def _peer_copy_resources(scheduler) -> list[str]:
        return [
            f"instance:{scheduler.instance_id}:peer-copy:{rank}"
            for rank in range(scheduler.num_npus)
        ]

    def _astra_fabric_resources(self, scheduler) -> tuple[str, ...]:
        """Return labels used to audit ASTRA/cold-copy overlap.

        ASTRA-Sim already models contention among TP/EP collectives and P->D
        point-to-point nodes.  The interactive frontend currently reports only
        whole-iteration completion, not communication-node start/end events.
        These node labels therefore support reporting only: ASTRA intervals do
        not update ``_resource_busy_until`` and cold-copy intervals do not gate
        ASTRA dispatch.  Charging a whole model iteration as unavailable would
        incorrectly turn asynchronous swap into an engine barrier.
        """
        if self.config.pd_peer_transfer_mode != "direct-fabric":
            return ()
        if (int(getattr(scheduler, "num_npus", 1)) <= 1
                and getattr(scheduler, "pd_type", None) is None):
            return ()
        return (f"node:{self._node_id(scheduler)}:pd-fabric",)

    def model_dispatch_blocked_until(
            self, instance_id: int, now_ns: int) -> Optional[int]:
        """Return no gate for asynchronous cold-copy resources.

        The owner request carries the restore-completion dependency.  A cold
        peer copy may overlap ASTRA model execution, so polling this hook must
        never turn it into a whole-iteration barrier for unrelated requests.
        """
        del instance_id, now_ns
        return None

    def model_dispatch_resource_ready_time(
            self, instance_id: int, eligibility_ns: int) -> int:
        """Return unchanged eligibility; async copies do not gate dispatch."""
        del instance_id
        return int(eligibility_ns)

    def _assert_direct_fabric_boundary(
            self, resources: Sequence[str], kind: str) -> None:
        """Compatibility hook: async peer copies need no model boundary."""
        del resources, kind

    def _after_completed_astra_windows(
            self, resources: Sequence[str], start_ns: int,
            service_ns: int) -> int:
        """Keep the logical issue time; model execution is independent."""
        del resources, service_ns
        return int(start_ns)

    def _insert_astra_calendar_window(
            self, resource: str, start_ns: int, finish_ns: int) -> None:
        """Insert and coalesce a window even when callbacks finish out of order."""
        start_ns = int(start_ns)
        finish_ns = int(finish_ns)
        if finish_ns < start_ns:
            raise RuntimeError(
                "ASTRA fabric calendar window finishes before it starts: "
                f"resource={resource}, start={start_ns}, finish={finish_ns}")
        windows = self._astra_fabric_calendar.setdefault(resource, [])
        index = bisect.bisect_left(windows, (start_ns, -math.inf))
        if index > 0 and windows[index - 1][1] >= start_ns:
            index -= 1
            start_ns = min(start_ns, windows[index][0])
            finish_ns = max(finish_ns, windows[index][1])
            windows.pop(index)
        while index < len(windows) and windows[index][0] <= finish_ns:
            start_ns = min(start_ns, windows[index][0])
            finish_ns = max(finish_ns, windows[index][1])
            windows.pop(index)
        windows.insert(index, (start_ns, finish_ns))

    def _transfer_resources(
            self, kind: str, source_instance_id: int,
            target_instance_id: Optional[int] = None) -> tuple[str, ...]:
        """Return gang-scheduled bottlenecks for one migration.

        Host DRAM is deliberately a single half-duplex node resource. This is
        a conservative baseline that captures read/write interference. SSD
        media exposes separate read and write queues, while each TP rank owns
        distinct PCIe and peer-copy resources. Direct P/D traffic also shares
        one node-fabric resource.
        """
        source = self._scheduler(source_instance_id)
        target = (
            source if target_instance_id is None
            else self._scheduler(target_instance_id)
        )
        source_node = self._node_id(source)
        target_node = self._node_id(target)
        peer_kinds = {
            "hbm_peer", "cpu_to_hbm_peer", "ssd_to_hbm_peer",
        }
        direct_peer = (
            self.config.pd_peer_transfer_mode == "direct-fabric"
            and kind in peer_kinds
        )
        if direct_peer and source_node != target_node:
            raise RuntimeError(
                "Cross-node agentic D->P HBM relocation is not modeled; "
                "configure a same-node P/D pair")
        resources: list[str] = []
        if kind in {
                "hbm_to_cpu", "hbm_to_ssd",
                "hbm_to_ssd_direct"}:
            resources.extend(self._pcie_resources(source))
        if kind in {"hbm_peer", "cpu_to_hbm_peer", "ssd_to_hbm_peer"}:
            # Legacy combined kinds retain their historical endpoint shape.
            # Current lower-tier restores use cpu_to_hbm/cpu_stage_to_hbm.
            if not direct_peer or kind != "hbm_peer":
                resources.extend(self._pcie_resources(source))
        if kind in {
                "cpu_to_hbm", "ssd_to_hbm",
                "ssd_to_hbm_direct", "cpu_stage_to_hbm"}:
            resources.extend(self._pcie_resources(target))
        if kind in {"hbm_peer", "cpu_to_hbm_peer", "ssd_to_hbm_peer"}:
            if not direct_peer:
                resources.extend(self._pcie_resources(target))
        if kind in {
                "hbm_to_cpu", "cpu_to_hbm", "hbm_to_ssd",
                "cpu_to_ssd", "ssd_to_hbm",
                "cpu_to_hbm_peer", "ssd_to_hbm_peer",
                # Retain the old event name as a staged compatibility path;
                # no SSD swap-in may bypass host DRAM.
                "ssd_to_hbm_direct", "ssd_to_cpu_stage",
                "cpu_stage_to_hbm"}:
            resources.append(f"node:{source_node}:dram")
            if target_node != source_node:
                resources.append(f"node:{target_node}:dram")
        if kind == "hbm_peer" and not direct_peer:
            resources.append(f"node:{source_node}:dram")
            if target_node != source_node:
                resources.append(f"node:{target_node}:dram")
        if direct_peer:
            resources.extend(self._peer_copy_resources(source))
            resources.extend(self._peer_copy_resources(target))
            resources.append(f"node:{source_node}:pd-fabric")
        if kind in {
                "hbm_to_ssd", "cpu_to_ssd", "hbm_to_ssd_direct"}:
            resources.append(self._ssd_pool_resource(source_node, "write"))
        elif kind in {
                "ssd_to_hbm", "ssd_to_hbm_peer", "ssd_to_hbm_direct",
                "ssd_to_cpu_stage"}:
            resources.append(self._ssd_pool_resource(source_node, "read"))
        return tuple(dict.fromkeys(resources))

    def _ssd_pool_resource(self, node_id: int, direction: str) -> str:
        """Return a node-local media queue while preserving one-node labels."""
        if direction not in {"read", "write"}:
            raise ValueError(f"Unknown SSD queue direction: {direction}")
        if len(self._ssd_node_ids) == 1:
            return f"ssd-pool:{direction}"
        return f"node:{int(node_id)}:ssd-pool:{direction}"

    @property
    def synchronous_swap_enabled(self) -> bool:
        return self.config.swap_execution_mode == "sync-engine-barrier"

    @property
    def async_decode_join_enabled(self) -> bool:
        return self.config.swap_execution_mode == "async-decode-join"

    @staticmethod
    def _sync_swap_direction(kind: str) -> Optional[str]:
        if kind in {
                "hbm_to_cpu", "hbm_to_ssd", "hbm_to_ssd_direct"}:
            return "out"
        if kind in {
                "cpu_to_hbm", "ssd_to_hbm", "ssd_to_hbm_direct",
                "cpu_to_hbm_peer", "ssd_to_hbm_peer",
                "ssd_to_hbm_peer_direct", "cpu_stage_to_hbm",
                "cpu_stage_to_hbm_peer", "ssd_staged_to_hbm",
                "ssd_staged_to_hbm_peer"}:
            return "in"
        return None

    def _sync_swap_instances(
            self, kind: str, source_instance_id: int,
            target_instance_id: Optional[int]) -> tuple[int, ...]:
        """Return model engines stopped by a GPU-facing swap.

        Local swap-out blocks its source and local swap-in blocks its target.
        Legacy combined peer-kind records use both P and D GPUs, so the
        conservative synchronous compatibility path blocks both. Current
        lower-tier restores target P directly. Pure HBM peer copies and
        CPU-to-SSD movement are not swap barriers.
        """
        direction = self._sync_swap_direction(kind)
        if direction == "out":
            return (int(source_instance_id),)
        if direction != "in":
            return ()
        target = (
            int(source_instance_id)
            if target_instance_id is None else int(target_instance_id)
        )
        if kind in {
                "cpu_to_hbm_peer", "ssd_to_hbm_peer",
                "ssd_to_hbm_peer_direct", "cpu_stage_to_hbm_peer",
                "ssd_staged_to_hbm_peer"}:
            return tuple(dict.fromkeys((int(source_instance_id), target)))
        return (target,)

    def _register_sync_swap_barrier(
            self, reservation: TransferReservation,
            source_instance_id: int,
            target_instance_id: Optional[int],
            session_id: Optional[str],
            exposes_owner_request: bool = False) -> None:
        """Stop affected engines from reservation arrival through commit.

        Queue wait is included: a synchronous cache operation queued behind
        an older copy still prevents the model iteration that issued it from
        advancing. Scheduler gating applies the interval without relabeling
        request-local restore time.
        """
        if not self.synchronous_swap_enabled:
            return
        direction = self._sync_swap_direction(reservation.kind)
        if direction is None:
            return
        start_ns = int(reservation.arrival_ns)
        end_ns = int(reservation.complete_ns)
        if end_ns <= start_ns:
            return
        instances = self._sync_swap_instances(
            reservation.kind, source_instance_id, target_instance_id)
        self.metrics.sync_swap_barrier_jobs += 1
        if direction == "in":
            self.metrics.sync_swap_in_barrier_jobs += 1
        else:
            self.metrics.sync_swap_out_barrier_jobs += 1
        self.metrics.sync_swap_engine_barrier_memberships += len(instances)
        owner_instance_id = (
            int(source_instance_id)
            if direction == "out" else (
                int(source_instance_id)
                if target_instance_id is None
                else int(target_instance_id)
            )
        )
        for instance_id in instances:
            barrier = (
                start_ns,
                end_ns,
                direction,
                reservation.kind,
                session_id,
            )
            self._sync_engine_barriers.setdefault(instance_id, []).append(
                barrier)
            sequence = self._sync_barrier_sequence
            self._sync_barrier_sequence += 1
            heapq.heappush(
                self._sync_barrier_upcoming.setdefault(instance_id, []),
                (start_ns, sequence, end_ns, direction,
                 reservation.kind, session_id),
            )
            if exposes_owner_request and instance_id == owner_instance_id:
                exposed = (start_ns, end_ns, (direction,))
                exposures = self._sync_exposed_barriers.setdefault(
                    instance_id, [])
                if not exposures or exposures[-1] != exposed:
                    exposures.append(exposed)
            self.events.append({
                "time_ns": start_ns,
                "event": "sync_swap_engine_barrier",
                "session_id": session_id,
                "instance_id": instance_id,
                "kind": reservation.kind,
                "direction": direction,
                "start_ns": start_ns,
                "complete_ns": end_ns,
                "queue_wait_ns": int(reservation.queue_wait_ns),
                "service_ns": int(reservation.service_ns),
                "exposes_owner_request": bool(
                    exposes_owner_request
                    and instance_id == owner_instance_id),
            })

    def _active_sync_swap_barriers(
            self, instance_id: int, now_ns: int) -> tuple[tuple, ...]:
        instance_id = int(instance_id)
        now_ns = int(now_ns)
        upcoming = self._sync_barrier_upcoming.setdefault(instance_id, [])
        active = self._sync_barrier_active.setdefault(instance_id, {})
        end_heap = self._sync_barrier_end_heap.setdefault(instance_id, [])
        while upcoming and upcoming[0][0] <= now_ns:
            start_ns, sequence, end_ns, direction, kind, session_id = (
                heapq.heappop(upcoming))
            if end_ns <= now_ns:
                continue
            active[sequence] = (
                start_ns, end_ns, direction, kind, session_id)
            heapq.heappush(end_heap, (end_ns, sequence))
        while end_heap and end_heap[0][0] <= now_ns:
            _, sequence = heapq.heappop(end_heap)
            active.pop(sequence, None)
        return tuple(active.values())

    def synchronous_swap_blocked_until(
            self, instance_id: int, now_ns: int) -> Optional[int]:
        """Return the end of the synchronous barrier covering ``now_ns``."""
        if not self.synchronous_swap_enabled:
            return None
        return max(
            (
                barrier[1]
                for barrier in self._active_sync_swap_barriers(
                    instance_id, now_ns)
            ),
            default=None,
        )

    def acquire_synchronous_prepare_lock(
            self, request_id: int, instance_ids: Sequence[int],
            session_id: Optional[str] = None) -> None:
        """Prevent new iterations while a foreground swap awaits idle GPUs."""
        if not self.synchronous_swap_enabled:
            return
        request_id = int(request_id)
        pinned_before = self._capacity_pinned_sessions()
        next_lock = tuple(dict.fromkeys(
            int(instance_id) for instance_id in instance_ids))
        self._sync_prepare_locks[request_id] = next_lock
        if session_id is not None:
            self._sync_prepare_sessions[request_id] = str(session_id)
        self._publish_capacity_pin_change(
            pinned_before, self._capacity_pinned_sessions())

    def release_synchronous_prepare_lock(self, request_id: int) -> None:
        request_id = int(request_id)
        pinned_before = self._capacity_pinned_sessions()
        self._sync_prepare_locks.pop(request_id, None)
        self._sync_prepare_sessions.pop(request_id, None)
        self._publish_capacity_pin_change(
            pinned_before, self._capacity_pinned_sessions())

    def _synchronous_prepare_pinned_sessions(self) -> set[str]:
        # The historical name is retained because capacity code already uses
        # this helper. Async direct-fabric copies acquire no prepare lock.
        return (
            set(self._sync_prepare_sessions.values())
            | set(self._fabric_prepare_sessions.values())
        )

    def _capacity_pinned_sessions(self) -> set[str]:
        """Return session sources that no capacity policy may reclaim."""
        return (
            self._synchronous_prepare_pinned_sessions()
            | set(self._pending_restore_sessions)
        )

    def _publish_capacity_pin_change(
            self, before: set[str], after: set[str]) -> bool:
        """Publish a capacity pin transition exactly once."""
        changed = set(before) != set(after)
        if changed:
            self._mark_hbm_admission_state_changed()
        return changed

    def _set_restore_capacity_pin(
            self, session_id: str, pinned: bool) -> bool:
        """Set one restore pin and publish eligibility changes exactly once."""
        session_id = str(session_id)
        pinned_before = self._capacity_pinned_sessions()
        if pinned:
            self._pending_restore_sessions.add(session_id)
        else:
            self._pending_restore_sessions.discard(session_id)
        pinned_after = self._capacity_pinned_sessions()
        self._publish_capacity_pin_change(pinned_before, pinned_after)
        return pinned_before != pinned_after

    def pending_prepare_retry_time(
            self, session_id: str) -> Optional[int]:
        """Return the exact retry event for a demotion-joining continuation."""
        session_id = str(session_id)
        if session_id not in self._pending_demotion_join_sessions:
            return None
        entry = self.entries.get(session_id)
        if entry is None or entry.migration_complete_ns is None:
            return None
        return int(entry.migration_complete_ns)

    def _begin_demotion_join(
            self, session_id: str, start_ns: int, complete_ns: int,
            migration_kind: str) -> bool:
        """Pin one immutable source-demotion tail; return True on first join."""
        session_id = str(session_id)
        window = (int(start_ns), int(complete_ns), str(migration_kind))
        previous = self._pending_demotion_join_windows.get(session_id)
        if previous is not None:
            first_start_ns, first_complete_ns, first_kind = previous
            if (window[1], window[2]) != (first_complete_ns, first_kind):
                raise RuntimeError(
                    "Pending source-demotion join changed identity: "
                    f"session={session_id}, first={previous}, retry={window}")
            if window[0] < first_start_ns:
                raise RuntimeError(
                    "Pending source-demotion join retry regressed time: "
                    f"session={session_id}, first={previous}, retry={window}")
            return False
        if window[1] <= window[0]:
            raise ValueError(
                "Source-demotion join requires a future completion: "
                f"session={session_id}, window={window}")
        self._pending_demotion_join_windows[session_id] = window
        self._pending_demotion_join_sessions.add(session_id)
        self._set_restore_capacity_pin(session_id, True)
        return True

    def _demotion_join_wait(self, session_id: str, now_ns: int) -> int:
        """Observe the immutable exposed tail without ending the dependency."""
        session_id = str(session_id)
        window = self._pending_demotion_join_windows.get(session_id)
        if window is None:
            return 0
        start_ns, complete_ns, _ = window
        if int(now_ns) < complete_ns:
            raise RuntimeError(
                "Source-demotion join observed before commit: "
                f"session={session_id}, now={now_ns}, complete={complete_ns}")
        return max(0, complete_ns - start_ns)

    def _consume_demotion_join(self, session_id: str, now_ns: int) -> int:
        """Finalize one exposed swap-out tail after preparation commits."""
        session_id = str(session_id)
        wait_ns = self._demotion_join_wait(session_id, now_ns)
        window = self._pending_demotion_join_windows.pop(session_id, None)
        self._pending_demotion_join_sessions.discard(session_id)
        if window is not None:
            self._source_demotion_join_intervals.append(
                (window[0], window[1]))
        return wait_ns

    def _clear_demotion_join(self, session_id: str) -> None:
        session_id = str(session_id)
        self._pending_demotion_join_sessions.discard(session_id)
        self._pending_demotion_join_windows.pop(session_id, None)

    def _censor_demotion_join(
            self, session_id: str, cutoff_ns: int) -> Optional[dict]:
        """Right-censor one open source-demotion dependency at ``cutoff``.

        This is deliberately distinct from consuming a successful
        preparation.  The elapsed interval contributes only to wall-clock
        migration exposure; completed-request latency counters remain
        unchanged.
        """
        session_id = str(session_id)
        cutoff_ns = int(cutoff_ns)
        has_session = session_id in self._pending_demotion_join_sessions
        has_window = session_id in self._pending_demotion_join_windows
        if has_session != has_window:
            raise RuntimeError(
                "Source-demotion join censor found inconsistent state: "
                f"session={session_id}, pending={has_session}, "
                f"window={has_window}")
        window = self._pending_demotion_join_windows.pop(session_id, None)
        self._pending_demotion_join_sessions.discard(session_id)
        if window is None:
            return None
        start_ns, complete_ns, migration_kind = window
        if cutoff_ns < start_ns:
            raise RuntimeError(
                "Measurement cutoff precedes source-demotion join: "
                f"session={session_id}, start={start_ns}, cutoff={cutoff_ns}")
        exposed_end_ns = min(cutoff_ns, complete_ns)
        elapsed_ns = max(0, exposed_end_ns - start_ns)
        remaining_ns = max(0, complete_ns - cutoff_ns)
        if elapsed_ns:
            self._censored_source_demotion_join_intervals.append(
                (start_ns, exposed_end_ns))
        audit = {
            "session_id": session_id,
            "cutoff_ns": cutoff_ns,
            "start_ns": start_ns,
            "complete_ns": complete_ns,
            "exposed_end_ns": exposed_end_ns,
            "elapsed_ns": elapsed_ns,
            "remaining_ns": remaining_ns,
            "migration_kind": migration_kind,
        }
        self._censored_source_demotion_join_audits.append(audit)
        self.events.append({
            "time_ns": cutoff_ns,
            "event": "source_demotion_join_censored",
            **audit,
        })
        return audit

    def _begin_transient_restore_wait(
            self, session_id: str, now_ns: int) -> None:
        self._pending_transient_restore_since.setdefault(
            str(session_id), int(now_ns))

    def _begin_destination_admission_wait(
            self, session_id: str, start_ns: int,
            operation_time_ns: int) -> None:
        """Pin the immutable start of an unissued restore admission wait."""
        session_id = str(session_id)
        start_ns = int(start_ns)
        operation_time_ns = int(operation_time_ns)
        if start_ns > operation_time_ns:
            raise RuntimeError(
                "Destination admission wait starts after its deferral: "
                f"session={session_id}, start={start_ns}, "
                f"operation={operation_time_ns}")
        previous = self._pending_destination_admission_since.setdefault(
            session_id, start_ns)
        if previous != start_ns:
            raise RuntimeError(
                "Destination admission retry changed its causal start: "
                f"session={session_id}, first={previous}, retry={start_ns}")

    def _clear_destination_admission_wait(self, session_id: str) -> None:
        self._pending_destination_admission_since.pop(str(session_id), None)

    def _consume_destination_admission_wait(
            self, session_id: str, expected_start_ns: int) -> bool:
        """Close a destination wait after admission or a terminal decision."""
        session_id = str(session_id)
        expected_start_ns = int(expected_start_ns)
        start_ns = self._pending_destination_admission_since.pop(
            session_id, None)
        if start_ns is None:
            return False
        if start_ns != expected_start_ns:
            raise RuntimeError(
                "Destination admission completed with a changed start: "
                f"session={session_id}, first={start_ns}, "
                f"expected={expected_start_ns}")
        return True

    def _censor_destination_admission_wait(
            self, session_id: str, cutoff_ns: int) -> Optional[dict]:
        session_id = str(session_id)
        cutoff_ns = int(cutoff_ns)
        start_ns = self._pending_destination_admission_since.pop(
            session_id, None)
        if start_ns is None:
            return None
        if cutoff_ns < start_ns:
            raise RuntimeError(
                "Measurement cutoff precedes destination admission: "
                f"session={session_id}, start={start_ns}, cutoff={cutoff_ns}")
        elapsed_ns = cutoff_ns - start_ns
        if elapsed_ns:
            self._censored_destination_admission_intervals.append(
                (start_ns, cutoff_ns))
        audit = {
            "session_id": session_id,
            "cutoff_ns": cutoff_ns,
            "start_ns": start_ns,
            "elapsed_ns": elapsed_ns,
            "remaining_ns": None,
        }
        self._censored_destination_admission_audits.append(audit)
        self.events.append({
            "time_ns": cutoff_ns,
            "event": "destination_restore_admission_censored",
            **audit,
        })
        return audit

    def _pause_transient_restore_wait(
            self, session_id: str, now_ns: int) -> None:
        session_id = str(session_id)
        start_ns = self._pending_transient_restore_since.pop(
            session_id, None)
        if start_ns is not None:
            self._pending_transient_restore_wait_ns[session_id] = (
                self._pending_transient_restore_wait_ns.get(session_id, 0)
                + max(0, int(now_ns) - int(start_ns))
            )

    def _consume_transient_restore_wait(
            self, session_id: str, now_ns: int) -> int:
        self._pause_transient_restore_wait(session_id, now_ns)
        return self._pending_transient_restore_wait_ns.pop(
            str(session_id), 0)

    def _clear_transient_restore_wait(self, session_id: str) -> None:
        self._pending_transient_restore_since.pop(str(session_id), None)
        self._pending_transient_restore_wait_ns.pop(str(session_id), None)

    def _censor_transient_restore_wait(
            self, session_id: str, cutoff_ns: int) -> Optional[dict]:
        """Report the labeled DRAM subset without adding it to exposure."""
        session_id = str(session_id)
        cutoff_ns = int(cutoff_ns)
        start_ns = self._pending_transient_restore_since.pop(
            session_id, None)
        accumulated_ns = self._pending_transient_restore_wait_ns.pop(
            session_id, 0)
        if start_ns is None and accumulated_ns == 0:
            return None
        if start_ns is not None and cutoff_ns < start_ns:
            raise RuntimeError(
                "Measurement cutoff precedes transient DRAM admission: "
                f"session={session_id}, start={start_ns}, cutoff={cutoff_ns}")
        active_ns = (
            0 if start_ns is None else cutoff_ns - int(start_ns))
        elapsed_ns = int(accumulated_ns) + active_ns
        audit = {
            "session_id": session_id,
            "cutoff_ns": cutoff_ns,
            "active_start_ns": start_ns,
            "accumulated_paused_ns": int(accumulated_ns),
            "active_ns": active_ns,
            "elapsed_ns": elapsed_ns,
            "remaining_ns": None,
        }
        self._censored_transient_restore_audits.append(audit)
        self.events.append({
            "time_ns": cutoff_ns,
            "event": "transient_dram_restore_admission_censored",
            **audit,
        })
        return audit

    @property
    def logical_frontier_ns(self) -> int:
        """Latest logical timestamp committed by the tier event engine."""
        return int(self._logical_frontier_ns)

    def enable_external_fabric(
            self, *, backend: str, physical_bandwidth_gbps: float,
            physical_latency_ns: int,
            physical_bandwidth_unit: str = "decimal_GBps") -> None:
        """Make congestion-aware ASTRA authoritative for direct peer copies."""
        if self.config.pd_peer_transfer_mode != "direct-fabric":
            raise ValueError(
                "External fabric mode requires pd_peer_transfer_mode="
                "'direct-fabric'")
        if backend != "analytical-congestion-aware":
            raise ValueError(
                "External cold-HBM fabric requires "
                "--network-backend analytical-congestion-aware")
        bandwidth = float(physical_bandwidth_gbps)
        latency_ns = int(physical_latency_ns)
        bandwidth_unit = str(physical_bandwidth_unit)
        if bandwidth_unit != "decimal_GBps":
            raise ValueError(
                "External cold-HBM fabric bandwidth must be declared in "
                "decimal_GBps, matching AgenticKVConfig bandwidth units; "
                f"got {bandwidth_unit!r}")
        if not math.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("Physical fabric bandwidth must be positive")
        if latency_ns < 0:
            raise ValueError("Physical fabric latency must be non-negative")
        if not math.isclose(
                bandwidth, float(self.config.pd_peer_bandwidth_gbps),
                rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                "Agentic pd_peer_bandwidth_gbps disagrees with cluster "
                f"link_bw: agentic={self.config.pd_peer_bandwidth_gbps}, "
                f"cluster={bandwidth}. The ASTRA cluster link is authoritative.")
        configured_latency_ns = int(round(
            float(self.config.pd_peer_latency_us) * 1_000))
        if configured_latency_ns != latency_ns:
            raise ValueError(
                "Agentic pd_peer_latency_us disagrees with cluster "
                f"link_latency: agentic_ns={configured_latency_ns}, "
                f"cluster_ns={latency_ns}. The ASTRA cluster link is authoritative.")
        self._external_fabric_enabled = True
        self._external_fabric_authority = {
            "backend": str(backend),
            "bandwidth_gbps": bandwidth,
            "bandwidth_unit": bandwidth_unit,
            "latency_ns": latency_ns,
            "completion_source": "astra_event_queue_callback",
        }

    @property
    def external_fabric_enabled(self) -> bool:
        return bool(self._external_fabric_enabled)

    def drain_external_fabric_jobs(self) -> list[dict]:
        """Return each newly queued ASTRA job exactly once."""
        jobs = []
        while self._external_fabric_outgoing:
            job_id = self._external_fabric_outgoing.pop(0)
            restore = self._external_fabric_by_job.get(job_id)
            if restore is None:
                raise RuntimeError(
                    f"Queued external fabric job disappeared: {job_id}")
            if restore.status != "queued":
                raise RuntimeError(
                    "External fabric job was queued more than once: "
                    f"job={job_id}, status={restore.status}")
            restore.status = "issued"
            self.metrics.external_fabric_jobs_issued += 1
            self.events.append({
                "time_ns": restore.arrival_time_ns,
                "session_id": restore.session_id,
                "event": "external_fabric_restore_issued",
                "job_id": restore.job_id,
                "source_instance_id": restore.source_instance_id,
                "target_instance_id": restore.target_instance_id,
                "bytes_per_lane": restore.bytes_per_lane,
                "lane_count": restore.lane_count,
                "bytes": restore.total_bytes,
            })
            jobs.append({
                "job_id": restore.job_id,
                "arrival_ns": restore.arrival_time_ns,
                "bytes_per_lane": restore.bytes_per_lane,
                "lane_count": restore.lane_count,
                "source_instance_id": restore.source_instance_id,
                "target_instance_id": restore.target_instance_id,
            })
        return jobs

    def has_pending_external_fabric_jobs(self) -> bool:
        return bool(self._external_fabric_by_job)

    def next_internal_event_time(self, after_ns: Optional[int] = None):
        """Return the next Python-owned tier event after ``after_ns``."""
        floor_ns = (
            self._logical_frontier_ns if after_ns is None else int(after_ns))
        pinned = self._capacity_pinned_sessions()
        candidates = [
            int(pending.ready_ns)
            for pending in self.pending_source_releases
            if int(pending.ready_ns) > floor_ns
        ]
        candidates.extend(
            int(pending.ready_ns)
            for pending in self.pending_hbm_allocations
            if int(pending.ready_ns) > floor_ns
        )
        for entry in self.entries.values():
            # Pinning prevents a *new* reclaim/TTL action.  It must never hide
            # the completion of a migration that the returning owner is
            # explicitly waiting for.
            if (entry.session_id in pinned
                    and entry.migration_kind is None):
                continue
            event_ns = self._next_entry_event_ns(entry)
            if event_ns is not None and int(event_ns) > floor_ns:
                candidates.append(int(event_ns))
        for session_id, record in self.ssd_records.items():
            if session_id in pinned:
                continue
            entry = self.entries.get(session_id)
            if entry is not None and entry.location == KVLocation.SSD:
                continue
            event_ns = self._ssd_record_expiry_ns(session_id, record)
            if int(event_ns) > floor_ns:
                candidates.append(int(event_ns))
        return min(candidates) if candidates else None

    def complete_external_fabric_job(
            self, *, job_id: str, arrival_ns: int, completion_ns: int,
            bytes_per_lane: int, lane_count: int,
            critical_lane_start_ns: int) -> bool:
        """Commit one exact ASTRA completion; duplicate callbacks are harmless."""
        callback = (
            int(arrival_ns), int(completion_ns), int(bytes_per_lane),
            int(lane_count), int(critical_lane_start_ns),
        )
        tombstone = self._external_fabric_tombstones.get(str(job_id))
        if tombstone is not None:
            if tombstone != callback:
                raise RuntimeError(
                    "External fabric duplicate callback changed metadata: "
                    f"job={job_id}, first={tombstone}, duplicate={callback}")
            return False
        try:
            restore = self._external_fabric_by_job[str(job_id)]
        except KeyError as exc:
            raise RuntimeError(
                f"Unknown external fabric completion job_id={job_id!r}") from exc
        expected = (
            restore.arrival_time_ns, restore.bytes_per_lane,
            restore.lane_count,
        )
        observed = (callback[0], callback[2], callback[3])
        if observed != expected:
            raise RuntimeError(
                "External fabric completion disagrees with issued job: "
                f"job={job_id}, expected={expected}, observed={observed}")
        if restore.status == "completed":
            prior = (
                restore.arrival_time_ns, restore.completion_time_ns,
                restore.bytes_per_lane, restore.lane_count,
                restore.critical_lane_start_ns,
            )
            if prior != callback:
                raise RuntimeError(
                    "External fabric duplicate callback changed metadata: "
                    f"job={job_id}, first={prior}, duplicate={callback}")
            return False
        if restore.status != "issued":
            raise RuntimeError(
                "External fabric completion preceded command issue: "
                f"job={job_id}, status={restore.status}")
        if not (restore.arrival_time_ns <= callback[4] <= callback[1]):
            raise RuntimeError(
                "External fabric critical lane timing is non-causal: "
                f"job={job_id}, arrival={restore.arrival_time_ns}, "
                f"start={callback[4]}, completion={callback[1]}")

        self.advance(callback[1])
        if self.entries.get(restore.session_id) is not restore.source_entry:
            raise RuntimeError(
                "External fabric source ownership changed before completion: "
                f"session={restore.session_id}")
        pending_target = any(
            allocation.entry is restore.target_entry
            for allocation in self.pending_hbm_allocations
        )
        if pending_target:
            raise RuntimeError(
                "External fabric completed before target HBM reservation: "
                f"job={job_id}")

        restore.status = "completed"
        restore.completion_time_ns = callback[1]
        restore.critical_lane_start_ns = callback[4]
        queue_wait_ns = callback[4] - callback[0]
        service_ns = callback[1] - callback[4]
        self.metrics.transfer_jobs += 1
        if queue_wait_ns:
            self.metrics.queued_transfer_jobs += 1
        self.metrics.migration_queue_wait_ns += queue_wait_ns
        self.metrics.migration_service_ns += service_ns
        self.metrics.external_fabric_jobs_completed += 1
        self.metrics.external_fabric_lane_bytes += (
            restore.bytes_per_lane * restore.lane_count)
        self._external_fabric_completion_generation[
            restore.target_instance_id] += 1
        event = {
            "time_ns": restore.arrival_time_ns,
            "session_id": restore.session_id,
            "event": "migration_reserve",
            "kind": "hbm_peer_external_astra",
            "arrival_ns": restore.arrival_time_ns,
            "job_arrival_ns": restore.release_time_ns,
            "start_ns": callback[4],
            "complete_ns": callback[1],
            "queue_wait_ns": queue_wait_ns,
            "service_ns": service_ns,
            "bytes": restore.total_bytes,
            "foreground": True,
            "resources": ["astra:shared-network-and-endpoints"],
            "job_id": restore.job_id,
            "source_instance_id": restore.source_instance_id,
            "target_instance_id": restore.target_instance_id,
            "bytes_per_lane": restore.bytes_per_lane,
            "lane_count": restore.lane_count,
        }
        self.events.append(event)
        self._external_fabric_history.append(dict(event))
        return True

    def censor_completed_external_fabric_job(
            self, job_id: str, now_ns: int) -> None:
        """Release a completed target reservation after measurement freeze."""
        try:
            restore = self._external_fabric_by_job[str(job_id)]
        except KeyError as exc:
            raise RuntimeError(
                f"Cannot censor unknown external fabric job {job_id!r}") from exc
        if (restore.status != "completed"
                or restore.completion_time_ns is None
                or restore.critical_lane_start_ns is None):
            raise RuntimeError(
                "Only a completed external fabric job can be censored: "
                f"job={job_id}, status={restore.status}")
        self.advance(int(now_ns))
        self._cancel_hbm_reservation(
            restore.target_entry, restore.arrival_time_ns)
        callback = (
            restore.arrival_time_ns,
            restore.completion_time_ns,
            restore.bytes_per_lane,
            restore.lane_count,
            restore.critical_lane_start_ns,
        )
        self._external_fabric_tombstones[restore.job_id] = callback
        self._external_fabric_by_job.pop(restore.job_id)
        self._external_fabric_by_session.pop(restore.session_id)
        self._set_restore_capacity_pin(restore.session_id, False)
        self.metrics.external_fabric_jobs_censored += 1
        self.metrics.external_fabric_censored_lane_bytes += (
            restore.bytes_per_lane * restore.lane_count)
        self.events.append({
            "time_ns": int(now_ns),
            "session_id": restore.session_id,
            "event": "external_fabric_restore_censored",
            "job_id": restore.job_id,
            "bytes": restore.total_bytes,
        })

    def synchronous_prepare_locked(self, instance_id: int) -> bool:
        instance_id = int(instance_id)
        return any(
            instance_id in instance_ids
            for instance_ids in self._sync_prepare_locks.values()
        )

    def synchronous_prepare_instances(
            self, session_id: str, target_instance_id: int,
            reuse_tokens: int, now_ns: int) -> tuple[int, ...]:
        """Return engines that must reach an iteration boundary before swap."""
        if not self.synchronous_swap_enabled or int(reuse_tokens) <= 0:
            return ()
        entry = self.entries.get(session_id)
        if entry is None or entry.location == KVLocation.DROPPED:
            return ()
        target_instance_id = int(target_instance_id)
        if entry.location == KVLocation.HBM:
            # A same-instance hit performs no swap. Cross-instance HBM reuse
            # can still force target-side LRU demotion before the peer copy.
            if entry.instance_id == target_instance_id:
                return ()
            target = self._scheduler(target_instance_id)
            reusable = min(int(reuse_tokens), int(entry.tokens))
            block_tokens = (
                (reusable + self.config.block_size - 1)
                // self.config.block_size
                * self.config.block_size
            )
            needed_per_rank = int(target.memory.get_kv(block_tokens))
            if self.synchronous_hbm_reclaim_needs_boundary(
                    target_instance_id, needed_per_rank, int(now_ns)):
                return (target_instance_id,)
            return ()
        return tuple(dict.fromkeys(
            (int(entry.instance_id), target_instance_id)))

    def direct_fabric_prepare_instances(
            self, session_id: str, target_instance_id: int,
            reuse_tokens: int, now_ns: int) -> tuple[int, ...]:
        """Return no model boundary for asynchronous cold peer copies."""
        del session_id, target_instance_id, reuse_tokens, now_ns
        return ()

    def prepare_boundary_instances(
            self, session_id: str, target_instance_id: int,
            reuse_tokens: int, now_ns: int) -> tuple[int, ...]:
        """Return all engine boundaries required before foreground restore."""
        return self.synchronous_prepare_instances(
            session_id, target_instance_id, reuse_tokens, now_ns)

    def acquire_prepare_lock(
            self, request_id: int, instance_ids: Sequence[int],
            session_id: Optional[str] = None) -> None:
        """Hold required model boundaries without changing current batches."""
        instance_ids = tuple(dict.fromkeys(
            int(instance_id) for instance_id in instance_ids))
        if self.synchronous_swap_enabled:
            self.acquire_synchronous_prepare_lock(
                request_id, instance_ids, session_id=session_id)

    def release_prepare_lock(self, request_id: int) -> None:
        self.release_synchronous_prepare_lock(request_id)
        request_id = int(request_id)
        pinned_before = self._capacity_pinned_sessions()
        self._fabric_prepare_locks.pop(request_id, None)
        self._fabric_prepare_sessions.pop(request_id, None)
        self._publish_capacity_pin_change(
            pinned_before, self._capacity_pinned_sessions())

    def prepare_locked(self, instance_id: int) -> bool:
        return self.synchronous_prepare_locked(int(instance_id))

    def prepare_boundary_busy(
            self, instance_ids: Sequence[int]) -> bool:
        """Whether a requested boundary still has actual execution in flight.

        A formed DP batch in ``scheduler.inflight`` may still be waiting in
        ``dp_pending`` and own no hardware. Treating it as a direct-fabric user
        creates a cycle: the restore waits for the undispatched batch while the
        batch later waits for the restore. Synchronous-swap sensitivity keeps
        its stricter engine rule; asynchronous direct-fabric mode consults only
        controller-dispatched ASTRA owners.
        """
        instance_ids = tuple(dict.fromkeys(
            int(instance_id) for instance_id in instance_ids))
        if self.synchronous_swap_enabled and any(
                self._scheduler(instance_id).inflight
                for instance_id in instance_ids):
            return True
        return False

    def synchronous_hbm_reclaim_needs_boundary(
            self, instance_id: int, needed_per_rank_bytes: int,
            now_ns: int) -> bool:
        """Whether a new GPU-facing LRU demotion must start for admission."""
        if (not self.synchronous_swap_enabled
                or int(needed_per_rank_bytes) <= 0
                or int(instance_id) in self._active_hbm_reclaim_claims
                or self.config.policy not in {
                    "cpu", "tiered", "tiered_queue_recompute",
                    "hbm_ssd_direct"
                }):
            return False
        instance_id = int(instance_id)
        needed_per_rank_bytes = int(needed_per_rank_bytes)
        if self._hbm_capacity_time(
                instance_id, needed_per_rank_bytes, int(now_ns)) is not None:
            return False
        scheduler = self._scheduler(instance_id)
        pinned = self._capacity_pinned_sessions()
        candidates = [
            victim for victim in self.entries.values()
            if victim.instance_id == instance_id
            and victim.location == KVLocation.HBM
            and victim.migration_kind is None
            and victim.session_id not in pinned
        ]
        scheduled = sum(
            victim.per_rank_bytes for victim in self.entries.values()
            if victim.instance_id == instance_id
            and victim.location == KVLocation.HBM
            and victim.migration_complete_ns is not None
            and victim.migration_kind in {
                "hbm_to_cpu", "hbm_to_ssd", "hbm_to_ssd_direct"
            }
        )
        reclaimable = sum(victim.per_rank_bytes for victim in candidates)
        can_reclaim = (
            self._hbm_avail(scheduler)
            + scheduled
            + reclaimable
            - self._hbm_logically_reserved(instance_id)
            >= needed_per_rank_bytes
        )
        return bool(candidates) and can_reclaim

    def record_synchronous_swap_dispatch_block(
            self, instance_id: int, now_ns: int,
            ready_requests: Sequence[object]) -> Optional[int]:
        """Record the exposed tail when runnable work hits a swap gate.

        The raw reservation begins when the copy is issued. A causal model-
        engine stall begins only when this scheduler could otherwise dispatch
        a batch. Repeated polls and overlapping reservations are retained as
        intervals and unioned at attribution/report time.
        """
        if not self.synchronous_swap_enabled or not ready_requests:
            return None
        instance_id = int(instance_id)
        now_ns = int(now_ns)
        active = list(self._active_sync_swap_barriers(instance_id, now_ns))
        if not active:
            return None
        blocked_until = max(barrier[1] for barrier in active)
        directions = tuple(sorted({barrier[2] for barrier in active}))
        exposed = (now_ns, blocked_until, directions)
        exposures = self._sync_exposed_barriers[instance_id]
        is_new_exposure = not exposures or exposures[-1] != exposed
        if is_new_exposure:
            exposures.append(exposed)

        blocked_sessions = {
            barrier[4] for barrier in active if barrier[4] is not None
        }
        victims = [
            request for request in ready_requests
            if getattr(request, "session_id", None) not in blocked_sessions
        ]
        if is_new_exposure:
            self.metrics.sync_swap_ready_victim_memberships += len(victims)
            self.metrics.sync_swap_max_ready_victims = max(
                self.metrics.sync_swap_max_ready_victims, len(victims))
        victim_ids = []
        for request in victims:
            request_id = int(request.id)
            victim_key = (instance_id, request_id)
            self._sync_ready_victim_intervals.setdefault(
                victim_key, []).append((now_ns, blocked_until))
            self._sync_unique_ready_victims.add(victim_key)
            victim_ids.append(request_id)
        if is_new_exposure:
            self.events.append({
                "time_ns": now_ns,
                "event": "sync_swap_dispatch_block",
                "instance_id": instance_id,
                "blocked_until_ns": blocked_until,
                "directions": list(directions),
                "ready_request_count": len(ready_requests),
                "ready_victim_request_ids": victim_ids,
            })
        return blocked_until

    def _sync_swap_barrier_for_batch(
            self, instance_id: int,
            batch_time_ns: int) -> tuple[int, tuple[str, ...]]:
        """Attach the contiguous barrier ending at this batch dispatch.

        The online simulator realizes synchronous swap as a pre-dispatch
        engine gate. Associating the gate with the immediately following
        iteration is causally equivalent to an in-iteration barrier while
        avoiding double-counting the returning request's ready-time delay.
        """
        if not self.synchronous_swap_enabled:
            return 0, ()
        exposed = self._sync_exposed_barriers.get(instance_id, ())
        cursor = self._sync_exposed_cursor.get(instance_id, 0)
        stop = cursor
        while stop < len(exposed) and exposed[stop][1] <= int(batch_time_ns):
            stop += 1
        pending = list(exposed[cursor:stop])
        self._sync_exposed_cursor[instance_id] = stop
        if not pending:
            return 0, ()
        components = []
        for start_ns, end_ns, directions in sorted(
                pending, key=lambda item: (item[0], item[1])):
            if components and start_ns <= components[-1][1]:
                old_start, old_end, old_directions = components[-1]
                components[-1] = (
                    old_start,
                    max(old_end, end_ns),
                    old_directions | set(directions),
                )
            else:
                components.append((start_ns, end_ns, set(directions)))
        matching = [
            component for component in components
            if component[1] == int(batch_time_ns)
        ]
        if not matching:
            return 0, ()
        start_ns, end_ns, directions = matching[-1]
        return end_ns - start_ns, tuple(sorted(directions))

    def _earliest_resource_slot(
            self, resources: Sequence[str], arrival_ns: int,
            service_ns: int) -> int:
        """Find the first common immutable gap across every resource.

        Older parent jobs may prebook a later stage. A newer short job can
        backfill a gap only when it finishes before every conflicting booked
        interval; it never moves an already-published completion timestamp.
        """
        candidate_ns = int(arrival_ns)
        service_ns = max(0, int(service_ns))
        if service_ns == 0 or not resources:
            return candidate_ns
        # A few compatibility tests and external callers seed a pre-existing
        # busy watermark without going through this manager. Treat it as an
        # opaque interval only when no detailed calendar exists.
        candidate_ns = max(
            candidate_ns,
            max((
                self._resource_busy_until.get(resource, 0)
                for resource in resources
                if not self._resource_intervals.get(resource)
            ), default=candidate_ns),
        )
        while True:
            shifted_ns = candidate_ns
            candidate_end_ns = candidate_ns + service_ns
            for resource in resources:
                for start_ns, end_ns, _, _ in self._resource_intervals.get(
                        resource, ()):
                    if end_ns <= candidate_ns:
                        continue
                    if start_ns >= candidate_end_ns:
                        break
                    shifted_ns = max(shifted_ns, int(end_ns))
            if shifted_ns == candidate_ns:
                return candidate_ns
            candidate_ns = shifted_ns

    def _insert_resource_interval(
            self, resources: Sequence[str], start_ns: int, complete_ns: int,
            reservation_sequence: int, kind: str) -> None:
        """Insert one non-empty interval and reject any scheduler overlap."""
        start_ns = int(start_ns)
        complete_ns = int(complete_ns)
        if complete_ns <= start_ns:
            return
        interval = (
            start_ns, complete_ns, int(reservation_sequence), str(kind))
        for resource in resources:
            calendar = self._resource_intervals.setdefault(resource, [])
            index = bisect.bisect_left(calendar, interval)
            if index > 0 and calendar[index - 1][1] > start_ns:
                raise RuntimeError(
                    "Migration resource calendar overlap before insertion: "
                    f"resource={resource}, previous={calendar[index - 1]}, "
                    f"new={interval}")
            if index < len(calendar) and calendar[index][0] < complete_ns:
                raise RuntimeError(
                    "Migration resource calendar overlap after insertion: "
                    f"resource={resource}, next={calendar[index]}, "
                    f"new={interval}")
            calendar.insert(index, interval)
            self._resource_busy_until[resource] = max(
                self._resource_busy_until.get(resource, 0), complete_ns)
            self._resource_busy_ns[resource] = (
                self._resource_busy_ns.get(resource, 0)
                + complete_ns - start_ns
            )
            self._resource_jobs[resource] = (
                self._resource_jobs.get(resource, 0) + 1)

    def _reserve_transfer(
            self, *, kind: str, arrival_ns: int, service_ns: int,
            source_instance_id: int, target_instance_id: Optional[int],
            num_bytes: int, background: bool,
            deadline_ns: Optional[int] = None,
            session_id: Optional[str] = None,
            ssd_write_phase_offset_ns: int = 0,
            ssd_write_phase_service_ns: Optional[int] = None,
            register_sync_barrier: bool = True,
            job_arrival_ns: Optional[int] = None,
            parent_reservation: Optional[TransferReservation] = None,
            ) -> TransferReservation:
        """Reserve a transfer directly on trace-time resources.

        The policy is deterministic non-preemptive parent-job FCFS. Top-level
        jobs must be submitted in logical-arrival order. An accepted job's
        stages are immutable; later jobs may fill calendar holes but cannot
        move a published owner-ready timestamp. Background copies have a
        next-use deadline. If they cannot commit by that deadline, the partial
        copy is cancelled and its source remains authoritative.
        """
        resources = self._transfer_resources(
            kind, source_instance_id, target_instance_id)
        self._assert_direct_fabric_boundary(resources, kind)
        arrival_ns = int(arrival_ns)
        logical_arrival_ns = int(
            arrival_ns if job_arrival_ns is None else job_arrival_ns)
        if parent_reservation is None:
            if logical_arrival_ns < self._last_transfer_job_arrival_ns:
                raise RuntimeError(
                    "Agentic migration job arrived behind a later job: "
                    f"kind={kind}, arrival={logical_arrival_ns}, "
                    "last_arrival="
                    f"{self._last_transfer_job_arrival_ns}. Route due "
                    "continuations before advancing later completion events.")
            self._last_transfer_job_arrival_ns = logical_arrival_ns
            parent_sequence = self._transfer_sequence
        else:
            parent_sequence = int(parent_reservation.parent_sequence)
            if logical_arrival_ns != int(parent_reservation.job_arrival_ns):
                raise RuntimeError(
                    "Atomic transfer child changed parent logical arrival: "
                    f"parent={parent_reservation.job_arrival_ns}, "
                    f"child={logical_arrival_ns}")
        reservation_sequence = self._transfer_sequence
        self._transfer_sequence += 1
        service_ns = max(0, int(service_ns))
        start_ns = self._earliest_resource_slot(
            resources, arrival_ns, service_ns)
        start_ns = self._after_completed_astra_windows(
            resources, start_ns, service_ns)
        complete_ns = start_ns + service_ns
        queue_wait_ns = max(0, start_ns - arrival_ns)
        self.metrics.transfer_jobs += 1

        if deadline_ns is not None and complete_ns > deadline_ns:
            deadline_ns = int(deadline_ns)
            effective_start_ns = min(start_ns, deadline_ns)
            effective_wait_ns = max(0, effective_start_ns - arrival_ns)
            active_ns = max(0, deadline_ns - start_ns)
            if effective_wait_ns:
                self.metrics.queued_transfer_jobs += 1
            self.metrics.migration_queue_wait_ns += effective_wait_ns
            self.metrics.background_queue_wait_ns += effective_wait_ns
            self.metrics.migration_service_ns += active_ns
            self.metrics.background_service_ns += active_ns
            self.metrics.background_cancelled_jobs += 1
            self.metrics.background_cancelled_bytes += max(0, int(num_bytes))
            wasted_bytes = (
                min(
                    max(0, int(num_bytes)),
                    math.ceil(num_bytes * active_ns / service_ns),
                )
                if service_ns > 0 else 0
            )
            self.metrics.background_wasted_bytes += wasted_bytes
            if kind in {
                    "hbm_to_ssd", "cpu_to_ssd", "hbm_to_ssd_direct"}:
                self.metrics.ssd_demotion_cancelled += 1
                # A cancelled copy can still have issued NAND writes. Model
                # progress only within the SSD media phase. For an HBM source,
                # an early cancellation during the preceding CPU stage has
                # issued no SSD host writes yet.
                write_service_ns = (
                    service_ns if ssd_write_phase_service_ns is None
                    else max(0, int(ssd_write_phase_service_ns))
                )
                write_active_ns = max(
                    0, active_ns - max(0, int(ssd_write_phase_offset_ns)))
                issued_write_bytes = (
                    min(
                        max(0, int(num_bytes)),
                        math.ceil(
                            num_bytes * write_active_ns / write_service_ns),
                    )
                    if write_service_ns > 0 else 0
                )
                self.metrics.ssd_host_write_bytes += issued_write_bytes
                self.metrics.ssd_cancelled_host_write_bytes += (
                    issued_write_bytes)
            if active_ns:
                self._insert_resource_interval(
                    resources, start_ns, deadline_ns,
                    reservation_sequence, kind)
            self.events.append({
                "time_ns": deadline_ns,
                "session_id": session_id,
                "event": "migration_cancel",
                "kind": kind,
                "arrival_ns": arrival_ns,
                "job_arrival_ns": logical_arrival_ns,
                "start_ns": effective_start_ns,
                "candidate_start_ns": start_ns,
                "queue_wait_ns": effective_wait_ns,
                "active_ns": active_ns,
                "bytes": int(num_bytes),
                "wasted_bytes": wasted_bytes,
                "foreground": not background,
                "complete_ns": deadline_ns,
                "resources": list(resources),
                "ssd_issued_write_bytes": (
                    issued_write_bytes
                    if kind in {
                        "hbm_to_ssd", "cpu_to_ssd", "hbm_to_ssd_direct"
                    } else 0),
            })
            reservation = TransferReservation(
                kind=kind,
                arrival_ns=arrival_ns,
                start_ns=effective_start_ns,
                complete_ns=deadline_ns,
                service_ns=service_ns,
                queue_wait_ns=effective_wait_ns,
                resources=resources,
                completed=False,
                active_ns_before_cancel=active_ns,
                reservation_sequence=reservation_sequence,
                parent_sequence=parent_sequence,
                job_arrival_ns=logical_arrival_ns,
            )
            if register_sync_barrier:
                self._register_sync_swap_barrier(
                    reservation,
                    source_instance_id,
                    target_instance_id,
                    session_id,
                    exposes_owner_request=not background,
                )
            return reservation

        if queue_wait_ns:
            self.metrics.queued_transfer_jobs += 1
        self.metrics.migration_queue_wait_ns += queue_wait_ns
        self.metrics.migration_service_ns += service_ns
        if background:
            self.metrics.background_queue_wait_ns += queue_wait_ns
            self.metrics.background_service_ns += service_ns
        else:
            self.metrics.critical_restore_queue_wait_ns += queue_wait_ns
            self.metrics.critical_restore_service_ns += service_ns
        self._insert_resource_interval(
            resources, start_ns, complete_ns,
            reservation_sequence, kind)
        self.events.append({
            "time_ns": arrival_ns,
            "session_id": session_id,
            "event": "migration_reserve",
            "kind": kind,
            "job_arrival_ns": logical_arrival_ns,
            "start_ns": start_ns,
            "complete_ns": complete_ns,
            "service_ns": service_ns,
            "queue_wait_ns": queue_wait_ns,
            "bytes": int(num_bytes),
            "foreground": not background,
            "resources": list(resources),
        })
        reservation = TransferReservation(
            kind=kind,
            arrival_ns=arrival_ns,
            start_ns=start_ns,
            complete_ns=complete_ns,
            service_ns=service_ns,
            queue_wait_ns=queue_wait_ns,
            resources=resources,
            reservation_sequence=reservation_sequence,
            parent_sequence=parent_sequence,
            job_arrival_ns=logical_arrival_ns,
        )
        if register_sync_barrier:
            self._register_sync_swap_barrier(
                reservation,
                source_instance_id,
                target_instance_id,
                session_id,
                exposes_owner_request=not background,
            )
        return reservation

    def _node_cpu_capacity_events(
            self, scheduler, now_ns: int,
            ) -> tuple[int, int, list[tuple[int, int]]]:
        """Return node DRAM capacity, current slack, and future slack deltas."""
        now_ns = int(now_ns)
        node_id = self._node_id(scheduler)
        capacity = self._cpu_capacity_bytes(scheduler)
        available = self._cpu_avail(scheduler)
        events: list[tuple[int, int]] = []
        for candidate in self.entries.values():
            if self._node_id(self._scheduler(candidate.instance_id)) != node_id:
                continue
            complete_ns = candidate.migration_complete_ns
            if complete_ns is None or int(complete_ns) <= now_ns:
                continue
            if (candidate.location == KVLocation.CPU
                    and candidate.migration_kind == "cpu_to_ssd"):
                events.append((int(complete_ns), candidate.total_bytes))
            elif (candidate.location == KVLocation.HBM
                    and candidate.migration_kind == "hbm_to_cpu"):
                events.append((int(complete_ns), -candidate.total_bytes))
        for pending in self.pending_source_releases:
            candidate = pending.entry
            if (candidate.location == KVLocation.CPU
                    and pending.ready_ns > now_ns
                    and self._node_id(
                        self._scheduler(candidate.instance_id)) == node_id):
                events.append((int(pending.ready_ns), candidate.total_bytes))
        for reservation in self._transient_dram_reservations.get(node_id, ()):
            if reservation.complete_ns <= now_ns:
                continue
            if reservation.start_ns <= now_ns:
                available -= reservation.bytes
            else:
                events.append((reservation.start_ns, -reservation.bytes))
            events.append((reservation.complete_ns, reservation.bytes))
        return capacity, available, events

    def _transient_dram_window_capacity(
            self, scheduler, start_ns: int, complete_ns: int,
            num_bytes: int, now_ns: int,
            ) -> tuple[Optional[int], int, list[tuple[int, int]]]:
        """Check full-object capacity over a proposed bounce-buffer lifetime."""
        capacity, available, raw_events = self._node_cpu_capacity_events(
            scheduler, now_ns)
        grouped: Dict[int, int] = {}
        for event_ns, delta in raw_events:
            grouped[int(event_ns)] = grouped.get(int(event_ns), 0) + int(delta)
        ordered = sorted(grouped.items())
        for event_ns, delta in ordered:
            if event_ns > start_ns:
                break
            available += delta
        peak_committed = capacity - available + int(num_bytes)
        if available < num_bytes:
            return int(start_ns), peak_committed, ordered
        for event_ns, delta in ordered:
            if event_ns <= start_ns:
                continue
            if event_ns >= complete_ns:
                break
            available += delta
            peak_committed = max(
                peak_committed, capacity - available + int(num_bytes))
            if available < num_bytes:
                return int(event_ns), peak_committed, ordered
        return None, peak_committed, ordered

    def _plan_ssd_restore_stages(
            self, *, arrival_ns: int, staging_instance_id: int,
            target_instance_id: int, per_rank_bytes: int,
            total_bytes: int) -> tuple[int, int, int, int, int, int]:
        media_service_ns = self._ssd_to_cpu_stage_ns(int(total_bytes))
        media_resources = self._transfer_resources(
            "ssd_to_cpu_stage", int(staging_instance_id), None)
        media_start_ns = self._earliest_resource_slot(
            media_resources, int(arrival_ns), media_service_ns)
        media_start_ns = self._after_completed_astra_windows(
            media_resources, media_start_ns, media_service_ns)
        media_complete_ns = media_start_ns + media_service_ns
        h2d_service_ns = self._cpu_transfer_ns(
            int(per_rank_bytes), int(total_bytes))
        h2d_resources = self._transfer_resources(
            "cpu_stage_to_hbm", int(staging_instance_id),
            int(target_instance_id))
        h2d_start_ns = self._earliest_resource_slot(
            h2d_resources, media_complete_ns, h2d_service_ns)
        h2d_start_ns = self._after_completed_astra_windows(
            h2d_resources, h2d_start_ns, h2d_service_ns)
        h2d_complete_ns = h2d_start_ns + h2d_service_ns
        return (
            media_start_ns, media_complete_ns, media_service_ns,
            h2d_start_ns, h2d_complete_ns, h2d_service_ns,
        )

    def _project_lower_tier_restore_queue(
            self, *, source: KVLocation, arrival_ns: int,
            staging_instance_id: int, target_instance_id: int,
            per_rank_bytes: int, total_bytes: int,
            ) -> tuple[int, int, tuple[str, ...]]:
        """Project immutable foreground queue wait without reserving I/O.

        The caller has already reserved destination HBM, so ``arrival_ns`` is
        the earliest physical transfer issue time. Existing resource-calendar
        intervals are immutable: this snapshot cannot move an accepted job or
        consume a queue slot. SSD restores retain their two serial stages and
        sum only the gaps before each stage, never dependency service.
        """
        arrival_ns = int(arrival_ns)
        staging_instance_id = int(staging_instance_id)
        target_instance_id = int(target_instance_id)
        per_rank_bytes = int(per_rank_bytes)
        total_bytes = int(total_bytes)
        if source == KVLocation.CPU:
            service_ns = self._cpu_transfer_ns(
                per_rank_bytes, total_bytes)
            resources = self._transfer_resources(
                "cpu_to_hbm", staging_instance_id, target_instance_id)
            start_ns = self._earliest_resource_slot(
                resources, arrival_ns, service_ns)
            start_ns = self._after_completed_astra_windows(
                resources, start_ns, service_ns)
            return (
                max(0, start_ns - arrival_ns),
                service_ns,
                ("cpu_to_hbm",),
            )
        if source == KVLocation.SSD:
            planned = self._plan_ssd_restore_stages(
                arrival_ns=arrival_ns,
                staging_instance_id=staging_instance_id,
                target_instance_id=target_instance_id,
                per_rank_bytes=per_rank_bytes,
                total_bytes=total_bytes,
            )
            queue_wait_ns = (
                planned[0] - arrival_ns
                + planned[3] - planned[1]
            )
            service_ns = planned[2] + planned[5]
            return (
                max(0, queue_wait_ns),
                service_ns,
                ("ssd_to_cpu_stage", "cpu_stage_to_hbm"),
            )
        raise ValueError(
            "Queue-aware restore projection requires a CPU or SSD source, "
            f"got {source.value!r}")

    def _fork_hbm_projection_manager(
            self, protected_session_id: str) -> "AgenticKVManager":
        """Return a purpose-built, mutation-isolated HBM reservation shadow.

        Deep-copying the full manager would recurse through live schedulers,
        loggers, and the external ASTRA authority.  The HBM admission path has
        a much smaller mutation surface: entry migration state, memory
        counters, transfer calendars, metrics/events, and synchronous swap
        barriers.  Clone exactly that surface and share only objects which the
        reservation path treats as read-only.
        """
        shadow = copy.copy(self)

        entry_clones: Dict[int, IdleKVEntry] = {}

        def clone_entry(entry: IdleKVEntry) -> IdleKVEntry:
            key = id(entry)
            clone = entry_clones.get(key)
            if clone is None:
                clone = copy.copy(entry)
                entry_clones[key] = clone
            return clone

        shadow.schedulers = {}
        for instance_id, scheduler in self.schedulers.items():
            shadow_scheduler = copy.copy(scheduler)
            shadow_scheduler.memory = copy.copy(scheduler.memory)
            shadow_scheduler.agentic_kv_manager = shadow
            shadow.schedulers[instance_id] = shadow_scheduler
        shadow.entries = {
            session_id: clone_entry(entry)
            for session_id, entry in self.entries.items()
        }
        shadow.ssd_records = {
            session_id: copy.copy(record)
            for session_id, record in self.ssd_records.items()
        }
        shadow.pending_source_releases = [
            PendingSourceRelease(
                entry=clone_entry(pending.entry),
                ready_ns=pending.ready_ns,
                remove_ssd_record=pending.remove_ssd_record,
            )
            for pending in self.pending_source_releases
        ]
        shadow.pending_hbm_allocations = [
            PendingHBMAllocation(
                entry=clone_entry(pending.entry),
                ready_ns=pending.ready_ns,
            )
            for pending in self.pending_hbm_allocations
        ]
        shadow._active_hbm_reclaim_claims = dict(
            self._active_hbm_reclaim_claims)
        shadow._direct_ssd_capacity_reservations = dict(
            self._direct_ssd_capacity_reservations)
        shadow._direct_ssd_capacity_reservation_nodes = dict(
            self._direct_ssd_capacity_reservation_nodes)
        shadow.metrics = copy.copy(self.metrics)
        # Projection events are useful for extracting the ordered LRU victims,
        # but historical live events must neither be copied nor appended to.
        shadow.events = []
        shadow._resource_busy_until = dict(self._resource_busy_until)
        shadow._resource_busy_ns = dict(self._resource_busy_ns)
        shadow._resource_jobs = dict(self._resource_jobs)
        shadow._resource_intervals = {
            resource: list(intervals)
            for resource, intervals in self._resource_intervals.items()
        }
        shadow._transient_dram_reservations = {
            node_id: list(reservations)
            for node_id, reservations
            in self._transient_dram_reservations.items()
        }
        shadow._transient_dram_history = {
            node_id: list(reservations)
            for node_id, reservations
            in self._transient_dram_history.items()
        }
        shadow._pending_restore_sessions = set(
            self._pending_restore_sessions)
        # The foreground source must not become a collateral CPU LRU victim.
        # Add the pin only to the shadow; a drop decision leaves live state
        # completely untouched.
        shadow._set_restore_capacity_pin(str(protected_session_id), True)
        shadow._sync_engine_barriers = {
            instance_id: list(barriers)
            for instance_id, barriers in self._sync_engine_barriers.items()
        }
        shadow._sync_barrier_upcoming = {
            instance_id: list(upcoming)
            for instance_id, upcoming in self._sync_barrier_upcoming.items()
        }
        shadow._sync_barrier_active = {
            instance_id: dict(active)
            for instance_id, active in self._sync_barrier_active.items()
        }
        shadow._sync_barrier_end_heap = {
            instance_id: list(end_heap)
            for instance_id, end_heap in self._sync_barrier_end_heap.items()
        }
        shadow._sync_exposed_barriers = {
            instance_id: list(exposed)
            for instance_id, exposed in self._sync_exposed_barriers.items()
        }
        shadow._sync_exposed_cursor = dict(self._sync_exposed_cursor)
        return shadow

    @staticmethod
    def _foreground_reservation_signature(
            reservations: Optional[Sequence[TransferReservation]]) -> tuple:
        """Return the exact immutable foreground reservation plan."""
        return tuple((
            reservation.kind,
            int(reservation.arrival_ns),
            int(reservation.start_ns),
            int(reservation.complete_ns),
            int(reservation.service_ns),
            int(reservation.queue_wait_ns),
            tuple(reservation.resources),
            bool(reservation.completed),
            int(reservation.reservation_sequence),
            int(reservation.parent_sequence),
            int(reservation.job_arrival_ns),
            int(reservation.transient_dram_capacity_wait_ns),
        ) for reservation in (reservations or ()))

    @staticmethod
    def _projection_lru_victims(
            events: Sequence[dict]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Extract ordered, unique HBM and CPU LRU victims from events."""
        hbm_kinds = {"hbm_capacity_demotion_scheduled"}
        cpu_kinds = {
            "cpu_capacity_eviction_scheduled",
            "transient_dram_cpu_lru_eviction_scheduled",
        }

        def ordered_unique(kinds: set[str]) -> tuple[str, ...]:
            seen = set()
            ordered = []
            for event in events:
                if event.get("event") not in kinds:
                    continue
                session_id = str(event["session_id"])
                if session_id in seen:
                    continue
                seen.add(session_id)
                ordered.append(session_id)
            return tuple(ordered)

        return ordered_unique(hbm_kinds), ordered_unique(cpu_kinds)

    def _hbm_reservation_fingerprint(self) -> tuple:
        """Return the state which an HBM reservation is allowed to mutate."""
        entry_migrations = tuple(sorted((
            session_id,
            entry.location.value,
            entry.migration_kind,
            entry.migration_start_ns,
            entry.migration_complete_ns,
            entry.migration_service_ns,
            entry.migration_queue_wait_ns,
            entry.migration_reason,
        ) for session_id, entry in self.entries.items()))
        ssd_pins = tuple(sorted(
            (session_id, record.pinned_until_ns)
            for session_id, record in self.ssd_records.items()
        ))
        memory_usage = tuple(sorted(
            (
                instance_id,
                int(scheduler.memory.npu_used),
                int(scheduler.memory.cpu_used),
            )
            for instance_id, scheduler in self.schedulers.items()
        ))
        pending_hbm = tuple(sorted((
            pending.entry.session_id,
            pending.entry.instance_id,
            pending.entry.per_rank_bytes,
            pending.entry.total_bytes,
            pending.ready_ns,
        ) for pending in self.pending_hbm_allocations))
        resource_intervals = tuple(sorted(
            (resource, tuple(intervals))
            for resource, intervals in self._resource_intervals.items()
        ))
        sync_barriers = tuple(sorted(
            (instance_id, tuple(barriers))
            for instance_id, barriers in self._sync_engine_barriers.items()
        ))
        sync_upcoming = tuple(sorted(
            (instance_id, tuple(upcoming))
            for instance_id, upcoming in self._sync_barrier_upcoming.items()
        ))
        sync_active = tuple(sorted(
            (instance_id, tuple(sorted(active.items())))
            for instance_id, active in self._sync_barrier_active.items()
        ))
        sync_end_heap = tuple(sorted(
            (instance_id, tuple(end_heap))
            for instance_id, end_heap in self._sync_barrier_end_heap.items()
        ))
        sync_exposed = tuple(sorted(
            (instance_id, tuple(exposed))
            for instance_id, exposed in self._sync_exposed_barriers.items()
        ))
        transient_dram = tuple(sorted(
            (
                node_id,
                tuple((
                    reservation.node_id,
                    reservation.session_id,
                    reservation.start_ns,
                    reservation.complete_ns,
                    reservation.bytes,
                    reservation.reservation_sequence,
                    reservation.peak_node_committed_bytes,
                ) for reservation in reservations),
            )
            for node_id, reservations
            in self._transient_dram_reservations.items()
        ))
        transient_dram_history = tuple(sorted(
            (
                node_id,
                tuple((
                    reservation.node_id,
                    reservation.session_id,
                    reservation.start_ns,
                    reservation.complete_ns,
                    reservation.bytes,
                    reservation.reservation_sequence,
                    reservation.peak_node_committed_bytes,
                ) for reservation in reservations),
            )
            for node_id, reservations
            in self._transient_dram_history.items()
        ))
        return (
            entry_migrations,
            ssd_pins,
            memory_usage,
            pending_hbm,
            resource_intervals,
            tuple(sorted(self._resource_busy_until.items())),
            tuple(sorted(self._resource_busy_ns.items())),
            tuple(sorted(self._resource_jobs.items())),
            int(self._transfer_sequence),
            int(self._last_transfer_job_arrival_ns),
            tuple(sorted(self._pending_restore_sessions)),
            tuple(sorted(self._direct_ssd_capacity_reservations.items())),
            tuple(sorted(
                self._direct_ssd_capacity_reservation_nodes.items())),
            sync_barriers,
            sync_upcoming,
            sync_active,
            sync_end_heap,
            sync_exposed,
            tuple(sorted(self._sync_exposed_cursor.items())),
            transient_dram,
            transient_dram_history,
            int(self._sync_barrier_sequence),
            int(self._hbm_admission_state_generation),
        )

    def _project_hbm_then_lower_tier_restore(
            self, *, candidate: IdleKVEntry, source: KVLocation,
            staging_instance_id: int, target_instance_id: int,
            per_rank_bytes: int, total_bytes: int,
            operation_time_ns: int) -> HBMRestoreProjection:
        """Forecast HBM/DRAM LRU cascades and the full restore without writes."""
        operation_time_ns = int(operation_time_ns)
        shadow = self._fork_hbm_projection_manager(candidate.session_id)
        shadow_candidate = copy.copy(candidate)
        ready_ns = shadow._reserve_hbm(
            shadow_candidate, operation_time_ns)
        if ready_ns is None:
            hbm_victims, cpu_victims = self._projection_lru_victims(
                shadow.events)
            return HBMRestoreProjection(
                hbm_ready_ns=None,
                foreground_arrival_ns=None,
                restore_ready_ns=None,
                hbm_admission_wait_ns=0,
                queue_wait_ns=0,
                service_ns=0,
                transfer_kinds=(
                    ("cpu_to_hbm",)
                    if source == KVLocation.CPU else
                    ("ssd_to_cpu_stage", "cpu_stage_to_hbm")
                ),
                hbm_victim_sessions=hbm_victims,
                cpu_victim_sessions=cpu_victims,
                post_reservation_fingerprint=(
                    shadow._hbm_reservation_fingerprint()),
            )

        reservations: Optional[tuple[TransferReservation, ...]]
        if source == KVLocation.CPU:
            reservation = shadow._reserve_transfer(
                kind="cpu_to_hbm",
                arrival_ns=ready_ns,
                service_ns=shadow._cpu_transfer_ns(
                    int(per_rank_bytes), int(total_bytes)),
                source_instance_id=int(staging_instance_id),
                target_instance_id=int(target_instance_id),
                num_bytes=int(total_bytes),
                background=False,
                session_id=candidate.session_id,
                job_arrival_ns=operation_time_ns,
            )
            reservations = (reservation,)
            transfer_kinds = ("cpu_to_hbm",)
        elif source == KVLocation.SSD:
            staged = shadow._reserve_ssd_restore_stages(
                arrival_ns=ready_ns,
                staging_instance_id=int(staging_instance_id),
                target_instance_id=int(target_instance_id),
                per_rank_bytes=int(per_rank_bytes),
                total_bytes=int(total_bytes),
                session_id=candidate.session_id,
                job_arrival_ns=operation_time_ns,
            )
            reservations = None if staged is None else tuple(staged)
            transfer_kinds = (
                "ssd_to_cpu_stage", "cpu_stage_to_hbm")
        else:
            raise ValueError(
                "Queue-aware restore projection requires a CPU or SSD source, "
                f"got {source.value!r}")

        hbm_victims, cpu_victims = self._projection_lru_victims(
            shadow.events)
        if not reservations:
            return HBMRestoreProjection(
                hbm_ready_ns=int(ready_ns),
                foreground_arrival_ns=None,
                restore_ready_ns=None,
                hbm_admission_wait_ns=0,
                queue_wait_ns=0,
                service_ns=0,
                transfer_kinds=transfer_kinds,
                hbm_victim_sessions=hbm_victims,
                cpu_victim_sessions=cpu_victims,
                post_reservation_fingerprint=(
                    shadow._hbm_reservation_fingerprint()),
            )

        foreground_arrival_ns = int(reservations[0].arrival_ns)
        restore_ready_ns = int(reservations[-1].complete_ns)
        hbm_admission_wait_ns = max(
            0, int(ready_ns) - operation_time_ns)
        transfer_queue_wait_ns = sum(
            int(reservation.queue_wait_ns) for reservation in reservations)
        service_ns = sum(
            int(reservation.service_ns) for reservation in reservations)
        transient_capacity_wait_ns = int(
            reservations[0].transient_dram_capacity_wait_ns)
        if (foreground_arrival_ns - int(ready_ns)
                != transient_capacity_wait_ns):
            raise RuntimeError(
                "Shadow SSD transient-capacity wait does not reconcile: "
                f"foreground_arrival={foreground_arrival_ns}, "
                f"hbm_ready={ready_ns}, "
                f"capacity_wait={transient_capacity_wait_ns}")
        # Queue-policy wait includes both waiting for a full-object DRAM bounce
        # buffer and waiting in the accepted transfer calendars. The explicit
        # transient field below remains an auditable subset, not an additive
        # fifth component.
        queue_wait_ns = (
            transient_capacity_wait_ns + transfer_queue_wait_ns)
        if (restore_ready_ns - operation_time_ns
                != hbm_admission_wait_ns + queue_wait_ns + service_ns):
            raise RuntimeError(
                "Shadow restore projection does not reconcile: "
                f"ready={restore_ready_ns}, operation={operation_time_ns}, "
                f"admission={hbm_admission_wait_ns}, "
                f"queue={queue_wait_ns}, service={service_ns}")
        return HBMRestoreProjection(
            hbm_ready_ns=int(ready_ns),
            foreground_arrival_ns=foreground_arrival_ns,
            restore_ready_ns=restore_ready_ns,
            hbm_admission_wait_ns=hbm_admission_wait_ns,
            queue_wait_ns=queue_wait_ns,
            service_ns=service_ns,
            transfer_kinds=transfer_kinds,
            transient_dram_capacity_wait_ns=transient_capacity_wait_ns,
            hbm_victim_sessions=hbm_victims,
            cpu_victim_sessions=cpu_victims,
            foreground_reservation_signature=(
                self._foreground_reservation_signature(reservations)),
            post_reservation_fingerprint=(
                shadow._hbm_reservation_fingerprint()),
        )

    def _assert_hbm_restore_projection_applied(
            self, *, projection: HBMRestoreProjection,
            candidate_ready_ns: Optional[int], source: KVLocation,
            staging_instance_id: int, target_instance_id: int,
            per_rank_bytes: int, total_bytes: int,
            reservations: Optional[Sequence[TransferReservation]] = None,
            event_start_index: Optional[int] = None) -> None:
        """Prove that live HBM and foreground reservations match the shadow."""
        del source, staging_instance_id, target_instance_id
        del per_rank_bytes, total_bytes
        if candidate_ready_ns != projection.hbm_ready_ns:
            raise RuntimeError(
                "Pure HBM projection changed before restore reservation: "
                f"projected={projection.hbm_ready_ns}, "
                f"reserved={candidate_ready_ns}")
        actual_signature = self._foreground_reservation_signature(
            reservations)
        if actual_signature != projection.foreground_reservation_signature:
            raise RuntimeError(
                "Ordinary foreground restore reservation diverged from its "
                "pure shadow projection: "
                f"projected={projection.foreground_reservation_signature}, "
                f"actual={actual_signature}")
        if event_start_index is not None:
            actual_hbm_victims, actual_cpu_victims = (
                self._projection_lru_victims(
                    self.events[int(event_start_index):]))
            if actual_hbm_victims != projection.hbm_victim_sessions:
                raise RuntimeError(
                    "Ordinary HBM LRU victims diverged from projection: "
                    f"projected={projection.hbm_victim_sessions}, "
                    f"actual={actual_hbm_victims}")
            if actual_cpu_victims != projection.cpu_victim_sessions:
                raise RuntimeError(
                    "Ordinary CPU LRU victims diverged from projection: "
                    f"projected={projection.cpu_victim_sessions}, "
                    f"actual={actual_cpu_victims}")
        expected_state = projection.post_reservation_fingerprint
        if (expected_state is not None
                and self._hbm_reservation_fingerprint() != expected_state):
            raise RuntimeError(
                "Ordinary HBM/DRAM/foreground reservation diverged from its "
                "pure LRU/cascade projection")

    @staticmethod
    def _queue_recompute_chunk_limit(scheduler) -> int:
        chunk_tokens = int(scheduler.max_num_batched_tokens)
        threshold = int(scheduler.long_prefill_token_threshold)
        if threshold > 0:
            chunk_tokens = min(chunk_tokens, threshold)
        if chunk_tokens <= 0:
            raise RuntimeError(
                "Queue-recompute prefix selection requires a positive "
                "prefill chunk limit")
        return chunk_tokens

    def _queue_recompute_incremental_comp_ns(
            self, *, target_instance_id: int, input_tokens: int,
            reusable_tokens: int, selected_tokens: int) -> Optional[int]:
        """Return singleton COMP added by recomputing ``[H, R)``."""
        multiplier = float(
            self.config.queue_recompute_cost_guard_multiplier)
        if multiplier <= 0:
            return None
        provider = self._queue_recompute_latency_providers.get(
            int(target_instance_id))
        if provider is None:
            raise RuntimeError(
                "Missing queue-recompute online latency provider for "
                f"target instance {target_instance_id}")
        scheduler = self._scheduler(int(target_instance_id))
        chunk_tokens = self._queue_recompute_chunk_limit(scheduler)
        selected_comp_ns = provider.singleton_prefill_comp_ns(
            input_tokens=int(input_tokens),
            hit_tokens=int(selected_tokens),
            max_chunk_tokens=chunk_tokens,
        )
        full_hit_comp_ns = provider.singleton_prefill_comp_ns(
            input_tokens=int(input_tokens),
            hit_tokens=int(reusable_tokens),
            max_chunk_tokens=chunk_tokens,
        )
        return max(0, int(selected_comp_ns) - int(full_hit_comp_ns))

    def _queue_recompute_capacity_snapshot(
            self, *, target_instance_id: int,
            pd_decode_instance_id: Optional[int], input_tokens: int,
            prefix_tokens: int, operation_time_ns: int,
            ) -> QueueRecomputeCapacitySnapshot:
        """Snapshot unreserved P/D HBM through the configured next chunk.

        This helper never reserves either endpoint.  The strict P/D router
        owns the later atomic chunk claim, so a feasible snapshot is evidence
        that a candidate was not knowingly impossible, not a promise that
        unrelated work cannot consume the bytes before DMA completion.
        """
        target_instance_id = int(target_instance_id)
        prefix_tokens = int(prefix_tokens)
        scheduler = self._scheduler(target_instance_id)
        chunk_limit = self._queue_recompute_chunk_limit(scheduler)
        configured_chunks = float(
            self.config.queue_recompute_prefill_headroom_chunks)
        horizon_tokens = int(math.ceil(chunk_limit * configured_chunks))
        next_chunk_tokens = min(
            max(0, int(input_tokens) - prefix_tokens), horizon_tokens)
        block_size = int(self.config.block_size)
        prefix_block_tokens = (
            (prefix_tokens + block_size - 1) // block_size * block_size
            if prefix_tokens else 0
        )
        through_tokens = prefix_tokens + next_chunk_tokens
        through_block_tokens = (
            (through_tokens + block_size - 1) // block_size * block_size
            if through_tokens else 0
        )
        prefix_per_rank_bytes = int(
            scheduler.memory.get_kv(prefix_block_tokens))
        through_per_rank_bytes = int(
            scheduler.memory.get_kv(through_block_tokens))
        growth_per_rank_bytes = max(
            0, through_per_rank_bytes - prefix_per_rank_bytes)
        prefill_unreserved = int(
            self.hbm_unreserved_per_rank_bytes(target_instance_id))

        decode_instance_id = None
        decode_unreserved = None
        decode_required = 0
        if getattr(scheduler, "pd_type", None) == "prefill":
            if pd_decode_instance_id is None:
                raise RuntimeError(
                    "Queue-recompute partial-prefix selection on a P/D "
                    "prefill instance requires its explicit fixed decode "
                    "instance id")
            decode_instance_id = int(pd_decode_instance_id)
            decode = self._scheduler(decode_instance_id)
            if getattr(decode, "pd_type", None) != "decode":
                raise RuntimeError(
                    "Queue-recompute P/D headroom target is not a decode "
                    f"instance: {decode_instance_id}")
            if self._node_id(decode) != self._node_id(scheduler):
                raise RuntimeError(
                    "Queue-recompute P/D headroom requires a same-node "
                    "fixed pair")
            layout_fields = (
                "model", "tp_size", "pp_size", "block_size", "fp",
                "kv_cache_dtype",
            )
            mismatches = [
                field for field in layout_fields
                if getattr(scheduler, field) != getattr(decode, field)
            ]
            if mismatches:
                raise RuntimeError(
                    "Queue-recompute P/D headroom requires identical KV "
                    f"layout; mismatched fields={mismatches}")
            decode_unreserved = int(
                self.hbm_unreserved_per_rank_bytes(decode_instance_id))
            decode_required = int(
                decode.memory.get_kv(through_block_tokens))
        elif pd_decode_instance_id is not None:
            raise RuntimeError(
                "pd_decode_instance_id is valid only for a prefill target")

        return QueueRecomputeCapacitySnapshot(
            time_ns=int(operation_time_ns),
            prefix_tokens=prefix_tokens,
            prefix_block_tokens=prefix_block_tokens,
            next_chunk_tokens=next_chunk_tokens,
            through_next_chunk_block_tokens=through_block_tokens,
            prefill_instance_id=target_instance_id,
            prefill_unreserved_per_rank_bytes=prefill_unreserved,
            prefill_prefix_per_rank_bytes=prefix_per_rank_bytes,
            prefill_growth_headroom_per_rank_bytes=growth_per_rank_bytes,
            prefill_required_through_chunk_per_rank_bytes=(
                through_per_rank_bytes),
            decode_instance_id=decode_instance_id,
            decode_unreserved_per_rank_bytes=decode_unreserved,
            decode_required_through_chunk_per_rank_bytes=decode_required,
        )

    def _queue_recompute_partial_candidates(
            self, *, reusable_tokens: int, target_instance_id: int,
            pd_decode_instance_id: Optional[int], input_tokens: int,
            operation_time_ns: int,
            ) -> tuple[
                tuple[int, ...],
                Dict[int, QueueRecomputeCapacitySnapshot],
            ]:
        """Return deterministic feasible block-prefix candidates.

        Capacity feasibility is monotone in H for a fixed next-chunk horizon,
        so a binary search finds the ceiling without running a shadow for
        every cache block.  Geometric interior points still expose cases in
        which a shorter gang reservation fits an earlier calendar hole.
        """
        block_size = int(self.config.block_size)
        max_partial_blocks = max(
            0, (int(reusable_tokens) - 1) // block_size)
        snapshots: Dict[int, QueueRecomputeCapacitySnapshot] = {}

        def snapshot(blocks: int) -> QueueRecomputeCapacitySnapshot:
            tokens = int(blocks) * block_size
            value = snapshots.get(tokens)
            if value is None:
                value = self._queue_recompute_capacity_snapshot(
                    target_instance_id=target_instance_id,
                    pd_decode_instance_id=pd_decode_instance_id,
                    input_tokens=input_tokens,
                    prefix_tokens=tokens,
                    operation_time_ns=operation_time_ns,
                )
                snapshots[tokens] = value
            return value

        low = 0
        high = max_partial_blocks
        while low < high:
            middle = (low + high + 1) // 2
            if snapshot(middle).feasible:
                low = middle
            else:
                high = middle - 1
        max_feasible_blocks = low
        if max_feasible_blocks <= 0:
            return (), snapshots

        candidate_blocks = {1, max_feasible_blocks}
        if max_feasible_blocks > 1:
            candidate_blocks.add(max_feasible_blocks - 1)
        cursor = max_feasible_blocks
        while cursor > 1:
            cursor //= 2
            candidate_blocks.add(max(1, cursor))
        candidates = tuple(sorted(
            (
                blocks * block_size
                for blocks in candidate_blocks
                if snapshot(blocks).feasible
            ),
            reverse=True,
        ))
        return candidates, snapshots

    def _evaluate_queue_recompute(
            self, *, session_id: str, source: KVLocation,
            projection: Optional[HBMRestoreProjection],
            staging_instance_id: int,
            target_instance_id: int,
            pd_decode_instance_id: Optional[int], per_rank_bytes: int,
            total_bytes: int, physical_entry_bytes: int,
            declared_reuse_tokens: int, reusable_tokens: int,
            policy_avoidable_tokens: int, input_tokens: int,
            operation_time_ns: int,
            ) -> QueueRecomputeSelection:
        """Choose a full, block-prefix, or zero restore before reservation.

        A modified decision is possible only after the full restore crosses
        the configured strict severe-queue threshold. Candidate costs are the
        causal prefix-restore projection plus singleton COMP for ``[H, R)``.
        P/D headroom is a timestamped slack snapshot and is never claimed by
        this method. Every nonzero decision is immutable across retries.
        """
        invocation = (
            source.value, int(staging_instance_id), int(target_instance_id),
            (
                None if pd_decode_instance_id is None
                else int(pd_decode_instance_id)
            ),
            int(per_rank_bytes), int(total_bytes),
            int(physical_entry_bytes), int(declared_reuse_tokens),
            int(reusable_tokens), int(policy_avoidable_tokens),
            int(input_tokens),
        )
        if not self.config.queue_recompute_enabled:
            return QueueRecomputeSelection(
                invocation=invocation,
                reusable_tokens=int(reusable_tokens),
                selected_tokens=int(reusable_tokens),
                selected_block_tokens=(
                    (int(reusable_tokens) + self.config.block_size - 1)
                    // self.config.block_size * self.config.block_size),
                selected_per_rank_bytes=int(per_rank_bytes),
                selected_total_bytes=int(total_bytes),
                full_total_bytes=int(total_bytes),
                full_projection=projection,
                selected_projection=projection,
                capacity_snapshot=None,
                estimated_suffix_recompute_ns=0,
                estimated_full_recompute_ns=0,
                predicted_resume_path_ns=None,
                full_predicted_resume_path_ns=None,
                selection_reason="policy_disabled",
            )
        committed = self._queue_recompute_restore_commitments.get(
            str(session_id))
        if committed is not None:
            if committed.invocation != invocation:
                raise RuntimeError(
                    "Queue-recompute retry changed its immutable invocation: "
                    f"session={session_id}, expected={committed.invocation}, "
                    f"observed={invocation}")
            self.events.append({
                "time_ns": int(operation_time_ns),
                "session_id": str(session_id),
                "event": "queue_recompute_restore_commitment_reused",
                "decision": (
                    "partial_restore_suffix_recompute"
                    if committed.partial else "restore"),
                "reusable_tokens_R": committed.reusable_tokens,
                "selected_prefix_tokens_H": committed.selected_tokens,
                "dropped_suffix_tokens": (
                    committed.dropped_suffix_tokens),
            })
            return replace(
                committed,
                full_projection=None,
                selected_projection=None,
            )

        transfer_kinds = (
            ("cpu_to_hbm",)
            if source == KVLocation.CPU else
            ("ssd_to_cpu_stage", "cpu_stage_to_hbm")
        )
        projection_available = bool(
            projection is not None and projection.available)
        if projection_available:
            projection_arrival_ns = int(projection.hbm_ready_ns)
            hbm_admission_wait_ns = int(
                projection.hbm_admission_wait_ns)
            queue_wait_ns = int(projection.queue_wait_ns)
            service_ns = int(projection.service_ns)
            transient_dram_capacity_wait_ns = int(
                projection.transient_dram_capacity_wait_ns)
            transfer_kinds = projection.transfer_kinds
            includes_new_lru_work = projection.includes_new_lru_work
            hbm_victim_sessions = projection.hbm_victim_sessions
            cpu_victim_sessions = projection.cpu_victim_sessions
        else:
            # The object cannot be admitted even after a pure LRU/cascade
            # attempt. Fail closed to restore and freeze that choice; ordinary
            # reservation retains the baseline's deterministic failure path.
            projection_arrival_ns = None
            hbm_admission_wait_ns = 0
            queue_wait_ns = 0
            service_ns = 0
            transient_dram_capacity_wait_ns = 0
            includes_new_lru_work = bool(
                projection is not None
                and projection.includes_new_lru_work)
            hbm_victim_sessions = (
                () if projection is None else
                projection.hbm_victim_sessions)
            cpu_victim_sessions = (
                () if projection is None else
                projection.cpu_victim_sessions)
        total_wait_ns = hbm_admission_wait_ns + queue_wait_ns
        ratio = float(self.config.queue_recompute_wait_service_ratio)
        ratio_threshold_ns = int(math.ceil(ratio * service_ns))
        minimum_threshold_ns = int(
            self.config.queue_recompute_min_wait_ns)
        threshold_ns = max(ratio_threshold_ns, minimum_threshold_ns)
        severe_gate_pass = (
            projection_available and total_wait_ns > threshold_ns)
        cost_multiplier = float(
            self.config.queue_recompute_cost_guard_multiplier)
        projected_restore_ns = int(total_wait_ns) + int(service_ns)
        full_predicted_ns = (
            projected_restore_ns if projection_available else None)

        selected_tokens = int(reusable_tokens)
        selected_block_tokens = (
            (selected_tokens + self.config.block_size - 1)
            // self.config.block_size * self.config.block_size
        )
        selected_per_rank_bytes = int(per_rank_bytes)
        selected_total_bytes = int(total_bytes)
        selected_projection = projection
        selected_snapshot = None
        selected_estimated_recompute_ns: Optional[int] = 0
        selected_predicted_ns = full_predicted_ns
        selection_reason = (
            "full_projection_unavailable_fail_closed"
            if not projection_available else
            "full_restore_below_severe_threshold"
            if not severe_gate_pass else
            "full_restore_lowest_predicted_path"
        )
        candidate_tokens: tuple[int, ...] = ()
        zero_recompute_ns: Optional[int] = None

        if severe_gate_pass:
            partial_tokens, snapshots = (
                self._queue_recompute_partial_candidates(
                    reusable_tokens=reusable_tokens,
                    target_instance_id=target_instance_id,
                    pd_decode_instance_id=pd_decode_instance_id,
                    input_tokens=input_tokens,
                    operation_time_ns=operation_time_ns,
                )
            )
            candidate_tokens = (
                int(reusable_tokens), *partial_tokens, 0)
            candidates = [(
                int(projected_restore_ns), -int(reusable_tokens),
                int(reusable_tokens), selected_block_tokens,
                int(per_rank_bytes), int(total_bytes), projection, None, 0,
            )]
            scheduler = self._scheduler(int(target_instance_id))
            for prefix_tokens in partial_tokens:
                prefix_block_tokens = (
                    (int(prefix_tokens) + self.config.block_size - 1)
                    // self.config.block_size * self.config.block_size
                )
                prefix_per_rank_bytes = int(
                    scheduler.memory.get_kv(prefix_block_tokens))
                prefix_total_bytes = (
                    prefix_per_rank_bytes * int(scheduler.num_npus))
                prefix_candidate = IdleKVEntry(
                    session_id=str(session_id),
                    instance_id=int(target_instance_id),
                    tokens=int(prefix_tokens),
                    block_tokens=prefix_block_tokens,
                    per_rank_bytes=prefix_per_rank_bytes,
                    total_bytes=prefix_total_bytes,
                    location=KVLocation.HBM,
                    tier_since_ns=int(operation_time_ns),
                    last_access_ns=int(operation_time_ns),
                )
                prefix_projection = (
                    self._project_hbm_then_lower_tier_restore(
                        candidate=prefix_candidate,
                        source=source,
                        staging_instance_id=staging_instance_id,
                        target_instance_id=target_instance_id,
                        per_rank_bytes=prefix_per_rank_bytes,
                        total_bytes=prefix_total_bytes,
                        operation_time_ns=operation_time_ns,
                    )
                )
                snapshot = snapshots[int(prefix_tokens)]
                if (not snapshot.feasible
                        or not prefix_projection.available
                        or prefix_projection.hbm_admission_wait_ns != 0
                        or prefix_projection.hbm_victim_sessions):
                    continue
                suffix_recompute_ns = (
                    self._queue_recompute_incremental_comp_ns(
                        target_instance_id=target_instance_id,
                        input_tokens=input_tokens,
                        reusable_tokens=reusable_tokens,
                        selected_tokens=prefix_tokens,
                    )
                )
                recompute_penalty_ns = (
                    0 if suffix_recompute_ns is None else
                    int(math.ceil(
                        cost_multiplier * suffix_recompute_ns))
                )
                prefix_restore_ns = (
                    prefix_projection.total_wait_ns
                    + prefix_projection.service_ns)
                predicted_ns = prefix_restore_ns + recompute_penalty_ns
                candidates.append((
                    int(predicted_ns), -int(prefix_tokens),
                    int(prefix_tokens), prefix_block_tokens,
                    prefix_per_rank_bytes, prefix_total_bytes,
                    prefix_projection, snapshot, suffix_recompute_ns,
                ))

            zero_recompute_ns = self._queue_recompute_incremental_comp_ns(
                target_instance_id=target_instance_id,
                input_tokens=input_tokens,
                reusable_tokens=reusable_tokens,
                selected_tokens=0,
            )
            zero_predicted_ns = (
                0 if zero_recompute_ns is None else
                int(math.ceil(cost_multiplier * zero_recompute_ns))
            )
            candidates.append((
                zero_predicted_ns, 0, 0, 0, 0, 0,
                None, None, zero_recompute_ns,
            ))
            best = min(candidates, key=lambda row: (row[0], row[1]))
            # Strict improvement keeps threshold equality on the baseline
            # full restore, just like the original whole-entry policy.
            if best[2] < int(reusable_tokens) and best[0] < projected_restore_ns:
                (
                    selected_predicted_ns, _, selected_tokens,
                    selected_block_tokens, selected_per_rank_bytes,
                    selected_total_bytes, selected_projection,
                    selected_snapshot, selected_estimated_recompute_ns,
                ) = best
                selection_reason = (
                    "partial_prefix_lowest_predicted_path"
                    if selected_tokens else
                    "zero_restore_lowest_predicted_path")

        cost_gate_pass = selected_tokens < int(reusable_tokens)
        cost_threshold_ns = (
            None if selected_estimated_recompute_ns is None else
            int(math.ceil(
                cost_multiplier * selected_estimated_recompute_ns))
        )
        selection = QueueRecomputeSelection(
            invocation=invocation,
            reusable_tokens=int(reusable_tokens),
            selected_tokens=int(selected_tokens),
            selected_block_tokens=int(selected_block_tokens),
            selected_per_rank_bytes=int(selected_per_rank_bytes),
            selected_total_bytes=int(selected_total_bytes),
            full_total_bytes=int(total_bytes),
            full_projection=projection,
            selected_projection=selected_projection,
            capacity_snapshot=selected_snapshot,
            estimated_suffix_recompute_ns=(
                selected_estimated_recompute_ns),
            estimated_full_recompute_ns=zero_recompute_ns,
            predicted_resume_path_ns=(
                None if selected_predicted_ns is None else
                int(selected_predicted_ns)),
            full_predicted_resume_path_ns=full_predicted_ns,
            selection_reason=selection_reason,
            candidate_tokens=tuple(int(value) for value in candidate_tokens),
        )
        self.metrics.queue_recompute_evaluation_attempts += 1
        if severe_gate_pass:
            self.metrics.queue_recompute_severe_gate_passes += 1
        if cost_gate_pass:
            self.metrics.queue_recompute_cost_gate_passes += 1
        snapshot_payload = (
            None if selection.capacity_snapshot is None else
            {
                **asdict(selection.capacity_snapshot),
                "feasible": selection.capacity_snapshot.feasible,
                "semantics": "causal_snapshot_not_reservation",
            }
        )
        selected_projection_available = bool(
            selection.selected_projection is not None
            and selection.selected_projection.available)
        self.events.append({
            "time_ns": int(operation_time_ns),
            "session_id": str(session_id),
            "event": "queue_recompute_evaluate",
            "source": source.value,
            "transfer_kinds": list(transfer_kinds),
            "bytes": int(total_bytes),
            "reusable_tokens_R": int(reusable_tokens),
            "selected_prefix_tokens_H": int(selection.selected_tokens),
            "selected_prefix_block_tokens": int(
                selection.selected_block_tokens),
            "dropped_suffix_tokens": selection.dropped_suffix_tokens,
            "selected_restore_bytes": selection.selected_total_bytes,
            "dropped_suffix_bytes": selection.dropped_suffix_bytes,
            "avoided_restore_bytes": selection.dropped_suffix_bytes,
            "physical_entry_dropped_bytes": (
                int(physical_entry_bytes) if selection.zero_restore else 0),
            "projection_arrival_ns": (
                None if projection_arrival_ns is None else
                int(projection_arrival_ns)),
            "projection_available": projection_available,
            "projection_available_without_new_lru_work": (
                projection_available and not includes_new_lru_work),
            "projection_includes_collateral_lru_work": (
                includes_new_lru_work),
            "projected_hbm_victim_sessions": list(hbm_victim_sessions),
            "projected_cpu_victim_sessions": list(cpu_victim_sessions),
            "projection_precedes_destination_hbm_reservation": True,
            "projected_hbm_admission_wait_ns": int(
                hbm_admission_wait_ns),
            "projected_transient_dram_capacity_wait_ns": int(
                transient_dram_capacity_wait_ns),
            "projected_queue_wait_ns": int(queue_wait_ns),
            "projected_total_wait_ns": int(total_wait_ns),
            "projected_service_ns": int(service_ns),
            "projected_restore_ns": projected_restore_ns,
            "estimated_incremental_recompute_comp_ns": (
                selection.estimated_full_recompute_ns),
            "estimated_suffix_recompute_comp_ns": (
                selection.estimated_suffix_recompute_ns),
            "selected_predicted_resume_path_ns": (
                selection.predicted_resume_path_ns),
            "full_predicted_resume_path_ns": (
                selection.full_predicted_resume_path_ns),
            "predicted_path_savings_ns": (
                None
                if (selection.predicted_resume_path_ns is None
                    or selection.full_predicted_resume_path_ns is None)
                else selection.full_predicted_resume_path_ns
                - selection.predicted_resume_path_ns),
            "candidate_prefix_tokens": list(selection.candidate_tokens),
            "full_projection_status": (
                "unavailable" if not projection_available else
                "available_with_collateral_lru"
                if includes_new_lru_work else
                "available_without_collateral_lru"),
            "prefix_projection_available": selected_projection_available,
            "prefix_projected_hbm_admission_wait_ns": (
                0 if not selected_projection_available else
                selection.selected_projection.hbm_admission_wait_ns),
            "prefix_projected_transient_dram_capacity_wait_ns": (
                0 if not selected_projection_available else
                selection.selected_projection
                .transient_dram_capacity_wait_ns),
            "prefix_projected_queue_wait_ns": (
                0 if not selected_projection_available else
                selection.selected_projection.queue_wait_ns),
            "prefix_projected_service_ns": (
                0 if not selected_projection_available else
                selection.selected_projection.service_ns),
            "prefix_projected_restore_ns": (
                0 if not selected_projection_available else
                selection.selected_projection.total_wait_ns
                + selection.selected_projection.service_ns),
            "capacity_headroom_snapshot": snapshot_payload,
            "capacity_headroom_snapshot_only": True,
            "capacity_headroom_claimed_by_policy": False,
            "pd_first_chunk_immediate_admission_guaranteed": False,
            "configured_wait_service_ratio": ratio,
            "configured_min_wait_ns": minimum_threshold_ns,
            "configured_cost_guard_multiplier": cost_multiplier,
            "ratio_threshold_ns": ratio_threshold_ns,
            "threshold_ns": threshold_ns,
            "cost_threshold_ns": cost_threshold_ns,
            "severe_gate_pass": severe_gate_pass,
            "cost_gate_pass": cost_gate_pass,
            "selection_reason": selection.selection_reason,
            "decision": (
                "drop_recompute" if selection.zero_restore else
                "partial_restore_suffix_recompute"
                if selection.partial else "restore"),
        })
        if not selection.modified:
            self.metrics.queue_recompute_full_restore_decisions += 1
            self._queue_recompute_restore_commitments[
                str(session_id)] = selection
            return selection

        if selection.partial:
            self.metrics.queue_recompute_partial_restore_decisions += 1
            if source == KVLocation.CPU:
                self.metrics.queue_recompute_partial_cpu_decisions += 1
            else:
                self.metrics.queue_recompute_partial_ssd_decisions += 1
        else:
            self.metrics.queue_recompute_zero_restore_decisions += 1
            self.metrics.queue_recompute_drop_decisions += 1
            if source == KVLocation.CPU:
                self.metrics.queue_recompute_cpu_drop_decisions += 1
            else:
                self.metrics.queue_recompute_ssd_drop_decisions += 1
            self.metrics.queue_recompute_physical_entry_dropped_bytes += int(
                physical_entry_bytes)
        self.metrics.queue_recompute_dropped_bytes += (
            selection.dropped_suffix_bytes)
        self.metrics.queue_recompute_avoided_restore_bytes += (
            selection.dropped_suffix_bytes)
        self.metrics.queue_recompute_selected_restore_tokens += int(
            selection.selected_tokens)
        self.metrics.queue_recompute_dropped_suffix_tokens += (
            selection.dropped_suffix_tokens)
        self.metrics.queue_recompute_selected_restore_bytes += int(
            selection.selected_total_bytes)
        self.metrics.queue_recompute_dropped_suffix_bytes += (
            selection.dropped_suffix_bytes)
        self.metrics.queue_recompute_projected_queue_wait_ns += int(
            queue_wait_ns)
        self.metrics.queue_recompute_projected_hbm_admission_wait_ns += int(
            hbm_admission_wait_ns)
        self.metrics.queue_recompute_projected_transient_dram_capacity_wait_ns += (
            int(transient_dram_capacity_wait_ns))
        self.metrics.queue_recompute_projected_service_ns += int(service_ns)
        if selection.partial and selection.selected_projection is not None:
            self.metrics.queue_recompute_prefix_projected_queue_wait_ns += int(
                selection.selected_projection.queue_wait_ns)
            self.metrics.queue_recompute_prefix_projected_hbm_admission_wait_ns += int(
                selection.selected_projection.hbm_admission_wait_ns)
            self.metrics.queue_recompute_prefix_projected_transient_dram_capacity_wait_ns += int(
                selection.selected_projection
                .transient_dram_capacity_wait_ns)
            self.metrics.queue_recompute_prefix_projected_service_ns += int(
                selection.selected_projection.service_ns)
        if selection.estimated_suffix_recompute_ns is not None:
            self.metrics.queue_recompute_estimated_recompute_ns += int(
                selection.estimated_suffix_recompute_ns)
        self.events.append({
            "time_ns": int(operation_time_ns),
            "session_id": str(session_id),
            "event": (
                "queue_recompute_partial"
                if selection.partial else "queue_recompute_drop"),
            "source": source.value,
            "transfer_kinds": list(transfer_kinds),
            "bytes": int(total_bytes),
            "reusable_tokens_R": int(reusable_tokens),
            "selected_prefix_tokens_H": int(selection.selected_tokens),
            "selected_prefix_block_tokens": int(
                selection.selected_block_tokens),
            "dropped_suffix_tokens": selection.dropped_suffix_tokens,
            "selected_restore_bytes": selection.selected_total_bytes,
            "dropped_suffix_bytes": selection.dropped_suffix_bytes,
            "avoided_restore_bytes": selection.dropped_suffix_bytes,
            "physical_entry_dropped_bytes": (
                int(physical_entry_bytes) if selection.zero_restore else 0),
            "physical_source_bytes_pinned_until_dma_complete": (
                int(physical_entry_bytes) if selection.partial else 0),
            "declared_reuse_tokens": int(declared_reuse_tokens),
            "reusable_tokens": int(reusable_tokens),
            "policy_avoidable_tokens": int(policy_avoidable_tokens),
            "projection_arrival_ns": int(projection_arrival_ns),
            "projection_available": True,
            "projection_available_without_new_lru_work": (
                not includes_new_lru_work),
            "projection_includes_collateral_lru_work": (
                includes_new_lru_work),
            "projected_hbm_victim_sessions": list(hbm_victim_sessions),
            "projected_cpu_victim_sessions": list(cpu_victim_sessions),
            "projection_precedes_destination_hbm_reservation": True,
            "projected_hbm_admission_wait_ns": int(
                hbm_admission_wait_ns),
            "projected_transient_dram_capacity_wait_ns": int(
                transient_dram_capacity_wait_ns),
            "projected_queue_wait_ns": int(queue_wait_ns),
            "projected_total_wait_ns": int(total_wait_ns),
            "projected_service_ns": int(service_ns),
            "projected_restore_ns": projected_restore_ns,
            "estimated_incremental_recompute_comp_ns": (
                selection.estimated_full_recompute_ns),
            "estimated_suffix_recompute_comp_ns": (
                selection.estimated_suffix_recompute_ns),
            "selected_predicted_resume_path_ns": (
                selection.predicted_resume_path_ns),
            "full_predicted_resume_path_ns": (
                selection.full_predicted_resume_path_ns),
            "candidate_prefix_tokens": list(selection.candidate_tokens),
            "full_projection_status": (
                "available_with_collateral_lru"
                if includes_new_lru_work else
                "available_without_collateral_lru"),
            "prefix_projection_available": selected_projection_available,
            "prefix_projected_hbm_admission_wait_ns": (
                0 if not selected_projection_available else
                selection.selected_projection.hbm_admission_wait_ns),
            "prefix_projected_transient_dram_capacity_wait_ns": (
                0 if not selected_projection_available else
                selection.selected_projection
                .transient_dram_capacity_wait_ns),
            "prefix_projected_queue_wait_ns": (
                0 if not selected_projection_available else
                selection.selected_projection.queue_wait_ns),
            "prefix_projected_service_ns": (
                0 if not selected_projection_available else
                selection.selected_projection.service_ns),
            "capacity_headroom_snapshot": snapshot_payload,
            "capacity_headroom_snapshot_only": True,
            "capacity_headroom_claimed_by_policy": False,
            "pd_first_chunk_immediate_admission_guaranteed": False,
            "configured_wait_service_ratio": ratio,
            "configured_min_wait_ns": minimum_threshold_ns,
            "configured_cost_guard_multiplier": cost_multiplier,
            "ratio_threshold_ns": ratio_threshold_ns,
            "threshold_ns": threshold_ns,
            "cost_threshold_ns": cost_threshold_ns,
            "severe_gate_pass": severe_gate_pass,
            "cost_gate_pass": cost_gate_pass,
            "object_scope": "kv_cache_entry",
            "selection_scope": (
                "contiguous_block_aligned_prefix"
                if selection.partial else "whole_reusable_entry"),
            "selection_reason": selection.selection_reason,
            "source_pin_scope": (
                "full_physical_source_until_prefix_dma_complete"
                if selection.partial else "not_applicable"),
            "recompute_scope": (
                "contiguous_suffix_H_to_R"
                if selection.partial else "whole_reusable_prefix"),
            "logical_session_effect": "none",
        })
        if selection.partial:
            self._queue_recompute_restore_commitments[
                str(session_id)] = selection
        return selection

    def _transient_cpu_lru_candidate(
            self, scheduler, exclude_session: Optional[str]):
        if not self.config.tiered_family:
            return None
        node_id = self._node_id(scheduler)
        candidates = sorted(
            (
                entry for entry in self.entries.values()
                if entry.location == KVLocation.CPU
                and self._node_id(
                    self._scheduler(entry.instance_id)) == node_id
                and entry.migration_kind is None
                and entry.session_id != exclude_session
                and entry.session_id not in self._capacity_pinned_sessions()
            ),
            key=lambda entry: (entry.last_access_ns, entry.session_id),
        )
        return candidates[0] if candidates else None

    def _record_transient_dram_reservation(
            self, *, scheduler, session_id: Optional[str],
            start_ns: int, complete_ns: int, num_bytes: int,
            reservation_sequence: int, peak_committed_bytes: int,
            capacity_wait_ns: int, pressure_stall_ns: int) -> None:
        node_id = self._node_id(scheduler)
        reservation = TransientDRAMReservation(
            node_id=node_id,
            session_id=session_id,
            start_ns=int(start_ns),
            complete_ns=int(complete_ns),
            bytes=int(num_bytes),
            reservation_sequence=int(reservation_sequence),
            peak_node_committed_bytes=int(peak_committed_bytes),
        )
        calendar = self._transient_dram_reservations.setdefault(node_id, [])
        index = bisect.bisect_left(
            [item.start_ns for item in calendar], reservation.start_ns)
        calendar.insert(index, reservation)
        self._transient_dram_history.setdefault(node_id, []).append(
            reservation)
        occupancy_events = []
        for item in calendar:
            occupancy_events.append((item.start_ns, 1, item.bytes))
            occupancy_events.append((item.complete_ns, 0, -item.bytes))
        occupancy = 0
        peak = 0
        for _, _, delta in sorted(occupancy_events):
            occupancy += delta
            peak = max(peak, occupancy)
        capacity = self._cpu_capacity_bytes(scheduler)
        if peak > capacity or peak_committed_bytes > capacity:
            raise RuntimeError(
                "Transient DRAM reservation exceeded node capacity: "
                f"node={node_id}, transient_peak={peak}, "
                f"committed_peak={peak_committed_bytes}, capacity={capacity}")
        self.metrics.transient_dram_reservations += 1
        self.metrics.transient_dram_reserved_bytes += int(num_bytes)
        self.metrics.transient_dram_byte_ns += (
            int(num_bytes) * (int(complete_ns) - int(start_ns)))
        self.metrics.transient_dram_capacity_wait_ns += int(capacity_wait_ns)
        self.metrics.transient_dram_pressure_stall_ns += int(
            pressure_stall_ns)
        self.metrics.peak_transient_dram_bytes = max(
            self.metrics.peak_transient_dram_bytes, peak)
        self.metrics.peak_cpu_committed_plus_transient_bytes = max(
            self.metrics.peak_cpu_committed_plus_transient_bytes,
            int(peak_committed_bytes),
        )
        self.events.append({
            "time_ns": int(start_ns),
            "session_id": session_id,
            "event": "transient_dram_reserve",
            "node_id": node_id,
            "start_ns": int(start_ns),
            "complete_ns": int(complete_ns),
            "bytes": int(num_bytes),
            "capacity_bytes": capacity,
            "capacity_wait_ns": int(capacity_wait_ns),
            "pressure_stall_ns": int(pressure_stall_ns),
            "peak_node_committed_bytes": int(peak_committed_bytes),
        })

    def _reserve_ssd_restore_stages(
            self, *, arrival_ns: int, staging_instance_id: int,
            target_instance_id: int, per_rank_bytes: int,
            total_bytes: int, session_id: Optional[str],
            register_sync_barrier: bool = True,
            job_arrival_ns: Optional[int] = None,
            ) -> Optional[
                tuple[TransferReservation, TransferReservation]]:
        """Reserve SSD->host and host->HBM as serial dependency stages.

        The first stage occupies SSD read and host DRAM but no GPU PCIe copy
        engine. The second begins only after the transient host buffer is
        complete and occupies host DRAM plus every target-rank PCIe engine,
        but no SSD queue. A full-object transient DRAM allocation is held from
        media start through H2D completion and shares the node capacity with
        persistent CPU cache. In the legacy synchronous sensitivity mode, the
        two physical reservations feed one media-arrival-to-HBM-completion
        engine barrier so splitting resources does not weaken that mode's
        original semantics.
        """
        arrival_ns = int(arrival_ns)
        total_bytes = int(total_bytes)
        staging_instance_id = int(staging_instance_id)
        target_instance_id = int(target_instance_id)
        logical_arrival_ns = int(
            arrival_ns if job_arrival_ns is None else job_arrival_ns)
        scheduler = self._scheduler(staging_instance_id)
        if total_bytes > self._cpu_capacity_bytes(scheduler):
            self.metrics.transient_dram_capacity_oversize += 1
            self.events.append({
                "time_ns": logical_arrival_ns,
                "session_id": session_id,
                "event": "transient_dram_capacity_oversize",
                "bytes": total_bytes,
                "capacity_bytes": self._cpu_capacity_bytes(scheduler),
            })
            return None

        candidate_arrival_ns = arrival_ns
        capacity_pressure = False
        temporarily_pinned = (
            session_id is not None
            and session_id not in self._pending_restore_sessions)
        if session_id is not None:
            self._set_restore_capacity_pin(session_id, True)
        try:
            while True:
                (
                    media_start_ns, media_complete_ns, _,
                    _, h2d_complete_ns, _,
                ) = self._plan_ssd_restore_stages(
                    arrival_ns=candidate_arrival_ns,
                    staging_instance_id=staging_instance_id,
                    target_instance_id=target_instance_id,
                    per_rank_bytes=per_rank_bytes,
                    total_bytes=total_bytes,
                )
                violation_ns, peak_committed_bytes, capacity_events = (
                    self._transient_dram_window_capacity(
                        scheduler,
                        media_start_ns,
                        h2d_complete_ns,
                        total_bytes,
                        self._logical_frontier_ns,
                    )
                )
                if violation_ns is None:
                    break
                capacity_pressure = True
                victim = self._transient_cpu_lru_candidate(
                    scheduler, session_id)
                if victim is not None:
                    self._schedule_entry_migration(
                        victim,
                        "cpu_to_ssd",
                        logical_arrival_ns,
                        reason="transient_dram_capacity",
                        job_arrival_ns=logical_arrival_ns,
                    )
                    if victim.migration_complete_ns is not None:
                        self.events.append({
                            "time_ns": logical_arrival_ns,
                            "session_id": victim.session_id,
                            "event": (
                                "transient_dram_cpu_lru_eviction_scheduled"),
                            "complete_ns": victim.migration_complete_ns,
                            "bytes": victim.total_bytes,
                            "restore_session_id": session_id,
                        })
                    continue
                relief_times = [
                    event_ns
                    for event_ns, delta in capacity_events
                    if event_ns > violation_ns and delta > 0
                ]
                if not relief_times:
                    self.metrics.transient_dram_capacity_deferrals += 1
                    self.events.append({
                        "time_ns": logical_arrival_ns,
                        "session_id": session_id,
                        "event": "transient_dram_capacity_deferred",
                        "bytes": total_bytes,
                        "capacity_bytes": self._cpu_capacity_bytes(scheduler),
                        "violation_ns": violation_ns,
                    })
                    return None
                next_arrival_ns = min(relief_times)
                if next_arrival_ns <= candidate_arrival_ns:
                    raise RuntimeError(
                        "Transient DRAM capacity search did not advance: "
                        f"candidate={candidate_arrival_ns}, "
                        f"relief={next_arrival_ns}")
                candidate_arrival_ns = next_arrival_ns
        finally:
            if temporarily_pinned and session_id is not None:
                self._set_restore_capacity_pin(session_id, False)

        planned = self._plan_ssd_restore_stages(
            arrival_ns=candidate_arrival_ns,
            staging_instance_id=staging_instance_id,
            target_instance_id=target_instance_id,
            per_rank_bytes=per_rank_bytes,
            total_bytes=total_bytes,
        )
        media = self._reserve_transfer(
            kind="ssd_to_cpu_stage",
            arrival_ns=candidate_arrival_ns,
            service_ns=planned[2],
            source_instance_id=staging_instance_id,
            target_instance_id=None,
            num_bytes=total_bytes,
            background=False,
            session_id=session_id,
            register_sync_barrier=False,
            job_arrival_ns=logical_arrival_ns,
        )
        h2d = self._reserve_transfer(
            kind="cpu_stage_to_hbm",
            arrival_ns=media.complete_ns,
            service_ns=planned[5],
            source_instance_id=staging_instance_id,
            target_instance_id=target_instance_id,
            num_bytes=total_bytes,
            background=False,
            session_id=session_id,
            register_sync_barrier=False,
            job_arrival_ns=logical_arrival_ns,
            parent_reservation=media,
        )
        if ((media.start_ns, media.complete_ns, h2d.start_ns,
             h2d.complete_ns)
                != (planned[0], planned[1], planned[3], planned[4])):
            raise RuntimeError(
                "Atomic SSD restore plan changed during commit: "
                f"planned={planned}, actual="
                f"{(media.start_ns, media.complete_ns, h2d.start_ns, h2d.complete_ns)}")
        capacity_wait_ns = candidate_arrival_ns - arrival_ns
        pressure_stall_ns = (
            media.start_ns - arrival_ns if capacity_pressure else 0)
        self._record_transient_dram_reservation(
            scheduler=scheduler,
            session_id=session_id,
            start_ns=media.start_ns,
            complete_ns=h2d.complete_ns,
            num_bytes=total_bytes,
            reservation_sequence=media.parent_sequence,
            peak_committed_bytes=peak_committed_bytes,
            capacity_wait_ns=capacity_wait_ns,
            pressure_stall_ns=pressure_stall_ns,
        )
        self.metrics.ssd_to_cpu_stage_bytes += total_bytes
        self.metrics.cpu_stage_to_hbm_bytes += total_bytes
        media = replace(
            media,
            transient_dram_capacity_wait_ns=capacity_wait_ns,
        )
        if register_sync_barrier:
            self._register_sync_swap_barrier(
                TransferReservation(
                    kind="ssd_staged_to_hbm",
                    arrival_ns=media.arrival_ns,
                    start_ns=media.start_ns,
                    complete_ns=h2d.complete_ns,
                    service_ns=media.service_ns + h2d.service_ns,
                    queue_wait_ns=(
                        media.queue_wait_ns + h2d.queue_wait_ns),
                    resources=tuple(dict.fromkeys(
                        media.resources + h2d.resources)),
                ),
                int(staging_instance_id),
                int(target_instance_id),
                session_id,
                exposes_owner_request=True,
            )
        return media, h2d

    def _schedule_entry_migration(
            self, entry: IdleKVEntry, kind: str, arrival_ns: int,
            reason: str = "ttl",
            job_arrival_ns: Optional[int] = None) -> None:
        direct_host_writes_before = self.metrics.ssd_host_write_bytes
        if kind == "hbm_to_cpu":
            service_ns = self._cpu_transfer_ns(
                entry.per_rank_bytes, entry.total_bytes)
            num_bytes = entry.total_bytes
        elif kind == "cpu_to_ssd":
            num_bytes = self._ssd_write_bytes(entry)
            service_ns = self._ssd_write_ns(num_bytes)
            ssd_write_phase_offset_ns = 0
            ssd_write_phase_service_ns = service_ns
            self.metrics.ssd_demotion_attempts += 1
        elif kind == "hbm_to_ssd":
            num_bytes = self._ssd_write_bytes(entry)
            num_ranks = self._scheduler(entry.instance_id).num_npus
            per_rank = (
                (num_bytes + num_ranks - 1) // num_ranks if num_bytes else 0)
            cpu_stage_ns = self._cpu_transfer_ns(per_rank, num_bytes)
            media_ns = self._ssd_write_ns(num_bytes)
            service_ns = cpu_stage_ns + media_ns
            ssd_write_phase_offset_ns = cpu_stage_ns
            ssd_write_phase_service_ns = media_ns
            self.metrics.ssd_demotion_attempts += 1
        elif kind == "hbm_to_ssd_direct":
            num_bytes, _, service_ns = self._direct_ssd_write_shape(entry)
            ssd_write_phase_offset_ns = min(
                service_ns,
                int(math.ceil(self.config.ssd_write_latency_us * 1_000)),
            )
            ssd_write_phase_service_ns = max(
                0, service_ns - ssd_write_phase_offset_ns)
            self.metrics.ssd_demotion_attempts += 1
        else:
            raise ValueError(f"Unknown entry migration kind: {kind}")
        # Capacity reclaim is a durable state transition, not an opportunistic
        # prefetch.  Once admitted, it must commit even if the session returns
        # first; that request joins the commit and then restores from the next
        # tier.  Only age/TTL movement retains the legacy cancellation
        # sensitivity behavior.
        deadline_ns = entry.next_use_ns if reason == "ttl" else None
        reservation = self._reserve_transfer(
            kind=kind,
            arrival_ns=arrival_ns,
            service_ns=service_ns,
            source_instance_id=entry.instance_id,
            target_instance_id=None,
            num_bytes=num_bytes,
            background=True,
            deadline_ns=deadline_ns,
            session_id=entry.session_id,
            ssd_write_phase_offset_ns=(
                ssd_write_phase_offset_ns
                if kind in {
                    "hbm_to_ssd", "cpu_to_ssd", "hbm_to_ssd_direct"
                } else 0),
            ssd_write_phase_service_ns=(
                ssd_write_phase_service_ns
                if kind in {
                    "hbm_to_ssd", "cpu_to_ssd", "hbm_to_ssd_direct"
                } else None),
            job_arrival_ns=job_arrival_ns,
        )
        if kind in {
                "hbm_to_ssd", "cpu_to_ssd", "hbm_to_ssd_direct"}:
            # Incremental append depends on the existing durable object. Keep
            # that base alive through the atomic commit even if its ordinary
            # TTL or an unrelated capacity admission fires meanwhile.
            previous = self.ssd_records.get(entry.session_id)
            if previous is not None:
                previous.pinned_until_ns = max(
                    previous.pinned_until_ns,
                    reservation.complete_ns + 1,
                )
            # Host writes are issued by the transfer, not by capacity
            # admission. A completed write therefore consumes endurance even
            # when the new durable object cannot be committed afterwards.
            # Cancelled transfers were already charged proportionally in
            # _reserve_transfer().
            if reservation.completed:
                self.metrics.ssd_host_write_bytes += num_bytes
        if kind == "hbm_to_ssd_direct":
            self.metrics.direct_ssd_write_bytes += (
                self.metrics.ssd_host_write_bytes
                - direct_host_writes_before)
        entry.migration_kind = (
            kind if reservation.completed else f"cancelled:{kind}")
        entry.migration_start_ns = reservation.start_ns
        entry.migration_complete_ns = (
            reservation.complete_ns if reservation.completed else None)
        entry.migration_service_ns = reservation.service_ns
        entry.migration_queue_wait_ns = reservation.queue_wait_ns
        entry.migration_reason = reason
        self._mark_hbm_admission_state_changed()

    def _clear_entry_migration(self, entry: IdleKVEntry) -> None:
        changed = any((
            entry.migration_kind is not None,
            entry.migration_start_ns is not None,
            entry.migration_complete_ns is not None,
            entry.migration_service_ns != 0,
            entry.migration_queue_wait_ns != 0,
            entry.migration_reason is not None,
        ))
        entry.migration_kind = None
        entry.migration_start_ns = None
        entry.migration_complete_ns = None
        entry.migration_service_ns = 0
        entry.migration_queue_wait_ns = 0
        entry.migration_reason = None
        if changed:
            self._mark_hbm_admission_state_changed()

    def _update_idle_peaks(self) -> None:
        hbm_bytes = sum(
            entry.total_bytes for entry in self.entries.values()
            if entry.location == KVLocation.HBM
        )
        cpu_bytes = sum(
            entry.total_bytes for entry in self.entries.values()
            if entry.location == KVLocation.CPU
        )
        self.metrics.peak_idle_hbm_bytes = max(
            self.metrics.peak_idle_hbm_bytes, hbm_bytes)
        self.metrics.peak_idle_cpu_bytes = max(
            self.metrics.peak_idle_cpu_bytes, cpu_bytes)

    def record_simulation_totals(
            self, *, total_request_latency_ns: int = 0,
            total_model_compute_ns: int = 0,
            recompute_model_compute_ns: int = 0,
            total_prompt_tokens: int = 0) -> None:
        """Attach denominators owned by the main serving simulation.

        The KV manager can measure restore stalls itself, but only the serving
        loop/trace backend knows total request latency and kernel compute time.
        Callers may update these monotonically or provide final totals once.
        """
        self.metrics.total_request_latency_ns = max(
            0, int(total_request_latency_ns))
        self.metrics.total_model_compute_ns = max(
            0, int(total_model_compute_ns))
        self.metrics.recompute_model_compute_ns = max(
            0, int(recompute_model_compute_ns))
        self.metrics.total_prompt_tokens = max(0, int(total_prompt_tokens))

    def record_active_preemption_totals(
            self, *, recompute_preemptions: int = 0,
            recompute_tokens: int = 0, cpu_swap_preemptions: int = 0,
            cpu_swap_write_bytes: int = 0,
            cpu_swap_read_bytes: int = 0) -> None:
        """Attach scheduler-owned active-KV preemption counters."""
        self.metrics.active_recompute_preemptions = max(
            0, int(recompute_preemptions))
        self.metrics.active_recompute_tokens = max(
            0, int(recompute_tokens))
        self.metrics.active_cpu_swap_preemptions = max(
            0, int(cpu_swap_preemptions))
        self.metrics.active_cpu_swap_write_bytes = max(
            0, int(cpu_swap_write_bytes))
        self.metrics.active_cpu_swap_read_bytes = max(
            0, int(cpu_swap_read_bytes))

    # ------------------------------------------------------------------
    # Memory ownership helpers.
    # ------------------------------------------------------------------

    def _scheduler(self, instance_id: int):
        try:
            return self.schedulers[instance_id]
        except KeyError as exc:
            raise KeyError(f"Unknown scheduler instance_id={instance_id}") from exc

    def _mark_hbm_admission_state_changed(self) -> None:
        """Publish one manager-side HBM admission dependency transition."""
        self._hbm_admission_state_generation += 1

    @staticmethod
    def _hbm_avail(scheduler) -> int:
        memory = scheduler.memory
        return max(0, memory.npu_mem - memory.npu_used)

    @staticmethod
    def _hbm_kv_ceiling(scheduler) -> int:
        """Return the maximum per-rank HBM capacity usable by KV state."""
        memory = scheduler.memory
        weight_bytes = max(0, int(getattr(memory, "weight", 0) or 0))
        return max(0, int(memory.npu_mem) - weight_bytes)

    def _hbm_logically_reserved(self, instance_id: int) -> int:
        pending = sum(
            allocation.entry.per_rank_bytes
            for allocation in self.pending_hbm_allocations
            if allocation.entry.instance_id == instance_id
        )
        claim = self._active_hbm_reclaim_claims.get(instance_id)
        return pending + (claim.per_rank_bytes if claim is not None else 0)

    def hbm_unreserved_per_rank_bytes(self, instance_id: int) -> int:
        """Return physical HBM slack not promised to manager admissions."""
        scheduler = self._scheduler(int(instance_id))
        return max(
            0,
            self._hbm_avail(scheduler)
            - self._hbm_logically_reserved(int(instance_id)),
        )

    def restore_capacity_state(self, instance_id: int) -> tuple[int, ...]:
        """Return retry-relevant HBM and node-DRAM capacity state."""
        scheduler = self._scheduler(int(instance_id))
        node_id = self._node_id(scheduler)
        now_ns = self._logical_frontier_ns
        transient_bytes = sum(
            reservation.bytes
            for reservation in self._transient_dram_reservations.get(
                node_id, ())
            if reservation.start_ns <= now_ns < reservation.complete_ns
        )
        node_cpu_used = sum(
            candidate.memory.cpu_used
            for candidate in self._node_schedulers(scheduler)
        )
        return (
            int(scheduler.memory.npu_used),
            int(self.hbm_unreserved_per_rank_bytes(instance_id)),
            int(node_cpu_used),
            int(transient_bytes),
            int(self._external_fabric_completion_generation.get(
                int(instance_id), 0)),
            int(self._hbm_admission_state_generation),
        )

    def active_cpu_swap_admissible(
            self, instance_id: int, total_bytes: int, now_ns: int) -> bool:
        """Protect immutable bounce buffers from active decode swap-outs."""
        self.advance(int(now_ns))
        scheduler = self._scheduler(int(instance_id))
        ready_ns = self._cpu_capacity_time(
            scheduler, int(total_bytes), int(now_ns))
        return ready_ns == int(now_ns)

    def record_active_cpu_swap_capacity_fallback(
            self, instance_id: int, request_id: int,
            total_bytes: int, now_ns: int) -> None:
        self.metrics.active_cpu_swap_capacity_fallbacks += 1
        self.events.append({
            "time_ns": int(now_ns),
            "event": "active_cpu_swap_capacity_fallback",
            "instance_id": int(instance_id),
            "request_id": int(request_id),
            "bytes": int(total_bytes),
            "fallback": "recompute",
        })

    def _cpu_avail(self, scheduler) -> int:
        """Return node-shared CPU capacity, not per-instance duplication."""
        node_id = getattr(scheduler, "node_id", scheduler.instance_id)
        peers = [
            candidate for candidate in self.schedulers.values()
            if getattr(candidate, "node_id", candidate.instance_id) == node_id
        ]
        capacity = min(candidate.memory.cpu_mem for candidate in peers)
        used = sum(candidate.memory.cpu_used for candidate in peers)
        return max(0, capacity - used)

    def _account_residence(self, entry: IdleKVEntry, until_ns: int) -> None:
        duration = max(0, until_ns - entry.tier_since_ns)
        if entry.location == KVLocation.HBM:
            self.metrics.hbm_byte_ns += entry.total_bytes * duration
        elif entry.location == KVLocation.CPU:
            self.metrics.cpu_byte_ns += entry.total_bytes * duration
        # Durable SSD records are accounted independently because a
        # keep-on-read copy can outlive the idle entry and remain resident
        # while the next turn runs from HBM.
        entry.tier_since_ns = until_ns

    def _account_ssd_record(self, record: SSDRecord, until_ns: int) -> None:
        duration = max(0, until_ns - record.accounted_until_ns)
        self.metrics.ssd_byte_ns += record.bytes * duration
        record.accounted_until_ns = max(record.accounted_until_ns, until_ns)

    @staticmethod
    def _capacity_time(
            available: int, events: Sequence[tuple[int, int]], needed: int,
            now_ns: int) -> Optional[int]:
        """Return the earliest timestamp with ``needed`` unreserved bytes."""
        if available >= needed:
            return int(now_ns)
        index = 0
        ordered = sorted(events)
        while index < len(ordered):
            event_ns = ordered[index][0]
            delta = 0
            while index < len(ordered) and ordered[index][0] == event_ns:
                delta += ordered[index][1]
                index += 1
            available += delta
            if available >= needed:
                return event_ns
        return None

    def _node_schedulers(self, scheduler) -> list[object]:
        node_id = self._node_id(scheduler)
        return [
            candidate for candidate in self.schedulers.values()
            if self._node_id(candidate) == node_id
        ]

    def _cpu_capacity_bytes(self, scheduler) -> int:
        peers = self._node_schedulers(scheduler)
        return min(candidate.memory.cpu_mem for candidate in peers)

    def _cpu_capacity_time(
            self, scheduler, needed_bytes: int, now_ns: int) -> Optional[int]:
        if needed_bytes > self._cpu_capacity_bytes(scheduler):
            return None
        node_id = self._node_id(scheduler)
        available = self._cpu_avail(scheduler)
        events: list[tuple[int, int]] = []
        for candidate in self.entries.values():
            if self._node_id(self._scheduler(candidate.instance_id)) != node_id:
                continue
            if candidate.migration_complete_ns is None:
                continue
            if (candidate.location == KVLocation.CPU
                    and candidate.migration_kind == "cpu_to_ssd"):
                events.append((
                    candidate.migration_complete_ns, candidate.total_bytes))
            elif (candidate.location == KVLocation.HBM
                    and candidate.migration_kind == "hbm_to_cpu"):
                # A scheduled HBM->CPU commit is already a reservation on the
                # node-shared CPU pool.
                events.append((
                    candidate.migration_complete_ns, -candidate.total_bytes))
        # A CPU hit remains physically pinned until its H2D DMA completes.
        # It is not present in ``entries`` after ownership preparation, but
        # its exact release is still usable for a future capacity reservation.
        if self.config.tiered_family:
            for pending in self.pending_source_releases:
                candidate = pending.entry
                if (candidate.location == KVLocation.CPU
                        and self._node_id(
                            self._scheduler(candidate.instance_id)) == node_id):
                    events.append((pending.ready_ns, candidate.total_bytes))
        for reservation in self._transient_dram_reservations.get(node_id, ()):
            if reservation.complete_ns <= now_ns:
                continue
            if reservation.start_ns <= now_ns:
                available -= reservation.bytes
            else:
                events.append((reservation.start_ns, -reservation.bytes))
            events.append((reservation.complete_ns, reservation.bytes))

        # A durable CPU-cache allocation must preserve every already-published
        # future consumer, not merely fit at one instant.  This includes both
        # full-object bounce buffers and HBM->CPU demotions that have already
        # reserved a later commit.  An instantaneous fit can otherwise admit
        # two copies against the same current slack; the first future commit
        # then invalidates the second demotion and any HBM reclaim claim that
        # depends on it.  Find the first event boundary whose entire known
        # suffix retains enough slack for the new object.
        grouped: Dict[int, int] = {}
        for event_ns, delta in events:
            if event_ns < now_ns:
                continue
            grouped[int(event_ns)] = grouped.get(int(event_ns), 0) + int(delta)
        states = [(int(now_ns), int(available))]
        for event_ns, delta in sorted(grouped.items()):
            if event_ns == now_ns:
                states[0] = (event_ns, states[0][1] + delta)
            else:
                states.append((event_ns, states[-1][1] + delta))
        suffix_min = [0] * len(states)
        running_min = math.inf
        for index in range(len(states) - 1, -1, -1):
            running_min = min(running_min, states[index][1])
            suffix_min[index] = int(running_min)
        for (event_ns, _), minimum_available in zip(states, suffix_min):
            if minimum_available >= needed_bytes:
                return int(event_ns)
        return None

    def _reserve_cpu_capacity(
            self, scheduler, needed_bytes: int, now_ns: int,
            exclude_session: Optional[str] = None) -> Optional[int]:
        """Reserve future CPU space, cascading CPU LRU victims to SSD."""
        ready_ns = self._cpu_capacity_time(
            scheduler, needed_bytes, now_ns)
        if ready_ns is not None:
            return ready_ns
        if not self.config.tiered_family:
            return None

        node_id = self._node_id(scheduler)
        candidates = sorted(
            (
                entry for entry in self.entries.values()
                if entry.location == KVLocation.CPU
                and self._node_id(self._scheduler(entry.instance_id)) == node_id
                and entry.migration_kind is None
                and entry.session_id != exclude_session
                and entry.session_id not in
                self._capacity_pinned_sessions()
            ),
            key=lambda entry: (entry.last_access_ns, entry.session_id),
        )
        for victim in candidates:
            self._schedule_entry_migration(
                victim, "cpu_to_ssd", now_ns, reason="cpu_capacity")
            if victim.migration_complete_ns is not None:
                self.events.append({
                    "time_ns": now_ns,
                    "session_id": victim.session_id,
                    "event": "cpu_capacity_eviction_scheduled",
                    "complete_ns": victim.migration_complete_ns,
                    "bytes": victim.total_bytes,
                })
            ready_ns = self._cpu_capacity_time(
                scheduler, needed_bytes, now_ns)
            if ready_ns is not None:
                return ready_ns
        return None

    def _schedule_hbm_demotion(
            self, entry: IdleKVEntry, now_ns: int, reason: str) -> bool:
        """Schedule an HBM victim to CPU, cascading CPU pressure to SSD."""
        # HBM-only policies must never acquire a CPU/SSD transfer merely
        # because host capacity happens to be available. Their pressure path
        # is the immediate whole-session LRU drop in _reserve_hbm().
        if self.config.policy in {"preserve", "hbm_lru_recompute"}:
            return False
        if self.config.policy == "hbm_ssd_direct":
            if not reason.startswith("hbm_capacity"):
                return False
            if not self._reserve_direct_ssd_capacity(entry, now_ns):
                # SSD is the terminal tier for this baseline. If no durable
                # destination can be admitted, discard the HBM LRU victim so
                # pressure still makes deterministic forward progress.
                self._drop_entry(entry, now_ns, "ssd_capacity")
                return True
            self._schedule_entry_migration(
                entry, "hbm_to_ssd_direct", now_ns, reason=reason)
            if entry.migration_complete_ns is None:
                self._release_direct_ssd_capacity(entry, now_ns)
            return entry.migration_complete_ns is not None
        scheduler = self._scheduler(entry.instance_id)
        cpu_ready_ns = self._reserve_cpu_capacity(
            scheduler, entry.total_bytes, now_ns,
            exclude_session=entry.session_id)
        if cpu_ready_ns is not None:
            self._schedule_entry_migration(
                entry, "hbm_to_cpu", cpu_ready_ns, reason=reason,
                job_arrival_ns=now_ns)
        elif self.config.policy == "cpu":
            # Preserve the legacy CPU-only behavior: issue the attempted copy
            # and decide admission at completion without looking ahead to a
            # future foreground source release.
            self._schedule_entry_migration(
                entry, "hbm_to_cpu", now_ns, reason=reason)
        elif self.config.tiered_family:
            # This is a necessary bypass only when CPU cannot admit the object
            # even after evicting every eligible idle CPU victim.
            bypass_reason = (
                f"{reason}_cpu_bypass"
                if reason.startswith("hbm_capacity")
                else "cpu_capacity_bypass")
            self._schedule_entry_migration(
                entry, "hbm_to_ssd", now_ns, reason=bypass_reason)
        else:
            return False
        return entry.migration_complete_ns is not None

    def _hbm_capacity_time(
            self, instance_id: int, needed_per_rank: int,
            now_ns: int) -> Optional[int]:
        scheduler = self._scheduler(instance_id)
        if needed_per_rank > self._hbm_kv_ceiling(scheduler):
            return None
        # Future allocations are physical only at ``ready_ns``, but their
        # bytes are logically reserved as soon as admission succeeds.  Without
        # this subtraction, a later request can consume today's physical slack
        # and leave the earlier reservation unable to commit after its victim
        # demotion completes.
        logically_reserved = self._hbm_logically_reserved(instance_id)
        events: list[tuple[int, int]] = []
        for candidate in self.entries.values():
            if (candidate.instance_id == instance_id
                    and candidate.location == KVLocation.HBM
                    and candidate.migration_complete_ns is not None
                    and candidate.migration_kind in {
                        "hbm_to_cpu", "hbm_to_ssd",
                        "hbm_to_ssd_direct"}):
                events.append((
                    candidate.migration_complete_ns,
                    candidate.per_rank_bytes,
                ))
        return self._capacity_time(
            self._hbm_avail(scheduler) - logically_reserved,
            events,
            needed_per_rank,
            now_ns,
        )

    def _reserve_hbm(
            self, entry: IdleKVEntry, now_ns: int) -> Optional[int]:
        """Reserve HBM now or after atomic LRU demotions complete."""
        scheduler = self._scheduler(entry.instance_id)
        ready_ns = self._hbm_capacity_time(
            entry.instance_id, entry.per_rank_bytes, now_ns)
        if ready_ns is None:
            candidates = sorted(
                (
                    victim for victim in self.entries.values()
                    if victim.instance_id == entry.instance_id
                    and victim.location == KVLocation.HBM
                    and victim.migration_kind is None
                    and victim.session_id != entry.session_id
                    and victim.session_id not in
                    self._capacity_pinned_sessions()
                ),
                key=lambda victim: (victim.last_access_ns, victim.session_id),
            )
            if self.config.policy == "hbm_ssd_direct":
                scheduled = sum(
                    victim.per_rank_bytes for victim in self.entries.values()
                    if victim.instance_id == entry.instance_id
                    and victim.location == KVLocation.HBM
                    and victim.migration_complete_ns is not None
                    and victim.migration_kind == "hbm_to_ssd_direct"
                )
                reclaimable = sum(
                    victim.per_rank_bytes for victim in candidates)
                if (self._hbm_avail(scheduler) + scheduled + reclaimable
                        - self._hbm_logically_reserved(entry.instance_id)
                        < entry.per_rank_bytes):
                    return None
            if self.config.policy in {
                    "preserve", "hbm_lru_recompute"}:
                reclaimable = sum(
                    victim.per_rank_bytes for victim in candidates)
                if (self._hbm_avail(scheduler) + reclaimable
                        - self._hbm_logically_reserved(entry.instance_id)
                        < entry.per_rank_bytes):
                    # Do not destroy useful idle sessions when the newcomer
                    # cannot fit even after every eligible LRU victim drops.
                    return None
            for victim in candidates:
                if self.config.policy in {
                        "preserve", "hbm_lru_recompute"}:
                    victim_bytes = victim.total_bytes
                    self._drop_entry(victim, now_ns, "hbm_capacity")
                    self.events.append({
                        "time_ns": now_ns,
                        "session_id": victim.session_id,
                        "event": "hbm_capacity_lru_drop",
                        "bytes": victim_bytes,
                    })
                    ready_ns = self._hbm_capacity_time(
                        entry.instance_id, entry.per_rank_bytes, now_ns)
                    if ready_ns is not None:
                        break
                    continue
                if not self._schedule_hbm_demotion(
                        victim, now_ns, reason="hbm_capacity"):
                    continue
                if victim.location == KVLocation.DROPPED:
                    ready_ns = self._hbm_capacity_time(
                        entry.instance_id, entry.per_rank_bytes, now_ns)
                    if ready_ns is not None:
                        break
                    continue
                self.events.append({
                    "time_ns": now_ns,
                    "session_id": victim.session_id,
                    "event": "hbm_capacity_demotion_scheduled",
                    "complete_ns": victim.migration_complete_ns,
                    "bytes": victim.total_bytes,
                })
                ready_ns = self._hbm_capacity_time(
                    entry.instance_id, entry.per_rank_bytes, now_ns)
                if ready_ns is not None:
                    break
        if ready_ns is None:
            return None
        if ready_ns == now_ns:
            scheduler.memory.allocate(entry.per_rank_bytes, Device.NPU)
            self._mark_hbm_admission_state_changed()
        else:
            self.pending_hbm_allocations.append(PendingHBMAllocation(
                entry=entry, ready_ns=ready_ns))
            self._mark_hbm_admission_state_changed()
        return ready_ns

    def _record_active_reclaim_rejection(
            self, *, now_ns: int, instance_id: int,
            needed_per_rank_bytes: int, reason: str,
            owner_kind: str, owner_id: Optional[int]) -> None:
        """Count every rejection while retaining only bounded diagnostics."""
        reason = str(reason)
        self._active_reclaim_rejection_counts[reason] = (
            self._active_reclaim_rejection_counts.get(reason, 0) + 1)
        if (len(self._active_reclaim_rejection_samples)
                >= self._active_reclaim_rejection_sample_limit):
            return
        sample = {
            "time_ns": int(now_ns),
            "event": "active_hbm_reclaim_rejected",
            "instance_id": int(instance_id),
            "per_rank_bytes": int(needed_per_rank_bytes),
            "reason": reason,
            "owner_kind": str(owner_kind),
            "owner_id": owner_id,
        }
        self._active_reclaim_rejection_samples.append(sample)
        self.events.append(sample)

    def claim_active_hbm_reclaim(
            self, instance_id: int, needed_per_rank_bytes: int,
            now_ns: int, owner_kind: str = "legacy",
            owner_id: Optional[int] = None) -> Optional[int]:
        """Claim HBM for active scheduler work, reclaiming idle LRU state.

        Exactly one claim may be outstanding per instance. Polling that claim
        with the same size is idempotent and does not enqueue more transfers.
        The return value is the current time, a future capacity-ready time, or
        ``None`` when even all eligible idle victims cannot make the request
        fit. Successful demotions retain their HBM source until atomic commit.
        """
        instance_id = int(instance_id)
        needed_per_rank_bytes = int(needed_per_rank_bytes)
        now_ns = int(now_ns)
        owner_kind = str(owner_kind or "legacy")
        owner_id = None if owner_id is None else int(owner_id)
        if needed_per_rank_bytes < 0:
            raise ValueError("needed_per_rank_bytes must be non-negative")
        scheduler = self._scheduler(instance_id)
        if needed_per_rank_bytes == 0:
            return now_ns

        existing = self._active_hbm_reclaim_claims.get(instance_id)
        if existing is not None:
            if (existing.owner_kind, existing.owner_id) != (
                    owner_kind, owner_id):
                return None
            if existing.per_rank_bytes != needed_per_rank_bytes:
                raise RuntimeError(
                    "Only one active HBM reclaim claim may be outstanding per "
                    f"instance: instance={instance_id}, "
                    f"claimed={existing.per_rank_bytes}, "
                    f"requested={needed_per_rank_bytes}")
            return max(now_ns, existing.ready_ns)

        if needed_per_rank_bytes > self._hbm_kv_ceiling(scheduler):
            self._record_active_reclaim_rejection(
                now_ns=now_ns,
                instance_id=instance_id,
                needed_per_rank_bytes=needed_per_rank_bytes,
                reason="kv_ceiling",
                owner_kind=owner_kind,
                owner_id=owner_id,
            )
            return None

        ready_ns = self._hbm_capacity_time(
            instance_id, needed_per_rank_bytes, now_ns)
        if ready_ns is None:
            if self.synchronous_swap_enabled and scheduler.inflight:
                self.events.append({
                    "time_ns": now_ns,
                    "event": "active_hbm_reclaim_deferred_for_iteration",
                    "instance_id": instance_id,
                    "per_rank_bytes": needed_per_rank_bytes,
                })
                return None
            candidates = sorted(
                (
                    victim for victim in self.entries.values()
                    if victim.instance_id == instance_id
                    and victim.location == KVLocation.HBM
                    and victim.migration_kind is None
                    and victim.session_id not in
                    self._capacity_pinned_sessions()
                ),
                key=lambda victim: (
                    victim.last_access_ns, victim.session_id),
            )
            scheduled = sum(
                victim.per_rank_bytes for victim in self.entries.values()
                if victim.instance_id == instance_id
                and victim.location == KVLocation.HBM
                and victim.migration_complete_ns is not None
                and victim.migration_kind in {
                    "hbm_to_cpu", "hbm_to_ssd", "hbm_to_ssd_direct"
                }
            )
            reclaimable = sum(
                victim.per_rank_bytes for victim in candidates)
            if (self._hbm_avail(scheduler) + scheduled + reclaimable
                    - self._hbm_logically_reserved(instance_id)
                    < needed_per_rank_bytes):
                self._record_active_reclaim_rejection(
                    now_ns=now_ns,
                    instance_id=instance_id,
                    needed_per_rank_bytes=needed_per_rank_bytes,
                    reason="capacity",
                    owner_kind=owner_kind,
                    owner_id=owner_id,
                )
                return None

            for victim in candidates:
                victim_per_rank_bytes = victim.per_rank_bytes
                victim_total_bytes = victim.total_bytes
                if self.config.policy in {
                        "preserve", "hbm_lru_recompute"}:
                    self._drop_entry(victim, now_ns, "hbm_capacity")
                    self.events.append({
                        "time_ns": now_ns,
                        "session_id": victim.session_id,
                        "event": "hbm_capacity_lru_drop",
                        "bytes": victim_total_bytes,
                        "reason": "active_hbm_reclaim",
                    })
                else:
                    demotion_scheduled = self._schedule_hbm_demotion(
                        victim, now_ns,
                        reason="hbm_capacity_active_reclaim")
                    cancelled_kind = victim.migration_kind
                    if (not demotion_scheduled
                            and cancelled_kind is not None
                            and cancelled_kind.startswith("cancelled:")):
                        raise RuntimeError(
                            "Capacity-triggered demotion was incorrectly "
                            "cancelled by next use: "
                            f"session={victim.session_id}, "
                            f"kind={cancelled_kind}")
                    elif victim.location == KVLocation.DROPPED:
                        self.events.append({
                            "time_ns": now_ns,
                            "session_id": victim.session_id,
                            "event": "active_hbm_reclaim_terminal_drop",
                            "reason": victim.drop_reason,
                            "bytes": victim_total_bytes,
                            "per_rank_bytes": victim_per_rank_bytes,
                        })
                    elif demotion_scheduled:
                        self.events.append({
                            "time_ns": now_ns,
                            "session_id": victim.session_id,
                            "event": "hbm_capacity_demotion_scheduled",
                            "complete_ns": victim.migration_complete_ns,
                            "bytes": victim_total_bytes,
                            "reason": "active_hbm_reclaim",
                        })

                ready_ns = self._hbm_capacity_time(
                    instance_id, needed_per_rank_bytes, now_ns)
                if ready_ns is not None:
                    break

        if ready_ns is None:
            self._record_active_reclaim_rejection(
                now_ns=now_ns,
                instance_id=instance_id,
                needed_per_rank_bytes=needed_per_rank_bytes,
                reason="capacity_after_reclaim",
                owner_kind=owner_kind,
                owner_id=owner_id,
            )
            return None

        total_bytes = needed_per_rank_bytes * int(scheduler.num_npus)
        claim = ActiveHBMReclaimClaim(
            instance_id=instance_id,
            per_rank_bytes=needed_per_rank_bytes,
            total_bytes=total_bytes,
            admitted_ns=now_ns,
            ready_ns=int(ready_ns),
            owner_kind=owner_kind,
            owner_id=owner_id,
        )
        self._active_hbm_reclaim_claims[instance_id] = claim
        self._mark_hbm_admission_state_changed()
        self.metrics.active_hbm_reclaim_admissions += 1
        self.metrics.active_hbm_reclaim_bytes += total_bytes
        self.metrics.active_hbm_reclaim_per_rank_bytes += needed_per_rank_bytes
        self.metrics.active_hbm_reclaim_wait_ns += max(
            0, claim.ready_ns - claim.admitted_ns)
        self.events.append({
            "time_ns": now_ns,
            "event": "active_hbm_reclaim_admit",
            "instance_id": instance_id,
            "ready_ns": claim.ready_ns,
            "wait_ns": max(0, claim.ready_ns - claim.admitted_ns),
            "bytes": total_bytes,
            "per_rank_bytes": needed_per_rank_bytes,
            "owner_kind": owner_kind,
            "owner_id": owner_id,
        })
        return claim.ready_ns

    def active_hbm_reclaim_claim(
            self, instance_id: int) -> Optional[ActiveHBMReclaimClaim]:
        """Return the outstanding claim without consuming its reservation."""
        return self._active_hbm_reclaim_claims.get(int(instance_id))

    def consume_active_hbm_reclaim(
            self, instance_id: int,
            now_ns: int, owner_kind: Optional[str] = None,
            owner_id: Optional[int] = None) -> Optional[ActiveHBMReclaimClaim]:
        """Consume a ready claim and release its logical reservation.

        The scheduler must allocate the returned ``per_rank_bytes``
        synchronously before invoking another manager admission operation.
        """
        instance_id = int(instance_id)
        now_ns = int(now_ns)
        claim = self._active_hbm_reclaim_claims.get(instance_id)
        if claim is None:
            return None
        if owner_kind is not None:
            expected_owner = (
                str(owner_kind or "legacy"),
                None if owner_id is None else int(owner_id),
            )
            if (claim.owner_kind, claim.owner_id) != expected_owner:
                return None
        if now_ns < claim.ready_ns:
            raise RuntimeError(
                "Active HBM reclaim claim is not ready: "
                f"instance={instance_id}, now={now_ns}, "
                f"ready={claim.ready_ns}")
        self.advance(now_ns)
        scheduler = self._scheduler(instance_id)
        available = self._hbm_avail(scheduler)
        if available < claim.per_rank_bytes:
            raise RuntimeError(
                "Active HBM reclaim claim was oversubscribed at consume: "
                f"instance={instance_id}, needed={claim.per_rank_bytes}, "
                f"available={available}")
        del self._active_hbm_reclaim_claims[instance_id]
        self._mark_hbm_admission_state_changed()
        self.events.append({
            "time_ns": now_ns,
            "event": "active_hbm_reclaim_consume",
            "instance_id": instance_id,
            "ready_ns": claim.ready_ns,
            "bytes": claim.total_bytes,
            "per_rank_bytes": claim.per_rank_bytes,
            "owner_kind": claim.owner_kind,
            "owner_id": claim.owner_id,
        })
        return claim

    def cancel_active_hbm_reclaim(
            self, instance_id: int,
            now_ns: int) -> Optional[ActiveHBMReclaimClaim]:
        """Cancel an outstanding claim without cancelling useful demotions."""
        instance_id = int(instance_id)
        claim = self._active_hbm_reclaim_claims.pop(instance_id, None)
        if claim is None:
            return None
        self._mark_hbm_admission_state_changed()
        self.events.append({
            "time_ns": int(now_ns),
            "event": "active_hbm_reclaim_cancel",
            "instance_id": instance_id,
            "ready_ns": claim.ready_ns,
            "bytes": claim.total_bytes,
            "per_rank_bytes": claim.per_rank_bytes,
            "owner_kind": claim.owner_kind,
            "owner_id": claim.owner_id,
        })
        return claim

    def _cancel_hbm_reservation(
            self, entry: IdleKVEntry, ready_ns: int) -> None:
        pending = next(
            (
                candidate for candidate in self.pending_hbm_allocations
                if candidate.entry is entry
            ),
            None,
        )
        if pending is not None:
            self.pending_hbm_allocations.remove(pending)
        else:
            self._scheduler(entry.instance_id).memory.free(
                entry.per_rank_bytes, Device.NPU)
        self._mark_hbm_admission_state_changed()

    def _allocate_hbm(self, entry: IdleKVEntry, now_ns: int) -> bool:
        """Compatibility helper for callers requiring immediate capacity."""
        ready_ns = self._reserve_hbm(entry, now_ns)
        if ready_ns is None:
            return False
        if ready_ns != now_ns:
            self._cancel_hbm_reservation(entry, ready_ns)
            return False
        return True

    def _ssd_node_for_session(self, session_id: str) -> int:
        entry = self.entries.get(session_id)
        if entry is not None:
            return int(self._node_id(self._scheduler(entry.instance_id)))
        record = self.ssd_records.get(session_id)
        if record is not None:
            return int(record.node_id)
        reserved_node = self._direct_ssd_capacity_reservation_nodes.get(
            session_id)
        if reserved_node is not None:
            return int(reserved_node)
        if len(self._ssd_node_ids) == 1:
            return int(self._ssd_node_ids[0])
        raise RuntimeError(
            "Cannot resolve node-local SSD ownership for session "
            f"{session_id!r}")

    def _ssd_used_bytes_on_node(self, node_id: int) -> int:
        return sum(
            record.bytes
            for record in self.ssd_records.values()
            if int(record.node_id) == int(node_id)
        )

    def _ssd_reserved_bytes(
            self, exclude_session: Optional[str] = None,
            node_id: Optional[int] = None) -> int:
        return sum(
            num_bytes
            for session_id, num_bytes
            in self._direct_ssd_capacity_reservations.items()
            if session_id != exclude_session
            and (
                node_id is None
                or self._ssd_node_for_session(session_id) == int(node_id)
            )
        )

    def _reserve_direct_ssd_capacity(
            self, entry: IdleKVEntry, now_ns: int) -> bool:
        """Reserve full-object SSD capacity before issuing a direct write."""
        if entry.session_id in self._direct_ssd_capacity_reservations:
            raise RuntimeError(
                f"Duplicate direct SSD capacity reservation: {entry.session_id}")
        node_id = int(self._node_id(self._scheduler(entry.instance_id)))
        previous = self.ssd_records.get(entry.session_id)
        if previous is not None and int(previous.node_id) != node_id:
            raise RuntimeError(
                "Cross-node SSD lineage replacement is not modeled: "
                f"session={entry.session_id}, source_node={previous.node_id}, "
                f"target_node={node_id}")
        old_bytes = previous.bytes if previous is not None else 0
        extra_bytes = max(0, entry.total_bytes - old_bytes)
        if entry.total_bytes > self.config.ssd_capacity_bytes:
            return False
        if not self._ensure_ssd_capacity(
                entry.session_id, extra_bytes, now_ns, node_id=node_id):
            return False
        self._direct_ssd_capacity_reservations[entry.session_id] = extra_bytes
        self._direct_ssd_capacity_reservation_nodes[entry.session_id] = node_id
        committed_reserved = (
            self.ssd_used_bytes + self._ssd_reserved_bytes())
        node_committed_reserved = (
            self._ssd_used_bytes_on_node(node_id)
            + self._ssd_reserved_bytes(node_id=node_id)
        )
        self.metrics.peak_ssd_committed_reserved_bytes = max(
            self.metrics.peak_ssd_committed_reserved_bytes,
            committed_reserved,
        )
        self.events.append({
            "time_ns": now_ns,
            "session_id": entry.session_id,
            "event": "ssd_capacity_reserve",
            "bytes": extra_bytes,
            "node_id": node_id,
            "node_committed_reserved_bytes": node_committed_reserved,
            "committed_reserved_bytes": committed_reserved,
        })
        return True

    def _release_direct_ssd_capacity(
            self, entry: IdleKVEntry, now_ns: int) -> None:
        reserved = self._direct_ssd_capacity_reservations.pop(
            entry.session_id, None)
        node_id = self._direct_ssd_capacity_reservation_nodes.pop(
            entry.session_id, None)
        if reserved is not None:
            self.events.append({
                "time_ns": now_ns,
                "session_id": entry.session_id,
                "event": "ssd_capacity_release",
                "bytes": reserved,
                "node_id": node_id,
            })

    def _ensure_ssd_capacity(
            self, session_id: str, extra_bytes: int, now_ns: int,
            node_id: Optional[int] = None) -> bool:
        if extra_bytes <= 0:
            return True
        if node_id is None:
            node_id = self._ssd_node_for_session(session_id)
        node_id = int(node_id)
        if node_id not in self._ssd_node_ids:
            raise RuntimeError(f"Unknown SSD host node: {node_id}")
        previous = self.ssd_records.get(session_id)
        if previous is not None and int(previous.node_id) != node_id:
            raise RuntimeError(
                "Cross-node SSD capacity update is not modeled: "
                f"session={session_id}, source_node={previous.node_id}, "
                f"target_node={node_id}")
        own_bytes = previous.bytes if previous is not None else 0
        if own_bytes + extra_bytes > self.config.ssd_capacity_bytes:
            return False
        reserved_sessions = set(self._direct_ssd_capacity_reservations)
        reserved_other = self._ssd_reserved_bytes(
            exclude_session=session_id, node_id=node_id)
        node_used_bytes = self._ssd_used_bytes_on_node(node_id)
        pinned_reads = {
            pending.entry.session_id
            for pending in self.pending_source_releases
            if (pending.entry.location == KVLocation.SSD
                or pending.remove_ssd_record)
        }
        pinned_prepares = self._capacity_pinned_sessions()
        victims = sorted(
            ((sid, record) for sid, record in self.ssd_records.items()
             if sid != session_id
             and int(record.node_id) == node_id
             and sid not in reserved_sessions
             and sid not in pinned_reads
             and sid not in pinned_prepares
             and not (
                 record.pinned_until_ns
                 and record.pinned_until_ns > now_ns
             )),
            key=lambda item: (item[1].last_access_ns, item[0]),
        )
        for sid, record in victims:
            if (node_used_bytes + reserved_other + extra_bytes
                    <= self.config.ssd_capacity_bytes):
                break
            self._account_ssd_record(record, now_ns)
            node_used_bytes -= record.bytes
            self.ssd_used_bytes -= record.bytes
            del self.ssd_records[sid]
            other = self.entries.get(sid)
            if other is not None and other.location == KVLocation.SSD:
                self._account_residence(other, now_ns)
                other.location = KVLocation.DROPPED
                other.drop_reason = "ssd_capacity_eviction"
                other.total_bytes = 0
                other.per_rank_bytes = 0
            self.metrics.capacity_drops += 1
            self.metrics.ssd_capacity_evictions += 1
            self.events.append({"time_ns": now_ns, "session_id": sid,
                                "event": "ssd_capacity_drop",
                                "node_id": node_id,
                                "bytes": record.bytes})
        return (node_used_bytes + reserved_other + extra_bytes
                <= self.config.ssd_capacity_bytes)

    # ------------------------------------------------------------------
    # Public lifecycle.
    # ------------------------------------------------------------------

    @staticmethod
    def _increment_cross_count(
            table: Dict[str, Dict[str, int]], left: str,
            right: str) -> None:
        row = table.setdefault(str(left), {})
        row[str(right)] = row.get(str(right), 0) + 1

    def record_agentic_request(self, req) -> None:
        """Count one routed session call under literal all-call scope."""
        if req.session_id is None or req.id in self._classified_request_ids:
            return
        self._classified_request_ids.add(req.id)
        gap_type = str(req.return_gap_type or "unknown")
        if int(req.sub_request_index or 0) == 0:
            residency = "session_start"
            source = "session_start"
        else:
            residency = str(
                req.agentic_kv_residency_at_return or "unknown")
            source = str(req.agentic_kv_source or "unknown")
        self._increment_cross_count(
            self._request_counts_by_residency_and_return,
            residency,
            gap_type,
        )
        self._increment_cross_count(
            self._request_counts_by_source_and_return,
            source,
            gap_type,
        )
        if (req.agentic_kv_async_decode_join
                and req.agentic_kv_restore_ns > 0):
            self.metrics.async_restore_gross_ns += int(
                req.agentic_kv_restore_ns)
            self._increment_async_restore_breakdown(
                source, gap_type, "request_count", 1)
            self._increment_async_restore_breakdown(
                source, gap_type, "gross_ns",
                int(req.agentic_kv_restore_ns))
            if req.agentic_kv_restore_gate_wait_ns > 0:
                self.record_async_restore_gate(
                    req, req.agentic_kv_restore_gate_start_ns)

    def _increment_async_restore_breakdown(
            self, source: str, gap_type: str, metric: str,
            value: int) -> None:
        source_row = self._async_restore_by_source_and_return.setdefault(
            str(source), {})
        cell = source_row.setdefault(str(gap_type), {})
        cell[str(metric)] = cell.get(str(metric), 0) + int(value)

    def record_async_restore_gate(self, req, gate_start_ns: int) -> None:
        """Record the request-local wait at the prefill/restore join once."""
        if (not req.agentic_kv_async_decode_join
                or req.agentic_kv_restore_gate_recorded):
            return
        wait_ns = max(
            0,
            int(req.agentic_kv_restore_ready_time_ns)
            - int(gate_start_ns),
        )
        req.agentic_kv_restore_gate_start_ns = int(gate_start_ns)
        req.agentic_kv_restore_gate_wait_ns = wait_ns
        req.agentic_kv_restore_gate_recorded = True
        self.metrics.async_restore_owner_barrier_ns += wait_ns
        if wait_ns:
            self._async_owner_barrier_intervals.append((
                int(gate_start_ns),
                int(gate_start_ns) + wait_ns,
            ))
        source = str(req.agentic_kv_source or "unknown")
        gap_type = str(req.return_gap_type or "unknown")
        self._increment_async_restore_breakdown(
            source, gap_type, "exposed_owner_barrier_ns", wait_ns)
        self.events.append({
            "time_ns": int(gate_start_ns),
            "event": "async_restore_decode_join",
            "session_id": req.session_id,
            "request_id": req.id,
            "source": source,
            "return_gap_type": gap_type,
            "restore_ready_ns": int(
                req.agentic_kv_restore_ready_time_ns),
            "wait_ns": wait_ns,
        })

    def record_pd_decode_receive_admission(
            self, req, instance_id: int, enqueued_ns: int,
            capacity_ready_ns: int, admitted_ns: int,
            restore_ready_ns: int, per_rank_bytes: int,
            chunk_admission: dict) -> None:
        """Record the strict D-side receive gate for the first P chunk."""
        if (chunk_admission.get(
                "cancelled_by_active_prefill_recompute", False)
                or int(chunk_admission.get("request_id", req.id))
                != int(req.id)):
            raise RuntimeError(
                "P/D decode admission did not receive the exact first "
                f"successful chunk history: request={req.id}")
        capacity_wait_ns = max(
            0, int(capacity_ready_ns) - int(enqueued_ns))
        wait_ns = max(0, int(admitted_ns) - int(enqueued_ns))
        critical_wait_ns = max(
            0, int(capacity_ready_ns)
            - max(int(restore_ready_ns), int(enqueued_ns)),
        )
        total_bytes = int(per_rank_bytes) * int(
            self._scheduler(instance_id).num_npus)
        self.metrics.pd_decode_receive_admissions += 1
        self.metrics.pd_decode_receive_reserved_bytes += total_bytes
        self.metrics.pd_decode_receive_capacity_wait_ns += capacity_wait_ns
        self.metrics.pd_decode_receive_admission_wait_ns += wait_ns
        self.metrics.pd_decode_receive_critical_wait_ns += critical_wait_ns
        self.events.append({
            "time_ns": int(admitted_ns),
            "event": "pd_decode_receive_admission",
            "admission_scope": "first_prefill_chunk",
            "chunk_index": 1,
            "session_id": req.session_id,
            "request_id": req.id,
            "target_instance_id": int(instance_id),
            "enqueued_ns": int(enqueued_ns),
            "capacity_ready_ns": int(capacity_ready_ns),
            "admitted_ns": int(admitted_ns),
            "ready_ns": int(admitted_ns),
            "capacity_wait_ns": capacity_wait_ns,
            "wait_ns": wait_ns,
            "critical_wait_after_restore_ns": critical_wait_ns,
            "critical_wait_basis": (
                "capacity_ready_non_additive_use_pd_launch_gate"),
            "per_rank_bytes": int(per_rank_bytes),
            "bytes": total_bytes,
            "full_per_rank_bytes": int(
                req.pd_decode_full_per_rank_bytes),
            "retained_per_rank_bytes": int(
                req.agentic_kv_retained_per_rank_bytes),
            "newly_reserved_per_rank_bytes": int(per_rank_bytes),
            "initial_owned_per_rank_bytes": int(
                chunk_admission[
                    "decode_current_per_rank_bytes"]),
            "target_owned_per_rank_bytes": int(
                chunk_admission[
                    "decode_target_per_rank_bytes"]),
            "active_prefill_recompute_generation": int(
                chunk_admission.get(
                    "active_prefill_recompute_generation", 0)),
            "source": req.agentic_kv_source,
            "residency_at_return": req.agentic_kv_residency_at_return,
            "return_gap_type": req.return_gap_type,
            "return_gap_source": req.return_gap_source,
        })

    def record_pd_prefill_admission(
            self, req, instance_id: int, enqueued_ns: int,
            capacity_ready_ns: int, admitted_ns: int,
            restore_ready_ns: int, per_rank_bytes: int,
            chunk_admission: dict) -> None:
        """Record P-side HBM admission for the first prefill chunk."""
        if (chunk_admission.get(
                "cancelled_by_active_prefill_recompute", False)
                or int(chunk_admission.get("request_id", req.id))
                != int(req.id)):
            raise RuntimeError(
                "P/D prefill admission did not receive the exact first "
                f"successful chunk history: request={req.id}")
        capacity_wait_ns = max(
            0, int(capacity_ready_ns) - int(enqueued_ns))
        wait_ns = max(0, int(admitted_ns) - int(enqueued_ns))
        critical_wait_ns = max(
            0, int(capacity_ready_ns)
            - max(int(restore_ready_ns), int(enqueued_ns)),
        )
        total_bytes = int(per_rank_bytes) * int(
            self._scheduler(instance_id).num_npus)
        self.metrics.pd_prefill_admissions += 1
        self.metrics.pd_prefill_reserved_bytes += total_bytes
        self.metrics.pd_prefill_capacity_wait_ns += capacity_wait_ns
        self.metrics.pd_prefill_admission_wait_ns += wait_ns
        self.metrics.pd_prefill_admission_critical_wait_ns += critical_wait_ns
        self.events.append({
            "time_ns": int(admitted_ns),
            "event": "pd_prefill_active_admission",
            "admission_scope": "first_prefill_chunk",
            "chunk_index": 1,
            "session_id": req.session_id,
            "request_id": req.id,
            "target_instance_id": int(instance_id),
            "enqueued_ns": int(enqueued_ns),
            "capacity_ready_ns": int(capacity_ready_ns),
            "admitted_ns": int(admitted_ns),
            "ready_ns": int(admitted_ns),
            "capacity_wait_ns": capacity_wait_ns,
            "wait_ns": wait_ns,
            "critical_wait_after_restore_ns": critical_wait_ns,
            "critical_wait_basis": (
                "capacity_ready_non_additive_use_pd_launch_gate"),
            "per_rank_bytes": int(per_rank_bytes),
            "bytes": total_bytes,
            "full_per_rank_bytes": int(
                req.pd_prefill_full_per_rank_bytes),
            "restored_prefix_per_rank_bytes": int(
                req.pd_prefill_initial_restored_per_rank_bytes),
            "newly_reserved_per_rank_bytes": int(per_rank_bytes),
            "initial_owned_per_rank_bytes": int(
                chunk_admission[
                    "prefill_current_per_rank_bytes"]),
            "target_owned_per_rank_bytes": int(
                chunk_admission[
                    "prefill_target_per_rank_bytes"]),
            "active_prefill_recompute_generation": int(
                chunk_admission.get(
                    "active_prefill_recompute_generation", 0)),
            "source": req.agentic_kv_source,
            "residency_at_return": req.agentic_kv_residency_at_return,
            "return_gap_type": req.return_gap_type,
            "return_gap_source": req.return_gap_source,
        })

    def record_pd_launch_admission(
            self, req, enqueued_ns: int, admitted_ns: int,
            restore_ready_ns: int) -> None:
        """Record the one causal P/D scheduler-visibility admission gate."""
        wait_ns = max(0, int(admitted_ns) - int(enqueued_ns))
        critical_wait_ns = max(
            0, int(admitted_ns)
            - max(int(restore_ready_ns), int(enqueued_ns)),
        )
        self.metrics.pd_launch_admissions += 1
        self.metrics.pd_launch_admission_wait_ns += wait_ns
        self.metrics.pd_launch_admission_critical_wait_ns += critical_wait_ns
        self.events.append({
            "time_ns": int(admitted_ns),
            "event": "pd_launch_admission",
            "admission_scope": "first_prefill_chunk",
            "chunk_index": 1,
            "session_id": req.session_id,
            "request_id": req.id,
            "enqueued_ns": int(enqueued_ns),
            "restore_ready_ns": int(restore_ready_ns),
            "prefill_capacity_ready_ns": int(
                req.pd_prefill_capacity_ready_ns),
            "decode_capacity_ready_ns": int(
                req.pd_decode_capacity_ready_ns),
            "admitted_ns": int(admitted_ns),
            "ready_ns": int(admitted_ns),
            "wait_ns": wait_ns,
            "critical_wait_after_restore_ns": critical_wait_ns,
            "critical_wait_scope": "canonical_non_additive_pd_launch_gate",
            "source": req.agentic_kv_source,
            "residency_at_return": req.agentic_kv_residency_at_return,
            "return_gap_type": req.return_gap_type,
            "return_gap_source": req.return_gap_source,
        })

    def record_pd_chunk_admission_cancellation(
            self, req, cancellation: dict) -> None:
        """Record one pre-commit chunk claim cancelled by P/D preemption."""
        required = (
            "request_id", "active_prefill_recompute_generation",
            "enqueued_ns", "cancelled_ns", "wait_ns",
            "critical_wait_after_restore_ns",
        )
        missing = [key for key in required if key not in cancellation]
        if missing:
            raise RuntimeError(
                "Cancelled P/D chunk admission record is incomplete: "
                f"request={req.id}, missing={missing}")
        values = {key: int(cancellation[key]) for key in required}
        if any(value < 0 for value in values.values()):
            raise RuntimeError(
                "Cancelled P/D chunk admission contains a negative value: "
                f"request={req.id}, values={values}")
        if values["request_id"] != int(req.id):
            raise RuntimeError(
                "Cancelled P/D chunk admission request identity changed: "
                f"expected={req.id}, observed={values['request_id']}")
        expected_wait_ns = max(
            0, values["cancelled_ns"] - values["enqueued_ns"])
        if values["wait_ns"] != expected_wait_ns:
            raise RuntimeError(
                "Cancelled P/D chunk wait is not timestamp-derived: "
                f"request={req.id}, recorded={values['wait_ns']}, "
                f"expected={expected_wait_ns}")
        expected_critical_ns = max(
            0,
            values["cancelled_ns"] - max(
                int(req.agentic_kv_restore_ready_time_ns),
                values["enqueued_ns"],
            ),
        )
        if (values["critical_wait_after_restore_ns"]
                != expected_critical_ns):
            raise RuntimeError(
                "Cancelled P/D chunk critical wait is not "
                f"timestamp-derived: request={req.id}, recorded="
                f"{values['critical_wait_after_restore_ns']}, "
                f"expected={expected_critical_ns}")
        if (not cancellation.get(
                "cancelled_by_active_prefill_recompute", False)
                or cancellation.get("committed", True)):
            raise RuntimeError(
                "Cancelled P/D chunk history lost its pre-commit "
                f"provenance: request={req.id}")

        self.metrics.pd_chunk_cancelled_admissions += 1
        if values["wait_ns"]:
            self.metrics.pd_chunk_cancelled_waiting_admissions += 1
        self.metrics.pd_chunk_cancelled_admission_wait_ns += values[
            "wait_ns"]
        self.metrics.pd_chunk_cancelled_admission_critical_wait_ns += values[
            "critical_wait_after_restore_ns"]
        event = dict(cancellation)
        event.update(values)
        event.update({
            "time_ns": values["cancelled_ns"],
            "event": (
                "pd_chunk_admission_cancelled_for_active_prefill_"
                "recompute"),
            "admission_scope": "one_prefill_chunk_atomic_pd_claim",
            "admission_semantics": "cancelled_before_graph_commit",
            "session_id": req.session_id,
            "request_id": int(req.id),
        })
        self.events.append(event)

    def validate_pd_chunk_admission(self, req, admission: dict):
        """Validate one chunk event without changing metrics or events."""
        required = (
            "request_id", "active_prefill_recompute_generation",
            "prefill_instance_id", "decode_instance_id",
            "computed_tokens", "chunk_tokens", "target_tokens",
            "prefill_current_per_rank_bytes",
            "decode_current_per_rank_bytes",
            "prefill_target_per_rank_bytes",
            "decode_target_per_rank_bytes",
            "prefill_delta_per_rank_bytes",
            "decode_delta_per_rank_bytes",
            "prefill_unreserved_per_rank_bytes",
            "decode_unreserved_per_rank_bytes",
            "enqueued_ns", "prefill_capacity_ready_ns",
            "decode_capacity_ready_ns", "admitted_ns", "wait_ns",
            "critical_wait_after_restore_ns",
            "prefill_peak_hbm_used_per_rank_bytes",
            "decode_peak_hbm_used_per_rank_bytes",
        )
        missing = [key for key in required if key not in admission]
        if missing:
            raise RuntimeError(
                "P/D chunk admission record is incomplete: "
                f"request={req.id}, missing={missing}")
        values = {key: int(admission[key]) for key in required}
        if any(value < 0 for value in values.values()):
            raise RuntimeError(
                "P/D chunk admission contains a negative value: "
                f"request={req.id}, values={values}")
        if values["request_id"] != int(req.id):
            raise RuntimeError(
                "P/D chunk admission request identity changed: "
                f"expected={req.id}, observed={values['request_id']}")
        if (values["prefill_target_per_rank_bytes"]
                != values["decode_target_per_rank_bytes"]):
            raise RuntimeError(
                "P/D chunk target block ownership is asymmetric: "
                f"request={req.id}, prefill="
                f"{values['prefill_target_per_rank_bytes']}, decode="
                f"{values['decode_target_per_rank_bytes']}")
        for role in ("prefill", "decode"):
            current = values[f"{role}_current_per_rank_bytes"]
            delta = values[f"{role}_delta_per_rank_bytes"]
            target = values[f"{role}_target_per_rank_bytes"]
            if current + delta != target:
                raise RuntimeError(
                    f"P/D {role} chunk byte conservation failed: "
                    f"request={req.id}, current={current}, delta={delta}, "
                    f"target={target}")
        expected_wait_ns = max(
            0, values["admitted_ns"] - values["enqueued_ns"])
        if values["wait_ns"] != expected_wait_ns:
            raise RuntimeError(
                "P/D chunk admission wait is not timestamp-derived: "
                f"request={req.id}, recorded={values['wait_ns']}, "
                f"expected={expected_wait_ns}")
        expected_critical_ns = max(
            0,
            values["admitted_ns"] - max(
                int(req.agentic_kv_restore_ready_time_ns),
                values["enqueued_ns"],
            ),
        )
        if values["critical_wait_after_restore_ns"] != expected_critical_ns:
            raise RuntimeError(
                "P/D chunk critical wait is not timestamp-derived: "
                f"request={req.id}, recorded="
                f"{values['critical_wait_after_restore_ns']}, "
                f"expected={expected_critical_ns}")

        prefill_total_bytes = (
            values["prefill_delta_per_rank_bytes"]
            * int(self._scheduler(
                values["prefill_instance_id"]).num_npus))
        decode_total_bytes = (
            values["decode_delta_per_rank_bytes"]
            * int(self._scheduler(
                values["decode_instance_id"]).num_npus))
        return values, prefill_total_bytes, decode_total_bytes

    def record_pd_chunk_admission(self, req, admission: dict) -> None:
        """Record one authoritative atomic P/D block claim.

        Queue-recompute capacity observations are deliberately snapshots, not
        reservations.  This event records the later, policy-independent pair
        claim at scheduler dispatch and, when available, links it to the most
        recent snapshot for the same session.  The two timestamps must remain
        distinct so an offline validator can detect snapshot-feasible chunks
        that nevertheless waited for real capacity.
        """
        values, prefill_total_bytes, decode_total_bytes = (
            self.validate_pd_chunk_admission(req, admission))
        self.metrics.pd_chunk_admissions += 1
        if values["wait_ns"]:
            self.metrics.pd_chunk_waiting_admissions += 1
        self.metrics.pd_chunk_admitted_tokens += values["chunk_tokens"]
        self.metrics.pd_chunk_prefill_reserved_bytes += prefill_total_bytes
        self.metrics.pd_chunk_decode_reserved_bytes += decode_total_bytes
        self.metrics.pd_chunk_admission_wait_ns += values["wait_ns"]
        self.metrics.pd_chunk_admission_critical_wait_ns += values[
            "critical_wait_after_restore_ns"]

        first_chunk = int(req.pd_chunk_admission_count) == 1
        decision_time_ns = int(req.pd_prefill_admission_enqueued_ns)
        snapshot_event = None
        if first_chunk:
            snapshot_event = next((
                event for event in reversed(self.events)
                if (event.get("event") == "queue_recompute_evaluate"
                    and str(event.get("session_id")) == str(req.session_id)
                    and event.get("capacity_headroom_snapshot") is not None
                    # Preparation and strict P/D pair binding share the same
                    # physical callback timestamp. Exact equality prevents a
                    # later request or chunk from inheriting a stale snapshot
                    # merely because it reuses the session identifier.
                    and int(event.get("time_ns", -1)) == decision_time_ns)
            ), None)
        snapshot = (
            None if snapshot_event is None else
            dict(snapshot_event["capacity_headroom_snapshot"])
        )
        snapshot_feasible = bool(
            snapshot is not None and snapshot.get("feasible", False))
        snapshot_feasible_but_waited = bool(
            snapshot_feasible and values["wait_ns"] > 0)
        if snapshot is not None:
            self.metrics.pd_chunk_snapshot_joined_admissions += 1
        if snapshot_feasible:
            self.metrics.pd_chunk_snapshot_feasible_admissions += 1
        if snapshot_feasible_but_waited:
            self.metrics.pd_chunk_snapshot_feasible_waiting_admissions += 1
            self.metrics.pd_chunk_snapshot_feasible_wait_ns += values[
                "wait_ns"]

        event = dict(values)
        event.update({
            "time_ns": values["admitted_ns"],
            "event": "pd_chunk_admission",
            "admission_scope": "one_prefill_chunk_atomic_pd_claim",
            "admission_semantics": (
                "policy_independent_authoritative_dispatch_claim"),
            "session_id": req.session_id,
            "request_id": int(req.id),
            "chunk_index": int(req.pd_chunk_admission_count),
            "first_chunk": first_chunk,
            "prefill_delta_bytes": prefill_total_bytes,
            "decode_delta_bytes": decode_total_bytes,
            "restore_ready_ns": int(
                req.agentic_kv_restore_ready_time_ns),
            "source": req.agentic_kv_source,
            "residency_at_return": req.agentic_kv_residency_at_return,
            "return_gap_type": req.return_gap_type,
            "return_gap_source": req.return_gap_source,
            "capacity_headroom_snapshot": snapshot,
            "capacity_headroom_snapshot_only": snapshot is not None,
            "capacity_headroom_claimed_by_policy": False,
            "capacity_snapshot_decision_time_ns": (
                None if snapshot_event is None else
                int(snapshot_event["time_ns"])),
            "capacity_snapshot_to_admission_ns": (
                None if snapshot_event is None else
                values["admitted_ns"] - int(snapshot_event["time_ns"])),
            "capacity_snapshot_feasible": snapshot_feasible,
            "snapshot_feasible_but_actual_waited": (
                snapshot_feasible_but_waited),
        })
        self.events.append(event)

    def record_agentic_batch_schedule(self, scheduler, batch) -> None:
        """Record source composition after all members are scheduler-ready."""
        fabric_resources = self._astra_fabric_resources(scheduler)
        batch.agentic_astra_fabric_resources = fabric_resources
        batch.agentic_astra_dispatch_time_ns = None
        source_counts: Dict[str, int] = {}
        return_counts: Dict[str, int] = {}
        source_return_counts: Dict[str, Dict[str, int]] = {}
        for req in batch.requests:
            source = str(req.agentic_kv_source or "initial_or_no_resume")
            source_counts[source] = source_counts.get(source, 0) + 1
            gap_type = str(req.return_gap_type or "unknown")
            return_counts[gap_type] = return_counts.get(gap_type, 0) + 1
            self._increment_cross_count(
                source_return_counts, source, gap_type)
            self._increment_cross_count(
                self._batch_memberships_by_source_and_return,
                source,
                gap_type,
            )
        mixed = (
            source_counts.get("hbm", 0) > 0
            and (
                source_counts.get("cpu", 0)
                + source_counts.get("ssd", 0)
            ) > 0
        )
        batch.agentic_source_counts = source_counts
        batch.agentic_return_gap_type_counts = return_counts
        batch.agentic_source_return_counts = source_return_counts
        batch.agentic_mixed_hbm_lower_tier = mixed
        sync_wait_ns, sync_directions = self._sync_swap_barrier_for_batch(
            scheduler.instance_id, int(batch.batch_time))
        batch.agentic_sync_swap_wait_ns = sync_wait_ns
        batch.agentic_sync_swap_directions = sync_directions
        batch.agentic_sync_swap_barrier_before_batch = sync_wait_ns > 0
        if sync_wait_ns:
            self.metrics.sync_swap_blocked_iterations += 1
            self.metrics.sync_swap_blocked_batch_memberships += len(
                batch.requests)
        if scheduler.pd_type == "prefill":
            self.metrics.agentic_prefill_batches += 1
            if mixed:
                self.metrics.agentic_mixed_hbm_lower_tier_prefill_batches += 1
        self.events.append({
            "time_ns": int(batch.batch_time),
            "event": "agentic_batch_schedule",
            "instance_id": scheduler.instance_id,
            "pd_type": scheduler.pd_type,
            "batch_id": batch.batch_id,
            "request_count": len(batch.requests),
            "source_counts": dict(sorted(source_counts.items())),
            "return_gap_type_counts": dict(sorted(return_counts.items())),
            "source_return_counts": {
                source: dict(sorted(counts.items()))
                for source, counts in sorted(source_return_counts.items())
            },
            "mixed_hbm_lower_tier_resume": mixed,
            "restore_barrier_inside_batch": False,
            "sync_swap_barrier_before_batch": (
                batch.agentic_sync_swap_barrier_before_batch),
            "sync_swap_wait_ns": sync_wait_ns,
            "sync_swap_directions": list(sync_directions),
            "sync_barrier_placement": (
                "pre_dispatch_equivalent"
                if sync_wait_ns else "none"
            ),
            "astra_shared_fabric_resources": list(fabric_resources),
        })

    def record_astra_workload_dispatch(
            self, scheduler, batch, dispatch_ns: int) -> None:
        """Open shared-resource ownership at controller dispatch, not batching.

        DP synchronization can leave a formed batch in ``dp_pending`` for many
        callbacks. Registering at batch formation would invent fabric occupancy
        and can deadlock a return waiting for a batch that ASTRA never received.
        Every actual controller workload submission must call this method once.
        """
        dispatch_ns = int(dispatch_ns)
        previous_dispatch = getattr(
            batch, "agentic_astra_dispatch_time_ns", None)
        if previous_dispatch is not None:
            raise RuntimeError(
                "Duplicate ASTRA workload dispatch registration: "
                f"instance={scheduler.instance_id}, batch={batch.batch_id}, "
                f"first_dispatch={previous_dispatch}, duplicate={dispatch_ns}")
        fabric_resources = tuple(getattr(
            batch,
            "agentic_astra_fabric_resources",
            self._astra_fabric_resources(scheduler),
        ))
        blocked_until = self.model_dispatch_blocked_until(
            scheduler.instance_id, dispatch_ns)
        if blocked_until is not None:
            raise RuntimeError(
                "ASTRA model batch was dispatched through a booked cold "
                "direct-fabric interval: "
                f"instance={scheduler.instance_id}, batch={batch.batch_id}, "
                f"dispatch={dispatch_ns}, blocked_until={blocked_until}")
        batch.agentic_astra_fabric_resources = fabric_resources
        batch.agentic_astra_dispatch_time_ns = dispatch_ns
        for resource in fabric_resources:
            owners = self._astra_fabric_inflight.setdefault(resource, {})
            owner = (int(scheduler.instance_id), int(batch.batch_id))
            if owner in owners:
                raise RuntimeError(
                    "Duplicate ASTRA shared-fabric batch registration: "
                    f"resource={resource}, owner={owner}")
            owners[owner] = dispatch_ns
        self.events.append({
            "time_ns": dispatch_ns,
            "event": "astra_workload_dispatch",
            "instance_id": int(scheduler.instance_id),
            "batch_id": int(batch.batch_id),
            "formed_ns": int(getattr(batch, "batch_time", dispatch_ns)),
            "dispatch_ns": dispatch_ns,
            "formation_to_dispatch_wait_ns": max(
                0, dispatch_ns - int(getattr(
                    batch, "batch_time", dispatch_ns))),
            "resources": list(fabric_resources),
        })

    def record_astra_workload_complete(
            self, scheduler, batch, finish_ns: int) -> int:
        """Close one real or dummy ASTRA workload resource window."""
        finish_ns = int(finish_ns)
        dispatch_ns = getattr(batch, "agentic_astra_dispatch_time_ns", None)
        if dispatch_ns is None:
            raise RuntimeError(
                "ASTRA workload completed without a dispatch dependency: "
                f"instance={scheduler.instance_id}, batch={batch.batch_id}")
        dispatch_ns = int(dispatch_ns)
        if finish_ns < dispatch_ns:
            raise RuntimeError(
                "ASTRA model iteration completed before controller dispatch: "
                f"dispatch={dispatch_ns}, finish={finish_ns}")
        fabric_resources = tuple(batch.agentic_astra_fabric_resources)
        for resource in fabric_resources:
            owner = (int(scheduler.instance_id), int(batch.batch_id))
            owners = self._astra_fabric_inflight.get(resource)
            if owners is None or owner not in owners:
                raise RuntimeError(
                    "ASTRA shared-fabric completion lost its dispatch owner: "
                    f"resource={resource}, owner={owner}")
            start_ns = owners.pop(owner)
            if int(start_ns) != dispatch_ns:
                raise RuntimeError(
                    "ASTRA shared-fabric owner has a mismatched dispatch time: "
                    f"owner={owner}, expected={dispatch_ns}, actual={start_ns}")
            if not owners:
                del self._astra_fabric_inflight[resource]
            self._astra_fabric_intervals.setdefault(resource, []).append((
                dispatch_ns, finish_ns,
                int(scheduler.instance_id), int(batch.batch_id),
            ))
            self._insert_astra_calendar_window(
                resource, dispatch_ns, finish_ns)
            self.metrics.astra_shared_fabric_windows += 1
            self.metrics.astra_shared_fabric_window_ns += (
                finish_ns - dispatch_ns)
            self.events.append({
                "time_ns": finish_ns,
                "event": "astra_shared_fabric_window",
                "resource": resource,
                "instance_id": int(scheduler.instance_id),
                "batch_id": int(batch.batch_id),
                "start_ns": dispatch_ns,
                "complete_ns": finish_ns,
                "duration_ns": finish_ns - dispatch_ns,
                "ownership": "astra_internal_tp_ep_pd",
            })
        batch.agentic_astra_dispatch_time_ns = None
        return finish_ns - dispatch_ns

    def record_agentic_batch_complete(
            self, scheduler, batch, finish_ns: int) -> None:
        dispatch_ns = int(batch.agentic_astra_dispatch_time_ns)
        fabric_resources = tuple(batch.agentic_astra_fabric_resources)
        duration_ns = self.record_astra_workload_complete(
            scheduler, batch, finish_ns)
        self.metrics.agentic_model_iteration_batches += 1
        self.metrics.agentic_model_iteration_execution_ns += duration_ns
        if duration_ns:
            self._model_iteration_intervals.setdefault(
                scheduler.instance_id, []).append((
                    dispatch_ns, int(finish_ns)))
        if scheduler.pd_type == "prefill":
            self.metrics.agentic_prefill_batch_execution_ns += duration_ns
            if batch.agentic_mixed_hbm_lower_tier:
                self.metrics.agentic_mixed_hbm_lower_tier_batch_execution_ns += (
                    duration_ns)
        for req in batch.requests:
            if (not req.agentic_kv_async_decode_join
                    or req.agentic_kv_restore_ns <= 0
                    or not req.is_prefill()):
                continue
            overlap_ns = max(
                0,
                min(int(finish_ns), req.agentic_kv_restore_ready_time_ns)
                - max(
                    dispatch_ns,
                    req.agentic_kv_restore_issue_time_ns,
                ),
            )
            if overlap_ns <= 0:
                continue
            req.agentic_kv_restore_compute_overlap_ns += overlap_ns
            self.metrics.async_restore_compute_overlap_ns += overlap_ns
            self._increment_async_restore_breakdown(
                str(req.agentic_kv_source or "unknown"),
                str(req.return_gap_type or "unknown"),
                "prefill_execution_overlap_ns",
                overlap_ns,
            )
        self.events.append({
            "time_ns": int(finish_ns),
            "event": "agentic_batch_complete",
            "instance_id": scheduler.instance_id,
            "pd_type": scheduler.pd_type,
            "batch_id": batch.batch_id,
            "duration_ns": duration_ns,
            "source_counts": dict(sorted(
                batch.agentic_source_counts.items())),
            "return_gap_type_counts": dict(sorted(
                batch.agentic_return_gap_type_counts.items())),
            "source_return_counts": {
                source: dict(sorted(counts.items()))
                for source, counts in sorted(
                    batch.agentic_source_return_counts.items())
            },
            "mixed_hbm_lower_tier_resume": (
                batch.agentic_mixed_hbm_lower_tier),
            "restore_barrier_inside_batch": False,
            "sync_swap_barrier_before_batch": bool(
                batch.agentic_sync_swap_barrier_before_batch),
            "sync_swap_wait_ns": int(batch.agentic_sync_swap_wait_ns),
            "sync_swap_directions": list(
                batch.agentic_sync_swap_directions),
            "astra_shared_fabric_resources": list(fabric_resources),
        })

    def on_tool_start(
            self, req, completion_time_ns: int, release_time_ns: int,
            return_gap_type: str = "unknown",
            return_gap_source: str = "unknown") -> None:
        """Backward-compatible alias for :meth:`on_idle_start`."""
        self.on_idle_start(
            req,
            completion_time_ns,
            release_time_ns,
            return_gap_type=return_gap_type,
            return_gap_source=return_gap_source,
        )

    def on_idle_start(
            self, req, completion_time_ns: int, release_time_ns: int,
            return_gap_type: str = "unknown",
            return_gap_source: str = "unknown") -> None:
        """Take ownership of a completed turn's KV during an inter-turn gap.

        ``Scheduler.add_done`` has already freed the active request's KV when
        this method runs. The callback atomically reclaims those exact bytes as
        idle HBM ownership before tied admissions run. Only the explicit
        recompute policy discards the completed turn.
        """
        if not self.config.enabled or req.session_id is None:
            return

        # Completion callbacks run before the main loop's ordinary manager
        # advance at this timestamp. Commit older migration/release events
        # first so capacity projection never treats a past, still-accounted
        # source as already free.
        self.advance(completion_time_ns)
        self.end_session(
            req.session_id, now_ns=completion_time_ns, keep_durable=True)
        scheduler = self._scheduler(req.instance_id)
        tokens = max(0, int(req.num_computed_tokens))
        blocks = (tokens + self.config.block_size - 1) // self.config.block_size
        block_tokens = blocks * self.config.block_size
        per_rank_bytes = scheduler.memory.get_kv(block_tokens)
        total_bytes = per_rank_bytes * scheduler.num_npus
        released_per_rank_bytes = (
            req.agentic_kv_completion_released_per_rank_bytes)
        if released_per_rank_bytes is None:
            raise RuntimeError(
                "Completed request lacks its scheduler KV-release audit: "
                f"session={req.session_id}")
        if int(released_per_rank_bytes) != per_rank_bytes:
            raise RuntimeError(
                "Completed request KV-release size does not match the idle "
                "handoff: "
                f"session={req.session_id}, released="
                f"{released_per_rank_bytes}, retained={per_rank_bytes}")
        req.agentic_kv_completion_released_per_rank_bytes = None
        entry = IdleKVEntry(
            session_id=req.session_id,
            instance_id=req.instance_id,
            tokens=tokens,
            block_tokens=block_tokens,
            per_rank_bytes=per_rank_bytes,
            total_bytes=total_bytes,
            location=KVLocation.HBM,
            tier_since_ns=completion_time_ns,
            last_access_ns=completion_time_ns,
            incremental_base_tokens=max(
                0, int(getattr(req, "prefix_reuse_tokens", 0) or 0)),
            next_use_ns=release_time_ns,
        )
        gap_type = str(return_gap_type or "unknown").strip().lower()
        if gap_type not in {"human", "tool", "mixed", "unknown"}:
            gap_type = "unknown"
        self.metrics.tool_pauses += 1
        self.metrics.idle_pauses += 1
        pause_counter = {
            "tool": "tool_return_pauses",
            "human": "human_return_pauses",
            "mixed": "mixed_return_pauses",
            "unknown": "unknown_return_pauses",
        }[gap_type]
        setattr(
            self.metrics,
            pause_counter,
            getattr(self.metrics, pause_counter) + 1,
        )

        if self.config.policy == "recompute" or tokens == 0:
            entry.location = KVLocation.DROPPED
            entry.per_rank_bytes = 0
            entry.total_bytes = 0
        else:
            available_per_rank = self._hbm_avail(scheduler)
            if available_per_rank < per_rank_bytes:
                raise RuntimeError(
                    "Completed active KV cannot be atomically handed to the "
                    "idle tier manager: "
                    f"session={req.session_id}, needed={per_rank_bytes}, "
                    f"available={available_per_rank}. Completion ownership "
                    "must precede tied HBM admissions.")
            # Future HBM reservations remain logically protected, but do not
            # subtract them here: their admission was backed by another source
            # release. These bytes are the exact active allocation freed by the
            # completion immediately before this callback.
            scheduler.memory.allocate(per_rank_bytes, Device.NPU)
            self.events.append({
                "time_ns": completion_time_ns,
                "session_id": req.session_id,
                "event": "completed_kv_idle_handoff",
                "per_rank_bytes": per_rank_bytes,
                "logically_reserved_per_rank_bytes": (
                    self._hbm_logically_reserved(req.instance_id)),
            })

        self.entries[req.session_id] = entry
        self._mark_hbm_admission_state_changed()
        self._update_idle_peaks()
        self.events.append({
            "time_ns": completion_time_ns,
            "session_id": req.session_id,
            "event": "tool_pause",
            "event_scope": "generic_inter_turn_idle_gap",
            "return_gap_type": gap_type,
            "return_gap_source": str(return_gap_source or "unknown"),
            "return_gap_ns": max(0, release_time_ns - completion_time_ns),
            "location": entry.location.value,
            "tokens": tokens,
            "bytes": total_bytes,
        })

    def advance(self, now_ns: int) -> None:
        """Advance background demotions up to ``now_ns``.

        Capacity-triggered background copies are non-preemptive and commit
        atomically. A known next-use timestamp affects only optional TTL
        sensitivity; a returning owner otherwise joins an in-flight demotion.
        """
        now_ns = int(now_ns)
        if now_ns < self._logical_frontier_ns:
            raise RuntimeError(
                "Agentic KV logical time regressed: "
                f"requested={now_ns}, frontier={self._logical_frontier_ns}. "
                "Route due continuations before later completion callbacks.")
        self._logical_frontier_ns = now_ns
        for node_id, reservations in list(
                self._transient_dram_reservations.items()):
            live = [
                reservation for reservation in reservations
                if reservation.complete_ns > now_ns
            ]
            if live:
                self._transient_dram_reservations[node_id] = live
            else:
                del self._transient_dram_reservations[node_id]
        if not self.config.enabled:
            return
        # A logical-time fast-forward can cross many independent migration
        # events at once. Process them globally by timestamp so capacity freed
        # at t=100 cannot be consumed by an event that actually occurred at
        # t=1. Deterministic ties process source release, durable-record TTL,
        # then tier migration and future HBM admission, with session ID
        # breaking ties within each kind.
        queue = []
        sequence = 0
        sync_pinned_sessions = self._capacity_pinned_sessions()
        for pending in self.pending_source_releases:
            if pending.ready_ns <= now_ns:
                heapq.heappush(queue, (
                    pending.ready_ns, 0, pending.entry.session_id,
                    sequence, "release", pending))
                sequence += 1
        for session_id, record in self.ssd_records.items():
            if session_id in sync_pinned_sessions:
                continue
            if (self.config.policy == "hbm_ssd_direct"
                    or not self.config.age_demotion_enabled):
                # Capacity-only baselines retain durable records until
                # explicit session teardown or SSD LRU pressure.
                continue
            entry = self.entries.get(session_id)
            if entry is not None and entry.location == KVLocation.SSD:
                # The live idle entry owns this record and schedules its own
                # SSD TTL transition below. Independent record expiry is for
                # keep-on-read shadow copies while the session is active or
                # its newer idle copy resides in HBM/CPU.
                continue
            event_ns = self._ssd_record_expiry_ns(session_id, record)
            if event_ns <= now_ns:
                heapq.heappush(queue, (
                    event_ns, 1, session_id,
                    sequence, "record_expiry", record))
                sequence += 1
        for entry in self.entries.values():
            if (entry.session_id in sync_pinned_sessions
                    and entry.migration_kind is None):
                continue
            event_ns = self._next_entry_event_ns(entry)
            if event_ns is not None and event_ns <= now_ns:
                heapq.heappush(queue, (
                    event_ns, 2, entry.session_id,
                    sequence, "entry", entry))
                sequence += 1
        for pending in self.pending_hbm_allocations:
            if pending.ready_ns <= now_ns:
                heapq.heappush(queue, (
                    pending.ready_ns, 3, pending.entry.session_id,
                    sequence, "hbm_allocation", pending))
                sequence += 1

        while queue:
            event_ns, _, event_session_id, _, kind, payload = heapq.heappop(queue)
            if kind == "release":
                pending = payload
                if pending not in self.pending_source_releases:
                    continue
                self.pending_source_releases.remove(pending)
                self._release_restore_source(pending)
                continue

            if kind == "record_expiry":
                record = payload
                if self.ssd_records.get(event_session_id) is not record:
                    continue
                expected_ns = self._ssd_record_expiry_ns(
                    event_session_id, record)
                if expected_ns != event_ns:
                    if expected_ns <= now_ns:
                        heapq.heappush(queue, (
                            expected_ns, 1, event_session_id,
                            sequence, "record_expiry", record))
                        sequence += 1
                    continue
                self._remove_ssd_record(event_session_id, event_ns)
                self.metrics.ttl_drops += 1
                self.events.append({
                    "time_ns": event_ns,
                    "session_id": event_session_id,
                    "event": "ssd_record_ttl",
                    "bytes": record.bytes,
                })
                continue

            if kind == "hbm_allocation":
                pending = payload
                if pending not in self.pending_hbm_allocations:
                    continue
                scheduler = self._scheduler(pending.entry.instance_id)
                if self._hbm_avail(scheduler) < pending.entry.per_rank_bytes:
                    raise RuntimeError(
                        "Future HBM reservation was oversubscribed at commit: "
                        f"session={pending.entry.session_id}, "
                        f"needed={pending.entry.per_rank_bytes}, "
                        f"available={self._hbm_avail(scheduler)}")
                scheduler.memory.allocate(
                    pending.entry.per_rank_bytes, Device.NPU)
                self.pending_hbm_allocations.remove(pending)
                self._mark_hbm_admission_state_changed()
                self.events.append({
                    "time_ns": event_ns,
                    "session_id": pending.entry.session_id,
                    "event": "hbm_capacity_reservation_commit",
                    "bytes": pending.entry.total_bytes,
                })
                self._update_idle_peaks()
                continue

            entry = payload
            if self.entries.get(entry.session_id) is not entry:
                continue
            if (self.synchronous_swap_enabled
                    and entry.location == KVLocation.HBM
                    and entry.migration_kind is None
                    and self.config.age_demotion_enabled
                    and self._scheduler(entry.instance_id).inflight):
                # Retry on the next main-loop event. Do not reinsert the same
                # expired timestamp into this advance() call: that would spin
                # forever while the source iteration is still in flight.
                self._sync_deferred_hbm_demotions.add(entry.session_id)
                continue
            expected_ns = self._next_entry_event_ns(entry)
            if expected_ns != event_ns:
                if expected_ns is not None and expected_ns <= now_ns:
                    heapq.heappush(queue, (
                        expected_ns, 2, entry.session_id,
                        sequence, "entry", entry))
                    sequence += 1
                continue
            if entry.session_id in self._sync_deferred_hbm_demotions:
                self._sync_deferred_hbm_demotions.remove(entry.session_id)
                self._advance_entry(entry, int(now_ns))
            else:
                self._advance_entry(entry, event_ns)
            # Capacity cascading can schedule a different CPU victim while
            # advancing this HBM entry. Discover every newly-created event;
            # stale duplicates are rejected by the expected timestamp check.
            for candidate in self.entries.values():
                if (candidate.session_id
                        in self._sync_deferred_hbm_demotions
                        or (candidate.session_id in sync_pinned_sessions
                            and candidate.migration_kind is None)):
                    continue
                next_ns = self._next_entry_event_ns(candidate)
                if next_ns is not None and next_ns <= now_ns:
                    heapq.heappush(queue, (
                        next_ns, 2, candidate.session_id,
                        sequence, "entry", candidate))
                    sequence += 1

    def _ssd_record_expiry_ns(
            self, session_id: str, record: SSDRecord) -> int:
        """Return durable-record TTL, postponed through an in-flight read."""
        expiry_ns = record.last_access_ns + self.config.ssd_ttl_ns
        pending_until = max(
            (
                pending.ready_ns
                for pending in self.pending_source_releases
                if pending.entry.session_id == session_id
                and pending.entry.location == KVLocation.SSD
            ),
            default=expiry_ns,
        )
        return max(expiry_ns, record.pinned_until_ns, pending_until)

    def _release_restore_source(self, pending: PendingSourceRelease) -> None:
        entry = pending.entry
        self._account_residence(entry, pending.ready_ns)
        if entry.location == KVLocation.CPU and entry.total_bytes:
            self._scheduler(entry.instance_id).memory.free(
                entry.total_bytes, Device.CPU)
        elif entry.location == KVLocation.HBM and entry.per_rank_bytes:
            self._scheduler(entry.instance_id).memory.free(
                entry.per_rank_bytes, Device.NPU)
        if pending.remove_ssd_record:
            self._remove_ssd_record(entry.session_id, pending.ready_ns)
        self.events.append({
            "time_ns": pending.ready_ns,
            "session_id": entry.session_id,
            "event": "restore_source_release",
            "source": entry.location.value,
        })
        self._update_idle_peaks()

    def _next_entry_event_ns(self, entry: IdleKVEntry) -> Optional[int]:
        if entry.location == KVLocation.DROPPED:
            return None
        policy = self.config.policy
        # Capacity pressure can schedule a migration even when age-based
        # demotion is disabled. Such migrations must always be allowed to
        # commit before considering whether a new TTL event may be created.
        if entry.migration_kind is not None:
            return entry.migration_complete_ns
        if not self.config.age_demotion_enabled:
            return None
        if entry.location in {KVLocation.HBM, KVLocation.CPU}:
            if policy in {"preserve", "hbm_lru_recompute"}:
                return None
            if entry.location == KVLocation.HBM and policy in {
                    "cpu", "tiered", "tiered_queue_recompute"}:
                ttl_ns = 0 if policy == "cpu" else self.config.hbm_ttl_ns
                return entry.tier_since_ns + ttl_ns
            if (entry.location == KVLocation.CPU
                    and self.config.tiered_family):
                return entry.tier_since_ns + self.config.cpu_ttl_ns
        if entry.location == KVLocation.SSD:
            if policy == "hbm_ssd_direct":
                return None
            event_ns = entry.tier_since_ns + self.config.ssd_ttl_ns
            if (entry.next_use_ns is not None
                    and event_ns > entry.next_use_ns):
                return None
            return event_ns
        return None

    def _advance_entry(self, entry: IdleKVEntry, now_ns: int) -> None:
        if entry.location == KVLocation.DROPPED:
            return
        policy = self.config.policy
        if policy in {"preserve", "hbm_lru_recompute"}:
            return

        if entry.location == KVLocation.HBM:
            if entry.migration_kind is None:
                if not self.config.age_demotion_enabled:
                    return
                if not self._schedule_hbm_demotion(
                        entry, now_ns, reason="ttl"):
                    self._drop_entry(entry, now_ns, "cpu_capacity")
                    return
            complete = entry.migration_complete_ns
            if (entry.migration_kind in {
                    "hbm_to_ssd", "hbm_to_ssd_direct"}
                    and complete is not None and now_ns >= complete):
                reason = entry.migration_reason or "ttl"
                self._clear_entry_migration(entry)
                self._move_hbm_to_ssd(entry, complete, reason)
                return
            if (entry.migration_kind == "hbm_to_cpu"
                    and complete is not None and now_ns >= complete):
                reason = entry.migration_reason or "ttl"
                self._clear_entry_migration(entry)
                scheduler = self._scheduler(entry.instance_id)
                if self._cpu_avail(scheduler) >= entry.total_bytes:
                    self._account_residence(entry, complete)
                    scheduler.memory.allocate(entry.total_bytes, Device.CPU)
                    scheduler.memory.free(entry.per_rank_bytes, Device.NPU)
                    entry.location = KVLocation.CPU
                    entry.tier_since_ns = complete
                    entry.last_access_ns = complete
                    self.metrics.hbm_to_cpu_bytes += entry.total_bytes
                    if reason.startswith("hbm_capacity"):
                        self.metrics.hbm_capacity_demotions += 1
                    self.events.append({"time_ns": complete, "session_id": entry.session_id,
                                        "event": "hbm_to_cpu", "bytes": entry.total_bytes,
                                        "reason": reason})
                    self._update_idle_peaks()
                else:
                    # A forecast can be invalidated by a same-timestamp source
                    # pin. Retry through the capacity cascade; never silently
                    # turn the ordinary path into a newcomer-only bypass.
                    if (not self.config.tiered_family
                            or not self._schedule_hbm_demotion(
                                entry, complete, reason=reason)):
                        self._drop_entry(entry, complete, "cpu_capacity")
                    return

        if (entry.location == KVLocation.CPU
                and self.config.tiered_family):
            cpu_expiry_ns = entry.tier_since_ns + self.config.cpu_ttl_ns
            if (self.config.age_demotion_enabled
                    and entry.migration_kind is None
                    and now_ns >= cpu_expiry_ns):
                self._schedule_entry_migration(
                    entry, "cpu_to_ssd", now_ns, reason="ttl")
            complete = entry.migration_complete_ns
            if (entry.migration_kind == "cpu_to_ssd"
                    and complete is not None and now_ns >= complete):
                reason = entry.migration_reason or "ttl"
                self._clear_entry_migration(entry)
                self._move_cpu_to_ssd(entry, complete, reason)

        if entry.location == KVLocation.SSD:
            if (policy == "hbm_ssd_direct"
                    or not self.config.age_demotion_enabled):
                return
            expiry = entry.tier_since_ns + self.config.ssd_ttl_ns
            if now_ns >= expiry:
                self._drop_entry(entry, expiry, "ssd_ttl")
                self.metrics.ttl_drops += 1

    def _ssd_write_bytes(self, entry: IdleKVEntry) -> int:
        previous = self.ssd_records.get(entry.session_id)
        if (self.config.ssd_write_mode == "incremental"
                and previous is not None
                and entry.tokens >= previous.tokens
                and entry.incremental_base_tokens >= previous.tokens):
            # Token-granular append is the optimistic endurance lower bound.
            # It still charges appends that fit inside an already allocated
            # cache block; a divergent/partial prefix forces a full rewrite.
            bytes_per_token = (
                entry.total_bytes // entry.block_tokens
                if entry.block_tokens else 0)
            return (entry.tokens - previous.tokens) * bytes_per_token
        return entry.total_bytes

    def _commit_ssd_record(self, entry: IdleKVEntry, now_ns: int) -> bool:
        node_id = int(self._node_id(self._scheduler(entry.instance_id)))
        previous = self.ssd_records.get(entry.session_id)
        if previous is not None and int(previous.node_id) != node_id:
            raise RuntimeError(
                "Cross-node SSD lineage replacement is not modeled: "
                f"session={entry.session_id}, source_node={previous.node_id}, "
                f"target_node={node_id}")
        old_bytes = previous.bytes if previous is not None else 0
        extra = max(0, entry.total_bytes - old_bytes)
        if not self._ensure_ssd_capacity(
                entry.session_id, extra, now_ns, node_id=node_id):
            return False
        if previous is not None:
            self._account_ssd_record(previous, now_ns)
        self.ssd_used_bytes += entry.total_bytes - old_bytes
        self.ssd_records[entry.session_id] = SSDRecord(
            tokens=entry.tokens,
            block_tokens=entry.block_tokens,
            bytes=entry.total_bytes,
            last_access_ns=now_ns,
            accounted_until_ns=now_ns,
            node_id=node_id,
        )
        self.metrics.peak_ssd_used_bytes = max(
            self.metrics.peak_ssd_used_bytes, self.ssd_used_bytes)
        self.metrics.peak_ssd_committed_reserved_bytes = max(
            self.metrics.peak_ssd_committed_reserved_bytes,
            self.ssd_used_bytes + self._ssd_reserved_bytes(),
        )
        return True

    def _move_hbm_to_ssd(
            self, entry: IdleKVEntry, complete: int, reason: str) -> None:
        direct = self.config.policy == "hbm_ssd_direct"
        if reason.startswith("hbm_capacity"):
            self.metrics.hbm_capacity_demotions += 1
        if "cpu_bypass" in reason:
            self.metrics.cpu_capacity_bypasses += 1
        if direct:
            # Releasing and committing are one logical event; no other event
            # can consume the reserved bytes between these operations.
            self._release_direct_ssd_capacity(entry, complete)
        if not self._commit_ssd_record(entry, complete):
            self._drop_entry(entry, complete, "ssd_capacity")
            return
        self._account_residence(entry, complete)
        self._scheduler(entry.instance_id).memory.free(entry.per_rank_bytes, Device.NPU)
        entry.location = KVLocation.SSD
        entry.hbm_ssd_start_ns = None
        entry.tier_since_ns = complete
        entry.last_access_ns = complete
        self.metrics.hbm_to_ssd_bytes += entry.total_bytes
        self.metrics.ssd_demotion_completions += 1
        self._update_idle_peaks()
        self.events.append({"time_ns": complete, "session_id": entry.session_id,
                            "event": (
                                "hbm_to_ssd_direct" if direct else "hbm_to_ssd"),
                            "bytes": entry.total_bytes,
                            "reason": reason})

    def _move_cpu_to_ssd(
            self, entry: IdleKVEntry, complete_ns: int, reason: str) -> None:
        if reason in {"cpu_capacity", "transient_dram_capacity"}:
            self.metrics.cpu_capacity_evictions += 1
        if reason == "transient_dram_capacity":
            self.metrics.transient_dram_cpu_lru_evictions += 1
        if not self._commit_ssd_record(entry, complete_ns):
            self._drop_entry(entry, complete_ns, "ssd_capacity")
            return
        self._account_residence(entry, complete_ns)
        self._scheduler(entry.instance_id).memory.free(entry.total_bytes, Device.CPU)
        entry.location = KVLocation.SSD
        entry.tier_since_ns = complete_ns
        entry.last_access_ns = complete_ns
        self.metrics.cpu_to_ssd_bytes += entry.total_bytes
        self.metrics.ssd_demotion_completions += 1
        self._update_idle_peaks()
        self.events.append({"time_ns": complete_ns, "session_id": entry.session_id,
                            "event": "cpu_to_ssd", "bytes": entry.total_bytes,
                            "reason": reason})

    def _drop_entry(
            self, entry: IdleKVEntry, now_ns: int, reason: str,
            keep_ssd_record: bool = False) -> None:
        try:
            drop_class = _KV_DROP_CLASS_BY_REASON[str(reason)]
        except KeyError as exc:
            raise RuntimeError(
                f"Unknown KV-cache entry drop reason {reason!r}") from exc
        was_dropped = entry.location == KVLocation.DROPPED
        self._sync_deferred_hbm_demotions.discard(entry.session_id)
        self._release_direct_ssd_capacity(entry, now_ns)
        self._account_residence(entry, now_ns)
        scheduler = self._scheduler(entry.instance_id)
        if entry.location == KVLocation.HBM and entry.per_rank_bytes:
            scheduler.memory.free(entry.per_rank_bytes, Device.NPU)
        elif entry.location == KVLocation.CPU and entry.total_bytes:
            scheduler.memory.free(entry.total_bytes, Device.CPU)
        elif entry.location == KVLocation.SSD and not keep_ssd_record:
            self._remove_ssd_record(entry.session_id, now_ns)
        if not keep_ssd_record and entry.location != KVLocation.SSD:
            # A keep-on-read durable snapshot must not occupy SSD capacity
            # after the newer HBM/CPU copy has been discarded.
            self._remove_ssd_record(entry.session_id, now_ns)
        entry.location = KVLocation.DROPPED
        entry.drop_reason = reason
        entry.hbm_ssd_start_ns = None
        self._clear_entry_migration(entry)
        entry.per_rank_bytes = 0
        entry.total_bytes = 0
        self._mark_hbm_admission_state_changed()
        if not was_dropped and "capacity" in reason:
            self.metrics.capacity_drops += 1
            if reason == "hbm_capacity":
                self.metrics.hbm_capacity_drops += 1
            elif reason == "ssd_capacity":
                self.metrics.ssd_capacity_admission_drops += 1
        self.events.append({
            "time_ns": now_ns,
            "session_id": entry.session_id,
            "event": "drop",
            "reason": reason,
            "object_scope": "kv_cache_entry",
            "logical_session_effect": "none",
            "drop_class": drop_class,
        })
        self._update_idle_peaks()

    @staticmethod
    def _foreground_restore_breakdown(
            reservation: TransferReservation,
            release_time_ns: int,
            pd_pair_fifo_wait_ns: int = 0,
            prepare_boundary_wait_ns: int = 0,
            source_demotion_join_wait_ns: int = 0,
            transient_dram_capacity_wait_ns: int = 0,
            ) -> tuple[int, int, int, int]:
        return AgenticKVManager._foreground_restore_chain_breakdown(
            (reservation,), release_time_ns, pd_pair_fifo_wait_ns,
            prepare_boundary_wait_ns, source_demotion_join_wait_ns,
            transient_dram_capacity_wait_ns)

    @staticmethod
    def _foreground_restore_chain_breakdown(
            reservations: Sequence[TransferReservation],
            release_time_ns: int,
            pd_pair_fifo_wait_ns: int = 0,
            prepare_boundary_wait_ns: int = 0,
            source_demotion_join_wait_ns: int = 0,
            transient_dram_capacity_wait_ns: int = 0,
            ) -> tuple[int, int, int, int]:
        """Return physical restore, HBM-admission, queue, and service time.

        Foreground transfer arrival is delayed until every HBM allocation and
        SSD bounce-buffer admission needed by the restore is ready. A strict
        P/D pair can hold
        a continuation in its FIFO, and an engine/scheduler boundary can then
        delay the first physical prepare attempt. Both waits precede the
        physical restore issue timestamp and are excluded from restore and
        destination-HBM admission. Multi-stage restores must hand each stage
        directly to the next one so no unaccounted gap appears between the
        SSD-to-DRAM and DRAM-to-P legs.
        """
        if not reservations:
            raise ValueError("Foreground restore requires at least one transfer")
        for previous, current in zip(reservations, reservations[1:]):
            if int(current.arrival_ns) != int(previous.complete_ns):
                raise RuntimeError(
                    "Foreground restore stages are not contiguous: "
                    f"previous_complete={previous.complete_ns}, "
                    f"next_arrival={current.arrival_ns}")
        first = reservations[0]
        last = reservations[-1]
        pd_pair_fifo_wait_ns = int(pd_pair_fifo_wait_ns)
        if pd_pair_fifo_wait_ns < 0:
            raise ValueError("P/D pair FIFO wait must be non-negative")
        prepare_boundary_wait_ns = int(prepare_boundary_wait_ns)
        if prepare_boundary_wait_ns < 0:
            raise ValueError("Prepare-boundary wait must be non-negative")
        source_demotion_join_wait_ns = int(source_demotion_join_wait_ns)
        if source_demotion_join_wait_ns < 0:
            raise ValueError("Source-demotion join wait must be non-negative")
        restore_issue_time_ns = (
            int(release_time_ns) + pd_pair_fifo_wait_ns
            + prepare_boundary_wait_ns + source_demotion_join_wait_ns)
        gross_admission_wait_ns = (
            int(first.arrival_ns) - restore_issue_time_ns)
        if gross_admission_wait_ns < 0:
            raise RuntimeError(
                "Pre-restore admission waits exceed the foreground "
                "restore's release-to-transfer admission interval: "
                f"arrival={first.arrival_ns}, "
                f"release={release_time_ns}, "
                f"pd_pair_fifo_wait={pd_pair_fifo_wait_ns}, "
                f"prepare_boundary_wait={prepare_boundary_wait_ns}, "
                f"source_demotion_join_wait="
                f"{source_demotion_join_wait_ns}")
        transient_dram_capacity_wait_ns = int(
            transient_dram_capacity_wait_ns)
        if (transient_dram_capacity_wait_ns < 0
                or transient_dram_capacity_wait_ns
                > gross_admission_wait_ns):
            raise RuntimeError(
                "Transient DRAM capacity wait is not a subset of the "
                "pre-transfer admission interval: "
                f"transient={transient_dram_capacity_wait_ns}, "
                f"gross_admission={gross_admission_wait_ns}")
        # HBM admission ends when destination bytes are reserved. A later wait
        # for a full-object SSD bounce buffer is lower-tier admission/queueing,
        # not HBM pressure. Keep it as an explicit subset of queue wait so the
        # four additive owner-gate components remain disjoint.
        hbm_admission_wait_ns = (
            gross_admission_wait_ns - transient_dram_capacity_wait_ns)
        restore_ns = int(last.complete_ns) - restore_issue_time_ns
        queue_wait_ns = (
            transient_dram_capacity_wait_ns
            + sum(int(reservation.queue_wait_ns)
                  for reservation in reservations)
        )
        service_ns = sum(
            int(reservation.service_ns) for reservation in reservations)
        accounted_ns = (
            hbm_admission_wait_ns
            + queue_wait_ns
            + service_ns
        )
        if restore_ns != accounted_ns:
            raise RuntimeError(
                "Foreground restore accounting does not reconcile: "
                f"physical_restore={restore_ns}, "
                f"pd_pair_fifo={pd_pair_fifo_wait_ns}, "
                f"prepare_boundary={prepare_boundary_wait_ns}, "
                f"source_demotion_join={source_demotion_join_wait_ns}, "
                f"hbm_admission={hbm_admission_wait_ns}, "
                f"queue={queue_wait_ns}, service={service_ns}")
        return (
            restore_ns,
            hbm_admission_wait_ns,
            queue_wait_ns,
            service_ns,
        )

    def _record_critical_restore_accounting(
            self, *, pd_pair_fifo_wait_ns: int,
            prepare_boundary_wait_ns: int,
            source_demotion_join_wait_ns: int,
            hbm_admission_wait_ns: int, queue_wait_ns: int,
            service_ns: int, expected_total_ns: int) -> None:
        """Accumulate one successful continuation's exact owner-gate split."""
        components = {
            "pd_pair_fifo": int(pd_pair_fifo_wait_ns),
            "prepare_boundary": int(prepare_boundary_wait_ns),
            "source_demotion_join": int(source_demotion_join_wait_ns),
            "hbm_admission": int(hbm_admission_wait_ns),
            "queue": int(queue_wait_ns),
            "service": int(service_ns),
        }
        if any(value < 0 for value in components.values()):
            raise RuntimeError(
                "Critical restore accounting contains a negative component: "
                f"{components}")
        physical_components_ns = (
            components["hbm_admission"]
            + components["queue"]
            + components["service"]
        )
        if int(expected_total_ns) != physical_components_ns:
            raise RuntimeError(
                "Critical restore accounting does not reconcile: "
                f"physical_restore={expected_total_ns}, "
                f"components={components}")
        self.metrics.critical_restore_ns += physical_components_ns
        self.metrics.pd_pair_fifo_admissions += 1
        self.metrics.pd_pair_fifo_wait_ns += components["pd_pair_fifo"]
        if components["pd_pair_fifo"]:
            self.metrics.pd_pair_fifo_waiting_admissions += 1
        self.metrics.prepare_boundary_admissions += 1
        self.metrics.prepare_boundary_wait_ns += components[
            "prepare_boundary"]
        if components["prepare_boundary"]:
            self.metrics.prepare_boundary_waiting_admissions += 1
        self.metrics.source_demotion_join_admissions += 1
        self.metrics.source_demotion_join_wait_ns += components[
            "source_demotion_join"]
        if components["source_demotion_join"]:
            self.metrics.source_demotion_join_waiting_admissions += 1
        self.metrics.critical_restore_hbm_admission_wait_ns += (
            components["hbm_admission"])
        # Analytical reservations account queue/service when they are issued;
        # external ASTRA callbacks do so when they complete. Do not add those
        # components again here.
        aggregate_components_ns = (
            self.metrics.critical_restore_hbm_admission_wait_ns
            + self.metrics.critical_restore_queue_wait_ns
            + self.metrics.critical_restore_service_ns
        )
        if self.metrics.critical_restore_ns != aggregate_components_ns:
            raise RuntimeError(
                "Aggregate foreground restore accounting does not reconcile: "
                f"physical_restore={self.metrics.critical_restore_ns}, "
                "hbm_admission="
                f"{self.metrics.critical_restore_hbm_admission_wait_ns}, "
                f"queue={self.metrics.critical_restore_queue_wait_ns}, "
                f"service={self.metrics.critical_restore_service_ns}")

    def snapshot_return_residency(
            self, session_id: str, return_time_ns: int) -> KVLocation:
        """Observe one continuation's tier at its exact request-ready event.

        This observation deliberately acquires no prepare lock and pins no
        cache object. A request waiting for P/D admission can therefore lose
        or demote its cache under the ordinary LRU policy, while reports still
        distinguish the original return residency from the eventual service
        source. The online event path must call this before advancing past the
        return timestamp; silently reconstructing historical residency from a
        later mutable entry would be incorrect.
        """
        return_time_ns = int(return_time_ns)
        if return_time_ns < self._logical_frontier_ns:
            raise RuntimeError(
                "Return-residency snapshot arrived after the tier timeline "
                f"advanced: session={session_id}, return={return_time_ns}, "
                f"frontier={self._logical_frontier_ns}")
        self.advance(return_time_ns)
        entry = self.entries.get(str(session_id))
        residency = (
            KVLocation.DROPPED if entry is None else entry.location)
        self.events.append({
            "time_ns": return_time_ns,
            "session_id": str(session_id),
            "event": "return_residency_snapshot",
            "residency_at_return": residency.value,
            "non_pinning": True,
        })
        return residency

    def prepare_request(
            self, session_id: str, instance_id: int, reuse_tokens: int,
            input_tokens: int, release_time_ns: int,
            return_gap_type: str = "unknown",
            return_gap_source: str = "unknown",
            return_gap_ns: int = 0,
            operation_time_ns: Optional[int] = None,
            defer_temporary_hbm_pressure: bool = False,
            residency_at_return: Optional[KVLocation | str] = None,
            pd_decode_instance_id: Optional[int] = None,
            pd_pair_fifo_wait_ns: int = 0,
            prepare_boundary_wait_ns: Optional[int] = None,
            request_id: Optional[int] = None,
            sub_request_index: Optional[int] = None,
            ) -> Optional[KVPreparation]:
        """Restore the longest reusable prefix and transfer ownership back.

        ``release_time_ns`` is the request-ready event used for latency
        accounting. ``operation_time_ns`` is when a previously capacity-
        blocked preparation is retried. ``residency_at_return`` is an exact,
        non-pinning snapshot taken when the request first became ready; the
        eventual restore source may differ after an admission wait. Returning
        ``None`` leaves the source pinned and asks the router to retry after
        another completion event. ``pd_pair_fifo_wait_ns`` is the exact subset
        of release-to-operation delay caused by strict P/D pair ordering.
        ``prepare_boundary_wait_ns`` is the subsequent non-I/O admission wait
        before the first physical restore attempt. Both precede
        ``restore_issue_time_ns`` and contribute to the owner-ready gate, but
        never to physical restore or HBM-capacity overhead. When the latter is
        omitted by a direct caller, every non-pair pre-operation delay is
        conservatively classified as prepare-boundary wait; the online router
        freezes it at the first manager attempt so later capacity retries are
        instead charged to HBM admission. ``pd_decode_instance_id`` identifies
        the fixed D peer used only for a causal partial-prefix capacity
        snapshot; the manager does not reserve its reported headroom.
        """
        # KV does not contain the final prompt token's hidden state/logits.
        # Match vLLM-style full-prefix handling by always executing at least
        # one prompt token unless the request itself is empty.
        declared_reuse = max(0, min(int(reuse_tokens), int(input_tokens)))
        requested_reuse = min(declared_reuse, max(0, int(input_tokens) - 1))
        release_time_ns = int(release_time_ns)
        operation_time_ns = int(
            release_time_ns
            if operation_time_ns is None else operation_time_ns)
        if operation_time_ns < release_time_ns:
            raise ValueError(
                "KV prepare operation cannot precede request release: "
                f"operation={operation_time_ns}, release={release_time_ns}")
        pd_pair_fifo_wait_ns = int(pd_pair_fifo_wait_ns)
        if pd_pair_fifo_wait_ns < 0:
            raise ValueError("pd_pair_fifo_wait_ns must be non-negative")
        if pd_pair_fifo_wait_ns > operation_time_ns - release_time_ns:
            raise ValueError(
                "P/D pair FIFO wait exceeds release-to-operation delay: "
                f"wait={pd_pair_fifo_wait_ns}, operation={operation_time_ns}, "
                f"release={release_time_ns}")
        known_external_restore = self._external_fabric_by_session.get(
            session_id)
        if prepare_boundary_wait_ns is None:
            prepare_boundary_wait_ns = (
                known_external_restore.prepare_boundary_wait_ns
                if known_external_restore is not None else
                operation_time_ns - release_time_ns - pd_pair_fifo_wait_ns
            )
        prepare_boundary_wait_ns = int(prepare_boundary_wait_ns)
        if prepare_boundary_wait_ns < 0:
            raise ValueError(
                "prepare_boundary_wait_ns must be non-negative")
        if (pd_pair_fifo_wait_ns + prepare_boundary_wait_ns
                > operation_time_ns - release_time_ns):
            raise ValueError(
                "Pre-restore admission waits exceed release-to-operation "
                f"delay: pair={pd_pair_fifo_wait_ns}, boundary="
                f"{prepare_boundary_wait_ns}, operation={operation_time_ns}, "
                f"release={release_time_ns}")
        self.advance(operation_time_ns)
        if residency_at_return is not None:
            try:
                return_residency = (
                    residency_at_return
                    if isinstance(residency_at_return, KVLocation)
                    else KVLocation(str(residency_at_return))
                )
            except ValueError as exc:
                raise ValueError(
                    "residency_at_return must be one of "
                    f"{[location.value for location in KVLocation]}, got "
                    f"{residency_at_return!r}"
                ) from exc
        else:
            return_residency = None
        external_restore = known_external_restore
        if external_restore is not None:
            # Backward-compatible direct callers may omit the observation on
            # an idempotent completion retry. The originally issued job is
            # authoritative; an explicitly changed observation still fails
            # the invocation identity check below.
            if return_residency is None:
                return_residency = external_restore.residency_at_return
            invocation = (
                int(instance_id), release_time_ns, declared_reuse,
                requested_reuse, int(input_tokens),
                str(return_gap_type or "unknown"),
                str(return_gap_source or "unknown"), int(return_gap_ns),
                (
                    None if return_residency is None
                    else return_residency.value
                ),
                pd_pair_fifo_wait_ns,
                prepare_boundary_wait_ns,
            )
            expected_invocation = (
                external_restore.target_instance_id,
                external_restore.release_time_ns,
                external_restore.declared_reuse_tokens,
                external_restore.requested_reuse_tokens,
                external_restore.input_tokens,
                external_restore.return_gap_type,
                external_restore.return_gap_source,
                external_restore.return_gap_ns,
                external_restore.residency_at_return.value,
                external_restore.pd_pair_fifo_wait_ns,
                external_restore.prepare_boundary_wait_ns,
            )
            if invocation != expected_invocation:
                raise RuntimeError(
                    "Retry changed an in-flight external fabric restore: "
                    f"session={session_id}, expected={expected_invocation}, "
                    f"observed={invocation}")
            if external_restore.status != "completed":
                return None
        entry = self.entries.get(session_id)
        if return_residency is None:
            return_residency = (
                KVLocation.DROPPED if entry is None else entry.location)
        if entry is not None and entry.migration_kind is not None:
            migration_kind = str(entry.migration_kind)
            complete_ns = entry.migration_complete_ns
            if migration_kind.startswith("cancelled:"):
                # TTL sensitivity may cancel an opportunistic copy at return;
                # the original tier remains authoritative.
                if complete_ns is not None:
                    raise RuntimeError(
                        "Cancelled migration retained a completion time: "
                        f"session={session_id}, kind={migration_kind}")
                self._clear_entry_migration(entry)
            elif complete_ns is None:
                raise RuntimeError(
                    "Live migration has no completion event: "
                    f"session={session_id}, kind={migration_kind}")
            elif int(complete_ns) > operation_time_ns:
                first_wait = self._begin_demotion_join(
                    session_id, operation_time_ns, int(complete_ns),
                    migration_kind)
                if first_wait:
                    self.events.append({
                        "time_ns": operation_time_ns,
                        "session_id": session_id,
                        "event": "demotion_commit_join_deferred",
                        "migration_kind": migration_kind,
                        "migration_complete_ns": int(complete_ns),
                        "release_time_ns": release_time_ns,
                    })
                return None
        source_demotion_join_wait_ns = self._demotion_join_wait(
            session_id, operation_time_ns)
        destination_admission_start_ns = (
            release_time_ns + pd_pair_fifo_wait_ns
            + prepare_boundary_wait_ns + source_demotion_join_wait_ns)
        record = self.ssd_records.get(session_id)
        defer_lineage_invalidation = (
            entry is not None
            and entry.location == KVLocation.SSD
            and record is not None
            and declared_reuse < record.tokens
            and requested_reuse > 0
        )
        # Use the declared LCP rather than the input-1 execution cap: the
        # latter forces one prompt token through the model but does not make an
        # otherwise identical durable prefix invalid.
        if not defer_lineage_invalidation:
            self._truncate_ssd_lineage(
                session_id, declared_reuse, operation_time_ns)
        if entry is None:
            self._queue_recompute_restore_commitments.pop(
                str(session_id), None)
            committed_join_wait_ns = self._consume_demotion_join(
                session_id, operation_time_ns)
            if committed_join_wait_ns != source_demotion_join_wait_ns:
                raise RuntimeError(
                    "Final source-demotion join wait changed during missing-"
                    f"entry preparation: session={session_id}, observed="
                    f"{source_demotion_join_wait_ns}, committed="
                    f"{committed_join_wait_ns}")
            self._set_restore_capacity_pin(session_id, False)
            self._clear_destination_admission_wait(session_id)
            self._clear_transient_restore_wait(session_id)
            self.metrics.resumed_prompt_tokens += max(0, int(input_tokens))
            recompute = declared_reuse
            policy_avoidable_recompute = requested_reuse
            if recompute:
                self.metrics.dropped_misses += 1
                self.metrics.recompute_tokens += recompute
                self.metrics.policy_avoidable_recompute_tokens += (
                    policy_avoidable_recompute)
                if return_residency in {
                        KVLocation.CPU, KVLocation.SSD,
                        KVLocation.DROPPED}:
                    scheduler = self._scheduler(instance_id)
                    blocks = (
                        requested_reuse + self.config.block_size - 1
                    ) // self.config.block_size
                    opportunity_bytes = (
                        scheduler.memory.get_kv(
                            blocks * self.config.block_size)
                        * scheduler.num_npus
                    )
                    self.metrics.hbf_eligible_resumes += 1
                    self.metrics.hbf_eligible_restore_bytes += (
                        opportunity_bytes)
                    self.metrics.hbf_dropped_recompute_tokens += (
                        policy_avoidable_recompute)
            # No object exists to admit or transfer. Any delay before this
            # lookup is routing/prepare-boundary delay, never physical HBM or
            # migration time.
            prepare_boundary_wait_ns = (
                operation_time_ns - release_time_ns
                - pd_pair_fifo_wait_ns
                - source_demotion_join_wait_ns)
            if prepare_boundary_wait_ns < 0:
                raise RuntimeError(
                    "Source-demotion join exceeds missing-entry owner delay: "
                    f"session={session_id}")
            restore_hbm_admission_wait_ns = 0
            restore_ns = 0
            owner_gate_ns = (
                pd_pair_fifo_wait_ns + prepare_boundary_wait_ns
                + source_demotion_join_wait_ns)
            restore_issue_time_ns = release_time_ns + owner_gate_ns
            self._record_critical_restore_accounting(
                pd_pair_fifo_wait_ns=pd_pair_fifo_wait_ns,
                prepare_boundary_wait_ns=prepare_boundary_wait_ns,
                source_demotion_join_wait_ns=(
                    source_demotion_join_wait_ns),
                hbm_admission_wait_ns=restore_hbm_admission_wait_ns,
                queue_wait_ns=0,
                service_ns=0,
                expected_total_ns=restore_ns,
            )
            self.events.append({
                "time_ns": release_time_ns,
                "operation_time_ns": operation_time_ns,
                "session_id": session_id,
                "request_id": request_id,
                "sub_request_index": sub_request_index,
                "event": "resume",
                "source": KVLocation.DROPPED.value,
                "residency_at_return": return_residency.value,
                "source_instance_id": None,
                "target_instance_id": instance_id,
                "source_node_id": None,
                "target_node_id": self._node_id(
                    self._scheduler(instance_id)),
                "hit_tokens": 0,
                "recompute_tokens": recompute,
                "restore_ns": restore_ns,
                "owner_gate_ns": owner_gate_ns,
                "pd_pair_fifo_wait_ns": pd_pair_fifo_wait_ns,
                "prepare_boundary_wait_ns": prepare_boundary_wait_ns,
                "source_demotion_join_wait_ns": (
                    source_demotion_join_wait_ns),
                "restore_issue_time_ns": restore_issue_time_ns,
                "target_hbm_ready_time_ns": restore_issue_time_ns,
                "restore_ready_time_ns": restore_issue_time_ns,
                "hbm_admission_wait_ns": (
                    restore_hbm_admission_wait_ns),
                "transient_dram_capacity_wait_ns": 0,
                "queue_wait_ns": 0,
                "restore_service_ns": 0,
                "return_gap_type": str(return_gap_type or "unknown"),
                "return_gap_source": str(return_gap_source or "unknown"),
                "return_gap_ns": int(return_gap_ns),
            })
            return KVPreparation(
                0, recompute, KVLocation.DROPPED, restore_ns,
                release_time_ns + owner_gate_ns, 0,
                owner_gate_ns=owner_gate_ns,
                restore_issue_time_ns=restore_issue_time_ns,
                target_hbm_ready_time_ns=restore_issue_time_ns,
                restore_ready_time_ns=restore_issue_time_ns,
                pd_pair_fifo_wait_ns=pd_pair_fifo_wait_ns,
                prepare_boundary_wait_ns=prepare_boundary_wait_ns,
                source_demotion_join_wait_ns=(
                    source_demotion_join_wait_ns),
                hbm_admission_wait_ns=(
                    restore_hbm_admission_wait_ns),
                residency_at_return=return_residency,
            )

        source = entry.location
        initial_drop_reason = entry.drop_reason
        reusable = max(0, min(requested_reuse, entry.tokens))
        # A completed autoregressive request normally lacks KV for its final
        # sampled token.  Cap policy opportunity by the physical prefix that
        # could have been retained, not merely by the workload's logical LCP.
        policy_reusable = max(0, min(requested_reuse, entry.tokens))
        scheduler = self._scheduler(instance_id)
        hit_blocks = (reusable + self.config.block_size - 1) // self.config.block_size
        hit_block_tokens = hit_blocks * self.config.block_size
        hit_per_rank = scheduler.memory.get_kv(hit_block_tokens) if reusable else 0
        hit_total = hit_per_rank * scheduler.num_npus
        restore_hbm_admission_wait_ns = 0
        restore_ns = 0
        restore_queue_wait_ns = 0
        restore_service_ns = 0
        restore_transient_dram_capacity_wait_ns = 0
        retained_instance_id = None
        retained_per_rank_bytes = 0

        restore_allocation_failed = False
        restore_failure_tier = None
        queue_recompute_selected = False
        queue_recompute_partial = False
        source_release_deferred = False
        source_residence_accounted = False
        if (source == KVLocation.HBM and entry.instance_id == instance_id
                and reusable):
            if source_demotion_join_wait_ns:
                raise RuntimeError(
                    "A committed source demotion cannot resume as local HBM: "
                    f"session={session_id}")
            # A local hit has no physical restore. If manager invocation was
            # delayed for any non-pair reason, expose it only as the neutral
            # prepare-boundary component.
            prepare_boundary_wait_ns = (
                operation_time_ns - release_time_ns
                - pd_pair_fifo_wait_ns)
            surplus = max(0, entry.per_rank_bytes - hit_per_rank)
            if surplus:
                self._scheduler(entry.instance_id).memory.free(surplus, Device.NPU)
            self.metrics.hbm_hits += 1
        elif source == KVLocation.HBM and reusable:
            if source_demotion_join_wait_ns:
                raise RuntimeError(
                    "A committed source demotion cannot resume from peer HBM: "
                    f"session={session_id}")
            if external_restore is not None:
                candidate = external_restore.target_entry
                candidate_ready_ns = external_restore.arrival_time_ns
                if (candidate.instance_id != instance_id
                        or candidate.tokens != reusable
                        or candidate.block_tokens != hit_block_tokens
                        or candidate.per_rank_bytes != hit_per_rank
                        or candidate.total_bytes != hit_total):
                    raise RuntimeError(
                        "Completed external fabric target reservation no longer "
                        f"matches retry session={session_id}")
            else:
                candidate = IdleKVEntry(
                    session_id=session_id, instance_id=instance_id,
                    tokens=reusable, block_tokens=hit_block_tokens,
                    per_rank_bytes=hit_per_rank, total_bytes=hit_total,
                    location=KVLocation.HBM, tier_since_ns=operation_time_ns,
                    last_access_ns=operation_time_ns,
                )
                candidate_ready_ns = self._reserve_hbm(
                    candidate, operation_time_ns)
            if candidate_ready_ns is None:
                if (defer_temporary_hbm_pressure
                        and candidate.per_rank_bytes
                        <= self._hbm_kv_ceiling(scheduler)):
                    self._begin_destination_admission_wait(
                        session_id, destination_admission_start_ns,
                        operation_time_ns)
                    self._pause_transient_restore_wait(
                        session_id, operation_time_ns)
                    self._set_restore_capacity_pin(session_id, True)
                    self.events.append({
                        "time_ns": operation_time_ns,
                        "session_id": session_id,
                        "event": "hbm_restore_admission_deferred",
                        "release_time_ns": release_time_ns,
                        "target_instance_id": instance_id,
                        "per_rank_bytes": candidate.per_rank_bytes,
                        "source": source.value,
                    })
                    return None
                source = KVLocation.DROPPED
                reusable = 0
                hit_total = 0
                restore_allocation_failed = True
                restore_failure_tier = "hbm"
            else:
                source_scheduler = self._scheduler(entry.instance_id)
                source_hit_per_rank = source_scheduler.memory.get_kv(
                    hit_block_tokens)
                source_hit_total = (
                    source_hit_per_rank * source_scheduler.num_npus)
                wire_bytes = max(source_hit_total, hit_total)
                if source_hit_per_rank > entry.per_rank_bytes:
                    self._cancel_hbm_reservation(
                        candidate, candidate_ready_ns)
                    raise RuntimeError(
                        "P/D source KV layout is smaller than the reusable "
                        "prefix; model, TP, PP, block size, and KV dtype must "
                        "match between the paired instances")
                if external_restore is not None:
                    if (external_restore.source_entry is not entry
                            or external_restore.bytes_per_lane
                            != source_hit_per_rank
                            or external_restore.lane_count
                            != source_scheduler.num_npus
                            or external_restore.total_bytes != wire_bytes
                            or external_restore.completion_time_ns is None
                            or external_restore.critical_lane_start_ns is None):
                        raise RuntimeError(
                            "Completed external fabric source metadata no longer "
                            f"matches retry session={session_id}")
                    restore_hbm_admission_wait_ns = (
                        external_restore.arrival_time_ns
                        - release_time_ns
                        - pd_pair_fifo_wait_ns
                        - prepare_boundary_wait_ns)
                    if restore_hbm_admission_wait_ns < 0:
                        raise RuntimeError(
                            "External pre-restore admission waits exceed the "
                            "release-to-fabric admission interval: "
                            f"session={session_id}, pair="
                            f"{pd_pair_fifo_wait_ns}, boundary="
                            f"{prepare_boundary_wait_ns}")
                    restore_queue_wait_ns = (
                        external_restore.critical_lane_start_ns
                        - external_restore.arrival_time_ns)
                    restore_service_ns = (
                        external_restore.completion_time_ns
                        - external_restore.critical_lane_start_ns)
                    restore_ns = (
                        external_restore.completion_time_ns
                        - release_time_ns
                        - pd_pair_fifo_wait_ns
                        - prepare_boundary_wait_ns)
                    if restore_ns != (
                            restore_hbm_admission_wait_ns
                            + restore_queue_wait_ns + restore_service_ns):
                        raise RuntimeError(
                            "External fabric restore accounting does not "
                            f"reconcile for session={session_id}")
                    self.metrics.critical_restore_queue_wait_ns += (
                        restore_queue_wait_ns)
                    self.metrics.critical_restore_service_ns += (
                        restore_service_ns)
                elif (self._external_fabric_enabled
                        and self.config.pd_peer_transfer_mode
                        == "direct-fabric"):
                    if (source_scheduler.num_npus != scheduler.num_npus
                            or source_hit_per_rank != hit_per_rank
                            or source_hit_total != hit_total):
                        self._cancel_hbm_reservation(
                            candidate, candidate_ready_ns)
                        raise RuntimeError(
                            "External P/D cold restore requires identical TP "
                            "lane count and per-rank KV layout")
                    job_id = f"coldkv.{self._external_fabric_sequence}"
                    self._external_fabric_sequence += 1
                    restore = ExternalFabricRestore(
                        job_id=job_id,
                        session_id=session_id,
                        source_entry=entry,
                        target_entry=candidate,
                        source_instance_id=entry.instance_id,
                        target_instance_id=instance_id,
                        release_time_ns=release_time_ns,
                        arrival_time_ns=int(candidate_ready_ns),
                        declared_reuse_tokens=declared_reuse,
                        requested_reuse_tokens=requested_reuse,
                        reusable_tokens=reusable,
                        input_tokens=int(input_tokens),
                        block_tokens=hit_block_tokens,
                        bytes_per_lane=source_hit_per_rank,
                        lane_count=source_scheduler.num_npus,
                        total_bytes=wire_bytes,
                        return_gap_type=str(return_gap_type or "unknown"),
                        return_gap_source=str(
                            return_gap_source or "unknown"),
                        return_gap_ns=int(return_gap_ns),
                        residency_at_return=return_residency,
                        pd_pair_fifo_wait_ns=pd_pair_fifo_wait_ns,
                        prepare_boundary_wait_ns=(
                            prepare_boundary_wait_ns),
                    )
                    if (session_id in self._external_fabric_by_session
                            or job_id in self._external_fabric_by_job):
                        self._cancel_hbm_reservation(
                            candidate, candidate_ready_ns)
                        raise RuntimeError(
                            "External fabric restore identity collision: "
                            f"session={session_id}, job={job_id}")
                    self._external_fabric_by_session[session_id] = restore
                    self._external_fabric_by_job[job_id] = restore
                    self._external_fabric_outgoing.append(job_id)
                    # The target HBM has been admitted and the physical ASTRA
                    # job is now authoritative.  Do not mislabel its later
                    # queue/service tail as destination-capacity admission if
                    # this request had an earlier capacity retry.
                    if self._consume_destination_admission_wait(
                            session_id, destination_admission_start_ns):
                        self._destination_admission_intervals.append((
                            destination_admission_start_ns,
                            int(candidate_ready_ns),
                        ))
                    self._set_restore_capacity_pin(session_id, True)
                    return None
                else:
                    try:
                        raw_service_ns = self._hbm_peer_transfer_ns(
                            source_scheduler, scheduler,
                            source_hit_per_rank, hit_per_rank, wire_bytes)
                        reservation = self._reserve_transfer(
                            kind="hbm_peer",
                            arrival_ns=candidate_ready_ns,
                            service_ns=raw_service_ns,
                            source_instance_id=entry.instance_id,
                            target_instance_id=instance_id,
                            num_bytes=wire_bytes,
                            background=False,
                            session_id=session_id,
                            job_arrival_ns=operation_time_ns,
                        )
                        (
                            restore_ns,
                            restore_hbm_admission_wait_ns,
                            restore_queue_wait_ns,
                            restore_service_ns,
                        ) = self._foreground_restore_breakdown(
                            reservation, release_time_ns,
                            pd_pair_fifo_wait_ns,
                            prepare_boundary_wait_ns,
                            source_demotion_join_wait_ns)
                    except Exception:
                        self._cancel_hbm_reservation(
                            candidate, candidate_ready_ns)
                        raise
                # The decode side remains authoritative for the old prefix.
                # Prefill receives a copy and later sends only newly computed
                # suffix KV back through the normal P->D converter path.
                self._account_residence(entry, operation_time_ns)
                source_residence_accounted = True
                surplus = entry.per_rank_bytes - source_hit_per_rank
                if surplus:
                    source_scheduler.memory.free(surplus, Device.NPU)
                retained_instance_id = entry.instance_id
                retained_per_rank_bytes = source_hit_per_rank
                self.metrics.hbm_hits += 1
                self.metrics.pd_hbm_to_hbm_bytes += wire_bytes
                self.metrics.pd_cross_instance_restore_ns += restore_ns
        elif source in {KVLocation.CPU, KVLocation.SSD} and reusable:
            # Reject an unsupported node/layout before HBM pressure can evict
            # an unrelated target-side entry for a restore that cannot run.
            self._validate_node_shared_restore_layout(
                entry, scheduler, hit_block_tokens,
                hit_per_rank, hit_total)
            durable_record = self.ssd_records.get(session_id)
            physical_entry_bytes = int(entry.total_bytes)
            if (durable_record is not None
                    and entry.location != KVLocation.SSD):
                physical_entry_bytes += int(durable_record.bytes)
            candidate = IdleKVEntry(
                session_id=session_id, instance_id=instance_id,
                tokens=reusable, block_tokens=hit_block_tokens,
                per_rank_bytes=hit_per_rank, total_bytes=hit_total,
                location=KVLocation.HBM,
                tier_since_ns=operation_time_ns,
                last_access_ns=operation_time_ns,
            )
            hbm_restore_projection = None
            projection_apply_event_start = None
            if (self.config.queue_recompute_enabled
                    and str(session_id) not in
                    self._queue_recompute_restore_commitments):
                hbm_restore_projection = (
                    self._project_hbm_then_lower_tier_restore(
                        candidate=candidate,
                        source=source,
                        staging_instance_id=entry.instance_id,
                        target_instance_id=instance_id,
                        per_rank_bytes=hit_per_rank,
                        total_bytes=hit_total,
                        operation_time_ns=operation_time_ns,
                    ))
            queue_selection = self._evaluate_queue_recompute(
                    session_id=session_id,
                    source=source,
                    projection=hbm_restore_projection,
                    staging_instance_id=entry.instance_id,
                    target_instance_id=instance_id,
                    pd_decode_instance_id=pd_decode_instance_id,
                    per_rank_bytes=hit_per_rank,
                    total_bytes=hit_total,
                    physical_entry_bytes=physical_entry_bytes,
                    declared_reuse_tokens=declared_reuse,
                    reusable_tokens=reusable,
                    policy_avoidable_tokens=policy_reusable,
                    input_tokens=input_tokens,
                    operation_time_ns=operation_time_ns,
                    )
            if queue_selection.modified:
                queue_recompute_selected = True
                queue_recompute_partial = queue_selection.partial
                reusable = int(queue_selection.selected_tokens)
                hit_block_tokens = int(
                    queue_selection.selected_block_tokens)
                hit_per_rank = int(
                    queue_selection.selected_per_rank_bytes)
                hit_total = int(queue_selection.selected_total_bytes)
                candidate = IdleKVEntry(
                    session_id=session_id, instance_id=instance_id,
                    tokens=reusable, block_tokens=hit_block_tokens,
                    per_rank_bytes=hit_per_rank, total_bytes=hit_total,
                    location=KVLocation.HBM,
                    tier_since_ns=operation_time_ns,
                    last_access_ns=operation_time_ns,
                )
                hbm_restore_projection = (
                    queue_selection.selected_projection)
            if queue_selection.zero_restore:
                candidate_ready_ns = None
                initial_drop_reason = "queue_pressure"
                source = KVLocation.DROPPED
                reusable = 0
                hit_total = 0
            else:
                if hbm_restore_projection is not None:
                    projection_apply_event_start = len(self.events)
                restore_pin_was_pending = (
                    session_id in self._pending_restore_sessions)
                self._set_restore_capacity_pin(session_id, True)
                try:
                    candidate_ready_ns = self._reserve_hbm(
                        candidate, operation_time_ns)
                    if (hbm_restore_projection is not None
                            and candidate_ready_ns is None):
                        self._assert_hbm_restore_projection_applied(
                            projection=hbm_restore_projection,
                            candidate_ready_ns=candidate_ready_ns,
                            source=source,
                            staging_instance_id=entry.instance_id,
                            target_instance_id=instance_id,
                            per_rank_bytes=hit_per_rank,
                            total_bytes=hit_total,
                            reservations=None,
                            event_start_index=(
                                projection_apply_event_start),
                        )
                except Exception:
                    if not restore_pin_was_pending:
                        self._set_restore_capacity_pin(session_id, False)
                    raise
            if queue_recompute_selected and not queue_recompute_partial:
                pass
            elif candidate_ready_ns is None:
                if (defer_temporary_hbm_pressure
                        and candidate.per_rank_bytes
                        <= self._hbm_kv_ceiling(scheduler)):
                    self._begin_destination_admission_wait(
                        session_id, destination_admission_start_ns,
                        operation_time_ns)
                    self._pause_transient_restore_wait(
                        session_id, operation_time_ns)
                    self._set_restore_capacity_pin(session_id, True)
                    self.events.append({
                        "time_ns": operation_time_ns,
                        "session_id": session_id,
                        "event": "hbm_restore_admission_deferred",
                        "release_time_ns": release_time_ns,
                        "target_instance_id": instance_id,
                        "per_rank_bytes": candidate.per_rank_bytes,
                        "source": source.value,
                    })
                    return None
                source = KVLocation.DROPPED
                reusable = 0
                hit_total = 0
                restore_allocation_failed = True
                restore_failure_tier = "hbm"
            elif source == KVLocation.CPU:
                try:
                    raw_service_ns = self._cpu_transfer_ns(
                        hit_per_rank, hit_total)
                    reservation = self._reserve_transfer(
                        kind="cpu_to_hbm",
                        arrival_ns=candidate_ready_ns,
                        service_ns=raw_service_ns,
                        source_instance_id=entry.instance_id,
                        target_instance_id=instance_id,
                        num_bytes=hit_total,
                        background=False,
                        session_id=session_id,
                        job_arrival_ns=operation_time_ns,
                    )
                    if hbm_restore_projection is not None:
                        self._assert_hbm_restore_projection_applied(
                            projection=hbm_restore_projection,
                            candidate_ready_ns=candidate_ready_ns,
                            source=source,
                            staging_instance_id=entry.instance_id,
                            target_instance_id=instance_id,
                            per_rank_bytes=hit_per_rank,
                            total_bytes=hit_total,
                            reservations=(reservation,),
                            event_start_index=(
                                projection_apply_event_start),
                        )
                except Exception:
                    self._cancel_hbm_reservation(
                        candidate, candidate_ready_ns)
                    self._set_restore_capacity_pin(session_id, False)
                    raise
                (
                    restore_ns,
                    restore_hbm_admission_wait_ns,
                    restore_queue_wait_ns,
                    restore_service_ns,
                ) = self._foreground_restore_breakdown(
                    reservation, release_time_ns,
                    pd_pair_fifo_wait_ns,
                    prepare_boundary_wait_ns,
                    source_demotion_join_wait_ns)
                self.pending_source_releases.append(PendingSourceRelease(
                    entry=entry,
                    ready_ns=(release_time_ns + pd_pair_fifo_wait_ns
                              + prepare_boundary_wait_ns
                              + source_demotion_join_wait_ns
                              + restore_ns),
                    remove_ssd_record=queue_recompute_partial,
                ))
                if queue_recompute_partial:
                    durable_record = self.ssd_records.get(session_id)
                    if durable_record is not None:
                        durable_record.pinned_until_ns = max(
                            durable_record.pinned_until_ns,
                            release_time_ns + pd_pair_fifo_wait_ns
                            + prepare_boundary_wait_ns
                            + source_demotion_join_wait_ns
                            + restore_ns,
                        )
                source_release_deferred = True
                self.metrics.cpu_hits += 1
                self.metrics.cpu_to_hbm_bytes += hit_total
            else:
                # Even the compatibility-named hbm_ssd_direct policy restores
                # through a transient host-DRAM stage before H2D PCIe. Only
                # its swap-out path remains direct-to-storage. The shared
                # lower tier restores directly to the selected P HBM; D HBM
                # is reserved later, in full, for the normal P->D handoff.
                try:
                    reservations = self._reserve_ssd_restore_stages(
                            arrival_ns=candidate_ready_ns,
                            staging_instance_id=entry.instance_id,
                            target_instance_id=instance_id,
                            per_rank_bytes=hit_per_rank,
                            total_bytes=hit_total,
                            session_id=session_id,
                            job_arrival_ns=operation_time_ns,
                        )
                    if hbm_restore_projection is not None:
                        self._assert_hbm_restore_projection_applied(
                            projection=hbm_restore_projection,
                            candidate_ready_ns=candidate_ready_ns,
                            source=source,
                            staging_instance_id=entry.instance_id,
                            target_instance_id=instance_id,
                            per_rank_bytes=hit_per_rank,
                            total_bytes=hit_total,
                            reservations=reservations,
                            event_start_index=(
                                projection_apply_event_start),
                        )
                except Exception:
                    self._cancel_hbm_reservation(
                        candidate, candidate_ready_ns)
                    self._set_restore_capacity_pin(session_id, False)
                    raise
                if reservations is None:
                    self._cancel_hbm_reservation(
                        candidate, candidate_ready_ns)
                    if (defer_temporary_hbm_pressure
                            and hit_total <= self._cpu_capacity_bytes(
                                self._scheduler(entry.instance_id))):
                        self._begin_destination_admission_wait(
                            session_id, destination_admission_start_ns,
                            operation_time_ns)
                        self._begin_transient_restore_wait(
                            session_id, operation_time_ns)
                        self._set_restore_capacity_pin(session_id, True)
                        self.events.append({
                            "time_ns": operation_time_ns,
                            "session_id": session_id,
                            "event": (
                                "transient_dram_restore_admission_deferred"),
                            "release_time_ns": release_time_ns,
                            "target_instance_id": instance_id,
                            "bytes": hit_total,
                            "source": source.value,
                        })
                        return None
                    source = KVLocation.DROPPED
                    reusable = 0
                    hit_total = 0
                    restore_allocation_failed = True
                    restore_failure_tier = "transient_dram"
                else:
                    ssd_media_reservation, h2d_reservation = reservations
                if not restore_allocation_failed:
                    deferred_transient_wait_ns = (
                        self._consume_transient_restore_wait(
                            session_id, operation_time_ns)
                    )
                    restore_transient_dram_capacity_wait_ns = (
                        deferred_transient_wait_ns
                        + ssd_media_reservation
                        .transient_dram_capacity_wait_ns
                    )
                    if deferred_transient_wait_ns:
                        self.metrics.transient_dram_capacity_wait_ns += (
                            deferred_transient_wait_ns)
                        self.metrics.transient_dram_pressure_stall_ns += (
                            deferred_transient_wait_ns)
                        self.events.append({
                            "time_ns": operation_time_ns,
                            "session_id": session_id,
                            "event": (
                                "transient_dram_deferred_wait_reconciled"),
                            "wait_ns": deferred_transient_wait_ns,
                        })
                    self.metrics.critical_restore_queue_wait_ns += (
                        restore_transient_dram_capacity_wait_ns)
                    (
                        restore_ns,
                        restore_hbm_admission_wait_ns,
                        restore_queue_wait_ns,
                        restore_service_ns,
                    ) = self._foreground_restore_chain_breakdown(
                        (ssd_media_reservation, h2d_reservation),
                        release_time_ns,
                        pd_pair_fifo_wait_ns,
                        prepare_boundary_wait_ns,
                        source_demotion_join_wait_ns,
                        restore_transient_dram_capacity_wait_ns,
                    )
                    self.metrics.ssd_hits += 1
                    self.metrics.ssd_to_hbm_bytes += hit_total
                    self.metrics.ssd_host_read_bytes += hit_total
                    record = self.ssd_records.get(session_id)
                    if record is not None:
                        record.last_access_ns = operation_time_ns
                        record.pinned_until_ns = max(
                            record.pinned_until_ns,
                            release_time_ns + pd_pair_fifo_wait_ns
                            + prepare_boundary_wait_ns
                            + source_demotion_join_wait_ns
                            + restore_ns,
                        )
                    # Every SSD read pins its source record until DMA completion.
                    # Without a pending event, an exact-prefix keep-on-read restore
                    # could be evicted by a concurrent SSD-capacity admission while
                    # the request was still reading it.
                    self.pending_source_releases.append(PendingSourceRelease(
                        entry=entry,
                        ready_ns=(release_time_ns + pd_pair_fifo_wait_ns
                                  + prepare_boundary_wait_ns
                                  + source_demotion_join_wait_ns
                                  + restore_ns),
                        remove_ssd_record=(
                            defer_lineage_invalidation
                            or queue_recompute_partial
                            or not self.config.keep_ssd_copy_on_read),
                    ))
                    source_release_deferred = True
        else:
            reusable = 0

        if source == KVLocation.DROPPED or reusable == 0:
            # A zero-overlap continuation or failed restore must release the
            # manager's old physical copy.
            # Otherwise the entry is removed below while its HBM/CPU counter
            # remains charged.  A valid durable SSD copy may survive a failed
            # HBM allocation for a later retry, but it is invalid when the
            # workload explicitly reports zero prefix reuse.
            if defer_lineage_invalidation:
                self._truncate_ssd_lineage(
                    session_id, declared_reuse, operation_time_ns)
                defer_lineage_invalidation = False
            keep_durable = (
                restore_allocation_failed
                and self.config.keep_ssd_copy_on_read
                and entry.session_id in self.ssd_records
            )
            if restore_allocation_failed:
                self.metrics.capacity_drops += 1
                if restore_failure_tier == "hbm":
                    self.metrics.hbm_capacity_drops += 1
                self.events.append({
                    "time_ns": operation_time_ns,
                    "release_time_ns": release_time_ns,
                    "session_id": session_id,
                    "event": (
                        "hbm_capacity_restore_drop"
                        if restore_failure_tier == "hbm" else
                        "transient_dram_capacity_restore_drop"
                    ),
                    "tokens": declared_reuse,
                })
            self._drop_entry(
                entry, operation_time_ns,
                "queue_pressure" if queue_recompute_selected else
                "resume_miss",
                keep_ssd_record=keep_durable)
            recompute = declared_reuse
            policy_avoidable_recompute = policy_reusable
            if recompute:
                self.metrics.dropped_misses += 1
                self.metrics.recompute_tokens += recompute
                self.metrics.policy_avoidable_recompute_tokens += (
                    policy_avoidable_recompute)
                if queue_recompute_selected:
                    self.metrics.queue_recompute_tokens += recompute
                    self.metrics.queue_recompute_policy_avoidable_tokens += (
                        policy_avoidable_recompute)
                if (restore_allocation_failed
                        or (initial_drop_reason is not None
                            and "capacity" in initial_drop_reason)):
                    self.metrics.capacity_induced_recompute_tokens += (
                        policy_avoidable_recompute)
            reusable = 0
            pending_destination_start_ns = (
                self._pending_destination_admission_since.get(session_id))
            terminal_destination_wait_ns = (
                0 if pending_destination_start_ns is None else
                operation_time_ns - int(pending_destination_start_ns)
            )
            if terminal_destination_wait_ns < 0:
                raise RuntimeError(
                    "Terminal restore decision precedes destination wait: "
                    f"session={session_id}, operation={operation_time_ns}, "
                    f"start={pending_destination_start_ns}")
            prepare_boundary_wait_ns = (
                operation_time_ns - release_time_ns
                - pd_pair_fifo_wait_ns
                - source_demotion_join_wait_ns
                - terminal_destination_wait_ns)
            if prepare_boundary_wait_ns < 0:
                raise RuntimeError(
                    "Source-demotion join exceeds failed-restore owner delay: "
                    f"session={session_id}")
            restore_hbm_admission_wait_ns = terminal_destination_wait_ns
            restore_ns = terminal_destination_wait_ns
            restore_queue_wait_ns = 0
            restore_service_ns = 0
            restore_transient_dram_capacity_wait_ns = 0
            hit_total = 0
            self._clear_transient_restore_wait(session_id)
        else:
            recompute = max(0, declared_reuse - reusable)
            policy_avoidable_recompute = max(0, policy_reusable - reusable)
            self.metrics.cache_hit_tokens += reusable
            self.metrics.recompute_tokens += recompute
            self.metrics.policy_avoidable_recompute_tokens += (
                policy_avoidable_recompute)
            if queue_recompute_selected:
                self.metrics.queue_recompute_tokens += recompute
                self.metrics.queue_recompute_policy_avoidable_tokens += (
                    policy_avoidable_recompute)

        if (entry.location != KVLocation.DROPPED
                and not source_release_deferred
                and not source_residence_accounted):
            self._account_residence(entry, operation_time_ns)
        committed_join_wait_ns = self._consume_demotion_join(
            session_id, operation_time_ns)
        if committed_join_wait_ns != source_demotion_join_wait_ns:
            raise RuntimeError(
                "Final source-demotion join wait changed during preparation: "
                f"session={session_id}, observed="
                f"{source_demotion_join_wait_ns}, committed="
                f"{committed_join_wait_ns}")
        self._consume_destination_admission_wait(
            session_id, destination_admission_start_ns)
        self._set_restore_capacity_pin(session_id, False)
        self.metrics.resumed_prompt_tokens += max(0, int(input_tokens))
        self._sync_deferred_hbm_demotions.discard(session_id)
        self._queue_recompute_restore_commitments.pop(
            str(session_id), None)
        del self.entries[session_id]
        self._mark_hbm_admission_state_changed()
        self._update_idle_peaks()
        self._record_critical_restore_accounting(
            pd_pair_fifo_wait_ns=pd_pair_fifo_wait_ns,
            prepare_boundary_wait_ns=prepare_boundary_wait_ns,
            source_demotion_join_wait_ns=source_demotion_join_wait_ns,
            hbm_admission_wait_ns=restore_hbm_admission_wait_ns,
            queue_wait_ns=restore_queue_wait_ns,
            service_ns=restore_service_ns,
            expected_total_ns=restore_ns,
        )
        owner_gate_ns = (
            pd_pair_fifo_wait_ns + prepare_boundary_wait_ns
            + source_demotion_join_wait_ns + restore_ns)
        restore_issue_time_ns = (
            release_time_ns + pd_pair_fifo_wait_ns
            + prepare_boundary_wait_ns + source_demotion_join_wait_ns)
        if restore_ns:
            self._critical_restore_intervals.append((
                restore_issue_time_ns,
                release_time_ns + owner_gate_ns))
        if restore_hbm_admission_wait_ns:
            self._destination_admission_intervals.append((
                restore_issue_time_ns,
                restore_issue_time_ns + restore_hbm_admission_wait_ns,
            ))
        if (policy_reusable > 0
                and return_residency in {
                    KVLocation.CPU, KVLocation.SSD, KVLocation.DROPPED
                }):
            opportunity_blocks = (
                policy_reusable + self.config.block_size - 1
            ) // self.config.block_size
            opportunity_bytes = (
                scheduler.memory.get_kv(
                    opportunity_blocks * self.config.block_size)
                * scheduler.num_npus
            )
            self.metrics.hbf_eligible_resumes += 1
            self.metrics.hbf_eligible_restore_bytes += opportunity_bytes
            self.metrics.hbf_gross_stall_upper_bound_ns += restore_ns
            if source == KVLocation.DROPPED:
                self.metrics.hbf_dropped_recompute_tokens += (
                    policy_avoidable_recompute)
        self.events.append({
            "time_ns": release_time_ns,
            "operation_time_ns": operation_time_ns,
            "session_id": session_id,
            "request_id": request_id,
            "sub_request_index": sub_request_index,
            "event": "resume",
            "source": source.value,
            "residency_at_return": return_residency.value,
            "source_instance_id": entry.instance_id,
            "target_instance_id": instance_id,
            "source_node_id": self._node_id(
                self._scheduler(entry.instance_id)),
            "target_node_id": self._node_id(scheduler),
            "hit_tokens": reusable,
            "recompute_tokens": recompute,
            "policy_avoidable_recompute_tokens": (
                policy_avoidable_recompute),
            "restore_ns": restore_ns,
            "owner_gate_ns": owner_gate_ns,
            "pd_pair_fifo_wait_ns": pd_pair_fifo_wait_ns,
            "prepare_boundary_wait_ns": prepare_boundary_wait_ns,
            "source_demotion_join_wait_ns": source_demotion_join_wait_ns,
            "restore_issue_time_ns": restore_issue_time_ns,
            "target_hbm_ready_time_ns": (
                restore_issue_time_ns
                + restore_hbm_admission_wait_ns),
            "restore_ready_time_ns": (
                restore_issue_time_ns + restore_ns),
            "hbm_admission_wait_ns": restore_hbm_admission_wait_ns,
            "transient_dram_capacity_wait_ns": (
                restore_transient_dram_capacity_wait_ns),
            "restore_service_ns": restore_service_ns,
            "queue_wait_ns": restore_queue_wait_ns,
            "bytes": hit_total,
            "return_gap_type": str(return_gap_type or "unknown"),
            "return_gap_source": str(return_gap_source or "unknown"),
            "return_gap_ns": int(return_gap_ns),
        })
        if external_restore is not None:
            if (external_restore.status != "completed"
                    or external_restore.completion_time_ns is None
                    or external_restore.critical_lane_start_ns is None):
                raise RuntimeError(
                    "External fabric restore was consumed before completion: "
                    f"job={external_restore.job_id}")
            callback = (
                external_restore.arrival_time_ns,
                external_restore.completion_time_ns,
                external_restore.bytes_per_lane,
                external_restore.lane_count,
                external_restore.critical_lane_start_ns,
            )
            self._external_fabric_tombstones[
                external_restore.job_id] = callback
            self._external_fabric_by_job.pop(external_restore.job_id)
            self._external_fabric_by_session.pop(session_id)
        return KVPreparation(
            hit_tokens=reusable,
            recompute_tokens=recompute,
            source=source,
            restore_ns=restore_ns,
            ready_time_ns=release_time_ns + owner_gate_ns,
            restored_bytes=hit_total,
            owner_gate_ns=owner_gate_ns,
            restore_issue_time_ns=restore_issue_time_ns,
            target_hbm_ready_time_ns=(
                restore_issue_time_ns
                + restore_hbm_admission_wait_ns),
            restore_ready_time_ns=release_time_ns + owner_gate_ns,
            pd_pair_fifo_wait_ns=pd_pair_fifo_wait_ns,
            prepare_boundary_wait_ns=prepare_boundary_wait_ns,
            source_demotion_join_wait_ns=source_demotion_join_wait_ns,
            hbm_admission_wait_ns=restore_hbm_admission_wait_ns,
            transient_dram_capacity_wait_ns=(
                restore_transient_dram_capacity_wait_ns),
            queue_wait_ns=restore_queue_wait_ns,
            service_ns=restore_service_ns,
            retained_instance_id=retained_instance_id,
            retained_per_rank_bytes=retained_per_rank_bytes,
            residency_at_return=return_residency,
        )

    def _remove_ssd_record(
            self, session_id: str, now_ns: Optional[int] = None) -> None:
        record = self.ssd_records.pop(session_id, None)
        if record is not None:
            if now_ns is not None:
                self._account_ssd_record(record, now_ns)
            self.ssd_used_bytes = max(0, self.ssd_used_bytes - record.bytes)

    def _truncate_ssd_lineage(
            self, session_id: str, valid_tokens: int, now_ns: int) -> None:
        """Invalidate a durable object after any transitive prefix divergence.

        A durable snapshot may remain on SSD while a newer turn runs from HBM
        or CPU. Immediate-pair reuse alone is then insufficient: once a turn
        diverges at token ``h``, bytes beyond ``h`` from any older snapshot can
        never be treated as an append base. The conservative incremental
        baseline stores whole objects, so it invalidates that snapshot rather
        than assuming block-level copy-on-write prefix sharing.
        """
        record = self.ssd_records.get(session_id)
        if record is None:
            return
        valid_tokens = max(0, min(int(valid_tokens), record.tokens))
        self._account_ssd_record(record, now_ns)
        if valid_tokens == record.tokens:
            record.last_access_ns = now_ns
            return
        old_tokens = record.tokens
        old_bytes = record.bytes
        self._remove_ssd_record(session_id, now_ns)
        self.events.append({
            "time_ns": now_ns,
            "session_id": session_id,
            "event": "ssd_lineage_invalidate",
            "valid_tokens": valid_tokens,
            "invalidated_tokens": old_tokens,
            "bytes": old_bytes,
        })

    def end_session(
            self, session_id: str, now_ns: Optional[int] = None,
            keep_durable: bool = False) -> None:
        session_id = str(session_id)
        open_waits = []
        has_join_session = session_id in self._pending_demotion_join_sessions
        has_join_window = session_id in self._pending_demotion_join_windows
        if has_join_session != has_join_window:
            raise RuntimeError(
                "Source-demotion join end found inconsistent state: "
                f"session={session_id}, pending={has_join_session}, "
                f"window={has_join_window}")
        if has_join_window:
            open_waits.append("source-demotion join")
        if session_id in self._pending_destination_admission_since:
            open_waits.append("destination admission")
        if (session_id in self._pending_transient_restore_since
                or session_id in self._pending_transient_restore_wait_ns):
            open_waits.append("transient DRAM admission")
        if open_waits:
            raise RuntimeError(
                "Cannot ordinarily end a session with open restore waits; "
                "use censor_session at a measurement cutoff: "
                f"session={session_id}, waits={open_waits}")
        external_restore = self._external_fabric_by_session.get(session_id)
        if external_restore is not None:
            if external_restore.status != "queued":
                raise RuntimeError(
                    "Cannot end a session while an ASTRA cold-fabric job is "
                    f"{external_restore.status}: session={session_id}, "
                    f"job={external_restore.job_id}")
            self._external_fabric_outgoing.remove(external_restore.job_id)
            self._external_fabric_by_job.pop(external_restore.job_id)
            self._external_fabric_by_session.pop(session_id)
            self._cancel_hbm_reservation(
                external_restore.target_entry,
                external_restore.arrival_time_ns)
        self._set_restore_capacity_pin(session_id, False)
        self._queue_recompute_restore_commitments.pop(session_id, None)
        self._clear_demotion_join(session_id)
        self._clear_transient_restore_wait(session_id)
        entry = self.entries.pop(session_id, None)
        if entry is not None:
            end_ns = entry.last_access_ns if now_ns is None else now_ns
            self._drop_entry(
                entry, end_ns, "session_end",
                keep_ssd_record=keep_durable)
        if not keep_durable:
            self._remove_ssd_record(session_id, now_ns)

    def censor_session(
            self, session_id: str, cutoff_ns: int) -> Optional[dict]:
        """End a censored session while retaining open causal exposure."""
        session_id = str(session_id)
        cutoff_ns = int(cutoff_ns)
        join_audit = self._censor_demotion_join(session_id, cutoff_ns)
        destination_audit = self._censor_destination_admission_wait(
            session_id, cutoff_ns)
        transient_audit = self._censor_transient_restore_wait(
            session_id, cutoff_ns)
        self.end_session(session_id, now_ns=cutoff_ns)
        if not any((join_audit, destination_audit, transient_audit)):
            return None
        return {
            "session_id": session_id,
            "cutoff_ns": cutoff_ns,
            "source_demotion_join": join_audit,
            "destination_admission": destination_audit,
            "transient_dram_admission": transient_audit,
        }

    def censor_prepared_request(self, request, now_ns: int) -> dict:
        """Release tier-manager ownership for an unlaunched P/D request.

        A successful ``prepare_request`` can own a restored P prefix and a
        retained D prefix before strict P/D suffix claims make the request
        scheduler-visible. Measurement early-stop must unwind those physical
        objects exactly once. Once ``pd_prefill_preallocated_per_rank_bytes``
        is nonzero, ownership has transferred to the scheduler and this API
        intentionally refuses to guess how to cancel queued/active work.
        """
        now_ns = int(now_ns)
        if int(request.pd_prefill_preallocated_per_rank_bytes) != 0:
            raise RuntimeError(
                "Cannot manager-censor a scheduler-visible P/D request: "
                f"request={request.id}, preallocated="
                f"{request.pd_prefill_preallocated_per_rank_bytes}")
        self.advance(now_ns)

        session_id = str(request.session_id)
        hit_tokens = max(0, int(request.agentic_kv_hit_tokens))
        owner_instance_id = request.agentic_kv_owner_instance_id
        retained_instance_id = request.agentic_kv_retained_instance_id
        retained_per_rank_bytes = int(
            request.agentic_kv_retained_per_rank_bytes)
        if hit_tokens > 0 and owner_instance_id is None:
            raise RuntimeError(
                "Prepared KV hit has no P-side owner during censoring: "
                f"request={request.id}, session={session_id}")
        if hit_tokens == 0 and owner_instance_id is not None:
            raise RuntimeError(
                "Zero-hit prepared request unexpectedly owns P HBM: "
                f"request={request.id}, owner={owner_instance_id}")
        if retained_per_rank_bytes < 0:
            raise RuntimeError(
                "Prepared request has negative retained D HBM bytes: "
                f"request={request.id}, bytes={retained_per_rank_bytes}")
        if ((retained_instance_id is None) !=
                (retained_per_rank_bytes == 0)):
            raise RuntimeError(
                "Retained P/D censor ownership is incomplete: "
                f"request={request.id}, instance={retained_instance_id}, "
                f"bytes={retained_per_rank_bytes}")

        released_owner_per_rank_bytes = 0
        cancelled_pending_target = False
        if hit_tokens > 0:
            owner = self._scheduler(int(owner_instance_id))
            block_tokens = (
                (hit_tokens + self.config.block_size - 1)
                // self.config.block_size
                * self.config.block_size
            )
            owner_per_rank_bytes = int(owner.memory.get_kv(block_tokens))
            pending_targets = [
                pending for pending in self.pending_hbm_allocations
                if pending.entry.session_id == session_id
                and pending.entry.instance_id == int(owner_instance_id)
            ]
            if len(pending_targets) > 1:
                raise RuntimeError(
                    "Prepared request owns multiple pending P HBM targets: "
                    f"request={request.id}, count={len(pending_targets)}")
            if pending_targets:
                pending = pending_targets[0]
                if int(pending.entry.per_rank_bytes) != owner_per_rank_bytes:
                    raise RuntimeError(
                        "Pending P HBM censor size does not match restored "
                        f"prefix: request={request.id}, pending="
                        f"{pending.entry.per_rank_bytes}, expected="
                        f"{owner_per_rank_bytes}")
                self.pending_hbm_allocations.remove(pending)
                self._mark_hbm_admission_state_changed()
                cancelled_pending_target = True
            else:
                owner.memory.free(owner_per_rank_bytes, Device.NPU)
                released_owner_per_rank_bytes = owner_per_rank_bytes

        released_retained_per_rank_bytes = 0
        if retained_instance_id is not None:
            retained = self._scheduler(int(retained_instance_id))
            retained.memory.free(retained_per_rank_bytes, Device.NPU)
            released_retained_per_rank_bytes = retained_per_rank_bytes

        pending_sources = [
            pending for pending in self.pending_source_releases
            if pending.entry.session_id == session_id
        ]
        if len(pending_sources) > 1:
            raise RuntimeError(
                "Prepared request owns multiple pending source releases: "
                f"request={request.id}, count={len(pending_sources)}")
        released_source = None
        if pending_sources:
            pending = pending_sources[0]
            self.pending_source_releases.remove(pending)
            released_source = pending.entry.location.value
            self._drop_entry(
                pending.entry, now_ns, "measurement_censor")
        # A keep-on-read durable SSD snapshot can survive the source release;
        # the whole session is censored, so no later continuation may use it.
        self._remove_ssd_record(session_id, now_ns)

        request.agentic_kv_owner_instance_id = None
        request.agentic_kv_retained_instance_id = None
        request.agentic_kv_retained_per_rank_bytes = 0
        audit = {
            "request_id": int(request.id),
            "session_id": session_id,
            "time_ns": now_ns,
            "cancelled_pending_target": cancelled_pending_target,
            "released_owner_per_rank_bytes": (
                released_owner_per_rank_bytes),
            "released_retained_per_rank_bytes": (
                released_retained_per_rank_bytes),
            "released_source": released_source,
        }
        self.events.append({
            "event": "prepared_request_censored",
            **audit,
        })
        self._update_idle_peaks()
        return audit

    def censor_preallocated_pd_request(
            self, request, prefill_instance_id: int,
            decode_instance_id: int, now_ns: int) -> dict:
        """Release an admitted but not running strict-P/D request exactly.

        The caller must first remove the request from every Router launch row
        or prefill scheduler queue. P/D admission has physically allocated the
        fresh P suffix and D receive suffix, while ``prepare_request`` still
        owns the restored P prefix, retained D prefix, and any restore source.
        Free the two suffix allocations, then reuse the prepared-request
        censor path for the remaining ownership. A restored P target can still
        be a future HBM allocation, so freeing the full P size blindly is not
        valid.
        """
        now_ns = int(now_ns)
        prefill_instance_id = int(prefill_instance_id)
        decode_instance_id = int(decode_instance_id)
        prefill = self._scheduler(prefill_instance_id)
        decode = self._scheduler(decode_instance_id)
        prefill_full = int(request.pd_prefill_full_per_rank_bytes)
        prefill_suffix = int(request.pd_prefill_reserved_per_rank_bytes)
        preallocated = int(
            request.pd_prefill_preallocated_per_rank_bytes)
        decode_full = int(request.pd_decode_full_per_rank_bytes)
        decode_suffix = int(request.pd_decode_reserved_per_rank_bytes)
        retained = int(request.agentic_kv_retained_per_rank_bytes)
        hit_tokens = max(0, int(request.agentic_kv_hit_tokens))
        hit_block_tokens = (
            (hit_tokens + self.config.block_size - 1)
            // self.config.block_size * self.config.block_size
            if hit_tokens else 0
        )
        restored = int(prefill.memory.get_kv(hit_block_tokens))
        if preallocated <= 0 or preallocated != prefill_full:
            raise RuntimeError(
                "Censored prelaunch P/D request lacks its exact full P "
                f"preallocation: request={request.id}, preallocated="
                f"{preallocated}, full={prefill_full}")
        if restored + prefill_suffix != prefill_full:
            raise RuntimeError(
                "Censored prelaunch P allocation does not reconcile: "
                f"request={request.id}, restored={restored}, "
                f"suffix={prefill_suffix}, full={prefill_full}")
        if retained + decode_suffix != decode_full:
            raise RuntimeError(
                "Censored prelaunch D allocation does not reconcile: "
                f"request={request.id}, retained={retained}, "
                f"suffix={decode_suffix}, full={decode_full}")
        if request.pd_decode_target_instance_id != decode_instance_id:
            raise RuntimeError(
                "Censored prelaunch D target changed: "
                f"request={request.id}, request_target="
                f"{request.pd_decode_target_instance_id}, "
                f"caller_target={decode_instance_id}")
        expected_owner = prefill_instance_id if restored else None
        if request.agentic_kv_owner_instance_id != expected_owner:
            raise RuntimeError(
                "Censored prelaunch restored P owner changed: "
                f"request={request.id}, expected={expected_owner}, "
                f"observed={request.agentic_kv_owner_instance_id}")
        expected_retained_instance = decode_instance_id if retained else None
        if request.agentic_kv_retained_instance_id != (
                expected_retained_instance):
            raise RuntimeError(
                "Censored prelaunch retained D owner changed: "
                f"request={request.id}, expected="
                f"{expected_retained_instance}, observed="
                f"{request.agentic_kv_retained_instance_id}")

        if prefill_suffix:
            prefill.memory.free(prefill_suffix, Device.NPU)
        if decode_suffix:
            decode.memory.free(decode_suffix, Device.NPU)
        # The prepared-request path now sees exactly the restored/retained
        # remainder and is allowed to unwind it.
        request.pd_prefill_preallocated_per_rank_bytes = 0
        prepared_audit = self.censor_prepared_request(request, now_ns)
        request.pd_prefill_full_per_rank_bytes = 0
        request.pd_prefill_reserved_per_rank_bytes = 0
        request.pd_decode_target_instance_id = None
        request.pd_decode_full_per_rank_bytes = 0
        request.pd_decode_reserved_per_rank_bytes = 0
        audit = {
            **prepared_audit,
            "prefill_instance_id": prefill_instance_id,
            "decode_instance_id": decode_instance_id,
            "released_prefill_suffix_per_rank_bytes": prefill_suffix,
            "released_decode_suffix_per_rank_bytes": decode_suffix,
            "released_prefill_full_per_rank_bytes": prefill_full,
            "released_decode_full_per_rank_bytes": decode_full,
        }
        self.events.append({
            "event": "preallocated_pd_request_censored",
            **audit,
        })
        self._update_idle_peaks()
        return audit

    def censor_completed_pd_prefill_request(
            self, request, prefill_instance_id: int,
            decode_instance_id: int, now_ns: int) -> dict:
        """Release D HBM after a drained P batch is not handed to decode.

        The prefill scheduler has already freed its complete active KV and
        cleared the P owner. Strict admission nevertheless left the complete D
        receive allocation physical. Measurement freeze skips the normal
        P-to-D handoff, so this path releases only D and invalidates any durable
        restore source.
        """
        now_ns = int(now_ns)
        prefill_instance_id = int(prefill_instance_id)
        decode_instance_id = int(decode_instance_id)
        self.advance(now_ns)
        if request.agentic_kv_owner_instance_id is not None:
            raise RuntimeError(
                "Completed prefill censoring observed live P ownership: "
                f"request={request.id}, owner="
                f"{request.agentic_kv_owner_instance_id}")
        if int(request.instance_id) != prefill_instance_id:
            raise RuntimeError(
                "Completed prefill censoring changed P instance: "
                f"request={request.id}, request_instance="
                f"{request.instance_id}, expected={prefill_instance_id}")
        prefill_full = int(request.pd_prefill_full_per_rank_bytes)
        preallocated = int(
            request.pd_prefill_preallocated_per_rank_bytes)
        if prefill_full <= 0 or preallocated != prefill_full:
            raise RuntimeError(
                "Completed prefill lost its historical full P admission: "
                f"request={request.id}, preallocated={preallocated}, "
                f"full={prefill_full}")
        if request.pd_decode_target_instance_id != decode_instance_id:
            raise RuntimeError(
                "Completed prefill D target changed before censoring: "
                f"request={request.id}, request_target="
                f"{request.pd_decode_target_instance_id}, "
                f"expected={decode_instance_id}")
        decode_full = int(request.pd_decode_full_per_rank_bytes)
        decode_suffix = int(request.pd_decode_reserved_per_rank_bytes)
        retained = int(request.agentic_kv_retained_per_rank_bytes)
        if decode_full <= 0 or retained + decode_suffix != decode_full:
            raise RuntimeError(
                "Completed prefill D ownership does not reconcile: "
                f"request={request.id}, retained={retained}, "
                f"suffix={decode_suffix}, full={decode_full}")
        expected_retained_instance = decode_instance_id if retained else None
        if request.agentic_kv_retained_instance_id != (
                expected_retained_instance):
            raise RuntimeError(
                "Completed prefill retained D owner changed: "
                f"request={request.id}, expected="
                f"{expected_retained_instance}, observed="
                f"{request.agentic_kv_retained_instance_id}")
        pending_targets = [
            pending for pending in self.pending_hbm_allocations
            if pending.entry.session_id == str(request.session_id)
        ]
        if pending_targets:
            raise RuntimeError(
                "Completed prefill still has an uncommitted P restore target: "
                f"request={request.id}, count={len(pending_targets)}")

        decode = self._scheduler(decode_instance_id)
        decode.memory.free(decode_full, Device.NPU)
        session_id = str(request.session_id)
        pending_sources = [
            pending for pending in self.pending_source_releases
            if pending.entry.session_id == session_id
        ]
        if len(pending_sources) > 1:
            raise RuntimeError(
                "Completed prefill owns multiple pending restore sources: "
                f"request={request.id}, count={len(pending_sources)}")
        if pending_sources:
            pending = pending_sources[0]
            self.pending_source_releases.remove(pending)
            self._drop_entry(
                pending.entry, now_ns, "measurement_censor")
        self._remove_ssd_record(session_id, now_ns)
        request.agentic_kv_retained_instance_id = None
        request.agentic_kv_retained_per_rank_bytes = 0
        request.pd_prefill_preallocated_per_rank_bytes = 0
        request.pd_prefill_full_per_rank_bytes = 0
        request.pd_prefill_reserved_per_rank_bytes = 0
        request.pd_decode_target_instance_id = None
        request.pd_decode_full_per_rank_bytes = 0
        request.pd_decode_reserved_per_rank_bytes = 0
        audit = {
            "request_id": int(request.id),
            "session_id": session_id,
            "time_ns": now_ns,
            "prefill_instance_id": prefill_instance_id,
            "decode_instance_id": decode_instance_id,
            "released_decode_full_per_rank_bytes": decode_full,
            "prefill_already_released_per_rank_bytes": prefill_full,
        }
        self.events.append({
            "event": "completed_pd_prefill_request_censored",
            **audit,
        })
        self._update_idle_peaks()
        return audit

    def validate_measurement_censoring_drained(self) -> dict:
        """Fail fast if early-stop cleanup left live KV ownership behind.

        Transfer calendars, completed ASTRA windows, residence accounting, and
        event histories are intentionally historical and may extend beyond a
        censored measurement cutoff. This audit covers only live ownership or
        an open dependency that could affect a future request: tier entries,
        pending allocation/source records, reclaim claims, prepare locks, and
        external-fabric jobs/windows.
        """
        open_astra_owners = sum(
            len(owners) for owners in self._astra_fabric_inflight.values())
        audit = {
            "idle_entries": len(self.entries),
            "idle_entry_session_ids": sorted(self.entries),
            "ssd_records": len(self.ssd_records),
            "ssd_record_session_ids": sorted(self.ssd_records),
            "ssd_used_bytes": int(self.ssd_used_bytes),
            "pending_hbm_allocations": len(self.pending_hbm_allocations),
            "pending_source_releases": len(self.pending_source_releases),
            "pending_restore_sessions": len(self._pending_restore_sessions),
            "pending_restore_session_ids": sorted(
                self._pending_restore_sessions),
            "pending_demotion_join_sessions": len(
                self._pending_demotion_join_sessions),
            "pending_demotion_join_session_ids": sorted(
                self._pending_demotion_join_sessions),
            "pending_demotion_join_windows": len(
                self._pending_demotion_join_windows),
            "pending_demotion_join_window_session_ids": sorted(
                self._pending_demotion_join_windows),
            "pending_destination_admission_waits": len(
                self._pending_destination_admission_since),
            "pending_destination_admission_session_ids": sorted(
                self._pending_destination_admission_since),
            "active_hbm_reclaim_claims": len(
                self._active_hbm_reclaim_claims),
            "active_hbm_reclaim_instance_ids": sorted(
                self._active_hbm_reclaim_claims),
            "synchronous_prepare_locks": len(self._sync_prepare_locks),
            "synchronous_prepare_sessions": len(
                self._sync_prepare_sessions),
            "direct_fabric_prepare_locks": len(
                self._fabric_prepare_locks),
            "direct_fabric_prepare_sessions": len(
                self._fabric_prepare_sessions),
            "external_fabric_jobs": len(self._external_fabric_by_job),
            "external_fabric_sessions": len(
                self._external_fabric_by_session),
            "external_fabric_outgoing_jobs": len(
                self._external_fabric_outgoing),
            "open_astra_model_windows": open_astra_owners,
            "transient_restore_wait_starts": len(
                self._pending_transient_restore_since),
            "transient_restore_wait_totals": len(
                self._pending_transient_restore_wait_ns),
            "deferred_hbm_demotions": len(
                self._sync_deferred_hbm_demotions),
            "direct_ssd_capacity_reservations": len(
                self._direct_ssd_capacity_reservations),
            "direct_ssd_capacity_reservation_nodes": len(
                self._direct_ssd_capacity_reservation_nodes),
        }
        count_keys = tuple(
            key for key in audit
            if not key.endswith("_ids")
            and not key.endswith("_session_ids")
            and key != "ssd_used_bytes"
        )
        nonzero = {
            key: audit[key] for key in count_keys if int(audit[key]) != 0
        }
        if int(audit["ssd_used_bytes"]) != 0:
            nonzero["ssd_used_bytes"] = int(audit["ssd_used_bytes"])
        audit["passed"] = not nonzero
        audit["live_state"] = nonzero
        if nonzero:
            raise RuntimeError(
                "Measurement censoring left live agentic-KV ownership or "
                f"dependencies: {nonzero}")
        return audit

    @staticmethod
    def _balanced_bytes(total_bytes: int, count: int):
        base, remainder = divmod(total_bytes, count)
        return [base + (1 if index < remainder else 0) for index in range(count)]

    @staticmethod
    def _interval_union_ns(intervals, end_ns: int) -> int:
        clipped = sorted(
            (max(0, start), min(end_ns, end))
            for start, end in intervals
            if end > 0 and start < end_ns and end > start
        )
        if not clipped:
            return 0
        total = 0
        current_start, current_end = clipped[0]
        for start, end in clipped[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                total += current_end - current_start
                current_start, current_end = start, end
        return total + current_end - current_start

    def validate_resource_timeline(self) -> dict:
        """Audit asynchronous cold-copy and ASTRA model timelines.

        ASTRA-to-ASTRA overlap is valid and remains modeled by ASTRA itself.
        PCIe/DRAM DMA and cold direct-fabric copies may overlap model execution.
        Cold peer copies must remain FCFS and non-overlapping with one another
        on their node-shared migration resource.
        """
        migration_conflicts = []
        for resource, intervals in self._resource_intervals.items():
            for previous, current in zip(intervals, intervals[1:]):
                if current[0] < previous[1]:
                    migration_conflicts.append({
                        "resource": resource,
                        "first": previous,
                        "second": current,
                    })
        if migration_conflicts:
            raise RuntimeError(
                "Agentic migration resource reservations overlap: "
                f"{migration_conflicts[:3]}")

        transient_dram_peaks = {}
        transient_dram_conflicts = []
        for node_id, reservations in sorted(
                self._transient_dram_history.items()):
            node_schedulers = [
                scheduler for scheduler in self.schedulers.values()
                if self._node_id(scheduler) == node_id
            ]
            capacity = min(
                scheduler.memory.cpu_mem for scheduler in node_schedulers)
            occupancy = 0
            peak = 0
            events = []
            for reservation in reservations:
                events.append((
                    reservation.start_ns, 1, reservation.bytes,
                    reservation.session_id))
                events.append((
                    reservation.complete_ns, 0, -reservation.bytes,
                    reservation.session_id))
                if reservation.peak_node_committed_bytes > capacity:
                    transient_dram_conflicts.append({
                        "node_id": node_id,
                        "session_id": reservation.session_id,
                        "peak_node_committed_bytes": (
                            reservation.peak_node_committed_bytes),
                        "capacity_bytes": capacity,
                    })
            for event_ns, _, delta, session_id in sorted(events):
                occupancy += delta
                peak = max(peak, occupancy)
                if occupancy > capacity:
                    transient_dram_conflicts.append({
                        "node_id": node_id,
                        "time_ns": event_ns,
                        "session_id": session_id,
                        "transient_bytes": occupancy,
                        "capacity_bytes": capacity,
                    })
            transient_dram_peaks[str(node_id)] = peak
        if transient_dram_conflicts:
            raise RuntimeError(
                "Transient DRAM reservations exceed node capacity: "
                f"{transient_dram_conflicts[:3]}")

        cold_by_resource: Dict[str, list[tuple[int, int, str]]] = {}
        for event in self.events:
            if event.get("event") != "migration_reserve":
                continue
            start_ns = int(event.get("start_ns", 0))
            complete_ns = int(event.get("complete_ns", start_ns))
            if complete_ns <= start_ns:
                continue
            for resource in event.get("resources", ()):
                if resource.endswith(":pd-fabric"):
                    cold_by_resource.setdefault(resource, []).append((
                        start_ns, complete_ns, str(event.get("kind")),
                    ))

        cold_conflicts = []
        for resource, cold_intervals in cold_by_resource.items():
            ordered = sorted(cold_intervals)
            for previous, current in zip(ordered, ordered[1:]):
                if current[0] < previous[1]:
                    cold_conflicts.append({
                        "resource": resource,
                        "first": previous,
                        "second": current,
                    })
        if cold_conflicts:
            raise RuntimeError(
                "Cold direct-fabric FCFS reservations overlap: "
                f"{cold_conflicts[:3]}")

        overlap_by_resource = {
            resource: [
                (start_ns, complete_ns, kind, None)
                for start_ns, complete_ns, kind in cold_intervals
            ]
            for resource, cold_intervals in cold_by_resource.items()
        }
        # Externally executed cold D->P copies use ASTRA's shared network and
        # endpoint model rather than the Python ``pd-fabric`` reservation
        # calendar.  Their completion history still has exact source/target
        # instance IDs, so attribute it to the same per-node audit label used
        # by model windows.  Keep it out of ``cold_by_resource``: multiple
        # external jobs may legitimately interleave inside ASTRA and therefore
        # are not subject to the analytical FCFS non-overlap assertion above.
        for event in self._external_fabric_history:
            start_ns = int(event.get("start_ns", 0))
            complete_ns = int(event.get("complete_ns", start_ns))
            if complete_ns <= start_ns:
                continue
            instance_ids = {
                int(event[instance_key])
                for instance_key in (
                    "source_instance_id", "target_instance_id")
                if event.get(instance_key) is not None
            }
            node_ids = {
                self._node_id(self._scheduler(instance_id))
                for instance_id in instance_ids
            }
            for node_id in node_ids:
                resource = f"node:{node_id}:pd-fabric"
                overlap_by_resource.setdefault(resource, []).append((
                    start_ns,
                    complete_ns,
                    str(event.get("kind")),
                    event.get("job_id"),
                ))

        model_overlaps = []
        for resource, cold_intervals in overlap_by_resource.items():
            calendar = self._astra_fabric_calendar.get(resource, ())
            for cold_start, cold_end, kind, job_id in cold_intervals:
                index = bisect.bisect_right(
                    calendar, (cold_start, math.inf)) - 1
                candidates = []
                if index >= 0:
                    candidates.append(calendar[index])
                if index + 1 < len(calendar):
                    candidates.append(calendar[index + 1])
                conflicting = next((
                    (model_start, model_end)
                    for model_start, model_end in candidates
                    if cold_start < model_end and model_start < cold_end
                ), None)
                if conflicting is None:
                    continue
                model_start, model_end = conflicting
                owner = next(
                    (
                        (instance_id, batch_id)
                        for detail_start, detail_end, instance_id, batch_id in
                        self._astra_fabric_intervals.get(resource, ())
                        if cold_start < detail_end
                        and detail_start < cold_end
                    ),
                    (None, None),
                )
                model_overlaps.append({
                    "resource": resource,
                    "kind": kind,
                    "cold_start_ns": cold_start,
                    "cold_complete_ns": cold_end,
                    "model_start_ns": model_start,
                    "model_complete_ns": model_end,
                    "instance_id": owner[0],
                    "batch_id": owner[1],
                    "external_fabric_job_id": job_id,
                })
        return {
            "mode": (
                "astra_shared_fabric_owner_ready_barrier"
                if self._external_fabric_enabled else
                "asynchronous_owner_ready_barrier"
                if self.config.pd_peer_transfer_mode == "direct-fabric"
                else "separate_cpu_staged_resources"
            ),
            "astra_window_count": sum(
                len(intervals)
                for intervals in self._astra_fabric_intervals.values()),
            "cold_direct_fabric_interval_count": sum(
                len(intervals) for intervals in cold_by_resource.values())
                + len(self._external_fabric_history),
            "allowed_model_overlap_count": len(model_overlaps),
            "allowed_model_overlaps": model_overlaps,
            "forbidden_overlap_count": 0,
            "migration_resource_overlap_count": 0,
            "migration_resource_interval_count": sum(
                len(intervals)
                for intervals in self._resource_intervals.values()),
            "transient_dram_capacity_violation_count": 0,
            "transient_dram_peak_bytes_by_node": transient_dram_peaks,
            "transient_dram_reservation_count": sum(
                len(reservations)
                for reservations in self._transient_dram_history.values()),
            "cold_peer_fcfs_overlap_count": 0,
            "open_astra_window_count": sum(
                len(owners)
                for owners in self._astra_fabric_inflight.values()),
            "pending_direct_fabric_prepare_locks": len(
                self._fabric_prepare_locks),
            # Exact per-node attribution is not exported by ASTRA. The model
            # completion already includes any contention, but this audit does
            # not claim that a particular completed batch was extended.
            "current_batch_latency_extended_by_cold_copy": False,
            "shared_fabric_contention_may_extend_model_communication": bool(
                self._external_fabric_enabled),
            "future_fabric_dispatch_is_gated": False,
            "pcie_dram_dma_may_overlap_model_execution": True,
            "granularity_limit": (
                "Cold peer chunks and Chakra communication share ASTRA topology "
                "links plus physical endpoint ingress/egress arbiters. Transfer "
                "completion gates only the owner request; unrelated continuous "
                "batches remain dispatchable. Contention is interleaved at the "
                "configured background chunk size."
                if self._external_fabric_enabled else
                "Cold peer copies use the Python analytical migration calendar; "
                "cold-copy completion gates only the owner request."
            ),
            "external_fabric_authority": self._external_fabric_authority,
            "external_fabric_pending_jobs": len(
                self._external_fabric_by_job),
        }

    def transfer_tail_at(self, cutoff_ns: int) -> dict:
        """Describe reservations issued by the cutoff but finishing later.

        Async swap-out is allowed to outlive the request measurement window;
        silently folding that service into makespan or pretending it vanished
        are both misleading. This audit distinguishes transfers already in
        service from jobs queued to start after the cutoff and reports their
        remaining demand without advancing the simulated clock.
        """
        cutoff_ns = int(cutoff_ns)
        active = []
        queued = []
        by_kind: Dict[str, dict] = {}
        for event in self.events:
            event_kind = event.get("event")
            if event_kind not in {"migration_reserve", "migration_cancel"}:
                continue
            issued_ns = int(event.get(
                "arrival_ns", event.get("time_ns", 0)))
            start_ns = int(event.get("start_ns", issued_ns))
            complete_ns = int(event.get("complete_ns", start_ns))
            if (event_kind == "migration_cancel"
                    and int(event.get("active_ns", 0)) <= 0):
                continue
            if issued_ns > cutoff_ns or complete_ns <= cutoff_ns:
                continue
            row = {
                "session_id": event.get("session_id"),
                "kind": str(event.get("kind") or "unknown"),
                "foreground": bool(event.get("foreground", False)),
                "bytes": int(event.get("bytes", 0)),
                "issued_ns": issued_ns,
                "start_ns": start_ns,
                "complete_ns": complete_ns,
                "remaining_queue_ns": max(0, start_ns - cutoff_ns),
                "remaining_service_ns": (
                    complete_ns - max(cutoff_ns, start_ns)),
                "tail_ns": complete_ns - cutoff_ns,
                "resources": list(event.get("resources", ())),
            }
            destination = active if start_ns <= cutoff_ns else queued
            destination.append(row)
            cell = by_kind.setdefault(row["kind"], {
                "jobs": 0,
                "bytes": 0,
                "remaining_queue_ns": 0,
                "remaining_service_ns": 0,
                "max_tail_ns": 0,
            })
            cell["jobs"] += 1
            cell["bytes"] += row["bytes"]
            cell["remaining_queue_ns"] += row["remaining_queue_ns"]
            cell["remaining_service_ns"] += row["remaining_service_ns"]
            cell["max_tail_ns"] = max(cell["max_tail_ns"], row["tail_ns"])
        rows = active + queued
        return {
            "cutoff_ns": cutoff_ns,
            "active_service_jobs": len(active),
            "queued_not_started_jobs": len(queued),
            "outstanding_jobs": len(rows),
            "foreground_jobs": sum(row["foreground"] for row in rows),
            "background_jobs": sum(not row["foreground"] for row in rows),
            "outstanding_bytes": sum(row["bytes"] for row in rows),
            "remaining_queue_ns_membership_sum": sum(
                row["remaining_queue_ns"] for row in rows),
            "remaining_service_ns_membership_sum": sum(
                row["remaining_service_ns"] for row in rows),
            "max_tail_ns": max(
                (row["tail_ns"] for row in rows), default=0),
            "by_kind": dict(sorted(by_kind.items())),
            "active": active,
            "queued": queued,
            "semantics": (
                "Issued analytical DMA reservations beyond the measurement "
                "cutoff are right-censored. Their remaining service is not "
                "added to request latency or the throughput denominator."
            ),
        }

    def _queue_recompute_accounting_audit(self) -> dict:
        """Fail closed on partial-prefix decision/accounting divergence."""
        errors = []
        metrics = self.metrics
        evaluations = [
            event for event in self.events
            if event.get("event") == "queue_recompute_evaluate"
        ]
        partials = [
            event for event in self.events
            if event.get("event") == "queue_recompute_partial"
        ]
        zero_restores = [
            event for event in self.events
            if event.get("event") == "queue_recompute_drop"
        ]

        def require(condition: bool, message: str) -> None:
            if not condition:
                errors.append(message)

        require(
            len(evaluations)
            == metrics.queue_recompute_evaluation_attempts,
            "evaluation event/counter mismatch",
        )
        require(
            len(partials)
            == metrics.queue_recompute_partial_restore_decisions,
            "partial event/counter mismatch",
        )
        require(
            len(zero_restores)
            == metrics.queue_recompute_zero_restore_decisions,
            "zero-restore event/counter mismatch",
        )
        require(
            metrics.queue_recompute_drop_decisions
            == metrics.queue_recompute_zero_restore_decisions,
            "legacy drop decisions must equal H=0 decisions",
        )
        require(
            metrics.queue_recompute_full_restore_decisions
            + metrics.queue_recompute_partial_restore_decisions
            + metrics.queue_recompute_zero_restore_decisions
            == metrics.queue_recompute_evaluation_attempts,
            "full/partial/zero decisions do not partition evaluations",
        )
        require(
            metrics.queue_recompute_partial_cpu_decisions
            + metrics.queue_recompute_partial_ssd_decisions
            == metrics.queue_recompute_partial_restore_decisions,
            "CPU/SSD partial decisions do not partition partial restores",
        )
        require(
            metrics.queue_recompute_cpu_drop_decisions
            + metrics.queue_recompute_ssd_drop_decisions
            == metrics.queue_recompute_zero_restore_decisions,
            "CPU/SSD drop decisions do not partition zero restores",
        )
        require(
            metrics.queue_recompute_dropped_bytes
            == metrics.queue_recompute_avoided_restore_bytes
            == metrics.queue_recompute_dropped_suffix_bytes,
            "avoided restore byte aliases diverged",
        )
        require(
            metrics.queue_recompute_dropped_suffix_tokens
            == metrics.queue_recompute_policy_avoidable_tokens,
            "selected suffix tokens diverged from executed policy recompute",
        )

        block_size = int(self.config.block_size)
        for event in partials + zero_restores:
            session_id = event.get("session_id")
            reusable = int(event.get("reusable_tokens_R", -1))
            selected = int(event.get("selected_prefix_tokens_H", -1))
            suffix = int(event.get("dropped_suffix_tokens", -1))
            selected_bytes = int(event.get("selected_restore_bytes", -1))
            suffix_bytes = int(event.get("dropped_suffix_bytes", -1))
            full_bytes = int(event.get("bytes", -1))
            require(
                0 <= selected < reusable,
                f"invalid R/H ordering for session {session_id}",
            )
            require(
                suffix == reusable - selected,
                f"suffix token identity failed for session {session_id}",
            )
            require(
                selected_bytes + suffix_bytes == full_bytes,
                f"suffix byte identity failed for session {session_id}",
            )
            require(
                bool(event.get("severe_gate_pass")),
                f"modified decision bypassed severe gate for {session_id}",
            )
            require(
                bool(event.get("cost_gate_pass")),
                f"modified decision bypassed cost gate for {session_id}",
            )
            require(
                event.get("logical_session_effect") == "none",
                f"modified KV decision changed logical session {session_id}",
            )
            if selected:
                require(
                    selected % block_size == 0,
                    f"partial H is not block aligned for {session_id}",
                )
                require(
                    event.get("selection_scope")
                    == "contiguous_block_aligned_prefix",
                    f"partial selection scope is not prefix-only for {session_id}",
                )
                require(
                    bool(event.get("prefix_projection_available")),
                    f"partial prefix lacks a restore projection for {session_id}",
                )
                require(
                    int(event.get(
                        "prefix_projected_hbm_admission_wait_ns", -1)) == 0,
                    f"partial prefix was not immediately HBM-fit for {session_id}",
                )
                snapshot = event.get("capacity_headroom_snapshot")
                require(
                    isinstance(snapshot, dict)
                    and bool(snapshot.get("feasible")),
                    f"partial prefix lacks feasible P/D snapshot for {session_id}",
                )
                require(
                    int(event.get(
                        "physical_source_bytes_pinned_until_dma_complete", 0))
                    > 0,
                    f"partial source pin is missing for {session_id}",
                )
            else:
                require(
                    selected_bytes == 0,
                    f"H=0 decision transferred bytes for {session_id}",
                )

        audit = {
            "passed": not errors,
            "errors": errors,
            "evaluation_events": len(evaluations),
            "partial_events": len(partials),
            "zero_restore_events": len(zero_restores),
            "block_size_tokens": block_size,
            "logical_session_drop_count": 0,
            "headroom_semantics": "causal_snapshot_not_reservation",
        }
        if errors:
            raise RuntimeError(
                "Queue-recompute partial-prefix accounting failed: "
                + "; ".join(errors))
        return audit

    def _pd_chunk_accounting_audit(self) -> dict:
        """Fail closed when incremental P/D event totals diverge."""
        metrics = self.metrics
        events = [
            event for event in self.events
            if event.get("event") == "pd_chunk_admission"
        ]
        cancelled_events = [
            event for event in self.events
            if event.get("event") == (
                "pd_chunk_admission_cancelled_for_active_prefill_"
                "recompute")
        ]
        first_chunks = [
            event for event in events if event.get("first_chunk", False)
        ]
        launch_events = [
            event for event in self.events
            if event.get("event") == "pd_launch_admission"
        ]
        prefill_events = [
            event for event in self.events
            if event.get("event") == "pd_prefill_active_admission"
        ]
        decode_events = [
            event for event in self.events
            if event.get("event") == "pd_decode_receive_admission"
        ]
        joined = [
            event for event in first_chunks
            if event.get("capacity_headroom_snapshot") is not None
        ]
        feasible = [
            event for event in joined
            if event.get("capacity_snapshot_feasible", False)
        ]
        feasible_waiting = [
            event for event in feasible
            if event.get("snapshot_feasible_but_actual_waited", False)
        ]
        checks = {
            "event_count_matches": (
                len(events) == metrics.pd_chunk_admissions),
            "first_chunk_events_match_launches": (
                len(first_chunks)
                == metrics.pd_launch_admissions
                == metrics.pd_prefill_admissions
                == metrics.pd_decode_receive_admissions),
            "first_chunk_legacy_event_partition_matches": (
                len(first_chunks)
                == len(launch_events)
                == len(prefill_events)
                == len(decode_events)),
            "token_sum_matches": (
                sum(int(event["chunk_tokens"]) for event in events)
                == metrics.pd_chunk_admitted_tokens),
            "prefill_byte_sum_matches": (
                sum(int(event["prefill_delta_bytes"]) for event in events)
                == metrics.pd_chunk_prefill_reserved_bytes),
            "decode_byte_sum_matches": (
                sum(int(event["decode_delta_bytes"]) for event in events)
                == metrics.pd_chunk_decode_reserved_bytes),
            "wait_sum_matches": (
                sum(int(event["wait_ns"]) for event in events)
                == metrics.pd_chunk_admission_wait_ns),
            "critical_wait_sum_matches": (
                sum(int(event["critical_wait_after_restore_ns"])
                    for event in events)
                == metrics.pd_chunk_admission_critical_wait_ns),
            "waiting_count_matches": (
                sum(int(event["wait_ns"]) > 0 for event in events)
                == metrics.pd_chunk_waiting_admissions),
            "cancelled_event_count_matches": (
                len(cancelled_events)
                == metrics.pd_chunk_cancelled_admissions),
            "cancelled_wait_sum_matches": (
                sum(int(event["wait_ns"]) for event in cancelled_events)
                == metrics.pd_chunk_cancelled_admission_wait_ns),
            "cancelled_critical_wait_sum_matches": (
                sum(int(event["critical_wait_after_restore_ns"])
                    for event in cancelled_events)
                == metrics.pd_chunk_cancelled_admission_critical_wait_ns),
            "cancelled_waiting_count_matches": (
                sum(int(event["wait_ns"]) > 0
                    for event in cancelled_events)
                == metrics.pd_chunk_cancelled_waiting_admissions),
            "attempt_wait_partition_matches": (
                sum(int(event["wait_ns"])
                    for event in events + cancelled_events)
                == metrics.pd_chunk_admission_wait_ns
                + metrics.pd_chunk_cancelled_admission_wait_ns),
            "attempt_critical_wait_partition_matches": (
                sum(int(event["critical_wait_after_restore_ns"])
                    for event in events + cancelled_events)
                == metrics.pd_chunk_admission_critical_wait_ns
                + metrics.pd_chunk_cancelled_admission_critical_wait_ns),
            "snapshot_join_count_matches": (
                len(joined)
                == metrics.pd_chunk_snapshot_joined_admissions),
            "snapshot_feasible_count_matches": (
                len(feasible)
                == metrics.pd_chunk_snapshot_feasible_admissions),
            "snapshot_feasible_wait_count_matches": (
                len(feasible_waiting)
                == metrics.pd_chunk_snapshot_feasible_waiting_admissions),
            "snapshot_feasible_wait_sum_matches": (
                sum(int(event["wait_ns"]) for event in feasible_waiting)
                == metrics.pd_chunk_snapshot_feasible_wait_ns),
            "snapshot_join_is_first_chunk_only": all(
                event.get("capacity_headroom_snapshot") is None
                for event in events if not event.get("first_chunk", False)),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                "Incremental P/D chunk accounting failed: "
                + ", ".join(failed))
        return {
            "status": "ok",
            "chunk_admissions": len(events),
            "cancelled_chunk_admissions": len(cancelled_events),
            "first_chunk_admissions": len(first_chunks),
            "snapshot_joined_first_chunks": len(joined),
            "snapshot_feasible_first_chunks": len(feasible),
            "snapshot_feasible_waiting_first_chunks": len(
                feasible_waiting),
            "checks": checks,
            "semantics": (
                "queue policy headroom is a causal snapshot; each chunk "
                "event is the later authoritative atomic P/D claim"),
        }

    def _pd_active_prefill_recompute_accounting_audit(self) -> dict:
        """Reconcile active-prefill replay events and restored-hit loss."""
        events = [
            event for event in self.events
            if event.get("event") == "pd_active_prefill_recompute_preempt"
        ]
        errors = []
        generation_by_request = {}
        cumulative_tokens_by_request = {}
        cumulative_restored_by_request = {}
        event_discarded_tokens = 0
        event_restored_discarded_tokens = 0
        for index, event in enumerate(events):
            prefix = f"pd_active_prefill_recompute_preempt[{index}]"
            required = (
                "request_id", "discarded_tokens",
                "restored_hit_tokens_discarded",
                "cumulative_active_prefill_recompute_tokens",
                "cumulative_restored_hit_tokens_discarded",
                "old_active_prefill_recompute_generation",
                "new_active_prefill_recompute_generation",
            )
            missing = [key for key in required if key not in event]
            if missing:
                errors.append(f"{prefix}: missing={missing}")
                continue
            invalid = {
                key: event[key]
                for key in required
                if (not isinstance(event[key], int)
                    or isinstance(event[key], bool)
                    or event[key] < 0)
            }
            if invalid:
                errors.append(f"{prefix}: invalid integers={invalid}")
                continue
            values = {key: event[key] for key in required}
            request_id = values["request_id"]
            old_generation = values[
                "old_active_prefill_recompute_generation"]
            new_generation = values[
                "new_active_prefill_recompute_generation"]
            expected_generation = generation_by_request.get(request_id, 0)
            if (old_generation != expected_generation
                    or new_generation != old_generation + 1):
                errors.append(
                    f"{prefix}: generation {old_generation}->{new_generation}, "
                    f"expected={expected_generation}->{expected_generation + 1}")
            discarded = values["discarded_tokens"]
            restored_delta = values["restored_hit_tokens_discarded"]
            if restored_delta > discarded or (
                    old_generation > 0 and restored_delta != 0):
                errors.append(
                    f"{prefix}: restored-hit delta={restored_delta}, "
                    f"discarded={discarded}, generation={old_generation}")
            expected_tokens = (
                cumulative_tokens_by_request.get(request_id, 0) + discarded)
            expected_restored = (
                cumulative_restored_by_request.get(request_id, 0)
                + restored_delta)
            if (values["cumulative_active_prefill_recompute_tokens"]
                    != expected_tokens):
                errors.append(f"{prefix}: cumulative replay tokens diverge")
            if (values["cumulative_restored_hit_tokens_discarded"]
                    != expected_restored):
                errors.append(f"{prefix}: cumulative restored hits diverge")
            generation_by_request[request_id] = new_generation
            cumulative_tokens_by_request[request_id] = expected_tokens
            cumulative_restored_by_request[request_id] = expected_restored
            event_discarded_tokens += discarded
            event_restored_discarded_tokens += restored_delta

        metrics = self.metrics
        checks = {
            "event_count_matches": (
                len(events)
                == metrics.pd_active_prefill_recompute_preemptions),
            "discarded_token_sum_matches": (
                event_discarded_tokens
                == metrics.pd_active_prefill_recompute_tokens),
            "restored_hit_discard_sum_matches": (
                event_restored_discarded_tokens
                == metrics
                .agentic_kv_restored_tokens_discarded_by_active_prefill_recompute),
        }
        errors.extend(
            name for name, passed in checks.items() if not passed)
        if errors:
            raise RuntimeError(
                "P/D active-prefill recompute accounting failed: "
                + "; ".join(errors))
        return {
            "status": "ok",
            "preemptions": len(events),
            "discarded_tokens": event_discarded_tokens,
            "restored_hit_tokens_discarded": (
                event_restored_discarded_tokens),
            "checks": checks,
            "semantics": (
                "physical resume source is retained; original restored hit "
                "tokens are charged exactly once at first active-prefill "
                "preemption"),
        }

    def summary(self, simulated_duration_ns: int = 0, dataset: Optional[str] = None,
                run_id: Optional[str] = None,
                measurement_censored: bool = False) -> dict:
        if simulated_duration_ns > 0:
            for entry in self.entries.values():
                self._account_residence(entry, simulated_duration_ns)
            for record in self.ssd_records.values():
                self._account_ssd_record(record, simulated_duration_ns)
        self._update_idle_peaks()
        resource_timeline_audit = self.validate_resource_timeline()
        queue_recompute_audit = self._queue_recompute_accounting_audit()
        pd_chunk_audit = self._pd_chunk_accounting_audit()
        pd_active_prefill_audit = (
            self._pd_active_prefill_recompute_accounting_audit())
        transfer_tail = self.transfer_tail_at(simulated_duration_ns)
        transfer_tail["measurement_censored"] = bool(measurement_censored)
        causal_restore_intervals = (
            self._source_demotion_join_intervals
            + self._destination_admission_intervals
            + self._censored_source_demotion_join_intervals
            + self._censored_destination_admission_intervals
            + self._async_owner_barrier_intervals
            if self.async_decode_join_enabled
            else self._source_demotion_join_intervals
            + self._destination_admission_intervals
            + self._censored_source_demotion_join_intervals
            + self._censored_destination_admission_intervals
            + self._critical_restore_intervals
        )
        restore_exposure_ns = self._interval_union_ns(
            causal_restore_intervals, int(simulated_duration_ns))
        exposure_fraction = (
            restore_exposure_ns / simulated_duration_ns
            if simulated_duration_ns > 0 else None)
        causal_restore_ns = self.metrics.source_demotion_join_wait_ns + (
            self.metrics.critical_restore_hbm_admission_wait_ns
            + self.metrics.async_restore_owner_barrier_ns
            if self.async_decode_join_enabled
            else self.metrics.critical_restore_ns
        )
        request_fraction = (
            causal_restore_ns
            / self.metrics.total_request_latency_ns
            if self.metrics.total_request_latency_ns > 0 else None)
        recompute_compute_fraction = (
            self.metrics.recompute_model_compute_ns
            / self.metrics.total_model_compute_ns
            if self.metrics.total_model_compute_ns > 0 else None)
        prompt_denominator = (
            self.metrics.total_prompt_tokens
            if self.metrics.total_prompt_tokens > 0
            else self.metrics.resumed_prompt_tokens)
        recompute_token_fraction = (
            self.metrics.recompute_tokens / prompt_denominator
            if prompt_denominator > 0 else None)
        executed_prefill_tokens = max(
            0, prompt_denominator - self.metrics.cache_hit_tokens)
        policy_recompute_prompt_fraction = (
            self.metrics.policy_avoidable_recompute_tokens / prompt_denominator
            if prompt_denominator > 0 else None)
        policy_recompute_executed_fraction = (
            self.metrics.policy_avoidable_recompute_tokens
            / executed_prefill_tokens
            if executed_prefill_tokens > 0 else None)
        barrier_horizon_ns = (
            int(simulated_duration_ns)
            if simulated_duration_ns > 0 else
            max(
                (
                    end_ns
                    for barriers in self._sync_engine_barriers.values()
                    for _, end_ns, _, _, _ in barriers
                ),
                default=0,
            )
        )
        reservation_union_by_instance = {
            str(instance_id): self._interval_union_ns(
                [(start_ns, end_ns) for start_ns, end_ns, _, _, _ in barriers],
                barrier_horizon_ns,
            )
            for instance_id, barriers in sorted(
                self._sync_engine_barriers.items())
        }
        reservation_union_by_direction = {
            direction: sum(
                self._interval_union_ns(
                    [
                        (start_ns, end_ns)
                        for start_ns, end_ns, item_direction, _, _ in barriers
                        if item_direction == direction
                    ],
                    barrier_horizon_ns,
                )
                for barriers in self._sync_engine_barriers.values()
            )
            for direction in ("in", "out")
        }
        aggregate_reservation_barrier_ns = sum(
            reservation_union_by_instance.values())
        global_reservation_barrier_ns = self._interval_union_ns(
            [
                (start_ns, end_ns)
                for barriers in self._sync_engine_barriers.values()
                for start_ns, end_ns, _, _, _ in barriers
            ],
            barrier_horizon_ns,
        )
        exposed_union_by_instance = {
            str(instance_id): self._interval_union_ns(
                [(start_ns, end_ns) for start_ns, end_ns, _ in barriers],
                barrier_horizon_ns,
            )
            for instance_id, barriers in sorted(
                self._sync_exposed_barriers.items())
        }
        exposed_union_by_direction = {
            direction: sum(
                self._interval_union_ns(
                    [
                        (start_ns, end_ns)
                        for start_ns, end_ns, directions in barriers
                        if direction in directions
                    ],
                    barrier_horizon_ns,
                )
                for barriers in self._sync_exposed_barriers.values()
            )
            for direction in ("in", "out")
        }
        aggregate_exposed_engine_wait_ns = sum(
            exposed_union_by_instance.values())
        global_exposed_engine_wait_ns = self._interval_union_ns(
            [
                (start_ns, end_ns)
                for barriers in self._sync_exposed_barriers.values()
                for start_ns, end_ns, _ in barriers
            ],
            barrier_horizon_ns,
        )
        aggregate_ready_victim_wait_ns = sum(
            self._interval_union_ns(intervals, barrier_horizon_ns)
            for intervals in self._sync_ready_victim_intervals.values()
        )
        iteration_execution_by_instance = {
            str(instance_id): self._interval_union_ns(
                intervals, barrier_horizon_ns)
            for instance_id, intervals in sorted(
                self._model_iteration_intervals.items())
        }
        aggregate_iteration_execution_ns = sum(
            iteration_execution_by_instance.values())
        global_model_execution_ns = self._interval_union_ns(
            [
                interval
                for intervals in self._model_iteration_intervals.values()
                for interval in intervals
            ],
            barrier_horizon_ns,
        )
        transfer_intervals = []
        for event in self.events:
            if (event.get("event") == "migration_reserve"
                    and event.get("complete_ns", 0)
                    > event.get("start_ns", 0)):
                transfer_intervals.append((
                    int(event["start_ns"]), int(event["complete_ns"])))
            elif (event.get("event") == "migration_cancel"
                    and event.get("active_ns", 0) > 0):
                start_ns = int(event["start_ns"])
                transfer_intervals.append((
                    start_ns, start_ns + int(event["active_ns"])))
        global_transfer_execution_ns = self._interval_union_ns(
            transfer_intervals, barrier_horizon_ns)
        global_model_or_transfer_ns = self._interval_union_ns(
            [
                interval
                for intervals in self._model_iteration_intervals.values()
                for interval in intervals
            ] + transfer_intervals,
            barrier_horizon_ns,
        )
        iteration_active_ns = (
            aggregate_iteration_execution_ns
            + aggregate_exposed_engine_wait_ns
        )
        devices_per_node = self.config.ssd_num_devices
        ssd_node_count = len(self._ssd_node_ids)
        total_ssd_devices = devices_per_node * ssd_node_count
        device_writes = self._balanced_bytes(
            self.metrics.ssd_host_write_bytes, total_ssd_devices)
        device_reads = self._balanced_bytes(
            self.metrics.ssd_host_read_bytes, total_ssd_devices)
        devices = []
        flat_index = 0
        for node_id in self._ssd_node_ids:
            for device_index in range(devices_per_node):
                device_id = (
                    f"ssd{device_index}"
                    if ssd_node_count == 1
                    else f"node{node_id}:ssd{device_index}"
                )
                devices.append({
                    "device_id": device_id,
                    "node_id": int(node_id),
                    "host_write_bytes": device_writes[flat_index],
                    "host_read_bytes": device_reads[flat_index],
                })
                flat_index += 1
        ssd_used_bytes_by_node = {
            str(node_id): self._ssd_used_bytes_on_node(node_id)
            for node_id in self._ssd_node_ids
        }
        ssd_reserved_bytes_by_node = {
            str(node_id): self._ssd_reserved_bytes(node_id=node_id)
            for node_id in self._ssd_node_ids
        }
        queue_provider_provenance = {}
        for instance_id, provider in sorted(
                self._queue_recompute_latency_providers.items()):
            metadata = dict(provider.metadata())
            queue_provider_provenance[str(instance_id)] = {
                key: metadata.get(key)
                for key in (
                    "name", "scope", "model", "hardware", "tp", "ep",
                    "dtype", "band", "target_config_sha256",
                    "source_sha256", "producer_source_sha256",
                )
            }
        return {
            "schema_version": 20,
            "run_id": run_id or "unknown",
            "dataset": dataset,
            "simulated_duration_ns": int(simulated_duration_ns),
            "trace_period_ns": int(simulated_duration_ns),
            "policy": self.config.policy,
            "config": asdict(self.config),
            "latency_model": {
                "contention": "trace_driven_resource_queue",
                "queue_policy": self.config.io_queue_policy,
                "pd_peer_transfer_mode": (
                    self.config.pd_peer_transfer_mode),
                "pd_peer_service_formula": (
                    "ASTRA shared-link and endpoint arbitration over exact "
                    "per-TP-lane bytes; completion supplied by callback"
                    if self._external_fabric_enabled else
                    "fixed + max(source_per_rank_bytes, "
                    "target_per_rank_bytes) / pd_peer_bandwidth"
                    if self.config.pd_peer_transfer_mode == "direct-fabric"
                    else (
                        "fixed + max(source_per_rank_bytes / pcie_bandwidth, "
                        "target_per_rank_bytes / pcie_bandwidth, "
                        "2 * total_bytes / cpu_bandwidth)"
                    )
                ),
                "storage_path": (
                    "gpu_ssd_direct_write_host_dram_staged_read_analytical"
                    if self.config.policy == "hbm_ssd_direct"
                    else "host_dram_staged_analytical"),
                "storage_service_formula": (
                    "write: fixed + max(per_rank_bytes / "
                    "pcie_bandwidth, total_bytes / ssd_bandwidth); read: "
                    "max(ssd_media, host_dram_write) stage + "
                    "max(host_dram_read, per_rank_pcie) stage"
                    if self.config.policy == "hbm_ssd_direct"
                    else (
                        "max(ssd_media, host_dram_write) stage + "
                        "max(host_dram_read, per_rank_pcie) stage"
                    )),
                "note": (
                    "Transfers gang-schedule per-rank PCIe/copy engines, "
                    "node-shared DRAM, and SSD read/write queues. Foreground "
                    "restores are measured on the critical path; running "
                    "capacity-triggered background copies are durable and "
                    "non-preemptive; a returning owner joins the immutable "
                    "commit. Only optional TTL-sensitivity copies may cancel "
                    "before commit. SSD restore "
                    "bounce objects reserve node DRAM capacity from media "
                    "start through H2D completion; tiered pressure cascades "
                    "CPU-cache LRU victims to SSD."
                    + (
                        " Same-node HBM-resident D->P copies use per-rank peer-copy "
                        "engines plus the node P/D fabric; they do not "
                        "reserve PCIe or host DRAM."
                        if self.config.pd_peer_transfer_mode == "direct-fabric"
                        else (
                            " Same-node HBM-resident D->P copies are CPU-staged and "
                            "reserve both endpoint PCIe engines plus host "
                            "DRAM."
                        )
                    )
                    + " For hbm_ssd_direct, swap-out retains an analytical "
                    "direct GPU-storage max-path model, while every swap-in "
                    "uses two dependency-linked reservations: SSD-to-transient-host "
                    "and host-to-HBM. The former has no GPU PCIe resource; "
                    "the latter has no SSD resource. It is not a measured "
                    "GDS implementation. Other SSD policies use the same "
                    "two-stage restore path."
                    " CPU/SSD migration queues remain Python analytical "
                    "calendars. In external direct-fabric mode, cold "
                    "HBM-resident D->P chunks use the same ASTRA links and "
                    "physical endpoint arbiters as normal P->D and TP/EP "
                    "traffic. They remain asynchronous with unrelated model "
                    "execution, but shared communication contention can extend "
                    "either flow. The exact ASTRA completion gates only the "
                    "returning owner. CPU/SSD restores bypass D HBM and load "
                    "the selected P HBM directly."
                    + (
                        " GPU-facing cold swaps additionally gate affected "
                        "model engines before the next batch dispatch."
                        if self.synchronous_swap_enabled else ""
                    )
                ),
            },
            "online_resource_bridge": resource_timeline_audit,
            "external_fabric": {
                "enabled": self._external_fabric_enabled,
                "authority": self._external_fabric_authority,
                "issued_jobs": self.metrics.external_fabric_jobs_issued,
                "completed_jobs": self.metrics.external_fabric_jobs_completed,
                "censored_jobs": self.metrics.external_fabric_jobs_censored,
                "censored_lane_bytes": (
                    self.metrics.external_fabric_censored_lane_bytes),
                "pending_jobs": len(self._external_fabric_by_job),
                "pending_sessions": sorted(
                    self._external_fabric_by_session),
                "completed_intervals": list(
                    self._external_fabric_history),
            },
            "measurement_cutoff_dma_tail": transfer_tail,
            "totals": asdict(self.metrics),
            "pd_chunk_accounting": pd_chunk_audit,
            "pd_active_prefill_recompute_accounting": (
                pd_active_prefill_audit),
            "host_dram_staging": {
                "capacity_scope": "node_shared_with_persistent_cpu_cache",
                "allocation_granularity": (
                    "selected_restore_prefix_object_block_aligned_when_partial"),
                "lifetime": "ssd_media_start_through_h2d_completion",
                "reservation_count": (
                    self.metrics.transient_dram_reservations),
                "reservation_bytes_membership_sum": (
                    self.metrics.transient_dram_reserved_bytes),
                "byte_ns_membership_sum": (
                    self.metrics.transient_dram_byte_ns),
                "aggregate_explicit_capacity_wait_ns": (
                    self.metrics.transient_dram_capacity_wait_ns),
                "aggregate_pressure_stall_upper_bound_ns": (
                    self.metrics.transient_dram_pressure_stall_ns),
                "capacity_deferrals": (
                    self.metrics.transient_dram_capacity_deferrals),
                "oversize_restore_failures": (
                    self.metrics.transient_dram_capacity_oversize),
                "pending_capacity_wait_sessions": len(
                    self._pending_transient_restore_since),
                "censored_capacity_wait_count": len(
                    self._censored_transient_restore_audits),
                "censored_capacity_wait_elapsed_ns_membership_sum": sum(
                    row["elapsed_ns"]
                    for row in self._censored_transient_restore_audits),
                "censored_capacity_wait_audits": list(
                    self._censored_transient_restore_audits),
                "cpu_lru_evictions": (
                    self.metrics.transient_dram_cpu_lru_evictions),
                "peak_transient_bytes": (
                    self.metrics.peak_transient_dram_bytes),
                "peak_persistent_plus_transient_bytes": (
                    self.metrics.peak_cpu_committed_plus_transient_bytes),
                "scope_note": (
                    "Explicit capacity wait is admission postponement beyond "
                    "the HBM-ready stage arrival. Pressure stall is an upper "
                    "bound from that arrival to SSD media start and may "
                    "overlap resource-queue delay caused by a CPU LRU write."
                ),
            },
            "synchronous_swap": {
                "mode": self.config.swap_execution_mode,
                "enabled": self.synchronous_swap_enabled,
                "barrier_placement": (
                    "pre_dispatch_engine_gate"
                    if self.synchronous_swap_enabled else "none"),
                "swap_in_and_swap_out_block": (
                    self.synchronous_swap_enabled),
                "cpu_to_ssd_blocks_model": False,
                "pure_pd_peer_copy_blocks_model": False,
                "pipelined_tp_critical_path": True,
                "reservation_barrier_union_ns_by_instance": (
                    reservation_union_by_instance),
                "aggregate_reservation_barrier_union_ns": (
                    aggregate_reservation_barrier_ns),
                "swap_in_reservation_barrier_union_ns": (
                    reservation_union_by_direction["in"]),
                "swap_out_reservation_barrier_union_ns": (
                    reservation_union_by_direction["out"]),
                "global_wall_reservation_barrier_union_ns": (
                    global_reservation_barrier_ns),
                "global_wall_reservation_barrier_fraction_of_makespan": (
                    global_reservation_barrier_ns / simulated_duration_ns
                    if simulated_duration_ns > 0 else None),
                "exposed_engine_wait_ns_by_instance": (
                    exposed_union_by_instance),
                "aggregate_exposed_engine_wait_ns": (
                    aggregate_exposed_engine_wait_ns),
                "swap_in_exposed_engine_wait_ns": (
                    exposed_union_by_direction["in"]),
                "swap_out_exposed_engine_wait_ns": (
                    exposed_union_by_direction["out"]),
                "directional_exposed_wait_is_non_additive": True,
                "global_wall_exposed_engine_wait_ns": (
                    global_exposed_engine_wait_ns),
                "global_wall_exposed_engine_wait_fraction_of_makespan": (
                    global_exposed_engine_wait_ns / simulated_duration_ns
                    if simulated_duration_ns > 0 else None),
                "model_iteration_execution_union_ns_by_instance": (
                    iteration_execution_by_instance),
                "aggregate_model_iteration_execution_union_ns": (
                    aggregate_iteration_execution_ns),
                "model_iteration_execution_membership_sum_ns": (
                    self.metrics.agentic_model_iteration_execution_ns),
                "batch_blocking_swap_wait_fraction_of_model_iteration_time": (
                    aggregate_exposed_engine_wait_ns / iteration_active_ns
                    if iteration_active_ns > 0 else None),
                "blocked_iteration_count": (
                    self.metrics.sync_swap_blocked_iterations),
                "blocked_batch_memberships": (
                    self.metrics.sync_swap_blocked_batch_memberships),
                "ready_victim_memberships_at_dispatch_block": (
                    self.metrics.sync_swap_ready_victim_memberships),
                "unique_ready_victim_requests": len(
                    self._sync_unique_ready_victims),
                "aggregate_ready_victim_wait_ns": (
                    aggregate_ready_victim_wait_ns),
                "pending_prepare_locks": len(self._sync_prepare_locks),
                "pending_prepare_pinned_sessions": len(
                    self._capacity_pinned_sessions()),
                "pending_capacity_restore_sessions": len(
                    self._pending_restore_sessions),
                "pending_source_demotion_join_sessions": len(
                    self._pending_demotion_join_sessions),
                "same_batch_membership_frozen_before_restore": False,
                "scope_note": (
                    (
                        "The scheduler realizes synchronous swap as an engine "
                        "gate and associates a contiguous gate with the batch "
                        "dispatched at its completion. This causally delays "
                        "runnable HBM work and queue progress, but does not "
                        "freeze the selected member list before restore. Raw "
                        "reservation unions include overlap with running or "
                        "idle engines; exposed waits include a foreground "
                        "owner's critical barrier plus failed collateral "
                        "dispatch opportunities and drive the blocking "
                        "fraction. The metric is therefore an engine-barrier "
                        "baseline, not a claim that the public InferCept CUDA "
                        "event stream was replayed."
                    ) if self.synchronous_swap_enabled else (
                        "Async modes never gate unrelated model engines. "
                        "async-pre-admission delays the owner request until "
                        "restore completes; async-decode-join permits an "
                        "idealized fresh-prefill region and gates only its "
                        "final prompt token/first output."
                    )
                ),
            },
            "asynchronous_restore": {
                "mode": self.config.swap_execution_mode,
                "swap_out_blocks_model": False,
                "swap_in_blocks_other_requests": False,
                "decode_requires_restore_complete": True,
                "overlap_model": (
                    "ideal_fresh_prefill_except_final_prompt_token"
                    if self.async_decode_join_enabled else "none"),
                "aggregate_swap_in_gross_ns": (
                    self.metrics.async_restore_gross_ns),
                "aggregate_prefill_execution_overlap_ns": (
                    self.metrics.async_restore_compute_overlap_ns),
                "aggregate_owner_decode_barrier_ns": (
                    self.metrics.async_restore_owner_barrier_ns),
                "aggregate_pre_admission_wait_ns": (
                    self.metrics.critical_restore_hbm_admission_wait_ns),
                "aggregate_destination_hbm_admission_wait_ns": (
                    self.metrics.critical_restore_hbm_admission_wait_ns),
                "aggregate_lower_tier_capacity_and_queue_wait_ns": (
                    self.metrics.critical_restore_queue_wait_ns),
                "aggregate_other_hidden_ns": max(
                    0,
                    self.metrics.async_restore_gross_ns
                    - self.metrics.critical_restore_hbm_admission_wait_ns
                    - self.metrics.async_restore_compute_overlap_ns
                    - self.metrics.async_restore_owner_barrier_ns,
                ),
                "by_source_and_return_gap_type": {
                    source: {
                        gap_type: dict(sorted(cell.items()))
                        for gap_type, cell in sorted(gaps.items())
                    }
                    for source, gaps in sorted(
                        self._async_restore_by_source_and_return.items())
                },
                "accuracy_note": (
                    "Bulk restore and the pre-attention-safe fresh-prompt "
                    "region are represented by a coarse ideal overlap. A "
                    "physical full-suffix overlap requires layerwise KV "
                    "streaming; compare async-pre-admission as the serial "
                    "bound."
                ),
            },
            "observed_load_activity": {
                "window_ns": int(simulated_duration_ns),
                "model_execution_union_ns_by_instance": (
                    iteration_execution_by_instance),
                "global_any_model_execution_ns": global_model_execution_ns,
                "global_all_model_engines_idle_ns": max(
                    0, int(simulated_duration_ns) - global_model_execution_ns),
                "global_transfer_execution_ns": global_transfer_execution_ns,
                "global_model_or_transfer_execution_ns": (
                    global_model_or_transfer_ns),
                "migration_only_no_model_execution_ns": max(
                    0,
                    global_model_or_transfer_ns - global_model_execution_ns,
                ),
                "fully_quiescent_ns": max(
                    0,
                    int(simulated_duration_ns) - global_model_or_transfer_ns,
                ),
                "global_any_model_busy_fraction": (
                    global_model_execution_ns / simulated_duration_ns
                    if simulated_duration_ns > 0 else None),
                "scope_note": (
                    "Observed online engine activity under the configured "
                    "session-arrival process and the workload's closed-loop "
                    "human/tool gaps. Interpret it together with the session "
                    "report's explicit trace, backlog, or Poisson admission "
                    "mode; an active-session backlog does not imply that an "
                    "LLM request is runnable during every closed-loop gap."
                ),
            },
            "time_breakdown": {
                "aggregate_request_migration_raw_elapsed_ns": (
                    self.metrics.source_demotion_join_wait_ns
                    + self.metrics.critical_restore_ns),
                "aggregate_request_migration_stall_ns": (
                    self.metrics.source_demotion_join_wait_ns
                    + self.metrics.critical_restore_hbm_admission_wait_ns
                    + self.metrics.async_restore_owner_barrier_ns
                    if self.async_decode_join_enabled
                    else self.metrics.source_demotion_join_wait_ns
                    + self.metrics.critical_restore_ns),
                "aggregate_request_migration_hbm_admission_wait_ns": (
                    self.metrics.critical_restore_hbm_admission_wait_ns),
                "aggregate_pd_pair_fifo_wait_ns": (
                    self.metrics.pd_pair_fifo_wait_ns),
                "aggregate_prepare_boundary_wait_ns": (
                    self.metrics.prepare_boundary_wait_ns),
                "aggregate_source_demotion_join_wait_ns": (
                    self.metrics.source_demotion_join_wait_ns),
                "censored_source_demotion_join_count": len(
                    self._censored_source_demotion_join_audits),
                "censored_source_demotion_join_elapsed_ns_membership_sum": (
                    sum(
                        row["elapsed_ns"]
                        for row in
                        self._censored_source_demotion_join_audits
                    )
                ),
                "censored_source_demotion_join_remaining_ns_membership_sum": (
                    sum(
                        row["remaining_ns"]
                        for row in
                        self._censored_source_demotion_join_audits
                    )
                ),
                "censored_source_demotion_join_audits": list(
                    self._censored_source_demotion_join_audits),
                "censored_destination_admission_count": len(
                    self._censored_destination_admission_audits),
                "censored_destination_admission_elapsed_ns_membership_sum": (
                    sum(
                        row["elapsed_ns"]
                        for row in
                        self._censored_destination_admission_audits
                    )
                ),
                "censored_destination_admission_audits": list(
                    self._censored_destination_admission_audits),
                "censored_transient_dram_admission_count": len(
                    self._censored_transient_restore_audits),
                "censored_transient_dram_admission_elapsed_ns_membership_sum": (
                    sum(
                        row["elapsed_ns"]
                        for row in self._censored_transient_restore_audits
                    )
                ),
                "censored_transient_dram_admission_audits": list(
                    self._censored_transient_restore_audits),
                "aggregate_owner_ready_gate_ns": (
                    self.metrics.pd_pair_fifo_wait_ns
                    + self.metrics.prepare_boundary_wait_ns
                    + self.metrics.source_demotion_join_wait_ns
                    + self.metrics.critical_restore_ns),
                "pd_pair_fifo_prepare_count": (
                    self.metrics.pd_pair_fifo_admissions),
                "pd_pair_fifo_waiting_prepare_count": (
                    self.metrics.pd_pair_fifo_waiting_admissions),
                "prepare_boundary_prepare_count": (
                    self.metrics.prepare_boundary_admissions),
                "prepare_boundary_waiting_prepare_count": (
                    self.metrics.prepare_boundary_waiting_admissions),
                "source_demotion_join_prepare_count": (
                    self.metrics.source_demotion_join_admissions),
                "source_demotion_join_waiting_prepare_count": (
                    self.metrics.source_demotion_join_waiting_admissions),
                "pd_pair_fifo_wait_fraction_of_total_request_latency": (
                    self.metrics.pd_pair_fifo_wait_ns
                    / self.metrics.total_request_latency_ns
                    if self.metrics.total_request_latency_ns > 0 else None),
                "pd_pair_fifo_fraction_denominator_scope": (
                    "Sum of completed non-prefill request latency over the "
                    "full online simulation. The numerator covers successful "
                    "continuation preparations, including a preparation later "
                    "right-censored at an early-stop boundary; use the session "
                    "report's measured completed-session cohort for the paper "
                    "window denominator."
                ),
                "prepare_boundary_wait_fraction_of_total_request_latency": (
                    self.metrics.prepare_boundary_wait_ns
                    / self.metrics.total_request_latency_ns
                    if self.metrics.total_request_latency_ns > 0 else None),
                "prepare_boundary_fraction_denominator_scope": (
                    "The same full-simulation completed-request latency "
                    "denominator as pd_pair_fifo_wait. This numerator is "
                    "scheduler/engine-boundary admission before physical "
                    "restore issue and is not migration time."
                ),
                "source_demotion_join_wait_fraction_of_total_request_latency": (
                    self.metrics.source_demotion_join_wait_ns
                    / self.metrics.total_request_latency_ns
                    if self.metrics.total_request_latency_ns > 0 else None),
                "source_demotion_join_fraction_denominator_scope": (
                    "The same full-simulation completed-request latency "
                    "denominator. The numerator is only the request-visible "
                    "tail of a capacity swap-out already in flight at return; "
                    "background service before return is excluded."
                ),
                "aggregate_transient_dram_capacity_wait_ns": (
                    self.metrics.transient_dram_capacity_wait_ns),
                "aggregate_request_migration_service_ns": (
                    self.metrics.critical_restore_service_ns),
                "aggregate_request_migration_queue_wait_ns": (
                    self.metrics.critical_restore_queue_wait_ns),
                "aggregate_request_migration_transfer_queue_wait_ns": max(
                    0,
                    self.metrics.critical_restore_queue_wait_ns
                    - self.metrics.transient_dram_capacity_wait_ns,
                ),
                "transient_dram_capacity_wait_is_subset_of_queue_wait": True,
                "migration_queue_wait_scope_note": (
                    "Lower-tier wait includes the disjoint transient-DRAM "
                    "capacity-admission subset plus accepted transfer-calendar "
                    "queue gaps. It excludes destination-HBM admission."),
                "aggregate_pd_decode_receive_admission_wait_ns": (
                    self.metrics.pd_decode_receive_admission_wait_ns),
                "aggregate_pd_decode_receive_capacity_wait_ns": (
                    self.metrics.pd_decode_receive_capacity_wait_ns),
                "aggregate_pd_decode_receive_critical_wait_ns": (
                    self.metrics.pd_decode_receive_critical_wait_ns),
                "aggregate_pd_prefill_admission_wait_ns": (
                    self.metrics.pd_prefill_admission_wait_ns),
                "aggregate_pd_prefill_capacity_wait_ns": (
                    self.metrics.pd_prefill_capacity_wait_ns),
                "aggregate_pd_prefill_admission_critical_wait_ns": (
                    self.metrics.pd_prefill_admission_critical_wait_ns),
                "aggregate_pd_launch_admission_wait_ns": (
                    self.metrics.pd_launch_admission_wait_ns),
                "aggregate_pd_launch_admission_critical_wait_ns": (
                    self.metrics.pd_launch_admission_critical_wait_ns),
                "aggregate_pd_chunk_admission_wait_ns": (
                    self.metrics.pd_chunk_admission_wait_ns),
                "aggregate_pd_chunk_admission_critical_wait_ns": (
                    self.metrics.pd_chunk_admission_critical_wait_ns),
                "aggregate_pd_chunk_cancelled_admission_wait_ns": (
                    self.metrics.pd_chunk_cancelled_admission_wait_ns),
                "aggregate_pd_chunk_cancelled_admission_critical_wait_ns": (
                    self.metrics
                    .pd_chunk_cancelled_admission_critical_wait_ns),
                "aggregate_pd_chunk_attempt_admission_wait_ns": (
                    self.metrics.pd_chunk_admission_wait_ns
                    + self.metrics.pd_chunk_cancelled_admission_wait_ns),
                "aggregate_pd_chunk_attempt_admission_critical_wait_ns": (
                    self.metrics.pd_chunk_admission_critical_wait_ns
                    + self.metrics
                    .pd_chunk_cancelled_admission_critical_wait_ns),
                "aggregate_pd_chunk_snapshot_feasible_wait_ns": (
                    self.metrics.pd_chunk_snapshot_feasible_wait_ns),
                "pd_side_wait_scope_note": (
                    "P and D side waits audit the same atomic launch gate and "
                    "must not be added. Use aggregate_pd_launch_admission_"
                    "critical_wait_ns for the first-chunk causal delay. "
                    "Aggregate chunk admission wait covers successful "
                    "incremental P/D claims; cancelled pre-commit attempts "
                    "are reported by the separate cancelled aggregates. "
                    "Both are non-additive with scheduler "
                    "queue delay because resource eligibility advances to "
                    "the admission timestamp."),
                "migration_restore_exposure_union_ns": restore_exposure_ns,
                "migration_restore_exposure_fraction_of_makespan": (
                    exposure_fraction),
                # Backward-compatible aliases. These are exposure, not an
                # exact makespan penalty: useful work may overlap the interval.
                "migration_critical_interval_union_ns": restore_exposure_ns,
                "migration_critical_interval_union_fraction_of_makespan": (
                    exposure_fraction),
                "migration_makespan_penalty_ns": None,
                "migration_makespan_penalty_fraction": None,
                "migration_stall_fraction_of_total_request_latency": (
                    request_fraction),
                "total_request_latency_ns": (
                    self.metrics.total_request_latency_ns or None),
                "recompute_model_compute_ns": (
                    self.metrics.recompute_model_compute_ns or None),
                "total_model_compute_ns": (
                    self.metrics.total_model_compute_ns or None),
                "recompute_fraction_of_total_model_compute": (
                    recompute_compute_fraction),
                "recompute_tokens": self.metrics.recompute_tokens,
                "policy_avoidable_recompute_tokens": (
                    self.metrics.policy_avoidable_recompute_tokens),
                "prompt_token_denominator": prompt_denominator or None,
                "prompt_token_denominator_scope": (
                    "all_prompt_tokens" if self.metrics.total_prompt_tokens > 0
                    else "resumed_agentic_prompts_only"),
                "recompute_token_fraction": recompute_token_fraction,
                "policy_avoidable_recompute_fraction_of_logical_prompt": (
                    policy_recompute_prompt_fraction),
                "executed_prefill_token_denominator": (
                    executed_prefill_tokens or None),
                "policy_avoidable_recompute_fraction_of_executed_prefill": (
                    policy_recompute_executed_fraction),
                "integration_note": (
                    "The serving loop supplies aggregate request latency and "
                    "prompt tokens. Exact recomputation/model-compute time "
                    "requires kernel-time attribution or a paired "
                    "counterfactual run; unavailable denominators remain null. "
                    "Raw recompute_tokens includes the mandatory one-token "
                    "full-prefix execution cap; policy_avoidable values remove "
                    "that baseline-independent work. "
                    "The restore interval union is exposure with at least one "
                    "blocked request, not a zero-migration makespan delta."
                    " Physical restore excludes strict P/D pair FIFO, "
                    "prepare-boundary admission, and the exposed tail of an "
                    "already-running source demotion; aggregate_owner_ready_"
                    "gate_ns is the exact membership sum of those three "
                    "components plus physical restore. The compatibility-"
                    "named HBM "
                    "admission wait is total "
                    "destination admission wait and includes the separately "
                    "reported transient DRAM capacity subset for SSD resumes."
                ),
            },
            "resource_queues": {
                key: {
                    "busy_until_ns": self._resource_busy_until.get(key, 0),
                    "service_demand_ns": self._resource_busy_ns.get(key, 0),
                    "jobs": self._resource_jobs.get(key, 0),
                }
                for key in sorted(self._resource_busy_until)
            },
            "active_hbm_reclaim_rejections": {
                "total": sum(
                    self._active_reclaim_rejection_counts.values()),
                "by_reason": dict(sorted(
                    self._active_reclaim_rejection_counts.items())),
                "sample_limit": (
                    self._active_reclaim_rejection_sample_limit),
                "sampled": len(
                    self._active_reclaim_rejection_samples),
                "suppressed": max(
                    0,
                    sum(self._active_reclaim_rejection_counts.values())
                    - len(self._active_reclaim_rejection_samples),
                ),
                "samples": list(
                    self._active_reclaim_rejection_samples),
            },
            "queue_recompute_policy": {
                "enabled": self.config.queue_recompute_enabled,
                "decision_point": (
                    "before_destination_hbm_reservation_and_foreground_io"),
                "decision_rule": (
                    "first require full projected_total_wait_ns > "
                    "max(ratio * full_service_ns, min_wait_ns); then choose "
                    "the minimum of prefix_restore_ns + cost_multiplier * "
                    "singleton_COMP(H)-singleton_COMP(R) over R, zero, and "
                    "deterministic block-prefix candidates which fit the "
                    "causal P/D next-chunk slack snapshot"),
                "projected_total_wait_semantics": (
                    "destination_hbm_admission_wait_plus_lower_tier_"
                    "capacity_and_transfer_queue_wait"),
                "strict_inequality": True,
                "configured_wait_service_ratio": (
                    self.config.queue_recompute_wait_service_ratio),
                "configured_min_wait_ns": (
                    self.config.queue_recompute_min_wait_ns),
                "configured_cost_guard_multiplier": (
                    self.config.queue_recompute_cost_guard_multiplier),
                "configured_prefill_headroom_chunks": (
                    self.config.queue_recompute_prefill_headroom_chunks),
                "headroom_semantics": (
                    "causal_unreserved_P_and_D_snapshot_not_reservation"),
                "headroom_owner": "ordinary_atomic_pd_chunk_admission",
                "cost_guard_enabled": (
                    self.config.queue_recompute_cost_guard_multiplier > 0),
                "cost_estimator": {
                    "method": (
                        "singleton chunked-prefill COMP critical-path "
                        "difference: hit=H minus hit=R"),
                    "collectives_included": False,
                    "future_batch_state_used": False,
                    "providers_by_target_instance": (
                        queue_provider_provenance),
                },
                "evaluation_attempts": (
                    self.metrics.queue_recompute_evaluation_attempts),
                "severe_gate_passes": (
                    self.metrics.queue_recompute_severe_gate_passes),
                "cost_gate_passes": (
                    self.metrics.queue_recompute_cost_gate_passes),
                "full_restore_decisions": (
                    self.metrics.queue_recompute_full_restore_decisions),
                "partial_restore_decisions": (
                    self.metrics.queue_recompute_partial_restore_decisions),
                "zero_restore_decisions": (
                    self.metrics.queue_recompute_zero_restore_decisions),
                "partial_cpu_decisions": (
                    self.metrics.queue_recompute_partial_cpu_decisions),
                "partial_ssd_decisions": (
                    self.metrics.queue_recompute_partial_ssd_decisions),
                "drop_decisions": (
                    self.metrics.queue_recompute_drop_decisions),
                "drop_decisions_legacy_semantics": (
                    "H_equals_zero_decisions_only"),
                "cpu_drop_decisions": (
                    self.metrics.queue_recompute_cpu_drop_decisions),
                "ssd_drop_decisions": (
                    self.metrics.queue_recompute_ssd_drop_decisions),
                "dropped_bytes": (
                    self.metrics.queue_recompute_dropped_bytes),
                "dropped_bytes_legacy_semantics": (
                    "alias_for_avoided_foreground_restore_bytes"),
                "avoided_restore_bytes": (
                    self.metrics.queue_recompute_avoided_restore_bytes),
                "physical_entry_dropped_bytes": (
                    self.metrics
                    .queue_recompute_physical_entry_dropped_bytes),
                "declared_recompute_tokens": (
                    self.metrics.queue_recompute_tokens),
                "policy_avoidable_recompute_tokens": (
                    self.metrics.queue_recompute_policy_avoidable_tokens),
                "selected_restore_tokens": (
                    self.metrics.queue_recompute_selected_restore_tokens),
                "dropped_suffix_tokens": (
                    self.metrics.queue_recompute_dropped_suffix_tokens),
                "selected_restore_bytes": (
                    self.metrics.queue_recompute_selected_restore_bytes),
                "dropped_suffix_bytes": (
                    self.metrics.queue_recompute_dropped_suffix_bytes),
                "modified_full_projected_queue_wait_ns": (
                    self.metrics.queue_recompute_projected_queue_wait_ns),
                "modified_full_projected_hbm_admission_wait_ns": (
                    self.metrics
                    .queue_recompute_projected_hbm_admission_wait_ns),
                "modified_full_projected_transient_dram_capacity_wait_ns": (
                    self.metrics
                    .queue_recompute_projected_transient_dram_capacity_wait_ns),
                "modified_full_projected_total_wait_ns": (
                    self.metrics.queue_recompute_projected_queue_wait_ns
                    + self.metrics
                    .queue_recompute_projected_hbm_admission_wait_ns),
                "modified_full_projected_service_ns": (
                    self.metrics.queue_recompute_projected_service_ns),
                # Schema-18 compatibility aliases. These refer to the full
                # rejected projection, not the selected H projection.
                "selected_projected_queue_wait_ns": (
                    self.metrics.queue_recompute_projected_queue_wait_ns),
                "selected_projected_hbm_admission_wait_ns": (
                    self.metrics
                    .queue_recompute_projected_hbm_admission_wait_ns),
                "selected_projected_transient_dram_capacity_wait_ns": (
                    self.metrics
                    .queue_recompute_projected_transient_dram_capacity_wait_ns),
                "selected_projected_total_wait_ns": (
                    self.metrics.queue_recompute_projected_queue_wait_ns
                    + self.metrics
                    .queue_recompute_projected_hbm_admission_wait_ns),
                "selected_projected_service_ns": (
                    self.metrics.queue_recompute_projected_service_ns),
                "partial_prefix_projected_queue_wait_ns": (
                    self.metrics
                    .queue_recompute_prefix_projected_queue_wait_ns),
                "partial_prefix_projected_hbm_admission_wait_ns": (
                    self.metrics
                    .queue_recompute_prefix_projected_hbm_admission_wait_ns),
                "partial_prefix_projected_transient_dram_capacity_wait_ns": (
                    self.metrics
                    .queue_recompute_prefix_projected_transient_dram_capacity_wait_ns),
                "partial_prefix_projected_service_ns": (
                    self.metrics
                    .queue_recompute_prefix_projected_service_ns),
                "selected_estimated_suffix_recompute_comp_ns": (
                    self.metrics.queue_recompute_estimated_recompute_ns),
                "accounting_invariants": queue_recompute_audit,
                "pending_restore_commitments": len(
                    self._queue_recompute_restore_commitments),
                "drop_fraction_of_all_agentic_requests": (
                    self.metrics.queue_recompute_drop_decisions
                    / len(self._classified_request_ids)
                    if self._classified_request_ids else None
                ),
                "modified_fraction_of_all_agentic_requests": (
                    (
                        self.metrics.queue_recompute_partial_restore_decisions
                        + self.metrics.queue_recompute_zero_restore_decisions
                    ) / len(self._classified_request_ids)
                    if self._classified_request_ids else None
                ),
                "scope_note": (
                    "Only CPU/SSD resumes are evaluated. HBM hits, zero-reuse "
                    "returns, and already accepted asynchronous demotions are "
                    "never cancelled by this policy. The first decision is "
                    "frozen across HBM-capacity retries. A purpose-built "
                    "non-mutating shadow projects HBM LRU demotions, CPU LRU "
                    "cascades, SSD transient-DRAM admission, and the complete "
                    "foreground transfer-calendar occupancy before the "
                    "decision. A missing full projection or a full projection "
                    "below the strict severe threshold fails closed to the "
                    "ordinary full restore. Transient-DRAM capacity wait is an explicit "
                    "subset of projected queue wait, not an additive fifth "
                    "component. Under severe pressure, the selector evaluates "
                    "R, zero, and deterministic contiguous block-aligned H<R "
                    "prefixes. Communication and destination ownership cover "
                    "only H; [H,R) follows ordinary prefill recomputation. The "
                    "full CPU/SSD source remains authoritative until prefix "
                    "DMA completion and is then fully released, including any "
                    "durable duplicate. The P/D headroom snapshot is not a "
                    "claim and does not guarantee zero later chunk-admission "
                    "wait. No KV selection drops the logical session."
                ),
            },
            "request_classification": {
                "all_agentic_request_count": len(
                    self._classified_request_ids),
                "denominator_scope": (
                    "Every routed agentic LLM call, including session starts, "
                    "zero-reuse calls, and all return classes."
                ),
                "by_residency_at_return_and_return_gap_type": {
                    residency: dict(sorted(counts.items()))
                    for residency, counts in sorted(
                        self._request_counts_by_residency_and_return.items())
                },
                "by_resume_source_and_return_gap_type": {
                    source: dict(sorted(counts.items()))
                    for source, counts in sorted(
                        self._request_counts_by_source_and_return.items())
                },
            },
            "batch_composition": {
                "membership_scope": (
                    "Batch memberships, not unique requests; chunked prefill "
                    "and decode can count one request more than once."
                ),
                "restore_barrier_inside_batch": False,
                "sync_swap_barrier_before_batch": (
                    self.metrics.sync_swap_blocked_iterations > 0),
                "restore_barrier_semantics": (
                    "pre_dispatch_engine_gate_associated_with_next_iteration"
                    if self.synchronous_swap_enabled
                    else "restore_completed_before_scheduler_visibility"
                ),
                "by_resume_source_and_return_gap_type": {
                    source: dict(sorted(counts.items()))
                    for source, counts in sorted(
                        self._batch_memberships_by_source_and_return.items())
                },
            },
            "active_hbm_reclaim": {
                "admission_count": (
                    self.metrics.active_hbm_reclaim_admissions),
                "admission_bytes": self.metrics.active_hbm_reclaim_bytes,
                "admission_per_rank_bytes": (
                    self.metrics.active_hbm_reclaim_per_rank_bytes),
                "aggregate_wait_ns": (
                    self.metrics.active_hbm_reclaim_wait_ns),
                "outstanding_claims": [
                    asdict(claim)
                    for _, claim in sorted(
                        self._active_hbm_reclaim_claims.items())
                ],
                "scope_note": (
                    "Admissions reserve per-rank HBM for active scheduler "
                    "work. Wait is the capacity-ready delay caused by idle "
                    "LRU reclaim; transfer queue and service remain in the "
                    "ordinary migration counters."
                ),
            },
            "idle_capacity_opportunity": {
                "average_idle_hbm_bytes": (
                    self.metrics.hbm_byte_ns / simulated_duration_ns
                    if simulated_duration_ns > 0 else None),
                "average_idle_cpu_bytes": (
                    self.metrics.cpu_byte_ns / simulated_duration_ns
                    if simulated_duration_ns > 0 else None),
                "average_idle_ssd_bytes": (
                    self.metrics.ssd_byte_ns / simulated_duration_ns
                    if simulated_duration_ns > 0 else None),
                "peak_idle_hbm_bytes": self.metrics.peak_idle_hbm_bytes,
                "peak_idle_cpu_bytes": self.metrics.peak_idle_cpu_bytes,
                "peak_idle_ssd_bytes": self.metrics.peak_ssd_used_bytes,
                "hbf_eligible_resumes_beyond_hbm": (
                    self.metrics.hbf_eligible_resumes),
                "hbf_eligible_restore_bytes": (
                    self.metrics.hbf_eligible_restore_bytes),
                "hbf_gross_restore_stall_upper_bound_ns": (
                    self.metrics.hbf_gross_stall_upper_bound_ns),
                "hbf_dropped_recompute_tokens": (
                    self.metrics.hbf_dropped_recompute_tokens),
                "hbf_gross_total_stall_upper_bound_ns": None,
                "scope_note": (
                    "HBF eligibility includes CPU, SSD, and dropped resumes. "
                    "The restore-stall value covers migration only and excludes "
                    "dropped-prefix recomputation, which is exposed separately "
                    "as tokens because online kernel-time attribution is not "
                    "available. A total HBF stall opportunity therefore remains "
                    "null until that attribution is supplied; partial-attention "
                    "service must then be subtracted."
                ),
            },
            "ssd": {
                "capacity_bytes": (
                    self.config.ssd_capacity_bytes * ssd_node_count),
                "capacity_bytes_per_node": self.config.ssd_capacity_bytes,
                "num_devices": total_ssd_devices,
                "devices_per_node": devices_per_node,
                "node_count": ssd_node_count,
                "node_ids": list(self._ssd_node_ids),
                "used_bytes": self.ssd_used_bytes,
                "used_bytes_by_node": ssd_used_bytes_by_node,
                "reserved_bytes": self._ssd_reserved_bytes(),
                "reserved_bytes_by_node": ssd_reserved_bytes_by_node,
                "committed_reserved_bytes": (
                    self.ssd_used_bytes + self._ssd_reserved_bytes()),
                "host_write_bytes": self.metrics.ssd_host_write_bytes,
                "cancelled_partial_host_write_bytes": (
                    self.metrics.ssd_cancelled_host_write_bytes),
                "host_read_bytes": self.metrics.ssd_host_read_bytes,
                "write_mode": self.config.ssd_write_mode,
            },
            # Direct input contract for ``python -m serving.endurance``.
            # Balanced striping is an explicit baseline assumption; measured
            # per-device counters can replace this list in future backends.
            "storage": {
                "totals": {
                    "aligned_host_write_bytes": self.metrics.ssd_host_write_bytes,
                    "host_read_bytes": self.metrics.ssd_host_read_bytes,
                },
                "devices": devices,
                "distribution": (
                    "balanced_across_node_local_pools_assumption"),
            },
            "event_semantics": {
                "drop": {
                    "object_scope": "kv_cache_entry",
                    "logical_session_effect": "none",
                    "classification_field": "drop_class",
                    "reason_field": "reason",
                    "classes": dict(_KV_DROP_CLASS_SEMANTICS),
                    "compatibility_note": (
                        "The legacy event name is retained for consumers of "
                        "earlier schemas. A drop record never removes a "
                        "logical session or releases its admission slot."
                    ),
                },
            },
            "events": list(self.events),
        }

    def save_metrics(self, path: str, simulated_duration_ns: int = 0,
                     dataset: Optional[str] = None,
                     run_id: Optional[str] = None,
                     measurement_censored: bool = False) -> None:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.summary(
                    simulated_duration_ns, dataset, run_id,
                    measurement_censored=measurement_censored),
                f, indent=2)
            f.write("\n")

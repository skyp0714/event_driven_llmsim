"""SSD-staged serving on one GPU server and one eight-card HBF server.

This module models the concrete two-server design:

* one P4D4 GPU server is the sole owner of model execution, P/D HBM, host
  DRAM, and local SSD before promotion;
* non-terminal GPU turns normally checkpoint D -> CPU -> SSD, after which a
  causal policy may promote the durable copy through CPU/NIC/RDMA;
* composite policies instead send a resumed stable D-HBM turn directly
  through GPU/NIC/RDMA at that turn's internal completion; and
* a resume always invalidates a not-yet-published promotion and restores
  through the GPU tiered node unless the HBF copy has already committed.

All compute and transfer components share one :class:`ResourceCalendar`.
The implementation is analytical-only and intentionally has no multi-HBF
path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import heapq
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .gpu_hbf_hybrid import (
    GPUHBFHybridSystem,
    HybridCall,
    HybridCallState,
    HybridDeadlockError,
    HybridExecution,
    HybridSession,
    HybridSystemMetrics,
)
from .gpu_pd_latency import P4D4GPUHardware
from .gpu_pd_tier_lifecycle import (
    RESTORE_EXECUTION_BULK,
    SSDExportStatus,
    SSDExportTicket,
    Tier,
    TierSessionState,
)
from .gpu_pd_tiered_node import (
    FiniteHBMTieredP4D4Node,
    TieredCallState,
    TieredNodeCall,
    TieredSessionLineage,
)
from .hbf_comparison_metrics import CompletedRequest
from .hbf_comparison_workload import (
    ScheduledSession,
    full_drain_hashes,
    stable_json_sha256,
)
from .hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
)
from .hbf_full_model_lifecycle import (
    FullModelHBFLifecycle,
    MigrationJob,
    PerGroupCapacityLedger,
    PlacementState,
    ResourceCalendar,
    ResumeExecution,
    ResumeRoute,
)
from .hbf_full_model_pool import (
    FullModelHBFServingPool,
    HBFServingRequest,
    derive_lpddr_workspace_bytes,
)


SUPPORTED_SSD_STAGED_HBF_LAYOUTS = frozenset({
    "dp8",
    "tp4",
    "tp8",
    "tp8_context",
})
SUPPORTED_SSD_PROMOTION_MODES = frozenset({
    "composite",
    "eager",
    "idle_delay",
    "load_aware",
    "never",
})


@dataclass(frozen=True)
class SSDPromotionPolicy:
    """Causal SSD-to-HBF promotion policy.

    ``idle_delay_ns`` is measured from the predecessor LLM request's user
    completion.  ``allowed_gap_types`` filters on metadata already known at
    that boundary.  The policy never inspects the future wait duration.
    """

    key: str = "eager"
    mode: str = "eager"
    idle_delay_ns: int = 0
    allowed_gap_types: tuple[str, ...] = ()
    retry_ns: int = 1_000_000
    load_hysteresis: float = 0.10
    direct_d_resume: bool = False
    ssd_checkpoint_age_ns: Optional[int] = None
    human_gap_broadcast: bool = False
    load_aware_admission: bool = False
    min_context_tokens: int = 0
    demotion_hysteresis: float = 0.0
    rank_mode: str = "context"

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("promotion policy key must be non-empty")
        if self.mode not in SUPPORTED_SSD_PROMOTION_MODES:
            raise ValueError(
                "promotion policy mode must be one of "
                f"{sorted(SUPPORTED_SSD_PROMOTION_MODES)}")
        for name in ("idle_delay_ns", "retry_ns", "min_context_tokens"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be non-negative")
        if self.retry_ns <= 0:
            raise ValueError("retry_ns must be positive")
        if self.mode == "eager" and self.idle_delay_ns:
            raise ValueError("eager promotion cannot have an idle delay")
        if self.mode == "idle_delay" and self.idle_delay_ns <= 0:
            raise ValueError(
                "idle-delay promotion requires a positive delay")
        if (
            self.ssd_checkpoint_age_ns is not None
            and (
                isinstance(self.ssd_checkpoint_age_ns, bool)
                or not isinstance(self.ssd_checkpoint_age_ns, int)
                or self.ssd_checkpoint_age_ns < 0
            )
        ):
            raise ValueError(
                "ssd_checkpoint_age_ns must be non-negative or None")
        for name in (
            "direct_d_resume",
            "human_gap_broadcast",
            "load_aware_admission",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.mode == "load_aware":
            object.__setattr__(self, "load_aware_admission", True)
        if self.mode == "never" and (
            self.direct_d_resume
            or self.ssd_checkpoint_age_ns is not None
            or self.human_gap_broadcast
            or self.load_aware_admission
        ):
            raise ValueError(
                "never promotion cannot enable composite triggers")
        if (
            isinstance(self.load_hysteresis, bool)
            or not isinstance(self.load_hysteresis, (int, float))
            or not math.isfinite(float(self.load_hysteresis))
            or not 0.0 <= float(self.load_hysteresis) <= 1.0
        ):
            raise ValueError(
                "load_hysteresis must be in [0, 1]")
        if (
            isinstance(self.demotion_hysteresis, bool)
            or not isinstance(self.demotion_hysteresis, (int, float))
            or not math.isfinite(float(self.demotion_hysteresis))
            or float(self.demotion_hysteresis) < 0.0
        ):
            raise ValueError(
                "demotion_hysteresis must be non-negative")
        if self.demotion_hysteresis and not self.load_aware_admission:
            raise ValueError(
                "demotion requires load-aware admission")
        if self.rank_mode not in {
            "context", "density", "density_oracle",
            "calls", "calls_oracle",
        }:
            raise ValueError(
                "rank_mode must be one of context, density, "
                "density_oracle, calls, calls_oracle")
        if (
            not isinstance(self.allowed_gap_types, tuple)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in self.allowed_gap_types
            )
        ):
            raise ValueError(
                "allowed_gap_types must be an immutable tuple of strings")
        normalized = tuple(
            value.strip().lower()
            for value in self.allowed_gap_types
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "allowed_gap_types cannot contain duplicates")
        object.__setattr__(self, "allowed_gap_types", normalized)

    @classmethod
    def for_key(cls, key: str) -> "SSDPromotionPolicy":
        fixed_delays = {
            "delay_25ms": 25_000_000,
            "delay_50ms": 50_000_000,
            "delay_100ms": 100_000_000,
            "delay_200ms": 200_000_000,
            "delay_500ms": 500_000_000,
            "delay_1000ms": 1_000_000_000,
            "delay_1s": 1_000_000_000,
            "delay_5s": 5_000_000_000,
            "delay_30s": 30_000_000_000,
            "delay_300s": 300_000_000_000,
        }
        if key == "eager":
            return cls()
        if key == "tool_immediate":
            return cls(
                key=key,
                mode="eager",
                allowed_gap_types=("tool",),
            )
        if key == "human_immediate":
            return cls(
                key=key,
                mode="eager",
                allowed_gap_types=("human",),
            )
        if key == "tool_or_human_immediate":
            return cls(
                key=key,
                mode="eager",
                allowed_gap_types=("tool", "human"),
            )
        if key in fixed_delays:
            return cls(
                key=key,
                mode="idle_delay",
                idle_delay_ns=fixed_delays[key],
            )
        if key == "load_aware":
            return cls(
                key=key,
                mode="load_aware",
                retry_ns=50_000_000,
            )
        if key == "load_aware_demote":
            return cls(
                key=key,
                mode="load_aware",
                retry_ns=50_000_000,
                demotion_hysteresis=0.5,
            )
        if key == "load_aware_demote_h2":
            return cls(
                key=key,
                mode="load_aware",
                retry_ns=50_000_000,
                demotion_hysteresis=2.0,
            )
        if key == "load_aware_density":
            return cls(
                key=key,
                mode="load_aware",
                retry_ns=50_000_000,
                demotion_hysteresis=0.5,
                rank_mode="density",
            )
        if key == "load_aware_density_oracle":
            return cls(
                key=key,
                mode="load_aware",
                retry_ns=50_000_000,
                demotion_hysteresis=0.5,
                rank_mode="density_oracle",
            )
        if key == "load_aware_calls":
            return cls(
                key=key,
                mode="load_aware",
                retry_ns=50_000_000,
                demotion_hysteresis=0.5,
                rank_mode="calls",
            )
        if key == "load_aware_calls_oracle":
            return cls(
                key=key,
                mode="load_aware",
                retry_ns=50_000_000,
                demotion_hysteresis=0.5,
                rank_mode="calls_oracle",
            )
        if key in {
            "composite",
            "composite_adaptive",
            "composite_ready",
            "composite_ready_adaptive",
        }:
            adaptive = key.endswith("_adaptive")
            promote_on_ssd_ready = key.startswith("composite_ready")
            return cls(
                key=key,
                mode="composite",
                retry_ns=(
                    50_000_000
                    if adaptive
                    else 1_000_000
                ),
                direct_d_resume=True,
                ssd_checkpoint_age_ns=(
                    0 if promote_on_ssd_ready else 50_000_000),
                human_gap_broadcast=True,
                load_aware_admission=adaptive,
            )
        if key == "never":
            return cls(key=key, mode="never")
        raise ValueError(
            f"unsupported SSD promotion policy {key!r}")

    def eligible_gap(self, gap_type: Optional[str]) -> bool:
        if self.mode == "never":
            return False
        if not self.allowed_gap_types:
            return True
        normalized = (
            "unknown"
            if gap_type is None
            else gap_type.strip().lower()
        )
        return normalized in self.allowed_gap_types


@dataclass
class SSDPromotionIntent:
    session_id: str
    predecessor_request_id: int
    snapshot_version: int
    user_completion_ns: int
    due_ns: Optional[int]
    gap_type: Optional[str]
    serial: int
    ssd_publication_ns: Optional[int] = None
    human_due_now: bool = False
    due_reached: bool = False


@dataclass(frozen=True)
class DirectMigrationTicket:
    job: MigrationJob
    source_copy_id: int


@dataclass
class SSDStagedHybridMetrics:
    submitted_calls: int = 0
    routed_calls: int = 0
    gpu_calls: int = 0
    hbf_calls: int = 0
    gpu_first_turn_calls: int = 0
    gpu_owned_calls: int = 0
    gpu_import_inflight_calls: int = 0
    gpu_recompute_calls: int = 0
    user_completed_calls: int = 0
    internal_completed_calls: int = 0
    proactive_offloads_requested: int = 0
    proactive_offloads_started: int = 0
    ssd_checkpoints_published: int = 0
    promotion_intents_scheduled: int = 0
    promotion_intents_filtered: int = 0
    promotion_intents_canceled: int = 0
    promotion_thresholds_reached: int = 0
    ssd_exports_started: int = 0
    ssd_export_capacity_deferrals: int = 0
    hbf_imports_started: int = 0
    hbf_import_capacity_deferrals: int = 0
    promotion_load_deferrals: int = 0
    hbf_imports_committed: int = 0
    hbf_imports_stale: int = 0
    direct_migrations_started: int = 0
    direct_migration_capacity_deferrals: int = 0
    direct_migrations_committed: int = 0
    direct_migrations_stale: int = 0
    human_gap_broadcasts: int = 0
    human_gap_broadcast_intents: int = 0
    hbf_load_demotions: int = 0
    max_pending_calls: int = 0


class SSDStagedGPUHBFNode:
    """One P4D4 GPU server plus one SSD-fed eight-card HBF server."""

    def __init__(
            self, *, repo_root: Path,
            gpu_hardware: P4D4GPUHardware,
            hbf_hardware: HBFServerHardware,
            hbf_layout: str | HBFParallelLayout = "tp4",
            gpu_node_id: int = 0,
            hbf_server_id: int = 0,
            p_capacity_bytes_per_rank: Optional[int] = None,
            d_capacity_bytes_per_rank: Optional[int] = None,
            cpu_capacity_bytes: Optional[int] = None,
            ssd_capacity_bytes: Optional[int] = None,
            max_num_batched_tokens: int = 8_192,
            max_num_seqs: int = 128,
            p_max_num_seqs: Optional[int] = None,
            d_max_num_seqs: Optional[int] = None,
            max_prefill_chunk_tokens: int = 4_096,
            hbf_mixed_batch_latency_limit_ns: Optional[int] = None,
            band: str = "central",
            restore_execution_mode: str = RESTORE_EXECUTION_BULK,
            validate_every_event: bool = True,
            promotion_policy: Optional[
                str | SSDPromotionPolicy] = None,
            migration_policy: Optional[
                str | SSDPromotionPolicy] = None) -> None:
        if not isinstance(validate_every_event, bool):
            raise ValueError("validate_every_event must be a boolean")
        if (
            isinstance(gpu_node_id, bool)
            or not isinstance(gpu_node_id, int)
            or gpu_node_id < 0
        ):
            raise ValueError("gpu_node_id must be non-negative")
        if (
            isinstance(hbf_server_id, bool)
            or not isinstance(hbf_server_id, int)
            or hbf_server_id < 0
        ):
            raise ValueError("hbf_server_id must be non-negative")
        layout = (
            HBFParallelLayout.for_key(hbf_layout)
            if isinstance(hbf_layout, str) else hbf_layout
        )
        if not isinstance(layout, HBFParallelLayout):
            raise TypeError(
                "hbf_layout must be a key or HBFParallelLayout")
        if (
            promotion_policy is not None
            and migration_policy is not None
        ):
            raise ValueError(
                "specify only one of promotion_policy and "
                "migration_policy")
        requested_policy = (
            "eager"
            if promotion_policy is None
            and migration_policy is None
            else (
                promotion_policy
                if promotion_policy is not None
                else migration_policy
            )
        )
        policy = (
            SSDPromotionPolicy.for_key(requested_policy)
            if isinstance(requested_policy, str)
            else requested_policy
        )
        if not isinstance(policy, SSDPromotionPolicy):
            raise TypeError(
                "promotion_policy must be a key or SSDPromotionPolicy")
        gpu_hardware.validate()
        hbf_hardware.validate()
        layout.validate(hbf_hardware.card_count)
        if layout.key not in SUPPORTED_SSD_STAGED_HBF_LAYOUTS:
            raise ValueError(
                f"unsupported HBF layout {layout.key!r}")
        if gpu_hardware.gpu_count != 8:
            raise ValueError(
                "the staged design requires one physical eight-GPU "
                "P4D4 server")
        if hbf_hardware.card_count != 8:
            raise ValueError(
                "the staged design requires one physical eight-card "
                "HBF server")

        self.repo_root = Path(repo_root)
        self.gpu_hardware = gpu_hardware
        self.hbf_hardware = hbf_hardware
        self.hbf_layout = layout
        self.hbf_layouts = (layout,)
        self.hbf_server_count = 1
        self.gpu_node_id = gpu_node_id
        self.hbf_server_id = hbf_server_id
        self.hbf_execution_backend = "analytical_calendar"
        self.promotion_policy = policy
        self.migration_policy = policy
        self.validate_every_event = validate_every_event
        self.calendar = ResourceCalendar(
            retain_reservations=validate_every_event)
        # Compatibility aliases intentionally point to the same calendar.
        self.gpu_calendar = self.calendar
        self.hbf_calendar = self.calendar

        self.gpu_node = FiniteHBMTieredP4D4Node(
            repo_root=self.repo_root,
            hardware=gpu_hardware,
            node_id=gpu_node_id,
            policy="ssd_direct",
            resource_calendar=self.calendar,
            p_capacity_bytes_per_rank=p_capacity_bytes_per_rank,
            d_capacity_bytes_per_rank=d_capacity_bytes_per_rank,
            cpu_capacity_bytes=cpu_capacity_bytes,
            ssd_capacity_bytes=ssd_capacity_bytes,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            p_max_num_seqs=p_max_num_seqs,
            d_max_num_seqs=d_max_num_seqs,
            max_prefill_chunk_tokens=max_prefill_chunk_tokens,
            band=band,
            restore_execution_mode=restore_execution_mode,
            validate_every_event=validate_every_event,
            retain_detailed_history=validate_every_event,
        )
        self.gpu_lifecycle = self.gpu_node.lifecycle
        self.gpu_pool = self.gpu_node.pool
        self.restore_execution_mode = (
            self.gpu_node.restore_execution_mode)

        workspace = derive_lpddr_workspace_bytes(
            layout,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        )
        hbf_lpddr_ledger = PerGroupCapacityLedger(
            group_count=layout.replicas,
            capacity_bytes=(
                hbf_hardware.lpddr_capacity_bytes_per_card
                - workspace
            ),
        )
        gpu_source_root_bandwidth_gbps = min(
            gpu_hardware.pcie_root_bandwidth_gbps,
            gpu_hardware.decode_gpu_count
            * gpu_hardware.pcie_bandwidth_gbps_per_gpu,
        )
        self.hbf_lifecycle = FullModelHBFLifecycle(
            hardware=hbf_hardware,
            layout=layout,
            resource_calendar=self.calendar,
            lpddr_ledger=hbf_lpddr_ledger,
            gpu_source_root_bandwidth_gbps=(
                gpu_source_root_bandwidth_gbps),
            gpu_source_node_id=gpu_node_id,
            gpu_source_cpu_bandwidth_gbps=(
                gpu_hardware.cpu_memory_bandwidth_gbps),
            gpu_source_nic_bandwidth_gbps=(
                hbf_hardware.rdma_bandwidth_gbps),
            validate_every_event=validate_every_event,
            execution_backend="analytical_calendar",
            server_id=hbf_server_id,
        )
        self.hbf_pool = FullModelHBFServingPool(
            repo_root=self.repo_root,
            hardware=hbf_hardware,
            layout=layout,
            resource_calendar=self.calendar,
            lpddr_ledger=hbf_lpddr_ledger,
            placement_resolver=(
                self.hbf_lifecycle.placement_snapshot),
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_prefill_chunk_tokens=max_prefill_chunk_tokens,
            mixed_batch_latency_limit_ns=(
                hbf_mixed_batch_latency_limit_ns),
            band=band,
            validate_every_event=validate_every_event,
            retain_detailed_history=validate_every_event,
            execution_backend="analytical_calendar",
            server_id=hbf_server_id,
        )

        self.calls: dict[int, HybridCall] = {}
        self.sessions: dict[str, HybridSession] = {}
        self.metrics = SSDStagedHybridMetrics()
        self.current_ns = 0
        self._pending_call_ids: deque[int] = deque()
        self._user_completed_ids: deque[int] = deque()
        self._tier_call_by_request: dict[int, TieredNodeCall] = {}
        self._gpu_user_seen: set[int] = set()
        self._gpu_internal_seen: set[int] = set()
        self._next_gpu_call_index: dict[str, int] = {}
        self._live_hbf_request_ids: set[int] = set()
        self._last_submitted_call_index: dict[str, int] = {}
        self._last_submitted_request_id: dict[str, int] = {}
        self._gap_type_by_request: dict[int, Optional[str]] = {}
        self._offload_version_by_session: dict[str, int] = {}
        self._intent_by_session: dict[str, SSDPromotionIntent] = {}
        self._promotion_heap: list[tuple[int, int, str]] = []
        self._export_by_session: dict[str, SSDExportTicket] = {}
        self._import_by_session: dict[
            str, tuple[MigrationJob, SSDExportTicket]] = {}
        self._direct_migrations: dict[
            int, DirectMigrationTicket] = {}
        self._next_promotion_serial = 1
        self._spec_total_calls: dict[str, int] = {}
        self._submitted_count_by_session: dict[str, int] = {}

    def set_spec_total_calls(
            self, session_id: str, count: int) -> None:
        """Register a session's total in-window call count.

        This is the oracle side-channel: the workload spec knows every
        call a scheduled session will make, and the ``density_oracle``
        ranking divides context by the calls still to come.  Causal
        policies never read it.
        """

        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be non-empty")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise ValueError("count must be a positive integer")
        self._spec_total_calls[session_id] = count

    def _session_calls_so_far(self, session_id: str) -> int:
        """Calls this session has made, seeded history included."""

        return self._last_submitted_call_index.get(session_id, -1) + 1

    def _session_remaining_calls(self, session_id: str) -> int:
        total = self._spec_total_calls.get(session_id)
        if total is None:
            return 1
        submitted = self._submitted_count_by_session.get(session_id, 0)
        return max(1, total - submitted)

    def _placement_value(self, session_id: str, placement) -> float:
        """HBF-residency worth: restore savings per request served.

        ``context`` is the raw restore size.  ``density`` divides by the
        session's observed call count -- past calls predict future calls
        under the trace's heavy-tailed session lengths, so this is the
        causal estimate of savings per future request.  ``density_oracle``
        divides by the spec's actual remaining calls.
        """

        tokens = float(placement.total_tokens)
        mode = self.promotion_policy.rank_mode
        if mode == "density":
            return tokens / max(1, self._session_calls_so_far(session_id))
        if mode == "density_oracle":
            return tokens / self._session_remaining_calls(session_id)
        if mode == "calls":
            # HBF-residency worth is the inverse of the session's call
            # mass: the call-heavy heads belong on the GPU, D-resident,
            # where their frequent resumes hit live KV.
            return 1.0 / max(1, self._session_calls_so_far(session_id))
        if mode == "calls_oracle":
            return 1.0 / self._session_remaining_calls(session_id)
        return tokens

    def set_gap_type(
            self, request_id: int,
            gap_type: Optional[str]) -> None:
        if request_id in self.calls:
            raise RuntimeError(
                "gap metadata must be registered before release")
        if gap_type is not None and (
                not isinstance(gap_type, str)
                or not gap_type.strip()):
            raise ValueError(
                "gap_type must be a non-empty string or None")
        self._gap_type_by_request[request_id] = gap_type

    def _sync_hybrid_from_tier(
            self, call: HybridCall,
            tier_call: TieredNodeCall) -> None:
        call.operational_reuse_tokens = (
            tier_call.operational_hit_tokens)
        call.gpu_hit_tokens = tier_call.operational_hit_tokens
        call.prepare_start_ns = tier_call.prepare_start_ns
        call.prepare_completion_ns = tier_call.prepare_completion_ns
        if tier_call.pool_request is not None:
            tier_call.pool_request.arrival_ns = call.release_ns
        call.gpu_request = tier_call.pool_request
        if tier_call.state in {
            TieredCallState.PENDING,
            TieredCallState.PREPARING,
        }:
            call.state = HybridCallState.GPU_PREPARING
        elif tier_call.state == TieredCallState.EXECUTING:
            call.state = HybridCallState.GPU_EXECUTING

    @staticmethod
    def _execution_for_route(
            call: HybridCall, route: ResumeRoute) -> HybridExecution:
        if call.call_index == 0:
            return HybridExecution.GPU_FIRST_TURN
        if route.execution == ResumeExecution.HBF:
            return HybridExecution.HBF_READY
        if route.execution == ResumeExecution.GPU_RECOMPUTE:
            return HybridExecution.GPU_RECOMPUTE
        if route.migration_inflight:
            return HybridExecution.GPU_MIGRATION_INFLIGHT
        return HybridExecution.GPU_OWNED

    def _cancel_intent_only(self, session_id: str) -> None:
        if self._intent_by_session.pop(session_id, None) is not None:
            self.metrics.promotion_intents_canceled += 1

    def _abort_export_for_resume(
            self, session_id: str, *,
            now_ns: int) -> None:
        ticket = self._export_by_session.get(session_id)
        if ticket is None:
            return
        if ticket.status in {
            SSDExportStatus.RUNNING,
            SSDExportStatus.ABORT_PENDING,
            SSDExportStatus.READY,
        }:
            released = self.gpu_lifecycle.abort_ssd_export(
                ticket, now_ns=now_ns)
            if released:
                self._export_by_session.pop(session_id, None)
        elif ticket.status == SSDExportStatus.ABORTED:
            self._export_by_session.pop(session_id, None)
        elif ticket.status == SSDExportStatus.FINISHED:
            raise RuntimeError(
                "resume found a finished SSD export without HBF commit")

    def _route_call(
            self, call: HybridCall) -> tuple[
                ResumeRoute, Optional[TieredNodeCall],
                Optional[HBFServingRequest]]:
        if call.call_index == 0:
            self.hbf_lifecycle.register_session(
                call.session_id, now_ns=self.current_ns)
        placement = self.hbf_lifecycle.sessions[call.session_id]
        if self._should_demote_for_load(
                placement, now_ns=self.current_ns):
            self.hbf_lifecycle.demote_session(
                call.session_id, now_ns=self.current_ns)
            self.metrics.hbf_load_demotions += 1
        operational_reuse = min(
            call.prefix_reuse_tokens,
            placement.total_tokens,
            call.input_tokens,
        )
        route = self.hbf_lifecycle.route_resume(
            call.session_id,
            now_ns=self.current_ns,
            request_id=call.request_id,
            prefix_reuse_tokens=operational_reuse,
            input_tokens=call.input_tokens,
            lpddr_growth_tokens=(
                call.input_tokens
                - operational_reuse
                + call.output_tokens
                - 1
            ),
        )
        call.execution = self._execution_for_route(call, route)
        call.route_reason = route.reason
        call.migration_inflight_at_route = route.migration_inflight
        call.operational_reuse_tokens = operational_reuse
        self.metrics.routed_calls += 1

        if route.execution == ResumeExecution.HBF:
            if route.group_id is None:
                raise RuntimeError(
                    "HBF route lacks a replica group")
            request = HBFServingRequest(
                request_id=call.request_id,
                session_id=call.session_id,
                arrival_ns=call.release_ns,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                hbf_prefix_tokens=route.hbf_tokens,
                lpddr_prefix_tokens=route.lpddr_tokens,
                group_id=route.group_id,
            )
            call.hbf_request = request
            call.hbf_server_index = 0
            call.state = HybridCallState.HBF_EXECUTING
            self._live_hbf_request_ids.add(call.request_id)
            self.metrics.hbf_calls += 1
            return route, None, request

        self._cancel_intent_only(call.session_id)
        self._abort_export_for_resume(
            call.session_id, now_ns=self.current_ns)
        self._offload_version_by_session.pop(
            call.session_id, None)
        gpu_lineage = self.gpu_node.sessions.get(call.session_id)
        restarted_gpu_lineage = False
        if gpu_lineage is not None and gpu_lineage.ended:
            self.gpu_node.restart_ended_session(
                call.session_id, now_ns=self.current_ns)
            self._next_gpu_call_index[call.session_id] = 0
            restarted_gpu_lineage = True
        gpu_call_index = self._next_gpu_call_index.get(
            call.session_id, 0)
        if (
            gpu_call_index == 0
            and not restarted_gpu_lineage
            and operational_reuse > 0
        ):
            # The session's first GPU-side call under this counter is a
            # resume: it has been living on the HBF host and is coming back
            # with KV that is genuinely reusable.  This index counts calls on
            # this node, not turns in the conversation, so leaving it at zero
            # would assert a first turn that reuses an earlier prefix.
            #
            # The node's own submission cursor is the authority on where the
            # lineage stands, so follow it when the two have drifted apart,
            # and open the lineage one call in when the node has never seen
            # this session.
            submitted = self.gpu_node._last_submitted_call_index.get(
                call.session_id)
            if submitted is None:
                lineage = self.gpu_node.sessions.get(call.session_id)
                if lineage is None:
                    self.gpu_node.sessions[call.session_id] = (
                        TieredSessionLineage(
                            session_id=call.session_id, last_call_index=0))
                else:
                    lineage.last_call_index = 0
                self.gpu_node._last_submitted_call_index[
                    call.session_id] = 0
                gpu_call_index = 1
            else:
                gpu_call_index = submitted + 1
        tier_call = TieredNodeCall(
            request_id=call.request_id,
            session_id=call.session_id,
            call_index=gpu_call_index,
            release_ns=self.current_ns,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            prefix_reuse_tokens=(
                0 if restarted_gpu_lineage
                else operational_reuse),
            has_successor=call.has_successor,
        )
        self._next_gpu_call_index[call.session_id] = (
            gpu_call_index + 1)
        self._tier_call_by_request[call.request_id] = tier_call
        self.metrics.gpu_calls += 1
        if call.execution == HybridExecution.GPU_FIRST_TURN:
            self.metrics.gpu_first_turn_calls += 1
        elif call.execution == HybridExecution.GPU_RECOMPUTE:
            self.metrics.gpu_recompute_calls += 1
        elif call.execution == HybridExecution.GPU_MIGRATION_INFLIGHT:
            self.metrics.gpu_import_inflight_calls += 1
        else:
            self.metrics.gpu_owned_calls += 1
        return route, tier_call, None

    def submit_many(
            self, calls: Iterable[HybridCall], *,
            now_ns: int) -> None:
        values = list(calls)
        seen_ids: set[int] = set()
        proposed_last = dict(self._last_submitted_call_index)
        proposed_predecessors = {
            session_id: self.calls[request_id]
            for session_id, request_id
            in self._last_submitted_request_id.items()
        }
        for call in values:
            call.validate()
            if call.release_ns != now_ns:
                raise ValueError(
                    "hybrid calls must be submitted at logical release")
            if call.request_id in self.calls or call.request_id in seen_ids:
                raise ValueError(
                    f"duplicate request_id={call.request_id}")
            prior_index = proposed_last.get(call.session_id, -1)
            if call.call_index != prior_index + 1:
                raise ValueError(
                    "session calls must be submitted in contiguous order")
            predecessor = proposed_predecessors.get(call.session_id)
            if predecessor is not None:
                if not predecessor.has_successor:
                    raise ValueError(
                        "cannot submit after a terminal predecessor")
                if predecessor.user_completion_ns is None:
                    raise ValueError(
                        "successor cannot precede predecessor completion")
                if call.release_ns < predecessor.user_completion_ns:
                    raise ValueError(
                        "successor release precedes predecessor completion")
            session = self.sessions.get(call.session_id)
            if session is not None and session.ended:
                raise ValueError(
                    f"session {call.session_id!r} already ended")
            proposed_last[call.session_id] = call.call_index
            proposed_predecessors[call.session_id] = call
            seen_ids.add(call.request_id)

        self.advance(now_ns, defer_schedule=True)
        for call in values:
            # Arrival visibility wins a same-timestamp promotion threshold,
            # even if the predecessor's internal handoff is still pending.
            if call.call_index > 0:
                self._cancel_intent_only(call.session_id)
            self.calls[call.request_id] = call
            self.sessions.setdefault(
                call.session_id,
                HybridSession(session_id=call.session_id),
            )
            self._last_submitted_call_index[call.session_id] = (
                call.call_index)
            self._last_submitted_request_id[call.session_id] = (
                call.request_id)
            self._submitted_count_by_session[call.session_id] = (
                self._submitted_count_by_session.get(
                    call.session_id, 0) + 1)
            self._pending_call_ids.append(call.request_id)
            self.metrics.submitted_calls += 1
        self.metrics.max_pending_calls = max(
            self.metrics.max_pending_calls,
            len(self._pending_call_ids),
        )
        self.flush_scheduling(now_ns)

    def submit(self, call: HybridCall, *, now_ns: int) -> None:
        self.submit_many((call,), now_ns=now_ns)

    def _admit_pending(self, now_ns: int) -> None:
        deferred: deque[int] = deque()
        gpu_calls: list[TieredNodeCall] = []
        hbf_requests: list[HBFServingRequest] = []
        while self._pending_call_ids:
            request_id = self._pending_call_ids.popleft()
            call = self.calls[request_id]
            session = self.sessions[call.session_id]
            if call.call_index != session.last_internal_call_index + 1:
                deferred.append(request_id)
                continue
            _, tier_call, hbf_request = self._route_call(call)
            if tier_call is not None:
                gpu_calls.append(tier_call)
            if hbf_request is not None:
                hbf_requests.append(hbf_request)
        self._pending_call_ids = deferred
        if gpu_calls:
            gpu_calls.sort(key=lambda value: value.request_id)
            self.gpu_node.submit_many(gpu_calls, now_ns=now_ns)
            for tier_call in gpu_calls:
                self._sync_hybrid_from_tier(
                    self.calls[tier_call.request_id],
                    tier_call,
                )
        if hbf_requests:
            hbf_requests.sort(key=lambda value: value.request_id)
            self.hbf_pool.submit_many(
                hbf_requests,
                now_ns=now_ns,
                defer_schedule=True,
            )

    def _schedule_promotion(
            self, call: HybridCall, *,
            completion_ns: int) -> None:
        gap_type = self._gap_type_by_request.get(
            call.request_id)
        if not self.promotion_policy.eligible_gap(gap_type):
            if self.promotion_policy.mode != "never":
                self.metrics.promotion_intents_filtered += 1
            return
        # The turn's own tokens are the post-turn context; the placement
        # record is only updated at internal completion, which has not
        # happened yet at this user-completion boundary.
        turn_total_tokens = call.input_tokens + call.output_tokens - 1
        if turn_total_tokens < self.promotion_policy.min_context_tokens:
            self.metrics.promotion_intents_filtered += 1
            return
        placement = self.hbf_lifecycle.sessions[call.session_id]
        serial = self._next_promotion_serial
        self._next_promotion_serial += 1
        due_ns = (
            None
            if self.promotion_policy.ssd_checkpoint_age_ns is not None
            else (
                completion_ns
                + self.promotion_policy.idle_delay_ns
            )
        )
        intent = SSDPromotionIntent(
            session_id=call.session_id,
            predecessor_request_id=call.request_id,
            snapshot_version=placement.version + 1,
            user_completion_ns=completion_ns,
            due_ns=due_ns,
            gap_type=gap_type,
            serial=serial,
        )
        self._intent_by_session[call.session_id] = intent
        if due_ns is not None:
            heapq.heappush(
                self._promotion_heap,
                (due_ns, intent.serial, intent.session_id),
            )
        self.metrics.promotion_intents_scheduled += 1

    def _arm_intent(
            self, intent: SSDPromotionIntent, *,
            due_ns: int) -> None:
        serial = self._next_promotion_serial
        self._next_promotion_serial += 1
        intent.serial = serial
        intent.due_ns = due_ns
        intent.due_reached = False
        heapq.heappush(
            self._promotion_heap,
            (due_ns, serial, intent.session_id),
        )

    def _broadcast_human_gap(self, *, now_ns: int) -> None:
        if not self.promotion_policy.human_gap_broadcast:
            return
        self.metrics.human_gap_broadcasts += 1
        broadcast_count = 0
        for session_id, intent in sorted(
                self._intent_by_session.items()):
            hbf_record = self.hbf_lifecycle.sessions.get(
                session_id)
            tier_record = self.gpu_lifecycle.sessions.get(
                session_id)
            if (
                hbf_record is None
                or tier_record is None
                or hbf_record.state != PlacementState.SSD_READY
                or tier_record.state != TierSessionState.SSD_READY
                or hbf_record.version != intent.snapshot_version
                or session_id in self._import_by_session
                or session_id in self._export_by_session
                or intent.due_reached
            ):
                continue
            intent.human_due_now = True
            self._arm_intent(intent, due_ns=now_ns)
            broadcast_count += 1
        self.metrics.human_gap_broadcast_intents += broadcast_count

    def _observe_gap_event(
            self, call: HybridCall, *, now_ns: int) -> None:
        if not self.promotion_policy.human_gap_broadcast:
            return
        gap_type = self._gap_type_by_request.get(call.request_id)
        if (
            gap_type is not None
            and gap_type.strip().lower() == "human"
        ):
            current_intent = self._intent_by_session.get(
                call.session_id)
            if (
                current_intent is not None
                and current_intent.predecessor_request_id
                == call.request_id
            ):
                current_intent.human_due_now = True
                self.metrics.human_gap_broadcast_intents += 1
            self._broadcast_human_gap(now_ns=now_ns)

    def _consume_gpu_notifications(self, now_ns: int) -> None:
        for tier_call in self.gpu_node.pop_completed():
            request_id = tier_call.request_id
            if request_id in self._gpu_user_seen:
                raise RuntimeError(
                    "duplicate GPU user completion")
            self._gpu_user_seen.add(request_id)
            call = self.calls[request_id]
            self._sync_hybrid_from_tier(call, tier_call)
            if tier_call.user_completion_ns != now_ns:
                raise RuntimeError(
                    "GPU completion at wrong node timestamp")
            call.user_completion_ns = now_ns
            call.state = HybridCallState.USER_COMPLETE
            self._user_completed_ids.append(request_id)
            self.metrics.user_completed_calls += 1
            if call.has_successor:
                self._schedule_promotion(
                    call, completion_ns=now_ns)
            self._observe_gap_event(
                call, now_ns=now_ns)

        for request_id, tier_call in sorted(
                self._tier_call_by_request.items()):
            if (
                request_id in self._gpu_internal_seen
                or tier_call.state
                != TieredCallState.INTERNAL_COMPLETE
            ):
                continue
            self._gpu_internal_seen.add(request_id)
            call = self.calls[request_id]
            self._sync_hybrid_from_tier(call, tier_call)
            if tier_call.internal_completion_ns != now_ns:
                raise RuntimeError(
                    "GPU internal completion at wrong node timestamp")
            total_tokens = (
                call.input_tokens + call.output_tokens - 1)
            direct_candidate = bool(
                self.promotion_policy.direct_d_resume
                and call.call_index > 0
                and tier_call.prepare_source == Tier.D
                and call.has_successor
                and total_tokens
                >= self.promotion_policy.min_context_tokens
            )
            start_direct = direct_candidate
            if (
                start_direct
                and self.promotion_policy.load_aware_admission
                and not self._load_aware_allows_promotion(now_ns)
            ):
                start_direct = False
                self.metrics.promotion_load_deferrals += 1
            direct_job = self.hbf_lifecycle.complete_gpu_turn(
                call.session_id,
                now_ns=now_ns,
                total_tokens=total_tokens,
                has_successor=call.has_successor,
                start_migration=start_direct,
            )
            if direct_job is not None:
                tier_record = self.gpu_lifecycle.sessions[
                    call.session_id]
                source_copy_id = tier_record.primary_copy_id
                source_copy = (
                    None
                    if source_copy_id is None
                    else self.gpu_lifecycle.copies.get(
                        source_copy_id)
                )
                if (
                    tier_record.state != TierSessionState.D_READY
                    or source_copy is None
                    or source_copy.tier != Tier.D
                ):
                    raise RuntimeError(
                        "direct migration lacks a committed D source")
                source_copy.foreground_pins += 1
                self._direct_migrations[
                    direct_job.job_id] = DirectMigrationTicket(
                        job=direct_job,
                        source_copy_id=source_copy_id,
                    )
                self.metrics.direct_migrations_started += 1
            elif start_direct:
                self.metrics.direct_migration_capacity_deferrals += 1
            session = self.sessions[call.session_id]
            session.last_internal_call_index = call.call_index
            session.ended = not call.has_successor
            call.internal_completion_ns = now_ns
            call.state = HybridCallState.INTERNAL_COMPLETE
            self.metrics.internal_completed_calls += 1
            if call.has_successor and direct_job is None:
                version = self.hbf_lifecycle.sessions[
                    call.session_id].version
                self._offload_version_by_session[
                    call.session_id] = version
                self.metrics.proactive_offloads_requested += 1
                self._try_start_offload(
                    call.session_id, now_ns=now_ns)

    def _sync_live_gpu_calls(self) -> None:
        for request_id, tier_call in (
                self._tier_call_by_request.items()):
            if request_id in self._gpu_internal_seen:
                continue
            self._sync_hybrid_from_tier(
                self.calls[request_id], tier_call)

    def _consume_hbf_notifications(self, now_ns: int) -> None:
        for request in self.hbf_pool.pop_completed():
            request_id = request.request_id
            if request_id not in self._live_hbf_request_ids:
                raise RuntimeError(
                    "unknown HBF completion")
            self._live_hbf_request_ids.remove(request_id)
            call = self.calls[request_id]
            call.user_completion_ns = request.completion_ns
            call.internal_completion_ns = request.completion_ns
            call.state = HybridCallState.INTERNAL_COMPLETE
            self._user_completed_ids.append(request_id)
            self.metrics.user_completed_calls += 1
            self.metrics.internal_completed_calls += 1
            self.hbf_lifecycle.complete_hbf_turn(
                call.session_id,
                now_ns=now_ns,
                total_tokens=(
                    call.input_tokens + call.output_tokens - 1),
                has_successor=call.has_successor,
            )
            session = self.sessions[call.session_id]
            session.last_internal_call_index = call.call_index
            session.ended = not call.has_successor
            self._observe_gap_event(
                call, now_ns=now_ns)

    def _try_start_offload(
            self, session_id: str, *,
            now_ns: int) -> None:
        snapshot_version = self._offload_version_by_session.get(
            session_id)
        if snapshot_version is None:
            return
        tier_record = self.gpu_lifecycle.sessions[session_id]
        hbf_record = self.hbf_lifecycle.sessions[session_id]
        if tier_record.state == TierSessionState.SSD_READY:
            published = self.hbf_lifecycle.publish_ssd_checkpoint(
                session_id,
                now_ns=now_ns,
                snapshot_version=snapshot_version,
            )
            self._offload_version_by_session.pop(session_id, None)
            if published:
                self.metrics.ssd_checkpoints_published += 1
                intent = self._intent_by_session.get(session_id)
                checkpoint_age_ns = (
                    self.promotion_policy.ssd_checkpoint_age_ns)
                if (
                    intent is not None
                    and checkpoint_age_ns is not None
                    and intent.snapshot_version == snapshot_version
                ):
                    intent.ssd_publication_ns = now_ns
                    self._arm_intent(
                        intent,
                        due_ns=(
                            now_ns
                            if intent.human_due_now
                            else now_ns + checkpoint_age_ns
                        ),
                    )
            return
        if (
            tier_record.state == TierSessionState.D_READY
            and hbf_record.state == PlacementState.GPU_READY
        ):
            started_before = (
                self.gpu_lifecycle.metrics.d_to_ssd_started)
            self.gpu_lifecycle.demote(
                session_id, now_ns=now_ns)
            if (
                self.gpu_lifecycle.metrics.d_to_ssd_started
                > started_before
            ):
                self.metrics.proactive_offloads_started += 1

    def _reconcile_offloads(self, now_ns: int) -> None:
        for session_id in tuple(
                self._offload_version_by_session):
            tier_record = self.gpu_lifecycle.sessions.get(
                session_id)
            hbf_record = self.hbf_lifecycle.sessions.get(
                session_id)
            if tier_record is None or hbf_record is None:
                self._offload_version_by_session.pop(
                    session_id, None)
                continue
            if hbf_record.state != PlacementState.GPU_READY:
                self._offload_version_by_session.pop(
                    session_id, None)
                continue
            self._try_start_offload(
                session_id, now_ns=now_ns)

    def _cleanup_aborted_exports(self) -> None:
        for session_id, ticket in tuple(
                self._export_by_session.items()):
            if ticket.status == SSDExportStatus.ABORTED:
                self._export_by_session.pop(session_id, None)

    def _reconcile_direct_migrations(self, now_ns: int) -> None:
        for job_id, ticket in tuple(
                self._direct_migrations.items()):
            job = ticket.job
            record = self.hbf_lifecycle.sessions[job.session_id]
            if job_id in record.migration_job_ids:
                continue
            committed = bool(
                record.state == PlacementState.HBF_READY
                and record.version == job.version
                and record.group_id == job.group_id
            )
            source_copy = self.gpu_lifecycle.copies.get(
                ticket.source_copy_id)
            if (
                source_copy is None
                or source_copy.foreground_pins <= 0
            ):
                raise RuntimeError(
                    "direct migration lost its D source pin")
            source_copy.foreground_pins -= 1
            self.gpu_lifecycle._maybe_release_retired(source_copy)
            if committed:
                tier_record = self.gpu_lifecycle.sessions[
                    job.session_id]
                if tier_record.state != TierSessionState.ENDED:
                    self.gpu_lifecycle.end(
                        job.session_id, now_ns=now_ns)
                gpu_lineage = self.gpu_node.sessions.get(
                    job.session_id)
                if gpu_lineage is not None:
                    gpu_lineage.ended = True
                self._offload_version_by_session.pop(
                    job.session_id, None)
                self._intent_by_session.pop(
                    job.session_id, None)
                self.metrics.direct_migrations_committed += 1
            else:
                self.metrics.direct_migrations_stale += 1
            self._direct_migrations.pop(job_id, None)

    def _reconcile_imports(self, now_ns: int) -> None:
        for session_id, (job, ticket) in tuple(
                self._import_by_session.items()):
            record = self.hbf_lifecycle.sessions[session_id]
            if job.job_id in record.migration_job_ids:
                continue
            if (
                record.state == PlacementState.HBF_READY
                and record.version == job.version
            ):
                if ticket.status != SSDExportStatus.READY:
                    raise RuntimeError(
                        "committed HBF import lost its SSD export")
                self.gpu_lifecycle.finish_ssd_export(
                    ticket, now_ns=now_ns)
                self.gpu_lifecycle.end(
                    session_id, now_ns=now_ns)
                gpu_lineage = self.gpu_node.sessions.get(session_id)
                if gpu_lineage is not None:
                    gpu_lineage.ended = True
                self.metrics.hbf_imports_committed += 1
            else:
                if ticket.status in {
                    SSDExportStatus.RUNNING,
                    SSDExportStatus.ABORT_PENDING,
                    SSDExportStatus.READY,
                }:
                    self.gpu_lifecycle.abort_ssd_export(
                        ticket, now_ns=now_ns)
                self.metrics.hbf_imports_stale += 1
            self._import_by_session.pop(session_id, None)
            self._export_by_session.pop(session_id, None)
            self._intent_by_session.pop(session_id, None)

    def _prune_promotion_heap(self) -> None:
        while self._promotion_heap:
            due_ns, serial, session_id = self._promotion_heap[0]
            intent = self._intent_by_session.get(session_id)
            if (
                intent is not None
                and intent.serial == serial
                and intent.due_ns == due_ns
                and not intent.due_reached
            ):
                break
            heapq.heappop(self._promotion_heap)

    def _queue_intent_retry(
            self, intent: SSDPromotionIntent, *,
            now_ns: int) -> None:
        self._arm_intent(
            intent,
            due_ns=now_ns + self.promotion_policy.retry_ns,
        )

    def _mark_due_promotions(self, now_ns: int) -> None:
        self._prune_promotion_heap()
        while (
            self._promotion_heap
            and self._promotion_heap[0][0] <= now_ns
        ):
            _, serial, session_id = heapq.heappop(
                self._promotion_heap)
            intent = self._intent_by_session.get(session_id)
            if intent is None or intent.serial != serial:
                self._prune_promotion_heap()
                continue
            intent.due_reached = True
            self.metrics.promotion_thresholds_reached += 1
            self._prune_promotion_heap()

    def _calendar_backlog_ns(
            self, *, prefixes: tuple[str, ...],
            now_ns: int) -> int:
        return max(
            (
                max(0, available_ns - now_ns)
                for resource, available_ns
                in self.calendar.available_ns.items()
                if resource.startswith(prefixes)
            ),
            default=0,
        )

    def _load_aware_allows_promotion(self, now_ns: int) -> bool:
        """Compare only causally visible queue and calendar pressure."""

        gpu_score, hbf_score = self._load_scores(now_ns)
        if hbf_score == 0.0:
            return True
        return hbf_score <= (
            gpu_score * (1.0 - self.promotion_policy.load_hysteresis)
        )

    def _load_scores(
            self, now_ns: int, *,
            include_pressure: bool = True) -> tuple[float, float]:
        # Both queue sums count outstanding work items per server; neither
        # side is normalised by worker count so the scales stay comparable
        # across layouts (tp8 has one HBF worker, tp4 has two).
        gpu_queue_work = sum((
            len(self.gpu_pool.p_worker.waiting),
            len(self.gpu_pool.d_worker.waiting),
            int(self.gpu_pool.p_worker.inflight is not None),
            int(self.gpu_pool.d_worker.inflight is not None),
            len(self.gpu_node._pending_call_ids),
            len(self.gpu_node._ready_call_ids),
        ))
        hbf_queue_work = sum(
            len(worker.waiting)
            + len(worker.prefill_drain)
            + len(worker.active_decode)
            + int(worker.inflight is not None)
            + int(worker.pending_launch_ns is not None)
            for worker in self.hbf_pool.workers
        )
        gpu_backlog = self._calendar_backlog_ns(
            prefixes=(f"gpu-node-{self.gpu_node_id}-p-model",
                      f"gpu-node-{self.gpu_node_id}-d-model",
                      f"gpu-node-{self.gpu_node_id}-pd-fabric"),
            now_ns=now_ns,
        )
        hbf_backlog = self._calendar_backlog_ns(
            prefixes=("hbf-group-", "hbf-card-"),
            now_ns=now_ns,
        )
        scale_ns = max(1, self.promotion_policy.retry_ns)
        gpu_hbm_pressure = (
            self.gpu_lifecycle.d_ledger.used_bytes
            / self.gpu_lifecycle.d_ledger.capacity_bytes
        )
        lpddr_ledger = self.hbf_lifecycle.lpddr_ledger
        hbf_lpddr_pressure = max(
            (
                lpddr_ledger.used_bytes(group.group_id)
                / lpddr_ledger.capacity_bytes
                for group in self.hbf_lifecycle.groups
            ),
            default=0.0,
        )
        hbf_media_pressure = max(
            (
                reserved / self.hbf_lifecycle.usable_bytes_per_card
                for card_bytes in (
                    self.hbf_lifecycle._reserved_bytes_by_card.values())
                for reserved in card_bytes.values()
            ),
            default=0.0,
        )
        gpu_dynamic = float(gpu_queue_work) + gpu_backlog / scale_ns
        hbf_dynamic = float(hbf_queue_work) + hbf_backlog / scale_ns
        if not include_pressure:
            return gpu_dynamic, hbf_dynamic
        return (
            gpu_dynamic + gpu_hbm_pressure,
            hbf_dynamic + hbf_lpddr_pressure + hbf_media_pressure,
        )

    def _should_demote_for_load(
            self, placement, *, now_ns: int) -> bool:
        """Demote an idle HBF resident on strong, persistent imbalance.

        The band is deliberately wider than the promotion gate's so the
        two valves cannot flap a session back and forth: promotion needs
        the HBF side to be the quieter one, demotion needs it to be
        overloaded by ``demotion_hysteresis`` beyond parity.  Only the
        lower half of the HBF population by placement value (see
        ``_placement_value``) is eligible: those are the residents whose
        avoided restore per served request is smallest, so shedding them
        buys the most relief per unit of recompute.
        """

        hysteresis = self.promotion_policy.demotion_hysteresis
        if not hysteresis:
            return False
        if (
            placement.state != PlacementState.HBF_READY
            or placement.append_job_ids
            or placement.lpddr_tokens != 0
            or placement.committed_hbf_tokens <= 0
        ):
            return False
        # Only dynamic load counts here: media occupancy is residency,
        # not overload, and against an idle GPU (score zero) any static
        # pressure term would demote unconditionally.  The floor of four
        # queued work items keeps mid-load transients from triggering
        # demotions whose recompute tail costs more than the relief --
        # the ablation showed the floor of one firing at loads the HBF
        # host was handling fine.
        gpu_score, hbf_score = self._load_scores(
            now_ns, include_pressure=False)
        if hbf_score <= max(4.0, gpu_score * (1.0 + hysteresis)):
            return False
        # A demoted session that does not fit in free D-HBM would pay a
        # restore on every resume instead of staying D-resident -- the
        # exact thrash the density ablation exposed.  Keep it on HBF.
        if (
            self.gpu_lifecycle._per_rank_bytes(placement.total_tokens)
            > self.gpu_lifecycle.d_ledger.free_bytes
        ):
            return False
        resident_values = sorted(
            self._placement_value(record.session_id, record)
            for record in self.hbf_lifecycle.sessions.values()
            if record.state == PlacementState.HBF_READY
        )
        if len(resident_values) < 2:
            return False
        median = (
            resident_values[len(resident_values) // 2]
            if len(resident_values) % 2
            else (
                resident_values[len(resident_values) // 2 - 1]
                + resident_values[len(resident_values) // 2]
            ) / 2
        )
        return self._placement_value(
            placement.session_id, placement) < median

    def _promotion_priority(
            self, item: tuple[str, SSDPromotionIntent]) -> tuple:
        """Order due intents by HBF-residency worth, highest first.

        When the load gate admits only k promotions they should go to
        the k sessions whose placement value (see ``_placement_value``)
        is largest, not the lexicographically first ones.  The serial
        tie-break keeps the order deterministic.
        """

        session_id, intent = item
        placement = self.hbf_lifecycle.sessions.get(session_id)
        value = (
            0.0 if placement is None
            else self._placement_value(session_id, placement)
        )
        return (-value, intent.serial, session_id)

    def _progress_promotions(self, now_ns: int) -> None:
        self._mark_due_promotions(now_ns)
        for session_id, intent in tuple(
                sorted(
                    self._intent_by_session.items(),
                    key=self._promotion_priority,
                )):
            if not intent.due_reached:
                continue
            if session_id in self._import_by_session:
                continue
            if (
                self.promotion_policy.load_aware_admission
                and not self._load_aware_allows_promotion(now_ns)
            ):
                self.metrics.promotion_load_deferrals += 1
                self._queue_intent_retry(
                    intent, now_ns=now_ns)
                continue
            hbf_record = self.hbf_lifecycle.sessions[session_id]
            tier_record = self.gpu_lifecycle.sessions[session_id]
            ticket = self._export_by_session.get(session_id)
            if ticket is None:
                if (
                    hbf_record.state != PlacementState.SSD_READY
                    or tier_record.state
                    != TierSessionState.SSD_READY
                ):
                    continue
                ticket = self.gpu_lifecycle.begin_ssd_export(
                    session_id, now_ns=now_ns)
                if ticket is None:
                    self.metrics.ssd_export_capacity_deferrals += 1
                    self._queue_intent_retry(
                        intent, now_ns=now_ns)
                    continue
                self._export_by_session[session_id] = ticket
                self.metrics.ssd_exports_started += 1
            if ticket.status == SSDExportStatus.RUNNING:
                continue
            if ticket.status != SSDExportStatus.READY:
                self._intent_by_session.pop(session_id, None)
                continue
            if not self.gpu_lifecycle.ssd_export_publication_valid(
                    ticket):
                self.gpu_lifecycle.abort_ssd_export(
                    ticket, now_ns=now_ns)
                self._export_by_session.pop(session_id, None)
                self._intent_by_session.pop(session_id, None)
                continue
            job = self.hbf_lifecycle.start_import_from_ssd(
                session_id, now_ns=now_ns)
            if job is None:
                self.metrics.hbf_import_capacity_deferrals += 1
                self._queue_intent_retry(
                    intent, now_ns=now_ns)
                continue
            self._import_by_session[session_id] = (job, ticket)
            self.metrics.hbf_imports_started += 1

    def _next_raw_event_ns(self) -> Optional[int]:
        values = []
        for value in (
            self.gpu_node.next_event_ns(),
            self.hbf_pool.next_event_ns(),
            self.hbf_lifecycle.next_completion_ns(),
        ):
            if value is not None:
                values.append(value)
        self._prune_promotion_heap()
        if self._promotion_heap:
            values.append(self._promotion_heap[0][0])
        return min(values) if values else None

    def advance(
            self, now_ns: int, *,
            defer_schedule: bool = False) -> None:
        if now_ns < self.current_ns:
            raise ValueError(
                "staged hybrid time cannot move backwards")
        while True:
            event_ns = self._next_raw_event_ns()
            if event_ns is None or event_ns > now_ns:
                break
            # Transfer and HBF publication callbacks win same-time arrivals.
            self.hbf_lifecycle.advance(event_ns)
            self._reconcile_direct_migrations(event_ns)
            self._reconcile_imports(event_ns)
            self.hbf_pool.advance(
                event_ns, defer_schedule=True)
            self.gpu_node.advance(
                event_ns, defer_schedule=True)
            self.current_ns = event_ns
            self._sync_live_gpu_calls()
            self._consume_hbf_notifications(event_ns)
            self._consume_gpu_notifications(event_ns)
            self._reconcile_offloads(event_ns)
            self._cleanup_aborted_exports()
            if defer_schedule and event_ns == now_ns:
                break
            self.flush_scheduling(event_ns)
        self.hbf_lifecycle.advance(now_ns)
        self._reconcile_direct_migrations(now_ns)
        self._reconcile_imports(now_ns)
        self.hbf_pool.advance(now_ns, defer_schedule=True)
        self.gpu_node.advance(now_ns, defer_schedule=True)
        self.current_ns = now_ns
        self._sync_live_gpu_calls()
        self._consume_hbf_notifications(now_ns)
        self._consume_gpu_notifications(now_ns)
        self._reconcile_offloads(now_ns)
        self._cleanup_aborted_exports()
        if not defer_schedule:
            self.flush_scheduling(now_ns)
        if self.validate_every_event:
            self.assert_invariants()

    def flush_scheduling(self, now_ns: int) -> None:
        if not (
            now_ns
            == self.current_ns
            == self.gpu_node.current_ns
            == self.hbf_pool.current_ns
            == self.hbf_lifecycle.current_ns
        ):
            raise ValueError(
                "flush_scheduling requires synchronized clocks")
        # Arrivals are admitted and routed before promotion thresholds.
        self._admit_pending(now_ns)
        self._reconcile_offloads(now_ns)
        self._progress_promotions(now_ns)
        self.gpu_node.flush_scheduling(now_ns)
        self._sync_live_gpu_calls()
        self.hbf_pool.flush_scheduling(now_ns)

    def next_event_ns(self) -> Optional[int]:
        return self._next_raw_event_ns()

    def pop_completed(self) -> list[HybridCall]:
        result = []
        while self._user_completed_ids:
            result.append(self.calls[
                self._user_completed_ids.popleft()])
        return result

    def has_pending_external(self) -> bool:
        return False

    def _deadlock_detail(self) -> str:
        return (
            f"pending_calls={list(self._pending_call_ids)[:8]}, "
            f"offloads={sorted(self._offload_version_by_session)}, "
            f"promotions={sorted(self._intent_by_session)}, "
            f"exports={sorted(self._export_by_session)}, "
            f"imports={sorted(self._import_by_session)}, "
            f"direct_migrations={sorted(self._direct_migrations)}"
        )

    def run_until_idle(self) -> list[HybridCall]:
        result = self.pop_completed()
        while self.next_event_ns() is not None:
            event_ns = self.next_event_ns()
            assert event_ns is not None
            self.advance(event_ns)
            result.extend(self.pop_completed())
        if (
            self._pending_call_ids
            or any(
                call.state != HybridCallState.INTERNAL_COMPLETE
                for call in self.calls.values()
            )
        ):
            raise HybridDeadlockError(self._deadlock_detail())
        self.assert_invariants()
        return result

    def assert_invariants(self) -> None:
        self.gpu_node.assert_invariants()
        self.hbf_lifecycle.assert_invariants()
        self.hbf_pool.assert_invariants()
        if not (
            self.calendar is self.gpu_node.calendar
            is self.gpu_lifecycle.calendar
            is self.gpu_pool.calendar
            is self.hbf_lifecycle.calendar
            is self.hbf_pool.calendar
        ):
            raise AssertionError(
                "staged hybrid components do not share one calendar")
        if not (
            self.current_ns
            == self.gpu_node.current_ns
            == self.hbf_lifecycle.current_ns
            == self.hbf_pool.current_ns
        ):
            raise AssertionError(
                "staged hybrid component clocks diverged")
        if self.hbf_server_count != 1:
            raise AssertionError(
                "staged hybrid must have exactly one HBF server")
        pending_ids = list(self._pending_call_ids)
        if len(pending_ids) != len(set(pending_ids)):
            raise AssertionError(
                "duplicate staged pending call")
        expected_pending = {
            call.request_id
            for call in self.calls.values()
            if call.state == HybridCallState.PENDING
        }
        if set(pending_ids) != expected_pending:
            raise AssertionError(
                "staged pending queue/state mismatch")
        for session_id, ticket in self._export_by_session.items():
            if ticket.session_id != session_id:
                raise AssertionError(
                    "SSD export session index mismatch")
        for session_id, (job, ticket) in (
                self._import_by_session.items()):
            if (
                job.session_id != session_id
                or ticket.session_id != session_id
            ):
                raise AssertionError(
                    "HBF import session index mismatch")
            placement = self.hbf_lifecycle.sessions[session_id]
            import_is_publishable = bool(
                placement.state == PlacementState.MIGRATING
                and job.job_id in placement.migration_job_ids
                and placement.generation == job.generation
                and placement.version == job.version
            )
            if import_is_publishable:
                if (
                    ticket.status != SSDExportStatus.READY
                    or ticket.resources_released
                ):
                    raise AssertionError(
                        "publishable HBF import released its SSD source")
            elif ticket.status not in {
                SSDExportStatus.ABORTED,
                SSDExportStatus.ABORT_PENDING,
            }:
                raise AssertionError(
                    "invalidated HBF import retained a live export")
        for job_id, ticket in self._direct_migrations.items():
            job = ticket.job
            source = self.gpu_lifecycle.copies.get(
                ticket.source_copy_id)
            placement = self.hbf_lifecycle.sessions.get(
                job.session_id)
            if (
                job.job_id != job_id
                or source is None
                or source.session_id != job.session_id
                or source.tier != Tier.D
                or source.foreground_pins <= 0
                or placement is None
                or job_id not in placement.migration_job_ids
            ):
                raise AssertionError(
                    "direct migration source ownership mismatch")
        if (
            self.metrics.submitted_calls != len(self.calls)
            or self.metrics.routed_calls
            != self.metrics.gpu_calls + self.metrics.hbf_calls
            or self.metrics.user_completed_calls
            < self.metrics.internal_completed_calls
            or self.metrics.direct_migrations_started
            != (
                self.metrics.direct_migrations_committed
                + self.metrics.direct_migrations_stale
                + len(self._direct_migrations)
            )
        ):
            raise AssertionError(
                "staged hybrid metric conservation failed")

    def report(self) -> Mapping[str, Any]:
        return {
            "mode": "ssd_staged_gpu_hbf_node",
            "architecture": {
                "gpu_server_count": 1,
                "gpu_server": "one_4p4d_h100_server",
                "gpu_tier_policy": "ssd_direct",
                "gpu_restore_execution_mode": (
                    self.restore_execution_mode),
                "hbf_server_count": 1,
                "hbf_server": "one_8card_full_model_hbf_server",
                "hbf_card_count": self.hbf_hardware.card_count,
                "hbf_layout": self.hbf_layout.key,
                "hbf_mixed_batch_latency_limit_ns": (
                    self.hbf_pool.mixed_batch_latency_limit_ns),
                "execution_backend": "analytical_calendar",
                "shared_resource_calendar": True,
            },
            "policy": asdict(self.promotion_policy),
            "metrics": asdict(self.metrics),
            "current_ns": self.current_ns,
            "calendar": self.calendar.report(),
            "gpu_node": self.gpu_node.report(),
            "hbf_lifecycle": self.hbf_lifecycle.report(),
            "hbf_pool": self.hbf_pool.report(),
            "active_state": {
                "pending_call_ids": list(self._pending_call_ids),
                "offload_sessions": sorted(
                    self._offload_version_by_session),
                "promotion_sessions": sorted(
                    self._intent_by_session),
                "export_sessions": sorted(
                    self._export_by_session),
                "import_sessions": sorted(
                    self._import_by_session),
                "direct_migration_job_ids": sorted(
                    self._direct_migrations),
            },
            "calls": {
                request_id: {
                    **asdict(call),
                    "state": call.state.value,
                    "execution": (
                        None if call.execution is None
                        else call.execution.value),
                    "ttft_ns": call.ttft_ns,
                    "tpot_ns": call.tpot_ns,
                }
                for request_id, call in sorted(self.calls.items())
            },
        }


class SSDStagedGPUHBFSystem(GPUHBFHybridSystem):
    """Agentic event loop for :class:`SSDStagedGPUHBFNode`."""

    def __init__(
            self, *, repo_root: Path,
            gpu_hardware: Optional[P4D4GPUHardware] = None,
            hbf_hardware: Optional[HBFServerHardware] = None,
            hbf_layout: str = "tp4",
            p_capacity_bytes_per_rank: Optional[int] = None,
            d_capacity_bytes_per_rank: Optional[int] = None,
            cpu_capacity_bytes: Optional[int] = None,
            ssd_capacity_bytes: Optional[int] = None,
            max_num_batched_tokens: int = 8_192,
            max_num_seqs: int = 128,
            p_max_num_seqs: Optional[int] = None,
            d_max_num_seqs: Optional[int] = None,
            max_prefill_chunk_tokens: int = 4_096,
            hbf_mixed_batch_latency_limit_ns: Optional[int] = None,
            band: str = "central",
            restore_execution_mode: str = RESTORE_EXECUTION_BULK,
            validate_every_event: bool = True,
            promotion_policy: Optional[
                str | SSDPromotionPolicy] = None,
            migration_policy: Optional[
                str | SSDPromotionPolicy] = None) -> None:
        self.node = SSDStagedGPUHBFNode(
            repo_root=repo_root,
            gpu_hardware=(
                P4D4GPUHardware()
                if gpu_hardware is None else gpu_hardware
            ),
            hbf_hardware=(
                HBFServerHardware()
                if hbf_hardware is None else hbf_hardware
            ),
            hbf_layout=hbf_layout,
            p_capacity_bytes_per_rank=p_capacity_bytes_per_rank,
            d_capacity_bytes_per_rank=d_capacity_bytes_per_rank,
            cpu_capacity_bytes=cpu_capacity_bytes,
            ssd_capacity_bytes=ssd_capacity_bytes,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            p_max_num_seqs=p_max_num_seqs,
            d_max_num_seqs=d_max_num_seqs,
            max_prefill_chunk_tokens=max_prefill_chunk_tokens,
            hbf_mixed_batch_latency_limit_ns=(
                hbf_mixed_batch_latency_limit_ns),
            band=band,
            restore_execution_mode=restore_execution_mode,
            validate_every_event=validate_every_event,
            promotion_policy=promotion_policy,
            migration_policy=migration_policy,
        )
        self.validate_every_event = validate_every_event
        self.metrics = HybridSystemMetrics()
        self.current_ns = 0
        self.call_specs = ()
        self._spec_by_request = {}
        self._request_by_identity = {}
        self._successor_by_request = {}
        self._release_heap = []
        self._queued_release_ids = set()
        self._released_ids = set()
        self._completed_ids = set()
        self._runtime_calls = {}
        self._completed_snapshots: dict[int, CompletedRequest] = {}
        self._completion_order = []
        self._completed_session_order = []
        self._offered_session_order = []
        self._loaded = False
        self._running = False
        self._finished = False

    def load(
            self,
            scheduled_sessions: Iterable[ScheduledSession]) -> None:
        values = tuple(scheduled_sessions)
        gap_types = {
            (scheduled.session.session_id, call.call_index):
            call.inter_turn_gap_type
            for scheduled in values
            for call in scheduled.session.calls
        }
        super().load(values)
        for spec in self.call_specs:
            self.node.set_gap_type(
                spec.request_id,
                gap_types[(spec.session_id, spec.call_index)],
            )

    def report(self) -> Mapping[str, Any]:
        spec_rows = [asdict(spec) for spec in self.call_specs]
        completed_identities = [
            self._spec_by_request[
                request_id].completion_identity
            for request_id in self._completion_order
        ]
        result: dict[str, Any] = {
            "mode": "ssd_staged_gpu_hbf_agentic_system",
            "architecture": {
                "gpu_server_count": 1,
                "gpu_server": "one_4p4d_h100_server",
                "gpu_restore_execution_mode": (
                    self.node.restore_execution_mode),
                "local_ssd_checkpoint": True,
                "hbf_server_count": 1,
                "hbf_server": "one_8card_full_model_hbf_server",
                "hbf_layout": self.node.hbf_layout.key,
                "hbf_mixed_batch_latency_limit_ns": (
                    self.node.hbf_pool.mixed_batch_latency_limit_ns),
                "execution_backend": "analytical_calendar",
                "shared_resource_calendar": True,
            },
            "policy": {
                "first_turn": "gpu",
                "gpu_turn_checkpoint": "d_to_cpu_to_local_ssd",
                "promotion_source": "local_ssd_via_cpu_rdma",
                "resume_at_threshold": "resume_wins",
                "promotion": asdict(
                    self.node.promotion_policy),
                "gpu_restore_execution_mode": (
                    self.node.restore_execution_mode),
            },
            "metrics": asdict(self.metrics),
            "current_ns": self.current_ns,
            "finished": self._finished,
            "call_specs": spec_rows,
            "call_specs_identity_sha256": stable_json_sha256(
                spec_rows),
            "completion_order": completed_identities,
            "completed_requests": [
                asdict(request)
                for request in self.completed_requests
            ],
            "node": self.node.report(),
        }
        if self._finished:
            call_drain = asdict(full_drain_hashes(
                [
                    spec.completion_identity
                    for spec in self.call_specs
                ],
                completed_identities,
            ))
            session_drain = asdict(full_drain_hashes(
                self._offered_session_order,
                self._completed_session_order,
            ))
            result["full_drain"] = call_drain
            result["call_full_drain"] = call_drain
            result["session_full_drain"] = session_drain
        return result


SSDStagedGPUHBFRunner = SSDStagedGPUHBFSystem


__all__ = [
    "SSDPromotionIntent",
    "SSDPromotionPolicy",
    "SSDStagedGPUHBFNode",
    "SSDStagedGPUHBFRunner",
    "SSDStagedGPUHBFSystem",
    "SSDStagedHybridMetrics",
    "SUPPORTED_SSD_PROMOTION_MODES",
    "SUPPORTED_SSD_STAGED_HBF_LAYOUTS",
]

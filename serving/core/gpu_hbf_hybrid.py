"""Online runner for one P4D4 GPU server plus one eight-card HBF server.

The GPU and HBF servers have independent compute/resource calendars.  The
only cross-server resource is the RDMA link used to persist completed GPU KV
at a turn boundary.  A first turn always executes on the GPU.  A resume that
arrives before that migration publishes executes on the GPU, while a resume
whose version is HBF-ready executes entirely on the HBF server.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from enum import Enum
import heapq
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .comparison_cutoff import ResumableCutoffEventLoopMixin
from .gpu_pd_hbm import AtomicPDHBM, PDHBMAdmission
from .gpu_pd_latency import P4D4GPUHardware
from .gpu_pd_pool import P4D4ServingPool, PDServingRequest
from .gpu_pd_tier_resources import TierNodeResources, TierTransferStage
from .hbf_comparison_metrics import CompletedRequest, RequestKey
from .hbf_comparison_workload import (
    ScheduledSession,
    full_drain_hashes,
    stable_json_sha256,
)
from .hbf_full_model_latency import HBFParallelLayout, HBFServerHardware
from .hbf_full_model_lifecycle import (
    FullModelHBFLifecycle,
    HBFLifecycleExternalDispatch,
    PerGroupCapacityLedger,
    PlacementState,
    ResourceCalendar,
    ResumeExecution,
    ResumeRoute,
)
from .hbf_full_model_pool import (
    FullModelHBFServingPool,
    HBFExternalDispatch,
    HBFServingRequest,
    derive_lpddr_workspace_bytes,
)


SUPPORTED_HBF_LAYOUTS = frozenset({
    "dp8",
    "tp4",
    "tp8",
    "tp8_context",
})
SUPPORTED_HBF_EXECUTION_BACKENDS = frozenset({
    "analytical_calendar",
    "external_astra",
})


class HybridDeadlockError(RuntimeError):
    """Raised when unfinished hybrid work has no future event."""


class HybridCallState(str, Enum):
    PENDING = "pending"
    GPU_PREPARING = "gpu_preparing"
    GPU_EXECUTING = "gpu_executing"
    HBF_EXECUTING = "hbf_executing"
    USER_COMPLETE = "user_complete"
    INTERNAL_COMPLETE = "internal_complete"


class HybridExecution(str, Enum):
    GPU_FIRST_TURN = "gpu_first_turn"
    GPU_MIGRATION_INFLIGHT = "gpu_migration_inflight_fallback"
    GPU_CAPACITY_FALLBACK = "gpu_hbf_capacity_fallback"
    GPU_RECOMPUTE = "gpu_recompute"
    GPU_OWNED = "gpu_owned"
    HBF_READY = "hbf_ready_resume"


@dataclass
class HybridCall:
    request_id: int
    session_id: str
    call_index: int
    release_ns: int
    input_tokens: int
    output_tokens: int
    prefix_reuse_tokens: int
    has_successor: bool
    state: HybridCallState = HybridCallState.PENDING
    execution: Optional[HybridExecution] = None
    route_reason: Optional[str] = None
    migration_inflight_at_route: bool = False
    operational_reuse_tokens: Optional[int] = None
    gpu_hit_tokens: Optional[int] = None
    admission_id: Optional[int] = None
    prepare_start_ns: Optional[int] = None
    prepare_completion_ns: Optional[int] = None
    gpu_request: Optional[PDServingRequest] = None
    hbf_request: Optional[HBFServingRequest] = None
    user_completion_ns: Optional[int] = None
    internal_completion_ns: Optional[int] = None

    def validate(self) -> None:
        for name in (
            "request_id",
            "call_index",
            "release_ns",
            "input_tokens",
            "output_tokens",
            "prefix_reuse_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.request_id < 0 or self.call_index < 0 or self.release_ns < 0:
            raise ValueError(
                "request_id/call_index/release_ns must be non-negative")
        if self.input_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("input/output tokens must be positive")
        if not 0 <= self.prefix_reuse_tokens <= self.input_tokens:
            raise ValueError(
                "prefix_reuse_tokens must be in 0..input_tokens")
        if self.input_tokens + self.output_tokens - 1 > 1_010_000:
            raise ValueError(
                "request output would exceed 1,010,000-token contract")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(self.has_successor, bool):
            raise ValueError("has_successor must be a boolean")
        if self.call_index == 0 and self.prefix_reuse_tokens:
            raise ValueError("first call cannot reuse an earlier prefix")
        if (
            self.state != HybridCallState.PENDING
            or self.execution is not None
            or self.route_reason is not None
            or self.migration_inflight_at_route
            or self.operational_reuse_tokens is not None
            or self.gpu_hit_tokens is not None
            or self.admission_id is not None
            or self.prepare_start_ns is not None
            or self.prepare_completion_ns is not None
            or self.gpu_request is not None
            or self.hbf_request is not None
            or self.user_completion_ns is not None
            or self.internal_completion_ns is not None
        ):
            raise ValueError("submitted hybrid call must be pristine")

    @property
    def serving_request(self) -> PDServingRequest | HBFServingRequest | None:
        if self.gpu_request is not None:
            return self.gpu_request
        return self.hbf_request

    @property
    def ttft_ns(self) -> Optional[int]:
        request = self.serving_request
        return None if request is None else request.ttft_ns

    @property
    def tpot_ns(self) -> Optional[float]:
        request = self.serving_request
        return None if request is None else request.tpot_ns


@dataclass
class HybridSession:
    session_id: str
    last_internal_call_index: int = -1
    ended: bool = False


@dataclass(frozen=True)
class HybridGPUPrepareJob:
    job_id: int
    request_id: int
    hit_tokens: int
    stage: TierTransferStage
    start_ns: int
    completion_ns: int


@dataclass(frozen=True)
class HybridHBFExternalDispatch:
    """One HBF-side ASTRA job owned by the hybrid node."""

    owner: str
    dispatch: HBFLifecycleExternalDispatch | HBFExternalDispatch

    def __post_init__(self) -> None:
        if self.owner not in {"lifecycle", "pool"}:
            raise ValueError(
                "external HBF dispatch owner must be lifecycle or pool")
        if (
            self.owner == "lifecycle"
            and not isinstance(
                self.dispatch, HBFLifecycleExternalDispatch)
        ):
            raise TypeError(
                "lifecycle owner requires a lifecycle dispatch")
        if (
            self.owner == "pool"
            and not isinstance(self.dispatch, HBFExternalDispatch)
        ):
            raise TypeError("pool owner requires a pool dispatch")

    @property
    def job_id(self) -> str:
        return self.dispatch.job_id

    @property
    def arrival_ns(self) -> int:
        return self.dispatch.arrival_ns

    @property
    def stage_count(self) -> int:
        return self.dispatch.stage_count

    @property
    def projection(self) -> Any:
        return self.dispatch.projection

    def controller_arguments(
            self,
    ) -> tuple[str, int, tuple[dict[str, Any], ...]]:
        return self.dispatch.controller_arguments()


@dataclass
class HybridNodeMetrics:
    submitted_calls: int = 0
    routed_calls: int = 0
    gpu_calls: int = 0
    hbf_calls: int = 0
    gpu_first_turn_calls: int = 0
    gpu_migration_inflight_calls: int = 0
    gpu_capacity_fallback_calls: int = 0
    gpu_recompute_calls: int = 0
    gpu_owned_calls: int = 0
    operational_prefix_cap_calls: int = 0
    operational_prefix_cap_tokens: int = 0
    gpu_hbm_capacity_deferrals: int = 0
    gpu_d_to_p_jobs: int = 0
    gpu_d_to_p_tokens: int = 0
    gpu_d_to_p_aggregate_bytes: int = 0
    user_completed_calls: int = 0
    internal_completed_calls: int = 0
    migration_hbm_releases: int = 0
    max_pending_calls: int = 0


class GPUHBFHybridNode:
    """One finite-HBM P4D4 server and one independent full-model HBF server."""

    def __init__(
            self, *, repo_root: Path,
            gpu_hardware: P4D4GPUHardware,
            hbf_hardware: HBFServerHardware,
            hbf_layout: str | HBFParallelLayout = "tp4",
            gpu_node_id: int = 0,
            max_num_batched_tokens: int = 8_192,
            max_num_seqs: int = 128,
            p_max_num_seqs: Optional[int] = None,
            d_max_num_seqs: Optional[int] = None,
            max_prefill_chunk_tokens: int = 4_096,
            band: str = "central",
            validate_every_event: bool = True,
            hbf_execution_backend: str = (
                "analytical_calendar"),
            hbf_server_id: Optional[int] = None,
            hbf_astra_chunk_bytes: int = 64 * 1024 ** 2) -> None:
        if not isinstance(validate_every_event, bool):
            raise ValueError("validate_every_event must be a boolean")
        if (
            not isinstance(hbf_execution_backend, str)
            or hbf_execution_backend
            not in SUPPORTED_HBF_EXECUTION_BACKENDS
        ):
            raise ValueError(
                "hbf_execution_backend must be one of "
                f"{sorted(SUPPORTED_HBF_EXECUTION_BACKENDS)}")
        resolved_hbf_server_id = (
            gpu_node_id
            if hbf_server_id is None else hbf_server_id
        )
        if (
            not isinstance(resolved_hbf_server_id, int)
            or isinstance(resolved_hbf_server_id, bool)
            or resolved_hbf_server_id < 0
        ):
            raise ValueError(
                "hbf_server_id must be a non-negative integer or None")
        if (
            not isinstance(hbf_astra_chunk_bytes, int)
            or isinstance(hbf_astra_chunk_bytes, bool)
            or hbf_astra_chunk_bytes <= 0
        ):
            raise ValueError(
                "hbf_astra_chunk_bytes must be a positive integer")
        layout = (
            HBFParallelLayout.for_key(hbf_layout)
            if isinstance(hbf_layout, str) else hbf_layout
        )
        if not isinstance(layout, HBFParallelLayout):
            raise TypeError(
                "hbf_layout must be a layout key or HBFParallelLayout")
        gpu_hardware.validate()
        hbf_hardware.validate()
        layout.validate(hbf_hardware.card_count)
        if layout.key not in SUPPORTED_HBF_LAYOUTS:
            raise ValueError(
                f"unsupported HBF layout {layout.key!r}")

        self.repo_root = Path(repo_root)
        self.gpu_hardware = gpu_hardware
        self.hbf_hardware = hbf_hardware
        self.hbf_layout = layout
        self.gpu_node_id = gpu_node_id
        self.hbf_execution_backend = hbf_execution_backend
        self.hbf_server_id = resolved_hbf_server_id
        self.hbf_astra_chunk_bytes = hbf_astra_chunk_bytes
        self.validate_every_event = validate_every_event
        self.retain_detailed_history = validate_every_event

        # The two server calendars are deliberately independent.  Migration
        # source PCIe, RDMA, destination PCIe, and HBF media are accounted in
        # the HBF-side calendar; no foreground request reserves RDMA.
        self.gpu_calendar = ResourceCalendar(
            retain_reservations=validate_every_event)
        self.hbf_calendar = (
            ResourceCalendar(
                retain_reservations=validate_every_event)
            if hbf_execution_backend == "analytical_calendar"
            else None
        )
        self.gpu_resources = TierNodeResources(
            hardware=gpu_hardware,
            node_id=gpu_node_id,
        )
        self.gpu_hbm = AtomicPDHBM(
            hardware=gpu_hardware,
            node_id=gpu_node_id,
        )
        self.gpu_pool = P4D4ServingPool(
            repo_root=self.repo_root,
            hardware=gpu_hardware,
            node_id=gpu_node_id,
            resource_calendar=self.gpu_calendar,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            p_max_num_seqs=p_max_num_seqs,
            d_max_num_seqs=d_max_num_seqs,
            max_prefill_chunk_tokens=max_prefill_chunk_tokens,
            band=band,
            validate_every_event=validate_every_event,
            retain_detailed_history=self.retain_detailed_history,
        )

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
        self.hbf_lifecycle = FullModelHBFLifecycle(
            hardware=hbf_hardware,
            layout=layout,
            resource_calendar=self.hbf_calendar,
            lpddr_ledger=hbf_lpddr_ledger,
            gpu_source_root_bandwidth_gbps=(
                min(
                    gpu_hardware.pcie_root_bandwidth_gbps,
                    gpu_hardware.decode_gpu_count
                    * gpu_hardware.pcie_bandwidth_gbps_per_gpu,
                )),
            validate_every_event=validate_every_event,
            execution_backend=hbf_execution_backend,
            server_id=self.hbf_server_id,
            astra_chunk_bytes=hbf_astra_chunk_bytes,
        )
        self.hbf_pool = FullModelHBFServingPool(
            repo_root=self.repo_root,
            hardware=hbf_hardware,
            layout=layout,
            resource_calendar=self.hbf_calendar,
            lpddr_ledger=hbf_lpddr_ledger,
            placement_resolver=self.hbf_lifecycle.placement_snapshot,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_prefill_chunk_tokens=max_prefill_chunk_tokens,
            band=band,
            validate_every_event=validate_every_event,
            retain_detailed_history=self.retain_detailed_history,
            execution_backend=hbf_execution_backend,
            server_id=self.hbf_server_id,
        )

        self.calls: dict[int, HybridCall] = {}
        self.sessions: dict[str, HybridSession] = {}
        self.metrics = HybridNodeMetrics()
        self.prepare_history: list[HybridGPUPrepareJob] = []
        self._pending_call_ids: deque[int] = deque()
        self._gpu_ready_call_ids: deque[int] = deque()
        self._prepare_jobs: dict[int, HybridGPUPrepareJob] = {}
        self._prepare_completion_heap: list[tuple[int, int, int]] = []
        self._admission_by_request: dict[int, PDHBMAdmission] = {}
        self._user_completed_ids: deque[int] = deque()
        self._gpu_user_seen: set[int] = set()
        self._gpu_handoff_seen: set[int] = set()
        self._live_hbf_request_ids: set[int] = set()
        self._external_hbf_pending: dict[
            str, HybridHBFExternalDispatch] = {}
        self._external_hbf_completed_job_ids: set[str] = set()
        self._last_submitted_call_index: dict[str, int] = {}
        self._last_submitted_request_id: dict[str, int] = {}
        self._next_prepare_job_id = 1
        self.current_ns = 0

    def _validate_gpu_capacity_contract(self, call: HybridCall) -> None:
        p_bytes = self.gpu_hardware.kv_capacity_bytes_per_rank(
            call.input_tokens)
        needs_d = call.output_tokens > 1 or call.has_successor
        final_tokens = call.input_tokens + call.output_tokens - 1
        d_bytes = (
            self.gpu_hardware.kv_capacity_bytes_per_rank(final_tokens)
            if needs_d else 0
        )
        if (
            p_bytes > self.gpu_hbm.p_capacity_bytes_per_rank
            or d_bytes > self.gpu_hbm.d_capacity_bytes_per_rank
        ):
            raise RuntimeError(
                "hybrid call is individually infeasible for GPU HBM: "
                f"request_id={call.request_id}, "
                f"p={p_bytes}/{self.gpu_hbm.p_capacity_bytes_per_rank}, "
                f"d={d_bytes}/{self.gpu_hbm.d_capacity_bytes_per_rank}")

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
            self._validate_gpu_capacity_contract(call)
            if call.release_ns != now_ns:
                raise ValueError(
                    "hybrid calls must be submitted at logical release")
            if call.request_id in self.calls or call.request_id in seen_ids:
                raise ValueError(
                    f"duplicate request_id={call.request_id}")
            prior_index = proposed_last.get(call.session_id, -1)
            if call.call_index != prior_index + 1:
                raise ValueError(
                    "session calls must be submitted in contiguous order: "
                    f"session={call.session_id!r}, "
                    f"got={call.call_index}, expected={prior_index + 1}")
            predecessor = proposed_predecessors.get(call.session_id)
            if predecessor is not None:
                if not predecessor.has_successor:
                    raise ValueError(
                        "cannot submit after a terminal predecessor")
                if predecessor.user_completion_ns is None:
                    raise ValueError(
                        "successor cannot be submitted before its "
                        "predecessor is user-complete")
                if call.release_ns < predecessor.user_completion_ns:
                    raise ValueError(
                        "successor release cannot precede predecessor "
                        "user completion")
            session = self.sessions.get(call.session_id)
            if session is not None and session.ended:
                raise ValueError(
                    f"session {call.session_id!r} already ended")
            proposed_last[call.session_id] = call.call_index
            proposed_predecessors[call.session_id] = call
            seen_ids.add(call.request_id)

        self.advance(now_ns, defer_schedule=True)
        for call in values:
            self.calls[call.request_id] = call
            self.sessions.setdefault(
                call.session_id,
                HybridSession(session_id=call.session_id),
            )
            self._last_submitted_call_index[call.session_id] = (
                call.call_index)
            self._last_submitted_request_id[call.session_id] = (
                call.request_id)
            self._pending_call_ids.append(call.request_id)
            self.metrics.submitted_calls += 1
        self.metrics.max_pending_calls = max(
            self.metrics.max_pending_calls,
            len(self._pending_call_ids),
        )
        self.flush_scheduling(now_ns)

    def submit(self, call: HybridCall, *, now_ns: int) -> None:
        self.submit_many((call,), now_ns=now_ns)

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
        if route.reason == "hbf_capacity_unavailable_gpu_retained":
            return HybridExecution.GPU_CAPACITY_FALLBACK
        return HybridExecution.GPU_OWNED

    def _resolve_route(self, call: HybridCall) -> ResumeRoute:
        if call.execution is not None:
            raise RuntimeError("hybrid call route was resolved twice")
        if call.call_index == 0:
            self.hbf_lifecycle.register_session(
                call.session_id, now_ns=self.current_ns)
        placement = self.hbf_lifecycle.sessions[call.session_id]
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
        execution = self._execution_for_route(call, route)
        call.execution = execution
        call.route_reason = route.reason
        call.migration_inflight_at_route = route.migration_inflight
        call.operational_reuse_tokens = operational_reuse
        if operational_reuse < call.prefix_reuse_tokens:
            self.metrics.operational_prefix_cap_calls += 1
            self.metrics.operational_prefix_cap_tokens += (
                call.prefix_reuse_tokens - operational_reuse)
        self.metrics.routed_calls += 1
        if execution == HybridExecution.HBF_READY:
            self.metrics.hbf_calls += 1
        else:
            self.metrics.gpu_calls += 1
        if execution == HybridExecution.GPU_FIRST_TURN:
            self.metrics.gpu_first_turn_calls += 1
        elif execution == HybridExecution.GPU_MIGRATION_INFLIGHT:
            self.metrics.gpu_migration_inflight_calls += 1
        elif execution == HybridExecution.GPU_CAPACITY_FALLBACK:
            self.metrics.gpu_capacity_fallback_calls += 1
        elif execution == HybridExecution.GPU_RECOMPUTE:
            self.metrics.gpu_recompute_calls += 1
        elif execution == HybridExecution.GPU_OWNED:
            self.metrics.gpu_owned_calls += 1
        return route

    def _make_hbf_request(
            self, call: HybridCall,
            route: ResumeRoute) -> HBFServingRequest:
        if route.group_id is None:
            raise RuntimeError("HBF route lacks a replica group")
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
        call.state = HybridCallState.HBF_EXECUTING
        self._live_hbf_request_ids.add(call.request_id)
        return request

    def _make_gpu_request(self, call: HybridCall) -> PDServingRequest:
        if call.gpu_hit_tokens is None:
            raise RuntimeError("GPU call lacks a resolved prefix hit")
        request = PDServingRequest(
            request_id=call.request_id,
            session_id=call.session_id,
            arrival_ns=call.release_ns,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            p_prefix_tokens=call.gpu_hit_tokens,
            d_prefix_tokens=call.gpu_hit_tokens,
            has_successor=call.has_successor,
        )
        call.gpu_request = request
        call.state = HybridCallState.GPU_EXECUTING
        return request

    def _queue_gpu_prepare(
            self, call: HybridCall, *,
            hit_tokens: int, now_ns: int) -> None:
        stage = self.gpu_resources.peer_stage(
            hit_tokens,
            direction="d_to_p",
            block_rounded=False,
        )
        job_id = self._next_prepare_job_id
        self._next_prepare_job_id += 1
        start_ns, completion_ns = stage.reserve(
            self.gpu_calendar,
            ready_ns=now_ns,
            job_id=job_id,
            namespace=f"hybrid-gpu-prepare-{self.gpu_node_id}",
        )
        job = HybridGPUPrepareJob(
            job_id=job_id,
            request_id=call.request_id,
            hit_tokens=hit_tokens,
            stage=stage,
            start_ns=start_ns,
            completion_ns=completion_ns,
        )
        call.state = HybridCallState.GPU_PREPARING
        call.prepare_start_ns = start_ns
        call.prepare_completion_ns = completion_ns
        if self.retain_detailed_history:
            self.prepare_history.append(job)
        self._prepare_jobs[job_id] = job
        heapq.heappush(
            self._prepare_completion_heap,
            (completion_ns, call.request_id, job_id),
        )
        self.metrics.gpu_d_to_p_jobs += 1
        self.metrics.gpu_d_to_p_tokens += hit_tokens
        self.metrics.gpu_d_to_p_aggregate_bytes += stage.aggregate_bytes

    def _admit_pending(self, now_ns: int) -> None:
        deferred: deque[int] = deque()
        hbf_requests: list[HBFServingRequest] = []
        gpu_requests: list[PDServingRequest] = []
        while self._pending_call_ids:
            request_id = self._pending_call_ids.popleft()
            call = self.calls[request_id]
            session = self.sessions[call.session_id]
            if call.call_index != session.last_internal_call_index + 1:
                deferred.append(request_id)
                continue

            route: Optional[ResumeRoute] = None
            if call.execution is None:
                route = self._resolve_route(call)
            if call.execution == HybridExecution.HBF_READY:
                if route is None:
                    raise RuntimeError(
                        "resolved HBF call remained pending")
                hbf_requests.append(
                    self._make_hbf_request(call, route))
                continue

            if call.admission_id is not None:
                raise RuntimeError(
                    "admitted GPU call remained in the pending queue")
            needs_d = call.output_tokens > 1 or call.has_successor
            admission = self.gpu_hbm.try_admit(
                session_id=call.session_id,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                needs_d=needs_d,
            )
            if admission is None:
                self.metrics.gpu_hbm_capacity_deferrals += 1
                deferred.append(request_id)
                continue
            self._admission_by_request[request_id] = admission
            call.admission_id = admission.admission_id
            hit_tokens = (
                0
                if call.execution == HybridExecution.GPU_RECOMPUTE
                else min(
                    call.operational_reuse_tokens,
                    call.input_tokens - 1,
                )
            )
            if hit_tokens is None:
                raise RuntimeError(
                    "GPU call lacks operational prefix reuse")
            call.gpu_hit_tokens = hit_tokens
            if hit_tokens:
                self._queue_gpu_prepare(
                    call,
                    hit_tokens=hit_tokens,
                    now_ns=now_ns,
                )
            else:
                self.gpu_hbm.release_d_source(admission)
                gpu_requests.append(self._make_gpu_request(call))
        self._pending_call_ids = deferred

        if hbf_requests:
            hbf_requests.sort(key=lambda request: request.request_id)
            self.hbf_pool.submit_many(
                hbf_requests, now_ns=now_ns)
        if gpu_requests:
            gpu_requests.sort(key=lambda request: request.request_id)
            self.gpu_pool.submit_many(
                gpu_requests, now_ns=now_ns)

    def _submit_gpu_ready(self, now_ns: int) -> bool:
        if not self._gpu_ready_call_ids:
            return False
        requests = []
        while self._gpu_ready_call_ids:
            call = self.calls[self._gpu_ready_call_ids.popleft()]
            requests.append(self._make_gpu_request(call))
        requests.sort(key=lambda request: request.request_id)
        self.gpu_pool.submit_many(requests, now_ns=now_ns)
        return True

    def _process_prepare_completions(self, now_ns: int) -> None:
        while (
            self._prepare_completion_heap
            and self._prepare_completion_heap[0][0] == now_ns
        ):
            _, request_id, job_id = heapq.heappop(
                self._prepare_completion_heap)
            job = self._prepare_jobs.pop(job_id)
            call = self.calls[request_id]
            if (
                call.state != HybridCallState.GPU_PREPARING
                or call.prepare_completion_ns != now_ns
            ):
                raise RuntimeError("stale hybrid GPU prepare completion")
            admission = self._admission_by_request[request_id]
            self.gpu_hbm.release_d_source(admission)
            self._gpu_ready_call_ids.append(request_id)

    def _mark_user_complete(
            self, call: HybridCall, *, completion_ns: int) -> None:
        if call.user_completion_ns is not None:
            raise RuntimeError("duplicate hybrid user completion")
        call.user_completion_ns = completion_ns
        call.state = HybridCallState.USER_COMPLETE
        self._user_completed_ids.append(call.request_id)
        self.metrics.user_completed_calls += 1

    def _finish_gpu_internal(
            self, call: HybridCall, *, now_ns: int) -> None:
        if call.internal_completion_ns is not None:
            return
        if call.user_completion_ns is None:
            raise RuntimeError(
                "GPU internal completion precedes user completion")
        admission = self._admission_by_request.pop(call.request_id)
        self.gpu_hbm.finish(
            admission,
            has_successor=call.has_successor,
        )
        if call.gpu_request is None:
            raise RuntimeError("GPU call lacks a serving request")
        self.hbf_lifecycle.complete_gpu_turn(
            call.session_id,
            now_ns=now_ns,
            total_tokens=(
                call.gpu_request.final_materialized_kv_tokens),
            has_successor=call.has_successor,
        )
        session = self.sessions[call.session_id]
        session.last_internal_call_index = call.call_index
        session.ended = not call.has_successor
        call.internal_completion_ns = now_ns
        call.state = HybridCallState.INTERNAL_COMPLETE
        self.metrics.internal_completed_calls += 1

    def _consume_gpu_notifications(self, now_ns: int) -> None:
        for request in self.gpu_pool.pop_handoff_completed():
            request_id = request.request_id
            if request_id in self._gpu_handoff_seen:
                raise RuntimeError("duplicate hybrid GPU handoff")
            self._gpu_handoff_seen.add(request_id)
            admission = self._admission_by_request[request_id]
            self.gpu_hbm.release_p(admission)
            call = self.calls[request_id]
            if request_id in self._gpu_user_seen:
                self._finish_gpu_internal(call, now_ns=now_ns)
        for request in self.gpu_pool.pop_completed():
            request_id = request.request_id
            if request_id in self._gpu_user_seen:
                raise RuntimeError("duplicate hybrid GPU completion")
            self._gpu_user_seen.add(request_id)
            call = self.calls[request_id]
            if request.completion_ns != now_ns:
                raise RuntimeError(
                    "GPU completion timestamp mismatch")
            self._mark_user_complete(
                call, completion_ns=now_ns)
            if (
                request.handoff_done
                or not (
                    request.output_tokens == 1
                    and request.has_successor
                )
            ):
                self._finish_gpu_internal(call, now_ns=now_ns)

    def _consume_hbf_notifications(self, now_ns: int) -> None:
        for request in self.hbf_pool.pop_completed():
            call = self.calls[request.request_id]
            if request.request_id not in self._live_hbf_request_ids:
                raise RuntimeError(
                    "completed HBF request lacks live-set ownership")
            self._live_hbf_request_ids.remove(request.request_id)
            if request.completion_ns != now_ns:
                raise RuntimeError(
                    "HBF completion timestamp mismatch")
            self._mark_user_complete(
                call, completion_ns=now_ns)
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
            call.internal_completion_ns = now_ns
            call.state = HybridCallState.INTERNAL_COMPLETE
            self.metrics.internal_completed_calls += 1

    def _reconcile_migrated_gpu_hbm(self) -> None:
        for session_id, placement in self.hbf_lifecycle.sessions.items():
            if (
                placement.gpu_retained_bytes == 0
                and self.gpu_hbm.d_bytes(session_id)
                and self.gpu_hbm.active_admission(session_id) is None
            ):
                self.gpu_hbm.release_idle_session(session_id)
                self.metrics.migration_hbm_releases += 1

    def _refresh_live_hbf_placements(self) -> None:
        """Publish completed appends before the HBF pool checks its ledger."""

        for request_id in tuple(self._live_hbf_request_ids):
            call = self.calls[request_id]
            request = call.hbf_request
            if (
                request is None
                or call.state != HybridCallState.HBF_EXECUTING
            ):
                continue
            hbf_tokens, lpddr_tokens, group_id = (
                self.hbf_lifecycle.placement_snapshot(
                    call.session_id)
            )
            if group_id != request.group_id:
                raise RuntimeError(
                    "active HBF request changed replica group")
            if hbf_tokens + lpddr_tokens != request.published_tokens:
                raise RuntimeError(
                    "append publication changed published request tokens")
            request.hbf_prefix_tokens = hbf_tokens
            request.lpddr_prefix_tokens = lpddr_tokens

    def _require_external_hbf(self, operation: str) -> None:
        if self.hbf_execution_backend != "external_astra":
            raise RuntimeError(
                f"{operation} requires "
                "hbf_execution_backend='external_astra'")

    @staticmethod
    def _callback_nonnegative_int(name: str, value: object) -> int:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    def drain_external_dispatches(
            self) -> tuple[HybridHBFExternalDispatch, ...]:
        """Drain lifecycle and foreground HBF jobs through one boundary."""

        self._require_external_hbf("drain_external_dispatches")
        # Preflight the combined namespace before either component marks a
        # dispatch issued.  The schemas also reserve disjoint structural
        # prefixes (hbf-migration/hbf-append versus hbf-model), so a failure
        # cannot strand one component in an issued-but-unowned state.
        lifecycle_ready = tuple(
            self.hbf_lifecycle._external_outbox)
        pool_ready = tuple(self.hbf_pool._external_outbox)
        prospective = [
            *(
                HybridHBFExternalDispatch(
                    owner="lifecycle", dispatch=dispatch)
                for dispatch in lifecycle_ready
            ),
            *(
                HybridHBFExternalDispatch(
                    owner="pool", dispatch=dispatch)
                for dispatch in pool_ready
            ),
        ]
        lifecycle_ids = {
            dispatch.job_id for dispatch in prospective
            if dispatch.owner == "lifecycle"
        }
        pool_ids = {
            dispatch.job_id for dispatch in prospective
            if dispatch.owner == "pool"
        }
        candidate_ids = [
            dispatch.job_id for dispatch in prospective
        ]
        if (
            lifecycle_ids & pool_ids
            or len(candidate_ids) != len(set(candidate_ids))
            or set(candidate_ids) & set(
                self._external_hbf_pending)
            or set(candidate_ids) & (
                self._external_hbf_completed_job_ids)
        ):
            raise RuntimeError(
                "hybrid HBF external ASTRA job-id namespace "
                "collision")
        if any(
                not job_id.startswith(
                    ("hbf-migration.", "hbf-append."))
                for job_id in lifecycle_ids
        ) or any(
                not job_id.startswith("hbf-model.")
                for job_id in pool_ids
        ):
            raise RuntimeError(
                "hybrid HBF external ASTRA job id violates its "
                "structural owner prefix")

        lifecycle_drained = (
            self.hbf_lifecycle.drain_external_dispatches())
        pool_drained = self.hbf_pool.drain_external_dispatches()
        if (
            len(lifecycle_drained) != len(lifecycle_ready)
            or any(
                actual is not expected
                for actual, expected in zip(
                    lifecycle_drained, lifecycle_ready)
            )
            or len(pool_drained) != len(pool_ready)
            or any(
                actual is not expected
                for actual, expected in zip(
                    pool_drained, pool_ready)
            )
        ):
            raise RuntimeError(
                "hybrid HBF component outbox changed during drain")

        for dispatch in prospective:
            if (
                dispatch.job_id in self._external_hbf_pending
                or dispatch.job_id
                in self._external_hbf_completed_job_ids
            ):
                raise RuntimeError(
                    "duplicate hybrid HBF external ASTRA job id "
                    f"{dispatch.job_id!r}")
            self._external_hbf_pending[
                dispatch.job_id] = dispatch
        if self.validate_every_event:
            self.assert_invariants()
        return tuple(prospective)

    def complete_external_dispatch(
            self, job_id: str, arrival_ns: int, completion_ns: int,
            stage_count: int, *,
            defer_schedule: bool = False,
    ) -> Any:
        """Route one strict ASTRA callback to its exact HBF owner."""

        self._require_external_hbf("complete_external_dispatch")
        if not isinstance(defer_schedule, bool):
            raise ValueError("defer_schedule must be a boolean")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        arrival = self._callback_nonnegative_int(
            "arrival_ns", arrival_ns)
        completion = self._callback_nonnegative_int(
            "completion_ns", completion_ns)
        stages = self._callback_nonnegative_int(
            "stage_count", stage_count)
        if job_id in self._external_hbf_completed_job_ids:
            raise RuntimeError(
                "duplicate hybrid HBF external ASTRA completion for "
                f"{job_id!r}")
        dispatch = self._external_hbf_pending.get(job_id)
        if dispatch is None:
            raise RuntimeError(
                "unknown hybrid HBF external ASTRA completion job "
                f"{job_id!r}")
        if arrival != dispatch.arrival_ns:
            raise RuntimeError(
                "hybrid HBF external ASTRA completion arrival mismatch: "
                f"job={job_id!r}, expected={dispatch.arrival_ns}, "
                f"actual={arrival}")
        if stages != dispatch.stage_count:
            raise RuntimeError(
                "hybrid HBF external ASTRA completion stage-count "
                f"mismatch: job={job_id!r}, "
                f"expected={dispatch.stage_count}, actual={stages}")
        minimum_completion = (
            arrival
            + dispatch.projection
            .dependency_critical_path_ns()
        )
        if completion < minimum_completion:
            raise RuntimeError(
                "hybrid HBF external ASTRA completion precedes the "
                "dependency critical path bound: "
                f"job={job_id!r}, minimum={minimum_completion}, "
                f"actual={completion}")
        if completion < self.current_ns:
            raise RuntimeError(
                "hybrid HBF external ASTRA completion moves time "
                f"backwards: current={self.current_ns}, "
                f"actual={completion}")

        # Drain every earlier Python-owned GPU event, but hold scheduling so
        # this HBF callback publishes before co-timed successor arrivals.
        self.advance(completion, defer_schedule=True)
        if self._external_hbf_pending.get(job_id) is not dispatch:
            raise RuntimeError(
                "hybrid HBF external ASTRA dispatch identity mismatch")
        if dispatch.owner == "lifecycle":
            result = self.hbf_lifecycle.complete_external_dispatch(
                job_id,
                arrival,
                completion,
                stages,
            )
            # Migration publication invalidates the retained GPU HBM copy.
            # Append publication may update a request already executing on
            # the HBF pool.  Both happen before any new scheduling.
            self._reconcile_migrated_gpu_hbm()
            self._refresh_live_hbf_placements()
        else:
            result = self.hbf_pool.complete_external_dispatch(
                job_id,
                arrival,
                completion,
                stages,
                defer_schedule=True,
            )
            self._consume_hbf_notifications(completion)
            self._refresh_live_hbf_placements()

        del self._external_hbf_pending[job_id]
        self._external_hbf_completed_job_ids.add(job_id)
        self.current_ns = completion
        if not defer_schedule:
            self.flush_scheduling(completion)
        if self.validate_every_event:
            self.assert_invariants()
        return result

    def has_pending_external(self) -> bool:
        """Return whether either HBF component awaits an ASTRA callback."""

        return (
            self.hbf_lifecycle.has_pending_external()
            or self.hbf_pool.has_pending_external_dispatches()
        )

    def _next_raw_event_ns(self) -> Optional[int]:
        values = []
        for value in (
            self.gpu_pool.next_event_ns(),
            self.hbf_pool.next_event_ns(),
            self.hbf_lifecycle.next_completion_ns(),
        ):
            if value is not None:
                values.append(value)
        if self._prepare_completion_heap:
            values.append(self._prepare_completion_heap[0][0])
        return min(values) if values else None

    def advance(
            self, now_ns: int, *,
            defer_schedule: bool = False) -> None:
        if now_ns < self.current_ns:
            raise ValueError(
                f"time cannot move backwards: current={self.current_ns}, "
                f"requested={now_ns}")
        while True:
            event_ns = self._next_raw_event_ns()
            if event_ns is None or event_ns > now_ns:
                break
            # Publication wins exact timestamp ties with arrivals.
            self.hbf_lifecycle.advance(event_ns)
            self._reconcile_migrated_gpu_hbm()
            self._refresh_live_hbf_placements()
            self.gpu_pool.advance(event_ns, defer_schedule=True)
            self.hbf_pool.advance(event_ns, defer_schedule=True)
            self._process_prepare_completions(event_ns)
            self._consume_gpu_notifications(event_ns)
            self._consume_hbf_notifications(event_ns)
            self.current_ns = event_ns
            if not (defer_schedule and event_ns == now_ns):
                self.flush_scheduling(event_ns)

        self.hbf_lifecycle.advance(now_ns)
        self._reconcile_migrated_gpu_hbm()
        self._refresh_live_hbf_placements()
        self.gpu_pool.advance(now_ns, defer_schedule=True)
        self.hbf_pool.advance(now_ns, defer_schedule=True)
        self.current_ns = now_ns
        if not defer_schedule:
            self.flush_scheduling(now_ns)
        if self.validate_every_event:
            self.assert_invariants()

    def flush_scheduling(self, now_ns: int) -> None:
        if (
            now_ns != self.current_ns
            or now_ns != self.gpu_pool.current_ns
            or now_ns != self.hbf_pool.current_ns
            or now_ns != self.hbf_lifecycle.current_ns
        ):
            raise ValueError(
                "flush_scheduling must run at the shared node timestamp")
        self._admit_pending(now_ns)
        self._submit_gpu_ready(now_ns)
        self.gpu_pool.flush_scheduling(now_ns)
        self.hbf_pool.flush_scheduling(now_ns)

    def next_event_ns(self) -> Optional[int]:
        """Return only events whose time is known outside ASTRA.

        External HBF work is intentionally absent.  Use
        :meth:`has_pending_external` to distinguish it from full idleness.
        """

        return self._next_raw_event_ns()

    def pop_completed(self) -> list[HybridCall]:
        result = []
        while self._user_completed_ids:
            result.append(
                self.calls[self._user_completed_ids.popleft()])
        return result

    def _deadlock_detail(self) -> str:
        unfinished = [
            call.request_id for call in self.calls.values()
            if call.state != HybridCallState.INTERNAL_COMPLETE
        ]
        return (
            f"unfinished={unfinished[:8]}, "
            f"pending={list(self._pending_call_ids)[:8]}, "
            f"hbf_external_pending={self.has_pending_external()}, "
            f"gpu_hbm_p={self.gpu_hbm.p_used_bytes_per_rank}, "
            f"gpu_hbm_d={self.gpu_hbm.d_used_bytes_per_rank}")

    def run_until_idle(self) -> list[HybridCall]:
        completed = self.pop_completed()
        while self.next_event_ns() is not None:
            event_ns = self.next_event_ns()
            assert event_ns is not None
            self.advance(event_ns)
            completed.extend(self.pop_completed())
        if self.has_pending_external():
            raise HybridDeadlockError(
                "external ASTRA HBF completions are pending; drain "
                "dispatches and apply complete_external_dispatch "
                "callbacks")
        if (
            self._pending_call_ids
            or self._gpu_ready_call_ids
            or self._prepare_completion_heap
            or any(
                call.state != HybridCallState.INTERNAL_COMPLETE
                for call in self.calls.values()
            )
        ):
            raise HybridDeadlockError(
                "hybrid node became idle before full drain: "
                + self._deadlock_detail())
        self.assert_invariants()
        return completed

    def assert_invariants(self) -> None:
        self.gpu_hbm.assert_invariants()
        self.gpu_pool.assert_invariants()
        self.hbf_lifecycle.assert_invariants()
        self.hbf_pool.assert_invariants()
        if self.hbf_execution_backend == "analytical_calendar":
            if self.hbf_calendar is None:
                raise AssertionError(
                    "analytical HBF backend lacks its shared calendar")
            if self.gpu_calendar is self.hbf_calendar:
                raise AssertionError(
                    "GPU and HBF server calendars must be independent")
            if (
                self._external_hbf_pending
                or self._external_hbf_completed_job_ids
            ):
                raise AssertionError(
                    "analytical hybrid node contains external HBF state")
            for reservation in self.hbf_calendar.reservations:
                if reservation.resource == "rdma-network" and (
                    reservation.namespace != "hbf-lifecycle"
                    or reservation.kind != "migration"
                ):
                    raise AssertionError(
                        "foreground HBF work consumed cross-server RDMA")
            rdma_count = (
                self.hbf_calendar
                .reservation_count_by_resource.get(
                    "rdma-network", 0)
            )
            rdma_bytes = (
                self.hbf_calendar
                .reservation_bytes_by_resource.get(
                    "rdma-network", 0)
            )
            if (
                rdma_count
                != self.hbf_lifecycle.metrics.migrations_started
                or rdma_bytes
                != self.hbf_lifecycle.metrics.migration_logical_bytes
            ):
                raise AssertionError(
                    "cross-server RDMA accounting includes "
                    "non-migration work")
        else:
            if self.hbf_calendar is not None:
                raise AssertionError(
                    "external ASTRA HBF backend exposed a shared Python "
                    "calendar")
            if (
                self.hbf_lifecycle.calendar
                is self.hbf_pool.calendar
            ):
                raise AssertionError(
                    "external HBF components share a Python calendar")
            lifecycle_issued = set(
                self.hbf_lifecycle._external_issued_job_ids)
            pool_issued = set(
                self.hbf_pool._external_issued_job_ids)
            if lifecycle_issued & pool_issued:
                raise AssertionError(
                    "lifecycle and pool issued the same ASTRA job id")
            expected_pending = lifecycle_issued | pool_issued
            if set(self._external_hbf_pending) != expected_pending:
                raise AssertionError(
                    "hybrid external HBF ownership disagrees with "
                    "component-issued jobs")
            expected_completed = (
                set(self.hbf_lifecycle
                    ._external_completed_job_ids)
                | set(self.hbf_pool
                      ._external_completed_job_ids)
            )
            if self._external_hbf_completed_job_ids != (
                    expected_completed):
                raise AssertionError(
                    "hybrid external HBF completion history disagrees "
                    "with its components")
            if (
                expected_pending
                & self._external_hbf_completed_job_ids
            ):
                raise AssertionError(
                    "completed hybrid external HBF job remains pending")
            for job_id, dispatch in (
                    self._external_hbf_pending.items()):
                if dispatch.job_id != job_id:
                    raise AssertionError(
                        "hybrid external HBF pending key mismatch")
                if dispatch.owner == "lifecycle":
                    if self.hbf_lifecycle._external_pending.get(
                            job_id) is not dispatch.dispatch:
                        raise AssertionError(
                            "hybrid lifecycle dispatch identity "
                            "mismatch")
                elif self.hbf_pool._external_pending.get(
                        job_id) is not dispatch.dispatch:
                    raise AssertionError(
                        "hybrid pool dispatch identity mismatch")
        if (
            self.gpu_calendar.reservation_count_by_resource.get(
                "rdma-network", 0)
            or self.gpu_calendar.reservation_bytes_by_resource.get(
                "rdma-network", 0)
        ):
            raise AssertionError(
                "GPU server calendar contains cross-server RDMA work")
        for call in self.calls.values():
            if (
                call.gpu_request is not None
                and call.hbf_request is not None
            ):
                raise AssertionError(
                    f"call executed on two servers: {call}")
            if call.execution == HybridExecution.HBF_READY:
                if (
                    call.hbf_request is None
                    and call.state
                    != HybridCallState.PENDING
                ):
                    raise AssertionError(
                        f"HBF route lacks HBF request: {call}")
                if call.admission_id is not None:
                    raise AssertionError(
                        f"HBF call owns GPU HBM: {call}")
            if call.state == HybridCallState.INTERNAL_COMPLETE:
                if (
                    call.user_completion_ns is None
                    or call.internal_completion_ns is None
                ):
                    raise AssertionError(
                        f"completed call lacks timestamps: {call}")
            if call.user_completion_ns is not None:
                request = call.serving_request
                if (
                    request is None
                    or request.completion_ns
                    != call.user_completion_ns
                ):
                    raise AssertionError(
                        f"user completion mismatch: {call}")
        expected_live_hbf = {
            call.request_id
            for call in self.calls.values()
            if call.state == HybridCallState.HBF_EXECUTING
        }
        if self._live_hbf_request_ids != expected_live_hbf:
            raise AssertionError(
                "live HBF request index disagrees with call states")
        for session_id, session in self.sessions.items():
            placement = self.hbf_lifecycle.sessions.get(session_id)
            if placement is None:
                if session.last_internal_call_index >= 0:
                    raise AssertionError(
                        "executed session lacks lifecycle state")
                continue
            if session.ended and placement.state != PlacementState.ENDED:
                raise AssertionError(
                    f"ended session retains lifecycle state: {placement}")
            if (
                self.gpu_hbm.active_admission(session_id) is None
                and self.gpu_hbm.d_bytes(session_id)
                and placement.gpu_retained_bytes == 0
            ):
                raise AssertionError(
                    "idle GPU HBM copy lacks lifecycle ownership")
        if self.metrics.user_completed_calls > self.metrics.submitted_calls:
            raise AssertionError(
                "hybrid completions exceed submissions")
        if self.metrics.internal_completed_calls > (
                self.metrics.user_completed_calls):
            raise AssertionError(
                "hybrid internal completions exceed user completions")

    def report(self) -> Mapping[str, Any]:
        execution_counts = Counter(
            call.execution.value
            for call in self.calls.values()
            if call.execution is not None
        )
        rdma_rows = (
            [
                asdict(row)
                for row in self.hbf_calendar.reservations
                if row.resource == "rdma-network"
            ]
            if self.hbf_calendar is not None else []
        )
        if self.hbf_calendar is None:
            rdma_detail_retained = False
            rdma_count = (
                self.hbf_lifecycle.metrics.migrations_started)
            rdma_logical_bytes = (
                self.hbf_lifecycle.metrics
                .migration_logical_bytes)
            rdma_accounting_source = "astra_causal_projection"
        else:
            rdma_detail_retained = (
                self.hbf_calendar.retain_reservations)
            rdma_count = (
                self.hbf_calendar
                .reservation_count_by_resource.get(
                    "rdma-network", 0))
            rdma_logical_bytes = (
                self.hbf_calendar
                .reservation_bytes_by_resource.get(
                    "rdma-network", 0))
            rdma_accounting_source = "python_analytical_calendar"
        return {
            "mode": "one_p4d4_gpu_plus_one_8card_full_model_hbf",
            "hbf_execution_backend": self.hbf_execution_backend,
            "hbf_server_id": self.hbf_server_id,
            "hbf_astra_chunk_bytes": self.hbf_astra_chunk_bytes,
            "hbf_completion_time_source": (
                "external_astra_callback"
                if self.hbf_execution_backend == "external_astra"
                else "python_analytical_calendar"
            ),
            "hbf_layout": asdict(self.hbf_layout),
            "hbf_topology_contract": {
                "card_count": self.hbf_hardware.card_count,
                "tp_group_count": self.hbf_layout.replicas,
                "cards_per_tp_group": self.hbf_layout.tp_size,
                "physical_kv_replication_factor": (
                    self.hbf_layout.physical_kv_replication_factor),
                "collective_fabric": (
                    "nonblocking intra-server fabric with per-card "
                    "endpoint bandwidth; disjoint TP4 groups may overlap "
                    "without a global switch-bisection penalty"),
                "pcie_roots": self.hbf_hardware.pcie_root_count,
                "cards_per_pcie_root": (
                    self.hbf_hardware.cards_per_pcie_root),
                "migration_ingress": (
                    "one RDMA stream, then bytes are charged to every "
                    "PCIe root and card interface touched by the selected "
                    "replica group"),
                "tp8_specific_assumption": (
                    "one eight-card TP group spans both four-card PCIe "
                    "roots; collectives use the intra-server fabric, and "
                    "GQA stores two physical KV copies"),
            },
            "server_resource_contract": {
                "gpu_compute_calendar_independent": True,
                "hbf_compute_calendar_independent": True,
                "shared_hbf_python_calendar": (
                    self.hbf_calendar is not None),
                "hbf_resource_timing_owner": (
                    "astra_shared_resource_dag"
                    if self.hbf_execution_backend == "external_astra"
                    else "python_analytical_calendar"
                ),
                "cross_server_link": "rdma-network",
                "cross_server_link_use": (
                    "turn-boundary GPU-KV migration only"),
                "hbf_ready_resume_network_round_trips": 0,
                "gpu_source_root_bandwidth_gbps": (
                    self.hbf_lifecycle
                    .gpu_source_root_bandwidth_gbps),
                "gpu_source_dma_contention": (
                    "serialized across migrations on a dedicated source "
                    "root/NIC DMA resource; GPU P/D peer traffic uses "
                    "NVLink and does not share that resource"),
                "same_timestamp_priority": (
                    "completion and its migration/append allocation "
                    "precede successor release and new arrivals"),
            },
            "validate_every_event": self.validate_every_event,
            "retain_detailed_history": self.retain_detailed_history,
            "retained_prepare_job_count": len(
                self.prepare_history),
            "metrics": asdict(self.metrics),
            "execution_counts": dict(sorted(execution_counts.items())),
            "rdma_migration_reservations": rdma_rows,
            "rdma_migration_summary": {
                "accounting_source": rdma_accounting_source,
                "reservation_detail_retained": rdma_detail_retained,
                "reservation_count": rdma_count,
                "logical_bytes": rdma_logical_bytes,
            },
            "gpu_resource_calendar": self.gpu_calendar.report(),
            "hbf_resource_calendar": (
                None
                if self.hbf_calendar is None
                else self.hbf_calendar.report()
            ),
            "external_hbf_pending_job_ids": sorted(
                self._external_hbf_pending),
            "external_hbf_completed_job_count": len(
                self._external_hbf_completed_job_ids),
            "external_hbf_undrained_dispatch_count": (
                self.hbf_lifecycle.report()[
                    "external_undrained_dispatch_count"]
                + self.hbf_pool.report()[
                    "external_undrained_dispatch_count"]
            ),
            "live_hbf_request_count": len(
                self._live_hbf_request_ids),
            "current_ns": self.current_ns,
            "gpu_hbm": self.gpu_hbm.report(),
            "gpu_pool": self.gpu_pool.report(),
            "hbf_lifecycle": self.hbf_lifecycle.report(),
            "hbf_pool": self.hbf_pool.report(),
            "sessions": {
                session_id: asdict(session)
                for session_id, session in sorted(self.sessions.items())
            },
            "calls": {
                request_id: {
                    "request_id": call.request_id,
                    "session_id": call.session_id,
                    "call_index": call.call_index,
                    "release_ns": call.release_ns,
                    "input_tokens": call.input_tokens,
                    "output_tokens": call.output_tokens,
                    "prefix_reuse_tokens": call.prefix_reuse_tokens,
                    "has_successor": call.has_successor,
                    "state": call.state.value,
                    "execution": (
                        None if call.execution is None
                        else call.execution.value
                    ),
                    "route_reason": call.route_reason,
                    "migration_inflight_at_route": (
                        call.migration_inflight_at_route),
                    "operational_reuse_tokens": (
                        call.operational_reuse_tokens),
                    "gpu_hit_tokens": call.gpu_hit_tokens,
                    "admission_id": call.admission_id,
                    "prepare_start_ns": call.prepare_start_ns,
                    "prepare_completion_ns": (
                        call.prepare_completion_ns),
                    "user_completion_ns": call.user_completion_ns,
                    "internal_completion_ns": (
                        call.internal_completion_ns),
                    "ttft_ns": call.ttft_ns,
                    "tpot_ns": call.tpot_ns,
                }
                for request_id, call in sorted(self.calls.items())
            },
        }


@dataclass(frozen=True)
class HybridCallSpec:
    request_id: int
    key: RequestKey
    source_index: int
    offer_index: int
    session_id: str
    call_index: int
    input_tokens: int
    output_tokens: int
    cached_prefix_tokens: int
    tool_duration_ns: int
    has_successor: bool

    @property
    def completion_identity(self) -> str:
        return (
            f"{self.key.session_id}::call-"
            f"{self.key.sub_request_index}"
        )


@dataclass
class HybridSystemMetrics:
    scheduled_sessions: int = 0
    scheduled_calls: int = 0
    released_calls: int = 0
    completed_calls: int = 0
    event_timestamps: int = 0
    fixed_point_rounds: int = 0
    max_release_heap: int = 0


class GPUHBFHybridSystem(ResumableCutoffEventLoopMixin):
    """Dynamic agentic-session event loop for the proposed two-server system."""

    def __init__(
            self, *, repo_root: Path,
            gpu_hardware: Optional[P4D4GPUHardware] = None,
            hbf_hardware: Optional[HBFServerHardware] = None,
            hbf_layout: str = "tp4",
            max_num_batched_tokens: int = 8_192,
            max_num_seqs: int = 128,
            p_max_num_seqs: Optional[int] = None,
            d_max_num_seqs: Optional[int] = None,
            max_prefill_chunk_tokens: int = 4_096,
            band: str = "central",
            validate_every_event: bool = True,
            hbf_execution_backend: str = (
                "analytical_calendar"),
            hbf_server_id: Optional[int] = None,
            hbf_astra_chunk_bytes: int = 64 * 1024 ** 2) -> None:
        self.node = GPUHBFHybridNode(
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
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            p_max_num_seqs=p_max_num_seqs,
            d_max_num_seqs=d_max_num_seqs,
            max_prefill_chunk_tokens=max_prefill_chunk_tokens,
            band=band,
            validate_every_event=validate_every_event,
            hbf_execution_backend=hbf_execution_backend,
            hbf_server_id=hbf_server_id,
            hbf_astra_chunk_bytes=hbf_astra_chunk_bytes,
        )
        self.validate_every_event = validate_every_event
        self.metrics = HybridSystemMetrics()
        self.current_ns = 0
        self.call_specs: tuple[HybridCallSpec, ...] = ()
        self._spec_by_request: dict[int, HybridCallSpec] = {}
        self._request_by_identity: dict[str, int] = {}
        self._successor_by_request: dict[int, int] = {}
        self._release_heap: list[tuple[int, int]] = []
        self._queued_release_ids: set[int] = set()
        self._released_ids: set[int] = set()
        self._completed_ids: set[int] = set()
        self._runtime_calls: dict[int, HybridCall] = {}
        self._completed_snapshots: dict[int, CompletedRequest] = {}
        self._completion_order: list[int] = []
        self._completed_session_order: list[str] = []
        self._offered_session_order: list[str] = []
        self._loaded = False
        self._running = False
        self._finished = False

    @property
    def completed_requests(self) -> tuple[CompletedRequest, ...]:
        return tuple(
            self._completed_snapshots[request_id]
            for request_id in self._completion_order
        )

    @staticmethod
    def _validate_scheduled_session(
            scheduled: ScheduledSession) -> None:
        if not isinstance(scheduled, ScheduledSession):
            raise TypeError(
                "scheduled sessions must be ScheduledSession values")
        for name in ("offer_index", "arrival_time_ns"):
            value = getattr(scheduled, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"scheduled {name} must be a non-negative integer")
        session = scheduled.session
        if not session.session_id or not session.calls:
            raise ValueError("scheduled session must have calls")
        for call_index, call in enumerate(session.calls):
            if call.session_id != session.session_id:
                raise ValueError(
                    "call/session identity mismatch")
            if call.call_index != call_index:
                raise ValueError(
                    "scheduled calls must use contiguous indices")
            if call_index == 0 and call.cached_prefix_tokens:
                raise ValueError(
                    "first call cannot reuse an earlier prefix")
            if not 0 <= call.cached_prefix_tokens <= call.input_tokens:
                raise ValueError(
                    "cached prefix must be in 0..input")
            if call.tool_duration_ns < 0:
                raise ValueError(
                    "tool duration must be non-negative")
            if call.input_tokens <= 0 or call.output_tokens <= 0:
                raise ValueError(
                    "input/output tokens must be positive")
            if call.input_tokens + call.output_tokens - 1 > 1_010_000:
                raise ValueError(
                    "scheduled call exceeds context contract")

    def load(
            self,
            scheduled_sessions: Iterable[ScheduledSession]) -> None:
        if self._loaded:
            raise RuntimeError("hybrid schedule is already loaded")
        values = list(scheduled_sessions)
        if not values:
            raise ValueError("scheduled_sessions cannot be empty")
        for scheduled in values:
            self._validate_scheduled_session(scheduled)
        offer_indices = [value.offer_index for value in values]
        session_ids = [value.session.session_id for value in values]
        source_indices = [value.session.source_index for value in values]
        if len(offer_indices) != len(set(offer_indices)):
            raise ValueError("duplicate scheduled offer indices")
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("duplicate scheduled session IDs")
        if len(source_indices) != len(set(source_indices)):
            raise ValueError("duplicate scheduled source indices")

        self._offered_session_order = [
            value.session.session_id for value in sorted(
                values, key=lambda item: item.offer_index)
        ]
        canonical = sorted(
            values,
            key=lambda item: (
                item.session.source_index,
                item.session.session_id,
            ),
        )
        specs = []
        identities = set()
        next_request_id = 0
        for scheduled in canonical:
            prior_request_id = None
            for call_index, call in enumerate(
                    scheduled.session.calls):
                if call.completion_identity in identities:
                    raise ValueError(
                        "duplicate completion identity")
                identities.add(call.completion_identity)
                request_id = next_request_id
                next_request_id += 1
                spec = HybridCallSpec(
                    request_id=request_id,
                    key=RequestKey(
                        scheduled.session.session_id,
                        call_index,
                    ),
                    source_index=scheduled.session.source_index,
                    offer_index=scheduled.offer_index,
                    session_id=scheduled.session.session_id,
                    call_index=call_index,
                    input_tokens=call.input_tokens,
                    output_tokens=call.output_tokens,
                    cached_prefix_tokens=call.cached_prefix_tokens,
                    tool_duration_ns=call.tool_duration_ns,
                    has_successor=(
                        call_index + 1
                        < len(scheduled.session.calls)
                    ),
                )
                specs.append(spec)
                self._spec_by_request[request_id] = spec
                self._request_by_identity[
                    call.completion_identity] = request_id
                if prior_request_id is None:
                    self._queue_release(
                        request_id,
                        release_ns=scheduled.arrival_time_ns,
                    )
                else:
                    self._successor_by_request[
                        prior_request_id] = request_id
                prior_request_id = request_id
        self.call_specs = tuple(specs)
        self.metrics.scheduled_sessions = len(values)
        self.metrics.scheduled_calls = len(specs)
        self._loaded = True
        self.assert_invariants()

    def request_id_for(self, completion_identity: str) -> int:
        try:
            return self._request_by_identity[completion_identity]
        except KeyError as exc:
            raise KeyError(
                f"unknown completion identity="
                f"{completion_identity!r}") from exc

    def _queue_release(
            self, request_id: int, *,
            release_ns: int) -> None:
        if request_id in self._queued_release_ids:
            raise RuntimeError(
                f"request_id={request_id} was released twice")
        if (
            isinstance(release_ns, bool)
            or not isinstance(release_ns, int)
            or release_ns < 0
        ):
            raise ValueError(
                "release_ns must be a non-negative integer")
        heapq.heappush(
            self._release_heap, (release_ns, request_id))
        self._queued_release_ids.add(request_id)
        self.metrics.max_release_heap = max(
            self.metrics.max_release_heap,
            len(self._release_heap),
        )

    def _consume_completions(self, now_ns: int) -> None:
        completed = sorted(
            self.node.pop_completed(),
            key=lambda call: (
                call.user_completion_ns,
                call.request_id,
            ),
        )
        for call in completed:
            request_id = call.request_id
            if request_id in self._completed_ids:
                raise RuntimeError(
                    "duplicate hybrid system completion")
            if call.user_completion_ns != now_ns:
                raise RuntimeError(
                    "hybrid completion at wrong global time")
            spec = self._spec_by_request[request_id]
            if (
                call.session_id != spec.session_id
                or call.call_index != spec.call_index
            ):
                raise RuntimeError(
                    "runtime call disagrees with frozen spec")
            request = call.serving_request
            if (
                request is None
                or request.first_token_ns is None
                or request.completion_ns is None
            ):
                raise RuntimeError(
                    "completed hybrid call lacks token timestamps")
            self._completed_ids.add(request_id)
            self._completion_order.append(request_id)
            self._completed_snapshots[request_id] = CompletedRequest(
                key=spec.key,
                release_ns=call.release_ns,
                first_token_ns=request.first_token_ns,
                completion_ns=request.completion_ns,
                output_tokens=call.output_tokens,
            )
            self.metrics.completed_calls += 1
            successor_id = self._successor_by_request.get(request_id)
            if successor_id is None:
                self._completed_session_order.append(
                    spec.session_id)
            else:
                self._queue_release(
                    successor_id,
                    release_ns=(
                        now_ns + spec.tool_duration_ns),
                )

    def _pop_releases(self, now_ns: int) -> list[HybridCall]:
        result = []
        while (
            self._release_heap
            and self._release_heap[0][0] == now_ns
        ):
            release_ns, request_id = heapq.heappop(
                self._release_heap)
            if request_id in self._released_ids:
                raise RuntimeError(
                    "duplicate hybrid runtime release")
            spec = self._spec_by_request[request_id]
            call = HybridCall(
                request_id=request_id,
                session_id=spec.session_id,
                call_index=spec.call_index,
                release_ns=release_ns,
                input_tokens=spec.input_tokens,
                output_tokens=spec.output_tokens,
                prefix_reuse_tokens=spec.cached_prefix_tokens,
                has_successor=spec.has_successor,
            )
            self._released_ids.add(request_id)
            self._runtime_calls[request_id] = call
            self.metrics.released_calls += 1
            result.append(call)
        if self._release_heap and self._release_heap[0][0] < now_ns:
            raise RuntimeError(
                "hybrid release heap fell behind current time")
        return result

    def _same_time_work_exists(self, now_ns: int) -> bool:
        return (
            bool(
                self._release_heap
                and self._release_heap[0][0] == now_ns
            )
            or self.node.next_event_ns() == now_ns
        )

    def _process_timestamp(self, now_ns: int) -> None:
        if now_ns < self.current_ns:
            raise ValueError(
                "hybrid system time cannot move backwards")
        self.metrics.event_timestamps += 1
        rounds = 0
        while True:
            rounds += 1
            if rounds > self.metrics.scheduled_calls + 4:
                raise HybridDeadlockError(
                    "same-timestamp hybrid fixed point "
                    "did not converge")
            self.node.advance(now_ns, defer_schedule=True)
            self._consume_completions(now_ns)
            arrivals = self._pop_releases(now_ns)
            if arrivals:
                arrivals.sort(key=lambda call: call.request_id)
                self.node.submit_many(
                    arrivals, now_ns=now_ns)
            else:
                self.node.flush_scheduling(now_ns)
            self.current_ns = now_ns
            self.metrics.fixed_point_rounds += 1
            if not self._same_time_work_exists(now_ns):
                break
        if self.validate_every_event:
            self.assert_invariants()

    def _next_event_ns(self) -> Optional[int]:
        values = []
        if self._release_heap:
            values.append(self._release_heap[0][0])
        node_event = self.node.next_event_ns()
        if node_event is not None:
            values.append(node_event)
        return min(values) if values else None

    def _deadlock_detail(self) -> str:
        unreleased = sorted(
            set(self._spec_by_request) - self._released_ids)
        unfinished = sorted(
            set(self._spec_by_request) - self._completed_ids)
        return (
            f"unreleased={unreleased[:8]}, "
            f"unfinished={unfinished[:8]}, "
            + self.node._deadlock_detail()
        )

    def run(
            self,
            scheduled_sessions: Optional[
                Iterable[ScheduledSession]] = None,
    ) -> list[CompletedRequest]:
        if scheduled_sessions is not None:
            self.load(scheduled_sessions)
        if not self._loaded:
            raise RuntimeError("load a schedule before running")
        if self._running:
            raise RuntimeError("hybrid system is already running")
        if self._finished:
            return list(self.completed_requests)

        self._running = True
        try:
            while True:
                event_ns = self._next_event_ns()
                if event_ns is None:
                    if self.node.has_pending_external():
                        raise HybridDeadlockError(
                            "external ASTRA HBF completions are pending; "
                            "drive node.drain_external_dispatches() "
                            "through ASTRA and apply "
                            "node.complete_external_dispatch() callbacks")
                    fully_drained = (
                        len(self._completed_ids)
                        == self.metrics.scheduled_calls
                        and len(self._released_ids)
                        == self.metrics.scheduled_calls
                        and all(
                            call.state
                            == HybridCallState.INTERNAL_COMPLETE
                            for call in self.node.calls.values()
                        )
                    )
                    if fully_drained:
                        break
                    raise HybridDeadlockError(
                        "unfinished hybrid work has no future event: "
                        + self._deadlock_detail())
                self._process_timestamp(event_ns)
        finally:
            self._running = False

        self._finished = True
        self.assert_invariants()
        return list(self.completed_requests)

    def assert_invariants(self) -> None:
        self.node.assert_invariants()
        if not self._completed_ids <= self._released_ids:
            raise AssertionError(
                "completed hybrid request was never released")
        if self.metrics.released_calls != len(self._released_ids):
            raise AssertionError(
                "hybrid released-call metric mismatch")
        if self.metrics.completed_calls != len(self._completed_ids):
            raise AssertionError(
                "hybrid completed-call metric mismatch")
        if self._completed_ids != set(self._completed_snapshots):
            raise AssertionError(
                "hybrid completion snapshot mismatch")
        if self._finished and len(self._completed_ids) != (
                len(self.call_specs)):
            raise AssertionError(
                "finished hybrid system lacks completions")
        if len(self._completed_session_order) != len(set(
                self._completed_session_order)):
            raise AssertionError(
                "hybrid session completed more than once")
        if self._finished and len(self._completed_session_order) != (
                self.metrics.scheduled_sessions):
            raise AssertionError(
                "finished hybrid system lacks session completions")

    def report(self) -> Mapping[str, Any]:
        spec_rows = [asdict(spec) for spec in self.call_specs]
        completed_identities = [
            self._spec_by_request[
                request_id].completion_identity
            for request_id in self._completion_order
        ]
        result: dict[str, Any] = {
            "mode": "gpu_hbf_hybrid_agentic_system",
            "architecture": {
                "gpu_server": "one_4p4d_h100_server",
                "hbf_server": (
                    "one_8card_full_model_hbf_npu_server"),
                "hbf_layout": self.node.hbf_layout.key,
                "hbf_execution_backend": (
                    self.node.hbf_execution_backend),
                "hbf_server_id": self.node.hbf_server_id,
                "hbf_astra_chunk_bytes": (
                    self.node.hbf_astra_chunk_bytes),
                "ssd_present": False,
            },
            "policy": {
                "first_turn": "gpu",
                "migration_inflight_resume": "gpu",
                "hbf_ready_resume": "hbf_end_to_end",
                "hbf_lpddr_finish_capacity_miss": (
                    "atomic_admission_then_full_gpu_recompute"),
                "hbf_ready_per_call_network_round_trip": False,
            },
            "metrics": asdict(self.metrics),
            "current_ns": self.current_ns,
            "finished": self._finished,
            "call_specs": spec_rows,
            "call_specs_identity_sha256": stable_json_sha256(
                spec_rows),
            "completion_order": completed_identities,
            "completed_requests": [
                asdict(request) for request in self.completed_requests],
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


GPUHBFHybridRunner = GPUHBFHybridSystem


__all__ = [
    "GPUHBFHybridNode",
    "GPUHBFHybridRunner",
    "GPUHBFHybridSystem",
    "HybridCall",
    "HybridCallSpec",
    "HybridCallState",
    "HybridDeadlockError",
    "HybridExecution",
    "HybridGPUPrepareJob",
    "HybridHBFExternalDispatch",
    "HybridNodeMetrics",
    "HybridSession",
    "HybridSystemMetrics",
    "SUPPORTED_HBF_LAYOUTS",
    "SUPPORTED_HBF_EXECUTION_BACKENDS",
]

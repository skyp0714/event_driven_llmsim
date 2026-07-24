"""Live Router adapter for full-model HBF serving through ASTRA.

This module owns the boundary between three otherwise independent pieces:

* raw agentic rows produced by :class:`serving.core.router.Router`;
* native GPU requests completed by the ordinary LLMServingSim scheduler;
* full-model HBF lifecycle and serving jobs completed by ASTRA callbacks.

The adapter deliberately does not reach into Router or Scheduler internals.
An integration must offer each due raw row before constructing a native GPU
``Request``.  GPU decisions continue through the existing path; HBF decisions
are removed from that path and submitted with :meth:`flush_admissions`.

All HBF timing is callback-owned.  The adapter never predicts an external
completion timestamp.  Lifecycle and foreground jobs share one global ASTRA
job namespace through :class:`HBFAstraJobMultiplexer`.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

from .hbf_astra_multiplexer import (
    HBFAstraJobMultiplexer,
    HBFAstraMultiplexedCompletion,
    HBFAstraMultiplexedJob,
)
from .hbf_full_model_lifecycle import (
    ActivePrefillDrainResult,
    ActivePrefillDrainStatus,
    AppendJob,
    FullModelHBFLifecycle,
    MigrationJob,
    PlacementState,
    ResumeExecution,
    ResumeRoute,
)
from .hbf_full_model_pool import (
    FullModelHBFServingPool,
    HBFRequestState,
    HBFServingRequest,
)


ONLINE_HBF_ADAPTER_SCHEMA = "full-model-hbf-online-adapter-v1"
SUPPORTED_GPU_RESUME_MODES = frozenset({
    "sticky_reuse",
    "recompute",
})


class OnlineHBFExecution(str, Enum):
    """Execution selected for one raw Router request."""

    GPU_FIRST_TURN = "gpu_first_turn"
    GPU_MIGRATION_INFLIGHT = "gpu_migration_inflight"
    GPU_CAPACITY_FALLBACK = "gpu_capacity_fallback"
    GPU_RECOMPUTE = "gpu_recompute"
    GPU_OWNED = "gpu_owned"
    HBF_READY = "hbf_ready"


class OnlineHBFCallState(str, Enum):
    GPU_ACTIVE = "gpu_active"
    HBF_STAGED = "hbf_staged"
    HBF_ACTIVE = "hbf_active"
    COMPLETE = "complete"


class GPUHBMEventKind(str, Enum):
    """Required native GPU-HBM ownership handoffs.

    These are accounting obligations, not synthetic transfer delays.  The
    live integration must map them to its finite-HBM owner:

    * ``TURN_RETAIN`` transfers completed active KV to idle migration
      ownership;
    * ``RESUME_CLAIM`` transfers retained idle KV back to the GPU request;
    * ``MIGRATION_RELEASE`` frees the retained source after valid HBF
      publication.
    * ``IDLE_RELEASE`` frees retained GPU KV when a measurement barrier
      censors the logical successor.

    ``logical_bytes`` follows the lifecycle's unrounded whole-model lineage.
    ``per_rank_bytes`` is GPU-TP-sharded and block-rounded for direct use by
    finite-HBM admission accounting.
    """

    TURN_RETAIN = "turn_retain"
    RESUME_CLAIM = "resume_claim"
    MIGRATION_RELEASE = "migration_release"
    IDLE_RELEASE = "idle_release"


@dataclass(frozen=True)
class GPUHBMOwnershipEvent:
    kind: GPUHBMEventKind
    session_id: str
    request_id: int
    gpu_instance_id: int
    time_ns: int
    token_count: int
    accounted_tokens_per_rank: int
    logical_bytes: int
    per_rank_bytes: int
    reason: str


@dataclass(frozen=True)
class RouterCompletionProxy:
    """HBF completion accepted by Router and materializable for reporting."""

    id: int
    session_id: str
    sub_request_index: int
    arrival: int
    end_time: int
    original_input: int
    input: int
    output: int
    num_computed_tokens: int
    generated_tokens: int
    ttft: int
    tpot: float
    latency: int
    admission_ns: int
    first_schedule_time_ns: int
    hbf_prefix_tokens: int
    lpddr_prefix_tokens: int
    token_completion_ns: tuple[int, ...]
    hbf_online_execution: str
    hbf_online_route_reason: str
    source_session_id: Optional[str] = None
    session_template_index: Optional[int] = None
    session_epoch: int = 0
    wakekv_has_successor: bool = False
    raw_metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}),
        compare=False,
        repr=False,
    )

    def materialize_request(
            self, *, model: str,
            instance_id: int) -> Any:
        """Return a completed native ``Request`` for metrics and CSV output."""

        from .request import Request

        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        instance = _integer("instance_id", instance_id)
        completion_times = tuple(
            _integer("token_completion_ns", value)
            for value in self.token_completion_ns
        )
        if len(completion_times) != self.generated_tokens:
            raise RuntimeError(
                "HBF token-completion count differs from generated tokens")
        if not completion_times:
            raise RuntimeError("completed HBF request has no output tokens")
        if completion_times[0] - self.arrival != self.ttft:
            raise RuntimeError(
                "HBF first-token timestamp does not reconcile with TTFT")
        if completion_times[-1] != self.end_time:
            raise RuntimeError(
                "HBF final-token timestamp differs from completion")
        if self.first_schedule_time_ns < self.admission_ns:
            raise RuntimeError(
                "HBF request was scheduled before physical admission")

        request = Request(
            self.id,
            model,
            self.original_input,
            self.output,
            self.arrival,
            instance,
        )
        raw = self.raw_metadata
        request.session_id = self.session_id
        request.sub_request_index = self.sub_request_index
        request.source_session_id = self.source_session_id
        request.session_template_index = self.session_template_index
        request.session_epoch = self.session_epoch
        offered = raw.get("session_offered_time_ns")
        admitted = raw.get("session_admission_time_ns")
        admission_wait = raw.get("session_admission_queue_wait_ns")
        request.session_offered_time_ns = int(
            self.arrival if offered is None else offered)
        request.session_admission_time_ns = int(
            self.arrival if admitted is None else admitted)
        request.session_admission_queue_wait_ns = int(
            0 if admission_wait is None else admission_wait)
        request.ready_time = self.admission_ns
        request.scheduler_resource_ready_time_ns = self.admission_ns
        request.first_schedule_time_ns = self.first_schedule_time_ns
        request.first_schedule_eligibility_time_ns = self.admission_ns
        request.first_schedule_request_ready_time_ns = self.admission_ns
        request.first_schedule_resource_ready_time_ns = self.admission_ns
        request.scheduler_queue_wait_ns = (
            self.first_schedule_time_ns - self.admission_ns)
        request.queuing_delay = (
            self.first_schedule_time_ns - self.arrival)

        cached_tokens = self.hbf_prefix_tokens + self.lpddr_prefix_tokens
        request.prefix_reuse_tokens = int(
            raw.get("prefix_reuse_toks") or 0)
        request.prefix_reuse_source = raw.get("prefix_reuse_source")
        request.return_gap_type = str(
            raw.get("return_gap_type") or "unknown")
        request.return_gap_source = str(
            raw.get("return_gap_source") or "unknown")
        request.return_gap_ns = int(raw.get("return_gap_ns") or 0)
        request.agentic_kv_hit_tokens = cached_tokens
        request.agentic_kv_recompute_tokens = 0
        request.agentic_kv_residency_at_return = "hbf"
        request.agentic_kv_source = "hbf"
        request.hbf_online_execution = self.hbf_online_execution
        request.hbf_online_route_reason = self.hbf_online_route_reason
        admission_delay_ns = self.admission_ns - self.arrival
        request.agentic_kv_owner_gate_ns = admission_delay_ns
        request.agentic_kv_prepare_boundary_wait_ns = admission_delay_ns
        request.agentic_kv_restore_issue_time_ns = self.admission_ns
        request.agentic_kv_target_hbm_ready_time_ns = self.admission_ns
        request.agentic_kv_restore_ready_time_ns = self.admission_ns
        request.agentic_kv_fresh_prompt_tokens = max(
            0, self.original_input - cached_tokens)

        request.num_computed_tokens = self.num_computed_tokens
        request.generated_tokens = self.generated_tokens
        request.end_time = self.end_time
        request.latency = self.latency
        request.ttft = int(self.ttft)
        request.tpot = int(self.tpot)
        request.itl = [
            current - previous
            for previous, current in zip(
                completion_times, completion_times[1:])
        ]
        request.recent_end = completion_times[-1]
        request.is_init = False
        request.wakekv_has_successor = self.wakekv_has_successor
        return request


@dataclass
class OnlineHBFCall:
    request_id: int
    session_id: str
    call_index: int
    trace_arrival_ns: int
    admission_ns: int
    input_tokens: int
    output_tokens: int
    requested_prefix_reuse_tokens: int
    operational_prefix_reuse_tokens: int
    has_successor: bool
    execution: OnlineHBFExecution
    route_reason: str
    migration_inflight: bool
    state: OnlineHBFCallState
    gpu_prefix_reuse_tokens: int
    gpu_instance_id: Optional[int]
    residency_at_return: Optional[str]
    kv_source: Optional[str]
    raw_metadata: Mapping[str, Any] = field(repr=False)
    hbf_request: Optional[HBFServingRequest] = None
    completion_ns: Optional[int] = None
    successor_censored: bool = False

    @property
    def final_materialized_tokens(self) -> int:
        return self.input_tokens + self.output_tokens - 1


@dataclass(frozen=True)
class OnlineHBFRouteDecision:
    request_id: int
    session_id: str
    call_index: int
    execution: OnlineHBFExecution
    route_reason: str
    operational_prefix_reuse_tokens: int
    gpu_prefix_reuse_tokens: int
    required_gpu_instance_id: Optional[int]
    migration_inflight: bool
    hbf_request: Optional[HBFServingRequest] = field(
        compare=False, repr=False)

    @property
    def divert_to_hbf(self) -> bool:
        return self.execution == OnlineHBFExecution.HBF_READY

    @property
    def run_on_gpu(self) -> bool:
        return not self.divert_to_hbf

    @property
    def force_gpu_recompute(self) -> bool:
        return self.execution == OnlineHBFExecution.GPU_RECOMPUTE


@dataclass(frozen=True)
class OnlineHBFAstraCompletion:
    """One routed ASTRA callback and any user-visible HBF completions."""

    multiplexed: HBFAstraMultiplexedCompletion
    router_completions: tuple[RouterCompletionProxy, ...]


@dataclass
class OnlineHBFAdapterMetrics:
    offered_requests: int = 0
    gpu_requests: int = 0
    hbf_requests: int = 0
    gpu_completions: int = 0
    hbf_completions: int = 0
    router_completion_proxies: int = 0
    astra_callbacks: int = 0
    astra_lifecycle_callbacks: int = 0
    astra_pool_callbacks: int = 0
    gpu_hbm_turn_retain_events: int = 0
    gpu_hbm_resume_claim_events: int = 0
    gpu_hbm_migration_release_events: int = 0
    gpu_hbm_idle_release_events: int = 0
    gpu_ready_hbm_pressure_reclaims: int = 0
    gpu_ready_hbm_pressure_reclaimed_logical_bytes: int = 0
    gpu_ready_hbm_pressure_reclaimed_per_rank_bytes: int = 0
    censored_successors: int = 0
    censored_active_gpu_requests: int = 0
    censored_queued_gpu_requests: int = 0


def _integer(
        name: str, value: object, *,
        minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer >= {minimum}")
    return value


def _raw_integer(
        row: Mapping[str, Any], name: str, *,
        default: Optional[int] = None,
        minimum: int = 0) -> int:
    if name not in row:
        if default is None:
            raise ValueError(f"raw Router row is missing {name!r}")
        value = default
    else:
        value = row[name]
    if isinstance(value, bool):
        raise ValueError(f"raw Router field {name!r} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"raw Router field {name!r} must be an integer") from exc
    if result < minimum:
        raise ValueError(
            f"raw Router field {name!r} must be >= {minimum}")
    return result


class FullModelHBFOnlineAdapter:
    """Route live agentic turns between native GPU and full-model HBF."""

    def __init__(
            self, *, lifecycle: FullModelHBFLifecycle,
            pool: FullModelHBFServingPool,
            lifecycle_source_name: str = "full-model-lifecycle",
            pool_source_name: str = "full-model-pool",
            gpu_tp_size: int = 4,
            gpu_block_size_tokens: int = 16,
            gpu_resume_mode: str = "sticky_reuse",
            validate_every_event: bool = True) -> None:
        if not isinstance(lifecycle, FullModelHBFLifecycle):
            raise TypeError(
                "lifecycle must be a FullModelHBFLifecycle")
        if not isinstance(pool, FullModelHBFServingPool):
            raise TypeError("pool must be a FullModelHBFServingPool")
        if lifecycle.execution_backend != "external_astra":
            raise ValueError(
                "online lifecycle must use external_astra")
        if pool.execution_backend != "external_astra":
            raise ValueError("online pool must use external_astra")
        if lifecycle.hardware != pool.hardware:
            raise ValueError("lifecycle/pool HBF hardware differs")
        if lifecycle.layout != pool.layout:
            raise ValueError("lifecycle/pool HBF layout differs")
        if lifecycle.server_id != pool.server_id:
            raise ValueError("lifecycle/pool ASTRA server_id differs")
        if lifecycle.lpddr_ledger is not pool.lpddr_ledger:
            raise ValueError(
                "lifecycle and pool must share one LPDDR ledger")
        expected_active_bytes = int(math.ceil(
            lifecycle.kv_bytes_per_token
            * lifecycle.layout.physical_kv_replication_factor
            / lifecycle.layout.tp_size
        ))
        if (
            pool.kv_bytes_per_active_token_per_card
            != expected_active_bytes
        ):
            raise ValueError(
                "lifecycle/pool per-card KV accounting differs")
        if lifecycle.current_ns != pool.current_ns:
            raise ValueError(
                "lifecycle/pool clocks must match at adapter creation")
        if lifecycle.sessions or pool.requests:
            raise ValueError(
                "online adapter requires pristine lifecycle and pool")
        self.gpu_tp_size = _integer(
            "gpu_tp_size", gpu_tp_size, minimum=1)
        self.gpu_block_size_tokens = _integer(
            "gpu_block_size_tokens",
            gpu_block_size_tokens,
            minimum=1,
        )
        if (
            not isinstance(gpu_resume_mode, str)
            or gpu_resume_mode not in SUPPORTED_GPU_RESUME_MODES
        ):
            raise ValueError(
                "gpu_resume_mode must be one of "
                f"{sorted(SUPPORTED_GPU_RESUME_MODES)}")
        self.gpu_resume_mode = gpu_resume_mode
        self.gpu_kv_bytes_per_token_per_rank = int(math.ceil(
            lifecycle.kv_bytes_per_token / self.gpu_tp_size))
        if not isinstance(validate_every_event, bool):
            raise ValueError("validate_every_event must be a boolean")

        resolver = pool.placement_resolver
        if resolver is None:
            pool.placement_resolver = lifecycle.placement_snapshot
        elif not (
            getattr(resolver, "__self__", None) is lifecycle
            and getattr(resolver, "__func__", None)
            is FullModelHBFLifecycle.placement_snapshot
        ):
            raise ValueError(
                "pool placement_resolver must be the paired lifecycle")

        self.lifecycle = lifecycle
        self.pool = pool
        self.validate_every_event = validate_every_event
        self.current_ns = lifecycle.current_ns
        self.calls: dict[int, OnlineHBFCall] = {}
        self._active_request_by_session: dict[str, int] = {}
        self._last_call_index_by_session: dict[str, int] = {}
        self._ended_sessions: set[str] = set()
        self._gpu_owner_instance_by_session: dict[str, int] = {}
        self._staged_hbf_by_time: dict[
            int, list[HBFServingRequest]] = {}
        self._completed_router_proxies: deque[
            RouterCompletionProxy] = deque()
        self._pending_hbf_completion_by_request: dict[
            int, tuple[HBFServingRequest, RouterCompletionProxy]] = {}
        self._prefill_drain_request_by_session: dict[str, int] = {}
        self._prefill_drain_request_by_job: dict[int, int] = {}
        self._prefill_drain_waiting_append_by_session: dict[
            str, tuple[int, ...]] = {}
        self._gpu_hbm_events: deque[GPUHBMOwnershipEvent] = deque()
        self._gpu_ready_hbm_pressure_reclaim_audits: list[
            dict[str, Any]] = []
        self.metrics = OnlineHBFAdapterMetrics()
        self._execution_counts: Counter[str] = Counter()

        self.multiplexer = HBFAstraJobMultiplexer()
        self.multiplexer.register_object(
            lifecycle_source_name,
            lifecycle,
            drain_method="drain_external_dispatches",
            complete_method="complete_external_dispatch",
            has_pending_method="has_pending_external",
        )
        self.multiplexer.register_object(
            pool_source_name,
            pool,
            drain_method="drain_external_dispatches",
            complete_method="complete_external_dispatch",
            has_pending_method="has_pending_external_dispatches",
            complete_kwargs={"defer_schedule": True},
        )
        self.lifecycle_source_name = lifecycle_source_name
        self.pool_source_name = pool_source_name
        if self.validate_every_event:
            self.assert_invariants()

    @staticmethod
    def _copy_raw_metadata(
            row: Mapping[str, Any]) -> Mapping[str, Any]:
        return MappingProxyType(dict(row))

    def _refresh_live_hbf_placements(self) -> None:
        """Publish lifecycle placement changes before pool validation."""

        for call in self.calls.values():
            if call.state not in {
                    OnlineHBFCallState.HBF_STAGED,
                    OnlineHBFCallState.HBF_ACTIVE}:
                continue
            request = call.hbf_request
            if request is None:
                raise RuntimeError(
                    "live HBF call lost its serving request")
            old_published = request.published_tokens
            hbf_tokens, lpddr_tokens, group_id = (
                self.lifecycle.placement_snapshot(call.session_id))
            if group_id != request.group_id:
                raise RuntimeError(
                    "active HBF request changed replica group")
            if hbf_tokens + lpddr_tokens != old_published:
                raise RuntimeError(
                    "append publication changed the published logical "
                    "request prefix")
            request.hbf_prefix_tokens = hbf_tokens
            request.lpddr_prefix_tokens = lpddr_tokens

    def _record_prefill_drain_intents(self) -> None:
        """Record newly gated requests without claiming lifecycle work."""

        for worker in self.pool.workers:
            for request_id in worker.prefill_drain:
                request = self.pool.requests[request_id]
                if request.state != HBFRequestState.PREFILL_DRAIN:
                    raise RuntimeError(
                        "prefill-drain queue contains an invalid request")
                self._record_prefill_drain_intent(request)

    def _record_prefill_drain_intent(
            self, request: HBFServingRequest) -> None:
        """Record one pool-owned gate at the same-time barrier boundary."""

        if request.state != HBFRequestState.PREFILL_DRAIN:
            raise RuntimeError(
                "cannot record a non-draining HBF request")
        call = self.calls.get(request.request_id)
        if (
            call is None
            or call.state != OnlineHBFCallState.HBF_ACTIVE
            or call.hbf_request is not request
            or call.session_id != request.session_id
        ):
            raise RuntimeError(
                "prefill drain lost its active adapter ownership")
        prior = self._prefill_drain_request_by_session.get(
            request.session_id)
        if prior is None:
            self._prefill_drain_request_by_session[
                request.session_id] = request.request_id
        elif prior != request.request_id:
            raise RuntimeError(
                "session acquired multiple prefill-drain gates: "
                f"session={request.session_id!r}, "
                f"prior_request={prior}, "
                f"new_request={request.request_id}")

    def _publish_prefill_drain_placement(
            self, request: HBFServingRequest) -> None:
        hbf_tokens, lpddr_tokens, group_id = (
            self.lifecycle.placement_snapshot(request.session_id))
        if group_id != request.group_id:
            raise RuntimeError(
                "prefill drain changed the request replica group")
        self.pool.publish_prefill_drain_placement(
            request.request_id,
            hbf_tokens=hbf_tokens,
            lpddr_tokens=lpddr_tokens,
        )

    def _release_prefill_drain(
            self, request: HBFServingRequest, *,
            now_ns: int, job_id: Optional[int],
            fallback: bool) -> None:
        session_id = request.session_id
        if (
            self._prefill_drain_request_by_session.get(session_id)
            != request.request_id
        ):
            raise RuntimeError(
                "prefill-drain release lost session ownership")
        if session_id in self._prefill_drain_waiting_append_by_session:
            raise RuntimeError(
                "prefill-drain release retained a prior-append wait")
        if job_id is None:
            if request.request_id in (
                    self._prefill_drain_request_by_job.values()):
                raise RuntimeError(
                    "jobless prefill-drain release retained a job owner")
        elif (
            self._prefill_drain_request_by_job.get(job_id)
            != request.request_id
        ):
            raise RuntimeError(
                "prefill-drain release job identity mismatch")
        self.pool.release_prefill_drain(
            request.request_id,
            now_ns=now_ns,
            job_id=job_id,
            fallback=fallback,
        )
        if job_id is not None:
            del self._prefill_drain_request_by_job[job_id]
        del self._prefill_drain_request_by_session[session_id]

    def _handle_prefill_drain_result(
            self, request: HBFServingRequest,
            result: ActivePrefillDrainResult, *,
            now_ns: int) -> None:
        """Bind one explicit lifecycle outcome to the pool decode gate."""

        if result.total_tokens != request.input_tokens:
            raise RuntimeError(
                "active prefill drain published an unexpected total")
        self._publish_prefill_drain_placement(request)
        status = result.status
        if status == ActivePrefillDrainStatus.STARTED:
            job = result.job
            if (
                job is None
                or result.append_tokens <= 0
                or result.blocking_append_job_ids
                or job.session_id != request.session_id
            ):
                raise RuntimeError(
                    "started prefill drain returned an invalid append job")
            if (
                job.job_id in self._prefill_drain_request_by_job
                or request.request_id in (
                    self._prefill_drain_request_by_job.values())
            ):
                raise RuntimeError(
                    "prefill drain acquired duplicate append ownership")
            self.pool.bind_prefill_drain_job(
                request.request_id,
                job_id=job.job_id,
                logical_tokens=result.append_tokens,
            )
            self._prefill_drain_request_by_job[
                job.job_id] = request.request_id
            return
        if result.job is not None:
            raise RuntimeError(
                "non-started prefill drain returned an append job")
        if status == ActivePrefillDrainStatus.WAIT_EXISTING_APPEND:
            blockers = result.blocking_append_job_ids
            if not blockers or len(blockers) != len(set(blockers)):
                raise RuntimeError(
                    "prefill drain returned an invalid prior-append wait")
            if (
                request.prefill_drain_job_id is not None
                or request.session_id in (
                    self._prefill_drain_waiting_append_by_session)
                or request.request_id in (
                    self._prefill_drain_request_by_job.values())
            ):
                raise RuntimeError(
                    "prefill drain prior-append wait has duplicate "
                    "ownership")
            self._prefill_drain_waiting_append_by_session[
                request.session_id] = tuple(blockers)
            return
        if result.blocking_append_job_ids:
            raise RuntimeError(
                "non-waiting prefill drain returned blocking jobs")
        if status == ActivePrefillDrainStatus.SATISFIED:
            if result.append_tokens:
                raise RuntimeError(
                    "satisfied prefill drain retained append work")
            self._release_prefill_drain(
                request,
                now_ns=now_ns,
                job_id=None,
                fallback=False,
            )
            return
        if status == ActivePrefillDrainStatus.CAPACITY_FALLBACK:
            self._release_prefill_drain(
                request,
                now_ns=now_ns,
                job_id=None,
                fallback=True,
            )
            return
        raise RuntimeError(
            f"unknown active prefill drain status {status!r}")

    def _start_or_retry_prefill_drain(
            self, request: HBFServingRequest, *,
            now_ns: int) -> None:
        if self.pool.prefill_drain_tail_tokens is None:
            raise RuntimeError(
                "prefill-drain gate exists without an enabled policy")
        if (
            self._prefill_drain_request_by_session.get(
                request.session_id)
            != request.request_id
        ):
            raise RuntimeError(
                "prefill-drain start lost session ownership")
        result = self.lifecycle.start_active_prefill_drain(
            request.session_id,
            request_id=request.request_id,
            now_ns=now_ns,
            total_tokens=request.input_tokens,
            tail_tokens=self.pool.prefill_drain_tail_tokens,
        )
        self._handle_prefill_drain_result(
            request, result, now_ns=now_ns)

    def _handle_lifecycle_append_callback(
            self, job: AppendJob, *,
            now_ns: int) -> None:
        """Release or retry an exact gate after one append callback."""

        request_id = self._prefill_drain_request_by_job.get(
            job.job_id)
        waiting_jobs = (
            self._prefill_drain_waiting_append_by_session.get(
                job.session_id)
        )
        if request_id is not None and waiting_jobs is not None:
            raise RuntimeError(
                "prefill drain owns both an active job and a prior wait")
        if request_id is not None:
            request = self.pool.requests.get(request_id)
            if (
                request is None
                or request.session_id != job.session_id
                or request.prefill_drain_job_id != job.job_id
            ):
                raise RuntimeError(
                    "active prefill-drain callback identity mismatch")
            tail_tokens = self.pool.prefill_drain_tail_tokens
            if tail_tokens is None:
                raise RuntimeError(
                    "active drain callback lost its configured tail")
            if request.active_lpddr_tokens <= tail_tokens:
                self._release_prefill_drain(
                    request,
                    now_ns=now_ns,
                    job_id=job.job_id,
                    fallback=False,
                )
            else:
                self.pool.clear_prefill_drain_job(
                    request.request_id,
                    job_id=job.job_id,
                )
                del self._prefill_drain_request_by_job[job.job_id]
                self._start_or_retry_prefill_drain(
                    request, now_ns=now_ns)
            return
        if waiting_jobs is None:
            return
        if job.job_id not in waiting_jobs:
            raise RuntimeError(
                "prior-append callback identity mismatch for waiting "
                f"prefill drain: session={job.session_id!r}, "
                f"job={job.job_id}, expected={waiting_jobs}")
        request_id = self._prefill_drain_request_by_session.get(
            job.session_id)
        if request_id is None:
            raise RuntimeError(
                "waiting prefill drain lost session ownership")
        request = self.pool.requests.get(request_id)
        if (
            request is None
            or request.session_id != job.session_id
            or not request.prefill_drain_claimed
            or request.prefill_drain_job_id is not None
        ):
            raise RuntimeError(
                "waiting prefill drain lost pool ownership")
        del self._prefill_drain_waiting_append_by_session[
            job.session_id]
        self._start_or_retry_prefill_drain(
            request, now_ns=now_ns)

    def _advance_to(self, now_ns: int) -> int:
        now = _integer("now_ns", now_ns)
        if now < self.current_ns:
            raise ValueError(
                f"adapter time cannot move backwards: "
                f"current={self.current_ns}, requested={now}")
        self.lifecycle.advance(now)
        self._refresh_live_hbf_placements()
        self.pool.advance(now, defer_schedule=True)
        self.current_ns = now
        return now

    def _parse_raw(
            self, row: Mapping[str, Any],
            now_ns: int) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            raise TypeError("raw Router request must be a mapping")
        request_id = _raw_integer(row, "index")
        session_value = row.get("session_id")
        if not isinstance(session_value, str) or not session_value:
            raise ValueError(
                "full-model HBF routing requires a non-empty session_id")
        call_index = _raw_integer(
            row, "sub_request_index", default=0)
        trace_arrival_ns = _raw_integer(row, "arrival_time_ns")
        if trace_arrival_ns > now_ns:
            raise ValueError(
                "raw Router request was offered before its arrival")
        input_tokens = _raw_integer(
            row, "input_toks", minimum=1)
        output_target = _raw_integer(
            row, "output_toks", minimum=1)
        output_tokens = output_target - input_tokens
        if output_tokens <= 0:
            raise ValueError(
                "Router output_toks must equal input plus a positive "
                "requested output")
        if input_tokens + output_tokens - 1 > 1_010_000:
            raise ValueError(
                "request exceeds the 1,010,000-token contract")
        prefix_reuse = _raw_integer(
            row, "prefix_reuse_toks", default=0)
        if prefix_reuse > input_tokens:
            raise ValueError("prefix reuse exceeds request input")
        has_successor = row.get("wakekv_has_successor", False)
        if not isinstance(has_successor, bool):
            raise ValueError(
                "wakekv_has_successor must be a boolean")
        return {
            "request_id": request_id,
            "session_id": session_value,
            "call_index": call_index,
            "trace_arrival_ns": trace_arrival_ns,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "prefix_reuse_tokens": prefix_reuse,
            "has_successor": has_successor,
        }

    @staticmethod
    def _execution_for_route(
            call_index: int,
            route: ResumeRoute) -> OnlineHBFExecution:
        if call_index == 0:
            return OnlineHBFExecution.GPU_FIRST_TURN
        if route.execution == ResumeExecution.HBF:
            return OnlineHBFExecution.HBF_READY
        if route.execution == ResumeExecution.GPU_RECOMPUTE:
            return OnlineHBFExecution.GPU_RECOMPUTE
        if route.migration_inflight:
            return OnlineHBFExecution.GPU_MIGRATION_INFLIGHT
        if route.reason == "hbf_capacity_unavailable_gpu_retained":
            return OnlineHBFExecution.GPU_CAPACITY_FALLBACK
        return OnlineHBFExecution.GPU_OWNED

    def _emit_gpu_hbm_event(
            self, *, kind: GPUHBMEventKind,
            call: OnlineHBFCall, time_ns: int,
            token_count: int, reason: str,
            gpu_instance_id: Optional[int] = None,
    ) -> GPUHBMOwnershipEvent:
        instance_id = (
            call.gpu_instance_id
            if gpu_instance_id is None else gpu_instance_id
        )
        if instance_id is None:
            raise RuntimeError(
                "GPU HBM ownership event lacks a native instance")
        instance_id = _integer(
            "gpu_instance_id", instance_id)
        accounted_tokens = int(math.ceil(
            token_count / self.gpu_block_size_tokens
        ) * self.gpu_block_size_tokens)
        event = GPUHBMOwnershipEvent(
            kind=kind,
            session_id=call.session_id,
            request_id=call.request_id,
            gpu_instance_id=instance_id,
            time_ns=time_ns,
            token_count=token_count,
            accounted_tokens_per_rank=accounted_tokens,
            logical_bytes=(
                token_count * self.lifecycle.kv_bytes_per_token),
            per_rank_bytes=(
                accounted_tokens
                * self.gpu_kv_bytes_per_token_per_rank
            ),
            reason=reason,
        )
        self._gpu_hbm_events.append(event)
        if kind == GPUHBMEventKind.TURN_RETAIN:
            self.metrics.gpu_hbm_turn_retain_events += 1
        elif kind == GPUHBMEventKind.RESUME_CLAIM:
            self.metrics.gpu_hbm_resume_claim_events += 1
        elif kind == GPUHBMEventKind.MIGRATION_RELEASE:
            self.metrics.gpu_hbm_migration_release_events += 1
        else:
            self.metrics.gpu_hbm_idle_release_events += 1
        return event

    def reclaim_gpu_ready_for_hbm_pressure(
            self, *, gpu_instance_id: int,
            now_ns: int) -> Optional[dict[str, Any]]:
        """Drop the oldest safe GPU_READY copy on one finite GPU owner.

        This is a progress path for a P/D decode reservation head. It never
        touches active calls or sessions with a pending migration/append
        callback. The next resume of the selected lineage therefore follows
        the lifecycle's explicit EVICTED-to-GPU_RECOMPUTE route.
        """

        now = self._advance_to(now_ns)
        instance_id = _integer(
            "gpu_instance_id", gpu_instance_id)
        candidates = []
        calls_by_session: dict[str, OnlineHBFCall] = {}
        for session_id, owner_instance_id in (
                self._gpu_owner_instance_by_session.items()):
            if owner_instance_id != instance_id:
                continue
            if not self.lifecycle.gpu_ready_pressure_reclaimable(
                    session_id):
                continue
            request_id = self._last_request_id(session_id)
            call = self.calls[request_id]
            if (
                call.state != OnlineHBFCallState.COMPLETE
                or not call.has_successor
                or call.successor_censored
                or call.completion_ns is None
                or call.gpu_instance_id != instance_id
                or session_id in self._active_request_by_session
            ):
                raise RuntimeError(
                    "GPU-ready pressure candidate has inconsistent adapter "
                    f"ownership: session={session_id!r}")
            candidates.append(session_id)
            calls_by_session[session_id] = call
        eviction = (
            self.lifecycle.evict_oldest_gpu_ready_for_hbm_pressure(
                candidates, now_ns=now)
        )
        if eviction is None:
            return None
        call = calls_by_session[eviction.session_id]
        event = self._emit_gpu_hbm_event(
            kind=GPUHBMEventKind.IDLE_RELEASE,
            call=call,
            gpu_instance_id=instance_id,
            time_ns=now,
            token_count=eviction.token_count,
            reason="pd_decode_hbm_pressure_reclaim",
        )
        del self._gpu_owner_instance_by_session[eviction.session_id]
        audit = {
            **asdict(eviction),
            "gpu_instance_id": instance_id,
            "owner_request_id": call.request_id,
            "accounted_tokens_per_rank": (
                event.accounted_tokens_per_rank),
            "per_rank_bytes": event.per_rank_bytes,
            "reason": event.reason,
        }
        self._gpu_ready_hbm_pressure_reclaim_audits.append(audit)
        self.metrics.gpu_ready_hbm_pressure_reclaims += 1
        self.metrics.gpu_ready_hbm_pressure_reclaimed_logical_bytes += (
            eviction.logical_bytes)
        self.metrics.gpu_ready_hbm_pressure_reclaimed_per_rank_bytes += (
            event.per_rank_bytes)
        if self.validate_every_event:
            self.assert_invariants()
        return dict(audit)

    def offer_raw_request(
            self, row: Mapping[str, Any], *,
            now_ns: int) -> OnlineHBFRouteDecision:
        """Classify one due Router row without launching an HBF batch.

        HBF decisions are staged so all co-timed Router rows can enter one
        continuous-batching scheduling pass.  Call
        :meth:`flush_admissions` after the Router finishes that arrival pass.
        """

        if self._pending_hbf_completion_by_request:
            raise RuntimeError(
                "cannot offer a Router row before deferred HBF "
                "completions reach the same-time barrier")
        now = self._advance_to(now_ns)
        values = self._parse_raw(row, now)
        request_id = values["request_id"]
        session_id = values["session_id"]
        call_index = values["call_index"]
        if request_id in self.calls:
            raise ValueError(f"duplicate request_id={request_id}")
        if session_id in self._ended_sessions:
            raise ValueError(
                f"session {session_id!r} already ended")
        active_id = self._active_request_by_session.get(session_id)
        if active_id is not None:
            raise ValueError(
                f"session {session_id!r} already has active "
                f"request {active_id}")
        expected_index = (
            self._last_call_index_by_session.get(session_id, -1) + 1)
        if call_index != expected_index:
            raise ValueError(
                "session calls must be offered in contiguous order: "
                f"session={session_id!r}, got={call_index}, "
                f"expected={expected_index}")
        if call_index == 0:
            if values["prefix_reuse_tokens"]:
                raise ValueError(
                    "first session turn cannot reuse session KV")
            self.lifecycle.register_session(
                session_id, now_ns=now)
        elif session_id not in self.lifecycle.sessions:
            raise RuntimeError(
                "resume has no lifecycle session")

        placement_before = self.lifecycle.sessions[session_id]
        residency_at_return = (
            None
            if call_index == 0
            else placement_before.state.value
        )
        retained_before = placement_before.gpu_retained_bytes
        retained_owner = self._gpu_owner_instance_by_session.get(
            session_id)
        if bool(retained_before) != (retained_owner is not None):
            raise RuntimeError(
                "GPU retained-byte lineage lost its owning instance")
        operational_reuse = min(
            values["prefix_reuse_tokens"],
            placement_before.total_tokens,
            values["input_tokens"],
        )
        # A native GPU prefill must execute at least the final prompt token
        # that produces output token one.  HBF-ready execution has its own
        # first-decode path and may consume the complete cached input.
        route_reuse = (
            operational_reuse
            if placement_before.state == PlacementState.HBF_READY
            else (
                0
                if (
                    call_index > 0
                    and self.gpu_resume_mode == "recompute"
                )
                else min(
                    operational_reuse,
                    max(0, values["input_tokens"] - 1),
                )
            )
        )
        route = self.lifecycle.route_resume(
            session_id,
            now_ns=now,
            request_id=request_id,
            prefix_reuse_tokens=route_reuse,
            input_tokens=values["input_tokens"],
            lpddr_growth_tokens=(
                values["input_tokens"]
                - route_reuse
                + values["output_tokens"]
                - 1
            ),
        )
        execution = self._execution_for_route(
            call_index, route)
        if (
            call_index > 0
            and execution != OnlineHBFExecution.HBF_READY
            and self.gpu_resume_mode == "recompute"
        ):
            execution = OnlineHBFExecution.GPU_RECOMPUTE
            route = ResumeRoute(
                execution=route.execution,
                session_id=route.session_id,
                group_id=route.group_id,
                hbf_tokens=route.hbf_tokens,
                lpddr_tokens=route.lpddr_tokens,
                migration_inflight=route.migration_inflight,
                reason=f"gpu_resume_recompute:{route.reason}",
            )
        gpu_prefix_reuse = (
            0
            if execution in {
                OnlineHBFExecution.GPU_RECOMPUTE,
                OnlineHBFExecution.HBF_READY,
            }
            else route_reuse
        )
        hbf_request = None
        state = OnlineHBFCallState.GPU_ACTIVE
        required_gpu_instance_id = None
        if execution == OnlineHBFExecution.HBF_READY:
            if route.group_id is None:
                raise RuntimeError("HBF route lacks a replica group")
            hbf_request = HBFServingRequest(
                request_id=request_id,
                session_id=session_id,
                # Pool arrival is the physical admission boundary.  The
                # original trace release remains on OnlineHBFCall for TTFT.
                arrival_ns=now,
                input_tokens=values["input_tokens"],
                output_tokens=values["output_tokens"],
                hbf_prefix_tokens=route.hbf_tokens,
                lpddr_prefix_tokens=route.lpddr_tokens,
                group_id=route.group_id,
            )
            self._staged_hbf_by_time.setdefault(
                now, []).append(hbf_request)
            state = OnlineHBFCallState.HBF_STAGED

        call = OnlineHBFCall(
            request_id=request_id,
            session_id=session_id,
            call_index=call_index,
            trace_arrival_ns=values["trace_arrival_ns"],
            input_tokens=values["input_tokens"],
            output_tokens=values["output_tokens"],
            requested_prefix_reuse_tokens=(
                values["prefix_reuse_tokens"]),
            has_successor=values["has_successor"],
            admission_ns=now,
            operational_prefix_reuse_tokens=operational_reuse,
            execution=execution,
            route_reason=route.reason,
            migration_inflight=route.migration_inflight,
            state=state,
            gpu_prefix_reuse_tokens=gpu_prefix_reuse,
            gpu_instance_id=None,
            residency_at_return=residency_at_return,
            kv_source=(
                None
                if call_index == 0
                else (
                    "hbf"
                    if execution == OnlineHBFExecution.HBF_READY
                    else (
                        "hbm"
                        if gpu_prefix_reuse > 0
                        else "dropped"
                    )
                )
            ),
            raw_metadata=self._copy_raw_metadata(row),
            hbf_request=hbf_request,
        )
        self.calls[request_id] = call
        self._active_request_by_session[session_id] = request_id
        self._last_call_index_by_session[session_id] = call_index
        self.metrics.offered_requests += 1
        self._execution_counts[execution.value] += 1
        if execution == OnlineHBFExecution.HBF_READY:
            self.metrics.hbf_requests += 1
        else:
            self.metrics.gpu_requests += 1
            retained_after = (
                self.lifecycle.sessions[
                    session_id].gpu_retained_bytes)
            if retained_before:
                assert retained_owner is not None
                if retained_after:
                    required_gpu_instance_id = retained_owner
                    call.gpu_instance_id = retained_owner
                    self._emit_gpu_hbm_event(
                        kind=GPUHBMEventKind.RESUME_CLAIM,
                        call=call,
                        time_ns=now,
                        token_count=gpu_prefix_reuse,
                        reason=route.reason,
                    )
                else:
                    self._emit_gpu_hbm_event(
                        kind=GPUHBMEventKind.IDLE_RELEASE,
                        call=call,
                        gpu_instance_id=retained_owner,
                        time_ns=now,
                        token_count=0,
                        reason="gpu_lineage_trimmed_to_zero",
                    )
                    del self._gpu_owner_instance_by_session[
                        session_id]

        decision = OnlineHBFRouteDecision(
            request_id=request_id,
            session_id=session_id,
            call_index=call_index,
            execution=execution,
            route_reason=route.reason,
            operational_prefix_reuse_tokens=operational_reuse,
            gpu_prefix_reuse_tokens=gpu_prefix_reuse,
            required_gpu_instance_id=required_gpu_instance_id,
            migration_inflight=route.migration_inflight,
            hbf_request=hbf_request,
        )
        if self.validate_every_event:
            self.assert_invariants()
        return decision

    def offer_raw_requests(
            self, rows: Iterable[Mapping[str, Any]], *,
            now_ns: int,
            flush: bool = True) -> tuple[OnlineHBFRouteDecision, ...]:
        """Offer a co-timed arrival set and optionally launch one HBF batch."""

        if not isinstance(flush, bool):
            raise ValueError("flush must be a boolean")
        decisions = tuple(
            self.offer_raw_request(row, now_ns=now_ns)
            for row in rows
        )
        if flush:
            self.flush_admissions(now_ns)
        return decisions

    def flush_admissions(self, now_ns: int) -> int:
        """Submit admissions and claim drains after the same-time barrier."""

        if self._pending_hbf_completion_by_request:
            raise RuntimeError(
                "cannot flush HBF admissions before deferred HBF "
                "completions reach the same-time barrier")
        now = self._advance_to(now_ns)
        stale = sorted(
            timestamp for timestamp in self._staged_hbf_by_time
            if timestamp < now
        )
        if stale:
            raise RuntimeError(
                "HBF admissions were not flushed at their exact "
                f"timestamp: {stale}")
        values = self._staged_hbf_by_time.pop(now, [])
        if values:
            values.sort(key=lambda request: request.request_id)
            self.pool.submit_many(
                values,
                now_ns=now,
                defer_schedule=True,
            )
            for request in values:
                call = self.calls[request.request_id]
                if call.state != OnlineHBFCallState.HBF_STAGED:
                    raise RuntimeError(
                        "staged HBF admission lost call ownership")
                call.state = OnlineHBFCallState.HBF_ACTIVE
        self._record_prefill_drain_intents()
        claimed = sorted(
            self.pool.claim_prefill_drain_requests(),
            key=lambda request: request.request_id,
        )
        for request in claimed:
            if (
                self._prefill_drain_request_by_session.get(
                    request.session_id)
                != request.request_id
            ):
                raise RuntimeError(
                    "claimed prefill drain lacks a recorded barrier intent")
            self._start_or_retry_prefill_drain(
                request, now_ns=now)
        # A deferred callback or an immediate satisfied/capacity outcome may
        # have left decode work ready. Scheduling occurs only at this barrier.
        self.pool.flush_scheduling(now)
        if self.validate_every_event:
            self.assert_invariants()
        return len(values)

    @staticmethod
    def _request_identity(request_or_id: object) -> int:
        raw_id = (
            getattr(request_or_id, "id")
            if hasattr(request_or_id, "id")
            else request_or_id
        )
        return _integer("request_id", raw_id)

    @staticmethod
    def _native_instance_id(
            request_or_id: object,
            explicit_instance_id: Optional[int]) -> Optional[int]:
        observed = (
            getattr(request_or_id, "instance_id")
            if hasattr(request_or_id, "instance_id")
            else None
        )
        if observed is not None:
            observed = _integer(
                "request.instance_id", observed)
        if explicit_instance_id is not None:
            explicit_instance_id = _integer(
                "gpu_instance_id", explicit_instance_id)
        if (
            observed is not None
            and explicit_instance_id is not None
            and observed != explicit_instance_id
        ):
            raise RuntimeError(
                "native request and explicit GPU instance differ")
        return (
            observed
            if observed is not None else explicit_instance_id
        )

    def bind_native_gpu_request(
            self, request_or_id: object, *,
            gpu_instance_id: Optional[int] = None) -> int:
        """Bind a GPU decision to the exact native Scheduler instance."""

        request_id = self._request_identity(request_or_id)
        call = self.calls.get(request_id)
        if call is None:
            raise KeyError(f"unknown GPU request_id={request_id}")
        if (
            call.state != OnlineHBFCallState.GPU_ACTIVE
            or call.execution == OnlineHBFExecution.HBF_READY
        ):
            raise RuntimeError(
                "only an active GPU call may bind a native instance")
        instance_id = self._native_instance_id(
            request_or_id, gpu_instance_id)
        if instance_id is None:
            instance_id = call.gpu_instance_id
        if instance_id is None:
            raise ValueError(
                "native GPU binding requires request.instance_id or "
                "gpu_instance_id")
        if (
            call.gpu_instance_id is not None
            and call.gpu_instance_id != instance_id
        ):
            raise RuntimeError(
                "native GPU route changed retained-KV ownership: "
                f"required={call.gpu_instance_id}, actual={instance_id}")
        call.gpu_instance_id = instance_id
        return instance_id

    def decorate_gpu_metadata(
            self, decision_or_id: OnlineHBFRouteDecision | int,
            row: Mapping[str, Any]) -> dict[str, Any]:
        """Return native-GPU metadata with exact reuse and owner binding.

        The caller must still force scheduler selection to
        ``hbf_gpu_required_instance_id`` when it is non-``None``, then call
        :meth:`bind_native_gpu_request` after constructing the native
        ``Request``.  This method never guesses an instance for a first turn
        or recomputation.
        """

        if not isinstance(row, Mapping):
            raise TypeError("raw Router request must be a mapping")
        request_id = (
            decision_or_id.request_id
            if isinstance(decision_or_id, OnlineHBFRouteDecision)
            else _integer("request_id", decision_or_id)
        )
        call = self.calls.get(request_id)
        if call is None:
            raise KeyError(f"unknown request_id={request_id}")
        if call.execution == OnlineHBFExecution.HBF_READY:
            raise RuntimeError(
                "HBF-diverted request has no native GPU metadata")
        result = dict(row)
        result["prefix_reuse_toks"] = call.gpu_prefix_reuse_tokens
        result["agentic_kv_hit_tokens"] = (
            call.gpu_prefix_reuse_tokens)
        result["agentic_kv_recompute_tokens"] = (
            call.operational_prefix_reuse_tokens
            if call.execution == OnlineHBFExecution.GPU_RECOMPUTE
            else 0
        )
        result["agentic_kv_owner_instance_id"] = (
            call.gpu_instance_id
            if call.gpu_prefix_reuse_tokens else None
        )
        result["agentic_kv_residency_at_return"] = (
            call.residency_at_return)
        result["agentic_kv_source"] = call.kv_source
        result["hbf_gpu_required_instance_id"] = call.gpu_instance_id
        result["hbf_online_execution"] = call.execution.value
        result["hbf_online_route_reason"] = call.route_reason
        return result

    def complete_native_gpu_request(
            self, request_or_id: object, *,
            completion_ns: int,
            materialized_tokens: Optional[int] = None,
            gpu_instance_id: Optional[int] = None,
            publish_successor: bool = True,
    ) -> Optional[MigrationJob]:
        """Apply a native GPU completion after same-time classification."""

        now = self._advance_to(completion_ns)
        if not isinstance(publish_successor, bool):
            raise ValueError("publish_successor must be a boolean")
        request_id = self._request_identity(request_or_id)
        call = self.calls.get(request_id)
        if call is None:
            raise KeyError(f"unknown GPU request_id={request_id}")
        if call.state != OnlineHBFCallState.GPU_ACTIVE:
            raise RuntimeError(
                f"request {request_id} is not GPU-active")
        if call.execution == OnlineHBFExecution.HBF_READY:
            raise RuntimeError("HBF-diverted request completed on GPU")
        instance_id = self.bind_native_gpu_request(
            request_or_id,
            gpu_instance_id=gpu_instance_id,
        )
        if (
            hasattr(request_or_id, "session_id")
            and getattr(request_or_id, "session_id") not in {
                None, call.session_id
            }
        ):
            raise RuntimeError(
                "native GPU completion changed session identity")
        if materialized_tokens is None:
            if not hasattr(request_or_id, "num_computed_tokens"):
                raise ValueError(
                    "materialized_tokens is required for an integer "
                    "request ID")
            materialized_tokens = getattr(
                request_or_id, "num_computed_tokens")
        tokens = _integer(
            "materialized_tokens", materialized_tokens,
            minimum=1)
        if tokens != call.final_materialized_tokens:
            raise RuntimeError(
                "native GPU completion has the wrong materialized "
                f"context: request={request_id}, expected="
                f"{call.final_materialized_tokens}, actual={tokens}")

        job = self.lifecycle.complete_gpu_turn(
            call.session_id,
            now_ns=now,
            total_tokens=tokens,
            has_successor=(
                call.has_successor and publish_successor),
        )
        call.completion_ns = now
        call.state = OnlineHBFCallState.COMPLETE
        del self._active_request_by_session[call.session_id]
        if call.has_successor and not publish_successor:
            call.successor_censored = True
            self._gpu_owner_instance_by_session.pop(
                call.session_id, None)
            self._ended_sessions.add(call.session_id)
            self.metrics.censored_successors += 1
        elif not call.has_successor:
            self._gpu_owner_instance_by_session.pop(
                call.session_id, None)
            self._ended_sessions.add(call.session_id)
        else:
            self._gpu_owner_instance_by_session[
                call.session_id] = instance_id
            self._emit_gpu_hbm_event(
                kind=GPUHBMEventKind.TURN_RETAIN,
                call=call,
                time_ns=now,
                token_count=tokens,
                reason=(
                    "migration_started"
                    if job is not None
                    else "hbf_capacity_unavailable_gpu_retained"
                ),
            )
        self.metrics.gpu_completions += 1
        if self.validate_every_event:
            self.assert_invariants()
        return job

    def _router_proxy(
            self, call: OnlineHBFCall,
            request: HBFServingRequest) -> RouterCompletionProxy:
        if (
            request.completion_ns is None
            or request.first_token_ns is None
            or request.first_scheduled_ns is None
        ):
            raise RuntimeError(
                "completed HBF request lacks latency timestamps")
        if len(request.token_completion_ns) != call.output_tokens:
            raise RuntimeError(
                "completed HBF request has incomplete token timestamps")
        completion_ns = request.completion_ns
        latency = completion_ns - call.trace_arrival_ns
        ttft = request.first_token_ns - call.trace_arrival_ns
        tpot = (
            0.0
            if call.output_tokens < 2
            else (
                (completion_ns - request.first_token_ns)
                // (call.output_tokens - 1)
            )
        )
        raw = call.raw_metadata
        return RouterCompletionProxy(
            id=call.request_id,
            session_id=call.session_id,
            sub_request_index=call.call_index,
            arrival=call.trace_arrival_ns,
            end_time=completion_ns,
            original_input=call.input_tokens,
            input=call.input_tokens,
            output=call.input_tokens + call.output_tokens,
            num_computed_tokens=call.final_materialized_tokens,
            generated_tokens=call.output_tokens,
            ttft=ttft,
            tpot=tpot,
            latency=latency,
            admission_ns=call.admission_ns,
            first_schedule_time_ns=request.first_scheduled_ns,
            hbf_prefix_tokens=request.admitted_hbf_prefix_tokens,
            lpddr_prefix_tokens=request.admitted_lpddr_prefix_tokens,
            token_completion_ns=tuple(request.token_completion_ns),
            hbf_online_execution=call.execution.value,
            hbf_online_route_reason=call.route_reason,
            source_session_id=raw.get("source_session_id"),
            session_template_index=raw.get(
                "session_template_index"),
            session_epoch=int(raw.get("session_epoch") or 0),
            wakekv_has_successor=call.has_successor,
            raw_metadata=raw,
        )

    def _finalize_hbf_pool_completion(
            self, call: OnlineHBFCall,
            request: HBFServingRequest,
            proxy: RouterCompletionProxy, *,
            publish_successor: bool) -> Optional[AppendJob]:
        if not isinstance(publish_successor, bool):
            raise ValueError("publish_successor must be a boolean")
        effective_successor = (
            call.has_successor and publish_successor)
        job = self.lifecycle.complete_hbf_turn(
            call.session_id,
            now_ns=int(request.completion_ns),
            total_tokens=call.final_materialized_tokens,
            has_successor=effective_successor,
        )
        call.completion_ns = int(request.completion_ns)
        call.state = OnlineHBFCallState.COMPLETE
        del self._active_request_by_session[call.session_id]
        if call.has_successor and not publish_successor:
            call.successor_censored = True
            self._ended_sessions.add(call.session_id)
            self.metrics.censored_successors += 1
        elif not call.has_successor:
            self._ended_sessions.add(call.session_id)
        self.metrics.hbf_completions += 1
        self.metrics.router_completion_proxies += 1
        return job

    def _consume_pool_completions(
            self, now_ns: int, *,
            defer_turn_finalization: bool = False,
    ) -> tuple[RouterCompletionProxy, ...]:
        if not isinstance(defer_turn_finalization, bool):
            raise ValueError(
                "defer_turn_finalization must be a boolean")
        proxies = []
        for request in self.pool.pop_completed():
            call = self.calls.get(request.request_id)
            if call is None:
                raise RuntimeError(
                    "HBF completion has no adapter call")
            if call.state != OnlineHBFCallState.HBF_ACTIVE:
                raise RuntimeError(
                    "HBF completion has no active adapter ownership")
            if request.completion_ns != now_ns:
                raise RuntimeError(
                    "HBF request and ASTRA callback timestamps differ")
            proxy = self._router_proxy(call, request)
            self._completed_router_proxies.append(proxy)
            proxies.append(proxy)
            if defer_turn_finalization:
                if request.request_id in (
                        self._pending_hbf_completion_by_request):
                    raise RuntimeError(
                        "HBF request completion was deferred twice")
                self._pending_hbf_completion_by_request[
                    request.request_id] = (request, proxy)
            else:
                self._finalize_hbf_pool_completion(
                    call, request, proxy,
                    publish_successor=True,
                )
        return tuple(proxies)

    def finalize_deferred_hbf_completion(
            self, request_or_id: object, *,
            completion_ns: int,
            publish_successor: bool = True,
    ) -> Optional[AppendJob]:
        """Commit one HBF turn after the global same-time barrier."""

        now = self._advance_to(completion_ns)
        request_id = self._request_identity(request_or_id)
        pending = self._pending_hbf_completion_by_request.get(
            request_id)
        if pending is None:
            raise KeyError(
                f"request {request_id} has no deferred HBF completion")
        request, proxy = pending
        if request.completion_ns != now or proxy.end_time != now:
            raise RuntimeError(
                "deferred HBF completion timestamp changed before commit")
        call = self.calls.get(request_id)
        if (
            call is None
            or call.state != OnlineHBFCallState.HBF_ACTIVE
        ):
            raise RuntimeError(
                "deferred HBF completion lost active call ownership")
        job = self._finalize_hbf_pool_completion(
            call, request, proxy,
            publish_successor=publish_successor,
        )
        del self._pending_hbf_completion_by_request[request_id]
        if self.validate_every_event:
            self.assert_invariants()
        return job

    def drain_astra_dispatches(
            self) -> tuple[HBFAstraMultiplexedJob, ...]:
        """Drain lifecycle and foreground jobs through one named boundary."""

        if self._staged_hbf_by_time:
            raise RuntimeError(
                "flush HBF admissions before draining ASTRA jobs")
        jobs = self.multiplexer.drain_jobs()
        if self.validate_every_event:
            self.assert_invariants()
        return jobs

    def drain_astra_commands(self) -> tuple[str, ...]:
        return tuple(
            job.controller_command
            for job in self.drain_astra_dispatches()
        )

    def complete_astra_dispatch(
            self, *, job_id: str, arrival_ns: int,
            completion_ns: int, stage_count: int,
            defer_turn_finalization: bool = False,
    ) -> OnlineHBFAstraCompletion:
        """Route one exact ASTRA callback without scheduling its successor.

        Apply every callback at a timestamp, offer tied Router arrivals, then
        call :meth:`flush_admissions`.  This preserves completion-before-
        arrival publication while letting tied arrivals batch together.
        """

        prior_gpu_retained = {
            session_id: record.gpu_retained_bytes
            for session_id, record in self.lifecycle.sessions.items()
        }
        result = self.multiplexer.complete(
            job_id=job_id,
            arrival_ns=arrival_ns,
            completion_ns=completion_ns,
            stage_count=stage_count,
        )
        now = _integer("completion_ns", completion_ns)
        if now < self.current_ns:
            raise RuntimeError(
                "ASTRA callback moved adapter time backwards")
        self.current_ns = now
        self.metrics.astra_callbacks += 1
        proxies: tuple[RouterCompletionProxy, ...] = ()
        if result.source_name == self.lifecycle_source_name:
            self._refresh_live_hbf_placements()
            self.pool.advance(now, defer_schedule=True)
            self.metrics.astra_lifecycle_callbacks += 1
            owner_result = result.owner_result
            if isinstance(owner_result, MigrationJob):
                before = prior_gpu_retained.get(
                    owner_result.session_id, 0)
                after = self.lifecycle.sessions[
                    owner_result.session_id].gpu_retained_bytes
                if after < before:
                    gpu_owner = (
                        self._gpu_owner_instance_by_session.get(
                            owner_result.session_id)
                    )
                    if gpu_owner is None:
                        raise RuntimeError(
                            "migration publication lost GPU owner")
                    call = self.calls.get(
                        self._last_request_id(owner_result.session_id))
                    if call is None:
                        raise RuntimeError(
                            "migration publication lost its source call")
                    released_tokens = (
                        (before - after)
                        // self.lifecycle.kv_bytes_per_token
                    )
                    self._emit_gpu_hbm_event(
                        kind=GPUHBMEventKind.MIGRATION_RELEASE,
                        call=call,
                        gpu_instance_id=gpu_owner,
                        time_ns=now,
                        token_count=released_tokens,
                        reason="hbf_migration_published",
                    )
                    del self._gpu_owner_instance_by_session[
                        owner_result.session_id]
            elif isinstance(owner_result, AppendJob):
                self._handle_lifecycle_append_callback(
                    owner_result, now_ns=now)
            else:
                raise RuntimeError(
                    "lifecycle ASTRA callback returned an unknown job")
        elif result.source_name == self.pool_source_name:
            self.lifecycle.advance(now)
            self.metrics.astra_pool_callbacks += 1
            proxies = self._consume_pool_completions(
                now,
                defer_turn_finalization=defer_turn_finalization,
            )
            self._record_prefill_drain_intents()
        else:
            raise RuntimeError(
                f"unknown online HBF source {result.source_name!r}")
        if self.validate_every_event:
            self.assert_invariants()
        return OnlineHBFAstraCompletion(
            multiplexed=result,
            router_completions=proxies,
        )

    def _last_request_id(self, session_id: str) -> int:
        candidates = [
            call.request_id for call in self.calls.values()
            if call.session_id == session_id
        ]
        if not candidates:
            raise RuntimeError(
                f"session {session_id!r} has no adapter calls")
        return max(
            candidates,
            key=lambda request_id: self.calls[
                request_id].call_index,
        )

    def pop_router_completions(self) -> list[RouterCompletionProxy]:
        values = list(self._completed_router_proxies)
        self._completed_router_proxies.clear()
        return values

    def pop_gpu_hbm_events(self) -> list[GPUHBMOwnershipEvent]:
        values = list(self._gpu_hbm_events)
        self._gpu_hbm_events.clear()
        return values

    def censor_completed_successor(
            self, request_or_id: object, *,
            now_ns: int) -> Optional[RouterCompletionProxy]:
        """End a frozen non-final session without releasing its successor.

        Already-issued migration or append DAGs remain ASTRA callback
        obligations.  Their generation becomes stale and their reserved HBF
        capacity is released by the ordinary strict callback path.
        """

        now = self._advance_to(now_ns)
        request_id = self._request_identity(request_or_id)
        call = self.calls.get(request_id)
        if call is None:
            raise KeyError(f"unknown request_id={request_id}")
        if call.state != OnlineHBFCallState.COMPLETE:
            raise RuntimeError(
                "only a completed request successor may be censored")
        if not call.has_successor:
            raise ValueError(
                "terminal request has no successor to censor")
        if call.successor_censored:
            raise RuntimeError("request successor was already censored")
        placement = self.lifecycle.sessions[call.session_id]
        retained_bytes = placement.gpu_retained_bytes
        retained_owner = self._gpu_owner_instance_by_session.get(
            call.session_id)
        if retained_bytes and retained_owner is None:
            raise RuntimeError(
                "censored GPU lineage lacks an owning instance")
        self.lifecycle.end_session(call.session_id, now_ns=now)
        if retained_bytes:
            retained_tokens = (
                retained_bytes // self.lifecycle.kv_bytes_per_token)
            self._emit_gpu_hbm_event(
                kind=GPUHBMEventKind.IDLE_RELEASE,
                call=call,
                gpu_instance_id=retained_owner,
                time_ns=now,
                token_count=retained_tokens,
                reason="measurement_successor_censored",
            )
        self._gpu_owner_instance_by_session.pop(
            call.session_id, None)
        call.successor_censored = True
        self._ended_sessions.add(call.session_id)
        self.metrics.censored_successors += 1
        discarded = None
        retained_proxies = deque()
        while self._completed_router_proxies:
            proxy = self._completed_router_proxies.popleft()
            if proxy.id == request_id:
                if discarded is not None:
                    raise RuntimeError(
                        "duplicate queued Router completion proxy")
                discarded = proxy
            else:
                retained_proxies.append(proxy)
        self._completed_router_proxies = retained_proxies
        if self.validate_every_event:
            self.assert_invariants()
        return discarded

    def censor_active_native_gpu_request(
            self, request_or_id: object, *,
            now_ns: int) -> None:
        """End a dispatched GPU call that cannot cross a source cutoff.

        This path is used only after the measurement source is frozen.  A
        prefill graph may already have been dispatched, yet launching its
        not-yet-issued decode graph would create new work beyond the cutoff.
        Scheduler completion has already released the graph's active KV, so
        the adapter ends only its logical lifecycle and must not emit a
        retained-HBM event or start migration.
        """

        now = self._advance_to(now_ns)
        request_id = self._request_identity(request_or_id)
        call = self.calls.get(request_id)
        if call is None:
            raise KeyError(f"unknown GPU request_id={request_id}")
        if call.state != OnlineHBFCallState.GPU_ACTIVE:
            raise RuntimeError(
                "only an active native GPU request may be censored")
        if call.execution == OnlineHBFExecution.HBF_READY:
            raise RuntimeError("HBF-diverted request is not native GPU work")
        if (
            self._gpu_owner_instance_by_session.get(call.session_id)
            is not None
        ):
            raise RuntimeError(
                "active GPU censoring found retained idle ownership")
        self.lifecycle.end_session(call.session_id, now_ns=now)
        call.completion_ns = now
        call.successor_censored = True
        call.state = OnlineHBFCallState.COMPLETE
        del self._active_request_by_session[call.session_id]
        self._ended_sessions.add(call.session_id)
        self.metrics.censored_active_gpu_requests += 1
        if self.validate_every_event:
            self.assert_invariants()

    def validate_queued_native_gpu_request(
            self, request_or_id: object, *,
            now_ns: int) -> dict[str, object]:
        """Validate a native GPU call before Scheduler queue cancellation.

        The Router calls this before mutating finite Scheduler memory.  It
        keeps the cross-component cutoff transactional: any identity,
        ownership, or timestamp error is reported while the physical request
        is still queued.
        """

        now = _integer("now_ns", now_ns)
        if now < self.current_ns:
            raise ValueError(
                "queued GPU censoring cannot move adapter time backwards: "
                f"current={self.current_ns}, requested={now}")
        request_id = self._request_identity(request_or_id)
        call = self.calls.get(request_id)
        if call is None:
            raise KeyError(f"unknown GPU request_id={request_id}")
        if call.state != OnlineHBFCallState.GPU_ACTIVE:
            raise RuntimeError(
                "only a GPU-active call may be censored from a native "
                f"Scheduler queue: request={request_id}, state="
                f"{call.state.value}")
        if call.execution == OnlineHBFExecution.HBF_READY:
            raise RuntimeError(
                "an HBF-diverted call cannot own a native Scheduler queue")
        observed_session = getattr(request_or_id, "session_id", None)
        if (
            observed_session is not None
            and str(observed_session) != call.session_id
        ):
            raise RuntimeError(
                "queued native GPU request changed session identity: "
                f"request={request_id}, expected={call.session_id!r}, "
                f"observed={observed_session!r}")
        observed_instance = getattr(
            request_or_id, "instance_id", None)
        if (
            observed_instance is not None
            and call.gpu_instance_id is not None
            and int(observed_instance) != call.gpu_instance_id
        ):
            raise RuntimeError(
                "queued native GPU request changed its required instance: "
                f"request={request_id}, required={call.gpu_instance_id}, "
                f"observed={observed_instance}")
        active_id = self._active_request_by_session.get(call.session_id)
        if active_id != request_id:
            raise RuntimeError(
                "queued native GPU call lost active-session ownership: "
                f"session={call.session_id!r}, expected={request_id}, "
                f"observed={active_id}")
        return {
            "request_id": request_id,
            "session_id": call.session_id,
            "execution": call.execution.value,
            "gpu_instance_id": call.gpu_instance_id,
            "cutoff_time_ns": now,
        }

    def censor_queued_native_gpu_request(
            self, request_or_id: object, *,
            now_ns: int) -> dict[str, object]:
        """End one GPU call after its idle Scheduler queue was unwound.

        Unlike :meth:`censor_active_native_gpu_request`, this path may cancel
        a colocated resume whose retained prefix was already adopted by a
        queued Scheduler request.  The Scheduler caller must release that
        active HBM or CPU-swap allocation first; consequently this method
        emits no ownership event and removes the adapter's sticky-owner index.
        Accepted HBF-pool requests are intentionally outside this API and
        remain drain obligations.
        """

        audit = self.validate_queued_native_gpu_request(
            request_or_id, now_ns=now_ns)
        now = self._advance_to(now_ns)
        request_id = int(audit["request_id"])
        call = self.calls[request_id]
        retained_owner = self._gpu_owner_instance_by_session.get(
            call.session_id)
        if (
            retained_owner is not None
            and call.gpu_instance_id is not None
            and retained_owner != call.gpu_instance_id
        ):
            raise RuntimeError(
                "queued native GPU cancellation found conflicting sticky "
                f"owners: session={call.session_id!r}, adapter="
                f"{call.gpu_instance_id}, lifecycle={retained_owner}")

        self.lifecycle.end_session(call.session_id, now_ns=now)
        call.completion_ns = now
        call.successor_censored = True
        call.state = OnlineHBFCallState.COMPLETE
        del self._active_request_by_session[call.session_id]
        self._gpu_owner_instance_by_session.pop(
            call.session_id, None)
        self._ended_sessions.add(call.session_id)
        self.metrics.censored_queued_gpu_requests += 1
        result = {
            **audit,
            "retained_gpu_owner_removed": retained_owner,
        }
        if self.validate_every_event:
            self.assert_invariants()
        return result

    def has_pending_astra_dispatches(self) -> bool:
        """Return whether accepted HBF work owns a callback obligation.

        This includes source-owned pool/lifecycle work that has not yet been
        drained into a Controller command.  It deliberately excludes native
        ``GPU_ACTIVE`` calls.
        """

        return bool(
            self._prefill_drain_request_by_session
            or self._prefill_drain_request_by_job
            or self._prefill_drain_waiting_append_by_session
            or self.multiplexer.has_pending()
        )

    def has_pending_native_gpu_requests(self) -> bool:
        """Return whether an offered native GPU call is still live."""

        return any(
            call.state == OnlineHBFCallState.GPU_ACTIVE
            for call in self.calls.values()
        )

    def has_deferred_hbf_completions(self) -> bool:
        """Return whether callback-complete HBF turns await the tie barrier."""

        return bool(self._pending_hbf_completion_by_request)

    def has_pending(self) -> bool:
        """Return whether any logical, Router, GPU, or ASTRA work is live."""

        return bool(
            self._active_request_by_session
            or self._staged_hbf_by_time
            or self._completed_router_proxies
            or self._pending_hbf_completion_by_request
            or self._prefill_drain_request_by_session
            or self._prefill_drain_request_by_job
            or self._prefill_drain_waiting_append_by_session
            or self._gpu_hbm_events
            or self.pool.has_pending()
            or self.lifecycle.has_pending_external()
            or self.multiplexer.has_pending()
        )

    def next_wakeup_ns(
            self, current_ns: int, *,
            router_arrival_ns: Optional[int] = None,
            extra_candidates: Iterable[int] = (),
    ) -> Optional[int]:
        """Return the next known Python-owned event, never an ASTRA guess.

        External HBF completion time is intentionally absent.  The Router's
        next raw arrival should be supplied so a control callback can preempt
        a long-running HBF DAG.
        """

        current = _integer("current_ns", current_ns)
        candidates = []
        if router_arrival_ns is not None:
            candidates.append(_integer(
                "router_arrival_ns", router_arrival_ns))
        for value in extra_candidates:
            candidates.append(_integer(
                "extra wakeup candidate", value))
        for value in (
            self.pool.next_event_ns(),
            self.lifecycle.next_completion_ns(),
        ):
            if value is not None:
                candidates.append(int(value))
        future = [value for value in candidates if value > current]
        return min(future) if future else None

    def assert_invariants(self) -> None:
        self.lifecycle.assert_invariants()
        self.pool.assert_invariants()
        if self.lifecycle.lpddr_ledger is not self.pool.lpddr_ledger:
            raise AssertionError("online adapter lost shared LPDDR ledger")
        if (
            self.current_ns != self.lifecycle.current_ns
            or self.current_ns != self.pool.current_ns
        ):
            raise AssertionError(
                "online adapter component clocks diverged")
        staged_ids = {
            request.request_id
            for requests in self._staged_hbf_by_time.values()
            for request in requests
        }
        if len(staged_ids) != sum(
                len(requests)
                for requests in self._staged_hbf_by_time.values()):
            raise AssertionError("duplicate staged HBF request")
        gated_by_session: dict[str, HBFServingRequest] = {}
        gated_by_request: dict[int, HBFServingRequest] = {}
        for worker in self.pool.workers:
            for request_id in worker.prefill_drain:
                request = self.pool.requests[request_id]
                if request.state != HBFRequestState.PREFILL_DRAIN:
                    raise AssertionError(
                        "prefill-drain queue contains invalid state")
                if request.session_id in gated_by_session:
                    raise AssertionError(
                        "session owns multiple pool prefill-drain gates")
                gated_by_session[request.session_id] = request
                gated_by_request[request.request_id] = request
        expected_pending_drains = {
            session_id: request.request_id
            for session_id, request in gated_by_session.items()
        }
        if (
            self._prefill_drain_request_by_session
            != expected_pending_drains
        ):
            raise AssertionError(
                "adapter/pool prefill-drain gate ownership differs")
        active_lifecycle_jobs = set(
            self.lifecycle._active_prefill_drain_job_ids)
        if (
            set(self._prefill_drain_request_by_job)
            != active_lifecycle_jobs
        ):
            raise AssertionError(
                "adapter/lifecycle active-drain job ownership differs")
        if len(set(
                self._prefill_drain_request_by_job.values())) != len(
                    self._prefill_drain_request_by_job):
            raise AssertionError(
                "one request owns multiple active drain jobs")
        active_job_by_request = {
            request_id: job_id
            for job_id, request_id in (
                self._prefill_drain_request_by_job.items())
        }
        for job_id, request_id in (
                self._prefill_drain_request_by_job.items()):
            request = gated_by_request.get(request_id)
            if (
                request is None
                or request.prefill_drain_job_id != job_id
                or not request.prefill_drain_claimed
                or request.session_id in (
                    self._prefill_drain_waiting_append_by_session)
            ):
                raise AssertionError(
                    "active prefill-drain job lost pool ownership")
            record = self.lifecycle.sessions.get(request.session_id)
            if (
                record is None
                or job_id not in record.append_job_ids
            ):
                raise AssertionError(
                    "active prefill-drain job lost lifecycle ownership")
        for session_id, blocker_ids in (
                self._prefill_drain_waiting_append_by_session.items()):
            request = gated_by_session.get(session_id)
            if (
                request is None
                or not request.prefill_drain_claimed
                or request.prefill_drain_job_id is not None
                or request.request_id in active_job_by_request
                or not blocker_ids
                or blocker_ids != tuple(sorted(set(blocker_ids)))
            ):
                raise AssertionError(
                    "waiting prefill drain has invalid pool ownership")
            record = self.lifecycle.sessions.get(session_id)
            if (
                record is None
                or not set(blocker_ids) <= record.append_job_ids
            ):
                raise AssertionError(
                    "waiting prefill drain has stale append blockers")
        for request in gated_by_request.values():
            active_job_id = active_job_by_request.get(
                request.request_id)
            waiting = (
                request.session_id
                in self._prefill_drain_waiting_append_by_session
            )
            if request.prefill_drain_claimed:
                if (active_job_id is not None) == waiting:
                    raise AssertionError(
                        "claimed prefill drain must own exactly one "
                        "active-job or prior-append wait")
                if (
                    request.prefill_drain_job_id
                    != active_job_id
                ):
                    raise AssertionError(
                        "claimed prefill-drain pool/job identity differs")
            elif (
                active_job_id is not None
                or waiting
                or request.prefill_drain_job_id is not None
            ):
                raise AssertionError(
                    "unclaimed prefill drain owns lifecycle work")
        for session_id, request_id in (
                self._active_request_by_session.items()):
            call = self.calls.get(request_id)
            if (
                call is None
                or call.session_id != session_id
                or call.state == OnlineHBFCallState.COMPLETE
            ):
                raise AssertionError(
                    "active session index is inconsistent")
        for request_id, (request, proxy) in (
                self._pending_hbf_completion_by_request.items()):
            call = self.calls.get(request_id)
            if (
                call is None
                or call.state != OnlineHBFCallState.HBF_ACTIVE
                or call.hbf_request is not request
                or proxy.id != request_id
                or request.completion_ns is None
            ):
                raise AssertionError(
                    "deferred HBF completion lost call/request ownership")
        for session_id, instance_id in (
                self._gpu_owner_instance_by_session.items()):
            if instance_id < 0:
                raise AssertionError("negative GPU owner instance")
            placement = self.lifecycle.sessions.get(session_id)
            if (
                placement is None
                or placement.gpu_retained_bytes <= 0
                or placement.state not in {
                    PlacementState.GPU_ACTIVE,
                    PlacementState.GPU_READY,
                    PlacementState.MIGRATING,
                }
            ):
                raise AssertionError(
                    "GPU owner index has no retained lifecycle lineage")
        for session_id, placement in self.lifecycle.sessions.items():
            if (
                placement.gpu_retained_bytes > 0
                and session_id not in self._gpu_owner_instance_by_session
            ):
                raise AssertionError(
                    "retained lifecycle lineage lacks a GPU owner")
        for request_id, call in self.calls.items():
            active = (
                self._active_request_by_session.get(call.session_id)
                == request_id
            )
            if call.state == OnlineHBFCallState.COMPLETE:
                if active or call.completion_ns is None:
                    raise AssertionError(
                        "completed call retains active ownership")
            elif not active:
                raise AssertionError(
                    "live call is absent from active session index")
            if call.state == OnlineHBFCallState.HBF_STAGED:
                if request_id not in staged_ids:
                    raise AssertionError(
                        "HBF-staged call is absent from admission set")
            elif request_id in staged_ids:
                raise AssertionError(
                    "non-staged call remains in admission set")
            if call.execution == OnlineHBFExecution.HBF_READY:
                if call.hbf_request is None:
                    raise AssertionError(
                        "HBF call lacks a serving request")
                if (
                    call.hbf_request.cached_tokens
                    != call.operational_prefix_reuse_tokens
                ):
                    raise AssertionError(
                        "HBF request changed its admitted prefix")
            elif call.hbf_request is not None:
                raise AssertionError(
                    "GPU call owns an HBF serving request")
        for session_id in self._ended_sessions:
            placement = self.lifecycle.sessions.get(session_id)
            if (
                placement is None
                or placement.state != PlacementState.ENDED
            ):
                raise AssertionError(
                    "ended adapter session is live in lifecycle")

    @staticmethod
    def integration_contract() -> dict[str, Any]:
        """Describe the exact Router/main hooks required by this core."""

        return {
            "schema": ONLINE_HBF_ADAPTER_SCHEMA,
            "router_arrival_hook": (
                "Offer each due raw agentic row before Request construction; "
                "remove HBF decisions from the native scheduler path and "
                "flush all co-timed admissions together."
            ),
            "gpu_route_hook": (
                "For GPU_RECOMPUTE, clear physical prefix reuse; for other "
                "GPU decisions use decorate_gpu_metadata, force scheduling "
                "to hbf_gpu_required_instance_id when present, and bind the "
                "constructed Request with bind_native_gpu_request. Retained "
                "KV must never move to a guessed GPU instance."
            ),
            "pd_disaggregated_gpu_fallback": (
                "Use gpu_resume_mode='recompute' when the native GPU server "
                "is P/D-disaggregated and no D-to-P restore transport is "
                "attached. This executes a migration-inflight or capacity "
                "fallback on the ordinary P-to-D path without pretending "
                "that decode-owned KV is locally visible to prefill."
            ),
            "gpu_completion_hook": (
                "Collect every final native GPU/P-D completion at the "
                "same-time barrier, classify the measurement cutoff, then "
                "call complete_native_gpu_request with the resulting "
                "successor-publication decision."
            ),
            "router_completion_hook": (
                "Defer an HBF turn's lifecycle commit until the same-time "
                "barrier, finalize each RouterCompletionProxy exactly once, "
                "and notify Router only when its successor is publishable. "
                "Queued proxies and deferred finalizations are live drain "
                "obligations."
            ),
            "astra_hook": (
                "Drain adapter ASTRA commands through Controller auxiliary "
                "commands and route hbf_background_complete callbacks back "
                "to complete_astra_dispatch."
            ),
            "tie_order": (
                "At one timestamp collect all GPU, P/D-prefill, and HBF "
                "completions, fence new model dispatch, sort their physical "
                "and logical commits deterministically, classify source "
                "cutoff before P-to-D launch, publish allowed successors, "
                "offer tied Router arrivals, then flush admissions. Active "
                "prefill-drain intents are claimed only by that final flush; "
                "an append callback releases the decode gate with deferred "
                "scheduling, so decode starts only at a later flush."
            ),
            "gpu_hbm_hook": (
                "Pop and apply every GPUHBMOwnershipEvent to the finite-HBM "
                "owner; queued ownership events are live drain obligations "
                "because the adapter cannot mutate Scheduler Memory after "
                "add_done."
            ),
            "liveness_hook": (
                "Use has_pending_astra_dispatches for accepted HBF/ASTRA "
                "ownership, has_pending_native_gpu_requests for native "
                "Scheduler ownership, and has_pending for final logical "
                "drain termination. Include next_wakeup_ns in exact control "
                "scheduling."
            ),
            "measurement_cutoff_hook": (
                "Freeze raw Router admission first. Accepted HBF-pool work "
                "remains a drain obligation. Whenever a native Scheduler "
                "has no inflight graph, release each queued request from its "
                "finite MemoryModel and call "
                "censor_queued_native_gpu_request; repeat until no native "
                "GPU request remains."
            ),
            "exclusive_tiering": (
                "Do not run the legacy SSD/CPU agentic-KV manager on rows "
                "diverted to the full-model HBF server."
            ),
        }

    def report(self) -> dict[str, Any]:
        calls = []
        for request_id in sorted(self.calls):
            call = self.calls[request_id]
            calls.append({
                "request_id": call.request_id,
                "session_id": call.session_id,
                "call_index": call.call_index,
                "trace_arrival_ns": call.trace_arrival_ns,
                "admission_ns": call.admission_ns,
                "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens,
                "requested_prefix_reuse_tokens": (
                    call.requested_prefix_reuse_tokens),
                "operational_prefix_reuse_tokens": (
                    call.operational_prefix_reuse_tokens),
                "gpu_prefix_reuse_tokens": (
                    call.gpu_prefix_reuse_tokens),
                "gpu_instance_id": call.gpu_instance_id,
                "residency_at_return": call.residency_at_return,
                "kv_source": call.kv_source,
                "has_successor": call.has_successor,
                "execution": call.execution.value,
                "route_reason": call.route_reason,
                "migration_inflight": call.migration_inflight,
                "state": call.state.value,
                "completion_ns": call.completion_ns,
                "successor_censored": call.successor_censored,
            })
        return {
            "schema": ONLINE_HBF_ADAPTER_SCHEMA,
            "current_ns": self.current_ns,
            "gpu_hbm_accounting": {
                "tp_size": self.gpu_tp_size,
                "block_size_tokens": self.gpu_block_size_tokens,
                "kv_bytes_per_token_per_rank": (
                    self.gpu_kv_bytes_per_token_per_rank),
                "resume_mode": self.gpu_resume_mode,
            },
            "metrics": asdict(self.metrics),
            "execution_counts": dict(sorted(
                self._execution_counts.items())),
            "active_request_by_session": dict(sorted(
                self._active_request_by_session.items())),
            "gpu_owner_instance_by_session": dict(sorted(
                self._gpu_owner_instance_by_session.items())),
            "ended_sessions": sorted(self._ended_sessions),
            "staged_hbf_admission_count": sum(
                len(values)
                for values in self._staged_hbf_by_time.values()),
            "pending_router_completion_count": len(
                self._completed_router_proxies),
            "pending_hbf_turn_finalization_count": len(
                self._pending_hbf_completion_by_request),
            "pending_hbf_turn_finalization_request_ids": sorted(
                self._pending_hbf_completion_by_request),
            "pending_prefill_drain_request_by_session": dict(sorted(
                self._prefill_drain_request_by_session.items())),
            "active_prefill_drain_request_by_job": dict(sorted(
                self._prefill_drain_request_by_job.items())),
            "waiting_prefill_drain_append_jobs_by_session": {
                session_id: list(job_ids)
                for session_id, job_ids in sorted(
                    self._prefill_drain_waiting_append_by_session.items())
            },
            "pending_gpu_hbm_event_count": len(
                self._gpu_hbm_events),
            "gpu_ready_hbm_pressure_reclaim_audits": [
                dict(audit)
                for audit in self._gpu_ready_hbm_pressure_reclaim_audits
            ],
            "multiplexer": self.multiplexer.report(),
            "lifecycle": self.lifecycle.report(),
            "pool": self.pool.report(),
            "calls": calls,
            "integration_contract": self.integration_contract(),
        }


__all__ = [
    "FullModelHBFOnlineAdapter",
    "GPUHBMEventKind",
    "GPUHBMOwnershipEvent",
    "ONLINE_HBF_ADAPTER_SCHEMA",
    "OnlineHBFAstraCompletion",
    "OnlineHBFAdapterMetrics",
    "OnlineHBFCall",
    "OnlineHBFCallState",
    "OnlineHBFExecution",
    "OnlineHBFRouteDecision",
    "RouterCompletionProxy",
    "SUPPORTED_GPU_RESUME_MODES",
]

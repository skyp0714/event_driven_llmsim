"""An HBF-only cluster: N eight-card HBF servers, no GPU host.

Every turn of every session -- including the first prefill -- executes
on the HBF server the session is pinned to.  Sessions are pinned round
robin at first sight and never migrate, so there is no staging tier and
no promotion machinery; the only cross-session coupling is the shared
analytical calendar.

The module name deliberately contains "hybrid" so the steady-state
campaign's lineage seeder recognises the node's session records.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .gpu_hbf_hybrid import (
    GPUHBFHybridSystem,
    HybridCall,
    HybridCallState,
    HybridExecution,
    HybridSession,
    HybridSystemMetrics,
)
from .gpu_ssd_hbf_hybrid import SSDPromotionPolicy
from .hbf_comparison_metrics import CompletedRequest
from .hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
)
from .hbf_full_model_lifecycle import (
    FullModelHBFLifecycle,
    PerGroupCapacityLedger,
    PlacementState,
    ResourceCalendar,
    ResumeExecution,
)
from .hbf_full_model_pool import (
    FullModelHBFServingPool,
    HBFServingRequest,
    derive_lpddr_workspace_bytes,
)


@dataclass
class HBFOnlyMetrics:
    submitted_calls: int = 0
    native_first_turns: int = 0
    hbf_resumes: int = 0
    user_completed_calls: int = 0
    internal_completed_calls: int = 0
    max_pending_calls: int = 0


class HBFOnlyClusterNode:
    """Session-pinned round-robin front end over N HBF server pairs."""

    def __init__(
            self, *, repo_root: Path,
            hbf_hardware: HBFServerHardware,
            hbf_layout: str | HBFParallelLayout = "tp4",
            server_count: int = 2,
            max_num_batched_tokens: int = 8_192,
            max_num_seqs: int = 128,
            max_prefill_chunk_tokens: int = 4_096,
            band: str = "central",
            validate_every_event: bool = True) -> None:
        layout = (
            HBFParallelLayout.for_key(hbf_layout)
            if isinstance(hbf_layout, str) else hbf_layout
        )
        hbf_hardware.validate()
        layout.validate(hbf_hardware.card_count)
        if server_count < 1:
            raise ValueError("server_count must be positive")
        self.repo_root = Path(repo_root)
        self.hbf_hardware = hbf_hardware
        self.hbf_layout = layout
        self.hbf_layouts = (layout,) * server_count
        self.hbf_server_count = server_count
        self.hbf_server_id = 0
        self.hbf_execution_backend = "analytical_calendar"
        self.migration_policy = SSDPromotionPolicy.for_key("never")
        self.promotion_policy = self.migration_policy
        self.validate_every_event = validate_every_event
        self.calendar = ResourceCalendar(
            retain_reservations=validate_every_event)
        workspace = derive_lpddr_workspace_bytes(
            layout,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        )
        self.lifecycles: list[FullModelHBFLifecycle] = []
        self.pools: list[FullModelHBFServingPool] = []
        for server_index in range(server_count):
            ledger = PerGroupCapacityLedger(
                group_count=layout.replicas,
                capacity_bytes=(
                    hbf_hardware.lpddr_capacity_bytes_per_card
                    - workspace
                ),
            )
            lifecycle = FullModelHBFLifecycle(
                hardware=hbf_hardware,
                layout=layout,
                resource_calendar=self.calendar,
                lpddr_ledger=ledger,
                gpu_source_root_bandwidth_gbps=1.0,
                gpu_source_node_id=0,
                validate_every_event=validate_every_event,
                execution_backend="analytical_calendar",
                server_id=server_index,
            )
            pool = FullModelHBFServingPool(
                repo_root=self.repo_root,
                hardware=hbf_hardware,
                layout=layout,
                resource_calendar=self.calendar,
                lpddr_ledger=ledger,
                placement_resolver=lifecycle.placement_snapshot,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                max_prefill_chunk_tokens=max_prefill_chunk_tokens,
                band=band,
                validate_every_event=validate_every_event,
                retain_detailed_history=validate_every_event,
                execution_backend="analytical_calendar",
                server_id=server_index,
            )
            self.lifecycles.append(lifecycle)
            self.pools.append(pool)
        # Aliases some shared tooling expects on hybrid-shaped nodes.
        self.hbf_lifecycle = self.lifecycles[0]
        self.hbf_pool = self.pools[0]

        self.calls: dict[int, HybridCall] = {}
        self.sessions: dict[str, HybridSession] = {}
        self.metrics = HBFOnlyMetrics()
        self.current_ns = 0
        self._pending_call_ids: deque[int] = deque()
        self._user_completed_ids: deque[int] = deque()
        self._last_submitted_call_index: dict[str, int] = {}
        self._server_by_session: dict[str, int] = {}
        self._live_request_server: dict[int, int] = {}
        self._next_pin = 0
        self._next_group_by_server = [0] * server_count
        self._gap_type_by_request: dict[int, Optional[str]] = {}

    # -- campaign integration hooks ------------------------------------
    def set_gap_type(
            self, request_id: int, gap_type: Optional[str]) -> None:
        self._gap_type_by_request[request_id] = gap_type

    def set_spec_total_calls(self, session_id: str, count: int) -> None:
        del session_id, count

    def pin_server(self, session_id: str) -> int:
        server = self._server_by_session.get(session_id)
        if server is None:
            server = self._next_pin % self.hbf_server_count
            self._next_pin += 1
            self._server_by_session[session_id] = server
        return server

    def preload_resident(
            self, session_id: str, tokens: int, *, now_ns: int,
            last_access_ns: Optional[int] = None) -> int:
        server = self.pin_server(session_id)
        self.lifecycles[server].preload_session(
            session_id, tokens, now_ns=now_ns,
            last_access_ns=last_access_ns)
        return server

    # -- submission ----------------------------------------------------
    def submit_many(
            self, calls: Iterable[HybridCall], *, now_ns: int) -> None:
        values = list(calls)
        for call in values:
            call.validate()
            if call.release_ns != now_ns:
                raise ValueError(
                    "calls must be submitted at logical release")
            if call.request_id in self.calls:
                raise ValueError(
                    f"duplicate request_id={call.request_id}")
            prior = self._last_submitted_call_index.get(
                call.session_id, -1)
            if call.call_index != prior + 1:
                raise ValueError(
                    "session calls must be submitted contiguously")
        self.advance(now_ns, defer_schedule=True)
        for call in values:
            self.calls[call.request_id] = call
            self.sessions.setdefault(
                call.session_id,
                HybridSession(session_id=call.session_id),
            )
            self._last_submitted_call_index[call.session_id] = (
                call.call_index)
            self._pending_call_ids.append(call.request_id)
            self.metrics.submitted_calls += 1
        self.metrics.max_pending_calls = max(
            self.metrics.max_pending_calls,
            len(self._pending_call_ids),
        )
        self.flush_scheduling(now_ns)

    def submit(self, call: HybridCall, *, now_ns: int) -> None:
        self.submit_many((call,), now_ns=now_ns)

    def _route_call(self, call: HybridCall) -> HBFServingRequest:
        server = self.pin_server(call.session_id)
        lifecycle = self.lifecycles[server]
        placement = lifecycle.sessions.get(call.session_id)
        if placement is None:
            group_id = (
                self._next_group_by_server[server]
                % self.hbf_layout.replicas)
            self._next_group_by_server[server] += 1
            lifecycle.begin_native_hbf_turn(
                call.session_id,
                group_id=group_id,
                now_ns=self.current_ns,
                request_id=call.request_id,
                lpddr_growth_tokens=(
                    call.input_tokens + call.output_tokens - 1),
            )
            call.execution = HybridExecution.HBF_READY
            call.operational_reuse_tokens = 0
            self.metrics.native_first_turns += 1
            request = HBFServingRequest(
                request_id=call.request_id,
                session_id=call.session_id,
                arrival_ns=call.release_ns,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                hbf_prefix_tokens=0,
                lpddr_prefix_tokens=0,
                group_id=group_id,
            )
        else:
            reuse = min(
                call.prefix_reuse_tokens,
                placement.total_tokens,
                call.input_tokens,
            )
            route = lifecycle.route_resume(
                call.session_id,
                now_ns=self.current_ns,
                request_id=call.request_id,
                prefix_reuse_tokens=reuse,
                input_tokens=call.input_tokens,
                lpddr_growth_tokens=(
                    call.input_tokens - reuse
                    + call.output_tokens - 1
                ),
            )
            if route.execution != ResumeExecution.HBF:
                raise RuntimeError(
                    "HBF-only cluster routed a resume off HBF: "
                    f"{route.reason}")
            call.execution = HybridExecution.HBF_READY
            call.operational_reuse_tokens = reuse
            call.route_reason = route.reason
            self.metrics.hbf_resumes += 1
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
        call.hbf_server_index = self._server_by_session[call.session_id]
        call.state = HybridCallState.HBF_EXECUTING
        self._live_request_server[call.request_id] = server
        return request

    def _admit_pending(self, now_ns: int) -> None:
        deferred: deque[int] = deque()
        per_server: dict[int, list[HBFServingRequest]] = {}
        while self._pending_call_ids:
            request_id = self._pending_call_ids.popleft()
            call = self.calls[request_id]
            session = self.sessions[call.session_id]
            if call.call_index != session.last_internal_call_index + 1:
                deferred.append(request_id)
                continue
            request = self._route_call(call)
            server = self._live_request_server[request_id]
            per_server.setdefault(server, []).append(request)
        self._pending_call_ids = deferred
        for server, requests in per_server.items():
            requests.sort(key=lambda value: value.request_id)
            self.pools[server].submit_many(
                requests, now_ns=now_ns, defer_schedule=True)

    # -- event loop ----------------------------------------------------
    def _consume_completions(self, now_ns: int) -> None:
        for server, pool in enumerate(self.pools):
            for request in pool.pop_completed():
                call = self.calls[request.request_id]
                self._live_request_server.pop(request.request_id, None)
                call.user_completion_ns = request.completion_ns
                call.internal_completion_ns = request.completion_ns
                call.state = HybridCallState.INTERNAL_COMPLETE
                self._user_completed_ids.append(request.request_id)
                self.metrics.user_completed_calls += 1
                self.metrics.internal_completed_calls += 1
                self.lifecycles[server].complete_hbf_turn(
                    call.session_id,
                    now_ns=now_ns,
                    total_tokens=(
                        call.input_tokens + call.output_tokens - 1),
                    has_successor=call.has_successor,
                )
                session = self.sessions[call.session_id]
                session.last_internal_call_index = call.call_index
                session.ended = not call.has_successor

    def _next_raw_event_ns(self) -> Optional[int]:
        values = []
        for pool in self.pools:
            value = pool.next_event_ns()
            if value is not None:
                values.append(value)
        for lifecycle in self.lifecycles:
            value = lifecycle.next_completion_ns()
            if value is not None:
                values.append(value)
        return min(values) if values else None

    def advance(
            self, now_ns: int, *,
            defer_schedule: bool = False) -> None:
        if now_ns < self.current_ns:
            raise ValueError("cluster time cannot move backwards")
        while True:
            event_ns = self._next_raw_event_ns()
            if event_ns is None or event_ns > now_ns:
                break
            for lifecycle in self.lifecycles:
                lifecycle.advance(event_ns)
            for pool in self.pools:
                pool.advance(event_ns, defer_schedule=True)
            self.current_ns = event_ns
            self._consume_completions(event_ns)
            if defer_schedule and event_ns == now_ns:
                break
            self.flush_scheduling(event_ns)
        for lifecycle in self.lifecycles:
            lifecycle.advance(now_ns)
        for pool in self.pools:
            pool.advance(now_ns, defer_schedule=True)
        self.current_ns = now_ns
        self._consume_completions(now_ns)
        if not defer_schedule:
            self.flush_scheduling(now_ns)

    def flush_scheduling(self, now_ns: int) -> None:
        self._admit_pending(now_ns)
        for pool in self.pools:
            pool.flush_scheduling(now_ns)

    def next_event_ns(self) -> Optional[int]:
        return self._next_raw_event_ns()

    def pop_completed(self) -> list[HybridCall]:
        result = []
        while self._user_completed_ids:
            result.append(
                self.calls[self._user_completed_ids.popleft()])
        return result

    def has_pending_external(self) -> bool:
        return False

    def _deadlock_detail(self) -> str:
        return f"pending={list(self._pending_call_ids)[:8]}"

    def assert_invariants(self) -> None:
        for lifecycle in self.lifecycles:
            lifecycle.assert_invariants()
        for pool in self.pools:
            pool.assert_invariants()

    def report(self) -> Mapping[str, Any]:
        return {
            "mode": "hbf_only_cluster_node",
            "architecture": {
                "gpu_server_count": 0,
                "hbf_server_count": self.hbf_server_count,
                "hbf_layout": self.hbf_layout.key,
                "execution_backend": "analytical_calendar",
            },
            "policy": asdict(self.migration_policy),
            "metrics": asdict(self.metrics),
            "current_ns": self.current_ns,
            "servers": [
                {
                    "lifecycle": lifecycle.report(),
                    "pool": pool.report(),
                }
                for lifecycle, pool in zip(self.lifecycles, self.pools)
            ],
        }


class HBFOnlyClusterSystem(GPUHBFHybridSystem):
    """Agentic event loop over :class:`HBFOnlyClusterNode`."""

    def __init__(
            self, *, repo_root: Path,
            hbf_hardware: Optional[HBFServerHardware] = None,
            hbf_layout: str = "tp4",
            server_count: int = 2,
            max_num_batched_tokens: int = 8_192,
            max_num_seqs: int = 128,
            max_prefill_chunk_tokens: int = 4_096,
            band: str = "central",
            validate_every_event: bool = True,
            **_ignored) -> None:
        self.node = HBFOnlyClusterNode(
            repo_root=repo_root,
            hbf_hardware=(
                HBFServerHardware()
                if hbf_hardware is None else hbf_hardware
            ),
            hbf_layout=hbf_layout,
            server_count=server_count,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_prefill_chunk_tokens=max_prefill_chunk_tokens,
            band=band,
            validate_every_event=validate_every_event,
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


__all__ = [
    "HBFOnlyClusterNode",
    "HBFOnlyClusterSystem",
]

"""Single-P4D4 causal systems for matched baseline/Oracle comparisons.

This module deliberately does not reuse or alter the frozen dual-node
comparison systems.  It provides a separate one-node event contract for:

* a finite-HBM P4D4 node with configurable local CPU/SSD tiering, and
* a strict infinite-HBM P4D4 performance Oracle.

Both wrappers consume the same immutable ``ScheduledSession`` values.  Only
first calls have offered arrival times; each successor is released at its
predecessor's user completion plus the trace-provided tool duration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .gpu_pd_latency import P4D4GPUHardware
from .gpu_pd_oracle_node import (
    OracleCallState,
    OracleNodeCall,
    StrictInfiniteHBMNode,
)
from .gpu_pd_tier_lifecycle import (
    RESTORE_EXECUTION_BULK,
    SUPPORTED_TIER_POLICIES,
)
from .gpu_pd_tiered_node import (
    FiniteHBMTieredP4D4Node,
    TieredCallState,
    TieredNodeCall,
)
from .hbf_comparison_metrics import CompletedRequest, RequestKey
from .hbf_comparison_workload import (
    ScheduledSession,
    full_drain_hashes,
    stable_json_sha256,
)


SINGLE_GPU_NODE_COUNT = 1
SINGLE_GPU_NODE_ID = 0
SINGLE_NODE_ROUTE_POLICY = "single_gpu_node_sticky"


class SingleP4D4DeadlockError(RuntimeError):
    """Raised when unfinished single-node work has no future event."""


@dataclass(frozen=True)
class SingleP4D4CallSpec:
    """Immutable identity and demand consumed by either one-node system."""

    request_id: int
    key: RequestKey
    source_index: int
    offer_index: int
    node_id: int
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
class SingleP4D4Metrics:
    scheduled_sessions: int = 0
    scheduled_calls: int = 0
    released_calls: int = 0
    completed_calls: int = 0
    event_timestamps: int = 0
    fixed_point_rounds: int = 0
    max_release_heap: int = 0


_RuntimeCall = OracleNodeCall | TieredNodeCall
_ServingNode = StrictInfiniteHBMNode | FiniteHBMTieredP4D4Node


class _SingleP4D4CausalSystem:
    """Common one-node schedule, release, and fixed-point event loop."""

    def __init__(
            self, *, repo_root: Path, hardware: P4D4GPUHardware,
            node: _ServingNode, validate_every_event: bool,
            mode: str) -> None:
        if not isinstance(validate_every_event, bool):
            raise ValueError("validate_every_event must be a boolean")
        if not isinstance(node, (
                StrictInfiniteHBMNode, FiniteHBMTieredP4D4Node)):
            raise TypeError("node must be a supported P4D4 serving node")
        if node.node_id != SINGLE_GPU_NODE_ID:
            raise ValueError("single-node systems require node_id=0")
        if not isinstance(mode, str) or not mode:
            raise ValueError("mode must be a non-empty string")

        self.repo_root = Path(repo_root)
        self.hardware = hardware
        self.node = node
        self.nodes = (node,)
        self.validate_every_event = validate_every_event
        self.route_policy = SINGLE_NODE_ROUTE_POLICY
        self.mode = mode
        self.metrics = SingleP4D4Metrics()
        self.current_ns = 0
        self.call_specs: tuple[SingleP4D4CallSpec, ...] = ()

        self._spec_by_request: dict[int, SingleP4D4CallSpec] = {}
        self._request_by_identity: dict[str, int] = {}
        self._successor_by_request: dict[int, int] = {}
        self._first_arrival_by_request: dict[int, int] = {}
        self._route_by_session: dict[str, int] = {}
        self._offer_by_session: dict[str, int] = {}
        self._release_heap: list[tuple[int, int]] = []
        self._queued_release_ids: set[int] = set()
        self._released_ids: set[int] = set()
        self._completed_ids: set[int] = set()
        self._runtime_calls: dict[int, _RuntimeCall] = {}
        self._completed_snapshots: dict[int, CompletedRequest] = {}
        self._completion_order: list[int] = []
        self._completed_session_order: list[str] = []
        self._loaded = False
        self._running = False
        self._finished = False
        self.assert_invariants()

    @staticmethod
    def _validate_scheduled_session(
            scheduled: ScheduledSession) -> None:
        if not isinstance(scheduled, ScheduledSession):
            raise TypeError(
                "scheduled sessions must be ScheduledSession values")
        if (
            isinstance(scheduled.offer_index, bool)
            or not isinstance(scheduled.offer_index, int)
            or scheduled.offer_index < 0
        ):
            raise ValueError(
                "scheduled offer_index must be a non-negative integer")
        if (
            isinstance(scheduled.arrival_time_ns, bool)
            or not isinstance(scheduled.arrival_time_ns, int)
            or scheduled.arrival_time_ns < 0
        ):
            raise ValueError(
                "scheduled arrival_time_ns must be a non-negative integer")
        session = scheduled.session
        if not session.session_id or not session.calls:
            raise ValueError("scheduled session must have calls")
        for call_index, call in enumerate(session.calls):
            if call.session_id != session.session_id:
                raise ValueError(
                    "call/session identity mismatch in scheduled session")
            if call.call_index != session.calls[0].call_index + call_index:
                raise ValueError(
                    "scheduled calls must use contiguous call indices")
            for name in (
                "input_tokens",
                "output_tokens",
                "cached_prefix_tokens",
                "tool_duration_ns",
            ):
                value = getattr(call, name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                ):
                    raise ValueError(f"{name} must be an integer")
            if call.input_tokens <= 0 or call.output_tokens <= 0:
                raise ValueError(
                    "scheduled input/output tokens must be positive")
            if not 0 <= call.cached_prefix_tokens <= call.input_tokens:
                raise ValueError(
                    "cached_prefix_tokens must be in 0..input_tokens")
            if call.call_index == 0 and call.cached_prefix_tokens:
                raise ValueError(
                    "first call cannot reuse an earlier prefix")
            if call.tool_duration_ns < 0:
                raise ValueError(
                    "tool_duration_ns must be non-negative")

    def _make_runtime_call(
            self, spec: SingleP4D4CallSpec, *,
            release_ns: int) -> _RuntimeCall:
        common = {
            "request_id": spec.request_id,
            "session_id": spec.session_id,
            "call_index": spec.call_index,
            "release_ns": release_ns,
            "input_tokens": spec.input_tokens,
            "output_tokens": spec.output_tokens,
            "prefix_reuse_tokens": spec.cached_prefix_tokens,
            "has_successor": spec.has_successor,
        }
        if isinstance(self.node, FiniteHBMTieredP4D4Node):
            return TieredNodeCall(**common)
        return OracleNodeCall(**common)

    def _preflight_call(
            self, spec: SingleP4D4CallSpec, *,
            release_ns: int) -> None:
        call = self._make_runtime_call(spec, release_ns=release_ns)
        call.validate()
        if isinstance(self.node, FiniteHBMTieredP4D4Node):
            if not isinstance(call, TieredNodeCall):
                raise AssertionError("tiered preflight call type mismatch")
            self.node._validate_capacity_contract(call)

    def load(
            self,
            scheduled_sessions: Iterable[ScheduledSession]) -> None:
        """Validate the complete cohort, then freeze IDs and first arrivals."""

        if self._loaded:
            raise RuntimeError("single-node schedule is already loaded")
        scheduled_values = list(scheduled_sessions)
        if not scheduled_values:
            raise ValueError("scheduled_sessions cannot be empty")
        for scheduled in scheduled_values:
            self._validate_scheduled_session(scheduled)

        offer_indices = [
            scheduled.offer_index for scheduled in scheduled_values]
        if len(offer_indices) != len(set(offer_indices)):
            raise ValueError("scheduled sessions contain duplicate offers")
        session_ids = [
            scheduled.session.session_id
            for scheduled in scheduled_values
        ]
        if len(session_ids) != len(set(session_ids)):
            raise ValueError(
                "scheduled sessions contain duplicate session IDs")
        source_indices = [
            scheduled.session.source_index
            for scheduled in scheduled_values
        ]
        if len(source_indices) != len(set(source_indices)):
            raise ValueError(
                "scheduled sessions contain duplicate source indices")

        canonical_order = sorted(
            scheduled_values,
            key=lambda value: (
                value.session.source_index,
                value.session.session_id,
            ),
        )
        provisional_specs = []
        next_request_id = 0
        completion_identities = set()
        for scheduled in canonical_order:
            session = scheduled.session
            for call_index, call in enumerate(session.calls):
                identity = call.completion_identity
                if identity in completion_identities:
                    raise ValueError(
                        "scheduled calls contain duplicate completion "
                        f"identity={identity!r}")
                completion_identities.add(identity)
                spec = SingleP4D4CallSpec(
                    request_id=next_request_id,
                    key=RequestKey(session.session_id, call.call_index),
                    source_index=session.source_index,
                    offer_index=scheduled.offer_index,
                    node_id=SINGLE_GPU_NODE_ID,
                    session_id=session.session_id,
                    call_index=call.call_index,
                    input_tokens=call.input_tokens,
                    output_tokens=call.output_tokens,
                    cached_prefix_tokens=call.cached_prefix_tokens,
                    tool_duration_ns=call.tool_duration_ns,
                    has_successor=call_index + 1 < len(session.calls),
                )
                self._preflight_call(
                    spec, release_ns=scheduled.arrival_time_ns)
                provisional_specs.append((scheduled, spec))
                next_request_id += 1

        prior_request_by_session: dict[str, int] = {}
        specs = []
        for scheduled, spec in provisional_specs:
            specs.append(spec)
            self._spec_by_request[spec.request_id] = spec
            self._request_by_identity[
                spec.completion_identity] = spec.request_id
            self._route_by_session[spec.session_id] = SINGLE_GPU_NODE_ID
            self._offer_by_session[spec.session_id] = spec.offer_index
            prior_request_id = prior_request_by_session.get(
                spec.session_id)
            if prior_request_id is None:
                self._first_arrival_by_request[spec.request_id] = (
                    scheduled.arrival_time_ns)
                self._queue_release(
                    spec.request_id,
                    release_ns=scheduled.arrival_time_ns,
                )
            else:
                self._successor_by_request[
                    prior_request_id] = spec.request_id
            prior_request_by_session[spec.session_id] = spec.request_id

        self.call_specs = tuple(specs)
        self.metrics.scheduled_sessions = len(scheduled_values)
        self.metrics.scheduled_calls = len(specs)
        self._loaded = True
        self.assert_invariants()

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
            raise ValueError("release_ns must be a non-negative integer")
        heapq.heappush(self._release_heap, (release_ns, request_id))
        self._queued_release_ids.add(request_id)
        self.metrics.max_release_heap = max(
            self.metrics.max_release_heap,
            len(self._release_heap),
        )

    def node_for_session(self, session_id: str) -> int:
        if session_id not in self._route_by_session:
            raise KeyError(f"unknown session_id={session_id!r}")
        return self._route_by_session[session_id]

    def request_id_for(self, completion_identity: str) -> int:
        if completion_identity not in self._request_by_identity:
            raise KeyError(
                f"unknown completion identity={completion_identity!r}")
        return self._request_by_identity[completion_identity]

    @property
    def completed_requests(self) -> tuple[CompletedRequest, ...]:
        return tuple(
            self._completed_snapshots[request_id]
            for request_id in self._completion_order
        )

    def _consume_node_completions(
            self, now_ns: int) -> list[_RuntimeCall]:
        completed = list(self.node.pop_completed())
        completed.sort(key=lambda call: (
            call.user_completion_ns,
            call.request_id,
        ))
        for call in completed:
            request_id = call.request_id
            if request_id in self._completed_ids:
                raise RuntimeError(
                    f"duplicate system completion request_id={request_id}")
            if call.user_completion_ns != now_ns:
                raise RuntimeError(
                    "node returned a completion at the wrong global time")
            spec = self._spec_by_request[request_id]
            if (
                call.session_id != spec.session_id
                or call.call_index != spec.call_index
            ):
                raise RuntimeError(
                    "runtime completion does not match frozen call spec")
            pool_request = call.pool_request
            if (
                pool_request is None
                or pool_request.first_token_ns is None
                or pool_request.completion_ns is None
            ):
                raise RuntimeError(
                    "completed single-node call lacks token timestamps")
            self._completed_ids.add(request_id)
            self._completion_order.append(request_id)
            self._completed_snapshots[request_id] = CompletedRequest(
                key=spec.key,
                release_ns=call.release_ns,
                first_token_ns=pool_request.first_token_ns,
                completion_ns=pool_request.completion_ns,
                output_tokens=call.output_tokens,
            )
            self.metrics.completed_calls += 1
            successor_id = self._successor_by_request.get(request_id)
            if successor_id is None:
                self._completed_session_order.append(spec.session_id)
            else:
                self._queue_release(
                    successor_id,
                    release_ns=now_ns + spec.tool_duration_ns,
                )
        return completed

    def _pop_releases(self, now_ns: int) -> list[_RuntimeCall]:
        arrivals = []
        while self._release_heap and self._release_heap[0][0] == now_ns:
            release_ns, request_id = heapq.heappop(self._release_heap)
            if request_id in self._released_ids:
                raise RuntimeError(
                    f"duplicate runtime release request_id={request_id}")
            spec = self._spec_by_request[request_id]
            call = self._make_runtime_call(spec, release_ns=release_ns)
            self._released_ids.add(request_id)
            self._runtime_calls[request_id] = call
            arrivals.append(call)
            self.metrics.released_calls += 1
        if self._release_heap and self._release_heap[0][0] < now_ns:
            raise RuntimeError("global release heap fell behind current time")
        arrivals.sort(key=lambda call: call.request_id)
        return arrivals

    def _same_time_work_exists(self, now_ns: int) -> bool:
        return (
            (
                bool(self._release_heap)
                and self._release_heap[0][0] == now_ns
            )
            or self.node.next_event_ns() == now_ns
        )

    def _process_timestamp(self, now_ns: int) -> None:
        if now_ns < self.current_ns:
            raise ValueError(
                "global single-node time cannot move backwards")
        self.metrics.event_timestamps += 1
        rounds = 0
        while True:
            rounds += 1
            if rounds > self.metrics.scheduled_calls + 4:
                raise SingleP4D4DeadlockError(
                    "same-timestamp fixed point did not converge")
            self.node.advance(now_ns, defer_schedule=True)
            self._consume_node_completions(now_ns)
            arrivals = self._pop_releases(now_ns)
            if arrivals:
                self.node.submit_many(arrivals, now_ns=now_ns)
            else:
                self.node.flush_scheduling(now_ns)
            self.metrics.fixed_point_rounds += 1
            self.current_ns = now_ns
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

    def _is_internal_complete(self, call: _RuntimeCall) -> bool:
        if isinstance(call, TieredNodeCall):
            return call.state == TieredCallState.INTERNAL_COMPLETE
        return call.state == OracleCallState.INTERNAL_COMPLETE

    def _deadlock_detail(self) -> str:
        unreleased = sorted(
            set(self._spec_by_request) - self._released_ids)
        unfinished = sorted(
            set(self._spec_by_request) - self._completed_ids)
        node_pending = sorted(
            call.request_id
            for call in self.node.calls.values()
            if not self._is_internal_complete(call)
        )
        return (
            f"mode={self.mode}, unreleased={unreleased[:8]}, "
            f"unfinished={unfinished[:8]}, "
            f"node_pending={node_pending[:8]}"
        )

    def run(
            self,
            scheduled_sessions: Optional[
                Iterable[ScheduledSession]] = None,
    ) -> list[CompletedRequest]:
        """Run through user completion and complete internal node drain."""

        if scheduled_sessions is not None:
            self.load(scheduled_sessions)
        if not self._loaded:
            raise RuntimeError("load a schedule before running")
        if self._running:
            raise RuntimeError("single-node system is already running")
        if self._finished:
            return list(self.completed_requests)

        self._running = True
        try:
            while True:
                next_event_ns = self._next_event_ns()
                if next_event_ns is None:
                    all_user_complete = (
                        len(self._completed_ids)
                        == self.metrics.scheduled_calls
                    )
                    all_internal_complete = all(
                        self._is_internal_complete(call)
                        for call in self.node.calls.values()
                    )
                    if (
                        all_user_complete
                        and all_internal_complete
                        and len(self._released_ids)
                        == self.metrics.scheduled_calls
                    ):
                        break
                    raise SingleP4D4DeadlockError(
                        "unfinished single-node work has no future event: "
                        + self._deadlock_detail())
                self._process_timestamp(next_event_ns)
        finally:
            self._running = False

        self._finished = True
        self.assert_invariants()
        return list(self.completed_requests)

    def assert_invariants(self) -> None:
        self.node.assert_invariants()
        if (
            len(self.nodes) != SINGLE_GPU_NODE_COUNT
            or self.nodes[0] is not self.node
            or self.node.node_id != SINGLE_GPU_NODE_ID
        ):
            raise AssertionError(
                "single-node system must own exactly GPU node 0")
        if set(self._route_by_session.values()) - {SINGLE_GPU_NODE_ID}:
            raise AssertionError("single-node system contains invalid route")
        for spec in self.call_specs:
            if spec.node_id != SINGLE_GPU_NODE_ID:
                raise AssertionError(
                    f"single-node spec has invalid route: {spec}")
            if self._route_by_session[spec.session_id] != spec.node_id:
                raise AssertionError(
                    f"non-sticky session route in spec={spec}")
            runtime = self._runtime_calls.get(spec.request_id)
            if (
                runtime is not None
                and self.node.calls.get(spec.request_id) is not runtime
            ):
                raise AssertionError(
                    "runtime call is not owned by the single node")
        if not self._completed_ids <= self._released_ids:
            raise AssertionError("completed request was never released")
        if self.metrics.released_calls != len(self._released_ids):
            raise AssertionError("released-call metric mismatch")
        if self.metrics.completed_calls != len(self._completed_ids):
            raise AssertionError("completed-call metric mismatch")
        if self._completed_ids != set(self._completed_snapshots):
            raise AssertionError("completed-request snapshot mismatch")
        if self._finished and (
            len(self._completed_ids) != len(self.call_specs)
            or len(self._released_ids) != len(self.call_specs)
        ):
            raise AssertionError("finished system lacks full call drain")
        if len(self._completed_session_order) != len(set(
                self._completed_session_order)):
            raise AssertionError("session completed more than once")
        if self._finished and len(self._completed_session_order) != (
                self.metrics.scheduled_sessions):
            raise AssertionError("finished system lacks session completions")

    def _routing_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "offer_index": self._offer_by_session[session_id],
                "session_id": session_id,
                "node_id": node_id,
            }
            for session_id, node_id in sorted(
                self._route_by_session.items(),
                key=lambda item: self._offer_by_session[item[0]],
            )
        ]

    def _system_specific_report(self) -> Mapping[str, Any]:
        return {}

    def report(self) -> Mapping[str, Any]:
        routing_rows = self._routing_rows()
        spec_rows = [asdict(spec) for spec in self.call_specs]
        completed_identities = [
            self._spec_by_request[request_id].completion_identity
            for request_id in self._completion_order
        ]
        result: dict[str, Any] = {
            "mode": self.mode,
            "node_count": SINGLE_GPU_NODE_COUNT,
            "gpu_server_count": SINGLE_GPU_NODE_COUNT,
            "routing_policy": self.route_policy,
            "routing_policy_scope": (
                "all sessions are sticky to the only physical GPU server"),
            "validate_every_event": self.validate_every_event,
            "routing": routing_rows,
            "routing_identity_sha256": stable_json_sha256(routing_rows),
            "call_specs": spec_rows,
            "call_specs_identity_sha256": stable_json_sha256(spec_rows),
            "metrics": asdict(self.metrics),
            "current_ns": self.current_ns,
            "finished": self._finished,
            "completion_order": completed_identities,
            "completed_requests": [
                asdict(request) for request in self.completed_requests],
            "node": self.node.report(),
            "nodes": [self.node.report()],
            **dict(self._system_specific_report()),
        }
        if self._finished:
            expected_calls = [
                spec.completion_identity for spec in self.call_specs]
            call_drain = asdict(full_drain_hashes(
                expected_calls, completed_identities))
            expected_sessions = [
                row["session_id"] for row in routing_rows]
            session_drain = asdict(full_drain_hashes(
                expected_sessions, self._completed_session_order))
            result["full_drain"] = call_drain
            result["call_full_drain"] = call_drain
            result["session_full_drain"] = session_drain
        return result


class SingleStrictInfiniteHBMOracle(_SingleP4D4CausalSystem):
    """One P4D4 server with nonbinding HBM capacity."""

    def __init__(
            self, *, repo_root: Path, hardware: P4D4GPUHardware,
            max_num_batched_tokens: int = 8_192,
            max_num_seqs: int = 128,
            p_max_num_seqs: Optional[int] = None,
            d_max_num_seqs: Optional[int] = None,
            max_prefill_chunk_tokens: int = 4_096,
            band: str = "central",
            validate_every_event: bool = True) -> None:
        node = StrictInfiniteHBMNode(
            repo_root=repo_root,
            hardware=hardware,
            node_id=SINGLE_GPU_NODE_ID,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            p_max_num_seqs=p_max_num_seqs,
            d_max_num_seqs=d_max_num_seqs,
            max_prefill_chunk_tokens=max_prefill_chunk_tokens,
            band=band,
            validate_every_event=validate_every_event,
            retain_detailed_history=validate_every_event,
        )
        super().__init__(
            repo_root=repo_root,
            hardware=hardware,
            node=node,
            validate_every_event=validate_every_event,
            mode="single_strict_infinite_hbm_residency_oracle",
        )

    def _system_specific_report(self) -> Mapping[str, Any]:
        return {
            "oracle_contract": (
                "one physical P4D4 compute server; infinite capacity is a "
                "performance-only HBM residency proof"),
        }


class SingleFiniteHBMTieredBaseline(_SingleP4D4CausalSystem):
    """One finite-HBM P4D4 server with node-local CPU/SSD tiering."""

    def __init__(
            self, *, repo_root: Path, hardware: P4D4GPUHardware,
            policy: str = "ssd_direct",
            p_capacity_bytes_per_rank: Optional[int] = None,
            d_capacity_bytes_per_rank: Optional[int] = None,
            cpu_capacity_bytes: Optional[int] = None,
            ssd_capacity_bytes: Optional[int] = None,
            max_num_batched_tokens: int = 8_192,
            max_num_seqs: int = 128,
            p_max_num_seqs: Optional[int] = None,
            d_max_num_seqs: Optional[int] = None,
            max_prefill_chunk_tokens: int = 4_096,
            band: str = "central",
            restore_execution_mode: str = RESTORE_EXECUTION_BULK,
            validate_every_event: bool = True) -> None:
        if policy not in SUPPORTED_TIER_POLICIES:
            raise ValueError(f"unsupported tier policy {policy!r}")
        node = FiniteHBMTieredP4D4Node(
            repo_root=repo_root,
            hardware=hardware,
            node_id=SINGLE_GPU_NODE_ID,
            policy=policy,
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
        self.policy = policy
        self.restore_execution_mode = restore_execution_mode
        super().__init__(
            repo_root=repo_root,
            hardware=hardware,
            node=node,
            validate_every_event=validate_every_event,
            mode="single_finite_hbm_p4d4_local_tiering",
        )

    def _system_specific_report(self) -> Mapping[str, Any]:
        if not isinstance(self.node, FiniteHBMTieredP4D4Node):
            raise AssertionError("baseline owns the wrong node type")
        return {
            "policy": self.policy,
            "restore_execution_mode": self.restore_execution_mode,
            "physical_isolation": (
                "one P4D4 server; HBM, CPU DRAM, PCIe, and SSD queues and "
                "capacity ledgers are node-local"),
            "local_ssd": {
                "device_count": self.hardware.ssd_device_count,
                "capacity_bytes_per_device": (
                    self.hardware.ssd_capacity_bytes_per_device),
                "aggregate_capacity_bytes": (
                    self.node.lifecycle.ssd_ledger.capacity_bytes),
                "read_bandwidth_gbps": (
                    self.hardware.ssd_read_bandwidth_gbps),
                "write_bandwidth_gbps": (
                    self.hardware.ssd_write_bandwidth_gbps),
            },
        }


SingleFiniteHBMTieredP4D4System = SingleFiniteHBMTieredBaseline
SingleStrictInfiniteHBMOracleSystem = SingleStrictInfiniteHBMOracle


__all__ = [
    "SINGLE_GPU_NODE_COUNT",
    "SINGLE_GPU_NODE_ID",
    "SINGLE_NODE_ROUTE_POLICY",
    "SingleFiniteHBMTieredBaseline",
    "SingleFiniteHBMTieredP4D4System",
    "SingleP4D4CallSpec",
    "SingleP4D4DeadlockError",
    "SingleP4D4Metrics",
    "SingleStrictInfiniteHBMOracle",
    "SingleStrictInfiniteHBMOracleSystem",
]

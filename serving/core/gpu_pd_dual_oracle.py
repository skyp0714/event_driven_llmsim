"""Two independent strict infinite-HBM P4D4 oracle nodes.

The system freezes routing and request identities before simulation, while
agentic successor release times remain dynamic:

``successor release = predecessor user completion + tool duration``.

All nodes are advanced to a shared timestamp before completions and arrivals
at that timestamp are collected.  Each node then receives at most one
``submit_many`` call in that fixed-point round.  This prevents node iteration
order from changing co-timed batching.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .comparison_cutoff import ResumableCutoffEventLoopMixin
from .gpu_pd_latency import P4D4GPUHardware
from .gpu_pd_oracle_node import (
    OracleCallState,
    OracleNodeCall,
    StrictInfiniteHBMNode,
)
from .hbf_comparison_workload import (
    ScheduledSession,
    full_drain_hashes,
    stable_json_sha256,
)
from .hbf_comparison_metrics import CompletedRequest, RequestKey


DUAL_ORACLE_NODE_COUNT = 2
ROUTE_OFFER_RR = "offer_index_mod_2_sticky"
ROUTE_BALANCED_TRACE_WORK = "balanced_trace_work_static"
SUPPORTED_ROUTE_POLICIES = frozenset({
    ROUTE_OFFER_RR,
    ROUTE_BALANCED_TRACE_WORK,
})


class DualOracleDeadlockError(RuntimeError):
    """Raised when unfinished work has no future event."""


@dataclass(frozen=True)
class DualOracleCallSpec:
    """Immutable request identity and demand fixed before simulation."""

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
class DualOracleMetrics:
    scheduled_sessions: int = 0
    scheduled_calls: int = 0
    released_calls: int = 0
    completed_calls: int = 0
    event_timestamps: int = 0
    fixed_point_rounds: int = 0
    max_release_heap: int = 0


class DualStrictInfiniteHBMOracle(ResumableCutoffEventLoopMixin):
    """Global event loop for two independent strict-oracle P4D4 nodes."""

    def __init__(
            self, *, repo_root: Path, hardware: P4D4GPUHardware,
            max_num_batched_tokens: int = 8_192,
            max_num_seqs: int = 128,
            p_max_num_seqs: Optional[int] = None,
            d_max_num_seqs: Optional[int] = None,
            max_prefill_chunk_tokens: int = 4_096,
            band: str = "central",
            validate_every_event: bool = True,
            route_policy: str = ROUTE_OFFER_RR) -> None:
        if not isinstance(validate_every_event, bool):
            raise ValueError("validate_every_event must be a boolean")
        if route_policy not in SUPPORTED_ROUTE_POLICIES:
            raise ValueError(
                "route_policy must be one of "
                f"{sorted(SUPPORTED_ROUTE_POLICIES)}")
        self.repo_root = Path(repo_root)
        self.hardware = hardware
        self.validate_every_event = validate_every_event
        self.route_policy = route_policy
        self.nodes = tuple(
            StrictInfiniteHBMNode(
                repo_root=self.repo_root,
                hardware=hardware,
                node_id=node_id,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                p_max_num_seqs=p_max_num_seqs,
                d_max_num_seqs=d_max_num_seqs,
                max_prefill_chunk_tokens=max_prefill_chunk_tokens,
                band=band,
                validate_every_event=validate_every_event,
                retain_detailed_history=validate_every_event,
            )
            for node_id in range(DUAL_ORACLE_NODE_COUNT)
        )
        self.metrics = DualOracleMetrics()
        self.current_ns = 0
        self.call_specs: tuple[DualOracleCallSpec, ...] = ()
        self._spec_by_request: dict[int, DualOracleCallSpec] = {}
        self._request_by_identity: dict[str, int] = {}
        self._successor_by_request: dict[int, int] = {}
        self._first_arrival_by_request: dict[int, int] = {}
        self._route_by_session: dict[str, int] = {}
        self._route_work_by_session: dict[str, int] = {}
        self._offer_by_session: dict[str, int] = {}
        self._release_heap: list[tuple[int, int]] = []
        self._queued_release_ids: set[int] = set()
        self._released_ids: set[int] = set()
        self._completed_ids: set[int] = set()
        self._runtime_calls: dict[int, OracleNodeCall] = {}
        self._completed_snapshots: dict[int, CompletedRequest] = {}
        self._completion_order: list[int] = []
        self._completed_session_order: list[str] = []
        self._loaded = False
        self._running = False
        self._finished = False

    @staticmethod
    def route_offer(offer_index: int) -> int:
        """Return the deterministic sticky node for an offered session."""

        if (
            isinstance(offer_index, bool)
            or not isinstance(offer_index, int)
            or offer_index < 0
        ):
            raise ValueError("offer_index must be a non-negative integer")
        return offer_index % DUAL_ORACLE_NODE_COUNT

    @staticmethod
    def trace_work_proxy(scheduled: ScheduledSession) -> int:
        """Return a deterministic attention-context service proxy.

        The proxy is used only to freeze a balanced experimental shard map.
        It is trace-informed and is not presented as an online router.
        """

        DualStrictInfiniteHBMOracle._validate_scheduled_session(
            scheduled)
        work = 0
        for call in scheduled.session.calls:
            fresh = call.input_tokens - call.cached_prefix_tokens
            # Sum the attended context across the fresh prefill suffix.
            prefill_context_work = (
                fresh
                * (
                    2 * call.cached_prefix_tokens
                    + fresh
                    + 1
                )
                // 2
            )
            decode_steps = call.output_tokens - 1
            # Token 1 is emitted by P.  D produces tokens 2..N at contexts
            # input, input+1, ..., input+N-2.
            decode_context_work = (
                decode_steps * call.input_tokens
                + decode_steps * (decode_steps - 1) // 2
            )
            # Retained-prefix movement is linear in the copied KV bytes.
            work += (
                prefill_context_work
                + decode_context_work
                + call.cached_prefix_tokens
                + fresh
                + call.output_tokens
            )
        return work

    def _freeze_routes(
            self,
            scheduled_values: list[ScheduledSession],
    ) -> dict[str, int]:
        if self.route_policy == ROUTE_OFFER_RR:
            return {
                scheduled.session.session_id: self.route_offer(
                    scheduled.offer_index)
                for scheduled in scheduled_values
            }

        target_counts = (
            (len(scheduled_values) + 1) // 2,
            len(scheduled_values) // 2,
        )
        counts = [0, 0]
        loads = [0, 0]
        routes = {}
        weighted = [
            (self.trace_work_proxy(scheduled), scheduled)
            for scheduled in scheduled_values
        ]
        for work, scheduled in sorted(
                weighted,
                key=lambda item: (
                    -item[0],
                    item[1].session.source_index,
                    item[1].session.session_id,
                )):
            candidates = [
                node_id for node_id in range(DUAL_ORACLE_NODE_COUNT)
                if counts[node_id] < target_counts[node_id]
            ]
            node_id = min(
                candidates,
                key=lambda candidate: (
                    loads[candidate],
                    candidate,
                ),
            )
            session_id = scheduled.session.session_id
            routes[session_id] = node_id
            counts[node_id] += 1
            loads[node_id] += work
        return routes

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
        """Immutable metric inputs in deterministic completion order."""

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
            if call.call_index != call_index:
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
            if call_index == 0 and call.cached_prefix_tokens:
                raise ValueError(
                    "first call cannot reuse an earlier prefix")
            if call.tool_duration_ns < 0:
                raise ValueError(
                    "tool_duration_ns must be non-negative")

    def load(
            self,
            scheduled_sessions: Iterable[ScheduledSession]) -> None:
        """Freeze all call IDs, demands, routes, and first arrivals."""

        if self._loaded:
            raise RuntimeError("dual oracle schedule is already loaded")
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
        frozen_routes = self._freeze_routes(scheduled_values)
        self._route_work_by_session = {
            scheduled.session.session_id: self.trace_work_proxy(
                scheduled)
            for scheduled in scheduled_values
        }

        offer_order = sorted(
            scheduled_values,
            key=lambda value: value.offer_index,
        )
        canonical_order = sorted(
            scheduled_values,
            key=lambda value: (
                value.session.source_index,
                value.session.session_id,
            ),
        )
        specs = []
        next_request_id = 0
        completion_identities = set()
        for scheduled in canonical_order:
            session = scheduled.session
            node_id = frozen_routes[session.session_id]
            self._route_by_session[session.session_id] = node_id
            self._offer_by_session[session.session_id] = (
                scheduled.offer_index)
            prior_request_id = None
            for call_index, call in enumerate(session.calls):
                identity = call.completion_identity
                if identity in completion_identities:
                    raise ValueError(
                        "scheduled calls contain duplicate completion "
                        f"identity={identity!r}")
                completion_identities.add(identity)
                request_id = next_request_id
                next_request_id += 1
                spec = DualOracleCallSpec(
                    request_id=request_id,
                    key=RequestKey(
                        session.session_id,
                        call_index,
                    ),
                    source_index=session.source_index,
                    offer_index=scheduled.offer_index,
                    node_id=node_id,
                    session_id=session.session_id,
                    call_index=call_index,
                    input_tokens=call.input_tokens,
                    output_tokens=call.output_tokens,
                    cached_prefix_tokens=call.cached_prefix_tokens,
                    tool_duration_ns=call.tool_duration_ns,
                    has_successor=call_index + 1 < len(session.calls),
                )
                specs.append(spec)
                self._spec_by_request[request_id] = spec
                self._request_by_identity[identity] = request_id
                if prior_request_id is None:
                    self._first_arrival_by_request[request_id] = (
                        scheduled.arrival_time_ns)
                    self._queue_release(
                        request_id,
                        release_ns=scheduled.arrival_time_ns,
                    )
                else:
                    self._successor_by_request[
                        prior_request_id] = request_id
                prior_request_id = request_id

        self.call_specs = tuple(specs)
        self.metrics.scheduled_sessions = len(offer_order)
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
        heapq.heappush(
            self._release_heap,
            (release_ns, request_id),
        )
        self._queued_release_ids.add(request_id)
        self.metrics.max_release_heap = max(
            self.metrics.max_release_heap,
            len(self._release_heap),
        )

    def _consume_node_completions(
            self, now_ns: int) -> list[OracleNodeCall]:
        completed = []
        for node in self.nodes:
            completed.extend(node.pop_completed())
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
            self._completed_ids.add(request_id)
            self._completion_order.append(request_id)
            pool_request = call.pool_request
            if (
                pool_request is None
                or pool_request.first_token_ns is None
                or pool_request.completion_ns is None
            ):
                raise RuntimeError(
                    "completed oracle call lacks token timestamps")
            self._completed_snapshots[request_id] = CompletedRequest(
                key=spec.key,
                release_ns=call.release_ns,
                first_token_ns=pool_request.first_token_ns,
                completion_ns=pool_request.completion_ns,
                output_tokens=call.output_tokens,
            )
            self.metrics.completed_calls += 1
            successor_id = self._successor_by_request.get(request_id)
            if successor_id is not None:
                self._queue_release(
                    successor_id,
                    release_ns=(
                        now_ns + spec.tool_duration_ns),
                )
            else:
                self._completed_session_order.append(spec.session_id)
        return completed

    def _pop_releases(
            self, now_ns: int) -> dict[int, list[OracleNodeCall]]:
        by_node = {
            node_id: [] for node_id in range(DUAL_ORACLE_NODE_COUNT)}
        while (
            self._release_heap
            and self._release_heap[0][0] == now_ns
        ):
            release_ns, request_id = heapq.heappop(
                self._release_heap)
            if request_id in self._released_ids:
                raise RuntimeError(
                    f"duplicate runtime release request_id={request_id}")
            spec = self._spec_by_request[request_id]
            call = OracleNodeCall(
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
            by_node[spec.node_id].append(call)
            self.metrics.released_calls += 1
        if self._release_heap and self._release_heap[0][0] < now_ns:
            raise RuntimeError("global release heap fell behind current time")
        return by_node

    def _same_time_work_exists(self, now_ns: int) -> bool:
        if (
            self._release_heap
            and self._release_heap[0][0] == now_ns
        ):
            return True
        return any(
            node.next_event_ns() == now_ns
            for node in self.nodes
        )

    def _process_timestamp(self, now_ns: int) -> None:
        if now_ns < self.current_ns:
            raise ValueError("global oracle time cannot move backwards")
        self.metrics.event_timestamps += 1
        rounds = 0
        while True:
            rounds += 1
            if rounds > self.metrics.scheduled_calls + 4:
                raise DualOracleDeadlockError(
                    "same-timestamp fixed point did not converge")

            # Advance every node before observing any completion.  Nodes do
            # not schedule independently until all co-timed arrivals exist.
            for node in self.nodes:
                node.advance(now_ns, defer_schedule=True)
            self._consume_node_completions(now_ns)
            arrivals = self._pop_releases(now_ns)

            # Exactly one scheduling boundary per node in this round.
            for node_id, node in enumerate(self.nodes):
                calls = arrivals[node_id]
                if calls:
                    calls.sort(key=lambda call: call.request_id)
                    node.submit_many(calls, now_ns=now_ns)
                else:
                    node.flush_scheduling(now_ns)

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
        for node in self.nodes:
            event_ns = node.next_event_ns()
            if event_ns is not None:
                values.append(event_ns)
        return min(values) if values else None

    def _deadlock_detail(self) -> str:
        unreleased = sorted(
            set(self._spec_by_request) - self._released_ids)
        unfinished = sorted(
            set(self._spec_by_request) - self._completed_ids)
        node_pending = {
            node.node_id: [
                call.request_id
                for call in node.calls.values()
                if call.state != OracleCallState.INTERNAL_COMPLETE
            ]
            for node in self.nodes
        }
        return (
            f"unreleased={unreleased[:8]}, "
            f"unfinished={unfinished[:8]}, "
            f"node_pending={node_pending}"
        )

    def run(
            self,
            scheduled_sessions: Optional[
                Iterable[ScheduledSession]] = None,
    ) -> list[CompletedRequest]:
        """Run the global event loop through complete internal drain."""

        if scheduled_sessions is not None:
            self.load(scheduled_sessions)
        if not self._loaded:
            raise RuntimeError("load a schedule before running")
        if self._running:
            raise RuntimeError("dual oracle is already running")
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
                        call.state
                        == OracleCallState.INTERNAL_COMPLETE
                        for node in self.nodes
                        for call in node.calls.values()
                    )
                    if (
                        all_user_complete
                        and all_internal_complete
                        and len(self._released_ids)
                        == self.metrics.scheduled_calls
                    ):
                        break
                    raise DualOracleDeadlockError(
                        "unfinished dual-oracle work has no future event: "
                        + self._deadlock_detail())
                self._process_timestamp(next_event_ns)
        finally:
            self._running = False

        self._finished = True
        self.assert_invariants()
        return list(self.completed_requests)

    def assert_invariants(self) -> None:
        for node in self.nodes:
            node.assert_invariants()
        if len(self.nodes) != DUAL_ORACLE_NODE_COUNT:
            raise AssertionError("dual oracle must have exactly two nodes")
        if len(set(self._route_by_session.values()) - {0, 1}):
            raise AssertionError("dual oracle contains an invalid route")
        for spec in self.call_specs:
            if (
                self.route_policy == ROUTE_OFFER_RR
                and spec.node_id != spec.offer_index % 2
            ):
                raise AssertionError(
                    f"non-deterministic route in spec={spec}")
            if self._route_by_session[spec.session_id] != spec.node_id:
                raise AssertionError(
                    f"non-sticky session route in spec={spec}")
            runtime = self._runtime_calls.get(spec.request_id)
            if runtime is not None:
                node = self.nodes[spec.node_id]
                if node.calls.get(spec.request_id) is not runtime:
                    raise AssertionError(
                        "runtime call is owned by the wrong node")
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
        ):
            raise AssertionError("finished system lacks completions")
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

    def report(self) -> Mapping[str, Any]:
        routing_rows = self._routing_rows()
        route_work_by_node = {
            node_id: sum(
                self._route_work_by_session[session_id]
                for session_id, route_node_id
                in self._route_by_session.items()
                if route_node_id == node_id
            )
            for node_id in range(DUAL_ORACLE_NODE_COUNT)
        }
        minimum_work = min(route_work_by_node.values())
        maximum_work = max(route_work_by_node.values())
        spec_rows = [asdict(spec) for spec in self.call_specs]
        completed_identities = [
            self._spec_by_request[request_id].completion_identity
            for request_id in self._completion_order
        ]
        result: dict[str, Any] = {
            "mode": "dual_strict_infinite_hbm_residency_oracle",
            "node_count": DUAL_ORACLE_NODE_COUNT,
            "routing_policy": self.route_policy,
            "routing_policy_scope": (
                "trace-informed static experimental partition"
                if self.route_policy == ROUTE_BALANCED_TRACE_WORK
                else "online-compatible session round-robin"
            ),
            "validate_every_event": self.validate_every_event,
            "routing": routing_rows,
            "routing_identity_sha256": stable_json_sha256(routing_rows),
            "routing_trace_work_proxy": {
                "definition": (
                    "fresh-prefill attended contexts + D decode attended "
                    "contexts + linear retained-prefix/fresh/output terms"
                ),
                "work_by_node": route_work_by_node,
                "max_over_min": (
                    maximum_work / minimum_work
                    if minimum_work else None
                ),
            },
            "call_specs": spec_rows,
            "call_specs_identity_sha256": stable_json_sha256(spec_rows),
            "metrics": asdict(self.metrics),
            "current_ns": self.current_ns,
            "finished": self._finished,
            "completion_order": completed_identities,
            "completed_requests": [
                asdict(request) for request in self.completed_requests],
            "nodes": [node.report() for node in self.nodes],
        }
        if self._finished:
            expected_identities = [
                spec.completion_identity for spec in self.call_specs]
            call_drain = asdict(full_drain_hashes(
                expected_identities,
                completed_identities,
            ))
            expected_sessions = [
                row["session_id"] for row in routing_rows]
            session_drain = asdict(full_drain_hashes(
                expected_sessions,
                self._completed_session_order,
            ))
            result["full_drain"] = call_drain
            result["call_full_drain"] = call_drain
            result["session_full_drain"] = session_drain
        return result


DualStrictInfiniteHBMOracleSystem = DualStrictInfiniteHBMOracle


__all__ = [
    "DUAL_ORACLE_NODE_COUNT",
    "ROUTE_BALANCED_TRACE_WORK",
    "ROUTE_OFFER_RR",
    "SUPPORTED_ROUTE_POLICIES",
    "DualOracleCallSpec",
    "DualOracleDeadlockError",
    "DualOracleMetrics",
    "DualStrictInfiniteHBMOracle",
    "DualStrictInfiniteHBMOracleSystem",
]

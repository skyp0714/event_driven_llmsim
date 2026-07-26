"""Two independent finite-HBM P4D4 servers with CPU/SSD KV tiering.

The global event contract intentionally matches the strict dual-oracle
runner: routes and request IDs are frozen before simulation, while each
successor is released at predecessor user completion plus its tool gap.
Every co-timed event is observed on both physical servers before either
server receives its next scheduling boundary.

The two servers share no resource calendar or capacity ledger.  Identical
resource names would still be harmless because calendars are node-local, but
the underlying transfer model also includes ``node_id`` in every physical
resource name to make accidental sharing visible in reports and tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .gpu_pd_dual_oracle import (
    DUAL_ORACLE_NODE_COUNT,
    ROUTE_BALANCED_TRACE_WORK,
    ROUTE_OFFER_RR,
    SUPPORTED_ROUTE_POLICIES,
    DualOracleCallSpec,
    DualStrictInfiniteHBMOracle,
)
from .gpu_pd_latency import P4D4GPUHardware
from .gpu_pd_tier_lifecycle import (
    RESTORE_EXECUTION_BULK,
    SUPPORTED_TIER_POLICIES,
)
from .gpu_pd_tiered_node import (
    FiniteHBMTieredP4D4Node,
    TieredCallState,
    TieredNodeCall,
)
from .hbf_comparison_metrics import CompletedRequest
from .hbf_comparison_workload import ScheduledSession


DUAL_TIERED_NODE_COUNT = DUAL_ORACLE_NODE_COUNT
DualTieredCallSpec = DualOracleCallSpec


class DualTieredDeadlockError(RuntimeError):
    """Raised when unfinished dual-tiered work has no future event."""


@dataclass
class DualTieredMetrics:
    scheduled_sessions: int = 0
    scheduled_calls: int = 0
    released_calls: int = 0
    completed_calls: int = 0
    event_timestamps: int = 0
    fixed_point_rounds: int = 0
    max_release_heap: int = 0


class DualFiniteHBMTieredBaseline(DualStrictInfiniteHBMOracle):
    """Global runner for two independent finite-HBM tiered P4D4 nodes."""

    def __init__(
            self, *, repo_root: Path, hardware: P4D4GPUHardware,
            policy: str,
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
            validate_every_event: bool = True,
            route_policy: str = ROUTE_OFFER_RR) -> None:
        if policy not in SUPPORTED_TIER_POLICIES:
            raise ValueError(f"unsupported tier policy {policy!r}")
        if not isinstance(validate_every_event, bool):
            raise ValueError("validate_every_event must be a boolean")
        if route_policy not in SUPPORTED_ROUTE_POLICIES:
            raise ValueError(
                "route_policy must be one of "
                f"{sorted(SUPPORTED_ROUTE_POLICIES)}")

        # This initialization mirrors the common global state in the strict
        # oracle without constructing and discarding oracle nodes.
        self.repo_root = Path(repo_root)
        self.hardware = hardware
        self.policy = policy
        self.restore_execution_mode = restore_execution_mode
        self.validate_every_event = validate_every_event
        self.route_policy = route_policy
        self.nodes = tuple(
            FiniteHBMTieredP4D4Node(
                repo_root=self.repo_root,
                hardware=hardware,
                node_id=node_id,
                policy=policy,
                p_capacity_bytes_per_rank=(
                    p_capacity_bytes_per_rank),
                d_capacity_bytes_per_rank=(
                    d_capacity_bytes_per_rank),
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
            for node_id in range(DUAL_TIERED_NODE_COUNT)
        )
        self.metrics = DualTieredMetrics()
        self.current_ns = 0
        self.call_specs: tuple[DualTieredCallSpec, ...] = ()
        self._spec_by_request: dict[int, DualTieredCallSpec] = {}
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
        self._runtime_calls: dict[int, TieredNodeCall] = {}
        self._completed_snapshots: dict[int, CompletedRequest] = {}
        self._completion_order: list[int] = []
        self._completed_session_order: list[str] = []
        self._loaded = False
        self._running = False
        self._finished = False
        self.assert_invariants()

    def load(
            self,
            scheduled_sessions: Iterable[ScheduledSession]) -> None:
        """Validate the finite-capacity contract, then freeze the schedule."""

        scheduled_values = list(scheduled_sessions)
        if self._loaded:
            raise RuntimeError("dual tiered schedule is already loaded")

        # Validate every node-level submission contract before the inherited
        # loader mutates IDs, routes, or release heaps.  Both nodes currently
        # have the same capacity configuration, but checking both preserves
        # this atomicity if per-node capacities are introduced later.
        for scheduled in scheduled_values:
            self._validate_scheduled_session(scheduled)
            calls = scheduled.session.calls
            for call_index, call_spec in enumerate(calls):
                probe = TieredNodeCall(
                    request_id=call_index,
                    session_id=scheduled.session.session_id,
                    call_index=call_index,
                    release_ns=scheduled.arrival_time_ns,
                    input_tokens=call_spec.input_tokens,
                    output_tokens=call_spec.output_tokens,
                    prefix_reuse_tokens=(
                        call_spec.cached_prefix_tokens),
                    has_successor=call_index + 1 < len(calls),
                )
                probe.validate()
                for node in self.nodes:
                    node._validate_capacity_contract(probe)

        super().load(scheduled_values)

    def _consume_node_completions(
            self, now_ns: int) -> list[TieredNodeCall]:
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
            pool_request = call.pool_request
            if (
                pool_request is None
                or pool_request.first_token_ns is None
                or pool_request.completion_ns is None
            ):
                raise RuntimeError(
                    "completed tiered call lacks token timestamps")

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
            if successor_id is not None:
                self._queue_release(
                    successor_id,
                    release_ns=now_ns + spec.tool_duration_ns,
                )
            else:
                self._completed_session_order.append(spec.session_id)
        return completed

    def _pop_releases(
            self, now_ns: int) -> dict[int, list[TieredNodeCall]]:
        by_node = {
            node_id: []
            for node_id in range(DUAL_TIERED_NODE_COUNT)
        }
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
            call = TieredNodeCall(
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

    def _process_timestamp(self, now_ns: int) -> None:
        if now_ns < self.current_ns:
            raise ValueError("global tiered time cannot move backwards")
        self.metrics.event_timestamps += 1
        rounds = 0
        while True:
            rounds += 1
            if rounds > self.metrics.scheduled_calls + 4:
                raise DualTieredDeadlockError(
                    "same-timestamp fixed point did not converge")

            for node in self.nodes:
                node.advance(now_ns, defer_schedule=True)
            self._consume_node_completions(now_ns)
            arrivals = self._pop_releases(now_ns)

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

    def _deadlock_detail(self) -> str:
        unreleased = sorted(
            set(self._spec_by_request) - self._released_ids)
        unfinished = sorted(
            set(self._spec_by_request) - self._completed_ids)
        node_pending = {
            node.node_id: [
                call.request_id
                for call in node.calls.values()
                if call.state != TieredCallState.INTERNAL_COMPLETE
            ]
            for node in self.nodes
        }
        return (
            f"policy={self.policy}, unreleased={unreleased[:8]}, "
            f"unfinished={unfinished[:8]}, "
            f"node_pending={node_pending}"
        )

    def run(
            self,
            scheduled_sessions: Optional[
                Iterable[ScheduledSession]] = None,
    ) -> list[CompletedRequest]:
        """Run both finite nodes through complete internal tier drain."""

        if scheduled_sessions is not None:
            self.load(scheduled_sessions)
        if not self._loaded:
            raise RuntimeError("load a schedule before running")
        if self._running:
            raise RuntimeError("dual tiered baseline is already running")
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
                        == TieredCallState.INTERNAL_COMPLETE
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
                    raise DualTieredDeadlockError(
                        "unfinished dual-tiered work has no future event: "
                        + self._deadlock_detail())
                self._process_timestamp(next_event_ns)
        finally:
            self._running = False

        self._finished = True
        self.assert_invariants()
        return list(self.completed_requests)

    @staticmethod
    def _aggregate_metric_dicts(
            rows: Iterable[Mapping[str, int]]) -> dict[str, int]:
        values = list(rows)
        keys = {
            key
            for row in values
            for key in row
        }
        result = {}
        for key in sorted(keys):
            samples = [int(row.get(key, 0)) for row in values]
            result[key] = (
                max(samples)
                if key.startswith("max_")
                else sum(samples)
            )
        return result

    def _bottleneck_report(self) -> Mapping[str, Any]:
        per_node = []
        resource_busy_by_class: dict[str, int] = {}
        for node in self.nodes:
            node_metrics = asdict(node.metrics)
            lifecycle_metrics = asdict(node.lifecycle.metrics)
            pool_metrics = asdict(node.pool.metrics)
            resource_busy = dict(sorted(node.calendar.busy_ns.items()))
            resource_utilization = {
                resource: (
                    busy_ns / self.current_ns
                    if self.current_ns else 0.0
                )
                for resource, busy_ns in resource_busy.items()
            }
            ledgers = {
                name: {
                    "capacity_bytes": ledger.capacity_bytes,
                    "peak_used_bytes": ledger.peak_used_bytes,
                    "peak_fraction": (
                        ledger.peak_used_bytes / ledger.capacity_bytes
                    ),
                    "final_used_bytes": ledger.used_bytes,
                }
                for name, ledger in (
                    ("p", node.lifecycle.p_ledger),
                    ("d", node.lifecycle.d_ledger),
                    ("cpu", node.lifecycle.cpu_ledger),
                    ("ssd", node.lifecycle.ssd_ledger),
                )
            }
            for resource, busy_ns in resource_busy.items():
                prefix = f"gpu-node-{node.node_id}-"
                resource_class = (
                    resource.removeprefix(prefix)
                    if resource.startswith(prefix)
                    else resource
                )
                resource_busy_by_class[resource_class] = (
                    resource_busy_by_class.get(resource_class, 0)
                    + busy_ns
                )
            per_node.append({
                "node_id": node.node_id,
                "node_metrics": node_metrics,
                "lifecycle_metrics": lifecycle_metrics,
                "pool_metrics": pool_metrics,
                "resource_busy_ns": resource_busy,
                "resource_utilization": resource_utilization,
                "ledgers": ledgers,
            })
        return {
            "aggregation": (
                "sum counters/durations; max max_* high-water marks"),
            "horizon_ns": self.current_ns,
            "deferral_counter_semantics": (
                "scheduler retry attempts, not unique delayed requests"),
            "per_node": per_node,
            "aggregate": {
                "node_metrics": self._aggregate_metric_dicts(
                    row["node_metrics"] for row in per_node),
                "lifecycle_metrics": self._aggregate_metric_dicts(
                    row["lifecycle_metrics"] for row in per_node),
                "pool_metrics": self._aggregate_metric_dicts(
                    row["pool_metrics"] for row in per_node),
                "resource_busy_ns_by_class": dict(sorted(
                    resource_busy_by_class.items())),
            },
        }

    def assert_invariants(self) -> None:
        super().assert_invariants()
        if len(self.nodes) != DUAL_TIERED_NODE_COUNT:
            raise AssertionError(
                "dual tiered baseline must have exactly two nodes")
        if self.nodes[0].calendar is self.nodes[1].calendar:
            raise AssertionError(
                "physical tiered nodes share a resource calendar")
        shared_resource_names = (
            set(self.nodes[0].calendar.available_ns)
            & set(self.nodes[1].calendar.available_ns)
        )
        if shared_resource_names:
            raise AssertionError(
                "physical tiered nodes share resource names: "
                f"{sorted(shared_resource_names)}")
        for left_name, left_ledger, right_ledger in (
            (
                "p",
                self.nodes[0].lifecycle.p_ledger,
                self.nodes[1].lifecycle.p_ledger,
            ),
            (
                "d",
                self.nodes[0].lifecycle.d_ledger,
                self.nodes[1].lifecycle.d_ledger,
            ),
            (
                "cpu",
                self.nodes[0].lifecycle.cpu_ledger,
                self.nodes[1].lifecycle.cpu_ledger,
            ),
            (
                "ssd",
                self.nodes[0].lifecycle.ssd_ledger,
                self.nodes[1].lifecycle.ssd_ledger,
            ),
        ):
            if left_ledger is right_ledger:
                raise AssertionError(
                    f"physical tiered nodes share {left_name} ledger")
        for node in self.nodes:
            if (
                node.policy != self.policy
                or node.lifecycle.policy != self.policy
                or node.restore_execution_mode
                != self.restore_execution_mode
            ):
                raise AssertionError(
                    "dual tiered policy/restore propagation failed")

    def report(self) -> Mapping[str, Any]:
        result = dict(super().report())
        result.update({
            "mode": "dual_finite_hbm_p4d4_tiering",
            "policy": self.policy,
            "restore_execution_mode": self.restore_execution_mode,
            "resource_calendars": [
                node.calendar.report() for node in self.nodes
            ],
            "resource_calendar_semantics": (
                "one lossless reservation-count/byte calendar per physical "
                "GPU server; use canonical aggregate resources rather than "
                "summing rank, root, and memory copies of the same payload"
            ),
            "routing_balance_limit": (
                "balanced_trace_work uses only the oracle compute/context "
                "proxy; it is static and does not balance tier I/O, "
                "capacity deferrals, eviction, or recomputation"),
            "physical_isolation": (
                "two independent 4P4D+CPU+SSD servers; no shared "
                "calendar, HBM, CPU, SSD, PCIe, or SSD queue"),
            "bottleneck_counters": self._bottleneck_report(),
        })
        return result


DualTieredP4D4BaselineSystem = DualFiniteHBMTieredBaseline


__all__ = [
    "DUAL_TIERED_NODE_COUNT",
    "ROUTE_BALANCED_TRACE_WORK",
    "ROUTE_OFFER_RR",
    "SUPPORTED_ROUTE_POLICIES",
    "DualFiniteHBMTieredBaseline",
    "DualTieredCallSpec",
    "DualTieredDeadlockError",
    "DualTieredMetrics",
    "DualTieredP4D4BaselineSystem",
]

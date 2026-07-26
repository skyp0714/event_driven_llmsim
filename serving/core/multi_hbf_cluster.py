"""Coordinator for multiple independent eight-card HBF serving servers.

The cluster deliberately keeps one lifecycle, serving pool, and active-memory
ledger per physical HBF server.  All servers share one analytical calendar so
the GPU source PCIe root and RDMA network can remain common bottlenecks, while
server-local resources are separated by the prefixes passed to each child.

This module is analytical-only.  The external ASTRA path already has
server-scoped resource identities, but requires a separate multi-producer
callback coordinator and is therefore kept fail-closed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
)
from .hbf_full_model_lifecycle import (
    FullModelHBFLifecycle,
    LifecycleMetrics,
    PerGroupCapacityLedger,
    PlacementState,
    ResourceCalendar,
    ResumeRoute,
)
from .hbf_full_model_pool import (
    FullModelHBFServingPool,
    HBFPoolMetrics,
    HBFServingRequest,
    derive_lpddr_workspace_bytes,
)


def _sum_metric_dataclass(metric_type: type, values: Sequence[object]):
    """Return the fieldwise sum of homogeneous integer metric records."""

    return metric_type(**{
        item.name: sum(int(getattr(value, item.name)) for value in values)
        for item in fields(metric_type)
    })


@dataclass(frozen=True)
class HBFServerBundle:
    """All state owned by one physical eight-card HBF server."""

    server_index: int
    server_id: int
    layout: HBFParallelLayout
    lpddr_ledger: PerGroupCapacityLedger
    lifecycle: FullModelHBFLifecycle
    pool: FullModelHBFServingPool


class MultiHBFCluster:
    """One shared ingress feeding multiple independent HBF servers."""

    def __init__(
            self, *, repo_root: Path, hardware: HBFServerHardware,
            layouts: Sequence[HBFParallelLayout],
            resource_calendar: ResourceCalendar,
            gpu_source_root_bandwidth_gbps: float,
            max_num_batched_tokens: int,
            max_num_seqs: int,
            max_prefill_chunk_tokens: int,
            band: str = "central",
            validate_every_event: bool = True,
            server_id_base: int = 0) -> None:
        if not layouts:
            raise ValueError("multi-HBF cluster requires at least one server")
        if (
            not isinstance(server_id_base, int)
            or isinstance(server_id_base, bool)
            or server_id_base < 0
        ):
            raise ValueError("server_id_base must be a non-negative integer")
        if not isinstance(resource_calendar, ResourceCalendar):
            raise TypeError("multi-HBF cluster requires a ResourceCalendar")
        hardware.validate()

        bundles = []
        for server_index, layout in enumerate(layouts):
            if not isinstance(layout, HBFParallelLayout):
                raise TypeError(
                    "multi-HBF layouts must be HBFParallelLayout values")
            layout.validate(hardware.card_count)
            server_id = server_id_base + server_index
            workspace = derive_lpddr_workspace_bytes(
                layout,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
            )
            kv_capacity = (
                hardware.lpddr_capacity_bytes_per_card - workspace)
            if kv_capacity <= 0:
                raise ValueError(
                    "active-memory capacity does not fit the derived "
                    f"workspace for HBF server {server_index}")
            ledger = PerGroupCapacityLedger(
                group_count=layout.replicas,
                capacity_bytes=kv_capacity,
            )
            resource_prefix = f"hbf-server-{server_id}-"
            lifecycle = FullModelHBFLifecycle(
                hardware=hardware,
                layout=layout,
                resource_calendar=resource_calendar,
                lpddr_ledger=ledger,
                gpu_source_root_bandwidth_gbps=(
                    gpu_source_root_bandwidth_gbps),
                validate_every_event=validate_every_event,
                execution_backend="analytical_calendar",
                server_id=server_id,
                analytical_resource_prefix=resource_prefix,
            )
            pool = FullModelHBFServingPool(
                repo_root=repo_root,
                hardware=hardware,
                layout=layout,
                resource_calendar=resource_calendar,
                lpddr_ledger=ledger,
                placement_resolver=lifecycle.placement_snapshot,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                max_prefill_chunk_tokens=max_prefill_chunk_tokens,
                band=band,
                validate_every_event=validate_every_event,
                retain_detailed_history=validate_every_event,
                execution_backend="analytical_calendar",
                server_id=server_id,
                analytical_resource_prefix=resource_prefix,
            )
            bundles.append(HBFServerBundle(
                server_index=server_index,
                server_id=server_id,
                layout=layout,
                lpddr_ledger=ledger,
                lifecycle=lifecycle,
                pool=pool,
            ))

        self.repo_root = Path(repo_root)
        self.hardware = hardware
        self.calendar = resource_calendar
        self.bundles = tuple(bundles)
        self.validate_every_event = validate_every_event
        self.session_server_index: dict[str, int] = {}
        self.lifecycle = MultiHBFLifecycleFacade(self)
        self.pool = MultiHBFPoolFacade(self)

    @property
    def server_count(self) -> int:
        return len(self.bundles)

    def bundle_for_session(self, session_id: str) -> HBFServerBundle:
        try:
            return self.bundles[self.session_server_index[session_id]]
        except KeyError as exc:
            raise KeyError(
                f"session {session_id!r} has no HBF server assignment"
            ) from exc

    @staticmethod
    def _pool_live_work(bundle: HBFServerBundle) -> int:
        return sum(
            len(worker.waiting)
            + len(worker.prefill_drain)
            + len(worker.active_decode)
            + (1 if worker.inflight is not None else 0)
            + (1 if worker.pending_launch_ns is not None else 0)
            for worker in bundle.pool.workers
        )

    @staticmethod
    def _live_session_count(bundle: HBFServerBundle) -> int:
        return sum(
            record.state != PlacementState.ENDED
            for record in bundle.lifecycle.sessions.values()
        )

    @staticmethod
    def _storage_fraction(bundle: HBFServerBundle) -> float:
        capacity = bundle.lifecycle.usable_bytes_per_card
        if capacity <= 0:
            return 1.0
        peak = max(
            (
                max(card_bytes.values(), default=0)
                for card_bytes in
                bundle.lifecycle._reserved_bytes_by_card.values()
            ),
            default=0,
        )
        return peak / capacity

    def server_score(self, server_index: int) -> tuple[float, ...]:
        """Return a deterministic causal destination score."""

        bundle = self.bundles[server_index]
        worker_count = len(bundle.pool.workers)
        return (
            self._pool_live_work(bundle) / worker_count,
            self._live_session_count(bundle) / worker_count,
            self._storage_fraction(bundle),
            float(server_index),
        )

    def choose_server(self) -> int:
        return min(
            range(self.server_count),
            key=self.server_score,
        )

    def normalized_compute_pressure(self) -> float:
        total_workers = sum(
            len(bundle.pool.workers) for bundle in self.bundles)
        if total_workers == 0:
            return 0.0
        return (
            sum(self._pool_live_work(bundle) for bundle in self.bundles)
            / total_workers
        )

    def report(self) -> dict[str, Any]:
        return {
            "mode": "multi_hbf_cluster",
            "server_count": self.server_count,
            "session_server_index": dict(sorted(
                self.session_server_index.items())),
            "server_scores": {
                bundle.server_index: list(
                    self.server_score(bundle.server_index))
                for bundle in self.bundles
            },
            "shared_resource_calendar": self.calendar.report(),
            "servers": {
                bundle.server_index: {
                    "server_id": bundle.server_id,
                    "layout": asdict(bundle.layout),
                    "lifecycle": bundle.lifecycle.report(),
                    "pool": bundle.pool.report(),
                }
                for bundle in self.bundles
            },
        }


class MultiHBFLifecycleFacade:
    """Lifecycle-compatible view over the cluster's child managers."""

    def __init__(self, cluster: MultiHBFCluster) -> None:
        self.cluster = cluster
        self.calendar = cluster.calendar
        self.execution_backend = "analytical_calendar"
        self.server_id = None
        self._external_outbox = ()
        self._external_pending: dict[str, object] = {}
        self._external_issued_job_ids: set[str] = set()
        self._external_completed_job_ids: set[str] = set()

    @property
    def current_ns(self) -> int:
        values = {
            bundle.lifecycle.current_ns
            for bundle in self.cluster.bundles
        }
        if len(values) != 1:
            raise RuntimeError(
                "multi-HBF lifecycle clocks are inconsistent")
        return next(iter(values))

    @property
    def sessions(self) -> dict[str, object]:
        return {
            session_id: self.cluster.bundles[
                server_index].lifecycle.sessions[session_id]
            for session_id, server_index
            in self.cluster.session_server_index.items()
        }

    @property
    def metrics(self) -> LifecycleMetrics:
        return _sum_metric_dataclass(
            LifecycleMetrics,
            [bundle.lifecycle.metrics
             for bundle in self.cluster.bundles],
        )

    @property
    def gpu_source_root_bandwidth_gbps(self) -> float:
        return self.cluster.bundles[
            0].lifecycle.gpu_source_root_bandwidth_gbps

    def register_session(
            self, session_id: str, *, now_ns: int = 0):
        if session_id in self.cluster.session_server_index:
            raise ValueError(f"duplicate session_id={session_id!r}")
        server_index = self.cluster.choose_server()
        record = self.cluster.bundles[
            server_index].lifecycle.register_session(
                session_id, now_ns=now_ns)
        self.cluster.session_server_index[session_id] = server_index
        return record

    def server_index_for_session(self, session_id: str) -> int:
        return self.cluster.session_server_index[session_id]

    def route_resume(self, session_id: str, **kwargs) -> ResumeRoute:
        self.advance(kwargs["now_ns"])
        return self.cluster.bundle_for_session(
            session_id).lifecycle.route_resume(session_id, **kwargs)

    def complete_gpu_turn(self, session_id: str, **kwargs):
        self.advance(kwargs["now_ns"])
        return self.cluster.bundle_for_session(
            session_id).lifecycle.complete_gpu_turn(
                session_id, **kwargs)

    def start_migration(self, session_id: str, **kwargs):
        self.advance(kwargs["now_ns"])
        return self.cluster.bundle_for_session(
            session_id).lifecycle.start_migration(
                session_id, **kwargs)

    def gpu_ready_pressure_reclaimable(
            self, session_id: str) -> bool:
        return self.cluster.bundle_for_session(
            session_id).lifecycle.gpu_ready_pressure_reclaimable(
                session_id)

    def evict_oldest_gpu_ready_for_hbm_pressure(
            self, session_ids: Sequence[str], *,
            now_ns: int):
        self.advance(now_ns)
        candidates = []
        for session_id in set(session_ids):
            bundle = self.cluster.bundle_for_session(session_id)
            record = bundle.lifecycle.sessions[session_id]
            if bundle.lifecycle.gpu_ready_pressure_reclaimable(
                    session_id):
                candidates.append((
                    record.last_access_ns,
                    record.session_id,
                    bundle,
                ))
        if not candidates:
            return None
        _, session_id, bundle = min(candidates)
        return bundle.lifecycle.evict_oldest_gpu_ready_for_hbm_pressure(
            (session_id,), now_ns=now_ns)

    def complete_hbf_turn(self, session_id: str, **kwargs):
        self.advance(kwargs["now_ns"])
        return self.cluster.bundle_for_session(
            session_id).lifecycle.complete_hbf_turn(
                session_id, **kwargs)

    def placement_snapshot(
            self, session_id: str) -> tuple[int, int, int]:
        return self.cluster.bundle_for_session(
            session_id).lifecycle.placement_snapshot(session_id)

    def advance(self, now_ns: int) -> None:
        for bundle in self.cluster.bundles:
            bundle.lifecycle.advance(now_ns)

    def next_completion_ns(self) -> Optional[int]:
        values = [
            value
            for value in (
                bundle.lifecycle.next_completion_ns()
                for bundle in self.cluster.bundles
            )
            if value is not None
        ]
        return min(values) if values else None

    def has_pending_external(self) -> bool:
        return False

    def drain_external_dispatches(self):
        raise RuntimeError(
            "multi-HBF cluster does not support external ASTRA")

    def assert_invariants(self) -> None:
        indexed_sessions = set(self.cluster.session_server_index)
        owned_sessions: dict[str, int] = {}
        for bundle in self.cluster.bundles:
            for session_id in bundle.lifecycle.sessions:
                if session_id in owned_sessions:
                    raise AssertionError(
                        "session is owned by multiple HBF servers")
                owned_sessions[session_id] = bundle.server_index
        if indexed_sessions != set(owned_sessions):
            raise AssertionError(
                "multi-HBF session assignment index is inconsistent")
        if any(
                self.cluster.session_server_index[session_id]
                != server_index
                for session_id, server_index in owned_sessions.items()):
            raise AssertionError(
                "multi-HBF session assignment points at the wrong server")
        for bundle in self.cluster.bundles:
            bundle.lifecycle.assert_invariants()

    def report(self) -> Mapping[str, Any]:
        return {
            "mode": "multi_hbf_lifecycle",
            "server_count": self.cluster.server_count,
            "metrics": asdict(self.metrics),
            "servers": {
                bundle.server_index: bundle.lifecycle.report()
                for bundle in self.cluster.bundles
            },
            "external_undrained_dispatch_count": 0,
        }


class MultiHBFPoolFacade:
    """Serving-pool-compatible view over all HBF server pools."""

    def __init__(self, cluster: MultiHBFCluster) -> None:
        self.cluster = cluster
        self.calendar = cluster.calendar
        first = cluster.bundles[0].pool
        self.hardware = cluster.hardware
        self.layout = cluster.bundles[0].layout
        self.execution_backend = "analytical_calendar"
        self.server_id = None
        self.max_num_batched_tokens = first.max_num_batched_tokens
        self.max_num_seqs = first.max_num_seqs
        self.max_prefill_chunk_tokens = (
            first.max_prefill_chunk_tokens)
        self.validate_every_event = cluster.validate_every_event
        self._external_outbox = ()
        self._external_pending: dict[str, object] = {}
        self._external_issued_job_ids: set[str] = set()
        self._external_completed_job_ids: set[str] = set()

    @property
    def current_ns(self) -> int:
        values = {
            bundle.pool.current_ns
            for bundle in self.cluster.bundles
        }
        if len(values) != 1:
            raise RuntimeError("multi-HBF pool clocks are inconsistent")
        return next(iter(values))

    @property
    def metrics(self) -> HBFPoolMetrics:
        return _sum_metric_dataclass(
            HBFPoolMetrics,
            [bundle.pool.metrics for bundle in self.cluster.bundles],
        )

    @property
    def workers(self) -> tuple[object, ...]:
        return tuple(
            worker
            for bundle in self.cluster.bundles
            for worker in bundle.pool.workers
        )

    @property
    def requests(self) -> dict[int, HBFServingRequest]:
        result: dict[int, HBFServingRequest] = {}
        for bundle in self.cluster.bundles:
            overlap = set(result) & set(bundle.pool.requests)
            if overlap:
                raise RuntimeError(
                    f"duplicate request IDs across HBF servers: {overlap}")
            result.update(bundle.pool.requests)
        return result

    def submit_many(
            self, requests: Iterable[HBFServingRequest], *,
            now_ns: int, defer_schedule: bool = False) -> None:
        if not isinstance(defer_schedule, bool):
            raise ValueError("defer_schedule must be a boolean")
        # Keep idle child clocks aligned and make every co-timed arrival
        # visible before any server launches its next batch.
        self.advance(now_ns, defer_schedule=True)
        values = list(requests)
        existing_request_ids = set(self.requests)
        seen_request_ids: set[int] = set()
        by_server: dict[int, list[HBFServingRequest]] = {
            bundle.server_index: []
            for bundle in self.cluster.bundles
        }
        for request in values:
            if (
                    request.request_id in existing_request_ids
                    or request.request_id in seen_request_ids):
                raise ValueError(
                    f"duplicate request_id={request.request_id}")
            seen_request_ids.add(request.request_id)
            server_index = self.cluster.session_server_index[
                request.session_id]
            by_server[server_index].append(request)
        for server_index, values in by_server.items():
            if values:
                values.sort(key=lambda item: item.request_id)
                self.cluster.bundles[
                    server_index].pool.submit_many(
                        values, now_ns=now_ns,
                        defer_schedule=True)
        if not defer_schedule:
            self.flush_scheduling(now_ns)

    def advance(
            self, now_ns: int, *,
            defer_schedule: bool = False) -> None:
        for bundle in self.cluster.bundles:
            bundle.pool.advance(
                now_ns, defer_schedule=defer_schedule)

    def flush_scheduling(self, now_ns: int) -> None:
        for bundle in self.cluster.bundles:
            bundle.pool.flush_scheduling(now_ns)

    def next_event_ns(self) -> Optional[int]:
        values = [
            value
            for value in (
                bundle.pool.next_event_ns()
                for bundle in self.cluster.bundles
            )
            if value is not None
        ]
        return min(values) if values else None

    def pop_completed(self) -> list[HBFServingRequest]:
        values = [
            request
            for bundle in self.cluster.bundles
            for request in bundle.pool.pop_completed()
        ]
        values.sort(key=lambda item: (
            item.completion_ns,
            item.request_id,
        ))
        return values

    def has_pending_external_dispatches(self) -> bool:
        return False

    def drain_external_dispatches(self):
        raise RuntimeError(
            "multi-HBF cluster does not support external ASTRA")

    def assert_invariants(self) -> None:
        seen_requests: set[int] = set()
        for bundle in self.cluster.bundles:
            bundle.pool.assert_invariants()
            overlap = seen_requests & set(bundle.pool.requests)
            if overlap:
                raise AssertionError(
                    "request is owned by multiple HBF servers")
            seen_requests.update(bundle.pool.requests)

    def report(self) -> Mapping[str, Any]:
        return {
            "mode": "multi_hbf_pool",
            "server_count": self.cluster.server_count,
            "metrics": asdict(self.metrics),
            "normalized_compute_pressure": (
                self.cluster.normalized_compute_pressure()),
            "servers": {
                bundle.server_index: bundle.pool.report()
                for bundle in self.cluster.bundles
            },
            "external_undrained_dispatch_count": 0,
        }

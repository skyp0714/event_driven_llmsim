"""Project one full-model HBF batch onto ASTRA shared-resource stages.

This is the first, deliberately narrow, full-model HBF-to-ASTRA vertical
slice.  It turns :class:`HBFModelBatchLatency` into the exact stage schema
accepted by ``Controller.hbf_background_command`` and by ASTRA-Sim's
``hbf-background-v1`` engine.  Card-local work is parallel across a TP group,
collectives are whole-group barriers, and every resource name is scoped to
the selected HBF server and physical card/root route.  The same PCIe
card/root/inter-root names are consumed by lifecycle migrations.

``aggregate-v1`` has important limits:

* kernel-family totals are grouped into six phases instead of one Chakra node
  per transformer operation;
* the three collective totals are analytical stages on explicit HBF fabric
  resources, not native ASTRA network collectives;
* HBF/LPDDR byte attribution is an exact proportional audit split, while the
  aggregate latency model remains the source of stage runtime;
* no overlap inside a card is inferred from aggregate totals.

Those limits make aggregate-v1 suitable for causal scheduler integration and
shared-resource contention, but not for claiming per-kernel ASTRA cycle
agreement.  ``ordered-v2`` is the additive, per-operation path: it expands
the immutable execution plan in order, forks every kernel across physical
cards, and joins every collective at a replica-wide shared-fabric barrier.
For context-striped TP8 it consumes the plan's exact odd/even per-card
attention ledger rather than multiplying one critical-rank scalar by eight.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cached_property
import heapq
import json
import math
from typing import Any, Sequence

from .hbf_full_model_latency import (
    HBFCollectiveExecutionOp,
    HBFKernelExecutionOp,
    HBFModelBatchExecutionPlan,
    HBFModelBatchLatency,
    HBFParallelLayout,
    HBFServerHardware,
)
from .hbf_pcie_topology import HBFPCIeTopology


PROJECTION_SCHEMA = "hbf-full-model-astra-v1/aggregate-v1"
PROJECTION_FIDELITY = "aggregate-v1"
ORDERED_V2_SCHEMA = "hbf-full-model-astra-v2/ordered-v2"
ORDERED_V2_FIDELITY = "per-operation-ordered-v2"
ORDERED_V2_STAGE_LIMIT = 1_000_000
ASTRA_NAMED_RESOURCE_TIMING_SEMANTICS = {
    "astra_dependency_critical_path_ns": (
        "sum of dependency-only DAG critical paths; named-resource "
        "reservations are excluded"
    ),
    "astra_solo_resource_serialized_completion_ns": (
        "sum of isolated-job ASTRA completion times with FIFO "
        "named-resource reservations, including conflicts within each job"
    ),
    "astra_actual_resource_serialized_completion_ns": (
        "sum of actual ASTRA callback elapsed times"
    ),
    "astra_internal_resource_serialization_wait_ns": (
        "solo resource-serialized completion minus dependency critical path"
    ),
    "astra_signed_interference_delta_ns": (
        "actual ASTRA callback elapsed time minus isolated-job "
        "resource-serialized completion; this signed value may be negative "
        "when inter-job event ordering reduces a job's internal serialization"
    ),
    "astra_completion_elapsed_ns": (
        "compatibility alias for actual resource-serialized completion"
    ),
    "astra_resource_delay_ns": (
        "compatibility total delay: actual completion minus dependency "
        "critical path, equal to internal serialization plus signed "
        "interference delta"
    ),
}
AGGREGATE_V1_LIMITATIONS = (
    "kernel-family totals are grouped rather than emitted per transformer "
    "layer",
    "collectives use analytical shared-resource stages rather than native "
    "ASTRA network collectives",
    "media bytes are apportioned for exact auditing but do not derive stage "
    "runtime",
    "intra-card compute and media overlap is not reconstructed",
)
ORDERED_V2_LIMITATIONS = (
    "collectives use analytical shared-resource barrier stages rather than "
    "native ASTRA COMM_COLL nodes",
    "PCIe routes are collapsed analytical stages rather than native ASTRA "
    "PCIe endpoints and links",
    "kernel runtimes remain calibrated analytical inputs rather than "
    "ASTRA-derived device compute cycles",
    "each source kernel already collapses intra-kernel compute and media "
    "overlap into one runtime",
)

_LAYOUT_CONTRACTS = {
    "dp8": (1, 8),
    "tp4": (4, 2),
    "tp8": (8, 1),
    "tp8_context": (8, 1),
}
_LOCAL_PHASE_FIELDS = (
    ("embedding", "embedding_ns"),
    ("dense", "dense_ns"),
    ("attention", "attention_ns"),
    ("router", "router_ns"),
    ("moe", "moe_ns"),
    ("final", "final_ns"),
)
_PRE_COLLECTIVE_PHASES = _LOCAL_PHASE_FIELDS[:-1]
_FINAL_PHASE = _LOCAL_PHASE_FIELDS[-1]
_COLLECTIVE_FIELDS = (
    ("tp_allreduce", "tp_allreduce_ns"),
    ("ep_allgather", "ep_allgather_ns"),
    ("ep_reduce_scatter", "ep_reduce_scatter_ns"),
)
_AGGREGATE_COLLECTIVE_TYPES = {
    "tp_allreduce": "ALLREDUCE",
    "ep_allgather": "ALLGATHER",
    "ep_reduce_scatter": "REDUCE_SCATTER",
}
_PAIR_COLLECTIVE_LATENCY_FIELDS = (
    "pair_query_exchange_ns",
    "pair_softmax_partial_exchange_ns",
)


class HBFModelAstraProjectionError(ValueError):
    """Raised when aggregate HBF work cannot be projected unambiguously."""


@dataclass(frozen=True)
class HBFNamedResourceTiming:
    """Solo-job timing under ASTRA's FIFO named-resource reservation."""

    dependency_critical_path_ns: int
    resource_serialized_completion_ns: int
    critical_path_internal_wait_ns: int
    cumulative_stage_resource_wait_ns: int


@dataclass(frozen=True)
class HBFAstraTimingAccounting:
    """Strict callback accounting relative to dependency and solo timing.

    Inter-job scheduling can change same-job resource acquisition order.
    Therefore actual elapsed time is bounded below by the dependency path,
    but it is not bounded below by the descriptor-order solo completion.
    """

    dependency_critical_path_ns: int
    solo_resource_serialized_completion_ns: int
    actual_resource_serialized_completion_ns: int

    def __post_init__(self) -> None:
        for name, value in (
            (
                "dependency_critical_path_ns",
                self.dependency_critical_path_ns,
            ),
            (
                "solo_resource_serialized_completion_ns",
                self.solo_resource_serialized_completion_ns,
            ),
            (
                "actual_resource_serialized_completion_ns",
                self.actual_resource_serialized_completion_ns,
            ),
        ):
            if type(value) is not int or value < 0:
                raise HBFModelAstraProjectionError(
                    f"{name} must be a finite non-negative integer")
        if (
            self.solo_resource_serialized_completion_ns
            < self.dependency_critical_path_ns
        ):
            raise HBFModelAstraProjectionError(
                "solo resource-serialized completion precedes the "
                "dependency critical path")
        if (
            self.actual_resource_serialized_completion_ns
            < self.dependency_critical_path_ns
        ):
            raise HBFModelAstraProjectionError(
                "actual resource-serialized completion precedes the "
                "dependency critical path")

    @property
    def resource_delay_ns(self) -> int:
        return (
            self.actual_resource_serialized_completion_ns
            - self.dependency_critical_path_ns
        )

    @property
    def internal_resource_serialization_wait_ns(self) -> int:
        return (
            self.solo_resource_serialized_completion_ns
            - self.dependency_critical_path_ns
        )

    @property
    def signed_interference_delta_ns(self) -> int:
        return (
            self.actual_resource_serialized_completion_ns
            - self.solo_resource_serialized_completion_ns
        )


def validate_hbf_astra_timing_metrics(metrics: object) -> None:
    """Validate accumulated timing types, bounds, and exact algebra."""

    names = (
        "astra_completion_elapsed_ns",
        "astra_resource_delay_ns",
        "astra_dependency_critical_path_ns",
        "astra_solo_resource_serialized_completion_ns",
        "astra_actual_resource_serialized_completion_ns",
        "astra_internal_resource_serialization_wait_ns",
        "astra_signed_interference_delta_ns",
    )
    values = {}
    for name in names:
        try:
            value = getattr(metrics, name)
        except AttributeError as exc:
            raise HBFModelAstraProjectionError(
                f"timing metrics are missing {name}") from exc
        if type(value) is not int:
            raise HBFModelAstraProjectionError(
                f"{name} must be a finite integer")
        values[name] = value

    accounting = HBFAstraTimingAccounting(
        dependency_critical_path_ns=values[
            "astra_dependency_critical_path_ns"],
        solo_resource_serialized_completion_ns=values[
            "astra_solo_resource_serialized_completion_ns"],
        actual_resource_serialized_completion_ns=values[
            "astra_actual_resource_serialized_completion_ns"],
    )
    expected = {
        "astra_completion_elapsed_ns": (
            accounting.actual_resource_serialized_completion_ns
        ),
        "astra_resource_delay_ns": accounting.resource_delay_ns,
        "astra_internal_resource_serialization_wait_ns": (
            accounting.internal_resource_serialization_wait_ns
        ),
        "astra_signed_interference_delta_ns": (
            accounting.signed_interference_delta_ns
        ),
    }
    for name, value in expected.items():
        if values[name] != value:
            raise HBFModelAstraProjectionError(
                "ASTRA timing metric accounting mismatch: "
                f"{name}={values[name]}, expected={value}")
    if accounting.resource_delay_ns != (
            accounting.internal_resource_serialization_wait_ns
            + accounting.signed_interference_delta_ns):
        raise AssertionError("ASTRA timing accounting identity changed")


def hbf_dependency_critical_path_ns(stages: Sequence[object]) -> int:
    """Return the DAG path length without named-resource serialization."""

    by_id = {stage.stage_id: stage for stage in stages}
    if len(by_id) != len(stages):
        raise HBFModelAstraProjectionError(
            "projection contains duplicate stage ids")
    remaining = {
        stage.stage_id: len(stage.dependencies)
        for stage in stages
    }
    dependents = {stage_id: [] for stage_id in by_id}
    for stage in stages:
        for dependency in stage.dependencies:
            if dependency not in dependents:
                raise HBFModelAstraProjectionError(
                    "projection stage graph contains an unknown dependency")
            dependents[dependency].append(stage.stage_id)
    finish: dict[str, int] = {}
    ready = deque(
        stage.stage_id
        for stage in stages
        if remaining[stage.stage_id] == 0
    )
    while ready:
        stage_id = ready.popleft()
        stage = by_id[stage_id]
        start = max(
            (finish[dependency]
             for dependency in stage.dependencies),
            default=0,
        )
        finish[stage_id] = start + stage.runtime_ns
        for dependent in dependents[stage_id]:
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                ready.append(dependent)
    if len(finish) != len(by_id):
        raise HBFModelAstraProjectionError(
            "projection stage graph contains a cycle")
    return max(finish.values(), default=0)


def hbf_solo_named_resource_timing(
        stages: Sequence[object]) -> HBFNamedResourceTiming:
    """Mirror one ASTRA HBF job on an initially idle resource calendar.

    The C++ engine submits descriptor-order roots, reserves every named
    resource atomically, and immediately submits a dependent when its final
    dependency completion event fires.  The event sequence below mirrors
    that contract.  It intentionally captures resource conflicts among
    stages of the same job; it does not include other foreground or
    background jobs.
    """

    materialized = tuple(stages)
    if not materialized:
        raise HBFModelAstraProjectionError(
            "projection must contain at least one stage")
    index_by_id = {
        stage.stage_id: index
        for index, stage in enumerate(materialized)
    }
    if len(index_by_id) != len(materialized):
        raise HBFModelAstraProjectionError(
            "projection contains duplicate stage ids")
    remaining = [len(stage.dependencies) for stage in materialized]
    dependents: list[list[int]] = [
        [] for _ in materialized
    ]
    for index, stage in enumerate(materialized):
        for dependency in stage.dependencies:
            try:
                dependency_index = index_by_id[dependency]
            except KeyError as exc:
                raise HBFModelAstraProjectionError(
                    "projection stage graph contains an unknown dependency"
                ) from exc
            dependents[dependency_index].append(index)

    resource_available_ns: dict[str, int] = {}
    stage_finish_ns: list[int | None] = [None] * len(materialized)
    events: list[tuple[int, int, int]] = []
    event_sequence = 0
    cumulative_wait_ns = 0

    def submit(stage_index: int, ready_ns: int) -> None:
        nonlocal event_sequence, cumulative_wait_ns
        stage = materialized[stage_index]
        start_ns = max(
            ready_ns,
            *(resource_available_ns.get(resource, 0)
              for resource in stage.resources),
        )
        finish_ns = start_ns + stage.runtime_ns
        cumulative_wait_ns += start_ns - ready_ns
        for resource in stage.resources:
            resource_available_ns[resource] = finish_ns
        stage_finish_ns[stage_index] = finish_ns
        heapq.heappush(
            events,
            (finish_ns, event_sequence, stage_index),
        )
        event_sequence += 1

    for index, dependency_count in enumerate(remaining):
        if dependency_count == 0:
            submit(index, 0)

    completed = 0
    while events:
        finish_ns, _, stage_index = heapq.heappop(events)
        completed += 1
        for dependent_index in dependents[stage_index]:
            remaining[dependent_index] -= 1
            if remaining[dependent_index] == 0:
                submit(dependent_index, finish_ns)
    if completed != len(materialized):
        raise HBFModelAstraProjectionError(
            "projection stage graph contains a cycle")

    dependency_ns = hbf_dependency_critical_path_ns(materialized)
    resource_ns = max(
        (value for value in stage_finish_ns if value is not None),
        default=0,
    )
    if resource_ns < dependency_ns:
        raise AssertionError(
            "named-resource completion preceded the dependency path")
    return HBFNamedResourceTiming(
        dependency_critical_path_ns=dependency_ns,
        resource_serialized_completion_ns=resource_ns,
        critical_path_internal_wait_ns=resource_ns - dependency_ns,
        cumulative_stage_resource_wait_ns=cumulative_wait_ns,
    )


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HBFModelAstraProjectionError(
            f"{name} must be a non-negative integer")
    return value


def _positive_int(name: str, value: object) -> int:
    value = _nonnegative_int(name, value)
    if value == 0:
        raise HBFModelAstraProjectionError(
            f"{name} must be a positive integer")
    return value


def _validate_layout(layout: HBFParallelLayout, card_count: int) -> None:
    layout.validate(card_count)
    expected = _LAYOUT_CONTRACTS.get(layout.key)
    if expected is None:
        raise HBFModelAstraProjectionError(
            f"layout key must be one of {sorted(_LAYOUT_CONTRACTS)}")
    actual = (layout.tp_size, layout.replicas)
    if actual != expected:
        raise HBFModelAstraProjectionError(
            f"layout {layout.key!r} must be tp={expected[0]}, "
            f"replicas={expected[1]}, got tp={actual[0]}, "
            f"replicas={actual[1]}")


@dataclass(frozen=True)
class HBFReplicaPlacement:
    """Physical cards and PCIe roots owned by one independent replica."""

    replica_id: int
    card_ids: tuple[int, ...]
    pcie_root_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "replica_id": self.replica_id,
            "card_ids": list(self.card_ids),
            "pcie_root_ids": list(self.pcie_root_ids),
        }


@dataclass(frozen=True)
class HBFServerPlacement:
    """Validated, exhaustive placement for an eight-card HBF server."""

    layout: str
    tp_size: int
    replicas: int
    server_id: int
    pcie_topology: HBFPCIeTopology
    groups: tuple[HBFReplicaPlacement, ...]

    def group(self, replica_id: int) -> HBFReplicaPlacement:
        replica = _nonnegative_int("replica_id", replica_id)
        if replica >= len(self.groups):
            raise HBFModelAstraProjectionError(
                f"replica_id={replica} is outside 0..{len(self.groups) - 1}")
        group = self.groups[replica]
        if group.replica_id != replica:
            raise HBFModelAstraProjectionError(
                "server placement group order is not canonical")
        return group

    def as_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "tp_size": self.tp_size,
            "replicas": self.replicas,
            "server_id": self.server_id,
            "pcie_topology": self.pcie_topology.as_dict(),
            "groups": [group.as_dict() for group in self.groups],
        }


def build_hbf_server_placement(
        *, hardware: HBFServerHardware, layout: HBFParallelLayout,
        server_id: int = 0) -> HBFServerPlacement:
    """Map DP8, two-TP4, or TP8 replicas to disjoint physical cards."""

    hardware.validate()
    _validate_layout(layout, hardware.card_count)
    server = _nonnegative_int("server_id", server_id)
    topology = HBFPCIeTopology.from_hardware(
        hardware, server_id=server)
    topology.validate_layout(
        layout_key=layout.key,
        tp_size=layout.tp_size,
        replicas=layout.replicas,
    )
    groups = []
    covered_cards: list[int] = []
    for replica_id in range(layout.replicas):
        first_card = replica_id * layout.tp_size
        cards = tuple(range(first_card, first_card + layout.tp_size))
        roots = topology.roots_for_cards(cards)
        groups.append(HBFReplicaPlacement(
            replica_id=replica_id,
            card_ids=cards,
            pcie_root_ids=roots,
        ))
        covered_cards.extend(cards)

    expected_cards = list(range(hardware.card_count))
    if covered_cards != expected_cards:
        raise HBFModelAstraProjectionError(
            "HBF replica placement must cover every card exactly once")
    for group in groups:
        if len(group.card_ids) != layout.tp_size:
            raise HBFModelAstraProjectionError(
                "HBF replica placement has the wrong TP width")
        if any(
                root_id < 0 or root_id >= hardware.pcie_root_count
                for root_id in group.pcie_root_ids):
            raise HBFModelAstraProjectionError(
                "HBF replica placement references an invalid PCIe root")

    return HBFServerPlacement(
        layout=layout.key,
        tp_size=layout.tp_size,
        replicas=layout.replicas,
        server_id=server,
        pcie_topology=topology,
        groups=tuple(groups),
    )


@dataclass(frozen=True)
class HBFModelAstraStage:
    """One controller-compatible stage plus an out-of-band byte ledger."""

    stage_id: str
    runtime_ns: int
    tensor_bytes: int
    resources: tuple[str, ...]
    dependencies: tuple[str, ...]
    hbf_read_bytes: int = 0
    lpddr_bytes: int = 0
    collective_bytes: int = 0
    operation_index: int | None = None
    operation_name: str | None = None
    operation_kind: str | None = None
    card_id: int | None = None
    collective_type: str | None = None
    pcie_route: str | None = None

    def __post_init__(self) -> None:
        _positive_int("stage.runtime_ns", self.runtime_ns)
        _nonnegative_int("stage.tensor_bytes", self.tensor_bytes)
        for name, value in (
            ("stage.hbf_read_bytes", self.hbf_read_bytes),
            ("stage.lpddr_bytes", self.lpddr_bytes),
            ("stage.collective_bytes", self.collective_bytes),
        ):
            _nonnegative_int(name, value)
        if self.tensor_bytes != (
                self.hbf_read_bytes
                + self.lpddr_bytes
                + self.collective_bytes):
            raise HBFModelAstraProjectionError(
                "stage tensor_bytes does not equal its byte ledger")
        if not self.stage_id:
            raise HBFModelAstraProjectionError(
                "stage id must be non-empty")
        if not self.resources or len(set(self.resources)) != len(
                self.resources):
            raise HBFModelAstraProjectionError(
                "stage resources must be non-empty and unique")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise HBFModelAstraProjectionError(
                "stage dependencies must be unique")
        if self.operation_index is not None:
            _nonnegative_int(
                "stage.operation_index", self.operation_index)
        if self.operation_name is not None and not self.operation_name:
            raise HBFModelAstraProjectionError(
                "stage operation_name must be non-empty")
        if self.operation_kind not in (None, "kernel", "collective"):
            raise HBFModelAstraProjectionError(
                "stage operation_kind must be kernel or collective")
        if self.card_id is not None:
            _nonnegative_int("stage.card_id", self.card_id)
        if self.collective_type is not None:
            if (
                self.operation_kind != "collective"
                or not self.collective_type
            ):
                raise HBFModelAstraProjectionError(
                    "collective_type requires a collective stage")
        if self.pcie_route is not None and (
                self.operation_kind != "collective"):
            raise HBFModelAstraProjectionError(
                "pcie_route requires a collective stage")

    def as_dict(self) -> dict[str, Any]:
        """Return exactly the schema consumed by ``Controller``."""

        return {
            "id": self.stage_id,
            "runtime_ns": self.runtime_ns,
            "tensor_bytes": self.tensor_bytes,
            "resources": list(self.resources),
            "deps": list(self.dependencies),
        }

    def audit_dict(self) -> dict[str, Any]:
        value = self.as_dict()
        value["byte_ledger"] = {
            "hbf_read_bytes": self.hbf_read_bytes,
            "lpddr_bytes": self.lpddr_bytes,
            "collective_bytes": self.collective_bytes,
        }
        if self.operation_index is not None:
            value["source_operation"] = {
                "index": self.operation_index,
                "name": self.operation_name,
                "kind": self.operation_kind,
                "card_id": self.card_id,
            }
        if self.collective_type is not None:
            value["collective_type"] = self.collective_type
            value["pcie_route"] = self.pcie_route
        return value


@dataclass(frozen=True)
class HBFModelAstraProjection:
    """Executable ASTRA descriptor with versioned projection audit."""

    placement: HBFServerPlacement
    replica: HBFReplicaPlacement
    batch_id: int
    stages: tuple[HBFModelAstraStage, ...]
    source_total_ns: int
    source_hbf_read_bytes_per_rank: int
    source_lpddr_bytes_per_rank: int
    source_collective_bytes_per_rank: int
    projection_schema: str = PROJECTION_SCHEMA
    projection_fidelity: str = PROJECTION_FIDELITY
    projection_limitations: tuple[str, ...] = AGGREGATE_V1_LIMITATIONS
    physical_hbf_read_bytes_override: int | None = None
    physical_lpddr_bytes_override: int | None = None
    physical_collective_bytes_override: int | None = None
    source_plan_operation_count: int | None = None

    @property
    def schema(self) -> str:
        return self.projection_schema

    @property
    def fidelity(self) -> str:
        return self.projection_fidelity

    @property
    def job_id(self) -> str:
        """Return the deterministic controller job id for this batch."""

        return (
            f"hbf-model.s{self.placement.server_id}."
            f"r{self.replica.replica_id}.b{self.batch_id}"
        )

    @property
    def physical_hbf_read_bytes(self) -> int:
        if self.physical_hbf_read_bytes_override is not None:
            return self.physical_hbf_read_bytes_override
        return (
            self.source_hbf_read_bytes_per_rank
            * self.placement.tp_size
        )

    @property
    def physical_lpddr_bytes(self) -> int:
        if self.physical_lpddr_bytes_override is not None:
            return self.physical_lpddr_bytes_override
        return (
            self.source_lpddr_bytes_per_rank
            * self.placement.tp_size
        )

    @property
    def physical_collective_bytes(self) -> int:
        if self.physical_collective_bytes_override is not None:
            return self.physical_collective_bytes_override
        return (
            self.source_collective_bytes_per_rank
            * self.placement.tp_size
        )

    def controller_stages(self) -> tuple[dict[str, Any], ...]:
        return tuple(stage.as_dict() for stage in self.stages)

    def controller_command_arguments(
            self, arrival_ns: int,
    ) -> tuple[str, int, tuple[dict[str, Any], ...]]:
        """Return arguments accepted by ``Controller.hbf_background_command``."""

        arrival = _nonnegative_int("arrival_ns", arrival_ns)
        return self.job_id, arrival, self.controller_stages()

    def descriptor(self) -> dict[str, Any]:
        return {
            "v": 1,
            "stages": list(self.controller_stages()),
        }

    def descriptor_json(self) -> str:
        return json.dumps(
            self.descriptor(), separators=(",", ":"), sort_keys=True)

    @cached_property
    def _dependency_critical_path_ns(self) -> int:
        return hbf_dependency_critical_path_ns(self.stages)

    def dependency_critical_path_ns(self) -> int:
        return self._dependency_critical_path_ns

    @cached_property
    def _solo_resource_timing(self) -> HBFNamedResourceTiming:
        return hbf_solo_named_resource_timing(self.stages)

    def solo_resource_timing(self) -> HBFNamedResourceTiming:
        return self._solo_resource_timing

    def solo_resource_serialized_completion_ns(self) -> int:
        return self.solo_resource_timing().resource_serialized_completion_ns

    def critical_path_ns(self) -> int:
        """Compatibility alias for the dependency-only DAG path."""

        return self.dependency_critical_path_ns()

    def audit_dict(self) -> dict[str, Any]:
        source_contract = {
            "total_ns": self.source_total_ns,
            "hbf_read_bytes_per_rank":
                self.source_hbf_read_bytes_per_rank,
            "lpddr_bytes_per_rank":
                self.source_lpddr_bytes_per_rank,
            "collective_bytes_per_rank":
                self.source_collective_bytes_per_rank,
        }
        if self.source_plan_operation_count is not None:
            source_contract["plan_operation_count"] = (
                self.source_plan_operation_count)
        timing = self.solo_resource_timing()
        return {
            "schema": self.schema,
            "fidelity": self.fidelity,
            "limitations": list(self.projection_limitations),
            "placement": self.placement.as_dict(),
            "selected_replica": self.replica.as_dict(),
            "batch_id": self.batch_id,
            "job_id": self.job_id,
            "source_contract": source_contract,
            "physical_byte_contract": {
                "hbf_read_bytes": self.physical_hbf_read_bytes,
                "lpddr_bytes": self.physical_lpddr_bytes,
                "collective_bytes": self.physical_collective_bytes,
            },
            "timing_contract": {
                "dependency_critical_path_ns":
                    timing.dependency_critical_path_ns,
                "solo_resource_serialized_completion_ns": (
                    timing.resource_serialized_completion_ns
                ),
                "solo_internal_resource_serialization_wait_ns": (
                    timing.critical_path_internal_wait_ns
                ),
                "solo_cumulative_stage_resource_wait_ns": (
                    timing.cumulative_stage_resource_wait_ns
                ),
            },
            "stages": [stage.audit_dict() for stage in self.stages],
        }

    def validate(self) -> None:
        _nonnegative_int("batch_id", self.batch_id)
        if not self.stages:
            raise HBFModelAstraProjectionError(
                "projection must contain at least one stage")
        if len(self.stages) > ORDERED_V2_STAGE_LIMIT:
            raise HBFModelAstraProjectionError(
                "projection exceeds ASTRA's "
                f"{ORDERED_V2_STAGE_LIMIT}-stage background-DAG limit")
        if not self.projection_schema or not self.projection_fidelity:
            raise HBFModelAstraProjectionError(
                "projection schema and fidelity must be non-empty")
        if (
            not isinstance(self.projection_limitations, tuple)
            or any(
                not isinstance(value, str) or not value
                for value in self.projection_limitations
            )
        ):
            raise HBFModelAstraProjectionError(
                "projection limitations must be non-empty strings")
        for name, value in (
            (
                "physical_hbf_read_bytes_override",
                self.physical_hbf_read_bytes_override,
            ),
            (
                "physical_lpddr_bytes_override",
                self.physical_lpddr_bytes_override,
            ),
            (
                "physical_collective_bytes_override",
                self.physical_collective_bytes_override,
            ),
        ):
            if value is not None:
                _nonnegative_int(name, value)
        if self.source_plan_operation_count is not None:
            _positive_int(
                "source_plan_operation_count",
                self.source_plan_operation_count,
            )
        ids = [stage.stage_id for stage in self.stages]
        if len(set(ids)) != len(ids):
            raise HBFModelAstraProjectionError(
                "projection contains duplicate stage ids")
        known = set(ids)
        for stage in self.stages:
            unknown = set(stage.dependencies) - known
            if unknown:
                raise HBFModelAstraProjectionError(
                    f"stage {stage.stage_id!r} has unknown dependencies "
                    f"{sorted(unknown)}")

        selected_cards = set(self.replica.card_ids)
        resource_prefix = (
            f"hbf-server:{self.placement.server_id}:")
        for stage in self.stages:
            if any(not resource.startswith(resource_prefix)
                   for resource in stage.resources):
                raise HBFModelAstraProjectionError(
                    "projection resource escaped its HBF server namespace")
            for resource in stage.resources:
                marker = ":card:"
                if marker not in resource:
                    continue
                suffix = resource.split(marker, 1)[1]
                try:
                    card_id = int(suffix.split(":", 1)[0])
                except (ValueError, IndexError) as exc:
                    raise HBFModelAstraProjectionError(
                        f"invalid card resource {resource!r}") from exc
                if card_id not in selected_cards:
                    raise HBFModelAstraProjectionError(
                        f"resource {resource!r} escaped replica "
                        f"{self.replica.replica_id}")
            if stage.collective_type is not None:
                expected_resources = (
                    self.placement.pcie_topology.collective_resources(
                        replica_id=self.replica.replica_id,
                        card_ids=self.replica.card_ids,
                        collective_type=stage.collective_type,
                    )
                )
                if stage.resources != expected_resources:
                    raise HBFModelAstraProjectionError(
                        f"collective stage {stage.stage_id!r} has "
                        "non-canonical PCIe resources")
                expected_route = (
                    self.placement.pcie_topology.collective_route(
                        card_ids=self.replica.card_ids,
                        collective_type=stage.collective_type,
                    )
                )
                if stage.pcie_route != expected_route:
                    raise HBFModelAstraProjectionError(
                        f"collective stage {stage.stage_id!r} has "
                        "non-canonical PCIe route")

        if self.dependency_critical_path_ns() != self.source_total_ns:
            raise HBFModelAstraProjectionError(
                "projection critical path does not equal source total_ns")
        byte_totals = {
            "hbf_read": sum(stage.hbf_read_bytes for stage in self.stages),
            "lpddr": sum(stage.lpddr_bytes for stage in self.stages),
            "collective": sum(
                stage.collective_bytes for stage in self.stages),
        }
        expected = {
            "hbf_read": self.physical_hbf_read_bytes,
            "lpddr": self.physical_lpddr_bytes,
            "collective": self.physical_collective_bytes,
        }
        if byte_totals != expected:
            raise HBFModelAstraProjectionError(
                f"projection byte ledger mismatch: actual={byte_totals}, "
                f"expected={expected}")


def _apportion(total: int, weighted_keys: Sequence[tuple[str, int]]) -> dict[str, int]:
    """Integer largest-remainder split with deterministic tie breaking."""

    value = _nonnegative_int("apportion total", total)
    positive = [
        (str(key), _positive_int(f"weight[{key}]", weight))
        for key, weight in weighted_keys
        if weight > 0
    ]
    if not positive:
        if value:
            raise HBFModelAstraProjectionError(
                "cannot apportion positive bytes across zero runtime")
        return {}
    denominator = sum(weight for _, weight in positive)
    allocations: dict[str, int] = {}
    remainders = []
    assigned = 0
    for index, (key, weight) in enumerate(positive):
        quotient, remainder = divmod(value * weight, denominator)
        allocations[key] = quotient
        assigned += quotient
        remainders.append((-remainder, index, key))
    for _, _, key in sorted(remainders)[:value - assigned]:
        allocations[key] += 1
    if sum(allocations.values()) != value:
        raise AssertionError("integer byte apportionment lost bytes")
    return allocations


def _exact_byte_count(name: str, value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise HBFModelAstraProjectionError(
            f"{name} must be non-negative and finite")
    rounded = int(round(float(value)))
    if not math.isclose(
            float(value), rounded, rel_tol=0.0, abs_tol=1e-6):
        raise HBFModelAstraProjectionError(
            f"{name} must resolve to an exact physical byte count")
    return rounded


def _validate_execution_plan(
        plan: HBFModelBatchExecutionPlan,
        layout: HBFParallelLayout) -> None:
    if not isinstance(plan, HBFModelBatchExecutionPlan):
        raise TypeError("plan must be HBFModelBatchExecutionPlan")
    if (
        plan.layout != layout.key
        or plan.tp_size != layout.tp_size
        or plan.replicas != layout.replicas
    ):
        raise HBFModelAstraProjectionError(
            "execution-plan layout metadata does not match placement layout")
    if plan.physical_kv_replication_factor != (
            layout.physical_kv_replication_factor):
        raise HBFModelAstraProjectionError(
            "execution-plan KV replication metadata does not match layout")
    if not plan.operations:
        raise HBFModelAstraProjectionError(
            "execution plan must contain operations")
    for index, operation in enumerate(plan.operations):
        if not isinstance(
                operation,
                (HBFKernelExecutionOp, HBFCollectiveExecutionOp)):
            raise HBFModelAstraProjectionError(
                f"unsupported execution operation at index {index}")
        _positive_int(
            f"plan.operations[{index}].latency_ns",
            operation.latency_ns,
        )
        if not operation.name:
            raise HBFModelAstraProjectionError(
                f"plan operation {index} has an empty name")
        if isinstance(operation, HBFKernelExecutionOp):
            if operation.kind != "kernel":
                raise HBFModelAstraProjectionError(
                    f"plan operation {index} has an invalid kernel kind")
            _exact_byte_count(
                f"plan.operations[{index}].hbf_read_bytes_per_rank",
                operation.hbf_read_bytes_per_rank,
            )
            _exact_byte_count(
                f"plan.operations[{index}].lpddr_bytes_per_rank",
                operation.lpddr_bytes_per_rank,
            )
        else:
            if operation.kind != "collective":
                raise HBFModelAstraProjectionError(
                    f"plan operation {index} has an invalid collective kind")
            _nonnegative_int(
                (
                    f"plan.operations[{index}]."
                    "transferred_bytes_per_rank"
                ),
                operation.transferred_bytes_per_rank,
            )
    if plan.total_ns != sum(
            operation.latency_ns for operation in plan.operations):
        raise HBFModelAstraProjectionError(
            "execution-plan total does not equal ordered operation runtimes")

    context_ranks = plan.context_attention_rank_executions
    if layout.is_context_striped:
        if (
            len(context_ranks) != 2
            or tuple(rank.pair_rank for rank in context_ranks) != (0, 1)
        ):
            raise HBFModelAstraProjectionError(
                "tp8_context plan requires pair-rank 0/1 attention metadata")
        if any(
                rank.q_heads != layout.attention_q_heads_per_rank
                or rank.kv_heads != layout.kv_heads_per_rank
                for rank in context_ranks):
            raise HBFModelAstraProjectionError(
                "tp8_context rank metadata has the wrong head mapping")
        for rank in context_ranks:
            _positive_int(
                f"context rank {rank.pair_rank} latency",
                rank.latency_ns,
            )
            _exact_byte_count(
                f"context rank {rank.pair_rank} HBF bytes",
                rank.hbf_read_bytes,
            )
            _exact_byte_count(
                f"context rank {rank.pair_rank} LPDDR bytes",
                rank.lpddr_bytes,
            )
        attention_latencies = {
            operation.latency_ns
            for operation in plan.kernel_operations
            if operation.category == "attention"
        }
        expected = max(rank.latency_ns for rank in context_ranks)
        if attention_latencies != {expected}:
            raise HBFModelAstraProjectionError(
                "tp8_context attention operation is not the pair-rank "
                "critical-path barrier")
    elif context_ranks:
        raise HBFModelAstraProjectionError(
            "conventional layout cannot carry context-rank metadata")


def _validate_latency(
        latency: HBFModelBatchLatency,
        layout: HBFParallelLayout) -> None:
    if not isinstance(latency, HBFModelBatchLatency):
        raise TypeError("latency must be HBFModelBatchLatency")
    if (
        latency.layout != layout.key
        or latency.tp_size != layout.tp_size
        or latency.replicas != layout.replicas
    ):
        raise HBFModelAstraProjectionError(
            "latency layout metadata does not match placement layout")
    total = _positive_int("latency.total_ns", latency.total_ns)
    component_fields = [
        field for _, field in _LOCAL_PHASE_FIELDS
    ] + ["collective_ns"]
    components = {
        field: _nonnegative_int(
            f"latency.{field}", getattr(latency, field))
        for field in component_fields
    }
    if sum(components.values()) != total:
        raise HBFModelAstraProjectionError(
            "latency total_ns is not the sum of aggregate components")
    collective_part_fields = (
        tuple(field for _, field in _COLLECTIVE_FIELDS)
        + _PAIR_COLLECTIVE_LATENCY_FIELDS
    )
    collective_parts = {
        field: _nonnegative_int(
            f"latency.{field}", getattr(latency, field))
        for field in collective_part_fields
    }
    if sum(collective_parts.values()) != latency.collective_ns:
        raise HBFModelAstraProjectionError(
            "collective component runtimes do not sum to collective_ns")
    for field in (
        "hbf_read_bytes_per_rank",
        "lpddr_bytes_per_rank",
        "collective_bytes_per_rank",
    ):
        _nonnegative_int(f"latency.{field}", getattr(latency, field))
    if layout.tp_size == 1 and (
        latency.collective_ns
        or latency.collective_bytes_per_rank
    ):
        raise HBFModelAstraProjectionError(
            "DP8/TP1 latency cannot contain collective work")
    if layout.tp_size > 1 and latency.collective_ns == 0:
        raise HBFModelAstraProjectionError(
            "TP layout must contain collective work")


def build_full_model_hbf_astra_projection(
        *, latency: HBFModelBatchLatency,
        hardware: HBFServerHardware, layout: HBFParallelLayout,
        replica_id: int, batch_id: int,
        server_id: int = 0) -> HBFModelAstraProjection:
    """Build one controller-compatible, card-isolated aggregate-v1 DAG."""

    if layout.is_context_striped:
        raise HBFModelAstraProjectionError(
            "aggregate-v1 cannot represent asymmetric tp8_context bytes; "
            "use build_ordered_full_model_hbf_astra_projection")
    placement = build_hbf_server_placement(
        hardware=hardware, layout=layout, server_id=server_id)
    replica = placement.group(replica_id)
    batch = _nonnegative_int("batch_id", batch_id)
    _validate_latency(latency, layout)

    local_weights = [
        (phase, getattr(latency, field))
        for phase, field in _LOCAL_PHASE_FIELDS
        if getattr(latency, field) > 0
    ]
    hbf_by_phase = _apportion(
        latency.hbf_read_bytes_per_rank, local_weights)
    lpddr_by_phase = _apportion(
        latency.lpddr_bytes_per_rank, local_weights)
    collective_weights = [
        (phase, getattr(latency, field))
        for phase, field in _COLLECTIVE_FIELDS
        if getattr(latency, field) > 0
    ]
    collective_per_rank = _apportion(
        latency.collective_bytes_per_rank,
        collective_weights,
    )

    stages: list[HBFModelAstraStage] = []
    previous_by_card: dict[int, str | None] = {
        card_id: None for card_id in replica.card_ids
    }
    prefix = f"hbf-server:{placement.server_id}"
    stage_prefix = (
        f"batch:{batch}:replica:{replica.replica_id}")

    for phase, runtime_field in _PRE_COLLECTIVE_PHASES:
        runtime = getattr(latency, runtime_field)
        if runtime == 0:
            continue
        for card_id in replica.card_ids:
            stage_id = (
                f"{stage_prefix}:card:{card_id}:{phase}")
            previous = previous_by_card[card_id]
            hbf_bytes = hbf_by_phase.get(phase, 0)
            lpddr_bytes = lpddr_by_phase.get(phase, 0)
            stages.append(HBFModelAstraStage(
                stage_id=stage_id,
                runtime_ns=runtime,
                tensor_bytes=hbf_bytes + lpddr_bytes,
                resources=(
                    f"{prefix}:card:{card_id}:npu",
                    f"{prefix}:card:{card_id}:hbf-read",
                    f"{prefix}:card:{card_id}:lpddr",
                ),
                dependencies=((previous,) if previous is not None else ()),
                hbf_read_bytes=hbf_bytes,
                lpddr_bytes=lpddr_bytes,
            ))
            previous_by_card[card_id] = stage_id

    collective_tail: str | None = None
    for phase, runtime_field in _COLLECTIVE_FIELDS:
        runtime = getattr(latency, runtime_field)
        if runtime == 0:
            continue
        stage_id = f"{stage_prefix}:collective:{phase}"
        if collective_tail is None:
            dependencies = tuple(
                previous_by_card[card_id]
                for card_id in replica.card_ids
                if previous_by_card[card_id] is not None
            )
        else:
            dependencies = (collective_tail,)
        bytes_total = (
            collective_per_rank.get(phase, 0)
            * layout.tp_size
        )
        collective_type = _AGGREGATE_COLLECTIVE_TYPES[phase]
        stages.append(HBFModelAstraStage(
            stage_id=stage_id,
            runtime_ns=runtime,
            tensor_bytes=bytes_total,
            resources=(
                placement.pcie_topology.collective_resources(
                    replica_id=replica.replica_id,
                    card_ids=replica.card_ids,
                    collective_type=collective_type,
                )
            ),
            dependencies=dependencies,
            collective_bytes=bytes_total,
            operation_kind="collective",
            collective_type=collective_type,
            pcie_route=placement.pcie_topology.collective_route(
                card_ids=replica.card_ids,
                collective_type=collective_type,
            ),
        ))
        collective_tail = stage_id

    final_phase, final_runtime_field = _FINAL_PHASE
    final_runtime = getattr(latency, final_runtime_field)
    if final_runtime > 0:
        for card_id in replica.card_ids:
            stage_id = (
                f"{stage_prefix}:card:{card_id}:"
                f"{final_phase}")
            previous = (
                collective_tail
                if collective_tail is not None
                else previous_by_card[card_id]
            )
            hbf_bytes = hbf_by_phase.get(final_phase, 0)
            lpddr_bytes = lpddr_by_phase.get(final_phase, 0)
            stages.append(HBFModelAstraStage(
                stage_id=stage_id,
                runtime_ns=final_runtime,
                tensor_bytes=hbf_bytes + lpddr_bytes,
                resources=(
                    f"{prefix}:card:{card_id}:npu",
                    f"{prefix}:card:{card_id}:hbf-read",
                    f"{prefix}:card:{card_id}:lpddr",
                ),
                dependencies=((previous,) if previous is not None else ()),
                hbf_read_bytes=hbf_bytes,
                lpddr_bytes=lpddr_bytes,
            ))
            previous_by_card[card_id] = stage_id

    projection = HBFModelAstraProjection(
        placement=placement,
        replica=replica,
        batch_id=batch,
        stages=tuple(stages),
        source_total_ns=latency.total_ns,
        source_hbf_read_bytes_per_rank=(
            latency.hbf_read_bytes_per_rank),
        source_lpddr_bytes_per_rank=(
            latency.lpddr_bytes_per_rank),
        source_collective_bytes_per_rank=(
            latency.collective_bytes_per_rank),
    )
    projection.validate()
    return projection


def build_ordered_full_model_hbf_astra_projection(
        *, plan: HBFModelBatchExecutionPlan,
        hardware: HBFServerHardware, layout: HBFParallelLayout,
        replica_id: int, batch_id: int,
        server_id: int = 0,
        latency: HBFModelBatchLatency | None = None,
) -> HBFModelAstraProjection:
    """Project every ordered source operation onto explicit ASTRA stages.

    Card-local kernels fork across the selected TP replica.  Every logical
    collective joins all card tails, reserves the replica/card fabric
    resources once, and becomes the dependency of the next card-local
    operation.  For ``tp8_context``, core-attention stages use the exact
    even/odd pair-rank records carried by ``plan``; the scalar critical-rank
    byte count is never multiplied across all eight cards.
    """

    placement = build_hbf_server_placement(
        hardware=hardware, layout=layout, server_id=server_id)
    replica = placement.group(replica_id)
    batch = _nonnegative_int("batch_id", batch_id)
    _validate_execution_plan(plan, layout)

    kernel_count = len(plan.kernel_operations)
    collective_count = len(plan.collective_operations)
    expected_stage_count = (
        kernel_count * len(replica.card_ids) + collective_count)
    if expected_stage_count > ORDERED_V2_STAGE_LIMIT:
        raise HBFModelAstraProjectionError(
            "ordered-v2 projection would exceed ASTRA's "
            f"{ORDERED_V2_STAGE_LIMIT}-stage background-DAG limit")

    scalar_hbf_per_rank = sum(
        _exact_byte_count(
            f"kernel {operation.name} HBF bytes",
            operation.hbf_read_bytes_per_rank,
        )
        for operation in plan.kernel_operations
    )
    scalar_lpddr_per_rank = sum(
        _exact_byte_count(
            f"kernel {operation.name} LPDDR bytes",
            operation.lpddr_bytes_per_rank,
        )
        for operation in plan.kernel_operations
    )
    scalar_collective_per_rank = sum(
        operation.transferred_bytes_per_rank
        for operation in plan.collective_operations
    )
    if latency is not None:
        _validate_latency(latency, layout)
        if latency.total_ns != plan.total_ns:
            raise HBFModelAstraProjectionError(
                "latency total_ns does not match execution plan")
        expected_source = (
            scalar_hbf_per_rank,
            scalar_lpddr_per_rank,
            scalar_collective_per_rank,
        )
        actual_source = (
            latency.hbf_read_bytes_per_rank,
            latency.lpddr_bytes_per_rank,
            latency.collective_bytes_per_rank,
        )
        if actual_source != expected_source:
            raise HBFModelAstraProjectionError(
                "latency byte audit does not match execution plan: "
                f"latency={actual_source}, plan={expected_source}")
        source_hbf_per_rank = latency.hbf_read_bytes_per_rank
        source_lpddr_per_rank = latency.lpddr_bytes_per_rank
        source_collective_per_rank = (
            latency.collective_bytes_per_rank)
    else:
        source_hbf_per_rank = scalar_hbf_per_rank
        source_lpddr_per_rank = scalar_lpddr_per_rank
        source_collective_per_rank = scalar_collective_per_rank

    context_by_pair_rank = {
        rank.pair_rank: rank
        for rank in plan.context_attention_rank_executions
    }
    prefix = f"hbf-server:{placement.server_id}"
    stage_prefix = (
        f"batch:{batch}:replica:{replica.replica_id}")
    previous_by_card: dict[int, str | None] = {
        card_id: None for card_id in replica.card_ids
    }
    stages: list[HBFModelAstraStage] = []

    for operation_index, operation in enumerate(plan.operations):
        operation_prefix = (
            f"{stage_prefix}:op:{operation_index}:"
            f"{operation.name}")
        if isinstance(operation, HBFKernelExecutionOp):
            for card_id in replica.card_ids:
                if (
                    layout.is_context_striped
                    and operation.category == "attention"
                ):
                    rank = context_by_pair_rank[card_id % 2]
                    runtime_ns = rank.latency_ns
                    hbf_bytes = _exact_byte_count(
                        (
                            f"context attention card {card_id} "
                            "HBF bytes"
                        ),
                        rank.hbf_read_bytes,
                    )
                    lpddr_bytes = _exact_byte_count(
                        (
                            f"context attention card {card_id} "
                            "LPDDR bytes"
                        ),
                        rank.lpddr_bytes,
                    )
                else:
                    runtime_ns = operation.latency_ns
                    hbf_bytes = _exact_byte_count(
                        f"kernel {operation.name} HBF bytes",
                        operation.hbf_read_bytes_per_rank,
                    )
                    lpddr_bytes = _exact_byte_count(
                        f"kernel {operation.name} LPDDR bytes",
                        operation.lpddr_bytes_per_rank,
                    )
                resources = [f"{prefix}:card:{card_id}:npu"]
                if hbf_bytes:
                    resources.append(
                        f"{prefix}:card:{card_id}:hbf-read")
                if lpddr_bytes:
                    resources.append(
                        f"{prefix}:card:{card_id}:lpddr")
                previous = previous_by_card[card_id]
                stage_id = f"{operation_prefix}:card:{card_id}"
                stages.append(HBFModelAstraStage(
                    stage_id=stage_id,
                    runtime_ns=runtime_ns,
                    tensor_bytes=hbf_bytes + lpddr_bytes,
                    resources=tuple(resources),
                    dependencies=(
                        (previous,) if previous is not None else ()),
                    hbf_read_bytes=hbf_bytes,
                    lpddr_bytes=lpddr_bytes,
                    operation_index=operation_index,
                    operation_name=operation.name,
                    operation_kind="kernel",
                    card_id=card_id,
                ))
                previous_by_card[card_id] = stage_id
            continue

        if not isinstance(operation, HBFCollectiveExecutionOp):
            raise AssertionError(
                "validated execution plan changed operation type")
        stage_id = f"{operation_prefix}:collective"
        dependencies = tuple(
            previous_by_card[card_id]
            for card_id in replica.card_ids
            if previous_by_card[card_id] is not None
        )
        physical_bytes = (
            operation.transferred_bytes_per_rank
            * len(replica.card_ids)
        )
        stages.append(HBFModelAstraStage(
            stage_id=stage_id,
            runtime_ns=operation.latency_ns,
            tensor_bytes=physical_bytes,
            resources=(
                placement.pcie_topology.collective_resources(
                    replica_id=replica.replica_id,
                    card_ids=replica.card_ids,
                    collective_type=operation.collective_type,
                )
            ),
            dependencies=dependencies,
            collective_bytes=physical_bytes,
            operation_index=operation_index,
            operation_name=operation.name,
            operation_kind="collective",
            collective_type=operation.collective_type,
            pcie_route=placement.pcie_topology.collective_route(
                card_ids=replica.card_ids,
                collective_type=operation.collective_type,
            ),
        ))
        for card_id in replica.card_ids:
            previous_by_card[card_id] = stage_id

    if len(stages) != expected_stage_count:
        raise AssertionError(
            "ordered-v2 stage expansion changed the expected stage count")
    physical_hbf_bytes = sum(stage.hbf_read_bytes for stage in stages)
    physical_lpddr_bytes = sum(stage.lpddr_bytes for stage in stages)
    physical_collective_bytes = sum(
        stage.collective_bytes for stage in stages)
    projection = HBFModelAstraProjection(
        placement=placement,
        replica=replica,
        batch_id=batch,
        stages=tuple(stages),
        source_total_ns=plan.total_ns,
        source_hbf_read_bytes_per_rank=source_hbf_per_rank,
        source_lpddr_bytes_per_rank=source_lpddr_per_rank,
        source_collective_bytes_per_rank=source_collective_per_rank,
        projection_schema=ORDERED_V2_SCHEMA,
        projection_fidelity=ORDERED_V2_FIDELITY,
        projection_limitations=ORDERED_V2_LIMITATIONS,
        physical_hbf_read_bytes_override=physical_hbf_bytes,
        physical_lpddr_bytes_override=physical_lpddr_bytes,
        physical_collective_bytes_override=physical_collective_bytes,
        source_plan_operation_count=len(plan.operations),
    )
    projection.validate()
    return projection


__all__ = [
    "AGGREGATE_V1_LIMITATIONS",
    "ASTRA_NAMED_RESOURCE_TIMING_SEMANTICS",
    "HBFAstraTimingAccounting",
    "HBFNamedResourceTiming",
    "HBFModelAstraProjection",
    "HBFModelAstraProjectionError",
    "HBFModelAstraStage",
    "HBFReplicaPlacement",
    "HBFServerPlacement",
    "PROJECTION_FIDELITY",
    "PROJECTION_SCHEMA",
    "ORDERED_V2_FIDELITY",
    "ORDERED_V2_LIMITATIONS",
    "ORDERED_V2_SCHEMA",
    "ORDERED_V2_STAGE_LIMIT",
    "build_full_model_hbf_astra_projection",
    "build_ordered_full_model_hbf_astra_projection",
    "build_hbf_server_placement",
    "hbf_dependency_critical_path_ns",
    "hbf_solo_named_resource_timing",
    "validate_hbf_astra_timing_metrics",
]

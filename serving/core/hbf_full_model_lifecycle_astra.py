"""ASTRA shared-resource DAGs for full-model HBF lifecycle traffic.

The legacy lifecycle calendar reserves every migration resource in parallel.
That is useful as a capacity ledger, but it is not a causal transfer model.
This module projects the immutable ``MigrationJob`` and ``AppendJob`` records
onto ``hbf-background-v1`` DAGs:

* migration: GPU source PCIe -> RDMA NIC -> destination PCIe route -> card
  link -> HBF write.  A destination away from the NIC root reserves the NIC
  root, inter-root link, and destination root in one collapsed route stage;
* append: card-local LPDDR read -> HBF write.

Transfers are chunked so adjacent hops can pipeline.  Bandwidth service is
charged once per chunk as ``ceil(bytes / GBps)`` nanoseconds.  RDMA fixed
latency is charged once per migration, on its first RDMA chunk.  HBF write
fixed latency is charged once per non-empty card stream, on that card's first
write chunk.  Thus changing ``chunk_bytes`` changes pipeline granularity but
does not multiply either configured fixed latency.

HBF writes intentionally reserve the same ``...:hbf-read`` resource used by
the foreground full-model projection.  The name reflects the first consumer
of that resource, while its contract is a shared, bidirectional HBF-media
calendar.  LPDDR reads likewise use the foreground ``...:lpddr`` resource.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cached_property
import json
import math
from typing import Any, Mapping, Sequence

from .controller import Controller
from .hbf_full_model_astra import (
    HBFNamedResourceTiming,
    HBFModelAstraProjectionError,
    HBFReplicaPlacement,
    HBFServerPlacement,
    build_hbf_server_placement,
    hbf_dependency_critical_path_ns,
    hbf_solo_named_resource_timing,
)
from .hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
)
from .hbf_full_model_lifecycle import (
    AppendJob,
    MigrationJob,
    canonical_card_bytes,
    hbf_kv_range_card_bytes,
)


PROJECTION_SCHEMA = "hbf-full-model-lifecycle-astra-v1/chunked-v1"
PROJECTION_FIDELITY = "causal-chunked-v1"
ASTRA_BACKGROUND_STAGE_LIMIT = 1_000_000
RDMA_FIXED_LATENCY_SEMANTICS = (
    "once_per_migration_on_first_rdma_chunk")
HBF_WRITE_FIXED_LATENCY_SEMANTICS = (
    "once_per_nonempty_card_stream_on_first_write_chunk")
PCIE_FIXED_LATENCY_SEMANTICS = (
    "once_per_nonempty_destination_root_stream")
LPDDR_FIXED_LATENCY_SEMANTICS = "none_configured"

_MIGRATION_ROLES = frozenset({
    "gpu_source_pcie",
    "rdma",
    "destination_pcie_root",
    "destination_pcie_card",
    "hbf_write",
})
_APPEND_ROLES = frozenset({"lpddr_read", "hbf_write"})


class HBFLifecycleAstraProjectionError(ValueError):
    """Raised when a lifecycle transfer cannot be projected exactly."""


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HBFLifecycleAstraProjectionError(
            f"{name} must be a non-negative integer")
    return value


def _positive_int(name: str, value: object) -> int:
    value = _nonnegative_int(name, value)
    if value == 0:
        raise HBFLifecycleAstraProjectionError(
            f"{name} must be a positive integer")
    return value


def _positive_float(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise HBFLifecycleAstraProjectionError(
            f"{name} must be positive and finite")
    return float(value)


def _fixed_latency_ns(value_us: float) -> int:
    if (
        isinstance(value_us, bool)
        or not isinstance(value_us, (int, float))
        or not math.isfinite(float(value_us))
        or float(value_us) < 0
    ):
        raise HBFLifecycleAstraProjectionError(
            "fixed latency must be non-negative and finite")
    return int(math.ceil(float(value_us) * 1_000.0))


def _service_ns(byte_count: int, bandwidth_gbps: float) -> int:
    """Return transfer service in ns (one GB/s equals one byte/ns)."""

    byte_count = _positive_int("byte_count", byte_count)
    bandwidth = _positive_float("bandwidth_gbps", bandwidth_gbps)
    return int(math.ceil(byte_count / bandwidth))


def _logical_chunks(total_bytes: int, chunk_bytes: int) -> tuple[int, ...]:
    total = _positive_int("logical_bytes", total_bytes)
    chunk = _positive_int("chunk_bytes", chunk_bytes)
    full, remainder = divmod(total, chunk)
    values = [chunk] * full
    if remainder:
        values.append(remainder)
    return tuple(values)


def _validate_stage_upper_bound(
        *, chunk_count: int, stages_per_chunk: int) -> None:
    """Reject pathological chunking before materializing a giant DAG."""

    count = _positive_int("chunk_count", chunk_count)
    per_chunk = _positive_int("stages_per_chunk", stages_per_chunk)
    if count * per_chunk > ASTRA_BACKGROUND_STAGE_LIMIT:
        raise HBFLifecycleAstraProjectionError(
            "lifecycle chunking may exceed ASTRA's "
            f"{ASTRA_BACKGROUND_STAGE_LIMIT}-stage background-DAG limit; "
            "increase chunk_bytes")


def _balanced_card_ranges(
        physical_bytes: int,
        card_ids: Sequence[int],
) -> tuple[tuple[int, int, int], ...]:
    """Return contiguous physical-byte stripes with an exact total."""

    total = _positive_int("physical_bytes", physical_bytes)
    cards = tuple(card_ids)
    if not cards or len(set(cards)) != len(cards):
        raise HBFLifecycleAstraProjectionError(
            "card placement must be non-empty and unique")
    quotient, remainder = divmod(total, len(cards))
    cursor = 0
    result = []
    for index, card_id in enumerate(cards):
        byte_count = quotient + (1 if index < remainder else 0)
        start = cursor
        cursor += byte_count
        result.append((card_id, start, cursor))
    if cursor != total:
        raise AssertionError("physical card striping lost bytes")
    return tuple(result)


def _chunk_card_bytes(
        *, logical_chunks: Sequence[int], replication_factor: int,
        target_card_bytes: Mapping[int, int],
) -> tuple[dict[int, int], ...]:
    """Split every chunk while preserving an exact final card quota.

    Each chunk is apportioned in proportion to the bytes still owed to every
    card.  Largest remainders break rounding ties in placement order.  This
    retains pipelining for conventional TP stripes and also supports the
    sparse even/odd card vectors required by ``tp8_context``.
    """

    factor = _positive_int("replication_factor", replication_factor)
    cards = tuple(target_card_bytes)
    if not cards or len(cards) != len(set(cards)):
        raise HBFLifecycleAstraProjectionError(
            "target card placement must be non-empty and unique")
    expected_by_card = {}
    for card_id in cards:
        expected_by_card[card_id] = _nonnegative_int(
            "target_card_bytes", target_card_bytes[card_id])
    expected_total = sum(expected_by_card.values())
    physical_total = sum(logical_chunks) * factor
    if expected_total != physical_total:
        raise HBFLifecycleAstraProjectionError(
            "target card quotas do not sum to the physical stream")
    observed_by_card = {card_id: 0 for card_id in cards}
    remaining_by_card = dict(expected_by_card)
    remaining_total = expected_total
    result = []
    for logical_bytes in logical_chunks:
        physical_chunk_bytes = logical_bytes * factor
        if not 0 < physical_chunk_bytes <= remaining_total:
            raise AssertionError(
                "physical chunk exceeds its remaining card quota")
        per_card: dict[int, int] = {}
        remainders = []
        floor_total = 0
        for order, card_id in enumerate(cards):
            quotient, remainder = divmod(
                physical_chunk_bytes * remaining_by_card[card_id],
                remaining_total,
            )
            if quotient:
                per_card[card_id] = quotient
            floor_total += quotient
            remainders.append((remainder, -order, card_id))
        rounding_bytes = physical_chunk_bytes - floor_total
        for _, _, card_id in sorted(remainders, reverse=True)[
                :rounding_bytes]:
            per_card[card_id] = per_card.get(card_id, 0) + 1
        if sum(per_card.values()) != physical_chunk_bytes:
            raise AssertionError("physical chunk striping lost bytes")
        for card_id, byte_count in per_card.items():
            if byte_count > remaining_by_card[card_id]:
                raise AssertionError(
                    "physical chunk exceeded a card's remaining quota")
            observed_by_card[card_id] += byte_count
            remaining_by_card[card_id] -= byte_count
        result.append(per_card)
        remaining_total -= physical_chunk_bytes
    if remaining_total or any(remaining_by_card.values()):
        raise AssertionError("physical chunk stream has the wrong length")
    if observed_by_card != expected_by_card:
        raise AssertionError("chunk striping changed exact card quotas")
    return tuple(result)


@dataclass(frozen=True)
class HBFLifecycleAstraStage:
    """One executable transfer stage plus service-time audit metadata."""

    stage_id: str
    role: str
    chunk_index: int
    runtime_ns: int
    tensor_bytes: int
    bandwidth_gbps: float
    service_ns: int
    fixed_latency_ns: int
    resources: tuple[str, ...]
    dependencies: tuple[str, ...]
    card_id: int | None = None
    root_id: int | None = None

    def __post_init__(self) -> None:
        if not self.stage_id:
            raise HBFLifecycleAstraProjectionError(
                "stage_id must be non-empty")
        _nonnegative_int("stage.chunk_index", self.chunk_index)
        _positive_int("stage.runtime_ns", self.runtime_ns)
        _positive_int("stage.tensor_bytes", self.tensor_bytes)
        _positive_float("stage.bandwidth_gbps", self.bandwidth_gbps)
        _positive_int("stage.service_ns", self.service_ns)
        _nonnegative_int(
            "stage.fixed_latency_ns", self.fixed_latency_ns)
        if self.runtime_ns != self.service_ns + self.fixed_latency_ns:
            raise HBFLifecycleAstraProjectionError(
                "stage runtime must equal service plus fixed latency")
        expected_service = _service_ns(
            self.tensor_bytes, self.bandwidth_gbps)
        if self.service_ns != expected_service:
            raise HBFLifecycleAstraProjectionError(
                "stage service does not equal ceil(bytes / GBps)")
        if not self.resources or len(set(self.resources)) != len(
                self.resources):
            raise HBFLifecycleAstraProjectionError(
                "stage resources must be non-empty and unique")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise HBFLifecycleAstraProjectionError(
                "stage dependencies must be unique")
        if self.card_id is not None:
            _nonnegative_int("stage.card_id", self.card_id)
        if self.root_id is not None:
            _nonnegative_int("stage.root_id", self.root_id)

    def as_dict(self) -> dict[str, Any]:
        """Return exactly the fields accepted by ``Controller``."""

        return {
            "id": self.stage_id,
            "runtime_ns": self.runtime_ns,
            "tensor_bytes": self.tensor_bytes,
            "resources": list(self.resources),
            "deps": list(self.dependencies),
        }

    def audit_dict(self) -> dict[str, Any]:
        value = self.as_dict()
        value.update({
            "role": self.role,
            "chunk_index": self.chunk_index,
            "bandwidth_gbps": self.bandwidth_gbps,
            "service_ns": self.service_ns,
            "fixed_latency_ns": self.fixed_latency_ns,
            "card_id": self.card_id,
            "root_id": self.root_id,
        })
        return value


@dataclass(frozen=True)
class HBFLifecycleByteLedger:
    """Bytes serviced at each causal hop of a lifecycle transfer."""

    gpu_source_pcie_bytes: int = 0
    rdma_bytes: int = 0
    destination_pcie_root_bytes: int = 0
    destination_pcie_card_bytes: int = 0
    lpddr_read_bytes: int = 0
    hbf_write_bytes: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "gpu_source_pcie_bytes": self.gpu_source_pcie_bytes,
            "rdma_bytes": self.rdma_bytes,
            "destination_pcie_root_bytes":
                self.destination_pcie_root_bytes,
            "destination_pcie_card_bytes":
                self.destination_pcie_card_bytes,
            "lpddr_read_bytes": self.lpddr_read_bytes,
            "hbf_write_bytes": self.hbf_write_bytes,
        }


@dataclass(frozen=True)
class HBFCardLifecycleByteLedger:
    """Exact physical bytes assigned to one card."""

    card_id: int
    destination_pcie_bytes: int
    lpddr_read_bytes: int
    hbf_write_bytes: int

    def as_dict(self) -> dict[str, int]:
        return {
            "card_id": self.card_id,
            "destination_pcie_bytes": self.destination_pcie_bytes,
            "lpddr_read_bytes": self.lpddr_read_bytes,
            "hbf_write_bytes": self.hbf_write_bytes,
        }


@dataclass(frozen=True)
class HBFLifecycleAstraProjection:
    """Controller-ready lifecycle DAG with immutable byte accounting."""

    kind: str
    placement: HBFServerPlacement
    replica: HBFReplicaPlacement
    lifecycle_job_id: int
    session_id: str
    generation: int
    version: int
    arrival_ns: int
    legacy_completion_ns: int
    logical_bytes: int
    physical_bytes: int
    per_card_capacity_bytes: int
    card_bytes: tuple[tuple[int, int], ...]
    chunk_bytes: int
    logical_chunks: tuple[int, ...]
    configured_rdma_fixed_latency_ns: int
    configured_hbf_write_fixed_latency_ns: int
    stages: tuple[HBFLifecycleAstraStage, ...]
    byte_ledger: HBFLifecycleByteLedger
    card_ledgers: tuple[HBFCardLifecycleByteLedger, ...]

    @property
    def schema(self) -> str:
        return PROJECTION_SCHEMA

    @property
    def fidelity(self) -> str:
        return PROJECTION_FIDELITY

    @property
    def job_id(self) -> str:
        return (
            f"hbf-{self.kind}.s{self.placement.server_id}."
            f"r{self.replica.replica_id}.j{self.lifecycle_job_id}"
        )

    def controller_stages(self) -> tuple[dict[str, Any], ...]:
        return tuple(stage.as_dict() for stage in self.stages)

    def controller_command_arguments(
            self, arrival_ns: int | None = None,
    ) -> tuple[str, int, tuple[dict[str, Any], ...]]:
        arrival = (
            self.arrival_ns
            if arrival_ns is None
            else _nonnegative_int("arrival_ns", arrival_ns)
        )
        return self.job_id, arrival, self.controller_stages()

    def descriptor(self) -> dict[str, Any]:
        return {"v": 1, "stages": list(self.controller_stages())}

    def descriptor_json(self) -> str:
        return json.dumps(
            self.descriptor(), separators=(",", ":"), sort_keys=True)

    @cached_property
    def _dependency_critical_path_ns(self) -> int:
        try:
            return hbf_dependency_critical_path_ns(self.stages)
        except HBFModelAstraProjectionError as exc:
            raise HBFLifecycleAstraProjectionError(str(exc)) from exc

    def dependency_critical_path_ns(self) -> int:
        return self._dependency_critical_path_ns

    @cached_property
    def _solo_resource_timing(self) -> HBFNamedResourceTiming:
        try:
            return hbf_solo_named_resource_timing(self.stages)
        except HBFModelAstraProjectionError as exc:
            raise HBFLifecycleAstraProjectionError(str(exc)) from exc

    def solo_resource_timing(self) -> HBFNamedResourceTiming:
        return self._solo_resource_timing

    def solo_resource_serialized_completion_ns(self) -> int:
        return self.solo_resource_timing().resource_serialized_completion_ns

    def critical_path_ns(self) -> int:
        """Compatibility alias for the dependency-only DAG path."""

        return self.dependency_critical_path_ns()

    def _expected_resources(
            self, stage: HBFLifecycleAstraStage) -> tuple[str, ...]:
        prefix = f"hbf-server:{self.placement.server_id}"
        if stage.role == "gpu_source_pcie":
            return (f"{prefix}:ingress:gpu-source-pcie-root",)
        if stage.role == "rdma":
            return (
                self.placement.pcie_topology.nic_resource(),)
        if stage.role == "destination_pcie_root":
            return (
                self.placement.pcie_topology
                .migration_root_resources(stage.root_id)
            )
        if stage.role == "destination_pcie_card":
            return (
                self.placement.pcie_topology.card_resource(
                    stage.card_id, domain="migration"),
            )
        if stage.role == "lpddr_read":
            return (f"{prefix}:card:{stage.card_id}:lpddr",)
        if stage.role == "hbf_write":
            # This is the foreground projection's shared HBF-media name.
            return (f"{prefix}:card:{stage.card_id}:hbf-read",)
        raise HBFLifecycleAstraProjectionError(
            f"unknown lifecycle stage role {stage.role!r}")

    def validate(self) -> None:
        if self.kind not in {"migration", "append"}:
            raise HBFLifecycleAstraProjectionError(
                "projection kind must be migration or append")
        _positive_int("lifecycle_job_id", self.lifecycle_job_id)
        if not self.session_id:
            raise HBFLifecycleAstraProjectionError(
                "session_id must be non-empty")
        _nonnegative_int("generation", self.generation)
        _nonnegative_int("version", self.version)
        _nonnegative_int("arrival_ns", self.arrival_ns)
        if self.legacy_completion_ns < self.arrival_ns:
            raise HBFLifecycleAstraProjectionError(
                "legacy completion precedes lifecycle arrival")
        _positive_int("logical_bytes", self.logical_bytes)
        _positive_int("physical_bytes", self.physical_bytes)
        _positive_int(
            "per_card_capacity_bytes", self.per_card_capacity_bytes)
        if tuple(
                card_id for card_id, _ in self.card_bytes
        ) != self.replica.card_ids:
            raise HBFLifecycleAstraProjectionError(
                "projection card bytes are not in placement order")
        expected_card_bytes = {}
        for card_id, byte_count in self.card_bytes:
            expected_card_bytes[card_id] = _nonnegative_int(
                "projection.card_bytes", byte_count)
        if sum(expected_card_bytes.values()) != self.physical_bytes:
            raise HBFLifecycleAstraProjectionError(
                "projection card bytes do not sum to physical_bytes")
        if max(expected_card_bytes.values(), default=0) != (
                self.per_card_capacity_bytes):
            raise HBFLifecycleAstraProjectionError(
                "projection per-card capacity is not the card-vector peak")
        _positive_int("chunk_bytes", self.chunk_bytes)
        _nonnegative_int(
            "configured_rdma_fixed_latency_ns",
            self.configured_rdma_fixed_latency_ns,
        )
        _nonnegative_int(
            "configured_hbf_write_fixed_latency_ns",
            self.configured_hbf_write_fixed_latency_ns,
        )
        if not self.logical_chunks:
            raise HBFLifecycleAstraProjectionError(
                "projection must contain logical chunks")
        if sum(self.logical_chunks) != self.logical_bytes:
            raise HBFLifecycleAstraProjectionError(
                "logical chunk ledger does not sum to logical_bytes")
        if any(
                value <= 0 or value > self.chunk_bytes
                for value in self.logical_chunks):
            raise HBFLifecycleAstraProjectionError(
                "logical chunk violates chunk_bytes")
        if not self.stages:
            raise HBFLifecycleAstraProjectionError(
                "projection must contain stages")

        ids = [stage.stage_id for stage in self.stages]
        if len(ids) != len(set(ids)):
            raise HBFLifecycleAstraProjectionError(
                "projection contains duplicate stage ids")
        known = set(ids)
        selected_cards = set(self.replica.card_ids)
        selected_roots = set(self.replica.pcie_root_ids)
        allowed_roles = (
            _MIGRATION_ROLES
            if self.kind == "migration" else _APPEND_ROLES
        )
        for stage in self.stages:
            if stage.role not in allowed_roles:
                raise HBFLifecycleAstraProjectionError(
                    f"stage role {stage.role!r} is invalid for {self.kind}")
            unknown = set(stage.dependencies) - known
            if unknown:
                raise HBFLifecycleAstraProjectionError(
                    f"stage {stage.stage_id!r} has unknown dependencies "
                    f"{sorted(unknown)}")
            if stage.stage_id in stage.dependencies:
                raise HBFLifecycleAstraProjectionError(
                    f"stage {stage.stage_id!r} depends on itself")
            if (
                stage.card_id is not None
                and stage.card_id not in selected_cards
            ):
                raise HBFLifecycleAstraProjectionError(
                    f"stage escaped replica card ownership: "
                    f"{stage.card_id}")
            if (
                stage.root_id is not None
                and stage.root_id not in selected_roots
            ):
                raise HBFLifecycleAstraProjectionError(
                    f"stage escaped replica root ownership: "
                    f"{stage.root_id}")
            if stage.resources != self._expected_resources(stage):
                raise HBFLifecycleAstraProjectionError(
                    f"stage {stage.stage_id!r} has non-canonical resources")

        if self.dependency_critical_path_ns() <= 0:
            raise HBFLifecycleAstraProjectionError(
                "projection critical path must be positive")

        observed = Counter()
        for stage in self.stages:
            observed[f"{stage.role}_bytes"] += stage.tensor_bytes
        expected = self.byte_ledger.as_dict()
        if {
            key: observed.get(key, 0) for key in expected
        } != expected:
            raise HBFLifecycleAstraProjectionError(
                "projection hop byte ledger does not match its stages")

        if tuple(ledger.card_id for ledger in self.card_ledgers) != (
                self.replica.card_ids):
            raise HBFLifecycleAstraProjectionError(
                "card ledgers are not in canonical placement order")
        for ledger in self.card_ledgers:
            expected_bytes = expected_card_bytes[ledger.card_id]
            if ledger.hbf_write_bytes != expected_bytes:
                raise HBFLifecycleAstraProjectionError(
                    "per-card HBF write ledger is not exact")
            if ledger.hbf_write_bytes > self.per_card_capacity_bytes:
                raise HBFLifecycleAstraProjectionError(
                    "per-card transfer exceeds lifecycle reservation")
            if self.kind == "migration":
                if (
                    ledger.destination_pcie_bytes != expected_bytes
                    or ledger.lpddr_read_bytes != 0
                ):
                    raise HBFLifecycleAstraProjectionError(
                        "migration per-card ledger is invalid")
            elif (
                ledger.destination_pcie_bytes != 0
                or ledger.lpddr_read_bytes != expected_bytes
            ):
                raise HBFLifecycleAstraProjectionError(
                    "append per-card ledger is invalid")

        rdma_stages = [
            stage for stage in self.stages if stage.role == "rdma"
        ]
        rdma_fixed = sum(stage.fixed_latency_ns for stage in rdma_stages)
        hbf_fixed_by_card = {
            card_id: sum(
                stage.fixed_latency_ns
                for stage in self.stages
                if stage.role == "hbf_write"
                and stage.card_id == card_id
            )
            for card_id in self.replica.card_ids
        }
        if self.kind == "migration":
            if (
                not rdma_stages
                or rdma_stages[0].fixed_latency_ns
                != self.configured_rdma_fixed_latency_ns
                or any(
                    stage.fixed_latency_ns
                    for stage in rdma_stages[1:]
                )
                or rdma_fixed != self.configured_rdma_fixed_latency_ns
            ):
                raise HBFLifecycleAstraProjectionError(
                    "RDMA fixed latency was not charged exactly once")
        elif rdma_fixed:
            raise HBFLifecycleAstraProjectionError(
                "append cannot contain RDMA fixed latency")
        expected_card_fixed = {
            ledger.card_id: (
                self.configured_hbf_write_fixed_latency_ns
                if ledger.hbf_write_bytes else 0
            )
            for ledger in self.card_ledgers
        }
        if hbf_fixed_by_card != expected_card_fixed:
            raise HBFLifecycleAstraProjectionError(
                "HBF fixed latency was not charged once per card stream")
        for root_id in self.replica.pcie_root_ids:
            root_stages = [
                stage for stage in self.stages
                if (
                    stage.role == "destination_pcie_root"
                    and stage.root_id == root_id
                )
            ]
            if not root_stages:
                continue
            expected_bandwidth, expected_fixed_us = (
                self.placement.pcie_topology
                .migration_root_service(root_id)
            )
            expected_fixed_ns = _fixed_latency_ns(expected_fixed_us)
            if (
                any(
                    stage.bandwidth_gbps != expected_bandwidth
                    for stage in root_stages
                )
                or root_stages[0].fixed_latency_ns
                != expected_fixed_ns
                or any(
                    stage.fixed_latency_ns
                    for stage in root_stages[1:]
                )
            ):
                raise HBFLifecycleAstraProjectionError(
                    "PCIe root route bandwidth/fixed latency is not "
                    "canonical")
        for card_id in self.replica.card_ids:
            write_stages = [
                stage for stage in self.stages
                if stage.role == "hbf_write"
                and stage.card_id == card_id
            ]
            if write_stages and (
                write_stages[0].fixed_latency_ns
                != self.configured_hbf_write_fixed_latency_ns
                or any(
                    stage.fixed_latency_ns
                    for stage in write_stages[1:]
                )
            ):
                raise HBFLifecycleAstraProjectionError(
                    "HBF fixed latency must appear only on the first "
                    "write chunk")

        # This is the final schema gate. It catches identifier, field, DAG,
        # and positive-runtime drift against the actual protocol encoder.
        Controller.hbf_background_command(
            *self.controller_command_arguments())

    def audit_dict(self) -> dict[str, Any]:
        timing = self.solo_resource_timing()
        return {
            "schema": self.schema,
            "fidelity": self.fidelity,
            "kind": self.kind,
            "job_id": self.job_id,
            "lifecycle_job_id": self.lifecycle_job_id,
            "session_id": self.session_id,
            "generation": self.generation,
            "version": self.version,
            "arrival_ns": self.arrival_ns,
            "legacy_completion_ns": self.legacy_completion_ns,
            "legacy_completion_is_not_used_for_astra": True,
            "placement": self.placement.as_dict(),
            "selected_replica": self.replica.as_dict(),
            "logical_bytes": self.logical_bytes,
            "physical_bytes": self.physical_bytes,
            "per_card_capacity_bytes": self.per_card_capacity_bytes,
            "card_bytes": dict(self.card_bytes),
            "capacity_padding_bytes": (
                self.per_card_capacity_bytes
                * len(self.replica.card_ids)
                - self.physical_bytes
            ),
            "chunk_bytes": self.chunk_bytes,
            "logical_chunks": list(self.logical_chunks),
            "fixed_latency_semantics": {
                "rdma": RDMA_FIXED_LATENCY_SEMANTICS,
                "hbf_write": HBF_WRITE_FIXED_LATENCY_SEMANTICS,
                "pcie": PCIE_FIXED_LATENCY_SEMANTICS,
                "lpddr": LPDDR_FIXED_LATENCY_SEMANTICS,
            },
            "configured_fixed_latency_ns": {
                "rdma": self.configured_rdma_fixed_latency_ns,
                "hbf_write": self.configured_hbf_write_fixed_latency_ns,
            },
            "byte_ledger": self.byte_ledger.as_dict(),
            "card_ledgers": [
                ledger.as_dict() for ledger in self.card_ledgers
            ],
            "timing_contract": {
                "dependency_critical_path_ns":
                    timing.dependency_critical_path_ns,
                "solo_resource_serialized_completion_ns":
                    timing.resource_serialized_completion_ns,
                "solo_internal_resource_serialization_wait_ns":
                    timing.critical_path_internal_wait_ns,
                "solo_cumulative_stage_resource_wait_ns":
                    timing.cumulative_stage_resource_wait_ns,
            },
            "stages": [stage.audit_dict() for stage in self.stages],
        }


def _validate_job(
        job: MigrationJob | AppendJob,
        *, layout: HBFParallelLayout, placement: HBFServerPlacement,
        expected_type: type[MigrationJob] | type[AppendJob],
) -> tuple[HBFReplicaPlacement, dict[int, int]]:
    if not isinstance(job, expected_type):
        raise TypeError(f"job must be {expected_type.__name__}")
    _positive_int("job.job_id", job.job_id)
    if not job.session_id:
        raise HBFLifecycleAstraProjectionError(
            "job.session_id must be non-empty")
    _nonnegative_int("job.generation", job.generation)
    _nonnegative_int("job.version", job.version)
    _positive_int("job.token_count", job.token_count)
    token_start = _nonnegative_int("job.token_start", job.token_start)
    logical = _positive_int("job.logical_bytes", job.logical_bytes)
    physical = _positive_int("job.physical_bytes", job.physical_bytes)
    per_card = _positive_int("job.per_card_bytes", job.per_card_bytes)
    _nonnegative_int("job.start_ns", job.start_ns)
    _nonnegative_int("job.completion_ns", job.completion_ns)
    if job.completion_ns < job.start_ns:
        raise HBFLifecycleAstraProjectionError(
            "job completion precedes start")
    expected_physical = (
        logical * layout.physical_kv_replication_factor)
    if physical != expected_physical:
        raise HBFLifecycleAstraProjectionError(
            "job physical_bytes does not include the layout's exact KV "
            "replication factor")
    replica = placement.group(job.group_id)
    if job.card_bytes:
        if tuple(
                card_id for card_id, _ in job.card_bytes
        ) != replica.card_ids:
            raise HBFLifecycleAstraProjectionError(
                "job card_bytes are not in replica placement order")
        card_bytes = {}
        for card_id, byte_count in job.card_bytes:
            card_bytes[card_id] = _nonnegative_int(
                "job.card_bytes", byte_count)
    elif layout.is_context_striped:
        if logical % job.token_count:
            raise HBFLifecycleAstraProjectionError(
                "context job logical_bytes must divide by token_count")
        card_bytes = hbf_kv_range_card_bytes(
            layout=layout,
            card_ids=replica.card_ids,
            kv_bytes_per_token=logical // job.token_count,
            token_start=token_start,
            token_count=job.token_count,
        )
    else:
        # Legacy manually-constructed jobs predate token offsets and exact
        # vectors. Preserve their balanced start-zero contract.
        card_bytes = {
            card_id: end - start
            for card_id, start, end in _balanced_card_ranges(
                physical, replica.card_ids)
        }
    if sum(card_bytes.values()) != physical:
        raise HBFLifecycleAstraProjectionError(
            "job card_bytes do not sum to physical_bytes")
    if job.card_bytes and logical % job.token_count == 0:
        expected_card_bytes = hbf_kv_range_card_bytes(
            layout=layout,
            card_ids=replica.card_ids,
            kv_bytes_per_token=logical // job.token_count,
            token_start=token_start,
            token_count=job.token_count,
        )
        if card_bytes != expected_card_bytes:
            raise HBFLifecycleAstraProjectionError(
                "job card_bytes do not match token-range placement")
    expected_per_card = max(card_bytes.values(), default=0)
    if per_card != expected_per_card:
        raise HBFLifecycleAstraProjectionError(
            "job per_card_bytes does not equal the card-vector peak")
    return replica, card_bytes


def _make_stage(
        *, stage_id: str, role: str, chunk_index: int,
        byte_count: int, bandwidth_gbps: float,
        resources: tuple[str, ...], dependencies: tuple[str, ...],
        fixed_latency_ns: int = 0, card_id: int | None = None,
        root_id: int | None = None,
) -> HBFLifecycleAstraStage:
    service = _service_ns(byte_count, bandwidth_gbps)
    return HBFLifecycleAstraStage(
        stage_id=stage_id,
        role=role,
        chunk_index=chunk_index,
        runtime_ns=service + fixed_latency_ns,
        tensor_bytes=byte_count,
        bandwidth_gbps=float(bandwidth_gbps),
        service_ns=service,
        fixed_latency_ns=fixed_latency_ns,
        resources=resources,
        dependencies=dependencies,
        card_id=card_id,
        root_id=root_id,
    )


def _card_ledgers(
        *, replica: HBFReplicaPlacement,
        card_bytes: Mapping[int, int],
        kind: str,
) -> tuple[HBFCardLifecycleByteLedger, ...]:
    result = []
    if tuple(card_bytes) != replica.card_ids:
        raise HBFLifecycleAstraProjectionError(
            "card ledger input is not in placement order")
    for card_id in replica.card_ids:
        byte_count = card_bytes[card_id]
        result.append(HBFCardLifecycleByteLedger(
            card_id=card_id,
            destination_pcie_bytes=(
                byte_count if kind == "migration" else 0),
            lpddr_read_bytes=(
                byte_count if kind == "append" else 0),
            hbf_write_bytes=byte_count,
        ))
    return tuple(result)


def build_migration_hbf_astra_projection(
        *, job: MigrationJob, hardware: HBFServerHardware,
        layout: HBFParallelLayout, chunk_bytes: int,
        server_id: int = 0,
        gpu_source_root_bandwidth_gbps: float | None = None,
) -> HBFLifecycleAstraProjection:
    """Project one migration as a causal, chunk-pipelined ASTRA DAG."""

    hardware.validate()
    layout.validate(hardware.card_count)
    chunk = _positive_int("chunk_bytes", chunk_bytes)
    source_bandwidth = _positive_float(
        "gpu_source_root_bandwidth_gbps",
        (
            hardware.pcie_root_bandwidth_gbps
            if gpu_source_root_bandwidth_gbps is None
            else gpu_source_root_bandwidth_gbps
        ),
    )
    placement = build_hbf_server_placement(
        hardware=hardware, layout=layout, server_id=server_id)
    replica, card_bytes = _validate_job(
        job, layout=layout, placement=placement,
        expected_type=MigrationJob)
    logical_chunks = _logical_chunks(job.logical_bytes, chunk)
    _validate_stage_upper_bound(
        chunk_count=len(logical_chunks),
        stages_per_chunk=(
            2 + len(replica.pcie_root_ids) + 2 * len(replica.card_ids)
        ),
    )
    physical_by_chunk = _chunk_card_bytes(
        logical_chunks=logical_chunks,
        replication_factor=layout.physical_kv_replication_factor,
        target_card_bytes=card_bytes,
    )

    prefix = f"hbf-server:{placement.server_id}"
    stage_prefix = (
        f"migration:{job.job_id}:replica:{replica.replica_id}")
    stages: list[HBFLifecycleAstraStage] = []
    previous_source: str | None = None
    previous_rdma: str | None = None
    previous_root: dict[int, str | None] = {
        root_id: None for root_id in replica.pcie_root_ids
    }
    root_started = {
        root_id: False for root_id in replica.pcie_root_ids
    }
    previous_card: dict[int, str | None] = {
        card_id: None for card_id in replica.card_ids
    }
    previous_write: dict[int, str | None] = {
        card_id: None for card_id in replica.card_ids
    }
    write_started = {card_id: False for card_id in replica.card_ids}
    rdma_fixed_ns = _fixed_latency_ns(
        hardware.rdma_one_way_latency_us)
    write_fixed_ns = _fixed_latency_ns(
        hardware.hbf_write_latency_us)

    for chunk_index, logical_byte_count in enumerate(logical_chunks):
        source_id = f"{stage_prefix}:chunk:{chunk_index}:source-pcie"
        stages.append(_make_stage(
            stage_id=source_id,
            role="gpu_source_pcie",
            chunk_index=chunk_index,
            byte_count=logical_byte_count,
            bandwidth_gbps=source_bandwidth,
            resources=(
                f"{prefix}:ingress:gpu-source-pcie-root",),
            dependencies=(
                (previous_source,) if previous_source else ()),
        ))
        previous_source = source_id

        rdma_id = f"{stage_prefix}:chunk:{chunk_index}:rdma"
        rdma_dependencies = [source_id]
        if previous_rdma is not None:
            rdma_dependencies.append(previous_rdma)
        stages.append(_make_stage(
            stage_id=rdma_id,
            role="rdma",
            chunk_index=chunk_index,
            byte_count=logical_byte_count,
            bandwidth_gbps=hardware.rdma_bandwidth_gbps,
            fixed_latency_ns=(
                rdma_fixed_ns if chunk_index == 0 else 0),
            resources=(placement.pcie_topology.nic_resource(),),
            dependencies=tuple(rdma_dependencies),
        ))
        previous_rdma = rdma_id

        bytes_by_card = physical_by_chunk[chunk_index]
        bytes_by_root: dict[int, int] = {}
        for card_id, byte_count in bytes_by_card.items():
            root_id = placement.pcie_topology.root_for_card(card_id)
            bytes_by_root[root_id] = (
                bytes_by_root.get(root_id, 0) + byte_count)

        root_stage_ids: dict[int, str] = {}
        for root_id in sorted(bytes_by_root):
            root_id_string = (
                f"{stage_prefix}:chunk:{chunk_index}:"
                f"pcie-root:{root_id}")
            root_dependencies = [rdma_id]
            if previous_root[root_id] is not None:
                root_dependencies.append(previous_root[root_id])
            root_bandwidth, root_fixed_us = (
                placement.pcie_topology.migration_root_service(root_id))
            root_fixed_ns = (
                0
                if root_started[root_id]
                else _fixed_latency_ns(root_fixed_us)
            )
            stages.append(_make_stage(
                stage_id=root_id_string,
                role="destination_pcie_root",
                chunk_index=chunk_index,
                byte_count=bytes_by_root[root_id],
                bandwidth_gbps=root_bandwidth,
                fixed_latency_ns=root_fixed_ns,
                resources=(
                    placement.pcie_topology
                    .migration_root_resources(root_id)
                ),
                dependencies=tuple(root_dependencies),
                root_id=root_id,
            ))
            root_started[root_id] = True
            previous_root[root_id] = root_id_string
            root_stage_ids[root_id] = root_id_string

        for card_id in replica.card_ids:
            byte_count = bytes_by_card.get(card_id, 0)
            if byte_count == 0:
                continue
            root_id = placement.pcie_topology.root_for_card(card_id)
            card_id_string = (
                f"{stage_prefix}:chunk:{chunk_index}:card:{card_id}:pcie")
            card_dependencies = [root_stage_ids[root_id]]
            if previous_card[card_id] is not None:
                card_dependencies.append(previous_card[card_id])
            stages.append(_make_stage(
                stage_id=card_id_string,
                role="destination_pcie_card",
                chunk_index=chunk_index,
                byte_count=byte_count,
                bandwidth_gbps=(
                    hardware.intra_fabric_bandwidth_gbps_per_card),
                resources=(
                    placement.pcie_topology.card_resource(
                        card_id, domain="migration"),
                ),
                dependencies=tuple(card_dependencies),
                card_id=card_id,
                root_id=root_id,
            ))
            previous_card[card_id] = card_id_string

            write_id = (
                f"{stage_prefix}:chunk:{chunk_index}:"
                f"card:{card_id}:hbf-write")
            write_dependencies = [card_id_string]
            if previous_write[card_id] is not None:
                write_dependencies.append(previous_write[card_id])
            fixed = 0 if write_started[card_id] else write_fixed_ns
            stages.append(_make_stage(
                stage_id=write_id,
                role="hbf_write",
                chunk_index=chunk_index,
                byte_count=byte_count,
                bandwidth_gbps=(
                    hardware.hbf_write_bandwidth_gbps_per_card),
                fixed_latency_ns=fixed,
                resources=(f"{prefix}:card:{card_id}:hbf-read",),
                dependencies=tuple(write_dependencies),
                card_id=card_id,
                root_id=root_id,
            ))
            write_started[card_id] = True
            previous_write[card_id] = write_id

    projection = HBFLifecycleAstraProjection(
        kind="migration",
        placement=placement,
        replica=replica,
        lifecycle_job_id=job.job_id,
        session_id=job.session_id,
        generation=job.generation,
        version=job.version,
        arrival_ns=job.start_ns,
        legacy_completion_ns=job.completion_ns,
        logical_bytes=job.logical_bytes,
        physical_bytes=job.physical_bytes,
        per_card_capacity_bytes=job.per_card_bytes,
        card_bytes=canonical_card_bytes(
            replica.card_ids, card_bytes),
        chunk_bytes=chunk,
        logical_chunks=logical_chunks,
        configured_rdma_fixed_latency_ns=rdma_fixed_ns,
        configured_hbf_write_fixed_latency_ns=write_fixed_ns,
        stages=tuple(stages),
        byte_ledger=HBFLifecycleByteLedger(
            gpu_source_pcie_bytes=job.logical_bytes,
            rdma_bytes=job.logical_bytes,
            destination_pcie_root_bytes=job.physical_bytes,
            destination_pcie_card_bytes=job.physical_bytes,
            hbf_write_bytes=job.physical_bytes,
        ),
        card_ledgers=_card_ledgers(
            replica=replica,
            card_bytes=card_bytes,
            kind="migration",
        ),
    )
    projection.validate()
    return projection


def build_append_hbf_astra_projection(
        *, job: AppendJob, hardware: HBFServerHardware,
        layout: HBFParallelLayout, chunk_bytes: int,
        server_id: int = 0,
) -> HBFLifecycleAstraProjection:
    """Project one LPDDR append as card-local pipelined ASTRA work."""

    hardware.validate()
    layout.validate(hardware.card_count)
    chunk = _positive_int("chunk_bytes", chunk_bytes)
    placement = build_hbf_server_placement(
        hardware=hardware, layout=layout, server_id=server_id)
    replica, card_bytes = _validate_job(
        job, layout=layout, placement=placement,
        expected_type=AppendJob)
    logical_chunks = _logical_chunks(job.logical_bytes, chunk)
    _validate_stage_upper_bound(
        chunk_count=len(logical_chunks),
        stages_per_chunk=2 * len(replica.card_ids),
    )
    physical_by_chunk = _chunk_card_bytes(
        logical_chunks=logical_chunks,
        replication_factor=layout.physical_kv_replication_factor,
        target_card_bytes=card_bytes,
    )

    prefix = f"hbf-server:{placement.server_id}"
    stage_prefix = f"append:{job.job_id}:replica:{replica.replica_id}"
    stages: list[HBFLifecycleAstraStage] = []
    previous_read: dict[int, str | None] = {
        card_id: None for card_id in replica.card_ids
    }
    previous_write: dict[int, str | None] = {
        card_id: None for card_id in replica.card_ids
    }
    write_started = {card_id: False for card_id in replica.card_ids}
    write_fixed_ns = _fixed_latency_ns(
        hardware.hbf_write_latency_us)

    for chunk_index, bytes_by_card in enumerate(physical_by_chunk):
        for card_id in replica.card_ids:
            byte_count = bytes_by_card.get(card_id, 0)
            if byte_count == 0:
                continue
            read_id = (
                f"{stage_prefix}:chunk:{chunk_index}:"
                f"card:{card_id}:lpddr-read")
            stages.append(_make_stage(
                stage_id=read_id,
                role="lpddr_read",
                chunk_index=chunk_index,
                byte_count=byte_count,
                bandwidth_gbps=(
                    hardware.lpddr_bandwidth_gbps_per_card),
                resources=(f"{prefix}:card:{card_id}:lpddr",),
                dependencies=(
                    (previous_read[card_id],)
                    if previous_read[card_id] else ()),
                card_id=card_id,
            ))
            previous_read[card_id] = read_id

            write_id = (
                f"{stage_prefix}:chunk:{chunk_index}:"
                f"card:{card_id}:hbf-write")
            write_dependencies = [read_id]
            if previous_write[card_id] is not None:
                write_dependencies.append(previous_write[card_id])
            fixed = 0 if write_started[card_id] else write_fixed_ns
            stages.append(_make_stage(
                stage_id=write_id,
                role="hbf_write",
                chunk_index=chunk_index,
                byte_count=byte_count,
                bandwidth_gbps=(
                    hardware.hbf_write_bandwidth_gbps_per_card),
                fixed_latency_ns=fixed,
                resources=(f"{prefix}:card:{card_id}:hbf-read",),
                dependencies=tuple(write_dependencies),
                card_id=card_id,
            ))
            write_started[card_id] = True
            previous_write[card_id] = write_id

    projection = HBFLifecycleAstraProjection(
        kind="append",
        placement=placement,
        replica=replica,
        lifecycle_job_id=job.job_id,
        session_id=job.session_id,
        generation=job.generation,
        version=job.version,
        arrival_ns=job.start_ns,
        legacy_completion_ns=job.completion_ns,
        logical_bytes=job.logical_bytes,
        physical_bytes=job.physical_bytes,
        per_card_capacity_bytes=job.per_card_bytes,
        card_bytes=canonical_card_bytes(
            replica.card_ids, card_bytes),
        chunk_bytes=chunk,
        logical_chunks=logical_chunks,
        configured_rdma_fixed_latency_ns=0,
        configured_hbf_write_fixed_latency_ns=write_fixed_ns,
        stages=tuple(stages),
        byte_ledger=HBFLifecycleByteLedger(
            lpddr_read_bytes=job.physical_bytes,
            hbf_write_bytes=job.physical_bytes,
        ),
        card_ledgers=_card_ledgers(
            replica=replica,
            card_bytes=card_bytes,
            kind="append",
        ),
    )
    projection.validate()
    return projection


__all__ = [
    "ASTRA_BACKGROUND_STAGE_LIMIT",
    "HBFCardLifecycleByteLedger",
    "HBFLifecycleAstraProjection",
    "HBFLifecycleAstraProjectionError",
    "HBFLifecycleAstraStage",
    "HBFLifecycleByteLedger",
    "HBF_WRITE_FIXED_LATENCY_SEMANTICS",
    "LPDDR_FIXED_LATENCY_SEMANTICS",
    "PCIE_FIXED_LATENCY_SEMANTICS",
    "PROJECTION_FIDELITY",
    "PROJECTION_SCHEMA",
    "RDMA_FIXED_LATENCY_SEMANTICS",
    "build_append_hbf_astra_projection",
    "build_migration_hbf_astra_projection",
]

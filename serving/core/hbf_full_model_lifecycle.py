"""Versioned lifecycle and resource accounting for full-model HBF serving."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import heapq
import math
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from .hbf_full_model_astra import (
    ASTRA_NAMED_RESOURCE_TIMING_SEMANTICS,
    HBFAstraTimingAccounting,
    HBFModelAstraProjectionError,
    validate_hbf_astra_timing_metrics,
)
from .hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
    qwen_logical_kv_bytes_per_token,
    qwen_model_weight_bytes_per_rank,
)
from .hbf_pcie_topology import HBFPCIeTopology

if TYPE_CHECKING:
    from .hbf_full_model_lifecycle_astra import (
        HBFLifecycleAstraProjection,
    )


class PlacementState(str, Enum):
    GPU_ACTIVE = "gpu_active"
    GPU_READY = "gpu_ready"
    SSD_READY = "ssd_ready"
    MIGRATING = "migrating"
    HBF_READY = "hbf_ready"
    HBF_ACTIVE = "hbf_active"
    EVICTED = "evicted"
    ENDED = "ended"


class ResumeExecution(str, Enum):
    GPU = "gpu"
    GPU_RESTORE = "gpu_restore"
    GPU_RECOMPUTE = "gpu_recompute"
    HBF = "hbf"


class MigrationSourceKind(str, Enum):
    """Authoritative snapshot source for one HBF migration job."""

    GPU = "gpu"
    SSD = "ssd"


class ActivePrefillDrainStatus(str, Enum):
    """Outcome of one active HBF prefill-drain attempt."""

    SATISFIED = "satisfied"
    STARTED = "started"
    WAIT_EXISTING_APPEND = "wait_existing_append"
    CAPACITY_FALLBACK = "capacity_fallback"


def hbf_request_headroom_owner(request_id: int) -> str:
    """Return the shared-ledger owner for one active request's KV growth."""

    if isinstance(request_id, bool) or not isinstance(request_id, int):
        raise ValueError("request_id must be an integer")
    if request_id < 0:
        raise ValueError("request_id must be non-negative")
    return f"hbf-request-headroom:{request_id}"


def hbf_kv_range_card_bytes(
        *, layout: HBFParallelLayout, card_ids: Sequence[int],
        kv_bytes_per_token: int, token_start: int,
        token_count: int) -> dict[int, int]:
    """Return exact physical KV bytes for one sequence-local token range.

    Conventional layouts use an additive byte stripe across their TP cards.
    ``tp8_context`` instead maps each of the four KV heads to one card pair
    and maps even/odd sequence positions to pair rank zero/one.  Keeping
    ``token_start`` explicit is essential for turn-boundary appends.
    """

    cards = tuple(card_ids)
    if (
        not cards
        or len(cards) != layout.tp_size
        or len(cards) != len(set(cards))
        or any(
            isinstance(card_id, bool)
            or not isinstance(card_id, int)
            or card_id < 0
            for card_id in cards
        )
    ):
        raise ValueError(
            "card_ids must be unique non-negative integers matching "
            "the layout TP group")
    for name, value in (
        ("kv_bytes_per_token", kv_bytes_per_token),
        ("token_start", token_start),
        ("token_count", token_count),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer")
    if kv_bytes_per_token == 0:
        raise ValueError("kv_bytes_per_token must be positive")
    result = {card_id: 0 for card_id in cards}
    if token_count == 0:
        return result

    if layout.is_context_striped:
        if len(cards) != 8:
            raise ValueError("tp8_context requires eight physical cards")
        head_base, head_remainder = divmod(kv_bytes_per_token, 4)
        head_bytes = tuple(
            head_base + (1 if index < head_remainder else 0)
            for index in range(4)
        )

        def even_prefix(length: int) -> int:
            return (length + 1) // 2

        even_tokens = (
            even_prefix(token_start + token_count)
            - even_prefix(token_start)
        )
        odd_tokens = token_count - even_tokens
        for pair_index, byte_count in enumerate(head_bytes):
            result[cards[2 * pair_index]] = even_tokens * byte_count
            result[cards[2 * pair_index + 1]] = (
                odd_tokens * byte_count)
        if sum(result.values()) != token_count * kv_bytes_per_token:
            raise AssertionError("context KV placement lost physical bytes")
        return result

    factor = layout.physical_kv_replication_factor
    physical_start = token_start * kv_bytes_per_token * factor
    physical_bytes = token_count * kv_bytes_per_token * factor
    quotient, remainder = divmod(physical_bytes, len(cards))
    start_index = physical_start % len(cards)
    for card_id in cards:
        result[card_id] = quotient
    for offset in range(remainder):
        result[cards[(start_index + offset) % len(cards)]] += 1
    if sum(result.values()) != physical_bytes:
        raise AssertionError("conventional KV placement lost physical bytes")
    return result


def canonical_card_bytes(
        card_ids: Sequence[int],
        card_bytes: Mapping[int, int]) -> tuple[tuple[int, int], ...]:
    """Freeze one physical-card byte vector in placement order."""

    cards = tuple(card_ids)
    if set(card_bytes) != set(cards):
        raise ValueError("card byte vector must cover every replica card")
    result = []
    for card_id in cards:
        byte_count = card_bytes[card_id]
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ValueError(
                "card byte vector values must be non-negative integers")
        result.append((card_id, byte_count))
    return tuple(result)


@dataclass(frozen=True)
class ResourceReservation:
    namespace: str
    resource: str
    start_ns: int
    end_ns: int
    service_ns: int
    byte_count: int
    job_id: int
    kind: str


class ResourceCalendar:
    """Deterministic gang reservation over named serial resources."""

    def __init__(self, *, retain_reservations: bool = True) -> None:
        if not isinstance(retain_reservations, bool):
            raise ValueError("retain_reservations must be a boolean")
        self.retain_reservations = retain_reservations
        self.available_ns: dict[str, int] = {}
        self.busy_ns: dict[str, int] = {}
        self.reservations: list[ResourceReservation] = []
        self.reservation_count_by_resource: dict[str, int] = {}
        self.reservation_bytes_by_resource: dict[str, int] = {}
        self.reservation_count_by_namespace_kind: dict[
            tuple[str, str], int] = {}
        self.reservation_bytes_by_namespace_kind: dict[
            tuple[str, str], int] = {}

    def reserve_parallel(
            self, *, arrival_ns: int, job_id: int, kind: str,
            demands: Mapping[str, tuple[int, int]],
            namespace: str = "default") -> tuple[int, int]:
        if arrival_ns < 0:
            raise ValueError("arrival_ns must be non-negative")
        if not namespace:
            raise ValueError("reservation namespace must be non-empty")
        if not demands:
            return arrival_ns, arrival_ns
        for resource, (service_ns, byte_count) in demands.items():
            if not resource:
                raise ValueError("resource name must be non-empty")
            if service_ns < 0 or byte_count < 0:
                raise ValueError("resource demand must be non-negative")
        start_ns = max(
            arrival_ns,
            max(self.available_ns.get(resource, 0)
                for resource in demands),
        )
        end_ns = start_ns
        for resource, (service_ns, byte_count) in sorted(demands.items()):
            resource_end = start_ns + service_ns
            self.available_ns[resource] = resource_end
            self.busy_ns[resource] = (
                self.busy_ns.get(resource, 0) + service_ns)
            self.reservation_count_by_resource[resource] = (
                self.reservation_count_by_resource.get(resource, 0) + 1)
            self.reservation_bytes_by_resource[resource] = (
                self.reservation_bytes_by_resource.get(resource, 0)
                + byte_count
            )
            namespace_kind = (namespace, kind)
            self.reservation_count_by_namespace_kind[namespace_kind] = (
                self.reservation_count_by_namespace_kind.get(
                    namespace_kind, 0
                ) + 1
            )
            self.reservation_bytes_by_namespace_kind[namespace_kind] = (
                self.reservation_bytes_by_namespace_kind.get(
                    namespace_kind, 0
                ) + byte_count
            )
            if self.retain_reservations:
                self.reservations.append(ResourceReservation(
                    namespace=namespace,
                    resource=resource,
                    start_ns=start_ns,
                    end_ns=resource_end,
                    service_ns=service_ns,
                    byte_count=byte_count,
                    job_id=job_id,
                    kind=kind,
                ))
            end_ns = max(end_ns, resource_end)
        return start_ns, end_ns

    def earliest_start(
            self, arrival_ns: int, resources: Sequence[str]) -> int:
        if arrival_ns < 0:
            raise ValueError("arrival_ns must be non-negative")
        if not resources:
            return arrival_ns
        if any(not resource for resource in resources):
            raise ValueError("resource name must be non-empty")
        return max(
            arrival_ns,
            max(self.available_ns.get(resource, 0)
                for resource in resources),
        )

    def utilization(self, resource: str, horizon_ns: int) -> float:
        if horizon_ns <= 0:
            return 0.0
        return self.busy_ns.get(resource, 0) / horizon_ns

    def report(self) -> dict[str, Any]:
        resources = sorted(
            set(self.available_ns)
            | set(self.busy_ns)
            | set(self.reservation_count_by_resource)
            | set(self.reservation_bytes_by_resource)
        )
        return {
            "retain_reservations": self.retain_reservations,
            "retained_reservation_count": len(self.reservations),
            "resources": {
                resource: {
                    "available_ns": self.available_ns.get(resource, 0),
                    "busy_ns": self.busy_ns.get(resource, 0),
                    "reservation_count": (
                        self.reservation_count_by_resource.get(resource, 0)
                    ),
                    "reservation_bytes": (
                        self.reservation_bytes_by_resource.get(resource, 0)
                    ),
                }
                for resource in resources
            },
            "namespace_kinds": [
                {
                    "namespace": namespace,
                    "kind": kind,
                    "reservation_count": (
                        self.reservation_count_by_namespace_kind[
                            (namespace, kind)]
                    ),
                    "reservation_bytes": (
                        self.reservation_bytes_by_namespace_kind[
                            (namespace, kind)]
                    ),
                }
                for namespace, kind in sorted(
                    self.reservation_count_by_namespace_kind)
            ],
        }


class PerGroupCapacityLedger:
    """Shared, exact physical-card capacity ownership for replica groups.

    The scalar methods remain backward compatible: ``set_bytes`` reserves
    the same byte count on every card in the selected group, while
    ``owner_bytes`` and ``used_bytes`` return the maximum physical-card
    occupancy.  Context-striped TP8 uses the vector methods so sequence-local
    even/odd placement is not averaged across the two cards in each KV-head
    pair.
    """

    def __init__(
            self, *, group_count: int, capacity_bytes: int,
            card_ids_by_group: Optional[
                Mapping[int, Sequence[int]]
            ] = None) -> None:
        if group_count <= 0 or capacity_bytes <= 0:
            raise ValueError(
                "capacity ledger dimensions must be positive")
        self.group_count = group_count
        self.capacity_bytes = capacity_bytes
        self._topology_explicit = card_ids_by_group is not None
        if card_ids_by_group is None:
            normalized_cards = {
                group_id: (group_id,)
                for group_id in range(group_count)
            }
        else:
            if set(card_ids_by_group) != set(range(group_count)):
                raise ValueError(
                    "capacity ledger card topology must cover every group")
            normalized_cards = {}
            covered: set[int] = set()
            for group_id in range(group_count):
                cards = tuple(card_ids_by_group[group_id])
                if (
                    not cards
                    or len(cards) != len(set(cards))
                    or any(
                        isinstance(card_id, bool)
                        or not isinstance(card_id, int)
                        or card_id < 0
                        for card_id in cards
                    )
                ):
                    raise ValueError(
                        "capacity ledger card ids must be non-negative "
                        "and unique")
                if covered & set(cards):
                    raise ValueError(
                        "capacity ledger cards cannot span two groups")
                covered.update(cards)
                normalized_cards[group_id] = cards
        self._card_ids_by_group = normalized_cards
        self._reservations: dict[int, dict[str, int]] = {
            group_id: {} for group_id in range(group_count)
        }
        self._reservation_vectors: dict[
            int, dict[str, dict[int, int]]
        ] = {
            group_id: {} for group_id in range(group_count)
        }
        self._used_by_card: dict[int, dict[int, int]] = {
            group_id: {
                card_id: 0
                for card_id in self._card_ids_by_group[group_id]
            }
            for group_id in range(group_count)
        }
        self._owner_group: dict[str, int] = {}
        self.peak_used_bytes: dict[int, int] = {
            group_id: 0 for group_id in range(group_count)
        }
        self.peak_used_bytes_by_card: dict[int, dict[int, int]] = {
            group_id: dict(self._used_by_card[group_id])
            for group_id in range(group_count)
        }

    def _validate_group(self, group_id: int) -> None:
        if not 0 <= group_id < self.group_count:
            raise ValueError(f"invalid capacity-ledger group {group_id}")

    def configure_cards(
            self, card_ids_by_group: Mapping[int, Sequence[int]]) -> None:
        """Attach physical cards, expanding legacy scalar reservations.

        A shared ledger is often constructed before its lifecycle and pool.
        The first consumer may therefore replace the implicit one-card
        topology. Existing scalar reservations are interpreted as uniform
        per-card ownership, preserving their historical contract.
        """

        candidate = PerGroupCapacityLedger(
            group_count=self.group_count,
            capacity_bytes=self.capacity_bytes,
            card_ids_by_group=card_ids_by_group,
        )
        requested = candidate._card_ids_by_group
        if self._topology_explicit:
            if requested != self._card_ids_by_group:
                raise ValueError(
                    "capacity ledger card topology does not match layout")
            return
        old_rows = [
            (group_id, owner, byte_count)
            for group_id, reservations in self._reservations.items()
            for owner, byte_count in reservations.items()
        ]
        self._card_ids_by_group = requested
        self._reservation_vectors = {
            group_id: {} for group_id in range(self.group_count)
        }
        self._used_by_card = {
            group_id: {
                card_id: 0 for card_id in requested[group_id]
            }
            for group_id in range(self.group_count)
        }
        self.peak_used_bytes_by_card = {
            group_id: dict(self._used_by_card[group_id])
            for group_id in range(self.group_count)
        }
        self._reservations = {
            group_id: {} for group_id in range(self.group_count)
        }
        self._owner_group.clear()
        self.peak_used_bytes = {
            group_id: 0 for group_id in range(self.group_count)
        }
        self._topology_explicit = True
        for group_id, owner, byte_count in old_rows:
            self.set_bytes(group_id, owner, byte_count)

    def card_ids(self, group_id: int) -> tuple[int, ...]:
        self._validate_group(group_id)
        return self._card_ids_by_group[group_id]

    def _normalize_card_bytes(
            self, group_id: int,
            card_bytes: Mapping[int, int]) -> dict[int, int]:
        self._validate_group(group_id)
        if not isinstance(card_bytes, Mapping):
            raise ValueError("card_bytes must be a mapping")
        expected = set(self._card_ids_by_group[group_id])
        if set(card_bytes) - expected:
            raise ValueError(
                "capacity reservation references a card outside its group")
        result = {}
        for card_id in self._card_ids_by_group[group_id]:
            value = card_bytes.get(card_id, 0)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    "capacity-ledger card bytes must be non-negative "
                    "integers")
            result[card_id] = value
        return result

    def used_bytes_by_card(self, group_id: int) -> Mapping[int, int]:
        self._validate_group(group_id)
        return dict(self._used_by_card[group_id])

    def used_bytes(self, group_id: int) -> int:
        self._validate_group(group_id)
        return max(self._used_by_card[group_id].values(), default=0)

    def owner_card_bytes(self, owner: str) -> Mapping[int, int]:
        group_id = self._owner_group.get(owner)
        if group_id is None:
            return {}
        return dict(self._reservation_vectors[group_id][owner])

    def owner_bytes(self, owner: str) -> int:
        return max(self.owner_card_bytes(owner).values(), default=0)

    def owner_group(self, owner: str) -> Optional[int]:
        return self._owner_group.get(owner)

    def can_set(self, group_id: int, owner: str, byte_count: int) -> bool:
        if (
            not owner
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ValueError(
                "capacity-ledger owner/bytes are invalid")
        return self.can_set_card_bytes(
            group_id,
            owner,
            {
                card_id: byte_count
                for card_id in self.card_ids(group_id)
            },
        )

    def can_set_card_bytes(
            self, group_id: int, owner: str,
            card_bytes: Mapping[int, int]) -> bool:
        if not owner:
            raise ValueError("capacity-ledger owner must be non-empty")
        normalized = self._normalize_card_bytes(group_id, card_bytes)
        current_group = self._owner_group.get(owner)
        current = (
            self._reservation_vectors[current_group][owner]
            if current_group is not None else {}
        )
        return all(
            (
                self._used_by_card[group_id][card_id]
                - (
                    current.get(card_id, 0)
                    if current_group == group_id else 0
                )
                + normalized[card_id]
            ) <= self.capacity_bytes
            for card_id in self.card_ids(group_id)
        )

    def set_bytes(
            self, group_id: int, owner: str, byte_count: int) -> None:
        self.set_card_bytes(
            group_id,
            owner,
            {
                card_id: byte_count
                for card_id in self.card_ids(group_id)
            },
        )

    def set_card_bytes(
            self, group_id: int, owner: str,
            card_bytes: Mapping[int, int]) -> None:
        normalized = self._normalize_card_bytes(group_id, card_bytes)
        if not self.can_set_card_bytes(group_id, owner, normalized):
            raise RuntimeError(
                f"LPDDR capacity exceeded: group={group_id}, "
                f"owner={owner!r}, requested={normalized}, "
                f"used={self.used_bytes_by_card(group_id)}, "
                f"capacity={self.capacity_bytes}")
        old_group = self._owner_group.get(owner)
        if old_group is not None:
            old_vector = self._reservation_vectors[
                old_group].pop(owner)
            for card_id, byte_count in old_vector.items():
                self._used_by_card[old_group][card_id] -= byte_count
            self._reservations[old_group].pop(owner)
            self._owner_group.pop(owner)
        if any(normalized.values()):
            self._reservation_vectors[group_id][owner] = normalized
            for card_id, byte_count in normalized.items():
                self._used_by_card[group_id][card_id] += byte_count
            self._reservations[group_id][owner] = max(
                normalized.values(), default=0)
            self._owner_group[owner] = group_id
        self.peak_used_bytes[group_id] = max(
            self.peak_used_bytes[group_id],
            self.used_bytes(group_id),
        )
        for card_id, byte_count in self._used_by_card[group_id].items():
            self.peak_used_bytes_by_card[group_id][card_id] = max(
                self.peak_used_bytes_by_card[group_id][card_id],
                byte_count,
            )

    def release(self, owner: str) -> None:
        old_group = self._owner_group.pop(owner, None)
        if old_group is not None:
            vector = self._reservation_vectors[old_group].pop(owner)
            for card_id, byte_count in vector.items():
                self._used_by_card[old_group][card_id] -= byte_count
            self._reservations[old_group].pop(owner)

    def shrink(self, owner: str, byte_count: int) -> None:
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ValueError("capacity-ledger shrink must be non-negative")
        group_id = self._owner_group.get(owner)
        if group_id is None:
            if byte_count:
                raise RuntimeError(
                    f"LPDDR reservation underflow: owner={owner!r}")
            return
        self.shrink_card_bytes(
            owner,
            {
                card_id: byte_count
                for card_id in self.card_ids(group_id)
            },
        )

    def shrink_card_bytes(
            self, owner: str,
            card_bytes: Mapping[int, int]) -> None:
        group_id = self._owner_group.get(owner)
        if group_id is None:
            if any(card_bytes.values()):
                raise RuntimeError(
                    f"LPDDR reservation underflow: owner={owner!r}")
            return
        release = self._normalize_card_bytes(group_id, card_bytes)
        current = dict(self._reservation_vectors[group_id][owner])
        if any(
                release[card_id] > current[card_id]
                for card_id in self.card_ids(group_id)):
            raise RuntimeError(
                f"LPDDR reservation underflow: owner={owner!r}, "
                f"release={release}, reserved={current}")
        self.set_card_bytes(
            group_id,
            owner,
            {
                card_id: current[card_id] - release[card_id]
                for card_id in self.card_ids(group_id)
            },
        )

    def reservations(self, group_id: int) -> Mapping[str, int]:
        self._validate_group(group_id)
        return dict(self._reservations[group_id])

    def card_reservations(
            self, group_id: int,
    ) -> Mapping[str, Mapping[int, int]]:
        self._validate_group(group_id)
        return {
            owner: dict(card_bytes)
            for owner, card_bytes in
            self._reservation_vectors[group_id].items()
        }

    def assert_invariants(self) -> None:
        for group_id in range(self.group_count):
            recomputed = {
                card_id: 0 for card_id in self.card_ids(group_id)
            }
            for owner, byte_count in self._reservations[group_id].items():
                if byte_count <= 0:
                    raise AssertionError(
                        "capacity ledger has a non-positive reservation "
                        f"for {owner!r}")
                if self._owner_group.get(owner) != group_id:
                    raise AssertionError(
                        f"capacity owner index mismatch for {owner!r}")
                vector = self._reservation_vectors[
                    group_id].get(owner)
                if vector is None:
                    raise AssertionError(
                        f"capacity owner lacks card vector for {owner!r}")
                if byte_count != max(vector.values(), default=0):
                    raise AssertionError(
                        f"capacity scalar/vector mismatch for {owner!r}")
                for card_id, value in vector.items():
                    if value < 0:
                        raise AssertionError(
                            f"negative card reservation for {owner!r}")
                    recomputed[card_id] += value
            if set(self._reservation_vectors[group_id]) != set(
                    self._reservations[group_id]):
                raise AssertionError(
                    "capacity vector/scalar owner index mismatch")
            if recomputed != self._used_by_card[group_id]:
                raise AssertionError(
                    f"capacity card usage mismatch: group={group_id}")
            for card_id, used in recomputed.items():
                if not 0 <= used <= self.capacity_bytes:
                    raise AssertionError(
                        f"capacity ledger overflow: group={group_id}, "
                        f"card={card_id}, used={used}, "
                        f"capacity={self.capacity_bytes}")
        indexed_owners = {
            owner
            for reservations in self._reservations.values()
            for owner in reservations
        }
        if indexed_owners != set(self._owner_group):
            raise AssertionError(
                "capacity owner/group reverse index mismatch")


@dataclass(frozen=True)
class HBFReplicaGroup:
    group_id: int
    card_ids: tuple[int, ...]


@dataclass
class SessionPlacement:
    session_id: str
    state: PlacementState = PlacementState.GPU_ACTIVE
    generation: int = 0
    version: int = 0
    total_tokens: int = 0
    committed_hbf_tokens: int = 0
    lpddr_tokens: int = 0
    group_id: Optional[int] = None
    gpu_retained_bytes: int = 0
    committed_per_card_bytes: int = 0
    pending_reserved_per_card_bytes: int = 0
    last_access_ns: int = 0
    active_request_id: Optional[int] = None
    migration_source_kind: Optional[MigrationSourceKind] = None
    migration_job_ids: set[int] = field(default_factory=set)
    append_job_ids: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class MigrationJob:
    job_id: int
    session_id: str
    generation: int
    version: int
    group_id: int
    token_count: int
    logical_bytes: int
    physical_bytes: int
    per_card_bytes: int
    start_ns: int
    completion_ns: int
    token_start: int = 0
    card_bytes: tuple[tuple[int, int], ...] = ()
    source_kind: MigrationSourceKind = MigrationSourceKind.GPU


@dataclass(frozen=True)
class AppendJob:
    job_id: int
    session_id: str
    generation: int
    version: int
    group_id: int
    token_count: int
    logical_bytes: int
    physical_bytes: int
    per_card_bytes: int
    start_ns: int
    completion_ns: int
    token_start: int = 0
    card_bytes: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class ActivePrefillDrainResult:
    """Explicit policy result for draining active fresh-prefill KV."""

    status: ActivePrefillDrainStatus
    job: Optional[AppendJob]
    total_tokens: int
    lpddr_tokens: int
    append_tokens: int
    retained_tail_tokens: int
    blocking_append_job_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class HBFLifecycleExternalDispatch:
    """One immutable lifecycle transfer awaiting ASTRA completion."""

    arrival_ns: int
    job: MigrationJob | AppendJob
    projection: "HBFLifecycleAstraProjection"

    @property
    def job_id(self) -> str:
        return self.projection.job_id

    @property
    def stage_count(self) -> int:
        return len(self.projection.stages)

    def controller_arguments(
            self,
    ) -> tuple[str, int, tuple[dict[str, Any], ...]]:
        return self.projection.controller_command_arguments(
            self.arrival_ns)


@dataclass(frozen=True)
class ResumeRoute:
    execution: ResumeExecution
    session_id: str
    group_id: Optional[int]
    hbf_tokens: int
    lpddr_tokens: int
    migration_inflight: bool
    reason: str


@dataclass(frozen=True)
class GPUReadyPressureEviction:
    """One idle GPU lineage dropped to unblock finite GPU HBM."""

    session_id: str
    eviction_ns: int
    last_access_ns: int
    token_count: int
    logical_bytes: int
    generation_before: int
    generation_after: int


@dataclass
class LifecycleMetrics:
    migrations_started: int = 0
    migrations_committed: int = 0
    migrations_stale: int = 0
    migration_logical_bytes: int = 0
    migration_physical_bytes: int = 0
    migration_wasted_physical_bytes: int = 0
    ssd_checkpoints_published: int = 0
    ssd_imports_started: int = 0
    ssd_imports_committed: int = 0
    ssd_imports_stale: int = 0
    ssd_import_logical_bytes: int = 0
    ssd_import_physical_bytes: int = 0
    ssd_import_wasted_physical_bytes: int = 0
    ssd_restore_resumes: int = 0
    gpu_fallback_resumes: int = 0
    gpu_recompute_resumes: int = 0
    lpddr_capacity_fallback_resumes: int = 0
    hbf_resumes: int = 0
    append_jobs_started: int = 0
    append_jobs_committed: int = 0
    append_jobs_stale: int = 0
    append_logical_bytes: int = 0
    append_physical_bytes: int = 0
    append_wasted_physical_bytes: int = 0
    active_prefill_drain_candidates: int = 0
    active_prefill_drain_started: int = 0
    active_prefill_drain_satisfied: int = 0
    active_prefill_drain_wait_existing_append: int = 0
    active_prefill_drain_capacity_fallback: int = 0
    active_prefill_drain_committed: int = 0
    active_prefill_drain_stale: int = 0
    capacity_evictions: int = 0
    gpu_ready_hbm_pressure_evictions: int = 0
    gpu_ready_hbm_pressure_evicted_bytes: int = 0
    gpu_retained_bytes_peak: int = 0
    hbf_reserved_bytes_peak: int = 0
    astra_completed_jobs: int = 0
    astra_completion_elapsed_ns: int = 0
    astra_resource_delay_ns: int = 0
    astra_dependency_critical_path_ns: int = 0
    astra_solo_resource_serialized_completion_ns: int = 0
    astra_actual_resource_serialized_completion_ns: int = 0
    astra_internal_resource_serialization_wait_ns: int = 0
    astra_signed_interference_delta_ns: int = 0


class FullModelHBFLifecycle:
    """Own session placement, migrations, appends, and HBF capacity."""

    _EXECUTION_BACKENDS = frozenset({
        "analytical_calendar",
        "external_astra",
    })

    def __init__(
            self, *, hardware: HBFServerHardware,
            layout: HBFParallelLayout,
            resource_calendar: Optional[ResourceCalendar] = None,
            lpddr_ledger: Optional[PerGroupCapacityLedger] = None,
            kv_bytes_per_token: Optional[int] = None,
            model_weight_bytes_per_rank: Optional[int] = None,
            gpu_source_root_bandwidth_gbps: float = 200.0,
            validate_every_event: bool = True,
            execution_backend: str = "analytical_calendar",
            server_id: int = 0,
            astra_chunk_bytes: int = 64 * 1024 ** 2,
            analytical_resource_prefix: str = "",
            gpu_source_node_id: int = 0,
            gpu_source_cpu_bandwidth_gbps: float = 200.0,
            gpu_source_nic_bandwidth_gbps: Optional[float] = None) -> None:
        hardware.validate()
        layout.validate(hardware.card_count)
        pcie_topology = HBFPCIeTopology.from_hardware(
            hardware, server_id=server_id)
        pcie_topology.validate_layout(
            layout_key=layout.key,
            tp_size=layout.tp_size,
            replicas=layout.replicas,
        )
        if (
            not isinstance(execution_backend, str)
            or execution_backend not in self._EXECUTION_BACKENDS
        ):
            raise ValueError(
                "execution_backend must be one of "
                f"{sorted(self._EXECUTION_BACKENDS)}")
        if not isinstance(analytical_resource_prefix, str):
            raise ValueError(
                "analytical_resource_prefix must be a string")
        if (
            execution_backend == "external_astra"
            and analytical_resource_prefix
        ):
            raise ValueError(
                "analytical_resource_prefix is unsupported with "
                "execution_backend='external_astra'")
        if (
            not isinstance(server_id, int)
            or isinstance(server_id, bool)
            or server_id < 0
        ):
            raise ValueError("server_id must be a non-negative integer")
        if (
            not isinstance(gpu_source_node_id, int)
            or isinstance(gpu_source_node_id, bool)
            or gpu_source_node_id < 0
        ):
            raise ValueError(
                "gpu_source_node_id must be a non-negative integer")
        if (
            not isinstance(astra_chunk_bytes, int)
            or isinstance(astra_chunk_bytes, bool)
            or astra_chunk_bytes <= 0
        ):
            raise ValueError(
                "astra_chunk_bytes must be a positive integer")
        if (
            execution_backend == "external_astra"
            and resource_calendar is not None
        ):
            raise ValueError(
                "external_astra owns lifecycle resource timing in ASTRA; "
                "resource_calendar must be omitted")
        if not isinstance(validate_every_event, bool):
            raise ValueError("validate_every_event must be a boolean")
        if (
            isinstance(gpu_source_root_bandwidth_gbps, bool)
            or not isinstance(
                gpu_source_root_bandwidth_gbps, (int, float))
            or not math.isfinite(
                float(gpu_source_root_bandwidth_gbps))
            or gpu_source_root_bandwidth_gbps <= 0
        ):
            raise ValueError(
                "gpu_source_root_bandwidth_gbps must be positive "
                "and finite")
        for name, value in (
            (
                "gpu_source_cpu_bandwidth_gbps",
                gpu_source_cpu_bandwidth_gbps,
            ),
            (
                "gpu_source_nic_bandwidth_gbps",
                (
                    hardware.rdma_bandwidth_gbps
                    if gpu_source_nic_bandwidth_gbps is None
                    else gpu_source_nic_bandwidth_gbps
                ),
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        self.hardware = hardware
        self.layout = layout
        self.pcie_topology = pcie_topology
        self.execution_backend = execution_backend
        self.server_id = server_id
        self.astra_chunk_bytes = astra_chunk_bytes
        self.analytical_resource_prefix = analytical_resource_prefix
        self.gpu_source_node_id = gpu_source_node_id
        self.validate_every_event = validate_every_event
        self.gpu_source_root_bandwidth_gbps = float(
            gpu_source_root_bandwidth_gbps)
        self.gpu_source_cpu_bandwidth_gbps = float(
            gpu_source_cpu_bandwidth_gbps)
        self.gpu_source_nic_bandwidth_gbps = float(
            hardware.rdma_bandwidth_gbps
            if gpu_source_nic_bandwidth_gbps is None
            else gpu_source_nic_bandwidth_gbps
        )
        self.calendar = (
            resource_calendar
            if resource_calendar is not None else ResourceCalendar()
        )
        self.kv_bytes_per_token = (
            qwen_logical_kv_bytes_per_token()
            if kv_bytes_per_token is None else kv_bytes_per_token
        )
        if self.kv_bytes_per_token <= 0:
            raise ValueError("kv_bytes_per_token must be positive")
        self.weight_bytes_per_rank = (
            qwen_model_weight_bytes_per_rank(layout.tp_size)
            if model_weight_bytes_per_rank is None
            else model_weight_bytes_per_rank
        )
        if not 0 <= self.weight_bytes_per_rank < (
                hardware.hbf_capacity_bytes_per_card):
            raise ValueError(
                "model weights must fit in each HBF card")
        self.groups = tuple(
            HBFReplicaGroup(
                group_id=index,
                card_ids=tuple(range(
                    index * layout.tp_size,
                    (index + 1) * layout.tp_size,
                )),
            )
            for index in range(layout.replicas)
        )
        card_ids = tuple(
            card_id
            for group in self.groups
            for card_id in group.card_ids
        )
        self._hbf_kv_write_bytes_by_card = {
            card_id: 0 for card_id in card_ids
        }
        self._hbf_migration_write_bytes_by_card = {
            card_id: 0 for card_id in card_ids
        }
        self._hbf_append_write_bytes_by_card = {
            card_id: 0 for card_id in card_ids
        }
        self._hbf_ssd_import_write_bytes_by_card = {
            card_id: 0 for card_id in card_ids
        }
        self._hbf_migration_wasted_write_bytes_by_card = {
            card_id: 0 for card_id in card_ids
        }
        self._hbf_append_wasted_write_bytes_by_card = {
            card_id: 0 for card_id in card_ids
        }
        self._hbf_ssd_import_wasted_write_bytes_by_card = {
            card_id: 0 for card_id in card_ids
        }
        self.lpddr_ledger = (
            lpddr_ledger
            if lpddr_ledger is not None
            else PerGroupCapacityLedger(
                group_count=layout.replicas,
                capacity_bytes=hardware.lpddr_capacity_bytes_per_card,
                card_ids_by_group={
                    group.group_id: group.card_ids
                    for group in self.groups
                },
            )
        )
        if self.lpddr_ledger.group_count != layout.replicas:
            raise ValueError(
                "LPDDR ledger group count does not match HBF layout")
        self.lpddr_ledger.configure_cards({
            group.group_id: group.card_ids
            for group in self.groups
        })
        if self.lpddr_ledger.capacity_bytes > (
                hardware.lpddr_capacity_bytes_per_card):
            raise ValueError(
                "LPDDR ledger capacity exceeds physical LPDDR")
        self.sessions: dict[str, SessionPlacement] = {}
        self.metrics = LifecycleMetrics()
        self._jobs: dict[int, MigrationJob | AppendJob] = {}
        self._active_prefill_drain_job_ids: set[int] = set()
        self._completion_heap: list[tuple[int, int]] = []
        self._external_outbox: deque[
            HBFLifecycleExternalDispatch] = deque()
        self._external_pending: dict[
            str, HBFLifecycleExternalDispatch] = {}
        self._external_issued_job_ids: set[str] = set()
        self._external_completed_job_ids: set[str] = set()
        self._next_job_id = 1
        self.current_ns = 0
        self._reserved_bytes_by_card = {
            group.group_id: {
                card_id: 0 for card_id in group.card_ids
            }
            for group in self.groups
        }
        self._reserved_per_card_by_group = {
            group.group_id: 0 for group in self.groups
        }

    @property
    def usable_bytes_per_card(self) -> int:
        return (
            self.hardware.hbf_capacity_bytes_per_card
            - self.weight_bytes_per_rank
        )

    def _next_id(self) -> int:
        result = self._next_job_id
        self._next_job_id += 1
        return result

    def _enqueue_external_job(
            self, job: MigrationJob | AppendJob,
            arrival_ns: int) -> None:
        if self.execution_backend != "external_astra":
            raise RuntimeError(
                "external lifecycle dispatch requires "
                "execution_backend='external_astra'")
        # Local imports keep the lifecycle job records available to the
        # projection module without creating an import cycle at module load.
        from .hbf_full_model_lifecycle_astra import (
            build_append_hbf_astra_projection,
            build_migration_hbf_astra_projection,
        )

        if isinstance(job, MigrationJob):
            if job.source_kind != MigrationSourceKind.GPU:
                raise RuntimeError(
                    "external ASTRA lifecycle does not support "
                    "SSD-origin migration jobs")
            projection = build_migration_hbf_astra_projection(
                job=job,
                hardware=self.hardware,
                layout=self.layout,
                chunk_bytes=self.astra_chunk_bytes,
                server_id=self.server_id,
                gpu_source_root_bandwidth_gbps=(
                    self.gpu_source_root_bandwidth_gbps),
            )
        else:
            projection = build_append_hbf_astra_projection(
                job=job,
                hardware=self.hardware,
                layout=self.layout,
                chunk_bytes=self.astra_chunk_bytes,
                server_id=self.server_id,
            )
        dispatch = HBFLifecycleExternalDispatch(
            arrival_ns=arrival_ns,
            job=job,
            projection=projection,
        )
        if dispatch.job_id in self._external_pending:
            raise RuntimeError(
                "duplicate HBF lifecycle external ASTRA job id "
                f"{dispatch.job_id!r}")
        self._external_pending[dispatch.job_id] = dispatch
        self._external_outbox.append(dispatch)

    def _physical_bytes(self, logical_bytes: int) -> int:
        return (
            logical_bytes * self.layout.physical_kv_replication_factor)

    def _per_card_bytes(self, logical_bytes: int) -> int:
        return int(math.ceil(
            self._physical_bytes(logical_bytes) / self.layout.tp_size))

    def _range_card_bytes(
            self, group_id: int, *,
            token_start: int, token_count: int) -> dict[int, int]:
        return hbf_kv_range_card_bytes(
            layout=self.layout,
            card_ids=self._group(group_id).card_ids,
            kv_bytes_per_token=self.kv_bytes_per_token,
            token_start=token_start,
            token_count=token_count,
        )

    def _prefix_card_bytes(
            self, group_id: int, token_count: int) -> dict[int, int]:
        return self._range_card_bytes(
            group_id, token_start=0, token_count=token_count)

    def _record_lpddr_card_bytes(
            self, record: SessionPlacement) -> dict[int, int]:
        if record.group_id is None:
            return {}
        return self._range_card_bytes(
            record.group_id,
            token_start=record.committed_hbf_tokens,
            token_count=record.lpddr_tokens,
        )

    def _job_card_bytes(
            self, job: MigrationJob | AppendJob) -> dict[int, int]:
        if job.card_bytes:
            return dict(job.card_bytes)
        return self._range_card_bytes(
            job.group_id,
            token_start=job.token_start,
            token_count=job.token_count,
        )

    def _record_hbf_kv_write(
            self, job: MigrationJob | AppendJob) -> None:
        """Charge every admitted KV media write, including stale jobs."""

        card_bytes = self._job_card_bytes(job)
        if sum(card_bytes.values()) != job.physical_bytes:
            raise AssertionError(
                "HBF per-card write vector changed physical byte count")
        if isinstance(job, MigrationJob):
            operation = self._hbf_migration_write_bytes_by_card
            if job.source_kind == MigrationSourceKind.SSD:
                for card_id, byte_count in card_bytes.items():
                    self._hbf_ssd_import_write_bytes_by_card[
                        card_id] += byte_count
        else:
            operation = self._hbf_append_write_bytes_by_card
        for card_id, byte_count in card_bytes.items():
            self._hbf_kv_write_bytes_by_card[card_id] += byte_count
            operation[card_id] += byte_count

    def _record_hbf_wasted_write(
            self, job: MigrationJob | AppendJob) -> None:
        card_bytes = self._job_card_bytes(job)
        if isinstance(job, MigrationJob):
            operation = (
                self._hbf_migration_wasted_write_bytes_by_card)
            if job.source_kind == MigrationSourceKind.SSD:
                for card_id, byte_count in card_bytes.items():
                    self._hbf_ssd_import_wasted_write_bytes_by_card[
                        card_id] += byte_count
        else:
            operation = self._hbf_append_wasted_write_bytes_by_card
        for card_id, byte_count in card_bytes.items():
            operation[card_id] += byte_count

    def _hbf_write_accounting_report(self) -> dict[str, Any]:
        card_ids = tuple(sorted(self._hbf_kv_write_bytes_by_card))
        values = tuple(
            self._hbf_kv_write_bytes_by_card[card_id]
            for card_id in card_ids
        )
        total = sum(values)
        mean = total / len(values) if values else 0.0
        variance = (
            sum((value - mean) ** 2 for value in values)
            / len(values)
            if values else 0.0
        )
        stddev = math.sqrt(variance)
        maximum = max(values, default=0)
        minimum = min(values, default=0)
        hottest_card_ids = [
            card_id
            for card_id in card_ids
            if maximum > 0
            and self._hbf_kv_write_bytes_by_card[card_id] == maximum
        ]
        wasted = (
            sum(self._hbf_migration_wasted_write_bytes_by_card.values())
            + sum(self._hbf_append_wasted_write_bytes_by_card.values())
        )
        return {
            "schema_version": 1,
            "accounting_basis": (
                "physical_media_payload_of_admitted_jobs"),
            "complete_for_endurance_projection": not self._jobs,
            "accounting_semantics": (
                "recurring KV payload bytes charged when migration or "
                "append jobs are admitted; stale jobs remain physical "
                "writes; SSD imports are a migration subset; payload "
                "bytes exclude any assumed flash write amplification"
            ),
            "total_physical_write_bytes": total,
            "migration_physical_write_bytes": sum(
                self._hbf_migration_write_bytes_by_card.values()),
            "append_physical_write_bytes": sum(
                self._hbf_append_write_bytes_by_card.values()),
            "ssd_import_physical_write_bytes_subset": sum(
                self._hbf_ssd_import_write_bytes_by_card.values()),
            "wasted_physical_write_bytes": wasted,
            "wasted_write_fraction": (
                wasted / total if total else None),
            "static_model_weight": {
                "bytes_per_card": self.weight_bytes_per_rank,
                "write_count": 1,
                "included_in_recurring_kv_wear": False,
            },
            "cards": [
                {
                    "server_id": self.server_id,
                    "card_id": card_id,
                    "device_id": (
                        f"hbf-server-{self.server_id}-card-{card_id}"),
                    "kv_region_capacity_bytes": (
                        self.usable_bytes_per_card),
                    "migration_write_bytes": (
                        self._hbf_migration_write_bytes_by_card[
                            card_id]),
                    "ssd_import_write_bytes_subset": (
                        self._hbf_ssd_import_write_bytes_by_card[
                            card_id]),
                    "append_write_bytes": (
                        self._hbf_append_write_bytes_by_card[card_id]),
                    "total_write_bytes": (
                        self._hbf_kv_write_bytes_by_card[card_id]),
                    "migration_wasted_write_bytes": (
                        self._hbf_migration_wasted_write_bytes_by_card[
                            card_id]),
                    "ssd_import_wasted_write_bytes_subset": (
                        self._hbf_ssd_import_wasted_write_bytes_by_card[
                            card_id]),
                    "append_wasted_write_bytes": (
                        self._hbf_append_wasted_write_bytes_by_card[
                            card_id]),
                    "wasted_write_bytes": (
                        self._hbf_migration_wasted_write_bytes_by_card[
                            card_id]
                        + self._hbf_append_wasted_write_bytes_by_card[
                            card_id]),
                }
                for card_id in card_ids
            ],
            "hotness": {
                "scope": (
                    "exact across cards; within each card, KV writes are "
                    "assumed randomly and uniformly spread over the "
                    "writable KV region; cell/page/block hotness is not "
                    "modeled"
                ),
                "card_count": len(values),
                "minimum_write_bytes": minimum,
                "mean_write_bytes": mean,
                "maximum_write_bytes": maximum,
                "population_stddev_write_bytes": stddev,
                "coefficient_of_variation": (
                    stddev / mean if mean else None),
                "maximum_to_mean": (
                    maximum / mean if mean else None),
                "hottest_card_share": (
                    maximum / total if total else None),
                "hottest_card_ids": hottest_card_ids,
            },
        }

    def _assert_hbf_write_accounting(self) -> None:
        expected_cards = set(self._hbf_kv_write_bytes_by_card)
        ledgers = (
            self._hbf_kv_write_bytes_by_card,
            self._hbf_migration_write_bytes_by_card,
            self._hbf_append_write_bytes_by_card,
            self._hbf_ssd_import_write_bytes_by_card,
            self._hbf_migration_wasted_write_bytes_by_card,
            self._hbf_append_wasted_write_bytes_by_card,
            self._hbf_ssd_import_wasted_write_bytes_by_card,
        )
        if any(set(ledger) != expected_cards for ledger in ledgers):
            raise AssertionError(
                "HBF write ledgers cover different physical cards")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for ledger in ledgers
            for value in ledger.values()
        ):
            raise AssertionError(
                "HBF write ledger contains invalid bytes")
        for card_id in expected_cards:
            migration = self._hbf_migration_write_bytes_by_card[
                card_id]
            append = self._hbf_append_write_bytes_by_card[card_id]
            ssd_import = self._hbf_ssd_import_write_bytes_by_card[
                card_id]
            migration_wasted = (
                self._hbf_migration_wasted_write_bytes_by_card[
                    card_id])
            append_wasted = (
                self._hbf_append_wasted_write_bytes_by_card[
                    card_id])
            ssd_import_wasted = (
                self._hbf_ssd_import_wasted_write_bytes_by_card[
                    card_id])
            if (
                self._hbf_kv_write_bytes_by_card[card_id]
                != migration + append
                or ssd_import > migration
                or migration_wasted > migration
                or append_wasted > append
                or ssd_import_wasted > ssd_import
                or ssd_import_wasted > migration_wasted
            ):
                raise AssertionError(
                    "HBF per-card write accounting is inconsistent")
        expected_totals = (
            (
                self._hbf_migration_write_bytes_by_card,
                self.metrics.migration_physical_bytes,
            ),
            (
                self._hbf_append_write_bytes_by_card,
                self.metrics.append_physical_bytes,
            ),
            (
                self._hbf_ssd_import_write_bytes_by_card,
                self.metrics.ssd_import_physical_bytes,
            ),
            (
                self._hbf_migration_wasted_write_bytes_by_card,
                self.metrics.migration_wasted_physical_bytes,
            ),
            (
                self._hbf_append_wasted_write_bytes_by_card,
                self.metrics.append_wasted_physical_bytes,
            ),
            (
                self._hbf_ssd_import_wasted_write_bytes_by_card,
                self.metrics.ssd_import_wasted_physical_bytes,
            ),
        )
        if any(
            sum(ledger.values()) != expected
            for ledger, expected in expected_totals
        ):
            raise AssertionError(
                "HBF per-card and aggregate write accounting diverged")
        if sum(self._hbf_kv_write_bytes_by_card.values()) != (
            self.metrics.migration_physical_bytes
            + self.metrics.append_physical_bytes
        ):
            raise AssertionError(
                "HBF total write accounting double-counted a component")
        pending_appends = sum(
            isinstance(job, AppendJob)
            for job in self._jobs.values()
        )
        if self.metrics.append_jobs_started != sum((
            self.metrics.append_jobs_committed,
            self.metrics.append_jobs_stale,
            pending_appends,
        )):
            raise AssertionError(
                "append completion accounting mismatch")

    @staticmethod
    def _peak_card_bytes(card_bytes: Mapping[int, int]) -> int:
        return max(card_bytes.values(), default=0)

    @staticmethod
    def lpddr_owner(session_id: str) -> str:
        return f"hbf-session:{session_id}"

    def _lpddr_bytes_for_tokens(self, token_count: int) -> int:
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        if self.layout.is_context_striped:
            return self._peak_card_bytes(
                self._prefix_card_bytes(0, token_count))
        return self._per_card_bytes(
            token_count * self.kv_bytes_per_token)

    def _group(self, group_id: int) -> HBFReplicaGroup:
        if not 0 <= group_id < len(self.groups):
            raise ValueError(f"invalid HBF group_id={group_id}")
        return self.groups[group_id]

    def _total_gpu_retained(self) -> int:
        return sum(
            record.gpu_retained_bytes for record in self.sessions.values())

    def _total_hbf_reserved(self) -> int:
        return sum(
            sum(card_bytes.values())
            for card_bytes in self._reserved_bytes_by_card.values()
        )

    def _update_peaks(self) -> None:
        self.metrics.gpu_retained_bytes_peak = max(
            self.metrics.gpu_retained_bytes_peak,
            self._total_gpu_retained(),
        )
        self.metrics.hbf_reserved_bytes_peak = max(
            self.metrics.hbf_reserved_bytes_peak,
            self._total_hbf_reserved(),
        )

    def _normalize_group_card_bytes(
            self, group_id: int,
            card_bytes: Mapping[int, int] | int) -> dict[int, int]:
        cards = self._group(group_id).card_ids
        if isinstance(card_bytes, int) and not isinstance(card_bytes, bool):
            if card_bytes < 0:
                raise ValueError(
                    "per-card reservation must be non-negative")
            return {card_id: card_bytes for card_id in cards}
        if not isinstance(card_bytes, Mapping):
            raise ValueError("card reservation must be a mapping")
        if set(card_bytes) != set(cards):
            raise ValueError(
                "card reservation must cover the selected replica")
        result = {}
        for card_id in cards:
            value = card_bytes[card_id]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    "card reservation bytes must be non-negative integers")
            result[card_id] = value
        return result

    def _sync_group_peak(self, group_id: int) -> None:
        self._reserved_per_card_by_group[group_id] = max(
            self._reserved_bytes_by_card[group_id].values(),
            default=0,
        )

    def _reserve_group(
            self, group_id: int,
            card_bytes: Mapping[int, int] | int) -> None:
        requested = self._normalize_group_card_bytes(
            group_id, card_bytes)
        used = self._reserved_bytes_by_card[group_id]
        overflow = {
            card_id: used[card_id] + requested[card_id]
            for card_id in used
            if (
                used[card_id] + requested[card_id]
                > self.usable_bytes_per_card
            )
        }
        if overflow:
            raise RuntimeError(
                f"HBF group {group_id} capacity exceeded: "
                f"requested={requested}, used={used}, "
                f"capacity={self.usable_bytes_per_card}")
        for card_id, byte_count in requested.items():
            used[card_id] += byte_count
        self._sync_group_peak(group_id)
        self._update_peaks()

    def _release_group(
            self, group_id: int,
            card_bytes: Mapping[int, int] | int) -> None:
        released = self._normalize_group_card_bytes(
            group_id, card_bytes)
        used = self._reserved_bytes_by_card[group_id]
        if any(
                released[card_id] > used[card_id]
                for card_id in used):
            raise RuntimeError(
                f"HBF group {group_id} double release: "
                f"release={released}, reserved={used}")
        for card_id, byte_count in released.items():
            used[card_id] -= byte_count
        self._sync_group_peak(group_id)

    def _record_committed_card_bytes(
            self, record: SessionPlacement) -> dict[int, int]:
        if record.group_id is None:
            return {}
        return self._prefix_card_bytes(
            record.group_id, record.committed_hbf_tokens)

    def _evict_one(self, group_id: int, now_ns: int) -> bool:
        candidates = [
            record
            for record in self.sessions.values()
            if (
                record.group_id == group_id
                and record.state == PlacementState.HBF_READY
                and not record.append_job_ids
                and record.lpddr_tokens == 0
                and record.committed_hbf_tokens > 0
            )
        ]
        if not candidates:
            return False
        victim = min(
            candidates, key=lambda item: (item.last_access_ns, item.session_id))
        self._release_group(
            group_id, self._record_committed_card_bytes(victim))
        victim.state = PlacementState.EVICTED
        victim.group_id = None
        victim.committed_hbf_tokens = 0
        victim.lpddr_tokens = 0
        self.lpddr_ledger.release(
            self.lpddr_owner(victim.session_id))
        victim.committed_per_card_bytes = 0
        victim.last_access_ns = now_ns
        victim.generation += 1
        self.metrics.capacity_evictions += 1
        return True

    def _ensure_capacity(
            self, group_id: int,
            card_bytes: Mapping[int, int],
            now_ns: int) -> bool:
        requested = self._normalize_group_card_bytes(
            group_id, card_bytes)
        if any(
                value > self.usable_bytes_per_card
                for value in requested.values()):
            return False
        while any(
                self._reserved_bytes_by_card[group_id][card_id]
                + requested[card_id]
                > self.usable_bytes_per_card
                for card_id in requested):
            if not self._evict_one(group_id, now_ns):
                return False
        return True

    def _choose_group(
            self, card_bytes_by_group: Mapping[
                int, Mapping[int, int]],
            now_ns: int) -> Optional[int]:
        if set(card_bytes_by_group) != {
                group.group_id for group in self.groups}:
            raise ValueError(
                "candidate card vectors must cover every replica group")

        def reclaimable_bytes(group_id: int) -> dict[int, int]:
            result = {
                card_id: 0 for card_id in self._group(group_id).card_ids
            }
            for record in self.sessions.values():
                if (
                    record.group_id == group_id
                    and record.state == PlacementState.HBF_READY
                    and not record.append_job_ids
                    and record.lpddr_tokens == 0
                    and record.committed_hbf_tokens > 0
                ):
                    for card_id, byte_count in (
                            self._record_committed_card_bytes(
                                record).items()):
                        result[card_id] += byte_count
            return result

        feasible = []
        for group in self.groups:
            requested = self._normalize_group_card_bytes(
                group.group_id,
                card_bytes_by_group[group.group_id],
            )
            reclaimable = reclaimable_bytes(group.group_id)
            if all(
                    self._reserved_bytes_by_card[
                        group.group_id][card_id]
                    + requested[card_id]
                    <= self.usable_bytes_per_card
                    + reclaimable[card_id]
                    for card_id in requested):
                feasible.append(group)
        if not feasible:
            return None
        group = min(
            feasible,
            key=lambda item: (
                max(
                    (
                        self._reserved_bytes_by_card[
                            item.group_id][card_id]
                        + card_bytes_by_group[
                            item.group_id][card_id]
                        for card_id in item.card_ids
                    ),
                    default=0,
                ),
                item.group_id,
            ),
        )
        if not self._ensure_capacity(
                group.group_id,
                card_bytes_by_group[group.group_id],
                now_ns):
            raise AssertionError(
                "preflight-feasible HBF group could not reclaim capacity")
        return group.group_id

    def _choose_group_without_eviction(
            self, card_bytes_by_group: Mapping[
                int, Mapping[int, int]]) -> Optional[int]:
        """Choose a replica only when its current free capacity fits.

        SSD checkpoints are durable source copies.  Importing one is an
        opportunistic promotion, so it must never destroy an already useful
        HBF placement merely to make the promotion fit.
        """

        if set(card_bytes_by_group) != {
                group.group_id for group in self.groups}:
            raise ValueError(
                "candidate card vectors must cover every replica group")
        feasible: list[HBFReplicaGroup] = []
        for group in self.groups:
            requested = self._normalize_group_card_bytes(
                group.group_id,
                card_bytes_by_group[group.group_id],
            )
            if all(
                    self._reserved_bytes_by_card[
                        group.group_id][card_id]
                    + requested[card_id]
                    <= self.usable_bytes_per_card
                    for card_id in group.card_ids):
                feasible.append(group)
        if not feasible:
            return None
        return min(
            feasible,
            key=lambda item: (
                max(
                    (
                        self._reserved_bytes_by_card[
                            item.group_id][card_id]
                        + card_bytes_by_group[
                            item.group_id][card_id]
                        for card_id in item.card_ids
                    ),
                    default=0,
                ),
                item.group_id,
            ),
        ).group_id

    @staticmethod
    def _service_ns(byte_count: int, bandwidth_gbps: float) -> int:
        if byte_count <= 0:
            return 0
        return int(math.ceil(byte_count / bandwidth_gbps))

    def _analytical_local_resource(self, resource: str) -> str:
        """Namespace one HBF-server-local analytical resource."""

        return f"{self.analytical_resource_prefix}{resource}"

    def _migration_demands(
            self, group_id: int, *, logical_bytes: int,
            physical_bytes: int, card_bytes: Mapping[int, int],
            job_id: int) -> Mapping[str, tuple[int, int]]:
        demands: dict[str, tuple[int, int]] = {
            "gpu-source-pcie-root": (
                self._service_ns(
                    logical_bytes,
                    self.gpu_source_root_bandwidth_gbps,
                ),
                logical_bytes,
            ),
            "rdma-network": (
                int(math.ceil(
                    self.hardware.rdma_one_way_latency_us * 1e3
                )) + self._service_ns(
                    logical_bytes, self.hardware.rdma_bandwidth_gbps),
                logical_bytes,
            ),
        }
        root_bytes: dict[int, int] = {}
        normalized = self._normalize_group_card_bytes(
            group_id, card_bytes)
        for card_id, byte_count in normalized.items():
            root_id = self.pcie_topology.root_for_card(card_id)
            root_bytes[root_id] = (
                root_bytes.get(root_id, 0) + byte_count)
        for root_id, byte_count in root_bytes.items():
            root_bandwidth, root_fixed_us = (
                self.pcie_topology.migration_root_service(root_id))
            demands[self._analytical_local_resource(
                f"hbf-pcie-root-{root_id}"
            )] = (
                int(math.ceil(root_fixed_us * 1e3))
                + self._service_ns(byte_count, root_bandwidth),
                byte_count,
            )
        for card_id, byte_count in normalized.items():
            if byte_count == 0:
                continue
            demands[self._analytical_local_resource(
                f"hbf-card-{card_id}-pcie"
            )] = (
                self._service_ns(
                    byte_count,
                    self.hardware.intra_fabric_bandwidth_gbps_per_card,
                ),
                byte_count,
            )
            demands[self._analytical_local_resource(
                f"hbf-card-{card_id}-media"
            )] = (
                int(math.ceil(
                    self.hardware.hbf_write_latency_us * 1e3
                )) + self._service_ns(
                    byte_count,
                    self.hardware.hbf_write_bandwidth_gbps_per_card,
                ),
                byte_count,
            )
        return demands

    def _ssd_import_demands(
            self, group_id: int, *, logical_bytes: int,
            physical_bytes: int, card_bytes: Mapping[int, int],
            job_id: int) -> Mapping[str, tuple[int, int]]:
        """Return SSD-origin import demand after SSD-to-CPU has completed."""

        demands = dict(self._migration_demands(
            group_id,
            logical_bytes=logical_bytes,
            physical_bytes=physical_bytes,
            card_bytes=card_bytes,
            job_id=job_id,
        ))
        # The external SSD stage has already populated CPU DRAM.  Replace
        # the direct GPU-HBM source root with the GPU host's CPU/NIC path.
        demands.pop("gpu-source-pcie-root")
        demands.update({
            f"gpu-node-{self.gpu_source_node_id}-cpu-dram": (
                self._service_ns(
                    logical_bytes,
                    self.gpu_source_cpu_bandwidth_gbps,
                ),
                logical_bytes,
            ),
            f"gpu-node-{self.gpu_source_node_id}-rdma-nic": (
                self._service_ns(
                    logical_bytes,
                    self.gpu_source_nic_bandwidth_gbps,
                ),
                logical_bytes,
            ),
        })
        return demands

    def _append_demands(
            self, group_id: int, card_bytes: Mapping[int, int],
    ) -> Mapping[str, tuple[int, int]]:
        demands: dict[str, tuple[int, int]] = {}
        normalized = self._normalize_group_card_bytes(
            group_id, card_bytes)
        for card_id, byte_count in normalized.items():
            if byte_count == 0:
                continue
            demands[self._analytical_local_resource(
                f"hbf-card-{card_id}-lpddr"
            )] = (
                self._service_ns(
                    byte_count,
                    self.hardware.lpddr_bandwidth_gbps_per_card,
                ),
                byte_count,
            )
            demands[self._analytical_local_resource(
                f"hbf-card-{card_id}-media"
            )] = (
                int(math.ceil(
                    self.hardware.hbf_write_latency_us * 1e3
                )) + self._service_ns(
                    byte_count,
                    self.hardware.hbf_write_bandwidth_gbps_per_card,
                ),
                byte_count,
            )
        return demands

    def register_session(
            self, session_id: str, *, now_ns: int = 0) -> SessionPlacement:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if now_ns < 0:
            raise ValueError("now_ns must be non-negative")
        if session_id in self.sessions:
            raise ValueError(f"duplicate session_id={session_id!r}")
        record = SessionPlacement(
            session_id=session_id,
            last_access_ns=now_ns,
        )
        self.sessions[session_id] = record
        return record

    def complete_gpu_turn(
            self, session_id: str, *, now_ns: int,
            total_tokens: int, has_successor: bool,
            start_migration: bool = True) -> Optional[MigrationJob]:
        if not isinstance(start_migration, bool):
            raise ValueError("start_migration must be a boolean")
        self.advance(now_ns)
        record = self.sessions[session_id]
        if record.state not in {
            PlacementState.GPU_ACTIVE,
            PlacementState.EVICTED,
        }:
            raise RuntimeError(
                f"GPU completion from invalid state {record.state}")
        if total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if total_tokens < record.total_tokens:
            raise ValueError("GPU completion cannot shrink session context")
        record.version += 1
        record.total_tokens = total_tokens
        record.gpu_retained_bytes = (
            total_tokens * self.kv_bytes_per_token)
        record.last_access_ns = now_ns
        record.active_request_id = None
        record.migration_source_kind = None
        self._update_peaks()
        if not has_successor:
            self.end_session(session_id, now_ns=now_ns)
            return None
        if not start_migration:
            record.state = PlacementState.GPU_READY
            record.group_id = None
            if self.validate_every_event:
                self.assert_invariants()
            return None
        return self._start_migration(record, now_ns)

    def publish_ssd_checkpoint(
            self, session_id: str, *, now_ns: int,
            snapshot_version: int) -> bool:
        """Atomically publish an externally committed SSD checkpoint.

        The SSD writer owns its transfer and capacity accounting.  This
        callback only changes the authoritative snapshot source after the
        complete object is durable.  A late callback is harmless: if the
        session resumed or produced a newer version, no lifecycle state is
        changed and ``False`` is returned.
        """

        if (
            isinstance(snapshot_version, bool)
            or not isinstance(snapshot_version, int)
            or snapshot_version < 0
        ):
            raise ValueError(
                "snapshot_version must be a non-negative integer")
        self.advance(now_ns)
        record = self.sessions[session_id]
        if (
            record.state != PlacementState.GPU_READY
            or record.version != snapshot_version
        ):
            return False
        if (
            record.active_request_id is not None
            or record.group_id is not None
            or record.migration_job_ids
            or record.append_job_ids
        ):
            raise RuntimeError(
                "GPU_READY SSD publication has unexpected live ownership")
        expected_bytes = record.total_tokens * self.kv_bytes_per_token
        if record.gpu_retained_bytes != expected_bytes:
            raise RuntimeError(
                "SSD publication lacks the complete retained GPU snapshot")
        record.state = PlacementState.SSD_READY
        record.gpu_retained_bytes = 0
        record.last_access_ns = now_ns
        self.metrics.ssd_checkpoints_published += 1
        if self.validate_every_event:
            self.assert_invariants()
        return True

    def start_migration(
            self, session_id: str, *,
            now_ns: int) -> Optional[MigrationJob]:
        """Start a previously deferred GPU-to-HBF migration.

        The caller owns eligibility timing and exact-timestamp ordering.  A
        resume that wins that ordering moves the placement out of GPU_READY,
        causing this method to reject the stale trigger without reserving any
        migration resource.
        """

        self.advance(now_ns)
        record = self.sessions[session_id]
        if record.state != PlacementState.GPU_READY:
            raise RuntimeError(
                "deferred migration requires a GPU_READY session: "
                f"session={session_id!r}, state={record.state}")
        if record.active_request_id is not None:
            raise RuntimeError(
                "deferred migration cannot start for an active request")
        if record.group_id is not None:
            raise RuntimeError(
                "GPU_READY session unexpectedly owns an HBF group")
        if record.total_tokens <= 0 or record.gpu_retained_bytes <= 0:
            raise RuntimeError(
                "deferred migration lacks retained GPU KV")
        return self._start_migration(record, now_ns)

    def start_import_from_ssd(
            self, session_id: str, *,
            now_ns: int) -> Optional[MigrationJob]:
        """Promote one CPU-staged SSD snapshot without evicting HBF data.

        ``now_ns`` is the completion time of the caller-owned SSD-to-CPU
        stage.  This method accounts for the GPU host CPU/NIC source path,
        the shared RDMA link, and all HBF destination resources.  The
        analytical calendar is required because the legacy ASTRA migration
        projection models a GPU-HBM source.
        """

        if self.execution_backend != "analytical_calendar":
            raise RuntimeError(
                "SSD-origin HBF import requires "
                "execution_backend='analytical_calendar'")
        self.advance(now_ns)
        record = self.sessions[session_id]
        if record.state != PlacementState.SSD_READY:
            raise RuntimeError(
                "SSD import requires an SSD_READY session: "
                f"session={session_id!r}, state={record.state}")
        if record.active_request_id is not None:
            raise RuntimeError(
                "SSD import cannot start for an active request")
        if record.group_id is not None:
            raise RuntimeError(
                "SSD_READY session unexpectedly owns an HBF group")
        if record.total_tokens <= 0 or record.gpu_retained_bytes != 0:
            raise RuntimeError(
                "SSD import requires an SSD-only snapshot")
        return self._start_migration(
            record,
            now_ns,
            source_kind=MigrationSourceKind.SSD,
            allow_eviction=False,
        )

    def _start_migration(
            self, record: SessionPlacement,
            now_ns: int, *,
            source_kind: MigrationSourceKind = MigrationSourceKind.GPU,
            allow_eviction: bool = True) -> Optional[MigrationJob]:
        if not isinstance(source_kind, MigrationSourceKind):
            raise ValueError(
                "source_kind must be a MigrationSourceKind")
        if not isinstance(allow_eviction, bool):
            raise ValueError("allow_eviction must be a boolean")
        if source_kind == MigrationSourceKind.GPU:
            if (
                record.state not in {
                    PlacementState.GPU_ACTIVE,
                    PlacementState.GPU_READY,
                }
                or record.gpu_retained_bytes <= 0
            ):
                raise RuntimeError(
                    "GPU migration requires retained GPU KV")
        elif (
            record.state != PlacementState.SSD_READY
            or record.gpu_retained_bytes != 0
        ):
            raise RuntimeError(
                "SSD import requires an SSD-only snapshot")
        logical_bytes = record.total_tokens * self.kv_bytes_per_token
        card_bytes_by_group = {
            group.group_id: self._range_card_bytes(
                group.group_id,
                token_start=0,
                token_count=record.total_tokens,
            )
            for group in self.groups
        }
        group_id = (
            self._choose_group(card_bytes_by_group, now_ns)
            if allow_eviction
            else self._choose_group_without_eviction(
                card_bytes_by_group)
        )
        if group_id is None:
            # The source copy remains authoritative and retryable.
            record.state = (
                PlacementState.GPU_READY
                if source_kind == MigrationSourceKind.GPU
                else PlacementState.SSD_READY
            )
            record.group_id = None
            return None
        card_bytes = card_bytes_by_group[group_id]
        per_card_bytes = self._peak_card_bytes(card_bytes)
        self._reserve_group(group_id, card_bytes)
        record.pending_reserved_per_card_bytes += per_card_bytes
        record.generation += 1
        record.state = PlacementState.MIGRATING
        record.group_id = group_id
        record.migration_source_kind = source_kind
        job_id = self._next_id()
        physical_bytes = self._physical_bytes(logical_bytes)
        if self.execution_backend == "analytical_calendar":
            demands = (
                self._migration_demands(
                    group_id,
                    logical_bytes=logical_bytes,
                    physical_bytes=physical_bytes,
                    card_bytes=card_bytes,
                    job_id=job_id,
                )
                if source_kind == MigrationSourceKind.GPU
                else self._ssd_import_demands(
                    group_id,
                    logical_bytes=logical_bytes,
                    physical_bytes=physical_bytes,
                    card_bytes=card_bytes,
                    job_id=job_id,
                )
            )
            start_ns, completion_ns = self.calendar.reserve_parallel(
                arrival_ns=now_ns,
                job_id=job_id,
                kind=(
                    "migration"
                    if source_kind == MigrationSourceKind.GPU
                    else "ssd-import"
                ),
                namespace="hbf-lifecycle",
                demands=demands,
            )
        else:
            # These fields preserve the immutable legacy job schema.  They
            # are deliberately non-authoritative in external mode: ASTRA's
            # callback is the sole completion-time source.
            start_ns = now_ns
            completion_ns = now_ns
        job = MigrationJob(
            job_id=job_id,
            session_id=record.session_id,
            generation=record.generation,
            version=record.version,
            group_id=group_id,
            token_count=record.total_tokens,
            logical_bytes=logical_bytes,
            physical_bytes=physical_bytes,
            per_card_bytes=per_card_bytes,
            start_ns=start_ns,
            completion_ns=completion_ns,
            token_start=0,
            card_bytes=canonical_card_bytes(
                self._group(group_id).card_ids, card_bytes),
            source_kind=source_kind,
        )
        self._jobs[job_id] = job
        if self.execution_backend == "analytical_calendar":
            heapq.heappush(
                self._completion_heap, (completion_ns, job_id))
        else:
            self._enqueue_external_job(job, now_ns)
        record.migration_job_ids.add(job_id)
        self.metrics.migrations_started += 1
        self.metrics.migration_logical_bytes += logical_bytes
        self.metrics.migration_physical_bytes += physical_bytes
        self._record_hbf_kv_write(job)
        if source_kind == MigrationSourceKind.SSD:
            self.metrics.ssd_imports_started += 1
            self.metrics.ssd_import_logical_bytes += logical_bytes
            self.metrics.ssd_import_physical_bytes += physical_bytes
        return job

    @staticmethod
    def _gpu_ready_pressure_reclaimable(
            record: SessionPlacement) -> bool:
        """Return whether an idle GPU-only lineage has no future job owner."""

        return bool(
            record.state == PlacementState.GPU_READY
            and record.active_request_id is None
            and record.group_id is None
            and record.gpu_retained_bytes > 0
            and record.committed_hbf_tokens == 0
            and record.lpddr_tokens == 0
            and record.committed_per_card_bytes == 0
            and record.pending_reserved_per_card_bytes == 0
            and not record.migration_job_ids
            and not record.append_job_ids
        )

    def gpu_ready_pressure_reclaimable(self, session_id: str) -> bool:
        """Return whether one GPU_READY session can be dropped immediately."""

        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be non-empty")
        if session_id not in self.sessions:
            raise KeyError(f"unknown session_id={session_id!r}")
        return self._gpu_ready_pressure_reclaimable(
            self.sessions[session_id])

    def evict_oldest_gpu_ready_for_hbm_pressure(
            self, session_ids: Sequence[str], *,
            now_ns: int) -> Optional[GPUReadyPressureEviction]:
        """Drop one deterministic idle GPU copy for later recomputation.

        ``session_ids`` is supplied by the adapter after filtering to one
        physical GPU owner. Only GPU_READY records with no active request and
        no pending migration/append callback are eligible.
        """

        self.advance(now_ns)
        unique_ids = set()
        for session_id in session_ids:
            if not isinstance(session_id, str) or not session_id:
                raise ValueError(
                    "GPU-ready pressure candidates must be session IDs")
            if session_id not in self.sessions:
                raise KeyError(
                    f"unknown GPU-ready pressure candidate {session_id!r}")
            unique_ids.add(session_id)
        candidates = [
            self.sessions[session_id]
            for session_id in unique_ids
            if self._gpu_ready_pressure_reclaimable(
                self.sessions[session_id])
        ]
        if not candidates:
            return None
        victim = min(
            candidates,
            key=lambda record: (
                record.last_access_ns,
                record.session_id,
            ),
        )
        prior_access = victim.last_access_ns
        prior_generation = victim.generation
        token_count = victim.total_tokens
        logical_bytes = victim.gpu_retained_bytes
        if logical_bytes != token_count * self.kv_bytes_per_token:
            raise RuntimeError(
                "GPU-ready pressure victim has inconsistent retained bytes: "
                f"session={victim.session_id!r}, "
                f"tokens={token_count}, bytes={logical_bytes}")

        victim.generation += 1
        victim.state = PlacementState.EVICTED
        victim.gpu_retained_bytes = 0
        victim.last_access_ns = now_ns
        self.metrics.gpu_ready_hbm_pressure_evictions += 1
        self.metrics.gpu_ready_hbm_pressure_evicted_bytes += logical_bytes
        result = GPUReadyPressureEviction(
            session_id=victim.session_id,
            eviction_ns=now_ns,
            last_access_ns=prior_access,
            token_count=token_count,
            logical_bytes=logical_bytes,
            generation_before=prior_generation,
            generation_after=victim.generation,
        )
        if self.validate_every_event:
            self.assert_invariants()
        return result

    @staticmethod
    def _reuse_tokens(
            record: SessionPlacement, *,
            prefix_reuse_tokens: Optional[int],
            input_tokens: Optional[int]) -> int:
        if prefix_reuse_tokens is None:
            reuse = record.total_tokens
        else:
            if (
                isinstance(prefix_reuse_tokens, bool)
                or not isinstance(prefix_reuse_tokens, int)
                or prefix_reuse_tokens < 0
            ):
                raise ValueError(
                    "prefix_reuse_tokens must be a non-negative integer")
            reuse = prefix_reuse_tokens
        if input_tokens is not None:
            if (
                isinstance(input_tokens, bool)
                or not isinstance(input_tokens, int)
                or input_tokens <= 0
            ):
                raise ValueError("input_tokens must be a positive integer")
            if reuse > input_tokens:
                raise ValueError(
                    "prefix reuse cannot exceed request input")
        if reuse > record.total_tokens:
            raise ValueError(
                "prefix reuse exceeds materialized session lineage")
        return reuse

    def _trim_hbf_lineage(
            self, record: SessionPlacement, reuse_tokens: int) -> bool:
        available = (
            record.committed_hbf_tokens + record.lpddr_tokens)
        if reuse_tokens > available:
            raise ValueError(
                "prefix reuse exceeds HBF/LPDDR-resident lineage")
        if reuse_tokens == available:
            return False
        if record.group_id is None:
            raise RuntimeError("cannot trim HBF lineage without a group")
        record.generation += 1
        record.version += 1
        old_committed = self._record_committed_card_bytes(record)
        new_hbf = min(reuse_tokens, record.committed_hbf_tokens)
        new_lpddr = reuse_tokens - new_hbf
        new_committed = self._prefix_card_bytes(
            record.group_id, new_hbf)
        released = {
            card_id: old_committed[card_id] - new_committed[card_id]
            for card_id in old_committed
        }
        if any(value < 0 for value in released.values()):
            raise RuntimeError("HBF lineage trim grew committed storage")
        if any(released.values()):
            self._release_group(record.group_id, released)
        record.total_tokens = reuse_tokens
        record.committed_hbf_tokens = new_hbf
        record.lpddr_tokens = new_lpddr
        record.committed_per_card_bytes = self._peak_card_bytes(
            new_committed)
        self.lpddr_ledger.set_card_bytes(
            record.group_id,
            self.lpddr_owner(record.session_id),
            self._range_card_bytes(
                record.group_id,
                token_start=new_hbf,
                token_count=new_lpddr,
            ),
        )
        return True

    def _trim_gpu_lineage(
            self, record: SessionPlacement, reuse_tokens: int) -> bool:
        if reuse_tokens == record.total_tokens:
            return False
        record.generation += 1
        record.version += 1
        record.total_tokens = reuse_tokens
        if record.gpu_retained_bytes:
            record.gpu_retained_bytes = (
                reuse_tokens * self.kv_bytes_per_token)
        return True

    def _route_hbf_lpddr_fallback(
            self, record: SessionPlacement, *,
            request_id: Optional[int], now_ns: int) -> ResumeRoute:
        """Atomically abandon an HBF copy before GPU recomputation."""

        if record.group_id is None:
            raise RuntimeError(
                "cannot abandon HBF placement without a group")
        group_id = record.group_id
        record.generation += 1
        record.version += 1
        if record.committed_per_card_bytes:
            self._release_group(
                group_id, self._record_committed_card_bytes(record))
        self.lpddr_ledger.release(
            self.lpddr_owner(record.session_id))
        if request_id is not None:
            self.lpddr_ledger.release(
                hbf_request_headroom_owner(request_id))
        record.state = PlacementState.GPU_ACTIVE
        record.group_id = None
        record.gpu_retained_bytes = 0
        record.committed_hbf_tokens = 0
        record.lpddr_tokens = 0
        record.committed_per_card_bytes = 0
        record.last_access_ns = now_ns
        record.active_request_id = request_id
        self.metrics.gpu_recompute_resumes += 1
        self.metrics.lpddr_capacity_fallback_resumes += 1
        return ResumeRoute(
            execution=ResumeExecution.GPU_RECOMPUTE,
            session_id=record.session_id,
            group_id=None,
            hbf_tokens=0,
            lpddr_tokens=0,
            migration_inflight=False,
            reason="hbf_lpddr_finish_capacity_fallback",
        )

    def placement_snapshot(
            self, session_id: str) -> tuple[int, int, int]:
        record = self.sessions[session_id]
        if (
            record.state not in {
                PlacementState.HBF_READY,
                PlacementState.HBF_ACTIVE,
            }
            or record.group_id is None
        ):
            raise RuntimeError(
                f"session {session_id!r} has no active HBF placement")
        return (
            record.committed_hbf_tokens,
            record.lpddr_tokens,
            record.group_id,
        )

    def route_resume(
            self, session_id: str, *, now_ns: int,
            request_id: Optional[int] = None,
            prefix_reuse_tokens: Optional[int] = None,
            input_tokens: Optional[int] = None,
            lpddr_growth_tokens: Optional[int] = None) -> ResumeRoute:
        # Completion callbacks win exact timestamp ties.
        self.advance(now_ns)
        if lpddr_growth_tokens is not None:
            if (
                isinstance(lpddr_growth_tokens, bool)
                or not isinstance(lpddr_growth_tokens, int)
                or lpddr_growth_tokens < 0
            ):
                raise ValueError(
                    "lpddr_growth_tokens must be a non-negative integer")
            if request_id is None:
                raise ValueError(
                    "LPDDR finish reservation requires request_id")
        record = self.sessions[session_id]
        reuse_tokens = self._reuse_tokens(
            record,
            prefix_reuse_tokens=prefix_reuse_tokens,
            input_tokens=input_tokens,
        )
        record.last_access_ns = now_ns
        if record.state == PlacementState.HBF_READY:
            trimmed = self._trim_hbf_lineage(record, reuse_tokens)
            if lpddr_growth_tokens is not None:
                if record.group_id is None:
                    raise RuntimeError(
                        "HBF-ready record lost its replica group")
                headroom_owner = hbf_request_headroom_owner(request_id)
                headroom_card_bytes = self._range_card_bytes(
                    record.group_id,
                    token_start=reuse_tokens,
                    token_count=lpddr_growth_tokens,
                )
                if not self.lpddr_ledger.can_set_card_bytes(
                        record.group_id,
                        headroom_owner,
                        headroom_card_bytes):
                    return self._route_hbf_lpddr_fallback(
                        record,
                        request_id=request_id,
                        now_ns=now_ns,
                    )
                self.lpddr_ledger.set_card_bytes(
                    record.group_id,
                    headroom_owner,
                    headroom_card_bytes,
                )
            record.state = PlacementState.HBF_ACTIVE
            record.active_request_id = request_id
            self.metrics.hbf_resumes += 1
            return ResumeRoute(
                execution=ResumeExecution.HBF,
                session_id=session_id,
                group_id=record.group_id,
                hbf_tokens=record.committed_hbf_tokens,
                lpddr_tokens=record.lpddr_tokens,
                migration_inflight=False,
                reason=(
                    "hbf_context_trimmed"
                    if trimmed
                    else (
                        "hbf_ready"
                        if not record.append_job_ids
                        else "hbf_append_inflight"
                    )
                ),
            )
        if record.state == PlacementState.MIGRATING:
            # Invalidate publication but do not cancel already consumed
            # network/media service. Its capacity reservation is released by
            # the stale completion callback.
            source_kind = record.migration_source_kind
            if source_kind is None:
                raise RuntimeError(
                    "migrating session lacks an authoritative source kind")
            record.generation += 1
            if reuse_tokens < record.total_tokens:
                record.version += 1
                record.total_tokens = reuse_tokens
            if source_kind == MigrationSourceKind.SSD:
                record.gpu_retained_bytes = (
                    reuse_tokens * self.kv_bytes_per_token)
            record.state = PlacementState.GPU_ACTIVE
            record.group_id = None
            record.active_request_id = request_id
            record.migration_source_kind = None
            self.metrics.gpu_fallback_resumes += 1
            if source_kind == MigrationSourceKind.SSD:
                self.metrics.ssd_restore_resumes += 1
            return ResumeRoute(
                execution=(
                    ResumeExecution.GPU
                    if source_kind == MigrationSourceKind.GPU
                    else ResumeExecution.GPU_RESTORE
                ),
                session_id=session_id,
                group_id=None,
                hbf_tokens=0,
                lpddr_tokens=0,
                migration_inflight=True,
                reason=(
                    "migration_inflight_gpu_fallback"
                    if source_kind == MigrationSourceKind.GPU
                    else "ssd_import_inflight_gpu_restore"
                ),
            )
        if record.state == PlacementState.EVICTED:
            self._trim_gpu_lineage(record, reuse_tokens)
            record.state = PlacementState.GPU_ACTIVE
            record.active_request_id = request_id
            self.metrics.gpu_recompute_resumes += 1
            return ResumeRoute(
                execution=ResumeExecution.GPU_RECOMPUTE,
                session_id=session_id,
                group_id=None,
                hbf_tokens=0,
                lpddr_tokens=0,
                migration_inflight=False,
                reason="hbf_capacity_evicted",
            )
        if record.state == PlacementState.GPU_READY:
            self._trim_gpu_lineage(record, reuse_tokens)
            record.state = PlacementState.GPU_ACTIVE
            record.active_request_id = request_id
            self.metrics.gpu_fallback_resumes += 1
            return ResumeRoute(
                execution=ResumeExecution.GPU,
                session_id=session_id,
                group_id=None,
                hbf_tokens=0,
                lpddr_tokens=0,
                migration_inflight=False,
                reason="hbf_capacity_unavailable_gpu_retained",
            )
        if record.state == PlacementState.SSD_READY:
            trimmed = self._trim_gpu_lineage(record, reuse_tokens)
            if not trimmed:
                # Invalidate any caller-side delayed import trigger even
                # when the entire SSD snapshot will be restored.
                record.generation += 1
            record.gpu_retained_bytes = (
                reuse_tokens * self.kv_bytes_per_token)
            record.state = PlacementState.GPU_ACTIVE
            record.active_request_id = request_id
            self.metrics.gpu_fallback_resumes += 1
            self.metrics.ssd_restore_resumes += 1
            return ResumeRoute(
                execution=ResumeExecution.GPU_RESTORE,
                session_id=session_id,
                group_id=None,
                hbf_tokens=0,
                lpddr_tokens=0,
                migration_inflight=False,
                reason="ssd_checkpoint_gpu_restore",
            )
        if record.state == PlacementState.GPU_ACTIVE:
            if record.active_request_id is not None:
                raise RuntimeError(
                    f"session {session_id!r} already has an active request")
            self._trim_gpu_lineage(record, reuse_tokens)
            record.active_request_id = request_id
            return ResumeRoute(
                execution=ResumeExecution.GPU,
                session_id=session_id,
                group_id=None,
                hbf_tokens=0,
                lpddr_tokens=0,
                migration_inflight=False,
                reason="gpu_owned",
            )
        raise RuntimeError(
            f"cannot resume session {session_id!r} from {record.state}")

    def _start_contiguous_append(
            self, record: SessionPlacement, *, now_ns: int,
            token_start: int,
            append_tokens: int) -> Optional[AppendJob]:
        """Reserve and launch one contiguous LPDDR-to-HBF append."""

        if record.group_id is None:
            raise RuntimeError(
                "cannot append an HBF placement without a replica group")
        if (
            isinstance(token_start, bool)
            or not isinstance(token_start, int)
            or token_start < 0
        ):
            raise ValueError("append token_start must be non-negative")
        if (
            isinstance(append_tokens, bool)
            or not isinstance(append_tokens, int)
            or append_tokens <= 0
        ):
            raise ValueError("append token count must be positive")
        group_id = record.group_id
        logical_bytes = append_tokens * self.kv_bytes_per_token
        physical_bytes = self._physical_bytes(logical_bytes)
        card_bytes = self._range_card_bytes(
            group_id,
            token_start=token_start,
            token_count=append_tokens,
        )
        per_card_bytes = self._peak_card_bytes(card_bytes)
        if not self._ensure_capacity(
                group_id, card_bytes, now_ns):
            # The record remains available in LPDDR. Backpressure is
            # explicit: no append is issued and policy may retry later.
            return None
        self._reserve_group(group_id, card_bytes)
        record.pending_reserved_per_card_bytes += per_card_bytes
        job_id = self._next_id()
        if self.execution_backend == "analytical_calendar":
            start_ns, completion_ns = self.calendar.reserve_parallel(
                arrival_ns=now_ns,
                job_id=job_id,
                kind="append",
                namespace="hbf-lifecycle",
                demands=self._append_demands(
                    group_id, card_bytes),
            )
        else:
            start_ns = now_ns
            completion_ns = now_ns
        job = AppendJob(
            job_id=job_id,
            session_id=record.session_id,
            generation=record.generation,
            version=record.version,
            group_id=group_id,
            token_count=append_tokens,
            logical_bytes=logical_bytes,
            physical_bytes=physical_bytes,
            per_card_bytes=per_card_bytes,
            start_ns=start_ns,
            completion_ns=completion_ns,
            token_start=token_start,
            card_bytes=canonical_card_bytes(
                self._group(group_id).card_ids,
                card_bytes,
            ),
        )
        self._jobs[job_id] = job
        if self.execution_backend == "analytical_calendar":
            heapq.heappush(
                self._completion_heap, (completion_ns, job_id))
        else:
            self._enqueue_external_job(job, now_ns)
        record.append_job_ids.add(job_id)
        self.metrics.append_jobs_started += 1
        self.metrics.append_logical_bytes += logical_bytes
        self.metrics.append_physical_bytes += physical_bytes
        self._record_hbf_kv_write(job)
        return job

    def start_active_prefill_drain(
            self, session_id: str, *, request_id: int,
            now_ns: int, total_tokens: int,
            tail_tokens: int = 0) -> ActivePrefillDrainResult:
        """Drain fresh-prefill KV while its HBF request stays active.

        The serving pool first transfers completed prefill KV from the
        request headroom owner to the session owner. This method verifies
        that exact per-card ownership before publishing the new placement.
        A bounded LPDDR tail may remain available for decode.
        """

        if (
            isinstance(request_id, bool)
            or not isinstance(request_id, int)
            or request_id < 0
        ):
            raise ValueError("request_id must be a non-negative integer")
        if (
            isinstance(total_tokens, bool)
            or not isinstance(total_tokens, int)
            or total_tokens < 0
        ):
            raise ValueError("total_tokens must be a non-negative integer")
        if (
            isinstance(tail_tokens, bool)
            or not isinstance(tail_tokens, int)
            or tail_tokens < 0
        ):
            raise ValueError("tail_tokens must be a non-negative integer")
        self.advance(now_ns)
        record = self.sessions[session_id]
        if record.state != PlacementState.HBF_ACTIVE:
            raise RuntimeError(
                "active prefill drain requires HBF_ACTIVE placement")
        if record.active_request_id != request_id:
            raise RuntimeError(
                "active prefill drain request does not own the session: "
                f"expected={record.active_request_id}, "
                f"actual={request_id}")
        if total_tokens < record.total_tokens:
            raise ValueError(
                "active prefill drain cannot shrink session context")
        if record.group_id is None:
            raise RuntimeError(
                "active prefill drain lost its replica group")

        delta_tokens = total_tokens - record.total_tokens
        new_lpddr_tokens = record.lpddr_tokens + delta_tokens
        group_id = record.group_id
        expected_card_bytes = self._range_card_bytes(
            group_id,
            token_start=record.committed_hbf_tokens,
            token_count=new_lpddr_tokens,
        )
        owner = self.lpddr_owner(session_id)
        actual_card_bytes = dict(
            self.lpddr_ledger.owner_card_bytes(owner))
        actual_group_id = self.lpddr_ledger.owner_group(owner)
        normalized_actual = {
            card_id: actual_card_bytes.get(card_id, 0)
            for card_id in self._group(group_id).card_ids
        }
        if (
            normalized_actual != expected_card_bytes
            or (
                any(expected_card_bytes.values())
                and actual_group_id != group_id
            )
            or set(actual_card_bytes) - set(expected_card_bytes)
        ):
            raise RuntimeError(
                "active HBF prefill did not conserve its LPDDR session "
                "owner vector: "
                f"session={session_id!r}, expected={expected_card_bytes}, "
                f"actual={actual_card_bytes}")

        record.total_tokens = total_tokens
        record.lpddr_tokens = new_lpddr_tokens
        record.version += 1
        record.last_access_ns = now_ns
        self.metrics.active_prefill_drain_candidates += 1
        retained_tail_tokens = min(
            tail_tokens, record.lpddr_tokens)
        append_tokens = (
            record.lpddr_tokens - retained_tail_tokens)
        blocking_job_ids = tuple(sorted(
            job_id for job_id in record.append_job_ids
            if isinstance(self._jobs.get(job_id), AppendJob)
        ))
        if blocking_job_ids:
            self.metrics.active_prefill_drain_wait_existing_append += 1
            result = ActivePrefillDrainResult(
                status=(
                    ActivePrefillDrainStatus.WAIT_EXISTING_APPEND),
                job=None,
                total_tokens=record.total_tokens,
                lpddr_tokens=record.lpddr_tokens,
                append_tokens=append_tokens,
                retained_tail_tokens=retained_tail_tokens,
                blocking_append_job_ids=blocking_job_ids,
            )
            if self.validate_every_event:
                self.assert_invariants()
            return result
        if append_tokens == 0:
            self.metrics.active_prefill_drain_satisfied += 1
            result = ActivePrefillDrainResult(
                status=ActivePrefillDrainStatus.SATISFIED,
                job=None,
                total_tokens=record.total_tokens,
                lpddr_tokens=record.lpddr_tokens,
                append_tokens=0,
                retained_tail_tokens=retained_tail_tokens,
            )
            if self.validate_every_event:
                self.assert_invariants()
            return result

        job = self._start_contiguous_append(
            record,
            now_ns=now_ns,
            token_start=record.committed_hbf_tokens,
            append_tokens=append_tokens,
        )
        if job is None:
            self.metrics.active_prefill_drain_capacity_fallback += 1
            result = ActivePrefillDrainResult(
                status=ActivePrefillDrainStatus.CAPACITY_FALLBACK,
                job=None,
                total_tokens=record.total_tokens,
                lpddr_tokens=record.lpddr_tokens,
                append_tokens=append_tokens,
                retained_tail_tokens=retained_tail_tokens,
            )
            if self.validate_every_event:
                self.assert_invariants()
            return result

        self._active_prefill_drain_job_ids.add(job.job_id)
        self.metrics.active_prefill_drain_started += 1
        result = ActivePrefillDrainResult(
            status=ActivePrefillDrainStatus.STARTED,
            job=job,
            total_tokens=record.total_tokens,
            lpddr_tokens=record.lpddr_tokens,
            append_tokens=append_tokens,
            retained_tail_tokens=retained_tail_tokens,
        )
        if self.validate_every_event:
            self.assert_invariants()
        return result

    def complete_hbf_turn(
            self, session_id: str, *, now_ns: int,
            total_tokens: int, has_successor: bool) -> Optional[AppendJob]:
        self.advance(now_ns)
        record = self.sessions[session_id]
        if record.state != PlacementState.HBF_ACTIVE:
            raise RuntimeError(
                f"HBF completion from invalid state {record.state}")
        if total_tokens < record.total_tokens:
            raise ValueError("HBF completion cannot shrink session context")
        delta_tokens = total_tokens - record.total_tokens
        new_lpddr_tokens = record.lpddr_tokens + delta_tokens
        headroom_owner = (
            None
            if record.active_request_id is None
            else hbf_request_headroom_owner(
                record.active_request_id)
        )
        headroom_card_bytes = (
            {}
            if headroom_owner is None
            else dict(self.lpddr_ledger.owner_card_bytes(
                headroom_owner))
        )
        if headroom_owner is not None and headroom_card_bytes:
            current_card_bytes = dict(
                self.lpddr_ledger.owner_card_bytes(
                    self.lpddr_owner(session_id)))
            target_card_bytes = self._range_card_bytes(
                record.group_id,
                token_start=record.committed_hbf_tokens,
                token_count=new_lpddr_tokens,
            )
            if any(
                    current_card_bytes.get(card_id, 0)
                    + headroom_card_bytes.get(card_id, 0)
                    != target_card_bytes[card_id]
                    for card_id in target_card_bytes):
                raise RuntimeError(
                    "active HBF request did not conserve its LPDDR "
                    "finish reservation")
            self.lpddr_ledger.release(headroom_owner)
        if has_successor:
            if record.group_id is None:
                raise RuntimeError(
                    "HBF completion lost its replica group")
            self.lpddr_ledger.set_card_bytes(
                record.group_id,
                self.lpddr_owner(session_id),
                self._range_card_bytes(
                    record.group_id,
                    token_start=record.committed_hbf_tokens,
                    token_count=new_lpddr_tokens,
                ),
            )
        record.total_tokens = total_tokens
        record.version += 1
        record.last_access_ns = now_ns
        record.active_request_id = None
        if not has_successor:
            self.end_session(session_id, now_ns=now_ns)
            return None
        record.state = PlacementState.HBF_READY
        pending_append_tokens = sum(
            self._jobs[job_id].token_count
            for job_id in record.append_job_ids
            if (
                isinstance(self._jobs.get(job_id), AppendJob)
                and self._jobs[job_id].generation == record.generation
            )
        )
        record.lpddr_tokens = new_lpddr_tokens
        append_tokens = record.lpddr_tokens - pending_append_tokens
        if append_tokens <= 0:
            return None
        token_start = (
            record.committed_hbf_tokens + pending_append_tokens)
        return self._start_contiguous_append(
            record,
            now_ns=now_ns,
            token_start=token_start,
            append_tokens=append_tokens,
        )

    def _finish_migration(self, job: MigrationJob) -> None:
        record = self.sessions[job.session_id]
        record.migration_job_ids.discard(job.job_id)
        record.pending_reserved_per_card_bytes -= job.per_card_bytes
        job_card_bytes = self._job_card_bytes(job)
        valid = (
            record.state == PlacementState.MIGRATING
            and record.generation == job.generation
            and record.version == job.version
            and record.group_id == job.group_id
        )
        if valid:
            record.state = PlacementState.HBF_READY
            record.migration_source_kind = None
            record.committed_hbf_tokens = job.token_count
            record.lpddr_tokens = 0
            self.lpddr_ledger.release(
                self.lpddr_owner(record.session_id))
            record.committed_per_card_bytes = self._peak_card_bytes(
                job_card_bytes)
            record.gpu_retained_bytes = 0
            self.metrics.migrations_committed += 1
            if job.source_kind == MigrationSourceKind.SSD:
                self.metrics.ssd_imports_committed += 1
        else:
            self._release_group(job.group_id, job_card_bytes)
            self.metrics.migrations_stale += 1
            self.metrics.migration_wasted_physical_bytes += (
                job.physical_bytes)
            self._record_hbf_wasted_write(job)
            if job.source_kind == MigrationSourceKind.SSD:
                self.metrics.ssd_imports_stale += 1
                self.metrics.ssd_import_wasted_physical_bytes += (
                    job.physical_bytes)

    def _finish_append(self, job: AppendJob) -> None:
        record = self.sessions[job.session_id]
        record.append_job_ids.discard(job.job_id)
        record.pending_reserved_per_card_bytes -= job.per_card_bytes
        active_prefill_drain = (
            job.job_id in self._active_prefill_drain_job_ids)
        self._active_prefill_drain_job_ids.discard(job.job_id)
        job_card_bytes = self._job_card_bytes(job)
        valid = (
            record.state != PlacementState.ENDED
            and record.generation == job.generation
            and record.group_id == job.group_id
            and job.version <= record.version
        )
        contiguous = (
            valid
            and job.token_start == record.committed_hbf_tokens
        )
        if contiguous:
            if job.token_count > record.lpddr_tokens:
                raise RuntimeError(
                    "append commits more tokens than LPDDR owns")
            record.committed_hbf_tokens += job.token_count
            record.lpddr_tokens -= job.token_count
            record.committed_per_card_bytes = self._peak_card_bytes(
                self._record_committed_card_bytes(record))
            self.lpddr_ledger.shrink_card_bytes(
                self.lpddr_owner(record.session_id),
                job_card_bytes,
            )
            self.metrics.append_jobs_committed += 1
            if active_prefill_drain:
                self.metrics.active_prefill_drain_committed += 1
        else:
            # An out-of-order completion cannot publish a non-contiguous
            # HBF prefix.  Its source remains in LPDDR, so releasing only the
            # speculative HBF reservation is lossless and a later turn can
            # retry the append.
            self._release_group(job.group_id, job_card_bytes)
            self.metrics.append_jobs_stale += 1
            self.metrics.append_wasted_physical_bytes += (
                job.physical_bytes)
            self._record_hbf_wasted_write(job)
            if active_prefill_drain:
                self.metrics.active_prefill_drain_stale += 1

    def _require_external_astra(self, operation: str) -> None:
        if self.execution_backend != "external_astra":
            raise RuntimeError(
                f"{operation} requires "
                "execution_backend='external_astra'")

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
            self) -> tuple[HBFLifecycleExternalDispatch, ...]:
        """Return each newly created ASTRA lifecycle job exactly once."""

        self._require_external_astra("drain_external_dispatches")
        dispatches = tuple(self._external_outbox)
        self._external_outbox.clear()
        for dispatch in dispatches:
            if self._external_pending.get(
                    dispatch.job_id) is not dispatch:
                raise RuntimeError(
                    "lifecycle external ASTRA outbox/pending "
                    "identity mismatch")
            if dispatch.job_id in self._external_issued_job_ids:
                raise RuntimeError(
                    "lifecycle external ASTRA job was issued "
                    "more than once")
            self._external_issued_job_ids.add(dispatch.job_id)
        if self.validate_every_event:
            self.assert_invariants()
        return dispatches

    def complete_external_dispatch(
            self, job_id: str, arrival_ns: int, completion_ns: int,
            stage_count: int,
    ) -> MigrationJob | AppendJob:
        """Apply the sole authoritative completion for one ASTRA job.

        The caller must deliver this callback before routing arrivals with
        the same timestamp.  That makes lifecycle publication win exact
        completion/arrival ties without inventing completions in
        :meth:`advance`.
        """

        self._require_external_astra("complete_external_dispatch")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        arrival = self._callback_nonnegative_int(
            "arrival_ns", arrival_ns)
        completion = self._callback_nonnegative_int(
            "completion_ns", completion_ns)
        stages = self._callback_nonnegative_int(
            "stage_count", stage_count)
        if job_id in self._external_completed_job_ids:
            raise RuntimeError(
                "duplicate lifecycle external ASTRA completion for "
                f"{job_id!r}")
        dispatch = self._external_pending.get(job_id)
        if dispatch is None:
            raise RuntimeError(
                "unknown lifecycle external ASTRA completion job "
                f"{job_id!r}")
        if job_id not in self._external_issued_job_ids:
            raise RuntimeError(
                f"lifecycle external ASTRA job {job_id!r} "
                "was not drained")
        if arrival != dispatch.arrival_ns:
            raise RuntimeError(
                "lifecycle external ASTRA completion arrival mismatch: "
                f"job={job_id!r}, expected={dispatch.arrival_ns}, "
                f"actual={arrival}")
        if stages != dispatch.stage_count:
            raise RuntimeError(
                "lifecycle external ASTRA completion stage-count "
                f"mismatch: job={job_id!r}, "
                f"expected={dispatch.stage_count}, actual={stages}")
        dependency_elapsed = (
            dispatch.projection.dependency_critical_path_ns())
        solo_resource_elapsed = (
            dispatch.projection
            .solo_resource_serialized_completion_ns()
        )
        minimum_completion = arrival + dependency_elapsed
        if completion < minimum_completion:
            raise RuntimeError(
                "lifecycle external ASTRA completion precedes the "
                "dependency critical path bound: "
                f"job={job_id!r}, minimum={minimum_completion}, "
                f"actual={completion}")
        if completion < self.current_ns:
            raise RuntimeError(
                "lifecycle external ASTRA completion moves time "
                f"backwards: current={self.current_ns}, "
                f"actual={completion}")
        if self._jobs.get(dispatch.job.job_id) is not dispatch.job:
            raise RuntimeError(
                "lifecycle external ASTRA job identity mismatch")

        # External advance only establishes the callback timestamp.  It
        # cannot consume this or any other pending ASTRA completion.
        self.advance(completion)
        del self._external_pending[job_id]
        self._external_issued_job_ids.remove(job_id)
        self._external_completed_job_ids.add(job_id)
        pending_job = self._jobs.pop(dispatch.job.job_id)
        if pending_job is not dispatch.job:
            raise RuntimeError(
                "lifecycle external ASTRA completion replaced its job")
        completed_job = replace(
            pending_job, completion_ns=completion)
        if isinstance(completed_job, MigrationJob):
            self._finish_migration(completed_job)
        else:
            self._finish_append(completed_job)
        self.metrics.astra_completed_jobs += 1
        actual_elapsed = completion - arrival
        timing = HBFAstraTimingAccounting(
            dependency_critical_path_ns=dependency_elapsed,
            solo_resource_serialized_completion_ns=(
                solo_resource_elapsed),
            actual_resource_serialized_completion_ns=actual_elapsed,
        )
        self.metrics.astra_completion_elapsed_ns += actual_elapsed
        self.metrics.astra_resource_delay_ns += timing.resource_delay_ns
        self.metrics.astra_dependency_critical_path_ns += (
            dependency_elapsed)
        self.metrics.astra_solo_resource_serialized_completion_ns += (
            solo_resource_elapsed)
        self.metrics.astra_actual_resource_serialized_completion_ns += (
            actual_elapsed)
        self.metrics.astra_internal_resource_serialization_wait_ns += (
            timing.internal_resource_serialization_wait_ns)
        self.metrics.astra_signed_interference_delta_ns += (
            timing.signed_interference_delta_ns)
        if self.validate_every_event:
            self.assert_invariants()
        return completed_job

    def has_pending_external(self) -> bool:
        """Return whether ASTRA still owns any lifecycle completion."""

        return bool(self._external_pending)

    def advance(self, now_ns: int) -> None:
        if now_ns < 0:
            raise ValueError("now_ns must be non-negative")
        if now_ns < self.current_ns:
            raise ValueError(
                f"time cannot move backwards: current={self.current_ns}, "
                f"requested={now_ns}")
        self.current_ns = now_ns
        if self.execution_backend == "analytical_calendar":
            while (
                self._completion_heap
                and self._completion_heap[0][0] <= now_ns
            ):
                _, job_id = heapq.heappop(self._completion_heap)
                job = self._jobs.pop(job_id)
                if isinstance(job, MigrationJob):
                    self._finish_migration(job)
                else:
                    self._finish_append(job)
        if self.validate_every_event:
            self.assert_invariants()

    def next_completion_ns(self) -> Optional[int]:
        if self.execution_backend == "external_astra":
            return None
        return (
            None if not self._completion_heap
            else self._completion_heap[0][0]
        )

    def run_until_idle(self) -> None:
        while self._completion_heap:
            event_ns = self.next_completion_ns()
            assert event_ns is not None
            self.advance(event_ns)
        if self._external_pending:
            raise RuntimeError(
                "external ASTRA lifecycle completions are pending; "
                "drain dispatches and apply "
                "complete_external_dispatch callbacks")
        self.assert_invariants()

    def end_session(self, session_id: str, *, now_ns: int) -> None:
        self.advance(now_ns)
        record = self.sessions[session_id]
        if record.state == PlacementState.ENDED:
            return
        record.generation += 1
        if record.active_request_id is not None:
            self.lpddr_ledger.release(
                hbf_request_headroom_owner(
                    record.active_request_id))
        self.lpddr_ledger.release(self.lpddr_owner(session_id))
        if record.group_id is not None:
            if record.committed_per_card_bytes:
                self._release_group(
                    record.group_id,
                    self._record_committed_card_bytes(record),
                )
            record.committed_per_card_bytes = 0
        record.state = PlacementState.ENDED
        record.group_id = None
        record.migration_source_kind = None
        record.gpu_retained_bytes = 0
        record.committed_hbf_tokens = 0
        record.lpddr_tokens = 0
        record.last_access_ns = now_ns
        record.active_request_id = None

    def assert_invariants(self) -> None:
        self.lpddr_ledger.assert_invariants()
        self._assert_hbf_write_accounting()
        try:
            validate_hbf_astra_timing_metrics(self.metrics)
        except HBFModelAstraProjectionError as exc:
            raise AssertionError(str(exc)) from exc
        if self.execution_backend == "analytical_calendar":
            if (
                self._external_outbox
                or self._external_pending
                or self._external_issued_job_ids
                or self._external_completed_job_ids
            ):
                raise AssertionError(
                    "analytical lifecycle contains external ASTRA state")
        else:
            if self._completion_heap:
                raise AssertionError(
                    "external ASTRA lifecycle contains Python timing "
                    "events")
            if (
                self.calendar.available_ns
                or self.calendar.busy_ns
                or self.calendar.reservations
                or self.calendar.reservation_count_by_resource
                or self.calendar.reservation_bytes_by_resource
                or self.calendar.reservation_count_by_namespace_kind
                or self.calendar.reservation_bytes_by_namespace_kind
            ):
                raise AssertionError(
                    "external ASTRA lifecycle used the Python resource "
                    "calendar")
            outbox_ids = [
                dispatch.job_id
                for dispatch in self._external_outbox
            ]
            if len(outbox_ids) != len(set(outbox_ids)):
                raise AssertionError(
                    "lifecycle external ASTRA outbox contains duplicate "
                    "jobs")
            pending_ids = set(self._external_pending)
            if not set(outbox_ids) <= pending_ids:
                raise AssertionError(
                    "lifecycle external ASTRA outbox contains a "
                    "non-pending job")
            if set(outbox_ids) & self._external_issued_job_ids:
                raise AssertionError(
                    "lifecycle external ASTRA job is both queued and "
                    "issued")
            if pending_ids != (
                    set(outbox_ids) | self._external_issued_job_ids):
                raise AssertionError(
                    "lifecycle external ASTRA pending job has no "
                    "ownership state")
            if (
                pending_ids & self._external_completed_job_ids
                or self._external_issued_job_ids
                & self._external_completed_job_ids
            ):
                raise AssertionError(
                    "completed lifecycle external ASTRA job remains live")
            pending_numeric_ids: set[int] = set()
            for job_id, dispatch in self._external_pending.items():
                if dispatch.job_id != job_id:
                    raise AssertionError(
                        "lifecycle external ASTRA pending key mismatch")
                if self._jobs.get(
                        dispatch.job.job_id) is not dispatch.job:
                    raise AssertionError(
                        "lifecycle external ASTRA pending job identity "
                        "mismatch")
                pending_numeric_ids.add(dispatch.job.job_id)
            if pending_numeric_ids != set(self._jobs):
                raise AssertionError(
                    "lifecycle external ASTRA pending/job registry "
                    "mismatch")
            for dispatch in self._external_outbox:
                if self._external_pending.get(
                        dispatch.job_id) is not dispatch:
                    raise AssertionError(
                        "lifecycle external ASTRA outbox identity "
                        "mismatch")
        if not self._active_prefill_drain_job_ids <= set(self._jobs):
            raise AssertionError(
                "active prefill drain tracks a non-pending append")
        for job_id in self._active_prefill_drain_job_ids:
            job = self._jobs[job_id]
            if (
                not isinstance(job, AppendJob)
                or job_id not in self.sessions[
                    job.session_id].append_job_ids
            ):
                raise AssertionError(
                    "active prefill drain job ownership mismatch")
        if self.metrics.active_prefill_drain_candidates != sum((
            self.metrics.active_prefill_drain_started,
            self.metrics.active_prefill_drain_satisfied,
            self.metrics.active_prefill_drain_wait_existing_append,
            self.metrics.active_prefill_drain_capacity_fallback,
        )):
            raise AssertionError(
                "active prefill drain outcome accounting mismatch")
        if self.metrics.active_prefill_drain_started != sum((
            self.metrics.active_prefill_drain_committed,
            self.metrics.active_prefill_drain_stale,
            len(self._active_prefill_drain_job_ids),
        )):
            raise AssertionError(
                "active prefill drain completion accounting mismatch")
        pending_migrations = tuple(
            job
            for job in self._jobs.values()
            if isinstance(job, MigrationJob)
        )
        if any(
                not isinstance(job.source_kind, MigrationSourceKind)
                for job in pending_migrations):
            raise AssertionError(
                "pending migration has an invalid source kind")
        if self.metrics.migrations_started != sum((
            self.metrics.migrations_committed,
            self.metrics.migrations_stale,
            len(pending_migrations),
        )):
            raise AssertionError(
                "migration completion accounting mismatch")
        pending_ssd_imports = sum(
            job.source_kind == MigrationSourceKind.SSD
            for job in pending_migrations
        )
        if self.metrics.ssd_imports_started != sum((
            self.metrics.ssd_imports_committed,
            self.metrics.ssd_imports_stale,
            pending_ssd_imports,
        )):
            raise AssertionError(
                "SSD import completion accounting mismatch")
        if (
            self.metrics.ssd_imports_started
            > self.metrics.migrations_started
            or self.metrics.ssd_imports_committed
            > self.metrics.migrations_committed
            or self.metrics.ssd_imports_stale
            > self.metrics.migrations_stale
            or self.metrics.ssd_import_physical_bytes
            > self.metrics.migration_physical_bytes
            or self.metrics.ssd_import_logical_bytes
            > self.metrics.migration_logical_bytes
            or self.metrics.ssd_import_wasted_physical_bytes
            > self.metrics.migration_wasted_physical_bytes
        ):
            raise AssertionError(
                "SSD import metrics exceed aggregate migration metrics")
        expected_group_reservations = {
            group.group_id: {
                card_id: 0 for card_id in group.card_ids
            }
            for group in self.groups
        }
        for record in self.sessions.values():
            if record.group_id is not None:
                committed = self._record_committed_card_bytes(record)
                for card_id, byte_count in committed.items():
                    expected_group_reservations[
                        record.group_id][card_id] += byte_count
        for job in self._jobs.values():
            for card_id, byte_count in self._job_card_bytes(job).items():
                expected_group_reservations[
                    job.group_id][card_id] += byte_count
        for group_id, reserved_by_card in (
                self._reserved_bytes_by_card.items()):
            if any(
                    not 0 <= byte_count <= self.usable_bytes_per_card
                    for byte_count in reserved_by_card.values()):
                raise AssertionError(
                    f"invalid HBF group reservation: "
                    f"group={group_id}, bytes={reserved_by_card}")
            if reserved_by_card != expected_group_reservations[group_id]:
                raise AssertionError(
                    f"HBF reservation ledger mismatch: "
                    f"group={group_id}, reserved={reserved_by_card}, "
                    f"expected={expected_group_reservations[group_id]}")
            expected_peak = max(
                reserved_by_card.values(), default=0)
            if (
                self._reserved_per_card_by_group[group_id]
                != expected_peak
            ):
                raise AssertionError(
                    "HBF reservation ledger mismatch "
                    "(scalar/vector): "
                    f"group={group_id}, "
                    f"scalar={self._reserved_per_card_by_group[group_id]}, "
                    f"vector_peak={expected_peak}")
        for record in self.sessions.values():
            lpddr_reserved = dict(
                self.lpddr_ledger.owner_card_bytes(
                    self.lpddr_owner(record.session_id)))
            expected_lpddr = self._record_lpddr_card_bytes(record)
            expected_pending = sum(
                self._jobs[job_id].per_card_bytes
                for job_id in (
                    record.migration_job_ids | record.append_job_ids)
                if job_id in self._jobs
            )
            for name in (
                "generation",
                "version",
                "total_tokens",
                "committed_hbf_tokens",
                "lpddr_tokens",
                "gpu_retained_bytes",
                "committed_per_card_bytes",
                "pending_reserved_per_card_bytes",
            ):
                if getattr(record, name) < 0:
                    raise AssertionError(
                        f"negative session field {name}: {record}")
            if record.pending_reserved_per_card_bytes != expected_pending:
                raise AssertionError(
                    f"pending HBF reservation mismatch: {record}, "
                    f"expected={expected_pending}")
            expected_committed_peak = self._peak_card_bytes(
                self._record_committed_card_bytes(record))
            if (
                record.committed_per_card_bytes
                != expected_committed_peak
            ):
                raise AssertionError(
                    f"committed HBF scalar/vector mismatch: {record}, "
                    f"expected_peak={expected_committed_peak}")
            if record.committed_hbf_tokens + record.lpddr_tokens > (
                    record.total_tokens):
                raise AssertionError(
                    f"session media tokens exceed total: {record}")
            if (
                record.state == PlacementState.HBF_READY
                and any(
                    lpddr_reserved.get(card_id, 0) != byte_count
                    for card_id, byte_count in expected_lpddr.items()
                )
            ):
                raise AssertionError(
                    f"idle LPDDR reservation mismatch: {record}, "
                    f"reserved={lpddr_reserved}, "
                    f"expected={expected_lpddr}")
            if (
                record.state == PlacementState.HBF_ACTIVE
                and any(
                    lpddr_reserved.get(card_id, 0) < byte_count
                    for card_id, byte_count in expected_lpddr.items()
                )
            ):
                raise AssertionError(
                    f"active LPDDR reservation lost committed delta: "
                    f"{record}, reserved={lpddr_reserved}, "
                    f"minimum={expected_lpddr}")
            if (
                record.state not in {
                    PlacementState.HBF_READY,
                    PlacementState.HBF_ACTIVE,
                }
                and any(lpddr_reserved.values())
            ):
                raise AssertionError(
                    f"non-HBF session owns LPDDR: {record}")
            if (
                record.state in {
                    PlacementState.HBF_READY,
                    PlacementState.HBF_ACTIVE,
                    PlacementState.MIGRATING,
                }
                and record.group_id is None
            ):
                raise AssertionError(
                    f"HBF state without group: {record}")
            if record.state == PlacementState.MIGRATING:
                current_jobs = [
                    job
                    for job_id in record.migration_job_ids
                    if (
                        isinstance(
                            (job := self._jobs.get(job_id)),
                            MigrationJob,
                        )
                        and job.generation == record.generation
                        and job.version == record.version
                        and job.group_id == record.group_id
                    )
                ]
                if len(current_jobs) != 1:
                    raise AssertionError(
                        "migrating session lacks exactly one current job: "
                        f"{record}")
                current_job = current_jobs[0]
                if (
                    record.migration_source_kind
                    != current_job.source_kind
                ):
                    raise AssertionError(
                        "migration source bookkeeping mismatch: "
                        f"{record}")
                if (
                    current_job.source_kind == MigrationSourceKind.GPU
                    and record.gpu_retained_bytes <= 0
                ):
                    raise AssertionError(
                        f"GPU migration lacks retained KV: {record}")
                if (
                    current_job.source_kind == MigrationSourceKind.SSD
                    and record.gpu_retained_bytes != 0
                ):
                    raise AssertionError(
                        f"SSD import unexpectedly retains GPU KV: {record}")
            elif record.migration_source_kind is not None:
                raise AssertionError(
                    "non-migrating session retains a migration source: "
                    f"{record}")
            if (
                record.state == PlacementState.GPU_READY
                and record.gpu_retained_bytes <= 0
            ):
                raise AssertionError(
                    f"GPU-ready session lacks retained KV: {record}")
            if (
                record.state == PlacementState.SSD_READY
                and (
                    record.gpu_retained_bytes
                    or record.group_id is not None
                    or record.committed_hbf_tokens
                    or record.lpddr_tokens
                    or record.committed_per_card_bytes
                    or record.pending_reserved_per_card_bytes
                    or record.migration_job_ids
                    or record.append_job_ids
                )
            ):
                raise AssertionError(
                    f"SSD-ready session has non-SSD ownership: {record}")
            if (
                record.state == PlacementState.ENDED
                and (
                    record.gpu_retained_bytes
                    or record.committed_per_card_bytes
                    or any(lpddr_reserved.values())
                )
            ):
                raise AssertionError(
                    f"ended session retains committed storage: {record}")

    def report(self) -> dict[str, Any]:
        if self.execution_backend == "external_astra":
            from .hbf_full_model_lifecycle_astra import (
                PROJECTION_FIDELITY,
                PROJECTION_SCHEMA,
            )
            projection_schema: Optional[str] = PROJECTION_SCHEMA
            projection_fidelity: Optional[str] = PROJECTION_FIDELITY
        else:
            projection_schema = None
            projection_fidelity = None
        result = {
            "layout": asdict(self.layout),
            "hardware": asdict(self.hardware),
            "execution_backend": self.execution_backend,
            "server_id": self.server_id,
            "astra_chunk_bytes": self.astra_chunk_bytes,
            "completion_time_source": (
                "external_astra_callback"
                if self.execution_backend == "external_astra"
                else "python_analytical_calendar"
            ),
            "astra_projection_schema": projection_schema,
            "astra_projection_fidelity": projection_fidelity,
            "astra_timing_semantics": (
                dict(ASTRA_NAMED_RESOURCE_TIMING_SEMANTICS)
                if self.execution_backend == "external_astra"
                else None
            ),
            "validate_every_event": self.validate_every_event,
            "gpu_source_root_bandwidth_gbps": (
                self.gpu_source_root_bandwidth_gbps),
            "gpu_source_node_id": self.gpu_source_node_id,
            "gpu_source_cpu_bandwidth_gbps": (
                self.gpu_source_cpu_bandwidth_gbps),
            "gpu_source_nic_bandwidth_gbps": (
                self.gpu_source_nic_bandwidth_gbps),
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "weight_bytes_per_rank": self.weight_bytes_per_rank,
            "usable_bytes_per_card": self.usable_bytes_per_card,
            "metrics": asdict(self.metrics),
            "hbf_write_accounting": (
                self._hbf_write_accounting_report()),
            "group_reserved_per_card_bytes": dict(
                self._reserved_per_card_by_group),
            "group_reserved_bytes_by_card": {
                group_id: dict(card_bytes)
                for group_id, card_bytes in
                self._reserved_bytes_by_card.items()
            },
            "resource_busy_ns": dict(self.calendar.busy_ns),
            "resource_calendar": self.calendar.report(),
            "lpddr_used_bytes_per_group": {
                group.group_id: self.lpddr_ledger.used_bytes(
                    group.group_id)
                for group in self.groups
            },
            "lpddr_peak_bytes_per_group": dict(
                self.lpddr_ledger.peak_used_bytes),
            "lpddr_used_bytes_by_card": {
                group.group_id: dict(
                    self.lpddr_ledger.used_bytes_by_card(
                        group.group_id))
                for group in self.groups
            },
            "lpddr_peak_bytes_by_card": {
                group_id: dict(card_bytes)
                for group_id, card_bytes in
                self.lpddr_ledger.peak_used_bytes_by_card.items()
            },
            "pending_job_count": len(self._jobs),
            "pending_migration_jobs": [
                {
                    "job_id": job.job_id,
                    "session_id": job.session_id,
                    "generation": job.generation,
                    "version": job.version,
                    "group_id": job.group_id,
                    "source_kind": job.source_kind.value,
                }
                for job in sorted(
                    (
                        pending
                        for pending in self._jobs.values()
                        if isinstance(pending, MigrationJob)
                    ),
                    key=lambda pending: pending.job_id,
                )
            ],
            "active_prefill_drain_pending_job_ids": sorted(
                self._active_prefill_drain_job_ids),
            "external_pending_job_ids": sorted(
                self._external_pending),
            "external_undrained_dispatch_count": len(
                self._external_outbox),
            "external_issued_dispatch_count": len(
                self._external_issued_job_ids),
            "external_completed_dispatch_count": len(
                self._external_completed_job_ids),
            "current_ns": self.current_ns,
            "sessions": {
                session_id: {
                    **asdict(record),
                    "state": record.state.value,
                    "migration_source_kind": (
                        None
                        if record.migration_source_kind is None
                        else record.migration_source_kind.value
                    ),
                    "migration_job_ids": sorted(
                        record.migration_job_ids),
                    "append_job_ids": sorted(record.append_job_ids),
                }
                for session_id, record in sorted(self.sessions.items())
            },
        }
        if self.analytical_resource_prefix:
            result["analytical_resource_prefix"] = (
                self.analytical_resource_prefix)
        return result


__all__ = [
    "ActivePrefillDrainResult",
    "ActivePrefillDrainStatus",
    "AppendJob",
    "FullModelHBFLifecycle",
    "GPUReadyPressureEviction",
    "HBFLifecycleExternalDispatch",
    "HBFReplicaGroup",
    "LifecycleMetrics",
    "MigrationJob",
    "MigrationSourceKind",
    "PerGroupCapacityLedger",
    "PlacementState",
    "ResourceCalendar",
    "ResourceReservation",
    "ResumeExecution",
    "ResumeRoute",
    "SessionPlacement",
    "canonical_card_bytes",
    "hbf_kv_range_card_bytes",
    "hbf_request_headroom_owner",
]

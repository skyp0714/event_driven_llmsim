"""Validated PCIe topology and shared resource names for HBF servers.

The HBF foreground and lifecycle projectors both submit named-resource DAGs
to the same ASTRA interactive engine.  They must therefore use one canonical
resource namespace for physical PCIe links.  Otherwise a model collective
and a KV migration can incorrectly overlap on the same card or root complex.

This module deliberately models only the topology required by the current
eight-card server.  It is not a replacement for native ASTRA PCIe endpoints:
route service remains an analytical stage duration, while the names below
provide deterministic contention and conservation boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence


SHARED_RESOURCE_MODE = "shared"
LEGACY_ISOLATED_RESOURCE_MODE = "legacy_isolated"
SUPPORTED_RESOURCE_MODES = frozenset({
    SHARED_RESOURCE_MODE,
    LEGACY_ISOLATED_RESOURCE_MODE,
})

CROSS_ROOT_P2P_MODE = "cross_root"
SAME_ROOT_ONLY_P2P_MODE = "same_root_only"
SUPPORTED_P2P_MODES = frozenset({
    CROSS_ROOT_P2P_MODE,
    SAME_ROOT_ONLY_P2P_MODE,
})

ROOT_LOCAL_ROUTE = "root_local"
CROSS_ROOT_ROUTE = "cross_root"
PAIR_COLLECTIVE_TYPES = frozenset({
    "PAIR_QUERY_EXCHANGE",
    "PAIR_SOFTMAX_PARTIAL_EXCHANGE",
})


class HBFPCIeTopologyError(ValueError):
    """Raised when an HBF PCIe topology or route is ambiguous."""


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HBFPCIeTopologyError(
            f"{name} must be a non-negative integer")
    return value


def _positive_int(name: str, value: object) -> int:
    value = _nonnegative_int(name, value)
    if value == 0:
        raise HBFPCIeTopologyError(
            f"{name} must be a positive integer")
    return value


def _positive_float(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise HBFPCIeTopologyError(
            f"{name} must be positive and finite")
    return float(value)


def _nonnegative_float(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise HBFPCIeTopologyError(
            f"{name} must be non-negative and finite")
    return float(value)


def _integer_tuple(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise HBFPCIeTopologyError(
            f"{name} must be a non-empty tuple")
    result = []
    for item in value:
        result.append(_nonnegative_int(name, item))
    return tuple(result)


@dataclass(frozen=True)
class HBFPCIeTopology:
    """Immutable topology and resource namespace for one HBF server."""

    server_id: int
    resource_mode: str
    card_to_root: tuple[int, ...]
    nic_to_root: tuple[int, ...]
    root_count: int
    cards_per_root: int
    card_bandwidth_gbps: float
    card_fixed_latency_us: float
    root_bandwidth_gbps: float
    root_fixed_latency_us: float
    inter_root_bandwidth_gbps: float
    inter_root_fixed_latency_us: float
    p2p_mode: str

    @classmethod
    def from_hardware(
            cls, hardware: Any, *, server_id: int,
    ) -> "HBFPCIeTopology":
        value = cls(
            server_id=_nonnegative_int("server_id", server_id),
            resource_mode=hardware.pcie_resource_mode,
            card_to_root=tuple(hardware.pcie_card_to_root),
            nic_to_root=tuple(hardware.pcie_nic_to_root),
            root_count=hardware.pcie_root_count,
            cards_per_root=hardware.cards_per_pcie_root,
            card_bandwidth_gbps=(
                hardware.intra_fabric_bandwidth_gbps_per_card),
            card_fixed_latency_us=(
                hardware.intra_fabric_fixed_latency_us),
            root_bandwidth_gbps=hardware.pcie_root_bandwidth_gbps,
            root_fixed_latency_us=(
                hardware.pcie_root_fixed_latency_us),
            inter_root_bandwidth_gbps=(
                hardware.pcie_inter_root_bandwidth_gbps),
            inter_root_fixed_latency_us=(
                hardware.pcie_inter_root_fixed_latency_us),
            p2p_mode=hardware.pcie_p2p_mode,
        )
        value.validate()
        return value

    def validate(self) -> None:
        _nonnegative_int("server_id", self.server_id)
        if self.resource_mode not in SUPPORTED_RESOURCE_MODES:
            raise HBFPCIeTopologyError(
                "pcie_resource_mode must be one of "
                f"{sorted(SUPPORTED_RESOURCE_MODES)}")
        roots = _integer_tuple(
            "pcie_card_to_root", self.card_to_root)
        nics = _integer_tuple(
            "pcie_nic_to_root", self.nic_to_root)
        root_count = _positive_int(
            "pcie_root_count", self.root_count)
        cards_per_root = _positive_int(
            "cards_per_pcie_root", self.cards_per_root)
        if root_count != 2 or cards_per_root != 4:
            raise HBFPCIeTopologyError(
                "the current HBF PCIe analytical model requires exactly "
                "two roots with four cards per root")
        if len(roots) != root_count * cards_per_root:
            raise HBFPCIeTopologyError(
                "pcie_card_to_root must map every card exactly once")
        if any(root_id >= root_count for root_id in (*roots, *nics)):
            raise HBFPCIeTopologyError(
                "PCIe card/NIC mapping references an unknown root")
        counts = tuple(roots.count(root_id)
                       for root_id in range(root_count))
        if counts != (cards_per_root,) * root_count:
            raise HBFPCIeTopologyError(
                "every PCIe root must own exactly cards_per_pcie_root "
                f"cards; observed={counts}")
        if len(nics) != 1:
            raise HBFPCIeTopologyError(
                "the current migration model requires exactly one RDMA NIC")
        for name, value in (
            ("card_bandwidth_gbps", self.card_bandwidth_gbps),
            ("root_bandwidth_gbps", self.root_bandwidth_gbps),
            (
                "inter_root_bandwidth_gbps",
                self.inter_root_bandwidth_gbps,
            ),
        ):
            _positive_float(name, value)
        for name, value in (
            ("card_fixed_latency_us", self.card_fixed_latency_us),
            ("root_fixed_latency_us", self.root_fixed_latency_us),
            (
                "inter_root_fixed_latency_us",
                self.inter_root_fixed_latency_us,
            ),
        ):
            _nonnegative_float(name, value)
        if self.p2p_mode not in SUPPORTED_P2P_MODES:
            raise HBFPCIeTopologyError(
                "pcie_p2p_mode must be one of "
                f"{sorted(SUPPORTED_P2P_MODES)}")

    @property
    def nic_root_id(self) -> int:
        return self.nic_to_root[0]

    def root_for_card(self, card_id: int) -> int:
        card = _nonnegative_int("card_id", card_id)
        if card >= len(self.card_to_root):
            raise HBFPCIeTopologyError(
                f"card_id={card} is outside the PCIe topology")
        return self.card_to_root[card]

    def roots_for_cards(
            self, card_ids: Sequence[int]) -> tuple[int, ...]:
        cards = tuple(card_ids)
        if not cards or len(cards) != len(set(cards)):
            raise HBFPCIeTopologyError(
                "card_ids must be non-empty and unique")
        return tuple(dict.fromkeys(
            self.root_for_card(card_id) for card_id in cards))

    def validate_layout(
            self, *, layout_key: str, tp_size: int,
            replicas: int) -> None:
        """Fail closed when a canonical layout cannot follow this topology."""

        self.validate()
        _positive_int("tp_size", tp_size)
        _positive_int("replicas", replicas)
        if tp_size * replicas != len(self.card_to_root):
            raise HBFPCIeTopologyError(
                "layout does not consume every PCIe-mapped card")
        groups = tuple(
            tuple(range(group * tp_size, (group + 1) * tp_size))
            for group in range(replicas)
        )
        if layout_key == "tp4":
            group_roots = tuple(
                self.roots_for_cards(cards) for cards in groups)
            if any(len(roots) != 1 for roots in group_roots):
                raise HBFPCIeTopologyError(
                    "each TP4 replica must be root-local")
            if len({roots[0] for roots in group_roots}) != replicas:
                raise HBFPCIeTopologyError(
                    "TP4 replicas must use independent PCIe roots")
        if layout_key in {"tp8", "tp8_context"}:
            roots = self.roots_for_cards(groups[0])
            if len(roots) < 2:
                raise HBFPCIeTopologyError(
                    f"{layout_key} must span PCIe roots")
            if self.p2p_mode != CROSS_ROOT_P2P_MODE:
                raise HBFPCIeTopologyError(
                    f"{layout_key} requires cross-root PCIe P2P")
        if layout_key == "tp8_context":
            for first in range(0, tp_size, 2):
                pair = groups[0][first:first + 2]
                if len(self.roots_for_cards(pair)) != 1:
                    raise HBFPCIeTopologyError(
                        "every tp8_context KV-head pair must be root-local")

    def root_resource(
            self, root_id: int, *, domain: str) -> str:
        root = _nonnegative_int("root_id", root_id)
        if root >= self.root_count:
            raise HBFPCIeTopologyError(
                f"root_id={root} is outside the PCIe topology")
        if self.resource_mode == SHARED_RESOURCE_MODE:
            return (
                f"hbf-server:{self.server_id}:pcie-root:{root}")
        if domain == "migration":
            return (
                f"hbf-server:{self.server_id}:pcie-root:{root}")
        if domain == "collective":
            return (
                f"hbf-server:{self.server_id}:legacy-collective:"
                f"pcie-root:{root}")
        raise HBFPCIeTopologyError(
            f"unknown PCIe resource domain {domain!r}")

    def card_resource(
            self, card_id: int, *, domain: str) -> str:
        card = _nonnegative_int("card_id", card_id)
        self.root_for_card(card)
        if self.resource_mode == SHARED_RESOURCE_MODE:
            return f"hbf-server:{self.server_id}:card:{card}:pcie"
        if domain == "migration":
            return f"hbf-server:{self.server_id}:card:{card}:pcie"
        if domain == "collective":
            return (
                f"hbf-server:{self.server_id}:card:{card}:fabric")
        raise HBFPCIeTopologyError(
            f"unknown PCIe resource domain {domain!r}")

    def inter_root_resource(
            self, first_root: int, second_root: int, *,
            domain: str) -> str:
        first = _nonnegative_int("first_root", first_root)
        second = _nonnegative_int("second_root", second_root)
        if first == second:
            raise HBFPCIeTopologyError(
                "inter-root resource requires two different roots")
        low, high = sorted((first, second))
        if high >= self.root_count:
            raise HBFPCIeTopologyError(
                "inter-root resource references an unknown root")
        if self.p2p_mode != CROSS_ROOT_P2P_MODE:
            raise HBFPCIeTopologyError(
                "cross-root route requested while PCIe P2P is disabled")
        if self.resource_mode == SHARED_RESOURCE_MODE:
            return (
                f"hbf-server:{self.server_id}:pcie-inter-root:"
                f"{low}-{high}")
        return (
            f"hbf-server:{self.server_id}:legacy-{domain}:"
            f"pcie-inter-root:{low}-{high}")

    def nic_resource(self, nic_id: int = 0) -> str:
        nic = _nonnegative_int("nic_id", nic_id)
        if nic >= len(self.nic_to_root):
            raise HBFPCIeTopologyError(
                f"nic_id={nic} is outside the PCIe topology")
        if self.resource_mode == SHARED_RESOURCE_MODE:
            return f"hbf-server:{self.server_id}:pcie-nic:{nic}"
        return f"hbf-server:{self.server_id}:ingress:rdma"

    def collective_route(
            self, *, card_ids: Sequence[int],
            collective_type: str) -> str:
        roots = self.roots_for_cards(card_ids)
        if collective_type in PAIR_COLLECTIVE_TYPES:
            cards = tuple(card_ids)
            for first in range(0, len(cards), 2):
                if len(self.roots_for_cards(cards[first:first + 2])) != 1:
                    raise HBFPCIeTopologyError(
                        "pair collective crossed a PCIe root")
            return ROOT_LOCAL_ROUTE
        if len(roots) == 1:
            return ROOT_LOCAL_ROUTE
        if self.p2p_mode != CROSS_ROOT_P2P_MODE:
            raise HBFPCIeTopologyError(
                "collective requires disabled cross-root PCIe P2P")
        return CROSS_ROOT_ROUTE

    def collective_resources(
            self, *, replica_id: int,
            card_ids: Sequence[int],
            collective_type: str,
    ) -> tuple[str, ...]:
        replica = _nonnegative_int("replica_id", replica_id)
        cards = tuple(card_ids)
        route = self.collective_route(
            card_ids=cards,
            collective_type=collective_type,
        )
        if self.resource_mode == LEGACY_ISOLATED_RESOURCE_MODE:
            return (
                f"hbf-server:{self.server_id}:replica:{replica}:fabric",
                *(self.card_resource(card_id, domain="collective")
                  for card_id in cards),
            )
        roots = self.roots_for_cards(cards)
        resources = [
            self.root_resource(root_id, domain="collective")
            for root_id in roots
        ]
        if route == CROSS_ROOT_ROUTE:
            for index, first_root in enumerate(roots):
                for second_root in roots[index + 1:]:
                    resources.append(self.inter_root_resource(
                        first_root,
                        second_root,
                        domain="collective",
                    ))
        resources.extend(
            self.card_resource(card_id, domain="collective")
            for card_id in cards
        )
        return tuple(resources)

    def migration_root_resources(
            self, destination_root: int) -> tuple[str, ...]:
        destination = _nonnegative_int(
            "destination_root", destination_root)
        if self.resource_mode == LEGACY_ISOLATED_RESOURCE_MODE:
            return (
                self.root_resource(destination, domain="migration"),)
        source = self.nic_root_id
        if destination == source:
            return (
                self.root_resource(destination, domain="migration"),)
        return (
            self.root_resource(source, domain="migration"),
            self.inter_root_resource(
                source, destination, domain="migration"),
            self.root_resource(destination, domain="migration"),
        )

    def migration_root_service(
            self, destination_root: int,
    ) -> tuple[float, float]:
        """Return collapsed route bandwidth and fixed latency.

        The route stage uses the slowest traversed bandwidth and the largest
        fixed hop latency.  This preserves each stage's analytical service
        floor.  In shared mode, two ready branches of one migration may still
        serialize on their common NIC-root resource; that intentional
        internal wait is outside the dependency-only DAG critical path.
        """

        destination = _nonnegative_int(
            "destination_root", destination_root)
        if destination >= self.root_count:
            raise HBFPCIeTopologyError(
                "migration destination references an unknown root")
        if destination == self.nic_root_id:
            return (
                self.root_bandwidth_gbps,
                self.root_fixed_latency_us,
            )
        if self.p2p_mode != CROSS_ROOT_P2P_MODE:
            raise HBFPCIeTopologyError(
                "migration requires disabled cross-root PCIe P2P")
        return (
            min(
                self.root_bandwidth_gbps,
                self.inter_root_bandwidth_gbps,
            ),
            max(
                self.root_fixed_latency_us,
                self.inter_root_fixed_latency_us,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "resource_mode": self.resource_mode,
            "card_to_root": list(self.card_to_root),
            "nic_to_root": list(self.nic_to_root),
            "root_count": self.root_count,
            "cards_per_root": self.cards_per_root,
            "card_bandwidth_gbps": self.card_bandwidth_gbps,
            "card_fixed_latency_us": self.card_fixed_latency_us,
            "root_bandwidth_gbps": self.root_bandwidth_gbps,
            "root_fixed_latency_us": self.root_fixed_latency_us,
            "inter_root_bandwidth_gbps":
                self.inter_root_bandwidth_gbps,
            "inter_root_fixed_latency_us":
                self.inter_root_fixed_latency_us,
            "p2p_mode": self.p2p_mode,
        }


def collective_runtime_ns(
        *, topology: HBFPCIeTopology, tp_size: int,
        payload_bytes: int, rank_multiplier: float,
        route: str) -> int:
    """Return a card/root/inter-root roofline for one collective."""

    tp = _positive_int("tp_size", tp_size)
    payload = _nonnegative_int("payload_bytes", payload_bytes)
    multiplier = _nonnegative_float(
        "rank_multiplier", rank_multiplier)
    if tp == 1 or payload == 0 or multiplier == 0:
        return 0
    if route not in {ROOT_LOCAL_ROUTE, CROSS_ROOT_ROUTE}:
        raise HBFPCIeTopologyError(
            f"unknown collective route {route!r}")
    per_rank_bytes = multiplier * payload
    active_per_root = min(tp, topology.cards_per_root)
    data_seconds = max(
        per_rank_bytes / (topology.card_bandwidth_gbps * 1e9),
        (
            active_per_root * per_rank_bytes
            / (topology.root_bandwidth_gbps * 1e9)
        ),
    )
    fixed_us = max(
        topology.card_fixed_latency_us,
        topology.root_fixed_latency_us,
    )
    if route == CROSS_ROOT_ROUTE:
        if topology.p2p_mode != CROSS_ROOT_P2P_MODE:
            raise HBFPCIeTopologyError(
                "collective requires disabled cross-root PCIe P2P")
        # A ring spanning two roots has two cut edges.  This is an explicit
        # analytical lower bound, not a native ASTRA collective.
        data_seconds = max(
            data_seconds,
            (
                2.0 * per_rank_bytes
                / (topology.inter_root_bandwidth_gbps * 1e9)
            ),
        )
        fixed_us = max(
            fixed_us,
            topology.inter_root_fixed_latency_us,
        )
    return int(math.ceil(data_seconds * 1e9 + fixed_us * 1e3))


__all__ = [
    "CROSS_ROOT_P2P_MODE",
    "CROSS_ROOT_ROUTE",
    "HBFPCIeTopology",
    "HBFPCIeTopologyError",
    "LEGACY_ISOLATED_RESOURCE_MODE",
    "PAIR_COLLECTIVE_TYPES",
    "ROOT_LOCAL_ROUTE",
    "SAME_ROOT_ONLY_P2P_MODE",
    "SHARED_RESOURCE_MODE",
    "collective_runtime_ns",
]

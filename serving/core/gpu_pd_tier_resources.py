"""Exact node-local transfer stages for a tiered P4D4 H100 server."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from .gpu_pd_latency import P4D4GPUHardware
from .hbf_full_model_lifecycle import ResourceCalendar


@dataclass(frozen=True)
class ResourceDemand:
    resource: str
    service_ns: int
    byte_count: int


@dataclass(frozen=True)
class TierTransferStage:
    kind: str
    direction: str
    token_count: int
    block_rounded: bool
    bytes_per_rank: int
    aggregate_bytes: int
    latency_ns: int
    demands: tuple[ResourceDemand, ...]

    @property
    def resources(self) -> tuple[str, ...]:
        return tuple(demand.resource for demand in self.demands)

    def calendar_demands(self) -> Mapping[str, tuple[int, int]]:
        return {
            demand.resource: (
                demand.service_ns,
                demand.byte_count,
            )
            for demand in self.demands
        }

    def reserve(
            self, calendar: ResourceCalendar, *, ready_ns: int,
            job_id: int, namespace: str,
    ) -> tuple[int, int]:
        return calendar.reserve_parallel(
            arrival_ns=ready_ns,
            job_id=job_id,
            kind=self.kind,
            namespace=namespace,
            demands=self.calendar_demands(),
        )


class TierNodeResources:
    """Physical transfer/capacity contract for one eight-H100 node."""

    def __init__(
            self, *, hardware: P4D4GPUHardware,
            node_id: int) -> None:
        hardware.validate()
        if (
            isinstance(node_id, bool)
            or not isinstance(node_id, int)
            or node_id < 0
        ):
            raise ValueError("node_id must be a non-negative integer")
        self.hardware = hardware
        self.node_id = node_id

    @property
    def hbm_kv_capacity_bytes_per_rank(self) -> int:
        return self.hardware.usable_hbm_bytes_per_rank

    @property
    def cpu_capacity_bytes(self) -> int:
        return self.hardware.cpu_memory_capacity_bytes

    @property
    def ssd_capacity_bytes(self) -> int:
        return self.hardware.ssd_capacity_bytes

    @property
    def max_hbm_kv_blocks_per_rank(self) -> int:
        block_bytes = self.hardware.kv_capacity_bytes_per_rank(
            self.hardware.block_size_tokens)
        return self.hbm_kv_capacity_bytes_per_rank // block_bytes

    @property
    def max_hbm_kv_tokens_per_rank(self) -> int:
        return (
            self.max_hbm_kv_blocks_per_rank
            * self.hardware.block_size_tokens
        )

    @staticmethod
    def _validate_tokens(token_count: int) -> None:
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            raise ValueError(
                "transfer token_count must be a non-negative integer")

    def _bytes(
            self, token_count: int, *,
            block_rounded: bool) -> tuple[int, int]:
        self._validate_tokens(token_count)
        if block_rounded:
            per_rank = self.hardware.kv_capacity_bytes_per_rank(
                token_count)
        else:
            per_rank = (
                token_count
                * self.hardware.kv_bytes_per_token_per_rank
            )
        return per_rank, per_rank * self.hardware.tp_size

    @staticmethod
    def _ns(seconds: float) -> int:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("transfer latency must be finite/non-negative")
        return int(math.ceil(seconds * 1e9))

    def _gpu_root(self, gpu_role: str) -> int:
        if gpu_role == "p":
            return 0
        if gpu_role == "d":
            return 1
        raise ValueError("gpu_role must be 'p' or 'd'")

    def gpu_cpu_stage(
            self, token_count: int, *,
            gpu_role: str,
            direction: str,
            block_rounded: bool = True) -> TierTransferStage:
        """Build one CPU<->TP4 HBM stage over four 50-GB/s lanes."""

        if direction not in {"cpu_to_gpu", "gpu_to_cpu"}:
            raise ValueError(
                "GPU/CPU direction must be cpu_to_gpu or gpu_to_cpu")
        root_id = self._gpu_root(gpu_role)
        per_rank, aggregate = self._bytes(
            token_count, block_rounded=block_rounded)
        if aggregate == 0:
            latency_ns = 0
        else:
            seconds = (
                self.hardware.cpu_transfer_latency_us * 1e-6
                + max(
                    per_rank
                    / (
                        self.hardware.pcie_bandwidth_gbps_per_gpu
                        * 1e9
                    ),
                    aggregate
                    / (
                        self.hardware.pcie_root_bandwidth_gbps
                        * 1e9
                    ),
                    aggregate
                    / (
                        self.hardware.cpu_memory_bandwidth_gbps
                        * 1e9
                    ),
                )
            )
            latency_ns = self._ns(seconds)
        demands = [
            ResourceDemand(
                f"gpu-node-{self.node_id}-{gpu_role}-pcie-rank-{rank}",
                latency_ns,
                per_rank,
            )
            for rank in range(self.hardware.tp_size)
        ]
        demands.extend((
            ResourceDemand(
                f"gpu-node-{self.node_id}-pcie-root-{root_id}",
                latency_ns,
                aggregate,
            ),
            ResourceDemand(
                f"gpu-node-{self.node_id}-cpu-dram",
                latency_ns,
                aggregate,
            ),
        ))
        return TierTransferStage(
            kind=f"{gpu_role}-{direction}",
            direction=direction,
            token_count=token_count,
            block_rounded=block_rounded,
            bytes_per_rank=per_rank,
            aggregate_bytes=aggregate,
            latency_ns=latency_ns,
            demands=tuple(demands),
        )

    def ssd_stage(
            self, token_count: int, *,
            direction: str) -> TierTransferStage:
        """Build one aggregate CPU<->SSD stage for a block-rounded object."""

        if direction not in {"ssd_to_cpu", "cpu_to_ssd"}:
            raise ValueError(
                "SSD direction must be ssd_to_cpu or cpu_to_ssd")
        per_rank, aggregate = self._bytes(
            token_count, block_rounded=True)
        if aggregate == 0:
            latency_ns = 0
        elif direction == "ssd_to_cpu":
            seconds = (
                self.hardware.ssd_read_latency_us * 1e-6
                + max(
                    aggregate
                    / (
                        self.hardware.ssd_read_bandwidth_gbps
                        * 1e9
                    ),
                    aggregate
                    / (
                        self.hardware.cpu_memory_bandwidth_gbps
                        * 1e9
                    ),
                )
            )
            latency_ns = self._ns(seconds)
        else:
            seconds = (
                self.hardware.ssd_write_latency_us * 1e-6
                + max(
                    aggregate
                    / (
                        self.hardware.ssd_write_bandwidth_gbps
                        * 1e9
                    ),
                    aggregate
                    / (
                        self.hardware.cpu_memory_bandwidth_gbps
                        * 1e9
                    ),
                )
            )
            latency_ns = self._ns(seconds)
        queue_name = (
            "ssd-read" if direction == "ssd_to_cpu"
            else "ssd-write"
        )
        return TierTransferStage(
            kind=direction.replace("_", "-"),
            direction=direction,
            token_count=token_count,
            block_rounded=True,
            bytes_per_rank=per_rank,
            aggregate_bytes=aggregate,
            latency_ns=latency_ns,
            demands=(
                ResourceDemand(
                    f"gpu-node-{self.node_id}-{queue_name}",
                    latency_ns,
                    aggregate,
                ),
                ResourceDemand(
                    f"gpu-node-{self.node_id}-cpu-dram",
                    latency_ns,
                    aggregate,
                ),
            ),
        )

    def peer_stage(
            self, token_count: int, *,
            direction: str,
            block_rounded: bool = False) -> TierTransferStage:
        """Build a same-node TP4 P<->D copy over pairwise 450-GB/s lanes."""

        if direction not in {"p_to_d", "d_to_p"}:
            raise ValueError(
                "peer direction must be p_to_d or d_to_p")
        per_rank, aggregate = self._bytes(
            token_count, block_rounded=block_rounded)
        if per_rank == 0:
            latency_ns = 0
        else:
            latency_ns = self._ns(
                self.hardware.pd_peer_fixed_latency_us * 1e-6
                + per_rank
                / (
                    self.hardware.nvlink_bandwidth_gbps_per_gpu
                    * 1e9
                )
            )
        demands = [
            ResourceDemand(
                f"gpu-node-{self.node_id}-pd-peer-rank-{rank}",
                latency_ns,
                per_rank,
            )
            for rank in range(self.hardware.tp_size)
        ]
        demands.append(ResourceDemand(
            f"gpu-node-{self.node_id}-pd-fabric",
            latency_ns,
            aggregate,
        ))
        return TierTransferStage(
            kind=direction.replace("_", "-"),
            direction=direction,
            token_count=token_count,
            block_rounded=block_rounded,
            bytes_per_rank=per_rank,
            aggregate_bytes=aggregate,
            latency_ns=latency_ns,
            demands=tuple(demands),
        )

    def metadata(self) -> Mapping[str, Any]:
        return {
            "node_id": self.node_id,
            "hardware": asdict(self.hardware),
            "p_pcie_root": 0,
            "d_pcie_root": 1,
            "ssd_restore_pipeline": (
                "ssd_to_cpu_then_cpu_to_p"),
            "capacity_units": {
                "hbm": "bytes_per_rank",
                "cpu": "aggregate_node_bytes",
                "ssd": "aggregate_node_bytes",
            },
        }


__all__ = [
    "ResourceDemand",
    "TierNodeResources",
    "TierTransferStage",
]

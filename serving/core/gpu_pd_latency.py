"""Analytical H100 latency and P/D handoff model for one 4P4D node."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .h100_kernel_calibrated_prompt import (
    BF16_BYTES,
    H100_DENSE_BF16_FLOPS_PER_SECOND,
    H100_HBM_BYTES_PER_SECOND,
    KernelWork,
    QWEN_EXPERTS,
    QWEN_HIDDEN_SIZE,
    QWEN_LAYERS,
    predict_kernel_seconds,
)
from .hbf_full_model_latency import (
    COLLECTIVE_FIXED_LATENCY_SEMANTICS,
    HBFModelBatchShape,
    qwen_logical_kv_bytes_per_token,
    qwen_model_weight_bytes_per_rank,
)
from .online_latency_model import (
    H100_QWEN3_TP4_KERNEL_CALIBRATED,
    OnlineBatchShape,
    build_online_latency_model,
)


SCHEMA_VERSION = 1
MODEL_CONFIG = (
    "configs/model/Qwen/Qwen3-30B-A3B-Instruct-2507.json")
P4D4_MODEL_LAYER_COUNT = QWEN_LAYERS


def _strict_fields(
        raw: Mapping[str, object], allowed: set[str], scope: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown {scope} field(s): {unknown}")


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _nonnegative_float(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be non-negative and finite")
    return float(value)


@dataclass(frozen=True)
class P4D4GPUHardware:
    """Physical and lower-tier inputs for one eight-H100 server."""

    gpu_count: int = 8
    prefill_gpu_count: int = 4
    decode_gpu_count: int = 4
    tp_size: int = 4
    hbm_capacity_bytes_per_gpu: int = 80_000_000_000
    hbm_bandwidth_gbps_per_gpu: float = 3_350.0
    gpu_peak_tflops_per_gpu: float = 989.5
    runtime_reserve_bytes_per_gpu: int = 19_893_012_480
    nvlink_bandwidth_gbps_per_gpu: float = 450.0
    collective_fixed_latency_us: float = 3.0
    pd_peer_fixed_latency_us: float = 1.0
    pcie_bandwidth_gbps_per_gpu: float = 50.0
    pcie_root_count: int = 2
    gpus_per_pcie_root: int = 4
    pcie_root_bandwidth_gbps: float = 200.0
    cpu_memory_capacity_bytes: int = 512_000_000_000
    cpu_memory_bandwidth_gbps: float = 200.0
    cpu_transfer_latency_us: float = 5.0
    ssd_capacity_bytes_per_device: int = 3_840_000_000_000
    ssd_device_count: int = 8
    ssd_read_bandwidth_gbps: float = 55.2
    ssd_write_bandwidth_gbps: float = 33.6
    ssd_read_latency_us: float = 20.0
    ssd_write_latency_us: float = 20.0
    block_size_tokens: int = 16

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "P4D4GPUHardware":
        _strict_fields(raw, set(cls.__dataclass_fields__), "GPU hardware")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self) -> None:
        for name in (
            "gpu_count",
            "prefill_gpu_count",
            "decode_gpu_count",
            "tp_size",
            "hbm_capacity_bytes_per_gpu",
            "runtime_reserve_bytes_per_gpu",
            "pcie_root_count",
            "gpus_per_pcie_root",
            "cpu_memory_capacity_bytes",
            "ssd_capacity_bytes_per_device",
            "ssd_device_count",
            "block_size_tokens",
        ):
            _positive_int(f"hardware.{name}", getattr(self, name))
        for name in (
            "hbm_bandwidth_gbps_per_gpu",
            "gpu_peak_tflops_per_gpu",
            "nvlink_bandwidth_gbps_per_gpu",
            "pcie_bandwidth_gbps_per_gpu",
            "pcie_root_bandwidth_gbps",
            "cpu_memory_bandwidth_gbps",
            "ssd_read_bandwidth_gbps",
            "ssd_write_bandwidth_gbps",
        ):
            _positive_float(f"hardware.{name}", getattr(self, name))
        for name in (
            "collective_fixed_latency_us",
            "pd_peer_fixed_latency_us",
            "cpu_transfer_latency_us",
            "ssd_read_latency_us",
            "ssd_write_latency_us",
        ):
            _nonnegative_float(f"hardware.{name}", getattr(self, name))
        if self.prefill_gpu_count + self.decode_gpu_count != self.gpu_count:
            raise ValueError(
                "P4D4 prefill and decode partitions must use every GPU")
        if (
            self.prefill_gpu_count != self.tp_size
            or self.decode_gpu_count != self.tp_size
        ):
            raise ValueError(
                "the P4D4 model requires one TP4 P and one TP4 D group")
        if self.pcie_root_count * self.gpus_per_pcie_root != self.gpu_count:
            raise ValueError(
                "PCIe roots must cover every GPU exactly once")
        if self.usable_hbm_bytes_per_rank <= 0:
            raise ValueError(
                "model weights and workspace do not fit GPU HBM")
        if not math.isclose(
            self.gpu_peak_tflops_per_gpu,
            H100_DENSE_BF16_FLOPS_PER_SECOND / 1e12,
        ):
            raise ValueError(
                "GPU compute must match the pinned H100 calibration")
        if not math.isclose(
            self.hbm_bandwidth_gbps_per_gpu,
            H100_HBM_BYTES_PER_SECOND / 1e9,
        ):
            raise ValueError(
                "GPU HBM bandwidth must match the pinned H100 calibration")

    @property
    def model_weight_bytes_per_rank(self) -> int:
        return qwen_model_weight_bytes_per_rank(self.tp_size)

    @property
    def usable_hbm_bytes_per_rank(self) -> int:
        return (
            self.hbm_capacity_bytes_per_gpu
            - self.model_weight_bytes_per_rank
            - self.runtime_reserve_bytes_per_gpu
        )

    @property
    def kv_bytes_per_token_per_rank(self) -> int:
        logical = qwen_logical_kv_bytes_per_token()
        if logical % self.tp_size:
            raise ValueError("logical KV does not divide TP ranks exactly")
        return logical // self.tp_size

    def kv_capacity_bytes_per_rank(self, token_count: int) -> int:
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            raise ValueError(
                "KV token count must be a non-negative integer")
        blocks = (
            (token_count + self.block_size_tokens - 1)
            // self.block_size_tokens
        )
        return (
            blocks
            * self.block_size_tokens
            * self.kv_bytes_per_token_per_rank
        )

    @property
    def ssd_capacity_bytes(self) -> int:
        return (
            self.ssd_capacity_bytes_per_device
            * self.ssd_device_count
        )


@dataclass(frozen=True)
class PDHandoffLatency:
    token_count: int
    bytes_per_rank: int
    aggregate_bytes: int
    latency_ns: int


@dataclass(frozen=True)
class GPUBatchLatency:
    total_ns: int
    comp_ns: int
    provider_comp_ns: int
    router_ns: int
    collective_ns: int
    tp_allreduce_ns: int
    ep_allgather_ns: int
    ep_reduce_scatter_ns: int
    collective_bytes_per_rank: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class GPUBatchPhaseLatency:
    """Exact full-batch timing around identical transformer layers."""

    prologue_ns: int
    layer_ns: int
    layer_count: int
    epilogue_ns: int
    total_ns: int

    def layer_start_offset_ns(self, layer_index: int) -> int:
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 0 <= layer_index < self.layer_count
        ):
            raise ValueError(
                "layer_index must identify a modeled transformer layer")
        return self.prologue_ns + layer_index * self.layer_ns


class P4D4LatencyModel:
    """Full-model TP4 GPU batches plus pairwise P-to-D KV copies."""

    def __init__(
            self, *, repo_root: Path, hardware: P4D4GPUHardware,
            band: str = "central") -> None:
        hardware.validate()
        self.hardware = hardware
        self.band = band
        model_config = repo_root / MODEL_CONFIG
        if not model_config.is_file():
            raise FileNotFoundError(
                f"missing target model config: {model_config}")
        digest = hashlib.sha256(model_config.read_bytes()).hexdigest()
        self._provider = build_online_latency_model(
            H100_QWEN3_TP4_KERNEL_CALIBRATED,
            str(repo_root.resolve()),
            digest,
            band,
        )

    @property
    def provider(self):
        return self._provider

    @staticmethod
    def _online_shape(shape: HBFModelBatchShape) -> OnlineBatchShape:
        shape.validate()
        return OnlineBatchShape(
            total_tokens=shape.total_tokens,
            prefill_q=shape.prefill_q,
            prefill_k=tuple(
                hbf_k + lpddr_k
                for hbf_k, lpddr_k in zip(
                    shape.prefill_hbf_k,
                    shape.prefill_lpddr_k,
                )
            ),
            decode_k=shape.decode_k,
            lm_head_sequences=shape.lm_head_sequences,
        )

    def _collective_ns(
            self, payload_bytes: int, rank_multiplier: float) -> int:
        if payload_bytes <= 0 or rank_multiplier <= 0:
            return 0
        seconds = (
            self.hardware.collective_fixed_latency_us * 1e-6
            + rank_multiplier
            * payload_bytes
            / (
                self.hardware.nvlink_bandwidth_gbps_per_gpu
                * 1e9
            )
        )
        return int(math.ceil(seconds * 1e9))

    def _router_ns(self, total_tokens: int) -> int:
        work = KernelWork(
            flops=(
                2.0
                * total_tokens
                * QWEN_HIDDEN_SIZE
                * QWEN_EXPERTS
            ),
            bytes=BF16_BYTES * (
                total_tokens * QWEN_HIDDEN_SIZE
                + QWEN_HIDDEN_SIZE * QWEN_EXPERTS
                + total_tokens * QWEN_EXPERTS
            ),
        )
        fit = self._provider.calibration.fits["router"]
        seconds = predict_kernel_seconds(
            fit, work,
            band=self.band,
            semantics=self._provider.semantics,
        )
        return int(math.ceil(seconds * 1e9))

    @lru_cache(maxsize=262_144)
    def batch_latency(
            self, shape: HBFModelBatchShape) -> GPUBatchLatency:
        online_shape = self._online_shape(shape)
        provider_comp_ns = self._provider.batch_kernel_critical_path_ns(
            online_shape)
        router_ns = QWEN_LAYERS * self._router_ns(
            shape.total_tokens)
        comp_ns = provider_comp_ns + router_ns

        tp = self.hardware.tp_size
        hidden_payload = (
            shape.total_tokens
            * QWEN_HIDDEN_SIZE
            * BF16_BYTES
        )
        dispatch_local_chunk = (
            max(1, shape.total_tokens // tp)
            * (QWEN_HIDDEN_SIZE + QWEN_EXPERTS)
            * BF16_BYTES
        )
        allreduce_per_layer = self._collective_ns(
            hidden_payload,
            2.0 * (tp - 1) / tp,
        )
        allgather_per_layer = self._collective_ns(
            dispatch_local_chunk,
            tp - 1,
        )
        reduce_scatter_per_layer = self._collective_ns(
            hidden_payload,
            (tp - 1) / tp,
        )
        tp_allreduce_ns = QWEN_LAYERS * allreduce_per_layer
        ep_allgather_ns = QWEN_LAYERS * allgather_per_layer
        ep_reduce_scatter_ns = (
            QWEN_LAYERS * reduce_scatter_per_layer)
        collective_ns = (
            tp_allreduce_ns
            + ep_allgather_ns
            + ep_reduce_scatter_ns
        )
        collective_bytes_per_rank = int(math.ceil(
            QWEN_LAYERS
            * (
                2.0 * (tp - 1) / tp * hidden_payload
                + (tp - 1) * dispatch_local_chunk
                + (tp - 1) / tp * hidden_payload
            )
        ))
        return GPUBatchLatency(
            total_ns=comp_ns + collective_ns,
            comp_ns=comp_ns,
            provider_comp_ns=provider_comp_ns,
            router_ns=router_ns,
            collective_ns=collective_ns,
            tp_allreduce_ns=tp_allreduce_ns,
            ep_allgather_ns=ep_allgather_ns,
            ep_reduce_scatter_ns=ep_reduce_scatter_ns,
            collective_bytes_per_rank=collective_bytes_per_rank,
        )

    @lru_cache(maxsize=262_144)
    def batch_phase_latency(
            self, shape: HBFModelBatchShape) -> GPUBatchPhaseLatency:
        """Decompose a batch without changing its aggregate reservation."""

        latency = self.batch_latency(shape)
        provider_phases = self._provider.batch_kernel_phases(
            self._online_shape(shape))
        layer_count = provider_phases.layer_count
        if layer_count != P4D4_MODEL_LAYER_COUNT:
            raise AssertionError("provider transformer-layer count changed")
        if provider_phases.total_ns != latency.provider_comp_ns:
            raise AssertionError(
                "provider phase decomposition changed aggregate compute")
        layer_scaled_fields = (
            latency.router_ns,
            latency.tp_allreduce_ns,
            latency.ep_allgather_ns,
            latency.ep_reduce_scatter_ns,
        )
        if any(value % layer_count for value in layer_scaled_fields):
            raise AssertionError(
                "per-layer GPU timing no longer divides exactly")
        layer_ns = (
            provider_phases.layer_ns
            + sum(
                value // layer_count
                for value in layer_scaled_fields
            )
        )
        phases = GPUBatchPhaseLatency(
            prologue_ns=provider_phases.prologue_ns,
            layer_ns=layer_ns,
            layer_count=layer_count,
            epilogue_ns=provider_phases.epilogue_ns,
            total_ns=latency.total_ns,
        )
        if (
            phases.prologue_ns
            + phases.layer_count * phases.layer_ns
            + phases.epilogue_ns
            != phases.total_ns
        ):
            raise AssertionError(
                "GPU phase decomposition changed aggregate latency")
        return phases

    def handoff_latency(self, token_count: int) -> PDHandoffLatency:
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            raise ValueError(
                "handoff token_count must be a non-negative integer")
        bytes_per_rank = (
            token_count * self.hardware.kv_bytes_per_token_per_rank)
        if bytes_per_rank == 0:
            latency_ns = 0
        else:
            seconds = (
                self.hardware.pd_peer_fixed_latency_us * 1e-6
                + bytes_per_rank
                / (
                    self.hardware.nvlink_bandwidth_gbps_per_gpu
                    * 1e9
                )
            )
            latency_ns = int(math.ceil(seconds * 1e9))
        return PDHandoffLatency(
            token_count=token_count,
            bytes_per_rank=bytes_per_rank,
            aggregate_bytes=bytes_per_rank * self.hardware.tp_size,
            latency_ns=latency_ns,
        )

    def metadata(self) -> Mapping[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "topology": "one_server_4p4d",
            "hardware": asdict(self.hardware),
            "full_model_provider": self._provider.metadata(),
            "collective_fixed_latency_semantics": (
                COLLECTIVE_FIXED_LATENCY_SEMANTICS),
            "ttft_boundary": "P sampler completion before P-to-D handoff",
            "handoff_scope": "materialized KV only",
        }


def load_p4d4_gpu_config(path: Path) -> P4D4GPUHardware:
    with path.open("r", encoding="utf-8") as source:
        raw = json.load(source)
    if not isinstance(raw, dict):
        raise ValueError("P4D4 GPU config must be a JSON object")
    _strict_fields(raw, {"schema_version", "hardware"}, "top-level")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SCHEMA_VERSION}")
    hardware = raw.get("hardware")
    if not isinstance(hardware, dict):
        raise ValueError("hardware must be an object")
    return P4D4GPUHardware.from_dict(hardware)


__all__ = [
    "GPUBatchLatency",
    "GPUBatchPhaseLatency",
    "P4D4_MODEL_LAYER_COUNT",
    "P4D4GPUHardware",
    "P4D4LatencyModel",
    "PDHandoffLatency",
    "load_p4d4_gpu_config",
]

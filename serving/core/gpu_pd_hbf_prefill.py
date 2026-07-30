"""Heterogeneous one-server P4D4: HBF+LPDDR prefill cards, H100 decode.

The prefill role is four HBF-GPU cards (one TP4 replica of the eight-card
full-model HBF server): model weights and committed session KV are read
from HBF at HBM-class bandwidth, activations and active-turn KV live in
LPDDR.  The decode role keeps four H100s with HBM.  Completed KV's durable
home is the P-side HBF; the D copy is produced by the existing layerwise
P-to-D NVLink handoff and is a cache, not the home.

The system reuses :class:`FiniteHBMTieredP4D4Node` unchanged.  Two seams
make it heterogeneous:

* the pool's P-stage latency model is swapped for
  :class:`HBFPrefillLatencyAdapter` (D batches and the handoff keep the
  calibrated GPU model), and
* the tier lifecycle's CPU tier is reinterpreted as the HBF home.  A
  "restore" from this tier moves no bytes -- the KV is already where
  prefill executes, and the physical read is charged inside the P-batch's
  HBF roofline -- so the tier's virtual bandwidth is set high enough that
  its transfer stages are ownership updates, not copies.  Demotion D-to-HBF
  is likewise free because prefill already wrote the KV into HBF.  SSD
  spill and SSD restore keep real SSD bandwidths and byte accounting.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from .gpu_pd_latency import (
    GPUBatchLatency,
    GPUBatchPhaseLatency,
    P4D4GPUHardware,
    P4D4LatencyModel,
)
from .gpu_pd_single_system import (
    SINGLE_GPU_NODE_ID,
    _SingleP4D4CausalSystem,
)
from .gpu_pd_tier_lifecycle import (
    RESTORE_EXECUTION_LAYERWISE,
    SUPPORTED_TIER_POLICIES,
)
from .gpu_pd_tiered_node import FiniteHBMTieredP4D4Node
from .hbf_full_model_latency import (
    FullModelHBFLatencyModel,
    HBFModelBatchShape,
    HBFParallelLayout,
    HBFServerHardware,
)


# The prefill role is one TP4 replica: four physical HBF cards.
HBF_PREFILL_LAYOUT = "tp4"
HBF_PREFILL_CARD_COUNT = 4

# The HBF-home tier reuses the lifecycle's CPU-tier machinery.  Transfers
# against this tier are ownership updates (the durable copy is already in
# HBF), so the virtual bandwidth only has to make their calendar occupancy
# negligible, never zero (zero-duration transfers would violate calendar
# invariants).
VIRTUAL_HBF_TIER_BANDWIDTH_GBPS = 1.0e6
VIRTUAL_HBF_TIER_LATENCY_US = 0.0

HETERO_MODE_NAME = "single_hetero_hbf_prefill_p4d4"


class HBFPrefillLatencyAdapter:
    """Present a FullModelHBFLatencyModel as a pool P-stage latency model.

    The adapter reshapes :class:`HBFModelBatchLatency` into the
    :class:`GPUBatchLatency` / :class:`GPUBatchPhaseLatency` contract the
    pool consumes, asserting the decomposition reproduces the aggregate
    exactly so layerwise handoff overlap and restore gating stay lossless.
    """

    def __init__(self, model: FullModelHBFLatencyModel) -> None:
        if model.layout.is_context_striped:
            raise ValueError(
                "the P4 prefill adapter requires a non-striped layout")
        self.model = model

    @lru_cache(maxsize=262_144)
    def batch_latency(self, shape: HBFModelBatchShape) -> GPUBatchLatency:
        raw = self.model.batch_latency(shape)
        if (
            raw.pair_query_exchange_ns
            or raw.pair_softmax_partial_exchange_ns
            or raw.pair_attention_merge_ns
        ):
            raise AssertionError(
                "non-striped HBF layout produced pair-attention terms")
        comp_ns = (
            raw.embedding_ns
            + raw.dense_ns
            + raw.attention_ns
            + raw.router_ns
            + raw.moe_ns
            + raw.final_ns
        )
        if comp_ns + raw.collective_ns != raw.total_ns:
            raise AssertionError(
                "HBF batch latency decomposition does not reproduce total")
        return GPUBatchLatency(
            total_ns=raw.total_ns,
            comp_ns=comp_ns,
            provider_comp_ns=comp_ns - raw.router_ns,
            router_ns=raw.router_ns,
            collective_ns=raw.collective_ns,
            tp_allreduce_ns=raw.tp_allreduce_ns,
            ep_allgather_ns=raw.ep_allgather_ns,
            ep_reduce_scatter_ns=raw.ep_reduce_scatter_ns,
            collective_bytes_per_rank=raw.collective_bytes_per_rank,
        )

    @lru_cache(maxsize=262_144)
    def batch_phase_latency(
            self, shape: HBFModelBatchShape) -> GPUBatchPhaseLatency:
        raw = self.model.batch_latency(shape)
        latency = self.batch_latency(shape)
        from .h100_kernel_calibrated_prompt import QWEN_LAYERS
        layer_scaled = (
            raw.dense_ns,
            raw.attention_ns,
            raw.router_ns,
            raw.moe_ns,
            raw.tp_allreduce_ns,
            raw.ep_allgather_ns,
            raw.ep_reduce_scatter_ns,
        )
        if any(value % QWEN_LAYERS for value in layer_scaled):
            raise AssertionError(
                "per-layer HBF timing no longer divides exactly")
        layer_ns = sum(value // QWEN_LAYERS for value in layer_scaled)
        phases = GPUBatchPhaseLatency(
            prologue_ns=raw.embedding_ns,
            layer_ns=layer_ns,
            layer_count=QWEN_LAYERS,
            epilogue_ns=raw.final_ns,
            total_ns=latency.total_ns,
        )
        if (
            phases.prologue_ns
            + phases.layer_count * phases.layer_ns
            + phases.epilogue_ns
            != phases.total_ns
        ):
            raise AssertionError(
                "HBF phase decomposition changed aggregate latency")
        return phases

    def metadata(self) -> Mapping[str, Any]:
        return {
            "kind": "hbf_prefill_p4_adapter",
            "layout": self.model.layout.key,
            "tp_size": self.model.layout.tp_size,
            "cards_in_replica": self.model.layout.tp_size,
            "hardware": {
                "hbf_read_bandwidth_gbps_per_card": (
                    self.model.hardware.hbf_read_bandwidth_gbps_per_card),
                "hbf_write_bandwidth_gbps_per_card": (
                    self.model.hardware.hbf_write_bandwidth_gbps_per_card),
                "lpddr_bandwidth_gbps_per_card": (
                    self.model.hardware.lpddr_bandwidth_gbps_per_card),
                "npu_peak_tflops_per_card": (
                    self.model.hardware.npu_peak_tflops_per_card),
            },
            "weight_residency": "hbf",
            "active_turn_kv_residency": "lpddr",
        }


def hbf_home_capacity_bytes(hbf_hardware: HBFServerHardware) -> int:
    """Return the P node's HBF home capacity: one TP4 replica's cards."""

    return (
        HBF_PREFILL_CARD_COUNT
        * hbf_hardware.hbf_capacity_bytes_per_card
    )


def virtual_hbf_tier_hardware(
        base: P4D4GPUHardware,
        hbf_hardware: HBFServerHardware) -> P4D4GPUHardware:
    """Reinterpret the CPU tier of ``base`` as the P-side HBF home."""

    value = replace(
        base,
        cpu_memory_capacity_bytes=hbf_home_capacity_bytes(hbf_hardware),
        cpu_memory_bandwidth_gbps=VIRTUAL_HBF_TIER_BANDWIDTH_GBPS,
        cpu_transfer_latency_us=VIRTUAL_HBF_TIER_LATENCY_US,
    )
    value.validate()
    return value


def load_hetero_hbf_prefill_config(
        path: Path) -> tuple[P4D4GPUHardware, HBFServerHardware, str]:
    """Load the heterogeneous server config.

    The JSON carries a ``gpu`` section (P4D4 substrate: D-side HBM, NVLink,
    PCIe, CPU-staging irrelevant fields, and the shared SSDs) plus an
    ``hbf`` section (the prefill cards) and the fixed ``hbf_layout``.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("gpu", "hbf"):
        if key not in raw or not isinstance(raw[key], Mapping):
            raise ValueError(
                f"hetero config {path} must carry a {key!r} object")
    layout_key = raw.get("hbf_layout", HBF_PREFILL_LAYOUT)
    if layout_key != HBF_PREFILL_LAYOUT:
        raise ValueError(
            "the heterogeneous P4D4 system models one TP4 HBF replica; "
            f"got hbf_layout={layout_key!r}")
    gpu = P4D4GPUHardware.from_dict(raw["gpu"])
    hbf = HBFServerHardware.from_dict(raw["hbf"])
    return gpu, hbf, layout_key


class SingleHBFPrefillTieredSystem(_SingleP4D4CausalSystem):
    """One heterogeneous server: HBF+LPDDR prefill role, H100 decode role."""

    def __init__(
            self, *, repo_root: Path,
            hardware: P4D4GPUHardware,
            hbf_hardware: Optional[HBFServerHardware] = None,
            policy: str = "cpu_ssd",
            p_capacity_bytes_per_rank: Optional[int] = None,
            d_capacity_bytes_per_rank: Optional[int] = None,
            ssd_capacity_bytes: Optional[int] = None,
            max_num_batched_tokens: int = 8_192,
            max_num_seqs: int = 128,
            p_max_num_seqs: Optional[int] = None,
            d_max_num_seqs: Optional[int] = None,
            max_prefill_chunk_tokens: int = 4_096,
            band: str = "central",
            restore_execution_mode: str = RESTORE_EXECUTION_LAYERWISE,
            validate_every_event: bool = True) -> None:
        if policy not in SUPPORTED_TIER_POLICIES:
            raise ValueError(f"unsupported tier policy {policy!r}")
        if policy != "cpu_ssd":
            raise ValueError(
                "the HBF-home tier reuses the CPU-tier machinery and "
                "therefore requires the cpu_ssd policy")
        resolved_hbf = (
            hbf_hardware if hbf_hardware is not None
            else HBFServerHardware()
        )
        resolved_hbf.validate()
        layout = HBFParallelLayout.for_key(HBF_PREFILL_LAYOUT)
        layout.validate(resolved_hbf.card_count)
        tier_hardware = virtual_hbf_tier_hardware(hardware, resolved_hbf)
        if p_capacity_bytes_per_rank is None:
            # The P ledger tracks per-active-turn KV visibility on the
            # prefill cards.  Prefix KV is HBF-resident and fresh KV is
            # bounded by the token budget, so the per-card HBF capacity is
            # the honest non-binding bound.
            p_capacity_bytes_per_rank = (
                resolved_hbf.hbf_capacity_bytes_per_card)

        def _p_latency_model_factory(gpu_model: P4D4LatencyModel):
            return HBFPrefillLatencyAdapter(
                FullModelHBFLatencyModel(
                    base_provider=gpu_model.provider,
                    hardware=resolved_hbf,
                    layout=layout,
                ))

        node = FiniteHBMTieredP4D4Node(
            repo_root=repo_root,
            hardware=tier_hardware,
            node_id=SINGLE_GPU_NODE_ID,
            policy=policy,
            p_capacity_bytes_per_rank=p_capacity_bytes_per_rank,
            d_capacity_bytes_per_rank=d_capacity_bytes_per_rank,
            cpu_capacity_bytes=hbf_home_capacity_bytes(resolved_hbf),
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
            p_latency_model_factory=_p_latency_model_factory,
        )
        self.policy = policy
        self.restore_execution_mode = restore_execution_mode
        self.hbf_hardware = resolved_hbf
        self.hbf_layout = layout
        super().__init__(
            repo_root=repo_root,
            hardware=tier_hardware,
            node=node,
            validate_every_event=validate_every_event,
            mode=HETERO_MODE_NAME,
        )

    def _system_specific_report(self) -> Mapping[str, Any]:
        if not isinstance(self.node, FiniteHBMTieredP4D4Node):
            raise AssertionError("hetero system owns the wrong node type")
        return {
            "policy": self.policy,
            "restore_execution_mode": self.restore_execution_mode,
            "prefill_role": {
                "cards": HBF_PREFILL_CARD_COUNT,
                "layout": self.hbf_layout.key,
                "hbf_capacity_bytes": hbf_home_capacity_bytes(
                    self.hbf_hardware),
                "hbf_read_bandwidth_gbps_per_card": (
                    self.hbf_hardware.hbf_read_bandwidth_gbps_per_card),
                "hbf_write_bandwidth_gbps_per_card": (
                    self.hbf_hardware.hbf_write_bandwidth_gbps_per_card),
                "lpddr_bandwidth_gbps_per_card": (
                    self.hbf_hardware.lpddr_bandwidth_gbps_per_card),
            },
            "decode_role": {
                "gpus": self.hardware.decode_gpu_count,
                "hbm_capacity_bytes_per_gpu": (
                    self.hardware.hbm_capacity_bytes_per_gpu),
            },
            "kv_home": (
                "P-side HBF; the D copy is a working cache produced by the "
                "layerwise NVLink handoff.  HBF-tier restores and D-to-HBF "
                "demotions are ownership updates: the modeled context read "
                "is inside the P-batch HBF roofline, and prefill already "
                "wrote the durable copy"),
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


def build_hetero_system_from_config(
        *, repo_root: Path, config_path: Path,
        **engine: Any) -> SingleHBFPrefillTieredSystem:
    gpu, hbf, _layout = load_hetero_hbf_prefill_config(config_path)
    return SingleHBFPrefillTieredSystem(
        repo_root=repo_root,
        hardware=gpu,
        hbf_hardware=hbf,
        **engine,
    )


__all__ = [
    "HBF_PREFILL_CARD_COUNT",
    "HBF_PREFILL_LAYOUT",
    "HETERO_MODE_NAME",
    "HBFPrefillLatencyAdapter",
    "SingleHBFPrefillTieredSystem",
    "build_hetero_system_from_config",
    "hbf_home_capacity_bytes",
    "load_hetero_hbf_prefill_config",
    "virtual_hbf_tier_hardware",
]

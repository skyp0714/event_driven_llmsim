"""Analytical full-model latency for an eight-card HBF-GPU server.

This module is intentionally separate from the attention-only WakeKV path.
Every HBF worker executes the complete transformer, including dense
projections, attention, MoE, the language-model head, and sampling.  The
H100 TP4 calibration and an H100-class 989.5-TFLOP/s compute peak are the
empirical anchors, while per-rank work is rebuilt for TP1, TP4, and TP8.

Model weights and committed KV are read from HBF.  Activations and active-turn
KV are read from LPDDR.  The two media have independent roofline terms.  TP
and EP communication is charged explicitly on the configured intra-server
fabric; it is not hidden inside compute time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .h100_kernel_calibrated_prompt import (
    BF16_BYTES,
    H100_SMS,
    QWEN_EXPERT_INTERMEDIATE,
    QWEN_EXPERTS,
    QWEN_HEAD_DIM,
    QWEN_HIDDEN_SIZE,
    QWEN_KV_HEADS,
    QWEN_LAYERS,
    QWEN_Q_HEADS,
    QWEN_TOP_K,
    QWEN_VOCAB_SIZE,
)
from .online_latency_model import (
    H100_QWEN3_TP4_KERNEL_CALIBRATED,
    OnlineLatencyModelError,
    build_online_latency_model,
)
from .hbf_pcie_topology import (
    CROSS_ROOT_ROUTE,
    HBFPCIeTopology,
    ROOT_LOCAL_ROUTE,
    collective_runtime_ns,
)


SCHEMA_VERSION = 1
SUPPORTED_TP_SIZES = frozenset({1, 4, 8})
COLLECTIVE_FIXED_LATENCY_SEMANTICS = (
    "end_to_end_fixed_per_logical_collective")
ONLINE_SOFTMAX_ACCUMULATOR_BYTES = 4
ONLINE_SOFTMAX_STATE_ELEMENTS = QWEN_HEAD_DIM + 2
DEFAULT_MODEL_CONFIG = (
    "configs/model/Qwen/Qwen3-30B-A3B-Instruct-2507.json"
)


def qwen_logical_kv_bytes_per_token(element_bytes: int = 2) -> int:
    _positive_int("element_bytes", element_bytes)
    return (
        2
        * QWEN_LAYERS
        * QWEN_KV_HEADS
        * QWEN_HEAD_DIM
        * element_bytes
    )


def qwen_model_weight_bytes_per_rank(
        tp_size: int, element_bytes: int = 2) -> int:
    """Return exact per-rank Qwen weights for the modeled TP/EP layout."""

    if tp_size not in SUPPORTED_TP_SIZES:
        raise ValueError(
            f"tp_size must be one of {sorted(SUPPORTED_TP_SIZES)}")
    _positive_int("element_bytes", element_bytes)
    q_dim = (QWEN_Q_HEADS // tp_size) * QWEN_HEAD_DIM
    kv_dim = (
        max(1, math.ceil(QWEN_KV_HEADS / tp_size))
        * QWEN_HEAD_DIM
    )
    embedding = (
        (QWEN_VOCAB_SIZE // tp_size)
        * QWEN_HIDDEN_SIZE
        * element_bytes
    )
    norm = QWEN_HIDDEN_SIZE * element_bytes
    qk_norm = 2 * QWEN_HEAD_DIM * element_bytes
    qkv = (
        QWEN_HIDDEN_SIZE * (q_dim + 2 * kv_dim) * element_bytes)
    o_proj = q_dim * QWEN_HIDDEN_SIZE * element_bytes
    router = QWEN_HIDDEN_SIZE * QWEN_EXPERTS * element_bytes
    experts = (
        (QWEN_EXPERTS // tp_size)
        * 3
        * QWEN_HIDDEN_SIZE
        * QWEN_EXPERT_INTERMEDIATE
        * element_bytes
    )
    block = 2 * norm + qk_norm + qkv + o_proj + router + experts
    lm_head = embedding
    return embedding + QWEN_LAYERS * block + norm + lm_head


def _strict_fields(
        raw: Mapping[str, object], allowed: set[str], scope: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown {scope} field(s): {unknown}")


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


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _two_way_zigzag_prefix_tokens(length: int, pair_rank: int) -> int:
    """Count tokens assigned to ``pair_rank`` in ``[0, length)``.

    Logical token positions alternate between the two cards.  This is the
    token-granular two-way zigzag used for HBF, active LPDDR KV, and causal
    current-turn KV.  Rank zero owns even positions and rank one owns odd
    positions within every sequence.
    """

    if length < 0:
        raise ValueError("zigzag prefix length must be non-negative")
    if pair_rank not in (0, 1):
        raise ValueError("zigzag pair_rank must be 0 or 1")
    return (length + (1 if pair_rank == 0 else 0)) // 2


def _two_way_zigzag_range_tokens(
        start: int, length: int, pair_rank: int) -> int:
    if start < 0 or length < 0:
        raise ValueError("zigzag range must be non-negative")
    return (
        _two_way_zigzag_prefix_tokens(start + length, pair_rank)
        - _two_way_zigzag_prefix_tokens(start, pair_rank)
    )


def _two_way_zigzag_causal_visible_pairs(
        prior_tokens: int, query_tokens: int, pair_rank: int) -> int:
    """Count exact causal Q/K pairs assigned to one physical card."""

    if prior_tokens < 0 or query_tokens < 0:
        raise ValueError("causal zigzag lengths must be non-negative")
    full_pairs = (
        query_tokens * prior_tokens
        + query_tokens * (query_tokens + 1) // 2
    )
    first_visible_length = prior_tokens + 1
    odd_prefixes = (
        query_tokens // 2
        + (
            1
            if (
                query_tokens % 2 == 1
                and first_visible_length % 2 == 1
            )
            else 0
        )
    )
    if pair_rank == 0:
        return (full_pairs + odd_prefixes) // 2
    if pair_rank == 1:
        return (full_pairs - odd_prefixes) // 2
    raise ValueError("zigzag pair_rank must be 0 or 1")


@dataclass(frozen=True)
class HBFServerHardware:
    """Hardware inputs shared by all parallel layouts.

    ``npu_peak_tflops_per_card`` is retained as a configuration-compatibility
    name.  The modeled compute device is an H100-class GPU and uses the same
    989.5-TFLOP/s peak as the pinned P4D4 H100 configuration.
    """

    card_count: int = 8
    hbf_capacity_bytes_per_card: int = 1_280_000_000_000
    hbf_read_bandwidth_gbps_per_card: float = 3_350.0
    hbf_read_latency_us: float = 5.0
    hbf_read_prefetch_enabled: bool = False
    hbf_write_bandwidth_gbps_per_card: float = 335.0
    hbf_write_latency_us: float = 20.0
    npu_peak_tflops_per_card: float = 989.5
    lpddr_capacity_bytes_per_card: int = 64 * 1024 ** 3
    lpddr_bandwidth_gbps_per_card: float = 204.8
    intra_fabric_bandwidth_gbps_per_card: float = 50.0
    intra_fabric_fixed_latency_us: float = 3.0
    pcie_root_count: int = 2
    cards_per_pcie_root: int = 4
    pcie_card_to_root: tuple[int, ...] = (
        0, 0, 0, 0, 1, 1, 1, 1)
    pcie_nic_to_root: tuple[int, ...] = (0,)
    pcie_resource_mode: str = "shared"
    pcie_p2p_mode: str = "cross_root"
    pcie_root_bandwidth_gbps: float = 200.0
    pcie_root_fixed_latency_us: float = 0.0
    pcie_inter_root_bandwidth_gbps: float = 100.0
    pcie_inter_root_fixed_latency_us: float = 0.0
    rdma_bandwidth_gbps: float = 80.0
    rdma_one_way_latency_us: float = 10.0

    @property
    def gpu_peak_tflops_per_card(self) -> float:
        """Return the H100-class GPU compute peak used by the HBF card."""

        return self.npu_peak_tflops_per_card

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "HBFServerHardware":
        _strict_fields(raw, set(cls.__dataclass_fields__), "hardware")
        normalized = dict(raw)
        for name in ("pcie_card_to_root", "pcie_nic_to_root"):
            if name in normalized:
                mapping = normalized[name]
                if not isinstance(mapping, list):
                    raise ValueError(
                        f"hardware.{name} must be a JSON array")
                normalized[name] = tuple(mapping)
        value = cls(**normalized)
        value.validate()
        return value

    def validate(self) -> None:
        if not isinstance(self.hbf_read_prefetch_enabled, bool):
            raise ValueError(
                "hardware.hbf_read_prefetch_enabled must be a boolean")
        for name in ("pcie_card_to_root", "pcie_nic_to_root"):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError(
                    f"hardware.{name} must be an immutable tuple")
        for name in (
            "card_count",
            "hbf_capacity_bytes_per_card",
            "lpddr_capacity_bytes_per_card",
            "pcie_root_count",
            "cards_per_pcie_root",
        ):
            _positive_int(f"hardware.{name}", getattr(self, name))
        for name in (
            "hbf_read_bandwidth_gbps_per_card",
            "hbf_write_bandwidth_gbps_per_card",
            "npu_peak_tflops_per_card",
            "lpddr_bandwidth_gbps_per_card",
            "intra_fabric_bandwidth_gbps_per_card",
            "pcie_root_bandwidth_gbps",
            "pcie_inter_root_bandwidth_gbps",
            "rdma_bandwidth_gbps",
        ):
            _positive_float(f"hardware.{name}", getattr(self, name))
        for name in (
            "hbf_read_latency_us",
            "hbf_write_latency_us",
            "intra_fabric_fixed_latency_us",
            "pcie_root_fixed_latency_us",
            "pcie_inter_root_fixed_latency_us",
            "rdma_one_way_latency_us",
        ):
            _nonnegative_float(f"hardware.{name}", getattr(self, name))
        if self.card_count != 8:
            raise ValueError("full-model HBF server currently requires 8 cards")
        if self.pcie_root_count * self.cards_per_pcie_root != (
                self.card_count):
            raise ValueError(
                "PCIe roots must cover every HBF card exactly once")
        HBFPCIeTopology.from_hardware(self, server_id=0)


@dataclass(frozen=True)
class HBFParallelLayout:
    """One of the supported eight-card full-model layouts.

    ``tp8`` is conventional GQA TP8 and replicates each KV head across its
    two Q ranks.  ``tp8_context`` keeps the same dense/model-weight TP8
    sharding, but maps each KV head to a two-card pair and work-balances that
    head's token context across the pair.  Each card therefore evaluates all
    eight Q heads associated with its pair's KV head over its exact
    even/odd-token half (ceil/floor for odd lengths), and the pair merges
    online-softmax partials.
    """

    key: str
    tp_size: int
    replicas: int

    @classmethod
    def for_key(cls, key: str) -> "HBFParallelLayout":
        layouts = {
            "dp8": cls(key="dp8", tp_size=1, replicas=8),
            "tp4": cls(key="tp4", tp_size=4, replicas=2),
            "tp8": cls(key="tp8", tp_size=8, replicas=1),
            "tp8_context": cls(
                key="tp8_context", tp_size=8, replicas=1),
        }
        try:
            return layouts[key]
        except KeyError as exc:
            raise ValueError(
                f"unsupported HBF parallel layout {key!r}; "
                f"supported={sorted(layouts)}"
            ) from exc

    def validate(self, card_count: int = 8) -> None:
        if self.tp_size not in SUPPORTED_TP_SIZES:
            raise ValueError(
                f"tp_size must be one of {sorted(SUPPORTED_TP_SIZES)}")
        _positive_int("layout.replicas", self.replicas)
        if self.is_context_striped and (
                self.tp_size != 8 or self.replicas != 1):
            raise ValueError(
                "tp8_context requires one TP8 replica")
        if self.tp_size * self.replicas != card_count:
            raise ValueError(
                "layout must consume every HBF card exactly once: "
                f"tp={self.tp_size}, replicas={self.replicas}, "
                f"cards={card_count}")

    @property
    def physical_kv_replication_factor(self) -> int:
        if self.is_context_striped:
            return 1
        # Qwen has four KV heads. Standard TP8 pairs two Q ranks with one
        # replicated KV head; TP1 and TP4 store one physical copy.
        return max(1, self.tp_size // QWEN_KV_HEADS)

    @property
    def is_context_striped(self) -> bool:
        return self.key == "tp8_context"

    @property
    def q_heads_per_rank(self) -> int:
        """Return the normal dense TP shard (also the rank-owned Q outputs)."""

        return QWEN_Q_HEADS // self.tp_size

    @property
    def attention_q_heads_per_rank(self) -> int:
        """Return Q heads evaluated by one rank's attention kernel."""

        if self.is_context_striped:
            return QWEN_Q_HEADS // QWEN_KV_HEADS
        return self.q_heads_per_rank

    @property
    def kv_heads_per_rank(self) -> int:
        return max(1, QWEN_KV_HEADS // self.tp_size)

    @property
    def context_partition_factor(self) -> int:
        return 2 if self.is_context_striped else 1

    @property
    def kv_layout_semantics(self) -> str:
        if self.is_context_striped:
            return (
                "four_kv_heads_mapped_to_card_pairs_with_unique_"
                "even_odd_token_zigzag_context_q8_attention")
        return {
            1: "unsharded_four_kv_heads_per_dp_replica",
            4: "one_unique_kv_head_per_tp_rank",
            8: "one_kv_head_replicated_across_each_two_q_ranks",
        }[self.tp_size]


@dataclass(frozen=True)
class HBFModelBatchShape:
    """Exact media placement for one full-model serving iteration.

    ``prefill_hbf_k`` and ``decode_hbf_k`` contain committed prefix tokens.
    Their LPDDR counterparts contain active-turn tokens that have not yet
    been persisted to HBF.  Current prefill query K/V is implicitly LPDDR.
    """

    total_tokens: int
    prefill_q: tuple[int, ...] = ()
    prefill_hbf_k: tuple[int, ...] = ()
    prefill_lpddr_k: tuple[int, ...] = ()
    decode_hbf_k: tuple[int, ...] = ()
    decode_lpddr_k: tuple[int, ...] = ()
    lm_head_sequences: int = 1

    @property
    def num_decode(self) -> int:
        return len(self.decode_hbf_k)

    @property
    def real_query_tokens(self) -> int:
        return sum(self.prefill_q) + self.num_decode

    @property
    def padding_tokens(self) -> int:
        return self.total_tokens - self.real_query_tokens

    @property
    def decode_k(self) -> tuple[int, ...]:
        return tuple(
            hbf + lpddr
            for hbf, lpddr in zip(
                self.decode_hbf_k, self.decode_lpddr_k)
        )

    def validate(self) -> None:
        if self.total_tokens <= 0:
            raise ValueError("batch total_tokens must be positive")
        if self.lm_head_sequences <= 0:
            raise ValueError("lm_head_sequences must be positive")
        if self.padding_tokens < 0:
            raise ValueError(
                "batch total_tokens is smaller than real query tokens")
        prefill_lengths = {
            len(self.prefill_q),
            len(self.prefill_hbf_k),
            len(self.prefill_lpddr_k),
        }
        if len(prefill_lengths) != 1:
            raise ValueError(
                "prefill q/HBF-K/LPDDR-K vectors must align")
        if len(self.decode_hbf_k) != len(self.decode_lpddr_k):
            raise ValueError("decode HBF-K/LPDDR-K vectors must align")
        if not self.prefill_q and not self.decode_hbf_k:
            raise ValueError("batch must contain prefill or decode work")
        for q, hbf_k, lpddr_k in zip(
                self.prefill_q,
                self.prefill_hbf_k,
                self.prefill_lpddr_k):
            if q <= 0 or q > 131_072:
                raise ValueError(
                    "prefill q must be in the range 1..131072")
            if hbf_k < 0 or lpddr_k < 0:
                raise ValueError("prefill K lengths must be non-negative")
            if q + hbf_k + lpddr_k > 1_010_000:
                raise ValueError("prefill context exceeds 1,010,000 tokens")
        for hbf_k, lpddr_k in zip(
                self.decode_hbf_k, self.decode_lpddr_k):
            if hbf_k < 0 or lpddr_k < 0:
                raise ValueError("decode K lengths must be non-negative")
            if not 1 <= hbf_k + lpddr_k <= 1_010_000:
                raise ValueError(
                    "decode total K must be in 1..1,010,000")
        if self.num_decode > 128:
            raise ValueError("decode batch exceeds calibrated support (128)")


@dataclass(frozen=True)
class KernelLatency:
    family: str
    latency_ns: int
    latency_seconds: float
    flops: float
    hbf_read_bytes: float
    lpddr_bytes: float
    compute_roof_ns: int
    hbf_roof_ns: int
    lpddr_roof_ns: int
    dominant_roof: str


@dataclass(frozen=True, slots=True)
class HBFContextAttentionRankExecution:
    """Exact work for one physical rank inside a KV-head card pair.

    The same pair-rank work applies independently to each of Qwen's four
    KV-head pairs.  Unlike the scalar critical-path kernel operation, these
    two records preserve odd-token asymmetry so a detailed ASTRA projector
    can emit card-specific runtime and byte stages without multiplying a
    maximum-rank byte count by eight.
    """

    pair_rank: int
    token_mapping: str
    q_heads: int
    kv_heads: int
    query_tokens: int
    visible_pairs: int
    hbf_kv_tokens: int
    lpddr_kv_tokens: int
    latency_ns: int
    flops: float
    hbf_read_bytes: float
    lpddr_bytes: float
    compute_roof_ns: int
    hbf_roof_ns: int
    lpddr_roof_ns: int
    dominant_roof: str


@dataclass(frozen=True, slots=True)
class HBFKernelExecutionOp:
    """One calibrated kernel in the ordered HBF batch execution plan.

    ``latency_ns`` already includes the calibrated maximum of the compute,
    HBF-read, and LPDDR rooflines.  The individual roofline fields and byte
    counts are audit metadata, not additional serialized operations.
    """

    kind: str
    name: str
    category: str
    layer_index: int | None
    family: str
    latency_ns: int
    flops: float
    hbf_read_bytes_per_rank: float
    lpddr_bytes_per_rank: float
    compute_roof_ns: int
    hbf_roof_ns: int
    lpddr_roof_ns: int
    dominant_roof: str
    latency_semantics: str
    kv_layout_semantics: str


@dataclass(frozen=True, slots=True)
class HBFCollectiveExecutionOp:
    """One logical TP/EP collective in execution order.

    ``logical_payload_bytes`` follows the named collective's input
    convention, while ``transferred_bytes_per_rank`` is the effective
    per-rank traffic used by the analytical latency model.
    """

    kind: str
    name: str
    category: str
    layer_index: int
    collective_type: str
    latency_ns: int
    logical_payload_bytes: int
    payload_semantics: str
    transferred_bytes_per_rank: int
    fixed_latency_semantics: str


HBFExecutionOp = HBFKernelExecutionOp | HBFCollectiveExecutionOp


@dataclass(frozen=True, slots=True)
class HBFModelBatchExecutionPlan:
    """Immutable, ordered operations for one full-model HBF batch."""

    layout: str
    tp_size: int
    replicas: int
    physical_kv_replication_factor: int
    kv_layout_semantics: str
    operations: tuple[HBFExecutionOp, ...]
    context_attention_rank_executions: tuple[
        HBFContextAttentionRankExecution, ...] = ()

    @property
    def total_ns(self) -> int:
        return sum(operation.latency_ns for operation in self.operations)

    @property
    def kernel_operations(self) -> tuple[HBFKernelExecutionOp, ...]:
        return tuple(
            operation
            for operation in self.operations
            if isinstance(operation, HBFKernelExecutionOp)
        )

    @property
    def collective_operations(
            self) -> tuple[HBFCollectiveExecutionOp, ...]:
        return tuple(
            operation
            for operation in self.operations
            if isinstance(operation, HBFCollectiveExecutionOp)
        )

    @property
    def context_attention_hbf_read_bytes_per_layer(self) -> float:
        """Return exact physical bytes across one two-card KV-head pair."""

        return sum(
            rank.hbf_read_bytes
            for rank in self.context_attention_rank_executions)

    @property
    def context_attention_lpddr_bytes_per_layer(self) -> float:
        """Return exact physical bytes across one two-card KV-head pair."""

        return sum(
            rank.lpddr_bytes
            for rank in self.context_attention_rank_executions)

    @property
    def context_attention_physical_hbf_read_bytes_per_layer(self) -> float:
        """Return exact HBF-read bytes across all four physical pairs."""

        return (
            self.context_attention_hbf_read_bytes_per_layer
            * QWEN_KV_HEADS
        )

    @property
    def context_attention_physical_lpddr_bytes_per_layer(self) -> float:
        """Return exact LPDDR bytes across all four physical pairs."""

        return (
            self.context_attention_lpddr_bytes_per_layer
            * QWEN_KV_HEADS
        )


@dataclass(frozen=True)
class HBFModelBatchLatency:
    layout: str
    tp_size: int
    replicas: int
    total_ns: int
    embedding_ns: int
    dense_ns: int
    attention_ns: int
    router_ns: int
    moe_ns: int
    final_ns: int
    collective_ns: int
    tp_allreduce_ns: int
    ep_allgather_ns: int
    ep_reduce_scatter_ns: int
    hbf_read_bytes_per_rank: int
    lpddr_bytes_per_rank: int
    collective_bytes_per_rank: int
    compute_roof_ns_sum: int
    hbf_roof_ns_sum: int
    lpddr_roof_ns_sum: int
    attention_compute_roof_ns: int
    attention_hbf_roof_ns: int
    attention_lpddr_roof_ns: int
    attention_dominant_roof: str
    dominant_kernel_counts: Mapping[str, int]
    pair_query_exchange_ns: int = 0
    pair_softmax_partial_exchange_ns: int = 0
    pair_attention_merge_ns: int = 0
    pair_query_exchange_bytes_per_rank: int = 0
    pair_softmax_partial_exchange_bytes_per_rank: int = 0
    pair_attention_merge_lpddr_bytes_per_rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class FullModelHBFLatencyModel:
    """Evaluate complete Qwen3 batches on one HBF TP replica."""

    def __init__(
            self, *, base_provider, hardware: HBFServerHardware,
            layout: HBFParallelLayout) -> None:
        hardware.validate()
        layout.validate(hardware.card_count)
        topology = HBFPCIeTopology.from_hardware(
            hardware, server_id=0)
        topology.validate_layout(
            layout_key=layout.key,
            tp_size=layout.tp_size,
            replicas=layout.replicas,
        )
        self.base_provider = base_provider
        self.hardware = hardware
        self.layout = layout
        self.pcie_topology = topology
        self.band = base_provider.band

    def _fit(self, family: str):
        try:
            return self.base_provider.calibration.fits[family]
        except KeyError as exc:
            raise OnlineLatencyModelError(
                f"missing H100 calibration family {family!r}") from exc

    @staticmethod
    def _ns(seconds: float) -> int:
        if not math.isfinite(seconds) or seconds < 0:
            raise OnlineLatencyModelError(
                f"invalid full-model HBF latency {seconds!r}")
        return max(1, int(math.ceil(seconds * 1e9)))

    def _hbf_read_seconds(
            self, byte_count: float, *,
            include_fixed_latency: bool = True) -> float:
        """Return HBF service while keeping fixed latency auditable.

        Perfect prefetch hides only the configured first-access latency.
        The byte transfer remains charged at the configured HBF bandwidth.
        ``include_fixed_latency=False`` is used for the incremental half of
        one fused mixed-attention kernel after its other half already paid
        the single fixed access latency.
        """

        if byte_count < 0:
            raise ValueError("HBF read bytes must be non-negative")
        if byte_count == 0:
            return 0.0
        fixed_seconds = (
            self.hardware.hbf_read_latency_us * 1e-6
            if (
                include_fixed_latency
                and not self.hardware.hbf_read_prefetch_enabled
            )
            else 0.0
        )
        return (
            fixed_seconds
            + byte_count
            / (
                self.hardware.hbf_read_bandwidth_gbps_per_card
                * 1e9
            )
        )

    def _kernel(
            self, family: str, *, flops: float,
            hbf_read_bytes: float = 0.0, lpddr_bytes: float = 0.0,
            wave_penalty: float = 1.0) -> KernelLatency:
        if (
            flops < 0
            or hbf_read_bytes < 0
            or lpddr_bytes < 0
            or wave_penalty < 1
        ):
            raise ValueError("kernel work must be non-negative")
        fit = self._fit(family)
        compute_seconds = (
            flops
            / (self.hardware.gpu_peak_tflops_per_card * 1e12)
            * wave_penalty
        )
        hbf_seconds = self._hbf_read_seconds(hbf_read_bytes)
        lpddr_seconds = (
            lpddr_bytes
            / (self.hardware.lpddr_bandwidth_gbps_per_card * 1e9)
        )
        roofs = {
            "compute": compute_seconds,
            "hbf_read": hbf_seconds,
            "lpddr": lpddr_seconds,
        }
        dominant = max(roofs, key=roofs.get)
        roof_seconds = roofs[dominant]
        seconds = max(
            fit.launch_floor_seconds,
            roof_seconds * fit.eta(self.band),
        )
        return KernelLatency(
            family=family,
            latency_ns=self._ns(seconds),
            latency_seconds=seconds,
            flops=float(flops),
            hbf_read_bytes=float(hbf_read_bytes),
            lpddr_bytes=float(lpddr_bytes),
            compute_roof_ns=int(math.ceil(compute_seconds * 1e9)),
            hbf_roof_ns=int(math.ceil(hbf_seconds * 1e9)),
            lpddr_roof_ns=int(math.ceil(lpddr_seconds * 1e9)),
            dominant_roof=dominant,
        )

    def _elementwise(
            self, family: str, tokens: int, width: int,
            flops_per_element: float, byte_passes: float) -> KernelLatency:
        return self._kernel(
            family,
            flops=flops_per_element * tokens * width,
            lpddr_bytes=(
                BF16_BYTES * byte_passes * tokens * width),
        )

    def _gemm(
            self, family: str, m: int, k: int, n: int) -> KernelLatency:
        return self._kernel(
            family,
            flops=2.0 * m * k * n,
            hbf_read_bytes=BF16_BYTES * k * n,
            lpddr_bytes=BF16_BYTES * (m * k + m * n),
        )

    def _attention_work(
            self, *, q_lengths: Sequence[int],
            hbf_k_lengths: Sequence[int],
            lpddr_k_lengths: Sequence[int],
            causal_prefill: bool,
            context_rank: int | None = None,
    ) -> tuple[float, float, float, float]:
        q_heads = self.layout.attention_q_heads_per_rank
        kv_heads = self.layout.kv_heads_per_rank
        q_total = sum(q_lengths)
        blocks = (
            sum(
                math.ceil(q / 128) * q_heads
                for q in q_lengths
            )
            if causal_prefill
            else len(q_lengths) * q_heads
        )
        wave_penalty = (
            math.ceil(max(1, blocks) / H100_SMS)
            * H100_SMS
            / max(1, blocks)
        )

        def build_work(
                pair_rank: int | None,
        ) -> tuple[float, float, float, float]:
            if pair_rank is None:
                hbf_total = sum(hbf_k_lengths)
                lpddr_total = sum(lpddr_k_lengths)
                if causal_prefill:
                    visible_pairs = sum(
                        q * (hbf_k + lpddr_k)
                        + q * (q + 1) / 2
                        for q, hbf_k, lpddr_k in zip(
                            q_lengths,
                            hbf_k_lengths,
                            lpddr_k_lengths,
                        )
                    )
                    lpddr_k_total = lpddr_total + q_total
                else:
                    visible_pairs = sum(
                        q * (hbf_k + lpddr_k)
                        for q, hbf_k, lpddr_k in zip(
                            q_lengths,
                            hbf_k_lengths,
                            lpddr_k_lengths,
                        )
                    )
                    lpddr_k_total = lpddr_total
            else:
                hbf_total = 0
                lpddr_k_total = 0
                visible_pairs = 0
                for q, hbf_k, lpddr_k in zip(
                        q_lengths,
                        hbf_k_lengths,
                        lpddr_k_lengths):
                    prior = hbf_k + lpddr_k
                    hbf_total += _two_way_zigzag_range_tokens(
                        0, hbf_k, pair_rank)
                    lpddr_k_total += _two_way_zigzag_range_tokens(
                        hbf_k, lpddr_k, pair_rank)
                    if causal_prefill:
                        lpddr_k_total += _two_way_zigzag_range_tokens(
                            prior, q, pair_rank)
                        visible_pairs += (
                            _two_way_zigzag_causal_visible_pairs(
                                prior, q, pair_rank))
                    else:
                        visible_pairs += (
                            q
                            * _two_way_zigzag_prefix_tokens(
                                prior, pair_rank)
                        )
            flops = (
                visible_pairs
                * q_heads
                * (4.0 * QWEN_HEAD_DIM + 5.0)
            )
            hbf_bytes = (
                BF16_BYTES
                * 2
                * hbf_total
                * kv_heads
                * QWEN_HEAD_DIM
            )
            lpddr_bytes = BF16_BYTES * (
                2 * q_total * q_heads * QWEN_HEAD_DIM
                + 2 * lpddr_k_total * kv_heads * QWEN_HEAD_DIM
            )
            return flops, hbf_bytes, lpddr_bytes, wave_penalty

        if not self.layout.is_context_striped:
            if context_rank is not None:
                raise ValueError(
                    "context_rank is only valid for tp8_context")
            return build_work(None)
        if context_rank is not None:
            return build_work(context_rank)
        by_rank = (build_work(0), build_work(1))
        return (
            max(work[0] for work in by_rank),
            max(work[1] for work in by_rank),
            max(work[2] for work in by_rank),
            wave_penalty,
        )

    def _prefill_attention(
            self, shape: HBFModelBatchShape, *,
            context_rank: int | None = None) -> KernelLatency | None:
        if not shape.prefill_q:
            return None
        work = self._attention_work(
            q_lengths=shape.prefill_q,
            hbf_k_lengths=shape.prefill_hbf_k,
            lpddr_k_lengths=shape.prefill_lpddr_k,
            causal_prefill=True,
            context_rank=context_rank,
        )
        return self._kernel(
            "prefill_attention",
            flops=work[0],
            hbf_read_bytes=work[1],
            lpddr_bytes=work[2],
            wave_penalty=work[3],
        )

    def _decode_kernel_for_batch(
            self, *, hbf_k: Sequence[int],
            lpddr_k: Sequence[int],
            context_rank: int | None = None) -> KernelLatency:
        batch_size = len(hbf_k)
        if batch_size <= 0:
            raise ValueError("decode kernel requires a non-empty batch")
        total_k = [
            hbf + lpddr for hbf, lpddr in zip(hbf_k, lpddr_k)
        ]
        mean_total = sum(total_k)
        padded_total = batch_size * max(total_k)
        skew_alpha = 0.093
        effective_total = int(round(
            mean_total + skew_alpha * (padded_total - mean_total)
        ))
        scale = effective_total / max(1, mean_total)
        scaled_hbf = tuple(max(0, int(round(value * scale)))
                           for value in hbf_k)
        scaled_lpddr = tuple(max(0, int(round(value * scale)))
                             for value in lpddr_k)
        work = self._attention_work(
            q_lengths=(1,) * batch_size,
            hbf_k_lengths=scaled_hbf,
            lpddr_k_lengths=scaled_lpddr,
            causal_prefill=False,
            context_rank=context_rank,
        )
        actual = self._kernel(
            "prefill_attention",
            flops=work[0],
            hbf_read_bytes=work[1],
            lpddr_bytes=work[2],
            wave_penalty=work[3],
        )

        # Decode has a separately fitted launch floor/residual. Re-evaluate
        # every smaller batch at the same per-sequence media mix to preserve
        # the calibrated monotone upper envelope.
        best_seconds = 0.0
        for candidate in range(1, batch_size + 1):
            fraction = candidate / batch_size
            candidate_hbf_total = int(math.ceil(
                sum(scaled_hbf) * fraction))
            candidate_lpddr_total = int(math.ceil(
                sum(scaled_lpddr) * fraction))
            candidate_hbf = _balanced_lengths(
                candidate_hbf_total, candidate)
            candidate_lpddr = _balanced_lengths(
                candidate_lpddr_total, candidate)
            candidate_work = self._attention_work(
                q_lengths=(1,) * candidate,
                hbf_k_lengths=candidate_hbf,
                lpddr_k_lengths=candidate_lpddr,
                causal_prefill=False,
                context_rank=context_rank,
            )
            compute_seconds = (
                candidate_work[0]
                / (self.hardware.gpu_peak_tflops_per_card * 1e12)
                * candidate_work[3]
            )
            hbf_seconds = self._hbf_read_seconds(candidate_work[1])
            lpddr_seconds = (
                candidate_work[2]
                / (self.hardware.lpddr_bandwidth_gbps_per_card * 1e9)
            )
            roof = max(compute_seconds, hbf_seconds, lpddr_seconds)
            fit = self.base_provider._decode_fit(candidate)
            best_seconds = max(
                best_seconds,
                roof,
                fit.launch_floor_seconds + roof * fit.eta(self.band),
            )
        return KernelLatency(
            family="decode_attention",
            latency_ns=self._ns(best_seconds),
            latency_seconds=best_seconds,
            flops=actual.flops,
            hbf_read_bytes=actual.hbf_read_bytes,
            lpddr_bytes=actual.lpddr_bytes,
            compute_roof_ns=actual.compute_roof_ns,
            hbf_roof_ns=actual.hbf_roof_ns,
            lpddr_roof_ns=actual.lpddr_roof_ns,
            dominant_roof=actual.dominant_roof,
        )

    def _attention(
            self, shape: HBFModelBatchShape, *,
            context_rank: int | None = None) -> KernelLatency:
        if self.layout.is_context_striped and context_rank is None:
            return max(
                (
                    self._attention(shape, context_rank=pair_rank)
                    for pair_rank in (0, 1)
                ),
                key=lambda kernel: kernel.latency_ns,
            )
        prefill = self._prefill_attention(
            shape, context_rank=context_rank)
        decode = None
        if shape.decode_hbf_k:
            decode = self._decode_kernel_for_batch(
                hbf_k=shape.decode_hbf_k,
                lpddr_k=shape.decode_lpddr_k,
                context_rank=context_rank,
            )
        if prefill is None:
            assert decode is not None
            return decode
        if decode is None:
            return prefill
        prefill_work = self._attention_work(
            q_lengths=shape.prefill_q,
            hbf_k_lengths=shape.prefill_hbf_k,
            lpddr_k_lengths=shape.prefill_lpddr_k,
            causal_prefill=True,
            context_rank=context_rank,
        )
        prefill_compute_seconds = (
            prefill_work[0]
            / (self.hardware.gpu_peak_tflops_per_card * 1e12)
            * prefill_work[3]
        )
        prefill_hbf_seconds = self._hbf_read_seconds(
            prefill_work[1],
            include_fixed_latency=(decode.hbf_read_bytes == 0),
        )
        prefill_lpddr_seconds = (
            prefill_work[2]
            / (self.hardware.lpddr_bandwidth_gbps_per_card * 1e9)
        )
        prefill_roof_seconds = max(
            prefill_compute_seconds,
            prefill_hbf_seconds,
            prefill_lpddr_seconds,
        )
        prefill_fit = self._fit("prefill_attention")
        latency_seconds = (
            decode.latency_seconds
            + prefill_roof_seconds
            * prefill_fit.eta(self.band)
        )
        latency_ns = self._ns(latency_seconds)
        combined_hbf_bytes = (
            prefill.hbf_read_bytes + decode.hbf_read_bytes)
        combined_lpddr_bytes = prefill.lpddr_bytes + decode.lpddr_bytes
        combined_compute_roof_ns = (
            prefill.compute_roof_ns + decode.compute_roof_ns)
        combined_hbf_roof_ns = int(math.ceil(
            self._hbf_read_seconds(combined_hbf_bytes) * 1e9))
        combined_lpddr_roof_ns = (
            prefill.lpddr_roof_ns + decode.lpddr_roof_ns)
        return KernelLatency(
            family="mixed_attention",
            latency_ns=latency_ns,
            latency_seconds=latency_seconds,
            flops=prefill.flops + decode.flops,
            hbf_read_bytes=combined_hbf_bytes,
            lpddr_bytes=combined_lpddr_bytes,
            compute_roof_ns=combined_compute_roof_ns,
            hbf_roof_ns=combined_hbf_roof_ns,
            lpddr_roof_ns=combined_lpddr_roof_ns,
            dominant_roof=max(
                (
                    ("compute", combined_compute_roof_ns),
                    ("hbf_read", combined_hbf_roof_ns),
                    ("lpddr", combined_lpddr_roof_ns),
                ),
                key=lambda item: item[1],
            )[0],
        )

    @lru_cache(maxsize=512)
    def context_attention_rank_executions(
            self, shape: HBFModelBatchShape,
    ) -> tuple[HBFContextAttentionRankExecution, ...]:
        """Return card-specific attention work for a context-striped pair."""

        shape.validate()
        if not self.layout.is_context_striped:
            return ()
        result = []
        queries = shape.real_query_tokens
        q_heads = self.layout.attention_q_heads_per_rank
        kv_heads = self.layout.kv_heads_per_rank
        visible_pair_flops = (
            q_heads * (4.0 * QWEN_HEAD_DIM + 5.0))
        kv_token_bytes = (
            BF16_BYTES * 2 * kv_heads * QWEN_HEAD_DIM)
        query_io_bytes = (
            BF16_BYTES * 2 * queries * q_heads * QWEN_HEAD_DIM)

        def exact_count(name: str, value: float) -> int:
            rounded = int(round(value))
            if not math.isclose(
                    value, rounded, rel_tol=0.0, abs_tol=1e-6):
                raise AssertionError(
                    f"context attention {name} is not token integral: "
                    f"{value}")
            return rounded

        for pair_rank in (0, 1):
            kernel = self._attention(shape, context_rank=pair_rank)
            result.append(HBFContextAttentionRankExecution(
                pair_rank=pair_rank,
                token_mapping=(
                    "sequence_local_even_odd_token_zigzag"),
                q_heads=q_heads,
                kv_heads=kv_heads,
                query_tokens=queries,
                visible_pairs=exact_count(
                    "visible-pair count",
                    kernel.flops / visible_pair_flops,
                ),
                hbf_kv_tokens=exact_count(
                    "HBF KV token count",
                    kernel.hbf_read_bytes / kv_token_bytes,
                ),
                lpddr_kv_tokens=exact_count(
                    "LPDDR KV token count",
                    (
                        kernel.lpddr_bytes - query_io_bytes
                    ) / kv_token_bytes,
                ),
                latency_ns=kernel.latency_ns,
                flops=kernel.flops,
                hbf_read_bytes=kernel.hbf_read_bytes,
                lpddr_bytes=kernel.lpddr_bytes,
                compute_roof_ns=kernel.compute_roof_ns,
                hbf_roof_ns=kernel.hbf_roof_ns,
                lpddr_roof_ns=kernel.lpddr_roof_ns,
                dominant_roof=kernel.dominant_roof,
            ))
        return tuple(result)

    def _pair_attention_exchange_bytes(
            self, shape: HBFModelBatchShape) -> tuple[int, int]:
        """Return per-rank pair traffic for context-striped attention.

        Dense TP8 produces four Q heads on each rank.  Before attention, both
        ranks send their owned BF16 Q shard to the peer, so each evaluates
        all eight Q heads associated with the pair's one KV head.  After the
        local half-context attention, a rank sends only the four FP32
        ``(O, max, sum)`` partials owned by its peer.  Sending all eight
        partials in both directions would duplicate traffic without changing
        the normal TP8 o-projection ownership.
        """

        if not self.layout.is_context_striped:
            return 0, 0
        queries = shape.real_query_tokens
        owned_q_heads = self.layout.q_heads_per_rank
        query_bytes = (
            queries
            * owned_q_heads
            * QWEN_HEAD_DIM
            * BF16_BYTES
        )
        partial_bytes = (
            queries
            * owned_q_heads
            * ONLINE_SOFTMAX_STATE_ELEMENTS
            * ONLINE_SOFTMAX_ACCUMULATOR_BYTES
        )
        return query_bytes, partial_bytes

    def _pair_attention_merge(
            self, shape: HBFModelBatchShape) -> KernelLatency | None:
        """Model the post-exchange online-softmax merge on owned Q heads."""

        if not self.layout.is_context_striped:
            return None
        queries = shape.real_query_tokens
        owned_q_heads = self.layout.q_heads_per_rank
        state_values = (
            queries * owned_q_heads * ONLINE_SOFTMAX_STATE_ELEMENTS)
        output_values = queries * owned_q_heads * QWEN_HEAD_DIM
        # Read local and received FP32 (O, max, sum) states, then write the
        # merged rank-owned BF16 output consumed by the normal TP8 o_proj.
        lpddr_bytes = (
            2 * state_values * ONLINE_SOFTMAX_ACCUMULATOR_BYTES
            + output_values * BF16_BYTES
        )
        # Per head: combine two max/sum states and rescale/add/divide the two
        # D-dimensional partial outputs.  Transcendentals are counted as one
        # analytical operation each; the calibrated residual below captures
        # the empirical efficiency/launch floor.
        flops = (
            queries
            * owned_q_heads
            * (4.0 * QWEN_HEAD_DIM + 10.0)
        )
        compute_seconds = (
            flops
            / (self.hardware.gpu_peak_tflops_per_card * 1e12)
        )
        lpddr_seconds = (
            lpddr_bytes
            / (self.hardware.lpddr_bandwidth_gbps_per_card * 1e9)
        )
        fit = self._fit("prefill_attention")
        roof_seconds = max(compute_seconds, lpddr_seconds)
        latency_seconds = max(
            fit.launch_floor_seconds,
            roof_seconds * fit.eta(self.band),
        )
        return KernelLatency(
            family="online_softmax_merge",
            latency_ns=self._ns(latency_seconds),
            latency_seconds=latency_seconds,
            flops=float(flops),
            hbf_read_bytes=0.0,
            lpddr_bytes=float(lpddr_bytes),
            compute_roof_ns=int(math.ceil(compute_seconds * 1e9)),
            hbf_roof_ns=0,
            lpddr_roof_ns=int(math.ceil(lpddr_seconds * 1e9)),
            dominant_roof=(
                "compute"
                if compute_seconds >= lpddr_seconds
                else "lpddr"
            ),
        )

    def _moe(self, total_tokens: int) -> tuple[KernelLatency, ...]:
        tp = self.layout.tp_size
        experts_per_rank = QWEN_EXPERTS // tp
        expert_pairs = total_tokens * QWEN_TOP_K / tp
        active_experts = min(
            experts_per_rank, max(1, math.ceil(expert_pairs)))
        up = self._kernel(
            "expert_up",
            flops=(
                4.0 * expert_pairs * QWEN_HIDDEN_SIZE
                * QWEN_EXPERT_INTERMEDIATE
            ),
            hbf_read_bytes=(
                BF16_BYTES * 2 * active_experts * QWEN_HIDDEN_SIZE
                * QWEN_EXPERT_INTERMEDIATE
            ),
            lpddr_bytes=BF16_BYTES * (
                expert_pairs * QWEN_HIDDEN_SIZE
                + 2 * expert_pairs * QWEN_EXPERT_INTERMEDIATE
            ),
        )
        activation = self._kernel(
            "expert_activation",
            flops=8.0 * expert_pairs * QWEN_EXPERT_INTERMEDIATE,
            lpddr_bytes=(
                BF16_BYTES * 3.0 * expert_pairs
                * QWEN_EXPERT_INTERMEDIATE
            ),
        )
        down = self._kernel(
            "expert_down",
            flops=(
                2.0 * expert_pairs * QWEN_EXPERT_INTERMEDIATE
                * QWEN_HIDDEN_SIZE
            ),
            hbf_read_bytes=(
                BF16_BYTES * active_experts
                * QWEN_EXPERT_INTERMEDIATE * QWEN_HIDDEN_SIZE
            ),
            lpddr_bytes=BF16_BYTES * (
                expert_pairs * QWEN_EXPERT_INTERMEDIATE
                + expert_pairs * QWEN_HIDDEN_SIZE
            ),
        )
        return up, activation, down

    def _collective_ns(
            self, payload_bytes: int, *,
            rank_multiplier: float, route: str) -> int:
        return collective_runtime_ns(
            topology=self.pcie_topology,
            tp_size=self.layout.tp_size,
            payload_bytes=payload_bytes,
            rank_multiplier=rank_multiplier,
            route=route,
        )

    # Each expanded plan has hundreds of operations.  Keep only a bounded
    # working set rather than mirroring the much lighter latency-result cache.
    @lru_cache(maxsize=512)
    def batch_execution_plan(
            self, shape: HBFModelBatchShape) -> HBFModelBatchExecutionPlan:
        """Return the immutable ordered operations for ``shape``.

        Kernel latencies are roofline-inclusive.  HBF/LPDDR byte fields are
        therefore diagnostics for conformance and accounting; callers must
        not serialize separate memory-delay operations in addition to the
        kernel latency.
        """

        shape.validate()
        total_tokens = shape.total_tokens
        sequences = shape.lm_head_sequences
        tp = self.layout.tp_size
        q_dim = self.layout.q_heads_per_rank * QWEN_HEAD_DIM
        kv_dim = self.layout.kv_heads_per_rank * QWEN_HEAD_DIM
        qkv_dim = q_dim + 2 * kv_dim

        embedding = self._elementwise(
            "embedding", total_tokens, QWEN_HIDDEN_SIZE, 0.0, 1.0)
        layernorm = self._elementwise(
            "norm", total_tokens, QWEN_HIDDEN_SIZE, 8.0, 2.0)
        q_norm = self._elementwise(
            "norm", total_tokens, q_dim, 8.0, 2.0)
        k_norm = self._elementwise(
            "norm", total_tokens, kv_dim, 8.0, 2.0)
        qkv = self._gemm(
            "q_projection", total_tokens, QWEN_HIDDEN_SIZE, qkv_dim)
        rope = self._elementwise(
            "rope", total_tokens, q_dim + kv_dim, 10.0, 2.0)
        context_attention_rank_executions = (
            self.context_attention_rank_executions(shape))
        attention = self._attention(shape)
        pair_query_bytes, pair_partial_bytes = (
            self._pair_attention_exchange_bytes(shape))
        pair_query_ns = self._collective_ns(
            pair_query_bytes,
            rank_multiplier=1.0,
            route=ROOT_LOCAL_ROUTE,
        )
        pair_partial_ns = self._collective_ns(
            pair_partial_bytes,
            rank_multiplier=1.0,
            route=ROOT_LOCAL_ROUTE,
        )
        pair_merge = self._pair_attention_merge(shape)
        o_proj = self._gemm(
            "o_projection", total_tokens, q_dim, QWEN_HIDDEN_SIZE)
        router = self._gemm(
            "router",
            total_tokens,
            QWEN_HIDDEN_SIZE,
            QWEN_EXPERTS,
        )
        moe = self._moe(total_tokens)

        allreduce_payload = (
            total_tokens * QWEN_HIDDEN_SIZE * BF16_BYTES)
        dispatch_tokens_per_rank = max(1, total_tokens // tp)
        allgather_local_chunk = (
            dispatch_tokens_per_rank
            * (QWEN_HIDDEN_SIZE + QWEN_EXPERTS)
            * BF16_BYTES
        )
        reduce_scatter_total = allreduce_payload
        standard_collective_route = (
            CROSS_ROOT_ROUTE if tp == 8 else ROOT_LOCAL_ROUTE)
        allreduce_ns = self._collective_ns(
            allreduce_payload,
            rank_multiplier=2.0 * (tp - 1) / tp,
            route=standard_collective_route,
        )
        allgather_ns = self._collective_ns(
            allgather_local_chunk,
            rank_multiplier=tp - 1,
            route=standard_collective_route,
        )
        reduce_scatter_ns = self._collective_ns(
            reduce_scatter_total,
            rank_multiplier=(tp - 1) / tp,
            route=standard_collective_route,
        )

        final_norm = self._elementwise(
            "norm", total_tokens, QWEN_HIDDEN_SIZE, 8.0, 2.0)
        lm_head = self._gemm(
            "lm_head",
            sequences,
            QWEN_HIDDEN_SIZE,
            QWEN_VOCAB_SIZE // tp,
        )
        sampler_fit = self._fit("lm_head")
        sampler_bytes = (
            sequences * (QWEN_VOCAB_SIZE // tp) * BF16_BYTES)
        sampler_seconds = max(
            sampler_fit.launch_floor_seconds,
            sampler_bytes
            / (self.hardware.lpddr_bandwidth_gbps_per_card * 1e9),
        )
        sampler = KernelLatency(
            family="sampler",
            latency_ns=self._ns(sampler_seconds),
            latency_seconds=sampler_seconds,
            flops=0.0,
            hbf_read_bytes=0.0,
            lpddr_bytes=float(sampler_bytes),
            compute_roof_ns=0,
            hbf_roof_ns=0,
            lpddr_roof_ns=self._ns(
                sampler_bytes
                / (self.hardware.lpddr_bandwidth_gbps_per_card * 1e9)
            ),
            dominant_roof="lpddr",
        )

        kv_layout_semantics = self.layout.kv_layout_semantics

        def kernel_op(
                name: str, category: str, layer_index: int | None,
                kernel: KernelLatency, *,
                kv_semantics: str = "not_applicable",
                latency_semantics: str = (
                    "calibrated_max_roofline_inclusive"),
        ) -> HBFKernelExecutionOp:
            return HBFKernelExecutionOp(
                kind="kernel",
                name=name,
                category=category,
                layer_index=layer_index,
                family=kernel.family,
                latency_ns=kernel.latency_ns,
                flops=kernel.flops,
                hbf_read_bytes_per_rank=kernel.hbf_read_bytes,
                lpddr_bytes_per_rank=kernel.lpddr_bytes,
                compute_roof_ns=kernel.compute_roof_ns,
                hbf_roof_ns=kernel.hbf_roof_ns,
                lpddr_roof_ns=kernel.lpddr_roof_ns,
                dominant_roof=kernel.dominant_roof,
                latency_semantics=latency_semantics,
                kv_layout_semantics=kv_semantics,
            )

        def collective_op(
                name: str, layer_index: int, collective_type: str,
                latency_ns: int, logical_payload_bytes: int,
                payload_semantics: str,
                transferred_bytes_per_rank: int,
        ) -> HBFCollectiveExecutionOp:
            return HBFCollectiveExecutionOp(
                kind="collective",
                name=name,
                category={
                    "ALLREDUCE": "tp_allreduce",
                    "ALLGATHER": "ep_allgather",
                    "REDUCE_SCATTER": "ep_reduce_scatter",
                    "PAIR_QUERY_EXCHANGE": "pair_query_exchange",
                    "PAIR_SOFTMAX_PARTIAL_EXCHANGE":
                        "pair_softmax_partial_exchange",
                }[collective_type],
                layer_index=layer_index,
                collective_type=collective_type,
                latency_ns=latency_ns,
                logical_payload_bytes=logical_payload_bytes,
                payload_semantics=payload_semantics,
                transferred_bytes_per_rank=(
                    transferred_bytes_per_rank),
                fixed_latency_semantics=(
                    COLLECTIVE_FIXED_LATENCY_SEMANTICS),
            )

        operations: list[HBFExecutionOp] = [
            kernel_op("embedding", "embedding", None, embedding),
        ]
        for layer_index in range(QWEN_LAYERS):
            prefix = f"layer_{layer_index}"
            operations.extend((
                kernel_op(
                    f"{prefix}.input_layernorm",
                    "dense",
                    layer_index,
                    layernorm,
                ),
                kernel_op(
                    f"{prefix}.qkv_proj",
                    "dense",
                    layer_index,
                    qkv,
                ),
                kernel_op(
                    f"{prefix}.q_norm",
                    "dense",
                    layer_index,
                    q_norm,
                ),
                kernel_op(
                    f"{prefix}.k_norm",
                    "dense",
                    layer_index,
                    k_norm,
                ),
                kernel_op(
                    f"{prefix}.rotary_emb",
                    "dense",
                    layer_index,
                    rope,
                ),
            ))
            if self.layout.is_context_striped:
                operations.append(collective_op(
                    f"{prefix}.pair_query_exchange",
                    layer_index,
                    "PAIR_QUERY_EXCHANGE",
                    pair_query_ns,
                    pair_query_bytes,
                    "rank_owned_q4_bf16_bytes_sent_to_pair_peer",
                    pair_query_bytes,
                ))
            operations.append(kernel_op(
                f"{prefix}.attention",
                "attention",
                layer_index,
                attention,
                kv_semantics=kv_layout_semantics,
            ))
            if self.layout.is_context_striped:
                assert pair_merge is not None
                operations.append(collective_op(
                    f"{prefix}.pair_softmax_partial_exchange",
                    layer_index,
                    "PAIR_SOFTMAX_PARTIAL_EXCHANGE",
                    pair_partial_ns,
                    pair_partial_bytes,
                    (
                        "rank_owned_q4_fp32_o_max_sum_partial_bytes_"
                        "sent_to_pair_peer"
                    ),
                    pair_partial_bytes,
                ))
                operations.append(kernel_op(
                    f"{prefix}.pair_online_softmax_merge",
                    "attention_merge",
                    layer_index,
                    pair_merge,
                    kv_semantics=kv_layout_semantics,
                    latency_semantics=(
                        "prefill_attention_calibrated_local_"
                        "online_softmax_merge_roofline"),
                ))
            operations.append(kernel_op(
                f"{prefix}.o_proj",
                "dense",
                layer_index,
                o_proj,
            ))
            if tp > 1:
                operations.append(collective_op(
                    f"{prefix}.tp_allreduce",
                    layer_index,
                    "ALLREDUCE",
                    allreduce_ns,
                    allreduce_payload,
                    "total_collective_input_bytes",
                    int(
                        2.0 * (tp - 1) / tp
                        * allreduce_payload
                    ),
                ))
            operations.extend((
                kernel_op(
                    f"{prefix}.post_attention_layernorm",
                    "dense",
                    layer_index,
                    layernorm,
                ),
                kernel_op(
                    f"{prefix}.router",
                    "router",
                    layer_index,
                    router,
                ),
            ))
            if tp > 1:
                operations.append(collective_op(
                    f"{prefix}.ep_allgather",
                    layer_index,
                    "ALLGATHER",
                    allgather_ns,
                    allgather_local_chunk,
                    "local_input_chunk_bytes_per_rank",
                    (tp - 1) * allgather_local_chunk,
                ))
            operations.extend((
                kernel_op(
                    f"{prefix}.expert_up",
                    "moe",
                    layer_index,
                    moe[0],
                ),
                kernel_op(
                    f"{prefix}.expert_activation",
                    "moe",
                    layer_index,
                    moe[1],
                ),
                kernel_op(
                    f"{prefix}.expert_down",
                    "moe",
                    layer_index,
                    moe[2],
                ),
            ))
            if tp > 1:
                operations.append(collective_op(
                    f"{prefix}.ep_reduce_scatter",
                    layer_index,
                    "REDUCE_SCATTER",
                    reduce_scatter_ns,
                    reduce_scatter_total,
                    "total_collective_input_bytes",
                    int(
                        (tp - 1) / tp
                        * reduce_scatter_total
                    ),
                ))
        operations.extend((
            kernel_op("final_layernorm", "final", None, final_norm),
            kernel_op("lm_head", "final", None, lm_head),
            kernel_op("sampler", "final", None, sampler),
        ))
        return HBFModelBatchExecutionPlan(
            layout=self.layout.key,
            tp_size=tp,
            replicas=self.layout.replicas,
            physical_kv_replication_factor=(
                self.layout.physical_kv_replication_factor),
            kv_layout_semantics=kv_layout_semantics,
            operations=tuple(operations),
            context_attention_rank_executions=(
                context_attention_rank_executions),
        )

    @lru_cache(maxsize=262_144)
    def batch_latency(
            self, shape: HBFModelBatchShape) -> HBFModelBatchLatency:
        """Return the compact analytical latency summary for ``shape``.

        The ordered execution plan intentionally remains available through
        :meth:`batch_execution_plan` for diagnostics and ASTRA projection.
        Serving only needs the aggregate summary, however, so this hot path
        computes each distinct per-layer kernel once and applies the fixed
        Qwen layer multiplicity directly.  It therefore avoids constructing
        and scanning hundreds of immutable operation records on every cache
        miss.
        """

        shape.validate()
        total_tokens = shape.total_tokens
        sequences = shape.lm_head_sequences
        tp = self.layout.tp_size
        q_dim = self.layout.q_heads_per_rank * QWEN_HEAD_DIM
        kv_dim = self.layout.kv_heads_per_rank * QWEN_HEAD_DIM
        qkv_dim = q_dim + 2 * kv_dim

        embedding = self._elementwise(
            "embedding", total_tokens, QWEN_HIDDEN_SIZE, 0.0, 1.0)
        layernorm = self._elementwise(
            "norm", total_tokens, QWEN_HIDDEN_SIZE, 8.0, 2.0)
        q_norm = self._elementwise(
            "norm", total_tokens, q_dim, 8.0, 2.0)
        k_norm = self._elementwise(
            "norm", total_tokens, kv_dim, 8.0, 2.0)
        qkv = self._gemm(
            "q_projection", total_tokens, QWEN_HIDDEN_SIZE, qkv_dim)
        rope = self._elementwise(
            "rope", total_tokens, q_dim + kv_dim, 10.0, 2.0)
        attention = self._attention(shape)
        pair_query_bytes, pair_partial_bytes = (
            self._pair_attention_exchange_bytes(shape))
        pair_merge = self._pair_attention_merge(shape)
        o_proj = self._gemm(
            "o_projection", total_tokens, q_dim, QWEN_HIDDEN_SIZE)
        router = self._gemm(
            "router",
            total_tokens,
            QWEN_HIDDEN_SIZE,
            QWEN_EXPERTS,
        )
        moe = self._moe(total_tokens)

        final_norm = self._elementwise(
            "norm", total_tokens, QWEN_HIDDEN_SIZE, 8.0, 2.0)
        lm_head = self._gemm(
            "lm_head",
            sequences,
            QWEN_HIDDEN_SIZE,
            QWEN_VOCAB_SIZE // tp,
        )
        sampler_fit = self._fit("lm_head")
        sampler_bytes = (
            sequences * (QWEN_VOCAB_SIZE // tp) * BF16_BYTES)
        sampler_seconds = max(
            sampler_fit.launch_floor_seconds,
            sampler_bytes
            / (self.hardware.lpddr_bandwidth_gbps_per_card * 1e9),
        )
        sampler = KernelLatency(
            family="sampler",
            latency_ns=self._ns(sampler_seconds),
            latency_seconds=sampler_seconds,
            flops=0.0,
            hbf_read_bytes=0.0,
            lpddr_bytes=float(sampler_bytes),
            compute_roof_ns=0,
            hbf_roof_ns=0,
            lpddr_roof_ns=self._ns(
                sampler_bytes
                / (
                    self.hardware.lpddr_bandwidth_gbps_per_card
                    * 1e9
                )
            ),
            dominant_roof="lpddr",
        )

        dense_per_layer = (
            2 * layernorm.latency_ns
            + qkv.latency_ns
            + q_norm.latency_ns
            + k_norm.latency_ns
            + rope.latency_ns
            + o_proj.latency_ns
        )
        attention_per_layer = (
            attention.latency_ns
            + (pair_merge.latency_ns if pair_merge is not None else 0)
        )
        moe_per_layer = sum(kernel.latency_ns for kernel in moe)

        embedding_ns = embedding.latency_ns
        dense_ns = QWEN_LAYERS * dense_per_layer
        attention_ns = QWEN_LAYERS * attention_per_layer
        router_ns = QWEN_LAYERS * router.latency_ns
        moe_ns = QWEN_LAYERS * moe_per_layer
        final_ns = (
            final_norm.latency_ns
            + lm_head.latency_ns
            + sampler.latency_ns
        )

        allreduce_payload = (
            total_tokens * QWEN_HIDDEN_SIZE * BF16_BYTES)
        dispatch_tokens_per_rank = max(1, total_tokens // tp)
        allgather_local_chunk = (
            dispatch_tokens_per_rank
            * (QWEN_HIDDEN_SIZE + QWEN_EXPERTS)
            * BF16_BYTES
        )
        standard_collective_route = (
            CROSS_ROOT_ROUTE if tp == 8 else ROOT_LOCAL_ROUTE)

        if tp > 1:
            allreduce_per_layer_ns = self._collective_ns(
                allreduce_payload,
                rank_multiplier=2.0 * (tp - 1) / tp,
                route=standard_collective_route,
            )
            allgather_per_layer_ns = self._collective_ns(
                allgather_local_chunk,
                rank_multiplier=tp - 1,
                route=standard_collective_route,
            )
            reduce_scatter_per_layer_ns = self._collective_ns(
                allreduce_payload,
                rank_multiplier=(tp - 1) / tp,
                route=standard_collective_route,
            )
            allreduce_bytes_per_layer = int(
                2.0 * (tp - 1) / tp * allreduce_payload)
            allgather_bytes_per_layer = (
                (tp - 1) * allgather_local_chunk)
            reduce_scatter_bytes_per_layer = int(
                (tp - 1) / tp * allreduce_payload)
        else:
            allreduce_per_layer_ns = 0
            allgather_per_layer_ns = 0
            reduce_scatter_per_layer_ns = 0
            allreduce_bytes_per_layer = 0
            allgather_bytes_per_layer = 0
            reduce_scatter_bytes_per_layer = 0

        if self.layout.is_context_striped:
            pair_query_per_layer_ns = self._collective_ns(
                pair_query_bytes,
                rank_multiplier=1.0,
                route=ROOT_LOCAL_ROUTE,
            )
            pair_partial_per_layer_ns = self._collective_ns(
                pair_partial_bytes,
                rank_multiplier=1.0,
                route=ROOT_LOCAL_ROUTE,
            )
        else:
            pair_query_per_layer_ns = 0
            pair_partial_per_layer_ns = 0

        tp_allreduce_ns = QWEN_LAYERS * allreduce_per_layer_ns
        ep_allgather_ns = QWEN_LAYERS * allgather_per_layer_ns
        ep_reduce_scatter_ns = (
            QWEN_LAYERS * reduce_scatter_per_layer_ns)
        pair_query_exchange_ns = (
            QWEN_LAYERS * pair_query_per_layer_ns)
        pair_softmax_partial_exchange_ns = (
            QWEN_LAYERS * pair_partial_per_layer_ns)
        collective_ns = (
            tp_allreduce_ns
            + ep_allgather_ns
            + ep_reduce_scatter_ns
            + pair_query_exchange_ns
            + pair_softmax_partial_exchange_ns
        )

        per_layer_kernel_multiplicities = (
            (layernorm, 2),
            (qkv, 1),
            (q_norm, 1),
            (k_norm, 1),
            (rope, 1),
            (attention, 1),
            (o_proj, 1),
            (router, 1),
            (moe[0], 1),
            (moe[1], 1),
            (moe[2], 1),
        ) + (((pair_merge, 1),) if pair_merge is not None else ())
        all_kernel_multiplicities = (
            (embedding, 1),
            *(
                (kernel, QWEN_LAYERS * multiplicity)
                for kernel, multiplicity
                in per_layer_kernel_multiplicities
            ),
            (final_norm, 1),
            (lm_head, 1),
            (sampler, 1),
        )

        def weighted_sum(attribute: str) -> float | int:
            return sum(
                getattr(kernel, attribute) * multiplicity
                for kernel, multiplicity in all_kernel_multiplicities
            )

        dominant_counts = {
            name: sum(
                multiplicity
                for kernel, multiplicity in all_kernel_multiplicities
                if kernel.dominant_roof == name
            )
            for name in ("compute", "hbf_read", "lpddr")
        }
        attention_roof_totals = {
            "compute": QWEN_LAYERS * (
                attention.compute_roof_ns
                + (
                    pair_merge.compute_roof_ns
                    if pair_merge is not None else 0
                )
            ),
            "hbf_read": QWEN_LAYERS * (
                attention.hbf_roof_ns
                + (
                    pair_merge.hbf_roof_ns
                    if pair_merge is not None else 0
                )
            ),
            "lpddr": QWEN_LAYERS * (
                attention.lpddr_roof_ns
                + (
                    pair_merge.lpddr_roof_ns
                    if pair_merge is not None else 0
                )
            ),
        }
        attention_dominant_roof = max(
            attention_roof_totals, key=attention_roof_totals.get)
        total_ns = (
            embedding_ns
            + dense_ns
            + attention_ns
            + router_ns
            + moe_ns
            + final_ns
            + collective_ns
        )
        collective_bytes_per_rank = QWEN_LAYERS * (
            allreduce_bytes_per_layer
            + allgather_bytes_per_layer
            + reduce_scatter_bytes_per_layer
            + pair_query_bytes
            + pair_partial_bytes
        )

        return HBFModelBatchLatency(
            layout=self.layout.key,
            tp_size=tp,
            replicas=self.layout.replicas,
            total_ns=total_ns,
            embedding_ns=embedding_ns,
            dense_ns=dense_ns,
            attention_ns=attention_ns,
            router_ns=router_ns,
            moe_ns=moe_ns,
            final_ns=final_ns,
            collective_ns=collective_ns,
            tp_allreduce_ns=tp_allreduce_ns,
            ep_allgather_ns=ep_allgather_ns,
            ep_reduce_scatter_ns=ep_reduce_scatter_ns,
            hbf_read_bytes_per_rank=int(math.ceil(
                weighted_sum("hbf_read_bytes"))),
            lpddr_bytes_per_rank=int(math.ceil(
                weighted_sum("lpddr_bytes"))),
            collective_bytes_per_rank=collective_bytes_per_rank,
            compute_roof_ns_sum=int(weighted_sum("compute_roof_ns")),
            hbf_roof_ns_sum=int(weighted_sum("hbf_roof_ns")),
            lpddr_roof_ns_sum=int(weighted_sum("lpddr_roof_ns")),
            attention_compute_roof_ns=attention_roof_totals["compute"],
            attention_hbf_roof_ns=attention_roof_totals["hbf_read"],
            attention_lpddr_roof_ns=attention_roof_totals["lpddr"],
            attention_dominant_roof=attention_dominant_roof,
            dominant_kernel_counts=dominant_counts,
            pair_query_exchange_ns=pair_query_exchange_ns,
            pair_softmax_partial_exchange_ns=(
                pair_softmax_partial_exchange_ns),
            pair_attention_merge_ns=QWEN_LAYERS * (
                pair_merge.latency_ns
                if pair_merge is not None else 0
            ),
            pair_query_exchange_bytes_per_rank=(
                QWEN_LAYERS * pair_query_bytes),
            pair_softmax_partial_exchange_bytes_per_rank=(
                QWEN_LAYERS * pair_partial_bytes),
            pair_attention_merge_lpddr_bytes_per_rank=int(math.ceil(
                QWEN_LAYERS * (
                    pair_merge.lpddr_bytes
                    if pair_merge is not None else 0.0
                )
            )),
        )

    def _batch_latency_from_execution_plan(
            self, shape: HBFModelBatchShape) -> HBFModelBatchLatency:
        """Reference summary built by expanding and scanning the full plan."""

        plan = self.batch_execution_plan(shape)
        kernel_operations = plan.kernel_operations
        collective_operations = plan.collective_operations

        def kernel_latency(category: str) -> int:
            return sum(
                operation.latency_ns
                for operation in kernel_operations
                if operation.category == category
            )

        def collective_latency(category: str) -> int:
            return sum(
                operation.latency_ns
                for operation in collective_operations
                if operation.category == category
            )

        dominant_counts = {
            name: sum(
                operation.dominant_roof == name
                for operation in kernel_operations
            )
            for name in ("compute", "hbf_read", "lpddr")
        }
        attention_core_operations = tuple(
            operation
            for operation in kernel_operations
            if operation.category == "attention"
        )
        if len(attention_core_operations) != QWEN_LAYERS:
            raise AssertionError(
                "full-model plan must contain one attention kernel per layer")
        attention_merge_operations = tuple(
            operation
            for operation in kernel_operations
            if operation.category == "attention_merge"
        )
        expected_merges = (
            QWEN_LAYERS if self.layout.is_context_striped else 0)
        if len(attention_merge_operations) != expected_merges:
            raise AssertionError(
                "full-model plan has an invalid pair attention merge count")
        attention_operations = (
            attention_core_operations + attention_merge_operations)

        embedding_ns = kernel_latency("embedding")
        dense_ns = kernel_latency("dense")
        attention_ns = (
            kernel_latency("attention")
            + kernel_latency("attention_merge")
        )
        router_ns = kernel_latency("router")
        moe_ns = kernel_latency("moe")
        final_ns = kernel_latency("final")
        tp_allreduce_ns = collective_latency("tp_allreduce")
        ep_allgather_ns = collective_latency("ep_allgather")
        ep_reduce_scatter_ns = collective_latency(
            "ep_reduce_scatter")
        pair_query_exchange_ns = collective_latency(
            "pair_query_exchange")
        pair_softmax_partial_exchange_ns = collective_latency(
            "pair_softmax_partial_exchange")
        collective_ns = sum(
            operation.latency_ns
            for operation in collective_operations
        )
        attention_roof_totals = {
            "compute": sum(
                operation.compute_roof_ns
                for operation in attention_operations),
            "hbf_read": sum(
                operation.hbf_roof_ns
                for operation in attention_operations),
            "lpddr": sum(
                operation.lpddr_roof_ns
                for operation in attention_operations),
        }
        attention_dominant_roof = max(
            attention_roof_totals, key=attention_roof_totals.get)
        total_ns = plan.total_ns
        return HBFModelBatchLatency(
            layout=plan.layout,
            tp_size=plan.tp_size,
            replicas=plan.replicas,
            total_ns=total_ns,
            embedding_ns=embedding_ns,
            dense_ns=dense_ns,
            attention_ns=attention_ns,
            router_ns=router_ns,
            moe_ns=moe_ns,
            final_ns=final_ns,
            collective_ns=collective_ns,
            tp_allreduce_ns=tp_allreduce_ns,
            ep_allgather_ns=ep_allgather_ns,
            ep_reduce_scatter_ns=ep_reduce_scatter_ns,
            hbf_read_bytes_per_rank=int(math.ceil(sum(
                operation.hbf_read_bytes_per_rank
                for operation in kernel_operations
            ))),
            lpddr_bytes_per_rank=int(math.ceil(sum(
                operation.lpddr_bytes_per_rank
                for operation in kernel_operations
            ))),
            collective_bytes_per_rank=sum(
                operation.transferred_bytes_per_rank
                for operation in collective_operations
            ),
            compute_roof_ns_sum=sum(
                operation.compute_roof_ns
                for operation in kernel_operations),
            hbf_roof_ns_sum=sum(
                operation.hbf_roof_ns
                for operation in kernel_operations),
            lpddr_roof_ns_sum=sum(
                operation.lpddr_roof_ns
                for operation in kernel_operations),
            attention_compute_roof_ns=sum(
                operation.compute_roof_ns
                for operation in attention_operations),
            attention_hbf_roof_ns=sum(
                operation.hbf_roof_ns
                for operation in attention_operations),
            attention_lpddr_roof_ns=sum(
                operation.lpddr_roof_ns
                for operation in attention_operations),
            attention_dominant_roof=attention_dominant_roof,
            dominant_kernel_counts=dominant_counts,
            pair_query_exchange_ns=pair_query_exchange_ns,
            pair_softmax_partial_exchange_ns=(
                pair_softmax_partial_exchange_ns),
            pair_attention_merge_ns=kernel_latency(
                "attention_merge"),
            pair_query_exchange_bytes_per_rank=sum(
                operation.transferred_bytes_per_rank
                for operation in collective_operations
                if operation.category == "pair_query_exchange"
            ),
            pair_softmax_partial_exchange_bytes_per_rank=sum(
                operation.transferred_bytes_per_rank
                for operation in collective_operations
                if operation.category == "pair_softmax_partial_exchange"
            ),
            pair_attention_merge_lpddr_bytes_per_rank=int(math.ceil(sum(
                operation.lpddr_bytes_per_rank
                for operation in attention_merge_operations
            ))),
        )

    def logical_hbf_capacity_bytes_per_replica(
            self, *, model_weight_bytes: int) -> int:
        if model_weight_bytes < 0:
            raise ValueError("model_weight_bytes must be non-negative")
        per_rank_weight = int(math.ceil(
            model_weight_bytes / self.layout.tp_size))
        per_rank_free = (
            self.hardware.hbf_capacity_bytes_per_card - per_rank_weight)
        if per_rank_free <= 0:
            return 0
        return int(
            per_rank_free
            * self.layout.tp_size
            / self.layout.physical_kv_replication_factor
        )

    def logical_hbf_capacity_bytes_server(
            self, *, model_weight_bytes: int) -> int:
        return (
            self.logical_hbf_capacity_bytes_per_replica(
                model_weight_bytes=model_weight_bytes)
            * self.layout.replicas
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "execution_scope": "full_transformer_end_to_end",
            "layout": asdict(self.layout),
            "hardware": asdict(self.hardware),
            "compute_device": {
                "kind": "h100_class_gpu",
                "peak_tflops_per_card": (
                    self.hardware.gpu_peak_tflops_per_card),
                "calibration_source": "h100_tp4_kernel_families",
            },
            "hbf_read_access": {
                "configured_fixed_latency_us": (
                    self.hardware.hbf_read_latency_us),
                "prefetch_enabled": (
                    self.hardware.hbf_read_prefetch_enabled),
                "exposed_fixed_latency_us_per_hbf_touching_kernel": (
                    0.0
                    if self.hardware.hbf_read_prefetch_enabled
                    else self.hardware.hbf_read_latency_us
                ),
                "bandwidth_cost_retained_with_prefetch": True,
                "semantics": (
                    "perfect_prefetch_hides_fixed_latency"
                    if self.hardware.hbf_read_prefetch_enabled
                    else "demand_read_charges_fixed_latency_once"
                ),
            },
            "weight_location": "hbf",
            "committed_kv_location": "hbf",
            "activation_location": "lpddr",
            "active_turn_kv_location": "lpddr",
            "collectives_included": True,
            "collective_fixed_latency_semantics": (
                COLLECTIVE_FIXED_LATENCY_SEMANTICS),
            "kv_layout_semantics": self.layout.kv_layout_semantics,
            "context_parallel_attention": (
                {
                    "enabled": True,
                    "kv_head_pairs": 4,
                    "cards_per_kv_head": 2,
                    "card_pairs": (
                        (0, 1), (2, 3), (4, 5), (6, 7)),
                    "physical_kv_replication_factor": 1,
                    "partition": (
                        "sequence-local even/odd token zigzag; "
                        "odd-token rank asymmetry preserved"),
                    "attention_q_heads_per_rank": 8,
                    "dense_q_heads_per_rank": 4,
                    "query_exchange_dtype": "bf16",
                    "softmax_partial": "fp32_(O,max,sum)",
                    "partial_exchange": (
                        "only peer-owned q4 partials per rank"),
                }
                if self.layout.is_context_striped
                else {"enabled": False}
            ),
            "calibration": (
                "legacy H100 TP4 kernel-family residuals with rebuilt "
                "TP1/TP4/TP8 per-rank work"
            ),
        }


def _balanced_lengths(total: int, count: int) -> tuple[int, ...]:
    if total < 0 or count <= 0:
        raise ValueError("balanced lengths require total >= 0 and count > 0")
    quotient, remainder = divmod(total, count)
    return tuple(
        quotient + (1 if index < remainder else 0)
        for index in range(count)
    )


def load_hbf_server_config(
        path: Path,
) -> tuple[HBFServerHardware, Mapping[str, HBFParallelLayout]]:
    with path.open("r", encoding="utf-8") as source:
        raw = json.load(source)
    if not isinstance(raw, dict):
        raise ValueError("HBF full-model config must be a JSON object")
    _strict_fields(
        raw, {"schema_version", "hardware", "layouts"}, "top-level")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"expected schema_version={SCHEMA_VERSION}")
    hardware_raw = raw.get("hardware")
    layouts_raw = raw.get("layouts")
    if not isinstance(hardware_raw, dict):
        raise ValueError("hardware must be an object")
    if not isinstance(layouts_raw, dict) or not layouts_raw:
        raise ValueError("layouts must be a non-empty object")
    hardware = HBFServerHardware.from_dict(hardware_raw)
    layouts = {}
    for key, item in layouts_raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"layout {key!r} must be an object")
        _strict_fields(item, {"tp_size", "replicas"}, f"layout {key}")
        layout = HBFParallelLayout(
            key=str(key),
            tp_size=item.get("tp_size"),
            replicas=item.get("replicas"),
        )
        layout.validate(hardware.card_count)
        HBFPCIeTopology.from_hardware(
            hardware, server_id=0).validate_layout(
                layout_key=layout.key,
                tp_size=layout.tp_size,
                replicas=layout.replicas,
            )
        layouts[str(key)] = layout
    return hardware, layouts


def build_full_model_hbf_latency(
        *, repo_root: Path, hardware: HBFServerHardware,
        layout: HBFParallelLayout, band: str = "central",
) -> FullModelHBFLatencyModel:
    model_config = repo_root / DEFAULT_MODEL_CONFIG
    if not model_config.is_file():
        raise FileNotFoundError(
            f"missing target model config: {model_config}")
    digest = hashlib.sha256(model_config.read_bytes()).hexdigest()
    provider = build_online_latency_model(
        H100_QWEN3_TP4_KERNEL_CALIBRATED,
        str(repo_root.resolve()),
        digest,
        band,
    )
    return FullModelHBFLatencyModel(
        base_provider=provider,
        hardware=hardware,
        layout=layout,
    )


__all__ = [
    "COLLECTIVE_FIXED_LATENCY_SEMANTICS",
    "FullModelHBFLatencyModel",
    "HBFCollectiveExecutionOp",
    "HBFContextAttentionRankExecution",
    "HBFExecutionOp",
    "HBFKernelExecutionOp",
    "HBFModelBatchExecutionPlan",
    "HBFModelBatchLatency",
    "HBFModelBatchShape",
    "HBFParallelLayout",
    "HBFServerHardware",
    "ONLINE_SOFTMAX_ACCUMULATOR_BYTES",
    "ONLINE_SOFTMAX_STATE_ELEMENTS",
    "build_full_model_hbf_latency",
    "load_hbf_server_config",
    "qwen_logical_kv_bytes_per_token",
    "qwen_model_weight_bytes_per_rank",
]

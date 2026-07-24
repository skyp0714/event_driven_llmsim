"""Standalone analytical model for agentic idle-KV overheads.

This module deliberately has no dependency on the simulator runtime.  It can
therefore be used while ASTRA-Sim, vLLM, or the legacy profiler environment is
unavailable.  The estimates are intended for sensitivity analysis and for
identifying measurements that are still needed; they are not a replacement for
TP8 kernel measurements.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SI_GB = 1_000_000_000.0
SI_TB = 1_000_000_000_000.0


class AnalysisConfigError(ValueError):
    """Raised when an analytical-model input is invalid."""


@dataclass(frozen=True)
class CpuTransferSpec:
    """GPU/CPU transfer limits, expressed as unidirectional SI GB/s."""

    gpu_to_host_gbps_per_rank: float = 50.0
    host_to_gpu_gbps_per_rank: float = 50.0
    dram_write_gbps_aggregate: float = 400.0
    dram_read_gbps_aggregate: float = 400.0
    fixed_latency_us: float = 5.0
    provenance: str = (
        "Uncalibrated analytical default: 50 GB/s effective per-rank PCIe "
        "and 400 GB/s aggregate host DRAM. Replace with platform memcpy data."
    )


@dataclass(frozen=True)
class SsdTransferSpec:
    """Aggregate SSD limits for a CPU-staged transfer path."""

    write_gbps_aggregate: float = 10.0
    read_gbps_aggregate: float = 14.0
    fixed_latency_us: float = 20.0
    staged_through_cpu: bool = True
    provenance: str = (
        "Uncalibrated single-device analytical default. Replace with fio or "
        "GDS measurements at the KV object sizes and queue depths of interest."
    )


@dataclass(frozen=True)
class HardwareSpec:
    """GPU roofline and storage-path assumptions for one platform."""

    name: str
    bf16_dense_tflops: float
    hbm_bandwidth_tbps: float
    compute_efficiency: float = 0.55
    memory_efficiency: float = 0.75
    launch_overhead_us_per_layer: float = 2.0
    cpu: CpuTransferSpec = CpuTransferSpec()
    ssd: SsdTransferSpec = SsdTransferSpec()
    nominal_provenance: str = ""
    calibration_provenance: str = (
        "No repository TP8 calibration applied; efficiency factors are explicit "
        "analytical assumptions."
    )

    def validate(self) -> None:
        positive = {
            "bf16_dense_tflops": self.bf16_dense_tflops,
            "hbm_bandwidth_tbps": self.hbm_bandwidth_tbps,
            "compute_efficiency": self.compute_efficiency,
            "memory_efficiency": self.memory_efficiency,
            "cpu.gpu_to_host_gbps_per_rank": self.cpu.gpu_to_host_gbps_per_rank,
            "cpu.host_to_gpu_gbps_per_rank": self.cpu.host_to_gpu_gbps_per_rank,
            "cpu.dram_write_gbps_aggregate": self.cpu.dram_write_gbps_aggregate,
            "cpu.dram_read_gbps_aggregate": self.cpu.dram_read_gbps_aggregate,
            "ssd.write_gbps_aggregate": self.ssd.write_gbps_aggregate,
            "ssd.read_gbps_aggregate": self.ssd.read_gbps_aggregate,
        }
        for field_name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise AnalysisConfigError(f"{field_name} must be positive")
        if self.compute_efficiency > 1 or self.memory_efficiency > 1:
            raise AnalysisConfigError("roofline efficiencies must be in (0, 1]")
        if self.launch_overhead_us_per_layer < 0:
            raise AnalysisConfigError("launch overhead cannot be negative")
        if self.cpu.fixed_latency_us < 0 or self.ssd.fixed_latency_us < 0:
            raise AnalysisConfigError("transfer fixed latency cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_HARDWARE_SPECS: dict[str, HardwareSpec] = {
    "H100": HardwareSpec(
        name="H100",
        # NVIDIA publishes 1,979 BF16 TFLOP/s with sparsity for H100 SXM.
        # The dense value below removes the advertised 2x sparsity factor.
        bf16_dense_tflops=989.5,
        hbm_bandwidth_tbps=3.35,
        nominal_provenance=(
            "NVIDIA H100 SXM public specifications; BF16 dense peak inferred "
            "as half of the 1,979 TFLOP/s with-sparsity figure; HBM 3.35 TB/s. "
            "https://www.nvidia.com/en-us/data-center/h100/"
        ),
    ),
    "H200": HardwareSpec(
        name="H200",
        # H200 uses the same Hopper compute specification as H100 SXM but HBM3e.
        bf16_dense_tflops=989.5,
        hbm_bandwidth_tbps=4.8,
        nominal_provenance=(
            "NVIDIA H200 SXM public specifications; BF16 dense peak inferred "
            "as half of the 1,979 TFLOP/s with-sparsity figure; HBM 4.8 TB/s. "
            "https://www.nvidia.com/en-us/data-center/h200/"
        ),
    ),
}


@dataclass(frozen=True)
class ModelShape:
    name: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    model_type: str = ""
    config_path: str = ""

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0 and self.num_experts_per_tok > 0

    @classmethod
    def from_config_file(cls, name: str, path: Path) -> "ModelShape":
        with path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        hidden_size = _positive_int(config, "hidden_size")
        num_heads = _positive_int(config, "num_attention_heads")
        head_dim = int(config.get("head_dim") or hidden_size // num_heads)
        intermediate = _positive_int(config, "intermediate_size")
        num_experts = int(
            config.get("num_local_experts", config.get("num_experts", 0)) or 0
        )
        top_k = int(config.get("num_experts_per_tok", 0) or 0)
        moe_intermediate = int(
            config.get("moe_intermediate_size", intermediate) or intermediate
        )
        shape = cls(
            name=name,
            hidden_size=hidden_size,
            num_hidden_layers=_positive_int(config, "num_hidden_layers"),
            num_attention_heads=num_heads,
            num_key_value_heads=int(
                config.get("num_key_value_heads", num_heads) or num_heads
            ),
            head_dim=head_dim,
            intermediate_size=intermediate,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            moe_intermediate_size=moe_intermediate,
            model_type=str(config.get("model_type", "")),
            config_path=str(path),
        )
        shape.validate()
        return shape

    def validate(self) -> None:
        values = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
        }
        for field_name, value in values.items():
            if value <= 0:
                raise AnalysisConfigError(
                    f"model {self.name}: {field_name} must be positive"
                )
        if self.is_moe and self.moe_intermediate_size <= 0:
            raise AnalysisConfigError(
                f"model {self.name}: MoE intermediate size must be positive"
            )


@dataclass(frozen=True)
class KvLayout:
    dtype_bytes: int
    tp_size: int
    logical_bytes_per_token: int
    physical_bytes_per_token_per_rank: int
    physical_bytes_per_token_cluster: int
    kv_heads_per_rank: float
    replication_factor: float
    mode: str
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolTransition:
    session_id: str
    sub_request_index: int
    tool_duration_ns: int
    cache_tokens_declared: int
    next_input_tokens: int
    observed_lcp_tokens: int
    declared_reuse_tokens: int
    effective_reuse_tokens: int
    reusable_allocation_tokens: int
    reuse_source: str
    token_identity_verified: bool
    previous_token_id_coverage: float
    next_token_id_coverage: float


@dataclass(frozen=True)
class WorkloadSummary:
    path: str
    sha256: str
    sessions: int
    sub_requests: int
    adjacent_transitions: int
    positive_tool_transitions: int
    zero_tool_transitions: int
    selected_tool_transitions: int
    transitions_excluded_context: int
    max_context_tokens_filter: int | None
    transitions_without_token_identity: int
    reuse_source_counts: Mapping[str, int]
    transitions: tuple[ToolTransition, ...]

    def metadata_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("transitions")
        return result


@dataclass(frozen=True)
class RooflineEstimate:
    tokens: int
    total_flops: float
    flops_per_rank: float
    hbm_bytes_per_rank: float
    compute_seconds: float
    memory_seconds: float
    launch_seconds: float
    total_seconds: float
    limiting_term: str
    expected_unique_experts_per_layer: float
    prefill_chunks: int


def _positive_int(mapping: Mapping[str, Any], key: str) -> int:
    try:
        value = int(mapping[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisConfigError(f"missing or invalid model field: {key}") from exc
    if value <= 0:
        raise AnalysisConfigError(f"model field {key} must be positive")
    return value


def repo_root_from_module() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_model_config(model_name: str, repo_root: Path | None = None) -> Path:
    root = repo_root or repo_root_from_module()
    path = root / "configs" / "model" / f"{model_name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"model config for {model_name!r} not found at {path}"
        )
    return path


def load_model_shape(
    model_name: str, repo_root: Path | None = None
) -> ModelShape:
    path = resolve_model_config(model_name, repo_root)
    return ModelShape.from_config_file(model_name, path)


def kv_layout(
    model: ModelShape,
    tp_size: int,
    dtype_bytes: int = 2,
    mode: str = "replicated",
) -> KvLayout:
    """Return logical and physical KV bytes per token.

    ``replicated`` models common tensor-parallel GQA behavior: if TP exceeds
    the number of KV heads, each rank still owns at least one KV head.  This
    makes Qwen3-30B-A3B at TP8 consume twice its logical KV size. The
    ``logical-even`` sensitivity ignores that replication. ``simulator-even``
    remains as a deprecated alias for artifacts produced before the runtime
    memory model was corrected.
    """

    if tp_size <= 0 or dtype_bytes <= 0:
        raise AnalysisConfigError("tp_size and dtype_bytes must be positive")
    logical = (
        2
        * model.num_key_value_heads
        * model.head_dim
        * model.num_hidden_layers
        * dtype_bytes
    )
    warning = ""
    if mode == "replicated":
        if model.num_key_value_heads >= tp_size:
            heads_per_rank = math.ceil(model.num_key_value_heads / tp_size)
            if model.num_key_value_heads % tp_size:
                warning = (
                    "KV heads do not divide TP exactly; ceil-based physical "
                    "layout is an upper bound and must be checked against the engine."
                )
        else:
            heads_per_rank = 1
            warning = (
                "TP exceeds KV-head count; physical KV-head replication is "
                "included; a logical-even sensitivity would undercount it."
            )
        per_rank = (
            2
            * heads_per_rank
            * model.head_dim
            * model.num_hidden_layers
            * dtype_bytes
        )
        physical_cluster = per_rank * tp_size
        reported_heads = float(heads_per_rank)
    elif mode in {"logical-even", "simulator-even"}:
        per_rank = logical // tp_size
        physical_cluster = per_rank * tp_size
        reported_heads = model.num_key_value_heads / tp_size
        if model.num_key_value_heads < tp_size:
            warning = (
                "Logical-even sensitivity permits fractional KV heads and can "
                "underestimate physical TP8 KV storage."
            )
    else:
        raise AnalysisConfigError(
            "kv layout mode must be 'replicated' or 'logical-even'"
        )
    return KvLayout(
        dtype_bytes=dtype_bytes,
        tp_size=tp_size,
        logical_bytes_per_token=logical,
        physical_bytes_per_token_per_rank=per_rank,
        physical_bytes_per_token_cluster=physical_cluster,
        kv_heads_per_rank=reported_heads,
        replication_factor=physical_cluster / logical,
        mode=mode,
        warning=warning,
    )


def _longest_common_prefix(left: Sequence[Any], right: Sequence[Any]) -> int:
    count = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        count += 1
    return count


def load_agentic_workload(
    path: Path,
    block_size: int = 16,
    include_zero_tool_duration: bool = False,
    max_context_tokens: int | None = None,
) -> WorkloadSummary:
    """Load adjacent tool-call transitions from an agentic JSONL workload."""

    if block_size <= 0:
        raise AnalysisConfigError("block_size must be positive")
    if max_context_tokens is not None and max_context_tokens <= 0:
        raise AnalysisConfigError("max_context_tokens must be positive when set")
    raw = path.read_bytes()
    sessions = 0
    sub_requests = 0
    adjacent = 0
    positive = 0
    zero = 0
    context_excluded = 0
    missing_identity = 0
    reuse_source_counts: dict[str, int] = {}
    transitions: list[ToolTransition] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalysisConfigError(
                f"{path}:{line_number}: invalid JSON: {exc}"
            ) from exc
        requests = record.get("sub_requests")
        if not isinstance(requests, list):
            continue
        sessions += 1
        sub_requests += len(requests)
        session_id = str(record.get("session_id", f"line-{line_number}"))
        for index in range(len(requests) - 1):
            current = requests[index]
            following = requests[index + 1]
            adjacent += 1
            tool_ns = int(current.get("tool_duration_ns", 0) or 0)
            if tool_ns < 0:
                raise AnalysisConfigError(
                    f"{path}:{line_number}: tool_duration_ns cannot be negative"
                )
            if tool_ns == 0:
                zero += 1
                if not include_zero_tool_duration:
                    continue
            else:
                positive += 1
            input_tokens = int(current.get("input_toks", 0) or 0)
            output_tokens = int(current.get("output_toks", 0) or 0)
            next_input_tokens = int(following.get("input_toks", 0) or 0)
            if input_tokens <= 0 or output_tokens < 0 or next_input_tokens <= 0:
                raise AnalysisConfigError(
                    f"{path}:{line_number}: invalid token count at sub-request {index}"
                )
            cache_tokens = max(0, input_tokens + output_tokens - 1)
            if (max_context_tokens is not None
                    and (cache_tokens > max_context_tokens
                         or next_input_tokens > max_context_tokens)):
                context_excluded += 1
                continue
            input_ids = current.get("input_tok_ids")
            output_ids = current.get("output_tok_ids")
            next_ids = following.get("input_tok_ids")
            identity_verified = (
                isinstance(input_ids, list)
                and isinstance(output_ids, list)
                and isinstance(next_ids, list)
                and bool(input_ids)
                and bool(next_ids)
            )
            observed_lcp = 0
            declared_reuse = 0
            effective_reuse = 0
            reuse_source = "unavailable"
            previous_coverage = 0.0
            next_coverage = 0.0
            if identity_verified:
                # Input IDs all have KV entries. Of the generated IDs, only
                # the first ``output_toks - 1`` do: the final generated token
                # has not yet passed back through the model. Do not blindly
                # drop the last *available* ID, because trace fixtures may
                # omit output IDs entirely; doing so would incorrectly remove
                # a verified input token and can invalidate durable lineage.
                previous_ids = list(input_ids)[:input_tokens]
                # Output IDs start after the declared input length. If the
                # input-ID array is truncated, concatenating outputs would
                # collapse that positional gap and could manufacture a false
                # longer common prefix when token values happen to match.
                if len(previous_ids) == input_tokens:
                    previous_ids += list(output_ids)[:max(0, output_tokens - 1)]
                observed_lcp = min(
                    _longest_common_prefix(previous_ids, next_ids),
                    cache_tokens,
                    next_input_tokens,
                )
                declared_reuse = observed_lcp
                effective_reuse = min(
                    declared_reuse, max(0, next_input_tokens - 1))
                reuse_source = "token_ids_exact"
                previous_coverage = min(1.0, len(previous_ids) / cache_tokens) if cache_tokens else 1.0
                next_coverage = min(1.0, len(next_ids) / next_input_tokens)
            else:
                missing_identity += 1
                explicit_reuse = following.get("prefix_reuse_toks")
                if explicit_reuse is not None:
                    try:
                        explicit_reuse = int(explicit_reuse)
                    except (TypeError, ValueError) as exc:
                        raise AnalysisConfigError(
                            f"{path}:{line_number}: invalid prefix_reuse_toks "
                            f"at sub-request {index + 1}"
                        ) from exc
                    if explicit_reuse < 0:
                        raise AnalysisConfigError(
                            f"{path}:{line_number}: prefix_reuse_toks cannot be negative"
                        )
                    declared_reuse = min(
                        explicit_reuse, cache_tokens, next_input_tokens)
                    effective_reuse = min(
                        declared_reuse, max(0, next_input_tokens - 1))
                    declared_source = str(
                        following.get("prefix_reuse_source") or "reported")
                    reuse_source = f"explicit_{declared_source}"
            # Logical hit tokens remain exact. Physical storage transfers the
            # page containing the partial tail, hence ceil block allocation.
            reusable_allocation = (
                (effective_reuse + block_size - 1) // block_size * block_size
                if effective_reuse else 0)
            reuse_source_counts[reuse_source] = (
                reuse_source_counts.get(reuse_source, 0) + 1)
            transitions.append(
                ToolTransition(
                    session_id=session_id,
                    sub_request_index=index,
                    tool_duration_ns=tool_ns,
                    cache_tokens_declared=cache_tokens,
                    next_input_tokens=next_input_tokens,
                    observed_lcp_tokens=observed_lcp,
                    declared_reuse_tokens=declared_reuse,
                    effective_reuse_tokens=effective_reuse,
                    reusable_allocation_tokens=reusable_allocation,
                    reuse_source=reuse_source,
                    token_identity_verified=identity_verified,
                    previous_token_id_coverage=previous_coverage,
                    next_token_id_coverage=next_coverage,
                )
            )
    if sessions == 0:
        raise AnalysisConfigError(f"{path} contains no agentic sessions")
    if not transitions:
        raise AnalysisConfigError(
            f"{path} contains no selected adjacent tool-call transitions"
        )
    return WorkloadSummary(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        sessions=sessions,
        sub_requests=sub_requests,
        adjacent_transitions=adjacent,
        positive_tool_transitions=positive,
        zero_tool_transitions=zero,
        selected_tool_transitions=len(transitions),
        transitions_excluded_context=context_excluded,
        max_context_tokens_filter=max_context_tokens,
        transitions_without_token_identity=missing_identity,
        reuse_source_counts=dict(sorted(reuse_source_counts.items())),
        transitions=tuple(transitions),
    )


def cpu_transfer_seconds(
    cluster_bytes: int,
    per_rank_bytes: int,
    spec: CpuTransferSpec,
    direction: str,
) -> float:
    """Model parallel rank copies constrained by both link and host DRAM."""

    if cluster_bytes <= 0:
        return 0.0
    if per_rank_bytes <= 0:
        raise AnalysisConfigError("per-rank bytes must be positive")
    if direction == "out":
        rank_bw = spec.gpu_to_host_gbps_per_rank
        aggregate_bw = spec.dram_write_gbps_aggregate
    elif direction == "in":
        rank_bw = spec.host_to_gpu_gbps_per_rank
        aggregate_bw = spec.dram_read_gbps_aggregate
    else:
        raise AnalysisConfigError("CPU transfer direction must be 'out' or 'in'")
    return spec.fixed_latency_us * 1e-6 + max(
        per_rank_bytes / (rank_bw * SI_GB),
        cluster_bytes / (aggregate_bw * SI_GB),
    )


def ssd_media_seconds(
    byte_count: int, spec: SsdTransferSpec, direction: str
) -> float:
    if byte_count <= 0:
        return 0.0
    if direction == "out":
        bandwidth = spec.write_gbps_aggregate
    elif direction == "in":
        bandwidth = spec.read_gbps_aggregate
    else:
        raise AnalysisConfigError("SSD transfer direction must be 'out' or 'in'")
    return spec.fixed_latency_us * 1e-6 + byte_count / (bandwidth * SI_GB)


def ssd_transfer_seconds(
    cluster_bytes: int,
    per_rank_bytes: int,
    hardware: HardwareSpec,
    direction: str,
) -> float:
    media = ssd_media_seconds(cluster_bytes, hardware.ssd, direction)
    if not hardware.ssd.staged_through_cpu:
        return media
    return media + cpu_transfer_seconds(
        cluster_bytes, per_rank_bytes, hardware.cpu, direction
    )


def _partial_service_bytes(
    byte_count: int, active_seconds: float, service_seconds: float
) -> int:
    """Return bytes issued by a cancelled proportional-service transfer.

    The standalone analyzer has no request/chunk-level storage trace, so a
    write cancelled after it starts uses the same explicit lower-level
    assumption as the online model: progress is linear over isolated service
    time.  Rounding up records any positive issued fraction while the result is
    always capped at the complete object size.
    """

    if byte_count <= 0 or active_seconds <= 0 or service_seconds <= 0:
        return 0
    fraction = min(1.0, active_seconds / service_seconds)
    return min(byte_count, int(math.ceil(byte_count * fraction)))


def _expected_unique_experts(num_experts: int, routes: int) -> float:
    if num_experts <= 0 or routes <= 0:
        return 0.0
    # Stable form of E * (1 - (1 - 1/E)^routes), assuming uniform routing.
    return num_experts * (-math.expm1(routes * math.log1p(-1.0 / num_experts)))


def roofline_recompute_seconds(
    model: ModelShape,
    hardware: HardwareSpec,
    tokens: int,
    tp_size: int,
    dtype_bytes: int = 2,
    kv: KvLayout | None = None,
    prefill_chunk_size: int = 2048,
) -> RooflineEstimate:
    """Estimate one causal prefill using a calibrated aggregate roofline.

    Transformer GEMM FLOPs and causal QK/PV FLOPs are included.  The memory
    term is a lower-bound tensor-traffic model and the result excludes TP
    collectives, CPU launch gaps, routing imbalance, and mixed-batch effects.
    """

    hardware.validate()
    if tokens <= 0:
        return RooflineEstimate(
            0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            "none", 0.0, 0)
    if tp_size <= 0 or dtype_bytes <= 0 or prefill_chunk_size <= 0:
        raise AnalysisConfigError(
            "tp_size, dtype_bytes, and prefill_chunk_size must be positive")
    layout = kv or kv_layout(model, tp_size, dtype_bytes)
    hidden = model.hidden_size
    q_dim = model.q_dim
    kv_dim = model.kv_dim
    layers = model.num_hidden_layers
    prefill_chunks = math.ceil(tokens / prefill_chunk_size)

    q_dim_per_rank = q_dim / tp_size
    physical_kv_dim_per_rank = layout.kv_heads_per_rank * model.head_dim
    attention_weight_elements_per_rank = (
        hidden * (q_dim_per_rank + 2 * physical_kv_dim_per_rank)
        + q_dim_per_rank * hidden
    )
    attention_linear_flops_per_token_per_rank = (
        2.0 * attention_weight_elements_per_rank
    )
    expected_experts = 0.0
    if model.is_moe:
        expected_experts = _expected_unique_experts(
            model.num_experts, tokens * model.num_experts_per_tok
        )
        ffn_flops_per_token_cluster = (
            6.0
            * hidden
            * model.moe_intermediate_size
            * model.num_experts_per_tok
            + 2.0 * hidden * model.num_experts
        )
        ffn_weight_elements_cluster = (
            hidden * model.num_experts
            + expected_experts
            * 3.0
            * hidden
            * model.moe_intermediate_size
        )
        activation_elements_per_token = (
            2 * hidden
            + 2 * q_dim
            + 2 * kv_dim
            + model.num_experts
            + model.num_experts_per_tok
            * (hidden + 3 * model.moe_intermediate_size)
        )
    else:
        ffn_flops_per_token_cluster = 6.0 * hidden * model.intermediate_size
        ffn_weight_elements_cluster = 3.0 * hidden * model.intermediate_size
        activation_elements_per_token = (
            4 * hidden + 2 * q_dim + 2 * kv_dim + 3 * model.intermediate_size
        )

    linear_flops_per_rank = (
        tokens
        * layers
        * (
            attention_linear_flops_per_token_per_rank
            + ffn_flops_per_token_cluster / tp_size
        )
    )
    # Causal pairs = T(T+1)/2; QK and PV each require two FLOPs per element.
    attention_flops_per_rank = (
        2.0 * q_dim_per_rank * tokens * (tokens + 1) * layers
    )
    flops_per_rank = linear_flops_per_rank + attention_flops_per_rank
    # This includes physical duplicated KV-projection work when TP exceeds the
    # number of KV heads, rather than reporting only logical model FLOPs.
    total_flops = flops_per_rank * tp_size

    active_weight_bytes_per_rank = (
        layers
        * (
            attention_weight_elements_per_rank
            + ffn_weight_elements_cluster / tp_size
        )
        * dtype_bytes
        * prefill_chunks
    )
    activation_bytes_cluster = (
        tokens * layers * activation_elements_per_token * dtype_bytes
    )
    # Activations are treated as ideally sharded. Each chunk reads at least
    # its full KV context once; FlashAttention tiling can read more, so this
    # remains a lower-bound memory term. Physical GQA replication is retained.
    chunk_context_tokens = sum(
        min(tokens, chunk_start + prefill_chunk_size)
        for chunk_start in range(0, tokens, prefill_chunk_size)
    )
    hbm_bytes_per_rank = (
        active_weight_bytes_per_rank
        + activation_bytes_cluster / tp_size
        + chunk_context_tokens * layout.physical_bytes_per_token_per_rank
    )
    compute_seconds = flops_per_rank / (
        hardware.bf16_dense_tflops * 1e12 * hardware.compute_efficiency
    )
    memory_seconds = hbm_bytes_per_rank / (
        hardware.hbm_bandwidth_tbps * SI_TB * hardware.memory_efficiency
    )
    launch_seconds = (
        layers * prefill_chunks * hardware.launch_overhead_us_per_layer * 1e-6)
    limiting = "compute" if compute_seconds >= memory_seconds else "memory"
    total_seconds = launch_seconds + max(compute_seconds, memory_seconds)
    return RooflineEstimate(
        tokens=tokens,
        total_flops=total_flops,
        flops_per_rank=flops_per_rank,
        hbm_bytes_per_rank=hbm_bytes_per_rank,
        compute_seconds=compute_seconds,
        memory_seconds=memory_seconds,
        launch_seconds=launch_seconds,
        total_seconds=total_seconds,
        limiting_term=limiting,
        expected_unique_experts_per_layer=expected_experts,
        prefill_chunks=prefill_chunks,
    )


def roofline_cached_prefill_seconds(
    model: ModelShape,
    hardware: HardwareSpec,
    total_tokens: int,
    cached_tokens: int,
    tp_size: int,
    dtype_bytes: int = 2,
    kv: KvLayout | None = None,
    prefill_chunk_size: int = 2048,
) -> RooflineEstimate:
    """Estimate suffix prefill when ``cached_tokens`` KV tokens already exist.

    Unlike subtracting two complete-prompt rooflines, this retains the weight
    reads and kernel launches needed for the uncached suffix.  Attention work
    uses the exact causal-pair delta, and every suffix chunk reads its cached
    prefix plus the suffix produced up to that chunk.
    """

    hardware.validate()
    if total_tokens < 0 or cached_tokens < 0 or cached_tokens > total_tokens:
        raise AnalysisConfigError(
            "cached prefill requires 0 <= cached_tokens <= total_tokens"
        )
    new_tokens = total_tokens - cached_tokens
    if new_tokens == 0:
        return RooflineEstimate(
            0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            "none", 0.0, 0,
        )
    if tp_size <= 0 or dtype_bytes <= 0 or prefill_chunk_size <= 0:
        raise AnalysisConfigError(
            "tp_size, dtype_bytes, and prefill_chunk_size must be positive"
        )
    layout = kv or kv_layout(model, tp_size, dtype_bytes)
    hidden = model.hidden_size
    q_dim = model.q_dim
    kv_dim = model.kv_dim
    layers = model.num_hidden_layers
    prefill_chunks = math.ceil(new_tokens / prefill_chunk_size)

    q_dim_per_rank = q_dim / tp_size
    physical_kv_dim_per_rank = layout.kv_heads_per_rank * model.head_dim
    attention_weight_elements_per_rank = (
        hidden * (q_dim_per_rank + 2 * physical_kv_dim_per_rank)
        + q_dim_per_rank * hidden
    )
    attention_linear_flops_per_token_per_rank = (
        2.0 * attention_weight_elements_per_rank
    )
    expected_experts = 0.0
    if model.is_moe:
        expected_experts = _expected_unique_experts(
            model.num_experts, new_tokens * model.num_experts_per_tok
        )
        ffn_flops_per_token_cluster = (
            6.0
            * hidden
            * model.moe_intermediate_size
            * model.num_experts_per_tok
            + 2.0 * hidden * model.num_experts
        )
        ffn_weight_elements_cluster = (
            hidden * model.num_experts
            + expected_experts
            * 3.0
            * hidden
            * model.moe_intermediate_size
        )
        activation_elements_per_token = (
            2 * hidden
            + 2 * q_dim
            + 2 * kv_dim
            + model.num_experts
            + model.num_experts_per_tok
            * (hidden + 3 * model.moe_intermediate_size)
        )
    else:
        ffn_flops_per_token_cluster = 6.0 * hidden * model.intermediate_size
        ffn_weight_elements_cluster = 3.0 * hidden * model.intermediate_size
        activation_elements_per_token = (
            4 * hidden + 2 * q_dim + 2 * kv_dim + 3 * model.intermediate_size
        )

    linear_flops_per_rank = (
        new_tokens
        * layers
        * (
            attention_linear_flops_per_token_per_rank
            + ffn_flops_per_token_cluster / tp_size
        )
    )
    causal_pair_delta_twice = (
        total_tokens * (total_tokens + 1)
        - cached_tokens * (cached_tokens + 1)
    )
    attention_flops_per_rank = (
        2.0 * q_dim_per_rank * causal_pair_delta_twice * layers
    )
    flops_per_rank = linear_flops_per_rank + attention_flops_per_rank
    total_flops = flops_per_rank * tp_size

    active_weight_bytes_per_rank = (
        layers
        * (
            attention_weight_elements_per_rank
            + ffn_weight_elements_cluster / tp_size
        )
        * dtype_bytes
        * prefill_chunks
    )
    activation_bytes_cluster = (
        new_tokens * layers * activation_elements_per_token * dtype_bytes
    )
    chunk_context_tokens = sum(
        min(total_tokens, cached_tokens + chunk_start + prefill_chunk_size)
        for chunk_start in range(0, new_tokens, prefill_chunk_size)
    )
    hbm_bytes_per_rank = (
        active_weight_bytes_per_rank
        + activation_bytes_cluster / tp_size
        + chunk_context_tokens * layout.physical_bytes_per_token_per_rank
    )
    compute_seconds = flops_per_rank / (
        hardware.bf16_dense_tflops * 1e12 * hardware.compute_efficiency
    )
    memory_seconds = hbm_bytes_per_rank / (
        hardware.hbm_bandwidth_tbps * SI_TB * hardware.memory_efficiency
    )
    launch_seconds = (
        layers * prefill_chunks * hardware.launch_overhead_us_per_layer * 1e-6
    )
    limiting = "compute" if compute_seconds >= memory_seconds else "memory"
    total_seconds = launch_seconds + max(compute_seconds, memory_seconds)
    return RooflineEstimate(
        tokens=new_tokens,
        total_flops=total_flops,
        flops_per_rank=flops_per_rank,
        hbm_bytes_per_rank=hbm_bytes_per_rank,
        compute_seconds=compute_seconds,
        memory_seconds=memory_seconds,
        launch_seconds=launch_seconds,
        total_seconds=total_seconds,
        limiting_term=limiting,
        expected_unique_experts_per_layer=expected_experts,
        prefill_chunks=prefill_chunks,
    )


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    total = sum(ordered)
    return {
        "count": len(ordered),
        "sum": total,
        "mean": total / len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def profile_provenance(
    repo_root: Path, model_name: str, hardware_name: str, tp_size: int
) -> dict[str, Any]:
    modern_root = repo_root / "profiler" / "perf" / hardware_name / model_name
    modern: dict[str, list[int]] = {}
    if modern_root.is_dir():
        for tp_dir in modern_root.glob("*/tp*"):
            try:
                tp = int(tp_dir.name.removeprefix("tp"))
            except ValueError:
                continue
            modern.setdefault(tp_dir.parent.name, []).append(tp)
    modern = {key: sorted(set(value)) for key, value in sorted(modern.items())}
    legacy_root = (
        repo_root / "profiler" / "v0" / "perf_models" / hardware_name / model_name
    )
    legacy_tps: list[int] = []
    if legacy_root.is_dir():
        for tp_dir in legacy_root.glob("tp*"):
            try:
                legacy_tps.append(int(tp_dir.name.removeprefix("tp")))
            except ValueError:
                continue
    modern_requested = any(tp_size in tps for tps in modern.values())
    legacy_requested = tp_size in legacy_tps
    limitations: list[str] = []
    if not modern_requested:
        limitations.append(
            f"No modern {hardware_name} TP{tp_size} category profile is bundled."
        )
    if legacy_tps and not legacy_requested:
        limitations.append(
            f"Legacy {hardware_name} data exists only at TP {sorted(set(legacy_tps))}; "
            "this analyzer does not silently treat it as TP8 measurement."
        )
    if hardware_name == "H200" and not modern and not legacy_tps:
        limitations.append("No bundled H200 profile was found for this model.")
    limitations.extend(
        [
            "Recompute is an analytical roofline, not a vLLM kernel replay.",
            "TP collectives, attention skew, kernel fusion, and MoE load imbalance are excluded.",
            "CPU/SSD defaults are uncalibrated until overridden with platform measurements.",
        ]
    )
    return {
        "mode_used": "analytical_roofline_only",
        "requested_tp_profile_available": modern_requested,
        "modern_variants_and_tps": modern,
        "legacy_tps": sorted(set(legacy_tps)),
        "legacy_requested_tp_available": legacy_requested,
        "limitations": limitations,
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def analyze_model_hardware(
    workload: WorkloadSummary,
    model: ModelShape,
    hardware: HardwareSpec,
    tp_size: int = 8,
    kv_dtype_bytes: int = 2,
    kv_layout_mode: str = "replicated",
    prefill_chunk_size: int = 2048,
    swap_out_mode: str = "cancellable",
    hbm_ttl_ms: float = 50.0,
    cpu_ttl_ms: float = 30_000.0,
    ssd_ttl_ms: float = 3_600_000.0,
    tiered_ssd_write_mode: str = "incremental",
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build one model/GPU summary row and its structured details."""

    hardware.validate()
    if swap_out_mode not in {"cancellable", "blocking"}:
        raise AnalysisConfigError(
            "swap_out_mode must be 'cancellable' or 'blocking'")
    if tiered_ssd_write_mode not in {"incremental", "full"}:
        raise AnalysisConfigError(
            "tiered_ssd_write_mode must be 'incremental' or 'full'")
    if min(hbm_ttl_ms, cpu_ttl_ms, ssd_ttl_ms) < 0:
        raise AnalysisConfigError("tier TTLs must be non-negative")
    layout = kv_layout(model, tp_size, kv_dtype_bytes, kv_layout_mode)
    cpu_service: list[float] = []
    cpu_exposed: list[float] = []
    cpu_cancellable_exposed: list[float] = []
    cpu_blocking_exposed: list[float] = []
    cpu_out_values: list[float] = []
    cpu_in_values: list[float] = []
    ssd_service: list[float] = []
    ssd_exposed: list[float] = []
    ssd_cancellable_exposed: list[float] = []
    ssd_blocking_exposed: list[float] = []
    ssd_out_values: list[float] = []
    ssd_in_values: list[float] = []
    recompute_full: list[float] = []
    recompute_reusable: list[float] = []
    tool_waits: list[float] = []
    cache_cluster_bytes: list[float] = []
    cache_rank_bytes: list[float] = []
    restore_cluster_bytes: list[float] = []
    restore_rank_bytes: list[float] = []
    cpu_write_overruns = 0
    ssd_write_overruns = 0
    cpu_faster_than_recompute = 0
    ssd_faster_than_recompute = 0
    limiting_terms: dict[str, int] = {"compute": 0, "memory": 0, "none": 0}
    full_rewrite_bytes = 0
    completed_full_rewrite_bytes = 0
    issued_full_rewrite_bytes = 0
    cancelled_partial_full_rewrite_bytes = 0
    optimistic_incremental_bytes = 0
    immediate_durable_tokens: dict[str, int] = {}
    # Migration stall and fallback recompute are distinct critical-path causes.
    # Keep a combined resume-overhead distribution for convenience, but never
    # use it as a migration numerator.
    tiered_migration_stall: list[float] = []
    tiered_resume_overhead: list[float] = []
    tiered_dropped_recompute: list[float] = []
    tiered_prompt_compute: list[float] = []
    tiered_sources = {"hbm": 0, "cpu": 0, "ssd": 0, "dropped": 0}
    tiered_restore_bytes_by_source = {
        "hbm": 0,
        "cpu": 0,
        "ssd": 0,
        "dropped": 0,
    }
    tiered_migration_stall_by_source = {
        "hbm": 0.0,
        "cpu": 0.0,
        "ssd": 0.0,
        "dropped": 0.0,
    }
    tiered_resume_overhead_by_source = {
        "hbm": 0.0,
        "cpu": 0.0,
        "ssd": 0.0,
        "dropped": 0.0,
    }
    tiered_ssd_host_write_bytes = 0
    tiered_ssd_completed_write_bytes = 0
    tiered_ssd_cancelled_partial_write_bytes = 0
    reuse_token_fractions: list[float] = []
    effective_reuse_tokens: list[float] = []
    # session -> (whole-object tokens, expiry on a per-session tool-time
    # clock). Active LLM latency is absent from the workload, so this clock is
    # an optimistic lower bound on object age; it still prevents a retained
    # keep-on-read snapshot from surviving its TTL across later tool waits.
    tiered_durable_records: dict[str, tuple[int, float]] = {}
    tiered_session_clocks: dict[str, float] = {}
    last_transition_index: dict[str, int] = {}

    for transition in workload.transitions:
        previous_index = last_transition_index.get(transition.session_id)
        if (previous_index is None
                or transition.sub_request_index != previous_index + 1):
            # Filtering a long-context or zero-duration intermediate turn
            # breaks the lineage chain. Do not extend an unseen snapshot.
            tiered_durable_records.pop(transition.session_id, None)
            immediate_durable_tokens.pop(transition.session_id, None)
        session_clock = tiered_session_clocks.get(transition.session_id, 0.0)
        durable_record = tiered_durable_records.get(transition.session_id)
        if durable_record is not None and durable_record[1] <= session_clock:
            tiered_durable_records.pop(transition.session_id, None)
        wait_seconds = transition.tool_duration_ns / 1e9
        cache_cluster = (
            transition.cache_tokens_declared
            * layout.physical_bytes_per_token_cluster
        )
        cache_rank = (
            transition.cache_tokens_declared
            * layout.physical_bytes_per_token_per_rank
        )
        restore_cluster = (
            transition.reusable_allocation_tokens
            * layout.physical_bytes_per_token_cluster
        )
        restore_rank = (
            transition.reusable_allocation_tokens
            * layout.physical_bytes_per_token_per_rank
        )
        cpu_out = cpu_transfer_seconds(
            cache_cluster, cache_rank, hardware.cpu, "out"
        )
        cpu_in = cpu_transfer_seconds(
            restore_cluster, restore_rank, hardware.cpu, "in"
        )
        ssd_out = ssd_transfer_seconds(
            cache_cluster, cache_rank, hardware, "out"
        )
        ssd_out_media = ssd_media_seconds(
            cache_cluster, hardware.ssd, "out"
        )
        ssd_out_write_start = max(0.0, ssd_out - ssd_out_media)
        ssd_in = ssd_transfer_seconds(
            restore_cluster, restore_rank, hardware, "in"
        )
        cpu_blocking_value = max(0.0, cpu_out - wait_seconds) + cpu_in
        ssd_blocking_value = max(0.0, ssd_out - wait_seconds) + ssd_in
        cpu_cancellable_value = cpu_in if cpu_out <= wait_seconds else 0.0
        ssd_cancellable_value = ssd_in if ssd_out <= wait_seconds else 0.0
        cpu_exposed_value = (
            cpu_cancellable_value
            if swap_out_mode == "cancellable"
            else cpu_blocking_value
        )
        ssd_exposed_value = (
            ssd_cancellable_value
            if swap_out_mode == "cancellable"
            else ssd_blocking_value
        )
        full_recompute = roofline_recompute_seconds(
            model,
            hardware,
            transition.next_input_tokens,
            tp_size,
            kv_dtype_bytes,
            layout,
            prefill_chunk_size,
        )
        reusable_recompute = roofline_recompute_seconds(
            model,
            hardware,
            transition.effective_reuse_tokens,
            tp_size,
            kv_dtype_bytes,
            layout,
            prefill_chunk_size,
        )

        # Online HBM -> CPU -> SSD age policy. Copies retain the upper-tier
        # source until commit, so a tool completion cancels an unfinished
        # migration without adding critical-path latency.
        cpu_commit = hbm_ttl_ms * 1e-3 + cpu_out
        if wait_seconds < cpu_commit:
            tiered_source = "hbm"
            tiered_migration_stall_value = 0.0
            tiered_dropped_recompute_value = 0.0
        else:
            durable_record = tiered_durable_records.get(
                transition.session_id)
            durable_tokens = (
                durable_record[0] if durable_record is not None else None)
            if (tiered_ssd_write_mode == "incremental"
                    and durable_tokens is not None
                    and transition.cache_tokens_declared >= durable_tokens):
                tiered_write_bytes = (
                    transition.cache_tokens_declared - durable_tokens
                ) * layout.physical_bytes_per_token_cluster
            else:
                tiered_write_bytes = cache_cluster
            tiered_ssd_write_start = cpu_commit + cpu_ttl_ms * 1e-3
            tiered_ssd_write_service = ssd_media_seconds(
                tiered_write_bytes, hardware.ssd, "out")
            ssd_commit = tiered_ssd_write_start + tiered_ssd_write_service
            if (durable_record is not None
                    and tiered_write_bytes < cache_cluster
                    and session_clock + tiered_ssd_write_start
                    >= durable_record[1]):
                # An append needs a live base when its media write starts.
                # Once started, that base is pinned through atomic commit,
                # matching the online manager. If it expired beforehand,
                # recompute service for a whole-object rewrite.
                tiered_durable_records.pop(transition.session_id, None)
                tiered_write_bytes = cache_cluster
                tiered_ssd_write_service = ssd_media_seconds(
                    tiered_write_bytes, hardware.ssd, "out")
                ssd_commit = (
                    tiered_ssd_write_start + tiered_ssd_write_service)
            if wait_seconds < ssd_commit:
                tiered_source = "cpu"
                tiered_migration_stall_value = cpu_in
                tiered_dropped_recompute_value = 0.0
                partial_write_bytes = _partial_service_bytes(
                    tiered_write_bytes,
                    max(0.0, wait_seconds - tiered_ssd_write_start),
                    tiered_ssd_write_service,
                )
                tiered_ssd_cancelled_partial_write_bytes += (
                    partial_write_bytes)
                tiered_ssd_host_write_bytes += partial_write_bytes
            else:
                tiered_ssd_host_write_bytes += tiered_write_bytes
                tiered_ssd_completed_write_bytes += tiered_write_bytes
                tiered_durable_records[transition.session_id] = (
                    transition.cache_tokens_declared,
                    session_clock + ssd_commit + ssd_ttl_ms * 1e-3,
                )
                if wait_seconds >= ssd_commit + ssd_ttl_ms * 1e-3:
                    tiered_source = "dropped"
                    tiered_migration_stall_value = 0.0
                    tiered_dropped_recompute_value = (
                        reusable_recompute.total_seconds)
                    tiered_durable_records.pop(transition.session_id, None)
                else:
                    tiered_source = "ssd"
                    tiered_migration_stall_value = ssd_in
                    tiered_dropped_recompute_value = 0.0
        tiered_resume_overhead_value = (
            tiered_migration_stall_value + tiered_dropped_recompute_value)
        tiered_sources[tiered_source] += 1
        tiered_restore_bytes_by_source[tiered_source] += restore_cluster
        tiered_migration_stall_by_source[tiered_source] += (
            tiered_migration_stall_value)
        tiered_resume_overhead_by_source[tiered_source] += (
            tiered_resume_overhead_value)
        tiered_migration_stall.append(tiered_migration_stall_value)
        tiered_resume_overhead.append(tiered_resume_overhead_value)
        if tiered_dropped_recompute_value > 0:
            tiered_dropped_recompute.append(tiered_dropped_recompute_value)
        # A successful HBM/CPU/SSD reuse avoids the prefix portion of the
        # next prefill; only the marginal suffix remains.  A dropped entry
        # performs the complete prefill.  The roofline is cumulative, so the
        # clamped difference is a prompt-only marginal-compute estimate.
        tiered_prompt_compute.append(
            full_recompute.total_seconds
            if tiered_source == "dropped"
            else max(
                0.0,
                full_recompute.total_seconds
                - reusable_recompute.total_seconds,
            )
        )

        full_rewrite_bytes += cache_cluster
        if swap_out_mode == "blocking" or ssd_out <= wait_seconds:
            completed_full_rewrite_bytes += cache_cluster
            issued_full_rewrite_bytes += cache_cluster
        else:
            partial_write_bytes = _partial_service_bytes(
                cache_cluster,
                max(0.0, wait_seconds - ssd_out_write_start),
                ssd_out_media,
            )
            issued_full_rewrite_bytes += partial_write_bytes
            cancelled_partial_full_rewrite_bytes += partial_write_bytes
        durable_tokens = immediate_durable_tokens.get(
            transition.session_id)
        if (durable_tokens is None or cache_cluster == 0
                or transition.cache_tokens_declared < durable_tokens):
            optimistic_incremental_bytes += cache_cluster
        else:
            optimistic_incremental_bytes += (
                max(0, transition.cache_tokens_declared - durable_tokens)
                * layout.physical_bytes_per_token_cluster
            )
        immediate_durable_tokens[transition.session_id] = (
            transition.cache_tokens_declared)

        # The following request can invalidate an older whole-object snapshot
        # even if this tool wait never reached SSD. The conservative
        # incremental baseline does not assume block-level copy-on-write: a
        # partial/divergent prefix invalidates the append base entirely.
        end_clock = session_clock + wait_seconds
        durable_record = tiered_durable_records.get(transition.session_id)
        if durable_record is not None:
            durable_tokens, durable_expiry = durable_record
            if (durable_expiry <= end_clock
                    or transition.declared_reuse_tokens < durable_tokens):
                tiered_durable_records.pop(transition.session_id, None)
            elif tiered_source == "ssd":
                # An exact keep-on-read restore refreshes the durable record's
                # last-access time. Raw declared LCP (not the input-1 hit cap)
                # decides whether the whole object remains a valid lineage.
                tiered_durable_records[transition.session_id] = (
                    durable_tokens,
                    end_clock + ssd_ttl_ms * 1e-3,
                )
        durable = immediate_durable_tokens.get(transition.session_id)
        if (durable is not None
                and transition.declared_reuse_tokens < durable):
            immediate_durable_tokens.pop(transition.session_id, None)
        tiered_session_clocks[transition.session_id] = end_clock
        last_transition_index[transition.session_id] = (
            transition.sub_request_index)

        tool_waits.append(wait_seconds)
        cache_cluster_bytes.append(cache_cluster)
        cache_rank_bytes.append(cache_rank)
        restore_cluster_bytes.append(restore_cluster)
        restore_rank_bytes.append(restore_rank)
        cpu_out_values.append(cpu_out)
        cpu_in_values.append(cpu_in)
        cpu_service.append(cpu_out + cpu_in)
        cpu_cancellable_exposed.append(cpu_cancellable_value)
        cpu_blocking_exposed.append(cpu_blocking_value)
        cpu_exposed.append(cpu_exposed_value)
        ssd_out_values.append(ssd_out)
        ssd_in_values.append(ssd_in)
        ssd_service.append(ssd_out + ssd_in)
        ssd_cancellable_exposed.append(ssd_cancellable_value)
        ssd_blocking_exposed.append(ssd_blocking_value)
        ssd_exposed.append(ssd_exposed_value)
        recompute_full.append(full_recompute.total_seconds)
        recompute_reusable.append(reusable_recompute.total_seconds)
        effective_reuse_tokens.append(transition.effective_reuse_tokens)
        reuse_token_fractions.append(
            transition.effective_reuse_tokens / transition.next_input_tokens
        )
        limiting_terms[reusable_recompute.limiting_term] += 1
        cpu_write_overruns += int(cpu_out > wait_seconds)
        ssd_write_overruns += int(ssd_out > wait_seconds)
        cpu_faster_than_recompute += int(
            cpu_exposed_value < reusable_recompute.total_seconds
        )
        ssd_faster_than_recompute += int(
            ssd_exposed_value < reusable_recompute.total_seconds
        )

    root = repo_root or repo_root_from_module()
    cpu_exposed_total = sum(cpu_exposed)
    ssd_exposed_total = sum(ssd_exposed)
    tiered_migration_stall_total = sum(tiered_migration_stall)
    tiered_resume_overhead_total = sum(tiered_resume_overhead)
    tiered_dropped_recompute_total = sum(tiered_dropped_recompute)
    recompute_reusable_total = sum(recompute_reusable)
    full_next_prefill_total = sum(recompute_full)
    tiered_prompt_compute_total = sum(tiered_prompt_compute)
    modeled_prompt_active_time = (
        tiered_prompt_compute_total + tiered_migration_stall_total
    )
    modeled_serialized_transition_time = (
        sum(tool_waits) + modeled_prompt_active_time
    )
    full_prefill_reference_time = (
        full_next_prefill_total + tiered_migration_stall_total
    )
    cold_sources = ("cpu", "ssd", "dropped")
    hbf_eligible_count = sum(tiered_sources[key] for key in cold_sources)
    hbf_eligible_bytes = sum(
        tiered_restore_bytes_by_source[key] for key in cold_sources
    )
    hbf_gross_migration_stall = sum(
        tiered_migration_stall_by_source[key] for key in cold_sources
    )
    hbf_gross_dropped_recompute = tiered_dropped_recompute_total
    hbf_gross_stall_upper_bound = (
        hbf_gross_migration_stall + hbf_gross_dropped_recompute)
    selected_count = len(workload.transitions)
    warnings = [
        "The roofline is a lower-bound sensitivity model, not a TP8 prediction validated by this repository.",
        "Swap-out uses declared completed-context tokens; swap-in transfers ceil-block physical storage while logical prefix hits remain exact.",
        "Swap-out can overlap tool wait, while restore and recompute are modeled as post-tool exposed latency.",
    ]
    if layout.warning:
        warnings.append(layout.warning)
    if workload.transitions_without_token_identity:
        warnings.append(
            f"{workload.transitions_without_token_identity} selected transitions lack token identity; explicit prefix metadata is used when available."
        )
    if any(
        transition.previous_token_id_coverage < 0.999
        or transition.next_token_id_coverage < 0.999
        for transition in workload.transitions
        if transition.token_identity_verified
    ):
        warnings.append(
            "Token-ID arrays do not cover every declared token; reusable-prefix bytes are conservative."
        )
    return {
        "model": model.name,
        "hardware": hardware.name,
        "tp_size": tp_size,
        "kv_dtype_bytes": kv_dtype_bytes,
        "prefill_chunk_size": prefill_chunk_size,
        "swap_out_mode": swap_out_mode,
        "model_shape": asdict(model),
        "hardware_spec": hardware.to_dict(),
        "kv_layout": layout.to_dict(),
        "profile_provenance": profile_provenance(
            root, model.name, hardware.name, tp_size
        ),
        "selected_tool_transitions": selected_count,
        "prefix_reuse": {
            "semantics": (
                "effective reusable tokens divided by the following request's "
                "input tokens; token_ids_exact is an observed tokenizer-level "
                "LCP, while explicit_* sources are reported metadata and are "
                "not relabeled as exact LCP"
            ),
            "effective_reuse_tokens": summarize_values(effective_reuse_tokens),
            "effective_reuse_fraction_of_next_input": summarize_values(
                reuse_token_fractions
            ),
            "source_counts": dict(workload.reuse_source_counts),
            "transitions_without_token_identity": (
                workload.transitions_without_token_identity
            ),
        },
        "tool_wait_seconds": summarize_values(tool_waits),
        "cache_bytes_physical_cluster": summarize_values(cache_cluster_bytes),
        "cache_bytes_physical_per_rank": summarize_values(cache_rank_bytes),
        "restore_bytes_physical_cluster": summarize_values(restore_cluster_bytes),
        "restore_bytes_physical_per_rank": summarize_values(restore_rank_bytes),
        "cpu_swap": {
            "swap_out_seconds": summarize_values(cpu_out_values),
            "swap_in_seconds": summarize_values(cpu_in_values),
            "service_seconds": summarize_values(cpu_service),
            "exposed_seconds": summarize_values(cpu_exposed),
            "cancellable_exposed_seconds": summarize_values(
                cpu_cancellable_exposed),
            "blocking_exposed_seconds": summarize_values(cpu_blocking_exposed),
            "write_overruns_tool_wait": cpu_write_overruns,
            "faster_than_recompute_count": cpu_faster_than_recompute,
        },
        "ssd_swap": {
            "swap_out_seconds": summarize_values(ssd_out_values),
            "swap_in_seconds": summarize_values(ssd_in_values),
            "service_seconds": summarize_values(ssd_service),
            "exposed_seconds": summarize_values(ssd_exposed),
            "cancellable_exposed_seconds": summarize_values(
                ssd_cancellable_exposed),
            "blocking_exposed_seconds": summarize_values(ssd_blocking_exposed),
            "write_overruns_tool_wait": ssd_write_overruns,
            "faster_than_recompute_count": ssd_faster_than_recompute,
            "host_write_bytes": {
                "full_rewrite_all_attempts": full_rewrite_bytes,
                "full_rewrite_completed_under_selected_mode": (
                    completed_full_rewrite_bytes),
                "full_rewrite_issued_under_selected_mode": (
                    issued_full_rewrite_bytes),
                "cancelled_partial_write_bytes_under_selected_mode": (
                    cancelled_partial_full_rewrite_bytes),
                "optimistic_incremental_append_lower_bound": (
                    optimistic_incremental_bytes),
            },
        },
        "tiered_policy": {
            "policy": "age_based_cancellable_hbm_cpu_ssd",
            "hbm_ttl_ms": hbm_ttl_ms,
            "cpu_ttl_ms": cpu_ttl_ms,
            "ssd_ttl_ms": ssd_ttl_ms,
            "ssd_write_mode": tiered_ssd_write_mode,
            "resume_source_counts": tiered_sources,
            "resume_source_fractions": {
                key: value / selected_count
                for key, value in tiered_sources.items()
            },
            "restore_bytes_by_source": tiered_restore_bytes_by_source,
            "aggregate_migration_stall_seconds_by_source": (
                tiered_migration_stall_by_source),
            "aggregate_resume_overhead_seconds_by_source": (
                tiered_resume_overhead_by_source),
            # Backward-compatible field: it now has the precise migration-only
            # meaning expected by its historical use as a migration numerator.
            "exposed_seconds": summarize_values(tiered_migration_stall),
            "exposed_seconds_scope": (
                "migration-only CPU/SSD restore stall; dropped fallback "
                "recompute is excluded"),
            "migration_stall_seconds": summarize_values(
                tiered_migration_stall),
            "resume_overhead_seconds": summarize_values(
                tiered_resume_overhead),
            "resume_overhead_seconds_scope": (
                "migration stall plus dropped-prefix fallback recompute; "
                "reported for resume-cost distribution only"),
            "dropped_fallback_recompute_seconds": summarize_values(
                tiered_dropped_recompute),
            "ssd_host_write_bytes": tiered_ssd_host_write_bytes,
            "ssd_completed_write_bytes": (
                tiered_ssd_completed_write_bytes),
            "ssd_cancelled_partial_write_bytes": (
                tiered_ssd_cancelled_partial_write_bytes),
        },
        "recompute": {
            "full_next_prefill_seconds": summarize_values(recompute_full),
            "avoidable_reusable_prefix_seconds": summarize_values(
                recompute_reusable
            ),
            "reusable_prefix_limiting_terms": limiting_terms,
            "avoidable_reusable_prefix_fraction_of_analyzed_next_prefill": (
                _safe_ratio(recompute_reusable_total, full_next_prefill_total)
            ),
            "avoidable_reusable_prefix_fraction_of_total_simulation_compute": None,
            "total_simulation_compute_fraction_note": (
                "Unavailable in the standalone trace analyzer: the workload "
                "does not contain cycle-level prefill/decode execution time. "
                "Use simulator-side compute accounting for this denominator."
            ),
        },
        "time_accounting": {
            "aggregate_tiered_request_stall_seconds": (
                tiered_migration_stall_total),
            "aggregate_tiered_migration_stall_seconds": (
                tiered_migration_stall_total),
            "aggregate_tiered_resume_overhead_seconds": (
                tiered_resume_overhead_total),
            "aggregate_dropped_fallback_recompute_seconds": (
                tiered_dropped_recompute_total),
            "aggregate_recompute_seconds": recompute_reusable_total,
            "observed_tool_wait_seconds": sum(tool_waits),
            "modeled_serialized_selected_transition_seconds": (
                modeled_serialized_transition_time
            ),
            "migration_stall_fraction_of_modeled_serialized_selected_transition_time": (
                _safe_ratio(
                    tiered_migration_stall_total,
                    modeled_serialized_transition_time,
                )
            ),
            "modeled_prompt_active_seconds": modeled_prompt_active_time,
            "migration_stall_fraction_of_modeled_prompt_active_time": (
                _safe_ratio(
                    tiered_migration_stall_total, modeled_prompt_active_time)
            ),
            "modeled_tiered_prompt_compute_seconds": (
                tiered_prompt_compute_total
            ),
            "full_prefill_reference_seconds": full_prefill_reference_time,
            "migration_stall_fraction_of_full_prefill_reference_time": (
                _safe_ratio(
                    tiered_migration_stall_total, full_prefill_reference_time)
            ),
            "modeled_fraction_scope": (
                "Prompt-only tiered lower bound: a cache hit executes the "
                "clamped analytical marginal full-prefill minus reusable-"
                "prefix compute, a dropped entry executes full prefill, and "
                "tiered restore stall is exposed. Serialized time also adds "
                "tool gaps. Decode, arrival overlap, and queueing are excluded."
            ),
            "migration_stall_fraction_of_total_simulated_request_time": None,
            "migration_stall_fraction_of_simulated_wall_time": None,
            "unavailable_fraction_note": (
                "The standalone workload has tool gaps but no complete LLM "
                "execution timeline or overlap schedule, so neither total "
                "request time nor wall-clock time is a valid denominator."
            ),
        },
        "hbf_npu_opportunity": {
            "scope": (
                "Gross upper bound for resumes beyond HBM (CPU, SSD, or "
                "dropped). It assumes their current restore/recompute stall "
                "could be eliminated and does not subtract HBF partial-"
                "attention latency."
            ),
            "eligible_resume_count": hbf_eligible_count,
            "eligible_resume_fraction": _safe_ratio(
                hbf_eligible_count, selected_count
            ),
            "eligible_restore_bytes": hbf_eligible_bytes,
            "gross_avoidable_migration_stall_seconds": (
                hbf_gross_migration_stall),
            "gross_avoidable_dropped_recompute_seconds": (
                hbf_gross_dropped_recompute),
            "gross_avoidable_stall_upper_bound_seconds": (
                hbf_gross_stall_upper_bound
            ),
            "ssd_only_resume_count": tiered_sources["ssd"],
            "ssd_only_resume_fraction": _safe_ratio(
                tiered_sources["ssd"], selected_count
            ),
            "ssd_only_restore_bytes": tiered_restore_bytes_by_source["ssd"],
            "why_ssd_resume_can_be_low": {
                "hbm_resume_count": tiered_sources["hbm"],
                "cpu_resume_count": tiered_sources["cpu"],
                "ssd_resume_count": tiered_sources["ssd"],
                "expired_or_dropped_count": tiered_sources["dropped"],
                "interpretation": (
                    "An SSD hit requires one tool gap to exceed the HBM-to-"
                    "CPU commit, CPU TTL, and SSD write completion, but remain "
                    "shorter than the SSD TTL. CPU-resident resumes therefore "
                    "represent HBF opportunity even when SSD hits are rare."
                ),
            },
        },
        "comparisons": {
            "cpu_exposed_to_recompute_ratio": _safe_ratio(
                cpu_exposed_total, recompute_reusable_total
            ),
            "ssd_exposed_to_recompute_ratio": _safe_ratio(
                ssd_exposed_total, recompute_reusable_total
            ),
            "tiered_exposed_to_recompute_ratio": _safe_ratio(
                tiered_migration_stall_total, recompute_reusable_total
            ),
            "tiered_migration_stall_to_recompute_ratio": _safe_ratio(
                tiered_migration_stall_total, recompute_reusable_total
            ),
            "tiered_resume_overhead_to_recompute_ratio": _safe_ratio(
                tiered_resume_overhead_total, recompute_reusable_total
            ),
            "cpu_faster_fraction": cpu_faster_than_recompute / selected_count,
            "ssd_faster_fraction": ssd_faster_than_recompute / selected_count,
            "cpu_swap_out_fully_hidden_fraction": 1.0
            - cpu_write_overruns / selected_count,
            "ssd_swap_out_fully_hidden_fraction": 1.0
            - ssd_write_overruns / selected_count,
        },
        "warnings": warnings,
    }


def build_report(
    workload: WorkloadSummary,
    models: Sequence[ModelShape],
    hardware_specs: Sequence[HardwareSpec],
    tp_size: int = 8,
    kv_dtype_bytes: int = 2,
    kv_layout_mode: str = "replicated",
    prefill_chunk_size: int = 2048,
    swap_out_mode: str = "cancellable",
    hbm_ttl_ms: float = 50.0,
    cpu_ttl_ms: float = 30_000.0,
    ssd_ttl_ms: float = 3_600_000.0,
    tiered_ssd_write_mode: str = "incremental",
    repo_root: Path | None = None,
) -> dict[str, Any]:
    if not models or not hardware_specs:
        raise AnalysisConfigError("at least one model and hardware spec is required")
    summaries = [
        analyze_model_hardware(
            workload,
            model,
            hardware,
            tp_size,
            kv_dtype_bytes,
            kv_layout_mode,
            prefill_chunk_size,
            swap_out_mode,
            hbm_ttl_ms,
            cpu_ttl_ms,
            ssd_ttl_ms,
            tiered_ssd_write_mode,
            repo_root,
        )
        for model in models
        for hardware in hardware_specs
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workload": workload.metadata_dict(),
        "assumptions": {
            "byte_units": "SI bytes for bandwidth; result byte counts are exact integers",
            "kv_formula": (
                "logical B/token = 2 * num_kv_heads * head_dim * layers * dtype_bytes; "
                "physical TP layout additionally accounts for KV-head replication"
            ),
            "prefix_restore_semantics": (
                "logical hit tokens are exact and capped at input_tokens-1; "
                "physical transfer bytes use ceil(block_size) allocation"
            ),
            "cpu_transfer_formula": (
                "fixed + max(per_rank_bytes/per_rank_link_Bps, "
                "cluster_bytes/aggregate_DRAM_Bps)"
            ),
            "ssd_transfer_formula": (
                "CPU stage + fixed + cluster_bytes/aggregate_SSD_Bps when staged_through_cpu"
            ),
            "exposed_swap_formula": (
                "cancellable: swap_in if swap_out <= tool_duration else 0; "
                "blocking sensitivity: max(0, swap_out-tool_duration)+swap_in"
            ),
            "selected_swap_out_mode": swap_out_mode,
            "prefill_chunk_size": prefill_chunk_size,
            "tiered_policy": {
                "hbm_ttl_ms": hbm_ttl_ms,
                "cpu_ttl_ms": cpu_ttl_ms,
                "ssd_ttl_ms": ssd_ttl_ms,
                "ssd_write_mode": tiered_ssd_write_mode,
                "migration_completion": (
                    "atomic and cancellable; unfinished copies retain upper-tier KV"
                ),
                "incremental_write": (
                    "token-granular append only when the durable record is a "
                    "transitively verified whole-prefix object; any skipped "
                    "turn or partial/divergent prefix invalidates the append "
                    "base and forces a full rewrite"
                ),
                "durable_ttl_clock": (
                    "per-session cumulative tool duration; active LLM and "
                    "restore wall time are unavailable and omitted, making "
                    "durable-record age an optimistic lower bound"
                ),
            },
            "recompute_roofline_formula": (
                "layers*chunks*launch + max(rank_FLOPs/(dense_peak*compute_eff), "
                "rank_HBM_bytes/(HBM_BW*memory_eff))"
            ),
            "causal_attention_flops": (
                "2 * q_dim * tokens * (tokens + 1) * layers for QK and PV"
            ),
            "scope_exclusions": [
                "TP collective/network time",
                "kernel-by-kernel fusion and launch gaps",
                "mixed-batch and FlashAttention skew effects",
                "MoE routing imbalance",
                "storage queueing across concurrent sessions",
                "shared HBM/CPU/SSD capacity and cross-session admission",
            ],
        },
        "summaries": summaries,
    }


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def hardware_spec_from_dict(
    name: str, data: Mapping[str, Any], base: HardwareSpec | None = None
) -> HardwareSpec:
    if base is None:
        required = {"bf16_dense_tflops", "hbm_bandwidth_tbps"}
        missing = sorted(required - set(data))
        if missing:
            raise AnalysisConfigError(
                f"hardware {name}: missing fields {', '.join(missing)}"
            )
        merged: dict[str, Any] = dict(data)
    else:
        merged = _deep_merge(base.to_dict(), data)
    cpu_data = merged.pop("cpu", {})
    ssd_data = merged.pop("ssd", {})
    merged["name"] = name
    try:
        spec = HardwareSpec(
            **merged,
            cpu=CpuTransferSpec(**cpu_data),
            ssd=SsdTransferSpec(**ssd_data),
        )
    except TypeError as exc:
        raise AnalysisConfigError(f"hardware {name}: invalid field: {exc}") from exc
    spec.validate()
    return spec


def load_hardware_config(path: Path | None = None) -> dict[str, HardwareSpec]:
    specs = dict(DEFAULT_HARDWARE_SPECS)
    if path is None:
        return specs
    with path.open("r", encoding="utf-8") as config_file:
        overrides = json.load(config_file)
    if not isinstance(overrides, dict):
        raise AnalysisConfigError("hardware config must be a JSON object")
    for name, data in overrides.items():
        if not isinstance(data, dict):
            raise AnalysisConfigError(f"hardware {name}: config must be an object")
        specs[name] = hardware_spec_from_dict(name, data, specs.get(name))
    return specs


def override_transfer_defaults(
    spec: HardwareSpec,
    cpu_rank_gbps: float | None = None,
    cpu_aggregate_gbps: float | None = None,
    ssd_read_gbps: float | None = None,
    ssd_write_gbps: float | None = None,
) -> HardwareSpec:
    cpu = spec.cpu
    ssd = spec.ssd
    if cpu_rank_gbps is not None:
        cpu = replace(
            cpu,
            gpu_to_host_gbps_per_rank=cpu_rank_gbps,
            host_to_gpu_gbps_per_rank=cpu_rank_gbps,
            provenance=cpu.provenance + " CLI per-rank bandwidth override applied.",
        )
    if cpu_aggregate_gbps is not None:
        cpu = replace(
            cpu,
            dram_write_gbps_aggregate=cpu_aggregate_gbps,
            dram_read_gbps_aggregate=cpu_aggregate_gbps,
            provenance=cpu.provenance + " CLI aggregate bandwidth override applied.",
        )
    if ssd_read_gbps is not None:
        ssd = replace(ssd, read_gbps_aggregate=ssd_read_gbps)
    if ssd_write_gbps is not None:
        ssd = replace(ssd, write_gbps_aggregate=ssd_write_gbps)
    result = replace(spec, cpu=cpu, ssd=ssd)
    result.validate()
    return result


def write_report_json(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


CSV_FIELDS = (
    "model",
    "hardware",
    "tp_size",
    "selected_tool_transitions",
    "kv_logical_bytes_per_token",
    "kv_physical_cluster_bytes_per_token",
    "kv_physical_rank_bytes_per_token",
    "kv_replication_factor",
    "total_swap_out_bytes",
    "ssd_completed_full_rewrite_bytes",
    "ssd_issued_full_rewrite_bytes",
    "ssd_cancelled_partial_write_bytes",
    "ssd_incremental_append_lower_bound_bytes",
    "total_restore_bytes",
    "cache_cluster_gb_p50",
    "cache_cluster_gb_p90",
    "tool_wait_ms_p50",
    "cpu_swap_service_ms_p50",
    "cpu_swap_exposed_ms_p50",
    "cpu_swap_cancellable_exposed_ms_p50",
    "cpu_swap_blocking_exposed_ms_p50",
    "ssd_swap_service_ms_p50",
    "ssd_swap_exposed_ms_p50",
    "ssd_swap_cancellable_exposed_ms_p50",
    "ssd_swap_blocking_exposed_ms_p50",
    "tiered_exposed_ms_p50",
    "tiered_resume_overhead_ms_p50",
    "tiered_dropped_recompute_seconds",
    "tiered_hbm_fraction",
    "tiered_cpu_fraction",
    "tiered_ssd_fraction",
    "tiered_dropped_fraction",
    "tiered_ssd_host_write_bytes",
    "tiered_ssd_completed_write_bytes",
    "tiered_ssd_cancelled_partial_write_bytes",
    "recompute_reusable_ms_p50",
    "recompute_full_prefill_ms_p50",
    "recompute_fraction_of_analyzed_next_prefill",
    "recompute_fraction_of_total_simulation_compute",
    "aggregate_tiered_request_stall_seconds",
    "aggregate_tiered_resume_overhead_seconds",
    "aggregate_dropped_fallback_recompute_seconds",
    "migration_stall_fraction_of_simulated_wall_time",
    "migration_stall_fraction_of_modeled_transition_time",
    "migration_stall_fraction_of_modeled_prompt_active_time",
    "migration_stall_fraction_of_full_prefill_reference_time",
    "prefix_reuse_fraction_p50",
    "prefix_reuse_fraction_p90",
    "prefix_reuse_source_counts",
    "hbf_eligible_resume_fraction",
    "hbf_eligible_restore_bytes",
    "hbf_gross_avoidable_migration_stall_seconds",
    "hbf_gross_avoidable_dropped_recompute_seconds",
    "hbf_gross_avoidable_stall_upper_bound_seconds",
    "ssd_only_resume_fraction",
    "cpu_exposed_to_recompute_ratio",
    "ssd_exposed_to_recompute_ratio",
    "cpu_swap_out_fully_hidden_fraction",
    "ssd_swap_out_fully_hidden_fraction",
    "requested_tp_profile_available",
    "profile_mode",
    "hardware_nominal_provenance",
    "hardware_calibration_provenance",
    "profile_limitations",
)


def _summary_csv_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    kv = summary["kv_layout"]
    cache = summary["cache_bytes_physical_cluster"]
    waits = summary["tool_wait_seconds"]
    cpu = summary["cpu_swap"]
    ssd = summary["ssd_swap"]
    tiered = summary["tiered_policy"]
    recompute = summary["recompute"]
    comparisons = summary["comparisons"]
    time_accounting = summary["time_accounting"]
    prefix_reuse = summary["prefix_reuse"]
    hbf_opportunity = summary["hbf_npu_opportunity"]
    profile = summary["profile_provenance"]
    return {
        "model": summary["model"],
        "hardware": summary["hardware"],
        "tp_size": summary["tp_size"],
        "selected_tool_transitions": summary["selected_tool_transitions"],
        "kv_logical_bytes_per_token": kv["logical_bytes_per_token"],
        "kv_physical_cluster_bytes_per_token": kv[
            "physical_bytes_per_token_cluster"
        ],
        "kv_physical_rank_bytes_per_token": kv[
            "physical_bytes_per_token_per_rank"
        ],
        "kv_replication_factor": kv["replication_factor"],
        "total_swap_out_bytes": cache["sum"],
        "ssd_completed_full_rewrite_bytes": ssd["host_write_bytes"][
            "full_rewrite_completed_under_selected_mode"
        ],
        "ssd_issued_full_rewrite_bytes": ssd["host_write_bytes"][
            "full_rewrite_issued_under_selected_mode"
        ],
        "ssd_cancelled_partial_write_bytes": ssd["host_write_bytes"][
            "cancelled_partial_write_bytes_under_selected_mode"
        ],
        "ssd_incremental_append_lower_bound_bytes": ssd["host_write_bytes"][
            "optimistic_incremental_append_lower_bound"
        ],
        "total_restore_bytes": summary["restore_bytes_physical_cluster"]["sum"],
        "cache_cluster_gb_p50": cache["p50"] / SI_GB,
        "cache_cluster_gb_p90": cache["p90"] / SI_GB,
        "tool_wait_ms_p50": waits["p50"] * 1e3,
        "cpu_swap_service_ms_p50": cpu["service_seconds"]["p50"] * 1e3,
        "cpu_swap_exposed_ms_p50": cpu["exposed_seconds"]["p50"] * 1e3,
        "cpu_swap_cancellable_exposed_ms_p50": cpu[
            "cancellable_exposed_seconds"
        ]["p50"] * 1e3,
        "cpu_swap_blocking_exposed_ms_p50": cpu[
            "blocking_exposed_seconds"
        ]["p50"] * 1e3,
        "ssd_swap_service_ms_p50": ssd["service_seconds"]["p50"] * 1e3,
        "ssd_swap_exposed_ms_p50": ssd["exposed_seconds"]["p50"] * 1e3,
        "ssd_swap_cancellable_exposed_ms_p50": ssd[
            "cancellable_exposed_seconds"
        ]["p50"] * 1e3,
        "ssd_swap_blocking_exposed_ms_p50": ssd[
            "blocking_exposed_seconds"
        ]["p50"] * 1e3,
        "tiered_exposed_ms_p50": tiered["exposed_seconds"]["p50"] * 1e3,
        "tiered_resume_overhead_ms_p50": tiered[
            "resume_overhead_seconds"
        ]["p50"] * 1e3,
        "tiered_dropped_recompute_seconds": tiered[
            "dropped_fallback_recompute_seconds"
        ]["sum"],
        "tiered_hbm_fraction": tiered["resume_source_fractions"]["hbm"],
        "tiered_cpu_fraction": tiered["resume_source_fractions"]["cpu"],
        "tiered_ssd_fraction": tiered["resume_source_fractions"]["ssd"],
        "tiered_dropped_fraction": tiered["resume_source_fractions"]["dropped"],
        "tiered_ssd_host_write_bytes": tiered["ssd_host_write_bytes"],
        "tiered_ssd_completed_write_bytes": tiered[
            "ssd_completed_write_bytes"
        ],
        "tiered_ssd_cancelled_partial_write_bytes": tiered[
            "ssd_cancelled_partial_write_bytes"
        ],
        "recompute_reusable_ms_p50": recompute[
            "avoidable_reusable_prefix_seconds"
        ]["p50"]
        * 1e3,
        "recompute_full_prefill_ms_p50": recompute[
            "full_next_prefill_seconds"
        ]["p50"]
        * 1e3,
        "recompute_fraction_of_analyzed_next_prefill": recompute[
            "avoidable_reusable_prefix_fraction_of_analyzed_next_prefill"
        ],
        "recompute_fraction_of_total_simulation_compute": recompute[
            "avoidable_reusable_prefix_fraction_of_total_simulation_compute"
        ],
        "aggregate_tiered_request_stall_seconds": time_accounting[
            "aggregate_tiered_request_stall_seconds"
        ],
        "aggregate_tiered_resume_overhead_seconds": time_accounting[
            "aggregate_tiered_resume_overhead_seconds"
        ],
        "aggregate_dropped_fallback_recompute_seconds": time_accounting[
            "aggregate_dropped_fallback_recompute_seconds"
        ],
        "migration_stall_fraction_of_simulated_wall_time": time_accounting[
            "migration_stall_fraction_of_simulated_wall_time"
        ],
        "migration_stall_fraction_of_modeled_transition_time": time_accounting[
            "migration_stall_fraction_of_modeled_serialized_selected_transition_time"
        ],
        "migration_stall_fraction_of_modeled_prompt_active_time": time_accounting[
            "migration_stall_fraction_of_modeled_prompt_active_time"
        ],
        "migration_stall_fraction_of_full_prefill_reference_time": time_accounting[
            "migration_stall_fraction_of_full_prefill_reference_time"
        ],
        "prefix_reuse_fraction_p50": prefix_reuse[
            "effective_reuse_fraction_of_next_input"
        ]["p50"],
        "prefix_reuse_fraction_p90": prefix_reuse[
            "effective_reuse_fraction_of_next_input"
        ]["p90"],
        "prefix_reuse_source_counts": json.dumps(
            prefix_reuse["source_counts"], sort_keys=True
        ),
        "hbf_eligible_resume_fraction": hbf_opportunity[
            "eligible_resume_fraction"
        ],
        "hbf_eligible_restore_bytes": hbf_opportunity[
            "eligible_restore_bytes"
        ],
        "hbf_gross_avoidable_migration_stall_seconds": hbf_opportunity[
            "gross_avoidable_migration_stall_seconds"
        ],
        "hbf_gross_avoidable_dropped_recompute_seconds": hbf_opportunity[
            "gross_avoidable_dropped_recompute_seconds"
        ],
        "hbf_gross_avoidable_stall_upper_bound_seconds": hbf_opportunity[
            "gross_avoidable_stall_upper_bound_seconds"
        ],
        "ssd_only_resume_fraction": hbf_opportunity[
            "ssd_only_resume_fraction"
        ],
        "cpu_exposed_to_recompute_ratio": comparisons[
            "cpu_exposed_to_recompute_ratio"
        ],
        "ssd_exposed_to_recompute_ratio": comparisons[
            "ssd_exposed_to_recompute_ratio"
        ],
        "cpu_swap_out_fully_hidden_fraction": comparisons[
            "cpu_swap_out_fully_hidden_fraction"
        ],
        "ssd_swap_out_fully_hidden_fraction": comparisons[
            "ssd_swap_out_fully_hidden_fraction"
        ],
        "requested_tp_profile_available": profile[
            "requested_tp_profile_available"
        ],
        "profile_mode": profile["mode_used"],
        "hardware_nominal_provenance": summary["hardware_spec"][
            "nominal_provenance"
        ],
        "hardware_calibration_provenance": summary["hardware_spec"][
            "calibration_provenance"
        ],
        "profile_limitations": " | ".join(profile["limitations"]),
    }


def write_report_csv(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for summary in report["summaries"]:
            writer.writerow(_summary_csv_row(summary))

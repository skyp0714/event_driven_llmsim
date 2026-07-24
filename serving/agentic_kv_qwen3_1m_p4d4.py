"""Reproducible Qwen3 1M P4+D4 cold-KV capacity experiment.

Prompt compute is projected with a kernel-decomposed analytical model fitted
once to the repository's legacy H100 measurements.  This is still an
extrapolation to Qwen3 and 1M-token attention, not a measured Qwen3 H100 run.
Each finite-capacity baseline uses the same prompt predictor object as its
paired infinite-HBM residency reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core.agentic_kv_capacity_replay import (
    CapacityReplayConfig,
    estimate_model_weight_bytes_per_rank,
    load_capacity_replay_workload,
    replay_capacity_aware_with_oracle,
    write_capacity_report,
)
from .core.agentic_kv_roofline import (
    AnalysisConfigError,
    kv_layout,
    load_hardware_config,
    load_model_shape,
    override_transfer_defaults,
)
from .core.h100_kernel_calibrated_prompt import (
    CALIBRATION_SOURCE_PATHS as KERNEL_CALIBRATION_SOURCE_PATHS,
    H100KernelCalibratedPromptModel,
    LEGACY_PRODUCER_SOURCE_PATHS as KERNEL_CALIBRATION_PRODUCER_SOURCE_PATHS,
    QWEN_EP,
    QWEN_EXPERTS,
    QWEN_EXPERT_INTERMEDIATE,
    QWEN_HEAD_DIM,
    QWEN_HIDDEN_SIZE,
    QWEN_KV_HEADS,
    QWEN_LAYERS,
    QWEN_Q_HEADS,
    QWEN_TOP_K,
    QWEN_TP,
    fit_h100_tp4_calibration,
)


MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
HARDWARE_NAME = "H100"
TP_SIZE = 4
MAX_CONTEXT_TOKENS = 1_010_000
PREFILL_CHUNK_SIZE = 131_072
BLOCK_SIZE = 16
KV_DTYPE_BYTES = 2
WEIGHT_DTYPE_BYTES = 2

HBM_CAPACITY_BYTES_PER_RANK = 80_000_000_000
CPU_CAPACITY_BYTES = 2_000_000_000_000
SSD_CAPACITY_BYTES = 30_720_000_000_000
CPU_PCIE_GBPS_PER_RANK = 50.0
CPU_DRAM_GBPS_AGGREGATE = 400.0
SSD_READ_GBPS_AGGREGATE = 55.2
SSD_WRITE_GBPS_AGGREGATE = 33.6
PD_NVLINK_GBPS_PER_RANK_ONE_WAY = 450.0
PD_FIXED_LATENCY_US = 3.0

QWEN_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
QWEN_MODEL_CONFIG_PATH = (
    "configs/model/Qwen/Qwen3-30B-A3B-Instruct-2507.json"
)
QWEN_REPOSITORY_URL = (
    "https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507"
)
QWEN_CONFIG_1M_SHA256 = (
    "bacd81916858d5b9f5daa616ee3aca13e3f888ceeb374cc804546de008dc85d0"
)
QWEN_CONFIG_1M_BYTES = 77_313
QWEN_MODEL_CARD_SHA256 = (
    "5d25d4b43cdfaf83e3d71df8f3436703ab26d92c07305053be667f577a49bb9e"
)
QWEN_MODEL_CARD_BYTES = 15_418
QWEN_CHECKPOINT_INDEX_SHA256 = (
    "8dde190b862c7c80ec7403c6495de00c60bbaf246ed479cee4506284989c584c"
)
QWEN_CHECKPOINT_INDEX_BYTES = 1_699_758
QWEN_CHECKPOINT_BYTES_TOTAL = 61_064_245_248
QWEN_CHECKPOINT_BYTES_PER_TP4_RANK = (
    QWEN_CHECKPOINT_BYTES_TOTAL // TP_SIZE
)

NVIDIA_DGX_H100_GUIDE_URL = (
    "https://docs.nvidia.com/dgx/dgxh100-user-guide/"
    "introduction-to-dgxh100.html"
)
NVIDIA_DGX_H100_GUIDE_OBSERVED_SHA256 = (
    "aa3dd7e72e910b03e697dfe7cfb76efc04bb9382f0c73cf620eba348eea6c945"
)
NVIDIA_DGX_H100_GUIDE_OBSERVED_BYTES = 45_788
NVIDIA_DGX_H100_NVME_GUIDE_URL = (
    "https://docs.nvidia.com/dgx/dgxh100-fw-update-guide/"
    "nvme-fw-update.html"
)
NVIDIA_DGX_H100_NVME_GUIDE_OBSERVED_SHA256 = (
    "44cbee198269ab8702247a1df1b3054be25201dec4dbf64c40ce6e7173a24c9a"
)
NVIDIA_DGX_H100_NVME_GUIDE_OBSERVED_BYTES = 29_431
NVIDIA_H100_SPEC_URL = "https://www.nvidia.com/en-us/data-center/h100/"
NVIDIA_H100_SPEC_OBSERVED_SHA256 = (
    "09cee4d569656179646f4aed350f39ec1a5f6b13be50de3b0f46183c9ab7a2f2"
)
NVIDIA_H100_SPEC_OBSERVED_BYTES = 288_443
KIOXIA_CM6_PRODUCT_BRIEF_URL = (
    "https://americas.kioxia.com/content/dam/kioxia/shared/business/ssd/"
    "enterprise-ssd/asset/productbrief/eSSD-CM6-R-product-brief.pdf"
)
KIOXIA_CM6_PRODUCT_BRIEF_SHA256 = (
    "1e22a2f5a19b89bbaa156bcdffaaecc3c99b67d1f10594126b42f5f968b91a3e"
)
KIOXIA_CM6_PRODUCT_BRIEF_BYTES = 133_166
KERNELSIGHT_LM_URL = "https://arxiv.org/abs/2606.28565"
KERNELSIGHT_LM_EQUATION = (
    "t=max(t_roof*eta,t0); "
    "t_roof=max((F/P_peak)*u,B/BW_peak); "
    "u=ceil(thread_blocks/num_sms)*num_sms/thread_blocks"
)

# The official model card says that 1M processing requires approximately
# 240 GB total GPU memory for weights, KV, and peak activations.  Splitting
# that approximate statement evenly across TP4 and subtracting exact BF16
# checkpoint and KV geometry yields a residual sensitivity, not a measurement.
MODEL_CARD_APPROX_ENGINE_BYTES_TOTAL = 240_000_000_000
MODEL_CARD_APPROX_ENGINE_BYTES_PER_TP4_RANK = (
    MODEL_CARD_APPROX_ENGINE_BYTES_TOTAL // TP_SIZE
)
KV_BYTES_PER_TOKEN_PER_TP4_RANK = 24_576
FULL_CONTEXT_KV_BYTES_PER_TP4_RANK = (
    MAX_CONTEXT_TOKENS * KV_BYTES_PER_TOKEN_PER_TP4_RANK
)
MODEL_CARD_INFERRED_RUNTIME_RESIDUAL_BYTES_PER_RANK = (
    MODEL_CARD_APPROX_ENGINE_BYTES_PER_TP4_RANK
    - QWEN_CHECKPOINT_BYTES_PER_TP4_RANK
    - FULL_CONTEXT_KV_BYTES_PER_TP4_RANK
)

POLICIES = (
    "hbm_lru_recompute",
    "hbm_ssd_direct",
    "tiered",
)

QWEN_RTX_PROFILE_META_PATH = (
    "profiler/perf/RTXPRO6000/Qwen/"
    "Qwen3-30B-A3B-Instruct-2507/bf16/meta.yaml"
)
LEGACY_H100_LLAMA70_TP4_ATTENTION_PATH = (
    "profiler/v0/perf_models/H100/meta-llama/"
    "Llama-3.1-70B/tp4/attention.csv"
)
LEGACY_H100_MIXTRAL_TP4_ATTENTION_PATH = (
    "profiler/v0/perf_models/H100/mistralai/"
    "Mixtral-8x7B-v0.1/tp4/attention.csv"
)
LEGACY_H100_LLAMA70_TP4_LAYERS_PATH = (
    "profiler/v0/perf_models/H100/meta-llama/"
    "Llama-3.1-70B/tp4/layers.csv"
)
LEGACY_H100_MIXTRAL_TP4_LAYERS_PATH = (
    "profiler/v0/perf_models/H100/mistralai/"
    "Mixtral-8x7B-v0.1/tp4/layers.csv"
)
LEGACY_H100_CALIBRATION_SOURCE_PATHS = (
    LEGACY_H100_LLAMA70_TP4_LAYERS_PATH,
    LEGACY_H100_LLAMA70_TP4_ATTENTION_PATH,
    LEGACY_H100_MIXTRAL_TP4_LAYERS_PATH,
    LEGACY_H100_MIXTRAL_TP4_ATTENTION_PATH,
)
LEGACY_H100_PRODUCER_SOURCE_PATHS = tuple(
    KERNEL_CALIBRATION_PRODUCER_SOURCE_PATHS
)


@dataclass(frozen=True)
class ComputeEndpoint:
    name: str
    band: str
    attention_multiplier: float
    provenance: str


COMPUTE_ENDPOINTS = (
    ComputeEndpoint(
        name="central_full_attention",
        band="central",
        attention_multiplier=1.0,
        provenance=(
            "Central fitted coefficients with full analytical causal-"
            "attention work. This is the primary compute endpoint."
        ),
    ),
    ComputeEndpoint(
        name="central_attention_one_third",
        band="central",
        attention_multiplier=1.0 / 3.0,
        provenance=(
            "Central fitted coefficients with only the attention component "
            "multiplied by one third. Non-attention kernels are unchanged. "
            "This is a labeled DCA/MInference sensitivity, not a measured "
            "Qwen3 DCA/MInference H100 result."
        ),
    ),
    ComputeEndpoint(
        name="fast_full_attention",
        band="fast",
        attention_multiplier=1.0,
        provenance=(
            "Fast coefficient band from legacy-H100 fit uncertainty with "
            "full analytical causal-attention work. Optional sensitivity."
        ),
    ),
    ComputeEndpoint(
        name="slow_full_attention",
        band="slow",
        attention_multiplier=1.0,
        provenance=(
            "Slow coefficient band from legacy-H100 fit uncertainty with "
            "full analytical causal-attention work. Optional sensitivity."
        ),
    ),
)
DEFAULT_COMPUTE_ENDPOINT_NAMES = (
    "central_full_attention",
    "central_attention_one_third",
)


def resolve_compute_endpoints(
    requested: Sequence[str] | None,
) -> tuple[ComputeEndpoint, ...]:
    """Resolve selected endpoint names while rejecting duplicates."""

    available = {endpoint.name: endpoint for endpoint in COMPUTE_ENDPOINTS}
    names = (
        DEFAULT_COMPUTE_ENDPOINT_NAMES
        if requested is None or len(requested) == 0
        else tuple(requested)
    )
    unknown = [name for name in names if name not in available]
    if unknown:
        raise AnalysisConfigError(
            "unknown compute endpoint(s): " + ", ".join(unknown)
        )
    if len(set(names)) != len(names):
        raise AnalysisConfigError("compute endpoints cannot be repeated")
    return tuple(available[name] for name in names)


@dataclass(frozen=True)
class ReserveCase:
    name: str
    common_bytes_per_rank: int
    prefill_bytes_per_rank: int | None = None
    decode_bytes_per_rank: int | None = None
    target_residual_fraction: float | None = None
    provenance: str = ""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _provenance_value(
    manifest: Mapping[str, Any], *paths: tuple[str, ...]
) -> Any:
    for path in paths:
        value: Any = manifest
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            return value
    return None


def _missing_workload_provenance(
    path: Path | None,
    discovery: str,
) -> dict[str, Any]:
    return {
        "status": "missing",
        "discovery": discovery,
        "sidecar": {
            "path": None if path is None else str(path.resolve()),
            "exists": False,
            "sha256": None,
            "bytes": None,
        },
        "tracelab_source": {
            "format": None,
            "location": None,
            "revision": None,
            "revision_status": "missing",
            "sha256": None,
            "sha256_status": "missing",
        },
        "converter": {
            "generator": None,
            "schema_version": None,
            "schema_version_status": "missing",
            "version": None,
            "version_status": "missing",
            "commit": None,
            "commit_status": "missing",
        },
        "validation": {
            "status": None,
            "status_provenance": "missing",
            "error_count": None,
            "warning_count": None,
            "warning_counts": {},
            "warning_count_matches_counters": None,
        },
        "output_binding": {
            "declared_sha256": None,
            "actual_sha256": None,
            "status": "unavailable",
        },
        "raw_manifest": None,
        "warning": (
            "No workload provenance sidecar was found. Trace revision, raw "
            "source hash, converter identity, and conversion validation are "
            "therefore unavailable."
        ),
    }


def load_workload_provenance(
    workload_path: Path,
    workload_sha256: str,
    explicit_path: Path | None = None,
) -> dict[str, Any]:
    """Load and bind a conversion sidecar to the exact replay workload."""

    discovery = "explicit" if explicit_path is not None else "automatic"
    sidecar_path = (
        explicit_path
        if explicit_path is not None
        else Path(f"{workload_path}.manifest.json")
    )
    if not sidecar_path.exists():
        if explicit_path is not None:
            raise AnalysisConfigError(
                f"explicit workload provenance sidecar does not exist: "
                f"{sidecar_path}"
            )
        missing = _missing_workload_provenance(sidecar_path, discovery)
        missing["output_binding"]["actual_sha256"] = workload_sha256
        return missing
    try:
        raw = sidecar_path.read_bytes()
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisConfigError(
            f"invalid workload provenance sidecar {sidecar_path}: {exc}"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise AnalysisConfigError(
            f"workload provenance sidecar must be a JSON object: {sidecar_path}"
        )

    declared_output_hash = _provenance_value(
        parsed,
        ("output", "sha256"),
        ("workload", "sha256"),
        ("output_sha256",),
    )
    if declared_output_hash is None:
        output_binding_status = "sidecar_output_hash_missing"
    elif str(declared_output_hash).lower() != workload_sha256.lower():
        raise AnalysisConfigError(
            "workload provenance sidecar output hash does not match replay "
            f"workload: {declared_output_hash} != {workload_sha256}"
        )
    else:
        output_binding_status = "verified"

    source_location = _provenance_value(
        parsed,
        ("source", "location"),
        ("raw_source", "path"),
        ("source_path",),
    )
    declared_source_hash = _provenance_value(
        parsed,
        ("source", "sha256"),
        ("source", "raw_sha256"),
        ("raw_source", "sha256"),
        ("source_sha256",),
    )
    source_path = None
    computed_source_hash = None
    if source_location:
        candidate = Path(str(source_location))
        source_path = (
            candidate
            if candidate.is_absolute()
            else sidecar_path.parent / candidate
        )
        if source_path.is_file():
            computed_source_hash = _sha256_file(source_path)
    if declared_source_hash is not None and computed_source_hash is not None:
        if str(declared_source_hash).lower() != computed_source_hash.lower():
            raise AnalysisConfigError(
                "workload provenance raw-source hash does not match the "
                f"declared source file: {declared_source_hash} != "
                f"{computed_source_hash}"
            )
        source_hash = computed_source_hash
        source_hash_status = "declared_and_verified"
    elif declared_source_hash is not None:
        source_hash = str(declared_source_hash)
        source_hash_status = "declared_unverified_source_unavailable"
    elif computed_source_hash is not None:
        source_hash = computed_source_hash
        source_hash_status = "computed_from_declared_source_location"
    else:
        source_hash = None
        source_hash_status = "missing"

    source_revision = _provenance_value(
        parsed,
        ("source", "revision"),
        ("raw_source", "revision"),
        ("source_revision",),
    )
    converter_schema = _provenance_value(
        parsed,
        ("converter", "schema_version"),
        ("schema_version",),
    )
    converter_version = _provenance_value(
        parsed,
        ("converter", "version"),
        ("generator_version",),
        ("converter_version",),
    )
    converter_commit = _provenance_value(
        parsed,
        ("converter", "commit"),
        ("converter", "git_commit"),
        ("generator_commit",),
        ("converter_commit",),
    )
    validation = parsed.get("validation", {})
    if not isinstance(validation, Mapping):
        validation = {}
    warning_counts = validation.get("warning_counts", {})
    if not isinstance(warning_counts, Mapping):
        warning_counts = {}
    normalized_warning_counts = {
        str(key): value for key, value in sorted(warning_counts.items())
    }
    warning_count = validation.get("warning_count")
    warning_sum = (
        sum(normalized_warning_counts.values())
        if all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in normalized_warning_counts.values()
        )
        else None
    )
    warning_count_matches = (
        None
        if warning_count is None or warning_sum is None
        else warning_count == warning_sum
    )
    validation_status = validation.get("status")
    return {
        "status": "loaded",
        "discovery": discovery,
        "sidecar": {
            "path": str(sidecar_path.resolve()),
            "exists": True,
            "sha256": _sha256_bytes(raw),
            "bytes": len(raw),
        },
        "tracelab_source": {
            "format": _provenance_value(parsed, ("source", "format")),
            "location": source_location,
            "resolved_local_path": (
                None if source_path is None else str(source_path.resolve())
            ),
            "revision": source_revision,
            "revision_status": (
                "declared" if source_revision is not None else "missing"
            ),
            "declared_sha256": declared_source_hash,
            "computed_sha256": computed_source_hash,
            "sha256": source_hash,
            "sha256_status": source_hash_status,
            "sha256_interpretation": (
                "A declared_and_verified hash is bound by both the sidecar "
                "and the currently readable source file. A computed-only "
                "hash identifies the source path at experiment time but does "
                "not prove that identical bytes were used at conversion time."
            ),
        },
        "converter": {
            "generator": _provenance_value(
                parsed,
                ("converter", "name"),
                ("converter", "module"),
                ("generator",),
            ),
            "schema_version": converter_schema,
            "schema_version_status": (
                "declared" if converter_schema is not None else "missing"
            ),
            "version": converter_version,
            "version_status": (
                "declared" if converter_version is not None else "missing"
            ),
            "commit": converter_commit,
            "commit_status": (
                "declared" if converter_commit is not None else "missing"
            ),
            "module_sha256": _provenance_value(
                parsed, ("converter", "module_sha256")
            ),
            "git_dirty_tracked_files": _provenance_value(
                parsed, ("converter", "git_dirty_tracked_files")
            ),
            "arguments": _provenance_value(
                parsed, ("converter", "arguments")
            ),
        },
        "validation": {
            "status": validation_status,
            "status_provenance": (
                "declared" if validation_status is not None else "missing"
            ),
            "error_count": validation.get("error_count"),
            "warning_count": warning_count,
            "warning_counts": normalized_warning_counts,
            "warning_count_matches_counters": warning_count_matches,
        },
        "output_binding": {
            "declared_sha256": declared_output_hash,
            "actual_sha256": workload_sha256,
            "status": output_binding_status,
        },
        "raw_manifest": dict(parsed),
        "warning": None,
    }


def _nested(
    value: Mapping[str, Any], *keys: str, default: Any = None
) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _token_distribution(
    values: Sequence[int],
) -> dict[str, int | float | str]:
    if not values:
        raise AnalysisConfigError("cannot summarize an empty token sample")
    ordered = sorted(values)

    def nearest_rank(quantile: float) -> int:
        index = max(0, math.ceil(quantile * len(ordered)) - 1)
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": nearest_rank(0.50),
        "p90": nearest_rank(0.90),
        "p95": nearest_rank(0.95),
        "p99": nearest_rank(0.99),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "percentile_method": "nearest_rank",
    }


def derive_reserve_cases(
    architecture_weight_bytes_per_rank: int,
    sweep: str,
    common_override: int | None = None,
    prefill_override: int | None = None,
    decode_override: int | None = None,
) -> tuple[ReserveCase, ...]:
    """Resolve exact core reserve inputs from model-card residual targets."""

    overrides = (common_override, prefill_override, decode_override)
    for value in overrides:
        if value is not None and value < 0:
            raise AnalysisConfigError("HBM reserves must be nonnegative bytes")
    if any(value is not None for value in overrides):
        if sweep != "full":
            raise AnalysisConfigError(
                "custom common/role reserves cannot be combined with a "
                "reserve sensitivity sweep"
            )
        full_target = (
            QWEN_CHECKPOINT_BYTES_PER_TP4_RANK
            + MODEL_CARD_INFERRED_RUNTIME_RESIDUAL_BYTES_PER_RANK
        )
        default_common = full_target - architecture_weight_bytes_per_rank
        if default_common < 0:
            raise AnalysisConfigError(
                "architecture-derived weights exceed the full non-KV target"
            )
        return (
            ReserveCase(
                name="custom",
                common_bytes_per_rank=(
                    default_common
                    if common_override is None
                    else common_override
                ),
                prefill_bytes_per_rank=prefill_override,
                decode_bytes_per_rank=decode_override,
                provenance=(
                    "User-specified exact-byte common and/or P/D role reserve; "
                    "unspecified common reserve defaults to the model-card "
                    "full-residual target adjusted for the simulator's "
                    "architecture-derived weight estimate."
                ),
            ),
        )

    fractions = {
        "zero": (0.0,),
        "half": (0.5,),
        "full": (1.0,),
        "all": (0.0, 0.5, 1.0),
    }[sweep]
    cases = []
    for fraction in fractions:
        target_non_kv = (
            QWEN_CHECKPOINT_BYTES_PER_TP4_RANK
            + int(
                MODEL_CARD_INFERRED_RUNTIME_RESIDUAL_BYTES_PER_RANK * fraction
            )
        )
        exact_adjusted = target_non_kv - architecture_weight_bytes_per_rank
        # At zero residual the architecture formula is 19 MB larger than the
        # checkpoint index, and the replay does not permit a negative reserve.
        core_reserve = max(0, exact_adjusted)
        name = {0.0: "zero_residual", 0.5: "half_residual", 1.0: "full_residual"}[
            fraction
        ]
        zero_note = (
            " At zero residual, a negative adjustment is not representable, "
            "so zero is used and the small weight-estimate delta is reported."
            if fraction == 0.0
            else ""
        )
        cases.append(
            ReserveCase(
                name=name,
                common_bytes_per_rank=core_reserve,
                target_residual_fraction=fraction,
                provenance=(
                    "Sensitivity derived from the official model card's "
                    "approximate 240 GB whole-engine statement. The target "
                    "non-KV bytes/rank equal exact checkpoint bytes/rank plus "
                    f"{fraction:g} times the inferred residual. The core "
                    "reserve is adjusted because the replay subtracts its "
                    "architecture-derived weight estimate."
                    + zero_note
                ),
            )
        )
    return tuple(cases)


def build_summary_row(
    report: Mapping[str, Any],
    run_id: str,
    endpoint: ComputeEndpoint,
    reserve: ReserveCase,
    policy: str,
    report_sha256: str,
) -> dict[str, Any]:
    """Flatten headline metrics without discarding the full JSON report."""

    resume = _nested(report, "resume", default={})
    sources = _nested(resume, "source_counts", default={})
    source_all = _nested(
        resume, "source_fractions_of_all_requests", default={}
    )
    workload = _nested(report, "workload", default={})
    all_requests = _nested(resume, "all_request_count", default=0)
    context_infeasible = report.get("context_infeasible_calls", 0)
    restore = _nested(resume, "restore_timing", default={})
    comparison = _nested(
        report, "infinite_hbm_oracle_comparison", default={}
    )
    transfer = _nested(report, "transfer_queue", default={})
    capacity = _nested(report, "capacity", default={})
    return {
        "run_id": run_id,
        "compute_endpoint": endpoint.name,
        "calibration_band": endpoint.band,
        "attention_multiplier": endpoint.attention_multiplier,
        "prompt_compute_scale": _nested(
            report,
            "execution_scope",
            "prompt_compute_scale",
            default=1.0,
        ),
        "prompt_compute_calibration_metadata_sha256": _nested(
            report,
            "experiment",
            "prompt_compute_calibration_metadata_sha256",
        ),
        "reserve_case": reserve.name,
        "baseline": policy,
        "report_schema_version": report.get("schema_version"),
        "tp_size": report.get("tp_size"),
        "max_context_tokens": workload.get("max_context_tokens"),
        "hbm_total_bytes_per_rank": capacity.get(
            "hbm_total_bytes_per_rank"
        ),
        "model_weight_bytes_per_rank_estimate": capacity.get(
            "model_weight_bytes_per_rank_estimate"
        ),
        "common_hbm_reserve_bytes_per_rank": capacity.get(
            "hbm_static_reserve_bytes_per_rank"
        ),
        "prefill_hbm_reserve_bytes_per_rank": capacity.get(
            "prefill_hbm_static_reserve_bytes_per_rank"
        ),
        "decode_hbm_reserve_bytes_per_rank": capacity.get(
            "decode_hbm_static_reserve_bytes_per_rank"
        ),
        "prefill_hbm_kv_budget_bytes_per_rank": capacity.get(
            "prefill_hbm_kv_budget_bytes_per_rank"
        ),
        "decode_hbm_kv_budget_bytes_per_rank": capacity.get(
            "decode_hbm_kv_budget_bytes_per_rank"
        ),
        "all_request_count": all_requests,
        "context_admissible_request_count": all_requests - context_infeasible,
        "reuse_eligible_transition_count": resume.get(
            "reuse_eligible_transition_count"
        ),
        "decode_hbm_resume_count": sources.get("decode_hbm", 0),
        "cpu_resume_count": sources.get("cpu", 0),
        "ssd_resume_count": sources.get("ssd", 0),
        "recompute_count": sources.get("recompute", 0),
        "decode_hbm_resume_fraction_all_requests": source_all.get(
            "decode_hbm", 0.0
        ),
        "cpu_resume_fraction_all_requests": source_all.get("cpu", 0.0),
        "ssd_resume_fraction_all_requests": source_all.get("ssd", 0.0),
        "cpu_or_ssd_resume_fraction_all_requests": resume.get(
            "cpu_or_ssd_resume_fraction_of_all_requests", 0.0
        ),
        "recompute_fraction_all_requests": source_all.get(
            "recompute", 0.0
        ),
        "recompute_fraction_executed_prompt_compute": _nested(
            report,
            "recompute",
            "analytical_time_fraction_of_executed_prompt_compute",
            default=0.0,
        ),
        "restore_raw_request_sum_seconds": restore.get(
            "request_summed_raw_elapsed_seconds", 0.0
        ),
        "restore_exposed_request_sum_seconds": restore.get(
            "request_summed_exposed_compute_admission_gate_seconds", 0.0
        ),
        "restore_exposed_wall_union_seconds": restore.get(
            "wall_clock_exposed_decode_barrier_union_seconds", 0.0
        ),
        "transfer_queue_wait_seconds": transfer.get(
            "aggregate_queue_wait_seconds", 0.0
        ),
        "transfer_service_seconds": transfer.get(
            "aggregate_service_seconds", 0.0
        ),
        "oracle_request_summed_service_slowdown_fraction": _nested(
            comparison,
            "all_calls",
            "slowdown_fraction_of_oracle_request_summed_service",
        ),
        "oracle_session_e2e_slowdown_fraction": _nested(
            comparison,
            "session_end_to_end",
            "slowdown_fraction_of_oracle",
        ),
        "oracle_trace_makespan_slowdown_fraction": _nested(
            comparison,
            "trace_makespan",
            "slowdown_fraction_of_oracle",
        ),
        "request_makespan_seconds": report.get("request_makespan_seconds"),
        "report_sha256": report_sha256,
    }


def build_return_source_rows(
    report: Mapping[str, Any],
    run_id: str,
    endpoint_name: str,
    reserve_name: str,
    policy: str,
) -> list[dict[str, Any]]:
    rows = []
    by_gap = _nested(report, "resume", "by_return_gap_type", default={})
    for gap_type, gap in sorted(by_gap.items()):
        counts = gap.get("source_counts", {})
        fraction_all = gap.get(
            "source_fractions_of_all_requests_in_return_class", {}
        )
        fraction_eligible = gap.get(
            "source_fractions_of_reuse_eligible_in_return_class", {}
        )
        tokens = gap.get("source_reusable_tokens", {})
        for source in ("decode_hbm", "cpu", "ssd", "recompute"):
            rows.append(
                {
                    "run_id": run_id,
                    "compute_endpoint": endpoint_name,
                    "reserve_case": reserve_name,
                    "baseline": policy,
                    "return_gap_type": gap_type,
                    "all_requests_in_class": gap.get("all_request_count", 0),
                    "reuse_eligible_in_class": gap.get(
                        "reuse_eligible_transition_count", 0
                    ),
                    "source": source,
                    "count": counts.get(source, 0),
                    "reusable_tokens": tokens.get(source, 0),
                    "fraction_of_all_requests_in_class": fraction_all.get(
                        source, 0.0
                    ),
                    "fraction_of_reuse_eligible_in_class": (
                        fraction_eligible.get(source, 0.0)
                    ),
                }
            )
    return rows


def build_transfer_stage_rows(
    report: Mapping[str, Any],
    run_id: str,
    endpoint_name: str,
    reserve_name: str,
    policy: str,
) -> list[dict[str, Any]]:
    transfer = _nested(report, "transfer_queue", default={})
    jobs = transfer.get("jobs_by_kind", {})
    byte_counts = transfer.get("bytes_by_kind", {})
    waits = transfer.get("queue_wait_seconds_by_kind", {})
    service = transfer.get("service_seconds_by_kind", {})
    stages = sorted(set(jobs) | set(byte_counts) | set(waits) | set(service))
    return [
        {
            "run_id": run_id,
            "compute_endpoint": endpoint_name,
            "reserve_case": reserve_name,
            "baseline": policy,
            "stage": stage,
            "jobs": jobs.get(stage, 0),
            "bytes": byte_counts.get(stage, 0),
            "queue_wait_seconds": waits.get(stage, 0.0),
            "service_seconds": service.get(stage, 0.0),
        }
        for stage in stages
    ]


def _legacy_h100_source_record(
    path: str, local_source_hashes: Mapping[str, str]
) -> dict[str, Any]:
    model = (
        "meta-llama/Llama-3.1-70B"
        if "/meta-llama/" in path
        else "mistralai/Mixtral-8x7B-v0.1"
    )
    tp_fragment = path.split("/tp", 1)[1].split("/", 1)[0]
    artifact_kind = "attention" if path.endswith("attention.csv") else "layers"
    return {
        "path": path,
        "sha256": local_source_hashes.get(path),
        "model": model,
        "tp_size": int(tp_fragment),
        "artifact_kind": artifact_kind,
        "measurement_dtype": "float16",
        "used": True,
        "use": (
            "Legacy component timing artifact labeled H100, used for "
            "calibration and/or held-out validation of the analytical Qwen "
            "projection. Exact measurement-system identity is unavailable."
        ),
    }


def build_manifest(
    *,
    command: str,
    workload: Mapping[str, Any],
    local_source_hashes: Mapping[str, str],
    git_provenance: Mapping[str, Any],
    architecture_weight_bytes_per_rank: int,
    reserve_cases: Sequence[ReserveCase],
    run_records: Sequence[Mapping[str, Any]],
    compute_endpoints: Sequence[ComputeEndpoint],
    prompt_compute_calibration_metadata: Mapping[
        str, Mapping[str, Any]
    ],
    prompt_compute_calibration_metadata_sha256: Mapping[str, str],
    calibration_artifact: Mapping[str, Any],
    workload_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the machine-readable evidence and calibration boundary."""

    endpoint_names = tuple(endpoint.name for endpoint in compute_endpoints)
    if set(prompt_compute_calibration_metadata) != set(endpoint_names):
        raise AnalysisConfigError(
            "prompt calibration metadata must cover every selected endpoint"
        )
    if set(prompt_compute_calibration_metadata_sha256) != set(endpoint_names):
        raise AnalysisConfigError(
            "prompt calibration metadata hashes must cover every selected "
            "endpoint"
        )
    for endpoint_name in endpoint_names:
        actual_hash = _json_sha256(
            prompt_compute_calibration_metadata[endpoint_name]
        )
        declared_hash = prompt_compute_calibration_metadata_sha256[
            endpoint_name
        ]
        if actual_hash != declared_hash:
            raise AnalysisConfigError(
                "prompt calibration metadata hash mismatch for "
                f"{endpoint_name}: {declared_hash} != {actual_hash}"
            )
    if calibration_artifact.get("endpoint_metadata_sha256") != dict(
        sorted(prompt_compute_calibration_metadata_sha256.items())
    ):
        raise AnalysisConfigError(
            "calibration artifact endpoint hashes do not match manifest "
            "metadata hashes"
        )
    weight_delta = (
        architecture_weight_bytes_per_rank
        - QWEN_CHECKPOINT_BYTES_PER_TP4_RANK
    )
    return {
        "schema_version": 3,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "working_directory": str(Path.cwd().resolve()),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "git": dict(git_provenance),
        "workload": dict(workload),
        "workload_provenance": dict(
            workload_provenance
            if workload_provenance is not None
            else _missing_workload_provenance(None, "not_provided_to_builder")
        ),
        "local_source_sha256": dict(sorted(local_source_hashes.items())),
        "calibration_artifact": dict(calibration_artifact),
        "experiment_contract": {
            "model": MODEL_NAME,
            "hardware": HARDWARE_NAME,
            "pd_layout": "one TP4 prefill replica plus one TP4 decode replica",
            "tp_size_per_role": TP_SIZE,
            "expert_parallel_size_per_role": TP_SIZE,
            "weight_layout_semantics": (
                "Attention and dense tensors are TP4-sharded; 128 MoE "
                "experts are evenly EP4-sharded on the same four GPUs."
            ),
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "prefill_chunk_size": PREFILL_CHUNK_SIZE,
            "block_size": BLOCK_SIZE,
            "kv_dtype_bytes": KV_DTYPE_BYTES,
            "hbm_capacity_bytes_per_rank": HBM_CAPACITY_BYTES_PER_RANK,
            "cpu_capacity_bytes": CPU_CAPACITY_BYTES,
            "ssd_capacity_bytes": SSD_CAPACITY_BYTES,
            "cpu_pcie_gbps_per_rank": CPU_PCIE_GBPS_PER_RANK,
            "cpu_dram_gbps_aggregate": CPU_DRAM_GBPS_AGGREGATE,
            "ssd_read_gbps_aggregate": SSD_READ_GBPS_AGGREGATE,
            "ssd_write_gbps_aggregate": SSD_WRITE_GBPS_AGGREGATE,
            "pd_nvlink_gbps_per_rank_one_way": (
                PD_NVLINK_GBPS_PER_RANK_ONE_WAY
            ),
            "policies": list(POLICIES),
            "demotion_mode": "capacity-only",
            "restore_execution_mode": "async-pre-admission",
            "infinite_hbm_oracle_paired_for_every_run": True,
            "same_prompt_predictor_object_for_finite_and_oracle": True,
            "whole_prompt_compute_scale": 1.0,
        },
        "model_geometry": {
            "qwen_revision": QWEN_REVISION,
            "local_simulator_config_path": QWEN_MODEL_CONFIG_PATH,
            "local_simulator_config_sha256": local_source_hashes.get(
                QWEN_MODEL_CONFIG_PATH
            ),
            "kv_bytes_per_token_per_tp4_rank": (
                KV_BYTES_PER_TOKEN_PER_TP4_RANK
            ),
            "full_context_kv_bytes_per_tp4_rank": (
                FULL_CONTEXT_KV_BYTES_PER_TP4_RANK
            ),
            "kv_storage_semantics": (
                "Retain BF16 K and V for every token at all 48 layers. DCA "
                "and MInference change the compute endpoint but are not "
                "assumed to compress or discard the reusable KV allocation."
            ),
            "tp4_kv_head_replication_factor": 1.0,
            "checkpoint_bytes_total": QWEN_CHECKPOINT_BYTES_TOTAL,
            "checkpoint_bytes_per_tp4_rank": (
                QWEN_CHECKPOINT_BYTES_PER_TP4_RANK
            ),
            "architecture_weight_bytes_per_rank": (
                architecture_weight_bytes_per_rank
            ),
            "architecture_minus_checkpoint_bytes_per_rank": weight_delta,
        },
        "reserve_derivation": {
            "model_card_approx_engine_bytes_total": (
                MODEL_CARD_APPROX_ENGINE_BYTES_TOTAL
            ),
            "model_card_approx_engine_bytes_per_tp4_rank": (
                MODEL_CARD_APPROX_ENGINE_BYTES_PER_TP4_RANK
            ),
            "formula": (
                "residual = 240,000,000,000 / 4 - "
                "61,064,245,248 / 4 - 1,010,000 * 24,576"
            ),
            "inferred_runtime_residual_bytes_per_rank": (
                MODEL_CARD_INFERRED_RUNTIME_RESIDUAL_BYTES_PER_RANK
            ),
            "full_residual_core_reserve_formula": (
                "checkpoint_bytes_per_rank + inferred_residual - "
                "architecture_weight_bytes_per_rank"
            ),
            "interpretation": (
                "Residual inferred from an approximate whole-engine model-card "
                "statement after exact checkpoint and KV subtraction. It is "
                "not a measured activation or runtime reserve."
            ),
            "pd_application": (
                "The sensitivity is applied independently to both P4 roles. "
                "P4+D4 duplicates weights and role-local runtime state; the "
                "model card's approximate 240 GB statement is not itself a "
                "P/D-topology measurement."
            ),
            "cases": [asdict(case) for case in reserve_cases],
        },
        "official_qwen_sources": {
            "repository": QWEN_REPOSITORY_URL,
            "revision": QWEN_REVISION,
            "config_1m": {
                "url": f"{QWEN_REPOSITORY_URL}/blob/{QWEN_REVISION}/config_1m.json",
                "sha256": QWEN_CONFIG_1M_SHA256,
                "bytes": QWEN_CONFIG_1M_BYTES,
                "evidence_class": "official_revision_pinned_model_artifact",
            },
            "model_card": {
                "url": f"{QWEN_REPOSITORY_URL}/blob/{QWEN_REVISION}/README.md",
                "sha256": QWEN_MODEL_CARD_SHA256,
                "bytes": QWEN_MODEL_CARD_BYTES,
                "evidence_class": "official_approximate_system_statement",
            },
            "checkpoint_index": {
                "url": (
                    f"{QWEN_REPOSITORY_URL}/blob/{QWEN_REVISION}/"
                    "model.safetensors.index.json"
                ),
                "sha256": QWEN_CHECKPOINT_INDEX_SHA256,
                "bytes": QWEN_CHECKPOINT_INDEX_BYTES,
                "metadata_total_size": QWEN_CHECKPOINT_BYTES_TOTAL,
                "evidence_class": "official_revision_pinned_model_artifact",
            },
        },
        "hardware_sources": {
            "hash_semantics": (
                "Observed source bytes on 2026-07-20 UTC. NVIDIA web pages "
                "are not revision pinned or vendored, so their hashes identify "
                "the consulted page representations rather than immutable "
                "artifacts. The KIOXIA PDF is a directly hashed product brief."
            ),
            "dgx_h100_system_guide": {
                "url": NVIDIA_DGX_H100_GUIDE_URL,
                "observed_sha256": (
                    NVIDIA_DGX_H100_GUIDE_OBSERVED_SHA256
                ),
                "observed_bytes": NVIDIA_DGX_H100_GUIDE_OBSERVED_BYTES,
                "supports": "80 GB GPUs, 2 TB system memory, 8 x 3.84 TB SSD",
                "evidence_class": "manufacturer_system_specification",
            },
            "dgx_h100_nvme_guide": {
                "url": NVIDIA_DGX_H100_NVME_GUIDE_URL,
                "observed_sha256": (
                    NVIDIA_DGX_H100_NVME_GUIDE_OBSERVED_SHA256
                ),
                "observed_bytes": NVIDIA_DGX_H100_NVME_GUIDE_OBSERVED_BYTES,
                "supports": "example KIOXIA KCM6DRUL3T84 device identity",
                "evidence_class": "manufacturer_system_guide",
            },
            "h100_specification": {
                "url": NVIDIA_H100_SPEC_URL,
                "observed_sha256": NVIDIA_H100_SPEC_OBSERVED_SHA256,
                "observed_bytes": NVIDIA_H100_SPEC_OBSERVED_BYTES,
                "supports": (
                    "H100 roofline and 900 GB/s aggregate NVLink source; "
                    "450 GB/s one-way is an explicit directional inference"
                ),
                "evidence_class": "manufacturer_accelerator_specification",
            },
            "kioxia_cm6_product_brief": {
                "url": KIOXIA_CM6_PRODUCT_BRIEF_URL,
                "sha256": KIOXIA_CM6_PRODUCT_BRIEF_SHA256,
                "bytes": KIOXIA_CM6_PRODUCT_BRIEF_BYTES,
                "supports": "6.9/4.2 GB/s per-device nameplate upper bounds",
                "evidence_class": "manufacturer_storage_specification",
            },
        },
        "evidence_classes": {
            "exact_trace_observation": (
                "Request geometry and inter-turn timing from the hashed input "
                "workload. Token counts remain source-tokenizer surrogates."
            ),
            "exact_model_geometry": (
                "Revision-pinned Qwen config/checkpoint metadata and exact TP4 "
                "KV-head geometry."
            ),
            "official_approximate_statement": (
                "Qwen model-card 240 GB whole-engine and up-to-3x speedup "
                "statements; the speedup defines only a labeled one-third "
                "attention-work sensitivity, not a whole-prompt multiplier."
            ),
            "nominal_capacity_and_manufacturer_upper_bound": (
                "H100 uses its marketed 80 GB SI capacity. Aggregate "
                "eight-drive CM6 55.2/33.6 GB/s is nameplate arithmetic, "
                "not an end-to-end fio measurement."
            ),
            "analytical_platform_assumption": (
                "50 GB/s per-rank PCIe, 400 GB/s shared DRAM, and 450 GB/s "
                "one-way NVLink are explicit unmeasured inputs."
            ),
            "kernel_calibrated_analytical_compute_endpoint": (
                "Kernel-decomposed analytical equations are fitted to legacy "
                "Llama-3.1-70B and Mixtral artifacts labeled H100. Exact H100 "
                "SKU and measurement software are unknown; Qwen geometry, 1M "
                "attention, and DCA/MInference remain extrapolations."
            ),
        },
        "repository_profile_evidence": {
            "legacy_h100_kernel_calibration": [
                _legacy_h100_source_record(path, local_source_hashes)
                for path in LEGACY_H100_CALIBRATION_SOURCE_PATHS
            ],
            "hash_binding": (
                "Every source is hashed from repository bytes at experiment "
                "start. Predictor metadata carries the same source hashes."
            ),
            "current_legacy_producer_snapshot": {
                "source_sha256": {
                    path: local_source_hashes.get(path)
                    for path in LEGACY_H100_PRODUCER_SOURCE_PATHS
                },
                "proven_to_be_measurement_revision": False,
                "scope": (
                    "Hash binding for the current repository implementation; "
                    "the CSVs do not carry a producer commit or environment "
                    "manifest that proves this was the generating revision."
                ),
            },
        },
        "excluded_repository_profile_evidence": {
            "same_model_current_profile": {
                "path": QWEN_RTX_PROFILE_META_PATH,
                "sha256": local_source_hashes.get(
                    QWEN_RTX_PROFILE_META_PATH
                ),
                "hardware": "RTX PRO 6000 Blackwell Server Edition",
                "tp_degrees": [1, 2],
                "attention_backend_scope": "standard vLLM attention",
                "maximum_profiled_attention_kv_tokens": 16_384,
                "used": False,
                "reason": (
                    "Wrong hardware, no TP4 measurement, and no 1M "
                    "DCA/MInference kernel surface. It cannot support an "
                    "absolute H100 1M latency claim."
                ),
            },
        },
        "prompt_compute_calibration": {
            "fit_invocations_this_driver_run": 1,
            "fit_shared_across_selected_endpoints": True,
            "same_endpoint_predictor_object_for_finite_and_oracle": True,
            "whole_prompt_compute_scale": 1.0,
            "method_reference": {
                "name": "KernelSight-LM kernel-level roofline",
                "url": KERNELSIGHT_LM_URL,
                "adapted_equation": KERNELSIGHT_LM_EQUATION,
            },
            "source_sha256": {
                path: local_source_hashes.get(path)
                for path in LEGACY_H100_CALIBRATION_SOURCE_PATHS
            },
            "producer_source_sha256": {
                path: local_source_hashes.get(path)
                for path in LEGACY_H100_PRODUCER_SOURCE_PATHS
            },
            "endpoint_metadata_sha256": dict(
                sorted(prompt_compute_calibration_metadata_sha256.items())
            ),
            "endpoint_metadata": {
                name: dict(prompt_compute_calibration_metadata[name])
                for name in endpoint_names
            },
            "holdout_validation_by_endpoint": {
                name: prompt_compute_calibration_metadata[name].get(
                    "validation"
                )
                for name in endpoint_names
            },
        },
        "calibration_boundary": {
            "measured_qwen3_1m_dca_h100_profile_used": False,
            "measured_qwen3_h100_tp4_profile_used": False,
            "same_model_rtxpro6000_tp1_tp2_amdahl_evidence_used": False,
            "legacy_h100_labeled_kernel_timing_evidence_used": True,
            "legacy_h100_measured_kernel_evidence_used": True,
            "legacy_h100_holdout_validation_used": True,
            "legacy_h100_measurement_dtype": "float16",
            "exact_legacy_h100_sku_known": False,
            "legacy_measurement_clock_state_known": False,
            "legacy_cuda_pytorch_flashattention_versions_known": False,
            "legacy_measurement_command_known": False,
            "current_producer_revision_proven_to_match_csv": False,
            "absolute_dgx_h100_latency_validated": False,
            "target_qwen_compute_dtype": "bfloat16",
            "fp16_to_bf16_kernel_efficiency_transfer_assumed": True,
            "attention_holdout_validates_long_k_or_1m": False,
            "component_holdout_is_target_end_to_end_accuracy": False,
            "collective_latency_measured_or_fitted": False,
            "ep_collective_backend": "allgather_reducescatter",
            "profiler_perf_bundle_created": False,
            "decode_compute_modeled": False,
            "llm_compute_queue_or_continuous_batching_modeled": False,
            "classification": (
                "Legacy-H100-labeled kernel-calibrated analytical capacity, "
                "communication, and transfer-queue sensitivity; not a "
                "measured Qwen3, 1M, or DCA/MInference H100 result."
            ),
        },
        "compute_endpoints": [asdict(item) for item in compute_endpoints],
        "runs": [dict(item) for item in run_records],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise AnalysisConfigError(f"refusing to write empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ("git", *args),
            cwd=repo_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.rstrip("\n")

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain=v1", "--untracked-files=normal")
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "available": False,
            "error": str(exc),
        }
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "status_sha256": _sha256_bytes(status.encode("utf-8")),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Qwen3-30B-A3B 1M P4+D4 cold-KV capacity baselines "
            "and paired infinite-HBM references."
        )
    )
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument(
        "--workload-manifest",
        type=Path,
        help=(
            "Conversion-provenance sidecar. When omitted, the driver checks "
            "<workload>.manifest.json. A missing sidecar is recorded rather "
            "than silently treated as complete provenance."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compute-endpoint",
        dest="compute_endpoints",
        action="append",
        choices=tuple(endpoint.name for endpoint in COMPUTE_ENDPOINTS),
        help=(
            "Compute endpoint to run; repeat to select multiple endpoints. "
            "The default runs central_full_attention and "
            "central_attention_one_third. Fast/slow full-attention bands are "
            "optional fit-uncertainty sensitivities."
        ),
    )
    parser.add_argument(
        "--reserve-sweep",
        choices=("full", "half", "zero", "all"),
        default="full",
        help=(
            "Runtime/activation residual sensitivity. The primary full case "
            "uses an exact 19,893,012,480-byte core reserve with the current "
            "architecture weight formula."
        ),
    )
    parser.add_argument(
        "--hbm-static-reserve-bytes-per-rank",
        type=int,
        help="Exact custom common non-weight, non-KV reserve.",
    )
    parser.add_argument(
        "--prefill-hbm-static-reserve-bytes-per-rank",
        type=int,
        help="Exact custom P-role non-weight, non-KV reserve.",
    )
    parser.add_argument(
        "--decode-hbm-static-reserve-bytes-per-rank",
        type=int,
        help="Exact custom D-role non-weight, non-KV reserve.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this driver's known output files if they exist.",
    )
    return parser


def _resolved_command(argv: Sequence[str] | None) -> str:
    supplied = list(sys.argv[1:] if argv is None else argv)
    return shlex.join(
        [sys.executable, "-m", "serving.agentic_kv_qwen3_1m_p4d4", *supplied]
    )


def run_experiment(
    args: argparse.Namespace, command: str
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    compute_endpoints = resolve_compute_endpoints(
        getattr(args, "compute_endpoints", None)
    )
    if tuple(KERNEL_CALIBRATION_SOURCE_PATHS) != (
        LEGACY_H100_CALIBRATION_SOURCE_PATHS
    ):
        raise AnalysisConfigError(
            "driver and kernel model disagree on H100 calibration sources"
        )
    if QWEN_TP != TP_SIZE or QWEN_EP != TP_SIZE:
        raise AnalysisConfigError(
            "driver and kernel model disagree on Qwen TP/EP geometry"
        )
    model = load_model_shape(MODEL_NAME, repo_root)
    expected_model_geometry = {
        "hidden_size": QWEN_HIDDEN_SIZE,
        "num_hidden_layers": QWEN_LAYERS,
        "num_attention_heads": QWEN_Q_HEADS,
        "num_key_value_heads": QWEN_KV_HEADS,
        "head_dim": QWEN_HEAD_DIM,
        "intermediate_size": 6_144,
        "num_experts": QWEN_EXPERTS,
        "num_experts_per_tok": QWEN_TOP_K,
        "moe_intermediate_size": QWEN_EXPERT_INTERMEDIATE,
        "model_type": "qwen3_moe",
    }
    actual_model_geometry = {
        field: getattr(model, field) for field in expected_model_geometry
    }
    if actual_model_geometry != expected_model_geometry:
        raise AnalysisConfigError(
            "Qwen target config geometry changed; update and re-review the "
            f"kernel model constants: {actual_model_geometry!r}"
        )
    layout = kv_layout(model, TP_SIZE, KV_DTYPE_BYTES)
    if (
        layout.physical_bytes_per_token_per_rank
        != KV_BYTES_PER_TOKEN_PER_TP4_RANK
        or layout.replication_factor != 1.0
    ):
        raise AnalysisConfigError(
            "Qwen3 TP4 KV geometry changed; update and re-review the evidence "
            "manifest constants before running"
        )
    architecture_weight = estimate_model_weight_bytes_per_rank(
        model, TP_SIZE, WEIGHT_DTYPE_BYTES, layout
    )
    reserve_cases = derive_reserve_cases(
        architecture_weight,
        args.reserve_sweep,
        args.hbm_static_reserve_bytes_per_rank,
        args.prefill_hbm_static_reserve_bytes_per_rank,
        args.decode_hbm_static_reserve_bytes_per_rank,
    )
    workload = load_capacity_replay_workload(
        args.workload,
        block_size=BLOCK_SIZE,
        max_context_tokens=MAX_CONTEXT_TOKENS,
    )
    workload_provenance = load_workload_provenance(
        args.workload,
        workload.sha256,
        args.workload_manifest,
    )
    context_infeasible_calls = sum(
        not call.context_eligible
        for session in workload.sessions
        for call in session.calls
    )
    if context_infeasible_calls:
        raise AnalysisConfigError(
            f"the 1.01M experiment would censor {context_infeasible_calls} "
            "calls; use a workload whose full prompt-plus-output sequences "
            "fit the declared model window"
        )
    hardware = override_transfer_defaults(
        load_hardware_config()[HARDWARE_NAME],
        cpu_rank_gbps=CPU_PCIE_GBPS_PER_RANK,
        cpu_aggregate_gbps=CPU_DRAM_GBPS_AGGREGATE,
        ssd_read_gbps=SSD_READ_GBPS_AGGREGATE,
        ssd_write_gbps=SSD_WRITE_GBPS_AGGREGATE,
    )
    hardware = replace(
        hardware,
        calibration_provenance=(
            "Kernel-decomposed analytical prompt model fitted to legacy "
            "Llama-3.1-70B and Mixtral timing artifacts labeled H100. Exact "
            "H100 SKU and software stack are unavailable. Projection to "
            "Qwen3, one-million-token attention, and DCA/MInference is "
            "analytical, not a measured Qwen3 H100 result."
        ),
        cpu=replace(
            hardware.cpu,
            provenance=(
                "Unmeasured experiment input: 50 GB/s effective per-rank "
                "PCIe and one shared 400 GB/s CPU-DRAM queue."
            ),
        ),
        ssd=replace(
            hardware.ssd,
            provenance=(
                "Manufacturer upper-bound sensitivity: eight KIOXIA CM6 "
                "drives at 6.9/4.2 GB/s each, CPU staged; not measured fio."
            ),
        ),
    )

    known_top_level = (
        "calibration.json",
        "summary.csv",
        "return_sources.csv",
        "transfer_stages.csv",
        "manifest.json",
    )
    if not args.overwrite:
        expected_reports = [
            args.output_dir
            / f"{reserve.name}__{endpoint.name}__{policy}.json"
            for reserve in reserve_cases
            for endpoint in compute_endpoints
            for policy in POLICIES
        ]
        conflicts = [
            str(args.output_dir / name)
            for name in known_top_level
            if (args.output_dir / name).exists()
        ]
        conflicts.extend(str(path) for path in expected_reports if path.exists())
        if conflicts:
            raise AnalysisConfigError(
                "output files already exist; pass --overwrite: "
                + ", ".join(conflicts)
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local_paths = (
        Path(__file__).resolve(),
        repo_root / "serving/core/agentic_kv_capacity_replay.py",
        repo_root / "serving/core/agentic_kv_roofline.py",
        repo_root / "serving/core/h100_kernel_calibrated_prompt.py",
        repo_root / QWEN_MODEL_CONFIG_PATH,
        repo_root / QWEN_RTX_PROFILE_META_PATH,
        *(repo_root / path for path in LEGACY_H100_CALIBRATION_SOURCE_PATHS),
        *(repo_root / path for path in LEGACY_H100_PRODUCER_SOURCE_PATHS),
    )
    local_hashes = {
        str(path.relative_to(repo_root)): _sha256_file(path)
        for path in local_paths
    }
    git_at_start = _git_provenance(repo_root)

    calibration = fit_h100_tp4_calibration(repo_root)
    target_config_sha256 = local_hashes[QWEN_MODEL_CONFIG_PATH]
    prompt_models: dict[str, H100KernelCalibratedPromptModel] = {}
    prompt_metadata: dict[str, dict[str, Any]] = {}
    prompt_metadata_hashes: dict[str, str] = {}
    for endpoint in compute_endpoints:
        predictor = H100KernelCalibratedPromptModel(
            calibration=calibration,
            band=endpoint.band,
            attention_multiplier=endpoint.attention_multiplier,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
            target_config_sha256=target_config_sha256,
        )
        metadata = predictor.metadata()
        if not isinstance(metadata, Mapping):
            raise AnalysisConfigError(
                f"prompt model metadata is not a mapping: {endpoint.name}"
            )
        prompt_models[endpoint.name] = predictor
        prompt_metadata[endpoint.name] = dict(metadata)
        prompt_metadata_hashes[endpoint.name] = _json_sha256(metadata)

    expected_calibration_hashes = {
        path: local_hashes[path]
        for path in LEGACY_H100_CALIBRATION_SOURCE_PATHS
    }
    if dict(calibration.source_sha256) != expected_calibration_hashes:
        raise AnalysisConfigError(
            "fitted calibration source hashes do not match experiment-time "
            "source hashes"
        )
    expected_producer_hashes = {
        path: local_hashes[path]
        for path in LEGACY_H100_PRODUCER_SOURCE_PATHS
    }
    if dict(calibration.producer_source_sha256) != expected_producer_hashes:
        raise AnalysisConfigError(
            "fitted calibration producer hashes do not match "
            "experiment-time source hashes"
        )
    for endpoint_name, metadata in prompt_metadata.items():
        if metadata.get("source_sha256") != expected_calibration_hashes:
            raise AnalysisConfigError(
                "prompt metadata source hashes do not match for endpoint "
                f"{endpoint_name}"
            )
        if metadata.get("producer_source_sha256") != expected_producer_hashes:
            raise AnalysisConfigError(
                "prompt metadata producer hashes do not match for endpoint "
                f"{endpoint_name}"
            )
        if _nested(metadata, "target_geometry", "config_sha256") != (
            target_config_sha256
        ):
            raise AnalysisConfigError(
                "prompt metadata target config hash does not match for "
                f"endpoint {endpoint_name}"
            )

    base_calibration_metadata = calibration.metadata()
    calibration_payload = {
        "schema_version": 3,
        "fit_invocations_this_driver_run": 1,
        "fit_shared_across_selected_endpoints": True,
        "source_unit_contract": {
            "layers.csv": (
                "latency(ns) is parsed as nanoseconds and multiplied by 1e-9"
            ),
            "attention.csv": (
                "time_stats.attn_prefill.median is parsed as milliseconds "
                "and multiplied by 1e-3"
            ),
        },
        "source_work_contract": {
            "measurement_dtype": "float16",
            "target_dtype": "bfloat16",
            "dtype_efficiency_transfer_assumed": True,
            "attention_causal_pairs": (
                "bottom-right aligned q*(k-q)+q*(q+1)/2"
            ),
            "ep_collective_backend": "allgather_reducescatter",
            "collective_terms_measured_or_fitted": False,
            "legacy_h100_artifact_label_only": True,
            "exact_h100_sku_and_software_stack_known": False,
            "current_producer_revision_proven_to_match_csv": False,
            "target_config_sha256": target_config_sha256,
        },
        "base_calibration_metadata_sha256": _json_sha256(
            base_calibration_metadata
        ),
        "base_calibration_metadata": base_calibration_metadata,
        "endpoints": {
            endpoint.name: {
                "configuration": asdict(endpoint),
                "metadata_sha256": prompt_metadata_hashes[endpoint.name],
                "metadata": prompt_metadata[endpoint.name],
            }
            for endpoint in compute_endpoints
        },
    }
    calibration_path = args.output_dir / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            calibration_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    calibration_artifact = {
        "path": calibration_path.name,
        "sha256": _sha256_file(calibration_path),
        "bytes": calibration_path.stat().st_size,
        "schema_version": calibration_payload["schema_version"],
        "base_calibration_metadata_sha256": calibration_payload[
            "base_calibration_metadata_sha256"
        ],
        "endpoint_metadata_sha256": dict(
            sorted(prompt_metadata_hashes.items())
        ),
    }

    summary_rows: list[dict[str, Any]] = []
    return_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    for reserve in reserve_cases:
        for endpoint in compute_endpoints:
            predictor = prompt_models[endpoint.name]
            metadata_hash = prompt_metadata_hashes[endpoint.name]
            for policy in POLICIES:
                run_id = f"{reserve.name}__{endpoint.name}__{policy}"
                report_path = args.output_dir / f"{run_id}.json"
                if report_path.exists() and not args.overwrite:
                    raise AnalysisConfigError(
                        f"output report already exists: {report_path}"
                    )
                print(f"Running {run_id}", flush=True)
                config = CapacityReplayConfig(
                    hbm_capacity_bytes_per_rank=HBM_CAPACITY_BYTES_PER_RANK,
                    cpu_capacity_bytes=CPU_CAPACITY_BYTES,
                    ssd_capacity_bytes=SSD_CAPACITY_BYTES,
                    hbm_static_reserve_bytes_per_rank=(
                        reserve.common_bytes_per_rank
                    ),
                    prefill_hbm_static_reserve_bytes_per_rank=(
                        reserve.prefill_bytes_per_rank
                    ),
                    decode_hbm_static_reserve_bytes_per_rank=(
                        reserve.decode_bytes_per_rank
                    ),
                    policy=policy,
                    demotion_mode="capacity-only",
                    block_size=BLOCK_SIZE,
                    prefill_chunk_size=PREFILL_CHUNK_SIZE,
                    enable_transfer_queueing=True,
                    cancel_migration_on_resume=False,
                    weight_dtype_bytes=WEIGHT_DTYPE_BYTES,
                    pd_disaggregated=True,
                    pd_link_gbps_per_rank=(
                        PD_NVLINK_GBPS_PER_RANK_ONE_WAY
                    ),
                    pd_fixed_latency_us=PD_FIXED_LATENCY_US,
                    restore_execution_mode="async-pre-admission",
                    prompt_compute_scale=1.0,
                    prompt_compute_scale_provenance=(
                        "Identity whole-prompt scale. Compute-band and "
                        "attention-only sensitivity are represented inside "
                        "the calibrated prompt predictor: "
                        + endpoint.provenance
                    ),
                )
                report = replay_capacity_aware_with_oracle(
                    workload,
                    model,
                    hardware,
                    TP_SIZE,
                    KV_DTYPE_BYTES,
                    config,
                    predictor,
                )
                if "infinite_hbm_oracle_comparison" not in report:
                    raise AssertionError(f"{run_id} is missing its paired oracle")
                embedded_metadata = _nested(
                    report,
                    "execution_scope",
                    "prompt_compute_calibration",
                    default={},
                )
                if _json_sha256(embedded_metadata) != metadata_hash:
                    raise AssertionError(
                        f"{run_id} prompt calibration metadata changed in replay"
                    )
                report["experiment"] = {
                    "driver": "serving.agentic_kv_qwen3_1m_p4d4",
                    "run_id": run_id,
                    "compute_endpoint": asdict(endpoint),
                    "prompt_compute_calibration_metadata_sha256": (
                        metadata_hash
                    ),
                    "prompt_predictor_shared_by_finite_and_oracle": True,
                    "reserve_case": asdict(reserve),
                    "classification": (
                        "Legacy-H100 kernel-calibrated analytical projection; "
                        "not a measured Qwen3 DCA/MInference H100 profile."
                    ),
                    "paired_infinite_hbm_oracle": True,
                }
                write_capacity_report(report, report_path)
                report_hash = _sha256_file(report_path)
                summary_rows.append(
                    build_summary_row(
                        report,
                        run_id,
                        endpoint,
                        reserve,
                        policy,
                        report_hash,
                    )
                )
                return_rows.extend(
                    build_return_source_rows(
                        report,
                        run_id,
                        endpoint.name,
                        reserve.name,
                        policy,
                    )
                )
                transfer_rows.extend(
                    build_transfer_stage_rows(
                        report,
                        run_id,
                        endpoint.name,
                        reserve.name,
                        policy,
                    )
                )
                run_records.append(
                    {
                        "run_id": run_id,
                        "report": report_path.name,
                        "report_sha256": report_hash,
                        "report_schema_version": report.get("schema_version"),
                        "policy": policy,
                        "compute_endpoint": endpoint.name,
                        "calibration_band": endpoint.band,
                        "attention_multiplier": endpoint.attention_multiplier,
                        "prompt_compute_calibration_metadata_sha256": (
                            metadata_hash
                        ),
                        "reserve_case": reserve.name,
                        "resolved_config": asdict(config),
                        "paired_infinite_hbm_oracle": True,
                    }
                )

    summary_path = args.output_dir / "summary.csv"
    returns_path = args.output_dir / "return_sources.csv"
    transfers_path = args.output_dir / "transfer_stages.csv"
    _write_csv(summary_path, summary_rows)
    _write_csv(returns_path, return_rows)
    _write_csv(transfers_path, transfer_rows)
    tables = {
        path.name: {
            "sha256": _sha256_file(path),
            "rows": sum(1 for _ in path.open("r", encoding="utf-8")) - 1,
        }
        for path in (summary_path, returns_path, transfers_path)
    }
    all_calls = [
        call
        for session in workload.sessions
        for call in session.calls
    ]
    input_distribution = _token_distribution(
        [call.input_tokens for call in all_calls]
    )
    total_sequence_distribution = _token_distribution(
        [call.total_sequence_tokens for call in all_calls]
    )
    above_native = sum(
        call.total_sequence_tokens > 262_144 for call in all_calls
    )
    manifest = build_manifest(
        command=command,
        workload={
            **workload.metadata_dict(),
            "resolved_path": str(args.workload.resolve()),
            "token_count_provenance": (
                "TraceLab source-provider token-count surrogates; not exact "
                "Qwen tokenizer IDs. No request is padded to 1M."
            ),
            "input_token_distribution": input_distribution,
            "prompt_plus_output_token_distribution": (
                total_sequence_distribution
            ),
            "calls_above_native_262144": above_native,
            "fraction_calls_above_native_262144": (
                above_native / len(all_calls)
            ),
        },
        local_source_hashes=local_hashes,
        git_provenance=git_at_start,
        architecture_weight_bytes_per_rank=architecture_weight,
        reserve_cases=reserve_cases,
        run_records=run_records,
        compute_endpoints=compute_endpoints,
        prompt_compute_calibration_metadata=prompt_metadata,
        prompt_compute_calibration_metadata_sha256=(
            prompt_metadata_hashes
        ),
        calibration_artifact=calibration_artifact,
        workload_provenance=workload_provenance,
    )
    manifest["tables"] = tables
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}", flush=True)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        run_experiment(args, _resolved_command(argv))
    except (AnalysisConfigError, FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

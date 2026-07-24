"""H100-measured kernel calibration for the Qwen3 TP4 prompt replay.

This module deliberately does not create a current-profiler bundle.  It fits
dimensionless roofline residuals to the repository's legacy H100 TP4 kernel
measurements, then evaluates Qwen3's own TP4/EP4 shapes.  The resulting model
is an analytical extrapolation, not a measured Qwen3 or DCA/MInference run.

The model follows the KernelSight-LM structure::

    t = max(t0, t_roof * eta)
    t_roof = max((F / P_peak) * u, bytes / BW_peak)

where ``u`` is a partial-wave penalty when a defensible thread-block proxy is
available.  Legacy GEMM traces do not preserve cuBLAS tile selection, so their
occupancy is intentionally folded into the fitted ``eta``.  Standard prefill
attention uses a documented 128-query tile proxy and H100's 132 SMs.
"""

from __future__ import annotations

import csv
import hashlib
import math
import statistics
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


H100_DENSE_BF16_FLOPS_PER_SECOND = 989.5e12
H100_HBM_BYTES_PER_SECOND = 3.35e12
H100_SMS = 132
BF16_BYTES = 2

QWEN_HIDDEN_SIZE = 2_048
QWEN_HEAD_DIM = 128
QWEN_Q_HEADS = 32
QWEN_KV_HEADS = 4
QWEN_LAYERS = 48
QWEN_EXPERTS = 128
QWEN_TOP_K = 8
QWEN_EXPERT_INTERMEDIATE = 768
QWEN_VOCAB_SIZE = 151_936
QWEN_TP = 4
QWEN_EP = 4

QWEN_NATIVE_MAX_POSITION_EMBEDDINGS = 262_144
QWEN_VALIDATED_RUNTIME_MAX_MODEL_LEN = 1_010_000
QWEN_LONG_CONTEXT_MODE = "dca_dense_full_attention_sensitivity"
QWEN_OFFICIAL_CONFIG_1M_REVISION = (
    "fcc1528445189b27dbe3139c2cf9a1139203a4ff"
)
QWEN_OFFICIAL_CONFIG_1M_SOURCE = (
    "https://huggingface.co/Qwen/"
    "Qwen3-30B-A3B-Instruct-2507/blob/"
    f"{QWEN_OFFICIAL_CONFIG_1M_REVISION}/config_1m.json"
)

TP_COLLECTIVE_BYTES_PER_SECOND = 450.0e9
TP_COLLECTIVE_FIXED_SECONDS = 3.0e-6
LATENCY_CACHE_MAX_ENTRIES = 131_072

LLAMA70_LAYERS = (
    "profiler/v0/perf_models/H100/meta-llama/"
    "Llama-3.1-70B/tp4/layers.csv"
)
LLAMA70_ATTENTION = (
    "profiler/v0/perf_models/H100/meta-llama/"
    "Llama-3.1-70B/tp4/attention.csv"
)
MIXTRAL_LAYERS = (
    "profiler/v0/perf_models/H100/mistralai/"
    "Mixtral-8x7B-v0.1/tp4/layers.csv"
)
MIXTRAL_ATTENTION = (
    "profiler/v0/perf_models/H100/mistralai/"
    "Mixtral-8x7B-v0.1/tp4/attention.csv"
)
CALIBRATION_SOURCE_PATHS = (
    LLAMA70_LAYERS,
    LLAMA70_ATTENTION,
    MIXTRAL_LAYERS,
    MIXTRAL_ATTENTION,
)
LEGACY_PRODUCER_SOURCE_PATHS = (
    "profiler/v0/profiler/attention/main.py",
    "profiler/v0/profiler/attention/attention_profiler.py",
    "profiler/v0/profiler/attention/attention_input.py",
    "profiler/v0/profiler/attention/batch_sampling.py",
    "profiler/v0/profiler/layers/main.py",
    "profiler/v0/profiler/common/timer.py",
    "profiler/v0/profiler/common/timer_stats_store.py",
    "profiler/v0/profiler/utils/__init__.py",
    "profiler/v0/profiler/utils/record_function_tracer.py",
    "profiler/v0/models/llama.py",
    "profiler/v0/models/mixtral.py",
)


def qwen_long_context_experiment_contract() -> dict[str, Any]:
    """Return the exact, paper-facing contract for the Qwen 1M sweep.

    The official 1M configuration enables sparse attention.  This simulator
    does not claim to reproduce that kernel.  It borrows the official DCA
    length-extension envelope, explicitly disables the sparse path for the
    experiment, and evaluates a dense/full-attention analytical sensitivity.
    Keeping the official setting and the experimental override in distinct
    fields prevents the sensitivity from being mistaken for a measured DCA
    or MInference result.
    """

    return {
        "mode": QWEN_LONG_CONTEXT_MODE,
        "native_max_position_embeddings": (
            QWEN_NATIVE_MAX_POSITION_EMBEDDINGS
        ),
        "validated_runtime_max_model_len": (
            QWEN_VALIDATED_RUNTIME_MAX_MODEL_LEN
        ),
        "official_config_1m": {
            "revision": QWEN_OFFICIAL_CONFIG_1M_REVISION,
            "source": QWEN_OFFICIAL_CONFIG_1M_SOURCE,
            "dual_chunk_attention_config": {
                "chunk_size": 131_072,
                "local_size": 4_096,
                "original_max_position_embeddings": 131_072,
                "sparse_attention_enabled": True,
            },
        },
        "experiment_override": {
            "sparse_attention_enabled": False,
            "attention_execution_model": (
                "standard_full_attention_roofline_extrapolation"
            ),
            "dca_or_minference_kernel_latency_calibrated": False,
        },
        "paper_interpretation": {
            "primary_claim_scope": (
                "matched-workload relative cache-policy comparison under "
                "one fixed calibration band"
            ),
            "absolute_1m_latency": "sensitivity_only",
        },
        "claim_boundary": (
            "Legacy-H100 kernel-family-fitted analytical dense/full-"
            "attention sensitivity using the DCA length envelope; not an "
            "official sparse-DCA kernel reproduction or a measured "
            "Qwen3/DCA 1M absolute-latency validation"
        ),
    }


@dataclass(frozen=True)
class _SourceSpec:
    name: str
    hidden: int
    q_heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tp: int = 4

    @property
    def q_dim_per_rank(self) -> int:
        return self.q_heads * self.head_dim // self.tp

    @property
    def kv_dim_per_rank(self) -> int:
        return self.kv_heads * self.head_dim // self.tp


LLAMA70_SPEC = _SourceSpec(
    name="meta-llama/Llama-3.1-70B",
    hidden=8_192,
    q_heads=64,
    kv_heads=8,
    head_dim=128,
    intermediate=28_672,
    vocab=128_256,
)
MIXTRAL_SPEC = _SourceSpec(
    name="mistralai/Mixtral-8x7B-v0.1",
    hidden=4_096,
    q_heads=32,
    kv_heads=8,
    head_dim=128,
    intermediate=14_336,
    vocab=32_000,
)


@dataclass(frozen=True)
class KernelWork:
    flops: float
    bytes: float
    wave_penalty: float = 1.0

    def roofline_seconds(self) -> float:
        if self.flops < 0 or self.bytes < 0 or self.wave_penalty < 1:
            raise ValueError("kernel work must be nonnegative with u >= 1")
        return max(
            self.flops
            / H100_DENSE_BF16_FLOPS_PER_SECOND
            * self.wave_penalty,
            self.bytes / H100_HBM_BYTES_PER_SECOND,
        )


@dataclass(frozen=True)
class KernelFit:
    family: str
    eta_fast: float
    eta_central: float
    eta_slow: float
    launch_floor_seconds: float
    training_rows: int
    holdout_rows: int
    holdout_mape: float
    holdout_p90_ape: float
    holdout_signed_mean: float
    source_models: tuple[str, ...]
    occupancy_contract: str

    def eta(self, band: str) -> float:
        if band == "fast":
            return self.eta_fast
        if band == "central":
            return self.eta_central
        if band == "slow":
            return self.eta_slow
        raise ValueError(f"unsupported calibration band: {band}")


@dataclass(frozen=True)
class H100TP4Calibration:
    fits: Mapping[str, KernelFit]
    source_sha256: Mapping[str, str]
    producer_source_sha256: Mapping[str, str]
    source_rows: Mapping[str, int]
    source_row_accounting: Mapping[str, Mapping[str, int]]
    duplicate_attention_points_coalesced: int
    validation: Mapping[str, Any]

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "target_long_context_experiment": (
                qwen_long_context_experiment_contract()
            ),
            "source_sha256": dict(sorted(self.source_sha256.items())),
            "producer_source_sha256": dict(
                sorted(self.producer_source_sha256.items())
            ),
            "source_identity": {
                "artifact_label": "legacy-H100-labeled",
                "exact_h100_sku_known": False,
                "measurement_clock_state_known": False,
                "cuda_pytorch_flashattention_versions_known": False,
                "measurement_command_known": False,
                "current_producer_revision_proven_to_have_generated_csv": (
                    False
                ),
                "producer_hash_scope": (
                    "Current repository snapshot of the relevant legacy "
                    "producer implementation; it is not a recovered "
                    "measurement-environment manifest."
                ),
            },
            "source_rows": dict(sorted(self.source_rows.items())),
            "source_row_accounting": {
                path: dict(sorted(accounting.items()))
                for path, accounting in sorted(
                    self.source_row_accounting.items()
                )
            },
            "duplicate_attention_points_coalesced": (
                self.duplicate_attention_points_coalesced
            ),
            "kernel_fits": {
                family: asdict(kernel_fit)
                for family, kernel_fit in sorted(self.fits.items())
            },
            "validation": dict(self.validation),
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _gemm_work(m: int, k: int, n: int) -> KernelWork:
    return KernelWork(
        flops=2.0 * m * k * n,
        bytes=BF16_BYTES * (m * k + k * n + m * n),
    )


def _elementwise_work(
    m: int, width: int, flops_per_element: float, byte_passes: float
) -> KernelWork:
    return KernelWork(
        flops=flops_per_element * m * width,
        bytes=BF16_BYTES * byte_passes * m * width,
    )


def _attention_work(
    *,
    q_tokens: int,
    k_tokens: int,
    causal_pairs: float,
    q_heads_per_rank: int,
    kv_heads_per_rank: int,
    head_dim: int,
) -> KernelWork:
    blocks = max(1, math.ceil(q_tokens / 128) * q_heads_per_rank)
    wave_penalty = math.ceil(blocks / H100_SMS) * H100_SMS / blocks
    return KernelWork(
        flops=(
            causal_pairs
            * q_heads_per_rank
            * (4.0 * head_dim + 5.0)
        ),
        bytes=BF16_BYTES
        * (
            2 * q_tokens * q_heads_per_rank * head_dim
            + 2 * k_tokens * kv_heads_per_rank * head_dim
        ),
        wave_penalty=wave_penalty,
    )


def _bottom_right_causal_pairs(q_tokens: int, k_tokens: int) -> float:
    """Return visible Q/K pairs for bottom-right aligned causal attention."""

    if q_tokens <= 0 or k_tokens < q_tokens:
        raise ValueError("causal attention requires k_tokens >= q_tokens > 0")
    prefix_tokens = k_tokens - q_tokens
    return float(
        q_tokens * prefix_tokens + q_tokens * (q_tokens + 1) / 2
    )


def _source_layer_family_and_work(
    spec: _SourceSpec, layer_name: str, tokens: int
) -> tuple[str, KernelWork] | None:
    if layer_name == "attn":
        # The standalone attention CSV is the only accepted attention source.
        return None
    if layer_name == "embedding":
        return "embedding", _elementwise_work(tokens, spec.hidden, 0.0, 1.0)
    if layer_name in {
        "input_layernorm",
        "post_layernorm",
        "final_layernorm",
    }:
        return "norm", _elementwise_work(tokens, spec.hidden, 8.0, 2.0)
    if layer_name == "q_proj":
        return "q_projection", _gemm_work(
            tokens, spec.hidden, spec.q_dim_per_rank
        )
    if layer_name in {"k_proj", "v_proj"}:
        return "kv_projection", _gemm_work(
            tokens, spec.hidden, spec.kv_dim_per_rank
        )
    if layer_name == "o_proj":
        return "o_projection", _gemm_work(
            tokens, spec.q_dim_per_rank, spec.hidden
        )
    if layer_name == "rope":
        return "rope", _elementwise_work(
            tokens,
            spec.q_dim_per_rank + spec.kv_dim_per_rank,
            10.0,
            2.0,
        )
    if layer_name in {"gate_proj", "up_proj"}:
        return "dense_ffn_up", _gemm_work(
            tokens, spec.hidden, spec.intermediate // spec.tp
        )
    if layer_name == "down_proj":
        return "dense_ffn_down", _gemm_work(
            tokens, spec.intermediate // spec.tp, spec.hidden
        )
    if layer_name == "gate":
        # Mixtral's router is not TP-sharded in the legacy model.
        return "router", _gemm_work(tokens, spec.hidden, 8)
    if layer_name in {"expert.w1", "expert.w3"}:
        return "expert_up", _gemm_work(
            tokens, spec.hidden, spec.intermediate
        )
    if layer_name == "expert.w2":
        return "expert_down", _gemm_work(
            tokens, spec.intermediate, spec.hidden
        )
    if layer_name == "act_fn":
        family = (
            "expert_activation"
            if spec.name == MIXTRAL_SPEC.name
            else "dense_activation"
        )
        width = (
            spec.intermediate
            if family == "expert_activation"
            else spec.intermediate // spec.tp
        )
        return family, _elementwise_work(tokens, width, 8.0, 3.0)
    if layer_name == "lm_head":
        return "lm_head", _gemm_work(
            tokens, spec.hidden, spec.vocab // spec.tp
        )
    return None


@dataclass(frozen=True)
class _FitRow:
    family: str
    source_model: str
    tokens: int
    measured_seconds: float
    work: KernelWork

    @property
    def eta(self) -> float:
        roof = self.work.roofline_seconds()
        return max(1.0, self.measured_seconds / max(roof, 1e-15))


def _fit_rows(
    family: str,
    rows: Sequence[_FitRow],
    *,
    training_predicate,
    holdout_predicate,
    occupancy_contract: str,
) -> KernelFit:
    training = [row for row in rows if training_predicate(row)]
    holdout = [row for row in rows if holdout_predicate(row)]
    if not training or not holdout:
        raise ValueError(
            f"calibration family {family} lacks train/holdout coverage"
        )
    etas = [row.eta for row in training]
    eta_fast = _quantile(etas, 0.10)
    eta_central = statistics.median(etas)
    eta_slow = _quantile(etas, 0.90)
    launch_candidates = [
        row.measured_seconds for row in rows if row.tokens <= 16
    ]
    if not launch_candidates:
        minimum_tokens = min(row.tokens for row in rows)
        launch_candidates = [
            row.measured_seconds
            for row in rows
            if row.tokens == minimum_tokens
        ]
    launch_floor = _quantile(launch_candidates, 0.10)
    signed_errors = []
    absolute_errors = []
    for row in holdout:
        prediction = max(
            launch_floor,
            row.work.roofline_seconds() * eta_central,
        )
        signed = (prediction - row.measured_seconds) / row.measured_seconds
        signed_errors.append(signed)
        absolute_errors.append(abs(signed))
    return KernelFit(
        family=family,
        eta_fast=eta_fast,
        eta_central=eta_central,
        eta_slow=eta_slow,
        launch_floor_seconds=launch_floor,
        training_rows=len(training),
        holdout_rows=len(holdout),
        holdout_mape=sum(absolute_errors) / len(absolute_errors),
        holdout_p90_ape=_quantile(absolute_errors, 0.90),
        holdout_signed_mean=sum(signed_errors) / len(signed_errors),
        source_models=tuple(sorted({row.source_model for row in rows})),
        occupancy_contract=occupancy_contract,
    )


def _load_layer_rows(
    path: Path, spec: _SourceSpec
) -> tuple[list[_FitRow], int, dict[str, int]]:
    rows: list[_FitRow] = []
    raw_count = 0
    with path.open("r", encoding="utf-8", newline="") as input_file:
        for raw in csv.DictReader(input_file):
            raw_count += 1
            if int(raw["tp_size"]) != spec.tp or int(raw["kv_cache"]) != 0:
                raise ValueError(
                    f"incompatible legacy layer row in {path}: expected "
                    f"tp_size={spec.tp} and kv_cache=0"
                )
            tokens = int(raw["input"])
            if tokens <= 0:
                raise ValueError(
                    f"incompatible legacy layer row in {path}: input <= 0"
                )
            resolved = _source_layer_family_and_work(
                spec, raw["layer_name"], tokens
            )
            if resolved is None:
                continue
            family, work = resolved
            rows.append(
                _FitRow(
                    family=family,
                    source_model=spec.name,
                    tokens=tokens,
                    measured_seconds=float(raw["latency(ns)"]) * 1e-9,
                    work=work,
                )
            )
    return rows, raw_count, {
        "raw_rows": raw_count,
        "eligible_component_rows": len(rows),
    }


def _load_attention_rows(
    path: Path, spec: _SourceSpec
) -> tuple[list[_FitRow], int, int, dict[str, int]]:
    raw_count = 0
    eligible_count = 0
    grouped: dict[tuple[int, int], list[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as input_file:
        for raw in csv.DictReader(input_file):
            raw_count += 1
            if raw["is_prefill"].strip().lower() != "true":
                continue
            measured_ms = raw["time_stats.attn_prefill.median"].strip()
            if not measured_ms:
                continue
            expected = {
                "n_embd": spec.hidden,
                "n_q_head": spec.q_heads,
                "n_kv_head": spec.kv_heads,
                "num_tensor_parallel_workers": spec.tp,
                "batch_size": 1,
            }
            mismatched = {
                field: (int(raw[field]), value)
                for field, value in expected.items()
                if int(raw[field]) != value
            }
            if mismatched or raw["attention_backend"] != "FLASH_ATTENTION":
                raise ValueError(
                    f"incompatible legacy attention row in {path}: "
                    f"fields={mismatched}, backend="
                    f"{raw['attention_backend']!r}"
                )
            q_tokens = int(raw["prefill_chunk_size"])
            k_tokens = int(raw["kv_cache_size"])
            _bottom_right_causal_pairs(q_tokens, k_tokens)
            eligible_count += 1
            grouped.setdefault((q_tokens, k_tokens), []).append(
                float(measured_ms) * 1e-3
            )
    rows = []
    duplicate_count = 0
    for (q_tokens, k_tokens), measurements in sorted(grouped.items()):
        duplicate_count += len(measurements) - 1
        work = _attention_work(
            q_tokens=q_tokens,
            k_tokens=k_tokens,
            causal_pairs=_bottom_right_causal_pairs(q_tokens, k_tokens),
            q_heads_per_rank=spec.q_heads // spec.tp,
            kv_heads_per_rank=spec.kv_heads // spec.tp,
            head_dim=spec.head_dim,
        )
        rows.append(
            _FitRow(
                family="prefill_attention",
                source_model=spec.name,
                tokens=q_tokens,
                measured_seconds=statistics.median(measurements),
                work=work,
            )
        )
    return rows, raw_count, duplicate_count, {
        "raw_rows": raw_count,
        "eligible_prefill_rows": eligible_count,
        "unique_prefill_points": len(rows),
        "fit_training_points": sum(
            1 for row in rows if 512 <= row.tokens <= 768
        ),
        "fit_holdout_points": sum(1 for row in rows if row.tokens >= 800),
    }


def _aggregate_validation(fits: Mapping[str, KernelFit]) -> dict[str, Any]:
    total_rows = sum(kernel_fit.holdout_rows for kernel_fit in fits.values())
    if total_rows == 0:
        raise ValueError("calibration has no holdout rows")
    weighted_mape = sum(
        kernel_fit.holdout_mape * kernel_fit.holdout_rows
        for kernel_fit in fits.values()
    ) / total_rows
    weighted_bias = sum(
        kernel_fit.holdout_signed_mean * kernel_fit.holdout_rows
        for kernel_fit in fits.values()
    ) / total_rows
    return {
        "holdout": {
            "split": (
                "layers: train M=1025..1536, contiguous holdout "
                "M=1537..2048; attention: train q=512..768, "
                "contiguous holdout q=800..1024"
            ),
            "interpretation": (
                "component-family interpolation check only; it is not an "
                "end-to-end Qwen3 or one-million-token accuracy estimate"
            ),
            "attention_boundary": (
                "the attention holdout varies q only and contains no "
                "long-K cached-suffix validation"
            ),
            "rows": total_rows,
            "weighted_mape": weighted_mape,
            "weighted_signed_mean": weighted_bias,
            "per_family": {
                family: {
                    "rows": kernel_fit.holdout_rows,
                    "mape": kernel_fit.holdout_mape,
                    "p90_ape": kernel_fit.holdout_p90_ape,
                    "signed_mean": kernel_fit.holdout_signed_mean,
                }
                for family, kernel_fit in sorted(fits.items())
            },
        }
    }


def fit_h100_tp4_calibration(repo_root: Path) -> H100TP4Calibration:
    """Fit robust per-family residuals from the four accepted TP4 sources."""

    paths = {relative: repo_root / relative for relative in CALIBRATION_SOURCE_PATHS}
    producer_paths = {
        relative: repo_root / relative
        for relative in LEGACY_PRODUCER_SOURCE_PATHS
    }
    missing = [
        relative
        for relative, path in (*paths.items(), *producer_paths.items())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing legacy H100 calibration source(s): " + ", ".join(missing)
        )

    all_rows: list[_FitRow] = []
    source_rows: dict[str, int] = {}
    source_row_accounting: dict[str, Mapping[str, int]] = {}
    duplicate_attention_points = 0
    for relative, spec in (
        (LLAMA70_LAYERS, LLAMA70_SPEC),
        (MIXTRAL_LAYERS, MIXTRAL_SPEC),
    ):
        rows, raw_count, accounting = _load_layer_rows(paths[relative], spec)
        all_rows.extend(rows)
        source_rows[relative] = raw_count
        source_row_accounting[relative] = accounting
    for relative, spec in (
        (LLAMA70_ATTENTION, LLAMA70_SPEC),
        (MIXTRAL_ATTENTION, MIXTRAL_SPEC),
    ):
        rows, raw_count, duplicates, accounting = _load_attention_rows(
            paths[relative], spec
        )
        all_rows.extend(rows)
        source_rows[relative] = raw_count
        source_row_accounting[relative] = accounting
        duplicate_attention_points += duplicates

    by_family: dict[str, list[_FitRow]] = {}
    for row in all_rows:
        by_family.setdefault(row.family, []).append(row)

    fits: dict[str, KernelFit] = {}
    for family, rows in sorted(by_family.items()):
        if family == "prefill_attention":
            fits[family] = _fit_rows(
                family,
                rows,
                training_predicate=lambda row: 512 <= row.tokens <= 768,
                holdout_predicate=lambda row: row.tokens >= 800,
                occupancy_contract=(
                    "u uses ceil(q/128) * local_q_heads thread-block proxy "
                    "and 132 H100 SMs"
                ),
            )
        else:
            fits[family] = _fit_rows(
                family,
                rows,
                training_predicate=lambda row: 1_025 <= row.tokens <= 1_536,
                holdout_predicate=lambda row: row.tokens >= 1_537,
                occupancy_contract=(
                    "legacy trace lacks selected CUDA tile/block geometry; "
                    "occupancy is folded into eta"
                ),
            )

    required = {
        "embedding",
        "norm",
        "q_projection",
        "o_projection",
        "rope",
        "router",
        "expert_up",
        "expert_down",
        "expert_activation",
        "lm_head",
        "prefill_attention",
    }
    missing_families = sorted(required - set(fits))
    if missing_families:
        raise ValueError(
            "legacy H100 calibration lacks required kernel families: "
            + ", ".join(missing_families)
        )
    return H100TP4Calibration(
        fits=fits,
        source_sha256={
            relative: _sha256(path) for relative, path in paths.items()
        },
        producer_source_sha256={
            relative: _sha256(path)
            for relative, path in producer_paths.items()
        },
        source_rows=source_rows,
        source_row_accounting=source_row_accounting,
        duplicate_attention_points_coalesced=duplicate_attention_points,
        validation=_aggregate_validation(fits),
    )


class H100KernelCalibratedPromptModel:
    """Kernel-decomposed Qwen3 TP4/EP4 analytical prompt predictor."""

    def __init__(
        self,
        *,
        calibration: H100TP4Calibration,
        band: str = "central",
        attention_multiplier: float = 1.0,
        prefill_chunk_size: int = 131_072,
        target_config_sha256: str | None = None,
    ) -> None:
        if band not in {"fast", "central", "slow"}:
            raise ValueError(f"unsupported calibration band: {band}")
        if not math.isfinite(attention_multiplier) or attention_multiplier <= 0:
            raise ValueError("attention_multiplier must be finite and positive")
        if prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if target_config_sha256 is not None and (
            len(target_config_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in target_config_sha256
            )
        ):
            raise ValueError(
                "target_config_sha256 must be a lowercase SHA-256 digest"
            )
        self.calibration = calibration
        self.band = band
        self.attention_multiplier = attention_multiplier
        self.prefill_chunk_size = prefill_chunk_size
        self.target_config_sha256 = target_config_sha256
        self._latency_cache: OrderedDict[tuple[int, int], float] = OrderedDict()

    def _predict(self, family: str, work: KernelWork) -> float:
        kernel_fit = self.calibration.fits[family]
        return max(
            kernel_fit.launch_floor_seconds,
            work.roofline_seconds() * kernel_fit.eta(self.band),
        )

    @staticmethod
    def _tp_allreduce_seconds(payload_bytes: float) -> float:
        return (
            TP_COLLECTIVE_FIXED_SECONDS
            + 2.0
            * (QWEN_TP - 1)
            / QWEN_TP
            * payload_bytes
            / TP_COLLECTIVE_BYTES_PER_SECOND
        )

    @staticmethod
    def _ep_allgather_seconds(local_chunk_bytes: float) -> float:
        return (
            TP_COLLECTIVE_FIXED_SECONDS
            + (QWEN_EP - 1)
            * local_chunk_bytes
            / TP_COLLECTIVE_BYTES_PER_SECOND
        )

    @staticmethod
    def _ep_reduce_scatter_seconds(total_buffer_bytes: float) -> float:
        return (
            TP_COLLECTIVE_FIXED_SECONDS
            + (QWEN_EP - 1)
            / QWEN_EP
            * total_buffer_bytes
            / TP_COLLECTIVE_BYTES_PER_SECOND
        )

    def _expert_seconds(self, q_tokens: int) -> tuple[float, float]:
        # Balanced EP4 conserves global expert-token pairs exactly.
        expert_token_pairs = q_tokens * QWEN_TOP_K / QWEN_EP
        active_local_experts = min(
            QWEN_EXPERTS // QWEN_EP,
            max(1, math.ceil(expert_token_pairs)),
        )
        up_work = KernelWork(
            flops=(
                4.0
                * expert_token_pairs
                * QWEN_HIDDEN_SIZE
                * QWEN_EXPERT_INTERMEDIATE
            ),
            bytes=BF16_BYTES
            * (
                expert_token_pairs * QWEN_HIDDEN_SIZE
                + 2
                * active_local_experts
                * QWEN_HIDDEN_SIZE
                * QWEN_EXPERT_INTERMEDIATE
                + 2 * expert_token_pairs * QWEN_EXPERT_INTERMEDIATE
            ),
        )
        activation_work = KernelWork(
            flops=8.0 * expert_token_pairs * QWEN_EXPERT_INTERMEDIATE,
            bytes=(
                BF16_BYTES
                * 3.0
                * expert_token_pairs
                * QWEN_EXPERT_INTERMEDIATE
            ),
        )
        down_work = KernelWork(
            flops=(
                2.0
                * expert_token_pairs
                * QWEN_EXPERT_INTERMEDIATE
                * QWEN_HIDDEN_SIZE
            ),
            bytes=BF16_BYTES
            * (
                expert_token_pairs * QWEN_EXPERT_INTERMEDIATE
                + active_local_experts
                * QWEN_EXPERT_INTERMEDIATE
                * QWEN_HIDDEN_SIZE
                + expert_token_pairs * QWEN_HIDDEN_SIZE
            ),
        )
        seconds = (
            self._predict("expert_up", up_work)
            + self._predict("expert_activation", activation_work)
            + self._predict("expert_down", down_work)
        )
        return seconds, expert_token_pairs

    def _chunk_breakdown(self, q_tokens: int, prefix_tokens: int) -> dict[str, float]:
        if q_tokens <= 0 or prefix_tokens < 0:
            raise ValueError("chunk requires q > 0 and prefix >= 0")
        q_dim_per_rank = QWEN_Q_HEADS * QWEN_HEAD_DIM // QWEN_TP
        kv_dim_per_rank = QWEN_KV_HEADS * QWEN_HEAD_DIM // QWEN_TP
        qkv_dim_per_rank = q_dim_per_rank + 2 * kv_dim_per_rank

        embedding = self._predict(
            "embedding",
            _elementwise_work(q_tokens, QWEN_HIDDEN_SIZE, 0.0, 1.0),
        )
        input_norm = self._predict(
            "norm",
            _elementwise_work(q_tokens, QWEN_HIDDEN_SIZE, 8.0, 2.0),
        )
        post_norm = input_norm
        q_norm = self._predict(
            "norm",
            _elementwise_work(q_tokens, q_dim_per_rank, 8.0, 2.0),
        )
        k_norm = self._predict(
            "norm",
            _elementwise_work(q_tokens, kv_dim_per_rank, 8.0, 2.0),
        )
        qkv_projection = self._predict(
            "q_projection",
            _gemm_work(q_tokens, QWEN_HIDDEN_SIZE, qkv_dim_per_rank),
        )
        rope = self._predict(
            "rope",
            _elementwise_work(
                q_tokens,
                q_dim_per_rank + kv_dim_per_rank,
                10.0,
                2.0,
            ),
        )
        causal_pairs = _bottom_right_causal_pairs(
            q_tokens,
            prefix_tokens + q_tokens,
        )
        attention_work = _attention_work(
            q_tokens=q_tokens,
            k_tokens=prefix_tokens + q_tokens,
            causal_pairs=causal_pairs,
            q_heads_per_rank=QWEN_Q_HEADS // QWEN_TP,
            kv_heads_per_rank=QWEN_KV_HEADS // QWEN_TP,
            head_dim=QWEN_HEAD_DIM,
        )
        attention = (
            self._predict("prefill_attention", attention_work)
            * self.attention_multiplier
        )
        o_projection = self._predict(
            "o_projection",
            _gemm_work(q_tokens, q_dim_per_rank, QWEN_HIDDEN_SIZE),
        )
        router = self._predict(
            "router",
            _gemm_work(q_tokens, QWEN_HIDDEN_SIZE, QWEN_EXPERTS),
        )
        experts, _ = self._expert_seconds(q_tokens)
        tp_allreduce = self._tp_allreduce_seconds(
            q_tokens * QWEN_HIDDEN_SIZE * BF16_BYTES
        )
        # Use the online trace generator's integer local-token contract for
        # the AG/RS backend. The latency remains an analytical sensitivity,
        # not an ASTRA-Sim collective or contention prediction.
        dispatch_tokens_per_rank = max(1, q_tokens // QWEN_EP)
        dispatch_local_chunk_bytes = (
            dispatch_tokens_per_rank
            * (QWEN_HIDDEN_SIZE + QWEN_EXPERTS)
            * BF16_BYTES
        )
        combine_total_buffer_bytes = (
            q_tokens * QWEN_HIDDEN_SIZE * BF16_BYTES
        )
        ep_allgather = self._ep_allgather_seconds(
            dispatch_local_chunk_bytes
        )
        ep_reduce_scatter = self._ep_reduce_scatter_seconds(
            combine_total_buffer_bytes
        )

        per_layer = {
            "normalization": input_norm + post_norm + q_norm + k_norm,
            "qkv_projection": qkv_projection,
            "rope": rope,
            "attention": attention,
            "o_projection": o_projection,
            "router": router,
            "experts": experts,
            "tp_allreduce": tp_allreduce,
            "ep_allgather": ep_allgather,
            "ep_reduce_scatter": ep_reduce_scatter,
        }
        result = {"embedding": embedding}
        result.update(
            {
                name: seconds * QWEN_LAYERS
                for name, seconds in per_layer.items()
            }
        )
        result["final_norm"] = self._predict(
            "norm",
            _elementwise_work(q_tokens, QWEN_HIDDEN_SIZE, 8.0, 2.0),
        )
        return result

    def _calculate(
        self, total_tokens: int, cached_tokens: int
    ) -> tuple[float, dict[str, float]]:
        if total_tokens < 0 or cached_tokens < 0 or cached_tokens > total_tokens:
            raise ValueError(
                "prompt estimate requires 0 <= cached_tokens <= total_tokens"
            )
        new_tokens = total_tokens - cached_tokens
        if new_tokens == 0:
            return 0.0, {"total": 0.0}

        totals: dict[str, float] = {}
        processed = 0
        while processed < new_tokens:
            q_tokens = min(
                self.prefill_chunk_size, new_tokens - processed
            )
            prefix_tokens = cached_tokens + processed
            chunk = self._chunk_breakdown(q_tokens, prefix_tokens)
            for name, seconds in chunk.items():
                totals[name] = totals.get(name, 0.0) + seconds
            processed += q_tokens

        # vLLM selects one final token per sequence for logits. It does not
        # apply the vocabulary projection to every prompt token.
        totals["lm_head"] = self._predict(
            "lm_head",
            _gemm_work(
                1,
                QWEN_HIDDEN_SIZE,
                QWEN_VOCAB_SIZE // QWEN_TP,
            ),
        )
        total_seconds = sum(totals.values())
        totals["total"] = total_seconds
        return total_seconds, totals

    def _estimate_seconds(self, total_tokens: int, cached_tokens: int) -> float:
        key = (total_tokens, cached_tokens)
        cached = self._latency_cache.get(key)
        if cached is not None:
            self._latency_cache.move_to_end(key)
            return cached
        total_seconds, _ = self._calculate(total_tokens, cached_tokens)
        self._latency_cache[key] = total_seconds
        if len(self._latency_cache) > LATENCY_CACHE_MAX_ENTRIES:
            self._latency_cache.popitem(last=False)
        return total_seconds

    def recompute_seconds(self, tokens: int) -> float:
        return self._estimate_seconds(tokens, 0)

    def cached_prefill_seconds(
        self, total_tokens: int, cached_tokens: int
    ) -> float:
        return self._estimate_seconds(total_tokens, cached_tokens)

    def estimate_breakdown(
        self, total_tokens: int, cached_tokens: int = 0
    ) -> Mapping[str, float]:
        return dict(self._calculate(total_tokens, cached_tokens)[1])

    def metadata(self) -> Mapping[str, Any]:
        calibration_metadata = self.calibration.metadata()
        return {
            "schema_version": 4,
            "model_kind": "h100_kernel_calibrated_prompt",
            "calibrated_from_measurements": True,
            "legacy_h100_labeled_component_measurements_used": True,
            "absolute_dgx_h100_calibration_validated": False,
            "legacy_h100_fp16_component_calibration": True,
            "target_qwen_shape_extrapolation": True,
            "long_context_experiment": (
                qwen_long_context_experiment_contract()
            ),
            "description": (
                "Kernel-decomposed Qwen3 TP4/EP4 prompt prediction fitted "
                "to legacy TP4 component artifacts labeled H100. Exact "
                "measurement-system identity is unavailable."
            ),
            "band": self.band,
            "attention_multiplier": self.attention_multiplier,
            "prefill_chunk_size": self.prefill_chunk_size,
            "source_sha256": calibration_metadata["source_sha256"],
            "producer_source_sha256": calibration_metadata[
                "producer_source_sha256"
            ],
            "source_identity": calibration_metadata["source_identity"],
            "source_rows": calibration_metadata["source_rows"],
            "source_row_accounting": calibration_metadata[
                "source_row_accounting"
            ],
            "analytical_model": {
                "reference": "KernelSight-LM, arXiv:2606.28565",
                "equation": (
                    "t=max(t0,t_roof*eta); "
                    "t_roof=max((F/P_peak)*u,bytes/BW_peak)"
                ),
                "eta_fit": (
                    "robust p50 of a contiguous upper-shape training tail; "
                    "fast=p10 and slow=p90"
                ),
                "gemm_occupancy": (
                    "folded into eta because legacy traces lack selected "
                    "cuBLAS tile and thread-block geometry"
                ),
                "attention_occupancy": (
                    "ceil(q/128)*local_q_heads partial-wave proxy"
                ),
                "source_attention_causal_pairs": (
                    "bottom-right aligned: q*(k-q)+q*(q+1)/2"
                ),
                "bands": (
                    "p10/p50/p90 fitted eta sensitivities, not confidence "
                    "intervals"
                ),
            },
            "hardware": {
                "dense_bf16_tflops": (
                    H100_DENSE_BF16_FLOPS_PER_SECOND / 1e12
                ),
                "hbm_tb_per_second": H100_HBM_BYTES_PER_SECOND / 1e12,
                "sms": H100_SMS,
                "collective_gb_per_second": (
                    TP_COLLECTIVE_BYTES_PER_SECOND / 1e9
                ),
                "collective_fixed_microseconds": (
                    TP_COLLECTIVE_FIXED_SECONDS * 1e6
                ),
            },
            "collective_model": {
                "backend": "allgather_reducescatter",
                "measured_or_fitted": False,
                "contention_modeled": False,
                "bandwidth_and_latency_contract": (
                    "analytical intra-node sensitivity using 450 GB/s and "
                    "3 us per collective"
                ),
                "ep_dispatch": (
                    "ring AllGather of max(1,floor(q/EP)) per-rank "
                    "(hidden+router_logits) tokens"
                ),
                "ep_combine": (
                    "ring ReduceScatter of the full hidden buffer"
                ),
            },
            "target_geometry": {
                "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "config_sha256": self.target_config_sha256,
                "tp": QWEN_TP,
                "ep": QWEN_EP,
                "layers": QWEN_LAYERS,
                "hidden_size": QWEN_HIDDEN_SIZE,
                "head_dim": QWEN_HEAD_DIM,
                "q_heads_per_rank": QWEN_Q_HEADS // QWEN_TP,
                "kv_heads_per_rank": QWEN_KV_HEADS // QWEN_TP,
                "fused_qkv_output_per_rank": 1_280,
                "experts": QWEN_EXPERTS,
                "experts_per_rank": QWEN_EXPERTS // QWEN_EP,
                "top_k": QWEN_TOP_K,
                "balanced_expert_pairs_per_rank_per_token": (
                    QWEN_TOP_K / QWEN_EP
                ),
            },
            "validation": calibration_metadata["validation"],
            "kernel_fits": calibration_metadata["kernel_fits"],
            "limitations": {
                "measured_qwen3_h100": False,
                "exact_legacy_h100_sku_known": False,
                "legacy_measurement_software_versions_known": False,
                "legacy_measurement_command_known": False,
                "current_producer_revision_is_measurement_revision": False,
                "absolute_dgx_h100_accuracy_validated": False,
                "absolute_1m_latency_sensitivity_only": True,
                "primary_claim_scope": (
                    "matched-workload relative cache-policy comparison "
                    "under one fixed calibration band"
                ),
                "measured_dca_or_minference": False,
                "one_million_token_attention_is_extrapolated": True,
                "source_prefill_q_max": 1_024,
                "source_prefill_kv_max": 2_016,
                "target_prefill_q_max": self.prefill_chunk_size,
                "target_context_max": 1_010_000,
                "source_dense_m_max": 2_048,
                "target_dense_m_max": self.prefill_chunk_size,
                "source_measurement_dtype": "float16",
                "target_model_dtype": "bfloat16",
                "fp16_to_bf16_efficiency_transfer_assumed": True,
                "attention_holdout_validates_long_k": False,
                "aggregate_holdout_is_target_end_to_end_accuracy": False,
                "fused_qkv_measured_as_target_shape": False,
                "qk_norm_head_width_measured": False,
                "router_e128_top8_measured": False,
                "moe_grouped_fusion_and_imbalance_measured": False,
                "collective_terms_measured_or_fitted": False,
                "decode_compute_modeled": False,
                "host_scheduler_and_continuous_batching_modeled": False,
                "sampler_and_serving_host_overhead_modeled": False,
                "attention_multiplier_scope": "prefill attention only",
                "latency_cache_max_scalar_entries": (
                    LATENCY_CACHE_MAX_ENTRIES
                ),
            },
        }


__all__ = [
    "CALIBRATION_SOURCE_PATHS",
    "H100KernelCalibratedPromptModel",
    "H100TP4Calibration",
    "KernelFit",
    "KernelWork",
    "LATENCY_CACHE_MAX_ENTRIES",
    "LEGACY_PRODUCER_SOURCE_PATHS",
    "_bottom_right_causal_pairs",
    "fit_h100_tp4_calibration",
]

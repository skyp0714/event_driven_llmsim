"""Online analytical latency providers for the ASTRA-Sim trace path.

The providers in this module supply only GPU ``COMP`` node durations.  They
must never add TP/EP/P-D communication time: the trace generator continues to
emit those collectives and ASTRA-Sim remains their sole timing authority.

The H100/Qwen3 provider is deliberately narrow.  It fits kernel-family
roofline residuals to the repository's legacy H100-labelled TP4 measurements,
then evaluates the exact Qwen3-30B-A3B TP4/EP4 shapes used by the cold-session
experiments.  It is an analytical extrapolation, not a measured Qwen3 result.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .h100_kernel_calibrated_prompt import (
    BF16_BYTES,
    H100TP4Calibration,
    H100_HBM_BYTES_PER_SECOND,
    H100_SMS,
    LLAMA70_ATTENTION,
    LLAMA70_SPEC,
    MIXTRAL_ATTENTION,
    MIXTRAL_SPEC,
    QWEN_EP,
    QWEN_EXPERTS,
    QWEN_EXPERT_INTERMEDIATE,
    QWEN_HEAD_DIM,
    QWEN_HIDDEN_SIZE,
    QWEN_KV_HEADS,
    QWEN_LAYERS,
    QWEN_LONG_CONTEXT_MODE,
    QWEN_NATIVE_MAX_POSITION_EMBEDDINGS,
    QWEN_VALIDATED_RUNTIME_MAX_MODEL_LEN,
    QWEN_Q_HEADS,
    QWEN_TOP_K,
    QWEN_TP,
    QWEN_VOCAB_SIZE,
    KernelWork,
    _elementwise_work,
    _gemm_work,
    fit_h100_tp4_calibration,
    qwen_long_context_experiment_contract,
)


H100_QWEN3_TP4_KERNEL_CALIBRATED = (
    "h100-qwen3-tp4-kernel-calibrated"
)
SUPPORTED_ONLINE_LATENCY_MODELS = (
    H100_QWEN3_TP4_KERNEL_CALIBRATED,
)
TARGET_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
TARGET_MAX_CONTEXT = QWEN_VALIDATED_RUNTIME_MAX_MODEL_LEN
TARGET_PREFILL_CHUNK = 131_072
DECODE_SKEW_ALPHA = 0.093


class OnlineLatencyModelError(ValueError):
    """Raised when an online analytical latency contract is violated."""


@dataclass(frozen=True)
class OnlineBatchKernelPhases:
    """Exact serial compute decomposition around the transformer layers."""

    prologue_ns: int
    layer_ns: int
    layer_count: int
    epilogue_ns: int

    @property
    def total_ns(self) -> int:
        return (
            self.prologue_ns
            + self.layer_count * self.layer_ns
            + self.epilogue_ns
        )


def validate_runtime_context_contract(
        *, config: Mapping[str, Any], max_model_len: int,
        model: str | None = None) -> Mapping[str, Any] | None:
    """Validate an explicit model-side contract above the native window.

    A larger KV allocation is not a context-extension mechanism.  Any runtime
    length above ``max_position_embeddings`` therefore requires a declared
    ``long_context_experiment`` mode with a finite validated runtime ceiling.
    The Qwen paper configuration is additionally pinned byte-for-byte to the
    provider's official-source and dense-sensitivity contract.
    """

    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int):
        raise OnlineLatencyModelError(
            f"max_model_len must be a positive integer, got "
            f"{max_model_len!r}"
        )
    if max_model_len <= 0:
        raise OnlineLatencyModelError(
            f"max_model_len must be positive, got {max_model_len}"
        )
    native = config.get("max_position_embeddings")
    if isinstance(native, bool) or not isinstance(native, int) or native <= 0:
        raise OnlineLatencyModelError(
            "model config must declare a positive integer "
            "max_position_embeddings"
        )
    if model == TARGET_MODEL:
        if native != QWEN_NATIVE_MAX_POSITION_EMBEDDINGS:
            raise OnlineLatencyModelError(
                "Qwen3 native context metadata changed: "
                f"expected {QWEN_NATIVE_MAX_POSITION_EMBEDDINGS}, "
                f"got {native}"
            )
        if max_model_len > QWEN_VALIDATED_RUNTIME_MAX_MODEL_LEN:
            raise OnlineLatencyModelError(
                f"max_model_len={max_model_len} exceeds the declared "
                "long-context runtime ceiling "
                f"{QWEN_VALIDATED_RUNTIME_MAX_MODEL_LEN} for mode "
                f"{QWEN_LONG_CONTEXT_MODE!r}"
            )
    if max_model_len <= native:
        return None

    contract = config.get("long_context_experiment")
    if not isinstance(contract, Mapping):
        raise OnlineLatencyModelError(
            f"max_model_len={max_model_len} exceeds native "
            f"max_position_embeddings={native} for {model or 'model'}; "
            "an explicit long_context_experiment mode is required"
        )
    mode = contract.get("mode")
    if not isinstance(mode, str) or not mode.strip():
        raise OnlineLatencyModelError(
            "long_context_experiment.mode must be an explicit non-empty "
            "string"
        )
    declared_native = contract.get("native_max_position_embeddings")
    if declared_native != native:
        raise OnlineLatencyModelError(
            "long_context_experiment.native_max_position_embeddings "
            f"must equal model max_position_embeddings={native}, got "
            f"{declared_native!r}"
        )
    ceiling = contract.get("validated_runtime_max_model_len")
    if isinstance(ceiling, bool) or not isinstance(ceiling, int):
        raise OnlineLatencyModelError(
            "long_context_experiment.validated_runtime_max_model_len must "
            "be a positive integer"
        )
    if ceiling <= native:
        raise OnlineLatencyModelError(
            "long-context runtime ceiling must exceed the native model "
            "window"
        )
    if max_model_len > ceiling:
        raise OnlineLatencyModelError(
            f"max_model_len={max_model_len} exceeds the declared "
            f"long-context runtime ceiling {ceiling} for mode {mode!r}"
        )

    if model == TARGET_MODEL:
        expected = qwen_long_context_experiment_contract()
        if dict(contract) != expected:
            raise OnlineLatencyModelError(
                "Qwen3 long_context_experiment metadata does not match "
                f"the pinned {QWEN_LONG_CONTEXT_MODE!r} provider contract"
            )
    return contract


@dataclass(frozen=True)
class OnlineBatchShape:
    """Exact token geometry consumed by one online scheduler iteration."""

    total_tokens: int
    prefill_q: tuple[int, ...]
    prefill_k: tuple[int, ...]
    decode_k: tuple[int, ...]
    lm_head_sequences: int
    recompute_q: tuple[int, ...] = ()

    @property
    def real_query_tokens(self) -> int:
        return sum(self.prefill_q) + len(self.decode_k)

    @property
    def padding_tokens(self) -> int:
        return self.total_tokens - self.real_query_tokens

    @property
    def num_decode(self) -> int:
        return len(self.decode_k)

    def without_recompute_queries(self) -> "OnlineBatchShape":
        """Return the cache-hit counterfactual for recompute attribution.

        Recomputed query tokens become an already-resident prefix.  Fresh
        suffix queries still attend to them, which is why the removed amount
        is added to ``prefill_k`` rather than simply disappearing.
        """

        if not self.recompute_q:
            return self
        if len(self.recompute_q) != len(self.prefill_q):
            raise OnlineLatencyModelError(
                "recompute_q must align one-to-one with prefill_q"
            )
        q_out = []
        k_out = []
        removed = 0
        for q_tokens, k_tokens, recompute_tokens in zip(
                self.prefill_q, self.prefill_k, self.recompute_q):
            if recompute_tokens < 0 or recompute_tokens > q_tokens:
                raise OnlineLatencyModelError(
                    "each recompute_q value must be in [0, prefill_q]"
                )
            fresh = q_tokens - recompute_tokens
            removed += recompute_tokens
            if fresh:
                q_out.append(fresh)
                k_out.append(k_tokens + recompute_tokens)
        return OnlineBatchShape(
            total_tokens=self.total_tokens - removed,
            prefill_q=tuple(q_out),
            prefill_k=tuple(k_out),
            decode_k=self.decode_k,
            lm_head_sequences=(
                self.lm_head_sequences
                - (len(self.prefill_q) - len(q_out))
            ),
            recompute_q=tuple(0 for _ in q_out),
        )


@dataclass(frozen=True)
class DecodeAttentionFit:
    batch_min: int
    batch_max: int
    eta_fast: float
    eta_central: float
    eta_slow: float
    launch_floor_seconds: float
    training_rows: int
    holdout_rows: int
    holdout_mape: float
    holdout_p90_ape: float
    source_k_max: int
    source_k_per_sequence_max: float

    def eta(self, band: str) -> float:
        if band == "fast":
            return self.eta_fast
        if band == "central":
            return self.eta_central
        if band == "slow":
            return self.eta_slow
        raise OnlineLatencyModelError(
            f"unsupported calibration band: {band!r}"
        )


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise OnlineLatencyModelError("cannot fit an empty measurement set")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _attention_work_for_sequences(
        q_lengths: Sequence[int], k_lengths: Sequence[int], *,
        q_heads_per_rank: int, kv_heads_per_rank: int,
        head_dim: int) -> KernelWork:
    if len(q_lengths) != len(k_lengths) or not q_lengths:
        raise OnlineLatencyModelError(
            "attention requires equally sized, non-empty Q/K length lists"
        )
    causal_pairs = 0.0
    q_total = 0
    k_total = 0
    blocks = 0
    for q_tokens, k_tokens in zip(q_lengths, k_lengths):
        if q_tokens <= 0 or k_tokens < q_tokens:
            raise OnlineLatencyModelError(
                "attention requires k_tokens >= q_tokens > 0 per sequence"
            )
        prefix = k_tokens - q_tokens
        causal_pairs += (
            q_tokens * prefix + q_tokens * (q_tokens + 1) / 2
        )
        q_total += q_tokens
        k_total += k_tokens
        blocks += math.ceil(q_tokens / 128) * q_heads_per_rank
    wave_penalty = math.ceil(blocks / H100_SMS) * H100_SMS / blocks
    return KernelWork(
        flops=(
            causal_pairs
            * q_heads_per_rank
            * (4.0 * head_dim + 5.0)
        ),
        bytes=BF16_BYTES * (
            2 * q_total * q_heads_per_rank * head_dim
            + 2 * k_total * kv_heads_per_rank * head_dim
        ),
        wave_penalty=wave_penalty,
    )


def _decode_work(
        batch_size: int, total_k_tokens: int, *,
        q_heads_per_rank: int, kv_heads_per_rank: int,
        head_dim: int) -> KernelWork:
    if batch_size <= 0 or total_k_tokens < batch_size:
        raise OnlineLatencyModelError(
            "decode attention requires total K >= batch size > 0"
        )
    blocks = batch_size * q_heads_per_rank
    wave_penalty = math.ceil(blocks / H100_SMS) * H100_SMS / blocks
    return KernelWork(
        flops=(
            total_k_tokens
            * q_heads_per_rank
            * (4.0 * head_dim + 5.0)
        ),
        bytes=BF16_BYTES * (
            2 * batch_size * q_heads_per_rank * head_dim
            + 2 * total_k_tokens * kv_heads_per_rank * head_dim
        ),
        wave_penalty=wave_penalty,
    )


def _fit_decode_attention(
        repo_root: Path) -> tuple[DecodeAttentionFit, ...]:
    """Fit batch-regime decode residuals with a long-K holdout.

    The legacy decode sweep records total K tokens across a uniform batch.
    We coalesce duplicate points and fit dimensionless roofline residuals in
    per measured batch size.  The largest 20% of per-sequence K values at
    every batch size are a contiguous holdout, so the reported error tests
    extrapolation in the direction used by the long-context experiment.
    """

    grouped: dict[tuple[int, int, int, int, int], list[float]] = {}
    for relative, spec in (
            (LLAMA70_ATTENTION, LLAMA70_SPEC),
            (MIXTRAL_ATTENTION, MIXTRAL_SPEC)):
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"missing legacy decode calibration source: {path}"
            )
        with path.open("r", encoding="utf-8", newline="") as input_file:
            for raw in csv.DictReader(input_file):
                if raw["is_prefill"].strip().lower() != "false":
                    continue
                if raw["attention_backend"] != "FLASH_ATTENTION":
                    raise OnlineLatencyModelError(
                        f"unsupported legacy attention backend in {path}"
                    )
                if int(raw["num_tensor_parallel_workers"]) != 4:
                    raise OnlineLatencyModelError(
                        f"non-TP4 decode row in {path}"
                    )
                batch = int(raw["batch_size"])
                total_k = int(raw["kv_cache_size"])
                # The legacy sampler admitted a small set of combinations
                # with total_k < batch. Its tensor builder integer-divided
                # those into zero-length K sequences, which are not valid
                # serving decode shapes and cannot calibrate this model.
                if total_k < batch:
                    continue
                measured_ms = raw[
                    "time_stats.attn_decode.median"
                ].strip()
                if not measured_ms:
                    raise OnlineLatencyModelError(
                        f"decode row without median in {path}"
                    )
                key = (
                    spec.q_heads // spec.tp,
                    spec.kv_heads // spec.tp,
                    spec.head_dim,
                    batch,
                    total_k,
                )
                grouped.setdefault(key, []).append(
                    float(measured_ms) * 1e-3
                )

    rows = []
    for (q_heads, kv_heads, head_dim, batch, total_k), samples in grouped.items():
        work = _decode_work(
            batch, total_k,
            q_heads_per_rank=q_heads,
            kv_heads_per_rank=kv_heads,
            head_dim=head_dim,
        )
        measured = statistics.median(samples)
        rows.append((
            batch, total_k, total_k / batch,
            measured, work.roofline_seconds(),
        ))

    fits = []
    for batch_anchor in sorted({row[0] for row in rows if row[0] <= 128}):
        regime = [row for row in rows if row[0] == batch_anchor]
        if not regime:
            raise OnlineLatencyModelError(
                f"decode calibration has no rows for batch {batch_anchor}"
            )
        k_values = sorted({row[2] for row in regime})
        holdout_boundary = k_values[max(1, math.floor(len(k_values) * 0.8))]
        training = [row for row in regime if row[2] < holdout_boundary]
        holdout = [row for row in regime if row[2] >= holdout_boundary]
        if not training or not holdout:
            raise OnlineLatencyModelError(
                f"decode calibration cannot split batch {batch_anchor}"
            )
        roof_values = [row[4] for row in training]
        measured_values = [row[3] for row in training]
        roof_mean = sum(roof_values) / len(roof_values)
        measured_mean = sum(measured_values) / len(measured_values)
        denominator = sum(
            (roof - roof_mean) ** 2 for roof in roof_values)
        if denominator <= 0:
            raise OnlineLatencyModelError(
                f"decode calibration has degenerate roofline at batch "
                f"{batch_anchor}"
            )
        central = max(0.0, sum(
            (roof - roof_mean) * (measured - measured_mean)
            for roof, measured in zip(roof_values, measured_values)
        ) / denominator)
        launch_floor = max(
            0.0, measured_mean - central * roof_mean)
        slope_samples = [
            max(0.0, (measured - launch_floor) / max(roof, 1e-15))
            for measured, roof in zip(measured_values, roof_values)
        ]
        eta_fast = min(central, _quantile(slope_samples, 0.10))
        eta_slow = max(central, _quantile(slope_samples, 0.90))
        errors = [
            abs((launch_floor + roof * central - measured) / measured)
            for _, _, _, measured, roof in holdout
        ]
        fits.append(DecodeAttentionFit(
            batch_min=batch_anchor,
            batch_max=batch_anchor,
            eta_fast=eta_fast,
            eta_central=central,
            eta_slow=eta_slow,
            launch_floor_seconds=launch_floor,
            training_rows=len(training),
            holdout_rows=len(holdout),
            holdout_mape=sum(errors) / len(errors),
            holdout_p90_ape=_quantile(errors, 0.90),
            source_k_max=max(row[1] for row in regime),
            source_k_per_sequence_max=max(k_values),
        ))
    return tuple(fits)


class H100Qwen3OnlineLatencyModel:
    """Kernel-decomposed online compute model for Qwen3 TP4/EP4."""

    def __init__(
            self, *, calibration: H100TP4Calibration,
            decode_attention_fits: Sequence[DecodeAttentionFit],
            band: str = "central", target_config_sha256: str) -> None:
        if band not in {"fast", "central", "slow"}:
            raise OnlineLatencyModelError(
                f"unsupported calibration band: {band!r}"
            )
        if len(target_config_sha256) != 64:
            raise OnlineLatencyModelError(
                "target_config_sha256 must be a SHA-256 digest"
            )
        self.calibration = calibration
        self.decode_attention_fits = tuple(decode_attention_fits)
        self.band = band
        self.target_config_sha256 = target_config_sha256

    @staticmethod
    def _ns(seconds: float) -> int:
        if not math.isfinite(seconds) or seconds <= 0:
            raise OnlineLatencyModelError(
                f"analytical kernel produced invalid latency {seconds!r}"
            )
        latency = int(round(seconds * 1e9))
        if latency <= 0 or latency > 3_600 * 1_000_000_000:
            raise OnlineLatencyModelError(
                f"analytical kernel latency out of sanity bounds: {latency} ns"
            )
        return latency

    def _predict_seconds(self, family: str, work: KernelWork) -> float:
        try:
            kernel_fit = self.calibration.fits[family]
        except KeyError as exc:
            raise OnlineLatencyModelError(
                f"missing fitted H100 kernel family {family!r}"
            ) from exc
        return max(
            kernel_fit.launch_floor_seconds,
            work.roofline_seconds() * kernel_fit.eta(self.band),
        )

    def _decode_fit(self, batch_size: int) -> DecodeAttentionFit:
        for fit in self.decode_attention_fits:
            if fit.batch_min == batch_size:
                return fit
        anchors = [fit.batch_min for fit in self.decode_attention_fits]
        index = bisect.bisect_right(anchors, batch_size)
        if index == 0 or index >= len(anchors):
            raise OnlineLatencyModelError(
                "decode batch size is outside the measured/fitted range "
                f"1..128: {batch_size}"
            )
        lower = self.decode_attention_fits[index - 1]
        upper = self.decode_attention_fits[index]
        fraction = (
            (math.log(batch_size) - math.log(lower.batch_min))
            / (math.log(upper.batch_min) - math.log(lower.batch_min))
        )

        def interpolate(field: str) -> float:
            low = float(getattr(lower, field))
            high = float(getattr(upper, field))
            return low + fraction * (high - low)

        return DecodeAttentionFit(
            batch_min=batch_size,
            batch_max=batch_size,
            eta_fast=interpolate("eta_fast"),
            eta_central=interpolate("eta_central"),
            eta_slow=interpolate("eta_slow"),
            launch_floor_seconds=interpolate(
                "launch_floor_seconds"),
            training_rows=min(lower.training_rows, upper.training_rows),
            holdout_rows=min(lower.holdout_rows, upper.holdout_rows),
            holdout_mape=interpolate("holdout_mape"),
            holdout_p90_ape=interpolate("holdout_p90_ape"),
            source_k_max=int(interpolate("source_k_max")),
            source_k_per_sequence_max=interpolate(
                "source_k_per_sequence_max"),
        )

    def validate_shape(self, shape: OnlineBatchShape) -> None:
        if shape.total_tokens <= 0:
            raise OnlineLatencyModelError("batch total_tokens must be positive")
        if len(shape.prefill_q) != len(shape.prefill_k):
            raise OnlineLatencyModelError(
                "prefill_q and prefill_k must have identical lengths"
            )
        if shape.padding_tokens < 0:
            raise OnlineLatencyModelError(
                "batch total_tokens is smaller than its real query count"
            )
        if shape.lm_head_sequences <= 0:
            raise OnlineLatencyModelError(
                "lm_head_sequences must be positive"
            )
        for q_tokens, prefix_tokens in zip(
                shape.prefill_q, shape.prefill_k):
            if q_tokens <= 0 or prefix_tokens < 0:
                raise OnlineLatencyModelError(
                    "prefill requires q > 0 and prefix >= 0"
                )
            if q_tokens > TARGET_PREFILL_CHUNK:
                raise OnlineLatencyModelError(
                    "prefill query chunk exceeds the online calibration "
                    f"contract: {q_tokens} > {TARGET_PREFILL_CHUNK}. "
                    "A 1M prompt must execute as scheduler-generated "
                    "131,072-token chunks, not one 1M-query kernel."
                )
            if prefix_tokens + q_tokens > TARGET_MAX_CONTEXT:
                raise OnlineLatencyModelError(
                    "prefill exceeds calibrated experiment context contract: "
                    f"{prefix_tokens + q_tokens} > {TARGET_MAX_CONTEXT}"
                )
        for k_tokens in shape.decode_k:
            if k_tokens <= 0 or k_tokens > TARGET_MAX_CONTEXT:
                raise OnlineLatencyModelError(
                    "decode K length outside 1..1,010,000: "
                    f"{k_tokens}"
                )
        if shape.num_decode > 128:
            raise OnlineLatencyModelError(
                "decode batch exceeds validated calibration support (128)"
            )
        if shape.recompute_q and len(shape.recompute_q) != len(shape.prefill_q):
            raise OnlineLatencyModelError(
                "recompute_q must align with prefill_q"
            )

    def dense_latency_ns(
            self, layer_name: str, total_tokens: int,
            lm_head_sequences: int) -> int:
        if total_tokens <= 0 or lm_head_sequences <= 0:
            raise OnlineLatencyModelError(
                "dense latency requires positive token and sequence counts"
            )
        q_dim = QWEN_Q_HEADS * QWEN_HEAD_DIM // QWEN_TP
        kv_dim = QWEN_KV_HEADS * QWEN_HEAD_DIM // QWEN_TP
        qkv_dim = q_dim + 2 * kv_dim
        if layer_name == "embedding":
            work = _elementwise_work(total_tokens, QWEN_HIDDEN_SIZE, 0.0, 1.0)
            family = "embedding"
            seconds = self._predict_seconds(family, work)
        elif layer_name in {"layernorm", "final_layernorm"}:
            work = _elementwise_work(total_tokens, QWEN_HIDDEN_SIZE, 8.0, 2.0)
            seconds = self._predict_seconds("norm", work)
        elif layer_name == "qk_norm":
            q_work = _elementwise_work(total_tokens, q_dim, 8.0, 2.0)
            k_work = _elementwise_work(total_tokens, kv_dim, 8.0, 2.0)
            seconds = (
                self._predict_seconds("norm", q_work)
                + self._predict_seconds("norm", k_work)
            )
        elif layer_name == "qkv_proj":
            work = _gemm_work(total_tokens, QWEN_HIDDEN_SIZE, qkv_dim)
            seconds = self._predict_seconds("q_projection", work)
        elif layer_name == "rotary_emb":
            work = _elementwise_work(
                total_tokens, q_dim + kv_dim, 10.0, 2.0
            )
            seconds = self._predict_seconds("rope", work)
        elif layer_name == "o_proj":
            work = _gemm_work(total_tokens, q_dim, QWEN_HIDDEN_SIZE)
            seconds = self._predict_seconds("o_projection", work)
        elif layer_name == "lm_head":
            work = _gemm_work(
                lm_head_sequences,
                QWEN_HIDDEN_SIZE,
                QWEN_VOCAB_SIZE // QWEN_TP,
            )
            seconds = self._predict_seconds("lm_head", work)
        elif layer_name == "sampler":
            # The legacy bundle has no standalone sampler event.  Keep it
            # explicit and conservative: one fitted launch floor plus a
            # memory pass over each rank's logits.  It is not folded into
            # lm_head, so the Chakra dependency remains visible.
            fit = self.calibration.fits["lm_head"]
            logits_bytes = (
                lm_head_sequences
                * (QWEN_VOCAB_SIZE // QWEN_TP)
                * BF16_BYTES
            )
            seconds = max(
                fit.launch_floor_seconds,
                logits_bytes / H100_HBM_BYTES_PER_SECOND,
            )
        else:
            raise OnlineLatencyModelError(
                f"unsupported calibrated dense layer: {layer_name!r}"
            )
        return self._ns(seconds)

    def attention_latency_ns(self, shape: OnlineBatchShape) -> int:
        self.validate_shape(shape)
        prefill_work = None
        if shape.prefill_q:
            prefill_work = _attention_work_for_sequences(
                shape.prefill_q,
                tuple(k + q for q, k in zip(
                    shape.prefill_q, shape.prefill_k)),
                q_heads_per_rank=QWEN_Q_HEADS // QWEN_TP,
                kv_heads_per_rank=QWEN_KV_HEADS // QWEN_TP,
                head_dim=QWEN_HEAD_DIM,
            )
        decode_work = None
        decode_fit = None
        if shape.decode_k:
            mean_total_k = sum(shape.decode_k)
            padded_total_k = len(shape.decode_k) * max(shape.decode_k)
            effective_total_k = int(round(
                mean_total_k
                + DECODE_SKEW_ALPHA * (padded_total_k - mean_total_k)
            ))
            decode_work = _decode_work(
                len(shape.decode_k), effective_total_k,
                q_heads_per_rank=QWEN_Q_HEADS // QWEN_TP,
                kv_heads_per_rank=QWEN_KV_HEADS // QWEN_TP,
                head_dim=QWEN_HEAD_DIM,
            )
            decode_fit = self._decode_fit(len(shape.decode_k))

        if prefill_work is None and decode_work is None:
            raise OnlineLatencyModelError(
                "attention requested for a batch without real queries"
            )
        if decode_work is None:
            seconds = self._predict_seconds(
                "prefill_attention", prefill_work
            )
        elif prefill_work is None:
            seconds = self._decode_attention_seconds(
                len(shape.decode_k), effective_total_k)
        else:
            prefill_fit = self.calibration.fits["prefill_attention"]
            # One mixed varlen kernel pays one launch floor.  Its analytical
            # work is decomposed by regime because legacy data provides
            # separate prefill and decode residuals but no mixed shot.
            seconds = (
                self._decode_attention_seconds(
                    len(shape.decode_k), effective_total_k)
                + (prefill_work.roofline_seconds()
                 * prefill_fit.eta(self.band))
            )
        return self._ns(seconds)

    @lru_cache(maxsize=131_072)
    def _decode_attention_seconds(
            self, batch_size: int, total_k_tokens: int) -> float:
        """Return a batch-monotone, roofline-bounded decode prediction.

        Separate legacy fits can cross at batch-regime boundaries.  Such a
        crossing made a larger batch with the same per-sequence context
        unrealistically faster in absolute latency.  Evaluate the fitted
        curve for every smaller batch at the same effective per-sequence K
        and take their upper envelope.  This preserves batching speedup in
        per-token throughput while enforcing nondecreasing batch latency.
        """

        if batch_size <= 0 or total_k_tokens < batch_size:
            raise OnlineLatencyModelError(
                "invalid decode shape for monotonic envelope"
            )
        per_sequence_k = total_k_tokens / batch_size
        envelope_seconds = 0.0
        for candidate_batch in range(1, batch_size + 1):
            candidate_total_k = max(
                candidate_batch,
                int(math.ceil(per_sequence_k * candidate_batch)),
            )
            work = _decode_work(
                candidate_batch, candidate_total_k,
                q_heads_per_rank=QWEN_Q_HEADS // QWEN_TP,
                kv_heads_per_rank=QWEN_KV_HEADS // QWEN_TP,
                head_dim=QWEN_HEAD_DIM,
            )
            fit = self._decode_fit(candidate_batch)
            fitted_seconds = (
                fit.launch_floor_seconds
                + work.roofline_seconds() * fit.eta(self.band)
            )
            envelope_seconds = max(
                envelope_seconds,
                work.roofline_seconds(),
                fitted_seconds,
            )
        return envelope_seconds

    def moe_rank_latency_ns(
            self, *, total_tokens: int, local_tokens: int,
            activated_experts: int, ep_total: int) -> int:
        if ep_total != QWEN_EP:
            raise OnlineLatencyModelError(
                f"Qwen online calibration requires EP={QWEN_EP}"
            )
        if total_tokens <= 0 or local_tokens <= 0:
            raise OnlineLatencyModelError(
                "MoE rank latency requires positive global/local tokens"
            )
        experts_per_rank = QWEN_EXPERTS // QWEN_EP
        if not 1 <= activated_experts <= experts_per_rank:
            raise OnlineLatencyModelError(
                "activated experts outside Qwen EP4 rank capacity: "
                f"{activated_experts}"
            )
        expert_pairs = total_tokens * QWEN_TOP_K / QWEN_EP
        up_work = KernelWork(
            flops=(4.0 * expert_pairs * QWEN_HIDDEN_SIZE
                   * QWEN_EXPERT_INTERMEDIATE),
            bytes=BF16_BYTES * (
                expert_pairs * QWEN_HIDDEN_SIZE
                + 2 * activated_experts * QWEN_HIDDEN_SIZE
                * QWEN_EXPERT_INTERMEDIATE
                + 2 * expert_pairs * QWEN_EXPERT_INTERMEDIATE
            ),
        )
        activation_work = KernelWork(
            flops=8.0 * expert_pairs * QWEN_EXPERT_INTERMEDIATE,
            bytes=(BF16_BYTES * 3.0 * expert_pairs
                   * QWEN_EXPERT_INTERMEDIATE),
        )
        down_work = KernelWork(
            flops=(2.0 * expert_pairs * QWEN_EXPERT_INTERMEDIATE
                   * QWEN_HIDDEN_SIZE),
            bytes=BF16_BYTES * (
                expert_pairs * QWEN_EXPERT_INTERMEDIATE
                + activated_experts * QWEN_EXPERT_INTERMEDIATE
                * QWEN_HIDDEN_SIZE
                + expert_pairs * QWEN_HIDDEN_SIZE
            ),
        )
        seconds = (
            self._predict_seconds("expert_up", up_work)
            + self._predict_seconds("expert_activation", activation_work)
            + self._predict_seconds("expert_down", down_work)
        )
        return self._ns(seconds)

    def batch_kernel_critical_path_ns(
            self, shape: OnlineBatchShape) -> int:
        """Return modeled serial COMP time, excluding every collective."""

        return self.batch_kernel_phases(shape).total_ns

    def batch_kernel_phases(
            self, shape: OnlineBatchShape) -> OnlineBatchKernelPhases:
        """Return prologue, one layer, and epilogue compute durations."""

        self.validate_shape(shape)
        prologue_ns = self.dense_latency_ns(
            "embedding", shape.total_tokens, shape.lm_head_sequences
        )
        layer_ns = sum((
            self.dense_latency_ns(
                "layernorm", shape.total_tokens, shape.lm_head_sequences),
            self.dense_latency_ns(
                "qkv_proj", shape.total_tokens, shape.lm_head_sequences),
            self.dense_latency_ns(
                "qk_norm", shape.total_tokens, shape.lm_head_sequences),
            self.dense_latency_ns(
                "rotary_emb", shape.total_tokens, shape.lm_head_sequences),
            self.attention_latency_ns(shape),
            self.dense_latency_ns(
                "o_proj", shape.total_tokens, shape.lm_head_sequences),
            self.dense_latency_ns(
                "layernorm", shape.total_tokens, shape.lm_head_sequences),
            self.moe_rank_latency_ns(
                total_tokens=shape.total_tokens,
                local_tokens=max(1, shape.total_tokens),
                activated_experts=min(
                    QWEN_EXPERTS // QWEN_EP,
                    max(1, math.ceil(
                        shape.total_tokens * QWEN_TOP_K / QWEN_EP
                    )),
                ),
                ep_total=QWEN_EP,
            ),
        ))
        epilogue_ns = sum((
            self.dense_latency_ns(
                "final_layernorm",
                shape.total_tokens,
                shape.lm_head_sequences,
            ),
            self.dense_latency_ns(
                "lm_head",
                shape.total_tokens,
                shape.lm_head_sequences,
            ),
            self.dense_latency_ns(
                "sampler",
                shape.total_tokens,
                shape.lm_head_sequences,
            ),
        ))
        return OnlineBatchKernelPhases(
            prologue_ns=prologue_ns,
            layer_ns=layer_ns,
            layer_count=QWEN_LAYERS,
            epilogue_ns=epilogue_ns,
        )

    def batch_kernel_comp_node_sum_ns(
            self, shape: OnlineBatchShape) -> int:
        """Return the exact sum of COMP durations emitted in the trace.

        The MoE trace contains one expert COMP node per EP rank.  Their sum
        is useful for work attribution, while ``batch_kernel_critical_path``
        uses the slowest rank and is the relevant serial compute projection.
        Neither quantity contains collective time.
        """

        self.validate_shape(shape)
        total = self.dense_latency_ns(
            "embedding", shape.total_tokens, shape.lm_head_sequences
        )
        dense_and_attention = sum((
            self.dense_latency_ns(
                "layernorm", shape.total_tokens,
                shape.lm_head_sequences),
            self.dense_latency_ns(
                "qkv_proj", shape.total_tokens,
                shape.lm_head_sequences),
            self.dense_latency_ns(
                "qk_norm", shape.total_tokens,
                shape.lm_head_sequences),
            self.dense_latency_ns(
                "rotary_emb", shape.total_tokens,
                shape.lm_head_sequences),
            self.attention_latency_ns(shape),
            self.dense_latency_ns(
                "o_proj", shape.total_tokens,
                shape.lm_head_sequences),
            self.dense_latency_ns(
                "layernorm", shape.total_tokens,
                shape.lm_head_sequences),
        ))
        activated = min(
            QWEN_EXPERTS // QWEN_EP,
            max(1, math.ceil(
                shape.total_tokens * QWEN_TOP_K / QWEN_EP
            )),
        )
        expert_per_rank = self.moe_rank_latency_ns(
            total_tokens=shape.total_tokens,
            local_tokens=max(1, shape.total_tokens),
            activated_experts=activated,
            ep_total=QWEN_EP,
        )
        total += QWEN_LAYERS * (
            dense_and_attention + QWEN_EP * expert_per_rank
        )
        total += self.dense_latency_ns(
            "final_layernorm", shape.total_tokens,
            shape.lm_head_sequences)
        total += self.dense_latency_ns(
            "lm_head", shape.total_tokens, shape.lm_head_sequences)
        total += self.dense_latency_ns(
            "sampler", shape.total_tokens, shape.lm_head_sequences)
        return total

    def recompute_attribution(
            self, shape: OnlineBatchShape) -> Mapping[str, int]:
        """Return an exact provider-level cache-hit counterfactual.

        These values are model ``COMP`` time only.  They are intentionally
        not described as ASTRA iteration time because removing recomputation
        can also change collective overlap and contention.
        """

        actual_critical = self.batch_kernel_critical_path_ns(shape)
        actual_sum = self.batch_kernel_comp_node_sum_ns(shape)
        counterfactual_shape = shape.without_recompute_queries()
        if counterfactual_shape.total_tokens == 0:
            counterfactual_critical = 0
            counterfactual_sum = 0
        else:
            counterfactual_critical = self.batch_kernel_critical_path_ns(
                counterfactual_shape
            )
            counterfactual_sum = self.batch_kernel_comp_node_sum_ns(
                counterfactual_shape
            )
        return {
            "modeled_comp_critical_path_ns": actual_critical,
            "cache_hit_counterfactual_comp_critical_path_ns": (
                counterfactual_critical),
            "modeled_comp_node_sum_ns": actual_sum,
            "cache_hit_counterfactual_comp_node_sum_ns": (
                counterfactual_sum),
            "recompute_marginal_comp_critical_path_ns": max(
                0, actual_critical - counterfactual_critical),
            "recompute_marginal_comp_node_sum_ns": max(
                0, actual_sum - counterfactual_sum),
            # Canonical paper metric: EP ranks execute in parallel, so this
            # alias is the critical-path marginal, not the rank sum.
            "recompute_marginal_comp_ns": max(
                0, actual_critical - counterfactual_critical),
            "recompute_query_tokens": sum(shape.recompute_q),
        }

    def singleton_prefill_comp_ns(
            self, *, input_tokens: int, hit_tokens: int,
            max_chunk_tokens: int) -> int:
        """Estimate one request's complete prefill COMP critical path.

        This uses the same provider and chunk geometry as the online trace
        path, but deliberately excludes collectives and queueing.  It is a
        causal policy estimate: no future batch composition or observed
        completion latency is consulted.
        """

        input_tokens = int(input_tokens)
        hit_tokens = int(hit_tokens)
        max_chunk_tokens = int(max_chunk_tokens)
        if input_tokens <= 0:
            raise OnlineLatencyModelError(
                "singleton prefill input_tokens must be positive")
        if hit_tokens < 0 or hit_tokens >= input_tokens:
            raise OnlineLatencyModelError(
                "singleton prefill hit_tokens must be in "
                "[0, input_tokens)")
        if max_chunk_tokens <= 0:
            raise OnlineLatencyModelError(
                "singleton prefill max_chunk_tokens must be positive")

        total_ns = 0
        computed = hit_tokens
        while computed < input_tokens:
            query_tokens = min(
                max_chunk_tokens, input_tokens - computed)
            shape = OnlineBatchShape(
                total_tokens=query_tokens,
                prefill_q=(query_tokens,),
                prefill_k=(computed,),
                decode_k=(),
                lm_head_sequences=1,
                recompute_q=(0,),
            )
            total_ns += self.batch_kernel_critical_path_ns(shape)
            computed += query_tokens
        return int(total_ns)

    def metadata(self) -> Mapping[str, Any]:
        base = self.calibration.metadata()
        return {
            "schema_version": 2,
            "name": H100_QWEN3_TP4_KERNEL_CALIBRATED,
            "scope": "online_trace_comp_nodes_only",
            "collectives_included_in_comp_time": False,
            "astra_collectives_remain_authoritative": True,
            "compute_accounting": {
                "paper_denominator": "modeled_comp_critical_path_ns",
                "paper_recompute_numerator": (
                    "recompute_marginal_comp_critical_path_ns"
                ),
                "emitted_comp_node_sum_ns": (
                    "validation-only sum across parallel EP rank nodes; "
                    "must not be used as elapsed model compute"
                ),
            },
            "model": TARGET_MODEL,
            "hardware": "H100",
            "tp": QWEN_TP,
            "ep": QWEN_EP,
            "dtype": "bfloat16",
            "band": self.band,
            "target_config_sha256": self.target_config_sha256,
            "long_context_experiment": (
                qwen_long_context_experiment_contract()
            ),
            "source_sha256": base["source_sha256"],
            "producer_source_sha256": base["producer_source_sha256"],
            "kernel_equation": (
                "prefill/dense: t=max(t_launch,eta*t_roof); "
                "decode-attention: t=t_launch+eta*t_roof; "
                "t_roof=max(F/P_peak*u,bytes/BW_peak)"
            ),
            "decode_attention_fits": [
                asdict(fit) for fit in self.decode_attention_fits
            ],
            "prefill_component_validation": base["validation"],
            "limits": {
                "native_max_position_embeddings": (
                    QWEN_NATIVE_MAX_POSITION_EMBEDDINGS
                ),
                "max_context_tokens": TARGET_MAX_CONTEXT,
                "max_prefill_query_chunk": TARGET_PREFILL_CHUNK,
                "one_million_prompt_execution": (
                    "online scheduler chunks at <=131072 query tokens; "
                    "prefix K grows between chunks"
                ),
                "max_decode_batch": 128,
                "source_prefill_q_max": 1_024,
                "source_prefill_k_max": 2_016,
                "long_context_is_analytical_extrapolation": True,
                "sparse_attention_enabled": False,
                "attention_execution_model": (
                    "standard_full_attention_roofline_extrapolation"
                ),
                "official_dca_kernel_reproduced": False,
                "mixed_prefill_decode_is_analytical_extrapolation": True,
                "decode_skew_alpha": DECODE_SKEW_ALPHA,
                "sampler_has_direct_measurement": False,
                "absolute_qwen_h100_accuracy_validated": False,
            },
        }

    def long_context_experiment_metadata(self) -> Mapping[str, Any]:
        """Return the compact claim contract attached to every online batch."""

        return qwen_long_context_experiment_contract()


def _target_config_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "model" / f"{TARGET_MODEL}.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_online_latency_contract(
        *, name: str, hardware: str, model: str, config: Mapping[str, Any],
        tp_size: int, pp_size: int, local_ep: int, ep_total: int,
        fp_bytes: int,
        dtype: str | None, kv_cache_dtype: str,
        enable_attn_offloading: bool) -> None:
    if name not in SUPPORTED_ONLINE_LATENCY_MODELS:
        raise OnlineLatencyModelError(
            f"unknown online latency model {name!r}; supported: "
            + ", ".join(SUPPORTED_ONLINE_LATENCY_MODELS)
        )
    errors = []
    if hardware != "H100":
        errors.append(f"hardware={hardware!r} (expected 'H100')")
    if model != TARGET_MODEL:
        errors.append(f"model={model!r} (expected {TARGET_MODEL!r})")
    if (tp_size != QWEN_TP or pp_size != 1
            or local_ep != QWEN_EP or ep_total != QWEN_EP):
        errors.append(
            f"parallelism=TP{tp_size}/PP{pp_size}/"
            f"localEP{local_ep}/totalEP{ep_total} "
            f"(expected TP{QWEN_TP}/PP1/"
            f"localEP{QWEN_EP}/totalEP{QWEN_EP})"
        )
    if fp_bytes != BF16_BYTES or dtype not in {None, "bfloat16", "bf16"}:
        errors.append(
            f"dtype={dtype!r}, fp_bytes={fp_bytes} (expected bfloat16/2)"
        )
    if kv_cache_dtype != "auto":
        errors.append(
            f"kv_cache_dtype={kv_cache_dtype!r} (expected 'auto')"
        )
    if enable_attn_offloading:
        errors.append("attention offloading is unsupported")
    expected = {
        "model_type": "qwen3_moe",
        "hidden_size": QWEN_HIDDEN_SIZE,
        "head_dim": QWEN_HEAD_DIM,
        "num_attention_heads": QWEN_Q_HEADS,
        "num_key_value_heads": QWEN_KV_HEADS,
        "num_hidden_layers": QWEN_LAYERS,
        "num_experts": QWEN_EXPERTS,
        "num_experts_per_tok": QWEN_TOP_K,
        "moe_intermediate_size": QWEN_EXPERT_INTERMEDIATE,
        "vocab_size": QWEN_VOCAB_SIZE,
        "max_position_embeddings": (
            QWEN_NATIVE_MAX_POSITION_EMBEDDINGS
        ),
    }
    mismatched = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatched:
        errors.append(f"model geometry mismatch={mismatched!r}")
    if config.get("long_context_experiment") != (
            qwen_long_context_experiment_contract()):
        errors.append(
            "long_context_experiment does not match the pinned official-"
            "source dense/full-attention sensitivity contract"
        )
    if errors:
        raise OnlineLatencyModelError(
            f"{name} contract violation: " + "; ".join(errors)
        )


@lru_cache(maxsize=3)
def build_online_latency_model(
        name: str, repo_root_text: str, target_config_sha256: str,
        band: str = "central") -> H100Qwen3OnlineLatencyModel:
    repo_root = Path(repo_root_text)
    if name != H100_QWEN3_TP4_KERNEL_CALIBRATED:
        raise OnlineLatencyModelError(
            f"unknown online latency model {name!r}"
        )
    target_path = _target_config_path(repo_root)
    if not target_path.is_file():
        raise FileNotFoundError(f"missing target model config: {target_path}")
    actual_hash = _sha256(target_path)
    if actual_hash != target_config_sha256:
        raise OnlineLatencyModelError(
            "target model config changed while constructing calibration: "
            f"expected={target_config_sha256}, actual={actual_hash}"
        )
    return H100Qwen3OnlineLatencyModel(
        calibration=fit_h100_tp4_calibration(repo_root),
        decode_attention_fits=_fit_decode_attention(repo_root),
        band=band,
        target_config_sha256=target_config_sha256,
    )


def resolve_online_latency_model(
        *, name: str | None, repo_root: Path, hardware: str, model: str,
        config: Mapping[str, Any], tp_size: int, pp_size: int,
        local_ep: int, ep_total: int, fp_bytes: int, dtype: str | None,
        kv_cache_dtype: str, enable_attn_offloading: bool,
        band: str = "central") -> H100Qwen3OnlineLatencyModel | None:
    if name in {None, "", "profile"}:
        return None
    validate_online_latency_contract(
        name=name, hardware=hardware, model=model, config=config,
        tp_size=tp_size, pp_size=pp_size, local_ep=local_ep,
        ep_total=ep_total,
        fp_bytes=fp_bytes, dtype=dtype, kv_cache_dtype=kv_cache_dtype,
        enable_attn_offloading=enable_attn_offloading,
    )
    config_path = _target_config_path(repo_root)
    return build_online_latency_model(
        name, str(repo_root.resolve()), _sha256(config_path), band
    )


def write_online_latency_provenance(
        path: Path, provider: H100Qwen3OnlineLatencyModel) -> None:
    """Write or verify a deterministic run-local provenance artifact."""

    payload = json.dumps(
        provider.metadata(), sort_keys=True, indent=2
    ) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != payload:
            raise OnlineLatencyModelError(
                f"online latency provenance changed within one run: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


__all__ = [
    "H100_QWEN3_TP4_KERNEL_CALIBRATED",
    "H100Qwen3OnlineLatencyModel",
    "OnlineBatchShape",
    "OnlineBatchKernelPhases",
    "OnlineLatencyModelError",
    "SUPPORTED_ONLINE_LATENCY_MODELS",
    "build_online_latency_model",
    "resolve_online_latency_model",
    "validate_runtime_context_contract",
    "validate_online_latency_contract",
    "write_online_latency_provenance",
]

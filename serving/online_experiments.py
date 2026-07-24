"""Bounded parallel experiment runner for the online serving simulator.

Every workload subprocess created here is exactly ``python -m serving``.  The
module never imports or invokes the standalone capacity-replay implementation.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import copy
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction

from .core.session_admission import MEASUREMENT_COHORT_SELECTIONS
from .core.session_metrics import TIMING_WARNING_CODES
from .core.agentic_kv import AgenticKVConfig


SCHEMA_VERSION = 12
MIN_CURRENT_SESSION_REPORT_SCHEMA_VERSION = 11
MIN_CURRENT_AGENTIC_REPORT_SCHEMA_VERSION = 20
DEFAULT_RUN_WALL_SECONDS = 600.0
MAX_RUN_WALL_SECONDS = 3_600.0
SUPPORTED_EXPERIMENT_MODES = ("backlog", "poisson")
DURABLE_CAPACITY_POLICIES = frozenset({"hbm_ssd_direct", "tiered"})
DURABLE_CAPACITY_CONTRACTS = frozenset({
    "terminal-ssd-lru",
    "lossless-working-set",
})
MANAGED_SERVING_FLAGS = {
    "--cluster-config", "--dataset", "--num-reqs", "--run-id", "--output",
    "--inputs-root", "--session-metrics", "--agentic-kv-metrics",
    "--agentic-kv-config", "--agentic-kv-policy",
    "--strict-infinite-hbm-oracle", "--no-strict-infinite-hbm-oracle",
    "--session-arrival-mode", "--session-arrival-rate-sps",
    "--session-arrival-seed", "--max-active-sessions",
    "--session-backlog-epochs", "--session-warmup-completions",
    "--session-measure-completions",
    "--session-measurement-cohort-selection",
    "--session-stop-after-measurement",
    "--no-session-stop-after-measurement",
}
_BACKLOG_LOAD_OVERRIDE_KEYS = frozenset({
    "backlog_epochs",
    "warmup_completions",
    "measure_completions",
    "min_fraction_at_configured_k",
})
_AGENTIC_POLICY_SPECIFIC_CONFIG_KEYS = frozenset({
    "policy",
    "queue_recompute_wait_service_ratio",
    "queue_recompute_min_wait_ms",
    "queue_recompute_cost_guard_multiplier",
    "queue_recompute_prefill_headroom_chunks",
})
_AGENTIC_HARDWARE_CONFIG_KEYS = frozenset({
    "pcie_bandwidth_gbps",
    "cpu_bandwidth_gbps",
    "cpu_transfer_latency_us",
    "pd_peer_transfer_mode",
    "pd_peer_bandwidth_gbps",
    "pd_peer_latency_us",
    "ssd_read_bandwidth_gbps",
    "ssd_write_bandwidth_gbps",
    "ssd_read_latency_us",
    "ssd_write_latency_us",
    "ssd_capacity_gb",
    "ssd_num_devices",
    "block_size",
    "io_queue_policy",
})


class ExperimentError(RuntimeError):
    pass


def _normalize_allowed_timing_warning_codes(value, setting_name):
    """Validate a stable, machine-readable timing-warning allowlist."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise ExperimentError(f"{setting_name} must be a JSON list")
    normalized = []
    for index, code in enumerate(value):
        if not isinstance(code, str) or not code.strip():
            raise ExperimentError(
                f"{setting_name}[{index}] must be a non-empty string")
        normalized.append(code.strip())
    duplicates = sorted({
        code for code in normalized if normalized.count(code) > 1
    })
    if duplicates:
        raise ExperimentError(
            f"{setting_name} contains duplicate codes: {duplicates}")
    unknown = sorted(set(normalized) - TIMING_WARNING_CODES)
    if unknown:
        raise ExperimentError(
            f"{setting_name} contains unknown timing warning codes: "
            f"{unknown}; supported={sorted(TIMING_WARNING_CODES)}")
    return normalized


def _normalize_mode_selection(requested_modes, configured_modes):
    """Return a stable, validated subset of the configured sweep modes."""
    configured = {
        str(mode) for mode in configured_modes
        if str(mode) in SUPPORTED_EXPERIMENT_MODES
    }
    if not configured:
        raise ExperimentError(
            "Experiment spec requires backlog and/or poisson mode")
    if requested_modes is None:
        return tuple(
            mode for mode in SUPPORTED_EXPERIMENT_MODES
            if mode in configured)
    if isinstance(requested_modes, str):
        requested_modes = [requested_modes]
    requested = [str(mode) for mode in requested_modes]
    if not requested:
        raise ExperimentError("At least one experiment mode must be selected")
    unsupported = sorted(
        set(requested) - set(SUPPORTED_EXPERIMENT_MODES))
    if unsupported:
        raise ExperimentError(
            f"Unsupported experiment mode selection: {unsupported}")
    duplicates = sorted({
        mode for mode in requested if requested.count(mode) > 1
    })
    if duplicates:
        raise ExperimentError(
            f"Experiment modes selected more than once: {duplicates}")
    absent = sorted(set(requested) - configured)
    if absent:
        raise ExperimentError(
            f"Selected experiment mode is absent from spec: {absent}")
    return tuple(
        mode for mode in SUPPORTED_EXPERIMENT_MODES
        if mode in requested)


def _normalize_plot_settings(value, configured_modes=None):
    """Validate optional paper-plot settings before any child is launched."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ExperimentError("plots must be a JSON object")
    unknown = sorted(set(value) - {
        "backlog_oracle_normalized", "poisson_rate_metrics",
    })
    if unknown:
        raise ExperimentError(f"Unsupported plot setting(s): {unknown}")
    settings = {}
    raw = value.get("backlog_oracle_normalized")
    if raw is not None and raw is not False:
        if raw is True:
            raw = {}
        if not isinstance(raw, dict):
            raise ExperimentError(
                "plots.backlog_oracle_normalized must be true or an object")
        unknown = sorted(set(raw) - {"minimum_k"})
        if unknown:
            raise ExperimentError(
                "Unsupported backlog oracle-normalized plot setting(s): "
                f"{unknown}")
        minimum_k = raw.get("minimum_k", 0)
        if (isinstance(minimum_k, bool)
                or not isinstance(minimum_k, (int, float))
                or not math.isfinite(float(minimum_k))
                or float(minimum_k) < 0
                or float(minimum_k) != int(minimum_k)):
            raise ExperimentError(
                "plots.backlog_oracle_normalized.minimum_k must be a "
                "non-negative integer")
        settings["backlog_oracle_normalized"] = {
            "minimum_k": int(minimum_k),
        }
        if configured_modes is not None:
            backlog = configured_modes.get("backlog")
            if not isinstance(backlog, dict):
                raise ExperimentError(
                    "backlog_oracle_normalized plot requires backlog mode")
            k_values = backlog.get("k_values")
            if not isinstance(k_values, list) or not k_values:
                raise ExperimentError(
                    "backlog_oracle_normalized plot requires non-empty "
                    "backlog.k_values")
            try:
                eligible = [
                    value for value in k_values
                    if float(value) >= int(minimum_k)
                ]
            except (TypeError, ValueError) as exc:
                raise ExperimentError(
                    "backlog.k_values must be numeric for the "
                    "oracle-normalized plot") from exc
            if not eligible:
                raise ExperimentError(
                    "backlog_oracle_normalized minimum_k excludes the entire "
                    f"backlog sweep: minimum_k={int(minimum_k)}, "
                    f"k_values={k_values}")

    raw = value.get("poisson_rate_metrics")
    if raw is not None and raw is not False:
        if raw is True:
            raw = {}
        if not isinstance(raw, dict):
            raise ExperimentError(
                "plots.poisson_rate_metrics must be true or an object")
        supported_slos = {"resume_ttft_slo", "tpot_slo"}
        unknown = sorted(set(raw) - supported_slos)
        if unknown:
            raise ExperimentError(
                "Unsupported Poisson rate-metric plot setting(s): "
                f"{unknown}")
        normalized = {}
        for key in sorted(supported_slos):
            if key not in raw or raw[key] is None:
                continue
            slo = raw[key]
            if not isinstance(slo, dict):
                raise ExperimentError(
                    f"plots.poisson_rate_metrics.{key} must be an object")
            unknown_slo = sorted(
                set(slo) - {"threshold_ms", "basis", "provenance"})
            if unknown_slo:
                raise ExperimentError(
                    f"Unsupported fields in plots.poisson_rate_metrics."
                    f"{key}: {unknown_slo}")
            threshold_ms = slo.get("threshold_ms")
            if (isinstance(threshold_ms, bool)
                    or not isinstance(threshold_ms, (int, float))
                    or not math.isfinite(float(threshold_ms))
                    or float(threshold_ms) <= 0):
                raise ExperimentError(
                    f"plots.poisson_rate_metrics.{key}.threshold_ms must "
                    "be a positive finite number")
            basis = slo.get("basis")
            if not isinstance(basis, str) or not basis.strip():
                raise ExperimentError(
                    f"plots.poisson_rate_metrics.{key}.basis must be a "
                    "non-empty string")
            provenance = slo.get("provenance")
            if not isinstance(provenance, dict) or not provenance:
                raise ExperimentError(
                    f"plots.poisson_rate_metrics.{key}.provenance must be "
                    "a non-empty JSON object")
            normalized[key] = {
                "threshold_ms": float(threshold_ms),
                "basis": basis.strip(),
                "provenance": copy.deepcopy(provenance),
            }
        settings["poisson_rate_metrics"] = normalized
        if (configured_modes is not None
                and not isinstance(configured_modes.get("poisson"), dict)):
            raise ExperimentError(
                "poisson_rate_metrics plot requires poisson mode")
    return settings


def _normalize_measurement_cohort_selection(value, mode):
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError(
            f"{mode} measurement_cohort_selection must be a non-empty "
            "string")
    normalized = value.strip().lower()
    if normalized not in MEASUREMENT_COHORT_SELECTIONS:
        raise ExperimentError(
            f"{mode} measurement_cohort_selection must be one of "
            f"{list(MEASUREMENT_COHORT_SELECTIONS)}")
    if normalized == "admission_order" and mode != "backlog":
        raise ExperimentError(
            "admission_order measurement cohorts require backlog mode")
    return normalized


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _is_sha256_digest(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_agentic_config_payload(
        payload, *, policy_override=None, strict_oracle=False):
    """Return the runtime-effective config in stable JSON scalar types."""
    try:
        unknown = sorted(
            set(payload) - set(AgenticKVConfig.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown field(s): {unknown}")
        config = AgenticKVConfig(**payload)
        if policy_override is not None:
            config.policy = str(policy_override)
        if strict_oracle:
            config.policy = "preserve"
            config.demotion_mode = "capacity-only"
        config.validate()
    except (TypeError, ValueError) as exc:
        raise ExperimentError(
            f"Invalid agentic KV config payload: {exc}") from exc
    defaults = AgenticKVConfig()
    effective = {
        name: getattr(config, name)
        for name in AgenticKVConfig.__dataclass_fields__
    }
    for name, value in tuple(effective.items()):
        default = getattr(defaults, name)
        if type(default) is float and isinstance(value, (int, float)):
            normalized = float(value)
            effective[name] = 0.0 if normalized == 0 else normalized
        elif (type(default) is int
              and isinstance(value, (int, float))
              and not isinstance(value, bool)
              and float(value).is_integer()):
            effective[name] = int(value)
        elif (type(default) is bool
              and (isinstance(value, bool) or value in (0, 1))):
            effective[name] = bool(value)
    return effective


def _agentic_config_fingerprints(
        path, *, policy_override=None, strict_oracle=False):
    """Hash normalized hardware and matched-control agentic KV settings."""
    try:
        payload = _load_json(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExperimentError(
            f"Invalid agentic KV config {path}: {exc}") from exc
    effective = _canonical_agentic_config_payload(
        payload,
        policy_override=policy_override,
        strict_oracle=strict_oracle,
    )
    hardware = {
        name: effective[name]
        for name in sorted(_AGENTIC_HARDWARE_CONFIG_KEYS)
    }
    shared_controls = {
        name: value
        for name, value in effective.items()
        if name not in _AGENTIC_POLICY_SPECIFIC_CONFIG_KEYS
    }
    return {
        "agentic_hardware_config_hash": _stable_json_hash(hardware),
        "agentic_shared_control_config_hash": _stable_json_hash(
            shared_controls),
        "agentic_effective_config_hash": _stable_json_hash(effective),
    }


def _resolve(repo_root, path):
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _apply_dataset_path_overrides(
        spec, repo_root, *, dataset_path=None, manifest_path=None):
    """Return an effective spec plus an auditable path-override record.

    Large converted traces are intentionally not committed to the repository.
    A local artifact may therefore replace only the dataset and companion
    manifest paths; the immutable SHA-256 and schema/count contract remains
    the one declared by the checked-in experiment spec.
    """
    effective = copy.deepcopy(spec)
    overrides = {}
    if dataset_path is not None:
        declared = effective.get("dataset")
        resolved = _resolve(repo_root, dataset_path)
        effective["dataset"] = str(resolved)
        overrides["dataset"] = {
            "declared_path": declared,
            "effective_path": str(resolved),
        }
    if manifest_path is not None:
        contract = effective.get("dataset_contract")
        if not isinstance(contract, dict):
            raise ExperimentError(
                "--dataset-manifest-override requires a dataset_contract")
        declared = contract.get("manifest")
        resolved = _resolve(repo_root, manifest_path)
        contract["manifest"] = str(resolved)
        overrides["dataset_manifest"] = {
            "declared_path": declared,
            "effective_path": str(resolved),
        }
    return effective, overrides


def _contract_integer(contract, key):
    value = contract.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExperimentError(f"dataset_contract.{key} must be an integer")
    if value < 0:
        raise ExperimentError(
            f"dataset_contract.{key} must be non-negative")
    return value


def _contract_sha256(contract, key):
    value = contract.get(key)
    if value is None:
        return None
    value = str(value).lower()
    if (len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ExperimentError(
            f"dataset_contract.{key} must be a 64-character SHA-256")
    return value


def _manifest_field(manifest, path, *, required=False):
    value = manifest
    for key in path:
        if not isinstance(value, dict) or key not in value:
            if required:
                raise ExperimentError(
                    "Dataset companion manifest is missing "
                    + ".".join(path))
            return None
        value = value[key]
    return value


def _prepare_dataset_contract(
        source_path, dataset_contract, repo_root):
    """Validate byte identity and companion-manifest provenance pre-parse."""
    if dataset_contract is None:
        return {"enabled": False}
    if not isinstance(dataset_contract, dict):
        raise ExperimentError("dataset_contract must be a JSON object")

    source_path = Path(source_path)
    if not source_path.is_file():
        raise ExperimentError(
            f"Dataset contract source file does not exist: {source_path}")
    actual_source_sha256 = _sha256_file(source_path)
    expected_source_sha256 = _contract_sha256(
        dataset_contract, "expected_sha256")
    if (expected_source_sha256 is not None
            and actual_source_sha256 != expected_source_sha256):
        raise ExperimentError(
            "dataset_contract.expected_sha256 mismatch for dataset: "
            f"expected={expected_source_sha256}, "
            f"actual={actual_source_sha256}")

    expected_schema_version = _contract_integer(
        dataset_contract, "expected_schema_version")
    expected_source_session_count = _contract_integer(
        dataset_contract, "expected_source_session_count")
    expected_manifest_sha256 = _contract_sha256(
        dataset_contract, "expected_manifest_sha256")
    manifest_value = dataset_contract.get("manifest")
    if expected_manifest_sha256 is not None and manifest_value is None:
        raise ExperimentError(
            "dataset_contract.expected_manifest_sha256 requires "
            "dataset_contract.manifest")

    manifest_path = None
    manifest_sha256 = None
    manifest = None
    schema_verified_by = None
    manifest_session_count = None
    if manifest_value is not None:
        if not isinstance(manifest_value, str) or not manifest_value.strip():
            raise ExperimentError(
                "dataset_contract.manifest must be a non-empty path")
        manifest_path = _resolve(repo_root, manifest_value)
        if not manifest_path.is_file():
            raise ExperimentError(
                "Dataset companion manifest does not exist: "
                f"{manifest_path}")
        manifest_sha256 = _sha256_file(manifest_path)
        if (expected_manifest_sha256 is not None
                and manifest_sha256 != expected_manifest_sha256):
            raise ExperimentError(
                "dataset_contract.expected_manifest_sha256 mismatch: "
                f"expected={expected_manifest_sha256}, "
                f"actual={manifest_sha256}")
        try:
            manifest = _load_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentError(
                f"Invalid dataset companion manifest {manifest_path}: "
                f"{exc}") from exc
        if not isinstance(manifest, dict):
            raise ExperimentError(
                "Dataset companion manifest must contain a JSON object")

        manifest_output_sha256 = str(_manifest_field(
            manifest, ("output", "sha256"), required=True)).lower()
        if manifest_output_sha256 != actual_source_sha256:
            raise ExperimentError(
                "Dataset companion manifest output.sha256 does not match "
                f"the dataset: manifest={manifest_output_sha256}, "
                f"actual={actual_source_sha256}")

        manifest_session_count = _manifest_field(
            manifest, ("summary", "sessions_emitted"), required=True)
        if (isinstance(manifest_session_count, bool)
                or not isinstance(manifest_session_count, int)
                or manifest_session_count < 0):
            raise ExperimentError(
                "Dataset companion manifest summary.sessions_emitted must "
                "be a non-negative integer")
        if (expected_source_session_count is not None
                and manifest_session_count != expected_source_session_count):
            raise ExperimentError(
                "Dataset companion manifest session count conflicts with "
                "dataset_contract.expected_source_session_count: "
                f"manifest={manifest_session_count}, "
                f"expected={expected_source_session_count}")

        if expected_schema_version is not None:
            manifest_schema_version = _manifest_field(
                manifest, ("schema_version",), required=True)
            if manifest_schema_version != expected_schema_version:
                raise ExperimentError(
                    "Dataset companion manifest schema_version conflicts "
                    "with dataset_contract.expected_schema_version: "
                    f"manifest={manifest_schema_version}, "
                    f"expected={expected_schema_version}")
            schema_verified_by = "companion_manifest"

        manifest_source = _manifest_field(
            manifest, ("source",), required=True)
        if not isinstance(manifest_source, dict):
            raise ExperimentError(
                "Dataset companion manifest source must be a JSON object")
        source_contract_fields = {
            "source_sha256": "sha256",
            "source_revision": "revision",
            "tracelab_reuse_mode": "tracelab_reuse_mode",
        }
        for contract_key, manifest_key in source_contract_fields.items():
            if contract_key not in dataset_contract:
                continue
            expected_value = (
                _contract_sha256(dataset_contract, contract_key)
                if contract_key == "source_sha256" else
                dataset_contract[contract_key]
            )
            actual_value = manifest_source.get(manifest_key)
            if contract_key == "source_sha256" and actual_value is not None:
                actual_value = str(actual_value).lower()
            if actual_value != expected_value:
                raise ExperimentError(
                    f"Dataset companion manifest source.{manifest_key} "
                    f"conflicts with dataset_contract.{contract_key}: "
                    f"manifest={actual_value!r}, expected={expected_value!r}")

    return {
        "enabled": True,
        "contract": dict(dataset_contract),
        "source_sha256": actual_source_sha256,
        "expected_schema_version": expected_schema_version,
        "expected_source_session_count": expected_source_session_count,
        "manifest_path": (
            str(manifest_path) if manifest_path is not None else None),
        "manifest_sha256": manifest_sha256,
        "manifest_session_count": manifest_session_count,
        "schema_verified_by": schema_verified_by,
    }


def _load_json(path):
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary, path)


def _git_provenance(repo_root):
    def command(*args):
        result = subprocess.run(
            list(args), cwd=repo_root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = command("git", "status", "--porcelain=v1")
    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "branch": command("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_sha256": (
            _sha256_bytes(status.encode("utf-8"))
            if status is not None else None
        ),
    }


def _session_features(row, index):
    sub_requests = row.get("sub_requests")
    if not sub_requests:
        raise ExperimentError(
            f"Online session experiment row {index} has no sub_requests")
    try:
        input_tokens = [int(sub["input_toks"]) for sub in sub_requests]
        output_tokens_by_request = [
            int(sub["output_toks"]) for sub in sub_requests]
        gaps_ns = [
            int(sub.get("tool_duration_ns", 0) or 0)
            for sub in sub_requests[:-1]
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentError(
            f"Invalid token or gap field in session row {index}: {exc}") from exc
    contexts = [
        prompt + generated
        for prompt, generated in zip(input_tokens, output_tokens_by_request)
    ]
    if (any(value < 0 for value in contexts)
            or any(value < 0 for value in input_tokens)
            or any(value < 0 for value in output_tokens_by_request)):
        raise ExperimentError(f"Negative token count in session row {index}")
    if any(value < 0 for value in gaps_ns):
        raise ExperimentError(f"Negative inter-turn gap in session row {index}")
    max_context = max(contexts)
    output_tokens = sum(output_tokens_by_request)
    gap_types = sorted({
        str(sub.get("inter_turn_gap_type") or "unknown").lower()
        for sub in sub_requests[:-1]
    }) or ["none"]
    reuse_eligible_transitions = 0
    reuse_eligible_tokens = 0
    for sub in sub_requests[1:]:
        value = sub.get(
            "policy_independent_reuse_toks",
            sub.get("prefix_reuse_toks", 0),
        )
        try:
            value = max(0, int(value or 0))
        except (TypeError, ValueError) as exc:
            raise ExperimentError(
                f"Invalid reuse token field in session row {index}: "
                f"{value!r}") from exc
        reuse_eligible_tokens += value
        reuse_eligible_transitions += int(value > 0)
    session_id = str(row.get("session_id", f"row-{index}"))
    return {
        "index": index,
        "session_id": session_id,
        "max_context_tokens": max_context,
        "context_log2_bucket": int(math.log2(max(1, max_context))),
        "gap_types": gap_types,
        "output_tokens": output_tokens,
        "request_count": len(sub_requests),
        "total_gap_ns": sum(gaps_ns),
        "max_gap_ns": max(gaps_ns, default=0),
        "reuse_eligible_transitions": reuse_eligible_transitions,
        "reuse_eligible_tokens": reuse_eligible_tokens,
    }


def _selection_filters(selection):
    integer_filters = {
        "min_context_tokens": (0, None),
        "max_context_tokens": (0, None),
        "max_output_tokens_per_session": (0, None),
        "min_requests_per_session": (1, None),
        "max_requests_per_session": (1, None),
        "min_total_gap_ns": (0, None),
        "max_total_gap_ns": (0, None),
        "max_single_gap_ns": (0, None),
        "min_reuse_eligible_transitions": (0, None),
    }
    filters = {}
    for key, (minimum, _) in integer_filters.items():
        raw = selection.get(key)
        if raw is None:
            filters[key] = None
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ExperimentError(
                f"workload_selection.{key} must be an integer") from exc
        if value < minimum:
            raise ExperimentError(
                f"workload_selection.{key} must be >= {minimum}")
        filters[key] = value

    for minimum_key, maximum_key in (
            ("min_context_tokens", "max_context_tokens"),
            ("min_requests_per_session", "max_requests_per_session"),
            ("min_total_gap_ns", "max_total_gap_ns")):
        minimum = filters[minimum_key]
        maximum = filters[maximum_key]
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ExperimentError(
                f"workload_selection.{minimum_key} exceeds {maximum_key}")

    include_indices = selection.get("include_source_indices")
    if include_indices is not None:
        if (not isinstance(include_indices, list)
                or any(isinstance(value, bool) for value in include_indices)):
            raise ExperimentError(
                "workload_selection.include_source_indices must be a list")
        try:
            include_indices = [int(value) for value in include_indices]
        except (TypeError, ValueError) as exc:
            raise ExperimentError(
                "workload_selection.include_source_indices must contain "
                "integers") from exc
        if any(value < 0 for value in include_indices):
            raise ExperimentError("include_source_indices cannot be negative")
        if len(set(include_indices)) != len(include_indices):
            raise ExperimentError("include_source_indices contains duplicates")
        filters["include_source_indices"] = include_indices
    else:
        filters["include_source_indices"] = None

    for key in ("allowed_gap_types", "required_gap_types"):
        raw = selection.get(key)
        if raw is None:
            filters[key] = None
            continue
        if not isinstance(raw, list) or not raw:
            raise ExperimentError(f"workload_selection.{key} must be a list")
        values = sorted({str(value).lower() for value in raw})
        filters[key] = values
    return filters


def _filter_rejection(feature, filters):
    include = filters["include_source_indices"]
    if include is not None and feature["index"] not in include:
        return "source_index_not_selected"
    comparisons = (
        ("min_context_tokens", "max_context_tokens", "max_context_tokens"),
        ("max_context_tokens", "max_context_tokens", "max_context_tokens"),
        ("max_output_tokens_per_session", "output_tokens", "output_tokens"),
        ("min_requests_per_session", "request_count", "request_count"),
        ("max_requests_per_session", "request_count", "request_count"),
        ("min_total_gap_ns", "total_gap_ns", "total_gap_ns"),
        ("max_total_gap_ns", "total_gap_ns", "total_gap_ns"),
        ("max_single_gap_ns", "max_gap_ns", "max_gap_ns"),
        (
            "min_reuse_eligible_transitions",
            "reuse_eligible_transitions",
            "reuse_eligible_transitions",
        ),
    )
    for filter_key, feature_key, label in comparisons:
        bound = filters[filter_key]
        if bound is None:
            continue
        value = feature[feature_key]
        if filter_key.startswith("min_") and value < bound:
            return f"below_{label}_minimum"
        if filter_key.startswith("max_") and value > bound:
            return f"above_{label}_maximum"
    gap_types = set(feature["gap_types"])
    allowed = filters["allowed_gap_types"]
    if allowed is not None and not gap_types.issubset(set(allowed)):
        return "disallowed_gap_type"
    required = filters["required_gap_types"]
    if required is not None and not set(required).issubset(gap_types):
        return "missing_required_gap_type"
    return None


def _context_scaling_lineage_break(sub_request_index, sub_request):
    """Return whether a source transition explicitly breaks KV lineage."""
    if sub_request_index == 0:
        return True
    status = str(
        sub_request.get(
            "lineage_status",
            sub_request.get("prefix_lineage_scope", ""),
        ) or ""
    ).strip().lower()
    if status in {"session_start", "context_shrink", "round_gap"}:
        return True
    normalized = status.replace("-", "_").replace(" ", "_")
    return (
        "compaction" in normalized
        or "context_reset" in normalized
        or "reset_context" in normalized
        or "lineage_break" in normalized
    )


def _scale_lineage_reuse(
        source_reuse, factor, *, source_input_tokens,
        scaled_input_tokens, previous_source_input_tokens,
        previous_output_tokens, previous_scaled_input_tokens,
        lineage_break):
    """Map one source prefix coordinate into the transformed lineage."""
    source_reuse = max(0, int(source_reuse))
    if lineage_break or previous_source_input_tokens is None:
        return 0, {
            "source_reuse_tokens": source_reuse,
            "source_reuse_after_bounds": 0,
            "predecessor_available_tokens": 0,
            "lineage_break": True,
        }

    source_predecessor_tokens = (
        int(previous_source_input_tokens) + int(previous_output_tokens))
    scaled_predecessor_tokens = (
        int(previous_scaled_input_tokens) + int(previous_output_tokens))
    bounded_source_reuse = min(
        source_reuse,
        int(source_input_tokens),
        source_predecessor_tokens,
    )

    # Prefix coordinates inside the predecessor prompt expand by the global
    # factor. A suffix that lies in the predecessor's generated output does
    # not expand because output_toks is deliberately unchanged. In
    # particular, a full adjacent prefix maps exactly to the transformed
    # predecessor context instead of inventing unavailable KV tokens.
    if bounded_source_reuse <= int(previous_source_input_tokens):
        scaled_reuse = (
            bounded_source_reuse * factor.numerator
            // factor.denominator
        )
    else:
        output_prefix_tokens = (
            bounded_source_reuse - int(previous_source_input_tokens))
        scaled_reuse = (
            int(previous_scaled_input_tokens) + output_prefix_tokens)
    scaled_reuse = min(
        scaled_reuse,
        int(scaled_input_tokens),
        scaled_predecessor_tokens,
    )
    return scaled_reuse, {
        "source_reuse_tokens": source_reuse,
        "source_reuse_after_bounds": bounded_source_reuse,
        "predecessor_available_tokens": scaled_predecessor_tokens,
        "lineage_break": False,
    }


def _apply_context_length_scaling(expanded, target_max_sequence_tokens):
    if target_max_sequence_tokens is None:
        return {
            "enabled": False,
            "semantics": "no length transform",
        }
    try:
        target = int(target_max_sequence_tokens)
    except (TypeError, ValueError) as exc:
        raise ExperimentError(
            "target_max_sequence_tokens must be an integer") from exc
    if target <= 0:
        raise ExperimentError(
            "target_max_sequence_tokens must be positive")

    calls = []
    for row, feature in expanded:
        for sub_request_index, sub in enumerate(row["sub_requests"]):
            try:
                input_tokens = int(sub["input_toks"])
                output_tokens = int(sub["output_toks"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ExperimentError(
                    "Context scaling encountered an invalid token field at "
                    f"source row {feature['index']}, request "
                    f"{sub_request_index}") from exc
            if input_tokens <= 0 or output_tokens < 0:
                raise ExperimentError(
                    "Context scaling requires positive input and non-negative "
                    "output token counts")
            if output_tokens >= target:
                raise ExperimentError(
                    f"Output tokens {output_tokens} leave no prompt capacity "
                    f"under target {target}")
            calls.append((
                row, feature, sub_request_index, sub,
                input_tokens, output_tokens,
            ))
    source_max = max(
        input_tokens + output_tokens
        for _, _, _, _, input_tokens, output_tokens in calls
    )
    if target < source_max:
        raise ExperimentError(
            "target_max_sequence_tokens cannot shrink the selected cohort; "
            f"source maximum is {source_max}, target is {target}")

    # The minimum per-call ratio is the one global rational factor that
    # maximizes all prompts without allowing any unchanged output to cross the
    # target. Integer floor is then exact and platform-independent.
    factor = min(
        Fraction(target - output_tokens, input_tokens)
        for _, _, _, _, input_tokens, output_tokens in calls
    )
    scaled_fields = (
        "input_toks",
        "prefix_reuse_toks",
        "policy_independent_reuse_toks",
    )
    required_fields = (*scaled_fields, "newly_append_toks")
    preserved_observation_fields = (
        "reported_input_toks",
        "observed_provider_hit_toks",
    )
    scaled_max = 0
    reuse_adjustments = {
        "lineage_break_fields_zeroed": 0,
        "source_predecessor_or_input_caps": 0,
        "scaled_predecessor_or_input_caps": 0,
    }
    for row, feature in expanded:
        previous_source_input_tokens = None
        previous_output_tokens = None
        previous_scaled_input_tokens = None
        for sub_request_index, sub in enumerate(row["sub_requests"]):
            missing = [key for key in required_fields if key not in sub]
            if missing:
                raise ExperimentError(
                    "Auditable context scaling requires TraceLab operational "
                    f"fields {missing} at source row {feature['index']}, "
                    f"request {sub_request_index}")
            original = {}
            for key in (*required_fields, *preserved_observation_fields):
                if key in sub:
                    try:
                        original[key] = int(sub[key])
                    except (TypeError, ValueError) as exc:
                        raise ExperimentError(
                            f"Context scaling field {key} is not an integer "
                            f"at source row {feature['index']}, request "
                            f"{sub_request_index}") from exc
            for key in required_fields:
                if original[key] < 0:
                    raise ExperimentError(
                        f"Context scaling field {key} cannot be negative")

            source_input_tokens = original["input_toks"]
            output_tokens = int(sub["output_toks"])
            scaled_input_tokens = (
                source_input_tokens * factor.numerator
                // factor.denominator
            )
            if scaled_input_tokens <= 0:
                raise ExperimentError(
                    "Context scaling produced a non-positive prompt")
            sub["input_toks"] = scaled_input_tokens

            lineage_break = _context_scaling_lineage_break(
                sub_request_index, sub)
            reuse_audit = {}
            for reuse_key in (
                    "prefix_reuse_toks",
                    "policy_independent_reuse_toks"):
                source_reuse = original[reuse_key]
                scaled_reuse, audit = _scale_lineage_reuse(
                    source_reuse,
                    factor,
                    source_input_tokens=source_input_tokens,
                    scaled_input_tokens=scaled_input_tokens,
                    previous_source_input_tokens=(
                        previous_source_input_tokens),
                    previous_output_tokens=previous_output_tokens,
                    previous_scaled_input_tokens=(
                        previous_scaled_input_tokens),
                    lineage_break=lineage_break,
                )
                if lineage_break and source_reuse > 0:
                    reuse_adjustments[
                        "lineage_break_fields_zeroed"] += 1
                if (audit["source_reuse_after_bounds"]
                        < audit["source_reuse_tokens"]):
                    reuse_adjustments[
                        "source_predecessor_or_input_caps"] += 1
                source_mapped_without_scaled_caps = (
                    scaled_reuse
                    if lineage_break
                    else (
                        audit["source_reuse_after_bounds"]
                        * factor.numerator // factor.denominator
                        if audit["source_reuse_after_bounds"]
                        <= int(previous_source_input_tokens)
                        else int(previous_scaled_input_tokens)
                        + audit["source_reuse_after_bounds"]
                        - int(previous_source_input_tokens)
                    )
                )
                if scaled_reuse < source_mapped_without_scaled_caps:
                    reuse_adjustments[
                        "scaled_predecessor_or_input_caps"] += 1
                sub[reuse_key] = scaled_reuse
                reuse_audit[reuse_key] = audit

            # This legacy field becomes an auditable operational suffix in a
            # synthetic-length workload. The untouched raw field and the
            # nested original retain the source converter's append count.
            sub["newly_append_toks"] = (
                scaled_input_tokens - int(sub["prefix_reuse_toks"]))
            if (int(sub["prefix_reuse_toks"])
                    + int(sub["newly_append_toks"])
                    != scaled_input_tokens):
                raise ExperimentError(
                    "Scaled prefix reuse and newly appended tokens do not "
                    "reconcile with input_toks")
            for reuse_key in (
                    "prefix_reuse_toks",
                    "policy_independent_reuse_toks"):
                if not 0 <= int(sub[reuse_key]) <= scaled_input_tokens:
                    raise ExperimentError(
                        f"Scaled {reuse_key} exceeds scaled input_toks at "
                        f"source row {feature['index']}, request "
                        f"{sub_request_index}")
                if (previous_scaled_input_tokens is None or lineage_break):
                    predecessor_available = 0
                else:
                    predecessor_available = (
                        int(previous_scaled_input_tokens)
                        + int(previous_output_tokens))
                if int(sub[reuse_key]) > predecessor_available:
                    raise ExperimentError(
                        f"Scaled {reuse_key} exceeds transformed predecessor "
                        f"KV at source row {feature['index']}, request "
                        f"{sub_request_index}")
            if scaled_input_tokens + output_tokens > target:
                raise ExperimentError(
                    "Global context factor failed its target bound")

            sub["online_length_scaling_original"] = original
            sub["online_length_scaling_lineage"] = {
                "lineage_status": sub.get("lineage_status"),
                "lineage_break": lineage_break,
                "predecessor_available_tokens": (
                    0 if previous_scaled_input_tokens is None
                    else int(previous_scaled_input_tokens)
                    + int(previous_output_tokens)
                ),
                "reuse_fields": reuse_audit,
                "newly_append_definition": (
                    "input_toks - prefix_reuse_toks"),
            }
            scaled_max = max(
                scaled_max, scaled_input_tokens + output_tokens)

            # Synthetic prompt lengths have no token-ID realization. The
            # source file remains content-addressed in the cohort manifest,
            # while the operational simulator uses explicit prefix reuse.
            if "input_tok_ids" in sub:
                original_ids = sub.pop("input_tok_ids")
                sub["online_length_scaling_original"][
                    "input_tok_ids_count"] = len(original_ids)
                sub["online_length_scaling_original"][
                    "input_tok_ids_sha256"] = _stable_json_hash(original_ids)

            previous_source_input_tokens = source_input_tokens
            previous_output_tokens = output_tokens
            previous_scaled_input_tokens = scaled_input_tokens

    transform = {
        "enabled": True,
        "label": "global_context_length_scaled_sensitivity",
        "empirical_length_distribution": False,
        "source_max_sequence_tokens": source_max,
        "target_max_sequence_tokens": target,
        "realized_max_sequence_tokens": scaled_max,
        "global_factor_numerator": factor.numerator,
        "global_factor_denominator": factor.denominator,
        "global_factor": float(factor),
        "rounding": (
            "integer floor for scaled prompt-prefix coordinates; generated "
            "output suffixes remain unscaled"),
        "scaled_operational_fields": list(scaled_fields),
        "derived_operational_fields": ["newly_append_toks"],
        "reuse_lineage_semantics": (
            "reuse is mapped within the transformed predecessor prompt plus "
            "its unchanged generated output, then bounded by predecessor KV "
            "and the current prompt; explicit lineage breaks map to zero"),
        "newly_append_semantics": (
            "operational input_toks - prefix_reuse_toks; source append "
            "provenance remains in raw_newly_append_toks and "
            "online_length_scaling_original.newly_append_toks"),
        "reuse_adjustments": reuse_adjustments,
        "unchanged_fields": [
            "output_toks", "tool_duration_ns", "inter_turn_gap_type",
            "lineage_status", "reported_input_toks",
            "observed_provider_hit_toks", "raw_newly_append_toks",
            "source/session ordering",
        ],
        "original_field": "sub_requests[].online_length_scaling_original",
        "input_token_ids": (
            "removed when present because scaled lengths have no synthetic "
            "token-ID realization; original count/hash and source-file hash "
            "provide provenance"
        ),
    }
    if scaled_max != target:
        raise ExperimentError(
            f"Context scaling failed to realize target exactly: "
            f"realized={scaled_max}, target={target}")
    for row, _ in expanded:
        if "online_experiment_context_transform" in row:
            raise ExperimentError(
                "Session already uses reserved field "
                "online_experiment_context_transform")
        row["online_experiment_context_transform"] = dict(transform)
    for row, feature in expanded:
        feature["operational_max_sequence_tokens"] = max(
            int(sub["input_toks"]) + int(sub["output_toks"])
            for sub in row["sub_requests"]
        )
        feature["context_length_scaled"] = True
    return transform


def materialize_session_cohort(
        source_path, output_dir, selection=None, *, dataset_contract=None,
        repo_root=None):
    """Select complete sessions and return an auditable cohort descriptor."""
    source_path = Path(source_path)
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    contract_validation = _prepare_dataset_contract(
        source_path, dataset_contract, Path(repo_root))
    selection = dict(selection or {})
    strategy = str(selection.get("strategy", "all"))
    if strategy not in {
            "all", "head", "stratified_context_gap",
            "long_context_low_output"}:
        raise ExperimentError(
            "workload_selection.strategy must be all, head, "
            "stratified_context_gap, or long_context_low_output")
    max_sessions = int(selection.get("max_sessions", 0) or 0)
    max_output_tokens = int(selection.get("max_total_output_tokens", 0) or 0)
    repetitions = int(selection.get("repetitions", 1))
    seed = int(selection.get("seed", 42))
    if max_sessions < 0 or max_output_tokens < 0:
        raise ExperimentError("workload selection budgets must be non-negative")
    if repetitions <= 0:
        raise ExperimentError("workload_selection.repetitions must be positive")
    filters = _selection_filters(selection)

    rows = []
    row_schema_versions = []
    try:
        with open(source_path, "r", encoding="utf-8") as source:
            for index, line in enumerate(source):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ExperimentError(
                        f"Invalid dataset JSON at source row {index}: "
                        f"{exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ExperimentError(
                        f"Dataset source row {index} must be a JSON object")
                row_schema_versions.append(row.get("schema_version"))
                rows.append((row, _session_features(row, index)))
    except OSError as exc:
        raise ExperimentError(
            f"Unable to read dataset source {source_path}: {exc}") from exc
    if not rows:
        raise ExperimentError("Session experiment workload is empty")

    parsed_source_session_count = len(rows)
    if contract_validation["enabled"]:
        expected_count = contract_validation[
            "expected_source_session_count"]
        if (expected_count is not None
                and parsed_source_session_count != expected_count):
            raise ExperimentError(
                "Parsed dataset row count conflicts with "
                "dataset_contract.expected_source_session_count: "
                f"parsed={parsed_source_session_count}, "
                f"expected={expected_count}")
        manifest_count = contract_validation["manifest_session_count"]
        if (manifest_count is not None
                and parsed_source_session_count != manifest_count):
            raise ExperimentError(
                "Parsed dataset row count conflicts with companion manifest "
                f"summary.sessions_emitted: parsed={parsed_source_session_count}, "
                f"manifest={manifest_count}")

        expected_schema = contract_validation["expected_schema_version"]
        present_row_schemas = [
            value for value in row_schema_versions if value is not None]
        if present_row_schemas:
            if len(present_row_schemas) != parsed_source_session_count:
                raise ExperimentError(
                    "Converted dataset row schema_version is present on only "
                    "a subset of source rows")
            if (expected_schema is not None
                    and any(value != expected_schema
                            for value in present_row_schemas)):
                mismatched = sorted({
                    repr(value) for value in present_row_schemas
                    if value != expected_schema
                })
                raise ExperimentError(
                    "Converted dataset row schema_version conflicts with "
                    "dataset_contract.expected_schema_version: "
                    f"observed={mismatched[:5]}, expected={expected_schema}")
            if expected_schema is not None:
                contract_validation["schema_verified_by"] = (
                    "companion_manifest_and_rows"
                    if contract_validation["schema_verified_by"] else
                    "converted_rows"
                )
        if (expected_schema is not None
                and contract_validation["schema_verified_by"] is None):
            raise ExperimentError(
                "dataset_contract.expected_schema_version cannot be "
                "verified: provide dataset_contract.manifest or a matching "
                "schema_version on every converted row")

        post_parse_sha256 = _sha256_file(source_path)
        if post_parse_sha256 != contract_validation["source_sha256"]:
            raise ExperimentError(
                "Dataset source changed while its contract was being "
                "validated")

    rejection_counts = {}
    filtered_rows = []
    for row, feature in rows:
        rejection = _filter_rejection(feature, filters)
        if rejection is None:
            filtered_rows.append((row, feature))
        else:
            rejection_counts[rejection] = rejection_counts.get(rejection, 0) + 1
    if filters["include_source_indices"] is not None:
        present = {feature["index"] for _, feature in rows}
        missing = sorted(
            set(filters["include_source_indices"]) - present)
        if missing:
            raise ExperimentError(
                "include_source_indices references missing row(s): "
                f"{missing[:10]}")
    if not filtered_rows:
        raise ExperimentError(
            "Workload filters produced no complete sessions; inspect the "
            "filter rejection counts or relax the bounds")
    rows = filtered_rows

    if strategy == "all":
        order = list(range(len(rows)))
    elif strategy == "head":
        order = list(range(len(rows)))
    elif strategy == "long_context_low_output":
        order = sorted(
            range(len(rows)),
            key=lambda position: (
                -rows[position][1]["max_context_tokens"],
                rows[position][1]["output_tokens"],
                rows[position][1]["total_gap_ns"],
                rows[position][1]["index"],
            ),
        )
    else:
        strata = {}
        for position, (_, feature) in enumerate(rows):
            key = (
                feature["context_log2_bucket"],
                tuple(feature["gap_types"]),
            )
            strata.setdefault(key, []).append(position)
        rng = random.Random(seed)
        for values in strata.values():
            rng.shuffle(values)
            values.sort(
                key=lambda position: rows[position][1][
                    "max_context_tokens"],
                reverse=True,
            )
        keys = sorted(strata, key=lambda key: (-key[0], key[1]))
        order = []
        while keys:
            next_keys = []
            for key in keys:
                values = strata[key]
                if values:
                    order.append(values.pop(0))
                if values:
                    next_keys.append(key)
            keys = next_keys

    selected = []
    selected_output_tokens = 0
    for position in order:
        row, feature = rows[position]
        if max_sessions and len(selected) >= max_sessions:
            break
        candidate_tokens = selected_output_tokens + feature["output_tokens"]
        if max_output_tokens and candidate_tokens > max_output_tokens:
            continue
        selected.append((row, feature))
        selected_output_tokens = candidate_tokens
    if not selected:
        raise ExperimentError(
            "Workload selection produced no complete sessions; increase the "
            "session or output-token budget")

    # Input ordering is restored after ranked selection so the simulator sees
    # a stable surrogate order rather than a selection traversal order.
    selected.sort(key=lambda item: item[1]["index"])
    duplicate_session_ids = {}
    for _, feature in selected:
        duplicate_session_ids.setdefault(feature["session_id"], []).append(
            feature["index"])
    duplicate_session_ids = {
        session_id: indices
        for session_id, indices in duplicate_session_ids.items()
        if len(indices) > 1
    }
    if duplicate_session_ids:
        examples = list(sorted(duplicate_session_ids.items()))[:5]
        raise ExperimentError(
            "Selected complete sessions have duplicate session_id values; "
            "online dependency tracking requires conversion-time collision "
            f"disambiguation. Examples: {examples}")
    expanded = []
    for repetition_index in range(repetitions):
        for row, feature in selected:
            materialized_row = copy.deepcopy(row)
            materialized_feature = dict(feature)
            materialized_feature["source_session_id"] = feature["session_id"]
            materialized_feature["repetition_index"] = repetition_index
            if repetitions > 1:
                materialized_session_id = (
                    f"{feature['session_id']}::online-rep-{repetition_index}")
                materialized_row["session_id"] = materialized_session_id
                materialized_feature["session_id"] = materialized_session_id
            provenance_key = "online_experiment_source"
            if provenance_key in materialized_row:
                raise ExperimentError(
                    f"Session row {feature['index']} already uses reserved "
                    f"field {provenance_key}")
            materialized_row[provenance_key] = {
                "source_index": feature["index"],
                "source_session_id": feature["session_id"],
                "repetition_index": repetition_index,
            }
            expanded.append((materialized_row, materialized_feature))

    context_transform = _apply_context_length_scaling(
        expanded, selection.get("target_max_sequence_tokens"))

    cohort_dir = Path(output_dir) / "cohort"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    workload_path = cohort_dir / "selected_sessions.jsonl"
    with open(workload_path, "w", encoding="utf-8") as output:
        for row, _ in expanded:
            output.write(json.dumps(row, sort_keys=True))
            output.write("\n")
    session_records = [feature for _, feature in expanded]
    sessions_path = cohort_dir / "selected_session_ids.json"
    _write_json(sessions_path, session_records)
    selected_session_identity_hash = _stable_json_hash([
        {
            "source_index": feature["index"],
            "session_id": feature["session_id"],
        }
        for feature in session_records
    ])
    selected_request_count = sum(
        feature["request_count"] for feature in session_records)
    if contract_validation["enabled"]:
        contract = contract_validation["contract"]
        expected_template_count = _contract_integer(
            contract, "expected_selected_template_count")
        if (expected_template_count is not None
                and len(selected) != expected_template_count):
            raise ExperimentError(
                "Selected cohort template count conflicts with "
                "dataset_contract.expected_selected_template_count: "
                f"selected={len(selected)}, "
                f"expected={expected_template_count}")
        expected_request_count = _contract_integer(
            contract, "expected_selected_request_count")
        if (expected_request_count is not None
                and selected_request_count != expected_request_count):
            raise ExperimentError(
                "Selected cohort request count conflicts with "
                "dataset_contract.expected_selected_request_count: "
                f"selected={selected_request_count}, "
                f"expected={expected_request_count}")
        expected_identity_hash = _contract_sha256(
            contract, "expected_selected_session_identity_hash")
        if (expected_identity_hash is not None
                and selected_session_identity_hash != expected_identity_hash):
            raise ExperimentError(
                "Selected cohort identity conflicts with dataset_contract."
                "expected_selected_session_identity_hash: "
                f"selected={selected_session_identity_hash}, "
                f"expected={expected_identity_hash}")

    descriptor = {
        "source_path": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "materialized_path": str(workload_path.resolve()),
        "materialized_sha256": _sha256_file(workload_path),
        "selected_session_records_path": str(sessions_path.resolve()),
        "selected_session_records_sha256": _sha256_file(sessions_path),
        "selected_session_ids_hash": _stable_json_hash([
            feature["session_id"] for feature in session_records
        ]),
        "selected_session_identity_hash": selected_session_identity_hash,
        "selected_template_count": len(selected),
        "selected_session_count": len(expanded),
        "selected_request_count": selected_request_count,
        "selected_output_tokens_per_repetition": selected_output_tokens,
        "selected_output_tokens": selected_output_tokens * repetitions,
        "context_length_transform": context_transform,
        "selection_rule": {
            "unit": "complete_session",
            "strategy": strategy,
            "max_sessions": max_sessions or None,
            "max_total_output_tokens": max_output_tokens or None,
            "repetitions": repetitions,
            "seed": seed,
            "filters": filters,
            "filter_and_transform_order": (
                "select complete source sessions using empirical lengths, "
                "then apply one global operational context factor"
            ),
            "source_session_count": (
                len(rows) + sum(rejection_counts.values())),
            "eligible_session_count_after_filters": len(rows),
            "filter_rejection_counts": dict(sorted(rejection_counts.items())),
            "strata": "floor(log2(max_context_tokens)) x gap_type_set",
            "long_context_low_output_order": (
                "max_context_desc, output_tokens_asc, total_gap_asc, "
                "source_index_asc"
            ),
            "post_selection_order": "source surrogate order",
            "repetition_order": (
                "repetition_index then source surrogate order; every copy "
                "has a unique session_id and online_experiment_source record"
            ),
            "sub_requests_truncated": False,
        },
        "dataset_contract_validation": {
            "enabled": contract_validation["enabled"],
            "passed": True,
            "source_sha256": contract_validation.get("source_sha256"),
            "parsed_source_session_count": parsed_source_session_count,
            "manifest_path": contract_validation.get("manifest_path"),
            "manifest_sha256": contract_validation.get("manifest_sha256"),
            "schema_verified_by": contract_validation.get(
                "schema_verified_by"),
        },
    }
    _write_json(cohort_dir / "cohort_manifest.json", descriptor)
    return descriptor


def _flag_name(argument):
    return str(argument).split("=", 1)[0]


def _validate_common_args(arguments):
    conflicts = sorted({
        _flag_name(argument)
        for argument in arguments
        if _flag_name(argument) in MANAGED_SERVING_FLAGS
    })
    if conflicts:
        raise ExperimentError(
            f"common_serving_args overrides managed flags: {conflicts}")


def _slug(value):
    return "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in str(value)
    ).strip("-")


def _new_execution_id(name):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    nonce = os.urandom(4).hex()
    return _slug(f"{name}-{timestamp}-{nonce}")


def _integer_setting(value, label, *, minimum):
    if isinstance(value, bool):
        fraction = None
    else:
        try:
            fraction = Fraction(str(value))
        except (ValueError, ZeroDivisionError):
            fraction = None
    expectation = "positive" if minimum == 1 else "non-negative"
    if fraction is None or fraction.denominator != 1:
        raise ExperimentError(f"{label} must be a {expectation} integer")
    result = int(fraction)
    if result < minimum:
        raise ExperimentError(f"{label} must be a {expectation} integer")
    return result


def _normalize_backlog_load_values(values):
    normalized = []
    source_by_value = {}
    for value in values:
        load = _integer_setting(value, "backlog K", minimum=1)
        if load in source_by_value:
            raise ExperimentError(
                "backlog k_values contains duplicate values after integer "
                f"normalization: {source_by_value[load]!r} and {value!r}")
        source_by_value[load] = value
        normalized.append(load)
    return normalized


def _normalize_backlog_load_overrides(raw_overrides, configured_loads):
    if not isinstance(raw_overrides, dict):
        raise ExperimentError("backlog load_overrides must be a JSON object")
    normalized = {}
    source_by_value = {}
    for raw_load, raw_override in raw_overrides.items():
        load = _integer_setting(
            raw_load, "backlog load_overrides key", minimum=1)
        if load in source_by_value:
            raise ExperimentError(
                "backlog load_overrides contains duplicate keys after "
                f"integer normalization: {source_by_value[load]!r} and "
                f"{raw_load!r}")
        source_by_value[load] = raw_load
        if not isinstance(raw_override, dict):
            raise ExperimentError(
                f"backlog load_overrides[{raw_load!r}] must be a JSON "
                "object")
        unsupported = sorted(
            set(raw_override) - _BACKLOG_LOAD_OVERRIDE_KEYS,
            key=str,
        )
        if unsupported:
            raise ExperimentError(
                f"backlog load_overrides[{raw_load!r}] contains "
                f"unsupported keys: {unsupported}")
        normalized[load] = dict(raw_override)

    unknown = sorted(set(normalized) - set(configured_loads))
    if unknown:
        raise ExperimentError(
            "backlog load_overrides contains K values absent from "
            f"k_values: {unknown}")
    return normalized


def _resolve_completion_window(
        mode, settings, *, available_sessions, require_complete,
        stop_after_measurement, load_description=None):
    description = str(mode)
    if load_description is not None:
        description += f" {load_description}"
    warmup = _integer_setting(
        settings.get("warmup_completions", 0),
        f"{description} warmup_completions",
        minimum=0,
    )
    measure_value = settings.get("measure_completions", 0)
    if (isinstance(measure_value, str)
            and measure_value.strip().lower() == "all"):
        measure = available_sessions - warmup
    else:
        measure = _integer_setting(
            measure_value,
            f"{description} measure_completions",
            minimum=1,
        )
    if measure <= 0:
        raise ExperimentError(
            f"{description} mode requires measure_completions > 0 or "
            "'all'")
    if available_sessions < warmup + measure:
        raise ExperimentError(
            f"{description} cohort has {available_sessions} sessions but "
            f"warmup + measurement requires {warmup + measure}")
    if (require_complete and stop_after_measurement
            and available_sessions != warmup + measure):
        raise ExperimentError(
            f"{description} stops at the measurement boundary and requires "
            "the complete generated cohort, but "
            f"available={available_sessions}, warmup+measure="
            f"{warmup + measure}")
    return warmup, measure


def _backlog_min_fraction(settings, *, load_description):
    raw_value = settings.get("min_fraction_at_configured_k")
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        value = math.nan
    else:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = math.nan
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ExperimentError(
            f"backlog {load_description} min_fraction_at_configured_k "
            "must be in [0, 1]")
    return value


def _materialize_mode_workload(
        cohort, output_dir, *, mode, session_repetitions):
    """Return a mode-local complete-session workload descriptor.

    Poisson steady-state runs often need more independent arrivals than the
    selected empirical template set contains. Repetition happens at the
    session row boundary and assigns every copy a distinct runtime ID; no
    sub-request chain is shortened or duplicated in isolation.
    """
    repetitions = int(session_repetitions)
    if repetitions <= 0:
        raise ExperimentError("session_repetitions must be positive")
    if repetitions == 1:
        return {
            "materialized_path": cohort["materialized_path"],
            "materialized_sha256": cohort["materialized_sha256"],
            "selected_session_count": cohort["selected_session_count"],
            "selected_request_count": cohort["selected_request_count"],
            "selected_session_ids_hash": cohort[
                "selected_session_ids_hash"],
            "selected_session_identity_hash": cohort[
                "selected_session_identity_hash"],
            "session_repetitions": 1,
        }

    rows = []
    with open(cohort["materialized_path"], "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            if not row.get("sub_requests"):
                raise ExperimentError(
                    f"Mode repetition requires a complete non-empty agentic "
                    f"session at workload row {line_number}")
            rows.append(row)
    if not rows:
        raise ExperimentError("Mode repetition received an empty workload")

    repeated = []
    identities = []
    for repetition_index in range(repetitions):
        for source_index, row in enumerate(rows):
            materialized = copy.deepcopy(row)
            source_session_id = str(row["session_id"])
            runtime_session_id = (
                f"{source_session_id}::{mode}-rep-{repetition_index}")
            materialized["session_id"] = runtime_session_id
            provenance_key = "online_experiment_mode_repetition"
            if provenance_key in materialized:
                raise ExperimentError(
                    f"Session {source_session_id!r} already uses reserved "
                    f"field {provenance_key}")
            materialized[provenance_key] = {
                "mode": str(mode),
                "source_session_id": source_session_id,
                "source_index": source_index,
                "repetition_index": repetition_index,
            }
            repeated.append(materialized)
            identities.append({
                "session_id": runtime_session_id,
                "source_session_id": source_session_id,
                "source_index": source_index,
                "repetition_index": repetition_index,
            })

    cohort_dir = Path(output_dir) / "cohort"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    workload_path = (
        cohort_dir / f"{mode}_sessions_rep-{repetitions}.jsonl")
    with open(workload_path, "w", encoding="utf-8") as output:
        for row in repeated:
            output.write(json.dumps(row, sort_keys=True))
            output.write("\n")
    return {
        "materialized_path": str(workload_path.resolve()),
        "materialized_sha256": _sha256_file(workload_path),
        "selected_session_count": len(repeated),
        "selected_request_count": sum(
            len(row["sub_requests"]) for row in repeated),
        "selected_session_ids_hash": _stable_json_hash([
            row["session_id"] for row in repeated]),
        "selected_session_identity_hash": _stable_json_hash(identities),
        "session_repetitions": repetitions,
    }


def build_run_descriptors(
        spec, repo_root, output_dir, cohort, *, execution_id=None,
        selected_modes=None):
    common_args = [str(value) for value in spec.get(
        "common_serving_args", [])]
    _validate_common_args(common_args)
    prefix_cache_flags = {
        _flag_name(argument)
        for argument in common_args
        if _flag_name(argument) in {
            "--enable-prefix-caching", "--no-enable-prefix-caching"}
    }
    if not prefix_cache_flags:
        # Agentic reuse must be accounted by AgenticKVManager; generic radix
        # prefix hits would otherwise contaminate every baseline and oracle.
        common_args.append("--no-enable-prefix-caching")
    elif len(prefix_cache_flags) > 1:
        raise ExperimentError(
            "common_serving_args contains contradictory prefix-cache flags")
    cluster_config = _resolve(repo_root, spec["cluster_config"])
    if not cluster_config.is_file():
        raise ExperimentError(f"Missing cluster config: {cluster_config}")
    policies = spec.get("policies", {})
    if not policies:
        raise ExperimentError("Experiment spec requires at least one policy")
    normalized_policies = {}
    for policy_order, (label, value) in enumerate(policies.items()):
        value = {"agentic_kv_config": value} if isinstance(value, str) else dict(value)
        config = _resolve(repo_root, value["agentic_kv_config"])
        if not config.is_file():
            raise ExperimentError(f"Missing policy config: {config}")
        config_payload = _load_json(config)
        expected_policy = value.get(
            "agentic_kv_policy", config_payload.get("policy"))
        if not expected_policy:
            raise ExperimentError(
                f"Policy {label!r} has no effective agentic KV policy")
        expected_policy = str(expected_policy)
        durable_capacity_contract = value.get(
            "durable_capacity_contract")
        if expected_policy in DURABLE_CAPACITY_POLICIES:
            durable_capacity_contract = str(
                durable_capacity_contract or "terminal-ssd-lru")
            if durable_capacity_contract not in DURABLE_CAPACITY_CONTRACTS:
                raise ExperimentError(
                    f"Policy {label!r} has unsupported durable capacity "
                    f"contract {durable_capacity_contract!r}; expected one "
                    f"of {sorted(DURABLE_CAPACITY_CONTRACTS)}")
        elif durable_capacity_contract is not None:
            raise ExperimentError(
                f"Policy {label!r} cannot set durable_capacity_contract "
                f"for non-durable policy {expected_policy!r}")
        normalized_policies[str(label)] = {
            "agentic_kv_config": config,
            "agentic_kv_policy": value.get("agentic_kv_policy"),
            "expected_agentic_policy": expected_policy,
            "durable_capacity_contract": durable_capacity_contract,
            "policy_order": policy_order,
            **_agentic_config_fingerprints(
                config,
                policy_override=value.get("agentic_kv_policy"),
            ),
        }
    oracle_config = _resolve(
        repo_root,
        spec.get(
            "oracle_agentic_kv_config",
            next(iter(normalized_policies.values()))["agentic_kv_config"],
        ),
    )
    if not oracle_config.is_file():
        raise ExperimentError(f"Missing oracle policy config: {oracle_config}")
    oracle_label = str(spec.get(
        "oracle_label", "infinite_hbm_oracle"))
    if oracle_label in normalized_policies:
        raise ExperimentError(
            f"oracle_label {oracle_label!r} collides with a baseline label")

    series = dict(normalized_policies)
    series[oracle_label] = {
        "agentic_kv_config": oracle_config,
        "agentic_kv_policy": "preserve",
        "strict_oracle": True,
        "expected_agentic_policy": "preserve",
        "durable_capacity_contract": None,
        "policy_order": len(normalized_policies),
        **_agentic_config_fingerprints(
            oracle_config,
            policy_override="preserve",
            strict_oracle=True,
        ),
    }
    for hash_field, description, compared_series in (
            ("agentic_hardware_config_hash", "hardware", series),
            ("agentic_shared_control_config_hash", "shared-control",
             normalized_policies)):
        hashes = {
            label: policy[hash_field]
            for label, policy in compared_series.items()
        }
        if len(set(hashes.values())) != 1:
            raise ExperimentError(
                f"Paired agentic {description} config mismatch before "
                f"launch: {hashes}")

    runs = []
    execution_slug = _slug(execution_id) if execution_id else None
    modes = spec.get("modes", {})
    selected_modes = _normalize_mode_selection(selected_modes, modes)
    for mode in selected_modes:
        mode_spec = dict(modes[mode])
        measurement_cohort_selection = (
            _normalize_measurement_cohort_selection(
                mode_spec.get(
                    "measurement_cohort_selection", "completion_order"),
                mode,
            )
        )
        values = mode_spec.get("k_values" if mode == "backlog" else "rates_sps", [])
        if not values:
            raise ExperimentError(f"{mode} mode requires non-empty values")
        if mode == "backlog":
            values = _normalize_backlog_load_values(values)
            load_overrides = _normalize_backlog_load_overrides(
                mode_spec.get("load_overrides", {}), values)
        else:
            if "load_overrides" in mode_spec:
                raise ExperimentError(
                    "load_overrides is supported only for backlog mode")
            load_overrides = {}
        session_repetitions = _integer_setting(
            mode_spec.get("session_repetitions", 1),
            f"{mode} session_repetitions",
            minimum=1,
        )
        if mode != "poisson" and session_repetitions != 1:
            raise ExperimentError(
                "session_repetitions is a Poisson mode option; use "
                "backlog_epochs for closed-backlog repetition")
        require_complete = bool(
            mode_spec.get("require_complete_session_cohort", False))
        stop_after_measurement = bool(
            mode_spec.get("stop_after_measurement", True))
        allow_timing_warnings = bool(mode_spec.get(
            "allow_timing_warnings",
            spec.get("allow_timing_warnings", False),
        ))
        allowed_timing_warning_codes = (
            _normalize_allowed_timing_warning_codes(
                mode_spec.get(
                    "allowed_timing_warning_codes",
                    spec.get("allowed_timing_warning_codes"),
                ),
                f"{mode}.allowed_timing_warning_codes",
            )
        )
        mode_cohort = _materialize_mode_workload(
            cohort,
            output_dir,
            mode=mode,
            session_repetitions=session_repetitions,
        )
        if mode == "poisson":
            poisson_max_active_sessions = _integer_setting(
                mode_spec.get("max_active_sessions", 0),
                "poisson max_active_sessions",
                minimum=0,
            )
            backlog_epochs = _integer_setting(
                mode_spec.get("backlog_epochs", 1),
                "poisson backlog_epochs",
                minimum=1,
            )
            available_sessions = mode_cohort["selected_session_count"]
            warmup, measure = _resolve_completion_window(
                mode,
                mode_spec,
                available_sessions=available_sessions,
                require_complete=require_complete,
                stop_after_measurement=stop_after_measurement,
            )
            raw_seeds = mode_spec.get("arrival_seeds")
            if raw_seeds is None:
                arrival_seeds = [int(mode_spec.get("arrival_seed", 42))]
            else:
                if (not isinstance(raw_seeds, list) or not raw_seeds):
                    raise ExperimentError(
                        "poisson arrival_seeds must be a non-empty list")
                arrival_seeds = [int(value) for value in raw_seeds]
            if len(arrival_seeds) != len(set(arrival_seeds)):
                raise ExperimentError(
                    "poisson arrival_seeds must not contain duplicates")
            if any(seed < 0 for seed in arrival_seeds):
                raise ExperimentError(
                    "poisson arrival seeds must be non-negative")
        else:
            poisson_max_active_sessions = None
            arrival_seeds = [None]

        for value in values:
            if mode == "backlog":
                normalized_value = value
                load_label = f"k-{normalized_value}"
                load_override_applied = normalized_value in load_overrides
                effective_settings = dict(mode_spec)
                effective_settings.update(
                    load_overrides.get(normalized_value, {}))
                backlog_epochs = _integer_setting(
                    effective_settings.get("backlog_epochs", 1),
                    f"backlog K={normalized_value} backlog_epochs",
                    minimum=1,
                )
                available_sessions = (
                    mode_cohort["selected_session_count"] * backlog_epochs)
                warmup, measure = _resolve_completion_window(
                    mode,
                    effective_settings,
                    available_sessions=available_sessions,
                    require_complete=require_complete,
                    stop_after_measurement=stop_after_measurement,
                    load_description=f"K={normalized_value}",
                )
                min_fraction_at_k = _backlog_min_fraction(
                    effective_settings,
                    load_description=f"K={normalized_value}",
                )
                effective_load_settings = {
                    "backlog_epochs": backlog_epochs,
                    "warmup_completions": warmup,
                    "measure_completions": measure,
                    "min_fraction_at_configured_k": min_fraction_at_k,
                }
            else:
                normalized_value = float(value)
                if not math.isfinite(normalized_value) or normalized_value <= 0:
                    raise ExperimentError(
                        f"Poisson rate must be finite and positive, got {value}")
                load_label = f"rate-{normalized_value:g}"
                min_fraction_at_k = None
                load_override_applied = False
                effective_load_settings = None
            for arrival_seed in arrival_seeds:
                seed_label = (
                    f"-seed-{arrival_seed}"
                    if mode == "poisson" else "")
                for label, policy in series.items():
                    run_name = (
                        f"{mode}-{load_label}{seed_label}-{label}")
                    run_id = _slug(
                        f"{execution_slug}-{run_name}"
                        if execution_slug else
                        f"{spec.get('name', 'online')}-{run_name}")
                    run_dir = Path(output_dir) / "runs" / run_id
                    session_metrics = run_dir / "session_metrics.json"
                    agentic_metrics = run_dir / "agentic_kv_metrics.json"
                    request_csv = run_dir / "requests.csv"
                    argv = [
                        sys.executable, "-m", "serving",
                        "--cluster-config", str(cluster_config),
                        "--dataset", mode_cohort["materialized_path"],
                        "--run-id", run_id,
                        "--session-metrics", str(session_metrics.resolve()),
                        "--agentic-kv-metrics", str(agentic_metrics.resolve()),
                        "--output", str(request_csv.resolve()),
                        "--agentic-kv-config",
                        str(policy["agentic_kv_config"]),
                        "--session-arrival-mode", mode,
                        "--session-warmup-completions", str(warmup),
                        "--session-measure-completions", str(measure),
                        "--session-measurement-cohort-selection",
                        measurement_cohort_selection,
                        (
                            "--session-stop-after-measurement"
                            if stop_after_measurement else
                            "--no-session-stop-after-measurement"
                        ),
                    ]
                    if policy.get("agentic_kv_policy"):
                        argv.extend([
                            "--agentic-kv-policy",
                            str(policy["agentic_kv_policy"]),
                        ])
                    if policy.get("strict_oracle"):
                        argv.append("--strict-infinite-hbm-oracle")
                    if mode == "backlog":
                        argv.extend([
                            "--max-active-sessions", str(normalized_value),
                            "--session-backlog-epochs", str(backlog_epochs),
                        ])
                    else:
                        argv.extend([
                            "--session-arrival-rate-sps",
                            str(normalized_value),
                            "--session-arrival-seed", str(arrival_seed),
                        ])
                        if poisson_max_active_sessions > 0:
                            argv.extend([
                                "--max-active-sessions",
                                str(poisson_max_active_sessions),
                            ])
                    argv.extend(common_args)
                    if argv[:3] != [sys.executable, "-m", "serving"]:
                        raise AssertionError(
                            "Experiment escaped online serving path")
                    expected_runtime_session_count = None
                    expected_runtime_session_ids_hash = None
                    expected_measurement_warmup_session_ids = None
                    expected_measurement_target_session_ids = None
                    expected_measurement_required_session_ids = None
                    if measurement_cohort_selection == "admission_order":
                        expected_runtime_sessions = _expected_runtime_sessions({
                            "workload_path": mode_cohort["materialized_path"],
                            "mode": mode,
                            "backlog_epochs": backlog_epochs,
                        })
                        expected_runtime_ids = list(
                            expected_runtime_sessions)
                        expected_runtime_session_count = len(
                            expected_runtime_ids)
                        expected_runtime_session_ids_hash = _stable_json_hash(
                            expected_runtime_ids)
                        expected_measurement_warmup_session_ids = (
                            expected_runtime_ids[:warmup]
                        )
                        expected_measurement_target_session_ids = (
                            expected_runtime_ids[warmup:warmup + measure]
                        )
                        expected_measurement_required_session_ids = (
                            expected_measurement_warmup_session_ids
                            + expected_measurement_target_session_ids
                        )
                        if (len(expected_measurement_target_session_ids)
                                != measure):
                            raise ExperimentError(
                                "Unable to materialize the complete "
                                "admission-order measurement target")
                        if (len(expected_measurement_required_session_ids)
                                != warmup + measure):
                            raise ExperimentError(
                                "Unable to materialize the complete fixed "
                                "admission-prefix warmup and target")
                    runs.append({
                        "run_id": run_id,
                        "run_dir": str(run_dir.resolve()),
                        "mode": mode,
                        "load_value": float(normalized_value),
                        "policy": label,
                        "strict_oracle": bool(policy.get("strict_oracle")),
                        "expected_agentic_policy": policy[
                            "expected_agentic_policy"],
                        "durable_capacity_contract": policy.get(
                            "durable_capacity_contract"),
                        "argv": argv,
                        "session_metrics": str(session_metrics.resolve()),
                        "agentic_kv_metrics": str(agentic_metrics.resolve()),
                        "request_csv": str(request_csv.resolve()),
                        "pair_key": (
                            f"{mode}:{normalized_value}:seed={arrival_seed}"),
                        "execution_id": execution_id,
                        "arrival_seed": arrival_seed,
                        "arrival_seed_count": len(arrival_seeds),
                        "session_repetitions": session_repetitions,
                        "max_active_sessions": (
                            int(normalized_value)
                            if mode == "backlog" else
                            int(poisson_max_active_sessions)
                        ),
                        "stop_after_measurement": stop_after_measurement,
                        "measurement_cohort_selection": (
                            measurement_cohort_selection),
                        "expected_runtime_session_count": (
                            expected_runtime_session_count),
                        "expected_runtime_session_ids_hash": (
                            expected_runtime_session_ids_hash),
                        "expected_measurement_warmup_session_ids": (
                            expected_measurement_warmup_session_ids),
                        "expected_measurement_warmup_session_ids_hash": (
                            _stable_json_hash(
                                expected_measurement_warmup_session_ids)
                            if expected_measurement_warmup_session_ids
                            is not None else None
                        ),
                        "expected_measurement_target_session_ids": (
                            expected_measurement_target_session_ids),
                        "expected_measurement_target_session_ids_hash": (
                            _stable_json_hash(
                                expected_measurement_target_session_ids)
                            if expected_measurement_target_session_ids
                            is not None else None
                        ),
                        "expected_measurement_required_session_ids": (
                            expected_measurement_required_session_ids),
                        "expected_measurement_required_session_ids_hash": (
                            _stable_json_hash(
                                expected_measurement_required_session_ids)
                            if expected_measurement_required_session_ids
                            is not None else None
                        ),
                        "min_fraction_at_configured_k": min_fraction_at_k,
                        "warmup_completions": warmup,
                        "measure_completions": measure,
                        "backlog_epochs": backlog_epochs,
                        "load_override_applied": load_override_applied,
                        "effective_load_settings": effective_load_settings,
                        "available_sessions": available_sessions,
                        "expected_request_count": (
                            mode_cohort["selected_request_count"]
                            * (backlog_epochs if mode == "backlog" else 1)
                        ),
                        "require_complete_session_cohort": require_complete,
                        "allow_timing_warnings": allow_timing_warnings,
                        "allowed_timing_warning_codes": (
                            allowed_timing_warning_codes),
                        "policy_order": int(policy["policy_order"]),
                        "cluster_config_path": str(cluster_config),
                        "agentic_kv_config_path": str(
                            policy["agentic_kv_config"]),
                        "config_sha256": _sha256_file(
                            policy["agentic_kv_config"]),
                        "agentic_hardware_config_hash": policy[
                            "agentic_hardware_config_hash"],
                        "agentic_shared_control_config_hash": policy[
                            "agentic_shared_control_config_hash"],
                        "agentic_effective_config_hash": policy[
                            "agentic_effective_config_hash"],
                        "cluster_config_sha256": _sha256_file(
                            cluster_config),
                        "workload_sha256": mode_cohort[
                            "materialized_sha256"],
                        "workload_path": mode_cohort["materialized_path"],
                        "selected_session_ids_hash": mode_cohort[
                            "selected_session_ids_hash"],
                        "selected_session_identity_hash": mode_cohort[
                            "selected_session_identity_hash"],
                    })
    if not runs:
        raise ExperimentError("Spec enables neither backlog nor poisson mode")
    run_ids = [run["run_id"] for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ExperimentError(
            "Experiment generated duplicate run IDs; check policy labels and "
            "load values after slug normalization")
    return runs


def _terminate_process_group(process, deadline):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        remaining = max(0.0, float(deadline) - time.monotonic())
        if remaining > 0:
            process.wait(timeout=min(1.0, remaining))
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        remaining = max(0.0, float(deadline) - time.monotonic())
        if remaining > 0:
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                # Never let an uninterruptible child defeat the advertised
                # per-run wall cap. poll() on later collection can reap it.
                pass


def execute_run(run, repo_root, timeout_seconds, provenance):
    timeout_seconds = float(timeout_seconds)
    if not 0 < timeout_seconds <= MAX_RUN_WALL_SECONDS:
        raise ExperimentError(
            "Per-run timeout must be positive and no greater than "
            f"{MAX_RUN_WALL_SECONDS:g} seconds")
    run_dir = Path(run["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    started = _utc_now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        **{key: value for key, value in run.items() if key != "run_dir"},
        "run_dir": str(run_dir),
        "started_at": started,
        "timeout_seconds": timeout_seconds,
        "provenance": provenance,
        "status": "running",
        "online_path_invariant": run["argv"][:3] == [
            sys.executable, "-m", "serving"],
    }
    _write_json(manifest_path, manifest)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    timed_out = False
    launch_error = None
    wall_start = time.monotonic()
    wall_deadline = wall_start + timeout_seconds
    try:
        with open(stdout_path, "w", encoding="utf-8") as stdout, open(
                stderr_path, "w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                run["argv"],
                cwd=repo_root,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            try:
                # Reserve two seconds inside the advertised wall cap for
                # process-group termination.
                run_budget = max(0.001, timeout_seconds - 2.0)
                return_code = process.wait(timeout=run_budget)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_group(process, wall_deadline)
                return_code = process.returncode
    except OSError as exc:
        return_code = None
        launch_error = f"{type(exc).__name__}: {exc}"
    manifest.update({
        "finished_at": _utc_now(),
        "return_code": return_code,
        "timed_out": timed_out,
        "wall_duration_seconds": time.monotonic() - wall_start,
        "status": (
            "launch_failed" if launch_error else
            "timeout" if timed_out else "succeeded" if return_code == 0
            else "failed"
        ),
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
    })
    if launch_error:
        manifest.setdefault("errors", []).append(launch_error)
    if return_code == 0:
        for key in ("session_metrics", "agentic_kv_metrics", "request_csv"):
            path = Path(run[key])
            if not path.is_file():
                manifest["status"] = "failed"
                manifest.setdefault("errors", []).append(
                    f"missing {key}: {path}")
            else:
                manifest[f"{key}_sha256"] = _sha256_file(path)
    _write_json(manifest_path, manifest)
    return manifest


def _measured_session_ids(report):
    values = [
        row["session_id"]
        for row in report["sessions"]["records"]
        if row.get("measurement_included")
    ]
    if len(values) != len(set(values)):
        raise ExperimentError(
            "Measured session cohort contains duplicate session IDs")
    return sorted(values)


def _distribution_columns(prefix, distribution):
    distribution = distribution or {}
    return {
        f"{prefix}_{stat}_ns": distribution.get(stat)
        for stat in ("mean", "p50", "p90", "p99", "max")
    }


def _percentile(sorted_values, percentile):
    """Match the simulator report's linear-interpolation percentile."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * float(percentile) / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _exact_distribution(values, *, name):
    """Build exact mean/p95 accounting without silently filtering values."""
    values = list(values)
    invalid = [
        value for value in values
        if value is None or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value)) or float(value) < 0
        or int(value) != value
    ]
    if invalid:
        raise ExperimentError(
            f"{name} contains invalid exact samples: {invalid[:5]}")
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"count": 0, "sum": 0, "mean": None, "p95": None}
    return {
        "count": len(ordered),
        "sum": sum(ordered),
        "mean": sum(ordered) / len(ordered),
        "p95": _percentile(ordered, 95),
    }


def _full_completed_cohort_scope(
        report, manifest, measured_session_ids):
    """Identify a drained all-session cohort without trusting request rows."""
    lifecycle_records = report.get("sessions", {}).get("records", [])
    completed_session_ids = {
        str(record.get("session_id"))
        for record in lifecycle_records
        if record.get("status") == "completed"
    }
    return bool(
        manifest.get("stop_after_measurement") is False
        and (report.get("session_admission") or {}).get(
            "cutoff_disposition") == "drain"
        and lifecycle_records
        and all(record.get("status") == "completed"
                for record in lifecycle_records)
        and set(measured_session_ids) == completed_session_ids
    )


def _measured_request_csv_rows(
        manifest, measured_session_ids, *, full_completed_cohort=False):
    """Load exact measured request rows used for composite HBM wait."""
    request_csv = manifest.get("request_csv")
    current_schema = int(manifest.get("schema_version", 0) or 0)
    if not request_csv:
        if current_schema >= SCHEMA_VERSION:
            raise ExperimentError(
                f"Current run {manifest['run_id']} is missing request_csv")
        return None
    path = Path(request_csv)
    if not path.is_file():
        raise ExperimentError(
            f"Missing request CSV for {manifest['run_id']}: {path}")
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {
            "request id", "session_id",
            "agentic_kv_hit_tokens", "agentic_kv_recompute_tokens",
            "agentic_kv_source",
            "return_gap_type",
            "agentic_kv_hbm_admission_wait_ns",
            "pd_chunk_admission_count",
            "pd_chunk_cancelled_admission_count",
            "pd_chunk_admission_wait_ns_total",
            "pd_chunk_admission_critical_wait_ns_total",
            "pd_chunk_successful_admission_wait_ns_total",
            "pd_chunk_successful_admission_critical_wait_ns_total",
            "pd_chunk_cancelled_admission_wait_ns_total",
            "pd_chunk_cancelled_admission_critical_wait_ns_total",
            "active_prefill_recompute_preemptions",
            "active_prefill_recompute_tokens",
            "active_prefill_recompute_frontier_tokens",
            "pd_active_prefill_recompute_generation",
            "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute",
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            if current_schema >= SCHEMA_VERSION:
                raise ExperimentError(
                    f"Current request CSV for {manifest['run_id']} is "
                    f"missing exact request accounting fields: {missing}")
            return None
        raw_rows = list(reader)
        outside = [
            row for row in raw_rows
            if str(row.get("session_id")) not in measured_session_ids
        ]
        if full_completed_cohort and outside:
            raise ExperimentError(
                f"Full-cohort request CSV for {manifest['run_id']} has "
                f"out-of-cohort rows: {outside[:3]}")
        rows = [
            row for row in raw_rows
            if str(row.get("session_id")) in measured_session_ids
        ]
    return rows


def _cross_layer_request_accounting_audit(
        exact_records, measured_session_ids, report, agentic_report,
        manifest, *, full_completed_cohort=False):
    """Join current per-request records to authoritative manager events."""
    current_schema = int(manifest.get("schema_version", 0) or 0)
    if current_schema < SCHEMA_VERSION:
        return {
            "performed": False,
            "passed": None,
            "reason": "legacy_online_artifact_schema",
        }
    run_id = manifest["run_id"]
    if int(agentic_report.get("schema_version", 0) or 0) < 20:
        raise ExperimentError(
            f"Current run {run_id} lacks schema-20 manager events")
    events = agentic_report.get("events")
    totals = agentic_report.get("totals")
    if not isinstance(events, list) or not isinstance(totals, dict):
        raise ExperimentError(
            f"Current run {run_id} lacks exact manager events/totals")

    records_by_id = {
        int(record["request_id"]): record for record in exact_records
    }
    full_cohort = bool(full_completed_cohort)
    event_names = {
        "active": "pd_active_prefill_recompute_preempt",
        "successful": "pd_chunk_admission",
        "cancelled": (
            "pd_chunk_admission_cancelled_for_active_prefill_recompute"),
        "resume": "resume",
    }
    selected_events = {
        ledger: {request_id: [] for request_id in records_by_id}
        for ledger in event_names
    }
    unmatched_scoped_event_count = 0

    def event_int(event, field, label):
        value = event.get(field)
        if (not isinstance(value, int) or isinstance(value, bool)
                or value < 0):
            raise ExperimentError(
                f"Malformed {label}.{field} for {run_id}: {value!r}")
        return value

    for ledger, event_name in event_names.items():
        for index, event in enumerate(events):
            if event.get("event") != event_name:
                continue
            label = f"{event_name}[{index}]"
            request_id = event_int(event, "request_id", label)
            event_session_id = str(event.get("session_id"))
            record = records_by_id.get(request_id)
            if record is None:
                if full_cohort or event_session_id in measured_session_ids:
                    raise ExperimentError(
                        f"{'Full-cohort' if full_cohort else 'Measured-session'} "
                        "manager event lacks an exact "
                        f"request record for {run_id}: {label}, "
                        f"request={request_id}, session={event_session_id}")
                unmatched_scoped_event_count += 1
                continue
            record_session_id = str(record.get("session_id"))
            if event_session_id != record_session_id:
                raise ExperimentError(
                    f"Manager event/request session mismatch for {run_id}: "
                    f"{label}, event={event_session_id}, "
                    f"record={record_session_id}")
            selected_events[ledger][request_id].append(event)

    for request_id, record in records_by_id.items():
        active = selected_events["active"][request_id]
        active_count = len(active)
        active_discarded_tokens = [
            event_int(event, "discarded_tokens", "active-prefill event")
            for event in active
        ]
        active_tokens = sum(active_discarded_tokens)
        restored_discarded = sum(event_int(
            event, "restored_hit_tokens_discarded",
            "active-prefill event") for event in active)
        active_expected = {
            "active_prefill_recompute_preemptions": active_count,
            "pd_active_prefill_recompute_generation": active_count,
            "active_prefill_recompute_tokens": active_tokens,
            "active_prefill_recompute_frontier_tokens": max(
                active_discarded_tokens, default=0),
            "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute": (
                restored_discarded),
        }
        for field, expected in active_expected.items():
            if int(record[field]) != expected:
                raise ExperimentError(
                    f"Active-prefill event/request mismatch for {run_id}, "
                    f"request={request_id}: {field}={record[field]!r}, "
                    f"events={expected}")
        if active:
            last = active[-1]
            if (event_int(
                    last, "cumulative_active_prefill_recompute_tokens",
                    "last active-prefill event") != active_tokens
                    or event_int(
                        last, "cumulative_restored_hit_tokens_discarded",
                        "last active-prefill event")
                    != restored_discarded):
                raise ExperimentError(
                    f"Active-prefill cumulative event/request mismatch for "
                    f"{run_id}, request={request_id}")

        successful = selected_events["successful"][request_id]
        cancelled = selected_events["cancelled"][request_id]
        successful_wait = sum(event_int(
            event, "wait_ns", "successful P/D chunk event")
            for event in successful)
        successful_critical = sum(event_int(
            event, "critical_wait_after_restore_ns",
            "successful P/D chunk event") for event in successful)
        cancelled_wait = sum(event_int(
            event, "wait_ns", "cancelled P/D chunk event")
            for event in cancelled)
        cancelled_critical = sum(event_int(
            event, "critical_wait_after_restore_ns",
            "cancelled P/D chunk event") for event in cancelled)
        chunk_expected = {
            "pd_chunk_admission_count": len(successful),
            "pd_chunk_cancelled_admission_count": len(cancelled),
            "pd_chunk_successful_admission_wait_ns_total": successful_wait,
            "pd_chunk_successful_admission_critical_wait_ns_total": (
                successful_critical),
            "pd_chunk_cancelled_admission_wait_ns_total": cancelled_wait,
            "pd_chunk_cancelled_admission_critical_wait_ns_total": (
                cancelled_critical),
            "pd_chunk_admission_wait_ns_total": (
                successful_wait + cancelled_wait),
            "pd_chunk_admission_critical_wait_ns_total": (
                successful_critical + cancelled_critical),
        }
        for field, expected in chunk_expected.items():
            if int(record[field]) != expected:
                raise ExperimentError(
                    f"P/D chunk event/request mismatch for {run_id}, "
                    f"request={request_id}: {field}={record[field]!r}, "
                    f"events={expected}")

        resume = selected_events["resume"][request_id]
        sub_request_index = int(record["sub_request_index"])
        expected_resume_events = int(sub_request_index > 0)
        if len(resume) != expected_resume_events:
            raise ExperimentError(
                f"Resume event/request cardinality mismatch for {run_id}, "
                f"request={request_id}: events={len(resume)}, "
                f"sub_request_index={sub_request_index}")
        if resume:
            event = resume[0]
            event_sub_index = event_int(
                event, "sub_request_index", "resume event")
            comparisons = {
                "sub_request_index": event_sub_index,
                "agentic_kv_hit_tokens": event_int(
                    event, "hit_tokens", "resume event"),
                "agentic_kv_recompute_tokens": event_int(
                    event, "recompute_tokens", "resume event"),
                "agentic_kv_hbm_admission_wait_ns": event_int(
                    event, "hbm_admission_wait_ns", "resume event"),
                "agentic_kv_source": str(event.get("source")),
                "return_gap_type": str(
                    event.get("return_gap_type") or "unknown"),
            }
            for field, observed in comparisons.items():
                expected = (
                    str(record.get(field) or "unknown")
                    if field in {"agentic_kv_source", "return_gap_type"}
                    else int(record[field])
                )
                if observed != expected:
                    raise ExperimentError(
                        f"Resume event/request mismatch for {run_id}, "
                        f"request={request_id}: {field} event={observed!r}, "
                        f"record={expected!r}")

    resume_events = [
        event for event in events if event.get("event") == "resume"
    ]
    global_source_counts = {source: 0 for source in ("hbm", "cpu", "ssd")}
    global_hit_tokens = 0
    global_recompute_tokens = 0
    for index, event in enumerate(resume_events):
        label = f"resume[{index}]"
        hit_tokens = event_int(event, "hit_tokens", label)
        recompute_tokens = event_int(event, "recompute_tokens", label)
        source = str(event.get("source"))
        if source not in {"hbm", "cpu", "ssd", "dropped"}:
            raise ExperimentError(
                f"Resume event has an invalid source for {run_id}: "
                f"{label}, source={source!r}")
        if hit_tokens > 0:
            if source not in global_source_counts:
                raise ExperimentError(
                    f"Positive-hit resume has no physical source for "
                    f"{run_id}: {label}, source={source!r}")
            global_source_counts[source] += 1
        global_hit_tokens += hit_tokens
        global_recompute_tokens += recompute_tokens
    global_expected = {
        "hbm_hits": global_source_counts["hbm"],
        "cpu_hits": global_source_counts["cpu"],
        "ssd_hits": global_source_counts["ssd"],
        "cache_hit_tokens": global_hit_tokens,
        "recompute_tokens": global_recompute_tokens,
    }
    for field, expected in global_expected.items():
        value = totals.get(field)
        if (not isinstance(value, int) or isinstance(value, bool)
                or value != expected):
            raise ExperimentError(
                f"Resume event/manager aggregate mismatch for {run_id}: "
                f"{field}={value!r}, events={expected}")

    return {
        "performed": True,
        "passed": True,
        "measured_request_count": len(records_by_id),
        "measured_resume_event_count": sum(
            len(rows) for rows in selected_events["resume"].values()),
        "global_resume_event_count": len(resume_events),
        "global_physical_resume_counts": global_source_counts,
        "global_hit_tokens": global_hit_tokens,
        "global_recompute_tokens": global_recompute_tokens,
        "full_completed_cohort": full_cohort,
        "unmatched_scoped_event_count": unmatched_scoped_event_count,
        "scope": (
            "per-request events are joined only to measured-session records; "
            "global resume event sums reconcile to manager totals"),
    }


def _derive_exact_rate_metrics(report, agentic_report, manifest):
    """Derive exact per-run latency and server-added-JCT metrics.

    Request percentiles use completed LLM calls in the measured session
    cohort. Resume TTFT includes every non-initial call. TPOT includes only
    calls with at least two generated tokens, for which a per-output-token
    interval is defined. Server-added JCT subtracts only the trace-declared
    closed-loop human/tool gaps from offer-to-final-completion JCT.
    """
    records = report.get("requests", {}).get("records")
    if not isinstance(records, list):
        raise ExperimentError(
            f"Missing exact request records for {manifest['run_id']}")
    measured_session_ids = set(_measured_session_ids(report))
    full_completed_cohort = _full_completed_cohort_scope(
        report, manifest, measured_session_ids)
    if full_completed_cohort:
        outside_json_records = [
            record for record in records
            if str(record.get("session_id")) not in measured_session_ids
        ]
        if outside_json_records:
            raise ExperimentError(
                f"Full-cohort session JSON for {manifest['run_id']} has "
                f"out-of-cohort request rows: {outside_json_records[:3]}")
        throughput = report.get("throughput") or {}
        expected_full_counts = {
            "completed_requests_total": len(records),
            "completed_sessions_total": len(measured_session_ids),
        }
        for field, expected in expected_full_counts.items():
            observed = throughput.get(field)
            if (not isinstance(observed, int) or isinstance(observed, bool)
                    or observed != expected):
                raise ExperimentError(
                    f"Full-cohort completion total mismatch for "
                    f"{manifest['run_id']}: {field}={observed!r}, "
                    f"expected={expected}")
    exact_records = [
        record for record in records
        if str(record.get("session_id")) in measured_session_ids
    ]
    current_schema = int(manifest.get("schema_version", 0) or 0)
    if current_schema >= SCHEMA_VERSION and not exact_records:
        raise ExperimentError(
            f"Current run {manifest['run_id']} has no measured request "
            "records")
    current_exact_fields = (
        "request_id", "session_id", "sub_request_index",
        "agentic_kv_hit_tokens", "agentic_kv_recompute_tokens",
        "agentic_kv_source", "return_gap_type",
        "agentic_kv_hbm_admission_wait_ns",
        "pd_chunk_admission_count",
        "pd_chunk_cancelled_admission_count",
        "pd_chunk_admission_wait_ns_total",
        "pd_chunk_admission_critical_wait_ns_total",
        "pd_chunk_successful_admission_wait_ns_total",
        "pd_chunk_successful_admission_critical_wait_ns_total",
        "pd_chunk_cancelled_admission_wait_ns_total",
        "pd_chunk_cancelled_admission_critical_wait_ns_total",
        "active_prefill_recompute_preemptions",
        "active_prefill_recompute_tokens",
        "active_prefill_recompute_frontier_tokens",
        "pd_active_prefill_recompute_generation",
        "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute",
    )
    if current_schema >= SCHEMA_VERSION:
        exact_integer_fields = tuple(
            field for field in current_exact_fields
            if field not in {
                "session_id", "agentic_kv_source", "return_gap_type"
            }
        )
        for record in exact_records:
            missing = [
                field for field in current_exact_fields
                if field not in record
            ]
            if missing:
                raise ExperimentError(
                    f"Current request record for {manifest['run_id']} is "
                    f"missing exact fields: request="
                    f"{record.get('request_id')}, missing={missing}")
            invalid_integers = {
                field: record[field]
                for field in exact_integer_fields
                if (not isinstance(record[field], int)
                    or isinstance(record[field], bool)
                    or record[field] < 0)
            }
            if invalid_integers:
                raise ExperimentError(
                    f"Current request record for {manifest['run_id']} has "
                    f"invalid exact integer fields: request="
                    f"{record.get('request_id')}, values={invalid_integers}")
            if (not isinstance(record["session_id"], str)
                    or not record["session_id"]):
                raise ExperimentError(
                    f"Current request record for {manifest['run_id']} has "
                    f"an invalid session_id: {record['session_id']!r}")
            for field in ("agentic_kv_source", "return_gap_type"):
                if (record[field] is not None
                        and not isinstance(record[field], str)):
                    raise ExperimentError(
                        f"Current request record for {manifest['run_id']} "
                        f"has invalid {field}: {record[field]!r}")
        exact_ids = [int(record["request_id"]) for record in exact_records]
        if len(exact_ids) != len(set(exact_ids)):
            raise ExperimentError(
                f"Duplicate exact request IDs for {manifest['run_id']}")
    cross_layer_accounting = _cross_layer_request_accounting_audit(
        exact_records, measured_session_ids, report, agentic_report,
        manifest, full_completed_cohort=full_completed_cohort)

    resume_records = [
        record for record in exact_records
        if int(record.get("sub_request_index", -1)) > 0
    ]
    expected_resume_count = int(
        report.get("requests", {}).get("resume", {}).get("count", 0))
    if exact_records and len(resume_records) != expected_resume_count:
        raise ExperimentError(
            f"Resume TTFT denominator mismatch for {manifest['run_id']}: "
            f"records={len(resume_records)}, report={expected_resume_count}")

    physical_sources = ("hbm", "cpu", "ssd")
    attempted_counts = {source: 0 for source in physical_sources}
    effective_counts = {source: 0 for source in physical_sources}
    attempted_by_gap_source = {}
    effective_by_gap_source = {}
    attempted_hit_tokens = 0
    restored_hit_tokens_discarded = 0
    kv_state_unavailable_count = 0
    zero_overlap_count = 0
    for record in resume_records:
        source = str(record.get("agentic_kv_source") or "unknown")
        hit_tokens = int(record.get("agentic_kv_hit_tokens", 0))
        discarded = int(record.get(
            "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute",
            0,
        ))
        recompute_tokens = int(record.get(
            "agentic_kv_recompute_tokens", 0))
        if source == "dropped" and recompute_tokens > 0:
            kv_state_unavailable_count += 1
        if hit_tokens == 0 and recompute_tokens == 0:
            zero_overlap_count += 1
        if hit_tokens < 0 or not 0 <= discarded <= hit_tokens:
            raise ExperimentError(
                f"Invalid attempted/surviving resume tokens for "
                f"{manifest['run_id']}, request={record.get('request_id')}: "
                f"hit={hit_tokens}, discarded={discarded}")
        if hit_tokens <= 0:
            if discarded:
                raise ExperimentError(
                    f"Restored-hit discard without attempted hit for "
                    f"{manifest['run_id']}, request="
                    f"{record.get('request_id')}")
            continue
        if source not in physical_sources:
            raise ExperimentError(
                f"Attempted restored hit lacks physical source for "
                f"{manifest['run_id']}, request={record.get('request_id')}: "
                f"source={source!r}")
        gap_type = str(record.get("return_gap_type") or "unknown")
        attempted_counts[source] += 1
        attempted_by_gap_source.setdefault(gap_type, {}).setdefault(
            source, 0)
        attempted_by_gap_source[gap_type][source] += 1
        attempted_hit_tokens += hit_tokens
        restored_hit_tokens_discarded += discarded
        if hit_tokens - discarded > 0:
            effective_counts[source] += 1
            effective_by_gap_source.setdefault(gap_type, {}).setdefault(
                source, 0)
            effective_by_gap_source[gap_type][source] += 1

    effective_hit_tokens = (
        attempted_hit_tokens - restored_hit_tokens_discarded)
    attempted_count = sum(attempted_counts.values())
    effective_count = sum(effective_counts.values())
    request_count = len(exact_records)
    source_accounting = {
        "attempted_counts_by_source": attempted_counts,
        "effective_surviving_counts_by_source": effective_counts,
        "attempted_count": attempted_count,
        "effective_surviving_count": effective_count,
        "attempted_fractions_of_all_requests": {
            source: attempted_counts[source] / request_count
            if request_count else None
            for source in physical_sources
        },
        "effective_surviving_fractions_of_all_requests": {
            source: effective_counts[source] / request_count
            if request_count else None
            for source in physical_sources
        },
        "attempted_fraction_of_all_requests": (
            attempted_count / request_count if request_count else None),
        "effective_surviving_fraction_of_all_requests": (
            effective_count / request_count if request_count else None),
        "attempted_restored_hit_tokens": attempted_hit_tokens,
        "restored_hit_tokens_discarded_by_active_prefill_recompute": (
            restored_hit_tokens_discarded),
        "effective_surviving_hit_tokens": effective_hit_tokens,
        "attempted_by_return_gap_type_and_source": (
            attempted_by_gap_source),
        "effective_surviving_by_return_gap_type_and_source": (
            effective_by_gap_source),
        "all_request_denominator": request_count,
        "kv_state_unavailable_resume_count": (
            kv_state_unavailable_count),
        "zero_overlap_resume_count": zero_overlap_count,
        "source_semantics": (
            "physical HBM/CPU/SSD resume provenance is retained even when "
            "active-prefill recomputation later discards restored tokens"),
    }

    requests_report = report.get("requests", {})
    if current_schema >= SCHEMA_VERSION:
        expected_scalars = {
            "attempted_physical_resume_count": attempted_count,
            "effective_surviving_resume_count": effective_count,
        }
        for field, expected in expected_scalars.items():
            if requests_report.get(field) != expected:
                raise ExperimentError(
                    f"Exact resume accounting does not reconcile for "
                    f"{manifest['run_id']}: {field}="
                    f"{requests_report.get(field)!r}, expected={expected}")
        expected_maps = {
            "attempted_physical_resume_counts_by_source": attempted_counts,
            "effective_surviving_resume_counts_by_source": effective_counts,
        }
        for field, expected in expected_maps.items():
            if requests_report.get(field) != expected:
                raise ExperimentError(
                    f"Exact resume source accounting does not reconcile for "
                    f"{manifest['run_id']}: {field}="
                    f"{requests_report.get(field)!r}, expected={expected}")
        expected_fraction_maps = {
            "attempted_physical_resume_fractions_of_all_requests": (
                source_accounting[
                    "attempted_fractions_of_all_requests"]),
            "effective_surviving_resume_fractions_of_all_requests": (
                source_accounting[
                    "effective_surviving_fractions_of_all_requests"]),
        }
        for field, expected in expected_fraction_maps.items():
            if requests_report.get(field) != expected:
                raise ExperimentError(
                    f"Exact all-request resume fractions do not reconcile "
                    f"for {manifest['run_id']}: {field}="
                    f"{requests_report.get(field)!r}, expected={expected}")

        def reported_cross_counts(field):
            cross = requests_report.get(field)
            if not isinstance(cross, dict):
                raise ExperimentError(
                    f"Missing {field} for {manifest['run_id']}")
            return {
                str(gap): {
                    str(source): int((group or {}).get("count", 0))
                    for source, group in (sources or {}).items()
                    if int((group or {}).get("count", 0)) > 0
                }
                for gap, sources in cross.items()
                if any(
                    int((group or {}).get("count", 0)) > 0
                    for group in (sources or {}).values())
            }

        cross_expectations = {
            "attempted_physical_resume_by_return_gap_type_and_source": (
                attempted_by_gap_source),
            "effective_surviving_resume_by_return_gap_type_and_source": (
                effective_by_gap_source),
        }
        for field, expected in cross_expectations.items():
            if reported_cross_counts(field) != expected:
                raise ExperimentError(
                    f"Exact human/tool resume cross-group does not "
                    f"reconcile for {manifest['run_id']}: {field}")
        reported_tokens = requests_report.get(
            "resume_reuse_token_accounting") or {}
        expected_tokens = {
            "attempted_restored_hit_tokens": attempted_hit_tokens,
            "restored_hit_tokens_discarded_by_active_prefill_recompute": (
                restored_hit_tokens_discarded),
            "effective_surviving_hit_tokens": effective_hit_tokens,
            "conservation_passed": True,
        }
        if reported_tokens != expected_tokens:
            raise ExperimentError(
                f"Exact resume token accounting does not reconcile for "
                f"{manifest['run_id']}: reported={reported_tokens}, "
                f"expected={expected_tokens}")
        expected_special_counts = {
            "kv_state_unavailable_resume_count": (
                kv_state_unavailable_count),
            "zero_overlap_resume_count": zero_overlap_count,
        }
        for field, expected in expected_special_counts.items():
            if requests_report.get(field) != expected:
                raise ExperimentError(
                    f"Exact zero-overlap/unavailable accounting does not "
                    f"reconcile for {manifest['run_id']}: {field}="
                    f"{requests_report.get(field)!r}, expected={expected}")
    resume_ttft = _exact_distribution(
        (record.get("ttft_ns") for record in resume_records),
        name=f"{manifest['run_id']} resume TTFT",
    )
    reported_resume_ttft = report.get(
        "requests", {}).get("resume", {}).get("ttft_ns", {})
    if resume_records:
        reported_mean = reported_resume_ttft.get("mean")
        if (reported_mean is None
                or not math.isclose(
                    float(reported_mean), float(resume_ttft["mean"]),
                    rel_tol=1e-12, abs_tol=1e-6)):
            raise ExperimentError(
                f"Exact resume TTFT does not reconcile with report for "
                f"{manifest['run_id']}: exact={resume_ttft['mean']}, "
                f"reported={reported_mean}")

    tpot_records = [
        record for record in exact_records
        if int(record.get("generated_tokens", 0)) >= 2
    ]
    tpot = _exact_distribution(
        (record.get("tpot_ns") for record in tpot_records),
        name=f"{manifest['run_id']} TPOT",
    )

    requests_by_session = {}
    for record in exact_records:
        requests_by_session.setdefault(
            str(record.get("session_id")), []).append(record)
    session_records = {
        str(record.get("session_id")): record
        for record in report.get("sessions", {}).get("records", [])
        if record.get("measurement_included")
    }
    if exact_records and set(session_records) != measured_session_ids:
        raise ExperimentError(
            f"Measured session record set mismatch for {manifest['run_id']}")
    server_added_values = []
    trace_gap_values = []
    for session_id in sorted(measured_session_ids):
        session = session_records.get(session_id)
        calls = requests_by_session.get(session_id, [])
        if session is None or not calls:
            if current_schema >= SCHEMA_VERSION:
                raise ExperimentError(
                    f"Missing exact server-added JCT inputs for "
                    f"{manifest['run_id']}, session={session_id}")
            continue
        required = ("offered_time_ns", "completion_time_ns")
        if any(session.get(field) is None for field in required):
            if current_schema >= SCHEMA_VERSION:
                raise ExperimentError(
                    f"Missing session offer/completion timestamp for "
                    f"{manifest['run_id']}, session={session_id}")
            continue
        total_jct_ns = (
            int(session["completion_time_ns"])
            - int(session["offered_time_ns"])
        )
        trace_gap_ns = sum(int(call.get("return_gap_ns", 0)) for call in calls)
        if total_jct_ns < 0 or trace_gap_ns < 0 or trace_gap_ns > total_jct_ns:
            raise ExperimentError(
                f"Server-added JCT does not reconcile for "
                f"{manifest['run_id']}, session={session_id}: "
                f"total={total_jct_ns}, trace_gaps={trace_gap_ns}")
        trace_gap_values.append(trace_gap_ns)
        server_added_values.append(total_jct_ns - trace_gap_ns)
    server_added = _exact_distribution(
        server_added_values,
        name=f"{manifest['run_id']} server-added session JCT",
    )
    trace_gaps = _exact_distribution(
        trace_gap_values,
        name=f"{manifest['run_id']} trace closed-loop idle gaps",
    )

    csv_rows = _measured_request_csv_rows(
        manifest,
        measured_session_ids,
        full_completed_cohort=full_completed_cohort,
    )
    if csv_rows is None:
        hbm_admission = {
            "count": 0, "sum": 0, "mean": None, "p95": None,
        }
        hbm_admission_scope = "unavailable_legacy_request_csv"
    else:
        records_by_id = {
            int(record["request_id"]): record for record in exact_records
        }
        record_ids = set(records_by_id)
        csv_ids = {int(row["request id"]) for row in csv_rows}
        if csv_ids != record_ids or len(csv_ids) != len(csv_rows):
            raise ExperimentError(
                f"Request CSV identity mismatch for {manifest['run_id']}: "
                f"json_only={sorted(record_ids-csv_ids)[:5]}, "
                f"csv_only={sorted(csv_ids-record_ids)[:5]}")
        restore_hbm_values = []
        pd_chunk_attempt_wait_values = []
        pd_chunk_successful_values = []
        pd_chunk_cancelled_values = []
        pd_chunk_gross_critical_values = []
        pd_chunk_successful_critical_values = []
        pd_chunk_cancelled_critical_values = []
        for row in csv_rows:
            request_id = int(row["request id"])
            record = records_by_id[request_id]
            restore_wait = int(row["agentic_kv_hbm_admission_wait_ns"])
            pd_chunk_wait = int(row["pd_chunk_admission_wait_ns_total"])
            successful_wait = int(
                row["pd_chunk_successful_admission_wait_ns_total"])
            cancelled_wait = int(
                row["pd_chunk_cancelled_admission_wait_ns_total"])
            gross_critical = int(
                row["pd_chunk_admission_critical_wait_ns_total"])
            successful_critical = int(row[
                "pd_chunk_successful_admission_critical_wait_ns_total"])
            cancelled_critical = int(row[
                "pd_chunk_cancelled_admission_critical_wait_ns_total"])
            integer_fields = (
                "agentic_kv_hit_tokens",
                "agentic_kv_recompute_tokens",
                "agentic_kv_hbm_admission_wait_ns",
                "pd_chunk_admission_count",
                "pd_chunk_cancelled_admission_count",
                "pd_chunk_admission_wait_ns_total",
                "pd_chunk_admission_critical_wait_ns_total",
                "pd_chunk_successful_admission_wait_ns_total",
                "pd_chunk_successful_admission_critical_wait_ns_total",
                "pd_chunk_cancelled_admission_wait_ns_total",
                "pd_chunk_cancelled_admission_critical_wait_ns_total",
                "active_prefill_recompute_preemptions",
                "active_prefill_recompute_tokens",
                "active_prefill_recompute_frontier_tokens",
                "pd_active_prefill_recompute_generation",
                "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute",
            )
            csv_values = {field: int(row[field]) for field in integer_fields}
            json_values = {
                field: int(record.get(field, -1))
                for field in integer_fields
            }
            if csv_values != json_values:
                mismatches = {
                    field: {
                        "csv": csv_values[field],
                        "json": json_values[field],
                    }
                    for field in integer_fields
                    if csv_values[field] != json_values[field]
                }
                raise ExperimentError(
                    f"Request CSV/JSON exact accounting mismatch for "
                    f"{manifest['run_id']}, request={request_id}: "
                    f"{mismatches}")
            csv_source = str(row.get("agentic_kv_source") or "")
            json_source = str(record.get("agentic_kv_source") or "")
            if csv_source != json_source:
                raise ExperimentError(
                    f"Request CSV/JSON physical source mismatch for "
                    f"{manifest['run_id']}, request={request_id}: "
                    f"csv={csv_source!r}, json={json_source!r}")
            csv_session_id = str(row.get("session_id"))
            json_session_id = str(record.get("session_id"))
            if csv_session_id != json_session_id:
                raise ExperimentError(
                    f"Request CSV/JSON session mismatch for "
                    f"{manifest['run_id']}, request={request_id}: "
                    f"csv={csv_session_id!r}, json={json_session_id!r}")
            csv_gap_type = str(row.get("return_gap_type") or "")
            json_gap_type = str(record.get("return_gap_type") or "")
            if csv_gap_type != json_gap_type:
                raise ExperimentError(
                    f"Request CSV/JSON return-gap type mismatch for "
                    f"{manifest['run_id']}, request={request_id}: "
                    f"csv={csv_gap_type!r}, json={json_gap_type!r}")
            if any(value < 0 for value in csv_values.values()):
                raise ExperimentError(
                    f"Negative exact request accounting component for "
                    f"{manifest['run_id']}, request={request_id}: "
                    f"{csv_values}")
            if successful_wait + cancelled_wait != pd_chunk_wait:
                raise ExperimentError(
                    f"P/D chunk wait partition mismatch for "
                    f"{manifest['run_id']}, request={request_id}: "
                    f"success={successful_wait}, cancel={cancelled_wait}, "
                    f"gross={pd_chunk_wait}")
            if successful_critical + cancelled_critical != gross_critical:
                raise ExperimentError(
                    f"P/D chunk critical-wait partition mismatch for "
                    f"{manifest['run_id']}, request={request_id}")
            if (csv_values["active_prefill_recompute_preemptions"]
                    != csv_values[
                        "pd_active_prefill_recompute_generation"]):
                raise ExperimentError(
                    f"Active-prefill count/generation mismatch for "
                    f"{manifest['run_id']}, request={request_id}")
            if (csv_values[
                    "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute"]
                    > csv_values["agentic_kv_hit_tokens"]):
                raise ExperimentError(
                    f"Restored-hit discard exceeds attempted hit for "
                    f"{manifest['run_id']}, request={request_id}")
            if (csv_values["active_prefill_recompute_tokens"]
                    < csv_values[
                        "active_prefill_recompute_frontier_tokens"]):
                raise ExperimentError(
                    f"Active-prefill cumulative tokens do not cover frontier "
                    f"for {manifest['run_id']}, request={request_id}")
            restore_hbm_values.append(restore_wait)
            pd_chunk_attempt_wait_values.append(pd_chunk_wait)
            pd_chunk_successful_values.append(successful_wait)
            pd_chunk_cancelled_values.append(cancelled_wait)
            pd_chunk_gross_critical_values.append(gross_critical)
            pd_chunk_successful_critical_values.append(successful_critical)
            pd_chunk_cancelled_critical_values.append(cancelled_critical)
        restore_hbm_admission = _exact_distribution(
            restore_hbm_values,
            name=f"{manifest['run_id']} restore HBM admission wait",
        )
        pd_chunk_attempt_admission = _exact_distribution(
            pd_chunk_attempt_wait_values,
            name=f"{manifest['run_id']} gross P/D chunk attempt wait",
        )
        pd_chunk_successful_admission = _exact_distribution(
            pd_chunk_successful_values,
            name=f"{manifest['run_id']} successful P/D chunk admission wait",
        )
        pd_chunk_cancelled_admission = _exact_distribution(
            pd_chunk_cancelled_values,
            name=f"{manifest['run_id']} cancelled P/D chunk admission wait",
        )
        pd_chunk_gross_critical = _exact_distribution(
            pd_chunk_gross_critical_values,
            name=f"{manifest['run_id']} gross P/D chunk critical wait",
        )
        # The canonical HBM-capacity component is the post-restore critical
        # tail.  Gross enqueue-to-admission wall time can overlap the restore
        # destination gate and therefore must not be added to it.
        pd_chunk_hbm_admission = dict(pd_chunk_gross_critical)
        pd_chunk_successful_critical = _exact_distribution(
            pd_chunk_successful_critical_values,
            name=f"{manifest['run_id']} successful P/D chunk critical wait",
        )
        pd_chunk_cancelled_critical = _exact_distribution(
            pd_chunk_cancelled_critical_values,
            name=f"{manifest['run_id']} cancelled P/D chunk critical wait",
        )
        hbm_admission = _exact_distribution(
            (
                restore_wait + pd_chunk_wait
                for restore_wait, pd_chunk_wait in zip(
                    restore_hbm_values, pd_chunk_gross_critical_values)
            ),
            name=f"{manifest['run_id']} total HBM admission wait",
        )
        hbm_admission_scope = (
            "request_critical_hbm_capacity_wait; restore_destination_plus_"
            "gross_successful_and_cancelled_pd_chunk_post_restore_critical_"
            "wait; "
            "excludes_pd_pair_fifo_restore_service_and_transfer_queue"
        )

    if csv_rows is None:
        restore_hbm_admission = {
            "count": 0, "sum": 0, "mean": None, "p95": None,
        }
        pd_chunk_hbm_admission = {
            "count": 0, "sum": 0, "mean": None, "p95": None,
        }
        pd_chunk_attempt_admission = dict(pd_chunk_hbm_admission)
        pd_chunk_successful_admission = dict(pd_chunk_hbm_admission)
        pd_chunk_cancelled_admission = dict(pd_chunk_hbm_admission)
        pd_chunk_gross_critical = dict(pd_chunk_hbm_admission)
        pd_chunk_successful_critical = dict(pd_chunk_hbm_admission)
        pd_chunk_cancelled_critical = dict(pd_chunk_hbm_admission)

    return {
        "resume_ttft": resume_ttft,
        "resume_ttft_denominator": (
            "all completed non-initial LLM calls in measured sessions"
        ),
        "tpot": tpot,
        "tpot_denominator": (
            "completed LLM calls with at least two generated tokens in "
            "measured sessions"
        ),
        "server_added_jct": server_added,
        "server_added_jct_denominator": "all measured completed sessions",
        "server_added_jct_definition": (
            "offer_to_final_completion_minus_sum_of_trace_declared_"
            "closed_loop_human_tool_gaps"
        ),
        "trace_idle_gaps": trace_gaps,
        "hbm_admission": hbm_admission,
        "restore_hbm_admission": restore_hbm_admission,
        "pd_chunk_hbm_admission": pd_chunk_hbm_admission,
        "pd_chunk_attempt_admission": pd_chunk_attempt_admission,
        "pd_chunk_successful_admission": pd_chunk_successful_admission,
        "pd_chunk_cancelled_admission": pd_chunk_cancelled_admission,
        "pd_chunk_gross_critical_wait": pd_chunk_gross_critical,
        "pd_chunk_successful_critical_wait": pd_chunk_successful_critical,
        "pd_chunk_cancelled_critical_wait": pd_chunk_cancelled_critical,
        "resume_source_accounting": source_accounting,
        "cross_layer_request_accounting": cross_layer_accounting,
        "hbm_admission_scope": hbm_admission_scope,
    }


def _operational_metric_sources(manifest, report):
    """Validate and flatten current HBM/batch measurement sources."""
    current_schema = int(manifest.get("schema_version", 0) or 0)
    report_schema = int(report.get("schema_version", 0) or 0)
    if (current_schema >= SCHEMA_VERSION
            and report_schema < MIN_CURRENT_SESSION_REPORT_SCHEMA_VERSION):
        raise ExperimentError(
            f"Online artifact schema {current_schema} requires session "
            f"report schema {MIN_CURRENT_SESSION_REPORT_SCHEMA_VERSION} "
            "operational sources for "
            f"{manifest['run_id']}; observed={report_schema}")
    if current_schema < SCHEMA_VERSION:
        return {
            "source_status": "unavailable_legacy_session_report",
            "average_active_batch_size": None,
            "average_active_batch_size_including_dummy": None,
            "active_batch_completed_count": None,
            "active_batch_dummy_count": None,
            "active_batch_scope": None,
            "hbm_occupancy": None,
        }

    run_id = manifest["run_id"]
    window = report.get("measurement_window") or {}
    occupancy = report.get("hbm_kv_occupancy")
    if not isinstance(occupancy, dict):
        raise ExperimentError(
            f"Current session report is missing hbm_kv_occupancy for "
            f"{run_id}")
    if occupancy.get("schema_version") != 1:
        raise ExperimentError(
            f"Unsupported HBM occupancy schema for {run_id}: "
            f"{occupancy.get('schema_version')!r}")
    if occupancy.get("units") != "per_rank_bytes":
        raise ExperimentError(
            f"HBM occupancy units must be per_rank_bytes for {run_id}")
    expected_bounds = (
        window.get("measurement_start_ns"),
        window.get("measurement_end_ns"),
        window.get("measurement_duration_ns"),
    )
    observed_bounds = (
        occupancy.get("window_start_ns"),
        occupancy.get("window_end_ns"),
        occupancy.get("window_duration_ns"),
    )
    if observed_bounds != expected_bounds:
        raise ExperimentError(
            f"HBM occupancy window mismatch for {run_id}: "
            f"observed={observed_bounds}, expected={expected_bounds}")
    coverage = occupancy.get("coverage") or {}
    if coverage.get("covers_window") is not True:
        raise ExperimentError(
            f"HBM occupancy does not cover measurement window for {run_id}")
    if not (occupancy.get("conservation") or {}).get("passed", False):
        raise ExperimentError(
            f"HBM occupancy conservation failed for {run_id}")
    physical_categories = (
        "physical_idle_reusable",
        "physical_non_idle_active",
        "physical_free",
    )
    overlay_categories = (
        "logical_destination_admission_reservation",
        "reserved_free_slack",
        "future_reclaim_backed_reservation",
        "unclaimed_allocatable_slack",
    )
    if tuple(occupancy.get("physical_capacity_breakdown") or ()) != (
            physical_categories):
        raise ExperimentError(
            f"HBM physical category contract mismatch for {run_id}")
    if tuple(occupancy.get("logical_reservation_overlay") or ()) != (
            overlay_categories):
        raise ExperimentError(
            f"HBM reservation-overlay contract mismatch for {run_id}")

    aggregate = occupancy.get("aggregate") or {}
    capacity = aggregate.get("capacity_per_rank_bytes_sum")
    if (not isinstance(capacity, int) or isinstance(capacity, bool)
            or capacity <= 0):
        raise ExperimentError(
            f"Invalid aggregate HBM KV capacity for {run_id}: {capacity!r}")
    category_reports = aggregate.get("categories") or {}
    required_categories = physical_categories + overlay_categories
    values = {}
    for category in required_categories:
        category_report = category_reports.get(category)
        if not isinstance(category_report, dict):
            raise ExperimentError(
                f"Missing HBM occupancy category {category!r} for {run_id}")
        for field in (
                "byte_ns", "average_per_rank_bytes",
                "peak_per_rank_bytes", "average_fraction_of_capacity",
                "peak_fraction_of_capacity"):
            value = category_report.get(field)
            if (not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value)) or value < 0):
                raise ExperimentError(
                    f"Invalid HBM {category}.{field} for {run_id}: "
                    f"{value!r}")
        average = float(category_report["average_per_rank_bytes"])
        fraction = float(category_report["average_fraction_of_capacity"])
        if not math.isclose(
                average / capacity, fraction,
                rel_tol=1e-9, abs_tol=1e-12):
            raise ExperimentError(
                f"HBM average fraction does not reconcile for {run_id}, "
                f"category={category}")
        values[category] = average
    tolerance = max(1e-6, capacity * 1e-9)
    if abs(sum(values[name] for name in physical_categories) - capacity) > (
            tolerance):
        raise ExperimentError(
            f"HBM physical capacity identity failed for {run_id}")
    if abs(
            values["reserved_free_slack"]
            + values["future_reclaim_backed_reservation"]
            - values["logical_destination_admission_reservation"]
    ) > tolerance:
        raise ExperimentError(
            f"HBM logical overlay identity failed for {run_id}")
    if abs(
            values["physical_idle_reusable"]
            + values["physical_non_idle_active"]
            + values["reserved_free_slack"]
            + values["unclaimed_allocatable_slack"]
            - capacity
    ) > tolerance:
        raise ExperimentError(
            f"HBM reservation-adjusted capacity identity failed for {run_id}")
    expected_physical_occupied = (
        values["physical_idle_reusable"]
        + values["physical_non_idle_active"]
    )
    if not math.isclose(
            float(aggregate.get(
                "average_physical_occupied_per_rank_bytes", -1)),
            expected_physical_occupied,
            rel_tol=1e-9, abs_tol=tolerance):
        raise ExperimentError(
            f"HBM physical occupied aggregate does not reconcile for {run_id}")
    per_instance = occupancy.get("per_instance")
    if not isinstance(per_instance, dict) or not per_instance:
        raise ExperimentError(
            f"HBM occupancy per_instance breakdown is missing for {run_id}")
    per_instance_capacities = []
    per_instance_average_sums = {
        category: 0.0 for category in required_categories}
    for instance_id, instance_report in per_instance.items():
        if not isinstance(instance_report, dict):
            raise ExperimentError(
                f"Invalid per-instance HBM report for {run_id}, "
                f"instance={instance_id}")
        instance_capacity = instance_report.get("capacity_per_rank_bytes")
        if (not isinstance(instance_capacity, int)
                or isinstance(instance_capacity, bool)
                or instance_capacity <= 0):
            raise ExperimentError(
                f"Invalid per-instance HBM capacity for {run_id}, "
                f"instance={instance_id}: {instance_capacity!r}")
        per_instance_capacities.append(instance_capacity)
        instance_categories = instance_report.get("categories") or {}
        for category in required_categories:
            category_report = instance_categories.get(category) or {}
            average = category_report.get("average_per_rank_bytes")
            if (not isinstance(average, (int, float))
                    or isinstance(average, bool)
                    or not math.isfinite(float(average)) or average < 0):
                raise ExperimentError(
                    f"Invalid per-instance HBM average for {run_id}, "
                    f"instance={instance_id}, category={category}")
            per_instance_average_sums[category] += float(average)
    if sum(per_instance_capacities) != capacity:
        raise ExperimentError(
            f"HBM per-instance capacities do not reconcile for {run_id}")
    for category in required_categories:
        if not math.isclose(
                per_instance_average_sums[category], values[category],
                rel_tol=1e-9, abs_tol=tolerance):
            raise ExperimentError(
                f"HBM per-instance {category} averages do not reconcile for "
                f"{run_id}")

    for byte_field, fraction_field in (
            (
                "average_physical_occupied_per_rank_bytes",
                "average_physical_occupied_utilization_fraction",
            ),
            (
                "average_reservation_adjusted_claim_per_rank_bytes",
                "average_reservation_adjusted_claim_fraction",
            )):
        byte_value = aggregate.get(byte_field)
        fraction_value = aggregate.get(fraction_field)
        if (not isinstance(byte_value, (int, float))
                or isinstance(byte_value, bool)
                or not isinstance(fraction_value, (int, float))
                or isinstance(fraction_value, bool)
                or not math.isfinite(float(byte_value))
                or not math.isfinite(float(fraction_value))
                or byte_value < 0
                or not 0 <= fraction_value <= 1
                or not math.isclose(
                    float(byte_value) / capacity, float(fraction_value),
                    rel_tol=1e-9, abs_tol=1e-12)):
            raise ExperimentError(
                f"HBM aggregate {byte_field}/{fraction_field} does not "
                f"reconcile for {run_id}")

    compute = report.get("online_model_compute") or {}
    batch = compute.get("real_batch_size")
    if not isinstance(batch, dict):
        raise ExperimentError(
            f"Current session report is missing real_batch_size for {run_id}")
    integer_fields = (
        "completed_batch_count", "non_dummy_completed_batch_count",
        "dp_dummy_completed_batch_count", "total_real_request_memberships",
    )
    for field in integer_fields:
        value = batch.get(field)
        if (not isinstance(value, int) or isinstance(value, bool)
                or value < 0):
            raise ExperimentError(
                f"Invalid real_batch_size.{field} for {run_id}: {value!r}")
    completed = batch["completed_batch_count"]
    non_dummy = batch["non_dummy_completed_batch_count"]
    dummy = batch["dp_dummy_completed_batch_count"]
    memberships = batch["total_real_request_memberships"]
    if completed <= 0 or non_dummy <= 0:
        raise ExperimentError(
            f"Measured run has no non-dummy completed batches for {run_id}")
    if non_dummy + dummy != completed or memberships < non_dummy:
        raise ExperimentError(
            f"Real batch-size counters do not reconcile for {run_id}")
    mean_non_dummy = batch.get("mean_real_requests_per_non_dummy_batch")
    mean_including_dummy = batch.get(
        "mean_real_requests_per_completed_batch_including_dummy")
    if (not isinstance(mean_non_dummy, (int, float))
            or isinstance(mean_non_dummy, bool)
            or not math.isclose(
                float(mean_non_dummy), memberships / non_dummy,
                rel_tol=1e-12, abs_tol=1e-12)):
        raise ExperimentError(
            f"Non-dummy mean batch size does not reconcile for {run_id}")
    if (not isinstance(mean_including_dummy, (int, float))
            or isinstance(mean_including_dummy, bool)
            or not math.isclose(
                float(mean_including_dummy), memberships / completed,
                rel_tol=1e-12, abs_tol=1e-12)):
        raise ExperimentError(
            f"Dummy-inclusive mean batch size does not reconcile for {run_id}")
    by_pd_type = batch.get("by_pd_type")
    if not isinstance(by_pd_type, dict) or not by_pd_type:
        raise ExperimentError(
            f"Real batch-size by_pd_type breakdown is missing for {run_id}")
    for field in integer_fields:
        if sum(int(value.get(field, -1)) for value in by_pd_type.values()) != (
                batch[field]):
            raise ExperimentError(
                f"Real batch-size by_pd_type {field} does not reconcile for "
                f"{run_id}")

    return {
        "source_status": "schema11_exact_measurement_window",
        "average_active_batch_size": float(mean_non_dummy),
        "average_active_batch_size_including_dummy": float(
            mean_including_dummy),
        "active_batch_completed_count": completed,
        "active_batch_dummy_count": dummy,
        "active_batch_scope": batch.get("membership_semantics"),
        "hbm_occupancy": {
            "capacity_per_rank_bytes_sum": capacity,
            "categories": values,
            "category_reports": category_reports,
            "average_physical_occupied_per_rank_bytes": (
                expected_physical_occupied),
            "average_physical_occupied_utilization_fraction": float(
                aggregate[
                    "average_physical_occupied_utilization_fraction"]),
            "average_reservation_adjusted_claim_per_rank_bytes": float(
                aggregate[
                    "average_reservation_adjusted_claim_per_rank_bytes"]),
            "average_reservation_adjusted_claim_fraction": float(
                aggregate["average_reservation_adjusted_claim_fraction"]),
            "scope": (
                "exact_time_weighted_measurement_window_per_rank; physical_"
                "categories_are_additive; logical_reservations_are_a_"
                "non_additive_overlay"
            ),
        },
    }


def _count_group(groups, key):
    return int((groups.get(key) or {}).get("count", 0))


def _safe_fraction(numerator, denominator):
    if denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _ratio_slowdown(value, oracle_value):
    if value is None or oracle_value is None or oracle_value == 0:
        return None
    return value / oracle_value - 1.0


def _validate_distribution(name, distribution):
    if not isinstance(distribution, dict):
        raise ExperimentError(f"Missing distribution {name}")
    count = distribution.get("count")
    total = distribution.get("sum")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ExperimentError(f"Invalid count in distribution {name}")
    if not isinstance(total, (int, float)) or isinstance(total, bool):
        raise ExperimentError(f"Invalid sum in distribution {name}")
    if not math.isfinite(float(total)) or total < 0:
        raise ExperimentError(f"Non-finite or negative sum in {name}")
    statistics = [
        distribution.get(key)
        for key in ("min", "p50", "p90", "p99", "max")
    ]
    mean = distribution.get("mean")
    if count == 0:
        if total != 0 or mean is not None or any(
                value is not None for value in statistics):
            raise ExperimentError(
                f"Empty distribution {name} has nonempty statistics")
        return
    numeric = [mean, *statistics]
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           for value in numeric):
        raise ExperimentError(f"Missing numeric statistics in {name}")
    if any(not math.isfinite(float(value)) or value < 0 for value in numeric):
        raise ExperimentError(
            f"Non-finite or negative latency statistic in {name}")
    if statistics != sorted(statistics):
        raise ExperimentError(f"Percentiles are not monotonic in {name}")
    if not statistics[0] <= mean <= statistics[-1]:
        raise ExperimentError(f"Mean lies outside min/max in {name}")
    expected_sum = float(mean) * count
    tolerance = max(1e-6, abs(float(total)) * 1e-9)
    if abs(expected_sum - float(total)) > tolerance:
        raise ExperimentError(
            f"Count/mean/sum do not reconcile in {name}")


def _validate_summary_row(row):
    for key, value in row.items():
        if value is None or isinstance(value, (str, bool)):
            continue
        if not isinstance(value, (int, float)):
            raise ExperimentError(
                f"Unexpected non-scalar summary metric {key}: "
                f"{type(value).__name__}")
        if not math.isfinite(float(value)):
            raise ExperimentError(f"Non-finite summary metric {key}={value}")
        if value < 0 and not key.startswith("oracle_"):
            raise ExperimentError(f"Negative summary metric {key}={value}")
        if ("fraction" in key and "slowdown" not in key
                and value > 1.0 + 1e-9):
            raise ExperimentError(
                f"Out-of-range fraction summary metric {key}={value}")


def _normalized_gap_type(value):
    value = str(value or "unknown").strip().lower()
    return value if value in {"human", "tool", "mixed", "unknown"} else "unknown"


def _expected_runtime_sessions(manifest):
    workload_path = manifest.get("workload_path")
    if not workload_path:
        return None
    rows = []
    with open(workload_path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            sub_requests = row.get("sub_requests")
            if not sub_requests:
                raise ExperimentError(
                    f"Materialized workload row {line_number} is not a "
                    "complete agentic session")
            rows.append(row)

    expected = {}
    epochs = (
        range(int(manifest.get("backlog_epochs", 1)))
        if manifest["mode"] == "backlog" else range(1)
    )
    next_request_id = 0
    for epoch in epochs:
        for template_index, row in enumerate(rows):
            source_session_id = str(row["session_id"])
            session_id = (
                f"{source_session_id}::template={template_index}::epoch={epoch}"
                if manifest["mode"] == "backlog" else source_session_id
            )
            if session_id in expected:
                raise ExperimentError(
                    f"Expected runtime session ID is duplicated: {session_id}")
            calls = []
            for sub_index, sub_request in enumerate(row["sub_requests"]):
                if sub_index == 0:
                    gap_type = "session_start"
                    gap_ns = 0
                else:
                    predecessor = row["sub_requests"][sub_index - 1]
                    gap_type = _normalized_gap_type(
                        predecessor.get("inter_turn_gap_type"))
                    gap_ns = int(
                        predecessor.get("tool_duration_ns", 0) or 0)
                calls.append({
                    "request_id": next_request_id,
                    "session_id": session_id,
                    "source_session_id": source_session_id,
                    "session_template_index": template_index,
                    "session_epoch": epoch,
                    "sub_request_index": sub_index,
                    "input_tokens": int(sub_request["input_toks"]),
                    "requested_output_tokens": int(sub_request["output_toks"]),
                    "prefix_reuse_tokens": int(
                        sub_request.get("prefix_reuse_toks", 0) or 0),
                    "return_gap_type": gap_type,
                    "return_gap_ns": gap_ns,
                })
                next_request_id += 1
            expected[session_id] = calls
    return expected


def _validate_trace_identity(manifest, report):
    expected_sessions = _expected_runtime_sessions(manifest)
    if expected_sessions is None:
        return {
            "performed": False,
            "reason": "manifest_has_no_materialized_workload_path",
        }
    request_records = report.get("requests", {}).get("records")
    if not isinstance(request_records, list):
        raise ExperimentError(
            f"Missing exact request records for {manifest['run_id']}")
    measured_ids = set(_measured_session_ids(report))
    unknown_sessions = sorted(measured_ids - set(expected_sessions))
    if unknown_sessions:
        raise ExperimentError(
            f"Measured runtime session IDs are absent from the materialized "
            f"workload for {manifest['run_id']}: {unknown_sessions[:5]}")

    observed_by_session = {}
    seen_request_ids = set()
    for record in request_records:
        request_id = int(record["request_id"])
        if request_id in seen_request_ids:
            raise ExperimentError(
                f"Duplicate completed request ID {request_id} in "
                f"{manifest['run_id']}")
        seen_request_ids.add(request_id)
        session_id = str(record["session_id"])
        if session_id not in measured_ids:
            raise ExperimentError(
                f"Request record for non-measured session {session_id} in "
                f"{manifest['run_id']}")
        observed_by_session.setdefault(session_id, []).append(record)

    lifecycle_by_session = {
        str(row["session_id"]): row
        for row in report["sessions"]["records"]
    }
    exact_fields = (
        "request_id", "session_id", "source_session_id",
        "session_template_index",
        "session_epoch", "sub_request_index", "input_tokens",
        "requested_output_tokens", "prefix_reuse_tokens",
        "return_gap_type", "return_gap_ns",
    )
    for session_id in sorted(measured_ids):
        expected_calls = expected_sessions[session_id]
        observed_calls = sorted(
            observed_by_session.get(session_id, []),
            key=lambda record: int(record["sub_request_index"]),
        )
        if len(observed_calls) != len(expected_calls):
            raise ExperimentError(
                f"Exact request count mismatch for session {session_id} in "
                f"{manifest['run_id']}: observed={len(observed_calls)}, "
                f"expected={len(expected_calls)}")
        lifecycle = lifecycle_by_session.get(session_id)
        if lifecycle is None:
            raise ExperimentError(
                f"Missing lifecycle for measured session {session_id}")
        for expected_call, observed_call in zip(
                expected_calls, observed_calls):
            for field in exact_fields:
                expected_value = expected_call[field]
                observed_value = observed_call.get(field)
                if observed_value != expected_value:
                    raise ExperimentError(
                        f"Trace identity mismatch for {manifest['run_id']}, "
                        f"session={session_id}, sub_request="
                        f"{expected_call['sub_request_index']}, field={field}: "
                        f"observed={observed_value!r}, "
                        f"expected={expected_value!r}")
            if int(observed_call["generated_tokens"]) != int(
                    expected_call["requested_output_tokens"]):
                raise ExperimentError(
                    f"Generated-token mismatch for {manifest['run_id']}, "
                    f"session={session_id}, sub_request="
                    f"{expected_call['sub_request_index']}")
        if int(observed_calls[0]["arrival_time_ns"]) != int(
                lifecycle["admission_time_ns"]):
            raise ExperimentError(
                f"Initial request/admission dependency mismatch for "
                f"{manifest['run_id']}, session={session_id}")
        for previous, current in zip(observed_calls, observed_calls[1:]):
            expected_arrival = (
                int(previous["end_time_ns"]) + int(current["return_gap_ns"]))
            if int(current["arrival_time_ns"]) != expected_arrival:
                raise ExperimentError(
                    f"Closed-loop dependency mismatch for {manifest['run_id']}, "
                    f"session={session_id}, sub_request="
                    f"{current['sub_request_index']}")
        if int(observed_calls[-1]["end_time_ns"]) != int(
                lifecycle["completion_time_ns"]):
            raise ExperimentError(
                f"Final request/session completion mismatch for "
                f"{manifest['run_id']}, session={session_id}")
    expected_record_count = sum(
        len(expected_sessions[session_id]) for session_id in measured_ids)
    if len(request_records) != expected_record_count:
        raise ExperimentError(
            f"Exact completed request set mismatch for {manifest['run_id']}")
    return {
        "performed": True,
        "passed": True,
        "checked_sessions": len(measured_ids),
        "checked_requests": len(request_records),
        "completed_identity_hash": _stable_json_hash([
            {
                key: record[key]
                for key in (*exact_fields, "generated_tokens")
            }
            for record in sorted(
                request_records,
                key=lambda record: (
                    str(record["session_id"]),
                    int(record["sub_request_index"]),
                ),
            )
        ]),
    }


def _expected_poisson_offered_arrivals(count, rate_sps, seed):
    """Reproduce Router's anchored Poisson process and its CRN identity."""
    count = int(count)
    rate_sps = float(rate_sps)
    seed = int(seed)
    if count <= 0:
        raise ExperimentError(
            "Poisson offered-arrival validation requires a positive count")
    if not math.isfinite(rate_sps) or rate_sps <= 0:
        raise ExperimentError(
            "Poisson offered-arrival validation requires a positive finite "
            "rate")
    if seed < 0:
        raise ExperimentError(
            "Poisson offered-arrival validation requires a non-negative seed")

    rate_rng = random.Random(seed)
    unit_rng = random.Random(seed)
    arrivals = [0]
    unit_draws = []
    arrival_ns = 0
    for _ in range(1, count):
        arrival_ns += int(rate_rng.expovariate(rate_sps) * 1_000_000_000)
        arrivals.append(arrival_ns)
        # Calling the same distribution at unit rate captures the underlying
        # common-random-number stream independently of the offered load.
        unit_draws.append(float.hex(unit_rng.expovariate(1.0)))
    unit_draw_trace_sha256 = _stable_json_hash({
        "algorithm": "python-random-expovariate-unit-rate-v1",
        "anchored_first_arrival_ns": 0,
        "session_count": count,
        "seed": seed,
        "unit_exponential_draws_hex": unit_draws,
    })
    return arrivals, unit_draw_trace_sha256


def _validate_offered_arrival_trace(manifest, report):
    """Recompute the offered trace and prove the Poisson CRN realization."""
    sessions = report.get("sessions") or {}
    lifecycle = sessions.get("records")
    if not isinstance(lifecycle, list) or not lifecycle:
        raise ExperimentError(
            f"Offered-arrival validation requires lifecycle records for "
            f"{manifest['run_id']}")

    reported_count = sessions.get("offered_arrival_trace_count")
    if (not isinstance(reported_count, int)
            or isinstance(reported_count, bool)
            or reported_count <= 0
            or reported_count != len(lifecycle)):
        raise ExperimentError(
            f"Offered-arrival trace count is missing or does not match the "
            f"lifecycle for {manifest['run_id']}: reported="
            f"{reported_count!r}, lifecycle={len(lifecycle)}")
    expected_available = manifest.get("available_sessions")
    if (expected_available is not None
            and reported_count != int(expected_available)):
        raise ExperimentError(
            f"Offered-arrival trace count does not match the generated "
            f"session cohort for {manifest['run_id']}: reported="
            f"{reported_count}, expected={expected_available}")

    offered_trace = []
    session_ids = set()
    for index, row in enumerate(lifecycle):
        if not isinstance(row, dict):
            raise ExperimentError(
                f"Offered-arrival lifecycle row {index} is not an object for "
                f"{manifest['run_id']}")
        session_id = row.get("session_id")
        offered_time_ns = row.get("offered_time_ns")
        if not isinstance(session_id, str) or not session_id:
            raise ExperimentError(
                f"Offered-arrival lifecycle row {index} has no session ID "
                f"for {manifest['run_id']}")
        if session_id in session_ids:
            raise ExperimentError(
                f"Offered-arrival lifecycle duplicates session {session_id!r} "
                f"for {manifest['run_id']}")
        if (not isinstance(offered_time_ns, int)
                or isinstance(offered_time_ns, bool)
                or offered_time_ns < 0):
            raise ExperimentError(
                f"Offered-arrival lifecycle row {index} has an invalid time "
                f"for {manifest['run_id']}: {offered_time_ns!r}")
        session_ids.add(session_id)
        offered_trace.append({
            "session_id": session_id,
            "offered_time_ns": offered_time_ns,
        })

    reported_hash = sessions.get("offered_arrival_trace_sha256")
    recomputed_hash = _stable_json_hash(offered_trace)
    if not _is_sha256_digest(reported_hash):
        raise ExperimentError(
            f"Offered-arrival trace hash is missing or malformed for "
            f"{manifest['run_id']}: {reported_hash!r}")
    if reported_hash != recomputed_hash:
        raise ExperimentError(
            f"Offered-arrival trace hash does not match lifecycle records for "
            f"{manifest['run_id']}")

    unit_draw_trace_sha256 = None
    if manifest.get("mode") == "poisson":
        rate_sps = manifest.get("load_value")
        seed = manifest.get("arrival_seed")
        if (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
            raise ExperimentError(
                f"Poisson offered-arrival validation requires an integer "
                f"seed for {manifest['run_id']}: {seed!r}")
        expected_times, unit_draw_trace_sha256 = (
            _expected_poisson_offered_arrivals(
                reported_count, rate_sps, seed))
        observed_times = [
            record["offered_time_ns"] for record in offered_trace]
        if observed_times != expected_times:
            mismatch = next(
                index for index, (observed, expected) in enumerate(zip(
                    observed_times, expected_times))
                if observed != expected
            )
            raise ExperimentError(
                f"Poisson offered-arrival process does not reproduce from "
                f"rate={rate_sps!r}, seed={seed} for {manifest['run_id']}; "
                f"index={mismatch}, observed={observed_times[mismatch]}, "
                f"expected={expected_times[mismatch]}")

    return {
        "performed": True,
        "passed": True,
        "count": reported_count,
        "offered_arrival_trace_sha256": recomputed_hash,
        "poisson_unit_draw_trace_sha256": unit_draw_trace_sha256,
        "poisson_reproduction_exact": manifest.get("mode") == "poisson",
    }


def _require_zero_totals(run_id, totals, fields):
    nonzero = {
        field: int(totals.get(field, 0) or 0)
        for field in fields
        if int(totals.get(field, 0) or 0) != 0
    }
    if nonzero:
        raise ExperimentError(
            f"Policy invariant failed for {run_id}; expected zero totals: "
            f"{nonzero}")


def _require_present_zero_totals(run_id, totals, fields):
    """Require an auditable counter to exist and be exactly zero."""
    missing = sorted(field for field in fields if field not in totals)
    if missing:
        raise ExperimentError(
            f"Policy invariant failed for {run_id}; missing required zero "
            f"totals: {missing}")
    _require_zero_totals(run_id, totals, fields)


def _validated_resume_timing(run_id, resume):
    """Return exact resume timestamps after checking the owner-gate split."""
    try:
        release_ns = int(resume["time_ns"])
        pair_ns = int(resume.get("pd_pair_fifo_wait_ns", 0) or 0)
        boundary_ns = int(
            resume.get("prepare_boundary_wait_ns", 0) or 0)
        source_join_ns = int(
            resume.get("source_demotion_join_wait_ns", 0) or 0)
        hbm_ns = int(resume.get("hbm_admission_wait_ns", 0) or 0)
        transient_dram_ns = int(
            resume.get("transient_dram_capacity_wait_ns", 0) or 0)
        queue_ns = int(resume.get("queue_wait_ns", 0) or 0)
        service_ns = int(resume.get("restore_service_ns", 0) or 0)
        restore_ns = int(resume["restore_ns"])
        owner_gate_ns = int(resume["owner_gate_ns"])
        issue_ns = int(resume["restore_issue_time_ns"])
        target_ready_ns = int(resume["target_hbm_ready_time_ns"])
        restore_ready_ns = int(resume["restore_ready_time_ns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentError(
            f"Malformed exact resume timing for {run_id}: {resume}") from exc
    components = (
        pair_ns, boundary_ns, source_join_ns, hbm_ns, transient_dram_ns,
        queue_ns,
        service_ns, restore_ns, owner_gate_ns,
    )
    if any(value < 0 for value in components):
        raise ExperimentError(
            f"Negative resume timing component for {run_id}: {resume}")
    if (restore_ns != hbm_ns + queue_ns + service_ns
            or owner_gate_ns
            != pair_ns + boundary_ns + source_join_ns + restore_ns
            or transient_dram_ns > queue_ns
            or issue_ns
            != release_ns + pair_ns + boundary_ns + source_join_ns
            or target_ready_ns != issue_ns + hbm_ns
            or restore_ready_ns != issue_ns + restore_ns):
        raise ExperimentError(
            f"Resume timing does not reconcile for {run_id}: {resume}")
    return {
        "release_ns": release_ns,
        "issue_ns": issue_ns,
        "target_ready_ns": target_ready_ns,
        "restore_ready_ns": restore_ready_ns,
        "hbm_ns": hbm_ns,
        "transient_dram_capacity_ns": transient_dram_ns,
        "queue_ns": queue_ns,
        "service_ns": service_ns,
        "restore_ns": restore_ns,
        "source_demotion_join_ns": source_join_ns,
    }


def _validated_transfer_timing(run_id, reservation):
    """Return a reservation timeline after exact queue/service validation."""
    try:
        arrival_ns = int(reservation["time_ns"])
        start_ns = int(reservation["start_ns"])
        complete_ns = int(reservation["complete_ns"])
        queue_ns = int(reservation["queue_wait_ns"])
        service_ns = int(reservation["service_ns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentError(
            f"Malformed migration reservation for {run_id}: "
            f"{reservation}") from exc
    if (min(arrival_ns, start_ns, complete_ns, queue_ns, service_ns) < 0
            or start_ns != arrival_ns + queue_ns
            or complete_ns != start_ns + service_ns):
        raise ExperimentError(
            f"Migration reservation timing does not reconcile for {run_id}: "
            f"{reservation}")
    return {
        "arrival_ns": arrival_ns,
        "start_ns": start_ns,
        "complete_ns": complete_ns,
        "queue_ns": queue_ns,
        "service_ns": service_ns,
    }


def _validate_ssd_restore_evidence(run_id, agentic_report):
    events = agentic_report.get("events") or []
    ssd_resumes = sorted(
        (
            event for event in events
            if event.get("event") == "resume"
            and event.get("source") == "ssd"
            and int(event.get("hit_tokens", 0) or 0) > 0
        ),
        key=lambda event: (
            int(event.get("time_ns", 0)), str(event.get("session_id"))),
    )
    reservations = [
        event for event in events
        if event.get("event") == "migration_reserve"
        and bool(event.get("foreground"))
    ]
    media_reservations = [
        event for event in reservations
        if event.get("kind") == "ssd_to_cpu_stage"
    ]
    h2d_reservations = [
        event for event in reservations
        if event.get("kind") == "cpu_stage_to_hbm"
    ]
    for reservation in media_reservations + h2d_reservations:
        _validated_transfer_timing(run_id, reservation)
    if (len(media_reservations) != len(ssd_resumes)
            or len(h2d_reservations) != len(ssd_resumes)):
        raise ExperimentError(
            f"SSD resume/stage counts do not reconcile for {run_id}: "
            f"resumes={len(ssd_resumes)}, media={len(media_reservations)}, "
            f"h2d={len(h2d_reservations)}")
    used = set()
    for resume in ssd_resumes:
        session_id = str(resume.get("session_id"))
        timing = _validated_resume_timing(run_id, resume)
        restore_complete_ns = timing["restore_ready_ns"]
        num_bytes = int(resume.get("bytes", 0))
        target_instance_id = int(resume.get("target_instance_id", -1))
        target_node_id = int(resume.get("target_node_id", -1))
        if target_instance_id < 0 or target_node_id < 0:
            raise ExperimentError(
                f"SSD resume lacks explicit target placement for {run_id}: "
                f"{resume}")
        media_match = None
        h2d_match = None
        for media_index, media in enumerate(reservations):
            if media_index in used:
                continue
            if (str(media.get("session_id")) != session_id
                    or media.get("kind") != "ssd_to_cpu_stage"
                    or int(media.get("time_ns", 0))
                    < timing["target_ready_ns"]
                    or int(media.get("bytes", 0)) != num_bytes):
                continue
            for h2d_index, h2d in enumerate(reservations):
                if h2d_index in used or h2d_index == media_index:
                    continue
                if (str(h2d.get("session_id")) != session_id
                        or h2d.get("kind") != "cpu_stage_to_hbm"
                        or int(h2d.get("bytes", 0)) != num_bytes
                        or int(h2d.get("time_ns", -1)) != int(
                            media.get("complete_ns", -2))
                        or int(h2d.get("start_ns", -1)) < int(
                            media.get("complete_ns", 0))
                        or int(h2d.get("complete_ns", -1))
                        != restore_complete_ns):
                    continue
                media_match = (media_index, media)
                h2d_match = (h2d_index, h2d)
                break
            if media_match is not None:
                break
        if media_match is None or h2d_match is None:
            raise ExperimentError(
                f"SSD resume lacks an exact serial SSD->DRAM->HBM transfer "
                f"chain for {run_id}, session={session_id}, "
                f"restore_issue={timing['issue_ns']}")
        media_index, media = media_match
        h2d_index, h2d = h2d_match
        expected_media_arrival = timing["target_ready_ns"]
        if int(media.get("time_ns", -1)) != expected_media_arrival:
            raise ExperimentError(
                f"SSD media stage does not begin after the exact HBM "
                f"admission wait for {run_id}, session={session_id}")
        stage_queue_ns = (
            int(media.get("queue_wait_ns", 0) or 0)
            + int(h2d.get("queue_wait_ns", 0) or 0))
        stage_service_ns = (
            int(media.get("service_ns", 0) or 0)
            + int(h2d.get("service_ns", 0) or 0))
        if (stage_queue_ns != int(resume.get("queue_wait_ns", 0) or 0)
                or stage_service_ns != int(
                    resume.get("restore_service_ns", 0) or 0)):
            raise ExperimentError(
                f"SSD stage queue/service accounting does not reconcile "
                f"for {run_id}, session={session_id}")
        media_resources = tuple(media.get("resources") or ())
        h2d_resources = tuple(h2d.get("resources") or ())
        expected_dram = f"node:{target_node_id}:dram"
        expected_pcie_prefix = (
            f"instance:{target_instance_id}:pcie-copy:")
        media_dram = {
            resource for resource in media_resources
            if resource.endswith(":dram")}
        h2d_dram = {
            resource for resource in h2d_resources
            if resource.endswith(":dram")}
        if (media_dram != {expected_dram}
                or "ssd-pool:read" not in media_resources
                or any(":pcie-copy:" in resource
                       for resource in media_resources)):
            raise ExperimentError(
                f"SSD media stage has invalid resources for {run_id}: "
                f"{media_resources}")
        if (h2d_dram != media_dram
                or not any(resource.startswith(expected_pcie_prefix)
                           for resource in h2d_resources)
                or any(resource.startswith("ssd-pool:")
                       for resource in h2d_resources)):
            raise ExperimentError(
                f"DRAM-to-HBM stage has invalid resources for {run_id}: "
                f"{h2d_resources}")
        used.update((media_index, h2d_index))
    return len(ssd_resumes)


def _validate_cpu_restore_evidence(run_id, agentic_report):
    """Require one exact CPU->HBM reservation for every CPU resume."""
    events = agentic_report.get("events") or []
    cpu_resumes = sorted(
        (
            event for event in events
            if event.get("event") == "resume"
            and event.get("source") == "cpu"
            and int(event.get("hit_tokens", 0) or 0) > 0
        ),
        key=lambda event: (
            int(event.get("time_ns", 0)), str(event.get("session_id"))),
    )
    reservations = [
        event for event in events
        if event.get("event") == "migration_reserve"
        and event.get("kind") == "cpu_to_hbm"
        and bool(event.get("foreground"))
    ]
    for reservation in reservations:
        _validated_transfer_timing(run_id, reservation)
    if len(reservations) != len(cpu_resumes):
        raise ExperimentError(
            f"CPU resume/reservation counts do not reconcile for {run_id}: "
            f"resumes={len(cpu_resumes)}, reservations={len(reservations)}")
    used = set()
    for resume in cpu_resumes:
        timing = _validated_resume_timing(run_id, resume)
        session_id = str(resume.get("session_id"))
        num_bytes = int(resume.get("bytes", 0) or 0)
        target_instance_id = int(resume.get("target_instance_id", -1))
        target_node_id = int(resume.get("target_node_id", -1))
        if target_instance_id < 0 or target_node_id < 0:
            raise ExperimentError(
                f"CPU resume lacks explicit target placement for {run_id}: "
                f"{resume}")
        matches = []
        for index, reservation in enumerate(reservations):
            if index in used:
                continue
            resources = tuple(reservation.get("resources") or ())
            if (str(reservation.get("session_id")) != session_id
                    or int(reservation.get("bytes", -1)) != num_bytes
                    or int(reservation.get("time_ns", -1))
                    != timing["target_ready_ns"]
                    or int(reservation.get("complete_ns", -1))
                    != timing["restore_ready_ns"]
                    or int(reservation.get("queue_wait_ns", -1))
                    != timing["queue_ns"]
                    or int(reservation.get("service_ns", -1))
                    != timing["service_ns"]
                    or f"node:{target_node_id}:dram" not in resources
                    or sum(resource.endswith(":dram")
                           for resource in resources) != 1
                    or not any(
                        resource.startswith(
                            f"instance:{target_instance_id}:pcie-copy:")
                        for resource in resources)
                    or any(resource.startswith("ssd-pool:")
                           for resource in resources)):
                continue
            matches.append((index, reservation))
        if len(matches) != 1:
            raise ExperimentError(
                "CPU resume lacks one exact CPU->HBM transfer for "
                f"{run_id}, session={session_id}, matches={len(matches)}")
        used.add(matches[0][0])
    return len(cpu_resumes)


def _validate_external_resume_evidence(run_id, agentic_report, intervals):
    """Map every completed cross-instance HBM resume to one ASTRA job."""
    resumes = [
        event for event in agentic_report.get("events") or ()
        if event.get("event") == "resume"
        and event.get("source") == "hbm"
        and int(event.get("hit_tokens", 0) or 0) > 0
        and int(event.get("source_instance_id", -1))
        != int(event.get("target_instance_id", -1))
    ]
    used_jobs = set()
    for resume in resumes:
        timing = _validated_resume_timing(run_id, resume)
        session_id = str(resume.get("session_id"))
        source_id = int(resume.get("source_instance_id", -1))
        target_id = int(resume.get("target_instance_id", -1))
        num_bytes = int(resume.get("bytes", 0) or 0)
        matches = []
        for interval in intervals:
            job_id = str(interval.get("job_id") or "")
            if job_id in used_jobs:
                continue
            if (str(interval.get("session_id")) != session_id
                    or int(interval.get("source_instance_id", -1)) != source_id
                    or int(interval.get("target_instance_id", -1)) != target_id
                    or int(interval.get("bytes", -1)) != num_bytes
                    or int(interval.get("arrival_ns", -1))
                    != timing["target_ready_ns"]
                    or int(interval.get("complete_ns", -1))
                    != timing["restore_ready_ns"]
                    or int(interval.get("queue_wait_ns", -1))
                    != timing["queue_ns"]
                    or int(interval.get("service_ns", -1))
                    != timing["service_ns"]):
                continue
            matches.append(interval)
        if len(matches) != 1:
            raise ExperimentError(
                "Cross-instance HBM resume lacks one exact external ASTRA "
                f"job for {run_id}, session={session_id}, "
                f"matches={len(matches)}")
        used_jobs.add(str(matches[0]["job_id"]))
    return {
        "resume_count": len(resumes),
        "matched_job_ids": sorted(used_jobs),
    }


def _external_fabric_model_coexecution_audit(agentic_report, intervals=None):
    """Cross-check external cold jobs against source/target model windows.

    The manager's resource-timeline report historically counted only Python-
    calendar peer copies. Congestion-aware direct restores instead live in
    ASTRA's event queue, so derive their overlap from exact completed-job and
    ``astra_shared_fabric_window`` records at result-validation time. This is
    an endpoint co-execution audit, not a claim that every overlapping graph
    was extended; a paired no-transfer control is required for that delta.
    """
    external = agentic_report.get("external_fabric") or {}
    if intervals is None:
        intervals = list(external.get("completed_intervals") or ())
    else:
        intervals = list(intervals)
    windows_by_instance = {}
    for event in agentic_report.get("events") or ():
        if event.get("event") != "astra_shared_fabric_window":
            continue
        try:
            instance_id = int(event["instance_id"])
            batch_id = int(event["batch_id"])
            start_ns = int(event["start_ns"])
            complete_ns = int(event["complete_ns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExperimentError(
                "Malformed ASTRA model window in external-fabric audit: "
                f"{event}") from exc
        if start_ns < 0 or complete_ns < start_ns:
            raise ExperimentError(
                "Non-causal ASTRA model window in external-fabric audit: "
                f"{event}")
        if complete_ns == start_ns:
            continue
        windows_by_instance.setdefault(instance_id, []).append({
            "instance_id": instance_id,
            "batch_id": batch_id,
            "resource": str(event.get("resource") or "unknown"),
            "start_ns": start_ns,
            "complete_ns": complete_ns,
        })

    window_ends_by_instance = {}
    for instance_id, windows in windows_by_instance.items():
        windows.sort(key=lambda row: (
            row["start_ns"], row["complete_ns"], row["batch_id"]))
        for previous, current in zip(windows, windows[1:]):
            if current["start_ns"] < previous["complete_ns"]:
                raise ExperimentError(
                    "One model instance has overlapping ASTRA windows in "
                    "external-fabric audit: "
                    f"instance={instance_id}, previous={previous}, "
                    f"current={current}")
        window_ends_by_instance[instance_id] = [
            row["complete_ns"] for row in windows]

    pair_count = 0
    overlap_membership_ns = 0
    overlap_intervals = []
    overlapped_jobs = set()
    overlapped_windows = set()
    samples = []
    for cold in intervals:
        try:
            cold_start_ns = int(cold["start_ns"])
            cold_complete_ns = int(cold["complete_ns"])
            source_instance_id = int(cold["source_instance_id"])
            target_instance_id = int(cold["target_instance_id"])
        except (KeyError, TypeError, ValueError):
            # Older synthetic policy-unit fixtures did not carry endpoint
            # identity. Their lifecycle remains validated, but no endpoint
            # overlap is claimed from incomplete evidence.
            continue
        if cold_complete_ns <= cold_start_ns:
            continue
        job_id = str(cold.get("job_id") or "unknown")
        for instance_id in sorted({source_instance_id, target_instance_id}):
            windows = windows_by_instance.get(instance_id, ())
            ends = window_ends_by_instance.get(instance_id, ())
            index = bisect_right(ends, cold_start_ns)
            while index < len(windows):
                window = windows[index]
                if window["start_ns"] >= cold_complete_ns:
                    break
                overlap_start_ns = max(
                    cold_start_ns, window["start_ns"])
                overlap_complete_ns = min(
                    cold_complete_ns, window["complete_ns"])
                if overlap_start_ns < overlap_complete_ns:
                    duration_ns = overlap_complete_ns - overlap_start_ns
                    pair_count += 1
                    overlap_membership_ns += duration_ns
                    overlap_intervals.append((
                        overlap_start_ns, overlap_complete_ns))
                    overlapped_jobs.add(job_id)
                    overlapped_windows.add((
                        instance_id, window["batch_id"],
                        window["start_ns"], window["complete_ns"],
                    ))
                    if len(samples) < 32:
                        samples.append({
                            "job_id": job_id,
                            "session_id": cold.get("session_id"),
                            "source_instance_id": source_instance_id,
                            "target_instance_id": target_instance_id,
                            "model_instance_id": instance_id,
                            "model_batch_id": window["batch_id"],
                            "cold_start_ns": cold_start_ns,
                            "cold_complete_ns": cold_complete_ns,
                            "model_start_ns": window["start_ns"],
                            "model_complete_ns": window["complete_ns"],
                            "overlap_start_ns": overlap_start_ns,
                            "overlap_complete_ns": overlap_complete_ns,
                            "overlap_ns": duration_ns,
                        })
                index += 1

    union_ns = 0
    if overlap_intervals:
        ordered = sorted(overlap_intervals)
        union_start, union_end = ordered[0]
        for start_ns, complete_ns in ordered[1:]:
            if start_ns <= union_end:
                union_end = max(union_end, complete_ns)
            else:
                union_ns += union_end - union_start
                union_start, union_end = start_ns, complete_ns
        union_ns += union_end - union_start
    return {
        "performed": bool(external.get("enabled")),
        "scope": "source_or_target_instance_model_window",
        "coexecution_pair_count": pair_count,
        "overlapped_job_count": len(overlapped_jobs),
        "overlapped_model_window_count": len(overlapped_windows),
        "coexecution_membership_ns": overlap_membership_ns,
        "coexecution_union_ns": union_ns,
        "samples": samples,
        "interpretation": (
            "Exact ASTRA job/model co-execution at a transfer endpoint. "
            "ASTRA charges their shared-link and endpoint contention; use a "
            "paired no-transfer control to attribute a model-latency delta."
        ),
    }


def _queue_recompute_projection_errors(
        event, *, ratio, min_wait_ns, cost_multiplier, block_size,
        decision_kind, selection_event=False):
    """Return schema-19 causal/accounting errors for one prefix choice."""
    errors = []

    def nonnegative_integer(field, *, nullable=False):
        value = event.get(field)
        if nullable and value is None:
            return None
        if (not isinstance(value, int) or isinstance(value, bool)
                or value < 0):
            errors.append(f"{field}={value!r}")
            return None
        return value

    event_time_ns = nonnegative_integer("time_ns")
    hbm_wait_ns = nonnegative_integer(
        "projected_hbm_admission_wait_ns")
    transient_dram_wait_ns = nonnegative_integer(
        "projected_transient_dram_capacity_wait_ns")
    queue_wait_ns = nonnegative_integer("projected_queue_wait_ns")
    total_wait_ns = nonnegative_integer("projected_total_wait_ns")
    service_ns = nonnegative_integer("projected_service_ns")
    restore_ns = nonnegative_integer("projected_restore_ns")
    ratio_threshold_ns = nonnegative_integer("ratio_threshold_ns")
    threshold_ns = nonnegative_integer("threshold_ns")
    configured_min_wait_ns = nonnegative_integer(
        "configured_min_wait_ns")
    cost_threshold_ns = nonnegative_integer(
        "cost_threshold_ns", nullable=True)
    estimated_full_recompute_ns = nonnegative_integer(
        "estimated_incremental_recompute_comp_ns", nullable=True)
    estimated_suffix_recompute_ns = nonnegative_integer(
        "estimated_suffix_recompute_comp_ns", nullable=True)
    selected_path_ns = nonnegative_integer(
        "selected_predicted_resume_path_ns", nullable=True)
    full_path_ns = nonnegative_integer(
        "full_predicted_resume_path_ns", nullable=True)
    projection_arrival_ns = nonnegative_integer(
        "projection_arrival_ns", nullable=True)
    full_bytes = nonnegative_integer("bytes")
    reusable_tokens = nonnegative_integer("reusable_tokens_R")
    selected_tokens = nonnegative_integer("selected_prefix_tokens_H")
    selected_block_tokens = nonnegative_integer(
        "selected_prefix_block_tokens")
    suffix_tokens = nonnegative_integer("dropped_suffix_tokens")
    selected_bytes = nonnegative_integer("selected_restore_bytes")
    suffix_bytes = nonnegative_integer("dropped_suffix_bytes")
    avoided_bytes = nonnegative_integer("avoided_restore_bytes")
    physical_dropped_bytes = nonnegative_integer(
        "physical_entry_dropped_bytes")
    if selection_event:
        for field in (
                "declared_reuse_tokens", "reusable_tokens",
                "policy_avoidable_tokens"):
            nonnegative_integer(field)

    session_id = event.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        errors.append(f"session_id={session_id!r}")
    source = event.get("source")
    if source not in {"cpu", "ssd"}:
        errors.append(f"source={source!r}")
    transfer_kinds = event.get("transfer_kinds")
    if (not isinstance(transfer_kinds, list) or not transfer_kinds
            or any(not isinstance(value, str) or not value
                   for value in transfer_kinds)):
        errors.append(f"transfer_kinds={transfer_kinds!r}")

    projection_available = event.get("projection_available")
    if not isinstance(projection_available, bool):
        errors.append(
            f"projection_available={projection_available!r}")
        projection_available = False
    if projection_available != (projection_arrival_ns is not None):
        errors.append(
            "projection_available/projection_arrival_ns mismatch")
    if (projection_available and event_time_ns is not None
            and projection_arrival_ns is not None
            and hbm_wait_ns is not None
            and projection_arrival_ns - event_time_ns != hbm_wait_ns):
        errors.append(
            "projected_hbm_admission_wait_ns != "
            "projection_arrival_ns - time_ns")
    if (not projection_available
            and any(value not in {None, 0} for value in (
                hbm_wait_ns, queue_wait_ns, total_wait_ns,
                transient_dram_wait_ns, service_ns, restore_ns))):
        errors.append(
            "unavailable projection has nonzero timing components")

    expected_decision = {
        "full": "restore",
        "partial": "partial_restore_suffix_recompute",
        "zero": "drop_recompute",
    }[decision_kind]
    if not selection_event and event.get("decision") != expected_decision:
        errors.append(
            f"decision={event.get('decision')!r}, "
            f"expected={expected_decision!r}")
    if all(value is not None for value in (
            reusable_tokens, selected_tokens, selected_block_tokens,
            suffix_tokens)):
        expected_block_tokens = (
            math.ceil(selected_tokens / block_size) * block_size
            if selected_tokens else 0)
        if reusable_tokens <= 0:
            errors.append(f"reusable_tokens_R={reusable_tokens!r}")
        if not 0 <= selected_tokens <= reusable_tokens:
            errors.append(
                f"selected_prefix_tokens_H={selected_tokens!r}, "
                f"R={reusable_tokens!r}")
        if selected_block_tokens != expected_block_tokens:
            errors.append(
                f"selected_prefix_block_tokens={selected_block_tokens!r}, "
                f"expected={expected_block_tokens!r}")
        if suffix_tokens != reusable_tokens - selected_tokens:
            errors.append(
                "dropped_suffix_tokens != reusable_tokens_R - "
                "selected_prefix_tokens_H")
        expected_kind = (
            "zero" if selected_tokens == 0 else
            "full" if selected_tokens == reusable_tokens else "partial")
        if decision_kind != expected_kind:
            errors.append(
                f"decision kind {decision_kind!r} conflicts with R/H "
                f"({reusable_tokens}/{selected_tokens})")
        if decision_kind == "partial" and selected_tokens % block_size:
            errors.append("partial selected_prefix_tokens_H is not block aligned")
    if all(value is not None for value in (
            full_bytes, selected_bytes, suffix_bytes, avoided_bytes)):
        if selected_bytes + suffix_bytes != full_bytes:
            errors.append(
                "selected_restore_bytes + dropped_suffix_bytes != bytes")
        if avoided_bytes != suffix_bytes:
            errors.append(
                "avoided_restore_bytes != dropped_suffix_bytes")
        if decision_kind == "full" and (
                selected_bytes != full_bytes or suffix_bytes != 0):
            errors.append("full restore does not preserve all bytes")
        if decision_kind == "zero" and selected_bytes != 0:
            errors.append("H=0 decision transfers nonzero bytes")
    if (physical_dropped_bytes is not None
            and decision_kind != "zero" and physical_dropped_bytes != 0):
        errors.append("nonzero physical entry drop outside H=0 path")

    required_victim_fields = {
        "projected_hbm_victim_sessions",
        "projected_cpu_victim_sessions",
    }
    victim_fields = sorted(required_victim_fields | {
        field for field in event
        if (field.startswith("projected_")
            and field.endswith("_victim_sessions"))
    })
    victim_lists = {}
    for field in victim_fields:
        victims = event.get(field)
        if (not isinstance(victims, list)
                or any(not isinstance(value, str) or not value
                       for value in victims)
                or len(victims) != len(set(victims))):
            errors.append(f"{field}={victims!r}")
        else:
            victim_lists[field] = victims
    if len(victim_lists) == len(victim_fields):
        projected_victims = [
            victim
            for victims in victim_lists.values()
            for victim in victims
        ]
        if session_id in projected_victims:
            errors.append("foreground session appears in victim list")
        includes_lru = bool(projected_victims)
        if (event.get("projection_includes_collateral_lru_work")
                is not includes_lru):
            errors.append(
                "projection_includes_collateral_lru_work/victim mismatch")
        expected_legacy_available = (
            projection_available and not includes_lru)
        if (event.get("projection_available_without_new_lru_work")
                is not expected_legacy_available):
            errors.append(
                "projection_available_without_new_lru_work is not the "
                "available-without-victims compatibility alias")

    if all(value is not None for value in (
            hbm_wait_ns, queue_wait_ns, total_wait_ns)):
        if total_wait_ns != hbm_wait_ns + queue_wait_ns:
            errors.append(
                "projected_total_wait_ns != "
                "projected_hbm_admission_wait_ns + "
                "projected_queue_wait_ns")
    if (transient_dram_wait_ns is not None and queue_wait_ns is not None
            and transient_dram_wait_ns > queue_wait_ns):
        errors.append(
            "projected_transient_dram_capacity_wait_ns exceeds "
            "projected_queue_wait_ns")
    if all(value is not None for value in (
            total_wait_ns, service_ns, restore_ns)):
        if restore_ns != total_wait_ns + service_ns:
            errors.append(
                "projected_restore_ns != projected_total_wait_ns + "
                "projected_service_ns")

    configured_ratio = event.get("configured_wait_service_ratio")
    configured_cost = event.get("configured_cost_guard_multiplier")
    if (not isinstance(configured_ratio, (int, float))
            or isinstance(configured_ratio, bool)
            or not math.isfinite(float(configured_ratio))
            or float(configured_ratio) != float(ratio)):
        errors.append(
            f"configured_wait_service_ratio={configured_ratio!r}")
    if (not isinstance(configured_cost, (int, float))
            or isinstance(configured_cost, bool)
            or not math.isfinite(float(configured_cost))
            or float(configured_cost) != float(cost_multiplier)):
        errors.append(
            f"configured_cost_guard_multiplier={configured_cost!r}")
    if configured_min_wait_ns != int(min_wait_ns):
        errors.append(
            f"configured_min_wait_ns={configured_min_wait_ns!r}")

    expected_ratio_threshold_ns = (
        math.ceil(float(ratio) * service_ns)
        if service_ns is not None else None)
    if ratio_threshold_ns != expected_ratio_threshold_ns:
        errors.append(
            f"ratio_threshold_ns={ratio_threshold_ns!r}, "
            f"expected={expected_ratio_threshold_ns!r}")
    expected_threshold_ns = (
        max(expected_ratio_threshold_ns, int(min_wait_ns))
        if expected_ratio_threshold_ns is not None else None)
    if threshold_ns != expected_threshold_ns:
        errors.append(
            f"threshold_ns={threshold_ns!r}, "
            f"expected={expected_threshold_ns!r}")

    expected_severe_gate = bool(
        projection_available
        and total_wait_ns is not None
        and expected_threshold_ns is not None
        and total_wait_ns > expected_threshold_ns)
    if event.get("severe_gate_pass") is not expected_severe_gate:
        errors.append(
            "severe_gate_pass does not use projected_total_wait_ns")

    expected_cost_threshold_ns = (
        math.ceil(cost_multiplier * estimated_suffix_recompute_ns)
        if estimated_suffix_recompute_ns is not None else None)
    if (decision_kind != "full" and cost_multiplier > 0
            and estimated_suffix_recompute_ns is None):
        errors.append("modified choice lacks suffix recompute estimate")
    if (decision_kind != "full" and cost_multiplier == 0
            and estimated_suffix_recompute_ns is not None):
        errors.append(
            "disabled cost guard unexpectedly reports a modified suffix "
            "estimate")
    if (decision_kind == "full"
            and estimated_suffix_recompute_ns != 0):
        errors.append("full restore suffix recompute estimate is not zero")
    if cost_threshold_ns != expected_cost_threshold_ns:
        errors.append(
            f"cost_threshold_ns={cost_threshold_ns!r}, "
            f"expected={expected_cost_threshold_ns!r}")
    expected_cost_gate = decision_kind != "full"
    if event.get("cost_gate_pass") is not expected_cost_gate:
        errors.append("cost_gate_pass does not match the modified path")
    if decision_kind != "full":
        if not expected_severe_gate:
            errors.append("modified choice does not pass the severe gate")
        if (selected_path_ns is None or full_path_ns is None
                or selected_path_ns >= full_path_ns):
            errors.append(
                "modified selected path does not strictly improve the full "
                "restore path")
        if restore_ns is not None and full_path_ns != restore_ns:
            errors.append(
                "full_predicted_resume_path_ns != projected_restore_ns")
    elif full_path_ns != selected_path_ns:
        errors.append("full restore selected/full predicted paths diverge")

    prefix_available = event.get("prefix_projection_available")
    if not isinstance(prefix_available, bool):
        errors.append(f"prefix_projection_available={prefix_available!r}")
        prefix_available = False
    prefix_hbm_wait_ns = nonnegative_integer(
        "prefix_projected_hbm_admission_wait_ns")
    prefix_transient_wait_ns = nonnegative_integer(
        "prefix_projected_transient_dram_capacity_wait_ns")
    prefix_queue_wait_ns = nonnegative_integer(
        "prefix_projected_queue_wait_ns")
    prefix_service_ns = nonnegative_integer(
        "prefix_projected_service_ns")
    if (prefix_transient_wait_ns is not None
            and prefix_queue_wait_ns is not None
            and prefix_transient_wait_ns > prefix_queue_wait_ns):
        errors.append(
            "prefix transient-DRAM wait exceeds prefix queue wait")
    if decision_kind == "partial":
        if prefix_available is not True:
            errors.append("partial decision lacks a prefix projection")
        if prefix_hbm_wait_ns != 0:
            errors.append("partial prefix has projected HBM admission wait")
        penalty_ns = expected_cost_threshold_ns or 0
        if all(value is not None for value in (
                prefix_hbm_wait_ns, prefix_queue_wait_ns,
                prefix_service_ns, selected_path_ns)):
            expected_selected_path_ns = (
                prefix_hbm_wait_ns + prefix_queue_wait_ns
                + prefix_service_ns + penalty_ns)
            if selected_path_ns != expected_selected_path_ns:
                errors.append(
                    "partial selected path does not equal prefix restore + "
                    "suffix recompute penalty")
    elif decision_kind == "zero":
        if prefix_available is not False:
            errors.append("H=0 unexpectedly has a prefix projection")
        if any(value not in {None, 0} for value in (
                prefix_hbm_wait_ns, prefix_transient_wait_ns,
                prefix_queue_wait_ns, prefix_service_ns)):
            errors.append("H=0 has nonzero prefix projection components")
        if selected_path_ns != (expected_cost_threshold_ns or 0):
            errors.append(
                "H=0 selected path does not equal recompute penalty")

    for field, expected in (
            ("capacity_headroom_snapshot_only", True),
            ("capacity_headroom_claimed_by_policy", False),
            ("pd_first_chunk_immediate_admission_guaranteed", False)):
        if event.get(field) is not expected:
            errors.append(f"{field}={event.get(field)!r}")
    snapshot = event.get("capacity_headroom_snapshot")
    if decision_kind == "partial":
        if not isinstance(snapshot, dict):
            errors.append("partial decision lacks capacity_headroom_snapshot")
        else:
            snapshot_integer_fields = (
                "time_ns", "prefix_tokens", "prefix_block_tokens",
                "next_chunk_tokens", "through_next_chunk_block_tokens",
                "prefill_instance_id",
                "prefill_unreserved_per_rank_bytes",
                "prefill_prefix_per_rank_bytes",
                "prefill_growth_headroom_per_rank_bytes",
                "prefill_required_through_chunk_per_rank_bytes",
                "decode_instance_id", "decode_unreserved_per_rank_bytes",
                "decode_required_through_chunk_per_rank_bytes",
            )
            values = {}
            for field in snapshot_integer_fields:
                value = snapshot.get(field)
                if (not isinstance(value, int) or isinstance(value, bool)
                        or value < 0):
                    errors.append(f"capacity snapshot {field}={value!r}")
                else:
                    values[field] = value
            if snapshot.get("semantics") != "causal_snapshot_not_reservation":
                errors.append("capacity snapshot semantics are not causal")
            if snapshot.get("feasible") is not True:
                errors.append("selected partial snapshot is not feasible")
            if (event_time_ns is not None
                    and values.get("time_ns") != event_time_ns):
                errors.append("capacity snapshot time does not match decision")
            if selected_tokens is not None and values.get(
                    "prefix_tokens") != selected_tokens:
                errors.append("capacity snapshot prefix_tokens != H")
            if selected_block_tokens is not None and values.get(
                    "prefix_block_tokens") != selected_block_tokens:
                errors.append("capacity snapshot prefix blocks != selected blocks")
            if all(field in values for field in (
                    "prefix_tokens", "next_chunk_tokens",
                    "through_next_chunk_block_tokens")):
                expected_through_blocks = math.ceil(
                    (values["prefix_tokens"] + values["next_chunk_tokens"])
                    / block_size) * block_size
                if (values["through_next_chunk_block_tokens"]
                        != expected_through_blocks):
                    errors.append("capacity snapshot through-chunk blocks diverge")
            if all(field in values for field in (
                    "prefill_prefix_per_rank_bytes",
                    "prefill_growth_headroom_per_rank_bytes",
                    "prefill_required_through_chunk_per_rank_bytes")):
                if (values["prefill_prefix_per_rank_bytes"]
                        + values["prefill_growth_headroom_per_rank_bytes"]
                        != values[
                            "prefill_required_through_chunk_per_rank_bytes"]):
                    errors.append("capacity snapshot P byte identity failed")
            if all(field in values for field in (
                    "prefill_unreserved_per_rank_bytes",
                    "prefill_required_through_chunk_per_rank_bytes")):
                if (values["prefill_required_through_chunk_per_rank_bytes"]
                        > values["prefill_unreserved_per_rank_bytes"]):
                    errors.append("capacity snapshot P headroom is infeasible")
            if all(field in values for field in (
                    "decode_unreserved_per_rank_bytes",
                    "decode_required_through_chunk_per_rank_bytes")):
                if (values["decode_required_through_chunk_per_rank_bytes"]
                        > values["decode_unreserved_per_rank_bytes"]):
                    errors.append("capacity snapshot D headroom is infeasible")
    elif snapshot is not None:
        errors.append("non-partial decision carries a capacity snapshot")

    if selection_event:
        if event.get("logical_session_effect") != "none":
            errors.append("KV choice changes the logical session")
        expected_scope = (
            "contiguous_block_aligned_prefix"
            if decision_kind == "partial" else "whole_reusable_entry")
        if event.get("selection_scope") != expected_scope:
            errors.append(
                f"selection_scope={event.get('selection_scope')!r}")
        expected_event = (
            "queue_recompute_partial"
            if decision_kind == "partial" else "queue_recompute_drop")
        if event.get("event") != expected_event:
            errors.append(f"event={event.get('event')!r}")
        if decision_kind == "partial":
            if event.get("recompute_scope") != "contiguous_suffix_H_to_R":
                errors.append("partial recompute scope is not suffix-only")
            if event.get("source_pin_scope") != (
                    "full_physical_source_until_prefix_dma_complete"):
                errors.append("partial physical source is not fully pinned")
            pinned = nonnegative_integer(
                "physical_source_bytes_pinned_until_dma_complete")
            if pinned is not None and pinned <= 0:
                errors.append("partial physical source pin has zero bytes")
        else:
            if event.get("recompute_scope") != "whole_reusable_prefix":
                errors.append("H=0 recompute scope changed")
    if event.get(
            "projection_precedes_destination_hbm_reservation") is not True:
        errors.append(
            "projection does not precede destination HBM reservation")
    return errors


def _queue_recompute_first_chunk_audit(events, partial_events):
    """Join each partial snapshot to its first emitted P/D chunk admission."""
    chunk_events = [
        event for event in events
        if (event.get("event") == "pd_chunk_admission"
            and event.get("first_chunk") is True)
    ]
    if not chunk_events:
        return {
            "performed": False,
            "reason": "agentic_report_has_no_pd_chunk_admission_events",
            "partial_snapshot_count": len(partial_events),
            "joined_count": 0,
        }
    available_by_session = {}
    for event in chunk_events:
        available_by_session.setdefault(str(event.get("session_id")), []).append(
            event)
    for rows in available_by_session.values():
        rows.sort(key=lambda row: (
            int(row.get("enqueued_ns", -1)), int(row.get("request_id", -1))))

    joined = []
    errors = []
    for selection in partial_events:
        session_id = str(selection.get("session_id"))
        snapshot = selection.get("capacity_headroom_snapshot") or {}
        decision_time_ns = int(selection.get("time_ns", -1))
        candidates = available_by_session.get(session_id, [])
        match_index = next((
            index for index, event in enumerate(candidates)
            if int(event.get("enqueued_ns", -1)) >= decision_time_ns
        ), None)
        if match_index is None:
            errors.append(
                f"partial snapshot has no subsequent first chunk: {session_id}")
            continue
        actual = candidates.pop(match_index)
        integer_fields = (
            "request_id", "prefill_instance_id", "decode_instance_id",
            "computed_tokens", "chunk_tokens", "target_tokens",
            "prefill_current_per_rank_bytes",
            "decode_current_per_rank_bytes",
            "prefill_target_per_rank_bytes",
            "decode_target_per_rank_bytes",
            "prefill_delta_per_rank_bytes", "decode_delta_per_rank_bytes",
            "prefill_unreserved_per_rank_bytes",
            "decode_unreserved_per_rank_bytes", "enqueued_ns",
            "prefill_capacity_ready_ns", "decode_capacity_ready_ns",
            "admitted_ns", "wait_ns", "critical_wait_after_restore_ns",
        )
        values = {}
        for field in integer_fields:
            value = actual.get(field)
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 0):
                errors.append(
                    f"pd_chunk_admission {session_id} {field}={value!r}")
            else:
                values[field] = value
        if (snapshot.get("prefill_instance_id")
                != actual.get("prefill_instance_id")
                or snapshot.get("decode_instance_id")
                != actual.get("decode_instance_id")):
            errors.append(
                f"snapshot/actual P/D pair mismatch for {session_id}")
        if actual.get("capacity_headroom_snapshot") != snapshot:
            errors.append(
                f"snapshot payload was not preserved for {session_id}")
        if (actual.get("capacity_headroom_snapshot_only") is not True
                or actual.get("capacity_headroom_claimed_by_policy") is not False
                or actual.get("capacity_snapshot_feasible") is not True
                or actual.get("capacity_snapshot_decision_time_ns")
                != snapshot.get("time_ns")):
            errors.append(
                f"snapshot/actual semantics diverged for {session_id}")
        if all(field in values for field in (
                "prefill_current_per_rank_bytes",
                "prefill_delta_per_rank_bytes",
                "prefill_target_per_rank_bytes")) and (
                values["prefill_current_per_rank_bytes"]
                + values["prefill_delta_per_rank_bytes"]
                != values["prefill_target_per_rank_bytes"]):
            errors.append(f"actual P byte identity failed for {session_id}")
        if all(field in values for field in (
                "decode_current_per_rank_bytes",
                "decode_delta_per_rank_bytes",
                "decode_target_per_rank_bytes")) and (
                values["decode_current_per_rank_bytes"]
                + values["decode_delta_per_rank_bytes"]
                != values["decode_target_per_rank_bytes"]):
            errors.append(f"actual D byte identity failed for {session_id}")
        if all(field in values for field in (
                "enqueued_ns", "admitted_ns", "wait_ns")) and (
                values["admitted_ns"] - values["enqueued_ns"]
                != values["wait_ns"]):
            errors.append(f"actual chunk wait identity failed for {session_id}")
        joined.append({
            "session_id": session_id,
            "request_id": actual.get("request_id"),
            "snapshot_time_ns": snapshot.get("time_ns"),
            "actual_enqueued_ns": actual.get("enqueued_ns"),
            "snapshot_prefill_unreserved_per_rank_bytes": snapshot.get(
                "prefill_unreserved_per_rank_bytes"),
            "actual_prefill_unreserved_per_rank_bytes": actual.get(
                "prefill_unreserved_per_rank_bytes"),
            "snapshot_decode_unreserved_per_rank_bytes": snapshot.get(
                "decode_unreserved_per_rank_bytes"),
            "actual_decode_unreserved_per_rank_bytes": actual.get(
                "decode_unreserved_per_rank_bytes"),
            "actual_wait_ns": actual.get("wait_ns"),
            "actual_critical_wait_after_restore_ns": actual.get(
                "critical_wait_after_restore_ns"),
        })
    waits = [int(row["actual_wait_ns"]) for row in joined]
    critical_waits = [
        int(row["actual_critical_wait_after_restore_ns"])
        for row in joined
    ]
    return {
        "performed": True,
        "passed": not errors,
        "errors": errors,
        "semantics": (
            "capacity_headroom_snapshot_is_observation_not_reservation; "
            "a_positive_actual_wait_is_valid"),
        "partial_snapshot_count": len(partial_events),
        "pd_chunk_event_count": len(chunk_events),
        "joined_count": len(joined),
        "waiting_count": sum(value > 0 for value in waits),
        "actual_wait_ns": sum(waits),
        "actual_critical_wait_after_restore_ns": sum(critical_waits),
        "max_actual_wait_ns": max(waits, default=0),
        "samples": joined[:32],
    }


def _pd_chunk_admission_audit(
        events, totals, reported_audit, *, schema_version=19):
    """Reconcile atomic P/D chunk attempts with exact counters."""
    rows = [
        event for event in events
        if event.get("event") == "pd_chunk_admission"
    ]
    cancelled_rows = [
        event for event in events
        if event.get("event") == (
            "pd_chunk_admission_cancelled_for_active_prefill_recompute")
    ]
    errors = []
    for index, event in enumerate(rows):
        prefix = f"pd_chunk_admission[{index}]"
        integer_fields = (
            "request_id", "prefill_instance_id", "decode_instance_id",
            "computed_tokens", "chunk_tokens", "target_tokens",
            "prefill_current_per_rank_bytes",
            "decode_current_per_rank_bytes",
            "prefill_target_per_rank_bytes",
            "decode_target_per_rank_bytes",
            "prefill_delta_per_rank_bytes", "decode_delta_per_rank_bytes",
            "prefill_unreserved_per_rank_bytes",
            "decode_unreserved_per_rank_bytes", "enqueued_ns",
            "prefill_capacity_ready_ns", "decode_capacity_ready_ns",
            "admitted_ns", "wait_ns", "critical_wait_after_restore_ns",
            "prefill_delta_bytes", "decode_delta_bytes", "restore_ready_ns",
            "time_ns",
        )
        values = {}
        for field in integer_fields:
            value = event.get(field)
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 0):
                errors.append(f"{prefix}.{field}={value!r}")
            else:
                values[field] = value
        if event.get("admission_scope") != (
                "one_prefill_chunk_atomic_pd_claim"):
            errors.append(f"{prefix}.admission_scope is invalid")
        if event.get("admission_semantics") != (
                "policy_independent_authoritative_dispatch_claim"):
            errors.append(f"{prefix}.admission_semantics is invalid")
        if event.get("capacity_headroom_claimed_by_policy") is not False:
            errors.append(f"{prefix} reports a policy capacity claim")
        if all(field in values for field in (
                "prefill_target_per_rank_bytes",
                "decode_target_per_rank_bytes")) and (
                values["prefill_target_per_rank_bytes"]
                != values["decode_target_per_rank_bytes"]):
            errors.append(f"{prefix} has asymmetric P/D target ownership")
        for role in ("prefill", "decode"):
            fields = (
                f"{role}_current_per_rank_bytes",
                f"{role}_delta_per_rank_bytes",
                f"{role}_target_per_rank_bytes",
            )
            if all(field in values for field in fields) and (
                    values[fields[0]] + values[fields[1]]
                    != values[fields[2]]):
                errors.append(f"{prefix} {role} byte identity failed")
        if all(field in values for field in (
                "time_ns", "admitted_ns")) and (
                values["time_ns"] != values["admitted_ns"]):
            errors.append(f"{prefix} time_ns != admitted_ns")
        if all(field in values for field in (
                "enqueued_ns", "admitted_ns", "wait_ns")) and (
                values["wait_ns"]
                != values["admitted_ns"] - values["enqueued_ns"]):
            errors.append(f"{prefix} wait identity failed")
        snapshot = event.get("capacity_headroom_snapshot")
        joined = snapshot is not None
        if event.get("capacity_headroom_snapshot_only") is not joined:
            errors.append(f"{prefix} snapshot-only flag diverged")
        feasible = bool(joined and snapshot.get("feasible") is True)
        if event.get("capacity_snapshot_feasible") is not feasible:
            errors.append(f"{prefix} snapshot-feasible flag diverged")
        feasible_waited = bool(feasible and event.get("wait_ns", 0) > 0)
        if event.get("snapshot_feasible_but_actual_waited") is not feasible_waited:
            errors.append(f"{prefix} feasible/waited flag diverged")
        if joined:
            if event.get("first_chunk") is not True:
                errors.append(f"{prefix} attaches a snapshot after first chunk")
            if (event.get("capacity_snapshot_decision_time_ns")
                    != snapshot.get("time_ns")):
                errors.append(f"{prefix} snapshot decision time diverged")
            if all(field in values for field in ("admitted_ns",)):
                expected_delta = (
                    values["admitted_ns"] - int(snapshot.get("time_ns", -1)))
                if event.get("capacity_snapshot_to_admission_ns") != expected_delta:
                    errors.append(f"{prefix} snapshot-to-admission time diverged")
        elif (event.get("capacity_snapshot_decision_time_ns") is not None
              or event.get("capacity_snapshot_to_admission_ns") is not None):
            errors.append(f"{prefix} has snapshot timestamps without snapshot")

    expected = {
        "pd_chunk_admissions": len(rows),
        "pd_chunk_waiting_admissions": sum(
            int(event.get("wait_ns", 0)) > 0 for event in rows),
        "pd_chunk_admitted_tokens": sum(
            int(event.get("chunk_tokens", 0)) for event in rows),
        "pd_chunk_prefill_reserved_bytes": sum(
            int(event.get("prefill_delta_bytes", 0)) for event in rows),
        "pd_chunk_decode_reserved_bytes": sum(
            int(event.get("decode_delta_bytes", 0)) for event in rows),
        "pd_chunk_admission_wait_ns": sum(
            int(event.get("wait_ns", 0)) for event in rows),
        "pd_chunk_admission_critical_wait_ns": sum(
            int(event.get("critical_wait_after_restore_ns", 0))
            for event in rows),
        "pd_chunk_snapshot_joined_admissions": sum(
            event.get("capacity_headroom_snapshot") is not None
            for event in rows),
        "pd_chunk_snapshot_feasible_admissions": sum(
            bool((event.get("capacity_headroom_snapshot") or {}).get(
                "feasible", False)) for event in rows),
        "pd_chunk_snapshot_feasible_waiting_admissions": sum(
            bool((event.get("capacity_headroom_snapshot") or {}).get(
                "feasible", False)) and int(event.get("wait_ns", 0)) > 0
            for event in rows),
        "pd_chunk_snapshot_feasible_wait_ns": sum(
            int(event.get("wait_ns", 0))
            for event in rows
            if bool((event.get("capacity_headroom_snapshot") or {}).get(
                "feasible", False)) and int(event.get("wait_ns", 0)) > 0),
    }
    if schema_version >= 20:
        for index, event in enumerate(cancelled_rows):
            prefix = f"pd_chunk_admission_cancelled[{index}]"
            integer_fields = (
                "request_id", "active_prefill_recompute_generation",
                "enqueued_ns", "cancelled_ns", "wait_ns",
                "critical_wait_after_restore_ns", "time_ns",
            )
            values = {}
            for field in integer_fields:
                value = event.get(field)
                if (not isinstance(value, int) or isinstance(value, bool)
                        or value < 0):
                    errors.append(f"{prefix}.{field}={value!r}")
                else:
                    values[field] = value
            if (event.get("cancelled_by_active_prefill_recompute") is not True
                    or event.get("preempted_before_commit") is not True
                    or event.get("committed") is not False
                    or event.get("admission_semantics")
                    != "cancelled_before_graph_commit"):
                errors.append(f"{prefix} cancellation provenance is invalid")
            if all(field in values for field in (
                    "time_ns", "cancelled_ns")) and (
                    values["time_ns"] != values["cancelled_ns"]):
                errors.append(f"{prefix} time_ns != cancelled_ns")
            if all(field in values for field in (
                    "enqueued_ns", "cancelled_ns", "wait_ns")) and (
                    values["cancelled_ns"] - values["enqueued_ns"]
                    != values["wait_ns"]):
                errors.append(f"{prefix} wait identity failed")
        expected.update({
            "pd_chunk_cancelled_admissions": len(cancelled_rows),
            "pd_chunk_cancelled_waiting_admissions": sum(
                int(event.get("wait_ns", 0)) > 0
                for event in cancelled_rows),
            "pd_chunk_cancelled_admission_wait_ns": sum(
                int(event.get("wait_ns", 0)) for event in cancelled_rows),
            "pd_chunk_cancelled_admission_critical_wait_ns": sum(
                int(event.get("critical_wait_after_restore_ns", 0))
                for event in cancelled_rows),
        })
    counter_mismatches = {
        field: {"observed": totals.get(field), "expected": value}
        for field, value in expected.items()
        if (not isinstance(totals.get(field), int)
            or isinstance(totals.get(field), bool)
            or int(totals.get(field)) != value)
    }
    if counter_mismatches:
        errors.append(f"counter mismatches={counter_mismatches}")
    first_chunks = sum(event.get("first_chunk") is True for event in rows)
    reported_counts = {
        "chunk_admissions": len(rows),
        "first_chunk_admissions": first_chunks,
        "snapshot_joined_first_chunks": expected[
            "pd_chunk_snapshot_joined_admissions"],
        "snapshot_feasible_first_chunks": expected[
            "pd_chunk_snapshot_feasible_admissions"],
        "snapshot_feasible_waiting_first_chunks": expected[
            "pd_chunk_snapshot_feasible_waiting_admissions"],
    }
    if schema_version >= 20:
        reported_counts["cancelled_chunk_admissions"] = len(cancelled_rows)
    if (not isinstance(reported_audit, dict)
            or reported_audit.get("status") != "ok"
            or not isinstance(reported_audit.get("checks"), dict)
            or not reported_audit.get("checks")
            or not all(value is True
                       for value in reported_audit["checks"].values())):
        errors.append(
            f"reported pd_chunk_accounting audit is invalid: "
            f"{reported_audit!r}")
    else:
        reported_mismatches = {
            field: {
                "observed": reported_audit.get(field),
                "expected": value,
            }
            for field, value in reported_counts.items()
            if reported_audit.get(field) != value
        }
        if reported_mismatches:
            errors.append(
                f"reported pd_chunk_accounting counts diverge: "
                f"{reported_mismatches}")
    return {
        "performed": True,
        "passed": not errors,
        "errors": errors,
        "event_count": len(rows),
        "cancelled_event_count": len(cancelled_rows),
        "counters": expected,
        "semantics": "atomic_policy_independent_P_D_chunk_claim",
    }


def _pd_active_prefill_recompute_audit(
        events, totals, reported_audit):
    """Reconcile schema-20 active-prefill replay and restored-hit loss."""
    rows = [
        event for event in events
        if event.get("event") == "pd_active_prefill_recompute_preempt"
    ]
    errors = []
    generation_by_request = {}
    tokens_by_request = {}
    restored_by_request = {}
    for index, event in enumerate(rows):
        prefix = f"pd_active_prefill_recompute_preempt[{index}]"
        fields = (
            "request_id", "discarded_tokens",
            "restored_hit_tokens_discarded",
            "cumulative_active_prefill_recompute_tokens",
            "cumulative_restored_hit_tokens_discarded",
            "old_active_prefill_recompute_generation",
            "new_active_prefill_recompute_generation",
        )
        values = {}
        for field in fields:
            value = event.get(field)
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 0):
                errors.append(f"{prefix}.{field}={value!r}")
            else:
                values[field] = value
        if len(values) != len(fields):
            continue
        request_id = values["request_id"]
        old_generation = values[
            "old_active_prefill_recompute_generation"]
        new_generation = values[
            "new_active_prefill_recompute_generation"]
        expected_generation = generation_by_request.get(request_id, 0)
        if (old_generation != expected_generation
                or new_generation != old_generation + 1):
            errors.append(f"{prefix} generation transition is invalid")
        discarded = values["discarded_tokens"]
        restored_delta = values["restored_hit_tokens_discarded"]
        if restored_delta > discarded or (
                old_generation > 0 and restored_delta != 0):
            errors.append(f"{prefix} restored-hit delta is invalid")
        expected_tokens = tokens_by_request.get(request_id, 0) + discarded
        expected_restored = (
            restored_by_request.get(request_id, 0) + restored_delta)
        if (values["cumulative_active_prefill_recompute_tokens"]
                != expected_tokens):
            errors.append(f"{prefix} cumulative replay tokens diverge")
        if (values["cumulative_restored_hit_tokens_discarded"]
                != expected_restored):
            errors.append(f"{prefix} cumulative restored hits diverge")
        generation_by_request[request_id] = new_generation
        tokens_by_request[request_id] = expected_tokens
        restored_by_request[request_id] = expected_restored

    expected = {
        "pd_active_prefill_recompute_preemptions": len(rows),
        "pd_active_prefill_recompute_tokens": sum(
            int(event.get("discarded_tokens", 0)) for event in rows),
        "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute": sum(
            int(event.get("restored_hit_tokens_discarded", 0))
            for event in rows),
    }
    mismatches = {
        field: {"observed": totals.get(field), "expected": expected_value}
        for field, expected_value in expected.items()
        if (not isinstance(totals.get(field), int)
            or isinstance(totals.get(field), bool)
            or int(totals.get(field)) != expected_value)
    }
    if mismatches:
        errors.append(f"counter mismatches={mismatches}")
    reported_counts = {
        "preemptions": len(rows),
        "discarded_tokens": expected[
            "pd_active_prefill_recompute_tokens"],
        "restored_hit_tokens_discarded": expected[
            "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute"],
    }
    if (not isinstance(reported_audit, dict)
            or reported_audit.get("status") != "ok"
            or not isinstance(reported_audit.get("checks"), dict)
            or not reported_audit.get("checks")
            or not all(reported_audit["checks"].values())):
        errors.append(
            "reported active-prefill recompute audit is invalid: "
            f"{reported_audit!r}")
    else:
        for field, expected_value in reported_counts.items():
            if reported_audit.get(field) != expected_value:
                errors.append(
                    f"reported active-prefill {field}="
                    f"{reported_audit.get(field)!r}, "
                    f"expected={expected_value}")
    return {
        "performed": True,
        "passed": not errors,
        "errors": errors,
        "event_count": len(rows),
        "counters": expected,
        "semantics": (
            "physical source provenance retained; restored hits charged "
            "once at first active-prefill preemption"),
    }


def _validate_policy_invariants(manifest, report, agentic_report):
    expected_policy = manifest.get("expected_agentic_policy")
    if expected_policy is None:
        if int(manifest.get("schema_version", 0) or 0) >= 7:
            raise ExperimentError(
                f"Schema-7 manifest {manifest.get('run_id')} is missing "
                "expected_agentic_policy")
        return {
            "performed": False,
            "reason": "manifest_has_no_expected_agentic_policy",
        }
    run_id = manifest["run_id"]
    if int(manifest.get("schema_version", 0) or 0) >= SCHEMA_VERSION:
        agentic_schema_version = agentic_report.get("schema_version")
        if (not isinstance(agentic_schema_version, int)
                or isinstance(agentic_schema_version, bool)
                or agentic_schema_version
                < MIN_CURRENT_AGENTIC_REPORT_SCHEMA_VERSION):
            raise ExperimentError(
                f"Online manifest schema {SCHEMA_VERSION} requires agentic "
                "KV report schema >= "
                f"{MIN_CURRENT_AGENTIC_REPORT_SCHEMA_VERSION} for "
                f"{run_id}; observed={agentic_schema_version!r}")
    effective_policy = str(agentic_report.get("policy"))
    if effective_policy != str(expected_policy):
        raise ExperimentError(
            f"Effective policy mismatch for {run_id}: "
            f"observed={effective_policy}, expected={expected_policy}")
    config = agentic_report.get("config") or {}
    expected_config = {
        "policy": expected_policy,
        "pressure_policy": "lru-drop",
        "demotion_mode": "capacity-only",
        "swap_execution_mode": "async-pre-admission",
        "active_preemption_mode": "recompute",
    }
    mismatches = {
        key: {"observed": config.get(key), "expected": value}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ExperimentError(
            f"Effective baseline config mismatch for {run_id}: {mismatches}")
    if int(manifest.get("schema_version", 0) or 0) >= 7:
        expected_effective_hash = manifest.get(
            "agentic_effective_config_hash")
        if not _is_sha256_digest(expected_effective_hash):
            raise ExperimentError(
                f"Missing or malformed effective agentic config hash for "
                f"{run_id}")
        observed_effective_hash = _stable_json_hash(
            _canonical_agentic_config_payload(config))
        if observed_effective_hash != expected_effective_hash:
            raise ExperimentError(
                f"Runtime effective agentic config hash mismatch for "
                f"{run_id}: observed={observed_effective_hash}, "
                f"expected={expected_effective_hash}")

    totals = agentic_report.get("totals") or {}
    durable_capacity_contract = manifest.get("durable_capacity_contract")
    if expected_policy in DURABLE_CAPACITY_POLICIES:
        durable_capacity_contract = str(
            durable_capacity_contract or "terminal-ssd-lru")
        if durable_capacity_contract not in DURABLE_CAPACITY_CONTRACTS:
            raise ExperimentError(
                f"Unsupported durable capacity contract for {run_id}: "
                f"{durable_capacity_contract!r}")
        durable_semantic_zero_fields = (
            "background_cancelled_jobs",
            "background_cancelled_bytes",
            "background_wasted_bytes",
            "ssd_demotion_cancelled",
            "ssd_cancelled_host_write_bytes",
            "hbm_capacity_drops",
            "ttl_drops",
        )
        _require_present_zero_totals(
            run_id, totals, durable_semantic_zero_fields)
        migration_cancellations = [
            event for event in (agentic_report.get("events") or [])
            if event.get("event") == "migration_cancel"
        ]
        if migration_cancellations:
            raise ExperimentError(
                f"Durable capacity policy cancelled migration(s) for "
                f"{run_id}: {migration_cancellations[:3]}")
    elif durable_capacity_contract is not None:
        raise ExperimentError(
            f"Non-durable policy {expected_policy!r} carries a durable "
            f"capacity contract for {run_id}")
    external_fabric_jobs = 0
    external_resume_evidence = {
        "resume_count": 0,
        "matched_job_ids": [],
    }
    external_fabric_model_coexecution = {
        "performed": False,
        "scope": "source_or_target_instance_model_window",
        "coexecution_pair_count": 0,
        "overlapped_job_count": 0,
        "overlapped_model_window_count": 0,
        "coexecution_membership_ns": 0,
        "coexecution_union_ns": 0,
        "samples": [],
        "interpretation": "No external congestion-aware fabric was enabled.",
    }
    queue_recompute_validation = {
        "performed": False,
        "reason": "effective_policy_is_not_tiered_queue_recompute",
    }
    pd_chunk_validation = {
        "performed": False,
        "reason": "agentic_report_schema_precedes_atomic_pd_chunk_events",
    }
    pd_active_prefill_validation = {
        "performed": False,
        "reason": "agentic_report_schema_precedes_active_prefill_audit",
    }
    if config.get("pd_peer_transfer_mode") == "direct-fabric":
        external = agentic_report.get("external_fabric")
        if not isinstance(external, dict) or external.get("enabled") is not True:
            raise ExperimentError(
                f"Direct-fabric run lacks an enabled external ASTRA bridge "
                f"for {run_id}")
        authority = external.get("authority") or {}
        if authority.get("backend") != "analytical-congestion-aware":
            raise ExperimentError(
                f"Direct-fabric authority is not congestion-aware ASTRA for "
                f"{run_id}: {authority}")
        if authority.get("bandwidth_unit") != "decimal_GBps":
            raise ExperimentError(
                f"Direct-fabric authority lacks a decimal GB/s contract for "
                f"{run_id}: {authority}")
        fabric_bandwidth_gbps = float(
            authority.get("bandwidth_gbps", 0) or 0)
        fabric_latency_ns = int(authority.get("latency_ns", -1))
        if (not math.isfinite(fabric_bandwidth_gbps)
                or fabric_bandwidth_gbps <= 0 or fabric_latency_ns < 0):
            raise ExperimentError(
                f"Invalid external cold-fabric authority for {run_id}: "
                f"{authority}")
        issued = int(external.get("issued_jobs", -1))
        completed = int(external.get("completed_jobs", -1))
        censored = int(external.get("censored_jobs", -1))
        censored_bytes = int(external.get("censored_lane_bytes", -1))
        pending = int(external.get("pending_jobs", -1))
        pending_sessions = list(external.get("pending_sessions") or ())
        intervals = list(external.get("completed_intervals") or ())
        if (issued < 0 or completed < 0 or censored < 0
                or censored > completed or censored_bytes < 0 or pending != 0
                or pending_sessions or issued != completed
                or completed != len(intervals)):
            raise ExperimentError(
                f"External cold-fabric job lifecycle does not drain for "
                f"{run_id}: issued={issued}, completed={completed}, "
                f"censored={censored}, censored_bytes={censored_bytes}, "
                f"pending={pending}, pending_sessions={pending_sessions}, "
                f"intervals={len(intervals)}")
        job_ids = []
        interval_bytes = 0
        for interval in intervals:
            job_id = str(interval.get("job_id") or "")
            arrival_ns = int(interval.get("arrival_ns", -1))
            start_ns = int(interval.get("start_ns", -1))
            complete_ns = int(interval.get("complete_ns", -1))
            queue_ns = int(interval.get("queue_wait_ns", -1))
            service_ns = int(interval.get("service_ns", -1))
            bytes_per_lane = int(interval.get("bytes_per_lane", -1))
            lane_count = int(interval.get("lane_count", -1))
            num_bytes = int(interval.get("bytes", -1))
            if (not job_id or arrival_ns < 0
                    or not arrival_ns <= start_ns <= complete_ns
                    or queue_ns != start_ns - arrival_ns
                    or service_ns != complete_ns - start_ns
                    or bytes_per_lane <= 0 or lane_count <= 0
                    or num_bytes != bytes_per_lane * lane_count):
                raise ExperimentError(
                    f"External cold-fabric interval is non-causal or has an "
                    f"invalid lane layout for {run_id}: {interval}")
            # ASTRA's integer event clock may truncate one fractional ns at
            # the final chunk boundary.  Keep that documented 1 ns tolerance
            # while rejecting any materially super-physical service time.
            uncongested_lower_bound_ns = max(0, (
                math.ceil(
                    bytes_per_lane
                    / (fabric_bandwidth_gbps * 1_000_000_000)
                    * 1_000_000_000)
                + fabric_latency_ns
                - 1
            ))
            if service_ns < uncongested_lower_bound_ns:
                raise ExperimentError(
                    f"External cold-fabric service is shorter than the ASTRA "
                    f"wire lower bound for {run_id}: observed={service_ns}, "
                    f"lower_bound={uncongested_lower_bound_ns}, "
                    f"interval={interval}")
            job_ids.append(job_id)
            interval_bytes += num_bytes
        if len(job_ids) != len(set(job_ids)):
            raise ExperimentError(
                f"External cold-fabric job IDs are not unique for {run_id}")
        counter_bytes = int(
            totals.get("external_fabric_lane_bytes", -1))
        counter_censored = int(
            totals.get("external_fabric_censored_lane_bytes", -1))
        counter_censored_jobs = int(
            totals.get("external_fabric_jobs_censored", -1))
        pd_bytes = int(totals.get("pd_hbm_to_hbm_bytes", -1))
        if (counter_bytes != interval_bytes
                or counter_censored != censored_bytes
                or counter_censored_jobs != censored
                or counter_bytes != pd_bytes + censored_bytes):
            raise ExperimentError(
                f"External cold-fabric byte counters do not reconcile for "
                f"{run_id}: intervals={interval_bytes}, "
                f"external_counter={counter_bytes}, pd_counter={pd_bytes}, "
                f"censored={censored_bytes}, "
                f"censored_jobs={censored}")
        if (manifest.get("require_complete_session_cohort")
                and completed - censored
                != int(totals.get("hbm_hits", -1))):
            raise ExperimentError(
                f"P/D external jobs do not match HBM resumes for {run_id}: "
                f"completed={completed}, censored={censored}, "
                f"hbm_hits={totals.get('hbm_hits')}")
        external_resume_evidence = _validate_external_resume_evidence(
            run_id, agentic_report, intervals)
        if external_resume_evidence["resume_count"] != completed - censored:
            raise ExperimentError(
                "External ASTRA jobs do not map one-to-one to completed "
                f"cross-instance HBM resumes for {run_id}: resumes="
                f"{external_resume_evidence['resume_count']}, completed="
                f"{completed}, censored={censored}")
        external_fabric_jobs = completed
        external_fabric_model_coexecution = (
            _external_fabric_model_coexecution_audit(
                agentic_report, intervals))
    if expected_policy == "hbm_lru_recompute":
        _require_zero_totals(run_id, totals, (
            "cpu_hits", "ssd_hits", "hbm_capacity_demotions",
            "hbm_to_cpu_bytes", "cpu_to_hbm_bytes", "cpu_to_ssd_bytes",
            "hbm_to_ssd_bytes", "ssd_to_hbm_bytes",
            "ssd_to_cpu_stage_bytes", "cpu_stage_to_hbm_bytes",
            "ssd_host_write_bytes", "ssd_host_read_bytes",
            "direct_ssd_write_bytes", "direct_ssd_read_bytes",
        ))
        invalid_sources = sorted({
            str(record.get("agentic_kv_source"))
            for record in report.get("requests", {}).get("records", [])
            if int(record.get("sub_request_index", 0)) > 0
            and str(record.get("agentic_kv_source")) not in {"hbm", "dropped"}
        })
        if invalid_sources:
            raise ExperimentError(
                f"HBM-only baseline observed lower-tier resume sources for "
                f"{run_id}: {invalid_sources}")
    elif expected_policy == "hbm_ssd_direct":
        _require_zero_totals(run_id, totals, (
            "cpu_hits", "hbm_to_cpu_bytes", "cpu_to_hbm_bytes",
            "cpu_to_ssd_bytes", "direct_ssd_read_bytes",
        ))
        invalid_sources = sorted({
            str(record.get("agentic_kv_source"))
            for record in report.get("requests", {}).get("records", [])
            if int(record.get("sub_request_index", 0)) > 0
            and str(record.get("agentic_kv_source"))
            not in {"hbm", "ssd", "dropped"}
        })
        if invalid_sources:
            raise ExperimentError(
                f"SSD-direct baseline observed forbidden resume sources for "
                f"{run_id}: {invalid_sources}")
    elif expected_policy == "tiered":
        _require_zero_totals(run_id, totals, (
            "direct_ssd_write_bytes", "direct_ssd_read_bytes",
        ))
    elif expected_policy == "tiered_queue_recompute":
        _require_present_zero_totals(run_id, totals, (
            "background_cancelled_jobs",
            "background_cancelled_bytes",
            "background_wasted_bytes",
            "ssd_demotion_cancelled",
            "ssd_cancelled_host_write_bytes",
            "hbm_capacity_drops",
            "ttl_drops",
            "direct_ssd_write_bytes",
            "direct_ssd_read_bytes",
        ))
        queue_report = agentic_report.get("queue_recompute_policy") or {}
        if queue_report.get("enabled") is not True:
            raise ExperimentError(
                f"Queue-recompute baseline is not enabled for {run_id}")
        ratio = float(config.get(
            "queue_recompute_wait_service_ratio", -1))
        if (not math.isfinite(ratio) or ratio < 0
                or float(queue_report.get(
                    "configured_wait_service_ratio", -1)) != ratio):
            raise ExperimentError(
                f"Queue-recompute ratio is invalid for {run_id}: "
                f"config={ratio}, report={queue_report}")
        min_wait_ms = float(config.get(
            "queue_recompute_min_wait_ms", 0))
        cost_multiplier = float(config.get(
            "queue_recompute_cost_guard_multiplier", 0))
        headroom_chunks = float(config.get(
            "queue_recompute_prefill_headroom_chunks", -1))
        if (not math.isfinite(min_wait_ms) or min_wait_ms < 0
                or not math.isfinite(cost_multiplier)
                or (cost_multiplier != 0 and cost_multiplier < 1)
                or not math.isfinite(headroom_chunks)
                or headroom_chunks < 1
                or int(queue_report.get("configured_min_wait_ns", -1))
                != int(math.ceil(min_wait_ms * 1_000_000))
                or float(queue_report.get(
                    "configured_cost_guard_multiplier", -1))
                != cost_multiplier
                or float(queue_report.get(
                    "configured_prefill_headroom_chunks", -1))
                != headroom_chunks
                or queue_report.get("headroom_semantics")
                != "causal_unreserved_P_and_D_snapshot_not_reservation"
                or queue_report.get("headroom_owner")
                != "ordinary_atomic_pd_chunk_admission"):
            raise ExperimentError(
                f"Queue-recompute severe/cost gates are invalid for "
                f"{run_id}: config={config}, report={queue_report}")
        events = agentic_report.get("events") or []
        evaluations = [
            event for event in events
            if event.get("event") == "queue_recompute_evaluate"
        ]
        partial_decisions = [
            event for event in events
            if event.get("event") == "queue_recompute_partial"
        ]
        zero_decisions = [
            event for event in events
            if event.get("event") == "queue_recompute_drop"
        ]
        queue_drops = [
            event for event in events
            if (event.get("event") == "drop"
                and event.get("reason") == "queue_pressure")
        ]
        min_wait_ns = int(math.ceil(min_wait_ms * 1_000_000))
        block_size = int(config.get("block_size", 0) or 0)
        if block_size <= 0:
            raise ExperimentError(
                f"Queue-recompute block size is invalid for {run_id}: "
                f"{block_size}")
        projection_errors = []
        evaluations_by_kind = {
            "full": [
                event for event in evaluations
                if event.get("decision") == "restore"],
            "partial": [
                event for event in evaluations
                if event.get("decision")
                == "partial_restore_suffix_recompute"],
            "zero": [
                event for event in evaluations
                if event.get("decision") == "drop_recompute"],
        }
        unknown_evaluations = [
            event for event in evaluations
            if event.get("decision") not in {
                "restore", "partial_restore_suffix_recompute",
                "drop_recompute",
            }
        ]
        if unknown_evaluations:
            projection_errors.append(
                "evaluation has unknown decision values: "
                f"{unknown_evaluations[:3]}")
        for decision_kind, projection_events in evaluations_by_kind.items():
            for index, event in enumerate(projection_events):
                for error in _queue_recompute_projection_errors(
                        event,
                        ratio=ratio,
                        min_wait_ns=min_wait_ns,
                        cost_multiplier=cost_multiplier,
                        block_size=block_size,
                        decision_kind=decision_kind,
                        selection_event=False):
                    projection_errors.append(
                        f"evaluation.{decision_kind}[{index}]: {error}")
        for decision_kind, selection_events in (
                ("partial", partial_decisions),
                ("zero", zero_decisions)):
            for index, event in enumerate(selection_events):
                for error in _queue_recompute_projection_errors(
                        event,
                        ratio=ratio,
                        min_wait_ns=min_wait_ns,
                        cost_multiplier=cost_multiplier,
                        block_size=block_size,
                        decision_kind=decision_kind,
                        selection_event=True):
                    projection_errors.append(
                        f"selection.{decision_kind}[{index}]: {error}")
        projection_pair_fields = (
            "time_ns", "session_id", "source", "transfer_kinds",
            "bytes", "reusable_tokens_R", "selected_prefix_tokens_H",
            "selected_prefix_block_tokens", "dropped_suffix_tokens",
            "selected_restore_bytes", "dropped_suffix_bytes",
            "avoided_restore_bytes",
            "physical_entry_dropped_bytes", "projection_arrival_ns",
            "projection_available",
            "projection_available_without_new_lru_work",
            "projection_includes_collateral_lru_work",
            "projected_hbm_victim_sessions",
            "projected_cpu_victim_sessions",
            "projection_precedes_destination_hbm_reservation",
            "projected_hbm_admission_wait_ns",
            "projected_transient_dram_capacity_wait_ns",
            "projected_queue_wait_ns", "projected_total_wait_ns",
            "projected_service_ns", "projected_restore_ns",
            "estimated_incremental_recompute_comp_ns",
            "estimated_suffix_recompute_comp_ns",
            "selected_predicted_resume_path_ns",
            "full_predicted_resume_path_ns", "candidate_prefix_tokens",
            "full_projection_status", "prefix_projection_available",
            "prefix_projected_hbm_admission_wait_ns",
            "prefix_projected_transient_dram_capacity_wait_ns",
            "prefix_projected_queue_wait_ns",
            "prefix_projected_service_ns", "capacity_headroom_snapshot",
            "capacity_headroom_snapshot_only",
            "capacity_headroom_claimed_by_policy",
            "pd_first_chunk_immediate_admission_guaranteed",
            "configured_wait_service_ratio", "configured_min_wait_ns",
            "configured_cost_guard_multiplier", "ratio_threshold_ns",
            "threshold_ns", "cost_threshold_ns", "severe_gate_pass",
            "cost_gate_pass",
        )
        for decision_kind, selection_events in (
                ("partial", partial_decisions),
                ("zero", zero_decisions)):
            selected_evaluations = evaluations_by_kind[decision_kind]
            evaluation_by_key = {
                (event.get("session_id"), event.get("time_ns")): event
                for event in selected_evaluations
            }
            selection_by_key = {
                (event.get("session_id"), event.get("time_ns")): event
                for event in selection_events
            }
            if (len(evaluation_by_key) != len(selected_evaluations)
                    or len(selection_by_key) != len(selection_events)
                    or set(evaluation_by_key) != set(selection_by_key)):
                projection_errors.append(
                    f"{decision_kind} evaluate/selection pairing differs: "
                    f"evaluations={sorted(evaluation_by_key)}, "
                    f"selections={sorted(selection_by_key)}")
            for key in sorted(set(evaluation_by_key) & set(selection_by_key)):
                evaluation = evaluation_by_key[key]
                decision = selection_by_key[key]
                victim_pair_fields = {
                    field for field in {*evaluation, *decision}
                    if (field.startswith("projected_")
                        and field.endswith("_victim_sessions"))
                }
                divergent_fields = [
                    field for field in (
                        *projection_pair_fields,
                        *sorted(victim_pair_fields))
                    if evaluation.get(field) != decision.get(field)
                ]
                if divergent_fields:
                    projection_errors.append(
                        f"{decision_kind} evaluate/selection {key} diverge "
                        f"in {divergent_fields}")
        decision_drop_keys = [
            (event.get("session_id"), event.get("time_ns"))
            for event in zero_decisions
        ]
        queue_drop_keys = [
            (event.get("session_id"), event.get("time_ns"))
            for event in queue_drops
        ]
        if decision_drop_keys != queue_drop_keys:
            projection_errors.append(
                "queue-pressure drop events do not pair with selected "
                f"projections: decisions={decision_drop_keys}, "
                f"drops={queue_drop_keys}")
        invalid_queue_drops = [
            event for event in queue_drops
            if (event.get("drop_class") != "policy_loss"
                or event.get("object_scope") != "kv_cache_entry"
                or event.get("logical_session_effect") != "none")
        ]
        if invalid_queue_drops:
            projection_errors.append(
                "queue-pressure drop event has invalid object semantics: "
                f"{invalid_queue_drops[:3]}")
        modified_decisions = partial_decisions + zero_decisions
        partial_cpu_decisions = sum(
            event.get("source") == "cpu" for event in partial_decisions)
        partial_ssd_decisions = sum(
            event.get("source") == "ssd" for event in partial_decisions)
        cpu_decisions = sum(
            event.get("source") == "cpu" for event in zero_decisions)
        ssd_decisions = sum(
            event.get("source") == "ssd" for event in zero_decisions)
        reconciliations = {
            "queue_recompute_evaluation_attempts": len(evaluations),
            "queue_recompute_severe_gate_passes": sum(
                bool(event.get("severe_gate_pass"))
                for event in evaluations),
            "queue_recompute_cost_gate_passes": sum(
                bool(event.get("cost_gate_pass"))
                for event in evaluations),
            "queue_recompute_full_restore_decisions": len(
                evaluations_by_kind["full"]),
            "queue_recompute_partial_restore_decisions": len(
                partial_decisions),
            "queue_recompute_zero_restore_decisions": len(zero_decisions),
            "queue_recompute_partial_cpu_decisions": partial_cpu_decisions,
            "queue_recompute_partial_ssd_decisions": partial_ssd_decisions,
            "queue_recompute_drop_decisions": len(zero_decisions),
            "queue_recompute_cpu_drop_decisions": cpu_decisions,
            "queue_recompute_ssd_drop_decisions": ssd_decisions,
            "queue_recompute_dropped_bytes": sum(
                int(event.get("dropped_suffix_bytes", 0))
                for event in modified_decisions),
            "queue_recompute_avoided_restore_bytes": sum(
                int(event.get("avoided_restore_bytes", 0))
                for event in modified_decisions),
            "queue_recompute_physical_entry_dropped_bytes": sum(
                int(event.get("physical_entry_dropped_bytes", 0))
                for event in zero_decisions),
            "queue_recompute_projected_queue_wait_ns": sum(
                int(event.get("projected_queue_wait_ns", 0))
                for event in modified_decisions),
            "queue_recompute_projected_hbm_admission_wait_ns": sum(
                int(event.get(
                    "projected_hbm_admission_wait_ns", 0))
                for event in modified_decisions),
            "queue_recompute_projected_transient_dram_capacity_wait_ns": sum(
                int(event.get(
                    "projected_transient_dram_capacity_wait_ns", 0))
                for event in modified_decisions),
            "queue_recompute_projected_service_ns": sum(
                int(event.get("projected_service_ns", 0))
                for event in modified_decisions),
            "queue_recompute_prefix_projected_queue_wait_ns": sum(
                int(event.get("prefix_projected_queue_wait_ns", 0))
                for event in partial_decisions),
            "queue_recompute_prefix_projected_hbm_admission_wait_ns": sum(
                int(event.get(
                    "prefix_projected_hbm_admission_wait_ns", 0))
                for event in partial_decisions),
            "queue_recompute_prefix_projected_transient_dram_capacity_wait_ns": sum(
                int(event.get(
                    "prefix_projected_transient_dram_capacity_wait_ns", 0))
                for event in partial_decisions),
            "queue_recompute_prefix_projected_service_ns": sum(
                int(event.get("prefix_projected_service_ns", 0))
                for event in partial_decisions),
            "queue_recompute_estimated_recompute_ns": sum(
                int(event.get(
                    "estimated_suffix_recompute_comp_ns") or 0)
                for event in modified_decisions),
            "queue_recompute_tokens": sum(
                max(0,
                    int(event.get("declared_reuse_tokens", 0))
                    - int(event.get("selected_prefix_tokens_H", 0)))
                for event in modified_decisions),
            "queue_recompute_policy_avoidable_tokens": sum(
                int(event.get("dropped_suffix_tokens", 0))
                for event in modified_decisions),
            "queue_recompute_selected_restore_tokens": sum(
                int(event.get("selected_prefix_tokens_H", 0))
                for event in modified_decisions),
            "queue_recompute_dropped_suffix_tokens": sum(
                int(event.get("dropped_suffix_tokens", 0))
                for event in modified_decisions),
            "queue_recompute_selected_restore_bytes": sum(
                int(event.get("selected_restore_bytes", 0))
                for event in modified_decisions),
            "queue_recompute_dropped_suffix_bytes": sum(
                int(event.get("dropped_suffix_bytes", 0))
                for event in modified_decisions),
        }
        mismatches = {
            field: {
                "observed": totals.get(field),
                "expected": expected,
            }
            for field, expected in reconciliations.items()
            if int(totals.get(field, -1)) != expected
        }
        selected_total_wait_ns = (
            reconciliations["queue_recompute_projected_queue_wait_ns"]
            + reconciliations[
                "queue_recompute_projected_hbm_admission_wait_ns"])
        event_total_wait_ns = sum(
            int(event.get("projected_total_wait_ns", 0))
            for event in modified_decisions)
        if selected_total_wait_ns != event_total_wait_ns:
            projection_errors.append(
                "selected projected total wait does not equal event "
                f"components: components={selected_total_wait_ns}, "
                f"events={event_total_wait_ns}")
        report_reconciliations = {
            "evaluation_attempts": len(evaluations),
            "severe_gate_passes": reconciliations[
                "queue_recompute_severe_gate_passes"],
            "cost_gate_passes": reconciliations[
                "queue_recompute_cost_gate_passes"],
            "full_restore_decisions": len(evaluations_by_kind["full"]),
            "partial_restore_decisions": len(partial_decisions),
            "zero_restore_decisions": len(zero_decisions),
            "partial_cpu_decisions": partial_cpu_decisions,
            "partial_ssd_decisions": partial_ssd_decisions,
            "drop_decisions": len(zero_decisions),
            "cpu_drop_decisions": cpu_decisions,
            "ssd_drop_decisions": ssd_decisions,
            "dropped_bytes": reconciliations[
                "queue_recompute_dropped_bytes"],
            "avoided_restore_bytes": reconciliations[
                "queue_recompute_avoided_restore_bytes"],
            "physical_entry_dropped_bytes": reconciliations[
                "queue_recompute_physical_entry_dropped_bytes"],
            "declared_recompute_tokens": reconciliations[
                "queue_recompute_tokens"],
            "policy_avoidable_recompute_tokens": reconciliations[
                "queue_recompute_policy_avoidable_tokens"],
            "selected_restore_tokens": reconciliations[
                "queue_recompute_selected_restore_tokens"],
            "dropped_suffix_tokens": reconciliations[
                "queue_recompute_dropped_suffix_tokens"],
            "selected_restore_bytes": reconciliations[
                "queue_recompute_selected_restore_bytes"],
            "dropped_suffix_bytes": reconciliations[
                "queue_recompute_dropped_suffix_bytes"],
            "modified_full_projected_queue_wait_ns": reconciliations[
                "queue_recompute_projected_queue_wait_ns"],
            "modified_full_projected_hbm_admission_wait_ns": reconciliations[
                "queue_recompute_projected_hbm_admission_wait_ns"],
            "modified_full_projected_transient_dram_capacity_wait_ns": (
                reconciliations[
                    "queue_recompute_projected_transient_dram_capacity_wait_ns"]),
            "modified_full_projected_total_wait_ns": selected_total_wait_ns,
            "modified_full_projected_service_ns": reconciliations[
                "queue_recompute_projected_service_ns"],
            "selected_projected_queue_wait_ns": reconciliations[
                "queue_recompute_projected_queue_wait_ns"],
            "selected_projected_hbm_admission_wait_ns": reconciliations[
                "queue_recompute_projected_hbm_admission_wait_ns"],
            "selected_projected_transient_dram_capacity_wait_ns": (
                reconciliations[
                    "queue_recompute_projected_transient_dram_capacity_wait_ns"]),
            "selected_projected_total_wait_ns": selected_total_wait_ns,
            "selected_projected_service_ns": reconciliations[
                "queue_recompute_projected_service_ns"],
            "partial_prefix_projected_queue_wait_ns": reconciliations[
                "queue_recompute_prefix_projected_queue_wait_ns"],
            "partial_prefix_projected_hbm_admission_wait_ns": reconciliations[
                "queue_recompute_prefix_projected_hbm_admission_wait_ns"],
            "partial_prefix_projected_transient_dram_capacity_wait_ns": (
                reconciliations[
                    "queue_recompute_prefix_projected_transient_dram_capacity_wait_ns"]),
            "partial_prefix_projected_service_ns": reconciliations[
                "queue_recompute_prefix_projected_service_ns"],
            "selected_estimated_suffix_recompute_comp_ns": (
                reconciliations[
                    "queue_recompute_estimated_recompute_ns"]),
        }
        report_mismatches = {
            field: {
                "observed": queue_report.get(field),
                "expected": expected,
            }
            for field, expected in report_reconciliations.items()
            if (not isinstance(queue_report.get(field), int)
                or isinstance(queue_report.get(field), bool)
                or int(queue_report.get(field)) != expected)
        }
        accounting_invariants = queue_report.get("accounting_invariants")
        expected_accounting = {
            "passed": True,
            "errors": [],
            "evaluation_events": len(evaluations),
            "partial_events": len(partial_decisions),
            "zero_restore_events": len(zero_decisions),
            "block_size_tokens": block_size,
            "logical_session_drop_count": 0,
            "headroom_semantics": "causal_snapshot_not_reservation",
        }
        if not isinstance(accounting_invariants, dict):
            projection_errors.append("missing report accounting_invariants")
        else:
            invariant_mismatches = {
                field: {
                    "observed": accounting_invariants.get(field),
                    "expected": expected,
                }
                for field, expected in expected_accounting.items()
                if accounting_invariants.get(field) != expected
            }
            if invariant_mismatches:
                projection_errors.append(
                    "report accounting_invariants diverge: "
                    f"{invariant_mismatches}")
        logical_session_drop_count = (report.get("session_admission") or {}).get(
            "logical_session_drop_count")
        if (int(manifest.get("schema_version", 0) or 0) >= SCHEMA_VERSION
                and logical_session_drop_count != 0):
            projection_errors.append(
                "queue-recompute run lacks an explicit zero logical-session "
                f"drop count: {logical_session_drop_count!r}")
        first_chunk_audit = _queue_recompute_first_chunk_audit(
            events, partial_decisions)
        if first_chunk_audit.get("performed") and not first_chunk_audit.get(
                "passed", False):
            projection_errors.extend(
                f"snapshot_first_chunk: {error}"
                for error in first_chunk_audit.get("errors", []))
        if int(queue_report.get("pending_restore_commitments", -1)) != 0:
            raise ExperimentError(
                f"Queue-recompute commitments leaked at completion for "
                f"{run_id}: {queue_report}")
        if mismatches or report_mismatches or projection_errors:
            raise ExperimentError(
                f"Queue-recompute accounting is invalid for {run_id}: "
                f"mismatches={mismatches}, "
                f"report_mismatches={report_mismatches}, "
                f"projection_errors={projection_errors[:8]}")
        queue_recompute_validation = {
            "performed": True,
            "passed": True,
            "full_restore_decisions": len(evaluations_by_kind["full"]),
            "partial_restore_decisions": len(partial_decisions),
            "zero_restore_decisions": len(zero_decisions),
            "accounting_invariants": accounting_invariants,
            "snapshot_to_first_chunk": first_chunk_audit,
        }

    if int(agentic_report.get("schema_version", 0) or 0) >= 19:
        agentic_schema_version = int(
            agentic_report.get("schema_version", 0) or 0)
        pd_chunk_validation = _pd_chunk_admission_audit(
            agentic_report.get("events") or [], totals,
            agentic_report.get("pd_chunk_accounting"),
            schema_version=agentic_schema_version)
        if not pd_chunk_validation["passed"]:
            raise ExperimentError(
                f"P/D chunk-admission accounting is invalid for {run_id}: "
                f"{pd_chunk_validation['errors'][:8]}")
        if agentic_schema_version >= 20:
            pd_active_prefill_validation = (
                _pd_active_prefill_recompute_audit(
                    agentic_report.get("events") or [], totals,
                    agentic_report.get(
                        "pd_active_prefill_recompute_accounting")))
            if not pd_active_prefill_validation["passed"]:
                raise ExperimentError(
                    "P/D active-prefill accounting is invalid for "
                    f"{run_id}: "
                    f"{pd_active_prefill_validation['errors'][:8]}")

    queue_counter_fields = (
        "queue_recompute_evaluation_attempts",
        "queue_recompute_severe_gate_passes",
        "queue_recompute_cost_gate_passes",
        "queue_recompute_full_restore_decisions",
        "queue_recompute_partial_restore_decisions",
        "queue_recompute_zero_restore_decisions",
        "queue_recompute_partial_cpu_decisions",
        "queue_recompute_partial_ssd_decisions",
        "queue_recompute_drop_decisions",
        "queue_recompute_cpu_drop_decisions",
        "queue_recompute_ssd_drop_decisions",
        "queue_recompute_dropped_bytes",
        "queue_recompute_avoided_restore_bytes",
        "queue_recompute_physical_entry_dropped_bytes",
        "queue_recompute_projected_queue_wait_ns",
        "queue_recompute_projected_hbm_admission_wait_ns",
        "queue_recompute_projected_transient_dram_capacity_wait_ns",
        "queue_recompute_projected_service_ns",
        "queue_recompute_prefix_projected_queue_wait_ns",
        "queue_recompute_prefix_projected_hbm_admission_wait_ns",
        "queue_recompute_prefix_projected_transient_dram_capacity_wait_ns",
        "queue_recompute_prefix_projected_service_ns",
        "queue_recompute_estimated_recompute_ns",
        "queue_recompute_tokens",
        "queue_recompute_policy_avoidable_tokens",
        "queue_recompute_selected_restore_tokens",
        "queue_recompute_dropped_suffix_tokens",
        "queue_recompute_selected_restore_bytes",
        "queue_recompute_dropped_suffix_bytes",
    )
    if expected_policy != "tiered_queue_recompute":
        _require_present_zero_totals(
            run_id, totals, queue_counter_fields)

    if durable_capacity_contract == "lossless-working-set":
        _require_present_zero_totals(run_id, totals, (
            "capacity_drops",
            "ssd_capacity_evictions",
            "ssd_capacity_admission_drops",
            "dropped_misses",
            "capacity_induced_recompute_tokens",
            "policy_avoidable_recompute_tokens",
            "hbf_dropped_recompute_tokens",
            "transient_dram_capacity_oversize",
        ))
        dropped_reusable = [
            {
                "session_id": record.get("session_id"),
                "sub_request_index": record.get("sub_request_index"),
                "prefix_reuse_tokens": record.get("prefix_reuse_tokens"),
            }
            for record in report.get("requests", {}).get("records", [])
            if int(record.get("sub_request_index", 0) or 0) > 0
            and int(record.get("prefix_reuse_tokens", 0) or 0) > 0
            and str(record.get("agentic_kv_source")) == "dropped"
        ]
        if dropped_reusable:
            raise ExperimentError(
                f"Lossless working-set contract observed dropped reusable "
                f"prefixes for {run_id}: {dropped_reusable[:5]}")

    ssd_evidence_count = _validate_ssd_restore_evidence(
        run_id, agentic_report)
    ssd_hits = int(totals.get("ssd_hits", 0) or 0)
    if ssd_evidence_count != ssd_hits:
        raise ExperimentError(
            f"SSD hit/evidence count mismatch for {run_id}: "
            f"events={ssd_evidence_count}, counter={ssd_hits}")
    cpu_evidence_count = _validate_cpu_restore_evidence(
        run_id, agentic_report)
    cpu_hits = int(totals.get("cpu_hits", 0) or 0)
    if cpu_evidence_count != cpu_hits:
        raise ExperimentError(
            f"CPU hit/evidence count mismatch for {run_id}: "
            f"events={cpu_evidence_count}, counter={cpu_hits}")
    staged_bytes = int(totals.get("ssd_to_cpu_stage_bytes", 0) or 0)
    h2d_bytes = int(totals.get("cpu_stage_to_hbm_bytes", 0) or 0)
    host_read_bytes = int(totals.get("ssd_host_read_bytes", 0) or 0)
    logical_ssd_restore_bytes = int(totals.get("ssd_to_hbm_bytes", 0) or 0)
    if not (staged_bytes == h2d_bytes == host_read_bytes
            == logical_ssd_restore_bytes):
        raise ExperimentError(
            f"SSD staged-read byte counters do not reconcile for {run_id}: "
            f"ssd_stage={staged_bytes}, h2d={h2d_bytes}, "
            f"host_read={host_read_bytes}, logical={logical_ssd_restore_bytes}")

    transfer_tail = agentic_report.get("measurement_cutoff_dma_tail")
    if (not isinstance(transfer_tail, dict)
            or "foreground_jobs" not in transfer_tail):
        raise ExperimentError(
            f"Missing measurement-cutoff DMA tail audit for {run_id}")
    if int(transfer_tail.get("foreground_jobs", 0) or 0) != 0:
        raise ExperimentError(
            f"Foreground restore remains outstanding at measurement cutoff "
            f"for {run_id}")

    if manifest.get("strict_oracle"):
        oracle = report.get("strict_infinite_hbm_oracle") or {}
        if oracle.get("enabled") is not True or oracle.get("passed") is not True:
            raise ExperimentError(
                f"Strict oracle proof is missing or failed for {run_id}")
        if oracle.get("violations"):
            raise ExperimentError(
                f"Strict oracle reports violations for {run_id}")
        if any(not cell.get("nonbinding", False)
               for cell in (oracle.get("per_instance") or {}).values()):
            raise ExperimentError(
                f"Strict oracle HBM bound became binding for {run_id}")
        if oracle.get("invalid_resume_sources"):
            raise ExperimentError(
                f"Strict oracle observed a non-HBM reusable resume for {run_id}")
        _require_zero_totals(run_id, totals, (
            "cpu_hits", "ssd_hits", "dropped_misses", "capacity_drops",
            "hbm_capacity_demotions", "hbm_capacity_drops",
            "cpu_capacity_evictions", "ssd_capacity_evictions",
            "ssd_capacity_admission_drops",
            "capacity_induced_recompute_tokens",
            "policy_avoidable_recompute_tokens",
            "hbm_to_cpu_bytes", "cpu_to_hbm_bytes", "cpu_to_ssd_bytes",
            "hbm_to_ssd_bytes", "ssd_to_hbm_bytes",
            "ssd_to_cpu_stage_bytes", "cpu_stage_to_hbm_bytes",
            "ssd_host_write_bytes", "ssd_host_read_bytes",
            "direct_ssd_write_bytes", "direct_ssd_read_bytes",
            "active_recompute_preemptions", "active_cpu_swap_preemptions",
            "pd_active_prefill_recompute_preemptions",
            "pd_active_prefill_recompute_tokens",
            "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute",
            "pd_chunk_cancelled_admissions",
            "pd_chunk_cancelled_admission_wait_ns",
            "pd_chunk_cancelled_admission_critical_wait_ns",
        ))
    return {
        "performed": True,
        "passed": True,
        "effective_policy": effective_policy,
        "durable_capacity_contract": durable_capacity_contract,
        "ssd_two_stage_restore_count": ssd_evidence_count,
        "cpu_restore_count": cpu_evidence_count,
        "foreground_transfer_tail_jobs": 0,
        "external_fabric_jobs": external_fabric_jobs,
        "external_resume_evidence": external_resume_evidence,
        "external_fabric_model_coexecution": (
            external_fabric_model_coexecution),
        "queue_recompute": queue_recompute_validation,
        "pd_chunk_admission": pd_chunk_validation,
        "pd_active_prefill_recompute": pd_active_prefill_validation,
    }


def _validate_measurement_cohort_contract(manifest, report):
    """Fail closed on the exact fixed cohort used by admission-order runs."""
    declared = "measurement_cohort_selection" in manifest
    selection = _normalize_measurement_cohort_selection(
        manifest.get("measurement_cohort_selection", "completion_order"),
        manifest.get("mode", "backlog"),
    )
    window = report.get("measurement_window") or {}
    admission_report = report.get("session_admission") or {}
    if declared:
        if window.get("measurement_cohort_selection") != selection:
            raise ExperimentError(
                f"Measurement cohort selection mismatch for "
                f"{manifest['run_id']}")
        if admission_report.get("measurement_cohort_selection") != selection:
            raise ExperimentError(
                f"Session-admission cohort selection mismatch for "
                f"{manifest['run_id']}")
    if not declared:
        # Legacy, completion-order reports predate the explicit fixed-target
        # contract. Their existing completion-order validation remains intact.
        return {
            "selection": "completion_order",
            "performed": False,
            "reason": "legacy_manifest_has_no_explicit_selection",
        }

    target_ids = window.get("measurement_target_session_ids")
    if (not isinstance(target_ids, list)
            or any(not isinstance(value, str) or not value
                   for value in target_ids)
            or len(target_ids) != len(set(target_ids))):
        raise ExperimentError(
            f"Malformed measurement target session-ID list for "
            f"{manifest['run_id']}")
    target_count = window.get("measurement_target_session_count")
    target_completed = window.get("measurement_target_completed_sessions")
    measure = int(manifest["measure_completions"])
    if (not isinstance(target_count, int) or isinstance(target_count, bool)
            or target_count != len(target_ids)
            or target_count != measure):
        raise ExperimentError(
            f"Measurement target count mismatch for {manifest['run_id']}")
    if (not isinstance(target_completed, int)
            or isinstance(target_completed, bool)
            or target_completed != target_count):
        raise ExperimentError(
            f"Measurement target completion count mismatch for "
            f"{manifest['run_id']}")
    target_hash = window.get("measurement_target_session_ids_hash")
    computed_hash = _stable_json_hash(target_ids)
    if target_hash != computed_hash:
        raise ExperimentError(
            f"Measurement target session-ID hash mismatch for "
            f"{manifest['run_id']}")
    for field in (
            "target_semantics", "target_order_and_hash_semantics",
            "start_semantics", "end_semantics"):
        value = window.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ExperimentError(
                f"Missing auditable measurement definition {field} for "
                f"{manifest['run_id']}")
    measured_ids = _measured_session_ids(report)
    if set(measured_ids) != set(target_ids):
        raise ExperimentError(
            f"Measurement target list and lifecycle membership differ for "
            f"{manifest['run_id']}")

    result = {
        "selection": selection,
        "performed": True,
        "target_session_count": target_count,
        "target_session_ids_hash": target_hash,
    }
    if selection == "completion_order":
        return result

    warmup = int(manifest.get("warmup_completions", -1))
    if warmup < 0:
        raise ExperimentError(
            "admission_order measurement cohorts require a non-negative "
            "warmup_completions admission-prefix size")
    expected_sessions = _expected_runtime_sessions(manifest)
    if expected_sessions is None:
        raise ExperimentError(
            f"Admission-order run {manifest['run_id']} has no materialized "
            "workload for target reconstruction")
    expected_runtime_ids = list(expected_sessions)
    expected_warmup_ids = expected_runtime_ids[:warmup]
    expected_ids = expected_runtime_ids[warmup:warmup + measure]
    expected_required_ids = expected_warmup_ids + expected_ids
    if (len(expected_warmup_ids) != warmup
            or len(expected_ids) != measure
            or len(expected_required_ids) != warmup + measure):
        raise ExperimentError(
            f"Admission-order warmup plus target exceeds the materialized "
            f"epoch-major session sequence for {manifest['run_id']}")
    expected_warmup_hash = _stable_json_hash(expected_warmup_ids)
    expected_hash = _stable_json_hash(expected_ids)
    expected_required_hash = _stable_json_hash(expected_required_ids)
    expected_runtime_hash = _stable_json_hash(expected_runtime_ids)
    manifest_warmup_ids = manifest.get(
        "expected_measurement_warmup_session_ids")
    manifest_warmup_hash = manifest.get(
        "expected_measurement_warmup_session_ids_hash")
    manifest_ids = manifest.get(
        "expected_measurement_target_session_ids")
    manifest_hash = manifest.get(
        "expected_measurement_target_session_ids_hash")
    manifest_required_ids = manifest.get(
        "expected_measurement_required_session_ids")
    manifest_required_hash = manifest.get(
        "expected_measurement_required_session_ids_hash")
    if (manifest.get("expected_runtime_session_count")
            != len(expected_runtime_ids)
            or manifest.get("expected_runtime_session_ids_hash")
            != expected_runtime_hash):
        raise ExperimentError(
            f"Admission-order manifest runtime order identity does not "
            f"match the epoch-major workload for {manifest['run_id']}")
    if (manifest_warmup_ids != expected_warmup_ids
            or manifest_warmup_hash != expected_warmup_hash
            or manifest_ids != expected_ids
            or manifest_hash != expected_hash
            or manifest_required_ids != expected_required_ids
            or manifest_required_hash != expected_required_hash):
        raise ExperimentError(
            f"Admission-order manifest fixed prefix does not match the "
            f"epoch-major warmup and target sessions for "
            f"{manifest['run_id']}")
    if target_ids != expected_ids or target_hash != expected_hash:
        raise ExperimentError(
            f"Admission-order report target does not match the fixed "
            f"epoch-major sessions after warmup for {manifest['run_id']}")

    warmup_ids = window.get("measurement_warmup_session_ids")
    warmup_hash = window.get("measurement_warmup_session_ids_hash")
    required_ids = window.get("measurement_required_session_ids")
    required_hash = window.get("measurement_required_session_ids_hash")
    if (warmup_ids != expected_warmup_ids
            or warmup_hash != expected_warmup_hash
            or required_ids != expected_required_ids
            or required_hash != expected_required_hash):
        raise ExperimentError(
            f"Admission-order report fixed prefix does not match the "
            f"manifest for {manifest['run_id']}")
    if (window.get("measurement_warmup_session_count") != warmup
            or window.get("measurement_warmup_completed_sessions") != warmup
            or window.get("measurement_required_session_count")
            != warmup + measure
            or window.get("measurement_required_completed_sessions")
            != warmup + measure
            or window.get("measurement_prefix_id_overlap_count") != 0
            or window.get("measurement_boundary_complete") is not True):
        raise ExperimentError(
            f"Admission-order fixed-prefix completion counters do not "
            f"reconcile for {manifest['run_id']}")

    lifecycle_by_id = {
        str(row.get("session_id")): row
        for row in report.get("sessions", {}).get("records", [])
    }
    try:
        warmup_rows = [
            lifecycle_by_id[session_id]
            for session_id in expected_warmup_ids
        ]
        target_rows = [
            lifecycle_by_id[session_id] for session_id in expected_ids
        ]
    except KeyError as exc:
        raise ExperimentError(
            f"Admission-order target lifecycle is missing for "
            f"{manifest['run_id']}: {exc.args[0]}") from exc
    for admission_index, row in enumerate(warmup_rows):
        if (row.get("status") != "completed"
                or row.get("measurement_warmup") is not True
                or row.get("measurement_target") is not False
                or row.get("measurement_required") is not True
                or row.get("measurement_included") is not False
                or row.get("measurement_role")
                != "fixed_admission_prefix_warmup"
                or row.get("planned_admission_index") != admission_index
                or row.get("admission_index") != admission_index):
            raise ExperimentError(
                f"Admission-order lifecycle does not prove warmup-prefix "
                f"index {admission_index} for {manifest['run_id']}")
    for offset, row in enumerate(target_rows):
        admission_index = warmup + offset
        if (row.get("status") != "completed"
                or row.get("measurement_target") is not True
                or row.get("measurement_warmup") is not False
                or row.get("measurement_required") is not True
                or row.get("measurement_included") is not True
                or row.get("measurement_role") != "measurement_target"
                or row.get("planned_admission_index") != admission_index
                or row.get("admission_index") != admission_index):
            raise ExperimentError(
                f"Admission-order lifecycle does not prove target index "
                f"{admission_index} for {manifest['run_id']}")
    try:
        expected_start_ns = min(
            int(row["admission_time_ns"]) for row in target_rows)
        expected_end_ns = max(
            int(row["completion_time_ns"]) for row in target_rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentError(
            f"Admission-order target timestamps are incomplete for "
            f"{manifest['run_id']}") from exc
    if (window.get("measurement_start_ns") != expected_start_ns
            or window.get("measurement_end_ns") != expected_end_ns):
        raise ExperimentError(
            f"Admission-order measurement bounds do not match target "
            f"admission/completion timestamps for {manifest['run_id']}")
    if (admission_report.get("measurement_target_session_count")
            != target_count
            or admission_report.get("measurement_target_completed_sessions")
            != target_count
            or admission_report.get("measurement_warmup_session_count")
            != warmup
            or admission_report.get("measurement_warmup_completed_sessions")
            != warmup
            or admission_report.get("measurement_required_session_count")
            != warmup + target_count
            or admission_report.get(
                "measurement_required_completed_sessions")
            != warmup + target_count
            or admission_report.get("measurement_prefix_id_overlap_count")
            != 0):
        raise ExperimentError(
            f"Router fixed-prefix counters do not reconcile for "
            f"{manifest['run_id']}")
    warmup_completion_boundary_ns = max(
        (int(row["completion_time_ns"]) for row in warmup_rows),
        default=None,
    )
    expected_admitted_overlap = (
        sum(
            int(row["admission_time_ns"]) < warmup_completion_boundary_ns
            for row in target_rows
        )
        if warmup_completion_boundary_ns is not None else 0
    )
    expected_completed_overlap = (
        sum(
            int(row["completion_time_ns"]) < warmup_completion_boundary_ns
            for row in target_rows
        )
        if warmup_completion_boundary_ns is not None else 0
    )
    if (window.get("warmup_completion_boundary_ns")
            != warmup_completion_boundary_ns
            or window.get(
                "target_admitted_before_warmup_complete_session_count")
            != expected_admitted_overlap
            or window.get(
                "target_completed_before_warmup_complete_session_count")
            != expected_completed_overlap
            or window.get(
                "target_execution_overlapped_unfinished_warmup")
            is not (expected_admitted_overlap > 0)):
        raise ExperimentError(
            f"Admission-order warmup/target temporal-overlap audit does not "
            f"reconcile for {manifest['run_id']}")
    result.update({
        "expected_runtime_session_count": len(expected_runtime_ids),
        "expected_runtime_session_ids_hash": expected_runtime_hash,
        "expected_warmup_session_ids": expected_warmup_ids,
        "expected_warmup_session_ids_hash": expected_warmup_hash,
        "expected_session_ids": expected_ids,
        "expected_session_ids_hash": expected_hash,
        "expected_required_session_ids": expected_required_ids,
        "expected_required_session_ids_hash": expected_required_hash,
        "prefix_id_overlap_count": 0,
        "target_admitted_before_warmup_complete_session_count": (
            expected_admitted_overlap),
        "target_completed_before_warmup_complete_session_count": (
            expected_completed_overlap),
        "measurement_start_ns": expected_start_ns,
        "measurement_end_ns": expected_end_ns,
    })
    return result


def _validate_session_queue_contract(manifest, report):
    """Fail closed on session waiting semantics emitted by schema 8+."""
    schema_version = report.get("schema_version")
    manifest_schema_version = int(
        manifest.get("schema_version", 0) or 0)
    requires_queue_contract = manifest_schema_version >= 8
    requires_current_contract = manifest_schema_version >= SCHEMA_VERSION
    if (not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version < 8):
        if requires_queue_contract:
            raise ExperimentError(
                f"Online manifest schema {manifest_schema_version} requires "
                "session report schema >= 8 for "
                f"{manifest['run_id']}; observed={schema_version!r}")
        return {
            "performed": False,
            "reason": "legacy_session_report_has_no_queue_contract",
        }
    if (requires_current_contract
            and schema_version < MIN_CURRENT_SESSION_REPORT_SCHEMA_VERSION):
        raise ExperimentError(
            f"Online manifest schema {manifest_schema_version} requires "
            "session report schema >= "
            f"{MIN_CURRENT_SESSION_REPORT_SCHEMA_VERSION} for "
            f"{manifest['run_id']}; observed={schema_version!r}")

    admission = report.get("session_admission") or {}
    mode = str(manifest.get("mode", admission.get("mode", "backlog")))
    configured_max_active = manifest.get(
        "max_active_sessions", admission.get("max_active_sessions", 0))
    if (not isinstance(configured_max_active, int)
            or isinstance(configured_max_active, bool)
            or configured_max_active < 0):
        raise ExperimentError(
            f"Invalid configured max_active_sessions for "
            f"{manifest['run_id']}: {configured_max_active!r}")
    capped_poisson = mode == "poisson" and configured_max_active > 0
    uses_fifo_slot_queue = mode == "backlog" or capped_poisson
    if mode == "backlog":
        expected_queue_policy = "fifo_wait_for_slot"
    elif capped_poisson:
        expected_queue_policy = "poisson_fifo_wait_for_slot"
    else:
        expected_queue_policy = "arrival_time_order"
    if admission.get("queue_policy") != expected_queue_policy:
        raise ExperimentError(
            f"Session queue policy mismatch for {manifest['run_id']}: "
            f"observed={admission.get('queue_policy')!r}, "
            f"expected={expected_queue_policy!r}")

    logical_drop_count = admission.get("logical_session_drop_count")
    if (not isinstance(logical_drop_count, int)
            or isinstance(logical_drop_count, bool)
            or logical_drop_count != 0):
        raise ExperimentError(
            f"Logical session drop contract failed for "
            f"{manifest['run_id']}: {logical_drop_count!r}")
    raw_lifecycle = (report.get("sessions") or {}).get("records", [])
    lifecycle = raw_lifecycle if isinstance(raw_lifecycle, list) else []
    lifecycle_drop_count = sum(
        str(row.get("status")) == "dropped"
        for row in lifecycle if isinstance(row, dict)
    )
    if lifecycle_drop_count != logical_drop_count:
        raise ExperimentError(
            f"Logical session drop counter does not reconcile with lifecycle "
            f"records for {manifest['run_id']}")

    stop_after_measurement = bool(
        manifest.get(
            "stop_after_measurement",
            admission.get("stop_after_measurement", True),
        )
    )
    expected_cutoff = "right_censor" if stop_after_measurement else "drain"
    if admission.get("cutoff_disposition") != expected_cutoff:
        raise ExperimentError(
            f"Session cutoff disposition mismatch for {manifest['run_id']}: "
            f"observed={admission.get('cutoff_disposition')!r}, "
            f"expected={expected_cutoff!r}")
    slot_release_event = admission.get("slot_release_event")
    precise_slot_events = {
        "final_request_completion_on_decode_owner",
        "final_request_completion_on_colocated_owner",
    }
    legacy_slot_events = {
        "final_decode_completion", "final_llm_request_completion",
    }
    supported_slot_events = (
        precise_slot_events
        if schema_version >= 9 else precise_slot_events | legacy_slot_events
    )
    if slot_release_event not in supported_slot_events:
        raise ExperimentError(
            f"Unsupported session slot release event for "
            f"{manifest['run_id']}: {slot_release_event!r}")
    legacy_slot_release_event = admission.get("slot_release_event_legacy")
    if schema_version >= 9:
        expected_legacy_event = {
            "final_request_completion_on_decode_owner": (
                "final_decode_completion"),
            "final_request_completion_on_colocated_owner": (
                "final_llm_request_completion"),
        }[slot_release_event]
        if legacy_slot_release_event != expected_legacy_event:
            raise ExperimentError(
                f"Session slot release compatibility alias does not "
                f"reconcile for {manifest['run_id']}: observed="
                f"{legacy_slot_release_event!r}, expected="
                f"{expected_legacy_event!r}")

    if schema_version >= 9:
        if not isinstance(raw_lifecycle, list) or not lifecycle:
            raise ExperimentError(
                f"Schema {schema_version} session report has no lifecycle "
                f"records for {manifest['run_id']}")
        if any(not isinstance(row, dict) for row in lifecycle):
            raise ExperimentError(
                f"Session lifecycle contains a non-object row for "
                f"{manifest['run_id']}")
        session_ids = [str(row.get("session_id")) for row in lifecycle]
        if (any(not row.get("session_id") for row in lifecycle)
                or len(session_ids) != len(set(session_ids))):
            raise ExperimentError(
                f"Session lifecycle IDs are missing or duplicated for "
                f"{manifest['run_id']}")
        statuses = [str(row.get("status")) for row in lifecycle]
        unsupported_statuses = sorted(
            set(statuses) - {"completed", "censored"})
        if unsupported_statuses:
            raise ExperimentError(
                f"Terminal session lifecycle contains unsupported status "
                f"for {manifest['run_id']}: {unsupported_statuses}")

        counter_names = (
            "planned_sessions", "offered_sessions", "admitted_sessions",
            "completed_sessions", "active_sessions",
            "remaining_unadmitted_sessions",
            "remaining_backlog_sessions",
        )
        counters = {}
        for name in counter_names:
            value = admission.get(name)
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 0):
                raise ExperimentError(
                    f"Invalid session admission counter {name} for "
                    f"{manifest['run_id']}: {value!r}")
            counters[name] = value

        completed_count = statuses.count("completed")
        censored_count = statuses.count("censored")
        admitted_rows = [
            row for row in lifecycle
            if row.get("admission_index") is not None
        ]
        if any(
                not isinstance(row["admission_index"], int)
                or isinstance(row["admission_index"], bool)
                or row["admission_index"] < 0
                for row in admitted_rows):
            raise ExperimentError(
                f"Session admission indices are malformed for "
                f"{manifest['run_id']}")
        expected_counters = {
            "planned_sessions": len(lifecycle),
            "offered_sessions": len(lifecycle),
            "admitted_sessions": len(admitted_rows),
            "completed_sessions": completed_count,
            "active_sessions": 0,
            "remaining_unadmitted_sessions": (
                len(lifecycle) - len(admitted_rows)),
        }
        if uses_fifo_slot_queue:
            expected_counters["remaining_backlog_sessions"] = (
                len(lifecycle) - len(admitted_rows))
        mismatches = {
            name: {"observed": counters[name], "expected": expected}
            for name, expected in expected_counters.items()
            if counters[name] != expected
        }
        if mismatches:
            raise ExperimentError(
                f"Session lifecycle/admission counters do not reconcile "
                f"for {manifest['run_id']}: {mismatches}")
        if any(
                row.get("status") == "completed"
                and row.get("admission_index") is None
                for row in lifecycle):
            raise ExperimentError(
                f"Completed session was never admitted for "
                f"{manifest['run_id']}")

        if uses_fifo_slot_queue:
            planned_indices = [
                row.get("planned_admission_index") for row in lifecycle]
            malformed_planned_indices = any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in planned_indices
            )
            if (malformed_planned_indices
                    or planned_indices != list(range(len(lifecycle)))):
                raise ExperimentError(
                    f"Slot-capped planned admission order is not an exact FIFO "
                    f"index sequence for {manifest['run_id']}")
            admitted_by_index = sorted(
                admitted_rows, key=lambda row: row["admission_index"])
            admission_indices = [
                row["admission_index"] for row in admitted_by_index]
            if admission_indices != list(range(len(admitted_rows))):
                raise ExperimentError(
                    f"Slot-capped admission indices are not a contiguous FIFO "
                    f"prefix for {manifest['run_id']}")
            if any(
                    row["planned_admission_index"] != admission_index
                    for admission_index, row in enumerate(admitted_by_index)):
                raise ExperimentError(
                    f"Slot-capped admitted sessions are not the planned FIFO "
                    f"prefix for {manifest['run_id']}")

            configured_k = admission.get("max_active_sessions")
            peak_active = (report.get("active_session_population") or {}).get(
                "peak_active_sessions")
            if (not isinstance(configured_k, int)
                    or isinstance(configured_k, bool)
                    or configured_k <= 0
                    or not isinstance(peak_active, (int, float))
                    or isinstance(peak_active, bool)
                    or not math.isfinite(float(peak_active))
                    or peak_active < 0
                    or peak_active > configured_k
                    or configured_k != configured_max_active):
                raise ExperimentError(
                    f"Slot-capped active-session peak exceeds or cannot be "
                    f"audited against K for {manifest['run_id']}: "
                    f"peak={peak_active!r}, report_K={configured_k!r}, "
                    f"configured_K={configured_max_active!r}")

        censoring = report.get("censoring") or {}
        if expected_cutoff == "drain":
            if censored_count != 0 or completed_count != len(lifecycle):
                raise ExperimentError(
                    f"Drain session lifecycle is incomplete for "
                    f"{manifest['run_id']}")
        else:
            reported_censored = censoring.get("censored_sessions")
            if (not isinstance(reported_censored, int)
                    or isinstance(reported_censored, bool)
                    or reported_censored != censored_count
                    or completed_count + censored_count != len(lifecycle)):
                raise ExperimentError(
                    f"Right-censored session lifecycle does not reconcile "
                    f"for {manifest['run_id']}: lifecycle={censored_count}, "
                    f"reported={reported_censored!r}")
    return {
        "performed": True,
        "queue_policy": expected_queue_policy,
        "logical_session_drop_count": 0,
        "slot_release_event": slot_release_event,
        "slot_release_event_legacy": legacy_slot_release_event,
        "cutoff_disposition": expected_cutoff,
    }


def _validate_completed_report(manifest, report, agentic_report):
    if report.get("run_id") != manifest["run_id"]:
        raise ExperimentError(
            f"Session report run_id mismatch for {manifest['run_id']}")
    if agentic_report.get("run_id") != manifest["run_id"]:
        raise ExperimentError(
            f"Agentic report run_id mismatch for {manifest['run_id']}")
    window = report.get("measurement_window", {})
    if not window.get("warmup_complete", False):
        raise ExperimentError(
            f"Warmup did not complete for {manifest['run_id']}")
    if not window.get("measurement_complete", False):
        raise ExperimentError(
            f"Measurement cohort did not complete for {manifest['run_id']}")
    if int(window.get("measure_completions_observed", -1)) != int(
            manifest["measure_completions"]):
        raise ExperimentError(
            f"Measured completion count mismatch for {manifest['run_id']}")
    duration_ns = window.get("measurement_duration_ns")
    if (not isinstance(duration_ns, (int, float))
            or isinstance(duration_ns, bool)
            or not math.isfinite(float(duration_ns))
            or duration_ns <= 0):
        raise ExperimentError(
            f"Invalid measurement duration for {manifest['run_id']}: "
            f"{duration_ns}")
    start_ns = window.get("measurement_start_ns")
    end_ns = window.get("measurement_end_ns")
    if (not isinstance(start_ns, (int, float))
            or not isinstance(end_ns, (int, float))
            or isinstance(start_ns, bool) or isinstance(end_ns, bool)
            or int(end_ns) - int(start_ns) != int(duration_ns)):
        raise ExperimentError(
            f"Measurement start/end/duration do not reconcile for "
            f"{manifest['run_id']}")
    simulated_ns = window.get("simulated_duration_ns")
    if int(agentic_report.get("simulated_duration_ns", -1)) != int(
            simulated_ns):
        raise ExperimentError(
            f"Session/agentic simulated duration mismatch for "
            f"{manifest['run_id']}")
    admission_report = report.get("session_admission") or {}
    cohort_validation = _validate_measurement_cohort_contract(
        manifest, report)
    session_queue_validation = _validate_session_queue_contract(
        manifest, report)
    manifest_schema_version = int(
        manifest.get("schema_version", 0) or 0)
    if (manifest_schema_version >= 8
            and session_queue_validation.get("performed") is not True):
        raise ExperimentError(
            f"Current online manifest did not perform the session queue "
            f"contract for {manifest['run_id']}")
    offered_arrival_validation = None
    if manifest_schema_version >= 8:
        offered_arrival_validation = _validate_offered_arrival_trace(
            manifest, report)
    stop_after_measurement = bool(
        manifest.get("stop_after_measurement", True))
    reported_stop = admission_report.get("stop_after_measurement")
    if ("stop_after_measurement" in manifest
            and reported_stop is not stop_after_measurement):
        raise ExperimentError(
            f"Session stop mode mismatch for {manifest['run_id']}")
    censoring = report.get("censoring") or {}
    censored_sessions = int(censoring.get("censored_sessions", 0) or 0)
    if (manifest.get("require_complete_session_cohort")
            and censored_sessions != 0):
        raise ExperimentError(
            f"Complete-cohort run {manifest['run_id']} censored "
            f"{censored_sessions} session(s)")
    if manifest.get("require_complete_session_cohort"):
        expected = int(manifest["available_sessions"])
        completed_total = int(
            report["throughput"]["completed_sessions_total"])
        if completed_total != expected:
            raise ExperimentError(
                f"Complete-cohort run {manifest['run_id']} completed "
                f"{completed_total}/{expected} generated sessions")
        expected_requests = int(manifest["expected_request_count"])
        completed_requests_total = int(
            report["throughput"]["completed_requests_total"])
        if completed_requests_total != expected_requests:
            raise ExperimentError(
                f"Complete-cohort run {manifest['run_id']} completed "
                f"{completed_requests_total}/{expected_requests} expected "
                "requests")
    if not stop_after_measurement:
        expected_sessions = int(manifest["available_sessions"])
        expected_requests = int(manifest["expected_request_count"])
        if window.get("measurement_early_stopped") is not False:
            raise ExperimentError(
                f"Drain-mode run early-stopped for {manifest['run_id']}")
        expected_admission_counters = {
            "planned_sessions": expected_sessions,
            "offered_sessions": expected_sessions,
            "admitted_sessions": expected_sessions,
            "completed_sessions": expected_sessions,
            "active_sessions": 0,
            "remaining_unadmitted_sessions": 0,
            "remaining_backlog_sessions": 0,
        }
        mismatches = {}
        for key, value in expected_admission_counters.items():
            observed = admission_report.get(key)
            if (not isinstance(observed, int) or isinstance(observed, bool)
                    or observed != value):
                mismatches[key] = {
                    "observed": observed,
                    "expected": value,
                }
        if mismatches:
            raise ExperimentError(
                f"Planned/admitted/completed drain counters do not "
                f"reconcile for {manifest['run_id']}: {mismatches}")
        if admission_report.get("admission_frozen") is not False:
            raise ExperimentError(
                f"Drain-mode admission unexpectedly froze for "
                f"{manifest['run_id']}")
        if (int(report["throughput"]["completed_sessions_total"])
                != expected_sessions
                or int(report["throughput"]["completed_requests_total"])
                != expected_requests
                or censored_sessions != 0):
            raise ExperimentError(
                f"Drain-mode run did not finish its complete generated "
                f"cohort for {manifest['run_id']}")
    measured_ids = _measured_session_ids(report)
    if len(measured_ids) != int(manifest["measure_completions"]):
        raise ExperimentError(
            f"Measured session-record count mismatch for "
            f"{manifest['run_id']}")
    if int(report["throughput"].get("completed_sessions", -1)) != int(
            manifest["measure_completions"]):
        raise ExperimentError(
            f"Throughput session count mismatch for {manifest['run_id']}")
    expected_session_rate = (
        int(manifest["measure_completions"]) * 1_000_000_000
        / float(duration_ns)
    )
    observed_session_rate = report["throughput"].get(
        "sessions_per_second_measurement_window")
    if (not isinstance(observed_session_rate, (int, float))
            or isinstance(observed_session_rate, bool)
            or not math.isfinite(float(observed_session_rate))
            or observed_session_rate <= 0
            or not math.isclose(
                float(observed_session_rate), expected_session_rate,
                rel_tol=1e-12, abs_tol=1e-12)):
        raise ExperimentError(
            f"Session throughput does not reconcile with completions and "
            f"duration for {manifest['run_id']}")
    if manifest.get("mode") == "poisson":
        configured_rate = admission_report.get("session_arrival_rate_sps")
        if (not isinstance(configured_rate, (int, float))
                or isinstance(configured_rate, bool)
                or not math.isclose(
                    float(configured_rate), float(manifest["load_value"]),
                    rel_tol=0, abs_tol=0)):
            raise ExperimentError(
                f"Configured Poisson rate mismatch for {manifest['run_id']}")
        configured_seed = admission_report.get("session_arrival_seed")
        if (not isinstance(configured_seed, int)
                or isinstance(configured_seed, bool)
                or configured_seed != int(manifest["arrival_seed"])):
            raise ExperimentError(
                f"Configured Poisson seed mismatch for {manifest['run_id']}")
        if (int(manifest.get("schema_version", 0) or 0) >= 8
                or "max_active_sessions" in manifest):
            configured_max_active = admission_report.get(
                "max_active_sessions")
            if (not isinstance(configured_max_active, int)
                    or isinstance(configured_max_active, bool)
                    or configured_max_active
                    != int(manifest.get("max_active_sessions", 0))):
                raise ExperimentError(
                    "Configured Poisson max-active session limit mismatch "
                    f"for {manifest['run_id']}")
        realized_rate = report["throughput"].get(
            "realized_session_offer_rate_sps")
        if (int(manifest.get("available_sessions", 0)) > 1
                and (not isinstance(realized_rate, (int, float))
                     or isinstance(realized_rate, bool)
                     or not math.isfinite(float(realized_rate))
                     or realized_rate <= 0)):
            raise ExperimentError(
                f"Missing positive realized Poisson offer rate for "
                f"{manifest['run_id']}")

    validation = report.get("validation", {}).get("timing")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ExperimentError(
            f"Timing validation failed or is missing for "
            f"{manifest['run_id']}")
    warnings = validation.get("warnings") or []
    if (not isinstance(warnings, list)
            or any(not isinstance(warning, str) or not warning
                   for warning in warnings)):
        raise ExperimentError(
            f"Timing warning list is malformed for {manifest['run_id']}")
    warning_codes_raw = validation.get("warning_codes")
    if warning_codes_raw is None:
        warning_codes = []
    elif not isinstance(warning_codes_raw, list):
        raise ExperimentError(
            f"Timing warning-code list is malformed for "
            f"{manifest['run_id']}")
    else:
        warning_codes = warning_codes_raw
        if (len(warning_codes) != len(warnings)
                or any(not isinstance(code, str) or not code
                       for code in warning_codes)):
            raise ExperimentError(
                f"Timing warning codes do not align one-to-one with "
                f"warnings for {manifest['run_id']}")
        unknown_codes = sorted(set(warning_codes) - TIMING_WARNING_CODES)
        if unknown_codes:
            raise ExperimentError(
                f"Timing validation emitted unknown warning codes for "
                f"{manifest['run_id']}: {unknown_codes}")
    allowed_codes = _normalize_allowed_timing_warning_codes(
        manifest.get("allowed_timing_warning_codes"),
        f"{manifest['run_id']}.allowed_timing_warning_codes",
    )
    if allowed_codes is not None:
        if warnings and warning_codes_raw is None:
            raise ExperimentError(
                f"Timing warning allowlist cannot audit legacy un-coded "
                f"warnings for {manifest['run_id']}: {warnings[:3]}")
        disallowed = [
            {"code": code, "warning": warning}
            for code, warning in zip(warning_codes, warnings)
            if code not in allowed_codes
        ]
        if disallowed:
            raise ExperimentError(
                f"Timing validation emitted warning code(s) outside the "
                f"allowlist for {manifest['run_id']}: {disallowed[:3]}")
    elif warnings and not manifest.get("allow_timing_warnings", False):
        raise ExperimentError(
            f"Timing validation emitted {len(warnings)} warning(s) for "
            f"{manifest['run_id']}: {warnings[:3]}")

    for group_name in ("all", "initial", "resume"):
        group = report.get("requests", {}).get(group_name)
        if not isinstance(group, dict):
            raise ExperimentError(
                f"Missing request group {group_name} for "
                f"{manifest['run_id']}")
        group_count = group.get("count")
        if (not isinstance(group_count, int) or isinstance(group_count, bool)
                or group_count < 0):
            raise ExperimentError(
                f"Invalid request count for group {group_name} in "
                f"{manifest['run_id']}")
        for metric_name in (
                "latency_ns", "release_to_first_schedule_ns",
                "scheduler_queue_wait_ns", "ttft_ns", "tpot_ns",
                "itl_ns", "restore_gate_wait_ns",
                "owner_ready_gate_ns", "pd_pair_fifo_wait_ns",
                "prepare_boundary_wait_ns",
                "source_demotion_join_wait_ns",
                "hbm_admission_wait_ns",
                "transient_dram_capacity_wait_ns",
                "restore_queue_wait_ns",
                "restore_service_ns", "pd_launch_admission_wait_ns",
                "pd_launch_admission_critical_wait_ns"):
            _validate_distribution(
                f"{manifest['run_id']}.requests.{group_name}.{metric_name}",
                group.get(metric_name),
            )
            if (metric_name != "itl_ns"
                    and group[metric_name]["count"] != group_count):
                raise ExperimentError(
                    f"Distribution count mismatch for {manifest['run_id']}."
                    f"requests.{group_name}.{metric_name}")
    _validate_distribution(
        f"{manifest['run_id']}.sessions.admission_queue_wait_ns",
        report.get("sessions", {}).get("admission_queue_wait_ns"),
    )
    for metric_name in (
            "e2e_from_admission_ns", "e2e_from_offer_ns"):
        distribution = report.get("sessions", {}).get(metric_name)
        if distribution is not None:
            _validate_distribution(
                f"{manifest['run_id']}.sessions.{metric_name}",
                distribution,
            )
    session_admission_count = report["sessions"][
        "admission_queue_wait_ns"]["count"]
    for metric_name in (
            "e2e_from_admission_ns", "e2e_from_offer_ns"):
        distribution = report.get("sessions", {}).get(metric_name)
        if (distribution is not None
                and int(distribution["count"]) != session_admission_count):
            raise ExperimentError(
                f"Session {metric_name} distribution count mismatch for "
                f"{manifest['run_id']}")
    execution_distribution = report.get("sessions", {}).get(
        "e2e_from_admission_ns")
    jct_distribution = report.get("sessions", {}).get(
        "e2e_from_offer_ns")
    if execution_distribution is not None and jct_distribution is not None:
        expected_jct_sum = (
            int(execution_distribution["sum"])
            + int(report["sessions"]["admission_queue_wait_ns"]["sum"])
        )
        if int(jct_distribution["sum"]) != expected_jct_sum:
            raise ExperimentError(
                "Session JCT does not reconcile with admission wait plus "
                f"admission-to-completion time for {manifest['run_id']}")
    if session_admission_count != int(manifest["measure_completions"]):
        raise ExperimentError(
            f"Session admission distribution count mismatch for "
            f"{manifest['run_id']}")
    all_count = int(report["requests"]["all"]["count"])
    initial_count = int(report["requests"]["initial"]["count"])
    resume_count = int(report["requests"]["resume"]["count"])
    if initial_count != int(manifest["measure_completions"]):
        raise ExperimentError(
            f"Initial request count mismatch for {manifest['run_id']}")
    if initial_count + resume_count != all_count:
        raise ExperimentError(
            f"Initial/resume request counts do not reconcile for "
            f"{manifest['run_id']}")
    if len(report["requests"].get("records") or []) not in {0, all_count}:
        raise ExperimentError(
            f"Exact request-record count does not reconcile for "
            f"{manifest['run_id']}")
    for classification_name in (
            "resume_by_return_gap_type",
            "resume_by_residency_at_return",
            "resume_by_source"):
        classification = report["requests"].get(classification_name)
        if not isinstance(classification, dict):
            raise ExperimentError(
                f"Missing {classification_name} for {manifest['run_id']}")
        classified_count = sum(
            int((group or {}).get("count", 0))
            for group in classification.values()
        )
        if classified_count != resume_count:
            raise ExperimentError(
                f"{classification_name} count does not reconcile for "
                f"{manifest['run_id']}")
    cross_classification = report["requests"].get(
        "resume_by_return_gap_type_and_source")
    if not isinstance(cross_classification, dict):
        raise ExperimentError(
            f"Missing return-gap/source cross-classification for "
            f"{manifest['run_id']}")
    cross_count = sum(
        int((group or {}).get("count", 0))
        for source_groups in cross_classification.values()
        for group in (source_groups or {}).values()
    )
    if cross_count != resume_count:
        raise ExperimentError(
            f"Return-gap/source cross-classification does not reconcile for "
            f"{manifest['run_id']}")
    for gap_type, gap_group in report["requests"][
            "resume_by_return_gap_type"].items():
        marginal = sum(
            int((group or {}).get("count", 0))
            for group in (cross_classification.get(gap_type) or {}).values()
        )
        if marginal != int(gap_group["count"]):
            raise ExperimentError(
                f"Return-gap/source row marginal does not reconcile for "
                f"{manifest['run_id']}, gap={gap_type}")
    for source, source_group in report["requests"][
            "resume_by_source"].items():
        marginal = sum(
            int(((groups or {}).get(source) or {}).get("count", 0))
            for groups in cross_classification.values()
        )
        if marginal != int(source_group["count"]):
            raise ExperimentError(
                f"Return-gap/source column marginal does not reconcile for "
                f"{manifest['run_id']}, source={source}")

    gap_residency = report["requests"].get(
        "resume_by_return_gap_type_and_residency_at_return")
    if not isinstance(gap_residency, dict):
        raise ExperimentError(
            f"Missing return-gap/residency cross-classification for "
            f"{manifest['run_id']}")
    residency_cross_count = sum(
        int((group or {}).get("count", 0))
        for residency_groups in gap_residency.values()
        for group in (residency_groups or {}).values()
    )
    if residency_cross_count != resume_count:
        raise ExperimentError(
            f"Return-gap/residency cross-classification does not reconcile "
            f"for {manifest['run_id']}")
    for gap_type, gap_group in report["requests"][
            "resume_by_return_gap_type"].items():
        marginal = sum(
            int((group or {}).get("count", 0))
            for group in (gap_residency.get(gap_type) or {}).values()
        )
        if marginal != int(gap_group["count"]):
            raise ExperimentError(
                f"Return-gap/residency row marginal does not reconcile for "
                f"{manifest['run_id']}, gap={gap_type}")
    for residency, residency_group in report["requests"][
            "resume_by_residency_at_return"].items():
        marginal = sum(
            int(((groups or {}).get(residency) or {}).get("count", 0))
            for groups in gap_residency.values()
        )
        if marginal != int(residency_group["count"]):
            raise ExperimentError(
                f"Return-gap/residency column marginal does not reconcile "
                f"for {manifest['run_id']}, residency={residency}")
    cohort_request_count = report["throughput"].get(
        "completed_requests_in_session_cohort")
    if cohort_request_count is None or int(cohort_request_count) != all_count:
        raise ExperimentError(
            f"Session-cohort request count mismatch for {manifest['run_id']}")
    min_fraction_at_k = manifest.get("min_fraction_at_configured_k")
    if min_fraction_at_k is not None:
        observed_fraction = (report.get("active_session_population") or {}).get(
            "fraction_at_configured_k")
        if (not isinstance(observed_fraction, (int, float))
                or isinstance(observed_fraction, bool)
                or not math.isfinite(float(observed_fraction))
                or observed_fraction < float(min_fraction_at_k)):
            raise ExperimentError(
                f"Backlog population was not sustained for "
                f"{manifest['run_id']}: fraction_at_k={observed_fraction}, "
                f"required={min_fraction_at_k}")

    trace_validation = _validate_trace_identity(manifest, report)
    policy_validation = _validate_policy_invariants(
        manifest, report, agentic_report)
    report.setdefault("validation", {})["trace_identity"] = trace_validation
    report["validation"]["policy_invariants"] = policy_validation
    report["validation"]["measurement_cohort"] = cohort_validation
    report["validation"]["session_queue"] = session_queue_validation
    if offered_arrival_validation is not None:
        report["validation"]["offered_arrival_trace"] = (
            offered_arrival_validation)


def _validate_poisson_common_random_numbers(rows, manifests):
    """Prove that every Poisson rate reuses one fixed seed/draw grid."""
    manifest_by_run = {
        manifest["run_id"]: manifest for manifest in manifests
        if manifest.get("status") == "succeeded"
    }
    current_rows = [
        row for row in rows
        if row.get("mode") == "poisson"
        and int(manifest_by_run[row["run_id"]].get(
            "schema_version", 0) or 0) >= 8
    ]
    if not current_rows:
        return {
            "performed": False,
            "reason": "no_current_schema_poisson_rows",
        }

    rate_seed_sets = {}
    policy_grid = {}
    for row in current_rows:
        rate = float(row["load_value"])
        seed = row.get("arrival_seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ExperimentError(
                f"Poisson CRN audit found an invalid seed in "
                f"{row['run_id']}: {seed!r}")
        rate_seed_sets.setdefault(rate, set()).add(seed)
        policy_grid.setdefault((rate, seed), set()).add(str(row["policy"]))

    seed_sets = {tuple(sorted(seeds)) for seeds in rate_seed_sets.values()}
    if len(seed_sets) != 1:
        raise ExperimentError(
            "Poisson rates do not use one fixed arrival-seed set: "
            f"{dict(sorted(rate_seed_sets.items()))}")
    policy_sets = {tuple(sorted(policies)) for policies in policy_grid.values()}
    if len(policy_sets) != 1:
        raise ExperimentError(
            "Poisson rate/seed cells do not contain one fixed policy grid: "
            f"{dict(sorted(policy_grid.items()))}")

    manifests_by_seed = {}
    rows_by_seed = {}
    for row in current_rows:
        seed = int(row["arrival_seed"])
        rows_by_seed.setdefault(seed, []).append(row)
        manifests_by_seed.setdefault(seed, []).append(
            manifest_by_run[row["run_id"]])

    signatures = {}
    for seed, seed_rows in sorted(rows_by_seed.items()):
        malformed = [
            row["run_id"] for row in seed_rows
            if not _is_sha256_digest(
                row.get("poisson_unit_draw_trace_sha256"))
        ]
        if malformed:
            raise ExperimentError(
                "Poisson common-random-number signature is missing or "
                f"malformed for seed={seed}: {malformed}")
        unit_hashes = {
            row["poisson_unit_draw_trace_sha256"] for row in seed_rows}
        counts = {row["offered_arrival_trace_count"] for row in seed_rows}
        seed_manifests = manifests_by_seed[seed]
        workload_hashes = {
            manifest.get("workload_sha256") for manifest in seed_manifests}
        identity_hashes = {
            manifest.get("selected_session_identity_hash")
            for manifest in seed_manifests
        }
        if (any(not _is_sha256_digest(value) for value in workload_hashes)
                or any(not _is_sha256_digest(value)
                       for value in identity_hashes)):
            raise ExperimentError(
                "Poisson cross-rate template identity proof is missing or "
                f"malformed for seed={seed}")
        if (len(unit_hashes) != 1 or len(counts) != 1
                or len(workload_hashes) != 1 or len(identity_hashes) != 1):
            raise ExperimentError(
                "Poisson rates do not share one template order and unit-rate "
                f"exponential draw stream for seed={seed}: "
                f"unit_hashes={sorted(unit_hashes)}, counts={sorted(counts)}, "
                f"workloads={sorted(workload_hashes, key=str)}, "
                f"identities={sorted(identity_hashes, key=str)}")
        signatures[seed] = next(iter(unit_hashes))

    return {
        "performed": True,
        "passed": True,
        "rate_count": len(rate_seed_sets),
        "rates_sps": sorted(rate_seed_sets),
        "arrival_seeds": list(next(iter(seed_sets))),
        "policy_count": len(next(iter(policy_sets))),
        "unit_draw_trace_sha256_by_seed": signatures,
    }


def collect_results(manifests, oracle_label="infinite_hbm_oracle"):
    rows = []
    reports = {}
    for manifest in manifests:
        if manifest.get("status") != "succeeded":
            continue
        report = _load_json(manifest["session_metrics"])
        agentic_report = _load_json(manifest["agentic_kv_metrics"])
        _validate_completed_report(manifest, report, agentic_report)
        reports[manifest["run_id"]] = report
        validation = report.get("validation", {}).get("timing", {})
        if validation and not validation.get("passed", False):
            raise ExperimentError(
                f"Timing validation failed for {manifest['run_id']}")
        if manifest["strict_oracle"]:
            oracle_validation = report.get("strict_infinite_hbm_oracle") or {}
            if not oracle_validation.get("passed", False):
                raise ExperimentError(
                    f"Oracle invariant failed for {manifest['run_id']}")
        throughput = report["throughput"]
        request_all = report["requests"]["all"]
        resume = report["requests"]["resume"]
        resume_by_source = report["requests"].get("resume_by_source", {})
        all_request_count = int(request_all["count"])
        resume_count = int(resume["count"])
        hbm_resume_count = _count_group(resume_by_source, "hbm")
        cpu_resume_count = _count_group(resume_by_source, "cpu")
        ssd_resume_count = _count_group(resume_by_source, "ssd")
        raw_dropped_resume_source_count = _count_group(
            resume_by_source, "dropped")
        classified_resume_count = sum(
            int((group or {}).get("count", 0))
            for group in resume_by_source.values()
        )
        if classified_resume_count != resume_count:
            raise ExperimentError(
                f"Resume source counts do not reconcile for "
                f"{manifest['run_id']}: classified={classified_resume_count}, "
                f"resume={resume_count}")
        other_resume_count = max(
            0,
            resume_count - hbm_resume_count - cpu_resume_count
            - ssd_resume_count - raw_dropped_resume_source_count,
        )
        time_breakdown = agentic_report.get("time_breakdown", {})
        totals = agentic_report.get("totals", {})
        queue_recompute = agentic_report.get(
            "queue_recompute_policy", {})
        asynchronous = agentic_report.get("asynchronous_restore", {})
        synchronous = agentic_report.get("synchronous_swap", {})
        load_activity = agentic_report.get("observed_load_activity", {})
        overhead_denominators = report.get("overhead_denominators") or {}
        cohort_overhead = overhead_denominators.get(
            "measured_session_cohort") or {}
        window_overhead = overhead_denominators.get(
            "strict_completion_window") or {}
        compute = report.get("online_model_compute") or {}
        active = report.get("active_session_population") or {}
        admission_report = report.get("session_admission") or {}
        policy_validation = report.get("validation", {}).get(
            "policy_invariants", {})
        external_coexecution = policy_validation.get(
            "external_fabric_model_coexecution", {})
        queue_validation = policy_validation.get("queue_recompute", {})
        snapshot_first_chunk = queue_validation.get(
            "snapshot_to_first_chunk", {})
        pd_chunk_validation = policy_validation.get(
            "pd_chunk_admission", {})
        pd_active_prefill_validation = policy_validation.get(
            "pd_active_prefill_recompute", {})
        timing_warnings = report["validation"]["timing"].get(
            "warnings") or []
        timing_warning_codes = report["validation"]["timing"].get(
            "warning_codes") or []
        measurement_window = report.get("measurement_window") or {}
        measurement_cohort_selection = manifest.get(
            "measurement_cohort_selection", "completion_order")
        measurement_warmup_session_ids = measurement_window.get(
            "measurement_warmup_session_ids")
        measurement_target_session_ids = measurement_window.get(
            "measurement_target_session_ids")
        measurement_required_session_ids = measurement_window.get(
            "measurement_required_session_ids")
        offered_arrival_validation = report.get("validation", {}).get(
            "offered_arrival_trace", {})
        exact_rate_metrics = _derive_exact_rate_metrics(
            report, agentic_report, manifest)
        resume_source_accounting = exact_rate_metrics[
            "resume_source_accounting"]
        cross_layer_accounting = exact_rate_metrics[
            "cross_layer_request_accounting"]
        attempted_counts = resume_source_accounting[
            "attempted_counts_by_source"]
        effective_counts = resume_source_accounting[
            "effective_surviving_counts_by_source"]
        dropped_resume_count = int(
            resume_source_accounting["kv_state_unavailable_resume_count"])
        operational_metrics = _operational_metric_sources(manifest, report)
        hbm_occupancy = operational_metrics["hbm_occupancy"]
        hbm_categories = (
            hbm_occupancy["categories"] if hbm_occupancy else {})
        hbm_category_reports = (
            hbm_occupancy["category_reports"] if hbm_occupancy else {})
        row = {
            "run_id": manifest["run_id"],
            "online_artifact_schema_version": manifest.get(
                "schema_version"),
            "session_report_schema_version": report.get("schema_version"),
            "agentic_report_schema_version": agentic_report.get(
                "schema_version"),
            "request_csv_path": manifest.get("request_csv"),
            "request_csv_sha256": manifest.get("request_csv_sha256"),
            "session_metrics_sha256": manifest.get(
                "session_metrics_sha256"),
            "agentic_kv_metrics_sha256": manifest.get(
                "agentic_kv_metrics_sha256"),
            "workload_sha256": manifest.get("workload_sha256"),
            "selected_session_ids_hash": manifest.get(
                "selected_session_ids_hash"),
            "selected_session_identity_hash": manifest.get(
                "selected_session_identity_hash"),
            "mode": manifest["mode"],
            "load_value": manifest["load_value"],
            "pair_key": manifest.get("pair_key"),
            "queue_policy": admission_report.get("queue_policy"),
            "configured_max_active_sessions": admission_report.get(
                "max_active_sessions"),
            "logical_session_drop_count": admission_report.get(
                "logical_session_drop_count"),
            "cutoff_disposition": admission_report.get(
                "cutoff_disposition"),
            "slot_release_event": admission_report.get(
                "slot_release_event"),
            "slot_release_event_legacy": admission_report.get(
                "slot_release_event_legacy"),
            "arrival_seed": manifest.get("arrival_seed"),
            "offered_arrival_trace_count": report.get(
                "sessions", {}).get("offered_arrival_trace_count"),
            "offered_arrival_trace_sha256": report.get(
                "sessions", {}).get("offered_arrival_trace_sha256"),
            "poisson_unit_draw_trace_sha256": (
                offered_arrival_validation.get(
                    "poisson_unit_draw_trace_sha256")),
            "session_repetitions": manifest.get("session_repetitions", 1),
            "stop_after_measurement": manifest.get(
                "stop_after_measurement", True),
            "measurement_cohort_selection": measurement_cohort_selection,
            "measurement_warmup_session_count": measurement_window.get(
                "measurement_warmup_session_count"),
            "measurement_warmup_completed_sessions": measurement_window.get(
                "measurement_warmup_completed_sessions"),
            "measurement_warmup_session_ids_hash": measurement_window.get(
                "measurement_warmup_session_ids_hash"),
            "measurement_warmup_session_ids_json": (
                json.dumps(
                    measurement_warmup_session_ids,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                if measurement_warmup_session_ids is not None else None
            ),
            "measurement_target_session_count": measurement_window.get(
                "measurement_target_session_count"),
            "measurement_target_completed_sessions": measurement_window.get(
                "measurement_target_completed_sessions"),
            "measurement_target_session_ids_hash": measurement_window.get(
                "measurement_target_session_ids_hash"),
            "measurement_target_session_ids_json": (
                json.dumps(
                    measurement_target_session_ids,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                if measurement_target_session_ids is not None else None
            ),
            "measurement_required_session_count": measurement_window.get(
                "measurement_required_session_count"),
            "measurement_required_completed_sessions": (
                measurement_window.get(
                    "measurement_required_completed_sessions")),
            "measurement_required_session_ids_hash": measurement_window.get(
                "measurement_required_session_ids_hash"),
            "measurement_required_session_ids_json": (
                json.dumps(
                    measurement_required_session_ids,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                if measurement_required_session_ids is not None else None
            ),
            "measurement_prefix_id_overlap_count": measurement_window.get(
                "measurement_prefix_id_overlap_count"),
            "target_admitted_before_warmup_complete_session_count": (
                measurement_window.get(
                    "target_admitted_before_warmup_complete_session_count")),
            "target_completed_before_warmup_complete_session_count": (
                measurement_window.get(
                    "target_completed_before_warmup_complete_session_count")),
            "target_execution_overlapped_unfinished_warmup": (
                measurement_window.get(
                    "target_execution_overlapped_unfinished_warmup")),
            "policy": manifest["policy"],
            "effective_agentic_policy": agentic_report.get("policy"),
            "swap_execution_mode": agentic_report.get(
                "config", {}).get("swap_execution_mode"),
            "sessions_per_second": throughput[
                "sessions_per_second_measurement_window"],
            "requests_per_second": throughput[
                "requests_per_second_measurement_window"],
            "total_tokens_per_second": throughput[
                "total_tokens_per_second_measurement_window"],
            "configured_session_arrival_rate_sps": (
                admission_report.get("session_arrival_rate_sps")
                if manifest["mode"] == "poisson" else None),
            "realized_session_offer_rate_sps": throughput.get(
                "realized_session_offer_rate_sps"),
            "measurement_duration_ns": report["measurement_window"][
                "measurement_duration_ns"],
            "simulated_duration_ns": report["measurement_window"][
                "simulated_duration_ns"],
            "completed_sessions": throughput["completed_sessions"],
            "completed_requests": throughput["completed_requests"],
            "session_cohort_request_count": all_request_count,
            "resume_request_count": resume_count,
            "hbm_resume_count": hbm_resume_count,
            "cpu_resume_count": cpu_resume_count,
            "ssd_resume_count": ssd_resume_count,
            "attempted_physical_resume_count": (
                resume_source_accounting["attempted_count"]),
            "effective_surviving_resume_count": (
                resume_source_accounting["effective_surviving_count"]),
            "attempted_hbm_resume_count": attempted_counts["hbm"],
            "attempted_cpu_resume_count": attempted_counts["cpu"],
            "attempted_ssd_resume_count": attempted_counts["ssd"],
            "effective_surviving_hbm_resume_count": effective_counts["hbm"],
            "effective_surviving_cpu_resume_count": effective_counts["cpu"],
            "effective_surviving_ssd_resume_count": effective_counts["ssd"],
            "attempted_physical_resume_fraction_of_all_requests": (
                resume_source_accounting[
                    "attempted_fraction_of_all_requests"]),
            "effective_surviving_resume_fraction_of_all_requests": (
                resume_source_accounting[
                    "effective_surviving_fraction_of_all_requests"]),
            "attempted_hbm_resume_fraction_of_all_requests": (
                resume_source_accounting[
                    "attempted_fractions_of_all_requests"]["hbm"]),
            "attempted_cpu_resume_fraction_of_all_requests": (
                resume_source_accounting[
                    "attempted_fractions_of_all_requests"]["cpu"]),
            "attempted_ssd_resume_fraction_of_all_requests": (
                resume_source_accounting[
                    "attempted_fractions_of_all_requests"]["ssd"]),
            "effective_surviving_hbm_resume_fraction_of_all_requests": (
                resume_source_accounting[
                    "effective_surviving_fractions_of_all_requests"]["hbm"]),
            "effective_surviving_cpu_resume_fraction_of_all_requests": (
                resume_source_accounting[
                    "effective_surviving_fractions_of_all_requests"]["cpu"]),
            "effective_surviving_ssd_resume_fraction_of_all_requests": (
                resume_source_accounting[
                    "effective_surviving_fractions_of_all_requests"]["ssd"]),
            "attempted_restored_hit_tokens": resume_source_accounting[
                "attempted_restored_hit_tokens"],
            "restored_hit_tokens_discarded_by_active_prefill_recompute": (
                resume_source_accounting[
                    "restored_hit_tokens_discarded_by_active_prefill_recompute"]),
            "effective_surviving_hit_tokens": resume_source_accounting[
                "effective_surviving_hit_tokens"],
            "attempted_resume_by_return_gap_type_and_source_json": (
                json.dumps(
                    resume_source_accounting[
                        "attempted_by_return_gap_type_and_source"],
                    sort_keys=True, separators=(",", ":"))),
            "effective_surviving_resume_by_return_gap_type_and_source_json": (
                json.dumps(
                    resume_source_accounting[
                        "effective_surviving_by_return_gap_type_and_source"],
                    sort_keys=True, separators=(",", ":"))),
            "resume_source_accounting_denominator": (
                "all_completed_requests_in_measured_session_cohort"),
            "resume_source_provenance_semantics": (
                resume_source_accounting["source_semantics"]),
            "cross_layer_request_accounting_audit_passed": (
                cross_layer_accounting.get("passed")),
            "cross_layer_request_accounting_scope": (
                cross_layer_accounting.get("scope")),
            "dropped_resume_count": dropped_resume_count,
            "raw_dropped_resume_source_count": (
                raw_dropped_resume_source_count),
            "kv_state_unavailable_resume_count": dropped_resume_count,
            "zero_overlap_resume_count": resume_source_accounting[
                "zero_overlap_resume_count"],
            "dropped_resume_columns_semantics": (
                "reusable_kv_unavailable_resume_only; excludes_zero_overlap; "
                "not_a_logical_session_drop"
            ),
            "legacy_physical_resume_columns_semantics": (
                "hbm_cpu_ssd_resume_columns_are_raw_manager_source_labels_"
                "and_may_include_zero_hit_object_release; use_attempted_"
                "columns_for_physical_resume_and_effective_surviving_"
                "columns_for_reuse_that_remains_after_active_prefill_"
                "preemption"
            ),
            "other_resume_count": other_resume_count,
            "resume_fraction_of_all_requests": _safe_fraction(
                resume_count, all_request_count),
            "hbm_resume_fraction_of_all_requests": _safe_fraction(
                hbm_resume_count, all_request_count),
            "cpu_resume_fraction_of_all_requests": _safe_fraction(
                cpu_resume_count, all_request_count),
            "ssd_resume_fraction_of_all_requests": _safe_fraction(
                ssd_resume_count, all_request_count),
            "dropped_resume_fraction_of_all_requests": _safe_fraction(
                dropped_resume_count, all_request_count),
            "kv_state_unavailable_resume_fraction_of_all_requests": (
                _safe_fraction(dropped_resume_count, all_request_count)),
            "zero_overlap_resume_fraction_of_all_requests": _safe_fraction(
                resume_source_accounting["zero_overlap_resume_count"],
                all_request_count,
            ),
            "other_resume_fraction_of_all_requests": _safe_fraction(
                other_resume_count, all_request_count),
            "hbm_resume_fraction_of_resume_requests": _safe_fraction(
                hbm_resume_count, resume_count),
            "cpu_resume_fraction_of_resume_requests": _safe_fraction(
                cpu_resume_count, resume_count),
            "ssd_resume_fraction_of_resume_requests": _safe_fraction(
                ssd_resume_count, resume_count),
            "dropped_resume_fraction_of_resume_requests": _safe_fraction(
                dropped_resume_count, resume_count),
            "kv_state_unavailable_resume_fraction_of_resume_requests": (
                _safe_fraction(dropped_resume_count, resume_count)),
            "other_resume_fraction_of_resume_requests": _safe_fraction(
                other_resume_count, resume_count),
            "mean_active_sessions": active.get("mean_active_sessions"),
            "peak_active_sessions": active.get("peak_active_sessions"),
            "fraction_at_configured_k": active.get(
                "fraction_at_configured_k"),
            "required_min_fraction_at_configured_k": manifest.get(
                "min_fraction_at_configured_k"),
            "online_model_compute_ns_measurement_window": compute.get(
                "total_model_compute_ns"),
            "recompute_model_compute_ns_measurement_window": compute.get(
                "recompute_model_compute_ns"),
            "recompute_fraction_of_model_compute_measurement_window": (
                compute.get("recompute_fraction_of_total_model_compute")),
            "model_compute_attribution": compute.get("attribution"),
            "migration_restore_exposure_fraction_of_makespan_full_simulation": (
                time_breakdown.get(
                    "migration_restore_exposure_fraction_of_makespan")),
            "migration_stall_fraction_of_request_latency_full_simulation": (
                time_breakdown.get(
                    "migration_stall_fraction_of_total_request_latency")),
            "aggregate_restore_stall_ns_full_simulation": time_breakdown.get(
                "aggregate_request_migration_stall_ns"),
            "aggregate_restore_hbm_admission_wait_ns_full_simulation": (
                time_breakdown.get(
                    "aggregate_request_migration_hbm_admission_wait_ns")),
            "aggregate_pd_pair_fifo_wait_ns_full_simulation": (
                time_breakdown.get("aggregate_pd_pair_fifo_wait_ns")),
            "aggregate_prepare_boundary_wait_ns_full_simulation": (
                time_breakdown.get("aggregate_prepare_boundary_wait_ns")),
            "aggregate_owner_ready_gate_ns_full_simulation": (
                time_breakdown.get("aggregate_owner_ready_gate_ns")),
            "pd_pair_fifo_wait_fraction_of_request_latency_full_simulation": (
                time_breakdown.get(
                    "pd_pair_fifo_wait_fraction_of_total_request_latency")),
            "prepare_boundary_wait_fraction_of_request_latency_full_simulation": (
                time_breakdown.get(
                    "prepare_boundary_wait_fraction_of_total_request_latency")),
            "aggregate_restore_queue_wait_ns_full_simulation": (
                time_breakdown.get(
                    "aggregate_request_migration_queue_wait_ns")),
            "aggregate_restore_service_ns_full_simulation": (
                time_breakdown.get(
                    "aggregate_request_migration_service_ns")),
            "aggregate_pd_launch_admission_wait_ns_full_simulation": (
                time_breakdown.get(
                    "aggregate_pd_launch_admission_wait_ns")),
            "aggregate_pd_launch_critical_wait_ns_full_simulation": (
                time_breakdown.get(
                    "aggregate_pd_launch_admission_critical_wait_ns")),
            "pd_pair_fifo_wait_ns_measurement_cohort": (
                cohort_overhead.get("pd_pair_fifo_wait_ns")),
            "pd_pair_fifo_wait_fraction_of_request_latency_measurement_cohort": (
                cohort_overhead.get(
                    "pd_pair_fifo_fraction_of_request_latency")),
            "pd_pair_fifo_wait_denominator_request_count_measurement_cohort": (
                cohort_overhead.get("request_count")),
            "pd_pair_fifo_wait_denominator_request_latency_ns_measurement_cohort": (
                cohort_overhead.get("request_latency_ns")),
            "prepare_boundary_wait_ns_measurement_cohort": (
                cohort_overhead.get("prepare_boundary_wait_ns")),
            "prepare_boundary_wait_fraction_of_request_latency_measurement_cohort": (
                cohort_overhead.get(
                    "prepare_boundary_fraction_of_request_latency")),
            "pd_pair_fifo_wait_ns_strict_completion_window": (
                window_overhead.get("pd_pair_fifo_wait_ns")),
            "pd_pair_fifo_wait_fraction_of_request_latency_strict_completion_window": (
                window_overhead.get(
                    "pd_pair_fifo_fraction_of_request_latency")),
            "pd_pair_fifo_wait_denominator_request_count_strict_completion_window": (
                window_overhead.get("request_count")),
            "pd_pair_fifo_wait_denominator_request_latency_ns_strict_completion_window": (
                window_overhead.get("request_latency_ns")),
            "prepare_boundary_wait_ns_strict_completion_window": (
                window_overhead.get("prepare_boundary_wait_ns")),
            "prepare_boundary_wait_fraction_of_request_latency_strict_completion_window": (
                window_overhead.get(
                    "prepare_boundary_fraction_of_request_latency")),
            "recompute_token_fraction_full_simulation": time_breakdown.get(
                "recompute_token_fraction"),
            "policy_avoidable_recompute_fraction_of_executed_prefill_full_simulation": (
                time_breakdown.get(
                    "policy_avoidable_recompute_fraction_of_executed_prefill")),
            "async_restore_gross_ns_full_simulation": asynchronous.get(
                "aggregate_swap_in_gross_ns"),
            "async_restore_owner_barrier_ns_full_simulation": asynchronous.get(
                "aggregate_owner_decode_barrier_ns"),
            "async_restore_prefill_overlap_ns_full_simulation": asynchronous.get(
                "aggregate_prefill_execution_overlap_ns"),
            "sync_swap_exposed_wait_fraction_of_makespan_full_simulation": (
                synchronous.get(
                    "global_wall_exposed_engine_wait_fraction_of_makespan")),
            "global_any_model_busy_fraction_full_simulation": load_activity.get(
                "global_any_model_busy_fraction"),
            "fully_quiescent_ns_full_simulation": load_activity.get(
                "fully_quiescent_ns"),
            "ssd_host_read_bytes_full_simulation": totals.get(
                "ssd_host_read_bytes"),
            "ssd_host_write_bytes_full_simulation": totals.get(
                "ssd_host_write_bytes"),
            "hbm_capacity_demotions_full_simulation": totals.get(
                "hbm_capacity_demotions"),
            "hbm_capacity_drops_full_simulation": totals.get(
                "hbm_capacity_drops"),
            "cpu_capacity_evictions_full_simulation": totals.get(
                "cpu_capacity_evictions"),
            "ssd_capacity_evictions_full_simulation": totals.get(
                "ssd_capacity_evictions"),
            "capacity_induced_recompute_tokens_full_simulation": totals.get(
                "capacity_induced_recompute_tokens"),
            "queue_recompute_evaluation_attempts_full_simulation": (
                queue_recompute.get("evaluation_attempts")),
            "queue_recompute_severe_gate_passes_full_simulation": (
                queue_recompute.get("severe_gate_passes")),
            "queue_recompute_cost_gate_passes_full_simulation": (
                queue_recompute.get("cost_gate_passes")),
            "queue_recompute_full_restore_decisions_full_simulation": (
                queue_recompute.get("full_restore_decisions")),
            "queue_recompute_partial_restore_decisions_full_simulation": (
                queue_recompute.get("partial_restore_decisions")),
            "queue_recompute_zero_restore_decisions_full_simulation": (
                queue_recompute.get("zero_restore_decisions")),
            "queue_recompute_partial_cpu_decisions_full_simulation": (
                queue_recompute.get("partial_cpu_decisions")),
            "queue_recompute_partial_ssd_decisions_full_simulation": (
                queue_recompute.get("partial_ssd_decisions")),
            "queue_recompute_drop_decisions_full_simulation": (
                queue_recompute.get("drop_decisions")),
            "queue_recompute_cpu_drop_decisions_full_simulation": (
                queue_recompute.get("cpu_drop_decisions")),
            "queue_recompute_ssd_drop_decisions_full_simulation": (
                queue_recompute.get("ssd_drop_decisions")),
            "queue_recompute_drop_fraction_of_all_agentic_requests": (
                queue_recompute.get(
                    "drop_fraction_of_all_agentic_requests")),
            "queue_recompute_dropped_bytes_full_simulation": (
                queue_recompute.get("dropped_bytes")),
            "queue_recompute_avoided_restore_bytes_full_simulation": (
                queue_recompute.get("avoided_restore_bytes")),
            "queue_recompute_physical_entry_dropped_bytes_full_simulation": (
                queue_recompute.get("physical_entry_dropped_bytes")),
            "queue_recompute_tokens_full_simulation": (
                queue_recompute.get("declared_recompute_tokens")),
            "queue_recompute_policy_avoidable_tokens_full_simulation": (
                queue_recompute.get(
                    "policy_avoidable_recompute_tokens")),
            "queue_recompute_selected_restore_tokens_full_simulation": (
                queue_recompute.get("selected_restore_tokens")),
            "queue_recompute_dropped_suffix_tokens_full_simulation": (
                queue_recompute.get("dropped_suffix_tokens")),
            "queue_recompute_selected_restore_bytes_full_simulation": (
                queue_recompute.get("selected_restore_bytes")),
            "queue_recompute_dropped_suffix_bytes_full_simulation": (
                queue_recompute.get("dropped_suffix_bytes")),
            "queue_recompute_modified_full_projected_queue_wait_ns_full_simulation": (
                queue_recompute.get(
                    "modified_full_projected_queue_wait_ns")),
            "queue_recompute_modified_full_projected_hbm_admission_wait_ns_full_simulation": (
                queue_recompute.get(
                    "modified_full_projected_hbm_admission_wait_ns")),
            "queue_recompute_modified_full_projected_transient_dram_capacity_wait_ns_full_simulation": (
                queue_recompute.get(
                    "modified_full_projected_transient_dram_capacity_wait_ns")),
            "queue_recompute_modified_full_projected_total_wait_ns_full_simulation": (
                queue_recompute.get(
                    "modified_full_projected_total_wait_ns")),
            "queue_recompute_modified_full_projected_service_ns_full_simulation": (
                queue_recompute.get(
                    "modified_full_projected_service_ns")),
            "queue_recompute_partial_prefix_projected_queue_wait_ns_full_simulation": (
                queue_recompute.get(
                    "partial_prefix_projected_queue_wait_ns")),
            "queue_recompute_partial_prefix_projected_hbm_admission_wait_ns_full_simulation": (
                queue_recompute.get(
                    "partial_prefix_projected_hbm_admission_wait_ns")),
            "queue_recompute_partial_prefix_projected_transient_dram_capacity_wait_ns_full_simulation": (
                queue_recompute.get(
                    "partial_prefix_projected_transient_dram_capacity_wait_ns")),
            "queue_recompute_partial_prefix_projected_service_ns_full_simulation": (
                queue_recompute.get(
                    "partial_prefix_projected_service_ns")),
            "queue_recompute_selected_estimated_suffix_recompute_ns_full_simulation": (
                queue_recompute.get(
                    "selected_estimated_suffix_recompute_comp_ns")),
            "queue_recompute_accounting_invariants_passed": (
                (queue_recompute.get("accounting_invariants") or {}).get(
                    "passed")),
            "queue_recompute_snapshot_first_chunk_audit_performed": (
                snapshot_first_chunk.get("performed")),
            "queue_recompute_snapshot_first_chunk_joined_count": (
                snapshot_first_chunk.get("joined_count")),
            "queue_recompute_snapshot_first_chunk_waiting_count": (
                snapshot_first_chunk.get("waiting_count")),
            "queue_recompute_snapshot_first_chunk_actual_wait_ns": (
                snapshot_first_chunk.get("actual_wait_ns")),
            "queue_recompute_snapshot_first_chunk_actual_critical_wait_ns": (
                snapshot_first_chunk.get(
                    "actual_critical_wait_after_restore_ns")),
            "queue_recompute_snapshot_first_chunk_max_actual_wait_ns": (
                snapshot_first_chunk.get("max_actual_wait_ns")),
            "pd_chunk_admission_audit_passed": pd_chunk_validation.get(
                "passed"),
            "pd_active_prefill_recompute_audit_passed": (
                pd_active_prefill_validation.get("passed")),
            "pd_chunk_admissions_full_simulation": totals.get(
                "pd_chunk_admissions"),
            "pd_chunk_waiting_admissions_full_simulation": totals.get(
                "pd_chunk_waiting_admissions"),
            "pd_chunk_admitted_tokens_full_simulation": totals.get(
                "pd_chunk_admitted_tokens"),
            "pd_chunk_prefill_reserved_bytes_full_simulation": totals.get(
                "pd_chunk_prefill_reserved_bytes"),
            "pd_chunk_decode_reserved_bytes_full_simulation": totals.get(
                "pd_chunk_decode_reserved_bytes"),
            "pd_chunk_admission_wait_ns_full_simulation": totals.get(
                "pd_chunk_admission_wait_ns"),
            "pd_chunk_admission_critical_wait_ns_full_simulation": totals.get(
                "pd_chunk_admission_critical_wait_ns"),
            "pd_chunk_cancelled_admissions_full_simulation": totals.get(
                "pd_chunk_cancelled_admissions"),
            "pd_chunk_cancelled_waiting_admissions_full_simulation": (
                totals.get("pd_chunk_cancelled_waiting_admissions")),
            "pd_chunk_cancelled_admission_wait_ns_full_simulation": (
                totals.get("pd_chunk_cancelled_admission_wait_ns")),
            "pd_chunk_cancelled_admission_critical_wait_ns_full_simulation": (
                totals.get(
                    "pd_chunk_cancelled_admission_critical_wait_ns")),
            "pd_chunk_attempt_admission_wait_ns_full_simulation": (
                int(totals.get("pd_chunk_admission_wait_ns", 0) or 0)
                + int(totals.get(
                    "pd_chunk_cancelled_admission_wait_ns", 0) or 0)),
            "pd_chunk_attempt_admission_critical_wait_ns_full_simulation": (
                int(totals.get(
                    "pd_chunk_admission_critical_wait_ns", 0) or 0)
                + int(totals.get(
                    "pd_chunk_cancelled_admission_critical_wait_ns", 0)
                    or 0)),
            "pd_chunk_legacy_wait_columns_semantics": (
                "pd_chunk_admission_wait_ns_full_simulation_is_successful_"
                "claims_only; attempt_wait_adds_cancelled_precommit_claims"),
            "pd_active_prefill_recompute_preemptions_full_simulation": (
                totals.get("pd_active_prefill_recompute_preemptions")),
            "pd_active_prefill_recompute_tokens_full_simulation": totals.get(
                "pd_active_prefill_recompute_tokens"),
            "restored_hit_tokens_discarded_by_active_prefill_recompute_full_simulation": (
                totals.get(
                    "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute")),
            "pd_chunk_snapshot_joined_admissions_full_simulation": totals.get(
                "pd_chunk_snapshot_joined_admissions"),
            "pd_chunk_snapshot_feasible_admissions_full_simulation": totals.get(
                "pd_chunk_snapshot_feasible_admissions"),
            "pd_chunk_snapshot_feasible_waiting_admissions_full_simulation": totals.get(
                "pd_chunk_snapshot_feasible_waiting_admissions"),
            "pd_chunk_snapshot_feasible_wait_ns_full_simulation": totals.get(
                "pd_chunk_snapshot_feasible_wait_ns"),
            "queue_recompute_selected_projected_queue_wait_ns_full_simulation": (
                queue_recompute.get(
                    "selected_projected_queue_wait_ns")),
            "queue_recompute_selected_projected_hbm_admission_wait_ns_full_simulation": (
                queue_recompute.get(
                    "selected_projected_hbm_admission_wait_ns")),
            "queue_recompute_selected_projected_transient_dram_capacity_wait_ns_full_simulation": (
                queue_recompute.get(
                    "selected_projected_transient_dram_capacity_wait_ns")),
            "queue_recompute_selected_projected_total_wait_ns_full_simulation": (
                queue_recompute.get(
                    "selected_projected_total_wait_ns")),
            "queue_recompute_selected_projected_service_ns_full_simulation": (
                queue_recompute.get("selected_projected_service_ns")),
            "queue_recompute_selected_estimated_recompute_ns_full_simulation": (
                queue_recompute.get(
                    "selected_estimated_incremental_recompute_comp_ns",
                    queue_recompute.get(
                        "selected_estimated_suffix_recompute_comp_ns"))),
            "measured_session_ids_hash": _stable_json_hash(
                _measured_session_ids(report)),
            "measured_completion_cohort_pairing_required": None,
            "oracle_latency_comparison_status": None,
            "input_session_ids_hash": manifest[
                "selected_session_ids_hash"],
            "oracle_throughput_slowdown_fraction": None,
            "oracle_measurement_time_slowdown_fraction": None,
            "oracle_ttft_mean_slowdown_fraction": None,
            "oracle_ttft_p99_slowdown_fraction": None,
            "oracle_tpot_mean_slowdown_fraction": None,
            "oracle_resume_ttft_mean_slowdown_fraction": None,
            "oracle_session_jct_mean_slowdown_fraction": None,
            "oracle_session_jct_p99_slowdown_fraction": None,
            "policy_order": manifest.get("policy_order"),
            "run_wall_duration_seconds": manifest.get(
                "wall_duration_seconds"),
            "timing_warning_count": len(timing_warnings),
            "timing_warnings_json": json.dumps(
                timing_warnings, sort_keys=True),
            "timing_warning_codes_json": json.dumps(
                timing_warning_codes, sort_keys=True),
            "allow_timing_warnings": bool(
                manifest.get("allow_timing_warnings", False)),
            "allowed_timing_warning_codes_json": json.dumps(
                manifest.get("allowed_timing_warning_codes"),
                sort_keys=True,
            ),
            "trace_identity_validation_passed": bool(
                report["validation"].get("trace_identity", {}).get(
                    "passed", False)),
            "policy_invariant_validation_passed": bool(
                report["validation"].get("policy_invariants", {}).get(
                    "passed", False)),
            "foreground_transfer_tail_jobs": int(
                (agentic_report.get("measurement_cutoff_dma_tail") or {}).get(
                    "foreground_jobs", 0) or 0),
            "background_transfer_tail_jobs": int(
                (agentic_report.get("measurement_cutoff_dma_tail") or {}).get(
                    "background_jobs", 0) or 0),
            "external_fabric_model_coexecution_pairs_full_simulation": int(
                external_coexecution.get("coexecution_pair_count", 0) or 0),
            "external_fabric_overlapped_jobs_full_simulation": int(
                external_coexecution.get("overlapped_job_count", 0) or 0),
            "external_fabric_overlapped_model_windows_full_simulation": int(
                external_coexecution.get(
                    "overlapped_model_window_count", 0) or 0),
            "external_fabric_model_coexecution_union_ns_full_simulation": int(
                external_coexecution.get("coexecution_union_ns", 0) or 0),
            "resume_ttft_exact_count": exact_rate_metrics[
                "resume_ttft"]["count"],
            "resume_ttft_exact_mean_ns": exact_rate_metrics[
                "resume_ttft"]["mean"],
            "resume_ttft_p95_ns": exact_rate_metrics[
                "resume_ttft"]["p95"],
            "resume_ttft_denominator": exact_rate_metrics[
                "resume_ttft_denominator"],
            "tpot_exact_eligible_count": exact_rate_metrics[
                "tpot"]["count"],
            "tpot_exact_mean_ns": exact_rate_metrics["tpot"]["mean"],
            "tpot_p95_ns": exact_rate_metrics["tpot"]["p95"],
            "tpot_denominator": exact_rate_metrics["tpot_denominator"],
            "server_added_session_jct_count": exact_rate_metrics[
                "server_added_jct"]["count"],
            "server_added_session_jct_sum_ns": exact_rate_metrics[
                "server_added_jct"]["sum"],
            "server_added_session_jct_mean_ns": exact_rate_metrics[
                "server_added_jct"]["mean"],
            "server_added_session_jct_p95_ns": exact_rate_metrics[
                "server_added_jct"]["p95"],
            "server_added_session_jct_denominator": exact_rate_metrics[
                "server_added_jct_denominator"],
            "server_added_session_jct_definition": exact_rate_metrics[
                "server_added_jct_definition"],
            "trace_closed_loop_idle_gap_sum_ns": exact_rate_metrics[
                "trace_idle_gaps"]["sum"],
            "trace_closed_loop_idle_gap_mean_ns": exact_rate_metrics[
                "trace_idle_gaps"]["mean"],
            "total_hbm_capacity_admission_wait_count": exact_rate_metrics[
                "hbm_admission"]["count"],
            "total_hbm_capacity_admission_wait_sum_ns": exact_rate_metrics[
                "hbm_admission"]["sum"],
            "total_hbm_capacity_admission_wait_mean_ns": exact_rate_metrics[
                "hbm_admission"]["mean"],
            "total_hbm_capacity_admission_wait_p95_ns": exact_rate_metrics[
                "hbm_admission"]["p95"],
            "restore_hbm_capacity_admission_wait_mean_ns": (
                exact_rate_metrics["restore_hbm_admission"]["mean"]),
            "restore_hbm_capacity_admission_wait_p95_ns": (
                exact_rate_metrics["restore_hbm_admission"]["p95"]),
            "pd_chunk_hbm_capacity_admission_wait_mean_ns": (
                exact_rate_metrics["pd_chunk_hbm_admission"]["mean"]),
            "pd_chunk_hbm_capacity_admission_wait_p95_ns": (
                exact_rate_metrics["pd_chunk_hbm_admission"]["p95"]),
            "pd_chunk_attempt_admission_wait_sum_ns": (
                exact_rate_metrics["pd_chunk_attempt_admission"]["sum"]),
            "pd_chunk_attempt_admission_wait_mean_ns": (
                exact_rate_metrics["pd_chunk_attempt_admission"]["mean"]),
            "pd_chunk_attempt_admission_wait_p95_ns": (
                exact_rate_metrics["pd_chunk_attempt_admission"]["p95"]),
            "pd_chunk_successful_admission_wait_sum_ns": (
                exact_rate_metrics["pd_chunk_successful_admission"]["sum"]),
            "pd_chunk_successful_admission_wait_mean_ns": (
                exact_rate_metrics["pd_chunk_successful_admission"]["mean"]),
            "pd_chunk_cancelled_admission_wait_sum_ns": (
                exact_rate_metrics["pd_chunk_cancelled_admission"]["sum"]),
            "pd_chunk_cancelled_admission_wait_mean_ns": (
                exact_rate_metrics["pd_chunk_cancelled_admission"]["mean"]),
            "pd_chunk_attempt_admission_critical_wait_sum_ns": (
                exact_rate_metrics["pd_chunk_gross_critical_wait"]["sum"]),
            "pd_chunk_attempt_admission_critical_wait_mean_ns": (
                exact_rate_metrics[
                    "pd_chunk_gross_critical_wait"]["mean"]),
            "pd_chunk_attempt_admission_critical_wait_p95_ns": (
                exact_rate_metrics[
                    "pd_chunk_gross_critical_wait"]["p95"]),
            "pd_chunk_successful_admission_critical_wait_sum_ns": (
                exact_rate_metrics[
                    "pd_chunk_successful_critical_wait"]["sum"]),
            "pd_chunk_cancelled_admission_critical_wait_sum_ns": (
                exact_rate_metrics[
                    "pd_chunk_cancelled_critical_wait"]["sum"]),
            "total_hbm_capacity_admission_wait_scope": exact_rate_metrics[
                "hbm_admission_scope"],
            "operational_metric_source_status": operational_metrics[
                "source_status"],
            "average_active_batch_size": operational_metrics[
                "average_active_batch_size"],
            "average_active_batch_size_including_dummy": (
                operational_metrics[
                    "average_active_batch_size_including_dummy"]),
            "active_batch_completed_count": operational_metrics[
                "active_batch_completed_count"],
            "active_batch_dummy_count": operational_metrics[
                "active_batch_dummy_count"],
            "active_batch_size_scope": operational_metrics[
                "active_batch_scope"],
            "hbm_kv_capacity_per_rank_bytes_sum": (
                hbm_occupancy["capacity_per_rank_bytes_sum"]
                if hbm_occupancy else None),
            "hbm_kv_average_physical_idle_reusable_per_rank_bytes": (
                hbm_categories.get("physical_idle_reusable")),
            "hbm_kv_average_physical_non_idle_active_per_rank_bytes": (
                hbm_categories.get("physical_non_idle_active")),
            "hbm_kv_average_physical_free_per_rank_bytes": (
                hbm_categories.get("physical_free")),
            "hbm_kv_average_logical_destination_reservation_per_rank_bytes": (
                hbm_categories.get(
                    "logical_destination_admission_reservation")),
            "hbm_kv_average_reserved_free_slack_per_rank_bytes": (
                hbm_categories.get("reserved_free_slack")),
            "hbm_kv_average_future_reclaim_backed_reservation_per_rank_bytes": (
                hbm_categories.get("future_reclaim_backed_reservation")),
            "hbm_kv_average_unclaimed_allocatable_slack_per_rank_bytes": (
                hbm_categories.get("unclaimed_allocatable_slack")),
            "hbm_kv_average_physical_idle_reusable_fraction": (
                (hbm_category_reports.get("physical_idle_reusable") or {}).get(
                    "average_fraction_of_capacity")),
            "hbm_kv_average_physical_non_idle_active_fraction": (
                (hbm_category_reports.get(
                    "physical_non_idle_active") or {}).get(
                    "average_fraction_of_capacity")),
            "hbm_kv_average_physical_free_fraction": (
                (hbm_category_reports.get("physical_free") or {}).get(
                    "average_fraction_of_capacity")),
            "hbm_kv_average_logical_destination_reservation_fraction": (
                (hbm_category_reports.get(
                    "logical_destination_admission_reservation") or {}).get(
                    "average_fraction_of_capacity")),
            "hbm_kv_average_reserved_free_slack_fraction": (
                (hbm_category_reports.get("reserved_free_slack") or {}).get(
                    "average_fraction_of_capacity")),
            "hbm_kv_average_future_reclaim_backed_reservation_fraction": (
                (hbm_category_reports.get(
                    "future_reclaim_backed_reservation") or {}).get(
                    "average_fraction_of_capacity")),
            "hbm_kv_average_unclaimed_allocatable_slack_fraction": (
                (hbm_category_reports.get(
                    "unclaimed_allocatable_slack") or {}).get(
                    "average_fraction_of_capacity")),
            "hbm_kv_average_physical_occupied_per_rank_bytes": (
                hbm_occupancy[
                    "average_physical_occupied_per_rank_bytes"]
                if hbm_occupancy else None),
            "hbm_kv_average_physical_occupied_utilization_fraction": (
                hbm_occupancy[
                    "average_physical_occupied_utilization_fraction"]
                if hbm_occupancy else None),
            "hbm_kv_average_reservation_adjusted_claim_per_rank_bytes": (
                hbm_occupancy[
                    "average_reservation_adjusted_claim_per_rank_bytes"]
                if hbm_occupancy else None),
            "hbm_kv_average_reservation_adjusted_claim_fraction": (
                hbm_occupancy[
                    "average_reservation_adjusted_claim_fraction"]
                if hbm_occupancy else None),
            "hbm_kv_occupancy_scope": (
                hbm_occupancy["scope"] if hbm_occupancy else None),
        }
        row.update(_distribution_columns("ttft", request_all.get("ttft_ns")))
        row.update(_distribution_columns("tpot", request_all.get("tpot_ns")))
        row.update(_distribution_columns(
            "resume_ttft", resume.get("ttft_ns")))
        row.update(_distribution_columns(
            "release_to_first_schedule",
            request_all.get("release_to_first_schedule_ns")))
        row.update(_distribution_columns(
            "scheduler_queue", request_all.get("scheduler_queue_wait_ns")))
        row.update(_distribution_columns(
            "restore_gate", request_all.get("restore_gate_wait_ns")))
        row.update(_distribution_columns(
            "owner_ready_gate", request_all.get("owner_ready_gate_ns")))
        row.update(_distribution_columns(
            "pd_pair_fifo", request_all.get("pd_pair_fifo_wait_ns")))
        row.update(_distribution_columns(
            "resume_pd_pair_fifo", resume.get("pd_pair_fifo_wait_ns")))
        row.update(_distribution_columns(
            "prepare_boundary",
            request_all.get("prepare_boundary_wait_ns")))
        row.update(_distribution_columns(
            "resume_prepare_boundary",
            resume.get("prepare_boundary_wait_ns")))
        row.update(_distribution_columns(
            "source_demotion_join",
            request_all.get("source_demotion_join_wait_ns")))
        row.update(_distribution_columns(
            "resume_source_demotion_join",
            resume.get("source_demotion_join_wait_ns")))
        row.update(_distribution_columns(
            "hbm_admission", request_all.get("hbm_admission_wait_ns")))
        row.update(_distribution_columns(
            "transient_dram_capacity",
            request_all.get("transient_dram_capacity_wait_ns")))
        row.update(_distribution_columns(
            "resume_transient_dram_capacity",
            resume.get("transient_dram_capacity_wait_ns")))
        row.update(_distribution_columns(
            "restore_queue", request_all.get("restore_queue_wait_ns")))
        row.update(_distribution_columns(
            "restore_service", request_all.get("restore_service_ns")))
        row.update(_distribution_columns(
            "pd_launch_admission",
            request_all.get("pd_launch_admission_wait_ns")))
        row.update(_distribution_columns(
            "pd_launch_admission_critical",
            request_all.get("pd_launch_admission_critical_wait_ns")))
        row.update(_distribution_columns(
            "session_admission_queue",
            report["sessions"].get("admission_queue_wait_ns")))
        row.update(_distribution_columns(
            "session_jct",
            report["sessions"].get("e2e_from_offer_ns")))
        row.update(_distribution_columns(
            "session_execution",
            report["sessions"].get("e2e_from_admission_ns")))
        for prefix, metric_name in (
                ("session_admission_queue", "admission_queue_wait_ns"),
                ("session_jct", "e2e_from_offer_ns"),
                ("session_execution", "e2e_from_admission_ns")):
            distribution = report["sessions"].get(metric_name) or {}
            row[f"{prefix}_count"] = distribution.get("count")
            row[f"{prefix}_sum_ns"] = distribution.get("sum")
        gap_groups = report["requests"].get(
            "resume_by_return_gap_type", {})
        gap_source_groups = report["requests"].get(
            "resume_by_return_gap_type_and_source", {})
        residency_groups = report["requests"].get(
            "resume_by_residency_at_return", {})
        for gap_type in ("tool", "human", "mixed", "unknown"):
            gap_count = _count_group(gap_groups, gap_type)
            row[f"{gap_type}_resume_count"] = gap_count
            row[f"{gap_type}_resume_fraction_of_all_requests"] = (
                _safe_fraction(gap_count, all_request_count))
            source_groups = gap_source_groups.get(gap_type, {})
            for source in ("hbm", "cpu", "ssd", "dropped"):
                count = _count_group(source_groups, source)
                row[f"{gap_type}_{source}_resume_count"] = count
                row[
                    f"{gap_type}_{source}_resume_fraction_of_all_requests"
                ] = _safe_fraction(count, all_request_count)
        for residency in ("hbm", "cpu", "ssd", "dropped"):
            count = _count_group(residency_groups, residency)
            row[f"{residency}_resident_at_return_count"] = count
            row[
                f"{residency}_resident_at_return_fraction_of_all_requests"
            ] = _safe_fraction(count, all_request_count)
        _validate_summary_row(row)
        rows.append(row)

    by_pair = {}
    for row in rows:
        by_pair.setdefault((
            row["mode"], row["load_value"], row.get("arrival_seed"),
        ), []).append(row)
    for pair, pair_rows in by_pair.items():
        oracle_rows = [row for row in pair_rows if row["policy"] == oracle_label]
        if len(oracle_rows) != 1:
            raise ExperimentError(
                f"Expected one oracle for pair {pair}, found {len(oracle_rows)}")
        oracle = oracle_rows[0]
        pair_manifests = [
            manifest for manifest in manifests
            if (manifest.get("status") == "succeeded"
                and manifest["mode"] == pair[0]
                and manifest["load_value"] == pair[1]
                and manifest.get("arrival_seed") == pair[2])
        ]
        immutable_fields = (
            "workload_sha256", "selected_session_ids_hash",
            "selected_session_identity_hash",
            "cluster_config_sha256", "arrival_seed", "pair_key",
            "max_active_sessions",
            "warmup_completions", "measure_completions",
            "measurement_cohort_selection",
            "expected_runtime_session_count",
            "expected_runtime_session_ids_hash",
            "expected_measurement_warmup_session_ids_hash",
            "expected_measurement_target_session_ids_hash",
            "expected_measurement_required_session_ids_hash",
        )
        for field in immutable_fields:
            values = {manifest.get(field) for manifest in pair_manifests}
            if len(values) != 1:
                raise ExperimentError(
                    f"Paired online provenance mismatch for {pair}, "
                    f"field={field}: {sorted(values, key=str)}")
        for manifest in pair_manifests:
            if int(manifest.get("schema_version", 0) or 0) < 7:
                continue
            for hash_field in (
                    "agentic_hardware_config_hash",
                    "agentic_shared_control_config_hash",
                    "agentic_effective_config_hash"):
                if not _is_sha256_digest(manifest.get(hash_field)):
                    raise ExperimentError(
                        f"Missing or malformed {hash_field} for "
                        f"{manifest.get('run_id')}")
        hardware_hashes = {
            manifest.get("agentic_hardware_config_hash")
            for manifest in pair_manifests
        }
        if len(hardware_hashes) != 1:
            raise ExperimentError(
                f"Paired agentic hardware config mismatch for {pair}")
        malformed_arrival_rows = [
            row["run_id"] for row in pair_rows
            if (not isinstance(row.get("offered_arrival_trace_count"), int)
                or isinstance(row.get("offered_arrival_trace_count"), bool)
                or row["offered_arrival_trace_count"] <= 0
                or not _is_sha256_digest(
                    row.get("offered_arrival_trace_sha256")))
        ]
        if malformed_arrival_rows:
            raise ExperimentError(
                "Paired offered-arrival trace proof is missing or malformed "
                f"for {pair}: {malformed_arrival_rows}")
        arrival_trace_counts = {
            row["offered_arrival_trace_count"] for row in pair_rows}
        arrival_trace_hashes = {
            row["offered_arrival_trace_sha256"] for row in pair_rows}
        if len(arrival_trace_counts) != 1 or len(arrival_trace_hashes) != 1:
            raise ExperimentError(
                "Paired offered-arrival traces differ for "
                f"{pair}: counts={sorted(arrival_trace_counts)}, "
                f"hashes={sorted(arrival_trace_hashes)}")
        shared_control_hashes = {
            manifest.get("agentic_shared_control_config_hash")
            for manifest in pair_manifests
            if not manifest.get("strict_oracle")
        }
        if len(shared_control_hashes) != 1:
            raise ExperimentError(
                f"Paired agentic shared-control config mismatch for {pair}")
        cohort_selection = pair_manifests[0].get(
            "measurement_cohort_selection")
        # Admission-order targets are fixed before execution and therefore
        # must remain identical across paired policies. Completion-order
        # targets are selected after execution by definition; requiring the
        # same IDs would reject valid steady-state throughput runs whenever
        # policies change completion order. Throughput remains comparable
        # over the same W-completion warmup and M-completion window, while
        # latency slowdowns are paired only for an exact target-ID match.
        cohort_pairing_required = cohort_selection != "completion_order"
        for row in pair_rows:
            row["measured_completion_cohort_matches_oracle"] = (
                row["measured_session_ids_hash"]
                == oracle["measured_session_ids_hash"]
            )
            row["measured_completion_cohort_pairing_required"] = (
                cohort_pairing_required)
            if (cohort_pairing_required
                    and not row[
                        "measured_completion_cohort_matches_oracle"]):
                row_report = reports[row["run_id"]]
                oracle_report = reports[oracle["run_id"]]
                row_ids = set(_measured_session_ids(row_report))
                oracle_ids = set(_measured_session_ids(oracle_report))
                raise ExperimentError(
                    f"Measured completion cohort differs from oracle for "
                    f"{pair}, policy={row['policy']}: only_policy="
                    f"{sorted(row_ids - oracle_ids)[:5]}, only_oracle="
                    f"{sorted(oracle_ids - row_ids)[:5]}")
            if row is oracle:
                row["oracle_throughput_slowdown_fraction"] = 0.0
                row["oracle_measurement_time_slowdown_fraction"] = 0.0
                row["oracle_ttft_mean_slowdown_fraction"] = 0.0
                row["oracle_ttft_p99_slowdown_fraction"] = 0.0
                row["oracle_tpot_mean_slowdown_fraction"] = 0.0
                row["oracle_resume_ttft_mean_slowdown_fraction"] = 0.0
                row["oracle_session_jct_mean_slowdown_fraction"] = 0.0
                row["oracle_session_jct_p99_slowdown_fraction"] = 0.0
                row["oracle_latency_comparison_status"] = "reference"
                continue
            throughput = row["sessions_per_second"]
            oracle_throughput = oracle["sessions_per_second"]
            row["oracle_throughput_slowdown_fraction"] = (
                oracle_throughput / throughput - 1.0
                if throughput and oracle_throughput is not None else None
            )
            duration = row["measurement_duration_ns"]
            oracle_duration = oracle["measurement_duration_ns"]
            row["oracle_measurement_time_slowdown_fraction"] = (
                duration / oracle_duration - 1.0
                if oracle_duration and duration is not None else None
            )
            if row["measured_completion_cohort_matches_oracle"]:
                row["oracle_ttft_mean_slowdown_fraction"] = _ratio_slowdown(
                    row["ttft_mean_ns"], oracle["ttft_mean_ns"])
                row["oracle_ttft_p99_slowdown_fraction"] = _ratio_slowdown(
                    row["ttft_p99_ns"], oracle["ttft_p99_ns"])
                row["oracle_tpot_mean_slowdown_fraction"] = _ratio_slowdown(
                    row["tpot_mean_ns"], oracle["tpot_mean_ns"])
                row["oracle_resume_ttft_mean_slowdown_fraction"] = (
                    _ratio_slowdown(
                        row["resume_ttft_mean_ns"],
                        oracle["resume_ttft_mean_ns"],
                    )
                )
                row["oracle_session_jct_mean_slowdown_fraction"] = (
                    _ratio_slowdown(
                        row["session_jct_mean_ns"],
                        oracle["session_jct_mean_ns"],
                    )
                )
                row["oracle_session_jct_p99_slowdown_fraction"] = (
                    _ratio_slowdown(
                        row["session_jct_p99_ns"],
                        oracle["session_jct_p99_ns"],
                    )
                )
                row["oracle_latency_comparison_status"] = (
                    "paired_exact_completion_cohort")
            else:
                row["oracle_ttft_mean_slowdown_fraction"] = None
                row["oracle_ttft_p99_slowdown_fraction"] = None
                row["oracle_tpot_mean_slowdown_fraction"] = None
                row["oracle_resume_ttft_mean_slowdown_fraction"] = None
                row["oracle_session_jct_mean_slowdown_fraction"] = None
                row["oracle_session_jct_p99_slowdown_fraction"] = None
                row["oracle_latency_comparison_status"] = (
                    "unpaired_policy_dependent_completion_order")
            _validate_summary_row(row)

    poisson_crn_validation = _validate_poisson_common_random_numbers(
        rows, manifests)
    for row in rows:
        is_poisson = row["mode"] == "poisson"
        row["poisson_crn_validation_performed"] = (
            bool(poisson_crn_validation.get("performed"))
            if is_poisson else False)
        row["poisson_crn_rate_count"] = (
            poisson_crn_validation.get("rate_count")
            if is_poisson else None)

    sample_counts = {}
    for row in rows:
        key = (row["mode"], row["load_value"], row["policy"])
        sample_counts[key] = sample_counts.get(key, 0) + 1
    for row in rows:
        row["arrival_seed_sample_count"] = sample_counts[
            (row["mode"], row["load_value"], row["policy"])]
    return rows


def save_results(rows, output_dir):
    output_dir = Path(output_dir)
    _write_json(output_dir / "summary.json", {
        "schema_version": SCHEMA_VERSION,
        "rows": rows,
    })
    def write_csv(path, selected_rows):
        if not selected_rows:
            return None
        with open(path, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(
                output, fieldnames=list(selected_rows[0]))
            writer.writeheader()
            writer.writerows(selected_rows)
        return path

    paths = {"combined": write_csv(output_dir / "summary.csv", rows)}
    for mode in ("backlog", "poisson"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        _write_json(output_dir / f"{mode}_summary.json", {
            "schema_version": SCHEMA_VERSION,
            "mode": mode,
            "rows": mode_rows,
        })
        paths[mode] = write_csv(
            output_dir / f"{mode}_summary.csv", mode_rows)
    return paths


def _ordered_plot_policies(rows):
    """Return one stable policy order or reject ambiguous plot metadata."""
    order_by_policy = {}
    policy_by_order = {}
    for row in rows:
        policy = str(row["policy"])
        order = int(row.get("policy_order") or 0)
        prior_order = order_by_policy.setdefault(policy, order)
        if prior_order != order:
            raise ExperimentError(
                "Plot rows assign inconsistent policy_order to policy "
                f"{policy!r}: {prior_order} versus {order}")
        prior_policy = policy_by_order.setdefault(order, policy)
        if prior_policy != policy:
            raise ExperimentError(
                "Plot rows assign one policy_order to multiple policies: "
                f"order={order}, policies={prior_policy!r}, {policy!r}")
    return [
        policy for policy, _ in sorted(
            order_by_policy.items(), key=lambda item: item[1])
    ]


def plot_grouped_throughput(rows, output_dir):
    """Write dependency-free grouped-bar SVGs for each load mode."""
    paths = []
    palette = (
        "#4C78A8", "#F58518", "#54A24B", "#E45756",
        "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D",
    )
    for mode in ("backlog", "poisson"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        if not mode_rows:
            continue
        loads = sorted({row["load_value"] for row in mode_rows})
        policies = _ordered_plot_policies(mode_rows)
        values_by_policy = {}
        for policy_index, policy in enumerate(policies):
            values = []
            for load in loads:
                matching = [
                    row for row in mode_rows
                    if row["load_value"] == load and row["policy"] == policy
                ]
                values.append(
                    sum(float(row["sessions_per_second"])
                        for row in matching) / len(matching)
                    if matching else 0)
            values_by_policy[policy] = values

        legend_item_widths = [
            max(120, len(policy) * 7 + 32) for policy in policies
        ]
        width = max(
            760,
            180 * len(loads) + 260,
            sum(legend_item_widths) + 120,
        )
        height = 520
        left, right, top, bottom = 90, 30, 65, 125
        plot_width = width - left - right
        plot_height = height - top - bottom
        all_values = [
            float(value)
            for values in values_by_policy.values()
            for value in values
        ]
        maximum = max(all_values, default=0.0)
        y_max = maximum * 1.1 if maximum > 0 else 1.0
        group_width = plot_width / max(1, len(loads))
        occupied_width = group_width * 0.8
        bar_width = occupied_width / max(1, len(policies))
        load_labels = [
            str(int(value)) if mode == "backlog" else f"{value:g}"
            for value in loads
        ]

        elements = [
            (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" viewBox="0 0 {width} {height}">'),
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#222}'
            '.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}'
            '.tick{font-size:12px}.label{font-size:14px}'
            '.title{font-size:18px;font-weight:600}.legend{font-size:12px}'
            '</style>',
            (f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" '
             f'class="title">Online {html.escape(mode)} throughput</text>'),
        ]
        for tick in range(6):
            fraction = tick / 5
            y = top + plot_height * (1 - fraction)
            value = y_max * fraction
            elements.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
                f'y2="{y:.2f}" class="grid"/>')
            elements.append(
                f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
                f'class="tick">{value:.3g}</text>')
        elements.extend([
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top+plot_height}" class="axis"/>',
            f'<line x1="{left}" y1="{top+plot_height}" '
            f'x2="{width-right}" y2="{top+plot_height}" class="axis"/>',
        ])
        for load_index, label in enumerate(load_labels):
            group_left = left + load_index * group_width
            for policy_index, policy in enumerate(policies):
                value = float(values_by_policy[policy][load_index])
                bar_height = plot_height * value / y_max
                x = (group_left + (group_width - occupied_width) / 2
                     + policy_index * bar_width)
                y = top + plot_height - bar_height
                color = palette[policy_index % len(palette)]
                elements.append(
                    f'<rect x="{x+1:.2f}" y="{y:.2f}" '
                    f'width="{max(1.0, bar_width-2):.2f}" '
                    f'height="{bar_height:.2f}" fill="{color}">'
                    f'<title>{html.escape(policy)}: {value:.6g}</title>'
                    '</rect>')
            center = group_left + group_width / 2
            elements.append(
                f'<text x="{center:.2f}" y="{top+plot_height+22}" '
                f'text-anchor="middle" class="tick">{html.escape(label)}</text>')
        x_label = (
            "Active sessions K" if mode == "backlog"
            else "Offered sessions/s"
        )
        elements.append(
            f'<text x="{left+plot_width/2:.2f}" y="{height-58}" '
            f'text-anchor="middle" class="label">{x_label}</text>')
        elements.append(
            f'<text x="20" y="{top+plot_height/2:.2f}" '
            f'text-anchor="middle" class="label" '
            f'transform="rotate(-90 20 {top+plot_height/2:.2f})">'
            'Completed sessions/s</text>')
        legend_x = left
        legend_y = height - 24
        for policy_index, policy in enumerate(policies):
            item_width = legend_item_widths[policy_index]
            color = palette[policy_index % len(palette)]
            elements.append(
                f'<rect x="{legend_x}" y="{legend_y-11}" width="12" '
                f'height="12" fill="{color}"/>')
            elements.append(
                f'<text x="{legend_x+18}" y="{legend_y}" '
                f'class="legend">{html.escape(policy)}</text>')
            legend_x += item_width
        elements.append('</svg>')

        path = Path(output_dir) / f"{mode}_throughput_grouped.svg"
        with open(path, "w", encoding="utf-8") as output:
            output.write("\n".join(elements))
            output.write("\n")
        paths.append(path)
    return paths


_STUDENT_T_975 = {
    1: 12.7062047364,
    2: 4.30265272975,
    3: 3.18244630528,
    4: 2.7764451052,
    5: 2.57058183564,
    6: 2.44691184879,
    7: 2.36462425101,
    8: 2.3060041352,
    9: 2.26215716285,
    10: 2.22813885196,
    11: 2.20098516008,
    12: 2.17881282966,
    13: 2.16036865646,
    14: 2.14478668792,
    15: 2.13144954556,
    16: 2.11990529922,
    17: 2.10981557783,
    18: 2.10092204024,
    19: 2.09302405441,
    20: 2.08596344727,
    21: 2.07961384473,
    22: 2.0738730679,
    23: 2.06865761042,
    24: 2.06389856163,
    25: 2.05953855275,
    26: 2.05552943864,
    27: 2.05183051648,
    28: 2.0484071418,
    29: 2.04522964213,
    30: 2.0422724563,
}


def _student_t_975(degrees_of_freedom):
    """Return the two-sided 95% Student-t critical value."""
    degrees_of_freedom = int(degrees_of_freedom)
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    exact = _STUDENT_T_975.get(degrees_of_freedom)
    if exact is not None:
        return exact

    # Cornish-Fisher expansion about the standard-normal 97.5th percentile.
    # This avoids a scipy dependency while remaining much more accurate than
    # silently substituting 1.96 for the Student-t interval.
    z = 1.959963984540054
    df = float(degrees_of_freedom)
    return (
        z
        + (z ** 3 + z) / (4 * df)
        + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
        + (
            3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z
        ) / (384 * df ** 3)
    )


def _poisson_session_jct_plot_cells(
        rows, *, oracle_label="infinite_hbm_oracle"):
    """Aggregate per-run session-JCT means with seeds as replicates."""
    poisson_rows = [row for row in rows if row.get("mode") == "poisson"]
    if not poisson_rows:
        return [], [], []
    policies = _ordered_plot_policies(poisson_rows)
    oracle_label = str(oracle_label)
    if oracle_label not in policies:
        raise ExperimentError(
            "Poisson session-JCT plot is missing reference policy "
            f"{oracle_label!r}")
    rates = sorted({float(row["load_value"]) for row in poisson_rows})
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ExperimentError(
            "Poisson session-JCT plot requires positive finite rates")

    pair_provenance = {}
    pair_key_by_provenance = {}
    samples = {}
    for row in poisson_rows:
        rate = float(row["load_value"])
        policy = str(row["policy"])
        raw_seed = row.get("arrival_seed")
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError):
            seed = None
        if (raw_seed is None or isinstance(raw_seed, bool)
                or seed is None or seed != raw_seed):
            raise ExperimentError(
                "Poisson session-JCT plot requires one integer arrival "
                f"seed per row: rate={rate:g}, policy={policy!r}, "
                f"seed={raw_seed!r}")
        provenance = (rate, seed)
        pair_key = str(
            row.get("pair_key") or f"poisson:{rate}:seed={seed}")
        prior_provenance = pair_provenance.setdefault(
            pair_key, provenance)
        if prior_provenance != provenance:
            raise ExperimentError(
                "Poisson session-JCT plot pair_key maps to conflicting "
                f"provenance: pair_key={pair_key!r}, "
                f"{prior_provenance} versus {provenance}")
        prior_pair_key = pair_key_by_provenance.setdefault(
            provenance, pair_key)
        if prior_pair_key != pair_key:
            raise ExperimentError(
                "Poisson session-JCT plot provenance maps to multiple pair "
                f"keys: provenance={provenance}, "
                f"pair_keys={prior_pair_key!r}, {pair_key!r}")

        raw_value = row.get("session_jct_mean_ns")
        try:
            value_ns = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ExperimentError(
                "Poisson session-JCT plot requires numeric "
                f"session_jct_mean_ns: rate={rate:g}, policy={policy!r}, "
                f"seed={seed}, value={raw_value!r}") from exc
        if not math.isfinite(value_ns) or value_ns <= 0:
            raise ExperimentError(
                "Poisson session-JCT plot requires positive finite "
                f"session_jct_mean_ns: rate={rate:g}, policy={policy!r}, "
                f"seed={seed}, value={raw_value!r}")
        key = (rate, policy)
        policy_samples = samples.setdefault(key, {})
        if seed in policy_samples:
            raise ExperimentError(
                "Poisson session-JCT plot has a duplicate seed row: "
                f"rate={rate:g}, policy={policy!r}, seed={seed}")
        policy_samples[seed] = value_ns / 1e9

    expected_seed_tuple = None
    cells = []
    for rate in rates:
        rate_seed_tuple = None
        for policy_order, policy in enumerate(policies):
            seed_values = samples.get((rate, policy))
            if not seed_values:
                raise ExperimentError(
                    "Poisson session-JCT plot has an incomplete policy "
                    f"grid: rate={rate:g}, policy={policy!r}")
            seed_tuple = tuple(sorted(seed_values))
            if rate_seed_tuple is None:
                rate_seed_tuple = seed_tuple
            elif seed_tuple != rate_seed_tuple:
                raise ExperimentError(
                    "Poisson session-JCT plot has unpaired seeds within "
                    f"rate={rate:g}: expected={rate_seed_tuple}, "
                    f"policy={policy!r}, observed={seed_tuple}")
            if expected_seed_tuple is None:
                expected_seed_tuple = seed_tuple
            elif seed_tuple != expected_seed_tuple:
                raise ExperimentError(
                    "Poisson session-JCT plot requires the same fixed seed "
                    f"set at every rate: expected={expected_seed_tuple}, "
                    f"rate={rate:g}, observed={seed_tuple}")

            values = [seed_values[seed] for seed in seed_tuple]
            sample_count = len(values)
            mean_seconds = sum(values) / sample_count
            if sample_count > 1:
                sample_variance = sum(
                    (value - mean_seconds) ** 2 for value in values
                ) / (sample_count - 1)
                sample_stddev = math.sqrt(sample_variance)
                half_width = (
                    _student_t_975(sample_count - 1)
                    * sample_stddev / math.sqrt(sample_count)
                )
                ci_lower = max(0.0, mean_seconds - half_width)
                ci_upper = mean_seconds + half_width
            else:
                sample_stddev = None
                half_width = None
                ci_lower = None
                ci_upper = None
            cells.append({
                "offered_rate_sessions_per_second": rate,
                "policy_order": policy_order,
                "policy": policy,
                "is_reference": policy == oracle_label,
                "seed_count": sample_count,
                "arrival_seeds": list(seed_tuple),
                "seed_level_mean_session_jct_seconds": values,
                "mean_session_jct_seconds": mean_seconds,
                "sample_stddev_seconds": sample_stddev,
                "ci95_half_width_seconds": half_width,
                "ci95_lower_seconds": ci_lower,
                "ci95_upper_seconds": ci_upper,
            })
    return rates, policies, cells


def _write_poisson_session_jct_svg(
        rates, policies, cells, output_dir):
    """Write the dependency-free form of the Poisson session-JCT plot."""
    palette = (
        "#4C78A8", "#F58518", "#54A24B", "#E45756",
        "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D",
    )
    values = {
        (cell["offered_rate_sessions_per_second"], cell["policy"]): cell
        for cell in cells
    }
    legend_item_widths = [
        max(120, len(policy) * 7 + 32) for policy in policies
    ]
    width = max(
        900,
        190 * len(rates) + 280,
        sum(legend_item_widths) + 120,
    )
    height = 580
    left, right, top, bottom = 100, 35, 70, 145
    plot_width = width - left - right
    plot_height = height - top - bottom
    upper_values = [
        cell["ci95_upper_seconds"]
        if cell["ci95_upper_seconds"] is not None
        else cell["mean_session_jct_seconds"]
        for cell in cells
    ]
    maximum = max(upper_values)
    y_max = maximum * 1.12 if maximum > 0 else 1.0
    group_width = plot_width / len(rates)
    occupied_width = group_width * 0.82
    bar_width = occupied_width / len(policies)

    def y_position(value):
        return top + plot_height * (1 - float(value) / y_max)

    elements = [
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
         f'height="{height}" viewBox="0 0 {width} {height}">'),
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}'
        '.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}'
        '.error{stroke:#222;stroke-width:1.25}.tick{font-size:12px}'
        '.label{font-size:14px}.title{font-size:18px;font-weight:600}'
        '.subtitle{font-size:11px}.legend{font-size:12px}</style>',
        (f'<text x="{width / 2:.1f}" y="27" text-anchor="middle" '
         'class="title">Poisson session JCT from offer to final '
         'completion</text>'),
        (f'<text x="{width / 2:.1f}" y="46" text-anchor="middle" '
         'class="subtitle">Bars are means of seed-level mean JCT; '
         'error bars are 95% Student-t confidence intervals</text>'),
    ]
    for tick in range(6):
        fraction = tick / 5
        value = y_max * fraction
        y = y_position(value)
        elements.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            f'y2="{y:.2f}" class="grid"/>')
        elements.append(
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
            f'class="tick">{value:.3g}</text>')
    elements.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top+plot_height}" class="axis"/>',
        f'<line x1="{left}" y1="{top+plot_height}" '
        f'x2="{width-right}" y2="{top+plot_height}" class="axis"/>',
    ])
    for rate_index, rate in enumerate(rates):
        group_left = left + rate_index * group_width
        for policy_index, policy in enumerate(policies):
            cell = values[(rate, policy)]
            mean_value = cell["mean_session_jct_seconds"]
            bar_height = plot_height * mean_value / y_max
            x = (
                group_left + (group_width - occupied_width) / 2
                + policy_index * bar_width
            )
            y = top + plot_height - bar_height
            color = palette[policy_index % len(palette)]
            elements.append(
                f'<rect x="{x+1:.2f}" y="{y:.2f}" '
                f'width="{max(1.0, bar_width-2):.2f}" '
                f'height="{bar_height:.2f}" fill="{color}">'
                f'<title>{html.escape(policy)}: {mean_value:.6g} s; '
                f'n={cell["seed_count"]}</title></rect>')
            half_width = cell["ci95_half_width_seconds"]
            if half_width is not None:
                center_x = x + bar_width / 2
                upper_y = y_position(mean_value + half_width)
                lower_y = y_position(max(0.0, mean_value - half_width))
                cap_width = min(8.0, max(3.0, bar_width * 0.28))
                elements.extend([
                    f'<line x1="{center_x:.2f}" y1="{upper_y:.2f}" '
                    f'x2="{center_x:.2f}" y2="{lower_y:.2f}" '
                    'class="error"/>',
                    f'<line x1="{center_x-cap_width:.2f}" '
                    f'y1="{upper_y:.2f}" x2="{center_x+cap_width:.2f}" '
                    f'y2="{upper_y:.2f}" class="error"/>',
                    f'<line x1="{center_x-cap_width:.2f}" '
                    f'y1="{lower_y:.2f}" x2="{center_x+cap_width:.2f}" '
                    f'y2="{lower_y:.2f}" class="error"/>',
                ])
        center = group_left + group_width / 2
        elements.append(
            f'<text x="{center:.2f}" y="{top+plot_height+22}" '
            f'text-anchor="middle" class="tick">{rate:g}</text>')
    elements.append(
        f'<text x="{left+plot_width/2:.2f}" y="{height-76}" '
        'text-anchor="middle" class="label">Offered sessions/s</text>')
    elements.append(
        f'<text x="22" y="{top+plot_height/2:.2f}" '
        f'text-anchor="middle" class="label" '
        f'transform="rotate(-90 22 {top+plot_height/2:.2f})">'
        'Mean session JCT (s; lower is better)</text>')
    legend_x = left
    legend_y = height - 27
    for policy_index, policy in enumerate(policies):
        color = palette[policy_index % len(palette)]
        elements.append(
            f'<rect x="{legend_x}" y="{legend_y-11}" width="12" '
            f'height="12" fill="{color}"/>')
        elements.append(
            f'<text x="{legend_x+18}" y="{legend_y}" '
            f'class="legend">{html.escape(policy)}</text>')
        legend_x += legend_item_widths[policy_index]
    elements.append('</svg>')

    path = Path(output_dir) / "poisson_session_jct_grouped.svg"
    with open(path, "w", encoding="utf-8") as output:
        output.write("\n".join(elements))
        output.write("\n")
    return path


def plot_poisson_session_jct(
        rows, output_dir, *, oracle_label="infinite_hbm_oracle"):
    """Write grouped Poisson offered-to-completion JCT paper artifacts.

    Every input row contributes one seed-level mean. Confidence intervals
    therefore describe variation across independent arrival seeds rather than
    incorrectly treating dependent sessions within one simulation as
    independent samples.
    """
    rates, policies, cells = _poisson_session_jct_plot_cells(
        rows, oracle_label=oracle_label)
    if not cells:
        return [], None
    output_dir = Path(output_dir)
    source_path = output_dir / "poisson_session_jct_plot_source.csv"
    source_rows = []
    for cell in cells:
        source_rows.append({
            "offered_rate_sessions_per_second": cell[
                "offered_rate_sessions_per_second"],
            "policy_order": cell["policy_order"],
            "policy": cell["policy"],
            "is_reference": cell["is_reference"],
            "seed_count": cell["seed_count"],
            "arrival_seeds_json": json.dumps(cell["arrival_seeds"]),
            "seed_level_mean_session_jct_seconds_json": json.dumps(
                cell["seed_level_mean_session_jct_seconds"]),
            "mean_session_jct_seconds": cell[
                "mean_session_jct_seconds"],
            "sample_stddev_seconds": cell["sample_stddev_seconds"],
            "ci95_half_width_seconds": cell[
                "ci95_half_width_seconds"],
            "ci95_lower_seconds": cell["ci95_lower_seconds"],
            "ci95_upper_seconds": cell["ci95_upper_seconds"],
            "source_metric": "session_jct_mean_ns",
            "aggregation_unit": "arrival_seed_level_mean",
            "confidence_interval": "two-sided 95% Student-t",
            "lower_is_better": True,
        })
    with open(source_path, "w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    svg_path = _write_poisson_session_jct_svg(
        rates, policies, cells, output_dir)
    return [svg_path], source_path


_POISSON_PAIRED_PLOT_PROVENANCE_FIELDS = (
    "offered_arrival_trace_sha256",
    "poisson_unit_draw_trace_sha256",
    "workload_sha256",
    "selected_session_ids_hash",
    "selected_session_identity_hash",
    "input_session_ids_hash",
    "measured_session_ids_hash",
)


def _paired_poisson_plot_grid(rows, *, reference_label):
    """Return a complete fixed-seed Poisson grid for paired paper plots."""
    poisson_rows = [row for row in rows if row.get("mode") == "poisson"]
    if not poisson_rows:
        return [], [], (), {}
    policies = _ordered_plot_policies(poisson_rows)
    reference_label = str(reference_label)
    if reference_label not in policies:
        raise ExperimentError(
            "Paired Poisson plot is missing residency reference policy "
            f"{reference_label!r}")

    rates = sorted({float(row["load_value"]) for row in poisson_rows})
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ExperimentError(
            "Paired Poisson plot requires positive finite offered rates")

    grid = {}
    pair_key_provenance = {}
    provenance_pair_key = {}
    rows_by_pair = {}
    for row in poisson_rows:
        rate = float(row["load_value"])
        policy = str(row["policy"])
        raw_seed = row.get("arrival_seed")
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError):
            seed = None
        if (raw_seed is None or isinstance(raw_seed, bool)
                or seed is None or seed != raw_seed):
            raise ExperimentError(
                "Paired Poisson plot requires one integer arrival seed per "
                f"row: rate={rate:g}, policy={policy!r}, "
                f"seed={raw_seed!r}")
        raw_pair_key = row.get("pair_key")
        if not isinstance(raw_pair_key, str) or not raw_pair_key.strip():
            raise ExperimentError(
                "Paired Poisson plot requires a non-empty pair_key for "
                f"provenance: rate={rate:g}, policy={policy!r}, seed={seed}")
        pair_key = raw_pair_key.strip()
        provenance = (rate, seed)
        previous = pair_key_provenance.setdefault(pair_key, provenance)
        if previous != provenance:
            raise ExperimentError(
                "Paired Poisson plot pair_key maps to conflicting "
                f"provenance: pair_key={pair_key!r}, {previous} versus "
                f"{provenance}")
        previous_pair_key = provenance_pair_key.setdefault(
            provenance, pair_key)
        if previous_pair_key != pair_key:
            raise ExperimentError(
                "Paired Poisson plot provenance maps to multiple pair keys: "
                f"provenance={provenance}, pair_keys={previous_pair_key!r}, "
                f"{pair_key!r}")

        key = (rate, seed, policy)
        if key in grid:
            raise ExperimentError(
                "Paired Poisson plot has a duplicate row: "
                f"rate={rate:g}, seed={seed}, policy={policy!r}")
        grid[key] = row
        rows_by_pair.setdefault(provenance, []).append(row)

    expected_seeds = None
    for rate in rates:
        seeds = tuple(sorted({
            seed for row_rate, seed, _ in grid if row_rate == rate
        }))
        if not seeds:
            raise ExperimentError(
                f"Paired Poisson plot has no seeds at rate={rate:g}")
        if expected_seeds is None:
            expected_seeds = seeds
        elif seeds != expected_seeds:
            raise ExperimentError(
                "Paired Poisson plot requires the same fixed seed set at "
                f"every rate: expected={expected_seeds}, rate={rate:g}, "
                f"observed={seeds}")
        for seed in seeds:
            missing = [
                policy for policy in policies
                if (rate, seed, policy) not in grid
            ]
            if missing:
                raise ExperimentError(
                    "Paired Poisson plot has an incomplete/unpaired grid: "
                    f"rate={rate:g}, seed={seed}, missing={missing}")
            pair_rows = rows_by_pair[(rate, seed)]
            for field in _POISSON_PAIRED_PLOT_PROVENANCE_FIELDS:
                present = [row.get(field) is not None for row in pair_rows]
                if any(present) and not all(present):
                    raise ExperimentError(
                        "Paired Poisson plot has incomplete provenance field "
                        f"{field!r}: rate={rate:g}, seed={seed}")
                if all(present):
                    values = {str(row[field]) for row in pair_rows}
                    if len(values) != 1:
                        raise ExperimentError(
                            "Paired Poisson plot has conflicting provenance "
                            f"field {field!r}: rate={rate:g}, seed={seed}, "
                            f"values={sorted(values)}")
    return rates, policies, expected_seeds or (), grid


def _positive_finite_metric(row, field, *, context):
    raw_value = row.get(field)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ExperimentError(
            f"{context} requires numeric {field}: value={raw_value!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ExperimentError(
            f"{context} requires positive finite {field}: value={raw_value!r}")
    return value


def _nonnegative_finite_metric(row, field, *, context):
    raw_value = row.get(field)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ExperimentError(
            f"{context} requires numeric {field}: value={raw_value!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ExperimentError(
            f"{context} requires non-negative finite {field}: "
            f"value={raw_value!r}")
    return value


def _nonnegative_integer_metric(row, field, *, context):
    raw_value = row.get(field)
    if (not isinstance(raw_value, int) or isinstance(raw_value, bool)
            or raw_value < 0):
        raise ExperimentError(
            f"{context} requires non-negative integer {field}: "
            f"value={raw_value!r}")
    return raw_value


def _seed_summary(values, *, clamp_lower_at_zero=False):
    values = [float(value) for value in values]
    count = len(values)
    if count <= 0:
        raise ValueError("Seed summary requires at least one value")
    mean = sum(values) / count
    if count == 1:
        return {
            "mean": mean,
            "sample_stddev": None,
            "ci95_half_width": None,
            "ci95_lower": None,
            "ci95_upper": None,
        }
    sample_variance = sum(
        (value - mean) ** 2 for value in values
    ) / (count - 1)
    sample_stddev = math.sqrt(sample_variance)
    half_width = (
        _student_t_975(count - 1) * sample_stddev / math.sqrt(count)
    )
    lower = mean - half_width
    if clamp_lower_at_zero:
        lower = max(0.0, lower)
    return {
        "mean": mean,
        "sample_stddev": sample_stddev,
        "ci95_half_width": half_width,
        "ci95_lower": lower,
        "ci95_upper": mean + half_width,
    }


def _write_poisson_reference_normalized_jct_svg(
        rates, policies, cells, output_dir):
    palette = (
        "#4C78A8", "#F58518", "#54A24B", "#E45756",
        "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D",
    )
    by_key = {
        (cell["offered_rate_sessions_per_second"], cell["policy"]): cell
        for cell in cells
    }
    legend_widths = [max(120, len(policy) * 7 + 32) for policy in policies]
    width = max(900, 190 * len(rates) + 280, sum(legend_widths) + 120)
    height = 580
    left, right, top, bottom = 100, 35, 74, 145
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(
        cell["ci95_upper_reference_jct_over_system_jct"]
        if cell["ci95_upper_reference_jct_over_system_jct"] is not None
        else cell["mean_reference_jct_over_system_jct"]
        for cell in cells
    )
    y_max = max(1.1, maximum * 1.12)
    group_width = plot_width / len(rates)
    occupied_width = group_width * 0.82
    bar_width = occupied_width / len(policies)

    def y_position(value):
        return top + plot_height * (1 - float(value) / y_max)

    elements = [
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
         f'height="{height}" viewBox="0 0 {width} {height}">'),
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}'
        '.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}'
        '.reference{stroke:#555;stroke-width:1.5;stroke-dasharray:5 4}'
        '.error{stroke:#222;stroke-width:1.25}.tick{font-size:12px}'
        '.label{font-size:14px}.title{font-size:18px;font-weight:600}'
        '.subtitle{font-size:11px}.legend{font-size:12px}</style>',
        (f'<text x="{width / 2:.1f}" y="27" text-anchor="middle" '
         'class="title">Poisson session JCT performance relative to '
         'infinite-HBM residency reference</text>'),
        (f'<text x="{width / 2:.1f}" y="47" text-anchor="middle" '
         'class="subtitle">Paired reference JCT / system JCT; mean and '
         '95% Student-t CI across fixed arrival seeds</text>'),
    ]
    for tick in range(6):
        fraction = tick / 5
        value = y_max * fraction
        y = y_position(value)
        elements.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            f'y2="{y:.2f}" class="grid"/>',
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
            f'class="tick">{value:.2f}</text>',
        ])
    reference_y = y_position(1.0)
    elements.append(
        f'<line x1="{left}" y1="{reference_y:.2f}" '
        f'x2="{width-right}" y2="{reference_y:.2f}" '
        'class="reference"/>')
    elements.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top+plot_height}" class="axis"/>',
        f'<line x1="{left}" y1="{top+plot_height}" '
        f'x2="{width-right}" y2="{top+plot_height}" class="axis"/>',
    ])
    for rate_index, rate in enumerate(rates):
        group_left = left + rate_index * group_width
        for policy_index, policy in enumerate(policies):
            cell = by_key[(rate, policy)]
            mean = cell["mean_reference_jct_over_system_jct"]
            bar_height = plot_height * mean / y_max
            x = (group_left + (group_width - occupied_width) / 2
                 + policy_index * bar_width)
            y = top + plot_height - bar_height
            color = palette[policy_index % len(palette)]
            elements.append(
                f'<rect x="{x+1:.2f}" y="{y:.2f}" '
                f'width="{max(1.0, bar_width-2):.2f}" '
                f'height="{bar_height:.2f}" fill="{color}">'
                f'<title>{html.escape(policy)}: {mean:.6g}; '
                f'n={cell["seed_count"]}</title></rect>')
            half_width = cell["ci95_half_width_reference_jct_over_system_jct"]
            if half_width is not None:
                center_x = x + bar_width / 2
                upper_y = y_position(mean + half_width)
                lower_y = y_position(max(0.0, mean - half_width))
                cap = min(8.0, max(3.0, bar_width * 0.28))
                elements.extend([
                    f'<line x1="{center_x:.2f}" y1="{upper_y:.2f}" '
                    f'x2="{center_x:.2f}" y2="{lower_y:.2f}" '
                    'class="error"/>',
                    f'<line x1="{center_x-cap:.2f}" y1="{upper_y:.2f}" '
                    f'x2="{center_x+cap:.2f}" y2="{upper_y:.2f}" '
                    'class="error"/>',
                    f'<line x1="{center_x-cap:.2f}" y1="{lower_y:.2f}" '
                    f'x2="{center_x+cap:.2f}" y2="{lower_y:.2f}" '
                    'class="error"/>',
                ])
        center = group_left + group_width / 2
        elements.append(
            f'<text x="{center:.2f}" y="{top+plot_height+22}" '
            f'text-anchor="middle" class="tick">{rate:g}</text>')
    elements.extend([
        f'<text x="{left+plot_width/2:.2f}" y="{height-76}" '
        'text-anchor="middle" class="label">Offered sessions/s</text>',
        f'<text x="22" y="{top+plot_height/2:.2f}" '
        f'text-anchor="middle" class="label" '
        f'transform="rotate(-90 22 {top+plot_height/2:.2f})">'
        'Reference JCT / system JCT (higher is better)</text>',
    ])
    legend_x = left
    legend_y = height - 27
    for policy_index, policy in enumerate(policies):
        color = palette[policy_index % len(palette)]
        elements.extend([
            f'<rect x="{legend_x}" y="{legend_y-11}" width="12" '
            f'height="12" fill="{color}"/>',
            f'<text x="{legend_x+18}" y="{legend_y}" '
            f'class="legend">{html.escape(policy)}</text>',
        ])
        legend_x += legend_widths[policy_index]
    elements.append('</svg>')
    path = Path(output_dir) / "poisson_session_jct_reference_normalized.svg"
    with open(path, "w", encoding="utf-8") as output:
        output.write("\n".join(elements))
        output.write("\n")
    return path


def plot_poisson_reference_normalized_jct(
        rows, output_dir, *, reference_label="infinite_hbm_oracle"):
    """Plot paired reference-JCT/system-JCT ratios across fixed seeds."""
    rates, policies, seeds, grid = _paired_poisson_plot_grid(
        rows, reference_label=reference_label)
    if not grid:
        return [], None
    cells = []
    for rate in rates:
        for policy_order, policy in enumerate(policies):
            ratios = []
            system_jct_seconds = []
            reference_jct_seconds = []
            pair_keys = []
            for seed in seeds:
                context = (
                    "Paired Poisson reference-normalized JCT plot at "
                    f"rate={rate:g}, seed={seed}, policy={policy!r}")
                row = grid[(rate, seed, policy)]
                reference_row = grid[(rate, seed, str(reference_label))]
                system_ns = _positive_finite_metric(
                    row, "session_jct_mean_ns", context=context)
                reference_ns = _positive_finite_metric(
                    reference_row, "session_jct_mean_ns", context=context)
                ratio = 1.0 if policy == str(reference_label) else (
                    reference_ns / system_ns)
                ratios.append(ratio)
                system_jct_seconds.append(system_ns / 1e9)
                reference_jct_seconds.append(reference_ns / 1e9)
                pair_keys.append(str(row["pair_key"]))
            summary = _seed_summary(ratios, clamp_lower_at_zero=True)
            if policy == str(reference_label):
                if any(value != 1.0 for value in ratios):
                    raise ExperimentError(
                        "Residency-reference normalized JCT must equal "
                        "exactly one")
                summary.update({
                    "mean": 1.0,
                    "sample_stddev": 0.0 if len(ratios) > 1 else None,
                    "ci95_half_width": 0.0 if len(ratios) > 1 else None,
                    "ci95_lower": 1.0 if len(ratios) > 1 else None,
                    "ci95_upper": 1.0 if len(ratios) > 1 else None,
                })
            cells.append({
                "offered_rate_sessions_per_second": rate,
                "policy_order": policy_order,
                "policy": policy,
                "is_reference": policy == str(reference_label),
                "seed_count": len(seeds),
                "arrival_seeds": list(seeds),
                "pair_keys": pair_keys,
                "seed_level_system_session_jct_seconds": system_jct_seconds,
                "seed_level_reference_session_jct_seconds": (
                    reference_jct_seconds),
                "seed_level_reference_jct_over_system_jct": ratios,
                "mean_reference_jct_over_system_jct": summary["mean"],
                "sample_stddev_reference_jct_over_system_jct": summary[
                    "sample_stddev"],
                "ci95_half_width_reference_jct_over_system_jct": summary[
                    "ci95_half_width"],
                "ci95_lower_reference_jct_over_system_jct": summary[
                    "ci95_lower"],
                "ci95_upper_reference_jct_over_system_jct": summary[
                    "ci95_upper"],
            })

    output_dir = Path(output_dir)
    source_path = (
        output_dir / "poisson_session_jct_reference_normalized_source.csv"
    )
    source_rows = []
    for cell in cells:
        source_rows.append({
            "offered_rate_sessions_per_second": cell[
                "offered_rate_sessions_per_second"],
            "policy_order": cell["policy_order"],
            "policy": cell["policy"],
            "is_reference": cell["is_reference"],
            "seed_count": cell["seed_count"],
            "arrival_seeds_json": json.dumps(cell["arrival_seeds"]),
            "pair_keys_json": json.dumps(cell["pair_keys"]),
            "seed_level_system_session_jct_seconds_json": json.dumps(
                cell["seed_level_system_session_jct_seconds"]),
            "seed_level_reference_session_jct_seconds_json": json.dumps(
                cell["seed_level_reference_session_jct_seconds"]),
            "seed_level_reference_jct_over_system_jct_json": json.dumps(
                cell["seed_level_reference_jct_over_system_jct"]),
            "mean_reference_jct_over_system_jct": cell[
                "mean_reference_jct_over_system_jct"],
            "sample_stddev_reference_jct_over_system_jct": cell[
                "sample_stddev_reference_jct_over_system_jct"],
            "ci95_half_width_reference_jct_over_system_jct": cell[
                "ci95_half_width_reference_jct_over_system_jct"],
            "ci95_lower_reference_jct_over_system_jct": cell[
                "ci95_lower_reference_jct_over_system_jct"],
            "ci95_upper_reference_jct_over_system_jct": cell[
                "ci95_upper_reference_jct_over_system_jct"],
            "formula": (
                "paired_reference_session_jct_mean_ns / "
                "system_session_jct_mean_ns"),
            "aggregation_unit": "arrival_seed_level_paired_ratio",
            "confidence_interval": "two-sided 95% Student-t",
            "higher_is_better": True,
            "reference_semantics": (
                "infinite-HBM residency reference; not a strict JCT oracle"),
        })
    with open(source_path, "w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    svg_path = _write_poisson_reference_normalized_jct_svg(
        rates, policies, cells, output_dir)
    return [svg_path], source_path


def _write_poisson_session_jct_decomposition_svg(
        rates, policies, cells, output_dir):
    palette = (
        "#4C78A8", "#F58518", "#54A24B", "#E45756",
        "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D",
    )
    by_key = {
        (cell["offered_rate_sessions_per_second"], cell["policy"]): cell
        for cell in cells
    }
    legend_widths = [max(120, len(policy) * 7 + 32) for policy in policies]
    width = max(940, 190 * len(rates) + 300, sum(legend_widths) + 120)
    height = 610
    left, right, top, bottom = 105, 35, 78, 170
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(
        cell["ci95_upper_total_session_jct_seconds"]
        if cell["ci95_upper_total_session_jct_seconds"] is not None
        else cell["mean_total_session_jct_seconds"]
        for cell in cells
    )
    y_max = maximum * 1.12 if maximum > 0 else 1.0
    group_width = plot_width / len(rates)
    occupied_width = group_width * 0.82
    bar_width = occupied_width / len(policies)

    def y_position(value):
        return top + plot_height * (1 - float(value) / y_max)

    elements = [
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
         f'height="{height}" viewBox="0 0 {width} {height}">'),
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}'
        '.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}'
        '.segment{stroke:#fff;stroke-width:.7}.error{stroke:#222;stroke-width:1.25}'
        '.tick{font-size:12px}.label{font-size:14px}'
        '.title{font-size:18px;font-weight:600}.subtitle{font-size:11px}'
        '.legend{font-size:12px}</style>',
        (f'<text x="{width / 2:.1f}" y="27" text-anchor="middle" '
         'class="title">Poisson offered-to-final session JCT '
         'decomposition</text>'),
        (f'<text x="{width / 2:.1f}" y="47" text-anchor="middle" '
         'class="subtitle">Stacked admission queue + post-admission '
         'execution; error bars are total-JCT 95% Student-t CI across '
         'arrival seeds</text>'),
    ]
    for tick in range(6):
        fraction = tick / 5
        value = y_max * fraction
        y = y_position(value)
        elements.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            f'y2="{y:.2f}" class="grid"/>',
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
            f'class="tick">{value:.3g}</text>',
        ])
    elements.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top+plot_height}" class="axis"/>',
        f'<line x1="{left}" y1="{top+plot_height}" '
        f'x2="{width-right}" y2="{top+plot_height}" class="axis"/>',
    ])
    for rate_index, rate in enumerate(rates):
        group_left = left + rate_index * group_width
        for policy_index, policy in enumerate(policies):
            cell = by_key[(rate, policy)]
            admission = cell["mean_admission_queue_seconds"]
            execution = cell["mean_session_execution_seconds"]
            total = cell["mean_total_session_jct_seconds"]
            admission_height = plot_height * admission / y_max
            execution_height = plot_height * execution / y_max
            x = (group_left + (group_width - occupied_width) / 2
                 + policy_index * bar_width)
            baseline_y = top + plot_height
            color = palette[policy_index % len(palette)]
            if admission_height > 0:
                elements.append(
                    f'<rect x="{x+1:.2f}" '
                    f'y="{baseline_y-admission_height:.2f}" '
                    f'width="{max(1.0, bar_width-2):.2f}" '
                    f'height="{admission_height:.2f}" fill="{color}" '
                    'class="segment admission-segment">'
                    f'<title>{html.escape(policy)} admission queue: '
                    f'{admission:.6g} s</title></rect>')
            execution_y = baseline_y - admission_height - execution_height
            elements.append(
                f'<rect x="{x+1:.2f}" y="{execution_y:.2f}" '
                f'width="{max(1.0, bar_width-2):.2f}" '
                f'height="{execution_height:.2f}" fill="{color}" '
                'fill-opacity="0.35" class="segment execution-segment">'
                f'<title>{html.escape(policy)} post-admission execution: '
                f'{execution:.6g} s; total: {total:.6g} s</title></rect>')
            half_width = cell["ci95_half_width_total_session_jct_seconds"]
            if half_width is not None:
                center_x = x + bar_width / 2
                upper_y = y_position(total + half_width)
                lower_y = y_position(max(0.0, total - half_width))
                cap = min(8.0, max(3.0, bar_width * 0.28))
                elements.extend([
                    f'<line x1="{center_x:.2f}" y1="{upper_y:.2f}" '
                    f'x2="{center_x:.2f}" y2="{lower_y:.2f}" '
                    'class="error"/>',
                    f'<line x1="{center_x-cap:.2f}" y1="{upper_y:.2f}" '
                    f'x2="{center_x+cap:.2f}" y2="{upper_y:.2f}" '
                    'class="error"/>',
                    f'<line x1="{center_x-cap:.2f}" y1="{lower_y:.2f}" '
                    f'x2="{center_x+cap:.2f}" y2="{lower_y:.2f}" '
                    'class="error"/>',
                ])
        center = group_left + group_width / 2
        elements.append(
            f'<text x="{center:.2f}" y="{top+plot_height+22}" '
            f'text-anchor="middle" class="tick">{rate:g}</text>')
    elements.extend([
        f'<text x="{left+plot_width/2:.2f}" y="{height-100}" '
        'text-anchor="middle" class="label">Offered sessions/s</text>',
        f'<text x="22" y="{top+plot_height/2:.2f}" '
        f'text-anchor="middle" class="label" '
        f'transform="rotate(-90 22 {top+plot_height/2:.2f})">'
        'Mean session JCT (s; lower is better)</text>',
    ])
    legend_x = left
    legend_y = height - 52
    for policy_index, policy in enumerate(policies):
        color = palette[policy_index % len(palette)]
        elements.extend([
            f'<rect x="{legend_x}" y="{legend_y-11}" width="12" '
            f'height="12" fill="{color}"/>',
            f'<text x="{legend_x+18}" y="{legend_y}" '
            f'class="legend">{html.escape(policy)}</text>',
        ])
        legend_x += legend_widths[policy_index]
    component_y = height - 24
    elements.extend([
        f'<rect x="{left}" y="{component_y-11}" width="12" '
        'height="12" fill="#555"/>',
        f'<text x="{left+18}" y="{component_y}" class="legend">'
        'Admission queue</text>',
        f'<rect x="{left+150}" y="{component_y-11}" width="12" '
        'height="12" fill="#555" fill-opacity="0.35"/>',
        f'<text x="{left+168}" y="{component_y}" class="legend">'
        'Post-admission execution</text>',
        '</svg>',
    ])
    path = Path(output_dir) / "poisson_session_jct_decomposition_stacked.svg"
    with open(path, "w", encoding="utf-8") as output:
        output.write("\n".join(elements))
        output.write("\n")
    return path


def plot_poisson_session_jct_decomposition(
        rows, output_dir, *, reference_label="infinite_hbm_oracle"):
    """Plot exact admission-queue and post-admission JCT components."""
    rates, policies, seeds, grid = _paired_poisson_plot_grid(
        rows, reference_label=reference_label)
    if not grid:
        return [], None
    cells = []
    for rate in rates:
        for policy_order, policy in enumerate(policies):
            totals = []
            admissions = []
            executions = []
            total_sums_ns = []
            admission_sums_ns = []
            execution_sums_ns = []
            session_counts = []
            pair_keys = []
            for seed in seeds:
                row = grid[(rate, seed, policy)]
                context = (
                    "Poisson session-JCT decomposition at "
                    f"rate={rate:g}, seed={seed}, policy={policy!r}")
                total_sum = _nonnegative_integer_metric(
                    row, "session_jct_sum_ns", context=context)
                admission_sum = _nonnegative_integer_metric(
                    row, "session_admission_queue_sum_ns", context=context)
                execution_sum = _nonnegative_integer_metric(
                    row, "session_execution_sum_ns", context=context)
                total_count = _nonnegative_integer_metric(
                    row, "session_jct_count", context=context)
                admission_count = _nonnegative_integer_metric(
                    row, "session_admission_queue_count", context=context)
                execution_count = _nonnegative_integer_metric(
                    row, "session_execution_count", context=context)
                if total_count <= 0 or total_sum <= 0:
                    raise ExperimentError(
                        f"{context} requires positive session JCT sum/count")
                if not total_count == admission_count == execution_count:
                    raise ExperimentError(
                        f"{context} has mismatched decomposition counts: "
                        f"total={total_count}, admission={admission_count}, "
                        f"execution={execution_count}")
                if total_sum != admission_sum + execution_sum:
                    raise ExperimentError(
                        f"{context} is nonadditive: session_jct_sum_ns="
                        f"{total_sum}, session_admission_queue_sum_ns="
                        f"{admission_sum}, "
                        f"session_execution_sum_ns={execution_sum}")
                total = total_sum / total_count
                admission = admission_sum / admission_count
                execution = execution_sum / execution_count
                reported_means = (
                    ("session_jct_mean_ns", total),
                    ("session_admission_queue_mean_ns", admission),
                    ("session_execution_mean_ns", execution),
                )
                for field, derived_mean in reported_means:
                    reported_mean = _nonnegative_finite_metric(
                        row, field, context=context)
                    if not math.isclose(
                            reported_mean, derived_mean,
                            rel_tol=1e-12, abs_tol=1e-9):
                        raise ExperimentError(
                            f"{context} {field} does not match exact "
                            f"sum/count: reported={reported_mean}, "
                            f"derived={derived_mean}")
                totals.append(total / 1_000_000_000)
                admissions.append(admission / 1_000_000_000)
                executions.append(execution / 1_000_000_000)
                total_sums_ns.append(total_sum)
                admission_sums_ns.append(admission_sum)
                execution_sums_ns.append(execution_sum)
                session_counts.append(total_count)
                pair_keys.append(str(row["pair_key"]))
            total_summary = _seed_summary(
                totals, clamp_lower_at_zero=True)
            mean_admission = sum(admissions) / len(admissions)
            mean_execution = sum(executions) / len(executions)
            mean_total = mean_admission + mean_execution
            cells.append({
                "offered_rate_sessions_per_second": rate,
                "policy_order": policy_order,
                "policy": policy,
                "is_reference": policy == str(reference_label),
                "seed_count": len(seeds),
                "arrival_seeds": list(seeds),
                "pair_keys": pair_keys,
                "seed_level_total_session_jct_seconds": totals,
                "seed_level_admission_queue_seconds": admissions,
                "seed_level_session_execution_seconds": executions,
                "seed_level_session_count": session_counts,
                "seed_level_total_session_jct_sum_ns": total_sums_ns,
                "seed_level_admission_queue_sum_ns": admission_sums_ns,
                "seed_level_session_execution_sum_ns": execution_sums_ns,
                "mean_total_session_jct_seconds": mean_total,
                "mean_admission_queue_seconds": mean_admission,
                "mean_session_execution_seconds": mean_execution,
                "sample_stddev_total_session_jct_seconds": total_summary[
                    "sample_stddev"],
                "ci95_half_width_total_session_jct_seconds": total_summary[
                    "ci95_half_width"],
                "ci95_lower_total_session_jct_seconds": total_summary[
                    "ci95_lower"],
                "ci95_upper_total_session_jct_seconds": total_summary[
                    "ci95_upper"],
            })

    output_dir = Path(output_dir)
    source_path = output_dir / "poisson_session_jct_decomposition_source.csv"
    source_rows = []
    for cell in cells:
        source_rows.append({
            "offered_rate_sessions_per_second": cell[
                "offered_rate_sessions_per_second"],
            "policy_order": cell["policy_order"],
            "policy": cell["policy"],
            "is_reference": cell["is_reference"],
            "seed_count": cell["seed_count"],
            "arrival_seeds_json": json.dumps(cell["arrival_seeds"]),
            "pair_keys_json": json.dumps(cell["pair_keys"]),
            "seed_level_total_session_jct_seconds_json": json.dumps(
                cell["seed_level_total_session_jct_seconds"]),
            "seed_level_admission_queue_seconds_json": json.dumps(
                cell["seed_level_admission_queue_seconds"]),
            "seed_level_session_execution_seconds_json": json.dumps(
                cell["seed_level_session_execution_seconds"]),
            "seed_level_session_count_json": json.dumps(
                cell["seed_level_session_count"]),
            "seed_level_total_session_jct_sum_ns_json": json.dumps(
                cell["seed_level_total_session_jct_sum_ns"]),
            "seed_level_admission_queue_sum_ns_json": json.dumps(
                cell["seed_level_admission_queue_sum_ns"]),
            "seed_level_session_execution_sum_ns_json": json.dumps(
                cell["seed_level_session_execution_sum_ns"]),
            "mean_total_session_jct_seconds": cell[
                "mean_total_session_jct_seconds"],
            "mean_admission_queue_seconds": cell[
                "mean_admission_queue_seconds"],
            "mean_session_execution_seconds": cell[
                "mean_session_execution_seconds"],
            "sample_stddev_total_session_jct_seconds": cell[
                "sample_stddev_total_session_jct_seconds"],
            "ci95_half_width_total_session_jct_seconds": cell[
                "ci95_half_width_total_session_jct_seconds"],
            "ci95_lower_total_session_jct_seconds": cell[
                "ci95_lower_total_session_jct_seconds"],
            "ci95_upper_total_session_jct_seconds": cell[
                "ci95_upper_total_session_jct_seconds"],
            "exact_row_identity": (
                "session_jct_mean_ns = session_admission_queue_mean_ns + "
                "session_execution_mean_ns"),
            "aggregation_unit": "arrival_seed_level_mean",
            "confidence_interval": (
                "two-sided 95% Student-t on total session JCT"),
            "lower_is_better": True,
        })
    with open(source_path, "w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    svg_path = _write_poisson_session_jct_decomposition_svg(
        rates, policies, cells, output_dir)
    return [svg_path], source_path


_POISSON_RATE_METRIC_DEFINITIONS = (
    {
        "name": "resume_ttft",
        "title": "Resume TTFT versus offered Poisson rate",
        "unit": "seconds",
        "statistics": (
            ("mean", "resume_ttft_exact_mean_ns", 1e-9),
            ("p95", "resume_ttft_p95_ns", 1e-9),
        ),
        "denominator_field": "resume_ttft_denominator",
        "slo_key": "resume_ttft_slo",
        "lower_is_better": True,
    },
    {
        "name": "tpot",
        "title": "TPOT versus offered Poisson rate",
        "unit": "seconds/token",
        "statistics": (
            ("mean", "tpot_exact_mean_ns", 1e-9),
            ("p95", "tpot_p95_ns", 1e-9),
        ),
        "denominator_field": "tpot_denominator",
        "slo_key": "tpot_slo",
        "lower_is_better": True,
    },
    {
        "name": "server_added_session_jct",
        "title": "Server-added session JCT versus offered Poisson rate",
        "unit": "seconds",
        "statistics": (
            ("mean", "server_added_session_jct_mean_ns", 1e-9),
            ("p95", "server_added_session_jct_p95_ns", 1e-9),
        ),
        "denominator_field": "server_added_session_jct_denominator",
        "lower_is_better": True,
    },
    {
        "name": "total_hbm_capacity_admission_wait",
        "title": "Total HBM capacity-admission wait versus offered rate",
        "unit": "seconds",
        "statistics": (
            ("mean", "total_hbm_capacity_admission_wait_mean_ns", 1e-9),
            ("p95", "total_hbm_capacity_admission_wait_p95_ns", 1e-9),
        ),
        "denominator_field": "total_hbm_capacity_admission_wait_scope",
        "lower_is_better": True,
    },
    {
        "name": "restore_hbm_capacity_admission_wait",
        "title": "Restore-destination HBM admission wait versus offered rate",
        "unit": "seconds",
        "statistics": (
            ("mean", "restore_hbm_capacity_admission_wait_mean_ns", 1e-9),
            ("p95", "restore_hbm_capacity_admission_wait_p95_ns", 1e-9),
        ),
        "denominator_field": "total_hbm_capacity_admission_wait_scope",
        "lower_is_better": True,
    },
    {
        "name": "pd_chunk_hbm_capacity_admission_wait",
        "title": "P/D chunk HBM admission wait versus offered rate",
        "unit": "seconds",
        "statistics": (
            ("mean", "pd_chunk_hbm_capacity_admission_wait_mean_ns", 1e-9),
            ("p95", "pd_chunk_hbm_capacity_admission_wait_p95_ns", 1e-9),
        ),
        "denominator_field": "total_hbm_capacity_admission_wait_scope",
        "lower_is_better": True,
    },
    {
        "name": "average_active_batch_size",
        "title": "Average active batch size versus offered Poisson rate",
        "unit": "real requests/non-dummy model iteration",
        "statistics": (
            ("mean", "average_active_batch_size", 1.0),
        ),
        "denominator_field": "active_batch_size_scope",
        "lower_is_better": False,
    },
)


def _poisson_rate_metric_cells(
        rows, *, reference_label, slo_settings=None):
    """Build fixed-seed cells for rate-dependent operational metrics."""
    incompatible = [
        str(row.get("run_id") or (
            f"{row.get('policy')}@{row.get('load_value')}/"
            f"seed={row.get('arrival_seed')}"
        ))
        for row in rows
        if row.get("mode") == "poisson"
        and (
            row.get("online_artifact_schema_version") != SCHEMA_VERSION
            or row.get("session_report_schema_version")
            != MIN_CURRENT_SESSION_REPORT_SCHEMA_VERSION
            or row.get("operational_metric_source_status")
            != "schema11_exact_measurement_window"
        )
    ]
    if incompatible:
        raise ExperimentError(
            "Poisson rate-metric plots require online artifact schema "
            f"{SCHEMA_VERSION}, session report schema "
            f"{MIN_CURRENT_SESSION_REPORT_SCHEMA_VERSION}, and exact "
            "measurement-window operational sources; legacy/incompatible "
            f"rows={incompatible[:5]}")
    rates, policies, seeds, grid = _paired_poisson_plot_grid(
        rows, reference_label=reference_label)
    if not grid:
        return rates, policies, []
    slo_settings = dict(slo_settings or {})
    cells = []
    for definition in _POISSON_RATE_METRIC_DEFINITIONS:
        denominators = set()
        for rate in rates:
            for policy in policies:
                for statistic, field, scale in definition["statistics"]:
                    values = []
                    for seed in seeds:
                        row = grid[(rate, seed, policy)]
                        context = (
                            f"Poisson {definition['name']} plot: "
                            f"rate={rate:g}, seed={seed}, policy={policy!r}"
                        )
                        value = _nonnegative_finite_metric(
                            row, field, context=context)
                        values.append(value * scale)
                        denominator = row.get(
                            definition["denominator_field"])
                        if not isinstance(denominator, str) or not denominator:
                            raise ExperimentError(
                                f"{context} requires non-empty denominator "
                                f"field {definition['denominator_field']}")
                        denominators.add(denominator)
                    summary = _seed_summary(
                        values, clamp_lower_at_zero=True)
                    slo_key = definition.get("slo_key")
                    slo_declaration = slo_settings.get(slo_key)
                    slo = (
                        float(slo_declaration["threshold_ms"]) / 1000.0
                        if slo_declaration is not None else None
                    )
                    cells.append({
                        "metric": definition["name"],
                        "title": definition["title"],
                        "unit": definition["unit"],
                        "statistic": statistic,
                        "source_field": field,
                        "offered_rate_sessions_per_second": rate,
                        "policy": policy,
                        "policy_order": policies.index(policy),
                        "is_reference": policy == str(reference_label),
                        "arrival_seeds": list(seeds),
                        "seed_values": values,
                        "seed_count": len(values),
                        "mean_across_seeds": summary["mean"],
                        "sample_stddev": summary["sample_stddev"],
                        "ci95_half_width": summary["ci95_half_width"],
                        "ci95_lower": summary["ci95_lower"],
                        "ci95_upper": summary["ci95_upper"],
                        "denominator": denominator,
                        "slo": slo,
                        "slo_source": (
                            f"spec.plots.poisson_rate_metrics.{slo_key}"
                            if slo is not None else None
                        ),
                        "slo_basis": (
                            slo_declaration["basis"]
                            if slo_declaration is not None else None
                        ),
                        "slo_provenance": (
                            slo_declaration["provenance"]
                            if slo_declaration is not None else None
                        ),
                        "lower_is_better": definition["lower_is_better"],
                    })
        if len(denominators) != 1:
            raise ExperimentError(
                f"Poisson {definition['name']} denominator semantics vary "
                f"across paired runs: {sorted(denominators)}")
    return rates, policies, cells


def _write_poisson_rate_metric_svg(
        definition, rates, policies, cells, output_dir):
    palette = (
        "#4C78A8", "#F58518", "#54A24B", "#E45756",
        "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D",
    )
    width = max(920, 150 * len(rates) + 320)
    height = 610
    left, right, top, bottom = 108, 42, 82, 170
    plot_width = width - left - right
    plot_height = height - top - bottom
    upper_values = [
        cell["ci95_upper"]
        if cell["ci95_upper"] is not None else cell["mean_across_seeds"]
        for cell in cells
    ]
    slo_values = {cell["slo"] for cell in cells if cell["slo"] is not None}
    if len(slo_values) > 1:
        raise ExperimentError(
            f"Poisson {definition['name']} plot has inconsistent SLOs")
    if slo_values:
        upper_values.append(next(iter(slo_values)))
    maximum = max(upper_values, default=0.0)
    y_max = maximum * 1.12 if maximum > 0 else 1.0
    x_step = plot_width / max(1, len(rates) - 1)

    def x_position(rate):
        return left + rates.index(rate) * x_step

    def y_position(value):
        return top + plot_height * (1.0 - float(value) / y_max)

    statistics = tuple(
        statistic for statistic, _, _ in definition["statistics"])
    statistic_note = (
        "solid=mean, dashed=p95"
        if statistics == ("mean", "p95") else
        f"statistic={','.join(statistics)}"
    )
    elements = [
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
         f'height="{height}" viewBox="0 0 {width} {height}">'),
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}'
        '.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}'
        '.mean{fill:none;stroke-width:2.4}.p95{fill:none;stroke-width:2.4;'
        'stroke-dasharray:7 4}.error{stroke-width:1}.slo{stroke:#222;'
        'stroke-width:1.5;stroke-dasharray:3 5}.tick{font-size:12px}'
        '.label{font-size:14px}.title{font-size:18px;font-weight:600}'
        '.subtitle{font-size:11px}.legend{font-size:12px}</style>',
        (f'<text x="{width/2:.1f}" y="28" text-anchor="middle" '
         f'class="title">{html.escape(definition["title"])}</text>'),
        (f'<text x="{width/2:.1f}" y="48" text-anchor="middle" '
         'class="subtitle">Each point aggregates one run statistic per '
         'fixed arrival seed; error bars are 95% Student-t CIs; '
         f'{html.escape(statistic_note)}</text>'),
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = y_position(value)
        elements.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            f'y2="{y:.2f}" class="grid"/>',
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
            f'class="tick">{value:.3g}</text>',
        ])
    elements.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top+plot_height}" class="axis"/>',
        f'<line x1="{left}" y1="{top+plot_height}" '
        f'x2="{width-right}" y2="{top+plot_height}" class="axis"/>',
    ])
    by_series = {}
    for cell in cells:
        by_series.setdefault(
            (cell["policy"], cell["statistic"]), []).append(cell)
    for policy_index, policy in enumerate(policies):
        color = palette[policy_index % len(palette)]
        for statistic in ("mean", "p95"):
            series = sorted(
                by_series.get((policy, statistic), []),
                key=lambda cell: cell["offered_rate_sessions_per_second"],
            )
            if not series:
                continue
            points = " ".join(
                f'{x_position(cell["offered_rate_sessions_per_second"]):.2f},'
                f'{y_position(cell["mean_across_seeds"]):.2f}'
                for cell in series
            )
            elements.append(
                f'<polyline points="{points}" class="{statistic}" '
                f'stroke="{color}"/>')
            for cell in series:
                x = x_position(cell["offered_rate_sessions_per_second"])
                y = y_position(cell["mean_across_seeds"])
                half_width = cell["ci95_half_width"]
                if half_width is not None:
                    upper = y_position(cell["mean_across_seeds"] + half_width)
                    lower = y_position(max(
                        0.0, cell["mean_across_seeds"] - half_width))
                    elements.extend([
                        f'<line x1="{x:.2f}" y1="{upper:.2f}" '
                        f'x2="{x:.2f}" y2="{lower:.2f}" '
                        f'class="error" stroke="{color}"/>',
                        f'<line x1="{x-4:.2f}" y1="{upper:.2f}" '
                        f'x2="{x+4:.2f}" y2="{upper:.2f}" '
                        f'class="error" stroke="{color}"/>',
                        f'<line x1="{x-4:.2f}" y1="{lower:.2f}" '
                        f'x2="{x+4:.2f}" y2="{lower:.2f}" '
                        f'class="error" stroke="{color}"/>',
                    ])
                elements.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" '
                    f'fill="{color}"><title>{html.escape(policy)} '
                    f'{statistic}: {cell["mean_across_seeds"]:.6g} '
                    f'{html.escape(definition["unit"])}; '
                    f'n={cell["seed_count"]} seeds</title></circle>')
    if slo_values:
        slo = next(iter(slo_values))
        y = y_position(slo)
        elements.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            f'y2="{y:.2f}" class="slo"/>',
            f'<text x="{width-right-4}" y="{y-6:.2f}" '
            f'text-anchor="end" class="tick">declared SLO {slo:.4g} '
            f'{html.escape(definition["unit"])}</text>',
        ])
    for rate in rates:
        x = x_position(rate)
        elements.append(
            f'<text x="{x:.2f}" y="{top+plot_height+22}" '
            f'text-anchor="middle" class="tick">{rate:g}</text>')
    elements.extend([
        f'<text x="{left+plot_width/2:.2f}" y="{height-112}" '
        'text-anchor="middle" class="label">Offered sessions/s</text>',
        f'<text x="24" y="{top+plot_height/2:.2f}" '
        f'text-anchor="middle" class="label" transform="rotate(-90 24 '
        f'{top+plot_height/2:.2f})">{html.escape(definition["unit"])}'
        f'{" (lower is better)" if definition["lower_is_better"] else ""}'
        '</text>',
    ])
    legend_x = left
    legend_y = height - 63
    for policy_index, policy in enumerate(policies):
        color = palette[policy_index % len(palette)]
        if legend_x + len(policy) * 7 + 42 > width - right:
            legend_x = left
            legend_y += 22
        elements.extend([
            f'<line x1="{legend_x}" y1="{legend_y-4}" '
            f'x2="{legend_x+20}" y2="{legend_y-4}" '
            f'stroke="{color}" stroke-width="3"/>',
            f'<text x="{legend_x+26}" y="{legend_y}" '
            f'class="legend">{html.escape(policy)}</text>',
        ])
        legend_x += max(145, len(policy) * 7 + 52)
    elements.append('</svg>')
    path = Path(output_dir) / f'poisson_{definition["name"]}_by_rate.svg'
    with open(path, "w", encoding="utf-8") as output:
        output.write("\n".join(elements))
        output.write("\n")
    return path


_HBM_OCCUPANCY_SERIES = (
    (
        "physical_idle_reusable",
        "hbm_kv_average_physical_idle_reusable_fraction",
        "additive_physical_stack",
    ),
    (
        "physical_non_idle_active",
        "hbm_kv_average_physical_non_idle_active_fraction",
        "additive_physical_stack",
    ),
    (
        "physical_free",
        "hbm_kv_average_physical_free_fraction",
        "additive_physical_stack",
    ),
    (
        "logical_destination_admission_reservation",
        "hbm_kv_average_logical_destination_reservation_fraction",
        "non_additive_logical_overlay",
    ),
    (
        "reserved_free_slack",
        "hbm_kv_average_reserved_free_slack_fraction",
        "non_additive_logical_overlay_component",
    ),
    (
        "future_reclaim_backed_reservation",
        "hbm_kv_average_future_reclaim_backed_reservation_fraction",
        "non_additive_logical_overlay_component",
    ),
    (
        "unclaimed_allocatable_slack",
        "hbm_kv_average_unclaimed_allocatable_slack_fraction",
        "reservation_adjusted_physical_slack",
    ),
    (
        "reservation_adjusted_claim",
        "hbm_kv_average_reservation_adjusted_claim_fraction",
        "non_additive_overlay_marker",
    ),
)


def _poisson_hbm_occupancy_cells(rows, *, reference_label):
    rates, policies, seeds, grid = _paired_poisson_plot_grid(
        rows, reference_label=reference_label)
    if not grid:
        return rates, policies, []
    cells = []
    for rate in rates:
        for policy_order, policy in enumerate(policies):
            scopes = set()
            for series, source_field, semantics in _HBM_OCCUPANCY_SERIES:
                values = []
                for seed in seeds:
                    row = grid[(rate, seed, policy)]
                    context = (
                        f"Poisson HBM occupancy plot: rate={rate:g}, "
                        f"seed={seed}, policy={policy!r}, series={series}"
                    )
                    value = _nonnegative_finite_metric(
                        row, source_field, context=context)
                    if value > 1.0 + 1e-9:
                        raise ExperimentError(
                            f"{context} exceeds capacity fraction: {value}")
                    values.append(value)
                    scope = row.get("hbm_kv_occupancy_scope")
                    if not isinstance(scope, str) or not scope:
                        raise ExperimentError(
                            f"{context} requires HBM occupancy scope")
                    scopes.add(scope)
                summary = _seed_summary(values, clamp_lower_at_zero=True)
                cells.append({
                    "metric": "hbm_kv_occupancy_breakdown",
                    "statistic": series,
                    "unit": "fraction_of_per_rank_kv_capacity",
                    "offered_rate_sessions_per_second": rate,
                    "policy_order": policy_order,
                    "policy": policy,
                    "is_reference": policy == str(reference_label),
                    "arrival_seeds": list(seeds),
                    "seed_values": values,
                    "seed_count": len(values),
                    "mean_across_seeds": summary["mean"],
                    "sample_stddev": summary["sample_stddev"],
                    "ci95_half_width": summary["ci95_half_width"],
                    "ci95_lower": summary["ci95_lower"],
                    "ci95_upper": summary["ci95_upper"],
                    "source_field": source_field,
                    "denominator": next(iter(scopes)),
                    "series_semantics": semantics,
                    "slo": None,
                    "slo_source": None,
                    "slo_basis": None,
                    "slo_provenance": None,
                    "lower_is_better": False,
                })
            if len(scopes) != 1:
                raise ExperimentError(
                    "HBM occupancy scope varies across paired runs: "
                    f"rate={rate:g}, policy={policy!r}, scopes={scopes}")
            physical = {
                cell["statistic"]: cell["mean_across_seeds"]
                for cell in cells
                if cell["offered_rate_sessions_per_second"] == rate
                and cell["policy"] == policy
                and cell["series_semantics"] == "additive_physical_stack"
            }
            if not math.isclose(
                    sum(physical.values()), 1.0,
                    rel_tol=1e-9, abs_tol=1e-9):
                raise ExperimentError(
                    "Seed-aggregated HBM physical stack does not sum to one: "
                    f"rate={rate:g}, policy={policy!r}, values={physical}")
    return rates, policies, cells


def _write_poisson_hbm_occupancy_svg(
        rates, policies, cells, output_dir):
    stack_series = (
        "physical_idle_reusable",
        "physical_non_idle_active",
        "physical_free",
    )
    colors = {
        "physical_idle_reusable": "#4C78A8",
        "physical_non_idle_active": "#F58518",
        "physical_free": "#D9E2EC",
    }
    by_key = {
        (
            cell["offered_rate_sessions_per_second"],
            cell["policy"],
            cell["statistic"],
        ): cell
        for cell in cells
    }
    width = max(1050, 125 * len(rates) * len(policies) + 220)
    height = 650
    left, right, top, bottom = 105, 40, 78, 205
    plot_width = width - left - right
    plot_height = height - top - bottom
    group_width = plot_width / len(rates)
    occupied = group_width * 0.88
    bar_width = occupied / len(policies)

    def y_position(fraction):
        return top + plot_height * (1.0 - float(fraction))

    elements = [
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
         f'height="{height}" viewBox="0 0 {width} {height}">'),
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}'
        '.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}'
        '.tick{font-size:11px}.label{font-size:14px}'
        '.title{font-size:18px;font-weight:600}.subtitle{font-size:11px}'
        '.legend{font-size:12px}.claim{fill:#111;stroke:white;'
        'stroke-width:.8}</style>',
        (f'<text x="{width/2:.1f}" y="28" text-anchor="middle" '
         'class="title">Time-weighted HBM KV occupancy breakdown versus '
         'offered Poisson rate</text>'),
        (f'<text x="{width/2:.1f}" y="48" text-anchor="middle" '
         'class="subtitle">Bars are additive physical ownership means; '
         'black diamonds are the non-additive reservation-adjusted claim; '
         'each value is averaged across fixed arrival seeds</text>'),
    ]
    for tick in range(6):
        fraction = tick / 5
        y = y_position(fraction)
        elements.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            f'y2="{y:.2f}" class="grid"/>',
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
            f'class="tick">{fraction:.0%}</text>',
        ])
    elements.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top+plot_height}" class="axis"/>',
        f'<line x1="{left}" y1="{top+plot_height}" '
        f'x2="{width-right}" y2="{top+plot_height}" class="axis"/>',
    ])
    for rate_index, rate in enumerate(rates):
        group_left = left + rate_index * group_width
        for policy_index, policy in enumerate(policies):
            x = (
                group_left + (group_width - occupied) / 2
                + policy_index * bar_width
            )
            cumulative = 0.0
            for series in stack_series:
                value = by_key[(rate, policy, series)]["mean_across_seeds"]
                y = y_position(cumulative + value)
                bar_height = plot_height * value
                elements.append(
                    f'<rect x="{x+1:.2f}" y="{y:.2f}" '
                    f'width="{max(1.0, bar_width-2):.2f}" '
                    f'height="{bar_height:.2f}" fill="{colors[series]}">'
                    f'<title>{html.escape(policy)}; {series}: '
                    f'{value:.3%}</title></rect>')
                cumulative += value
            claim = by_key[
                (rate, policy, "reservation_adjusted_claim")
            ]["mean_across_seeds"]
            center = x + bar_width / 2
            claim_y = y_position(claim)
            points = (
                f'{center:.2f},{claim_y-5:.2f} '
                f'{center+5:.2f},{claim_y:.2f} '
                f'{center:.2f},{claim_y+5:.2f} '
                f'{center-5:.2f},{claim_y:.2f}'
            )
            elements.append(
                f'<polygon points="{points}" class="claim"><title>'
                f'{html.escape(policy)} reservation-adjusted claim: '
                f'{claim:.3%}</title></polygon>')
            elements.append(
                f'<text x="{center:.2f}" y="{top+plot_height+16}" '
                f'text-anchor="end" class="tick" '
                f'transform="rotate(-38 {center:.2f} '
                f'{top+plot_height+16})">{html.escape(policy)}</text>')
        center = group_left + group_width / 2
        elements.append(
            f'<text x="{center:.2f}" y="{height-92}" '
            f'text-anchor="middle" class="label">rate {rate:g}</text>')
    elements.extend([
        f'<text x="24" y="{top+plot_height/2:.2f}" '
        f'text-anchor="middle" class="label" transform="rotate(-90 24 '
        f'{top+plot_height/2:.2f})">Fraction of per-rank HBM KV '
        'capacity</text>',
    ])
    legend_x = left
    legend_y = height - 38
    labels = (
        ("physical_idle_reusable", "idle reusable physical KV"),
        ("physical_non_idle_active", "non-idle active physical KV"),
        ("physical_free", "physical free"),
    )
    for series, label in labels:
        elements.extend([
            f'<rect x="{legend_x}" y="{legend_y-11}" width="12" '
            f'height="12" fill="{colors[series]}"/>',
            f'<text x="{legend_x+18}" y="{legend_y}" '
            f'class="legend">{html.escape(label)}</text>',
        ])
        legend_x += max(190, len(label) * 7 + 45)
    elements.extend([
        f'<polygon points="{legend_x+6},{legend_y-13} '
        f'{legend_x+12},{legend_y-7} {legend_x+6},{legend_y-1} '
        f'{legend_x},{legend_y-7}" class="claim"/>',
        f'<text x="{legend_x+20}" y="{legend_y}" class="legend">'
        'reservation-adjusted claim (non-additive overlay)</text>',
        '</svg>',
    ])
    path = Path(output_dir) / "poisson_hbm_kv_occupancy_breakdown_by_rate.svg"
    with open(path, "w", encoding="utf-8") as output:
        output.write("\n".join(elements))
        output.write("\n")
    return path


def plot_poisson_rate_metrics(
        rows, output_dir, *, reference_label="infinite_hbm_oracle",
        slo_settings=None):
    """Write fixed-seed Poisson operational-metric CSV and SVG artifacts."""
    rates, policies, cells = _poisson_rate_metric_cells(
        rows,
        reference_label=reference_label,
        slo_settings=slo_settings,
    )
    if not cells:
        return [], None
    occupancy_rates, occupancy_policies, occupancy_cells = (
        _poisson_hbm_occupancy_cells(
            rows, reference_label=reference_label))
    if occupancy_rates != rates or occupancy_policies != policies:
        raise ExperimentError(
            "Poisson HBM occupancy grid differs from rate-metric grid")
    cells.extend(occupancy_cells)
    output_dir = Path(output_dir)
    source_path = output_dir / "poisson_rate_metrics_source.csv"
    source_rows = []
    for cell in cells:
        source_rows.append({
            "online_artifact_schema_version": SCHEMA_VERSION,
            "required_session_report_schema_version": (
                MIN_CURRENT_SESSION_REPORT_SCHEMA_VERSION),
            "operational_source_status": (
                "schema11_exact_measurement_window"),
            "metric": cell["metric"],
            "statistic": cell["statistic"],
            "unit": cell["unit"],
            "offered_rate_sessions_per_second": cell[
                "offered_rate_sessions_per_second"],
            "policy_order": cell["policy_order"],
            "policy": cell["policy"],
            "is_reference": cell["is_reference"],
            "seed_count": cell["seed_count"],
            "arrival_seeds_json": json.dumps(cell["arrival_seeds"]),
            "seed_level_run_statistics_json": json.dumps(
                cell["seed_values"]),
            "mean_across_seeds": cell["mean_across_seeds"],
            "sample_stddev": cell["sample_stddev"],
            "ci95_half_width": cell["ci95_half_width"],
            "ci95_lower": cell["ci95_lower"],
            "ci95_upper": cell["ci95_upper"],
            "source_field": cell["source_field"],
            "denominator": cell["denominator"],
            "slo": cell["slo"],
            "slo_source": cell["slo_source"],
            "slo_basis": cell["slo_basis"],
            "slo_provenance_json": (
                json.dumps(cell["slo_provenance"], sort_keys=True)
                if cell["slo_provenance"] is not None else None
            ),
            "aggregation_unit": "arrival_seed_level_run_statistic",
            "confidence_interval": "two-sided 95% Student-t",
            "lower_is_better": cell["lower_is_better"],
            "series_semantics": cell.get("series_semantics"),
        })
    with open(source_path, "w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    paths = []
    for definition in _POISSON_RATE_METRIC_DEFINITIONS:
        metric_cells = [
            cell for cell in cells if cell["metric"] == definition["name"]
        ]
        paths.append(_write_poisson_rate_metric_svg(
            definition, rates, policies, metric_cells, output_dir))
    paths.append(_write_poisson_hbm_occupancy_svg(
        rates, policies, occupancy_cells, output_dir))
    return paths, source_path


def plot_backlog_oracle_normalized_throughput(
        rows, output_dir, *, oracle_label="infinite_hbm_oracle",
        minimum_k=10):
    """Write a paired, oracle-normalized backlog grouped-bar SVG."""
    minimum_k = int(minimum_k)
    mode_rows = [
        row for row in rows
        if (row.get("mode") == "backlog"
            and float(row["load_value"]) >= minimum_k)
    ]
    if not mode_rows:
        raise ExperimentError(
            "Oracle-normalized backlog plot has no rows at or above "
            f"K={minimum_k}")
    loads = sorted({float(row["load_value"]) for row in mode_rows})
    policies = _ordered_plot_policies(mode_rows)
    if oracle_label not in policies:
        raise ExperimentError(
            "Oracle-normalized backlog plot is missing oracle policy "
            f"{oracle_label!r}")

    values_by_policy = {policy: [] for policy in policies}
    derived_cells = []
    for load in loads:
        load_rows = [
            row for row in mode_rows
            if float(row["load_value"]) == load
        ]
        provenance_by_pair_key = {}
        pair_key_by_provenance = {}
        for row in load_rows:
            pair_key = str(
                row.get("pair_key")
                or f"load={load}:seed={row.get('arrival_seed')}")
            provenance = (load, row.get("arrival_seed"))
            prior = provenance_by_pair_key.setdefault(pair_key, provenance)
            if prior != provenance:
                raise ExperimentError(
                    "Oracle-normalized backlog plot pair_key maps to "
                    "conflicting provenance: "
                    f"pair_key={pair_key!r}, {prior} versus {provenance}")
            prior_key = pair_key_by_provenance.setdefault(
                provenance, pair_key)
            if prior_key != pair_key:
                raise ExperimentError(
                    "Oracle-normalized backlog plot provenance maps to "
                    "multiple pair keys: "
                    f"provenance={provenance}, pair_keys={prior_key!r}, "
                    f"{pair_key!r}")
        by_policy = {}
        for policy in policies:
            cells = {}
            for row in load_rows:
                if str(row["policy"]) != policy:
                    continue
                pair_key = str(
                    row.get("pair_key")
                    or f"load={load}:seed={row.get('arrival_seed')}")
                if pair_key in cells:
                    raise ExperimentError(
                        "Oracle-normalized backlog plot has duplicate paired "
                        f"row: load={load}, policy={policy!r}, "
                        f"pair_key={pair_key!r}")
                value = float(row["sessions_per_second"])
                if not math.isfinite(value) or value <= 0:
                    raise ExperimentError(
                        "Oracle-normalized backlog plot requires positive "
                        f"finite throughput: load={load}, policy={policy!r}, "
                        f"value={value}")
                cells[pair_key] = value
            by_policy[policy] = cells
        oracle_cells = by_policy[oracle_label]
        if not oracle_cells:
            raise ExperimentError(
                f"Oracle-normalized backlog plot has no oracle row at K={load:g}")
        oracle_pair_keys = set(oracle_cells)
        for policy in policies:
            cells = by_policy[policy]
            if set(cells) != oracle_pair_keys:
                raise ExperimentError(
                    "Oracle-normalized backlog plot has unpaired rows: "
                    f"load={load}, policy={policy!r}, "
                    f"policy_pairs={sorted(cells)}, "
                    f"oracle_pairs={sorted(oracle_pair_keys)}")
            ratios = [
                cells[pair_key] / oracle_cells[pair_key]
                for pair_key in sorted(oracle_pair_keys)
            ]
            mean_ratio = sum(ratios) / len(ratios)
            values_by_policy[policy].append(mean_ratio)
            derived_cells.append({
                "load_k": int(load),
                "policy": policy,
                "mean_paired_throughput_ratio": mean_ratio,
                "pairs": [
                    {
                        "pair_key": pair_key,
                        "arrival_seed": provenance_by_pair_key[pair_key][1],
                        "policy_sessions_per_second": cells[pair_key],
                        "oracle_sessions_per_second": oracle_cells[pair_key],
                        "throughput_ratio": (
                            cells[pair_key] / oracle_cells[pair_key]),
                    }
                    for pair_key in sorted(oracle_pair_keys)
                ],
            })

    palette = (
        "#4C78A8", "#F58518", "#54A24B", "#E45756",
        "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D",
    )
    legend_item_widths = [
        max(120, len(policy) * 7 + 32) for policy in policies
    ]
    width = max(
        840,
        200 * len(loads) + 280,
        sum(legend_item_widths) + 120,
    )
    height = 560
    left, right, top, bottom = 90, 30, 70, 145
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(
        value
        for values in values_by_policy.values()
        for value in values
    )
    y_max = max(1.1, maximum * 1.14)
    group_width = plot_width / len(loads)
    occupied_width = group_width * 0.82
    bar_width = occupied_width / len(policies)
    elements = [
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
         f'height="{height}" viewBox="0 0 {width} {height}">'),
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}'
        '.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}'
        '.oracle{stroke:#555;stroke-width:1.5;stroke-dasharray:5 4}'
        '.tick{font-size:12px}.value{font-size:10px}.label{font-size:14px}'
        '.title{font-size:18px;font-weight:600}.legend{font-size:12px}'
        '</style>',
        (f'<text x="{width / 2:.1f}" y="29" text-anchor="middle" '
         'class="title">Backlog throughput normalized to infinite-HBM '
         'residency reference</text>'),
    ]
    for tick in range(6):
        fraction = tick / 5
        y = top + plot_height * (1 - fraction)
        value = y_max * fraction
        elements.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            f'y2="{y:.2f}" class="grid"/>')
        elements.append(
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
            f'class="tick">{value:.2f}</text>')
    oracle_y = top + plot_height * (1 - 1 / y_max)
    elements.append(
        f'<line x1="{left}" y1="{oracle_y:.2f}" x2="{width-right}" '
        f'y2="{oracle_y:.2f}" class="oracle"/>')
    elements.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top+plot_height}" class="axis"/>',
        f'<line x1="{left}" y1="{top+plot_height}" '
        f'x2="{width-right}" y2="{top+plot_height}" class="axis"/>',
    ])
    for load_index, load in enumerate(loads):
        group_left = left + load_index * group_width
        for policy_index, policy in enumerate(policies):
            value = values_by_policy[policy][load_index]
            bar_height = plot_height * value / y_max
            x = (group_left + (group_width - occupied_width) / 2
                 + policy_index * bar_width)
            y = top + plot_height - bar_height
            color = palette[policy_index % len(palette)]
            elements.append(
                f'<rect x="{x+1:.2f}" y="{y:.2f}" '
                f'width="{max(1.0, bar_width-2):.2f}" '
                f'height="{bar_height:.2f}" fill="{color}">'
                f'<title>{html.escape(policy)}: {value:.6g}</title>'
                '</rect>')
            elements.append(
                f'<text x="{x+bar_width/2:.2f}" y="{max(top+10, y-4):.2f}" '
                f'text-anchor="middle" class="value">{value:.2f}</text>')
        center = group_left + group_width / 2
        elements.append(
            f'<text x="{center:.2f}" y="{top+plot_height+22}" '
            f'text-anchor="middle" class="tick">{int(load)}</text>')
    elements.append(
        f'<text x="{left+plot_width/2:.2f}" y="{height-76}" '
        'text-anchor="middle" class="label">Active sessions K</text>')
    elements.append(
        f'<text x="20" y="{top+plot_height/2:.2f}" '
        f'text-anchor="middle" class="label" '
        f'transform="rotate(-90 20 {top+plot_height/2:.2f})">'
        'Completed sessions/s ÷ HBM reference</text>')
    legend_x = left
    legend_y = height - 28
    for policy_index, policy in enumerate(policies):
        color = palette[policy_index % len(palette)]
        elements.append(
            f'<rect x="{legend_x}" y="{legend_y-11}" width="12" '
            f'height="12" fill="{color}"/>')
        elements.append(
            f'<text x="{legend_x+18}" y="{legend_y}" '
            f'class="legend">{html.escape(policy)}</text>')
        legend_x += legend_item_widths[policy_index]
    elements.append('</svg>')

    path = (
        Path(output_dir)
        / f"backlog_throughput_oracle_normalized_k{minimum_k}plus.svg"
    )
    with open(path, "w", encoding="utf-8") as output:
        output.write("\n".join(elements))
        output.write("\n")
    _write_json(path.with_suffix(".json"), {
        "schema_version": SCHEMA_VERSION,
        "metric": "sessions_per_second",
        "formula": (
            "mean_over_pairs(policy_sessions_per_second / "
            "oracle_sessions_per_second)"
        ),
        "oracle_label": oracle_label,
        "reference_semantics": (
            "Infinite HBM makes capacity nonbinding and preserves reusable "
            "KV residency, but retains modeled P/D transfer and compute "
            "dependencies; it is not asserted to be a universal throughput "
            "upper bound."
        ),
        "minimum_k": minimum_k,
        "cells": derived_cells,
    })
    return path


def build_backlog_slowdown_audit(
        rows, *, oracle_label="infinite_hbm_oracle",
        required_max_slowdown_fraction=None):
    baselines = sorted({
        row["policy"]
        for row in rows
        if row["mode"] == "backlog" and row["policy"] != oracle_label
    })
    thresholds = (0.5, 1.0)
    per_policy = {}
    for policy in baselines:
        policy_rows = sorted(
            (
                row for row in rows
                if row["mode"] == "backlog" and row["policy"] == policy
            ),
            key=lambda row: row["load_value"],
        )
        cells = {}
        for threshold in thresholds:
            crossing = next((
                row for row in policy_rows
                if row["oracle_throughput_slowdown_fraction"] is not None
                and row["oracle_throughput_slowdown_fraction"] >= threshold
            ), None)
            cells[str(threshold)] = {
                "reached": crossing is not None,
                "first_k": (
                    int(crossing["load_value"])
                    if crossing is not None else None),
                "observed_slowdown_fraction": (
                    crossing["oracle_throughput_slowdown_fraction"]
                    if crossing is not None else None),
            }
        slowdowns = [
            row["oracle_throughput_slowdown_fraction"]
            for row in policy_rows
            if row["oracle_throughput_slowdown_fraction"] is not None
        ]
        per_policy[policy] = {
            "threshold_crossings": cells,
            "max_observed_slowdown_fraction": max(slowdowns, default=None),
            "swept_k_values": [
                int(row["load_value"]) for row in policy_rows],
        }

    required = None
    passed = True
    if required_max_slowdown_fraction is not None:
        try:
            required = float(required_max_slowdown_fraction)
        except (TypeError, ValueError) as exc:
            raise ExperimentError(
                "required_max_slowdown_fraction must be numeric") from exc
        if not math.isfinite(required) or required < 0:
            raise ExperimentError(
                "required_max_slowdown_fraction must be finite and "
                "non-negative")
        passed = any(
            value["max_observed_slowdown_fraction"] is not None
            and value["max_observed_slowdown_fraction"] >= required
            for value in per_policy.values()
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "metric": "oracle_throughput_slowdown_fraction",
        "definition": "oracle_sessions_per_second / policy_sessions_per_second - 1",
        "thresholds": list(thresholds),
        "per_policy": per_policy,
        "required_max_slowdown_fraction": required,
        "required_target_reached_by_any_non_oracle": passed,
        "passed": passed,
    }


def _normalize_ssd_resume_opportunity_contract(
        contract, *, configured_modes=None, configured_policies=None):
    if contract is None:
        return None
    if not isinstance(contract, dict):
        raise ExperimentError(
            "ssd_resume_opportunity_contract must be a JSON object")
    allowed_keys = {
        "mode", "policy", "minimum_fraction_of_all_requests",
    }
    unknown_keys = sorted(set(contract) - allowed_keys)
    if unknown_keys:
        raise ExperimentError(
            "ssd_resume_opportunity_contract has unsupported fields: "
            f"{unknown_keys}")

    mode = contract.get("mode", "backlog")
    if not isinstance(mode, str) or mode not in SUPPORTED_EXPERIMENT_MODES:
        raise ExperimentError(
            "ssd_resume_opportunity_contract.mode must be one of "
            f"{list(SUPPORTED_EXPERIMENT_MODES)}")
    policy = contract.get("policy")
    if not isinstance(policy, str) or not policy.strip():
        raise ExperimentError(
            "ssd_resume_opportunity_contract.policy must be a non-empty "
            "policy label")
    policy = policy.strip()
    required = contract.get("minimum_fraction_of_all_requests")
    if (not isinstance(required, (int, float))
            or isinstance(required, bool)):
        raise ExperimentError(
            "ssd_resume_opportunity_contract."
            "minimum_fraction_of_all_requests must be numeric")
    required = float(required)
    if not math.isfinite(required) or not 0.0 <= required <= 1.0:
        raise ExperimentError(
            "ssd_resume_opportunity_contract."
            "minimum_fraction_of_all_requests must be finite and in [0, 1]")
    if configured_modes is not None and mode not in configured_modes:
        raise ExperimentError(
            "ssd_resume_opportunity_contract references unconfigured mode "
            f"{mode!r}")
    if configured_policies is not None and policy not in configured_policies:
        raise ExperimentError(
            "ssd_resume_opportunity_contract references unknown policy "
            f"label {policy!r}")
    return {
        "mode": mode,
        "policy": policy,
        "minimum_fraction_of_all_requests": required,
    }


def _prepare_ssd_resume_opportunity_contract(spec, selected_modes):
    """Validate the declared contract and select it for this partial run."""
    contract = _normalize_ssd_resume_opportunity_contract(
        spec.get("ssd_resume_opportunity_contract"),
        configured_modes=spec.get("modes", {}),
        configured_policies=spec.get("policies", {}),
    )
    active_contract = (
        contract
        if contract is not None and contract["mode"] in selected_modes
        else None
    )
    return contract, active_contract


def build_ssd_resume_opportunity_audit(rows, contract):
    """Audit a policy's SSD resumes against all measured requests.

    Replicated arrival seeds are pooled from their integer counts at each
    load.  This intentionally avoids averaging per-run fractions with unequal
    request denominators.
    """
    contract = _normalize_ssd_resume_opportunity_contract(contract)
    mode = contract["mode"]
    policy = contract["policy"]
    required = contract["minimum_fraction_of_all_requests"]

    target_rows = [
        row for row in rows
        if row.get("mode") == mode and row.get("policy") == policy
    ]
    if not target_rows:
        raise ExperimentError(
            "SSD resume opportunity contract has no result rows for "
            f"mode={mode!r}, policy={policy!r}")

    by_load = {}
    for row in target_rows:
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ExperimentError(
                "SSD resume opportunity contract encountered a target row "
                "without a run_id")
        load_value = row.get("load_value")
        if (not isinstance(load_value, (int, float))
                or isinstance(load_value, bool)
                or not math.isfinite(float(load_value))):
            raise ExperimentError(
                f"Invalid load_value for SSD opportunity audit in {run_id}")
        load_value = float(load_value)

        ssd_count = row.get("attempted_ssd_resume_count")
        request_count = row.get("session_cohort_request_count")
        if (not isinstance(ssd_count, int) or isinstance(ssd_count, bool)
                or ssd_count < 0):
            raise ExperimentError(
                f"Invalid attempted_ssd_resume_count for SSD opportunity "
                f"audit in "
                f"{run_id}")
        if (not isinstance(request_count, int)
                or isinstance(request_count, bool)
                or request_count <= 0):
            raise ExperimentError(
                "SSD opportunity audit requires a positive integer "
                f"session_cohort_request_count in {run_id}")
        if ssd_count > request_count:
            raise ExperimentError(
                "attempted_ssd_resume_count exceeds the all-request "
                "denominator in "
                f"{run_id}")

        reported_fraction = row.get(
            "attempted_ssd_resume_fraction_of_all_requests")
        expected_fraction = ssd_count / request_count
        if (not isinstance(reported_fraction, (int, float))
                or isinstance(reported_fraction, bool)
                or not math.isfinite(float(reported_fraction))
                or not math.isclose(
                    float(reported_fraction), expected_fraction,
                    rel_tol=1e-12, abs_tol=1e-12)):
            raise ExperimentError(
                "SSD resume fraction does not reconcile with its integer "
                f"counts in {run_id}: reported={reported_fraction}, "
                f"expected={expected_fraction}")

        cell = by_load.setdefault(load_value, {
            "attempted_ssd_resume_count": 0,
            "all_request_count": 0,
            "run_ids": [],
            "arrival_seeds": [],
        })
        cell["attempted_ssd_resume_count"] += ssd_count
        cell["all_request_count"] += request_count
        cell["run_ids"].append(run_id)
        cell["arrival_seeds"].append(row.get("arrival_seed"))

    per_load = []
    for load_value, cell in sorted(by_load.items()):
        observed = (
            cell["attempted_ssd_resume_count"]
            / cell["all_request_count"])
        per_load.append({
            "load_value": load_value,
            "run_count": len(cell["run_ids"]),
            "run_ids": sorted(cell["run_ids"]),
            "arrival_seeds": sorted(
                cell["arrival_seeds"],
                key=lambda value: (value is not None, str(value)),
            ),
            "attempted_ssd_resume_count": cell[
                "attempted_ssd_resume_count"],
            "all_request_count": cell["all_request_count"],
            "attempted_ssd_resume_fraction_of_all_requests": observed,
            "target_reached": observed >= required,
        })

    crossing = next(
        (cell for cell in per_load if cell["target_reached"]), None)
    max_observed = max(
        cell["attempted_ssd_resume_fraction_of_all_requests"]
        for cell in per_load)
    return {
        "schema_version": SCHEMA_VERSION,
        "metric": "attempted_ssd_resume_fraction_of_all_requests",
        "definition": (
            "sum(attempted_ssd_resume_count) / sum(all_request_count)"),
        "denominator_scope": (
            "all completed requests in the measured session cohort"),
        "seed_aggregation": (
            "integer counts pooled independently at each load value"),
        "mode": mode,
        "policy": policy,
        "minimum_fraction_of_all_requests": required,
        "per_load": per_load,
        "max_observed_fraction_of_all_requests": max_observed,
        "first_reaching_load_value": (
            crossing["load_value"] if crossing is not None else None),
        "passed": crossing is not None,
    }


def run_suite(
        spec_path, *, max_parallel=None, timeout_seconds=None,
        dataset_path=None, dataset_manifest_path=None, modes=None):
    repo_root = Path(__file__).resolve().parents[1]
    spec_path = _resolve(repo_root, spec_path)
    declared_spec = _load_json(spec_path)
    spec, dataset_path_overrides = _apply_dataset_path_overrides(
        declared_spec,
        repo_root,
        dataset_path=dataset_path,
        manifest_path=dataset_manifest_path,
    )
    selected_modes = _normalize_mode_selection(modes, spec.get("modes", {}))
    plot_settings = _normalize_plot_settings(
        spec.get("plots"), spec.get("modes", {}))
    if plot_settings:
        spec["plots"] = plot_settings
    else:
        spec.pop("plots", None)
    parallelism = int(
        spec.get("max_parallel", 1)
        if max_parallel is None else max_parallel)
    timeout = float(
        spec.get("timeout_seconds", DEFAULT_RUN_WALL_SECONDS)
        if timeout_seconds is None else timeout_seconds)
    if parallelism <= 0:
        raise ExperimentError("max_parallel must be positive")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ExperimentError(
            "timeout_seconds must be positive and finite")
    if timeout > MAX_RUN_WALL_SECONDS:
        raise ExperimentError(
            "timeout_seconds exceeds the hard per-run wall cap of "
            f"{MAX_RUN_WALL_SECONDS:g} seconds")
    ssd_resume_contract, active_ssd_resume_contract = (
        _prepare_ssd_resume_opportunity_contract(spec, selected_modes))
    if ssd_resume_contract is not None:
        spec["ssd_resume_opportunity_contract"] = ssd_resume_contract
    base_output_dir = _resolve(
        repo_root,
        spec.get("output_dir", f"results/online/{spec_path.stem}"),
    )
    execution_id = str(
        spec.get("execution_id")
        or _new_execution_id(spec.get("name", spec_path.stem))
    )
    execution_id = _slug(execution_id)
    if not execution_id:
        raise ExperimentError("execution_id becomes empty after normalization")
    output_dir = base_output_dir / "executions" / execution_id
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ExperimentError(
            f"Refusing to overwrite existing execution: {output_dir}") from exc
    spec_snapshot_path = output_dir / "spec_snapshot.json"
    _write_json(spec_snapshot_path, spec)
    cohort = materialize_session_cohort(
        _resolve(repo_root, spec["dataset"]),
        output_dir,
        spec.get("workload_selection"),
        dataset_contract=spec.get("dataset_contract"),
        repo_root=repo_root,
    )
    runs = build_run_descriptors(
        spec, repo_root, output_dir, cohort, execution_id=execution_id,
        selected_modes=selected_modes)
    provenance = {
        "created_at": _utc_now(),
        "execution_id": execution_id,
        "base_output_dir": str(base_output_dir),
        "execution_output_dir": str(output_dir),
        "git": _git_provenance(repo_root),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "spec_path": str(spec_path),
        "spec_sha256": _sha256_file(spec_path),
        "spec_snapshot_path": str(spec_snapshot_path.resolve()),
        "spec_snapshot_sha256": _sha256_file(spec_snapshot_path),
        "dataset_path_overrides": dataset_path_overrides,
        "selected_modes": list(selected_modes),
        "cohort": cohort,
    }
    suite_manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": spec.get("name", spec_path.stem),
        "execution_id": execution_id,
        "output_dir": str(output_dir),
        "status": "running",
        "max_parallel": parallelism,
        "timeout_seconds_per_run": timeout,
        "selected_modes": list(selected_modes),
        "ssd_resume_opportunity_contract": ssd_resume_contract,
        "ssd_resume_opportunity_contract_mode_selected": (
            active_ssd_resume_contract is not None),
        "online_subprocess_prefix": [sys.executable, "-m", "serving"],
        "provenance": provenance,
        "runs": runs,
    }
    _write_json(output_dir / "suite_manifest.json", suite_manifest)

    manifests = []
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = {
            executor.submit(
                execute_run, run, repo_root, timeout, provenance): run
            for run in runs
        }
        for future in as_completed(futures):
            try:
                manifests.append(future.result())
            except Exception as exc:
                run = futures[future]
                failed_manifest = {
                    "schema_version": SCHEMA_VERSION,
                    **{key: value for key, value in run.items()
                       if key != "run_dir"},
                    "run_dir": run["run_dir"],
                    "status": "runner_failed",
                    "finished_at": _utc_now(),
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "provenance": provenance,
                }
                _write_json(
                    Path(run["run_dir"]) / "run_manifest.json",
                    failed_manifest,
                )
                manifests.append(failed_manifest)
    manifests.sort(key=lambda item: item["run_id"])
    failures = [
        manifest for manifest in manifests
        if manifest.get("status") != "succeeded"
    ]
    rows = []
    plot_paths = []
    plot_data_paths = []
    result_paths = {}
    slowdown_audit = None
    ssd_resume_opportunity_audit = None
    validation_error = None
    if not failures:
        try:
            oracle_label = str(spec.get(
                "oracle_label", "infinite_hbm_oracle"))
            rows = collect_results(
                manifests,
                oracle_label=oracle_label,
            )
            result_paths = save_results(rows, output_dir)
            plot_paths = plot_grouped_throughput(rows, output_dir)
            if "poisson" in selected_modes:
                jct_plot_paths, jct_source_path = (
                    plot_poisson_session_jct(
                        rows,
                        output_dir,
                        oracle_label=oracle_label,
                    )
                )
                plot_paths.extend(jct_plot_paths)
                if jct_source_path is not None:
                    plot_data_paths.append(jct_source_path)
                normalized_jct_paths, normalized_jct_source = (
                    plot_poisson_reference_normalized_jct(
                        rows,
                        output_dir,
                        reference_label=oracle_label,
                    )
                )
                plot_paths.extend(normalized_jct_paths)
                if normalized_jct_source is not None:
                    plot_data_paths.append(normalized_jct_source)
                decomposition_paths, decomposition_source = (
                    plot_poisson_session_jct_decomposition(
                        rows,
                        output_dir,
                        reference_label=oracle_label,
                    )
                )
                plot_paths.extend(decomposition_paths)
                if decomposition_source is not None:
                    plot_data_paths.append(decomposition_source)
                rate_metric_paths, rate_metric_source = (
                    plot_poisson_rate_metrics(
                        rows,
                        output_dir,
                        reference_label=oracle_label,
                        slo_settings=plot_settings.get(
                            "poisson_rate_metrics", {}),
                    )
                )
                plot_paths.extend(rate_metric_paths)
                if rate_metric_source is not None:
                    plot_data_paths.append(rate_metric_source)
            normalized_backlog = plot_settings.get(
                "backlog_oracle_normalized")
            if normalized_backlog is not None and "backlog" in selected_modes:
                normalized_plot_path = (
                    plot_backlog_oracle_normalized_throughput(
                        rows,
                        output_dir,
                        oracle_label=oracle_label,
                        minimum_k=normalized_backlog["minimum_k"],
                    )
                )
                plot_paths.append(normalized_plot_path)
                plot_data_paths.append(
                    normalized_plot_path.with_suffix(".json"))
            if "backlog" in selected_modes:
                slowdown_audit = build_backlog_slowdown_audit(
                    rows,
                    oracle_label=oracle_label,
                    required_max_slowdown_fraction=spec.get(
                        "required_max_slowdown_fraction"),
                )
                _write_json(
                    output_dir / "backlog_slowdown_targets.json",
                    slowdown_audit,
                )
                if not slowdown_audit["passed"]:
                    validation_error = (
                        "No non-oracle backlog baseline reached required "
                        "maximum slowdown fraction "
                        f"{slowdown_audit['required_max_slowdown_fraction']}"
                    )
            if active_ssd_resume_contract is not None:
                ssd_resume_opportunity_audit = (
                    build_ssd_resume_opportunity_audit(
                        rows, active_ssd_resume_contract))
                _write_json(
                    output_dir / "ssd_resume_opportunity_contract.json",
                    ssd_resume_opportunity_audit,
                )
                if not ssd_resume_opportunity_audit["passed"]:
                    required_fraction = ssd_resume_opportunity_audit[
                        "minimum_fraction_of_all_requests"]
                    max_observed_fraction = ssd_resume_opportunity_audit[
                        "max_observed_fraction_of_all_requests"]
                    opportunity_error = (
                        "Policy "
                        f"{ssd_resume_opportunity_audit['policy']!r} never "
                        "reached required SSD resume fraction of all "
                        f"requests {required_fraction} in "
                        f"{ssd_resume_opportunity_audit['mode']} mode; "
                        "maximum observed fraction was "
                        f"{max_observed_fraction}"
                    )
                    validation_error = (
                        f"{validation_error}; {opportunity_error}"
                        if validation_error else opportunity_error)
        except ExperimentError as exc:
            validation_error = str(exc)
        except Exception as exc:
            validation_error = (
                f"Online result postprocessing failed with "
                f"{type(exc).__name__}: {exc}")
    status = (
        "failed" if failures else
        "failed_validation" if validation_error else "succeeded"
    )
    suite_manifest.update({
        "status": status,
        "finished_at": _utc_now(),
        "run_manifests": [
            str(Path(manifest["run_dir"]) / "run_manifest.json")
            for manifest in manifests
        ],
        "failed_run_ids": [manifest["run_id"] for manifest in failures],
        "summary_csv": (
            str(result_paths["combined"].resolve()) if rows else None),
        "mode_summary_csvs": {
            mode: str(path.resolve()) if path is not None else None
            for mode, path in result_paths.items()
            if mode != "combined"
        },
        "plots": [str(path.resolve()) for path in plot_paths],
        "plot_data": [str(path.resolve()) for path in plot_data_paths],
        "backlog_slowdown_audit": (
            str((output_dir / "backlog_slowdown_targets.json").resolve())
            if slowdown_audit is not None else None
        ),
        "ssd_resume_opportunity_audit": (
            str((
                output_dir / "ssd_resume_opportunity_contract.json"
            ).resolve())
            if ssd_resume_opportunity_audit is not None else None
        ),
        "validation_error": validation_error,
    })
    _write_json(output_dir / "suite_manifest.json", suite_manifest)
    if failures:
        raise ExperimentError(
            f"{len(failures)} online run(s) failed or timed out: "
            + ", ".join(item["run_id"] for item in failures))
    if validation_error:
        raise ExperimentError(validation_error)
    return suite_manifest


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m serving.online_experiments",
        description=(
            "Run paired backlog/Poisson sweeps exclusively through "
            "python -m serving"
        ),
    )
    parser.add_argument("--spec", required=True, help="experiment JSON spec")
    parser.add_argument(
        "--mode", dest="modes", action="append",
        choices=SUPPORTED_EXPERIMENT_MODES,
        help=(
            "run only this configured mode; repeat to select both "
            "(default: every mode in the spec)"
        ),
    )
    parser.add_argument(
        "--dataset-override", default=None,
        help=(
            "local converted-dataset path; checked-in SHA/schema/count "
            "contract remains authoritative"
        ),
    )
    parser.add_argument(
        "--dataset-manifest-override", default=None,
        help=(
            "local companion-manifest path; checked-in manifest SHA remains "
            "authoritative"
        ),
    )
    parser.add_argument(
        "--max-parallel", type=int, default=None,
        help="override bounded subprocess parallelism",
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=None,
        help=(
            "override per-simulation wall timeout; the runner enforces a "
            "hard 3600-second cap (default: 600 seconds)"
        ),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        manifest = run_suite(
            args.spec,
            max_parallel=args.max_parallel,
            timeout_seconds=args.timeout_seconds,
            dataset_path=args.dataset_override,
            dataset_manifest_path=args.dataset_manifest_override,
            modes=args.modes,
        )
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(manifest["summary_csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

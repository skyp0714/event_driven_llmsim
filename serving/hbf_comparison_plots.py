"""Validate, aggregate, and plot a completed HBF comparison sweep.

The input to this module is the immutable bundle emitted by
``serving.hbf_comparison_sweep``.  Loading is deliberately fail-closed:
top-level self-hashes, paired-schedule contracts, every cell completion
marker, every artifact hash, live trace/config/source hashes, the analytical
backend declaration, full-drain identities, and metric formulas are checked
before a seed value is accepted.

The statistical replicate is one arrival seed.  Request rows are never pooled.
For every ``(rate, system, metric)`` point, the arithmetic mean and two-sided
Student-t 95% confidence interval are computed from the seed-level cell
summaries through :func:`aggregate_seed_values`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Callable, Mapping, Optional, Sequence

from .core.hbf_comparison_cell import (
    ASTRA_CYCLES_USED,
    CELL_SCHEMA_VERSION,
    HBF_CONFIG_PATHS,
    SIMULATION_BACKEND,
    SYSTEM_KEYS,
    write_json_atomic,
)
from .core.hbf_comparison_metrics import (
    SeedAggregate,
    aggregate_seed_values,
)
from .core.hbf_comparison_workload import stable_json_sha256
from .hbf_comparison_sweep import (
    CELL_JSON,
    COMPLETION_JSON,
    COMPLETION_MARKER_SCHEMA_VERSION,
    REQUEST_CSV,
    SWEEP_SCHEMA_VERSION,
    TOP_LEVEL_MANIFEST,
    ComparisonSweepError,
    _artifact_record,
    _rate_directory,
    _rate_text,
    _strict_json_object,
    _validate_csv_row_count,
)


AGGREGATE_SCHEMA_VERSION = 1
FONT_SIZE = 24
PNG_DPI = 200
LATENCY_FIGSIZE = (12, 18)
THROUGHPUT_FIGSIZE = (12, 24)

MAIN_SYSTEMS = (
    "recompute",
    "ssd_direct",
    "cpu_ssd",
    "oracle",
    "hbf_tp4_wide",
)
HBF_LAYOUT_SYSTEMS = (
    "hbf_dp8",
    "hbf_tp4",
    "hbf_tp8",
    "hbf_tp4_wide",
)

SYSTEM_LABELS = {
    "recompute": "Recompute",
    "ssd_direct": "SSD-direct",
    "cpu_ssd": "CPU+SSD",
    "oracle": "Infinite-HBM oracle",
    "hbf_dp8": "HBF DP8",
    "hbf_tp4": "HBF 2×TP4",
    "hbf_tp8": "HBF TP8",
    "hbf_tp4_wide": "HBF 2×TP4, wide LPDDR",
}

SYSTEM_STYLES = {
    "recompute": {"color": "#d62728", "marker": "o", "linestyle": "-"},
    "ssd_direct": {"color": "#ff7f0e", "marker": "s", "linestyle": "-"},
    "cpu_ssd": {"color": "#9467bd", "marker": "^", "linestyle": "-"},
    "oracle": {"color": "#111111", "marker": "D", "linestyle": "--"},
    "hbf_dp8": {"color": "#8c564b", "marker": "v", "linestyle": "-"},
    "hbf_tp4": {"color": "#2ca02c", "marker": "P", "linestyle": "-"},
    "hbf_tp8": {"color": "#17becf", "marker": "X", "linestyle": "-"},
    "hbf_tp4_wide": {
        "color": "#1f77b4", "marker": "*", "linestyle": "-",
    },
}

MATPLOTLIB_RC = {
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
}


class ComparisonPlotInputError(RuntimeError):
    """Raised when an input artifact fails a provenance or schema check."""


class ComparisonPlotRenderError(RuntimeError):
    """Raised when Matplotlib cannot render the validated aggregate."""


@dataclass(frozen=True)
class MetricSpec:
    """Definition of one seed-level metric consumed from ``cell.json``."""

    key: str
    title: str
    y_label: str
    source_path: tuple[str, ...]
    scale: float
    unit: str
    positive: bool
    bounded_fraction: bool = False
    log_scale: bool = False


METRIC_SPECS = (
    MetricSpec(
        key="first_ttft_p95_seconds",
        title="First-turn TTFT p95",
        y_label="First TTFT p95 (s)",
        source_path=(
            "summary", "latency_distributions_ns",
            "first_ttft", "p95_ns",
        ),
        scale=1e-9,
        unit="seconds",
        positive=True,
        log_scale=True,
    ),
    MetricSpec(
        key="resume_ttft_p95_seconds",
        title="Resume TTFT p95",
        y_label="Resume TTFT p95 (s)",
        source_path=(
            "summary", "latency_distributions_ns",
            "resume_ttft", "p95_ns",
        ),
        scale=1e-9,
        unit="seconds",
        positive=True,
        log_scale=True,
    ),
    MetricSpec(
        key="tpot_p95_milliseconds",
        title="TPOT p95",
        y_label="TPOT p95 (ms/token)",
        source_path=(
            "summary", "latency_distributions_ns",
            "tpot_eligible", "p95_ns",
        ),
        scale=1e-6,
        unit="milliseconds_per_token",
        positive=True,
        log_scale=True,
    ),
    MetricSpec(
        key="joint_slo_pass_fraction",
        title="Joint TTFT+TPOT SLO attainment",
        y_label="Joint SLO pass fraction",
        source_path=("summary", "slo", "all_slo_pass_fraction"),
        scale=1.0,
        unit="fraction",
        positive=False,
        bounded_fraction=True,
    ),
    MetricSpec(
        key="slo_request_goodput_per_second",
        title="SLO request goodput",
        y_label="SLO-good requests/s",
        source_path=(
            "summary",
            "offered_load_normalized_request_goodput",
            "value",
        ),
        scale=1.0,
        unit="requests_per_second",
        positive=False,
    ),
    MetricSpec(
        key="slo_output_token_goodput_per_second",
        title="SLO output-token goodput",
        y_label="SLO-good output tokens/s",
        source_path=(
            "summary",
            "offered_load_normalized_output_token_goodput",
            "value",
        ),
        scale=1.0,
        unit="output_tokens_per_second",
        positive=False,
    ),
    MetricSpec(
        key="observed_request_throughput_per_second",
        title="Observed inter-completion throughput",
        y_label="Observed requests/s",
        source_path=(
            "summary",
            "observed_completion_span_throughput",
            "requests_per_second",
        ),
        scale=1.0,
        unit="requests_per_second",
        positive=True,
    ),
)

METRIC_BY_KEY = {spec.key: spec for spec in METRIC_SPECS}
LATENCY_METRICS = tuple(spec.key for spec in METRIC_SPECS[:3])
THROUGHPUT_METRICS = tuple(spec.key for spec in METRIC_SPECS[3:])


@dataclass(frozen=True)
class SeedCellMetrics:
    """Validated seed-level measurements from one immutable sweep cell."""

    seed: int
    session_rate: float
    rate_text: str
    system_key: str
    schedule_pair_sha256: str
    values: Mapping[str, float]


@dataclass(frozen=True)
class ValidatedSweep:
    """Provenance plus the minimal metrics retained from all cells."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, object]
    rates: tuple[float, ...]
    rate_texts: tuple[str, ...]
    seeds: tuple[int, ...]
    system_keys: tuple[str, ...]
    cells: tuple[SeedCellMetrics, ...]
    metric_keys: tuple[str, ...] = tuple(
        spec.key for spec in METRIC_SPECS)


@dataclass(frozen=True)
class AggregatePoint:
    """One rate/system point with the selected seed-level aggregates."""

    session_rate: float
    rate_text: str
    system_key: str
    metrics: Mapping[str, SeedAggregate]


@dataclass(frozen=True)
class ComparisonAggregate:
    """Complete paired grid ready for tables and rendering."""

    source_manifest_path: str
    source_manifest_sha256: str
    source_cells_sha256: str
    simulation_backend: str
    astra_cycles_used: bool
    rates: tuple[float, ...]
    rate_texts: tuple[str, ...]
    seeds: tuple[int, ...]
    system_keys: tuple[str, ...]
    points: tuple[AggregatePoint, ...]
    metric_keys: tuple[str, ...] = tuple(
        spec.key for spec in METRIC_SPECS)


_TOP_KEYS = {
    "schema_version",
    "scenario",
    "grid",
    "slo_thresholds_ns",
    "execution",
    "code_revision_hashes",
    "system_config_contracts",
    "system_config_contracts_sha256",
    "pairing",
    "cells",
    "cells_sha256",
}
_SCENARIO_KEYS = {
    "scenario_id",
    "source_path",
    "source_sha256",
    "manifest",
    "manifest_sha256",
}
_GRID_KEYS = {
    "rates",
    "rate_texts",
    "rates_sha256",
    "seeds",
    "seeds_sha256",
    "system_keys",
    "system_keys_sha256",
    "cell_count",
}
_EXECUTION_KEYS = {
    "executor",
    "multiprocessing_start_method",
    "max_tasks_per_child",
    "one_isolated_cell_per_process",
    "workers",
    "detected_physical_cores",
    "simulation_backend",
    "astra_cycles_used",
}
_CODE_REVISION_KEYS = {
    "repository_git_head",
    "astra_sim_git_head",
    "python_implementation",
    "python_version",
    "serving_python_files",
    "serving_python_tree_sha256",
    "execution_code_sha256",
}
_PAIRING_KEYS = {
    "semantics",
    "measurement_identities_sha256",
    "schedule_pairs",
    "schedule_pairs_sha256",
}
_PAIR_KEYS = {
    "scenario_id",
    "seed",
    "session_rate",
    "rate_text",
    "offered_session_ids_sha256",
    "unit_draws_sha256",
    "session_count",
    "call_count",
    "call_specs_sha256",
    "schedule_sha256",
    "expected_call_identities_sha256",
    "expected_call_identity_set_sha256",
    "expected_session_ids_sha256",
    "schedule_pair_sha256",
    "system_keys",
    "result_schedule_sha256",
    "result_call_specs_sha256",
    "measurement_identities_sha256",
    "completion_call_set_sha256",
}
_PAIR_CONTRACT_KEYS = (
    "scenario_id",
    "seed",
    "session_rate",
    "rate_text",
    "offered_session_ids_sha256",
    "unit_draws_sha256",
    "session_count",
    "call_count",
    "call_specs_sha256",
    "schedule_sha256",
    "expected_call_identities_sha256",
    "expected_call_identity_set_sha256",
    "expected_session_ids_sha256",
)
_CELL_RECORD_KEYS = {
    "seed",
    "session_rate",
    "rate_text",
    "system_key",
    "relative_directory",
    "cell_contract_sha256",
    "schedule_pair_sha256",
    "result_contract",
    "artifacts",
}
_RESULT_CONTRACT_KEYS = {
    "cell_schema_version",
    "call_specs_sha256",
    "schedule_sha256",
    "measurement_identities_sha256",
    "completion_call_set_sha256",
    "completion_call_order_sha256",
    "request_count",
    "simulation_backend",
    "astra_cycles_used",
}
_COMPLETION_KEYS = {
    "schema_version",
    "status",
    "cell_contract",
    "cell_contract_sha256",
    "result_contract",
    "artifacts",
}
_CELL_CONTRACT_KEYS = {
    "schema_version",
    "scenario_id",
    "scenario_manifest_sha256",
    "seed",
    "session_rate",
    "rate_text",
    "system_key",
    "schedule_pair_sha256",
    "expected_call_specs_sha256",
    "expected_schedule_sha256",
    "measurement_identities_sha256",
    "system_config_contract_sha256",
    "execution_code_sha256",
    "thresholds_ns",
}
_CELL_JSON_KEYS = {
    "schema_version",
    "system_key",
    "session_rate",
    "simulation_contract",
    "frozen_workload",
    "measurement_roster",
    "summary",
    "full_drain",
    "bottleneck_report",
    "execution_observation",
    "requests",
}
_SUMMARY_KEYS = {
    "counts",
    "latency_distributions_ns",
    "request_kind_summaries",
    "slo",
    "offered_load_normalized_request_goodput",
    "offered_load_normalized_output_token_goodput",
    "observed_completion_span_throughput",
}
_COUNTS_KEYS = {
    "measurement_sessions",
    "measurement_calls",
    "first_calls",
    "resume_calls",
    "tpot_eligible_calls",
    "output_tokens",
}
_DISTRIBUTION_KEYS = {
    "count",
    "mean_ns",
    "p50_ns",
    "p90_ns",
    "p95_ns",
    "p99_ns",
    "percentile_method",
}
_SLO_KEYS = {
    "thresholds_ns",
    "first_ttft_pass_count",
    "first_ttft_pass_fraction",
    "resume_ttft_pass_count",
    "resume_ttft_pass_fraction",
    "ttft_pass_count",
    "ttft_pass_fraction",
    "tpot_pass_count",
    "tpot_pass_fraction_of_eligible",
    "all_slo_pass_count",
    "all_slo_pass_fraction",
    "all_slo_pass_output_tokens",
}


def _fail(message: str) -> None:
    raise ComparisonPlotInputError(message)


def _exact_keys(
        value: object,
        expected: set[str],
        context: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{context} must be an object")
    observed = set(value)
    if observed != expected:
        _fail(
            f"{context} schema mismatch: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ComparisonPlotInputError(
            f"cannot hash required file {path}: {exc}") from exc
    return digest.hexdigest()


def _strict_json(path: Path, context: str) -> dict[str, object]:
    try:
        return _strict_json_object(path)
    except ComparisonSweepError as exc:
        raise ComparisonPlotInputError(
            f"cannot validate {context}: {exc}") from exc


def _observed_artifact(
        path: Path,
        context: str,
) -> dict[str, object]:
    try:
        return _artifact_record(path)
    except ComparisonSweepError as exc:
        raise ComparisonPlotInputError(
            f"cannot validate {context}: {exc}") from exc


def _validate_csv(path: Path, expected_rows: int, context: str) -> None:
    try:
        _validate_csv_row_count(path, expected_rows)
    except ComparisonSweepError as exc:
        raise ComparisonPlotInputError(
            f"cannot validate {context}: {exc}") from exc


def _require_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_git_oid(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{context} must be a lowercase Git object ID")
    return value


def _require_integer(
        value: object,
        context: str,
        *,
        minimum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        _fail(f"{context} must be at least {minimum}")
    return value


def _require_float(
        value: object,
        context: str,
        *,
        positive: bool = False,
        nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{context} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        _fail(f"{context} must be a finite number")
    if positive and parsed <= 0:
        _fail(f"{context} must be positive")
    if nonnegative and parsed < 0:
        _fail(f"{context} must be nonnegative")
    return parsed


def _safe_relative_path(value: object, context: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"{context} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or any(
            part in ("", ".", "..") for part in relative.parts):
        _fail(f"{context} is not a safe relative path: {value!r}")
    return relative


def _regular_file_below(
        root: Path,
        relative: Path,
        context: str,
) -> Path:
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink() or not current.is_dir():
            _fail(f"{context} has a missing or symlinked parent: {current}")
    target = root / relative
    if target.is_symlink() or not target.is_file():
        _fail(f"{context} is not a regular file: {target}")
    try:
        target.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise ComparisonPlotInputError(
            f"{context} escapes its declared root: {target}") from exc
    return target


def _validate_file_record(
        record: object,
        *,
        repo_root: Path,
        context: str,
) -> Mapping[str, object]:
    value = _exact_keys(
        record,
        {"repo_relative_path", "content_sha256", "size_bytes"},
        context,
    )
    relative = _safe_relative_path(
        value["repo_relative_path"], f"{context}.repo_relative_path")
    target = _regular_file_below(repo_root, relative, context)
    expected_hash = _require_sha256(
        value["content_sha256"], f"{context}.content_sha256")
    expected_size = _require_integer(
        value["size_bytes"], f"{context}.size_bytes", minimum=0)
    try:
        size = target.stat().st_size
    except OSError as exc:
        raise ComparisonPlotInputError(
            f"cannot stat {context} file {target}: {exc}") from exc
    if size != expected_size or _sha256_file(target) != expected_hash:
        _fail(f"{context} live file differs from the sweep provenance")
    return value


def _validate_code_revision(
        value: object,
        repo_root: Path,
) -> Mapping[str, object]:
    code = _exact_keys(value, _CODE_REVISION_KEYS, "code_revision_hashes")
    source_files = code["serving_python_files"]
    if not isinstance(source_files, Mapping) or not source_files:
        _fail("code_revision_hashes.serving_python_files must be non-empty")
    normalized = {}
    for raw_path, raw_hash in source_files.items():
        if not isinstance(raw_path, str):
            _fail("serving source paths must be strings")
        relative = _safe_relative_path(raw_path, "serving source path")
        expected = _require_sha256(
            raw_hash, f"serving source hash for {raw_path}")
        normalized[raw_path] = expected
    repository_head = _require_git_oid(
        code["repository_git_head"],
        "code_revision_hashes.repository_git_head",
    )
    for raw_path, expected in normalized.items():
        relative = _safe_relative_path(raw_path, "serving source path")
        live_target = repo_root / relative
        if (
            not live_target.is_symlink()
            and live_target.is_file()
            and _sha256_file(live_target) == expected
        ):
            continue
        try:
            completed = subprocess.run(
                ["git", "show", f"{repository_head}:{raw_path}"],
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            raise ComparisonPlotInputError(
                f"cannot inspect recorded source revision: {exc}") from exc
        if (
            completed.returncode != 0
            or hashlib.sha256(completed.stdout).hexdigest() != expected
        ):
            _fail(
                f"recorded Git revision does not reproduce source hash: "
                f"{raw_path}"
            )
    if stable_json_sha256(normalized) != code["serving_python_tree_sha256"]:
        _fail("serving_python_tree_sha256 is inconsistent")
    execution_payload = {
        "python_implementation": code["python_implementation"],
        "python_version": code["python_version"],
        "serving_python_files": normalized,
    }
    if stable_json_sha256(execution_payload) != code["execution_code_sha256"]:
        _fail("execution_code_sha256 is inconsistent")
    if (
        not isinstance(code["python_implementation"], str)
        or not code["python_implementation"]
        or not isinstance(code["python_version"], str)
        or not code["python_version"]
    ):
        _fail("recorded Python runtime provenance is invalid")
    astra_head = code["astra_sim_git_head"]
    if astra_head is not None:
        _require_git_oid(
            astra_head, "code_revision_hashes.astra_sim_git_head")
        try:
            completed = subprocess.run(
                ["git", "cat-file", "-e", f"{astra_head}^{{commit}}"],
                cwd=repo_root / "astra-sim",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            raise ComparisonPlotInputError(
                f"cannot inspect recorded ASTRA-Sim revision: {exc}") from exc
        if completed.returncode != 0:
            _fail("recorded ASTRA-Sim Git revision is unavailable")
    return code


def _validate_grid(
        value: object,
) -> tuple[
        Mapping[str, object],
        tuple[float, ...],
        tuple[str, ...],
        tuple[int, ...],
        tuple[str, ...],
]:
    grid = _exact_keys(value, _GRID_KEYS, "grid")
    raw_rates = grid["rates"]
    raw_texts = grid["rate_texts"]
    raw_seeds = grid["seeds"]
    raw_systems = grid["system_keys"]
    if not all(isinstance(item, list) for item in (
            raw_rates, raw_texts, raw_seeds, raw_systems)):
        _fail("grid axes must be JSON arrays")
    if not raw_rates or not raw_seeds or not raw_systems:
        _fail("grid axes must be non-empty")
    rates = tuple(
        _require_float(rate, f"grid.rates[{index}]", positive=True)
        for index, rate in enumerate(raw_rates)
    )
    rate_texts = tuple(raw_texts)
    if (
        any(not isinstance(text, str) or not text for text in rate_texts)
        or rate_texts != tuple(_rate_text(rate) for rate in raw_rates)
        or len(set(rate_texts)) != len(rate_texts)
    ):
        _fail("grid rate_texts are not canonical and unique")
    seeds = tuple(
        _require_integer(seed, f"grid.seeds[{index}]", minimum=0)
        for index, seed in enumerate(raw_seeds)
    )
    systems = tuple(raw_systems)
    if (
        len(set(seeds)) != len(seeds)
        or len(set(systems)) != len(systems)
        or any(
            not isinstance(system, str) or system not in SYSTEM_KEYS
            for system in systems
        )
    ):
        _fail("grid seeds/systems must be unique and supported")
    for name, payload, digest in (
        ("rates", list(rate_texts), grid["rates_sha256"]),
        ("seeds", list(seeds), grid["seeds_sha256"]),
        ("system_keys", list(systems), grid["system_keys_sha256"]),
    ):
        _require_sha256(digest, f"grid.{name}_sha256")
        if stable_json_sha256(payload) != digest:
            _fail(f"grid.{name}_sha256 is inconsistent")
    expected_count = len(rates) * len(seeds) * len(systems)
    if _require_integer(
            grid["cell_count"], "grid.cell_count", minimum=1
    ) != expected_count:
        _fail("grid.cell_count does not equal the Cartesian-product size")
    return grid, rates, rate_texts, seeds, systems


def _validate_system_configs(
        value: object,
        *,
        systems: Sequence[str],
        aggregate_hash: object,
        repo_root: Path,
) -> Mapping[str, Mapping[str, object]]:
    if not isinstance(value, Mapping) or set(value) != set(systems):
        _fail("system_config_contracts do not match grid.system_keys")
    _require_sha256(
        aggregate_hash, "system_config_contracts_sha256")
    if stable_json_sha256(value) != aggregate_hash:
        _fail("system_config_contracts_sha256 is inconsistent")
    validated = {}
    for system in systems:
        context = f"system_config_contracts.{system}"
        contract = _exact_keys(
            value[system],
            {
                "system_key",
                "system_class",
                "tiering_policy",
                "hbf_layout",
                "model_config",
                "gpu_config",
                "hbf_config",
            },
            context,
        )
        if contract["system_key"] != system:
            _fail(f"{context}.system_key mismatch")
        _validate_file_record(
            contract["model_config"],
            repo_root=repo_root,
            context=f"{context}.model_config",
        )
        _validate_file_record(
            contract["gpu_config"],
            repo_root=repo_root,
            context=f"{context}.gpu_config",
        )
        hbf_record = contract["hbf_config"]
        if system in HBF_CONFIG_PATHS:
            _validate_file_record(
                hbf_record,
                repo_root=repo_root,
                context=f"{context}.hbf_config",
            )
        elif hbf_record is not None:
            _fail(f"{context}.hbf_config must be null")
        validated[system] = contract
    return validated


def _validate_pairing(
        value: object,
        *,
        scenario_id: str,
        rates: Sequence[float],
        rate_texts: Sequence[str],
        seeds: Sequence[int],
        systems: Sequence[str],
        measurement_sha256: str,
) -> Mapping[tuple[int, str], Mapping[str, object]]:
    pairing = _exact_keys(value, _PAIRING_KEYS, "pairing")
    if pairing["measurement_identities_sha256"] != measurement_sha256:
        _fail("pairing measurement roster hash is inconsistent")
    pairs = pairing["schedule_pairs"]
    if not isinstance(pairs, list):
        _fail("pairing.schedule_pairs must be an array")
    _require_sha256(
        pairing["schedule_pairs_sha256"],
        "pairing.schedule_pairs_sha256",
    )
    if stable_json_sha256(pairs) != pairing["schedule_pairs_sha256"]:
        _fail("pairing.schedule_pairs_sha256 is inconsistent")
    expected_coordinates = [
        (seed, rate, rate_text)
        for seed in seeds
        for rate, rate_text in zip(rates, rate_texts)
    ]
    if len(pairs) != len(expected_coordinates):
        _fail("pairing schedule count does not match the seed/rate grid")
    by_coordinate = {}
    for index, (raw, coordinate) in enumerate(
            zip(pairs, expected_coordinates)):
        pair = _exact_keys(raw, _PAIR_KEYS, f"schedule_pairs[{index}]")
        seed, rate, rate_text = coordinate
        if (
            pair["scenario_id"] != scenario_id
            or pair["seed"] != seed
            or pair["session_rate"] != rate
            or pair["rate_text"] != rate_text
            or pair["system_keys"] != list(systems)
            or pair["measurement_identities_sha256"]
            != measurement_sha256
        ):
            _fail(f"schedule_pairs[{index}] coordinate/roster mismatch")
        for count_key in ("session_count", "call_count"):
            _require_integer(
                pair[count_key],
                f"schedule_pairs[{index}].{count_key}",
                minimum=1,
            )
        for key in (
            "offered_session_ids_sha256",
            "unit_draws_sha256",
            "call_specs_sha256",
            "schedule_sha256",
            "expected_call_identities_sha256",
            "expected_call_identity_set_sha256",
            "expected_session_ids_sha256",
            "schedule_pair_sha256",
            "result_schedule_sha256",
            "result_call_specs_sha256",
            "measurement_identities_sha256",
            "completion_call_set_sha256",
        ):
            _require_sha256(pair[key], f"schedule_pairs[{index}].{key}")
        base_contract = {
            key: pair[key] for key in _PAIR_CONTRACT_KEYS
        }
        if stable_json_sha256(base_contract) != pair["schedule_pair_sha256"]:
            _fail(f"schedule_pairs[{index}] contract hash is inconsistent")
        if (
            pair["result_schedule_sha256"] != pair["schedule_sha256"]
            or pair["result_call_specs_sha256"]
            != pair["call_specs_sha256"]
            or pair["completion_call_set_sha256"]
            != pair["expected_call_identity_set_sha256"]
        ):
            _fail(f"schedule_pairs[{index}] result hashes diverge")
        by_coordinate[(seed, rate_text)] = pair
    return by_coordinate


def _nested(
        value: Mapping[str, object],
        path: Sequence[str],
        context: str,
) -> object:
    current: object = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            _fail(f"{context} is missing {'.'.join(path)}")
        current = current[key]
    return current


def _validate_distribution(
        value: object,
        *,
        expected_count: int,
        context: str,
) -> Mapping[str, object]:
    distribution = _exact_keys(
        value, _DISTRIBUTION_KEYS, context)
    if distribution["count"] != expected_count or expected_count < 0:
        _fail(f"{context}.count is inconsistent")
    if distribution["percentile_method"] != "inclusive_nearest_rank":
        _fail(f"{context} uses an unsupported percentile method")
    if expected_count == 0:
        for key in ("mean_ns", "p50_ns", "p90_ns", "p95_ns", "p99_ns"):
            if distribution[key] is not None:
                _fail(
                    f"{context}.{key} must be null for an empty cohort")
        return distribution
    previous = None
    for key in ("p50_ns", "p90_ns", "p95_ns", "p99_ns"):
        parsed = _require_float(
            distribution[key], f"{context}.{key}", positive=True)
        if previous is not None and parsed < previous:
            _fail(f"{context} percentiles are not monotone")
        previous = parsed
    _require_float(
        distribution["mean_ns"], f"{context}.mean_ns", positive=True)
    return distribution


def _validate_metric_summary(
        cell: Mapping[str, object],
        *,
        rate: float,
        thresholds: Mapping[str, object],
        context: str,
        metric_keys: Sequence[str],
) -> Mapping[str, float]:
    summary = _exact_keys(cell["summary"], _SUMMARY_KEYS, f"{context}.summary")
    counts = _exact_keys(
        summary["counts"], _COUNTS_KEYS, f"{context}.summary.counts")
    parsed_counts = {}
    for key in _COUNTS_KEYS:
        minimum = (
            1
            if key in {
                "measurement_sessions",
                "measurement_calls",
                "output_tokens",
            }
            else 0
        )
        parsed_counts[key] = _require_integer(
            counts[key], f"{context}.summary.counts.{key}", minimum=minimum)
    if (
        parsed_counts["first_calls"] + parsed_counts["resume_calls"]
        != parsed_counts["measurement_calls"]
        or parsed_counts["tpot_eligible_calls"]
        > parsed_counts["measurement_calls"]
    ):
        _fail(f"{context} measurement counts are inconsistent")

    distributions = summary["latency_distributions_ns"]
    if not isinstance(distributions, Mapping):
        _fail(f"{context}.summary.latency_distributions_ns must be an object")
    required_distributions = {
        "first_ttft": parsed_counts["first_calls"],
        "resume_ttft": parsed_counts["resume_calls"],
        "tpot_eligible": parsed_counts["tpot_eligible_calls"],
    }
    for name, count in required_distributions.items():
        if name not in distributions:
            _fail(f"{context} is missing latency distribution {name}")
        _validate_distribution(
            distributions[name],
            expected_count=count,
            context=f"{context}.summary.latency_distributions_ns.{name}",
        )

    slo = _exact_keys(summary["slo"], _SLO_KEYS, f"{context}.summary.slo")
    if slo["thresholds_ns"] != thresholds:
        _fail(f"{context} SLO thresholds differ from the sweep manifest")
    pass_count = _require_integer(
        slo["all_slo_pass_count"],
        f"{context}.summary.slo.all_slo_pass_count",
        minimum=0,
    )
    pass_tokens = _require_integer(
        slo["all_slo_pass_output_tokens"],
        f"{context}.summary.slo.all_slo_pass_output_tokens",
        minimum=0,
    )
    pass_fraction = _require_float(
        slo["all_slo_pass_fraction"],
        f"{context}.summary.slo.all_slo_pass_fraction",
        nonnegative=True,
    )
    if (
        pass_fraction > 1.0
        or pass_count > parsed_counts["measurement_calls"]
        or not math.isclose(
            pass_fraction,
            pass_count / parsed_counts["measurement_calls"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        _fail(f"{context} joint SLO pass fraction is inconsistent")

    request_goodput = _exact_keys(
        summary["offered_load_normalized_request_goodput"],
        {"label", "unit", "value", "formula"},
        f"{context}.summary.request_goodput",
    )
    output_goodput = _exact_keys(
        summary["offered_load_normalized_output_token_goodput"],
        {"label", "unit", "value", "formula"},
        f"{context}.summary.output_goodput",
    )
    observed = _exact_keys(
        summary["observed_completion_span_throughput"],
        {
            "label",
            "semantics",
            "completion_start_ns",
            "completion_end_ns",
            "completion_span_ns",
            "completion_event_count",
            "inter_completion_interval_count",
            "interval_output_tokens",
            "requests_per_second",
            "output_tokens_per_second",
            "zero_span_value",
        },
        f"{context}.summary.observed_throughput",
    )
    if (
        request_goodput["unit"] != "requests/s"
        or output_goodput["unit"] != "output tokens/s"
        or request_goodput["formula"] != (
            "session_rate * measured_calls / measured_sessions "
            "* all_SLO_pass_fraction"
        )
        or output_goodput["formula"] != (
            "session_rate * all_SLO_pass_output_tokens "
            "/ measured_sessions"
        )
        or observed["zero_span_value"] is not None
    ):
        _fail(f"{context} throughput semantics changed")

    expected_request_goodput = (
        rate * pass_count / parsed_counts["measurement_sessions"]
    )
    expected_output_goodput = (
        rate * pass_tokens / parsed_counts["measurement_sessions"]
    )
    actual_request_goodput = _require_float(
        request_goodput["value"],
        f"{context}.summary.request_goodput.value",
        nonnegative=True,
    )
    actual_output_goodput = _require_float(
        output_goodput["value"],
        f"{context}.summary.output_goodput.value",
        nonnegative=True,
    )
    if (
        not math.isclose(
            actual_request_goodput, expected_request_goodput,
            rel_tol=1e-12, abs_tol=1e-12,
        )
        or not math.isclose(
            actual_output_goodput, expected_output_goodput,
            rel_tol=1e-12, abs_tol=1e-12,
        )
    ):
        _fail(f"{context} offered-load-normalized goodput is inconsistent")

    span = _require_integer(
        observed["completion_span_ns"],
        f"{context}.summary.observed_throughput.completion_span_ns",
        minimum=1,
    )
    events = _require_integer(
        observed["completion_event_count"],
        f"{context}.summary.observed_throughput.completion_event_count",
        minimum=2,
    )
    intervals = _require_integer(
        observed["inter_completion_interval_count"],
        f"{context}.summary.observed_throughput.interval_count",
        minimum=1,
    )
    completion_start = _require_integer(
        observed["completion_start_ns"],
        f"{context}.summary.observed_throughput.completion_start_ns",
        minimum=0,
    )
    completion_end = _require_integer(
        observed["completion_end_ns"],
        f"{context}.summary.observed_throughput.completion_end_ns",
        minimum=1,
    )
    interval_output_tokens = _require_integer(
        observed["interval_output_tokens"],
        f"{context}.summary.observed_throughput.interval_output_tokens",
        minimum=0,
    )
    observed_output_rate = _require_float(
        observed["output_tokens_per_second"],
        f"{context}.summary.observed_throughput.output_tokens_per_second",
        nonnegative=True,
    )
    if (
        events != parsed_counts["measurement_calls"]
        or intervals != events - 1
        or completion_end - completion_start != span
    ):
        _fail(f"{context} observed throughput event span is inconsistent")
    observed_request_rate = _require_float(
        observed["requests_per_second"],
        f"{context}.summary.observed_throughput.requests_per_second",
        positive=True,
    )
    if not math.isclose(
        observed_request_rate,
        intervals * 1_000_000_000 / span,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        _fail(f"{context} observed request throughput formula changed")
    if not math.isclose(
        observed_output_rate,
        interval_output_tokens * 1_000_000_000 / span,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        _fail(f"{context} observed output throughput formula changed")

    values = {}
    for key in metric_keys:
        spec = METRIC_BY_KEY[key]
        raw = _nested(cell, spec.source_path, context)
        if raw is None:
            _fail(
                f"{context}.{spec.key} is unavailable because its "
                "measurement cohort is empty"
            )
        parsed = _require_float(
            raw,
            f"{context}.{'.'.join(spec.source_path)}",
            positive=spec.positive,
            nonnegative=not spec.positive,
        )
        scaled = parsed * spec.scale
        if spec.bounded_fraction and scaled > 1.0:
            _fail(f"{context}.{spec.key} must be within [0, 1]")
        values[spec.key] = scaled
    return values


def _validate_cell(
        *,
        root: Path,
        record: object,
        expected_coordinate: tuple[int, float, str, str],
        pair: Mapping[str, object],
        scenario: Mapping[str, object],
        thresholds: Mapping[str, object],
        execution_code_sha256: str,
        system_config: Mapping[str, object],
        metric_keys: Sequence[str],
) -> SeedCellMetrics:
    seed, rate, rate_text, system = expected_coordinate
    context = f"cell[{seed},{rate_text},{system}]"
    item = _exact_keys(record, _CELL_RECORD_KEYS, context)
    expected_relative = (
        Path("cells") / _rate_directory(rate_text)
        / f"seed_{seed}" / system
    )
    if (
        item["seed"] != seed
        or item["session_rate"] != rate
        or item["rate_text"] != rate_text
        or item["system_key"] != system
        or item["relative_directory"] != expected_relative.as_posix()
        or item["schedule_pair_sha256"] != pair["schedule_pair_sha256"]
    ):
        _fail(f"{context} coordinate or path mismatch")
    directory = root / expected_relative
    if directory.is_symlink() or not directory.is_dir():
        _fail(f"{context} directory is missing or symlinked")
    expected_children = {CELL_JSON, REQUEST_CSV, COMPLETION_JSON}
    try:
        children = {child.name for child in directory.iterdir()}
    except OSError as exc:
        raise ComparisonPlotInputError(
            f"cannot inspect {context}: {exc}") from exc
    if children != expected_children:
        _fail(f"{context} directory contents are incomplete or unexpected")

    artifacts = _exact_keys(
        item["artifacts"],
        {CELL_JSON, REQUEST_CSV, COMPLETION_JSON},
        f"{context}.artifacts",
    )
    for filename in (CELL_JSON, REQUEST_CSV, COMPLETION_JSON):
        expected_artifact = _exact_keys(
            artifacts[filename],
            {"sha256", "size_bytes"},
            f"{context}.artifacts.{filename}",
        )
        _require_sha256(
            expected_artifact["sha256"],
            f"{context}.artifacts.{filename}.sha256",
        )
        _require_integer(
            expected_artifact["size_bytes"],
            f"{context}.artifacts.{filename}.size_bytes",
            minimum=0,
        )
        if _observed_artifact(
                directory / filename,
                f"{context}.artifacts.{filename}",
        ) != expected_artifact:
            _fail(f"{context} artifact hash mismatch for {filename}")

    marker = _strict_json(
        directory / COMPLETION_JSON, f"{context}.completion")
    marker = _exact_keys(marker, _COMPLETION_KEYS, f"{context}.completion")
    marker_artifacts = _exact_keys(
        marker["artifacts"],
        {CELL_JSON, REQUEST_CSV},
        f"{context}.completion.artifacts",
    )
    if (
        marker["schema_version"] != COMPLETION_MARKER_SCHEMA_VERSION
        or marker["status"] != "complete"
        or marker_artifacts[CELL_JSON] != artifacts[CELL_JSON]
        or marker_artifacts[REQUEST_CSV] != artifacts[REQUEST_CSV]
    ):
        _fail(f"{context} completion marker mismatch")

    config_hash = stable_json_sha256(system_config)
    expected_cell_contract = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "scenario_id": scenario["scenario_id"],
        "scenario_manifest_sha256": scenario["manifest_sha256"],
        "seed": seed,
        "session_rate": rate,
        "rate_text": rate_text,
        "system_key": system,
        "schedule_pair_sha256": pair["schedule_pair_sha256"],
        "expected_call_specs_sha256": pair["call_specs_sha256"],
        "expected_schedule_sha256": pair["schedule_sha256"],
        "measurement_identities_sha256": (
            pair["measurement_identities_sha256"]),
        "system_config_contract_sha256": config_hash,
        "execution_code_sha256": execution_code_sha256,
        "thresholds_ns": dict(thresholds),
    }
    contract = _exact_keys(
        marker["cell_contract"],
        _CELL_CONTRACT_KEYS,
        f"{context}.completion.cell_contract",
    )
    contract_hash = _require_sha256(
        item["cell_contract_sha256"],
        f"{context}.cell_contract_sha256",
    )
    if (
        contract != expected_cell_contract
        or marker["cell_contract_sha256"] != contract_hash
        or stable_json_sha256(contract) != contract_hash
    ):
        _fail(f"{context} cell contract mismatch")

    result_contract = _exact_keys(
        item["result_contract"],
        _RESULT_CONTRACT_KEYS,
        f"{context}.result_contract",
    )
    if marker["result_contract"] != result_contract:
        _fail(f"{context} result contract differs from completion marker")
    for key in (
        "call_specs_sha256",
        "schedule_sha256",
        "measurement_identities_sha256",
        "completion_call_set_sha256",
        "completion_call_order_sha256",
    ):
        _require_sha256(
            result_contract[key], f"{context}.result_contract.{key}")
    if (
        result_contract["cell_schema_version"] != CELL_SCHEMA_VERSION
        or result_contract["call_specs_sha256"] != pair["call_specs_sha256"]
        or result_contract["schedule_sha256"] != pair["schedule_sha256"]
        or result_contract["measurement_identities_sha256"]
        != pair["measurement_identities_sha256"]
        or result_contract["completion_call_set_sha256"]
        != pair["expected_call_identity_set_sha256"]
        or result_contract["simulation_backend"] != SIMULATION_BACKEND
        or result_contract["astra_cycles_used"] is not ASTRA_CYCLES_USED
        or _require_integer(
            result_contract["request_count"],
            f"{context}.result_contract.request_count",
            minimum=1,
        ) != pair["call_count"]
    ):
        _fail(f"{context} result contract diverges from its paired schedule")

    cell = _strict_json(directory / CELL_JSON, f"{context}.cell_json")
    cell = _exact_keys(cell, _CELL_JSON_KEYS, f"{context}.cell_json")
    if (
        cell["schema_version"] != CELL_SCHEMA_VERSION
        or cell["system_key"] != system
        or cell["session_rate"] != rate
    ):
        _fail(f"{context} cell.json coordinate mismatch")
    frozen = cell["frozen_workload"]
    roster = cell["measurement_roster"]
    drain = cell["full_drain"]
    requests = cell["requests"]
    if (
        not isinstance(frozen, Mapping)
        or not isinstance(roster, Mapping)
        or not isinstance(drain, Mapping)
        or not isinstance(requests, list)
    ):
        _fail(f"{context} lacks workload/drain/request provenance")
    calls = drain.get("calls")
    if not isinstance(calls, Mapping):
        _fail(f"{context} lacks full_drain.calls")
    if (
        frozen.get("session_count") != pair["session_count"]
        or frozen.get("call_count") != pair["call_count"]
        or frozen.get("call_specs_sha256") != pair["call_specs_sha256"]
        or frozen.get("schedule_sha256") != pair["schedule_sha256"]
        or frozen.get("expected_call_identities_sha256")
        != pair["expected_call_identities_sha256"]
        or roster.get("ordered_identities_sha256")
        != pair["measurement_identities_sha256"]
        or calls.get("identity_count") != pair["call_count"]
        or calls.get("expected_set_sha256")
        != pair["expected_call_identity_set_sha256"]
        or calls.get("completion_set_sha256")
        != pair["expected_call_identity_set_sha256"]
        or calls.get("completion_order_sha256")
        != result_contract["completion_call_order_sha256"]
    ):
        _fail(f"{context} frozen workload/full-drain hashes mismatch")
    identities = []
    for row_index, row in enumerate(requests):
        if not isinstance(row, Mapping):
            _fail(f"{context}.requests[{row_index}] must be an object")
        identity = row.get("completion_identity")
        if (
            not isinstance(identity, str)
            or not identity
            or row.get("system_key") != system
        ):
            _fail(f"{context}.requests[{row_index}] identity/system mismatch")
        identities.append(identity)
    if (
        len(identities) != pair["call_count"]
        or len(set(identities)) != len(identities)
        or stable_json_sha256(sorted(identities))
        != pair["expected_call_identity_set_sha256"]
    ):
        _fail(f"{context} request rows differ from the full-drain call set")
    _validate_csv(
        directory / REQUEST_CSV,
        pair["call_count"],
        f"{context}.{REQUEST_CSV}",
    )

    simulation = cell["simulation_contract"]
    if not isinstance(simulation, Mapping):
        _fail(f"{context}.simulation_contract must be an object")
    backend = simulation.get("execution_backend")
    hardware = simulation.get("hardware")
    if (
        simulation.get("system_key") != system
        or not isinstance(backend, Mapping)
        or backend.get("name") != SIMULATION_BACKEND
        or backend.get("astra_cycles_used") is not ASTRA_CYCLES_USED
        or not isinstance(hardware, Mapping)
    ):
        _fail(f"{context} simulation backend provenance mismatch")
    for device in ("gpu", "hbf"):
        expected_device = system_config[f"{device}_config"]
        observed_device = hardware.get(device)
        if expected_device is None:
            if observed_device is not None:
                _fail(f"{context} unexpectedly reports {device} hardware")
        elif (
            not isinstance(observed_device, Mapping)
            or observed_device.get("repo_relative_path")
            != expected_device["repo_relative_path"]
            or observed_device.get("content_sha256")
            != expected_device["content_sha256"]
        ):
            _fail(f"{context} {device} hardware provenance mismatch")

    metrics = _validate_metric_summary(
        cell,
        rate=rate,
        thresholds=thresholds,
        context=context,
        metric_keys=metric_keys,
    )
    return SeedCellMetrics(
        seed=seed,
        session_rate=rate,
        rate_text=rate_text,
        system_key=system,
        schedule_pair_sha256=pair["schedule_pair_sha256"],
        values=metrics,
    )


def load_validated_sweep(
        sweep_root: str | Path,
        *,
        repo_root: str | Path,
        metric_keys: Optional[Sequence[str]] = None,
) -> ValidatedSweep:
    """Load every cell only after strict artifact/provenance validation.

    ``metric_keys`` may select a strict subset when a preregistered
    measurement window intentionally has an empty cohort for another metric
    (for example, a resume-only long-cold stress has no first-turn TTFT).
    Every summary count, empty-distribution sentinel, SLO formula, and
    throughput formula is still validated.
    """

    root = Path(sweep_root).expanduser().resolve()
    repository = Path(repo_root).expanduser().resolve()
    selected_metric_keys = (
        tuple(spec.key for spec in METRIC_SPECS)
        if metric_keys is None
        else tuple(metric_keys)
    )
    if (
        not selected_metric_keys
        or len(selected_metric_keys) != len(set(selected_metric_keys))
        or any(key not in METRIC_BY_KEY for key in selected_metric_keys)
    ):
        _fail(
            "metric_keys must be a non-empty unique subset of the "
            "validated metric catalog"
        )
    if root.is_symlink() or not root.is_dir():
        _fail(f"sweep root is missing or symlinked: {root}")
    if repository.is_symlink() or not repository.is_dir():
        _fail(f"repository root is missing or symlinked: {repository}")
    manifest_path = root / TOP_LEVEL_MANIFEST
    manifest = _strict_json(manifest_path, "top-level manifest")
    manifest = _exact_keys(manifest, _TOP_KEYS, "top-level manifest")
    if manifest["schema_version"] != SWEEP_SCHEMA_VERSION:
        _fail(
            f"unsupported sweep schema {manifest['schema_version']!r}; "
            f"required={SWEEP_SCHEMA_VERSION}"
        )
    scenario = _exact_keys(
        manifest["scenario"], _SCENARIO_KEYS, "scenario")
    if (
        not isinstance(scenario["scenario_id"], str)
        or not scenario["scenario_id"]
        or not isinstance(scenario["manifest"], Mapping)
    ):
        _fail("scenario ID/manifest is invalid")
    _require_sha256(scenario["source_sha256"], "scenario.source_sha256")
    _require_sha256(
        scenario["manifest_sha256"], "scenario.manifest_sha256")
    if stable_json_sha256(
            scenario["manifest"]) != scenario["manifest_sha256"]:
        _fail("scenario.manifest_sha256 is inconsistent")
    source_path = Path(str(scenario["source_path"])).expanduser()
    if source_path.is_symlink() or not source_path.is_file():
        _fail(f"scenario source trace is missing or symlinked: {source_path}")
    if _sha256_file(source_path) != scenario["source_sha256"]:
        _fail("live scenario trace differs from sweep provenance")

    grid, rates, rate_texts, seeds, systems = _validate_grid(
        manifest["grid"])
    thresholds = _exact_keys(
        manifest["slo_thresholds_ns"],
        {"first_ttft_ns", "resume_ttft_ns", "tpot_ns"},
        "slo_thresholds_ns",
    )
    for key, raw in thresholds.items():
        _require_integer(raw, f"slo_thresholds_ns.{key}", minimum=1)
    execution = _exact_keys(
        manifest["execution"], _EXECUTION_KEYS, "execution")
    if (
        execution["simulation_backend"] != SIMULATION_BACKEND
        or execution["astra_cycles_used"] is not ASTRA_CYCLES_USED
        or execution["one_isolated_cell_per_process"] is not True
    ):
        _fail("sweep execution backend/isolation provenance mismatch")
    _require_integer(execution["workers"], "execution.workers", minimum=1)
    _require_integer(
        execution["detected_physical_cores"],
        "execution.detected_physical_cores",
        minimum=1,
    )
    code = _validate_code_revision(
        manifest["code_revision_hashes"], repository)
    configs = _validate_system_configs(
        manifest["system_config_contracts"],
        systems=systems,
        aggregate_hash=manifest["system_config_contracts_sha256"],
        repo_root=repository,
    )
    pairing = _exact_keys(
        manifest["pairing"], _PAIRING_KEYS, "pairing")
    measurement_sha = _require_sha256(
        pairing["measurement_identities_sha256"],
        "pairing.measurement_identities_sha256",
    )
    pairs = _validate_pairing(
        pairing,
        scenario_id=scenario["scenario_id"],
        rates=rates,
        rate_texts=rate_texts,
        seeds=seeds,
        systems=systems,
        measurement_sha256=measurement_sha,
    )

    records = manifest["cells"]
    if not isinstance(records, list):
        _fail("cells must be an array")
    _require_sha256(manifest["cells_sha256"], "cells_sha256")
    if (
        len(records) != grid["cell_count"]
        or stable_json_sha256(records) != manifest["cells_sha256"]
    ):
        _fail("cells list count/hash is inconsistent")
    expected_coordinates = [
        (seed, rate, rate_text, system)
        for seed in seeds
        for rate, rate_text in zip(rates, rate_texts)
        for system in systems
    ]
    cells = []
    for record, coordinate in zip(records, expected_coordinates):
        seed, _, rate_text, system = coordinate
        cells.append(_validate_cell(
            root=root,
            record=record,
            expected_coordinate=coordinate,
            pair=pairs[(seed, rate_text)],
            scenario=scenario,
            thresholds=thresholds,
            execution_code_sha256=code["execution_code_sha256"],
            system_config=configs[system],
            metric_keys=selected_metric_keys,
        ))
    return ValidatedSweep(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        manifest=manifest,
        rates=rates,
        rate_texts=rate_texts,
        seeds=seeds,
        system_keys=systems,
        metric_keys=selected_metric_keys,
        cells=tuple(cells),
    )


def aggregate_validated_sweep(
        sweep: ValidatedSweep,
) -> ComparisonAggregate:
    """Aggregate the sweep's selected metrics across the complete seed grid."""

    by_coordinate = {}
    for cell in sweep.cells:
        key = (cell.session_rate, cell.system_key, cell.seed)
        if key in by_coordinate:
            _fail(f"duplicate seed cell coordinate {key!r}")
        by_coordinate[key] = cell
    expected = {
        (rate, system, seed)
        for rate in sweep.rates
        for system in sweep.system_keys
        for seed in sweep.seeds
    }
    if set(by_coordinate) != expected:
        missing = sorted(expected - set(by_coordinate))
        extra = sorted(set(by_coordinate) - expected)
        _fail(
            f"aggregate grid is incomplete: "
            f"missing={missing[:3]}, unexpected={extra[:3]}"
        )

    points = []
    for rate, rate_text in zip(sweep.rates, sweep.rate_texts):
        for system in sweep.system_keys:
            metrics = {}
            for key in sweep.metric_keys:
                spec = METRIC_BY_KEY[key]
                values_by_seed = {
                    seed: by_coordinate[
                        (rate, system, seed)
                    ].values[spec.key]
                    for seed in sweep.seeds
                }
                aggregate = aggregate_seed_values(values_by_seed)
                if set(aggregate.seed_ids) != set(sweep.seeds):
                    _fail("seed aggregate roster differs from sweep manifest")
                metrics[spec.key] = aggregate
            points.append(AggregatePoint(
                session_rate=rate,
                rate_text=rate_text,
                system_key=system,
                metrics=metrics,
            ))
    execution = sweep.manifest["execution"]
    return ComparisonAggregate(
        source_manifest_path=str(sweep.manifest_path),
        source_manifest_sha256=sweep.manifest_sha256,
        source_cells_sha256=str(sweep.manifest["cells_sha256"]),
        simulation_backend=str(execution["simulation_backend"]),
        astra_cycles_used=bool(execution["astra_cycles_used"]),
        rates=sweep.rates,
        rate_texts=sweep.rate_texts,
        seeds=sweep.seeds,
        system_keys=sweep.system_keys,
        metric_keys=sweep.metric_keys,
        points=tuple(points),
    )


def aggregate_to_dict(
        aggregate: ComparisonAggregate,
) -> dict[str, object]:
    """Serialize an aggregate with explicit estimand and source paths."""

    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "source": {
            "manifest_path": aggregate.source_manifest_path,
            "manifest_sha256": aggregate.source_manifest_sha256,
            "cells_sha256": aggregate.source_cells_sha256,
            "simulation_backend": aggregate.simulation_backend,
            "astra_cycles_used": aggregate.astra_cycles_used,
        },
        "statistics": {
            "replicate": "one_seed_level_cell_summary",
            "estimand": "arithmetic_mean_across_independent_arrival_seeds",
            "confidence_interval": (
                "two_sided_student_t_95_percent_across_seed_values"
            ),
            "request_rows_pooled": False,
        },
        "grid": {
            "rates": list(aggregate.rates),
            "rate_texts": list(aggregate.rate_texts),
            "seeds": list(aggregate.seeds),
            "system_keys": list(aggregate.system_keys),
            "metric_keys": list(aggregate.metric_keys),
        },
        "metrics": {
            spec.key: {
                "title": spec.title,
                "unit": spec.unit,
                "cell_source_path": ".".join(spec.source_path),
                "cell_value_scale": spec.scale,
            }
            for spec in METRIC_SPECS
            if spec.key in aggregate.metric_keys
        },
        "points": [
            {
                "session_rate": point.session_rate,
                "rate_text": point.rate_text,
                "system_key": point.system_key,
                "metrics": {
                    key: asdict(value)
                    for key, value in point.metrics.items()
                },
            }
            for point in aggregate.points
        ],
    }


def _atomic_csv(
        path: Path,
        write: Callable[[object], None],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False) as temporary:
            temporary_name = temporary.name
            write(temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return path


def write_statistics_csv(
        path: str | Path,
        aggregate: ComparisonAggregate,
) -> Path:
    """Write every seed sample and Student-t interval to a flat table."""

    target = Path(path)
    fields = (
        "session_rate",
        "rate_text",
        "system_key",
        "metric",
        "unit",
        "seed_count",
        "seed_ids",
        "seed_values",
        "mean",
        "sample_stddev",
        "ci95_half_width",
        "ci95_lower",
        "ci95_upper",
        "ci_method",
    )

    def write(handle: object) -> None:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in aggregate.points:
            for key in aggregate.metric_keys:
                spec = METRIC_BY_KEY[key]
                value = point.metrics[spec.key]
                writer.writerow({
                    "session_rate": point.session_rate,
                    "rate_text": point.rate_text,
                    "system_key": point.system_key,
                    "metric": spec.key,
                    "unit": spec.unit,
                    "seed_count": len(value.seed_ids),
                    "seed_ids": ";".join(map(str, value.seed_ids)),
                    "seed_values": ";".join(
                        format(sample, ".17g")
                        for sample in value.values
                    ),
                    "mean": format(value.mean, ".17g"),
                    "sample_stddev": (
                        ""
                        if value.sample_stddev is None
                        else format(value.sample_stddev, ".17g")
                    ),
                    "ci95_half_width": (
                        ""
                        if value.ci95_half_width is None
                        else format(value.ci95_half_width, ".17g")
                    ),
                    "ci95_lower": (
                        ""
                        if value.ci95_lower is None
                        else format(value.ci95_lower, ".17g")
                    ),
                    "ci95_upper": (
                        ""
                        if value.ci95_upper is None
                        else format(value.ci95_upper, ".17g")
                    ),
                    "ci_method": value.ci_method,
                })

    return _atomic_csv(target, write)


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, RuntimeError) as exc:
        raise ComparisonPlotRenderError(
            f"Matplotlib is required to render comparison figures: {exc}"
        ) from exc
    return plt


def _point_index(
        aggregate: ComparisonAggregate,
) -> Mapping[tuple[str, float], AggregatePoint]:
    index = {
        (point.system_key, point.session_rate): point
        for point in aggregate.points
    }
    if len(index) != len(aggregate.points):
        raise ComparisonPlotRenderError(
            "aggregate contains duplicate rate/system points")
    return index


def _plot_metric(
        axis: object,
        *,
        aggregate: ComparisonAggregate,
        index: Mapping[tuple[str, float], AggregatePoint],
        systems: Sequence[str],
        metric_key: str,
) -> None:
    spec = METRIC_BY_KEY[metric_key]
    for system in systems:
        points = [
            index[(system, rate)].metrics[metric_key]
            for rate in aggregate.rates
        ]
        means = [point.mean for point in points]
        lower_errors = []
        upper_errors = []
        for point in points:
            if point.ci95_lower is None or point.ci95_upper is None:
                lower_errors.append(0.0)
                upper_errors.append(0.0)
                continue
            lower_endpoint = point.ci95_lower
            upper_endpoint = point.ci95_upper
            if spec.bounded_fraction:
                lower_endpoint = max(0.0, lower_endpoint)
                upper_endpoint = min(1.0, upper_endpoint)
            if spec.log_scale:
                lower_endpoint = max(
                    point.mean * 1e-6, lower_endpoint)
            lower_errors.append(point.mean - lower_endpoint)
            upper_errors.append(upper_endpoint - point.mean)
        axis.errorbar(
            aggregate.rates,
            means,
            yerr=(lower_errors, upper_errors),
            linewidth=2.4,
            markersize=9,
            capsize=5,
            label=SYSTEM_LABELS[system],
            **SYSTEM_STYLES[system],
        )
    axis.set_title(spec.title)
    axis.set_ylabel(spec.y_label)
    axis.grid(True, alpha=0.3)
    if spec.log_scale:
        axis.set_yscale("log")
    if spec.bounded_fraction:
        axis.set_ylim(-0.03, 1.03)


def _save_figure_atomic(
        figure: object,
        path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
                prefix=f".{path.stem}.",
                suffix=".png",
                dir=path.parent,
                delete=False) as temporary:
            temporary_name = temporary.name
        figure.savefig(
            temporary_name,
            dpi=PNG_DPI,
            bbox_inches="tight",
            metadata={
                "Title": path.stem,
                "Software": "LLMServingSim hbf_comparison_plots",
            },
        )
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return path


def _render_metric_group(
        aggregate: ComparisonAggregate,
        *,
        systems: Sequence[str],
        metric_keys: Sequence[str],
        output_path: Path,
        figure_size: tuple[int, int],
        figure_title: str,
) -> Path:
    missing = [system for system in systems
               if system not in aggregate.system_keys]
    if missing:
        raise ComparisonPlotRenderError(
            f"aggregate lacks required systems: {missing}")
    missing_metrics = [
        metric_key
        for metric_key in metric_keys
        if metric_key not in aggregate.metric_keys
    ]
    if missing_metrics:
        raise ComparisonPlotRenderError(
            f"aggregate lacks required metrics: {missing_metrics}")
    plt = _load_matplotlib()
    index = _point_index(aggregate)
    with plt.rc_context(MATPLOTLIB_RC):
        figure, axes = plt.subplots(
            len(metric_keys),
            1,
            figsize=figure_size,
            sharex=True,
            constrained_layout=True,
        )
        if len(metric_keys) == 1:
            axes = [axes]
        for axis, metric_key in zip(axes, metric_keys):
            _plot_metric(
                axis,
                aggregate=aggregate,
                index=index,
                systems=systems,
                metric_key=metric_key,
            )
        axes[-1].set_xlabel("Offered session rate (sessions/s)")
        axes[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.42),
            ncol=2,
            frameon=False,
        )
        figure.suptitle(figure_title, y=1.015)
        try:
            return _save_figure_atomic(figure, output_path)
        finally:
            plt.close(figure)


def render_figures(
        aggregate: ComparisonAggregate,
        output_dir: str | Path,
) -> tuple[Path, ...]:
    """Render main comparison and HBF-layout appendix figures."""

    output = Path(output_dir)
    return (
        _render_metric_group(
            aggregate,
            systems=MAIN_SYSTEMS,
            metric_keys=LATENCY_METRICS,
            output_path=output / "balanced_main_latency.png",
            figure_size=LATENCY_FIGSIZE,
            figure_title="Balanced TraceLab: latency",
        ),
        _render_metric_group(
            aggregate,
            systems=MAIN_SYSTEMS,
            metric_keys=THROUGHPUT_METRICS,
            output_path=output / "balanced_main_slo_goodput.png",
            figure_size=THROUGHPUT_FIGSIZE,
            figure_title="Balanced TraceLab: SLO and throughput",
        ),
        _render_metric_group(
            aggregate,
            systems=HBF_LAYOUT_SYSTEMS,
            metric_keys=LATENCY_METRICS,
            output_path=output / "balanced_hbf_layout_latency.png",
            figure_size=LATENCY_FIGSIZE,
            figure_title="HBF layout appendix: latency",
        ),
        _render_metric_group(
            aggregate,
            systems=HBF_LAYOUT_SYSTEMS,
            metric_keys=THROUGHPUT_METRICS,
            output_path=output / "balanced_hbf_layout_slo_goodput.png",
            figure_size=THROUGHPUT_FIGSIZE,
            figure_title="HBF layout appendix: SLO and throughput",
        ),
    )


def write_aggregate_artifacts(
        aggregate: ComparisonAggregate,
        output_dir: str | Path,
        *,
        render: bool = True,
) -> Mapping[str, object]:
    """Write auditable JSON/CSV statistics and optional PNG figures."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aggregate_path = write_json_atomic(
        output / "balanced_comparison_aggregate.json",
        aggregate_to_dict(aggregate),
    )
    statistics_path = write_statistics_csv(
        output / "balanced_comparison_seed_statistics.csv",
        aggregate,
    )
    figure_paths = render_figures(aggregate, output) if render else ()
    artifacts = {
        "aggregate_json": {
            "path": str(aggregate_path),
            **_artifact_record(aggregate_path),
        },
        "statistics_csv": {
            "path": str(statistics_path),
            **_artifact_record(statistics_path),
        },
        "figures": [
            {
                "path": str(path),
                **_artifact_record(path),
            }
            for path in figure_paths
        ],
    }
    manifest_path = write_json_atomic(
        output / "plot_manifest.json",
        {
            "schema_version": AGGREGATE_SCHEMA_VERSION,
            "source_manifest_path": aggregate.source_manifest_path,
            "source_manifest_sha256": aggregate.source_manifest_sha256,
            "source_cells_sha256": aggregate.source_cells_sha256,
            "simulation_backend": aggregate.simulation_backend,
            "astra_cycles_used": aggregate.astra_cycles_used,
            "font_size": FONT_SIZE,
            "figure_width_inches": 12,
            "latency_y_scale": "log",
            "confidence_interval": "student_t_95_across_seed_cells",
            "artifacts": artifacts,
        },
    )
    return {
        **artifacts,
        "plot_manifest": {
            "path": str(manifest_path),
            **_artifact_record(manifest_path),
        },
    }


def _parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate, aggregate, and plot a completed balanced "
            "HBF comparison sweep."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=(
            repo_root / "results" / "wakekv_hbf"
            / "balanced-comparison-schema1-20260723"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repo_root / "results" / "wakekv_hbf"
            / "balanced-comparison-schema1-20260723" / "plots"
        ),
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="write validated aggregate JSON/CSV without importing Matplotlib",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    validated = load_validated_sweep(
        args.sweep_root,
        repo_root=args.repo_root,
    )
    aggregate = aggregate_validated_sweep(validated)
    artifacts = write_aggregate_artifacts(
        aggregate,
        args.output_dir,
        render=not args.no_render,
    )
    print(json.dumps(
        {
            "source_manifest_sha256": aggregate.source_manifest_sha256,
            "seed_count": len(aggregate.seeds),
            "rate_count": len(aggregate.rates),
            "system_count": len(aggregate.system_keys),
            "point_count": len(aggregate.points),
            "simulation_backend": aggregate.simulation_backend,
            "astra_cycles_used": aggregate.astra_cycles_used,
            "artifacts": artifacts,
        },
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

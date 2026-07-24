"""Resumable live ASTRA-Sim sweep for the tiering/HBF comparison.

This runner intentionally does not use ``hbf_comparison_sweep``: that older
driver is a Python analytical discrete-event model.  Every cell launched here
is a real ``python -m serving`` process and therefore exercises the normal
LLMServingSim scheduler, Chakra conversion, and ASTRA-Sim controller.

One TraceLab schedule is materialized per ``(seed, rate)`` and reused byte for
byte by every system.  A cell is considered complete only after its process
exits successfully, every expected request appears exactly once in the native
CSV, all numeric results are finite, and the required runtime reports parse as
strict JSON.

Campaign identity deliberately hashes a bounded implementation surface:
every ``serving/**/*.py`` file, the Chakra LLM converter, and the selected
congestion-aware ASTRA executable.  Runtime configs and the TraceLab source are
hashed separately.  This avoids hashing an entire checkout or build tree while
ensuring that simulator or executable changes cannot silently reuse old cells.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Iterable, Mapping, Sequence

from .core.live_comparison_metrics import (
    DEFAULT_TPOT_SLO_NS,
    DEFAULT_TTFT_SLO_NS,
    LiveComparisonMetricsError,
    compute_live_comparison_metrics,
    expected_request_identities,
    materialize_scheduled_sessions,
    parse_serving_requests_csv,
)
from .core.tracelab_comparison_scenarios import (
    load_balanced_causal_prefix_scenario,
)


SCHEMA_VERSION = 2
NETWORK_BACKEND = "analytical-congestion-aware"
LATENCY_MODEL = "h100-qwen3-tp4-kernel-calibrated"

DEFAULT_TRACE = Path(
    os.environ.get(
        "LLMSIM_DATA",
        str(Path.home() / "llmsim-data"),
    )
) / "tracelab-schema3-sps0.2-final.jsonl"
DEFAULT_OUTPUT_ROOT = Path("results/live_astra_hbf_comparison")
DEFAULT_INPUTS_ROOT = Path("/dev/shm/llmsim-live-astra-comparison")
DEFAULT_SCENARIO_FACTORY = "balanced"
DEFAULT_LOG_INTERVAL_SECONDS = 60.0

PILOT_RATES = (0.02, 0.1, 0.5)
PILOT_SEEDS = (101,)
FULL_RATES = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
FULL_SEEDS = tuple(range(101, 113))

DUAL_CLUSTER = Path(
    "configs/cluster/dual_node_qwen3_1m_pd_p4d4_h100.json")
SINGLE_CLUSTER = Path(
    "configs/cluster/single_node_qwen3_1m_pd_p4d4_h100.json")
TIERED_CONFIG = Path(
    "configs/agentic_kv/qwen3_1m_p4d4/tiered_fullprompt.json")
ORACLE_CONFIG = Path(
    "configs/agentic_kv/qwen3_1m_p4d4/"
    "infinite_hbm_oracle_fullprompt.json")
HBF_CONFIG = Path("configs/wakekv_hbf/full_model_8card_server.json")

_CAMPAIGN_SOURCE_ROOTS = (Path("serving"),)
_CAMPAIGN_SOURCE_FILES = (
    Path(
        "astra-sim/extern/graph_frontend/chakra/src/"
        "converter/llm_converter.py"
    ),
)
_CAMPAIGN_ASTRA_BINARY = Path(
    "astra-sim/build/astra_analytical/build/"
    "AstraCongestion/bin/AstraCongestion"
)
_REQUIRED_RESULT_ARTIFACTS = (
    "requests",
    "session_report",
    "runtime_report",
    "stdout",
    "stderr",
)

_BOTTLENECK_TERMS = (
    "admission",
    "bandwidth",
    "busy",
    "capacity",
    "fallback",
    "hbf",
    "hit",
    "lpddr",
    "miss",
    "occup",
    "pcie",
    "queue",
    "rdma",
    "recompute",
    "ssd",
    "stall",
    "transfer",
    "util",
)


class LiveAstraSweepError(RuntimeError):
    """Raised when a campaign or cell violates its fail-closed contract."""


@dataclass(frozen=True)
class SystemSpec:
    """One physical comparison system and its serving-specific arguments."""

    key: str
    cluster_config: Path
    runtime_kind: str
    policy_config: Path
    layout: str | None = None

    def __post_init__(self) -> None:
        if self.runtime_kind not in {"agentic_kv", "oracle", "full_model_hbf"}:
            raise ValueError(f"unsupported runtime kind {self.runtime_kind!r}")
        if self.runtime_kind == "full_model_hbf":
            if self.layout not in {"tp4", "tp8", "tp8_context"}:
                raise ValueError("full-model HBF requires a supported layout")
        elif self.layout is not None:
            raise ValueError("only full-model HBF systems have a layout")


SYSTEMS = {
    "ssd_tiering": SystemSpec(
        key="ssd_tiering",
        cluster_config=DUAL_CLUSTER,
        runtime_kind="agentic_kv",
        policy_config=TIERED_CONFIG,
    ),
    "oracle": SystemSpec(
        key="oracle",
        cluster_config=DUAL_CLUSTER,
        runtime_kind="oracle",
        policy_config=ORACLE_CONFIG,
    ),
    "hbf_tp4": SystemSpec(
        key="hbf_tp4",
        cluster_config=SINGLE_CLUSTER,
        runtime_kind="full_model_hbf",
        policy_config=HBF_CONFIG,
        layout="tp4",
    ),
    "hbf_tp8": SystemSpec(
        key="hbf_tp8",
        cluster_config=SINGLE_CLUSTER,
        runtime_kind="full_model_hbf",
        policy_config=HBF_CONFIG,
        layout="tp8",
    ),
    "hbf_tp8_context": SystemSpec(
        key="hbf_tp8_context",
        cluster_config=SINGLE_CLUSTER,
        runtime_kind="full_model_hbf",
        policy_config=HBF_CONFIG,
        layout="tp8_context",
    ),
}
DEFAULT_SYSTEMS = tuple(SYSTEMS)


@dataclass(frozen=True)
class Cell:
    """Fully resolved live process cell."""

    cell_id: str
    system: SystemSpec
    seed: int
    rate: float
    workload_path: Path
    workload_sha256: str
    cell_dir: Path
    inputs_dir: Path
    request_count: int
    session_count: int
    last_external_guard_offer_ns: int | None = None
    expected_measurement_resume_count: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.last_external_guard_offer_ns,
            self.expected_measurement_resume_count,
        )
        if (values[0] is None) != (values[1] is None):
            raise ValueError(
                "runtime guard cutoff and expected resume count must "
                "be provided together")
        if values[0] is not None:
            if type(values[0]) is not int or values[0] < 0:
                raise ValueError(
                    "last_external_guard_offer_ns must be non-negative")
            if type(values[1]) is not int or values[1] <= 0:
                raise ValueError(
                    "expected_measurement_resume_count must be positive")

    @property
    def requests_csv(self) -> Path:
        return self.cell_dir / "requests.csv"

    @property
    def session_report(self) -> Path:
        return self.cell_dir / "session_metrics.json"

    @property
    def runtime_report(self) -> Path:
        if self.system.runtime_kind == "full_model_hbf":
            return self.cell_dir / "hbf_runtime.json"
        return self.cell_dir / "agentic_kv.json"

    @property
    def result_path(self) -> Path:
        return self.cell_dir / "result.json"


def _cell_runtime_guard_contract(
    cell: Cell,
) -> dict[str, object] | None:
    if cell.last_external_guard_offer_ns is None:
        return None
    assert cell.expected_measurement_resume_count is not None
    return {
        "seed": cell.seed,
        "offered_session_rate_per_second": cell.rate,
        "last_external_guard_offer_ns": (
            cell.last_external_guard_offer_ns),
        "expected_measurement_resume_count": (
            cell.expected_measurement_resume_count),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _strict_json(path: Path) -> Mapping[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise LiveAstraSweepError(
            f"cannot parse strict JSON report {path}") from exc
    if not isinstance(value, dict):
        raise LiveAstraSweepError(f"JSON report {path} is not an object")
    _require_finite_tree(value, str(path))
    return value


def _require_finite_tree(value: object, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise LiveAstraSweepError(f"{name} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite_tree(child, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_tree(child, f"{name}[{index}]")


def _parse_csv_list(raw: str, *, converter, name: str) -> tuple:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = converter(item)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                f"{name} contains invalid value {item!r}") from exc
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return tuple(values)


def _validate_log_interval_seconds(value: float) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise LiveAstraSweepError(
            "log_interval_seconds must be positive and finite") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise LiveAstraSweepError(
            "log_interval_seconds must be positive and finite")
    return normalized


def _positive_finite_log_interval(raw: str) -> float:
    try:
        return _validate_log_interval_seconds(float(raw))
    except (TypeError, ValueError, LiveAstraSweepError) as exc:
        raise argparse.ArgumentTypeError(
            "--log-interval must be positive and finite") from exc


def _validate_rates(rates: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(rate) for rate in rates)
    if (
        not normalized
        or len(normalized) != len(set(normalized))
        or any(not math.isfinite(rate) or rate <= 0.0 or rate > 5.0
               for rate in normalized)
    ):
        raise LiveAstraSweepError(
            "rates must be unique, finite, in (0, 5]")
    return tuple(sorted(normalized))


def _validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if (
        not normalized
        or len(normalized) != len(set(normalized))
        or any(seed < 0 for seed in normalized)
    ):
        raise LiveAstraSweepError(
            "seeds must be unique non-negative integers")
    return tuple(sorted(normalized))


def _rate_tag(rate: float) -> str:
    return format(rate, ".12g").replace(".", "p")


def build_serving_command(
    *,
    repo_root: Path,
    python_executable: Path,
    cell: Cell,
    log_interval_seconds: float = DEFAULT_LOG_INTERVAL_SECONDS,
) -> tuple[str, ...]:
    """Build the exact live serving command for a cell."""

    log_interval_seconds = _validate_log_interval_seconds(
        log_interval_seconds)
    command = [
        str(python_executable),
        "-m",
        "serving",
        "--cluster-config",
        str((repo_root / cell.system.cluster_config).resolve()),
        "--dataset",
        str(cell.workload_path.resolve()),
        "--num-reqs",
        str(cell.session_count),
        "--network-backend",
        NETWORK_BACKEND,
        "--latency-model",
        LATENCY_MODEL,
        "--no-enable-prefix-caching",
        "--request-routing-policy",
        "RR",
        "--run-id",
        cell.cell_id,
        "--inputs-root",
        str(cell.inputs_dir.resolve()),
        "--output",
        str(cell.requests_csv.resolve()),
        "--session-metrics",
        str(cell.session_report.resolve()),
        "--log-interval",
        str(log_interval_seconds),
        "--log-level",
        "WARNING",
    ]
    policy_path = str((repo_root / cell.system.policy_config).resolve())
    if cell.system.runtime_kind in {"agentic_kv", "oracle"}:
        command.extend([
            "--agentic-kv-config",
            policy_path,
            "--agentic-kv-metrics",
            str(cell.runtime_report.resolve()),
        ])
        if cell.system.runtime_kind == "oracle":
            command.append("--strict-infinite-hbm-oracle")
    else:
        command.extend([
            "--full-model-hbf-config",
            policy_path,
            "--full-model-hbf-layout",
            str(cell.system.layout),
            "--full-model-hbf-metrics",
            str(cell.runtime_report.resolve()),
        ])
    return tuple(command)


def _runtime_environment(repo_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    chakra_build = (
        repo_root
        / "astra-sim/extern/graph_frontend/chakra/build/lib"
    )
    chakra_source = (
        repo_root
        / "astra-sim/extern/graph_frontend/chakra"
    )
    additions = os.pathsep.join((str(chakra_build), str(chakra_source)))
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        additions if not current else additions + os.pathsep + current)
    return environment


def _extract_bottlenecks(
    reports: Mapping[str, Mapping[str, object]],
    *,
    limit: int = 512,
) -> dict[str, object]:
    selected: dict[str, object] = {}

    def visit(prefix: str, value: object) -> None:
        if len(selected) >= limit:
            return
        if isinstance(value, Mapping):
            for key in sorted(value, key=str):
                visit(f"{prefix}.{key}" if prefix else str(key), value[key])
        elif isinstance(value, (list, tuple)):
            if len(value) <= 16 and all(
                    isinstance(item, (str, int, float, bool, type(None)))
                    for item in value):
                if any(term in prefix.lower() for term in _BOTTLENECK_TERMS):
                    selected[prefix] = list(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            if any(term in prefix.lower() for term in _BOTTLENECK_TERMS):
                selected[prefix] = value

    for report_name, report in sorted(reports.items()):
        visit(report_name, report)
    if len(selected) >= limit:
        selected["_truncated"] = True
    return selected


def _artifact(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise LiveAstraSweepError(f"required artifact is missing: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _required_artifact_paths(cell: Cell) -> dict[str, Path]:
    return {
        "requests": cell.requests_csv,
        "session_report": cell.session_report,
        "runtime_report": cell.runtime_report,
        "stdout": cell.cell_dir / "stdout.log",
        "stderr": cell.cell_dir / "stderr.log",
    }


def _artifact_record_matches(
    record: object,
    expected_path: Path,
) -> bool:
    if not isinstance(record, Mapping):
        return False
    recorded_path = record.get("path")
    recorded_bytes = record.get("bytes")
    recorded_sha256 = record.get("sha256")
    if (
        not isinstance(recorded_path, str)
        or not recorded_path
        or type(recorded_bytes) is not int
        or recorded_bytes < 0
        or not isinstance(recorded_sha256, str)
        or len(recorded_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in recorded_sha256
        )
    ):
        return False
    try:
        if Path(recorded_path).resolve() != expected_path.resolve():
            return False
        if not expected_path.is_file():
            return False
        stat = expected_path.stat()
        return (
            stat.st_size == recorded_bytes
            and _sha256_file(expected_path) == recorded_sha256
        )
    except OSError:
        return False


def _campaign_implementation_identity(
    repo_root: Path,
) -> dict[str, object]:
    """Hash the deterministic, bounded implementation surface for a run.

    The recursive portion is restricted to Python sources below ``serving/``.
    Chakra's LLM converter is the only additional source file, and the
    congestion-aware executable is hashed by content through its normal
    launcher path.  Generated ASTRA inputs, unrelated build products, tests,
    results, and the rest of the checkout are intentionally outside the set.
    """

    relative_sources: set[Path] = set()
    for relative_root in _CAMPAIGN_SOURCE_ROOTS:
        source_root = repo_root / relative_root
        if not source_root.is_dir():
            raise LiveAstraSweepError(
                f"campaign source root is missing: {source_root}")
        relative_sources.update(
            path.relative_to(repo_root)
            for path in source_root.rglob("*.py")
            if path.is_file()
        )
    relative_sources.update(_CAMPAIGN_SOURCE_FILES)
    relative_paths = tuple(sorted(
        relative_sources,
        key=lambda path: path.as_posix(),
    ))
    if not relative_paths:
        raise LiveAstraSweepError("campaign source file set is empty")

    source_files: dict[str, object] = {}
    for relative in relative_paths:
        resolved = repo_root / relative
        if not resolved.is_file():
            raise LiveAstraSweepError(
                f"campaign source file is missing: {resolved}")
        source_files[relative.as_posix()] = {
            "bytes": resolved.stat().st_size,
            "sha256": _sha256_file(resolved),
        }

    astra_binary = repo_root / _CAMPAIGN_ASTRA_BINARY
    if not astra_binary.is_file():
        raise LiveAstraSweepError(
            f"campaign ASTRA binary is missing: {astra_binary}")
    return {
        "source_scope": {
            "recursive_python_roots": [
                path.as_posix() for path in _CAMPAIGN_SOURCE_ROOTS
            ],
            "explicit_source_files": [
                path.as_posix() for path in _CAMPAIGN_SOURCE_FILES
            ],
        },
        "source_files": source_files,
        "astra_binary": {
            "path": _CAMPAIGN_ASTRA_BINARY.as_posix(),
            "bytes": astra_binary.stat().st_size,
            "sha256": _sha256_file(astra_binary),
        },
    }


def _safe_remove_cell_inputs(cell_inputs: Path, campaign_inputs: Path) -> None:
    inputs = cell_inputs.resolve()
    root = campaign_inputs.resolve()
    if inputs.name != "inputs" or root not in inputs.parents:
        raise LiveAstraSweepError(
            f"refusing unsafe inputs cleanup: {inputs}")
    cell_root = inputs.parent
    if cell_root.parent != root:
        raise LiveAstraSweepError(
            f"refusing non-cell inputs cleanup: {cell_root}")
    if cell_root.exists():
        shutil.rmtree(cell_root)


def _run_cell(
    *,
    repo_root: Path,
    python_executable: Path,
    cell: Cell,
    scheduled_sessions: tuple,
    measurement_session_ids: tuple[str, ...],
    ttft_slo_ns: int,
    tpot_slo_ns: int,
    campaign_inputs: Path,
    timeout_seconds: float | None,
    keep_failed_inputs: bool,
    log_interval_seconds: float,
) -> dict[str, object]:
    log_interval_seconds = _validate_log_interval_seconds(
        log_interval_seconds)
    cell.cell_dir.mkdir(parents=True, exist_ok=True)
    cell.inputs_dir.mkdir(parents=True, exist_ok=True)
    command = build_serving_command(
        repo_root=repo_root,
        python_executable=python_executable,
        cell=cell,
        log_interval_seconds=log_interval_seconds,
    )
    command_record = {
        "schema_version": SCHEMA_VERSION,
        "cell_id": cell.cell_id,
        "command": list(command),
        "cwd": str(repo_root),
        "workload_sha256": cell.workload_sha256,
        "log_interval_seconds": log_interval_seconds,
    }
    _atomic_json(cell.cell_dir / "command.json", command_record)

    stdout_path = cell.cell_dir / "stdout.log"
    stderr_path = cell.cell_dir / "stderr.log"
    started = time.time()
    try:
        try:
            with (
                stdout_path.open("wb") as stdout,
                stderr_path.open("wb") as stderr,
            ):
                completed = subprocess.run(
                    command,
                    cwd=repo_root,
                    env=_runtime_environment(repo_root),
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    timeout=timeout_seconds,
                )
        except subprocess.TimeoutExpired as exc:
            raise LiveAstraSweepError(
                f"{cell.cell_id} exceeded timeout "
                f"{timeout_seconds}s") from exc
        elapsed = time.time() - started
        if completed.returncode != 0:
            raise LiveAstraSweepError(
                f"{cell.cell_id} exited with status "
                f"{completed.returncode}; see {stderr_path}")

        expected = expected_request_identities(scheduled_sessions)
        requests = parse_serving_requests_csv(
            cell.requests_csv,
            expected_identities=expected,
        )
        metrics = compute_live_comparison_metrics(
            scheduled_sessions,
            requests,
            measurement_session_ids=measurement_session_ids,
            ttft_slo_ns=ttft_slo_ns,
            tpot_slo_ns=tpot_slo_ns,
        )
        session_report = _strict_json(cell.session_report)
        runtime_report = _strict_json(cell.runtime_report)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "cell_id": cell.cell_id,
            "system": cell.system.key,
            "runtime_kind": cell.system.runtime_kind,
            "layout": cell.system.layout,
            "seed": cell.seed,
            "offered_session_rate_per_second": cell.rate,
            "runtime_guard_contract": _cell_runtime_guard_contract(cell),
            "wall_time_seconds": elapsed,
            "network_backend": NETWORK_BACKEND,
            "latency_model": LATENCY_MODEL,
            "request_routing_policy": "RR",
            "log_interval_seconds": log_interval_seconds,
            "astra_cycles_required": True,
            "workload": {
                "path": str(cell.workload_path),
                "sha256": cell.workload_sha256,
                "session_count": cell.session_count,
                "request_count": cell.request_count,
            },
            "metrics": asdict(metrics),
            "bottleneck_fields": _extract_bottlenecks({
                "session": session_report,
                "runtime": runtime_report,
            }),
            "artifacts": {
                name: _artifact(path)
                for name, path in _required_artifact_paths(cell).items()
            },
        }
        _require_finite_tree(result, cell.cell_id)
        _atomic_json(cell.result_path, result)
    except BaseException:
        if not keep_failed_inputs:
            _safe_remove_cell_inputs(cell.inputs_dir, campaign_inputs)
        raise
    _safe_remove_cell_inputs(cell.inputs_dir, campaign_inputs)
    return result


def _resolved_specs(
    repo_root: Path,
    system_keys: Sequence[str],
) -> tuple[SystemSpec, ...]:
    if not system_keys or len(system_keys) != len(set(system_keys)):
        raise LiveAstraSweepError(
            "systems must be a non-empty unique sequence")
    unknown = set(system_keys) - set(SYSTEMS)
    if unknown:
        raise LiveAstraSweepError(
            f"unknown systems: {', '.join(sorted(unknown))}")
    specs = tuple(SYSTEMS[key] for key in system_keys)
    for spec in specs:
        for path in (spec.cluster_config, spec.policy_config):
            resolved = repo_root / path
            if not resolved.is_file():
                raise LiveAstraSweepError(
                    f"required config is missing: {resolved}")
    return specs


def _scenario_contract(scenario) -> tuple[str, str, tuple[str, ...]]:
    manifest = getattr(scenario, "manifest", None)
    if manifest is None:
        raise LiveAstraSweepError("scenario has no manifest")
    scenario_id = getattr(manifest, "scenario_id", None)
    source_sha256 = getattr(manifest, "source_sha256", None)
    measurement_session_ids = tuple(
        getattr(manifest, "measurement_session_ids", ()))
    if not isinstance(scenario_id, str) or not scenario_id:
        raise LiveAstraSweepError("scenario manifest has no scenario_id")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef"
               for character in source_sha256)
    ):
        raise LiveAstraSweepError(
            "scenario manifest has an invalid source_sha256")
    if (
        not measurement_session_ids
        or len(measurement_session_ids) != len(set(measurement_session_ids))
        or any(not isinstance(value, str) or not value
               for value in measurement_session_ids)
    ):
        raise LiveAstraSweepError(
            "scenario manifest has invalid measurement_session_ids")
    if not callable(getattr(scenario, "build_offered_plan", None)):
        raise LiveAstraSweepError(
            "scenario must expose build_offered_plan(seed=...)")
    return scenario_id, source_sha256, measurement_session_ids


def _runtime_guard_contract(
    scenario,
    *,
    seed: int,
    rate: float,
    schedule: Sequence | None = None,
) -> dict[str, object] | None:
    """Resolve and optionally bind a scenario's live-arrival guard to a schedule."""

    manifest = scenario.manifest
    required = getattr(
        manifest, "runtime_guard_validation_required", False)
    if type(required) is not bool:
        raise LiveAstraSweepError(
            "scenario runtime_guard_validation_required must be boolean")
    if not required:
        return None
    provider = getattr(scenario, "runtime_guard_contract", None)
    if not callable(provider):
        raise LiveAstraSweepError(
            "scenario requires runtime guard validation but exposes no "
            "runtime_guard_contract(seed=..., sessions_per_second=...)")
    raw = provider(seed=seed, sessions_per_second=rate)
    if not isinstance(raw, Mapping):
        raise LiveAstraSweepError(
            "scenario runtime guard contract must be an object")
    cutoff = raw.get("last_external_guard_offer_ns")
    expected_resumes = raw.get("expected_measurement_resume_count")
    if type(cutoff) is not int or cutoff < 0:
        raise LiveAstraSweepError(
            "scenario runtime guard cutoff must be a non-negative integer")
    if type(expected_resumes) is not int or expected_resumes <= 0:
        raise LiveAstraSweepError(
            "scenario runtime guard expected resume count must be positive")
    manifest_expected = getattr(
        manifest, "runtime_guard_expected_measurement_resume_count", None)
    if manifest_expected != expected_resumes:
        raise LiveAstraSweepError(
            "scenario runtime guard resume count disagrees with manifest")
    contract = {
        "seed": seed,
        "offered_session_rate_per_second": rate,
        "last_external_guard_offer_ns": cutoff,
        "expected_measurement_resume_count": expected_resumes,
    }
    for key in ("seed", "offered_session_rate_per_second"):
        if raw.get(key) != contract[key]:
            raise LiveAstraSweepError(
                f"scenario runtime guard {key} disagrees with schedule")
    if schedule is not None:
        if not schedule:
            raise LiveAstraSweepError(
                "runtime-guarded schedule cannot be empty")
        scheduled_tail = getattr(schedule[-1], "arrival_time_ns", None)
        if scheduled_tail != cutoff:
            raise LiveAstraSweepError(
                "scenario runtime guard cutoff disagrees with the final "
                "scheduled external offer")
    return contract


def load_scenario(
    trace_path: Path,
    factory_spec: str = DEFAULT_SCENARIO_FACTORY,
):
    """Load the pinned balanced scenario or a local custom scenario factory.

    A custom spec is ``module:function`` or ``/path/to/file.py:function``.
    The function receives the resolved TraceLab path and returns any object
    implementing the same small protocol as ``TraceLabComparisonScenario``:
    ``manifest`` plus ``build_offered_plan(seed=...).at_rate(rate)``.  This is
    the extension point for pressure-balanced or other preregistered cohorts;
    the runner remains responsible for exact materialization and validation.
    """

    if factory_spec == DEFAULT_SCENARIO_FACTORY:
        scenario = load_balanced_causal_prefix_scenario(trace_path)
        _scenario_contract(scenario)
        return scenario
    if ":" not in factory_spec:
        raise LiveAstraSweepError(
            "custom scenario factory must be module:function or "
            "/path/file.py:function")
    source, function_name = factory_spec.rsplit(":", 1)
    if not source or not function_name:
        raise LiveAstraSweepError("invalid custom scenario factory")
    source_path = Path(source)
    try:
        if source_path.suffix == ".py" or source_path.is_file():
            resolved = source_path.resolve()
            if not resolved.is_file():
                raise LiveAstraSweepError(
                    f"scenario factory file is missing: {resolved}")
            module_name = (
                "_llmsim_live_scenario_"
                + hashlib.sha256(str(resolved).encode()).hexdigest()[:16]
            )
            module_spec = importlib.util.spec_from_file_location(
                module_name, resolved)
            if module_spec is None or module_spec.loader is None:
                raise LiveAstraSweepError(
                    f"cannot load scenario factory file {resolved}")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(source)
        factory = getattr(module, function_name)
    except (ImportError, AttributeError, OSError) as exc:
        raise LiveAstraSweepError(
            f"cannot resolve scenario factory {factory_spec!r}") from exc
    if not callable(factory):
        raise LiveAstraSweepError(
            f"scenario factory {factory_spec!r} is not callable")
    scenario = factory(trace_path.resolve())
    _scenario_contract(scenario)
    return scenario


def _campaign_identity(
    *,
    repo_root: Path,
    trace_path: Path,
    scenario,
    scenario_factory: str,
    specs: Sequence[SystemSpec],
    rates: Sequence[float],
    seeds: Sequence[int],
    ttft_slo_ns: int,
    tpot_slo_ns: int,
    log_interval_seconds: float,
) -> dict[str, object]:
    log_interval_seconds = _validate_log_interval_seconds(
        log_interval_seconds)
    scenario_id, source_sha256, measurement_session_ids = (
        _scenario_contract(scenario))
    scenario_manifest = scenario.manifest
    if callable(getattr(scenario_manifest, "to_dict", None)):
        manifest_payload = scenario_manifest.to_dict()
    elif hasattr(scenario_manifest, "__dataclass_fields__"):
        manifest_payload = asdict(scenario_manifest)
    else:
        manifest_payload = {
            "scenario_id": scenario_id,
            "source_sha256": source_sha256,
            "measurement_session_ids": list(measurement_session_ids),
        }
    files = {str(trace_path): _sha256_file(trace_path)}
    for spec in specs:
        for relative in (spec.cluster_config, spec.policy_config):
            resolved = (repo_root / relative).resolve()
            files[str(relative)] = _sha256_file(resolved)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "scenario_source_sha256": source_sha256,
        "scenario_factory": scenario_factory,
        "scenario_manifest_sha256": _stable_json_sha256(manifest_payload),
        "measurement_session_ids_sha256": _stable_json_sha256(
            list(measurement_session_ids)),
        "trace_path": str(trace_path),
        "systems": [asdict(spec) for spec in specs],
        "rates": list(rates),
        "seeds": list(seeds),
        "ttft_slo_ns": ttft_slo_ns,
        "tpot_slo_ns": tpot_slo_ns,
        "network_backend": NETWORK_BACKEND,
        "latency_model": LATENCY_MODEL,
        "request_routing_policy": "RR",
        "log_interval_seconds": log_interval_seconds,
        "files": files,
        "simulator_implementation": _campaign_implementation_identity(
            repo_root),
    }
    runtime_guard_contracts = [
        contract
        for seed in seeds
        for rate in rates
        for contract in (
            _runtime_guard_contract(
                scenario,
                seed=seed,
                rate=rate,
            ),
        )
        if contract is not None
    ]
    if runtime_guard_contracts:
        identity["runtime_guard_validation_required"] = True
        identity["runtime_guard_contracts"] = runtime_guard_contracts
    # Dataclass paths need a canonical string projection.
    identity["systems"] = [
        {
            **row,
            "cluster_config": str(row["cluster_config"]),
            "policy_config": str(row["policy_config"]),
        }
        for row in identity["systems"]
    ]
    return identity


def _load_or_initialize_manifest(
    path: Path,
    *,
    identity: Mapping[str, object],
    cells: Sequence[Cell],
) -> dict[str, object]:
    digest = _stable_json_sha256(identity)
    log_interval_seconds = _validate_log_interval_seconds(
        identity.get("log_interval_seconds"))
    if path.exists():
        existing = dict(_strict_json(path))
        if existing.get("campaign_sha256") != digest:
            raise LiveAstraSweepError(
                "existing manifest belongs to a different campaign")
        recorded_cells = existing.get("cells")
        if not isinstance(recorded_cells, dict):
            raise LiveAstraSweepError(
                "existing manifest has invalid cells")
        if set(recorded_cells) != {cell.cell_id for cell in cells}:
            raise LiveAstraSweepError(
                "existing manifest cell roster changed")
        for cell in cells:
            entry = recorded_cells[cell.cell_id]
            if (
                not isinstance(entry, dict)
                or entry.get("workload_sha256") != cell.workload_sha256
                or entry.get("request_count") != cell.request_count
                or entry.get("session_count") != cell.session_count
                or entry.get("log_interval_seconds")
                != log_interval_seconds
                or entry.get("runtime_guard_contract")
                != _cell_runtime_guard_contract(cell)
            ):
                raise LiveAstraSweepError(
                    f"existing manifest schedule changed for {cell.cell_id}")
        return existing
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "campaign_sha256": digest,
        "campaign": identity,
        "created_unix_seconds": time.time(),
        "updated_unix_seconds": time.time(),
        "cells": {
            cell.cell_id: {
                "status": "pending",
                "system": cell.system.key,
                "seed": cell.seed,
                "rate": cell.rate,
                "workload_sha256": cell.workload_sha256,
                "request_count": cell.request_count,
                "session_count": cell.session_count,
                "log_interval_seconds": log_interval_seconds,
                "runtime_guard_contract": _cell_runtime_guard_contract(cell),
                "result": str(cell.result_path),
            }
            for cell in cells
        },
    }
    _atomic_json(path, manifest)
    return manifest


def _is_resumable_completion(cell: Cell, entry: Mapping[str, object]) -> bool:
    if entry.get("status") != "completed" or not cell.result_path.is_file():
        return False
    recorded_result_sha256 = entry.get("result_sha256")
    recorded_result_bytes = entry.get("result_bytes")
    if (
        not isinstance(recorded_result_sha256, str)
        or len(recorded_result_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in recorded_result_sha256
        )
        or type(recorded_result_bytes) is not int
        or recorded_result_bytes < 0
    ):
        return False
    try:
        if (
            cell.result_path.stat().st_size != recorded_result_bytes
            or _sha256_file(cell.result_path) != recorded_result_sha256
        ):
            return False
        result = _strict_json(cell.result_path)
    except (LiveAstraSweepError, OSError):
        return False
    workload = result.get("workload")
    artifacts = result.get("artifacts")
    if not (
        result.get("status") == "completed"
        and result.get("schema_version") == SCHEMA_VERSION
        and result.get("cell_id") == cell.cell_id
        and result.get("system") == cell.system.key
        and result.get("seed") == cell.seed
        and result.get("offered_session_rate_per_second") == cell.rate
        and result.get("log_interval_seconds")
        == entry.get("log_interval_seconds")
        and result.get("runtime_guard_contract")
        == _cell_runtime_guard_contract(cell)
        and isinstance(result.get("metrics"), dict)
        and isinstance(workload, Mapping)
        and workload.get("sha256") == cell.workload_sha256
        and workload.get("request_count") == cell.request_count
        and workload.get("session_count") == cell.session_count
        and isinstance(artifacts, Mapping)
        and set(_REQUIRED_RESULT_ARTIFACTS).issubset(artifacts)
    ):
        return False
    expected_paths = _required_artifact_paths(cell)
    return all(
        _artifact_record_matches(artifacts[name], expected_paths[name])
        for name in _REQUIRED_RESULT_ARTIFACTS
    )


def _build_cells_and_schedules(
    *,
    scenario,
    specs: Sequence[SystemSpec],
    rates: Sequence[float],
    seeds: Sequence[int],
    output_root: Path,
    campaign_inputs: Path,
) -> tuple[list[Cell], dict[tuple[int, float], tuple]]:
    cells = []
    schedules = {}
    workload_root = output_root / "_workloads"
    for seed in seeds:
        plan = scenario.build_offered_plan(seed=seed)
        for rate in rates:
            schedule = plan.at_rate(rate)
            runtime_guard = _runtime_guard_contract(
                scenario,
                seed=seed,
                rate=rate,
                schedule=schedule,
            )
            schedule_key = (seed, rate)
            schedules[schedule_key] = schedule
            stem = f"seed{seed}-rate{_rate_tag(rate)}"
            workload = materialize_scheduled_sessions(
                schedule,
                workload_root / f"{stem}.jsonl",
                source_sha256=scenario.manifest.source_sha256,
            )
            for spec in specs:
                cell_id = f"{stem}-{spec.key}"
                cells.append(Cell(
                    cell_id=cell_id,
                    system=spec,
                    seed=seed,
                    rate=rate,
                    workload_path=workload.path,
                    workload_sha256=workload.sha256,
                    cell_dir=output_root / "cells" / cell_id,
                    inputs_dir=campaign_inputs / cell_id / "inputs",
                    request_count=workload.request_count,
                    session_count=workload.session_count,
                    last_external_guard_offer_ns=(
                        None if runtime_guard is None else int(
                            runtime_guard[
                                "last_external_guard_offer_ns"])),
                    expected_measurement_resume_count=(
                        None if runtime_guard is None else int(
                            runtime_guard[
                                "expected_measurement_resume_count"])),
                ))
    return cells, schedules


def run_campaign(
    *,
    repo_root: Path,
    trace_path: Path,
    output_root: Path,
    inputs_root: Path,
    python_executable: Path,
    system_keys: Sequence[str],
    rates: Sequence[float],
    seeds: Sequence[int],
    max_parallel: int,
    ttft_slo_ns: int = DEFAULT_TTFT_SLO_NS,
    tpot_slo_ns: int = DEFAULT_TPOT_SLO_NS,
    log_interval_seconds: float = DEFAULT_LOG_INTERVAL_SECONDS,
    timeout_seconds: float | None = None,
    keep_failed_inputs: bool = True,
    dry_run: bool = False,
    scenario=None,
    scenario_factory: str = DEFAULT_SCENARIO_FACTORY,
) -> Mapping[str, object]:
    """Execute or describe a complete paired live-ASTRA campaign."""

    repo_root = repo_root.resolve()
    trace_path = trace_path.resolve()
    output_root = output_root.resolve()
    inputs_root = inputs_root.resolve()
    if not trace_path.is_file():
        raise LiveAstraSweepError(f"TraceLab file is missing: {trace_path}")
    if max_parallel < 1 or max_parallel > 128:
        raise LiveAstraSweepError("max_parallel must be in [1, 128]")
    if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0):
        raise LiveAstraSweepError(
            "timeout_seconds must be positive and finite")
    log_interval_seconds = _validate_log_interval_seconds(
        log_interval_seconds)
    rates = _validate_rates(rates)
    seeds = _validate_seeds(seeds)
    specs = _resolved_specs(repo_root, system_keys)

    if scenario is None:
        scenario = load_scenario(trace_path, scenario_factory)
    _, _, measurement_session_ids = _scenario_contract(scenario)
    manifest_contract = scenario.manifest
    output_root.mkdir(parents=True, exist_ok=True)
    campaign_identity = _campaign_identity(
        repo_root=repo_root,
        trace_path=trace_path,
        scenario=scenario,
        scenario_factory=scenario_factory,
        specs=specs,
        rates=rates,
        seeds=seeds,
        ttft_slo_ns=ttft_slo_ns,
        tpot_slo_ns=tpot_slo_ns,
        log_interval_seconds=log_interval_seconds,
    )
    campaign_sha = _stable_json_sha256(campaign_identity)
    campaign_inputs = inputs_root / campaign_sha[:16]
    campaign_inputs.mkdir(parents=True, exist_ok=True)
    cells, schedules = _build_cells_and_schedules(
        scenario=scenario,
        specs=specs,
        rates=rates,
        seeds=seeds,
        output_root=output_root,
        campaign_inputs=campaign_inputs,
    )
    manifest_path = output_root / "manifest.json"
    manifest = _load_or_initialize_manifest(
        manifest_path, identity=campaign_identity, cells=cells)
    if dry_run:
        return {
            "campaign_sha256": campaign_sha,
            "manifest": str(manifest_path),
            "cell_count": len(cells),
            "commands": [
                list(build_serving_command(
                    repo_root=repo_root,
                    python_executable=python_executable,
                    cell=cell,
                    log_interval_seconds=log_interval_seconds,
                ))
                for cell in cells
            ],
        }

    pending = [
        cell
        for cell in cells
        if not _is_resumable_completion(
            cell, manifest["cells"][cell.cell_id])
    ]
    for cell in cells:
        if cell not in pending:
            manifest["cells"][cell.cell_id]["status"] = "completed"
            manifest["cells"][cell.cell_id]["resumed"] = True
    manifest["updated_unix_seconds"] = time.time()
    _atomic_json(manifest_path, manifest)

    failures = {}
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for cell in pending:
            entry = manifest["cells"][cell.cell_id]
            entry["status"] = "running"
            entry["started_unix_seconds"] = time.time()
            future = executor.submit(
                _run_cell,
                repo_root=repo_root,
                python_executable=python_executable,
                cell=cell,
                scheduled_sessions=schedules[(cell.seed, cell.rate)],
                measurement_session_ids=(
                    measurement_session_ids),
                ttft_slo_ns=ttft_slo_ns,
                tpot_slo_ns=tpot_slo_ns,
                campaign_inputs=campaign_inputs,
                timeout_seconds=timeout_seconds,
                keep_failed_inputs=keep_failed_inputs,
                log_interval_seconds=log_interval_seconds,
            )
            futures[future] = cell
        manifest["updated_unix_seconds"] = time.time()
        _atomic_json(manifest_path, manifest)

        for future in as_completed(futures):
            cell = futures[future]
            entry = manifest["cells"][cell.cell_id]
            try:
                result = future.result()
            except BaseException as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                failures[cell.cell_id] = entry["error"]
            else:
                entry["status"] = "completed"
                result_artifact = _artifact(cell.result_path)
                entry["result_sha256"] = result_artifact["sha256"]
                entry["result_bytes"] = result_artifact["bytes"]
                entry["operational_request_goodput_per_second"] = (
                    result["metrics"][
                        "operational_request_goodput_per_second"])
                entry.pop("error", None)
            entry["finished_unix_seconds"] = time.time()
            manifest["updated_unix_seconds"] = time.time()
            _atomic_json(manifest_path, manifest)

    manifest["status"] = "failed" if failures else "completed"
    manifest["updated_unix_seconds"] = time.time()
    manifest["failure_count"] = len(failures)
    _atomic_json(manifest_path, manifest)
    if failures:
        rendered = "; ".join(
            f"{cell_id}: {error}"
            for cell_id, error in sorted(failures.items())
        )
        raise LiveAstraSweepError(
            f"{len(failures)} live ASTRA cells failed: {rendered}")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m serving.live_astra_comparison_sweep",
        description=(
            "Run paired TraceLab cells through live LLMServingSim + ASTRA-Sim"
        ),
    )
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument(
        "--scenario-factory",
        default=DEFAULT_SCENARIO_FACTORY,
        help=(
            "'balanced', module:function, or /path/file.py:function; custom "
            "factories receive the TraceLab path and return a scenario"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--inputs-root", type=Path, default=DEFAULT_INPUTS_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--systems",
        default=",".join(DEFAULT_SYSTEMS),
        help="comma-separated subset of " + ",".join(DEFAULT_SYSTEMS),
    )
    parser.add_argument(
        "--rates",
        default=None,
        help="comma-separated system-wide session-start rates in (0,5]",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="comma-separated non-negative seeds",
    )
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--ttft-slo-seconds", type=float, default=30.0)
    parser.add_argument("--tpot-slo-milliseconds", type=float, default=300.0)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--log-interval",
        type=_positive_finite_log_interval,
        default=DEFAULT_LOG_INTERVAL_SECONDS,
        help=(
            "seconds between serving progress logs; this is pinned in "
            "campaign and cell provenance"
        ),
    )
    parser.add_argument(
        "--keep-failed-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode_rates = PILOT_RATES if args.mode == "pilot" else FULL_RATES
    mode_seeds = PILOT_SEEDS if args.mode == "pilot" else FULL_SEEDS
    rates = (
        mode_rates
        if args.rates is None
        else _parse_csv_list(args.rates, converter=float, name="rates")
    )
    seeds = (
        mode_seeds
        if args.seeds is None
        else _parse_csv_list(args.seeds, converter=int, name="seeds")
    )
    systems = tuple(
        item.strip() for item in args.systems.split(",") if item.strip())
    if (
        not math.isfinite(args.ttft_slo_seconds)
        or args.ttft_slo_seconds <= 0.0
        or not math.isfinite(args.tpot_slo_milliseconds)
        or args.tpot_slo_milliseconds <= 0.0
    ):
        raise SystemExit("SLO thresholds must be positive and finite")
    try:
        result = run_campaign(
            repo_root=Path(__file__).resolve().parents[1],
            trace_path=args.trace,
            output_root=args.output_root,
            inputs_root=args.inputs_root,
            python_executable=args.python,
            system_keys=systems,
            rates=rates,
            seeds=seeds,
            max_parallel=args.max_parallel,
            ttft_slo_ns=int(round(
                args.ttft_slo_seconds * 1_000_000_000)),
            tpot_slo_ns=int(round(
                args.tpot_slo_milliseconds * 1_000_000)),
            log_interval_seconds=args.log_interval,
            timeout_seconds=args.timeout_seconds,
            keep_failed_inputs=args.keep_failed_inputs,
            dry_run=args.dry_run,
            scenario_factory=args.scenario_factory,
        )
    except (
        LiveAstraSweepError,
        LiveComparisonMetricsError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

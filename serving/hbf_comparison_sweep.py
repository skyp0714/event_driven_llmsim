"""Reproducible parallel sweep for the balanced TraceLab comparison.

The sweep unit is deliberately one process per ``(rate, seed, system)`` cell.
The parent constructs each seeded Poisson plan once and gives every system in
the pairwise group the exact same immutable schedule.  Workers publish
``cell.json`` and ``requests.csv`` through the comparison cell bundle API,
then atomically add ``completion.json`` as the validated commit marker.

An existing directory is never accepted merely because it exists.  Resume
validates the marker, artifact hashes, frozen workload hashes, measurement
roster, hardware configuration, and full-drain identity set before reusing a
cell.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Callable, Mapping, Optional, Sequence

from .core.hbf_comparison_cell import (
    ASTRA_CYCLES_USED,
    BASELINE_POLICIES,
    CELL_SCHEMA_VERSION,
    DEFAULT_FIRST_TTFT_SECONDS,
    DEFAULT_RESUME_TTFT_SECONDS,
    DEFAULT_TPOT_MILLISECONDS,
    HBF_CONFIG_PATHS,
    HBF_LAYOUTS,
    PINNED_GPU_CONFIG,
    SIMULATION_BACKEND,
    SYSTEM_KEYS,
    build_slo_thresholds,
    json_safe,
    run_comparison_cell,
    write_cell_output_bundle_atomic,
    write_json_atomic,
)
from .core.hbf_comparison_workload import (
    ScheduledSession,
    stable_json_sha256,
)
from .core.tracelab_comparison_scenarios import (
    BALANCED_DEFAULT_RATES,
    BalancedCausalPrefixManifest,
    LONG_COLD_ANCHOR_RATES,
    LongColdContextStressManifest,
    TraceLabComparisonScenario,
    load_balanced_causal_prefix_scenario,
    load_long_cold_context_stress_scenario,
)


SWEEP_SCHEMA_VERSION = 1
COMPLETION_MARKER_SCHEMA_VERSION = 1
DEFAULT_TRACE_FILENAME = "tracelab-schema3-sps0.2-final.jsonl"
DEFAULT_SEEDS = (
    101, 211, 307, 401, 503, 607,
    701, 809, 907, 1009, 1103, 1201,
)
TOP_LEVEL_MANIFEST = "manifest.json"
CELL_JSON = "cell.json"
REQUEST_CSV = "requests.csv"
COMPLETION_JSON = "completion.json"
PINNED_MODEL_CONFIG = Path(
    "configs/model/Qwen/Qwen3-30B-A3B-Instruct-2507.json")

# This is the import closure used by ``run_comparison_cell``.  Keeping the
# list explicit avoids making an unrelated analysis module part of a long
# sweep merely because a caller happened to import it in the parent process.
COMPARISON_SOURCE_FILES = (
    Path("serving/__init__.py"),
    Path("serving/core/__init__.py"),
    Path("serving/core/gpu_hbf_hybrid.py"),
    Path("serving/core/gpu_pd_dual_oracle.py"),
    Path("serving/core/gpu_pd_dual_tiered.py"),
    Path("serving/core/gpu_pd_hbm.py"),
    Path("serving/core/gpu_pd_latency.py"),
    Path("serving/core/gpu_pd_oracle_node.py"),
    Path("serving/core/gpu_pd_pool.py"),
    Path("serving/core/gpu_pd_tier_lifecycle.py"),
    Path("serving/core/gpu_pd_tier_resources.py"),
    Path("serving/core/gpu_pd_tiered_node.py"),
    Path("serving/core/h100_kernel_calibrated_prompt.py"),
    Path("serving/core/hbf_comparison_cell.py"),
    Path("serving/core/hbf_comparison_metrics.py"),
    Path("serving/core/hbf_comparison_workload.py"),
    Path("serving/core/hbf_full_model_latency.py"),
    Path("serving/core/hbf_full_model_lifecycle.py"),
    Path("serving/core/hbf_full_model_pool.py"),
    Path("serving/core/online_latency_model.py"),
    Path("serving/core/tracelab_comparison_scenarios.py"),
    Path("serving/hbf_comparison_sweep.py"),
)


class ComparisonSweepError(RuntimeError):
    """Raised when a sweep cannot preserve its preregistered contract."""


@dataclass(frozen=True)
class CellTask:
    """Pickle-safe description of one isolated worker process."""

    repo_root: Path
    output_root: Path
    output_dir: Path
    scenario_id: str
    scenario_manifest_sha256: str
    seed: int
    session_rate: float
    rate_text: str
    system_key: str
    scheduled_sessions: tuple[ScheduledSession, ...]
    measurement_identities: tuple[str, ...]
    schedule_pair_sha256: str
    expected_call_specs_sha256: str
    expected_schedule_sha256: str
    expected_call_identities_sha256: str
    expected_call_identity_set_sha256: str
    expected_call_count: int
    expected_session_count: int
    measurement_identities_sha256: str
    system_config_contract: Mapping[str, object]
    system_config_contract_sha256: str
    execution_code_sha256: str
    first_ttft_seconds: float
    resume_ttft_seconds: float
    tpot_milliseconds: float
    cell_contract: Mapping[str, object]
    cell_contract_sha256: str


@dataclass(frozen=True)
class SweepPlan:
    """Fully frozen sweep inputs before any worker is launched."""

    repo_root: Path
    trace_path: Path
    output_root: Path
    scenario: TraceLabComparisonScenario
    scenario_manifest: Mapping[str, object]
    scenario_manifest_sha256: str
    rates: tuple[float, ...]
    rate_texts: tuple[str, ...]
    seeds: tuple[int, ...]
    system_keys: tuple[str, ...]
    workers: int
    detected_physical_cores: int
    first_ttft_seconds: float
    resume_ttft_seconds: float
    tpot_milliseconds: float
    thresholds_ns: Mapping[str, int]
    code_revision: Mapping[str, object]
    execution_code_sha256: str
    system_config_contracts: Mapping[str, Mapping[str, object]]
    system_config_contracts_sha256: str
    schedule_pairs: tuple[Mapping[str, object], ...]
    tasks: tuple[CellTask, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ComparisonSweepError(
            f"cannot hash required file {path}: {exc}") from exc
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, object]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ComparisonSweepError(
            f"cell artifact must be a regular file: {target}")
    try:
        size_bytes = target.stat().st_size
    except OSError as exc:
        raise ComparisonSweepError(
            f"cannot stat cell artifact {target}: {exc}") from exc
    return {
        "sha256": _sha256_file(target),
        "size_bytes": size_bytes,
    }


def _strict_json_object(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicates(
            pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ComparisonSweepError(
            f"required JSON artifact is not a regular file: {target}")
    try:
        with target.open("r", encoding="utf-8") as source:
            value = json.load(
                source,
                parse_constant=reject_constant,
                object_pairs_hook=reject_duplicates,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ComparisonSweepError(
            f"cannot parse strict JSON artifact {target}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonSweepError(
            f"JSON artifact must contain an object: {target}")
    return value


def _require_exact_keys(
        value: Mapping[str, object],
        expected: set[str],
        context: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise ComparisonSweepError(
            f"{context} schema mismatch: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}")


def default_trace_path(
        environment: Optional[Mapping[str, str]] = None) -> Path:
    """Resolve the pinned filename below ``LLMSIM_DATA``."""

    values = os.environ if environment is None else environment
    raw = values.get("LLMSIM_DATA")
    data_root = (
        Path.home() / "llmsim-data"
        if raw is None
        else Path(raw).expanduser()
    )
    return data_root / DEFAULT_TRACE_FILENAME


def discover_physical_core_count() -> int:
    """Count affinity-visible physical cores, with portable fallbacks."""

    try:
        allowed = set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        logical = os.cpu_count()
        return max(1, 1 if logical is None else logical)

    physical = set()
    for cpu in sorted(allowed):
        topology = Path(
            f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package_id = int(
                (topology / "physical_package_id").read_text(
                    encoding="utf-8").strip()
            )
            core_id = int(
                (topology / "core_id").read_text(
                    encoding="utf-8").strip()
            )
        except (OSError, ValueError):
            physical = set()
            break
        physical.add((package_id, core_id))
    if physical:
        return len(physical)
    return max(1, len(allowed))


def default_worker_count(physical_cores: Optional[int] = None) -> int:
    """Leave a small host headroom while using physical cores only."""

    count = (
        discover_physical_core_count()
        if physical_cores is None
        else physical_cores
    )
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("physical_cores must be a positive integer")
    if count == 1:
        return 1
    headroom = min(6, max(2, count // 8))
    return max(1, count - headroom)


def _rate_text(rate: object) -> str:
    if isinstance(rate, bool):
        raise ComparisonSweepError("rates must be positive finite numbers")
    try:
        decimal = Decimal(str(rate))
    except (InvalidOperation, ValueError) as exc:
        raise ComparisonSweepError(
            f"invalid session rate {rate!r}") from exc
    if not decimal.is_finite() or decimal <= 0:
        raise ComparisonSweepError(
            f"invalid session rate {rate!r}")
    rendered = format(decimal.normalize(), "f")
    return "0" if rendered == "-0" else rendered


def _rate_directory(rate_text: str) -> str:
    return f"rate_{rate_text.replace('.', 'p')}"


def _git_dir(worktree: Path) -> Optional[Path]:
    marker = worktree / ".git"
    if marker.is_dir():
        return marker.resolve()
    if marker.is_file():
        try:
            content = marker.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        prefix = "gitdir:"
        if not content.startswith(prefix):
            return None
        return (worktree / content[len(prefix):].strip()).resolve()
    return None


def _git_head(worktree: Path) -> Optional[str]:
    git_dir = _git_dir(worktree)
    if git_dir is None:
        return None
    try:
        head = (git_dir / "HEAD").read_text(
            encoding="utf-8").strip()
    except OSError:
        return None
    if not head.startswith("ref: "):
        return head if len(head) == 40 else None
    ref = head[5:]
    try:
        value = (git_dir / ref).read_text(
            encoding="utf-8").strip()
        return value if len(value) == 40 else None
    except OSError:
        pass
    try:
        lines = (git_dir / "packed-refs").read_text(
            encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith(("#", "^")):
            continue
        fields = line.split(" ", 1)
        if len(fields) == 2 and fields[1] == ref:
            return fields[0] if len(fields[0]) == 40 else None
    return None


def _code_revision_contract(repo_root: Path) -> dict[str, object]:
    root = Path(repo_root).resolve()
    source_hashes = {
        relative.as_posix(): _sha256_file(root / relative)
        for relative in COMPARISON_SOURCE_FILES
    }
    execution_payload = {
        "python_implementation": sys.implementation.name,
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "serving_python_files": source_hashes,
    }
    return {
        "repository_git_head": _git_head(root),
        "astra_sim_git_head": _git_head(root / "astra-sim"),
        **execution_payload,
        "serving_python_tree_sha256": stable_json_sha256(
            source_hashes),
        "execution_code_sha256": stable_json_sha256(
            execution_payload),
    }


def _config_file_record(
        repo_root: Path,
        relative_path: Path,
) -> dict[str, object]:
    target = repo_root / relative_path
    if target.is_symlink() or not target.is_file():
        raise ComparisonSweepError(
            f"required hardware config is not a regular file: {target}")
    return {
        "repo_relative_path": relative_path.as_posix(),
        "content_sha256": _sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def _system_config_contract(
        repo_root: Path,
        system_key: str,
) -> dict[str, object]:
    if system_key not in SYSTEM_KEYS:
        raise ComparisonSweepError(
            f"unsupported system key {system_key!r}")
    hbf_path = HBF_CONFIG_PATHS.get(system_key)
    return {
        "system_key": system_key,
        "system_class": (
            "dual_finite_hbm_tiered"
            if system_key in BASELINE_POLICIES
            else (
                "dual_strict_infinite_hbm_oracle"
                if system_key == "oracle"
                else "gpu_hbf_hybrid"
            )
        ),
        "tiering_policy": BASELINE_POLICIES.get(system_key),
        "hbf_layout": HBF_LAYOUTS.get(system_key),
        "model_config": _config_file_record(
            repo_root, PINNED_MODEL_CONFIG),
        "gpu_config": _config_file_record(
            repo_root, PINNED_GPU_CONFIG),
        "hbf_config": (
            None
            if hbf_path is None
            else _config_file_record(repo_root, hbf_path)
        ),
    }


def _frozen_schedule_contract(
        scheduled_sessions: tuple[ScheduledSession, ...],
) -> dict[str, object]:
    call_rows = []
    schedule_rows = []
    identities = []
    for scheduled_rank, scheduled in enumerate(scheduled_sessions):
        session_identities = []
        for call in scheduled.session.calls:
            identity = call.completion_identity
            identities.append(identity)
            session_identities.append(identity)
            call_rows.append({
                "offer_index": scheduled.offer_index,
                "source_index": scheduled.session.source_index,
                "session_id": scheduled.session.session_id,
                "source_session_identity_sha256": (
                    scheduled.session.source_session_identity_sha256
                ),
                "call_index": call.call_index,
                "completion_identity": identity,
                "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens,
                "tool_duration_ns": call.tool_duration_ns,
                "cached_prefix_tokens": call.cached_prefix_tokens,
                "fresh_input_tokens": call.fresh_input_tokens,
                "lineage_status": call.lineage_status,
                "inter_turn_gap_type": call.inter_turn_gap_type,
            })
        schedule_rows.append({
            "scheduled_rank": scheduled_rank,
            "offer_index": scheduled.offer_index,
            "source_index": scheduled.session.source_index,
            "session_id": scheduled.session.session_id,
            "arrival_time_ns": scheduled.arrival_time_ns,
            "unit_interarrival_hex": float(
                scheduled.unit_interarrival).hex(),
            "unit_arrival_time_hex": float(
                scheduled.unit_arrival_time).hex(),
            "completion_identities": session_identities,
        })
    session_ids = [
        scheduled.session.session_id
        for scheduled in scheduled_sessions
    ]
    return {
        "session_count": len(scheduled_sessions),
        "call_count": len(call_rows),
        "call_specs_sha256": stable_json_sha256(call_rows),
        "schedule_sha256": stable_json_sha256(schedule_rows),
        "expected_call_identities_sha256": stable_json_sha256(
            identities),
        "expected_call_identity_set_sha256": stable_json_sha256(
            sorted(identities)),
        "expected_session_ids_sha256": stable_json_sha256(
            session_ids),
    }


def _schedule_pair_contract(
        *,
        scenario_id: str,
        seed: int,
        session_rate: float,
        rate_text: str,
        offered_session_ids_sha256: str,
        unit_draws_sha256: str,
        scheduled_sessions: tuple[ScheduledSession, ...],
) -> dict[str, object]:
    frozen = _frozen_schedule_contract(scheduled_sessions)
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "session_rate": session_rate,
        "rate_text": rate_text,
        "offered_session_ids_sha256": offered_session_ids_sha256,
        "unit_draws_sha256": unit_draws_sha256,
        **frozen,
    }


def _validate_int_list(
        values: Sequence[int],
        name: str,
) -> tuple[int, ...]:
    result = tuple(values)
    if not result:
        raise ComparisonSweepError(f"{name} cannot be empty")
    if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in result):
        raise ComparisonSweepError(
            f"{name} must contain non-negative integers")
    if len(result) != len(set(result)):
        raise ComparisonSweepError(
            f"{name} cannot contain duplicates")
    return result


def _validate_system_keys(
        values: Sequence[str],
) -> tuple[str, ...]:
    result = tuple(values)
    if not result:
        raise ComparisonSweepError("system_keys cannot be empty")
    if any(
            not isinstance(value, str) or value not in SYSTEM_KEYS
            for value in result):
        raise ComparisonSweepError(
            f"system_keys must be selected from {SYSTEM_KEYS}")
    if len(result) != len(set(result)):
        raise ComparisonSweepError(
            "system_keys cannot contain duplicates")
    return result


def _validate_rates(
        values: Sequence[float],
        scenario: TraceLabComparisonScenario,
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    if not values:
        raise ComparisonSweepError("rates cannot be empty")
    rates = []
    texts = []
    for value in values:
        if isinstance(value, bool):
            raise ComparisonSweepError(
                "rates must be positive finite numbers")
        try:
            rate = float(value)
        except (TypeError, ValueError) as exc:
            raise ComparisonSweepError(
                f"invalid session rate {value!r}") from exc
        scenario.manifest.arrival_contract.validate_rate(rate)
        rates.append(rate)
        texts.append(_rate_text(value))
    if len(texts) != len(set(texts)):
        raise ComparisonSweepError(
            "rates cannot contain exact duplicates")
    return tuple(rates), tuple(texts)


def build_sweep_plan(
        *,
        repo_root: Path,
        trace_path: Path,
        output_root: Path,
        rates: Sequence[float] = BALANCED_DEFAULT_RATES,
        seeds: Sequence[int] = DEFAULT_SEEDS,
        system_keys: Sequence[str] = SYSTEM_KEYS,
        workers: Optional[int] = None,
        first_ttft_seconds: float = DEFAULT_FIRST_TTFT_SECONDS,
        resume_ttft_seconds: float = DEFAULT_RESUME_TTFT_SECONDS,
        tpot_milliseconds: float = DEFAULT_TPOT_MILLISECONDS,
        scenario: Optional[TraceLabComparisonScenario] = None,
) -> SweepPlan:
    """Freeze every schedule, contract hash, and deterministic cell path."""

    root = Path(repo_root).resolve()
    trace = Path(trace_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    selected_scenario = (
        load_balanced_causal_prefix_scenario(trace)
        if scenario is None
        else scenario
    )
    if not isinstance(
            selected_scenario.manifest,
            (
                BalancedCausalPrefixManifest,
                LongColdContextStressManifest,
            )):
        raise ComparisonSweepError(
            "the sweep requires a balanced causal-prefix or long-cold "
            "context scenario")

    rate_values, rate_texts = _validate_rates(
        rates, selected_scenario)
    seed_values = _validate_int_list(seeds, "seeds")
    systems = _validate_system_keys(system_keys)
    detected_physical_cores = discover_physical_core_count()
    worker_count = (
        default_worker_count(detected_physical_cores)
        if workers is None
        else workers
    )
    if (
        isinstance(worker_count, bool)
        or not isinstance(worker_count, int)
        or worker_count <= 0
    ):
        raise ComparisonSweepError(
            "workers must be a positive integer")

    thresholds = build_slo_thresholds(
        first_ttft_seconds=first_ttft_seconds,
        resume_ttft_seconds=resume_ttft_seconds,
        tpot_milliseconds=tpot_milliseconds,
    )
    thresholds_ns = {
        "first_ttft_ns": thresholds.first_ttft_ns,
        "resume_ttft_ns": thresholds.resume_ttft_ns,
        "tpot_ns": thresholds.tpot_ns,
    }
    scenario_manifest = json_safe(
        selected_scenario.manifest.to_dict())
    if not isinstance(scenario_manifest, dict):
        raise AssertionError("scenario manifest did not serialize to an object")
    scenario_manifest_sha256 = stable_json_sha256(
        scenario_manifest)
    measurement_ids = tuple(
        selected_scenario.manifest.measurement_request_identities)
    measurement_sha256 = stable_json_sha256(
        list(measurement_ids))
    if measurement_sha256 != (
            selected_scenario.manifest
            .measurement_request_identities_sha256):
        raise ComparisonSweepError(
            "scenario measurement roster hash is inconsistent")

    code_revision = _code_revision_contract(root)
    execution_code_sha256 = str(
        code_revision["execution_code_sha256"])
    config_contracts = {
        system_key: _system_config_contract(root, system_key)
        for system_key in systems
    }
    config_contracts_sha256 = stable_json_sha256(
        config_contracts)

    schedule_pairs = []
    tasks = []
    for seed in seed_values:
        offered_plan = selected_scenario.build_offered_plan(
            seed=seed)
        for rate, rate_text in zip(rate_values, rate_texts):
            scheduled = offered_plan.at_rate(rate)
            pair = _schedule_pair_contract(
                scenario_id=selected_scenario.manifest.scenario_id,
                seed=seed,
                session_rate=rate,
                rate_text=rate_text,
                offered_session_ids_sha256=(
                    offered_plan.offered_session_ids_sha256),
                unit_draws_sha256=offered_plan.unit_draws_sha256,
                scheduled_sessions=scheduled,
            )
            pair_sha256 = stable_json_sha256(pair)
            pair_manifest_row = {
                **pair,
                "schedule_pair_sha256": pair_sha256,
            }
            schedule_pairs.append(pair_manifest_row)
            for system_key in systems:
                config = config_contracts[system_key]
                config_sha256 = stable_json_sha256(config)
                cell_contract = {
                    "schema_version": SWEEP_SCHEMA_VERSION,
                    "scenario_id": (
                        selected_scenario.manifest.scenario_id),
                    "scenario_manifest_sha256": (
                        scenario_manifest_sha256),
                    "seed": seed,
                    "session_rate": rate,
                    "rate_text": rate_text,
                    "system_key": system_key,
                    "schedule_pair_sha256": pair_sha256,
                    "expected_call_specs_sha256": (
                        pair["call_specs_sha256"]),
                    "expected_schedule_sha256": (
                        pair["schedule_sha256"]),
                    "measurement_identities_sha256": (
                        measurement_sha256),
                    "system_config_contract_sha256": (
                        config_sha256),
                    "execution_code_sha256": (
                        execution_code_sha256),
                    "thresholds_ns": thresholds_ns,
                }
                cell_contract_sha256 = stable_json_sha256(
                    cell_contract)
                directory = (
                    output
                    / "cells"
                    / _rate_directory(rate_text)
                    / f"seed_{seed}"
                    / system_key
                )
                tasks.append(CellTask(
                    repo_root=root,
                    output_root=output,
                    output_dir=directory,
                    scenario_id=(
                        selected_scenario.manifest.scenario_id),
                    scenario_manifest_sha256=(
                        scenario_manifest_sha256),
                    seed=seed,
                    session_rate=rate,
                    rate_text=rate_text,
                    system_key=system_key,
                    scheduled_sessions=scheduled,
                    measurement_identities=measurement_ids,
                    schedule_pair_sha256=pair_sha256,
                    expected_call_specs_sha256=str(
                        pair["call_specs_sha256"]),
                    expected_schedule_sha256=str(
                        pair["schedule_sha256"]),
                    expected_call_identities_sha256=str(
                        pair["expected_call_identities_sha256"]),
                    expected_call_identity_set_sha256=str(
                        pair["expected_call_identity_set_sha256"]),
                    expected_call_count=int(pair["call_count"]),
                    expected_session_count=int(pair["session_count"]),
                    measurement_identities_sha256=measurement_sha256,
                    system_config_contract=config,
                    system_config_contract_sha256=config_sha256,
                    execution_code_sha256=execution_code_sha256,
                    first_ttft_seconds=first_ttft_seconds,
                    resume_ttft_seconds=resume_ttft_seconds,
                    tpot_milliseconds=tpot_milliseconds,
                    cell_contract=cell_contract,
                    cell_contract_sha256=cell_contract_sha256,
                ))

    return SweepPlan(
        repo_root=root,
        trace_path=trace,
        output_root=output,
        scenario=selected_scenario,
        scenario_manifest=scenario_manifest,
        scenario_manifest_sha256=scenario_manifest_sha256,
        rates=rate_values,
        rate_texts=rate_texts,
        seeds=seed_values,
        system_keys=systems,
        workers=worker_count,
        detected_physical_cores=detected_physical_cores,
        first_ttft_seconds=first_ttft_seconds,
        resume_ttft_seconds=resume_ttft_seconds,
        tpot_milliseconds=tpot_milliseconds,
        thresholds_ns=thresholds_ns,
        code_revision=code_revision,
        execution_code_sha256=execution_code_sha256,
        system_config_contracts=config_contracts,
        system_config_contracts_sha256=(
            config_contracts_sha256),
        schedule_pairs=tuple(schedule_pairs),
        tasks=tuple(tasks),
    )


def _validate_execution_inputs(task: CellTask) -> None:
    code = _code_revision_contract(task.repo_root)
    if code["execution_code_sha256"] != task.execution_code_sha256:
        raise ComparisonSweepError(
            "serving source code changed after the sweep was planned")
    config = _system_config_contract(
        task.repo_root, task.system_key)
    if (
        stable_json_sha256(config)
        != task.system_config_contract_sha256
        or config != task.system_config_contract
    ):
        raise ComparisonSweepError(
            f"hardware configuration changed before {task.system_key} ran")


def _validate_result_contract(
        result: Mapping[str, object],
        task: CellTask,
) -> dict[str, object]:
    if result.get("schema_version") != CELL_SCHEMA_VERSION:
        raise ComparisonSweepError(
            f"{task.system_key} returned the wrong cell schema")
    if (
        result.get("system_key") != task.system_key
        or result.get("session_rate") != task.session_rate
    ):
        raise ComparisonSweepError(
            f"{task.system_key} returned the wrong cell coordinates")

    frozen = result.get("frozen_workload")
    roster = result.get("measurement_roster")
    drain = result.get("full_drain")
    requests = result.get("requests")
    if not isinstance(frozen, Mapping):
        raise ComparisonSweepError("cell lacks frozen_workload")
    if not isinstance(roster, Mapping):
        raise ComparisonSweepError("cell lacks measurement_roster")
    if not isinstance(drain, Mapping):
        raise ComparisonSweepError("cell lacks full_drain")
    if not isinstance(requests, list):
        raise ComparisonSweepError("cell lacks request rows")

    expected_frozen = {
        "session_count": task.expected_session_count,
        "call_count": task.expected_call_count,
        "call_specs_sha256": task.expected_call_specs_sha256,
        "schedule_sha256": task.expected_schedule_sha256,
        "expected_call_identities_sha256": (
            task.expected_call_identities_sha256),
    }
    for key, expected in expected_frozen.items():
        if frozen.get(key) != expected:
            raise ComparisonSweepError(
                f"cell frozen workload mismatch for {key}: "
                f"observed={frozen.get(key)!r}, expected={expected!r}")
    if roster.get("ordered_identities_sha256") != (
            task.measurement_identities_sha256):
        raise ComparisonSweepError(
            "cell measurement roster differs from scenario manifest")

    calls = drain.get("calls")
    if not isinstance(calls, Mapping):
        raise ComparisonSweepError("cell lacks call full-drain hashes")
    if (
        calls.get("identity_count") != task.expected_call_count
        or calls.get("expected_set_sha256")
        != task.expected_call_identity_set_sha256
        or calls.get("completion_set_sha256")
        != task.expected_call_identity_set_sha256
    ):
        raise ComparisonSweepError(
            "cell did not preserve the exact full-drain call set")

    request_identities = []
    for row_index, row in enumerate(requests):
        if not isinstance(row, Mapping):
            raise ComparisonSweepError(
                f"request row {row_index} is not an object")
        identity = row.get("completion_identity")
        if not isinstance(identity, str) or not identity:
            raise ComparisonSweepError(
                f"request row {row_index} lacks an identity")
        if row.get("system_key") != task.system_key:
            raise ComparisonSweepError(
                f"request row {row_index} has the wrong system key")
        request_identities.append(identity)
    if (
        len(request_identities) != task.expected_call_count
        or len(request_identities) != len(set(request_identities))
        or stable_json_sha256(sorted(request_identities))
        != task.expected_call_identity_set_sha256
    ):
        raise ComparisonSweepError(
            "request rows differ from the full-drain identity set")

    simulation = result.get("simulation_contract")
    backend = (
        simulation.get("execution_backend")
        if isinstance(simulation, Mapping)
        else None
    )
    if (
        not isinstance(backend, Mapping)
        or backend.get("name") != SIMULATION_BACKEND
        or backend.get("astra_cycles_used") is not ASTRA_CYCLES_USED
    ):
        raise ComparisonSweepError(
            "cell execution backend differs from the planned analytical "
            "comparison backend")
    hardware = (
        simulation.get("hardware")
        if isinstance(simulation, Mapping)
        else None
    )
    if not isinstance(hardware, Mapping):
        raise ComparisonSweepError(
            "cell lacks simulation hardware provenance")
    expected_gpu = task.system_config_contract["gpu_config"]
    observed_gpu = hardware.get("gpu")
    if (
        not isinstance(observed_gpu, Mapping)
        or observed_gpu.get("repo_relative_path")
        != expected_gpu["repo_relative_path"]
        or observed_gpu.get("content_sha256")
        != expected_gpu["content_sha256"]
    ):
        raise ComparisonSweepError(
            "cell GPU config differs from the planned config")
    expected_hbf = task.system_config_contract["hbf_config"]
    observed_hbf = hardware.get("hbf")
    if expected_hbf is None:
        if observed_hbf is not None:
            raise ComparisonSweepError(
                "non-HBF cell unexpectedly reports an HBF config")
    elif (
        not isinstance(observed_hbf, Mapping)
        or observed_hbf.get("repo_relative_path")
        != expected_hbf["repo_relative_path"]
        or observed_hbf.get("content_sha256")
        != expected_hbf["content_sha256"]
    ):
        raise ComparisonSweepError(
            "cell HBF config differs from the planned config")

    return {
        "cell_schema_version": result["schema_version"],
        "call_specs_sha256": frozen["call_specs_sha256"],
        "schedule_sha256": frozen["schedule_sha256"],
        "measurement_identities_sha256": (
            roster["ordered_identities_sha256"]),
        "completion_call_set_sha256": (
            calls["completion_set_sha256"]),
        "completion_call_order_sha256": (
            calls["completion_order_sha256"]),
        "request_count": len(requests),
        "simulation_backend": SIMULATION_BACKEND,
        "astra_cycles_used": ASTRA_CYCLES_USED,
    }


def _completion_marker(
        *,
        task: CellTask,
        result_contract: Mapping[str, object],
        artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": COMPLETION_MARKER_SCHEMA_VERSION,
        "status": "complete",
        "cell_contract": dict(task.cell_contract),
        "cell_contract_sha256": task.cell_contract_sha256,
        "result_contract": dict(result_contract),
        "artifacts": {
            name: dict(record)
            for name, record in artifacts.items()
        },
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_csv_row_count(
        path: Path,
        expected_rows: int,
) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.reader(source)
            header = next(reader, None)
            count = sum(1 for _ in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ComparisonSweepError(
            f"cannot validate request CSV {path}: {exc}") from exc
    if not header or count != expected_rows:
        raise ComparisonSweepError(
            f"request CSV row count mismatch: "
            f"observed={count}, expected={expected_rows}")


def validate_completed_cell(task: CellTask) -> dict[str, object]:
    """Validate a committed cell before accepting it for resume."""

    directory = task.output_dir
    if directory.is_symlink() or not directory.is_dir():
        raise ComparisonSweepError(
            f"cell path is not a regular directory: {directory}")
    try:
        children = {path.name for path in directory.iterdir()}
    except OSError as exc:
        raise ComparisonSweepError(
            f"cannot inspect cell directory {directory}: {exc}") from exc
    expected_children = {CELL_JSON, REQUEST_CSV, COMPLETION_JSON}
    if children != expected_children:
        raise ComparisonSweepError(
            f"cell directory contents are incomplete or unexpected: "
            f"{directory}, observed={sorted(children)}")

    marker_path = directory / COMPLETION_JSON
    marker = _strict_json_object(marker_path)
    _require_exact_keys(
        marker,
        {
            "schema_version",
            "status",
            "cell_contract",
            "cell_contract_sha256",
            "result_contract",
            "artifacts",
        },
        "completion marker",
    )
    if (
        marker["schema_version"] != COMPLETION_MARKER_SCHEMA_VERSION
        or marker["status"] != "complete"
        or marker["cell_contract"] != task.cell_contract
        or marker["cell_contract_sha256"]
        != task.cell_contract_sha256
        or stable_json_sha256(marker["cell_contract"])
        != task.cell_contract_sha256
    ):
        raise ComparisonSweepError(
            f"completion marker contract mismatch: {directory}")

    artifacts = marker["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise ComparisonSweepError(
            f"completion marker lacks artifact hashes: {directory}")
    _require_exact_keys(
        artifacts, {CELL_JSON, REQUEST_CSV},
        "completion marker artifacts")
    for filename in (CELL_JSON, REQUEST_CSV):
        expected = artifacts[filename]
        if not isinstance(expected, Mapping):
            raise ComparisonSweepError(
                f"invalid artifact record for {filename}")
        _require_exact_keys(
            expected, {"sha256", "size_bytes"},
            f"artifact record {filename}")
        observed = _artifact_record(directory / filename)
        if observed != expected:
            raise ComparisonSweepError(
                f"cell artifact hash mismatch: {directory / filename}")

    result = _strict_json_object(directory / CELL_JSON)
    observed_result_contract = _validate_result_contract(
        result, task)
    if marker["result_contract"] != observed_result_contract:
        raise ComparisonSweepError(
            f"completion marker result hashes mismatch: {directory}")
    _validate_csv_row_count(
        directory / REQUEST_CSV, task.expected_call_count)

    relative_dir = directory.relative_to(
        task.output_root).as_posix()
    return {
        "seed": task.seed,
        "session_rate": task.session_rate,
        "rate_text": task.rate_text,
        "system_key": task.system_key,
        "relative_directory": relative_dir,
        "cell_contract_sha256": task.cell_contract_sha256,
        "schedule_pair_sha256": task.schedule_pair_sha256,
        "result_contract": observed_result_contract,
        "artifacts": {
            CELL_JSON: dict(artifacts[CELL_JSON]),
            REQUEST_CSV: dict(artifacts[REQUEST_CSV]),
            COMPLETION_JSON: _artifact_record(marker_path),
        },
    }


def _execute_cell(task: CellTask) -> dict[str, object]:
    """Run and commit one cell; invoked once in each spawned process."""

    _validate_execution_inputs(task)
    if os.path.lexists(task.output_dir):
        raise ComparisonSweepError(
            f"refusing to replace existing cell directory "
            f"{task.output_dir}")
    result = run_comparison_cell(
        repo_root=task.repo_root,
        system_key=task.system_key,
        scheduled_sessions=task.scheduled_sessions,
        session_rate=task.session_rate,
        measurement_identities=task.measurement_identities,
        first_ttft_seconds=task.first_ttft_seconds,
        resume_ttft_seconds=task.resume_ttft_seconds,
        tpot_milliseconds=task.tpot_milliseconds,
    )
    result_contract = _validate_result_contract(result, task)
    write_cell_output_bundle_atomic(task.output_dir, result)
    artifacts = {
        CELL_JSON: _artifact_record(task.output_dir / CELL_JSON),
        REQUEST_CSV: _artifact_record(task.output_dir / REQUEST_CSV),
    }
    marker = _completion_marker(
        task=task,
        result_contract=result_contract,
        artifacts=artifacts,
    )
    write_json_atomic(task.output_dir / COMPLETION_JSON, marker)
    _fsync_directory(task.output_dir)
    return validate_completed_cell(task)


def _preflight_cells(
        plan: SweepPlan,
        *,
        resume: bool,
) -> tuple[list[dict[str, object]], list[CellTask]]:
    if plan.output_root.is_symlink():
        raise ComparisonSweepError(
            f"output root cannot be a symlink: {plan.output_root}")
    if plan.output_root.exists() and not plan.output_root.is_dir():
        raise ComparisonSweepError(
            f"output root is not a directory: {plan.output_root}")
    completed = []
    pending = []
    for task in plan.tasks:
        if os.path.lexists(task.output_dir):
            if not resume:
                raise ComparisonSweepError(
                    "existing cell requires --resume after strict "
                    f"validation: {task.output_dir}")
            completed.append(validate_completed_cell(task))
        else:
            pending.append(task)
    return completed, pending


def describe_sweep_plan(
        plan: SweepPlan,
) -> dict[str, object]:
    """Return a no-write plan, validating any already completed cells."""

    statuses = []
    complete_count = 0
    for task in plan.tasks:
        if os.path.lexists(task.output_dir):
            validate_completed_cell(task)
            status = "complete_validated"
            complete_count += 1
        else:
            status = "pending"
        statuses.append({
            "rate_text": task.rate_text,
            "seed": task.seed,
            "system_key": task.system_key,
            "relative_directory": task.output_dir.relative_to(
                plan.output_root).as_posix(),
            "cell_contract_sha256": task.cell_contract_sha256,
            "status": status,
        })
    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "mode": "dry_run",
        "trace_path": str(plan.trace_path),
        "output_root": str(plan.output_root),
        "scenario_id": plan.scenario.manifest.scenario_id,
        "scenario_manifest_sha256": (
            plan.scenario_manifest_sha256),
        "rates": list(plan.rates),
        "seeds": list(plan.seeds),
        "system_keys": list(plan.system_keys),
        "workers": plan.workers,
        "detected_physical_cores": plan.detected_physical_cores,
        "cell_count": len(plan.tasks),
        "complete_validated_count": complete_count,
        "pending_count": len(plan.tasks) - complete_count,
        "execution_code_sha256": plan.execution_code_sha256,
        "system_config_contracts_sha256": (
            plan.system_config_contracts_sha256),
        "cells": statuses,
    }


def _validate_pairing(
        plan: SweepPlan,
        records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    by_coordinate = {}
    for record in records:
        key = (record["seed"], record["rate_text"], record["system_key"])
        if key in by_coordinate:
            raise ComparisonSweepError(
                f"duplicate completed cell coordinate {key!r}")
        by_coordinate[key] = record

    pair_rows = []
    for pair in plan.schedule_pairs:
        seed = pair["seed"]
        rate_text = pair["rate_text"]
        group = []
        for system_key in plan.system_keys:
            key = (seed, rate_text, system_key)
            if key not in by_coordinate:
                raise ComparisonSweepError(
                    f"missing paired cell {key!r}")
            record = by_coordinate[key]
            if record["schedule_pair_sha256"] != (
                    pair["schedule_pair_sha256"]):
                raise ComparisonSweepError(
                    f"paired schedule contract mismatch for {key!r}")
            group.append(record)
        schedule_hashes = {
            record["result_contract"]["schedule_sha256"]
            for record in group
        }
        call_hashes = {
            record["result_contract"]["call_specs_sha256"]
            for record in group
        }
        measurement_hashes = {
            record["result_contract"][
                "measurement_identities_sha256"]
            for record in group
        }
        completion_sets = {
            record["result_contract"][
                "completion_call_set_sha256"]
            for record in group
        }
        if (
            schedule_hashes != {pair["schedule_sha256"]}
            or call_hashes != {pair["call_specs_sha256"]}
            or measurement_hashes != {
                plan.scenario.manifest
                .measurement_request_identities_sha256}
            or completion_sets != {
                pair["expected_call_identity_set_sha256"]}
        ):
            raise ComparisonSweepError(
                "systems in a seed/rate group did not consume the exact "
                f"same schedule: seed={seed}, rate={rate_text}")
        pair_rows.append({
            **dict(pair),
            "system_keys": list(plan.system_keys),
            "result_schedule_sha256": next(iter(schedule_hashes)),
            "result_call_specs_sha256": next(iter(call_hashes)),
            "measurement_identities_sha256": next(
                iter(measurement_hashes)),
            "completion_call_set_sha256": next(
                iter(completion_sets)),
        })
    return tuple(pair_rows)


def _top_manifest(
        plan: SweepPlan,
        records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    paired = _validate_pairing(plan, records)
    record_by_key = {
        (record["seed"], record["rate_text"], record["system_key"]):
        record
        for record in records
    }
    ordered_records = [
        record_by_key[(task.seed, task.rate_text, task.system_key)]
        for task in plan.tasks
    ]
    rates_payload = list(plan.rate_texts)
    seeds_payload = list(plan.seeds)
    systems_payload = list(plan.system_keys)
    cells_sha256 = stable_json_sha256(ordered_records)
    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "scenario": {
            "scenario_id": plan.scenario.manifest.scenario_id,
            "source_path": str(plan.trace_path),
            "source_sha256": plan.scenario.workload.source_sha256,
            "manifest": dict(plan.scenario_manifest),
            "manifest_sha256": plan.scenario_manifest_sha256,
        },
        "grid": {
            "rates": list(plan.rates),
            "rate_texts": rates_payload,
            "rates_sha256": stable_json_sha256(rates_payload),
            "seeds": seeds_payload,
            "seeds_sha256": stable_json_sha256(seeds_payload),
            "system_keys": systems_payload,
            "system_keys_sha256": stable_json_sha256(
                systems_payload),
            "cell_count": len(plan.tasks),
        },
        "slo_thresholds_ns": dict(plan.thresholds_ns),
        "execution": {
            "executor": "concurrent.futures.ProcessPoolExecutor",
            "multiprocessing_start_method": "spawn",
            "max_tasks_per_child": 1,
            "one_isolated_cell_per_process": True,
            "workers": plan.workers,
            "detected_physical_cores": (
                plan.detected_physical_cores),
            "simulation_backend": SIMULATION_BACKEND,
            "astra_cycles_used": ASTRA_CYCLES_USED,
        },
        "code_revision_hashes": dict(plan.code_revision),
        "system_config_contracts": {
            key: dict(value)
            for key, value
            in plan.system_config_contracts.items()
        },
        "system_config_contracts_sha256": (
            plan.system_config_contracts_sha256),
        "pairing": {
            "semantics": (
                "one_seeded_unit_rate_poisson_plan_is_scaled_once_per_"
                "rate_and_consumed_identically_by_every_system"
            ),
            "measurement_identities_sha256": (
                plan.scenario.manifest
                .measurement_request_identities_sha256),
            "schedule_pairs": list(paired),
            "schedule_pairs_sha256": stable_json_sha256(
                paired),
        },
        "cells": ordered_records,
        "cells_sha256": cells_sha256,
    }


def _assert_inputs_unchanged(plan: SweepPlan) -> None:
    current_code = _code_revision_contract(plan.repo_root)
    if current_code["execution_code_sha256"] != (
            plan.execution_code_sha256):
        raise ComparisonSweepError(
            "serving source code changed during the sweep")
    current_configs = {
        system_key: _system_config_contract(
            plan.repo_root, system_key)
        for system_key in plan.system_keys
    }
    if (
        current_configs != plan.system_config_contracts
        or stable_json_sha256(current_configs)
        != plan.system_config_contracts_sha256
    ):
        raise ComparisonSweepError(
            "hardware configuration changed during the sweep")


def _publish_top_manifest(
        plan: SweepPlan,
        manifest: Mapping[str, object],
        *,
        resume: bool,
) -> Path:
    path = plan.output_root / TOP_LEVEL_MANIFEST
    if os.path.lexists(path):
        if not resume:
            raise ComparisonSweepError(
                f"top-level manifest already exists: {path}")
        existing = _strict_json_object(path)
        if existing != manifest:
            raise ComparisonSweepError(
                "existing top-level manifest differs from the fully "
                "validated current sweep")
        return path
    plan.output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, manifest)
    _fsync_directory(plan.output_root)
    return path


def _new_process_pool(max_workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
    )


def run_sweep(
        plan: SweepPlan,
        *,
        resume: bool = False,
        progress: Optional[Callable[[Mapping[str, object]], None]] = None,
) -> tuple[dict[str, object], Path]:
    """Execute pending cells, validate every pair, and publish the manifest."""

    manifest_path = plan.output_root / TOP_LEVEL_MANIFEST
    if os.path.lexists(manifest_path) and not resume:
        raise ComparisonSweepError(
            f"top-level manifest already exists: {manifest_path}")
    completed, pending = _preflight_cells(
        plan, resume=resume)
    if pending:
        with _new_process_pool(plan.workers) as executor:
            future_tasks = {
                executor.submit(_execute_cell, task): task
                for task in pending
            }
            for future in as_completed(future_tasks):
                task = future_tasks[future]
                try:
                    record = future.result()
                except BaseException:
                    for other in future_tasks:
                        other.cancel()
                    raise
                completed.append(record)
                if progress is not None:
                    progress({
                        "completed": len(completed),
                        "total": len(plan.tasks),
                        "seed": task.seed,
                        "rate_text": task.rate_text,
                        "system_key": task.system_key,
                    })

    _assert_inputs_unchanged(plan)
    manifest = _top_manifest(plan, completed)
    path = _publish_top_manifest(
        plan, manifest, resume=resume)
    return manifest, path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned balanced TraceLab comparison with one "
            "spawned process per cell."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=default_trace_path(),
        help=(
            "schema-3 TraceLab JSONL (default: "
            "$LLMSIM_DATA/tracelab-schema3-sps0.2-final.jsonl)"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        choices=("balanced", "long-cold"),
        default="balanced",
        help=(
            "pinned TraceLab scenario; long-cold is a finite "
            "non-equilibrium tier-pressure sensitivity"
        ),
    )
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=SYSTEM_KEYS,
        default=list(SYSTEM_KEYS),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "concurrent physical-core workers; default leaves host "
            "headroom"
        ),
    )
    parser.add_argument(
        "--first-ttft-seconds",
        type=float,
        default=DEFAULT_FIRST_TTFT_SECONDS,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse only strictly validated completed cells",
    )
    parser.add_argument(
        "--resume-ttft-seconds",
        type=float,
        default=DEFAULT_RESUME_TTFT_SECONDS,
    )
    parser.add_argument(
        "--tpot-milliseconds",
        type=float,
        default=DEFAULT_TPOT_MILLISECONDS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the complete no-write plan",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    scenario = (
        load_balanced_causal_prefix_scenario(args.trace)
        if args.scenario == "balanced"
        else load_long_cold_context_stress_scenario(args.trace)
    )
    default_rates = (
        BALANCED_DEFAULT_RATES
        if args.scenario == "balanced"
        else LONG_COLD_ANCHOR_RATES
    )
    plan = build_sweep_plan(
        repo_root=args.repo_root,
        trace_path=args.trace,
        output_root=args.output,
        rates=default_rates if args.rates is None else args.rates,
        seeds=args.seeds,
        system_keys=args.systems,
        workers=args.workers,
        first_ttft_seconds=args.first_ttft_seconds,
        resume_ttft_seconds=args.resume_ttft_seconds,
        tpot_milliseconds=args.tpot_milliseconds,
        scenario=scenario,
    )
    if args.dry_run:
        print(json.dumps(
            describe_sweep_plan(plan),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ))
        return 0

    already_complete = sum(
        os.path.lexists(task.output_dir)
        for task in plan.tasks
    )

    def report_progress(event: Mapping[str, object]) -> None:
        print(
            "completed "
            f"{event['completed']}/{event['total']} "
            f"rate={event['rate_text']} "
            f"seed={event['seed']} "
            f"system={event['system_key']}",
            flush=True,
        )

    manifest, path = run_sweep(
        plan,
        resume=args.resume,
        progress=report_progress,
    )
    print(json.dumps({
        "manifest": str(path),
        "manifest_sha256": _sha256_file(path),
        "cell_count": manifest["grid"]["cell_count"],
        "resumed_cell_count": already_complete,
        "executed_cell_count": (
            manifest["grid"]["cell_count"] - already_complete),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

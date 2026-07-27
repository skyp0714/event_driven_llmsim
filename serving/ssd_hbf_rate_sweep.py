"""TP8 arrival-rate sweep using the frozen SSD+HBF design selection.

The wrapper intentionally delegates each rate to
``ssd_hbf_design_sweep.run_design_space``.  It does not select policies from
rate-sweep results: the policy/read/restore coordinates are loaded from the
discovery-frozen selection manifest before any rate cell is launched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .core.hbf_comparison_cell import (
    DEFAULT_FIRST_TTFT_SECONDS,
    DEFAULT_RESUME_TTFT_SECONDS,
    DEFAULT_TPOT_MILLISECONDS,
    write_json_atomic,
)
from .core.hbf_comparison_workload import stable_json_sha256
from .core.tracelab_comparison_scenarios import (
    BALANCED_DEFAULT_RATES,
    TraceLabComparisonScenario,
    load_balanced_causal_prefix_scenario,
)
from .hbf_comparison_sweep import (
    default_trace_path,
    default_worker_count,
)
from .ssd_hbf_design_sweep import (
    SSDHBFDesignSpec,
    SSDHBFDesignSweepError,
    SUPPORTED_HBF_READ_MODES,
    SUPPORTED_RESTORE_EXECUTION_MODES,
    make_design_spec,
    parse_active_memory_spec,
    run_design_space,
    validate_scenario_contract,
)


SSD_HBF_RATE_SWEEP_SCHEMA_VERSION = 2
SSD_HBF_RATE_SWEEP_CONTRACT_KEY = (
    "ssd-hbf-tp8-frozen-selection-rate-sweep-v2")
TP8_CONTEXT_LAYOUT = "tp8_context"
DEFAULT_FROZEN_SELECTION_PATH = Path(
    "configs/experiments/ssd_hbf_final_selection.json")
DEFAULT_ACTIVE_MEMORY_SPEC = "lpddr:16:409.6"
RATE_SWEEP_MANIFEST_NAME = "rate_sweep_manifest.json"


@dataclass(frozen=True)
class FrozenTP8Selection:
    """Validated four-coordinate TP8 projection of the frozen selection."""

    path: Path
    sha256: str
    schema_version: int
    selection_status: str
    designs: tuple[SSDHBFDesignSpec, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_selection_path(
        repo_root: Path,
        selection_path: Path,
) -> Path:
    path = Path(selection_path).expanduser()
    if not path.is_absolute():
        path = Path(repo_root) / path
    path = path.resolve()
    if not path.is_file():
        raise SSDHBFDesignSweepError(
            f"frozen selection is not a file: {path}")
    return path


def _selection_display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return str(path)


def load_frozen_tp8_selection(
        *,
        repo_root: Path,
        selection_path: Path = DEFAULT_FROZEN_SELECTION_PATH,
) -> FrozenTP8Selection:
    """Load exactly the four TP8 coordinates frozen before held-out runs."""

    root = Path(repo_root).resolve()
    path = _resolve_selection_path(root, selection_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SSDHBFDesignSweepError(
            f"cannot load frozen selection {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SSDHBFDesignSweepError(
            "frozen selection root must be a JSON object")
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version <= 0
    ):
        raise SSDHBFDesignSweepError(
            "frozen selection has an invalid schema_version")
    status = payload.get("selection_status")
    if status != "frozen_before_heldout":
        raise SSDHBFDesignSweepError(
            "selection_status must be 'frozen_before_heldout'")
    policies = payload.get("migration_policies")
    if (
        not isinstance(policies, list)
        or len(policies) != 2
        or any(not isinstance(policy, str) for policy in policies)
        or len(set(policies)) != len(policies)
    ):
        raise SSDHBFDesignSweepError(
            "frozen selection must declare two distinct migration policies")
    guard_ms = payload.get("mixed_batch_latency_limit_ms")
    if (
        guard_ms is not None
        and (
            isinstance(guard_ms, bool)
            or not isinstance(guard_ms, int)
            or guard_ms <= 0
        )
    ):
        raise SSDHBFDesignSweepError(
            "mixed_batch_latency_limit_ms must be positive or null")
    coordinates = payload.get("restore_by_coordinate")
    if not isinstance(coordinates, list):
        raise SSDHBFDesignSweepError(
            "restore_by_coordinate must be a JSON array")
    tp8_rows = [
        row for row in coordinates
        if (
            isinstance(row, Mapping)
            and row.get("hbf_layout") == TP8_CONTEXT_LAYOUT
        )
    ]
    if len(tp8_rows) != 4:
        raise SSDHBFDesignSweepError(
            "frozen selection must contain exactly four TP8 coordinates")
    expected_axes = {
        (policy, read_mode)
        for policy in policies
        for read_mode in SUPPORTED_HBF_READ_MODES
    }
    observed_axes = set()
    memory = parse_active_memory_spec(DEFAULT_ACTIVE_MEMORY_SPEC)
    designs = []
    for row in tp8_rows:
        policy = row.get("migration_policy")
        read_mode = row.get("hbf_read_mode")
        restore_mode = row.get("restore_execution_mode")
        if not isinstance(policy, str) or not isinstance(read_mode, str):
            raise SSDHBFDesignSweepError(
                "TP8 policy and read mode must be strings")
        if restore_mode not in SUPPORTED_RESTORE_EXECUTION_MODES:
            raise SSDHBFDesignSweepError(
                f"unsupported frozen restore mode {restore_mode!r}")
        axis = (policy, read_mode)
        if axis in observed_axes:
            raise SSDHBFDesignSweepError(
                f"duplicate frozen TP8 coordinate {axis!r}")
        observed_axes.add(axis)
        designs.append(make_design_spec(
            hbf_layout=TP8_CONTEXT_LAYOUT,
            migration_policy=policy,
            active_memory=memory,
            hbf_read_mode=read_mode,
            restore_execution_mode=str(restore_mode),
            mixed_batch_latency_limit_ms=guard_ms,
        ))
    if observed_axes != expected_axes:
        raise SSDHBFDesignSweepError(
            "frozen TP8 coordinates do not form the declared "
            f"policy/read grid: missing={sorted(expected_axes - observed_axes)}, "
            f"extra={sorted(observed_axes - expected_axes)}")
    designs.sort(
        key=lambda design: (
            design.migration_policy,
            design.hbf_read_mode,
            design.restore_execution_mode,
        )
    )
    return FrozenTP8Selection(
        path=path,
        sha256=_sha256_file(path),
        schema_version=schema_version,
        selection_status=status,
        designs=tuple(designs),
    )


def _validated_rates(
        scenario: TraceLabComparisonScenario,
        rates: Sequence[float],
) -> tuple[float, ...]:
    values = []
    for value in rates:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise SSDHBFDesignSweepError(
                "rates must be positive finite numbers")
        rate = float(value)
        validate_scenario_contract(
            scenario, session_rate=rate)
        values.append(rate)
    if not values:
        raise SSDHBFDesignSweepError("rates cannot be empty")
    if len(values) != len(set(values)):
        raise SSDHBFDesignSweepError("rates contain duplicates")
    return tuple(sorted(values))


def _validated_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    values = tuple(seeds)
    if len(values) < 2:
        raise SSDHBFDesignSweepError(
            "at least two seeds are required for paired confidence intervals")
    if (
        any(isinstance(seed, bool) or not isinstance(seed, int)
            for seed in values)
        or len(values) != len(set(values))
    ):
        raise SSDHBFDesignSweepError(
            "seeds must be distinct integers")
    return tuple(sorted(values))


def run_rate_sweep(
        *,
        repo_root: Path,
        output_root: Path,
        scenario: TraceLabComparisonScenario,
        selection_path: Path = DEFAULT_FROZEN_SELECTION_PATH,
        rates: Sequence[float] = BALANCED_DEFAULT_RATES,
        seeds: Sequence[int] = (201, 202, 203),
        workers: int = 1,
        first_ttft_seconds: float = DEFAULT_FIRST_TTFT_SECONDS,
        resume_ttft_seconds: float = DEFAULT_RESUME_TTFT_SECONDS,
        tpot_milliseconds: float = DEFAULT_TPOT_MILLISECONDS,
        resume: bool = False,
        require_eligibility: bool = False,
        require_runtime_energy: bool = True,
) -> tuple[dict[str, object], Path]:
    """Run one existing single-rate design space per validated rate."""

    root = Path(repo_root).resolve()
    output = Path(output_root).expanduser().resolve()
    selection = load_frozen_tp8_selection(
        repo_root=root,
        selection_path=selection_path,
    )
    rate_values = _validated_rates(scenario, rates)
    seed_values = _validated_seeds(seeds)
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers <= 0
    ):
        raise SSDHBFDesignSweepError(
            "workers must be a positive integer")

    output.mkdir(parents=True, exist_ok=True)
    rate_aggregates = []
    scenario_contract = validate_scenario_contract(
        scenario, session_rate=rate_values[0])
    execution_inputs_sha256 = None
    for rate in rate_values:
        rate_root = output / f"rate-{rate:g}"
        aggregate, aggregate_path = run_design_space(
            repo_root=root,
            output_root=rate_root,
            scenario=scenario,
            designs=selection.designs,
            seeds=seed_values,
            workers=workers,
            session_rate=rate,
            first_ttft_seconds=first_ttft_seconds,
            resume_ttft_seconds=resume_ttft_seconds,
            tpot_milliseconds=tpot_milliseconds,
            resume=resume,
            require_eligibility=require_eligibility,
            require_runtime_energy=require_runtime_energy,
        )
        aggregate_path = Path(aggregate_path).resolve()
        aggregate_rates = aggregate.get("rates")
        if (
            not isinstance(aggregate_rates, list)
            or len(aggregate_rates) != 1
            or float(aggregate_rates[0].get("session_rate")) != rate
        ):
            raise SSDHBFDesignSweepError(
                f"per-rate aggregate disagrees with requested rate {rate:g}")
        aggregate_scenario = aggregate.get("scenario")
        if not isinstance(aggregate_scenario, Mapping):
            raise SSDHBFDesignSweepError(
                f"per-rate aggregate lacks scenario provenance at rate "
                f"{rate:g}")
        scenario_fields = (
            "scenario_id",
            "scenario_manifest_type",
            "manifest_sha256",
            "measurement_roster_sha256",
            "measurement_identity_count",
            "declared_session_rates",
        )
        mismatched_scenario_fields = [
            field for field in scenario_fields
            if aggregate_scenario.get(field) != scenario_contract.get(field)
        ]
        if aggregate_scenario.get("required_session_rate") != rate:
            mismatched_scenario_fields.append("required_session_rate")
        if mismatched_scenario_fields:
            raise SSDHBFDesignSweepError(
                f"per-rate scenario provenance mismatch at rate {rate:g}: "
                f"{sorted(set(mismatched_scenario_fields))}")
        aggregate_execution_hash = aggregate.get(
            "execution_inputs_sha256")
        if (
            not isinstance(aggregate_execution_hash, str)
            or len(aggregate_execution_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in aggregate_execution_hash
            )
        ):
            raise SSDHBFDesignSweepError(
                f"per-rate aggregate has invalid execution input hash at "
                f"rate {rate:g}")
        if execution_inputs_sha256 is None:
            execution_inputs_sha256 = aggregate_execution_hash
        elif aggregate_execution_hash != execution_inputs_sha256:
            raise SSDHBFDesignSweepError(
                "execution source or hardware config changed between "
                "rate points")
        try:
            relative_path = aggregate_path.relative_to(output)
        except ValueError as exc:
            raise SSDHBFDesignSweepError(
                "per-rate aggregate escaped the rate-sweep output root"
            ) from exc
        rate_aggregates.append({
            "session_rate": rate,
            "relative_path": relative_path.as_posix(),
            "sha256": _sha256_file(aggregate_path),
        })

    manifest = {
        "schema_version": SSD_HBF_RATE_SWEEP_SCHEMA_VERSION,
        "rate_sweep_contract": SSD_HBF_RATE_SWEEP_CONTRACT_KEY,
        "scenario": {
            "scenario_id": scenario_contract["scenario_id"],
            "manifest_type": scenario_contract[
                "scenario_manifest_type"],
            "manifest_sha256": scenario_contract["manifest_sha256"],
            "measurement_roster_sha256": scenario_contract[
                "measurement_roster_sha256"],
            "measurement_identity_count": scenario_contract[
                "measurement_identity_count"],
            "declared_session_rates": scenario_contract[
                "declared_session_rates"],
        },
        "hbf_layout": TP8_CONTEXT_LAYOUT,
        "selection": {
            "path": _selection_display_path(
                selection.path, root),
            "sha256": selection.sha256,
            "schema_version": selection.schema_version,
            "selection_status": selection.selection_status,
        },
        "rates": list(rate_values),
        "seeds": list(seed_values),
        "designs": [
            design.to_json_dict() for design in selection.designs],
        "rate_aggregates": rate_aggregates,
        "execution_inputs_sha256": execution_inputs_sha256,
        "reference_eligibility_required": require_eligibility,
        "runtime_energy_tco_required": require_runtime_energy,
    }
    manifest["manifest_payload_sha256"] = stable_json_sha256(
        manifest)
    manifest_path = output / RATE_SWEEP_MANIFEST_NAME
    write_json_atomic(manifest_path, manifest)
    return manifest, manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the four discovery-frozen TP8 SSD+HBF coordinates over "
            "the balanced causal-prefix scenario's arrival-rate anchors."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--trace", type=Path, default=default_trace_path())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        default=DEFAULT_FROZEN_SELECTION_PATH,
    )
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=None,
        help=(
            "arrival rates; defaults to all anchors declared by the "
            "loaded balanced scenario"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "paired seeds; defaults to heldout.seeds in the frozen "
            "selection"
        ),
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--first-ttft-seconds",
        type=float,
        default=DEFAULT_FIRST_TTFT_SECONDS,
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--require-reference-eligibility",
        action="store_true",
        help=(
            "abort when a per-rate baseline/Oracle audit gate fails; "
            "the default retains every rate as a labelled audit"
        ),
    )
    return parser


def _selection_default_seeds(path: Path) -> tuple[int, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        heldout = payload["heldout"]
        seeds = heldout["seeds"]
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as exc:
        raise SSDHBFDesignSweepError(
            "cannot read heldout.seeds from frozen selection") from exc
    return _validated_seeds(seeds)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    selection_path = _resolve_selection_path(
        repo_root, args.selection)
    scenario = load_balanced_causal_prefix_scenario(args.trace)
    rates = (
        tuple(scenario.manifest.arrival_contract.rates)
        if args.rates is None else tuple(args.rates)
    )
    seeds = (
        _selection_default_seeds(selection_path)
        if args.seeds is None else tuple(args.seeds)
    )
    workers = (
        default_worker_count()
        if args.workers is None else args.workers
    )

    manifest, path = run_rate_sweep(
        repo_root=repo_root,
        output_root=args.output,
        scenario=scenario,
        selection_path=selection_path,
        rates=rates,
        seeds=seeds,
        workers=workers,
        first_ttft_seconds=args.first_ttft_seconds,
        resume_ttft_seconds=args.resume_ttft_seconds,
        tpot_milliseconds=args.tpot_milliseconds,
        resume=args.resume,
        require_eligibility=args.require_reference_eligibility,
    )
    print(json.dumps({
        "manifest": str(path),
        "manifest_payload_sha256": manifest[
            "manifest_payload_sha256"],
        "rates": manifest["rates"],
        "design_count": len(manifest["designs"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ACTIVE_MEMORY_SPEC",
    "DEFAULT_FROZEN_SELECTION_PATH",
    "FrozenTP8Selection",
    "RATE_SWEEP_MANIFEST_NAME",
    "SSD_HBF_RATE_SWEEP_CONTRACT_KEY",
    "SSD_HBF_RATE_SWEEP_SCHEMA_VERSION",
    "TP8_CONTEXT_LAYOUT",
    "load_frozen_tp8_selection",
    "run_rate_sweep",
]

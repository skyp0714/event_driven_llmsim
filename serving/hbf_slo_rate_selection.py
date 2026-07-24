"""Adapt a validated HBF comparison sweep to the SLO rate selector.

This module deliberately starts from
:func:`serving.hbf_comparison_plots.load_validated_sweep`.  It does not parse
cell JSON independently or weaken that loader's artifact, pairing, full-drain,
metric-formula, and provenance checks.

The comparison sweep predates the generic rate selector, so a few selector
identity fields do not exist under the same names.  Every such field is either
mapped from an exact sweep digest or content-addressed from an explicit payload
recorded in the output artifact.  Scenario equilibrium is never inferred:
``scenario.manifest.equilibrium_workload`` must exist and be a boolean.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .core.hbf_comparison_cell import (
    CELL_SCHEMA_VERSION,
    write_json_atomic,
)
from .core.hbf_comparison_workload import stable_json_sha256
from .core.hbf_slo_rate_selection import (
    RATE_SELECTION_SCHEMA_VERSION,
    SCENARIO_FAMILY_BALANCED,
    SCENARIO_FAMILY_LONG_COLD,
    RateGridManifestIdentity,
    SeedRateMetricRow,
    SystemProvenanceIdentity,
    select_rate_grid_operating_points,
)
from .hbf_comparison_plots import (
    AGGREGATE_SCHEMA_VERSION,
    METRIC_BY_KEY,
    ComparisonAggregate,
    SeedCellMetrics,
    ValidatedSweep,
    aggregate_validated_sweep,
    load_validated_sweep,
)
from .hbf_comparison_sweep import SWEEP_SCHEMA_VERSION


ADAPTER_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_FILENAME = "slo_rate_selection.json"
RESULT_SCHEMA_REVISION = (
    f"hbf-comparison-sweep-v{SWEEP_SCHEMA_VERSION}/"
    f"cell-v{CELL_SCHEMA_VERSION}/"
    f"plot-aggregate-v{AGGREGATE_SCHEMA_VERSION}/"
    f"seed-metric-adapter-v{ADAPTER_SCHEMA_VERSION}"
)

_BALANCED_SCENARIO_ID = re.compile(
    r"^tracelab-balanced-(?P<calls>[1-9][0-9]*)-call-causal-prefix-v1$")
_LONG_COLD_SCENARIO_ID = re.compile(
    r"^tracelab-long-cold-(?P<threshold>[1-9][0-9]*)-cached-"
    r"native-prefix-v1$"
)
_SELECTOR_METRICS = (
    "joint_slo_pass_fraction",
    "slo_request_goodput_per_second",
    "slo_output_token_goodput_per_second",
)


class HBFSLORateSelectionAdapterError(RuntimeError):
    """Raised when a validated sweep lacks an exact selector contract."""


def _fail(message: str) -> None:
    raise HBFSLORateSelectionAdapterError(message)


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{context} must be an object")
    return value


def _positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{context} must be a positive integer")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{context} must be a lowercase SHA-256 digest")
    return value


def _derive_scenario_family(
        scenario_id: object,
        scenario_manifest: Mapping[str, object],
) -> tuple[str, Mapping[str, object]]:
    """Return a family only when exact serialized discriminators agree."""

    if not isinstance(scenario_id, str) or not scenario_id:
        _fail("scenario.scenario_id must be a non-empty string")
    if scenario_manifest.get("scenario_id") != scenario_id:
        _fail("scenario manifest ID differs from scenario.scenario_id")

    balanced = _BALANCED_SCENARIO_ID.fullmatch(scenario_id)
    long_cold = _LONG_COLD_SCENARIO_ID.fullmatch(scenario_id)
    if balanced is not None:
        calls = _positive_integer(
            scenario_manifest.get("calls_per_session"),
            "scenario.manifest.calls_per_session",
        )
        if calls != int(balanced.group("calls")):
            _fail(
                "balanced scenario ID and calls_per_session disagree")
        if "cached_prefix_threshold" in scenario_manifest:
            _fail(
                "balanced scenario unexpectedly has a long-cold "
                "cached_prefix_threshold discriminator"
            )
        return SCENARIO_FAMILY_BALANCED, {
            "rule": (
                "anchored balanced scenario_id plus exact "
                "calls_per_session agreement"
            ),
            "scenario_id": scenario_id,
            "calls_per_session": calls,
        }

    if long_cold is not None:
        threshold = _positive_integer(
            scenario_manifest.get("cached_prefix_threshold"),
            "scenario.manifest.cached_prefix_threshold",
        )
        if threshold != int(long_cold.group("threshold")):
            _fail(
                "long-cold scenario ID and cached_prefix_threshold disagree")
        successor_count = scenario_manifest.get("successor_call_count")
        if (
            isinstance(successor_count, bool)
            or not isinstance(successor_count, int)
            or successor_count < 0
        ):
            _fail(
                "scenario.manifest.successor_call_count must be a "
                "non-negative integer"
            )
        if "calls_per_session" in scenario_manifest:
            _fail(
                "long-cold scenario unexpectedly has a balanced "
                "calls_per_session discriminator"
            )
        return SCENARIO_FAMILY_LONG_COLD, {
            "rule": (
                "anchored long-cold scenario_id plus exact cached-prefix "
                "threshold agreement"
            ),
            "scenario_id": scenario_id,
            "cached_prefix_threshold": threshold,
            "successor_call_count": successor_count,
        }

    _fail(
        "scenario family is not explicitly derivable from a supported "
        "TraceLab scenario ID and discriminator fields"
    )


def _metric_contract_payload() -> Mapping[str, object]:
    metrics = []
    for key in _SELECTOR_METRICS:
        spec = METRIC_BY_KEY.get(key)
        if spec is None:
            _fail(f"validated plot metric {key!r} is unavailable")
        metrics.append({
            "key": spec.key,
            "cell_source_path": list(spec.source_path),
            "cell_value_scale": spec.scale,
            "unit": spec.unit,
            "bounded_fraction": spec.bounded_fraction,
        })
    return {
        "replicate": "one_seed_level_cell_summary",
        "request_rows_pooled": False,
        "seed_aggregation": (
            "serving.core.hbf_comparison_metrics.aggregate_seed_values"
        ),
        "metrics": metrics,
        "validated_goodput_formulas": {
            "slo_request_goodput_per_second": (
                "session_rate * measured_calls / measured_sessions "
                "* all_SLO_pass_fraction"
            ),
            "slo_output_token_goodput_per_second": (
                "session_rate * all_SLO_pass_output_tokens "
                "/ measured_sessions"
            ),
        },
    }


def _slo_contract_payload(
        thresholds: Mapping[str, object],
) -> Mapping[str, object]:
    expected = {"first_ttft_ns", "resume_ttft_ns", "tpot_ns"}
    if set(thresholds) != expected:
        _fail("slo_thresholds_ns has an unexpected field set")
    parsed = {
        key: _positive_integer(
            thresholds[key], f"slo_thresholds_ns.{key}")
        for key in sorted(expected)
    }
    spec = METRIC_BY_KEY.get("joint_slo_pass_fraction")
    if spec is None:
        _fail("joint SLO metric is unavailable from the validated sweep")
    return {
        "metric_scope": "all",
        "thresholds_ns": parsed,
        "joint_slo_metric": {
            "key": spec.key,
            "cell_source_path": list(spec.source_path),
            "cell_value_scale": spec.scale,
        },
    }


def _system_provenance_payloads(
        manifest: Mapping[str, object],
        system_keys: Sequence[str],
) -> tuple[
    tuple[SystemProvenanceIdentity, ...],
    Mapping[str, Mapping[str, object]],
]:
    configs = _mapping(
        manifest.get("system_config_contracts"),
        "system_config_contracts",
    )
    code = _mapping(
        manifest.get("code_revision_hashes"),
        "code_revision_hashes",
    )
    execution = _mapping(manifest.get("execution"), "execution")
    execution_code_sha256 = _sha256(
        code.get("execution_code_sha256"),
        "code_revision_hashes.execution_code_sha256",
    )
    if set(configs) != set(system_keys):
        _fail("system configuration roster differs from validated sweep")

    identities = []
    payloads = {}
    for system_key in system_keys:
        config = _mapping(
            configs[system_key],
            f"system_config_contracts.{system_key}",
        )
        if config.get("system_key") != system_key:
            _fail(f"system config key mismatch for {system_key!r}")
        payload = {
            "system_key": system_key,
            "system_config_contract_sha256": stable_json_sha256(config),
            "execution_code_sha256": execution_code_sha256,
            "simulation_backend": execution.get("simulation_backend"),
            "astra_cycles_used": execution.get("astra_cycles_used"),
            "sweep_schema_version": SWEEP_SCHEMA_VERSION,
            "cell_schema_version": CELL_SCHEMA_VERSION,
        }
        digest = stable_json_sha256(payload)
        identities.append(SystemProvenanceIdentity(
            system_key=system_key,
            provenance_sha256=digest,
        ))
        payloads[system_key] = {
            **payload,
            "provenance_sha256": digest,
        }
    return tuple(identities), payloads


def _index_pairs(
        manifest: Mapping[str, object],
        sweep: ValidatedSweep,
) -> Mapping[tuple[int, str], Mapping[str, object]]:
    pairing = _mapping(manifest.get("pairing"), "pairing")
    raw_pairs = pairing.get("schedule_pairs")
    if not isinstance(raw_pairs, list):
        _fail("pairing.schedule_pairs must be an array")
    indexed = {}
    for index, raw in enumerate(raw_pairs):
        pair = _mapping(raw, f"pairing.schedule_pairs[{index}]")
        seed = pair.get("seed")
        rate_text = pair.get("rate_text")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(rate_text, str)
            or not rate_text
        ):
            _fail(f"pairing.schedule_pairs[{index}] has invalid coordinates")
        coordinate = (seed, rate_text)
        if coordinate in indexed:
            _fail(f"duplicate schedule pair coordinate {coordinate!r}")
        indexed[coordinate] = pair
    expected = {
        (seed, rate_text)
        for seed in sweep.seeds
        for rate_text in sweep.rate_texts
    }
    if set(indexed) != expected:
        _fail("schedule-pair grid differs from validated sweep")
    return indexed


def _index_cell_records(
        manifest: Mapping[str, object],
        sweep: ValidatedSweep,
) -> Mapping[tuple[int, str, str], Mapping[str, object]]:
    records = manifest.get("cells")
    if not isinstance(records, list):
        _fail("cells must be an array")
    indexed = {}
    for index, raw in enumerate(records):
        record = _mapping(raw, f"cells[{index}]")
        seed = record.get("seed")
        rate_text = record.get("rate_text")
        system_key = record.get("system_key")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(rate_text, str)
            or not isinstance(system_key, str)
        ):
            _fail(f"cells[{index}] has invalid coordinates")
        coordinate = (seed, rate_text, system_key)
        if coordinate in indexed:
            _fail(f"duplicate cell record coordinate {coordinate!r}")
        indexed[coordinate] = record
    expected = {
        (seed, rate_text, system_key)
        for seed in sweep.seeds
        for rate_text in sweep.rate_texts
        for system_key in sweep.system_keys
    }
    if set(indexed) != expected:
        _fail("cell-record grid differs from validated sweep")
    return indexed


def _unit_rate_plan_hashes(
        pairs: Mapping[tuple[int, str], Mapping[str, object]],
        sweep: ValidatedSweep,
        scenario_id: str,
) -> tuple[Mapping[int, str], Mapping[int, Mapping[str, object]]]:
    hashes = {}
    payloads = {}
    for seed in sweep.seeds:
        observed = []
        for rate_text in sweep.rate_texts:
            pair = pairs[(seed, rate_text)]
            payload = {
                "scenario_id": scenario_id,
                "seed": seed,
                "offered_session_ids_sha256": _sha256(
                    pair.get("offered_session_ids_sha256"),
                    "pair.offered_session_ids_sha256",
                ),
                "unit_draws_sha256": _sha256(
                    pair.get("unit_draws_sha256"),
                    "pair.unit_draws_sha256",
                ),
            }
            observed.append((stable_json_sha256(payload), payload))
        unique = {digest for digest, _ in observed}
        if len(unique) != 1:
            _fail(
                f"seed {seed} does not have one exact unit-rate plan "
                "across all rates"
            )
        digest, payload = observed[0]
        hashes[seed] = digest
        payloads[seed] = {**payload, "unit_rate_plan_sha256": digest}
    return hashes, payloads


def _validate_aggregate(
        sweep: ValidatedSweep,
        aggregate: ComparisonAggregate,
) -> None:
    if not isinstance(sweep, ValidatedSweep):
        _fail("sweep must be a ValidatedSweep returned by the strict loader")
    if not isinstance(aggregate, ComparisonAggregate):
        _fail("aggregate must be a ComparisonAggregate")
    expected = aggregate_validated_sweep(sweep)
    if aggregate != expected:
        _fail(
            "aggregate does not exactly equal aggregate_validated_sweep("
            "validated_sweep)"
        )


def build_rate_selection_artifact(
        sweep: ValidatedSweep,
        aggregate: ComparisonAggregate,
) -> Mapping[str, object]:
    """Build a self-hashed selector artifact from strict sweep objects."""

    _validate_aggregate(sweep, aggregate)
    manifest = _mapping(sweep.manifest, "validated sweep manifest")
    source_manifest_sha256 = _sha256(
        sweep.manifest_sha256, "validated sweep manifest SHA-256")
    source_cells_sha256 = _sha256(
        aggregate.source_cells_sha256,
        "validated sweep cells SHA-256",
    )
    if manifest.get("cells_sha256") != source_cells_sha256:
        _fail("aggregate cell-list hash differs from the sweep manifest")
    if (
        not isinstance(aggregate.simulation_backend, str)
        or not aggregate.simulation_backend
        or not isinstance(aggregate.astra_cycles_used, bool)
    ):
        _fail("aggregate backend provenance is invalid")
    scenario = _mapping(manifest.get("scenario"), "scenario")
    scenario_manifest = _mapping(
        scenario.get("manifest"), "scenario.manifest")
    scenario_id = scenario.get("scenario_id")
    family, family_derivation = _derive_scenario_family(
        scenario_id, scenario_manifest)
    scenario_schema = _positive_integer(
        scenario_manifest.get("schema_version"),
        "scenario.manifest.schema_version",
    )
    equilibrium = scenario_manifest.get("equilibrium_workload")
    if not isinstance(equilibrium, bool):
        _fail(
            "scenario.manifest.equilibrium_workload must be an explicit "
            "boolean; it is never inferred"
        )
    scenario_manifest_sha256 = _sha256(
        scenario.get("manifest_sha256"),
        "scenario.manifest_sha256",
    )
    if stable_json_sha256(scenario_manifest) != scenario_manifest_sha256:
        _fail("scenario manifest content hash changed after validation")

    pairing = _mapping(manifest.get("pairing"), "pairing")
    measurement_roster_sha256 = _sha256(
        pairing.get("measurement_identities_sha256"),
        "pairing.measurement_identities_sha256",
    )
    scenario_measurement_sha256 = _sha256(
        scenario_manifest.get("measurement_request_identities_sha256"),
        "scenario.manifest.measurement_request_identities_sha256",
    )
    if scenario_measurement_sha256 != measurement_roster_sha256:
        _fail(
            "scenario measurement roster differs from paired sweep roster")

    thresholds = _mapping(
        manifest.get("slo_thresholds_ns"), "slo_thresholds_ns")
    slo_payload = _slo_contract_payload(thresholds)
    metric_payload = _metric_contract_payload()
    slo_contract_sha256 = stable_json_sha256(slo_payload)
    metric_contract_sha256 = stable_json_sha256(metric_payload)
    system_identities, system_payloads = _system_provenance_payloads(
        manifest, sweep.system_keys)

    selector_manifest = RateGridManifestIdentity(
        schema_version=RATE_SELECTION_SCHEMA_VERSION,
        scenario_family=family,
        scenario_id=str(scenario_id),
        scenario_manifest_schema_version=scenario_schema,
        scenario_manifest_sha256=scenario_manifest_sha256,
        equilibrium_workload=equilibrium,
        measurement_roster_sha256=measurement_roster_sha256,
        metric_scope="all",
        slo_contract_sha256=slo_contract_sha256,
        metric_contract_sha256=metric_contract_sha256,
        result_schema_revision=RESULT_SCHEMA_REVISION,
        system_keys=tuple(sweep.system_keys),
        rates=tuple(sweep.rates),
        seed_ids=tuple(sweep.seeds),
        system_provenance=system_identities,
    )

    pairs = _index_pairs(manifest, sweep)
    cell_records = _index_cell_records(manifest, sweep)
    plan_hashes, plan_payloads = _unit_rate_plan_hashes(
        pairs, sweep, str(scenario_id))
    system_hashes = selector_manifest.system_provenance_by_key
    rows = []
    for cell in sweep.cells:
        if not isinstance(cell, SeedCellMetrics):
            _fail("validated sweep cells must be SeedCellMetrics values")
        pair = pairs[(cell.seed, cell.rate_text)]
        record = cell_records[
            (cell.seed, cell.rate_text, cell.system_key)]
        schedule_sha256 = _sha256(
            pair.get("schedule_sha256"), "pair.schedule_sha256")
        if cell.schedule_pair_sha256 != pair.get("schedule_pair_sha256"):
            _fail("validated cell schedule pair changed after validation")
        rows.append(SeedRateMetricRow(
            scenario_id=str(scenario_id),
            scenario_manifest_sha256=scenario_manifest_sha256,
            measurement_roster_sha256=measurement_roster_sha256,
            metric_scope="all",
            slo_contract_sha256=slo_contract_sha256,
            metric_contract_sha256=metric_contract_sha256,
            result_schema_revision=RESULT_SCHEMA_REVISION,
            system_key=cell.system_key,
            system_provenance_sha256=system_hashes[cell.system_key],
            offered_session_rate=cell.session_rate,
            seed_id=cell.seed,
            unit_rate_plan_sha256=plan_hashes[cell.seed],
            rate_scaled_schedule_sha256=schedule_sha256,
            # The sweep has no field named "cell manifest".  Its exact,
            # unique, content-addressed cell contract is the closest and
            # stricter identity, so the mapping is disclosed below.
            cell_manifest_sha256=_sha256(
                record.get("cell_contract_sha256"),
                "cell.cell_contract_sha256",
            ),
            joint_slo_pass_fraction=cell.values[
                "joint_slo_pass_fraction"],
            slo_request_goodput_per_second=cell.values[
                "slo_request_goodput_per_second"],
            slo_output_token_goodput_per_second=cell.values[
                "slo_output_token_goodput_per_second"],
        ))
    selection = select_rate_grid_operating_points(
        selector_manifest, rows)

    payload: dict[str, object] = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "source": {
            "validated_sweep_manifest_path": str(sweep.manifest_path),
            "validated_sweep_manifest_sha256": source_manifest_sha256,
            "validated_sweep_cells_sha256": source_cells_sha256,
            "simulation_backend": aggregate.simulation_backend,
            "astra_cycles_used": aggregate.astra_cycles_used,
        },
        "adapter_contract": {
            "input_validation": (
                "serving.hbf_comparison_plots.load_validated_sweep"
            ),
            "aggregate_validation": (
                "exact equality with aggregate_validated_sweep("
                "validated_sweep)"
            ),
            "scenario_family_derivation": family_derivation,
            "equilibrium_derivation": {
                "source_field": (
                    "scenario.manifest.equilibrium_workload"),
                "inferred": False,
                "value": equilibrium,
            },
            "measurement_roster_sha256": measurement_roster_sha256,
            "slo_contract": {
                "payload": slo_payload,
                "sha256": slo_contract_sha256,
            },
            "metric_contract": {
                "payload": metric_payload,
                "sha256": metric_contract_sha256,
            },
            "system_provenance": system_payloads,
            "unit_rate_plan_provenance_by_seed": {
                str(seed): plan_payloads[seed]
                for seed in sweep.seeds
            },
            "field_mappings": {
                "rate_scaled_schedule_sha256": (
                    "pairing.schedule_pairs[].schedule_sha256"
                ),
                "cell_manifest_sha256": (
                    "cells[].cell_contract_sha256"
                ),
            },
            "result_schema_revision": RESULT_SCHEMA_REVISION,
        },
        "selection": selection.to_dict(),
        "selection_sha256": stable_json_sha256(selection.to_dict()),
        "artifact_sha256_semantics": (
            "sha256_of_canonical_json_with_artifact_sha256_omitted"
        ),
    }
    payload["artifact_sha256"] = stable_json_sha256(payload)
    return payload


def write_rate_selection_artifact(
        path: str | Path,
        artifact: Mapping[str, object],
) -> Path:
    """Atomically write an already-built, self-consistent artifact."""

    value = _mapping(artifact, "artifact")
    expected = _sha256(
        value.get("artifact_sha256"), "artifact.artifact_sha256")
    unhashed = {
        key: item
        for key, item in value.items()
        if key != "artifact_sha256"
    }
    if stable_json_sha256(unhashed) != expected:
        _fail("artifact_sha256 is inconsistent")
    return write_json_atomic(Path(path), value)


def _parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate an HBF comparison sweep and write an "
            "auditable SLO rate-selection artifact."
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
        "--output",
        type=Path,
        default=None,
        help=(
            "output JSON path; defaults to "
            "<sweep-root>/slo_rate_selection.json"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    sweep = load_validated_sweep(
        args.sweep_root,
        repo_root=args.repo_root,
        metric_keys=_SELECTOR_METRICS,
    )
    aggregate = aggregate_validated_sweep(sweep)
    artifact = build_rate_selection_artifact(sweep, aggregate)
    output = (
        args.output
        if args.output is not None
        else Path(args.sweep_root) / DEFAULT_OUTPUT_FILENAME
    )
    written = write_rate_selection_artifact(output, artifact)
    print(json.dumps(
        {
            "output": str(written),
            "artifact_sha256": artifact["artifact_sha256"],
            "source_manifest_sha256": sweep.manifest_sha256,
            "scenario_family": (
                artifact["selection"]["manifest_identity"][
                    "scenario_family"]
            ),
            "equilibrium_workload": (
                artifact["selection"]["manifest_identity"][
                    "equilibrium_workload"]
            ),
            "systems": len(sweep.system_keys),
            "rates": len(sweep.rates),
            "seeds": len(sweep.seeds),
        },
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "DEFAULT_OUTPUT_FILENAME",
    "HBFSLORateSelectionAdapterError",
    "RESULT_SCHEMA_REVISION",
    "build_rate_selection_artifact",
    "main",
    "write_rate_selection_artifact",
]

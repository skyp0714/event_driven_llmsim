"""Audited TP8 rate-scaling plots for the SSD-staged HBF comparison.

The input is the immutable multi-rate manifest emitted by
``ssd_hbf_rate_sweep``.  This module never selects a winner from measured
rate points: it renders the four TP8 coordinates frozen in the input
selection, both finite-HBM baselines, and the performance-only Oracle.

The primary figure preserves the seven metrics used by the earlier HBF
comparison while placing offered session rate on the x-axis.  Runtime power,
five-year facility energy, five-year TCO, and HBF write endurance are emitted
as companion rate-scaling figures.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

from .core.hbf_comparison_workload import stable_json_sha256
from .ssd_hbf_design_sweep import (
    BASELINE_CANDIDATE_KEYS,
    ORACLE_CANDIDATE_KEY,
    SSD_HBF_CONTRACT_KEY,
    SSD_HBF_SWEEP_SCHEMA_VERSION,
    SUPPORTED_HBF_READ_MODES,
    SUPPORTED_RESTORE_EXECUTION_MODES,
)
from .ssd_hbf_final_plots import (
    CENTRAL_ENDURANCE_SCENARIO,
    PERFORMANCE_METRIC_SPECS,
    RUNTIME_TCO_REPORT_SCHEMA,
)
from .ssd_hbf_rate_sweep import (
    RATE_SWEEP_MANIFEST_NAME,
    SSD_HBF_RATE_SWEEP_CONTRACT_KEY,
    SSD_HBF_RATE_SWEEP_SCHEMA_VERSION,
    load_frozen_tp8_selection,
)


RATE_PLOT_SCHEMA_VERSION = 1
TP8_LAYOUT = "tp8_context"
EXPECTED_DESIGN_COUNT = 4

_PERFORMANCE_BY_KEY = {
    spec.aggregate_key: spec for spec in PERFORMANCE_METRIC_SPECS
}
RATE_PERFORMANCE_KEYS = (
    "slo_good_output_tokens_per_second",
    "joint_slo_pass_fraction",
    "observed_request_throughput_per_second",
    "first_ttft_p95_ns",
    "resume_ttft_p95_ns",
    "tpot_p95_ns",
    "slo_request_goodput_per_second",
)
RUNTIME_METRICS = (
    (
        "average_it_power_kw",
        "trace_average_it_power_w",
        "Average runtime IT power",
        "Average IT power (kW)",
        1e-3,
    ),
    (
        "five_year_facility_energy_mwh",
        "five_year_facility_energy_kwh",
        "Modeled 5-year facility energy",
        "5-year facility energy (MWh)",
        1e-3,
    ),
    (
        "five_year_tco_thousand_usd",
        "five_year_tco_usd",
        "Modeled 5-year TCO",
        "5-year TCO ($ thousands)",
        1e-3,
    ),
)
ENDURANCE_METRICS = (
    (
        "hottest_card_payload_write_tb_per_day",
        "Hottest-card recurring KV writes",
        "Payload writes (TB/day)",
    ),
    (
        "five_year_endurance_budget_percent",
        "5-year HBF endurance budget",
        "100K P/E, WAF=1 budget used (%)",
    ),
)

_REFERENCE_ORDER = (
    BASELINE_CANDIDATE_KEYS["bulk"],
    BASELINE_CANDIDATE_KEYS["layerwise_streaming"],
    ORACLE_CANDIDATE_KEY,
)
_REFERENCE_LABELS = {
    BASELINE_CANDIDATE_KEYS["bulk"]:
        "2×GPU-host baseline | bulk restore",
    BASELINE_CANDIDATE_KEYS["layerwise_streaming"]:
        "2×GPU-host baseline | streaming restore",
    ORACLE_CANDIDATE_KEY: "2×GPU-host infinite-HBM Oracle",
}


class SSDHBFRatePlotError(ValueError):
    """Raised when a rate sweep cannot support the claimed plot."""


@dataclass(frozen=True)
class Statistic:
    mean: float
    ci95_lower: Optional[float]
    ci95_upper: Optional[float]


@dataclass(frozen=True)
class RatePoint:
    session_rate: float
    performance: Mapping[str, Statistic]
    runtime: Mapping[str, Statistic]
    endurance: Mapping[str, Statistic]


@dataclass(frozen=True)
class RateSeries:
    key: str
    label: str
    kind: str
    migration_policy: Optional[str]
    hbf_read_mode: Optional[str]
    restore_execution_mode: Optional[str]
    points: tuple[RatePoint, ...]


@dataclass(frozen=True)
class LoadedRateSweep:
    source_path: Path
    source_file_sha256: str
    source_payload_sha256: str
    selection_file_sha256: str
    execution_inputs_sha256: str
    scenario_id: str
    rates: tuple[float, ...]
    seeds: tuple[int, ...]
    series: tuple[RateSeries, ...]
    reference_eligible_at_all_rates: bool
    reference_eligibility_failures: tuple[str, ...]
    aggregate_file_sha256_by_rate: Mapping[float, str]


@dataclass(frozen=True)
class RatePlotArtifacts:
    source_csv: Path
    artifact_manifest_json: Path
    performance_png: Optional[Path]
    runtime_power_energy_tco_png: Optional[Path]
    hbf_endurance_png: Optional[Path]
    rendered: bool
    matplotlib_available: bool
    source_manifest_sha256: str

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return {
            key: (
                str(item) if isinstance(item, Path) else item
            )
            for key, item in value.items()
        }


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SSDHBFRatePlotError(f"{path} must be an object")
    return value


def _sequence(value: object, path: str) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise SSDHBFRatePlotError(f"{path} must be an array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise SSDHBFRatePlotError(f"{path} must be a non-empty string")
    return value


def _finite(
        value: object,
        path: str,
        *,
        positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SSDHBFRatePlotError(f"{path} must be finite")
    converted = float(value)
    if positive and converted <= 0.0:
        raise SSDHBFRatePlotError(f"{path} must be positive")
    return converted


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise SSDHBFRatePlotError(
            f"required JSON cannot be a symlink: {unresolved}")
    target = unresolved.resolve()
    if not target.is_file():
        raise SSDHBFRatePlotError(
            f"required JSON must be a regular file: {target}")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicates(
            pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = target.read_bytes()
        value = json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SSDHBFRatePlotError(
            f"cannot load strict JSON {target}: {exc}") from exc
    return dict(_mapping(value, str(target))), payload


def _rate_tuple(value: object, path: str) -> tuple[float, ...]:
    rates = tuple(
        _finite(item, f"{path}[{index}]", positive=True)
        for index, item in enumerate(_sequence(value, path))
    )
    if len(rates) < 2:
        raise SSDHBFRatePlotError(
            f"{path} must contain at least two rates")
    if rates != tuple(sorted(rates)) or len(rates) != len(set(rates)):
        raise SSDHBFRatePlotError(
            f"{path} must be strictly increasing and unique")
    return rates


def _seed_tuple(value: object, path: str) -> tuple[int, ...]:
    seeds = tuple(_sequence(value, path))
    if (
        len(seeds) < 2
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in seeds
        )
        or seeds != tuple(sorted(seeds))
        or len(seeds) != len(set(seeds))
    ):
        raise SSDHBFRatePlotError(
            f"{path} must contain at least two sorted unique integers")
    return seeds


def _statistic(
        value: object,
        path: str,
        *,
        seeds: tuple[int, ...],
        scale: float = 1.0,
) -> Statistic:
    statistic = _mapping(value, path)
    if statistic.get("ci_method") != "student_t_95":
        raise SSDHBFRatePlotError(
            f"{path}.ci_method must be student_t_95")
    if tuple(_sequence(
            statistic.get("seed_ids"), f"{path}.seed_ids")) != seeds:
        raise SSDHBFRatePlotError(
            f"{path}.seed_ids do not match the rate sweep seeds")
    values = tuple(_sequence(statistic.get("values"), f"{path}.values"))
    if len(values) != len(seeds):
        raise SSDHBFRatePlotError(
            f"{path}.values must contain one value per seed")
    for index, item in enumerate(values):
        _finite(item, f"{path}.values[{index}]")
    mean = _finite(statistic.get("mean"), f"{path}.mean")
    lower = _finite(
        statistic.get("ci95_lower"), f"{path}.ci95_lower")
    upper = _finite(
        statistic.get("ci95_upper"), f"{path}.ci95_upper")
    if lower > mean or upper < mean:
        raise SSDHBFRatePlotError(
            f"{path} confidence interval does not contain its mean")
    return Statistic(
        mean=mean * scale,
        ci95_lower=lower * scale,
        ci95_upper=upper * scale,
    )


def _performance(
        value: object,
        path: str,
        *,
        seeds: tuple[int, ...],
) -> dict[str, Statistic]:
    metrics = _mapping(value, path)
    result = {}
    for key in RATE_PERFORMANCE_KEYS:
        spec = _PERFORMANCE_BY_KEY[key]
        raw = metrics.get(key)
        if raw is None:
            raise SSDHBFRatePlotError(
                f"{path}.{key} is required for the balanced rate study")
        statistic = _statistic(
            raw,
            f"{path}.{key}",
            seeds=seeds,
            scale=spec.scale,
        )
        if spec.positive and statistic.mean <= 0.0:
            raise SSDHBFRatePlotError(
                f"{path}.{key}.mean must be positive")
        if (
            spec.bounded_fraction
            and not 0.0 <= statistic.mean <= 1.0
        ):
            raise SSDHBFRatePlotError(
                f"{path}.{key}.mean must be in [0, 1]")
        result[key] = statistic
    return result


def _runtime(
        value: object,
        path: str,
        *,
        seeds: tuple[int, ...],
        prefix: str,
) -> dict[str, Statistic]:
    report = _mapping(value, path)
    if report.get("report_schema") != RUNTIME_TCO_REPORT_SCHEMA:
        raise SSDHBFRatePlotError(
            f"{path}.report_schema must be {RUNTIME_TCO_REPORT_SCHEMA!r}")
    aggregation = _mapping(
        report.get("aggregation"), f"{path}.aggregation")
    statistics = _mapping(
        aggregation.get("student_t_95_by_seed"),
        f"{path}.aggregation.student_t_95_by_seed",
    )
    result = {}
    for plot_key, report_key, _title, _label, scale in RUNTIME_METRICS:
        statistic_key = f"{prefix}_{report_key}"
        statistic = _statistic(
            statistics.get(statistic_key),
            f"{path}.aggregation.student_t_95_by_seed.{statistic_key}",
            seeds=seeds,
            scale=scale,
        )
        if statistic.mean <= 0.0:
            raise SSDHBFRatePlotError(
                f"{path}.{statistic_key}.mean must be positive")
        result[plot_key] = statistic
    return result


def _endurance(
        value: object,
        path: str,
) -> dict[str, Statistic]:
    report = _mapping(value, path)
    scenarios = _mapping(
        report.get("scenarios"), f"{path}.scenarios")
    central = _mapping(
        scenarios.get(CENTRAL_ENDURANCE_SCENARIO),
        f"{path}.scenarios.{CENTRAL_ENDURANCE_SCENARIO}",
    )
    cards = tuple(_sequence(
        central.get("cards"),
        f"{path}.scenarios.{CENTRAL_ENDURANCE_SCENARIO}.cards",
    ))
    if len(cards) != 8:
        raise SSDHBFRatePlotError(
            f"{path} must contain exactly eight TP8 HBF cards")
    writes = []
    budgets = []
    for index, raw_card in enumerate(cards):
        card = _mapping(raw_card, f"{path}.cards[{index}]")
        writes.append(_finite(
            card.get("payload_write_bytes_per_day"),
            f"{path}.cards[{index}].payload_write_bytes_per_day",
        ))
        budgets.append(_finite(
            card.get("service_lifetime_budget_fraction"),
            f"{path}.cards[{index}].service_lifetime_budget_fraction",
        ))
    if min(writes) < 0.0 or min(budgets) < 0.0:
        raise SSDHBFRatePlotError(
            f"{path} endurance values cannot be negative")
    return {
        "hottest_card_payload_write_tb_per_day": Statistic(
            max(writes) / 1e12, None, None),
        "five_year_endurance_budget_percent": Statistic(
            100.0 * max(budgets), None, None),
    }


def _design_label(design: Mapping[str, Any]) -> str:
    policy = _string(
        design.get("migration_policy"), "design.migration_policy")
    read_mode = _string(
        design.get("hbf_read_mode"), "design.hbf_read_mode")
    restore = _string(
        design.get("restore_execution_mode"),
        "design.restore_execution_mode",
    )
    restore_label = (
        "streaming" if restore == "layerwise_streaming" else "bulk")
    return (
        f"GPU+HBF TP8 | {policy.replace('_', ' ')} | "
        f"{read_mode} | {restore_label}"
    )


def _validate_top_designs(
        value: object,
) -> tuple[Mapping[str, Any], ...]:
    designs = tuple(
        _mapping(item, f"designs[{index}]")
        for index, item in enumerate(_sequence(value, "designs"))
    )
    if len(designs) != EXPECTED_DESIGN_COUNT:
        raise SSDHBFRatePlotError(
            f"rate sweep must freeze exactly {EXPECTED_DESIGN_COUNT} designs")
    keys = []
    coordinates = []
    for index, design in enumerate(designs):
        path = f"designs[{index}]"
        key = _string(design.get("key"), f"{path}.key")
        layout = _string(
            design.get("hbf_layout"), f"{path}.hbf_layout")
        policy = _string(
            design.get("migration_policy"), f"{path}.migration_policy")
        read_mode = _string(
            design.get("hbf_read_mode"), f"{path}.hbf_read_mode")
        restore = _string(
            design.get("restore_execution_mode"),
            f"{path}.restore_execution_mode",
        )
        if layout != TP8_LAYOUT:
            raise SSDHBFRatePlotError(
                "rate plots are TP8-only; TP4 coordinates are rejected")
        if read_mode not in SUPPORTED_HBF_READ_MODES:
            raise SSDHBFRatePlotError(
                f"{path} has unsupported HBF read mode")
        if restore not in SUPPORTED_RESTORE_EXECUTION_MODES:
            raise SSDHBFRatePlotError(
                f"{path} has unsupported restore mode")
        keys.append(key)
        coordinates.append((policy, read_mode))
    if len(keys) != len(set(keys)):
        raise SSDHBFRatePlotError("design keys must be unique")
    policies = {policy for policy, _read in coordinates}
    if (
        len(policies) != 2
        or set(coordinates)
        != {
            (policy, read_mode)
            for policy in policies
            for read_mode in SUPPORTED_HBF_READ_MODES
        }
    ):
        raise SSDHBFRatePlotError(
            "frozen TP8 grid must contain demand and prefetch for each "
            "of exactly two migration policies")
    return tuple(sorted(designs, key=lambda design: (
        str(design["migration_policy"]),
        str(design["hbf_read_mode"]),
    )))


def _aggregate_path(root: Path, relative_text: object) -> Path:
    text = _string(relative_text, "rate_aggregates[].relative_path")
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts:
        raise SSDHBFRatePlotError(
            "aggregate relative paths must stay below the manifest root")
    target = root / Path(*pure.parts)
    try:
        target.resolve().relative_to(root)
    except ValueError as exc:
        raise SSDHBFRatePlotError(
            "aggregate relative path escapes the manifest root") from exc
    return target


def _same_design_sets(
        expected: Sequence[Mapping[str, Any]],
        observed: object,
        path: str,
) -> None:
    rows = tuple(
        _mapping(item, f"{path}[{index}]")
        for index, item in enumerate(_sequence(observed, path))
    )
    expected_by_key = {
        str(design["key"]): stable_json_sha256(design)
        for design in expected
    }
    observed_by_key = {
        _string(design.get("key"), f"{path}[].key"):
        stable_json_sha256(design)
        for design in rows
    }
    if len(rows) != len(observed_by_key) or (
        expected_by_key != observed_by_key
    ):
        raise SSDHBFRatePlotError(
            f"{path} does not match the top-level frozen designs")


def load_rate_sweep(
        path: Path | str,
        *,
        repo_root: Optional[Path | str] = None,
) -> LoadedRateSweep:
    """Load every rate aggregate and fail closed on an incomplete TP8 grid."""

    source_input = Path(path).expanduser()
    manifest, payload = _strict_json(source_input)
    source_path = source_input.resolve()
    if manifest.get("schema_version") != SSD_HBF_RATE_SWEEP_SCHEMA_VERSION:
        raise SSDHBFRatePlotError(
            "unsupported SSD-HBF rate-sweep schema")
    if (
        manifest.get("rate_sweep_contract")
        != SSD_HBF_RATE_SWEEP_CONTRACT_KEY
    ):
        raise SSDHBFRatePlotError(
            "unexpected SSD-HBF rate-sweep contract")
    claimed_payload_hash = manifest.get("manifest_payload_sha256")
    if not _is_sha256(claimed_payload_hash):
        raise SSDHBFRatePlotError(
            "manifest_payload_sha256 must be a SHA-256 digest")
    unhashed = dict(manifest)
    unhashed.pop("manifest_payload_sha256", None)
    if stable_json_sha256(unhashed) != claimed_payload_hash:
        raise SSDHBFRatePlotError(
            "rate-sweep manifest payload hash mismatch")
    if manifest.get("hbf_layout") != TP8_LAYOUT:
        raise SSDHBFRatePlotError(
            "rate-scaling plots require hbf_layout=tp8_context")
    selection = _mapping(manifest.get("selection"), "selection")
    selection_path = _string(
        selection.get("path"), "selection.path")
    if not _is_sha256(selection.get("sha256")):
        raise SSDHBFRatePlotError(
            "selection.sha256 must be a SHA-256 digest")
    selection_schema = selection.get("schema_version")
    if (
        isinstance(selection_schema, bool)
        or not isinstance(selection_schema, int)
        or selection_schema <= 0
    ):
        raise SSDHBFRatePlotError(
            "selection.schema_version must be a positive integer")
    if selection.get("selection_status") != "frozen_before_heldout":
        raise SSDHBFRatePlotError(
            "selection must have been frozen before heldout evaluation")
    root_for_selection = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root).expanduser().resolve()
    )
    try:
        frozen_selection = load_frozen_tp8_selection(
            repo_root=root_for_selection,
            selection_path=Path(selection_path),
        )
    except (OSError, ValueError) as exc:
        raise SSDHBFRatePlotError(
            f"cannot verify frozen TP8 selection: {exc}") from exc
    if (
        frozen_selection.sha256 != selection.get("sha256")
        or frozen_selection.schema_version != selection_schema
        or frozen_selection.selection_status
        != selection.get("selection_status")
    ):
        raise SSDHBFRatePlotError(
            "frozen selection file disagrees with the rate manifest")

    scenario = _mapping(manifest.get("scenario"), "scenario")
    if scenario.get("manifest_type") != "BalancedCausalPrefixManifest":
        raise SSDHBFRatePlotError(
            "rate-scaling plots require the balanced causal-prefix scenario "
            "so first-turn and resume metrics are both defined")
    scenario_id = _string(
        scenario.get("scenario_id"), "scenario.scenario_id")
    scenario_manifest_sha256 = scenario.get("manifest_sha256")
    if not _is_sha256(scenario_manifest_sha256):
        raise SSDHBFRatePlotError(
            "scenario.manifest_sha256 must be a SHA-256 digest")
    scenario_roster_sha256 = scenario.get(
        "measurement_roster_sha256")
    if not _is_sha256(scenario_roster_sha256):
        raise SSDHBFRatePlotError(
            "scenario.measurement_roster_sha256 must be a SHA-256 digest")
    scenario_identity_count = scenario.get(
        "measurement_identity_count")
    if (
        isinstance(scenario_identity_count, bool)
        or not isinstance(scenario_identity_count, int)
        or scenario_identity_count <= 0
    ):
        raise SSDHBFRatePlotError(
            "scenario.measurement_identity_count must be positive")
    rates = _rate_tuple(manifest.get("rates"), "rates")
    declared_rates = tuple(
        _finite(item, "scenario.declared_session_rates[]", positive=True)
        for item in _sequence(
            scenario.get("declared_session_rates"),
            "scenario.declared_session_rates",
        )
    )
    if any(rate not in declared_rates for rate in rates):
        raise SSDHBFRatePlotError(
            "one or more plotted rates are outside the scenario contract")
    seeds = _seed_tuple(manifest.get("seeds"), "seeds")
    designs = _validate_top_designs(manifest.get("designs"))
    frozen_designs = tuple(
        design.to_json_dict() for design in frozen_selection.designs)
    _same_design_sets(
        frozen_designs, designs, "designs")
    design_by_key = {
        str(design["key"]): design for design in designs}
    execution_inputs_sha256 = manifest.get(
        "execution_inputs_sha256")
    if not _is_sha256(execution_inputs_sha256):
        raise SSDHBFRatePlotError(
            "execution_inputs_sha256 must be a SHA-256 digest")

    aggregate_entries = tuple(_sequence(
        manifest.get("rate_aggregates"), "rate_aggregates"))
    if len(aggregate_entries) != len(rates):
        raise SSDHBFRatePlotError(
            "rate_aggregates must contain exactly one aggregate per rate")
    points_by_series: dict[str, list[RatePoint]] = {
        key: [] for key in _REFERENCE_ORDER
    }
    points_by_series.update({
        key: [] for key in design_by_key})
    eligibility_failures = []
    all_eligible = True
    aggregate_hashes: dict[float, str] = {}
    root = source_path.parent

    for expected_rate, raw_entry in zip(rates, aggregate_entries):
        entry_path = f"rate_aggregates[{len(aggregate_hashes)}]"
        entry = _mapping(raw_entry, entry_path)
        rate = _finite(
            entry.get("session_rate"),
            f"{entry_path}.session_rate",
            positive=True,
        )
        if rate != expected_rate:
            raise SSDHBFRatePlotError(
                "rate_aggregates must be sorted exactly like rates")
        claimed_hash = entry.get("sha256")
        if not _is_sha256(claimed_hash):
            raise SSDHBFRatePlotError(
                f"{entry_path}.sha256 must be a SHA-256 digest")
        aggregate_path = _aggregate_path(
            root, entry.get("relative_path"))
        aggregate, aggregate_payload = _strict_json(aggregate_path)
        actual_hash = _sha256(aggregate_payload)
        if actual_hash != claimed_hash:
            raise SSDHBFRatePlotError(
                f"aggregate hash mismatch at rate {rate:g}")
        aggregate_hashes[rate] = actual_hash
        if aggregate.get("schema_version") != SSD_HBF_SWEEP_SCHEMA_VERSION:
            raise SSDHBFRatePlotError(
                f"unsupported per-rate aggregate schema at rate {rate:g}")
        if aggregate.get("comparison_contract") != SSD_HBF_CONTRACT_KEY:
            raise SSDHBFRatePlotError(
                f"per-rate comparison contract mismatch at rate {rate:g}")
        if (
            aggregate.get("execution_inputs_sha256")
            != execution_inputs_sha256
        ):
            raise SSDHBFRatePlotError(
                f"execution input hash mismatch at rate {rate:g}")
        aggregate_scenario = _mapping(
            aggregate.get("scenario"),
            f"rate[{rate:g}].scenario",
        )
        expected_scenario = {
            "scenario_id": scenario_id,
            "scenario_manifest_type": "BalancedCausalPrefixManifest",
            "manifest_sha256": scenario_manifest_sha256,
            "measurement_roster_sha256": scenario_roster_sha256,
            "measurement_identity_count": scenario_identity_count,
            "declared_session_rates": list(declared_rates),
            "required_session_rate": rate,
        }
        scenario_mismatches = [
            key for key, expected in expected_scenario.items()
            if aggregate_scenario.get(key) != expected
        ]
        if scenario_mismatches:
            raise SSDHBFRatePlotError(
                f"scenario provenance mismatch at rate {rate:g}: "
                f"{scenario_mismatches}")

        grid = _mapping(aggregate.get("grid"), f"rate[{rate:g}].grid")
        if _finite(
                grid.get("session_rate"),
                f"rate[{rate:g}].grid.session_rate",
                positive=True,
        ) != rate:
            raise SSDHBFRatePlotError(
                f"grid rate mismatch at rate {rate:g}")
        if _seed_tuple(
                grid.get("seeds"),
                f"rate[{rate:g}].grid.seeds",
        ) != seeds:
            raise SSDHBFRatePlotError(
                f"grid seeds mismatch at rate {rate:g}")
        _same_design_sets(
            designs,
            grid.get("designs"),
            f"rate[{rate:g}].grid.designs",
        )

        rate_rows = tuple(_sequence(
            aggregate.get("rates"), f"rate[{rate:g}].rates"))
        if len(rate_rows) != 1:
            raise SSDHBFRatePlotError(
                f"aggregate at rate {rate:g} must contain one rate row")
        rate_row = _mapping(
            rate_rows[0], f"rate[{rate:g}].rates[0]")
        if _finite(
                rate_row.get("session_rate"),
                f"rate[{rate:g}].rates[0].session_rate",
                positive=True,
        ) != rate:
            raise SSDHBFRatePlotError(
                f"rate row mismatch at rate {rate:g}")

        eligibility = _mapping(
            rate_row.get("reference_eligibility"),
            f"rate[{rate:g}].reference_eligibility",
        )
        eligible = eligibility.get("eligible")
        if not isinstance(eligible, bool):
            raise SSDHBFRatePlotError(
                f"rate {rate:g} reference eligibility must be boolean")
        if not eligible:
            all_eligible = False
            failures = tuple(_sequence(
                eligibility.get("failures"),
                f"rate[{rate:g}].reference_eligibility.failures",
            ))
            eligibility_failures.extend(
                f"rate={rate:g}:{failure}" for failure in failures)

        references = _mapping(
            rate_row.get("references"),
            f"rate[{rate:g}].references",
        )
        if set(references) != set(_REFERENCE_ORDER):
            raise SSDHBFRatePlotError(
                f"rate {rate:g} must contain both baselines and the Oracle")
        reference_performance = {
            key: _performance(
                references[key],
                f"rate[{rate:g}].references.{key}",
                seeds=seeds,
            )
            for key in _REFERENCE_ORDER
        }

        design_rows = tuple(
            _mapping(item, f"rate[{rate:g}].designs[{index}]")
            for index, item in enumerate(_sequence(
                rate_row.get("designs"),
                f"rate[{rate:g}].designs",
            ))
        )
        if len(design_rows) != EXPECTED_DESIGN_COUNT:
            raise SSDHBFRatePlotError(
                f"rate {rate:g} must contain four frozen TP8 design rows")
        observed_keys = {
            _string(
                _mapping(row.get("design"), "design row.design").get("key"),
                "design row.design.key",
            )
            for row in design_rows
        }
        if observed_keys != set(design_by_key):
            raise SSDHBFRatePlotError(
                f"rate {rate:g} design rows do not match the frozen grid")

        baseline_runtime: dict[str, dict[str, Statistic]] = {}
        for row in design_rows:
            design = _mapping(row.get("design"), "design row.design")
            key = str(design["key"])
            if (
                stable_json_sha256(design)
                != stable_json_sha256(design_by_key[key])
            ):
                raise SSDHBFRatePlotError(
                    f"rate {rate:g} mutated frozen design {key!r}")
            restore_mode = str(design["restore_execution_mode"])
            expected_baseline = BASELINE_CANDIDATE_KEYS[restore_mode]
            if row.get("baseline_candidate_key") != expected_baseline:
                raise SSDHBFRatePlotError(
                    f"rate {rate:g} design {key!r} has an unmatched baseline")
            runtime_report = row.get("runtime_energy_tco")
            proposed_runtime = _runtime(
                runtime_report,
                f"rate[{rate:g}].design[{key}].runtime_energy_tco",
                seeds=seeds,
                prefix="proposed",
            )
            matched_runtime = _runtime(
                runtime_report,
                f"rate[{rate:g}].design[{key}].runtime_energy_tco",
                seeds=seeds,
                prefix="baseline",
            )
            previous = baseline_runtime.get(expected_baseline)
            if previous is not None and previous != matched_runtime:
                raise SSDHBFRatePlotError(
                    f"rate {rate:g} has inconsistent matched baseline "
                    f"runtime statistics for {expected_baseline!r}")
            baseline_runtime[expected_baseline] = matched_runtime
            points_by_series[key].append(RatePoint(
                session_rate=rate,
                performance=_performance(
                    row.get("metrics"),
                    f"rate[{rate:g}].design[{key}].metrics",
                    seeds=seeds,
                ),
                runtime=proposed_runtime,
                endurance=_endurance(
                    row.get("hbf_endurance"),
                    f"rate[{rate:g}].design[{key}].hbf_endurance",
                ),
            ))

        if set(baseline_runtime) != set(BASELINE_CANDIDATE_KEYS.values()):
            raise SSDHBFRatePlotError(
                f"rate {rate:g} lacks one matched baseline runtime mode")
        for key in _REFERENCE_ORDER:
            points_by_series[key].append(RatePoint(
                session_rate=rate,
                performance=reference_performance[key],
                runtime=baseline_runtime.get(key, {}),
                endurance={},
            ))

    series = []
    for key in _REFERENCE_ORDER:
        series.append(RateSeries(
            key=key,
            label=_REFERENCE_LABELS[key],
            kind="oracle" if key == ORACLE_CANDIDATE_KEY else "baseline",
            migration_policy=None,
            hbf_read_mode=None,
            restore_execution_mode=(
                None if key == ORACLE_CANDIDATE_KEY
                else "bulk"
                if key == BASELINE_CANDIDATE_KEYS["bulk"]
                else "layerwise_streaming"
            ),
            points=tuple(points_by_series[key]),
        ))
    for design in designs:
        key = str(design["key"])
        series.append(RateSeries(
            key=key,
            label=_design_label(design),
            kind="design",
            migration_policy=str(design["migration_policy"]),
            hbf_read_mode=str(design["hbf_read_mode"]),
            restore_execution_mode=str(
                design["restore_execution_mode"]),
            points=tuple(points_by_series[key]),
        ))
    for item in series:
        if tuple(point.session_rate for point in item.points) != rates:
            raise SSDHBFRatePlotError(
                f"series {item.key!r} is incomplete across rates")

    return LoadedRateSweep(
        source_path=source_path,
        source_file_sha256=_sha256(payload),
        source_payload_sha256=str(claimed_payload_hash),
        selection_file_sha256=frozen_selection.sha256,
        execution_inputs_sha256=str(execution_inputs_sha256),
        scenario_id=scenario_id,
        rates=rates,
        seeds=seeds,
        series=tuple(series),
        reference_eligible_at_all_rates=all_eligible,
        reference_eligibility_failures=tuple(eligibility_failures),
        aggregate_file_sha256_by_rate=aggregate_hashes,
    )


_SOURCE_FIELDS = (
    "rate_plot_schema_version",
    "source_manifest_sha256",
    "selection_file_sha256",
    "execution_inputs_sha256",
    "aggregate_file_sha256",
    "result_status",
    "reference_eligibility_failures",
    "scenario_id",
    "session_rate",
    "seed_count",
    "seed_ids",
    "series_key",
    "series_kind",
    "series_label",
    "hbf_layout",
    "migration_policy",
    "hbf_read_mode",
    "restore_execution_mode",
    "metric_group",
    "metric_key",
    "metric_unit",
    "mean",
    "ci95_lower",
    "ci95_upper",
)


def _metric_unit(group: str, metric_key: str) -> str:
    if group == "performance":
        return _PERFORMANCE_BY_KEY[metric_key].y_label
    if group == "runtime":
        return next(
            y_label
            for key, _report, _title, y_label, _scale in RUNTIME_METRICS
            if key == metric_key
        )
    return next(
        y_label
        for key, _title, y_label in ENDURANCE_METRICS
        if key == metric_key
    )


def build_source_rows(
        loaded: LoadedRateSweep,
) -> tuple[dict[str, Any], ...]:
    """Return one long-form audited row per series/rate/metric."""

    rows = []
    for series in loaded.series:
        groups = (
            ("performance", "performance"),
            ("runtime", "runtime"),
            ("endurance", "endurance"),
        )
        for point in series.points:
            for group_name, field_name in groups:
                statistics = getattr(point, field_name)
                for metric_key in sorted(statistics):
                    statistic = statistics[metric_key]
                    rows.append({
                        "rate_plot_schema_version":
                            RATE_PLOT_SCHEMA_VERSION,
                        "source_manifest_sha256":
                            loaded.source_file_sha256,
                        "selection_file_sha256":
                            loaded.selection_file_sha256,
                        "execution_inputs_sha256":
                            loaded.execution_inputs_sha256,
                        "aggregate_file_sha256":
                            loaded.aggregate_file_sha256_by_rate[
                                point.session_rate],
                        "result_status": (
                            "eligible"
                            if loaded.reference_eligible_at_all_rates
                            else "audit_reference_ineligible"
                        ),
                        "reference_eligibility_failures": "|".join(
                            loaded.reference_eligibility_failures),
                        "scenario_id": loaded.scenario_id,
                        "session_rate": point.session_rate,
                        "seed_count": len(loaded.seeds),
                        "seed_ids": ",".join(
                            str(seed) for seed in loaded.seeds),
                        "series_key": series.key,
                        "series_kind": series.kind,
                        "series_label": series.label,
                        "hbf_layout": (
                            TP8_LAYOUT
                            if series.kind == "design" else ""),
                        "migration_policy":
                            series.migration_policy or "",
                        "hbf_read_mode": series.hbf_read_mode or "",
                        "restore_execution_mode":
                            series.restore_execution_mode or "",
                        "metric_group": group_name,
                        "metric_key": metric_key,
                        "metric_unit": _metric_unit(
                            group_name, metric_key),
                        "mean": statistic.mean,
                        "ci95_lower": (
                            "" if statistic.ci95_lower is None
                            else statistic.ci95_lower),
                        "ci95_upper": (
                            "" if statistic.ci95_upper is None
                            else statistic.ci95_upper),
                    })
    return tuple(rows)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=_SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_pyplot():
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        return importlib.import_module("matplotlib.pyplot")
    except (ImportError, ModuleNotFoundError):
        return None


def _save_figure(figure, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(
        temporary,
        format="png",
        dpi=180,
        bbox_inches="tight",
    )
    temporary.replace(path)


def _series_style(series: RateSeries) -> dict[str, Any]:
    if series.key == BASELINE_CANDIDATE_KEYS["bulk"]:
        return {
            "color": "#E07A1F", "marker": "o", "linestyle": "-"}
    if series.key == BASELINE_CANDIDATE_KEYS["layerwise_streaming"]:
        return {
            "color": "#A05A2C", "marker": "s", "linestyle": "--"}
    if series.key == ORACLE_CANDIDATE_KEY:
        return {
            "color": "#111111", "marker": "D", "linestyle": ":"}
    colors = {
        ("composite_ready", "demand"): "#1F77B4",
        ("composite_ready", "prefetch"): "#17BECF",
        ("composite_ready_adaptive", "demand"): "#2CA02C",
        ("composite_ready_adaptive", "prefetch"): "#9467BD",
    }
    color = colors.get(
        (series.migration_policy, series.hbf_read_mode), "#7F7F7F")
    return {
        "color": color,
        "marker": "o" if series.hbf_read_mode == "demand" else "^",
        "linestyle": "-",
    }


def _plot_metric(
        axis,
        loaded: LoadedRateSweep,
        *,
        group: str,
        metric_key: str,
        title: str,
        y_label: str,
        log_y: bool = False,
        bounded_fraction: bool = False,
) -> None:
    for series in loaded.series:
        statistics = [
            getattr(point, group).get(metric_key)
            for point in series.points
        ]
        if not any(statistic is not None for statistic in statistics):
            continue
        if any(statistic is None for statistic in statistics):
            raise SSDHBFRatePlotError(
                f"series {series.key!r} is partially missing {metric_key!r}")
        concrete = [
            statistic for statistic in statistics
            if statistic is not None
        ]
        means = [statistic.mean for statistic in concrete]
        style = _series_style(series)
        axis.plot(
            loaded.rates,
            means,
            label=series.label,
            linewidth=1.8,
            markersize=4.8,
            **style,
        )
        if all(
            statistic.ci95_lower is not None
            and statistic.ci95_upper is not None
            for statistic in concrete
        ):
            lower = [
                float(statistic.ci95_lower) for statistic in concrete]
            upper = [
                float(statistic.ci95_upper) for statistic in concrete]
            if bounded_fraction:
                lower = [max(0.0, value) for value in lower]
                upper = [min(1.0, value) for value in upper]
            if log_y:
                lower = [
                    max(value, mean * 1e-6)
                    for value, mean in zip(lower, means)
                ]
            axis.fill_between(
                loaded.rates,
                lower,
                upper,
                color=style["color"],
                alpha=0.09,
                linewidth=0,
            )
    axis.set_xscale("log")
    axis.set_xticks(loaded.rates)
    axis.set_xticklabels(f"{rate:g}" for rate in loaded.rates)
    if log_y:
        axis.set_yscale("log")
    if bounded_fraction:
        axis.set_ylim(0.0, 1.05)
    axis.set_title(title, fontsize=10, loc="left")
    axis.set_xlabel("Offered session rate (sessions/s)")
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.22)


def _audit_prefix(loaded: LoadedRateSweep) -> str:
    return (
        ""
        if loaded.reference_eligible_at_all_rates
        else "AUDIT — preregistered reference-opportunity gate failed\n"
    )


def _render_performance(
        pyplot,
        loaded: LoadedRateSweep,
        path: Path,
) -> None:
    figure, axes = pyplot.subplots(3, 3, figsize=(18, 14))
    plots = (
        (
            "slo_good_output_tokens_per_second",
            "Offered-rate-normalized SLO-good output rate",
        ),
        ("joint_slo_pass_fraction", "Joint TTFT + TPOT SLO attainment"),
        (
            "observed_request_throughput_per_second",
            "Measured inter-completion output rate",
        ),
        ("first_ttft_p95_ns", "First-turn TTFT"),
        ("resume_ttft_p95_ns", "Resume TTFT"),
        ("tpot_p95_ns", "Per-output-token latency"),
        (
            "slo_request_goodput_per_second",
            "Offered-rate-normalized SLO-good request rate",
        ),
    )
    for axis, (metric_key, title) in zip(axes.flat, plots):
        spec = _PERFORMANCE_BY_KEY[metric_key]
        _plot_metric(
            axis,
            loaded,
            group="performance",
            metric_key=metric_key,
            title=title,
            y_label=spec.y_label,
            log_y=spec.log_scale,
            bounded_fraction=spec.bounded_fraction,
        )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes[2, 1].axis("off")
    axes[2, 1].legend(
        handles,
        labels,
        loc="center",
        fontsize=8,
        frameon=False,
    )
    axes[2, 2].axis("off")
    failures = (
        "none"
        if loaded.reference_eligible_at_all_rates
        else "\n".join(loaded.reference_eligibility_failures[:5])
    )
    axes[2, 2].text(
        0.0,
        1.0,
        "\n".join((
            "Study contract",
            f"scenario: {loaded.scenario_id}",
            "layout: TP8 fixed",
            f"rates: {', '.join(f'{rate:g}' for rate in loaded.rates)}",
            f"paired seeds/rate: {len(loaded.seeds)}",
            "bands: two-sided Student-t 95% CI",
            "selection: frozen once; no per-rate winner selection",
            "references: bulk baseline, streaming baseline, Oracle",
            f"reference gate failures: {failures}",
        )),
        va="top",
        fontsize=9,
    )
    figure.suptitle(
        _audit_prefix(loaded)
        + "SSD-staged GPU+HBF TP8 load-scaling comparison",
        fontsize=15,
    )
    figure.tight_layout()
    _save_figure(figure, path)
    pyplot.close(figure)


def _render_runtime(
        pyplot,
        loaded: LoadedRateSweep,
        path: Path,
) -> None:
    figure, axes = pyplot.subplots(1, 3, figsize=(18, 5.5))
    for axis, (
        metric_key,
        _report_key,
        title,
        y_label,
        _scale,
    ) in zip(axes, RUNTIME_METRICS):
        _plot_metric(
            axis,
            loaded,
            group="runtime",
            metric_key=metric_key,
            title=title,
            y_label=y_label,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    figure.suptitle(
        _audit_prefix(loaded)
        + "TP8 runtime power, facility energy, and 5-year TCO",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.95))
    _save_figure(figure, path)
    pyplot.close(figure)


def _render_endurance(
        pyplot,
        loaded: LoadedRateSweep,
        path: Path,
) -> None:
    figure, axes = pyplot.subplots(1, 2, figsize=(13, 5.5))
    for axis, (metric_key, title, y_label) in zip(
            axes, ENDURANCE_METRICS):
        _plot_metric(
            axis,
            loaded,
            group="endurance",
            metric_key=metric_key,
            title=title,
            y_label=y_label,
        )
    axes[1].axhline(
        100.0,
        color="#666666",
        linestyle="--",
        linewidth=1.0,
        label="5-year wear budget",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    figure.suptitle(
        _audit_prefix(loaded)
        + "TP8 HBF recurring-write and endurance scaling",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.95))
    _save_figure(figure, path)
    pyplot.close(figure)


def write_rate_plot_artifacts(
        loaded: LoadedRateSweep,
        output_dir: Path | str,
        *,
        render: bool = True,
) -> RatePlotArtifacts:
    """Write audited source data and attached-style rate-scaling figures."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "rate_plot_source.csv"
    artifact_manifest_path = root / "rate_plot_artifacts.json"
    rows = build_source_rows(loaded)
    _write_csv(source_path, rows)

    pyplot = _load_pyplot() if render else None
    matplotlib_available = pyplot is not None
    performance_path = runtime_path = endurance_path = None
    if render and pyplot is not None:
        prefix = (
            "" if loaded.reference_eligible_at_all_rates else "audit_")
        performance_path = root / f"{prefix}01_tp8_rate_performance.png"
        runtime_path = root / f"{prefix}02_tp8_rate_power_tco.png"
        endurance_path = root / f"{prefix}03_tp8_rate_endurance.png"
        _render_performance(pyplot, loaded, performance_path)
        _render_runtime(pyplot, loaded, runtime_path)
        _render_endurance(pyplot, loaded, endurance_path)

    artifact_payload = {
        "schema_version": RATE_PLOT_SCHEMA_VERSION,
        "source_manifest_path": str(loaded.source_path),
        "source_manifest_file_sha256": loaded.source_file_sha256,
        "source_manifest_payload_sha256": loaded.source_payload_sha256,
        "selection_file_sha256": loaded.selection_file_sha256,
        "execution_inputs_sha256": loaded.execution_inputs_sha256,
        "aggregate_file_sha256_by_rate": {
            f"{rate:g}": digest
            for rate, digest in sorted(
                loaded.aggregate_file_sha256_by_rate.items())
        },
        "hbf_layout": TP8_LAYOUT,
        "rates": list(loaded.rates),
        "seeds": list(loaded.seeds),
        "series_keys": [series.key for series in loaded.series],
        "reference_eligible_at_all_rates":
            loaded.reference_eligible_at_all_rates,
        "reference_eligibility_failures":
            list(loaded.reference_eligibility_failures),
        "source_csv": source_path.name,
        "source_csv_sha256": _sha256(source_path.read_bytes()),
        "performance_png": (
            None if performance_path is None else performance_path.name),
        "runtime_power_energy_tco_png": (
            None if runtime_path is None else runtime_path.name),
        "hbf_endurance_png": (
            None if endurance_path is None else endurance_path.name),
        "rendered": render and pyplot is not None,
    }
    artifact_payload["artifact_payload_sha256"] = stable_json_sha256(
        artifact_payload)
    _write_json(artifact_manifest_path, artifact_payload)
    return RatePlotArtifacts(
        source_csv=source_path,
        artifact_manifest_json=artifact_manifest_path,
        performance_png=performance_path,
        runtime_power_energy_tco_png=runtime_path,
        hbf_endurance_png=endurance_path,
        rendered=render and pyplot is not None,
        matplotlib_available=matplotlib_available,
        source_manifest_sha256=loaded.source_file_sha256,
    )


def generate_rate_plots(
        manifest_path: Path | str,
        output_dir: Path | str,
        *,
        repo_root: Optional[Path | str] = None,
        render: bool = True,
) -> RatePlotArtifacts:
    loaded = load_rate_sweep(
        manifest_path, repo_root=repo_root)
    return write_rate_plot_artifacts(
        loaded, output_dir, render=render)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render the frozen TP8 SSD-HBF selection across offered rates."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(RATE_SWEEP_MANIFEST_NAME),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    artifacts = generate_rate_plots(
        args.manifest,
        args.output,
        repo_root=args.repo_root,
        render=not args.no_render,
    )
    print(json.dumps(
        artifacts.to_json_dict(),
        sort_keys=True,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENDURANCE_METRICS",
    "EXPECTED_DESIGN_COUNT",
    "LoadedRateSweep",
    "RATE_PERFORMANCE_KEYS",
    "RATE_PLOT_SCHEMA_VERSION",
    "RUNTIME_METRICS",
    "RatePlotArtifacts",
    "RatePoint",
    "RateSeries",
    "SSDHBFRatePlotError",
    "Statistic",
    "build_source_rows",
    "generate_rate_plots",
    "load_rate_sweep",
    "main",
    "write_rate_plot_artifacts",
]

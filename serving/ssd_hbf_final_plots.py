"""Frozen final-selection plots for the SSD-staged HBF campaign.

This module accepts only the aggregate produced by ``ssd_hbf_design_sweep``
and an explicit pre-heldout selection manifest.  The manifest, rather than
heldout measurements, chooses the eight rendered design coordinates.  Every
other heldout coordinate remains visible in the audit CSV with an explicit
exclusion reason.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .core.hbf_comparison_workload import stable_json_sha256
from .ssd_hbf_design_sweep import (
    BASELINE_CANDIDATE_KEYS,
    ORACLE_CANDIDATE_KEY,
    REQUIRED_SESSION_RATE,
    SSD_HBF_CONTRACT_KEY,
    SSD_HBF_SWEEP_SCHEMA_VERSION,
    SUPPORTED_HBF_READ_MODES,
    SUPPORTED_LAYOUTS,
    SUPPORTED_MIGRATION_POLICIES,
    SUPPORTED_RESTORE_EXECUTION_MODES,
)


FINAL_RESULTS_SCHEMA_VERSION = 3
POLICY_SELECTION_SCHEMA_VERSION = 3
PLOT_SOURCE_SCHEMA_VERSION = 3
FROZEN_SELECTION_SCHEMA_VERSION = 1
CENTRAL_ENDURANCE_SCENARIO = "slc_100k_pe_waf1"
RUNTIME_TCO_REPORT_SCHEMA = "ssd-hbf-runtime-tco-v1"
FINAL_PLOT_DESIGN_CELL_COUNT = 8


@dataclass(frozen=True)
class PerformanceMetricSpec:
    aggregate_key: str
    row_field: str
    plot_key: str
    filename_stem: str
    title: str
    y_label: str
    scale: float = 1.0
    optional: bool = False
    positive: bool = False
    bounded_fraction: bool = False
    log_scale: bool = False


PERFORMANCE_METRIC_SPECS = (
    PerformanceMetricSpec(
        "first_ttft_p95_ns",
        "first_ttft_p95_ns",
        "first_ttft_p95_seconds",
        "01_first_ttft_p95",
        "First-turn TTFT p95",
        "First TTFT p95 (s)",
        scale=1e-9,
        optional=True,
        positive=True,
        log_scale=True,
    ),
    PerformanceMetricSpec(
        "resume_ttft_p95_ns",
        "resume_ttft_p95_ns",
        "resume_ttft_p95_seconds",
        "02_resume_ttft_p95",
        "Resume TTFT p95",
        "Resume TTFT p95 (s)",
        scale=1e-9,
        positive=True,
        log_scale=True,
    ),
    PerformanceMetricSpec(
        "tpot_p95_ns",
        "tpot_p95_ns",
        "tpot_p95_milliseconds",
        "03_tpot_p95",
        "TPOT p95",
        "TPOT p95 (ms/token)",
        scale=1e-6,
        positive=True,
        log_scale=True,
    ),
    PerformanceMetricSpec(
        "joint_slo_pass_fraction",
        "joint_slo_pass_fraction",
        "joint_slo_pass_fraction",
        "04_joint_slo_pass_fraction",
        "Joint TTFT+TPOT SLO attainment",
        "Joint SLO pass fraction",
        bounded_fraction=True,
    ),
    PerformanceMetricSpec(
        "slo_request_goodput_per_second",
        "slo_request_goodput_per_second",
        "slo_request_goodput_per_second",
        "05_slo_request_goodput",
        "SLO request goodput",
        "SLO-good requests/s",
    ),
    PerformanceMetricSpec(
        "slo_good_output_tokens_per_second",
        "goodput_mean",
        "slo_output_token_goodput_per_second",
        "06_slo_output_token_goodput",
        "SLO output-token goodput",
        "SLO-good output tokens/s",
    ),
    PerformanceMetricSpec(
        "observed_request_throughput_per_second",
        "observed_request_throughput_per_second",
        "observed_request_throughput_per_second",
        "07_observed_request_throughput",
        "Observed inter-completion throughput",
        "Observed requests/s",
        positive=True,
    ),
)
PERFORMANCE_PLOT_COUNT = len(PERFORMANCE_METRIC_SPECS)

# The final aggregate carries the committed RuntimeTCOComparison shape here.
# Extra uncertainty and per-seed fields are allowed alongside its core
# projections, but alternate report names are not inferred.
RUNTIME_REPORT_FIELD_KEYS = ("runtime_energy_tco",)
RUNTIME_PROJECTION_METRIC_KEYS = {
    "average_it_power_w": "trace_average_it_power_w",
    "five_year_facility_energy_kwh": (
        "five_year_facility_energy_kwh"),
    "five_year_tco_usd": "five_year_tco_usd",
}

_REQUIRED_REFERENCE_KEYS = frozenset({
    *BASELINE_CANDIDATE_KEYS.values(),
    ORACLE_CANDIDATE_KEY,
})
_REQUIRED_OPTIONS = frozenset(
    (read_mode, restore_mode)
    for read_mode in SUPPORTED_HBF_READ_MODES
    for restore_mode in SUPPORTED_RESTORE_EXECUTION_MODES
)
class SSDHBFFinalResultsError(ValueError):
    """Raised when an aggregate cannot support final-result claims."""


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SSDHBFFinalResultsError(f"{path} must be an object")
    return value


def _sequence(value: object, path: str) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise SSDHBFFinalResultsError(f"{path} must be an array")
    return value


def _finite(
        value: object,
        path: str,
        *,
        minimum: Optional[float] = None,
        positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SSDHBFFinalResultsError(
            f"{path} must be a finite number")
    converted = float(value)
    if positive and converted <= 0.0:
        raise SSDHBFFinalResultsError(f"{path} must be positive")
    if minimum is not None and converted < minimum:
        raise SSDHBFFinalResultsError(
            f"{path} must be at least {minimum}")
    return converted


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SSDHBFFinalResultsError(
            f"{path} must be a positive integer")
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise SSDHBFFinalResultsError(f"{path} must be non-empty")
    return value


def _unique_ints(value: object, path: str) -> tuple[int, ...]:
    values = tuple(_sequence(value, path))
    if (
        not values
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in values
        )
        or len(values) != len(set(values))
    ):
        raise SSDHBFFinalResultsError(
            f"{path} must contain unique integers")
    return values


@dataclass(frozen=True)
class FrozenFinalSelection:
    source_path: Path
    source_file_sha256: str
    source_payload_sha256: str
    repo_root: Path
    session_rate: float
    discovery_seed_ids: tuple[int, ...]
    discovery_aggregate_path: Path
    discovery_aggregate_sha256: str
    heldout_seed_ids: tuple[int, ...]
    heldout_aggregate_path: Path
    migration_policies: tuple[str, ...]
    mixed_batch_latency_limit_ms: Optional[int]
    coordinates: tuple[tuple[str, str, str, str], ...]


def load_frozen_selection(
        path: Path | str,
        *,
        repo_root: Optional[Path | str] = None,
) -> FrozenFinalSelection:
    """Load and verify the pre-heldout eight-coordinate selection."""

    source_path = Path(path).expanduser().resolve()
    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root).expanduser().resolve()
    )
    try:
        payload = source_path.read_bytes()
        raw = json.loads(
            payload,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}"),
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SSDHBFFinalResultsError(
            f"cannot load frozen selection {source_path}: {exc}") from exc
    selection = _mapping(raw, "selection")
    if selection.get("schema_version") != FROZEN_SELECTION_SCHEMA_VERSION:
        raise SSDHBFFinalResultsError(
            "frozen selection schema is unsupported")
    if selection.get("selection_status") != "frozen_before_heldout":
        raise SSDHBFFinalResultsError(
            "selection_status must be 'frozen_before_heldout'")
    session_rate = _finite(
        selection.get("session_rate"),
        "selection.session_rate",
        positive=True,
    )
    if not math.isclose(
        session_rate,
        REQUIRED_SESSION_RATE,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise SSDHBFFinalResultsError(
            "frozen selection must use the pinned 3 sessions/s rate")

    discovery = _mapping(
        selection.get("discovery"), "selection.discovery")
    discovery_seeds = _unique_ints(
        discovery.get("seeds"), "selection.discovery.seeds")
    if discovery.get("selection_metric") != (
            "slo_good_output_tokens_per_second"):
        raise SSDHBFFinalResultsError(
            "discovery selection_metric is unsupported")
    if discovery.get("selection_direction") != "maximize":
        raise SSDHBFFinalResultsError(
            "discovery selection_direction must be 'maximize'")
    discovery_relative = Path(_nonempty_string(
        discovery.get("aggregate_path"),
        "selection.discovery.aggregate_path",
    ))
    discovery_path = (
        discovery_relative
        if discovery_relative.is_absolute()
        else root / discovery_relative
    ).resolve()
    discovery_sha256 = discovery.get("aggregate_sha256")
    if not _is_sha256(discovery_sha256):
        raise SSDHBFFinalResultsError(
            "selection.discovery.aggregate_sha256 is invalid")
    try:
        observed_discovery_sha256 = _sha256(discovery_path.read_bytes())
    except OSError as exc:
        raise SSDHBFFinalResultsError(
            f"cannot read frozen discovery aggregate {discovery_path}: "
            f"{exc}") from exc
    if observed_discovery_sha256 != discovery_sha256:
        raise SSDHBFFinalResultsError(
            "frozen discovery aggregate hash does not match the manifest")

    heldout = _mapping(
        selection.get("heldout"), "selection.heldout")
    heldout_seeds = _unique_ints(
        heldout.get("seeds"), "selection.heldout.seeds")
    if set(discovery_seeds) & set(heldout_seeds):
        raise SSDHBFFinalResultsError(
            "discovery and heldout seed sets must be disjoint")
    heldout_relative = Path(_nonempty_string(
        heldout.get("output_path"),
        "selection.heldout.output_path",
    ))
    heldout_root = (
        heldout_relative
        if heldout_relative.is_absolute()
        else root / heldout_relative
    ).resolve()

    raw_policies = tuple(_sequence(
        selection.get("migration_policies"),
        "selection.migration_policies",
    ))
    if (
        not raw_policies
        or any(
            not isinstance(policy, str)
            or policy not in SUPPORTED_MIGRATION_POLICIES
            for policy in raw_policies
        )
        or len(raw_policies) != len(set(raw_policies))
    ):
        raise SSDHBFFinalResultsError(
            "selection.migration_policies must contain unique supported "
            "policies")
    policies = tuple(str(policy) for policy in raw_policies)
    mixed_limit = selection.get("mixed_batch_latency_limit_ms")
    if (
        mixed_limit is not None
        and (
            isinstance(mixed_limit, bool)
            or not isinstance(mixed_limit, int)
            or mixed_limit <= 0
        )
    ):
        raise SSDHBFFinalResultsError(
            "selection.mixed_batch_latency_limit_ms must be a positive "
            "integer or null")

    raw_coordinates = _sequence(
        selection.get("restore_by_coordinate"),
        "selection.restore_by_coordinate",
    )
    coordinates = []
    for index, raw_coordinate in enumerate(raw_coordinates):
        coordinate = _mapping(
            raw_coordinate,
            f"selection.restore_by_coordinate[{index}]",
        )
        if set(coordinate) != {
            "migration_policy",
            "hbf_layout",
            "hbf_read_mode",
            "restore_execution_mode",
        }:
            raise SSDHBFFinalResultsError(
                "each frozen coordinate must contain exactly policy, "
                "layout, read mode, and restore mode")
        policy = coordinate.get("migration_policy")
        layout = coordinate.get("hbf_layout")
        read_mode = coordinate.get("hbf_read_mode")
        restore_mode = coordinate.get("restore_execution_mode")
        if policy not in policies:
            raise SSDHBFFinalResultsError(
                "frozen coordinate uses an undeclared migration policy")
        if layout not in SUPPORTED_LAYOUTS:
            raise SSDHBFFinalResultsError(
                "frozen coordinate uses an unsupported layout")
        if read_mode not in SUPPORTED_HBF_READ_MODES:
            raise SSDHBFFinalResultsError(
                "frozen coordinate uses an unsupported read mode")
        if restore_mode not in SUPPORTED_RESTORE_EXECUTION_MODES:
            raise SSDHBFFinalResultsError(
                "frozen coordinate uses an unsupported restore mode")
        coordinates.append((
            str(policy),
            str(layout),
            str(read_mode),
            str(restore_mode),
        ))
    frozen_coordinates = tuple(coordinates)
    expected_projection = {
        (policy, layout, read_mode)
        for policy in policies
        for layout in SUPPORTED_LAYOUTS
        for read_mode in SUPPORTED_HBF_READ_MODES
    }
    observed_projection = {
        coordinate[:3] for coordinate in frozen_coordinates}
    if (
        len(frozen_coordinates) != FINAL_PLOT_DESIGN_CELL_COUNT
        or len(frozen_coordinates) != len(set(frozen_coordinates))
        or observed_projection != expected_projection
    ):
        raise SSDHBFFinalResultsError(
            "frozen selection must contain exactly one restore choice for "
            "each policy/layout/read-mode coordinate")

    return FrozenFinalSelection(
        source_path=source_path,
        source_file_sha256=_sha256(payload),
        source_payload_sha256=stable_json_sha256(selection),
        repo_root=root,
        session_rate=session_rate,
        discovery_seed_ids=discovery_seeds,
        discovery_aggregate_path=discovery_path,
        discovery_aggregate_sha256=str(discovery_sha256),
        heldout_seed_ids=heldout_seeds,
        heldout_aggregate_path=(heldout_root / "aggregate.json").resolve(),
        migration_policies=policies,
        mixed_batch_latency_limit_ms=mixed_limit,
        coordinates=tuple(sorted(frozen_coordinates)),
    )


def _memory_identity(memory: Mapping[str, Any]) -> tuple[object, ...]:
    kind = memory.get("kind")
    if not isinstance(kind, str) or not kind:
        raise SSDHBFFinalResultsError(
            "design.active_memory.kind must be non-empty")
    capacity = _finite(
        memory.get("capacity_gib_per_card"),
        "design.active_memory.capacity_gib_per_card",
        positive=True,
    )
    bandwidth = _finite(
        memory.get("bandwidth_gbps_per_card"),
        "design.active_memory.bandwidth_gbps_per_card",
        positive=True,
    )
    capex = _finite(
        memory.get("capex_usd_per_gib"),
        "design.active_memory.capex_usd_per_gib",
        minimum=0.0,
    )
    power = _finite(
        memory.get("power_w_per_gib"),
        "design.active_memory.power_w_per_gib",
        minimum=0.0,
    )
    return (kind, capacity, bandwidth, capex, power)


def _group_id(
        layout: str,
        memory_identity: tuple[object, ...],
) -> str:
    memory_json = json.dumps(
        list(memory_identity),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return (
        f"{layout}|memory-"
        f"{hashlib.sha256(memory_json.encode('utf-8')).hexdigest()[:12]}"
    )


def _stat_mean(
        metrics: Mapping[str, Any],
        metric_key: str,
        path: str,
        *,
        positive: bool = False,
        expected_seed_ids: Optional[tuple[int, ...]] = None,
) -> tuple[float, Optional[float], Optional[float]]:
    statistic = _mapping(
        metrics.get(metric_key), f"{path}.{metric_key}")
    mean = _finite(
        statistic.get("mean"), f"{path}.{metric_key}.mean",
        positive=positive,
    )
    lower_value = statistic.get("ci95_lower")
    upper_value = statistic.get("ci95_upper")
    if (lower_value is None) != (upper_value is None):
        raise SSDHBFFinalResultsError(
            f"{path}.{metric_key} has a partial confidence interval")
    if lower_value is None:
        lower = upper = None
    else:
        lower = _finite(
            lower_value, f"{path}.{metric_key}.ci95_lower")
        upper = _finite(
            upper_value, f"{path}.{metric_key}.ci95_upper")
        if lower > mean or upper < mean or lower > upper:
            raise SSDHBFFinalResultsError(
                f"{path}.{metric_key} has an invalid confidence interval")
    if expected_seed_ids is not None:
        raw_seed_ids = _sequence(
            statistic.get("seed_ids"),
            f"{path}.{metric_key}.seed_ids",
        )
        seed_ids = tuple(raw_seed_ids)
        if (
            any(
                isinstance(seed, bool) or not isinstance(seed, int)
                for seed in seed_ids
            )
            or seed_ids != expected_seed_ids
        ):
            raise SSDHBFFinalResultsError(
                f"{path}.{metric_key}.seed_ids must exactly match "
                "grid.seeds")
        raw_values = _sequence(
            statistic.get("values"),
            f"{path}.{metric_key}.values",
        )
        values = tuple(
            _finite(
                value,
                f"{path}.{metric_key}.values[{index}]",
            )
            for index, value in enumerate(raw_values)
        )
        if len(values) != len(expected_seed_ids):
            raise SSDHBFFinalResultsError(
                f"{path}.{metric_key}.values must cover every grid seed")
        values_mean = math.fsum(values) / len(values)
        if not math.isclose(
            values_mean,
            mean,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise SSDHBFFinalResultsError(
                f"{path}.{metric_key}.mean disagrees with seed values")
    return mean, lower, upper


def _optional_stat_mean(
        metrics: Mapping[str, Any],
        metric_key: str,
        path: str,
        *,
        positive: bool = False,
        expected_seed_ids: Optional[tuple[int, ...]] = None,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if metrics.get(metric_key) is None:
        return None, None, None
    return _stat_mean(
        metrics,
        metric_key,
        path,
        positive=positive,
        expected_seed_ids=expected_seed_ids,
    )


@dataclass(frozen=True)
class AggregateStatistic:
    mean: Optional[float]
    ci95_lower: Optional[float]
    ci95_upper: Optional[float]


def _performance_statistics(
        metrics: Mapping[str, Any],
        path: str,
        *,
        expected_seed_ids: tuple[int, ...],
) -> Mapping[str, AggregateStatistic]:
    result = {}
    for spec in PERFORMANCE_METRIC_SPECS:
        if spec.optional:
            mean, lower, upper = _optional_stat_mean(
                metrics,
                spec.aggregate_key,
                path,
                positive=spec.positive,
                expected_seed_ids=expected_seed_ids,
            )
        else:
            mean, lower, upper = _stat_mean(
                metrics,
                spec.aggregate_key,
                path,
                positive=spec.positive,
                expected_seed_ids=expected_seed_ids,
            )
        if (
            spec.bounded_fraction
            and mean is not None
            and not 0.0 <= mean <= 1.0
        ):
            raise SSDHBFFinalResultsError(
                f"{path}.{spec.aggregate_key}.mean must be in [0, 1]")
        result[spec.aggregate_key] = AggregateStatistic(
            mean=mean,
            ci95_lower=lower,
            ci95_upper=upper,
        )
    return result


@dataclass(frozen=True)
class RuntimeObjectives:
    baseline_average_it_power_w: float
    proposed_average_it_power_w: float
    baseline_five_year_facility_energy_kwh: float
    proposed_five_year_facility_energy_kwh: float
    baseline_five_year_tco_usd: float
    proposed_five_year_tco_usd: float
    report_field: str

    @property
    def power_ratio(self) -> float:
        return (
            self.proposed_average_it_power_w
            / self.baseline_average_it_power_w
        )

    @property
    def energy_ratio(self) -> float:
        return (
            self.proposed_five_year_facility_energy_kwh
            / self.baseline_five_year_facility_energy_kwh
        )

    @property
    def tco_ratio(self) -> float:
        return (
            self.proposed_five_year_tco_usd
            / self.baseline_five_year_tco_usd
        )


def _runtime_objectives(
        row: Mapping[str, Any],
        path: str,
) -> Optional[RuntimeObjectives]:
    present = [
        key for key in RUNTIME_REPORT_FIELD_KEYS
        if row.get(key) is not None
    ]
    if not present:
        return None
    if len(present) != 1:
        raise SSDHBFFinalResultsError(
            f"{path} contains ambiguous runtime report fields {present}")
    report_field = present[0]
    report = _mapping(row[report_field], f"{path}.{report_field}")
    if report.get("report_schema") != RUNTIME_TCO_REPORT_SCHEMA:
        raise SSDHBFFinalResultsError(
            f"{path}.{report_field}.report_schema must equal "
            f"{RUNTIME_TCO_REPORT_SCHEMA!r}")
    baseline = _mapping(
        report.get("baseline"), f"{path}.{report_field}.baseline")
    proposed = _mapping(
        report.get("proposed"), f"{path}.{report_field}.proposed")
    baseline_system = baseline.get("system_key")
    proposed_system = proposed.get("system_key")
    if baseline_system != "two_gpu_local_ssd_baseline":
        raise SSDHBFFinalResultsError(
            f"{path}.{report_field}.baseline has the wrong system_key")
    if proposed_system != "one_gpu_local_ssd_plus_one_hbf":
        raise SSDHBFFinalResultsError(
            f"{path}.{report_field}.proposed has the wrong system_key")
    if baseline.get("report_schema") != RUNTIME_TCO_REPORT_SCHEMA:
        raise SSDHBFFinalResultsError(
            f"{path}.{report_field}.baseline has the wrong report_schema")
    if proposed.get("report_schema") != RUNTIME_TCO_REPORT_SCHEMA:
        raise SSDHBFFinalResultsError(
            f"{path}.{report_field}.proposed has the wrong report_schema")

    values: dict[str, float] = {}
    for side_name, side in (("baseline", baseline), ("proposed", proposed)):
        for output_name, input_name in (
            RUNTIME_PROJECTION_METRIC_KEYS.items()
        ):
            values[f"{side_name}_{output_name}"] = _finite(
                side.get(input_name),
                f"{path}.{report_field}.{side_name}.{input_name}",
                positive=True,
            )
    return RuntimeObjectives(
        baseline_average_it_power_w=values[
            "baseline_average_it_power_w"],
        proposed_average_it_power_w=values[
            "proposed_average_it_power_w"],
        baseline_five_year_facility_energy_kwh=values[
            "baseline_five_year_facility_energy_kwh"],
        proposed_five_year_facility_energy_kwh=values[
            "proposed_five_year_facility_energy_kwh"],
        baseline_five_year_tco_usd=values[
            "baseline_five_year_tco_usd"],
        proposed_five_year_tco_usd=values[
            "proposed_five_year_tco_usd"],
        report_field=report_field,
    )


@dataclass(frozen=True)
class WearObjectives:
    five_year_budget_fraction: float
    payload_write_bytes_per_second: float
    hottest_card_write_bytes_per_day: float
    hottest_card_share: Optional[float]
    card_write_cv: Optional[float]


def _optional_nonnegative(
        value: object,
        path: str,
) -> Optional[float]:
    if value is None:
        return None
    return _finite(value, path, minimum=0.0)


def _wear_objectives(
        row: Mapping[str, Any],
        path: str,
) -> WearObjectives:
    endurance = _mapping(
        row.get("hbf_endurance"), f"{path}.hbf_endurance")
    if endurance.get("schema_version") != 1:
        raise SSDHBFFinalResultsError(
            f"{path}.hbf_endurance has an unsupported schema")
    observed = _finite(
        endurance.get("total_observed_seconds"),
        f"{path}.hbf_endurance.total_observed_seconds",
        positive=True,
    )
    write_bytes = _finite(
        endurance.get("total_physical_write_bytes"),
        f"{path}.hbf_endurance.total_physical_write_bytes",
        minimum=0.0,
    )
    hotness = _mapping(
        endurance.get("hotness"), f"{path}.hbf_endurance.hotness")
    if _positive_int(
        hotness.get("card_count"),
        f"{path}.hbf_endurance.hotness.card_count",
    ) != 8:
        raise SSDHBFFinalResultsError(
            f"{path}.hbf_endurance must cover eight physical HBF cards")
    scenarios = _mapping(
        endurance.get("scenarios"), f"{path}.hbf_endurance.scenarios")
    central = _mapping(
        scenarios.get(CENTRAL_ENDURANCE_SCENARIO),
        (
            f"{path}.hbf_endurance.scenarios."
            f"{CENTRAL_ENDURANCE_SCENARIO}"
        ),
    )
    if not math.isclose(
        _finite(
            central.get("service_lifetime_years"),
            (
                f"{path}.hbf_endurance.scenarios."
                f"{CENTRAL_ENDURANCE_SCENARIO}."
                "service_lifetime_years"
            ),
            positive=True,
        ),
        5.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SSDHBFFinalResultsError(
            f"{path}.hbf_endurance is not a five-year projection")
    cards = _sequence(
        central.get("cards"),
        (
            f"{path}.hbf_endurance.scenarios."
            f"{CENTRAL_ENDURANCE_SCENARIO}.cards"
        ),
    )
    if len(cards) != 8:
        raise SSDHBFFinalResultsError(
            f"{path}.hbf_endurance central scenario needs eight cards")
    budgets = []
    per_day = []
    for card_index, raw_card in enumerate(cards):
        card = _mapping(
            raw_card,
            f"{path}.hbf_endurance.central.cards[{card_index}]",
        )
        budgets.append(_finite(
            card.get("service_lifetime_budget_fraction"),
            (
                f"{path}.hbf_endurance.central.cards[{card_index}]."
                "service_lifetime_budget_fraction"
            ),
            minimum=0.0,
        ))
        per_day.append(_finite(
            card.get("payload_write_bytes_per_day"),
            (
                f"{path}.hbf_endurance.central.cards[{card_index}]."
                "payload_write_bytes_per_day"
            ),
            minimum=0.0,
        ))
    return WearObjectives(
        five_year_budget_fraction=max(budgets),
        payload_write_bytes_per_second=write_bytes / observed,
        hottest_card_write_bytes_per_day=max(per_day),
        hottest_card_share=_optional_nonnegative(
            hotness.get("hottest_card_share"),
            f"{path}.hbf_endurance.hotness.hottest_card_share",
        ),
        card_write_cv=_optional_nonnegative(
            hotness.get("coefficient_of_variation"),
            (
                f"{path}.hbf_endurance.hotness."
                "coefficient_of_variation"
            ),
        ),
    )


@dataclass(frozen=True)
class FinalCandidate:
    key: str
    group_id: str
    hbf_layout: str
    memory_identity: tuple[object, ...]
    migration_policy: str
    canonical_migration_policy: str
    hbf_read_mode: str
    restore_execution_mode: str
    mixed_batch_latency_limit_ms: Optional[int]
    baseline_candidate_key: str
    goodput_mean: float
    goodput_ci95_lower: Optional[float]
    goodput_ci95_upper: Optional[float]
    baseline_goodput_mean: float
    oracle_goodput_mean: float
    first_ttft_p95_ns: Optional[float]
    resume_ttft_p95_ns: Optional[float]
    tpot_p95_ns: Optional[float]
    joint_slo_pass_fraction: float
    performance_statistics: Mapping[str, AggregateStatistic]
    metrics_sha256: str
    runtime: Optional[RuntimeObjectives]
    wear: WearObjectives
    source_index: int

    @property
    def option(self) -> tuple[str, str]:
        return (self.hbf_read_mode, self.restore_execution_mode)

    @property
    def coordinate(self) -> tuple[str, str, str, str]:
        return (
            self.group_id,
            self.canonical_migration_policy,
            self.hbf_read_mode,
            self.restore_execution_mode,
        )


@dataclass(frozen=True)
class LoadedStagedResults:
    source_path: Path
    source_aggregate_sha256: str
    source_payload_sha256: str
    aggregate: Mapping[str, Any]
    frozen_selection: FrozenFinalSelection
    session_rate: float
    seed_ids: tuple[int, ...]
    references: Mapping[str, Mapping[str, Any]]
    candidates: tuple[FinalCandidate, ...]
    runtime_available: bool
    reference_eligible: bool
    reference_eligibility_failures: tuple[str, ...]
    audit_mode: bool


def _validate_reference(
        key: str,
        reference: Mapping[str, Any],
        expected_seed_ids: tuple[int, ...],
) -> float:
    statistics = _performance_statistics(
        reference,
        f"references.{key}",
        expected_seed_ids=expected_seed_ids,
    )
    goodput = statistics[
        "slo_good_output_tokens_per_second"].mean
    if goodput is None or goodput <= 0.0:
        raise SSDHBFFinalResultsError(
            f"references.{key} must have positive output-token goodput")
    return goodput


def _validate_design_spec(
        raw: object,
        path: str,
) -> tuple[
    Mapping[str, Any],
    str,
    str,
    tuple[object, ...],
    str,
    str,
    str,
    Optional[int],
]:
    design = _mapping(raw, path)
    key = design.get("key")
    if not isinstance(key, str) or not key:
        raise SSDHBFFinalResultsError(f"{path}.key must be non-empty")
    layout = design.get("hbf_layout")
    if layout not in SUPPORTED_LAYOUTS:
        raise SSDHBFFinalResultsError(
            f"{path}.hbf_layout is unsupported")
    policy = design.get("migration_policy")
    if (
        not isinstance(policy, str)
        or policy not in SUPPORTED_MIGRATION_POLICIES
    ):
        raise SSDHBFFinalResultsError(
            f"{path}.migration_policy is unsupported")
    read_mode = design.get("hbf_read_mode")
    if read_mode not in SUPPORTED_HBF_READ_MODES:
        raise SSDHBFFinalResultsError(
            f"{path}.hbf_read_mode is unsupported")
    restore_mode = design.get("restore_execution_mode")
    if restore_mode not in SUPPORTED_RESTORE_EXECUTION_MODES:
        raise SSDHBFFinalResultsError(
            f"{path}.restore_execution_mode is unsupported")
    mixed_limit = design.get("mixed_batch_latency_limit_ms")
    if (
        mixed_limit is not None
        and (
            isinstance(mixed_limit, bool)
            or not isinstance(mixed_limit, int)
            or mixed_limit <= 0
        )
    ):
        raise SSDHBFFinalResultsError(
            f"{path}.mixed_batch_latency_limit_ms must be a positive "
            "integer or null")
    for field, expected in (
        ("gpu_host_count", 1),
        ("hbf_host_count", 1),
        ("hbf_card_count", 8),
    ):
        if design.get(field) != expected:
            raise SSDHBFFinalResultsError(
                f"{path}.{field} must equal {expected}")
    memory = _mapping(
        design.get("active_memory"), f"{path}.active_memory")
    identity = _memory_identity(memory)
    return (
        design,
        key,
        str(layout),
        identity,
        str(policy),
        str(read_mode),
        str(restore_mode),
        mixed_limit,
    )


def _candidate_from_row(
        row: object,
        index: int,
        references: Mapping[str, Mapping[str, Any]],
        reference_goodput: Mapping[str, float],
        expected_seed_ids: tuple[int, ...],
        *,
        allow_ineligible_reference: bool,
) -> FinalCandidate:
    path = f"rates[0].designs[{index}]"
    design_row = _mapping(row, path)
    (
        _,
        key,
        layout,
        memory_identity,
        policy,
        read_mode,
        restore_mode,
        mixed_limit,
    ) = _validate_design_spec(design_row.get("design"), f"{path}.design")
    baseline_key = design_row.get("baseline_candidate_key")
    expected_baseline = BASELINE_CANDIDATE_KEYS[restore_mode]
    if baseline_key != expected_baseline:
        raise SSDHBFFinalResultsError(
            f"{path}.baseline_candidate_key does not match restore mode")
    if baseline_key not in references:
        raise SSDHBFFinalResultsError(
            f"{path} references an absent baseline")
    eligibility = _mapping(
        design_row.get("matched_reference_eligibility"),
        f"{path}.matched_reference_eligibility",
    )
    if (
        eligibility.get("eligible") is not True
        and not allow_ineligible_reference
    ):
        raise SSDHBFFinalResultsError(
            f"{path} failed matched reference eligibility")
    metrics = _mapping(design_row.get("metrics"), f"{path}.metrics")
    performance_statistics = _performance_statistics(
        metrics,
        f"{path}.metrics",
        expected_seed_ids=expected_seed_ids,
    )
    goodput_statistic = performance_statistics[
        "slo_good_output_tokens_per_second"]
    goodput = goodput_statistic.mean
    if goodput is None:
        raise AssertionError("required goodput statistic is unavailable")
    lower = goodput_statistic.ci95_lower
    upper = goodput_statistic.ci95_upper
    if goodput < 0.0:
        raise SSDHBFFinalResultsError(
            f"{path} has negative SLO-good output-token goodput")
    joint_slo = performance_statistics[
        "joint_slo_pass_fraction"].mean
    first_ttft = performance_statistics["first_ttft_p95_ns"].mean
    resume_ttft = performance_statistics["resume_ttft_p95_ns"].mean
    tpot = performance_statistics["tpot_p95_ns"].mean
    if joint_slo is None or resume_ttft is None or tpot is None:
        raise AssertionError("required performance statistic is unavailable")
    runtime = _runtime_objectives(design_row, path)
    wear = _wear_objectives(design_row, path)
    return FinalCandidate(
        key=key,
        group_id=_group_id(layout, memory_identity),
        hbf_layout=layout,
        memory_identity=memory_identity,
        migration_policy=policy,
        canonical_migration_policy=policy,
        hbf_read_mode=read_mode,
        restore_execution_mode=restore_mode,
        mixed_batch_latency_limit_ms=mixed_limit,
        baseline_candidate_key=str(baseline_key),
        goodput_mean=goodput,
        goodput_ci95_lower=lower,
        goodput_ci95_upper=upper,
        baseline_goodput_mean=reference_goodput[str(baseline_key)],
        oracle_goodput_mean=reference_goodput[ORACLE_CANDIDATE_KEY],
        first_ttft_p95_ns=first_ttft,
        resume_ttft_p95_ns=resume_ttft,
        tpot_p95_ns=tpot,
        joint_slo_pass_fraction=joint_slo,
        performance_statistics=performance_statistics,
        metrics_sha256=stable_json_sha256(metrics),
        runtime=runtime,
        wear=wear,
        source_index=index,
    )


def _validate_complete_roster(
        candidates: Sequence[FinalCandidate],
        frozen_selection: FrozenFinalSelection,
) -> None:
    by_group: dict[str, list[FinalCandidate]] = {}
    for candidate in candidates:
        by_group.setdefault(candidate.group_id, []).append(candidate)
    if not by_group:
        raise SSDHBFFinalResultsError(
            "staged aggregate contains no design groups")
    expected = {
        (policy, read_mode, restore_mode)
        for policy in frozen_selection.migration_policies
        for read_mode, restore_mode in _REQUIRED_OPTIONS
    }
    for group_id, rows in sorted(by_group.items()):
        observed = {
            (
                candidate.canonical_migration_policy,
                candidate.hbf_read_mode,
                candidate.restore_execution_mode,
            )
            for candidate in rows
        }
        mixed_limits = {
            candidate.mixed_batch_latency_limit_ms
            for candidate in rows
        }
        if mixed_limits != {
                frozen_selection.mixed_batch_latency_limit_ms}:
            raise SSDHBFFinalResultsError(
                f"design group {group_id} does not match the frozen "
                "mixed-batch latency limit")
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise SSDHBFFinalResultsError(
                f"incomplete frozen policy/option roster for "
                f"{group_id}: missing={missing}, extra={extra}")


def load_staged_aggregate(
        path: Path | str,
        *,
        frozen_selection: FrozenFinalSelection,
        allow_ineligible_reference: bool = False,
) -> LoadedStagedResults:
    """Load one SSD-staged aggregate.

    The default remains fail-closed.  An explicitly audit-only aggregate
    may be loaded with ``allow_ineligible_reference=True``; this never
    changes the stored gate outcome and downstream artifacts remain
    visibly marked as ineligible.
    """

    if not isinstance(frozen_selection, FrozenFinalSelection):
        raise SSDHBFFinalResultsError(
            "frozen_selection must be a validated FrozenFinalSelection")
    source_path = Path(path).expanduser().resolve()
    if source_path != frozen_selection.heldout_aggregate_path:
        raise SSDHBFFinalResultsError(
            "aggregate path does not match the frozen heldout output")
    try:
        payload = source_path.read_bytes()
        aggregate = json.loads(
            payload,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}"),
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SSDHBFFinalResultsError(
            f"cannot load staged aggregate {source_path}: {exc}") from exc
    root = _mapping(aggregate, "aggregate")
    if root.get("schema_version") != SSD_HBF_SWEEP_SCHEMA_VERSION:
        raise SSDHBFFinalResultsError(
            "aggregate schema does not match the staged sweep")
    if root.get("comparison_contract") != SSD_HBF_CONTRACT_KEY:
        raise SSDHBFFinalResultsError(
            "aggregate comparison contract is not the SSD-staged "
            "one-GPU plus one-HBF comparison")
    if not _is_sha256(root.get("measurement_roster_sha256")):
        raise SSDHBFFinalResultsError(
            "aggregate has an invalid measurement roster hash")
    if not _is_sha256(root.get("execution_inputs_sha256")):
        raise SSDHBFFinalResultsError(
            "aggregate has an invalid execution-input hash")
    scenario = _mapping(root.get("scenario"), "scenario")
    scenario_id = scenario.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise SSDHBFFinalResultsError(
            "scenario.scenario_id must be non-empty")
    if not _is_sha256(scenario.get("manifest_sha256")):
        raise SSDHBFFinalResultsError(
            "scenario.manifest_sha256 is invalid")
    if scenario.get("measurement_roster_sha256") != (
            root["measurement_roster_sha256"]):
        raise SSDHBFFinalResultsError(
            "scenario and aggregate measurement rosters disagree")
    _positive_int(
        scenario.get("measurement_identity_count"),
        "scenario.measurement_identity_count",
    )
    if _finite(
        scenario.get("required_session_rate"),
        "scenario.required_session_rate",
        positive=True,
    ) != REQUIRED_SESSION_RATE:
        raise SSDHBFFinalResultsError(
            "scenario does not pin the required 3 sessions/s rate")
    endurance_proxy = _mapping(
        root.get("hbf_endurance_proxy_profile"),
        "hbf_endurance_proxy_profile",
    )
    for key in ("profile_id", "vendor", "model", "source_url", "semantics"):
        value = endurance_proxy.get(key)
        if not isinstance(value, str) or not value:
            raise SSDHBFFinalResultsError(
                f"hbf_endurance_proxy_profile.{key} must be non-empty")

    grid = _mapping(root.get("grid"), "grid")
    session_rate = _finite(
        grid.get("session_rate"), "grid.session_rate", positive=True)
    if not math.isclose(
        session_rate,
        frozen_selection.session_rate,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise SSDHBFFinalResultsError(
            "final staged results require the pinned 3 sessions/s rate")
    seeds = _sequence(grid.get("seeds"), "grid.seeds")
    if (
        len(seeds) < 2
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        raise SSDHBFFinalResultsError(
            "grid.seeds must contain at least two unique integers")
    seed_ids = tuple(int(seed) for seed in seeds)
    if seed_ids != frozen_selection.heldout_seed_ids:
        raise SSDHBFFinalResultsError(
            "grid.seeds do not match the frozen heldout seed roster")
    raw_grid_designs = _sequence(grid.get("designs"), "grid.designs")
    if grid.get("design_count") != len(raw_grid_designs):
        raise SSDHBFFinalResultsError(
            "grid.design_count does not match grid.designs")
    if grid.get("reference_count") != len(_REQUIRED_REFERENCE_KEYS):
        raise SSDHBFFinalResultsError(
            "grid.reference_count must cover two matched baselines "
            "and one Oracle")
    expected_cells = (
        len(raw_grid_designs) + len(_REQUIRED_REFERENCE_KEYS)
    ) * len(seeds)
    if grid.get("cell_count") != expected_cells:
        raise SSDHBFFinalResultsError(
            "grid.cell_count is inconsistent with designs, references, "
            "and seeds")
    resumed_count = grid.get("resumed_cell_count")
    executed_count = grid.get("executed_cell_count")
    if (
        isinstance(resumed_count, bool)
        or not isinstance(resumed_count, int)
        or resumed_count < 0
        or isinstance(executed_count, bool)
        or not isinstance(executed_count, int)
        or executed_count < 0
        or resumed_count + executed_count != expected_cells
    ):
        raise SSDHBFFinalResultsError(
            "grid resumed/executed cell counts do not conserve cells")
    grid_designs = {}
    for index, raw_design in enumerate(raw_grid_designs):
        design, key, *_ = _validate_design_spec(
            raw_design, f"grid.designs[{index}]")
        if key in grid_designs:
            raise SSDHBFFinalResultsError(
                f"grid contains duplicate design key {key!r}")
        grid_designs[key] = design

    rates = _sequence(root.get("rates"), "rates")
    if len(rates) != 1:
        raise SSDHBFFinalResultsError(
            "final staged aggregate must contain exactly one pinned rate")
    rate_row = _mapping(rates[0], "rates[0]")
    observed_rate = _finite(
        rate_row.get("session_rate"),
        "rates[0].session_rate",
        positive=True,
    )
    if observed_rate != session_rate:
        raise SSDHBFFinalResultsError(
            "rate row disagrees with grid.session_rate")
    eligibility = _mapping(
        rate_row.get("reference_eligibility"),
        "rates[0].reference_eligibility",
    )
    reference_eligible = eligibility.get("eligible") is True
    raw_failures = _sequence(
        eligibility.get("failures"),
        "rates[0].reference_eligibility.failures",
    )
    if any(
        not isinstance(failure, str) or not failure
        for failure in raw_failures
    ):
        raise SSDHBFFinalResultsError(
            "reference eligibility failures must be non-empty strings")
    reference_failures = tuple(str(value) for value in raw_failures)
    if reference_eligible and reference_failures:
        raise SSDHBFFinalResultsError(
            "eligible reference cannot retain failure reasons")
    if not reference_eligible and not allow_ineligible_reference:
        raise SSDHBFFinalResultsError(
            "final staged aggregate failed reference eligibility")
    if (
        not reference_eligible
        and root.get("reference_eligibility_required") is not False
    ):
        raise SSDHBFFinalResultsError(
            "ineligible reference can only be loaded from an explicit "
            "audit aggregate")
    if not reference_eligible and not reference_failures:
        raise SSDHBFFinalResultsError(
            "ineligible reference must retain failure reasons")
    by_restore = _mapping(
        eligibility.get("by_restore_execution_mode"),
        (
            "rates[0].reference_eligibility."
            "by_restore_execution_mode"
        ),
    )
    if set(by_restore) != set(SUPPORTED_RESTORE_EXECUTION_MODES):
        raise SSDHBFFinalResultsError(
            "reference eligibility must cover bulk and layerwise baselines")
    restore_mode_failed = any(
        _mapping(
            by_restore[mode],
            f"reference_eligibility.by_restore.{mode}",
        ).get("eligible") is not True
        for mode in SUPPORTED_RESTORE_EXECUTION_MODES
    )
    if restore_mode_failed and not allow_ineligible_reference:
        raise SSDHBFFinalResultsError(
            "a matched restore-mode baseline failed eligibility")
    if reference_eligible and restore_mode_failed:
        raise SSDHBFFinalResultsError(
            "aggregate eligibility disagrees with a matched baseline")

    references = _mapping(rate_row.get("references"), "rates[0].references")
    if set(references) != _REQUIRED_REFERENCE_KEYS:
        raise SSDHBFFinalResultsError(
            "references must be exactly the bulk baseline, layerwise "
            "baseline, and performance-only Oracle")
    reference_goodput = {
        key: _validate_reference(
            key,
            _mapping(value, f"references.{key}"),
            seed_ids,
        )
        for key, value in references.items()
    }
    design_rows = _sequence(
        rate_row.get("designs"), "rates[0].designs")
    if len(design_rows) != len(raw_grid_designs):
        raise SSDHBFFinalResultsError(
            "rate design rows do not match grid design count")
    raw_candidates = tuple(
        _candidate_from_row(
            row,
            index,
            references,
            reference_goodput,
            seed_ids,
            allow_ineligible_reference=(
                allow_ineligible_reference),
        )
        for index, row in enumerate(design_rows)
    )
    row_keys = {candidate.key for candidate in raw_candidates}
    if len(row_keys) != len(raw_candidates):
        raise SSDHBFFinalResultsError(
            "rate design rows contain duplicate keys")
    if row_keys != set(grid_designs):
        raise SSDHBFFinalResultsError(
            "rate design keys do not exactly match grid.designs")
    for candidate in raw_candidates:
        row_design = _mapping(
            design_rows[candidate.source_index],
            f"rates[0].designs[{candidate.source_index}]",
        ).get("design")
        if row_design != grid_designs[candidate.key]:
            raise SSDHBFFinalResultsError(
                f"design semantics differ between grid and rate row "
                f"for {candidate.key!r}")

    runtime_presence = {
        candidate.runtime is not None for candidate in raw_candidates
    }
    if len(runtime_presence) != 1:
        raise SSDHBFFinalResultsError(
            "runtime energy/TCO reports are only partially populated")
    runtime_available = runtime_presence == {True}
    candidates = tuple(sorted(
        raw_candidates, key=lambda candidate: candidate.key))
    _validate_complete_roster(candidates, frozen_selection)

    # A matched baseline is a physical reference and therefore cannot vary
    # with the design policy/read mode within a restore-mode cohort.
    if runtime_available:
        for restore_mode in SUPPORTED_RESTORE_EXECUTION_MODES:
            runtime_rows = [
                candidate.runtime
                for candidate in candidates
                if candidate.restore_execution_mode == restore_mode
            ]
            baseline_values = {
                (
                    runtime.baseline_average_it_power_w,
                    runtime.baseline_five_year_facility_energy_kwh,
                    runtime.baseline_five_year_tco_usd,
                )
                for runtime in runtime_rows
                if runtime is not None
            }
            if len(baseline_values) != 1:
                raise SSDHBFFinalResultsError(
                    "runtime matched-baseline values vary within "
                    f"restore mode {restore_mode!r}")

    return LoadedStagedResults(
        source_path=source_path,
        source_aggregate_sha256=_sha256(payload),
        source_payload_sha256=stable_json_sha256(root),
        aggregate=root,
        frozen_selection=frozen_selection,
        session_rate=session_rate,
        seed_ids=seed_ids,
        references={
            key: _mapping(value, f"references.{key}")
            for key, value in references.items()
        },
        candidates=candidates,
        runtime_available=runtime_available,
        reference_eligible=reference_eligible,
        reference_eligibility_failures=reference_failures,
        audit_mode=not reference_eligible,
    )


def _candidate_objectives(
        candidate: FinalCandidate,
) -> dict[str, Optional[float]]:
    runtime = candidate.runtime
    return {
        **{
            spec.aggregate_key: candidate.performance_statistics[
                spec.aggregate_key].mean
            for spec in PERFORMANCE_METRIC_SPECS
        },
        "runtime_five_year_tco_usd": (
            None if runtime is None
            else runtime.proposed_five_year_tco_usd
        ),
        "runtime_five_year_facility_energy_kwh": (
            None if runtime is None
            else runtime.proposed_five_year_facility_energy_kwh
        ),
        "hbf_five_year_budget_fraction_100k_pe_waf1": (
            candidate.wear.five_year_budget_fraction
        ),
    }


def select_meaningful_policies(
        loaded: LoadedStagedResults,
) -> dict[str, Any]:
    """Select exactly the coordinates frozen before heldout execution."""

    if not isinstance(loaded, LoadedStagedResults):
        raise SSDHBFFinalResultsError(
            "loaded must be a validated LoadedStagedResults")
    frozen = loaded.frozen_selection
    frozen_coordinates = set(frozen.coordinates)
    by_coordinate: dict[
        tuple[str, str, str, str], FinalCandidate
    ] = {}
    for candidate in loaded.candidates:
        coordinate = (
            candidate.migration_policy,
            candidate.hbf_layout,
            candidate.hbf_read_mode,
            candidate.restore_execution_mode,
        )
        if coordinate in by_coordinate:
            raise SSDHBFFinalResultsError(
                f"duplicate heldout coordinate {coordinate!r}")
        by_coordinate[coordinate] = candidate
    missing = sorted(frozen_coordinates - set(by_coordinate))
    if missing:
        raise SSDHBFFinalResultsError(
            f"heldout aggregate is missing frozen coordinates: {missing}")
    selected_keys = {
        by_coordinate[coordinate].key
        for coordinate in frozen_coordinates
    }
    frozen_restore_by_projection = {
        coordinate[:3]: coordinate[3]
        for coordinate in frozen_coordinates
    }
    if len(selected_keys) != FINAL_PLOT_DESIGN_CELL_COUNT:
        raise SSDHBFFinalResultsError(
            "frozen selection did not resolve to exactly eight candidates")

    audit_by_key = {}
    for candidate in sorted(
            loaded.candidates, key=lambda value: value.key):
        selected = candidate.key in selected_keys
        projection = (
            candidate.migration_policy,
            candidate.hbf_layout,
            candidate.hbf_read_mode,
        )
        excluded_restore_sibling = (
            not selected
            and projection in frozen_restore_by_projection
            and candidate.restore_execution_mode
            != frozen_restore_by_projection[projection]
        )
        audit_by_key[candidate.key] = {
            "candidate_key": candidate.key,
            "group_id": candidate.group_id,
            "hbf_layout": candidate.hbf_layout,
            "migration_policy": candidate.migration_policy,
            "canonical_migration_policy": (
                candidate.canonical_migration_policy),
            "hbf_read_mode": candidate.hbf_read_mode,
            "restore_execution_mode": (
                candidate.restore_execution_mode),
            "mixed_batch_latency_limit_ms": (
                candidate.mixed_batch_latency_limit_ms),
            "objectives": _candidate_objectives(candidate),
            "selected": selected,
            "selection_reasons": (
                [
                    "exact_coordinate_in_frozen_pre_heldout_manifest:"
                    f"{frozen.source_file_sha256}"
                ]
                if selected else []
            ),
            "exclusion_reasons": (
                []
                if selected else [
                    (
                        "restore_mode_not_frozen_from_discovery"
                        if excluded_restore_sibling
                        else
                        "coordinate_not_in_frozen_pre_heldout_manifest"
                    )
                ]
            ),
            "dominators": [],
        }
    groups = []
    by_group: dict[str, list[FinalCandidate]] = {}
    for candidate in loaded.candidates:
        by_group.setdefault(candidate.group_id, []).append(candidate)
    for group_id, candidates in sorted(by_group.items()):
        sample = candidates[0]
        groups.append({
            "group_id": group_id,
            "hbf_layout": sample.hbf_layout,
            "active_memory_identity": list(sample.memory_identity),
            "candidate_count": len(candidates),
            "selected_candidate_keys": sorted(
                candidate.key
                for candidate in candidates
                if candidate.key in selected_keys
            ),
        })

    report: dict[str, Any] = {
        "final_results_schema_version": FINAL_RESULTS_SCHEMA_VERSION,
        "schema_version": POLICY_SELECTION_SCHEMA_VERSION,
        "source": {
            "aggregate_path": str(loaded.source_path),
            "aggregate_file_sha256": loaded.source_aggregate_sha256,
            "aggregate_payload_sha256": loaded.source_payload_sha256,
            "comparison_contract": SSD_HBF_CONTRACT_KEY,
            "session_rate": loaded.session_rate,
            "result_status": (
                "audit_reference_ineligible"
                if loaded.audit_mode else "eligible_final"
            ),
            "reference_eligible": loaded.reference_eligible,
            "reference_eligibility_failures": list(
                loaded.reference_eligibility_failures),
            "seed_ids": list(loaded.seed_ids),
            "frozen_selection_path": str(frozen.source_path),
            "frozen_selection_file_sha256": (
                frozen.source_file_sha256),
            "frozen_selection_payload_sha256": (
                frozen.source_payload_sha256),
            "discovery_aggregate_path": str(
                frozen.discovery_aggregate_path),
            "discovery_aggregate_sha256": (
                frozen.discovery_aggregate_sha256),
        },
        "selection_algorithm": {
            "selection_status": "frozen_before_heldout",
            "retention_rule": (
                "exact membership in restore_by_coordinate from the "
                "validated frozen selection manifest"),
            "heldout_metrics_used_for_selection": False,
            "rendered_design_cell_count": FINAL_PLOT_DESIGN_CELL_COUNT,
            "migration_policies": list(frozen.migration_policies),
            "mixed_batch_latency_limit_ms": (
                frozen.mixed_batch_latency_limit_ms),
            "frozen_coordinates": [
                {
                    "migration_policy": coordinate[0],
                    "hbf_layout": coordinate[1],
                    "hbf_read_mode": coordinate[2],
                    "restore_execution_mode": coordinate[3],
                }
                for coordinate in frozen.coordinates
            ],
            "oracle_semantics": (
                "performance-only upper reference; excluded from power, "
                "energy, TCO, and endurance"),
        },
        "runtime_objectives_available": loaded.runtime_available,
        "audit_mode": loaded.audit_mode,
        "groups": groups,
        "selected_candidate_keys": sorted(selected_keys),
        "candidate_audit": [
            audit_by_key[key] for key in sorted(audit_by_key)
        ],
    }
    report["policy_selection_sha256"] = stable_json_sha256(report)
    return report


_PERFORMANCE_PLOT_SOURCE_FIELDS = tuple(
    field
    for spec in PERFORMANCE_METRIC_SPECS
    for field in (
        spec.row_field,
        f"{spec.row_field}_ci95_lower",
        f"{spec.row_field}_ci95_upper",
    )
)


_PLOT_SOURCE_FIELDS = (
    "plot_source_schema_version",
    "source_aggregate_sha256",
    "result_status",
    "reference_eligible",
    "reference_eligibility_failures",
    "seed_count",
    "seed_ids",
    "candidate_kind",
    "candidate_key",
    "group_id",
    "hbf_layout",
    "active_memory_kind",
    "active_memory_gib_per_card",
    "active_memory_gbps_per_card",
    "migration_policy",
    "canonical_migration_policy",
    "hbf_read_mode",
    "restore_execution_mode",
    "baseline_candidate_key",
    "include_in_final_plots",
    "selection_reasons",
    "exclusion_reasons",
    *_PERFORMANCE_PLOT_SOURCE_FIELDS,
    "baseline_goodput_mean",
    "oracle_goodput_mean",
    "goodput_ratio_to_baseline",
    "oracle_goodput_ratio_to_baseline",
    "baseline_runtime_average_it_power_w",
    "runtime_average_it_power_w",
    "runtime_power_ratio_to_baseline",
    "baseline_runtime_five_year_facility_energy_kwh",
    "runtime_five_year_facility_energy_kwh",
    "runtime_energy_ratio_to_baseline",
    "baseline_runtime_five_year_tco_usd",
    "runtime_five_year_tco_usd",
    "runtime_tco_ratio_to_baseline",
    "hbf_payload_write_bytes_per_second",
    "hbf_hottest_card_write_bytes_per_day",
    "hbf_five_year_budget_fraction_100k_pe_waf1",
    "hbf_hottest_card_share",
    "hbf_card_write_cv",
    "oracle_performance_only",
)


def _performance_row_values(
        statistics: Mapping[str, AggregateStatistic],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for spec in PERFORMANCE_METRIC_SPECS:
        statistic = statistics[spec.aggregate_key]
        result[spec.row_field] = (
            "" if statistic.mean is None else statistic.mean)
        result[f"{spec.row_field}_ci95_lower"] = (
            "" if statistic.ci95_lower is None
            else statistic.ci95_lower
        )
        result[f"{spec.row_field}_ci95_upper"] = (
            "" if statistic.ci95_upper is None
            else statistic.ci95_upper
        )
    return result


def _reference_plot_rows(
        loaded: LoadedStagedResults,
) -> list[dict[str, Any]]:
    rows = []
    runtime_by_baseline: dict[str, RuntimeObjectives] = {}
    if loaded.runtime_available:
        for candidate in loaded.candidates:
            if candidate.runtime is not None:
                runtime_by_baseline.setdefault(
                    candidate.baseline_candidate_key,
                    candidate.runtime,
                )
    for key in sorted(loaded.references):
        reference = loaded.references[key]
        performance_statistics = _performance_statistics(
            reference,
            f"references.{key}",
            expected_seed_ids=loaded.seed_ids,
        )
        is_oracle = key == ORACLE_CANDIDATE_KEY
        runtime = None if is_oracle else runtime_by_baseline.get(key)
        rows.append({
            "candidate_kind": "oracle" if is_oracle else "baseline",
            "candidate_key": key,
            "group_id": "reference",
            "restore_execution_mode": (
                ""
                if is_oracle else next(
                    mode for mode, baseline_key
                    in BASELINE_CANDIDATE_KEYS.items()
                    if baseline_key == key
                )
            ),
            "include_in_final_plots": True,
            "selection_reasons": (
                "performance_only_upper_reference"
                if is_oracle
                else "matched_restore_mode_physical_baseline"
            ),
            "exclusion_reasons": "",
            **_performance_row_values(performance_statistics),
            "baseline_runtime_average_it_power_w": (
                ""
                if runtime is None
                else runtime.baseline_average_it_power_w
            ),
            "runtime_average_it_power_w": (
                ""
                if runtime is None
                else runtime.baseline_average_it_power_w
            ),
            "runtime_power_ratio_to_baseline": (
                "" if runtime is None else 1.0),
            "baseline_runtime_five_year_facility_energy_kwh": (
                ""
                if runtime is None
                else runtime.baseline_five_year_facility_energy_kwh
            ),
            "runtime_five_year_facility_energy_kwh": (
                ""
                if runtime is None
                else runtime.baseline_five_year_facility_energy_kwh
            ),
            "runtime_energy_ratio_to_baseline": (
                "" if runtime is None else 1.0),
            "baseline_runtime_five_year_tco_usd": (
                ""
                if runtime is None
                else runtime.baseline_five_year_tco_usd
            ),
            "runtime_five_year_tco_usd": (
                ""
                if runtime is None
                else runtime.baseline_five_year_tco_usd
            ),
            "runtime_tco_ratio_to_baseline": (
                "" if runtime is None else 1.0),
            "oracle_performance_only": is_oracle,
        })
    return rows


def _validate_selection_for_loaded(
        loaded: LoadedStagedResults,
        selection: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    if selection.get("final_results_schema_version") != (
            FINAL_RESULTS_SCHEMA_VERSION):
        raise SSDHBFFinalResultsError(
            "selection final-results schema is unsupported")
    if selection.get("schema_version") != POLICY_SELECTION_SCHEMA_VERSION:
        raise SSDHBFFinalResultsError(
            "selection policy schema is unsupported")
    claimed_digest = selection.get("policy_selection_sha256")
    if not _is_sha256(claimed_digest):
        raise SSDHBFFinalResultsError(
            "selection has an invalid policy-selection hash")
    unhashed = dict(selection)
    unhashed.pop("policy_selection_sha256", None)
    if stable_json_sha256(unhashed) != claimed_digest:
        raise SSDHBFFinalResultsError(
            "selection policy-selection hash does not match its payload")
    source = _mapping(selection.get("source"), "selection.source")
    if (
        source.get("aggregate_file_sha256")
        != loaded.source_aggregate_sha256
        or source.get("aggregate_payload_sha256")
        != loaded.source_payload_sha256
        or source.get("frozen_selection_file_sha256")
        != loaded.frozen_selection.source_file_sha256
        or source.get("frozen_selection_payload_sha256")
        != loaded.frozen_selection.source_payload_sha256
    ):
        raise SSDHBFFinalResultsError(
            "selection does not belong to the loaded aggregate and frozen "
            "manifest")
    audit_values = tuple(
        _mapping(row, "selection.candidate_audit[]")
        for row in _sequence(
            selection.get("candidate_audit"),
            "selection.candidate_audit",
        )
    )
    audit_keys = tuple(
        str(row.get("candidate_key")) for row in audit_values)
    expected_audit_keys = {
        candidate.key for candidate in loaded.candidates
    }
    if (
        len(audit_keys) != len(set(audit_keys))
        or set(audit_keys) != expected_audit_keys
    ):
        raise SSDHBFFinalResultsError(
            "selection candidate audit does not exactly cover the raw "
            "design roster")
    selected_keys = {
        str(value)
        for value in _sequence(
            selection.get("selected_candidate_keys"),
            "selection.selected_candidate_keys",
        )
    }
    audited_selected_keys = {
        str(row["candidate_key"])
        for row in audit_values
        if row.get("selected") is True
    }
    if (
        selected_keys != audited_selected_keys
        or len(selected_keys) != FINAL_PLOT_DESIGN_CELL_COUNT
    ):
        raise SSDHBFFinalResultsError(
            "selection selected keys disagree with its candidate audit")
    for row in audit_values:
        selected = row.get("selected")
        selection_reasons = _sequence(
            row.get("selection_reasons"),
            "selection.candidate_audit[].selection_reasons",
        )
        exclusion_reasons = _sequence(
            row.get("exclusion_reasons"),
            "selection.candidate_audit[].exclusion_reasons",
        )
        if selected is True:
            if not selection_reasons or exclusion_reasons:
                raise SSDHBFFinalResultsError(
                    "selected audit rows require selection reasons and "
                    "cannot contain exclusion reasons")
        elif selected is False:
            if selection_reasons or not exclusion_reasons:
                raise SSDHBFFinalResultsError(
                    "excluded audit rows require exclusion reasons and "
                    "cannot contain selection reasons")
        else:
            raise SSDHBFFinalResultsError(
                "candidate audit selected must be boolean")
    return audit_values


def build_plot_source_rows(
        loaded: LoadedStagedResults,
        selection: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return deterministic full-audit rows; renderers filter selected."""

    audit_values = _validate_selection_for_loaded(loaded, selection)
    audit_by_key = {
        str(_mapping(row, "candidate_audit[]").get("candidate_key")):
        _mapping(row, "candidate_audit[]")
        for row in audit_values
    }
    candidate_by_key = {
        candidate.key: candidate for candidate in loaded.candidates
    }
    rows = _reference_plot_rows(loaded)
    for key in sorted(audit_by_key):
        audit = audit_by_key[key]
        candidate = candidate_by_key.get(key)
        if candidate is None:
            raise SSDHBFFinalResultsError(
                f"audit contains unknown candidate {key!r}")
        memory = candidate.memory_identity
        runtime = candidate.runtime
        rows.append({
            "candidate_kind": "design",
            "candidate_key": key,
            "group_id": candidate.group_id,
            "hbf_layout": candidate.hbf_layout,
            "active_memory_kind": memory[0],
            "active_memory_gib_per_card": memory[1],
            "active_memory_gbps_per_card": memory[2],
            "migration_policy": audit["migration_policy"],
            "canonical_migration_policy": (
                audit["canonical_migration_policy"]),
            "hbf_read_mode": candidate.hbf_read_mode,
            "restore_execution_mode": (
                candidate.restore_execution_mode),
            "baseline_candidate_key": (
                candidate.baseline_candidate_key),
            "include_in_final_plots": audit["selected"],
            "selection_reasons": "|".join(
                audit["selection_reasons"]),
            "exclusion_reasons": "|".join(
                audit["exclusion_reasons"]),
            **_performance_row_values(
                candidate.performance_statistics),
            "baseline_goodput_mean": (
                candidate.baseline_goodput_mean),
            "oracle_goodput_mean": candidate.oracle_goodput_mean,
            "goodput_ratio_to_baseline": (
                candidate.goodput_mean
                / candidate.baseline_goodput_mean
            ),
            "oracle_goodput_ratio_to_baseline": (
                candidate.oracle_goodput_mean
                / candidate.baseline_goodput_mean
            ),
            "baseline_runtime_average_it_power_w": (
                "" if runtime is None
                else runtime.baseline_average_it_power_w
            ),
            "runtime_average_it_power_w": (
                "" if runtime is None
                else runtime.proposed_average_it_power_w
            ),
            "runtime_power_ratio_to_baseline": (
                "" if runtime is None else runtime.power_ratio),
            "baseline_runtime_five_year_facility_energy_kwh": (
                "" if runtime is None
                else runtime.baseline_five_year_facility_energy_kwh
            ),
            "runtime_five_year_facility_energy_kwh": (
                "" if runtime is None
                else runtime.proposed_five_year_facility_energy_kwh
            ),
            "runtime_energy_ratio_to_baseline": (
                "" if runtime is None else runtime.energy_ratio),
            "baseline_runtime_five_year_tco_usd": (
                "" if runtime is None
                else runtime.baseline_five_year_tco_usd
            ),
            "runtime_five_year_tco_usd": (
                "" if runtime is None
                else runtime.proposed_five_year_tco_usd
            ),
            "runtime_tco_ratio_to_baseline": (
                "" if runtime is None else runtime.tco_ratio),
            "hbf_payload_write_bytes_per_second": (
                candidate.wear.payload_write_bytes_per_second),
            "hbf_hottest_card_write_bytes_per_day": (
                candidate.wear.hottest_card_write_bytes_per_day),
            "hbf_five_year_budget_fraction_100k_pe_waf1": (
                candidate.wear.five_year_budget_fraction),
            "hbf_hottest_card_share": (
                ""
                if candidate.wear.hottest_card_share is None
                else candidate.wear.hottest_card_share
            ),
            "hbf_card_write_cv": (
                ""
                if candidate.wear.card_write_cv is None
                else candidate.wear.card_write_cv
            ),
            "oracle_performance_only": False,
        })
    normalized = []
    for row in rows:
        normalized.append({
            field: (
                PLOT_SOURCE_SCHEMA_VERSION
                if field == "plot_source_schema_version"
                else loaded.source_aggregate_sha256
                if field == "source_aggregate_sha256"
                else (
                    "audit_reference_ineligible"
                    if loaded.audit_mode
                    else "eligible_final"
                )
                if field == "result_status"
                else loaded.reference_eligible
                if field == "reference_eligible"
                else "|".join(
                    loaded.reference_eligibility_failures)
                if field == "reference_eligibility_failures"
                else len(loaded.seed_ids)
                if field == "seed_count"
                else ",".join(str(seed) for seed in loaded.seed_ids)
                if field == "seed_ids"
                else row.get(field, "")
            )
            for field in _PLOT_SOURCE_FIELDS
        })
    return tuple(normalized)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv_atomic(
        path: Path,
        rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=_PLOT_SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _selected_design_rows(
        rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    selected = [
        row for row in rows
        if (
            row["candidate_kind"] == "design"
            and row["include_in_final_plots"] is True
        )
    ]
    if len(selected) != FINAL_PLOT_DESIGN_CELL_COUNT:
        raise SSDHBFFinalResultsError(
            "plot source must select exactly "
            f"{FINAL_PLOT_DESIGN_CELL_COUNT} design cells")
    observed = {
        (
            row["hbf_layout"],
            row["canonical_migration_policy"],
            row["hbf_read_mode"],
            row["restore_execution_mode"],
        )
        for row in selected
    }
    projected = {
        coordinate[:3] for coordinate in observed}
    if (
        len(observed) != FINAL_PLOT_DESIGN_CELL_COUNT
        or len(projected) != FINAL_PLOT_DESIGN_CELL_COUNT
    ):
        raise SSDHBFFinalResultsError(
            "selected plot rows must contain one frozen restore choice per "
            "policy/layout/read-mode coordinate")
    return sorted(
        selected,
        key=lambda row: (
            str(row["hbf_layout"]),
            str(row["canonical_migration_policy"]),
            str(row["hbf_read_mode"]),
            str(row["restore_execution_mode"]),
        ),
    )


def _short_label(row: Mapping[str, Any]) -> str:
    read = "prefetch" if row["hbf_read_mode"] == "prefetch" else "demand"
    layout = (
        "TP4×2"
        if row["hbf_layout"] == "tp4x2"
        else "TP8-context"
    )
    policy = str(row["canonical_migration_policy"]).replace("_", " ")
    restore = (
        "stream"
        if row["restore_execution_mode"] == "layerwise_streaming"
        else "bulk"
    )
    return (
        f"{layout} | {policy} | {read} | {restore}"
    )


def _reference_row(
        rows: Sequence[Mapping[str, Any]],
        candidate_key: str,
) -> Mapping[str, Any]:
    matches = [
        row for row in rows
        if row.get("candidate_key") == candidate_key
    ]
    if len(matches) != 1:
        raise SSDHBFFinalResultsError(
            f"plot source must contain one {candidate_key!r} reference")
    return matches[0]


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


def _render_endurance(
        pyplot,
        rows: Sequence[Mapping[str, Any]],
        output_path: Path,
) -> None:
    selected = _selected_design_rows(rows)
    labels = [_short_label(row) for row in selected]
    positions = list(range(len(selected)))
    write_gb_day = [
        float(row["hbf_hottest_card_write_bytes_per_day"]) / 1e9
        for row in selected
    ]
    budget_percent = [
        100.0
        * float(row[
            "hbf_five_year_budget_fraction_100k_pe_waf1"])
        for row in selected
    ]
    width = max(9.5, 0.44 * len(selected) + 3.0)
    figure, axes = pyplot.subplots(
        2, 1, figsize=(width, 6.6), sharex=True)
    axes[0].bar(positions, write_gb_day, color="#B279A2")
    axes[0].set_ylabel("GB/day")
    axes[0].set_title(
        "Hottest-card recurring KV write rate", loc="left", fontsize=10)
    axes[1].bar(positions, budget_percent, color="#E45756")
    axes[1].axhline(
        100.0, color="#666666", linestyle="--", linewidth=1,
        label="5-year wear budget")
    axes[1].set_ylabel("Budget used (%)")
    axes[1].set_title(
        "5-year 100K P/E, WAF=1 endurance budget",
        loc="left",
        fontsize=10,
    )
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
    axes[-1].set_xticks(
        positions, labels=labels, rotation=55, ha="right", fontsize=7)
    figure.suptitle(
        _audit_title_prefix(rows)
        + "Frozen heldout coordinates: HBF endurance from recurring writes "
        "(uniform within-card spreading)")
    figure.tight_layout()
    _save_figure(figure, output_path)
    pyplot.close(figure)


def _audit_title_prefix(
        rows: Sequence[Mapping[str, Any]],
) -> str:
    statuses = {row.get("result_status") for row in rows}
    if len(statuses) != 1:
        raise SSDHBFFinalResultsError(
            "plot source contains inconsistent result status")
    return (
        "AUDIT — reference eligibility failed\n"
        if statuses == {"audit_reference_ineligible"} else ""
    )


def _scaled_statistic_from_row(
        row: Mapping[str, Any],
        spec: PerformanceMetricSpec,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    value = row[spec.row_field]
    if value == "":
        return None, None, None
    mean = float(value) * spec.scale
    lower_value = row[f"{spec.row_field}_ci95_lower"]
    upper_value = row[f"{spec.row_field}_ci95_upper"]
    lower = (
        None if lower_value == "" else float(lower_value) * spec.scale)
    upper = (
        None if upper_value == "" else float(upper_value) * spec.scale)
    return mean, lower, upper


def _render_performance_metric(
        pyplot,
        rows: Sequence[Mapping[str, Any]],
        spec: PerformanceMetricSpec,
        output_path: Path,
) -> None:
    selected = _selected_design_rows(rows)
    labels = [_short_label(row) for row in selected]
    positions = list(range(len(selected)))
    colors = [
        "#4C78A8" if row["hbf_read_mode"] == "demand"
        else "#F58518"
        for row in selected
    ]
    width = max(11.0, 0.62 * len(selected) + 4.0)
    figure, axis = pyplot.subplots(figsize=(width, 5.8))
    any_value = False
    for position, row, color in zip(positions, selected, colors):
        mean, lower, upper = _scaled_statistic_from_row(row, spec)
        if mean is None:
            continue
        any_value = True
        yerr = None
        if lower is not None and upper is not None:
            yerr = [[max(0.0, mean - lower)], [max(0.0, upper - mean)]]
        axis.bar(
            position,
            mean,
            color=color,
            alpha=0.88,
            yerr=yerr,
            capsize=4 if yerr is not None else 0,
        )

    seen_baseline_modes = set()
    for position, row in zip(positions, selected):
        baseline = _reference_row(
            rows, str(row["baseline_candidate_key"]))
        mean, lower, upper = _scaled_statistic_from_row(
            baseline, spec)
        if mean is None:
            continue
        mode = str(row["restore_execution_mode"])
        yerr = None
        if lower is not None and upper is not None:
            yerr = [[max(0.0, mean - lower)], [max(0.0, upper - mean)]]
        axis.errorbar(
            [position],
            [mean],
            yerr=yerr,
            fmt="x" if mode == "bulk" else "+",
            color="#555555" if mode == "bulk" else "#111111",
            markersize=8,
            capsize=3,
            label=(
                f"Matched {mode.replace('_', ' ')} baseline"
                if mode not in seen_baseline_modes else None
            ),
            zorder=4,
        )
        seen_baseline_modes.add(mode)

    oracle = _reference_row(rows, ORACLE_CANDIDATE_KEY)
    oracle_mean, oracle_lower, oracle_upper = (
        _scaled_statistic_from_row(oracle, spec))
    if oracle_mean is not None:
        axis.axhline(
            oracle_mean,
            color="#111111",
            linestyle=":",
            linewidth=1.4,
            label="Infinite-HBM Oracle",
        )
        if oracle_lower is not None and oracle_upper is not None:
            axis.axhspan(
                oracle_lower,
                oracle_upper,
                color="#111111",
                alpha=0.06,
            )
    if not any_value:
        axis.text(
            0.5,
            0.5,
            "N/A — no eligible first-turn samples",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
        axis.set_yticks([])
    elif spec.log_scale:
        axis.set_yscale("log")
    if spec.bounded_fraction:
        axis.set_ylim(0.0, 1.05)
    axis.set_xticks(
        positions, labels=labels, rotation=48, ha="right", fontsize=7)
    axis.set_ylabel(spec.y_label)
    axis.set_title(
        _audit_title_prefix(rows)
        + f"Frozen heldout coordinates: {spec.title}")
    axis.grid(axis="y", alpha=0.22)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    _save_figure(figure, output_path)
    pyplot.close(figure)


def _render_power_energy(
        pyplot,
        rows: Sequence[Mapping[str, Any]],
        output_path: Path,
) -> None:
    selected = _selected_design_rows(rows)
    labels = [_short_label(row) for row in selected]
    metrics = (
        ("runtime_power_ratio_to_baseline", "Average IT power"),
        ("runtime_energy_ratio_to_baseline", "5-year facility energy"),
    )
    positions = list(range(len(selected)))
    width = max(10.0, 0.55 * len(selected) + 3.0)
    figure, axes = pyplot.subplots(
        2, 1, figsize=(width, 6.8), sharex=True)
    for axis, (field, title) in zip(axes, metrics):
        axis.bar(
            positions,
            [float(row[field]) for row in selected],
            color="#54A24B",
            alpha=0.88,
        )
        axis.axhline(1.0, color="#666666", linestyle="--", linewidth=1)
        axis.set_ylabel("Design / matched baseline")
        axis.set_title(title, loc="left", fontsize=10)
        axis.grid(axis="y", alpha=0.22)
    axes[-1].set_xticks(
        positions, labels=labels, rotation=48, ha="right", fontsize=7)
    figure.suptitle(
        _audit_title_prefix(rows)
        + "Frozen heldout coordinates: runtime power and energy")
    figure.tight_layout()
    _save_figure(figure, output_path)
    pyplot.close(figure)


def _render_five_year_tco(
        pyplot,
        rows: Sequence[Mapping[str, Any]],
        output_path: Path,
) -> None:
    selected = _selected_design_rows(rows)
    labels = [_short_label(row) for row in selected]
    positions = list(range(len(selected)))
    figure, axis = pyplot.subplots(
        figsize=(max(10.0, 0.55 * len(selected) + 3.0), 5.4))
    axis.bar(
        positions,
        [
            float(row["runtime_tco_ratio_to_baseline"])
            for row in selected
        ],
        color="#ECA82C",
        alpha=0.9,
    )
    axis.axhline(
        1.0,
        color="#666666",
        linestyle="--",
        linewidth=1,
        label="Matched two-GPU baseline",
    )
    axis.set_xticks(
        positions, labels=labels, rotation=48, ha="right", fontsize=7)
    axis.set_ylabel("Design / matched baseline")
    axis.set_title(
        _audit_title_prefix(rows)
        + "Frozen heldout coordinates: 5-year TCO")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    _save_figure(figure, output_path)
    pyplot.close(figure)


@dataclass(frozen=True)
class FinalPlotArtifacts:
    policy_selection_json: Path
    plot_source_csv: Path
    performance_metric_pngs: Mapping[str, Path]
    runtime_power_energy_png: Optional[Path]
    five_year_tco_png: Optional[Path]
    hbf_endurance_png: Optional[Path]
    rendered: bool
    matplotlib_available: bool
    source_aggregate_sha256: str
    policy_selection_sha256: str

    def to_json_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in tuple(result.items()):
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, Mapping):
                result[key] = {
                    nested_key: (
                        str(nested_value)
                        if isinstance(nested_value, Path)
                        else nested_value
                    )
                    for nested_key, nested_value in value.items()
                }
        return result


def write_final_artifacts(
        loaded: LoadedStagedResults,
        output_dir: Path | str,
        *,
        render: bool = True,
) -> FinalPlotArtifacts:
    """Write audited sources and exactly ten final graphs when rendered."""

    if render and not loaded.runtime_available:
        raise SSDHBFFinalResultsError(
            "final rendering requires complete event-derived runtime "
            "power, five-year facility energy, and five-year TCO fields; "
            "use render=False only for a provisional selection audit")
    selection = select_meaningful_policies(loaded)
    rows = build_plot_source_rows(loaded, selection)
    root = Path(output_dir).expanduser().resolve()
    selection_path = root / "policy_selection.json"
    source_path = root / "plot_source.csv"
    _write_json_atomic(selection_path, selection)
    _write_csv_atomic(source_path, rows)

    pyplot = _load_pyplot() if render else None
    matplotlib_available = pyplot is not None
    performance_paths: dict[str, Path] = {}
    power_energy_path = tco_path = endurance_path = None
    if render and pyplot is not None:
        prefix = "audit_" if loaded.audit_mode else ""
        for spec in PERFORMANCE_METRIC_SPECS:
            path = root / f"{prefix}{spec.filename_stem}.png"
            _render_performance_metric(pyplot, rows, spec, path)
            performance_paths[spec.plot_key] = path
        power_energy_path = (
            root / f"{prefix}08_power_energy.png")
        tco_path = root / f"{prefix}09_five_year_tco.png"
        endurance_path = root / f"{prefix}10_endurance.png"
        _render_power_energy(pyplot, rows, power_energy_path)
        _render_five_year_tco(pyplot, rows, tco_path)
        _render_endurance(pyplot, rows, endurance_path)
    return FinalPlotArtifacts(
        policy_selection_json=selection_path,
        plot_source_csv=source_path,
        performance_metric_pngs=performance_paths,
        runtime_power_energy_png=power_energy_path,
        five_year_tco_png=tco_path,
        hbf_endurance_png=endurance_path,
        rendered=render and pyplot is not None,
        matplotlib_available=matplotlib_available,
        source_aggregate_sha256=loaded.source_aggregate_sha256,
        policy_selection_sha256=str(
            selection["policy_selection_sha256"]),
    )


def generate_final_results(
        aggregate_path: Path | str,
        output_dir: Path | str,
        *,
        selection_config_path: Path | str,
        repo_root: Optional[Path | str] = None,
        render: bool = True,
        allow_ineligible_reference_audit: bool = False,
) -> FinalPlotArtifacts:
    """Convenience API: strict load, select, export, and render."""

    frozen_selection = load_frozen_selection(
        selection_config_path,
        repo_root=repo_root,
    )
    return write_final_artifacts(
        load_staged_aggregate(
            aggregate_path,
            frozen_selection=frozen_selection,
            allow_ineligible_reference=(
                allow_ineligible_reference_audit),
        ),
        output_dir,
        render=render,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select and render final results from the SSD-staged "
            "two-GPU-vs-one-GPU-one-HBF aggregate."
        )
    )
    parser.add_argument("aggregate", type=Path)
    parser.add_argument(
        "--selection-config",
        required=True,
        type=Path,
        help="pre-heldout frozen eight-coordinate selection JSON",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root used to resolve paths in the selection JSON",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Render PNG graphs; requires complete runtime energy/TCO "
            "fields. Use --no-render only for provisional audit files."
        ),
    )
    parser.add_argument(
        "--allow-ineligible-reference-audit",
        action="store_true",
        help=(
            "Render an explicitly audit-only aggregate whose baseline/"
            "Oracle eligibility gate failed. Output PNG names and titles "
            "are marked AUDIT; strict mode remains the default."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    artifacts = generate_final_results(
        args.aggregate,
        args.output_dir,
        selection_config_path=args.selection_config,
        repo_root=args.repo_root,
        render=args.render,
        allow_ineligible_reference_audit=(
            args.allow_ineligible_reference_audit),
    )
    print(json.dumps(
        artifacts.to_json_dict(),
        sort_keys=True,
        allow_nan=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CENTRAL_ENDURANCE_SCENARIO",
    "FINAL_PLOT_DESIGN_CELL_COUNT",
    "FINAL_RESULTS_SCHEMA_VERSION",
    "FinalCandidate",
    "FinalPlotArtifacts",
    "FrozenFinalSelection",
    "LoadedStagedResults",
    "PERFORMANCE_METRIC_SPECS",
    "PERFORMANCE_PLOT_COUNT",
    "POLICY_SELECTION_SCHEMA_VERSION",
    "PLOT_SOURCE_SCHEMA_VERSION",
    "RUNTIME_PROJECTION_METRIC_KEYS",
    "RUNTIME_REPORT_FIELD_KEYS",
    "SSDHBFFinalResultsError",
    "build_plot_source_rows",
    "generate_final_results",
    "load_frozen_selection",
    "load_staged_aggregate",
    "main",
    "select_meaningful_policies",
    "write_final_artifacts",
]

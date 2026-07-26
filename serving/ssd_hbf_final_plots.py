"""Final-policy selection and plots for the SSD-staged HBF campaign.

This module intentionally accepts only the aggregate produced by
``ssd_hbf_design_sweep``.  It does not merge the older direct HBM-to-HBF
migration sweep.  Selection is auditable rather than hand-curated:

* every comparable layout/active-memory group must contain the complete
  canonical migration-policy roster;
* every policy must contain demand/prefetch crossed with bulk/layerwise
  restore execution;
* the best-goodput candidate in each read/restore option is retained;
* all candidates that are nondominated in goodput, runtime five-year TCO,
  runtime five-year facility energy, and HBF wear are retained.

The historical ``delay_1s`` spelling is an alias for ``delay_1000ms``.
When both are present, the canonical record is used and the alias remains
visible in the audit with an explicit exclusion reason.
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
    CANONICAL_MIGRATION_POLICIES,
    ORACLE_CANDIDATE_KEY,
    REQUIRED_SESSION_RATE,
    SSD_HBF_CONTRACT_KEY,
    SSD_HBF_SWEEP_SCHEMA_VERSION,
    SUPPORTED_HBF_READ_MODES,
    SUPPORTED_LAYOUTS,
    SUPPORTED_RESTORE_EXECUTION_MODES,
)


FINAL_RESULTS_SCHEMA_VERSION = 2
POLICY_SELECTION_SCHEMA_VERSION = 2
PLOT_SOURCE_SCHEMA_VERSION = 2
DELAY_POLICY_ALIASES = {"delay_1s": "delay_1000ms"}
CENTRAL_ENDURANCE_SCENARIO = "slc_100k_pe_waf1"
RUNTIME_TCO_REPORT_SCHEMA = "ssd-hbf-runtime-tco-v1"

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
_SELECTION_OBJECTIVES_WITH_RUNTIME = (
    "goodput_max",
    "runtime_five_year_tco_min",
    "runtime_five_year_facility_energy_min",
    "hbf_five_year_wear_min",
)
_SELECTION_OBJECTIVES_WITHOUT_RUNTIME = (
    "goodput_max",
    "hbf_five_year_wear_min",
)
MIN_FINAL_GOODPUT_RATIO_TO_BASELINE = 1.0
_POLICY_PRIORITY = {
    policy: index
    for index, policy in enumerate(CANONICAL_MIGRATION_POLICIES)
}


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


def _canonical_policy(policy: str) -> str:
    return DELAY_POLICY_ALIASES.get(policy, policy)


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
    return mean, lower, upper


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
    baseline_candidate_key: str
    goodput_mean: float
    goodput_ci95_lower: Optional[float]
    goodput_ci95_upper: Optional[float]
    baseline_goodput_mean: float
    oracle_goodput_mean: float
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
class AliasCollapse:
    excluded_key: str
    retained_key: str
    coordinate: tuple[str, str, str, str]


@dataclass(frozen=True)
class LoadedStagedResults:
    source_path: Path
    source_aggregate_sha256: str
    source_payload_sha256: str
    aggregate: Mapping[str, Any]
    session_rate: float
    references: Mapping[str, Mapping[str, Any]]
    candidates: tuple[FinalCandidate, ...]
    alias_collapses: tuple[AliasCollapse, ...]
    runtime_available: bool
    reference_eligible: bool
    reference_eligibility_failures: tuple[str, ...]
    audit_mode: bool


def _validate_reference(
        key: str,
        reference: Mapping[str, Any],
) -> float:
    goodput, _, _ = _stat_mean(
        reference,
        "slo_good_output_tokens_per_second",
        f"references.{key}",
        positive=True,
    )
    _stat_mean(
        reference,
        "joint_slo_pass_fraction",
        f"references.{key}",
    )
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
        or _canonical_policy(policy)
        not in CANONICAL_MIGRATION_POLICIES
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
    )


def _candidate_from_row(
        row: object,
        index: int,
        references: Mapping[str, Mapping[str, Any]],
        reference_goodput: Mapping[str, float],
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
    goodput, lower, upper = _stat_mean(
        metrics,
        "slo_good_output_tokens_per_second",
        f"{path}.metrics",
    )
    if goodput < 0.0:
        raise SSDHBFFinalResultsError(
            f"{path} has negative SLO-good output-token goodput")
    _stat_mean(
        metrics,
        "joint_slo_pass_fraction",
        f"{path}.metrics",
    )
    runtime = _runtime_objectives(design_row, path)
    wear = _wear_objectives(design_row, path)
    return FinalCandidate(
        key=key,
        group_id=_group_id(layout, memory_identity),
        hbf_layout=layout,
        memory_identity=memory_identity,
        migration_policy=policy,
        canonical_migration_policy=_canonical_policy(policy),
        hbf_read_mode=read_mode,
        restore_execution_mode=restore_mode,
        baseline_candidate_key=str(baseline_key),
        goodput_mean=goodput,
        goodput_ci95_lower=lower,
        goodput_ci95_upper=upper,
        baseline_goodput_mean=reference_goodput[str(baseline_key)],
        oracle_goodput_mean=reference_goodput[ORACLE_CANDIDATE_KEY],
        runtime=runtime,
        wear=wear,
        source_index=index,
    )


def _alias_equivalent(
        first: FinalCandidate,
        second: FinalCandidate,
) -> bool:
    first_values = (
        first.goodput_mean,
        first.goodput_ci95_lower,
        first.goodput_ci95_upper,
        first.baseline_goodput_mean,
        first.oracle_goodput_mean,
        first.runtime,
        first.wear,
    )
    second_values = (
        second.goodput_mean,
        second.goodput_ci95_lower,
        second.goodput_ci95_upper,
        second.baseline_goodput_mean,
        second.oracle_goodput_mean,
        second.runtime,
        second.wear,
    )
    return first_values == second_values


def _collapse_aliases(
        candidates: Sequence[FinalCandidate],
) -> tuple[tuple[FinalCandidate, ...], tuple[AliasCollapse, ...]]:
    by_coordinate: dict[
        tuple[str, str, str, str], list[FinalCandidate]
    ] = {}
    for candidate in candidates:
        by_coordinate.setdefault(candidate.coordinate, []).append(candidate)
    retained = []
    collapses = []
    for coordinate, coordinate_candidates in sorted(
            by_coordinate.items()):
        if len(coordinate_candidates) == 1:
            retained.append(coordinate_candidates[0])
            continue
        original_policies = {
            candidate.migration_policy
            for candidate in coordinate_candidates
        }
        if (
            len(coordinate_candidates) != 2
            or original_policies
            != {"delay_1s", "delay_1000ms"}
        ):
            raise SSDHBFFinalResultsError(
                "duplicate canonical design coordinate "
                f"{coordinate!r}")
        canonical = next(
            candidate for candidate in coordinate_candidates
            if candidate.migration_policy == "delay_1000ms"
        )
        alias = next(
            candidate for candidate in coordinate_candidates
            if candidate.migration_policy == "delay_1s"
        )
        if not _alias_equivalent(canonical, alias):
            raise SSDHBFFinalResultsError(
                "delay_1s alias disagrees with delay_1000ms at "
                f"{coordinate!r}")
        retained.append(canonical)
        collapses.append(AliasCollapse(
            excluded_key=alias.key,
            retained_key=canonical.key,
            coordinate=coordinate,
        ))
    return (
        tuple(sorted(retained, key=lambda candidate: candidate.key)),
        tuple(sorted(
            collapses,
            key=lambda collapse: collapse.excluded_key,
        )),
    )


def _validate_complete_roster(
        candidates: Sequence[FinalCandidate],
) -> None:
    by_group: dict[str, list[FinalCandidate]] = {}
    for candidate in candidates:
        by_group.setdefault(candidate.group_id, []).append(candidate)
    if not by_group:
        raise SSDHBFFinalResultsError(
            "staged aggregate contains no design groups")
    expected = {
        (policy, read_mode, restore_mode)
        for policy in CANONICAL_MIGRATION_POLICIES
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
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise SSDHBFFinalResultsError(
                f"incomplete canonical policy/option roster for "
                f"{group_id}: missing={missing}, extra={extra}")


def load_staged_aggregate(
        path: Path | str,
        *,
        allow_ineligible_reference: bool = False,
) -> LoadedStagedResults:
    """Load one SSD-staged aggregate.

    The default remains fail-closed.  An explicitly audit-only aggregate
    may be loaded with ``allow_ineligible_reference=True``; this never
    changes the stored gate outcome and downstream artifacts remain
    visibly marked as ineligible.
    """

    source_path = Path(path).expanduser().resolve()
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
        REQUIRED_SESSION_RATE,
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
            key, _mapping(value, f"references.{key}"))
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
    candidates, alias_collapses = _collapse_aliases(raw_candidates)
    _validate_complete_roster(candidates)

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
        session_rate=session_rate,
        references={
            key: _mapping(value, f"references.{key}")
            for key, value in references.items()
        },
        candidates=candidates,
        alias_collapses=alias_collapses,
        runtime_available=runtime_available,
        reference_eligible=reference_eligible,
        reference_eligibility_failures=reference_failures,
        audit_mode=not reference_eligible,
    )


def _dominates(
        first: FinalCandidate,
        second: FinalCandidate,
        *,
        runtime_available: bool,
) -> bool:
    no_worse = (
        first.goodput_mean >= second.goodput_mean
        and first.wear.five_year_budget_fraction
        <= second.wear.five_year_budget_fraction
    )
    strictly_better = (
        first.goodput_mean > second.goodput_mean
        or first.wear.five_year_budget_fraction
        < second.wear.five_year_budget_fraction
    )
    if runtime_available:
        if first.runtime is None or second.runtime is None:
            raise AssertionError("runtime cohort is partially populated")
        no_worse = (
            no_worse
            and first.runtime.proposed_five_year_tco_usd
            <= second.runtime.proposed_five_year_tco_usd
            and first.runtime.proposed_five_year_facility_energy_kwh
            <= second.runtime.proposed_five_year_facility_energy_kwh
        )
        strictly_better = (
            strictly_better
            or first.runtime.proposed_five_year_tco_usd
            < second.runtime.proposed_five_year_tco_usd
            or first.runtime.proposed_five_year_facility_energy_kwh
            < second.runtime.proposed_five_year_facility_energy_kwh
        )
    return no_worse and strictly_better


def _candidate_objectives(
        candidate: FinalCandidate,
) -> dict[str, Optional[float]]:
    runtime = candidate.runtime
    return {
        "goodput_output_tokens_per_second": candidate.goodput_mean,
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


def _policy_objectives(
        candidates: Sequence[FinalCandidate],
) -> dict[str, Any]:
    ordered = tuple(sorted(
        candidates,
        key=lambda value: (
            value.hbf_read_mode,
            value.restore_execution_mode,
        ),
    ))
    if (
        len(ordered) != len(_REQUIRED_OPTIONS)
        or {candidate.option for candidate in ordered}
        != set(_REQUIRED_OPTIONS)
    ):
        raise SSDHBFFinalResultsError(
            "policy summary requires all four read/restore options")
    ratios = tuple(
        candidate.goodput_mean / candidate.baseline_goodput_mean
        for candidate in ordered
    )
    runtime_ratios = tuple(
        None if candidate.runtime is None
        else candidate.runtime.tco_ratio
        for candidate in ordered
    )
    if any(value is None for value in runtime_ratios):
        mean_runtime_tco_ratio = None
    else:
        mean_runtime_tco_ratio = math.fsum(
            float(value) for value in runtime_ratios
        ) / len(runtime_ratios)
    signature = tuple(
        (
            candidate.option,
            candidate.goodput_mean,
            candidate.runtime,
            candidate.wear,
        )
        for candidate in ordered
    )
    return {
        "policy": ordered[0].canonical_migration_policy,
        "candidate_keys": [
            candidate.key for candidate in ordered],
        "minimum_goodput_ratio_to_baseline": min(ratios),
        "mean_goodput_ratio_to_baseline": (
            math.fsum(ratios) / len(ratios)),
        "mean_runtime_tco_ratio_to_baseline": (
            mean_runtime_tco_ratio),
        "maximum_five_year_wear_budget_fraction": max(
            candidate.wear.five_year_budget_fraction
            for candidate in ordered
        ),
        "robust_across_all_options": (
            min(ratios)
            >= MIN_FINAL_GOODPUT_RATIO_TO_BASELINE
        ),
        "_signature": signature,
    }


def _normalized_distance_to_ideal(
        summary: Mapping[str, Any],
        robust_summaries: Sequence[Mapping[str, Any]],
        *,
        runtime_available: bool,
) -> float:
    axes = (
        ("mean_goodput_ratio_to_baseline", True),
        ("maximum_five_year_wear_budget_fraction", False),
    )
    if runtime_available:
        axes = (
            axes[0],
            ("mean_runtime_tco_ratio_to_baseline", False),
            axes[1],
        )
    squared = []
    for field, maximize in axes:
        values = [float(row[field]) for row in robust_summaries]
        lower = min(values)
        upper = max(values)
        observed = float(summary[field])
        if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-15):
            cost = 0.0
        elif maximize:
            cost = (upper - observed) / (upper - lower)
        else:
            cost = (observed - lower) / (upper - lower)
        squared.append(cost * cost)
    return math.sqrt(math.fsum(squared) / len(squared))


def select_meaningful_policies(
        loaded: LoadedStagedResults,
) -> dict[str, Any]:
    """Select auditable policy-level performance, knee, and wear anchors."""

    if not isinstance(loaded, LoadedStagedResults):
        raise SSDHBFFinalResultsError(
            "loaded must be a validated LoadedStagedResults")
    candidates_by_group: dict[str, list[FinalCandidate]] = {}
    for candidate in loaded.candidates:
        candidates_by_group.setdefault(
            candidate.group_id, []).append(candidate)

    audit_by_key: dict[str, dict[str, Any]] = {}
    group_reports = []
    selected_keys = set()
    for group_id, group_candidates in sorted(
            candidates_by_group.items()):
        option_winners: dict[str, list[str]] = {}
        for read_mode, restore_mode in sorted(_REQUIRED_OPTIONS):
            option_candidates = [
                candidate for candidate in group_candidates
                if (
                    candidate.option == (read_mode, restore_mode)
                    and candidate.goodput_mean > 0.0
                )
            ]
            if not option_candidates:
                raise SSDHBFFinalResultsError(
                    "no positive-goodput policy remains for option "
                    f"{read_mode}|{restore_mode} in {group_id}")
            maximum = max(
                candidate.goodput_mean
                for candidate in option_candidates
            )
            winners = sorted(
                candidate.key
                for candidate in option_candidates
                if candidate.goodput_mean == maximum
            )
            option_winners[
                f"{read_mode}|{restore_mode}"
            ] = winners

        dominators: dict[str, list[str]] = {}
        frontier = []
        for candidate in group_candidates:
            if candidate.goodput_mean <= 0.0:
                dominators[candidate.key] = []
                continue
            dominating_keys = sorted(
                other.key
                for other in group_candidates
                if (
                    other.key != candidate.key
                    and other.goodput_mean > 0.0
                    and _dominates(
                        other,
                        candidate,
                        runtime_available=loaded.runtime_available,
                    )
                )
            )
            dominators[candidate.key] = dominating_keys
            if not dominating_keys:
                frontier.append(candidate.key)
        frontier = sorted(frontier)

        candidates_by_policy: dict[
            str, list[FinalCandidate]
        ] = {}
        for candidate in group_candidates:
            candidates_by_policy.setdefault(
                candidate.canonical_migration_policy,
                [],
            ).append(candidate)
        policy_summaries = {
            policy: _policy_objectives(rows)
            for policy, rows in candidates_by_policy.items()
        }
        robust_summaries = [
            summary for summary in policy_summaries.values()
            if summary["robust_across_all_options"]
        ]
        if not robust_summaries:
            raise SSDHBFFinalResultsError(
                "no migration policy meets the matched baseline in all "
                f"four options for {group_id}")

        def priority(summary: Mapping[str, Any]) -> int:
            return _POLICY_PRIORITY[str(summary["policy"])]

        maximum_goodput = max(
            float(summary["mean_goodput_ratio_to_baseline"])
            for summary in robust_summaries
        )
        performance = min(
            (
                summary for summary in robust_summaries
                if math.isclose(
                    float(summary[
                        "mean_goodput_ratio_to_baseline"]),
                    maximum_goodput,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ),
            key=priority,
        )
        endurance = min(
            robust_summaries,
            key=lambda summary: (
                float(summary[
                    "maximum_five_year_wear_budget_fraction"]),
                (
                    float(summary[
                        "mean_runtime_tco_ratio_to_baseline"])
                    if loaded.runtime_available
                    else 0.0
                ),
                -float(summary[
                    "mean_goodput_ratio_to_baseline"]),
                priority(summary),
            ),
        )
        knee_distances = {
            str(summary["policy"]): _normalized_distance_to_ideal(
                summary,
                robust_summaries,
                runtime_available=loaded.runtime_available,
            )
            for summary in robust_summaries
        }
        knee = min(
            robust_summaries,
            key=lambda summary: (
                knee_distances[str(summary["policy"])],
                priority(summary),
            ),
        )
        anchor_policies = {
            "performance_across_all_options": str(
                performance["policy"]),
            "normalized_multiobjective_knee": str(knee["policy"]),
            "minimum_wear_with_goodput_ge_baseline": str(
                endurance["policy"]),
        }
        selected_policies = set(anchor_policies.values())
        retained = {
            candidate.key
            for candidate in group_candidates
            if candidate.canonical_migration_policy
            in selected_policies
        }
        selected_keys.update(retained)
        selected_signature_policy = {
            policy_summaries[policy]["_signature"]: policy
            for policy in selected_policies
        }

        for candidate in sorted(
                group_candidates, key=lambda value: value.key):
            selection_reasons = []
            exclusion_reasons = []
            option_key = (
                f"{candidate.hbf_read_mode}|"
                f"{candidate.restore_execution_mode}"
            )
            policy = candidate.canonical_migration_policy
            summary = policy_summaries[policy]
            if candidate.key in retained:
                selection_reasons.extend(
                    f"policy_anchor:{anchor_name}"
                    for anchor_name, anchor_policy
                    in sorted(anchor_policies.items())
                    if anchor_policy == policy
                )
            elif not summary["robust_across_all_options"]:
                exclusion_reasons.append(
                    "policy_minimum_goodput_below_matched_baseline:"
                    f"{summary['minimum_goodput_ratio_to_baseline']:.12g}"
                )
            else:
                equivalent = selected_signature_policy.get(
                    summary["_signature"])
                if equivalent is not None:
                    exclusion_reasons.append(
                        "metric_equivalent_policy_not_retained:"
                        f"{equivalent}")
                else:
                    exclusion_reasons.append(
                        "not_a_policy_level_anchor")

            if candidate.key in option_winners[option_key]:
                if candidate.key in retained:
                    selection_reasons.append(
                        f"best_goodput_for_option:{option_key}")
                else:
                    exclusion_reasons.append(
                        f"option_goodput_tie_not_retained:{option_key}")
            if candidate.goodput_mean <= 0.0:
                exclusion_reasons.append(
                    "zero_slo_goodput_ineligible_for_final_selection")
            elif candidate.key in frontier:
                if candidate.key in retained:
                    selection_reasons.append(
                        "nondominated_on_declared_objectives")
                else:
                    exclusion_reasons.append(
                        "individual_nondominated_but_not_policy_anchor")
            else:
                exclusion_reasons.append(
                    "dominated_by:" + ",".join(
                        dominators[candidate.key]))
            audit_by_key[candidate.key] = {
                "candidate_key": candidate.key,
                "group_id": group_id,
                "migration_policy": candidate.migration_policy,
                "canonical_migration_policy": (
                    candidate.canonical_migration_policy),
                "hbf_read_mode": candidate.hbf_read_mode,
                "restore_execution_mode": (
                    candidate.restore_execution_mode),
                "objectives": _candidate_objectives(candidate),
                "selected": candidate.key in retained,
                "selection_reasons": selection_reasons,
                "exclusion_reasons": exclusion_reasons,
                "dominators": dominators[candidate.key],
            }
        sample = group_candidates[0]
        group_reports.append({
            "group_id": group_id,
            "hbf_layout": sample.hbf_layout,
            "active_memory_identity": list(sample.memory_identity),
            "candidate_count_after_alias_collapse": len(
                group_candidates),
            "option_best_goodput_candidate_keys": option_winners,
            "nondominated_candidate_keys": frontier,
            "minimum_final_goodput_ratio_to_baseline": (
                MIN_FINAL_GOODPUT_RATIO_TO_BASELINE),
            "robust_policy_keys": sorted(
                str(summary["policy"])
                for summary in robust_summaries
            ),
            "selected_policy_anchors": anchor_policies,
            "policy_knee_distances": dict(sorted(knee_distances.items())),
            "policy_summaries": [
                {
                    key: value
                    for key, value in summary.items()
                    if key != "_signature"
                }
                for summary in sorted(
                    policy_summaries.values(),
                    key=priority,
                )
            ],
            "selected_candidate_keys": sorted(retained),
        })

    for collapse in loaded.alias_collapses:
        audit_by_key[collapse.excluded_key] = {
            "candidate_key": collapse.excluded_key,
            "group_id": collapse.coordinate[0],
            "migration_policy": "delay_1s",
            "canonical_migration_policy": "delay_1000ms",
            "hbf_read_mode": collapse.coordinate[2],
            "restore_execution_mode": collapse.coordinate[3],
            "objectives": dict(
                audit_by_key[collapse.retained_key]["objectives"]),
            "selected": False,
            "selection_reasons": [],
            "exclusion_reasons": [
                "canonical_alias_duplicate_of:"
                f"{collapse.retained_key}"
            ],
            "dominators": [],
        }

    report: dict[str, Any] = {
        "final_results_schema_version": FINAL_RESULTS_SCHEMA_VERSION,
        "schema_version": POLICY_SELECTION_SCHEMA_VERSION,
        "source": {
            "aggregate_path": str(loaded.source_path),
            "aggregate_file_sha256": (
                loaded.source_aggregate_sha256),
            "aggregate_payload_sha256": (
                loaded.source_payload_sha256),
            "comparison_contract": SSD_HBF_CONTRACT_KEY,
            "session_rate": loaded.session_rate,
            "result_status": (
                "audit_reference_ineligible"
                if loaded.audit_mode
                else "eligible_final"
            ),
            "reference_eligible": loaded.reference_eligible,
            "reference_eligibility_failures": list(
                loaded.reference_eligibility_failures),
        },
        "selection_algorithm": {
            "scope": (
                "migration policy within each identical HBF layout and "
                "active-memory hardware/economic group; every selected "
                "policy is rendered in all four read/restore options"),
            "required_migration_policies": list(
                CANONICAL_MIGRATION_POLICIES),
            "required_read_restore_options": [
                {
                    "hbf_read_mode": read_mode,
                    "restore_execution_mode": restore_mode,
                }
                for read_mode, restore_mode in sorted(_REQUIRED_OPTIONS)
            ],
            "objectives": list(
                _SELECTION_OBJECTIVES_WITH_RUNTIME
                if loaded.runtime_available
                else _SELECTION_OBJECTIVES_WITHOUT_RUNTIME
            ),
            "retention_rule": (
                "retain the performance, normalized multiobjective-knee, "
                "and minimum-wear policy anchors after requiring every "
                "read/restore option to meet the matched baseline; exact "
                "metric-equivalent policies use canonical policy order "
                "as a deterministic representative"),
            "minimum_goodput_ratio_to_baseline_in_every_option": (
                MIN_FINAL_GOODPUT_RATIO_TO_BASELINE),
            "oracle_semantics": (
                "performance-only upper reference; excluded from "
                "power, energy, TCO, endurance, and Pareto selection"),
            "delay_alias_semantics": (
                "delay_1s canonicalizes to delay_1000ms; a duplicate "
                "alias is excluded without metric-based cherry-picking"),
        },
        "runtime_objectives_available": loaded.runtime_available,
        "audit_mode": loaded.audit_mode,
        "groups": group_reports,
        "selected_candidate_keys": sorted(selected_keys),
        "candidate_audit": [
            audit_by_key[key] for key in sorted(audit_by_key)
        ],
    }
    report["policy_selection_sha256"] = stable_json_sha256(report)
    return report


_PLOT_SOURCE_FIELDS = (
    "plot_source_schema_version",
    "source_aggregate_sha256",
    "result_status",
    "reference_eligible",
    "reference_eligibility_failures",
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
    "goodput_mean",
    "goodput_ci95_lower",
    "goodput_ci95_upper",
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
        goodput, lower, upper = _stat_mean(
            reference,
            "slo_good_output_tokens_per_second",
            f"references.{key}",
            positive=True,
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
                if is_oracle else "matched_physical_baseline"
            ),
            "exclusion_reasons": "",
            "goodput_mean": goodput,
            "goodput_ci95_lower": lower,
            "goodput_ci95_upper": upper,
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


def build_plot_source_rows(
        loaded: LoadedStagedResults,
        selection: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return deterministic full-audit rows; renderers filter selected."""

    audit_values = _sequence(
        selection.get("candidate_audit"),
        "selection.candidate_audit",
    )
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
            collapse = next(
                (
                    value for value in loaded.alias_collapses
                    if value.excluded_key == key
                ),
                None,
            )
            if collapse is None:
                raise SSDHBFFinalResultsError(
                    f"audit contains unknown candidate {key!r}")
            candidate = candidate_by_key[collapse.retained_key]
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
            "goodput_mean": candidate.goodput_mean,
            "goodput_ci95_lower": (
                ""
                if candidate.goodput_ci95_lower is None
                else candidate.goodput_ci95_lower
            ),
            "goodput_ci95_upper": (
                ""
                if candidate.goodput_ci95_upper is None
                else candidate.goodput_ci95_upper
            ),
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
    return [
        row for row in rows
        if (
            row["candidate_kind"] == "design"
            and row["include_in_final_plots"] is True
        )
    ]


def _short_label(row: Mapping[str, Any]) -> str:
    restore = (
        "stream"
        if row["restore_execution_mode"] == "layerwise_streaming"
        else "bulk"
    )
    read = "prefetch" if row["hbf_read_mode"] == "prefetch" else "demand"
    return (
        f"{row['canonical_migration_policy']} | {read}/{restore} | "
        f"{row['hbf_layout']}"
    )


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


def _render_performance(
        pyplot,
        rows: Sequence[Mapping[str, Any]],
        output_path: Path,
) -> None:
    selected = _selected_design_rows(rows)
    labels = [_short_label(row) for row in selected]
    values = [
        float(row["goodput_ratio_to_baseline"])
        for row in selected
    ]
    oracle = [
        float(row["oracle_goodput_ratio_to_baseline"])
        for row in selected
    ]
    height = max(4.0, 0.31 * len(selected) + 1.5)
    figure, axis = pyplot.subplots(figsize=(10.5, height))
    positions = list(range(len(selected)))
    colors = [
        "#4C78A8" if row["hbf_read_mode"] == "demand"
        else "#F58518"
        for row in selected
    ]
    axis.barh(positions, values, color=colors, alpha=0.88)
    axis.scatter(
        oracle,
        positions,
        color="#333333",
        marker="|",
        s=90,
        label="Oracle / matched baseline",
        zorder=3,
    )
    axis.axvline(
        1.0, color="#666666", linestyle="--", linewidth=1,
        label="Matched two-GPU baseline")
    axis.set_yticks(positions, labels=labels, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlabel("SLO-good output-token goodput / matched baseline")
    audit_prefix = (
        "AUDIT — reference eligibility failed\n"
        if rows[0].get("result_status")
        == "audit_reference_ineligible"
        else ""
    )
    axis.set_title(
        audit_prefix
        + "Selected staged HBF policies: read and restore sensitivity")
    axis.grid(axis="x", alpha=0.22)
    axis.legend(fontsize=8, loc="best")
    _save_figure(figure, output_path)
    pyplot.close(figure)


def _render_runtime(
        pyplot,
        rows: Sequence[Mapping[str, Any]],
        output_path: Path,
) -> None:
    selected = _selected_design_rows(rows)
    labels = [_short_label(row) for row in selected]
    metrics = (
        ("runtime_power_ratio_to_baseline", "Average IT power"),
        ("runtime_energy_ratio_to_baseline", "5-year facility energy"),
        ("runtime_tco_ratio_to_baseline", "5-year TCO"),
    )
    width = max(9.5, 0.44 * len(selected) + 3.0)
    figure, axes = pyplot.subplots(
        3, 1, figsize=(width, 8.0), sharex=True)
    positions = list(range(len(selected)))
    for axis, (field, title) in zip(axes, metrics):
        axis.bar(
            positions,
            [float(row[field]) for row in selected],
            color="#54A24B",
            alpha=0.88,
        )
        axis.axhline(1.0, color="#666666", linestyle="--", linewidth=1)
        axis.set_ylabel("Design / baseline")
        axis.set_title(title, loc="left", fontsize=10)
        axis.grid(axis="y", alpha=0.22)
    axes[-1].set_xticks(
        positions, labels=labels, rotation=55, ha="right", fontsize=7)
    audit_prefix = (
        "AUDIT — reference eligibility failed\n"
        if rows[0].get("result_status")
        == "audit_reference_ineligible"
        else ""
    )
    figure.suptitle(
        audit_prefix
        + "Event-derived runtime power, energy, and TCO "
        "(Oracle excluded)")
    figure.tight_layout()
    _save_figure(figure, output_path)
    pyplot.close(figure)


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
    audit_prefix = (
        "AUDIT — reference eligibility failed\n"
        if rows[0].get("result_status")
        == "audit_reference_ineligible"
        else ""
    )
    figure.suptitle(
        audit_prefix
        + "HBF endurance from measured recurring writes "
        "(uniform within-card spreading)")
    figure.tight_layout()
    _save_figure(figure, output_path)
    pyplot.close(figure)


@dataclass(frozen=True)
class FinalPlotArtifacts:
    policy_selection_json: Path
    plot_source_csv: Path
    performance_sensitivity_png: Optional[Path]
    runtime_power_energy_tco_png: Optional[Path]
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
        return result


def write_final_artifacts(
        loaded: LoadedStagedResults,
        output_dir: Path | str,
        *,
        render: bool = True,
) -> FinalPlotArtifacts:
    """Write audited sources and, when requested, the three final graphs."""

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
    performance_path = runtime_path = endurance_path = None
    if render and pyplot is not None:
        prefix = "audit_" if loaded.audit_mode else ""
        performance_path = (
            root / f"{prefix}performance_sensitivity.png")
        runtime_path = (
            root / f"{prefix}runtime_power_energy_tco.png")
        endurance_path = root / f"{prefix}hbf_endurance.png"
        _render_performance(pyplot, rows, performance_path)
        _render_runtime(pyplot, rows, runtime_path)
        _render_endurance(pyplot, rows, endurance_path)
    return FinalPlotArtifacts(
        policy_selection_json=selection_path,
        plot_source_csv=source_path,
        performance_sensitivity_png=performance_path,
        runtime_power_energy_tco_png=runtime_path,
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
        render: bool = True,
        allow_ineligible_reference_audit: bool = False,
) -> FinalPlotArtifacts:
    """Convenience API: strict load, select, export, and render."""

    return write_final_artifacts(
        load_staged_aggregate(
            aggregate_path,
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
    "DELAY_POLICY_ALIASES",
    "FINAL_RESULTS_SCHEMA_VERSION",
    "FinalCandidate",
    "FinalPlotArtifacts",
    "LoadedStagedResults",
    "POLICY_SELECTION_SCHEMA_VERSION",
    "PLOT_SOURCE_SCHEMA_VERSION",
    "RUNTIME_PROJECTION_METRIC_KEYS",
    "RUNTIME_REPORT_FIELD_KEYS",
    "SSDHBFFinalResultsError",
    "build_plot_source_rows",
    "generate_final_results",
    "load_staged_aggregate",
    "main",
    "select_meaningful_policies",
    "write_final_artifacts",
]

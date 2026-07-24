"""Auditable operating-point selection over paired comparison rate grids.

The selector consumes one already-aggregated metric row per
``(system, offered rate, seed)``.  It does not infer a grid from the rows:
the caller supplies an explicit manifest identity containing the complete
system, rate, seed, scenario, metric, and provenance contract.

Only an equilibrium balanced scenario is eligible for a sustainable-rate
selection.  Its selected point is the highest *tested* offered rate whose
lower Student-t 95% confidence bound for seed-level joint-SLO attainment is
at least 0.95.  A qualifying top grid point is right-censored; it is not an
observed saturation boundary.  Request- and output-token-goodput maxima are
always descriptive maxima over the finite tested grid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Optional, Sequence

from .hbf_comparison_metrics import SeedAggregate, aggregate_seed_values
from .hbf_comparison_workload import stable_json_sha256


RATE_SELECTION_SCHEMA_VERSION = 1
JOINT_SLO_CI_LOWER_THRESHOLD = 0.95
SCENARIO_FAMILY_BALANCED = "balanced"
SCENARIO_FAMILY_LONG_COLD = "long_cold"
SUPPORTED_SCENARIO_FAMILIES = frozenset({
    SCENARIO_FAMILY_BALANCED,
    SCENARIO_FAMILY_LONG_COLD,
})


class HBFSLORateSelectionError(ValueError):
    """Raised when a rate-grid selection contract is incomplete or unsafe."""


def _nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HBFSLORateSelectionError(
            f"{name} must be a non-empty string")
    return value


def _sha256(name: str, value: object) -> str:
    text = _nonempty_string(name, value)
    if (
        len(text) != 64
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise HBFSLORateSelectionError(
            f"{name} must be a lowercase 64-character SHA-256 digest")
    return text


def _finite_number(
        name: str,
        value: object,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        strictly_positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise HBFSLORateSelectionError(
            f"{name} must be a finite number, got {value!r}")
    converted = float(value)
    if strictly_positive and converted <= 0.0:
        raise HBFSLORateSelectionError(
            f"{name} must be positive, got {value!r}")
    if minimum is not None and converted < minimum:
        raise HBFSLORateSelectionError(
            f"{name} must be at least {minimum}, got {value!r}")
    if maximum is not None and converted > maximum:
        raise HBFSLORateSelectionError(
            f"{name} must be at most {maximum}, got {value!r}")
    return converted


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HBFSLORateSelectionError(
            f"{name} must be a positive integer, got {value!r}")
    return value


def _seed_id(name: str, value: object) -> int | str:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, str))
        or isinstance(value, str) and not value
    ):
        raise HBFSLORateSelectionError(
            f"{name} must be an integer or non-empty string, got {value!r}")
    return value


def _seed_sort_key(seed: int | str) -> tuple[str, str]:
    return type(seed).__name__, repr(seed)


@dataclass(frozen=True)
class SystemProvenanceIdentity:
    """Pinned policy/configuration/code identity for one compared system."""

    system_key: str
    provenance_sha256: str

    def __post_init__(self) -> None:
        _nonempty_string("system_key", self.system_key)
        _sha256("provenance_sha256", self.provenance_sha256)


@dataclass(frozen=True)
class RateGridManifestIdentity:
    """Expected Cartesian grid and common experiment provenance."""

    schema_version: int
    scenario_family: str
    scenario_id: str
    scenario_manifest_schema_version: int
    scenario_manifest_sha256: str
    equilibrium_workload: bool
    measurement_roster_sha256: str
    metric_scope: str
    slo_contract_sha256: str
    metric_contract_sha256: str
    result_schema_revision: str
    system_keys: tuple[str, ...]
    rates: tuple[float, ...]
    seed_ids: tuple[int | str, ...]
    system_provenance: tuple[SystemProvenanceIdentity, ...]

    def __post_init__(self) -> None:
        if self.schema_version != RATE_SELECTION_SCHEMA_VERSION:
            raise HBFSLORateSelectionError(
                f"expected schema_version={RATE_SELECTION_SCHEMA_VERSION}")
        if self.scenario_family not in SUPPORTED_SCENARIO_FAMILIES:
            raise HBFSLORateSelectionError(
                "scenario_family must be one of "
                f"{sorted(SUPPORTED_SCENARIO_FAMILIES)}")
        _nonempty_string("scenario_id", self.scenario_id)
        _positive_integer(
            "scenario_manifest_schema_version",
            self.scenario_manifest_schema_version,
        )
        _sha256(
            "scenario_manifest_sha256",
            self.scenario_manifest_sha256,
        )
        if not isinstance(self.equilibrium_workload, bool):
            raise HBFSLORateSelectionError(
                "equilibrium_workload must be a boolean")
        if (
            self.scenario_family == SCENARIO_FAMILY_LONG_COLD
            and self.equilibrium_workload
        ):
            raise HBFSLORateSelectionError(
                "long_cold scenarios must be marked non-equilibrium")
        _sha256(
            "measurement_roster_sha256",
            self.measurement_roster_sha256,
        )
        if self.metric_scope != "all":
            raise HBFSLORateSelectionError(
                "rate selection requires metric_scope='all'")
        _sha256("slo_contract_sha256", self.slo_contract_sha256)
        _sha256("metric_contract_sha256", self.metric_contract_sha256)
        _nonempty_string(
            "result_schema_revision", self.result_schema_revision)

        for name in (
            "system_keys",
            "rates",
            "seed_ids",
            "system_provenance",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise HBFSLORateSelectionError(
                    f"{name} must be an immutable tuple")
        if not self.system_keys:
            raise HBFSLORateSelectionError(
                "system_keys cannot be empty")
        for index, system_key in enumerate(self.system_keys):
            _nonempty_string(f"system_keys[{index}]", system_key)
        if len(self.system_keys) != len(set(self.system_keys)):
            raise HBFSLORateSelectionError(
                "system_keys cannot contain duplicates")

        if not self.rates:
            raise HBFSLORateSelectionError("rates cannot be empty")
        normalized_rates = tuple(
            _finite_number(
                f"rates[{index}]", rate, strictly_positive=True)
            for index, rate in enumerate(self.rates)
        )
        if any(
                left >= right
                for left, right in zip(
                    normalized_rates, normalized_rates[1:])
        ):
            raise HBFSLORateSelectionError(
                "rates must be strictly increasing")
        object.__setattr__(self, "rates", normalized_rates)

        if not self.seed_ids:
            raise HBFSLORateSelectionError("seed_ids cannot be empty")
        for index, seed in enumerate(self.seed_ids):
            _seed_id(f"seed_ids[{index}]", seed)
        if len(self.seed_ids) != len(set(self.seed_ids)):
            raise HBFSLORateSelectionError(
                "seed_ids cannot contain duplicates")

        if any(
                not isinstance(item, SystemProvenanceIdentity)
                for item in self.system_provenance
        ):
            raise HBFSLORateSelectionError(
                "system_provenance entries must be "
                "SystemProvenanceIdentity values")
        provenance_keys = tuple(
            item.system_key for item in self.system_provenance)
        if len(provenance_keys) != len(set(provenance_keys)):
            raise HBFSLORateSelectionError(
                "system_provenance contains duplicate system keys")
        if provenance_keys != self.system_keys:
            raise HBFSLORateSelectionError(
                "system_provenance keys differ from system_keys "
                "(including order): "
                f"observed={provenance_keys!r}, "
                f"expected={self.system_keys!r}"
            )

    @property
    def system_provenance_by_key(self) -> Mapping[str, str]:
        return {
            item.system_key: item.provenance_sha256
            for item in self.system_provenance
        }


@dataclass(frozen=True)
class SeedRateMetricRow:
    """One seed-level summary for a system and tested offered rate."""

    scenario_id: str
    scenario_manifest_sha256: str
    measurement_roster_sha256: str
    metric_scope: str
    slo_contract_sha256: str
    metric_contract_sha256: str
    result_schema_revision: str
    system_key: str
    system_provenance_sha256: str
    offered_session_rate: float
    seed_id: int | str
    unit_rate_plan_sha256: str
    rate_scaled_schedule_sha256: str
    cell_manifest_sha256: str
    joint_slo_pass_fraction: float
    slo_request_goodput_per_second: float
    slo_output_token_goodput_per_second: float

    def __post_init__(self) -> None:
        _nonempty_string("scenario_id", self.scenario_id)
        for name in (
            "scenario_manifest_sha256",
            "measurement_roster_sha256",
            "slo_contract_sha256",
            "metric_contract_sha256",
            "system_provenance_sha256",
            "unit_rate_plan_sha256",
            "rate_scaled_schedule_sha256",
            "cell_manifest_sha256",
        ):
            _sha256(name, getattr(self, name))
        if self.metric_scope != "all":
            raise HBFSLORateSelectionError(
                "row metric_scope must be 'all'")
        _nonempty_string(
            "result_schema_revision", self.result_schema_revision)
        _nonempty_string("system_key", self.system_key)
        offered_rate = _finite_number(
            "offered_session_rate",
            self.offered_session_rate,
            strictly_positive=True,
        )
        _seed_id("seed_id", self.seed_id)
        joint_slo = _finite_number(
            "joint_slo_pass_fraction",
            self.joint_slo_pass_fraction,
            minimum=0.0,
            maximum=1.0,
        )
        request_goodput = _finite_number(
            "slo_request_goodput_per_second",
            self.slo_request_goodput_per_second,
            minimum=0.0,
        )
        output_goodput = _finite_number(
            "slo_output_token_goodput_per_second",
            self.slo_output_token_goodput_per_second,
            minimum=0.0,
        )
        object.__setattr__(
            self, "offered_session_rate", offered_rate)
        object.__setattr__(
            self, "joint_slo_pass_fraction", joint_slo)
        object.__setattr__(
            self,
            "slo_request_goodput_per_second",
            request_goodput,
        )
        object.__setattr__(
            self,
            "slo_output_token_goodput_per_second",
            output_goodput,
        )


@dataclass(frozen=True)
class RatePointAggregate:
    """Three seed aggregates for one system/rate cell column."""

    offered_session_rate: float
    joint_slo_pass_fraction: SeedAggregate
    slo_request_goodput_per_second: SeedAggregate
    slo_output_token_goodput_per_second: SeedAggregate
    joint_slo_ci_lower_qualifies: Optional[bool]


@dataclass(frozen=True)
class SustainableRateSelection:
    """Conservative joint-SLO operating point or explicit rejection."""

    eligible: bool
    status: str
    selected_rate: Optional[float]
    selected_joint_slo_mean: Optional[float]
    selected_joint_slo_ci95_lower: Optional[float]
    joint_slo_ci_lower_threshold: float
    right_censored: Optional[bool]
    semantics: str


@dataclass(frozen=True)
class DescriptiveGoodputMaximum:
    """Finite-grid mean maximum that is not a sustainable-rate claim."""

    metric: str
    selected_rate: float
    mean_value: float
    tied_rates: tuple[float, ...]
    seed_aggregate: SeedAggregate
    sustainable_ceiling_claim: bool
    semantics: str


@dataclass(frozen=True)
class SystemRateSelection:
    """All operating-point results for one system."""

    system_key: str
    rate_points: tuple[RatePointAggregate, ...]
    sustainable_joint_slo_rate: SustainableRateSelection
    descriptive_request_goodput_maximum: DescriptiveGoodputMaximum
    descriptive_output_token_goodput_maximum: DescriptiveGoodputMaximum


@dataclass(frozen=True)
class RateGridSelectionArtifact:
    """JSON-serializable selection result with input identity hashes."""

    schema_version: int
    manifest_identity: RateGridManifestIdentity
    manifest_identity_sha256: str
    canonical_input_rows_sha256: str
    joint_slo_ci_lower_threshold: float
    systems: tuple[SystemRateSelection, ...]
    interpretation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _aggregate(values_by_seed: Mapping[int | str, float]) -> SeedAggregate:
    try:
        return aggregate_seed_values(values_by_seed)
    except ValueError as exc:
        raise HBFSLORateSelectionError(
            f"seed aggregation failed: {exc}") from exc


def _validate_rows(
        manifest: RateGridManifestIdentity,
        rows: Sequence[SeedRateMetricRow],
) -> dict[tuple[str, float, int | str], SeedRateMetricRow]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise HBFSLORateSelectionError(
            "rows must be a sequence of SeedRateMetricRow values")
    expected_systems = set(manifest.system_keys)
    expected_rates = {float(rate) for rate in manifest.rates}
    expected_seeds = set(manifest.seed_ids)
    expected_provenance = manifest.system_provenance_by_key
    indexed: dict[
        tuple[str, float, int | str], SeedRateMetricRow
    ] = {}
    cell_manifest_owners: dict[str, tuple[str, float, int | str]] = {}

    for row_index, row in enumerate(rows):
        if not isinstance(row, SeedRateMetricRow):
            raise HBFSLORateSelectionError(
                f"rows[{row_index}] must be a SeedRateMetricRow")
        common_fields = {
            "scenario_id": (
                row.scenario_id, manifest.scenario_id),
            "scenario_manifest_sha256": (
                row.scenario_manifest_sha256,
                manifest.scenario_manifest_sha256,
            ),
            "measurement_roster_sha256": (
                row.measurement_roster_sha256,
                manifest.measurement_roster_sha256,
            ),
            "metric_scope": (
                row.metric_scope, manifest.metric_scope),
            "slo_contract_sha256": (
                row.slo_contract_sha256,
                manifest.slo_contract_sha256,
            ),
            "metric_contract_sha256": (
                row.metric_contract_sha256,
                manifest.metric_contract_sha256,
            ),
            "result_schema_revision": (
                row.result_schema_revision,
                manifest.result_schema_revision,
            ),
        }
        mismatched = {
            name: (observed, expected)
            for name, (observed, expected) in common_fields.items()
            if observed != expected
        }
        if mismatched:
            raise HBFSLORateSelectionError(
                f"rows[{row_index}] common provenance mismatch: "
                f"{mismatched!r}")
        if row.system_key not in expected_systems:
            raise HBFSLORateSelectionError(
                f"rows[{row_index}] has unexpected system "
                f"{row.system_key!r}")
        rate = float(row.offered_session_rate)
        if rate not in expected_rates:
            raise HBFSLORateSelectionError(
                f"rows[{row_index}] has unexpected rate {rate!r}")
        if row.seed_id not in expected_seeds:
            raise HBFSLORateSelectionError(
                f"rows[{row_index}] has unexpected seed "
                f"{row.seed_id!r}")
        if (
            row.system_provenance_sha256
            != expected_provenance[row.system_key]
        ):
            raise HBFSLORateSelectionError(
                f"rows[{row_index}] system provenance differs from "
                f"the manifest for {row.system_key!r}")

        key = (row.system_key, rate, row.seed_id)
        if key in indexed:
            raise HBFSLORateSelectionError(
                f"duplicate rate-grid cell {key!r}")
        owner = cell_manifest_owners.get(row.cell_manifest_sha256)
        if owner is not None:
            raise HBFSLORateSelectionError(
                "cell manifest digest is reused by distinct grid cells: "
                f"first={owner!r}, second={key!r}"
            )
        indexed[key] = row
        cell_manifest_owners[row.cell_manifest_sha256] = key

    expected_keys = {
        (system_key, float(rate), seed_id)
        for system_key in manifest.system_keys
        for rate in manifest.rates
        for seed_id in manifest.seed_ids
    }
    observed_keys = set(indexed)
    if observed_keys != expected_keys:
        missing = sorted(
            expected_keys - observed_keys,
            key=lambda key: (
                key[0], key[1], _seed_sort_key(key[2])),
        )
        extra = sorted(
            observed_keys - expected_keys,
            key=lambda key: (
                key[0], key[1], _seed_sort_key(key[2])),
        )
        raise HBFSLORateSelectionError(
            "rate grid is not the exact manifest Cartesian product: "
            f"missing={missing[:8]!r}, extra={extra[:8]!r}"
        )

    # A seed identifies one frozen unit-rate permutation/draw vector.  Rate
    # scaling may change the concrete schedule, but neither the system nor
    # the selected rate may redraw that unit-rate plan.
    for seed_id in manifest.seed_ids:
        plan_hashes = {
            row.unit_rate_plan_sha256
            for key, row in indexed.items()
            if key[2] == seed_id
        }
        if len(plan_hashes) != 1:
            raise HBFSLORateSelectionError(
                f"unit-rate plan provenance differs for seed {seed_id!r}")

    # At a concrete rate and seed, every system must consume the exact same
    # rate-scaled schedule.
    for rate in manifest.rates:
        for seed_id in manifest.seed_ids:
            schedule_hashes = {
                indexed[(
                    system_key, float(rate), seed_id
                )].rate_scaled_schedule_sha256
                for system_key in manifest.system_keys
            }
            if len(schedule_hashes) != 1:
                raise HBFSLORateSelectionError(
                    "rate-scaled schedule provenance differs across "
                    f"systems for rate={float(rate)!r}, seed={seed_id!r}"
                )
    return indexed


def _descriptive_maximum(
        rate_points: Sequence[RatePointAggregate],
        *,
        metric: str,
) -> DescriptiveGoodputMaximum:
    if metric == "slo_request_goodput_per_second":
        aggregate_for = lambda point: (
            point.slo_request_goodput_per_second)
    elif metric == "slo_output_token_goodput_per_second":
        aggregate_for = lambda point: (
            point.slo_output_token_goodput_per_second)
    else:
        raise AssertionError(f"unsupported descriptive metric {metric!r}")
    maximum = max(aggregate_for(point).mean for point in rate_points)
    tied_rates = tuple(
        point.offered_session_rate
        for point in rate_points
        if aggregate_for(point).mean == maximum
    )
    selected_rate = max(tied_rates)
    selected_point = next(
        point
        for point in rate_points
        if point.offered_session_rate == selected_rate
    )
    return DescriptiveGoodputMaximum(
        metric=metric,
        selected_rate=selected_rate,
        mean_value=maximum,
        tied_rates=tied_rates,
        seed_aggregate=aggregate_for(selected_point),
        sustainable_ceiling_claim=False,
        semantics=(
            "arithmetic mean of seed-level SLO goodput; descriptive "
            "maximum over the finite tested rate grid; highest-rate "
            "tie break; not a maximum sustainable throughput claim"
        ),
    )


def _sustainable_selection(
        manifest: RateGridManifestIdentity,
        rate_points: Sequence[RatePointAggregate],
) -> SustainableRateSelection:
    semantics = (
        "highest tested offered rate whose lower two-sided Student-t 95% "
        "confidence bound for seed-level joint-SLO pass fraction is at "
        "least 0.95"
    )
    if (
        manifest.scenario_family != SCENARIO_FAMILY_BALANCED
        or not manifest.equilibrium_workload
    ):
        return SustainableRateSelection(
            eligible=False,
            status="rejected_non_equilibrium_or_non_balanced_scenario",
            selected_rate=None,
            selected_joint_slo_mean=None,
            selected_joint_slo_ci95_lower=None,
            joint_slo_ci_lower_threshold=(
                JOINT_SLO_CI_LOWER_THRESHOLD),
            right_censored=None,
            semantics=(
                semantics
                + "; rejected because sustainable selection is only "
                "defined for an equilibrium balanced scenario"
            ),
        )
    if len(manifest.seed_ids) < 2:
        raise HBFSLORateSelectionError(
            "equilibrium sustainable selection requires at least two "
            "seeds for a Student-t confidence interval"
        )
    qualified = [
        point
        for point in rate_points
        if point.joint_slo_ci_lower_qualifies
    ]
    if not qualified:
        return SustainableRateSelection(
            eligible=True,
            status="no_tested_rate_meets_joint_slo_ci_floor",
            selected_rate=None,
            selected_joint_slo_mean=None,
            selected_joint_slo_ci95_lower=None,
            joint_slo_ci_lower_threshold=(
                JOINT_SLO_CI_LOWER_THRESHOLD),
            right_censored=False,
            semantics=semantics,
        )
    selected = max(
        qualified, key=lambda point: point.offered_session_rate)
    lower = selected.joint_slo_pass_fraction.ci95_lower
    if lower is None:
        raise AssertionError("qualified rate lacks a confidence bound")
    top_rate = max(float(rate) for rate in manifest.rates)
    return SustainableRateSelection(
        eligible=True,
        status="selected",
        selected_rate=selected.offered_session_rate,
        selected_joint_slo_mean=(
            selected.joint_slo_pass_fraction.mean),
        selected_joint_slo_ci95_lower=lower,
        joint_slo_ci_lower_threshold=JOINT_SLO_CI_LOWER_THRESHOLD,
        right_censored=selected.offered_session_rate == top_rate,
        semantics=(
            semantics
            + "; right_censored=true means the top tested rate still "
            "qualified and no saturation boundary was observed"
        ),
    )


def select_rate_grid_operating_points(
        manifest: RateGridManifestIdentity,
        rows: Sequence[SeedRateMetricRow],
) -> RateGridSelectionArtifact:
    """Validate a complete paired grid and select auditable rate points."""

    if not isinstance(manifest, RateGridManifestIdentity):
        raise HBFSLORateSelectionError(
            "manifest must be a RateGridManifestIdentity")
    indexed = _validate_rows(manifest, rows)
    system_results = []
    canonical_rows = []
    for system_key in manifest.system_keys:
        rate_points = []
        for raw_rate in manifest.rates:
            rate = float(raw_rate)
            cell_rows = tuple(
                indexed[(system_key, rate, seed_id)]
                for seed_id in manifest.seed_ids
            )
            joint = _aggregate({
                row.seed_id: float(row.joint_slo_pass_fraction)
                for row in cell_rows
            })
            request_goodput = _aggregate({
                row.seed_id: float(
                    row.slo_request_goodput_per_second)
                for row in cell_rows
            })
            output_goodput = _aggregate({
                row.seed_id: float(
                    row.slo_output_token_goodput_per_second)
                for row in cell_rows
            })
            qualifies: Optional[bool]
            if (
                manifest.scenario_family == SCENARIO_FAMILY_BALANCED
                and manifest.equilibrium_workload
            ):
                if joint.ci95_lower is None:
                    qualifies = False
                else:
                    qualifies = (
                        joint.ci95_lower
                        >= JOINT_SLO_CI_LOWER_THRESHOLD
                    )
            else:
                qualifies = None
            rate_points.append(RatePointAggregate(
                offered_session_rate=rate,
                joint_slo_pass_fraction=joint,
                slo_request_goodput_per_second=request_goodput,
                slo_output_token_goodput_per_second=output_goodput,
                joint_slo_ci_lower_qualifies=qualifies,
            ))
            canonical_rows.extend(asdict(row) for row in cell_rows)
        points = tuple(rate_points)
        system_results.append(SystemRateSelection(
            system_key=system_key,
            rate_points=points,
            sustainable_joint_slo_rate=_sustainable_selection(
                manifest, points),
            descriptive_request_goodput_maximum=(
                _descriptive_maximum(
                    points,
                    metric="slo_request_goodput_per_second",
                )
            ),
            descriptive_output_token_goodput_maximum=(
                _descriptive_maximum(
                    points,
                    metric="slo_output_token_goodput_per_second",
                )
            ),
        ))

    artifact = RateGridSelectionArtifact(
        schema_version=RATE_SELECTION_SCHEMA_VERSION,
        manifest_identity=manifest,
        manifest_identity_sha256=stable_json_sha256(
            asdict(manifest)),
        canonical_input_rows_sha256=stable_json_sha256(
            canonical_rows),
        joint_slo_ci_lower_threshold=JOINT_SLO_CI_LOWER_THRESHOLD,
        systems=tuple(system_results),
        interpretation=(
            "Sustainable selections, when eligible, are conservative "
            "operating points on the tested grid. Descriptive goodput "
            "maxima and right-censored top-grid selections are not "
            "maximum sustainable throughput estimates."
        ),
    )
    # Exercise primitive serialization before returning an artifact used by
    # downstream TCO code.
    stable_json_sha256(artifact.to_dict())
    return artifact


__all__ = [
    "DescriptiveGoodputMaximum",
    "HBFSLORateSelectionError",
    "JOINT_SLO_CI_LOWER_THRESHOLD",
    "RATE_SELECTION_SCHEMA_VERSION",
    "RateGridManifestIdentity",
    "RateGridSelectionArtifact",
    "RatePointAggregate",
    "SCENARIO_FAMILY_BALANCED",
    "SCENARIO_FAMILY_LONG_COLD",
    "SUPPORTED_SCENARIO_FAMILIES",
    "SeedRateMetricRow",
    "SustainableRateSelection",
    "SystemProvenanceIdentity",
    "SystemRateSelection",
    "select_rate_grid_operating_points",
]

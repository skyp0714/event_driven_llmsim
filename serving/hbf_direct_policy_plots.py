"""Strict appendix selection and plotting for direct GPU-HBM to HBF migration.

This module accepts only the narrow direct-migration campaign emitted by
``hbf_design_space_sweep``:

* one TraceLab rate;
* three paired arrival seeds;
* one eight-card TP4 HBF host and one active-memory point;
* all 11 migration policies crossed with demand and prefetch HBF reads;
* the CPU+SSD baseline and infinite-HBM Oracle references.

The direct campaign is deliberately kept separate from the SSD-staged HBF
campaign.  Its absolute metrics must not be merged with staged results.
``jit_oracle`` remains in the audit exports, but is excluded from main-policy
selection because it uses future tool-gap knowledge.  For each read mode, all
causal policies tied for the highest mean joint-SLO output-token goodput are
selected.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .hbf_design_space_sweep import (
    BASELINE_CANDIDATE_KEY,
    DESIGN_SPACE_SCHEMA_VERSION,
    ORACLE_CANDIDATE_KEY,
    SUPPORTED_HBF_READ_MODES,
    SUPPORTED_MIGRATION_POLICIES,
)


DIRECT_POLICY_SELECTION_SCHEMA = "hbf-direct-policy-selection-v1"
DIRECT_PLOT_SOURCE_SCHEMA = "hbf-direct-policy-plot-source-v1"
EXPECTED_SCENARIO_ID = "tracelab-balanced-3-call-causal-prefix-v1"
EXPECTED_LAYOUT = ("tp4",)
EXPECTED_SEED_COUNT = 3
FUTURE_LOOKING_POLICIES = frozenset({"jit_oracle"})
GOODPUT_METRIC = "slo_good_output_tokens_per_second"
GOODPUT_TIE_REL_TOL = 1e-12
GOODPUT_TIE_ABS_TOL = 1e-12

SELECTION_FILENAME = "direct_policy_selection.json"
PLOT_SOURCE_FILENAME = "direct_policy_plot_source.csv"
PLOT_FILENAME = "direct_policy_appendix.png"

_REFERENCE_KEYS = frozenset({
    BASELINE_CANDIDATE_KEY,
    ORACLE_CANDIDATE_KEY,
})
_POLICY_PRIORITY = {
    policy: index
    for index, policy in enumerate(SUPPORTED_MIGRATION_POLICIES)
}
_EXPECTED_ROSTER = frozenset(
    (policy, read_mode)
    for policy in SUPPORTED_MIGRATION_POLICIES
    for read_mode in SUPPORTED_HBF_READ_MODES
)


class HBFDirectPolicyPlotError(ValueError):
    """Raised when a direct-policy aggregate or output request is invalid."""


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HBFDirectPolicyPlotError(f"{path} must be an object")
    return value


def _sequence(value: object, path: str) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise HBFDirectPolicyPlotError(f"{path} must be an array")
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
        raise HBFDirectPolicyPlotError(
            f"{path} must be a finite number")
    result = float(value)
    if positive and result <= 0.0:
        raise HBFDirectPolicyPlotError(f"{path} must be positive")
    if minimum is not None and result < minimum:
        raise HBFDirectPolicyPlotError(
            f"{path} must be at least {minimum}")
    return result


def _integer(
        value: object,
        path: str,
        *,
        minimum: int = 0,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise HBFDirectPolicyPlotError(
            f"{path} must be an integer at least {minimum}")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json(payload: bytes, path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise HBFDirectPolicyPlotError(
            f"{path} contains non-finite JSON constant {value}")

    def reject_duplicates(
            pairs: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise HBFDirectPolicyPlotError(
                    f"{path} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        decoded = payload.decode("utf-8")
        value = json.loads(
            decoded,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except HBFDirectPolicyPlotError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HBFDirectPolicyPlotError(
            f"invalid aggregate JSON {path}: {exc}") from exc
    return _mapping(value, str(path))


def _same_float(first: float, second: float) -> bool:
    return math.isclose(
        first,
        second,
        rel_tol=GOODPUT_TIE_REL_TOL,
        abs_tol=GOODPUT_TIE_ABS_TOL,
    )


@dataclass(frozen=True)
class GoodputStatistic:
    mean: float
    ci95_lower: float
    ci95_upper: float
    seed_ids: tuple[int, ...]
    values: tuple[float, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "ci95_lower": self.ci95_lower,
            "ci95_upper": self.ci95_upper,
            "seed_ids": list(self.seed_ids),
            "values": list(self.values),
        }


def _goodput_statistic(
        raw: object,
        path: str,
        expected_seeds: tuple[int, ...],
) -> GoodputStatistic:
    statistic = _mapping(raw, path)
    mean = _finite(
        statistic.get("mean"), f"{path}.mean", minimum=0.0)
    lower = _finite(
        statistic.get("ci95_lower"), f"{path}.ci95_lower")
    upper = _finite(
        statistic.get("ci95_upper"), f"{path}.ci95_upper")
    if lower > mean or upper < mean or lower > upper:
        raise HBFDirectPolicyPlotError(
            f"{path} has an invalid confidence interval")
    if statistic.get("ci_method") != "student_t_95":
        raise HBFDirectPolicyPlotError(
            f"{path}.ci_method must equal 'student_t_95'")

    raw_seed_ids = _sequence(
        statistic.get("seed_ids"), f"{path}.seed_ids")
    seed_ids = tuple(
        _integer(seed, f"{path}.seed_ids[{index}]", minimum=0)
        for index, seed in enumerate(raw_seed_ids)
    )
    if seed_ids != expected_seeds:
        raise HBFDirectPolicyPlotError(
            f"{path}.seed_ids must match the three grid seeds")
    raw_values = _sequence(
        statistic.get("values"), f"{path}.values")
    values = tuple(
        _finite(
            value,
            f"{path}.values[{index}]",
            minimum=0.0,
        )
        for index, value in enumerate(raw_values)
    )
    if len(values) != len(seed_ids):
        raise HBFDirectPolicyPlotError(
            f"{path}.values must contain one value per seed")
    recomputed_mean = sum(values) / len(values)
    if not _same_float(mean, recomputed_mean):
        raise HBFDirectPolicyPlotError(
            f"{path}.mean does not match its seed values")
    return GoodputStatistic(
        mean=mean,
        ci95_lower=lower,
        ci95_upper=upper,
        seed_ids=seed_ids,
        values=values,
    )


def _memory_identity(
        raw: object,
        path: str,
) -> tuple[object, ...]:
    memory = _mapping(raw, path)
    kind = memory.get("kind")
    if not isinstance(kind, str) or not kind:
        raise HBFDirectPolicyPlotError(f"{path}.kind must be non-empty")
    capacity = _finite(
        memory.get("capacity_gib_per_card"),
        f"{path}.capacity_gib_per_card",
        positive=True,
    )
    bandwidth = _finite(
        memory.get("bandwidth_gbps_per_card"),
        f"{path}.bandwidth_gbps_per_card",
        positive=True,
    )
    capex = _finite(
        memory.get("capex_usd_per_gib"),
        f"{path}.capex_usd_per_gib",
        minimum=0.0,
    )
    power = _finite(
        memory.get("power_w_per_gib"),
        f"{path}.power_w_per_gib",
        minimum=0.0,
    )
    return (kind, capacity, bandwidth, capex, power)


@dataclass(frozen=True)
class DirectPolicyCandidate:
    key: str
    migration_policy: str
    hbf_read_mode: str
    active_memory: Mapping[str, Any]
    goodput: GoodputStatistic
    source_index: int

    @property
    def future_looking(self) -> bool:
        return self.migration_policy in FUTURE_LOOKING_POLICIES


@dataclass(frozen=True)
class LoadedDirectPolicyResults:
    source_path: Path
    source_aggregate_sha256: str
    aggregate: Mapping[str, Any]
    session_rate: float
    seed_ids: tuple[int, ...]
    hbf_server_layouts: tuple[str, ...]
    active_memory: Mapping[str, Any]
    scenario_id: str
    references: Mapping[str, GoodputStatistic]
    candidates: tuple[DirectPolicyCandidate, ...]


def _design_fields(
        raw: object,
        path: str,
) -> tuple[
    Mapping[str, Any],
    str,
    str,
    str,
    tuple[str, ...],
    tuple[object, ...],
]:
    design = _mapping(raw, path)
    key = design.get("key")
    if not isinstance(key, str) or not key:
        raise HBFDirectPolicyPlotError(f"{path}.key must be non-empty")
    policy = design.get("migration_policy")
    if policy not in SUPPORTED_MIGRATION_POLICIES:
        raise HBFDirectPolicyPlotError(
            f"{path}.migration_policy is unsupported")
    read_mode = design.get("hbf_read_mode")
    if read_mode not in SUPPORTED_HBF_READ_MODES:
        raise HBFDirectPolicyPlotError(
            f"{path}.hbf_read_mode is unsupported")
    raw_layouts = _sequence(
        design.get("hbf_server_layouts"),
        f"{path}.hbf_server_layouts",
    )
    layouts = tuple(raw_layouts)
    if (
        any(not isinstance(layout, str) for layout in layouts)
        or layouts != EXPECTED_LAYOUT
    ):
        raise HBFDirectPolicyPlotError(
            f"{path}.hbf_server_layouts must equal "
            f"{list(EXPECTED_LAYOUT)!r}")
    memory_identity = _memory_identity(
        design.get("active_memory"), f"{path}.active_memory")
    return (
        design,
        key,
        str(policy),
        str(read_mode),
        layouts,
        memory_identity,
    )


def load_direct_policy_aggregate(
        path: Path | str,
) -> LoadedDirectPolicyResults:
    """Load and strictly validate one complete direct-policy aggregate."""

    source_path = Path(path).expanduser().resolve()
    try:
        payload = source_path.read_bytes()
    except OSError as exc:
        raise HBFDirectPolicyPlotError(
            f"cannot read aggregate {source_path}: {exc}") from exc
    aggregate = _strict_json(payload, source_path)

    if aggregate.get("schema_version") != DESIGN_SPACE_SCHEMA_VERSION:
        raise HBFDirectPolicyPlotError(
            "aggregate.schema_version does not match "
            "hbf_design_space_sweep")
    if aggregate.get("performance_metric") != (
        "offered-load-normalized joint-SLO-good output tokens/s"
    ):
        raise HBFDirectPolicyPlotError(
            "aggregate.performance_metric is not the direct campaign metric")
    execution_hash = aggregate.get("execution_inputs_sha256")
    if not _is_sha256(execution_hash):
        raise HBFDirectPolicyPlotError(
            "aggregate.execution_inputs_sha256 must be a lowercase SHA-256")

    scenario = _mapping(aggregate.get("scenario"), "aggregate.scenario")
    scenario_id = scenario.get("scenario_id")
    if scenario_id != EXPECTED_SCENARIO_ID:
        raise HBFDirectPolicyPlotError(
            "aggregate.scenario.scenario_id is not the direct appendix "
            "scenario")
    if not _is_sha256(scenario.get("manifest_sha256")):
        raise HBFDirectPolicyPlotError(
            "aggregate.scenario.manifest_sha256 must be a lowercase SHA-256")

    grid = _mapping(aggregate.get("grid"), "aggregate.grid")
    raw_rates = _sequence(grid.get("rates"), "aggregate.grid.rates")
    if len(raw_rates) != 1:
        raise HBFDirectPolicyPlotError(
            "direct appendix requires exactly one session rate")
    session_rate = _finite(
        raw_rates[0], "aggregate.grid.rates[0]", positive=True)

    raw_seeds = _sequence(grid.get("seeds"), "aggregate.grid.seeds")
    seed_ids = tuple(
        _integer(seed, f"aggregate.grid.seeds[{index}]", minimum=0)
        for index, seed in enumerate(raw_seeds)
    )
    if (
        len(seed_ids) != EXPECTED_SEED_COUNT
        or len(set(seed_ids)) != EXPECTED_SEED_COUNT
    ):
        raise HBFDirectPolicyPlotError(
            "direct appendix requires exactly three unique seeds")

    expected_design_count = len(_EXPECTED_ROSTER)
    if _integer(
        grid.get("design_count"),
        "aggregate.grid.design_count",
    ) != expected_design_count:
        raise HBFDirectPolicyPlotError(
            f"direct appendix requires exactly {expected_design_count} "
            "designs")
    if _integer(
        grid.get("reference_count"),
        "aggregate.grid.reference_count",
    ) != len(_REFERENCE_KEYS):
        raise HBFDirectPolicyPlotError(
            "direct appendix requires exactly two references")
    expected_cell_count = (
        expected_design_count + len(_REFERENCE_KEYS)
    ) * EXPECTED_SEED_COUNT
    if _integer(
        grid.get("cell_count"),
        "aggregate.grid.cell_count",
    ) != expected_cell_count:
        raise HBFDirectPolicyPlotError(
            f"aggregate.grid.cell_count must equal {expected_cell_count}")
    executed = _integer(
        grid.get("executed_cell_count"),
        "aggregate.grid.executed_cell_count",
    )
    resumed = _integer(
        grid.get("resumed_cell_count"),
        "aggregate.grid.resumed_cell_count",
    )
    if executed + resumed != expected_cell_count:
        raise HBFDirectPolicyPlotError(
            "executed and resumed cell counts do not cover the grid")

    rate_rows = _sequence(aggregate.get("rates"), "aggregate.rates")
    if len(rate_rows) != 1:
        raise HBFDirectPolicyPlotError(
            "direct appendix requires exactly one aggregate rate row")
    rate_row = _mapping(rate_rows[0], "aggregate.rates[0]")
    row_rate = _finite(
        rate_row.get("session_rate"),
        "aggregate.rates[0].session_rate",
        positive=True,
    )
    if not _same_float(row_rate, session_rate):
        raise HBFDirectPolicyPlotError(
            "grid and aggregate row session rates disagree")
    raw_rate_seeds = _sequence(
        rate_row.get("seed_ids"), "aggregate.rates[0].seed_ids")
    if tuple(raw_rate_seeds) != seed_ids:
        raise HBFDirectPolicyPlotError(
            "rate-row seed_ids must match the three grid seeds")

    raw_references = _mapping(
        rate_row.get("references"), "aggregate.rates[0].references")
    if set(raw_references) != _REFERENCE_KEYS:
        raise HBFDirectPolicyPlotError(
            "references must be exactly baseline_cpu_ssd and oracle")
    references = {
        key: _goodput_statistic(
            _mapping(raw_references[key], f"references.{key}").get(
                GOODPUT_METRIC),
            f"references.{key}.{GOODPUT_METRIC}",
            seed_ids,
        )
        for key in sorted(_REFERENCE_KEYS)
    }

    raw_design_rows = _sequence(
        rate_row.get("designs"), "aggregate.rates[0].designs")
    if len(raw_design_rows) != expected_design_count:
        raise HBFDirectPolicyPlotError(
            f"rate row must contain exactly {expected_design_count} designs")
    candidates = []
    observed_roster = set()
    observed_keys = set()
    memory_identities = set()
    rate_designs_by_key: dict[str, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(raw_design_rows):
        path_prefix = f"aggregate.rates[0].designs[{index}]"
        row = _mapping(raw_row, path_prefix)
        (
            design,
            key,
            policy,
            read_mode,
            _,
            memory_identity,
        ) = _design_fields(row.get("design"), f"{path_prefix}.design")
        coordinate = (policy, read_mode)
        if coordinate in observed_roster:
            raise HBFDirectPolicyPlotError(
                f"duplicate direct-policy coordinate {coordinate!r}")
        if key in observed_keys:
            raise HBFDirectPolicyPlotError(
                f"duplicate direct-policy design key {key!r}")
        observed_roster.add(coordinate)
        observed_keys.add(key)
        memory_identities.add(memory_identity)
        rate_designs_by_key[key] = design
        metrics = _mapping(row.get("metrics"), f"{path_prefix}.metrics")
        goodput = _goodput_statistic(
            metrics.get(GOODPUT_METRIC),
            f"{path_prefix}.metrics.{GOODPUT_METRIC}",
            seed_ids,
        )
        candidates.append(DirectPolicyCandidate(
            key=key,
            migration_policy=policy,
            hbf_read_mode=read_mode,
            active_memory=_mapping(
                design.get("active_memory"),
                f"{path_prefix}.design.active_memory",
            ),
            goodput=goodput,
            source_index=index,
        ))
    if observed_roster != _EXPECTED_ROSTER:
        raise HBFDirectPolicyPlotError(
            "incomplete 11-policy by demand/prefetch roster: "
            f"missing={sorted(_EXPECTED_ROSTER - observed_roster)}, "
            f"extra={sorted(observed_roster - _EXPECTED_ROSTER)}")
    if len(memory_identities) != 1:
        raise HBFDirectPolicyPlotError(
            "direct appendix requires exactly one active-memory point")

    raw_grid_designs = _sequence(
        grid.get("designs"), "aggregate.grid.designs")
    if len(raw_grid_designs) != expected_design_count:
        raise HBFDirectPolicyPlotError(
            "aggregate.grid.designs does not cover the direct roster")
    grid_designs_by_key = {}
    for index, raw_design in enumerate(raw_grid_designs):
        design, key, _, _, _, _ = _design_fields(
            raw_design, f"aggregate.grid.designs[{index}]")
        if key in grid_designs_by_key:
            raise HBFDirectPolicyPlotError(
                f"duplicate aggregate.grid design key {key!r}")
        grid_designs_by_key[key] = design
    if grid_designs_by_key != rate_designs_by_key:
        raise HBFDirectPolicyPlotError(
            "aggregate.grid.designs and rate-row designs disagree")

    candidates.sort(key=lambda candidate: (
        _POLICY_PRIORITY[candidate.migration_policy],
        SUPPORTED_HBF_READ_MODES.index(candidate.hbf_read_mode),
    ))
    return LoadedDirectPolicyResults(
        source_path=source_path,
        source_aggregate_sha256=_sha256_bytes(payload),
        aggregate=aggregate,
        session_rate=session_rate,
        seed_ids=seed_ids,
        hbf_server_layouts=EXPECTED_LAYOUT,
        active_memory=candidates[0].active_memory,
        scenario_id=str(scenario_id),
        references=references,
        candidates=tuple(candidates),
    )


def _policy_label(policy: str) -> str:
    labels = {
        "eager": "Eager",
        "after_first_tool": "After first tool",
        "load_aware": "Load aware",
        "jit_oracle": "JIT oracle",
        "never": "Never migrate",
    }
    if policy in labels:
        return labels[policy]
    if policy.startswith("delay_") and policy.endswith("ms"):
        return f"Delay {policy[6:-2]} ms"
    return policy.replace("_", " ")


def select_direct_policies(
        loaded: LoadedDirectPolicyResults,
) -> dict[str, object]:
    """Select every tied best causal policy independently per read mode."""

    selected_keys = set()
    best_by_mode = {}
    causal_ranks: dict[str, int] = {}
    for read_mode in SUPPORTED_HBF_READ_MODES:
        candidates = [
            candidate
            for candidate in loaded.candidates
            if (
                candidate.hbf_read_mode == read_mode
                and not candidate.future_looking
            )
        ]
        if not candidates:
            raise HBFDirectPolicyPlotError(
                f"no causal candidates for read mode {read_mode!r}")
        best_mean = max(
            candidate.goodput.mean for candidate in candidates)
        tied = [
            candidate
            for candidate in candidates
            if _same_float(candidate.goodput.mean, best_mean)
        ]
        selected_keys.update(candidate.key for candidate in tied)
        for candidate in candidates:
            causal_ranks[candidate.key] = 1 + sum(
                1
                for other in candidates
                if (
                    other.goodput.mean > candidate.goodput.mean
                    and not _same_float(
                        other.goodput.mean,
                        candidate.goodput.mean,
                    )
                )
            )
        best_by_mode[read_mode] = {
            "mean_slo_good_output_tokens_per_second": best_mean,
            "selected_design_keys": [
                candidate.key
                for candidate in sorted(
                    tied,
                    key=lambda item: _POLICY_PRIORITY[
                        item.migration_policy],
                )
            ],
            "selected_migration_policies": [
                candidate.migration_policy
                for candidate in sorted(
                    tied,
                    key=lambda item: _POLICY_PRIORITY[
                        item.migration_policy],
                )
            ],
        }

    policy_rows = []
    for candidate in loaded.candidates:
        selected = candidate.key in selected_keys
        if candidate.future_looking:
            reason = (
                "excluded_future_looking_policy_uses_tool_gap_knowledge"
            )
        elif selected:
            reason = (
                "selected_tied_best_causal_mean_slo_goodput_for_read_mode"
            )
        else:
            reason = (
                "not_selected_lower_mean_slo_goodput_than_best_causal_policy"
            )
        policy_rows.append({
            "design_key": candidate.key,
            "migration_policy": candidate.migration_policy,
            "migration_policy_label": _policy_label(
                candidate.migration_policy),
            "hbf_read_mode": candidate.hbf_read_mode,
            "future_looking": candidate.future_looking,
            "eligible_for_main_selection": (
                not candidate.future_looking),
            "selected_for_appendix_plot": selected,
            "causal_rank_within_read_mode": (
                None
                if candidate.future_looking
                else causal_ranks[candidate.key]
            ),
            "selection_reason": reason,
            "goodput": candidate.goodput.to_json_dict(),
        })

    references = {
        key: {
            "candidate_key": key,
            "label": (
                "CPU+SSD baseline"
                if key == BASELINE_CANDIDATE_KEY
                else "Infinite-HBM Oracle"
            ),
            "performance_reference_only": (
                key == ORACLE_CANDIDATE_KEY),
            "goodput": loaded.references[key].to_json_dict(),
        }
        for key in (BASELINE_CANDIDATE_KEY, ORACLE_CANDIDATE_KEY)
    }
    return {
        "report_schema": DIRECT_POLICY_SELECTION_SCHEMA,
        "source_aggregate": {
            "path": str(loaded.source_path),
            "sha256": loaded.source_aggregate_sha256,
            "schema_version": loaded.aggregate["schema_version"],
            "scenario_id": loaded.scenario_id,
        },
        "scenario_scope": {
            "kind": "direct_gpu_hbm_to_hbf_migration_appendix",
            "session_rate": loaded.session_rate,
            "seed_ids": list(loaded.seed_ids),
            "hbf_host_count": len(loaded.hbf_server_layouts),
            "hbf_server_layouts": list(loaded.hbf_server_layouts),
            "active_memory": dict(loaded.active_memory),
            "separation_rule": (
                "Do not merge these absolute metrics with the SSD-staged "
                "HBF campaign."
            ),
        },
        "selection_contract": {
            "objective": (
                "maximum arithmetic-mean offered-load-normalized joint-SLO "
                "good output tokens per second"
            ),
            "independent_groups": list(SUPPORTED_HBF_READ_MODES),
            "causal_only": True,
            "tie_relative_tolerance": GOODPUT_TIE_REL_TOL,
            "tie_absolute_tolerance": GOODPUT_TIE_ABS_TOL,
            "future_looking_policies_excluded": sorted(
                FUTURE_LOOKING_POLICIES),
            "expected_policy_count": len(SUPPORTED_MIGRATION_POLICIES),
            "expected_read_mode_count": len(SUPPORTED_HBF_READ_MODES),
            "raw_design_count": len(loaded.candidates),
        },
        "references": references,
        "best_by_read_mode": best_by_mode,
        "selected_design_keys": sorted(selected_keys),
        "policies": policy_rows,
    }


def build_direct_plot_source_rows(
        loaded: LoadedDirectPolicyResults,
        selection: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Return references and all 22 raw policies for CSV and plotting."""

    selected_keys = set(_sequence(
        selection.get("selected_design_keys"),
        "selection.selected_design_keys",
    ))
    selection_rows = {
        row["design_key"]: row
        for row in _sequence(
            selection.get("policies"), "selection.policies")
    }
    baseline = loaded.references[BASELINE_CANDIDATE_KEY]
    oracle = loaded.references[ORACLE_CANDIDATE_KEY]
    common = {
        "plot_source_schema": DIRECT_PLOT_SOURCE_SCHEMA,
        "scenario_kind": "direct_gpu_hbm_to_hbf_migration_appendix",
        "session_rate": loaded.session_rate,
        "seed_count": len(loaded.seed_ids),
        "seed_ids": ",".join(str(seed) for seed in loaded.seed_ids),
        "hbf_host_count": len(loaded.hbf_server_layouts),
        "hbf_server_layouts": "+".join(
            loaded.hbf_server_layouts),
        "source_aggregate_sha256": loaded.source_aggregate_sha256,
    }
    rows = []
    rows.append({
        **common,
        "candidate_kind": "baseline_reference",
        "candidate_key": BASELINE_CANDIDATE_KEY,
        "migration_policy": "",
        "hbf_read_mode": "",
        "future_looking": False,
        "eligible_for_main_selection": False,
        "causal_rank_within_read_mode": "",
        "selected_for_appendix_plot": False,
        "included_in_appendix_plot": True,
        "selection_reason": "included_cpu_ssd_baseline_reference",
        "goodput_mean": baseline.mean,
        "goodput_ci95_lower": baseline.ci95_lower,
        "goodput_ci95_upper": baseline.ci95_upper,
        "goodput_seed_values": json.dumps(list(baseline.values)),
        "goodput_ratio_to_baseline": 1.0,
        "goodput_ratio_to_oracle": (
            baseline.mean / oracle.mean
            if oracle.mean > 0.0 else ""
        ),
        "plot_order": 0,
        "plot_label": "CPU+SSD\nbaseline",
    })

    selected_candidates = [
        candidate
        for candidate in loaded.candidates
        if candidate.key in selected_keys
    ]
    selected_candidates.sort(key=lambda candidate: (
        SUPPORTED_HBF_READ_MODES.index(candidate.hbf_read_mode),
        _POLICY_PRIORITY[candidate.migration_policy],
    ))
    selected_plot_order = {
        candidate.key: index + 1
        for index, candidate in enumerate(selected_candidates)
    }
    for candidate in loaded.candidates:
        audit = selection_rows[candidate.key]
        selected = candidate.key in selected_keys
        rows.append({
            **common,
            "candidate_kind": "direct_hbm_to_hbf_policy",
            "candidate_key": candidate.key,
            "migration_policy": candidate.migration_policy,
            "hbf_read_mode": candidate.hbf_read_mode,
            "future_looking": candidate.future_looking,
            "eligible_for_main_selection": (
                not candidate.future_looking),
            "causal_rank_within_read_mode": (
                ""
                if candidate.future_looking
                else audit["causal_rank_within_read_mode"]
            ),
            "selected_for_appendix_plot": selected,
            "included_in_appendix_plot": selected,
            "selection_reason": audit["selection_reason"],
            "goodput_mean": candidate.goodput.mean,
            "goodput_ci95_lower": candidate.goodput.ci95_lower,
            "goodput_ci95_upper": candidate.goodput.ci95_upper,
            "goodput_seed_values": json.dumps(
                list(candidate.goodput.values)),
            "goodput_ratio_to_baseline": (
                candidate.goodput.mean / baseline.mean
                if baseline.mean > 0.0 else ""
            ),
            "goodput_ratio_to_oracle": (
                candidate.goodput.mean / oracle.mean
                if oracle.mean > 0.0 else ""
            ),
            "plot_order": (
                selected_plot_order[candidate.key]
                if selected else ""
            ),
            "plot_label": (
                f"{candidate.hbf_read_mode.title()}\n"
                f"{_policy_label(candidate.migration_policy)}"
                if selected else ""
            ),
        })

    rows.append({
        **common,
        "candidate_kind": "oracle_reference",
        "candidate_key": ORACLE_CANDIDATE_KEY,
        "migration_policy": "",
        "hbf_read_mode": "",
        "future_looking": False,
        "eligible_for_main_selection": False,
        "causal_rank_within_read_mode": "",
        "selected_for_appendix_plot": False,
        "included_in_appendix_plot": True,
        "selection_reason": "included_infinite_hbm_performance_reference",
        "goodput_mean": oracle.mean,
        "goodput_ci95_lower": oracle.ci95_lower,
        "goodput_ci95_upper": oracle.ci95_upper,
        "goodput_seed_values": json.dumps(list(oracle.values)),
        "goodput_ratio_to_baseline": (
            oracle.mean / baseline.mean
            if baseline.mean > 0.0 else ""
        ),
        "goodput_ratio_to_oracle": 1.0,
        "plot_order": len(selected_candidates) + 1,
        "plot_label": "Infinite-HBM\nOracle",
    })
    return tuple(rows)


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                value,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv_atomic(
        path: Path,
        rows: Sequence[Mapping[str, object]],
) -> None:
    if not rows:
        raise HBFDirectPolicyPlotError("plot-source rows cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(
                target, fieldnames=tuple(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot
    except Exception as exc:
        raise HBFDirectPolicyPlotError(
            f"Matplotlib is required to render the appendix: {exc}") from exc
    return pyplot


def _render_appendix(
        path: Path,
        rows: Sequence[Mapping[str, object]],
) -> None:
    plot_rows = sorted(
        (
            row for row in rows
            if row["included_in_appendix_plot"] is True
        ),
        key=lambda row: int(row["plot_order"]),
    )
    if len(plot_rows) < 4:
        raise HBFDirectPolicyPlotError(
            "appendix plot requires two references and both read modes")
    modes = {
        row["hbf_read_mode"]
        for row in plot_rows
        if row["candidate_kind"] == "direct_hbm_to_hbf_policy"
    }
    if modes != set(SUPPORTED_HBF_READ_MODES):
        raise HBFDirectPolicyPlotError(
            "appendix plot is missing a selected read mode")

    pyplot = _load_pyplot()
    figure, axis = pyplot.subplots(figsize=(9.4, 5.8))
    labels = [str(row["plot_label"]) for row in plot_rows]
    means = [float(row["goodput_mean"]) for row in plot_rows]
    lower_errors = [
        mean - float(row["goodput_ci95_lower"])
        for mean, row in zip(means, plot_rows)
    ]
    upper_errors = [
        float(row["goodput_ci95_upper"]) - mean
        for mean, row in zip(means, plot_rows)
    ]
    colors = []
    hatches = []
    for row in plot_rows:
        if row["candidate_kind"] == "baseline_reference":
            colors.append("#7f7f7f")
            hatches.append("")
        elif row["candidate_kind"] == "oracle_reference":
            colors.append("#252525")
            hatches.append("//")
        elif row["hbf_read_mode"] == "demand":
            colors.append("#d95f02")
            hatches.append("")
        else:
            colors.append("#1b9e77")
            hatches.append("")
    bars = axis.bar(
        range(len(plot_rows)),
        means,
        yerr=[lower_errors, upper_errors],
        capsize=5,
        color=colors,
        edgecolor="#202020",
        linewidth=0.8,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    offset = max(means) * 0.018 if max(means) > 0.0 else 1.0
    for bar, mean in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + offset,
            f"{mean:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_xticks(range(len(plot_rows)), labels)
    axis.set_ylabel("Joint-SLO good output tokens/s")
    axis.set_title(
        "Direct GPU-HBM \u2192 HBF migration policy appendix\n"
        "(separate scenario; three paired arrival seeds)"
    )
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.set_ylim(
        0.0,
        max(
            mean + upper
            for mean, upper in zip(means, upper_errors)
        ) * 1.16,
    )
    figure.text(
        0.5,
        0.015,
        (
            "Direct-migration appendix only. Absolute values must not be "
            "combined with the SSD-staged HBF campaign."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.065, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    try:
        figure.savefig(
            temporary,
            format="png",
            dpi=200,
            bbox_inches="tight",
        )
        temporary.replace(path)
    finally:
        pyplot.close(figure)
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class DirectPolicyPlotArtifacts:
    selection_json: Path
    plot_source_csv: Path
    appendix_png: Path
    selected_design_keys: tuple[str, ...]


def write_direct_policy_artifacts(
        loaded: LoadedDirectPolicyResults,
        output_dir: Path | str,
        *,
        overwrite: bool = False,
) -> DirectPolicyPlotArtifacts:
    """Select policies and write the JSON, CSV, and compact appendix PNG."""

    if not isinstance(overwrite, bool):
        raise HBFDirectPolicyPlotError("overwrite must be a boolean")
    root = Path(output_dir).expanduser().resolve()
    targets = (
        root / SELECTION_FILENAME,
        root / PLOT_SOURCE_FILENAME,
        root / PLOT_FILENAME,
    )
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise HBFDirectPolicyPlotError(
            "refusing to overwrite existing appendix artifacts: "
            f"{existing}")
    selection = select_direct_policies(loaded)
    rows = build_direct_plot_source_rows(loaded, selection)
    _write_json_atomic(targets[0], selection)
    _write_csv_atomic(targets[1], rows)
    _render_appendix(targets[2], rows)
    return DirectPolicyPlotArtifacts(
        selection_json=targets[0],
        plot_source_csv=targets[1],
        appendix_png=targets[2],
        selected_design_keys=tuple(
            selection["selected_design_keys"]),
    )


def generate_direct_policy_appendix(
        aggregate_path: Path | str,
        output_dir: Path | str,
        *,
        overwrite: bool = False,
) -> DirectPolicyPlotArtifacts:
    """Validate a completed direct campaign and emit appendix artifacts."""

    loaded = load_direct_policy_aggregate(aggregate_path)
    return write_direct_policy_artifacts(
        loaded, output_dir, overwrite=overwrite)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select and plot the best causal policies from the separate "
            "direct GPU-HBM-to-HBF migration campaign."
        )
    )
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only this module's three existing output artifacts",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    artifacts = generate_direct_policy_appendix(
        args.aggregate,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"selection: {artifacts.selection_json}")
    print(f"plot source: {artifacts.plot_source_csv}")
    print(f"appendix plot: {artifacts.appendix_png}")
    print(
        "selected designs: "
        + ", ".join(artifacts.selected_design_keys)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

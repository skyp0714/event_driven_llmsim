from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from serving.core.hbf_comparison_workload import stable_json_sha256
from serving.hbf_comparison_plots import (
    ComparisonAggregate,
    ComparisonPlotInputError,
    ComparisonPlotRenderError,
    SeedCellMetrics,
    ValidatedSweep,
    _render_metric_group,
    _validate_distribution,
    aggregate_to_dict,
    aggregate_validated_sweep,
    load_validated_sweep,
)
from serving.hbf_slo_rate_selection import (
    HBFSLORateSelectionAdapterError,
    build_rate_selection_artifact,
    write_rate_selection_artifact,
)
from tests.test_hbf_comparison_plots import SyntheticSweep


SELECTOR_METRICS = (
    "joint_slo_pass_fraction",
    "slo_request_goodput_per_second",
    "slo_output_token_goodput_per_second",
)


def _digest(label: object) -> str:
    return stable_json_sha256(label)


def _fake_sweep(
        *,
        family: str = "balanced",
        equilibrium: bool = False,
) -> tuple[ValidatedSweep, ComparisonAggregate]:
    rates = (1.0, 2.0, 3.0)
    rate_texts = ("1", "2", "3")
    seeds = (101, 103, 107)
    systems = ("tiering", "hbf")
    measurement_sha = _digest(["measurement:0", "measurement:1"])
    if family == "balanced":
        scenario_id = "tracelab-balanced-3-call-causal-prefix-v1"
        scenario_manifest = {
            "schema_version": 1,
            "scenario_id": scenario_id,
            "calls_per_session": 3,
            "equilibrium_workload": equilibrium,
            "measurement_request_identities_sha256": measurement_sha,
        }
    elif family == "long_cold":
        scenario_id = (
            "tracelab-long-cold-100000-cached-native-prefix-v1")
        scenario_manifest = {
            "schema_version": 1,
            "scenario_id": scenario_id,
            "cached_prefix_threshold": 100_000,
            "successor_call_count": 2,
            "equilibrium_workload": equilibrium,
            "measurement_request_identities_sha256": measurement_sha,
        }
    else:
        raise AssertionError(f"unknown fixture family {family!r}")
    scenario_manifest_sha = stable_json_sha256(scenario_manifest)

    pairs = []
    cells = []
    cell_metrics = []
    joint_by_rate = {
        1.0: (1.0, 1.0, 1.0),
        2.0: (0.98, 0.98, 0.98),
        3.0: (0.40, 0.50, 0.60),
    }
    for seed_index, seed in enumerate(seeds):
        unit_draws_sha = _digest(["unit-draws", seed])
        offered_sessions_sha = _digest(["offered-sessions", seed])
        for rate, rate_text in zip(rates, rate_texts):
            schedule_sha = _digest(["schedule", seed, rate])
            pair = {
                "scenario_id": scenario_id,
                "seed": seed,
                "session_rate": rate,
                "rate_text": rate_text,
                "offered_session_ids_sha256": offered_sessions_sha,
                "unit_draws_sha256": unit_draws_sha,
                "schedule_sha256": schedule_sha,
                "schedule_pair_sha256": _digest(
                    ["schedule-pair", seed, rate]),
            }
            pairs.append(pair)
            for system_index, system in enumerate(systems):
                cell_contract_sha = _digest(
                    ["cell-contract", seed, rate, system])
                cells.append({
                    "seed": seed,
                    "session_rate": rate,
                    "rate_text": rate_text,
                    "system_key": system,
                    "cell_contract_sha256": cell_contract_sha,
                })
                joint = joint_by_rate[rate][seed_index]
                cell_metrics.append(SeedCellMetrics(
                    seed=seed,
                    session_rate=rate,
                    rate_text=rate_text,
                    system_key=system,
                    schedule_pair_sha256=pair["schedule_pair_sha256"],
                    values={
                        "joint_slo_pass_fraction": joint,
                        "slo_request_goodput_per_second": (
                            rate * joint + system_index * 0.01
                        ),
                        "slo_output_token_goodput_per_second": (
                            rate * joint * 100.0 + system_index
                        ),
                    },
                ))

    configs = {
        system: {
            "system_key": system,
            "policy": f"{system}-policy",
        }
        for system in systems
    }
    manifest = {
        "schema_version": 1,
        "scenario": {
            "scenario_id": scenario_id,
            "manifest": scenario_manifest,
            "manifest_sha256": scenario_manifest_sha,
        },
        "slo_thresholds_ns": {
            "first_ttft_ns": 30_000_000_000,
            "resume_ttft_ns": 30_000_000_000,
            "tpot_ns": 300_000_000,
        },
        "execution": {
            "simulation_backend": "python_analytical_discrete_event",
            "astra_cycles_used": False,
        },
        "code_revision_hashes": {
            "execution_code_sha256": _digest("execution-code"),
        },
        "system_config_contracts": configs,
        "pairing": {
            "measurement_identities_sha256": measurement_sha,
            "schedule_pairs": pairs,
        },
        "cells": cells,
        "cells_sha256": stable_json_sha256(cells),
    }
    sweep = ValidatedSweep(
        root=Path("/validated/fake"),
        manifest_path=Path("/validated/fake/manifest.json"),
        manifest_sha256=stable_json_sha256(manifest),
        manifest=manifest,
        rates=rates,
        rate_texts=rate_texts,
        seeds=seeds,
        system_keys=systems,
        metric_keys=SELECTOR_METRICS,
        cells=tuple(cell_metrics),
    )
    return sweep, aggregate_validated_sweep(sweep)


class HBFSLORateSelectionAdapterTests(unittest.TestCase):

    def test_non_equilibrium_balanced_rejects_sustainable_selection(self):
        sweep, aggregate = _fake_sweep(
            family="balanced", equilibrium=False)
        artifact = build_rate_selection_artifact(sweep, aggregate)

        identity = artifact["selection"]["manifest_identity"]
        self.assertEqual(identity["scenario_family"], "balanced")
        self.assertFalse(identity["equilibrium_workload"])
        self.assertFalse(
            artifact["adapter_contract"]["equilibrium_derivation"][
                "inferred"])
        for system in artifact["selection"]["systems"]:
            sustainable = system["sustainable_joint_slo_rate"]
            self.assertFalse(sustainable["eligible"])
            self.assertEqual(
                sustainable["status"],
                "rejected_non_equilibrium_or_non_balanced_scenario",
            )
            self.assertEqual(
                system["descriptive_request_goodput_maximum"][
                    "selected_rate"],
                2.0,
            )

    def test_non_equilibrium_long_cold_keeps_descriptive_maxima(self):
        sweep, aggregate = _fake_sweep(
            family="long_cold", equilibrium=False)
        artifact = build_rate_selection_artifact(sweep, aggregate)

        identity = artifact["selection"]["manifest_identity"]
        self.assertEqual(identity["scenario_family"], "long_cold")
        self.assertFalse(identity["equilibrium_workload"])
        for system in artifact["selection"]["systems"]:
            self.assertFalse(
                system["sustainable_joint_slo_rate"]["eligible"])
            self.assertEqual(
                system["descriptive_output_token_goodput_maximum"][
                    "selected_rate"],
                2.0,
            )

    def test_explicit_equilibrium_true_is_preserved_not_inferred(self):
        sweep, aggregate = _fake_sweep(
            family="balanced", equilibrium=True)
        artifact = build_rate_selection_artifact(sweep, aggregate)
        for system in artifact["selection"]["systems"]:
            sustainable = system["sustainable_joint_slo_rate"]
            self.assertTrue(sustainable["eligible"])
            self.assertEqual(sustainable["selected_rate"], 2.0)
            self.assertFalse(sustainable["right_censored"])

    def test_missing_or_mismatched_scenario_discriminators_fail_closed(self):
        sweep, aggregate = _fake_sweep()
        del sweep.manifest["scenario"]["manifest"]["equilibrium_workload"]
        with self.assertRaisesRegex(
                HBFSLORateSelectionAdapterError,
                "explicit boolean"):
            build_rate_selection_artifact(sweep, aggregate)

        sweep, aggregate = _fake_sweep(family="long_cold")
        sweep.manifest["scenario"]["manifest"][
            "cached_prefix_threshold"] = 99_999
        with self.assertRaisesRegex(
                HBFSLORateSelectionAdapterError, "disagree"):
            build_rate_selection_artifact(sweep, aggregate)

    def test_missing_exact_roster_and_unit_plan_drift_fail_closed(self):
        sweep, aggregate = _fake_sweep()
        scenario = sweep.manifest["scenario"]
        scenario["manifest"].pop(
            "measurement_request_identities_sha256")
        scenario["manifest_sha256"] = stable_json_sha256(
            scenario["manifest"])
        with self.assertRaisesRegex(
                HBFSLORateSelectionAdapterError, "SHA-256"):
            build_rate_selection_artifact(sweep, aggregate)

        sweep, aggregate = _fake_sweep()
        pair = next(
            value
            for value in sweep.manifest["pairing"]["schedule_pairs"]
            if value["seed"] == 101 and value["session_rate"] == 2.0
        )
        pair["unit_draws_sha256"] = _digest("redrawn")
        with self.assertRaisesRegex(
                HBFSLORateSelectionAdapterError, "unit-rate plan"):
            build_rate_selection_artifact(sweep, aggregate)

    def test_aggregate_must_exactly_match_validated_sweep(self):
        sweep, aggregate = _fake_sweep()
        changed = replace(
            aggregate, source_cells_sha256=_digest("different-cells"))
        with self.assertRaisesRegex(
                HBFSLORateSelectionAdapterError, "does not exactly equal"):
            build_rate_selection_artifact(sweep, changed)

    def test_atomic_writer_validates_self_hash(self):
        sweep, aggregate = _fake_sweep()
        artifact = build_rate_selection_artifact(sweep, aggregate)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "selection.json"
            written = write_rate_selection_artifact(output, artifact)
            self.assertEqual(written, output)
            observed = json.loads(output.read_text(encoding="utf-8"))
            digest = observed.pop("artifact_sha256")
            self.assertEqual(stable_json_sha256(observed), digest)
            self.assertEqual(
                list(output.parent.glob(f".{output.name}.*.tmp")), [])

            changed = dict(artifact)
            changed["selection_sha256"] = _digest("tampered")
            with self.assertRaisesRegex(
                    HBFSLORateSelectionAdapterError, "inconsistent"):
                write_rate_selection_artifact(output, changed)

    def test_strict_loader_can_select_only_rate_selector_metrics(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticSweep(Path(directory), repo_root)
            sweep = load_validated_sweep(
                fixture.root,
                repo_root=repo_root,
                metric_keys=SELECTOR_METRICS,
            )
            aggregate = aggregate_validated_sweep(sweep)

        self.assertEqual(sweep.metric_keys, SELECTOR_METRICS)
        self.assertEqual(aggregate.metric_keys, SELECTOR_METRICS)
        self.assertTrue(all(
            set(point.metrics) == set(SELECTOR_METRICS)
            for point in aggregate.points
        ))
        self.assertEqual(
            tuple(aggregate_to_dict(aggregate)["grid"]["metric_keys"]),
            SELECTOR_METRICS,
        )

    def test_strict_loader_validates_empty_distribution_sentinel(self):
        empty = {
            "count": 0,
            "mean_ns": None,
            "p50_ns": None,
            "p90_ns": None,
            "p95_ns": None,
            "p99_ns": None,
            "percentile_method": "inclusive_nearest_rank",
        }
        self.assertEqual(
            _validate_distribution(
                empty, expected_count=0, context="empty"),
            empty,
        )
        changed = dict(empty)
        changed["p95_ns"] = 1.0
        with self.assertRaisesRegex(
                ComparisonPlotInputError, "must be null"):
            _validate_distribution(
                changed, expected_count=0, context="empty")

    def test_partial_aggregate_render_fails_with_missing_metric(self):
        _, aggregate = _fake_sweep()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                    ComparisonPlotRenderError, "lacks required metrics"):
                _render_metric_group(
                    aggregate,
                    systems=("tiering", "hbf"),
                    metric_keys=("first_ttft_p95_seconds",),
                    output_path=Path(directory) / "should-not-exist.png",
                    figure_size=(12, 4),
                    figure_title="partial",
                )


if __name__ == "__main__":
    unittest.main()

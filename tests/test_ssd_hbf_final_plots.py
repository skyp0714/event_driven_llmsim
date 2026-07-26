from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from serving.core.hbf_comparison_workload import stable_json_sha256
from serving.ssd_hbf_design_sweep import (
    BASELINE_CANDIDATE_KEYS,
    CANONICAL_MIGRATION_POLICIES,
    ORACLE_CANDIDATE_KEY,
    REQUIRED_SESSION_RATE,
    SSD_HBF_CONTRACT_KEY,
    SSD_HBF_SWEEP_SCHEMA_VERSION,
    SUPPORTED_HBF_READ_MODES,
    SUPPORTED_RESTORE_EXECUTION_MODES,
)
from serving.ssd_hbf_final_plots import (
    PLOT_SOURCE_SCHEMA_VERSION,
    SSDHBFFinalResultsError,
    build_plot_source_rows,
    generate_final_results,
    load_staged_aggregate,
    select_meaningful_policies,
    write_final_artifacts,
)


def _stats(mean: float) -> dict[str, object]:
    return {
        "n": 3,
        "mean": mean,
        "sample_stddev": 1.0,
        "standard_error": 0.5,
        "ci95_lower": mean - 1.0,
        "ci95_upper": mean + 1.0,
        "ci_method": "student_t_95",
    }


def _metrics(goodput: float) -> dict[str, object]:
    return {
        "slo_good_output_tokens_per_second": _stats(goodput),
        "joint_slo_pass_fraction": _stats(0.98),
        "first_ttft_p95_ns": _stats(10_000.0),
        "resume_ttft_p95_ns": _stats(20_000.0),
        "tpot_p95_ns": _stats(30_000.0),
    }


def _runtime_report(
        *,
        proposed_power: float,
        proposed_energy: float,
        proposed_tco: float,
) -> dict[str, object]:
    def projection(
            system_key: str,
            power: float,
            energy: float,
            tco: float,
    ) -> dict[str, object]:
        return {
            "report_schema": "ssd-hbf-runtime-tco-v1",
            "system_key": system_key,
            "capex_usd": tco - 10_000.0,
            "trace_average_it_power_w": power,
            "five_year_it_energy_kwh": energy / 1.2,
            "five_year_facility_energy_kwh": energy,
            "five_year_runtime_electricity_opex_usd": 10_000.0,
            "five_year_tco_usd": tco,
            "replaced_static_electricity_opex_usd": 9_000.0,
            "pue": 1.2,
            "electricity_usd_per_kwh": 0.10,
        }

    return {
        "report_schema": "ssd-hbf-runtime-tco-v1",
        "baseline": projection(
            "two_gpu_local_ssd_baseline",
            14_000.0,
            735_840.0,
            640_000.0,
        ),
        "proposed": projection(
            "one_gpu_local_ssd_plus_one_hbf",
            proposed_power,
            proposed_energy,
            proposed_tco,
        ),
        "baseline_runtime": {
            "report_schema": "ssd-hbf-runtime-energy-v1",
        },
        "proposed_runtime": {
            "report_schema": "ssd-hbf-runtime-energy-v1",
        },
        "proposed_average_it_power_ratio_to_baseline": (
            proposed_power / 14_000.0),
        "proposed_five_year_it_energy_ratio_to_baseline": (
            proposed_energy / 735_840.0),
        "proposed_five_year_tco_ratio_to_baseline": (
            proposed_tco / 640_000.0),
        "incremental_average_it_power_w": (
            proposed_power - 14_000.0),
        "incremental_five_year_it_energy_kwh": (
            proposed_energy - 735_840.0),
        "incremental_five_year_tco_usd": (
            proposed_tco - 640_000.0),
    }


def _endurance(budget: float) -> dict[str, object]:
    per_card_day = budget * 1e9
    return {
        "schema_version": 1,
        "accounting_semantics": "duration weighted",
        "wear_distribution_assumption": "uniform within card",
        "sample_count": 3,
        "sample_run_ids": ["seed-1", "seed-2", "seed-3"],
        "total_observed_seconds": 30.0,
        "total_physical_write_bytes": int(
            per_card_day / 86_400.0 * 30.0 * 8),
        "total_wasted_write_bytes": 0,
        "wasted_write_fraction": 0.0,
        "model_weight_bytes_per_card_excluded": 100,
        "hotness": {
            "card_count": 8,
            "minimum_write_bytes": 10,
            "mean_write_bytes": 10.0,
            "maximum_write_bytes": 10,
            "population_stddev_write_bytes": 0.0,
            "coefficient_of_variation": 0.0,
            "maximum_to_mean": 1.0,
            "hottest_card_share": 0.125,
            "hottest_device_ids": ["card-0"],
        },
        "scenarios": {
            "slc_100k_pe_waf1": {
                "scenario": {
                    "key": "slc_100k_pe_waf1",
                },
                "service_lifetime_years": 5.0,
                "pool_years_to_first_card_eol": 5.0 / budget,
                "pool_endurance_unbounded_at_observed_write_rate": False,
                "pool_meets_service_lifetime": budget <= 1.0,
                "limiting_device_ids": ["card-0"],
                "cards": [
                    {
                        "device_id": f"card-{card_id}",
                        "server_id": 0,
                        "card_id": card_id,
                        "kv_region_capacity_bytes": 1_000_000,
                        "trace_write_bytes": 10,
                        "trace_wasted_write_bytes": 0,
                        "payload_write_bytes_per_second": (
                            per_card_day / 86_400.0),
                        "payload_write_bytes_per_day": per_card_day,
                        "wear_adjusted_write_bytes_per_day": per_card_day,
                        "full_region_writes_per_day": 0.1,
                        "years_to_eol": 5.0 / budget,
                        "endurance_unbounded_at_observed_write_rate": False,
                        "service_lifetime_budget_fraction": budget,
                        "meets_service_lifetime": budget <= 1.0,
                    }
                    for card_id in range(8)
                ],
            },
        },
    }


def _policy_values(
        policy: str,
) -> tuple[float, float, float, float]:
    if policy == "eager":
        return (120.0, 13_000.0, 680_000.0, 0.12)
    if policy == "delay_50ms":
        return (115.0, 12_000.0, 620_000.0, 0.08)
    if policy == "never":
        return (80.0, 11_000.0, 580_000.0, 0.02)
    return (100.0, 13_500.0, 700_000.0, 0.20)


def _design(
        policy: str,
        read_mode: str,
        restore_mode: str,
) -> dict[str, object]:
    policy_key = policy.replace("_", "-")
    restore_key = restore_mode.replace("_", "-")
    return {
        "key": (
            f"ssd-hbf-tp4x2-{policy_key}-{read_mode}-"
            f"{restore_key}-lpddr"),
        "hbf_layout": "tp4x2",
        "migration_policy": policy,
        "active_memory": {
            "kind": "lpddr",
            "capacity_gib_per_card": 16.0,
            "bandwidth_gbps_per_card": 409.6,
            "capex_usd_per_gib": 8.0,
            "power_w_per_gib": 0.08,
            "assumption": "synthetic",
        },
        "hbf_read_mode": read_mode,
        "restore_execution_mode": restore_mode,
        "gpu_host_count": 1,
        "hbf_host_count": 1,
        "hbf_card_count": 8,
        "simulator_layout": "tp4",
        "tco_layout": "tp4x2",
    }


def _design_row(
        design: dict[str, object],
        *,
        runtime: bool,
) -> dict[str, object]:
    policy = str(design["migration_policy"])
    canonical = "delay_1000ms" if policy == "delay_1s" else policy
    goodput, power, tco, budget = _policy_values(canonical)
    restore_mode = str(design["restore_execution_mode"])
    baseline_key = BASELINE_CANDIDATE_KEYS[restore_mode]
    row = {
        "design": copy.deepcopy(design),
        "metrics": _metrics(goodput),
        "baseline_candidate_key": baseline_key,
        "matched_reference_eligibility": {
            "eligible": True,
            "failures": [],
        },
        "hbf_endurance": _endurance(budget),
        "paired_vs_baseline_goodput": {
            "candidate_over_reference": {
                "mean": goodput / 10.0,
            },
        },
        "paired_vs_oracle_goodput": {
            "candidate_over_reference": {
                "mean": goodput / 140.0,
            },
        },
        "tco": {"static": "not used by final runtime selection"},
        "tco_unavailable_reason": None,
        "performance_tco_pareto": False,
    }
    if runtime:
        row["runtime_energy_tco"] = _runtime_report(
            proposed_power=power,
            proposed_energy=power * 52.56,
            proposed_tco=tco,
        )
    return row


def _reference(goodput: float) -> dict[str, object]:
    return _metrics(goodput)


def _aggregate(
        *,
        runtime: bool = True,
        aliases: bool = False,
) -> dict[str, object]:
    policies = list(CANONICAL_MIGRATION_POLICIES)
    if aliases:
        policies.append("delay_1s")
    designs = [
        _design(policy, read_mode, restore_mode)
        for policy in policies
        for read_mode in SUPPORTED_HBF_READ_MODES
        for restore_mode in SUPPORTED_RESTORE_EXECUTION_MODES
    ]
    design_rows = [
        _design_row(design, runtime=runtime)
        for design in designs
    ]
    references = {
        BASELINE_CANDIDATE_KEYS["bulk"]: _reference(10.0),
        BASELINE_CANDIDATE_KEYS[
            "layerwise_streaming"]: _reference(12.0),
        ORACLE_CANDIDATE_KEY: _reference(140.0),
    }
    seeds = [101, 102, 103]
    return {
        "schema_version": SSD_HBF_SWEEP_SCHEMA_VERSION,
        "comparison_contract": SSD_HBF_CONTRACT_KEY,
        "measurement_roster_sha256": "a" * 64,
        "hbf_endurance_proxy_profile": {
            "profile_id": "synthetic",
            "vendor": "Synthetic",
            "model": "Proxy",
            "source_url": "https://example.com/endurance",
            "semantics": "Synthetic SSD proxy for unit tests",
        },
        "rates": [{
            "session_rate": REQUIRED_SESSION_RATE,
            "reference_eligibility": {
                "eligible": True,
                "failures": [],
                "by_restore_execution_mode": {
                    mode: {"eligible": True, "failures": []}
                    for mode in SUPPORTED_RESTORE_EXECUTION_MODES
                },
            },
            "references": references,
            "designs": design_rows,
            "performance_tco_pareto_design_keys": [],
            "performance_ranking": [
                design["key"] for design in designs],
        }],
        "scenario": {
            "scenario_id": "synthetic-long-cold",
            "manifest_sha256": "c" * 64,
            "measurement_roster_sha256": "a" * 64,
            "measurement_identity_count": 10,
            "required_session_rate": REQUIRED_SESSION_RATE,
        },
        "grid": {
            "session_rate": REQUIRED_SESSION_RATE,
            "seeds": seeds,
            "design_count": len(designs),
            "reference_count": 3,
            "cell_count": (len(designs) + 3) * len(seeds),
            "resumed_cell_count": 0,
            "executed_cell_count": (len(designs) + 3) * len(seeds),
            "designs": copy.deepcopy(designs),
        },
        "execution_inputs_sha256": "b" * 64,
    }


def _write_aggregate(
        root: Path,
        aggregate: dict[str, object],
) -> Path:
    path = root / "aggregate.json"
    path.write_text(
        json.dumps(aggregate, sort_keys=True),
        encoding="utf-8",
    )
    return path


class SSDHBFFinalPlotsTests(unittest.TestCase):
    def test_strict_loader_accepts_complete_staged_runtime_cohort(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_aggregate(root, _aggregate())
            expected_file_hash = hashlib.sha256(
                path.read_bytes()).hexdigest()
            loaded = load_staged_aggregate(path)

        self.assertTrue(loaded.runtime_available)
        self.assertTrue(loaded.reference_eligible)
        self.assertFalse(loaded.audit_mode)
        self.assertEqual(
            len(loaded.candidates),
            len(CANONICAL_MIGRATION_POLICIES) * 4,
        )
        self.assertEqual(len(loaded.alias_collapses), 0)
        self.assertEqual(
            loaded.source_aggregate_sha256,
            expected_file_hash,
        )

    def test_ineligible_reference_requires_explicit_audit_mode(self):
        aggregate = _aggregate()
        aggregate["reference_eligibility_required"] = False
        eligibility = aggregate["rates"][0][
            "reference_eligibility"]
        eligibility["eligible"] = False
        eligibility["failures"] = [
            "bulk:baseline_over_oracle_ci95_upper_above_0.10",
            (
                "layerwise_streaming:"
                "baseline_over_oracle_ci95_upper_above_0.10"
            ),
        ]
        for mode in SUPPORTED_RESTORE_EXECUTION_MODES:
            eligibility["by_restore_execution_mode"][mode] = {
                "eligible": False,
                "failures": [
                    "baseline_over_oracle_ci95_upper_above_0.10"],
            }
        for row in aggregate["rates"][0]["designs"]:
            row["matched_reference_eligibility"] = {
                "eligible": False,
                "failures": [
                    "baseline_over_oracle_ci95_upper_above_0.10"],
            }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_aggregate(root, aggregate)
            with self.assertRaisesRegex(
                    SSDHBFFinalResultsError,
                    "failed reference eligibility"):
                load_staged_aggregate(path)
            loaded = load_staged_aggregate(
                path, allow_ineligible_reference=True)
            selection = select_meaningful_policies(loaded)
            rows = build_plot_source_rows(loaded, selection)

        self.assertFalse(loaded.reference_eligible)
        self.assertTrue(loaded.audit_mode)
        self.assertEqual(
            selection["source"]["result_status"],
            "audit_reference_ineligible",
        )
        self.assertTrue(all(
            row["result_status"] == "audit_reference_ineligible"
            and row["reference_eligible"] is False
            for row in rows
        ))

    def test_alias_is_collapsed_without_erasing_audit_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = load_staged_aggregate(
                _write_aggregate(root, _aggregate(aliases=True)))
            selection = select_meaningful_policies(loaded)

        self.assertEqual(len(loaded.alias_collapses), 4)
        self.assertEqual(
            len(loaded.candidates),
            len(CANONICAL_MIGRATION_POLICIES) * 4,
        )
        alias_audits = [
            row for row in selection["candidate_audit"]
            if row["migration_policy"] == "delay_1s"
        ]
        self.assertEqual(len(alias_audits), 4)
        self.assertTrue(all(not row["selected"] for row in alias_audits))
        self.assertTrue(all(
            row["exclusion_reasons"][0].startswith(
                "canonical_alias_duplicate_of:")
            for row in alias_audits
        ))

    def test_selection_keeps_option_winners_and_nondominated_tradeoffs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = load_staged_aggregate(
                _write_aggregate(root, _aggregate()))
            selection = select_meaningful_policies(loaded)

        selected_audit = [
            row for row in selection["candidate_audit"]
            if row["selected"]
        ]
        selected_policies = {
            row["canonical_migration_policy"]
            for row in selected_audit
        }
        self.assertEqual(
            selected_policies,
            {"eager", "delay_50ms", "never"},
        )
        eager = [
            row for row in selected_audit
            if row["canonical_migration_policy"] == "eager"
        ]
        self.assertEqual(len(eager), 4)
        self.assertTrue(all(any(
            reason.startswith("best_goodput_for_option:")
            for reason in row["selection_reasons"]
        ) for row in eager))
        ordinary = next(
            row for row in selection["candidate_audit"]
            if (
                row["canonical_migration_policy"] == "delay_25ms"
                and row["hbf_read_mode"] == "demand"
                and row["restore_execution_mode"] == "bulk"
            )
        )
        self.assertFalse(ordinary["selected"])
        self.assertTrue(any(
            reason.startswith("dominated_by:")
            for reason in ordinary["exclusion_reasons"]
        ))
        digest = selection.pop("policy_selection_sha256")
        self.assertEqual(digest, stable_json_sha256(selection))

    def test_zero_goodput_policy_remains_audited_but_is_not_selected(self):
        aggregate = _aggregate()
        target = aggregate["rates"][0]["designs"][0]
        target["metrics"] = _metrics(0.0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = load_staged_aggregate(
                _write_aggregate(root, aggregate))
            selection = select_meaningful_policies(loaded)

        audit = next(
            row for row in selection["candidate_audit"]
            if row["candidate_key"] == target["design"]["key"]
        )
        self.assertFalse(audit["selected"])
        self.assertIn(
            "zero_slo_goodput_ineligible_for_final_selection",
            audit["exclusion_reasons"],
        )

    def test_incomplete_policy_roster_fails_closed(self):
        aggregate = _aggregate()
        missing_policy = "delay_300s"
        aggregate["grid"]["designs"] = [
            row for row in aggregate["grid"]["designs"]
            if row["migration_policy"] != missing_policy
        ]
        aggregate["rates"][0]["designs"] = [
            row for row in aggregate["rates"][0]["designs"]
            if row["design"]["migration_policy"] != missing_policy
        ]
        count = len(aggregate["grid"]["designs"])
        aggregate["grid"]["design_count"] = count
        aggregate["grid"]["cell_count"] = (count + 3) * 3
        aggregate["grid"]["executed_cell_count"] = (count + 3) * 3
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_aggregate(Path(temporary), aggregate)
            with self.assertRaisesRegex(
                    SSDHBFFinalResultsError,
                    "incomplete canonical policy/option roster"):
                load_staged_aggregate(path)

    def test_missing_read_restore_option_fails_closed(self):
        aggregate = _aggregate()
        removed_key = next(
            design["key"] for design in aggregate["grid"]["designs"]
            if (
                design["migration_policy"] == "eager"
                and design["hbf_read_mode"] == "prefetch"
                and design["restore_execution_mode"]
                == "layerwise_streaming"
            )
        )
        aggregate["grid"]["designs"] = [
            row for row in aggregate["grid"]["designs"]
            if row["key"] != removed_key
        ]
        aggregate["rates"][0]["designs"] = [
            row for row in aggregate["rates"][0]["designs"]
            if row["design"]["key"] != removed_key
        ]
        count = len(aggregate["grid"]["designs"])
        aggregate["grid"]["design_count"] = count
        aggregate["grid"]["cell_count"] = (count + 3) * 3
        aggregate["grid"]["executed_cell_count"] = (count + 3) * 3
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_aggregate(Path(temporary), aggregate)
            with self.assertRaisesRegex(
                    SSDHBFFinalResultsError,
                    "incomplete canonical policy/option roster"):
                load_staged_aggregate(path)

    def test_direct_sweep_contract_is_rejected(self):
        aggregate = _aggregate()
        aggregate["comparison_contract"] = (
            "direct-hbm-to-hbf-historical-comparison")
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_aggregate(Path(temporary), aggregate)
            with self.assertRaisesRegex(
                    SSDHBFFinalResultsError,
                    "not the SSD-staged"):
                load_staged_aggregate(path)

    def test_partial_runtime_population_is_rejected(self):
        aggregate = _aggregate()
        aggregate["rates"][0]["designs"][0].pop(
            "runtime_energy_tco")
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_aggregate(Path(temporary), aggregate)
            with self.assertRaisesRegex(
                    SSDHBFFinalResultsError,
                    "partially populated"):
                load_staged_aggregate(path)

    def test_alias_disagreement_is_rejected(self):
        aggregate = _aggregate(aliases=True)
        alias = next(
            row for row in aggregate["rates"][0]["designs"]
            if (
                row["design"]["migration_policy"] == "delay_1s"
                and row["design"]["hbf_read_mode"] == "demand"
                and row["design"]["restore_execution_mode"] == "bulk"
            )
        )
        alias["metrics"][
            "slo_good_output_tokens_per_second"]["mean"] += 1.0
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_aggregate(Path(temporary), aggregate)
            with self.assertRaisesRegex(
                    SSDHBFFinalResultsError,
                    "alias disagrees"):
                load_staged_aggregate(path)

    def test_provisional_artifacts_include_full_audit_and_oracle_is_perf_only(
            self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = load_staged_aggregate(
                _write_aggregate(root, _aggregate()))
            output = root / "final"
            artifacts = write_final_artifacts(
                loaded, output, render=False)
            selection = json.loads(
                artifacts.policy_selection_json.read_text(
                    encoding="utf-8"))
            with artifacts.plot_source_csv.open(
                    encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))

        self.assertFalse(artifacts.rendered)
        self.assertTrue(artifacts.policy_selection_sha256)
        self.assertEqual(
            len(selection["candidate_audit"]),
            len(CANONICAL_MIGRATION_POLICIES) * 4,
        )
        self.assertEqual(
            {int(row["plot_source_schema_version"]) for row in rows},
            {PLOT_SOURCE_SCHEMA_VERSION},
        )
        oracle = next(
            row for row in rows
            if row["candidate_kind"] == "oracle")
        self.assertEqual(oracle["oracle_performance_only"], "True")
        self.assertNotEqual(oracle["goodput_mean"], "")
        self.assertEqual(oracle["runtime_average_it_power_w"], "")
        self.assertEqual(oracle["runtime_five_year_tco_usd"], "")
        self.assertEqual(
            oracle["hbf_five_year_budget_fraction_100k_pe_waf1"],
            "",
        )
        excluded = [
            row for row in rows
            if (
                row["candidate_kind"] == "design"
                and row["include_in_final_plots"] == "False"
            )
        ]
        self.assertTrue(excluded)
        self.assertTrue(all(
            row["exclusion_reasons"] for row in excluded))

    def test_render_fails_closed_before_writing_without_runtime_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = load_staged_aggregate(
                _write_aggregate(
                    root, _aggregate(runtime=False)))
            output = root / "final"
            with self.assertRaisesRegex(
                    SSDHBFFinalResultsError,
                    "final rendering requires complete"):
                write_final_artifacts(loaded, output, render=True)
            self.assertFalse(output.exists())

    def test_missing_matplotlib_still_writes_validated_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate_path = _write_aggregate(root, _aggregate())
            output = root / "final"
            with patch(
                    "serving.ssd_hbf_final_plots._load_pyplot",
                    return_value=None):
                artifacts = generate_final_results(
                    aggregate_path, output, render=True)

            self.assertFalse(artifacts.rendered)
            self.assertFalse(artifacts.matplotlib_available)
            self.assertTrue(artifacts.policy_selection_json.is_file())
            self.assertTrue(artifacts.plot_source_csv.is_file())
            self.assertIsNone(
                artifacts.performance_sensitivity_png)

    def test_plot_rows_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = load_staged_aggregate(
                _write_aggregate(root, _aggregate()))
            selection = select_meaningful_policies(loaded)
            first = build_plot_source_rows(loaded, selection)
            second = build_plot_source_rows(loaded, selection)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

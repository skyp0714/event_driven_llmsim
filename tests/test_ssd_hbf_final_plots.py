from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from serving.ssd_hbf_design_sweep import (
    BASELINE_CANDIDATE_KEYS,
    ORACLE_CANDIDATE_KEY,
    REQUIRED_SESSION_RATE,
    SSD_HBF_CONTRACT_KEY,
    SSD_HBF_SWEEP_SCHEMA_VERSION,
    SUPPORTED_HBF_READ_MODES,
    SUPPORTED_LAYOUTS,
    SUPPORTED_RESTORE_EXECUTION_MODES,
)
from serving.ssd_hbf_final_plots import (
    FINAL_PLOT_DESIGN_CELL_COUNT,
    PERFORMANCE_METRIC_SPECS,
    PERFORMANCE_PLOT_COUNT,
    SSDHBFFinalResultsError,
    build_plot_source_rows,
    load_frozen_selection,
    load_staged_aggregate,
    select_meaningful_policies,
    write_final_artifacts,
)


POLICIES = ("composite_ready", "composite_ready_adaptive")
HELDOUT_SEEDS = (201, 202, 203)
FROZEN_COORDINATES = (
    ("composite_ready", "tp4x2", "demand", "layerwise_streaming"),
    ("composite_ready", "tp4x2", "prefetch", "layerwise_streaming"),
    ("composite_ready", "tp8_context", "demand", "layerwise_streaming"),
    ("composite_ready", "tp8_context", "prefetch", "bulk"),
    (
        "composite_ready_adaptive",
        "tp4x2",
        "demand",
        "layerwise_streaming",
    ),
    ("composite_ready_adaptive", "tp4x2", "prefetch", "bulk"),
    (
        "composite_ready_adaptive",
        "tp8_context",
        "demand",
        "layerwise_streaming",
    ),
    (
        "composite_ready_adaptive",
        "tp8_context",
        "prefetch",
        "layerwise_streaming",
    ),
)


def _stats(mean: float) -> dict[str, object]:
    return {
        "mean": mean,
        "sample_stddev": 0.01,
        "ci95_lower": mean - 0.01,
        "ci95_upper": mean + 0.01,
        "ci_method": "student_t_95",
        "seed_ids": list(HELDOUT_SEEDS),
        "values": [mean, mean, mean],
    }


def _metrics(goodput: float) -> dict[str, object]:
    return {
        "first_ttft_p95_ns": None,
        "resume_ttft_p95_ns": _stats(20_000_000_000.0),
        "tpot_p95_ns": _stats(300_000_000.0),
        "joint_slo_pass_fraction": _stats(0.8),
        "slo_request_goodput_per_second": _stats(7.5),
        "slo_good_output_tokens_per_second": _stats(goodput),
        "observed_request_throughput_per_second": _stats(0.1),
    }


def _runtime_report(restore_mode: str, offset: float) -> dict[str, object]:
    baseline_power = 14_000.0 + (
        100.0 if restore_mode == "layerwise_streaming" else 0.0)
    baseline_energy = baseline_power * 52.56
    baseline_tco = 640_000.0

    def projection(
            system_key: str,
            power: float,
            energy: float,
            tco: float,
    ) -> dict[str, object]:
        return {
            "report_schema": "ssd-hbf-runtime-tco-v1",
            "system_key": system_key,
            "trace_average_it_power_w": power,
            "five_year_facility_energy_kwh": energy,
            "five_year_tco_usd": tco,
        }

    return {
        "report_schema": "ssd-hbf-runtime-tco-v1",
        "baseline": projection(
            "two_gpu_local_ssd_baseline",
            baseline_power,
            baseline_energy,
            baseline_tco,
        ),
        "proposed": projection(
            "one_gpu_local_ssd_plus_one_hbf",
            12_000.0 + offset,
            (12_000.0 + offset) * 52.56,
            580_000.0 + offset,
        ),
    }


def _endurance(offset: float) -> dict[str, object]:
    per_day = 1_000_000_000.0 + offset
    return {
        "schema_version": 1,
        "total_observed_seconds": 30.0,
        "total_physical_write_bytes": 8_000_000,
        "model_weight_bytes_per_card_excluded": 100,
        "hotness": {
            "card_count": 8,
            "coefficient_of_variation": 0.0,
            "hottest_card_share": 0.125,
        },
        "scenarios": {
            "slc_100k_pe_waf1": {
                "service_lifetime_years": 5.0,
                "cards": [
                    {
                        "payload_write_bytes_per_day": per_day,
                        "service_lifetime_budget_fraction": 0.1,
                    }
                    for _ in range(8)
                ],
            },
        },
    }


def _design(
        policy: str,
        layout: str,
        read_mode: str,
        restore_mode: str,
) -> dict[str, object]:
    return {
        "key": (
            f"ssd-hbf-{layout}-{policy}-{read_mode}-{restore_mode}-lpddr"),
        "hbf_layout": layout,
        "migration_policy": policy,
        "active_memory": {
            "kind": "lpddr",
            "capacity_gib_per_card": 16.0,
            "bandwidth_gbps_per_card": 409.6,
            "capex_usd_per_gib": 5.0,
            "power_w_per_gib": 0.08,
            "assumption": "synthetic",
        },
        "hbf_read_mode": read_mode,
        "restore_execution_mode": restore_mode,
        "mixed_batch_latency_limit_ms": None,
        "gpu_host_count": 1,
        "hbf_host_count": 1,
        "hbf_card_count": 8,
        "simulator_layout": (
            "tp4" if layout == "tp4x2" else "tp8_context"),
        "tco_layout": "tp4x2" if layout == "tp4x2" else "tp8",
    }


def _aggregate() -> dict[str, object]:
    designs = [
        _design(policy, layout, read_mode, restore_mode)
        for policy in POLICIES
        for layout in SUPPORTED_LAYOUTS
        for read_mode in SUPPORTED_HBF_READ_MODES
        for restore_mode in SUPPORTED_RESTORE_EXECUTION_MODES
    ]
    design_rows = []
    for index, design in enumerate(designs):
        restore_mode = str(design["restore_execution_mode"])
        design_rows.append({
            "design": copy.deepcopy(design),
            "metrics": _metrics(2_700.0 + index),
            "baseline_candidate_key": (
                BASELINE_CANDIDATE_KEYS[restore_mode]),
            "matched_reference_eligibility": {
                "eligible": False,
                "failures": [
                    "baseline_over_oracle_ci95_upper_above_0.10"],
            },
            "hbf_endurance": _endurance(float(index)),
            "runtime_energy_tco": _runtime_report(
                restore_mode, float(index)),
            "paired_vs_baseline_goodput": {},
            "paired_vs_oracle_goodput": {},
            "tco": {},
            "tco_unavailable_reason": None,
            "performance_tco_pareto": False,
        })
    references = {
        BASELINE_CANDIDATE_KEYS["bulk"]: _metrics(600.0),
        BASELINE_CANDIDATE_KEYS[
            "layerwise_streaming"]: _metrics(550.0),
        ORACLE_CANDIDATE_KEY: _metrics(3_275.4),
    }
    cell_count = (len(designs) + 3) * len(HELDOUT_SEEDS)
    failures = [
        f"{mode}:baseline_over_oracle_ci95_upper_above_0.10"
        for mode in SUPPORTED_RESTORE_EXECUTION_MODES
    ]
    return {
        "schema_version": SSD_HBF_SWEEP_SCHEMA_VERSION,
        "comparison_contract": SSD_HBF_CONTRACT_KEY,
        "reference_eligibility_required": False,
        "measurement_roster_sha256": "a" * 64,
        "execution_inputs_sha256": "b" * 64,
        "hbf_endurance_proxy_profile": {
            "profile_id": "synthetic",
            "vendor": "Synthetic",
            "model": "Proxy",
            "source_url": "https://example.com/endurance",
            "semantics": "synthetic",
        },
        "scenario": {
            "scenario_id": "synthetic-long-cold",
            "manifest_sha256": "c" * 64,
            "measurement_roster_sha256": "a" * 64,
            "measurement_identity_count": 10,
            "required_session_rate": REQUIRED_SESSION_RATE,
        },
        "grid": {
            "session_rate": REQUIRED_SESSION_RATE,
            "seeds": list(HELDOUT_SEEDS),
            "design_count": len(designs),
            "reference_count": 3,
            "cell_count": cell_count,
            "resumed_cell_count": 0,
            "executed_cell_count": cell_count,
            "designs": copy.deepcopy(designs),
        },
        "rates": [{
            "session_rate": REQUIRED_SESSION_RATE,
            "reference_eligibility": {
                "eligible": False,
                "failures": failures,
                "by_restore_execution_mode": {
                    mode: {
                        "eligible": False,
                        "failures": [
                            "baseline_over_oracle_ci95_upper_above_0.10"],
                    }
                    for mode in SUPPORTED_RESTORE_EXECUTION_MODES
                },
            },
            "references": references,
            "designs": design_rows,
        }],
    }


def _write_campaign(root: Path) -> tuple[Path, Path]:
    discovery = root / "results/discovery/aggregate.json"
    discovery.parent.mkdir(parents=True, exist_ok=True)
    discovery.write_bytes(b"frozen discovery aggregate\n")
    discovery_sha256 = hashlib.sha256(
        discovery.read_bytes()).hexdigest()
    heldout = root / "results/heldout/aggregate.json"
    heldout.parent.mkdir(parents=True, exist_ok=True)
    heldout.write_text(
        json.dumps(_aggregate(), sort_keys=True),
        encoding="utf-8",
    )
    selection = {
        "schema_version": 1,
        "selection_status": "frozen_before_heldout",
        "session_rate": REQUIRED_SESSION_RATE,
        "discovery": {
            "seeds": [101, 102],
            "aggregate_path": "results/discovery/aggregate.json",
            "aggregate_sha256": discovery_sha256,
            "selection_metric": "slo_good_output_tokens_per_second",
            "selection_direction": "maximize",
        },
        "heldout": {
            "seeds": list(HELDOUT_SEEDS),
            "output_path": "results/heldout",
        },
        "migration_policies": list(POLICIES),
        "mixed_batch_latency_limit_ms": None,
        "restore_by_coordinate": [
            {
                "migration_policy": coordinate[0],
                "hbf_layout": coordinate[1],
                "hbf_read_mode": coordinate[2],
                "restore_execution_mode": coordinate[3],
            }
            for coordinate in FROZEN_COORDINATES
        ],
    }
    selection_path = root / "selection.json"
    selection_path.write_text(
        json.dumps(selection, sort_keys=True),
        encoding="utf-8",
    )
    return heldout, selection_path


def _load_campaign(root: Path):
    aggregate_path, selection_path = _write_campaign(root)
    frozen = load_frozen_selection(
        selection_path, repo_root=root)
    loaded = load_staged_aggregate(
        aggregate_path,
        frozen_selection=frozen,
        allow_ineligible_reference=True,
    )
    return aggregate_path, selection_path, frozen, loaded


class SSDHBFFinalPlotsTests(unittest.TestCase):
    def test_frozen_manifest_selects_exact_eight_without_heldout_optimization(
            self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, _, loaded = _load_campaign(Path(temporary))
            selection = select_meaningful_policies(loaded)

        selected = {
            (
                row["migration_policy"],
                row["hbf_layout"],
                row["hbf_read_mode"],
                row["restore_execution_mode"],
            )
            for row in selection["candidate_audit"]
            if row["selected"]
        }
        self.assertEqual(selected, set(FROZEN_COORDINATES))
        self.assertEqual(len(selected), FINAL_PLOT_DESIGN_CELL_COUNT)
        self.assertFalse(
            selection["selection_algorithm"][
                "heldout_metrics_used_for_selection"])
        self.assertTrue(all(
            row["exclusion_reasons"]
            == ["restore_mode_not_frozen_from_discovery"]
            for row in selection["candidate_audit"]
            if not row["selected"]
        ))

    def test_heldout_metric_change_cannot_change_frozen_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate_path, selection_path = _write_campaign(root)
            aggregate = json.loads(
                aggregate_path.read_text(encoding="utf-8"))
            unselected = aggregate["rates"][0]["designs"][0]
            unselected["metrics"][
                "slo_good_output_tokens_per_second"
            ] = _stats(1_000_000.0)
            aggregate_path.write_text(
                json.dumps(aggregate, sort_keys=True),
                encoding="utf-8",
            )
            frozen = load_frozen_selection(
                selection_path, repo_root=root)
            loaded = load_staged_aggregate(
                aggregate_path,
                frozen_selection=frozen,
                allow_ineligible_reference=True,
            )
            selection = select_meaningful_policies(loaded)

        selected = {
            (
                row["migration_policy"],
                row["hbf_layout"],
                row["hbf_read_mode"],
                row["restore_execution_mode"],
            )
            for row in selection["candidate_audit"]
            if row["selected"]
        }
        self.assertEqual(selected, set(FROZEN_COORDINATES))
        self.assertNotIn(
            (
                unselected["design"]["migration_policy"],
                unselected["design"]["hbf_layout"],
                unselected["design"]["hbf_read_mode"],
                unselected["design"]["restore_execution_mode"],
            ),
            selected,
        )

    def test_schema10_ineligible_audit_has_seven_metrics_and_both_baselines(
            self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate_path, _, frozen, loaded = _load_campaign(root)
            with self.assertRaisesRegex(
                    SSDHBFFinalResultsError,
                    "failed reference eligibility"):
                load_staged_aggregate(
                    aggregate_path,
                    frozen_selection=frozen,
                )
            rows = build_plot_source_rows(
                loaded, select_meaningful_policies(loaded))

        self.assertTrue(loaded.audit_mode)
        self.assertTrue(all(
            row["result_status"] == "audit_reference_ineligible"
            for row in rows
        ))
        baselines = {
            row["candidate_key"]
            for row in rows
            if row["candidate_kind"] == "baseline"
            and row["include_in_final_plots"] is True
        }
        self.assertEqual(baselines, set(BASELINE_CANDIDATE_KEYS.values()))
        selected = [
            row for row in rows
            if row["candidate_kind"] == "design"
            and row["include_in_final_plots"] is True
        ]
        self.assertEqual(len(selected), FINAL_PLOT_DESIGN_CELL_COUNT)
        for spec in PERFORMANCE_METRIC_SPECS:
            self.assertIn(spec.row_field, selected[0])
        self.assertTrue(all(
            row["first_ttft_p95_ns"] == "" for row in rows))

    def test_render_writes_exactly_ten_audit_pngs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, loaded = _load_campaign(root)
            artifacts = write_final_artifacts(
                loaded, root / "plots", render=True)
            paths = [
                *artifacts.performance_metric_pngs.values(),
                artifacts.runtime_power_energy_png,
                artifacts.five_year_tco_png,
                artifacts.hbf_endurance_png,
            ]

            self.assertEqual(
                len(artifacts.performance_metric_pngs),
                PERFORMANCE_PLOT_COUNT,
            )
            self.assertEqual(len(paths), 10)
            self.assertEqual(
                [path.name for path in paths],
                [
                    "audit_01_first_ttft_p95.png",
                    "audit_02_resume_ttft_p95.png",
                    "audit_03_tpot_p95.png",
                    "audit_04_joint_slo_pass_fraction.png",
                    "audit_05_slo_request_goodput.png",
                    "audit_06_slo_output_token_goodput.png",
                    "audit_07_observed_request_throughput.png",
                    "audit_08_power_energy.png",
                    "audit_09_five_year_tco.png",
                    "audit_10_endurance.png",
                ],
            )
            self.assertTrue(all(
                path is not None
                and path.is_file()
                and path.name.startswith("audit_")
                for path in paths
            ))

    def test_manifest_hash_and_seed_mismatches_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate_path, selection_path = _write_campaign(root)
            raw = json.loads(selection_path.read_text(encoding="utf-8"))
            raw["discovery"]["aggregate_sha256"] = "0" * 64
            selection_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(
                    SSDHBFFinalResultsError,
                    "hash does not match"):
                load_frozen_selection(selection_path, repo_root=root)

            _, selection_path = _write_campaign(root)
            raw = json.loads(selection_path.read_text(encoding="utf-8"))
            raw["heldout"]["seeds"] = [301, 302, 303]
            selection_path.write_text(json.dumps(raw), encoding="utf-8")
            frozen = load_frozen_selection(
                selection_path, repo_root=root)
            with self.assertRaisesRegex(
                    SSDHBFFinalResultsError,
                    "heldout seed roster"):
                load_staged_aggregate(
                    aggregate_path,
                    frozen_selection=frozen,
                    allow_ineligible_reference=True,
                )


if __name__ == "__main__":
    unittest.main()

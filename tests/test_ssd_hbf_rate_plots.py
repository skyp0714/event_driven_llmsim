from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from serving.core.hbf_comparison_workload import stable_json_sha256
from serving.ssd_hbf_design_sweep import (
    BASELINE_CANDIDATE_KEYS,
    ORACLE_CANDIDATE_KEY,
)
from serving.ssd_hbf_rate_plots import (
    RATE_PERFORMANCE_KEYS,
    SSDHBFRatePlotError,
    build_source_rows,
    load_rate_sweep,
    write_rate_plot_artifacts,
)
from serving.ssd_hbf_rate_sweep import (
    SSD_HBF_RATE_SWEEP_CONTRACT_KEY,
    SSD_HBF_RATE_SWEEP_SCHEMA_VERSION,
    load_frozen_tp8_selection,
    run_rate_sweep,
)
from tests.test_ssd_hbf_final_plots import (
    FROZEN_COORDINATES,
    _aggregate,
    _stats,
)


RATES = (1.0, 3.0)
REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_MANIFEST_SHA256 = "d" * 64
MEASUREMENT_ROSTER_SHA256 = "c" * 64
EXECUTION_INPUTS_SHA256 = "f" * 64
TP8_COORDINATES = {
    coordinate for coordinate in FROZEN_COORDINATES
    if coordinate[1] == "tp8_context"
}
FROZEN_DESIGNS_BY_COORDINATE = {
    (
        design.migration_policy,
        design.hbf_layout,
        design.hbf_read_mode,
        design.restore_execution_mode,
    ): design.to_json_dict()
    for design in load_frozen_tp8_selection(
        repo_root=REPO_ROOT).designs
}


def _add_first_turn_and_runtime_statistics(
        row: dict[str, object],
        rate: float,
) -> None:
    metrics = row["metrics"]
    metrics["first_ttft_p95_ns"] = _stats(
        4_000_000_000.0 + rate)
    runtime = row["runtime_energy_tco"]
    statistics = {}
    for system_prefix in ("baseline", "proposed"):
        projection = runtime[system_prefix]
        for report_key in (
            "trace_average_it_power_w",
            "five_year_facility_energy_kwh",
            "five_year_tco_usd",
        ):
            statistics[f"{system_prefix}_{report_key}"] = _stats(
                float(projection[report_key]) + rate)
    runtime["aggregation"] = {
        "student_t_95_by_seed": statistics,
    }


def _rate_aggregate(rate: float) -> dict[str, object]:
    aggregate = _aggregate()
    rate_row = aggregate["rates"][0]
    selected = []
    for row in rate_row["designs"]:
        design = row["design"]
        coordinate = (
            design["migration_policy"],
            design["hbf_layout"],
            design["hbf_read_mode"],
            design["restore_execution_mode"],
        )
        if coordinate in TP8_COORDINATES:
            row["design"] = copy.deepcopy(
                FROZEN_DESIGNS_BY_COORDINATE[coordinate])
            selected.append(row)
    for row in selected:
        _add_first_turn_and_runtime_statistics(row, rate)
    for metrics in rate_row["references"].values():
        metrics["first_ttft_p95_ns"] = _stats(
            3_000_000_000.0 + rate)
    rate_row["session_rate"] = rate
    rate_row["designs"] = selected
    designs = [copy.deepcopy(row["design"]) for row in selected]
    aggregate["grid"].update({
        "session_rate": rate,
        "design_count": len(designs),
        "reference_count": 3,
        "cell_count": 21,
        "executed_cell_count": 21,
        "designs": designs,
    })
    aggregate["scenario"] = {
        "scenario_id": "balanced-synthetic",
        "scenario_manifest_type": "BalancedCausalPrefixManifest",
        "manifest_sha256": SCENARIO_MANIFEST_SHA256,
        "measurement_roster_sha256": MEASUREMENT_ROSTER_SHA256,
        "measurement_identity_count": 10,
        "required_session_rate": rate,
        "declared_session_rates": list(RATES),
    }
    aggregate["execution_inputs_sha256"] = EXECUTION_INPUTS_SHA256
    return aggregate


def _write_rate_sweep(root: Path) -> Path:
    selection_path = root / "selection.json"
    selection_path.write_bytes((
        REPO_ROOT
        / "configs/experiments/ssd_hbf_final_selection.json"
    ).read_bytes())
    entries = []
    designs = None
    for rate in RATES:
        aggregate = _rate_aggregate(rate)
        designs = aggregate["grid"]["designs"]
        directory = root / f"rate-{rate:g}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "aggregate.json"
        path.write_text(
            json.dumps(aggregate, sort_keys=True),
            encoding="utf-8",
        )
        entries.append({
            "session_rate": rate,
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    manifest = {
        "schema_version": SSD_HBF_RATE_SWEEP_SCHEMA_VERSION,
        "rate_sweep_contract": SSD_HBF_RATE_SWEEP_CONTRACT_KEY,
        "scenario": {
            "scenario_id": "balanced-synthetic",
            "manifest_type": "BalancedCausalPrefixManifest",
            "manifest_sha256": SCENARIO_MANIFEST_SHA256,
            "measurement_roster_sha256": MEASUREMENT_ROSTER_SHA256,
            "measurement_identity_count": 10,
            "declared_session_rates": list(RATES),
        },
        "hbf_layout": "tp8_context",
        "selection": {
            "path": "selection.json",
            "sha256": hashlib.sha256(
                selection_path.read_bytes()).hexdigest(),
            "schema_version": 1,
            "selection_status": "frozen_before_heldout",
        },
        "rates": list(RATES),
        "seeds": [201, 202, 203],
        "designs": designs,
        "rate_aggregates": entries,
        "execution_inputs_sha256": EXECUTION_INPUTS_SHA256,
        "reference_eligibility_required": False,
        "runtime_energy_tco_required": True,
    }
    manifest["manifest_payload_sha256"] = stable_json_sha256(
        manifest)
    path = root / "rate_sweep_manifest.json"
    path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return path


class SSDHBFRatePlotsTests(unittest.TestCase):
    def test_loads_complete_tp8_rate_grid_with_original_seven_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            loaded = load_rate_sweep(
                _write_rate_sweep(Path(temporary)),
                repo_root=temporary,
            )
            rows = build_source_rows(loaded)

        self.assertEqual(loaded.rates, RATES)
        self.assertEqual(len(loaded.series), 7)
        self.assertEqual(
            [series.key for series in loaded.series[:3]],
            [
                BASELINE_CANDIDATE_KEYS["bulk"],
                BASELINE_CANDIDATE_KEYS["layerwise_streaming"],
                ORACLE_CANDIDATE_KEY,
            ],
        )
        self.assertTrue(all(
            len(series.points) == len(RATES)
            for series in loaded.series
        ))
        self.assertEqual(
            {
                row["metric_key"]
                for row in rows
                if row["metric_group"] == "performance"
            },
            set(RATE_PERFORMANCE_KEYS),
        )
        self.assertTrue(all(
            row["hbf_layout"] == "tp8_context"
            for row in rows
            if row["series_kind"] == "design"
        ))

    def test_loader_accepts_manifest_from_actual_rate_sweep_producer(self):
        scenario_contract = {
            "scenario_id": "balanced-synthetic",
            "scenario_manifest_type": "BalancedCausalPrefixManifest",
            "manifest_sha256": SCENARIO_MANIFEST_SHA256,
            "measurement_roster_sha256": MEASUREMENT_ROSTER_SHA256,
            "measurement_identity_count": 10,
            "required_session_rate": RATES[0],
            "declared_session_rates": list(RATES),
        }
        scenario = SimpleNamespace(
            manifest=SimpleNamespace(
                arrival_contract=SimpleNamespace(rates=RATES),
            ),
        )

        def fake_run_design_space(**kwargs):
            aggregate = _rate_aggregate(kwargs["session_rate"])
            output = Path(kwargs["output_root"])
            output.mkdir(parents=True, exist_ok=True)
            path = output / "aggregate.json"
            path.write_text(
                json.dumps(aggregate, sort_keys=True),
                encoding="utf-8",
            )
            return aggregate, path

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection_path = root / "selection.json"
            selection_path.write_bytes((
                REPO_ROOT
                / "configs/experiments/ssd_hbf_final_selection.json"
            ).read_bytes())
            with (
                patch(
                    "serving.ssd_hbf_rate_sweep."
                    "validate_scenario_contract",
                    return_value=scenario_contract,
                ),
                patch(
                    "serving.ssd_hbf_rate_sweep.run_design_space",
                    side_effect=fake_run_design_space,
                ),
            ):
                _manifest, manifest_path = run_rate_sweep(
                    repo_root=root,
                    output_root=root / "rate-output",
                    scenario=scenario,
                    selection_path=selection_path,
                    rates=RATES,
                    seeds=(201, 202, 203),
                    require_eligibility=False,
                )
            loaded = load_rate_sweep(
                manifest_path, repo_root=root)

        self.assertEqual(loaded.rates, RATES)
        self.assertEqual(
            loaded.execution_inputs_sha256,
            EXECUTION_INPUTS_SHA256,
        )
        self.assertEqual(len(loaded.series), 7)

    def test_writes_audited_source_without_running_or_rendering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = load_rate_sweep(
                _write_rate_sweep(root), repo_root=root)
            artifacts = write_rate_plot_artifacts(
                loaded, root / "plots", render=False)

            self.assertTrue(artifacts.source_csv.is_file())
            self.assertTrue(
                artifacts.artifact_manifest_json.is_file())
            self.assertFalse(artifacts.rendered)
            self.assertIsNone(artifacts.performance_png)
            payload = json.loads(
                artifacts.artifact_manifest_json.read_text(
                    encoding="utf-8"))
            self.assertEqual(payload["rates"], list(RATES))
            self.assertEqual(len(payload["series_keys"]), 7)
            self.assertFalse(
                payload["reference_eligible_at_all_rates"])

    def test_synthetic_render_writes_attached_style_rate_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = load_rate_sweep(
                _write_rate_sweep(root), repo_root=root)
            artifacts = write_rate_plot_artifacts(
                loaded, root / "plots", render=True)

            self.assertTrue(artifacts.matplotlib_available)
            self.assertTrue(artifacts.rendered)
            self.assertEqual(
                [
                    artifacts.performance_png.name,
                    artifacts.runtime_power_energy_tco_png.name,
                    artifacts.hbf_endurance_png.name,
                ],
                [
                    "audit_01_tp8_rate_performance.png",
                    "audit_02_tp8_rate_power_tco.png",
                    "audit_03_tp8_rate_endurance.png",
                ],
            )
            self.assertTrue(all(
                path.is_file()
                for path in (
                    artifacts.performance_png,
                    artifacts.runtime_power_energy_tco_png,
                    artifacts.hbf_endurance_png,
                )
            ))

    def test_rejects_tp4_or_incomplete_rate_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_rate_sweep(root)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["hbf_layout"] = "tp4x2"
            manifest.pop("manifest_payload_sha256")
            manifest["manifest_payload_sha256"] = stable_json_sha256(
                manifest)
            path.write_text(
                json.dumps(manifest, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    SSDHBFRatePlotError, "tp8_context"):
                load_rate_sweep(path, repo_root=root)

            path = _write_rate_sweep(root)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["rate_aggregates"].pop()
            manifest.pop("manifest_payload_sha256")
            manifest["manifest_payload_sha256"] = stable_json_sha256(
                manifest)
            path.write_text(
                json.dumps(manifest, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    SSDHBFRatePlotError,
                    "exactly one aggregate per rate"):
                load_rate_sweep(path, repo_root=root)

    def test_manifest_and_aggregate_hashes_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_rate_sweep(root)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["rates"] = [1.0, 2.0]
            path.write_text(
                json.dumps(manifest, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    SSDHBFRatePlotError, "payload hash mismatch"):
                load_rate_sweep(path, repo_root=root)

            path = _write_rate_sweep(root)
            aggregate_path = root / "rate-1/aggregate.json"
            aggregate_path.write_text(
                aggregate_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    SSDHBFRatePlotError, "aggregate hash mismatch"):
                load_rate_sweep(path, repo_root=root)

    def test_selection_and_scenario_provenance_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_rate_sweep(root)
            selection_path = root / "selection.json"
            selection = json.loads(
                selection_path.read_text(encoding="utf-8"))
            selection["migration_policies"].reverse()
            selection_path.write_text(
                json.dumps(selection, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    SSDHBFRatePlotError,
                    "frozen selection file disagrees"):
                load_rate_sweep(path, repo_root=root)

            path = _write_rate_sweep(root)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            aggregate_path = root / manifest[
                "rate_aggregates"][0]["relative_path"]
            aggregate = json.loads(
                aggregate_path.read_text(encoding="utf-8"))
            aggregate["scenario"]["scenario_id"] = "other-scenario"
            aggregate_path.write_text(
                json.dumps(aggregate, sort_keys=True),
                encoding="utf-8",
            )
            manifest["rate_aggregates"][0]["sha256"] = hashlib.sha256(
                aggregate_path.read_bytes()).hexdigest()
            manifest.pop("manifest_payload_sha256")
            manifest["manifest_payload_sha256"] = stable_json_sha256(
                manifest)
            path.write_text(
                json.dumps(manifest, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    SSDHBFRatePlotError,
                    "scenario provenance mismatch"):
                load_rate_sweep(path, repo_root=root)


if __name__ == "__main__":
    unittest.main()

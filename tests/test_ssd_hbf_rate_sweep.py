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
from serving.core.tracelab_comparison_scenarios import (
    BALANCED_DEFAULT_RATES,
)
from serving.ssd_hbf_design_sweep import SSDHBFDesignSweepError
from serving.ssd_hbf_rate_sweep import (
    DEFAULT_FROZEN_SELECTION_PATH,
    RATE_SWEEP_MANIFEST_NAME,
    SSD_HBF_RATE_SWEEP_CONTRACT_KEY,
    TP8_CONTEXT_LAYOUT,
    _parser,
    load_frozen_tp8_selection,
    run_rate_sweep,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class SSDHBFRateSweepTests(unittest.TestCase):
    def test_frozen_selection_projects_exactly_four_tp8_coordinates(self):
        selection = load_frozen_tp8_selection(repo_root=REPO_ROOT)

        self.assertEqual(len(selection.designs), 4)
        self.assertTrue(all(
            design.hbf_layout == TP8_CONTEXT_LAYOUT
            for design in selection.designs
        ))
        self.assertEqual(
            {
                (
                    design.migration_policy,
                    design.hbf_read_mode,
                    design.restore_execution_mode,
                )
                for design in selection.designs
            },
            {
                (
                    "composite_ready",
                    "demand",
                    "layerwise_streaming",
                ),
                ("composite_ready", "prefetch", "bulk"),
                (
                    "composite_ready_adaptive",
                    "demand",
                    "layerwise_streaming",
                ),
                (
                    "composite_ready_adaptive",
                    "prefetch",
                    "layerwise_streaming",
                ),
            },
        )
        expected_sha = hashlib.sha256(
            (REPO_ROOT / DEFAULT_FROZEN_SELECTION_PATH).read_bytes()
        ).hexdigest()
        self.assertEqual(selection.sha256, expected_sha)

    def test_selection_fails_closed_when_a_tp8_coordinate_is_missing(self):
        source = json.loads(
            (REPO_ROOT / DEFAULT_FROZEN_SELECTION_PATH).read_text(
                encoding="utf-8"
            )
        )
        modified = copy.deepcopy(source)
        for index, row in enumerate(
                modified["restore_by_coordinate"]):
            if row["hbf_layout"] == TP8_CONTEXT_LAYOUT:
                modified["restore_by_coordinate"].pop(index)
                break
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(
                json.dumps(modified), encoding="utf-8")
            with self.assertRaisesRegex(
                    SSDHBFDesignSweepError,
                    "exactly four TP8 coordinates"):
                load_frozen_tp8_selection(
                    repo_root=REPO_ROOT,
                    selection_path=path,
                )

    def test_wrapper_delegates_each_sorted_rate_and_hash_pins_outputs(self):
        scenario = SimpleNamespace(
            manifest=SimpleNamespace(
                arrival_contract=SimpleNamespace(
                    rates=BALANCED_DEFAULT_RATES),
            ),
        )
        scenario_contract = {
            "scenario_id": "balanced-test",
            "scenario_manifest_type": (
                "BalancedCausalPrefixManifest"),
            "manifest_sha256": "a" * 64,
            "measurement_roster_sha256": "b" * 64,
            "measurement_identity_count": 12,
            "declared_session_rates": list(
                BALANCED_DEFAULT_RATES),
        }

        def fake_run_design_space(**kwargs):
            rate = kwargs["session_rate"]
            output = Path(kwargs["output_root"])
            output.mkdir(parents=True, exist_ok=True)
            aggregate = {
                "rates": [{"session_rate": rate}],
                "scenario": {
                    **scenario_contract,
                    "required_session_rate": rate,
                },
                "execution_inputs_sha256": "c" * 64,
                "test_payload": f"rate-{rate:g}",
            }
            path = output / "aggregate.json"
            path.write_text(
                json.dumps(aggregate, sort_keys=True),
                encoding="utf-8",
            )
            return aggregate, path

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rate-sweep"
            with (
                patch(
                    "serving.ssd_hbf_rate_sweep."
                    "validate_scenario_contract",
                    return_value=scenario_contract,
                ),
                patch(
                    "serving.ssd_hbf_rate_sweep.run_design_space",
                    side_effect=fake_run_design_space,
                ) as run,
            ):
                manifest, path = run_rate_sweep(
                    repo_root=REPO_ROOT,
                    output_root=output,
                    scenario=scenario,
                    rates=(5.0, 1.0, 3.0),
                    seeds=(203, 201, 202),
                    workers=2,
                    require_runtime_energy=False,
                )

            self.assertEqual(path, output / RATE_SWEEP_MANIFEST_NAME)
            self.assertTrue(path.is_file())
            self.assertEqual(
                manifest["rate_sweep_contract"],
                SSD_HBF_RATE_SWEEP_CONTRACT_KEY,
            )
            self.assertEqual(manifest["hbf_layout"], TP8_CONTEXT_LAYOUT)
            self.assertEqual(manifest["rates"], [1.0, 3.0, 5.0])
            self.assertEqual(manifest["seeds"], [201, 202, 203])
            self.assertEqual(len(manifest["designs"]), 4)
            self.assertEqual(
                manifest["execution_inputs_sha256"], "c" * 64)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(
                [
                    call.kwargs["session_rate"]
                    for call in run.call_args_list
                ],
                [1.0, 3.0, 5.0],
            )
            self.assertTrue(all(
                all(
                    design.hbf_layout == TP8_CONTEXT_LAYOUT
                    for design in call.kwargs["designs"]
                )
                for call in run.call_args_list
            ))
            self.assertEqual(
                [
                    row["relative_path"]
                    for row in manifest["rate_aggregates"]
                ],
                [
                    "rate-1/aggregate.json",
                    "rate-3/aggregate.json",
                    "rate-5/aggregate.json",
                ],
            )
            for row in manifest["rate_aggregates"]:
                aggregate_path = output / row["relative_path"]
                self.assertEqual(
                    row["sha256"],
                    hashlib.sha256(
                        aggregate_path.read_bytes()).hexdigest(),
                )
            unsealed = dict(manifest)
            observed = unsealed.pop("manifest_payload_sha256")
            self.assertEqual(
                observed, stable_json_sha256(unsealed))

    def test_cli_defers_default_rates_to_loaded_scenario(self):
        args = _parser().parse_args(["--output", "/tmp/rates"])

        self.assertIsNone(args.rates)
        self.assertIsNone(args.seeds)
        self.assertEqual(
            args.selection, DEFAULT_FROZEN_SELECTION_PATH)

    def test_wrapper_rejects_execution_source_drift_between_rates(self):
        scenario = SimpleNamespace(
            manifest=SimpleNamespace(
                arrival_contract=SimpleNamespace(rates=(1.0, 3.0)),
            ),
        )
        scenario_contract = {
            "scenario_id": "balanced-test",
            "scenario_manifest_type": "BalancedCausalPrefixManifest",
            "manifest_sha256": "a" * 64,
            "measurement_roster_sha256": "b" * 64,
            "measurement_identity_count": 12,
            "declared_session_rates": [1.0, 3.0],
        }

        def drifting_run_design_space(**kwargs):
            rate = kwargs["session_rate"]
            output = Path(kwargs["output_root"])
            output.mkdir(parents=True, exist_ok=True)
            aggregate = {
                "rates": [{"session_rate": rate}],
                "scenario": {
                    **scenario_contract,
                    "required_session_rate": rate,
                },
                "execution_inputs_sha256": (
                    "c" * 64 if rate == 1.0 else "d" * 64),
            }
            path = output / "aggregate.json"
            path.write_text(
                json.dumps(aggregate, sort_keys=True),
                encoding="utf-8",
            )
            return aggregate, path

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "serving.ssd_hbf_rate_sweep."
                    "validate_scenario_contract",
                    return_value=scenario_contract,
                ),
                patch(
                    "serving.ssd_hbf_rate_sweep.run_design_space",
                    side_effect=drifting_run_design_space,
                ),
                self.assertRaisesRegex(
                    SSDHBFDesignSweepError,
                    "changed between rate points",
                ),
            ):
                run_rate_sweep(
                    repo_root=REPO_ROOT,
                    output_root=Path(directory) / "rates",
                    scenario=scenario,
                    rates=(1.0, 3.0),
                    seeds=(201, 202),
                    require_runtime_energy=False,
                )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from serving.hbf_design_space_sweep import (
    BASELINE_CANDIDATE_KEY,
    DESIGN_SPACE_SCHEMA_VERSION,
    ORACLE_CANDIDATE_KEY,
    SUPPORTED_HBF_READ_MODES,
    SUPPORTED_MIGRATION_POLICIES,
)
from serving.hbf_direct_policy_plots import (
    EXPECTED_SCENARIO_ID,
    HBFDirectPolicyPlotError,
    build_direct_plot_source_rows,
    generate_direct_policy_appendix,
    load_direct_policy_aggregate,
    select_direct_policies,
)


def _statistic(mean: float, seeds=(101, 102, 103)):
    values = [mean - 1.0, mean, mean + 1.0]
    return {
        "ci95_half_width": 2.0,
        "ci95_lower": mean - 2.0,
        "ci95_upper": mean + 2.0,
        "ci_method": "student_t_95",
        "mean": mean,
        "sample_stddev": 1.0,
        "seed_ids": list(seeds),
        "values": values,
    }


def _design(policy: str, read_mode: str):
    return {
        "active_memory": {
            "assumption": "synthetic test point",
            "bandwidth_gbps_per_card": 204.8,
            "capacity_gib_per_card": 16.0,
            "capex_usd_per_gib": 5.0,
            "kind": "lpddr",
            "power_w_per_gib": 0.08,
        },
        "hbf_read_mode": read_mode,
        "hbf_server_layouts": ["tp4"],
        "key": f"hbf1-tp4-{policy}-{read_mode}-synthetic",
        "migration_policy": policy,
    }


def _aggregate():
    seeds = [101, 102, 103]
    means = {}
    for policy_index, policy in enumerate(SUPPORTED_MIGRATION_POLICIES):
        for read_mode in SUPPORTED_HBF_READ_MODES:
            means[(policy, read_mode)] = (
                100.0 + policy_index
                + (0.25 if read_mode == "prefetch" else 0.0)
            )
    # The future-looking result must never win selection.
    means[("jit_oracle", "demand")] = 999.0
    means[("jit_oracle", "prefetch")] = 999.0
    # Demand has a two-policy best tie; prefetch has one best causal policy.
    means[("eager", "demand")] = 250.0
    means[("delay_25ms", "demand")] = 250.0
    means[("delay_50ms", "prefetch")] = 260.0

    designs = []
    rows = []
    for policy in SUPPORTED_MIGRATION_POLICIES:
        for read_mode in SUPPORTED_HBF_READ_MODES:
            design = _design(policy, read_mode)
            designs.append(design)
            rows.append({
                "design": design,
                "metrics": {
                    "slo_good_output_tokens_per_second": _statistic(
                        means[(policy, read_mode)]),
                },
            })
    references = {
        BASELINE_CANDIDATE_KEY: {
            "slo_good_output_tokens_per_second": _statistic(200.0),
        },
        ORACLE_CANDIDATE_KEY: {
            "slo_good_output_tokens_per_second": _statistic(300.0),
        },
    }
    return {
        "aggregation": "synthetic Student-t seed aggregation",
        "execution_inputs_sha256": "a" * 64,
        "grid": {
            "cell_count": 72,
            "design_count": 22,
            "designs": designs,
            "executed_cell_count": 72,
            "rates": [3.0],
            "reference_count": 2,
            "resumed_cell_count": 0,
            "seeds": seeds,
        },
        "performance_metric": (
            "offered-load-normalized joint-SLO-good output tokens/s"),
        "rates": [{
            "designs": rows,
            "references": references,
            "seed_ids": seeds,
            "session_rate": 3.0,
        }],
        "scenario": {
            "manifest_sha256": "b" * 64,
            "scenario_id": EXPECTED_SCENARIO_ID,
        },
        "schema_version": DESIGN_SPACE_SCHEMA_VERSION,
        "tco_semantics": "synthetic",
    }


def _write(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


class HBFDirectPolicyPlotTests(unittest.TestCase):

    def test_selects_tied_best_causal_and_retains_jit_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            _write(path, _aggregate())
            loaded = load_direct_policy_aggregate(path)
            selection = select_direct_policies(loaded)

        selected = {
            (
                row["migration_policy"],
                row["hbf_read_mode"],
            )
            for row in selection["policies"]
            if row["selected_for_appendix_plot"]
        }
        self.assertEqual(selected, {
            ("eager", "demand"),
            ("delay_25ms", "demand"),
            ("delay_50ms", "prefetch"),
        })
        self.assertEqual(len(selection["policies"]), 22)
        jit_rows = [
            row for row in selection["policies"]
            if row["migration_policy"] == "jit_oracle"
        ]
        self.assertEqual(len(jit_rows), 2)
        self.assertTrue(all(
            row["future_looking"]
            and not row["eligible_for_main_selection"]
            and row["selection_reason"]
            == "excluded_future_looking_policy_uses_tool_gap_knowledge"
            for row in jit_rows
        ))

        source = build_direct_plot_source_rows(loaded, selection)
        self.assertEqual(len(source), 24)
        self.assertEqual(
            sum(
                row["candidate_kind"] == "direct_hbm_to_hbf_policy"
                for row in source
            ),
            22,
        )
        self.assertEqual(
            sum(bool(row["included_in_appendix_plot"]) for row in source),
            5,
        )

    def test_fails_closed_on_incomplete_or_wrong_campaign_shape(self):
        malformed = _aggregate()
        malformed["rates"][0]["designs"].pop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            _write(path, malformed)
            with self.assertRaisesRegex(
                    HBFDirectPolicyPlotError, "exactly 22 designs"):
                load_direct_policy_aggregate(path)

        malformed = _aggregate()
        malformed["grid"]["seeds"] = [101, 102]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seeds.json"
            _write(path, malformed)
            with self.assertRaisesRegex(
                    HBFDirectPolicyPlotError, "three unique seeds"):
                load_direct_policy_aggregate(path)

        malformed = _aggregate()
        malformed["grid"]["designs"][0][
            "hbf_server_layouts"] = ["tp8_context"]
        malformed["rates"][0]["designs"][0]["design"][
            "hbf_server_layouts"] = ["tp8_context"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            _write(path, malformed)
            with self.assertRaisesRegex(
                    HBFDirectPolicyPlotError, "must equal"):
                load_direct_policy_aggregate(path)

    def test_generates_json_csv_png_without_touching_source(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("Matplotlib is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "aggregate.json"
            output = root / "appendix"
            _write(source, _aggregate())
            source_before = source.read_bytes()

            artifacts = generate_direct_policy_appendix(source, output)

            self.assertEqual(source.read_bytes(), source_before)
            selection = json.loads(
                artifacts.selection_json.read_text(encoding="utf-8"))
            self.assertEqual(len(selection["policies"]), 22)
            with artifacts.plot_source_csv.open(
                    encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 24)
            self.assertEqual(
                sum(
                    row["candidate_kind"]
                    == "direct_hbm_to_hbf_policy"
                    for row in rows
                ),
                22,
            )
            self.assertEqual(
                artifacts.appendix_png.read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )
            with self.assertRaisesRegex(
                    HBFDirectPolicyPlotError, "refusing to overwrite"):
                generate_direct_policy_appendix(source, output)

    def test_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(
                '{"schema_version": 2, "schema_version": 2}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    HBFDirectPolicyPlotError, "duplicate JSON key"):
                load_direct_policy_aggregate(path)


if __name__ == "__main__":
    unittest.main()

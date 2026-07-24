from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import unittest

from serving.core.hbf_slo_rate_selection import (
    HBFSLORateSelectionError,
    JOINT_SLO_CI_LOWER_THRESHOLD,
    RATE_SELECTION_SCHEMA_VERSION,
    SCENARIO_FAMILY_BALANCED,
    SCENARIO_FAMILY_LONG_COLD,
    RateGridManifestIdentity,
    SeedRateMetricRow,
    SystemProvenanceIdentity,
    select_rate_grid_operating_points,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _manifest(
        *,
        scenario_family: str = SCENARIO_FAMILY_BALANCED,
        equilibrium_workload: bool = True,
        system_keys: tuple[str, ...] = ("tiering", "hbf"),
        rates: tuple[float, ...] = (1.0, 2.0, 3.0),
        seed_ids: tuple[int | str, ...] = (101, 103, 107, 109),
) -> RateGridManifestIdentity:
    scenario_id = (
        "tracelab-balanced-equilibrium-v1"
        if scenario_family == SCENARIO_FAMILY_BALANCED
        else "tracelab-long-cold-native-prefix-v1"
    )
    return RateGridManifestIdentity(
        schema_version=RATE_SELECTION_SCHEMA_VERSION,
        scenario_family=scenario_family,
        scenario_id=scenario_id,
        scenario_manifest_schema_version=1,
        scenario_manifest_sha256=_digest(
            f"{scenario_id}-manifest"),
        equilibrium_workload=equilibrium_workload,
        measurement_roster_sha256=_digest(
            f"{scenario_id}-measurement"),
        metric_scope="all",
        slo_contract_sha256=_digest("slo-contract"),
        metric_contract_sha256=_digest("metric-contract"),
        result_schema_revision="hbf-comparison-cell-v1/aggregate-seed-v1",
        system_keys=system_keys,
        rates=rates,
        seed_ids=seed_ids,
        system_provenance=tuple(
            SystemProvenanceIdentity(
                system_key=system_key,
                provenance_sha256=_digest(
                    f"system-provenance-{system_key}"),
            )
            for system_key in system_keys
        ),
    )


def _rows(
        manifest: RateGridManifestIdentity,
        *,
        top_joint: tuple[float, ...] = (0.94, 0.95, 0.96, 0.95),
) -> list[SeedRateMetricRow]:
    joint = {
        1.0: (0.99,) * len(manifest.seed_ids),
        2.0: (0.96, 0.97, 0.98, 0.97)[:len(manifest.seed_ids)],
        3.0: top_joint[:len(manifest.seed_ids)],
    }
    request_goodput = {1.0: 2.0, 2.0: 5.0, 3.0: 7.0}
    output_goodput = {1.0: 100.0, 2.0: 300.0, 3.0: 250.0}
    rows = []
    provenance = manifest.system_provenance_by_key
    for system_index, system_key in enumerate(manifest.system_keys):
        for rate in manifest.rates:
            values = joint[float(rate)]
            if len(values) != len(manifest.seed_ids):
                raise AssertionError("fixture joint values lack a seed")
            for seed_index, seed_id in enumerate(manifest.seed_ids):
                rows.append(SeedRateMetricRow(
                    scenario_id=manifest.scenario_id,
                    scenario_manifest_sha256=(
                        manifest.scenario_manifest_sha256),
                    measurement_roster_sha256=(
                        manifest.measurement_roster_sha256),
                    metric_scope=manifest.metric_scope,
                    slo_contract_sha256=(
                        manifest.slo_contract_sha256),
                    metric_contract_sha256=(
                        manifest.metric_contract_sha256),
                    result_schema_revision=(
                        manifest.result_schema_revision),
                    system_key=system_key,
                    system_provenance_sha256=(
                        provenance[system_key]),
                    offered_session_rate=float(rate),
                    seed_id=seed_id,
                    unit_rate_plan_sha256=_digest(
                        f"unit-plan-{seed_id}"),
                    rate_scaled_schedule_sha256=_digest(
                        f"schedule-{float(rate)}-{seed_id}"),
                    cell_manifest_sha256=_digest(
                        f"cell-{system_key}-{float(rate)}-{seed_id}"),
                    joint_slo_pass_fraction=values[seed_index],
                    slo_request_goodput_per_second=(
                        request_goodput[float(rate)] + system_index * 0.1
                    ),
                    slo_output_token_goodput_per_second=(
                        output_goodput[float(rate)]
                        + system_index * 1.0
                    ),
                ))
    return rows


class HBFSLORateSelectionTests(unittest.TestCase):

    def test_selects_conservative_rate_and_separate_descriptive_maxima(self):
        manifest = _manifest()
        rows = _rows(manifest)
        result = select_rate_grid_operating_points(manifest, rows)

        self.assertEqual(result.schema_version, 1)
        self.assertEqual(
            result.joint_slo_ci_lower_threshold,
            JOINT_SLO_CI_LOWER_THRESHOLD,
        )
        self.assertEqual(len(result.systems), 2)
        for system in result.systems:
            sustainable = system.sustainable_joint_slo_rate
            self.assertTrue(sustainable.eligible)
            self.assertEqual(sustainable.status, "selected")
            self.assertEqual(sustainable.selected_rate, 2.0)
            self.assertGreaterEqual(
                sustainable.selected_joint_slo_ci95_lower, 0.95)
            self.assertFalse(sustainable.right_censored)
            self.assertEqual(
                system.descriptive_request_goodput_maximum.selected_rate,
                3.0,
            )
            self.assertEqual(
                system.descriptive_output_token_goodput_maximum.selected_rate,
                2.0,
            )
            self.assertFalse(
                system.descriptive_request_goodput_maximum
                .sustainable_ceiling_claim
            )
            self.assertFalse(
                system.descriptive_output_token_goodput_maximum
                .sustainable_ceiling_claim
            )
            self.assertEqual(
                [
                    point.joint_slo_ci_lower_qualifies
                    for point in system.rate_points
                ],
                [True, True, False],
            )

        # Canonical row ordering makes the artifact content-address stable.
        reversed_result = select_rate_grid_operating_points(
            manifest, list(reversed(rows)))
        self.assertEqual(
            result.canonical_input_rows_sha256,
            reversed_result.canonical_input_rows_sha256,
        )
        self.assertEqual(result.systems, reversed_result.systems)
        json.dumps(result.to_dict(), sort_keys=True, allow_nan=False)

    def test_top_qualifying_grid_point_is_right_censored(self):
        manifest = _manifest()
        result = select_rate_grid_operating_points(
            manifest,
            _rows(manifest, top_joint=(0.98, 0.98, 0.98, 0.98)),
        )
        for system in result.systems:
            selected = system.sustainable_joint_slo_rate
            self.assertEqual(selected.selected_rate, 3.0)
            self.assertTrue(selected.right_censored)
            self.assertIn("no saturation boundary", selected.semantics)

    def test_no_qualifying_tested_rate_is_explicit(self):
        manifest = _manifest()
        rows = [
            replace(row, joint_slo_pass_fraction=0.90)
            for row in _rows(manifest)
        ]
        result = select_rate_grid_operating_points(manifest, rows)
        for system in result.systems:
            selected = system.sustainable_joint_slo_rate
            self.assertTrue(selected.eligible)
            self.assertEqual(
                selected.status,
                "no_tested_rate_meets_joint_slo_ci_floor",
            )
            self.assertIsNone(selected.selected_rate)
            self.assertFalse(selected.right_censored)

    def test_long_cold_rejects_sustainable_but_keeps_descriptive_maxima(self):
        manifest = _manifest(
            scenario_family=SCENARIO_FAMILY_LONG_COLD,
            equilibrium_workload=False,
        )
        result = select_rate_grid_operating_points(
            manifest, _rows(manifest))
        for system in result.systems:
            selected = system.sustainable_joint_slo_rate
            self.assertFalse(selected.eligible)
            self.assertEqual(
                selected.status,
                "rejected_non_equilibrium_or_non_balanced_scenario",
            )
            self.assertIsNone(selected.selected_rate)
            self.assertIsNone(selected.right_censored)
            self.assertEqual(
                system.descriptive_request_goodput_maximum.selected_rate,
                3.0,
            )
            self.assertTrue(all(
                point.joint_slo_ci_lower_qualifies is None
                for point in system.rate_points
            ))

    def test_exact_cartesian_grid_rejects_missing_duplicate_and_extra(self):
        manifest = _manifest()
        rows = _rows(manifest)
        with self.assertRaisesRegex(
                HBFSLORateSelectionError, "Cartesian product"):
            select_rate_grid_operating_points(manifest, rows[:-1])
        with self.assertRaisesRegex(
                HBFSLORateSelectionError, "duplicate rate-grid cell"):
            select_rate_grid_operating_points(
                manifest, rows + [rows[0]])

        unexpected = replace(
            rows[0],
            offered_session_rate=4.0,
            cell_manifest_sha256=_digest("unexpected-cell"),
        )
        with self.assertRaisesRegex(
                HBFSLORateSelectionError, "unexpected rate"):
            select_rate_grid_operating_points(
                manifest, [unexpected] + rows[1:])

    def test_paired_schedule_and_plan_provenance_fail_closed(self):
        manifest = _manifest()
        rows = _rows(manifest)
        target = next(
            index
            for index, row in enumerate(rows)
            if (
                row.system_key == "hbf"
                and row.offered_session_rate == 2.0
                and row.seed_id == 101
            )
        )

        changed_schedule = list(rows)
        changed_schedule[target] = replace(
            changed_schedule[target],
            rate_scaled_schedule_sha256=_digest("different-schedule"),
        )
        with self.assertRaisesRegex(
                HBFSLORateSelectionError,
                "schedule provenance differs"):
            select_rate_grid_operating_points(
                manifest, changed_schedule)

        changed_plan = list(rows)
        changed_plan[target] = replace(
            changed_plan[target],
            unit_rate_plan_sha256=_digest("different-plan"),
        )
        with self.assertRaisesRegex(
                HBFSLORateSelectionError,
                "unit-rate plan provenance differs"):
            select_rate_grid_operating_points(manifest, changed_plan)

    def test_manifest_and_row_provenance_fail_closed(self):
        manifest = _manifest()
        rows = _rows(manifest)

        changed = list(rows)
        changed[0] = replace(
            changed[0],
            scenario_manifest_sha256=_digest("different-manifest"),
        )
        with self.assertRaisesRegex(
                HBFSLORateSelectionError,
                "common provenance mismatch"):
            select_rate_grid_operating_points(manifest, changed)

        changed = list(rows)
        changed[0] = replace(
            changed[0],
            system_provenance_sha256=_digest("different-system"),
        )
        with self.assertRaisesRegex(
                HBFSLORateSelectionError,
                "system provenance differs"):
            select_rate_grid_operating_points(manifest, changed)

        changed = list(rows)
        changed[1] = replace(
            changed[1],
            cell_manifest_sha256=changed[0].cell_manifest_sha256,
        )
        with self.assertRaisesRegex(
                HBFSLORateSelectionError,
                "cell manifest digest is reused"):
            select_rate_grid_operating_points(manifest, changed)

    def test_nonfinite_out_of_range_and_numeric_strings_are_rejected(self):
        manifest = _manifest()
        row = _rows(manifest)[0]
        for field, value in (
            ("joint_slo_pass_fraction", math.nan),
            ("joint_slo_pass_fraction", 1.01),
            ("slo_request_goodput_per_second", -1.0),
            ("slo_output_token_goodput_per_second", math.inf),
            ("offered_session_rate", "3.0"),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(HBFSLORateSelectionError):
                    replace(row, **{field: value})

    def test_equilibrium_needs_two_seeds_but_non_equilibrium_does_not(self):
        equilibrium = _manifest(seed_ids=(101,))
        with self.assertRaisesRegex(
                HBFSLORateSelectionError, "at least two seeds"):
            select_rate_grid_operating_points(
                equilibrium, _rows(equilibrium))

        long_cold = _manifest(
            scenario_family=SCENARIO_FAMILY_LONG_COLD,
            equilibrium_workload=False,
            seed_ids=(101,),
        )
        result = select_rate_grid_operating_points(
            long_cold, _rows(long_cold))
        aggregate = (
            result.systems[0]
            .descriptive_request_goodput_maximum
            .seed_aggregate
        )
        self.assertEqual(aggregate.ci_method, "unavailable_single_seed")

    def test_manifest_rejects_contradictory_or_incomplete_provenance(self):
        with self.assertRaisesRegex(
                HBFSLORateSelectionError, "non-equilibrium"):
            _manifest(
                scenario_family=SCENARIO_FAMILY_LONG_COLD,
                equilibrium_workload=True,
            )
        with self.assertRaisesRegex(
                HBFSLORateSelectionError,
                "system_provenance keys differ"):
            replace(
                _manifest(),
                system_provenance=(
                    SystemProvenanceIdentity(
                        "tiering", _digest("only-tiering")),
                ),
            )
        with self.assertRaisesRegex(
                HBFSLORateSelectionError, "strictly increasing"):
            replace(_manifest(), rates=(1.0, 1.0))


if __name__ == "__main__":
    unittest.main()

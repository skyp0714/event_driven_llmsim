import json
from pathlib import Path
import unittest

from serving.core.endurance_model import DeviceProfile
from serving.core.hbf_endurance import (
    HBFEnduranceError,
    HBFEnduranceScenario,
    HBFWriteSample,
    default_hbf_endurance_scenarios,
    project_hbf_endurance,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MICRON_PROFILE = (
    REPO_ROOT / "configs" / "storage"
    / "micron_9550_pro_3_84tb.json"
)


def _accounting(
        writes, *, capacity=1_000, complete=True,
        wasted=None):
    write_values = tuple(writes)
    wasted_values = (
        (0,) * len(write_values)
        if wasted is None else tuple(wasted)
    )
    return {
        "schema_version": 1,
        "accounting_basis": (
            "physical_media_payload_of_admitted_jobs"),
        "complete_for_endurance_projection": complete,
        "total_physical_write_bytes": sum(write_values),
        "wasted_physical_write_bytes": sum(wasted_values),
        "static_model_weight": {
            "bytes_per_card": 100,
            "write_count": 1,
            "included_in_recurring_kv_wear": False,
        },
        "cards": [
            {
                "device_id": f"hbf-server-0-card-{index}",
                "server_id": 0,
                "card_id": index,
                "kv_region_capacity_bytes": capacity,
                "total_write_bytes": write_bytes,
                "wasted_write_bytes": wasted_values[index],
            }
            for index, write_bytes in enumerate(write_values)
        ],
    }


def _raw_scenario(*, pe=100_000.0, waf=1.0):
    return HBFEnduranceScenario(
        key=f"pe{pe:g}-waf{waf:g}",
        rated_full_region_writes=pe,
        write_amplification_factor=waf,
        accounting_basis="raw_hbf_pe_cycles",
        waf_affects_lifetime=True,
        assumption="synthetic raw P/E test",
        source_url="https://example.com/raw-pe",
    )


class HBFEnduranceTests(unittest.TestCase):
    def test_exact_one_full_write_per_day_projection(self):
        sample = HBFWriteSample.from_write_accounting(
            run_id="one-day",
            duration_seconds=86_400,
            write_accounting=_accounting((1_000, 1_000)),
        )
        report = project_hbf_endurance(
            (sample,), (_raw_scenario(),))
        scenario = report["scenarios"]["pe100000-waf1"]

        self.assertAlmostEqual(
            scenario["pool_years_to_first_card_eol"],
            100_000 / 365,
        )
        self.assertTrue(
            scenario["pool_meets_service_lifetime"])
        self.assertTrue(all(
            card["full_region_writes_per_day"] == 1.0
            for card in scenario["cards"]
        ))
        self.assertEqual(
            report["model_weight_bytes_per_card_excluded"],
            100,
        )

    def test_hottest_card_limits_pool_and_samples_weight_by_duration(self):
        first = HBFWriteSample.from_write_accounting(
            run_id="first",
            duration_seconds=10,
            write_accounting=_accounting((2_000, 1_000)),
        )
        second = HBFWriteSample.from_write_accounting(
            run_id="second",
            duration_seconds=30,
            write_accounting=_accounting((0, 500)),
        )
        report = project_hbf_endurance(
            (first, second), (_raw_scenario(),))
        scenario = report["scenarios"]["pe100000-waf1"]
        by_id = {
            card["device_id"]: card
            for card in scenario["cards"]
        }

        self.assertEqual(report["total_observed_seconds"], 40)
        self.assertEqual(
            by_id["hbf-server-0-card-0"][
                "payload_write_bytes_per_second"],
            50.0,
        )
        self.assertEqual(
            by_id["hbf-server-0-card-1"][
                "payload_write_bytes_per_second"],
            37.5,
        )
        self.assertEqual(
            scenario["limiting_device_ids"],
            ["hbf-server-0-card-0"],
        )

    def test_raw_pe_and_waf_sensitivities_are_monotonic(self):
        sample = HBFWriteSample.from_write_accounting(
            run_id="sensitivity",
            duration_seconds=1,
            write_accounting=_accounting((100, 50)),
        )
        report = project_hbf_endurance(
            (sample,),
            (
                _raw_scenario(pe=100_000, waf=1.0),
                _raw_scenario(pe=100_000, waf=1.3),
                _raw_scenario(pe=100_000, waf=2.0),
                _raw_scenario(pe=1_000_000, waf=1.0),
            ),
        )
        scenarios = report["scenarios"]
        life_100k = scenarios[
            "pe100000-waf1"]["pool_years_to_first_card_eol"]
        life_1m = scenarios[
            "pe1e+06-waf1"]["pool_years_to_first_card_eol"]

        self.assertAlmostEqual(life_1m, life_100k * 10)
        self.assertGreater(
            life_100k,
            scenarios["pe100000-waf1.3"][
                "pool_years_to_first_card_eol"],
        )
        self.assertGreater(
            scenarios["pe100000-waf1.3"][
                "pool_years_to_first_card_eol"],
            scenarios["pe100000-waf2"][
                "pool_years_to_first_card_eol"],
        )

    def test_ssd_proxy_full_write_anchors_are_exact(self):
        profile = DeviceProfile.from_json_file(MICRON_PROFILE)
        scenarios = {
            scenario.key: scenario
            for scenario in default_hbf_endurance_scenarios(profile)
        }
        random_proxy = scenarios["ssd_proxy_random_4k"]
        sequential_proxy = scenarios[
            "ssd_proxy_sequential_128k"]

        self.assertAlmostEqual(
            random_proxy.rated_full_region_writes,
            1_825.0,
        )
        self.assertAlmostEqual(
            sequential_proxy.rated_full_region_writes,
            7_656.25,
        )
        self.assertFalse(random_proxy.waf_affects_lifetime)

    def test_zero_write_projection_uses_null_not_infinity(self):
        sample = HBFWriteSample.from_write_accounting(
            run_id="zero",
            duration_seconds=1,
            write_accounting=_accounting((0, 0)),
        )
        report = project_hbf_endurance(
            (sample,), (_raw_scenario(),))
        scenario = report["scenarios"]["pe100000-waf1"]

        self.assertIsNone(
            scenario["pool_years_to_first_card_eol"])
        self.assertTrue(
            scenario[
                "pool_endurance_unbounded_at_observed_write_rate"])
        self.assertTrue(
            scenario["pool_meets_service_lifetime"])
        self.assertIsNone(
            report["hotness"]["coefficient_of_variation"])
        json.dumps(report, allow_nan=False, sort_keys=True)

    def test_pending_or_inconsistent_accounting_fails_closed(self):
        with self.assertRaisesRegex(
                HBFEnduranceError, "pending jobs"):
            HBFWriteSample.from_write_accounting(
                run_id="pending",
                duration_seconds=1,
                write_accounting=_accounting(
                    (1, 1), complete=False),
            )
        invalid = _accounting((1, 1))
        invalid["total_physical_write_bytes"] = 3
        with self.assertRaisesRegex(
                HBFEnduranceError, "per-card"):
            HBFWriteSample.from_write_accounting(
                run_id="invalid",
                duration_seconds=1,
                write_accounting=invalid,
            )


if __name__ == "__main__":
    unittest.main()

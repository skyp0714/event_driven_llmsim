import csv
import json
import tempfile
import unittest
from pathlib import Path

from serving.core.endurance_model import (
    DECIMAL_TB,
    DeviceProfile,
    EnduranceConfigError,
    ProjectionAssumptions,
    RunWriteStats,
    project_endurance,
    write_report_csv,
    write_report_json,
)
from serving.endurance import main as endurance_main


REPO_ROOT = Path(__file__).resolve().parents[1]
STORAGE_CONFIG = REPO_ROOT / "configs" / "storage"
MICRON_PROFILE = STORAGE_CONFIG / "micron_9550_pro_3_84tb.json"
SOLIDIGM_1_DWPD = STORAGE_CONFIG / "solidigm_d7_ps1010_3_84tb.json"
SOLIDIGM_3_DWPD = STORAGE_CONFIG / "solidigm_d7_ps1030_3_2tb.json"


class EnduranceModelTest(unittest.TestCase):
    def test_catalog_uses_decimal_si_and_resolves_dwpd(self):
        micron = DeviceProfile.from_json_file(MICRON_PROFILE)
        self.assertEqual(micron.capacity_bytes, 3_840_000_000_000)
        conservative = micron.select_rating()
        self.assertEqual(
            conservative.resolve_tbw_bytes(micron.capacity_bytes),
            7_008_000_000_000_000,
        )
        self.assertAlmostEqual(
            conservative.resolve_dwpd(micron.capacity_bytes), 1.0
        )

        solidigm_1 = DeviceProfile.from_json_file(SOLIDIGM_1_DWPD)
        self.assertEqual(
            solidigm_1.select_rating().resolve_tbw_bytes(
                solidigm_1.capacity_bytes
            ),
            7_008_000_000_000_000,
        )
        solidigm_3 = DeviceProfile.from_json_file(SOLIDIGM_3_DWPD)
        self.assertEqual(solidigm_3.capacity_bytes, 3_200_000_000_000)
        self.assertEqual(
            solidigm_3.select_rating().resolve_tbw_bytes(
                solidigm_3.capacity_bytes
            ),
            17_520_000_000_000_000,
        )

    def test_micron_sequential_rating_is_explicit_sensitivity(self):
        profile = DeviceProfile.from_json_file(MICRON_PROFILE)
        default = profile.select_rating()
        sequential = profile.select_rating("sequential_128k_sensitivity")
        self.assertFalse(default.sensitivity_only)
        self.assertTrue(sequential.sensitivity_only)
        self.assertEqual(
            sequential.resolve_tbw_bytes(profile.capacity_bytes),
            29_400_000_000_000_000,
        )
        self.assertGreater(
            sequential.resolve_tbw_bytes(profile.capacity_bytes),
            default.resolve_tbw_bytes(profile.capacity_bytes),
        )

    def test_host_tbw_lifetime_does_not_double_count_waf(self):
        profile = DeviceProfile.from_json_file(MICRON_PROFILE)
        stats = RunWriteStats(
            run_id="one-tb-hour",
            host_write_bytes=DECIMAL_TB,
            trace_period_seconds=3600.0,
        )
        waf_1 = project_endurance(
            stats, profile, ProjectionAssumptions(waf=1.0)
        )
        waf_3 = project_endurance(
            stats, profile, ProjectionAssumptions(waf=3.0)
        )

        first = waf_1.devices[0]
        amplified = waf_3.devices[0]
        self.assertAlmostEqual(first.years_to_tbw, 0.8)
        self.assertEqual(first.years_to_tbw, amplified.years_to_tbw)
        self.assertEqual(
            amplified.trace_estimated_nand_write_bytes,
            3 * DECIMAL_TB,
        )
        self.assertEqual(waf_3.accounting_mode, "host_tbw")
        self.assertFalse(waf_3.to_dict()["assumptions"]["waf_affects_lifetime"])

    def test_period_and_duty_cycle_resolve_replays_per_day(self):
        profile = DeviceProfile.from_json_file(MICRON_PROFILE)
        stats = RunWriteStats("duty", DECIMAL_TB)
        report = project_endurance(
            stats,
            profile,
            ProjectionAssumptions(
                trace_period_seconds=3600.0,
                duty_cycle=0.5,
            ),
        )
        self.assertAlmostEqual(report.effective_replays_per_day, 12.0)
        self.assertAlmostEqual(
            report.devices[0].host_write_bytes_per_day,
            12 * DECIMAL_TB,
        )

    def test_direct_replays_per_day(self):
        profile = DeviceProfile.from_json_file(MICRON_PROFILE)
        stats = RunWriteStats("direct", 1000)
        report = project_endurance(
            stats,
            profile,
            ProjectionAssumptions(replays_per_day=7.5),
        )
        self.assertEqual(report.effective_replays_per_day, 7.5)
        self.assertEqual(report.devices[0].host_write_bytes_per_day, 7500)

    def test_balanced_multi_device_distribution_preserves_bytes(self):
        profile = DeviceProfile.from_json_file(MICRON_PROFILE)
        stats = RunWriteStats("balanced", 803, 83)
        report = project_endurance(
            stats,
            profile,
            ProjectionAssumptions(replays_per_day=1),
            num_devices=8,
        )
        writes = [device.trace_host_write_bytes for device in report.devices]
        reads = [device.trace_host_read_bytes for device in report.devices]
        self.assertEqual(sum(writes), 803)
        self.assertEqual(sum(reads), 83)
        self.assertLessEqual(max(writes) - min(writes), 1)
        self.assertLessEqual(max(reads) - min(reads), 1)
        self.assertEqual(report.distribution_mode, "balanced")

    def test_explicit_device_writes_set_first_device_eol(self):
        profile = DeviceProfile.from_json_file(MICRON_PROFILE)
        stats = RunWriteStats.from_dict({
            "run_id": "uneven",
            "host_write_bytes": 400 * DECIMAL_TB,
            "devices": [
                {"device_id": "cold", "host_write_bytes": 100 * DECIMAL_TB},
                {"device_id": "hot", "host_write_bytes": 300 * DECIMAL_TB},
            ],
        })
        report = project_endurance(
            stats,
            profile,
            ProjectionAssumptions(replays_per_day=1),
        )
        by_id = {device.device_id: device for device in report.devices}
        self.assertEqual(report.distribution_mode, "explicit")
        self.assertLess(by_id["hot"].years_to_tbw, by_id["cold"].years_to_tbw)
        self.assertEqual(
            report.pool_years_to_first_device_eol,
            by_id["hot"].years_to_tbw,
        )

    def test_nested_stats_and_device_mapping_are_accepted(self):
        stats = RunWriteStats.from_dict({
            "run_id": "nested",
            "offered_trace_period_ns": 2_000_000_000,
            "storage": {
                "totals": {
                    "aligned_host_write_bytes": 40,
                    "host_read_bytes": 12,
                },
                "device_host_write_bytes": {"a": 10, "b": 30},
                "device_host_read_bytes": {"a": 2, "b": 10},
            },
        })
        self.assertEqual(stats.trace_period_seconds, 2.0)
        self.assertEqual(stats.host_write_bytes, 40)
        self.assertEqual(stats.host_read_bytes, 12)
        self.assertEqual([device.device_id for device in stats.devices], ["a", "b"])

    def test_aggregate_reads_can_accompany_explicit_device_writes(self):
        stats = RunWriteStats.from_dict({
            "host_write_bytes": 40,
            "host_read_bytes": 11,
            "devices": [
                {"device_id": "a", "host_write_bytes": 10},
                {"device_id": "b", "host_write_bytes": 30},
            ],
        })
        self.assertEqual(sum(d.host_read_bytes for d in stats.devices), 11)
        self.assertLessEqual(
            max(d.host_read_bytes for d in stats.devices)
            - min(d.host_read_bytes for d in stats.devices),
            1,
        )

    def test_inconsistent_explicit_total_is_rejected(self):
        with self.assertRaises(EnduranceConfigError):
            RunWriteStats.from_dict({
                "host_write_bytes": 10,
                "devices": [{"device_id": "a", "host_write_bytes": 9}],
            })

    def test_replay_modes_are_mutually_exclusive(self):
        with self.assertRaises(EnduranceConfigError):
            ProjectionAssumptions(
                replays_per_day=1,
                trace_period_seconds=3600,
            )
        with self.assertRaises(EnduranceConfigError):
            ProjectionAssumptions(replays_per_day=1, duty_cycle=0.5)

    def test_zero_write_rate_serializes_without_nonstandard_infinity(self):
        profile = DeviceProfile.from_json_file(MICRON_PROFILE)
        report = project_endurance(
            RunWriteStats("idle", 0),
            profile,
            ProjectionAssumptions(replays_per_day=0),
        )
        self.assertIsNone(report.devices[0].years_to_tbw)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "report.json"
            write_report_json(report, output)
            text = output.read_text(encoding="utf-8")
            self.assertNotIn("Infinity", text)
            parsed = json.loads(text)
            self.assertIsNone(parsed["devices"][0]["years_to_tbw"])
            self.assertTrue(
                parsed["devices"][0][
                    "endurance_unbounded_at_projected_write_rate"
                ]
            )

    def test_json_and_csv_have_one_row_per_device(self):
        profile = DeviceProfile.from_json_file(MICRON_PROFILE)
        report = project_endurance(
            RunWriteStats("outputs", 2 * DECIMAL_TB),
            profile,
            ProjectionAssumptions(replays_per_day=2, waf=1.2),
            num_devices=2,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "nested" / "report.json"
            csv_path = Path(temp_dir) / "nested" / "report.csv"
            write_report_json(report, json_path)
            write_report_csv(report, csv_path)

            parsed = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["device_profile"]["num_devices"], 2)
            self.assertEqual(len(parsed["devices"]), 2)
            with csv_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["accounting_mode"] for row in rows}, {"host_tbw"})
            self.assertEqual({row["waf_affects_lifetime"] for row in rows}, {"False"})

    def test_cli_writes_json_and_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            stats_path = temp / "stats.json"
            json_path = temp / "report.json"
            csv_path = temp / "report.csv"
            stats_path.write_text(json.dumps({
                "run_id": "cli",
                "host_write_bytes": DECIMAL_TB,
                "trace_period_seconds": 3600,
            }), encoding="utf-8")

            result = endurance_main([
                "--stats", str(stats_path),
                "--device-profile", str(MICRON_PROFILE),
                "--num-devices", "2",
                "--duty-cycle", "0.5",
                "--waf", "2",
                "--output-json", str(json_path),
                "--output-csv", str(csv_path),
            ])
            self.assertEqual(result, 0)
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            parsed = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["run_id"], "cli")
            self.assertEqual(parsed["assumptions"]["effective_replays_per_day"], 12)

    def test_cli_accepts_agentic_analysis_tiered_counter(self):
        analysis = {
            "summaries": [{
                "model": "org/model",
                "hardware": "H100",
                "tiered_policy": {"ssd_host_write_bytes": 8 * DECIMAL_TB},
                "ssd_swap": {
                    "host_write_bytes": {
                        "full_rewrite_all_attempts": 16 * DECIMAL_TB,
                        "full_rewrite_issued_under_selected_mode": 12 * DECIMAL_TB,
                        "optimistic_incremental_append_lower_bound": DECIMAL_TB,
                    }
                },
            }]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            analysis_path = Path(temp_dir) / "analysis.json"
            output = Path(temp_dir) / "endurance.json"
            analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
            result = endurance_main([
                "--analysis-report", str(analysis_path),
                "--analysis-model", "org/model",
                "--analysis-hardware", "H100",
                "--analysis-traffic", "tiered",
                "--device-profile", str(MICRON_PROFILE),
                "--num-devices", "8",
                "--replays-per-day", "1",
                "--output-json", str(output),
            ])
            parsed = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(
            sum(device["trace_host_write_bytes"] for device in parsed["devices"]),
            8 * DECIMAL_TB,
        )

    def test_full_rewrite_analysis_uses_issued_not_attempted_bytes(self):
        analysis = {
            "summaries": [{
                "model": "org/model",
                "hardware": "H100",
                "tiered_policy": {"ssd_host_write_bytes": 8 * DECIMAL_TB},
                "ssd_swap": {
                    "host_write_bytes": {
                        "full_rewrite_all_attempts": 16 * DECIMAL_TB,
                        "full_rewrite_issued_under_selected_mode": 12 * DECIMAL_TB,
                        "optimistic_incremental_append_lower_bound": DECIMAL_TB,
                    }
                },
            }]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            analysis_path = temp / "analysis.json"
            output = temp / "endurance.json"
            analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
            result = endurance_main([
                "--analysis-report", str(analysis_path),
                "--analysis-model", "org/model",
                "--analysis-hardware", "H100",
                "--analysis-traffic", "full-rewrite",
                "--device-profile", str(MICRON_PROFILE),
                "--num-devices", "8",
                "--replays-per-day", "1",
                "--output-json", str(output),
            ])
            parsed = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(
            sum(device["trace_host_write_bytes"] for device in parsed["devices"]),
            12 * DECIMAL_TB,
        )


if __name__ == "__main__":
    unittest.main()

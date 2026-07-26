from __future__ import annotations

import json
import math
import unittest

from serving.core.ssd_hbf_runtime_energy import (
    BASELINE_SYSTEM_KEY,
    BYTES_PER_GIB,
    HOURS_PER_YEAR,
    PROPOSED_SYSTEM_KEY,
    RuntimePowerAssumptions,
    SSDHBFRuntimeEnergyError,
    account_one_gpu_one_hbf_runtime_energy,
    account_two_gpu_runtime_energy,
    evaluate_ssd_hbf_runtime_tco,
    project_five_year_runtime_tco,
)


HORIZON_NS = 1_000_000_000


def _resource(
        *, busy_ns: int = 0, byte_count: int = 0,
) -> dict[str, int]:
    return {
        "available_ns": busy_ns,
        "busy_ns": busy_ns,
        "reservation_count": int(busy_ns > 0 or byte_count > 0),
        "reservation_bytes": byte_count,
    }


def _calendar(
        resources: dict[str, dict[str, int]],
) -> dict[str, object]:
    return {
        "retain_reservations": False,
        "retained_reservation_count": 0,
        "resources": resources,
        "namespace_kinds": [],
    }


def _baseline_node(node_id: int) -> dict[str, object]:
    return {
        "mode": "finite_hbm_p4d4_tiering",
        "node_id": node_id,
        "pool": {
            "hardware": {
                "gpu_count": 8,
                "prefill_gpu_count": 4,
                "decode_gpu_count": 4,
                "cpu_memory_capacity_bytes": 512_000_000_000,
                "ssd_device_count": 8,
            },
            "metrics": {
                "p_modeled_ns": 100_000_000,
                "d_modeled_ns": 200_000_000,
            },
        },
        "lifecycle": {
            "metrics": {"transfer_bytes": 999_999},
        },
    }


def _baseline_calendar(node_id: int) -> dict[str, object]:
    prefix = f"gpu-node-{node_id}"
    return _calendar({
        f"{prefix}-p-pcie-rank-0": _resource(byte_count=25),
        f"{prefix}-p-pcie-rank-1": _resource(byte_count=25),
        f"{prefix}-p-pcie-rank-2": _resource(byte_count=25),
        f"{prefix}-p-pcie-rank-3": _resource(byte_count=25),
        f"{prefix}-pcie-root-0": _resource(
            busy_ns=20_000_000, byte_count=100),
        f"{prefix}-cpu-dram": _resource(
            busy_ns=80_000_000, byte_count=200),
        f"{prefix}-ssd-read": _resource(
            busy_ns=100_000_000, byte_count=40),
        f"{prefix}-ssd-write": _resource(
            busy_ns=50_000_000, byte_count=20),
        f"{prefix}-pd-peer-rank-0": _resource(byte_count=30),
        f"{prefix}-pd-fabric": _resource(
            busy_ns=20_000_000, byte_count=30),
    })


def _baseline_report() -> dict[str, object]:
    return {
        "mode": "dual_finite_hbm_p4d4_tiering",
        "current_ns": HORIZON_NS,
        "finished": True,
        "nodes": [_baseline_node(0), _baseline_node(1)],
    }


def _proposed_report() -> dict[str, object]:
    resources = {
        "gpu-node-0-p-pcie-rank-0": _resource(byte_count=25),
        "gpu-node-0-p-pcie-rank-1": _resource(byte_count=25),
        "gpu-node-0-p-pcie-rank-2": _resource(byte_count=25),
        "gpu-node-0-p-pcie-rank-3": _resource(byte_count=25),
        "gpu-node-0-pcie-root-0": _resource(
            busy_ns=20_000_000, byte_count=100),
        "gpu-node-0-cpu-dram": _resource(
            busy_ns=100_000_000, byte_count=200),
        "gpu-node-0-ssd-read": _resource(
            busy_ns=50_000_000, byte_count=40),
        "gpu-node-0-ssd-write": _resource(
            busy_ns=20_000_000, byte_count=20),
        "gpu-node-0-pd-peer-rank-0": _resource(byte_count=30),
        "gpu-node-0-pd-fabric": _resource(
            busy_ns=10_000_000, byte_count=30),
        "hbf-pcie-root-0": _resource(
            busy_ns=20_000_000, byte_count=80),
        "hbf-pcie-root-1": _resource(
            busy_ns=20_000_000, byte_count=80),
        "hbf-group-0-fabric": _resource(
            busy_ns=50_000_000, byte_count=50),
        "hbf-group-1-fabric": _resource(
            busy_ns=50_000_000, byte_count=50),
        "rdma-network": _resource(
            busy_ns=40_000_000, byte_count=160),
        "gpu-node-0-rdma-nic": _resource(
            busy_ns=40_000_000, byte_count=160),
    }
    for card_id in range(8):
        resources[f"hbf-card-{card_id}-pcie"] = _resource(
            byte_count=20)
        resources[f"hbf-card-{card_id}-media"] = _resource(
            busy_ns=100_000_000, byte_count=100)
        resources[f"hbf-card-{card_id}-lpddr"] = _resource(
            busy_ns=10_000_000, byte_count=10)
    calendar = _calendar(resources)
    hbf_hardware = {
        "card_count": 8,
        "lpddr_capacity_bytes_per_card": 64 * BYTES_PER_GIB,
    }
    return {
        "mode": "ssd_staged_gpu_hbf_agentic_system",
        "current_ns": HORIZON_NS,
        "finished": True,
        "node": {
            "calendar": calendar,
            "gpu_node": {
                "pool": {
                    "hardware": {
                        "gpu_count": 8,
                        "prefill_gpu_count": 4,
                        "decode_gpu_count": 4,
                        "cpu_memory_capacity_bytes": 512_000_000_000,
                        "ssd_device_count": 8,
                    },
                    "metrics": {
                        "p_modeled_ns": 100_000_000,
                        "d_modeled_ns": 150_000_000,
                    },
                },
            },
            "hbf_pool": {
                "hardware": hbf_hardware,
                "layout": {
                    "key": "tp4",
                    "tp_size": 4,
                    "replicas": 2,
                },
                "metrics": {
                    "modeled_batch_ns": 200_000_000,
                    "hbf_read_bytes_per_rank": 160,
                },
            },
            "hbf_lifecycle": {
                "hardware": hbf_hardware,
                "hbf_write_accounting": {
                    "schema_version": 1,
                    "complete_for_endurance_projection": True,
                    "total_physical_write_bytes": 160,
                    "cards": [
                        {
                            "card_id": card_id,
                            "total_write_bytes": 20,
                        }
                        for card_id in range(8)
                    ],
                },
            },
        },
    }


class RuntimePowerAssumptionTests(unittest.TestCase):
    def test_central_hbf_power_matches_flashaccel_cli_ratio(self):
        assumptions = RuntimePowerAssumptions()

        self.assertAlmostEqual(
            assumptions.hbf_full_activity_power_ratio_to_h100,
            860.0 / 700.0,
        )
        self.assertAlmostEqual(
            assumptions.hbf_full_activity_power_ratio_to_h100,
            1.23,
            places=2,
        )
        self.assertEqual(assumptions.lifetime_years, 5.0)
        self.assertEqual(
            {source.source_key for source in assumptions.sources},
            {
                "nvidia_h100_sxm",
                "flashaccel_hbf",
                "micron_9550_pro_3_84tb",
                "repository_power_config",
            },
        )

    def test_non_five_year_projection_is_rejected(self):
        with self.assertRaisesRegex(
                SSDHBFRuntimeEnergyError, "five-year"):
            RuntimePowerAssumptions(lifetime_years=3.0)


class BaselineRuntimeEnergyTests(unittest.TestCase):
    def test_requires_exact_calendar_bytes_not_aggregate_transfer_metric(self):
        with self.assertRaisesRegex(
                SSDHBFRuntimeEnergyError, "calendar.report"):
            account_two_gpu_runtime_energy(_baseline_report())

    def test_component_exclusive_counters_ignore_rank_lane_duplicates(self):
        report = account_two_gpu_runtime_energy(
            _baseline_report(),
            baseline_calendar_reports=(
                _baseline_calendar(0),
                _baseline_calendar(1),
            ),
        )

        self.assertEqual(report.system_key, BASELINE_SYSTEM_KEY)
        self.assertEqual(report.horizon_ns, HORIZON_NS)
        self.assertEqual(
            report.component("h100_gpu_hbm_cards").active_device_ns,
            2_400_000_000,
        )
        self.assertEqual(
            report.component("pcie_root_data_path").transfer_bytes,
            200,
        )
        self.assertEqual(
            report.component("host_dram").transfer_bytes,
            400,
        )
        self.assertEqual(
            report.component("gpu_intraserver_fabric").transfer_bytes,
            60,
        )
        ssd = report.component("local_nvme_ssd")
        self.assertEqual(ssd.read_bytes, 80)
        self.assertEqual(ssd.write_bytes, 40)
        self.assertEqual(
            ssd.active_device_ns,
            2_400_000_000,
        )
        self.assertEqual(
            report.component("external_network_nics").transfer_bytes,
            0,
        )
        self.assertGreater(report.total_it_energy_j, 0)
        self.assertAlmostEqual(
            report.average_it_power_w,
            report.total_it_energy_j,
        )
        json.dumps(report.to_json_dict(), allow_nan=False)

    def test_current_baseline_report_embeds_resource_calendars(self):
        raw = _baseline_report()
        raw["resource_calendars"] = [
            _baseline_calendar(0),
            _baseline_calendar(1),
        ]

        report = account_two_gpu_runtime_energy(raw)

        self.assertEqual(
            report.input_summary["pcie_root_transfer_bytes"],
            200,
        )


class ProposedRuntimeEnergyTests(unittest.TestCase):
    def test_hbf_uses_card_time_and_canonical_physical_bytes(self):
        report = account_one_gpu_one_hbf_runtime_energy(
            _proposed_report())

        self.assertEqual(report.system_key, PROPOSED_SYSTEM_KEY)
        self.assertEqual(
            report.component("h100_gpu_hbm_cards").active_device_ns,
            1_000_000_000,
        )
        self.assertEqual(
            report.component("hbf_gpu_logic").active_device_ns,
            800_000_000,
        )
        media = report.component("hbf_media_controller")
        self.assertEqual(media.active_device_ns, 800_000_000)
        self.assertEqual(media.transfer_bytes, 800)
        self.assertEqual(media.write_bytes, 160)
        self.assertEqual(media.read_bytes, 640)
        self.assertEqual(
            report.component("hbf_lpddr").transfer_bytes,
            80,
        )
        self.assertEqual(
            report.component("pcie_root_data_path").transfer_bytes,
            100,
        )
        self.assertEqual(
            report.component("hbf_intraserver_fabric").transfer_bytes,
            260,
        )
        self.assertEqual(
            report.component("external_network_nics").transfer_bytes,
            320,
        )
        self.assertEqual(
            report.component("external_network_nics").physical_quantity,
            4,
        )
        self.assertEqual(
            report.component("external_network_fabric").transfer_bytes,
            160,
        )
        self.assertEqual(
            report.component("hbf_intraserver_fabric").active_device_ns,
            120_000_000,
        )
        self.assertEqual(
            report.input_summary["hbf_media_write_bytes"],
            160,
        )

    def test_hbf_write_ledger_cannot_exceed_media_counter(self):
        raw = _proposed_report()
        raw["node"]["hbf_lifecycle"]["hbf_write_accounting"][  # type: ignore[index]
            "total_physical_write_bytes"
        ] = 801

        with self.assertRaisesRegex(
                SSDHBFRuntimeEnergyError, "write ledger"):
            account_one_gpu_one_hbf_runtime_energy(raw)


class RuntimeTCOTests(unittest.TestCase):
    def setUp(self):
        self.baseline = account_two_gpu_runtime_energy(
            _baseline_report(),
            baseline_calendar_reports=(
                _baseline_calendar(0),
                _baseline_calendar(1),
            ),
        )
        self.proposed = account_one_gpu_one_hbf_runtime_energy(
            _proposed_report())

    def test_trace_average_power_projects_without_utilization_multiplier(self):
        projection = project_five_year_runtime_tco(
            self.baseline,
            capex_usd=500_000.0,
            replaced_static_electricity_opex_usd=9_999_999.0,
        )
        expected_it_kwh = (
            self.baseline.average_it_power_w
            * 5.0
            * HOURS_PER_YEAR
            / 1000.0
        )
        expected_electricity = expected_it_kwh * 1.20 * 0.10

        self.assertAlmostEqual(
            projection.five_year_it_energy_kwh,
            expected_it_kwh,
        )
        self.assertAlmostEqual(
            projection.five_year_runtime_electricity_opex_usd,
            expected_electricity,
        )
        self.assertAlmostEqual(
            projection.five_year_tco_usd,
            500_000.0 + expected_electricity,
        )
        self.assertEqual(
            projection.replaced_static_electricity_opex_usd,
            9_999_999.0,
        )
        self.assertNotAlmostEqual(
            projection.five_year_tco_usd,
            500_000.0 + 9_999_999.0 + expected_electricity,
        )

    def test_paired_comparison_exposes_canonical_power_energy_tco_fields(self):
        comparison = evaluate_ssd_hbf_runtime_tco(
            baseline_system_report=_baseline_report(),
            proposed_system_report=_proposed_report(),
            baseline_calendar_reports=(
                _baseline_calendar(0),
                _baseline_calendar(1),
            ),
            baseline_capex_usd=500_000.0,
            proposed_capex_usd=450_000.0,
            baseline_static_electricity_opex_usd=50_000.0,
            proposed_static_electricity_opex_usd=60_000.0,
        )

        self.assertTrue(math.isfinite(
            comparison.proposed_average_it_power_ratio_to_baseline))
        self.assertAlmostEqual(
            comparison.proposed_five_year_it_energy_ratio_to_baseline,
            comparison.proposed_average_it_power_ratio_to_baseline,
        )
        self.assertAlmostEqual(
            comparison.incremental_average_it_power_w,
            (
                comparison.proposed.trace_average_it_power_w
                - comparison.baseline.trace_average_it_power_w
            ),
        )
        self.assertEqual(
            comparison.baseline.replaced_static_electricity_opex_usd,
            50_000.0,
        )
        self.assertEqual(
            comparison.proposed.replaced_static_electricity_opex_usd,
            60_000.0,
        )
        json.dumps(comparison.to_json_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()

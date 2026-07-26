from dataclasses import replace
import json
import math
import unittest

from serving.core.hbf_comparison_tco import (
    EvaluationAssumptions,
    HardwareAnchors,
)
from serving.core.hbf_design_tco import (
    ActiveMemorySpec,
    lpddr_active_memory,
    proposed_hbf_design_cost,
)
from serving.core.ssd_hbf_tco import (
    HBFServerLayout,
    SSDHBFTCOError,
    TwoGPUOneHBFComparisonTopology,
    evaluate_ssd_hbf_tco,
    one_gpu_one_hbf_cost,
    two_gpu_local_ssd_baseline_cost,
)


class SSDHBFTopologyTests(unittest.TestCase):
    def test_layouts_change_replicas_not_physical_counts(self):
        tp4 = TwoGPUOneHBFComparisonTopology.for_layout("tp4x2")
        tp8 = TwoGPUOneHBFComparisonTopology.for_layout("tp8")

        self.assertEqual(tp4.hbf_layout.tensor_parallel_size, 4)
        self.assertEqual(
            tp4.hbf_layout.independent_serving_replicas, 2)
        self.assertEqual(tp8.hbf_layout.tensor_parallel_size, 8)
        self.assertEqual(
            tp8.hbf_layout.independent_serving_replicas, 1)
        self.assertEqual(tp4.baseline, tp8.baseline)
        self.assertEqual(tp4.proposed, tp8.proposed)
        self.assertEqual(tp4.baseline.gpu_hosts, 2)
        self.assertEqual(tp4.baseline.h100_cards, 16)
        self.assertEqual(tp4.baseline.local_ssd_devices, 16)
        self.assertEqual(tp4.proposed.gpu_hosts, 1)
        self.assertEqual(tp4.proposed.hbf_hosts, 1)
        self.assertEqual(tp4.proposed.h100_cards, 8)
        self.assertEqual(tp4.proposed.hbf_cards, 8)
        self.assertEqual(tp4.proposed.local_ssd_devices, 8)
        self.assertIn("do not change host", tp4.count_semantics)

    def test_tp4_aliases_are_canonicalized_and_invalid_layout_rejected(self):
        for key in ("tp4x2", "tp4*2", "tp4"):
            with self.subTest(key=key):
                self.assertEqual(
                    HBFServerLayout.for_key(key).key,
                    "tp4x2",
                )
        with self.assertRaisesRegex(
                SSDHBFTCOError, "layout key"):
            HBFServerLayout.for_key("dp8")


class SSDHBFComponentCostTests(unittest.TestCase):
    def test_baseline_is_exactly_two_gpu_hosts_with_sixteen_local_ssds(self):
        cost = two_gpu_local_ssd_baseline_cost()

        self.assertEqual(cost.counts.cpu_hosts, 2)
        self.assertEqual(cost.counts.gpu_hosts, 2)
        self.assertEqual(cost.counts.hbf_hosts, 0)
        self.assertEqual(cost.counts.h100_cards, 16)
        self.assertEqual(cost.counts.hbf_cards, 0)
        self.assertEqual(cost.counts.local_ssd_devices, 16)
        self.assertEqual(
            cost.component("gpu_cpu_host_base").quantity, 2)
        self.assertEqual(
            cost.component("h100_gpu_logic").quantity, 16)
        self.assertEqual(
            cost.component("h100_hbm_stack").quantity, 16)
        self.assertEqual(
            cost.component("gpu_local_nvme_ssd").quantity, 16)

    def test_proposed_replaces_one_gpu_ssd_host_with_one_hbf_host(self):
        baseline = two_gpu_local_ssd_baseline_cost()
        proposed = one_gpu_one_hbf_cost(
            hbf_layout="tp4x2",
            active_memory=lpddr_active_memory(),
        )

        baseline_components = {
            line.component_key: line for line in baseline.bom
        }
        proposed_components = {
            line.component_key: line for line in proposed.bom
        }
        for key, line in baseline_components.items():
            with self.subTest(component=key):
                proposed_line = proposed_components[key]
                expected_ratio = (
                    1.0
                    if key == "network_fabric"
                    else 2.0
                )
                self.assertEqual(
                    line.quantity,
                    expected_ratio * proposed_line.quantity,
                )
                self.assertEqual(
                    line.unit_capex_usd,
                    proposed_line.unit_capex_usd,
                )
                self.assertEqual(
                    line.unit_it_power_w,
                    proposed_line.unit_it_power_w,
                )
        self.assertEqual(proposed.counts.gpu_hosts, 1)
        self.assertEqual(proposed.counts.hbf_hosts, 1)
        self.assertEqual(proposed.counts.h100_cards, 8)
        self.assertEqual(proposed.counts.hbf_cards, 8)
        self.assertEqual(proposed.counts.local_ssd_devices, 8)
        self.assertEqual(
            proposed.component("hbf_cpu_host_base").quantity, 1)
        self.assertEqual(
            proposed.component("hbf_gpu_logic").quantity, 8)
        self.assertEqual(
            proposed.component(
                "hbf_media_controller_subsystem").quantity,
            8,
        )
        self.assertEqual(
            proposed.component("hbf_host_rdma_nic").quantity, 1)

        additions = tuple(
            line
            for line in proposed.bom
            if line.component_key not in baseline_components
        )
        removed_gpu_host_capex = math.fsum(
            baseline_components[key].capex_usd
            - proposed_components[key].capex_usd
            for key in baseline_components
        )
        removed_gpu_host_power = math.fsum(
            baseline_components[key].it_power_w
            - proposed_components[key].it_power_w
            for key in baseline_components
        )
        self.assertAlmostEqual(
            proposed.capex_usd - baseline.capex_usd,
            (
                math.fsum(line.capex_usd for line in additions)
                - removed_gpu_host_capex
            ),
        )
        self.assertAlmostEqual(
            proposed.it_power_w - baseline.it_power_w,
            (
                math.fsum(line.it_power_w for line in additions)
                - removed_gpu_host_power
            ),
        )

    def test_baseline_and_proposal_use_their_network_price_anchors(self):
        anchors = HardwareAnchors(
            baseline_nic_capex_usd=111.0,
            baseline_nic_power_w=11.0,
            baseline_fabric_capex_usd=222.0,
            baseline_fabric_power_w=22.0,
            rdma_nic_capex_usd=333.0,
            rdma_nic_power_w=33.0,
            rdma_fabric_capex_usd=444.0,
            rdma_fabric_power_w=44.0,
        )
        baseline = two_gpu_local_ssd_baseline_cost(
            anchors=anchors)
        proposed = one_gpu_one_hbf_cost(
            hbf_layout="tp4x2",
            active_memory=lpddr_active_memory(anchors=anchors),
            anchors=anchors,
        )

        self.assertEqual(
            baseline.component(
                "gpu_host_network_nic").unit_capex_usd,
            111.0,
        )
        self.assertEqual(
            baseline.component(
                "network_fabric").unit_it_power_w,
            22.0,
        )
        self.assertEqual(
            proposed.component(
                "gpu_host_network_nic").unit_capex_usd,
            333.0,
        )
        self.assertEqual(
            proposed.component(
                "network_fabric").unit_it_power_w,
            44.0,
        )

    def test_cost_records_reject_crossed_physical_topologies(self):
        baseline = two_gpu_local_ssd_baseline_cost()
        proposed = one_gpu_one_hbf_cost(
            hbf_layout="tp4x2",
            active_memory=lpddr_active_memory(),
        )

        with self.assertRaisesRegex(
                SSDHBFTCOError, "exactly two"):
            replace(baseline, counts=proposed.counts)
        with self.assertRaisesRegex(
                SSDHBFTCOError, "exactly one GPU"):
            replace(proposed, counts=baseline.counts)

    def test_tp4x2_and_tp8_have_identical_bom_and_tco(self):
        memory = lpddr_active_memory()
        tp4 = one_gpu_one_hbf_cost(
            hbf_layout="tp4x2", active_memory=memory)
        tp8 = one_gpu_one_hbf_cost(
            hbf_layout="tp8", active_memory=memory)

        self.assertEqual(tp4.bom, tp8.bom)
        self.assertEqual(tp4.capex_usd, tp8.capex_usd)
        self.assertEqual(tp4.it_power_w, tp8.it_power_w)
        self.assertEqual(
            tp4.five_year_tco_usd,
            tp8.five_year_tco_usd,
        )
        self.assertNotEqual(
            tp4.hbf_layout.independent_serving_replicas,
            tp8.hbf_layout.independent_serving_replicas,
        )

    def test_active_memory_kind_capacity_bandwidth_and_cost_are_explicit(self):
        memory = ActiveMemorySpec(
            kind="sram_like",
            capacity_gib_per_card=12.0,
            bandwidth_gbps_per_card=8_000.0,
            capex_usd_per_gib=2_000.0,
            power_w_per_gib=20.0,
            assumption="Counterfactual package SRAM-like memory.",
        )
        cost = one_gpu_one_hbf_cost(
            hbf_layout="tp8", active_memory=memory)
        line = cost.component("hbf_card_active_memory")

        self.assertEqual(cost.active_memory.kind, "sram_like")
        self.assertEqual(
            cost.active_memory.bandwidth_gbps_per_card, 8_000.0)
        self.assertEqual(line.quantity, 8 * 12)
        self.assertEqual(line.unit_capex_usd, 2_000.0)
        self.assertEqual(line.unit_it_power_w, 20.0)
        self.assertIn("8000 GB/s/card", line.assumption)

    def test_proposed_reuses_existing_one_hbf_component_assumptions(self):
        anchors = HardwareAnchors()
        evaluation = EvaluationAssumptions()
        memory = lpddr_active_memory(anchors=anchors)
        existing = proposed_hbf_design_cost(
            hbf_host_count=1,
            active_memory=memory,
            anchors=anchors,
            evaluation=evaluation,
        )
        with_ssd = one_gpu_one_hbf_cost(
            hbf_layout="tp4x2",
            active_memory=memory,
            anchors=anchors,
            evaluation=evaluation,
        )
        ssd_capex = (
            8 * anchors.nvme_ssd_capex_usd_per_device)
        ssd_power = (
            8 * anchors.nvme_ssd_power_w_per_device)

        self.assertAlmostEqual(
            with_ssd.capex_usd,
            existing.capex_usd + ssd_capex,
        )
        self.assertAlmostEqual(
            with_ssd.it_power_w,
            existing.it_power_w + ssd_power,
        )
        expected_ssd_energy_opex = (
            ssd_power
            * evaluation.pue
            * evaluation.lifetime_powered_equivalent_full_load_hours
            / 1_000.0
            * evaluation.electricity_usd_per_kwh
        )
        self.assertAlmostEqual(
            with_ssd.five_year_tco_usd,
            (
                existing.lifetime_tco_usd
                + ssd_capex
                + expected_ssd_energy_opex
            ),
        )

    def test_non_five_year_evaluation_is_rejected(self):
        with self.assertRaisesRegex(
                SSDHBFTCOError, "five-year"):
            two_gpu_local_ssd_baseline_cost(
                evaluation=EvaluationAssumptions(
                    lifetime_years=4.0),
            )

    def test_hbf_compute_uses_full_h100_gpu_logic_power_and_capex(self):
        anchors = HardwareAnchors()
        proposed = one_gpu_one_hbf_cost(
            hbf_layout="tp4x2",
            active_memory=lpddr_active_memory(anchors=anchors),
            anchors=anchors,
        )
        compute = proposed.component("hbf_gpu_logic")

        self.assertEqual(
            compute.unit_it_power_w,
            anchors.gpu_logic_power_w_per_card,
        )
        self.assertEqual(
            compute.unit_capex_usd,
            anchors.gpu_logic_capex_usd_per_card,
        )
        self.assertEqual(proposed.evaluation.lifetime_years, 5.0)


class SSDHBFEconomicsTests(unittest.TestCase):
    def test_delta_and_required_goodput_break_even_are_auditable(self):
        baseline_goodput = 100.0
        baseline = two_gpu_local_ssd_baseline_cost()
        proposed = one_gpu_one_hbf_cost(
            hbf_layout="tp8",
            active_memory=lpddr_active_memory(),
        )
        tco_ratio = (
            proposed.five_year_tco_usd
            / baseline.five_year_tco_usd
        )
        required_goodput = baseline_goodput * tco_ratio
        report = evaluate_ssd_hbf_tco(
            hbf_layout="tp8",
            active_memory=lpddr_active_memory(),
            baseline_slo_good_output_tokens_per_second=(
                baseline_goodput),
            proposed_slo_good_output_tokens_per_second=(
                required_goodput),
            oracle_slo_good_output_tokens_per_second=200.0,
        )

        self.assertAlmostEqual(
            report.proposed_tco_ratio_to_baseline,
            tco_ratio,
        )
        self.assertAlmostEqual(
            report.required_goodput_ratio_for_equal_token_cost,
            tco_ratio,
        )
        self.assertAlmostEqual(
            report.required_proposed_goodput_tokens_per_second,
            required_goodput,
        )
        self.assertTrue(
            report.proposed_meets_or_exceeds_goodput_break_even)
        self.assertAlmostEqual(
            report.cost_delta_proposed_minus_baseline
            .five_year_tco_usd,
            proposed.five_year_tco_usd
            - baseline.five_year_tco_usd,
        )
        self.assertAlmostEqual(
            report.baseline_token_economics
            .tco_usd_per_million_slo_good_output_tokens,
            report.proposed_token_economics
            .tco_usd_per_million_slo_good_output_tokens,
        )
        power = report.power_energy_comparison
        self.assertEqual(power.lifetime_years, 5.0)
        self.assertEqual(
            power.incremental_it_power_w,
            proposed.it_power_w - baseline.it_power_w,
        )
        self.assertAlmostEqual(
            power.proposed_it_power_ratio_to_baseline,
            proposed.it_power_w / baseline.it_power_w,
        )
        self.assertAlmostEqual(
            power.proposed_facility_energy_ratio_to_baseline,
            proposed.five_year_facility_energy_kwh
            / baseline.five_year_facility_energy_kwh,
        )

    def test_oracle_is_performance_only_and_json_safe(self):
        report = evaluate_ssd_hbf_tco(
            hbf_layout="tp4x2",
            active_memory=lpddr_active_memory(),
            baseline_slo_good_output_tokens_per_second=100.0,
            proposed_slo_good_output_tokens_per_second=0.0,
            oracle_slo_good_output_tokens_per_second=300.0,
        )

        self.assertFalse(
            report.oracle_reference.physical_bom_available)
        self.assertFalse(
            report.oracle_reference.included_in_tco_comparison)
        self.assertIsNone(
            report.oracle_reference.five_year_tco_usd)
        self.assertIsNone(
            report.oracle_reference
            .tco_usd_per_million_slo_good_output_tokens)
        self.assertEqual(
            report.oracle_reference
            .slo_good_output_tokens_per_second,
            300.0,
        )
        self.assertIsNone(
            report.proposed_token_economics
            .tco_usd_per_million_slo_good_output_tokens)
        self.assertFalse(
            report.proposed_meets_or_exceeds_goodput_break_even)
        json.dumps(report.to_json_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()

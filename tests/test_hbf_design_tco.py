import json
import unittest

from serving.core.hbf_comparison_tco import (
    EvaluationAssumptions,
    HardwareAnchors,
    SensitivityPoint,
    proposed_hbf_cost,
    tiering_baseline_cost,
)
from serving.core.hbf_design_tco import (
    CENTRAL_SENSITIVITY_POINT,
    ActiveMemorySpec,
    HBFDesignTCOError,
    HBFDesignTopology,
    evaluate_hbf_design_tco,
    lpddr_active_memory,
    proposed_hbf_design_cost,
)


class HBFDesignInputTests(unittest.TestCase):
    def test_hbf_host_count_must_be_positive_integer(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                        HBFDesignTCOError, "positive integer"):
                    HBFDesignTopology(hbf_host_count=value)

    def test_active_memory_requires_explicit_valid_assumptions(self):
        with self.assertRaisesRegex(
                HBFDesignTCOError, "kind must be one of"):
            ActiveMemorySpec(
                kind="hbm",
                capacity_gib_per_card=16.0,
                bandwidth_gbps_per_card=1_000.0,
                capex_usd_per_gib=10.0,
                power_w_per_gib=1.0,
            )
        with self.assertRaisesRegex(HBFDesignTCOError, "must be positive"):
            ActiveMemorySpec(
                kind="sram_like",
                capacity_gib_per_card=0.0,
                bandwidth_gbps_per_card=8_000.0,
                capex_usd_per_gib=2_000.0,
                power_w_per_gib=20.0,
            )
        with self.assertRaisesRegex(HBFDesignTCOError, "at least 0"):
            ActiveMemorySpec(
                kind="lpddr",
                capacity_gib_per_card=16.0,
                bandwidth_gbps_per_card=204.8,
                capex_usd_per_gib=-1.0,
                power_w_per_gib=0.08,
            )

    def test_central_sensitivity_point_is_explicit(self):
        self.assertEqual(
            CENTRAL_SENSITIVITY_POINT,
            SensitivityPoint(
                npu_logic_capex_ratio_to_gpu_logic=1.0,
                hbf_subsystem_capex_ratio_to_hbm_stack=0.5,
                npu_logic_power_ratio_to_gpu_logic=1.0,
                hbf_subsystem_power_ratio_to_hbm_stack=3.5,
            ),
        )


class HBFDesignCostTests(unittest.TestCase):
    def test_one_host_lpddr_matches_existing_central_proposal(self):
        anchors = HardwareAnchors()
        evaluation = EvaluationAssumptions()
        memory = lpddr_active_memory(anchors=anchors)
        exploratory = proposed_hbf_design_cost(
            hbf_host_count=1,
            active_memory=memory,
            sensitivity_point=CENTRAL_SENSITIVITY_POINT,
            anchors=anchors,
            evaluation=evaluation,
        )
        strict = proposed_hbf_cost(
            CENTRAL_SENSITIVITY_POINT,
            anchors=anchors,
            evaluation=evaluation,
        )

        self.assertAlmostEqual(exploratory.capex_usd, strict.capex_usd)
        self.assertAlmostEqual(exploratory.it_power_w, strict.it_power_w)
        self.assertAlmostEqual(
            exploratory.lifetime_tco_usd,
            strict.lifetime_tco_usd,
        )
        self.assertEqual(exploratory.topology.h100_card_count, 8)
        self.assertEqual(exploratory.topology.hbf_card_count, 8)

    def test_multiple_hbf_hosts_scale_host_card_and_memory_bom(self):
        memory = lpddr_active_memory(capacity_gib_per_card=32.0)
        one = proposed_hbf_design_cost(
            hbf_host_count=1,
            active_memory=memory,
        )
        three = proposed_hbf_design_cost(
            hbf_host_count=3,
            active_memory=memory,
        )

        self.assertEqual(
            three.component("cpu_host_base").quantity, 4)
        self.assertEqual(
            three.component("hbf_gpu_logic").quantity, 24)
        self.assertEqual(
            three.component("hbf_gpu_intraserver_fabric").quantity, 3)
        self.assertEqual(
            three.component("hbf_card_active_memory").quantity,
            24 * 32,
        )
        self.assertEqual(
            three.component("rdma_network_nic").quantity, 8)
        self.assertGreater(three.capex_usd, one.capex_usd)
        self.assertGreater(three.lifetime_tco_usd, one.lifetime_tco_usd)

    def test_sram_like_cost_power_and_bandwidth_are_not_inherited(self):
        memory = ActiveMemorySpec(
            kind="sram_like",
            capacity_gib_per_card=8.0,
            bandwidth_gbps_per_card=8_000.0,
            capex_usd_per_gib=2_000.0,
            power_w_per_gib=20.0,
            assumption="Counterfactual package SRAM-like active memory.",
        )
        cost = proposed_hbf_design_cost(
            hbf_host_count=2,
            active_memory=memory,
        )
        line = cost.component("hbf_card_active_memory")

        self.assertEqual(line.quantity, 16 * 8)
        self.assertEqual(line.unit_capex_usd, 2_000.0)
        self.assertEqual(line.unit_it_power_w, 20.0)
        self.assertIn("8000 GB/s/card", line.assumption)
        self.assertEqual(
            cost.active_memory.bandwidth_gbps_per_card, 8_000.0)


class HBFDesignEvaluationTests(unittest.TestCase):
    def test_report_exposes_break_even_and_token_cost(self):
        evaluation = EvaluationAssumptions()
        report = evaluate_hbf_design_tco(
            hbf_host_count=2,
            active_memory=lpddr_active_memory(),
            baseline_slo_good_output_tokens_per_second=100.0,
            proposed_slo_good_output_tokens_per_second=120.0,
            oracle_slo_good_output_tokens_per_second=150.0,
            evaluation=evaluation,
        )
        expected_ratio = (
            report.proposed_cost.lifetime_tco_usd
            / report.baseline_cost.lifetime_tco_usd
        )
        expected_proposed_dollars_per_million = (
            report.proposed_cost.lifetime_tco_usd
            / (120.0 * evaluation.lifetime_loaded_seconds)
            * 1_000_000.0
        )

        self.assertAlmostEqual(
            report.goodput_break_even_ratio_vs_baseline,
            expected_ratio,
        )
        self.assertAlmostEqual(
            report.break_even_proposed_goodput_tokens_per_second,
            100.0 * expected_ratio,
        )
        self.assertAlmostEqual(
            report.proposed_token_cost
            .dollars_per_million_slo_good_output_tokens,
            expected_proposed_dollars_per_million,
        )
        self.assertEqual(
            report.oracle_reference
            .slo_good_output_tokens_per_second,
            150.0,
        )
        self.assertIsNone(report.oracle_reference.lifetime_tco_usd)
        self.assertIsNone(
            report.oracle_reference
            .dollars_per_million_slo_good_output_tokens)
        json.dumps(report.to_json_dict(), allow_nan=False)

    def test_zero_proposed_goodput_has_no_dollars_per_million(self):
        report = evaluate_hbf_design_tco(
            hbf_host_count=1,
            active_memory=lpddr_active_memory(),
            baseline_slo_good_output_tokens_per_second=100.0,
            proposed_slo_good_output_tokens_per_second=0.0,
        )

        self.assertIsNone(
            report.proposed_token_cost
            .dollars_per_million_slo_good_output_tokens)
        self.assertFalse(
            report.proposed_meets_or_exceeds_token_value_break_even)

    def test_fixed_baseline_is_the_existing_tiering_bom(self):
        anchors = HardwareAnchors()
        evaluation = EvaluationAssumptions()
        report = evaluate_hbf_design_tco(
            hbf_host_count=1,
            active_memory=lpddr_active_memory(anchors=anchors),
            baseline_slo_good_output_tokens_per_second=100.0,
            proposed_slo_good_output_tokens_per_second=100.0,
            anchors=anchors,
            evaluation=evaluation,
        )
        expected = tiering_baseline_cost(
            anchors=anchors,
            evaluation=evaluation,
        )

        self.assertEqual(report.baseline_cost, expected)


if __name__ == "__main__":
    unittest.main()

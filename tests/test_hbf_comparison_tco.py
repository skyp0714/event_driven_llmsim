from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest

from serving.core.hbf_comparison_tco import (
    ComparisonPerformanceProvenance,
    BYTES_PER_GIB,
    DEFAULT_HBF_HARDWARE_VARIANT,
    ECONOMIC_SYSTEM_KEYS,
    GoodputResultProvenance,
    HBFHardwareVariant,
    ORACLE_SYSTEM_KEY,
    P4D4_CPU_MEMORY_BYTES_PER_HOST,
    PINNED_GPU_CONFIG_SHA256,
    PINNED_HBF_CONFIG_SHA256,
    PINNED_HBF_WIDE_LPDDR_CONFIG_SHA256,
    OUTPUT_TOKEN_GOODPUT_DEFINITION,
    PROPOSED_SYSTEM_KEY,
    TIERING_SYSTEM_KEY,
    DeploymentTopology,
    EvaluationAssumptions,
    HBFComparisonTCOError,
    HardwareAnchors,
    SensitivityAxes,
    SensitivityPoint,
    evaluate_tco_sensitivity,
    proposed_hbf_cost,
    sensitivity_points,
    tiering_baseline_cost,
    token_economics,
    WIDE_LPDDR_HBF_HARDWARE_VARIANT,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def result_provenance(
        system_key, goodput, *, schedule_sha256="a" * 64,
        measurement_sha256="b" * 64, session_rate=2.0,
        scenario_id="balanced-causal-prefix-v1",
        cohort_id="measurement-epochs-2-through-5"):
    return GoodputResultProvenance(
        system_key=system_key,
        slo_good_output_tokens_per_second=goodput,
        offered_session_rate_per_second=session_rate,
        scenario_id=scenario_id,
        cohort_id=cohort_id,
        schedule_sha256=schedule_sha256,
        schedule_hash_semantics=(
            "ordered_paired_seed_schedule_set_manifest"),
        measurement_cohort_sha256=measurement_sha256,
        result_goodput_origin=(
            f"results/comparison/{system_key}/aggregate.json"),
        result_manifest_sha256=hashlib.sha256(
            f"{system_key}-result-manifest".encode("utf-8")
        ).hexdigest(),
        result_schema_revision="hbf-comparison-cell-v1/aggregate-v1",
        simulator_code_revision="0123456789abcdef" * 2 + "01234567",
        metric_scope="all",
        metric_json_path=(
            "summary.request_kind_summaries.all."
            "offered_load_normalized_output_token_goodput.value"
        ),
        metric_definition=OUTPUT_TOKEN_GOODPUT_DEFINITION,
        aggregation_method="arithmetic_mean_across_paired_seeds",
        seed_count=8,
        confidence_interval_method="student_t_95",
        confidence_interval_lower_tokens_per_second=goodput * 0.9,
        confidence_interval_upper_tokens_per_second=goodput * 1.1,
    )


def performance_provenance(
        tiering_goodput, proposed_goodput, oracle_goodput=None, *,
        hbf_hardware_variant=DEFAULT_HBF_HARDWARE_VARIANT,
        hbf_layout_key="hbf_tp4"):
    return ComparisonPerformanceProvenance(
        selected_tiering_policy_key="cpu_ssd",
        hbf_layout_key=hbf_layout_key,
        hbf_policy_key=(
            "first_gpu__migration_inflight_resume_gpu__"
            "hbf_ready_resume_hbf__turn_boundary_lpddr_v1"),
        hbf_policy_contract_sha256="c" * 64,
        gpu_config_sha256=PINNED_GPU_CONFIG_SHA256,
        hbf_hardware_variant=hbf_hardware_variant,
        first_ttft_slo_ns=30_000_000_000,
        resume_ttft_slo_ns=30_000_000_000,
        tpot_slo_ns=300_000_000,
        operating_point_mode="matched_single_operating_point",
        rate_selection_semantics=(
            "matched offered-rate cell; not a sustainable-rate ceiling"),
        maximum_slo_sustainable_claim=False,
        tiering_result=result_provenance(
            TIERING_SYSTEM_KEY, tiering_goodput),
        proposed_result=result_provenance(
            PROPOSED_SYSTEM_KEY, proposed_goodput),
        oracle_result=(
            None
            if oracle_goodput is None
            else result_provenance(ORACLE_SYSTEM_KEY, oracle_goodput)
        ),
    )


def evaluate_with_provenance(values, **kwargs):
    kwargs["performance_provenance"] = performance_provenance(
        values[TIERING_SYSTEM_KEY],
        values[PROPOSED_SYSTEM_KEY],
        values.get(ORACLE_SYSTEM_KEY),
    )
    return evaluate_tco_sensitivity(values, **kwargs)


def central_point():
    return SensitivityPoint(
        npu_logic_capex_ratio_to_gpu_logic=1.00,
        hbf_subsystem_capex_ratio_to_hbm_stack=0.50,
        npu_logic_power_ratio_to_gpu_logic=1.00,
        hbf_subsystem_power_ratio_to_hbm_stack=3.50,
    )


class TopologyAndBOMTests(unittest.TestCase):
    def test_default_topology_matches_the_two_physical_systems(self):
        topology = DeploymentTopology()
        self.assertEqual(topology.tiering_cpu_hosts, 2)
        self.assertEqual(topology.tiering_h100_cards, 16)
        self.assertEqual(topology.tiering_ssd_devices, 16)
        self.assertEqual(
            topology.tiering_gpu_intraserver_fabric_units, 2)
        self.assertEqual(topology.proposed_cpu_hosts, 2)
        self.assertEqual(topology.proposed_h100_cards, 8)
        self.assertEqual(topology.proposed_hbf_npu_cards, 8)
        self.assertEqual(topology.proposed_lpddr_gib, 512)
        self.assertEqual(
            topology.proposed_gpu_intraserver_fabric_units, 1)
        self.assertEqual(
            topology.proposed_hbf_intraserver_fabric_units, 1)
        self.assertEqual(
            topology.host_dram_gib_per_host * BYTES_PER_GIB,
            P4D4_CPU_MEMORY_BYTES_PER_HOST,
        )

    def test_h100_anchor_is_decomposed_without_double_counting(self):
        anchors = HardwareAnchors()
        self.assertAlmostEqual(
            anchors.gpu_logic_capex_usd_per_card
            + anchors.hbm_stack_capex_usd_per_card,
            anchors.h100_card_capex_usd,
        )
        self.assertAlmostEqual(
            anchors.gpu_logic_power_w_per_card
            + anchors.hbm_stack_power_w_per_card,
            anchors.h100_card_power_w,
        )

    def test_hbm_credit_uses_manufacturing_share_of_purchase_price(self):
        anchors = HardwareAnchors()
        self.assertEqual(
            anchors.hbm_capex_accounting_basis,
            "manufacturing_cost_fraction_applied_to_purchase_price",
        )
        self.assertAlmostEqual(
            anchors.hbm_manufacturing_cost_fraction_of_h100_card,
            1_350.0 / 3_320.0,
        )
        self.assertAlmostEqual(
            anchors.hbm_stack_capex_usd_per_card,
            30_000.0 * (1_350.0 / 3_320.0),
        )
        self.assertAlmostEqual(
            anchors.hbm_capex_share_of_h100_purchase_price,
            1_350.0 / 3_320.0,
        )
        self.assertNotAlmostEqual(
            anchors.hbm_capex_share_of_h100_purchase_price,
            anchors.hbm_capex_fraction_of_h100_card,
        )
        self.assertTrue(
            anchors.hbm_component_cost_source_url.startswith("https://"))
        self.assertTrue(
            anchors.h100_purchase_price_source_url.startswith("https://"))
        self.assertTrue(
            anchors.h100_tdp_source_url.startswith("https://"))

        absolute = replace(
            anchors,
            hbm_capex_accounting_basis=(
                "absolute_avoided_purchase_credit"),
        )
        self.assertEqual(
            absolute.hbm_stack_capex_usd_per_card, 1_350.0)
        self.assertAlmostEqual(
            absolute.hbm_capex_share_of_h100_purchase_price, 0.045)

        legacy = replace(
            anchors,
            hbm_capex_accounting_basis=(
                "legacy_fraction_of_purchase_price"),
        )
        self.assertEqual(
            legacy.hbm_capex_share_of_h100_purchase_price, 0.30)
        self.assertAlmostEqual(
            legacy.gpu_logic_capex_usd_per_card
            + legacy.hbm_stack_capex_usd_per_card,
            legacy.h100_card_capex_usd,
        )

    def test_pinned_hardware_hashes_match_the_effective_configs(self):
        gpu_bytes = (
            REPO_ROOT / "configs/wakekv_hbf/p4d4_gpu_server.json"
        ).read_bytes()
        hbf_bytes = (
            REPO_ROOT / "configs/wakekv_hbf/full_model_8card_server.json"
        ).read_bytes()
        wide_hbf_bytes = (
            REPO_ROOT
            / "configs/wakekv_hbf/full_model_8card_server_wide_lpddr.json"
        ).read_bytes()
        self.assertEqual(
            hashlib.sha256(gpu_bytes).hexdigest(),
            PINNED_GPU_CONFIG_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(hbf_bytes).hexdigest(),
            PINNED_HBF_CONFIG_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(wide_hbf_bytes).hexdigest(),
            PINNED_HBF_WIDE_LPDDR_CONFIG_SHA256,
        )

    def test_baseline_bom_has_two_hosts_dram_ssds_and_network(self):
        cost = tiering_baseline_cost()
        topology = DeploymentTopology()
        self.assertEqual(cost.component("cpu_host_base").quantity, 2)
        self.assertEqual(
            cost.component("host_dram").quantity,
            2 * topology.host_dram_gib_per_host,
        )
        self.assertEqual(cost.component("h100_gpu_logic").quantity, 16)
        self.assertEqual(cost.component("h100_hbm_stack").quantity, 16)
        self.assertEqual(
            cost.component("gpu_intraserver_fabric").quantity, 2)
        self.assertGreater(
            cost.component("gpu_intraserver_fabric").capex_usd, 0)
        self.assertGreater(
            cost.component("gpu_intraserver_fabric").it_power_w, 0)
        self.assertEqual(cost.component("nvme_ssd_tier").quantity, 16)
        self.assertEqual(cost.component("baseline_network_nic").quantity, 2)
        self.assertEqual(
            cost.component("baseline_network_fabric").quantity, 1)

    def test_proposed_bom_has_both_hosts_lpddr_rdma_and_no_ssd(self):
        cost = proposed_hbf_cost(central_point())
        topology = DeploymentTopology()
        self.assertEqual(cost.component("cpu_host_base").quantity, 2)
        self.assertEqual(
            cost.component("host_dram").quantity,
            2 * topology.host_dram_gib_per_host,
        )
        self.assertEqual(cost.component("h100_gpu_logic").quantity, 8)
        self.assertEqual(
            cost.component("gpu_intraserver_fabric").quantity, 1)
        self.assertGreater(
            cost.component("gpu_intraserver_fabric").capex_usd, 0)
        self.assertGreater(
            cost.component("gpu_intraserver_fabric").it_power_w, 0)
        self.assertEqual(cost.component("hbf_npu_logic").quantity, 8)
        self.assertEqual(
            cost.component("hbf_media_controller_subsystem").quantity, 8)
        self.assertEqual(
            cost.component("hbf_npu_intraserver_fabric").quantity, 1)
        self.assertGreater(
            cost.component("hbf_npu_intraserver_fabric").capex_usd, 0)
        self.assertGreater(
            cost.component("hbf_npu_intraserver_fabric").it_power_w, 0)
        self.assertEqual(cost.component("hbf_card_lpddr").quantity, 512)
        self.assertGreater(cost.component("hbf_card_lpddr").capex_usd, 0)
        self.assertGreater(cost.component("hbf_card_lpddr").it_power_w, 0)
        self.assertEqual(cost.component("nvme_ssd_tier").quantity, 0)
        self.assertEqual(cost.component("rdma_network_nic").quantity, 4)
        self.assertEqual(cost.component("rdma_network_fabric").quantity, 1)

    def test_both_deployments_use_the_same_two_host_anchor(self):
        baseline = tiering_baseline_cost()
        proposed = proposed_hbf_cost(central_point())
        for component in ("cpu_host_base", "host_dram"):
            left = baseline.component(component)
            right = proposed.component(component)
            self.assertEqual(left.quantity, right.quantity)
            self.assertEqual(left.capex_usd, right.capex_usd)
            self.assertEqual(left.it_power_w, right.it_power_w)


class SensitivityTests(unittest.TestCase):
    def test_default_grid_is_full_cartesian_product(self):
        axes = SensitivityAxes()
        points = sensitivity_points(axes)
        self.assertEqual(
            axes.npu_logic_capex_ratios_to_gpu_logic, (1.0,))
        self.assertEqual(
            axes.npu_logic_power_ratios_to_gpu_logic, (1.0,))
        self.assertEqual(axes.cartesian_size, 9)
        self.assertEqual(len(points), 9)
        self.assertEqual(len({point.key for point in points}), 9)

    def test_sensitivity_keys_do_not_alias_nearby_custom_floats(self):
        left = SensitivityPoint(0.50000001, 0.5, 0.5, 3.5)
        right = SensitivityPoint(0.50000002, 0.5, 0.5, 3.5)
        self.assertNotEqual(left.key, right.key)

    def test_hbf_subsystem_is_cheaper_than_hbm_per_installed_byte(self):
        anchors = HardwareAnchors()
        for point in sensitivity_points():
            cost = proposed_hbf_cost(point, anchors)
            hbf = cost.component("hbf_media_controller_subsystem")
            self.assertLess(
                (
                    hbf.unit_capex_usd
                    / DEFAULT_HBF_HARDWARE_VARIANT
                    .hbf_capacity_ratio_to_hbm
                ),
                anchors.hbm_stack_capex_usd_per_card,
            )

    def test_central_hbf_card_power_matches_whole_card_anchor(self):
        anchors = HardwareAnchors()
        cost = proposed_hbf_cost(central_point(), anchors)
        media = cost.component("hbf_media_controller_subsystem")
        logic = cost.component("hbf_npu_logic")
        lpddr = cost.component("hbf_card_lpddr")
        self.assertEqual(media.unit_capex_usd, 4_500.0)
        self.assertEqual(media.unit_it_power_w, 300.0)
        whole_card_power = (
            logic.unit_it_power_w
            + media.unit_it_power_w
            + lpddr.it_power_w / logic.quantity
        )
        self.assertAlmostEqual(
            whole_card_power / anchors.h100_card_power_w,
            1.2358857142857143,
        )

    def test_hbf_power_sensitivity_changes_energy_opex_and_tco(self):
        low = proposed_hbf_cost(SensitivityPoint(
            0.5, 0.5, 0.5, 3.0))
        high = proposed_hbf_cost(SensitivityPoint(
            0.5, 0.5, 0.5, 4.0))
        self.assertEqual(low.capex_usd, high.capex_usd)
        self.assertLess(low.it_power_w, high.it_power_w)
        self.assertLess(
            low.lifetime_electricity_opex_usd,
            high.lifetime_electricity_opex_usd,
        )
        self.assertLess(low.lifetime_tco_usd, high.lifetime_tco_usd)

    def test_hbf_fabric_kind_and_wide_lpddr_change_explicit_bom_lines(self):
        pcie = DEFAULT_HBF_HARDWARE_VARIANT
        ualink_wide = HBFHardwareVariant(
            variant_key="ualink100_lpddr409p6",
            hbf_config_sha256="f" * 64,
            card_count=8,
            hbf_capacity_bytes_per_card=1_280_000_000_000,
            hbf_capacity_ratio_to_hbm=16.0,
            intra_fabric_kind="ualink",
            intra_fabric_bandwidth_gbps_per_card=100.0,
            intra_fabric_capex_multiplier=1.25,
            intra_fabric_power_multiplier=1.20,
            lpddr_reference_bandwidth_gbps_per_card=204.8,
            lpddr_effective_bandwidth_gbps_per_card=409.6,
            lpddr_capacity_gib_per_card=64.0,
            lpddr_bandwidth_multiplier=2.0,
            lpddr_capex_multiplier=1.5,
            lpddr_power_multiplier=1.75,
            rdma_bandwidth_gbps=100.0,
            rdma_one_way_latency_us=5.0,
            rdma_capex_multiplier=1.25,
            rdma_power_multiplier=1.20,
            cost_power_assumption=(
                "Analytical wide-LPDDR and UALink multipliers."
            ),
        )
        pcie_cost = proposed_hbf_cost(
            central_point(), hbf_hardware_variant=pcie)
        wide_cost = proposed_hbf_cost(
            central_point(), hbf_hardware_variant=ualink_wide)
        pcie_fabric = pcie_cost.component(
            "hbf_npu_intraserver_fabric")
        wide_fabric = wide_cost.component(
            "hbf_npu_intraserver_fabric")
        self.assertIn("PCIE", pcie_fabric.component_label)
        self.assertIn("UALINK", wide_fabric.component_label)
        self.assertGreater(
            wide_fabric.unit_capex_usd,
            pcie_fabric.unit_capex_usd,
        )
        self.assertGreater(
            wide_fabric.unit_it_power_w,
            pcie_fabric.unit_it_power_w,
        )
        pcie_lpddr = pcie_cost.component("hbf_card_lpddr")
        wide_lpddr = wide_cost.component("hbf_card_lpddr")
        self.assertAlmostEqual(
            wide_lpddr.unit_capex_usd,
            pcie_lpddr.unit_capex_usd * 1.5,
        )
        self.assertAlmostEqual(
            wide_lpddr.unit_it_power_w,
            pcie_lpddr.unit_it_power_w * 1.75,
        )
        self.assertIn("2x reference bandwidth", wide_lpddr.assumption)
        self.assertAlmostEqual(
            wide_cost.component("rdma_network_nic").unit_capex_usd,
            pcie_cost.component("rdma_network_nic").unit_capex_usd * 1.25,
        )
        self.assertAlmostEqual(
            wide_cost.component("rdma_network_nic").unit_it_power_w,
            pcie_cost.component("rdma_network_nic").unit_it_power_w * 1.20,
        )
        self.assertIn(
            "16x the H100 HBM anchor",
            wide_cost.component(
                "hbf_media_controller_subsystem").assumption,
        )

    def test_effective_hbf_variant_must_match_physical_bom_counts(self):
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "card count mismatches"):
            proposed_hbf_cost(
                central_point(),
                hbf_hardware_variant=replace(
                    DEFAULT_HBF_HARDWARE_VARIANT,
                    card_count=4,
                ),
            )
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "LPDDR topology capacity mismatches"):
            proposed_hbf_cost(
                central_point(),
                hbf_hardware_variant=replace(
                    DEFAULT_HBF_HARDWARE_VARIANT,
                    lpddr_capacity_gib_per_card=128.0,
                ),
            )
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "HBF capacity ratio mismatches"):
            proposed_hbf_cost(
                central_point(),
                hbf_hardware_variant=replace(
                    DEFAULT_HBF_HARDWARE_VARIANT,
                    hbf_capacity_ratio_to_hbm=15.0,
                ),
            )

    def test_report_keeps_baseline_invariant_across_rows(self):
        report = evaluate_with_provenance({
            TIERING_SYSTEM_KEY: 1_000.0,
            PROPOSED_SYSTEM_KEY: 900.0,
        })
        self.assertEqual(len(report.sensitivity_rows), 9)
        self.assertTrue(all(
            row.tiering_cost == report.tiering_cost
            for row in report.sensitivity_rows
        ))


class TokenEconomicsTests(unittest.TestCase):
    def test_unit_safe_lifetime_token_economics(self):
        evaluation = EvaluationAssumptions(
            lifetime_years=1,
            average_utilization=0.5,
            pue=1,
            electricity_usd_per_kwh=0,
        )
        cost = tiering_baseline_cost(evaluation=evaluation)
        value = token_economics(cost, 2_000.0)
        expected_seconds = 8_760 * 3_600 * 0.5
        expected_tokens = 2_000 * expected_seconds
        self.assertEqual(value.lifetime_loaded_seconds, expected_seconds)
        self.assertEqual(
            value.lifetime_slo_good_output_tokens, expected_tokens)
        self.assertAlmostEqual(
            value.lifetime_slo_good_output_tokens_per_tco_usd,
            expected_tokens / cost.lifetime_tco_usd,
        )
        self.assertAlmostEqual(
            value.tco_usd_per_million_slo_good_output_tokens,
            cost.lifetime_tco_usd / expected_tokens * 1_000_000,
        )

    def test_idle_power_is_explicit_and_only_changes_energy(self):
        zero_idle = EvaluationAssumptions(
            average_utilization=0.5,
            idle_power_fraction_of_active=0.0,
        )
        half_idle = EvaluationAssumptions(
            average_utilization=0.5,
            idle_power_fraction_of_active=0.5,
        )
        zero_cost = tiering_baseline_cost(evaluation=zero_idle)
        half_cost = tiering_baseline_cost(evaluation=half_idle)
        self.assertEqual(zero_cost.capex_usd, half_cost.capex_usd)
        self.assertEqual(zero_cost.it_power_w, half_cost.it_power_w)
        self.assertLess(
            zero_cost.lifetime_facility_energy_kwh,
            half_cost.lifetime_facility_energy_kwh,
        )
        self.assertIn("idle_power_fraction", half_cost.power_utilization_semantics)

    def test_zero_goodput_has_zero_tokens_per_dollar_and_no_inverse(self):
        cost = proposed_hbf_cost(central_point())
        value = token_economics(cost, 0)
        self.assertEqual(
            value.lifetime_slo_good_output_tokens_per_tco_usd, 0)
        self.assertIsNone(
            value.tco_usd_per_million_slo_good_output_tokens)

    def test_break_even_is_tco_ratio_times_baseline_goodput(self):
        baseline_goodput = 1_000.0
        proposed_goodput = 700.0
        axes = SensitivityAxes(
            npu_logic_capex_ratios_to_gpu_logic=(0.5,),
            hbf_subsystem_capex_ratios_to_hbm_stack=(0.5,),
            npu_logic_power_ratios_to_gpu_logic=(0.5,),
            hbf_subsystem_power_ratios_to_hbm_stack=(3.5,),
        )
        row = evaluate_with_provenance({
            TIERING_SYSTEM_KEY: baseline_goodput,
            PROPOSED_SYSTEM_KEY: proposed_goodput,
        }, axes=axes).sensitivity_rows[0]
        self.assertAlmostEqual(
            row.break_even_proposed_goodput_ratio_to_tiering,
            row.proposed_tco_ratio_to_tiering,
        )
        self.assertAlmostEqual(
            row.break_even_proposed_goodput_tokens_per_second,
            baseline_goodput * row.proposed_tco_ratio_to_tiering,
        )
        self.assertAlmostEqual(
            row.proposed_tokens_per_usd_ratio_to_tiering,
            (
                proposed_goodput / baseline_goodput
                / row.proposed_tco_ratio_to_tiering
            ),
        )


class DisclosureAndValidationTests(unittest.TestCase):
    def test_oracle_is_performance_only_and_excluded_from_economics(self):
        report = evaluate_with_provenance({
            TIERING_SYSTEM_KEY: 1_000,
            PROPOSED_SYSTEM_KEY: 900,
            ORACLE_SYSTEM_KEY: 1_100,
        })
        self.assertEqual(report.economic_system_keys, ECONOMIC_SYSTEM_KEYS)
        self.assertNotIn(
            ORACLE_SYSTEM_KEY, report.economic_system_keys)
        oracle = report.oracle_performance_reference
        self.assertEqual(oracle.slo_good_output_tokens_per_second, 1_100)
        self.assertTrue(oracle.infinite_hbm_capacity)
        self.assertFalse(oracle.physical_bom_available)
        self.assertFalse(oracle.included_in_main_tco_comparison)
        self.assertIsNone(oracle.tco_usd)
        self.assertIsNone(oracle.tokens_per_usd)
        self.assertIn("unphysical", oracle.exclusion_reason)

    def test_report_is_strict_json_safe(self):
        report = evaluate_with_provenance({
            TIERING_SYSTEM_KEY: 1_000,
            PROPOSED_SYSTEM_KEY: 900,
            ORACLE_SYSTEM_KEY: 1_100,
        })
        payload = report.to_json_dict()
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
        self.assertIn('"report_schema": "hbf-comparison-tco-v1"', encoded)
        self.assertEqual(
            payload["performance_provenance"]
            ["selected_tiering_policy_key"],
            "cpu_ssd",
        )
        provenance = payload["performance_provenance"]
        self.assertEqual(
            provenance["first_ttft_slo_ns"], 30_000_000_000)
        self.assertEqual(
            provenance["resume_ttft_slo_ns"], 30_000_000_000)
        self.assertEqual(provenance["tpot_slo_ns"], 300_000_000)
        self.assertEqual(
            provenance["gpu_config_sha256"],
            PINNED_GPU_CONFIG_SHA256,
        )
        self.assertEqual(
            provenance["hbf_hardware_variant"]["hbf_config_sha256"],
            PINNED_HBF_CONFIG_SHA256,
        )
        self.assertEqual(
            provenance["hbf_hardware_variant"]
            ["intra_fabric_bandwidth_gbps_per_card"],
            50.0,
        )
        self.assertEqual(
            provenance["hbf_hardware_variant"]
            ["lpddr_effective_bandwidth_gbps_per_card"],
            204.8,
        )
        self.assertEqual(
            provenance["tiering_result"]["metric_scope"], "all")
        self.assertEqual(
            provenance["tiering_result"]["metric_definition"],
            OUTPUT_TOKEN_GOODPUT_DEFINITION,
        )
        self.assertEqual(
            provenance["tiering_result"]["schedule_hash_semantics"],
            "ordered_paired_seed_schedule_set_manifest",
        )
        self.assertEqual(
            provenance["operating_point_mode"],
            "matched_single_operating_point",
        )
        self.assertFalse(
            provenance["maximum_slo_sustainable_claim"])
        self.assertIn(
            "not a maximum",
            payload["token_economics_scope"],
        )
        self.assertEqual(
            payload["memory_capacity"]
            ["hbf_capacity_ratio_to_hbm_per_card"],
            16.0,
        )
        self.assertEqual(
            payload["memory_capacity"]
            ["proposed_raw_hbf_capacity_bytes"],
            8 * 1_280_000_000_000,
        )

    def test_typed_provenance_is_required_and_goodput_must_match(self):
        values = {
            TIERING_SYSTEM_KEY: 1_000,
            PROPOSED_SYSTEM_KEY: 900,
        }
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "performance_provenance is required"):
            evaluate_tco_sensitivity(values)
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "goodput mismatches"):
            evaluate_tco_sensitivity(
                values,
                performance_provenance=performance_provenance(999, 900),
            )

    def test_cross_system_provenance_mismatch_is_rejected(self):
        with self.assertRaisesRegex(
                HBFComparisonTCOError,
                "mismatched performance provenance field schedule_sha256"):
            ComparisonPerformanceProvenance(
                selected_tiering_policy_key="cpu_ssd",
                hbf_layout_key="hbf_tp4",
                hbf_policy_key=(
                    "first_gpu__migration_inflight_resume_gpu__"
                    "hbf_ready_resume_hbf__turn_boundary_lpddr_v1"),
                hbf_policy_contract_sha256="c" * 64,
                gpu_config_sha256=PINNED_GPU_CONFIG_SHA256,
                hbf_hardware_variant=DEFAULT_HBF_HARDWARE_VARIANT,
                first_ttft_slo_ns=30_000_000_000,
                resume_ttft_slo_ns=30_000_000_000,
                tpot_slo_ns=300_000_000,
                operating_point_mode="matched_single_operating_point",
                rate_selection_semantics=(
                    "matched offered-rate cell; "
                    "not a sustainable-rate ceiling"),
                maximum_slo_sustainable_claim=False,
                tiering_result=result_provenance(
                    TIERING_SYSTEM_KEY, 1_000),
                proposed_result=result_provenance(
                    PROPOSED_SYSTEM_KEY,
                    900,
                    schedule_sha256="c" * 64,
                ),
            )

    def test_invalid_ratios_and_evaluation_are_rejected(self):
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "below 1"):
            SensitivityAxes(
                hbf_subsystem_capex_ratios_to_hbm_stack=(1.0,))
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "above 1"):
            SensitivityAxes(
                hbf_subsystem_power_ratios_to_hbm_stack=(1.0,))
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "finite number"):
            HardwareAnchors(h100_card_capex_usd=True)
        with self.assertRaisesRegex(
                HBFComparisonTCOError, r"\(0, 1\]"):
            EvaluationAssumptions(average_utilization=0)

    def test_numeric_strings_are_rejected_at_every_public_input_layer(self):
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "finite number"):
            HardwareAnchors(h100_card_capex_usd="30000")
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "finite number"):
            DeploymentTopology(host_dram_gib_per_host="512")
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "finite number"):
            EvaluationAssumptions(average_utilization="0.7")
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "finite number"):
            SensitivityPoint("0.5", 0.5, 0.5, 3.5)
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "finite number"):
            replace(
                result_provenance(TIERING_SYSTEM_KEY, 1_000),
                offered_session_rate_per_second="2",
            )
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "finite number"):
            evaluate_tco_sensitivity(
                {
                    TIERING_SYSTEM_KEY: "1000",
                    PROPOSED_SYSTEM_KEY: 900,
                },
                performance_provenance=performance_provenance(1_000, 900),
            )

    def test_missing_provenance_identity_or_origin_is_rejected(self):
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "result_goodput_origin"):
            replace(
                result_provenance(TIERING_SYSTEM_KEY, 1_000),
                result_goodput_origin="",
            )
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "SHA-256"):
            replace(
                result_provenance(TIERING_SYSTEM_KEY, 1_000),
                measurement_cohort_sha256="not-a-hash",
            )
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "selected_tiering_policy_key"):
            ComparisonPerformanceProvenance(
                selected_tiering_policy_key="",
                hbf_layout_key="hbf_tp4",
                hbf_policy_key=(
                    "first_gpu__migration_inflight_resume_gpu__"
                    "hbf_ready_resume_hbf__turn_boundary_lpddr_v1"),
                hbf_policy_contract_sha256="c" * 64,
                gpu_config_sha256=PINNED_GPU_CONFIG_SHA256,
                hbf_hardware_variant=DEFAULT_HBF_HARDWARE_VARIANT,
                first_ttft_slo_ns=30_000_000_000,
                resume_ttft_slo_ns=30_000_000_000,
                tpot_slo_ns=300_000_000,
                operating_point_mode="matched_single_operating_point",
                rate_selection_semantics=(
                    "matched offered-rate cell; "
                    "not a sustainable-rate ceiling"),
                maximum_slo_sustainable_claim=False,
                tiering_result=result_provenance(
                    TIERING_SYSTEM_KEY, 1_000),
                proposed_result=result_provenance(
                    PROPOSED_SYSTEM_KEY, 900),
            )

    def test_provenance_policy_layout_slo_and_config_are_validated(self):
        valid = performance_provenance(1_000, 900)
        with self.assertRaisesRegex(
                HBFComparisonTCOError,
                "selected_tiering_policy_key must be one of"):
            replace(valid, selected_tiering_policy_key="typo")
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "hbf_layout_key must be one of"):
            replace(valid, hbf_layout_key="tp6")
        with self.assertRaisesRegex(
                HBFComparisonTCOError,
                "wide-LPDDR hardware variant must be selected together"):
            replace(valid, hbf_layout_key="hbf_tp4_wide")
        with self.assertRaisesRegex(
                HBFComparisonTCOError,
                "wide-LPDDR hardware variant must be selected together"):
            replace(
                valid,
                hbf_hardware_variant=WIDE_LPDDR_HBF_HARDWARE_VARIANT,
            )
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "hbf_policy_key must be one of"):
            replace(valid, hbf_policy_key="opaque-policy")
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "SHA-256"):
            replace(valid, hbf_policy_contract_sha256="not-a-hash")
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "positive integer"):
            replace(valid, first_ttft_slo_ns="30000000000")
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "SHA-256"):
            replace(valid, gpu_config_sha256="not-a-config-hash")
        with self.assertRaisesRegex(
                HBFComparisonTCOError,
                "only supports matched_single_operating_point"):
            replace(valid, operating_point_mode="maximum_slo_sustainable")
        with self.assertRaisesRegex(
                HBFComparisonTCOError,
                "cannot claim maximum SLO-sustainable"):
            replace(valid, maximum_slo_sustainable_claim=True)

    def test_metric_artifact_and_schedule_set_provenance_fail_closed(self):
        valid = performance_provenance(1_000, 900)
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "result_manifest_sha256"):
            replace(
                valid.tiering_result,
                result_manifest_sha256="not-a-hash",
            )
        with self.assertRaisesRegex(
                HBFComparisonTCOError,
                "schedule_hash_semantics disagrees with seed_count"):
            replace(
                valid.tiering_result,
                schedule_hash_semantics="single_frozen_schedule",
            )
        with self.assertRaisesRegex(
                HBFComparisonTCOError,
                "metric_json_path does not match metric_scope"):
            replace(valid.tiering_result, metric_scope="resume")
        resume_result = replace(
            valid.proposed_result,
            metric_scope="resume",
            metric_json_path=(
                "summary.request_kind_summaries.resume."
                "offered_load_normalized_output_token_goodput.value"
            ),
        )
        with self.assertRaisesRegex(
                HBFComparisonTCOError,
                "mismatched performance provenance field metric_scope"):
            replace(valid, proposed_result=resume_result)

    def test_wide_variant_is_retained_and_drives_report_bom(self):
        variant = WIDE_LPDDR_HBF_HARDWARE_VARIANT
        values = {
            TIERING_SYSTEM_KEY: 1_000,
            PROPOSED_SYSTEM_KEY: 900,
        }
        report = evaluate_tco_sensitivity(
            values,
            performance_provenance=performance_provenance(
                1_000,
                900,
                hbf_hardware_variant=variant,
                hbf_layout_key="hbf_tp4_wide",
            ),
            axes=SensitivityAxes(
                npu_logic_capex_ratios_to_gpu_logic=(0.5,),
                hbf_subsystem_capex_ratios_to_hbm_stack=(0.5,),
                npu_logic_power_ratios_to_gpu_logic=(0.5,),
                hbf_subsystem_power_ratios_to_hbm_stack=(3.5,),
            ),
        )
        self.assertEqual(
            report.performance_provenance.hbf_hardware_variant,
            variant,
        )
        self.assertEqual(
            report.performance_provenance.hbf_layout_key,
            "hbf_tp4_wide",
        )
        lpddr = report.sensitivity_rows[0].proposed_cost.component(
            "hbf_card_lpddr")
        self.assertEqual(variant.hbf_config_sha256,
                         PINNED_HBF_WIDE_LPDDR_CONFIG_SHA256)
        self.assertEqual(
            variant.lpddr_effective_bandwidth_gbps_per_card, 409.6)
        self.assertEqual(variant.lpddr_bandwidth_multiplier, 2.0)
        self.assertEqual(
            lpddr.unit_capex_usd,
            HardwareAnchors().lpddr_capex_usd_per_gib * 1.5,
        )
        self.assertEqual(
            lpddr.unit_it_power_w,
            HardwareAnchors().lpddr_power_w_per_gib * 1.75,
        )
        self.assertIn(
            "409.6 GB/s per card", lpddr.assumption)

    def test_goodput_map_rejects_missing_unknown_and_zero_baseline(self):
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "missing systems"):
            evaluate_tco_sensitivity(
                {TIERING_SYSTEM_KEY: 1_000},
                performance_provenance=performance_provenance(1_000, 900),
            )
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "unknown systems"):
            evaluate_tco_sensitivity(
                {
                    TIERING_SYSTEM_KEY: 1_000,
                    PROPOSED_SYSTEM_KEY: 900,
                    "oracle_with_fake_tco": 1_100,
                },
                performance_provenance=performance_provenance(1_000, 900),
            )
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "positive"):
            evaluate_with_provenance({
                TIERING_SYSTEM_KEY: 0,
                PROPOSED_SYSTEM_KEY: 900,
            })


if __name__ == "__main__":
    unittest.main()

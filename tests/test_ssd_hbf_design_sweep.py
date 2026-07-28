from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from serving.core.gpu_pd_dual_oracle import (
    ROUTE_BALANCED_TRACE_WORK,
    DualStrictInfiniteHBMOracle,
)
from serving.core.gpu_pd_dual_tiered import (
    DualFiniteHBMTieredBaseline,
)
from serving.core.hbf_comparison_workload import (
    CallSpec,
    ScheduledSession,
    SessionSpec,
    stable_json_sha256,
)
from serving.core.tracelab_comparison_scenarios import (
    ArrivalRateContract,
    BalancedCausalPrefixManifest,
    TraceLabComparisonScenario,
)
from serving.ssd_hbf_design_sweep import (
    BASELINE_CANDIDATE_KEY,
    BASELINE_CANDIDATE_KEYS,
    BASELINE_RESTORE_MODES,
    CANONICAL_MIGRATION_POLICIES,
    DEFAULT_MIGRATION_POLICIES,
    ORACLE_CANDIDATE_KEY,
    REQUIRED_SESSION_RATE,
    SSDHBFDesignSweepError,
    SSD_HBF_CONTRACT_KEY,
    STREAMING_BASELINE_CANDIDATE_KEY,
    SUPPORTED_MIGRATION_POLICIES,
    _CellTask,
    _design_runtime_energy_tco,
    _execute_task,
    _load_resumable_cell,
    _parser,
    _seal_record,
    _task_contract,
    aggregate_cell_records,
    build_design_grid,
    build_tasks,
    evaluate_reference_eligibility,
    make_design_system,
    make_design_spec,
    make_reference_system,
    parse_active_memory_spec,
    run_reference_cell,
    run_design_space,
    validate_scenario_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ROSTER = ("measure::call-0",)
ROSTER_HASH = stable_json_sha256(list(ROSTER))


def _scheduled_session(
        *, seed: int = 0,
) -> tuple[ScheduledSession, ...]:
    session_id = f"session-{seed}"
    call = CallSpec(
        session_id=session_id,
        source_index=seed,
        call_index=0,
        input_tokens=64,
        output_tokens=2,
        tool_duration_ns=0,
        cached_prefix_tokens=0,
        fresh_input_tokens=64,
        lineage_status=None,
        inter_turn_gap_type=None,
    )
    session = SessionSpec(
        source_index=seed,
        session_id=session_id,
        source_arrival_time_ns=0,
        source_session_identity_sha256=None,
        calls=(call,),
    )
    return (ScheduledSession(
        offer_index=0,
        session=session,
        arrival_time_ns=0,
        unit_interarrival=0.0,
        unit_arrival_time=0.0,
    ),)


def _summary(
        goodput: float, *,
        joint: float,
        latency_scale: float = 1.0,
) -> dict[str, object]:
    return {
        "offered_load_normalized_output_token_goodput": {
            "value": goodput,
        },
        "offered_load_normalized_request_goodput": {
            "value": goodput / 10.0,
        },
        "observed_completion_span_throughput": {
            "requests_per_second": goodput / 20.0,
        },
        "slo": {
            "all_slo_pass_fraction": joint,
        },
        "latency_distributions_ns": {
            "first_ttft": {"p95_ns": 10.0 * latency_scale},
            "resume_ttft": {"p95_ns": 20.0 * latency_scale},
            "tpot_eligible": {"p95_ns": 30.0 * latency_scale},
        },
    }


def _record(
        candidate_key: str,
        seed: int,
        goodput: float,
        *,
        joint: float,
        hbf_card_writes=(0,) * 8,
        horizon_ns: int = 1_000_000_000,
) -> dict[str, object]:
    restore_execution_mode = (
        BASELINE_RESTORE_MODES[candidate_key]
        if candidate_key in BASELINE_RESTORE_MODES
        else None
        if candidate_key == ORACLE_CANDIDATE_KEY
        else (
            "layerwise_streaming"
            if "layerwise-streaming" in candidate_key
            else "bulk"
        )
    )
    return {
        "candidate_kind": (
            "baseline"
            if candidate_key in BASELINE_CANDIDATE_KEYS.values()
            else "oracle"
            if candidate_key == ORACLE_CANDIDATE_KEY
            else "design"
        ),
        "candidate_key": candidate_key,
        "restore_execution_mode": restore_execution_mode,
        "seed": seed,
        "session_rate": REQUIRED_SESSION_RATE,
        "measurement_roster": {
            "identity_count": len(ROSTER),
            "ordered_identities_sha256": ROSTER_HASH,
        },
        "summary": _summary(goodput, joint=joint),
        "execution_observation": {
            "simulated_horizon_ns": horizon_ns,
            "elapsed_wall_time_ns": 1,
        },
        "system_report": {
            "node": {
                "hbf_lifecycle": {
                    "hbf_write_accounting": {
                        "schema_version": 1,
                        "accounting_basis": (
                            "physical_media_payload_of_admitted_jobs"),
                        "complete_for_endurance_projection": True,
                        "total_physical_write_bytes": sum(
                            hbf_card_writes),
                        "wasted_physical_write_bytes": 0,
                        "static_model_weight": {
                            "bytes_per_card": 100,
                            "write_count": 1,
                            "included_in_recurring_kv_wear": False,
                        },
                        "cards": [
                            {
                                "device_id": (
                                    f"hbf-server-0-card-{card_id}"),
                                "server_id": 0,
                                "card_id": card_id,
                                "kv_region_capacity_bytes": 1_000,
                                "total_write_bytes": write_bytes,
                                "wasted_write_bytes": 0,
                            }
                            for card_id, write_bytes
                            in enumerate(hbf_card_writes)
                        ],
                    },
                },
            },
        },
    }


class _FakeOfferedPlan:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.returned_schedule = _scheduled_session(seed=seed)
        self.requested_rates: list[float] = []

    def at_rate(self, rate: float):
        self.requested_rates.append(rate)
        return self.returned_schedule


class _FakeScenario:
    def __init__(self) -> None:
        self.manifest = SimpleNamespace(
            measurement_request_identities=ROSTER,
        )
        self.plans: dict[int, _FakeOfferedPlan] = {}

    def build_offered_plan(self, *, seed: int):
        plan = _FakeOfferedPlan(seed)
        self.plans[seed] = plan
        return plan


class SSDHBFDesignSweepTests(unittest.TestCase):
    def setUp(self):
        self.memory12 = parse_active_memory_spec(
            "lpddr:12:409.6")
        self.memory16 = parse_active_memory_spec(
            "lpddr:16:409.6")
        self.tp4 = make_design_spec(
            hbf_layout="tp4x2",
            migration_policy="delay_1000ms",
            active_memory=self.memory12,
        )
        self.tp8 = make_design_spec(
            hbf_layout="tp8_context",
            migration_policy="tool_or_human_immediate",
            active_memory=self.memory16,
        )

    def test_defaults_cover_all_distinct_supported_migration_policies(self):
        self.assertEqual(
            DEFAULT_MIGRATION_POLICIES,
            CANONICAL_MIGRATION_POLICIES,
        )
        self.assertEqual(
            set(SUPPORTED_MIGRATION_POLICIES)
            - set(CANONICAL_MIGRATION_POLICIES),
            {
                "delay_1s",
                "load_aware_demote",
                "load_aware_demote_h2",
                "load_aware_density",
                "load_aware_density_oracle",
                "load_aware_calls",
                "load_aware_calls_oracle",
                "composite",
                "composite_adaptive",
                "composite_ready",
                "composite_ready_adaptive",
            },
        )
        self.assertIn("delay_1000ms", CANONICAL_MIGRATION_POLICIES)
        self.assertNotIn("delay_1s", CANONICAL_MIGRATION_POLICIES)
        self.assertIn("composite", SUPPORTED_MIGRATION_POLICIES)
        self.assertIn(
            "composite_adaptive", SUPPORTED_MIGRATION_POLICIES)
        self.assertIn(
            "composite_ready", SUPPORTED_MIGRATION_POLICIES)
        self.assertIn(
            "composite_ready_adaptive",
            SUPPORTED_MIGRATION_POLICIES,
        )
        self.assertNotIn(
            "composite", DEFAULT_MIGRATION_POLICIES)

    def test_ineligible_reference_audit_is_explicit_opt_in(self):
        default = _parser().parse_args(["--output", "/tmp/out"])
        audit = _parser().parse_args([
            "--output", "/tmp/out",
            "--allow-ineligible-reference-audit",
        ])

        self.assertFalse(default.allow_ineligible_reference_audit)
        self.assertTrue(audit.allow_ineligible_reference_audit)
        self.assertEqual(default.layouts, ["tp8_context"])

    def test_balanced_scenario_accepts_rates_from_its_arrival_contract(self):
        manifest = object.__new__(BalancedCausalPrefixManifest)
        object.__setattr__(manifest, "scenario_id", "balanced-test")
        object.__setattr__(manifest, "equilibrium_workload", False)
        object.__setattr__(
            manifest, "measurement_request_identities", ROSTER)
        object.__setattr__(
            manifest,
            "measurement_request_identities_sha256",
            ROSTER_HASH,
        )
        object.__setattr__(
            manifest,
            "arrival_contract",
            ArrivalRateContract(
                rates=(3.0, 5.0),
                maximum_rate=5.0,
                enumerated_only=True,
                rate_unit="sessions_per_second",
                process="test",
                first_arrival_semantics="test",
                offer_order_semantics="test",
            ),
        )
        object.__setattr__(
            manifest, "to_dict", lambda: {"scenario_id": "balanced-test"})
        scenario = TraceLabComparisonScenario(
            workload=None,
            manifest=manifest,
            shuffle_session_starts=False,
        )

        contract = validate_scenario_contract(
            scenario, session_rate=5.0)

        self.assertEqual(contract["required_session_rate"], 5.0)
        self.assertEqual(
            contract["declared_session_rates"], [3.0, 5.0])
        with self.assertRaisesRegex(
                SSDHBFDesignSweepError,
                "violates the scenario arrival contract"):
            validate_scenario_contract(
                scenario, session_rate=4.0)

    def test_grid_distinguishes_layouts_but_keeps_one_physical_host(self):
        grid = build_design_grid(
            layouts=("tp4x2", "tp8_context"),
            migration_policies=(
                "tool_or_human_immediate", "delay_1000ms"),
            active_memories=(self.memory12,),
        )

        self.assertEqual(len(grid), 4)
        self.assertEqual(len({design.key for design in grid}), 4)
        self.assertEqual(
            {design.simulator_layout for design in grid},
            {"tp4", "tp8_context"},
        )
        self.assertTrue(all(
            design.gpu_host_count == 1
            and design.hbf_host_count == 1
            and design.hbf_card_count == 8
            for design in grid
        ))
        self.assertEqual(self.tp4.tco_layout, "tp4x2")
        self.assertEqual(self.tp8.tco_layout, "tp8")
        with self.assertRaisesRegex(
                SSDHBFDesignSweepError, "unsupported migration"):
            make_design_spec(
                hbf_layout="tp4x2",
                migration_policy="direct_gpu_hbm_to_hbf",
                active_memory=self.memory12,
            )

    def test_design_factory_materializes_one_gpu_and_one_hbf(self):
        for design, expected_layout in (
                (self.tp4, "tp4"),
                (self.tp8, "tp8_context")):
            with self.subTest(layout=design.hbf_layout):
                system = make_design_system(
                    repo_root=REPO_ROOT,
                    spec=design,
                )
                report = system.report()
                self.assertEqual(
                    report["architecture"]["gpu_server_count"], 1)
                self.assertEqual(
                    report["architecture"]["hbf_server_count"], 1)
                self.assertTrue(
                    report["architecture"][
                        "local_ssd_checkpoint"])
                self.assertEqual(
                    report["architecture"]["hbf_layout"],
                    expected_layout,
                )
                self.assertEqual(
                    report["policy"]["promotion"]["key"],
                    design.migration_policy,
                )

    def test_reference_factories_materialize_two_balanced_gpu_clusters(self):
        baseline = make_reference_system(
            repo_root=REPO_ROOT,
            candidate_kind="baseline",
        )
        oracle = make_reference_system(
            repo_root=REPO_ROOT,
            candidate_kind="oracle",
        )

        self.assertIsInstance(
            baseline, DualFiniteHBMTieredBaseline)
        self.assertIsInstance(
            oracle, DualStrictInfiniteHBMOracle)
        self.assertEqual(len(baseline.nodes), 2)
        self.assertEqual(len(oracle.nodes), 2)
        self.assertEqual(
            baseline.route_policy, ROUTE_BALANCED_TRACE_WORK)
        self.assertEqual(
            oracle.route_policy, ROUTE_BALANCED_TRACE_WORK)
        self.assertEqual(baseline.policy, "ssd_direct")

    def test_reference_cell_reports_dual_gpu_physical_topology(self):
        result = run_reference_cell(
            repo_root=REPO_ROOT,
            candidate_kind="baseline",
            scheduled_sessions=_scheduled_session(),
            session_rate=REQUIRED_SESSION_RATE,
            seed=0,
            measurement_identities=("session-0::call-0",),
        )

        self.assertEqual(
            result["physical_topology"],
            {
                "gpu_host_count": 2,
                "hbf_host_count": 0,
                "h100_card_count": 16,
                "hbf_card_count": 0,
                "local_ssd_device_count": 16,
            },
        )

    def test_hbf_read_mode_is_a_distinct_propagated_axis(self):
        grid = build_design_grid(
            layouts=("tp8_context",),
            migration_policies=("eager",),
            active_memories=(self.memory16,),
            hbf_read_modes=("demand", "prefetch"),
        )

        self.assertEqual(len(grid), 2)
        self.assertEqual(
            {design.hbf_read_mode for design in grid},
            {"demand", "prefetch"},
        )
        self.assertEqual(len({design.key for design in grid}), 2)
        for design in grid:
            system = make_design_system(
                repo_root=REPO_ROOT,
                spec=design,
            )
            self.assertEqual(
                system.node.hbf_hardware.hbf_read_prefetch_enabled,
                design.hbf_read_mode == "prefetch",
            )

    def test_mixed_batch_latency_guard_is_a_distinct_propagated_axis(self):
        grid = build_design_grid(
            layouts=("tp8_context",),
            migration_policies=("composite_ready",),
            active_memories=(self.memory16,),
            mixed_batch_latency_limits_ms=(None, 225, 250),
        )

        self.assertEqual(len(grid), 3)
        self.assertEqual(
            {
                design.mixed_batch_latency_limit_ms
                for design in grid
            },
            {None, 225, 250},
        )
        self.assertEqual(len({design.key for design in grid}), 3)
        for design in grid:
            system = make_design_system(
                repo_root=REPO_ROOT,
                spec=design,
            )
            expected = (
                None
                if design.mixed_batch_latency_limit_ms is None
                else (
                    design.mixed_batch_latency_limit_ms
                    * 1_000_000
                )
            )
            self.assertEqual(
                system.node.hbf_pool.mixed_batch_latency_limit_ns,
                expected,
            )

    def test_restore_mode_is_propagated_and_gets_a_matched_baseline(self):
        grid = build_design_grid(
            layouts=("tp8_context",),
            migration_policies=("eager",),
            active_memories=(self.memory16,),
            restore_execution_modes=(
                "bulk", "layerwise_streaming"),
        )

        self.assertEqual(len(grid), 2)
        self.assertEqual(
            {design.restore_execution_mode for design in grid},
            {"bulk", "layerwise_streaming"},
        )
        self.assertEqual(len({design.key for design in grid}), 2)
        for design in grid:
            system = make_design_system(
                repo_root=REPO_ROOT,
                spec=design,
            )
            self.assertEqual(
                system.node.restore_execution_mode,
                design.restore_execution_mode,
            )
        streaming_baseline = make_reference_system(
            repo_root=REPO_ROOT,
            candidate_kind="baseline",
            restore_execution_mode="layerwise_streaming",
        )
        self.assertEqual(
            streaming_baseline.restore_execution_mode,
            "layerwise_streaming",
        )
        self.assertTrue(all(
            node.restore_execution_mode == "layerwise_streaming"
            for node in streaming_baseline.nodes
        ))

    def test_tasks_share_oracle_but_match_each_restore_mode(self):
        scenario = _FakeScenario()
        streaming = make_design_spec(
            hbf_layout="tp4x2",
            migration_policy="delay_1000ms",
            active_memory=self.memory12,
            restore_execution_mode="layerwise_streaming",
        )
        contract = {
            "manifest_sha256": "a" * 64,
            "measurement_roster_sha256": ROSTER_HASH,
        }
        with (
            patch(
                "serving.ssd_hbf_design_sweep."
                "validate_scenario_contract",
                return_value=contract,
            ),
            patch(
                "serving.ssd_hbf_design_sweep."
                "_execution_inputs_sha256",
                return_value="b" * 64,
            ),
        ):
            tasks = build_tasks(
                repo_root=REPO_ROOT,
                scenario=scenario,
                designs=(self.tp4, streaming),
                seeds=(7, 11),
            )

        self.assertEqual(len(tasks), 10)
        for seed in (7, 11):
            cohort = [
                task for task in tasks if task.seed == seed]
            self.assertEqual(
                [task.candidate_key for task in cohort],
                [
                    BASELINE_CANDIDATE_KEY,
                    STREAMING_BASELINE_CANDIDATE_KEY,
                    ORACLE_CANDIDATE_KEY,
                    self.tp4.key,
                    streaming.key,
                ],
            )
            self.assertEqual(
                {
                    _task_contract(task)[
                        "restore_execution_mode"]
                    for task in cohort
                },
                {None, "bulk", "layerwise_streaming"},
            )

    def test_aggregation_pairs_designs_to_matching_restore_baselines(self):
        streaming = make_design_spec(
            hbf_layout="tp4x2",
            migration_policy="delay_1000ms",
            active_memory=self.memory12,
            restore_execution_mode="layerwise_streaming",
        )
        records = []
        for seed in (7, 11, 13):
            records.extend((
                _record(
                    BASELINE_CANDIDATE_KEY,
                    seed, 5.0, joint=0.05,
                ),
                _record(
                    STREAMING_BASELINE_CANDIDATE_KEY,
                    seed, 8.0, joint=0.08,
                ),
                _record(
                    ORACLE_CANDIDATE_KEY,
                    seed, 100.0, joint=0.98,
                ),
                _record(
                    self.tp4.key,
                    seed, 60.0, joint=0.60,
                ),
                _record(
                    streaming.key,
                    seed, 64.0, joint=0.64,
                ),
            ))

        aggregate = aggregate_cell_records(
            records, (self.tp4, streaming))
        rate = aggregate["rates"][0]
        self.assertTrue(
            rate["reference_eligibility"]["eligible"])
        self.assertEqual(
            set(rate["references"]),
            {
                BASELINE_CANDIDATE_KEY,
                STREAMING_BASELINE_CANDIDATE_KEY,
                ORACLE_CANDIDATE_KEY,
            },
        )
        rows = {
            row["design"]["restore_execution_mode"]: row
            for row in rate["designs"]
        }
        self.assertEqual(
            rows["bulk"]["baseline_candidate_key"],
            BASELINE_CANDIDATE_KEY,
        )
        self.assertEqual(
            rows["layerwise_streaming"][
                "baseline_candidate_key"],
            STREAMING_BASELINE_CANDIDATE_KEY,
        )
        self.assertEqual(
            rows["bulk"]["paired_vs_baseline_goodput"][
                "candidate_over_reference"]["mean"],
            12.0,
        )
        self.assertEqual(
            rows["layerwise_streaming"][
                "paired_vs_baseline_goodput"][
                    "candidate_over_reference"]["mean"],
            8.0,
        )

    def test_endurance_aggregation_weights_bytes_by_trace_duration(self):
        records = []
        for seed, writes, horizon_ns in (
                (7, (100,) + (0,) * 7, 1_000_000_000),
                (11, (300,) + (0,) * 7, 3_000_000_000)):
            records.extend((
                _record(
                    BASELINE_CANDIDATE_KEY,
                    seed, 5.0, joint=0.05,
                ),
                _record(
                    ORACLE_CANDIDATE_KEY,
                    seed, 100.0, joint=0.98,
                ),
                _record(
                    self.tp4.key,
                    seed, 60.0, joint=0.60,
                    hbf_card_writes=writes,
                    horizon_ns=horizon_ns,
                ),
            ))

        aggregate = aggregate_cell_records(
            records, (self.tp4,))
        endurance = aggregate["rates"][0]["designs"][0][
            "hbf_endurance"]
        central = endurance["scenarios"][
            "slc_100k_pe_waf1"]

        self.assertEqual(endurance["sample_count"], 2)
        self.assertEqual(
            endurance["total_observed_seconds"], 4.0)
        self.assertEqual(
            endurance["total_physical_write_bytes"], 400)
        limiting = central["cards"][0]
        self.assertEqual(
            limiting["payload_write_bytes_per_second"],
            100.0,
        )
        self.assertEqual(
            central["limiting_device_ids"],
            ["hbf-server-0-card-0"],
        )
        self.assertEqual(
            endurance["hotness"]["hottest_card_share"],
            1.0,
        )

    def test_runtime_tco_is_pooled_and_keeps_per_seed_audit(self):
        baseline_projection = SimpleNamespace(
            trace_average_it_power_w=10_000.0,
            five_year_facility_energy_kwh=525_600.0,
            five_year_tco_usd=600_000.0,
        )
        proposed_projection = SimpleNamespace(
            trace_average_it_power_w=11_000.0,
            five_year_facility_energy_kwh=578_160.0,
            five_year_tco_usd=620_000.0,
        )

        class FakeComparison:
            baseline = baseline_projection
            proposed = proposed_projection

            def to_json_dict(self):
                return {
                    "report_schema": "ssd-hbf-runtime-tco-v1",
                    "baseline": {
                        "report_schema": "ssd-hbf-runtime-tco-v1",
                        "system_key": "two_gpu_local_ssd_baseline",
                        "trace_average_it_power_w": 10_000.0,
                        "five_year_facility_energy_kwh": 525_600.0,
                        "five_year_tco_usd": 600_000.0,
                    },
                    "proposed": {
                        "report_schema": "ssd-hbf-runtime-tco-v1",
                        "system_key": (
                            "one_gpu_local_ssd_plus_one_hbf"),
                        "trace_average_it_power_w": 11_000.0,
                        "five_year_facility_energy_kwh": 578_160.0,
                        "five_year_tco_usd": 620_000.0,
                    },
                }

        fake = FakeComparison()
        records = {
            seed: {"system_report": {"seed": seed}}
            for seed in (7, 11, 13)
        }
        static_tco = {
            "baseline_cost": {
                "capex_usd": 500_000.0,
                "five_year_electricity_opex_usd": 50_000.0,
            },
            "proposed_cost": {
                "capex_usd": 510_000.0,
                "five_year_electricity_opex_usd": 55_000.0,
            },
        }
        with (
            patch(
                "serving.ssd_hbf_design_sweep."
                "evaluate_ssd_hbf_runtime_tco",
                return_value=fake,
            ) as evaluate,
            patch(
                "serving.ssd_hbf_design_sweep."
                "aggregate_runtime_tco_comparisons",
                return_value=fake,
            ) as pool,
        ):
            result, reason = _design_runtime_energy_tco(
                baseline_records_by_seed=records,
                proposed_records_by_seed=records,
                static_tco=static_tco,
                require_runtime_energy=True,
            )

        self.assertIsNone(reason)
        self.assertEqual(evaluate.call_count, 3)
        pool.assert_called_once()
        self.assertEqual(
            result["aggregation"]["method"],
            "pooled_energy_over_pooled_simulated_horizon",
        )
        self.assertEqual(
            set(result["per_seed"]), {"7", "11", "13"})
        self.assertEqual(
            result["aggregation"]["paired_by_seed"]["power"][
                "candidate_over_reference"]["mean"],
            1.1,
        )

    def test_tasks_share_one_schedule_object_and_hash_per_seed(self):
        scenario = _FakeScenario()
        contract = {
            "manifest_sha256": "a" * 64,
            "measurement_roster_sha256": ROSTER_HASH,
        }
        with (
            patch(
                "serving.ssd_hbf_design_sweep."
                "validate_scenario_contract",
                return_value=contract,
            ),
            patch(
                "serving.ssd_hbf_design_sweep."
                "_execution_inputs_sha256",
                return_value="b" * 64,
            ),
        ):
            tasks = build_tasks(
                repo_root=REPO_ROOT,
                scenario=scenario,
                designs=(self.tp4, self.tp8),
                seeds=(7, 11),
            )

        self.assertEqual(len(tasks), 8)
        for seed in (7, 11):
            cohort = [task for task in tasks if task.seed == seed]
            self.assertEqual(
                [task.candidate_kind for task in cohort],
                ["baseline", "oracle", "design", "design"],
            )
            schedule = scenario.plans[seed].returned_schedule
            self.assertTrue(all(
                task.scheduled_sessions is schedule
                for task in cohort
            ))
            self.assertTrue(all(
                task.measurement_identities == ROSTER
                for task in cohort
            ))
            schedule_hashes = {
                _task_contract(task)["schedule_sha256"]
                for task in cohort
            }
            roster_hashes = {
                _task_contract(task)[
                    "measurement_identities_sha256"]
                for task in cohort
            }
            self.assertEqual(len(schedule_hashes), 1)
            self.assertEqual(roster_hashes, {ROSTER_HASH})
        with (
            patch(
                "serving.ssd_hbf_design_sweep."
                "validate_scenario_contract",
                return_value=contract,
            ),
            patch(
                "serving.ssd_hbf_design_sweep."
                "_execution_inputs_sha256",
                return_value="b" * 64,
            ),
        ):
            rate_five_tasks = build_tasks(
                repo_root=REPO_ROOT,
                scenario=scenario,
                designs=(self.tp4,),
                seeds=(7, 11),
                session_rate=5.0,
            )
        self.assertEqual(len(rate_five_tasks), 6)
        self.assertTrue(all(
            task.session_rate == 5.0
            for task in rate_five_tasks
        ))
        self.assertEqual(
            scenario.plans[7].requested_rates, [5.0])
        self.assertEqual(
            scenario.plans[11].requested_rates, [5.0])

    def test_reference_eligibility_accepts_only_healthy_oracle_and_gap(self):
        accepted = evaluate_reference_eligibility(
            baseline_goodput_by_seed={7: 5.0, 11: 5.0, 13: 5.0},
            oracle_goodput_by_seed={
                7: 100.0, 11: 100.0, 13: 100.0},
            oracle_joint_slo_by_seed={
                7: 0.96, 11: 0.97, 13: 0.98},
        )
        self.assertTrue(accepted["eligible"])
        self.assertLessEqual(
            accepted[
                "observed_baseline_over_oracle_ci95_upper"],
            0.10,
        )

        low_oracle = evaluate_reference_eligibility(
            baseline_goodput_by_seed={7: 5.0, 11: 5.0, 13: 5.0},
            oracle_goodput_by_seed={
                7: 100.0, 11: 100.0, 13: 100.0},
            oracle_joint_slo_by_seed={
                7: 0.89, 11: 1.0, 13: 1.0},
        )
        self.assertFalse(low_oracle["eligible"])
        self.assertIn(
            "oracle_seed_joint_slo_below_0.90",
            low_oracle["failures"],
        )

        small_gap = evaluate_reference_eligibility(
            baseline_goodput_by_seed={
                7: 20.0, 11: 20.0, 13: 20.0},
            oracle_goodput_by_seed={
                7: 100.0, 11: 100.0, 13: 100.0},
            oracle_joint_slo_by_seed={
                7: 0.98, 11: 0.98, 13: 0.98},
        )
        self.assertFalse(small_gap["eligible"])
        self.assertIn(
            "baseline_over_oracle_ci95_upper_above_0.10",
            small_gap["failures"],
        )

    def test_aggregation_attaches_corrected_tco_and_pareto(self):
        cheap = self.tp4
        dominated = make_design_spec(
            hbf_layout="tp4x2",
            migration_policy="delay_1000ms",
            active_memory=self.memory16,
        )
        records = []
        for seed in (7, 11, 13):
            records.extend((
                _record(
                    BASELINE_CANDIDATE_KEY,
                    seed, 5.0, joint=0.05,
                ),
                _record(
                    ORACLE_CANDIDATE_KEY,
                    seed, 100.0, joint=0.98,
                ),
                _record(
                    cheap.key, seed, 60.0, joint=0.60,
                ),
                _record(
                    dominated.key, seed, 55.0, joint=0.55,
                ),
            ))

        aggregate = aggregate_cell_records(
            records, (cheap, dominated))
        rate = aggregate["rates"][0]
        self.assertTrue(
            rate["reference_eligibility"]["eligible"])
        by_key = {
            row["design"]["key"]: row
            for row in rate["designs"]
        }
        cheap_tco = by_key[cheap.key]["tco"]
        self.assertIsNotNone(cheap_tco)
        self.assertEqual(
            cheap_tco["topology"]["baseline"]["gpu_hosts"], 2)
        self.assertEqual(
            cheap_tco["topology"]["baseline"]["hbf_hosts"], 0)
        self.assertEqual(
            cheap_tco["topology"]["baseline"]["h100_cards"], 16)
        self.assertEqual(
            cheap_tco["topology"]["baseline"][
                "local_ssd_devices"], 16)
        self.assertEqual(
            cheap_tco["topology"]["proposed"]["gpu_hosts"], 1)
        self.assertEqual(
            cheap_tco["topology"]["proposed"]["hbf_hosts"], 1)
        self.assertEqual(
            cheap_tco["topology"]["proposed"][
                "local_ssd_devices"], 8)
        self.assertEqual(
            rate["performance_tco_pareto_design_keys"],
            [cheap.key],
        )
        self.assertTrue(
            by_key[cheap.key]["performance_tco_pareto"])
        self.assertFalse(
            by_key[dominated.key]["performance_tco_pareto"])

        rejected = copy.deepcopy(records)
        for record in rejected:
            if record["candidate_key"] == BASELINE_CANDIDATE_KEY:
                record["summary"] = _summary(20.0, joint=0.20)
        with self.assertRaisesRegex(
                SSDHBFDesignSweepError,
                "reference eligibility gate failed",
        ):
            aggregate_cell_records(
                rejected, (cheap, dominated))
        audit_only = aggregate_cell_records(
            rejected, (cheap, dominated),
            require_eligibility=False,
        )
        self.assertFalse(
            audit_only["rates"][0][
                "reference_eligibility"]["eligible"])
        self.assertTrue(all(
            row["tco"] is None
            for row in audit_only["rates"][0]["designs"]
        ))

    def test_mock_task_execution_seals_reference_and_design_cells(self):
        schedule = _scheduled_session()

        def task(kind: str, key: str, design=None):
            return _CellTask(
                repo_root=REPO_ROOT,
                candidate_kind=kind,
                candidate_key=key,
                seed=7,
                session_rate=REQUIRED_SESSION_RATE,
                scheduled_sessions=schedule,
                measurement_identities=ROSTER,
                design=design,
                first_ttft_seconds=30.0,
                resume_ttft_seconds=30.0,
                tpot_milliseconds=300.0,
                scenario_contract_sha256="a" * 64,
                execution_inputs_sha256="b" * 64,
            )

        baseline_task = task(
            "baseline", BASELINE_CANDIDATE_KEY)
        design_task = task("design", self.tp4.key, self.tp4)
        with patch(
            "serving.ssd_hbf_design_sweep.run_reference_cell",
            return_value=_record(
                BASELINE_CANDIDATE_KEY, 7, 5.0, joint=0.05),
        ) as reference:
            baseline = _execute_task(baseline_task)
        with patch(
            "serving.ssd_hbf_design_sweep.run_design_cell",
            return_value=_record(
                self.tp4.key, 7, 60.0, joint=0.60),
        ) as design:
            proposed = _execute_task(design_task)

        reference.assert_called_once()
        design.assert_called_once()
        for record in (baseline, proposed):
            self.assertEqual(
                record["cell_contract"][
                    "comparison_contract"],
                SSD_HBF_CONTRACT_KEY,
            )
            unsealed = dict(record)
            observed = unsealed.pop("result_payload_sha256")
            self.assertEqual(
                observed, stable_json_sha256(unsealed))

    def test_resume_requires_contract_payload_and_roster_hashes(self):
        task = _CellTask(
            repo_root=REPO_ROOT,
            candidate_kind="baseline",
            candidate_key=BASELINE_CANDIDATE_KEY,
            seed=7,
            session_rate=REQUIRED_SESSION_RATE,
            scheduled_sessions=_scheduled_session(),
            measurement_identities=ROSTER,
            design=None,
            first_ttft_seconds=30.0,
            resume_ttft_seconds=30.0,
            tpot_milliseconds=300.0,
            scenario_contract_sha256="a" * 64,
            execution_inputs_sha256="b" * 64,
        )
        record = _seal_record(
            task,
            _record(
                BASELINE_CANDIDATE_KEY, 7, 5.0, joint=0.05),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cell.json"
            path.write_text(
                json.dumps(record, allow_nan=False),
                encoding="utf-8",
            )
            self.assertEqual(
                _load_resumable_cell(path, task), record)

            tampered = copy.deepcopy(record)
            tampered["summary"][
                "offered_load_normalized_output_token_goodput"][
                    "value"] = 6.0
            path.write_text(
                json.dumps(tampered, allow_nan=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    SSDHBFDesignSweepError,
                    "payload hash mismatch"):
                _load_resumable_cell(path, task)

            wrong_roster = copy.deepcopy(record)
            wrong_roster["measurement_roster"][
                "ordered_identities_sha256"] = "c" * 64
            unsealed = dict(wrong_roster)
            unsealed.pop("result_payload_sha256")
            wrong_roster["result_payload_sha256"] = (
                stable_json_sha256(unsealed))
            path.write_text(
                json.dumps(wrong_roster, allow_nan=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    SSDHBFDesignSweepError,
                    "measurement roster mismatch"):
                _load_resumable_cell(path, task)

    def test_full_runner_publishes_and_reuses_only_sealed_cells(self):
        tasks = []
        for seed in (7, 11, 13):
            schedule = _scheduled_session(seed=seed)
            common = {
                "repo_root": REPO_ROOT,
                "seed": seed,
                "session_rate": REQUIRED_SESSION_RATE,
                "scheduled_sessions": schedule,
                "measurement_identities": ROSTER,
                "first_ttft_seconds": 30.0,
                "resume_ttft_seconds": 30.0,
                "tpot_milliseconds": 300.0,
                "scenario_contract_sha256": "a" * 64,
                "execution_inputs_sha256": "b" * 64,
            }
            tasks.extend((
                _CellTask(
                    candidate_kind="baseline",
                    candidate_key=BASELINE_CANDIDATE_KEY,
                    design=None,
                    **common,
                ),
                _CellTask(
                    candidate_kind="oracle",
                    candidate_key=ORACLE_CANDIDATE_KEY,
                    design=None,
                    **common,
                ),
                _CellTask(
                    candidate_kind="design",
                    candidate_key=self.tp4.key,
                    design=self.tp4,
                    **common,
                ),
            ))

        def execute(task):
            goodput, joint = {
                "baseline": (5.0, 0.05),
                "oracle": (100.0, 0.98),
                "design": (60.0, 0.60),
            }[task.candidate_kind]
            return _seal_record(
                task,
                _record(
                    task.candidate_key,
                    task.seed,
                    goodput,
                    joint=joint,
                ),
            )

        scenario = _FakeScenario()
        scenario_contract = {
            "scenario_id": "fake-long-cold",
            "manifest_sha256": "a" * 64,
            "measurement_roster_sha256": ROSTER_HASH,
            "measurement_identity_count": 1,
            "required_session_rate": REQUIRED_SESSION_RATE,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            with (
                patch(
                    "serving.ssd_hbf_design_sweep."
                    "validate_scenario_contract",
                    return_value=scenario_contract,
                ),
                patch(
                    "serving.ssd_hbf_design_sweep.build_tasks",
                    return_value=tuple(tasks),
                ),
                patch(
                    "serving.ssd_hbf_design_sweep."
                    "_execution_inputs_sha256",
                    return_value="b" * 64,
                ),
                patch(
                    "serving.ssd_hbf_design_sweep._execute_task",
                    side_effect=execute,
                ) as first_execute,
            ):
                manifest, aggregate_path = run_design_space(
                    repo_root=REPO_ROOT,
                    output_root=output,
                    scenario=scenario,
                    designs=(self.tp4,),
                    seeds=(7, 11, 13),
                    workers=1,
                    require_runtime_energy=False,
                )
            self.assertEqual(first_execute.call_count, 9)
            self.assertTrue(aggregate_path.is_file())
            self.assertTrue(
                (output / "summary.csv").is_file())
            with (output / "summary.csv").open(
                    "r", encoding="utf-8", newline="") as source:
                summary_rows = list(csv.DictReader(source))
            self.assertEqual(len(summary_rows), 1)
            summary = summary_rows[0]
            self.assertEqual(summary["tco_lifetime_years"], "5.0")
            self.assertNotEqual(summary["five_year_tco_usd"], "")
            self.assertNotEqual(summary["baseline_it_power_w"], "")
            self.assertNotEqual(summary["proposed_it_power_w"], "")
            self.assertNotEqual(
                summary["proposed_five_year_facility_energy_kwh"],
                "",
            )
            self.assertEqual(
                summary["hbf_total_write_bytes_across_seeds"],
                "0",
            )
            self.assertEqual(
                summary[
                    "hbf_lifetime_years_100k_pe_waf1"],
                "",
            )
            self.assertEqual(
                summary[
                    "hbf_meets_five_year_endurance_100k_pe_waf1"],
                "True",
            )
            self.assertEqual(
                len(tuple((output / "cells").rglob("*.json"))),
                9,
            )
            self.assertEqual(
                manifest["grid"]["executed_cell_count"], 9)

            with (
                patch(
                    "serving.ssd_hbf_design_sweep."
                    "validate_scenario_contract",
                    return_value=scenario_contract,
                ),
                patch(
                    "serving.ssd_hbf_design_sweep.build_tasks",
                    return_value=tuple(tasks),
                ),
                patch(
                    "serving.ssd_hbf_design_sweep."
                    "_execution_inputs_sha256",
                    return_value="b" * 64,
                ),
                patch(
                    "serving.ssd_hbf_design_sweep._execute_task",
                    side_effect=AssertionError(
                        "sealed resume reran a cell"),
                ) as resumed_execute,
            ):
                resumed, _ = run_design_space(
                    repo_root=REPO_ROOT,
                    output_root=output,
                    scenario=scenario,
                    designs=(self.tp4,),
                    seeds=(7, 11, 13),
                    workers=1,
                    resume=True,
                    require_runtime_energy=False,
                )
            resumed_execute.assert_not_called()
            self.assertEqual(
                resumed["grid"]["resumed_cell_count"], 9)
            self.assertEqual(
                resumed["grid"]["executed_cell_count"], 0)


if __name__ == "__main__":
    unittest.main()

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from serving.core.hbf_comparison_cell import (
    ASTRA_CYCLES_USED,
    D_MAX_NUM_SEQS,
    HBF_LAYOUTS,
    MAX_NUM_BATCHED_TOKENS,
    MAX_PREFILL_CHUNK_TOKENS,
    P_MAX_NUM_SEQS,
    SHARED_MAX_NUM_SEQS,
    SIMULATION_BACKEND,
    SYSTEM_KEYS,
    ComparisonCellError,
    build_slo_thresholds,
    json_safe,
    make_comparison_system,
    run_comparison_cell,
    summarize_measurement_requests,
    validate_causal_release_contract,
    validate_system_call_projection,
    write_cell_output_bundle_atomic,
    write_cell_outputs_atomic,
)
from serving.core.gpu_hbf_hybrid import GPUHBFHybridSystem
from serving.core.gpu_pd_dual_oracle import (
    ROUTE_BALANCED_TRACE_WORK,
    DualStrictInfiniteHBMOracle,
)
from serving.core.gpu_pd_dual_tiered import DualFiniteHBMTieredBaseline
from serving.core.hbf_comparison_metrics import (
    CompletedRequest,
    RequestKey,
    SLOThresholds,
)
from serving.core.hbf_comparison_workload import (
    CallSpec,
    ScheduledSession,
    SessionSpec,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_scheduled_session(
        *,
        source_index,
        offer_index,
        arrival_ns,
        calls,
        unit_interarrival=0.0,
        unit_arrival_time=0.0,
        session_id=None):
    if session_id is None:
        session_id = f"cell-session-{source_index}"
    call_specs = tuple(
        CallSpec(
            session_id=session_id,
            source_index=source_index,
            call_index=call_index,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_duration_ns=tool_duration_ns,
            cached_prefix_tokens=cached_prefix_tokens,
            fresh_input_tokens=input_tokens - cached_prefix_tokens,
            lineage_status=None,
            inter_turn_gap_type=None,
        )
        for call_index, (
            input_tokens,
            output_tokens,
            tool_duration_ns,
            cached_prefix_tokens,
        ) in enumerate(calls)
    )
    return ScheduledSession(
        offer_index=offer_index,
        session=SessionSpec(
            source_index=source_index,
            session_id=session_id,
            source_arrival_time_ns=arrival_ns,
            source_session_identity_sha256=None,
            calls=call_specs,
        ),
        arrival_time_ns=arrival_ns,
        unit_interarrival=unit_interarrival,
        unit_arrival_time=unit_arrival_time,
    )


def tiny_schedule():
    return (
        make_scheduled_session(
            source_index=10,
            offer_index=0,
            arrival_ns=0,
            calls=((32, 2, 1_000_000_000, 0),
                   (40, 2, 0, 33)),
        ),
        make_scheduled_session(
            source_index=11,
            offer_index=1,
            # 0.5 unit seconds scaled at two sessions/second.
            arrival_ns=250_000_000,
            unit_interarrival=0.5,
            unit_arrival_time=0.5,
            calls=((24, 1, 0, 0),),
        ),
    )


class HBFComparisonCellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schedule = tiny_schedule()
        cls.measurement_ids = (
            "cell-session-10::call-0",
            "cell-session-10::call-1",
        )
        cls.results = {
            system_key: run_comparison_cell(
                repo_root=REPO_ROOT,
                system_key=system_key,
                scheduled_sessions=cls.schedule,
                session_rate=2.0,
                measurement_identities=cls.measurement_ids,
            )
            for system_key in SYSTEM_KEYS
        }

    def test_all_systems_share_exact_frozen_identities_and_hashes(self):
        self.assertEqual(set(self.results), set(SYSTEM_KEYS))
        call_spec_hashes = {
            result["frozen_workload"]["call_specs_sha256"]
            for result in self.results.values()
        }
        schedule_hashes = {
            result["frozen_workload"]["schedule_sha256"]
            for result in self.results.values()
        }
        call_set_hashes = {
            result["full_drain"]["calls"]["completion_set_sha256"]
            for result in self.results.values()
        }
        self.assertEqual(len(call_spec_hashes), 1)
        self.assertEqual(len(schedule_hashes), 1)
        self.assertEqual(len(call_set_hashes), 1)
        for result in self.results.values():
            self.assertEqual(
                result["frozen_workload"][
                    "expected_system_call_projection_sha256"],
                result["frozen_workload"][
                    "normalized_system_call_projection_sha256"],
            )

        expected_identities = {
            "cell-session-10::call-0",
            "cell-session-10::call-1",
            "cell-session-11::call-0",
        }
        for system_key, result in self.results.items():
            with self.subTest(system_key=system_key):
                self.assertEqual(
                    result["full_drain"]["calls"]["identity_count"], 3)
                self.assertEqual(
                    {
                        row["completion_identity"]
                        for row in result["requests"]
                    },
                    expected_identities,
                )
                measured = [
                    row for row in result["requests"]
                    if row["is_measurement"]
                ]
                self.assertEqual(
                    [row["completion_identity"] for row in measured],
                    list(self.measurement_ids),
                )
                self.assertEqual(
                    result["summary"]["counts"]["measurement_calls"], 2)
                self.assertEqual(
                    result["summary"]["counts"]["measurement_sessions"], 1)
                self.assertEqual(
                    result["summary"][
                        "offered_load_normalized_request_goodput"
                    ]["label"],
                    "offered-load-normalized request goodput",
                )
                self.assertIn(
                    "observed_completion_span_throughput",
                    result["summary"],
                )
                first_row = result["requests"][0]
                self.assertEqual(first_row["input_tokens"], 32)
                self.assertEqual(first_row["cached_prefix_tokens"], 0)
                self.assertEqual(first_row["fresh_input_tokens"], 32)
                self.assertEqual(
                    first_row["tool_duration_ns"], 1_000_000_000)
                self.assertIsNotNone(first_row["execution_target"])
                self.assertIsNotNone(first_row["execution_node_id"])
                self.assertIsNotNone(first_row["execution_instance_id"])
                self.assertIsNotNone(first_row["execution_policy"])
                self.assertIn(
                    "request_kind_summaries", result["summary"])
                self.assertIn(
                    "resume_tpot_eligible",
                    result["summary"]["latency_distributions_ns"],
                )
        for system_key in (
                "hbf_dp8", "hbf_tp4", "hbf_tp8", "hbf_tp4_wide"):
            rows = self.results[system_key]["requests"]
            self.assertEqual(rows[0]["execution_target"], "gpu")
            self.assertEqual(rows[0]["execution_node_id"], 0)
            self.assertIsNone(rows[0]["execution_group_id"])
            self.assertEqual(rows[1]["execution_target"], "hbf")
            self.assertEqual(rows[1]["execution_node_id"], 1)
            self.assertIsNotNone(rows[1]["execution_group_id"])
            self.assertTrue(
                rows[1]["execution_instance_id"].startswith("hbf-group-"))

    def test_factories_pin_documented_limits_and_topologies(self):
        for system_key in SYSTEM_KEYS:
            with self.subTest(system_key=system_key):
                system = make_comparison_system(
                    repo_root=REPO_ROOT,
                    system_key=system_key,
                )
                self.assertFalse(system.validate_every_event)
                if isinstance(system, GPUHBFHybridSystem):
                    pools = (system.node.gpu_pool,)
                    self.assertEqual(
                        system.node.hbf_pool.max_num_batched_tokens,
                        MAX_NUM_BATCHED_TOKENS,
                    )
                    self.assertEqual(
                        system.node.hbf_pool.max_num_seqs,
                        SHARED_MAX_NUM_SEQS,
                    )
                    self.assertEqual(
                        system.node.hbf_pool.max_prefill_chunk_tokens,
                        MAX_PREFILL_CHUNK_TOKENS,
                    )
                    self.assertEqual(
                        system.node.hbf_layout.key,
                        HBF_LAYOUTS[system_key],
                    )
                else:
                    self.assertIsInstance(
                        system,
                        (DualFiniteHBMTieredBaseline,
                         DualStrictInfiniteHBMOracle),
                    )
                    self.assertEqual(len(system.nodes), 2)
                    self.assertEqual(
                        system.route_policy,
                        ROUTE_BALANCED_TRACE_WORK,
                    )
                    pools = tuple(node.pool for node in system.nodes)
                for pool in pools:
                    self.assertEqual(
                        pool.max_num_batched_tokens,
                        MAX_NUM_BATCHED_TOKENS,
                    )
                    self.assertEqual(
                        pool.max_num_seqs,
                        SHARED_MAX_NUM_SEQS,
                    )
                    self.assertEqual(
                        pool.p_max_num_seqs,
                        P_MAX_NUM_SEQS,
                    )
                    self.assertEqual(
                        pool.d_max_num_seqs,
                        D_MAX_NUM_SEQS,
                    )
                    self.assertEqual(
                        pool.max_prefill_chunk_tokens,
                        MAX_PREFILL_CHUNK_TOKENS,
                    )
                simulation_contract = self.results[system_key][
                    "simulation_contract"]
                backend = simulation_contract["execution_backend"]
                self.assertEqual(
                    backend["name"], SIMULATION_BACKEND)
                self.assertIs(
                    backend["astra_cycles_used"], ASTRA_CYCLES_USED)
                self.assertFalse(ASTRA_CYCLES_USED)
                self.assertIn(
                    "cycle equality is not claimed",
                    backend["astra_conformance_scope"],
                )
                contract = simulation_contract["hardware"]
                gpu_path = (
                    REPO_ROOT
                    / "configs/wakekv_hbf/p4d4_gpu_server.json"
                )
                self.assertEqual(
                    contract["gpu"]["content_sha256"],
                    hashlib.sha256(gpu_path.read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    contract["gpu"]["effective_values"]["gpu_count"], 8)
                if system_key.startswith("hbf_"):
                    hbf_config_name = (
                        "full_model_8card_server_wide_lpddr.json"
                        if system_key == "hbf_tp4_wide"
                        else "full_model_8card_server.json"
                    )
                    hbf_path = (
                        REPO_ROOT
                        / "configs/wakekv_hbf"
                        / hbf_config_name
                    )
                    self.assertEqual(
                        contract["hbf"]["content_sha256"],
                        hashlib.sha256(hbf_path.read_bytes()).hexdigest(),
                    )
                    self.assertEqual(
                        contract["hbf"]["effective_values"]["card_count"], 8)
                else:
                    self.assertIsNone(contract["hbf"])

    def test_wide_lpddr_is_one_explicit_variant_of_legacy_key_set(self):
        legacy_keys = (
            "recompute",
            "ssd_direct",
            "cpu_ssd",
            "oracle",
            "hbf_dp8",
            "hbf_tp4",
            "hbf_tp8",
        )
        self.assertEqual(SYSTEM_KEYS, legacy_keys + ("hbf_tp4_wide",))

        default_path = (
            REPO_ROOT
            / "configs/wakekv_hbf/full_model_8card_server.json"
        )
        wide_path = (
            REPO_ROOT
            / "configs/wakekv_hbf/full_model_8card_server_wide_lpddr.json"
        )
        for system_key in legacy_keys[4:]:
            with self.subTest(system_key=system_key):
                contract = self.results[system_key][
                    "simulation_contract"]["hardware"]["hbf"]
                self.assertEqual(
                    contract["repo_relative_path"],
                    "configs/wakekv_hbf/full_model_8card_server.json",
                )
                self.assertEqual(
                    contract["content_sha256"],
                    hashlib.sha256(default_path.read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    contract["effective_values"][
                        "lpddr_bandwidth_gbps_per_card"],
                    204.8,
                )

        wide_contract = self.results["hbf_tp4_wide"][
            "simulation_contract"]["hardware"]["hbf"]
        self.assertEqual(
            wide_contract["repo_relative_path"],
            "configs/wakekv_hbf/full_model_8card_server_wide_lpddr.json",
        )
        self.assertEqual(
            wide_contract["content_sha256"],
            hashlib.sha256(wide_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            wide_contract["effective_values"][
                "lpddr_bandwidth_gbps_per_card"],
            409.6,
        )
        wide_system = make_comparison_system(
            repo_root=REPO_ROOT,
            system_key="hbf_tp4_wide",
        )
        self.assertEqual(wide_system.node.hbf_layout.key, "tp4")
        self.assertEqual(
            wide_system.node.hbf_hardware.lpddr_bandwidth_gbps_per_card,
            409.6,
        )

    def test_slo_boundaries_and_one_token_tpot_semantics(self):
        threshold = SLOThresholds(
            first_ttft_ns=30_000_000_000,
            resume_ttft_ns=30_000_000_000,
            tpot_ns=300_000_000,
        )
        requests = (
            CompletedRequest(
                key=RequestKey("a", 0),
                release_ns=0,
                first_token_ns=30_000_000_000,
                completion_ns=30_000_000_000,
                output_tokens=1,
            ),
            CompletedRequest(
                key=RequestKey("b", 1),
                release_ns=0,
                first_token_ns=30_000_000_000,
                completion_ns=30_300_000_000,
                output_tokens=2,
            ),
            CompletedRequest(
                key=RequestKey("c", 1),
                release_ns=0,
                first_token_ns=30_000_000_001,
                completion_ns=30_300_000_002,
                output_tokens=2,
            ),
        )
        summary = summarize_measurement_requests(
            requests,
            session_rate=3.0,
            thresholds=threshold,
        )
        self.assertEqual(
            summary["slo"]["first_ttft_pass_fraction"], 1.0)
        self.assertEqual(
            summary["slo"]["resume_ttft_pass_fraction"], 0.5)
        self.assertEqual(
            summary["slo"]["tpot_pass_fraction_of_eligible"], 0.5)
        self.assertEqual(summary["slo"]["all_slo_pass_count"], 2)
        self.assertEqual(
            summary["counts"]["tpot_eligible_calls"], 2)
        self.assertEqual(
            summary["latency_distributions_ns"][
                "resume_ttft"]["p95_ns"],
            30_000_000_001.0,
        )
        self.assertAlmostEqual(
            summary["offered_load_normalized_request_goodput"]["value"],
            2.0,
        )
        self.assertEqual(
            summary["request_kind_summaries"]["resume"][
                "slo"]["joint_pass_count"],
            1,
        )
        self.assertEqual(
            summary["observed_completion_span_throughput"][
                "inter_completion_interval_count"],
            2,
        )
        self.assertIn(
            "(N-1)",
            summary["observed_completion_span_throughput"]["semantics"],
        )
        converted = build_slo_thresholds()
        self.assertEqual(converted.first_ttft_ns, 30_000_000_000)
        self.assertEqual(converted.resume_ttft_ns, 30_000_000_000)
        self.assertEqual(converted.tpot_ns, 300_000_000)

    def test_strict_json_and_atomic_json_csv_outputs(self):
        result = self.results["hbf_tp4"]
        json.dumps(result, allow_nan=False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path, csv_path = write_cell_outputs_atomic(
                json_path=root / "nested" / "cell.json",
                csv_path=root / "nested" / "requests.csv",
                result=result,
            )
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["system_key"], "hbf_tp4")
            with csv_path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 3)
            self.assertFalse(any(
                path.suffix == ".tmp"
                for path in (root / "nested").iterdir()
            ))

            bundle = write_cell_output_bundle_atomic(
                root / "bundle", result)
            self.assertEqual(
                json.loads(
                    (bundle / "cell.json").read_text(encoding="utf-8")
                )["system_key"],
                "hbf_tp4",
            )
            with (bundle / "requests.csv").open(
                    newline="", encoding="utf-8") as source:
                self.assertEqual(len(list(csv.DictReader(source))), 3)
            with self.assertRaises(FileExistsError):
                write_cell_output_bundle_atomic(bundle, result)

            existing_json = root / "existing.json"
            existing_csv = root / "existing.csv"
            existing_json.write_text(
                '{"old": true}\n', encoding="utf-8")
            existing_csv.write_text(
                "old,csv\n1,2\n", encoding="utf-8")
            invalid_result = dict(result)
            invalid_result["requests"] = [{"system_key": "hbf_tp4"}]
            with self.assertRaises(ComparisonCellError):
                write_cell_outputs_atomic(
                    json_path=existing_json,
                    csv_path=existing_csv,
                    result=invalid_result,
                )
            self.assertEqual(
                existing_json.read_text(encoding="utf-8"),
                '{"old": true}\n',
            )
            self.assertEqual(
                existing_csv.read_text(encoding="utf-8"),
                "old,csv\n1,2\n",
            )
            same_path = root / "same-output"
            same_path.write_text("unchanged\n", encoding="utf-8")
            with self.assertRaisesRegex(
                    ComparisonCellError, "distinct files"):
                write_cell_outputs_atomic(
                    json_path=same_path,
                    csv_path=same_path,
                    result=result,
                )
            self.assertEqual(
                same_path.read_text(encoding="utf-8"),
                "unchanged\n",
            )

            invalid_bundle = root / "invalid-bundle"
            with self.assertRaises(ComparisonCellError):
                write_cell_output_bundle_atomic(
                    invalid_bundle, invalid_result)
            self.assertFalse(invalid_bundle.exists())
            self.assertFalse(any(
                path.name.startswith(".invalid-bundle.")
                for path in root.iterdir()
            ))

        with self.assertRaises(ComparisonCellError):
            json_safe(float("nan"))

    def test_frozen_tuple_and_roster_fail_closed(self):
        with self.assertRaises(TypeError):
            run_comparison_cell(
                repo_root=REPO_ROOT,
                system_key="oracle",
                scheduled_sessions=list(self.schedule),
                session_rate=1.0,
            )
        with self.assertRaises(ComparisonCellError):
            run_comparison_cell(
                repo_root=REPO_ROOT,
                system_key="oracle",
                scheduled_sessions=self.schedule,
                session_rate=2.0,
                measurement_identities=("missing::call-0",),
            )
        with self.assertRaisesRegex(
                ComparisonCellError,
                "inconsistent with session_rate"):
            run_comparison_cell(
                repo_root=REPO_ROOT,
                system_key="oracle",
                scheduled_sessions=self.schedule,
                session_rate=1.0,
            )
        with self.assertRaisesRegex(
                ComparisonCellError,
                "offer_index order"):
            validate_causal_release_contract(
                tuple(reversed(self.schedule)),
                (),
            )
        broken_coordinate = (
            self.schedule[0],
            replace(
                self.schedule[1],
                unit_arrival_time=0.75,
            ),
        )
        with self.assertRaisesRegex(
                ComparisonCellError, "cumulative"):
            validate_causal_release_contract(
                broken_coordinate, ())

    def test_causal_release_and_output_contract_fail_closed(self):
        first_completion = 100
        resume_release = first_completion + 1_000_000_000
        completed = (
            CompletedRequest(
                key=RequestKey("cell-session-10", 0),
                release_ns=0,
                first_token_ns=90,
                completion_ns=first_completion,
                output_tokens=2,
            ),
            CompletedRequest(
                key=RequestKey("cell-session-10", 1),
                release_ns=resume_release,
                first_token_ns=resume_release + 10,
                completion_ns=resume_release + 20,
                output_tokens=2,
            ),
            CompletedRequest(
                key=RequestKey("cell-session-11", 0),
                release_ns=250_000_000,
                first_token_ns=250_000_010,
                completion_ns=250_000_010,
                output_tokens=1,
            ),
        )
        drain = validate_causal_release_contract(
            self.schedule, completed)
        self.assertEqual(drain["identity_count"], 3)

        bad_release = (
            completed[0],
            replace(completed[1], release_ns=resume_release + 1),
            completed[2],
        )
        with self.assertRaisesRegex(
                ComparisonCellError, "causal successor timing"):
            validate_causal_release_contract(
                self.schedule, bad_release)

        bad_first_release = (
            replace(completed[0], release_ns=1),
            completed[1],
            completed[2],
        )
        with self.assertRaisesRegex(
                ComparisonCellError, "scheduled arrival"):
            validate_causal_release_contract(
                self.schedule, bad_first_release)

        bad_output = (
            replace(completed[0], output_tokens=3),
            completed[1],
            completed[2],
        )
        with self.assertRaisesRegex(
                ComparisonCellError, "output-token mismatch"):
            validate_causal_release_contract(
                self.schedule, bad_output)

    def test_system_call_projection_fails_closed_on_consumed_work_mutation(
            self):
        system = make_comparison_system(
            repo_root=REPO_ROOT,
            system_key="oracle",
        )
        system.load(self.schedule)
        digest = validate_system_call_projection(
            self.schedule, system.call_specs)
        self.assertEqual(len(digest), 64)

        changed_input = (
            replace(
                system.call_specs[0],
                input_tokens=system.call_specs[0].input_tokens + 1,
            ),
            *system.call_specs[1:],
        )
        with self.assertRaisesRegex(
                ComparisonCellError, "input_tokens"):
            validate_system_call_projection(
                self.schedule, changed_input)

        changed_tool_gap = (
            replace(
                system.call_specs[0],
                tool_duration_ns=(
                    system.call_specs[0].tool_duration_ns + 1),
            ),
            *system.call_specs[1:],
        )
        with self.assertRaisesRegex(
                ComparisonCellError, "tool_duration_ns"):
            validate_system_call_projection(
                self.schedule, changed_tool_gap)


if __name__ == "__main__":
    unittest.main()

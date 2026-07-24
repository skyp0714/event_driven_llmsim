import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from serving.core.session_metrics import (
    _active_population,
    _timing_validation,
    build_session_metrics,
    save_session_metrics,
)


def _request(request_id, sub_index, source, end_time, arrival=None):
    if arrival is None:
        arrival = end_time - 100
    latency = end_time - arrival
    ttft = min(60, latency)
    remaining = latency - ttft
    base, extra = divmod(remaining, 9)
    itl = [base + (1 if index < extra else 0) for index in range(9)]
    restore_ns = 10 if sub_index else 0
    return SimpleNamespace(
        id=request_id,
        session_id="session-0",
        sub_request_index=sub_index,
        end_time=end_time,
        arrival=arrival,
        input=100,
        original_input=100,
        output=110,
        requested_output_tokens=10,
        generated_tokens=10,
        latency=latency,
        queuing_delay=20,
        first_schedule_time_ns=arrival + 20,
        first_schedule_eligibility_time_ns=arrival + 15,
        first_schedule_request_ready_time_ns=arrival,
        first_schedule_resource_ready_time_ns=arrival + 15,
        scheduler_queue_wait_ns=5,
        ttft=ttft,
        tpot=remaining // 9,
        itl=itl,
        source_session_id="session-0",
        session_template_index=0,
        session_epoch=0,
        return_gap_type=("tool" if sub_index else "session_start"),
        agentic_kv_residency_at_return=(source if sub_index else None),
        agentic_kv_source=(source if sub_index else None),
        agentic_kv_restore_gate_wait_ns=(10 if sub_index else 0),
        agentic_kv_restore_gate_start_ns=(arrival if sub_index else 0),
        agentic_kv_async_decode_join=False,
        agentic_kv_overlap_cutoff_tokens=None,
        agentic_kv_hit_tokens=(50 if sub_index else 0),
        agentic_kv_recompute_tokens=0,
        agentic_kv_restored_tokens_discarded_by_active_prefill_recompute=0,
        agentic_kv_hbm_admission_wait_ns=(2 if sub_index else 0),
        agentic_kv_transient_dram_capacity_wait_ns=(
            1 if sub_index and source == "ssd" else 0),
        agentic_kv_restore_queue_wait_ns=(3 if sub_index else 0),
        agentic_kv_restore_service_ns=(5 if sub_index else 0),
        agentic_kv_restore_ns=restore_ns,
        agentic_kv_owner_gate_ns=restore_ns,
        pd_pair_fifo_wait_ns=0,
        agentic_kv_prepare_boundary_wait_ns=0,
        agentic_kv_source_demotion_join_wait_ns=0,
        agentic_kv_restore_issue_time_ns=arrival,
        agentic_kv_target_hbm_ready_time_ns=(
            arrival + (2 if sub_index else 0)),
        agentic_kv_restore_ready_time_ns=(arrival + restore_ns),
        return_gap_ns=0,
        prefix_reuse_tokens=(50 if sub_index else 0),
        pd_launch_admission_wait_ns=(7 if sub_index else 0),
        pd_launch_admission_critical_wait_ns=(6 if sub_index else 0),
        pd_chunk_admission_count=0,
        pd_chunk_cancelled_admission_count=0,
        pd_chunk_admission_wait_ns_total=0,
        pd_chunk_admission_critical_wait_ns_total=0,
        pd_chunk_successful_admission_wait_ns_total=0,
        pd_chunk_successful_admission_critical_wait_ns_total=0,
        pd_chunk_cancelled_admission_wait_ns_total=0,
        pd_chunk_cancelled_admission_critical_wait_ns_total=0,
        active_prefill_recompute_preemptions=0,
        active_prefill_recompute_tokens=0,
        active_prefill_recompute_frontier_tokens=0,
        pd_active_prefill_recompute_generation=0,
    )


class _Router:
    session_admission = SimpleNamespace(
        warmup_completions=0,
        measure_completions=0,
        max_active_sessions=1,
    )

    def session_admission_summary(self):
        return {
            "mode": "backlog",
            "completed_sessions": 1,
        }

    def session_lifecycle_records(self):
        return [{
            "session_id": "session-0",
            "status": "completed",
            "offered_time_ns": 0,
            "admission_time_ns": 10,
            "admission_queue_wait_ns": 10,
            "completion_time_ns": 310,
            "e2e_ns": 300,
        }]


class SessionMetricsTests(unittest.TestCase):
    def test_long_request_warning_has_a_stable_machine_code(self):
        end_time_ns = 3_600_000_000_011
        request = _request(
            0, 0, None, end_time_ns, arrival=10)
        lifecycle = {
            "session-0": {
                "session_id": "session-0",
                "status": "completed",
                "offered_time_ns": 0,
                "admission_time_ns": 10,
                "admission_queue_wait_ns": 10,
                "completion_time_ns": end_time_ns,
                "e2e_ns": end_time_ns - 10,
            },
        }

        validation = _timing_validation(
            [request], end_time_ns, lifecycle)

        self.assertTrue(validation["passed"])
        self.assertEqual(validation["violations"], [])
        self.assertEqual(
            validation["warnings"],
            [
                "request=0, session=session-0: request latency exceeds "
                "one hour",
            ],
        )
        self.assertEqual(
            validation["warning_codes"],
            ["request_latency_over_one_hour"],
        )

    def test_active_population_counts_live_and_censored_sessions(self):
        population = _active_population(
            [
                {
                    "admission_time_ns": 0,
                    "completion_time_ns": 200,
                },
                {
                    "admission_time_ns": 50,
                    "completion_time_ns": None,
                },
                {
                    "admission_time_ns": 75,
                    "completion_time_ns": None,
                    "censored_time_ns": 250,
                },
            ],
            start_ns=100,
            end_ns=300,
            configured_k=3,
        )

        self.assertEqual(population["peak_active_sessions"], 3)
        self.assertEqual(population["mean_active_sessions"], 2.25)
        self.assertEqual(population["fraction_at_configured_k"], 0.5)

    def test_active_population_clips_lifecycles_to_window(self):
        population = _active_population(
            [
                {
                    "admission_time_ns": 0,
                    "completion_time_ns": 400,
                },
                {
                    "admission_time_ns": 150,
                    "completion_time_ns": 250,
                },
            ],
            start_ns=100,
            end_ns=300,
            configured_k=2,
        )

        self.assertEqual(population["peak_active_sessions"], 2)
        self.assertEqual(population["mean_active_sessions"], 1.5)
        self.assertEqual(population["fraction_at_configured_k"], 0.5)

    def test_active_population_ignores_invalid_and_zero_intervals(self):
        population = _active_population(
            [
                {
                    "admission_time_ns": None,
                    "completion_time_ns": None,
                },
                {
                    "admission_time_ns": 125,
                    "completion_time_ns": 125,
                },
                {
                    "admission_time_ns": 175,
                    "completion_time_ns": 150,
                },
                {
                    "admission_time_ns": 225,
                    "completion_time_ns": None,
                    "censored_time_ns": 200,
                },
                {
                    "admission_time_ns": 300,
                    "completion_time_ns": None,
                },
                {
                    "admission_time_ns": 125,
                    "completion_time_ns": 175,
                },
            ],
            start_ns=100,
            end_ns=300,
            configured_k=1,
        )

        self.assertEqual(population["peak_active_sessions"], 1)
        self.assertEqual(population["mean_active_sessions"], 0.25)
        self.assertEqual(population["fraction_at_configured_k"], 0.25)

    def test_active_population_handles_invalid_measurement_window(self):
        expected = {
            "mean_active_sessions": None,
            "peak_active_sessions": None,
            "fraction_at_configured_k": None,
        }
        lifecycle = [{
            "admission_time_ns": 0,
            "completion_time_ns": None,
        }]

        self.assertEqual(_active_population(lifecycle, None, 100, 1), expected)
        self.assertEqual(_active_population(lifecycle, 100, None, 1), expected)
        self.assertEqual(_active_population(lifecycle, 100, 100, 1), expected)
        self.assertEqual(_active_population(lifecycle, 101, 100, 1), expected)

    def test_pre_admission_resume_cannot_schedule_before_restore_ready(self):
        request = _request(0, 1, "cpu", 200)
        request.agentic_kv_restore_ready_time_ns = (
            request.first_schedule_time_ns + 1)
        scheduler = SimpleNamespace(pd_type="decode", done=[request])

        with self.assertRaisesRegex(
                RuntimeError, "scheduled before restore-ready"):
            build_session_metrics(
                _Router(), [scheduler], 220,
                dataset="workload.jsonl", run_id="bad-dependency")

    def test_report_uses_all_completed_requests_as_resume_denominator(self):
        cpu_resume = _request(1, 1, "cpu", 210)
        cpu_resume.active_prefill_recompute_preemptions = 1
        cpu_resume.active_prefill_recompute_tokens = 60
        cpu_resume.active_prefill_recompute_frontier_tokens = 60
        cpu_resume.pd_active_prefill_recompute_generation = 1
        cpu_resume.agentic_kv_restored_tokens_discarded_by_active_prefill_recompute = (
            50)
        cpu_resume.pd_chunk_cancelled_admission_count = 1
        cpu_resume.pd_chunk_cancelled_admission_wait_ns_total = 4
        cpu_resume.pd_chunk_cancelled_admission_critical_wait_ns_total = 3
        cpu_resume.pd_chunk_admission_wait_ns_total = 4
        cpu_resume.pd_chunk_admission_critical_wait_ns_total = 3
        ssd_resume = _request(2, 2, "ssd", 310)
        ssd_resume.return_gap_type = "human"
        scheduler = SimpleNamespace(
            pd_type="decode",
            done=[
                _request(0, 0, None, 110),
                cpu_resume,
                ssd_resume,
            ],
        )
        prefill = SimpleNamespace(
            pd_type="prefill",
            done=[_request(0, 0, None, 50)],
        )
        report = build_session_metrics(
            _Router(), [prefill, scheduler], 320,
            dataset="workload.jsonl", run_id="test-run",
        )

        self.assertEqual(report["schema_version"], 11)
        self.assertIsNone(report["hbm_kv_occupancy"])
        self.assertEqual(
            report["validation"]["timing"]["warning_codes"], [])
        self.assertEqual(report["throughput"]["completed_sessions"], 1)
        self.assertEqual(report["throughput"]["completed_requests"], 3)
        self.assertEqual(report["requests"]["resume"]["count"], 2)
        self.assertAlmostEqual(
            report["requests"][
                "resume_source_fractions_of_all_requests"
            ]["cpu"],
            1 / 3,
        )
        self.assertAlmostEqual(
            report["requests"][
                "resume_source_fractions_of_resume_requests"
            ]["ssd"],
            1 / 2,
        )
        self.assertEqual(
            report["requests"]["attempted_physical_resume_count"], 2)
        self.assertEqual(
            report["requests"]["effective_surviving_resume_count"], 1)
        self.assertAlmostEqual(
            report["requests"][
                "attempted_physical_resume_fractions_of_all_requests"
            ]["cpu"],
            1 / 3,
        )
        self.assertEqual(
            report["requests"][
                "effective_surviving_resume_fractions_of_all_requests"
            ]["cpu"],
            0,
        )
        self.assertEqual(
            report["requests"][
                "effective_surviving_resume_by_return_gap_type_and_source"
            ]["human"]["ssd"]["count"],
            1,
        )
        self.assertNotIn(
            "tool",
            report["requests"][
                "effective_surviving_resume_by_return_gap_type_and_source"
            ],
        )
        self.assertEqual(
            report["requests"][
                "attempted_physical_resume_by_return_gap_type_and_source"
            ]["tool"]["cpu"]["count"],
            1,
        )
        self.assertEqual(
            report["requests"][
                "attempted_physical_resume_by_return_gap_type_and_source"
            ]["human"]["ssd"]["count"],
            1,
        )
        self.assertEqual(
            report["requests"]["resume_reuse_token_accounting"],
            {
                "attempted_restored_hit_tokens": 100,
                "restored_hit_tokens_discarded_by_active_prefill_recompute": 50,
                "effective_surviving_hit_tokens": 50,
                "conservation_passed": True,
            },
        )
        self.assertEqual(
            report["sessions"]["admission_queue_wait_ns"]["sum"], 10
        )
        self.assertEqual(
            report["sessions"]["e2e_from_offer_ns"]["sum"], 310
        )
        self.assertEqual(
            report["sessions"]["offered_arrival_trace_count"], 1
        )
        self.assertEqual(
            len(report["sessions"]["offered_arrival_trace_sha256"]), 64
        )
        self.assertEqual(
            report["requests"]["resume_by_return_gap_type"]["tool"][
                "ttft_ns"
            ]["count"],
            1,
        )
        self.assertEqual(
            report["requests"]["resume_by_return_gap_type"]["human"][
                "ttft_ns"
            ]["count"],
            1,
        )
        self.assertEqual(
            report["requests"][
                "resume_by_return_gap_type_and_source"
            ]["tool"]["cpu"]["count"],
            1,
        )
        self.assertEqual(
            report["requests"]["all"]["scheduler_queue_wait_ns"]["sum"],
            15,
        )
        self.assertEqual(
            report["active_session_population"]["peak_active_sessions"], 1
        )
        records = report["requests"]["records"]
        self.assertEqual(
            [(row["sub_request_index"], row["input_tokens"],
              row["requested_output_tokens"], row["generated_tokens"])
             for row in records],
            [(0, 100, 10, 10), (1, 100, 10, 10), (2, 100, 10, 10)],
        )
        self.assertEqual(records[1]["agentic_kv_source"], "cpu")
        self.assertEqual(records[2]["agentic_kv_source"], "ssd")
        self.assertEqual(
            report["requests"]["resume"][
                "transient_dram_capacity_wait_ns"]["sum"],
            1,
        )
        self.assertEqual(
            records[2]["agentic_kv_transient_dram_capacity_wait_ns"], 1)

    def test_zero_overlap_source_label_is_not_a_physical_resume_or_drop(self):
        initial = _request(0, 0, None, 110)
        zero_overlap = _request(1, 1, "dropped", 310, arrival=110)
        zero_overlap.prefix_reuse_tokens = 0
        zero_overlap.agentic_kv_hit_tokens = 0
        zero_overlap.agentic_kv_recompute_tokens = 0
        zero_overlap.agentic_kv_restore_ns = 0
        zero_overlap.agentic_kv_owner_gate_ns = 0
        zero_overlap.agentic_kv_hbm_admission_wait_ns = 0
        zero_overlap.agentic_kv_restore_queue_wait_ns = 0
        zero_overlap.agentic_kv_restore_service_ns = 0
        zero_overlap.agentic_kv_restore_issue_time_ns = zero_overlap.arrival
        zero_overlap.agentic_kv_target_hbm_ready_time_ns = zero_overlap.arrival
        zero_overlap.agentic_kv_restore_ready_time_ns = zero_overlap.arrival
        scheduler = SimpleNamespace(
            pd_type="decode", done=[initial, zero_overlap])
        prefill = SimpleNamespace(
            pd_type="prefill", done=[_request(0, 0, None, 50)])

        report = build_session_metrics(
            _Router(), [prefill, scheduler], 320,
            dataset="workload.jsonl", run_id="zero-overlap",
        )

        requests = report["requests"]
        self.assertEqual(
            requests["resume_by_source"]["dropped"]["count"], 1)
        self.assertEqual(requests["attempted_physical_resume_count"], 0)
        self.assertEqual(requests["effective_surviving_resume_count"], 0)
        self.assertEqual(requests["kv_state_unavailable_resume_count"], 0)
        self.assertEqual(requests["zero_overlap_resume_count"], 1)
        self.assertEqual(
            requests["resume_reuse_token_accounting"][
                "attempted_restored_hit_tokens"],
            0,
        )

    def test_report_clips_hbm_occupancy_to_measurement_window(self):
        scheduler = SimpleNamespace(
            pd_type="decode",
            done=[
                _request(0, 0, None, 110),
                _request(1, 1, "cpu", 210),
                _request(2, 2, "ssd", 310),
            ],
        )

        class _Occupancy:
            def __init__(self):
                self.window = None

            def summary(self, start_ns, end_ns):
                self.window = (start_ns, end_ns)
                return {
                    "schema_version": 1,
                    "window_start_ns": start_ns,
                    "window_end_ns": end_ns,
                    "conservation": {"passed": True},
                }

        occupancy = _Occupancy()
        report = build_session_metrics(
            _Router(), [scheduler], 320,
            hbm_occupancy_accounting=occupancy,
        )
        self.assertEqual(occupancy.window, (10, 310))
        self.assertEqual(
            report["hbm_kv_occupancy"]["window_start_ns"], 10)
        self.assertEqual(
            report["hbm_kv_occupancy"]["window_end_ns"], 310)

    def test_admission_order_uses_the_exact_fixed_target_and_bounds(self):
        lifecycle = [
            {
                "session_id": "target-a",
                "status": "completed",
                "offered_time_ns": 0,
                "admission_time_ns": 10,
                "admission_queue_wait_ns": 10,
                "completion_time_ns": 500,
                "e2e_ns": 490,
                "planned_admission_index": 0,
                "admission_index": 0,
                "measurement_target": True,
            },
            {
                "session_id": "target-b",
                "status": "completed",
                "offered_time_ns": 0,
                "admission_time_ns": 20,
                "admission_queue_wait_ns": 20,
                "completion_time_ns": 100,
                "e2e_ns": 80,
                "planned_admission_index": 1,
                "admission_index": 1,
                "measurement_target": True,
            },
            {
                "session_id": "non-target",
                "status": "completed",
                "offered_time_ns": 0,
                "admission_time_ns": 30,
                "admission_queue_wait_ns": 30,
                "completion_time_ns": 200,
                "e2e_ns": 170,
                "planned_admission_index": 2,
                "admission_index": 2,
                "measurement_target": False,
            },
        ]

        class AdmissionOrderRouter:
            session_admission = SimpleNamespace(
                warmup_completions=0,
                measure_completions=2,
                max_active_sessions=2,
                measurement_cohort_selection="admission_order",
            )

            def session_lifecycle_records(self):
                return lifecycle

            def measurement_target_session_ids(self):
                return ("target-a", "target-b")

            def session_admission_summary(self):
                return {
                    "mode": "backlog",
                    "measurement_cohort_selection": "admission_order",
                    "measurement_target_session_count": 2,
                    "measurement_target_completed_sessions": 2,
                }

        first = _request(0, 0, None, 500, arrival=10)
        first.session_id = "target-a"
        first.source_session_id = "target-a"
        second = _request(1, 0, None, 100, arrival=20)
        second.session_id = "target-b"
        second.source_session_id = "target-b"
        scheduler = SimpleNamespace(pd_type="decode", done=[first, second])

        report = build_session_metrics(
            AdmissionOrderRouter(), [scheduler], 500,
            dataset="workload.jsonl", run_id="fixed-cohort",
        )

        window = report["measurement_window"]
        self.assertEqual(
            window["measurement_target_session_ids"],
            ["target-a", "target-b"],
        )
        self.assertEqual(window["measurement_target_session_count"], 2)
        self.assertEqual(window["measurement_target_completed_sessions"], 2)
        self.assertEqual(window["measurement_start_ns"], 10)
        self.assertEqual(window["measurement_end_ns"], 500)
        self.assertEqual(window["measurement_duration_ns"], 490)
        expected_hash = hashlib.sha256(json.dumps(
            ["target-a", "target-b"],
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        self.assertEqual(
            window["measurement_target_session_ids_hash"], expected_hash)
        included = {
            row["session_id"]: row["measurement_included"]
            for row in report["sessions"]["records"]
        }
        self.assertEqual(included, {
            "target-a": True,
            "target-b": True,
            "non-target": False,
        })
        self.assertEqual(report["requests"]["all"]["count"], 2)

    def test_admission_prefix_warmup_can_overlap_but_never_enters_metrics(self):
        warmup_id = "slow-warmup"
        target_id = "fast-target"
        lifecycle = [
            {
                "session_id": warmup_id,
                "status": "completed",
                "offered_time_ns": 0,
                "admission_time_ns": 10,
                "admission_queue_wait_ns": 10,
                "completion_time_ns": 500,
                "e2e_ns": 490,
                "planned_admission_index": 0,
                "admission_index": 0,
            },
            {
                "session_id": target_id,
                "status": "completed",
                "offered_time_ns": 0,
                "admission_time_ns": 20,
                "admission_queue_wait_ns": 20,
                "completion_time_ns": 100,
                "e2e_ns": 80,
                "planned_admission_index": 1,
                "admission_index": 1,
            },
        ]

        class FixedPrefixRouter:
            session_admission = SimpleNamespace(
                warmup_completions=1,
                measure_completions=1,
                max_active_sessions=2,
                measurement_cohort_selection="admission_order",
            )

            def session_lifecycle_records(self):
                return lifecycle

            def measurement_warmup_session_ids(self):
                return (warmup_id,)

            def measurement_target_session_ids(self):
                return (target_id,)

            def measurement_required_session_ids(self):
                return (warmup_id, target_id)

            def session_admission_summary(self):
                return {
                    "mode": "backlog",
                    "measurement_cohort_selection": "admission_order",
                    "measurement_warmup_session_count": 1,
                    "measurement_warmup_completed_sessions": 1,
                    "measurement_target_session_count": 1,
                    "measurement_target_completed_sessions": 1,
                    "measurement_required_session_count": 2,
                    "measurement_required_completed_sessions": 2,
                    "measurement_prefix_id_overlap_count": 0,
                }

        warmup_request = _request(0, 0, None, 500, arrival=10)
        warmup_request.session_id = warmup_id
        warmup_request.source_session_id = warmup_id
        target_request = _request(1, 0, None, 100, arrival=20)
        target_request.session_id = target_id
        target_request.source_session_id = target_id
        scheduler = SimpleNamespace(
            pd_type="decode", done=[warmup_request, target_request])

        report = build_session_metrics(
            FixedPrefixRouter(), [scheduler], 500,
            dataset="workload.jsonl", run_id="fixed-prefix",
        )

        window = report["measurement_window"]
        self.assertEqual(window["measurement_warmup_session_ids"], [warmup_id])
        self.assertEqual(window["measurement_target_session_ids"], [target_id])
        self.assertEqual(
            window["measurement_required_session_ids"],
            [warmup_id, target_id],
        )
        self.assertTrue(window["measurement_boundary_complete"])
        self.assertEqual(window["warmup_completion_boundary_ns"], 500)
        self.assertEqual(
            window["target_admitted_before_warmup_complete_session_count"],
            1,
        )
        self.assertEqual(
            window["target_completed_before_warmup_complete_session_count"],
            1,
        )
        self.assertTrue(
            window["target_execution_overlapped_unfinished_warmup"])
        self.assertEqual(report["requests"]["all"]["count"], 1)
        self.assertEqual(report["throughput"]["completed_requests"], 1)
        records = {
            row["session_id"]: row
            for row in report["sessions"]["records"]
        }
        self.assertFalse(records[warmup_id]["measurement_included"])
        self.assertEqual(
            records[warmup_id]["measurement_role"],
            "fixed_admission_prefix_warmup",
        )
        self.assertTrue(records[target_id]["measurement_included"])

    def test_hbm_local_pair_and_boundary_wait_are_not_physical_restore(self):
        initial = _request(0, 0, None, 110, arrival=10)
        resume = _request(1, 1, "hbm", 310, arrival=110)
        resume.pd_pair_fifo_wait_ns = 7
        resume.agentic_kv_prepare_boundary_wait_ns = 8
        resume.agentic_kv_restore_ns = 0
        resume.agentic_kv_owner_gate_ns = 15
        resume.agentic_kv_hbm_admission_wait_ns = 0
        resume.agentic_kv_restore_queue_wait_ns = 0
        resume.agentic_kv_restore_service_ns = 0
        resume.agentic_kv_restore_issue_time_ns = 125
        resume.agentic_kv_target_hbm_ready_time_ns = 125
        resume.agentic_kv_restore_ready_time_ns = 125
        scheduler = SimpleNamespace(
            pd_type="decode", done=[initial, resume])

        report = build_session_metrics(
            _Router(), [scheduler], 320,
            dataset="workload.jsonl", run_id="pair-boundary")

        self.assertEqual(
            report["requests"]["resume"]["pd_pair_fifo_wait_ns"]["sum"],
            7,
        )
        self.assertEqual(
            report["requests"]["resume"][
                "prepare_boundary_wait_ns"]["sum"],
            8,
        )
        self.assertEqual(
            report["requests"]["resume"]["hbm_admission_wait_ns"]["sum"],
            0,
        )
        self.assertEqual(
            report["requests"]["resume"]["owner_ready_gate_ns"]["sum"],
            15,
        )
        overhead = report["overhead_denominators"][
            "measured_session_cohort"]
        self.assertEqual(overhead["pd_pair_fifo_wait_ns"], 7)
        self.assertEqual(overhead["prepare_boundary_wait_ns"], 8)

    def test_source_demotion_join_is_separate_from_physical_restore(self):
        initial = _request(0, 0, None, 110, arrival=10)
        resume = _request(1, 1, "cpu", 310, arrival=110)
        resume.agentic_kv_source_demotion_join_wait_ns = 8
        resume.agentic_kv_owner_gate_ns = 18
        resume.agentic_kv_restore_issue_time_ns = 118
        resume.agentic_kv_target_hbm_ready_time_ns = 120
        resume.agentic_kv_restore_ready_time_ns = 128
        resume.first_schedule_resource_ready_time_ns = 128
        resume.first_schedule_eligibility_time_ns = 128
        resume.scheduler_queue_wait_ns = 2
        scheduler = SimpleNamespace(pd_type="decode", done=[initial, resume])

        report = build_session_metrics(
            _Router(), [scheduler], 320,
            dataset="workload.jsonl", run_id="demotion-join")

        distribution = report["requests"]["all"]
        self.assertEqual(
            distribution["source_demotion_join_wait_ns"]["sum"], 8)
        overhead = report["overhead_denominators"][
            "measured_session_cohort"]
        self.assertEqual(overhead["source_demotion_join_wait_ns"], 8)
        request = report["requests"]["records"][1]
        self.assertEqual(
            request["agentic_kv_source_demotion_join_wait_ns"], 8)

    def test_transient_dram_wait_is_lower_tier_queue_subset(self):
        initial = _request(0, 0, None, 110, arrival=10)
        request = _request(1, 1, "ssd", 310, arrival=110)
        request.agentic_kv_transient_dram_capacity_wait_ns = 3
        scheduler = SimpleNamespace(
            pd_type="decode", done=[initial, request])

        report = build_session_metrics(_Router(), [scheduler], 320)

        self.assertTrue(report["validation"]["timing"]["passed"])
        self.assertEqual(
            report["requests"]["resume"]
            ["transient_dram_capacity_wait_ns"]["sum"],
            3,
        )
        self.assertIn(
            "subset of restore_queue_wait_ns",
            report["denominator_notes"]["transient_dram_capacity_wait"],
        )

    def test_transient_dram_wait_cannot_exceed_restore_queue_wait(self):
        initial = _request(0, 0, None, 110, arrival=10)
        request = _request(1, 1, "ssd", 310, arrival=110)
        request.agentic_kv_transient_dram_capacity_wait_ns = 4
        scheduler = SimpleNamespace(
            pd_type="decode", done=[initial, request])

        with self.assertRaisesRegex(
                RuntimeError, "exceeds total lower-tier restore queue wait"):
            build_session_metrics(_Router(), [scheduler], 320)

    def test_negative_itl_is_rejected_instead_of_silently_filtered(self):
        request = _request(0, 0, None, 310, arrival=10)
        request.itl[0] = -1
        scheduler = SimpleNamespace(pd_type="decode", done=[request])

        with self.assertRaisesRegex(RuntimeError, "negative inter-token"):
            build_session_metrics(_Router(), [scheduler], 320)

    def test_closed_loop_successor_dependency_is_exact(self):
        requests = [
            _request(0, 0, None, 110),
            _request(1, 1, "cpu", 210),
            _request(2, 2, "ssd", 310),
        ]
        requests[1].return_gap_ns = 1
        scheduler = SimpleNamespace(pd_type="decode", done=requests)

        with self.assertRaisesRegex(RuntimeError, "successor arrival"):
            build_session_metrics(_Router(), [scheduler], 320)

    def test_completion_window_excludes_warmup_and_drain(self):
        router = _Router()
        router.session_admission = SimpleNamespace(
            warmup_completions=1,
            measure_completions=1,
            max_active_sessions=1,
        )
        router.session_lifecycle_records = lambda: [
            {
                "session_id": "warmup",
                "status": "completed",
                "offered_time_ns": 0,
                "admission_time_ns": 0,
                "admission_queue_wait_ns": 0,
                "completion_time_ns": 100,
                "e2e_ns": 100,
            },
            {
                "session_id": "measured",
                "status": "completed",
                "offered_time_ns": 0,
                "admission_time_ns": 200,
                "admission_queue_wait_ns": 200,
                "completion_time_ns": 300,
                "e2e_ns": 100,
            },
            {
                "session_id": "drain",
                "status": "completed",
                "offered_time_ns": 0,
                "admission_time_ns": 800,
                "admission_queue_wait_ns": 800,
                "completion_time_ns": 900,
                "e2e_ns": 100,
            },
        ]
        scheduler = SimpleNamespace(
            pd_type="decode",
            done=[
                SimpleNamespace(**{
                    **_request(0, 0, None, 100).__dict__,
                    "session_id": "warmup",
                }),
                SimpleNamespace(**{
                    **_request(1, 0, None, 300, arrival=200).__dict__,
                    "session_id": "measured",
                }),
                SimpleNamespace(**{
                    **_request(2, 0, None, 900, arrival=800).__dict__,
                    "session_id": "drain",
                }),
            ],
        )

        report = build_session_metrics(router, [scheduler], 900)

        window = report["measurement_window"]
        self.assertEqual(window["measurement_start_ns"], 100)
        self.assertEqual(window["measurement_end_ns"], 300)
        self.assertEqual(window["measurement_duration_ns"], 200)
        self.assertEqual(report["throughput"]["completed_sessions"], 1)
        self.assertEqual(report["throughput"]["completed_sessions_total"], 3)
        self.assertEqual(report["requests"]["all"]["count"], 1)
        self.assertEqual(report["throughput"]["completed_requests"], 1)
        self.assertEqual(
            report["throughput"]["sessions_per_second_measurement_window"],
            5_000_000,
        )

    def test_request_throughput_uses_completion_events_not_whole_cohort(self):
        router = _Router()
        router.session_admission = SimpleNamespace(
            warmup_completions=1,
            measure_completions=1,
            max_active_sessions=2,
        )
        router.session_lifecycle_records = lambda: [
            {
                "session_id": "warmup",
                "status": "completed",
                "offered_time_ns": 0,
                "admission_time_ns": 0,
                "admission_queue_wait_ns": 0,
                "completion_time_ns": 100,
                "e2e_ns": 100,
            },
            {
                "session_id": "measured",
                "status": "completed",
                "offered_time_ns": 0,
                "admission_time_ns": 0,
                "admission_queue_wait_ns": 0,
                "completion_time_ns": 300,
                "e2e_ns": 300,
            },
            {
                "session_id": "later",
                "status": "completed",
                "offered_time_ns": 0,
                "admission_time_ns": 400,
                "admission_queue_wait_ns": 400,
                "completion_time_ns": 500,
                "e2e_ns": 100,
            },
        ]
        scheduler = SimpleNamespace(
            pd_type="decode",
            done=[
                SimpleNamespace(**{
                    **_request(0, 0, None, 80, arrival=0).__dict__,
                    "session_id": "measured",
                }),
                SimpleNamespace(**{
                    **_request(1, 1, "cpu", 180, arrival=80).__dict__,
                    "session_id": "measured",
                }),
                SimpleNamespace(**{
                    **_request(3, 2, "hbm", 300, arrival=180).__dict__,
                    "session_id": "measured",
                }),
                SimpleNamespace(**{
                    **_request(2, 0, None, 500, arrival=400).__dict__,
                    "session_id": "later",
                }),
            ],
        )

        report = build_session_metrics(router, [scheduler], 500)

        self.assertEqual(
            report["throughput"]["completed_requests_in_session_cohort"],
            3,
        )
        self.assertEqual(report["throughput"]["completed_requests"], 2)
        self.assertEqual(report["requests"]["all"]["count"], 3)
        self.assertEqual(report["throughput"]["prompt_tokens"], 200)
        self.assertEqual(
            report["throughput"]["requests_per_second_measurement_window"],
            10_000_000,
        )

    def test_save_session_metrics_writes_valid_json(self):
        report = {"schema_version": 1, "mode": "trace"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            save_session_metrics(report, str(path))
            self.assertEqual(json.loads(path.read_text()), report)


if __name__ == "__main__":
    unittest.main()

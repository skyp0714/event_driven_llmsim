import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from serving.core.router import Router
from serving.core.session_admission import (
    SessionAdmissionConfig,
    add_session_admission_arguments,
    session_admission_from_args,
)

from test_agentic_router import FakeScheduler, FakeTierManager


def _session(session_id, arrival_ns, calls=1, gap_ns=0):
    sub_requests = []
    for index in range(calls):
        sub_requests.append({
            "input_toks": 10 + index,
            "output_toks": 2,
            "tool_duration_ns": gap_ns if index + 1 < calls else 0,
        })
    return {
        "session_id": session_id,
        "arrival_time_ns": arrival_ns,
        "sub_requests": sub_requests,
    }


def _write_workload(rows):
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    with handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return Path(handle.name)


class SessionAdmissionTests(unittest.TestCase):
    def test_cli_defaults_preserve_trace_arrivals(self):
        parser = argparse.ArgumentParser()
        add_session_admission_arguments(parser)
        config = session_admission_from_args(parser.parse_args([]))
        self.assertEqual(config, SessionAdmissionConfig())

        scheduler = FakeScheduler()
        router = Router(1, [scheduler], 0, session_admission=config)
        path = _write_workload([
            _session("late", 20),
            _session("early", 10),
        ])
        self.addCleanup(path.unlink)
        router.load_requests(str(path))

        self.assertEqual(router.route_arrived_requests(9), 0)
        self.assertEqual(router.route_arrived_requests(10), 1)
        self.assertEqual(scheduler.added[0][0][4], 10)
        self.assertEqual(scheduler.added[0][1]["session_id"], "early")
        self.assertEqual(router.route_arrived_requests(20), 1)

    def test_poisson_arrivals_are_deterministic_and_replace_trace_times(self):
        config = SessionAdmissionConfig(
            mode="poisson",
            session_arrival_rate_sps=2.0,
            session_arrival_seed=7,
        )
        rows = [
            _session("a", 999),
            _session("b", 1),
            _session("c", 2),
        ]
        arrivals = []
        for _ in range(2):
            scheduler = FakeScheduler()
            router = Router(1, [scheduler], 0, session_admission=config)
            path = _write_workload(rows)
            self.addCleanup(path.unlink)
            router.load_requests(str(path))
            arrivals.append([
                item["arrival_time_ns"]
                for item in router._pending_requests
            ])
        self.assertEqual(arrivals[0], arrivals[1])
        self.assertEqual(arrivals[0][0], 0)
        self.assertEqual(arrivals[0], sorted(arrivals[0]))
        self.assertNotEqual(arrivals[0], [999, 1, 2])
        summary = router.session_admission_summary()
        self.assertEqual(summary["mode"], "poisson")
        self.assertEqual(summary["offered_sessions"], 3)
        self.assertEqual(summary["session_arrival_rate_sps"], 2.0)
        self.assertEqual(summary["session_arrival_seed"], 7)

    def test_poisson_active_limit_builds_fifo_admission_backlog(self):
        scheduler = FakeScheduler()
        router = Router(
            1,
            [scheduler],
            0,
            session_admission=SessionAdmissionConfig(
                mode="poisson",
                max_active_sessions=1,
                session_arrival_rate_sps=2.0,
                session_arrival_seed=7,
            ),
        )
        path = _write_workload([
            _session("a", 999),
            _session("b", 1),
            _session("c", 2),
        ])
        self.addCleanup(path.unlink)
        router.load_requests(str(path))

        lifecycle = router.session_lifecycle_records()
        offered = [row["offered_time_ns"] for row in lifecycle]
        self.assertEqual(offered, sorted(offered))
        self.assertEqual(
            [row["status"] for row in lifecycle],
            ["active", "waiting_for_admission", "waiting_for_admission"],
        )
        self.assertEqual(router.route_arrived_requests(0), 1)
        first = scheduler.request.pop(0)

        completion_ns = offered[-1] + 10
        router.notify_request_completed(first, completion_ns)
        summary = router.session_admission_summary()
        self.assertEqual(summary["active_sessions"], 1)
        self.assertEqual(summary["admitted_sessions"], 2)
        self.assertEqual(summary["remaining_backlog_sessions"], 1)
        self.assertEqual(
            summary["queue_policy"], "poisson_fifo_wait_for_slot")
        self.assertEqual(summary["logical_session_drop_count"], 0)

        lifecycle = router.session_lifecycle_records()
        self.assertEqual(
            lifecycle[1]["admission_time_ns"], completion_ns)
        self.assertEqual(
            lifecycle[1]["admission_queue_wait_ns"],
            completion_ns - offered[1],
        )
        self.assertEqual(lifecycle[2]["admission_time_ns"], None)

    def test_poisson_strict_pd_stages_only_arrived_sessions(self):
        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        router = Router(
            2,
            [prefill, decode],
            0,
            session_admission=SessionAdmissionConfig(
                mode="poisson",
                session_arrival_rate_sps=1.0,
                session_arrival_seed=5,
            ),
            agentic_kv_manager=FakeTierManager(),
        )
        path = _write_workload([
            _session("first", 999),
            _session("future", 1),
        ])
        self.addCleanup(path.unlink)
        router.load_requests(str(path))

        self.assertEqual(router.route_arrived_requests(0), 1)
        self.assertEqual(len(prefill.request), 1)
        next_arrival = router.get_next_pending_arrival()
        self.assertGreater(next_arrival, 0)
        self.assertEqual(router.route_arrived_requests(next_arrival - 1), 0)
        self.assertEqual(len(prefill.request), 1)

        # The second prefill is admitted at its own offer time while the first
        # remains runnable, so the scheduler may continuously batch both.
        self.assertEqual(router.route_arrived_requests(next_arrival), 1)
        self.assertEqual(len(prefill.request), 2)
        self.assertEqual(
            [metadata["source_session_id"] for _, metadata in prefill.added],
            ["first", "future"],
        )

    def test_backlog_holds_slot_across_closed_loop_gap(self):
        scheduler = FakeScheduler()
        router = Router(
            1,
            [scheduler],
            0,
            session_admission=SessionAdmissionConfig(
                mode="backlog", max_active_sessions=1),
        )
        path = _write_workload([
            _session("two-call", 999, calls=2, gap_ns=50),
            _session("next", 1, calls=1),
        ])
        self.addCleanup(path.unlink)
        router.load_requests(str(path))

        summary = router.session_admission_summary()
        self.assertEqual(summary["active_sessions"], 1)
        self.assertEqual(summary["remaining_backlog_sessions"], 1)
        self.assertEqual(router.route_arrived_requests(0), 1)
        first = scheduler.request.pop(0)
        first.num_computed_tokens = 12
        router.notify_request_completed(first, 100)

        # The same session owns K during its recorded wait; no replacement is
        # admitted until its second and final call completes.
        summary = router.session_admission_summary()
        self.assertEqual(summary["active_sessions"], 1)
        self.assertEqual(summary["admitted_sessions"], 1)
        self.assertEqual(router.get_next_pending_arrival(), 150)
        self.assertEqual(router.route_arrived_requests(149), 0)
        self.assertEqual(router.route_arrived_requests(150), 1)
        second = scheduler.request.pop(0)
        router.notify_request_completed(second, 200)

        summary = router.session_admission_summary()
        self.assertEqual(summary["active_sessions"], 1)
        self.assertEqual(summary["admitted_sessions"], 2)
        self.assertEqual(summary["completed_sessions"], 1)
        self.assertEqual(summary["remaining_backlog_sessions"], 0)
        self.assertEqual(router.get_next_pending_arrival(), 200)
        self.assertEqual(router.route_arrived_requests(200), 1)
        replacement_metadata = scheduler.added[-1][1]
        self.assertEqual(replacement_metadata["source_session_id"], "next")
        self.assertEqual(replacement_metadata["session_admission_time_ns"], 200)
        self.assertEqual(replacement_metadata["session_offered_time_ns"], 0)
        self.assertEqual(
            replacement_metadata["session_admission_queue_wait_ns"], 200)

    def test_backlog_refills_every_freed_slot_without_losing_waiters(self):
        scheduler = FakeScheduler()
        router = Router(
            1,
            [scheduler],
            0,
            session_admission=SessionAdmissionConfig(
                mode="backlog", max_active_sessions=2),
        )
        path = _write_workload([
            _session("a", 0),
            _session("b", 0),
            _session("c", 0),
            _session("d", 0),
            _session("e", 0),
        ])
        self.addCleanup(path.unlink)
        router.load_requests(str(path))
        router.route_arrived_requests(0)

        initial = {
            request.session_id: request for request in scheduler.request
        }
        request_a = initial["a::template=0::epoch=0"]
        request_b = initial["b::template=1::epoch=0"]
        scheduler.request.remove(request_a)
        scheduler.request.remove(request_b)

        # Each completion releases exactly one slot. Even when completions
        # share a timestamp, the next templates stay ordered and no waiter is
        # discarded or admitted above K.
        router.notify_request_completed(request_a, 100)
        summary = router.session_admission_summary()
        self.assertEqual(summary["active_sessions"], 2)
        self.assertEqual(summary["remaining_backlog_sessions"], 2)

        router.notify_request_completed(request_b, 100)
        summary = router.session_admission_summary()
        self.assertEqual(summary["active_sessions"], 2)
        self.assertEqual(summary["admitted_sessions"], 4)
        self.assertEqual(summary["completed_sessions"], 2)
        self.assertEqual(summary["remaining_backlog_sessions"], 1)
        self.assertEqual(summary["queue_policy"], "fifo_wait_for_slot")
        self.assertEqual(summary["logical_session_drop_count"], 0)
        self.assertEqual(
            summary["slot_release_event"],
            "final_request_completion_on_colocated_owner",
        )
        self.assertEqual(
            summary["slot_release_event_legacy"],
            "final_llm_request_completion",
        )
        self.assertEqual(summary["cutoff_disposition"], "drain")
        self.assertEqual(
            summary["admitted_sessions"],
            summary["completed_sessions"] + summary["active_sessions"],
        )

        router.route_arrived_requests(100)
        self.assertEqual(
            [request.session_id for request in scheduler.request],
            [
                "c::template=2::epoch=0",
                "d::template=3::epoch=0",
            ],
        )
        lifecycle = router.session_lifecycle_records()
        self.assertEqual(
            [row["status"] for row in lifecycle],
            ["completed", "completed", "active", "active", "backlog"],
        )
        self.assertNotIn("dropped", {row["status"] for row in lifecycle})

    def test_strict_pd_backlog_releases_slot_only_after_decode_completion(self):
        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = FakeTierManager()
        router = Router(
            2,
            [prefill, decode],
            0,
            session_admission=SessionAdmissionConfig(
                mode="backlog", max_active_sessions=1),
            agentic_kv_manager=manager,
        )
        path = _write_workload([
            _session("first", 100),
            _session("replacement", 200),
        ])
        self.addCleanup(path.unlink)
        router.load_requests(str(path))

        self.assertEqual(router.route_arrived_requests(0), 1)
        first = prefill.request.pop(0)
        self.assertEqual(first.pd_decode_target_instance_id, 1)
        self.assertEqual(
            router.session_admission_summary()["active_sessions"], 1)

        # P completion and the P->D handoff are not session completion. The
        # closed-population slot remains owned while decode is active.
        router.transfer_prefill_request([first], current_time_ns=10)
        self.assertEqual(decode.decoded, [first])
        self.assertEqual(router.route_arrived_requests(10), 0)
        summary = router.session_admission_summary()
        self.assertEqual(summary["active_sessions"], 1)
        self.assertEqual(summary["admitted_sessions"], 1)
        self.assertEqual(summary["remaining_backlog_sessions"], 1)
        self.assertEqual(len(prefill.added), 1)

        # Only D completion ends the old session. The replacement is admitted
        # at that same logical time and becomes routable independently.
        router.notify_request_completed(first, 20)
        summary = router.session_admission_summary()
        self.assertEqual(manager.ended, [first.session_id])
        self.assertEqual(summary["completed_sessions"], 1)
        self.assertEqual(summary["active_sessions"], 1)
        self.assertEqual(summary["admitted_sessions"], 2)
        self.assertEqual(summary["remaining_backlog_sessions"], 0)
        self.assertEqual(
            summary["slot_release_event"],
            "final_request_completion_on_decode_owner")
        self.assertEqual(
            summary["slot_release_event_legacy"],
            "final_decode_completion")
        self.assertEqual(router.get_next_pending_arrival(), 20)

        self.assertEqual(router.route_arrived_requests(20), 1)
        self.assertEqual(len(prefill.added), 2)
        self.assertEqual(
            prefill.added[-1][1]["source_session_id"], "replacement")

    def test_backlog_epochs_have_deterministic_unique_ids_and_drain(self):
        scheduler = FakeScheduler()
        router = Router(
            1,
            [scheduler],
            0,
            session_admission=SessionAdmissionConfig(
                mode="backlog",
                max_active_sessions=2,
                backlog_epochs=2,
            ),
        )
        path = _write_workload([
            _session("a", 100),
            _session("b", 200),
        ])
        self.addCleanup(path.unlink)
        router.load_requests(str(path))

        expected_ids = [
            "a::template=0::epoch=0",
            "b::template=1::epoch=0",
            "a::template=0::epoch=1",
            "b::template=1::epoch=1",
        ]
        offered = router.session_lifecycle_records()
        self.assertEqual([row["session_id"] for row in offered], expected_ids)
        self.assertEqual(
            [row["status"] for row in offered],
            ["active", "active", "backlog", "backlog"],
        )
        observed_ids = []
        now = 0
        while router.has_deferred_sessions():
            router.route_arrived_requests(now)
            while scheduler.request:
                request = scheduler.request.pop(0)
                observed_ids.append(request.session_id)
                now += 10
                router.notify_request_completed(request, now)

        self.assertEqual(observed_ids, expected_ids)
        self.assertFalse(router.has_pending_requests())
        self.assertEqual(
            router.session_admission_summary()["completed_sessions"], 4)
        records = router.session_lifecycle_records()
        self.assertEqual([row["session_id"] for row in records], expected_ids)
        self.assertTrue(all(row["status"] == "completed" for row in records))
        self.assertTrue(all(
            row["e2e_ns"]
            == row["completion_time_ns"] - row["admission_time_ns"]
            for row in records
        ))

    def test_backlog_rejects_flat_rows_before_routing(self):
        router = Router(
            1,
            [FakeScheduler()],
            0,
            session_admission=SessionAdmissionConfig(
                mode="backlog", max_active_sessions=1),
        )
        path = _write_workload([{
            "input_toks": 10,
            "output_toks": 2,
            "arrival_time_ns": 0,
        }])
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "only agentic rows"):
            router.load_requests(str(path))
        self.assertFalse(router.has_pending_requests())

    def test_poisson_rejects_flat_rows(self):
        router = Router(
            1,
            [FakeScheduler()],
            0,
            session_admission=SessionAdmissionConfig(
                mode="poisson", session_arrival_rate_sps=1.0),
        )
        path = _write_workload([{
            "input_toks": 10,
            "output_toks": 2,
            "arrival_time_ns": 0,
        }])
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "poisson mode accepts only"):
            router.load_requests(str(path))
        self.assertFalse(router.has_pending_requests())

    def test_load_control_rejects_empty_agentic_sessions(self):
        for mode, kwargs in (
                ("poisson", {"session_arrival_rate_sps": 1.0}),
                ("backlog", {"max_active_sessions": 1})):
            with self.subTest(mode=mode):
                router = Router(
                    1,
                    [FakeScheduler()],
                    0,
                    session_admission=SessionAdmissionConfig(
                        mode=mode, **kwargs),
                )
                path = _write_workload([{
                    "session_id": "empty",
                    "arrival_time_ns": 0,
                    "sub_requests": [],
                }])
                self.addCleanup(path.unlink)
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    router.load_requests(str(path))

    def test_completion_measurement_window_arguments_are_validated(self):
        parser = argparse.ArgumentParser()
        add_session_admission_arguments(parser)
        config = session_admission_from_args(parser.parse_args([
            "--session-warmup-completions", "10",
            "--session-measure-completions", "100",
        ]))
        self.assertEqual(config.warmup_completions, 10)
        self.assertEqual(config.measure_completions, 100)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            SessionAdmissionConfig(warmup_completions=-1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            SessionAdmissionConfig(measure_completions=-1)
        with self.assertRaisesRegex(
                ValueError, "requires measure_completions > 0"):
            SessionAdmissionConfig(stop_after_measurement=True)

        admission_order = session_admission_from_args(parser.parse_args([
            "--session-arrival-mode", "backlog",
            "--max-active-sessions", "2",
            "--session-measure-completions", "1",
            "--session-stop-after-measurement",
            "--session-measurement-cohort-selection", "admission_order",
        ]))
        self.assertEqual(
            admission_order.measurement_cohort_selection,
            "admission_order",
        )
        with self.assertRaisesRegex(ValueError, "require backlog mode"):
            SessionAdmissionConfig(
                mode="poisson",
                session_arrival_rate_sps=1.0,
                measurement_cohort_selection="admission_order",
            )
        fixed_prefix = SessionAdmissionConfig(
            mode="backlog",
            max_active_sessions=1,
            warmup_completions=1,
            measure_completions=1,
            measurement_cohort_selection="admission_order",
        )
        self.assertEqual(fixed_prefix.warmup_completions, 1)
        with self.assertRaisesRegex(
                ValueError, "Unknown measurement cohort selection"):
            SessionAdmissionConfig(
                measurement_cohort_selection="policy_dependent",
            )

    def test_admission_order_target_waits_for_fixed_session_and_refills_k(self):
        scheduler = FakeScheduler()
        router = Router(
            1,
            [scheduler],
            0,
            session_admission=SessionAdmissionConfig(
                mode="backlog",
                max_active_sessions=2,
                measure_completions=1,
                stop_after_measurement=True,
                measurement_cohort_selection="admission_order",
            ),
        )
        path = _write_workload([
            _session("a", 0),
            _session("b", 0),
            _session("c", 0),
            _session("d", 0),
            _session("e", 0),
        ])
        self.addCleanup(path.unlink)
        router.load_requests(str(path))
        router.route_arrived_requests(0)

        requests = {
            request.session_id: request
            for request in scheduler.request
        }
        target_id = "a::template=0::epoch=0"
        self.assertEqual(
            router.measurement_target_session_ids(), (target_id,))
        self.assertFalse(router.measurement_boundary_would_be_reached([
            requests["b::template=1::epoch=0"]]))

        scheduler.request.remove(requests["b::template=1::epoch=0"])
        router.notify_request_completed(
            requests["b::template=1::epoch=0"], 10)
        router.route_arrived_requests(10)
        requests = {
            request.session_id: request
            for request in scheduler.request
        }
        self.assertEqual(
            router.session_admission_summary()["active_sessions"], 2)
        self.assertFalse(router.measurement_boundary_would_be_reached([
            requests["c::template=2::epoch=0"]]))

        scheduler.request.remove(requests["c::template=2::epoch=0"])
        router.notify_request_completed(
            requests["c::template=2::epoch=0"], 20)
        router.route_arrived_requests(20)
        requests = {
            request.session_id: request
            for request in scheduler.request
        }
        self.assertTrue(router.measurement_boundary_would_be_reached([
            requests[target_id]]))
        self.assertFalse(router.measurement_target_reached())

        router.freeze_session_admission()
        scheduler.request.remove(requests[target_id])
        router.notify_request_completed(requests[target_id], 30)
        self.assertTrue(router.measurement_target_reached())
        summary = router.session_admission_summary()
        self.assertEqual(summary["active_sessions"], 1)
        self.assertEqual(summary["admitted_sessions"], 4)
        self.assertEqual(summary["remaining_backlog_sessions"], 1)
        self.assertEqual(summary["measurement_target_session_count"], 1)
        self.assertEqual(summary["measurement_target_completed_sessions"], 1)

        lifecycle = router.session_lifecycle_records()
        self.assertEqual(
            [row["planned_admission_index"] for row in lifecycle],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            [row["admission_index"] for row in lifecycle],
            [0, 1, 2, 3, None],
        )
        self.assertEqual(
            [row["session_id"] for row in lifecycle
             if row["measurement_target"]],
            [target_id],
        )

    def test_fixed_admission_prefix_waits_for_slow_warmup_not_target_only(self):
        scheduler = FakeScheduler()
        router = Router(
            1,
            [scheduler],
            0,
            session_admission=SessionAdmissionConfig(
                mode="backlog",
                max_active_sessions=2,
                warmup_completions=1,
                measure_completions=1,
                stop_after_measurement=True,
                measurement_cohort_selection="admission_order",
            ),
        )
        path = _write_workload([
            _session("slow-warmup", 0),
            _session("fast-target", 0),
            _session("outside", 0),
        ])
        self.addCleanup(path.unlink)
        router.load_requests(str(path))
        router.route_arrived_requests(0)

        warmup_id = "slow-warmup::template=0::epoch=0"
        target_id = "fast-target::template=1::epoch=0"
        self.assertEqual(
            router.measurement_warmup_session_ids(), (warmup_id,))
        self.assertEqual(
            router.measurement_target_session_ids(), (target_id,))
        self.assertEqual(
            router.measurement_required_session_ids(),
            (warmup_id, target_id),
        )

        requests = {
            request.session_id: request for request in scheduler.request
        }
        target = requests[target_id]
        self.assertFalse(router.measurement_boundary_would_be_reached([
            target]))
        scheduler.request.remove(target)
        router.notify_request_completed(target, 10)
        router.route_arrived_requests(10)
        self.assertFalse(router.measurement_target_reached())

        requests = {
            request.session_id: request for request in scheduler.request
        }
        warmup = requests[warmup_id]
        outside = requests["outside::template=2::epoch=0"]
        self.assertFalse(router.measurement_boundary_would_be_reached([
            outside]))
        self.assertTrue(router.measurement_boundary_would_be_reached([
            warmup]))
        scheduler.request.remove(warmup)
        router.notify_request_completed(warmup, 20)
        self.assertTrue(router.measurement_target_reached())

        summary = router.session_admission_summary()
        self.assertEqual(summary["measurement_warmup_session_count"], 1)
        self.assertEqual(summary["measurement_target_session_count"], 1)
        self.assertEqual(summary["measurement_required_session_count"], 2)
        self.assertEqual(summary["measurement_required_completed_sessions"], 2)
        self.assertEqual(summary["measurement_prefix_id_overlap_count"], 0)
        roles = {
            row["session_id"]: row["measurement_role"]
            for row in router.session_lifecycle_records()
        }
        self.assertEqual(
            roles[warmup_id], "fixed_admission_prefix_warmup")
        self.assertEqual(roles[target_id], "measurement_target")
        self.assertEqual(
            roles["outside::template=2::epoch=0"],
            "outside_required_prefix",
        )

    def test_completion_order_boundary_remains_policy_dependent_default(self):
        scheduler = FakeScheduler()
        router = Router(
            1,
            [scheduler],
            0,
            session_admission=SessionAdmissionConfig(
                mode="backlog",
                max_active_sessions=2,
                measure_completions=1,
                stop_after_measurement=True,
            ),
        )
        path = _write_workload([
            _session("a", 0),
            _session("b", 0),
            _session("c", 0),
        ])
        self.addCleanup(path.unlink)
        router.load_requests(str(path))
        router.route_arrived_requests(0)
        request_b = next(
            request for request in scheduler.request
            if request.session_id == "b::template=1::epoch=0")

        self.assertEqual(router.measurement_target_session_ids(), ())
        self.assertTrue(
            router.measurement_boundary_would_be_reached([request_b]))

    def test_admission_order_boundary_requires_every_unfinished_target_final(self):
        scheduler = FakeScheduler()
        router = Router(
            1,
            [scheduler],
            0,
            session_admission=SessionAdmissionConfig(
                mode="backlog",
                max_active_sessions=3,
                measure_completions=2,
                stop_after_measurement=True,
                measurement_cohort_selection="admission_order",
            ),
        )
        path = _write_workload([
            _session("a", 0, calls=2),
            _session("b", 0),
            _session("c", 0),
        ])
        self.addCleanup(path.unlink)
        router.load_requests(str(path))
        router.route_arrived_requests(0)
        first_a = next(
            request for request in scheduler.request
            if request.session_id == "a::template=0::epoch=0")
        request_b = next(
            request for request in scheduler.request
            if request.session_id == "b::template=1::epoch=0")

        self.assertFalse(router.measurement_boundary_would_be_reached([
            first_a, request_b]))
        scheduler.request.remove(first_a)
        router.notify_request_completed(first_a, 10)
        router.route_arrived_requests(10)
        final_a = next(
            request for request in scheduler.request
            if request.session_id == "a::template=0::epoch=0")

        self.assertFalse(
            router.measurement_boundary_would_be_reached([final_a]))
        self.assertTrue(router.measurement_boundary_would_be_reached([
            final_a, request_b]))

    def test_fixed_boundary_completes_tied_finals_and_censors_other_waiters(self):
        scheduler = FakeScheduler()
        router = Router(
            1,
            [scheduler],
            0,
            session_admission=SessionAdmissionConfig(
                mode="backlog",
                max_active_sessions=3,
                measure_completions=1,
                stop_after_measurement=True,
                measurement_cohort_selection="admission_order",
            ),
        )
        path = _write_workload([
            _session("target", 0),
            _session("other-final", 0),
            _session("other-nonfinal", 0, calls=2),
            _session("waiting-a", 0),
            _session("waiting-b", 0),
        ])
        self.addCleanup(path.unlink)
        router.load_requests(str(path))
        router.route_arrived_requests(0)

        requests = {
            request.session_id: request for request in scheduler.request
        }
        target = requests["target::template=0::epoch=0"]
        other_final = requests["other-final::template=1::epoch=0"]
        other_nonfinal = requests["other-nonfinal::template=2::epoch=0"]
        tied = [target, other_final, other_nonfinal]
        self.assertTrue(router.measurement_boundary_would_be_reached(tied))

        # Mirror the serving-loop boundary: freeze before callbacks, record
        # every tied final call, and do not release a successor from a tied
        # non-final call. The unfinished population is right-censored later;
        # it is not silently dropped and no replacement crosses the boundary.
        router.freeze_session_admission()
        for request in tied:
            scheduler.request.remove(request)
        router.notify_request_completed(target, 100)
        router.notify_request_completed(other_final, 100)

        summary = router.session_admission_summary()
        self.assertTrue(router.measurement_target_reached())
        self.assertEqual(summary["admitted_sessions"], 3)
        self.assertEqual(summary["completed_sessions"], 2)
        self.assertEqual(summary["active_sessions"], 1)
        self.assertEqual(summary["remaining_backlog_sessions"], 2)
        self.assertTrue(summary["admission_frozen"])
        self.assertEqual(summary["logical_session_drop_count"], 0)
        self.assertEqual(summary["cutoff_disposition"], "right_censor")

        censoring = router.finalize_measurement_censoring(100)
        self.assertEqual(censoring["censored_sessions"], 3)
        self.assertEqual(
            censoring["status_counts_before_censoring"],
            {"active": 1, "backlog": 2},
        )
        lifecycle = router.session_lifecycle_records()
        self.assertEqual(
            [row["status"] for row in lifecycle],
            ["completed", "completed", "censored", "censored", "censored"],
        )
        self.assertNotIn("dropped", {row["status"] for row in lifecycle})
        self.assertFalse(router.has_deferred_sessions())
        self.assertEqual(router._request_to_session, {})

    def test_admission_order_target_must_fit_finite_backlog(self):
        router = Router(
            1,
            [FakeScheduler()],
            0,
            session_admission=SessionAdmissionConfig(
                mode="backlog",
                max_active_sessions=1,
                measure_completions=2,
                stop_after_measurement=True,
                measurement_cohort_selection="admission_order",
            ),
        )
        path = _write_workload([_session("only", 0)])
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(
                ValueError, "backlog contains only 1"):
            router.load_requests(str(path))

    def test_measurement_freeze_censors_without_backlog_replacement(self):
        scheduler = FakeScheduler()
        router = Router(
            1,
            [scheduler],
            0,
            session_admission=SessionAdmissionConfig(
                mode="backlog",
                max_active_sessions=1,
                measure_completions=1,
                stop_after_measurement=True,
            ),
        )
        path = _write_workload([
            _session("first", 0),
            _session("second", 0),
            _session("third", 0),
        ])
        self.addCleanup(path.unlink)
        router.load_requests(str(path))
        router.route_arrived_requests(0)
        request = scheduler.request.pop(0)

        router.freeze_session_admission()
        router.notify_request_completed(request, 100)

        summary = router.session_admission_summary()
        self.assertEqual(summary["completed_sessions"], 1)
        self.assertEqual(summary["admitted_sessions"], 1)
        self.assertTrue(summary["admission_frozen"])
        censoring = router.finalize_measurement_censoring(100)
        self.assertEqual(censoring["censored_sessions"], 2)
        self.assertEqual(
            censoring["status_counts_before_censoring"], {"backlog": 2})

    def test_measurement_freeze_blocks_nonfinal_successor_at_api_boundary(self):
        scheduler = FakeScheduler()
        router = Router(
            1,
            [scheduler],
            0,
            session_admission=SessionAdmissionConfig(
                mode="backlog", max_active_sessions=1,
                measure_completions=1, stop_after_measurement=True),
        )
        path = _write_workload([
            _session("active", 0, calls=2, gap_ns=50),
            _session("waiting", 0),
        ])
        self.addCleanup(path.unlink)
        router.load_requests(str(path))
        router.route_arrived_requests(0)
        first = scheduler.request.pop(0)

        router.freeze_session_admission()
        router.notify_request_completed(first, 100)

        # The guard is in notify_request_completed itself, not only in the
        # serving loop: the successor is not materialized and the active
        # logical session keeps K until audited right-censor cleanup.
        self.assertEqual(router.get_next_pending_arrival(), None)
        self.assertIn(first.id, router._request_to_session)
        summary = router.session_admission_summary()
        self.assertEqual(summary["admitted_sessions"], 1)
        self.assertEqual(summary["completed_sessions"], 0)
        self.assertEqual(summary["active_sessions"], 1)
        self.assertEqual(summary["remaining_backlog_sessions"], 1)

        censoring = router.finalize_measurement_censoring(100)
        self.assertEqual(censoring["censored_sessions"], 2)
        self.assertEqual(router._request_to_session, {})

    def test_backlog_requires_positive_k(self):
        with self.assertRaisesRegex(ValueError, "max_active_sessions > 0"):
            SessionAdmissionConfig(mode="backlog")

    def test_poisson_requires_positive_rate(self):
        with self.assertRaisesRegex(ValueError, "arrival_rate_sps > 0"):
            SessionAdmissionConfig(mode="poisson")


if __name__ == "__main__":
    unittest.main()

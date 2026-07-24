import unittest
from types import SimpleNamespace

from serving.__main__ import (
    _analytical_advance_command,
    _analytical_report_time,
    _idle_fast_forward_delta,
    _measurement_drain_complete,
    _next_exact_control_wakeup_ns,
    _next_idle_wakeup_ns,
    _route_strictly_older_arrivals_at_callback,
    _throughput_interval_scale,
    _uniform_cluster_link_value,
)
from serving.core.controller import ExactControlSchedule


class _Router:
    def __init__(self, next_pending=None, next_handoff=None):
        self.next_pending = next_pending
        self.next_handoff = next_handoff

    def get_next_pending_arrival(self):
        return self.next_pending

    def get_next_decode_handoff_wakeup(self):
        return self.next_handoff


class _Manager:
    def __init__(
            self, blocked_until=None, internal_event=None,
            external_fabric_pending=False):
        self.blocked_until = blocked_until
        self.internal_event = internal_event
        self.external_fabric_pending = external_fabric_pending

    def synchronous_swap_blocked_until(self, instance_id, now_ns):
        if self.blocked_until is None or now_ns >= self.blocked_until:
            return None
        return self.blocked_until

    def next_internal_event_time(self, now_ns):
        if self.internal_event is None or now_ns >= self.internal_event:
            return None
        return self.internal_event

    def has_pending_external_fabric_jobs(self):
        return self.external_fabric_pending


def _scheduler(*, request_ready=None, memory_wait=None, swap_wait=None,
               fabric_wait=None, inflight=False, manager=None,
               instance_id=0):
    requests = (
        [] if request_ready is None
        else [SimpleNamespace(ready_time=request_ready)]
    )
    return SimpleNamespace(
        request=requests,
        inflight=[object()] if inflight else [],
        memory_wait_until_ns=memory_wait,
        model_fabric_wait_until_ns=fabric_wait,
        instance_id=instance_id,
        agentic_kv_manager=(
            manager if manager is not None else
            (None if swap_wait is None else _Manager(swap_wait))
        ),
    )


class AgenticTimeTest(unittest.TestCase):
    def test_model_callback_separates_older_cutoff_from_physical_time(self):
        class CallbackRouter:
            def __init__(self, manager):
                self.manager = manager
                self.calls = []

            def route_arrived_requests(
                    self, cutoff_ns, *, operation_time_ns=None):
                self.calls.append((cutoff_ns, operation_time_ns))
                self.manager.logical_frontier_ns = operation_time_ns
                return 1

        manager = SimpleNamespace(logical_frontier_ns=90)
        manager.advance = lambda now_ns: setattr(
            manager, "logical_frontier_ns", int(now_ns))
        router = CallbackRouter(manager)

        self.assertEqual(
            _route_strictly_older_arrivals_at_callback(
                101, router, manager),
            1,
        )
        self.assertEqual(router.calls, [(100, 101)])
        self.assertEqual(manager.logical_frontier_ns, 101)

    def test_measurement_drain_waits_for_external_and_control_callbacks(self):
        manager = _Manager(external_fabric_pending=True)
        controls = ExactControlSchedule()
        self.assertEqual(
            controls.arm(200, 100),
            "control-at\tpython-ready.0\t200",
        )
        schedulers = [_scheduler()]
        arguments = (
            True, schedulers, {"dp": {}}, {}, manager, controls)

        # Freeze while an external job is the only physical operation. Its
        # completion cannot exit early because a pre-freeze control callback
        # is still an outstanding ASTRA protocol obligation.
        self.assertFalse(_measurement_drain_complete(*arguments))
        manager.external_fabric_pending = False
        self.assertFalse(_measurement_drain_complete(*arguments))

        controls.complete("python-ready.0", 200)
        self.assertTrue(_measurement_drain_complete(*arguments))

    def test_uniform_cluster_link_contract_is_explicit(self):
        self.assertEqual(_uniform_cluster_link_value(450, "link_bw"), 450)
        self.assertEqual(
            _uniform_cluster_link_value([450, 450], "link_bw"), 450)
        with self.assertRaisesRegex(ValueError, "uniform"):
            _uniform_cluster_link_value([450, 100], "link_bw")

    def test_exact_control_wakeup_arms_earliest_live_dependency(self):
        manager = _Manager(blocked_until=700, internal_event=350)
        wakeup = _next_exact_control_wakeup_ns(
            100,
            [_scheduler(
                request_ready=800, memory_wait=650,
                fabric_wait=550, inflight=True)],
            _Router(next_pending=400, next_handoff=450),
            manager,
        )

        self.assertEqual(wakeup, 350)

    def test_throughput_interval_supports_long_logging_periods(self):
        interval_ns, per_second_scale = _throughput_interval_scale(1_000_000)
        self.assertEqual(interval_ns, 1_000_000_000_000_000)
        self.assertEqual(per_second_scale, 1e-6)

    def test_throughput_interval_rejects_nonpositive_or_nonfinite_values(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _throughput_interval_scale(value)

    def test_analytical_report_cannot_regress_below_python_floor(self):
        with self.assertRaisesRegex(RuntimeError, "regressed"):
            _analytical_report_time(900, 1_000)
        self.assertEqual(_analytical_report_time(1_000, 1_000), 1_000)
        self.assertEqual(_analytical_report_time(1_100, 1_000), 1_100)
        with self.assertRaises(ValueError):
            _analytical_report_time(-1, 0)

    def test_analytical_advance_protocol_requires_forward_target(self):
        self.assertEqual(
            _analytical_advance_command(100, 625),
            "advance-to:625",
        )
        for target in (99, 100):
            with self.subTest(target=target), self.assertRaises(ValueError):
                _analytical_advance_command(100, target)

    def test_long_idle_gap_keeps_one_astra_poll(self):
        self.assertEqual(
            _idle_fast_forward_delta(5_000_000, 10_005_000_000),
            9_999_000_000,
        )

    def test_sub_millisecond_gap_cancels_idle_poll_overshoot(self):
        adjustment = _idle_fast_forward_delta(0, 999_999)
        self.assertEqual(adjustment, -1)
        self.assertEqual(0 + 1_000_000 + adjustment, 999_999)

    def test_ns3_keeps_its_100ns_idle_tick(self):
        self.assertEqual(
            _idle_fast_forward_delta(5_000, 1_005_000, "ns3"),
            999_900,
        )

    def test_ready_request_waiting_for_hbm_reclaim_is_idle_wakeup(self):
        wakeup = _next_idle_wakeup_ns(
            100,
            [_scheduler(request_ready=0, memory_wait=1_000)],
            _Router(),
        )
        self.assertEqual(wakeup, 1_000)

    def test_runnable_peer_prevents_fast_forward(self):
        wakeup = _next_idle_wakeup_ns(
            100,
            [
                _scheduler(request_ready=0, memory_wait=1_000),
                _scheduler(request_ready=100),
            ],
            _Router(next_pending=500),
        )
        self.assertIsNone(wakeup)

    def test_attempted_blocked_head_keeps_exact_handoff_wakeup(self):
        wakeup = _next_idle_wakeup_ns(
            100,
            [_scheduler(request_ready=100, instance_id=1)],
            _Router(next_handoff=500),
            known_nonrunnable_instance_id=1,
        )

        self.assertEqual(wakeup, 500)

    def test_unattempted_runnable_peer_still_prevents_fast_forward(self):
        wakeup = _next_idle_wakeup_ns(
            100,
            [
                _scheduler(request_ready=100, instance_id=0),
                _scheduler(request_ready=100, instance_id=1),
            ],
            _Router(next_handoff=500),
            known_nonrunnable_instance_id=1,
        )

        self.assertIsNone(wakeup)

    def test_attempted_blocked_head_uses_earliest_dependency(self):
        manager = _Manager(internal_event=425)
        wakeup = _next_idle_wakeup_ns(
            100,
            [_scheduler(
                request_ready=100, manager=manager, instance_id=1)],
            _Router(next_pending=600, next_handoff=500),
            known_nonrunnable_instance_id=1,
        )

        self.assertEqual(wakeup, 425)

    def test_earliest_request_or_reclaim_event_wins(self):
        wakeup = _next_idle_wakeup_ns(
            100,
            [
                _scheduler(request_ready=0, memory_wait=1_000),
                _scheduler(request_ready=800),
            ],
            _Router(next_pending=600),
        )
        self.assertEqual(wakeup, 600)

    def test_inflight_work_prevents_fast_forward(self):
        wakeup = _next_idle_wakeup_ns(
            100,
            [_scheduler(memory_wait=1_000, inflight=True)],
            _Router(next_pending=500),
        )
        self.assertIsNone(wakeup)

    def test_idle_prefill_cannot_jump_past_inflight_decode(self):
        wakeup = _next_idle_wakeup_ns(
            100,
            [
                _scheduler(request_ready=1_000),
                _scheduler(inflight=True),
            ],
            _Router(next_pending=500),
        )
        self.assertIsNone(wakeup)

    def test_decode_handoff_reclaim_is_idle_wakeup(self):
        wakeup = _next_idle_wakeup_ns(
            100,
            [_scheduler()],
            _Router(next_pending=900, next_handoff=500),
        )
        self.assertEqual(wakeup, 500)

    def test_ready_request_waiting_for_sync_swap_is_idle_wakeup(self):
        wakeup = _next_idle_wakeup_ns(
            100,
            [_scheduler(request_ready=0, swap_wait=750)],
            _Router(),
        )
        self.assertEqual(wakeup, 750)

    def test_ready_request_waiting_for_model_fabric_is_idle_wakeup(self):
        wakeup = _next_idle_wakeup_ns(
            100,
            [_scheduler(request_ready=0, fabric_wait=625)],
            _Router(),
        )
        self.assertEqual(wakeup, 625)

    def test_sync_swap_does_not_hide_runnable_peer(self):
        wakeup = _next_idle_wakeup_ns(
            100,
            [
                _scheduler(request_ready=0, swap_wait=750),
                _scheduler(request_ready=100),
            ],
            _Router(),
        )
        self.assertIsNone(wakeup)

    def test_all_dependency_sources_compete_for_earliest_idle_wakeup(self):
        wakeup = _next_idle_wakeup_ns(
            100,
            [
                _scheduler(
                    request_ready=700,
                    memory_wait=650,
                    swap_wait=600,
                    fabric_wait=550,
                ),
            ],
            _Router(next_pending=500, next_handoff=450),
        )
        self.assertEqual(wakeup, 450)

    def test_empty_scheduler_dependency_sources_all_compete(self):
        wakeup = _next_idle_wakeup_ns(
            100,
            [
                _scheduler(
                    memory_wait=700,
                    swap_wait=600,
                    fabric_wait=500,
                ),
            ],
            _Router(),
        )
        self.assertEqual(wakeup, 500)

    def test_idle_capacity_waiter_wakes_for_manager_migration_commit(self):
        manager = _Manager(internal_event=425)
        wakeup = _next_idle_wakeup_ns(
            100,
            [_scheduler(manager=manager)],
            _Router(),
        )
        self.assertEqual(wakeup, 425)


if __name__ == "__main__":
    unittest.main()

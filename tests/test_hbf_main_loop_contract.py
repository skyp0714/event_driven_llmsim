import unittest
from types import SimpleNamespace

from serving.__main__ import (
    _analytical_idle_endpoint_command,
    _full_model_hbf_arrival_routing_blocked,
    _full_model_hbf_dispatch_blocked,
    _measurement_drain_complete,
    _next_exact_control_wakeup_ns,
)
from serving.core.controller import (
    ExactControlSchedule,
    SameTimeControlBarrier,
)
from serving.core.hbf_online_runtime import FullModelHBFOnlineRuntime


class _Router:
    def __init__(self, *, next_arrival=None, next_handoff=None):
        self.next_arrival = next_arrival
        self.next_handoff = next_handoff

    def get_next_pending_arrival(self):
        return self.next_arrival

    def get_next_decode_handoff_wakeup(self):
        return self.next_handoff


class _Adapter:
    def __init__(
            self, *, pending=False, wakeup=None,
            deferred_completion=False):
        self.pending = pending
        self.wakeup = wakeup
        self.deferred_completion = deferred_completion
        self.wakeup_calls = []

    def has_pending(self):
        return self.pending

    def has_deferred_hbf_completions(self):
        return self.deferred_completion

    def next_wakeup_ns(
            self, current_ns, *, router_arrival_ns=None,
            extra_candidates=()):
        self.wakeup_calls.append((
            current_ns,
            router_arrival_ns,
            tuple(extra_candidates),
        ))
        return self.wakeup


def _scheduler(*, inflight=False):
    return SimpleNamespace(
        inflight=[object()] if inflight else [],
        request=[],
        memory_wait_until_ns=None,
        model_fabric_wait_until_ns=None,
        instance_id=0,
    )


class FullModelHBFMainLoopContractTests(unittest.TestCase):
    def test_tie_barrier_fences_older_arrival_routing(self):
        adapter = _Adapter()
        runtime = SimpleNamespace(adapter=adapter)
        barrier = SameTimeControlBarrier()

        self.assertFalse(
            _full_model_hbf_arrival_routing_blocked(runtime, barrier))
        barrier.arm(50)
        self.assertTrue(
            _full_model_hbf_arrival_routing_blocked(runtime, barrier))
        barrier.complete("python-tie.0", 50)
        adapter.deferred_completion = True
        self.assertTrue(
            _full_model_hbf_arrival_routing_blocked(runtime, barrier))
        adapter.deferred_completion = False
        self.assertFalse(
            _full_model_hbf_arrival_routing_blocked(runtime, barrier))
        self.assertFalse(
            _full_model_hbf_arrival_routing_blocked(None, barrier))

    def test_pending_tie_barrier_fences_new_gpu_dispatch(self):
        runtime = object()
        barrier = SameTimeControlBarrier()
        self.assertFalse(
            _full_model_hbf_dispatch_blocked(runtime, barrier))
        barrier.arm(50)
        self.assertTrue(
            _full_model_hbf_dispatch_blocked(runtime, barrier))
        barrier.complete("python-tie.0", 50)
        self.assertFalse(
            _full_model_hbf_dispatch_blocked(runtime, barrier))
        self.assertFalse(
            _full_model_hbf_dispatch_blocked(None, barrier))

    def test_measurement_drain_waits_for_all_adapter_and_tie_state(self):
        adapter = _Adapter(pending=True)
        runtime = SimpleNamespace(adapter=adapter)
        exact = ExactControlSchedule()
        barrier = SameTimeControlBarrier()
        schedulers = [_scheduler()]

        arguments = (
            True,
            schedulers,
            {"dp": {}},
            {},
            None,
            exact,
            runtime,
            barrier,
        )
        self.assertFalse(_measurement_drain_complete(*arguments))

        adapter.pending = False
        command = barrier.arm(40)
        self.assertEqual(
            command,
            "control-after-endpoints\tpython-tie.0\t40",
        )
        self.assertFalse(_measurement_drain_complete(*arguments))

        self.assertTrue(barrier.complete("python-tie.0", 40))
        self.assertTrue(_measurement_drain_complete(*arguments))

    def test_exact_wakeup_includes_adapter_owned_python_event(self):
        adapter = _Adapter(wakeup=250)
        runtime = SimpleNamespace(adapter=adapter)
        router = _Router(next_arrival=500, next_handoff=600)

        self.assertEqual(
            _next_exact_control_wakeup_ns(
                100,
                [_scheduler(inflight=True)],
                router,
                full_model_hbf_runtime=runtime,
            ),
            250,
        )
        self.assertEqual(adapter.wakeup_calls, [(100, 500, ())])

    def test_same_time_barrier_namespace_is_disjoint_from_future_control(self):
        exact = ExactControlSchedule()
        barrier = SameTimeControlBarrier()

        self.assertEqual(
            exact.arm(80, 20),
            "control-at\tpython-ready.0\t80",
        )
        self.assertEqual(
            barrier.arm(20),
            "control-after-endpoints\tpython-tie.0\t20",
        )
        self.assertTrue(barrier.owns("python-tie.0"))
        self.assertFalse(barrier.owns("python-ready.0"))
        self.assertTrue(barrier.complete("python-tie.0", 20))
        self.assertTrue(exact.complete("python-ready.0", 80))

    def test_hbf_callback_is_a_causal_wakeup_for_parked_gpu_group(self):
        self.assertEqual(
            _analytical_idle_endpoint_command(
                "analytical-congestion-aware",
                True,
                True,
                False,
            ),
            "park",
        )
        self.assertEqual(
            _analytical_idle_endpoint_command(
                "analytical-congestion-aware",
                True,
                True,
                True,
            ),
            "pass",
        )

    def test_reporting_schedulers_include_materialized_hbf_requests(self):
        hbf_request = SimpleNamespace(id=9)
        gpu_scheduler = SimpleNamespace(
            instance_id=0, pd_type="decode", done=[])
        runtime = FullModelHBFOnlineRuntime.__new__(
            FullModelHBFOnlineRuntime)
        runtime.options = SimpleNamespace(server_id=3)
        runtime.completed_requests = [hbf_request]

        sources = runtime.reporting_schedulers([gpu_scheduler])

        self.assertEqual(len(sources), 2)
        self.assertIs(sources[0], gpu_scheduler)
        self.assertEqual(sources[1].pd_type, "hbf")
        self.assertEqual(sources[1].instance_id, "hbf:3")
        self.assertEqual(sources[1].done, [hbf_request])


if __name__ == "__main__":
    unittest.main()

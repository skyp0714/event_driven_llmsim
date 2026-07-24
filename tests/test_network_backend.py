import os
import unittest

from serving.__main__ import (
    _NETWORK_BACKEND_CHOICES,
    _analytical_idle_endpoint_command,
    _has_dispatched_model_work,
    _has_partially_observed_model_completion,
    _idle_fast_forward_delta,
    _resolve_analytical_binary,
)


class NetworkBackendTest(unittest.TestCase):
    def test_congestion_aware_backend_is_a_cli_choice(self):
        self.assertIn(
            "analytical-congestion-aware", _NETWORK_BACKEND_CHOICES)

    def test_default_analytical_binary_is_congestion_unaware(self):
        path = _resolve_analytical_binary("/astra", "analytical")

        self.assertEqual(
            path,
            os.path.join(
                "/astra",
                "build/astra_analytical/build/AnalyticalAstra/bin/"
                "AnalyticalAstra",
            ),
        )

    def test_congestion_aware_binary_uses_dedicated_build_target(self):
        path = _resolve_analytical_binary(
            "/astra", "analytical-congestion-aware")

        self.assertEqual(
            path,
            os.path.join(
                "/astra",
                "build/astra_analytical/build/AstraCongestion/bin/"
                "AstraCongestion",
            ),
        )

    def test_non_analytical_backend_is_rejected_by_binary_resolver(self):
        with self.assertRaisesRegex(ValueError, "Not an analytical"):
            _resolve_analytical_binary("/astra", "ns3")

    def test_congestion_aware_idle_poll_matches_analytical_frontend(self):
        self.assertEqual(
            _idle_fast_forward_delta(
                5_000_000,
                10_005_000_000,
                "analytical-congestion-aware",
            ),
            9_999_000_000,
        )

    def test_only_analytical_group_controller_parks_for_backend_wakeup(self):
        self.assertEqual(
            _analytical_idle_endpoint_command(
                "analytical-congestion-aware", True, True),
            "park",
        )
        for backend, controller, wakeup in (
                ("analytical-congestion-aware", False, True),
                ("analytical", True, False),
                ("ns3", True, True)):
            with self.subTest(
                    backend=backend, controller=controller, wakeup=wakeup):
                self.assertEqual(
                    _analytical_idle_endpoint_command(
                        backend, controller, wakeup),
                    "pass",
                )

    def test_formed_agentic_batch_is_not_a_live_astra_graph(self):
        formed = type("Batch", (), {
            "agentic_astra_dispatch_time_ns": None,
        })()
        dispatched = type("Batch", (), {
            "agentic_astra_dispatch_time_ns": 0,
        })()
        scheduler = type("Scheduler", (), {"inflight": [formed]})()

        self.assertFalse(_has_dispatched_model_work([scheduler], object()))
        scheduler.inflight.append(dispatched)
        self.assertTrue(_has_dispatched_model_work([scheduler], object()))

    def test_partial_peer_completion_suppresses_endpoint_park(self):
        partial = type("Batch", (), {"end": [11]})()
        complete_callback_not_seen = type("Batch", (), {"end": []})()
        scheduler = type("Scheduler", (), {
            "inflight": [complete_callback_not_seen],
        })()

        self.assertFalse(
            _has_partially_observed_model_completion([scheduler]))
        scheduler.inflight = [partial]
        self.assertTrue(
            _has_partially_observed_model_completion([scheduler]))
        self.assertEqual(
            _analytical_idle_endpoint_command(
                "analytical-congestion-aware",
                True,
                True,
                has_partially_observed_completion=True,
            ),
            "pass",
        )


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import random
import unittest

from serving.core.gpu_pd_dual_oracle import (
    DualOracleDeadlockError,
    DualStrictInfiniteHBMOracle,
    ROUTE_BALANCED_TRACE_WORK,
)
from serving.core.gpu_pd_latency import load_p4d4_gpu_config
from serving.core.hbf_comparison_workload import (
    CallSpec,
    ScheduledSession,
    SessionSpec,
    build_offered_plan,
    load_fixed_comparison_workload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_CONFIG = (
    REPO_ROOT / "configs" / "wakekv_hbf"
    / "p4d4_gpu_server.json"
)
TRACE_PATH = (
    Path.home()
    / "llmsim-data"
    / "tracelab-schema3-sps0.2-final.jsonl"
)


class DualStrictInfiniteHBMOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = load_p4d4_gpu_config(GPU_CONFIG)

    def make_system(
            self, *, max_tokens=256, chunk=64,
            validate_every_event=True,
            route_policy="offer_index_mod_2_sticky"):
        return DualStrictInfiniteHBMOracle(
            repo_root=REPO_ROOT,
            hardware=self.hardware,
            max_num_batched_tokens=max_tokens,
            max_num_seqs=16,
            max_prefill_chunk_tokens=chunk,
            validate_every_event=validate_every_event,
            route_policy=route_policy,
        )

    @staticmethod
    def session(
            offer_index, *, session_id=None, arrival_ns=0,
            calls=((100, 2, 0, 0),), source_index=None):
        if session_id is None:
            session_id = f"s{offer_index}"
        if source_index is None:
            source_index = offer_index
        call_specs = []
        for call_index, (
                input_tokens, output_tokens, prefix_tokens,
                tool_duration_ns) in enumerate(calls):
            call_specs.append(CallSpec(
                session_id=session_id,
                source_index=source_index,
                call_index=call_index,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_duration_ns=tool_duration_ns,
                cached_prefix_tokens=prefix_tokens,
                fresh_input_tokens=input_tokens - prefix_tokens,
                lineage_status=None,
                inter_turn_gap_type=None,
            ))
        spec = SessionSpec(
            source_index=source_index,
            session_id=session_id,
            source_arrival_time_ns=arrival_ns,
            source_session_identity_sha256=None,
            calls=tuple(call_specs),
        )
        return ScheduledSession(
            offer_index=offer_index,
            session=spec,
            arrival_time_ns=arrival_ns,
            unit_interarrival=0.0,
            unit_arrival_time=0.0,
        )

    def test_sticky_modulo_routing_and_stable_preassigned_ids(self):
        scheduled = [
            self.session(3, source_index=10),
            self.session(0, source_index=40),
            self.session(2, source_index=20),
            self.session(1, source_index=30),
        ]
        system = self.make_system()
        system.load(scheduled)

        self.assertEqual(
            [spec.request_id for spec in system.call_specs],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            [spec.offer_index for spec in system.call_specs],
            [3, 2, 1, 0],
        )
        self.assertEqual(
            [spec.source_index for spec in system.call_specs],
            [10, 20, 30, 40],
        )
        self.assertEqual(system.node_for_session("s0"), 0)
        self.assertEqual(system.node_for_session("s1"), 1)
        self.assertEqual(system.node_for_session("s2"), 0)
        self.assertEqual(system.node_for_session("s3"), 1)

        reordered = self.make_system()
        reordered.load(reversed(scheduled))
        self.assertEqual(system.call_specs, reordered.call_specs)
        self.assertEqual(
            system.report()["routing_identity_sha256"],
            reordered.report()["routing_identity_sha256"],
        )

    def test_thirty_two_sessions_split_sixteen_per_node(self):
        system = self.make_system()
        system.load([
            self.session(index) for index in range(32)
        ])
        counts = {
            node_id: sum(
                spec.call_index == 0 and spec.node_id == node_id
                for spec in system.call_specs
            )
            for node_id in range(2)
        }
        self.assertEqual(counts, {0: 16, 1: 16})

    @unittest.skipUnless(
        TRACE_PATH.exists(), "TraceLab release not present")
    def test_balanced_static_route_is_seed_invariant_and_balanced(self):
        workload = load_fixed_comparison_workload(TRACE_PATH)
        mappings = []
        ratios = []
        for seed in (101, 1201):
            scheduled = build_offered_plan(
                workload.sessions,
                seed=seed,
            ).at_rate(5.0)
            system = self.make_system(
                route_policy=ROUTE_BALANCED_TRACE_WORK)
            system.load(scheduled)
            mappings.append(dict(system._route_by_session))
            report = system.report()
            ratios.append(
                report["routing_trace_work_proxy"]["max_over_min"])
            counts = [
                sum(
                    node_id == candidate
                    for node_id in system._route_by_session.values()
                )
                for candidate in (0, 1)
            ]
            self.assertEqual(counts, [16, 16])
        self.assertEqual(mappings[0], mappings[1])
        self.assertTrue(all(ratio <= 1.07 for ratio in ratios))

    def test_successor_release_is_completion_plus_tool_gap(self):
        gap_ns = 123_456_789
        scheduled = [self.session(
            0,
            calls=(
                (100, 2, 0, gap_ns),
                (110, 2, 101, 0),
            ),
        )]
        system = self.make_system()
        completed = system.run(scheduled)

        first_id = system.request_id_for("s0::call-0")
        second_id = system.request_id_for("s0::call-1")
        first = system._runtime_calls[first_id]
        second = system._runtime_calls[second_id]
        self.assertEqual(
            second.release_ns,
            first.user_completion_ns + gap_ns,
        )
        self.assertEqual(second.operational_hit_tokens, 101)
        self.assertEqual(
            [call.key.sub_request_index for call in completed],
            [0, 1],
        )

    def test_zero_gap_successor_waits_for_lineage_handoff(self):
        scheduled = [self.session(
            0,
            calls=(
                (1_000, 1, 0, 0),
                (1_010, 1, 1_000, 0),
            ),
        )]
        system = self.make_system()
        system.run(scheduled)

        first = system._runtime_calls[
            system.request_id_for("s0::call-0")]
        second = system._runtime_calls[
            system.request_id_for("s0::call-1")]
        self.assertEqual(second.release_ns, first.user_completion_ns)
        self.assertGreater(
            first.internal_completion_ns,
            first.user_completion_ns,
        )
        self.assertGreaterEqual(
            second.prepare_start_ns,
            first.internal_completion_ns,
        )

    def test_cotimed_arrivals_batch_once_per_node(self):
        scheduled = [
            self.session(index, arrival_ns=1_000)
            for index in range(4)
        ]
        system = self.make_system(max_tokens=1_024, chunk=1_024)
        system.run(scheduled)

        for node_id, node in enumerate(system.nodes):
            first_p_batch = next(
                batch for batch in node.pool.batch_history
                if batch.stage == "p")
            request_ids = [
                item.request_id for item in first_p_batch.items]
            expected = [
                spec.request_id for spec in system.call_specs
                if spec.node_id == node_id
            ]
            self.assertEqual(request_ids, expected)

    def test_cotimed_cross_node_successors_share_global_boundary(self):
        scheduled = [
            self.session(
                offer_index,
                calls=(
                    (100, 2, 0, 0),
                    (50, 1, 0, 0),
                ),
            )
            for offer_index in (0, 1)
        ]
        system = self.make_system(max_tokens=1_024, chunk=1_024)
        system.run(scheduled)
        first_calls = [
            system._runtime_calls[
                system.request_id_for(f"s{index}::call-0")]
            for index in (0, 1)
        ]
        successors = [
            system._runtime_calls[
                system.request_id_for(f"s{index}::call-1")]
            for index in (0, 1)
        ]
        self.assertEqual(
            first_calls[0].user_completion_ns,
            first_calls[1].user_completion_ns,
        )
        self.assertEqual(
            successors[0].release_ns,
            successors[1].release_ns,
        )
        self.assertEqual(
            successors[0].release_ns,
            first_calls[0].user_completion_ns,
        )

    def test_nodes_are_independent_and_never_run_locally_to_idle(self):
        scheduled = [
            self.session(0, calls=((100, 2, 0, 1_000), (110, 1, 101, 0))),
            self.session(1, calls=((100, 2, 0, 1_000), (110, 1, 101, 0))),
        ]
        system = self.make_system()
        self.assertIsNot(
            system.nodes[0].calendar,
            system.nodes[1].calendar,
        )

        def forbidden():
            raise AssertionError("node.run_until_idle must not be called")

        for node in system.nodes:
            node.run_until_idle = forbidden
        completed = system.run(scheduled)
        self.assertEqual(len(completed), 4)

    def test_completion_order_is_deterministic_for_cotimed_nodes(self):
        scheduled = [
            self.session(0),
            self.session(1),
        ]
        system = self.make_system()
        completed = system.run(scheduled)
        self.assertEqual(
            [call.key.session_id for call in completed],
            ["s0", "s1"],
        )

    def test_completed_requests_are_immediate_immutable_snapshots(self):
        system = self.make_system()
        system.run([self.session(
            0,
            calls=(
                (1_000, 1, 0, 0),
                (1_010, 1, 1_000, 0),
            ),
        )])
        first = system.completed_requests[0]
        runtime = system._runtime_calls[
            system.request_id_for("s0::call-0")]
        self.assertEqual(first.key.session_id, runtime.session_id)
        self.assertEqual(first.release_ns, runtime.release_ns)
        self.assertEqual(
            first.first_token_ns,
            runtime.user_completion_ns,
        )
        self.assertEqual(
            runtime.state.value,
            "internal_complete",
        )

    def test_full_drain_report_preserves_routing_identity(self):
        scheduled = [
            self.session(0),
            self.session(
                1,
                calls=((100, 1, 0, 0), (105, 2, 100, 0)),
            ),
        ]
        system = self.make_system()
        system.run(scheduled)
        report = system.report()

        self.assertEqual(
            report["mode"],
            "dual_strict_infinite_hbm_residency_oracle",
        )
        self.assertEqual(
            report["routing_policy"],
            "offer_index_mod_2_sticky",
        )
        self.assertTrue(report["finished"])
        self.assertEqual(report["metrics"]["scheduled_sessions"], 2)
        self.assertEqual(report["metrics"]["scheduled_calls"], 3)
        self.assertEqual(report["metrics"]["completed_calls"], 3)
        self.assertEqual(report["full_drain"]["identity_count"], 3)
        self.assertEqual(
            report["call_full_drain"],
            report["full_drain"],
        )
        self.assertEqual(
            report["session_full_drain"]["identity_count"],
            2,
        )
        self.assertEqual(
            report["full_drain"]["expected_set_sha256"],
            report["full_drain"]["completion_set_sha256"],
        )
        self.assertEqual(
            [row["node_id"] for row in report["routing"]],
            [0, 1],
        )
        json.dumps(report)

    def test_deterministic_randomized_dual_node_stress(self):
        rng = random.Random(20260723)
        scheduled = []
        for session_index in range(32):
            calls = []
            prior_final = 0
            for call_index in range(5):
                output_tokens = rng.randint(1, 5)
                if call_index == 0:
                    input_tokens = rng.randint(32, 2_048)
                    prefix_tokens = 0
                elif call_index % 3:
                    input_tokens = prior_final + rng.randint(1, 64)
                    prefix_tokens = prior_final
                else:
                    input_tokens = max(
                        1,
                        prior_final - rng.randint(1, prior_final),
                    )
                    prefix_tokens = 0
                calls.append((
                    input_tokens,
                    output_tokens,
                    prefix_tokens,
                    rng.choice((0, 1, 10_000, 1_000_000)),
                ))
                prior_final = input_tokens + output_tokens - 1
            scheduled.append(self.session(
                session_index,
                arrival_ns=rng.randrange(0, 10_000_000),
                calls=tuple(calls),
                source_index=1_000 + session_index,
            ))

        first = self.make_system(max_tokens=1_024, chunk=256)
        second = self.make_system(max_tokens=1_024, chunk=256)
        first_completed = first.run(scheduled)
        second_completed = second.run(reversed(scheduled))
        self.assertEqual(first_completed, second_completed)
        self.assertEqual(len(first_completed), 160)
        self.assertEqual(
            first.report()["call_full_drain"],
            second.report()["call_full_drain"],
        )
        self.assertEqual(
            first.report()["session_full_drain"],
            second.report()["session_full_drain"],
        )
        for system in (first, second):
            self.assertEqual(
                sum(
                    node.hbm.p_used_bytes_per_rank
                    for node in system.nodes
                ),
                0,
            )
            self.assertEqual(
                sum(
                    node.hbm.d_used_bytes_per_rank
                    for node in system.nodes
                ),
                0,
            )
            self.assertEqual(
                system.metrics.completed_calls,
                system.metrics.scheduled_calls,
            )

    def test_deadlock_is_detected_when_future_release_is_lost(self):
        system = self.make_system()
        system.load([self.session(0)])
        system._release_heap.clear()
        with self.assertRaisesRegex(
                DualOracleDeadlockError,
                "no future event"):
            system.run()

    def test_invalid_duplicate_offer_is_rejected_before_mutation(self):
        system = self.make_system()
        with self.assertRaisesRegex(ValueError, "duplicate offers"):
            system.load([
                self.session(0, session_id="a"),
                self.session(0, session_id="b"),
            ])
        self.assertEqual(system.call_specs, ())

    def test_system_can_only_be_loaded_once(self):
        system = self.make_system()
        scheduled = [self.session(0)]
        system.load(scheduled)
        with self.assertRaisesRegex(RuntimeError, "already loaded"):
            system.load(scheduled)

    def test_sweep_mode_matches_strict_mode_and_propagates(self):
        scheduled = [
            self.session(
                index,
                arrival_ns=index * 1_000,
                calls=(
                    (100 + index, 3, 0, 0),
                    (110 + index, 2, 102 + index, 0),
                ),
            )
            for index in range(4)
        ]
        strict = self.make_system()
        sweep = self.make_system(validate_every_event=False)
        self.assertEqual(strict.run(scheduled), sweep.run(scheduled))
        self.assertFalse(sweep.validate_every_event)
        self.assertTrue(all(
            not node.validate_every_event
            and not node.pool.validate_every_event
            and not node.pool.retain_detailed_history
            and not node.pool.batch_history
            and not node.pool.handoff_history
            for node in sweep.nodes
        ))
        sweep.assert_invariants()

    def test_validate_every_event_must_be_boolean(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            DualStrictInfiniteHBMOracle(
                repo_root=REPO_ROOT,
                hardware=self.hardware,
                validate_every_event=1,
            )
        with self.assertRaisesRegex(ValueError, "route_policy"):
            DualStrictInfiniteHBMOracle(
                repo_root=REPO_ROOT,
                hardware=self.hardware,
                route_policy="unknown",
            )


if __name__ == "__main__":
    unittest.main()

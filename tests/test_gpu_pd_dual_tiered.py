import json
from pathlib import Path
import random
import unittest

from serving.core.gpu_pd_dual_oracle import (
    ROUTE_BALANCED_TRACE_WORK,
)
from serving.core.gpu_pd_dual_tiered import (
    DualFiniteHBMTieredBaseline,
    DualTieredDeadlockError,
)
from serving.core.gpu_pd_latency import load_p4d4_gpu_config
from serving.core.gpu_pd_tiered_node import TieredCallState
from serving.core.hbf_comparison_workload import (
    CallSpec,
    ScheduledSession,
    SessionSpec,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_CONFIG = (
    REPO_ROOT / "configs" / "wakekv_hbf"
    / "p4d4_gpu_server.json"
)
POLICIES = (
    "hbm_lru_recompute",
    "ssd_direct",
    "cpu_ssd",
)
EXPECTED_32_CALL_SET_SHA256 = (
    "108ff19c7d2361d8006d15bebab3037ee716abb7f81bbc3928ea31713888dd16"
)
EXPECTED_32_SESSION_SET_SHA256 = (
    "9e55735a822213008f526eeff838e41f7cb83710e6152b67a139ceee87e18b49"
)
EXPECTED_32_COMPLETION_ORDER_SHA256 = {
    "hbm_lru_recompute": (
        "64ef7232d53ea0bb966f4b691dcfc617fe344f46eb17696855f9406390038906"
    ),
    "ssd_direct": (
        "dfb6f52f21b0e32a989687b4871a987445c9d825768a50a834f3e701e10dc27d"
    ),
    "cpu_ssd": (
        "dfb6f52f21b0e32a989687b4871a987445c9d825768a50a834f3e701e10dc27d"
    ),
}


class DualFiniteHBMTieredBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = load_p4d4_gpu_config(GPU_CONFIG)
        cls.block_per_rank = (
            cls.hardware.kv_capacity_bytes_per_rank(16))
        cls.block_aggregate = (
            cls.block_per_rank * cls.hardware.tp_size)

    def make_system(
            self, policy="cpu_ssd", *,
            p_blocks=64, d_blocks=8,
            cpu_blocks=64, ssd_blocks=512,
            max_tokens=512, chunk=128,
            restore_execution_mode="bulk",
            validate_every_event=True,
            route_policy="offer_index_mod_2_sticky"):
        return DualFiniteHBMTieredBaseline(
            repo_root=REPO_ROOT,
            hardware=self.hardware,
            policy=policy,
            p_capacity_bytes_per_rank=(
                p_blocks * self.block_per_rank),
            d_capacity_bytes_per_rank=(
                d_blocks * self.block_per_rank),
            cpu_capacity_bytes=(
                cpu_blocks * self.block_aggregate),
            ssd_capacity_bytes=(
                ssd_blocks * self.block_aggregate),
            max_num_batched_tokens=max_tokens,
            max_num_seqs=32,
            max_prefill_chunk_tokens=chunk,
            restore_execution_mode=restore_execution_mode,
            validate_every_event=validate_every_event,
            route_policy=route_policy,
        )

    def test_restore_execution_mode_propagates_to_both_nodes(self):
        system = self.make_system(
            restore_execution_mode="layerwise_streaming")

        self.assertEqual(
            system.restore_execution_mode,
            "layerwise_streaming",
        )
        self.assertTrue(all(
            node.restore_execution_mode == "layerwise_streaming"
            and node.lifecycle.restore_execution_mode
            == "layerwise_streaming"
            for node in system.nodes
        ))
        self.assertEqual(
            system.report()["restore_execution_mode"],
            "layerwise_streaming",
        )

    @staticmethod
    def session(
            offer_index, *, session_id=None, arrival_ns=0,
            calls=((64, 2, 0, 0),), source_index=None):
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

    def cohort_32(self):
        sessions = []
        for index in range(32):
            first_input = 32 + 16 * (index % 4)
            first_final = first_input + 1
            resume_input = first_final + 8
            sessions.append(self.session(
                index,
                arrival_ns=index * 1_000,
                source_index=1_000 + index,
                calls=(
                    (first_input, 2, 0, index % 3),
                    (
                        resume_input,
                        2,
                        first_final,
                        (index % 5) * 10,
                    ),
                    (32 + index % 8, 1, 0, 0),
                ),
            ))
        return sessions

    def test_all_policies_propagate_and_physical_nodes_are_isolated(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                system = self.make_system(policy)
                left, right = system.nodes
                self.assertEqual(
                    [node.node_id for node in system.nodes],
                    [0, 1],
                )
                self.assertIsNot(left.calendar, right.calendar)
                self.assertIs(
                    left.calendar, left.lifecycle.calendar)
                self.assertIs(left.calendar, left.pool.calendar)
                self.assertIs(
                    right.calendar, right.lifecycle.calendar)
                self.assertIs(right.calendar, right.pool.calendar)
                for ledger_name in ("p", "d", "cpu", "ssd"):
                    self.assertIsNot(
                        getattr(left.lifecycle, f"{ledger_name}_ledger"),
                        getattr(right.lifecycle, f"{ledger_name}_ledger"),
                    )
                self.assertTrue(all(
                    node.policy == policy
                    and node.lifecycle.policy == policy
                    for node in system.nodes
                ))

    def test_sticky_route_and_request_ids_are_frozen_before_run(self):
        scheduled = [
            self.session(3, source_index=10),
            self.session(0, source_index=40),
            self.session(2, source_index=20),
            self.session(1, source_index=30),
        ]
        system = self.make_system()
        system.load(scheduled)
        reordered = self.make_system()
        reordered.load(reversed(scheduled))

        self.assertEqual(system.call_specs, reordered.call_specs)
        self.assertEqual(
            [spec.request_id for spec in system.call_specs],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            [spec.offer_index for spec in system.call_specs],
            [3, 2, 1, 0],
        )
        self.assertEqual(
            {
                session_id: system.node_for_session(session_id)
                for session_id in ("s0", "s1", "s2", "s3")
            },
            {"s0": 0, "s1": 1, "s2": 0, "s3": 1},
        )
        self.assertEqual(
            system.report()["routing_identity_sha256"],
            reordered.report()["routing_identity_sha256"],
        )

    def test_balanced_static_route_is_precomputed_and_sixteen_each(self):
        scheduled = self.cohort_32()
        first = self.make_system(
            route_policy=ROUTE_BALANCED_TRACE_WORK)
        second = self.make_system(
            route_policy=ROUTE_BALANCED_TRACE_WORK)
        first.load(scheduled)
        second.load(reversed(scheduled))
        self.assertEqual(
            first._route_by_session,
            second._route_by_session,
        )
        self.assertEqual(
            [
                sum(node_id == candidate for node_id in (
                    first._route_by_session.values()))
                for candidate in (0, 1)
            ],
            [16, 16],
        )

    def test_global_boundary_batches_cotimed_arrivals_per_node(self):
        scheduled = [
            self.session(index, arrival_ns=1_000)
            for index in range(4)
        ]
        system = self.make_system(
            d_blocks=32, max_tokens=1_024, chunk=1_024)
        system.run(scheduled)
        for node_id, node in enumerate(system.nodes):
            first_p_batch = next(
                batch for batch in node.pool.batch_history
                if batch.stage == "p")
            self.assertEqual(
                [item.request_id for item in first_p_batch.items],
                [
                    spec.request_id
                    for spec in system.call_specs
                    if spec.node_id == node_id
                ],
            )

    def test_successor_release_uses_user_completion_plus_tool_gap(self):
        gap_ns = 123_456
        scheduled = [self.session(
            0,
            calls=(
                (64, 2, 0, gap_ns),
                (70, 2, 65, 0),
            ),
        )]
        system = self.make_system()
        system.run(scheduled)
        first = system._runtime_calls[
            system.request_id_for("s0::call-0")]
        second = system._runtime_calls[
            system.request_id_for("s0::call-1")]
        self.assertEqual(
            second.release_ns,
            first.user_completion_ns + gap_ns,
        )
        self.assertEqual(second.operational_hit_tokens, 65)
        self.assertEqual(
            first.state, TieredCallState.INTERNAL_COMPLETE)
        self.assertEqual(
            second.state, TieredCallState.INTERNAL_COMPLETE)

    def test_exact_32_session_full_drain_for_every_policy(self):
        scheduled = self.cohort_32()
        for policy in POLICIES:
            with self.subTest(policy=policy):
                system = self.make_system(
                    policy, validate_every_event=False)
                completed = system.run(scheduled)
                report = system.report()
                self.assertEqual(len(completed), 96)
                self.assertEqual(
                    report["call_full_drain"]["identity_count"], 96)
                self.assertEqual(
                    report["session_full_drain"]["identity_count"], 32)
                self.assertEqual(
                    report["call_full_drain"][
                        "expected_set_sha256"],
                    EXPECTED_32_CALL_SET_SHA256,
                )
                self.assertEqual(
                    report["call_full_drain"][
                        "completion_set_sha256"],
                    EXPECTED_32_CALL_SET_SHA256,
                )
                self.assertEqual(
                    report["call_full_drain"][
                        "completion_order_sha256"],
                    EXPECTED_32_COMPLETION_ORDER_SHA256[policy],
                )
                self.assertEqual(
                    report["session_full_drain"][
                        "expected_set_sha256"],
                    EXPECTED_32_SESSION_SET_SHA256,
                )
                self.assertEqual(
                    report["session_full_drain"][
                        "completion_set_sha256"],
                    EXPECTED_32_SESSION_SET_SHA256,
                )
                for node in system.nodes:
                    self.assertTrue(all(
                        lineage.ended
                        for lineage in node.sessions.values()
                    ))
                    self.assertEqual(
                        node.lifecycle.p_ledger.used_bytes, 0)
                    self.assertEqual(
                        node.lifecycle.d_ledger.used_bytes, 0)
                    self.assertEqual(
                        node.lifecycle.cpu_ledger.used_bytes, 0)
                    self.assertEqual(
                        node.lifecycle.ssd_ledger.used_bytes, 0)

    def test_deterministic_randomized_stress_for_every_policy(self):
        rng = random.Random(20260723)
        scheduled = []
        for session_index in range(8):
            calls = []
            prior_final = 0
            for call_index in range(4):
                output_tokens = rng.randint(1, 3)
                if call_index == 0:
                    input_tokens = rng.randint(32, 64)
                    prefix_tokens = 0
                elif call_index % 3:
                    input_tokens = min(
                        112,
                        prior_final + rng.randint(0, 16),
                    )
                    prefix_tokens = min(
                        prior_final, input_tokens)
                else:
                    input_tokens = rng.randint(24, 48)
                    prefix_tokens = 0
                calls.append((
                    input_tokens,
                    output_tokens,
                    prefix_tokens,
                    rng.choice((0, 1, 1_000, 100_000)),
                ))
                prior_final = input_tokens + output_tokens - 1
            scheduled.append(self.session(
                session_index,
                arrival_ns=rng.randrange(0, 1_000_000),
                calls=tuple(calls),
                source_index=5_000 + session_index,
            ))

        for policy in POLICIES:
            with self.subTest(policy=policy):
                first = self.make_system(
                    policy, d_blocks=7,
                    validate_every_event=False)
                second = self.make_system(
                    policy, d_blocks=7,
                    validate_every_event=False)
                first_completed = first.run(scheduled)
                second_completed = second.run(reversed(scheduled))
                self.assertEqual(first_completed, second_completed)
                self.assertEqual(len(first_completed), 32)
                self.assertEqual(
                    first.report()["call_full_drain"],
                    second.report()["call_full_drain"],
                )
                self.assertEqual(
                    first.report()["session_full_drain"],
                    second.report()["session_full_drain"],
                )

    def test_sweep_mode_matches_strict_and_propagates_to_all_layers(self):
        scheduled = self.cohort_32()[:4]
        strict = self.make_system()
        sweep = self.make_system(validate_every_event=False)
        self.assertEqual(strict.run(scheduled), sweep.run(scheduled))
        self.assertFalse(sweep.validate_every_event)
        self.assertTrue(all(
            not node.validate_every_event
            and not node.lifecycle.validate_every_event
            and not node.pool.validate_every_event
            and not node.pool.retain_detailed_history
            and not node.pool.batch_history
            and not node.pool.handoff_history
            for node in sweep.nodes
        ))
        sweep.assert_invariants()

    def test_invalid_schedule_is_rejected_atomically(self):
        system = self.make_system(p_blocks=1)
        valid = self.session(
            0, calls=((16, 1, 0, 0),))
        invalid = self.session(
            1, calls=((17, 1, 0, 0),))
        with self.assertRaisesRegex(ValueError, "P-HBM"):
            system.load((valid, invalid))
        self.assertEqual(system.call_specs, ())
        self.assertEqual(system._spec_by_request, {})
        self.assertEqual(system._route_by_session, {})
        self.assertEqual(system._release_heap, [])
        self.assertTrue(all(not node.calls for node in system.nodes))
        self.assertFalse(system._loaded)

    def test_deadlock_is_detected_if_first_releases_are_lost(self):
        system = self.make_system()
        system.load([self.session(0)])
        system._release_heap.clear()
        with self.assertRaisesRegex(
                DualTieredDeadlockError, "no future event"):
            system.run()

    def test_report_has_per_node_and_aggregate_bottleneck_counters(self):
        system = self.make_system(
            "ssd_direct", validate_every_event=False)
        system.run(self.cohort_32())
        report = system.report()
        bottlenecks = report["bottleneck_counters"]
        self.assertEqual(
            report["mode"], "dual_finite_hbm_p4d4_tiering")
        self.assertEqual(report["policy"], "ssd_direct")
        self.assertEqual(len(bottlenecks["per_node"]), 2)
        aggregate = bottlenecks["aggregate"]
        for metric_group in (
                "node_metrics",
                "lifecycle_metrics",
                "pool_metrics"):
            per_node = bottlenecks["per_node"]
            for key, value in aggregate[metric_group].items():
                samples = [
                    row[metric_group][key]
                    for row in per_node
                ]
                expected = (
                    max(samples)
                    if key.startswith("max_")
                    else sum(samples)
                )
                self.assertEqual(value, expected)
        resources_by_node = [
            set(row["resource_busy_ns"])
            for row in bottlenecks["per_node"]
        ]
        self.assertTrue(resources_by_node[0])
        self.assertTrue(resources_by_node[1])
        self.assertTrue(all(
            name.startswith("gpu-node-0-")
            for name in resources_by_node[0]
        ))
        self.assertTrue(all(
            name.startswith("gpu-node-1-")
            for name in resources_by_node[1]
        ))
        self.assertFalse(
            resources_by_node[0] & resources_by_node[1])
        self.assertIn(
            "does not balance tier I/O",
            report["routing_balance_limit"],
        )
        self.assertEqual(
            bottlenecks["horizon_ns"], system.current_ns)
        self.assertIn(
            "retry attempts",
            bottlenecks["deferral_counter_semantics"],
        )
        for row in bottlenecks["per_node"]:
            for utilization in row["resource_utilization"].values():
                self.assertGreaterEqual(utilization, 0.0)
                self.assertLessEqual(utilization, 1.0)
            for ledger in row["ledgers"].values():
                self.assertGreater(ledger["capacity_bytes"], 0)
                self.assertGreaterEqual(ledger["peak_fraction"], 0.0)
                self.assertLessEqual(ledger["peak_fraction"], 1.0)
                self.assertEqual(ledger["final_used_bytes"], 0)
        json.dumps(report)

    def test_constructor_rejects_invalid_policy_and_boolean(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.make_system("unknown")
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.make_system(validate_every_event=1)


if __name__ == "__main__":
    unittest.main()

from dataclasses import asdict
from pathlib import Path
import unittest

from serving.core.gpu_pd_dual_oracle import DUAL_ORACLE_NODE_COUNT
from serving.core.gpu_pd_latency import load_p4d4_gpu_config
from serving.core.gpu_pd_single_system import (
    SINGLE_GPU_NODE_COUNT,
    SINGLE_NODE_ROUTE_POLICY,
    SingleFiniteHBMTieredBaseline,
    SingleP4D4DeadlockError,
    SingleStrictInfiniteHBMOracle,
)
from serving.core.hbf_comparison_cell import (
    validate_causal_release_contract,
    validate_system_call_projection,
)
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


class SingleP4D4SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = load_p4d4_gpu_config(GPU_CONFIG)
        cls.block_per_rank = (
            cls.hardware.kv_capacity_bytes_per_rank(16))
        cls.block_aggregate = (
            cls.block_per_rank * cls.hardware.tp_size)

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
        session = SessionSpec(
            source_index=source_index,
            session_id=session_id,
            source_arrival_time_ns=arrival_ns,
            source_session_identity_sha256=None,
            calls=tuple(call_specs),
        )
        return ScheduledSession(
            offer_index=offer_index,
            session=session,
            arrival_time_ns=arrival_ns,
            unit_interarrival=(
                0.0 if offer_index == 0 else 1.0),
            unit_arrival_time=float(offer_index),
        )

    def make_baseline(
            self, *, policy="ssd_direct",
            p_blocks=128, d_blocks=128,
            cpu_blocks=512, ssd_blocks=2_048,
            validate_every_event=True):
        return SingleFiniteHBMTieredBaseline(
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
            max_num_batched_tokens=1_024,
            max_num_seqs=32,
            max_prefill_chunk_tokens=256,
            validate_every_event=validate_every_event,
        )

    def make_oracle(self, *, validate_every_event=True):
        return SingleStrictInfiniteHBMOracle(
            repo_root=REPO_ROOT,
            hardware=self.hardware,
            max_num_batched_tokens=1_024,
            max_num_seqs=32,
            max_prefill_chunk_tokens=256,
            validate_every_event=validate_every_event,
        )

    def paired_schedule(self):
        return (
            self.session(
                0,
                session_id="later-source",
                arrival_ns=1_000,
                source_index=20,
                calls=(
                    (64, 2, 0, 123_456),
                    (72, 2, 65, 0),
                ),
            ),
            self.session(
                1,
                session_id="earlier-source",
                arrival_ns=1_000,
                source_index=10,
                calls=((80, 2, 0, 0),),
            ),
        )

    def test_is_one_physical_gpu_and_leaves_frozen_dual_count_unchanged(self):
        baseline = self.make_baseline()
        oracle = self.make_oracle()

        self.assertEqual(SINGLE_GPU_NODE_COUNT, 1)
        self.assertEqual(DUAL_ORACLE_NODE_COUNT, 2)
        for system in (baseline, oracle):
            self.assertEqual(len(system.nodes), 1)
            self.assertIs(system.nodes[0], system.node)
            self.assertEqual(system.node.node_id, 0)
            self.assertEqual(
                system.route_policy, SINGLE_NODE_ROUTE_POLICY)
        self.assertEqual(baseline.policy, "ssd_direct")
        self.assertEqual(
            baseline.node.lifecycle.ssd_ledger.capacity_bytes,
            2_048 * self.block_aggregate,
        )

    def test_baseline_and_oracle_freeze_identical_single_node_projection(self):
        scheduled = self.paired_schedule()
        original = tuple(asdict(item) for item in scheduled)
        baseline = self.make_baseline()
        oracle = self.make_oracle()

        baseline.load(scheduled)
        oracle.load(scheduled)

        self.assertEqual(baseline.call_specs, oracle.call_specs)
        self.assertTrue(all(
            spec.node_id == 0 for spec in baseline.call_specs))
        self.assertEqual(
            [spec.source_index for spec in baseline.call_specs],
            [10, 20, 20],
        )
        self.assertEqual(
            validate_system_call_projection(
                scheduled, baseline.call_specs),
            validate_system_call_projection(
                scheduled, oracle.call_specs),
        )
        self.assertEqual(
            tuple(asdict(item) for item in scheduled), original)

    def test_successor_release_is_completion_plus_tool_gap_in_both(self):
        scheduled = self.paired_schedule()
        for system in (self.make_baseline(), self.make_oracle()):
            with self.subTest(mode=system.mode):
                completed = system.run(scheduled)
                first = system._runtime_calls[
                    system.request_id_for(
                        "later-source::call-0")]
                resume = system._runtime_calls[
                    system.request_id_for(
                        "later-source::call-1")]
                self.assertEqual(
                    resume.release_ns,
                    first.user_completion_ns + 123_456,
                )
                self.assertEqual(len(completed), 3)
                validate_causal_release_contract(
                    scheduled, completed)

    def test_reports_exact_call_and_session_full_drain(self):
        scheduled = self.paired_schedule()
        for system in (
                self.make_baseline(validate_every_event=False),
                self.make_oracle(validate_every_event=False)):
            with self.subTest(mode=system.mode):
                system.run(scheduled)
                report = system.report()
                self.assertTrue(report["finished"])
                self.assertEqual(report["gpu_server_count"], 1)
                self.assertEqual(
                    report["call_full_drain"]["identity_count"], 3)
                self.assertEqual(
                    report["call_full_drain"][
                        "expected_set_sha256"],
                    report["call_full_drain"][
                        "completion_set_sha256"],
                )
                self.assertEqual(
                    report["session_full_drain"]["identity_count"], 2)
                self.assertEqual(
                    report["session_full_drain"][
                        "expected_set_sha256"],
                    report["session_full_drain"][
                        "completion_set_sha256"],
                )
                self.assertEqual(len(report["nodes"]), 1)

    def test_cotimed_arrivals_share_one_gpu_batch_boundary(self):
        scheduled = tuple(
            self.session(
                index, arrival_ns=5_000,
                calls=((32 + index, 1, 0, 0),))
            for index in range(4)
        )
        for system in (self.make_baseline(), self.make_oracle()):
            with self.subTest(mode=system.mode):
                system.run(scheduled)
                first_p_batch = next(
                    batch for batch in system.node.pool.batch_history
                    if batch.stage == "p"
                )
                self.assertEqual(
                    [item.request_id for item in first_p_batch.items],
                    [spec.request_id for spec in system.call_specs],
                )

    def test_ssd_direct_baseline_uses_only_its_local_ssd_under_pressure(self):
        scheduled = tuple(
            self.session(
                index,
                arrival_ns=0,
                calls=(
                    (64, 2, 0, 1_000_000_000),
                    (72, 2, 65, 0),
                ),
            )
            for index in range(2)
        )
        system = self.make_baseline(
            policy="ssd_direct",
            d_blocks=5,
            validate_every_event=False,
        )
        system.run(scheduled)

        lifecycle_metrics = system.node.lifecycle.metrics
        report = system.report()
        self.assertGreater(
            lifecycle_metrics.d_to_ssd_started, 0)
        self.assertGreater(
            lifecycle_metrics.demotions_committed, 0)
        self.assertEqual(report["policy"], "ssd_direct")
        self.assertEqual(
            report["local_ssd"]["device_count"],
            self.hardware.ssd_device_count,
        )
        self.assertTrue(all(
            resource.startswith("gpu-node-0-")
            for resource in system.node.calendar.available_ns
        ))

    def test_invalid_capacity_schedule_is_rejected_before_load_mutation(self):
        system = self.make_baseline(p_blocks=1)
        scheduled = (
            self.session(0, calls=((16, 1, 0, 0),)),
            self.session(1, calls=((17, 1, 0, 0),)),
        )

        with self.assertRaisesRegex(ValueError, "P-HBM"):
            system.load(scheduled)

        self.assertFalse(system._loaded)
        self.assertEqual(system.call_specs, ())
        self.assertEqual(system._spec_by_request, {})
        self.assertEqual(system._release_heap, [])
        self.assertFalse(system.node.calls)

    def test_missing_first_release_is_reported_as_deadlock(self):
        system = self.make_oracle()
        system.load((self.session(0),))
        system._release_heap.clear()

        with self.assertRaisesRegex(
                SingleP4D4DeadlockError, "no future event"):
            system.run()


if __name__ == "__main__":
    unittest.main()

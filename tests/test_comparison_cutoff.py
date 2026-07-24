from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

from serving.core.gpu_hbf_hybrid import GPUHBFHybridSystem
from serving.core.gpu_pd_dual_oracle import DualStrictInfiniteHBMOracle
from serving.core.gpu_pd_dual_tiered import DualFiniteHBMTieredBaseline
from serving.core.gpu_pd_latency import load_p4d4_gpu_config
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


def make_schedule(
        source_index, calls, *,
        arrival_ns=0, offer_index=None):
    session_id = f"session-{source_index}"
    call_specs = tuple(
        CallSpec(
            session_id=session_id,
            source_index=source_index,
            call_index=call_index,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_duration_ns=tool_duration_ns,
            cached_prefix_tokens=cached_prefix_tokens,
            fresh_input_tokens=(
                input_tokens - cached_prefix_tokens),
            lineage_status=None,
            inter_turn_gap_type=None,
        )
        for call_index, (
            input_tokens,
            output_tokens,
            cached_prefix_tokens,
            tool_duration_ns,
        ) in enumerate(calls)
    )
    return ScheduledSession(
        offer_index=(
            source_index if offer_index is None else offer_index),
        session=SessionSpec(
            source_index=source_index,
            session_id=session_id,
            source_arrival_time_ns=arrival_ns,
            source_session_identity_sha256=None,
            calls=call_specs,
        ),
        arrival_time_ns=arrival_ns,
        unit_interarrival=0.0,
        unit_arrival_time=0.0,
    )


class ComparisonCutoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = load_p4d4_gpu_config(GPU_CONFIG)
        cls.block_per_rank = (
            cls.hardware.kv_capacity_bytes_per_rank(16))
        cls.block_aggregate = (
            cls.block_per_rank * cls.hardware.tp_size)

    def make_oracle(self):
        return DualStrictInfiniteHBMOracle(
            repo_root=REPO_ROOT,
            hardware=self.hardware,
            max_num_batched_tokens=256,
            max_num_seqs=16,
            max_prefill_chunk_tokens=128,
        )

    def make_tiered(self):
        return DualFiniteHBMTieredBaseline(
            repo_root=REPO_ROOT,
            hardware=self.hardware,
            policy="cpu_ssd",
            p_capacity_bytes_per_rank=(
                64 * self.block_per_rank),
            d_capacity_bytes_per_rank=(
                64 * self.block_per_rank),
            cpu_capacity_bytes=(
                64 * self.block_aggregate),
            ssd_capacity_bytes=(
                512 * self.block_aggregate),
            max_num_batched_tokens=256,
            max_num_seqs=16,
            max_prefill_chunk_tokens=128,
        )

    def make_hybrid(self):
        return GPUHBFHybridSystem(
            repo_root=REPO_ROOT,
            gpu_hardware=self.hardware,
            hbf_layout="tp4",
            max_num_batched_tokens=256,
            max_num_seqs=16,
            max_prefill_chunk_tokens=128,
        )

    @property
    def factories(self):
        return (
            ("oracle", self.make_oracle),
            ("tiered", self.make_tiered),
            ("hybrid", self.make_hybrid),
        )

    @staticmethod
    def chain_schedule():
        # output_tokens=1 lets the user completion precede the lineage
        # handoff.  A zero tool gap releases call 1 at that exact timestamp.
        return make_schedule(
            0,
            (
                (64, 1, 0, 0),
                (64, 1, 64, 0),
            ),
        )

    @staticmethod
    def by_key(completed):
        return {request.key: request for request in completed}

    def first_completion_ns(self, factory):
        system = factory()
        completed = system.run((self.chain_schedule(),))
        return completed[0].completion_ns

    def test_exact_cutoff_partition_resume_and_full_run_equivalence(self):
        for name, factory in self.factories:
            with self.subTest(system=name):
                cutoff_ns = self.first_completion_ns(factory)
                schedules = (
                    self.chain_schedule(),
                    make_schedule(
                        1,
                        ((32, 2, 0, 0),),
                        arrival_ns=cutoff_ns + 1_000,
                    ),
                )

                reference = factory()
                reference_completed = reference.run(schedules)
                reference_by_key = self.by_key(reference_completed)

                partial = factory()
                audit = partial.run_until(
                    cutoff_ns,
                    scheduled_sessions=schedules,
                )

                self.assertFalse(partial._finished)
                self.assertFalse(audit.system_finished)
                self.assertEqual(audit.current_ns, cutoff_ns)
                self.assertEqual(
                    audit.last_processed_event_ns, cutoff_ns)
                self.assertEqual(
                    audit.scheduled_request_ids, (0, 1, 2))
                self.assertEqual(
                    audit.unreleased_request_ids, (2,))
                self.assertEqual(
                    audit.released_live_request_ids, (1,))
                self.assertEqual(
                    audit.user_completed_request_ids, (0,))
                self.assertEqual(
                    audit.internal_work_request_ids, (0, 1))
                self.assertEqual(
                    audit.internal_complete_request_ids, ())
                self.assertEqual(
                    audit.user_completed_internal_work_request_ids,
                    (0,),
                )
                self.assertIsNotNone(audit.next_event_ns)
                self.assertGreater(audit.next_event_ns, cutoff_ns)
                self.assertEqual(
                    partial._runtime_calls[1].release_ns,
                    cutoff_ns,
                )
                self.assertTrue(all(
                    request.completion_ns <= cutoff_ns
                    for request in partial.completed_requests
                ))

                for request in partial.completed_requests:
                    self.assertEqual(
                        request,
                        reference_by_key[request.key],
                    )

                resumed_completed = partial.run()
                self.assertTrue(partial._finished)
                self.assertEqual(
                    self.by_key(resumed_completed),
                    reference_by_key,
                )
                partial.assert_invariants()

    def test_exclusive_cutoff_leaves_tie_then_inclusive_resume_takes_it(self):
        for name, factory in self.factories:
            with self.subTest(system=name):
                cutoff_ns = self.first_completion_ns(factory)
                system = factory()
                before = system.run_until(
                    cutoff_ns,
                    inclusive=False,
                    scheduled_sessions=(self.chain_schedule(),),
                )
                self.assertEqual(
                    before.user_completed_request_ids, ())
                self.assertEqual(before.next_event_ns, cutoff_ns)
                self.assertLess(
                    before.last_processed_event_ns, cutoff_ns)
                self.assertFalse(system._finished)

                at = system.run_until(cutoff_ns)
                self.assertEqual(
                    at.user_completed_request_ids, (0,))
                self.assertEqual(
                    at.released_live_request_ids, (1,))
                self.assertEqual(
                    at.user_completed_internal_work_request_ids,
                    (0,),
                )
                self.assertEqual(at.current_ns, cutoff_ns)
                self.assertFalse(system._finished)

    def test_cutoff_past_natural_drain_still_does_not_finish_system(self):
        schedule = make_schedule(
            0,
            ((16, 1, 0, 0),),
        )
        for name, factory in self.factories:
            with self.subTest(system=name):
                reference = factory()
                completion_ns = reference.run(
                    (schedule,))[0].completion_ns

                system = factory()
                audit = system.run_until(
                    completion_ns + 1_000_000_000,
                    scheduled_sessions=(schedule,),
                )
                self.assertFalse(system._finished)
                self.assertFalse(audit.system_finished)
                self.assertEqual(audit.next_event_ns, None)
                self.assertEqual(
                    audit.user_completed_request_ids, (0,))
                self.assertEqual(
                    audit.internal_complete_request_ids, (0,))
                self.assertEqual(
                    audit.internal_work_request_ids, ())

                before = tuple(system.completed_requests)
                self.assertEqual(system.run(), list(before))
                self.assertTrue(system._finished)

    def test_audit_is_frozen_and_cutoff_arguments_fail_closed(self):
        system = self.make_oracle()
        with self.assertRaises(RuntimeError):
            system.run_until(0)
        with self.assertRaises(ValueError):
            system.run_until(
                True,
                scheduled_sessions=(self.chain_schedule(),),
            )

        system = self.make_oracle()
        audit = system.run_until(
            0,
            scheduled_sessions=(self.chain_schedule(),),
        )
        with self.assertRaises(FrozenInstanceError):
            audit.current_ns = 1
        with self.assertRaises(ValueError):
            system.run_until(-1)
        with self.assertRaises(ValueError):
            system.run_until(0, inclusive=1)

        cutoff_ns = self.first_completion_ns(self.make_oracle)
        system = self.make_oracle()
        system.run_until(
            cutoff_ns,
            scheduled_sessions=(self.chain_schedule(),),
        )
        with self.assertRaises(ValueError):
            system.run_until(cutoff_ns, inclusive=False)
        with self.assertRaises(ValueError):
            system.run_until(cutoff_ns - 1)


if __name__ == "__main__":
    unittest.main()

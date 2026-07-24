from pathlib import Path
import unittest

from serving.core.gpu_hbf_hybrid import (
    GPUHBFHybridNode,
    GPUHBFHybridSystem,
    HybridCall,
    HybridCallState,
    HybridDeadlockError,
    HybridExecution,
)
from serving.core.gpu_pd_latency import P4D4GPUHardware
from serving.core.hbf_comparison_workload import (
    CallSpec,
    ScheduledSession,
    SessionSpec,
)
from serving.core.hbf_full_model_latency import HBFServerHardware
from serving.core.hbf_full_model_lifecycle import PlacementState


REPO_ROOT = Path(__file__).resolve().parents[1]


class GPUHBFHybridExternalTests(unittest.TestCase):
    def node(
            self, *, backend="external_astra",
            hbf_server_id=None,
            hbf_astra_chunk_bytes=64 * 1024 ** 2):
        return GPUHBFHybridNode(
            repo_root=REPO_ROOT,
            gpu_hardware=P4D4GPUHardware(),
            hbf_hardware=HBFServerHardware(),
            hbf_layout="tp4",
            max_num_batched_tokens=256,
            max_num_seqs=16,
            max_prefill_chunk_tokens=128,
            hbf_execution_backend=backend,
            hbf_server_id=hbf_server_id,
            hbf_astra_chunk_bytes=hbf_astra_chunk_bytes,
        )

    @staticmethod
    def call(
            request_id, call_index, release_ns, *,
            input_tokens, output_tokens=1,
            prefix_reuse_tokens=0, has_successor=False):
        return HybridCall(
            request_id=request_id,
            session_id="session",
            call_index=call_index,
            release_ns=release_ns,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prefix_reuse_tokens=prefix_reuse_tokens,
            has_successor=has_successor,
        )

    @staticmethod
    def complete_at_critical_path(
            node, dispatch, *, defer_schedule=True):
        completion_ns = (
            dispatch.arrival_ns
            + dispatch.projection
            .dependency_critical_path_ns()
        )
        node.complete_external_dispatch(
            dispatch.job_id,
            dispatch.arrival_ns,
            completion_ns,
            dispatch.stage_count,
            defer_schedule=defer_schedule,
        )
        return completion_ns

    @staticmethod
    def drain_python_events(node):
        while node.next_event_ns() is not None:
            node.advance(node.next_event_ns())

    def start_gpu_first_turn(self, node):
        first = self.call(
            0,
            0,
            0,
            input_tokens=16,
            output_tokens=1,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        self.drain_python_events(node)
        self.assertEqual(
            first.state, HybridCallState.INTERNAL_COMPLETE)
        self.assertTrue(node.has_pending_external())
        return first

    def test_migration_foreground_and_append_share_one_dispatch_api(self):
        node = self.node()
        self.start_gpu_first_turn(node)

        migration, = node.drain_external_dispatches()
        self.assertEqual(migration.owner, "lifecycle")
        self.assertTrue(
            migration.job_id.startswith("hbf-migration.s0."))
        self.assertEqual(
            migration.controller_arguments()[0],
            migration.job_id,
        )
        migration_completion = self.complete_at_critical_path(
            node, migration)
        placement = node.hbf_lifecycle.sessions["session"]
        self.assertEqual(placement.state, PlacementState.HBF_READY)
        self.assertEqual(node.gpu_hbm.d_bytes("session"), 0)
        self.assertEqual(node.metrics.migration_hbm_releases, 1)

        resume = self.call(
            1,
            1,
            migration_completion,
            input_tokens=17,
            output_tokens=1,
            prefix_reuse_tokens=16,
            has_successor=True,
        )
        node.submit(resume, now_ns=migration_completion)
        self.assertEqual(
            resume.execution, HybridExecution.HBF_READY)
        foreground, = node.drain_external_dispatches()
        self.assertEqual(foreground.owner, "pool")
        self.assertTrue(
            foreground.job_id.startswith("hbf-model.s0."))
        foreground_completion = self.complete_at_critical_path(
            node, foreground)
        self.assertEqual(
            resume.state, HybridCallState.INTERNAL_COMPLETE)
        self.assertEqual(
            resume.user_completion_ns, foreground_completion)

        append, = node.drain_external_dispatches()
        self.assertEqual(append.owner, "lifecycle")
        self.assertTrue(
            append.job_id.startswith("hbf-append.s0."))
        self.complete_at_critical_path(node, append)
        self.assertFalse(node.has_pending_external())
        self.assertEqual(placement.committed_hbf_tokens, 17)
        self.assertEqual(placement.lpddr_tokens, 0)
        ids = {
            migration.job_id,
            foreground.job_id,
            append.job_id,
        }
        self.assertEqual(len(ids), 3)
        report = node.report()
        self.assertEqual(
            report["hbf_execution_backend"], "external_astra")
        self.assertEqual(
            report["external_hbf_completed_job_count"], 3)
        self.assertEqual(
            report["rdma_migration_summary"]["accounting_source"],
            "astra_causal_projection",
        )

    def test_deferred_pool_callback_does_not_launch_next_batch(self):
        node = self.node()
        self.start_gpu_first_turn(node)
        migration, = node.drain_external_dispatches()
        ready_ns = self.complete_at_critical_path(node, migration)

        resume = self.call(
            1,
            1,
            ready_ns,
            input_tokens=16,
            output_tokens=2,
            prefix_reuse_tokens=16,
            has_successor=False,
        )
        node.submit(resume, now_ns=ready_ns)
        first_batch, = node.drain_external_dispatches()
        self.assertEqual(first_batch.owner, "pool")
        first_completion = self.complete_at_critical_path(
            node, first_batch, defer_schedule=True)
        self.assertEqual(
            node.drain_external_dispatches(), ())
        self.assertEqual(resume.state, HybridCallState.HBF_EXECUTING)

        node.flush_scheduling(first_completion)
        second_batch, = node.drain_external_dispatches()
        self.assertEqual(second_batch.owner, "pool")
        self.complete_at_critical_path(
            node, second_batch, defer_schedule=True)
        self.assertEqual(
            resume.state, HybridCallState.INTERNAL_COMPLETE)
        self.assertFalse(node.has_pending_external())

    def test_unknown_duplicate_and_python_timing_are_rejected(self):
        node = self.node()
        self.start_gpu_first_turn(node)
        self.assertIsNone(node.hbf_calendar)
        self.assertEqual(node.hbf_lifecycle._completion_heap, [])
        self.assertEqual(node.hbf_pool._completion_heap, [])
        self.assertIsNone(node.next_event_ns())
        with self.assertRaisesRegex(
                HybridDeadlockError, "external ASTRA HBF"):
            node.run_until_idle()

        dispatch, = node.drain_external_dispatches()
        completion = (
            dispatch.arrival_ns
            + dispatch.projection
            .dependency_critical_path_ns()
        )
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            node.complete_external_dispatch(
                "hbf-unknown.s0.r0.j0",
                dispatch.arrival_ns,
                completion,
                dispatch.stage_count,
            )
        node.complete_external_dispatch(
            dispatch.job_id,
            dispatch.arrival_ns,
            completion,
            dispatch.stage_count,
            defer_schedule=True,
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            node.complete_external_dispatch(
                dispatch.job_id,
                dispatch.arrival_ns,
                completion,
                dispatch.stage_count,
            )

    def test_server_identity_and_chunk_size_reach_both_components(self):
        node = self.node(
            hbf_server_id=9,
            hbf_astra_chunk_bytes=131_072,
        )
        self.assertEqual(node.hbf_server_id, 9)
        self.assertEqual(node.hbf_lifecycle.server_id, 9)
        self.assertEqual(node.hbf_pool.server_id, 9)
        self.assertEqual(
            node.hbf_lifecycle.astra_chunk_bytes, 131_072)
        self.start_gpu_first_turn(node)
        migration, = node.drain_external_dispatches()
        self.assertTrue(
            migration.job_id.startswith("hbf-migration.s9."))
        self.assertEqual(
            migration.projection.chunk_bytes, 131_072)
        report = node.report()
        self.assertEqual(report["hbf_server_id"], 9)
        self.assertEqual(
            report["hbf_astra_chunk_bytes"], 131_072)

    def test_legacy_backend_keeps_shared_hbf_calendar(self):
        node = self.node(backend="analytical_calendar")
        self.assertIsNotNone(node.hbf_calendar)
        self.assertIs(
            node.hbf_lifecycle.calendar, node.hbf_calendar)
        self.assertIs(node.hbf_pool.calendar, node.hbf_calendar)
        self.assertEqual(
            node.report()["hbf_completion_time_source"],
            "python_analytical_calendar",
        )
        with self.assertRaisesRegex(
                RuntimeError, "external_astra"):
            node.drain_external_dispatches()

    def test_backend_validation(self):
        with self.assertRaisesRegex(
                ValueError, "hbf_execution_backend"):
            self.node(backend="not-a-backend")
        with self.assertRaisesRegex(ValueError, "hbf_server_id"):
            self.node(hbf_server_id=-1)
        with self.assertRaisesRegex(
                ValueError, "hbf_astra_chunk_bytes"):
            self.node(hbf_astra_chunk_bytes=0)

    def test_system_run_requires_an_external_astra_driver(self):
        system = GPUHBFHybridSystem(
            repo_root=REPO_ROOT,
            hbf_execution_backend="external_astra",
            hbf_server_id=11,
            hbf_astra_chunk_bytes=262_144,
            max_num_batched_tokens=256,
            max_num_seqs=16,
            max_prefill_chunk_tokens=128,
        )
        session_id = "agent"
        calls = (
            CallSpec(
                session_id=session_id,
                source_index=0,
                call_index=0,
                input_tokens=16,
                output_tokens=1,
                tool_duration_ns=0,
                cached_prefix_tokens=0,
                fresh_input_tokens=16,
                lineage_status=None,
                inter_turn_gap_type=None,
            ),
            CallSpec(
                session_id=session_id,
                source_index=0,
                call_index=1,
                input_tokens=17,
                output_tokens=1,
                tool_duration_ns=0,
                cached_prefix_tokens=16,
                fresh_input_tokens=1,
                lineage_status=None,
                inter_turn_gap_type=None,
            ),
        )
        scheduled = ScheduledSession(
            offer_index=0,
            session=SessionSpec(
                source_index=0,
                session_id=session_id,
                source_arrival_time_ns=0,
                source_session_identity_sha256=None,
                calls=calls,
            ),
            arrival_time_ns=0,
            unit_interarrival=0.0,
            unit_arrival_time=0.0,
        )
        with self.assertRaisesRegex(
                HybridDeadlockError,
                "drive node.drain_external_dispatches"):
            system.run((scheduled,))
        self.assertTrue(system.node.has_pending_external())
        self.assertEqual(system.node.hbf_server_id, 11)
        self.assertEqual(
            system.node.hbf_astra_chunk_bytes, 262_144)


if __name__ == "__main__":
    unittest.main()

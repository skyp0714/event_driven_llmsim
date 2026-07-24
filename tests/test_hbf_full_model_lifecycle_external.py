import dataclasses
import unittest

from serving.core.hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
)
from serving.core.hbf_full_model_lifecycle import (
    FullModelHBFLifecycle,
    PlacementState,
    ResourceCalendar,
    ResumeExecution,
)


class FullModelHBFLifecycleExternalTests(unittest.TestCase):
    def manager(self, **overrides):
        values = {
            "hardware": HBFServerHardware(),
            "layout": HBFParallelLayout.for_key("tp4"),
            "kv_bytes_per_token": 1,
            "execution_backend": "external_astra",
            "server_id": 7,
            "astra_chunk_bytes": 11,
        }
        values.update(overrides)
        return FullModelHBFLifecycle(**values)

    @staticmethod
    def completion(dispatch, extra_ns=0):
        return (
            dispatch.arrival_ns
            + dispatch.projection
            .dependency_critical_path_ns()
            + extra_ns
        )

    def test_default_backend_remains_analytical(self):
        manager = FullModelHBFLifecycle(
            hardware=HBFServerHardware(),
            layout=HBFParallelLayout.for_key("tp4"),
            kv_bytes_per_token=1,
        )
        manager.register_session("legacy")
        job = manager.complete_gpu_turn(
            "legacy", now_ns=3, total_tokens=100,
            has_successor=True)
        self.assertGreater(job.completion_ns, job.start_ns)
        self.assertIsNotNone(manager.next_completion_ns())
        manager.advance(job.completion_ns)
        self.assertEqual(
            manager.sessions["legacy"].state,
            PlacementState.HBF_READY,
        )
        self.assertEqual(
            manager.report()["completion_time_source"],
            "python_analytical_calendar",
        )

    def test_external_migration_and_append_are_callback_driven(self):
        manager = self.manager()
        record = manager.register_session("s")
        migration = manager.complete_gpu_turn(
            "s", now_ns=5, total_tokens=53,
            has_successor=True)
        self.assertEqual(migration.start_ns, 5)
        self.assertEqual(migration.completion_ns, 5)
        self.assertEqual(record.state, PlacementState.MIGRATING)
        self.assertIsNone(manager.next_completion_ns())
        self.assertTrue(manager.has_pending_external())

        migration_dispatch, = manager.drain_external_dispatches()
        self.assertIs(migration_dispatch.job, migration)
        self.assertEqual(migration_dispatch.projection.kind, "migration")
        self.assertEqual(
            migration_dispatch.projection.placement.server_id, 7)
        self.assertGreater(migration_dispatch.stage_count, 0)
        self.assertEqual(
            migration_dispatch.controller_arguments()[0],
            migration_dispatch.job_id,
        )
        migration_completion = self.completion(migration_dispatch)
        manager.advance(migration_completion)
        self.assertEqual(record.state, PlacementState.MIGRATING)
        completed_migration = manager.complete_external_dispatch(
            migration_dispatch.job_id,
            migration_dispatch.arrival_ns,
            migration_completion,
            migration_dispatch.stage_count,
        )
        self.assertEqual(
            completed_migration.completion_ns, migration_completion)
        self.assertEqual(record.state, PlacementState.HBF_READY)
        self.assertEqual(record.committed_hbf_tokens, 53)

        route = manager.route_resume(
            "s", now_ns=migration_completion, request_id=1)
        self.assertEqual(route.execution, ResumeExecution.HBF)
        append = manager.complete_hbf_turn(
            "s",
            now_ns=migration_completion + 1,
            total_tokens=66,
            has_successor=True,
        )
        append_dispatch, = manager.drain_external_dispatches()
        self.assertIs(append_dispatch.job, append)
        self.assertEqual(append_dispatch.projection.kind, "append")
        roles = {
            stage.role for stage in append_dispatch.projection.stages
        }
        self.assertEqual(roles, {"lpddr_read", "hbf_write"})
        append_completion = self.completion(append_dispatch)
        completed_append = manager.complete_external_dispatch(
            append_dispatch.job_id,
            append_dispatch.arrival_ns,
            append_completion,
            append_dispatch.stage_count,
        )
        self.assertEqual(completed_append.completion_ns, append_completion)
        self.assertEqual(record.committed_hbf_tokens, 66)
        self.assertEqual(record.lpddr_tokens, 0)
        self.assertFalse(manager.has_pending_external())
        report = manager.report()
        self.assertEqual(report["metrics"]["astra_completed_jobs"], 2)
        self.assertEqual(report["external_completed_dispatch_count"], 2)
        self.assertEqual(
            report["astra_projection_fidelity"],
            "causal-chunked-v1",
        )
        self.assertIn(
            "astra_signed_interference_delta_ns",
            report["astra_timing_semantics"],
        )

    def test_tp8_dependency_bound_allows_negative_interference_delta(self):
        manager = self.manager(
            layout=HBFParallelLayout.for_key("tp8"),
            kv_bytes_per_token=1024,
            astra_chunk_bytes=16 * 1024,
        )
        manager.register_session("s")
        manager.complete_gpu_turn(
            "s", now_ns=5, total_tokens=16,
            has_successor=True,
        )
        dispatch, = manager.drain_external_dispatches()
        dependency_ns = (
            dispatch.projection.dependency_critical_path_ns())
        solo_resource_ns = (
            dispatch.projection
            .solo_resource_serialized_completion_ns()
        )
        self.assertGreater(solo_resource_ns, dependency_ns)

        actual_elapsed_ns = dependency_ns
        signed_interference_delta_ns = (
            actual_elapsed_ns - solo_resource_ns)
        manager.complete_external_dispatch(
            dispatch.job_id,
            dispatch.arrival_ns,
            dispatch.arrival_ns + actual_elapsed_ns,
            dispatch.stage_count,
        )
        metrics = manager.metrics
        self.assertEqual(metrics.astra_completed_jobs, 1)
        self.assertEqual(
            metrics.astra_dependency_critical_path_ns,
            dependency_ns,
        )
        self.assertEqual(
            metrics.astra_solo_resource_serialized_completion_ns,
            solo_resource_ns,
        )
        self.assertEqual(
            metrics.astra_actual_resource_serialized_completion_ns,
            actual_elapsed_ns,
        )
        self.assertEqual(
            metrics.astra_internal_resource_serialization_wait_ns,
            solo_resource_ns - dependency_ns,
        )
        self.assertEqual(
            metrics.astra_signed_interference_delta_ns,
            signed_interference_delta_ns,
        )
        self.assertEqual(
            metrics.astra_resource_delay_ns,
            actual_elapsed_ns - dependency_ns,
        )
        self.assertEqual(
            metrics.astra_resource_delay_ns,
            (
                metrics.astra_internal_resource_serialization_wait_ns
                + metrics.astra_signed_interference_delta_ns
            ),
        )
        self.assertEqual(
            metrics.astra_completion_elapsed_ns,
            metrics.astra_actual_resource_serialized_completion_ns,
        )
        manager.assert_invariants()
        metrics.astra_signed_interference_delta_ns = -1.0
        with self.assertRaisesRegex(
                AssertionError, "finite integer"):
            manager.assert_invariants()

    def test_resume_during_external_migration_stales_publication(self):
        manager = self.manager()
        record = manager.register_session("s")
        job = manager.complete_gpu_turn(
            "s", now_ns=100, total_tokens=37,
            has_successor=True)
        dispatch, = manager.drain_external_dispatches()
        route = manager.route_resume(
            "s", now_ns=101, request_id=9)
        self.assertEqual(route.execution, ResumeExecution.GPU)
        self.assertTrue(route.migration_inflight)
        retained = record.gpu_retained_bytes

        manager.complete_external_dispatch(
            dispatch.job_id,
            dispatch.arrival_ns,
            self.completion(dispatch),
            dispatch.stage_count,
        )
        self.assertEqual(record.state, PlacementState.GPU_ACTIVE)
        self.assertEqual(record.gpu_retained_bytes, retained)
        self.assertEqual(record.committed_hbf_tokens, 0)
        self.assertEqual(manager.metrics.migrations_stale, 1)
        self.assertEqual(
            manager.report()["group_reserved_per_card_bytes"],
            {0: 0, 1: 0},
        )
        self.assertNotIn(job.job_id, manager._jobs)

    def test_callback_before_equal_timestamp_resume_wins_tie(self):
        manager = self.manager()
        record = manager.register_session("s")
        manager.complete_gpu_turn(
            "s", now_ns=0, total_tokens=29,
            has_successor=True)
        dispatch, = manager.drain_external_dispatches()
        completion = self.completion(dispatch)
        manager.complete_external_dispatch(
            dispatch.job_id,
            dispatch.arrival_ns,
            completion,
            dispatch.stage_count,
        )
        route = manager.route_resume(
            "s", now_ns=completion, request_id=4)
        self.assertEqual(route.execution, ResumeExecution.HBF)
        self.assertEqual(record.state, PlacementState.HBF_ACTIVE)

    def test_strict_external_callback_rejects_bad_identity_and_metadata(self):
        manager = self.manager()
        manager.register_session("s")
        job = manager.complete_gpu_turn(
            "s", now_ns=10, total_tokens=31,
            has_successor=True)
        pending_id = next(iter(manager._external_pending))
        pending = manager._external_pending[pending_id]
        minimum = self.completion(pending)
        with self.assertRaisesRegex(RuntimeError, "was not drained"):
            manager.complete_external_dispatch(
                pending_id, pending.arrival_ns, minimum,
                pending.stage_count)

        dispatch, = manager.drain_external_dispatches()
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            manager.complete_external_dispatch(
                "hbf-migration.unknown", dispatch.arrival_ns,
                minimum, dispatch.stage_count)
        with self.assertRaisesRegex(RuntimeError, "arrival mismatch"):
            manager.complete_external_dispatch(
                dispatch.job_id, dispatch.arrival_ns + 1,
                minimum, dispatch.stage_count)
        with self.assertRaisesRegex(RuntimeError, "stage-count"):
            manager.complete_external_dispatch(
                dispatch.job_id, dispatch.arrival_ns,
                minimum, dispatch.stage_count + 1)
        with self.assertRaisesRegex(RuntimeError, "critical path"):
            manager.complete_external_dispatch(
                dispatch.job_id, dispatch.arrival_ns,
                minimum - 1, dispatch.stage_count)

        manager._jobs[job.job_id] = dataclasses.replace(job)
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            manager.complete_external_dispatch(
                dispatch.job_id, dispatch.arrival_ns,
                minimum, dispatch.stage_count)
        manager._jobs[job.job_id] = job
        manager.complete_external_dispatch(
            dispatch.job_id, dispatch.arrival_ns,
            minimum, dispatch.stage_count)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            manager.complete_external_dispatch(
                dispatch.job_id, dispatch.arrival_ns,
                minimum, dispatch.stage_count)

    def test_external_mode_never_uses_python_timing_or_calendar(self):
        with self.assertRaisesRegex(ValueError, "must be omitted"):
            self.manager(resource_calendar=ResourceCalendar())
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.manager(astra_chunk_bytes=0)
        manager = self.manager()
        manager.register_session("s")
        manager.complete_gpu_turn(
            "s", now_ns=0, total_tokens=23,
            has_successor=True)
        self.assertEqual(manager._completion_heap, [])
        self.assertEqual(manager.calendar.reservations, [])
        self.assertEqual(manager.calendar.busy_ns, {})
        manager.advance(10**9)
        self.assertEqual(
            manager.sessions["s"].state, PlacementState.MIGRATING)
        with self.assertRaisesRegex(
                RuntimeError, "external ASTRA lifecycle completions"):
            manager.run_until_idle()


if __name__ == "__main__":
    unittest.main()

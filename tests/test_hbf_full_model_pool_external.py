from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

from serving.core.controller import Controller
from serving.core.hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
)
from serving.core.hbf_full_model_pool import (
    FullModelHBFServingPool,
    HBFExternalDispatch,
    HBFRequestState,
    HBFServingRequest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class ExternalAstraFullModelPoolTests(unittest.TestCase):
    def make_pool(self, layout="tp4", **kwargs):
        return FullModelHBFServingPool(
            repo_root=REPO_ROOT,
            hardware=HBFServerHardware(),
            layout=HBFParallelLayout.for_key(layout),
            execution_backend="external_astra",
            max_num_batched_tokens=256,
            max_num_seqs=16,
            max_prefill_chunk_tokens=64,
            **kwargs,
        )

    @staticmethod
    def request(
            request_id=1, *, group_id=0, output_tokens=1,
            arrival_ns=0):
        return HBFServingRequest(
            request_id=request_id,
            session_id=f"session-{request_id}",
            arrival_ns=arrival_ns,
            input_tokens=100,
            output_tokens=output_tokens,
            hbf_prefix_tokens=100,
            lpddr_prefix_tokens=0,
            group_id=group_id,
        )

    @staticmethod
    def complete_at_minimum(pool, dispatch, extra_ns=0):
        completion_ns = (
            dispatch.arrival_ns
            + dispatch.projection
            .dependency_critical_path_ns()
            + extra_ns
        )
        return pool.complete_external_dispatch(
            dispatch.job_id,
            dispatch.arrival_ns,
            completion_ns,
            dispatch.stage_count,
        )

    def test_dispatch_is_immutable_controller_compatible_and_not_modeled(self):
        pool = self.make_pool(server_id=7)
        request = self.request()
        pool.submit(request, now_ns=0)

        self.assertIsNone(pool.next_completion_ns())
        self.assertIsNone(pool.next_event_ns())
        self.assertTrue(pool.has_pending())
        self.assertTrue(pool.has_pending_external_dispatches())
        self.assertEqual(pool.calendar.reservations, [])
        self.assertEqual(len(pool.batch_history), 1)
        self.assertIsNone(pool.batch_history[0].completion_ns)
        with self.assertRaisesRegex(RuntimeError, "completions are pending"):
            pool.run_until_idle()

        dispatches = pool.drain_external_dispatches()
        self.assertEqual(len(dispatches), 1)
        dispatch = dispatches[0]
        self.assertIsInstance(dispatch, HBFExternalDispatch)
        self.assertEqual(dispatch.batch, pool.workers[0].inflight)
        self.assertIn("hbf-model.s7.r0", dispatch.job_id)
        with self.assertRaises(FrozenInstanceError):
            dispatch.arrival_ns = 1

        job_id, arrival_ns, stages = dispatch.controller_arguments()
        self.assertEqual(job_id, dispatch.job_id)
        self.assertEqual(arrival_ns, 0)
        self.assertEqual(len(stages), dispatch.stage_count)
        command = Controller.hbf_background_command(
            job_id, arrival_ns, stages)
        self.assertTrue(command.startswith(
            f"hbf-background\t{dispatch.job_id}\t0\t"))
        self.assertEqual(pool.drain_external_dispatches(), ())
        pool.assert_invariants()

    def test_actual_callback_drives_ttft_tpot_and_follow_on_dispatch(self):
        pool = self.make_pool()
        request = self.request(output_tokens=2)
        pool.submit(request, now_ns=0)
        first = pool.drain_external_dispatches()[0]
        first_batch = self.complete_at_minimum(
            pool, first, extra_ns=123)

        self.assertEqual(
            first_batch.completion_ns,
            first.arrival_ns
            + first.projection
            .dependency_critical_path_ns()
            + 123,
        )
        self.assertEqual(
            pool.batch_history[0].completion_ns,
            first_batch.completion_ns,
        )
        self.assertEqual(request.first_token_ns, first_batch.completion_ns)
        self.assertEqual(request.state, HBFRequestState.DECODE)
        self.assertIsNone(pool.next_event_ns())

        second = pool.drain_external_dispatches()[0]
        self.assertEqual(second.arrival_ns, first_batch.completion_ns)
        second_batch = self.complete_at_minimum(
            pool, second, extra_ns=77)
        self.assertEqual(
            pool.pop_completed(), [request])
        self.assertEqual(request.state, HBFRequestState.COMPLETE)
        self.assertEqual(request.completion_ns, second_batch.completion_ns)
        self.assertEqual(
            request.tpot_ns,
            second_batch.completion_ns - first_batch.completion_ns,
        )
        self.assertEqual(pool.metrics.astra_completed_batches, 2)
        dependency_total_ns = (
            first.projection.dependency_critical_path_ns()
            + second.projection.dependency_critical_path_ns()
        )
        solo_resource_total_ns = (
            first.projection
            .solo_resource_serialized_completion_ns()
            + second.projection
            .solo_resource_serialized_completion_ns()
        )
        self.assertEqual(
            pool.metrics.astra_resource_delay_ns,
            123 + 77,
        )
        self.assertEqual(
            pool.metrics.astra_completion_elapsed_ns,
            dependency_total_ns + 123 + 77,
        )
        self.assertEqual(
            pool.metrics.modeled_batch_ns,
            dependency_total_ns,
        )
        self.assertEqual(
            pool.metrics.astra_dependency_critical_path_ns,
            dependency_total_ns,
        )
        self.assertEqual(
            pool.metrics
            .astra_solo_resource_serialized_completion_ns,
            solo_resource_total_ns,
        )
        self.assertEqual(
            pool.metrics
            .astra_actual_resource_serialized_completion_ns,
            dependency_total_ns + 123 + 77,
        )
        self.assertEqual(
            pool.metrics
            .astra_internal_resource_serialization_wait_ns,
            solo_resource_total_ns - dependency_total_ns,
        )
        self.assertEqual(
            pool.metrics.astra_signed_interference_delta_ns,
            dependency_total_ns + 123 + 77 - solo_resource_total_ns,
        )
        self.assertEqual(
            pool.metrics.astra_resource_delay_ns,
            (
                pool.metrics
                .astra_internal_resource_serialization_wait_ns
                + pool.metrics.astra_signed_interference_delta_ns
            ),
        )
        self.assertEqual(pool.run_until_idle(), [])
        pool.assert_invariants()

    def test_callback_can_defer_launch_for_cotimed_arrival_batching(self):
        pool = self.make_pool()
        first_request = self.request(output_tokens=2)
        pool.submit(first_request, now_ns=0)
        first_dispatch = pool.drain_external_dispatches()[0]
        first_completion = (
            first_dispatch.arrival_ns
            + first_dispatch.projection
            .dependency_critical_path_ns())
        pool.complete_external_dispatch(
            first_dispatch.job_id,
            first_dispatch.arrival_ns,
            first_completion,
            first_dispatch.stage_count,
            defer_schedule=True,
        )
        self.assertTrue(pool.has_pending())
        self.assertFalse(pool.has_pending_external_dispatches())

        cotimed = self.request(
            request_id=2,
            arrival_ns=first_completion,
            output_tokens=1,
        )
        pool.submit(cotimed, now_ns=first_completion)
        mixed_dispatch = pool.drain_external_dispatches()[0]
        self.assertEqual(
            {item.request_id for item in mixed_dispatch.batch.items},
            {1, 2},
        )
        pool.assert_invariants()

    def test_callback_validation_is_strict_and_non_mutating(self):
        pool = self.make_pool()
        request = self.request()
        pool.submit(request, now_ns=0)
        pending = pool._external_outbox[0]
        minimum = (
            pending.arrival_ns
            + pending.projection
            .dependency_critical_path_ns())

        with self.assertRaisesRegex(RuntimeError, "was not drained"):
            pool.complete_external_dispatch(
                pending.job_id,
                pending.arrival_ns,
                minimum,
                pending.stage_count,
            )
        dispatch = pool.drain_external_dispatches()[0]
        failures = (
            (
                (dispatch.job_id, dispatch.arrival_ns + 1,
                 minimum, dispatch.stage_count),
                "arrival mismatch",
            ),
            (
                (dispatch.job_id, dispatch.arrival_ns,
                 minimum, dispatch.stage_count + 1),
                "stage-count mismatch",
            ),
            (
                (dispatch.job_id, dispatch.arrival_ns,
                 minimum - 1, dispatch.stage_count),
                "critical path",
            ),
        )
        for arguments, message in failures:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    pool.complete_external_dispatch(*arguments)
                self.assertIs(pool.workers[0].inflight, dispatch.batch)
                self.assertEqual(pool.metrics.astra_completed_batches, 0)
                pool.assert_invariants()
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            pool.complete_external_dispatch(
                "unknown-job", 0, minimum, dispatch.stage_count)
        with self.assertRaisesRegex(ValueError, "stage_count"):
            pool.complete_external_dispatch(
                dispatch.job_id, 0, minimum, True)

        completed = pool.complete_external_dispatch(
            dispatch.job_id,
            dispatch.arrival_ns,
            minimum,
            dispatch.stage_count,
        )
        self.assertEqual(completed.completion_ns, minimum)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            pool.complete_external_dispatch(
                dispatch.job_id,
                dispatch.arrival_ns,
                minimum,
                dispatch.stage_count,
            )
        self.assertEqual(pool.pop_completed(), [request])
        self.assertFalse(pool.has_pending())
        self.assertFalse(pool.has_pending_external_dispatches())
        pool.assert_invariants()

    def test_external_configuration_and_report_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "execution_backend"):
            FullModelHBFServingPool(
                repo_root=REPO_ROOT,
                hardware=HBFServerHardware(),
                layout=HBFParallelLayout.for_key("tp4"),
                execution_backend="not-a-backend",
            )
        with self.assertRaisesRegex(ValueError, "resource_calendar"):
            from serving.core.hbf_full_model_lifecycle import ResourceCalendar
            FullModelHBFServingPool(
                repo_root=REPO_ROOT,
                hardware=HBFServerHardware(),
                layout=HBFParallelLayout.for_key("tp4"),
                execution_backend="external_astra",
                resource_calendar=ResourceCalendar(),
            )

        pool = self.make_pool()
        with self.assertRaisesRegex(RuntimeError, "external_astra"):
            FullModelHBFServingPool(
                repo_root=REPO_ROOT,
                hardware=HBFServerHardware(),
                layout=HBFParallelLayout.for_key("tp4"),
            ).drain_external_dispatches()
        pool.submit(self.request(), now_ns=0)
        report = pool.report()
        self.assertEqual(report["execution_backend"], "external_astra")
        self.assertEqual(
            report["completion_time_source"],
            "external_astra_callback",
        )
        self.assertEqual(
            report["astra_projection_fidelity"],
            "per-operation-ordered-v2",
        )
        self.assertEqual(
            report["astra_projection_schema"],
            "hbf-full-model-astra-v2/ordered-v2",
        )
        self.assertIn(
            "astra_internal_resource_serialization_wait_ns",
            report["astra_timing_semantics"],
        )
        self.assertEqual(report["pending_batch_count"], 1)
        self.assertEqual(report["external_undrained_dispatch_count"], 1)
        self.assertIsNone(report["group_telemetry"][0]["npu_busy_ns"])
        self.assertIsNone(
            report["group_telemetry"][0]["npu_utilization"])

    def test_tp8_context_uses_ordered_exact_card_projection(self):
        pool = self.make_pool(layout="tp8_context")
        request = self.request()
        pool.submit(request, now_ns=0)
        dispatch, = pool.drain_external_dispatches()

        self.assertEqual(
            dispatch.projection.schema,
            "hbf-full-model-astra-v2/ordered-v2",
        )
        self.assertEqual(
            dispatch.projection.placement.layout, "tp8_context")
        attention_stages = [
            stage
            for stage in dispatch.projection.stages
            if (
                stage.operation_name is not None
                and stage.operation_name.endswith(".attention")
            )
        ]
        self.assertEqual(len(attention_stages), 48 * 8)
        self.assertEqual(
            {stage.card_id for stage in attention_stages},
            set(range(8)),
        )
        self.assertEqual(
            dispatch.projection.physical_hbf_read_bytes,
            sum(
                stage.hbf_read_bytes
                for stage in dispatch.projection.stages
            ),
        )
        completed = self.complete_at_minimum(pool, dispatch)
        self.assertEqual(completed.completion_ns, request.completion_ns)
        self.assertEqual(pool.pop_completed(), [request])


if __name__ == "__main__":
    unittest.main()

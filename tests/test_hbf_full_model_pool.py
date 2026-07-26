import dataclasses
from pathlib import Path
import random
import unittest

from serving.core.hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
    qwen_logical_kv_bytes_per_token,
)
from serving.core.hbf_full_model_lifecycle import (
    FullModelHBFLifecycle,
    PerGroupCapacityLedger,
    PlacementState,
    ResourceCalendar,
    ResumeExecution,
)
from serving.core.hbf_full_model_pool import (
    FullModelHBFServingPool,
    HBFRequestState,
    HBFServingRequest,
    derive_lpddr_workspace_bytes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class FullModelHBFServingPoolTests(unittest.TestCase):
    def make_pool(
            self, layout="tp4", *, hardware=None, calendar=None,
            max_tokens=256, max_seqs=16, chunk=64,
            validate_every_event=True,
            retain_detailed_history=True,
            execution_backend="analytical_calendar",
            analytical_resource_prefix=""):
        return FullModelHBFServingPool(
            repo_root=REPO_ROOT,
            hardware=hardware or HBFServerHardware(),
            layout=HBFParallelLayout.for_key(layout),
            resource_calendar=calendar,
            max_num_batched_tokens=max_tokens,
            max_num_seqs=max_seqs,
            max_prefill_chunk_tokens=chunk,
            validate_every_event=validate_every_event,
            retain_detailed_history=retain_detailed_history,
            execution_backend=execution_backend,
            analytical_resource_prefix=analytical_resource_prefix,
        )

    @staticmethod
    def request(
            request_id, *, group=0, arrival=0, input_tokens=100,
            output_tokens=4, hbf=90, lpddr=0):
        return HBFServingRequest(
            request_id=request_id,
            session_id=f"s-{request_id}",
            arrival_ns=arrival,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            hbf_prefix_tokens=hbf,
            lpddr_prefix_tokens=lpddr,
            group_id=group,
        )

    def test_workspace_is_derived_and_fits_default_lpddr(self):
        values = {
            key: derive_lpddr_workspace_bytes(
                HBFParallelLayout.for_key(key),
                max_num_batched_tokens=8_192,
                max_num_seqs=128,
            )
            for key in ("dp8", "tp4", "tp8")
        }
        self.assertGreater(values["dp8"], values["tp4"])
        self.assertGreater(values["tp4"], 2 * 1024 ** 3)
        self.assertTrue(all(
            value < 64 * 1024 ** 3 for value in values.values()))

    def test_output_one_has_ttft_and_no_tpot(self):
        pool = self.make_pool()
        request = self.request(1, output_tokens=1)
        pool.submit(request, now_ns=0)
        completed = pool.run_until_idle()
        self.assertEqual(completed, [request])
        self.assertEqual(request.state, HBFRequestState.COMPLETE)
        self.assertIsNotNone(request.ttft_ns)
        self.assertGreater(request.ttft_ns, 0)
        self.assertIsNone(request.tpot_ns)
        self.assertEqual(len(request.token_completion_ns), 1)

    def test_logical_arrival_may_precede_internal_submit_time(self):
        pool = self.make_pool()
        request = self.request(
            1, arrival=10, output_tokens=1)
        pool.submit(request, now_ns=25)
        completed = pool.run_until_idle()
        self.assertEqual(completed, [request])
        self.assertGreater(request.first_token_ns, 25)
        self.assertEqual(
            request.ttft_ns,
            request.first_token_ns - request.arrival_ns,
        )

    def test_fully_cached_request_executes_first_decode_iteration(self):
        pool = self.make_pool()
        request = self.request(
            1, input_tokens=100, hbf=90, lpddr=10,
            output_tokens=2)
        pool.submit(request, now_ns=0)
        pool.run_until_idle()
        self.assertEqual(request.prefill_processed_tokens, 0)
        self.assertEqual(request.generated_tokens, 2)
        self.assertGreater(request.ttft_ns, 0)
        self.assertGreater(request.tpot_ns, 0)
        self.assertEqual(request.active_lpddr_tokens, 11)
        self.assertEqual(
            pool.batch_history[0].items[0].kind, "first_decode")

    def test_chunked_prefill_then_decode_preserves_token_counts(self):
        pool = self.make_pool(max_tokens=32, chunk=16)
        request = self.request(
            1, input_tokens=100, hbf=50, output_tokens=5)
        pool.submit(request, now_ns=0)
        pool.run_until_idle()
        self.assertEqual(request.prefill_processed_tokens, 50)
        self.assertEqual(request.generated_tokens, 5)
        self.assertEqual(request.active_lpddr_tokens, 54)
        self.assertEqual(len(request.token_completion_ns), 5)
        self.assertEqual(
            [item.query_tokens
             for batch in pool.batch_history
             for item in batch.items
             if item.kind == "prefill"],
            [16, 16, 16, 2],
        )
        self.assertEqual(
            sum(item.kind == "decode"
                for batch in pool.batch_history
                for item in batch.items),
            4,
        )

    def test_prefill_drain_gates_decode_without_changing_admission_hit(self):
        pool = FullModelHBFServingPool(
            repo_root=REPO_ROOT,
            hardware=HBFServerHardware(),
            layout=HBFParallelLayout.for_key("tp4"),
            max_num_batched_tokens=32,
            max_num_seqs=4,
            max_prefill_chunk_tokens=16,
            prefill_drain_tail_tokens=2,
            prefill_drain_min_tokens=1,
        )
        request = self.request(
            1,
            input_tokens=100,
            output_tokens=3,
            hbf=90,
            lpddr=0,
        )
        pool.submit(request, now_ns=0)
        while request.state != HBFRequestState.PREFILL_DRAIN:
            pool.advance(pool.next_event_ns())

        first_token_ns = request.first_token_ns
        self.assertIsNotNone(first_token_ns)
        self.assertEqual(request.generated_tokens, 1)
        self.assertEqual(request.cached_tokens, 90)
        self.assertEqual(request.fresh_tokens, 10)
        self.assertEqual(request.prefill_processed_tokens, 10)
        self.assertEqual(request.active_lpddr_tokens, 10)
        self.assertEqual(
            sum(len(worker.active_decode) for worker in pool.workers), 0)

        self.assertEqual(
            pool.claim_prefill_drain_requests(), (request,))
        with self.assertRaisesRegex(ValueError, "integers"):
            pool.publish_prefill_drain_placement(
                request.request_id,
                hbf_tokens=90,
                lpddr_tokens=True,
            )
        original_placement = (
            request.hbf_prefix_tokens,
            request.lpddr_prefix_tokens,
            request.published_growth_tokens,
        )
        with self.assertRaisesRegex(RuntimeError, "LPDDR ledger"):
            pool.publish_prefill_drain_placement(
                request.request_id,
                hbf_tokens=98,
                lpddr_tokens=2,
            )
        self.assertEqual(
            (
                request.hbf_prefix_tokens,
                request.lpddr_prefix_tokens,
                request.published_growth_tokens,
            ),
            original_placement,
        )
        pool.publish_prefill_drain_placement(
            request.request_id,
            hbf_tokens=90,
            lpddr_tokens=10,
        )
        with self.assertRaisesRegex(RuntimeError, "back to LPDDR"):
            pool.publish_prefill_drain_placement(
                request.request_id,
                hbf_tokens=85,
                lpddr_tokens=15,
            )
        with self.assertRaisesRegex(RuntimeError, "token count"):
            pool.bind_prefill_drain_job(
                request.request_id,
                job_id=16,
                logical_tokens=7,
            )
        pool.bind_prefill_drain_job(
            request.request_id,
            job_id=17,
            logical_tokens=8,
        )
        with self.assertRaisesRegex(RuntimeError, "identity"):
            pool.clear_prefill_drain_job(
                request.request_id, job_id=18)
        pool.clear_prefill_drain_job(
            request.request_id, job_id=17)
        pool.bind_prefill_drain_job(
            request.request_id,
            job_id=18,
            logical_tokens=8,
        )
        with self.assertRaisesRegex(RuntimeError, "LPDDR tail"):
            pool.release_prefill_drain(
                request.request_id,
                now_ns=first_token_ns,
                job_id=18,
            )
        with self.assertRaisesRegex(RuntimeError, "bound job"):
            pool.release_prefill_drain(
                request.request_id,
                now_ns=first_token_ns,
                job_id=18,
                fallback=True,
            )

        pool.lpddr_ledger.set_card_bytes(
            request.group_id,
            pool._lpddr_owner(request.session_id),
            pool._range_card_bytes(
                request.group_id,
                token_start=98,
                token_count=2,
            ),
        )
        pool.publish_prefill_drain_placement(
            request.request_id,
            hbf_tokens=98,
            lpddr_tokens=2,
        )
        drain_done_ns = first_token_ns + 1_000
        pool.advance(drain_done_ns, defer_schedule=True)
        pool.release_prefill_drain(
            request.request_id,
            now_ns=drain_done_ns,
            job_id=18,
        )
        pool.flush_scheduling(drain_done_ns)

        decode = pool.batch_history[-1]
        self.assertEqual(decode.shape.decode_hbf_k, (98,))
        self.assertEqual(decode.shape.decode_lpddr_k, (2,))
        self.assertEqual(request.first_token_ns, first_token_ns)
        self.assertEqual(request.cached_tokens, 90)
        self.assertEqual(
            request.admitted_hbf_prefix_tokens, 90)
        self.assertEqual(
            request.admitted_lpddr_prefix_tokens, 0)
        self.assertEqual(pool.metrics.prefill_drain_candidates, 1)
        self.assertEqual(pool.metrics.prefill_drain_started, 2)
        self.assertEqual(pool.metrics.prefill_drain_completed, 1)
        self.assertEqual(pool.metrics.prefill_drain_wait_ns, 1_000)
        pool.run_until_idle()

    def test_prefill_drain_output_one_bypasses_gate(self):
        pool = FullModelHBFServingPool(
            repo_root=REPO_ROOT,
            hardware=HBFServerHardware(),
            layout=HBFParallelLayout.for_key("tp4"),
            max_num_batched_tokens=32,
            max_num_seqs=4,
            max_prefill_chunk_tokens=16,
            prefill_drain_tail_tokens=0,
            prefill_drain_min_tokens=0,
        )
        request = self.request(
            1,
            input_tokens=100,
            output_tokens=1,
            hbf=90,
        )
        self.assertEqual(pool.run_until_idle(), [])
        pool.submit(request, now_ns=0)
        self.assertEqual(pool.run_until_idle(), [request])
        self.assertEqual(pool.metrics.prefill_drain_candidates, 0)

    def test_fully_cached_lpddr_prefix_can_gate_after_first_decode(self):
        pool = FullModelHBFServingPool(
            repo_root=REPO_ROOT,
            hardware=HBFServerHardware(),
            layout=HBFParallelLayout.for_key("tp4"),
            max_num_batched_tokens=32,
            max_num_seqs=4,
            max_prefill_chunk_tokens=16,
            prefill_drain_tail_tokens=2,
            prefill_drain_min_tokens=1,
        )
        request = self.request(
            1,
            input_tokens=100,
            output_tokens=2,
            hbf=90,
            lpddr=10,
        )
        pool.submit(request, now_ns=0)
        pool.advance(pool.next_event_ns())
        self.assertEqual(
            request.state, HBFRequestState.PREFILL_DRAIN)
        self.assertEqual(request.prefill_processed_tokens, 0)
        self.assertEqual(request.generated_tokens, 1)
        self.assertEqual(
            pool.claim_prefill_drain_requests(), (request,))
        pool.publish_prefill_drain_placement(
            request.request_id,
            hbf_tokens=90,
            lpddr_tokens=10,
        )
        pool.release_prefill_drain(
            request.request_id,
            now_ns=request.first_token_ns,
            fallback=True,
        )
        pool.flush_scheduling(request.first_token_ns)
        self.assertEqual(pool.run_until_idle(), [request])
        self.assertEqual(pool.metrics.prefill_drain_fallbacks, 1)

    def test_prefill_drain_queue_class_is_exact(self):
        pool = FullModelHBFServingPool(
            repo_root=REPO_ROOT,
            hardware=HBFServerHardware(),
            layout=HBFParallelLayout.for_key("tp4"),
            max_num_batched_tokens=32,
            max_num_seqs=4,
            max_prefill_chunk_tokens=16,
            prefill_drain_tail_tokens=2,
            prefill_drain_min_tokens=1,
        )
        request = self.request(
            1,
            input_tokens=100,
            output_tokens=2,
            hbf=90,
        )
        pool.submit(request, now_ns=0)
        while request.state != HBFRequestState.PREFILL_DRAIN:
            pool.advance(pool.next_event_ns())
        worker = pool.workers[request.group_id]
        worker.prefill_drain.remove(request.request_id)
        worker.active_decode.append(request.request_id)
        with self.assertRaisesRegex(
                AssertionError, "queue ownership"):
            pool.assert_invariants()

    def test_same_timestamp_submissions_form_one_batch(self):
        pool = self.make_pool()
        requests = [
            self.request(index, input_tokens=100, hbf=90)
            for index in range(4)
        ]
        pool.submit_many(requests, now_ns=0)
        first = pool.batch_history[0]
        self.assertEqual(len(first.items), 4)
        pool.run_until_idle()
        self.assertEqual(pool.metrics.completed_requests, 4)
        self.assertGreater(pool.metrics.max_batch_size, 1)

    def test_submission_can_defer_one_same_time_schedule_barrier(self):
        pool = self.make_pool()
        requests = [
            self.request(index, input_tokens=100, hbf=90)
            for index in range(2)
        ]
        pool.submit_many(
            requests, now_ns=0, defer_schedule=True)
        self.assertEqual(pool.batch_history, [])
        self.assertEqual(
            sum(len(worker.waiting) for worker in pool.workers), 2)
        pool.flush_scheduling(0)
        self.assertEqual(len(pool.batch_history[0].items), 2)
        self.assertEqual(pool.run_until_idle(), requests)
        with self.assertRaisesRegex(ValueError, "defer_schedule"):
            self.make_pool().submit_many(
                (), now_ns=0, defer_schedule=1)

    def test_decode_first_mixes_active_decode_with_new_prefill(self):
        pool = self.make_pool(max_tokens=64, chunk=32)
        long_output = self.request(
            1, input_tokens=100, hbf=99, output_tokens=4)
        pool.submit(long_output, now_ns=0)
        first_done = pool.next_completion_ns()
        pool.advance(first_done, defer_schedule=True)
        newcomer = self.request(
            2, arrival=first_done, input_tokens=120, hbf=100,
            output_tokens=1)
        pool.submit(newcomer, now_ns=first_done)
        mixed = pool.batch_history[-1]
        self.assertTrue(any(
            item.kind == "decode" for item in mixed.items))
        self.assertTrue(any(
            item.kind == "prefill" for item in mixed.items))
        pool.run_until_idle()

    def test_dp8_groups_execute_on_disjoint_resources(self):
        pool = self.make_pool("dp8")
        requests = (
            self.request(1, group=0),
            self.request(2, group=1),
        )
        pool.submit_many(requests, now_ns=0)
        batches = pool.batch_history[:2]
        self.assertEqual({batch.group_id for batch in batches}, {0, 1})
        self.assertEqual(batches[0].start_ns, batches[1].start_ns)

    def test_tp_collectives_are_present_in_batch_telemetry(self):
        values = {}
        for key in ("dp8", "tp4", "tp8"):
            pool = self.make_pool(key)
            pool.submit(self.request(1), now_ns=0)
            values[key] = pool.batch_history[0].latency
        self.assertEqual(values["dp8"].collective_ns, 0)
        self.assertGreater(values["tp4"].collective_ns, 0)
        self.assertGreater(values["tp8"].collective_ns, 0)

    def test_compact_metrics_preserve_latency_component_audit(self):
        pool = self.make_pool(
            "tp4", retain_detailed_history=False)
        requests = (
            self.request(1, input_tokens=100, hbf=90),
            self.request(2, input_tokens=120, hbf=100),
        )
        pool.submit_many(requests, now_ns=0)
        pool.run_until_idle()
        metrics = pool.metrics
        self.assertEqual(
            metrics.modeled_batch_ns,
            (
                metrics.embedding_modeled_ns
                + metrics.dense_modeled_ns
                + metrics.attention_modeled_ns
                + metrics.router_modeled_ns
                + metrics.moe_modeled_ns
                + metrics.final_modeled_ns
                + metrics.collective_modeled_ns
            ),
        )
        self.assertEqual(
            metrics.batches,
            (
                metrics.attention_compute_dominant_batches
                + metrics.attention_hbf_dominant_batches
                + metrics.attention_lpddr_dominant_batches
            ),
        )
        self.assertGreater(metrics.attention_modeled_ns, 0)
        self.assertGreater(metrics.attention_lpddr_roof_ns, 0)
        self.assertEqual(pool.batch_history, [])

    def test_foreground_waits_for_existing_hbf_media_write(self):
        calendar = ResourceCalendar()
        calendar.reserve_parallel(
            arrival_ns=0,
            job_id=999,
            kind="append",
            demands={
                f"hbf-card-{card}-media": (1_000_000, 100)
                for card in range(4)
            },
        )
        pool = self.make_pool("tp4", calendar=calendar)
        pool.submit(self.request(1), now_ns=0)
        self.assertEqual(pool.batch_history, [])
        self.assertEqual(pool.next_event_ns(), 1_000_000)
        pool.advance(1_000_000)
        batch = pool.batch_history[0]
        self.assertEqual(batch.start_ns, 1_000_000)
        self.assertEqual(pool.metrics.resource_delay_ns, 1_000_000)

    def test_prefixed_pools_share_calendar_but_not_server_local_resources(self):
        calendar = ResourceCalendar()
        first = self.make_pool(
            "tp4",
            calendar=calendar,
            analytical_resource_prefix="server-0:",
        )
        second = self.make_pool(
            "tp4",
            calendar=calendar,
            analytical_resource_prefix="server-1:",
        )

        first.submit(self.request(1), now_ns=0)
        second.submit(self.request(1), now_ns=0)

        self.assertEqual(first.batch_history[0].start_ns, 0)
        self.assertEqual(second.batch_history[0].start_ns, 0)
        resources = set(calendar.available_ns)
        self.assertIn("server-0:hbf-group-0-npu", resources)
        self.assertIn("server-1:hbf-group-0-npu", resources)
        self.assertIn("server-0:hbf-card-0-media", resources)
        self.assertIn("server-1:hbf-card-0-media", resources)
        self.assertNotIn("hbf-group-0-npu", resources)
        self.assertEqual(
            first.report()["analytical_resource_prefix"],
            "server-0:",
        )
        telemetry = first.report()["group_telemetry"][0]
        self.assertEqual(telemetry["compute_device"], "h100_class_gpu")
        self.assertEqual(telemetry["gpu_busy_ns"], telemetry["npu_busy_ns"])
        self.assertEqual(
            telemetry["gpu_utilization"], telemetry["npu_utilization"])
        self.assertGreater(telemetry["gpu_busy_ns"], 0)

    def test_prefixed_lifecycle_write_blocks_only_matching_pool(self):
        calendar = ResourceCalendar()
        calendar.reserve_parallel(
            arrival_ns=0,
            job_id=999,
            kind="append",
            demands={
                f"server-0:hbf-card-{card}-media": (1_000_000, 100)
                for card in range(4)
            },
        )
        blocked = self.make_pool(
            "tp4",
            calendar=calendar,
            analytical_resource_prefix="server-0:",
        )
        independent = self.make_pool(
            "tp4",
            calendar=calendar,
            analytical_resource_prefix="server-1:",
        )

        blocked.submit(self.request(1), now_ns=0)
        independent.submit(self.request(1), now_ns=0)

        self.assertEqual(blocked.batch_history, [])
        self.assertEqual(blocked.next_event_ns(), 1_000_000)
        self.assertEqual(independent.batch_history[0].start_ns, 0)

    def test_empty_analytical_prefix_preserves_pool_report_and_names(self):
        pool = self.make_pool(analytical_resource_prefix="")
        pool.submit(self.request(1), now_ns=0)
        report = pool.report()
        self.assertNotIn("analytical_resource_prefix", report)
        self.assertIn("hbf-group-0-npu", pool.calendar.available_ns)
        self.assertNotIn("server-0:hbf-group-0-npu",
                         pool.calendar.available_ns)

    def test_external_pool_rejects_analytical_resource_prefix(self):
        with self.assertRaisesRegex(
                ValueError, "analytical_resource_prefix"):
            self.make_pool(
                execution_backend="external_astra",
                analytical_resource_prefix="server-0:",
            )

    def test_arrival_before_delayed_launch_joins_batch(self):
        calendar = ResourceCalendar()
        calendar.reserve_parallel(
            arrival_ns=0,
            job_id=999,
            kind="append",
            demands={
                f"hbf-card-{card}-media": (1_000_000, 100)
                for card in range(4)
            },
        )
        pool = self.make_pool("tp4", calendar=calendar)
        first = self.request(1, arrival=0)
        second = self.request(2, arrival=500_000)
        pool.submit(first, now_ns=0)
        pool.submit(second, now_ns=500_000)
        self.assertEqual(pool.batch_history, [])
        pool.advance(1_000_000)
        self.assertEqual(
            [item.request_id
             for item in pool.batch_history[0].items],
            [1, 2],
        )

    def test_lifecycle_migration_resume_append_integration(self):
        hardware = HBFServerHardware()
        layout = HBFParallelLayout.for_key("tp4")
        calendar = ResourceCalendar()
        workspace = derive_lpddr_workspace_bytes(
            layout,
            max_num_batched_tokens=256,
            max_num_seqs=16,
        )
        ledger = PerGroupCapacityLedger(
            group_count=layout.replicas,
            capacity_bytes=(
                hardware.lpddr_capacity_bytes_per_card - workspace),
        )
        lifecycle = FullModelHBFLifecycle(
            hardware=hardware,
            layout=layout,
            resource_calendar=calendar,
            lpddr_ledger=ledger,
        )
        pool = FullModelHBFServingPool(
            repo_root=REPO_ROOT,
            hardware=hardware,
            layout=layout,
            resource_calendar=calendar,
            lpddr_ledger=ledger,
            placement_resolver=lifecycle.placement_snapshot,
            max_num_batched_tokens=256,
            max_num_seqs=16,
            max_prefill_chunk_tokens=64,
        )

        record = lifecycle.register_session("session")
        migration = lifecycle.complete_gpu_turn(
            "session",
            now_ns=0,
            total_tokens=1_000,
            has_successor=True,
        )
        self.assertIsNotNone(migration)
        lifecycle.advance(migration.completion_ns)
        route = lifecycle.route_resume(
            "session",
            now_ns=migration.completion_ns,
            request_id=1,
        )
        self.assertEqual(route.execution, ResumeExecution.HBF)
        self.assertEqual(route.hbf_tokens, 1_000)
        self.assertEqual(route.lpddr_tokens, 0)

        first = HBFServingRequest(
            request_id=1,
            session_id="session",
            arrival_ns=migration.completion_ns,
            input_tokens=1_005,
            output_tokens=3,
            hbf_prefix_tokens=route.hbf_tokens,
            lpddr_prefix_tokens=route.lpddr_tokens,
            group_id=route.group_id,
        )
        pool.submit(first, now_ns=first.arrival_ns)
        completed = pool.run_until_idle()
        self.assertEqual(completed, [first])
        self.assertGreater(first.ttft_ns, 0)
        self.assertGreaterEqual(
            pool.batch_history[0].start_ns,
            migration.completion_ns,
        )

        append = lifecycle.complete_hbf_turn(
            "session",
            now_ns=first.completion_ns,
            total_tokens=first.input_tokens + first.output_tokens - 1,
            has_successor=True,
        )
        self.assertIsNotNone(append)
        self.assertEqual(record.lpddr_tokens, 7)
        next_arrival = max(
            first.completion_ns,
            append.completion_ns - 1,
        )
        second_route = lifecycle.route_resume(
            "session",
            now_ns=next_arrival,
            request_id=2,
        )
        self.assertEqual(second_route.execution, ResumeExecution.HBF)
        self.assertEqual(second_route.hbf_tokens, 1_000)
        self.assertEqual(second_route.lpddr_tokens, 7)
        self.assertEqual(second_route.reason, "hbf_append_inflight")
        self.assertEqual(record.state, PlacementState.HBF_ACTIVE)

        second = HBFServingRequest(
            request_id=2,
            session_id="session",
            arrival_ns=next_arrival,
            input_tokens=1_010,
            output_tokens=1,
            hbf_prefix_tokens=second_route.hbf_tokens,
            lpddr_prefix_tokens=second_route.lpddr_tokens,
            group_id=second_route.group_id,
        )
        batch_count_before_second = len(pool.batch_history)
        pool.submit(second, now_ns=next_arrival)
        self.assertEqual(
            len(pool.batch_history), batch_count_before_second)
        self.assertEqual(pool.next_event_ns(), append.completion_ns)
        lifecycle.advance(append.completion_ns)
        pool.advance(append.completion_ns)
        launched = pool.batch_history[-1]
        self.assertEqual(launched.shape.prefill_hbf_k, (1_007,))
        self.assertEqual(launched.shape.prefill_lpddr_k, (0,))
        second_done = pool.run_until_idle()
        self.assertEqual(second_done, [second])
        self.assertGreaterEqual(
            pool.batch_history[-1].start_ns,
            append.completion_ns,
        )
        lifecycle.complete_hbf_turn(
            "session",
            now_ns=second.completion_ns,
            total_tokens=second.input_tokens + second.output_tokens - 1,
            has_successor=False,
        )
        self.assertEqual(record.state, PlacementState.ENDED)
        lifecycle.assert_invariants()
        pool.assert_invariants()
        append_resources = {
            row.resource for row in calendar.reservations
            if (
                row.namespace == "hbf-lifecycle"
                and row.job_id == append.job_id
                and row.kind == "append"
            )
        }
        self.assertTrue(any(
            resource.endswith("-lpddr")
            for resource in append_resources))
        self.assertTrue(any(
            resource.endswith("-media")
            for resource in append_resources))
        self.assertTrue(all(
            row.namespace in {
                "hbf-lifecycle", "hbf-pool",
            }
            for row in calendar.reservations
            if row.kind in {
                "migration", "append", "hbf-model-batch",
            }
        ))

    def test_idle_lpddr_and_active_request_share_exact_ledger(self):
        layout = HBFParallelLayout.for_key("tp4")
        workspace = derive_lpddr_workspace_bytes(
            layout,
            max_num_batched_tokens=256,
            max_num_seqs=16,
        )
        per_token = 98_304 // 4
        hardware = dataclasses.replace(
            HBFServerHardware(),
            lpddr_capacity_bytes_per_card=workspace + 10 * per_token,
        )
        ledger = PerGroupCapacityLedger(
            group_count=layout.replicas,
            capacity_bytes=10 * per_token,
        )
        ledger.set_bytes(0, "idle-session", 8 * per_token)
        pool = FullModelHBFServingPool(
            repo_root=REPO_ROOT,
            hardware=hardware,
            layout=layout,
            lpddr_ledger=ledger,
            max_num_batched_tokens=256,
            max_num_seqs=16,
            max_prefill_chunk_tokens=64,
        )
        request = self.request(
            1, input_tokens=4, hbf=1, lpddr=3,
            output_tokens=1)
        with self.assertRaisesRegex(RuntimeError, "LPDDR"):
            pool.submit(request, now_ns=0)

        ledger.set_bytes(0, "idle-session", 7 * per_token)
        request = self.request(
            2, input_tokens=4, hbf=1, lpddr=3,
            output_tokens=1)
        pool.submit(request, now_ns=0)
        self.assertEqual(ledger.used_bytes(0), 10 * per_token)
        pool.run_until_idle()

    def test_finish_headroom_is_reserved_atomically_at_admission(self):
        layout = HBFParallelLayout.for_key("tp8")
        workspace = derive_lpddr_workspace_bytes(
            layout,
            max_num_batched_tokens=16,
            max_num_seqs=4,
        )
        per_token = 98_304 * 2 // 8
        hardware = dataclasses.replace(
            HBFServerHardware(),
            lpddr_capacity_bytes_per_card=(
                workspace + 4 * per_token),
        )
        ledger = PerGroupCapacityLedger(
            group_count=1,
            capacity_bytes=4 * per_token,
        )
        pool = FullModelHBFServingPool(
            repo_root=REPO_ROOT,
            hardware=hardware,
            layout=layout,
            lpddr_ledger=ledger,
            max_num_batched_tokens=16,
            max_num_seqs=4,
            max_prefill_chunk_tokens=16,
        )
        requests = [
            self.request(
                request_id,
                input_tokens=1,
                output_tokens=4,
                hbf=1,
                lpddr=0,
            )
            for request_id in (1, 2)
        ]
        with self.assertRaisesRegex(RuntimeError, "LPDDR"):
            pool.submit_many(requests, now_ns=0)
        self.assertEqual(ledger.used_bytes(0), 0)
        self.assertEqual(pool.requests, {})

        pool.submit(requests[0], now_ns=0)
        self.assertEqual(ledger.used_bytes(0), 3 * per_token)
        pool.run_until_idle()
        self.assertEqual(
            ledger.owner_bytes("hbf-request-headroom:1"), 0)
        self.assertEqual(ledger.used_bytes(0), 3 * per_token)

    def test_tp8_context_disjoint_one_token_parities_share_capacity(self):
        layout = HBFParallelLayout.for_key("tp8_context")
        per_head = qwen_logical_kv_bytes_per_token() // 4
        workspace = derive_lpddr_workspace_bytes(
            layout,
            max_num_batched_tokens=8,
            max_num_seqs=2,
        )
        hardware = dataclasses.replace(
            HBFServerHardware(),
            lpddr_capacity_bytes_per_card=workspace + per_head,
        )
        ledger = PerGroupCapacityLedger(
            group_count=1,
            capacity_bytes=per_head,
        )
        pool = FullModelHBFServingPool(
            repo_root=REPO_ROOT,
            hardware=hardware,
            layout=layout,
            lpddr_ledger=ledger,
            max_num_batched_tokens=8,
            max_num_seqs=2,
            max_prefill_chunk_tokens=8,
        )
        even = HBFServingRequest(
            request_id=1,
            session_id="even",
            arrival_ns=0,
            input_tokens=1,
            output_tokens=1,
            hbf_prefix_tokens=0,
            lpddr_prefix_tokens=1,
            group_id=0,
        )
        odd = HBFServingRequest(
            request_id=2,
            session_id="odd",
            arrival_ns=0,
            input_tokens=2,
            output_tokens=1,
            hbf_prefix_tokens=1,
            lpddr_prefix_tokens=1,
            group_id=0,
        )
        pool.submit_many((even, odd), now_ns=0)
        self.assertEqual(
            ledger.used_bytes_by_card(0),
            {card_id: per_head for card_id in range(8)},
        )
        completed = pool.run_until_idle()
        self.assertEqual(
            {request.request_id for request in completed}, {1, 2})
        pool.assert_invariants()

    def test_tp8_context_same_parity_one_token_sessions_overflow(self):
        layout = HBFParallelLayout.for_key("tp8_context")
        per_head = qwen_logical_kv_bytes_per_token() // 4
        workspace = derive_lpddr_workspace_bytes(
            layout,
            max_num_batched_tokens=8,
            max_num_seqs=2,
        )
        hardware = dataclasses.replace(
            HBFServerHardware(),
            lpddr_capacity_bytes_per_card=workspace + per_head,
        )
        ledger = PerGroupCapacityLedger(
            group_count=1,
            capacity_bytes=per_head,
        )
        pool = FullModelHBFServingPool(
            repo_root=REPO_ROOT,
            hardware=hardware,
            layout=layout,
            lpddr_ledger=ledger,
            max_num_batched_tokens=8,
            max_num_seqs=2,
            max_prefill_chunk_tokens=8,
        )
        requests = tuple(
            HBFServingRequest(
                request_id=request_id,
                session_id=f"even-{request_id}",
                arrival_ns=0,
                input_tokens=1,
                output_tokens=1,
                hbf_prefix_tokens=0,
                lpddr_prefix_tokens=1,
                group_id=0,
            )
            for request_id in (1, 2)
        )
        with self.assertRaisesRegex(
                RuntimeError, "LPDDR exceeds capacity"):
            pool.submit_many(requests, now_ns=0)
        self.assertEqual(ledger.used_bytes(0), 0)
        self.assertEqual(pool.requests, {})

    def test_model_media_occupancy_uses_roof_not_wall_time(self):
        pool = self.make_pool("tp4")
        pool.submit(self.request(1), now_ns=0)
        batch = pool.batch_history[0]
        rows = [
            row for row in pool.calendar.reservations
            if (
                row.namespace == "hbf-pool"
                and row.job_id == batch.batch_id
            )
        ]
        npu = [
            row for row in rows if row.resource.endswith("-npu")][0]
        media = [
            row for row in rows if row.resource.endswith("-media")][0]
        lpddr = [
            row for row in rows if row.resource.endswith("-lpddr")][0]
        self.assertEqual(npu.service_ns, batch.latency.total_ns)
        self.assertEqual(
            media.service_ns,
            min(
                batch.latency.total_ns,
                batch.latency.hbf_roof_ns_sum,
            ),
        )
        self.assertEqual(
            lpddr.service_ns,
            min(
                batch.latency.total_ns,
                batch.latency.lpddr_roof_ns_sum,
            ),
        )

    def test_tiny_lpddr_fails_fast_when_workspace_does_not_fit(self):
        base = HBFServerHardware()
        tiny = dataclasses.replace(
            base, lpddr_capacity_bytes_per_card=1024)
        with self.assertRaisesRegex(ValueError, "workspace"):
            self.make_pool(hardware=tiny)

    def test_max_sequences_and_final_context_are_bounded(self):
        with self.assertRaisesRegex(ValueError, "calibrated"):
            self.make_pool(max_seqs=129)
        request = self.request(
            1,
            input_tokens=1_010_000,
            output_tokens=2,
            hbf=1_010_000,
        )
        with self.assertRaisesRegex(ValueError, "output"):
            self.make_pool().submit(request, now_ns=0)

    def test_reused_mutable_request_is_rejected(self):
        pool = self.make_pool()
        request = self.request(1)
        pool.submit(request, now_ns=0)
        pool.run_until_idle()
        with self.assertRaisesRegex(ValueError, "pristine"):
            self.make_pool().submit(request, now_ns=0)

    def test_deterministic_random_workload_conserves_tokens(self):
        def run_once():
            rng = random.Random(17)
            pool = self.make_pool(
                "tp4", max_tokens=128, max_seqs=32, chunk=32)
            requests = []
            for request_id in range(100):
                input_tokens = rng.randint(20, 300)
                cached = rng.randint(0, input_tokens)
                request = self.request(
                    request_id,
                    group=request_id % 2,
                    input_tokens=input_tokens,
                    hbf=cached,
                    output_tokens=rng.randint(1, 12),
                )
                requests.append(request)
            pool.submit_many(requests, now_ns=0)
            completed = pool.run_until_idle()
            self.assertEqual(len(completed), len(requests))
            self.assertEqual(
                sum(item.prefill_processed_tokens
                    for item in completed),
                sum(item.fresh_tokens for item in requests),
            )
            self.assertEqual(
                sum(item.generated_tokens for item in completed),
                sum(item.output_tokens for item in requests),
            )
            pool.assert_invariants()
            return [
                (
                    item.request_id,
                    item.first_token_ns,
                    item.completion_ns,
                    tuple(item.token_completion_ns),
                )
                for item in sorted(
                    completed, key=lambda value: value.request_id)
            ]

        self.assertEqual(run_once(), run_once())

    def test_strict_and_sweep_pool_are_event_equivalent(self):
        def run(validate_every_event):
            pool = self.make_pool(
                "tp4",
                max_tokens=128,
                max_seqs=16,
                chunk=32,
                validate_every_event=validate_every_event,
            )
            requests = [
                self.request(
                    request_id,
                    group=request_id % 2,
                    input_tokens=80 + request_id * 3,
                    output_tokens=1 + request_id % 5,
                    hbf=40 + request_id,
                )
                for request_id in range(24)
            ]
            pool.submit_many(requests, now_ns=0)
            completed = pool.run_until_idle()
            return {
                "completed": [
                    (
                        request.request_id,
                        request.first_token_ns,
                        request.completion_ns,
                        tuple(request.token_completion_ns),
                    )
                    for request in sorted(
                        completed, key=lambda item: item.request_id)
                ],
                "metrics": dataclasses.asdict(pool.metrics),
                "batches": list(pool.batch_history),
                "available_ns": dict(pool.calendar.available_ns),
                "busy_ns": dict(pool.calendar.busy_ns),
                "calendar_counts": dict(
                    pool.calendar.reservation_count_by_resource),
                "calendar_bytes": dict(
                    pool.calendar.reservation_bytes_by_resource),
            }

        self.assertEqual(run(True), run(False))

    def test_sweep_pool_still_checks_final_drain(self):
        pool = self.make_pool(validate_every_event=False)
        pool.lpddr_ledger._reservations[0]["broken"] = -1
        pool.lpddr_ledger._owner_group["broken"] = 0
        pool.advance(0)
        with self.assertRaisesRegex(
                AssertionError, "capacity ledger"):
            pool.run_until_idle()

    def test_compact_history_preserves_metrics_and_latency_endpoints(self):
        pool = self.make_pool(
            validate_every_event=False,
            retain_detailed_history=False,
        )
        request = self.request(
            1,
            input_tokens=100,
            output_tokens=5,
            hbf=50,
        )
        pool.submit(request, now_ns=0)
        self.assertEqual(pool.run_until_idle(), [request])
        self.assertIsNotNone(request.first_token_ns)
        self.assertIsNotNone(request.completion_ns)
        self.assertGreater(request.tpot_ns, 0)
        self.assertEqual(request.token_completion_ns, [])
        self.assertEqual(pool.batch_history, [])
        self.assertGreater(pool.metrics.batches, 0)
        report = pool.report()
        self.assertFalse(report["retain_detailed_history"])
        self.assertEqual(report["retained_batch_count"], 0)
        pool.assert_invariants()

    def test_compact_batch_history_can_retain_token_timestamps(self):
        pool = FullModelHBFServingPool(
            repo_root=REPO_ROOT,
            hardware=HBFServerHardware(),
            layout=HBFParallelLayout.for_key("tp4"),
            max_num_batched_tokens=64,
            max_num_seqs=8,
            max_prefill_chunk_tokens=64,
            validate_every_event=False,
            retain_detailed_history=False,
            retain_token_completion_history=True,
        )
        request = self.request(
            1,
            input_tokens=100,
            output_tokens=5,
            hbf=50,
        )

        pool.submit(request, now_ns=0)
        self.assertEqual(pool.run_until_idle(), [request])

        self.assertEqual(len(request.token_completion_ns), 5)
        self.assertEqual(request.token_completion_ns[-1], request.completion_ns)
        self.assertEqual(pool.batch_history, [])
        report = pool.report()
        self.assertFalse(report["retain_detailed_history"])
        self.assertTrue(report["retain_token_completion_history"])
        pool.assert_invariants()

    def test_pool_validation_mode_must_be_boolean(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.make_pool(validate_every_event=1)

    def test_pool_history_mode_must_be_boolean(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.make_pool(retain_detailed_history=1)
        with self.assertRaisesRegex(ValueError, "boolean"):
            FullModelHBFServingPool(
                repo_root=REPO_ROOT,
                hardware=HBFServerHardware(),
                layout=HBFParallelLayout.for_key("tp4"),
                retain_token_completion_history=1,
            )

    def test_prefill_drain_policy_values_are_validated(self):
        with self.assertRaisesRegex(
                ValueError, "prefill_drain_tail_tokens"):
            FullModelHBFServingPool(
                repo_root=REPO_ROOT,
                hardware=HBFServerHardware(),
                layout=HBFParallelLayout.for_key("tp4"),
                prefill_drain_tail_tokens=-1,
            )
        with self.assertRaisesRegex(
                ValueError, "prefill_drain_min_tokens"):
            FullModelHBFServingPool(
                repo_root=REPO_ROOT,
                hardware=HBFServerHardware(),
                layout=HBFParallelLayout.for_key("tp4"),
                prefill_drain_min_tokens=True,
            )


if __name__ == "__main__":
    unittest.main()

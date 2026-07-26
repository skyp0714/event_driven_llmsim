import dataclasses
from pathlib import Path
import random
import unittest

from serving.core.gpu_pd_latency import load_p4d4_gpu_config
from serving.core.gpu_pd_pool import (
    P4D4ServingPool,
    PDRequestState,
    PDServingRequest,
)
from serving.core.hbf_full_model_lifecycle import ResourceCalendar
from serving.core.hbf_full_model_latency import HBFModelBatchShape


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_CONFIG = (
    REPO_ROOT / "configs" / "wakekv_hbf"
    / "p4d4_gpu_server.json"
)


class P4D4ServingPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = load_p4d4_gpu_config(GPU_CONFIG)

    def make_pool(
            self, *, node_id=0, calendar=None,
            max_tokens=256, max_seqs=16, chunk=64,
            p_max_seqs=None, d_max_seqs=None,
            validate_every_event=True,
            retain_detailed_history=True):
        return P4D4ServingPool(
            repo_root=REPO_ROOT,
            hardware=self.hardware,
            node_id=node_id,
            resource_calendar=calendar,
            max_num_batched_tokens=max_tokens,
            max_num_seqs=max_seqs,
            p_max_num_seqs=p_max_seqs,
            d_max_num_seqs=d_max_seqs,
            max_prefill_chunk_tokens=chunk,
            validate_every_event=validate_every_event,
            retain_detailed_history=retain_detailed_history,
        )

    @staticmethod
    def request(
            request_id, *, session_id=None, arrival=0,
            input_tokens=100, output_tokens=4,
            p_prefix=90, d_prefix=90, has_successor=True,
            restore_layer_ready_ns=()):
        return PDServingRequest(
            request_id=request_id,
            session_id=session_id or f"s-{request_id}",
            arrival_ns=arrival,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            p_prefix_tokens=p_prefix,
            d_prefix_tokens=d_prefix,
            has_successor=has_successor,
            restore_layer_ready_ns=restore_layer_ready_ns,
        )

    def test_output_one_completes_at_ttft_while_handoff_continues(self):
        pool = self.make_pool()
        request = self.request(
            1,
            input_tokens=1_000,
            output_tokens=1,
            p_prefix=999,
            d_prefix=0,
        )
        pool.submit(request, now_ns=0)
        first_token_ns = pool.batch_history[0].completion_ns
        pool.advance(first_token_ns)
        self.assertEqual(pool.pop_completed(), [request])
        self.assertEqual(request.state, PDRequestState.COMPLETE)
        self.assertEqual(request.completion_ns, request.first_token_ns)
        self.assertEqual(request.handoff_start_ns, request.first_token_ns)
        self.assertGreater(
            request.handoff_completion_ns,
            request.first_token_ns,
        )
        self.assertFalse(request.handoff_done)
        self.assertIsNone(request.tpot_ns)
        self.assertEqual(pool.metrics.d_batches, 0)
        self.assertEqual(pool.run_until_idle(), [])
        self.assertTrue(request.handoff_done)
        self.assertEqual(pool.pop_handoff_completed(), [request])
        self.assertEqual(pool.pop_handoff_completed(), [])

    def test_terminal_output_one_skips_unneeded_lineage_handoff(self):
        pool = self.make_pool()
        request = self.request(
            1,
            input_tokens=1_000,
            output_tokens=1,
            p_prefix=999,
            d_prefix=0,
            has_successor=False,
        )
        pool.submit(request, now_ns=0)
        self.assertEqual(pool.run_until_idle(), [request])
        self.assertEqual(request.completion_ns, request.first_token_ns)
        self.assertIsNone(request.handoff_start_ns)
        self.assertEqual(pool.handoff_history, [])

    def test_output_two_first_decode_context_is_exact_input(self):
        pool = self.make_pool()
        request = self.request(
            1,
            input_tokens=100,
            output_tokens=2,
            p_prefix=99,
            d_prefix=90,
        )
        pool.submit(request, now_ns=0)
        pool.run_until_idle()
        d_batches = [
            batch for batch in pool.batch_history if batch.stage == "d"
        ]
        self.assertEqual(len(d_batches), 1)
        self.assertEqual(d_batches[0].shape.decode_k, (100,))
        self.assertGreaterEqual(
            d_batches[0].start_ns,
            request.handoff_completion_ns,
        )
        self.assertEqual(request.generated_tokens, 2)
        self.assertEqual(request.final_materialized_kv_tokens, 101)

    def test_output_five_decode_contexts_and_final_kv(self):
        pool = self.make_pool()
        request = self.request(
            1,
            input_tokens=1_000,
            output_tokens=5,
            p_prefix=999,
            d_prefix=1_000,
        )
        pool.submit(request, now_ns=0)
        pool.run_until_idle()
        contexts = [
            batch.shape.decode_k[0]
            for batch in pool.batch_history
            if batch.stage == "d"
        ]
        self.assertEqual(contexts, [1_000, 1_001, 1_002, 1_003])
        self.assertEqual(request.generated_tokens, 5)
        self.assertEqual(len(request.token_completion_ns), 5)
        self.assertEqual(request.final_materialized_kv_tokens, 1_004)

    def test_full_prefix_hit_still_runs_one_p_query_token(self):
        pool = self.make_pool()
        request = self.request(
            1,
            input_tokens=100,
            output_tokens=2,
            p_prefix=100,
            d_prefix=100,
        )
        pool.submit(request, now_ns=0)
        self.assertEqual(
            request.operational_p_prefix_tokens, 99)
        self.assertEqual(request.p_fresh_tokens, 1)
        self.assertEqual(pool.batch_history[0].shape.prefill_q, (1,))
        pool.run_until_idle()
        self.assertEqual(request.p_processed_tokens, 1)
        self.assertEqual(request.handoff_tokens, 0)
        self.assertEqual(pool.metrics.zero_byte_handoffs, 1)

    def test_layer_ready_vector_overlaps_restore_with_one_p_batch(self):
        pool = self.make_pool()
        shape = HBFModelBatchShape(
            total_tokens=1,
            prefill_q=(1,),
            prefill_hbf_k=(99,),
            prefill_lpddr_k=(0,),
            lm_head_sequences=1,
        )
        phases = pool.model.batch_phase_latency(shape)
        target_start_ns = 1_000_000
        layer_ready_ns = tuple(
            target_start_ns
            + phases.layer_start_offset_ns(layer_index)
            for layer_index in range(phases.layer_count)
        )
        request = self.request(
            1,
            input_tokens=100,
            output_tokens=1,
            p_prefix=99,
            d_prefix=0,
            has_successor=False,
            restore_layer_ready_ns=layer_ready_ns,
        )

        pool.submit(request, now_ns=0)
        batch = pool.batch_history[0]

        self.assertEqual(batch.restore_gate_ns, target_start_ns)
        self.assertEqual(batch.start_ns, target_start_ns)
        self.assertEqual(
            batch.completion_ns,
            target_start_ns + batch.latency.total_ns,
        )
        self.assertLess(batch.start_ns, layer_ready_ns[-1])
        self.assertEqual(pool.metrics.streaming_restore_requests, 1)
        self.assertEqual(pool.metrics.p_streaming_batches, 1)
        self.assertEqual(
            pool.metrics.p_restore_gate_delay_ns,
            target_start_ns,
        )
        pool.run_until_idle()

    def test_layer_ready_vector_validation_is_strict(self):
        with self.assertRaisesRegex(ValueError, "one ready timestamp"):
            self.request(
                1,
                restore_layer_ready_ns=(1, 2),
            ).validate()
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            self.request(
                2,
                restore_layer_ready_ns=tuple(
                    [10] * 47 + [9]),
            ).validate()

    def test_chunked_prefill_emits_token_only_on_final_chunk(self):
        pool = self.make_pool(max_tokens=32, chunk=16)
        request = self.request(
            1,
            input_tokens=100,
            output_tokens=3,
            p_prefix=50,
            d_prefix=50,
        )
        pool.submit(request, now_ns=0)
        first_p_done = pool.batch_history[0].completion_ns
        pool.advance(first_p_done)
        self.assertEqual(request.generated_tokens, 0)
        self.assertIsNone(request.first_token_ns)
        pool.run_until_idle()
        p_chunks = [
            item.query_tokens
            for batch in pool.batch_history
            if batch.stage == "p"
            for item in batch.items
        ]
        self.assertEqual(p_chunks, [16, 16, 16, 2])
        self.assertEqual(request.p_processed_tokens, 50)
        self.assertEqual(request.generated_tokens, 3)

    def test_same_timestamp_submissions_form_one_p_batch(self):
        pool = self.make_pool()
        requests = [
            self.request(index)
            for index in range(1, 5)
        ]
        pool.submit_many(requests, now_ns=0)
        self.assertEqual(pool.batch_history[0].stage, "p")
        self.assertEqual(len(pool.batch_history[0].items), 4)
        pool.run_until_idle()
        self.assertEqual(pool.metrics.completed_requests, 4)
        self.assertGreater(pool.metrics.max_d_batch_size, 1)

    def test_documented_stage_caps_apply_to_co_timed_work(self):
        calendar = ResourceCalendar()
        calendar.reserve_parallel(
            arrival_ns=0,
            job_id=999,
            kind="external-d-work",
            namespace="test",
            demands={
                "gpu-node-0-d-model": (1_000_000_000_000, 0),
            },
        )
        pool = self.make_pool(
            calendar=calendar,
            max_tokens=131_072,
            max_seqs=128,
            p_max_seqs=32,
            d_max_seqs=128,
            chunk=131_072,
        )
        requests = [
            self.request(
                request_id,
                input_tokens=10,
                output_tokens=2,
                p_prefix=10,
                d_prefix=10,
                has_successor=False,
            )
            for request_id in range(1, 131)
        ]
        pool.submit_many(requests, now_ns=0)
        self.assertEqual(len(pool.batch_history[0].items), 32)

        pool.advance(1_000_000_000_000)
        p_batches = [
            batch for batch in pool.batch_history if batch.stage == "p"
        ]
        d_batches = [
            batch for batch in pool.batch_history if batch.stage == "d"
        ]
        self.assertEqual([len(batch.items) for batch in p_batches],
                         [32, 32, 32, 32, 2])
        self.assertEqual(len(d_batches[0].items), 128)
        pool.run_until_idle()
        self.assertEqual(pool.metrics.max_p_batch_size, 32)
        self.assertEqual(pool.metrics.max_d_batch_size, 128)

    def test_shared_sequence_limit_remains_backward_compatible(self):
        pool = self.make_pool(max_seqs=7)
        self.assertEqual(pool.max_num_seqs, 7)
        self.assertEqual(pool.p_max_num_seqs, 7)
        self.assertEqual(pool.d_max_num_seqs, 7)
        report = pool.report()
        self.assertEqual(report["max_num_seqs"], 7)
        self.assertEqual(report["p_max_num_seqs"], 7)
        self.assertEqual(report["d_max_num_seqs"], 7)

    def test_invalid_stage_sequence_limits_are_rejected(self):
        invalid_values = (0, True, 1.5, 129)
        for stage_name in ("p_max_seqs", "d_max_seqs"):
            for value in invalid_values:
                with self.subTest(stage=stage_name, value=value):
                    with self.assertRaisesRegex(
                            ValueError, "max_num_seqs"):
                        self.make_pool(**{stage_name: value})

    def test_131072_token_budget_and_chunk_are_supported(self):
        pool = self.make_pool(
            max_tokens=131_072,
            max_seqs=32,
            chunk=131_072,
        )
        request = self.request(
            1,
            input_tokens=131_072,
            output_tokens=1,
            p_prefix=0,
            d_prefix=0,
            has_successor=False,
        )
        pool.submit(request, now_ns=0)
        self.assertEqual(
            pool.batch_history[0].shape.prefill_q,
            (131_072,),
        )
        self.assertEqual(pool.run_until_idle(), [request])

    def test_zero_byte_handoff_completions_coalesce_d_batch(self):
        pool = self.make_pool()
        requests = [
            self.request(
                index,
                input_tokens=100,
                output_tokens=2,
                p_prefix=99,
                d_prefix=100,
            )
            for index in (1, 2)
        ]
        pool.submit_many(requests, now_ns=0)
        pool.run_until_idle()
        d_batches = [
            batch for batch in pool.batch_history if batch.stage == "d"
        ]
        self.assertEqual(len(d_batches), 1)
        self.assertEqual(
            [item.request_id for item in d_batches[0].items],
            [1, 2],
        )

    def test_p_and_d_compute_overlap_on_disjoint_gpu_groups(self):
        pool = self.make_pool()
        decode_request = self.request(
            1,
            input_tokens=100,
            output_tokens=3,
            p_prefix=99,
            d_prefix=100,
        )
        pool.submit(decode_request, now_ns=0)
        first_token_ns = pool.batch_history[0].completion_ns
        pool.advance(first_token_ns, defer_schedule=True)
        newcomer = self.request(
            2,
            arrival=first_token_ns,
            input_tokens=120,
            output_tokens=1,
            p_prefix=100,
            d_prefix=0,
        )
        pool.submit(newcomer, now_ns=first_token_ns)
        launched = pool.batch_history[-2:]
        self.assertEqual({batch.stage for batch in launched}, {"p", "d"})
        self.assertEqual(launched[0].start_ns, launched[1].start_ns)
        self.assertGreater(
            min(batch.completion_ns for batch in launched),
            launched[0].start_ns,
        )
        pool.run_until_idle()

    def test_handoffs_serialize_without_changing_shared_ttft(self):
        pool = self.make_pool()
        requests = [
            self.request(
                index,
                input_tokens=1_000,
                output_tokens=2,
                p_prefix=999,
                d_prefix=0,
            )
            for index in (1, 2)
        ]
        pool.submit_many(requests, now_ns=0)
        first_token_ns = pool.batch_history[0].completion_ns
        pool.advance(first_token_ns)
        first, second = pool.handoff_history
        self.assertEqual(requests[0].first_token_ns, first_token_ns)
        self.assertEqual(requests[1].first_token_ns, first_token_ns)
        self.assertEqual(first.start_ns, first_token_ns)
        self.assertEqual(second.start_ns, first.completion_ns)
        self.assertGreater(second.completion_ns, first.completion_ns)
        pool.run_until_idle()

    def test_handoff_size_does_not_change_ttft(self):
        full_copy_pool = self.make_pool()
        zero_copy_pool = self.make_pool()
        full_copy = self.request(
            1,
            input_tokens=1_000,
            output_tokens=2,
            p_prefix=999,
            d_prefix=0,
        )
        zero_copy = self.request(
            1,
            input_tokens=1_000,
            output_tokens=2,
            p_prefix=999,
            d_prefix=1_000,
        )
        full_copy_pool.submit(full_copy, now_ns=0)
        zero_copy_pool.submit(zero_copy, now_ns=0)
        full_copy_pool.run_until_idle()
        zero_copy_pool.run_until_idle()
        self.assertEqual(full_copy.first_token_ns, zero_copy.first_token_ns)
        self.assertGreater(
            full_copy.completion_ns,
            zero_copy.completion_ns,
        )

    def test_submit_after_restore_preserves_logical_release_for_ttft(self):
        pool = self.make_pool()
        request = self.request(1, arrival=100)
        pool.submit(request, now_ns=1_000)
        pool.run_until_idle()
        self.assertEqual(
            request.ttft_ns,
            request.first_token_ns - 100,
        )
        self.assertGreaterEqual(request.ttft_ns, 900)
        future = self.request(2, arrival=request.completion_ns + 1)
        with self.assertRaisesRegex(ValueError, "logical arrival"):
            pool.submit(future, now_ns=request.completion_ns)

    def test_arrival_before_delayed_p_launch_joins_batch(self):
        calendar = ResourceCalendar()
        calendar.reserve_parallel(
            arrival_ns=0,
            job_id=999,
            kind="external-p-work",
            namespace="test",
            demands={
                "gpu-node-0-p-model": (1_000_000, 0),
            },
        )
        pool = self.make_pool(calendar=calendar)
        first = self.request(1, arrival=0)
        second = self.request(2, arrival=500_000)
        pool.submit(first, now_ns=0)
        pool.submit(second, now_ns=500_000)
        self.assertEqual(pool.batch_history, [])
        pool.advance(1_000_000)
        self.assertEqual(
            [item.request_id for item in pool.batch_history[0].items],
            [1, 2],
        )

    def test_two_nodes_use_disjoint_resource_namespaces(self):
        calendar = ResourceCalendar()
        node0 = self.make_pool(node_id=0, calendar=calendar)
        node1 = self.make_pool(node_id=1, calendar=calendar)
        node0.submit(self.request(1), now_ns=0)
        node1.submit(self.request(2), now_ns=0)
        self.assertEqual(
            node0.batch_history[0].start_ns,
            node1.batch_history[0].start_ns,
        )
        resources = {
            row.resource for row in calendar.reservations
            if row.kind == "p-model-batch"
        }
        self.assertEqual(resources, {
            "gpu-node-0-p-model",
            "gpu-node-1-p-model",
        })
        node0.run_until_idle()
        node1.run_until_idle()

    def test_invalid_or_reused_request_is_rejected(self):
        pool = self.make_pool()
        invalid = self.request(1)
        invalid.output_tokens = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            pool.submit(invalid, now_ns=0)
        request = self.request(2)
        pool.submit(request, now_ns=0)
        pool.run_until_idle()
        with self.assertRaisesRegex(ValueError, "pristine"):
            pool.submit(request, now_ns=request.completion_ns)

    def test_invalid_submission_cannot_strand_pool_event(self):
        pool = self.make_pool()
        valid = self.request(
            1,
            input_tokens=1_000,
            p_prefix=0,
            d_prefix=0,
            output_tokens=1,
            has_successor=False,
        )
        pool.submit(valid, now_ns=0)
        first_completion_ns = pool.next_event_ns()
        invalid = self.request(2, arrival=first_completion_ns)
        invalid.output_tokens = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            pool.submit(invalid, now_ns=first_completion_ns)
        self.assertEqual(pool.current_ns, 0)
        self.assertEqual(pool.next_event_ns(), first_completion_ns)
        self.assertEqual(pool.run_until_idle(), [valid])

    def test_sweep_mode_still_validates_at_full_drain(self):
        pool = self.make_pool(validate_every_event=False)
        requests = [
            self.request(
                request_id,
                input_tokens=100 + request_id,
                output_tokens=3,
                p_prefix=50,
                d_prefix=50,
                has_successor=False,
            )
            for request_id in range(1, 5)
        ]
        pool.submit_many(requests, now_ns=0)
        self.assertEqual(len(pool.run_until_idle()), 4)
        self.assertFalse(pool.validate_every_event)
        pool.assert_invariants()

    def test_compact_history_preserves_metrics_and_latency_endpoints(self):
        pool = self.make_pool(
            validate_every_event=False,
            retain_detailed_history=False,
        )
        request = self.request(
            1,
            input_tokens=80,
            output_tokens=5,
            p_prefix=40,
            d_prefix=40,
            has_successor=False,
        )
        pool.submit(request, now_ns=0)
        self.assertEqual(pool.run_until_idle(), [request])
        self.assertIsNotNone(request.first_token_ns)
        self.assertIsNotNone(request.completion_ns)
        self.assertGreater(request.tpot_ns, 0)
        self.assertEqual(request.token_completion_ns, [])
        self.assertEqual(pool.batch_history, [])
        self.assertEqual(pool.handoff_history, [])
        self.assertEqual(pool._handoff_jobs, {})
        self.assertGreater(pool.metrics.p_batches, 0)
        self.assertGreater(pool.metrics.d_batches, 0)
        report = pool.report()
        self.assertFalse(report["retain_detailed_history"])
        self.assertEqual(report["retained_batch_count"], 0)
        self.assertEqual(report["live_handoff_job_count"], 0)
        pool.assert_invariants()

    def test_history_mode_must_be_boolean(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.make_pool(retain_detailed_history=1)

    def test_hardware_is_not_mutated_by_pool(self):
        before = dataclasses.asdict(self.hardware)
        pool = self.make_pool()
        pool.submit(self.request(1), now_ns=0)
        pool.run_until_idle()
        self.assertEqual(dataclasses.asdict(self.hardware), before)

    def test_randomized_batching_conserves_all_query_tokens(self):
        rng = random.Random(17)
        pool = self.make_pool(
            max_tokens=512,
            max_seqs=32,
            chunk=128,
        )
        requests = []
        for request_id in range(1, 201):
            input_tokens = rng.randint(1, 1_000)
            requests.append(self.request(
                request_id,
                input_tokens=input_tokens,
                output_tokens=rng.choice((1, 2, 5)),
                p_prefix=rng.randint(0, input_tokens),
                d_prefix=rng.randint(0, input_tokens),
                has_successor=bool(rng.getrandbits(1)),
            ))
        pool.submit_many(requests, now_ns=0)
        completed = pool.run_until_idle()
        self.assertEqual(
            {request.request_id for request in completed},
            set(range(1, 201)),
        )
        self.assertEqual(
            pool.metrics.p_query_tokens,
            sum(request.p_fresh_tokens for request in requests),
        )
        self.assertEqual(
            pool.metrics.d_query_tokens,
            sum(request.output_tokens - 1 for request in requests),
        )
        self.assertEqual(
            pool.metrics.handoff_jobs,
            sum(
                request.output_tokens > 1 or request.has_successor
                for request in requests
            ),
        )
        self.assertTrue(all(
            request.generated_tokens == request.output_tokens
            and request.p_processed_tokens == request.p_fresh_tokens
            and request.final_materialized_kv_tokens
            == request.input_tokens + request.output_tokens - 1
            for request in requests
        ))
        self.assertTrue(all(
            len(batch.items) <= 32
            and batch.shape.total_tokens <= 512
            for batch in pool.batch_history
        ))
        for resource in (
            "gpu-node-0-p-model",
            "gpu-node-0-d-model",
            "gpu-node-0-pd-fabric",
        ):
            rows = sorted(
                (
                    row for row in pool.calendar.reservations
                    if row.resource == resource
                ),
                key=lambda row: row.start_ns,
            )
            self.assertTrue(all(
                current.end_ns <= following.start_ns
                for current, following in zip(rows, rows[1:])
            ))
        pool.assert_invariants()


if __name__ == "__main__":
    unittest.main()

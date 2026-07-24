import heapq
from pathlib import Path
import random
import unittest

from serving.core.gpu_pd_latency import load_p4d4_gpu_config
from serving.core.gpu_pd_oracle_node import (
    OracleCallState,
    OracleNodeCall,
    StrictInfiniteHBMNode,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_CONFIG = (
    REPO_ROOT / "configs" / "wakekv_hbf"
    / "p4d4_gpu_server.json"
)


class StrictInfiniteHBMNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = load_p4d4_gpu_config(GPU_CONFIG)

    def make_node(self, *, node_id=0, max_tokens=256, chunk=64):
        return StrictInfiniteHBMNode(
            repo_root=REPO_ROOT,
            hardware=self.hardware,
            node_id=node_id,
            max_num_batched_tokens=max_tokens,
            max_num_seqs=16,
            max_prefill_chunk_tokens=chunk,
        )

    def advance_until_user_completion(self, node, call):
        while call.user_completion_ns is None:
            event_ns = node.next_event_ns()
            self.assertIsNotNone(event_ns)
            node.advance(event_ns)
        self.assertEqual(node.current_ns, call.user_completion_ns)

    @staticmethod
    def call(
            request_id, *, session_id="s", call_index=0,
            release=0, input_tokens=100, output_tokens=3,
            prefix=0, has_successor=False):
        return OracleNodeCall(
            request_id=request_id,
            session_id=session_id,
            call_index=call_index,
            release_ns=release,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prefix_reuse_tokens=prefix,
            has_successor=has_successor,
        )

    def test_first_call_has_no_prepare_and_uses_full_handoff(self):
        node = self.make_node()
        call = self.call(
            1,
            input_tokens=100,
            output_tokens=3,
            has_successor=True,
        )
        node.submit(call, now_ns=0)
        self.assertEqual(node.run_until_idle(), [call])
        self.assertEqual(call.state, OracleCallState.INTERNAL_COMPLETE)
        self.assertEqual(call.operational_hit_tokens, 0)
        self.assertEqual(node.prepare_history, [])
        self.assertEqual(call.pool_request.p_fresh_tokens, 100)
        self.assertEqual(call.pool_request.handoff_tokens, 100)
        self.assertEqual(
            node.sessions["s"].materialized_tokens, 102)
        self.assertTrue(node.sessions["s"].d_resident)

    def test_resume_pays_d_to_p_and_handoffs_only_fresh_suffix(self):
        node = self.make_node()
        first = self.call(
            1,
            input_tokens=100,
            output_tokens=2,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        node.run_until_idle()
        release = first.user_completion_ns + 1_000
        resume = self.call(
            2,
            call_index=1,
            release=release,
            input_tokens=110,
            output_tokens=2,
            prefix=101,
            has_successor=False,
        )
        node.submit(resume, now_ns=release)
        self.assertEqual(len(node.prepare_history), 1)
        prepare = node.prepare_history[0]
        self.assertEqual(prepare.hit_tokens, 101)
        self.assertGreater(prepare.completion_ns, release)
        node.run_until_idle()
        self.assertEqual(resume.operational_hit_tokens, 101)
        self.assertEqual(resume.pool_request.p_fresh_tokens, 9)
        self.assertEqual(resume.pool_request.handoff_tokens, 9)
        resume_p_batches = [
            batch for batch in node.pool.batch_history
            if any(
                item.request_id == resume.request_id
                for item in batch.items
            ) and batch.stage == "p"
        ]
        self.assertGreaterEqual(
            resume_p_batches[0].start_ns,
            prepare.completion_ns,
        )
        self.assertTrue(node.sessions["s"].ended)
        self.assertEqual(node.hbm.p_used_bytes_per_rank, 0)
        self.assertEqual(node.hbm.d_used_bytes_per_rank, 0)

    def test_full_prefix_is_capped_to_leave_one_p_token(self):
        node = self.make_node()
        first = self.call(
            1,
            input_tokens=100,
            output_tokens=1,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        self.advance_until_user_completion(node, first)
        release = first.user_completion_ns
        resume = self.call(
            2,
            call_index=1,
            release=release,
            input_tokens=100,
            output_tokens=1,
            prefix=100,
            has_successor=False,
        )
        node.submit(resume, now_ns=release)
        node.run_until_idle()
        self.assertEqual(resume.operational_hit_tokens, 99)
        self.assertEqual(resume.pool_request.p_fresh_tokens, 1)
        self.assertEqual(resume.pool_request.handoff_tokens, 1)
        self.assertEqual(node.metrics.full_prefix_cap_calls, 1)

    def test_context_shrink_does_not_restore_old_prefix(self):
        node = self.make_node()
        first = self.call(
            1,
            input_tokens=100,
            output_tokens=2,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        node.run_until_idle()
        release = first.user_completion_ns
        resume = self.call(
            2,
            call_index=1,
            release=release,
            input_tokens=40,
            output_tokens=1,
            prefix=0,
            has_successor=False,
        )
        node.submit(resume, now_ns=release)
        node.run_until_idle()
        self.assertEqual(resume.operational_hit_tokens, 0)
        self.assertEqual(resume.pool_request.p_fresh_tokens, 40)
        self.assertEqual(resume.pool_request.d_prefix_tokens, 0)
        self.assertEqual(node.metrics.context_shrink_calls, 1)
        self.assertEqual(node.metrics.d_to_p_jobs, 0)

    def test_resume_ownership_transitions_are_exact(self):
        node = self.make_node()
        first = self.call(
            1,
            input_tokens=100,
            output_tokens=2,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        node.run_until_idle()
        prior_d_bytes = node.hbm.d_bytes("s")
        release_ns = first.user_completion_ns
        resume = self.call(
            2,
            call_index=1,
            release=release_ns,
            input_tokens=80,
            output_tokens=2,
            prefix=70,
            has_successor=False,
        )
        node.submit(resume, now_ns=release_ns)
        admission = node.hbm.active_admission("s")
        self.assertIsNotNone(admission)
        self.assertEqual(
            node.hbm.p_bytes("s"),
            admission.p_bytes_per_rank,
        )
        self.assertEqual(node.hbm.d_bytes("s"), prior_d_bytes)
        self.assertGreater(
            prior_d_bytes,
            admission.d_target_bytes_per_rank,
        )
        self.assertEqual(node.metrics.context_shrink_calls, 1)

        node.advance(resume.prepare_completion_ns)
        self.assertEqual(
            node.hbm.d_bytes("s"),
            admission.d_target_bytes_per_rank,
        )
        self.assertEqual(
            node.hbm.p_bytes("s"),
            admission.p_bytes_per_rank,
        )
        while resume.pool_request.handoff_completion_ns is None:
            event_ns = node.next_event_ns()
            self.assertIsNotNone(event_ns)
            node.advance(event_ns)
        node.advance(resume.pool_request.handoff_completion_ns)
        self.assertEqual(node.hbm.p_bytes("s"), 0)
        self.assertEqual(
            node.hbm.d_bytes("s"),
            admission.d_target_bytes_per_rank,
        )
        node.run_until_idle()
        self.assertEqual(node.hbm.p_bytes("s"), 0)
        self.assertEqual(node.hbm.d_bytes("s"), 0)

    def test_resume_ttft_includes_physical_prepare_copy(self):
        node = self.make_node()
        first = self.call(
            1,
            input_tokens=1_000,
            output_tokens=2,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        node.run_until_idle()
        release = first.user_completion_ns
        resume = self.call(
            2,
            call_index=1,
            release=release,
            input_tokens=1_010,
            output_tokens=1,
            prefix=1_001,
            has_successor=False,
        )
        node.submit(resume, now_ns=release)
        prepare = node.prepare_history[-1]
        node.run_until_idle()
        self.assertEqual(resume.prepare_start_ns, release)
        self.assertEqual(
            resume.prepare_completion_ns - resume.prepare_start_ns,
            prepare.stage.latency_ns,
        )
        self.assertGreater(
            resume.ttft_ns,
            prepare.stage.latency_ns,
        )

    def test_zero_gap_successor_waits_for_output_one_lineage_tail(self):
        node = self.make_node()
        first = self.call(
            1,
            input_tokens=1_000,
            output_tokens=1,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        self.advance_until_user_completion(node, first)
        first_token_ns = first.user_completion_ns
        self.assertEqual(node.pop_completed(), [first])
        self.assertEqual(first.state, OracleCallState.USER_COMPLETE)
        self.assertGreater(
            first.pool_request.handoff_completion_ns,
            first_token_ns,
        )
        resume = self.call(
            2,
            call_index=1,
            release=first_token_ns,
            input_tokens=1_001,
            output_tokens=1,
            prefix=1_000,
            has_successor=False,
        )
        node.submit(resume, now_ns=first_token_ns)
        self.assertEqual(resume.state, OracleCallState.PENDING)
        self.assertIsNone(resume.admission_id)
        lineage_ready = first.pool_request.handoff_completion_ns
        node.advance(lineage_ready)
        self.assertEqual(first.state, OracleCallState.INTERNAL_COMPLETE)
        self.assertNotEqual(resume.state, OracleCallState.PENDING)
        node.run_until_idle()

    def test_lineage_blocked_head_does_not_block_another_session(self):
        node = self.make_node()
        first = self.call(
            1,
            input_tokens=1_000,
            output_tokens=1,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        self.advance_until_user_completion(node, first)
        now_ns = first.user_completion_ns
        node.pop_completed()
        blocked_resume = self.call(
            2,
            call_index=1,
            release=now_ns,
            input_tokens=1_001,
            output_tokens=1,
            prefix=1_000,
            has_successor=False,
        )
        independent = self.call(
            3,
            session_id="other",
            release=now_ns,
            input_tokens=32,
            output_tokens=1,
            has_successor=False,
        )
        node.submit_many(
            (blocked_resume, independent),
            now_ns=now_ns,
        )
        self.assertEqual(
            blocked_resume.state, OracleCallState.PENDING)
        self.assertIsNone(blocked_resume.admission_id)
        self.assertEqual(
            independent.state, OracleCallState.EXECUTING)
        self.assertIsNotNone(independent.admission_id)
        node.run_until_idle()

    def test_same_timestamp_first_calls_coalesce_on_p(self):
        node = self.make_node()
        calls = [
            self.call(
                index,
                session_id=f"s-{index}",
                has_successor=False,
            )
            for index in range(1, 5)
        ]
        node.submit_many(calls, now_ns=0)
        first_batch = node.pool.batch_history[0]
        self.assertEqual(first_batch.stage, "p")
        self.assertEqual(len(first_batch.items), 4)
        node.run_until_idle()
        self.assertEqual(node.metrics.user_completed_calls, 4)

    def test_deterministic_multisession_stress_conserves_work(self):
        rng = random.Random(20260723)
        node = self.make_node(
            max_tokens=1_024,
            chunk=256,
        )
        num_sessions = 24
        calls_per_session = 5
        specs = {}
        gaps = {}
        for session_index in range(num_sessions):
            prior_final = 0
            for call_index in range(calls_per_session):
                output_tokens = rng.randint(1, 6)
                if call_index == 0:
                    input_tokens = rng.randint(64, 2_048)
                    prefix_tokens = 0
                elif call_index % 3 == 1:
                    input_tokens = prior_final + rng.randint(1, 128)
                    prefix_tokens = prior_final
                elif call_index % 3 == 2:
                    input_tokens = prior_final
                    prefix_tokens = prior_final
                else:
                    input_tokens = max(
                        1,
                        prior_final - rng.randint(1, prior_final),
                    )
                    prefix_tokens = 0
                specs[(session_index, call_index)] = (
                    input_tokens,
                    output_tokens,
                    prefix_tokens,
                )
                gaps[(session_index, call_index)] = rng.choice(
                    (0, 1, 10_000, 1_000_000))
                prior_final = input_tokens + output_tokens - 1

        arrivals = [
            (0, session_index, 0)
            for session_index in range(num_sessions)
        ]
        heapq.heapify(arrivals)
        completed_ids = []
        all_calls = {}
        expected_calls = num_sessions * calls_per_session

        while len(completed_ids) < expected_calls:
            internal_ns = node.next_event_ns()
            external_ns = arrivals[0][0] if arrivals else None
            candidates = [
                value for value in (internal_ns, external_ns)
                if value is not None
            ]
            self.assertTrue(candidates)
            now_ns = min(candidates)
            node.advance(now_ns, defer_schedule=True)
            for completed in node.pop_completed():
                completed_ids.append(completed.request_id)
                if completed.has_successor:
                    next_index = completed.call_index + 1
                    session_index = int(
                        completed.session_id.removeprefix("session-"))
                    release_ns = (
                        completed.user_completion_ns
                        + gaps[(session_index, completed.call_index)]
                    )
                    heapq.heappush(
                        arrivals,
                        (release_ns, session_index, next_index),
                    )

            ready_calls = []
            while arrivals and arrivals[0][0] == now_ns:
                release_ns, session_index, call_index = heapq.heappop(
                    arrivals)
                input_tokens, output_tokens, prefix_tokens = specs[
                    (session_index, call_index)]
                request_id = session_index * 100 + call_index
                call = self.call(
                    request_id,
                    session_id=f"session-{session_index}",
                    call_index=call_index,
                    release=release_ns,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    prefix=prefix_tokens,
                    has_successor=(
                        call_index + 1 < calls_per_session),
                )
                all_calls[request_id] = call
                ready_calls.append(call)
            if ready_calls:
                node.submit_many(ready_calls, now_ns=now_ns)
            else:
                node.flush_scheduling(now_ns)

        self.assertEqual(node.run_until_idle(), [])
        self.assertEqual(len(completed_ids), expected_calls)
        self.assertEqual(len(set(completed_ids)), expected_calls)
        self.assertEqual(node.metrics.internal_completed_calls, expected_calls)
        self.assertTrue(all(
            call.state == OracleCallState.INTERNAL_COMPLETE
            for call in all_calls.values()
        ))
        expected_p_queries = sum(
            call.input_tokens - call.operational_hit_tokens
            for call in all_calls.values()
        )
        actual_p_queries = sum(
            item.query_tokens
            for batch in node.pool.batch_history
            if batch.stage == "p"
            for item in batch.items
        )
        expected_d_queries = sum(
            call.output_tokens - 1
            for call in all_calls.values()
        )
        actual_d_queries = sum(
            item.query_tokens
            for batch in node.pool.batch_history
            if batch.stage == "d"
            for item in batch.items
        )
        self.assertEqual(actual_p_queries, expected_p_queries)
        self.assertEqual(actual_d_queries, expected_d_queries)
        self.assertTrue(all(
            placement.ended
            for placement in node.sessions.values()
        ))
        self.assertEqual(node.hbm.p_used_bytes_per_rank, 0)
        self.assertEqual(node.hbm.d_used_bytes_per_rank, 0)

    def test_oracle_uses_no_cpu_ssd_or_pcie_resources(self):
        node = self.make_node()
        first = self.call(1, output_tokens=2, has_successor=True)
        node.submit(first, now_ns=0)
        node.run_until_idle()
        resume = self.call(
            2,
            call_index=1,
            release=first.user_completion_ns,
            input_tokens=110,
            output_tokens=2,
            prefix=101,
            has_successor=False,
        )
        node.submit(resume, now_ns=resume.release_ns)
        node.run_until_idle()
        resources = {
            row.resource for row in node.calendar.reservations
        }
        self.assertFalse(any(
            "ssd" in resource
            or "cpu" in resource
            or "pcie" in resource
            for resource in resources
        ))
        self.assertEqual(node.hbm.metrics.capacity_deferrals, 0)

    def test_call_order_and_first_prefix_are_validated(self):
        node = self.make_node()
        with self.assertRaisesRegex(ValueError, "first call"):
            node.submit(self.call(1, prefix=1), now_ns=0)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            node.submit(
                self.call(2, call_index=1),
                now_ns=0,
            )
        with self.assertRaisesRegex(ValueError, "terminal predecessor"):
            node.submit_many(
                (
                    self.call(
                        3,
                        session_id="terminal",
                        has_successor=False,
                    ),
                    self.call(
                        4,
                        session_id="terminal",
                        call_index=1,
                        has_successor=False,
                    ),
                ),
                now_ns=0,
            )

    def test_successor_cannot_be_injected_before_user_completion(self):
        node = self.make_node()
        first = self.call(1, has_successor=True)
        node.submit(first, now_ns=0)
        successor = self.call(
            2,
            call_index=1,
            release=0,
            has_successor=False,
        )
        with self.assertRaisesRegex(ValueError, "user-complete"):
            node.submit(successor, now_ns=0)
        self.assertEqual(node.metrics.submitted_calls, 1)
        self.assertEqual(node.run_until_idle(), [first])

    def test_invalid_submission_cannot_strand_deferred_work(self):
        node = self.make_node()
        valid = self.call(
            1,
            input_tokens=1_000,
            output_tokens=1,
            has_successor=False,
        )
        node.submit(valid, now_ns=0)
        first_completion_ns = node.next_event_ns()
        self.assertIsNotNone(first_completion_ns)
        invalid = self.call(
            2,
            session_id="invalid",
            release=first_completion_ns,
            prefix=1,
        )
        with self.assertRaisesRegex(ValueError, "first call"):
            node.submit(invalid, now_ns=first_completion_ns)
        self.assertEqual(node.current_ns, 0)
        self.assertEqual(node.next_event_ns(), first_completion_ns)
        self.assertEqual(node.run_until_idle(), [valid])

    def test_report_names_strict_nonmirrored_oracle(self):
        node = self.make_node(node_id=3)
        report = node.report()
        self.assertEqual(
            report["mode"],
            "strict_infinite_hbm_residency_oracle",
        )
        self.assertIn(
            "D-to-P copy",
            report["physical_resume_contract"],
        )
        self.assertEqual(report["node_id"], 3)


if __name__ == "__main__":
    unittest.main()

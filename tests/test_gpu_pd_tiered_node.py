import heapq
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from serving.core.gpu_pd_latency import load_p4d4_gpu_config
from serving.core.gpu_pd_tier_lifecycle import (
    Tier,
    TierJobStatus,
    TierSessionState,
)
from serving.core.gpu_pd_tiered_node import (
    FiniteHBMTieredP4D4Node,
    TieredCallState,
    TieredNodeCall,
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


class FiniteHBMTieredP4D4NodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = load_p4d4_gpu_config(GPU_CONFIG)
        cls.block_per_rank = (
            cls.hardware.kv_capacity_bytes_per_rank(16))
        cls.block_aggregate = (
            cls.block_per_rank * cls.hardware.tp_size)

    def make_node(
            self, policy, *, node_id=0,
            p_blocks=128, d_blocks=64,
            cpu_blocks=128, ssd_blocks=256,
            max_tokens=512, chunk=128,
            validate_every_event=True):
        return FiniteHBMTieredP4D4Node(
            repo_root=REPO_ROOT,
            hardware=self.hardware,
            node_id=node_id,
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
            validate_every_event=validate_every_event,
        )

    @staticmethod
    def call(
            request_id, *, session_id="s", call_index=0,
            release=0, input_tokens=100, output_tokens=3,
            prefix=0, has_successor=False):
        return TieredNodeCall(
            request_id=request_id,
            session_id=session_id,
            call_index=call_index,
            release_ns=release,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prefix_reuse_tokens=prefix,
            has_successor=has_successor,
        )

    def advance_until_user_completion(self, node, call):
        while call.user_completion_ns is None:
            event_ns = node.next_event_ns()
            self.assertIsNotNone(event_ns)
            node.advance(event_ns)

    def test_first_call_and_stable_d_resume_use_exact_handoffs(self):
        node = self.make_node("cpu_ssd")
        first = self.call(
            0,
            input_tokens=100,
            output_tokens=2,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        self.assertEqual(node.run_until_idle(), [first])
        self.assertEqual(first.prepare_source, None)
        self.assertEqual(first.operational_hit_tokens, 0)
        self.assertEqual(first.pool_request.d_prefix_tokens, 0)
        self.assertEqual(first.pool_request.handoff_tokens, 100)
        self.assertEqual(
            node.lifecycle.sessions["s"].state,
            TierSessionState.D_READY,
        )
        self.assertEqual(node.lifecycle.p_ledger.used_bytes, 0)

        release_ns = first.user_completion_ns + 1
        resume = self.call(
            1,
            call_index=1,
            release=release_ns,
            input_tokens=110,
            output_tokens=2,
            prefix=101,
            has_successor=False,
        )
        node.submit(resume, now_ns=release_ns)
        ticket = node._ticket_by_request[resume.request_id]
        self.assertEqual(ticket.source, Tier.D)
        self.assertFalse(ticket.full_d_reservation)
        node.run_until_idle()
        self.assertEqual(resume.operational_hit_tokens, 101)
        self.assertEqual(resume.pool_request.d_prefix_tokens, 101)
        self.assertEqual(resume.pool_request.handoff_tokens, 9)
        self.assertEqual(node.metrics.stable_d_hits, 1)
        self.assertEqual(node.metrics.fresh_suffix_handoffs, 1)
        self.assertEqual(node.lifecycle.p_ledger.used_bytes, 0)
        self.assertEqual(node.lifecycle.d_ledger.used_bytes, 0)

    def test_policy_specific_lower_tier_resume_or_recompute(self):
        expected = {
            "hbm_lru_recompute": (
                None,
                (),
                TierSessionState.LOST,
            ),
            "ssd_direct": (
                Tier.SSD,
                ("ssd-to-cpu", "p-cpu_to_gpu"),
                TierSessionState.SSD_READY,
            ),
            "cpu_ssd": (
                Tier.CPU,
                ("p-cpu_to_gpu",),
                TierSessionState.CPU_READY,
            ),
        }
        for policy in POLICIES:
            with self.subTest(policy=policy):
                node = self.make_node(policy)
                first = self.call(
                    0,
                    input_tokens=64,
                    output_tokens=2,
                    has_successor=True,
                )
                node.submit(first, now_ns=0)
                node.run_until_idle()
                demotion = node.lifecycle.demote(
                    "s", now_ns=node.current_ns)
                if demotion is not None:
                    node.advance(demotion.completion_ns)
                    self.assertEqual(
                        demotion.status, TierJobStatus.COMMITTED)
                source, stages, state = expected[policy]
                self.assertEqual(
                    node.lifecycle.sessions["s"].state, state)

                release_ns = node.current_ns + 1
                resume = self.call(
                    1,
                    call_index=1,
                    release=release_ns,
                    input_tokens=70,
                    output_tokens=2,
                    prefix=65,
                    has_successor=False,
                )
                node.submit(resume, now_ns=release_ns)
                ticket = node._ticket_by_request[1]
                self.assertEqual(ticket.source, source)
                self.assertEqual(ticket.transfer_kinds, stages)
                node.run_until_idle()
                self.assertEqual(
                    resume.pool_request.d_prefix_tokens, 0)
                self.assertEqual(
                    resume.pool_request.handoff_tokens, 70)
                if policy == "hbm_lru_recompute":
                    self.assertEqual(
                        resume.operational_hit_tokens, 0)
                    self.assertEqual(
                        node.metrics.recompute_resumes, 1)
                else:
                    self.assertEqual(
                        resume.operational_hit_tokens, 65)
                    self.assertEqual(
                        node.metrics.lower_tier_hits, 1)
                self.assertEqual(
                    node.lifecycle.p_ledger.used_bytes, 0)
                self.assertEqual(
                    node.lifecycle.d_ledger.used_bytes, 0)
                self.assertEqual(
                    node.lifecycle.cpu_ledger.used_bytes, 0)
                self.assertEqual(
                    node.lifecycle.ssd_ledger.used_bytes, 0)

    def test_resume_racing_demotion_uses_full_new_d_destination(self):
        node = self.make_node(
            "ssd_direct",
            p_blocks=64,
            d_blocks=64,
            cpu_blocks=64,
            ssd_blocks=128,
        )
        first = self.call(
            0,
            input_tokens=128,
            output_tokens=2,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        node.run_until_idle()
        demotion = node.lifecycle.demote(
            "s", now_ns=node.current_ns)
        old_source_id = demotion.source_copy_id
        release_ns = node.current_ns + 1
        self.assertLess(release_ns, demotion.completion_ns)
        resume = self.call(
            1,
            call_index=1,
            release=release_ns,
            input_tokens=140,
            output_tokens=2,
            prefix=129,
            has_successor=True,
        )
        node.submit(resume, now_ns=release_ns)
        ticket = node._ticket_by_request[1]
        self.assertEqual(ticket.source, Tier.D)
        self.assertTrue(ticket.full_d_reservation)
        node.run_until_idle()
        self.assertEqual(resume.operational_hit_tokens, 129)
        self.assertEqual(resume.pool_request.d_prefix_tokens, 0)
        self.assertEqual(resume.pool_request.handoff_tokens, 140)
        self.assertEqual(demotion.status, TierJobStatus.STALE)
        self.assertGreater(
            node.lifecycle.metrics.stale_transfer_bytes, 0)
        record = node.lifecycle.sessions["s"]
        self.assertEqual(record.state, TierSessionState.D_READY)
        self.assertNotEqual(record.primary_copy_id, old_source_id)
        self.assertEqual(record.tokens, 141)
        self.assertNotIn(old_source_id, node.lifecycle.copies)

        final = self.call(
            2,
            call_index=2,
            release=node.current_ns,
            input_tokens=142,
            output_tokens=1,
            prefix=141,
            has_successor=False,
        )
        node.submit(final, now_ns=node.current_ns)
        node.run_until_idle()
        self.assertEqual(final.operational_hit_tokens, 141)
        self.assertEqual(node.lifecycle.d_ledger.used_bytes, 0)
        self.assertEqual(node.lifecycle.ssd_ledger.used_bytes, 0)

    def test_demotion_completion_wins_exact_release_timestamp(self):
        node = self.make_node("ssd_direct")
        first = self.call(
            0,
            input_tokens=64,
            output_tokens=2,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        node.run_until_idle()
        demotion = node.lifecycle.demote(
            "s", now_ns=node.current_ns)
        resume = self.call(
            1,
            call_index=1,
            release=demotion.completion_ns,
            input_tokens=70,
            output_tokens=1,
            prefix=65,
            has_successor=False,
        )
        node.submit(resume, now_ns=demotion.completion_ns)
        ticket = node._ticket_by_request[1]
        self.assertEqual(demotion.status, TierJobStatus.COMMITTED)
        self.assertEqual(ticket.source, Tier.SSD)
        node.run_until_idle()

    def test_zero_gap_output_one_successor_waits_for_handoff_tail(self):
        node = self.make_node("cpu_ssd")
        first = self.call(
            0,
            input_tokens=1_000,
            output_tokens=1,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        self.advance_until_user_completion(node, first)
        user_ns = first.user_completion_ns
        self.assertEqual(node.pop_completed(), [first])
        self.assertEqual(first.state, TieredCallState.USER_COMPLETE)
        self.assertGreater(
            first.pool_request.handoff_completion_ns, user_ns)
        self.assertGreater(node.lifecycle.p_ledger.used_bytes, 0)

        resume = self.call(
            1,
            call_index=1,
            release=user_ns,
            input_tokens=1_001,
            output_tokens=1,
            prefix=1_000,
            has_successor=False,
        )
        independent = self.call(
            2,
            session_id="other",
            release=user_ns,
            input_tokens=16,
            output_tokens=1,
            has_successor=False,
        )
        node.submit_many(
            (resume, independent), now_ns=user_ns)
        self.assertEqual(resume.state, TieredCallState.PENDING)
        self.assertEqual(
            independent.state, TieredCallState.EXECUTING)
        lineage_ready = first.pool_request.handoff_completion_ns
        node.advance(lineage_ready)
        self.assertEqual(
            first.state, TieredCallState.INTERNAL_COMPLETE)
        self.assertNotEqual(resume.state, TieredCallState.PENDING)
        node.run_until_idle()
        self.assertEqual(node.lifecycle.p_ledger.used_bytes, 0)

    def test_p_capacity_releases_at_handoff_before_decode_finishes(self):
        node = self.make_node("cpu_ssd")
        call = self.call(
            0,
            input_tokens=1_000,
            output_tokens=8,
            has_successor=False,
        )
        node.submit(call, now_ns=0)
        while call.pool_request.handoff_completion_ns is None:
            event_ns = node.next_event_ns()
            self.assertIsNotNone(event_ns)
            node.advance(event_ns)
        handoff_ns = call.pool_request.handoff_completion_ns
        node.advance(handoff_ns)
        ticket = node._ticket_by_request[call.request_id]
        self.assertTrue(ticket.p_released)
        self.assertEqual(node.lifecycle.p_ledger.used_bytes, 0)
        self.assertIsNone(call.user_completion_ns)
        self.assertGreater(node.lifecycle.d_ledger.used_bytes, 0)
        node.run_until_idle()
        self.assertEqual(node.lifecycle.d_ledger.used_bytes, 0)

    def test_stable_d_context_shrink_releases_bytes_after_prepare(self):
        node = self.make_node(
            "cpu_ssd", p_blocks=64, d_blocks=64)
        first = self.call(
            0,
            input_tokens=256,
            output_tokens=1,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        node.run_until_idle()
        old_bytes = node.lifecycle.d_ledger.used_bytes
        release_ns = node.current_ns + 1
        resume = self.call(
            1,
            call_index=1,
            release=release_ns,
            input_tokens=32,
            output_tokens=3,
            prefix=31,
            has_successor=False,
        )
        node.submit(resume, now_ns=release_ns)
        prepare_ns = resume.prepare_completion_ns
        self.assertGreater(prepare_ns, release_ns)
        node.advance(prepare_ns, defer_schedule=True)
        target_bytes = self.hardware.kv_capacity_bytes_per_rank(34)
        self.assertLess(target_bytes, old_bytes)
        self.assertEqual(
            node.lifecycle.d_ledger.used_bytes, target_bytes)
        self.assertGreater(
            node.lifecycle.metrics.d_source_bytes_released_early, 0)
        node.flush_scheduling(prepare_ns)
        node.run_until_idle()
        self.assertEqual(node.metrics.context_shrink_calls, 1)

    def test_cpu_ssd_spills_cpu_lru_to_make_ssd_restore_bounce(self):
        node = self.make_node(
            "cpu_ssd",
            p_blocks=64,
            d_blocks=64,
            cpu_blocks=8,
            ssd_blocks=64,
        )
        first = self.call(
            0,
            input_tokens=64,
            output_tokens=1,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        node.run_until_idle()
        s_to_cpu = node.lifecycle.demote(
            "s", now_ns=node.current_ns)
        node.advance(s_to_cpu.completion_ns)

        node.lifecycle.register_d_ready(
            "tier-t", 64, now_ns=node.current_ns)
        t_to_cpu = node.lifecycle.demote(
            "tier-t", now_ns=node.current_ns)
        node.advance(t_to_cpu.completion_ns)
        self.assertEqual(
            node.lifecycle.cpu_ledger.free_bytes, 0)

        node.lifecycle.register_d_ready(
            "tier-u", 64, now_ns=node.current_ns)
        self.assertIsNone(node.lifecycle.demote(
            "tier-u", now_ns=node.current_ns))
        spill_s_ns = node.lifecycle.next_event_ns()
        self.assertIsNotNone(spill_s_ns)
        node.advance(spill_s_ns)
        self.assertEqual(
            node.lifecycle.sessions["s"].state,
            TierSessionState.SSD_READY,
        )
        u_to_cpu = node.lifecycle.demote(
            "tier-u", now_ns=node.current_ns)
        node.advance(u_to_cpu.completion_ns)
        self.assertEqual(
            node.lifecycle.cpu_ledger.free_bytes, 0)

        release_ns = node.current_ns + 1
        resume = self.call(
            1,
            call_index=1,
            release=release_ns,
            input_tokens=70,
            output_tokens=1,
            prefix=64,
            has_successor=False,
        )
        node.submit(resume, now_ns=release_ns)
        self.assertEqual(resume.state, TieredCallState.PENDING)
        self.assertGreater(node.metrics.cpu_bounce_deferrals, 0)
        spill_t_ns = node.lifecycle.next_event_ns()
        self.assertIsNotNone(spill_t_ns)
        node.advance(spill_t_ns)
        self.assertNotEqual(resume.state, TieredCallState.PENDING)
        self.assertEqual(resume.prepare_source, Tier.SSD)
        node.run_until_idle()
        self.assertEqual(
            node.lifecycle.sessions["tier-t"].state,
            TierSessionState.SSD_READY,
        )
        self.assertEqual(
            node.lifecycle.sessions["tier-u"].state,
            TierSessionState.CPU_READY,
        )
        for session_id in ("tier-t", "tier-u"):
            node.lifecycle.end(
                session_id, now_ns=node.current_ns)
        node.assert_invariants()
        self.assertEqual(node.lifecycle.cpu_ledger.used_bytes, 0)
        self.assertEqual(node.lifecycle.ssd_ledger.used_bytes, 0)

    def test_ssd_capacity_eviction_explains_tiered_recompute(self):
        node = self.make_node(
            "ssd_direct",
            p_blocks=32,
            d_blocks=32,
            cpu_blocks=8,
            ssd_blocks=4,
        )
        first_a = self.call(
            0,
            session_id="a",
            input_tokens=64,
            output_tokens=1,
            has_successor=True,
        )
        node.submit(first_a, now_ns=0)
        node.run_until_idle()
        demote_a = node.lifecycle.demote(
            "a", now_ns=node.current_ns)
        node.advance(demote_a.completion_ns)

        first_b = self.call(
            1,
            session_id="b",
            release=node.current_ns,
            input_tokens=64,
            output_tokens=1,
            has_successor=True,
        )
        node.submit(first_b, now_ns=node.current_ns)
        node.run_until_idle()
        demote_b = node.lifecycle.demote(
            "b", now_ns=node.current_ns)
        self.assertEqual(
            node.lifecycle.sessions["a"].state,
            TierSessionState.LOST,
        )
        node.advance(demote_b.completion_ns)

        resume_a = self.call(
            2,
            session_id="a",
            call_index=1,
            release=node.current_ns,
            input_tokens=65,
            output_tokens=1,
            prefix=64,
            has_successor=False,
        )
        node.submit(resume_a, now_ns=node.current_ns)
        self.assertIsNone(resume_a.prepare_source)
        node.run_until_idle()
        self.assertEqual(resume_a.operational_hit_tokens, 0)
        self.assertEqual(node.metrics.recompute_resumes, 1)
        self.assertEqual(node.lifecycle.metrics.ssd_evictions, 1)
        node.lifecycle.end("b", now_ns=node.current_ns)
        node.assert_invariants()
        self.assertEqual(node.lifecycle.ssd_ledger.used_bytes, 0)

    def test_d_capacity_blocked_head_does_not_block_terminal_call(self):
        node = self.make_node(
            "hbm_lru_recompute",
            p_blocks=64,
            d_blocks=7,
            cpu_blocks=8,
            ssd_blocks=16,
        )
        occupying = self.call(
            0,
            session_id="occupying",
            input_tokens=100,
            output_tokens=3,
            has_successor=True,
        )
        node.submit(occupying, now_ns=0)
        blocked = self.call(
            1,
            session_id="blocked",
            input_tokens=100,
            output_tokens=2,
            has_successor=False,
        )
        terminal = self.call(
            2,
            session_id="terminal",
            input_tokens=16,
            output_tokens=1,
            has_successor=False,
        )
        node.submit_many((blocked, terminal), now_ns=0)
        self.assertEqual(blocked.state, TieredCallState.PENDING)
        self.assertEqual(terminal.state, TieredCallState.EXECUTING)
        self.assertGreater(blocked.capacity_deferrals, 0)
        completed = node.run_until_idle()
        self.assertCountEqual(
            [call.request_id for call in completed],
            [0, 1, 2],
        )
        self.assertGreater(
            node.metrics.d_reclamation_immediate_drops, 0)

    def test_same_timestamp_first_calls_coalesce_on_p(self):
        node = self.make_node("cpu_ssd")
        calls = [
            self.call(
                request_id,
                session_id=f"session-{request_id}",
                output_tokens=1,
                has_successor=False,
            )
            for request_id in range(4)
        ]
        node.submit_many(calls, now_ns=0)
        first_batch = node.pool.batch_history[0]
        self.assertEqual(first_batch.stage, "p")
        self.assertEqual(len(first_batch.items), 4)
        node.run_until_idle()

    def test_full_prefix_cap_leaves_one_p_token_and_needs_no_terminal_d(self):
        node = self.make_node("cpu_ssd")
        first = self.call(
            0,
            input_tokens=100,
            output_tokens=1,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        node.run_until_idle()
        resume = self.call(
            1,
            call_index=1,
            release=node.current_ns,
            input_tokens=100,
            output_tokens=1,
            prefix=100,
            has_successor=False,
        )
        node.submit(resume, now_ns=node.current_ns)
        ticket = node._ticket_by_request[1]
        self.assertFalse(ticket.needs_d)
        node.run_until_idle()
        self.assertEqual(resume.operational_hit_tokens, 99)
        self.assertEqual(resume.pool_request.p_fresh_tokens, 1)
        self.assertEqual(resume.pool_request.d_prefix_tokens, 0)
        self.assertIsNone(
            resume.pool_request.handoff_completion_ns)
        self.assertEqual(node.metrics.full_prefix_cap_calls, 1)
        self.assertEqual(node.metrics.fresh_suffix_handoffs, 0)
        self.assertEqual(node.metrics.full_prompt_handoffs, 1)
        self.assertEqual(node.lifecycle.p_ledger.used_bytes, 0)
        self.assertEqual(node.lifecycle.d_ledger.used_bytes, 0)

    def test_call_order_and_successor_release_are_validated(self):
        node = self.make_node("cpu_ssd")
        with self.assertRaisesRegex(ValueError, "first call"):
            node.submit(self.call(0, prefix=1), now_ns=0)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            node.submit(
                self.call(1, call_index=1), now_ns=0)

        first = self.call(
            2,
            session_id="lineage",
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        successor = self.call(
            3,
            session_id="lineage",
            call_index=1,
            has_successor=False,
        )
        with self.assertRaisesRegex(ValueError, "user-complete"):
            node.submit(successor, now_ns=0)
        self.assertEqual(node.metrics.submitted_calls, 1)
        node.run_until_idle()

    def test_invalid_submission_does_not_advance_existing_event(self):
        node = self.make_node("cpu_ssd")
        valid = self.call(
            0,
            input_tokens=1_000,
            output_tokens=1,
            has_successor=False,
        )
        node.submit(valid, now_ns=0)
        event_ns = node.next_event_ns()
        self.assertIsNotNone(event_ns)
        invalid = self.call(
            1,
            session_id="invalid",
            release=event_ns,
            prefix=1,
        )
        with self.assertRaisesRegex(ValueError, "first call"):
            node.submit(invalid, now_ns=event_ns)
        self.assertEqual(node.current_ns, 0)
        self.assertEqual(node.next_event_ns(), event_ns)
        self.assertEqual(node.run_until_idle(), [valid])

    def test_deterministic_stress_conserves_work_for_every_policy(self):
        rng = random.Random(20260723)
        num_sessions = 8
        calls_per_session = 4
        specs = {}
        gaps = {}
        for session_index in range(num_sessions):
            prior_final = 0
            for call_index in range(calls_per_session):
                output_tokens = rng.randint(1, 4)
                if call_index == 0:
                    input_tokens = rng.randint(48, 96)
                    prefix_tokens = 0
                elif call_index % 3 == 1:
                    input_tokens = prior_final + rng.randint(1, 32)
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

        for policy in POLICIES:
            with self.subTest(policy=policy):
                node = self.make_node(
                    policy,
                    p_blocks=64,
                    d_blocks=24,
                    cpu_blocks=128,
                    ssd_blocks=256,
                    max_tokens=512,
                    chunk=128,
                    validate_every_event=False,
                )
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
                    external_ns = (
                        arrivals[0][0] if arrivals else None)
                    candidates = [
                        value
                        for value in (internal_ns, external_ns)
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
                                completed.session_id.removeprefix(
                                    "session-"))
                            release_ns = (
                                completed.user_completion_ns
                                + gaps[(
                                    session_index,
                                    completed.call_index,
                                )]
                            )
                            heapq.heappush(
                                arrivals,
                                (
                                    release_ns,
                                    session_index,
                                    next_index,
                                ),
                            )

                    ready = []
                    while (
                        arrivals
                        and arrivals[0][0] == now_ns
                    ):
                        (
                            release_ns,
                            session_index,
                            call_index,
                        ) = heapq.heappop(arrivals)
                        (
                            input_tokens,
                            output_tokens,
                            prefix_tokens,
                        ) = specs[(session_index, call_index)]
                        request_id = (
                            session_index * calls_per_session
                            + call_index
                        )
                        call = self.call(
                            request_id,
                            session_id=f"session-{session_index}",
                            call_index=call_index,
                            release=release_ns,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            prefix=prefix_tokens,
                            has_successor=(
                                call_index + 1
                                < calls_per_session),
                        )
                        all_calls[request_id] = call
                        ready.append(call)
                    if ready:
                        node.submit_many(ready, now_ns=now_ns)
                    else:
                        node.flush_scheduling(now_ns)

                self.assertEqual(node.run_until_idle(), [])
                self.assertEqual(len(completed_ids), expected_calls)
                self.assertEqual(
                    len(set(completed_ids)), expected_calls)
                expected_p = sum(
                    call.input_tokens - call.operational_hit_tokens
                    for call in all_calls.values()
                )
                actual_p = sum(
                    item.query_tokens
                    for batch in node.pool.batch_history
                    if batch.stage == "p"
                    for item in batch.items
                )
                expected_d = sum(
                    call.output_tokens - 1
                    for call in all_calls.values()
                )
                actual_d = sum(
                    item.query_tokens
                    for batch in node.pool.batch_history
                    if batch.stage == "d"
                    for item in batch.items
                )
                self.assertEqual(actual_p, expected_p)
                self.assertEqual(actual_d, expected_d)
                self.assertEqual(
                    node.metrics.internal_completed_calls,
                    expected_calls,
                )
                self.assertGreater(
                    node.metrics.d_reclamation_attempts, 0)
                self.assertTrue(all(
                    call.state == TieredCallState.INTERNAL_COMPLETE
                    for call in all_calls.values()
                ))
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

    def test_invalid_batch_is_atomic_and_capacity_is_validated(self):
        node = self.make_node(
            "cpu_ssd", p_blocks=1, d_blocks=1)
        valid = self.call(
            0,
            input_tokens=16,
            output_tokens=1,
            has_successor=False,
        )
        invalid = self.call(
            1,
            session_id="invalid",
            input_tokens=17,
            output_tokens=1,
            has_successor=False,
        )
        with self.assertRaisesRegex(ValueError, "P-HBM"):
            node.submit_many((valid, invalid), now_ns=0)
        self.assertEqual(node.calls, {})
        self.assertEqual(node.metrics.submitted_calls, 0)
        self.assertEqual(node.current_ns, 0)

    def test_tier_staging_infeasibility_is_rejected_before_advance(self):
        cases = (
            ("cpu", 1, 8, "CPU staging"),
            ("ssd", 8, 1, "SSD tier"),
        )
        for policy in ("ssd_direct", "cpu_ssd"):
            for name, cpu_blocks, ssd_blocks, message in cases:
                with self.subTest(policy=policy, tier=name):
                    node = self.make_node(
                        policy,
                        p_blocks=8,
                        d_blocks=8,
                        cpu_blocks=cpu_blocks,
                        ssd_blocks=ssd_blocks,
                    )
                    valid = self.call(
                        0,
                        input_tokens=16,
                        output_tokens=1,
                        has_successor=False,
                    )
                    node.submit(valid, now_ns=0)
                    event_ns = node.next_event_ns()
                    self.assertIsNotNone(event_ns)
                    invalid = self.call(
                        1,
                        session_id="oversized",
                        release=event_ns,
                        input_tokens=32,
                        output_tokens=1,
                        has_successor=True,
                    )
                    with self.assertRaisesRegex(
                            ValueError, message):
                        node.submit(invalid, now_ns=event_ns)
                    self.assertEqual(node.current_ns, 0)
                    self.assertEqual(node.next_event_ns(), event_ns)
                    self.assertNotIn(1, node.calls)
                    self.assertEqual(
                        node.metrics.submitted_calls, 1)
                    self.assertEqual(
                        node.run_until_idle(), [valid])

    def test_prepare_exception_restores_full_pending_order(self):
        node = self.make_node("cpu_ssd")
        calls = [
            self.call(
                request_id,
                session_id=f"session-{request_id}",
                output_tokens=1,
                has_successor=False,
            )
            for request_id in range(3)
        ]
        original = node._try_prepare

        def injected(call, *, now_ns):
            if call.request_id == 0:
                return None
            if call.request_id == 1:
                raise RuntimeError("injected prepare failure")
            return original(call, now_ns=now_ns)

        with patch.object(
                node, "_try_prepare", side_effect=injected):
            with self.assertRaisesRegex(
                    RuntimeError, "injected prepare failure"):
                node.submit_many(calls, now_ns=0)
        self.assertEqual(
            list(node._pending_call_ids), [0, 1, 2])
        self.assertTrue(all(
            call.state == TieredCallState.PENDING
            for call in calls
        ))
        self.assertEqual(node.lifecycle.prepares, {})
        node.flush_scheduling(0)
        completed = node.run_until_idle()
        self.assertEqual(
            [call.request_id for call in completed],
            [0, 1, 2],
        )

    def test_report_is_json_safe_and_names_single_capacity_owner(self):
        node = self.make_node(
            "ssd_direct", node_id=3,
            validate_every_event=False)
        call = self.call(0, output_tokens=1)
        node.submit(call, now_ns=0)
        node.run_until_idle()
        report = node.report()
        json.dumps(report)
        self.assertEqual(report["mode"], "finite_hbm_p4d4_tiering")
        self.assertEqual(report["policy"], "ssd_direct")
        self.assertEqual(report["node_id"], 3)
        self.assertIn("sole P/D KV ledger", report["capacity_owner"])
        self.assertFalse(hasattr(node, "hbm"))
        self.assertFalse(report["validate_every_event"])
        self.assertFalse(node.lifecycle.validate_every_event)
        self.assertFalse(node.pool.validate_every_event)


if __name__ == "__main__":
    unittest.main()

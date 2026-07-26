from pathlib import Path
import json
import random
import unittest

from serving.core.gpu_pd_latency import load_p4d4_gpu_config
from serving.core.gpu_pd_tier_lifecycle import (
    SharedByteLedger,
    Tier,
    TierJobStatus,
    TierSessionState,
    TieredPDKVLifecycle,
)
from serving.core.hbf_full_model_lifecycle import ResourceCalendar


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_CONFIG = (
    REPO_ROOT / "configs" / "wakekv_hbf"
    / "p4d4_gpu_server.json"
)


class SharedByteLedgerTests(unittest.TestCase):
    def test_reservations_and_atomic_replacement_conserve_capacity(self):
        ledger = SharedByteLedger("shared", 100)
        self.assertTrue(ledger.reserve("source", 60))
        self.assertTrue(ledger.reserve("destination", 40))
        self.assertFalse(ledger.reserve("overflow", 1))
        ledger.replace(
            remove_owners=("source", "destination"),
            owner="committed",
            byte_count=70,
        )
        self.assertEqual(ledger.used_bytes, 70)
        self.assertEqual(ledger.peak_used_bytes, 100)
        self.assertEqual(ledger.owners, {"committed": 70})

    def test_failed_replacement_is_atomic_on_owner_collision(self):
        ledger = SharedByteLedger("shared", 100)
        ledger.set_bytes("keep", 10)
        ledger.set_bytes("remove", 20)
        before = dict(ledger.owners)
        with self.assertRaisesRegex(
                RuntimeError, "replacement owner already exists"):
            ledger.replace(
                remove_owners=("remove",),
                owner="keep",
                byte_count=30,
            )
        self.assertEqual(ledger.owners, before)


class TieredPDKVLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = load_p4d4_gpu_config(GPU_CONFIG)
        cls.block_per_rank = (
            cls.hardware.kv_capacity_bytes_per_rank(16))
        cls.block_aggregate = (
            cls.block_per_rank * cls.hardware.tp_size)

    def make_lifecycle(
            self, policy, *, node_id=0, blocks=64,
            cpu_blocks=64, ssd_blocks=128, calendar=None,
            restore_execution_mode="bulk",
            validate_every_event=True):
        return TieredPDKVLifecycle(
            hardware=self.hardware,
            node_id=node_id,
            policy=policy,
            calendar=calendar,
            p_capacity_bytes_per_rank=(
                blocks * self.block_per_rank),
            d_capacity_bytes_per_rank=(
                blocks * self.block_per_rank),
            cpu_capacity_bytes=(
                cpu_blocks * self.block_aggregate),
            ssd_capacity_bytes=(
                ssd_blocks * self.block_aggregate),
            restore_execution_mode=restore_execution_mode,
            validate_every_event=validate_every_event,
        )

    @staticmethod
    def finish_prepare(lifecycle, ticket, *, successor=True):
        lifecycle.advance(ticket.completion_ns)
        lifecycle.pop_prepare_completed()
        lifecycle.mark_active(
            ticket, now_ns=ticket.completion_ns)
        if ticket.needs_d:
            lifecycle.release_p_after_handoff(
                ticket, now_ns=ticket.completion_ns)
        lifecycle.commit_d_ready(
            ticket,
            now_ns=ticket.completion_ns,
            has_successor=successor,
        )

    def test_recompute_policy_drops_whole_lru_object(self):
        lifecycle = self.make_lifecycle(
            "hbm_lru_recompute", blocks=2)
        lifecycle.register_d_ready("b", 16, now_ns=0)
        lifecycle.register_d_ready("a", 16, now_ns=0)
        self.assertEqual(
            lifecycle.ensure_d_headroom(
                self.block_per_rank, now_ns=0),
            0,
        )
        self.assertEqual(
            lifecycle.sessions["a"].state,
            TierSessionState.LOST,
        )
        self.assertEqual(
            lifecycle.sessions["b"].state,
            TierSessionState.D_READY,
        )
        self.assertEqual(
            lifecycle.d_ledger.free_bytes,
            self.block_per_rank,
        )

    def test_ssd_direct_reserves_source_destination_and_bounce(self):
        lifecycle = self.make_lifecycle(
            "ssd_direct", blocks=4,
            cpu_blocks=1, ssd_blocks=2)
        lifecycle.register_d_ready("s", 16, now_ns=0)
        job = lifecycle.demote("s", now_ns=0)
        self.assertIsNotNone(job)
        source = lifecycle.copies[job.source_copy_id]
        self.assertEqual(source.demotion_pins, 1)
        self.assertEqual(
            lifecycle.d_ledger.used_bytes,
            self.block_per_rank,
        )
        self.assertEqual(
            lifecycle.cpu_ledger.used_bytes,
            self.block_aggregate,
        )
        self.assertEqual(
            lifecycle.ssd_ledger.used_bytes,
            self.block_aggregate,
        )
        self.assertEqual(
            job.transfer_kinds,
            ("d-gpu_to_cpu", "cpu-to-ssd"),
        )
        self.assertEqual(
            job.stages[0].completion_ns,
            job.stages[1].start_ns,
        )
        lifecycle.advance(job.completion_ns)
        self.assertEqual(job.status, TierJobStatus.COMMITTED)
        self.assertEqual(
            lifecycle.sessions["s"].state,
            TierSessionState.SSD_READY,
        )
        self.assertEqual(lifecycle.d_ledger.used_bytes, 0)
        self.assertEqual(lifecycle.cpu_ledger.used_bytes, 0)

    def test_cpu_ssd_pressure_spills_cpu_lru_then_retries_demotion(self):
        lifecycle = self.make_lifecycle(
            "cpu_ssd", blocks=4,
            cpu_blocks=1, ssd_blocks=4)
        lifecycle.register_d_ready("a", 16, now_ns=0)
        first = lifecycle.demote("a", now_ns=0)
        lifecycle.advance(first.completion_ns)
        lifecycle.register_d_ready(
            "b", 16, now_ns=first.completion_ns)

        self.assertIsNone(
            lifecycle.demote("b", now_ns=first.completion_ns))
        spill_ns = lifecycle.next_event_ns()
        self.assertIsNotNone(spill_ns)
        self.assertEqual(
            lifecycle.sessions["a"].state,
            TierSessionState.CPU_DEMOTING_SSD,
        )
        lifecycle.advance(spill_ns)
        self.assertEqual(
            lifecycle.sessions["a"].state,
            TierSessionState.SSD_READY,
        )
        second = lifecycle.demote("b", now_ns=spill_ns)
        self.assertIsNotNone(second)
        lifecycle.advance(second.completion_ns)
        self.assertEqual(
            lifecycle.sessions["b"].state,
            TierSessionState.CPU_READY,
        )

    def test_capacity_lru_tie_breaks_by_session_id(self):
        lifecycle = self.make_lifecycle(
            "cpu_ssd", blocks=6,
            cpu_blocks=2, ssd_blocks=8)
        lifecycle.register_d_ready("b", 16, now_ns=0)
        lifecycle.register_d_ready("a", 16, now_ns=0)
        lifecycle.register_d_ready("c", 16, now_ns=0)
        for session_id in ("b", "a"):
            job = lifecycle.demote(
                session_id, now_ns=lifecycle.current_ns)
            lifecycle.advance(job.completion_ns)
        now_ns = lifecycle.current_ns
        self.assertIsNone(
            lifecycle.demote("c", now_ns=now_ns))
        self.assertEqual(
            lifecycle.sessions["a"].state,
            TierSessionState.CPU_DEMOTING_SSD,
        )
        self.assertEqual(
            lifecycle.sessions["b"].state,
            TierSessionState.CPU_READY,
        )

    def test_resume_during_demotion_makes_publish_stale_but_keeps_work(self):
        lifecycle = self.make_lifecycle(
            "ssd_direct", blocks=32,
            cpu_blocks=16, ssd_blocks=32)
        lifecycle.register_d_ready("s", 128, now_ns=0)
        demotion = lifecycle.demote("s", now_ns=0)
        source = lifecycle.copies[demotion.source_copy_id]
        ticket = lifecycle.begin_prepare(
            "s",
            request_id=1,
            now_ns=1,
            input_tokens=129,
            output_tokens=2,
            reusable_tokens=128,
            has_successor=True,
        )
        self.assertIsNotNone(ticket)
        self.assertEqual(source.demotion_pins, 1)
        self.assertEqual(source.foreground_pins, 1)
        self.assertTrue(ticket.full_d_reservation)
        self.assertEqual(
            lifecycle.d_ledger.owner_bytes(ticket.d_owner),
            ticket.d_reserved_bytes_per_rank,
        )
        self.assertGreater(
            lifecycle.d_ledger.used_bytes,
            source.byte_count,
        )

        lifecycle.advance(ticket.completion_ns)
        self.assertEqual(source.foreground_pins, 0)
        self.assertEqual(source.demotion_pins, 1)
        lifecycle.mark_active(
            ticket, now_ns=ticket.completion_ns)
        lifecycle.release_p_after_handoff(
            ticket, now_ns=ticket.completion_ns)
        lifecycle.commit_d_ready(
            ticket,
            now_ns=ticket.completion_ns,
            has_successor=True,
        )
        self.assertIn(source.copy_id, lifecycle.copies)
        lifecycle.advance(demotion.completion_ns)
        self.assertEqual(demotion.status, TierJobStatus.STALE)
        self.assertNotIn(source.copy_id, lifecycle.copies)
        self.assertEqual(lifecycle.ssd_ledger.used_bytes, 0)
        self.assertGreater(
            lifecycle.metrics.stale_transfer_bytes, 0)

    def test_full_d_destination_wins_after_early_stale_demotion(self):
        lifecycle = self.make_lifecycle(
            "ssd_direct", blocks=64,
            cpu_blocks=64, ssd_blocks=64)
        lifecycle.register_d_ready("s", 128, now_ns=0)
        demotion = lifecycle.demote("s", now_ns=0)
        old_copy_id = demotion.source_copy_id
        ticket = lifecycle.begin_prepare(
            "s",
            request_id=1,
            now_ns=1,
            input_tokens=129,
            output_tokens=2,
            reusable_tokens=128,
            has_successor=True,
        )
        self.assertTrue(ticket.full_d_reservation)
        self.assertLess(
            ticket.completion_ns, demotion.completion_ns)

        lifecycle.advance(demotion.completion_ns)
        self.assertEqual(demotion.status, TierJobStatus.STALE)
        self.assertIn(old_copy_id, lifecycle.copies)
        lifecycle.pop_prepare_completed()
        lifecycle.mark_active(
            ticket, now_ns=demotion.completion_ns)
        lifecycle.release_p_after_handoff(
            ticket, now_ns=demotion.completion_ns)
        lifecycle.commit_d_ready(
            ticket,
            now_ns=demotion.completion_ns,
            has_successor=True,
        )

        new_copy_id = lifecycle.sessions["s"].primary_copy_id
        self.assertNotEqual(new_copy_id, old_copy_id)
        self.assertNotIn(old_copy_id, lifecycle.copies)
        self.assertEqual(
            lifecycle.copies[new_copy_id].tokens,
            ticket.final_tokens,
        )

    def test_completion_wins_exact_resume_timestamp(self):
        lifecycle = self.make_lifecycle(
            "ssd_direct", blocks=16,
            cpu_blocks=8, ssd_blocks=16)
        lifecycle.register_d_ready("s", 64, now_ns=0)
        demotion = lifecycle.demote("s", now_ns=0)
        source = lifecycle.peek_resume_source(
            "s", now_ns=demotion.completion_ns)
        self.assertEqual(source.source, Tier.SSD)
        self.assertFalse(source.demotion_inflight)
        ticket = lifecycle.begin_prepare(
            "s",
            request_id=1,
            now_ns=demotion.completion_ns,
            input_tokens=65,
            output_tokens=2,
            reusable_tokens=64,
            has_successor=True,
        )
        self.assertEqual(ticket.source, Tier.SSD)
        self.assertEqual(demotion.status, TierJobStatus.COMMITTED)

    def test_cpu_source_is_retired_until_stale_spill_reader_finishes(self):
        lifecycle = self.make_lifecycle(
            "cpu_ssd", blocks=32,
            cpu_blocks=16, ssd_blocks=32)
        lifecycle.register_d_ready("s", 128, now_ns=0)
        d2c = lifecycle.demote("s", now_ns=0)
        lifecycle.advance(d2c.completion_ns)
        source_id = lifecycle.sessions["s"].primary_copy_id
        spill = lifecycle._start_cpu_spill(
            lifecycle.sessions["s"],
            now_ns=lifecycle.current_ns,
        )
        ticket = lifecycle.begin_prepare(
            "s",
            request_id=1,
            now_ns=lifecycle.current_ns + 1,
            input_tokens=129,
            output_tokens=2,
            reusable_tokens=128,
            has_successor=True,
        )
        self.assertEqual(ticket.source, Tier.CPU)
        self.assertLess(spill.completion_ns, ticket.completion_ns)
        lifecycle.advance(spill.completion_ns)
        source = lifecycle.copies[source_id]
        self.assertTrue(source.retired)
        self.assertEqual(source.foreground_pins, 1)
        self.assertEqual(source.demotion_pins, 0)
        self.assertGreater(lifecycle.cpu_ledger.used_bytes, 0)
        self.assertEqual(spill.status, TierJobStatus.STALE)
        lifecycle.advance(ticket.completion_ns)
        self.assertNotIn(source_id, lifecycle.copies)
        self.assertEqual(lifecycle.cpu_ledger.used_bytes, 0)

    def test_cpu_prepare_is_one_stage_and_ssd_prepare_keeps_shadow(self):
        cpu = self.make_lifecycle(
            "cpu_ssd", blocks=16,
            cpu_blocks=8, ssd_blocks=16)
        cpu.register_d_ready("s", 64, now_ns=0)
        d2c = cpu.demote("s", now_ns=0)
        cpu.advance(d2c.completion_ns)
        ticket = cpu.begin_prepare(
            "s",
            request_id=1,
            now_ns=cpu.current_ns,
            input_tokens=65,
            output_tokens=2,
            reusable_tokens=64,
            has_successor=True,
        )
        self.assertEqual(
            ticket.transfer_kinds, ("p-cpu_to_gpu",))
        cpu.advance(ticket.completion_ns)
        self.assertEqual(cpu.cpu_ledger.used_bytes, 0)

        ssd = self.make_lifecycle(
            "ssd_direct", blocks=16,
            cpu_blocks=8, ssd_blocks=16)
        ssd.register_d_ready("s", 64, now_ns=0)
        d2s = ssd.demote("s", now_ns=0)
        ssd.advance(d2s.completion_ns)
        shadow_id = ssd.sessions["s"].primary_copy_id
        ticket = ssd.begin_prepare(
            "s",
            request_id=1,
            now_ns=ssd.current_ns,
            input_tokens=65,
            output_tokens=2,
            reusable_tokens=64,
            has_successor=True,
        )
        self.assertEqual(
            ticket.transfer_kinds,
            ("ssd-to-cpu", "p-cpu_to_gpu"),
        )
        self.assertEqual(
            ticket.stages[0].completion_ns,
            ticket.stages[1].start_ns,
        )
        ssd.advance(ticket.completion_ns)
        self.assertTrue(ssd.copies[shadow_id].shadow)
        self.assertGreater(ssd.ssd_ledger.used_bytes, 0)
        self.finish_prepare_after_advance(
            ssd, ticket, successor=True)
        self.assertNotIn(shadow_id, ssd.copies)
        self.assertEqual(ssd.ssd_ledger.used_bytes, 0)

    def test_layerwise_cpu_and_ssd_restore_publish_before_full_transfer(self):
        for policy, expected_source, expected_stage_count in (
                ("cpu_ssd", Tier.CPU, 48),
                ("ssd_direct", Tier.SSD, 96)):
            with self.subTest(policy=policy):
                lifecycle = self.make_lifecycle(
                    policy,
                    blocks=16,
                    cpu_blocks=8,
                    ssd_blocks=16,
                    restore_execution_mode="layerwise_streaming",
                )
                lifecycle.register_d_ready("s", 64, now_ns=0)
                demotion = lifecycle.demote("s", now_ns=0)
                lifecycle.advance(demotion.completion_ns)
                source_id = lifecycle.sessions[
                    "s"].primary_copy_id
                ticket = lifecycle.begin_prepare(
                    "s",
                    request_id=1,
                    now_ns=lifecycle.current_ns,
                    input_tokens=65,
                    output_tokens=2,
                    reusable_tokens=64,
                    has_successor=True,
                )

                self.assertEqual(ticket.source, expected_source)
                self.assertEqual(
                    ticket.restore_execution_mode,
                    "layerwise_streaming",
                )
                self.assertEqual(
                    len(ticket.stages), expected_stage_count)
                self.assertEqual(
                    len(ticket.restore_layer_ready_ns), 48)
                self.assertEqual(
                    ticket.restore_layer_ready_ns[-1],
                    ticket.completion_ns,
                )
                self.assertEqual(
                    tuple(sorted(ticket.restore_layer_ready_ns)),
                    ticket.restore_layer_ready_ns,
                )
                transferred = sum(
                    stage.stage.aggregate_bytes
                    for stage in ticket.stages
                )
                expected_object_bytes = (
                    lifecycle._aggregate_bytes(ticket.hit_tokens))
                self.assertEqual(
                    transferred,
                    expected_object_bytes
                    * (2 if expected_source == Tier.SSD else 1),
                )

                lifecycle.release_prepare_to_pool(
                    ticket, now_ns=lifecycle.current_ns)
                self.assertTrue(ticket.pool_released)
                self.assertFalse(ticket.completed)
                source = lifecycle.copies[source_id]
                self.assertEqual(source.foreground_pins, 1)
                if expected_source == Tier.SSD:
                    self.assertEqual(
                        lifecycle.cpu_ledger.owner_bytes(
                            ticket.bounce_owner),
                        expected_object_bytes,
                    )
                lifecycle.advance(ticket.completion_ns - 1)
                self.assertEqual(source.foreground_pins, 1)
                self.assertFalse(ticket.completed)
                lifecycle.advance(ticket.completion_ns)
                lifecycle.pop_prepare_completed()
                self.assertTrue(ticket.completed)
                if expected_source == Tier.SSD:
                    self.assertEqual(
                        lifecycle.cpu_ledger.owner_bytes(
                            ticket.bounce_owner),
                        0,
                    )
                    self.assertIn(source_id, lifecycle.copies)
                else:
                    self.assertNotIn(source_id, lifecycle.copies)
                lifecycle.mark_active(
                    ticket, now_ns=ticket.completion_ns)
                lifecycle.release_p_after_handoff(
                    ticket, now_ns=ticket.completion_ns)
                lifecycle.commit_d_ready(
                    ticket,
                    now_ns=ticket.completion_ns,
                    has_successor=True,
                )
                lifecycle.assert_invariants()

    def test_invalid_restore_execution_mode_is_rejected(self):
        with self.assertRaisesRegex(
                ValueError, "restore_execution_mode"):
            self.make_lifecycle(
                "ssd_direct",
                restore_execution_mode="implicit_streaming",
            )

    @staticmethod
    def finish_prepare_after_advance(
            lifecycle, ticket, *, successor):
        lifecycle.pop_prepare_completed()
        lifecycle.mark_active(
            ticket, now_ns=ticket.completion_ns)
        if ticket.needs_d:
            lifecycle.release_p_after_handoff(
                ticket, now_ns=ticket.completion_ns)
        lifecycle.commit_d_ready(
            ticket,
            now_ns=ticket.completion_ns,
            has_successor=successor,
        )

    def test_ssd_capacity_evicts_whole_lru_and_resume_misses(self):
        lifecycle = self.make_lifecycle(
            "ssd_direct", blocks=8,
            cpu_blocks=2, ssd_blocks=1)
        lifecycle.register_d_ready("a", 16, now_ns=0)
        first = lifecycle.demote("a", now_ns=0)
        lifecycle.advance(first.completion_ns)
        lifecycle.register_d_ready(
            "b", 16, now_ns=lifecycle.current_ns)
        second = lifecycle.demote(
            "b", now_ns=lifecycle.current_ns)
        self.assertEqual(
            lifecycle.sessions["a"].state,
            TierSessionState.LOST,
        )
        lifecycle.advance(second.completion_ns)
        source = lifecycle.peek_resume_source(
            "a", now_ns=lifecycle.current_ns)
        self.assertIsNone(source.source)
        ticket = lifecycle.begin_prepare(
            "a",
            request_id=1,
            now_ns=lifecycle.current_ns,
            input_tokens=16,
            output_tokens=1,
            reusable_tokens=15,
            has_successor=False,
        )
        self.assertIsNone(ticket.source)
        self.assertEqual(ticket.hit_tokens, 0)

    def test_destination_overcommit_is_atomic_and_does_not_bump_generation(self):
        lifecycle = self.make_lifecycle(
            "ssd_direct", blocks=1,
            cpu_blocks=4, ssd_blocks=4)
        lifecycle.register_d_ready("s", 16, now_ns=0)
        demotion = lifecycle.demote("s", now_ns=0)
        record = lifecycle.sessions["s"]
        generation = record.generation
        source = lifecycle.copies[record.primary_copy_id]
        reservations_before = list(
            lifecycle.calendar.reservations)
        ticket = lifecycle.begin_prepare(
            "s",
            request_id=1,
            now_ns=1,
            input_tokens=16,
            output_tokens=1,
            reusable_tokens=15,
            has_successor=True,
        )
        self.assertIsNone(ticket)
        self.assertEqual(record.generation, generation)
        self.assertEqual(record.state, TierSessionState.D_DEMOTING_SSD)
        self.assertEqual(source.demotion_pins, 1)
        self.assertEqual(source.foreground_pins, 0)
        self.assertEqual(lifecycle.p_ledger.used_bytes, 0)
        self.assertEqual(
            lifecycle.d_ledger.used_bytes,
            self.block_per_rank,
        )
        self.assertEqual(
            lifecycle.calendar.reservations,
            reservations_before,
        )
        lifecycle.advance(demotion.completion_ns)

    def test_terminal_output_one_reserves_no_d_destination(self):
        lifecycle = self.make_lifecycle(
            "hbm_lru_recompute", blocks=1)
        ticket = lifecycle.begin_prepare(
            "new",
            request_id=1,
            now_ns=0,
            input_tokens=16,
            output_tokens=1,
            reusable_tokens=0,
            has_successor=False,
        )
        self.assertIsNotNone(ticket)
        self.assertFalse(ticket.needs_d)
        self.assertIsNone(ticket.d_owner)
        self.assertEqual(ticket.d_reserved_bytes_per_rank, 0)
        self.assertEqual(ticket.d_target_bytes_per_rank, 0)
        self.assertEqual(lifecycle.d_ledger.used_bytes, 0)
        self.finish_prepare(
            lifecycle, ticket, successor=False)
        self.assertEqual(
            lifecycle.sessions["new"].state,
            TierSessionState.ENDED,
        )
        self.assertEqual(lifecycle.d_ledger.used_bytes, 0)

    def test_handoff_releases_p_before_decode_commit(self):
        lifecycle = self.make_lifecycle(
            "hbm_lru_recompute", blocks=4)
        ticket = lifecycle.begin_prepare(
            "multi",
            request_id=0,
            now_ns=0,
            input_tokens=16,
            output_tokens=2,
            reusable_tokens=0,
            has_successor=False,
        )
        lifecycle.advance(ticket.completion_ns)
        lifecycle.pop_prepare_completed()
        lifecycle.mark_active(
            ticket, now_ns=ticket.completion_ns)
        self.assertEqual(
            lifecycle.p_ledger.used_bytes,
            self.block_per_rank,
        )
        with self.assertRaisesRegex(
                RuntimeError, "handoff must complete"):
            lifecycle.commit_d_ready(
                ticket,
                now_ns=ticket.completion_ns,
                has_successor=False,
            )
        lifecycle.release_p_after_handoff(
            ticket, now_ns=ticket.completion_ns)
        self.assertEqual(lifecycle.p_ledger.used_bytes, 0)
        self.assertEqual(
            lifecycle.d_ledger.used_bytes,
            2 * self.block_per_rank,
        )
        d_owner = ticket.d_owner
        d_bytes = ticket.d_reserved_bytes_per_rank
        lifecycle.d_ledger.release(d_owner)
        with self.assertRaisesRegex(
                AssertionError, "D destination ownership"):
            lifecycle.assert_invariants()
        self.assertTrue(
            lifecycle.d_ledger.reserve(d_owner, d_bytes))
        with self.assertRaisesRegex(
                RuntimeError, "already released"):
            lifecycle.release_p_after_handoff(
                ticket, now_ns=ticket.completion_ns)
        lifecycle.commit_d_ready(
            ticket,
            now_ns=ticket.completion_ns,
            has_successor=False,
        )
        self.assertEqual(lifecycle.d_ledger.used_bytes, 0)

    def test_stable_d_source_is_released_or_shrunk_after_prepare(self):
        terminal = self.make_lifecycle(
            "cpu_ssd", blocks=8)
        terminal.register_d_ready("terminal", 64, now_ns=0)
        ticket = terminal.begin_prepare(
            "terminal",
            request_id=1,
            now_ns=0,
            input_tokens=16,
            output_tokens=1,
            reusable_tokens=15,
            has_successor=False,
        )
        terminal.advance(ticket.completion_ns)
        self.assertEqual(terminal.d_ledger.used_bytes, 0)
        self.assertIsNone(
            terminal.sessions["terminal"].primary_copy_id)
        self.finish_prepare_after_advance(
            terminal, ticket, successor=False)

        shrink = self.make_lifecycle(
            "cpu_ssd", blocks=8)
        shrink.register_d_ready("shrink", 64, now_ns=0)
        ticket = shrink.begin_prepare(
            "shrink",
            request_id=2,
            now_ns=0,
            input_tokens=16,
            output_tokens=2,
            reusable_tokens=15,
            has_successor=True,
        )
        shrink.advance(ticket.completion_ns)
        expected = shrink.hardware.kv_capacity_bytes_per_rank(17)
        self.assertEqual(shrink.d_ledger.used_bytes, expected)
        self.assertGreater(
            shrink.metrics.d_source_bytes_released_early, 0)
        self.finish_prepare_after_advance(
            shrink, ticket, successor=True)

    def test_zero_request_id_is_valid_and_input_contract_is_checked(self):
        lifecycle = self.make_lifecycle(
            "hbm_lru_recompute", blocks=1)
        ticket = lifecycle.begin_prepare(
            "zero",
            request_id=0,
            now_ns=0,
            input_tokens=16,
            output_tokens=1,
            reusable_tokens=0,
            has_successor=False,
        )
        self.assertIsNotNone(ticket)
        self.finish_prepare(
            lifecycle, ticket, successor=False)

        with self.assertRaisesRegex(
                ValueError, "cannot exceed input_tokens"):
            lifecycle.begin_prepare(
                "bad-reuse",
                request_id=1,
                now_ns=ticket.completion_ns,
                input_tokens=16,
                output_tokens=1,
                reusable_tokens=17,
                has_successor=False,
            )
        with self.assertRaisesRegex(
                ValueError, "final context exceeds"):
            lifecycle.begin_prepare(
                "too-long",
                request_id=2,
                now_ns=ticket.completion_ns,
                input_tokens=1_010_000,
                output_tokens=2,
                reusable_tokens=0,
                has_successor=False,
            )
        with self.assertRaisesRegex(
                ValueError, "version must be a positive integer"):
            lifecycle.register_d_ready(
                "bad-version",
                16,
                now_ns=ticket.completion_ns,
                version=False,
            )

    def test_deferral_is_phantom_free_and_infeasible_fails_fast(self):
        lifecycle = self.make_lifecycle(
            "hbm_lru_recompute", blocks=1)
        busy = lifecycle.begin_prepare(
            "busy",
            request_id=0,
            now_ns=0,
            input_tokens=16,
            output_tokens=1,
            reusable_tokens=0,
            has_successor=False,
        )
        self.assertIsNotNone(busy)
        deferred = lifecycle.begin_prepare(
            "deferred",
            request_id=1,
            now_ns=0,
            input_tokens=16,
            output_tokens=1,
            reusable_tokens=0,
            has_successor=False,
        )
        self.assertIsNone(deferred)
        self.assertNotIn("deferred", lifecycle.sessions)
        self.assertNotIn(1, lifecycle.report()["seen_request_ids"])

        with self.assertRaisesRegex(
                RuntimeError, "individually infeasible"):
            lifecycle.begin_prepare(
                "too-large",
                request_id=2,
                now_ns=0,
                input_tokens=17,
                output_tokens=1,
                reusable_tokens=0,
                has_successor=False,
            )
        self.assertNotIn("too-large", lifecycle.sessions)
        with self.assertRaisesRegex(
                ValueError, "context exceeds"):
            lifecycle.register_d_ready(
                "too-long",
                1_010_001,
                now_ns=0,
            )
        self.finish_prepare(
            lifecycle, busy, successor=False)

    def test_request_ids_are_globally_unique_after_admission(self):
        lifecycle = self.make_lifecycle(
            "hbm_lru_recompute", blocks=2)
        first = lifecycle.begin_prepare(
            "first",
            request_id=0,
            now_ns=0,
            input_tokens=16,
            output_tokens=1,
            reusable_tokens=0,
            has_successor=False,
        )
        self.finish_prepare(
            lifecycle, first, successor=False)
        with self.assertRaisesRegex(
                ValueError, "duplicate request_id=0"):
            lifecycle.begin_prepare(
                "second",
                request_id=0,
                now_ns=lifecycle.current_ns,
                input_tokens=16,
                output_tokens=1,
                reusable_tokens=0,
                has_successor=False,
            )
        self.assertNotIn("second", lifecycle.sessions)

    def test_cpu_bounce_headroom_spills_lru_and_protects_resume(self):
        lifecycle = self.make_lifecycle(
            "cpu_ssd", blocks=4,
            cpu_blocks=1, ssd_blocks=2)

        lifecycle.register_d_ready("old", 16, now_ns=0)
        old_to_cpu = lifecycle.demote("old", now_ns=0)
        lifecycle.advance(old_to_cpu.completion_ns)

        lifecycle.register_d_ready(
            "resume", 16, now_ns=lifecycle.current_ns)
        self.assertIsNone(lifecycle.demote(
            "resume", now_ns=lifecycle.current_ns))
        spill_old_ns = lifecycle.next_event_ns()
        lifecycle.advance(spill_old_ns)
        resume_to_cpu = lifecycle.demote(
            "resume", now_ns=lifecycle.current_ns)
        lifecycle.advance(resume_to_cpu.completion_ns)

        lifecycle.register_d_ready(
            "occupant", 16, now_ns=lifecycle.current_ns)
        self.assertIsNone(lifecycle.demote(
            "occupant", now_ns=lifecycle.current_ns))
        spill_resume_ns = lifecycle.next_event_ns()
        lifecycle.advance(spill_resume_ns)
        occupant_to_cpu = lifecycle.demote(
            "occupant", now_ns=lifecycle.current_ns)
        lifecycle.advance(occupant_to_cpu.completion_ns)

        self.assertEqual(
            lifecycle.sessions["resume"].state,
            TierSessionState.SSD_READY,
        )
        blocked = lifecycle.begin_prepare(
            "resume",
            request_id=10,
            now_ns=lifecycle.current_ns,
            input_tokens=16,
            output_tokens=1,
            reusable_tokens=15,
            has_successor=False,
        )
        self.assertIsNone(blocked)
        progress_ns = lifecycle.ensure_cpu_bounce_headroom(
            15,
            now_ns=lifecycle.current_ns,
            protected_session="resume",
        )
        self.assertGreater(progress_ns, lifecycle.current_ns)
        lifecycle.advance(progress_ns)
        self.assertEqual(
            lifecycle.sessions["resume"].state,
            TierSessionState.SSD_READY,
        )
        ticket = lifecycle.begin_prepare(
            "resume",
            request_id=10,
            now_ns=lifecycle.current_ns,
            input_tokens=16,
            output_tokens=1,
            reusable_tokens=15,
            has_successor=False,
        )
        self.assertIsNotNone(ticket)
        self.finish_prepare(
            lifecycle, ticket, successor=False)
        lifecycle.end(
            "occupant", now_ns=lifecycle.current_ns)
        lifecycle.end(
            "old", now_ns=lifecycle.current_ns)
        lifecycle.run_until_idle()
        self.assertEqual(lifecycle.cpu_ledger.used_bytes, 0)
        self.assertEqual(lifecycle.ssd_ledger.used_bytes, 0)

    def test_oversized_ssd_destination_does_not_evict_existing_copy(self):
        lifecycle = self.make_lifecycle(
            "ssd_direct", blocks=4,
            cpu_blocks=4, ssd_blocks=1)
        lifecycle.register_d_ready("keep", 16, now_ns=0)
        keep_job = lifecycle.demote("keep", now_ns=0)
        lifecycle.advance(keep_job.completion_ns)
        keep_copy = lifecycle.sessions["keep"].primary_copy_id
        lifecycle.register_d_ready(
            "large", 32, now_ns=lifecycle.current_ns)
        with self.assertRaisesRegex(
                RuntimeError, "exceeds node SSD capacity"):
            lifecycle.demote(
                "large", now_ns=lifecycle.current_ns)
        self.assertEqual(
            lifecycle.sessions["keep"].primary_copy_id,
            keep_copy,
        )
        self.assertEqual(
            lifecycle.sessions["keep"].state,
            TierSessionState.SSD_READY,
        )

    def test_zero_reuse_d_source_still_backs_delta_only_d_capacity(self):
        lifecycle = self.make_lifecycle(
            "cpu_ssd", blocks=1)
        lifecycle.register_d_ready("s", 16, now_ns=0)
        source_id = lifecycle.sessions["s"].primary_copy_id
        ticket = lifecycle.begin_prepare(
            "s",
            request_id=1,
            now_ns=0,
            input_tokens=16,
            output_tokens=1,
            reusable_tokens=0,
            has_successor=True,
        )
        self.assertIsNotNone(ticket)
        self.assertIsNone(ticket.source)
        self.assertEqual(ticket.hit_tokens, 0)
        self.assertEqual(ticket.d_reuse_copy_id, source_id)
        self.assertIsNone(ticket.d_owner)
        self.assertIn(source_id, lifecycle.copies)
        self.assertEqual(
            lifecycle.d_ledger.used_bytes,
            self.block_per_rank,
        )
        self.finish_prepare(
            lifecycle, ticket, successor=True)
        self.assertEqual(
            lifecycle.sessions["s"].state,
            TierSessionState.D_READY,
        )
        self.assertEqual(
            lifecycle.d_ledger.used_bytes,
            self.block_per_rank,
        )

    def test_existing_ended_id_is_rejected_while_stale_job_drains(self):
        lifecycle = self.make_lifecycle(
            "ssd_direct", blocks=8,
            cpu_blocks=4, ssd_blocks=8)
        lifecycle.register_d_ready("s", 64, now_ns=0)
        job = lifecycle.demote("s", now_ns=0)
        lifecycle.end("s", now_ns=1)
        source = lifecycle.copies[job.source_copy_id]
        self.assertTrue(source.retired)
        self.assertEqual(source.demotion_pins, 1)
        with self.assertRaisesRegex(
                RuntimeError, "already registered"):
            lifecycle.register_d_ready("s", 16, now_ns=1)
        lifecycle.advance(job.completion_ns)
        self.assertEqual(job.status, TierJobStatus.STALE)
        self.assertNotIn(source.copy_id, lifecycle.copies)

    def test_two_nodes_share_calendar_without_sharing_physical_queues(self):
        calendar = ResourceCalendar()
        node0 = self.make_lifecycle(
            "ssd_direct", node_id=0,
            cpu_blocks=2, ssd_blocks=4,
            calendar=calendar)
        node1 = self.make_lifecycle(
            "ssd_direct", node_id=1,
            cpu_blocks=2, ssd_blocks=4,
            calendar=calendar)
        node0.register_d_ready("a", 16, now_ns=0)
        node1.register_d_ready("b", 16, now_ns=0)
        job0 = node0.demote("a", now_ns=0)
        job1 = node1.demote("b", now_ns=0)
        self.assertEqual(job0.start_ns, job1.start_ns)
        self.assertEqual(job0.completion_ns, job1.completion_ns)
        resources0 = {
            demand.resource
            for stage in job0.stages
            for demand in stage.stage.demands
        }
        resources1 = {
            demand.resource
            for stage in job1.stages
            for demand in stage.stage.demands
        }
        self.assertTrue(resources0.isdisjoint(resources1))

    def test_randomized_small_capacity_stress_conserves_every_ledger(self):
        rng = random.Random(20260723)
        for policy in (
                "hbm_lru_recompute",
                "ssd_direct",
                "cpu_ssd"):
            lifecycle = self.make_lifecycle(
                policy, blocks=12,
                cpu_blocks=6, ssd_blocks=12)
            now_ns = 0
            for index in range(40):
                session_id = f"{policy}-{index}"
                tokens = rng.choice((16, 32))
                needed = self.hardware.kv_capacity_bytes_per_rank(
                    tokens)
                while lifecycle.d_ledger.free_bytes < needed:
                    event_ns = lifecycle.ensure_d_headroom(
                        needed, now_ns=now_ns)
                    self.assertIsNotNone(event_ns)
                    now_ns = event_ns
                    lifecycle.advance(now_ns)
                lifecycle.register_d_ready(
                    session_id, tokens, now_ns=now_ns)
                if rng.random() < 0.75:
                    job = lifecycle.demote(
                        session_id, now_ns=now_ns)
                    if job is None and (
                            lifecycle.sessions[session_id].state
                            == TierSessionState.D_READY):
                        event_ns = lifecycle.next_event_ns()
                        if event_ns is not None:
                            now_ns = event_ns
                            lifecycle.advance(now_ns)
                            job = lifecycle.demote(
                                session_id, now_ns=now_ns)
                    if job is not None:
                        if rng.random() < 0.35:
                            resume_ns = min(
                                job.completion_ns,
                                job.start_ns + 1,
                            )
                        else:
                            resume_ns = job.completion_ns
                        now_ns = max(now_ns, resume_ns)
                record = lifecycle.sessions[session_id]
                if record.state == TierSessionState.D_READY:
                    lifecycle.demote(
                        session_id, now_ns=now_ns)
                source = lifecycle.peek_resume_source(
                    session_id, now_ns=now_ns)
                ticket = lifecycle.begin_prepare(
                    session_id,
                    request_id=index + 1,
                    now_ns=now_ns,
                    input_tokens=tokens + 1,
                    output_tokens=1,
                    reusable_tokens=tokens,
                    has_successor=False,
                )
                if ticket is None:
                    lifecycle.end(session_id, now_ns=now_ns)
                else:
                    now_ns = ticket.completion_ns
                    lifecycle.advance(now_ns)
                    lifecycle.pop_prepare_completed()
                    lifecycle.mark_active(
                        ticket, now_ns=now_ns)
                    lifecycle.commit_d_ready(
                        ticket,
                        now_ns=now_ns,
                        has_successor=False,
                    )
                lifecycle.assert_invariants()
            lifecycle.run_until_idle()
            lifecycle.assert_invariants()
            self.assertEqual(lifecycle.p_ledger.used_bytes, 0)
            self.assertEqual(lifecycle.d_ledger.used_bytes, 0)
            self.assertEqual(lifecycle.cpu_ledger.used_bytes, 0)
            self.assertEqual(lifecycle.ssd_ledger.used_bytes, 0)
            self.assertEqual(lifecycle.copies, {})
            self.assertEqual(lifecycle.next_event_ns(), None)
            self.assertLessEqual(
                lifecycle.d_ledger.peak_used_bytes,
                lifecycle.d_ledger.capacity_bytes,
            )
            self.assertLessEqual(
                lifecycle.cpu_ledger.peak_used_bytes,
                lifecycle.cpu_ledger.capacity_bytes,
            )
            self.assertLessEqual(
                lifecycle.ssd_ledger.peak_used_bytes,
                lifecycle.ssd_ledger.capacity_bytes,
            )

    def test_report_declares_sole_hbm_ledger_and_tie_order(self):
        lifecycle = self.make_lifecycle("cpu_ssd")
        report = lifecycle.report()
        json.dumps(report)
        self.assertTrue(report["validate_every_event"])
        self.assertIn("sole", report["d_hbm_integration"])
        self.assertIn(
            "must not", report["d_hbm_integration"])
        self.assertEqual(
            report["completion_order"],
            "transfer_completion_before_same_timestamp_arrival",
        )

    def test_constructor_rejects_unknown_policy(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.make_lifecycle("unknown")
        with self.assertRaisesRegex(
                ValueError, "validate_every_event"):
            self.make_lifecycle(
                "cpu_ssd", validate_every_event=1)


if __name__ == "__main__":
    unittest.main()

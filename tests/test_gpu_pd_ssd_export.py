from pathlib import Path
import unittest

from serving.core.gpu_pd_latency import load_p4d4_gpu_config
from serving.core.gpu_pd_tier_lifecycle import (
    SSDExportStatus,
    TierJobStatus,
    TierSessionState,
    TieredPDKVLifecycle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_CONFIG = (
    REPO_ROOT / "configs" / "wakekv_hbf"
    / "p4d4_gpu_server.json"
)


class SSDExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = load_p4d4_gpu_config(GPU_CONFIG)
        cls.block_per_rank = (
            cls.hardware.kv_capacity_bytes_per_rank(16))
        cls.block_aggregate = (
            cls.block_per_rank * cls.hardware.tp_size)

    def make_lifecycle(
            self, *, cpu_blocks=8,
            d_blocks=8, ssd_blocks=8):
        return TieredPDKVLifecycle(
            hardware=self.hardware,
            node_id=0,
            policy="ssd_direct",
            p_capacity_bytes_per_rank=(
                d_blocks * self.block_per_rank),
            d_capacity_bytes_per_rank=(
                d_blocks * self.block_per_rank),
            cpu_capacity_bytes=(
                cpu_blocks * self.block_aggregate),
            ssd_capacity_bytes=(
                ssd_blocks * self.block_aggregate),
        )

    def make_ssd_ready(self, lifecycle, *, session_id="s"):
        lifecycle.register_d_ready(
            session_id, 16, now_ns=lifecycle.current_ns)
        demotion = lifecycle.demote(
            session_id, now_ns=lifecycle.current_ns)
        self.assertIsNotNone(demotion)
        lifecycle.advance(demotion.completion_ns)
        self.assertEqual(
            lifecycle.sessions[session_id].state,
            TierSessionState.SSD_READY,
        )
        return lifecycle.sessions[session_id]

    @staticmethod
    def finish_prepare(lifecycle, ticket):
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
            has_successor=ticket.has_successor,
        )

    def test_exact_completion_finish_releases_pin_and_bounce(self):
        lifecycle = self.make_lifecycle()
        record = self.make_ssd_ready(lifecycle)
        source_id = record.primary_copy_id
        source = lifecycle.copies[source_id]
        ssd_bytes = lifecycle.ssd_ledger.used_bytes

        ticket = lifecycle.begin_ssd_export(
            "s", now_ns=lifecycle.current_ns)
        self.assertIsNotNone(ticket)
        self.assertEqual(ticket.transfer_kinds, ("ssd-to-cpu",))
        self.assertFalse(ticket.physical_complete)
        self.assertTrue(
            lifecycle.ssd_export_publication_valid(ticket))
        self.assertEqual(source.export_pins, 1)
        self.assertEqual(
            lifecycle.cpu_ledger.used_bytes, ticket.byte_count)

        lifecycle.finish_ssd_export(
            ticket, now_ns=ticket.completion_ns)

        self.assertEqual(ticket.status, SSDExportStatus.FINISHED)
        self.assertTrue(ticket.physical_complete)
        self.assertTrue(ticket.resources_released)
        self.assertEqual(
            lifecycle.jobs[ticket.job_id].status,
            TierJobStatus.COMPLETE,
        )
        self.assertEqual(source.export_pins, 0)
        self.assertEqual(lifecycle.cpu_ledger.used_bytes, 0)
        self.assertEqual(lifecycle.ssd_ledger.used_bytes, ssd_bytes)
        lifecycle.assert_invariants()

    def test_resume_during_read_invalidates_but_does_not_cancel_io(self):
        lifecycle = self.make_lifecycle(cpu_blocks=8)
        record = self.make_ssd_ready(lifecycle)
        source_id = record.primary_copy_id
        source = lifecycle.copies[source_id]
        export = lifecycle.begin_ssd_export(
            "s", now_ns=lifecycle.current_ns)
        resume_ns = export.completion_ns - 1

        prepare = lifecycle.begin_prepare(
            "s",
            request_id=1,
            now_ns=resume_ns,
            input_tokens=17,
            output_tokens=2,
            reusable_tokens=16,
            has_successor=True,
        )

        self.assertIsNotNone(prepare)
        self.assertFalse(
            lifecycle.ssd_export_publication_valid(export))
        self.assertEqual(source.export_pins, 1)
        self.assertEqual(source.foreground_pins, 1)
        self.assertFalse(
            lifecycle.abort_ssd_export(
                export, now_ns=resume_ns))
        self.assertEqual(
            export.status, SSDExportStatus.ABORT_PENDING)
        self.assertGreater(lifecycle.cpu_ledger.used_bytes, 0)
        self.assertEqual(
            lifecycle.jobs[export.job_id].status,
            TierJobStatus.RUNNING,
        )

        lifecycle.advance(export.completion_ns)

        self.assertEqual(export.status, SSDExportStatus.ABORTED)
        self.assertTrue(export.resources_released)
        self.assertEqual(source.export_pins, 0)
        self.assertEqual(source.foreground_pins, 1)
        self.assertEqual(
            lifecycle.jobs[export.job_id].status,
            TierJobStatus.COMPLETE,
        )
        self.finish_prepare(lifecycle, prepare)
        self.assertNotIn(source_id, lifecycle.copies)
        self.assertEqual(lifecycle.cpu_ledger.used_bytes, 0)
        self.assertEqual(lifecycle.ssd_ledger.used_bytes, 0)
        lifecycle.assert_invariants()

    def test_resume_at_exact_read_completion_invalidates_ready_export(self):
        lifecycle = self.make_lifecycle(cpu_blocks=8)
        self.make_ssd_ready(lifecycle)
        export = lifecycle.begin_ssd_export(
            "s", now_ns=lifecycle.current_ns)

        prepare = lifecycle.begin_prepare(
            "s",
            request_id=2,
            now_ns=export.completion_ns,
            input_tokens=17,
            output_tokens=1,
            reusable_tokens=16,
            has_successor=False,
        )

        self.assertEqual(export.status, SSDExportStatus.READY)
        self.assertTrue(export.physical_complete)
        self.assertFalse(
            lifecycle.ssd_export_publication_valid(export))
        self.assertTrue(
            lifecycle.abort_ssd_export(
                export, now_ns=export.completion_ns))
        self.assertEqual(export.status, SSDExportStatus.ABORTED)
        self.finish_prepare(lifecycle, prepare)
        self.assertEqual(lifecycle.cpu_ledger.used_bytes, 0)
        self.assertEqual(lifecycle.ssd_ledger.used_bytes, 0)
        lifecycle.assert_invariants()

    def test_inflight_abort_defers_release_until_physical_completion(self):
        lifecycle = self.make_lifecycle()
        record = self.make_ssd_ready(lifecycle)
        source = lifecycle.copies[record.primary_copy_id]
        ticket = lifecycle.begin_ssd_export(
            "s", now_ns=lifecycle.current_ns)

        self.assertFalse(
            lifecycle.abort_ssd_export(
                ticket, now_ns=lifecycle.current_ns))
        self.assertEqual(
            ticket.status, SSDExportStatus.ABORT_PENDING)
        self.assertEqual(source.export_pins, 1)
        self.assertEqual(
            lifecycle.cpu_ledger.used_bytes, ticket.byte_count)

        lifecycle.advance(ticket.completion_ns)

        self.assertEqual(ticket.status, SSDExportStatus.ABORTED)
        self.assertEqual(source.export_pins, 0)
        self.assertEqual(lifecycle.cpu_ledger.used_bytes, 0)
        self.assertGreater(
            lifecycle.calendar.reservation_bytes_by_namespace_kind[
                ("gpu-tier-node-0", "ssd-to-cpu")],
            0,
        )
        lifecycle.assert_invariants()

    def test_cpu_capacity_deferral_is_state_and_calendar_pure(self):
        lifecycle = self.make_lifecycle(cpu_blocks=1)
        record = self.make_ssd_ready(lifecycle)
        source = lifecycle.copies[record.primary_copy_id]
        self.assertTrue(lifecycle.cpu_ledger.reserve(
            "test:block-export", self.block_aggregate))
        before_generation = record.generation
        before_jobs = dict(lifecycle.jobs)
        before_reservations = tuple(
            lifecycle.calendar.reservations)
        before_next_event = lifecycle.next_event_ns()

        ticket = lifecycle.begin_ssd_export(
            "s", now_ns=lifecycle.current_ns)

        self.assertIsNone(ticket)
        self.assertEqual(record.state, TierSessionState.SSD_READY)
        self.assertEqual(record.generation, before_generation)
        self.assertEqual(source.export_pins, 0)
        self.assertEqual(lifecycle.jobs, before_jobs)
        self.assertEqual(
            tuple(lifecycle.calendar.reservations),
            before_reservations,
        )
        self.assertEqual(
            lifecycle.next_event_ns(), before_next_event)
        self.assertEqual(lifecycle.ssd_exports, {})
        self.assertEqual(
            lifecycle.metrics.ssd_export_capacity_deferrals, 1)
        lifecycle.cpu_ledger.release("test:block-export")
        lifecycle.assert_invariants()


if __name__ == "__main__":
    unittest.main()

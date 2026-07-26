import unittest

from serving.core.hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
)
from serving.core.hbf_full_model_lifecycle import (
    FullModelHBFLifecycle,
    MigrationSourceKind,
    PlacementState,
    ResourceCalendar,
    ResumeExecution,
    hbf_kv_range_card_bytes,
)


class HBFSSDImportTests(unittest.TestCase):
    def make_manager(
            self, layout="tp4", *, hardware=None,
            kv_bytes_per_token=32,
            model_weight_bytes_per_rank=None,
            calendar=None):
        return FullModelHBFLifecycle(
            hardware=hardware or HBFServerHardware(),
            layout=HBFParallelLayout.for_key(layout),
            resource_calendar=calendar,
            kv_bytes_per_token=kv_bytes_per_token,
            model_weight_bytes_per_rank=model_weight_bytes_per_rank,
        )

    def publish_checkpoint(
            self, manager, session_id, *,
            now_ns, total_tokens):
        record = manager.register_session(
            session_id, now_ns=now_ns)
        result = manager.complete_gpu_turn(
            session_id,
            now_ns=now_ns,
            total_tokens=total_tokens,
            has_successor=True,
            start_migration=False,
        )
        self.assertIsNone(result)
        version = record.version
        generation = record.generation
        self.assertTrue(manager.publish_ssd_checkpoint(
            session_id,
            now_ns=now_ns,
            snapshot_version=version,
        ))
        self.assertEqual(record.state, PlacementState.SSD_READY)
        self.assertEqual(record.version, version)
        self.assertEqual(record.generation, generation)
        self.assertEqual(record.gpu_retained_bytes, 0)
        return record

    def test_tp4_import_uses_one_replica_and_cpu_nic_source(self):
        calendar = ResourceCalendar()
        manager = self.make_manager(calendar=calendar)
        record = self.publish_checkpoint(
            manager, "s", now_ns=10, total_tokens=100)

        job = manager.start_import_from_ssd("s", now_ns=20)

        self.assertIsNotNone(job)
        self.assertEqual(job.source_kind, MigrationSourceKind.SSD)
        self.assertEqual(job.group_id, 0)
        self.assertEqual(
            tuple(card_id for card_id, _ in job.card_bytes),
            (0, 1, 2, 3),
        )
        self.assertEqual(
            manager.report()["pending_migration_jobs"][0][
                "source_kind"],
            "ssd",
        )
        resources = set(calendar.available_ns)
        self.assertIn("gpu-node-0-cpu-dram", resources)
        self.assertIn("gpu-node-0-rdma-nic", resources)
        self.assertIn("rdma-network", resources)
        self.assertIn("hbf-card-0-media", resources)
        self.assertNotIn("gpu-source-pcie-root", resources)
        self.assertFalse(any("ssd" in resource for resource in resources))
        self.assertEqual(
            manager._reserved_per_card_by_group[1], 0)

        manager.advance(job.completion_ns)
        self.assertEqual(record.state, PlacementState.HBF_READY)
        self.assertEqual(record.committed_hbf_tokens, 100)
        self.assertEqual(manager.metrics.ssd_imports_committed, 1)

    def test_tp8_context_import_keeps_exact_card_vector(self):
        manager = self.make_manager(
            "tp8_context", kv_bytes_per_token=8)
        self.publish_checkpoint(
            manager, "s", now_ns=0, total_tokens=3)

        job = manager.start_import_from_ssd("s", now_ns=1)

        self.assertIsNotNone(job)
        expected = hbf_kv_range_card_bytes(
            layout=manager.layout,
            card_ids=tuple(range(8)),
            kv_bytes_per_token=8,
            token_start=0,
            token_count=3,
        )
        self.assertEqual(dict(job.card_bytes), expected)
        self.assertEqual(
            tuple(expected.values()),
            (4, 2, 4, 2, 4, 2, 4, 2),
        )
        self.assertEqual(job.physical_bytes, job.logical_bytes)
        manager.advance(job.completion_ns)
        self.assertEqual(
            manager.report()["group_reserved_bytes_by_card"][0],
            expected,
        )

    def test_capacity_deferral_preserves_ssd_and_hbf_placements(self):
        hardware = HBFServerHardware(
            hbf_capacity_bytes_per_card=100)
        manager = self.make_manager(
            hardware=hardware,
            kv_bytes_per_token=4,
            model_weight_bytes_per_rank=0,
        )
        first = self.publish_checkpoint(
            manager, "first", now_ns=0, total_tokens=100)
        first_job = manager.start_import_from_ssd(
            "first", now_ns=0)
        manager.advance(first_job.completion_ns)
        second = self.publish_checkpoint(
            manager,
            "second",
            now_ns=first_job.completion_ns,
            total_tokens=100,
        )
        second_job = manager.start_import_from_ssd(
            "second", now_ns=first_job.completion_ns)
        manager.advance(second_job.completion_ns)
        deferred = self.publish_checkpoint(
            manager,
            "deferred",
            now_ns=second_job.completion_ns,
            total_tokens=100,
        )

        before = manager.report()["group_reserved_bytes_by_card"]
        job = manager.start_import_from_ssd(
            "deferred", now_ns=second_job.completion_ns)

        self.assertIsNone(job)
        self.assertEqual(deferred.state, PlacementState.SSD_READY)
        self.assertEqual(first.state, PlacementState.HBF_READY)
        self.assertEqual(second.state, PlacementState.HBF_READY)
        self.assertEqual(
            manager.report()["group_reserved_bytes_by_card"],
            before,
        )
        self.assertEqual(manager.metrics.capacity_evictions, 0)
        self.assertEqual(manager.metrics.ssd_imports_started, 2)

    def test_resume_from_ssd_ready_has_distinct_restore_route(self):
        manager = self.make_manager()
        record = self.publish_checkpoint(
            manager, "s", now_ns=0, total_tokens=100)
        generation = record.generation

        route = manager.route_resume(
            "s",
            now_ns=10,
            request_id=7,
            prefix_reuse_tokens=90,
            input_tokens=100,
        )

        self.assertEqual(route.execution, ResumeExecution.GPU_RESTORE)
        self.assertFalse(route.migration_inflight)
        self.assertEqual(route.reason, "ssd_checkpoint_gpu_restore")
        self.assertEqual(record.state, PlacementState.GPU_ACTIVE)
        self.assertGreater(record.generation, generation)
        self.assertEqual(record.total_tokens, 90)
        self.assertEqual(record.gpu_retained_bytes, 90 * 32)
        self.assertEqual(manager.metrics.ssd_restore_resumes, 1)

    def test_inflight_resume_invalidates_import_and_accounts_waste(self):
        manager = self.make_manager()
        record = self.publish_checkpoint(
            manager, "s", now_ns=0, total_tokens=10_000)
        job = manager.start_import_from_ssd("s", now_ns=1)
        generation = record.generation

        route = manager.route_resume(
            "s",
            now_ns=job.completion_ns - 1,
            request_id=9,
        )

        self.assertEqual(route.execution, ResumeExecution.GPU_RESTORE)
        self.assertTrue(route.migration_inflight)
        self.assertEqual(record.state, PlacementState.GPU_ACTIVE)
        self.assertGreater(record.generation, generation)
        self.assertEqual(record.group_id, None)
        self.assertEqual(
            record.gpu_retained_bytes, job.logical_bytes)
        self.assertEqual(
            manager.metrics.ssd_import_wasted_physical_bytes, 0)

        manager.advance(job.completion_ns)
        self.assertEqual(manager.metrics.migrations_stale, 1)
        self.assertEqual(manager.metrics.ssd_imports_stale, 1)
        self.assertEqual(
            manager.metrics.migration_wasted_physical_bytes,
            job.physical_bytes,
        )
        self.assertEqual(
            manager.metrics.ssd_import_wasted_physical_bytes,
            job.physical_bytes,
        )
        self.assertEqual(record.pending_reserved_per_card_bytes, 0)
        self.assertTrue(all(
            byte_count == 0
            for group in manager._reserved_bytes_by_card.values()
            for byte_count in group.values()
        ))

    def test_exact_import_completion_tie_routes_to_hbf(self):
        manager = self.make_manager()
        record = self.publish_checkpoint(
            manager, "s", now_ns=0, total_tokens=1_000)
        job = manager.start_import_from_ssd("s", now_ns=1)

        route = manager.route_resume(
            "s", now_ns=job.completion_ns, request_id=11)

        self.assertEqual(route.execution, ResumeExecution.HBF)
        self.assertEqual(route.reason, "hbf_ready")
        self.assertEqual(record.state, PlacementState.HBF_ACTIVE)
        self.assertEqual(manager.metrics.ssd_imports_committed, 1)
        self.assertEqual(manager.metrics.ssd_restore_resumes, 0)

    def test_ended_inflight_import_releases_destination_on_completion(self):
        manager = self.make_manager()
        record = self.publish_checkpoint(
            manager, "s", now_ns=0, total_tokens=1_000)
        job = manager.start_import_from_ssd("s", now_ns=1)

        manager.end_session("s", now_ns=job.completion_ns - 1)
        manager.advance(job.completion_ns)

        self.assertEqual(record.state, PlacementState.ENDED)
        self.assertEqual(record.pending_reserved_per_card_bytes, 0)
        self.assertEqual(manager.report()["pending_job_count"], 0)
        self.assertTrue(all(
            byte_count == 0
            for group in manager._reserved_bytes_by_card.values()
            for byte_count in group.values()
        ))
        manager.assert_invariants()


if __name__ == "__main__":
    unittest.main()

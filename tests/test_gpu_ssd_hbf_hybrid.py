import dataclasses
from pathlib import Path
import unittest

from serving.core.gpu_hbf_hybrid import (
    HybridCall,
    HybridExecution,
)
from serving.core.gpu_pd_tier_lifecycle import (
    SSDExportStatus,
    Tier,
    TierSessionState,
)
from serving.core.gpu_pd_latency import P4D4GPUHardware
from serving.core.gpu_ssd_hbf_hybrid import (
    SSDPromotionPolicy,
    SSDStagedGPUHBFNode,
    SSDStagedGPUHBFSystem,
)
from serving.core.hbf_comparison_workload import (
    CallSpec,
    ScheduledSession,
    SessionSpec,
)
from serving.core.hbf_full_model_lifecycle import PlacementState
from serving.core.hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
)
from serving.core.hbf_full_model_pool import derive_lpddr_workspace_bytes


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_schedule(
        source_index, calls, *,
        arrival_ns=0, session_id=None):
    resolved_session_id = (
        f"ssd-staged-{source_index}"
        if session_id is None else session_id
    )
    call_specs = tuple(
        CallSpec(
            session_id=resolved_session_id,
            source_index=source_index,
            call_index=call_index,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_duration_ns=tool_duration_ns,
            cached_prefix_tokens=cached_prefix_tokens,
            fresh_input_tokens=(
                input_tokens - cached_prefix_tokens),
            lineage_status=None,
            inter_turn_gap_type=gap_type,
        )
        for call_index, (
            input_tokens,
            output_tokens,
            tool_duration_ns,
            cached_prefix_tokens,
            gap_type,
        ) in enumerate(calls)
    )
    return ScheduledSession(
        offer_index=source_index,
        session=SessionSpec(
            source_index=source_index,
            session_id=resolved_session_id,
            source_arrival_time_ns=arrival_ns,
            source_session_identity_sha256=None,
            calls=call_specs,
        ),
        arrival_time_ns=arrival_ns,
        unit_interarrival=0.0,
        unit_arrival_time=0.0,
    )


class SSDStagedGPUHBFTests(unittest.TestCase):
    @staticmethod
    def make_node(
            *, policy="eager", layout="tp4",
            restore_execution_mode="bulk"):
        return SSDStagedGPUHBFNode(
            repo_root=REPO_ROOT,
            gpu_hardware=P4D4GPUHardware(),
            hbf_hardware=HBFServerHardware(),
            hbf_layout=layout,
            max_num_batched_tokens=256,
            max_num_seqs=16,
            max_prefill_chunk_tokens=128,
            restore_execution_mode=restore_execution_mode,
            promotion_policy=policy,
        )

    @staticmethod
    def make_system(
            *, policy="eager", layout="tp4",
            restore_execution_mode="bulk"):
        return SSDStagedGPUHBFSystem(
            repo_root=REPO_ROOT,
            hbf_layout=layout,
            max_num_batched_tokens=256,
            max_num_seqs=16,
            max_prefill_chunk_tokens=128,
            restore_execution_mode=restore_execution_mode,
            promotion_policy=policy,
        )

    @staticmethod
    def runtime_call(
            request_id, call_index, release_ns, *,
            input_tokens, output_tokens=1,
            prefix_reuse_tokens=0, has_successor=False,
            session_id="session"):
        return HybridCall(
            request_id=request_id,
            session_id=session_id,
            call_index=call_index,
            release_ns=release_ns,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prefix_reuse_tokens=prefix_reuse_tokens,
            has_successor=has_successor,
        )

    def test_exactly_one_gpu_and_one_eight_card_hbf_share_calendar(self):
        node = self.make_node(layout="tp8_context")

        self.assertEqual(node.gpu_hardware.gpu_count, 8)
        self.assertEqual(node.hbf_server_count, 1)
        self.assertEqual(node.hbf_hardware.card_count, 8)
        self.assertEqual(node.hbf_layout.key, "tp8_context")
        self.assertIs(node.calendar, node.gpu_node.calendar)
        self.assertIs(node.calendar, node.gpu_lifecycle.calendar)

    def test_gpu_restore_mode_propagates_without_changing_hbf_import(self):
        node = self.make_node(
            restore_execution_mode="layerwise_streaming")
        system = self.make_system(
            restore_execution_mode="layerwise_streaming")

        self.assertEqual(
            node.restore_execution_mode,
            "layerwise_streaming",
        )
        self.assertEqual(
            node.gpu_node.restore_execution_mode,
            "layerwise_streaming",
        )
        self.assertEqual(
            node.report()["architecture"][
                "gpu_restore_execution_mode"],
            "layerwise_streaming",
        )
        report = system.report()
        self.assertEqual(
            report["architecture"]["gpu_restore_execution_mode"],
            "layerwise_streaming",
        )
        self.assertNotIn(
            "restore_execution_mode",
            node.hbf_lifecycle.report(),
        )
        self.assertIs(node.calendar, node.gpu_pool.calendar)
        self.assertIs(node.calendar, node.hbf_lifecycle.calendar)
        self.assertIs(node.calendar, node.hbf_pool.calendar)
        self.assertEqual(node.hbf_execution_backend,
                         "analytical_calendar")
        self.assertFalse(node.has_pending_external())

    def test_first_gpu_turn_proactively_publishes_local_ssd(self):
        node = self.make_node(policy="delay_1s")
        node.set_gap_type(0, "tool")
        first = self.runtime_call(
            0,
            0,
            0,
            input_tokens=64,
            output_tokens=2,
            has_successor=True,
        )
        node.submit(first, now_ns=0)

        while (
            node.hbf_lifecycle.sessions["session"].state
            != PlacementState.SSD_READY
        ):
            event_ns = node.next_event_ns()
            self.assertIsNotNone(event_ns)
            node.advance(event_ns)

        self.assertEqual(
            first.execution, HybridExecution.GPU_FIRST_TURN)
        self.assertEqual(
            node.gpu_lifecycle.sessions["session"].state,
            TierSessionState.SSD_READY,
        )
        self.assertEqual(
            node.hbf_lifecycle.sessions["session"].state,
            PlacementState.SSD_READY,
        )
        self.assertEqual(
            node.gpu_lifecycle.metrics.d_to_ssd_started, 1)
        demotions = [
            job for job in node.gpu_lifecycle.jobs.values()
            if job.transfer_kinds
            == ("d-gpu_to_cpu", "cpu-to-ssd")
        ]
        self.assertEqual(len(demotions), 1)
        self.assertEqual(
            node.hbf_lifecycle.metrics.ssd_imports_started, 0)
        self.assertFalse(any(
            reservation.resource == "rdma-network"
            for reservation in node.calendar.reservations
        ))

    def test_short_wait_restores_ssd_to_gpu_without_rdma(self):
        system = self.make_system(policy="delay_1s")
        system.run((
            make_schedule(
                0,
                (
                    (64, 2, 100_000_000, 0, "tool"),
                    (67, 1, 0, 65, None),
                ),
            ),
        ))

        first, second = system.node.calls[0], system.node.calls[1]
        tier_second = system.node._tier_call_by_request[1]
        self.assertEqual(
            first.execution, HybridExecution.GPU_FIRST_TURN)
        self.assertEqual(
            second.execution, HybridExecution.GPU_OWNED)
        self.assertEqual(
            second.route_reason, "ssd_checkpoint_gpu_restore")
        self.assertEqual(tier_second.prepare_source, Tier.SSD)
        self.assertEqual(
            system.node.metrics.hbf_imports_started, 0)
        self.assertFalse(any(
            reservation.resource == "rdma-network"
            for reservation in system.node.calendar.reservations
        ))
        self.assert_gpu_tiers_empty(system.node)

    def test_long_wait_promotes_ssd_to_hbf(self):
        system = self.make_system(policy="tool_immediate")
        system.run((
            make_schedule(
                0,
                (
                    (64, 2, 10_000_000_000, 0, "tool"),
                    (67, 1, 0, 65, None),
                ),
            ),
        ))

        first, second = system.node.calls[0], system.node.calls[1]
        self.assertEqual(
            first.execution, HybridExecution.GPU_FIRST_TURN)
        self.assertEqual(
            second.execution, HybridExecution.HBF_READY)
        self.assertEqual(
            system.node.metrics.ssd_checkpoints_published, 1)
        self.assertEqual(
            system.node.metrics.ssd_exports_started, 1)
        self.assertEqual(
            system.node.metrics.hbf_imports_started, 1)
        self.assertEqual(
            system.node.metrics.hbf_imports_committed, 1)
        self.assertEqual(
            system.node.hbf_lifecycle.metrics.ssd_imports_committed,
            1,
        )
        self.assertTrue(any(
            reservation.resource == "rdma-network"
            for reservation in system.node.calendar.reservations
        ))
        self.assert_gpu_tiers_empty(system.node)
        self.assertEqual(
            system.node.hbf_lifecycle.sessions[
                "ssd-staged-0"].state,
            PlacementState.ENDED,
        )

    def test_no_rdma_before_fixed_delay(self):
        node = self.make_node(policy="delay_1s")
        node.set_gap_type(0, "human")
        first = self.runtime_call(
            0,
            0,
            0,
            input_tokens=64,
            output_tokens=2,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        while first.user_completion_ns is None:
            event_ns = node.next_event_ns()
            self.assertIsNotNone(event_ns)
            node.advance(event_ns)
        threshold_ns = first.user_completion_ns + 1_000_000_000
        node.advance(threshold_ns - 1)

        self.assertEqual(
            node.hbf_lifecycle.metrics.ssd_imports_started, 0)
        self.assertFalse(any(
            reservation.resource == "rdma-network"
            for reservation in node.calendar.reservations
        ))

    def test_resume_during_hbf_import_invalidates_to_gpu_restore(self):
        node = self.make_node(policy="tool_immediate")
        node.set_gap_type(0, "tool")
        first = self.runtime_call(
            0,
            0,
            0,
            input_tokens=1_000,
            output_tokens=2,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        while "session" not in node._import_by_session:
            event_ns = node.next_event_ns()
            self.assertIsNotNone(event_ns)
            node.advance(event_ns)
        import_job, export = node._import_by_session["session"]
        resume_ns = (
            max(node.current_ns, import_job.start_ns)
            + import_job.completion_ns
        ) // 2
        self.assertLess(resume_ns, import_job.completion_ns)
        second = self.runtime_call(
            1,
            1,
            resume_ns,
            input_tokens=1_002,
            output_tokens=1,
            prefix_reuse_tokens=1_001,
            has_successor=False,
        )
        node.submit(second, now_ns=resume_ns)

        self.assertEqual(
            second.execution,
            HybridExecution.GPU_MIGRATION_INFLIGHT,
        )
        self.assertEqual(
            second.route_reason,
            "ssd_import_inflight_gpu_restore",
        )
        self.assertIn(
            export.status,
            {SSDExportStatus.ABORTED,
             SSDExportStatus.ABORT_PENDING},
        )
        node.assert_invariants()
        node.run_until_idle()
        self.assertEqual(node.metrics.hbf_imports_committed, 0)
        self.assertEqual(node.metrics.hbf_imports_stale, 1)
        self.assertEqual(
            node.hbf_lifecycle.metrics.ssd_imports_stale, 1)
        self.assert_gpu_tiers_empty(node)

    def test_resume_at_exact_delay_threshold_wins_without_export(self):
        delay_ns = 100_000_000
        system = self.make_system(policy="delay_100ms")
        system.run((
            make_schedule(
                0,
                (
                    (64, 2, delay_ns, 0, "tool"),
                    (67, 1, 0, 65, None),
                ),
            ),
        ))

        first, second = system.node.calls[0], system.node.calls[1]
        self.assertEqual(
            second.release_ns,
            first.user_completion_ns + delay_ns,
        )
        self.assertEqual(
            second.execution, HybridExecution.GPU_OWNED)
        self.assertEqual(
            second.route_reason, "ssd_checkpoint_gpu_restore")
        self.assertEqual(
            system.node.metrics.ssd_exports_started, 0)
        self.assertEqual(
            system.node.metrics.hbf_imports_started, 0)
        self.assertEqual(
            system.node.metrics.promotion_intents_canceled, 1)

    def test_delayed_gpu_admission_preserves_logical_release_ttft(self):
        system = self.make_system(policy="never")
        completed = system.run((
            make_schedule(
                77,
                (
                    (1_000, 1, 0, 0, "tool"),
                    (1_001, 1, 0, 1_000, None),
                ),
                session_id="late-tier-submit",
            ),
        ))

        first = system.node.calls[0]
        second = system.node.calls[1]
        tier_second = system.node._tier_call_by_request[1]
        completed_second = next(
            request for request in completed
            if request.key.sub_request_index == 1
        )
        self.assertLess(
            first.user_completion_ns,
            first.internal_completion_ns,
        )
        self.assertEqual(
            second.release_ns,
            first.user_completion_ns,
        )
        self.assertEqual(
            tier_second.release_ns,
            first.internal_completion_ns,
        )
        self.assertGreater(
            tier_second.release_ns,
            second.release_ns,
        )
        self.assertEqual(
            second.gpu_request.arrival_ns,
            second.release_ns,
        )
        self.assertEqual(
            completed_second.release_ns,
            second.release_ns,
        )
        expected_ttft_ns = (
            second.gpu_request.first_token_ns
            - second.release_ns
        )
        self.assertEqual(second.ttft_ns, expected_ttft_ns)
        self.assertEqual(
            completed_second.ttft_ns,
            expected_ttft_ns,
        )

    def test_hbf_import_commit_wins_exact_resume_tie(self):
        node = self.make_node(policy="tool_immediate")
        node.set_gap_type(0, "tool")
        first = self.runtime_call(
            0,
            0,
            0,
            input_tokens=1_000,
            output_tokens=2,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        while "session" not in node._import_by_session:
            event_ns = node.next_event_ns()
            self.assertIsNotNone(event_ns)
            node.advance(event_ns)
        import_job, _ = node._import_by_session["session"]
        second = self.runtime_call(
            1,
            1,
            import_job.completion_ns,
            input_tokens=1_002,
            output_tokens=1,
            prefix_reuse_tokens=1_001,
            has_successor=False,
        )
        node.submit(second, now_ns=import_job.completion_ns)

        self.assertEqual(
            second.execution, HybridExecution.HBF_READY)
        self.assertEqual(node.metrics.hbf_imports_committed, 1)
        self.assertEqual(node.metrics.hbf_imports_stale, 0)
        node.run_until_idle()
        self.assert_gpu_tiers_empty(node)

    def test_gpu_hbf_gpu_restart_uses_gpu_local_call_index(self):
        layout = HBFParallelLayout.for_key("tp8")
        workspace = derive_lpddr_workspace_bytes(
            layout,
            max_num_batched_tokens=16,
            max_num_seqs=4,
        )
        per_token_per_card = 98_304 * 2 // 8
        hardware = dataclasses.replace(
            HBFServerHardware(),
            lpddr_capacity_bytes_per_card=(
                workspace + 4 * per_token_per_card),
        )
        system = SSDStagedGPUHBFSystem(
            repo_root=REPO_ROOT,
            hbf_hardware=hardware,
            hbf_layout="tp8",
            max_num_batched_tokens=16,
            max_num_seqs=4,
            max_prefill_chunk_tokens=16,
            promotion_policy="tool_immediate",
        )
        system.run((
            make_schedule(
                0,
                (
                    (1, 1, 10_000_000_000, 0, "tool"),
                    (1, 2, 10_000_000_000, 1, "tool"),
                    (2, 6, 0, 2, None),
                ),
            ),
        ))

        calls = system.node.calls
        self.assertEqual(
            tuple(calls[index].call_index for index in range(3)),
            (0, 1, 2),
        )
        self.assertEqual(
            tuple(calls[index].execution for index in range(3)),
            (
                HybridExecution.GPU_FIRST_TURN,
                HybridExecution.HBF_READY,
                HybridExecution.GPU_RECOMPUTE,
            ),
        )
        self.assertEqual(
            calls[2].route_reason,
            "hbf_lpddr_finish_capacity_fallback",
        )
        self.assertEqual(
            system.node._tier_call_by_request[0].call_index, 0)
        self.assertNotIn(1, system.node._tier_call_by_request)
        self.assertEqual(
            system.node._tier_call_by_request[2].call_index, 0)
        self.assertEqual(
            system.node._tier_call_by_request[2].session_id,
            "ssd-staged-0",
        )
        self.assertEqual(
            system.node.gpu_node.metrics.session_restarts, 1)
        self.assertEqual(
            system.node.gpu_lifecycle.metrics.session_restarts, 1)
        self.assertEqual(
            system.node.hbf_lifecycle.metrics.
            lpddr_capacity_fallback_resumes,
            1,
        )
        self.assert_gpu_tiers_empty(system.node)

    def test_tool_filter_uses_gap_metadata_not_future_duration(self):
        system = self.make_system(policy="tool_immediate")
        system.run((
            make_schedule(
                0,
                (
                    (64, 2, 2_000_000_000, 0, "human"),
                    (67, 1, 0, 65, None),
                ),
            ),
        ))

        self.assertEqual(
            system.node.calls[1].execution,
            HybridExecution.GPU_OWNED,
        )
        self.assertEqual(
            system.node.metrics.promotion_intents_filtered, 1)
        self.assertEqual(
            system.node.metrics.hbf_imports_started, 0)

    def test_composite_policy_fields_preserve_load_aware_admission(self):
        composite = SSDPromotionPolicy.for_key("composite")
        adaptive = SSDPromotionPolicy.for_key(
            "composite_adaptive")
        ready = SSDPromotionPolicy.for_key(
            "composite_ready")
        ready_adaptive = SSDPromotionPolicy.for_key(
            "composite_ready_adaptive")

        self.assertTrue(composite.direct_d_resume)
        self.assertEqual(
            composite.ssd_checkpoint_age_ns, 50_000_000)
        self.assertTrue(composite.human_gap_broadcast)
        self.assertFalse(composite.load_aware_admission)
        self.assertTrue(adaptive.direct_d_resume)
        self.assertEqual(
            adaptive.ssd_checkpoint_age_ns, 50_000_000)
        self.assertTrue(adaptive.human_gap_broadcast)
        self.assertTrue(adaptive.load_aware_admission)
        self.assertEqual(ready.ssd_checkpoint_age_ns, 0)
        self.assertFalse(ready.load_aware_admission)
        self.assertEqual(
            ready_adaptive.ssd_checkpoint_age_ns, 0)
        self.assertTrue(ready_adaptive.load_aware_admission)
        self.assertTrue(
            SSDPromotionPolicy.for_key(
                "load_aware").load_aware_admission)

    def test_composite_ssd_age_and_human_due_are_publication_anchored(
            self):
        age_node = self.make_node(policy="composite")
        age_node.set_gap_type(0, "tool")
        age_call = self.runtime_call(
            0, 0, 0,
            input_tokens=64,
            output_tokens=2,
            has_successor=True,
        )
        age_node.submit(age_call, now_ns=0)
        while (
            age_node.hbf_lifecycle.sessions["session"].state
            != PlacementState.SSD_READY
        ):
            age_node.advance(age_node.next_event_ns())
        age_intent = age_node._intent_by_session["session"]
        publication_ns = age_intent.ssd_publication_ns
        self.assertIsNotNone(publication_ns)
        self.assertGreater(
            publication_ns, age_call.user_completion_ns)
        self.assertEqual(
            age_intent.due_ns, publication_ns + 50_000_000)
        age_node.advance(age_intent.due_ns - 1)
        self.assertEqual(age_node.metrics.ssd_exports_started, 0)
        age_node.advance(age_intent.due_ns)
        self.assertEqual(
            age_node._export_by_session["session"].start_ns,
            age_intent.due_ns,
        )

        human_node = self.make_node(policy="composite")
        human_node.set_gap_type(10, "tool")
        older = self.runtime_call(
            10, 0, 0,
            input_tokens=64,
            output_tokens=2,
            has_successor=True,
            session_id="older",
        )
        human_node.submit(older, now_ns=0)
        while (
            human_node.hbf_lifecycle.sessions["older"].state
            != PlacementState.SSD_READY
        ):
            human_node.advance(human_node.next_event_ns())
        original_due_ns = human_node._intent_by_session[
            "older"].due_ns
        human_node.set_gap_type(11, "human")
        current = self.runtime_call(
            11, 0, human_node.current_ns,
            input_tokens=8,
            output_tokens=1,
            has_successor=True,
            session_id="current",
        )
        human_node.submit(current, now_ns=human_node.current_ns)
        while current.user_completion_ns is None:
            human_node.advance(human_node.next_event_ns())
        self.assertLess(current.user_completion_ns, original_due_ns)
        self.assertEqual(
            human_node._export_by_session["older"].start_ns,
            current.user_completion_ns,
        )
        while (
            human_node.hbf_lifecycle.sessions["current"].state
            != PlacementState.SSD_READY
        ):
            human_node.advance(human_node.next_event_ns())
        current_intent = human_node._intent_by_session["current"]
        self.assertTrue(current_intent.human_due_now)
        self.assertEqual(
            current_intent.due_ns,
            current_intent.ssd_publication_ns,
        )
        self.assertIn("current", human_node._export_by_session)
        self.assertEqual(
            human_node.metrics.human_gap_broadcast_intents, 2)

    def test_composite_stable_d_resume_commits_directly_to_hbf(self):
        system = self.make_system(policy="composite")
        system.run((
            make_schedule(
                0,
                (
                    (64, 2, 0, 0, "tool"),
                    (67, 2, 1_000_000_000, 65, "tool"),
                    (70, 1, 0, 68, None),
                ),
            ),
        ))

        node = system.node
        self.assertEqual(
            node._tier_call_by_request[1].prepare_source, Tier.D)
        self.assertEqual(
            node.calls[2].execution, HybridExecution.HBF_READY)
        self.assertEqual(node.metrics.direct_migrations_started, 1)
        self.assertEqual(node.metrics.direct_migrations_committed, 1)
        self.assertEqual(node.metrics.direct_migrations_stale, 0)
        self.assertEqual(node.metrics.ssd_checkpoints_published, 0)
        self.assertTrue(any(
            reservation.kind == "migration"
            and reservation.resource == "rdma-network"
            for reservation in node.calendar.reservations
        ))
        self.assert_gpu_tiers_empty(node)

    def test_composite_direct_inflight_resume_is_cow_and_stale_safe(
            self):
        node = self.make_node(policy="composite")
        node.set_gap_type(0, "tool")
        first = self.runtime_call(
            0, 0, 0,
            input_tokens=64,
            output_tokens=2,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        while first.user_completion_ns is None:
            node.advance(node.next_event_ns())
        node.set_gap_type(1, "tool")
        second = self.runtime_call(
            1, 1, first.user_completion_ns,
            input_tokens=67,
            output_tokens=2,
            prefix_reuse_tokens=65,
            has_successor=True,
        )
        node.submit(second, now_ns=first.user_completion_ns)
        while not node._direct_migrations:
            node.advance(node.next_event_ns())
        direct = next(iter(node._direct_migrations.values()))
        resume_ns = (
            max(node.current_ns, direct.job.start_ns)
            + direct.job.completion_ns
        ) // 2
        node.set_gap_type(2, "tool")
        third = self.runtime_call(
            2, 2, resume_ns,
            input_tokens=70,
            output_tokens=2,
            prefix_reuse_tokens=68,
            has_successor=True,
        )
        node.submit(third, now_ns=resume_ns)

        tier_third = node._tier_call_by_request[2]
        prepare = node.gpu_lifecycle.prepares[
            tier_third.prepare_id]
        self.assertEqual(
            third.execution,
            HybridExecution.GPU_MIGRATION_INFLIGHT,
        )
        self.assertEqual(tier_third.prepare_source, Tier.D)
        self.assertTrue(prepare.full_d_reservation)
        self.assertIsNotNone(prepare.d_owner)
        node.run_until_idle()
        self.assertEqual(node.metrics.direct_migrations_started, 2)
        self.assertEqual(node.metrics.direct_migrations_stale, 1)
        self.assertEqual(node.metrics.direct_migrations_committed, 1)
        self.assertNotIn(
            direct.source_copy_id, node.gpu_lifecycle.copies)
        self.assert_gpu_tiers_empty(node)

    def test_policy_grid_includes_long_delays_and_load_aware(self):
        for key, delay_ns in (
            ("delay_1s", 1_000_000_000),
            ("delay_5s", 5_000_000_000),
            ("delay_30s", 30_000_000_000),
            ("delay_300s", 300_000_000_000),
        ):
            with self.subTest(key=key):
                policy = SSDPromotionPolicy.for_key(key)
                self.assertEqual(policy.idle_delay_ns, delay_ns)
        load_aware = SSDPromotionPolicy.for_key("load_aware")
        self.assertEqual(load_aware.mode, "load_aware")
        self.assertGreater(load_aware.retry_ns, 0)
        alias_system = SSDStagedGPUHBFSystem(
            repo_root=REPO_ROOT,
            max_num_batched_tokens=256,
            max_num_seqs=16,
            max_prefill_chunk_tokens=128,
            migration_policy="delay_5s",
        )
        self.assertEqual(
            alias_system.node.promotion_policy.idle_delay_ns,
            5_000_000_000,
        )

    def test_load_aware_defers_while_hbf_calendar_is_busy(self):
        node = self.make_node(policy="load_aware")
        node.calendar.reserve_parallel(
            arrival_ns=0,
            job_id=99_999,
            kind="test-hbf-load",
            namespace="test",
            demands={
                "hbf-group-0-npu": (1_000_000_000, 0),
            },
        )
        node.set_gap_type(0, "tool")
        first = self.runtime_call(
            0,
            0,
            0,
            input_tokens=64,
            output_tokens=2,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        while (
            node.next_event_ns() is not None
            and node.next_event_ns() < 250_000_000
        ):
            node.advance(node.next_event_ns())

        self.assertGreater(
            node.metrics.promotion_load_deferrals, 0)
        self.assertEqual(node.metrics.hbf_imports_started, 0)
        self.assertFalse(any(
            reservation.resource == "rdma-network"
            for reservation in node.calendar.reservations
        ))

        while node.metrics.hbf_imports_started == 0:
            event_ns = node.next_event_ns()
            self.assertIsNotNone(event_ns)
            node.advance(event_ns)
        self.assertGreaterEqual(node.current_ns, 1_000_000_000)

    def assert_gpu_tiers_empty(self, node):
        self.assertEqual(node.gpu_lifecycle.p_ledger.used_bytes, 0)
        self.assertEqual(node.gpu_lifecycle.d_ledger.used_bytes, 0)
        self.assertEqual(node.gpu_lifecycle.cpu_ledger.used_bytes, 0)
        self.assertEqual(node.gpu_lifecycle.ssd_ledger.used_bytes, 0)
        self.assertFalse(node._export_by_session)
        self.assertFalse(node._import_by_session)
        node.assert_invariants()


if __name__ == "__main__":
    unittest.main()

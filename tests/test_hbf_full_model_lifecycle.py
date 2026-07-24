import dataclasses
import random
import unittest

from serving.core.hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
    qwen_logical_kv_bytes_per_token,
    qwen_model_weight_bytes_per_rank,
)
from serving.core.hbf_full_model_lifecycle import (
    ActivePrefillDrainStatus,
    FullModelHBFLifecycle,
    PlacementState,
    ResourceCalendar,
    ResumeExecution,
    hbf_kv_range_card_bytes,
    hbf_request_headroom_owner,
)


class FullModelHBFLifecycleTests(unittest.TestCase):
    def make_manager(
            self, layout="tp4", *, hardware=None,
            kv_bytes_per_token=None, validate_every_event=True,
            execution_backend="analytical_calendar"):
        return FullModelHBFLifecycle(
            hardware=hardware or HBFServerHardware(),
            layout=HBFParallelLayout.for_key(layout),
            kv_bytes_per_token=kv_bytes_per_token,
            validate_every_event=validate_every_event,
            execution_backend=execution_backend,
        )

    def complete_one_external_dispatch(self, manager):
        dispatches = manager.drain_external_dispatches()
        self.assertEqual(len(dispatches), 1)
        dispatch = dispatches[0]
        completion_ns = (
            dispatch.arrival_ns
            + dispatch.projection
            .dependency_critical_path_ns()
        )
        completed = manager.complete_external_dispatch(
            dispatch.job_id,
            dispatch.arrival_ns,
            completion_ns,
            dispatch.stage_count,
        )
        return completed

    def prepare_active_hbf(
            self, manager, *, initial_tokens, request_id,
            growth_tokens):
        record = manager.register_session("s")
        migration = manager.complete_gpu_turn(
            "s", now_ns=0, total_tokens=initial_tokens,
            has_successor=True,
        )
        self.assertIsNotNone(migration)
        if manager.execution_backend == "external_astra":
            migration = self.complete_one_external_dispatch(manager)
            ready_ns = migration.completion_ns
        else:
            manager.advance(migration.completion_ns)
            ready_ns = migration.completion_ns
        route = manager.route_resume(
            "s",
            now_ns=ready_ns,
            request_id=request_id,
            lpddr_growth_tokens=growth_tokens,
        )
        self.assertEqual(route.execution, ResumeExecution.HBF)
        return record, ready_ns

    def commit_active_lpddr_growth(
            self, manager, record, *, request_id, delta_tokens):
        self.assertIsNotNone(record.group_id)
        token_start = (
            record.committed_hbf_tokens + record.lpddr_tokens)
        delta_card_bytes = manager._range_card_bytes(
            record.group_id,
            token_start=token_start,
            token_count=delta_tokens,
        )
        manager.lpddr_ledger.shrink_card_bytes(
            hbf_request_headroom_owner(request_id),
            delta_card_bytes,
        )
        manager.lpddr_ledger.set_card_bytes(
            record.group_id,
            manager.lpddr_owner(record.session_id),
            manager._range_card_bytes(
                record.group_id,
                token_start=record.committed_hbf_tokens,
                token_count=record.lpddr_tokens + delta_tokens,
            ),
        )

    def test_qwen_geometry_includes_tp8_kv_replication(self):
        self.assertEqual(qwen_logical_kv_bytes_per_token(), 98_304)
        weights = {
            tp: qwen_model_weight_bytes_per_rank(tp)
            for tp in (1, 4, 8)
        }
        self.assertEqual(weights[1], 61_064_245_248)
        self.assertEqual(weights[4], 15_285_252_096)
        self.assertEqual(weights[8], 7_680_585_728)
        self.assertGreater(weights[8] * 8, weights[4] * 4)

    def test_resume_during_migration_falls_back_and_stale_job_cannot_publish(self):
        manager = self.make_manager()
        record = manager.register_session("s")
        job = manager.complete_gpu_turn(
            "s", now_ns=100, total_tokens=100_000,
            has_successor=True)
        self.assertIsNotNone(job)
        self.assertEqual(record.state, PlacementState.MIGRATING)
        retained = record.gpu_retained_bytes

        route = manager.route_resume(
            "s", now_ns=job.completion_ns - 1, request_id=7)
        self.assertEqual(route.execution, ResumeExecution.GPU)
        self.assertTrue(route.migration_inflight)
        self.assertEqual(record.state, PlacementState.GPU_ACTIVE)
        self.assertEqual(record.gpu_retained_bytes, retained)

        manager.advance(job.completion_ns)
        self.assertEqual(record.state, PlacementState.GPU_ACTIVE)
        self.assertEqual(record.gpu_retained_bytes, retained)
        self.assertEqual(record.committed_hbf_tokens, 0)
        self.assertEqual(manager.metrics.migrations_stale, 1)
        self.assertEqual(manager.metrics.migrations_committed, 0)

    def test_exact_completion_tie_routes_to_hbf(self):
        manager = self.make_manager()
        record = manager.register_session("s")
        job = manager.complete_gpu_turn(
            "s", now_ns=0, total_tokens=10_000, has_successor=True)
        route = manager.route_resume(
            "s", now_ns=job.completion_ns, request_id=8)
        self.assertEqual(route.execution, ResumeExecution.HBF)
        self.assertEqual(route.hbf_tokens, 10_000)
        self.assertEqual(record.gpu_retained_bytes, 0)
        self.assertEqual(manager.metrics.migrations_committed, 1)

    def test_new_generation_survives_older_migration_completion(self):
        manager = self.make_manager()
        record = manager.register_session("s")
        first = manager.complete_gpu_turn(
            "s", now_ns=0, total_tokens=20_000, has_successor=True)
        manager.route_resume("s", now_ns=first.start_ns + 1)
        second = manager.complete_gpu_turn(
            "s", now_ns=first.start_ns + 2,
            total_tokens=21_000, has_successor=True)
        self.assertGreater(second.generation, first.generation)

        manager.advance(first.completion_ns)
        self.assertEqual(record.state, PlacementState.MIGRATING)
        self.assertEqual(record.generation, second.generation)
        manager.advance(second.completion_ns)
        self.assertEqual(record.state, PlacementState.HBF_READY)
        self.assertEqual(record.committed_hbf_tokens, 21_000)
        self.assertEqual(manager.metrics.migrations_stale, 1)
        self.assertEqual(manager.metrics.migrations_committed, 1)

    def test_hbf_append_inflight_does_not_block_resume(self):
        manager = self.make_manager()
        record = manager.register_session("s")
        migration = manager.complete_gpu_turn(
            "s", now_ns=0, total_tokens=100_000, has_successor=True)
        manager.advance(migration.completion_ns)
        manager.route_resume(
            "s", now_ns=migration.completion_ns, request_id=1)
        append = manager.complete_hbf_turn(
            "s",
            now_ns=migration.completion_ns + 1,
            total_tokens=100_500,
            has_successor=True,
        )
        self.assertIsNotNone(append)
        self.assertEqual(record.lpddr_tokens, 500)
        route = manager.route_resume(
            "s", now_ns=append.completion_ns - 1, request_id=2)
        self.assertEqual(route.execution, ResumeExecution.HBF)
        self.assertEqual(route.hbf_tokens, 100_000)
        self.assertEqual(route.lpddr_tokens, 500)
        self.assertEqual(route.reason, "hbf_append_inflight")

        # Finish the zero-delta active request before observing append commit.
        manager.complete_hbf_turn(
            "s", now_ns=append.completion_ns - 1,
            total_tokens=100_500, has_successor=True)
        manager.advance(append.completion_ns)
        self.assertEqual(record.committed_hbf_tokens, 100_500)
        self.assertEqual(record.lpddr_tokens, 0)

    def test_active_prefill_drain_analytical_retains_tail(self):
        manager = self.make_manager(kv_bytes_per_token=16)
        record, ready_ns = self.prepare_active_hbf(
            manager,
            initial_tokens=100,
            request_id=11,
            growth_tokens=10,
        )
        self.commit_active_lpddr_growth(
            manager,
            record,
            request_id=11,
            delta_tokens=10,
        )
        expected_before = manager._range_card_bytes(
            record.group_id,
            token_start=100,
            token_count=10,
        )
        self.assertEqual(
            manager.lpddr_ledger.owner_card_bytes(
                manager.lpddr_owner("s")),
            expected_before,
        )

        result = manager.start_active_prefill_drain(
            "s",
            request_id=11,
            now_ns=ready_ns,
            total_tokens=110,
            tail_tokens=2,
        )

        self.assertEqual(
            result.status, ActivePrefillDrainStatus.STARTED)
        self.assertIsNotNone(result.job)
        self.assertEqual(result.append_tokens, 8)
        self.assertEqual(result.retained_tail_tokens, 2)
        self.assertEqual(result.job.token_start, 100)
        self.assertEqual(result.job.token_count, 8)
        self.assertEqual(record.state, PlacementState.HBF_ACTIVE)
        self.assertEqual(record.active_request_id, 11)
        self.assertEqual(record.total_tokens, 110)
        self.assertEqual(record.lpddr_tokens, 10)
        self.assertEqual(
            manager.report()[
                "active_prefill_drain_pending_job_ids"],
            [result.job.job_id],
        )

        manager.advance(result.job.completion_ns)
        self.assertEqual(record.committed_hbf_tokens, 108)
        self.assertEqual(record.lpddr_tokens, 2)
        expected_tail = manager._range_card_bytes(
            record.group_id,
            token_start=108,
            token_count=2,
        )
        self.assertEqual(
            manager.lpddr_ledger.owner_card_bytes(
                manager.lpddr_owner("s")),
            expected_tail,
        )
        satisfied = manager.start_active_prefill_drain(
            "s",
            request_id=11,
            now_ns=result.job.completion_ns,
            total_tokens=110,
            tail_tokens=2,
        )
        self.assertEqual(
            satisfied.status, ActivePrefillDrainStatus.SATISFIED)
        self.assertIsNone(satisfied.job)
        self.assertEqual(manager.metrics.active_prefill_drain_candidates, 2)
        self.assertEqual(manager.metrics.active_prefill_drain_started, 1)
        self.assertEqual(manager.metrics.active_prefill_drain_satisfied, 1)
        self.assertEqual(manager.metrics.active_prefill_drain_committed, 1)
        self.assertEqual(manager.metrics.active_prefill_drain_stale, 0)
        manager.assert_invariants()

    def test_active_prefill_drain_external_astra_commits(self):
        manager = self.make_manager(
            kv_bytes_per_token=16,
            execution_backend="external_astra",
        )
        record, ready_ns = self.prepare_active_hbf(
            manager,
            initial_tokens=200,
            request_id=21,
            growth_tokens=6,
        )
        self.commit_active_lpddr_growth(
            manager,
            record,
            request_id=21,
            delta_tokens=6,
        )

        result = manager.start_active_prefill_drain(
            "s",
            request_id=21,
            now_ns=ready_ns,
            total_tokens=206,
            tail_tokens=1,
        )

        self.assertEqual(
            result.status, ActivePrefillDrainStatus.STARTED)
        self.assertEqual(result.job.token_count, 5)
        self.assertEqual(manager.report()["pending_job_count"], 1)
        completed = self.complete_one_external_dispatch(manager)
        self.assertEqual(completed.job_id, result.job.job_id)
        self.assertGreater(completed.completion_ns, ready_ns)
        self.assertEqual(record.committed_hbf_tokens, 205)
        self.assertEqual(record.lpddr_tokens, 1)
        self.assertEqual(manager.metrics.astra_completed_jobs, 2)
        self.assertEqual(manager.metrics.active_prefill_drain_committed, 1)
        self.assertEqual(
            manager.report()[
                "active_prefill_drain_pending_job_ids"],
            [],
        )
        self.assertFalse(manager.calendar.reservations)
        manager.assert_invariants()

    def test_active_prefill_drain_capacity_fallback_keeps_lpddr(self):
        weight = qwen_model_weight_bytes_per_rank(1)
        hardware = dataclasses.replace(
            HBFServerHardware(),
            hbf_capacity_bytes_per_card=weight + 100,
        )
        manager = self.make_manager(
            "dp8",
            hardware=hardware,
            kv_bytes_per_token=1,
        )
        record, ready_ns = self.prepare_active_hbf(
            manager,
            initial_tokens=90,
            request_id=31,
            growth_tokens=20,
        )
        self.commit_active_lpddr_growth(
            manager,
            record,
            request_id=31,
            delta_tokens=20,
        )
        hbf_before = dict(
            manager.report()["group_reserved_bytes_by_card"][
                record.group_id])
        lpddr_before = dict(
            manager.lpddr_ledger.owner_card_bytes(
                manager.lpddr_owner("s")))

        result = manager.start_active_prefill_drain(
            "s",
            request_id=31,
            now_ns=ready_ns,
            total_tokens=110,
            tail_tokens=0,
        )

        self.assertEqual(
            result.status,
            ActivePrefillDrainStatus.CAPACITY_FALLBACK,
        )
        self.assertIsNone(result.job)
        self.assertEqual(record.committed_hbf_tokens, 90)
        self.assertEqual(record.lpddr_tokens, 20)
        self.assertEqual(
            manager.lpddr_ledger.owner_card_bytes(
                manager.lpddr_owner("s")),
            lpddr_before,
        )
        self.assertEqual(
            manager.report()["group_reserved_bytes_by_card"][
                record.group_id],
            hbf_before,
        )
        self.assertEqual(
            manager.metrics.active_prefill_drain_capacity_fallback, 1)
        self.assertEqual(manager.metrics.active_prefill_drain_started, 0)
        manager.assert_invariants()

    def test_active_prefill_drain_waits_then_retries_after_append(self):
        manager = self.make_manager(kv_bytes_per_token=16)
        record = manager.register_session("s")
        migration = manager.complete_gpu_turn(
            "s", now_ns=0, total_tokens=100,
            has_successor=True,
        )
        manager.advance(migration.completion_ns)
        manager.route_resume(
            "s", now_ns=migration.completion_ns, request_id=40)
        older_append = manager.complete_hbf_turn(
            "s",
            now_ns=migration.completion_ns,
            total_tokens=110,
            has_successor=True,
        )
        manager.route_resume(
            "s",
            now_ns=migration.completion_ns,
            request_id=41,
            lpddr_growth_tokens=10,
        )
        self.commit_active_lpddr_growth(
            manager,
            record,
            request_id=41,
            delta_tokens=10,
        )

        waiting = manager.start_active_prefill_drain(
            "s",
            request_id=41,
            now_ns=migration.completion_ns,
            total_tokens=120,
            tail_tokens=0,
        )

        self.assertEqual(
            waiting.status,
            ActivePrefillDrainStatus.WAIT_EXISTING_APPEND,
        )
        self.assertIsNone(waiting.job)
        self.assertEqual(
            waiting.blocking_append_job_ids,
            (older_append.job_id,),
        )
        self.assertEqual(record.lpddr_tokens, 20)
        manager.advance(older_append.completion_ns)
        self.assertEqual(record.committed_hbf_tokens, 110)
        self.assertEqual(record.lpddr_tokens, 10)

        retried = manager.start_active_prefill_drain(
            "s",
            request_id=41,
            now_ns=older_append.completion_ns,
            total_tokens=120,
            tail_tokens=0,
        )
        self.assertEqual(
            retried.status, ActivePrefillDrainStatus.STARTED)
        self.assertEqual(retried.job.token_start, 110)
        self.assertEqual(retried.job.token_count, 10)
        manager.advance(retried.job.completion_ns)
        self.assertEqual(record.committed_hbf_tokens, 120)
        self.assertEqual(record.lpddr_tokens, 0)
        self.assertEqual(
            manager.metrics.active_prefill_drain_candidates, 2)
        self.assertEqual(
            manager.metrics.active_prefill_drain_wait_existing_append, 1)
        self.assertEqual(manager.metrics.active_prefill_drain_started, 1)
        self.assertEqual(manager.metrics.active_prefill_drain_committed, 1)
        manager.assert_invariants()

    def test_active_prefill_drain_rejects_state_request_and_ledger_errors(self):
        inactive = self.make_manager(kv_bytes_per_token=1)
        inactive.register_session("s")
        with self.assertRaisesRegex(RuntimeError, "HBF_ACTIVE"):
            inactive.start_active_prefill_drain(
                "s",
                request_id=1,
                now_ns=0,
                total_tokens=0,
            )

        manager = self.make_manager(kv_bytes_per_token=1)
        record, ready_ns = self.prepare_active_hbf(
            manager,
            initial_tokens=10,
            request_id=51,
            growth_tokens=4,
        )
        with self.assertRaisesRegex(RuntimeError, "does not own"):
            manager.start_active_prefill_drain(
                "s",
                request_id=52,
                now_ns=ready_ns,
                total_tokens=10,
            )
        with self.assertRaisesRegex(ValueError, "cannot shrink"):
            manager.start_active_prefill_drain(
                "s",
                request_id=51,
                now_ns=ready_ns,
                total_tokens=9,
            )
        with self.assertRaisesRegex(ValueError, "tail_tokens"):
            manager.start_active_prefill_drain(
                "s",
                request_id=51,
                now_ns=ready_ns,
                total_tokens=10,
                tail_tokens=-1,
            )

        self.commit_active_lpddr_growth(
            manager,
            record,
            request_id=51,
            delta_tokens=3,
        )
        version_before = record.version
        with self.assertRaisesRegex(RuntimeError, "session owner vector"):
            manager.start_active_prefill_drain(
                "s",
                request_id=51,
                now_ns=ready_ns,
                total_tokens=14,
            )
        self.assertEqual(record.total_tokens, 10)
        self.assertEqual(record.lpddr_tokens, 0)
        self.assertEqual(record.version, version_before)
        self.assertEqual(
            manager.metrics.active_prefill_drain_candidates, 0)

    def test_active_prefill_drain_tp8_context_preserves_uneven_stripe(self):
        manager = self.make_manager("tp8_context")
        record, ready_ns = self.prepare_active_hbf(
            manager,
            initial_tokens=1,
            request_id=61,
            growth_tokens=4,
        )
        self.commit_active_lpddr_growth(
            manager,
            record,
            request_id=61,
            delta_tokens=4,
        )
        per_head = qwen_logical_kv_bytes_per_token() // 4

        result = manager.start_active_prefill_drain(
            "s",
            request_id=61,
            now_ns=ready_ns,
            total_tokens=5,
            tail_tokens=1,
        )

        self.assertEqual(
            result.status, ActivePrefillDrainStatus.STARTED)
        self.assertEqual(result.job.token_start, 1)
        self.assertEqual(result.job.token_count, 3)
        self.assertEqual(
            dict(result.job.card_bytes),
            {
                0: per_head, 1: 2 * per_head,
                2: per_head, 3: 2 * per_head,
                4: per_head, 5: 2 * per_head,
                6: per_head, 7: 2 * per_head,
            },
        )
        manager.advance(result.job.completion_ns)
        self.assertEqual(record.committed_hbf_tokens, 4)
        self.assertEqual(record.lpddr_tokens, 1)
        self.assertEqual(
            manager.lpddr_ledger.owner_card_bytes(
                manager.lpddr_owner("s")),
            {
                0: per_head, 1: 0,
                2: per_head, 3: 0,
                4: per_head, 5: 0,
                6: per_head, 7: 0,
            },
        )
        manager.assert_invariants()

    def test_context_shrink_invalidates_old_append_lineage(self):
        manager = self.make_manager()
        record = manager.register_session("s")
        migration = manager.complete_gpu_turn(
            "s", now_ns=0, total_tokens=1_000,
            has_successor=True)
        manager.advance(migration.completion_ns)
        manager.route_resume(
            "s", now_ns=migration.completion_ns, request_id=1)
        append = manager.complete_hbf_turn(
            "s",
            now_ns=migration.completion_ns + 1,
            total_tokens=1_007,
            has_successor=True,
        )
        old_generation = append.generation
        route = manager.route_resume(
            "s",
            now_ns=append.completion_ns - 1,
            request_id=2,
            prefix_reuse_tokens=900,
            input_tokens=905,
        )
        self.assertEqual(route.execution, ResumeExecution.HBF)
        self.assertEqual(route.hbf_tokens, 900)
        self.assertEqual(route.lpddr_tokens, 0)
        self.assertEqual(route.reason, "hbf_context_trimmed")
        self.assertGreater(record.generation, old_generation)
        self.assertEqual(record.total_tokens, 900)

        manager.advance(append.completion_ns)
        self.assertEqual(record.committed_hbf_tokens, 900)
        self.assertEqual(record.lpddr_tokens, 0)
        self.assertEqual(manager.metrics.append_jobs_committed, 0)
        self.assertEqual(manager.report()["pending_job_count"], 0)
        manager.assert_invariants()

    def test_append_reads_lpddr_and_uses_namespaced_job_identity(self):
        manager = self.make_manager()
        manager.register_session("s")
        migration = manager.complete_gpu_turn(
            "s", now_ns=0, total_tokens=1_000,
            has_successor=True)
        manager.advance(migration.completion_ns)
        manager.route_resume(
            "s", now_ns=migration.completion_ns, request_id=1)
        append = manager.complete_hbf_turn(
            "s",
            now_ns=migration.completion_ns,
            total_tokens=1_010,
            has_successor=True,
        )
        rows = [
            row for row in manager.calendar.reservations
            if (
                row.namespace == "hbf-lifecycle"
                and row.job_id == append.job_id
                and row.kind == "append"
            )
        ]
        self.assertEqual(
            len([row for row in rows
                 if row.resource.endswith("-lpddr")]),
            4,
        )
        self.assertEqual(
            len([row for row in rows
                 if row.resource.endswith("-media")]),
            4,
        )

    def test_tp8_migration_writes_two_physical_kv_copies(self):
        logical = 123_456
        managers = {
            key: self.make_manager(key, kv_bytes_per_token=1)
            for key in ("dp8", "tp4", "tp8")
        }
        jobs = {}
        for key, manager in managers.items():
            manager.register_session("s")
            jobs[key] = manager.complete_gpu_turn(
                "s", now_ns=0, total_tokens=logical,
                has_successor=True)
        self.assertEqual(jobs["dp8"].physical_bytes, logical)
        self.assertEqual(jobs["tp4"].physical_bytes, logical)
        self.assertEqual(jobs["tp8"].physical_bytes, 2 * logical)
        self.assertEqual(jobs["dp8"].per_card_bytes, logical)
        self.assertEqual(
            jobs["tp4"].per_card_bytes,
            jobs["tp8"].per_card_bytes,
        )

    def test_tp8_context_one_token_ranges_alternate_card_pairs(self):
        layout = HBFParallelLayout.for_key("tp8_context")
        cards = tuple(range(8))
        per_token = qwen_logical_kv_bytes_per_token()
        per_head = per_token // 4
        for token_start in range(33):
            with self.subTest(token_start=token_start):
                vector = hbf_kv_range_card_bytes(
                    layout=layout,
                    card_ids=cards,
                    kv_bytes_per_token=per_token,
                    token_start=token_start,
                    token_count=1,
                )
                expected_cards = (
                    {0, 2, 4, 6}
                    if token_start % 2 == 0
                    else {1, 3, 5, 7}
                )
                self.assertEqual(
                    {
                        card_id for card_id, byte_count in vector.items()
                        if byte_count
                    },
                    expected_cards,
                )
                self.assertTrue(all(
                    vector[card_id] == per_head
                    for card_id in expected_cards
                ))
                self.assertEqual(sum(vector.values()), per_token)

    def test_card_range_vectors_are_additive_across_odd_boundaries(self):
        rng = random.Random(71)
        for layout_key in (
                "dp8", "tp4", "tp8", "tp8_context"):
            layout = HBFParallelLayout.for_key(layout_key)
            cards = tuple(range(layout.tp_size))
            for _ in range(100):
                token_start = rng.randrange(0, 101)
                first_count = rng.randrange(0, 31)
                second_count = rng.randrange(0, 31)
                kv_bytes_per_token = rng.choice((1, 17, 98_304))
                first = hbf_kv_range_card_bytes(
                    layout=layout,
                    card_ids=cards,
                    kv_bytes_per_token=kv_bytes_per_token,
                    token_start=token_start,
                    token_count=first_count,
                )
                second = hbf_kv_range_card_bytes(
                    layout=layout,
                    card_ids=cards,
                    kv_bytes_per_token=kv_bytes_per_token,
                    token_start=token_start + first_count,
                    token_count=second_count,
                )
                combined = hbf_kv_range_card_bytes(
                    layout=layout,
                    card_ids=cards,
                    kv_bytes_per_token=kv_bytes_per_token,
                    token_start=token_start,
                    token_count=first_count + second_count,
                )
                self.assertEqual(
                    combined,
                    {
                        card_id: first[card_id] + second[card_id]
                        for card_id in cards
                    },
                )

    def test_tp8_context_one_token_capacity_is_not_averaged(self):
        per_token = qwen_logical_kv_bytes_per_token()
        per_head = per_token // 4
        hardware = dataclasses.replace(
            HBFServerHardware(),
            hbf_capacity_bytes_per_card=per_head,
        )
        manager = FullModelHBFLifecycle(
            hardware=hardware,
            layout=HBFParallelLayout.for_key("tp8_context"),
            model_weight_bytes_per_rank=0,
        )
        first = manager.register_session("first")
        first_job = manager.complete_gpu_turn(
            "first", now_ns=0, total_tokens=1,
            has_successor=True)
        self.assertIsNotNone(first_job)
        self.assertEqual(
            dict(first_job.card_bytes),
            {
                0: per_head, 1: 0,
                2: per_head, 3: 0,
                4: per_head, 5: 0,
                6: per_head, 7: 0,
            },
        )
        self.assertEqual(first.state, PlacementState.MIGRATING)

        second = manager.register_session("second")
        second_job = manager.complete_gpu_turn(
            "second", now_ns=0, total_tokens=1,
            has_successor=True)
        self.assertIsNone(second_job)
        self.assertEqual(second.state, PlacementState.GPU_READY)
        self.assertEqual(
            manager.report()["group_reserved_bytes_by_card"][0],
            dict(first_job.card_bytes),
        )
        manager.advance(first_job.completion_ns)
        manager.assert_invariants()

    def test_tp8_context_append_uses_next_token_parity(self):
        per_token = qwen_logical_kv_bytes_per_token()
        per_head = per_token // 4
        manager = self.make_manager("tp8_context")
        manager.register_session("s")
        migration = manager.complete_gpu_turn(
            "s", now_ns=0, total_tokens=1,
            has_successor=True)
        manager.advance(migration.completion_ns)
        manager.route_resume(
            "s", now_ns=migration.completion_ns, request_id=1)
        append = manager.complete_hbf_turn(
            "s",
            now_ns=migration.completion_ns + 1,
            total_tokens=2,
            has_successor=True,
        )
        self.assertEqual(append.token_start, 1)
        self.assertEqual(
            dict(append.card_bytes),
            {
                0: 0, 1: per_head,
                2: 0, 3: per_head,
                4: 0, 5: per_head,
                6: 0, 7: per_head,
            },
        )
        self.assertEqual(
            manager.lpddr_ledger.owner_card_bytes(
                manager.lpddr_owner("s")),
            dict(append.card_bytes),
        )
        manager.advance(append.completion_ns)
        self.assertEqual(
            manager.report()["group_reserved_bytes_by_card"][0],
            {card_id: per_head for card_id in range(8)},
        )
        manager.assert_invariants()

    def test_dp8_migrations_to_disjoint_cards_still_share_rdma(self):
        manager = self.make_manager("dp8", kv_bytes_per_token=1)
        for session_id in ("a", "b", "c"):
            manager.register_session(session_id)
        first = manager.complete_gpu_turn(
            "a", now_ns=0, total_tokens=1_000_000,
            has_successor=True)
        second = manager.complete_gpu_turn(
            "b", now_ns=0, total_tokens=1_000_000,
            has_successor=True)
        self.assertNotEqual(first.group_id, second.group_id)
        first_card = manager.groups[first.group_id].card_ids[0]
        second_card = manager.groups[second.group_id].card_ids[0]
        first_media = [
            row for row in manager.calendar.reservations
            if row.job_id == first.job_id
            and row.resource == f"hbf-card-{first_card}-media"
        ][0]
        second_media = [
            row for row in manager.calendar.reservations
            if row.job_id == second.job_id
            and row.resource == f"hbf-card-{second_card}-media"
        ][0]
        second_rdma = [
            row for row in manager.calendar.reservations
            if row.job_id == second.job_id
            and row.resource == "rdma-network"
        ][0]
        first_rdma = [
            row for row in manager.calendar.reservations
            if row.job_id == first.job_id
            and row.resource == "rdma-network"
        ][0]
        self.assertGreaterEqual(second_rdma.start_ns, first_rdma.end_ns)
        self.assertEqual(second_media.start_ns, second_rdma.start_ns)
        self.assertLess(first_media.start_ns, second_media.start_ns)

    def test_dp8_appends_on_disjoint_cards_overlap(self):
        manager = self.make_manager("dp8", kv_bytes_per_token=1)
        jobs = []
        for session_id in ("a", "b"):
            manager.register_session(session_id)
            jobs.append(manager.complete_gpu_turn(
                session_id, now_ns=0, total_tokens=1_000_000,
                has_successor=True))
        ready_ns = max(job.completion_ns for job in jobs)
        manager.advance(ready_ns)
        appends = []
        for session_id in ("a", "b"):
            manager.route_resume(session_id, now_ns=ready_ns)
            appends.append(manager.complete_hbf_turn(
                session_id,
                now_ns=ready_ns,
                total_tokens=1_001_000,
                has_successor=True,
            ))
        media = []
        for append in appends:
            card_id = manager.groups[append.group_id].card_ids[0]
            media.append([
                row for row in manager.calendar.reservations
                if row.job_id == append.job_id
                and row.resource == f"hbf-card-{card_id}-media"
            ][0])
        self.assertEqual(media[0].start_ns, media[1].start_ns)

    def test_capacity_lru_evicts_only_idle_hbf_record(self):
        # One byte per token and 100 usable bytes per card make capacity
        # transitions exact and cheap.
        base = HBFServerHardware()
        weight = qwen_model_weight_bytes_per_rank(1)
        hardware = dataclasses.replace(
            base,
            hbf_capacity_bytes_per_card=weight + 100,
        )
        manager = FullModelHBFLifecycle(
            hardware=hardware,
            layout=HBFParallelLayout.for_key("dp8"),
            kv_bytes_per_token=1,
        )
        for session_id in ("old", "new"):
            manager.register_session(session_id)
        old = manager.complete_gpu_turn(
            "old", now_ns=0, total_tokens=80, has_successor=True)
        manager.advance(old.completion_ns)
        # Fill every other group so the next migration must revisit group 0.
        fillers = []
        for index in range(1, 8):
            session_id = f"fill-{index}"
            manager.register_session(session_id)
            job = manager.complete_gpu_turn(
                session_id, now_ns=old.completion_ns,
                total_tokens=100, has_successor=True)
            fillers.append(job)
        manager.advance(max(job.completion_ns for job in fillers))
        new = manager.complete_gpu_turn(
            "new",
            now_ns=max(job.completion_ns for job in fillers) + 1,
            total_tokens=90,
            has_successor=True,
        )
        self.assertIsNotNone(new)
        self.assertEqual(new.group_id, old.group_id)
        self.assertEqual(
            manager.sessions["old"].state, PlacementState.EVICTED)
        self.assertEqual(manager.metrics.capacity_evictions, 1)
        route = manager.route_resume(
            "old", now_ns=new.start_ns, request_id=9)
        self.assertEqual(route.execution, ResumeExecution.GPU_RECOMPUTE)

    def test_capacity_miss_keeps_valid_gpu_copy(self):
        base = HBFServerHardware()
        weight = qwen_model_weight_bytes_per_rank(8)
        hardware = dataclasses.replace(
            base,
            hbf_capacity_bytes_per_card=weight + 10,
        )
        manager = FullModelHBFLifecycle(
            hardware=hardware,
            layout=HBFParallelLayout.for_key("tp8"),
            kv_bytes_per_token=1,
        )
        record = manager.register_session("s")
        job = manager.complete_gpu_turn(
            "s", now_ns=0, total_tokens=100, has_successor=True)
        self.assertIsNone(job)
        self.assertEqual(record.state, PlacementState.GPU_READY)
        self.assertEqual(record.gpu_retained_bytes, 100)
        route = manager.route_resume("s", now_ns=1)
        self.assertEqual(route.execution, ResumeExecution.GPU)
        self.assertEqual(
            route.reason, "hbf_capacity_unavailable_gpu_retained")

    def test_gpu_hbm_pressure_reclaims_gpu_ready_lru_stably(self):
        weight = qwen_model_weight_bytes_per_rank(8)
        hardware = dataclasses.replace(
            HBFServerHardware(),
            hbf_capacity_bytes_per_card=weight + 1,
        )
        manager = FullModelHBFLifecycle(
            hardware=hardware,
            layout=HBFParallelLayout.for_key("tp8"),
            kv_bytes_per_token=1,
        )
        for session_id in ("z-session", "a-session"):
            manager.register_session(session_id)
            self.assertIsNone(manager.complete_gpu_turn(
                session_id,
                now_ns=10,
                total_tokens=10,
                has_successor=True,
            ))

        eviction = manager.evict_oldest_gpu_ready_for_hbm_pressure(
            ("z-session", "a-session"),
            now_ns=20,
        )

        self.assertEqual(eviction.session_id, "a-session")
        self.assertEqual(eviction.last_access_ns, 10)
        self.assertEqual(eviction.logical_bytes, 10)
        self.assertEqual(
            manager.sessions["a-session"].state,
            PlacementState.EVICTED,
        )
        self.assertEqual(
            manager.sessions["a-session"].gpu_retained_bytes, 0)
        self.assertEqual(
            manager.sessions["z-session"].state,
            PlacementState.GPU_READY,
        )
        self.assertEqual(
            manager.metrics.gpu_ready_hbm_pressure_evictions, 1)
        self.assertEqual(
            manager.metrics.gpu_ready_hbm_pressure_evicted_bytes, 10)
        route = manager.route_resume(
            "a-session", now_ns=21, request_id=7)
        self.assertEqual(route.execution, ResumeExecution.GPU_RECOMPUTE)
        manager.assert_invariants()

    def test_gpu_hbm_pressure_skips_gpu_ready_with_pending_callback(self):
        weight = qwen_model_weight_bytes_per_rank(8)
        hardware = dataclasses.replace(
            HBFServerHardware(),
            hbf_capacity_bytes_per_card=weight + 25,
        )
        manager = FullModelHBFLifecycle(
            hardware=hardware,
            layout=HBFParallelLayout.for_key("tp8"),
            kv_bytes_per_token=1,
        )
        record = manager.register_session("pending")
        migration = manager.complete_gpu_turn(
            "pending", now_ns=0, total_tokens=100,
            has_successor=True)
        manager.route_resume("pending", now_ns=0, request_id=1)
        self.assertIsNone(manager.complete_gpu_turn(
            "pending", now_ns=0, total_tokens=100,
            has_successor=True))
        self.assertEqual(record.state, PlacementState.GPU_READY)
        self.assertTrue(record.migration_job_ids)
        self.assertFalse(
            manager.gpu_ready_pressure_reclaimable("pending"))
        self.assertIsNone(
            manager.evict_oldest_gpu_ready_for_hbm_pressure(
                ("pending",), now_ns=0))

        manager.advance(migration.completion_ns)
        self.assertTrue(
            manager.gpu_ready_pressure_reclaimable("pending"))
        manager.assert_invariants()

    def test_oversized_migration_does_not_evict_ready_session(self):
        base = HBFServerHardware()
        weight = qwen_model_weight_bytes_per_rank(1)
        hardware = dataclasses.replace(
            base,
            hbf_capacity_bytes_per_card=weight + 100,
        )
        manager = FullModelHBFLifecycle(
            hardware=hardware,
            layout=HBFParallelLayout.for_key("dp8"),
            kv_bytes_per_token=1,
        )
        victim = manager.register_session("victim")
        victim_job = manager.complete_gpu_turn(
            "victim", now_ns=0, total_tokens=40,
            has_successor=True)
        manager.advance(victim_job.completion_ns)
        self.assertEqual(victim.state, PlacementState.HBF_READY)

        oversized = manager.register_session("oversized")
        job = manager.complete_gpu_turn(
            "oversized",
            now_ns=victim_job.completion_ns + 1,
            total_tokens=200,
            has_successor=True,
        )
        self.assertIsNone(job)
        self.assertEqual(victim.state, PlacementState.HBF_READY)
        self.assertEqual(victim.committed_hbf_tokens, 40)
        self.assertEqual(oversized.state, PlacementState.GPU_READY)
        self.assertEqual(manager.metrics.capacity_evictions, 0)
        manager.assert_invariants()

    def test_group_preflight_avoids_evictions_on_infeasible_group(self):
        base = HBFServerHardware()
        weight = qwen_model_weight_bytes_per_rank(1)
        hardware = dataclasses.replace(
            base,
            hbf_capacity_bytes_per_card=weight + 100,
        )
        manager = FullModelHBFLifecycle(
            hardware=hardware,
            layout=HBFParallelLayout.for_key("dp8"),
            kv_bytes_per_token=1,
        )

        def place(
                session_id, group_id, tokens, *,
                state=PlacementState.HBF_ACTIVE):
            record = manager.register_session(session_id)
            record.state = state
            record.total_tokens = tokens
            record.committed_hbf_tokens = tokens
            record.group_id = group_id
            record.committed_per_card_bytes = tokens
            manager._reserve_group(group_id, tokens)
            return record

        group0_victim = place(
            "group0-victim", 0, 10,
            state=PlacementState.HBF_READY)
        place("group0-active", 0, 80)
        group1_victim = place(
            "group1-victim", 1, 30,
            state=PlacementState.HBF_READY)
        place("group1-active", 1, 65)
        for group_id in range(2, 8):
            place(f"full-{group_id}", group_id, 100)
        manager.assert_invariants()

        manager.register_session("new")
        job = manager.complete_gpu_turn(
            "new", now_ns=1, total_tokens=30,
            has_successor=True)
        self.assertIsNotNone(job)
        self.assertEqual(job.group_id, 1)
        self.assertEqual(
            group0_victim.state, PlacementState.HBF_READY)
        self.assertEqual(
            group1_victim.state, PlacementState.EVICTED)
        self.assertEqual(manager.metrics.capacity_evictions, 1)
        manager.assert_invariants()

    def test_gpu_source_bandwidth_is_independent_of_hbf_fabric(self):
        hardware = dataclasses.replace(
            HBFServerHardware(),
            intra_fabric_bandwidth_gbps_per_card=7.0,
        )
        manager = FullModelHBFLifecycle(
            hardware=hardware,
            layout=HBFParallelLayout.for_key("tp4"),
            kv_bytes_per_token=1,
            gpu_source_root_bandwidth_gbps=123.0,
        )
        manager.register_session("s")
        job = manager.complete_gpu_turn(
            "s", now_ns=0, total_tokens=12_300,
            has_successor=True)
        source = [
            row for row in manager.calendar.reservations
            if (
                row.job_id == job.job_id
                and row.resource == "gpu-source-pcie-root"
            )
        ][0]
        self.assertEqual(source.service_ns, 100)
        self.assertEqual(
            manager.report()["gpu_source_root_bandwidth_gbps"],
            123.0,
        )

    def test_final_cleanup_waits_for_stale_jobs_without_double_free(self):
        manager = self.make_manager(kv_bytes_per_token=1)
        record = manager.register_session("s")
        job = manager.complete_gpu_turn(
            "s", now_ns=0, total_tokens=1_000_000,
            has_successor=True)
        manager.end_session("s", now_ns=job.start_ns + 1)
        self.assertEqual(record.state, PlacementState.ENDED)
        manager.advance(job.completion_ns)
        self.assertEqual(manager.report()["pending_job_count"], 0)
        self.assertTrue(all(
            value == 0
            for value in manager.report()[
                "group_reserved_per_card_bytes"].values()
        ))

    def test_randomized_race_sequence_preserves_invariants(self):
        rng = random.Random(73)
        manager = self.make_manager("tp4", kv_bytes_per_token=16)
        manager.register_session("s")
        now = 0
        total = 1
        for _ in range(500):
            record = manager.sessions["s"]
            if record.state in {
                PlacementState.GPU_ACTIVE,
                PlacementState.GPU_READY,
                PlacementState.EVICTED,
            }:
                if record.state == PlacementState.GPU_READY:
                    manager.route_resume("s", now_ns=now)
                total += rng.randint(1, 100)
                manager.complete_gpu_turn(
                    "s", now_ns=now, total_tokens=total,
                    has_successor=True)
            elif record.state == PlacementState.MIGRATING:
                if rng.random() < 0.5:
                    manager.route_resume("s", now_ns=now)
                else:
                    now = manager.next_completion_ns()
                    manager.advance(now)
            elif record.state == PlacementState.HBF_READY:
                manager.route_resume("s", now_ns=now)
            elif record.state == PlacementState.HBF_ACTIVE:
                total += rng.randint(0, 100)
                manager.complete_hbf_turn(
                    "s", now_ns=now, total_tokens=total,
                    has_successor=True)
            manager.assert_invariants()
            now += 1

    def test_time_cannot_move_backwards(self):
        manager = self.make_manager()
        manager.register_session("s")
        manager.advance(100)
        with self.assertRaisesRegex(ValueError, "backwards"):
            manager.route_resume("s", now_ns=99)

    def test_second_resume_while_hbf_request_active_is_rejected(self):
        manager = self.make_manager()
        manager.register_session("s")
        job = manager.complete_gpu_turn(
            "s", now_ns=0, total_tokens=100, has_successor=True)
        manager.advance(job.completion_ns)
        manager.route_resume(
            "s", now_ns=job.completion_ns, request_id=11)
        with self.assertRaisesRegex(RuntimeError, "cannot resume"):
            manager.route_resume(
                "s", now_ns=job.completion_ns, request_id=12)
        self.assertEqual(
            manager.sessions["s"].active_request_id, 11)

    def test_strict_and_sweep_lifecycle_are_event_equivalent(self):
        def run(validate_every_event):
            manager = self.make_manager(
                kv_bytes_per_token=16,
                validate_every_event=validate_every_event,
            )
            manager.register_session("s")
            migration = manager.complete_gpu_turn(
                "s", now_ns=0, total_tokens=2_000,
                has_successor=True)
            manager.advance(migration.completion_ns)
            manager.route_resume(
                "s", now_ns=migration.completion_ns, request_id=1)
            append = manager.complete_hbf_turn(
                "s",
                now_ns=migration.completion_ns + 1,
                total_tokens=2_100,
                has_successor=True,
            )
            manager.advance(append.completion_ns)
            manager.route_resume(
                "s", now_ns=append.completion_ns, request_id=2)
            manager.complete_hbf_turn(
                "s",
                now_ns=append.completion_ns + 1,
                total_tokens=2_110,
                has_successor=False,
            )
            manager.run_until_idle()
            report = manager.report()
            return {
                "metrics": report["metrics"],
                "group_reserved": (
                    report["group_reserved_per_card_bytes"]),
                "resource_busy": report["resource_busy_ns"],
                "calendar_counts": dict(
                    manager.calendar.reservation_count_by_resource),
                "calendar_bytes": dict(
                    manager.calendar.reservation_bytes_by_resource),
                "sessions": report["sessions"],
                "current_ns": report["current_ns"],
            }

        self.assertEqual(run(True), run(False))

    def test_sweep_lifecycle_still_checks_final_drain(self):
        manager = self.make_manager(validate_every_event=False)
        manager._reserved_per_card_by_group[0] = 1
        manager.advance(0)
        with self.assertRaisesRegex(
                AssertionError, "reservation ledger mismatch"):
            manager.run_until_idle()

    def test_lifecycle_validation_mode_must_be_boolean(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.make_manager(validate_every_event=1)


class ResourceCalendarTests(unittest.TestCase):
    def test_gang_reservation_waits_for_every_resource(self):
        calendar = ResourceCalendar()
        first = calendar.reserve_parallel(
            arrival_ns=0,
            job_id=1,
            kind="one",
            demands={"a": (10, 10), "b": (20, 20)},
        )
        second = calendar.reserve_parallel(
            arrival_ns=5,
            job_id=2,
            kind="two",
            demands={"a": (3, 3), "b": (4, 4)},
        )
        self.assertEqual(first, (0, 20))
        self.assertEqual(second, (20, 24))
        self.assertEqual(calendar.utilization("a", 24), 13 / 24)
        self.assertEqual(calendar.utilization("b", 24), 1.0)

    def test_compact_calendar_preserves_exact_aggregates(self):
        retained = ResourceCalendar()
        compact = ResourceCalendar(retain_reservations=False)
        for calendar in (retained, compact):
            calendar.reserve_parallel(
                arrival_ns=0,
                job_id=1,
                kind="migration",
                namespace="lifecycle",
                demands={"a": (10, 100), "b": (20, 200)},
            )
            calendar.reserve_parallel(
                arrival_ns=5,
                job_id=2,
                kind="batch",
                namespace="pool",
                demands={"a": (3, 30), "b": (4, 40)},
            )
        self.assertEqual(compact.reservations, [])
        self.assertEqual(len(retained.reservations), 4)
        self.assertEqual(compact.available_ns, retained.available_ns)
        self.assertEqual(compact.busy_ns, retained.busy_ns)
        self.assertEqual(
            compact.reservation_count_by_resource,
            retained.reservation_count_by_resource,
        )
        self.assertEqual(
            compact.reservation_bytes_by_resource,
            retained.reservation_bytes_by_resource,
        )
        self.assertEqual(
            compact.reservation_count_by_namespace_kind,
            retained.reservation_count_by_namespace_kind,
        )
        self.assertEqual(
            compact.reservation_bytes_by_namespace_kind,
            retained.reservation_bytes_by_namespace_kind,
        )
        report = compact.report()
        self.assertFalse(report["retain_reservations"])
        self.assertEqual(report["retained_reservation_count"], 0)
        self.assertEqual(
            report["resources"]["a"]["reservation_count"], 2)
        self.assertEqual(
            report["resources"]["a"]["reservation_bytes"], 130)
        self.assertEqual(
            report["namespace_kinds"],
            [
                {
                    "namespace": "lifecycle",
                    "kind": "migration",
                    "reservation_count": 2,
                    "reservation_bytes": 300,
                },
                {
                    "namespace": "pool",
                    "kind": "batch",
                    "reservation_count": 2,
                    "reservation_bytes": 70,
                },
            ],
        )

    def test_calendar_mode_must_be_boolean(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            ResourceCalendar(retain_reservations=1)


if __name__ == "__main__":
    unittest.main()

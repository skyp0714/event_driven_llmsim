import dataclasses
import json
from pathlib import Path
import random
import unittest

from serving.core.gpu_hbf_hybrid import (
    GPUHBFHybridNode,
    GPUHBFHybridSystem,
    HybridCall,
    HybridCallState,
    HybridExecution,
)
from serving.core.gpu_pd_latency import P4D4GPUHardware
from serving.core.hbf_comparison_workload import (
    CallSpec,
    ScheduledSession,
    SessionSpec,
)
from serving.core.hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
)
from serving.core.hbf_full_model_lifecycle import PlacementState
from serving.core.hbf_full_model_pool import derive_lpddr_workspace_bytes


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_call_spec(
        session_id, source_index, call_index, input_tokens,
        output_tokens, tool_duration_ns, cached_prefix_tokens):
    return CallSpec(
        session_id=session_id,
        source_index=source_index,
        call_index=call_index,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_duration_ns=tool_duration_ns,
        cached_prefix_tokens=cached_prefix_tokens,
        fresh_input_tokens=input_tokens - cached_prefix_tokens,
        lineage_status=None,
        inter_turn_gap_type=None,
    )


def make_schedule(
        source_index, calls, *,
        arrival_ns=0, offer_index=None):
    session_id = f"session-{source_index}"
    specs = tuple(
        make_call_spec(
            session_id,
            source_index,
            call_index,
            input_tokens,
            output_tokens,
            tool_duration_ns,
            cached_prefix_tokens,
        )
        for call_index, (
            input_tokens,
            output_tokens,
            tool_duration_ns,
            cached_prefix_tokens,
        ) in enumerate(calls)
    )
    return ScheduledSession(
        offer_index=(
            source_index if offer_index is None else offer_index),
        session=SessionSpec(
            source_index=source_index,
            session_id=session_id,
            source_arrival_time_ns=arrival_ns,
            source_session_identity_sha256=None,
            calls=specs,
        ),
        arrival_time_ns=arrival_ns,
        unit_interarrival=0.0,
        unit_arrival_time=0.0,
    )


class GPUHBFHybridTests(unittest.TestCase):
    def make_system(
            self, layout="tp4", *,
            gpu_hardware=None, hbf_hardware=None,
            validate_every_event=True):
        return GPUHBFHybridSystem(
            repo_root=REPO_ROOT,
            gpu_hardware=(
                P4D4GPUHardware()
                if gpu_hardware is None else gpu_hardware
            ),
            hbf_hardware=(
                HBFServerHardware()
                if hbf_hardware is None else hbf_hardware
            ),
            hbf_layout=layout,
            max_num_batched_tokens=256,
            max_num_seqs=16,
            max_prefill_chunk_tokens=128,
            validate_every_event=validate_every_event,
        )

    def make_node(self, layout="tp4", *, validate_every_event=True):
        return GPUHBFHybridNode(
            repo_root=REPO_ROOT,
            gpu_hardware=P4D4GPUHardware(),
            hbf_hardware=HBFServerHardware(),
            hbf_layout=layout,
            max_num_batched_tokens=256,
            max_num_seqs=16,
            max_prefill_chunk_tokens=128,
            validate_every_event=validate_every_event,
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

    def test_first_turn_gpu_and_ready_resume_hbf_for_all_layouts(self):
        for layout in ("dp8", "tp4", "tp8"):
            with self.subTest(layout=layout):
                system = self.make_system(layout)
                schedule = make_schedule(
                    0,
                    (
                        (32, 2, 10_000_000_000, 0),
                        (34, 2, 0, 33),
                    ),
                )
                completed = system.run((schedule,))
                self.assertEqual(len(completed), 2)
                calls = system.node.calls
                self.assertEqual(
                    calls[0].execution,
                    HybridExecution.GPU_FIRST_TURN,
                )
                self.assertEqual(
                    calls[1].execution,
                    HybridExecution.HBF_READY,
                )
                self.assertEqual(calls[0].state,
                                 HybridCallState.INTERNAL_COMPLETE)
                self.assertEqual(calls[1].state,
                                 HybridCallState.INTERNAL_COMPLETE)
                self.assertEqual(
                    system.node.hbf_layout.key, layout)
                self.assertEqual(
                    system.node.hbf_layout.tp_size
                    * system.node.hbf_layout.replicas,
                    8,
                )
                self.assertEqual(
                    system.node.gpu_hbm.p_used_bytes_per_rank, 0)
                self.assertEqual(
                    system.node.gpu_hbm.d_used_bytes_per_rank, 0)
                self.assertEqual(
                    system.node.hbf_lifecycle.sessions[
                        "session-0"].state,
                    PlacementState.ENDED,
                )

    def test_zero_tool_gap_routes_migration_inflight_resume_to_gpu(self):
        system = self.make_system()
        schedule = make_schedule(
            0,
            (
                (1_000, 1, 0, 0),
                (1_001, 1, 0, 1_000),
            ),
        )
        completed = system.run((schedule,))
        self.assertEqual(len(completed), 2)
        first, second = (
            system.node.calls[0], system.node.calls[1])
        self.assertEqual(
            first.execution, HybridExecution.GPU_FIRST_TURN)
        self.assertEqual(
            second.execution,
            HybridExecution.GPU_MIGRATION_INFLIGHT,
        )
        self.assertTrue(second.migration_inflight_at_route)
        self.assertEqual(
            second.route_reason,
            "migration_inflight_gpu_fallback",
        )
        metrics = system.node.hbf_lifecycle.metrics
        self.assertEqual(metrics.migrations_started, 1)
        self.assertEqual(metrics.migrations_committed, 0)
        self.assertEqual(metrics.migrations_stale, 1)
        self.assertEqual(metrics.hbf_resumes, 0)
        lifecycle_report = system.node.hbf_lifecycle.report()
        self.assertEqual(lifecycle_report["pending_job_count"], 0)
        self.assertTrue(all(
            value == 0
            for value in lifecycle_report[
                "group_reserved_per_card_bytes"].values()
        ))
        self.assertEqual(
            system.node.gpu_hbm.p_used_bytes_per_rank, 0)
        self.assertEqual(
            system.node.gpu_hbm.d_used_bytes_per_rank, 0)

    def test_migration_completion_wins_exact_timestamp_tie(self):
        node = self.make_node()
        first = self.runtime_call(
            0,
            0,
            0,
            input_tokens=1_000,
            output_tokens=2,
            has_successor=True,
        )
        node.submit(first, now_ns=0)
        while first.internal_completion_ns is None:
            event_ns = node.next_event_ns()
            self.assertIsNotNone(event_ns)
            node.advance(event_ns)
        migration_completion_ns = (
            node.hbf_lifecycle.next_completion_ns())
        self.assertIsNotNone(migration_completion_ns)
        second = self.runtime_call(
            1,
            1,
            migration_completion_ns,
            input_tokens=1_002,
            output_tokens=1,
            prefix_reuse_tokens=1_001,
            has_successor=False,
        )
        node.submit(second, now_ns=migration_completion_ns)
        self.assertEqual(
            second.execution, HybridExecution.HBF_READY)
        self.assertFalse(second.migration_inflight_at_route)
        completed = node.run_until_idle()
        self.assertIn(second, completed)
        self.assertEqual(
            node.hbf_lifecycle.metrics.migrations_committed, 1)
        self.assertEqual(
            node.hbf_lifecycle.metrics.migrations_stale, 0)

    def test_ready_hbf_call_has_no_per_call_network_round_trip(self):
        system = self.make_system("tp4")
        schedule = make_schedule(
            0,
            (
                (128, 3, 5_000_000_000, 0),
                (133, 3, 0, 130),
            ),
        )
        system.run((schedule,))
        self.assertEqual(
            system.node.calls[1].execution,
            HybridExecution.HBF_READY,
        )
        rdma_rows = [
            row
            for row in system.node.hbf_calendar.reservations
            if row.resource == "rdma-network"
        ]
        self.assertEqual(len(rdma_rows), 1)
        self.assertTrue(all(
            row.kind == "migration"
            and row.namespace == "hbf-lifecycle"
            for row in rdma_rows
        ))
        self.assertFalse(any(
            row.resource == "rdma-network"
            for row in system.node.gpu_calendar.reservations
        ))
        contract = system.report()["policy"]
        self.assertFalse(
            contract["hbf_ready_per_call_network_round_trip"])

    def test_output_one_lineage_is_exact_on_gpu_and_hbf(self):
        system = self.make_system()
        schedule = make_schedule(
            0,
            (
                (64, 1, 5_000_000_000, 0),
                (64, 1, 0, 64),
            ),
        )
        completed = system.run((schedule,))
        self.assertEqual(len(completed), 2)
        self.assertTrue(all(
            row.first_token_ns == row.completion_ns
            for row in completed
        ))
        self.assertTrue(all(row.tpot_ns is None for row in completed))
        first = system.node.calls[0].gpu_request
        second = system.node.calls[1].hbf_request
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.final_materialized_kv_tokens, 64)
        self.assertEqual(
            system.node.calls[1].prefix_reuse_tokens, 64)
        self.assertEqual(
            system.node.calls[1].operational_reuse_tokens, 64)
        self.assertEqual(second.hbf_prefix_tokens, 64)
        self.assertEqual(second.prefill_processed_tokens, 0)
        self.assertEqual(
            system.node.hbf_lifecycle.sessions[
                "session-0"].state,
            PlacementState.ENDED,
        )

    def test_raw_next_prefix_includes_unsaved_sampled_token(self):
        system = self.make_system()
        schedule = make_schedule(
            0,
            (
                (100, 3, 5_000_000_000, 0),
                # TraceLab commonly records input+output as the raw reuse.
                # The predecessor only materializes input+output-1 KV.
                (105, 1, 0, 103),
            ),
        )
        completed = system.run((schedule,))
        self.assertEqual(len(completed), 2)
        second = system.node.calls[1]
        self.assertEqual(second.prefix_reuse_tokens, 103)
        self.assertEqual(second.operational_reuse_tokens, 102)
        self.assertEqual(
            second.execution, HybridExecution.HBF_READY)
        self.assertEqual(second.hbf_request.hbf_prefix_tokens, 102)
        self.assertEqual(second.hbf_request.fresh_tokens, 3)
        self.assertEqual(
            system.node.metrics.operational_prefix_cap_calls, 1)
        self.assertEqual(
            system.node.metrics.operational_prefix_cap_tokens, 1)

    def test_finite_gpu_hbm_defers_instead_of_becoming_infinite(self):
        base = P4D4GPUHardware()
        gpu_hardware = dataclasses.replace(
            base,
            hbm_capacity_bytes_per_gpu=(
                base.model_weight_bytes_per_rank
                + base.runtime_reserve_bytes_per_gpu
                + 2_700_000_000
            ),
        )
        system = self.make_system(
            gpu_hardware=gpu_hardware,
            validate_every_event=False,
        )
        schedules = (
            make_schedule(
                0,
                (
                    (60_000, 1, 5_000_000_000, 0),
                    (60_000, 1, 0, 60_000),
                ),
                offer_index=0,
            ),
            make_schedule(
                1,
                (
                    (60_000, 1, 5_000_000_000, 0),
                    (60_000, 1, 0, 60_000),
                ),
                offer_index=1,
            ),
        )
        completed = system.run(schedules)
        self.assertEqual(len(completed), 4)
        self.assertGreater(
            system.node.metrics.gpu_hbm_capacity_deferrals, 0)
        hbm = system.node.gpu_hbm
        self.assertLessEqual(
            hbm.metrics.peak_p_bytes_per_rank,
            hbm.p_capacity_bytes_per_rank,
        )
        self.assertLessEqual(
            hbm.metrics.peak_d_bytes_per_rank,
            hbm.d_capacity_bytes_per_rank,
        )
        self.assertEqual(hbm.p_used_bytes_per_rank, 0)
        self.assertEqual(hbm.d_used_bytes_per_rank, 0)
        self.assertGreaterEqual(
            system.node.metrics.migration_hbm_releases, 2)

    def test_tp8_report_pins_replication_fabric_and_root_assumptions(self):
        system = self.make_system("tp8")
        system.run((
            make_schedule(0, ((16, 1, 0, 0),)),
        ))
        topology = system.node.report()["hbf_topology_contract"]
        self.assertEqual(topology["cards_per_tp_group"], 8)
        self.assertEqual(
            topology["physical_kv_replication_factor"], 2)
        self.assertEqual(topology["pcie_roots"], 2)
        self.assertEqual(topology["cards_per_pcie_root"], 4)
        self.assertIn(
            "spans both", topology["tp8_specific_assumption"])
        self.assertIn(
            "GQA", topology["tp8_specific_assumption"])

    def test_documented_gpu_stage_caps_are_plumbed_independently(self):
        system = GPUHBFHybridSystem(
            repo_root=REPO_ROOT,
            hbf_layout="tp4",
            max_num_batched_tokens=131_072,
            max_num_seqs=128,
            p_max_num_seqs=32,
            d_max_num_seqs=128,
            max_prefill_chunk_tokens=131_072,
        )
        system.run((
            make_schedule(0, ((16, 1, 0, 0),)),
        ))
        self.assertEqual(system.node.gpu_pool.p_max_num_seqs, 32)
        self.assertEqual(system.node.gpu_pool.d_max_num_seqs, 128)
        self.assertEqual(system.node.hbf_pool.max_num_seqs, 128)
        report = system.report()
        self.assertEqual(
            report["node"]["gpu_pool"]["p_max_num_seqs"], 32)
        self.assertEqual(
            report["node"]["gpu_pool"]["d_max_num_seqs"], 128)

    def test_sweep_mode_uses_compact_native_hbf_primitives(self):
        system = self.make_system(
            "tp4", validate_every_event=False)
        schedule = make_schedule(
            0,
            (
                (64, 2, 5_000_000_000, 0),
                (66, 1, 0, 65),
            ),
        )
        completed = system.run((schedule,))
        self.assertEqual(len(completed), 2)
        node = system.node
        self.assertFalse(node.hbf_lifecycle.validate_every_event)
        self.assertFalse(node.hbf_pool.validate_every_event)
        self.assertFalse(node.gpu_calendar.retain_reservations)
        self.assertFalse(node.hbf_calendar.retain_reservations)
        self.assertFalse(node.retain_detailed_history)
        self.assertEqual(node.gpu_calendar.reservations, [])
        self.assertEqual(node.hbf_calendar.reservations, [])
        self.assertEqual(node.gpu_pool.batch_history, [])
        self.assertEqual(node.gpu_pool.handoff_history, [])
        self.assertEqual(node.hbf_pool.batch_history, [])
        self.assertEqual(node.prepare_history, [])
        self.assertTrue(all(
            call.serving_request.token_completion_ns == []
            for call in node.calls.values()
        ))
        report = system.report()["node"]
        rdma = report["rdma_migration_summary"]
        self.assertFalse(rdma["reservation_detail_retained"])
        self.assertEqual(
            rdma["reservation_count"],
            node.hbf_lifecycle.metrics.migrations_started,
        )
        self.assertEqual(
            rdma["logical_bytes"],
            node.hbf_lifecycle.metrics.migration_logical_bytes,
        )
        node.assert_invariants()

    def test_tiny_lpddr_uses_gpu_fallback_without_decode_deadlock(self):
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
        system = GPUHBFHybridSystem(
            repo_root=REPO_ROOT,
            hbf_hardware=hardware,
            hbf_layout="tp8",
            max_num_batched_tokens=16,
            max_num_seqs=4,
            max_prefill_chunk_tokens=16,
        )
        schedules = tuple(
            make_schedule(
                source_index,
                (
                    (1, 1, 10_000_000_000, 0),
                    (1, 4, 0, 1),
                ),
                offer_index=source_index,
            )
            for source_index in range(2)
        )
        completed = system.run(schedules)
        self.assertEqual(len(completed), 4)
        resumes = [
            call for call in system.node.calls.values()
            if call.call_index == 1
        ]
        self.assertEqual(
            sum(call.execution == HybridExecution.HBF_READY
                for call in resumes),
            1,
        )
        fallback = [
            call for call in resumes
            if call.execution == HybridExecution.GPU_RECOMPUTE
        ]
        self.assertEqual(len(fallback), 1)
        self.assertEqual(
            fallback[0].route_reason,
            "hbf_lpddr_finish_capacity_fallback",
        )
        lifecycle = system.node.hbf_lifecycle
        self.assertEqual(
            lifecycle.metrics.lpddr_capacity_fallback_resumes, 1)
        self.assertTrue(all(
            lifecycle.lpddr_ledger.used_bytes(group_id) == 0
            for group_id in range(
                lifecycle.lpddr_ledger.group_count)
        ))
        system.assert_invariants()

    def test_context_shrink_invalidates_append_without_losing_lineage(self):
        system = self.make_system()
        schedule = make_schedule(
            0,
            (
                (200, 2, 5_000_000_000, 0),
                (202, 2, 0, 201),
                (40, 1, 0, 0),
            ),
        )
        completed = system.run((schedule,))
        self.assertEqual(len(completed), 3)
        self.assertEqual(
            system.node.calls[1].execution,
            HybridExecution.HBF_READY,
        )
        self.assertEqual(
            system.node.calls[2].execution,
            HybridExecution.HBF_READY,
        )
        third = system.node.calls[2].hbf_request
        self.assertIsNotNone(third)
        self.assertEqual(third.cached_tokens, 0)
        self.assertEqual(third.prefill_processed_tokens, 40)
        self.assertEqual(
            system.node.hbf_lifecycle.sessions[
                "session-0"].state,
            PlacementState.ENDED,
        )

    def test_randomized_multi_session_stress_is_deterministic_and_drains(self):
        def build_schedules():
            rng = random.Random(29)
            schedules = []
            for source_index in range(16):
                calls = []
                materialized = 0
                for call_index in range(5):
                    if call_index == 0:
                        input_tokens = rng.randint(8, 64)
                        cached = 0
                    elif rng.random() < 0.15:
                        input_tokens = rng.randint(8, 40)
                        cached = 0
                    else:
                        cached = materialized
                        input_tokens = cached + rng.randint(0, 8)
                    output_tokens = rng.randint(1, 4)
                    tool_duration_ns = (
                        0 if call_index % 2 == 0
                        else 2_000_000_000
                    )
                    calls.append((
                        input_tokens,
                        output_tokens,
                        tool_duration_ns,
                        cached,
                    ))
                    materialized = (
                        input_tokens + output_tokens - 1)
                schedules.append(make_schedule(
                    source_index,
                    calls,
                    arrival_ns=(source_index % 4) * 1_000_000,
                    offer_index=source_index,
                ))
            return tuple(schedules)

        def run_once():
            system = self.make_system(
                "tp4", validate_every_event=False)
            completed = system.run(build_schedules())
            system.assert_invariants()
            self.assertEqual(len(completed), 80)
            self.assertEqual(
                system.node.metrics.gpu_first_turn_calls, 16)
            self.assertGreater(
                system.node.metrics.gpu_migration_inflight_calls, 0)
            self.assertGreater(system.node.metrics.hbf_calls, 0)
            self.assertEqual(
                system.node.gpu_hbm.p_used_bytes_per_rank, 0)
            self.assertEqual(
                system.node.gpu_hbm.d_used_bytes_per_rank, 0)
            self.assertEqual(
                system.node.hbf_lifecycle.report()[
                    "pending_job_count"],
                0,
            )
            report = system.report()
            json.dumps(report, sort_keys=True)
            self.assertEqual(
                report["call_full_drain"]["identity_count"], 80)
            self.assertEqual(
                report["session_full_drain"]["identity_count"], 16)
            return (
                report["call_full_drain"]["completion_order_sha256"],
                report["session_full_drain"][
                    "completion_order_sha256"],
                tuple(
                    (request.key, request.first_token_ns,
                     request.completion_ns)
                    for request in completed
                ),
            )

        self.assertEqual(run_once(), run_once())


if __name__ == "__main__":
    unittest.main()

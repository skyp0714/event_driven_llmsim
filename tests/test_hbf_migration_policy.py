import dataclasses
from pathlib import Path
import unittest

from serving.core.gpu_hbf_hybrid import (
    GPUHBFHybridSystem,
    HybridExecution,
    MigrationPolicy,
)
from serving.core.hbf_comparison_workload import (
    CallSpec,
    ScheduledSession,
    SessionSpec,
)
from serving.core.hbf_full_model_lifecycle import PlacementState


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_schedule(source_index, calls, *, arrival_ns=0):
    session_id = f"policy-session-{source_index}"
    call_specs = tuple(
        CallSpec(
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
        for call_index, (
            input_tokens,
            output_tokens,
            tool_duration_ns,
            cached_prefix_tokens,
        ) in enumerate(calls)
    )
    return ScheduledSession(
        offer_index=source_index,
        session=SessionSpec(
            source_index=source_index,
            session_id=session_id,
            source_arrival_time_ns=arrival_ns,
            source_session_identity_sha256=None,
            calls=call_specs,
        ),
        arrival_time_ns=arrival_ns,
        unit_interarrival=0.0,
        unit_arrival_time=0.0,
    )


class HBFMigrationPolicyTests(unittest.TestCase):
    @staticmethod
    def make_system(*, migration_policy=None):
        kwargs = {
            "repo_root": REPO_ROOT,
            "hbf_layout": "tp4",
            "max_num_batched_tokens": 256,
            "max_num_seqs": 16,
            "max_prefill_chunk_tokens": 128,
        }
        if migration_policy is not None:
            kwargs["migration_policy"] = migration_policy
        return GPUHBFHybridSystem(**kwargs)

    @staticmethod
    def completion_signature(system):
        return tuple(
            (
                row.key,
                row.release_ns,
                row.first_token_ns,
                row.completion_ns,
            )
            for row in system.completed_requests
        )

    @staticmethod
    def execution_signature(system):
        return tuple(
            system.node.calls[request_id].execution
            for request_id in sorted(system.node.calls)
        )

    @staticmethod
    def rdma_signature(system):
        return tuple(
            (
                row.start_ns,
                row.end_ns,
                row.byte_count,
                row.job_id,
            )
            for row in system.node.hbf_calendar.reservations
            if row.resource == "rdma-network"
        )

    def test_explicit_eager_is_backward_compatible_with_default(self):
        schedules = (
            make_schedule(
                0,
                (
                    (1_000, 2, 0, 0),
                    (1_002, 1, 0, 1_001),
                ),
            ),
            make_schedule(
                1,
                (
                    (96, 2, 2_000_000_000, 0),
                    (99, 1, 0, 97),
                ),
            ),
        )
        default = self.make_system()
        explicit = self.make_system(migration_policy="eager")
        default.run(schedules)
        explicit.run(schedules)

        self.assertEqual(
            self.completion_signature(default),
            self.completion_signature(explicit),
        )
        self.assertEqual(
            self.execution_signature(default),
            self.execution_signature(explicit),
        )
        self.assertEqual(
            dataclasses.asdict(default.node.metrics),
            dataclasses.asdict(explicit.node.metrics),
        )
        self.assertEqual(
            dataclasses.asdict(default.node.hbf_lifecycle.metrics),
            dataclasses.asdict(explicit.node.hbf_lifecycle.metrics),
        )
        self.assertEqual(
            self.rdma_signature(default),
            self.rdma_signature(explicit),
        )
        self.assertGreater(
            default.node.hbf_lifecycle.metrics.migrations_started, 0)

    def test_idle_delay_exact_tie_resume_wins_without_migration(self):
        delay_ns = 100_000_000
        system = self.make_system(migration_policy="delay_100ms")
        system.run((
            make_schedule(
                0,
                (
                    (64, 2, delay_ns, 0),
                    (67, 1, 0, 65),
                ),
            ),
        ))

        first = system.node.calls[0]
        second = system.node.calls[1]
        self.assertEqual(first.tool_duration_ns, delay_ns)
        self.assertEqual(
            second.release_ns,
            first.user_completion_ns + delay_ns,
        )
        self.assertEqual(
            second.execution,
            HybridExecution.GPU_CAPACITY_FALLBACK,
        )
        self.assertFalse(second.migration_inflight_at_route)
        self.assertEqual(
            second.route_reason,
            "hbf_capacity_unavailable_gpu_retained",
        )
        self.assertEqual(
            system.node.metrics.migration_triggers_scheduled, 1)
        self.assertEqual(
            system.node.metrics.migration_triggers_canceled, 1)
        self.assertEqual(
            system.node.metrics.migration_triggers_launched, 0)
        self.assertEqual(
            system.node.hbf_lifecycle.metrics.migrations_started, 0)
        self.assertEqual(self.rdma_signature(system), ())

    def test_after_first_tool_migrates_only_after_first_resume(self):
        system = self.make_system(
            migration_policy="after_first_tool")
        system.run((
            make_schedule(
                0,
                (
                    (64, 2, 100_000_000, 0),
                    (70, 2, 5_000_000_000, 65),
                    (74, 1, 0, 71),
                ),
            ),
        ))

        executions = self.execution_signature(system)
        self.assertEqual(
            executions,
            (
                HybridExecution.GPU_FIRST_TURN,
                HybridExecution.GPU_CAPACITY_FALLBACK,
                HybridExecution.HBF_READY,
            ),
        )
        lifecycle = system.node.hbf_lifecycle
        self.assertEqual(lifecycle.metrics.migrations_started, 1)
        self.assertEqual(lifecycle.metrics.migrations_committed, 1)
        self.assertEqual(lifecycle.metrics.migrations_stale, 0)
        self.assertEqual(
            system.node.metrics.migration_triggers_scheduled, 0)
        self.assertEqual(
            lifecycle.sessions["policy-session-0"].state,
            PlacementState.ENDED,
        )

    def test_never_policy_keeps_all_resumes_on_gpu_and_drains(self):
        system = self.make_system(migration_policy="never")
        system.run((
            make_schedule(
                0,
                (
                    (64, 2, 1_000_000_000, 0),
                    (70, 2, 1_000_000_000, 65),
                    (74, 1, 0, 71),
                ),
            ),
        ))

        self.assertEqual(
            self.execution_signature(system),
            (
                HybridExecution.GPU_FIRST_TURN,
                HybridExecution.GPU_CAPACITY_FALLBACK,
                HybridExecution.GPU_CAPACITY_FALLBACK,
            ),
        )
        lifecycle = system.node.hbf_lifecycle
        self.assertEqual(lifecycle.metrics.migrations_started, 0)
        self.assertEqual(lifecycle.metrics.hbf_resumes, 0)
        self.assertEqual(
            system.node.metrics.migration_triggers_scheduled, 0)
        self.assertEqual(self.rdma_signature(system), ())
        self.assertEqual(
            system.node.gpu_hbm.p_used_bytes_per_rank, 0)
        self.assertEqual(
            system.node.gpu_hbm.d_used_bytes_per_rank, 0)
        self.assertEqual(
            lifecycle.sessions["policy-session-0"].state,
            PlacementState.ENDED,
        )
        system.assert_invariants()

    def test_load_aware_is_deterministic_and_ignores_future_gap(self):
        policy = MigrationPolicy(
            key="test_load_aware",
            mode="load_aware",
            idle_delay_ns=10_000_000,
            load_retry_ns=5_000_000,
            load_hysteresis=0.1,
            gpu_hbm_high_watermark=1.0,
            max_load_deferrals=2,
        )

        def run(gap_ns):
            system = self.make_system(migration_policy=policy)
            # Pin only current-load observations.  The policy must retry
            # deterministically and must not consult the future release gap.
            system.node._hbf_compute_pressure = lambda: 1.0
            system.node._gpu_compute_pressure = lambda: 0.0
            system.run((
                make_schedule(
                    0,
                    (
                        (128, 2, gap_ns, 0),
                        (132, 1, 0, 129),
                    ),
                ),
            ))
            return system

        one_second = run(1_000_000_000)
        same_again = run(1_000_000_000)
        two_seconds = run(2_000_000_000)

        self.assertEqual(
            self.completion_signature(one_second),
            self.completion_signature(same_again),
        )
        self.assertEqual(
            self.rdma_signature(one_second),
            self.rdma_signature(same_again),
        )
        self.assertEqual(
            self.rdma_signature(one_second),
            self.rdma_signature(two_seconds),
        )
        for system in (one_second, same_again, two_seconds):
            self.assertEqual(
                system.node.metrics.migration_load_deferrals, 2)
            self.assertEqual(
                system.node.metrics.migration_triggers_scheduled, 1)
            self.assertEqual(
                system.node.metrics.migration_triggers_launched, 1)
            self.assertEqual(
                system.node.hbf_lifecycle.metrics.migrations_started, 1)
            self.assertEqual(
                system.node.calls[1].execution,
                HybridExecution.HBF_READY,
            )

    def test_delayed_policy_forces_migration_under_gpu_hbm_pressure(self):
        system = self.make_system(migration_policy="delay_1000ms")
        system.node.gpu_hbm.d_capacity_bytes_per_rank = (
            system.node.gpu_hardware.kv_capacity_bytes_per_rank(70)
        )
        schedules = tuple(
            make_schedule(
                source_index,
                (
                    (64, 2, 5_000_000_000, 0),
                    (67, 1, 0, 65),
                ),
            )
            for source_index in range(3)
        )

        self.assertEqual(len(system.run(schedules)), 6)
        self.assertGreater(
            system.node.metrics.gpu_hbm_pressure_migration_starts, 0)
        self.assertEqual(
            system.node.metrics.gpu_hbm_pressure_evictions, 0)
        self.assertEqual(
            system.node.gpu_hbm.d_used_bytes_per_rank, 0)
        system.assert_invariants()

    def test_never_policy_evicts_idle_gpu_kv_under_hbm_pressure(self):
        system = self.make_system(migration_policy="never")
        system.node.gpu_hbm.d_capacity_bytes_per_rank = (
            system.node.gpu_hardware.kv_capacity_bytes_per_rank(70)
        )
        schedules = tuple(
            make_schedule(
                source_index,
                (
                    (64, 2, 5_000_000_000, 0),
                    (67, 1, 0, 65),
                ),
            )
            for source_index in range(3)
        )

        self.assertEqual(len(system.run(schedules)), 6)
        self.assertEqual(
            system.node.hbf_lifecycle.metrics.migrations_started, 0)
        self.assertGreater(
            system.node.metrics.gpu_hbm_pressure_evictions, 0)
        self.assertGreater(
            system.node.hbf_lifecycle.metrics
            .gpu_ready_hbm_pressure_evictions,
            0,
        )
        self.assertEqual(
            system.node.gpu_hbm.d_used_bytes_per_rank, 0)
        system.assert_invariants()


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from serving.core.agentic_kv import AgenticKVConfig
from serving.core.online_measurement import (
    OnlineHBMOccupancyAccounting,
    OnlineModelComputeAccounting,
    StrictInfiniteHBMOracle,
    configure_strict_oracle,
    measurement_target_reached,
)
from serving.core.request import Batch, Request


class _Memory:
    weight = 100
    npu_mem = 1000
    npu_physical_mem = 1200
    npu_runtime_reserve_bytes = 200
    npu_allocatable_mem = 1000
    npu_used = 100

    @staticmethod
    def get_kv(tokens):
        return int(tokens) * 10


class _Manager:
    def __init__(self):
        fields = (
            "cpu_hits", "ssd_hits", "dropped_misses", "capacity_drops",
            "hbm_capacity_demotions", "hbm_capacity_drops",
            "cpu_capacity_evictions", "ssd_capacity_evictions",
            "ssd_capacity_admission_drops",
            "capacity_induced_recompute_tokens",
            "policy_avoidable_recompute_tokens",
            "active_recompute_preemptions", "active_cpu_swap_preemptions",
            "pd_active_prefill_recompute_preemptions",
            "pd_active_prefill_recompute_tokens",
            "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute",
            "pd_chunk_cancelled_admissions",
            "pd_chunk_cancelled_admission_wait_ns",
            "pd_chunk_cancelled_admission_critical_wait_ns",
        )
        self.metrics = SimpleNamespace(**{field: 0 for field in fields})

    @staticmethod
    def _hbm_logically_reserved(instance_id):
        return 0


class OnlineMeasurementTests(unittest.TestCase):
    def test_queue_wait_is_write_once_across_chunks(self):
        request = Request(1, "model", 100, 110, 0, 0)
        request.ready_time = 25
        request.scheduler_resource_ready_time_ns = 30
        request.set_que_delay(40)
        request.set_que_delay(90)
        self.assertEqual(request.first_schedule_time_ns, 40)
        self.assertEqual(request.queuing_delay, 40)
        self.assertEqual(request.scheduler_queue_wait_ns, 10)

    def test_provider_compute_and_completion_window(self):
        request = Request(1, "model", 100, 110, 0, 0)
        request.agentic_kv_recompute_tokens = 50
        batch = Batch(
            0, "model", 100, 0, [100], [], 1, 0,
            [100], [0], [], 10, 0,
        )
        batch.requests.append(request)
        batch.scheduled_tokens = {request.id: 100}
        batch.model_compute_ns = 80
        batch.recompute_model_compute_ns = 30
        batch.online_long_context_experiment = {
            "mode": "dca_dense_full_attention_sensitivity",
            "paper_interpretation": {
                "absolute_1m_latency": "sensitivity_only",
            },
        }
        scheduler = SimpleNamespace(
            inflight=[batch], start_npu=0, num_npus=1,
            pd_type=None, instance_id=0,
        )
        accounting = OnlineModelComputeAccounting()
        observation = accounting.prepare_completion(
            scheduler, 1, 0, 110)
        self.assertEqual(observation.recompute_query_tokens, 50)
        accounting.record_completion(observation, batch)
        summary = accounting.summary(0, 110)
        self.assertEqual(summary["attribution"], "provider_comp_critical_path")
        self.assertEqual(summary["total_model_compute_ns"], 80)
        self.assertEqual(summary["recompute_model_compute_ns"], 30)
        self.assertEqual(
            summary["long_context_experiment"]["paper_interpretation"][
                "absolute_1m_latency"
            ],
            "sensitivity_only",
        )
        self.assertEqual(accounting.summary(110, 200)["completed_batches"], 0)
        batch_size = summary["real_batch_size"]
        self.assertEqual(batch_size["completed_batch_count"], 1)
        self.assertEqual(batch_size["non_dummy_completed_batch_count"], 1)
        self.assertEqual(batch_size["dp_dummy_completed_batch_count"], 0)
        self.assertEqual(
            batch_size["mean_real_requests_per_non_dummy_batch"], 1)
        self.assertEqual(
            batch_size["by_pd_type"]["colocated"]
            ["total_real_request_memberships"],
            1,
        )

    def test_active_prefill_frontier_unions_with_agentic_recompute(self):
        request = Request(2, "model", 100, 101, 0, 0)
        request.active_prefill_recompute_frontier_tokens = 60
        request.agentic_kv_hit_tokens = 50
        request.agentic_kv_recompute_tokens = 50
        batch = Batch(
            0, "model", 80, 0, [80], [], 1, 0,
            [80], [0], [], 10, 0,
        )
        batch.requests.append(request)
        batch.scheduled_tokens = {request.id: 80}
        scheduler = SimpleNamespace(
            inflight=[batch], start_npu=0, num_npus=1,
            pd_type=None, instance_id=0,
        )

        observation = OnlineModelComputeAccounting().prepare_completion(
            scheduler, 1, 0, 110)

        # [0,60) active replay union [50,100) session recompute covers the
        # complete [0,80) chunk exactly once, not 60+30 with overlap.
        self.assertEqual(observation.total_query_tokens, 80)
        self.assertEqual(observation.recompute_query_tokens, 80)

    def test_real_batch_size_separates_zero_request_dp_dummy(self):
        accounting = OnlineModelComputeAccounting()
        scheduler = SimpleNamespace(
            inflight=[], start_npu=0, num_npus=1,
            pd_type="decode", instance_id=0,
        )
        real_request = Request(1, "model", 1, 2, 0, 0)
        for batch_id, requests in ((0, [real_request]), (1, [])):
            batch = Batch(
                batch_id, "model", 1, 1, [1], [], 0, 1,
                [], [], [1], batch_id * 10, 0,
            )
            batch.requests.extend(requests)
            batch.scheduled_tokens = {
                request.id: 1 for request in requests
            }
            scheduler.inflight = [batch]
            observation = accounting.prepare_completion(
                scheduler, batch_id + 1, 0, batch_id * 10 + 5)
            accounting.record_completion(observation, batch)

        batch_size = accounting.summary()["real_batch_size"]
        self.assertEqual(batch_size["completed_batch_count"], 2)
        self.assertEqual(batch_size["non_dummy_completed_batch_count"], 1)
        self.assertEqual(batch_size["dp_dummy_completed_batch_count"], 1)
        self.assertEqual(batch_size["total_real_request_memberships"], 1)
        self.assertEqual(
            batch_size["mean_real_requests_per_non_dummy_batch"], 1)
        self.assertEqual(
            batch_size[
                "mean_real_requests_per_completed_batch_including_dummy"],
            0.5,
        )

    def test_hbm_occupancy_clips_and_replaces_same_timestamp(self):
        memory = SimpleNamespace(
            weight=100,
            npu_allocatable_mem=1000,
            npu_used=100,
        )
        scheduler = SimpleNamespace(
            instance_id=0, pd_type="decode", memory=memory)
        manager = SimpleNamespace(entries={}, logical=0)
        manager._hbm_logically_reserved = lambda instance_id: manager.logical
        accounting = OnlineHBMOccupancyAccounting([scheduler])
        accounting.observe(0, manager)

        memory.npu_used = 500
        manager.entries = {
            "idle": SimpleNamespace(
                instance_id=0,
                location="hbm",
                per_rank_bytes=100,
            ),
        }
        manager.logical = 200
        accounting.observe(10, manager)

        # The final state at t=10 is authoritative. Its reservation exceeds
        # current physical slack, so 100 bytes are backed by a still-physical
        # reclaim victim and must remain a non-additive overlay.
        manager.entries["idle"].per_rank_bytes = 150
        manager.logical = 600
        accounting.observe(10, manager)

        memory.npu_used = 300
        manager.entries = {}
        manager.logical = 0
        accounting.observe(20, manager)
        accounting.observe(30, manager)

        report = accounting.summary(5, 25)
        categories = report["aggregate"]["categories"]
        self.assertEqual(report["window_duration_ns"], 20)
        self.assertEqual(
            report["coverage"]["same_timestamp_replacement_count"], 1)
        self.assertEqual(
            categories["physical_idle_reusable"]
            ["average_per_rank_bytes"],
            75,
        )
        self.assertEqual(
            categories["physical_non_idle_active"]
            ["average_per_rank_bytes"],
            175,
        )
        self.assertEqual(
            categories["physical_free"]["average_per_rank_bytes"], 650)
        self.assertEqual(
            categories["logical_destination_admission_reservation"]
            ["average_per_rank_bytes"],
            300,
        )
        self.assertEqual(
            categories["reserved_free_slack"]["average_per_rank_bytes"],
            250,
        )
        self.assertEqual(
            categories["future_reclaim_backed_reservation"]
            ["average_per_rank_bytes"],
            50,
        )
        self.assertEqual(
            categories["unclaimed_allocatable_slack"]
            ["average_per_rank_bytes"],
            400,
        )
        self.assertTrue(report["conservation"]["passed"])
        self.assertAlmostEqual(
            report["aggregate"]
            ["average_physical_occupied_utilization_fraction"],
            250 / 900,
        )
        self.assertEqual(
            report["per_instance"]["0"]
            ["peak_physical_occupied_per_rank_bytes"],
            400,
        )
        self.assertEqual(
            report["per_instance"]["0"]
            ["peak_reservation_adjusted_claim_per_rank_bytes"],
            900,
        )

    def test_hbm_occupancy_fails_closed_on_unowned_physical_deficit(self):
        memory = SimpleNamespace(
            weight=100,
            npu_allocatable_mem=1000,
            npu_used=150,
        )
        scheduler = SimpleNamespace(
            instance_id=0, pd_type=None, memory=memory)
        manager = SimpleNamespace(
            entries={
                "idle": SimpleNamespace(
                    instance_id=0,
                    location="hbm",
                    per_rank_bytes=100,
                ),
            },
            _hbm_logically_reserved=lambda instance_id: 0,
        )
        accounting = OnlineHBMOccupancyAccounting([scheduler])
        with self.assertRaisesRegex(RuntimeError, "negative ownership"):
            accounting.observe(0, manager)

    def test_hbm_occupancy_requires_window_coverage(self):
        memory = SimpleNamespace(
            weight=100,
            npu_allocatable_mem=1000,
            npu_used=100,
        )
        scheduler = SimpleNamespace(
            instance_id=0, pd_type=None, memory=memory)
        manager = SimpleNamespace(
            entries={},
            _hbm_logically_reserved=lambda instance_id: 0,
        )
        accounting = OnlineHBMOccupancyAccounting([scheduler])
        accounting.observe(10, manager)
        accounting.observe(20, manager)
        with self.assertRaisesRegex(RuntimeError, "do not cover"):
            accounting.summary(0, 20)

    def test_provider_contract_cannot_change_between_batches(self):
        accounting = OnlineModelComputeAccounting()
        observation = SimpleNamespace(
            iteration_service_ns=1,
            attributed_recompute_iteration_service_ns=0,
        )
        first = SimpleNamespace(
            model_compute_ns=1,
            recompute_model_compute_ns=0,
            online_long_context_experiment={"mode": "first"},
        )
        accounting.record_completion(observation, first)
        second = SimpleNamespace(
            model_compute_ns=1,
            recompute_model_compute_ns=0,
            online_long_context_experiment={"mode": "second"},
        )
        with self.assertRaisesRegex(RuntimeError, "changed between"):
            accounting.record_completion(observation, second)
        self.assertEqual(accounting.completed_batches, 1)

    def test_strict_oracle_installs_proof_bound_and_validates(self):
        scheduler = SimpleNamespace(
            instance_id=0, block_size=16, memory=_Memory())
        oracle = StrictInfiniteHBMOracle([scheduler], [100, 200])
        self.assertGreater(scheduler.memory.npu_mem, 1000)
        self.assertEqual(
            scheduler.memory.npu_allocatable_mem,
            scheduler.memory.npu_mem,
        )
        request = Request(1, "model", 100, 110, 0, 0)
        request.session_id = "s"
        request.sub_request_index = 1
        request.prefix_reuse_tokens = 80
        request.agentic_kv_source = "hbm"
        report = oracle.validate(_Manager(), [request])
        self.assertTrue(report["passed"])
        self.assertGreater(
            report["per_instance"]["0"]["minimum_slack_per_rank_bytes"],
            0,
        )
        self.assertEqual(
            report["per_instance"]["0"]["physical_per_rank_bytes"], 1200)
        self.assertEqual(
            report["per_instance"]["0"]["runtime_reserve_per_rank_bytes"],
            200,
        )

    def test_oracle_config_and_measurement_target(self):
        config = configure_strict_oracle(AgenticKVConfig(policy="tiered"))
        self.assertEqual(config.policy, "preserve")
        self.assertEqual(config.demotion_mode, "capacity-only")
        admission = SimpleNamespace(
            warmup_completions=2, measure_completions=3)
        router = SimpleNamespace(session_admission_summary=lambda: {
            "completed_sessions": 5,
        })
        self.assertTrue(measurement_target_reached(router, admission))

        fixed_admission = SimpleNamespace(
            warmup_completions=1,
            measure_completions=2,
            measurement_cohort_selection="admission_order",
        )
        fixed_router = SimpleNamespace(
            measurement_target_reached=lambda: True,
            session_admission_summary=lambda: {"completed_sessions": 0},
        )
        self.assertTrue(measurement_target_reached(
            fixed_router, fixed_admission))
        with self.assertRaisesRegex(TypeError, "fixed-target tracking"):
            measurement_target_reached(router, fixed_admission)


if __name__ == "__main__":
    unittest.main()

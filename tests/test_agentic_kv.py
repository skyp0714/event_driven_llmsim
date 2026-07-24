import unittest
from pathlib import Path
from types import SimpleNamespace

from serving.core.agentic_kv import (
    AgenticKVConfig,
    AgenticKVManager,
    IdleKVEntry,
    KVLocation,
    PendingHBMAllocation,
    PendingSourceRelease,
    SSDRecord,
    TransferReservation,
)
from serving.core.endurance_model import RunWriteStats
from serving.core.memory_model import Device
from serving.core.memory_model import full_cluster_kv_bytes_per_token
from serving.core.request import Request
from serving.core.scheduler import Scheduler


class FakeMemory:
    def __init__(self, bytes_per_token=100, npu_mem=10**12, cpu_mem=10**13):
        self.bytes_per_token = bytes_per_token
        self.npu_mem = npu_mem
        self.cpu_mem = cpu_mem
        self.npu_used = 0
        self.cpu_used = 0

    def get_kv(self, tokens):
        return tokens * self.bytes_per_token

    def get_total_kv(self, request):
        blocks = (request.num_computed_tokens + 15) // 16
        return self.get_kv(blocks * 16)

    def allocate(self, num_bytes, device):
        if device == Device.NPU:
            if self.npu_used + num_bytes > self.npu_mem:
                raise RuntimeError("NPU capacity exceeded")
            self.npu_used += num_bytes
        elif device == Device.CPU:
            if self.cpu_used + num_bytes > self.cpu_mem:
                raise RuntimeError("CPU capacity exceeded")
            self.cpu_used += num_bytes
        else:
            raise AssertionError(device)

    def free(self, num_bytes, device):
        if device == Device.NPU:
            self.npu_used -= num_bytes
            if self.npu_used < 0:
                raise RuntimeError("negative NPU usage")
        elif device == Device.CPU:
            self.cpu_used -= num_bytes
            if self.cpu_used < 0:
                raise RuntimeError("negative CPU usage")
        else:
            raise AssertionError(device)


class FakeScheduler:
    def __init__(
            self, instance_id=0, node_id=None, num_npus=8,
            max_num_batched_tokens=131_072,
            long_prefill_token_threshold=131_072,
            pd_type=None, model="test-model", tp_size=1, pp_size=1,
            block_size=16, fp=2, kv_cache_dtype="auto",
            **memory_kwargs):
        self.instance_id = instance_id
        self.node_id = instance_id if node_id is None else node_id
        self.num_npus = num_npus
        self.memory = FakeMemory(**memory_kwargs)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.pd_type = pd_type
        self.model = model
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.block_size = block_size
        self.fp = fp
        self.kv_cache_dtype = kv_cache_dtype
        self.request = []
        self.inflight = []


class LinearPrefillProvider:
    def __init__(self, ns_per_token=10):
        self.ns_per_token = int(ns_per_token)

    def singleton_prefill_comp_ns(
            self, *, input_tokens, hit_tokens, max_chunk_tokens):
        del max_chunk_tokens
        return (
            (int(input_tokens) - int(hit_tokens)) * self.ns_per_token)

    def metadata(self):
        return {
            "name": "linear-test-provider",
            "scope": "online_trace_comp_nodes_only",
            "model": "test",
            "hardware": "test",
            "tp": 1,
            "ep": 1,
            "dtype": "test",
            "band": "test",
            "target_config_sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "producer_source_sha256": "c" * 64,
        }


class FakeRequest:
    def __init__(
            self, session_id="s", instance_id=0, tokens=100,
            prefix_reuse_tokens=0):
        self.session_id = session_id
        self.instance_id = instance_id
        self.num_computed_tokens = tokens
        self.prefix_reuse_tokens = prefix_reuse_tokens
        blocks = (int(tokens) + 15) // 16
        self.agentic_kv_completion_released_per_rank_bytes = (
            blocks * 16 * 100)


class AgenticKVTest(unittest.TestCase):
    def test_manager_rejects_generic_prefix_cache_double_accounting(self):
        scheduler = FakeScheduler()
        scheduler.enable_prefix_caching = True

        with self.assertRaisesRegex(
                ValueError, "generic Radix prefix caching"):
            AgenticKVManager(
                [scheduler], AgenticKVConfig(policy="preserve"))

    def test_measurement_cutoff_reports_active_and_queued_dma_tail(self):
        scheduler = FakeScheduler()
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(policy="preserve"))
        manager._reserve_transfer(
            kind="cpu_to_hbm", arrival_ns=0, service_ns=100,
            source_instance_id=0, target_instance_id=0,
            num_bytes=10, background=True, session_id="active")
        manager._reserve_transfer(
            kind="cpu_to_hbm", arrival_ns=10, service_ns=100,
            source_instance_id=0, target_instance_id=0,
            num_bytes=20, background=True, session_id="queued")

        tail = manager.transfer_tail_at(50)

        self.assertEqual(tail["active_service_jobs"], 1)
        self.assertEqual(tail["queued_not_started_jobs"], 1)
        self.assertEqual(tail["outstanding_bytes"], 30)
        self.assertEqual(tail["max_tail_ns"], 150)
        summary = manager.summary(50, measurement_censored=True)
        self.assertTrue(
            summary["measurement_cutoff_dma_tail"][
                "measurement_censored"])

    def test_measurement_cutoff_preserves_open_demotion_join_exposure(self):
        scheduler = FakeScheduler()
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(policy="tiered"))
        manager._begin_demotion_join(
            "joining", start_ns=100, complete_ns=1_600,
            migration_kind="hbm_to_cpu")

        audit = manager.censor_session("joining", cutoff_ns=1_000)
        summary = manager.summary(1_000, measurement_censored=True)
        breakdown = summary["time_breakdown"]

        join_audit = audit["source_demotion_join"]
        self.assertEqual(join_audit["elapsed_ns"], 900)
        self.assertEqual(join_audit["remaining_ns"], 600)
        self.assertEqual(
            manager.metrics.source_demotion_join_wait_ns, 0)
        self.assertEqual(
            breakdown["aggregate_source_demotion_join_wait_ns"], 0)
        self.assertEqual(
            breakdown["censored_source_demotion_join_count"], 1)
        self.assertEqual(
            breakdown[
                "censored_source_demotion_join_elapsed_ns_membership_sum"],
            900,
        )
        self.assertEqual(
            breakdown[
                "censored_source_demotion_join_remaining_ns_membership_sum"],
            600,
        )
        self.assertEqual(
            breakdown["migration_restore_exposure_union_ns"], 900)
        self.assertTrue(
            manager.validate_measurement_censoring_drained()["passed"])

    def test_ordinary_session_end_rejects_open_demotion_join(self):
        scheduler = FakeScheduler()
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(policy="tiered"))
        manager._begin_demotion_join(
            "joining", start_ns=100, complete_ns=1_600,
            migration_kind="hbm_to_cpu")

        with self.assertRaisesRegex(RuntimeError, "use censor_session"):
            manager.end_session("joining", now_ns=1_000)

    def test_ordinary_session_end_rejects_open_admission_waits(self):
        destination_manager = AgenticKVManager(
            [FakeScheduler()], AgenticKVConfig(policy="tiered"))
        destination_manager._begin_destination_admission_wait(
            "waiting", start_ns=0, operation_time_ns=0)
        with self.assertRaisesRegex(RuntimeError, "destination admission"):
            destination_manager.end_session("waiting", now_ns=1)

        transient_manager = AgenticKVManager(
            [FakeScheduler()], AgenticKVConfig(policy="tiered"))
        transient_manager._begin_transient_restore_wait("waiting", 0)
        with self.assertRaisesRegex(RuntimeError, "transient DRAM admission"):
            transient_manager.end_session("waiting", now_ns=1)

    def test_censor_counts_paused_transient_subset_without_double_count(self):
        manager = AgenticKVManager(
            [FakeScheduler()], AgenticKVConfig(policy="tiered"))
        manager._begin_destination_admission_wait(
            "waiting", start_ns=0, operation_time_ns=0)
        manager._begin_transient_restore_wait("waiting", 0)
        manager._pause_transient_restore_wait("waiting", 50)

        audit = manager.censor_session("waiting", cutoff_ns=100)
        breakdown = manager.summary(
            100, measurement_censored=True)["time_breakdown"]

        self.assertEqual(
            audit["destination_admission"]["elapsed_ns"], 100)
        self.assertEqual(
            audit["transient_dram_admission"]["active_ns"], 0)
        self.assertEqual(
            audit["transient_dram_admission"][
                "accumulated_paused_ns"],
            50,
        )
        self.assertEqual(
            audit["transient_dram_admission"]["elapsed_ns"], 50)
        self.assertEqual(
            breakdown["migration_restore_exposure_union_ns"], 100)

    def manager(self, **overrides):
        scheduler = FakeScheduler()
        values = {
            "policy": "tiered",
            "hbm_ttl_ms": 10,
            "cpu_ttl_ms": 20,
            "ssd_ttl_ms": 1000,
            "pcie_bandwidth_gbps": 50,
            "cpu_bandwidth_gbps": 200,
            "ssd_read_bandwidth_gbps": 100,
            "ssd_write_bandwidth_gbps": 80,
            "ssd_num_devices": 8,
        }
        values.update(overrides)
        return AgenticKVManager([scheduler], AgenticKVConfig(**values)), scheduler

    def queue_recompute_manager(
            self, source, ratio=1.0, min_wait_ms=0.0,
            cost_multiplier=0.0, provider=None):
        scheduler = FakeScheduler(
            instance_id=0, node_id=0, num_npus=1,
            bytes_per_token=100, npu_mem=10**9, cpu_mem=10**9,
        )
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="tiered_queue_recompute",
                demotion_mode="capacity-only",
                pcie_bandwidth_gbps=1e9,
                cpu_bandwidth_gbps=1e9,
                cpu_transfer_latency_us=0,
                ssd_read_bandwidth_gbps=1e9,
                ssd_read_latency_us=0,
                queue_recompute_wait_service_ratio=ratio,
                queue_recompute_min_wait_ms=min_wait_ms,
                queue_recompute_cost_guard_multiplier=cost_multiplier,
            ),
            queue_recompute_latency_providers=(
                {} if provider is None else {0: provider}),
        )
        entry = IdleKVEntry(
            session_id="queued", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=source, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[entry.session_id] = entry
        if source == KVLocation.CPU:
            scheduler.memory.allocate(entry.total_bytes, Device.CPU)
        elif source == KVLocation.SSD:
            manager.ssd_records[entry.session_id] = SSDRecord(
                tokens=entry.tokens,
                block_tokens=entry.block_tokens,
                bytes=entry.total_bytes,
                last_access_ns=0,
                accounted_until_ns=0,
            )
            manager.ssd_used_bytes = entry.total_bytes
        else:
            scheduler.memory.allocate(entry.per_rank_bytes, Device.NPU)
        return manager, scheduler, entry

    def partial_queue_recompute_manager(
            self, source, *, pd=False, decode_npu_mem=16_000,
            durable_cpu_copy=False):
        prefill = FakeScheduler(
            instance_id=0, node_id=0, num_npus=1,
            bytes_per_token=100, npu_mem=16_000, cpu_mem=100_000,
            max_num_batched_tokens=32,
            long_prefill_token_threshold=32,
            pd_type="prefill" if pd else None,
        )
        schedulers = [prefill]
        decode = None
        if pd:
            decode = FakeScheduler(
                instance_id=1, node_id=0, num_npus=1,
                bytes_per_token=100, npu_mem=decode_npu_mem,
                cpu_mem=100_000,
                max_num_batched_tokens=32,
                long_prefill_token_threshold=32,
                pd_type="decode",
            )
            schedulers.append(decode)
        provider_ns_per_token = (
            200 if source == KVLocation.SSD else 100)
        manager = AgenticKVManager(
            schedulers,
            AgenticKVConfig(
                policy="tiered_queue_recompute",
                demotion_mode="capacity-only",
                pcie_bandwidth_gbps=1,
                cpu_bandwidth_gbps=1,
                cpu_transfer_latency_us=0,
                ssd_read_bandwidth_gbps=1,
                ssd_read_latency_us=0,
                queue_recompute_wait_service_ratio=0,
                queue_recompute_cost_guard_multiplier=1,
                queue_recompute_prefill_headroom_chunks=1,
            ),
            queue_recompute_latency_providers={
                0: LinearPrefillProvider(provider_ns_per_token)},
        )
        source_entry = IdleKVEntry(
            session_id="queued", instance_id=0, tokens=128,
            block_tokens=128, per_rank_bytes=12_800,
            total_bytes=12_800, location=source,
            tier_since_ns=0, last_access_ns=0,
        )
        victim = IdleKVEntry(
            session_id="hbm-victim", instance_id=0, tokens=64,
            block_tokens=64, per_rank_bytes=6_400,
            total_bytes=6_400, location=KVLocation.HBM,
            tier_since_ns=0, last_access_ns=-1,
        )
        manager.entries[source_entry.session_id] = source_entry
        manager.entries[victim.session_id] = victim
        prefill.memory.allocate(victim.per_rank_bytes, Device.NPU)
        if source == KVLocation.CPU:
            prefill.memory.allocate(source_entry.total_bytes, Device.CPU)
        else:
            manager.ssd_records[source_entry.session_id] = SSDRecord(
                tokens=source_entry.tokens,
                block_tokens=source_entry.block_tokens,
                bytes=source_entry.total_bytes,
                last_access_ns=0,
                accounted_until_ns=0,
            )
            manager.ssd_used_bytes = source_entry.total_bytes
        if durable_cpu_copy:
            manager.ssd_records[source_entry.session_id] = SSDRecord(
                tokens=source_entry.tokens,
                block_tokens=source_entry.block_tokens,
                bytes=source_entry.total_bytes,
                last_access_ns=0,
                accounted_until_ns=0,
            )
            manager.ssd_used_bytes = source_entry.total_bytes
        return manager, prefill, decode, source_entry, victim

    def test_queue_recompute_rejects_invalid_ratio(self):
        for ratio in (-1, float("nan"), float("inf")):
            with self.subTest(ratio=ratio):
                with self.assertRaisesRegex(
                        ValueError,
                        "queue_recompute_wait_service_ratio"):
                    AgenticKVConfig(
                        policy="tiered_queue_recompute",
                        queue_recompute_wait_service_ratio=ratio,
                    ).validate()
        for multiplier in (-1, 0.5, float("nan"), float("inf"), True):
            with self.subTest(multiplier=multiplier):
                with self.assertRaisesRegex(
                        ValueError,
                        "queue_recompute_cost_guard_multiplier|"
                        "non-negative and finite"):
                    AgenticKVConfig(
                        policy="tiered_queue_recompute",
                        queue_recompute_cost_guard_multiplier=multiplier,
                    ).validate()
        for chunks in (0, 0.5, float("nan"), float("inf"), True):
            with self.subTest(headroom_chunks=chunks):
                with self.assertRaisesRegex(
                        ValueError,
                        "queue_recompute_prefill_headroom_chunks|"
                        "non-negative and finite"):
                    AgenticKVConfig(
                        policy="tiered_queue_recompute",
                        queue_recompute_prefill_headroom_chunks=chunks,
                    ).validate()

    def test_cost_aware_queue_recompute_requires_online_provider(self):
        with self.assertRaisesRegex(
                ValueError, "requires an online latency provider"):
            AgenticKVManager(
                [FakeScheduler()],
                AgenticKVConfig(
                    policy="tiered_queue_recompute",
                    queue_recompute_cost_guard_multiplier=1.25,
                ),
            )

    def test_config_rejects_nonfinite_boolean_and_fractional_hardware_values(self):
        invalid_cases = (
            ({"pcie_bandwidth_gbps": True}, "positive and finite"),
            ({"pcie_bandwidth_gbps": float("nan")},
             "positive and finite"),
            ({"cpu_transfer_latency_us": float("inf")},
             "non-negative and finite"),
            ({"ssd_num_devices": 1.5}, "positive integers"),
            ({"block_size": True}, "positive integers"),
            ({"keep_ssd_copy_on_read": 1}, "must be boolean"),
            ({"queue_recompute_wait_service_ratio": True},
             "queue_recompute_wait_service_ratio"),
        )
        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides):
                config = AgenticKVConfig(policy="tiered", **overrides)
                with self.assertRaisesRegex(ValueError, message):
                    config.validate()

    def test_queue_recompute_partial_cpu_restores_only_block_prefix(self):
        manager, scheduler, _, source_entry, victim = (
            self.partial_queue_recompute_manager(
                KVLocation.CPU, durable_cpu_copy=True))
        source_bytes = source_entry.total_bytes

        prep = manager.prepare_request(
            "queued", 0, 128, 1024, 0,
            residency_at_return=KVLocation.CPU,
        )

        self.assertEqual(prep.source, KVLocation.CPU)
        self.assertEqual(prep.hit_tokens, 64)
        self.assertEqual(prep.recompute_tokens, 64)
        self.assertEqual(prep.restored_bytes, 6400)
        self.assertEqual(manager.metrics.cpu_to_hbm_bytes, 6400)
        self.assertEqual(
            manager.metrics.queue_recompute_partial_restore_decisions, 1)
        self.assertEqual(manager.metrics.queue_recompute_drop_decisions, 0)
        self.assertEqual(victim.location, KVLocation.HBM)
        self.assertIsNone(victim.migration_kind)
        self.assertEqual(scheduler.memory.cpu_used, source_bytes)
        self.assertIn("queued", manager.ssd_records)
        pending = next(
            item for item in manager.pending_source_releases
            if item.entry is source_entry)
        self.assertEqual(pending.ready_ns, prep.ready_time_ns)
        self.assertTrue(pending.remove_ssd_record)
        manager.config.ssd_capacity_gb = 0.00002
        self.assertFalse(manager._ensure_ssd_capacity(
            "competing", 12_800, prep.ready_time_ns - 1))
        self.assertIn("queued", manager.ssd_records)
        event = next(
            item for item in manager.events
            if item.get("event") == "queue_recompute_partial")
        self.assertEqual(event["reusable_tokens_R"], 128)
        self.assertEqual(event["selected_prefix_tokens_H"], 64)
        self.assertEqual(event["dropped_suffix_tokens"], 64)
        self.assertEqual(event["selected_restore_bytes"], 6400)
        self.assertEqual(event["dropped_suffix_bytes"], 6400)
        self.assertEqual(
            event["selection_scope"],
            "contiguous_block_aligned_prefix",
        )
        self.assertEqual(
            event["source_pin_scope"],
            "full_physical_source_until_prefix_dma_complete",
        )
        self.assertTrue(event["capacity_headroom_snapshot"]["feasible"])
        self.assertFalse(
            event["pd_first_chunk_immediate_admission_guaranteed"])

        manager.advance(prep.ready_time_ns - 1)
        self.assertEqual(scheduler.memory.cpu_used, source_bytes)
        self.assertIn("queued", manager.ssd_records)
        manager.advance(prep.ready_time_ns)
        self.assertEqual(scheduler.memory.cpu_used, 0)
        self.assertNotIn("queued", manager.ssd_records)
        self.assertEqual(manager.ssd_used_bytes, 0)
        self.assertTrue(manager._ensure_ssd_capacity(
            "competing", 12_800, prep.ready_time_ns))
        audit = manager.summary(
            prep.ready_time_ns)["queue_recompute_policy"][
                "accounting_invariants"]
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["logical_session_drop_count"], 0)

    def test_queue_recompute_partial_ssd_stages_only_selected_prefix(self):
        manager, scheduler, _, source_entry, _ = (
            self.partial_queue_recompute_manager(KVLocation.SSD))

        prep = manager.prepare_request(
            "queued", 0, 128, 1024, 0,
            residency_at_return=KVLocation.SSD,
        )

        self.assertEqual(prep.source, KVLocation.SSD)
        self.assertEqual((prep.hit_tokens, prep.recompute_tokens), (64, 64))
        self.assertEqual(prep.restored_bytes, 6400)
        self.assertEqual(manager.metrics.ssd_to_cpu_stage_bytes, 6400)
        self.assertEqual(manager.metrics.cpu_stage_to_hbm_bytes, 6400)
        self.assertEqual(manager.metrics.ssd_to_hbm_bytes, 6400)
        self.assertEqual(manager.metrics.ssd_host_read_bytes, 6400)
        transient = next(
            item for item in manager.events
            if item.get("event") == "transient_dram_reserve")
        self.assertEqual(transient["bytes"], 6400)
        pending = next(
            item for item in manager.pending_source_releases
            if item.entry is source_entry)
        self.assertTrue(pending.remove_ssd_record)
        self.assertIn("queued", manager.ssd_records)
        self.assertEqual(manager.ssd_used_bytes, 12_800)
        manager.advance(prep.ready_time_ns - 1)
        self.assertIn("queued", manager.ssd_records)
        manager.advance(prep.ready_time_ns)
        self.assertNotIn("queued", manager.ssd_records)
        self.assertEqual(manager.ssd_used_bytes, 0)
        self.assertEqual(scheduler.memory.cpu_used, 0)

    def test_queue_recompute_partial_candidates_are_block_aligned(self):
        manager, _, _, _, _ = self.partial_queue_recompute_manager(
            KVLocation.CPU)

        candidates, snapshots = manager._queue_recompute_partial_candidates(
            reusable_tokens=70,
            target_instance_id=0,
            pd_decode_instance_id=None,
            input_tokens=1024,
            operation_time_ns=0,
        )

        self.assertTrue(candidates)
        self.assertTrue(all(0 < value < 70 for value in candidates))
        self.assertTrue(all(value % 16 == 0 for value in candidates))
        self.assertEqual(candidates, tuple(sorted(set(candidates), reverse=True)))
        self.assertTrue(all(
            snapshots[value].feasible for value in candidates))

    def test_queue_recompute_severe_false_is_full_tiering(self):
        manager, _, entry = self.queue_recompute_manager(
            KVLocation.CPU,
            ratio=4.0,
            cost_multiplier=1.0,
            provider=LinearPrefillProvider(),
        )

        prep = manager.prepare_request(
            "queued", 0, 16, 17, 0,
            residency_at_return=KVLocation.CPU,
        )

        self.assertEqual(prep.source, KVLocation.CPU)
        self.assertEqual(prep.hit_tokens, 16)
        self.assertEqual(prep.recompute_tokens, 0)
        self.assertEqual(prep.restored_bytes, entry.total_bytes)
        evaluation = next(
            item for item in manager.events
            if item.get("event") == "queue_recompute_evaluate")
        self.assertFalse(evaluation["severe_gate_pass"])
        self.assertEqual(evaluation["decision"], "restore")
        self.assertEqual(
            evaluation["selection_reason"],
            "full_restore_below_severe_threshold",
        )
        self.assertEqual(
            manager.metrics.queue_recompute_full_restore_decisions, 1)
        self.assertEqual(
            manager.metrics.queue_recompute_partial_restore_decisions, 0)

    def test_queue_recompute_unavailable_full_projection_fails_closed(self):
        scheduler = FakeScheduler(
            instance_id=0, node_id=0, num_npus=1,
            bytes_per_token=100, npu_mem=6400, cpu_mem=100_000,
            max_num_batched_tokens=32,
            long_prefill_token_threshold=32,
        )
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="tiered_queue_recompute",
                demotion_mode="capacity-only",
                queue_recompute_wait_service_ratio=0,
                queue_recompute_cost_guard_multiplier=1,
            ),
            queue_recompute_latency_providers={
                0: LinearPrefillProvider(100)},
        )
        entry = IdleKVEntry(
            session_id="queued", instance_id=0, tokens=128,
            block_tokens=128, per_rank_bytes=12_800,
            total_bytes=12_800, location=KVLocation.CPU,
            tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[entry.session_id] = entry
        scheduler.memory.allocate(entry.total_bytes, Device.CPU)

        prep = manager.prepare_request(
            "queued", 0, 128, 1024, 0,
            residency_at_return=KVLocation.CPU,
        )

        self.assertEqual(prep.source, KVLocation.DROPPED)
        self.assertEqual(manager.metrics.queue_recompute_drop_decisions, 0)
        self.assertEqual(
            manager.metrics.queue_recompute_partial_restore_decisions, 0)
        evaluation = next(
            item for item in manager.events
            if item.get("event") == "queue_recompute_evaluate")
        self.assertFalse(evaluation["projection_available"])
        self.assertFalse(evaluation["severe_gate_pass"])
        self.assertEqual(evaluation["decision"], "restore")
        self.assertEqual(
            evaluation["selection_reason"],
            "full_projection_unavailable_fail_closed",
        )
        capacity_drop = next(
            item for item in manager.events
            if item.get("event") == "hbm_capacity_restore_drop")
        self.assertEqual(capacity_drop["session_id"], "queued")
        self.assertFalse(any(
            item.get("event") == "queue_recompute_drop"
            for item in manager.events))

    def test_queue_recompute_pd_snapshot_uses_decode_capacity_ceiling(self):
        manager, _, decode, _, _ = self.partial_queue_recompute_manager(
            KVLocation.CPU, pd=True, decode_npu_mem=8000)
        decode.memory.allocate(1600, Device.NPU)

        candidates, snapshots = manager._queue_recompute_partial_candidates(
            reusable_tokens=128,
            target_instance_id=0,
            pd_decode_instance_id=1,
            input_tokens=1024,
            operation_time_ns=0,
        )

        self.assertEqual(candidates[0], 32)
        selected_snapshot = snapshots[32]
        self.assertEqual(
            selected_snapshot.prefill_unreserved_per_rank_bytes, 9600)
        self.assertEqual(
            selected_snapshot.decode_unreserved_per_rank_bytes, 6400)
        self.assertEqual(
            selected_snapshot.decode_required_through_chunk_per_rank_bytes,
            6400,
        )
        self.assertTrue(selected_snapshot.feasible)
        too_large = snapshots[64]
        self.assertEqual(
            too_large.decode_required_through_chunk_per_rank_bytes, 9600)
        self.assertFalse(too_large.feasible)
        with self.assertRaisesRegex(RuntimeError, "explicit fixed decode"):
            manager._queue_recompute_partial_candidates(
                reusable_tokens=128,
                target_instance_id=0,
                pd_decode_instance_id=None,
                input_tokens=1024,
                operation_time_ns=0,
            )

    def test_queue_recompute_partial_projection_is_pure(self):
        manager, scheduler, _, source_entry, victim = (
            self.partial_queue_recompute_manager(KVLocation.CPU))
        candidate = IdleKVEntry(
            session_id="queued", instance_id=0, tokens=128,
            block_tokens=128, per_rank_bytes=12_800,
            total_bytes=12_800, location=KVLocation.HBM,
            tier_since_ns=0, last_access_ns=0,
        )
        projection = manager._project_hbm_then_lower_tier_restore(
            candidate=candidate,
            source=KVLocation.CPU,
            staging_instance_id=0,
            target_instance_id=0,
            per_rank_bytes=12_800,
            total_bytes=12_800,
            operation_time_ns=0,
        )
        state_before = (
            scheduler.memory.npu_used,
            scheduler.memory.cpu_used,
            victim.migration_kind,
            source_entry.migration_kind,
            dict(manager._resource_intervals),
            list(manager.pending_hbm_allocations),
        )

        selection = manager._evaluate_queue_recompute(
            session_id="queued",
            source=KVLocation.CPU,
            projection=projection,
            staging_instance_id=0,
            target_instance_id=0,
            pd_decode_instance_id=None,
            per_rank_bytes=12_800,
            total_bytes=12_800,
            physical_entry_bytes=12_800,
            declared_reuse_tokens=128,
            reusable_tokens=128,
            policy_avoidable_tokens=128,
            input_tokens=1024,
            operation_time_ns=0,
        )

        self.assertTrue(selection.partial)
        self.assertEqual(selection.selected_tokens, 64)
        self.assertEqual(
            state_before,
            (
                scheduler.memory.npu_used,
                scheduler.memory.cpu_used,
                victim.migration_kind,
                source_entry.migration_kind,
                dict(manager._resource_intervals),
                list(manager.pending_hbm_allocations),
            ),
        )

    def test_queue_recompute_partial_retry_commitment_keeps_same_H(self):
        manager, _, _, _, _ = self.partial_queue_recompute_manager(
            KVLocation.CPU)
        candidate = IdleKVEntry(
            session_id="queued", instance_id=0, tokens=128,
            block_tokens=128, per_rank_bytes=12_800,
            total_bytes=12_800, location=KVLocation.HBM,
            tier_since_ns=0, last_access_ns=0,
        )
        projection = manager._project_hbm_then_lower_tier_restore(
            candidate=candidate,
            source=KVLocation.CPU,
            staging_instance_id=0,
            target_instance_id=0,
            per_rank_bytes=12_800,
            total_bytes=12_800,
            operation_time_ns=0,
        )
        kwargs = dict(
            session_id="queued",
            source=KVLocation.CPU,
            staging_instance_id=0,
            target_instance_id=0,
            pd_decode_instance_id=None,
            per_rank_bytes=12_800,
            total_bytes=12_800,
            physical_entry_bytes=12_800,
            declared_reuse_tokens=128,
            reusable_tokens=128,
            policy_avoidable_tokens=128,
            input_tokens=1024,
        )

        first = manager._evaluate_queue_recompute(
            projection=projection, operation_time_ns=0, **kwargs)
        manager._reserve_transfer(
            kind="cpu_to_hbm", arrival_ns=0, service_ns=1_000_000,
            source_instance_id=0, target_instance_id=0,
            num_bytes=1, background=True, session_id="later",
        )
        retry = manager._evaluate_queue_recompute(
            projection=None, operation_time_ns=1, **kwargs)

        self.assertTrue(first.partial)
        self.assertEqual(first.selected_tokens, 64)
        self.assertEqual(retry.selected_tokens, first.selected_tokens)
        self.assertIsNone(retry.selected_projection)
        self.assertEqual(
            manager.metrics.queue_recompute_evaluation_attempts, 1)
        reused = next(
            item for item in manager.events
            if item.get("event")
            == "queue_recompute_restore_commitment_reused")
        self.assertEqual(reused["selected_prefix_tokens_H"], 64)

    def test_queue_recompute_drops_queued_cpu_restore_without_foreground_io(self):
        manager, scheduler, _ = self.queue_recompute_manager(
            KVLocation.CPU)
        self.assertEqual(
            manager.claim_active_hbm_reclaim(
                0, 1600, 0, owner_kind="pd", owner_id=99),
            0,
        )
        manager._reserve_transfer(
            kind="cpu_to_hbm", arrival_ns=0, service_ns=1_000_000,
            source_instance_id=0, target_instance_id=0,
            num_bytes=1, background=True, session_id="occupier",
        )
        jobs_before = manager.metrics.transfer_jobs

        prep = manager.prepare_request(
            "queued", 0, 16, 17, 0,
            residency_at_return=KVLocation.CPU,
        )

        self.assertEqual(prep.source, KVLocation.DROPPED)
        self.assertEqual(prep.recompute_tokens, 16)
        self.assertEqual(manager.metrics.transfer_jobs, jobs_before)
        self.assertEqual(manager.metrics.queue_recompute_drop_decisions, 1)
        self.assertEqual(manager.metrics.queue_recompute_cpu_drop_decisions, 1)
        self.assertEqual(manager.metrics.queue_recompute_ssd_drop_decisions, 0)
        self.assertEqual(manager.metrics.queue_recompute_tokens, 16)
        self.assertEqual(
            manager.metrics.queue_recompute_policy_avoidable_tokens, 16)
        self.assertEqual(scheduler.memory.npu_used, 0)
        self.assertEqual(scheduler.memory.cpu_used, 0)
        self.assertEqual(manager.pending_hbm_allocations, [])
        claim = manager.active_hbm_reclaim_claim(0)
        self.assertEqual(
            (claim.owner_kind, claim.owner_id, claim.per_rank_bytes),
            ("pd", 99, 1600),
        )
        manager.consume_active_hbm_reclaim(
            0, 0, owner_kind="pd", owner_id=99)
        decision = next(
            event for event in manager.events
            if event.get("event") == "queue_recompute_drop")
        self.assertEqual(decision["source"], "cpu")
        self.assertGreater(
            decision["projected_queue_wait_ns"],
            decision["projected_service_ns"],
        )
        drop = next(
            event for event in manager.events
            if (event.get("event") == "drop"
                and event.get("reason") == "queue_pressure"))
        self.assertEqual(drop["drop_class"], "policy_loss")
        self.assertEqual(drop["logical_session_effect"], "none")

    def test_queue_recompute_drops_queued_two_stage_ssd_restore(self):
        manager, scheduler, _ = self.queue_recompute_manager(
            KVLocation.SSD)
        self.assertEqual(
            manager.claim_active_hbm_reclaim(
                0, 1600, 0, owner_kind="pd", owner_id=99),
            0,
        )
        manager._reserve_transfer(
            kind="ssd_to_cpu_stage", arrival_ns=0,
            service_ns=1_000_000, source_instance_id=0,
            target_instance_id=None, num_bytes=1, background=True,
            session_id="occupier",
        )
        jobs_before = manager.metrics.transfer_jobs

        prep = manager.prepare_request(
            "queued", 0, 16, 17, 0,
            residency_at_return=KVLocation.SSD,
        )

        self.assertEqual(prep.source, KVLocation.DROPPED)
        self.assertEqual(manager.metrics.transfer_jobs, jobs_before)
        self.assertEqual(manager.metrics.queue_recompute_ssd_drop_decisions, 1)
        self.assertEqual(manager.ssd_records, {})
        self.assertEqual(manager.ssd_used_bytes, 0)
        self.assertEqual(scheduler.memory.npu_used, 0)
        claim = manager.active_hbm_reclaim_claim(0)
        self.assertEqual(
            (claim.owner_kind, claim.owner_id, claim.per_rank_bytes),
            ("pd", 99, 1600),
        )
        manager.consume_active_hbm_reclaim(
            0, 0, owner_kind="pd", owner_id=99)
        decision = next(
            event for event in manager.events
            if event.get("event") == "queue_recompute_drop")
        self.assertEqual(
            decision["transfer_kinds"],
            ["ssd_to_cpu_stage", "cpu_stage_to_hbm"],
        )

    def test_queue_recompute_strict_threshold_equality_restores(self):
        manager, _, entry = self.queue_recompute_manager(
            KVLocation.CPU)
        _, service_ns, _ = manager._project_lower_tier_restore_queue(
            source=KVLocation.CPU,
            arrival_ns=0,
            staging_instance_id=entry.instance_id,
            target_instance_id=0,
            per_rank_bytes=entry.per_rank_bytes,
            total_bytes=entry.total_bytes,
        )
        manager._reserve_transfer(
            kind="cpu_to_hbm", arrival_ns=0, service_ns=service_ns,
            source_instance_id=0, target_instance_id=0,
            num_bytes=1, background=True, session_id="occupier",
        )

        prep = manager.prepare_request(
            "queued", 0, 16, 17, 0,
            residency_at_return=KVLocation.CPU,
        )

        self.assertEqual(prep.source, KVLocation.CPU)
        self.assertEqual(prep.queue_wait_ns, service_ns)
        self.assertEqual(manager.metrics.queue_recompute_evaluation_attempts, 1)
        self.assertEqual(manager.metrics.queue_recompute_drop_decisions, 0)
        evaluation = next(
            event for event in manager.events
            if event.get("event") == "queue_recompute_evaluate")
        self.assertEqual(evaluation["decision"], "restore")
        self.assertEqual(
            evaluation["projected_queue_wait_ns"],
            evaluation["threshold_ns"],
        )

    def test_queue_recompute_cost_guard_rejects_expensive_recompute(self):
        manager, _, _ = self.queue_recompute_manager(
            KVLocation.CPU,
            ratio=1.0,
            cost_multiplier=1.25,
            provider=LinearPrefillProvider(),
        )
        manager._reserve_transfer(
            kind="cpu_to_hbm", arrival_ns=0, service_ns=101,
            source_instance_id=0, target_instance_id=0,
            num_bytes=1, background=True, session_id="occupier",
        )

        prep = manager.prepare_request(
            "queued", 0, 16, 17, 0,
            residency_at_return=KVLocation.CPU,
        )

        self.assertEqual(prep.source, KVLocation.CPU)
        self.assertEqual(manager.metrics.queue_recompute_severe_gate_passes, 1)
        self.assertEqual(manager.metrics.queue_recompute_cost_gate_passes, 0)
        self.assertEqual(manager.metrics.queue_recompute_drop_decisions, 0)
        evaluation = next(
            event for event in manager.events
            if event.get("event") == "queue_recompute_evaluate")
        self.assertEqual(
            evaluation["estimated_incremental_recompute_comp_ns"], 160)
        self.assertTrue(evaluation["severe_gate_pass"])
        self.assertFalse(evaluation["cost_gate_pass"])
        self.assertEqual(evaluation["decision"], "restore")

    def test_queue_recompute_cost_guard_selects_profitable_drop(self):
        manager, scheduler, _ = self.queue_recompute_manager(
            KVLocation.CPU,
            ratio=4.0,
            cost_multiplier=1.25,
            provider=LinearPrefillProvider(),
        )
        manager.ssd_records["queued"] = SSDRecord(
            tokens=16, block_tokens=16, bytes=1600,
            last_access_ns=0, accounted_until_ns=0,
        )
        manager.ssd_used_bytes = 1600
        manager._reserve_transfer(
            kind="cpu_to_hbm", arrival_ns=0, service_ns=1_000,
            source_instance_id=0, target_instance_id=0,
            num_bytes=1, background=True, session_id="occupier",
        )
        transfer_jobs_before = manager.metrics.transfer_jobs

        prep = manager.prepare_request(
            "queued", 0, 16, 17, 0,
            residency_at_return=KVLocation.CPU,
        )

        self.assertEqual(prep.source, KVLocation.DROPPED)
        self.assertEqual(manager.metrics.transfer_jobs, transfer_jobs_before)
        self.assertEqual(manager.metrics.queue_recompute_drop_decisions, 1)
        self.assertEqual(
            manager.metrics.queue_recompute_avoided_restore_bytes, 1600)
        self.assertEqual(
            manager.metrics.queue_recompute_physical_entry_dropped_bytes,
            3200,
        )
        self.assertEqual(
            manager.metrics.queue_recompute_estimated_recompute_ns, 160)
        self.assertEqual(scheduler.memory.npu_used, 0)
        self.assertEqual(manager.ssd_used_bytes, 0)
        decision = next(
            event for event in manager.events
            if event.get("event") == "queue_recompute_drop")
        self.assertTrue(
            decision["projection_precedes_destination_hbm_reservation"])
        self.assertEqual(
            decision["selection_scope"], "whole_reusable_entry")

    def test_queue_recompute_drop_does_not_mutate_unrelated_hbm_lru(self):
        manager, scheduler, _ = self.queue_recompute_manager(
            KVLocation.CPU, ratio=4.0)
        scheduler.memory.npu_mem = 1600
        scheduler.memory.cpu_mem = 3200
        hbm_victim = IdleKVEntry(
            session_id="hbm-victim", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        cpu_victim = IdleKVEntry(
            session_id="cpu-victim", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[hbm_victim.session_id] = hbm_victim
        manager.entries[cpu_victim.session_id] = cpu_victim
        scheduler.memory.allocate(1600, Device.NPU)
        scheduler.memory.allocate(1600, Device.CPU)
        manager._reserve_transfer(
            kind="cpu_to_hbm", arrival_ns=0, service_ns=1_000,
            source_instance_id=0, target_instance_id=0,
            num_bytes=1, background=True, session_id="occupier",
        )
        jobs_before = manager.metrics.transfer_jobs
        intervals_before = {
            key: list(value)
            for key, value in manager._resource_intervals.items()
        }

        prep = manager.prepare_request(
            "queued", 0, 16, 17, 0,
            residency_at_return=KVLocation.CPU,
        )

        self.assertEqual(prep.source, KVLocation.DROPPED)
        self.assertEqual(hbm_victim.location, KVLocation.HBM)
        self.assertIsNone(hbm_victim.migration_kind)
        self.assertEqual(cpu_victim.location, KVLocation.CPU)
        self.assertIsNone(cpu_victim.migration_kind)
        self.assertEqual(manager.metrics.transfer_jobs, jobs_before)
        self.assertEqual(manager._resource_intervals, intervals_before)
        self.assertEqual(manager.pending_hbm_allocations, [])
        self.assertEqual(scheduler.memory.npu_used, 1600)
        self.assertEqual(scheduler.memory.cpu_used, 1600)
        decision = next(
            event for event in manager.events
            if event.get("event") == "queue_recompute_drop")
        self.assertEqual(
            decision["projected_hbm_victim_sessions"], ["hbm-victim"])
        self.assertEqual(
            decision["projected_cpu_victim_sessions"], ["cpu-victim"])
        self.assertGreater(decision["projected_hbm_admission_wait_ns"], 0)
        self.assertGreater(
            decision["projected_total_wait_ns"],
            decision["projected_service_ns"],
        )
        self.assertTrue(decision["projection_available"])
        self.assertTrue(
            decision["projection_includes_collateral_lru_work"])
        self.assertFalse(
            decision["projection_available_without_new_lru_work"])

    def test_hbm_lru_projection_is_pure_and_matches_restore_apply(self):
        manager, scheduler, source_entry = self.queue_recompute_manager(
            KVLocation.CPU, ratio=1e9)
        scheduler.memory.npu_mem = 1600
        scheduler.memory.cpu_mem = 3200
        hbm_victim = IdleKVEntry(
            session_id="hbm-victim", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        cpu_victim = IdleKVEntry(
            session_id="cpu-victim", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[hbm_victim.session_id] = hbm_victim
        manager.entries[cpu_victim.session_id] = cpu_victim
        scheduler.memory.allocate(1600, Device.NPU)
        scheduler.memory.allocate(1600, Device.CPU)
        manager._reserve_transfer(
            kind="cpu_to_ssd", arrival_ns=0, service_ns=100,
            source_instance_id=0, target_instance_id=None,
            num_bytes=1, background=True, session_id="occupier",
        )
        candidate = IdleKVEntry(
            session_id="queued", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        state_before = manager._hbm_reservation_fingerprint()
        metrics_before = vars(manager.metrics).copy()
        events_before = list(manager.events)
        generation_before = manager._hbm_admission_state_generation

        projection = manager._project_hbm_then_lower_tier_restore(
            candidate=candidate,
            source=KVLocation.CPU,
            staging_instance_id=0,
            target_instance_id=0,
            per_rank_bytes=1600,
            total_bytes=1600,
            operation_time_ns=0,
        )

        self.assertEqual(manager._hbm_reservation_fingerprint(), state_before)
        self.assertEqual(vars(manager.metrics), metrics_before)
        self.assertEqual(manager.events, events_before)
        self.assertEqual(
            manager._hbm_admission_state_generation, generation_before)
        self.assertEqual(projection.hbm_victim_sessions, ("hbm-victim",))
        self.assertEqual(projection.cpu_victim_sessions, ("cpu-victim",))
        self.assertGreater(projection.hbm_admission_wait_ns, 0)

        apply_event_start = len(manager.events)
        manager._set_restore_capacity_pin("queued", True)
        ready_ns = manager._reserve_hbm(candidate, 0)
        foreground = manager._reserve_transfer(
            kind="cpu_to_hbm",
            arrival_ns=ready_ns,
            service_ns=manager._cpu_transfer_ns(1600, 1600),
            source_instance_id=0,
            target_instance_id=0,
            num_bytes=1600,
            background=False,
            session_id="queued",
            job_arrival_ns=0,
        )
        manager._assert_hbm_restore_projection_applied(
            projection=projection,
            candidate_ready_ns=ready_ns,
            source=KVLocation.CPU,
            staging_instance_id=0,
            target_instance_id=0,
            per_rank_bytes=1600,
            total_bytes=1600,
            reservations=(foreground,),
            event_start_index=apply_event_start,
        )

        self.assertEqual(ready_ns, projection.hbm_ready_ns)
        self.assertEqual(cpu_victim.migration_kind, "cpu_to_ssd")
        self.assertEqual(hbm_victim.migration_kind, "hbm_to_cpu")
        self.assertIsNone(source_entry.migration_kind)
        source_background_kinds = [
            event["kind"]
            for event in manager.events
            if (event.get("event") == "migration_reserve"
                and event.get("session_id") == "queued"
                and not event.get("foreground"))
        ]
        self.assertEqual(source_background_kinds, [])

    def test_foreground_cpu_source_is_pinned_across_hbm_lru_cascade(self):
        manager, scheduler, source_entry = self.queue_recompute_manager(
            KVLocation.CPU, ratio=1e9)
        scheduler.memory.npu_mem = 1600
        scheduler.memory.cpu_mem = 3200
        hbm_victim = IdleKVEntry(
            session_id="hbm-victim", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        cpu_victim = IdleKVEntry(
            session_id="cpu-victim", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[hbm_victim.session_id] = hbm_victim
        manager.entries[cpu_victim.session_id] = cpu_victim
        scheduler.memory.allocate(1600, Device.NPU)
        scheduler.memory.allocate(1600, Device.CPU)

        prep = manager.prepare_request(
            "queued", 0, 16, 17, 0,
            residency_at_return=KVLocation.CPU,
        )

        self.assertEqual(prep.source, KVLocation.CPU)
        self.assertEqual(cpu_victim.migration_kind, "cpu_to_ssd")
        self.assertEqual(hbm_victim.migration_kind, "hbm_to_cpu")
        self.assertIsNone(source_entry.migration_kind)
        self.assertNotIn("queued", manager._pending_restore_sessions)
        source_reservations = [
            event
            for event in manager.events
            if (event.get("event") == "migration_reserve"
                and event.get("session_id") == "queued")
        ]
        self.assertEqual(
            [event["kind"] for event in source_reservations],
            ["cpu_to_hbm"],
        )
        self.assertTrue(source_reservations[0]["foreground"])

    def test_ssd_projection_includes_transient_cpu_lru_and_is_pure(self):
        for swap_mode in ("async-pre-admission", "sync-engine-barrier"):
            with self.subTest(swap_mode=swap_mode):
                manager, scheduler, _ = self.queue_recompute_manager(
                    KVLocation.SSD, ratio=1e9)
                manager.config.swap_execution_mode = swap_mode
                scheduler.memory.cpu_mem = 1600
                cpu_victim = IdleKVEntry(
                    session_id="cpu-victim", instance_id=0, tokens=16,
                    block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
                    location=KVLocation.CPU, tier_since_ns=0,
                    last_access_ns=-1,
                )
                manager.entries[cpu_victim.session_id] = cpu_victim
                scheduler.memory.allocate(1600, Device.CPU)
                candidate = IdleKVEntry(
                    session_id="queued", instance_id=0, tokens=16,
                    block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
                    location=KVLocation.HBM, tier_since_ns=0,
                    last_access_ns=0,
                )
                state_before = manager._hbm_reservation_fingerprint()
                metrics_before = vars(manager.metrics).copy()
                events_before = list(manager.events)
                history_before = {
                    node_id: list(reservations)
                    for node_id, reservations
                    in manager._transient_dram_history.items()
                }

                projection = manager._project_hbm_then_lower_tier_restore(
                    candidate=candidate,
                    source=KVLocation.SSD,
                    staging_instance_id=0,
                    target_instance_id=0,
                    per_rank_bytes=1600,
                    total_bytes=1600,
                    operation_time_ns=0,
                )

                self.assertTrue(projection.available)
                self.assertEqual(
                    projection.cpu_victim_sessions, ("cpu-victim",))
                self.assertEqual(projection.hbm_admission_wait_ns, 0)
                self.assertGreater(projection.queue_wait_ns, 0)
                self.assertEqual(
                    projection.total_wait_ns,
                    projection.hbm_admission_wait_ns
                    + projection.queue_wait_ns,
                )
                self.assertEqual(
                    projection.restore_ready_ns,
                    projection.total_wait_ns + projection.service_ns,
                )
                self.assertEqual(
                    manager._hbm_reservation_fingerprint(), state_before)
                self.assertEqual(vars(manager.metrics), metrics_before)
                self.assertEqual(manager.events, events_before)
                self.assertEqual(
                    manager._transient_dram_history, history_before)
                self.assertIsNone(cpu_victim.migration_kind)

                prep = manager.prepare_request(
                    "queued", 0, 16, 17, 0,
                    residency_at_return=KVLocation.SSD,
                )

                self.assertEqual(prep.source, KVLocation.SSD)
                self.assertEqual(
                    prep.queue_wait_ns, projection.queue_wait_ns)
                self.assertEqual(
                    cpu_victim.migration_kind, "cpu_to_ssd")
                evaluation = next(
                    event for event in manager.events
                    if event.get("event") == "queue_recompute_evaluate")
                self.assertEqual(
                    evaluation["projected_cpu_victim_sessions"],
                    ["cpu-victim"],
                )
                self.assertEqual(
                    evaluation["projected_queue_wait_ns"],
                    projection.queue_wait_ns,
                )
                if swap_mode == "sync-engine-barrier":
                    self.assertEqual(
                        [barrier[3] for barrier in
                         manager._sync_engine_barriers[0]],
                        ["ssd_staged_to_hbm"],
                    )

    def test_ssd_projection_labels_transient_capacity_wait_as_queue(self):
        manager, scheduler, _ = self.queue_recompute_manager(
            KVLocation.SSD, ratio=1e9)
        scheduler.memory.cpu_mem = 1600
        held = IdleKVEntry(
            session_id="held", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
        )
        scheduler.memory.allocate(1600, Device.CPU)
        manager.pending_source_releases.append(PendingSourceRelease(
            entry=held, ready_ns=5000))
        candidate = IdleKVEntry(
            session_id="queued", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )

        projection = manager._project_hbm_then_lower_tier_restore(
            candidate=candidate,
            source=KVLocation.SSD,
            staging_instance_id=0,
            target_instance_id=0,
            per_rank_bytes=1600,
            total_bytes=1600,
            operation_time_ns=0,
        )

        self.assertEqual(projection.hbm_ready_ns, 0)
        self.assertEqual(projection.foreground_arrival_ns, 5000)
        self.assertEqual(projection.hbm_admission_wait_ns, 0)
        self.assertEqual(projection.transient_dram_capacity_wait_ns, 5000)
        self.assertEqual(projection.queue_wait_ns, 5000)
        self.assertEqual(projection.total_wait_ns, 5000)
        prep = manager.prepare_request(
            "queued", 0, 16, 17, 0,
            residency_at_return=KVLocation.SSD,
        )
        self.assertEqual(prep.hbm_admission_wait_ns, 0)
        self.assertEqual(prep.transient_dram_capacity_wait_ns, 5000)
        self.assertEqual(prep.queue_wait_ns, 5000)
        self.assertEqual(prep.target_hbm_ready_time_ns, 0)
        resume = next(
            event for event in manager.events
            if event.get("event") == "resume"
            and event.get("session_id") == "queued")
        self.assertEqual(
            resume["transient_dram_capacity_wait_ns"], 5000)
        self.assertEqual(
            resume["restore_ns"],
            resume["hbm_admission_wait_ns"]
            + resume["queue_wait_ns"]
            + resume["restore_service_ns"],
        )
        evaluation = next(
            event for event in manager.events
            if event.get("event") == "queue_recompute_evaluate")
        self.assertEqual(evaluation["projection_arrival_ns"], 0)
        self.assertEqual(
            evaluation["projected_hbm_admission_wait_ns"], 0)
        self.assertEqual(
            evaluation["projected_transient_dram_capacity_wait_ns"], 5000)
        self.assertEqual(evaluation["projected_queue_wait_ns"], 5000)
        self.assertEqual(evaluation["projected_total_wait_ns"], 5000)
        breakdown = manager.summary(5002)["time_breakdown"]
        self.assertEqual(
            breakdown["aggregate_request_migration_hbm_admission_wait_ns"],
            0,
        )
        self.assertEqual(
            breakdown["aggregate_request_migration_queue_wait_ns"], 5000)
        self.assertEqual(
            breakdown["aggregate_request_migration_transfer_queue_wait_ns"],
            0,
        )
        self.assertTrue(
            breakdown["transient_dram_capacity_wait_is_subset_of_queue_wait"])

    def test_queue_recompute_drop_reports_transient_capacity_queue_subset(self):
        manager, scheduler, _ = self.queue_recompute_manager(
            KVLocation.SSD, ratio=1.0)
        scheduler.memory.cpu_mem = 1600
        held = IdleKVEntry(
            session_id="held", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
        )
        scheduler.memory.allocate(1600, Device.CPU)
        manager.pending_source_releases.append(PendingSourceRelease(
            entry=held, ready_ns=5000))

        prep = manager.prepare_request(
            "queued", 0, 16, 17, 0,
            residency_at_return=KVLocation.SSD,
        )

        self.assertEqual(prep.source, KVLocation.DROPPED)
        resume = next(
            event for event in manager.events
            if event.get("event") == "resume"
            and event.get("session_id") == "queued")
        self.assertEqual(resume["transient_dram_capacity_wait_ns"], 0)
        self.assertEqual(
            manager.metrics
            .queue_recompute_projected_transient_dram_capacity_wait_ns,
            5000,
        )
        policy = manager.summary(5000)["queue_recompute_policy"]
        self.assertEqual(
            policy["selected_projected_transient_dram_capacity_wait_ns"],
            5000,
        )
        self.assertEqual(policy["selected_projected_queue_wait_ns"], 5000)
        self.assertEqual(policy["selected_projected_hbm_admission_wait_ns"], 0)
        self.assertEqual(policy["selected_projected_total_wait_ns"], 5000)

    def test_drop_events_are_explicitly_kv_scoped_and_classified(self):
        manager, _ = self.manager()
        expected = {
            "hbm_capacity": "capacity_loss",
            "cpu_capacity": "capacity_loss",
            "ssd_capacity": "capacity_loss",
            "ssd_ttl": "ttl_loss",
            "resume_miss": "resume_recompute_cleanup",
            "queue_pressure": "policy_loss",
            "session_end": "normal_session_cleanup",
            "measurement_censor": "measurement_cleanup",
        }
        for now_ns, (reason, drop_class) in enumerate(expected.items()):
            entry = IdleKVEntry(
                session_id=reason,
                instance_id=0,
                tokens=0,
                block_tokens=0,
                per_rank_bytes=0,
                total_bytes=0,
                location=KVLocation.DROPPED,
                tier_since_ns=0,
                last_access_ns=0,
            )
            manager._drop_entry(entry, now_ns, reason)
            event = manager.events[-1]
            self.assertEqual(event["event"], "drop")
            self.assertEqual(event["reason"], reason)
            self.assertEqual(event["drop_class"], drop_class)
            self.assertEqual(event["object_scope"], "kv_cache_entry")
            self.assertEqual(event["logical_session_effect"], "none")

        with self.assertRaisesRegex(
                RuntimeError, "Unknown KV-cache entry drop reason"):
            manager._drop_entry(entry, 99, "ambiguous")

        summary = manager.summary(100)
        self.assertEqual(summary["schema_version"], 20)
        semantics = summary["event_semantics"]["drop"]
        self.assertEqual(semantics["object_scope"], "kv_cache_entry")
        self.assertEqual(semantics["logical_session_effect"], "none")
        self.assertEqual(semantics["classification_field"], "drop_class")
        self.assertEqual(set(semantics["classes"]), set(expected.values()))

    def assert_restore_components(self, prep):
        self.assertEqual(
            prep.restore_ns,
            prep.hbm_admission_wait_ns + prep.queue_wait_ns + prep.service_ns,
        )

    def test_nonphysical_resume_delay_is_prepare_boundary_not_restore(self):
        manager, scheduler = self.manager(policy="preserve")
        manager.on_tool_start(
            FakeRequest(tokens=16), 0, 1_000_000_000)

        prep = manager.prepare_request(
            "s", 0, 16, 17, 0,
            operation_time_ns=100,
            pd_pair_fifo_wait_ns=40,
        )

        self.assertEqual(prep.source, KVLocation.HBM)
        self.assertEqual(prep.pd_pair_fifo_wait_ns, 40)
        self.assertEqual(prep.prepare_boundary_wait_ns, 60)
        self.assertEqual(prep.restore_ns, 0)
        self.assertEqual(prep.hbm_admission_wait_ns, 0)
        self.assertEqual(prep.owner_gate_ns, 100)
        self.assertEqual(prep.restore_issue_time_ns, 100)
        self.assertEqual(prep.target_hbm_ready_time_ns, 100)
        self.assertEqual(prep.restore_ready_time_ns, 100)
        self.assertEqual(manager.metrics.critical_restore_ns, 0)
        self.assertEqual(manager.metrics.pd_pair_fifo_wait_ns, 40)
        self.assertEqual(manager.metrics.prepare_boundary_wait_ns, 60)
        scheduler.memory.free(prep.restored_bytes // 8, Device.NPU)

        dropped, _ = self.manager(policy="recompute")
        miss = dropped.prepare_request(
            "missing", 0, 16, 17, 0,
            operation_time_ns=100,
            pd_pair_fifo_wait_ns=40,
        )
        self.assertEqual(miss.source, KVLocation.DROPPED)
        self.assertEqual(miss.prepare_boundary_wait_ns, 60)
        self.assertEqual(miss.restore_ns, 0)
        self.assertEqual(miss.hbm_admission_wait_ns, 0)
        self.assertEqual(miss.owner_gate_ns, 100)
        self.assertEqual(miss.restore_issue_time_ns, 100)

    def test_completed_kv_handoff_precedes_future_hbm_reservation(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=3200)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(
                policy="tiered", demotion_mode="capacity-only"))
        waiting = IdleKVEntry(
            session_id="waiting", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        manager.pending_hbm_allocations.append(PendingHBMAllocation(
            entry=waiting, ready_ns=100))

        manager.on_idle_start(
            FakeRequest(session_id="completed", tokens=16),
            completion_time_ns=10,
            release_time_ns=1_000,
        )

        completed = manager.entries["completed"]
        self.assertEqual(completed.location, KVLocation.HBM)
        self.assertEqual(scheduler.memory.npu_used, 1600)
        self.assertEqual(manager.metrics.hbm_capacity_drops, 0)
        self.assertEqual(len(manager.pending_hbm_allocations), 1)

        manager.advance(100)
        self.assertEqual(scheduler.memory.npu_used, 3200)
        self.assertEqual(manager.pending_hbm_allocations, [])

    def test_completed_kv_handoff_fails_loudly_after_tied_steal(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=1600)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(
                policy="tiered", demotion_mode="capacity-only"))
        scheduler.memory.allocate(1600, Device.NPU)

        with self.assertRaisesRegex(
                RuntimeError, "Completion ownership must precede"):
            manager.on_idle_start(
                FakeRequest(session_id="completed", tokens=16),
                completion_time_ns=10,
                release_time_ns=1_000,
            )

        self.assertNotIn("completed", manager.entries)
        self.assertEqual(manager.metrics.hbm_capacity_drops, 0)

    def test_capacity_retry_begins_after_pair_and_frozen_boundary_wait(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100,
            npu_mem=1600, cpu_mem=1600)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(
                policy="tiered", demotion_mode="capacity-only",
                pcie_bandwidth_gbps=1e9,
                cpu_bandwidth_gbps=1e9,
                cpu_transfer_latency_us=0,
            ))
        source = IdleKVEntry(
            session_id="waiting", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[source.session_id] = source
        scheduler.memory.allocate(1600, Device.CPU)
        scheduler.memory.allocate(1600, Device.NPU)

        self.assertIsNone(manager.prepare_request(
            "waiting", 0, 16, 17, 0,
            operation_time_ns=60,
            pd_pair_fifo_wait_ns=40,
            prepare_boundary_wait_ns=20,
            defer_temporary_hbm_pressure=True,
        ))
        scheduler.memory.free(1600, Device.NPU)
        prep = manager.prepare_request(
            "waiting", 0, 16, 17, 0,
            operation_time_ns=100,
            pd_pair_fifo_wait_ns=40,
            prepare_boundary_wait_ns=20,
            defer_temporary_hbm_pressure=True,
        )

        self.assertEqual(prep.restore_issue_time_ns, 60)
        self.assertEqual(prep.hbm_admission_wait_ns, 40)
        self.assertEqual(prep.target_hbm_ready_time_ns, 100)
        self.assertEqual(
            prep.owner_gate_ns,
            40 + 20 + prep.restore_ns,
        )
        self.assertEqual(
            prep.restore_ready_time_ns,
            prep.restore_issue_time_ns + prep.restore_ns,
        )
        self.assert_restore_components(prep)

    def test_default_swap_execution_waits_before_compute_admission(self):
        self.assertEqual(
            AgenticKVConfig().swap_execution_mode,
            "async-pre-admission",
        )

    def test_pd_peer_transfer_defaults_to_cpu_staged(self):
        self.assertEqual(
            AgenticKVConfig().pd_peer_transfer_mode,
            "cpu-staged",
        )

    def test_rejects_unknown_demotion_mode(self):
        config = AgenticKVConfig(demotion_mode="future-aware")
        with self.assertRaisesRegex(ValueError, "demotion_mode"):
            config.validate()

    def test_rejects_unknown_swap_execution_mode(self):
        config = AgenticKVConfig(swap_execution_mode="sometimes-sync")
        with self.assertRaisesRegex(ValueError, "swap_execution_mode"):
            config.validate()

    def test_rejects_invalid_pd_peer_transfer_settings(self):
        with self.assertRaisesRegex(ValueError, "pd_peer_transfer_mode"):
            AgenticKVConfig(pd_peer_transfer_mode="magic").validate()
        with self.assertRaisesRegex(ValueError, "pd_peer_bandwidth_gbps"):
            AgenticKVConfig(pd_peer_bandwidth_gbps=0).validate()
        with self.assertRaisesRegex(ValueError, "pd_peer_latency_us"):
            AgenticKVConfig(pd_peer_latency_us=-1).validate()

    def test_pd_peer_modes_use_distinct_latency_and_resources(self):
        source = FakeScheduler(instance_id=0, node_id=0, num_npus=4)
        target = FakeScheduler(instance_id=1, node_id=0, num_npus=4)
        cpu_staged = AgenticKVManager(
            [source, target],
            AgenticKVConfig(
                policy="preserve",
                pcie_bandwidth_gbps=50,
                cpu_bandwidth_gbps=400,
                cpu_transfer_latency_us=5,
            ),
        )
        direct = AgenticKVManager(
            [source, target],
            AgenticKVConfig(
                policy="preserve",
                pd_peer_transfer_mode="direct-fabric",
                pd_peer_bandwidth_gbps=450,
                pd_peer_latency_us=3,
            ),
        )
        source_per_rank = 90_000_000_000
        target_per_rank = 100_000_000_000
        total_bytes = 400_000_000_000

        self.assertEqual(
            cpu_staged._hbm_peer_transfer_ns(
                source, target, source_per_rank, target_per_rank,
                total_bytes),
            2_000_010_000,
        )
        self.assertEqual(
            direct._hbm_peer_transfer_ns(
                source, target, source_per_rank, target_per_rank,
                total_bytes),
            222_225_223,
        )

        staged_resources = cpu_staged._transfer_resources(
            "hbm_peer", source.instance_id, target.instance_id)
        self.assertEqual(
            sum("pcie-copy" in resource for resource in staged_resources),
            8,
        )
        self.assertIn("node:0:dram", staged_resources)
        self.assertFalse(any(
            "peer-copy" in resource for resource in staged_resources))
        self.assertNotIn("node:0:pd-fabric", staged_resources)

        direct_resources = direct._transfer_resources(
            "hbm_peer", source.instance_id, target.instance_id)
        self.assertEqual(
            sum("peer-copy" in resource for resource in direct_resources),
            8,
        )
        self.assertIn("node:0:pd-fabric", direct_resources)
        self.assertFalse(any(
            "pcie-copy" in resource for resource in direct_resources))
        self.assertFalse(any(
            resource.endswith(":dram") for resource in direct_resources))

    def test_direct_pd_peer_jobs_contend_on_node_fabric(self):
        source = FakeScheduler(instance_id=0, node_id=0, num_npus=4)
        target = FakeScheduler(instance_id=1, node_id=0, num_npus=4)
        manager = AgenticKVManager(
            [source, target],
            AgenticKVConfig(
                policy="preserve",
                pd_peer_transfer_mode="direct-fabric",
            ),
        )
        first = manager._reserve_transfer(
            kind="hbm_peer", arrival_ns=0, service_ns=100,
            source_instance_id=0, target_instance_id=1,
            num_bytes=1, background=False, session_id="first",
        )
        second = manager._reserve_transfer(
            kind="hbm_peer", arrival_ns=0, service_ns=100,
            source_instance_id=0, target_instance_id=1,
            num_bytes=1, background=False, session_id="second",
        )

        self.assertEqual(first.start_ns, 0)
        self.assertEqual(first.complete_ns, 100)
        self.assertEqual(second.start_ns, 100)
        self.assertEqual(second.queue_wait_ns, 100)

    def test_external_pd_restore_waits_for_exact_astra_callback(self):
        prefill = FakeScheduler(instance_id=0, node_id=0, num_npus=4)
        decode = FakeScheduler(instance_id=1, node_id=0, num_npus=4)
        manager = AgenticKVManager(
            [prefill, decode],
            AgenticKVConfig(
                policy="preserve",
                pd_peer_transfer_mode="direct-fabric",
                pd_peer_bandwidth_gbps=450,
                pd_peer_latency_us=3,
            ),
        )
        manager.enable_external_fabric(
            backend="analytical-congestion-aware",
            physical_bandwidth_gbps=450,
            physical_latency_ns=3_000,
        )
        manager.on_tool_start(
            FakeRequest(instance_id=1, tokens=100), 0, 1_000_000_000)
        source_before = decode.memory.npu_used

        self.assertIsNone(
            manager.prepare_request("s", 0, 96, 120, 10_000))
        jobs = manager.drain_external_fabric_jobs()

        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job["source_instance_id"], 1)
        self.assertEqual(job["target_instance_id"], 0)
        self.assertEqual(job["lane_count"], 4)
        self.assertEqual(
            job["bytes_per_lane"], prefill.memory.get_kv(96))
        self.assertEqual(decode.memory.npu_used, source_before)
        self.assertEqual(prefill.memory.npu_used, job["bytes_per_lane"])
        self.assertIn("s", manager._capacity_pinned_sessions())
        self.assertIsNone(
            manager.prepare_request(
                "s", 0, 96, 120, 10_000,
                operation_time_ns=job["arrival_ns"]))
        self.assertEqual(manager.drain_external_fabric_jobs(), [])

        start_ns = job["arrival_ns"] + 25
        complete_ns = start_ns + 100
        state_before = manager.restore_capacity_state(0)
        self.assertTrue(manager.complete_external_fabric_job(
            job_id=job["job_id"],
            arrival_ns=job["arrival_ns"],
            completion_ns=complete_ns,
            bytes_per_lane=job["bytes_per_lane"],
            lane_count=job["lane_count"],
            critical_lane_start_ns=start_ns,
        ))
        self.assertNotEqual(
            state_before, manager.restore_capacity_state(0))
        self.assertEqual(decode.memory.npu_used, source_before)
        self.assertIn("s", manager._capacity_pinned_sessions())

        prep = manager.prepare_request(
            "s", 0, 96, 120, 10_000,
            operation_time_ns=complete_ns)

        self.assertEqual(prep.ready_time_ns, complete_ns)
        self.assertEqual(prep.queue_wait_ns, 25)
        self.assertEqual(prep.service_ns, 100)
        self.assertEqual(
            prep.restore_ns,
            prep.hbm_admission_wait_ns + 25 + 100)
        self.assertEqual(prep.retained_instance_id, 1)
        self.assertEqual(
            decode.memory.npu_used, prep.retained_per_rank_bytes)
        self.assertFalse(manager.has_pending_external_fabric_jobs())
        self.assertFalse(manager.complete_external_fabric_job(
            job_id=job["job_id"],
            arrival_ns=job["arrival_ns"],
            completion_ns=complete_ns,
            bytes_per_lane=job["bytes_per_lane"],
            lane_count=job["lane_count"],
            critical_lane_start_ns=start_ns,
        ))

    def test_external_fabric_rejects_cluster_authority_mismatch(self):
        manager = AgenticKVManager(
            [FakeScheduler()],
            AgenticKVConfig(
                policy="preserve",
                pd_peer_transfer_mode="direct-fabric",
                pd_peer_bandwidth_gbps=450,
                pd_peer_latency_us=3,
            ),
        )

        with self.assertRaisesRegex(ValueError, "link_bw"):
            manager.enable_external_fabric(
                backend="analytical-congestion-aware",
                physical_bandwidth_gbps=400,
                physical_latency_ns=3_000,
            )
        with self.assertRaisesRegex(ValueError, "link_latency"):
            manager.enable_external_fabric(
                backend="analytical-congestion-aware",
                physical_bandwidth_gbps=450,
                physical_latency_ns=1_000,
            )
        with self.assertRaisesRegex(ValueError, "decimal_GBps"):
            manager.enable_external_fabric(
                backend="analytical-congestion-aware",
                physical_bandwidth_gbps=450,
                physical_latency_ns=3_000,
                physical_bandwidth_unit="astra_GBps_legacy",
            )

    def test_censored_external_restore_is_not_request_critical(self):
        prefill = FakeScheduler(instance_id=0, node_id=0, num_npus=4)
        decode = FakeScheduler(instance_id=1, node_id=0, num_npus=4)
        manager = AgenticKVManager(
            [prefill, decode],
            AgenticKVConfig(
                policy="preserve",
                pd_peer_transfer_mode="direct-fabric",
                pd_peer_bandwidth_gbps=450,
                pd_peer_latency_us=1,
            ),
        )
        manager.enable_external_fabric(
            backend="analytical-congestion-aware",
            physical_bandwidth_gbps=450,
            physical_latency_ns=1_000,
        )
        manager.on_tool_start(
            FakeRequest(instance_id=1, tokens=100), 0, 1_000_000_000)
        self.assertIsNone(
            manager.prepare_request("s", 0, 96, 120, 10_000))
        job = manager.drain_external_fabric_jobs()[0]
        start_ns = job["arrival_ns"] + 25
        complete_ns = start_ns + 100
        manager.complete_external_fabric_job(
            job_id=job["job_id"], arrival_ns=job["arrival_ns"],
            completion_ns=complete_ns,
            bytes_per_lane=job["bytes_per_lane"],
            lane_count=job["lane_count"],
            critical_lane_start_ns=start_ns,
        )

        manager.censor_completed_external_fabric_job(
            job["job_id"], complete_ns)

        self.assertEqual(prefill.memory.npu_used, 0)
        self.assertGreater(decode.memory.npu_used, 0)
        self.assertEqual(manager.metrics.critical_restore_ns, 0)
        self.assertEqual(manager.metrics.critical_restore_queue_wait_ns, 0)
        self.assertEqual(manager.metrics.critical_restore_service_ns, 0)
        self.assertEqual(manager.metrics.external_fabric_jobs_censored, 1)
        self.assertEqual(
            manager.metrics.external_fabric_censored_lane_bytes,
            job["bytes_per_lane"] * job["lane_count"],
        )
        self.assertFalse(manager.has_pending_external_fabric_jobs())
        manager.end_session("s", now_ns=complete_ns)
        self.assertEqual(decode.memory.npu_used, 0)

    def test_external_censor_preserves_pre_enqueue_destination_admission(self):
        prefill = FakeScheduler(
            instance_id=0, node_id=0, num_npus=1,
            bytes_per_token=100, npu_mem=1600)
        decode = FakeScheduler(
            instance_id=1, node_id=0, num_npus=1,
            bytes_per_token=100, npu_mem=1600)
        manager = AgenticKVManager(
            [prefill, decode], AgenticKVConfig(
                policy="preserve",
                pd_peer_transfer_mode="direct-fabric",
                pd_peer_bandwidth_gbps=450,
                pd_peer_latency_us=1,
            ))
        manager.enable_external_fabric(
            backend="analytical-congestion-aware",
            physical_bandwidth_gbps=450,
            physical_latency_ns=1_000,
        )
        source = IdleKVEntry(
            session_id="waiting", instance_id=1, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[source.session_id] = source
        decode.memory.allocate(1600, Device.NPU)
        prefill.memory.allocate(1600, Device.NPU)

        self.assertIsNone(manager.prepare_request(
            "waiting", 0, 16, 17, 0,
            operation_time_ns=0,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
            residency_at_return=KVLocation.HBM,
        ))
        prefill.memory.free(1600, Device.NPU)
        self.assertIsNone(manager.prepare_request(
            "waiting", 0, 16, 17, 0,
            operation_time_ns=100,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
            residency_at_return=KVLocation.HBM,
        ))

        job = manager.drain_external_fabric_jobs()[0]
        self.assertEqual(job["arrival_ns"], 100)
        start_ns = job["arrival_ns"] + 25
        complete_ns = start_ns + 100
        manager.complete_external_fabric_job(
            job_id=job["job_id"],
            arrival_ns=job["arrival_ns"],
            completion_ns=complete_ns,
            bytes_per_lane=job["bytes_per_lane"],
            lane_count=job["lane_count"],
            critical_lane_start_ns=start_ns,
        )
        manager.censor_completed_external_fabric_job(
            job["job_id"], complete_ns)
        manager.censor_session("waiting", cutoff_ns=complete_ns)

        self.assertEqual(manager.metrics.critical_restore_ns, 0)
        breakdown = manager.summary(
            complete_ns,
            measurement_censored=True,
        )["time_breakdown"]
        self.assertEqual(
            breakdown["migration_restore_exposure_union_ns"], 100)
        self.assertTrue(
            manager.validate_measurement_censoring_drained()["passed"])

    def test_external_fabric_history_reports_allowed_model_overlap(self):
        prefill = FakeScheduler(instance_id=0, node_id=0, num_npus=4)
        decode = FakeScheduler(instance_id=1, node_id=0, num_npus=4)
        prefill.pd_type = "prefill"
        decode.pd_type = "decode"
        manager = AgenticKVManager(
            [prefill, decode],
            AgenticKVConfig(
                policy="preserve",
                pd_peer_transfer_mode="direct-fabric",
                pd_peer_bandwidth_gbps=450,
                pd_peer_latency_us=1,
            ),
        )
        manager.enable_external_fabric(
            backend="analytical-congestion-aware",
            physical_bandwidth_gbps=450,
            physical_latency_ns=1_000,
        )
        manager.on_tool_start(
            FakeRequest(instance_id=1, tokens=100), 0, 1_000_000_000)
        self.assertIsNone(
            manager.prepare_request("s", 0, 96, 120, 10_000))
        job = manager.drain_external_fabric_jobs()[0]

        model = self._fabric_batch(71, job["arrival_ns"])
        manager.record_agentic_batch_schedule(prefill, model)
        manager.record_astra_workload_dispatch(
            prefill, model, job["arrival_ns"])
        copy_start_ns = job["arrival_ns"] + 25
        copy_complete_ns = copy_start_ns + 100
        self.assertTrue(manager.complete_external_fabric_job(
            job_id=job["job_id"],
            arrival_ns=job["arrival_ns"],
            completion_ns=copy_complete_ns,
            bytes_per_lane=job["bytes_per_lane"],
            lane_count=job["lane_count"],
            critical_lane_start_ns=copy_start_ns,
        ))
        manager.record_agentic_batch_complete(
            prefill, model, copy_complete_ns + 100)

        audit = manager.validate_resource_timeline()

        self.assertEqual(audit["allowed_model_overlap_count"], 1)
        overlap = audit["allowed_model_overlaps"][0]
        self.assertEqual(overlap["resource"], "node:0:pd-fabric")
        self.assertEqual(overlap["kind"], "hbm_peer_external_astra")
        self.assertEqual(overlap["external_fabric_job_id"], job["job_id"])
        self.assertEqual(overlap["instance_id"], prefill.instance_id)
        self.assertEqual(overlap["batch_id"], model.batch_id)

    def test_atomic_ssd_chain_allows_safe_future_stage_hole_backfill(self):
        scheduler = FakeScheduler(instance_id=0, node_id=0, num_npus=4)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="tiered",
                pcie_bandwidth_gbps=0.25,
                cpu_bandwidth_gbps=1,
                cpu_transfer_latency_us=0,
                ssd_read_bandwidth_gbps=1,
                ssd_read_latency_us=0,
                ssd_write_bandwidth_gbps=1,
                ssd_write_latency_us=0,
            ),
        )
        media, h2d = manager._reserve_ssd_restore_stages(
            arrival_ns=0,
            staging_instance_id=0,
            target_instance_id=0,
            per_rank_bytes=25,
            total_bytes=100,
            session_id="older-chain",
            job_arrival_ns=0,
        )
        short = manager._reserve_transfer(
            kind="hbm_to_ssd_direct",
            arrival_ns=10,
            service_ns=20,
            source_instance_id=0,
            target_instance_id=None,
            num_bytes=20,
            background=True,
            session_id="short-backfill",
            job_arrival_ns=10,
        )
        long = manager._reserve_transfer(
            kind="hbm_to_ssd_direct",
            arrival_ns=40,
            service_ns=150,
            source_instance_id=0,
            target_instance_id=None,
            num_bytes=150,
            background=True,
            session_id="long-after-chain",
            job_arrival_ns=40,
        )

        self.assertEqual((media.start_ns, media.complete_ns), (0, 100))
        self.assertEqual((h2d.start_ns, h2d.complete_ns), (100, 200))
        self.assertEqual((short.start_ns, short.complete_ns), (10, 30))
        self.assertEqual((long.start_ns, long.complete_ns), (200, 350))
        self.assertEqual(media.parent_sequence, h2d.parent_sequence)
        for intervals in manager._resource_intervals.values():
            for previous, current in zip(intervals, intervals[1:]):
                self.assertLessEqual(previous[1], current[0])

    def test_direct_ssd_demotion_preflight_uses_calendar_hole(self):
        scheduler = FakeScheduler(
            instance_id=0, node_id=0, num_npus=1)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="hbm_ssd_direct",
                pcie_bandwidth_gbps=0.25,
                cpu_bandwidth_gbps=1,
                cpu_transfer_latency_us=0,
                ssd_read_bandwidth_gbps=1,
                ssd_read_latency_us=0,
                ssd_write_bandwidth_gbps=1,
                ssd_write_latency_us=0,
            ),
        )
        manager._reserve_ssd_restore_stages(
            arrival_ns=0,
            staging_instance_id=0,
            target_instance_id=0,
            per_rank_bytes=25,
            total_bytes=100,
            session_id="older-chain",
            job_arrival_ns=0,
        )
        victim = IdleKVEntry(
            session_id="victim", instance_id=0, tokens=1,
            block_tokens=1, per_rank_bytes=20, total_bytes=20,
            location=KVLocation.HBM, tier_since_ns=10,
            last_access_ns=10, next_use_ns=50,
        )
        manager.entries[victim.session_id] = victim
        manager._direct_ssd_write_shape = lambda entry: (20, 20, 20)

        self.assertTrue(manager._schedule_hbm_demotion(
            victim, 10, "hbm_capacity"))
        self.assertEqual(victim.migration_start_ns, 10)
        self.assertEqual(victim.migration_complete_ns, 30)

    def test_transient_dram_capacity_allows_bounded_concurrent_bounces(self):
        source = FakeScheduler(
            instance_id=0, node_id=0, num_npus=1, cpu_mem=250)
        target = FakeScheduler(
            instance_id=1, node_id=0, num_npus=1, cpu_mem=250)
        manager = AgenticKVManager(
            [source, target], AgenticKVConfig(
                policy="tiered",
                pcie_bandwidth_gbps=1,
                cpu_bandwidth_gbps=1,
                cpu_transfer_latency_us=0,
                ssd_read_bandwidth_gbps=1,
                ssd_read_latency_us=0,
            ))
        manager._insert_resource_interval(
            ("instance:0:pcie-copy:0",), 100, 300, 999, "test")

        first = manager._reserve_ssd_restore_stages(
            arrival_ns=0, staging_instance_id=0, target_instance_id=0,
            per_rank_bytes=100, total_bytes=100, session_id="first",
            job_arrival_ns=0)
        second = manager._reserve_ssd_restore_stages(
            arrival_ns=100, staging_instance_id=0, target_instance_id=1,
            per_rank_bytes=100, total_bytes=100, session_id="second",
            job_arrival_ns=100)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        reservations = manager._transient_dram_history[0]
        self.assertEqual(
            [(item.start_ns, item.complete_ns) for item in reservations],
            [(0, 400), (100, 300)],
        )
        self.assertEqual(manager.metrics.peak_transient_dram_bytes, 200)
        self.assertEqual(
            manager.metrics.peak_cpu_committed_plus_transient_bytes, 200)
        audit = manager.validate_resource_timeline()
        self.assertEqual(audit["transient_dram_capacity_violation_count"], 0)

    def test_transient_dram_capacity_serializes_overlapping_bounces(self):
        source = FakeScheduler(
            instance_id=0, node_id=0, num_npus=1, cpu_mem=150)
        target = FakeScheduler(
            instance_id=1, node_id=0, num_npus=1, cpu_mem=150)
        manager = AgenticKVManager(
            [source, target], AgenticKVConfig(
                policy="tiered",
                pcie_bandwidth_gbps=1,
                cpu_bandwidth_gbps=1,
                cpu_transfer_latency_us=0,
                ssd_read_bandwidth_gbps=1,
                ssd_read_latency_us=0,
            ))
        manager._insert_resource_interval(
            ("instance:0:pcie-copy:0",), 100, 300, 999, "test")
        manager._reserve_ssd_restore_stages(
            arrival_ns=0, staging_instance_id=0, target_instance_id=0,
            per_rank_bytes=100, total_bytes=100, session_id="first",
            job_arrival_ns=0)

        second = manager._reserve_ssd_restore_stages(
            arrival_ns=100, staging_instance_id=0, target_instance_id=1,
            per_rank_bytes=100, total_bytes=100, session_id="second",
            job_arrival_ns=100)

        self.assertIsNotNone(second)
        reservations = manager._transient_dram_history[0]
        self.assertEqual(
            [(item.start_ns, item.complete_ns) for item in reservations],
            [(0, 400), (400, 600)],
        )
        self.assertEqual(manager.metrics.peak_transient_dram_bytes, 100)
        self.assertEqual(manager.metrics.transient_dram_capacity_wait_ns, 300)
        self.assertEqual(manager.metrics.transient_dram_pressure_stall_ns, 300)

    def test_transient_dram_full_cpu_cascades_lru_to_ssd(self):
        scheduler = FakeScheduler(
            instance_id=0, node_id=0, num_npus=1, cpu_mem=100)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(
                policy="tiered",
                pcie_bandwidth_gbps=1,
                cpu_bandwidth_gbps=1,
                cpu_transfer_latency_us=0,
                ssd_read_bandwidth_gbps=1,
                ssd_read_latency_us=0,
                ssd_write_bandwidth_gbps=1,
                ssd_write_latency_us=0,
            ))
        victim = IdleKVEntry(
            session_id="cpu-lru", instance_id=0, tokens=1,
            block_tokens=1, per_rank_bytes=100, total_bytes=100,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[victim.session_id] = victim
        scheduler.memory.allocate(100, Device.CPU)

        stages = manager._reserve_ssd_restore_stages(
            arrival_ns=0, staging_instance_id=0, target_instance_id=0,
            per_rank_bytes=100, total_bytes=100, session_id="restore",
            job_arrival_ns=0)

        self.assertIsNotNone(stages)
        self.assertEqual(victim.migration_kind, "cpu_to_ssd")
        self.assertEqual(
            (stages[0].start_ns, stages[1].complete_ns), (100, 300))
        self.assertEqual(manager.metrics.transient_dram_pressure_stall_ns, 100)
        manager.advance(300)
        self.assertEqual(victim.location, KVLocation.SSD)
        self.assertEqual(manager.metrics.transient_dram_cpu_lru_evictions, 1)
        self.assertEqual(scheduler.memory.cpu_used, 0)

    def test_transient_dram_unknown_active_use_defers_without_overbooking(self):
        scheduler = FakeScheduler(
            instance_id=0, node_id=0, num_npus=1, cpu_mem=100)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(policy="hbm_ssd_direct"))
        scheduler.memory.allocate(100, Device.CPU)

        stages = manager._reserve_ssd_restore_stages(
            arrival_ns=0, staging_instance_id=0, target_instance_id=0,
            per_rank_bytes=100, total_bytes=100, session_id="restore",
            job_arrival_ns=0)

        self.assertIsNone(stages)
        self.assertEqual(manager.metrics.transient_dram_capacity_deferrals, 1)
        self.assertFalse(manager._transient_dram_history)

    def test_transfer_rejects_regressed_top_level_job_arrival(self):
        manager, _ = self.manager()
        manager._reserve_transfer(
            kind="cpu_to_hbm", arrival_ns=100, service_ns=10,
            source_instance_id=0, target_instance_id=0,
            num_bytes=1, background=False, session_id="later",
            job_arrival_ns=100)

        with self.assertRaisesRegex(RuntimeError, "arrived behind"):
            manager._reserve_transfer(
                kind="cpu_to_hbm", arrival_ns=10, service_ns=10,
                source_instance_id=0, target_instance_id=0,
                num_bytes=1, background=False, session_id="earlier",
                job_arrival_ns=10)

    def test_queued_deadline_cancellation_has_ordered_timestamps(self):
        manager, _ = self.manager()
        manager._reserve_transfer(
            kind="cpu_to_hbm", arrival_ns=0, service_ns=100,
            source_instance_id=0, target_instance_id=0,
            num_bytes=1, background=False, session_id="occupier")
        cancelled = manager._reserve_transfer(
            kind="hbm_to_cpu", arrival_ns=10, service_ns=10,
            source_instance_id=0, target_instance_id=None,
            num_bytes=1, background=True, deadline_ns=50,
            session_id="cancelled")

        self.assertFalse(cancelled.completed)
        self.assertEqual(cancelled.start_ns, 50)
        self.assertEqual(cancelled.complete_ns, 50)
        self.assertEqual(cancelled.active_ns_before_cancel, 0)
        self.assertLessEqual(cancelled.start_ns, cancelled.complete_ns)

        active_manager, _ = self.manager()
        active_manager._reserve_transfer(
            kind="hbm_to_cpu", arrival_ns=0, service_ns=100,
            source_instance_id=0, target_instance_id=None,
            num_bytes=100, background=True, deadline_ns=40,
            session_id="active-cancel")
        tail = active_manager.transfer_tail_at(20)
        self.assertEqual(tail["active_service_jobs"], 1)
        self.assertEqual(tail["outstanding_bytes"], 100)
        self.assertEqual(tail["max_tail_ns"], 20)

    @staticmethod
    def _fabric_batch(batch_id, batch_time):
        request = SimpleNamespace(
            id=batch_id,
            session_id=f"batch-{batch_id}",
            agentic_kv_source="hbm",
            return_gap_type="tool",
            agentic_kv_async_decode_join=False,
            agentic_kv_restore_ns=0,
            is_prefill=lambda: False,
        )
        return SimpleNamespace(
            batch_id=batch_id,
            batch_time=batch_time,
            requests=[request],
        )

    def test_direct_fabric_overlaps_astra_without_gating_dispatch(self):
        source = FakeScheduler(instance_id=0, node_id=0, num_npus=4)
        target = FakeScheduler(instance_id=1, node_id=0, num_npus=4)
        other_node = FakeScheduler(instance_id=2, node_id=1, num_npus=4)
        source.pd_type = "decode"
        target.pd_type = "prefill"
        other_node.pd_type = "decode"
        manager = AgenticKVManager(
            [source, target, other_node],
            AgenticKVConfig(
                policy="preserve",
                pd_peer_transfer_mode="direct-fabric",
            ),
        )
        entry = IdleKVEntry(
            session_id="returning", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=6400,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[entry.session_id] = entry
        current = self._fabric_batch(10, 100)
        manager.record_agentic_batch_schedule(source, current)
        manager.record_astra_workload_dispatch(source, current, 100)

        boundary_instances = manager.prepare_boundary_instances(
            "returning", 1, 16, 150)
        self.assertEqual(boundary_instances, ())
        manager.acquire_prepare_lock(
            99, boundary_instances, session_id="returning")
        self.assertFalse(manager.prepare_locked(0))
        self.assertFalse(manager.prepare_locked(1))
        self.assertFalse(manager.prepare_locked(2))
        reservation = manager._reserve_transfer(
            kind="hbm_peer", arrival_ns=150, service_ns=100,
            source_instance_id=0, target_instance_id=1,
            num_bytes=6400, background=False,
            session_id="returning",
        )
        self.assertEqual((reservation.start_ns, reservation.complete_ns),
                         (150, 250))
        self.assertIsNone(manager.model_dispatch_blocked_until(0, 175))
        self.assertIsNone(manager.model_dispatch_blocked_until(1, 175))
        self.assertIsNone(manager.model_dispatch_blocked_until(2, 175))
        self.assertEqual(manager.metrics.direct_fabric_dispatch_blocks, 0)
        self.assertEqual(manager.metrics.direct_fabric_dispatch_wait_ns, 0)
        self.assertEqual(
            manager.model_dispatch_resource_ready_time(0, 175), 175)
        self.assertIsNone(manager.synchronous_swap_blocked_until(0, 175))

        # An unrelated target batch dispatches while the copy is in flight.
        future = self._fabric_batch(11, 175)
        manager.record_agentic_batch_schedule(target, future)
        manager.record_astra_workload_dispatch(target, future, 175)
        manager.record_agentic_batch_complete(target, future, 275)
        manager.record_agentic_batch_complete(source, current, 300)
        self.assertEqual(
            manager.metrics.agentic_model_iteration_execution_ns, 300)
        manager.release_prepare_lock(99)
        audit = manager.validate_resource_timeline()
        self.assertEqual(audit["forbidden_overlap_count"], 0)
        self.assertEqual(audit["cold_peer_fcfs_overlap_count"], 0)
        self.assertGreater(audit["allowed_model_overlap_count"], 0)
        self.assertFalse(audit["current_batch_latency_extended_by_cold_copy"])
        self.assertFalse(audit["future_fabric_dispatch_is_gated"])

    def test_late_direct_fabric_event_keeps_logical_issue_time(self):
        source = FakeScheduler(instance_id=0, node_id=0, num_npus=4)
        target = FakeScheduler(instance_id=1, node_id=0, num_npus=4)
        source.pd_type = "decode"
        target.pd_type = "prefill"
        manager = AgenticKVManager(
            [source, target],
            AgenticKVConfig(
                policy="preserve",
                pd_peer_transfer_mode="direct-fabric",
            ),
        )
        batch = self._fabric_batch(20, 100)
        manager.record_agentic_batch_schedule(source, batch)
        manager.record_astra_workload_dispatch(source, batch, 100)
        manager.record_agentic_batch_complete(source, batch, 300)

        # A retrospective event keeps its logical issue time. ASTRA execution
        # is an independent contention domain and may overlap it.
        reservation = manager._reserve_transfer(
            kind="hbm_peer", arrival_ns=150, service_ns=50,
            source_instance_id=0, target_instance_id=1,
            num_bytes=1, background=False, session_id="late",
        )
        self.assertEqual(reservation.start_ns, 150)
        self.assertEqual(reservation.queue_wait_ns, 0)
        self.assertEqual(reservation.complete_ns, 200)
        audit = manager.validate_resource_timeline()
        self.assertEqual(audit["forbidden_overlap_count"], 0)
        self.assertEqual(audit["allowed_model_overlap_count"], 1)

    def test_pcie_restore_can_overlap_running_astra_batch(self):
        scheduler = FakeScheduler(instance_id=0, node_id=0, num_npus=4)
        scheduler.pd_type = "decode"
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="tiered",
                pd_peer_transfer_mode="direct-fabric",
                swap_execution_mode="async-pre-admission",
            ),
        )
        batch = self._fabric_batch(30, 100)
        manager.record_agentic_batch_schedule(scheduler, batch)
        manager.record_astra_workload_dispatch(scheduler, batch, 100)
        reservation = manager._reserve_transfer(
            kind="cpu_to_hbm", arrival_ns=150, service_ns=50,
            source_instance_id=0, target_instance_id=0,
            num_bytes=1, background=False, session_id="owner",
        )
        self.assertEqual((reservation.start_ns, reservation.complete_ns),
                         (150, 200))
        self.assertIsNone(manager.model_dispatch_blocked_until(0, 175))
        self.assertIsNone(manager.synchronous_swap_blocked_until(0, 175))
        manager.record_agentic_batch_complete(scheduler, batch, 300)
        audit = manager.validate_resource_timeline()
        self.assertTrue(audit["pcie_dram_dma_may_overlap_model_execution"])

    def test_formed_dp_batch_is_not_a_fabric_owner_or_boundary_deadlock(self):
        source = FakeScheduler(instance_id=0, node_id=0, num_npus=4)
        target = FakeScheduler(instance_id=1, node_id=0, num_npus=4)
        source.pd_type = "decode"
        target.pd_type = "prefill"
        manager = AgenticKVManager(
            [source, target],
            AgenticKVConfig(
                policy="preserve",
                pd_peer_transfer_mode="direct-fabric",
            ),
        )
        manager.entries["returning"] = IdleKVEntry(
            session_id="returning", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=6400,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        formed = self._fabric_batch(40, 100)
        manager.record_agentic_batch_schedule(source, formed)
        source.inflight.append(formed)  # Mirrors scheduler -> dp_pending.

        instances = manager.prepare_boundary_instances(
            "returning", 1, 16, 150)
        self.assertEqual(instances, ())
        self.assertFalse(manager.prepare_boundary_busy(instances))
        self.assertEqual(manager._astra_fabric_inflight, {})

        reservation = manager._reserve_transfer(
            kind="hbm_peer", arrival_ns=150, service_ns=50,
            source_instance_id=0, target_instance_id=1,
            num_bytes=1, background=False, session_id="returning",
        )
        self.assertEqual(reservation.complete_ns, 200)
        self.assertIsNone(manager.model_dispatch_blocked_until(0, 175))
        # The already-formed unrelated wave may dispatch during the copy.
        manager.record_astra_workload_dispatch(source, formed, 175)
        manager.record_agentic_batch_complete(source, formed, 300)
        dispatch = next(
            event for event in manager.events
            if event.get("event") == "astra_workload_dispatch"
        )
        self.assertEqual(dispatch["formation_to_dispatch_wait_ns"], 75)
        self.assertEqual(
            manager.metrics.agentic_model_iteration_execution_ns, 125)
        self.assertEqual(
            manager.validate_resource_timeline()["open_astra_window_count"],
            0,
        )

    def test_dummy_workload_uses_explicit_dispatch_and_completion(self):
        scheduler = FakeScheduler(instance_id=0, node_id=0, num_npus=4)
        scheduler.pd_type = "decode"
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="preserve",
                pd_peer_transfer_mode="direct-fabric",
            ),
        )
        dummy = SimpleNamespace(batch_id=50, batch_time=100, requests=[])
        manager.record_astra_workload_dispatch(scheduler, dummy, 125)
        self.assertFalse(manager.prepare_boundary_busy((0,)))
        duration_ns = manager.record_astra_workload_complete(
            scheduler, dummy, 175)
        self.assertEqual(duration_ns, 50)
        self.assertFalse(manager.prepare_boundary_busy((0,)))
        audit = manager.validate_resource_timeline()
        self.assertEqual(audit["astra_window_count"], 1)
        self.assertEqual(audit["open_astra_window_count"], 0)

    def test_model_fabric_wait_does_not_overwrite_hbm_memory_wait(self):
        manager = SimpleNamespace(
            model_dispatch_blocked_until=lambda instance_id, now_ns: 200,
        )
        scheduler = SimpleNamespace(
            agentic_kv_manager=manager,
            instance_id=0,
            memory_wait_until_ns=777,
            model_fabric_wait_until_ns=None,
        )
        self.assertTrue(Scheduler._model_resource_blocked(scheduler, 100))
        self.assertEqual(scheduler.model_fabric_wait_until_ns, 200)
        self.assertEqual(scheduler.memory_wait_until_ns, 777)

    def test_astra_calendar_does_not_shift_async_copy(self):
        manager = AgenticKVManager(
            [FakeScheduler(instance_id=0, node_id=0, num_npus=4)],
            AgenticKVConfig(
                policy="preserve",
                pd_peer_transfer_mode="direct-fabric",
            ),
        )

        class CountingSequence:
            def __init__(self, values):
                self.values = values
                self.reads = 0

            def __len__(self):
                return len(self.values)

            def __getitem__(self, index):
                self.reads += 1
                return self.values[index]

        windows = CountingSequence([
            (index * 20, index * 20 + 10)
            for index in range(50_000)
        ])
        manager._astra_fabric_calendar["node:0:pd-fabric"] = windows
        start_ns = 49_999 * 20 + 5
        ready_ns = manager._after_completed_astra_windows(
            ("node:0:pd-fabric",), start_ns, 1)
        self.assertEqual(ready_ns, start_ns)
        self.assertEqual(windows.reads, 0)

    def test_fabric_calendar_coalesces_out_of_order_completions(self):
        manager = AgenticKVManager(
            [FakeScheduler(instance_id=0, node_id=0, num_npus=4)],
            AgenticKVConfig(
                policy="preserve",
                pd_peer_transfer_mode="direct-fabric",
            ),
        )
        resource = "node:0:pd-fabric"
        manager._insert_astra_calendar_window(resource, 0, 10)
        manager._insert_astra_calendar_window(resource, 20, 30)
        # This callback arrives late and overlaps only the first interval. It
        # must not erase the real 15..20 idle gap before the second interval.
        manager._insert_astra_calendar_window(resource, 5, 15)
        self.assertEqual(
            manager._astra_fabric_calendar[resource],
            [(0, 15), (20, 30)],
        )
        self.assertEqual(
            manager._after_completed_astra_windows((resource,), 12, 4),
            12,
        )

    def test_sync_swap_barriers_cover_gpu_facing_in_and_out_paths(self):
        manager, _ = self.manager(
            swap_execution_mode="sync-engine-barrier")
        swap_out = manager._reserve_transfer(
            kind="hbm_to_cpu",
            arrival_ns=100,
            service_ns=50,
            source_instance_id=0,
            target_instance_id=None,
            num_bytes=1,
            background=True,
            session_id="out",
        )
        self.assertEqual(swap_out.complete_ns, 150)
        self.assertEqual(manager.synchronous_swap_blocked_until(0, 100), 150)
        self.assertEqual(manager.synchronous_swap_blocked_until(0, 149), 150)
        self.assertIsNone(manager.synchronous_swap_blocked_until(0, 150))

        swap_in = manager._reserve_transfer(
            kind="cpu_to_hbm",
            arrival_ns=200,
            service_ns=75,
            source_instance_id=0,
            target_instance_id=0,
            num_bytes=1,
            background=False,
            session_id="in",
        )
        self.assertEqual(swap_in.complete_ns, 275)
        self.assertEqual(manager.synchronous_swap_blocked_until(0, 250), 275)

    def test_sync_swap_does_not_block_cpu_to_ssd_or_hbm_peer(self):
        manager, _ = self.manager(
            swap_execution_mode="sync-engine-barrier")
        manager._reserve_transfer(
            kind="cpu_to_ssd",
            arrival_ns=100,
            service_ns=50,
            source_instance_id=0,
            target_instance_id=None,
            num_bytes=1,
            background=True,
            session_id="ssd",
        )
        manager._reserve_transfer(
            kind="hbm_peer",
            arrival_ns=200,
            service_ns=50,
            source_instance_id=0,
            target_instance_id=0,
            num_bytes=1,
            background=False,
            session_id="peer",
        )
        self.assertIsNone(manager.synchronous_swap_blocked_until(0, 125))
        self.assertIsNone(manager.synchronous_swap_blocked_until(0, 225))

    def test_async_pre_admission_has_no_engine_barrier(self):
        manager, _ = self.manager(
            swap_execution_mode="async-pre-admission")
        manager._reserve_transfer(
            kind="cpu_stage_to_hbm",
            arrival_ns=100,
            service_ns=50,
            source_instance_id=0,
            target_instance_id=0,
            num_bytes=1,
            background=False,
            session_id="async",
        )
        self.assertIsNone(manager.synchronous_swap_blocked_until(0, 125))

    def test_async_pre_admission_reserves_hbm_before_ssd_load(self):
        manager, scheduler = self.manager(
            swap_execution_mode="async-pre-admission",
            ssd_read_bandwidth_gbps=0.001,
            ssd_read_latency_us=0,
        )
        entry = IdleKVEntry(
            session_id="cold", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=12800,
            location=KVLocation.SSD, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[entry.session_id] = entry
        manager.ssd_records[entry.session_id] = SSDRecord(
            tokens=16, block_tokens=16, bytes=12800,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 12800

        prep = manager.prepare_request("cold", 0, 16, 32, 100)

        self.assertGreater(prep.restore_ready_time_ns, 100)
        self.assertEqual(scheduler.memory.npu_used, 1600)
        self.assertEqual(
            manager.hbm_unreserved_per_rank_bytes(0),
            scheduler.memory.npu_mem - 1600,
        )
        self.assertIsNone(
            manager.synchronous_swap_blocked_until(0, 101))
        media = next(
            event for event in manager.events
            if event.get("kind") == "ssd_to_cpu_stage")
        h2d = next(
            event for event in manager.events
            if event.get("kind") == "cpu_stage_to_hbm")
        self.assertEqual(media["complete_ns"], h2d["time_ns"])
        self.assertEqual(h2d["complete_ns"], prep.restore_ready_time_ns)
        self.assertIn("ssd-pool:read", media["resources"])
        self.assertIn("node:0:dram", media["resources"])
        self.assertFalse(any(
            "pcie-copy" in resource for resource in media["resources"]))
        self.assertNotIn("ssd-pool:read", h2d["resources"])
        self.assertIn("node:0:dram", h2d["resources"])
        self.assertEqual(
            sum("pcie-copy" in resource
                for resource in h2d["resources"]),
            scheduler.num_npus,
        )
        self.assertEqual(manager.metrics.ssd_to_cpu_stage_bytes, 12800)
        self.assertEqual(manager.metrics.cpu_stage_to_hbm_bytes, 12800)

    def test_staged_ssd_restore_keeps_one_full_legacy_sync_barrier(self):
        manager, _ = self.manager(
            swap_execution_mode="sync-engine-barrier",
            ssd_read_bandwidth_gbps=0.001,
            ssd_read_latency_us=0,
        )
        entry = IdleKVEntry(
            session_id="sync-cold", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=12800,
            location=KVLocation.SSD, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[entry.session_id] = entry
        manager.ssd_records[entry.session_id] = SSDRecord(
            tokens=16, block_tokens=16, bytes=12800,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 12800

        prep = manager.prepare_request("sync-cold", 0, 16, 32, 100)

        foreground = [
            event for event in manager.events
            if event.get("event") == "migration_reserve"
            and event.get("foreground")
        ]
        self.assertEqual(
            [event["kind"] for event in foreground],
            ["ssd_to_cpu_stage", "cpu_stage_to_hbm"],
        )
        self.assertEqual(manager.metrics.sync_swap_barrier_jobs, 1)
        self.assertEqual(
            manager.synchronous_swap_blocked_until(0, 100),
            prep.restore_ready_time_ns,
        )
        barriers = [
            event for event in manager.events
            if event.get("event") == "sync_swap_engine_barrier"
        ]
        self.assertEqual(len(barriers), 1)
        self.assertEqual(barriers[0]["kind"], "ssd_staged_to_hbm")
        self.assertEqual(barriers[0]["start_ns"], 100)
        self.assertEqual(
            barriers[0]["complete_ns"], prep.restore_ready_time_ns)

    def test_async_decode_join_separates_gross_overlap_and_owner_barrier(self):
        manager, scheduler = self.manager(
            swap_execution_mode="async-decode-join")
        scheduler.pd_type = "prefill"
        request = Request(1, "model", 10, 12, 100, 0)
        request.session_id = "async"
        request.sub_request_index = 1
        request.return_gap_type = "tool"
        request.agentic_kv_source = "cpu"
        request.agentic_kv_residency_at_return = "cpu"
        request.agentic_kv_async_decode_join = True
        request.agentic_kv_restore_ns = 100
        request.agentic_kv_restore_issue_time_ns = 100
        request.agentic_kv_restore_ready_time_ns = 200
        batch = SimpleNamespace(
            batch_time=120,
            batch_id=1,
            requests=[request],
        )

        manager.record_agentic_request(request)
        manager.record_agentic_batch_schedule(scheduler, batch)
        manager.record_astra_workload_dispatch(scheduler, batch, 120)
        manager.record_agentic_batch_complete(scheduler, batch, 180)
        manager.record_async_restore_gate(request, 180)
        summary = manager.summary(250)
        async_restore = summary["asynchronous_restore"]

        self.assertEqual(async_restore["aggregate_swap_in_gross_ns"], 100)
        self.assertEqual(
            async_restore["aggregate_prefill_execution_overlap_ns"], 60)
        self.assertEqual(
            async_restore["aggregate_owner_decode_barrier_ns"], 20)
        self.assertEqual(async_restore["aggregate_other_hidden_ns"], 20)
        self.assertFalse(async_restore["swap_out_blocks_model"])
        self.assertFalse(async_restore["swap_in_blocks_other_requests"])
        self.assertEqual(
            summary["observed_load_activity"][
                "global_any_model_execution_ns"],
            60,
        )

    def test_async_one_fresh_late_gate_keeps_gross_restore_time(self):
        manager, _ = self.manager(
            swap_execution_mode="async-decode-join")
        request = Request(2, "model", 10, 12, 100, 0)
        request.session_id = "async-late"
        request.sub_request_index = 1
        request.return_gap_type = "tool"
        request.agentic_kv_source = "ssd"
        request.agentic_kv_residency_at_return = "ssd"
        request.agentic_kv_async_decode_join = True
        request.agentic_kv_restore_ns = 100
        request.agentic_kv_restore_issue_time_ns = 100
        request.agentic_kv_restore_ready_time_ns = 200

        manager.record_agentic_request(request)
        manager.record_async_restore_gate(request, 250)
        async_restore = manager.summary(300)["asynchronous_restore"]

        self.assertEqual(async_restore["aggregate_swap_in_gross_ns"], 100)
        self.assertEqual(
            async_restore["aggregate_owner_decode_barrier_ns"], 0)
        self.assertEqual(async_restore["aggregate_other_hidden_ns"], 100)

    def test_sync_foreground_restore_is_exposed_before_next_batch(self):
        manager, scheduler = self.manager(
            swap_execution_mode="sync-engine-barrier")
        scheduler.pd_type = "prefill"
        manager._reserve_transfer(
            kind="cpu_to_hbm",
            arrival_ns=100,
            service_ns=50,
            source_instance_id=0,
            target_instance_id=0,
            num_bytes=1,
            background=False,
            session_id="returning",
        )
        request = SimpleNamespace(
            id=1,
            session_id="returning",
            agentic_kv_source="cpu",
            return_gap_type="tool",
            agentic_kv_async_decode_join=False,
            agentic_kv_restore_ns=0,
            is_prefill=lambda: False,
        )
        batch = SimpleNamespace(
            batch_time=150,
            batch_id=1,
            requests=[request],
        )

        manager.record_agentic_batch_schedule(scheduler, batch)
        manager.record_astra_workload_dispatch(scheduler, batch, 150)
        manager.record_agentic_batch_complete(scheduler, batch, 250)
        summary = manager.summary(250)["synchronous_swap"]

        self.assertEqual(batch.agentic_sync_swap_wait_ns, 50)
        self.assertTrue(batch.agentic_sync_swap_barrier_before_batch)
        scheduled = next(
            event for event in manager.events
            if event.get("event") == "agentic_batch_schedule")
        self.assertFalse(scheduled["restore_barrier_inside_batch"])
        self.assertTrue(scheduled["sync_swap_barrier_before_batch"])
        self.assertEqual(summary["aggregate_exposed_engine_wait_ns"], 50)
        self.assertEqual(
            summary[
                "batch_blocking_swap_wait_fraction_of_model_iteration_time"],
            1 / 3,
        )

    def test_sync_background_swap_counts_only_exposed_dispatch_tail(self):
        manager, scheduler = self.manager(
            swap_execution_mode="sync-engine-barrier")
        scheduler.pd_type = "decode"
        manager._register_sync_swap_barrier(
            TransferReservation(
                kind="hbm_to_cpu",
                arrival_ns=0,
                start_ns=0,
                complete_ns=100,
                service_ns=100,
                queue_wait_ns=0,
                resources=(),
            ),
            0,
            None,
            "demoted",
        )
        victim = SimpleNamespace(
            id=2,
            session_id="hbm-victim",
            agentic_kv_source="hbm",
            return_gap_type="human",
            agentic_kv_async_decode_join=False,
            agentic_kv_restore_ns=0,
            is_prefill=lambda: False,
        )

        self.assertEqual(
            manager.record_synchronous_swap_dispatch_block(
                0, 50, [victim]),
            100,
        )
        batch = SimpleNamespace(
            batch_time=100,
            batch_id=2,
            requests=[victim],
        )
        manager.record_agentic_batch_schedule(scheduler, batch)
        manager.record_astra_workload_dispatch(scheduler, batch, 100)
        manager.record_agentic_batch_complete(scheduler, batch, 200)
        summary = manager.summary(200)["synchronous_swap"]

        self.assertEqual(batch.agentic_sync_swap_wait_ns, 50)
        self.assertEqual(
            summary["aggregate_reservation_barrier_union_ns"], 100)
        self.assertEqual(summary["aggregate_exposed_engine_wait_ns"], 50)
        self.assertEqual(summary["unique_ready_victim_requests"], 1)
        self.assertEqual(summary["aggregate_ready_victim_wait_ns"], 50)

    def test_sync_background_swap_without_runnable_work_is_not_exposed(self):
        manager, scheduler = self.manager(
            swap_execution_mode="sync-engine-barrier")
        scheduler.pd_type = "prefill"
        manager._reserve_transfer(
            kind="hbm_to_cpu",
            arrival_ns=0,
            service_ns=100,
            source_instance_id=0,
            target_instance_id=None,
            num_bytes=1,
            background=True,
            session_id="idle",
        )
        batch = SimpleNamespace(
            batch_time=100,
            batch_id=3,
            requests=[SimpleNamespace(
                id=3,
                session_id="new",
                agentic_kv_source=None,
                return_gap_type="session_start",
            )],
        )
        manager.record_agentic_batch_schedule(scheduler, batch)

        summary = manager.summary(100)["synchronous_swap"]
        self.assertEqual(batch.agentic_sync_swap_wait_ns, 0)
        self.assertEqual(
            summary["aggregate_reservation_barrier_union_ns"], 100)
        self.assertEqual(summary["aggregate_exposed_engine_wait_ns"], 0)

    def test_sync_nested_barriers_preserve_all_directions(self):
        manager, scheduler = self.manager(
            swap_execution_mode="sync-engine-barrier")
        scheduler.pd_type = "decode"
        for kind, start, end in (
                ("hbm_to_cpu", 0, 100),
                ("cpu_to_hbm", 50, 60)):
            manager._register_sync_swap_barrier(
                TransferReservation(
                    kind=kind,
                    arrival_ns=start,
                    start_ns=start,
                    complete_ns=end,
                    service_ns=end - start,
                    queue_wait_ns=0,
                    resources=(),
                ),
                0,
                0,
                kind,
            )
        victim = SimpleNamespace(
            id=4,
            session_id="victim",
            agentic_kv_source="hbm",
            return_gap_type="tool",
        )
        manager.record_synchronous_swap_dispatch_block(0, 50, [victim])
        batch = SimpleNamespace(
            batch_time=100,
            batch_id=4,
            requests=[victim],
        )
        manager.record_agentic_batch_schedule(scheduler, batch)

        self.assertEqual(
            batch.agentic_sync_swap_directions, ("in", "out"))

    def test_headline_ssd_baselines_share_hardware_assumptions(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs" / "agentic_kv"
        hbm_only = AgenticKVConfig.from_json(
            str(config_dir / "hbm_lru_recompute.json"))
        direct = AgenticKVConfig.from_json(
            str(config_dir / "hbm_ssd_direct_8ssd.json"))
        tiered = AgenticKVConfig.from_json(
            str(config_dir / "tiered_capacity_fullwrite_8ssd.json"))

        for field in (
                "pcie_bandwidth_gbps",
                "ssd_read_bandwidth_gbps",
                "ssd_write_bandwidth_gbps",
                "ssd_read_latency_us",
                "ssd_write_latency_us",
                "ssd_capacity_gb",
                "ssd_num_devices",
                "ssd_write_mode",
                "block_size"):
            self.assertEqual(getattr(direct, field), getattr(tiered, field))
        self.assertEqual(direct.demotion_mode, "capacity-only")
        self.assertEqual(tiered.demotion_mode, "capacity-only")
        self.assertEqual(direct.pcie_bandwidth_gbps, 50.0)
        self.assertEqual(direct.cpu_bandwidth_gbps, 400.0)
        self.assertEqual(direct.ssd_read_bandwidth_gbps, 55.2)
        self.assertEqual(direct.ssd_write_bandwidth_gbps, 33.6)
        self.assertEqual(direct.ssd_capacity_bytes, 30_720_000_000_000)
        self.assertEqual(direct.active_preemption_mode, "recompute")
        self.assertEqual(tiered.active_preemption_mode, "recompute")
        for config in (hbm_only, direct, tiered):
            self.assertEqual(
                config.swap_execution_mode, "async-pre-admission")

    def test_qwen_1m_p4d4_baselines_share_direct_peer_contract(self):
        config_dir = (
            Path(__file__).resolve().parents[1]
            / "configs" / "agentic_kv" / "qwen3_1m_p4d4"
        )
        configs = [
            AgenticKVConfig.from_json(str(config_dir / filename))
            for filename in (
                "hbm_lru_recompute.json",
                "hbm_ssd_direct.json",
                "tiered.json",
                "tiered_queue_recompute.json",
            )
        ]

        self.assertEqual(
            [config.policy for config in configs],
            [
                "hbm_lru_recompute", "hbm_ssd_direct", "tiered",
                "tiered_queue_recompute",
            ],
        )
        for config in configs:
            self.assertEqual(config.pd_peer_transfer_mode, "direct-fabric")
            self.assertEqual(config.pd_peer_bandwidth_gbps, 450.0)
            self.assertEqual(config.pd_peer_latency_us, 1.0)
            self.assertEqual(config.pcie_bandwidth_gbps, 50.0)
            self.assertEqual(config.cpu_bandwidth_gbps, 200.0)
            self.assertEqual(config.ssd_read_bandwidth_gbps, 55.2)
            self.assertEqual(config.ssd_write_bandwidth_gbps, 33.6)
            self.assertEqual(config.ssd_num_devices, 8)
            self.assertEqual(config.demotion_mode, "capacity-only")
            self.assertEqual(
                config.swap_execution_mode, "async-pre-admission"
            )
        self.assertEqual(
            configs[-1].queue_recompute_wait_service_ratio, 4.0)
        self.assertEqual(
            configs[-1].queue_recompute_cost_guard_multiplier, 1.25)
        self.assertEqual(
            configs[-1].queue_recompute_prefill_headroom_chunks, 1.0)

    def test_qwen_partial_policy_has_conservative_threshold_candidates(self):
        config_dir = (
            Path(__file__).resolve().parents[1]
            / "configs" / "agentic_kv" / "qwen3_1m_p4d4"
        )
        candidates = [
            AgenticKVConfig.from_json(str(config_dir / filename))
            for filename in (
                "tiered_partial_recompute_r8_g1_25.json",
                "tiered_partial_recompute_r12_g1_5.json",
            )
        ]

        self.assertEqual(
            [item.queue_recompute_wait_service_ratio for item in candidates],
            [8.0, 12.0],
        )
        self.assertEqual(
            [item.queue_recompute_cost_guard_multiplier for item in candidates],
            [1.25, 1.5],
        )
        self.assertTrue(all(
            item.policy == "tiered_queue_recompute"
            and item.queue_recompute_prefill_headroom_chunks == 1.0
            for item in candidates
        ))

    def test_capacity_only_tiered_policy_ignores_all_ttls(self):
        manager, scheduler = self.manager(
            demotion_mode="capacity-only",
            hbm_ttl_ms=0,
            cpu_ttl_ms=0,
            ssd_ttl_ms=0,
        )
        hbm = IdleKVEntry(
            session_id="hbm", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=12800,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        cpu = IdleKVEntry(
            session_id="cpu", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=12800,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
        )
        ssd = IdleKVEntry(
            session_id="ssd", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=12800,
            location=KVLocation.SSD, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries = {entry.session_id: entry for entry in (hbm, cpu, ssd)}
        manager.ssd_records["ssd"] = SSDRecord(
            tokens=16, block_tokens=16, bytes=12800,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_records["shadow"] = SSDRecord(
            tokens=16, block_tokens=16, bytes=12800,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 25600
        scheduler.memory.allocate(1600, Device.NPU)
        scheduler.memory.allocate(12800, Device.CPU)

        manager.advance(10**18)

        self.assertEqual(hbm.location, KVLocation.HBM)
        self.assertEqual(cpu.location, KVLocation.CPU)
        self.assertEqual(ssd.location, KVLocation.SSD)
        self.assertEqual(set(manager.ssd_records), {"ssd", "shadow"})
        self.assertEqual(manager.metrics.ttl_drops, 0)

    def test_sync_ttl_swap_out_waits_for_iteration_boundary(self):
        manager, scheduler = self.manager(
            swap_execution_mode="sync-engine-barrier",
            hbm_ttl_ms=0,
        )
        manager.on_tool_start(
            FakeRequest(session_id="ttl", tokens=16), 0, 1_000_000)
        scheduler.inflight.append(object())

        manager.advance(1)

        entry = manager.entries["ttl"]
        self.assertIsNone(entry.migration_kind)
        self.assertEqual(manager.metrics.transfer_jobs, 0)

        scheduler.inflight.clear()
        manager.advance(100)
        self.assertEqual(entry.migration_kind, "hbm_to_cpu")
        self.assertEqual(manager.metrics.transfer_jobs, 1)
        migration = next(
            event for event in manager.events
            if event.get("event") == "migration_reserve")
        self.assertEqual(migration["start_ns"], 100)

    def test_sync_hbm_resume_clears_deferred_ttl_marker(self):
        manager, scheduler = self.manager(
            swap_execution_mode="sync-engine-barrier",
            hbm_ttl_ms=0,
        )
        manager.on_tool_start(
            FakeRequest(session_id="repeat", tokens=16), 0, 1_000_000)
        scheduler.inflight.append(object())
        manager.advance(1)
        self.assertIn("repeat", manager._sync_deferred_hbm_demotions)

        prep = manager.prepare_request("repeat", 0, 16, 17, 10)

        self.assertEqual(prep.source, KVLocation.HBM)
        self.assertNotIn("repeat", manager._sync_deferred_hbm_demotions)

    def test_sync_pending_resume_pins_its_idle_lru_source(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=1600)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="hbm_lru_recompute",
                swap_execution_mode="sync-engine-barrier",
            ),
        )
        returning = IdleKVEntry(
            session_id="returning",
            instance_id=0,
            tokens=16,
            block_tokens=16,
            per_rank_bytes=1600,
            total_bytes=1600,
            location=KVLocation.HBM,
            tier_since_ns=0,
            last_access_ns=0,
        )
        manager.entries[returning.session_id] = returning
        scheduler.memory.allocate(1600, Device.NPU)
        manager.acquire_synchronous_prepare_lock(
            1, (0,), session_id="returning")
        newcomer = IdleKVEntry(
            session_id="newcomer",
            instance_id=0,
            tokens=16,
            block_tokens=16,
            per_rank_bytes=1600,
            total_bytes=1600,
            location=KVLocation.HBM,
            tier_since_ns=10,
            last_access_ns=10,
        )

        self.assertIsNone(manager._reserve_hbm(newcomer, 10))
        self.assertEqual(returning.location, KVLocation.HBM)

        manager.release_synchronous_prepare_lock(1)
        self.assertEqual(manager._reserve_hbm(newcomer, 10), 10)
        self.assertEqual(returning.location, KVLocation.DROPPED)

    def test_sync_active_reclaim_defers_new_swap_until_boundary(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=1600)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="tiered",
                demotion_mode="capacity-only",
                swap_execution_mode="sync-engine-barrier",
            ),
        )
        victim = IdleKVEntry(
            session_id="victim",
            instance_id=0,
            tokens=16,
            block_tokens=16,
            per_rank_bytes=1600,
            total_bytes=1600,
            location=KVLocation.HBM,
            tier_since_ns=0,
            last_access_ns=0,
        )
        manager.entries[victim.session_id] = victim
        scheduler.memory.allocate(1600, Device.NPU)
        scheduler.inflight.append(object())

        self.assertTrue(manager.synchronous_hbm_reclaim_needs_boundary(
            0, 1600, 50))
        self.assertIsNone(manager.claim_active_hbm_reclaim(0, 1600, 50))
        self.assertEqual(manager.metrics.transfer_jobs, 0)

        scheduler.inflight.clear()
        ready_ns = manager.claim_active_hbm_reclaim(0, 1600, 100)
        self.assertGreater(ready_ns, 100)
        migration = next(
            event for event in manager.events
            if event.get("event") == "migration_reserve")
        self.assertEqual(migration["start_ns"], 100)
        self.assertEqual(manager.metrics.hbm_to_cpu_bytes, 0)
        self.assertEqual(manager.metrics.cpu_to_ssd_bytes, 0)

    def test_tp_transfer_uses_parallel_gpu_links_and_aggregate_cpu(self):
        manager, _ = self.manager()
        per_rank = 100_000_000_000
        total = 8 * per_rank
        # PCIe takes 2 s per rank; aggregate DRAM takes 4 s. Add 5 us.
        self.assertEqual(
            manager._cpu_transfer_ns(per_rank, total),
            4_000_005_000,
        )

    def test_active_preemption_totals_are_exposed_in_summary(self):
        manager, _ = self.manager()
        manager.record_active_preemption_totals(
            recompute_preemptions=2,
            recompute_tokens=123,
            cpu_swap_preemptions=0,
            cpu_swap_write_bytes=0,
            cpu_swap_read_bytes=0,
        )

        summary = manager.summary(1)

        self.assertEqual(summary["config"]["active_preemption_mode"], "recompute")
        self.assertEqual(summary["totals"]["active_recompute_preemptions"], 2)
        self.assertEqual(summary["totals"]["active_recompute_tokens"], 123)
        self.assertEqual(summary["totals"]["active_cpu_swap_write_bytes"], 0)

    def test_cpu_capacity_is_shared_across_colocated_instances(self):
        left = FakeScheduler(instance_id=0, node_id=0, cpu_mem=1000)
        right = FakeScheduler(instance_id=1, node_id=0, cpu_mem=1000)
        manager = AgenticKVManager(
            [left, right], AgenticKVConfig(policy="cpu"))
        left.memory.allocate(800, Device.CPU)
        self.assertEqual(manager._cpu_avail(right), 200)

    def test_tp8_kv_head_replication_matches_physical_layout(self):
        qwen_bytes = full_cluster_kv_bytes_per_token(
            "Qwen/Qwen3-30B-A3B-Instruct-2507", 16, tp_size=8)
        llama_bytes = full_cluster_kv_bytes_per_token(
            "meta-llama/Llama-3.1-8B", 16, tp_size=8)
        self.assertEqual(qwen_bytes, 196_608)
        self.assertEqual(llama_bytes, 131_072)

    def test_tiered_transition_and_ssd_restore(self):
        manager, scheduler = self.manager()
        request = FakeRequest(tokens=100)
        manager.on_tool_start(request, 0, 1_000_000_000)
        entry = manager.entries["s"]
        self.assertEqual(entry.location, KVLocation.HBM)
        self.assertEqual(scheduler.memory.npu_used, entry.per_rank_bytes)

        hbm_complete = (
            manager.config.hbm_ttl_ns
            + manager._cpu_transfer_ns(entry.per_rank_bytes, entry.total_bytes)
        )
        manager.advance(hbm_complete)
        self.assertEqual(entry.location, KVLocation.CPU)
        self.assertEqual(scheduler.memory.npu_used, 0)
        self.assertEqual(scheduler.memory.cpu_used, entry.total_bytes)
        self.assertIsNone(entry.migration_kind)

        ssd_complete = (
            hbm_complete + manager.config.cpu_ttl_ns
            + manager._ssd_write_ns(entry.total_bytes)
        )
        manager.advance(ssd_complete)
        self.assertEqual(entry.location, KVLocation.SSD)
        self.assertEqual(scheduler.memory.cpu_used, 0)
        self.assertEqual(manager.metrics.ssd_host_write_bytes, entry.total_bytes)

        prep = manager.prepare_request("s", 0, 96, 120, 1_000_000_000)
        self.assertEqual(prep.source, KVLocation.SSD)
        self.assertEqual(prep.hit_tokens, 96)
        self.assertGreater(prep.restore_ns, 0)
        self.assert_restore_components(prep)
        self.assertEqual(scheduler.memory.npu_used, prep.restored_bytes // 8)

    def test_idle_pause_and_resume_preserve_human_return_class(self):
        manager, _ = self.manager(demotion_mode="capacity-only")
        request = FakeRequest(tokens=100)
        manager.on_idle_start(
            request,
            100,
            1_000_100,
            return_gap_type="human",
            return_gap_source="request_ready_boundary",
        )
        prep = manager.prepare_request(
            session_id="s",
            instance_id=0,
            reuse_tokens=100,
            input_tokens=108,
            release_time_ns=1_000_100,
            return_gap_type="human",
            return_gap_source="request_ready_boundary",
            return_gap_ns=1_000_000,
        )

        self.assertEqual(manager.metrics.idle_pauses, 1)
        self.assertEqual(manager.metrics.human_return_pauses, 1)
        self.assertEqual(manager.metrics.tool_return_pauses, 0)
        self.assertGreater(prep.hit_tokens, 0)
        pause = next(
            event for event in manager.events
            if event.get("event") == "tool_pause"
        )
        resume = next(
            event for event in manager.events
            if event.get("event") == "resume"
        )
        self.assertEqual(pause["return_gap_type"], "human")
        self.assertEqual(resume["return_gap_type"], "human")
        self.assertEqual(resume["return_gap_ns"], 1_000_000)

    def test_mixed_prefill_batch_records_ready_source_composition(self):
        manager, scheduler = self.manager()
        scheduler.pd_type = "prefill"
        requests = [
            SimpleNamespace(
                agentic_kv_source="hbm",
                return_gap_type="tool",
                agentic_kv_async_decode_join=False,
                agentic_kv_restore_ns=0,
                is_prefill=lambda: True,
            ),
            SimpleNamespace(
                agentic_kv_source="ssd",
                return_gap_type="human",
                agentic_kv_async_decode_join=False,
                agentic_kv_restore_ns=0,
                is_prefill=lambda: True,
            ),
        ]
        batch = SimpleNamespace(
            batch_time=100,
            batch_id=7,
            requests=requests,
            agentic_source_counts={},
            agentic_return_gap_type_counts={},
            agentic_mixed_hbm_lower_tier=False,
        )

        manager.record_agentic_batch_schedule(scheduler, batch)
        manager.record_astra_workload_dispatch(scheduler, batch, 100)
        manager.record_agentic_batch_complete(scheduler, batch, 250)

        self.assertTrue(batch.agentic_mixed_hbm_lower_tier)
        self.assertEqual(
            manager.metrics.agentic_mixed_hbm_lower_tier_prefill_batches,
            1,
        )
        self.assertEqual(
            manager.metrics.agentic_mixed_hbm_lower_tier_batch_execution_ns,
            150,
        )
        scheduled = next(
            event for event in manager.events
            if event.get("event") == "agentic_batch_schedule"
        )
        self.assertEqual(scheduled["source_counts"], {"hbm": 1, "ssd": 1})
        self.assertEqual(
            scheduled["source_return_counts"],
            {"hbm": {"tool": 1}, "ssd": {"human": 1}},
        )
        self.assertFalse(scheduled["restore_barrier_inside_batch"])
        summary = manager.summary(250)
        self.assertEqual(
            summary["batch_composition"][
                "by_resume_source_and_return_gap_type"
            ],
            {"hbm": {"tool": 1}, "ssd": {"human": 1}},
        )

    def test_all_request_cross_tab_separates_residency_from_resume_source(self):
        manager, _ = self.manager()
        first = SimpleNamespace(
            id=0,
            session_id="s",
            sub_request_index=0,
            return_gap_type="session_start",
            agentic_kv_residency_at_return=None,
            agentic_kv_source=None,
            agentic_kv_async_decode_join=False,
            agentic_kv_restore_ns=0,
        )
        failed_ssd_restore = SimpleNamespace(
            id=1,
            session_id="s",
            sub_request_index=1,
            return_gap_type="human",
            agentic_kv_residency_at_return="ssd",
            agentic_kv_source="dropped",
            agentic_kv_async_decode_join=False,
            agentic_kv_restore_ns=0,
        )

        manager.record_agentic_request(first)
        manager.record_agentic_request(failed_ssd_restore)
        manager.record_agentic_request(failed_ssd_restore)
        classification = manager.summary(1)["request_classification"]

        self.assertEqual(classification["all_agentic_request_count"], 2)
        self.assertEqual(
            classification[
                "by_residency_at_return_and_return_gap_type"
            ],
            {"session_start": {"session_start": 1}, "ssd": {"human": 1}},
        )
        self.assertEqual(
            classification["by_resume_source_and_return_gap_type"],
            {
                "dropped": {"human": 1},
                "session_start": {"session_start": 1},
            },
        )

    def test_pd_launch_gate_records_actual_atomic_admission_once(self):
        base_manager, _ = self.manager()
        prefill = FakeScheduler(instance_id=0, node_id=0)
        decode = FakeScheduler(instance_id=1, node_id=0)
        manager = AgenticKVManager(
            [prefill, decode], base_manager.config)
        request = SimpleNamespace(
            id=7,
            session_id="s",
            pd_prefill_capacity_ready_ns=100,
            pd_decode_capacity_ready_ns=500,
            pd_prefill_full_per_rank_bytes=112,
            pd_prefill_initial_restored_per_rank_bytes=80,
            pd_prefill_reserved_per_rank_bytes=32,
            pd_decode_full_per_rank_bytes=112,
            agentic_kv_retained_per_rank_bytes=80,
            agentic_kv_restore_ready_time_ns=100,
            agentic_kv_source="ssd",
            agentic_kv_residency_at_return="ssd",
            return_gap_type="human",
            return_gap_source="request_ready_boundary",
            pd_prefill_admission_enqueued_ns=100,
            pd_chunk_admission_count=1,
            pd_chunk_admission_history=[{
                "prefill_current_per_rank_bytes": 80,
                "prefill_target_per_rank_bytes": 112,
                "decode_current_per_rank_bytes": 80,
                "decode_target_per_rank_bytes": 112,
            }],
        )

        manager.record_pd_prefill_admission(
            request, 0, 100, 100, 500, 100, 32,
            request.pd_chunk_admission_history[0])
        manager.record_pd_decode_receive_admission(
            request, 1, 100, 500, 500, 100, 32,
            request.pd_chunk_admission_history[0])
        manager.record_pd_launch_admission(request, 100, 500, 100)
        manager.record_pd_chunk_admission(request, {
            "request_id": 7,
            "active_prefill_recompute_generation": 0,
            "prefill_instance_id": 0,
            "decode_instance_id": 1,
            "computed_tokens": 80,
            "chunk_tokens": 20,
            "target_tokens": 100,
            "prefill_current_per_rank_bytes": 80,
            "decode_current_per_rank_bytes": 80,
            "prefill_target_per_rank_bytes": 112,
            "decode_target_per_rank_bytes": 112,
            "prefill_delta_per_rank_bytes": 32,
            "decode_delta_per_rank_bytes": 32,
            "prefill_unreserved_per_rank_bytes": 1024,
            "decode_unreserved_per_rank_bytes": 1024,
            "enqueued_ns": 100,
            "prefill_capacity_ready_ns": 100,
            "decode_capacity_ready_ns": 500,
            "admitted_ns": 500,
            "wait_ns": 400,
            "critical_wait_after_restore_ns": 400,
            "prefill_peak_hbm_used_per_rank_bytes": 112,
            "decode_peak_hbm_used_per_rank_bytes": 112,
        })

        prefill_event = next(
            event for event in manager.events
            if event["event"] == "pd_prefill_active_admission")
        self.assertEqual(prefill_event["capacity_ready_ns"], 100)
        self.assertEqual(prefill_event["admitted_ns"], 500)
        self.assertEqual(prefill_event["time_ns"], 500)
        decode_event = next(
            event for event in manager.events
            if event["event"] == "pd_decode_receive_admission")
        self.assertEqual(decode_event["capacity_ready_ns"], 500)
        self.assertEqual(decode_event["admitted_ns"], 500)
        launch_event = next(
            event for event in manager.events
            if event["event"] == "pd_launch_admission")
        self.assertEqual(launch_event["critical_wait_after_restore_ns"], 400)
        chunk_event = next(
            event for event in manager.events
            if event["event"] == "pd_chunk_admission")
        self.assertTrue(chunk_event["first_chunk"])
        self.assertEqual(chunk_event["prefill_delta_per_rank_bytes"], 32)
        self.assertEqual(chunk_event["decode_delta_per_rank_bytes"], 32)
        self.assertEqual(chunk_event["wait_ns"], 400)
        self.assertEqual(
            chunk_event["active_prefill_recompute_generation"], 0)
        self.assertIsNone(chunk_event["capacity_headroom_snapshot"])

        summary = manager.summary(500)
        self.assertEqual(summary["totals"]["pd_launch_admissions"], 1)
        self.assertEqual(summary["totals"]["pd_chunk_admissions"], 1)
        self.assertEqual(summary["totals"]["pd_chunk_admitted_tokens"], 20)
        self.assertEqual(summary["totals"]["pd_chunk_admission_wait_ns"], 400)
        self.assertEqual(
            summary["time_breakdown"][
                "aggregate_pd_launch_admission_critical_wait_ns"],
            400,
        )
        self.assertEqual(
            summary["totals"]["pd_prefill_admission_critical_wait_ns"],
            0,
        )
        self.assertEqual(
            summary["totals"]["pd_decode_receive_critical_wait_ns"],
            400,
        )

    def test_pd_chunk_snapshot_join_is_first_chunk_and_timestamp_exact(self):
        base_manager, _ = self.manager()
        manager = AgenticKVManager(
            [
                FakeScheduler(instance_id=0, node_id=0),
                FakeScheduler(instance_id=1, node_id=0),
            ],
            base_manager.config,
        )
        request = SimpleNamespace(
            id=8,
            session_id="snapshot-session",
            pd_prefill_admission_enqueued_ns=100,
            pd_prefill_capacity_ready_ns=100,
            pd_decode_capacity_ready_ns=150,
            pd_prefill_full_per_rank_bytes=112,
            pd_prefill_initial_restored_per_rank_bytes=80,
            pd_prefill_reserved_per_rank_bytes=32,
            pd_decode_full_per_rank_bytes=112,
            agentic_kv_retained_per_rank_bytes=80,
            pd_chunk_admission_count=1,
            pd_chunk_admission_history=[{
                "prefill_current_per_rank_bytes": 80,
                "prefill_target_per_rank_bytes": 112,
                "decode_current_per_rank_bytes": 80,
                "decode_target_per_rank_bytes": 112,
            }],
            agentic_kv_restore_ready_time_ns=100,
            agentic_kv_source="ssd",
            agentic_kv_residency_at_return="ssd",
            return_gap_type="tool",
            return_gap_source="request_ready_boundary",
        )
        manager.events.append({
            "time_ns": 100,
            "event": "queue_recompute_evaluate",
            "session_id": "snapshot-session",
            "capacity_headroom_snapshot": {
                "time_ns": 100,
                "feasible": True,
                "semantics": "causal_snapshot_not_reservation",
            },
        })

        first = {
            "request_id": 8,
            "active_prefill_recompute_generation": 0,
            "prefill_instance_id": 0,
            "decode_instance_id": 1,
            "computed_tokens": 80,
            "chunk_tokens": 20,
            "target_tokens": 100,
            "prefill_current_per_rank_bytes": 80,
            "decode_current_per_rank_bytes": 80,
            "prefill_target_per_rank_bytes": 112,
            "decode_target_per_rank_bytes": 112,
            "prefill_delta_per_rank_bytes": 32,
            "decode_delta_per_rank_bytes": 32,
            "prefill_unreserved_per_rank_bytes": 32,
            "decode_unreserved_per_rank_bytes": 32,
            "enqueued_ns": 100,
            "prefill_capacity_ready_ns": 100,
            "decode_capacity_ready_ns": 150,
            "admitted_ns": 150,
            "wait_ns": 50,
            "critical_wait_after_restore_ns": 50,
            "prefill_peak_hbm_used_per_rank_bytes": 112,
            "decode_peak_hbm_used_per_rank_bytes": 112,
        }
        manager.record_pd_prefill_admission(
            request, 0, 100, 100, 150, 100, 32, first)
        manager.record_pd_decode_receive_admission(
            request, 1, 100, 150, 150, 100, 32, first)
        manager.record_pd_launch_admission(request, 100, 150, 100)
        manager.record_pd_chunk_admission(request, first)
        request.pd_chunk_admission_count = 2
        later = dict(first)
        later.update({
            "computed_tokens": 100,
            "chunk_tokens": 10,
            "target_tokens": 110,
            "prefill_current_per_rank_bytes": 112,
            "decode_current_per_rank_bytes": 112,
            "prefill_delta_per_rank_bytes": 0,
            "decode_delta_per_rank_bytes": 0,
            "enqueued_ns": 200,
            "prefill_capacity_ready_ns": 200,
            "decode_capacity_ready_ns": 220,
            "admitted_ns": 220,
            "wait_ns": 20,
            "critical_wait_after_restore_ns": 20,
        })
        manager.record_pd_chunk_admission(request, later)

        chunk_events = [
            event for event in manager.events
            if event.get("event") == "pd_chunk_admission"
        ]
        self.assertTrue(chunk_events[0]["capacity_snapshot_feasible"])
        self.assertTrue(
            chunk_events[0]["snapshot_feasible_but_actual_waited"])
        self.assertIsNone(chunk_events[1]["capacity_headroom_snapshot"])
        self.assertEqual(
            manager.metrics.pd_chunk_snapshot_joined_admissions, 1)
        self.assertEqual(
            manager.metrics.pd_chunk_snapshot_feasible_wait_ns, 50)

        audit = manager._pd_chunk_accounting_audit()
        self.assertEqual(audit["snapshot_joined_first_chunks"], 1)
        self.assertEqual(audit["snapshot_feasible_waiting_first_chunks"], 1)

    def test_short_tool_wait_cancels_demotion_and_hits_hbm(self):
        manager, scheduler = self.manager(hbm_ttl_ms=50)
        manager.on_tool_start(FakeRequest(tokens=128), 0, 10_000_000)
        manager.advance(10_000_000)
        prep = manager.prepare_request("s", 0, 112, 144, 10_000_000)
        self.assertEqual(prep.source, KVLocation.HBM)
        self.assertEqual(prep.restore_ns, 0)
        self.assert_restore_components(prep)
        self.assertEqual(prep.hbm_admission_wait_ns, 0)
        self.assertEqual(manager.metrics.ssd_host_write_bytes, 0)
        self.assertEqual(scheduler.memory.npu_used, prep.restored_bytes // 8)

    def test_future_background_does_not_block_earlier_foreground_restore(self):
        manager, _ = self.manager(
            hbm_ttl_ms=100, ssd_read_bandwidth_gbps=0.001,
            ssd_read_latency_us=0)
        manager.on_tool_start(
            FakeRequest(session_id="future", tokens=128), 0, 1_000_000_000)
        source = IdleKVEntry(
            session_id="reading", instance_id=0, tokens=8,
            block_tokens=16, per_rank_bytes=1600, total_bytes=12800,
            location=KVLocation.SSD, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries["reading"] = source
        manager.ssd_records["reading"] = SSDRecord(
            tokens=8, block_tokens=16, bytes=12800,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 12800

        prep = manager.prepare_request("reading", 0, 8, 9, 10_000_000)
        media = next(
            event for event in manager.events
            if event.get("kind") == "ssd_to_cpu_stage")
        h2d = next(
            event for event in manager.events
            if event.get("kind") == "cpu_stage_to_hbm")
        self.assertEqual(media["start_ns"], 10_000_000)
        self.assertEqual(media["complete_ns"], h2d["time_ns"])
        self.assertEqual(h2d["complete_ns"], prep.ready_time_ns)

        manager.advance(100_000_000)
        background = next(
            event for event in manager.events
            if event.get("kind") == "hbm_to_cpu")
        self.assertGreaterEqual(
            background["start_ns"], h2d["complete_ns"])
        self.assertLess(prep.ready_time_ns, background["complete_ns"])

    def test_concurrent_ssd_restores_queue_on_shared_resources(self):
        manager, _ = self.manager(
            ssd_read_bandwidth_gbps=0.001,
            ssd_read_latency_us=0,
            swap_execution_mode="async-pre-admission",
        )
        for session_id in ("left", "right"):
            manager.entries[session_id] = IdleKVEntry(
                session_id=session_id, instance_id=0, tokens=8,
                block_tokens=16, per_rank_bytes=1600, total_bytes=12800,
                location=KVLocation.SSD, tier_since_ns=0, last_access_ns=0,
            )
            manager.ssd_records[session_id] = SSDRecord(
                tokens=8, block_tokens=16, bytes=12800,
                last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 25600

        left = manager.prepare_request("left", 0, 8, 9, 0)
        right = manager.prepare_request("right", 0, 8, 9, 0)
        self.assert_restore_components(left)
        self.assert_restore_components(right)
        self.assertEqual(left.queue_wait_ns, 0)
        self.assertEqual(right.queue_wait_ns, left.service_ns)
        self.assertEqual(right.restore_ns, left.service_ns + right.service_ns)
        self.assertEqual(
            manager.metrics.critical_restore_queue_wait_ns,
            left.service_ns,
        )
        foreground = [
            event for event in manager.events
            if event.get("event") == "migration_reserve"
            and event.get("foreground")
        ]
        self.assertEqual(
            [event["kind"] for event in foreground],
            [
                "ssd_to_cpu_stage", "cpu_stage_to_hbm",
                "ssd_to_cpu_stage", "cpu_stage_to_hbm",
            ],
        )
        for media, h2d in zip(foreground[::2], foreground[1::2]):
            self.assertEqual(media["complete_ns"], h2d["time_ns"])
            self.assertIn("ssd-pool:read", media["resources"])
            self.assertFalse(any(
                "pcie-copy" in resource
                for resource in media["resources"]))
            self.assertNotIn("ssd-pool:read", h2d["resources"])
            self.assertTrue(any(
                "pcie-copy" in resource
                for resource in h2d["resources"]))
        self.assertEqual(manager.metrics.ssd_to_cpu_stage_bytes, 25600)
        self.assertEqual(manager.metrics.cpu_stage_to_hbm_bytes, 25600)

        summary = manager.summary(right.ready_time_ns)
        breakdown = summary["time_breakdown"]
        self.assertEqual(summary["schema_version"], 20)
        self.assertEqual(
            breakdown[
                "aggregate_request_migration_hbm_admission_wait_ns"],
            0,
        )
        self.assertEqual(
            breakdown["aggregate_request_migration_stall_ns"],
            breakdown[
                "aggregate_request_migration_hbm_admission_wait_ns"]
            + breakdown["aggregate_request_migration_queue_wait_ns"]
            + breakdown["aggregate_request_migration_service_ns"],
        )
        self.assertEqual(
            breakdown["migration_critical_interval_union_ns"],
            right.ready_time_ns,
        )
        self.assertEqual(
            breakdown["migration_critical_interval_union_fraction_of_makespan"],
            1.0,
        )
        self.assertIsNone(
            breakdown["recompute_fraction_of_total_model_compute"])

    def test_pd_resume_moves_decode_hbm_to_prefill_before_release(self):
        prefill = FakeScheduler(instance_id=0, node_id=0)
        decode = FakeScheduler(instance_id=1, node_id=0)
        manager = AgenticKVManager(
            [prefill, decode],
            AgenticKVConfig(policy="preserve"),
        )
        manager.on_tool_start(
            FakeRequest(instance_id=1, tokens=100), 0, 1_000_000_000)
        source_bytes = decode.memory.npu_used

        prep = manager.prepare_request("s", 0, 96, 120, 10_000)

        self.assertEqual(prep.source, KVLocation.HBM)
        self.assertGreater(prep.restore_ns, 0)
        self.assert_restore_components(prep)
        self.assertEqual(prefill.memory.npu_used, prep.restored_bytes // 8)
        self.assertLess(decode.memory.npu_used, source_bytes)
        self.assertEqual(
            decode.memory.npu_used, prep.retained_per_rank_bytes)
        self.assertEqual(prep.retained_instance_id, decode.instance_id)
        self.assertEqual(
            manager.metrics.pd_hbm_to_hbm_bytes, prep.restored_bytes)
        self.assertEqual(
            manager.metrics.pd_cross_instance_restore_ns, prep.restore_ns)

        manager.advance(prep.ready_time_ns - 1)
        self.assertEqual(
            decode.memory.npu_used, prep.retained_per_rank_bytes)
        manager.advance(prep.ready_time_ns)
        self.assertEqual(
            decode.memory.npu_used, prep.retained_per_rank_bytes)
        self.assertEqual(prefill.memory.npu_used, prep.restored_bytes // 8)

    def test_pd_cpu_resume_restores_directly_to_prefill_hbm(self):
        prefill = FakeScheduler(instance_id=0, node_id=0)
        decode = FakeScheduler(instance_id=1, node_id=0)
        manager = AgenticKVManager(
            [prefill, decode],
            AgenticKVConfig(
                policy="cpu", pcie_bandwidth_gbps=50,
                cpu_bandwidth_gbps=200),
        )
        manager.on_tool_start(
            FakeRequest(instance_id=1, tokens=100), 0, 1_000_000_000)
        entry = manager.entries["s"]
        manager.advance(manager._cpu_transfer_ns(
            entry.per_rank_bytes, entry.total_bytes))
        self.assertEqual(entry.location, KVLocation.CPU)

        prep = manager.prepare_request("s", 0, 96, 120, 100_000)

        self.assertEqual(prep.source, KVLocation.CPU)
        self.assert_restore_components(prep)
        self.assertIsNone(prep.retained_instance_id)
        self.assertEqual(prep.retained_per_rank_bytes, 0)
        self.assertEqual(decode.memory.npu_used, 0)
        self.assertEqual(prefill.memory.npu_used, prep.restored_bytes // 8)
        self.assertGreater(decode.memory.cpu_used, 0)
        self.assertEqual(
            prep.service_ns,
            manager._cpu_transfer_ns(
                prep.restored_bytes // prefill.num_npus,
                prep.restored_bytes,
            ),
        )
        self.assertEqual(manager.metrics.pd_hbm_to_hbm_bytes, 0)
        self.assertEqual(manager.metrics.pd_cross_instance_restore_ns, 0)
        foreground = [
            event for event in manager.events
            if event.get("event") == "migration_reserve"
            and event.get("foreground")
        ]
        self.assertEqual(
            [event["kind"] for event in foreground], ["cpu_to_hbm"])
        self.assertTrue(any(
            resource.startswith("instance:0:pcie-copy")
            for resource in foreground[0]["resources"]
        ))
        self.assertFalse(any(
            resource.startswith("instance:1:pcie-copy")
            for resource in foreground[0]["resources"]
        ))
        manager.advance(prep.ready_time_ns)
        self.assertEqual(decode.memory.cpu_used, 0)
        self.assertEqual(decode.memory.npu_used, 0)

    def test_pd_retained_prefix_plus_suffix_has_exact_decode_ownership(self):
        prefill = FakeScheduler(instance_id=0, node_id=0)
        decode = FakeScheduler(instance_id=1, node_id=0)
        manager = AgenticKVManager(
            [prefill, decode], AgenticKVConfig(policy="preserve"))
        manager.on_tool_start(
            FakeRequest(instance_id=1, tokens=100), 0, 1_000_000_000)
        prep = manager.prepare_request("s", 0, 96, 120, 10_000)

        full_per_rank = prefill.memory.get_kv(128)
        prefill.memory.allocate(
            full_per_rank - prep.restored_bytes // 8, Device.NPU)
        prefill.memory.free(full_per_rank, Device.NPU)
        self.assertEqual(prefill.memory.npu_used, 0)

        request = Request(0, "model", 120, 130, 0, 0)
        request.num_computed_tokens = 120
        request.agentic_kv_retained_instance_id = prep.retained_instance_id
        request.agentic_kv_retained_per_rank_bytes = (
            prep.retained_per_rank_bytes)
        decode_scheduler = Scheduler.__new__(Scheduler)
        decode_scheduler.instance_id = decode.instance_id
        decode_scheduler.enable_prefix_caching = False
        decode_scheduler.request = []
        decode_scheduler.agentic_kv_manager = None
        decode_scheduler.memory = decode.memory

        decode_scheduler.add_decode(request)
        self.assertEqual(decode.memory.npu_used, full_per_rank)
        decode.memory.free(full_per_rank, Device.NPU)
        self.assertEqual(decode.memory.npu_used, 0)

    def test_measurement_censor_releases_prepared_cpu_restore_ownership(self):
        prefill = FakeScheduler(instance_id=0, node_id=0)
        decode = FakeScheduler(instance_id=1, node_id=0)
        manager = AgenticKVManager(
            [prefill, decode],
            AgenticKVConfig(
                policy="cpu",
                pcie_bandwidth_gbps=50,
                cpu_bandwidth_gbps=200,
            ),
        )
        manager.on_tool_start(
            FakeRequest(instance_id=1, tokens=100), 0, 1_000_000_000)
        entry = manager.entries["s"]
        manager.advance(manager._cpu_transfer_ns(
            entry.per_rank_bytes, entry.total_bytes))
        self.assertEqual(entry.location, KVLocation.CPU)

        prep = manager.prepare_request("s", 0, 96, 120, 100_000)
        request = Request(91, "model", 120, 1, 100_000, 0)
        request.session_id = "s"
        request.agentic_kv_hit_tokens = prep.hit_tokens
        request.agentic_kv_owner_instance_id = prefill.instance_id
        request.agentic_kv_retained_instance_id = prep.retained_instance_id
        request.agentic_kv_retained_per_rank_bytes = (
            prep.retained_per_rank_bytes)
        self.assertGreater(prefill.memory.npu_used, 0)
        self.assertGreater(decode.memory.cpu_used, 0)
        self.assertEqual(len(manager.pending_source_releases), 1)

        audit = manager.censor_prepared_request(request, 100_001)

        self.assertEqual(prefill.memory.npu_used, 0)
        self.assertEqual(decode.memory.cpu_used, 0)
        self.assertEqual(manager.pending_source_releases, [])
        self.assertIsNone(request.agentic_kv_owner_instance_id)
        self.assertIsNone(request.agentic_kv_retained_instance_id)
        self.assertEqual(request.agentic_kv_retained_per_rank_bytes, 0)
        self.assertEqual(audit["released_source"], "cpu")
        self.assertGreater(audit["released_owner_per_rank_bytes"], 0)
        self.assertTrue(
            manager.validate_measurement_censoring_drained()["passed"])
        with self.assertRaisesRegex(RuntimeError, "has no P-side owner"):
            manager.censor_prepared_request(request, 100_001)

    def test_measurement_censor_drain_audit_rejects_live_tier_ownership(self):
        manager, scheduler = self.manager(policy="preserve")
        manager.on_tool_start(
            FakeRequest(session_id="live", tokens=16),
            0, 1_000_000_000)

        with self.assertRaisesRegex(
                RuntimeError, "left live agentic-KV ownership"):
            manager.validate_measurement_censoring_drained()

        manager.end_session("live", now_ns=1)
        self.assertEqual(scheduler.memory.npu_used, 0)
        self.assertTrue(
            manager.validate_measurement_censoring_drained()["passed"])

    def test_measurement_censor_releases_prepared_pd_hbm_copies(self):
        prefill = FakeScheduler(instance_id=0, node_id=0)
        decode = FakeScheduler(instance_id=1, node_id=0)
        manager = AgenticKVManager(
            [prefill, decode], AgenticKVConfig(policy="preserve"))
        manager.on_tool_start(
            FakeRequest(instance_id=1, tokens=100), 0, 1_000_000_000)
        prep = manager.prepare_request("s", 0, 96, 120, 10_000)
        request = Request(92, "model", 120, 1, 10_000, 0)
        request.session_id = "s"
        request.agentic_kv_hit_tokens = prep.hit_tokens
        request.agentic_kv_owner_instance_id = prefill.instance_id
        request.agentic_kv_retained_instance_id = prep.retained_instance_id
        request.agentic_kv_retained_per_rank_bytes = (
            prep.retained_per_rank_bytes)
        self.assertGreater(prefill.memory.npu_used, 0)
        self.assertGreater(decode.memory.npu_used, 0)

        audit = manager.censor_prepared_request(request, 10_001)

        self.assertEqual(prefill.memory.npu_used, 0)
        self.assertEqual(decode.memory.npu_used, 0)
        self.assertGreater(audit["released_owner_per_rank_bytes"], 0)
        self.assertEqual(
            audit["released_retained_per_rank_bytes"],
            prep.retained_per_rank_bytes,
        )

    def test_measurement_censor_cancels_future_target_without_freeing_it(self):
        scheduler = FakeScheduler(
            instance_id=0, node_id=0, num_npus=1,
            bytes_per_token=100, npu_mem=1600)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(policy="tiered"))
        target = IdleKVEntry(
            session_id="pending", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=100, last_access_ns=100,
        )
        manager.pending_hbm_allocations.append(PendingHBMAllocation(
            entry=target, ready_ns=200))
        request = Request(93, "model", 17, 1, 100, 0)
        request.session_id = "pending"
        request.agentic_kv_hit_tokens = 16
        request.agentic_kv_owner_instance_id = scheduler.instance_id

        audit = manager.censor_prepared_request(request, 100)

        self.assertTrue(audit["cancelled_pending_target"])
        self.assertEqual(audit["released_owner_per_rank_bytes"], 0)
        self.assertEqual(manager.pending_hbm_allocations, [])
        self.assertEqual(scheduler.memory.npu_used, 0)

    def test_measurement_censor_refuses_scheduler_visible_pd_request(self):
        manager, _ = self.manager()
        request = Request(94, "model", 17, 1, 0, 0)
        request.session_id = "visible"
        request.pd_prefill_preallocated_per_rank_bytes = 1600

        with self.assertRaisesRegex(RuntimeError, "scheduler-visible"):
            manager.censor_prepared_request(request, 0)

    def test_pd_ssd_resume_restores_directly_to_prefill_hbm(self):
        prefill = FakeScheduler(instance_id=0, node_id=0)
        decode = FakeScheduler(instance_id=1, node_id=0)
        manager = AgenticKVManager(
            [prefill, decode],
            AgenticKVConfig(
                policy="tiered", hbm_ttl_ms=0, cpu_ttl_ms=0,
                pcie_bandwidth_gbps=50, cpu_bandwidth_gbps=200,
                ssd_read_bandwidth_gbps=100,
                ssd_write_bandwidth_gbps=80),
        )
        manager.on_tool_start(
            FakeRequest(instance_id=1, tokens=100), 0, 1_000_000_000)
        manager.advance(100_000_000)
        self.assertEqual(manager.entries["s"].location, KVLocation.SSD)

        prep = manager.prepare_request("s", 0, 96, 120, 100_000_000)

        self.assertEqual(prep.source, KVLocation.SSD)
        self.assert_restore_components(prep)
        self.assertIsNone(prep.retained_instance_id)
        self.assertEqual(prep.retained_per_rank_bytes, 0)
        self.assertEqual(decode.memory.npu_used, 0)
        self.assertEqual(prefill.memory.npu_used, prep.restored_bytes // 8)
        self.assertEqual(manager.metrics.pd_hbm_to_hbm_bytes, 0)
        foreground = [
            event for event in manager.events
            if event.get("event") == "migration_reserve"
            and event.get("foreground")
        ]
        self.assertEqual(
            [event["kind"] for event in foreground],
            ["ssd_to_cpu_stage", "cpu_stage_to_hbm"],
        )
        self.assertEqual(
            foreground[0]["complete_ns"], foreground[1]["start_ns"])
        self.assertEqual(
            prep.service_ns,
            manager._ssd_to_cpu_stage_ns(prep.restored_bytes)
            + manager._cpu_transfer_ns(
                prep.restored_bytes // prefill.num_npus,
                prep.restored_bytes,
            ),
        )
        manager.advance(prep.ready_time_ns)
        self.assertEqual(decode.memory.npu_used, 0)

    def test_pd_cross_node_hbm_resume_is_rejected_without_leaking_target(self):
        prefill = FakeScheduler(instance_id=0, node_id=0)
        decode = FakeScheduler(instance_id=1, node_id=1)
        manager = AgenticKVManager(
            [prefill, decode],
            AgenticKVConfig(policy="preserve"),
        )
        manager.on_tool_start(
            FakeRequest(instance_id=1, tokens=100), 0, 1_000_000_000)
        source_bytes = decode.memory.npu_used

        with self.assertRaisesRegex(RuntimeError, "Cross-node agentic D->P"):
            manager.prepare_request("s", 0, 96, 120, 10_000)

        self.assertEqual(prefill.memory.npu_used, 0)
        self.assertEqual(decode.memory.npu_used, source_bytes)
        self.assertIn("s", manager.entries)

    def test_pd_cross_node_lower_tier_restore_is_rejected_without_leak(self):
        for location in (KVLocation.CPU, KVLocation.SSD):
            with self.subTest(location=location.value):
                prefill = FakeScheduler(instance_id=0, node_id=0)
                decode = FakeScheduler(instance_id=1, node_id=1)
                manager = AgenticKVManager(
                    [prefill, decode],
                    AgenticKVConfig(policy="tiered"),
                )
                entry = IdleKVEntry(
                    session_id="s", instance_id=1, tokens=16,
                    block_tokens=16, per_rank_bytes=1600,
                    total_bytes=12800, location=location,
                    tier_since_ns=0, last_access_ns=0,
                )
                manager.entries["s"] = entry
                if location == KVLocation.CPU:
                    decode.memory.allocate(12800, Device.CPU)
                else:
                    manager.ssd_records["s"] = SSDRecord(
                        tokens=16, block_tokens=16, bytes=12800,
                        last_access_ns=0, accounted_until_ns=0,
                    )
                    manager.ssd_used_bytes = 12800

                with self.assertRaisesRegex(
                        RuntimeError, "Cross-node lower-tier KV restore"):
                    manager.prepare_request("s", 0, 16, 17, 100)

                self.assertEqual(prefill.memory.npu_used, 0)
                self.assertEqual(decode.memory.npu_used, 0)
                self.assertIn("s", manager.entries)
                foreground = [
                    event for event in manager.events
                    if event.get("event") == "migration_reserve"
                    and event.get("foreground")
                ]
                self.assertEqual(foreground, [])

        missing_manager, _ = self.manager()
        missing = missing_manager.prepare_request(
            "missing", 0, 0, 120, 0,
            request_id=8, sub_request_index=1)
        missing_resume = next(
            event for event in missing_manager.events
            if event.get("event") == "resume")
        self.assertEqual(missing.source, KVLocation.DROPPED)
        self.assertEqual(missing.hit_tokens, 0)
        self.assertEqual(missing.recompute_tokens, 0)
        self.assertEqual(missing_resume["source"], "dropped")
        self.assertEqual(missing_manager.metrics.dropped_misses, 0)
        self.assertEqual(missing_manager.metrics.recompute_tokens, 0)
        self.assertEqual(missing_manager.metrics.hbf_eligible_resumes, 0)
        self.assertEqual(missing_manager.metrics.hbf_eligible_restore_bytes, 0)

    def test_full_prefix_hit_still_executes_final_prompt_token(self):
        manager, _ = self.manager(hbm_ttl_ms=50)
        manager.on_tool_start(FakeRequest(tokens=128), 0, 10_000_000)
        prep = manager.prepare_request("s", 0, 128, 128, 10_000_000)
        self.assertEqual(prep.hit_tokens, 127)
        self.assertEqual(prep.recompute_tokens, 1)
        self.assertEqual(manager.metrics.policy_avoidable_recompute_tokens, 0)

    def test_full_prefix_block_boundary_caps_operational_hit_at_input_minus_one(self):
        manager, _ = self.manager(hbm_ttl_ms=50, block_size=16)
        manager.on_tool_start(FakeRequest(tokens=16), 0, 10_000_000)

        prep = manager.prepare_request("s", 0, 16, 16, 10_000_000)

        self.assertEqual(prep.hit_tokens, 15)
        self.assertEqual(prep.recompute_tokens, 1)
        self.assertEqual(manager.metrics.cache_hit_tokens, 15)
        self.assertEqual(manager.metrics.policy_avoidable_recompute_tokens, 0)
        self.assertEqual(manager.metrics.capacity_induced_recompute_tokens, 0)

    def test_capacity_drop_excludes_mandatory_final_token_from_policy_miss(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=1600)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(
                policy="hbm_lru_recompute", block_size=16))
        victim = IdleKVEntry(
            session_id="victim", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[victim.session_id] = victim
        scheduler.memory.allocate(1600, Device.NPU)
        newcomer = IdleKVEntry(
            session_id="new", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=1, last_access_ns=1,
        )
        self.assertEqual(manager._reserve_hbm(newcomer, 1), 1)

        prep = manager.prepare_request("victim", 0, 16, 16, 2)

        self.assertEqual(prep.hit_tokens, 0)
        self.assertEqual(prep.recompute_tokens, 16)
        self.assertEqual(manager.metrics.policy_avoidable_recompute_tokens, 15)
        self.assertEqual(manager.metrics.capacity_induced_recompute_tokens, 15)

    def test_recompute_baseline_does_not_reserve_memory(self):
        manager, scheduler = self.manager(policy="recompute")
        manager.on_tool_start(FakeRequest(tokens=100), 0, 1_000_000)
        self.assertEqual(scheduler.memory.npu_used, 0)
        prep = manager.prepare_request("s", 0, 80, 120, 1_000_000)
        self.assertEqual(prep.source, KVLocation.DROPPED)
        self.assertEqual(prep.hit_tokens, 0)
        self.assertEqual(prep.recompute_tokens, 80)

    def test_zero_reuse_releases_cpu_residence(self):
        manager, scheduler = self.manager(hbm_ttl_ms=0, cpu_ttl_ms=1000)
        manager.on_tool_start(FakeRequest(tokens=100), 0, 1_000_000_000)
        entry = manager.entries["s"]
        cpu_complete = manager._cpu_transfer_ns(
            entry.per_rank_bytes, entry.total_bytes)
        manager.advance(cpu_complete)
        self.assertEqual(entry.location, KVLocation.CPU)
        self.assertGreater(scheduler.memory.cpu_used, 0)

        prep = manager.prepare_request("s", 0, 0, 120, 1_000_000_000)
        self.assertEqual(prep.hit_tokens, 0)
        self.assertEqual(scheduler.memory.cpu_used, 0)
        self.assertNotIn("s", manager.entries)

    def test_zero_reuse_preserves_raw_tier_without_hit_miss_or_restore(self):
        cases = []

        hbm_manager, _ = self.manager(
            hbm_ttl_ms=2000, cpu_ttl_ms=2000)
        hbm_manager.on_tool_start(
            FakeRequest(tokens=100), 0, 1_000_000_000)
        cases.append(("hbm", hbm_manager, 1_000_000_000))

        cpu_manager, _ = self.manager(
            hbm_ttl_ms=0, cpu_ttl_ms=2000)
        cpu_manager.on_tool_start(
            FakeRequest(tokens=100), 0, 1_000_000_000)
        cpu_entry = cpu_manager.entries["s"]
        cpu_manager.advance(cpu_manager._cpu_transfer_ns(
            cpu_entry.per_rank_bytes, cpu_entry.total_bytes))
        self.assertEqual(cpu_entry.location, KVLocation.CPU)
        cases.append(("cpu", cpu_manager, 1_000_000_000))

        ssd_manager, _ = self.manager(hbm_ttl_ms=0, cpu_ttl_ms=0)
        ssd_manager.on_tool_start(
            FakeRequest(tokens=100), 0, 1_000_000_000)
        ssd_manager.advance(100_000_000)
        self.assertEqual(
            ssd_manager.entries["s"].location, KVLocation.SSD)
        cases.append(("ssd", ssd_manager, 100_000_000))

        for source, manager, now_ns in cases:
            with self.subTest(source=source):
                prep = manager.prepare_request(
                    "s", 0, 0, 120, now_ns,
                    request_id=7, sub_request_index=1)
                resume = next(
                    event for event in reversed(manager.events)
                    if event.get("event") == "resume")
                self.assertEqual(prep.source.value, source)
                self.assertEqual(prep.residency_at_return.value, source)
                self.assertEqual(prep.hit_tokens, 0)
                self.assertEqual(prep.recompute_tokens, 0)
                self.assertEqual(prep.restore_ns, 0)
                self.assertEqual(resume["source"], source)
                self.assertEqual(resume["residency_at_return"], source)
                self.assertEqual(resume["hit_tokens"], 0)
                self.assertEqual(resume["recompute_tokens"], 0)
                self.assertEqual(manager.metrics.hbm_hits, 0)
                self.assertEqual(manager.metrics.cpu_hits, 0)
                self.assertEqual(manager.metrics.ssd_hits, 0)
                self.assertEqual(manager.metrics.cache_hit_tokens, 0)
                self.assertEqual(manager.metrics.recompute_tokens, 0)
                self.assertEqual(manager.metrics.dropped_misses, 0)
                self.assertEqual(manager.metrics.hbf_eligible_resumes, 0)
                self.assertEqual(
                    manager.metrics.hbf_eligible_restore_bytes, 0)
                self.assertEqual(
                    manager.metrics.hbf_gross_stall_upper_bound_ns, 0)
                self.assertEqual(
                    manager.metrics.hbf_dropped_recompute_tokens, 0)
                foreground = [
                    event for event in manager.events
                    if (event.get("event") == "migration_reserve"
                        and event.get("foreground"))
                ]
                self.assertEqual(foreground, [])

    def test_cpu_restore_releases_source_only_at_dma_completion(self):
        manager, scheduler = self.manager(hbm_ttl_ms=0, cpu_ttl_ms=1000)
        manager.on_tool_start(FakeRequest(tokens=100), 0, 1_000_000_000)
        entry = manager.entries["s"]
        cpu_complete = manager._cpu_transfer_ns(
            entry.per_rank_bytes, entry.total_bytes)
        manager.advance(cpu_complete)

        prep = manager.prepare_request("s", 0, 96, 120, cpu_complete)
        self.assertGreater(scheduler.memory.cpu_used, 0)
        manager.advance(prep.ready_time_ns - 1)
        self.assertGreater(scheduler.memory.cpu_used, 0)
        manager.advance(prep.ready_time_ns)
        self.assertEqual(scheduler.memory.cpu_used, 0)

    def test_incremental_write_charges_same_block_append(self):
        manager, scheduler = self.manager(hbm_ttl_ms=0, cpu_ttl_ms=0)
        manager.on_tool_start(FakeRequest(tokens=15), 0, 1_000_000_000)
        manager.advance(1_000_000_000)
        first_write = manager.metrics.ssd_host_write_bytes
        prep = manager.prepare_request("s", 0, 15, 20, 1_000_000_000)
        manager.advance(prep.ready_time_ns)
        scheduler.memory.free(prep.restored_bytes // 8, Device.NPU)

        manager.on_tool_start(
            FakeRequest(tokens=16, prefix_reuse_tokens=15),
            prep.ready_time_ns, 2_000_000_000)
        entry = manager.entries["s"]
        expected_append = entry.total_bytes // entry.block_tokens
        self.assertEqual(manager._ssd_write_bytes(entry), expected_append)
        manager.advance(2_000_000_000)
        self.assertEqual(
            manager.metrics.ssd_host_write_bytes - first_write,
            expected_append,
        )

    def test_incremental_write_pins_base_until_atomic_commit(self):
        manager, scheduler = self.manager(
            ssd_ttl_ms=0.000001,
            ssd_write_bandwidth_gbps=0.000001,
            ssd_write_latency_us=0,
        )
        manager.ssd_records["s"] = SSDRecord(
            tokens=8, block_tokens=8, bytes=6400,
            last_access_ns=0, accounted_until_ns=0,
        )
        manager.ssd_used_bytes = 6400
        entry = IdleKVEntry(
            session_id="s", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=12800,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
            incremental_base_tokens=8, next_use_ns=10**18,
        )
        scheduler.memory.allocate(entry.total_bytes, Device.CPU)
        manager.entries["s"] = entry

        expected_delta = 6400
        manager._schedule_entry_migration(entry, "cpu_to_ssd", 0)
        complete_ns = entry.migration_complete_ns
        self.assertIsNotNone(complete_ns)
        self.assertGreater(
            manager.ssd_records["s"].pinned_until_ns, complete_ns)
        manager.advance(complete_ns)

        self.assertEqual(manager.metrics.ssd_host_write_bytes, expected_delta)
        self.assertEqual(manager.ssd_records["s"].tokens, 16)
        self.assertEqual(manager.ssd_records["s"].bytes, 12800)

    def test_completed_write_counts_endurance_when_capacity_commit_fails(self):
        manager, scheduler = self.manager(
            ssd_capacity_gb=0.000000001,
            ssd_write_latency_us=0,
        )
        entry = IdleKVEntry(
            session_id="s", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=12800,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
            next_use_ns=10**18,
        )
        scheduler.memory.allocate(entry.total_bytes, Device.CPU)
        manager.entries["s"] = entry

        manager._schedule_entry_migration(entry, "cpu_to_ssd", 0)
        manager.advance(entry.migration_complete_ns)

        self.assertEqual(manager.metrics.ssd_host_write_bytes, 12800)
        self.assertEqual(entry.location, KVLocation.DROPPED)
        self.assertNotIn("s", manager.ssd_records)
        self.assertEqual(scheduler.memory.cpu_used, 0)

    def test_partial_prefix_forces_full_ssd_rewrite(self):
        manager, scheduler = self.manager(hbm_ttl_ms=0, cpu_ttl_ms=0)
        manager.on_tool_start(FakeRequest(tokens=15), 0, 1_000_000_000)
        manager.advance(1_000_000_000)
        prep = manager.prepare_request("s", 0, 10, 20, 1_000_000_000)
        manager.advance(prep.ready_time_ns)
        scheduler.memory.free(prep.restored_bytes // 8, Device.NPU)

        manager.on_tool_start(
            FakeRequest(tokens=16, prefix_reuse_tokens=10),
            prep.ready_time_ns, 2_000_000_000)
        entry = manager.entries["s"]
        self.assertEqual(manager._ssd_write_bytes(entry), entry.total_bytes)

    def test_partial_turn_invalidates_transitive_ssd_append_base(self):
        manager, scheduler = self.manager(hbm_ttl_ms=0, cpu_ttl_ms=0)
        manager.on_tool_start(FakeRequest(tokens=100), 0, 1_000_000_000)
        manager.advance(1_000_000_000)
        self.assertIn("s", manager.ssd_records)

        # B shares only half of durable A. The SSD read may finish, but A is
        # no longer a valid append base for B or any later turn.
        prep = manager.prepare_request("s", 0, 50, 120, 1_000_000_000)
        self.assertIn("s", manager.ssd_records)
        manager.advance(prep.ready_time_ns)
        self.assertNotIn("s", manager.ssd_records)
        scheduler.memory.free(prep.restored_bytes // 8, Device.NPU)

        manager.on_tool_start(
            FakeRequest(tokens=120, prefix_reuse_tokens=50),
            prep.ready_time_ns, 2_000_000_000)
        entry = manager.entries["s"]
        self.assertEqual(manager._ssd_write_bytes(entry), entry.total_bytes)

    def test_fast_forward_processes_capacity_events_chronologically(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, cpu_mem=1600)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="cpu", pcie_bandwidth_gbps=1e9,
                cpu_bandwidth_gbps=1e9, cpu_transfer_latency_us=0),
        )
        cpu_source = IdleKVEntry(
            session_id="old", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
        )
        scheduler.memory.allocate(1600, Device.CPU)
        manager.pending_source_releases.append(PendingSourceRelease(
            entry=cpu_source, ready_ns=100))

        hbm_entry = IdleKVEntry(
            session_id="new", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
            next_use_ns=1000,
        )
        scheduler.memory.allocate(1600, Device.NPU)
        manager.entries["new"] = hbm_entry
        manager.advance(100)

        ordered = [
            (event["event"], event["time_ns"])
            for event in manager.events
            if event["event"] in {"drop", "restore_source_release"}
        ]
        self.assertEqual(
            ordered,
            [("drop", 1), ("restore_source_release", 100)],
        )
        self.assertEqual(scheduler.memory.cpu_used, 0)

    def test_ssd_capacity_does_not_evict_an_inflight_restore(self):
        manager, _ = self.manager(
            ssd_num_devices=1, ssd_capacity_gb=0.000001)
        source = IdleKVEntry(
            session_id="reading", instance_id=0, tokens=8,
            block_tokens=8, per_rank_bytes=100, total_bytes=800,
            location=KVLocation.SSD, tier_since_ns=0, last_access_ns=0,
        )
        manager.ssd_records["reading"] = SSDRecord(
            tokens=8, block_tokens=8, bytes=800,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 800
        manager.pending_source_releases.append(PendingSourceRelease(
            entry=source, ready_ns=100))

        self.assertFalse(manager._ensure_ssd_capacity("new", 500, 50))
        self.assertIn("reading", manager.ssd_records)

    def test_exact_keep_on_read_restore_pins_ssd_until_dma_completion(self):
        manager, _ = self.manager(
            ssd_num_devices=1, ssd_capacity_gb=0.000013,
            ssd_read_bandwidth_gbps=0.000001,
            ssd_read_latency_us=0,
            ssd_ttl_ms=1,
            keep_ssd_copy_on_read=True,
        )
        source = IdleKVEntry(
            session_id="reading", instance_id=0, tokens=8,
            block_tokens=16, per_rank_bytes=1600, total_bytes=12800,
            location=KVLocation.SSD, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries["reading"] = source
        manager.ssd_records["reading"] = SSDRecord(
            tokens=8, block_tokens=16, bytes=12800,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 12800

        prep = manager.prepare_request("reading", 0, 8, 9, 0)
        self.assertGreater(prep.ready_time_ns, 0)
        self.assertTrue(any(
            pending.entry.session_id == "reading"
            for pending in manager.pending_source_releases
        ))
        self.assertFalse(manager._ensure_ssd_capacity(
            "new", 500, prep.ready_time_ns - 1))
        self.assertIn("reading", manager.ssd_records)

        manager.advance(prep.ready_time_ns)
        self.assertFalse(any(
            pending.entry.session_id == "reading"
            for pending in manager.pending_source_releases
        ))
        restore_and_ttl = [
            event for event in manager.events
            if event["event"] in {"restore_source_release", "ssd_record_ttl"}
        ]
        self.assertEqual(
            [event["time_ns"] for event in restore_and_ttl],
            [prep.ready_time_ns, prep.ready_time_ns],
        )
        self.assertEqual(
            manager.metrics.ssd_byte_ns,
            12800 * prep.ready_time_ns,
        )
        self.assertTrue(manager._ensure_ssd_capacity(
            "new", 500, prep.ready_time_ns))
        self.assertNotIn("reading", manager.ssd_records)

    def test_keep_on_read_shadow_record_expires_independently(self):
        manager, scheduler = self.manager(
            hbm_ttl_ms=10_000, ssd_ttl_ms=1,
            keep_ssd_copy_on_read=True)
        source = IdleKVEntry(
            session_id="s", instance_id=0, tokens=8,
            block_tokens=16, per_rank_bytes=1600, total_bytes=12800,
            location=KVLocation.SSD, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries["s"] = source
        manager.ssd_records["s"] = SSDRecord(
            tokens=8, block_tokens=16, bytes=12800,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 12800

        prep = manager.prepare_request("s", 0, 8, 9, 0)
        manager.advance(prep.ready_time_ns)
        self.assertIn("s", manager.ssd_records)
        scheduler.memory.free(prep.restored_bytes // scheduler.num_npus, Device.NPU)

        # A newer idle HBM copy must not keep the old durable append base
        # alive past the SSD record's own TTL.
        manager.on_tool_start(
            FakeRequest(tokens=10, prefix_reuse_tokens=8),
            prep.ready_time_ns + 100_000,
            prep.ready_time_ns + 20_000_000,
        )
        manager.advance(2_000_000)
        self.assertEqual(manager.entries["s"].location, KVLocation.HBM)
        self.assertNotIn("s", manager.ssd_records)
        self.assertEqual(manager.metrics.ttl_drops, 1)
        self.assertTrue(any(
            event["event"] == "ssd_record_ttl"
            for event in manager.events
        ))

    def test_failed_cpu_restore_releases_cpu_residence(self):
        scheduler = FakeScheduler(npu_mem=20_000)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="tiered", hbm_ttl_ms=0, cpu_ttl_ms=1000,
                ssd_num_devices=8),
        )
        manager.on_tool_start(FakeRequest(tokens=100), 0, 1_000_000_000)
        entry = manager.entries["s"]
        cpu_complete = manager._cpu_transfer_ns(
            entry.per_rank_bytes, entry.total_bytes)
        manager.advance(cpu_complete)
        self.assertEqual(entry.location, KVLocation.CPU)

        scheduler.memory.allocate(scheduler.memory.npu_mem, Device.NPU)
        prep = manager.prepare_request("s", 0, 96, 120, 1_000_000_000)
        self.assertEqual(prep.source, KVLocation.DROPPED)
        self.assertEqual(prep.recompute_tokens, 96)
        self.assertEqual(scheduler.memory.cpu_used, 0)

    def test_temporary_hbm_full_defers_lower_tier_restore_without_drop(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100,
            npu_mem=1600, cpu_mem=1600)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(
                policy="tiered", demotion_mode="capacity-only",
                pcie_bandwidth_gbps=1e9,
                cpu_bandwidth_gbps=1e9,
                cpu_transfer_latency_us=0,
            ))
        source = IdleKVEntry(
            session_id="waiting", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[source.session_id] = source
        scheduler.memory.allocate(1600, Device.CPU)
        scheduler.memory.allocate(1600, Device.NPU)

        deferred = manager.prepare_request(
            "waiting", 0, 16, 17, 0,
            operation_time_ns=0,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
        )
        self.assertIsNone(deferred)
        self.assertIs(manager.entries["waiting"], source)
        self.assertIn("waiting", manager._pending_restore_sessions)
        self.assertEqual(scheduler.memory.cpu_used, 1600)
        self.assertEqual(manager.metrics.recompute_tokens, 0)

        scheduler.memory.free(1600, Device.NPU)
        prep = manager.prepare_request(
            "waiting", 0, 16, 17, 0,
            operation_time_ns=100,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
        )
        self.assertIsNotNone(prep)
        self.assertEqual(prep.source, KVLocation.CPU)
        self.assertEqual(prep.recompute_tokens, 0)
        self.assertEqual(prep.hbm_admission_wait_ns, 100)
        self.assertNotIn("waiting", manager._pending_restore_sessions)
        manager.advance(prep.ready_time_ns)
        self.assertEqual(scheduler.memory.cpu_used, 0)

    def test_censor_preserves_destination_hbm_admission_exposure(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100,
            npu_mem=1600, cpu_mem=1600)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(
                policy="tiered", demotion_mode="capacity-only"))
        source = IdleKVEntry(
            session_id="waiting", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[source.session_id] = source
        scheduler.memory.allocate(1600, Device.CPU)
        scheduler.memory.allocate(1600, Device.NPU)

        deferred = manager.prepare_request(
            "waiting", 0, 16, 17, 0,
            operation_time_ns=0,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
        )
        self.assertIsNone(deferred)

        scheduler.memory.free(1600, Device.NPU)
        audit = manager.censor_session("waiting", cutoff_ns=100)
        summary = manager.summary(100, measurement_censored=True)
        breakdown = summary["time_breakdown"]

        self.assertIsNone(audit["source_demotion_join"])
        self.assertEqual(
            audit["destination_admission"]["elapsed_ns"], 100)
        self.assertIsNone(audit["transient_dram_admission"])
        self.assertEqual(
            manager.metrics.critical_restore_hbm_admission_wait_ns, 0)
        self.assertEqual(
            breakdown["censored_destination_admission_count"], 1)
        self.assertEqual(
            breakdown[
                "censored_destination_admission_elapsed_ns_membership_sum"],
            100,
        )
        self.assertEqual(
            breakdown["migration_restore_exposure_union_ns"], 100)
        self.assertTrue(
            manager.validate_measurement_censoring_drained()["passed"])

    def test_async_decode_join_exposes_completed_destination_admission_wait(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100,
            npu_mem=1600, cpu_mem=1600)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(
                policy="tiered",
                demotion_mode="capacity-only",
                swap_execution_mode="async-decode-join",
                pcie_bandwidth_gbps=1e9,
                cpu_bandwidth_gbps=1e9,
                cpu_transfer_latency_us=0,
            ))
        source = IdleKVEntry(
            session_id="waiting", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[source.session_id] = source
        scheduler.memory.allocate(1600, Device.CPU)
        scheduler.memory.allocate(1600, Device.NPU)

        self.assertIsNone(manager.prepare_request(
            "waiting", 0, 16, 17, 0,
            operation_time_ns=0,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
        ))
        scheduler.memory.free(1600, Device.NPU)
        prep = manager.prepare_request(
            "waiting", 0, 16, 17, 0,
            operation_time_ns=100,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
        )

        self.assertIsNotNone(prep)
        self.assertEqual(prep.hbm_admission_wait_ns, 100)
        self.assertEqual(prep.service_ns, 1)
        self.assertEqual(prep.restore_ns, 101)
        self.assertEqual(manager._critical_restore_intervals, [(0, 101)])
        self.assertEqual(manager._async_owner_barrier_intervals, [])
        summary = manager.summary(200)
        breakdown = summary["time_breakdown"]
        self.assertEqual(
            breakdown["migration_restore_exposure_union_ns"], 100)
        self.assertEqual(
            breakdown["aggregate_request_migration_stall_ns"], 100)
        self.assertEqual(
            summary["asynchronous_restore"][
                "aggregate_pre_admission_wait_ns"],
            100,
        )

    def test_queue_recompute_freezes_restore_across_capacity_retry(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100,
            npu_mem=1600, cpu_mem=1600)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(
                policy="tiered_queue_recompute",
                demotion_mode="capacity-only",
                swap_execution_mode="async-pre-admission",
                queue_recompute_wait_service_ratio=1.0,
                pcie_bandwidth_gbps=1e9,
                cpu_bandwidth_gbps=1e9,
                cpu_transfer_latency_us=0,
            ))
        source = IdleKVEntry(
            session_id="waiting", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[source.session_id] = source
        scheduler.memory.allocate(1600, Device.CPU)
        scheduler.memory.allocate(1600, Device.NPU)
        manager._reserve_transfer(
            kind="cpu_to_hbm", arrival_ns=0, service_ns=1000,
            source_instance_id=0, target_instance_id=0,
            num_bytes=1, background=True, session_id="other",
        )

        self.assertIsNone(manager.prepare_request(
            "waiting", 0, 16, 17, 0,
            operation_time_ns=0,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
        ))
        scheduler.memory.free(1600, Device.NPU)
        prep = manager.prepare_request(
            "waiting", 0, 16, 17, 0,
            operation_time_ns=100,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
        )

        self.assertIsNotNone(prep)
        self.assertEqual(prep.source, KVLocation.CPU)
        self.assertEqual(prep.recompute_tokens, 0)
        self.assertEqual(manager.metrics.queue_recompute_drop_decisions, 0)
        self.assertEqual(
            manager.metrics.queue_recompute_evaluation_attempts, 1)
        self.assertEqual(prep.owner_gate_ns, 1001)
        self.assertEqual(prep.prepare_boundary_wait_ns, 0)
        self.assertEqual(prep.hbm_admission_wait_ns, 100)
        self.assertEqual(prep.queue_wait_ns, 900)
        self.assertEqual(prep.service_ns, 1)
        self.assertEqual(prep.restore_ns, 1001)
        self.assertTrue(any(
            event.get("event")
            == "queue_recompute_restore_commitment_reused"
            for event in manager.events
        ))
        summary = manager.summary(1100)
        breakdown = summary["time_breakdown"]
        self.assertEqual(
            summary["queue_recompute_policy"][
                "pending_restore_commitments"],
            0,
        )
        self.assertEqual(
            breakdown["aggregate_prepare_boundary_wait_ns"], 0)
        self.assertEqual(
            breakdown[
                "aggregate_request_migration_hbm_admission_wait_ns"],
            100,
        )
        self.assertEqual(
            breakdown["migration_restore_exposure_union_ns"], 1001)

    def test_temporary_transient_dram_full_defers_ssd_restore_without_drop(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100,
            npu_mem=1600, cpu_mem=1600)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(
                policy="hbm_ssd_direct",
                pcie_bandwidth_gbps=1e9,
                cpu_bandwidth_gbps=1e9,
                cpu_transfer_latency_us=0,
                ssd_read_bandwidth_gbps=1e9,
                ssd_read_latency_us=0,
            ))
        source = IdleKVEntry(
            session_id="waiting", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.SSD, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[source.session_id] = source
        manager.ssd_records[source.session_id] = SSDRecord(
            tokens=16, block_tokens=16, bytes=1600,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 1600
        scheduler.memory.allocate(1600, Device.CPU)

        deferred = manager.prepare_request(
            "waiting", 0, 16, 17, 0,
            operation_time_ns=0,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
        )

        self.assertIsNone(deferred)
        self.assertIs(manager.entries["waiting"], source)
        self.assertIn("waiting", manager.ssd_records)
        self.assertIn("waiting", manager._pending_restore_sessions)
        self.assertEqual(scheduler.memory.npu_used, 0)
        self.assertEqual(manager.metrics.recompute_tokens, 0)

        scheduler.memory.free(1600, Device.CPU)
        prep = manager.prepare_request(
            "waiting", 0, 16, 17, 0,
            operation_time_ns=100,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
        )

        self.assertIsNotNone(prep)
        self.assertEqual(prep.source, KVLocation.SSD)
        self.assertEqual(prep.recompute_tokens, 0)
        self.assertEqual(prep.hbm_admission_wait_ns, 0)
        self.assertGreaterEqual(prep.queue_wait_ns, 100)
        self.assertEqual(prep.target_hbm_ready_time_ns, 0)
        self.assertEqual(prep.transient_dram_capacity_wait_ns, 100)
        self.assertEqual(
            manager.metrics.transient_dram_capacity_wait_ns, 100)
        self.assertNotIn("waiting", manager._pending_restore_sessions)

    def test_censor_preserves_transient_dram_admission_subset_once(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100,
            npu_mem=1600, cpu_mem=1600)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(
                policy="hbm_ssd_direct",
                pcie_bandwidth_gbps=1e9,
                cpu_bandwidth_gbps=1e9,
                cpu_transfer_latency_us=0,
                ssd_read_bandwidth_gbps=1e9,
                ssd_read_latency_us=0,
            ))
        source = IdleKVEntry(
            session_id="waiting", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.SSD, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[source.session_id] = source
        manager.ssd_records[source.session_id] = SSDRecord(
            tokens=16, block_tokens=16, bytes=1600,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 1600
        scheduler.memory.allocate(1600, Device.CPU)

        deferred = manager.prepare_request(
            "waiting", 0, 16, 17, 0,
            operation_time_ns=0,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
        )
        self.assertIsNone(deferred)

        scheduler.memory.free(1600, Device.CPU)
        audit = manager.censor_session("waiting", cutoff_ns=100)
        summary = manager.summary(100, measurement_censored=True)
        breakdown = summary["time_breakdown"]

        self.assertEqual(
            audit["destination_admission"]["elapsed_ns"], 100)
        self.assertEqual(
            audit["transient_dram_admission"]["elapsed_ns"], 100)
        self.assertEqual(
            manager.metrics.transient_dram_capacity_wait_ns, 0)
        self.assertEqual(
            manager.metrics.critical_restore_hbm_admission_wait_ns, 0)
        self.assertEqual(
            breakdown["censored_destination_admission_count"], 1)
        self.assertEqual(
            breakdown["censored_transient_dram_admission_count"], 1)
        # Transient DRAM admission is a labeled subset of the continuous
        # destination wait, so it must not be arithmetically double-counted.
        self.assertEqual(
            breakdown["migration_restore_exposure_union_ns"], 100)
        self.assertTrue(
            manager.validate_measurement_censoring_drained()["passed"])

    def test_delayed_prepare_preserves_hbm_return_residency_across_demotion(self):
        for policy, expected_source in (
                ("tiered", KVLocation.CPU),
                ("hbm_ssd_direct", KVLocation.SSD)):
            with self.subTest(policy=policy):
                scheduler = FakeScheduler(
                    num_npus=1, bytes_per_token=100,
                    npu_mem=10_000, cpu_mem=10_000)
                manager = AgenticKVManager(
                    [scheduler], AgenticKVConfig(
                        policy=policy,
                        demotion_mode="capacity-only",
                        pcie_bandwidth_gbps=1e9,
                        cpu_bandwidth_gbps=1e9,
                        cpu_transfer_latency_us=0,
                        ssd_read_bandwidth_gbps=1e9,
                        ssd_write_bandwidth_gbps=1e9,
                        ssd_read_latency_us=0,
                        ssd_write_latency_us=0,
                    ))
                source = IdleKVEntry(
                    session_id="delayed", instance_id=0, tokens=16,
                    block_tokens=16, per_rank_bytes=1600,
                    total_bytes=1600, location=KVLocation.HBM,
                    tier_since_ns=0, last_access_ns=0,
                    next_use_ns=1_000_000,
                )
                manager.entries[source.session_id] = source
                scheduler.memory.allocate(1600, Device.NPU)

                observed = manager.snapshot_return_residency(
                    source.session_id, 10)
                self.assertEqual(observed, KVLocation.HBM)
                self.assertTrue(manager._schedule_hbm_demotion(
                    source, 20, reason="hbm_capacity"))
                operation_ns = int(source.migration_complete_ns)
                manager.advance(operation_ns)
                self.assertEqual(source.location, expected_source)

                prep = manager.prepare_request(
                    source.session_id, 0, 16, 17, 10,
                    operation_time_ns=operation_ns,
                    residency_at_return=observed,
                    defer_temporary_hbm_pressure=True,
                )

                self.assertIsNotNone(prep)
                self.assertEqual(prep.residency_at_return, KVLocation.HBM)
                self.assertEqual(prep.source, expected_source)
                self.assertEqual(prep.hbm_admission_wait_ns, 0)
                self.assertEqual(
                    prep.prepare_boundary_wait_ns, operation_ns - 10)
                manager.advance(prep.ready_time_ns)
                self.assertGreaterEqual(
                    source.tier_since_ns, operation_ns)
                resume = next(
                    event for event in manager.events
                    if event.get("event") == "resume"
                    and event.get("session_id") == source.session_id)
                self.assertEqual(resume["time_ns"], 10)
                self.assertEqual(
                    resume["operation_time_ns"], operation_ns)
                self.assertEqual(
                    resume["residency_at_return"], "hbm")
                self.assertEqual(resume["source"], expected_source.value)
                if expected_source == KVLocation.SSD:
                    self.assertEqual(
                        manager.ssd_records[source.session_id].last_access_ns,
                        operation_ns,
                    )

    def test_entry_missing_after_hbm_return_recomputes_without_fake_restore_drop(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=10_000)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(
                policy="hbm_lru_recompute",
                demotion_mode="capacity-only"))
        source = IdleKVEntry(
            session_id="disappeared", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[source.session_id] = source
        scheduler.memory.allocate(1600, Device.NPU)
        observed = manager.snapshot_return_residency(
            source.session_id, 10)
        manager._drop_entry(source, 20, "hbm_capacity")
        del manager.entries[source.session_id]

        prep = manager.prepare_request(
            source.session_id, 0, 16, 17, 10,
            operation_time_ns=20,
            residency_at_return=observed,
        )

        self.assertEqual(prep.residency_at_return, KVLocation.HBM)
        self.assertEqual(prep.source, KVLocation.DROPPED)
        self.assertEqual(prep.recompute_tokens, 16)
        self.assertEqual(manager.metrics.dropped_misses, 1)
        self.assertEqual(manager.metrics.recompute_tokens, 16)
        self.assertEqual(manager.metrics.policy_avoidable_recompute_tokens, 16)
        self.assertEqual(manager.metrics.hbf_eligible_resumes, 0)
        self.assertFalse(any(
            event.get("event") in {
                "hbm_capacity_restore_drop",
                "transient_dram_capacity_restore_drop",
            }
            for event in manager.events
        ))
        resume = next(
            event for event in manager.events
            if event.get("event") == "resume")
        self.assertEqual(resume["residency_at_return"], "hbm")
        self.assertEqual(resume["source"], "dropped")
        self.assertEqual(resume["operation_time_ns"], 20)

    def test_oversize_lower_tier_restore_remains_terminal_recompute(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100,
            npu_mem=800, cpu_mem=1600)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(
                policy="tiered", demotion_mode="capacity-only"))
        source = IdleKVEntry(
            session_id="oversize", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[source.session_id] = source
        scheduler.memory.allocate(1600, Device.CPU)

        prep = manager.prepare_request(
            "oversize", 0, 16, 17, 0,
            defer_temporary_hbm_pressure=True,
        )

        self.assertIsNotNone(prep)
        self.assertEqual(prep.source, KVLocation.DROPPED)
        self.assertEqual(prep.recompute_tokens, 16)
        self.assertEqual(scheduler.memory.cpu_used, 0)
        self.assertNotIn("oversize", manager._pending_restore_sessions)

    def test_tiered_cpu_capacity_forecasts_pinned_restore_release(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, cpu_mem=1600)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(
                policy="tiered", demotion_mode="capacity-only"))
        source = IdleKVEntry(
            session_id="source", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=0,
        )
        scheduler.memory.allocate(1600, Device.CPU)
        manager.pending_source_releases.append(PendingSourceRelease(
            entry=source, ready_ns=50))

        self.assertEqual(
            manager._cpu_capacity_time(scheduler, 1600, 0), 50)

    def test_active_reclaim_rejection_diagnostics_are_bounded(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=800)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(policy="tiered"))
        for index in range(200):
            self.assertIsNone(manager.claim_active_hbm_reclaim(
                0, 1600, index, owner_kind="pd", owner_id=1))

        diagnostics = manager.summary(200)[
            "active_hbm_reclaim_rejections"]
        self.assertEqual(diagnostics["total"], 200)
        self.assertEqual(diagnostics["by_reason"], {"kv_ceiling": 200})
        self.assertEqual(diagnostics["sampled"], 128)
        self.assertEqual(diagnostics["suppressed"], 72)
        self.assertEqual(sum(
            event["event"] == "active_hbm_reclaim_rejected"
            for event in manager.events), 128)

    def test_ssd_ttl_releases_capacity_and_forces_recompute(self):
        manager, _ = self.manager(hbm_ttl_ms=0, cpu_ttl_ms=0, ssd_ttl_ms=1)
        manager.on_tool_start(FakeRequest(tokens=64), 0, 1_000_000_000)
        manager.advance(1_000_000_000)
        self.assertEqual(manager.entries["s"].location, KVLocation.DROPPED)
        self.assertEqual(manager.ssd_used_bytes, 0)
        self.assertNotIn("s", manager.ssd_records)

    def test_metrics_are_direct_endurance_input_with_balanced_bytes(self):
        manager, _ = self.manager()
        manager.metrics.ssd_host_write_bytes = 803
        manager.metrics.ssd_host_read_bytes = 83
        summary = manager.summary(2_000_000_000, "trace.jsonl", "run-a")
        self.assertEqual(summary["schema_version"], 20)
        stats = RunWriteStats.from_dict(summary)
        self.assertEqual(stats.run_id, "run-a")
        self.assertEqual(stats.trace_period_seconds, 2.0)
        self.assertEqual(sum(device.host_write_bytes for device in stats.devices), 803)
        self.assertLessEqual(
            max(device.host_write_bytes for device in stats.devices)
            - min(device.host_write_bytes for device in stats.devices),
            1,
        )

    def test_cancelled_ssd_write_counts_partial_endurance_bytes(self):
        manager, _ = self.manager(
            ssd_write_latency_us=0, cpu_transfer_latency_us=0)
        reservation = manager._reserve_transfer(
            kind="cpu_to_ssd",
            arrival_ns=0,
            service_ns=1000,
            source_instance_id=0,
            target_instance_id=None,
            num_bytes=1000,
            background=True,
            deadline_ns=500,
            session_id="partial",
        )
        self.assertFalse(reservation.completed)
        self.assertEqual(manager.metrics.ssd_host_write_bytes, 500)
        self.assertEqual(
            manager.metrics.ssd_cancelled_host_write_bytes, 500)

    def test_cancelled_background_copy_counts_as_transfer_activity(self):
        manager, _ = self.manager()
        reservation = manager._reserve_transfer(
            kind="hbm_to_cpu",
            arrival_ns=0,
            service_ns=100,
            source_instance_id=0,
            target_instance_id=None,
            num_bytes=100,
            background=True,
            deadline_ns=40,
            session_id="partial-activity",
            register_sync_barrier=False,
        )
        self.assertFalse(reservation.completed)
        self.assertEqual(reservation.active_ns_before_cancel, 40)

        activity = manager.summary(100)["observed_load_activity"]
        self.assertEqual(activity["global_transfer_execution_ns"], 40)
        self.assertEqual(
            activity["migration_only_no_model_execution_ns"], 40)
        self.assertEqual(activity["fully_quiescent_ns"], 60)
        self.assertIn("backlog, or Poisson", activity["scope_note"])
        self.assertNotIn("open-loop", activity["scope_note"])

    def test_cancelled_hbm_to_ssd_during_cpu_stage_issues_no_ssd_write(self):
        manager, _ = self.manager(
            ssd_write_latency_us=0, cpu_transfer_latency_us=0)
        reservation = manager._reserve_transfer(
            kind="hbm_to_ssd",
            arrival_ns=0,
            service_ns=2000,
            source_instance_id=0,
            target_instance_id=None,
            num_bytes=1000,
            background=True,
            deadline_ns=500,
            session_id="cpu-stage",
            ssd_write_phase_offset_ns=1000,
            ssd_write_phase_service_ns=1000,
        )
        self.assertFalse(reservation.completed)
        self.assertEqual(manager.metrics.ssd_host_write_bytes, 0)
        self.assertEqual(manager.metrics.ssd_cancelled_host_write_bytes, 0)

    def test_hbf_drop_opportunity_does_not_claim_unknown_time(self):
        manager, _ = self.manager()
        prep = manager.prepare_request("missing", 0, 64, 80, 0)
        self.assertEqual(prep.recompute_tokens, 64)
        self.assertEqual(prep.hbm_admission_wait_ns, 0)
        self.assert_restore_components(prep)
        summary = manager.summary(1)
        opportunity = summary["idle_capacity_opportunity"]
        self.assertEqual(opportunity["hbf_dropped_recompute_tokens"], 64)
        self.assertIsNone(opportunity["hbf_gross_total_stall_upper_bound_ns"])

    def test_hbm_capacity_does_not_drop_inflight_migration_source(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=1600)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(policy="tiered"))
        source = IdleKVEntry(
            session_id="moving", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
            migration_kind="hbm_to_cpu", migration_complete_ns=100,
        )
        scheduler.memory.allocate(1600, Device.NPU)
        manager.entries[source.session_id] = source
        candidate = IdleKVEntry(
            session_id="new", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=1, last_access_ns=1,
        )

        self.assertFalse(manager._allocate_hbm(candidate, 1))
        self.assertIn("moving", manager.entries)
        self.assertEqual(scheduler.memory.npu_used, 1600)

    def test_future_hbm_reservation_blocks_later_immediate_admission(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100,
            npu_mem=4000, cpu_mem=10_000)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="cpu",
                pcie_bandwidth_gbps=1e9,
                cpu_bandwidth_gbps=1e9,
                cpu_transfer_latency_us=0,
            ),
        )
        oldest = IdleKVEntry(
            session_id="oldest", instance_id=0, tokens=8,
            block_tokens=8, per_rank_bytes=800, total_bytes=800,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        recent = IdleKVEntry(
            session_id="recent", instance_id=0, tokens=8,
            block_tokens=8, per_rank_bytes=800, total_bytes=800,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=1,
        )
        manager.entries = {
            entry.session_id: entry for entry in (oldest, recent)
        }
        # The remaining 1,600 bytes represent active scheduler-owned KV. There
        # are 800 physically free bytes, but the first future reservation needs
        # that slack plus the oldest victim's eventual 800-byte release.
        scheduler.memory.allocate(3200, Device.NPU)
        first = IdleKVEntry(
            session_id="first", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=2,
        )
        first_ready_ns = manager._reserve_hbm(first, 0)
        self.assertIsNotNone(first_ready_ns)
        self.assertGreater(first_ready_ns, 0)

        second = IdleKVEntry(
            session_id="second", instance_id=0, tokens=8,
            block_tokens=8, per_rank_bytes=800, total_bytes=800,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=3,
        )
        second_ready_ns = manager._reserve_hbm(second, 0)

        # The physical 800-byte slack is already part of ``first``'s logical
        # reservation. ``second`` must wait for a separate victim instead of
        # allocating immediately and making ``first`` fail at commit.
        self.assertIsNotNone(second_ready_ns)
        self.assertGreater(second_ready_ns, first_ready_ns)
        self.assertEqual(scheduler.memory.npu_used, 3200)
        self.assertEqual(
            {pending.entry.session_id for pending
             in manager.pending_hbm_allocations},
            {"first", "second"},
        )

        manager.advance(first_ready_ns)
        self.assertEqual(scheduler.memory.npu_used, 4000)
        self.assertEqual(
            [pending.entry.session_id
             for pending in manager.pending_hbm_allocations],
            ["second"],
        )

        manager.advance(second_ready_ns)
        self.assertEqual(scheduler.memory.npu_used, 4000)
        self.assertEqual(manager.pending_hbm_allocations, [])
        commits = [
            event["session_id"] for event in manager.events
            if event.get("event") == "hbm_capacity_reservation_commit"
        ]
        self.assertEqual(commits, ["first", "second"])

    def test_active_hbm_reclaim_immediately_drops_deterministic_hbm_lru(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=3200)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(policy="hbm_lru_recompute"))
        alpha = IdleKVEntry(
            session_id="alpha", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=1,
        )
        zeta = IdleKVEntry(
            session_id="zeta", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=1,
        )
        manager.entries = {
            entry.session_id: entry for entry in (zeta, alpha)
        }
        scheduler.memory.allocate(3200, Device.NPU)

        ready_ns = manager.claim_active_hbm_reclaim(0, 1600, 10)
        jobs_after_admit = manager.metrics.transfer_jobs

        self.assertEqual(ready_ns, 10)
        self.assertEqual(alpha.location, KVLocation.DROPPED)
        self.assertEqual(zeta.location, KVLocation.HBM)
        self.assertEqual(scheduler.memory.npu_used, 1600)
        self.assertEqual(manager.metrics.hbm_capacity_drops, 1)
        self.assertEqual(manager.metrics.active_hbm_reclaim_admissions, 1)
        self.assertEqual(manager.metrics.active_hbm_reclaim_bytes, 1600)
        self.assertEqual(manager.metrics.active_hbm_reclaim_wait_ns, 0)

        # Same-size polling reuses the claim and cannot duplicate reclaim.
        self.assertEqual(
            manager.claim_active_hbm_reclaim(0, 1600, 10), 10)
        self.assertEqual(manager.metrics.transfer_jobs, jobs_after_admit)
        self.assertEqual(manager.metrics.active_hbm_reclaim_admissions, 1)

        claim = manager.consume_active_hbm_reclaim(0, 10)
        self.assertEqual(claim.per_rank_bytes, 1600)
        self.assertIsNone(manager.consume_active_hbm_reclaim(0, 10))
        summary = manager.summary(10)
        self.assertEqual(summary["schema_version"], 20)
        self.assertEqual(
            summary["active_hbm_reclaim"]["admission_count"], 1)
        self.assertEqual(
            summary["active_hbm_reclaim"]["outstanding_claims"], [])

    def test_active_hbm_reclaim_waits_for_atomic_tiered_demotion(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100,
            npu_mem=1600, cpu_mem=1600)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="tiered",
                pcie_bandwidth_gbps=1e9,
                cpu_bandwidth_gbps=1e9,
                cpu_transfer_latency_us=0,
            ),
        )
        victim = IdleKVEntry(
            session_id="victim", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[victim.session_id] = victim
        scheduler.memory.allocate(1600, Device.NPU)

        ready_ns = manager.claim_active_hbm_reclaim(0, 1600, 0)
        jobs_after_admit = manager.metrics.transfer_jobs

        self.assertGreater(ready_ns, 0)
        self.assertEqual(victim.location, KVLocation.HBM)
        self.assertEqual(victim.migration_kind, "hbm_to_cpu")
        self.assertEqual(scheduler.memory.npu_used, 1600)
        self.assertEqual(
            manager.claim_active_hbm_reclaim(0, 1600, 0), ready_ns)
        self.assertEqual(manager.metrics.transfer_jobs, jobs_after_admit)
        with self.assertRaisesRegex(RuntimeError, "one active HBM reclaim"):
            manager.claim_active_hbm_reclaim(0, 800, 0)

        manager.advance(ready_ns)
        self.assertEqual(victim.location, KVLocation.CPU)
        self.assertEqual(scheduler.memory.npu_used, 0)
        self.assertEqual(scheduler.memory.cpu_used, 1600)
        outstanding = manager.summary(ready_ns)[
            "active_hbm_reclaim"]["outstanding_claims"]
        self.assertEqual(len(outstanding), 1)
        self.assertEqual(outstanding[0]["ready_ns"], ready_ns)
        self.assertEqual(
            manager.metrics.active_hbm_reclaim_wait_ns, ready_ns)

        claim = manager.consume_active_hbm_reclaim(0, ready_ns)
        self.assertEqual(claim.ready_ns, ready_ns)
        self.assertEqual(
            manager.summary(ready_ns)[
                "active_hbm_reclaim"]["outstanding_claims"],
            [],
        )

    def test_active_hbm_reclaim_preserves_future_cpu_commit_capacity(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100,
            npu_mem=1600, cpu_mem=2400)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="tiered",
                demotion_mode="capacity-only",
                pcie_bandwidth_gbps=1e9,
                cpu_bandwidth_gbps=1e9,
                cpu_transfer_latency_us=1,
                ssd_write_bandwidth_gbps=1e9,
                ssd_write_latency_us=0,
                ssd_num_devices=1,
                ssd_write_mode="full",
            ),
        )
        cpu_old = IdleKVEntry(
            session_id="cpu-old", instance_id=0, tokens=12,
            block_tokens=12, per_rank_bytes=1200, total_bytes=1200,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=-1,
        )
        hbm_a = IdleKVEntry(
            session_id="hbm-a", instance_id=0, tokens=8,
            block_tokens=8, per_rank_bytes=800, total_bytes=800,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        hbm_b = IdleKVEntry(
            session_id="hbm-b", instance_id=0, tokens=8,
            block_tokens=8, per_rank_bytes=800, total_bytes=800,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=1,
        )
        manager.entries = {
            entry.session_id: entry
            for entry in (cpu_old, hbm_a, hbm_b)
        }
        scheduler.memory.allocate(cpu_old.total_bytes, Device.CPU)
        scheduler.memory.allocate(
            hbm_a.per_rank_bytes + hbm_b.per_rank_bytes, Device.NPU)

        ready_ns = manager.claim_active_hbm_reclaim(0, 1600, 0)

        # The first HBM victim has reserved 800 of the currently available CPU
        # bytes at its future commit.  The second cannot reuse that same slack:
        # it must wait for the CPU LRU to reach SSD before starting its copy.
        self.assertEqual(hbm_a.migration_kind, "hbm_to_cpu")
        self.assertEqual(cpu_old.migration_kind, "cpu_to_ssd")
        self.assertEqual(hbm_b.migration_kind, "hbm_to_cpu")
        self.assertGreater(
            hbm_b.migration_start_ns, hbm_a.migration_complete_ns)
        self.assertEqual(ready_ns, hbm_b.migration_complete_ns)

        manager.advance(ready_ns)
        claim = manager.consume_active_hbm_reclaim(0, ready_ns)

        self.assertEqual(claim.per_rank_bytes, 1600)
        self.assertEqual(scheduler.memory.npu_used, 0)
        self.assertEqual(scheduler.memory.cpu_used, 1600)
        self.assertEqual(cpu_old.location, KVLocation.SSD)
        self.assertEqual(hbm_a.location, KVLocation.CPU)
        self.assertEqual(hbm_b.location, KVLocation.CPU)

    def test_active_hbm_claim_blocks_idle_admission_until_cancelled(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=1600)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(policy="tiered"))

        self.assertEqual(
            manager.claim_active_hbm_reclaim(0, 1600, 7), 7)
        idle = IdleKVEntry(
            session_id="idle", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=7, last_access_ns=7,
        )
        self.assertIsNone(manager._reserve_hbm(idle, 7))
        self.assertEqual(scheduler.memory.npu_used, 0)

        cancelled = manager.cancel_active_hbm_reclaim(0, 8)
        self.assertEqual(cancelled.per_rank_bytes, 1600)
        self.assertIsNone(manager.cancel_active_hbm_reclaim(0, 8))
        self.assertEqual(manager._reserve_hbm(idle, 8), 8)
        self.assertEqual(scheduler.memory.npu_used, 1600)

    def test_hbm_unreserved_bytes_keep_future_idle_allocation_protected(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=4000)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(policy="tiered"))
        scheduler.memory.allocate(2400, Device.NPU)
        pending = IdleKVEntry(
            session_id="pending", instance_id=0, tokens=8,
            block_tokens=8, per_rank_bytes=800, total_bytes=800,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        manager.pending_hbm_allocations.append(PendingHBMAllocation(
            entry=pending, ready_ns=100))

        self.assertEqual(manager.hbm_unreserved_per_rank_bytes(0), 800)
        self.assertEqual(
            manager.claim_active_hbm_reclaim(0, 800, 0), 0)
        self.assertEqual(manager.hbm_unreserved_per_rank_bytes(0), 0)

        manager.consume_active_hbm_reclaim(0, 0)

        # Raw physical slack is 1,600 bytes, but half remains promised to the
        # future idle admission and cannot be stolen by a larger active batch.
        self.assertEqual(scheduler.memory.npu_mem - scheduler.memory.npu_used, 1600)
        self.assertEqual(manager.hbm_unreserved_per_rank_bytes(0), 800)

    def test_active_hbm_reclaim_return_joins_durable_direct_demotion(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=1600)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="hbm_ssd_direct",
                pcie_bandwidth_gbps=1e9,
                ssd_write_bandwidth_gbps=0.000001,
                ssd_write_latency_us=0,
                ssd_num_devices=1,
                ssd_write_mode="full",
            ),
        )
        victim = IdleKVEntry(
            session_id="victim", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
            next_use_ns=800_000_000,
        )
        manager.entries[victim.session_id] = victim
        scheduler.memory.allocate(1600, Device.NPU)

        ready_ns = manager.claim_active_hbm_reclaim(0, 1600, 0)
        jobs_after_admit = manager.metrics.transfer_jobs

        self.assertGreater(ready_ns, victim.next_use_ns)
        self.assertEqual(victim.location, KVLocation.HBM)
        self.assertEqual(victim.migration_kind, "hbm_to_ssd_direct")
        self.assertEqual(scheduler.memory.npu_used, 1600)
        self.assertEqual(manager.metrics.ssd_cancelled_host_write_bytes, 0)
        self.assertEqual(
            manager.metrics.direct_ssd_write_bytes,
            manager.metrics.ssd_host_write_bytes,
        )
        self.assertEqual(manager._ssd_reserved_bytes(), 1600)

        prep = manager.prepare_request(
            "victim", 0, 16, 17, victim.next_use_ns,
            operation_time_ns=victim.next_use_ns,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
            residency_at_return=KVLocation.HBM,
        )
        self.assertIsNone(prep)
        self.assertEqual(
            manager.pending_prepare_retry_time("victim"), ready_ns)
        self.assertEqual(
            manager.next_internal_event_time(victim.next_use_ns), ready_ns)
        self.assertEqual(victim.location, KVLocation.HBM)
        self.assertEqual(manager.metrics.hbm_capacity_drops, 0)
        self.assertEqual(manager.metrics.transfer_jobs, jobs_after_admit)

        retry_ns = (
            victim.next_use_ns
            + (ready_ns - victim.next_use_ns) // 2
        )
        retry = manager.prepare_request(
            "victim", 0, 16, 17, victim.next_use_ns,
            operation_time_ns=retry_ns,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
            residency_at_return=KVLocation.HBM,
        )
        self.assertIsNone(retry)
        self.assertEqual(
            manager.pending_prepare_retry_time("victim"), ready_ns)
        join_events = [
            event for event in manager.events
            if event.get("event") == "demotion_commit_join_deferred"
        ]
        self.assertEqual(len(join_events), 1)
        self.assertEqual(join_events[0]["time_ns"], victim.next_use_ns)

        manager.advance(ready_ns)
        self.assertEqual(victim.location, KVLocation.SSD)
        self.assertEqual(scheduler.memory.npu_used, 0)
        self.assertEqual(manager._ssd_reserved_bytes(), 0)
        capacity_deferred = manager.prepare_request(
            "victim", 0, 16, 17, victim.next_use_ns,
            operation_time_ns=ready_ns,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
            residency_at_return=KVLocation.HBM,
        )
        self.assertIsNone(capacity_deferred)
        join_wait_ns = ready_ns - victim.next_use_ns
        self.assertEqual(
            manager._demotion_join_wait("victim", ready_ns),
            join_wait_ns,
        )
        manager.cancel_active_hbm_reclaim(0, ready_ns)
        restored = manager.prepare_request(
            "victim", 0, 16, 17, victim.next_use_ns,
            operation_time_ns=ready_ns + 1,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
            residency_at_return=KVLocation.HBM,
        )
        self.assertEqual(restored.source, KVLocation.SSD)
        self.assertEqual(restored.recompute_tokens, 0)
        self.assertEqual(
            restored.source_demotion_join_wait_ns, join_wait_ns)
        self.assertEqual(restored.restore_issue_time_ns, ready_ns)
        self.assertEqual(
            restored.owner_gate_ns,
            join_wait_ns + restored.restore_ns,
        )
        self.assertEqual(
            restored.hbm_admission_wait_ns, 1)
        self.assertNotIn(
            "victim", manager._pending_demotion_join_windows)
        self.assertEqual(
            manager.metrics.source_demotion_join_wait_ns, join_wait_ns)
        self.assertEqual(
            manager.metrics.source_demotion_join_waiting_admissions, 1)
        self.assertEqual(manager.metrics.dropped_misses, 0)
        breakdown = manager.summary(
            restored.ready_time_ns)["time_breakdown"]
        self.assertEqual(
            breakdown["aggregate_source_demotion_join_wait_ns"],
            join_wait_ns,
        )
        self.assertEqual(
            breakdown["aggregate_request_migration_raw_elapsed_ns"],
            join_wait_ns + restored.restore_ns,
        )

    def test_censor_preserves_join_then_destination_admission_without_gap(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=1600)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="hbm_ssd_direct",
                pcie_bandwidth_gbps=1e9,
                ssd_write_bandwidth_gbps=0.000001,
                ssd_write_latency_us=0,
                ssd_num_devices=1,
                ssd_write_mode="full",
            ),
        )
        return_ns = 800_000_000
        victim = IdleKVEntry(
            session_id="victim", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
            next_use_ns=return_ns,
        )
        manager.entries[victim.session_id] = victim
        scheduler.memory.allocate(1600, Device.NPU)
        demotion_ready_ns = manager.claim_active_hbm_reclaim(0, 1600, 0)

        self.assertIsNone(manager.prepare_request(
            "victim", 0, 16, 17, return_ns,
            operation_time_ns=return_ns,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
            residency_at_return=KVLocation.HBM,
        ))
        manager.advance(demotion_ready_ns)
        self.assertIsNone(manager.prepare_request(
            "victim", 0, 16, 17, return_ns,
            operation_time_ns=demotion_ready_ns,
            prepare_boundary_wait_ns=0,
            defer_temporary_hbm_pressure=True,
            residency_at_return=KVLocation.HBM,
        ))

        cutoff_ns = demotion_ready_ns + 100
        manager.cancel_active_hbm_reclaim(0, cutoff_ns)
        audit = manager.censor_session("victim", cutoff_ns=cutoff_ns)
        breakdown = manager.summary(
            cutoff_ns, measurement_censored=True)["time_breakdown"]
        join_elapsed_ns = demotion_ready_ns - return_ns

        self.assertEqual(
            audit["source_demotion_join"]["elapsed_ns"],
            join_elapsed_ns,
        )
        self.assertEqual(
            audit["source_demotion_join"]["remaining_ns"], 0)
        self.assertEqual(
            audit["destination_admission"]["start_ns"],
            demotion_ready_ns,
        )
        self.assertEqual(
            audit["destination_admission"]["elapsed_ns"], 100)
        self.assertEqual(manager.metrics.source_demotion_join_wait_ns, 0)
        self.assertEqual(
            manager.metrics.critical_restore_hbm_admission_wait_ns, 0)
        self.assertEqual(
            breakdown["migration_restore_exposure_union_ns"],
            join_elapsed_ns + 100,
        )
        self.assertTrue(
            manager.validate_measurement_censoring_drained()["passed"])

    def test_demotion_join_retry_keeps_first_exposed_tail_identity(self):
        manager, _ = self.manager()

        self.assertTrue(manager._begin_demotion_join(
            "joined", 10, 100, "cpu_to_ssd"))
        self.assertFalse(manager._begin_demotion_join(
            "joined", 40, 100, "cpu_to_ssd"))
        with self.assertRaisesRegex(RuntimeError, "changed identity"):
            manager._begin_demotion_join(
                "joined", 40, 101, "cpu_to_ssd")
        with self.assertRaisesRegex(RuntimeError, "changed identity"):
            manager._begin_demotion_join(
                "joined", 40, 100, "hbm_to_cpu")
        with self.assertRaisesRegex(RuntimeError, "regressed time"):
            manager._begin_demotion_join(
                "joined", 9, 100, "cpu_to_ssd")

        self.assertEqual(manager._consume_demotion_join("joined", 100), 90)

    def test_active_hbm_reclaim_rejects_above_weight_adjusted_ceiling(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=1600)
        scheduler.memory.weight = 400
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(policy="hbm_lru_recompute"))
        victim = IdleKVEntry(
            session_id="victim", instance_id=0, tokens=12,
            block_tokens=12, per_rank_bytes=1200, total_bytes=1200,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[victim.session_id] = victim
        scheduler.memory.allocate(1600, Device.NPU)

        self.assertIsNone(
            manager.claim_active_hbm_reclaim(0, 1201, 0))
        self.assertEqual(victim.location, KVLocation.HBM)
        self.assertEqual(manager.metrics.hbm_capacity_drops, 0)

    def test_hbm_lru_recompute_drops_only_oldest_session_on_pressure(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=3200)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(policy="hbm_lru_recompute"))
        oldest = IdleKVEntry(
            session_id="oldest", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=1,
        )
        recent = IdleKVEntry(
            session_id="recent", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=2,
        )
        manager.entries = {
            entry.session_id: entry for entry in (oldest, recent)
        }
        scheduler.memory.allocate(3200, Device.NPU)
        newcomer = IdleKVEntry(
            session_id="new", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=3, last_access_ns=3,
        )

        ready_ns = manager._reserve_hbm(newcomer, 3)

        self.assertEqual(ready_ns, 3)
        self.assertEqual(oldest.location, KVLocation.DROPPED)
        self.assertEqual(oldest.drop_reason, "hbm_capacity")
        self.assertEqual(recent.location, KVLocation.HBM)
        self.assertEqual(scheduler.memory.npu_used, 3200)
        self.assertEqual(scheduler.memory.cpu_used, 0)
        self.assertEqual(manager.ssd_used_bytes, 0)
        self.assertEqual(manager.metrics.hbm_capacity_drops, 1)
        self.assertEqual(manager.metrics.transfer_jobs, 0)

        prep = manager.prepare_request("oldest", 0, 16, 17, 4)
        self.assertEqual(prep.source, KVLocation.DROPPED)
        self.assertEqual(prep.recompute_tokens, 16)
        self.assertEqual(
            manager.metrics.capacity_induced_recompute_tokens, 16)

    def test_hbm_lru_recompute_has_no_ttl_demotion(self):
        manager, scheduler = self.manager(
            policy="hbm_lru_recompute", hbm_ttl_ms=0)
        manager.on_tool_start(
            FakeRequest(tokens=16), 0, 1_000_000_000)

        manager.advance(999_999_999)

        self.assertEqual(manager.entries["s"].location, KVLocation.HBM)
        self.assertGreater(scheduler.memory.npu_used, 0)
        self.assertEqual(scheduler.memory.cpu_used, 0)
        self.assertEqual(manager.metrics.transfer_jobs, 0)

    def test_hbm_lru_recompute_public_lifecycle_admits_without_pending_copy(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=3200)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(policy="hbm_lru_recompute"))
        manager.on_tool_start(
            FakeRequest(session_id="a", tokens=16), 1, 1_000_000)
        manager.on_tool_start(
            FakeRequest(session_id="b", tokens=16), 2, 1_000_000)

        # Admit C as active work exactly as Scheduler does under full HBM:
        # reclaim the idle LRU, allocate C, then free C at completion before
        # the manager receives the completion callback.
        self.assertEqual(
            manager.claim_active_hbm_reclaim(0, 1600, 3), 3)
        manager.consume_active_hbm_reclaim(0, 3)
        scheduler.memory.allocate(1600, Device.NPU)
        scheduler.memory.free(1600, Device.NPU)
        manager.on_tool_start(
            FakeRequest(session_id="c", tokens=16), 3, 1_000_000)

        self.assertEqual(manager.entries["a"].location, KVLocation.DROPPED)
        self.assertEqual(manager.entries["b"].location, KVLocation.HBM)
        self.assertEqual(manager.entries["c"].location, KVLocation.HBM)
        self.assertEqual(manager.pending_hbm_allocations, [])
        self.assertIsNone(manager.entries["b"].migration_kind)
        self.assertIsNone(manager.entries["c"].migration_kind)
        self.assertEqual(scheduler.memory.npu_used, 3200)
        self.assertEqual(scheduler.memory.cpu_used, 0)
        self.assertEqual(manager.ssd_used_bytes, 0)
        self.assertEqual(manager.metrics.transfer_jobs, 0)

        prep = manager.prepare_request("a", 0, 16, 17, 4)
        self.assertEqual(prep.recompute_tokens, 16)

    def test_hbm_lru_recompute_does_not_drop_victims_for_unfit_newcomer(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=2400)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(policy="hbm_lru_recompute"))
        victim = IdleKVEntry(
            session_id="victim", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[victim.session_id] = victim
        scheduler.memory.allocate(2400, Device.NPU)
        newcomer = IdleKVEntry(
            session_id="new", instance_id=0, tokens=32,
            block_tokens=32, per_rank_bytes=3200, total_bytes=3200,
            location=KVLocation.HBM, tier_since_ns=1, last_access_ns=1,
        )

        self.assertIsNone(manager._reserve_hbm(newcomer, 1))
        self.assertEqual(victim.location, KVLocation.HBM)
        self.assertEqual(scheduler.memory.npu_used, 2400)
        self.assertEqual(manager.metrics.hbm_capacity_drops, 0)

    def test_preserve_alias_uses_hbm_lru_drop_without_cpu_transfer(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=1600)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(policy="preserve"))

        manager.on_tool_start(
            FakeRequest(session_id="a", tokens=16), 1, 1_000_000)
        self.assertEqual(
            manager.claim_active_hbm_reclaim(0, 1600, 2), 2)
        manager.consume_active_hbm_reclaim(0, 2)
        scheduler.memory.allocate(1600, Device.NPU)
        scheduler.memory.free(1600, Device.NPU)
        manager.on_tool_start(
            FakeRequest(session_id="b", tokens=16), 2, 1_000_000)

        self.assertEqual(manager.entries["a"].location, KVLocation.DROPPED)
        self.assertEqual(manager.entries["b"].location, KVLocation.HBM)
        self.assertEqual(scheduler.memory.cpu_used, 0)
        self.assertEqual(manager.ssd_used_bytes, 0)
        self.assertEqual(manager.metrics.transfer_jobs, 0)

    def test_direct_ssd_write_bypasses_dram_but_read_is_host_staged(self):
        scheduler = FakeScheduler(num_npus=8)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="hbm_ssd_direct",
                pcie_bandwidth_gbps=50,
                cpu_bandwidth_gbps=200,
                ssd_write_bandwidth_gbps=100,
                ssd_read_bandwidth_gbps=200,
                ssd_write_latency_us=20,
                ssd_read_latency_us=20,
            ),
        )
        per_rank = 100_000_000_000
        total = 800_000_000_000

        self.assertEqual(
            manager._direct_ssd_write_ns(per_rank, total),
            8_000_020_000,
        )
        self.assertEqual(
            manager._ssd_read_ns(total, per_rank),
            8_000_025_000,
        )
        write_resources = manager._transfer_resources(
            "hbm_to_ssd_direct", 0, 0)
        self.assertIn("ssd-pool:write", write_resources)
        self.assertEqual(
            sum("pcie-copy" in resource for resource in write_resources), 8)
        self.assertFalse(any(
            "dram" in resource for resource in write_resources))

        media_resources = manager._transfer_resources(
            "ssd_to_cpu_stage", 0, None)
        self.assertIn("ssd-pool:read", media_resources)
        self.assertIn("node:0:dram", media_resources)
        self.assertFalse(any(
            "pcie-copy" in resource for resource in media_resources))

        h2d_resources = manager._transfer_resources(
            "cpu_stage_to_hbm", 0, 0)
        self.assertNotIn("ssd-pool:read", h2d_resources)
        self.assertIn("node:0:dram", h2d_resources)
        self.assertEqual(
            sum("pcie-copy" in resource for resource in h2d_resources), 8)

    def test_hbm_ssd_direct_write_and_host_staged_restore_are_atomic(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=1600)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="hbm_ssd_direct",
                hbm_ttl_ms=0,
                ssd_ttl_ms=0,
                ssd_capacity_gb=0.00001,
                ssd_write_mode="full",
            ),
        )
        old = IdleKVEntry(
            session_id="old", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
            next_use_ns=1_000_000_000,
        )
        newcomer = IdleKVEntry(
            session_id="new", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=1, last_access_ns=1,
            next_use_ns=1_000_000_000,
        )
        manager.entries["old"] = old
        scheduler.memory.allocate(1600, Device.NPU)

        ready_ns = manager._reserve_hbm(newcomer, 1)

        self.assertGreater(ready_ns, 1)
        self.assertEqual(old.location, KVLocation.HBM)
        self.assertEqual(old.migration_kind, "hbm_to_ssd_direct")
        self.assertEqual(scheduler.memory.npu_used, 1600)
        self.assertEqual(manager.ssd_used_bytes, 0)
        self.assertEqual(manager._ssd_reserved_bytes(), 1600)

        manager.advance(ready_ns)
        self.assertEqual(old.location, KVLocation.SSD)
        self.assertEqual(scheduler.memory.npu_used, 1600)
        self.assertEqual(scheduler.memory.cpu_used, 0)
        self.assertEqual(manager.ssd_used_bytes, 1600)
        self.assertEqual(manager._ssd_reserved_bytes(), 0)
        self.assertEqual(manager.metrics.hbm_to_ssd_bytes, 1600)
        self.assertEqual(manager.metrics.direct_ssd_write_bytes, 1600)

        # Remove the synthetic foreground newcomer, then exercise a clean
        # capacity-only SSD restore of the old session.
        scheduler.memory.free(1600, Device.NPU)
        prep = manager.prepare_request("old", 0, 16, 17, ready_ns + 1)
        self.assertEqual(prep.source, KVLocation.SSD)
        self.assertEqual(
            prep.service_ns,
            manager._ssd_read_ns(1600, 1600),
        )
        self.assertEqual(manager.metrics.ssd_to_hbm_bytes, 1600)
        self.assertEqual(manager.metrics.direct_ssd_read_bytes, 0)
        self.assertEqual(scheduler.memory.cpu_used, 0)
        self.assertEqual(manager.metrics.hbm_to_cpu_bytes, 0)
        self.assertEqual(manager.metrics.cpu_to_hbm_bytes, 0)
        write_events = [
            event for event in manager.events
            if event.get("kind") == "hbm_to_ssd_direct"
        ]
        self.assertTrue(write_events)
        self.assertFalse(any(
            "dram" in resource
            for event in write_events
            for resource in event.get("resources", [])
        ))
        media_event = next(
            event for event in manager.events
            if event.get("kind") == "ssd_to_cpu_stage")
        h2d_event = next(
            event for event in manager.events
            if event.get("kind") == "cpu_stage_to_hbm")
        self.assertEqual(media_event["complete_ns"], h2d_event["time_ns"])
        self.assertIn("node:0:dram", media_event["resources"])
        self.assertIn("ssd-pool:read", media_event["resources"])
        self.assertFalse(any(
            "pcie-copy" in resource
            for resource in media_event["resources"]))
        self.assertIn("node:0:dram", h2d_event["resources"])
        self.assertNotIn("ssd-pool:read", h2d_event["resources"])
        self.assertTrue(any(
            "pcie-copy" in resource
            for resource in h2d_event["resources"]))
        self.assertEqual(manager.metrics.ssd_to_cpu_stage_bytes, 1600)
        self.assertEqual(manager.metrics.cpu_stage_to_hbm_bytes, 1600)

    def test_direct_restore_accounts_nonzero_hbm_admission_wait(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=1600)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="hbm_ssd_direct",
                pcie_bandwidth_gbps=1e9,
                ssd_read_bandwidth_gbps=0.001,
                ssd_write_bandwidth_gbps=0.001,
                ssd_read_latency_us=0,
                ssd_write_latency_us=0,
                ssd_capacity_gb=0.00001,
                ssd_num_devices=1,
                ssd_write_mode="full",
                swap_execution_mode="async-pre-admission",
            ),
        )
        source = IdleKVEntry(
            session_id="source", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.SSD, tier_since_ns=0, last_access_ns=0,
        )
        victim = IdleKVEntry(
            session_id="victim", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=1,
            next_use_ns=1_000_000_000,
        )
        manager.entries = {"source": source, "victim": victim}
        manager.ssd_records["source"] = SSDRecord(
            tokens=16, block_tokens=16, bytes=1600,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 1600
        scheduler.memory.allocate(1600, Device.NPU)

        prep = manager.prepare_request("source", 0, 16, 17, 10)

        self.assertEqual(prep.source, KVLocation.SSD)
        self.assertGreater(prep.hbm_admission_wait_ns, 0)
        self.assertEqual(prep.queue_wait_ns, 0)
        self.assert_restore_components(prep)
        self.assertEqual(
            manager.metrics.critical_restore_hbm_admission_wait_ns,
            prep.hbm_admission_wait_ns,
        )
        resume = next(
            event for event in manager.events
            if event.get("event") == "resume"
            and event.get("session_id") == "source")
        self.assertEqual(
            resume["hbm_admission_wait_ns"], prep.hbm_admission_wait_ns)

        manager.advance(prep.ready_time_ns)
        summary = manager.summary(
            prep.ready_time_ns, "trace.jsonl", "direct-pressure")
        breakdown = summary["time_breakdown"]
        self.assertEqual(summary["schema_version"], 20)
        self.assertEqual(
            breakdown["aggregate_request_migration_stall_ns"],
            breakdown[
                "aggregate_request_migration_hbm_admission_wait_ns"]
            + breakdown["aggregate_request_migration_queue_wait_ns"]
            + breakdown["aggregate_request_migration_service_ns"],
        )
        self.assertEqual(
            summary["totals"]["direct_ssd_write_bytes"],
            summary["totals"]["ssd_host_write_bytes"],
        )
        self.assertEqual(
            summary["totals"]["direct_ssd_read_bytes"],
            0,
        )
        self.assertEqual(
            summary["totals"]["ssd_to_cpu_stage_bytes"], 1600)
        self.assertEqual(
            summary["totals"]["cpu_stage_to_hbm_bytes"], 1600)
        self.assertEqual(summary["totals"]["hbm_to_cpu_bytes"], 0)
        self.assertEqual(summary["totals"]["cpu_to_hbm_bytes"], 0)
        self.assertEqual(summary["totals"]["cpu_byte_ns"], 0)
        self.assertTrue(any(
            "dram" in resource for resource in summary["resource_queues"]))
        self.assertLessEqual(
            summary["ssd"]["committed_reserved_bytes"],
            summary["ssd"]["capacity_bytes"],
        )
        self.assertLessEqual(
            summary["totals"]["peak_ssd_committed_reserved_bytes"],
            summary["ssd"]["capacity_bytes"],
        )
        stats = RunWriteStats.from_dict(summary)
        self.assertEqual(stats.run_id, "direct-pressure")
        self.assertEqual(stats.host_write_bytes, 1600)
        self.assertEqual(stats.host_read_bytes, 1600)

    def test_hbm_ssd_direct_ssd_capacity_uses_lru_then_recompute(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=1600)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="hbm_ssd_direct",
                ssd_capacity_gb=0.0000016,
                ssd_write_mode="full",
            ),
        )
        ssd_old = IdleKVEntry(
            session_id="ssd-old", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.SSD, tier_since_ns=0, last_access_ns=0,
        )
        hbm_old = IdleKVEntry(
            session_id="hbm-old", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=1,
            next_use_ns=1_000_000_000,
        )
        manager.entries = {
            ssd_old.session_id: ssd_old,
            hbm_old.session_id: hbm_old,
        }
        manager.ssd_records[ssd_old.session_id] = SSDRecord(
            tokens=16, block_tokens=16, bytes=1600,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 1600
        scheduler.memory.allocate(1600, Device.NPU)
        newcomer = IdleKVEntry(
            session_id="new", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=2, last_access_ns=2,
        )

        ready_ns = manager._reserve_hbm(newcomer, 2)
        manager.advance(ready_ns)

        self.assertEqual(ssd_old.location, KVLocation.DROPPED)
        self.assertEqual(hbm_old.location, KVLocation.SSD)
        self.assertEqual(manager.ssd_used_bytes, 1600)
        self.assertEqual(manager.metrics.ssd_capacity_evictions, 1)
        self.assertEqual(scheduler.memory.cpu_used, 0)
        self.assertEqual(manager.metrics.hbm_to_cpu_bytes, 0)
        self.assertEqual(manager.metrics.cpu_to_ssd_bytes, 0)
        prep = manager.prepare_request("ssd-old", 0, 16, 17, ready_ns)
        self.assertEqual(prep.recompute_tokens, 16)

    def test_hbm_ssd_direct_capacity_reservations_prevent_overbooking(self):
        manager, _ = self.manager(
            policy="hbm_ssd_direct",
            ssd_num_devices=1,
            ssd_capacity_gb=0.0000016,
        )
        left = IdleKVEntry(
            session_id="left", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=200, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        right = IdleKVEntry(
            session_id="right", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=200, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=1,
        )

        self.assertTrue(manager._reserve_direct_ssd_capacity(left, 0))
        self.assertFalse(manager._reserve_direct_ssd_capacity(right, 0))
        self.assertEqual(manager._ssd_reserved_bytes(), 1600)
        self.assertLessEqual(
            manager.ssd_used_bytes + manager._ssd_reserved_bytes(),
            manager.config.ssd_capacity_bytes,
        )
        manager._release_direct_ssd_capacity(left, 1)
        self.assertTrue(manager._reserve_direct_ssd_capacity(right, 1))

    def test_hbm_ssd_direct_oversized_object_preserves_existing_ssd_lru(self):
        manager, _ = self.manager(
            policy="hbm_ssd_direct",
            ssd_num_devices=1,
            ssd_capacity_gb=0.0000016,
        )
        manager.ssd_records["useful"] = SSDRecord(
            tokens=16, block_tokens=16, bytes=1600,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 1600
        oversized = IdleKVEntry(
            session_id="oversized", instance_id=0, tokens=32,
            block_tokens=32, per_rank_bytes=400, total_bytes=3200,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )

        self.assertFalse(manager._reserve_direct_ssd_capacity(oversized, 1))
        self.assertIn("useful", manager.ssd_records)
        self.assertEqual(manager.ssd_used_bytes, 1600)

    def test_hbm_ssd_direct_ignores_ttls_without_capacity_pressure(self):
        manager, scheduler = self.manager(
            policy="hbm_ssd_direct",
            hbm_ttl_ms=0,
            ssd_ttl_ms=0,
        )
        manager.on_tool_start(
            FakeRequest(session_id="hbm", tokens=16), 0, 10**15)
        ssd = IdleKVEntry(
            session_id="ssd", instance_id=0, tokens=8,
            block_tokens=8, per_rank_bytes=100, total_bytes=800,
            location=KVLocation.SSD, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries["ssd"] = ssd
        manager.ssd_records["ssd"] = SSDRecord(
            tokens=8, block_tokens=8, bytes=800,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 800

        manager.advance(10**12)

        self.assertEqual(manager.entries["hbm"].location, KVLocation.HBM)
        self.assertEqual(manager.entries["ssd"].location, KVLocation.SSD)
        self.assertIn("ssd", manager.ssd_records)
        self.assertEqual(scheduler.memory.cpu_used, 0)

    def test_cancelled_direct_ssd_write_charges_media_from_start(self):
        manager, _ = self.manager()
        reservation = manager._reserve_transfer(
            kind="hbm_to_ssd_direct",
            arrival_ns=0,
            service_ns=1000,
            source_instance_id=0,
            target_instance_id=None,
            num_bytes=1000,
            background=True,
            deadline_ns=500,
            session_id="direct-partial",
            ssd_write_phase_offset_ns=0,
            ssd_write_phase_service_ns=1000,
        )
        self.assertFalse(reservation.completed)
        self.assertEqual(manager.metrics.ssd_host_write_bytes, 500)
        self.assertEqual(manager.metrics.ssd_cancelled_host_write_bytes, 500)

    def test_durable_direct_capacity_write_uses_terminal_ssd_lru(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100, npu_mem=1600)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="hbm_ssd_direct",
                ssd_capacity_gb=0.0000016,
                ssd_write_bandwidth_gbps=0.000001,
                ssd_write_latency_us=0,
                pcie_bandwidth_gbps=1e9,
            ),
        )
        manager.ssd_records["useful"] = SSDRecord(
            tokens=16, block_tokens=16, bytes=1600,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 1600
        victim = IdleKVEntry(
            session_id="victim", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
            next_use_ns=800_000_000,
        )
        manager.entries["victim"] = victim
        scheduler.memory.allocate(1600, Device.NPU)

        self.assertTrue(manager._schedule_hbm_demotion(
            victim, 0, "hbm_capacity"))

        self.assertEqual(victim.location, KVLocation.HBM)
        self.assertEqual(victim.migration_kind, "hbm_to_ssd_direct")
        self.assertNotIn("useful", manager.ssd_records)
        self.assertEqual(manager.ssd_used_bytes, 0)
        self.assertEqual(manager._ssd_reserved_bytes(), 1600)
        self.assertEqual(manager.metrics.ssd_cancelled_host_write_bytes, 0)
        manager.advance(victim.migration_complete_ns)
        self.assertEqual(victim.location, KVLocation.SSD)
        self.assertEqual(manager.ssd_used_bytes, 1600)

    def test_direct_capacity_write_is_not_cancelled_during_fixed_latency(self):
        scheduler = FakeScheduler(num_npus=1)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="hbm_ssd_direct",
                pcie_bandwidth_gbps=1e9,
                ssd_write_bandwidth_gbps=1e9,
                ssd_write_latency_us=100,
            ),
        )
        entry = IdleKVEntry(
            session_id="latency", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
            next_use_ns=50_000,
        )
        manager.entries[entry.session_id] = entry
        scheduler.memory.allocate(1600, Device.NPU)

        self.assertTrue(manager._schedule_hbm_demotion(
            entry, 0, "hbm_capacity"))
        self.assertGreater(entry.migration_complete_ns, entry.next_use_ns)
        self.assertEqual(manager.metrics.ssd_host_write_bytes, 1600)
        self.assertEqual(manager.metrics.direct_ssd_write_bytes, 1600)
        self.assertEqual(manager.metrics.ssd_cancelled_host_write_bytes, 0)

    def test_pd_hbm_ssd_direct_restores_ssd_to_dram_to_prefill(self):
        source = FakeScheduler(instance_id=0, node_id=0)
        target = FakeScheduler(instance_id=1, node_id=0)
        manager = AgenticKVManager(
            [source, target], AgenticKVConfig(
                policy="hbm_ssd_direct",
                swap_execution_mode="sync-engine-barrier",
            ))
        entry = IdleKVEntry(
            session_id="s", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=12800,
            location=KVLocation.SSD, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries["s"] = entry
        manager.ssd_records["s"] = SSDRecord(
            tokens=16, block_tokens=16, bytes=12800,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 12800
        # Exercise queueing on both stages, not only their raw service time.
        manager._resource_busy_until["ssd-pool:read"] = 50_000
        manager._resource_busy_until["node:0:dram"] = 200_000

        prep = manager.prepare_request("s", 1, 16, 17, 1)

        self.assertEqual(prep.source, KVLocation.SSD)
        self.assert_restore_components(prep)
        self.assertIsNone(prep.retained_instance_id)
        self.assertEqual(prep.retained_per_rank_bytes, 0)
        self.assertEqual(source.memory.npu_used, 0)
        self.assertEqual(target.memory.npu_used, 1600)
        self.assertEqual(source.memory.cpu_used, 0)
        self.assertEqual(target.memory.cpu_used, 0)
        self.assertEqual(manager.metrics.ssd_to_hbm_bytes, 12800)
        self.assertEqual(manager.metrics.direct_ssd_read_bytes, 0)
        self.assertEqual(manager.metrics.ssd_host_read_bytes, 12800)
        self.assertEqual(manager.metrics.pd_hbm_to_hbm_bytes, 0)
        self.assertEqual(manager.metrics.pd_cross_instance_restore_ns, 0)
        self.assertEqual(manager.metrics.ssd_to_cpu_stage_bytes, 12800)
        self.assertEqual(manager.metrics.cpu_stage_to_hbm_bytes, 12800)
        foreground = [
            event for event in manager.events
            if event["event"] == "migration_reserve"
            and event["foreground"]
        ]
        self.assertEqual(
            [event["kind"] for event in foreground],
            ["ssd_to_cpu_stage", "cpu_stage_to_hbm"],
        )
        media, h2d = foreground
        self.assertEqual(media["complete_ns"], h2d["time_ns"])
        self.assertEqual(h2d["complete_ns"], prep.ready_time_ns)
        self.assertIn("ssd-pool:read", media["resources"])
        self.assertFalse(any(
            "pcie-copy" in resource for resource in media["resources"]))
        self.assertNotIn("ssd-pool:read", h2d["resources"])
        self.assertTrue(any(
            "pcie-copy" in resource for resource in h2d["resources"]))

        # The legacy synchronous sensitivity remains one conservative engine
        # barrier across the complete staged chain, while the physical queue
        # events above retain their disjoint resource sets.
        self.assertEqual(manager.metrics.sync_swap_barrier_jobs, 1)
        barrier_start_ns = media["time_ns"]
        self.assertIsNone(manager.synchronous_swap_blocked_until(
            0, barrier_start_ns))
        self.assertEqual(
            manager.synchronous_swap_blocked_until(1, barrier_start_ns),
            prep.ready_time_ns,
        )
        sync = manager.summary(prep.ready_time_ns)["synchronous_swap"]
        self.assertEqual(
            sync["exposed_engine_wait_ns_by_instance"],
            {"0": 0, "1": prep.ready_time_ns - barrier_start_ns},
        )
        self.assertEqual(
            sync["aggregate_exposed_engine_wait_ns"],
            prep.ready_time_ns - barrier_start_ns)
        self.assertGreater(prep.queue_wait_ns, 0)
        self.assertEqual(
            prep.queue_wait_ns,
            sum(event["queue_wait_ns"] for event in foreground),
        )
        self.assertEqual(
            prep.service_ns,
            sum(event["service_ns"] for event in foreground),
        )
        self.assertIn("s", manager.ssd_records)
        self.assertNotIn("s", manager.entries)

    def test_capacity_pressure_cascades_hbm_to_cpu_to_ssd_lru(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100,
            npu_mem=1600, cpu_mem=1600)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="tiered",
                demotion_mode="capacity-only",
                hbm_ttl_ms=0,
                cpu_ttl_ms=0,
                ssd_ttl_ms=0,
                pcie_bandwidth_gbps=1e9,
                cpu_bandwidth_gbps=1e9,
                cpu_transfer_latency_us=0,
                ssd_write_bandwidth_gbps=1e9,
                ssd_write_latency_us=0,
                ssd_capacity_gb=0.0000016,
            ),
        )
        ssd_old = IdleKVEntry(
            session_id="ssd-old", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.SSD, tier_since_ns=0, last_access_ns=0,
        )
        cpu_old = IdleKVEntry(
            session_id="cpu-old", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.CPU, tier_since_ns=0, last_access_ns=1,
        )
        hbm_old = IdleKVEntry(
            session_id="hbm-old", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=2,
        )
        manager.entries = {
            entry.session_id: entry
            for entry in (ssd_old, cpu_old, hbm_old)
        }
        manager.ssd_records[ssd_old.session_id] = SSDRecord(
            tokens=16, block_tokens=16, bytes=1600,
            last_access_ns=0, accounted_until_ns=0)
        manager.ssd_used_bytes = 1600
        scheduler.memory.allocate(1600, Device.CPU)
        scheduler.memory.allocate(1600, Device.NPU)

        newcomer = IdleKVEntry(
            session_id="new", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=3,
        )
        ready_ns = manager._reserve_hbm(newcomer, 0)
        self.assertIsNotNone(ready_ns)
        self.assertGreater(ready_ns, 0)
        # Atomicity: neither source tier is released at migration start.
        self.assertEqual(scheduler.memory.npu_used, 1600)
        self.assertEqual(scheduler.memory.cpu_used, 1600)

        manager.advance(ready_ns)
        self.assertEqual(hbm_old.location, KVLocation.CPU)
        self.assertEqual(cpu_old.location, KVLocation.SSD)
        self.assertEqual(ssd_old.location, KVLocation.DROPPED)
        self.assertEqual(scheduler.memory.npu_used, 1600)
        self.assertEqual(scheduler.memory.cpu_used, 1600)
        self.assertEqual(manager.ssd_used_bytes, 1600)
        self.assertEqual(manager.metrics.hbm_capacity_demotions, 1)
        self.assertEqual(manager.metrics.cpu_capacity_evictions, 1)
        self.assertEqual(manager.metrics.ssd_capacity_evictions, 1)
        self.assertEqual(manager.metrics.capacity_drops, 1)
        self.assertEqual(manager.metrics.cpu_capacity_bypasses, 0)

        prep = manager.prepare_request("ssd-old", 0, 16, 17, ready_ns)
        self.assertEqual(prep.source, KVLocation.DROPPED)
        self.assertEqual(prep.recompute_tokens, 16)
        self.assertEqual(
            manager.metrics.capacity_induced_recompute_tokens, 16)

    def test_hbm_pressure_bypasses_cpu_only_when_object_cannot_fit(self):
        scheduler = FakeScheduler(
            num_npus=1, bytes_per_token=100,
            npu_mem=1600, cpu_mem=800)
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="tiered",
                pcie_bandwidth_gbps=1e9,
                cpu_bandwidth_gbps=1e9,
                cpu_transfer_latency_us=0,
                ssd_write_bandwidth_gbps=1e9,
                ssd_write_latency_us=0,
            ),
        )
        victim = IdleKVEntry(
            session_id="victim", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=0,
        )
        manager.entries[victim.session_id] = victim
        scheduler.memory.allocate(1600, Device.NPU)
        newcomer = IdleKVEntry(
            session_id="new", instance_id=0, tokens=16,
            block_tokens=16, per_rank_bytes=1600, total_bytes=1600,
            location=KVLocation.HBM, tier_since_ns=0, last_access_ns=1,
        )

        ready_ns = manager._reserve_hbm(newcomer, 0)
        self.assertIsNotNone(ready_ns)
        manager.advance(ready_ns)
        self.assertEqual(victim.location, KVLocation.SSD)
        self.assertEqual(manager.metrics.hbm_capacity_demotions, 1)
        self.assertEqual(manager.metrics.cpu_capacity_bypasses, 1)
        self.assertEqual(manager.metrics.hbm_capacity_drops, 0)

    def test_durable_ssd_residence_is_counted_during_active_turn(self):
        manager, scheduler = self.manager(hbm_ttl_ms=0, cpu_ttl_ms=0)
        manager.on_tool_start(FakeRequest(tokens=64), 0, 1_000_000_000)
        manager.advance(100_000_000)
        entry = manager.entries["s"]
        self.assertEqual(entry.location, KVLocation.SSD)
        commit_ns = manager.ssd_records["s"].accounted_until_ns
        prep = manager.prepare_request("s", 0, 64, 80, 100_000_000)
        manager.advance(prep.ready_time_ns)
        scheduler.memory.free(prep.restored_bytes // 8, Device.NPU)
        end_ns = prep.ready_time_ns + 1_000_000
        summary = manager.summary(end_ns)
        expected = manager.ssd_records["s"].bytes * (end_ns - commit_ns)
        self.assertEqual(summary["totals"]["ssd_byte_ns"], expected)


if __name__ == "__main__":
    unittest.main()

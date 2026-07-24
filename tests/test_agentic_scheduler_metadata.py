import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from serving.core.memory_model import Device, MemoryModel
from serving.core.request import Request
from serving.core.scheduler import Scheduler


class FakeDecodeMemory:
    def __init__(self):
        self.allocated = []

    @staticmethod
    def get_total_kv(request):
        return request.num_computed_tokens * 10

    def allocate(self, size, device):
        self.allocated.append((size, device))


class AgenticSchedulerMetadataTest(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.request = []
        self.scheduler.agentic_kv_manager = None
        self.scheduler.pd_prefill_reclaimability_generation = 0

    def test_optional_agentic_metadata_defaults_and_request_order(self):
        metadata = {
            "session_id": "session-a",
            "sub_request_index": 0,
            "ready_time_ns": None,
            "prefix_reuse_toks": None,
            "agentic_kv_hit_tokens": None,
            "agentic_kv_recompute_tokens": None,
            "agentic_kv_restore_ns": None,
            "agentic_kv_source_demotion_join_wait_ns": None,
            "agentic_kv_hbm_admission_wait_ns": None,
            "agentic_kv_transient_dram_capacity_wait_ns": None,
            "agentic_kv_restore_queue_wait_ns": None,
            "agentic_kv_restore_service_ns": None,
        }
        request = self.scheduler.add_request(
            [2, "model", 8, 10, 10, 0], metadata=metadata)

        self.assertEqual(request.ready_time, 10)
        self.assertEqual(request.prefix_reuse_tokens, 0)
        self.assertEqual(request.agentic_kv_hit_tokens, 0)
        self.assertEqual(request.agentic_kv_recompute_tokens, 0)
        self.assertEqual(request.agentic_kv_restore_ns, 0)
        self.assertEqual(
            request.agentic_kv_source_demotion_join_wait_ns, 0)
        self.assertEqual(request.agentic_kv_hbm_admission_wait_ns, 0)
        self.assertEqual(
            request.agentic_kv_transient_dram_capacity_wait_ns, 0)
        self.assertEqual(request.agentic_kv_restore_queue_wait_ns, 0)
        self.assertEqual(request.agentic_kv_restore_service_ns, 0)

        propagated = self.scheduler.add_request(
            [5, "model", 8, 10, 20, 0],
            metadata={
                "agentic_kv_source_demotion_join_wait_ns": 6,
                "agentic_kv_hbm_admission_wait_ns": 9,
                "agentic_kv_transient_dram_capacity_wait_ns": 4,
            },
            enqueue=False,
        )
        self.assertEqual(
            propagated.agentic_kv_source_demotion_join_wait_ns, 6)
        self.assertEqual(propagated.agentic_kv_hbm_admission_wait_ns, 9)
        self.assertEqual(
            propagated.agentic_kv_transient_dram_capacity_wait_ns, 4)

        self.scheduler.add_request([1, "model", 8, 10, 5, 0])
        self.scheduler.add_request([0, "model", 8, 10, 5, 0])
        self.assertEqual(
            [(item.arrival, item.id) for item in self.scheduler.request],
            [(5, 0), (5, 1), (10, 2)],
        )

        delayed = self.scheduler.add_request(
            [3, "model", 8, 10, 0, 0],
            metadata={"ready_time_ns": 100})
        earlier = self.scheduler.add_request(
            [4, "model", 8, 10, 50, 0],
            metadata={"ready_time_ns": 50})
        self.assertLess(
            self.scheduler.request.index(earlier),
            self.scheduler.request.index(delayed),
        )

    def test_request_context_limit_includes_prompt_and_output(self):
        self.scheduler.max_model_len = 128

        accepted = self.scheduler.add_request(
            [10, "model", 120, 128, 0, 0]
        )
        self.assertEqual(accepted.output, 128)

        with self.assertRaisesRegex(
                ValueError, "total sequence length 129"):
            self.scheduler.add_request(
                [11, "model", 120, 129, 0, 0]
            )

    def test_transient_dram_wait_is_preserved_in_request_csv(self):
        self.scheduler.instance_id = 0
        request = self.scheduler.add_request(
            [20, "model", 8, 10, 0, 0],
            metadata={
                "agentic_kv_source": "ssd",
                "agentic_kv_restore_ns": 10,
                "agentic_kv_hbm_admission_wait_ns": 0,
                "agentic_kv_transient_dram_capacity_wait_ns": 4,
                "agentic_kv_restore_queue_wait_ns": 4,
                "agentic_kv_restore_service_ns": 6,
            },
            enqueue=False,
        )
        self.scheduler.done = [request]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.csv"
            self.scheduler.save_output(str(path))
            with path.open(newline="") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(
            row["agentic_kv_transient_dram_capacity_wait_ns"], "4")
        self.assertEqual(row["agentic_kv_hbm_admission_wait_ns"], "0")
        self.assertEqual(row["agentic_kv_restore_queue_wait_ns"], "4")
        self.assertEqual(row["agentic_kv_restore_service_ns"], "6")
        self.assertEqual(
            int(row["agentic_kv_restore_ns"]),
            int(row["agentic_kv_hbm_admission_wait_ns"])
            + int(row["agentic_kv_restore_queue_wait_ns"])
            + int(row["agentic_kv_restore_service_ns"]),
        )

    def test_request_csv_tail_fields_round_trip_without_column_shift(self):
        self.scheduler.instance_id = 0
        request = self.scheduler.add_request(
            [21, "model", 8, 10, 0, 0], enqueue=False)
        sentinels = {
            "pd_launch_admission_wait_ns": 101,
            "pd_launch_admission_critical_wait_ns": 102,
            "pd_chunk_admission_count": 103,
            "pd_chunk_cancelled_admission_count": 104,
            "pd_chunk_admitted_tokens_total": 105,
            "pd_chunk_prefill_admitted_per_rank_bytes": 106,
            "pd_chunk_decode_admitted_per_rank_bytes": 107,
            "pd_chunk_admission_wait_ns_total": 108,
            "pd_chunk_admission_critical_wait_ns_total": 109,
            "pd_chunk_successful_admission_wait_ns_total": 110,
            "pd_chunk_successful_admission_critical_wait_ns_total": 111,
            "pd_chunk_cancelled_admission_wait_ns_total": 112,
            "pd_chunk_cancelled_admission_critical_wait_ns_total": 113,
            "pd_chunk_prefill_peak_hbm_used_per_rank_bytes": 114,
            "pd_chunk_decode_peak_hbm_used_per_rank_bytes": 115,
            "pd_prefill_initial_restored_per_rank_bytes": 116,
            "pd_prefill_handoff_released_per_rank_bytes": 117,
            "pd_decode_handoff_owned_per_rank_bytes": 118,
            "active_prefill_recompute_preemptions": 119,
            "active_prefill_recompute_tokens": 120,
            "active_prefill_recompute_frontier_tokens": 121,
            "pd_active_prefill_recompute_generation": 122,
            "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute": (
                123),
            "pd_kv_ownership_state": "tail-sentinel",
        }
        for field, value in sentinels.items():
            setattr(request, field, value)
        self.scheduler.done = [request]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.csv"
            self.scheduler.save_output(str(path))
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                row = next(reader)
                fieldnames = list(reader.fieldnames)

        self.assertNotIn(None, row)
        self.assertEqual(len(fieldnames), len(row))
        self.assertEqual(set(fieldnames), set(row))
        for field, value in sentinels.items():
            self.assertEqual(row[field], str(value), field)

    def test_enqueue_snapshots_booked_model_resource_dependency(self):
        self.scheduler.instance_id = 0
        self.scheduler.agentic_kv_manager = type("Manager", (), {
            "model_dispatch_resource_ready_time": staticmethod(
                lambda instance_id, eligibility_ns: 200),
        })()

        request = self.scheduler.add_request(
            [12, "model", 8, 10, 100, 0],
            metadata={"ready_time_ns": 150},
        )
        request.set_que_delay(200)

        self.assertEqual(request.scheduler_resource_ready_time_ns, 200)
        self.assertEqual(request.first_schedule_eligibility_time_ns, 200)
        self.assertEqual(request.scheduler_queue_wait_ns, 0)

    def test_pd_decode_allocation_requires_prefill_owner_release(self):
        decode = Scheduler.__new__(Scheduler)
        decode.instance_id = 1
        decode.enable_prefix_caching = False
        decode.request = []
        decode.agentic_kv_manager = None
        decode.memory = FakeDecodeMemory()
        request = Request(0, "model", 8, 10, 0, 0)
        request.num_computed_tokens = 8
        request.agentic_kv_owner_instance_id = 0

        with self.assertRaisesRegex(RuntimeError, "still owned by instance 0"):
            decode.add_decode(request)
        self.assertEqual(decode.memory.allocated, [])

        request.agentic_kv_owner_instance_id = None
        decode.add_decode(request)
        self.assertEqual(decode.memory.allocated, [(80, Device.NPU)])
        self.assertEqual(request.agentic_kv_owner_instance_id, 1)

    def test_pd_decode_allocates_only_suffix_when_prefix_is_retained(self):
        decode = Scheduler.__new__(Scheduler)
        decode.instance_id = 1
        decode.enable_prefix_caching = False
        decode.request = []
        decode.agentic_kv_manager = None
        decode.memory = FakeDecodeMemory()
        request = Request(0, "model", 8, 10, 0, 0)
        request.num_computed_tokens = 8
        request.agentic_kv_retained_instance_id = 1
        request.agentic_kv_retained_per_rank_bytes = 30

        decode.add_decode(request)

        self.assertEqual(decode.memory.allocated, [(50, Device.NPU)])
        self.assertIsNone(request.agentic_kv_retained_instance_id)
        self.assertEqual(request.agentic_kv_retained_per_rank_bytes, 0)
        self.assertEqual(request.agentic_kv_owner_instance_id, 1)

    def test_pd_decode_consumes_preallocated_receive_without_allocating_twice(self):
        decode = Scheduler.__new__(Scheduler)
        decode.instance_id = 1
        decode.enable_prefix_caching = False
        decode.request = []
        decode.agentic_kv_manager = None
        decode.memory = FakeDecodeMemory()
        request = Request(0, "model", 8, 10, 0, 0)
        request.num_computed_tokens = 8
        request.agentic_kv_retained_instance_id = 1
        request.agentic_kv_retained_per_rank_bytes = 30
        request.pd_decode_target_instance_id = 1
        request.pd_decode_full_per_rank_bytes = 80
        request.pd_decode_reserved_per_rank_bytes = 50
        request.pd_prefill_full_per_rank_bytes = 80
        request.pd_prefill_reserved_per_rank_bytes = 50
        request.pd_prefill_preallocated_per_rank_bytes = 80

        decode.add_decode(request, preallocated_hbm_bytes=50)

        self.assertEqual(decode.memory.allocated, [])
        self.assertEqual(request.instance_id, 1)
        self.assertEqual(request.agentic_kv_owner_instance_id, 1)
        self.assertEqual(request.pd_decode_reserved_per_rank_bytes, 0)
        self.assertEqual(request.pd_prefill_preallocated_per_rank_bytes, 0)

    def test_pd_decode_rejects_preallocation_mismatch_before_mutation(self):
        decode = Scheduler.__new__(Scheduler)
        decode.instance_id = 1
        decode.enable_prefix_caching = False
        decode.request = []
        decode.agentic_kv_manager = None
        decode.memory = FakeDecodeMemory()
        request = Request(0, "model", 8, 10, 0, 0)
        request.num_computed_tokens = 8
        request.pd_decode_target_instance_id = 1
        request.pd_decode_full_per_rank_bytes = 80
        request.pd_decode_reserved_per_rank_bytes = 79

        with self.assertRaisesRegex(RuntimeError, "reservation"):
            decode.add_decode(request, preallocated_hbm_bytes=80)

        self.assertEqual(request.instance_id, 0)
        self.assertEqual(decode.memory.allocated, [])

    def test_measurement_censor_releases_queued_active_hbm(self):
        class CensorMemory:
            def __init__(self, used):
                self.used = used

            @staticmethod
            def get_evict_kv(request):
                return int(request.num_computed_tokens) * 10

            def free(self, size, device):
                if device != Device.NPU or size > self.used:
                    raise AssertionError((size, device, self.used))
                self.used -= size

        decode = Scheduler.__new__(Scheduler)
        decode.instance_id = 1
        decode.enable_prefix_caching = False
        decode.inflight = []
        request = Request(30, "model", 8, 10, 0, 1)
        request.session_id = "queued-active"
        request.num_computed_tokens = 8
        request.agentic_kv_owner_instance_id = 1
        decode.request = [request]
        decode.memory = CensorMemory(80)

        audit = decode.censor_queued_request(request, 100)

        self.assertEqual(audit["released_per_rank_bytes"], 80)
        self.assertEqual(decode.memory.used, 0)
        self.assertEqual(decode.request, [])
        self.assertIsNone(request.agentic_kv_owner_instance_id)

    def test_measurement_censor_does_not_free_recompute_twice(self):
        class EmptyMemory:
            @staticmethod
            def get_evict_kv(request):
                raise AssertionError("recompute KV was already released")

            @staticmethod
            def free(size, device):
                raise AssertionError((size, device))

        decode = Scheduler.__new__(Scheduler)
        decode.instance_id = 1
        decode.enable_prefix_caching = False
        decode.inflight = []
        request = Request(31, "model", 8, 10, 0, 1)
        request.session_id = "queued-recompute"
        request.num_computed_tokens = 0
        request.recompute_target_tokens = 8
        request.agentic_kv_owner_instance_id = 1
        decode.request = [request]
        decode.memory = EmptyMemory()

        audit = decode.censor_queued_request(request, 100)

        self.assertEqual(audit["released_per_rank_bytes"], 0)
        self.assertTrue(audit["active_recompute_already_released"])
        self.assertEqual(decode.request, [])
        self.assertIsNone(request.agentic_kv_owner_instance_id)

    def test_prefill_full_preallocation_is_not_allocated_by_chunks_again(self):
        memory = MemoryModel.__new__(MemoryModel)
        memory.block_size = 16
        memory.enable_prefix_caching = False
        memory.get_kv = lambda tokens: int(tokens) * 10
        request = Request(0, "model", 100, 110, 0, 0)
        request.num_computed_tokens = 80
        request.pd_prefill_preallocated_per_rank_bytes = 1120

        first_chunk = memory.get_block_kv(
            [request], 1, {request.id: 16})
        final_chunk = memory.get_block_kv(
            [request], 1, {request.id: 20})

        self.assertEqual(first_chunk, 0)
        self.assertEqual(final_chunk, 0)

        request.pd_prefill_preallocated_per_rank_bytes = 0
        self.assertEqual(
            memory.get_block_kv([request], 1, {request.id: 20}),
            320,
        )

    def test_async_restore_cutoff_keeps_final_prompt_token_behind_join(self):
        request = Request(0, "model", 140, 142, 0, 0)
        request.num_computed_tokens = 80
        request.agentic_kv_overlap_cutoff_tokens = 139
        request.agentic_kv_restore_ready_time_ns = 1000

        self.assertEqual(
            self.scheduler._prefill_schedule_target(request, 999), 139)
        self.assertEqual(
            self.scheduler._prefill_schedule_target(request, 1000), 140)

    def _pd_prefill_request(
            self, request_id, input_tokens, hit_tokens, *,
            restore_ready_ns=100, retained_instance_id=None,
            retained_per_rank_bytes=0):
        self.scheduler.pd_type = "prefill"
        self.scheduler.instance_id = 0
        self.scheduler.agentic_kv_manager = object()
        return self.scheduler.add_request(
            [
                request_id, "model", input_tokens, input_tokens + 2,
                0, 0,
            ],
            metadata={
                "session_id": f"session-{request_id}",
                "agentic_kv_hit_tokens": hit_tokens,
                "agentic_kv_source": (
                    "hbm" if retained_instance_id is not None else "cpu"),
                "agentic_kv_restore_ready_time_ns": restore_ready_ns,
                "agentic_kv_owner_instance_id": (
                    0 if hit_tokens > 0 else None),
                "agentic_kv_retained_instance_id": retained_instance_id,
                "agentic_kv_retained_per_rank_bytes": (
                    retained_per_rank_bytes),
            },
            enqueue=False,
        )

    @staticmethod
    def _handoff_batch(request, batch_id, batch_time, q_tokens):
        return SimpleNamespace(
            batch_id=batch_id,
            batch_time=batch_time,
            requests=[request],
            pd_restored_prefix_handoff_tokens=0,
            pd_restored_prefix_handoff_by_request={},
            pd_new_kv_handoff_tokens=0,
            pd_new_kv_handoff_by_request={},
            pd_kv_handoff_committed=False,
        ), {request.id: q_tokens}

    def test_lower_tier_prefix_is_staged_once_after_restore_and_committed(self):
        request = self._pd_prefill_request(
            20, input_tokens=17, hit_tokens=15, restore_ready_ns=100)
        self.assertTrue(request.pd_kv_handoff_tracking_enabled)
        self.assertEqual(
            request.pd_restored_prefix_handoff_pending_tokens, 15)

        # Async-decode-join may form an early suffix batch while DMA is live.
        early, scheduled = self._handoff_batch(request, 0, 99, 1)
        self.scheduler._stage_pd_kv_handoff(early, scheduled)
        self.assertEqual(early.pd_restored_prefix_handoff_tokens, 0)
        self.assertEqual(
            request.pd_restored_prefix_handoff_pending_tokens, 15)
        self.assertEqual(request.pd_new_kv_handoff_sent_tokens, 0)

        self.scheduler._commit_pd_kv_handoff(early)
        self.assertEqual(request.pd_new_kv_handoff_sent_tokens, 1)
        self.assertEqual(
            request.pd_restored_prefix_handoff_pending_tokens, 15)

        request.num_computed_tokens = 16
        ready, scheduled = self._handoff_batch(request, 1, 100, 1)
        self.scheduler._stage_pd_kv_handoff(ready, scheduled)
        self.assertEqual(ready.pd_restored_prefix_handoff_tokens, 15)
        # Formation is non-destructive; a failed ASTRA graph can retry.
        self.assertEqual(
            request.pd_restored_prefix_handoff_pending_tokens, 15)
        self.assertEqual(
            request.pd_restored_prefix_handoff_sent_tokens, 0)

        self.scheduler._commit_pd_kv_handoff(ready)
        request.num_computed_tokens = 17
        self.scheduler._validate_pd_prompt_kv_handoff(request)
        self.assertEqual(
            request.pd_restored_prefix_handoff_pending_tokens, 0)
        self.assertEqual(
            request.pd_restored_prefix_handoff_sent_tokens, 15)
        self.assertEqual(request.pd_new_kv_handoff_sent_tokens, 2)

        later, scheduled = self._handoff_batch(request, 2, 101, 1)
        self.scheduler._stage_pd_kv_handoff(later, scheduled)
        self.assertEqual(later.pd_restored_prefix_handoff_tokens, 0)

    def test_hbm_retained_prefix_sends_only_final_prompt_token(self):
        request = self._pd_prefill_request(
            21, input_tokens=16, hit_tokens=15,
            retained_instance_id=1, retained_per_rank_bytes=160)
        self.assertEqual(
            request.pd_restored_prefix_handoff_pending_tokens, 0)

        batch, scheduled = self._handoff_batch(request, 0, 100, 1)
        self.scheduler._stage_pd_kv_handoff(batch, scheduled)
        self.scheduler._commit_pd_kv_handoff(batch)
        request.num_computed_tokens = 16
        self.scheduler._validate_pd_prompt_kv_handoff(request)

        self.assertEqual(batch.pd_restored_prefix_handoff_tokens, 0)
        self.assertEqual(
            request.pd_restored_prefix_handoff_sent_tokens, 0)
        self.assertEqual(request.pd_new_kv_handoff_sent_tokens, 1)

    def test_input_minus_one_cap_holds_across_block_boundary(self):
        one_block = self._pd_prefill_request(
            22, input_tokens=16, hit_tokens=15)
        block_plus_one = self._pd_prefill_request(
            23, input_tokens=17, hit_tokens=16)

        self.assertEqual(
            one_block.pd_restored_prefix_handoff_pending_tokens, 15)
        self.assertEqual(
            block_plus_one.pd_restored_prefix_handoff_pending_tokens, 16)
        with self.assertRaisesRegex(RuntimeError, "final prompt token"):
            self._pd_prefill_request(
                24, input_tokens=16, hit_tokens=16)

    def test_duplicate_pd_handoff_commit_is_rejected(self):
        request = self._pd_prefill_request(
            25, input_tokens=16, hit_tokens=15)
        batch, scheduled = self._handoff_batch(request, 0, 100, 1)
        self.scheduler._stage_pd_kv_handoff(batch, scheduled)
        self.scheduler._commit_pd_kv_handoff(batch)

        with self.assertRaisesRegex(RuntimeError, "committed twice"):
            self.scheduler._commit_pd_kv_handoff(batch)


if __name__ == "__main__":
    unittest.main()

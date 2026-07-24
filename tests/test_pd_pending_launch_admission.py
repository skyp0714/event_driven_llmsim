import unittest

from serving.core.memory_model import Device
from serving.core.request import Request
from serving.core.router import Router
from tests.test_agentic_router import (
    FakeDecodeAdmissionManager,
    FakeScheduler,
    PrefillRequest,
)


class CapacityAwareAdmissionManager(FakeDecodeAdmissionManager):
    """Minimal exact-capacity manager for pending-restore admission tests."""

    def __init__(self, schedulers):
        super().__init__(ready_ns=0)
        self.schedulers = {
            int(scheduler.instance_id): scheduler
            for scheduler in schedulers
        }
        self.cancelled = []
        self.censored = []

    def restore_capacity_state(self, instance_id):
        scheduler = self.schedulers[int(instance_id)]
        return (
            int(scheduler.memory.npu_used),
            int(scheduler.memory.npu_mem - scheduler.memory.npu_used),
        )

    def hbm_unreserved_per_rank_bytes(self, instance_id):
        return self.restore_capacity_state(instance_id)[1]

    def claim_active_hbm_reclaim(
            self, instance_id, needed_per_rank_bytes, now_ns,
            owner_kind="legacy", owner_id=None):
        instance_id = int(instance_id)
        needed_per_rank_bytes = int(needed_per_rank_bytes)
        if instance_id in self.claims:
            return self.claims[instance_id].ready_ns
        scheduler = self.schedulers[instance_id]
        available = (
            int(scheduler.memory.npu_mem)
            - int(scheduler.memory.npu_used)
        )
        if needed_per_rank_bytes > available:
            self.claim_calls.append(
                (instance_id, needed_per_rank_bytes, int(now_ns)))
            return None
        return super().claim_active_hbm_reclaim(
            instance_id,
            needed_per_rank_bytes,
            now_ns,
            owner_kind=owner_kind,
            owner_id=owner_id,
        )

    def cancel_active_hbm_reclaim(self, instance_id, now_ns):
        claim = super().cancel_active_hbm_reclaim(instance_id, now_ns)
        if claim is not None:
            self.cancelled.append((
                int(instance_id),
                int(claim.owner_id),
                int(now_ns),
            ))
        return claim

    def censor_preallocated_pd_request(
            self, request, prefill_instance_id,
            decode_instance_id, now_ns):
        prefill = self.schedulers[int(prefill_instance_id)]
        decode = self.schedulers[int(decode_instance_id)]
        prefill_bytes = int(
            request.pd_prefill_preallocated_per_rank_bytes)
        decode_bytes = int(request.pd_decode_full_per_rank_bytes)
        prefill.memory.free(prefill_bytes, Device.NPU)
        decode.memory.free(decode_bytes, Device.NPU)
        request.agentic_kv_owner_instance_id = None
        request.agentic_kv_retained_instance_id = None
        request.agentic_kv_retained_per_rank_bytes = 0
        request.pd_prefill_preallocated_per_rank_bytes = 0
        request.pd_decode_target_instance_id = None
        request.pd_decode_full_per_rank_bytes = 0
        request.pd_decode_reserved_per_rank_bytes = 0
        audit = {
            "request_id": int(request.id),
            "session_id": str(request.session_id),
            "time_ns": int(now_ns),
            "released_prefill_per_rank_bytes": prefill_bytes,
            "released_decode_per_rank_bytes": decode_bytes,
        }
        self.censored.append(audit)
        return dict(audit)

    def censor_prepared_request(self, request, now_ns):
        restored = int(
            request.pd_prefill_initial_restored_per_rank_bytes)
        retained = int(request.agentic_kv_retained_per_rank_bytes)
        if restored:
            self.schedulers[int(request.instance_id)].memory.free(
                restored, Device.NPU)
        if retained:
            self.schedulers[
                int(request.agentic_kv_retained_instance_id)
            ].memory.free(retained, Device.NPU)
        request.agentic_kv_owner_instance_id = None
        request.agentic_kv_retained_instance_id = None
        request.agentic_kv_retained_per_rank_bytes = 0
        audit = {
            "request_id": int(request.id),
            "session_id": str(request.session_id),
            "time_ns": int(now_ns),
        }
        self.censored.append(audit)
        return dict(audit)


class PDIncrementalChunkAdmissionTest(unittest.TestCase):
    @staticmethod
    def _system(*, prefill_capacity=224, decode_capacity=224):
        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        prefill.memory.npu_mem = int(prefill_capacity)
        decode.memory.npu_mem = int(decode_capacity)
        manager = CapacityAwareAdmissionManager([prefill, decode])
        router = Router(
            2,
            [prefill, decode],
            0,
            "RR",
            agentic_kv_manager=manager,
        )
        return router, prefill, decode, manager

    @staticmethod
    def _request(
            request_id, prefill, *, input_tokens=100, hit_tokens=0,
            retained=False, ready_time_ns=100):
        request = Request(
            request_id, "model", input_tokens, input_tokens + 1,
            0, prefill.instance_id,
        )
        request.session_id = f"session-{request_id}"
        request.ready_time = int(ready_time_ns)
        request.agentic_kv_hit_tokens = int(hit_tokens)
        request.num_computed_tokens = int(hit_tokens)
        request.agentic_kv_restore_ns = max(0, int(ready_time_ns) - 100)
        request.agentic_kv_restore_ready_time_ns = int(ready_time_ns)
        request.pd_kv_handoff_tracking_enabled = True
        if hit_tokens:
            restored_bytes = (
                (int(hit_tokens) + prefill.block_size - 1)
                // prefill.block_size * prefill.block_size)
            prefill.memory.allocate(restored_bytes, Device.NPU)
            request.agentic_kv_owner_instance_id = prefill.instance_id
            if retained:
                request.agentic_kv_retained_instance_id = 1
                request.agentic_kv_retained_per_rank_bytes = restored_bytes
        return request

    @staticmethod
    def _admit(router, prefill, request, now_ns=100):
        pair = (prefill.instance_id, 1)
        router._pd_admission_owner[pair] = request.id
        router._stage_pd_receive_admission(request, prefill, now_ns)
        return router.process_pending_decode_handoffs(now_ns)

    def test_no_full_prompt_allocation_at_binding(self):
        router, prefill, decode, _ = self._system()
        request = self._request(1, prefill, input_tokens=100)
        self.assertEqual(self._admit(router, prefill, request), 1)
        self.assertEqual(prefill.memory.npu_used, 0)
        self.assertEqual(decode.memory.npu_used, 0)
        self.assertEqual(request.pd_prefill_full_per_rank_bytes, 112)
        self.assertEqual(request.pd_decode_full_per_rank_bytes, 112)
        self.assertEqual(request.pd_prefill_owned_per_rank_bytes, 0)
        self.assertEqual(request.pd_decode_owned_per_rank_bytes, 0)

    def test_lower_tier_first_chunk_exposes_asymmetric_pd_bytes(self):
        router, prefill, decode, _ = self._system()
        request = self._request(
            2, prefill, input_tokens=100, hit_tokens=80)
        self.assertEqual(self._admit(router, prefill, request), 1)
        requirements = router.pd_prefill_chunk_requirements(
            request, prefill, 16)
        self.assertEqual(
            requirements["prefill_current_per_rank_bytes"], 80)
        self.assertEqual(
            requirements["decode_current_per_rank_bytes"], 0)
        self.assertEqual(
            requirements["prefill_delta_per_rank_bytes"], 16)
        self.assertEqual(
            requirements["decode_delta_per_rank_bytes"], 96)
        self.assertTrue(router.admit_pd_prefill_chunk(
            prefill, request, 16, 100))
        self.assertEqual(prefill.memory.npu_used, 96)
        self.assertEqual(decode.memory.npu_used, 96)
        self.assertEqual(request.pd_chunk_admitted_tokens, 16)
        self.assertEqual(
            request.pd_chunk_admission_history[0][
                "decode_delta_per_rank_bytes"], 96)

    def test_one_sided_capacity_rolls_back_without_blocking_pair_claim(self):
        router, prefill, decode, manager = self._system(
            prefill_capacity=224,
            decode_capacity=112,
        )
        decode.memory.npu_used = 32
        request = self._request(
            3, prefill, input_tokens=100, hit_tokens=80)
        self.assertEqual(self._admit(router, prefill, request), 1)
        self.assertFalse(router.admit_pd_prefill_chunk(
            prefill, request, 16, 100))
        self.assertEqual(manager.claims, {})
        self.assertEqual(manager.cancelled, [(0, request.id, 100)])
        self.assertEqual(prefill.memory.npu_used, 80)
        self.assertEqual(decode.memory.npu_used, 32)
        self.assertTrue(router.has_pending_decode_handoffs())
        decode.memory.npu_used = 0
        router.process_pending_decode_handoffs(101)
        self.assertEqual(request.pd_chunk_admitted_tokens, 16)
        self.assertEqual(prefill.memory.npu_used, 96)
        self.assertEqual(decode.memory.npu_used, 96)

    def test_cutoff_before_first_chunk_releases_restored_prefix_once(self):
        router, prefill, decode, manager = self._system()
        request = self._request(
            4, prefill, input_tokens=100, hit_tokens=80,
            ready_time_ns=500)
        self.assertEqual(self._admit(router, prefill, request), 0)

        router.freeze_session_admission()
        summary = router.finalize_measurement_censoring(200)

        self.assertEqual(summary["pending_prefill_launches_at_cutoff"], 1)
        self.assertEqual(summary["censored_pending_prefill_launches"], 1)
        self.assertEqual([row["request_id"] for row in manager.censored], [4])
        self.assertEqual(prefill.memory.npu_used, 0)
        self.assertEqual(decode.memory.npu_used, 0)
        self.assertEqual(router._pending_prefill_launches, [])
        self.assertEqual(router._pd_admission_owner, {})
        self.assertEqual(manager.claims, {})

    def test_same_pair_pending_head_precedes_tail(self):
        router, prefill, decode, manager = self._system(
            prefill_capacity=16, decode_capacity=16)
        decode.memory.npu_used = 16
        head = self._request(10, prefill, input_tokens=16)
        tail = self._request(11, prefill, input_tokens=16)
        self.assertEqual(self._admit(router, prefill, head), 1)
        self.assertEqual(self._admit(router, prefill, tail), 1)
        self.assertFalse(router.admit_pd_prefill_chunk(
            prefill, head, 16, 100))
        self.assertFalse(router.admit_pd_prefill_chunk(
            prefill, tail, 16, 100))
        pair = (prefill.instance_id, decode.instance_id)
        self.assertEqual(
            [row["request"].id
             for row in router._pending_pd_chunk_admissions[pair]],
            [head.id, tail.id],
        )
        decode.memory.npu_used = 0
        router.process_pending_decode_handoffs(101)
        self.assertEqual(head.pd_chunk_admitted_tokens, 16)
        self.assertEqual(tail.pd_chunk_admitted_tokens, 0)
        self.assertEqual(
            router._pending_pd_chunk_admissions[pair][0]["request"], tail)
        self.assertEqual(manager.claims, {})

    def test_blocked_pair_does_not_block_independent_pair(self):
        p0 = FakeScheduler(0, "prefill", node_id=0)
        d0 = FakeScheduler(1, "decode", node_id=0)
        p1 = FakeScheduler(2, "prefill", node_id=1)
        d1 = FakeScheduler(3, "decode", node_id=1)
        for scheduler in (p0, d0, p1, d1):
            scheduler.memory.npu_mem = 16
        d0.memory.npu_used = 16
        manager = CapacityAwareAdmissionManager([p0, d0, p1, d1])
        router = Router(
            4, [p0, d0, p1, d1], 0, "RR",
            agentic_kv_manager=manager)
        blocked = self._request(20, p0, input_tokens=16)
        ready = self._request(21, p1, input_tokens=16)
        router._stage_pd_receive_admission(blocked, p0, 100)
        router._stage_pd_receive_admission(ready, p1, 100)
        router.process_pending_decode_handoffs(100)
        self.assertFalse(router.admit_pd_prefill_chunk(
            p0, blocked, 16, 100))
        self.assertTrue(router.admit_pd_prefill_chunk(
            p1, ready, 16, 100))
        self.assertEqual(ready.pd_chunk_admitted_tokens, 16)
        self.assertEqual(p1.memory.npu_used, 16)
        self.assertEqual(d1.memory.npu_used, 16)
        self.assertEqual(blocked.pd_chunk_admitted_tokens, 0)

    def test_cutoff_cancels_pending_chunk_claim_then_frees_current_once(self):
        router, prefill, decode, manager = self._system()
        manager.ready_ns = {prefill.instance_id: 500,
                            decode.instance_id: 500}
        request = self._request(
            30, prefill, input_tokens=100, hit_tokens=80)
        self.assertEqual(self._admit(router, prefill, request), 1)
        self.assertFalse(router.admit_pd_prefill_chunk(
            prefill, request, 16, 100))
        self.assertEqual(len(manager.claims), 2)
        router.freeze_session_admission()
        summary = router.finalize_measurement_censoring(200)
        self.assertEqual(summary["pending_pd_chunk_admissions_at_cutoff"], 1)
        self.assertEqual(prefill.memory.npu_used, 0)
        self.assertEqual(decode.memory.npu_used, 0)
        self.assertEqual(manager.claims, {})
        self.assertEqual(request.pd_kv_ownership_state, "censored")

    def test_cutoff_frees_admitted_uncommitted_chunk(self):
        router, prefill, decode, manager = self._system()
        request = self._request(40, prefill, input_tokens=100)
        self.assertEqual(self._admit(router, prefill, request), 1)
        self.assertTrue(router.admit_pd_prefill_chunk(
            prefill, request, 16, 100))
        self.assertEqual(prefill.memory.npu_used, 16)
        self.assertEqual(decode.memory.npu_used, 16)
        router.freeze_session_admission()
        router.finalize_measurement_censoring(200)
        self.assertEqual(prefill.memory.npu_used, 0)
        self.assertEqual(decode.memory.npu_used, 0)
        self.assertEqual(request.pd_kv_ownership_state, "censored")
        self.assertEqual(len(manager.censored), 1)

    def test_first_chunk_wait_is_resource_eligibility_not_scheduler_queue(self):
        router, prefill, decode, manager = self._system()
        manager.ready_ns = {prefill.instance_id: 500,
                            decode.instance_id: 300}
        request = self._request(50, prefill, input_tokens=16)
        request.agentic_kv_async_decode_join = True
        request.agentic_kv_restore_ns = 100
        request.agentic_kv_restore_ready_time_ns = 200
        self.assertEqual(self._admit(router, prefill, request), 1)
        self.assertFalse(router.admit_pd_prefill_chunk(
            prefill, request, 16, 100))
        router.process_pending_decode_handoffs(500)
        self.assertEqual(request.pd_chunk_admission_wait_ns, 400)
        self.assertEqual(request.pd_chunk_admission_critical_wait_ns, 300)
        self.assertEqual(request.scheduler_resource_ready_time_ns, 500)
        request.set_que_delay(600)
        self.assertEqual(request.scheduler_queue_wait_ns, 100)
        self.assertEqual(manager.async_restore_gates, [(50, 500, 0)])

    def test_failed_sync_chunk_attempt_releases_prepare_lock(self):
        router, prefill, decode, manager = self._system(
            prefill_capacity=112, decode_capacity=112)
        decode.memory.npu_used = 32
        manager.synchronous_swap_enabled = True
        manager.sync_hbm_boundary_instances = {
            prefill.instance_id, decode.instance_id}
        request = self._request(
            60, prefill, input_tokens=100, hit_tokens=80)
        self.assertEqual(self._admit(router, prefill, request), 1)
        self.assertFalse(router.admit_pd_prefill_chunk(
            prefill, request, 16, 100))
        self.assertEqual(manager.sync_prepare_locks, {})
        router.process_pending_decode_handoffs(100)
        self.assertEqual(manager.sync_prepare_locks, {})


if __name__ == "__main__":
    unittest.main()

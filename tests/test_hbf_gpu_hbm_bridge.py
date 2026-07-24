from pathlib import Path
import unittest

from serving.core.hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
)
from serving.core.hbf_full_model_lifecycle import (
    FullModelHBFLifecycle,
    PerGroupCapacityLedger,
)
from serving.core.hbf_full_model_pool import (
    FullModelHBFServingPool,
    derive_lpddr_workspace_bytes,
)
from serving.core.hbf_gpu_hbm_bridge import (
    FullModelHBFGPUHBMBridge,
    GPUHBMBridgeCapacityError,
    GPUHBMBridgeError,
    GPUHBMBridgeStaleEventError,
    GPUHBMBridgeUnderflowError,
    GPUHBMBridgeUnsupportedReuseError,
    GPU_HBM_BRIDGE_SCHEMA,
)
from serving.core.hbf_online_adapter import (
    FullModelHBFOnlineAdapter,
    GPUHBMEventKind,
    GPUHBMOwnershipEvent,
)
from serving.core.memory_model import Device, MemoryModel
from serving.core.scheduler import Scheduler


REPO_ROOT = Path(__file__).resolve().parents[1]


class NullLogger:
    def info(self, *args, **kwargs):
        pass


def finite_scheduler(
        instance_id, pd_type, *,
        allocatable_bytes=4096, weight_bytes=128,
        kv_bytes_per_token_per_rank=2):
    if (
        kv_bytes_per_token_per_rank <= 0
        or kv_bytes_per_token_per_rank % 2
    ):
        raise ValueError(
            "test KV bytes per token must be a positive even integer")
    memory = MemoryModel.__new__(MemoryModel)
    memory.instance_id = instance_id
    memory.node_id = 0
    memory.block_size = 16
    memory.kv_heads_per_tp_rank = 1
    memory.head_dim = kv_bytes_per_token_per_rank // 2
    memory.layers_per_pp_rank = 1
    memory.kv_fp = 1
    memory.weight = weight_bytes
    memory.npu_used = weight_bytes
    memory.npu_peak_used = weight_bytes
    memory.npu_allocatable_mem = allocatable_bytes
    memory.npu_mem = allocatable_bytes
    memory.logger = NullLogger()
    memory.enable_prefix_caching = False

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.instance_id = instance_id
    scheduler.node_id = 0
    scheduler.block_size = 16
    scheduler.enable_prefix_caching = False
    scheduler.pd_type = pd_type
    scheduler.memory = memory
    scheduler.request = []
    scheduler.agentic_kv_manager = None
    scheduler.pd_prefill_reclaimability_generation = 0
    scheduler.max_model_len = 4096
    return scheduler


def ownership_event(
        memory, kind, *,
        session_id="session-a",
        request_id=1,
        instance_id=None,
        time_ns=10,
        token_count=17,
        reason=None):
    if instance_id is None:
        instance_id = memory.instance_id
    accounted_tokens = (
        (token_count + memory.block_size - 1)
        // memory.block_size
        * memory.block_size
        if token_count else 0
    )
    return GPUHBMOwnershipEvent(
        kind=kind,
        session_id=session_id,
        request_id=request_id,
        gpu_instance_id=instance_id,
        time_ns=time_ns,
        token_count=token_count,
        accounted_tokens_per_rank=accounted_tokens,
        logical_bytes=token_count * 8,
        per_rank_bytes=memory.get_kv(accounted_tokens),
        reason=reason or kind.value,
    )


def build_recompute_adapter():
    hardware = HBFServerHardware()
    layout = HBFParallelLayout.for_key("tp4")
    workspace = derive_lpddr_workspace_bytes(
        layout,
        max_num_batched_tokens=16,
        max_num_seqs=4,
    )
    ledger = PerGroupCapacityLedger(
        group_count=layout.replicas,
        capacity_bytes=(
            hardware.lpddr_capacity_bytes_per_card - workspace),
    )
    lifecycle = FullModelHBFLifecycle(
        hardware=hardware,
        layout=layout,
        lpddr_ledger=ledger,
        execution_backend="external_astra",
        server_id=7,
        astra_chunk_bytes=1024 * 1024,
    )
    pool = FullModelHBFServingPool(
        repo_root=REPO_ROOT,
        hardware=hardware,
        layout=layout,
        lpddr_ledger=ledger,
        placement_resolver=lifecycle.placement_snapshot,
        max_num_batched_tokens=16,
        max_num_seqs=4,
        max_prefill_chunk_tokens=16,
        execution_backend="external_astra",
        server_id=7,
    )
    return FullModelHBFOnlineAdapter(
        lifecycle=lifecycle,
        pool=pool,
        gpu_resume_mode="recompute",
    )


def raw_request(
        request_id, session_id, call_index, arrival_ns,
        input_tokens, output_tokens, prefix_reuse_tokens,
        has_successor):
    return {
        "index": request_id,
        "session_id": session_id,
        "sub_request_index": call_index,
        "arrival_time_ns": arrival_ns,
        "input_toks": input_tokens,
        "output_toks": input_tokens + output_tokens,
        "prefix_reuse_toks": prefix_reuse_tokens,
        "prefix_reuse_source": "test",
        "wakekv_has_successor": has_successor,
    }


class FullModelHBFGPUHBMBridgePDTests(unittest.TestCase):
    def setUp(self):
        self.prefill = finite_scheduler(0, "prefill")
        self.decode = finite_scheduler(1, "decode")
        self.bridge = FullModelHBFGPUHBMBridge(
            {0: self.prefill, 1: self.decode},
            pd_pairs=[(0, 1)],
        )

    def retain(self, *, token_count=17):
        event = ownership_event(
            self.decode.memory,
            GPUHBMEventKind.TURN_RETAIN,
            token_count=token_count,
        )
        self.bridge.apply_event(event)
        return event

    def test_quiescent_bridge_survives_post_run_weight_release(self):
        self.prefill.memory.npu_used = 0
        self.decode.memory.npu_used = 0

        self.bridge.assert_invariants()

        report = self.bridge.report()
        self.assertEqual(
            report["memory_by_instance"][0][
                "dynamic_used_per_rank_bytes"],
            0,
        )
        self.assertEqual(
            report["memory_by_instance"][1][
                "dynamic_used_per_rank_bytes"],
            0,
        )

    def test_turn_retain_and_migration_release_hit_exact_decode_memory(self):
        baseline = self.decode.memory.npu_used
        retained = self.retain()
        self.assertEqual(retained.accounted_tokens_per_rank, 32)
        self.assertEqual(retained.per_rank_bytes, 64)
        self.assertEqual(self.decode.memory.npu_used, baseline + 64)
        self.assertEqual(self.prefill.memory.npu_used, baseline)

        released = ownership_event(
            self.decode.memory,
            GPUHBMEventKind.MIGRATION_RELEASE,
            request_id=retained.request_id,
            time_ns=20,
            token_count=17,
        )
        result = self.bridge.apply_event(released)
        self.assertEqual(result["freed_per_rank_bytes"], 64)
        self.assertEqual(self.decode.memory.npu_used, baseline)
        self.assertEqual(
            self.bridge.report()["idle_allocations"], [])

        with self.assertRaisesRegex(
                GPUHBMBridgeStaleEventError, "duplicate"):
            self.bridge.apply_event(released)

    def test_pd_decode_reservation_projects_completed_prompt_kv(self):
        request = self.prefill.add_request(
            [90, "model", 17, 19, 0, 0],
            metadata={"session_id": "session-reserve"},
            enqueue=False,
        )
        decode_baseline = self.decode.memory.npu_used

        self.assertTrue(self.bridge.try_reserve_pd_decode(
            request,
            prefill_instance_id=0,
            decode_instance_id=1,
        ))
        reservation = self.bridge.pd_decode_reservation(request)
        self.assertEqual(reservation.projected_context_tokens, 17)
        self.assertEqual(reservation.full_per_rank_bytes, 64)
        self.assertEqual(reservation.reserved_per_rank_bytes, 64)
        self.assertEqual(
            self.decode.memory.npu_used,
            decode_baseline + reservation.reserved_per_rank_bytes,
        )

        request.num_computed_tokens = request.original_input
        request.generated_tokens = 1
        self.assertIsNone(self.decode.add_decode(
            request,
            preallocated_hbm_bytes=reservation.reserved_per_rank_bytes,
            completion_time_ns=10,
        ))
        consumed = self.bridge.consume_pd_decode_reservation(request)
        self.assertEqual(consumed, reservation)
        self.assertEqual(self.decode.request, [request])
        self.assertEqual(
            self.decode.memory.npu_used,
            decode_baseline + reservation.reserved_per_rank_bytes,
        )
        self.assertEqual(
            self.bridge.report()["pending_pd_decode_reservations"], [])
        self.assertEqual(
            self.bridge.metrics
            .pd_decode_transferred_to_scheduler_per_rank_bytes,
            reservation.reserved_per_rank_bytes,
        )
        self.bridge.assert_invariants()

        self.decode.request.clear()
        self.decode.memory.free(
            reservation.reserved_per_rank_bytes, Device.NPU)

    def test_pd_decode_reservation_waits_then_cancels_exact_bytes(self):
        request = self.prefill.add_request(
            [91, "model", 17, 19, 0, 0],
            metadata={"session_id": "session-wait"},
            enqueue=False,
        )
        self.decode.memory.npu_allocatable_mem = (
            self.decode.memory.weight + 64)
        self.decode.memory.npu_mem = (
            self.decode.memory.npu_allocatable_mem)
        self.decode.memory.allocate(32, Device.NPU)
        occupied = self.decode.memory.npu_used

        self.assertFalse(self.bridge.try_reserve_pd_decode(
            request,
            prefill_instance_id=0,
            decode_instance_id=1,
        ))
        self.assertEqual(self.decode.memory.npu_used, occupied)
        self.assertIsNone(self.bridge.pd_decode_reservation(request))

        self.decode.memory.free(32, Device.NPU)
        self.assertTrue(self.bridge.try_reserve_pd_decode(
            request,
            prefill_instance_id=0,
            decode_instance_id=1,
        ))
        self.assertEqual(self.decode.memory.npu_used, occupied + 32)
        audit = self.bridge.cancel_pd_decode_reservation(request)
        self.assertEqual(audit["reserved_per_rank_bytes"], 64)
        self.assertEqual(
            self.decode.memory.npu_used, self.decode.memory.weight)
        self.assertEqual(request.pd_kv_ownership_state, "censored")
        self.assertEqual(
            self.bridge.metrics.pd_decode_cancelled_per_rank_bytes, 64)
        self.bridge.assert_invariants()

    def test_zero_lineage_idle_release_frees_whole_retained_record(self):
        baseline = self.decode.memory.npu_used
        self.retain()
        release = ownership_event(
            self.decode.memory,
            GPUHBMEventKind.IDLE_RELEASE,
            request_id=2,
            time_ns=11,
            token_count=0,
            reason="gpu_lineage_trimmed_to_zero",
        )
        result = self.bridge.apply_event(release)
        self.assertEqual(result["freed_per_rank_bytes"], 64)
        self.assertEqual(self.decode.memory.npu_used, baseline)

    def test_pd_resume_claim_is_rejected_without_d_to_p_model(self):
        baseline = self.decode.memory.npu_used
        self.retain()
        retained_used = self.decode.memory.npu_used
        claim = ownership_event(
            self.decode.memory,
            GPUHBMEventKind.RESUME_CLAIM,
            request_id=2,
            time_ns=11,
            token_count=16,
        )
        with self.assertRaisesRegex(
                GPUHBMBridgeUnsupportedReuseError,
                "no D-to-P restore"):
            self.bridge.apply_event(claim)
        self.assertEqual(self.decode.memory.npu_used, retained_used)
        report = self.bridge.report()
        self.assertEqual(len(report["idle_allocations"]), 1)
        self.assertEqual(report["pending_colocated_claims"], [])
        self.assertEqual(report["metrics"]["rejected_events"], 1)
        self.assertEqual(
            report["memory_by_instance"][1][
                "bridge_owned_per_rank_bytes"],
            retained_used - baseline,
        )

    def test_pd_recompute_decoration_requires_prior_idle_release(self):
        self.retain()
        metadata = {
            "index": 2,
            "session_id": "session-a",
            "prefix_reuse_toks": 16,
            "agentic_kv_hit_tokens": 16,
            "agentic_kv_recompute_tokens": 0,
            "hbf_gpu_required_instance_id": 1,
        }
        with self.assertRaisesRegex(
                GPUHBMBridgeStaleEventError, "IDLE_RELEASE first"):
            self.bridge.decorate_pd_recompute(
                2, metadata,
                prefill_instance_id=0,
                decode_instance_id=1,
            )

        self.bridge.apply_event(ownership_event(
            self.decode.memory,
            GPUHBMEventKind.IDLE_RELEASE,
            request_id=2,
            time_ns=11,
            token_count=0,
        ))
        decorated = self.bridge.decorate_pd_recompute(
            2, metadata,
            prefill_instance_id=0,
            decode_instance_id=1,
        )
        self.assertEqual(decorated["prefix_reuse_toks"], 0)
        self.assertEqual(decorated["agentic_kv_hit_tokens"], 0)
        self.assertEqual(decorated["agentic_kv_recompute_tokens"], 16)
        self.assertIsNone(decorated["agentic_kv_owner_instance_id"])
        self.assertIsNone(decorated["agentic_kv_retained_instance_id"])
        self.assertEqual(
            decorated["agentic_kv_retained_per_rank_bytes"], 0)
        self.assertEqual(decorated["_pd_prefill_instance_id"], 0)
        self.assertIsNone(
            decorated["hbf_gpu_required_instance_id"])
        self.assertEqual(
            decorated["hbf_gpu_required_prefill_instance_id"], 0)
        self.assertEqual(
            decorated["hbf_gpu_required_decode_instance_id"], 1)
        self.assertTrue(
            decorated["hbf_gpu_unmodeled_d2p_restore"])

        request = self.prefill.add_request(
            [2, "model", 17, 20, 11, 0],
            metadata=decorated,
            enqueue=False,
        )
        binding = self.bridge.bind_pd_recompute(request)
        self.assertEqual(binding["prefill_instance_id"], 0)
        self.assertEqual(binding["decode_instance_id"], 1)
        self.assertEqual(request.num_computed_tokens, 0)
        self.assertIsNone(request.agentic_kv_owner_instance_id)
        self.assertIsNone(request.agentic_kv_retained_instance_id)

        with self.assertRaisesRegex(
                GPUHBMBridgeStaleEventError, "no P/D recompute binding"):
            self.bridge.bind_pd_recompute(request)

    def test_actual_adapter_recompute_events_bridge_to_pd_scheduler(self):
        adapter = build_recompute_adapter()
        kv_bytes = adapter.gpu_kv_bytes_per_token_per_rank
        prefill = finite_scheduler(
            0, "prefill",
            allocatable_bytes=10**15,
            kv_bytes_per_token_per_rank=kv_bytes,
        )
        decode = finite_scheduler(
            1, "decode",
            allocatable_bytes=10**15,
            kv_bytes_per_token_per_rank=kv_bytes,
        )
        bridge = FullModelHBFGPUHBMBridge(
            {0: prefill, 1: decode},
            pd_pairs=[(0, 1)],
            adapter=adapter,
        )
        baseline = decode.memory.npu_used

        first = raw_request(
            80, "session-pd", 0, 0,
            input_tokens=17,
            output_tokens=1,
            prefix_reuse_tokens=0,
            has_successor=True,
        )
        adapter.offer_raw_requests((first,), now_ns=0)
        adapter.complete_native_gpu_request(
            80,
            completion_ns=1,
            materialized_tokens=17,
            gpu_instance_id=1,
        )
        retain, = adapter.pop_gpu_hbm_events()
        self.assertEqual(retain.kind, GPUHBMEventKind.TURN_RETAIN)
        bridge.apply_event(retain)
        self.assertEqual(
            decode.memory.npu_used,
            baseline + retain.per_rank_bytes,
        )

        resume = raw_request(
            81, "session-pd", 1, 2,
            input_tokens=17,
            output_tokens=1,
            prefix_reuse_tokens=17,
            has_successor=False,
        )
        decision = adapter.offer_raw_request(resume, now_ns=2)
        self.assertTrue(decision.force_gpu_recompute)
        release, = adapter.pop_gpu_hbm_events()
        self.assertEqual(release.kind, GPUHBMEventKind.IDLE_RELEASE)
        self.assertEqual(release.token_count, 0)
        bridge.apply_event(release)
        self.assertEqual(decode.memory.npu_used, baseline)

        adapter_metadata = adapter.decorate_gpu_metadata(
            decision, resume)
        metadata = bridge.decorate_pd_recompute(
            81, adapter_metadata,
            prefill_instance_id=0,
            decode_instance_id=1,
        )
        request = prefill.add_request(
            [81, "model", 17, 18, 2, 0],
            metadata=metadata,
            enqueue=False,
        )
        bridge.bind_pd_recompute(request)
        self.assertEqual(request.num_computed_tokens, 0)
        self.assertEqual(
            request.agentic_kv_recompute_tokens, 17)
        self.assertIsNone(request.agentic_kv_owner_instance_id)
        self.assertIsNone(request.agentic_kv_retained_instance_id)
        self.assertEqual(
            bridge.report()["metrics"]["applied_events"], 2)
        self.assertEqual(
            bridge.report()["adapter_contract"][
                "gpu_resume_mode"],
            "recompute",
        )
        adapter.gpu_resume_mode = "sticky_reuse"
        with self.assertRaisesRegex(
                GPUHBMBridgeUnsupportedReuseError,
                "gpu_resume_mode='recompute'"):
            bridge.validate_adapter_contract(adapter)

    def test_wrong_pair_capacity_and_underflow_fail_before_state_change(self):
        metadata = {
            "index": 2,
            "session_id": "session-a",
            "prefix_reuse_toks": 4,
        }
        with self.assertRaisesRegex(
                GPUHBMBridgeStaleEventError, "unconfigured pair"):
            self.bridge.decorate_pd_recompute(
                2, metadata,
                prefill_instance_id=0,
                decode_instance_id=0,
            )

        small_decode = finite_scheduler(
            3, "decode",
            allocatable_bytes=180,
            weight_bytes=128,
        )
        small_prefill = finite_scheduler(2, "prefill")
        bridge = FullModelHBFGPUHBMBridge(
            {2: small_prefill, 3: small_decode},
            pd_pairs=[(2, 3)],
        )
        event = ownership_event(
            small_decode.memory,
            GPUHBMEventKind.TURN_RETAIN,
            instance_id=3,
            token_count=17,
        )
        with self.assertRaisesRegex(
                GPUHBMBridgeCapacityError, "required=64"):
            bridge.apply_event(event)
        self.assertEqual(small_decode.memory.npu_used, 128)
        self.assertEqual(bridge.report()["idle_allocations"], [])

        self.retain()
        self.decode.memory.npu_used = self.decode.memory.weight + 32
        release = ownership_event(
            self.decode.memory,
            GPUHBMEventKind.IDLE_RELEASE,
            request_id=2,
            time_ns=11,
            token_count=0,
        )
        with self.assertRaisesRegex(
                GPUHBMBridgeUnderflowError, "dynamic allocation"):
            self.bridge.apply_event(release)
        self.assertEqual(
            len(self.bridge.report()["idle_allocations"]), 1)

    def test_request_capacity_includes_every_materialized_output_token(self):
        decode = finite_scheduler(
            3, "decode",
            allocatable_bytes=192,
            weight_bytes=128,
        )
        prefill = finite_scheduler(2, "prefill")
        bridge = FullModelHBFGPUHBMBridge(
            {2: prefill, 3: decode},
            pd_pairs=[(2, 3)],
        )

        self.assertEqual(
            bridge.validate_pd_decode_prompt_capacity(
                17, decode_instance_id=3),
            64,
        )
        self.assertEqual(
            bridge.validate_pd_decode_request_capacity(
                17, 16, decode_instance_id=3),
            64,
        )
        with self.assertRaisesRegex(
                GPUHBMBridgeCapacityError,
                "terminal_materialized_tokens=33"):
            bridge.validate_pd_decode_request_capacity(
                17, 17, decode_instance_id=3)
        with self.assertRaisesRegex(
                ValueError, "requested_output_tokens"):
            bridge.validate_pd_decode_request_capacity(
                17, 0, decode_instance_id=3)

    def test_event_geometry_and_decode_ownership_are_strict(self):
        invalid = ownership_event(
            self.decode.memory,
            GPUHBMEventKind.TURN_RETAIN,
        )
        invalid = GPUHBMOwnershipEvent(
            **{
                **invalid.__dict__,
                "per_rank_bytes": invalid.per_rank_bytes + 1,
            },
        )
        with self.assertRaisesRegex(
                GPUHBMBridgeError, "exact MemoryModel"):
            self.bridge.apply_event(invalid)
        self.assertEqual(self.decode.memory.npu_used, 128)

        wrong_owner = ownership_event(
            self.prefill.memory,
            GPUHBMEventKind.TURN_RETAIN,
            instance_id=0,
        )
        with self.assertRaisesRegex(
                GPUHBMBridgeStaleEventError, "decode Scheduler"):
            self.bridge.apply_event(wrong_owner)

        self.retain()
        stale = ownership_event(
            self.decode.memory,
            GPUHBMEventKind.IDLE_RELEASE,
            request_id=2,
            time_ns=9,
            token_count=0,
        )
        with self.assertRaisesRegex(
                GPUHBMBridgeStaleEventError, "backwards"):
            self.bridge.apply_event(stale)
        self.assertEqual(
            len(self.bridge.report()["idle_allocations"]), 1)


class FullModelHBFGPUHBMBridgeColocatedTests(unittest.TestCase):
    def test_colocated_claim_is_adopted_without_prefix_double_allocation(self):
        scheduler = finite_scheduler(3, None)
        bridge = FullModelHBFGPUHBMBridge({3: scheduler})
        baseline = scheduler.memory.npu_used
        bridge.apply_event(ownership_event(
            scheduler.memory,
            GPUHBMEventKind.TURN_RETAIN,
            instance_id=3,
            token_count=17,
        ))
        self.assertEqual(scheduler.memory.npu_used, baseline + 64)

        bridge.apply_event(ownership_event(
            scheduler.memory,
            GPUHBMEventKind.RESUME_CLAIM,
            request_id=2,
            instance_id=3,
            time_ns=11,
            token_count=16,
        ))
        # Trimming from 17 logical tokens (two blocks) to 16 (one block)
        # releases exactly one block before Scheduler ownership transfers.
        self.assertEqual(scheduler.memory.npu_used, baseline + 32)
        metadata = bridge.decorate_colocated_continuation(2, {
            "index": 2,
            "session_id": "session-a",
            "prefix_reuse_toks": 16,
            "agentic_kv_hit_tokens": 16,
            "hbf_gpu_required_instance_id": 3,
        })
        request = scheduler.add_request(
            [2, "model", 17, 20, 11, 3],
            metadata=metadata,
            enqueue=False,
        )
        bridge.bind_colocated_continuation(request)
        self.assertEqual(request.num_computed_tokens, 16)
        self.assertEqual(request.agentic_kv_owner_instance_id, 3)

        missing_suffix = scheduler.memory.get_block_kv(
            [request], 1, {request.id: 1})
        self.assertEqual(missing_suffix, 32)
        scheduler.memory.allocate(missing_suffix, Device.NPU)
        self.assertEqual(scheduler.memory.npu_used, baseline + 64)
        report = bridge.report()
        self.assertEqual(report["schema"], GPU_HBM_BRIDGE_SCHEMA)
        self.assertEqual(report["topology"], "colocated")
        self.assertEqual(
            report["adopted_colocated_request_ids"], [2])
        self.assertEqual(
            report["memory_by_instance"][3][
                "bridge_owned_per_rank_bytes"],
            0,
        )

    def test_colocated_metadata_mismatch_and_duplicate_are_rejected(self):
        scheduler = finite_scheduler(3, None)
        bridge = FullModelHBFGPUHBMBridge({3: scheduler})
        bridge.apply_event(ownership_event(
            scheduler.memory,
            GPUHBMEventKind.TURN_RETAIN,
            instance_id=3,
            token_count=16,
        ))
        bridge.apply_event(ownership_event(
            scheduler.memory,
            GPUHBMEventKind.RESUME_CLAIM,
            request_id=2,
            instance_id=3,
            time_ns=11,
            token_count=16,
        ))
        with self.assertRaisesRegex(
                GPUHBMBridgeStaleEventError, "differs"):
            bridge.decorate_colocated_continuation(2, {
                "index": 2,
                "session_id": "session-a",
                "prefix_reuse_toks": 15,
            })
        metadata = bridge.decorate_colocated_continuation(2, {
            "index": 2,
            "session_id": "session-a",
            "prefix_reuse_toks": 16,
        })
        with self.assertRaisesRegex(
                GPUHBMBridgeStaleEventError, "already decorated"):
            bridge.decorate_colocated_continuation(2, metadata)


if __name__ == "__main__":
    unittest.main()

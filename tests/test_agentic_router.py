import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from serving.core.agentic_kv import KVLocation, KVPreparation
from serving.core.memory_model import Device
from serving.core.request import Request
from serving.core.router import Router


class FakeScheduler:
    def __init__(self, instance_id=0, pd_type=None, node_id=0):
        self.pd_type = pd_type
        self.instance_id = instance_id
        self.node_id = node_id
        self.tp_size = 1
        self.pp_size = 1
        self.block_size = 16
        self.fp = 16
        self.kv_cache_dtype = "auto"
        self.max_num_seqs = 128
        self.request = []
        self.inflight = []
        self.enable_prefix_caching = False
        self.model = "model"
        self.memory = SimpleNamespace(
            npu_mem=10_000,
            weight=0,
            npu_used=0,
        )
        self.memory.get_kv = lambda tokens: int(tokens)
        self.memory.get_evict_kv = lambda request: self.memory.get_kv(
            (
                (int(request.num_computed_tokens) + self.block_size - 1)
                // self.block_size
                * self.block_size
            )
        )

        def allocate(byte_count, device):
            if device != Device.NPU:
                raise AssertionError(device)
            if self.memory.npu_used + byte_count > self.memory.npu_mem:
                raise RuntimeError("NPU capacity exceeded")
            self.memory.npu_used += byte_count

        def free(byte_count, device):
            if device != Device.NPU:
                raise AssertionError(device)
            if byte_count < 0 or byte_count > self.memory.npu_used:
                raise RuntimeError(
                    f"invalid NPU free: used={self.memory.npu_used}, "
                    f"bytes={byte_count}")
            self.memory.npu_used -= byte_count

        self.memory.allocate = allocate
        self.memory.free = free
        self.memory_wait_until_ns = None
        self.decode_handoff_claim_pending = False
        self.pd_prefill_reclaimability_generation = 0
        self.added = []
        self.decoded = []
        self.decode_admissions = []

    def add_request(
            self, values, is_init=True, metadata=None, enqueue=True):
        self.added.append((values, metadata))
        request = Request(*values, is_init=is_init)
        if metadata:
            request.session_id = metadata.get("session_id")
            request.sub_request_index = metadata.get("sub_request_index")
            request.ready_time = int(
                metadata.get("ready_time_ns")
                if metadata.get("ready_time_ns") is not None
                else request.arrival
            )
            request.agentic_kv_hit_tokens = int(
                metadata.get("agentic_kv_hit_tokens") or 0
            )
            request.agentic_kv_restore_issue_time_ns = int(
                metadata.get("agentic_kv_restore_issue_time_ns")
                or request.arrival)
            request.agentic_kv_target_hbm_ready_time_ns = int(
                metadata.get("agentic_kv_target_hbm_ready_time_ns")
                or request.ready_time)
            request.agentic_kv_restore_ready_time_ns = int(
                metadata.get("agentic_kv_restore_ready_time_ns")
                or request.ready_time)
            request.agentic_kv_fresh_prompt_tokens = int(
                metadata.get("agentic_kv_fresh_prompt_tokens") or 0)
            request.agentic_kv_overlap_cutoff_tokens = metadata.get(
                "agentic_kv_overlap_cutoff_tokens")
            request.agentic_kv_async_decode_join = bool(
                metadata.get("agentic_kv_async_decode_join", False))
            request.agentic_kv_restore_ns = int(
                metadata.get("agentic_kv_restore_ns") or 0)
            request.agentic_kv_owner_gate_ns = int(
                metadata.get("agentic_kv_owner_gate_ns") or 0)
            request.pd_pair_fifo_wait_ns = int(
                metadata.get("pd_pair_fifo_wait_ns") or 0)
            request.agentic_kv_prepare_boundary_wait_ns = int(
                metadata.get(
                    "agentic_kv_prepare_boundary_wait_ns") or 0)
            request.agentic_kv_source_demotion_join_wait_ns = int(
                metadata.get(
                    "agentic_kv_source_demotion_join_wait_ns") or 0)
            request.agentic_kv_restore_gate_start_ns = int(
                metadata.get("agentic_kv_restore_gate_start_ns") or 0)
            request.agentic_kv_restore_gate_wait_ns = int(
                metadata.get("agentic_kv_restore_gate_wait_ns") or 0)
            request.num_computed_tokens = request.agentic_kv_hit_tokens
            request.agentic_kv_owner_instance_id = metadata.get(
                "agentic_kv_owner_instance_id")
            request.agentic_kv_retained_instance_id = metadata.get(
                "agentic_kv_retained_instance_id")
            request.agentic_kv_retained_per_rank_bytes = int(
                metadata.get("agentic_kv_retained_per_rank_bytes") or 0
            )
        if enqueue:
            self.enqueue_request(request)
        return request

    def enqueue_request(self, request):
        self.request.append(request)
        self.request.sort(key=lambda item: (item.ready_time, item.id))

    def censor_queued_request(self, request, cutoff_time_ns):
        del cutoff_time_ns
        if request not in self.request:
            raise RuntimeError("request is not queued")
        if request.agentic_kv_owner_instance_id != self.instance_id:
            raise RuntimeError("request owner mismatch")
        released = (
            0 if request.recompute_target_tokens is not None
            else int(self.memory.get_evict_kv(request))
        )
        if released:
            self.memory.free(released, Device.NPU)
        self.request.remove(request)
        request.agentic_kv_owner_instance_id = None
        return {
            "request_id": int(request.id),
            "session_id": str(request.session_id),
            "instance_id": int(self.instance_id),
            "released_per_rank_bytes": released,
        }

    @staticmethod
    def decode_handoff_hbm_bytes(request):
        return int(getattr(request, "decode_handoff_bytes", 100))

    def add_decode(
            self, request, admitted_hbm_bytes=None,
            preallocated_hbm_bytes=None, completion_time_ns=None):
        del completion_time_ns
        if request.agentic_kv_owner_instance_id is not None:
            raise RuntimeError("source owner was not released")
        request.instance_id = self.instance_id
        request.agentic_kv_owner_instance_id = self.instance_id
        self.decoded.append(request)
        self.decode_admissions.append(
            preallocated_hbm_bytes
            if preallocated_hbm_bytes is not None
            else admitted_hbm_bytes
        )
        return None


class FakeTierManager:
    def __init__(self):
        self.synchronous_swap_enabled = False
        self.async_decode_join_enabled = False
        self.prepared = []
        self.started = []
        self.ended = []
        self.claims = {}
        self.pd_admissions = []
        self.pd_prefill_admissions = []
        self.pd_launch_admissions = []
        self.async_restore_gates = []
        self.classified = []
        self.sync_prepare_instances = ()
        self.sync_prepare_locks = {}
        self.sync_hbm_boundary_instances = set()
        self.return_residency_snapshots = []
        self.censor_audits = {}

    def snapshot_return_residency(self, session_id, return_time_ns):
        self.return_residency_snapshots.append(
            (str(session_id), int(return_time_ns)))
        return KVLocation.CPU

    def prepare_request(self, **kwargs):
        self.prepared.append(kwargs)
        release = kwargs["release_time_ns"]
        pair_wait = int(kwargs.get("pd_pair_fifo_wait_ns") or 0)
        boundary_wait = int(
            kwargs.get("prepare_boundary_wait_ns") or 0)
        restore_issue = release + pair_wait + boundary_wait
        return KVPreparation(
            hit_tokens=80,
            recompute_tokens=20,
            source=KVLocation.CPU,
            restore_ns=123,
            ready_time_ns=restore_issue + 123,
            restored_bytes=1000,
            owner_gate_ns=pair_wait + boundary_wait + 123,
            restore_issue_time_ns=restore_issue,
            target_hbm_ready_time_ns=restore_issue + 23,
            restore_ready_time_ns=restore_issue + 123,
            pd_pair_fifo_wait_ns=pair_wait,
            prepare_boundary_wait_ns=boundary_wait,
            hbm_admission_wait_ns=23,
            transient_dram_capacity_wait_ns=7,
            queue_wait_ns=30,
            service_ns=70,
            residency_at_return=KVLocation(
                kwargs.get('residency_at_return', KVLocation.CPU)),
        )

    def synchronous_prepare_instances(
            self, session_id, target_instance_id, reuse_tokens, now_ns):
        return tuple(self.sync_prepare_instances)

    def acquire_synchronous_prepare_lock(
            self, request_id, instance_ids, session_id=None):
        self.sync_prepare_locks[int(request_id)] = tuple(instance_ids)

    def release_synchronous_prepare_lock(self, request_id):
        self.sync_prepare_locks.pop(int(request_id), None)

    def synchronous_hbm_reclaim_needs_boundary(
            self, instance_id, needed_per_rank_bytes, now_ns):
        return instance_id in self.sync_hbm_boundary_instances

    def on_idle_start(
            self, request, completion_time_ns, release_time_ns, **metadata):
        self.started.append((
            request.id, completion_time_ns, release_time_ns, metadata))

    def end_session(self, session_id, now_ns=None):
        self.ended.append(session_id)

    def censor_session(self, session_id, cutoff_ns):
        del cutoff_ns
        self.end_session(session_id)
        return self.censor_audits.get(str(session_id))

    def claim_active_hbm_reclaim(
            self, instance_id, needed_per_rank_bytes, now_ns,
            owner_kind="legacy", owner_id=None):
        self.claims[instance_id] = SimpleNamespace(
            instance_id=instance_id,
            per_rank_bytes=needed_per_rank_bytes,
            ready_ns=int(now_ns),
            owner_kind=owner_kind,
            owner_id=owner_id,
        )
        return int(now_ns)

    def consume_active_hbm_reclaim(
            self, instance_id, now_ns, owner_kind=None, owner_id=None):
        return self.claims.pop(instance_id, None)

    def active_hbm_reclaim_claim(self, instance_id):
        return self.claims.get(int(instance_id))

    def cancel_active_hbm_reclaim(self, instance_id, now_ns):
        del now_ns
        return self.claims.pop(int(instance_id), None)

    def advance(self, now_ns):
        return None

    def record_agentic_request(self, request):
        self.classified.append(request.id)
    def record_pd_decode_receive_admission(self, *args):
        self.pd_admissions.append(args)

    def record_pd_prefill_admission(self, *args):
        self.pd_prefill_admissions.append(args)

    def record_pd_launch_admission(self, *args):
        self.pd_launch_admissions.append(args)

    def record_async_restore_gate(self, request, gate_start_ns):
        wait_ns = max(
            0,
            int(request.agentic_kv_restore_ready_time_ns)
            - int(gate_start_ns),
        )
        request.agentic_kv_restore_gate_start_ns = int(gate_start_ns)
        request.agentic_kv_restore_gate_wait_ns = wait_ns
        request.agentic_kv_restore_gate_recorded = True
        self.async_restore_gates.append((request.id, int(gate_start_ns), wait_ns))


class FakeFabricBoundaryManager(FakeTierManager):
    """Models controller-dispatched ownership separately from formed batches."""

    def __init__(self):
        super().__init__()
        self.fabric_boundary_instances = ()
        self.fabric_busy = False

    def prepare_boundary_instances(
            self, session_id, target_instance_id, reuse_tokens, now_ns):
        return tuple(self.fabric_boundary_instances)

    def prepare_boundary_busy(self, instance_ids):
        return bool(instance_ids) and self.fabric_busy

    def acquire_prepare_lock(
            self, request_id, instance_ids, session_id=None):
        self.acquire_synchronous_prepare_lock(
            request_id, instance_ids, session_id=session_id)

    def release_prepare_lock(self, request_id):
        self.release_synchronous_prepare_lock(request_id)


class FakeDecodeAdmissionManager(FakeTierManager):
    def __init__(self, ready_ns):
        super().__init__()
        self.ready_ns = ready_ns
        self.claims = {}
        self.claim_calls = []
        self.consume_calls = []

    def claim_active_hbm_reclaim(
            self, instance_id, needed_per_rank_bytes, now_ns,
            owner_kind="legacy", owner_id=None):
        self.claim_calls.append(
            (instance_id, needed_per_rank_bytes, now_ns))
        configured_ready_ns = (
            self.ready_ns[instance_id]
            if isinstance(self.ready_ns, dict)
            else self.ready_ns
        )
        if instance_id not in self.claims:
            self.claims[instance_id] = SimpleNamespace(
                instance_id=instance_id,
                per_rank_bytes=needed_per_rank_bytes,
                ready_ns=max(int(now_ns), int(configured_ready_ns)),
                owner_kind=owner_kind,
                owner_id=owner_id,
            )
        return self.claims[instance_id].ready_ns

    def consume_active_hbm_reclaim(
            self, instance_id, now_ns, owner_kind=None, owner_id=None):
        self.consume_calls.append((instance_id, now_ns))
        claim = self.claims.get(instance_id)
        if claim is None or int(now_ns) < claim.ready_ns:
            return None
        del self.claims[instance_id]
        return claim


class FinishedRequest:
    id = 0
    session_id = "session-a"
    num_computed_tokens = 120


class PrefillRequest(Request):
    def __init__(self, request_id, session_id="session-a", instance_id=0):
        super().__init__(
            request_id, "model", 100, 101, 0, instance_id)
        self.id = request_id
        self.session_id = session_id
        self.instance_id = instance_id
        self.model = "model"
        self.agentic_kv_owner_instance_id = None
        self.agentic_kv_retained_instance_id = None
        self.agentic_kv_retained_per_rank_bytes = 0
        self.pd_decode_target_instance_id = None
        self.pd_decode_full_per_rank_bytes = 0
        self.pd_decode_reserved_per_rank_bytes = 0
        self.pd_decode_admission_enqueued_ns = 0
        self.pd_decode_capacity_ready_ns = 0
        self.pd_decode_capacity_wait_ns = 0
        self.pd_decode_admission_ready_ns = 0
        self.pd_decode_admission_wait_ns = 0
        self.pd_decode_admission_critical_wait_ns = 0
        self.pd_prefill_full_per_rank_bytes = 0
        self.pd_prefill_reserved_per_rank_bytes = 0
        self.pd_prefill_admission_enqueued_ns = 0
        self.pd_prefill_capacity_ready_ns = 0
        self.pd_prefill_capacity_wait_ns = 0
        self.pd_prefill_admission_ready_ns = 0
        self.pd_prefill_admission_wait_ns = 0
        self.pd_prefill_admission_critical_wait_ns = 0
        self.pd_prefill_preallocated_per_rank_bytes = 0
        self.pd_launch_admission_ready_ns = 0
        self.pd_launch_admission_wait_ns = 0
        self.pd_launch_admission_critical_wait_ns = 0
        self.original_input = 100
        self.ready_time = 0
        self.agentic_kv_source = None
        self.agentic_kv_residency_at_return = None
        self.agentic_kv_hit_tokens = 0
        self.agentic_kv_restore_ns = 0
        self.agentic_kv_restore_ready_time_ns = 0
        self.agentic_kv_overlap_cutoff_tokens = None
        self.agentic_kv_async_decode_join = False
        self.agentic_kv_restore_gate_start_ns = 0
        self.agentic_kv_restore_gate_wait_ns = 0
        self.agentic_kv_restore_gate_recorded = False
        self.return_gap_type = "session_start"
        self.return_gap_source = "session_start"
        self.decode_handoff_bytes = 100


class AgenticRouterTest(unittest.TestCase):
    def test_late_callback_uses_physical_frontier_for_due_restore(self):
        class FrontierManager(FakeTierManager):
            def __init__(self):
                super().__init__()
                self.frontier = 0
                self.timeline = []

            def snapshot_return_residency(
                    self, session_id, return_time_ns):
                if int(return_time_ns) < self.frontier:
                    raise RuntimeError("snapshot regressed")
                self.frontier = int(return_time_ns)
                self.timeline.append(("snapshot", str(session_id),
                                      int(return_time_ns)))
                return super().snapshot_return_residency(
                    session_id, return_time_ns)

            def prepare_request(self, **kwargs):
                operation_ns = int(kwargs["operation_time_ns"])
                if operation_ns < self.frontier:
                    raise RuntimeError("operation regressed")
                self.frontier = operation_ns
                self.timeline.append(("prepare", kwargs["session_id"],
                                      operation_ns))
                return super().prepare_request(**kwargs)

        scheduler = FakeScheduler()
        manager = FrontierManager()
        router = Router(
            1, [scheduler], 0, "RR", agentic_kv_manager=manager)
        router._pending_requests = [
            {
                "index": request_id,
                "input_toks": 100,
                "output_toks": 101,
                "arrival_time_ns": 100,
                "session_id": session_id,
                "sub_request_index": 1,
                "prefix_reuse_toks": 80,
            }
            for request_id, session_id in ((1, "older-a"), (2, "older-b"))
        ]

        self.assertEqual(
            router.route_arrived_requests(
                100, operation_time_ns=101),
            0,
        )
        self.assertEqual(
            manager.timeline[:2],
            [("snapshot", "older-a", 100),
             ("snapshot", "older-b", 100)],
        )
        self.assertEqual(
            [attempt["operation_time_ns"] for attempt in manager.prepared],
            [101, 101],
        )
        self.assertEqual(
            [attempt["prepare_boundary_wait_ns"]
             for attempt in manager.prepared],
            [1, 1],
        )
        self.assertTrue(all(
            request["agentic_kv_owner_gate_ns"] == 124
            for request in router._pending_requests[router._pending_idx:]
        ))

    def test_routing_rejects_operation_before_arrival_cutoff(self):
        router = Router(1, [FakeScheduler()], 0, "RR")
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            router.route_arrived_requests(101, operation_time_ns=100)

    def _one_fresh_async_router(self, session_id):
        scheduler = FakeScheduler()
        manager = FakeTierManager()
        manager.async_decode_join_enabled = True
        manager.prepare_request = lambda **kwargs: KVPreparation(
            hit_tokens=139,
            recompute_tokens=0,
            source=KVLocation.SSD,
            restore_ns=123,
            ready_time_ns=kwargs["release_time_ns"] + 123,
            restored_bytes=1000,
            restore_issue_time_ns=kwargs["release_time_ns"],
            target_hbm_ready_time_ns=kwargs["release_time_ns"] + 23,
            restore_ready_time_ns=kwargs["release_time_ns"] + 123,
            hbm_admission_wait_ns=23,
            queue_wait_ns=30,
            service_ns=70,
        )
        router = Router(
            1, [scheduler], 0, "RR", agentic_kv_manager=manager)
        workload = {
            "session_id": session_id,
            "arrival_time_ns": 0,
            "sub_requests": [
                {"input_toks": 100, "output_toks": 2,
                 "tool_duration_ns": 1000},
                {"input_toks": 140, "output_toks": 1,
                 "tool_duration_ns": 0, "prefix_reuse_toks": 100},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(workload) + "\n", encoding="utf-8")
            router.load_requests(str(path))
        router.route_arrived_requests(0)
        router.notify_request_completed(FinishedRequest(), 5000)
        return router, scheduler, manager

    def test_prefix_reuse_preserves_token_id_positions(self):
        complete = {
            "input_toks": 4,
            "output_toks": 2,
            "input_tok_ids": [10, 11],
            "output_tok_ids": [12],
        }
        following = {
            "input_toks": 4,
            "input_tok_ids": [10, 11, 12, 99],
        }
        reuse, source = Router._prefix_reuse(
            complete, following, FinishedRequest())
        self.assertEqual((reuse, source), (2, "exact"))

        # Missing output IDs must not cause the last verified input token to
        # be removed when the completed cache contains only input KV.
        complete = {
            "input_toks": 4,
            "output_toks": 1,
            "input_tok_ids": [0, 1, 2, 3],
            "output_tok_ids": [],
        }
        following = {
            "input_toks": 4,
            "input_tok_ids": [0, 1, 2, 3],
        }
        reuse, source = Router._prefix_reuse(
            complete, following, FinishedRequest())
        self.assertEqual((reuse, source), (4, "exact"))

    def test_agentic_session_is_sticky_without_tier_manager(self):
        schedulers = [FakeScheduler(0), FakeScheduler(1)]
        router = Router(2, schedulers, 0, "RR")
        workload = {
            "session_id": "sticky",
            "arrival_time_ns": 0,
            "sub_requests": [
                {"input_toks": 10, "output_toks": 2,
                 "tool_duration_ns": 100},
                {"input_toks": 12, "output_toks": 2,
                 "tool_duration_ns": 0},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(workload) + "\n", encoding="utf-8")
            router.load_requests(str(path))

        self.assertEqual(router.route_arrived_requests(0), 1)
        router.notify_request_completed(FinishedRequest(), 1000)
        self.assertEqual(router.route_arrived_requests(1100), 1)
        self.assertEqual(len(schedulers[0].added), 2)
        self.assertEqual(len(schedulers[1].added), 0)

    def test_restore_delays_only_continuations_and_metadata_reaches_scheduler(self):
        scheduler = FakeScheduler()
        manager = FakeTierManager()
        router = Router(1, [scheduler], 0, "RR", agentic_kv_manager=manager)
        workload = {
            "session_id": "session-a",
            "arrival_time_ns": 0,
            "sub_requests": [
                {
                    "input_toks": 100,
                    "output_toks": 20,
                    "tool_duration_ns": 1000,
                    "inter_turn_gap_type": "human",
                    "tool_wait_source": "request_ready_boundary",
                    "prefix_reuse_toks": 0,
                },
                {
                    "input_toks": 140,
                    "output_toks": 10,
                    "tool_duration_ns": 0,
                    # This is the second call's outgoing class and must not
                    # be attached to the second call itself.
                    "inter_turn_gap_type": "tool",
                    "prefix_reuse_toks": 100,
                    "prefix_reuse_source": "reported",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(workload) + "\n", encoding="utf-8")
            router.load_requests(str(path))

        self.assertEqual(router.route_arrived_requests(0), 1)
        self.assertEqual(manager.prepared, [])
        self.assertEqual(scheduler.added[0][1]["session_id"], "session-a")
        self.assertEqual(
            scheduler.added[0][1]["return_gap_type"], "session_start")

        router.notify_request_completed(FinishedRequest(), 5000)
        self.assertEqual(manager.started, [(
            0,
            5000,
            6000,
            {
                "return_gap_type": "human",
                "return_gap_source": "request_ready_boundary",
            },
        )])
        self.assertEqual(router.route_arrived_requests(5999), 0)
        self.assertEqual(router.route_arrived_requests(6000), 0)
        self.assertEqual(len(manager.prepared), 1)
        self.assertEqual(manager.prepared[0]["sub_request_index"], 1)
        self.assertEqual(router.get_next_pending_arrival(), 6123)
        self.assertEqual(router.route_arrived_requests(6123), 1)
        self.assertEqual(
            manager.prepared[0]["request_id"], scheduler.added[1][0][0])
        metadata = scheduler.added[1][1]
        self.assertEqual(metadata["agentic_kv_hit_tokens"], 80)
        self.assertEqual(metadata["agentic_kv_recompute_tokens"], 20)
        self.assertEqual(metadata["agentic_kv_source"], "cpu")
        self.assertEqual(metadata["agentic_kv_restore_ns"], 123)
        self.assertEqual(
            metadata["agentic_kv_source_demotion_join_wait_ns"], 0)
        self.assertEqual(
            metadata["agentic_kv_hbm_admission_wait_ns"], 23)
        self.assertEqual(
            metadata["agentic_kv_transient_dram_capacity_wait_ns"], 7)
        self.assertEqual(
            metadata["agentic_kv_restore_queue_wait_ns"], 30)
        self.assertEqual(metadata["agentic_kv_restore_service_ns"], 70)
        self.assertEqual(metadata["return_gap_type"], "human")
        self.assertEqual(
            metadata["return_gap_source"], "request_ready_boundary")
        self.assertEqual(metadata["return_gap_ns"], 1000)

    def test_pre_admission_restore_does_not_hold_back_ready_peer_request(self):
        scheduler = FakeScheduler()
        manager = FakeTierManager()
        router = Router(
            1, [scheduler], 0, "RR", agentic_kv_manager=manager)
        session = {
            "session_id": "cold-owner",
            "arrival_time_ns": 0,
            "sub_requests": [
                {
                    "input_toks": 100,
                    "output_toks": 2,
                    "tool_duration_ns": 1000,
                    "inter_turn_gap_type": "tool",
                },
                {
                    "input_toks": 140,
                    "output_toks": 2,
                    "tool_duration_ns": 0,
                    "prefix_reuse_toks": 100,
                },
            ],
        }
        ready_peer = {
            "input_toks": 8,
            "output_toks": 1,
            "arrival_time_ns": 6050,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(
                json.dumps(session) + "\n" + json.dumps(ready_peer) + "\n",
                encoding="utf-8",
            )
            router.load_requests(str(path))

        self.assertEqual(router.route_arrived_requests(0), 1)
        router.notify_request_completed(FinishedRequest(), 5000)

        # The cold owner issues its restore at request-ready time but remains
        # outside the scheduler until all KV has arrived at 6123.
        self.assertEqual(router.route_arrived_requests(6000), 0)
        self.assertEqual(manager.prepared[0]["release_time_ns"], 6000)
        self.assertEqual([values[0][0] for values in scheduler.added], [0])

        # A peer request that becomes ready while the load is in flight still
        # reaches the continuous-batching scheduler immediately.
        self.assertEqual(router.route_arrived_requests(6050), 1)
        self.assertEqual([values[0][0] for values in scheduler.added], [0, 2])
        self.assertEqual(router.get_next_pending_arrival(), 6123)

        self.assertEqual(router.route_arrived_requests(6123), 1)
        self.assertEqual(
            [values[0][0] for values in scheduler.added], [0, 2, 1])
        cold_metadata = scheduler.added[-1][1]
        self.assertEqual(cold_metadata["ready_time_ns"], 6123)
        self.assertIsNone(
            cold_metadata["agentic_kv_overlap_cutoff_tokens"])
        self.assertFalse(cold_metadata["agentic_kv_async_decode_join"])

    def test_temporary_restore_capacity_deferral_retries_without_head_block(self):
        scheduler = FakeScheduler()
        manager = FakeTierManager()
        attempts = []

        def prepare(**kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                return None
            release = int(kwargs["release_time_ns"])
            operation = int(kwargs["operation_time_ns"])
            return KVPreparation(
                hit_tokens=80,
                recompute_tokens=20,
                source=KVLocation.CPU,
                restore_ns=operation - release,
                ready_time_ns=operation,
                restored_bytes=1000,
                restore_issue_time_ns=release,
                target_hbm_ready_time_ns=operation,
                restore_ready_time_ns=operation,
                hbm_admission_wait_ns=operation - release,
            )

        manager.prepare_request = prepare
        router = Router(
            1, [scheduler], 0, "RR", agentic_kv_manager=manager)
        router._pending_requests = [
            {
                "index": 1,
                "input_toks": 100,
                "output_toks": 110,
                "arrival_time_ns": 100,
                "session_id": "cold",
                "sub_request_index": 1,
                "prefix_reuse_toks": 80,
            },
            {
                "index": 2,
                "input_toks": 10,
                "output_toks": 11,
                "arrival_time_ns": 100,
            },
        ]

        self.assertEqual(router.route_arrived_requests(100), 1)
        self.assertEqual([values[0][0] for values in scheduler.added], [2])
        self.assertIsNone(router.get_next_pending_arrival())
        # The cold owner is no longer in the arrival list, but remains a
        # causal capacity waiter until a reclaim-generation change retries
        # preparation. It must therefore keep the simulation alive.
        self.assertTrue(router.has_pending_requests())
        self.assertEqual(len(router._pending_capacity_preparations), 1)
        scheduler.memory.npu_used += 1
        self.assertEqual(router.route_arrived_requests(101), 1)
        self.assertEqual(
            [values[0][0] for values in scheduler.added], [2, 1])
        self.assertEqual(attempts[0]["release_time_ns"], 100)
        self.assertEqual(attempts[1]["release_time_ns"], 100)
        self.assertEqual(attempts[1]["operation_time_ns"], 101)

    def test_async_decode_join_exposes_prefill_before_restore_completion(self):
        scheduler = FakeScheduler()
        manager = FakeTierManager()
        manager.async_decode_join_enabled = True
        router = Router(
            1, [scheduler], 0, "RR", agentic_kv_manager=manager)
        workload = {
            "session_id": "async-join",
            "arrival_time_ns": 0,
            "sub_requests": [
                {
                    "input_toks": 100,
                    "output_toks": 2,
                    "tool_duration_ns": 1000,
                    "inter_turn_gap_type": "tool",
                },
                {
                    "input_toks": 140,
                    "output_toks": 2,
                    "tool_duration_ns": 0,
                    "prefix_reuse_toks": 100,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(workload) + "\n", encoding="utf-8")
            router.load_requests(str(path))

        self.assertEqual(router.route_arrived_requests(0), 1)
        router.notify_request_completed(FinishedRequest(), 5000)
        self.assertEqual(router.route_arrived_requests(6000), 0)
        self.assertEqual(manager.prepared[0]["release_time_ns"], 6000)
        self.assertEqual(router.get_next_pending_arrival(), 6023)
        self.assertEqual(router.route_arrived_requests(6023), 1)
        metadata = scheduler.added[1][1]
        self.assertEqual(metadata["ready_time_ns"], 6023)
        self.assertEqual(metadata["agentic_kv_restore_ready_time_ns"], 6123)
        self.assertEqual(metadata["agentic_kv_overlap_cutoff_tokens"], 139)
        self.assertEqual(metadata["agentic_kv_fresh_prompt_tokens"], 60)
        self.assertTrue(metadata["agentic_kv_async_decode_join"])

    def test_async_decode_join_with_only_final_prompt_token_waits_in_full(self):
        router, scheduler, _ = self._one_fresh_async_router(
            "one-token-join")
        self.assertEqual(router.route_arrived_requests(6000), 0)
        self.assertEqual(router.get_next_pending_arrival(), 6123)
        self.assertEqual(router.route_arrived_requests(6122), 0)
        self.assertEqual(router.route_arrived_requests(6123), 1)
        metadata = scheduler.added[1][1]
        self.assertIsNone(metadata["agentic_kv_overlap_cutoff_tokens"])
        self.assertEqual(metadata["agentic_kv_restore_gate_wait_ns"], 123)

    def test_one_fresh_gate_late_route_hides_completed_restore(self):
        router, scheduler, _ = self._one_fresh_async_router(
            "one-token-late")

        self.assertEqual(router.route_arrived_requests(6200), 1)
        metadata = scheduler.added[1][1]
        self.assertEqual(metadata["agentic_kv_restore_ns"], 123)
        self.assertEqual(metadata["agentic_kv_restore_gate_wait_ns"], 0)
        self.assertEqual(metadata["agentic_kv_restore_gate_start_ns"], 0)

    def test_one_fresh_gate_counts_only_wait_after_route_observation(self):
        router, scheduler, _ = self._one_fresh_async_router(
            "one-token-partial")

        self.assertEqual(router.route_arrived_requests(6050), 0)
        self.assertEqual(router.get_next_pending_arrival(), 6123)
        self.assertEqual(router.route_arrived_requests(6123), 1)
        metadata = scheduler.added[1][1]
        self.assertEqual(metadata["agentic_kv_restore_ns"], 123)
        self.assertEqual(metadata["agentic_kv_restore_gate_wait_ns"], 73)
        self.assertEqual(metadata["agentic_kv_restore_gate_start_ns"], 6050)

    def test_sync_restore_waits_for_iteration_boundary_before_prepare(self):
        scheduler = FakeScheduler()
        manager = FakeTierManager()
        manager.sync_prepare_instances = (0,)
        router = Router(
            1, [scheduler], 0, "RR", agentic_kv_manager=manager)
        workload = {
            "session_id": "session-sync",
            "arrival_time_ns": 0,
            "sub_requests": [
                {
                    "input_toks": 100,
                    "output_toks": 20,
                    "tool_duration_ns": 1000,
                },
                {
                    "input_toks": 140,
                    "output_toks": 10,
                    "tool_duration_ns": 0,
                    "prefix_reuse_toks": 100,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(workload) + "\n", encoding="utf-8")
            router.load_requests(str(path))

        self.assertEqual(router.route_arrived_requests(0), 1)
        scheduler.request.clear()
        scheduler.inflight.append(object())
        router.notify_request_completed(FinishedRequest(), 5000)

        self.assertEqual(router.route_arrived_requests(6000), 0)
        self.assertEqual(manager.prepared, [])
        self.assertTrue(router.has_pending_requests())
        self.assertEqual(manager.sync_prepare_locks, {1: (0,)})

        scheduler.inflight.clear()
        self.assertEqual(router.route_arrived_requests(7000), 0)
        self.assertEqual(len(manager.prepared), 1)
        self.assertEqual(manager.prepared[0]["release_time_ns"], 6000)
        self.assertEqual(
            manager.prepared[0]["prepare_boundary_wait_ns"], 1000)
        self.assertEqual(manager.sync_prepare_locks, {})
        self.assertEqual(router.get_next_pending_arrival(), 7123)
        self.assertEqual(router.route_arrived_requests(7123), 1)

    def test_direct_fabric_boundary_ignores_formed_undispatched_dp_batch(self):
        scheduler = FakeScheduler()
        manager = FakeFabricBoundaryManager()
        manager.fabric_boundary_instances = (0,)
        router = Router(
            1, [scheduler], 0, "RR", agentic_kv_manager=manager)
        workload = {
            "session_id": "session-fabric",
            "arrival_time_ns": 0,
            "sub_requests": [
                {
                    "input_toks": 100,
                    "output_toks": 20,
                    "tool_duration_ns": 1000,
                },
                {
                    "input_toks": 140,
                    "output_toks": 10,
                    "tool_duration_ns": 0,
                    "prefix_reuse_toks": 100,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(workload) + "\n", encoding="utf-8")
            router.load_requests(str(path))

        self.assertEqual(router.route_arrived_requests(0), 1)
        scheduler.request.clear()
        # This mirrors a batch already formed by Scheduler but still parked in
        # main.dp_pending. It owns no controller-dispatched fabric window.
        scheduler.inflight.append(object())
        manager.fabric_busy = False
        router.notify_request_completed(FinishedRequest(), 5000)

        self.assertEqual(router.route_arrived_requests(6000), 0)
        self.assertEqual(len(manager.prepared), 1)
        self.assertEqual(manager.prepared[0]["release_time_ns"], 6000)
        self.assertEqual(manager.sync_prepare_locks, {})
        self.assertEqual(router.get_next_pending_arrival(), 6123)

    def test_direct_fabric_actual_owner_lock_releases_at_callback_boundary(self):
        scheduler = FakeScheduler()
        manager = FakeFabricBoundaryManager()
        manager.fabric_boundary_instances = (0,)
        manager.fabric_busy = True
        router = Router(
            1, [scheduler], 0, "RR", agentic_kv_manager=manager)
        workload = {
            "session_id": "session-fabric-running",
            "arrival_time_ns": 0,
            "sub_requests": [
                {
                    "input_toks": 100,
                    "output_toks": 20,
                    "tool_duration_ns": 1000,
                },
                {
                    "input_toks": 140,
                    "output_toks": 10,
                    "tool_duration_ns": 0,
                    "prefix_reuse_toks": 100,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(workload) + "\n", encoding="utf-8")
            router.load_requests(str(path))

        self.assertEqual(router.route_arrived_requests(0), 1)
        scheduler.request.clear()
        router.notify_request_completed(FinishedRequest(), 5000)
        self.assertEqual(router.route_arrived_requests(6000), 0)
        self.assertEqual(manager.prepared, [])
        self.assertEqual(manager.sync_prepare_locks, {1: (0,)})

        manager.fabric_busy = False
        self.assertEqual(router.route_arrived_requests(7000), 0)
        self.assertEqual(len(manager.prepared), 1)
        self.assertEqual(manager.prepared[0]["release_time_ns"], 6000)
        self.assertEqual(
            manager.prepared[0]["prepare_boundary_wait_ns"], 1000)
        self.assertEqual(manager.sync_prepare_locks, {})

    def test_sync_restore_does_not_backdate_a_late_idle_callback(self):
        scheduler = FakeScheduler()
        manager = FakeTierManager()
        manager.sync_prepare_instances = (0,)
        router = Router(
            1, [scheduler], 0, "RR", agentic_kv_manager=manager)
        workload = {
            "session_id": "session-late",
            "arrival_time_ns": 0,
            "sub_requests": [
                {
                    "input_toks": 100,
                    "output_toks": 20,
                    "tool_duration_ns": 1000,
                },
                {
                    "input_toks": 140,
                    "output_toks": 10,
                    "tool_duration_ns": 0,
                    "prefix_reuse_toks": 100,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(workload) + "\n", encoding="utf-8")
            router.load_requests(str(path))

        self.assertEqual(router.route_arrived_requests(0), 1)
        scheduler.request.clear()
        router.notify_request_completed(FinishedRequest(), 5000)

        self.assertEqual(router.route_arrived_requests(7000), 0)
        self.assertEqual(manager.prepared[0]["release_time_ns"], 6000)
        self.assertEqual(
            manager.prepared[0]["prepare_boundary_wait_ns"], 1000)
        self.assertEqual(router.get_next_pending_arrival(), 7123)

    def test_pd_continuation_prepares_on_prefill_and_keeps_decode_affinity(self):
        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = FakeTierManager()
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        workload = {
            "session_id": "session-a",
            "arrival_time_ns": 0,
            "sub_requests": [
                {"input_toks": 100, "output_toks": 20,
                 "tool_duration_ns": 1000},
                {"input_toks": 140, "output_toks": 10,
                 "tool_duration_ns": 0, "prefix_reuse_toks": 100},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(workload) + "\n", encoding="utf-8")
            router.load_requests(str(path))

        self.assertEqual(router.route_arrived_requests(0), 1)
        self.assertEqual(len(prefill.request), 1)
        first = prefill.request.pop()
        self.assertEqual(first.pd_decode_target_instance_id, 1)
        self.assertEqual(decode.memory.npu_used, 0)
        first.num_computed_tokens = first.original_input
        first.agentic_kv_owner_instance_id = None
        before_handoff = decode.memory.npu_used
        router.transfer_prefill_request([first], current_time_ns=100)
        self.assertEqual(decode.memory.npu_used, before_handoff)

        # Stand in for normal decode completion, which frees the first active
        # allocation before the next turn is released.
        decode.memory.npu_used = 0
        completed = FinishedRequest()
        completed.instance_id = 1
        router.notify_request_completed(completed, 5000)
        self.assertEqual(router.route_arrived_requests(6000), 0)
        self.assertEqual(manager.prepared[0]["instance_id"], 0)
        self.assertEqual(router.route_arrived_requests(6123), 1)
        self.assertEqual(
            prefill.added[1][1]["agentic_kv_owner_instance_id"], 0)
        second = prefill.request.pop()
        second.num_computed_tokens = second.original_input
        second.agentic_kv_owner_instance_id = None
        before_handoff = decode.memory.npu_used
        router.transfer_prefill_request([second], current_time_ns=6200)
        self.assertEqual(decode.memory.npu_used, before_handoff)
        self.assertEqual([request.id for request in decode.decoded], [0, 1])

    def test_sync_pd_slack_admission_does_not_drain_running_iterations(self):
        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        prefill.inflight.append(object())
        decode.inflight.append(object())
        manager = FakeTierManager()
        manager.synchronous_swap_enabled = True
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        workload = {
            "session_id": "session-slack",
            "arrival_time_ns": 0,
            "sub_requests": [{
                "input_toks": 100,
                "output_toks": 20,
                "tool_duration_ns": 0,
            }],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(workload) + "\n", encoding="utf-8")
            router.load_requests(str(path))

        self.assertEqual(router.route_arrived_requests(0), 1)
        self.assertEqual(len(prefill.request), 1)
        self.assertEqual(len(prefill.inflight), 1)
        self.assertEqual(len(decode.inflight), 1)
        self.assertEqual(manager.sync_prepare_locks, {})

    def test_pd_decode_handoff_consumes_immediate_hbm_claim(self):
        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = FakeDecodeAdmissionManager(ready_ns=100)
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        request = PrefillRequest(7)
        request.original_input = 256

        router._stage_pd_receive_admission(request, prefill, 100)
        self.assertTrue(router.has_pending_decode_handoffs())
        self.assertEqual(decode.decoded, [])

        self.assertEqual(router.process_pending_decode_handoffs(100), 1)
        self.assertFalse(router.has_pending_decode_handoffs())
        self.assertEqual(prefill.request, [request])
        self.assertEqual(prefill.memory.npu_used, 0)
        self.assertEqual(decode.memory.npu_used, 0)
        self.assertEqual(manager.claim_calls, [])
        self.assertTrue(router.admit_pd_prefill_chunk(
            prefill, request, 256, 100))
        self.assertEqual(prefill.memory.npu_used, 256)
        self.assertEqual(decode.memory.npu_used, 256)
        self.assertEqual(
            manager.claim_calls,
            [(0, 256, 100), (1, 256, 100)],
        )
        self.assertEqual(
            manager.consume_calls,
            [(0, 100), (1, 100)],
        )
        self.assertEqual(
            request.pd_prefill_preallocated_per_rank_bytes, 256)
        self.assertFalse(prefill.decode_handoff_claim_pending)
        self.assertFalse(decode.decode_handoff_claim_pending)
        request.agentic_kv_owner_instance_id = None
        before_handoff = decode.memory.npu_used
        router.transfer_prefill_request([request], current_time_ns=101)
        self.assertEqual(decode.memory.npu_used, before_handoff)
        self.assertEqual(decode.decoded, [request])
        self.assertEqual(decode.decode_admissions, [256])

    def test_pd_decode_handoff_waits_for_delayed_lru_reclaim(self):
        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = FakeDecodeAdmissionManager(ready_ns=500)
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        request = PrefillRequest(8)
        request.ready_time = 100

        router._stage_pd_receive_admission(request, prefill, 100)
        self.assertEqual(router.process_pending_decode_handoffs(100), 1)
        self.assertFalse(router.admit_pd_prefill_chunk(
            prefill, request, 100, 100))
        self.assertEqual(router.get_next_decode_handoff_wakeup(), 500)
        self.assertTrue(decode.decode_handoff_claim_pending)
        self.assertEqual(router.process_pending_decode_handoffs(499), 0)
        self.assertEqual(prefill.request, [request])

        self.assertEqual(router.process_pending_decode_handoffs(500), 0)
        self.assertEqual(prefill.request, [request])
        self.assertEqual(decode.memory.npu_used, 112)
        self.assertEqual(request.pd_decode_admission_wait_ns, 400)
        self.assertEqual(
            request.pd_decode_admission_critical_wait_ns, 400)
        self.assertFalse(router.has_pending_decode_handoffs())
        self.assertIsNone(router.get_next_decode_handoff_wakeup())
        self.assertFalse(decode.decode_handoff_claim_pending)

    def test_pd_launch_uses_later_capacity_time_for_atomic_admission(self):
        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = FakeDecodeAdmissionManager(
            ready_ns={0: 100, 1: 500})
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        request = PrefillRequest(11)
        request.ready_time = 100

        router._stage_pd_receive_admission(request, prefill, 100)
        self.assertEqual(router.process_pending_decode_handoffs(100), 1)
        self.assertFalse(router.admit_pd_prefill_chunk(
            prefill, request, 100, 100))
        self.assertEqual(router.get_next_decode_handoff_wakeup(), 500)

        self.assertEqual(router.process_pending_decode_handoffs(500), 0)
        self.assertEqual(request.pd_prefill_capacity_ready_ns, 100)
        self.assertEqual(request.pd_decode_capacity_ready_ns, 500)
        self.assertEqual(request.pd_prefill_capacity_wait_ns, 0)
        self.assertEqual(request.pd_decode_capacity_wait_ns, 400)
        self.assertEqual(request.pd_prefill_admission_ready_ns, 500)
        self.assertEqual(request.pd_decode_admission_ready_ns, 500)
        self.assertEqual(request.pd_prefill_admission_wait_ns, 400)
        self.assertEqual(request.pd_decode_admission_wait_ns, 400)
        self.assertEqual(request.pd_prefill_admission_critical_wait_ns, 0)
        self.assertEqual(request.pd_decode_admission_critical_wait_ns, 400)
        self.assertEqual(request.pd_launch_admission_wait_ns, 400)
        self.assertEqual(
            request.pd_launch_admission_critical_wait_ns, 400)
        self.assertEqual(len(manager.pd_launch_admissions), 1)
        self.assertFalse(manager.claims)

    def test_one_fresh_strict_pd_admission_after_restore_hides_gate(self):
        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = FakeDecodeAdmissionManager(ready_ns=150)
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        request = PrefillRequest(12)
        request.ready_time = 100
        request.agentic_kv_restore_ns = 100
        request.agentic_kv_restore_ready_time_ns = 100
        request.agentic_kv_async_decode_join = True

        router._stage_pd_receive_admission(request, prefill, 0)
        self.assertEqual(router.process_pending_decode_handoffs(0), 0)
        self.assertEqual(router.process_pending_decode_handoffs(100), 1)
        self.assertFalse(router.admit_pd_prefill_chunk(
            prefill, request, 100, 100))
        self.assertEqual(router.process_pending_decode_handoffs(150), 0)

        self.assertEqual(manager.async_restore_gates, [(12, 150, 0)])
        self.assertEqual(request.agentic_kv_restore_ns, 100)
        self.assertEqual(request.agentic_kv_restore_gate_wait_ns, 0)

    def test_one_fresh_strict_pd_admission_before_restore_counts_remainder(self):
        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = FakeDecodeAdmissionManager(ready_ns=40)
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        request = PrefillRequest(13)
        request.ready_time = 40
        request.agentic_kv_restore_ns = 100
        request.agentic_kv_restore_ready_time_ns = 100
        request.agentic_kv_async_decode_join = True

        router._stage_pd_receive_admission(request, prefill, 0)
        self.assertEqual(router.process_pending_decode_handoffs(0), 0)
        self.assertEqual(router.process_pending_decode_handoffs(40), 1)
        self.assertTrue(router.admit_pd_prefill_chunk(
            prefill, request, 100, 40))

        self.assertEqual(manager.async_restore_gates, [(13, 40, 60)])
        self.assertEqual(request.agentic_kv_restore_ns, 100)
        self.assertEqual(request.agentic_kv_restore_gate_wait_ns, 60)
        self.assertEqual(router.process_pending_decode_handoffs(100), 0)

    def test_pd_reclaim_poll_is_coalesced_until_capacity_state_changes(self):
        class RejectingManager(FakeDecodeAdmissionManager):
            def claim_active_hbm_reclaim(
                    self, instance_id, needed_per_rank_bytes, now_ns,
                    owner_kind="legacy", owner_id=None):
                self.claim_calls.append(
                    (instance_id, needed_per_rank_bytes, now_ns))
                return None

        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = RejectingManager(ready_ns=0)
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        request = PrefillRequest(14)
        router._stage_pd_receive_admission(request, prefill, 100)

        self.assertEqual(router.process_pending_decode_handoffs(100), 1)
        self.assertFalse(router.admit_pd_prefill_chunk(
            prefill, request, 100, 100))
        self.assertEqual(len(manager.claim_calls), 2)
        for _ in range(10_000):
            self.assertEqual(
                router.process_pending_decode_handoffs(100), 0)
        self.assertEqual(len(manager.claim_calls), 2)

        prefill.memory.npu_used += 1
        self.assertEqual(router.process_pending_decode_handoffs(100), 0)
        # Pair admission is atomic, so any exact pair-state change retries
        # both incomplete roles once. The outer pair cache coalesces all
        # unchanged polls; a second per-role cache would hide manager-only
        # reclaimability generations.
        self.assertEqual(len(manager.claim_calls), 4)

    def test_capacity_waiter_retries_at_explicit_demotion_commit(self):
        class DemotionJoinManager(FakeTierManager):
            def __init__(self):
                super().__init__()
                self.commit_ns = 50

            def prepare_request(self, **kwargs):
                self.prepared.append(kwargs)
                if int(kwargs["operation_time_ns"]) < self.commit_ns:
                    return None
                # Avoid appending twice when delegating to the shared fixture.
                self.prepared.pop()
                return super().prepare_request(**kwargs)

            def pending_prepare_retry_time(self, session_id):
                del session_id
                return self.commit_ns

        scheduler = FakeScheduler(0)
        manager = DemotionJoinManager()
        router = Router(
            1, [scheduler], 0, "RR", agentic_kv_manager=manager)
        router._pending_requests = [{
            "index": 44,
            "input_toks": 100,
            "output_toks": 101,
            "arrival_time_ns": 0,
            "session_id": "joining",
            "sub_request_index": 1,
            "prefix_reuse_toks": 100,
        }]

        # A capacity deferral is not a routed request. The only work remains
        # in the explicit preparation-wait queue until the demotion commits.
        self.assertEqual(router.route_arrived_requests(0), 0)
        self.assertEqual(len(manager.prepared), 1)
        self.assertEqual(router.get_next_pending_arrival(), 50)
        self.assertEqual(router.route_arrived_requests(49), 0)
        self.assertEqual(len(manager.prepared), 1)

        self.assertEqual(router.route_arrived_requests(50), 0)
        self.assertEqual(len(manager.prepared), 2)
        self.assertEqual(router._pending_capacity_preparations, [])
        self.assertEqual(manager.prepared[-1]["operation_time_ns"], 50)
        self.assertEqual(router.get_next_pending_arrival(), 123)
        self.assertEqual(router.route_arrived_requests(123), 1)
        self.assertEqual([values[0][0] for values in scheduler.added], [44])

    def test_strict_pd_pair_fcfs_defers_tail_restore_until_head_admits(self):
        class InitiallyBlockedManager(FakeTierManager):
            def __init__(self):
                super().__init__()
                self.blocked = True

            def claim_active_hbm_reclaim(
                    self, instance_id, needed_per_rank_bytes, now_ns,
                    owner_kind="legacy", owner_id=None):
                if self.blocked:
                    return None
                return super().claim_active_hbm_reclaim(
                    instance_id,
                    needed_per_rank_bytes,
                    now_ns,
                    owner_kind=owner_kind,
                    owner_id=owner_id,
                )

        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = InitiallyBlockedManager()
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        router._pending_requests = [
            {
                "index": 100,
                "input_toks": 256,
                "output_toks": 257,
                "arrival_time_ns": 0,
                "session_id": "older-fresh",
                "sub_request_index": 0,
                "prefix_reuse_toks": 0,
            },
            {
                "index": 101,
                "input_toks": 140,
                "output_toks": 141,
                "arrival_time_ns": 0,
                "session_id": "later-resume",
                "sub_request_index": 1,
                "prefix_reuse_toks": 100,
            },
        ]

        # Binding no longer claims the full prompt. The fresh head becomes
        # runnable, while the continuation can prepare independently; exact
        # block FIFO is enforced later by the chunk-admission queue.
        self.assertEqual(router.route_arrived_requests(0), 1)
        self.assertEqual(len(manager.prepared), 1)
        self.assertEqual(router._pending_pd_admission_waits, [])
        self.assertFalse(router.has_pending_requests())
        self.assertTrue(router.has_pending_decode_handoffs())

        # Capacity changes do not re-run preparation; only actual chunk
        # scheduling will exercise the pair's block claim.
        manager.blocked = False
        self.assertEqual(router.route_arrived_requests(1), 0)
        self.assertEqual(len(manager.prepared), 1)
        self.assertEqual(
            manager.prepared[0]["session_id"], "later-resume")
        self.assertEqual(manager.prepared[0]["release_time_ns"], 0)
        self.assertEqual(manager.prepared[0]["operation_time_ns"], 0)
        self.assertEqual(
            manager.prepared[0]["residency_at_return"], "cpu")
        self.assertEqual(
            manager.return_residency_snapshots,
            [("later-resume", 0)],
        )
        self.assertEqual(router._pending_pd_admission_waits, [])
        deferred_request = router._pending_prefill_launches[0]["request"]
        self.assertEqual(deferred_request.pd_pair_fifo_wait_ns, 0)
        self.assertEqual(deferred_request.agentic_kv_owner_gate_ns, 123)
        self.assertEqual(
            deferred_request.agentic_kv_restore_issue_time_ns, 0)
        self.assertEqual(
            deferred_request.agentic_kv_restore_ready_time_ns, 123)

    def test_initial_pd_pair_fifo_wait_is_visible_without_restore(self):
        class InitiallyBlockedManager(FakeTierManager):
            def __init__(self):
                super().__init__()
                self.blocked = True

            def claim_active_hbm_reclaim(
                    self, instance_id, needed_per_rank_bytes, now_ns,
                    owner_kind="legacy", owner_id=None):
                if self.blocked:
                    return None
                return super().claim_active_hbm_reclaim(
                    instance_id,
                    needed_per_rank_bytes,
                    now_ns,
                    owner_kind=owner_kind,
                    owner_id=owner_id,
                )

        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = InitiallyBlockedManager()
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        router._pending_requests = [
            {
                "index": 500,
                "input_toks": 100,
                "output_toks": 101,
                "arrival_time_ns": 0,
                "session_id": "initial-head",
                "sub_request_index": 0,
                "prefix_reuse_toks": 0,
            },
            {
                "index": 501,
                "input_toks": 100,
                "output_toks": 101,
                "arrival_time_ns": 0,
                "session_id": "initial-tail",
                "sub_request_index": 0,
                "prefix_reuse_toks": 0,
            },
        ]

        self.assertEqual(router.route_arrived_requests(0), 2)
        manager.blocked = False
        self.assertEqual(router.route_arrived_requests(1), 0)

        metadata = next(
            row_metadata
            for values, row_metadata in prefill.added
            if values[0] == 501
        )
        self.assertEqual(metadata["pd_pair_fifo_wait_ns"], 0)
        self.assertEqual(metadata["agentic_kv_owner_gate_ns"], 0)
        self.assertEqual(metadata["agentic_kv_restore_ns"], 0)
        self.assertEqual(
            metadata["agentic_kv_restore_issue_time_ns"], 0)
        self.assertEqual(
            metadata["agentic_kv_target_hbm_ready_time_ns"], 0)
        self.assertEqual(
            metadata["agentic_kv_restore_ready_time_ns"], 0)
        self.assertEqual(manager.prepared, [])

    def test_frozen_pd_wait_and_capacity_owner_are_censored_once(self):
        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = FakeTierManager()
        manager.censor_audits["capacity-owner"] = {
            "session_id": "capacity-owner",
            "source_demotion_join": {
                "session_id": "capacity-owner", "elapsed_ns": 10},
            "destination_admission": {
                "session_id": "capacity-owner", "elapsed_ns": 20},
            "transient_dram_admission": {
                "session_id": "capacity-owner", "elapsed_ns": 5},
        }
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        router._pending_pd_admission_waits = [{
            "request": {
                "index": 301,
                "session_id": "pair-wait",
            },
            "pair": (0, 1),
        }]
        router._pending_capacity_preparations = [{
            "request": {
                "index": 300,
                "session_id": "capacity-owner",
            },
            "instance_id": 0,
            "last_state": (0, 0),
        }]
        router._pending_sync_preparations = [{
            "request": {
                "index": 299,
                "session_id": "sync-owner",
            },
            "instance_ids": (0,),
        }]
        manager.sync_prepare_locks[299] = (0,)
        router._pd_admission_owner[(0, 1)] = 300

        router.freeze_session_admission()
        summary = router.finalize_measurement_censoring(999)

        self.assertEqual(
            summary["pending_capacity_prepare_rows_at_cutoff"], 1)
        self.assertEqual(
            summary["pending_pd_admission_waits_at_cutoff"], 1)
        self.assertEqual(summary["pending_prepare_rows_at_cutoff"], 1)
        self.assertEqual(summary["released_prepare_rows_at_freeze"], 3)
        self.assertEqual(
            sorted(manager.ended),
            ["capacity-owner", "pair-wait", "sync-owner"])
        self.assertEqual(len(manager.ended), len(set(manager.ended)))
        self.assertEqual(manager.sync_prepare_locks, {})
        self.assertEqual(router._pending_capacity_preparations, [])
        self.assertEqual(router._pending_pd_admission_waits, [])
        self.assertEqual(router._pd_admission_owner, {})
        self.assertEqual(summary["censored_source_demotion_joins"], 1)
        self.assertEqual(summary["censored_destination_admissions"], 1)
        self.assertEqual(summary["censored_transient_dram_admissions"], 1)
        self.assertEqual(
            summary["censored_source_demotion_join_audits"][0][
                "elapsed_ns"],
            10,
        )
        self.assertEqual(
            summary["censored_destination_admission_audits"][0][
                "elapsed_ns"],
            20,
        )
        self.assertEqual(
            summary["censored_transient_dram_admission_audits"][0][
                "elapsed_ns"],
            5,
        )

    def test_frozen_pending_handoff_cancels_exact_claims_before_kv(self):
        class CensoringManager(FakeDecodeAdmissionManager):
            def __init__(self):
                super().__init__(ready_ns=500)
                self.cancelled = []
                self.censored = []

            def active_hbm_reclaim_claim(self, instance_id):
                return self.claims.get(int(instance_id))

            def cancel_active_hbm_reclaim(self, instance_id, now_ns):
                claim = self.claims.pop(int(instance_id), None)
                if claim is not None:
                    self.cancelled.append((int(instance_id), int(now_ns)))
                return claim

            def censor_prepared_request(self, request, now_ns):
                self.assert_claims_released()
                owner = prefill.memory.get_kv(80)
                prefill.memory.npu_used -= owner
                decode.memory.npu_used -= int(
                    request.agentic_kv_retained_per_rank_bytes)
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

            def assert_claims_released(self):
                if self.claims:
                    raise AssertionError(
                        f"prepared KV censored before claims: {self.claims}")

        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = CensoringManager()
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        request = PrefillRequest(400, session_id="censored-handoff")
        request.agentic_kv_hit_tokens = 80
        request.num_computed_tokens = 80
        request.agentic_kv_owner_instance_id = 0
        request.agentic_kv_retained_instance_id = 1
        request.agentic_kv_retained_per_rank_bytes = 80
        prefill.memory.npu_used = 80
        decode.memory.npu_used = 80

        router._pd_admission_owner[(0, 1)] = request.id
        router._stage_pd_receive_admission(request, prefill, 100)
        self.assertEqual(router.process_pending_decode_handoffs(100), 1)
        self.assertFalse(router.admit_pd_prefill_chunk(
            prefill, request, 20, 100))
        self.assertEqual(set(manager.claims), {0, 1})

        router.freeze_session_admission()
        summary = router.finalize_measurement_censoring(101)

        self.assertEqual(summary["pending_decode_handoffs_at_cutoff"], 0)
        self.assertEqual(summary["pending_pd_chunk_admissions_at_cutoff"], 1)
        self.assertEqual(summary["censored_pending_decode_handoffs"], 1)
        self.assertEqual(manager.cancelled, [(0, 101), (1, 101)])
        self.assertEqual(len(manager.censored), 1)
        self.assertEqual(prefill.memory.npu_used, 0)
        self.assertEqual(decode.memory.npu_used, 0)
        self.assertFalse(router.has_pending_decode_handoffs())
        self.assertEqual(router._pd_admission_owner, {})

    def test_frozen_pending_prefill_launch_releases_consumed_pd_allocations(self):
        class LaunchCensorManager(FakeDecodeAdmissionManager):
            def __init__(self):
                super().__init__(ready_ns=0)
                self.censored_launches = []

            def censor_prepared_request(self, request, now_ns):
                self.assert_no_claims()
                if request.pd_prefill_preallocated_per_rank_bytes != 0:
                    raise AssertionError(
                        request.pd_prefill_preallocated_per_rank_bytes)
                if request.pd_prefill_initial_restored_per_rank_bytes != 80:
                    raise AssertionError(
                        request.pd_prefill_initial_restored_per_rank_bytes)
                prefill.memory.npu_used -= 80
                decode.memory.npu_used -= 80
                request.agentic_kv_owner_instance_id = None
                request.agentic_kv_retained_instance_id = None
                request.agentic_kv_retained_per_rank_bytes = 0
                request.pd_prefill_preallocated_per_rank_bytes = 0
                audit = {
                    "request_id": int(request.id),
                    "session_id": str(request.session_id),
                    "time_ns": int(now_ns),
                }
                self.censored_launches.append(audit)
                return dict(audit)

            def assert_no_claims(self):
                if self.claims:
                    raise AssertionError(
                        f"launch censored before claims consumed: "
                        f"{self.claims}")

        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = LaunchCensorManager()
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        request = PrefillRequest(401, session_id="censored-launch")
        request.ready_time = 500
        request.agentic_kv_hit_tokens = 80
        request.agentic_kv_owner_instance_id = 0
        request.agentic_kv_retained_instance_id = 1
        request.agentic_kv_retained_per_rank_bytes = 80
        prefill.memory.npu_used = 80
        decode.memory.npu_used = 80

        router._pd_admission_owner[(0, 1)] = request.id
        router._stage_pd_receive_admission(request, prefill, 100)
        self.assertEqual(router.process_pending_decode_handoffs(100), 0)
        self.assertEqual(len(router._pending_prefill_launches), 1)
        self.assertEqual(prefill.memory.npu_used, 80)
        self.assertEqual(decode.memory.npu_used, 80)
        self.assertFalse(manager.claims)

        router.freeze_session_admission()
        summary = router.finalize_measurement_censoring(101)

        self.assertEqual(summary["pending_prefill_launches_at_cutoff"], 1)
        self.assertEqual(summary["censored_pending_prefill_launches"], 1)
        self.assertEqual(len(manager.censored_launches), 1)
        self.assertEqual(prefill.memory.npu_used, 0)
        self.assertEqual(decode.memory.npu_used, 0)
        self.assertFalse(router.has_pending_decode_handoffs())
        self.assertEqual(router._pd_admission_owner, {})

    def test_frozen_queued_prefill_releases_pd_allocations_once(self):
        class QueuedCensorManager(FakeDecodeAdmissionManager):
            def __init__(self):
                super().__init__(ready_ns=0)
                self.censored = []

            def censor_prepared_request(self, request, now_ns):
                self.assertFalse(request in prefill.request)
                self.assertEqual(
                    request.pd_prefill_preallocated_per_rank_bytes, 0)
                self.assertEqual(
                    request.pd_prefill_initial_restored_per_rank_bytes, 80)
                prefill.memory.free(80, Device.NPU)
                decode.memory.free(80, Device.NPU)
                request.agentic_kv_owner_instance_id = None
                request.agentic_kv_retained_instance_id = None
                request.agentic_kv_retained_per_rank_bytes = 0
                request.pd_prefill_preallocated_per_rank_bytes = 0
                audit = {
                    "request_id": int(request.id),
                    "session_id": str(request.session_id),
                    "time_ns": int(now_ns),
                }
                self.censored.append(audit)
                return dict(audit)

            @staticmethod
            def assertFalse(value):
                if value:
                    raise AssertionError(value)

            @staticmethod
            def assertEqual(observed, expected):
                if observed != expected:
                    raise AssertionError((observed, expected))

        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = QueuedCensorManager()
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        request = PrefillRequest(402, session_id="censored-queued-p")
        request.ready_time = 100
        request.agentic_kv_hit_tokens = 80
        request.agentic_kv_owner_instance_id = 0
        request.agentic_kv_retained_instance_id = 1
        request.agentic_kv_retained_per_rank_bytes = 80
        prefill.memory.npu_used = 80
        decode.memory.npu_used = 80

        router._pd_admission_owner[(0, 1)] = request.id
        router._stage_pd_receive_admission(request, prefill, 100)
        self.assertEqual(router.process_pending_decode_handoffs(100), 1)
        self.assertEqual(prefill.request, [request])
        self.assertEqual(router._pending_prefill_launches, [])
        self.assertEqual(prefill.memory.npu_used, 80)
        self.assertEqual(decode.memory.npu_used, 80)

        router.freeze_session_admission()
        summary = router.finalize_measurement_censoring(101)

        self.assertEqual(summary["queued_requests_at_cutoff"], 1)
        self.assertEqual(summary["pending_prefill_launches_at_cutoff"], 0)
        self.assertEqual(summary["censored_queued_pd_prefill_requests"], 1)
        self.assertEqual(summary["censored_pending_prefill_launches"], 0)
        self.assertEqual(len(manager.censored), 1)
        self.assertEqual(prefill.request, [])
        self.assertEqual(prefill.memory.npu_used, 0)
        self.assertEqual(decode.memory.npu_used, 0)

    def test_post_freeze_prefill_completion_releases_orphaned_decode_hbm(self):
        class CompletionCensorManager(FakeDecodeAdmissionManager):
            def __init__(self):
                super().__init__(ready_ns=0)
                self.completed_censors = []

            def censor_completed_pd_prefill_request(
                    self, request, prefill_instance_id,
                    decode_instance_id, now_ns):
                if request.agentic_kv_owner_instance_id is not None:
                    raise AssertionError(
                        request.agentic_kv_owner_instance_id)
                if prefill.memory.npu_used != 0:
                    raise AssertionError(prefill.memory.npu_used)
                if request.pd_prefill_preallocated_per_rank_bytes != 112:
                    raise AssertionError(
                        request.pd_prefill_preallocated_per_rank_bytes)
                if request.pd_decode_full_per_rank_bytes != 112:
                    raise AssertionError(
                        request.pd_decode_full_per_rank_bytes)
                decode.memory.free(112, Device.NPU)
                request.agentic_kv_retained_instance_id = None
                request.agentic_kv_retained_per_rank_bytes = 0
                request.pd_prefill_preallocated_per_rank_bytes = 0
                request.pd_decode_target_instance_id = None
                request.pd_decode_full_per_rank_bytes = 0
                request.pd_decode_reserved_per_rank_bytes = 0
                audit = {
                    "request_id": int(request.id),
                    "prefill_instance_id": int(prefill_instance_id),
                    "decode_instance_id": int(decode_instance_id),
                    "time_ns": int(now_ns),
                }
                self.completed_censors.append(audit)
                return dict(audit)

        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = CompletionCensorManager()
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        request = PrefillRequest(403, session_id="drained-p")
        request.ready_time = 100

        router._pd_admission_owner[(0, 1)] = request.id
        router._stage_pd_receive_admission(request, prefill, 100)
        self.assertEqual(router.process_pending_decode_handoffs(100), 1)
        # Binding fixes the pair but owns only the initial prefixes. The
        # first (and here complete) prefill chunk performs the authoritative
        # atomic P/D block claim.
        self.assertTrue(router.admit_pd_prefill_chunk(
            prefill, request, 100, 100))
        self.assertEqual(prefill.memory.npu_used, 112)
        self.assertEqual(decode.memory.npu_used, 112)

        # Model the post-freeze add_done() transition: P releases its full
        # prompt allocation, but main must not transfer the request into D.
        request.num_computed_tokens = 100
        request.pd_chunk_admitted_tokens = 0
        request.pd_chunk_admission_target_tokens = 0
        request.pd_chunk_admission_history[-1]["committed"] = True
        prefill.request.remove(request)
        prefill.memory.free(112, Device.NPU)
        request.pd_prefill_owned_per_rank_bytes = 0
        request.pd_prefill_handoff_released_per_rank_bytes = 112
        request.pd_kv_ownership_state = "handoff_pending"
        request.agentic_kv_owner_instance_id = None
        router.freeze_session_admission()
        audits = router.censor_completed_pd_prefill_requests(
            [request], 150)

        self.assertEqual(len(audits), 1)
        self.assertEqual(len(manager.completed_censors), 1)
        self.assertEqual(prefill.memory.npu_used, 0)
        self.assertEqual(decode.memory.npu_used, 0)
        summary = router.finalize_measurement_censoring(150)
        self.assertEqual(
            summary["censored_completed_pd_prefill_requests"], 1)
        self.assertEqual(summary["queued_requests_at_cutoff"], 0)

    def test_frozen_decode_queue_releases_active_hbm_and_exact_claim(self):
        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = FakeTierManager()
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        request = Request(404, "model", 100, 110, 0, 1)
        request.session_id = "queued-d"
        request.num_computed_tokens = 100
        request.agentic_kv_owner_instance_id = 1
        decode.memory.npu_used = 112
        decode.enqueue_request(request)
        manager.claims[1] = SimpleNamespace(
            instance_id=1,
            per_rank_bytes=16,
            ready_ns=200,
            owner_kind="scheduler",
            owner_id=request.id,
        )

        router.freeze_session_admission()
        summary = router.finalize_measurement_censoring(150)

        self.assertEqual(summary["queued_requests_at_cutoff"], 1)
        self.assertEqual(summary["cancelled_scheduler_hbm_claims"], 1)
        self.assertEqual(summary["censored_queued_active_requests"], 1)
        self.assertEqual(decode.request, [])
        self.assertEqual(decode.memory.npu_used, 0)
        self.assertIsNone(request.agentic_kv_owner_instance_id)
        self.assertEqual(manager.claims, {})

    def test_finalize_ends_future_censored_session_once_and_preserves_cutoff(self):
        scheduler = FakeScheduler(0, "prefill", node_id=0)
        manager = FakeTierManager()
        router = Router(
            1, [scheduler], 0, "RR",
            agentic_kv_manager=manager)
        router._session_lifecycle["future-session"] = {
            "session_id": "future-session",
            "status": "active",
        }
        router._active_sessions.add("future-session")
        router._pending_requests = [{
            "index": 600,
            "input_toks": 10,
            "output_toks": 11,
            "arrival_time_ns": 1_000,
            "session_id": "future-session",
            "sub_request_index": 1,
        }]

        router.freeze_session_admission()
        summary = router.finalize_measurement_censoring(100)

        self.assertEqual(summary["active_session_ids_at_cutoff"], [
            "future-session"])
        self.assertEqual(summary["pending_request_rows_at_cutoff"], 1)
        self.assertEqual(summary["ended_censored_sessions"], 1)
        self.assertEqual(manager.ended, ["future-session"])
        self.assertEqual(router._active_sessions, set())
        self.assertFalse(router.has_pending_requests())

    def test_strict_pd_pair_wait_does_not_block_independent_pair(self):
        class FirstPairBlockedManager(FakeTierManager):
            def claim_active_hbm_reclaim(
                    self, instance_id, needed_per_rank_bytes, now_ns,
                    owner_kind="legacy", owner_id=None):
                if int(instance_id) in {0, 1}:
                    return None
                return super().claim_active_hbm_reclaim(
                    instance_id,
                    needed_per_rank_bytes,
                    now_ns,
                    owner_kind=owner_kind,
                    owner_id=owner_id,
                )

        schedulers = [
            FakeScheduler(0, "prefill", node_id=0),
            FakeScheduler(1, "decode", node_id=0),
            FakeScheduler(2, "prefill", node_id=1),
            FakeScheduler(3, "decode", node_id=1),
        ]
        router = Router(
            4, schedulers, 0, "RR",
            agentic_kv_manager=FirstPairBlockedManager())
        router._pending_requests = [
            {
                "index": 200,
                "input_toks": 100,
                "output_toks": 101,
                "arrival_time_ns": 0,
                "session_id": "node-zero",
                "sub_request_index": 0,
                "prefix_reuse_toks": 0,
            },
            {
                "index": 201,
                "input_toks": 100,
                "output_toks": 101,
                "arrival_time_ns": 0,
                "session_id": "node-one",
                "sub_request_index": 0,
                "prefix_reuse_toks": 0,
            },
        ]

        self.assertEqual(router.route_arrived_requests(0), 2)
        self.assertEqual(
            [request.id for request in schedulers[0].request], [200])
        self.assertEqual(
            [request.id for request in schedulers[2].request], [201])
        self.assertFalse(router.has_pending_decode_handoffs())

    def test_strict_pd_immediate_admission_drain_is_iterative(self):
        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        prefill.memory.npu_mem = 1_000_000_000
        decode.memory.npu_mem = 1_000_000_000
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=FakeTierManager())
        request_count = 1_100
        router._pending_requests = [
            {
                "index": request_id,
                "input_toks": 1,
                "output_toks": 2,
                "arrival_time_ns": 0,
                "session_id": f"bulk-{request_id}",
                "sub_request_index": 0,
                "prefix_reuse_toks": 0,
            }
            for request_id in range(request_count)
        ]

        self.assertEqual(
            router.route_arrived_requests(0), request_count)
        self.assertEqual(len(prefill.request), request_count)
        self.assertFalse(router.has_pending_decode_handoffs())
        self.assertEqual(router._pd_admission_owner, {})

    def test_strict_pd_rejects_ambiguous_decode_destinations(self):
        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode_a = FakeScheduler(1, "decode", node_id=0)
        decode_b = FakeScheduler(2, "decode", node_id=0)
        router = Router(
            3,
            [prefill, decode_a, decode_b],
            0,
            "RR",
            agentic_kv_manager=FakeTierManager(),
        )
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            router._stage_pd_receive_admission(
                PrefillRequest(9), prefill, 0
            )

    def test_strict_pd_hbm_retained_prefix_preallocates_only_suffix(self):
        for source in ("hbm",):
            with self.subTest(source=source):
                prefill = FakeScheduler(0, "prefill", node_id=0)
                decode = FakeScheduler(1, "decode", node_id=0)
                manager = FakeDecodeAdmissionManager(ready_ns=100)
                router = Router(
                    2,
                    [prefill, decode],
                    0,
                    "RR",
                    agentic_kv_manager=manager,
                )
                request = PrefillRequest(10)
                request.ready_time = 100
                request.agentic_kv_source = source
                request.agentic_kv_residency_at_return = source
                request.agentic_kv_hit_tokens = 80
                request.num_computed_tokens = 80
                request.agentic_kv_owner_instance_id = 0
                request.agentic_kv_retained_instance_id = 1
                request.agentic_kv_retained_per_rank_bytes = 80
                prefill.memory.npu_used = 80
                decode.memory.npu_used = 80

                router._stage_pd_receive_admission(request, prefill, 100)
                self.assertEqual(
                    request.pd_decode_reserved_per_rank_bytes, 0
                )
                self.assertEqual(
                    request.pd_prefill_reserved_per_rank_bytes, 0
                )
                self.assertEqual(
                    router.process_pending_decode_handoffs(100), 1
                )
                self.assertTrue(router.admit_pd_prefill_chunk(
                    prefill, request, 20, 100))
                self.assertEqual(decode.memory.npu_used, 112)
                self.assertEqual(prefill.memory.npu_used, 112)
                self.assertEqual(
                    manager.claim_calls,
                    [(0, 32, 100), (1, 32, 100)],
                )

                request.agentic_kv_owner_instance_id = None
                before_handoff = decode.memory.npu_used
                router.transfer_prefill_request(
                    [request], current_time_ns=101
                )
                self.assertEqual(decode.memory.npu_used, before_handoff)

    def test_strict_pd_lower_tier_restore_preallocates_full_decode(self):
        for source in ("cpu", "ssd"):
            with self.subTest(source=source):
                prefill = FakeScheduler(0, "prefill", node_id=0)
                decode = FakeScheduler(1, "decode", node_id=0)
                manager = FakeDecodeAdmissionManager(ready_ns=100)
                router = Router(
                    2,
                    [prefill, decode],
                    0,
                    "RR",
                    agentic_kv_manager=manager,
                )
                request = PrefillRequest(10)
                request.ready_time = 100
                request.agentic_kv_source = source
                request.agentic_kv_residency_at_return = source
                request.agentic_kv_hit_tokens = 80
                request.num_computed_tokens = 80
                request.agentic_kv_owner_instance_id = 0
                # CPU/SSD restored directly into P HBM. D has no retained
                # prefix and must reserve its complete normal handoff buffer.
                request.agentic_kv_retained_instance_id = None
                request.agentic_kv_retained_per_rank_bytes = 0
                prefill.memory.npu_used = 80

                router._stage_pd_receive_admission(request, prefill, 100)
                self.assertEqual(
                    request.pd_decode_reserved_per_rank_bytes, 0
                )
                self.assertEqual(
                    request.pd_prefill_reserved_per_rank_bytes, 0
                )
                self.assertEqual(
                    router.process_pending_decode_handoffs(100), 1
                )
                self.assertTrue(router.admit_pd_prefill_chunk(
                    prefill, request, 20, 100))
                self.assertEqual(decode.memory.npu_used, 112)
                self.assertEqual(prefill.memory.npu_used, 112)
                self.assertEqual(
                    manager.claim_calls,
                    [(0, 32, 100), (1, 112, 100)],
                )

                request.agentic_kv_owner_instance_id = None
                before_handoff = decode.memory.npu_used
                router.transfer_prefill_request(
                    [request], current_time_ns=101
                )
                self.assertEqual(decode.memory.npu_used, before_handoff)

    def test_restore_pending_request_does_not_block_hbm_ready_prefill(self):
        prefill = FakeScheduler(0, "prefill", node_id=0)
        decode = FakeScheduler(1, "decode", node_id=0)
        manager = FakeTierManager()
        router = Router(
            2,
            [prefill, decode],
            0,
            "RR",
            agentic_kv_manager=manager,
        )
        cold = PrefillRequest(20)
        cold.ready_time = 500
        cold.agentic_kv_source = "ssd"
        hbm = PrefillRequest(21)
        hbm.ready_time = 100
        hbm.agentic_kv_source = "hbm"

        router._stage_pd_receive_admission(cold, prefill, 100)
        router._stage_pd_receive_admission(hbm, prefill, 100)

        self.assertEqual(router.process_pending_decode_handoffs(100), 1)
        self.assertEqual([request.id for request in prefill.request], [21])
        self.assertTrue(router.has_pending_decode_handoffs())
        self.assertEqual(router.get_next_decode_handoff_wakeup(), 500)

        self.assertEqual(router.process_pending_decode_handoffs(500), 1)
        self.assertEqual(
            [request.id for request in prefill.request], [21, 20]
        )


if __name__ == "__main__":
    unittest.main()

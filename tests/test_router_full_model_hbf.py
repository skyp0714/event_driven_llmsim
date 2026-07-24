import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from serving.core.hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
    qwen_model_weight_bytes_per_rank,
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
)
from serving.core.hbf_online_adapter import FullModelHBFOnlineAdapter
from serving.core.memory_model import Device, MemoryModel
from serving.core.request import Request
from serving.core.router import Router
from serving.core.scheduler import Scheduler


REPO_ROOT = Path(__file__).resolve().parents[1]


class RouteDecision:
    def __init__(
            self, request_id, *,
            divert_to_hbf=False,
            force_gpu_recompute=False,
            required_gpu_instance_id=None):
        self.request_id = int(request_id)
        self.divert_to_hbf = bool(divert_to_hbf)
        self.run_on_gpu = not self.divert_to_hbf
        self.force_gpu_recompute = bool(force_gpu_recompute)
        self.required_gpu_instance_id = required_gpu_instance_id


class RecordingAdapter:
    def __init__(self, *, gpu_resume_mode, divert_sessions=()):
        self.gpu_resume_mode = gpu_resume_mode
        self.divert_sessions = set(divert_sessions)
        self.offers = []
        self.flush_calls = []
        self.bind_calls = []
        self.completions = []
        self.censor_calls = []
        self.calls = {}
        self._staged_by_time = {}
        self._events = []
        self.gpu_ready_reclaim_calls = []

    def offer_raw_request(self, row, *, now_ns):
        copied = dict(row)
        self.offers.append((copied, int(now_ns)))
        request_id = int(row["index"])
        session_id = str(row["session_id"])
        call_index = int(row.get("sub_request_index", 0))
        divert = session_id in self.divert_sessions
        force_recompute = (
            call_index > 0
            and self.gpu_resume_mode == "recompute"
            and not divert
        )
        self.calls[request_id] = SimpleNamespace(
            gpu_instance_id=None,
            final_materialized_tokens=int(row["output_toks"]) - 1,
        )
        if divert:
            self._staged_by_time.setdefault(int(now_ns), []).append(
                request_id)
        if force_recompute:
            self._events.append({
                "kind": "idle_release",
                "request_id": request_id,
                "time_ns": int(now_ns),
            })
        return RouteDecision(
            request_id,
            divert_to_hbf=divert,
            force_gpu_recompute=force_recompute,
        )

    def decorate_gpu_metadata(self, decision, row):
        result = dict(row)
        prefix_tokens = int(row.get("prefix_reuse_toks", 0))
        hit_tokens = 0 if decision.force_gpu_recompute else prefix_tokens
        result["prefix_reuse_toks"] = hit_tokens
        result["agentic_kv_hit_tokens"] = hit_tokens
        result["agentic_kv_recompute_tokens"] = (
            prefix_tokens if decision.force_gpu_recompute else 0)
        result["agentic_kv_owner_instance_id"] = None
        result["hbf_gpu_required_instance_id"] = (
            decision.required_gpu_instance_id)
        return result

    def flush_admissions(self, now_ns):
        now_ns = int(now_ns)
        self.flush_calls.append(now_ns)
        return len(self._staged_by_time.pop(now_ns, ()))

    def pop_gpu_hbm_events(self):
        result = list(self._events)
        self._events.clear()
        return result

    def bind_native_gpu_request(
            self, request, *, gpu_instance_id=None):
        instance_id = (
            int(request.instance_id)
            if gpu_instance_id is None else int(gpu_instance_id)
        )
        self.calls[int(request.id)].gpu_instance_id = instance_id
        self.bind_calls.append((int(request.id), instance_id))
        return instance_id

    def complete_native_gpu_request(
            self, request, *, completion_ns,
            materialized_tokens=None, gpu_instance_id=None):
        request_id = int(request.id)
        call = self.calls[request_id]
        observed_tokens = (
            int(request.num_computed_tokens)
            if materialized_tokens is None else int(materialized_tokens)
        )
        if observed_tokens != call.final_materialized_tokens:
            raise RuntimeError("test completion has wrong materialized tokens")
        instance_id = self.bind_native_gpu_request(
            request, gpu_instance_id=gpu_instance_id)
        self.completions.append((
            request_id, int(completion_ns), instance_id))
        self._events.append({
            "kind": "turn_retain",
            "request_id": request_id,
            "gpu_instance_id": instance_id,
            "time_ns": int(completion_ns),
        })
        return f"migration-{request_id}"

    def validate_queued_native_gpu_request(
            self, request, *, now_ns):
        if int(request.id) not in self.calls:
            raise KeyError(request.id)
        return {
            "request_id": int(request.id),
            "session_id": str(request.session_id),
            "cutoff_time_ns": int(now_ns),
        }

    def censor_queued_native_gpu_request(
            self, request, *, now_ns):
        audit = self.validate_queued_native_gpu_request(
            request, now_ns=now_ns)
        self.censor_calls.append((
            int(request.id), int(now_ns)))
        del self.calls[int(request.id)]
        return audit

    def has_pending_native_gpu_requests(self):
        return bool(self.calls)

    def has_pending_astra_dispatches(self):
        return False

    def reclaim_gpu_ready_for_hbm_pressure(
            self, *, gpu_instance_id, now_ns):
        self.gpu_ready_reclaim_calls.append((
            int(gpu_instance_id), int(now_ns)))
        return None


class RecordingBridge:
    def __init__(
            self, schedulers, *, pd_pairs=(),
            pd_decode_capacity_available=True):
        self.schedulers = {
            int(scheduler.instance_id): scheduler
            for scheduler in schedulers
        }
        self.pd_pairs = tuple(pd_pairs)
        self.topology = "pd" if self.pd_pairs else "colocated"
        self.fallback_reuse_mode = (
            "recompute" if self.topology == "pd" else "sticky_reuse")
        self.validated_adapters = []
        self.applied_events = []
        self.pd_decorations = []
        self.pd_bindings = []
        self.colocated_decorations = []
        self.colocated_bindings = []
        self.pd_decode_reservations = {}
        self.pd_decode_reservation_attempts = []
        self.pd_decode_request_capacity_validations = []
        self.pd_decode_capacity_available = bool(
            pd_decode_capacity_available)

    def validate_adapter_contract(self, adapter):
        self.validated_adapters.append(adapter)
        return {"gpu_resume_mode": adapter.gpu_resume_mode}

    def apply_events(self, events):
        events = tuple(events)
        self.applied_events.extend(events)
        return tuple({"applied": event} for event in events)

    def decorate_pd_recompute(
            self, request_id, metadata, *,
            prefill_instance_id, decode_instance_id):
        if not self.applied_events:
            raise RuntimeError(
                "P/D recompute decoration preceded ownership event drain")
        result = dict(metadata)
        result["_pd_prefill_instance_id"] = int(prefill_instance_id)
        result["prefix_reuse_toks"] = 0
        result["agentic_kv_hit_tokens"] = 0
        result["agentic_kv_owner_instance_id"] = None
        result["agentic_kv_retained_instance_id"] = None
        result["agentic_kv_retained_per_rank_bytes"] = 0
        self.pd_decorations.append((
            int(request_id),
            int(prefill_instance_id),
            int(decode_instance_id),
        ))
        return result

    def bind_pd_recompute(self, request):
        if int(request.instance_id) != int(self.pd_pairs[0][0]):
            raise RuntimeError("P/D recompute was not constructed on P")
        if (
            int(request.agentic_kv_hit_tokens) != 0
            or int(request.num_computed_tokens) != 0
        ):
            raise RuntimeError("P/D recompute silently retained prefix KV")
        self.pd_bindings.append(int(request.id))
        return {"request_id": int(request.id)}

    def validate_pd_decode_prompt_capacity(
            self, input_tokens, *, decode_instance_id):
        del decode_instance_id
        if int(input_tokens) <= 0:
            raise ValueError("input_tokens must be positive")
        return 0

    def validate_pd_decode_request_capacity(
            self, input_tokens, requested_output_tokens, *,
            decode_instance_id):
        input_tokens = int(input_tokens)
        requested_output_tokens = int(requested_output_tokens)
        decode_instance_id = int(decode_instance_id)
        if input_tokens <= 0:
            raise ValueError("input_tokens must be positive")
        if requested_output_tokens <= 0:
            raise ValueError(
                "requested_output_tokens must be positive")
        self.pd_decode_request_capacity_validations.append((
            input_tokens,
            requested_output_tokens,
            decode_instance_id,
        ))
        return 0

    def try_reserve_pd_decode(
            self, request, *, prefill_instance_id,
            decode_instance_id):
        self.pd_decode_reservation_attempts.append(int(request.id))
        if not self.pd_decode_capacity_available:
            return False
        request_id = int(request.id)
        reservation = SimpleNamespace(
            request_id=request_id,
            prefill_instance_id=int(prefill_instance_id),
            decode_instance_id=int(decode_instance_id),
            reserved_per_rank_bytes=0,
        )
        self.pd_decode_reservations[request_id] = reservation
        request.pd_decode_target_instance_id = int(decode_instance_id)
        request.pd_decode_full_per_rank_bytes = 0
        request.pd_decode_reserved_per_rank_bytes = 0
        return True

    def pd_decode_reservation(self, request):
        return self.pd_decode_reservations.get(int(request.id))

    def consume_pd_decode_reservation(self, request):
        return self.pd_decode_reservations.pop(int(request.id))

    def cancel_pd_decode_reservation(self, request):
        reservation = self.pd_decode_reservations.pop(
            int(request.id), None)
        return None if reservation is None else {
            "request_id": reservation.request_id,
        }

    def decorate_colocated_continuation(self, request_id, metadata):
        result = dict(metadata)
        self.colocated_decorations.append(int(request_id))
        return result

    def bind_colocated_continuation(self, request):
        self.colocated_bindings.append(int(request.id))
        return {"request_id": int(request.id)}


class RecordingScheduler:
    def __init__(self, instance_id, pd_type):
        self.instance_id = int(instance_id)
        self.pd_type = pd_type
        self.enable_prefix_caching = False
        self.model = "test/model"
        self.request = []
        self.inflight = []
        self.pd_chunk_admission_callback = None
        self.max_model_len = 4096

    def add_request(
            self, values, is_init=True, metadata=None, enqueue=True):
        request = Request(*values, is_init=is_init)
        metadata = {} if metadata is None else metadata
        request.session_id = metadata.get("session_id")
        request.sub_request_index = metadata.get("sub_request_index")
        request.source_session_id = metadata.get("source_session_id")
        request.session_template_index = metadata.get(
            "session_template_index")
        request.session_epoch = int(metadata.get("session_epoch") or 0)
        request.ready_time = int(
            metadata.get("ready_time_ns") or request.arrival)
        request.prefix_reuse_tokens = int(
            metadata.get("prefix_reuse_toks") or 0)
        request.agentic_kv_hit_tokens = int(
            metadata.get("agentic_kv_hit_tokens") or 0)
        request.agentic_kv_recompute_tokens = int(
            metadata.get("agentic_kv_recompute_tokens") or 0)
        request.agentic_kv_owner_instance_id = metadata.get(
            "agentic_kv_owner_instance_id")
        request.agentic_kv_retained_instance_id = metadata.get(
            "agentic_kv_retained_instance_id")
        request.agentic_kv_retained_per_rank_bytes = int(
            metadata.get("agentic_kv_retained_per_rank_bytes") or 0)
        request.num_computed_tokens = request.agentic_kv_hit_tokens
        if enqueue:
            self.request.append(request)
        return request

    def enqueue_request(self, request):
        self.request.append(request)

    def censor_full_model_hbf_queued_request(
            self, request, cutoff_time_ns):
        del cutoff_time_ns
        self.request.remove(request)
        return {
            "request_id": int(request.id),
            "instance_id": self.instance_id,
            "released_npu_per_rank_bytes": 0,
            "released_cpu_cluster_bytes": 0,
        }


def write_workload(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class NullLogger:
    def info(self, *args, **kwargs):
        pass


def build_real_recompute_adapter(*, hardware=None):
    hardware = hardware or HBFServerHardware()
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


def real_finite_scheduler(
        instance_id, pd_type, kv_bytes_per_token_per_rank):
    memory = MemoryModel.__new__(MemoryModel)
    memory.instance_id = int(instance_id)
    memory.node_id = 0
    memory.block_size = 16
    memory.kv_heads_per_tp_rank = 1
    memory.head_dim = int(kv_bytes_per_token_per_rank) // 2
    memory.layers_per_pp_rank = 1
    memory.kv_fp = 1
    memory.weight = 128
    memory.npu_used = 128
    memory.npu_peak_used = 128
    memory.npu_allocatable_mem = 1 << 40
    memory.npu_mem = 1 << 40
    memory.cpu_used = 0
    memory.cpu_mem = 1 << 40
    memory.prefix_storage = None
    memory.enable_prefix_sharing = False
    memory.logger = NullLogger()
    memory.enable_prefix_caching = False

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.instance_id = int(instance_id)
    scheduler.node_id = 0
    scheduler.block_size = 16
    scheduler.enable_prefix_caching = False
    scheduler.pd_type = pd_type
    scheduler.memory = memory
    scheduler.request = []
    scheduler.inflight = []
    scheduler.agentic_kv_manager = None
    scheduler.num_npus = 1
    scheduler.pd_prefill_reclaimability_generation = 0
    scheduler.max_model_len = 4096
    scheduler.model = "test/model"
    return scheduler


class RouterFullModelHBFTests(unittest.TestCase):
    def test_full_model_hbf_rejects_flat_and_oversized_rows_before_offer(self):
        scheduler = RecordingScheduler(0, "colocated")
        adapter = RecordingAdapter(gpu_resume_mode="sticky_reuse")
        bridge = RecordingBridge([scheduler])
        router = Router(
            1, [scheduler], 0,
            full_model_hbf_adapter=adapter,
            full_model_hbf_gpu_hbm_bridge=bridge,
        )

        with TemporaryDirectory() as directory:
            flat = Path(directory) / "flat.jsonl"
            write_workload(flat, [{
                "input_toks": 3,
                "output_toks": 2,
                "arrival_time_ns": 0,
            }])
            with self.assertRaisesRegex(ValueError, "only non-empty agentic"):
                router.load_requests(str(flat))

            oversized = Path(directory) / "oversized.jsonl"
            write_workload(oversized, [{
                "session_id": "too-long",
                "arrival_time_ns": 0,
                "sub_requests": [{
                    "input_toks": 4096,
                    "output_toks": 1,
                }],
            }])
            with self.assertRaisesRegex(ValueError, "max_model_len=4096"):
                router.load_requests(str(oversized))

        self.assertEqual(adapter.offers, [])
        self.assertEqual(adapter.calls, {})

    def test_pd_gpu_launch_waits_for_decode_hbm_before_prefill_visibility(self):
        prefill = RecordingScheduler(0, "prefill")
        decode = RecordingScheduler(1, "decode")
        adapter = RecordingAdapter(gpu_resume_mode="recompute")
        bridge = RecordingBridge(
            [prefill, decode],
            pd_pairs=((0, 1),),
            pd_decode_capacity_available=False,
        )
        router = Router(
            2, [prefill, decode], 0,
            full_model_hbf_adapter=adapter,
            full_model_hbf_gpu_hbm_bridge=bridge,
        )

        with TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.jsonl"
            write_workload(workload, [{
                "session_id": "session-capacity",
                "arrival_time_ns": 0,
                "sub_requests": [{
                    "input_toks": 4,
                    "output_toks": 2,
                }],
            }])
            router.load_requests(str(workload))

        self.assertEqual(router.route_arrived_requests(0), 1)
        self.assertEqual(prefill.request, [])
        self.assertEqual(
            len(router._pending_full_model_hbf_prefill_launches), 1)
        request = router._pending_full_model_hbf_prefill_launches[
            0]["request"]
        self.assertIsNone(bridge.pd_decode_reservation(request))

        bridge.pd_decode_capacity_available = True
        self.assertEqual(router.process_pending_decode_handoffs(10), 1)
        self.assertEqual(prefill.request, [request])
        self.assertEqual(
            router._pending_full_model_hbf_prefill_launches, [])
        self.assertIsNotNone(bridge.pd_decode_reservation(request))

    def test_pd_capacity_waiting_prefill_is_censored_outside_scheduler(self):
        prefill = RecordingScheduler(0, "prefill")
        decode = RecordingScheduler(1, "decode")
        adapter = RecordingAdapter(gpu_resume_mode="recompute")
        bridge = RecordingBridge(
            [prefill, decode],
            pd_pairs=((0, 1),),
            pd_decode_capacity_available=False,
        )
        router = Router(
            2, [prefill, decode], 0,
            full_model_hbf_adapter=adapter,
            full_model_hbf_gpu_hbm_bridge=bridge,
        )

        with TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.jsonl"
            write_workload(workload, [{
                "session_id": "session-capacity-cutoff",
                "arrival_time_ns": 0,
                "sub_requests": [{
                    "input_toks": 4,
                    "output_toks": 2,
                }],
            }])
            router.load_requests(str(workload))

        self.assertEqual(router.route_arrived_requests(0), 1)
        self.assertEqual(prefill.request, [])
        router.freeze_session_admission()
        audit = router.censor_idle_full_model_hbf_native_queues(10)
        self.assertEqual(audit["censored_requests"], 1)
        self.assertEqual(
            router._pending_full_model_hbf_prefill_launches, [])
        self.assertEqual(adapter.calls, {})
        self.assertEqual(bridge.pd_decode_reservations, {})

    def test_pd_capacity_waiter_keeps_fifo_priority_over_new_arrival(self):
        prefill = RecordingScheduler(0, "prefill")
        decode = RecordingScheduler(1, "decode")
        adapter = RecordingAdapter(gpu_resume_mode="recompute")
        bridge = RecordingBridge(
            [prefill, decode],
            pd_pairs=((0, 1),),
            pd_decode_capacity_available=False,
        )
        router = Router(
            2, [prefill, decode], 0,
            full_model_hbf_adapter=adapter,
            full_model_hbf_gpu_hbm_bridge=bridge,
        )

        with TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.jsonl"
            write_workload(workload, [
                {
                    "session_id": "older",
                    "arrival_time_ns": 0,
                    "sub_requests": [{
                        "input_toks": 4,
                        "output_toks": 2,
                    }],
                },
                {
                    "session_id": "newer",
                    "arrival_time_ns": 10,
                    "sub_requests": [{
                        "input_toks": 4,
                        "output_toks": 2,
                    }],
                },
            ])
            router.load_requests(str(workload))

        self.assertEqual(router.route_arrived_requests(0), 1)
        older = router._pending_full_model_hbf_prefill_launches[
            0]["request"]
        bridge.pd_decode_capacity_available = True
        attempts_before = len(bridge.pd_decode_reservation_attempts)

        self.assertEqual(router.route_arrived_requests(10), 1)
        attempts = bridge.pd_decode_reservation_attempts[attempts_before:]
        self.assertGreaterEqual(len(attempts), 1)
        self.assertEqual(attempts[0], int(older.id))
        self.assertEqual(prefill.request[0], older)

    def test_intrinsic_pd_decode_oversize_fails_before_adapter_mutation(self):
        adapter = build_real_recompute_adapter()
        geometry = adapter.gpu_kv_bytes_per_token_per_rank
        prefill = real_finite_scheduler(0, "prefill", geometry)
        decode = real_finite_scheduler(1, "decode", geometry)
        one_block = decode.memory.get_kv(decode.memory.block_size)
        decode.memory.npu_allocatable_mem = (
            decode.memory.weight + one_block)
        decode.memory.npu_mem = decode.memory.npu_allocatable_mem
        bridge = FullModelHBFGPUHBMBridge(
            {0: prefill, 1: decode},
            pd_pairs=((0, 1),),
        )
        router = Router(
            2, [prefill, decode], 0,
            full_model_hbf_adapter=adapter,
            full_model_hbf_gpu_hbm_bridge=bridge,
        )

        with TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.jsonl"
            write_workload(workload, [{
                "session_id": "session-oversize",
                "arrival_time_ns": 0,
                "sub_requests": [{
                    "input_toks": 17,
                    "output_toks": 1,
                }],
            }])
            with self.assertRaises(GPUHBMBridgeCapacityError):
                router.load_requests(str(workload))

        self.assertEqual(adapter.calls, {})
        self.assertEqual(adapter.lifecycle.sessions, {})
        self.assertEqual(router._pending_requests, [])
        self.assertEqual(router._session_templates_loaded, 0)

    def test_terminal_pd_decode_oversize_fails_when_prompt_alone_fits(self):
        adapter = build_real_recompute_adapter()
        geometry = adapter.gpu_kv_bytes_per_token_per_rank
        prefill = real_finite_scheduler(0, "prefill", geometry)
        decode = real_finite_scheduler(1, "decode", geometry)
        one_block = decode.memory.get_kv(decode.memory.block_size)
        decode.memory.npu_allocatable_mem = (
            decode.memory.weight + one_block)
        decode.memory.npu_mem = decode.memory.npu_allocatable_mem
        bridge = FullModelHBFGPUHBMBridge(
            {0: prefill, 1: decode},
            pd_pairs=((0, 1),),
        )
        router = Router(
            2, [prefill, decode], 0,
            full_model_hbf_adapter=adapter,
            full_model_hbf_gpu_hbm_bridge=bridge,
        )

        with TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.jsonl"
            write_workload(workload, [{
                "session_id": "terminal-oversize",
                "arrival_time_ns": 0,
                "sub_requests": [{
                    "input_toks": 16,
                    "output_toks": 2,
                }],
            }])
            with self.assertRaisesRegex(
                    GPUHBMBridgeCapacityError,
                    "terminal_materialized_tokens=17"):
                router.load_requests(str(workload))

        self.assertEqual(adapter.calls, {})
        self.assertEqual(adapter.lifecycle.sessions, {})
        self.assertEqual(router._pending_requests, [])
        self.assertEqual(router._session_templates_loaded, 0)

    def test_real_pd_first_turn_reservation_survives_prefill_handoff(self):
        adapter = build_real_recompute_adapter()
        geometry = adapter.gpu_kv_bytes_per_token_per_rank
        prefill = real_finite_scheduler(0, "prefill", geometry)
        decode = real_finite_scheduler(1, "decode", geometry)
        bridge = FullModelHBFGPUHBMBridge(
            {0: prefill, 1: decode},
            pd_pairs=((0, 1),),
        )
        router = Router(
            2, [prefill, decode], 0,
            full_model_hbf_adapter=adapter,
            full_model_hbf_gpu_hbm_bridge=bridge,
        )

        with TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.jsonl"
            write_workload(workload, [{
                "session_id": "session-handoff",
                "arrival_time_ns": 0,
                "sub_requests": [{
                    "input_toks": 17,
                    "output_toks": 2,
                }],
            }])
            router.load_requests(str(workload))

        self.assertEqual(router.route_arrived_requests(0), 1)
        request = prefill.request.pop()
        reservation = bridge.pd_decode_reservation(request)
        self.assertIsNotNone(reservation)
        self.assertGreater(reservation.reserved_per_rank_bytes, 0)
        request.num_computed_tokens = request.original_input
        request.generated_tokens = 1

        self.assertEqual(
            router.transfer_prefill_request([request], 10), [])
        self.assertEqual(decode.request, [request])
        self.assertIsNone(bridge.pd_decode_reservation(request))
        self.assertEqual(
            decode.memory.npu_used,
            decode.memory.weight + reservation.reserved_per_rank_bytes,
        )
        bridge.assert_invariants()

    def test_pd_reservation_pressure_reclaims_idle_gpu_ready_lru(self):
        hardware = HBFServerHardware(
            hbf_capacity_bytes_per_card=(
                qwen_model_weight_bytes_per_rank(4) + 1),
        )
        adapter = build_real_recompute_adapter(hardware=hardware)
        geometry = adapter.gpu_kv_bytes_per_token_per_rank
        prefill = real_finite_scheduler(0, "prefill", geometry)
        decode = real_finite_scheduler(1, "decode", geometry)

        adapter.offer_raw_request({
            "index": 100,
            "session_id": "idle-victim",
            "sub_request_index": 0,
            "arrival_time_ns": 0,
            "input_toks": 4,
            "output_toks": 6,
            "prefix_reuse_toks": 0,
            "wakekv_has_successor": True,
        }, now_ns=0)
        self.assertIsNone(adapter.complete_native_gpu_request(
            100,
            completion_ns=1,
            materialized_tokens=5,
            gpu_instance_id=1,
        ))
        retain, = adapter.pop_gpu_hbm_events()
        decode.memory.npu_allocatable_mem = (
            decode.memory.weight + retain.per_rank_bytes)
        decode.memory.npu_mem = decode.memory.npu_allocatable_mem
        bridge = FullModelHBFGPUHBMBridge(
            {0: prefill, 1: decode},
            pd_pairs=((0, 1),),
        )
        bridge.apply_event(retain)
        self.assertEqual(
            adapter.lifecycle.sessions["idle-victim"].state.value,
            "gpu_ready",
        )
        self.assertFalse(adapter.has_pending_astra_dispatches())

        router = Router(
            2, [prefill, decode], 0,
            full_model_hbf_adapter=adapter,
            full_model_hbf_gpu_hbm_bridge=bridge,
        )
        with TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.jsonl"
            write_workload(workload, [{
                "session_id": "new-session",
                "arrival_time_ns": 1,
                "sub_requests": [{
                    "input_toks": 4,
                    "output_toks": 2,
                }],
            }])
            router.load_requests(str(workload))

        self.assertEqual(router.route_arrived_requests(1), 1)
        self.assertEqual(len(prefill.request), 1)
        request = prefill.request[0]
        reservation = bridge.pd_decode_reservation(request)
        self.assertIsNotNone(reservation)
        self.assertEqual(
            decode.memory.npu_used,
            decode.memory.weight + reservation.reserved_per_rank_bytes,
        )
        self.assertEqual(
            adapter.lifecycle.sessions["idle-victim"].state.value,
            "evicted",
        )
        report = adapter.report()
        self.assertEqual(
            report["metrics"]["gpu_ready_hbm_pressure_reclaims"], 1)
        reclaim, = report["gpu_ready_hbm_pressure_reclaim_audits"]
        self.assertEqual(reclaim["session_id"], "idle-victim")
        self.assertEqual(reclaim["gpu_instance_id"], 1)
        self.assertEqual(
            bridge.report()["idle_allocations"], [])

        future_resume = adapter.offer_raw_request({
            "index": 101,
            "session_id": "idle-victim",
            "sub_request_index": 1,
            "arrival_time_ns": 2,
            "input_toks": 6,
            "output_toks": 7,
            "prefix_reuse_toks": 5,
            "wakekv_has_successor": False,
        }, now_ns=2)
        self.assertTrue(future_resume.force_gpu_recompute)
        self.assertEqual(future_resume.gpu_prefix_reuse_tokens, 0)
        adapter.assert_invariants()
        bridge.assert_invariants()

    def test_constructor_requires_a_consistent_exclusive_pair(self):
        scheduler = RecordingScheduler(0, "colocated")
        adapter = RecordingAdapter(gpu_resume_mode="sticky_reuse")
        bridge = RecordingBridge([scheduler])

        with self.assertRaisesRegex(ValueError, "requires both"):
            Router(
                1, [scheduler], 0,
                full_model_hbf_adapter=adapter,
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            Router(
                1, [scheduler], 0,
                agentic_kv_manager=object(),
                full_model_hbf_adapter=adapter,
                full_model_hbf_gpu_hbm_bridge=bridge,
            )
        adapter.reclaim_gpu_ready_for_hbm_pressure = None
        with self.assertRaisesRegex(
                TypeError, "reclaim_gpu_ready_for_hbm_pressure"):
            Router(
                1, [scheduler], 0,
                full_model_hbf_adapter=adapter,
                full_model_hbf_gpu_hbm_bridge=bridge,
            )

    def test_all_cotimed_hbf_rows_flush_once_and_keep_session_maps(self):
        scheduler = RecordingScheduler(0, "colocated")
        adapter = RecordingAdapter(
            gpu_resume_mode="sticky_reuse",
            divert_sessions={"session-a", "session-b"},
        )
        bridge = RecordingBridge([scheduler])
        router = Router(
            1, [scheduler], 0,
            full_model_hbf_adapter=adapter,
            full_model_hbf_gpu_hbm_bridge=bridge,
        )

        with TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.jsonl"
            write_workload(workload, [
                {
                    "session_id": "session-a",
                    "arrival_time_ns": 0,
                    "sub_requests": [
                        {"input_toks": 4, "output_toks": 2},
                    ],
                },
                {
                    "session_id": "session-b",
                    "arrival_time_ns": 0,
                    "sub_requests": [
                        {"input_toks": 5, "output_toks": 1},
                    ],
                },
            ])
            router.load_requests(str(workload))

        self.assertEqual(router.route_arrived_requests(0), 2)
        self.assertEqual(adapter.flush_calls, [0])
        self.assertEqual(len(adapter.offers), 2)
        self.assertEqual(scheduler.request, [])
        self.assertEqual(
            router._request_to_session,
            {0: ("session-a", 0), 1: ("session-b", 0)},
        )
        self.assertEqual(
            set(router._deferred_sessions),
            {"session-a", "session-b"},
        )
        self.assertEqual(
            router._active_sessions,
            {"session-a", "session-b"},
        )

    def test_pd_gpu_call_binds_only_after_decode_completion(self):
        prefill = RecordingScheduler(0, "prefill")
        decode = RecordingScheduler(1, "decode")
        adapter = RecordingAdapter(gpu_resume_mode="recompute")
        bridge = RecordingBridge(
            [prefill, decode], pd_pairs=((0, 1),))
        router = Router(
            2, [prefill, decode], 0,
            full_model_hbf_adapter=adapter,
            full_model_hbf_gpu_hbm_bridge=bridge,
        )

        with TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.jsonl"
            write_workload(workload, [{
                "session_id": "session-a",
                "arrival_time_ns": 0,
                "sub_requests": [
                    {
                        "input_toks": 4,
                        "output_toks": 2,
                        "tool_duration_ns": 5,
                    },
                    {
                        "input_toks": 6,
                        "output_toks": 1,
                        "prefix_reuse_toks": 5,
                    },
                ],
            }])
            router.load_requests(str(workload))

        self.assertEqual(router.route_arrived_requests(0), 1)
        first = prefill.request[0]
        self.assertEqual(first.instance_id, 0)
        self.assertIsNone(adapter.calls[first.id].gpu_instance_id)
        self.assertEqual(adapter.bind_calls, [])
        self.assertEqual(bridge.pd_bindings, [])
        self.assertEqual(router._session_decode_affinity["session-a"], 1)

        first.instance_id = 1
        first.num_computed_tokens = first.output - 1
        self.assertEqual(
            adapter.complete_native_gpu_request(
                first, completion_ns=10),
            "migration-0",
        )
        router.drain_full_model_hbf_gpu_hbm_events()
        self.assertEqual(adapter.calls[first.id].gpu_instance_id, 1)
        self.assertEqual(adapter.bind_calls, [(0, 1)])
        self.assertEqual(
            bridge.applied_events[-1]["gpu_instance_id"], 1)

        router.notify_request_completed(first, 10)
        self.assertEqual(router.route_arrived_requests(15), 1)
        continuation = prefill.request[-1]
        self.assertEqual(continuation.id, 1)
        self.assertEqual(continuation.instance_id, 0)
        self.assertEqual(continuation.agentic_kv_hit_tokens, 0)
        self.assertEqual(continuation.num_computed_tokens, 0)
        self.assertEqual(continuation.agentic_kv_recompute_tokens, 5)
        self.assertIsNone(
            adapter.calls[continuation.id].gpu_instance_id)
        self.assertEqual(adapter.bind_calls, [(0, 1)])
        self.assertEqual(bridge.pd_decorations, [(1, 0, 1)])
        self.assertEqual(bridge.pd_bindings, [1])
        self.assertEqual(adapter.flush_calls, [0, 15])
        self.assertEqual(
            router._request_to_session,
            {1: ("session-a", 1)},
        )

    def test_real_adapter_bridge_pd_resume_recomputes_on_prefill(self):
        adapter = build_real_recompute_adapter()
        geometry = adapter.gpu_kv_bytes_per_token_per_rank
        prefill = real_finite_scheduler(0, "prefill", geometry)
        decode = real_finite_scheduler(1, "decode", geometry)
        bridge = FullModelHBFGPUHBMBridge(
            {0: prefill, 1: decode},
            pd_pairs=((0, 1),),
        )
        router = Router(
            2, [prefill, decode], 0,
            full_model_hbf_adapter=adapter,
            full_model_hbf_gpu_hbm_bridge=bridge,
        )

        with TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.jsonl"
            write_workload(workload, [{
                "session_id": "session-real",
                "arrival_time_ns": 0,
                "sub_requests": [
                    {
                        "input_toks": 4,
                        "output_toks": 2,
                        "tool_duration_ns": 5,
                    },
                    {
                        "input_toks": 6,
                        "output_toks": 1,
                        "prefix_reuse_toks": 5,
                    },
                ],
            }])
            router.load_requests(str(workload))

        self.assertEqual(router.route_arrived_requests(0), 1)
        first = prefill.request.pop()
        self.assertIsNone(adapter.calls[first.id].gpu_instance_id)
        first.instance_id = 1
        first.num_computed_tokens = first.output - 1
        migration = adapter.complete_native_gpu_request(
            first, completion_ns=100)
        router.drain_full_model_hbf_gpu_hbm_events()
        self.assertIsNotNone(migration)
        self.assertEqual(adapter.calls[first.id].gpu_instance_id, 1)
        self.assertEqual(
            bridge.report()["idle_allocations"][0]["gpu_instance_id"], 1)

        router.notify_request_completed(first, 100)
        self.assertEqual(router.route_arrived_requests(105), 1)
        continuation = prefill.request[0]
        self.assertEqual(continuation.instance_id, 0)
        self.assertEqual(continuation.agentic_kv_hit_tokens, 0)
        self.assertEqual(continuation.agentic_kv_recompute_tokens, 5)
        self.assertEqual(continuation.num_computed_tokens, 0)
        self.assertIsNone(
            adapter.calls[continuation.id].gpu_instance_id)
        report = bridge.report()
        self.assertEqual(report["idle_allocations"], [])
        self.assertEqual(
            report["bound_pd_recompute_request_ids"],
            [continuation.id],
        )

    def test_measurement_freeze_retries_busy_scheduler_then_censors_queue(self):
        adapter = build_real_recompute_adapter()
        geometry = adapter.gpu_kv_bytes_per_token_per_rank
        prefill = real_finite_scheduler(0, "prefill", geometry)
        decode = real_finite_scheduler(1, "decode", geometry)
        bridge = FullModelHBFGPUHBMBridge(
            {0: prefill, 1: decode},
            pd_pairs=((0, 1),),
        )
        router = Router(
            2, [prefill, decode], 0,
            full_model_hbf_adapter=adapter,
            full_model_hbf_gpu_hbm_bridge=bridge,
        )

        with TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.jsonl"
            write_workload(workload, [{
                "session_id": "session-busy-cutoff",
                "arrival_time_ns": 0,
                "sub_requests": [{
                    "input_toks": 4,
                    "output_toks": 2,
                }],
            }])
            router.load_requests(str(workload))
        self.assertEqual(router.route_arrived_requests(0), 1)

        router.freeze_session_admission()
        prefill.inflight.append(object())
        first = router.censor_idle_full_model_hbf_native_queues(10)
        self.assertEqual(first["censored_requests"], 0)
        self.assertEqual(
            first["skipped_busy_schedulers"][0]["instance_id"], 0)
        self.assertTrue(first["remaining_native_gpu_requests"])
        self.assertFalse(first["accepted_hbf_work_drains"])

        prefill.inflight.clear()
        second = router.censor_idle_full_model_hbf_native_queues(11)
        self.assertEqual(second["censored_requests"], 1)
        self.assertFalse(second["remaining_native_gpu_requests"])
        self.assertEqual(prefill.request, [])
        self.assertFalse(adapter.has_pending())
        summary = router.finalize_measurement_censoring(11)
        self.assertEqual(
            summary["censored_full_model_hbf_native_gpu_requests"], 1)
        self.assertEqual(
            summary["full_model_hbf_accepted_work_policy"],
            "drain_accepted_hbf_censor_native_gpu_queue",
        )
        adapter.assert_invariants()
        bridge.assert_invariants()

    def test_measurement_freeze_releases_hbm_and_cpu_swap_d_queues(self):
        adapter = build_real_recompute_adapter()
        geometry = adapter.gpu_kv_bytes_per_token_per_rank
        prefill = real_finite_scheduler(0, "prefill", geometry)
        decode = real_finite_scheduler(1, "decode", geometry)
        bridge = FullModelHBFGPUHBMBridge(
            {0: prefill, 1: decode},
            pd_pairs=((0, 1),),
        )
        router = Router(
            2, [prefill, decode], 0,
            full_model_hbf_adapter=adapter,
            full_model_hbf_gpu_hbm_bridge=bridge,
        )

        with TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.jsonl"
            write_workload(workload, [
                {
                    "session_id": "session-hbm-queue",
                    "arrival_time_ns": 0,
                    "sub_requests": [{
                        "input_toks": 17,
                        "output_toks": 2,
                    }],
                },
                {
                    "session_id": "session-cpu-queue",
                    "arrival_time_ns": 0,
                    "sub_requests": [{
                        "input_toks": 17,
                        "output_toks": 2,
                    }],
                },
            ])
            router.load_requests(str(workload))
        self.assertEqual(router.route_arrived_requests(0), 2)

        hbm_request, cpu_request = list(prefill.request)
        prefill.request.clear()
        for request in (hbm_request, cpu_request):
            request.instance_id = decode.instance_id
            request.num_computed_tokens = request.output - 1
            request.agentic_kv_owner_instance_id = decode.instance_id
            decode.request.append(request)
        hbm_bytes = decode.memory.get_evict_kv(hbm_request)
        cpu_bytes = decode.memory.get_evict_kv(cpu_request)
        decode.memory.allocate(hbm_bytes, Device.NPU)
        cpu_request.evict = True
        decode.memory.allocate(
            cpu_bytes * decode.num_npus, Device.CPU)

        router.freeze_session_admission()
        audit = router.censor_idle_full_model_hbf_native_queues(20)
        self.assertEqual(audit["censored_requests"], 2)
        by_request = {
            row["request_id"]: row for row in
            audit["censored_request_audits"]
        }
        self.assertEqual(
            by_request[hbm_request.id]["memory"][
                "released_npu_per_rank_bytes"],
            hbm_bytes,
        )
        self.assertEqual(
            by_request[cpu_request.id]["memory"][
                "released_cpu_cluster_bytes"],
            cpu_bytes * decode.num_npus,
        )
        self.assertEqual(decode.memory.npu_used, decode.memory.weight)
        self.assertEqual(decode.memory.cpu_used, 0)
        self.assertEqual(decode.request, [])
        self.assertFalse(adapter.has_pending())
        adapter.assert_invariants()
        bridge.assert_invariants()

    def test_measurement_freeze_keeps_accepted_hbf_astra_obligation(self):
        adapter = build_real_recompute_adapter()
        geometry = adapter.gpu_kv_bytes_per_token_per_rank
        prefill = real_finite_scheduler(0, "prefill", geometry)
        decode = real_finite_scheduler(1, "decode", geometry)
        bridge = FullModelHBFGPUHBMBridge(
            {0: prefill, 1: decode},
            pd_pairs=((0, 1),),
        )
        router = Router(
            2, [prefill, decode], 0,
            full_model_hbf_adapter=adapter,
            full_model_hbf_gpu_hbm_bridge=bridge,
        )

        with TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.jsonl"
            write_workload(workload, [{
                "session_id": "session-accepted-hbf",
                "arrival_time_ns": 0,
                "sub_requests": [
                    {
                        "input_toks": 4,
                        "output_toks": 2,
                        "tool_duration_ns": 5,
                    },
                    {
                        "input_toks": 6,
                        "output_toks": 1,
                    },
                ],
            }])
            router.load_requests(str(workload))
        self.assertEqual(router.route_arrived_requests(0), 1)
        request = prefill.request.pop()
        request.instance_id = decode.instance_id
        request.num_computed_tokens = request.output - 1
        adapter.complete_native_gpu_request(
            request, completion_ns=10)
        router.drain_full_model_hbf_gpu_hbm_events()
        self.assertTrue(adapter.has_pending_astra_dispatches())
        self.assertFalse(adapter.has_pending_native_gpu_requests())

        router.freeze_session_admission()
        audit = router.censor_idle_full_model_hbf_native_queues(10)
        self.assertEqual(audit["censored_requests"], 0)
        self.assertTrue(audit["accepted_hbf_work_drains"])
        self.assertTrue(adapter.has_pending_astra_dispatches())

    def test_legacy_routing_is_unchanged_without_adapter(self):
        scheduler = RecordingScheduler(0, "colocated")
        router = Router(1, [scheduler], 0)

        with TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.jsonl"
            write_workload(workload, [{
                "input_toks": 3,
                "output_toks": 2,
                "arrival_time_ns": 0,
            }])
            router.load_requests(str(workload))

        self.assertEqual(router.route_arrived_requests(0), 1)
        self.assertEqual([request.id for request in scheduler.request], [0])
        self.assertIsNone(router.full_model_hbf_adapter)


if __name__ == "__main__":
    unittest.main()

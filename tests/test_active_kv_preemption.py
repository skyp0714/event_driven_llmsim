import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from serving.core import trace_generator
from serving.core.agentic_kv import (
    AgenticKVConfig,
    AgenticKVManager,
    IdleKVEntry,
    KVLocation,
)
from serving.core.memory_model import Device
from serving.core.request import Request
from serving.core.router import Router
from serving.core.scheduler import Scheduler


class _Logger:
    def info(self, *args, **kwargs):
        pass


class _Memory:
    """Byte-per-token memory model for focused scheduler state tests."""

    def __init__(self, npu_mem, npu_used=0, cpu_mem=1_000_000):
        self.npu_mem = npu_mem
        self.npu_used = npu_used
        self.weight = 0
        self.cpu_mem = cpu_mem
        self.cpu_used = 0

    def get_evict_kv(self, req):
        return int(req.num_computed_tokens)

    def get_total_kv(self, req):
        return int(req.num_computed_tokens)

    def get_kv(self, tokens):
        return int(tokens)

    def get_block_kv(self, batch_req, batch_len, scheduled_tokens=None):
        size = 0
        for req in batch_req[:batch_len]:
            if req.is_prefill():
                size += int(scheduled_tokens[req.id])
            elif not req.evict:
                size += 1
        return size

    def is_avail(self, size, device):
        if device == Device.NPU:
            return self.npu_used + size <= self.npu_mem
        if device == Device.CPU:
            return self.cpu_used + size <= self.cpu_mem
        raise AssertionError(f"unexpected device: {device}")

    def allocate(self, size, device):
        if device == Device.NPU:
            if not self.is_avail(size, device):
                raise RuntimeError("NPU capacity exceeded")
            self.npu_used += size
        elif device == Device.CPU:
            if not self.is_avail(size, device):
                raise RuntimeError("CPU capacity exceeded")
            self.cpu_used += size
        else:
            raise AssertionError(f"unexpected device: {device}")

    def free(self, size, device):
        if device == Device.NPU:
            if self.npu_used - size < self.weight:
                raise RuntimeError("NPU model-weight floor crossed")
            self.npu_used -= size
        elif device == Device.CPU:
            self.cpu_used -= size
            if self.cpu_used < 0:
                raise RuntimeError("negative CPU allocation")
        else:
            raise AssertionError(f"unexpected device: {device}")


def _scheduler(memory, mode="cpu-swap", num_npus=1, token_budget=4):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = "test-model"
    scheduler.instance_id = 0
    scheduler.node_id = 0
    scheduler.start_npu = 0
    scheduler.pd_type = None
    scheduler.num_npus = num_npus
    scheduler.tp_size = num_npus
    scheduler.pp_size = 1
    scheduler.max_num_seqs = 16
    scheduler.max_num_batched_tokens = token_budget
    scheduler.long_prefill_token_threshold = 0
    scheduler.enable_prefix_caching = False
    scheduler.enable_prefix_sharing = False
    scheduler.enable_chunked_prefill = True
    scheduler.prefix_storage = None
    scheduler.prioritize_prefill = False
    scheduler.request = []
    scheduler.inflight = []
    scheduler.done = []
    scheduler.batch_ids = -1
    scheduler.memory = memory
    scheduler.logger = _Logger()
    scheduler.active_preemption_mode = mode
    scheduler.active_recompute_preemptions = 0
    scheduler.active_recompute_tokens = 0
    scheduler.active_cpu_swap_preemptions = 0
    scheduler.active_cpu_swap_write_bytes = 0
    scheduler.active_cpu_swap_read_bytes = 0
    scheduler.agentic_kv_manager = None
    scheduler.memory_wait_until_ns = None
    scheduler.pd_chunk_admission_pass_protected_request_ids = set()
    scheduler.pd_prefill_reclaimability_generation = 0
    return scheduler


def _manager_with_idle_entry(
        scheduler, *, policy="hbm_lru_recompute", next_use_ns=None):
    manager = AgenticKVManager(
        [scheduler],
        AgenticKVConfig(
            policy=policy,
            demotion_mode="capacity-only",
            pcie_bandwidth_gbps=1_000,
            cpu_bandwidth_gbps=1_000,
            cpu_transfer_latency_us=1,
            ssd_read_bandwidth_gbps=1_000,
            ssd_write_bandwidth_gbps=1_000,
            ssd_read_latency_us=1,
            ssd_write_latency_us=1,
        ),
    )
    idle = IdleKVEntry(
        session_id="idle",
        instance_id=scheduler.instance_id,
        tokens=2,
        block_tokens=2,
        per_rank_bytes=2,
        total_bytes=2 * scheduler.num_npus,
        location=KVLocation.HBM,
        tier_since_ns=0,
        last_access_ns=0,
        next_use_ns=next_use_ns,
    )
    manager.entries[idle.session_id] = idle
    return manager, idle


class ActiveKVPreemptionTests(unittest.TestCase):
    @staticmethod
    def _complete_batch(scheduler, request, finish, chunk_len=0):
        request.chunk_len = int(chunk_len)
        batch_id = scheduler.batch_ids + 1
        scheduler.batch_ids = batch_id
        scheduler.inflight.append(SimpleNamespace(
            batch_id=batch_id,
            requests=[request],
            end=[],
        ))
        if scheduler.pd_type == "prefill":
            scheduler.add_done(batch_id + 1, scheduler.start_npu, finish)
            return scheduler.add_done(
                batch_id + 1,
                scheduler.start_npu + scheduler.num_npus * 2 - 1,
                finish,
            )
        return scheduler.add_done(
            batch_id + 1, scheduler.start_npu, finish)

    def _run_pd_request(self, output_tokens, *, chunks, prefix_hit=0):
        prompt_tokens = 4
        request = Request(
            99, "test-model", prompt_tokens,
            prompt_tokens + output_tokens, 0, 0)
        request.prefix_cache_hit = int(prefix_hit)
        request.num_computed_tokens = int(prefix_hit)

        prefill_memory = _Memory(
            npu_mem=100, npu_used=prompt_tokens)
        prefill = _scheduler(prefill_memory, token_budget=prompt_tokens)
        prefill.pd_type = "prefill"
        total_prompt = 0
        generated_total = 0
        for index, chunk in enumerate(chunks):
            prompt, generated, handed_off = self._complete_batch(
                prefill, request, 100 + index * 100, chunk)
            total_prompt += prompt
            generated_total += generated
            if index + 1 < len(chunks):
                self.assertEqual(generated, 0)
                self.assertEqual(handed_off, [])
            else:
                self.assertEqual(generated, 1)
                self.assertEqual(handed_off, [request])

        self.assertEqual(total_prompt, prompt_tokens)
        self.assertEqual(request.generated_tokens, 1)
        prefill_finish = 100 + (len(chunks) - 1) * 100
        self.assertEqual(request.ttft, prefill_finish)
        self.assertEqual(request.itl, [])
        self.assertEqual(prefill_memory.npu_used, 0)

        decode_memory = _Memory(npu_mem=100)
        decode = _scheduler(decode_memory, token_budget=prompt_tokens)
        decode.pd_type = "decode"
        decode.instance_id = 1
        first_decode_finish = 100 + len(chunks) * 100
        handoff_completed = decode.add_decode(
            request, completion_time_ns=prefill_finish)
        if output_tokens == 1:
            self.assertIs(handoff_completed, request)
            self.assertEqual(decode.request, [])
            self.assertEqual(decode.done, [request])
        else:
            self.assertIsNone(handoff_completed)
        for output_index in range(output_tokens - 1):
            decode.request.clear()
            decode_memory.allocate(1, Device.NPU)
            prompt, generated, completed = self._complete_batch(
                decode, request,
                first_decode_finish + output_index * 100)
            self.assertEqual(prompt, 0)
            generated_total += generated
            if output_index + 1 < output_tokens - 1:
                self.assertEqual(completed, [])
            else:
                self.assertEqual(completed, [request])

        self.assertEqual(generated_total, output_tokens)
        self.assertEqual(request.generated_tokens, output_tokens)
        self.assertEqual(request.ttft, prefill_finish)
        self.assertEqual(len(request.itl), max(0, output_tokens - 1))
        self.assertEqual(decode_memory.npu_used, 0)
        return request

    def test_pending_long_pd_head_does_not_consume_small_peer_budget(self):
        scheduler = _scheduler(_Memory(100), token_budget=4)
        scheduler.pd_type = "prefill"
        long_request = Request(1, "test-model", 8, 9, 0, 0)
        small_request = Request(2, "test-model", 1, 2, 0, 0)
        for request in (long_request, small_request):
            request.pd_kv_handoff_tracking_enabled = True
            request.pd_kv_ownership_state = "prefill_active"
            request.agentic_kv_restore_ready_time_ns = 0
        scheduler.request = [long_request, small_request]

        def admit(_scheduler, request, chunk_tokens, _now_ns):
            if request is long_request:
                request.pd_chunk_claim_pending = True
                return False
            request.pd_chunk_admitted_tokens = int(chunk_tokens)
            request.pd_chunk_admission_target_tokens = (
                int(request.num_computed_tokens) + int(chunk_tokens))
            request.pd_chunk_admission_history.append({
                "committed": False,
            })
            return True

        scheduler.pd_chunk_admission_callback = admit
        batch = scheduler.schedule_base(0, 0)
        self.assertIsNotNone(batch)
        self.assertEqual(batch.requests, [small_request])
        self.assertEqual(batch.scheduled_tokens, {small_request.id: 1})
        self.assertTrue(long_request.pd_chunk_claim_pending)

    def test_pd_prefill_emits_first_output_and_decode_emits_remaining(self):
        request = self._run_pd_request(3, chunks=[4])

        self.assertEqual(request.itl, [100, 100])
        self.assertEqual(request.tpot, 100)

    def test_pd_single_output_completes_handoff_without_decode_iteration(self):
        request = self._run_pd_request(1, chunks=[4])

        self.assertEqual(request.generated_tokens, 1)
        self.assertEqual(request.itl, [])
        self.assertEqual(request.tpot, 0)

    def test_pd_chunked_reused_prompt_records_ttft_at_prefill_completion(self):
        request = self._run_pd_request(1, chunks=[2], prefix_hit=2)

        self.assertEqual(request.generated_tokens, 1)
        self.assertEqual(request.ttft, 100)

    def test_colocated_prefill_emits_exactly_one_single_token_output(self):
        memory = _Memory(npu_mem=100, npu_used=4)
        scheduler = _scheduler(memory, token_budget=4)
        request = Request(98, "test-model", 4, 5, 0, 0)

        prompt, generated, completed = self._complete_batch(
            scheduler, request, 100, 4)

        self.assertEqual((prompt, generated), (4, 1))
        self.assertEqual(completed, [request])
        self.assertEqual(request.generated_tokens, 1)
        self.assertEqual(request.ttft, 100)
        self.assertEqual(request.itl, [])
        self.assertEqual(memory.npu_used, 0)

    def test_pd_prefill_trace_contains_output_head_and_sampler(self):
        emitted = []
        ctx = SimpleNamespace(
            perf_db={
                "architecture": {
                    "sequence": {
                        "head": [
                            "final_layernorm", "lm_head", "sampler"]
                    }
                }
            },
            pd_type="prefill",
            node_id=0,
            power_model=None,
        )

        def capture(
                _ctx, _bctx, layer_name, lines, _power_acc,
                _batch_tag, **kwargs):
            emitted.append((layer_name, kwargs.get("output_loc")))
            lines.append(f"{layer_name}\n")

        output = StringIO()
        with patch.object(trace_generator, "_emit_layer", side_effect=capture):
            trace_generator._emit_final_layers(
                ctx, SimpleNamespace(), output)

        self.assertEqual(
            [layer for layer, _ in emitted],
            ["final_layernorm", "lm_head", "sampler"],
        )
        self.assertEqual(emitted[-1][1], "REMOTE:0")

    def _decode_request(self, request_id=1, computed=6):
        req = Request(
            request_id, "test-model", 4, 10, 0, 0, is_init=False)
        req.num_computed_tokens = computed
        req.generated_tokens = computed - req.original_input + 1
        req.ttft = 40
        req.recent_end = 50
        return req

    @staticmethod
    def _pd_partial_request(
            request_id, *, prompt_tokens, computed_tokens,
            prefill_instance_id=0, decode_instance_id=1,
            arrival=0, restored_tokens=0, retained_tokens=0,
            frozen_chunk_tokens=0):
        request = Request(
            request_id, "test-model", prompt_tokens,
            prompt_tokens + 1, arrival, prefill_instance_id)
        request.ready_time = int(arrival)
        request.num_computed_tokens = int(computed_tokens)
        request.agentic_kv_hit_tokens = int(restored_tokens)
        request.prefix_cache_hit = int(restored_tokens)
        request.agentic_kv_restore_ready_time_ns = 0
        request.agentic_kv_owner_instance_id = (
            prefill_instance_id if computed_tokens else None)
        request.agentic_kv_retained_instance_id = (
            decode_instance_id if retained_tokens else None)
        request.agentic_kv_retained_per_rank_bytes = int(retained_tokens)
        request.pd_kv_handoff_tracking_enabled = True
        request.pd_decode_target_instance_id = int(decode_instance_id)
        request.pd_decode_full_per_rank_bytes = int(prompt_tokens)
        request.pd_prefill_full_per_rank_bytes = int(prompt_tokens)
        request.pd_prefill_initial_restored_per_rank_bytes = int(
            restored_tokens)
        request.pd_prefill_reserved_per_rank_bytes = (
            int(computed_tokens) + int(frozen_chunk_tokens)
            - int(restored_tokens))
        request.pd_prefill_owned_per_rank_bytes = (
            int(computed_tokens) + int(frozen_chunk_tokens))
        request.pd_prefill_preallocated_per_rank_bytes = (
            request.pd_prefill_owned_per_rank_bytes)
        request.pd_decode_reserved_per_rank_bytes = (
            int(computed_tokens) + int(frozen_chunk_tokens)
            - int(retained_tokens))
        request.pd_decode_owned_per_rank_bytes = (
            int(computed_tokens) + int(frozen_chunk_tokens))
        request.pd_kv_ownership_state = "prefill_active"
        request.pd_restored_prefix_handoff_sent_tokens = (
            0 if retained_tokens else int(restored_tokens))
        request.pd_new_kv_handoff_sent_tokens = max(
            0, int(computed_tokens) - int(restored_tokens))
        if frozen_chunk_tokens:
            request.pd_chunk_admitted_tokens = int(frozen_chunk_tokens)
            request.pd_chunk_admission_target_tokens = (
                int(computed_tokens) + int(frozen_chunk_tokens))
            request.pd_chunk_admission_history.append({
                "active_prefill_recompute_generation": 0,
                "committed": False,
            })
        return request

    @staticmethod
    def _pd_pair(prefill_capacity, decode_capacity, used):
        prefill = _scheduler(
            _Memory(npu_mem=prefill_capacity, npu_used=used),
            mode="recompute")
        decode = _scheduler(
            _Memory(npu_mem=decode_capacity, npu_used=used),
            mode="recompute")
        prefill.pd_type = "prefill"
        decode.pd_type = "decode"
        decode.instance_id = 1
        decode.start_npu = 1
        for scheduler in (prefill, decode):
            scheduler.block_size = 1
            scheduler.fp = 16
            scheduler.kv_cache_dtype = "auto"
            scheduler.decode_handoff_claim_pending = False
        manager = AgenticKVManager(
            [prefill, decode],
            AgenticKVConfig(
                policy="tiered",
                demotion_mode="capacity-only",
                block_size=1,
            ),
        )
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        return prefill, decode, manager, router

    def test_twenty_partial_pd_prefills_preempt_tail_and_admit_fifo_head(self):
        prefill, decode, manager, router = self._pd_pair(100, 100, 100)
        owner = self._pd_partial_request(
            1, prompt_tokens=100, computed_tokens=81)
        victims = [
            self._pd_partial_request(
                request_id, prompt_tokens=100, computed_tokens=1,
                arrival=request_id)
            for request_id in range(2, 21)
        ]
        original_order = [owner, *victims]
        prefill.request = list(original_order)

        # This is the finite-HBM no-wakeup state from the online run: all P
        # and D bytes belong to active partial prefills, there are no idle LRU
        # entries, no in-flight graph, and the FIFO head needs one more block.
        self.assertIsNone(manager.next_internal_event_time(100))
        self.assertTrue(router.admit_pd_prefill_chunk(
            prefill, owner, 1, 100))

        victim = victims[-1]
        self.assertEqual(prefill.request, original_order)
        self.assertEqual(owner.num_computed_tokens, 81)
        self.assertEqual(owner.pd_chunk_admitted_tokens, 1)
        self.assertEqual(owner.pd_chunk_admission_target_tokens, 82)
        self.assertEqual(owner.pd_prefill_owned_per_rank_bytes, 82)
        self.assertEqual(owner.pd_decode_owned_per_rank_bytes, 82)
        self.assertEqual(victim.num_computed_tokens, 0)
        self.assertEqual(
            victim.active_prefill_recompute_frontier_tokens, 1)
        self.assertEqual(victim.pd_active_prefill_recompute_generation, 1)
        self.assertEqual(victim.pd_prefill_owned_per_rank_bytes, 0)
        self.assertEqual(victim.pd_decode_owned_per_rank_bytes, 0)
        self.assertEqual(prefill.memory.npu_used, 100)
        self.assertEqual(decode.memory.npu_used, 100)
        self.assertEqual(prefill.active_recompute_preemptions, 1)
        self.assertEqual(prefill.active_recompute_tokens, 1)
        self.assertFalse(router._pending_pd_chunk_admissions)
        events = [
            event for event in manager.events
            if event.get("event")
            == "pd_active_prefill_recompute_preempt"
        ]
        self.assertEqual([event["request_id"] for event in events], [20])

    def test_fifo_head_uses_explicit_higher_progress_fallback(self):
        prefill, decode, manager, router = self._pd_pair(10, 10, 10)
        owner = self._pd_partial_request(
            1, prompt_tokens=10, computed_tokens=2, arrival=0)
        advanced_peer = self._pd_partial_request(
            2, prompt_tokens=10, computed_tokens=8, arrival=1)
        prefill.request = [owner, advanced_peer]

        self.assertTrue(router.admit_pd_prefill_chunk(
            prefill, owner, 1, 100))

        self.assertEqual(prefill.request, [owner, advanced_peer])
        self.assertEqual(owner.pd_chunk_admission_target_tokens, 3)
        self.assertEqual(advanced_peer.num_computed_tokens, 0)
        self.assertEqual(
            advanced_peer.active_prefill_recompute_frontier_tokens, 8)
        fallback = [
            event for event in manager.events
            if event.get("event")
            == "pd_active_prefill_fifo_liveness_fallback"
        ]
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0]["request_id"], owner.id)
        self.assertEqual(
            fallback[0]["victim_request_ids"], [advanced_peer.id])

    def test_same_pass_admitted_owner_is_not_reclaimed_by_later_owner(self):
        prefill, decode, _, router = self._pd_pair(10, 10, 9)
        first = self._pd_partial_request(
            1, prompt_tokens=10, computed_tokens=2, arrival=0)
        second = self._pd_partial_request(
            2, prompt_tokens=10, computed_tokens=4, arrival=1)
        tail = self._pd_partial_request(
            3, prompt_tokens=10, computed_tokens=3, arrival=2)
        prefill.request = [first, second, tail]
        scheduled = {first.id: 1, second.id: 1}

        admitted = prefill._admit_pd_prefill_chunks(
            100, [first, second], scheduled)

        self.assertEqual(admitted, [first, second])
        self.assertEqual(scheduled, {first.id: 1, second.id: 1})
        self.assertEqual(first.num_computed_tokens, 2)
        self.assertEqual(first.pd_chunk_admitted_tokens, 1)
        self.assertEqual(first.pd_prefill_owned_per_rank_bytes, 3)
        self.assertEqual(first.pd_decode_owned_per_rank_bytes, 3)
        self.assertEqual(first.pd_active_prefill_recompute_generation, 0)
        self.assertEqual(second.pd_chunk_admitted_tokens, 1)
        self.assertEqual(tail.num_computed_tokens, 0)
        self.assertEqual(tail.pd_active_prefill_recompute_generation, 1)
        self.assertEqual(prefill.request, [first, second, tail])
        self.assertFalse(
            prefill.pd_chunk_admission_pass_protected_request_ids)

    def test_graph_commit_retries_pending_head_at_unchanged_capacity(self):
        prefill, decode, manager, router = self._pd_pair(4, 4, 4)
        owner = self._pd_partial_request(
            30, prompt_tokens=4, computed_tokens=1, arrival=0)
        frozen_victim = self._pd_partial_request(
            31, prompt_tokens=4, computed_tokens=2,
            frozen_chunk_tokens=1, arrival=1)
        prefill.request = [owner, frozen_victim]

        # Both engines are physically full. The only potential victim still
        # owns an admitted graph, so the FIFO head must remain pending and an
        # identical controller poll must be coalesced.
        self.assertFalse(router.admit_pd_prefill_chunk(
            prefill, owner, 1, 100))
        pair = (prefill.instance_id, decode.instance_id)
        handoff = router._pending_pd_chunk_admissions[pair][0]
        cached_state = handoff["last_pair_claim_attempt_state"]
        self.assertEqual(
            cached_state,
            router._pd_handoff_capacity_state(handoff, 100),
        )
        admissions = manager.metrics.active_hbm_reclaim_admissions
        self.assertEqual(router._process_pending_pd_chunk_admissions(100), 0)
        self.assertEqual(
            manager.metrics.active_hbm_reclaim_admissions, admissions)

        manager_state_before = tuple(
            manager.restore_capacity_state(instance_id)
            for instance_id in pair
        )
        batch = SimpleNamespace(
            batch_id=7,
            requests=[frozen_victim],
            pd_restored_prefix_handoff_by_request={},
            pd_new_kv_handoff_by_request={frozen_victim.id: 1},
            pd_kv_handoff_committed=False,
        )
        prefill._commit_pd_kv_handoff(batch)
        frozen_victim.num_computed_tokens += 1

        # Graph commit changes only reclaimability: physical/manager capacity
        # is byte-for-byte identical, while the router retry key advances.
        self.assertEqual(
            manager_state_before,
            tuple(
                manager.restore_capacity_state(instance_id)
                for instance_id in pair
            ),
        )
        self.assertNotEqual(
            cached_state,
            router._pd_handoff_capacity_state(handoff, 100),
        )

        self.assertEqual(router._process_pending_pd_chunk_admissions(100), 1)
        self.assertFalse(router._pending_pd_chunk_admissions)
        self.assertEqual(owner.pd_chunk_admission_target_tokens, 2)
        self.assertEqual(frozen_victim.num_computed_tokens, 0)
        self.assertEqual(
            frozen_victim.active_prefill_recompute_frontier_tokens, 3)
        self.assertEqual(
            frozen_victim.pd_active_prefill_recompute_generation, 1)
        self.assertEqual(prefill.memory.npu_used, 2)
        self.assertEqual(decode.memory.npu_used, 2)

    def test_pd_graph_commit_publishes_one_generation_per_batch(self):
        prefill, _, _, _ = self._pd_pair(8, 8, 4)
        first = self._pd_partial_request(
            32, prompt_tokens=4, computed_tokens=1,
            frozen_chunk_tokens=1, arrival=0)
        second = self._pd_partial_request(
            33, prompt_tokens=4, computed_tokens=1,
            frozen_chunk_tokens=1, arrival=1)
        batch = SimpleNamespace(
            batch_id=8,
            requests=[first, second],
            pd_restored_prefix_handoff_by_request={},
            pd_new_kv_handoff_by_request={first.id: 1, second.id: 1},
            pd_kv_handoff_committed=False,
        )

        prefill._commit_pd_kv_handoff(batch)

        self.assertEqual(prefill.pd_prefill_reclaimability_generation, 1)
        self.assertEqual(first.pd_chunk_admitted_tokens, 0)
        self.assertEqual(second.pd_chunk_admitted_tokens, 0)

        ordinary = Request(34, "test-model", 1, 2, 0, 0)
        ordinary_batch = SimpleNamespace(
            batch_id=9,
            requests=[ordinary],
            pd_restored_prefix_handoff_by_request={},
            pd_new_kv_handoff_by_request={ordinary.id: 1},
            pd_kv_handoff_committed=False,
        )
        prefill._commit_pd_kv_handoff(ordinary_batch)
        self.assertEqual(prefill.pd_prefill_reclaimability_generation, 1)

    def test_partial_pd_filter_reforms_batch_and_refills_token_budget(self):
        prefill = _scheduler(
            _Memory(npu_mem=100, npu_used=2),
            mode="recompute", token_budget=2)
        prefill.pd_type = "prefill"
        prefill.long_prefill_token_threshold = 1
        first = self._pd_partial_request(
            4, prompt_tokens=8, computed_tokens=1, arrival=0)
        blocked = self._pd_partial_request(
            5, prompt_tokens=8, computed_tokens=1, arrival=1)
        ordinary = Request(6, "test-model", 8, 9, 2, 0)
        ordinary.ready_time = 2
        prefill.request = [first, blocked, ordinary]
        callback_counts = {}

        def callback(scheduler, request, chunk_tokens, now_ns):
            del scheduler, now_ns
            callback_counts[request.id] = (
                callback_counts.get(request.id, 0) + 1)
            if request is blocked:
                request.pd_chunk_claim_pending = True
                return False
            if request.pd_chunk_admitted_tokens:
                return True
            request.pd_chunk_admitted_tokens = int(chunk_tokens)
            request.pd_chunk_admission_target_tokens = (
                int(request.num_computed_tokens) + int(chunk_tokens))
            request.pd_chunk_admission_history.append({
                "active_prefill_recompute_generation": 0,
                "committed": False,
            })
            return True

        prefill.pd_chunk_admission_callback = callback
        batch = prefill.schedule_base(100, prefill.start_npu)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.requests, [first, ordinary])
        self.assertEqual(
            batch.scheduled_tokens, {first.id: 1, ordinary.id: 1})
        self.assertEqual(sum(batch.scheduled_tokens.values()), 2)
        self.assertEqual(callback_counts, {first.id: 2, blocked.id: 1})
        self.assertTrue(blocked.pd_chunk_claim_pending)
        self.assertEqual(blocked.chunk_len, 0)

    def test_prefill_preemption_cancels_pending_chunk_and_charges_wait(self):
        prefill, decode, manager, router = self._pd_pair(8, 8, 1)
        victim = self._pd_partial_request(
            7, prompt_tokens=8, computed_tokens=1)
        prefill.request = [victim]
        victim.pd_chunk_claim_pending = True
        pair = (prefill.instance_id, decode.instance_id)
        requirements = router.pd_prefill_chunk_requirements(
            victim, prefill, 1)
        router._pending_pd_chunk_admissions[pair] = [{
            "request": victim,
            "prefill_scheduler": prefill,
            "decode_scheduler": decode,
            "prefill_needed_per_rank_bytes": 1,
            "decode_needed_per_rank_bytes": 1,
            "prefill_claim_ready_ns": None,
            "decode_claim_ready_ns": None,
            "enqueued_ns": 50,
            "requirements": requirements,
        }]

        released = router._preempt_one_pd_prefill(
            victim, prefill, decode, 100)

        self.assertEqual(released, (1, 1, 1))
        self.assertFalse(router._pending_pd_chunk_admissions)
        self.assertFalse(victim.pd_chunk_claim_pending)
        self.assertEqual(victim.pd_chunk_admitted_tokens, 0)
        self.assertEqual(victim.pd_chunk_admission_target_tokens, 0)
        history = victim.pd_chunk_admission_history[-1]
        self.assertTrue(history["preempted_before_commit"])
        self.assertTrue(
            history["invalidated_by_active_prefill_recompute"])
        self.assertTrue(
            history["cancelled_by_active_prefill_recompute"])
        self.assertEqual(victim.pd_chunk_admission_wait_ns_total, 50)
        self.assertEqual(
            victim.pd_chunk_admission_critical_wait_ns_total, 50)
        self.assertEqual(victim.pd_chunk_cancelled_admission_count, 1)
        self.assertEqual(
            victim.pd_chunk_cancelled_admission_wait_ns_total, 50)
        self.assertEqual(
            victim.pd_chunk_successful_admission_wait_ns_total, 0)
        self.assertEqual(
            victim.pd_chunk_admission_wait_ns_total,
            victim.pd_chunk_successful_admission_wait_ns_total
            + victim.pd_chunk_cancelled_admission_wait_ns_total,
        )
        self.assertEqual(prefill.memory.npu_used, 0)
        self.assertEqual(decode.memory.npu_used, 0)
        self.assertEqual(manager.metrics.pd_chunk_cancelled_admissions, 1)
        self.assertEqual(
            manager.metrics.pd_chunk_cancelled_admission_wait_ns, 50)

        # The first successful admission is generation 1 and must drive the
        # legacy first-chunk P/D events. The cancelled generation-0 row above
        # must never be reused through history[0].
        self.assertTrue(router.admit_pd_prefill_chunk(
            prefill, victim, 1, 100))
        chunk_event = next(
            event for event in reversed(manager.events)
            if event.get("event") == "pd_chunk_admission")
        prefill_event = next(
            event for event in reversed(manager.events)
            if event.get("event") == "pd_prefill_active_admission")
        decode_event = next(
            event for event in reversed(manager.events)
            if event.get("event") == "pd_decode_receive_admission")
        self.assertEqual(
            (chunk_event["active_prefill_recompute_generation"],
             chunk_event["prefill_current_per_rank_bytes"],
             chunk_event["prefill_target_per_rank_bytes"]),
            (1, 0, 1),
        )
        self.assertEqual(
            (prefill_event["active_prefill_recompute_generation"],
             prefill_event["initial_owned_per_rank_bytes"],
             prefill_event["target_owned_per_rank_bytes"]),
            (1, 0, 1),
        )
        self.assertEqual(
            (decode_event["active_prefill_recompute_generation"],
             decode_event["initial_owned_per_rank_bytes"],
             decode_event["target_owned_per_rank_bytes"]),
            (1, 0, 1),
        )
        audit = manager._pd_chunk_accounting_audit()
        self.assertEqual(audit["chunk_admissions"], 1)
        self.assertEqual(audit["cancelled_chunk_admissions"], 1)

    def test_frozen_pd_chunk_is_immutable_until_graph_commit(self):
        prefill, decode, _, router = self._pd_pair(8, 8, 2)
        frozen = self._pd_partial_request(
            8, prompt_tokens=8, computed_tokens=1,
            frozen_chunk_tokens=1)
        prefill.request = [frozen]

        with self.assertRaisesRegex(RuntimeError, "frozen until graph"):
            router._preempt_one_pd_prefill(
                frozen, prefill, decode, 100)

        self.assertEqual(frozen.num_computed_tokens, 1)
        self.assertEqual(frozen.pd_chunk_admitted_tokens, 1)
        self.assertEqual(frozen.pd_prefill_owned_per_rank_bytes, 2)
        self.assertEqual(frozen.pd_decode_owned_per_rank_bytes, 2)
        self.assertEqual(prefill.memory.npu_used, 2)
        self.assertEqual(decode.memory.npu_used, 2)

    def test_pending_pair_does_not_reclaim_prior_fifo_frozen_chunk(self):
        prefill, decode, _, router = self._pd_pair(3, 3, 2)
        first = self._pd_partial_request(
            13, prompt_tokens=3, computed_tokens=1, arrival=0)
        second = self._pd_partial_request(
            14, prompt_tokens=3, computed_tokens=1, arrival=1)
        prefill.request = [first, second]
        pair = (prefill.instance_id, decode.instance_id)

        def handoff(request):
            requirements = router.pd_prefill_chunk_requirements(
                request, prefill, 1)
            request.pd_chunk_claim_pending = True
            return {
                "request": request,
                "prefill_scheduler": prefill,
                "decode_scheduler": decode,
                "prefill_needed_per_rank_bytes": 1,
                "decode_needed_per_rank_bytes": 1,
                "enqueued_ns": 100,
                "prefill_claim_ready_ns": None,
                "decode_claim_ready_ns": None,
                "last_pair_claim_attempt_state": None,
                "requirements": requirements,
            }

        first_handoff = handoff(first)
        second_handoff = handoff(second)
        router._pending_pd_chunk_admissions[pair] = [
            first_handoff, second_handoff]

        admitted = router._process_pending_pd_chunk_admissions(100)

        self.assertEqual(admitted, 1)
        self.assertEqual(first.pd_chunk_admitted_tokens, 1)
        self.assertEqual(first.pd_chunk_admission_target_tokens, 2)
        self.assertFalse(first.pd_chunk_claim_pending)
        self.assertEqual(first.num_computed_tokens, 1)
        self.assertTrue(second.pd_chunk_claim_pending)
        self.assertEqual(
            router._pending_pd_chunk_admissions[pair], [second_handoff])
        self.assertEqual(prefill.active_recompute_preemptions, 0)
        self.assertEqual(prefill.memory.npu_used, 3)
        self.assertEqual(decode.memory.npu_used, 3)

    def test_later_same_pass_victim_restores_exact_chunk_before_dispatch(self):
        prefill, decode, _, router = self._pd_pair(5, 5, 5)
        first = self._pd_partial_request(
            15, prompt_tokens=5, computed_tokens=3, arrival=0)
        later = self._pd_partial_request(
            16, prompt_tokens=5, computed_tokens=2, arrival=1)
        prefill.request = [first, later]
        prefill.max_num_batched_tokens = 2
        prefill.long_prefill_token_threshold = 1
        # Router already allocated every proposed P block atomically with D;
        # the scheduler must not allocate those bytes a second time.
        prefill.memory.get_block_kv = (
            lambda batch_req, batch_len, scheduled_tokens=None: 0)

        batch = prefill.schedule_base(100, prefill.start_npu)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.requests, [first, later])
        self.assertEqual(
            batch.scheduled_tokens, {first.id: 1, later.id: 1})
        self.assertEqual(first.chunk_len, 1)
        self.assertEqual(later.chunk_len, 1)
        self.assertEqual(later.num_computed_tokens, 0)
        self.assertEqual(
            later.active_prefill_recompute_frontier_tokens, 2)
        self.assertEqual(later.pd_chunk_admission_target_tokens, 1)

        batch.agentic_astra_dispatch_time_ns = 100
        prefill.add_done(
            batch.batch_id + 1, prefill.start_npu, 200)
        prompt_tokens, generated_tokens, completed = prefill.add_done(
            batch.batch_id + 1,
            prefill.start_npu + prefill.num_npus * 2 - 1,
            200,
        )

        self.assertEqual((prompt_tokens, generated_tokens), (1, 0))
        self.assertEqual(completed, [])
        self.assertEqual(first.num_computed_tokens, 4)
        self.assertEqual(later.num_computed_tokens, 1)
        self.assertEqual(first.pd_new_kv_handoff_sent_tokens, 4)
        self.assertEqual(later.pd_new_kv_handoff_sent_tokens, 1)
        self.assertEqual(prefill.memory.npu_used, 5)
        self.assertEqual(decode.memory.npu_used, 5)
        self.assertFalse(router._pending_pd_chunk_admissions)

    def test_router_rejects_current_pass_victim_without_freeing_pd_kv(self):
        prefill, decode, _, router = self._pd_pair(2, 2, 2)
        protected = self._pd_partial_request(
            9, prompt_tokens=3, computed_tokens=2)
        prefill.request = [protected]
        prefill.pd_chunk_admission_pass_protected_request_ids.add(
            protected.id)

        with self.assertRaisesRegex(RuntimeError, "current batch"):
            router._preempt_one_pd_prefill(
                protected, prefill, decode, 100)

        self.assertEqual(protected.num_computed_tokens, 2)
        self.assertEqual(protected.pd_prefill_owned_per_rank_bytes, 2)
        self.assertEqual(protected.pd_decode_owned_per_rank_bytes, 2)
        self.assertEqual(prefill.memory.npu_used, 2)
        self.assertEqual(decode.memory.npu_used, 2)

    def test_atomic_pd_prefill_rejects_colocated_roles(self):
        scheduler = _scheduler(
            _Memory(npu_mem=4, npu_used=2), mode="recompute")
        scheduler.pd_type = "prefill"
        scheduler.block_size = 1
        scheduler.decode_handoff_claim_pending = False
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="tiered", demotion_mode="capacity-only",
                block_size=1),
        )
        router = Router(
            1, [scheduler], 0, "RR", agentic_kv_manager=manager)
        request = self._pd_partial_request(
            12, prompt_tokens=4, computed_tokens=2,
            decode_instance_id=0)
        scheduler.request = [request]

        with self.assertRaisesRegex(RuntimeError, "distinct prefill"):
            router.pd_prefill_chunk_requirements(
                request, scheduler, 1)
        with self.assertRaisesRegex(RuntimeError, "distinct prefill"):
            router._preempt_one_pd_prefill(
                request, scheduler, scheduler, 100)

        self.assertEqual(scheduler.memory.npu_used, 2)
        self.assertEqual(request.num_computed_tokens, 2)

    def test_strict_pd_rejects_many_to_one_prefill_mapping(self):
        first_prefill = _scheduler(
            _Memory(npu_mem=4), mode="recompute")
        second_prefill = _scheduler(
            _Memory(npu_mem=4), mode="recompute")
        decode = _scheduler(_Memory(npu_mem=4), mode="recompute")
        first_prefill.pd_type = "prefill"
        second_prefill.pd_type = "prefill"
        second_prefill.instance_id = 1
        second_prefill.start_npu = 1
        decode.pd_type = "decode"
        decode.instance_id = 2
        decode.start_npu = 2
        for scheduler in (first_prefill, second_prefill, decode):
            scheduler.block_size = 1
            scheduler.fp = 16
            scheduler.kv_cache_dtype = "auto"

        manager = AgenticKVManager(
            [first_prefill, second_prefill, decode],
            AgenticKVConfig(
                policy="tiered", demotion_mode="capacity-only",
                block_size=1),
        )
        with self.assertRaisesRegex(RuntimeError, "injective P-to-D"):
            Router(
                3, [first_prefill, second_prefill, decode], 0, "RR",
                agentic_kv_manager=manager)

    def test_pd_preemption_preflights_both_model_weight_floors_atomically(self):
        for constrained_role in ("prefill", "decode"):
            with self.subTest(constrained_role=constrained_role):
                prefill, decode, _, router = self._pd_pair(8, 8, 2)
                victim = self._pd_partial_request(
                    17, prompt_tokens=8, computed_tokens=2)
                prefill.request = [victim]
                constrained = (
                    prefill if constrained_role == "prefill" else decode)
                constrained.memory.weight = 1

                with self.assertRaisesRegex(
                        RuntimeError, "model-weight floor"):
                    router._preempt_one_pd_prefill(
                        victim, prefill, decode, 100)

                self.assertEqual(prefill.memory.npu_used, 2)
                self.assertEqual(decode.memory.npu_used, 2)
                self.assertEqual(victim.num_computed_tokens, 2)
                self.assertEqual(
                    victim.pd_prefill_owned_per_rank_bytes, 2)
                self.assertEqual(
                    victim.pd_decode_owned_per_rank_bytes, 2)
                self.assertEqual(
                    victim.pd_active_prefill_recompute_generation, 0)

    def test_pd_preemption_rejects_bad_physical_owner_without_mutation(self):
        for corrupt_role in ("prefill", "retained_decode"):
            with self.subTest(corrupt_role=corrupt_role):
                prefill, decode, manager, router = self._pd_pair(4, 4, 2)
                retained = 1 if corrupt_role == "retained_decode" else 0
                victim = self._pd_partial_request(
                    18, prompt_tokens=4, computed_tokens=2,
                    retained_tokens=retained)
                prefill.request = [victim]
                if corrupt_role == "prefill":
                    victim.agentic_kv_owner_instance_id = 99
                    error = "physical P owner changed"
                else:
                    victim.agentic_kv_retained_instance_id = 99
                    error = "retained D owner changed"

                requirements = router.pd_prefill_chunk_requirements(
                    victim, prefill, 1)
                pair = (prefill.instance_id, decode.instance_id)
                handoff = {
                    "request": victim,
                    "prefill_scheduler": prefill,
                    "decode_scheduler": decode,
                    "prefill_needed_per_rank_bytes": 1,
                    "decode_needed_per_rank_bytes": 1,
                    "prefill_claim_ready_ns": 100,
                    "decode_claim_ready_ns": 100,
                    "enqueued_ns": 100,
                    "last_pair_claim_attempt_state": None,
                    "requirements": requirements,
                }
                victim.pd_chunk_claim_pending = True
                router._pending_pd_chunk_admissions[pair] = [handoff]
                for scheduler in (prefill, decode):
                    self.assertEqual(
                        manager.claim_active_hbm_reclaim(
                            scheduler.instance_id, 1, 100,
                            owner_kind="pd", owner_id=victim.id),
                        100,
                    )
                    scheduler.decode_handoff_claim_pending = True
                prefill_claim = manager.active_hbm_reclaim_claim(
                    prefill.instance_id)
                decode_claim = manager.active_hbm_reclaim_claim(
                    decode.instance_id)
                events_before = list(manager.events)
                history_before = list(victim.pd_chunk_admission_history)

                with self.assertRaisesRegex(RuntimeError, error):
                    router._preempt_one_pd_prefill(
                        victim, prefill, decode, 100)

                self.assertEqual(
                    router._pending_pd_chunk_admissions[pair], [handoff])
                self.assertIs(
                    manager.active_hbm_reclaim_claim(prefill.instance_id),
                    prefill_claim)
                self.assertIs(
                    manager.active_hbm_reclaim_claim(decode.instance_id),
                    decode_claim)
                self.assertEqual(manager.events, events_before)
                self.assertEqual(
                    victim.pd_chunk_admission_history, history_before)
                self.assertTrue(victim.pd_chunk_claim_pending)
                self.assertEqual(prefill.memory.npu_used, 2)
                self.assertEqual(decode.memory.npu_used, 2)
                self.assertEqual(victim.num_computed_tokens, 2)
                self.assertEqual(
                    victim.pd_active_prefill_recompute_generation, 0)

    def test_pd_chunk_finalize_preflights_decode_before_consuming_prefill(self):
        for corruption in (
                "decode_claim", "decode_headroom", "target_parity",
                "ready_timestamp"):
            with self.subTest(corruption=corruption):
                prefill, decode, manager, router = self._pd_pair(4, 4, 2)
                request = self._pd_partial_request(
                    19, prompt_tokens=4, computed_tokens=2)
                prefill.request = [request]
                requirements = router.pd_prefill_chunk_requirements(
                    request, prefill, 1)
                handoff = {
                    "request": request,
                    "prefill_scheduler": prefill,
                    "decode_scheduler": decode,
                    "prefill_needed_per_rank_bytes": 1,
                    "decode_needed_per_rank_bytes": 1,
                    "prefill_claim_ready_ns": 100,
                    "decode_claim_ready_ns": 100,
                    "enqueued_ns": 100,
                    "last_pair_claim_attempt_state": None,
                    "requirements": requirements,
                }
                request.pd_chunk_claim_pending = True
                pair = (prefill.instance_id, decode.instance_id)
                router._pending_pd_chunk_admissions[pair] = [handoff]
                self.assertEqual(manager.claim_active_hbm_reclaim(
                    prefill.instance_id, 1, 100,
                    owner_kind="pd", owner_id=request.id), 100)
                decode_owner = (
                    request.id + 1
                    if corruption == "decode_claim" else request.id)
                self.assertEqual(manager.claim_active_hbm_reclaim(
                    decode.instance_id, 1, 100,
                    owner_kind="pd", owner_id=decode_owner), 100)
                if corruption == "decode_headroom":
                    decode.memory.npu_used = decode.memory.npu_mem
                    error = "physical allocator headroom"
                elif corruption == "target_parity":
                    requirements["decode_target_per_rank_bytes"] += 1
                    error = "block parity"
                elif corruption == "ready_timestamp":
                    handoff["decode_claim_ready_ns"] = None
                    error = "capacity-ready timestamp"
                else:
                    error = "exact HBM claim"
                prefill_claim = manager.active_hbm_reclaim_claim(
                    prefill.instance_id)
                decode_claim = manager.active_hbm_reclaim_claim(
                    decode.instance_id)
                events_before = list(manager.events)

                with self.assertRaisesRegex(RuntimeError, error):
                    router._finalize_pd_chunk_admission(handoff, 100)

                self.assertIs(
                    manager.active_hbm_reclaim_claim(prefill.instance_id),
                    prefill_claim)
                self.assertIs(
                    manager.active_hbm_reclaim_claim(decode.instance_id),
                    decode_claim)
                self.assertEqual(manager.events, events_before)
                self.assertEqual(
                    router._pending_pd_chunk_admissions[pair], [handoff])
                self.assertEqual(prefill.memory.npu_used, 2)
                self.assertEqual(
                    decode.memory.npu_used,
                    4 if corruption == "decode_headroom" else 2)
                self.assertEqual(request.num_computed_tokens, 2)
                self.assertEqual(
                    request.pd_prefill_owned_per_rank_bytes, 2)
                self.assertEqual(
                    request.pd_decode_owned_per_rank_bytes, 2)
                self.assertTrue(request.pd_chunk_claim_pending)
                self.assertFalse(request.pd_chunk_admission_history)

    def test_restored_prefix_replay_counts_logical_prompt_exactly_once(self):
        request = Request(88, "test-model", 4, 5, 0, 0)
        request.prefix_cache_hit = 2
        request.agentic_kv_hit_tokens = 2
        request.num_computed_tokens = 3
        discarded = request.begin_active_prefill_recompute()
        self.assertEqual(discarded, 3)

        prefill = _scheduler(_Memory(npu_mem=8, npu_used=4), token_budget=2)
        prefill.pd_type = "prefill"
        first_prompt, _, _ = self._complete_batch(
            prefill, request, 100, chunk_len=2)
        second_prompt, _, handed_off = self._complete_batch(
            prefill, request, 200, chunk_len=2)

        # One fresh suffix token was credited before preemption. Replayed
        # [0,3) contributes zero, the new final token contributes one, and the
        # original two-token cache hit is credited once at completion.
        self.assertEqual(first_prompt, 0)
        self.assertEqual(second_prompt, 3)
        self.assertEqual(1 + first_prompt + second_prompt, 4)
        self.assertEqual(handed_off, [request])
        self.assertEqual(request.prefix_cache_hit, 2)
        self.assertEqual(request.agentic_kv_hit_tokens, 2)
        self.assertEqual(request.active_recompute_tokens, 3)

    def test_restore_ready_prefix_and_retained_decode_copy_are_discarded(self):
        prefill, decode, manager, router = self._pd_pair(3, 3, 3)
        request = self._pd_partial_request(
            10, prompt_tokens=4, computed_tokens=3,
            restored_tokens=2, retained_tokens=2)
        request.pd_new_kv_handoff_sent_tokens = 1
        prefill.request = [request]

        released = router._preempt_one_pd_prefill(
            request, prefill, decode, 100)

        self.assertEqual(released, (3, 3, 3))
        self.assertEqual(prefill.memory.npu_used, 0)
        self.assertEqual(decode.memory.npu_used, 0)
        self.assertEqual(request.num_computed_tokens, 0)
        self.assertEqual(request.agentic_kv_hit_tokens, 2)
        self.assertEqual(request.prefix_cache_hit, 2)
        self.assertIsNone(request.agentic_kv_owner_instance_id)
        self.assertIsNone(request.agentic_kv_retained_instance_id)
        self.assertEqual(request.agentic_kv_retained_per_rank_bytes, 0)
        self.assertEqual(
            request.pd_prefill_initial_restored_per_rank_bytes, 0)
        self.assertEqual(request.pd_restored_prefix_handoff_sent_tokens, 0)
        self.assertEqual(request.pd_new_kv_handoff_sent_tokens, 0)
        self.assertEqual(
            manager.metrics.pd_active_prefill_recompute_preemptions, 1)
        self.assertEqual(
            manager.metrics.pd_active_prefill_recompute_tokens, 3)
        self.assertEqual(
            manager.metrics
            .agentic_kv_restored_tokens_discarded_by_active_prefill_recompute,
            2,
        )
        progress = next(
            event for event in manager.events
            if event.get("event") == "pd_active_prefill_recompute_preempt")
        self.assertEqual(progress["restored_hit_tokens_discarded"], 2)
        self.assertEqual(
            progress["cumulative_active_prefill_recompute_tokens"], 3)
        self.assertEqual(
            progress["cumulative_restored_hit_tokens_discarded"], 2)
        self.assertEqual(
            manager._pd_active_prefill_recompute_accounting_audit()[
                "status"],
            "ok",
        )
        for invalid in (True, "3"):
            with self.subTest(invalid=invalid):
                progress["discarded_tokens"] = invalid
                with self.assertRaisesRegex(
                        RuntimeError, "invalid integers"):
                    manager._pd_active_prefill_recompute_accounting_audit()
        progress["discarded_tokens"] = 3

        # The logical hit remains in the request/report, but the next graph
        # now computes and sends that prefix as ordinary new KV.
        self.assertTrue(router.admit_pd_prefill_chunk(
            prefill, request, 2, 100))
        self.assertEqual(request.pd_chunk_admission_target_tokens, 2)
        self.assertEqual(request.pd_prefill_owned_per_rank_bytes, 2)
        self.assertEqual(request.pd_decode_owned_per_rank_bytes, 2)

    def test_repeated_prefill_preemption_counts_each_discarded_replay(self):
        request = Request(11, "test-model", 10, 11, 0, 0)
        request.agentic_kv_hit_tokens = 3
        request.num_computed_tokens = 6
        self.assertEqual(request.begin_active_prefill_recompute(), 6)
        self.assertEqual(
            request
            .agentic_kv_restored_tokens_discarded_by_active_prefill_recompute,
            3,
        )
        # A later replay can be preempted before it reaches the original
        # restored-prefix length; those hits were already charged above.
        request.num_computed_tokens = 2
        self.assertEqual(request.begin_active_prefill_recompute(), 2)

        self.assertEqual(
            request.active_prefill_recompute_frontier_tokens, 6)
        self.assertEqual(request.active_prefill_recompute_preemptions, 2)
        self.assertEqual(request.active_prefill_recompute_tokens, 8)
        self.assertEqual(request.active_recompute_tokens, 8)
        self.assertEqual(
            request
            .agentic_kv_restored_tokens_discarded_by_active_prefill_recompute,
            3,
        )
        self.assertEqual(request.pd_active_prefill_recompute_generation, 2)
        self.assertEqual(request.num_computed_tokens, 0)

    def test_router_repeated_preemption_below_restored_hit_frontier(self):
        prefill, decode, manager, router = self._pd_pair(10, 10, 6)
        request = self._pd_partial_request(
            12,
            prompt_tokens=10,
            computed_tokens=6,
            restored_tokens=3,
            retained_tokens=3,
        )
        request.session_id = "two-generation"
        prefill.request = [request]

        self.assertEqual(
            router._preempt_one_pd_prefill(
                request, prefill, decode, 100),
            (6, 6, 6),
        )
        self.assertEqual(request.pd_active_prefill_recompute_generation, 1)
        self.assertEqual(
            request
            .agentic_kv_restored_tokens_discarded_by_active_prefill_recompute,
            3,
        )

        # Admit and commit a two-token generation-1 replay. This frontier is
        # deliberately below the original three-token restored hit. The hit
        # ownership was already discarded by generation 0, so a second
        # router-level preemption must remain valid.
        self.assertTrue(router.admit_pd_prefill_chunk(
            prefill, request, 2, 150))
        replay_batch = SimpleNamespace(
            batch_id=1,
            requests=[request],
            pd_restored_prefix_handoff_by_request={},
            pd_new_kv_handoff_by_request={request.id: 2},
            pd_kv_handoff_committed=False,
        )
        prefill._commit_pd_kv_handoff(replay_batch)
        request.num_computed_tokens = 2

        self.assertEqual(request.pd_prefill_owned_per_rank_bytes, 2)
        self.assertEqual(request.pd_decode_owned_per_rank_bytes, 2)
        self.assertEqual(prefill.memory.npu_used, 2)
        self.assertEqual(decode.memory.npu_used, 2)
        self.assertEqual(
            router._preempt_one_pd_prefill(
                request, prefill, decode, 200),
            (2, 2, 2),
        )

        self.assertEqual(prefill.request, [request])
        self.assertEqual(request.pd_kv_ownership_state, "prefill_active")
        self.assertIsNone(request.agentic_kv_owner_instance_id)
        self.assertIsNone(request.agentic_kv_retained_instance_id)
        self.assertEqual(request.pd_prefill_owned_per_rank_bytes, 0)
        self.assertEqual(request.pd_decode_owned_per_rank_bytes, 0)
        self.assertEqual(prefill.memory.npu_used, 0)
        self.assertEqual(decode.memory.npu_used, 0)
        self.assertEqual(request.pd_active_prefill_recompute_generation, 2)
        self.assertEqual(request.active_prefill_recompute_preemptions, 2)
        self.assertEqual(request.active_prefill_recompute_tokens, 8)
        self.assertEqual(request.active_prefill_recompute_frontier_tokens, 6)
        self.assertEqual(
            request
            .agentic_kv_restored_tokens_discarded_by_active_prefill_recompute,
            3,
        )
        self.assertEqual(
            manager.metrics.pd_active_prefill_recompute_preemptions, 2)
        self.assertEqual(
            manager.metrics.pd_active_prefill_recompute_tokens, 8)
        self.assertEqual(
            manager.metrics
            .agentic_kv_restored_tokens_discarded_by_active_prefill_recompute,
            3,
        )
        progress = [
            event for event in manager.events
            if event.get("event") == "pd_active_prefill_recompute_preempt"
        ]
        self.assertEqual(
            [event["restored_hit_tokens_discarded"] for event in progress],
            [3, 0],
        )
        self.assertEqual(
            [
                (event["old_active_prefill_recompute_generation"],
                 event["new_active_prefill_recompute_generation"])
                for event in progress
            ],
            [(0, 1), (1, 2)],
        )
        self.assertEqual(
            manager._pd_active_prefill_recompute_accounting_audit()[
                "status"],
            "ok",
        )

        # Generation 2 can immediately reacquire atomically paired P/D
        # capacity; the session and request were never dropped or reordered.
        self.assertTrue(router.admit_pd_prefill_chunk(
            prefill, request, 1, 200))
        self.assertEqual(request.pd_prefill_owned_per_rank_bytes, 1)
        self.assertEqual(request.pd_decode_owned_per_rank_bytes, 1)
        self.assertEqual(
            request.pd_chunk_admission_history[-1][
                "active_prefill_recompute_generation"],
            2,
        )

    def test_recompute_preemption_rebuilds_context_without_cpu_traffic(self):
        memory = _Memory(npu_mem=6, npu_used=6)
        scheduler = _scheduler(memory, mode="recompute", token_budget=4)
        req = self._decode_request(computed=6)
        scheduler.request.append(req)

        # The next decode block cannot fit. The first scheduling attempt
        # discards physical KV and leaves the request queued for recomputation.
        self.assertIsNone(scheduler.schedule_base(100, 0))
        self.assertEqual(memory.npu_used, 0)
        self.assertEqual(memory.cpu_used, 0)
        self.assertFalse(req.evict)
        self.assertEqual(req.recompute_target_tokens, 6)
        self.assertEqual(req.num_computed_tokens, 0)
        self.assertEqual(req.active_recompute_tokens, 6)
        self.assertEqual(scheduler.active_recompute_preemptions, 1)
        self.assertEqual(scheduler.active_recompute_tokens, 6)
        self.assertEqual(scheduler.active_cpu_swap_write_bytes, 0)

        first = scheduler.schedule_base(100, 0)
        self.assertIsNotNone(first)
        self.assertEqual(first.scheduled_tokens, {req.id: 4})
        self.assertEqual(first.evict, 0)
        self.assertEqual(first.load, 0)
        prompt, generated, done = scheduler.add_done(1, 0, 200)
        self.assertEqual((prompt, generated, done), (0, 0, []))
        self.assertEqual(req.num_computed_tokens, 4)
        self.assertEqual(req.recompute_target_tokens, 6)
        self.assertEqual(req.ttft, 40)
        self.assertEqual(req.itl, [])

        second = scheduler.schedule_base(200, 0)
        self.assertIsNotNone(second)
        self.assertEqual(second.scheduled_tokens, {req.id: 2})
        self.assertEqual(second.evict, 0)
        self.assertEqual(second.load, 0)
        prompt, generated, done = scheduler.add_done(2, 0, 300)

        # Rebuilding the final context token also emits the next output. It is
        # not a second prompt and must not overwrite the original TTFT.
        self.assertEqual((prompt, generated, done), (0, 1, []))
        self.assertEqual(req.num_computed_tokens, 6)
        self.assertIsNone(req.recompute_target_tokens)
        self.assertFalse(req.is_prefill())
        self.assertEqual(req.ttft, 40)
        self.assertEqual(req.itl, [250])
        self.assertEqual(memory.npu_used, 6)
        self.assertEqual(memory.cpu_used, 0)
        self.assertEqual(scheduler.active_cpu_swap_read_bytes, 0)

    def test_cpu_swap_mode_preserves_legacy_spill_and_trace_bytes(self):
        memory = _Memory(npu_mem=12, npu_used=12)
        scheduler = _scheduler(
            memory, mode="cpu-swap", num_npus=2, token_budget=4)
        first = self._decode_request(request_id=1, computed=6)
        second = self._decode_request(request_id=2, computed=6)
        scheduler.request.extend([first, second])

        batch = scheduler.schedule_base(100, 0)

        self.assertIsNotNone(batch)
        self.assertEqual([req.id for req in batch.requests], [first.id])
        self.assertEqual(batch.evict, 6)
        self.assertEqual(batch.load, 0)
        self.assertTrue(second.evict)
        self.assertEqual(second.num_computed_tokens, 6)
        self.assertIsNone(second.recompute_target_tokens)
        self.assertEqual(memory.cpu_used, 12)
        self.assertEqual(scheduler.active_cpu_swap_preemptions, 1)
        self.assertEqual(scheduler.active_cpu_swap_write_bytes, 12)
        self.assertEqual(scheduler.active_recompute_preemptions, 0)
        self.assertEqual(scheduler.active_recompute_tokens, 0)

    def test_active_cpu_swap_falls_back_before_overbooking_bounce_dram(self):
        memory = _Memory(npu_mem=6, npu_used=6, cpu_mem=6)
        scheduler = _scheduler(
            memory, mode="cpu-swap", num_npus=1, token_budget=4)
        manager = AgenticKVManager(
            [scheduler], AgenticKVConfig(
                policy="tiered", demotion_mode="capacity-only"))
        manager._record_transient_dram_reservation(
            scheduler=scheduler,
            session_id="restore",
            start_ns=0,
            complete_ns=1_000,
            num_bytes=6,
            reservation_sequence=0,
            peak_committed_bytes=6,
            capacity_wait_ns=0,
            pressure_stall_ns=0,
        )
        req = self._decode_request(computed=6)

        released = scheduler._preempt_decode_request(req, 100)

        self.assertEqual(released, 6)
        self.assertFalse(req.evict)
        self.assertEqual(req.recompute_target_tokens, 6)
        self.assertEqual(memory.cpu_used, 0)
        self.assertEqual(scheduler.active_cpu_swap_preemptions, 0)
        self.assertEqual(scheduler.active_recompute_preemptions, 1)
        self.assertEqual(manager.metrics.active_cpu_swap_capacity_fallbacks, 1)

    def test_request_rejects_recompute_before_decode(self):
        req = Request(1, "test-model", 8, 12, 0, 0)
        req.num_computed_tokens = 7

        with self.assertRaisesRegex(RuntimeError, "during prefill"):
            req.begin_active_recompute()

    def test_idle_hbm_lru_is_dropped_before_active_decode_recompute(self):
        memory = _Memory(npu_mem=8, npu_used=8)
        scheduler = _scheduler(memory, mode="recompute")
        manager, idle = _manager_with_idle_entry(scheduler)
        req = self._decode_request(computed=6)
        scheduler.request.append(req)

        batch = scheduler.schedule_base(100, 0)

        self.assertIsNotNone(batch)
        self.assertEqual(idle.location, KVLocation.DROPPED)
        self.assertEqual(idle.drop_reason, "hbm_capacity")
        self.assertEqual(memory.npu_used, 7)
        self.assertEqual(scheduler.active_recompute_preemptions, 0)
        self.assertIsNone(scheduler.memory_wait_until_ns)
        self.assertEqual(manager.metrics.active_hbm_reclaim_admissions, 1)
        self.assertEqual(manager.metrics.active_hbm_reclaim_wait_ns, 0)

    def test_tiered_idle_demotion_blocks_without_duplicate_active_preemption(self):
        memory = _Memory(npu_mem=8, npu_used=8)
        scheduler = _scheduler(memory, mode="recompute")
        manager, idle = _manager_with_idle_entry(scheduler, policy="tiered")
        req = self._decode_request(computed=6)
        scheduler.request.append(req)

        self.assertIsNone(scheduler.schedule_base(100, 0))
        ready_ns = req.agentic_hbm_reclaim_ready_time_ns
        self.assertIsNotNone(ready_ns)
        self.assertGreater(ready_ns, 100)
        self.assertIsNone(scheduler.memory_wait_until_ns)
        self.assertEqual(idle.location, KVLocation.HBM)
        self.assertEqual(manager.metrics.transfer_jobs, 1)
        self.assertEqual(scheduler.active_recompute_preemptions, 0)

        self.assertIsNone(scheduler.schedule_base(ready_ns - 1, 0))
        self.assertEqual(manager.metrics.transfer_jobs, 1)
        self.assertEqual(idle.location, KVLocation.HBM)

        batch = scheduler.schedule_base(ready_ns, 0)
        self.assertIsNotNone(batch)
        self.assertEqual(idle.location, KVLocation.CPU)
        self.assertEqual(memory.npu_used, 7)
        self.assertEqual(memory.cpu_used, 2)
        self.assertEqual(scheduler.active_recompute_preemptions, 0)
        self.assertEqual(manager.metrics.active_hbm_reclaim_admissions, 1)
        self.assertEqual(
            manager.metrics.active_hbm_reclaim_wait_ns,
            ready_ns - 100,
        )

    def test_future_reclaim_defers_only_owner_and_dispatches_hbm_peer(self):
        memory = _Memory(npu_mem=8, npu_used=8)
        scheduler = _scheduler(memory, mode="recompute")
        manager, _ = _manager_with_idle_entry(
            scheduler, policy="tiered")
        owner = self._decode_request(request_id=1, computed=6)
        peer = self._decode_request(request_id=2, computed=6)
        scheduler.request.extend([owner, peer])
        default_get_block_kv = memory.get_block_kv

        def request_local_block_kv(batch_req, batch_len, scheduled_tokens=None):
            selected = batch_req[:batch_len]
            if selected and all(req.id == peer.id for req in selected):
                # The peer remains within its already allocated KV block.
                return 0
            return default_get_block_kv(
                batch_req, batch_len, scheduled_tokens)

        memory.get_block_kv = request_local_block_kv

        batch = scheduler.schedule_base(100, 0)

        self.assertIsNotNone(batch)
        self.assertEqual([req.id for req in batch.requests], [peer.id])
        ready_ns = owner.agentic_hbm_reclaim_ready_time_ns
        self.assertGreater(ready_ns, 100)
        claim = manager.active_hbm_reclaim_claim(0)
        self.assertEqual((claim.owner_kind, claim.owner_id), ("scheduler", 1))
        self.assertIsNone(scheduler.memory_wait_until_ns)

        scheduler.inflight.clear()
        owner_batch = scheduler.schedule_base(ready_ns, 0)
        self.assertEqual(
            [req.id for req in owner_batch.requests], [owner.id])
        self.assertIsNone(manager.active_hbm_reclaim_claim(0))
        consume_events = [
            event for event in manager.events
            if event.get("event") == "active_hbm_reclaim_consume"
        ]
        self.assertEqual(len(consume_events), 1)
        self.assertEqual(consume_events[0]["owner_id"], owner.id)

    def test_pd_claims_do_not_block_unreserved_work_on_either_engine(self):
        memories = [_Memory(npu_mem=8, npu_used=8) for _ in range(2)]
        schedulers = [
            _scheduler(memory, mode="recompute")
            for memory in memories
        ]
        for instance_id, scheduler in enumerate(schedulers):
            scheduler.instance_id = instance_id
            scheduler.decode_handoff_claim_pending = True
        manager = AgenticKVManager(
            schedulers,
            AgenticKVConfig(
                policy="tiered",
                demotion_mode="capacity-only",
                pcie_bandwidth_gbps=1_000,
                cpu_bandwidth_gbps=1_000,
                cpu_transfer_latency_us=1,
            ),
        )
        for instance_id, scheduler in enumerate(schedulers):
            idle = IdleKVEntry(
                session_id=f"idle-{instance_id}",
                instance_id=instance_id,
                tokens=2,
                block_tokens=2,
                per_rank_bytes=2,
                total_bytes=2,
                location=KVLocation.HBM,
                tier_since_ns=0,
                last_access_ns=0,
            )
            manager.entries[idle.session_id] = idle
            ready_ns = manager.claim_active_hbm_reclaim(
                instance_id,
                1,
                100,
                owner_kind="pd",
                owner_id=99,
            )
            self.assertGreater(ready_ns, 100)
            request = self._decode_request(
                request_id=10 + instance_id, computed=6)
            scheduler.request.append(request)
            scheduler.memory.get_block_kv = (
                lambda batch_req, batch_len, scheduled_tokens=None: 0)

        batches = [
            scheduler.schedule_base(100, scheduler.start_npu)
            for scheduler in schedulers
        ]

        self.assertTrue(all(batch is not None for batch in batches))
        self.assertEqual(
            [[req.id for req in batch.requests] for batch in batches],
            [[10], [11]],
        )
        self.assertEqual(
            [manager.active_hbm_reclaim_claim(i).owner_kind for i in range(2)],
            ["pd", "pd"],
        )

    def test_pd_claim_limits_peer_batch_to_unreserved_hbm_slack(self):
        memory = _Memory(npu_mem=8, npu_used=8)
        scheduler = _scheduler(memory, mode="recompute")
        scheduler.decode_handoff_claim_pending = True
        manager, _ = _manager_with_idle_entry(
            scheduler, policy="tiered")
        ready_ns = manager.claim_active_hbm_reclaim(
            0,
            1,
            100,
            owner_kind="pd",
            owner_id=99,
        )
        manager.advance(ready_ns)
        self.assertEqual(manager.hbm_unreserved_per_rank_bytes(0), 1)

        fitting_peer = self._decode_request(request_id=20, computed=6)
        oversized_peer = self._decode_request(request_id=21, computed=6)

        def peer_block_kv(batch_req, batch_len, scheduled_tokens=None):
            per_request = {
                fitting_peer.id: 1,
                oversized_peer.id: 2,
            }
            return sum(
                per_request[req.id] for req in batch_req[:batch_len])

        memory.get_block_kv = peer_block_kv
        scheduler.request.append(oversized_peer)

        # Physical HBM has two bytes free, but one belongs to the P/D claim.
        # A standalone two-byte peer must not consume that reservation.
        self.assertIsNone(scheduler.schedule_base(ready_ns, 0))
        self.assertEqual(memory.npu_used, 6)
        self.assertEqual(oversized_peer.num_computed_tokens, 6)
        self.assertEqual(scheduler.active_recompute_preemptions, 0)

        scheduler.request.clear()
        scheduler.request.append(fitting_peer)
        batch = scheduler.schedule_base(ready_ns, 0)

        self.assertEqual(
            [req.id for req in batch.requests], [fitting_peer.id])
        self.assertEqual(manager.hbm_unreserved_per_rank_bytes(0), 0)

        scheduler.inflight.clear()
        scheduler.request.append(oversized_peer)
        self.assertIsNone(scheduler.schedule_base(ready_ns, 0))
        self.assertEqual(oversized_peer.num_computed_tokens, 6)
        self.assertEqual(scheduler.active_recompute_preemptions, 0)
        self.assertEqual(
            (manager.active_hbm_reclaim_claim(0).owner_kind,
             manager.active_hbm_reclaim_claim(0).owner_id),
            ("pd", 99),
        )

        claim = manager.consume_active_hbm_reclaim(
            0, ready_ns, owner_kind="pd", owner_id=99)
        self.assertEqual(claim.per_rank_bytes, 1)
        consume_events = [
            event for event in manager.events
            if event.get("event") == "active_hbm_reclaim_consume"
        ]
        self.assertEqual(len(consume_events), 1)

    def test_partial_pd_claim_rolls_back_for_same_time_decode_progress(self):
        prefill = _scheduler(
            _Memory(npu_mem=8, npu_used=8), mode="recompute")
        decode = _scheduler(
            _Memory(npu_mem=8, npu_used=7), mode="recompute")
        prefill.pd_type = "prefill"
        decode.pd_type = "decode"
        decode.instance_id = 1
        decode.start_npu = 1
        for scheduler in (prefill, decode):
            scheduler.block_size = 1
            scheduler.fp = 16
            scheduler.kv_cache_dtype = "auto"
            scheduler.decode_handoff_claim_pending = False

        manager = AgenticKVManager(
            [prefill, decode],
            AgenticKVConfig(
                policy="tiered",
                demotion_mode="capacity-only",
            ),
        )
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        pending = Request(99, "test-model", 1, 2, 100, 0)
        pending.ready_time = 100
        router._stage_pd_receive_admission(pending, prefill, 100)

        active = self._decode_request(request_id=7, computed=7)
        active.instance_id = decode.instance_id
        active.agentic_kv_owner_instance_id = decode.instance_id
        decode.request.append(active)

        # P cannot reserve its suffix, while D initially can. The D-side
        # reservation must be rolled back atomically or it consumes the final
        # byte needed by this already-admitted decode request.
        self.assertEqual(router.process_pending_decode_handoffs(100), 1)
        self.assertFalse(router.admit_pd_prefill_chunk(
            prefill, pending, 1, 100))
        handoff = router._pending_pd_chunk_admissions[(0, 1)][0]
        self.assertIsNone(handoff["prefill_claim_ready_ns"])
        self.assertIsNone(handoff["decode_claim_ready_ns"])
        self.assertFalse(prefill.decode_handoff_claim_pending)
        self.assertFalse(decode.decode_handoff_claim_pending)
        self.assertIsNone(manager.active_hbm_reclaim_claim(0))
        self.assertIsNone(manager.active_hbm_reclaim_claim(1))
        self.assertEqual(router._pd_admission_owner, {})
        cancel_events = [
            event for event in manager.events
            if event.get("event") == "active_hbm_reclaim_cancel"
        ]
        self.assertEqual(len(cancel_events), 1)
        self.assertEqual(
            (cancel_events[0]["owner_kind"], cancel_events[0]["owner_id"]),
            ("pd", pending.id),
        )

        # Identical capacity state is coalesced, not controller-polled.
        reclaim_admissions = manager.metrics.active_hbm_reclaim_admissions
        self.assertEqual(router.process_pending_decode_handoffs(100), 0)
        self.assertEqual(
            manager.metrics.active_hbm_reclaim_admissions,
            reclaim_admissions,
        )

        batch = decode.schedule_base(100, decode.start_npu)
        self.assertIsNotNone(batch)
        self.assertEqual([request.id for request in batch.requests], [active.id])

        # A real capacity change on P retries the pair at the same logical
        # timestamp. Release only the new D block to model this batch's
        # completion; its prior seven-token context remains resident.
        decode.inflight.clear()
        decode.memory.free(1, Device.NPU)
        prefill.memory.free(1, Device.NPU)
        self.assertEqual(router.process_pending_decode_handoffs(100), 0)
        self.assertEqual(prefill.request, [pending])
        self.assertEqual(pending.pd_chunk_admitted_tokens, 1)
        self.assertFalse(router.has_pending_decode_handoffs())
        self.assertEqual(router._pd_admission_owner, {})

    def test_partial_pd_claim_cancellation_keeps_started_demotion(self):
        prefill = _scheduler(
            _Memory(npu_mem=8, npu_used=8), mode="recompute")
        decode = _scheduler(
            _Memory(npu_mem=8, npu_used=8), mode="recompute")
        prefill.pd_type = "prefill"
        decode.pd_type = "decode"
        decode.instance_id = 1
        decode.start_npu = 1
        for scheduler in (prefill, decode):
            scheduler.block_size = 1
            scheduler.fp = 16
            scheduler.kv_cache_dtype = "auto"
            scheduler.decode_handoff_claim_pending = False

        manager = AgenticKVManager(
            [prefill, decode],
            AgenticKVConfig(
                policy="tiered",
                demotion_mode="capacity-only",
                pcie_bandwidth_gbps=1_000,
                cpu_bandwidth_gbps=1_000,
                cpu_transfer_latency_us=1,
            ),
        )
        idle = IdleKVEntry(
            session_id="decode-idle",
            instance_id=decode.instance_id,
            tokens=2,
            block_tokens=2,
            per_rank_bytes=2,
            total_bytes=2,
            location=KVLocation.HBM,
            tier_since_ns=0,
            last_access_ns=0,
        )
        manager.entries[idle.session_id] = idle
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        pending = Request(100, "test-model", 1, 2, 100, 0)
        pending.ready_time = 100
        router._stage_pd_receive_admission(pending, prefill, 100)

        self.assertEqual(router.process_pending_decode_handoffs(100), 1)
        self.assertFalse(router.admit_pd_prefill_chunk(
            prefill, pending, 1, 100))
        complete_ns = idle.migration_complete_ns
        self.assertIsNotNone(complete_ns)
        self.assertEqual(idle.migration_kind, "hbm_to_cpu")
        self.assertIsNone(manager.active_hbm_reclaim_claim(1))
        self.assertFalse(decode.decode_handoff_claim_pending)
        self.assertEqual(manager.metrics.transfer_jobs, 1)
        self.assertEqual(manager.next_internal_event_time(100), complete_ns)

        manager.advance(complete_ns)
        self.assertEqual(idle.location, KVLocation.CPU)
        self.assertEqual(idle.migration_kind, None)
        self.assertEqual(manager.metrics.transfer_jobs, 1)

    def test_net_neutral_active_to_idle_publication_retries_pd_pair(self):
        prefill = _scheduler(
            _Memory(npu_mem=8, npu_used=8), mode="recompute")
        decode = _scheduler(
            _Memory(npu_mem=8, npu_used=7), mode="recompute")
        prefill.pd_type = "prefill"
        decode.pd_type = "decode"
        decode.instance_id = 1
        decode.start_npu = 1
        for scheduler in (prefill, decode):
            scheduler.block_size = 1
            scheduler.fp = 16
            scheduler.kv_cache_dtype = "auto"
            scheduler.decode_handoff_claim_pending = False

        manager = AgenticKVManager(
            [prefill, decode],
            AgenticKVConfig(
                policy="tiered",
                demotion_mode="capacity-only",
                block_size=1,
                pcie_bandwidth_gbps=1_000,
                cpu_bandwidth_gbps=1_000,
                cpu_transfer_latency_us=1,
            ),
        )
        router = Router(
            2, [prefill, decode], 0, "RR",
            agentic_kv_manager=manager)
        pending = Request(101, "test-model", 1, 2, 100, 0)
        pending.ready_time = 100
        router._stage_pd_receive_admission(pending, prefill, 100)

        # P is physically full and has no idle victim. D's one-sided claim is
        # rolled back, and an unchanged state does not cause controller-poll
        # retries.
        self.assertEqual(router.process_pending_decode_handoffs(100), 1)
        self.assertFalse(router.admit_pd_prefill_chunk(
            prefill, pending, 1, 100))
        state_before_publication = manager.restore_capacity_state(0)
        admissions = manager.metrics.active_hbm_reclaim_admissions
        self.assertEqual(router.process_pending_decode_handoffs(100), 0)
        self.assertEqual(
            manager.metrics.active_hbm_reclaim_admissions, admissions)

        # Completion frees one active byte and immediately publishes that
        # same byte as idle KV. Physical usage and unreserved slack are net
        # neutral, but the byte is now an eligible LRU victim.
        completed = Request(7, "test-model", 1, 2, 0, 0)
        completed.session_id = "new-idle-victim"
        completed.num_computed_tokens = 1
        completed.agentic_kv_completion_released_per_rank_bytes = 1
        prefill.memory.free(1, Device.NPU)
        manager.on_idle_start(
            completed,
            completion_time_ns=100,
            release_time_ns=1_000,
            return_gap_type="tool",
            return_gap_source="test",
        )
        state_after_publication = manager.restore_capacity_state(0)
        self.assertEqual(
            state_before_publication[:-1],
            state_after_publication[:-1],
        )
        self.assertGreater(
            state_after_publication[-1], state_before_publication[-1])

        # The generation change bypasses the pair coalescing cache. Both
        # claims are admitted without waiting for an unrelated clock tick.
        self.assertEqual(router.process_pending_decode_handoffs(100), 0)
        self.assertIsNotNone(manager.active_hbm_reclaim_claim(0))
        self.assertIsNotNone(manager.active_hbm_reclaim_claim(1))
        self.assertEqual(
            manager.entries["new-idle-victim"].migration_kind,
            "hbm_to_cpu",
        )
        self.assertEqual(
            manager.metrics.active_hbm_reclaim_admissions,
            admissions + 2,
        )

    def test_capacity_pin_generation_changes_without_physical_delta(self):
        scheduler = _scheduler(
            _Memory(npu_mem=8, npu_used=2), mode="recompute")
        manager = AgenticKVManager(
            [scheduler],
            AgenticKVConfig(
                policy="tiered",
                demotion_mode="capacity-only",
                swap_execution_mode="sync-engine-barrier",
            ),
        )
        manager.entries["idle"] = IdleKVEntry(
            session_id="idle",
            instance_id=scheduler.instance_id,
            tokens=2,
            block_tokens=2,
            per_rank_bytes=2,
            total_bytes=2,
            location=KVLocation.HBM,
            tier_since_ns=0,
            last_access_ns=0,
        )

        unpinned = manager.restore_capacity_state(scheduler.instance_id)
        manager.acquire_synchronous_prepare_lock(
            77, [scheduler.instance_id], session_id="idle")
        pinned = manager.restore_capacity_state(scheduler.instance_id)
        self.assertEqual(unpinned[:-1], pinned[:-1])
        self.assertGreater(pinned[-1], unpinned[-1])

        # Reasserting the same logical pin is idempotent.
        manager.acquire_synchronous_prepare_lock(
            77, [scheduler.instance_id], session_id="idle")
        self.assertEqual(
            manager.restore_capacity_state(scheduler.instance_id), pinned)

        manager.release_synchronous_prepare_lock(77)
        released = manager.restore_capacity_state(scheduler.instance_id)
        self.assertEqual(pinned[:-1], released[:-1])
        self.assertGreater(released[-1], pinned[-1])

    def test_removed_scheduler_claim_owner_releases_reservation(self):
        memory = _Memory(npu_mem=8, npu_used=8)
        scheduler = _scheduler(memory, mode="recompute")
        manager, _ = _manager_with_idle_entry(
            scheduler, policy="tiered")
        owner = self._decode_request(request_id=30, computed=6)
        scheduler.request.append(owner)

        self.assertIsNone(scheduler.schedule_base(100, 0))
        self.assertIsNotNone(manager.active_hbm_reclaim_claim(0))
        scheduler.request.remove(owner)

        self.assertIsNone(scheduler.schedule_base(101, 0))
        self.assertIsNone(manager.active_hbm_reclaim_claim(0))
        cancel_events = [
            event for event in manager.events
            if event.get("event") == "active_hbm_reclaim_cancel"
        ]
        self.assertEqual(len(cancel_events), 1)
        self.assertEqual(cancel_events[0]["owner_kind"], "scheduler")
        self.assertEqual(cancel_events[0]["owner_id"], owner.id)

    def test_first_turn_prefill_reclaims_idle_hbm_instead_of_deadlocking(self):
        memory = _Memory(npu_mem=2, npu_used=2)
        scheduler = _scheduler(memory, mode="recompute")
        _, idle = _manager_with_idle_entry(scheduler)
        req = Request(1, "test-model", 1, 2, 0, 0)
        scheduler.request.append(req)

        batch = scheduler.schedule_base(100, 0)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.scheduled_tokens, {req.id: 1})
        self.assertEqual(idle.location, KVLocation.DROPPED)
        self.assertEqual(memory.npu_used, 1)
        self.assertEqual(scheduler.active_recompute_preemptions, 0)


if __name__ == "__main__":
    unittest.main()

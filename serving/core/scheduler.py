import pandas as pd
from time import time
import csv
import os

from .request import *
from .utils import *
from .controller import *
from .memory_model import *
from .graph_generator import *
from .trace_generator import *
from .online_latency_model import validate_runtime_context_contract
from .logger import print_markup, print_rule
from .pim_model import *
import numpy as np

# class that shedules request of astra-sim
class Scheduler:
    def __init__(self, model, node_id, instance_id, max_num_seqs, max_num_batched_tokens,
                 num_npus, tp_size, pp_size, npu_mem, cpu_mem,
                 start_npu, pd_type, fp, block_size, req_num,
                 prioritize_prefill, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, enable_chunked_prefill=False,
                 long_prefill_token_threshold=0, cxl_mem=0, ep_size=1,
                 kv_cache_dtype='auto', active_preemption_mode='cpu-swap',
                 max_model_len=None, npu_runtime_reserve_bytes=0):
        self.model = model
        self.config = get_config(model)
        if max_model_len is None:
            max_model_len = self.config['max_position_embeddings']
        validate_runtime_context_contract(
            config=self.config,
            max_model_len=max_model_len,
            model=model,
        )
        self.max_model_len = int(max_model_len)
        if self.max_model_len <= 0:
            raise ValueError(
                f"max_model_len must be positive, got {self.max_model_len}")
        self.node_id = node_id
        self.instance_id = instance_id
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = min(
            max_num_batched_tokens, self.max_model_len)
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.num_npus = num_npus
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.req_num = req_num
        self.start_npu = start_npu
        self.pd_type = pd_type
        self.block_size = block_size
        self.fp = fp
        self.kv_cache_dtype = kv_cache_dtype
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_prefix_sharing = enable_prefix_sharing
        self.enable_chunked_prefill = enable_chunked_prefill
        self.prefix_storage = prefix_storage
        self.prioritize_prefill = prioritize_prefill
        if active_preemption_mode not in {'cpu-swap', 'recompute'}:
            raise ValueError(
                "active_preemption_mode must be 'cpu-swap' or 'recompute', "
                f"got {active_preemption_mode!r}")
        self.active_preemption_mode = active_preemption_mode
        self.active_recompute_preemptions = 0
        self.active_recompute_tokens = 0
        self.active_cpu_swap_preemptions = 0
        self.active_cpu_swap_write_bytes = 0
        self.active_cpu_swap_read_bytes = 0
        # Installed by AgenticKVManager when idle-session ownership is active.
        # ``memory_wait_until_ns`` is retained as a compatibility surface but
        # active-HBM reclaim is request-local: only the claim owner is deferred.
        self.agentic_kv_manager = None
        self.memory_wait_until_ns = None
        self.model_fabric_wait_until_ns = None
        # Router-side P/D handoff admission shares the manager's single
        # per-instance active-HBM claim with ordinary scheduler admission.
        # While this flag is set, schedule() may use unreserved slack but
        # must not issue or consume a second claim for the same instance.
        self.decode_handoff_claim_pending = False
        # Router installs this only for strict agentic P/D. It admits the
        # block-rounded P and D ownership for one proposed prefill chunk.
        self.pd_chunk_admission_callback = None
        # Requests admitted earlier in the current local batch-formation pass
        # are still present in ``self.request`` until the Batch is built. A
        # later candidate must never reclaim their frozen P/D ownership.
        self.pd_chunk_admission_pass_protected_request_ids = set()
        # A committed P graph turns its formerly frozen partial-prefill KV
        # into a safe progress-preemption victim without changing HBM usage.
        # Router retry coalescing must observe that dependency transition.
        self.pd_prefill_reclaimability_generation = 0
        # Lists are sorted by scheduler eligibility time. For ordinary
        # requests ready_time == arrival; an agentic restore can make it later.
        self.request = []
        self.inflight = []
        self.done = []
        self.batch_ids = -1

        # memory model
        self.memory = MemoryModel(
            model, instance_id, node_id, num_npus, tp_size, npu_mem,
            cpu_mem, block_size, fp, enable_prefix_caching,
            enable_prefix_sharing, prefix_pool, prefix_storage, cxl_mem,
            ep_size=ep_size, pp_size=pp_size,
            kv_cache_dtype=kv_cache_dtype,
            npu_runtime_reserve_bytes=npu_runtime_reserve_bytes)

        # logger
        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)
    
 
    def schedule(self, current, sys, batch_id=-1):
        if self.enable_prefix_caching:
            return self.schedule_with_prefix(current, sys, batch_id)
        else:
            return self.schedule_base(current, sys, batch_id)

    def _configure_pd_kv_handoff(self, req):
        """Initialize exact prompt-KV ownership for an agentic P request.

        Lower-tier restores materialize the reusable prefix only on P. The
        decode receive buffer is preallocated, but it contains no data until a
        P graph sends the prefix. Conversely, a cross-instance HBM hit retains
        an authoritative D-side prefix and must not send that prefix back.
        """
        if (getattr(self, "pd_type", None) != "prefill"
                or getattr(self, "agentic_kv_manager", None) is None):
            return

        req.pd_kv_handoff_tracking_enabled = True
        hit_tokens = int(req.agentic_kv_hit_tokens)
        if hit_tokens < 0:
            raise RuntimeError(
                f"Request #{req.id} has negative reusable KV tokens: "
                f"{hit_tokens}")
        max_operational_hit = max(0, int(req.original_input) - 1)
        if hit_tokens > max_operational_hit:
            raise RuntimeError(
                "Agentic P/D reuse must leave the final prompt token for P "
                f"execution: request={req.id}, hit={hit_tokens}, "
                f"input={req.original_input}")

        retained_instance = req.agentic_kv_retained_instance_id
        retained_bytes = int(req.agentic_kv_retained_per_rank_bytes)
        if retained_instance is None:
            if retained_bytes != 0:
                raise RuntimeError(
                    "Retained P/D KV bytes require a decode instance: "
                    f"request={req.id}, bytes={retained_bytes}")
            if (hit_tokens > 0
                    and req.agentic_kv_owner_instance_id
                    != self.instance_id):
                raise RuntimeError(
                    "A non-retained restored prefix must be owned by its P "
                    f"instance before launch: request={req.id}, "
                    f"owner={req.agentic_kv_owner_instance_id}, "
                    f"prefill={self.instance_id}")
            req.pd_restored_prefix_handoff_pending_tokens = hit_tokens
        else:
            if hit_tokens <= 0 or retained_bytes <= 0:
                raise RuntimeError(
                    "A retained decode prefix requires positive logical and "
                    f"physical ownership: request={req.id}, hit={hit_tokens}, "
                    f"bytes={retained_bytes}")
            req.pd_restored_prefix_handoff_pending_tokens = 0

    def _stage_pd_kv_handoff(self, batch, scheduled_tokens):
        """Attach one graph's P->D KV ownership without consuming it.

        A lower-tier DMA may overlap early suffix work in async-decode-join
        mode. In that case the restored prefix is staged on the first batch
        formed at or after restore completion; the final prompt-token gate
        guarantees such a batch exists. Newly computed KV is staged on every
        P batch. Request counters remain unchanged until add_done().
        """
        if getattr(self, "pd_type", None) != "prefill":
            return

        restored_by_request = {}
        new_by_request = {}
        for req in batch.requests:
            if not req.pd_kv_handoff_tracking_enabled:
                continue
            q_tokens = int(scheduled_tokens.get(req.id, 0))
            if q_tokens <= 0:
                raise RuntimeError(
                    "Tracked P/D request was batched without positive model "
                    f"work: request={req.id}, scheduled={q_tokens}")
            new_by_request[req.id] = q_tokens

            pending = int(
                req.pd_restored_prefix_handoff_pending_tokens)
            if pending < 0:
                raise RuntimeError(
                    f"Request #{req.id} has negative pending P/D KV tokens")
            if pending == 0:
                continue
            if not req.is_prefill():
                raise RuntimeError(
                    "A restored P/D prefix reached a decode-style P batch: "
                    f"request={req.id}, pending={pending}")
            if int(batch.batch_time) < int(
                    req.agentic_kv_restore_ready_time_ns):
                # The source DMA has not produced the prefix yet. Keep its
                # one-shot ownership on Request for the post-restore batch.
                continue
            restored_by_request[req.id] = pending

        batch.pd_restored_prefix_handoff_by_request = (
            restored_by_request)
        batch.pd_restored_prefix_handoff_tokens = sum(
            restored_by_request.values())
        batch.pd_new_kv_handoff_by_request = new_by_request
        batch.pd_new_kv_handoff_tokens = sum(new_by_request.values())

    def _admit_pd_prefill_chunks(
            self, current, batch_requests, scheduled_tokens):
        """Filter only P/D owners whose exact chunk claim is not ready."""
        callback = getattr(self, "pd_chunk_admission_callback", None)
        if callback is None or self.pd_type != "prefill":
            return batch_requests
        if self.pd_chunk_admission_pass_protected_request_ids:
            raise RuntimeError(
                "Nested P/D chunk-admission pass retained protected owners: "
                f"instance={self.instance_id}, protected="
                f"{sorted(self.pd_chunk_admission_pass_protected_request_ids)}")
        admitted = []
        try:
            for req in batch_requests:
                if (not req.is_prefill()
                        or not req.pd_kv_handoff_tracking_enabled):
                    admitted.append(req)
                    continue
                chunk_tokens = int(scheduled_tokens.get(req.id, 0))
                if chunk_tokens <= 0:
                    raise RuntimeError(
                        "P/D prefill candidate lacks a positive proposed "
                        f"chunk: request={req.id}, "
                        f"scheduled={chunk_tokens}")
                # An earlier callback in this same pass may have selected a
                # later candidate as the active-prefill victim, which resets
                # its chunk_len together with num_computed_tokens. Reassert
                # the immutable proposal immediately before admission.
                req.chunk_len = chunk_tokens
                if callback(self, req, chunk_tokens, int(current)):
                    if int(req.pd_chunk_admitted_tokens) != chunk_tokens:
                        raise RuntimeError(
                            "P/D chunk callback returned ready without "
                            "freezing the proposal: request="
                            f"{req.id}, proposed={chunk_tokens}, admitted="
                            f"{req.pd_chunk_admitted_tokens}")
                    expected_target = (
                        int(req.num_computed_tokens) + chunk_tokens)
                    if int(req.pd_chunk_admission_target_tokens) != (
                            expected_target):
                        raise RuntimeError(
                            "P/D chunk callback froze the wrong target: "
                            f"request={req.id}, expected={expected_target}, "
                            "observed="
                            f"{req.pd_chunk_admission_target_tokens}")
                    req.chunk_len = chunk_tokens
                    admitted.append(req)
                    self.pd_chunk_admission_pass_protected_request_ids.add(
                        int(req.id))
                else:
                    req.chunk_len = 0
                    scheduled_tokens.pop(req.id, None)
        finally:
            self.pd_chunk_admission_pass_protected_request_ids.clear()
        return admitted

    def _commit_pd_kv_handoff(self, batch):
        """Commit staged P->D KV only after all ASTRA ranks succeed."""
        # A few scheduler unit fixtures predate Batch and intentionally use a
        # minimal SimpleNamespace. They cannot carry P/D ownership; preserve a
        # strict no-op for those non-P/D doubles while real Batch instances
        # always expose the fields initialized in request.py.
        if not hasattr(batch, "pd_kv_handoff_committed"):
            return
        if batch.pd_kv_handoff_committed:
            raise RuntimeError(
                f"P/D KV handoff for batch #{batch.batch_id} committed twice")
        reclaimability_changed = False
        requests = {req.id: req for req in batch.requests}
        for request_id, tokens in (
                batch.pd_restored_prefix_handoff_by_request.items()):
            req = requests.get(request_id)
            if req is None:
                raise RuntimeError(
                    "P/D restored-prefix batch ownership lost request "
                    f"#{request_id}")
            tokens = int(tokens)
            if (tokens <= 0
                    or req.pd_restored_prefix_handoff_pending_tokens
                    != tokens):
                raise RuntimeError(
                    "P/D restored-prefix commit does not match pending "
                    f"ownership: request={request_id}, staged={tokens}, "
                    "pending="
                    f"{req.pd_restored_prefix_handoff_pending_tokens}")
            req.pd_restored_prefix_handoff_pending_tokens = 0
            req.pd_restored_prefix_handoff_sent_tokens += tokens

        for request_id, tokens in batch.pd_new_kv_handoff_by_request.items():
            req = requests.get(request_id)
            if req is None:
                raise RuntimeError(
                    "P/D new-KV batch ownership lost request "
                    f"#{request_id}")
            tokens = int(tokens)
            if tokens <= 0:
                raise RuntimeError(
                    "P/D new-KV commit requires positive tokens: "
                    f"request={request_id}, staged={tokens}")
            if (req.pd_kv_handoff_tracking_enabled
                    and req.pd_kv_ownership_state == "prefill_active"):
                admitted_tokens = int(req.pd_chunk_admitted_tokens)
                expected_target = int(req.num_computed_tokens) + tokens
                if (admitted_tokens != tokens
                        or int(req.pd_chunk_admission_target_tokens)
                        != expected_target):
                    raise RuntimeError(
                        "Committed P graph does not match its admitted P/D "
                        f"chunk: request={request_id}, staged={tokens}, "
                        f"admitted={admitted_tokens}, target="
                        f"{req.pd_chunk_admission_target_tokens}, "
                        f"expected_target={expected_target}")
                if not req.pd_chunk_admission_history:
                    raise RuntimeError(
                        f"P/D request #{request_id} has no chunk admission "
                        "history at graph commit")
                history = req.pd_chunk_admission_history[-1]
                if history.get("committed", False):
                    raise RuntimeError(
                        f"P/D request #{request_id} committed one chunk twice")
                history["committed"] = True
                history["commit_batch_id"] = int(batch.batch_id)
                req.pd_chunk_admitted_tokens = 0
                req.pd_chunk_admission_target_tokens = 0
                reclaimability_changed = True
            req.pd_new_kv_handoff_sent_tokens += tokens
        batch.pd_kv_handoff_committed = True
        if reclaimability_changed:
            # One batch completion is one scheduler-visible dependency
            # transition even when several partial prefills become victims.
            self.pd_prefill_reclaimability_generation += 1

    def _validate_pd_prompt_kv_handoff(self, req):
        """Prove that D owns every logical prompt token exactly once."""
        if not req.pd_kv_handoff_tracking_enabled:
            return
        hit_tokens = int(req.agentic_kv_hit_tokens)
        active_prefill_recomputed = (
            int(req.pd_active_prefill_recompute_generation) > 0)
        expected_restored = (
            0 if (active_prefill_recomputed
                  or req.agentic_kv_retained_instance_id is not None)
            else hit_tokens)
        expected_new = (
            int(req.original_input)
            if active_prefill_recomputed
            else int(req.original_input) - hit_tokens)
        actual_restored = int(
            req.pd_restored_prefix_handoff_sent_tokens)
        actual_new = int(req.pd_new_kv_handoff_sent_tokens)
        pending = int(req.pd_restored_prefix_handoff_pending_tokens)
        if (pending != 0
                or actual_restored != expected_restored
                or actual_new != expected_new):
            raise RuntimeError(
                "P/D prompt-KV handoff does not reconcile at prefill "
                f"completion: request={req.id}, input={req.original_input}, "
                f"hit={hit_tokens}, retained="
                f"{req.agentic_kv_retained_instance_id is not None}, "
                f"active_prefill_recomputed={active_prefill_recomputed}, "
                f"restored_sent={actual_restored}/{expected_restored}, "
                f"new_sent={actual_new}/{expected_new}, pending={pending}")
        if req.pd_kv_ownership_state == "prefill_active":
            if req.pd_chunk_claim_pending or req.pd_chunk_admitted_tokens:
                raise RuntimeError(
                    "P/D prefill completed with an uncommitted chunk: "
                    f"request={req.id}, claim_pending="
                    f"{req.pd_chunk_claim_pending}, admitted_tokens="
                    f"{req.pd_chunk_admitted_tokens}")
            prefill_owned = int(req.pd_prefill_owned_per_rank_bytes)
            decode_owned = int(req.pd_decode_owned_per_rank_bytes)
            expected = int(req.pd_prefill_full_per_rank_bytes)
            if (prefill_owned != expected
                    or decode_owned != int(
                        req.pd_decode_full_per_rank_bytes)
                    or prefill_owned != decode_owned):
                raise RuntimeError(
                    "P/D prompt block ownership does not reconcile at "
                    f"handoff: request={req.id}, prefill="
                    f"{prefill_owned}/{expected}, decode={decode_owned}/"
                    f"{req.pd_decode_full_per_rank_bytes}")

    def _synchronous_swap_blocked(self, current, ready_requests):
        if self.agentic_kv_manager is None:
            return False
        prepare_locked = getattr(
            self.agentic_kv_manager,
            "prepare_locked",
            self.agentic_kv_manager.synchronous_prepare_locked,
        )
        if prepare_locked(self.instance_id):
            return True
        blocked_until = (
            self.agentic_kv_manager.record_synchronous_swap_dispatch_block(
                self.instance_id, current, ready_requests)
        )
        if blocked_until is not None:
            for request in ready_requests:
                if request.first_schedule_time_ns is None:
                    request.scheduler_resource_ready_time_ns = max(
                        int(request.scheduler_resource_ready_time_ns),
                        int(blocked_until),
                    )
        return blocked_until is not None and current < blocked_until

    def _model_resource_blocked(self, current, ready_requests=()):
        """Gate only future ASTRA dispatches behind cold fabric traffic."""
        if self.agentic_kv_manager is None:
            self.model_fabric_wait_until_ns = None
            return False
        dispatch_gate = getattr(
            self.agentic_kv_manager, "model_dispatch_blocked_until", None)
        if dispatch_gate is None:
            self.model_fabric_wait_until_ns = None
            return False
        blocked_until = dispatch_gate(self.instance_id, current)
        self.model_fabric_wait_until_ns = blocked_until
        if blocked_until is not None:
            for request in ready_requests:
                if request.first_schedule_time_ns is None:
                    request.scheduler_resource_ready_time_ns = max(
                        int(request.scheduler_resource_ready_time_ns),
                        int(blocked_until),
                    )
        return blocked_until is not None and int(current) < blocked_until

    def _get_reload_size(self, batch_req, batch_len):
        load_size = 0
        for req in batch_req[:batch_len]:
            if req.evict:
                load_size += self.memory.get_evict_kv(req)
        return load_size

    @staticmethod
    def _prefill_schedule_target(req, current):
        """Return the furthest prompt token this iteration may execute.

        Async cold restore may expose a reusable-prefix destination before
        its bytes are complete. All fresh tokens except the final prompt token
        form the idealized overlap region; the final token remains behind the
        request-local restore join because it can produce the first output.
        """
        target = int(req.prefill_target_tokens)
        cutoff = req.agentic_kv_overlap_cutoff_tokens
        if (cutoff is not None
                and int(current) < req.agentic_kv_restore_ready_time_ns):
            target = min(target, int(cutoff))
        return target

    def _is_active_hbm_available(self, size):
        """Check physical HBM after manager-side logical reservations."""
        if self.agentic_kv_manager is None:
            return self.memory.is_avail(size, Device.NPU)
        return (
            self.agentic_kv_manager.hbm_unreserved_per_rank_bytes(
                self.instance_id)
            >= size
        )

    def _consume_ready_scheduler_hbm_claim(self, current):
        """Return the request owning a ready scheduler-side HBM claim."""
        if self.agentic_kv_manager is None:
            return None, None
        claim = self.agentic_kv_manager.active_hbm_reclaim_claim(
            self.instance_id)
        if (claim is None or claim.owner_kind != "scheduler"
                or int(current) < claim.ready_ns):
            return None, None
        owners = [
            req for req in self.request if req.id == claim.owner_id
        ]
        if len(owners) != 1:
            raise RuntimeError(
                "Scheduler HBM reclaim claim lost its request owner: "
                f"instance={self.instance_id}, owner={claim.owner_id}")
        owner = owners[0]
        consumed = self.agentic_kv_manager.consume_active_hbm_reclaim(
            self.instance_id,
            current,
            owner_kind="scheduler",
            owner_id=owner.id,
        )
        if consumed is None:
            raise RuntimeError(
                "Ready scheduler HBM reclaim claim was not consumable: "
                f"instance={self.instance_id}, owner={owner.id}")
        original_ready = owner.agentic_hbm_reclaim_original_ready_time_ns
        owner.scheduler_resource_ready_time_ns = max(
            int(owner.scheduler_resource_ready_time_ns),
            int(claim.ready_ns),
        )
        if original_ready is not None:
            owner.ready_time = int(original_ready)
        owner.agentic_hbm_reclaim_ready_time_ns = None
        owner.agentic_hbm_reclaim_original_ready_time_ns = None
        return owner, consumed

    def _cancel_orphaned_scheduler_hbm_claim(self, current):
        """Release a scheduler reservation whose request left the queue."""
        if self.agentic_kv_manager is None:
            return
        claim = self.agentic_kv_manager.active_hbm_reclaim_claim(
            self.instance_id)
        if (claim is None or claim.owner_kind != "scheduler"
                or any(req.id == claim.owner_id for req in self.request)):
            return
        self.agentic_kv_manager.cancel_active_hbm_reclaim(
            self.instance_id, current)

    def _defer_scheduler_hbm_claim_owner(self, req, ready_ns):
        """Make a future HBM claim delay only its owning request."""
        if req.agentic_hbm_reclaim_original_ready_time_ns is None:
            req.agentic_hbm_reclaim_original_ready_time_ns = int(
                req.ready_time)
        req.agentic_hbm_reclaim_ready_time_ns = int(ready_ns)
        req.ready_time = max(int(req.ready_time), int(ready_ns))
        self.request.sort(key=lambda item: (item.ready_time, item.id))

    def _preempt_decode_request(self, req, current):
        """Release one active decode request according to the configured mode.

        Returns the per-rank KV bytes released from HBM.  CPU swap preserves
        the legacy ``req.evict`` reload path. Recompute instead resets the
        physical context and lets subsequent prefill batches rebuild it.
        """
        evicted_kv_size = self.memory.get_evict_kv(req)
        self.memory.free(evicted_kv_size, Device.NPU)
        use_cpu_swap = self.active_preemption_mode == 'cpu-swap'
        cluster_bytes = evicted_kv_size * self.num_npus
        if use_cpu_swap and self.agentic_kv_manager is not None:
            admissible = getattr(
                self.agentic_kv_manager,
                'active_cpu_swap_admissible',
                None,
            )
            if (admissible is not None
                    and not admissible(
                        self.instance_id, cluster_bytes, int(current))):
                use_cpu_swap = False
                record_fallback = getattr(
                    self.agentic_kv_manager,
                    'record_active_cpu_swap_capacity_fallback',
                    None,
                )
                if record_fallback is not None:
                    record_fallback(
                        self.instance_id, req.id,
                        cluster_bytes, int(current))
        if use_cpu_swap:
            req.evict = True
            self.memory.allocate(cluster_bytes, Device.CPU)
            self.active_cpu_swap_preemptions += 1
            self.active_cpu_swap_write_bytes += cluster_bytes
            self.logger.info("Eviction of the request #%d to CPU", req.id)
        else:
            recompute_tokens = int(req.num_computed_tokens)
            req.begin_active_recompute()
            self.active_recompute_preemptions += 1
            self.active_recompute_tokens += recompute_tokens
            self.logger.info(
                "Preemption of request #%d with %d-token recomputation",
                req.id, recompute_tokens)
        return evicted_kv_size

    # batch the request scheduling method
    def schedule_base(self, current, sys, batch_id=-1):
        # first NPU to process new batch
        if sys == self.start_npu:
            self._cancel_orphaned_scheduler_hbm_claim(current)
            # nothing to batch return None
            if len(self.request) != 0 and self.request[0].ready_time > current:
                return None
            # constraint of inflight batches considering parallelism
            if len(self.inflight) >= self.pp_size:
                # wait it to be done
                return None

            claim_owner, consumed_hbm_claim = (
                self._consume_ready_scheduler_hbm_claim(current))
            # scheduling start
            batch_req = (
                [claim_owner]
                if claim_owner is not None else
                [
                    req for req in self.request
                    if req.ready_time <= current
                    and not (
                        req.pd_chunk_claim_pending
                        and not req.pd_chunk_admitted_tokens)
                ]
            )

            # max_num_seqs limits total running requests (vLLM behavior)
            running_reqs = sum(len(b.requests) for b in self.inflight)
            available_slots = max(0, int(self.max_num_seqs) - running_reqs)
            batch_len = min(len(batch_req), available_slots)

            # nothing to batch
            if batch_len == 0:
                return None
            if self._model_resource_blocked(
                    current, batch_req[:batch_len]):
                return None
            if self._synchronous_swap_blocked(
                    current, batch_req[:batch_len]):
                return None

            # can make batch and proceed
            batch_req = batch_req[:batch_len]

            kv_size = 0
            evict_size = 0

            # Get decode requests for preemption decisions
            gen_req = [req for req in batch_req if not req.is_prefill()]
            
            if self.prioritize_prefill and not self.enable_chunked_prefill:
                prefill_req = [req for req in batch_req if req.is_prefill()]

                if len(prefill_req) != 0:
                    batch_req = prefill_req
                    batch_len = min(len(batch_req), available_slots)
                    batch_req = batch_req[:batch_len]
            
            # Chunked prefill: process decode requests first, then prefill requests
            if self.enable_chunked_prefill:
                prefills = [req for req in batch_req if req.is_prefill()]
                decodes = [req for req in batch_req if not req.is_prefill()]
                batch_req = decodes + prefills
                batch_len = len(batch_req)
            
            # ============ STEP 1: Token budget allocation (FIRST) ============
            # Build scheduled_tokens dict: req.id -> tokens to process this step
            scheduled_tokens = {}
            
            if self.enable_chunked_prefill:
                # vLLM-style chunked prefill: schedule running (decode + ongoing prefill)
                # first, then waiting (new prefill) requests. Token budget is the main
                # constraint; long_prefill_token_threshold caps per-request tokens per step.
                token_budget = self.max_num_batched_tokens
                new_batch_req = []
                threshold = self.long_prefill_token_threshold
                # Decode requests first (each decode request = 1 token)
                for req in batch_req:
                    if not req.is_prefill():
                        if token_budget <= 0:
                            break
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = 1
                        token_budget -= 1
                # Then prefill requests (chunked)
                for req in batch_req:
                    if req.is_prefill():
                        if token_budget <= 0:
                            break
                        frozen_chunk = int(req.pd_chunk_admitted_tokens)
                        remaining = (
                            self._prefill_schedule_target(req, current)
                            - req.num_computed_tokens)
                        # Per-request cap: long_prefill_token_threshold
                        if frozen_chunk:
                            if frozen_chunk > remaining:
                                raise RuntimeError(
                                    "Frozen P/D chunk exceeds remaining "
                                    f"prefill: request={req.id}, frozen="
                                    f"{frozen_chunk}, remaining={remaining}")
                            if frozen_chunk > token_budget:
                                continue
                            remaining = frozen_chunk
                        elif 0 < threshold < remaining:
                            remaining = threshold
                        chunk = min(remaining, token_budget)
                        if chunk <= 0:
                            break
                        req.chunk_len = chunk
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = chunk
                        token_budget -= chunk
                batch_req = new_batch_req
                batch_len = len(batch_req)

            else:
                # Non-chunked: compute scheduled tokens for each request
                total_len = 0
                for req in batch_req:
                    if req.is_prefill():
                        tokens_to_compute = (
                            self._prefill_schedule_target(req, current)
                            - req.num_computed_tokens)
                        scheduled_tokens[req.id] = tokens_to_compute
                        req.chunk_len = tokens_to_compute
                        total_len += tokens_to_compute
                    else:
                        scheduled_tokens[req.id] = 1
                        total_len += 1

                while total_len > self.max_num_batched_tokens:
                    # print(f"[NON_CHUNKED] total_len({total_len} = sum([req 0 ~ {batch_len - 1}])) exceed 'max_num_batched_tokens'")
                    last_req = batch_req[-1]
                    total_len -= scheduled_tokens[last_req.id]
                    del scheduled_tokens[last_req.id]
                    batch_req = batch_req[:-1]
                    batch_len -= 1
                
                # DEBUG: Check if total_len reached max
                # if total_len >= self.max_num_batched_tokens * 0.9:
                #     print(f"[NON-CHUNKED] Near max tokens! total_len: {total_len}/{self.max_num_batched_tokens}")
                #     print(f"              Batch: {batch_len} reqs, scheduled_tokens: {scheduled_tokens}")
            
                # Early return due to max_num_batched_tokens limitation (It occurs only when No chunked-prefill)
                if batch_len == 0:
                    print("     [WARNNING] Cannot load the request to batch due to max_num_batched_tokens limitation")
                    return None
            proposed_batch_len = len(batch_req)
            batch_req = self._admit_pd_prefill_chunks(
                current, batch_req, scheduled_tokens)
            batch_len = len(batch_req)
            if batch_len < proposed_batch_len:
                # Re-form at the same timestamp. Newly gated owners are now
                # skipped before token budgeting, so their unused budget can
                # admit smaller peers instead of causing head-of-line block.
                return self.schedule_base(current, sys, batch_id)
            if batch_len == 0:
                return None
            # ============ STEP 2: KV size calculation (with scheduled_tokens) ============
            temp_len = batch_len
            for i in range(batch_len, -1, -1):
                kv_size = self.memory.get_block_kv(batch_req, i, scheduled_tokens)
                load_size = self._get_reload_size(batch_req, i)
                if self._is_active_hbm_available(kv_size + load_size):
                    temp_len = i
                    break

            # Idle session state is lower priority than runnable active work.
            # If even the head request cannot enter HBM, reclaim the oldest
            # idle whole-session object before preempting an active decode.
            if (temp_len == 0 and self.agentic_kv_manager is not None
                    and not getattr(
                        self, "decode_handoff_claim_pending", False)):
                if consumed_hbm_claim is not None:
                    raise RuntimeError(
                        "Consumed scheduler HBM claim did not admit its owner: "
                        f"instance={self.instance_id}, "
                        f"owner={consumed_hbm_claim.owner_id}")
                minimum_kv_size = self.memory.get_block_kv(
                    batch_req, 1, scheduled_tokens)
                minimum_load_size = self._get_reload_size(batch_req, 1)
                if (self.inflight
                        and self.agentic_kv_manager
                        .synchronous_hbm_reclaim_needs_boundary(
                            self.instance_id,
                            minimum_kv_size + minimum_load_size,
                            current,
                        )):
                    return None
                ready_ns = self.agentic_kv_manager.claim_active_hbm_reclaim(
                    self.instance_id,
                    minimum_kv_size + minimum_load_size,
                    current,
                    owner_kind="scheduler",
                    owner_id=batch_req[0].id,
                )
                if ready_ns is not None:
                    if ready_ns > current:
                        self._defer_scheduler_hbm_claim_owner(
                            batch_req[0], ready_ns)
                        # Re-run admission at the same timestamp. The owner is
                        # now future-ready, so fit-capable peers can dispatch
                        # without waiting for its background demotion.
                        return self.schedule_base(current, sys, batch_id)
                    claim = self.agentic_kv_manager.consume_active_hbm_reclaim(
                        self.instance_id,
                        current,
                        owner_kind="scheduler",
                        owner_id=batch_req[0].id,
                    )
                    if claim is None:
                        raise RuntimeError(
                            "Immediate agentic HBM reclaim lost its claim")
                    consumed_hbm_claim = claim
                    for i in range(batch_len, -1, -1):
                        kv_size = self.memory.get_block_kv(
                            batch_req, i, scheduled_tokens)
                        load_size = self._get_reload_size(batch_req, i)
                        if self._is_active_hbm_available(
                                kv_size + load_size):
                            temp_len = i
                            break

            # A request-local or P/D admission claim owns its logical HBM
            # reservation. Smaller runnable batches may use unreserved slack,
            # but a non-fitting peer must wait instead of preempting another
            # active context to compete with the outstanding claim.
            if temp_len == 0 and self.agentic_kv_manager is not None:
                outstanding_claim = (
                    self.agentic_kv_manager.active_hbm_reclaim_claim(
                        self.instance_id))
                if outstanding_claim is not None:
                    return None
            if (temp_len == 0
                    and getattr(self, "decode_handoff_claim_pending", False)):
                return None
            
            # ============ STEP 3: Eviction if needed ============
            while temp_len == 0:
                # print("Evict Request to CPU due to memory limitation")
                # preempt request one by one until there is enough space
                if len(gen_req) == 0:
                    return None
                
                # check already evicted request
                if gen_req[-1].evict:
                    gen_req = gen_req[:-1]
                    continue

                req_to_evict = gen_req[-1]
                evicted_kv_size = self._preempt_decode_request(
                    req_to_evict, current)
                if req_to_evict.evict:
                    evict_size += evicted_kv_size
                gen_req = gen_req[:-1]

                if len(gen_req) < batch_len:
                    batch_len = len(gen_req)

                # check if can batch
                for i in range(batch_len, -1, -1):
                    kv_size = self.memory.get_block_kv(batch_req, i, scheduled_tokens)
                    load_size = self._get_reload_size(batch_req, i)
                    if self._is_active_hbm_available(kv_size + load_size):
                        temp_len = i
                        break

            batch_len = temp_len
            batch_req = batch_req[:batch_len]

            # Recompute kv_size for final batch
            kv_size = self.memory.get_block_kv(batch_req, batch_len, scheduled_tokens)
            load_size = self._get_reload_size(batch_req, batch_len)

            # delete from request queue
            for req in batch_req:
                for i, req_ in enumerate(self.request):
                    if req_.id == req.id:
                        del self.request[i]
                        break

                if req.evict:
                    req.evict = False
                    self.logger.info("Loading the request #%d", req.id)

            # ============ STEP 4: Allocate memory ============
            if kv_size > 0:
                self.memory.allocate(kv_size, Device.NPU)

            # Reload evicted KV to NPU and remove the spilled copy from CPU.
            # load_size is per-rank, cpu_used is full-cluster.
            if load_size > 0:
                self.memory.allocate(load_size, Device.NPU)
                self.memory.free(load_size * self.num_npus, Device.CPU)
                self.active_cpu_swap_read_bytes += load_size * self.num_npus
            
            # ============ STEP 5: Build batch with lists ============
            total_len = 0
            kv_len = 0
            num_prefill = 0
            num_decode = 0
            q_list = []
            k_list = []
            prefill_q_list = []
            prefill_k_list = []
            decode_k_list = []
            for req in batch_req:
                req.set_que_delay(current)
                if req.is_prefill():
                    # Use scheduled_tokens for chunk size
                    chunk_size = scheduled_tokens.get(
                        req.id,
                        req.prefill_target_tokens - req.num_computed_tokens)

                    total_len += chunk_size
                    q_list.append(chunk_size)
                    prefill_q_list.append(chunk_size)
                    # prefill_k_list: already computed tokens (k_cache from previous chunks)
                    prefill_k_list.append(req.num_computed_tokens)
                    # k_list: total kv cache after this step (computed + new)
                    # k_list.append(req.num_computed_tokens + chunk_size)
                    num_prefill += 1

                else:
                    # Decode
                    total_len += 1
                    q_list.append(1)
                    num_decode += 1
                    kv_len += req.num_computed_tokens
                    decode_k_list.append(req.num_computed_tokens)
                    # k_list.append(req.num_computed_tokens)

            # make batch, output doesn't matter here!! always one iteration
            # batch is also 1
            batch = Batch(self.get_batch_id(), self.model, total_len, kv_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, current, kv_size, evict_size, load_size)
            # add already fired system
            batch.fired.append(sys)
            batch.requests.extend(batch_req)
            batch.scheduled_tokens = scheduled_tokens
            self._stage_pd_kv_handoff(batch, scheduled_tokens)
            if self.agentic_kv_manager is not None:
                self.agentic_kv_manager.record_agentic_batch_schedule(
                    self, batch)
            self.inflight.append(batch)
            self.logger.info(
                "Scheduling new batch #%d to NPU[%d]",
                batch.batch_id,
                sys,
            )
            # print(f"[BATCH DEBUG] Batch: {len(new_batch_req)} reqs, scheduled_tokens: {scheduled_tokens}")
            # batch.log()
            return batch
        
        # Schedule already batched request
        else:
            if len(self.inflight) == 0:
                return None
            else:
                batch = None
                # find batch
                for b in self.inflight:
                    if b.batch_id == batch_id:
                        batch = b
                if batch == None:
                    return None
                # check if this has been runned in the system
                if sys in batch.fired:
                    return None
                else:
                    batch.fired.append(sys)
                    self.logger.info(
                        "Scheduling existing batch #%d to NPU[%d]",
                        batch.batch_id,
                        sys,
                    )
                    return batch
    
    def schedule_with_prefix(self, current, sys, batch_id=-1):
        if sys == self.start_npu:
            # nothing to batch return None
            if len(self.request) != 0 and self.request[0].ready_time > current:
                return None
            # constraint of inflight batches considering parallelism
            if len(self.inflight) >= self.pp_size:
                # wait it to be done
                return None

            # scheduling start
            batch_req = [
                req for req in self.request
                if req.ready_time <= current
                and not (
                    req.pd_chunk_claim_pending
                    and not req.pd_chunk_admitted_tokens)
            ]

            # max_num_seqs limits total running requests (vLLM behavior)
            running_reqs = sum(len(b.requests) for b in self.inflight)
            available_slots = max(0, int(self.max_num_seqs) - running_reqs)
            batch_len = min(len(batch_req), available_slots)

            # nothing to batch
            if batch_len == 0:
                return None
            if self._model_resource_blocked(
                    current, batch_req[:batch_len]):
                return None
            if self._synchronous_swap_blocked(
                    current, batch_req[:batch_len]):
                return None

            # can make batch and proceed
            batch_req = batch_req[:batch_len]

            # Prioritize prefill (without chunked prefill) or reorder for chunked prefill
            if self.prioritize_prefill and not self.enable_chunked_prefill:
                prefill_req = [req for req in batch_req if req.is_prefill()]
                if len(prefill_req) != 0:
                    batch_req = prefill_req
                    batch_len = min(len(batch_req), available_slots)
                    batch_req = batch_req[:batch_len]
            
            # Chunked prefill: process decode requests first, then prefill requests
            if self.enable_chunked_prefill:
                prefills = [req for req in batch_req if req.is_prefill()]
                decodes = [req for req in batch_req if not req.is_prefill()]
                batch_req = decodes + prefills
                batch_len = len(batch_req)

            # Get decode requests for preemption decisions
            gen_req = [req for req in batch_req if not req.is_prefill()]
            # gen_req = [req for req in batch_req if not (req.num_computed_tokens >= req.original_input)]
            
            # ============ STEP 0: Prefix Matching ============
            # Only match prefix for NEW prefill requests (first chunk)
            # Ongoing chunked prefills already have their prefix cache info
            # for req in batch_req:
            #     if req.is_prefill():
            #         self.memory.prefix_match(req)
            
            # ============ STEP 1: Token budget allocation ============
            scheduled_tokens = {}
            
            if self.enable_chunked_prefill:
                # Chunked prefill: assign token budget to requests
                token_budget = self.max_num_batched_tokens
                new_batch_req = []
                
                # Decode requests first (each decode request = 1 token)
                for req in batch_req:
                    if not req.is_prefill():
                        if token_budget <= 0:
                            break
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = 1
                        token_budget -= 1
                
                # Then prefill requests (chunked)
                threshold = self.long_prefill_token_threshold
                for req in batch_req:
                    if req.is_prefill():
                        if token_budget <= 0:
                            break
                        # Calculate remaining tokens without considering prefix cache
                        # because it is already considered in "self.memory.prefix_match(req)" -> req.num_computed_tokens
                        if req.num_computed_tokens == 0:
                            self.memory.prefix_match(req)
                        remaining = (
                            self._prefill_schedule_target(req, current)
                            - req.num_computed_tokens)
                        # Per-request cap: long_prefill_token_threshold
                        frozen_chunk = int(req.pd_chunk_admitted_tokens)
                        if frozen_chunk:
                            if frozen_chunk > remaining:
                                raise RuntimeError(
                                    "Frozen P/D chunk exceeds remaining "
                                    f"prefill: request={req.id}, frozen="
                                    f"{frozen_chunk}, remaining={remaining}")
                            if frozen_chunk > token_budget:
                                continue
                            remaining = frozen_chunk
                        elif 0 < threshold < remaining:
                            remaining = threshold
                        chunk = min(remaining, token_budget)
                        if chunk <= 0:
                            break

                        req.chunk_len = chunk
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = chunk
                        token_budget -= chunk

                batch_req = new_batch_req
                batch_len = len(batch_req)
            else:
                # Non-chunked: compute scheduled tokens for each request
                total_len = 0
                for req in batch_req:
                    if req.is_prefill():
                        if req.num_computed_tokens == 0:
                            self.memory.prefix_match(req)
                        tokens_to_compute = max(
                            self._prefill_schedule_target(req, current)
                            - req.num_computed_tokens,
                            1,
                        )
                        scheduled_tokens[req.id] = tokens_to_compute
                        req.chunk_len = tokens_to_compute  # Set chunk_len for add_done()
                        total_len += tokens_to_compute
                    else:
                        scheduled_tokens[req.id] = 1
                        total_len += 1

                while total_len > self.max_num_batched_tokens:
                    last_req = batch_req[-1]
                    total_len -= scheduled_tokens[last_req.id]
                    del scheduled_tokens[last_req.id]
                    batch_req = batch_req[:-1]
                    batch_len -= 1

            proposed_batch_len = len(batch_req)
            batch_req = self._admit_pd_prefill_chunks(
                current, batch_req, scheduled_tokens)
            batch_len = len(batch_req)
            if batch_len < proposed_batch_len:
                return self.schedule_with_prefix(current, sys, batch_id)
            if batch_len == 0:
                return None
            
            # ============ STEP 1.5: Lock prefix for scheduled requests ============
            newly_locked = set()
            for req in batch_req:
                # if req.is_prefill() and req.num_computed_tokens == 0:
                if req.is_prefill() and req.npu_last_node is not None and not req._prefix_locked:
                    self.memory.lock_prefix(req, Device.NPU)
                    req._prefix_locked = True
                    newly_locked.add(req.id)
            
            # ============ STEP 2: KV size calculation ============
            kv_size = 0
            evict_size = 0
            temp_len = batch_len
            total_useable_size = self.memory.avail_size(Device.NPU) + self.memory.evictable_size(Device.NPU)
            
            for i in range(batch_len, -1, -1):
                kv_size = self.memory.get_block_kv(batch_req, i, scheduled_tokens)
                if total_useable_size >= kv_size:
                    temp_len = i
                    break
            
            # ============ STEP 3: Eviction if needed ============
            evicted_req = []
            while temp_len == 0:
                # print("eviction occurs!!")
                if len(gen_req) == 0:
                    # print("gen_req length == 0 (No decode) => return None (No Batch)")
                    # No request to evict but no memory - rollback prefix cache lock
                    for req in batch_req:
                        if req.is_prefill() and req._prefix_locked:
                            
                            self.memory.unlock_prefix(req, Device.NPU)
                            self.memory.erase_prefix_info(req)
                            req._prefix_locked = False
                    return None
                
                # Check already evicted request
                if gen_req[-1].evict:
                    gen_req = gen_req[:-1]
                    continue
                
                # Evict the last decode request
                # (DEPRECATED) self.memory.unlock_prefix(gen_req[-1], Device.NPU)
                # (DEPRECATED) self.memory.erase_prefix_info(gen_req[-1])
                if gen_req[-1].is_prefill() and getattr(gen_req[-1], '_prefix_locked', False):
                    self.memory.unlock_prefix(gen_req[-1], Device.NPU)
                    # self.memory.erase_prefix_info(gen_req[-1])
                    gen_req[-1]._prefix_locked = False
                
                current_usable_size = self.memory.avail_size(Device.NPU) + self.memory.evictable_size(Device.NPU)
                
                gen_req[-1].evict = True
                evicted_req.append(gen_req[-1])
                self.logger.info("Eviction of the request #%d", gen_req[-1].id)
                gen_req = gen_req[:-1]
                
                if len(gen_req) < batch_len:
                    batch_len = len(gen_req)
                
                # Check if can batch now
                for i in range(batch_len, -1, -1):
                    kv_size = self.memory.get_block_kv(batch_req, i, scheduled_tokens)
                    if current_usable_size >= kv_size:
                        temp_len = i
                        break

            # Unlock prefix for requests that didn't make it into the batch
            for req in batch_req[temp_len:]:
                if req.is_prefill() and req._prefix_locked:
                    self.memory.unlock_prefix(req, Device.NPU)
                    self.memory.erase_prefix_info(req)
                    req._prefix_locked = False

            batch_len = temp_len
            batch_req = batch_req[:batch_len]
            
            # Recompute kv_size for final batch
            kv_size = self.memory.get_block_kv(batch_req, batch_len, scheduled_tokens)
            evict_size = (kv_size - self.memory.avail_size(Device.NPU)) if kv_size > self.memory.avail_size(Device.NPU) else 0
            
            if evict_size > 0:
                self.memory.evict_prefix_cache(evict_size, Device.NPU)

            # ============ STEP 4: Allocate memory & handle evicted requests ============
            evict_load_size = 0
            prefix_load_size = 0
            
            for req in batch_req:
                # Remove from request queue
                for i, req_ in enumerate(self.request):
                    if req_.id == req.id:
                        del self.request[i]
                        break

                # Load prefix cache from storage if needed
                if req.is_prefill() and req.storage_cache_hit > req.npu_cache_hit:
                    prefix_load_size += (req.storage_cache_hit - req.npu_cache_hit) * self.memory.get_kv(1)

                # Handle evicted requests
                if req.evict:
                    self.memory.prefix_match(req)
                    self.memory.lock_prefix(req, Device.NPU)
                    if self.prefix_storage is not None:
                        self.memory.unlock_prefix(req, Device.CPU)
                    evict_load_size += self.memory.get_evict_kv(req)
                    req.evict = False
                    self.logger.info("Loading the request #%d", req.id)

            # ============ STEP 5: Build batch with lists ============
            total_len = 0
            kv_len = 0
            num_prefill = 0
            num_decode = 0
            q_list = []
            k_list = []
            prefill_q_list = []
            prefill_k_list = []
            decode_k_list = []
            
            # Evict storage prefix cache if needed
            total_size = 0
            for req in batch_req:
                total_size += self.memory.get_total_kv(req) * self.num_npus
            for req in evicted_req:
                total_size += self.memory.get_total_kv(req) * self.num_npus
            
            if self.prefix_storage is not None:
                storage_evict_size = (total_size - self.memory.avail_size(self.prefix_storage)) if total_size > self.memory.avail_size(self.prefix_storage) else 0
                if storage_evict_size > 0:
                    self.memory.evict_prefix_cache(storage_evict_size, self.prefix_storage)

            for req in batch_req:
                req.set_que_delay(current)
                # Update the prefix cache for incoming batch
                # NOTE: Moved to add_done() to ensure prefix cache is updated after chunk computation
                # self.memory.cache_unfinished_req(req, Device.NPU)
                # if self.prefix_storage is not None:
                #     self.memory.cache_unfinished_req(req, self.prefix_storage)
                
                if req.is_prefill():
                    # Use scheduled_tokens for chunk size. num_computed_tokens
                    # already includes any prefix-cache hit (memory_model.py
                    # bumps it on first prefix_match), so chunk_size is already
                    # the count of tokens actually computed this iteration —
                    # no further prefix-hit subtraction is needed downstream.
                    chunk_size = scheduled_tokens.get(
                        req.id,
                        req.prefill_target_tokens - req.num_computed_tokens)
                    if chunk_size > self.max_num_batched_tokens:
                        raise Exception("Chunk length exceeds max num batched tokens")

                    total_len += chunk_size
                    q_list.append(chunk_size)
                    num_prefill += 1
                    prefill_q_list.append(chunk_size)
                    # prefill_k_list: already computed tokens (k_cache from previous chunks)
                    prefill_k_list.append(req.num_computed_tokens)
                else:
                    # Decode: use num_computed_tokens (inevitable modification)
                    total_len += 1
                    q_list.append(1)
                    num_decode += 1
                    kv_len += req.num_computed_tokens  # inevitable modification: was req.input
                    decode_k_list.append(req.num_computed_tokens)  # inevitable modification: was req.input
                
                k_list.append(req.num_computed_tokens)  # inevitable modification: was req.input
            
            # Storage needs to hold evicted cache
            if self.prefix_storage is not None:
                for req in evicted_req:
                    self.memory.storage_cache_evicted_req(req)

            
            # For debugging
            # self.memory.npu_prefix_cache.pretty_print()
            # self.memory.npu_prefix_cache.print_prefix_info()
            batch = Batch(self.get_batch_id(), self.model, total_len, kv_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, current, kv_size, evict_size, evict_load_size + prefix_load_size)
            batch.fired.append(sys)
            batch.requests.extend(batch_req)
            batch.scheduled_tokens = scheduled_tokens
            self._stage_pd_kv_handoff(batch, scheduled_tokens)
            if self.agentic_kv_manager is not None:
                self.agentic_kv_manager.record_agentic_batch_schedule(
                    self, batch)
            self.inflight.append(batch)
            self.logger.info(
                "Scheduling new batch #%d to NPU[%d]",
                batch.batch_id,
                sys,
            )
            # print(f"[BATCH DEBUG] Batch: {len(new_batch_req)} reqs, scheduled_tokens: {scheduled_tokens}")
            # batch.log()
            return batch
        # Schedule already batched request
        else:
            if len(self.inflight) == 0:
                return None
            else:
                batch = None
                # find batch
                for b in self.inflight:
                    if b.batch_id == batch_id:
                        batch = b
                if batch is None or sys in batch.fired:
                    return None
                else:
                    batch.fired.append(sys)
                    self.logger.info(
                        "Scheduling existing batch #%d to NPU[%d]",
                        batch.batch_id,
                        sys,
                    )
                    return batch
        
    def _finish_request(self, req, finish):
        """Release active KV and finalize one request exactly once."""
        self.logger.info("Request #%d is done", req.id)
        if self.enable_prefix_caching:
            self.memory.cache_finished_req(req, Device.NPU)
            if self.prefix_storage is not None:
                self.memory.cache_finished_req(req, Device.CPU)
        else:
            kv_size = self.memory.get_evict_kv(req)
            self.memory.free(kv_size, Device.NPU)
            req.agentic_kv_completion_released_per_rank_bytes = int(kv_size)
        req.agentic_kv_owner_instance_id = None
        req.add_latency(finish)
        self.done.append(req)
        return req

    def censor_queued_request(self, req, cutoff_time_ns):
        """Remove one non-running request and release its active KV.

        Measurement early-stop drains every dispatched batch but deliberately
        does not schedule queued work. Prefix-cache ownership and CPU-swapped
        active requests require separate cancellation protocols, so this
        exact path is limited to the prefix-off, recompute-preemption mode used
        by online cold-session experiments. Strict P-side requests with an
        unconsumed D receive allocation are handled by Router/AgenticKVManager
        before this generic active-request path.
        """
        del cutoff_time_ns  # Kept in the API for timestamped audit callers.
        if self.inflight:
            raise RuntimeError(
                "Cannot censor a queued request while its scheduler still "
                f"has inflight batches: instance={self.instance_id}")
        matches = [candidate for candidate in self.request if candidate is req]
        if len(matches) != 1:
            raise RuntimeError(
                "Queued request censoring requires one identity match: "
                f"instance={self.instance_id}, request={req.id}, "
                f"matches={len(matches)}")
        if self.enable_prefix_caching:
            raise RuntimeError(
                "Measurement censoring of queued prefix-cache requests is "
                "not implemented")
        if req.evict:
            raise RuntimeError(
                "Measurement censoring of queued CPU-swapped active "
                "requests is not implemented")
        if int(req.pd_prefill_preallocated_per_rank_bytes) != 0:
            raise RuntimeError(
                "Strict P/D preallocation must be censored by the router "
                f"ownership path: request={req.id}")
        if req.agentic_kv_owner_instance_id != self.instance_id:
            raise RuntimeError(
                "Queued active request lost its scheduler HBM owner: "
                f"request={req.id}, expected={self.instance_id}, "
                f"observed={req.agentic_kv_owner_instance_id}")

        # Recompute preemption already freed the complete active context and
        # reset num_computed_tokens to zero. Every other queued decode owns its
        # current block-rounded context on this scheduler.
        if req.recompute_target_tokens is not None:
            released_per_rank_bytes = 0
        else:
            released_per_rank_bytes = int(self.memory.get_evict_kv(req))
            self.memory.free(released_per_rank_bytes, Device.NPU)
        self.request.remove(req)
        req.agentic_kv_owner_instance_id = None
        return {
            "request_id": int(req.id),
            "session_id": (
                None if req.session_id is None else str(req.session_id)),
            "instance_id": int(self.instance_id),
            "released_per_rank_bytes": released_per_rank_bytes,
            "active_recompute_already_released": (
                req.recompute_target_tokens is not None),
        }

    def censor_full_model_hbf_queued_request(
            self, req, cutoff_time_ns):
        """Cancel one queued full-model-HBF GPU request exactly.

        This is the finite-memory counterpart of the online HBF adapter's
        logical cutoff.  It is deliberately narrower than generic
        measurement censoring:

        * no model graph may remain inflight on this Scheduler;
        * generic prefix caching and the legacy agentic-KV manager are
          forbidden by the full-model-HBF composition;
        * an active CPU-swap copy, a recompute-preempted request, an
          unlaunched zero-byte request, and an ordinary HBM-resident request
          are unwound separately.

        A queued request may have completed earlier chunks even when
        ``agentic_kv_owner_instance_id`` is ``None``.  Therefore ownership is
        reconciled from the Scheduler's physical preemption state instead of
        treating that metadata field as an allocation oracle.
        """
        del cutoff_time_ns  # Retained for a timestamped Router audit.
        if self.inflight:
            raise RuntimeError(
                "Cannot censor a full-model HBF queue while its Scheduler "
                f"has inflight batches: instance={self.instance_id}")
        matches = [candidate for candidate in self.request if candidate is req]
        if len(matches) != 1:
            raise RuntimeError(
                "Full-model HBF queue censoring requires one identity "
                f"match: instance={self.instance_id}, request={req.id}, "
                f"matches={len(matches)}")
        if self.enable_prefix_caching:
            raise RuntimeError(
                "Full-model HBF queue censoring cannot unwind generic "
                "prefix-cache ownership")
        if self.agentic_kv_manager is not None:
            raise RuntimeError(
                "Full-model HBF queue censoring cannot share ownership "
                "with the legacy agentic-KV manager")
        if int(req.pd_prefill_preallocated_per_rank_bytes) != 0:
            raise RuntimeError(
                "Full-model HBF queue censoring found an unexpected strict "
                f"P/D preallocation: request={req.id}")
        if (
            req.agentic_kv_retained_instance_id is not None
            or int(req.agentic_kv_retained_per_rank_bytes) != 0
        ):
            raise RuntimeError(
                "Full-model HBF queue censoring found unadopted retained "
                f"ownership: request={req.id}")
        owner = req.agentic_kv_owner_instance_id
        if owner not in (None, self.instance_id):
            raise RuntimeError(
                "Full-model HBF queued request changed GPU owner: "
                f"request={req.id}, expected={self.instance_id}, "
                f"observed={owner}")

        resident_per_rank_bytes = 0
        swapped_cluster_bytes = 0
        if req.evict:
            swapped_per_rank_bytes = int(self.memory.get_evict_kv(req))
            swapped_cluster_bytes = (
                swapped_per_rank_bytes * int(self.num_npus))
            if swapped_cluster_bytes:
                self.memory.free(swapped_cluster_bytes, Device.CPU)
            req.evict = False
        elif req.recompute_target_tokens is None:
            resident_per_rank_bytes = int(self.memory.get_evict_kv(req))
            if resident_per_rank_bytes:
                self.memory.free(resident_per_rank_bytes, Device.NPU)

        self.request.remove(req)
        req.agentic_kv_owner_instance_id = None
        return {
            "request_id": int(req.id),
            "session_id": (
                None if req.session_id is None else str(req.session_id)),
            "instance_id": int(self.instance_id),
            "released_npu_per_rank_bytes": resident_per_rank_bytes,
            "released_cpu_cluster_bytes": swapped_cluster_bytes,
            "active_recompute_already_released": (
                req.recompute_target_tokens is not None),
        }

    # pop inflight, add to done
    def add_done(self, id, sys, finish):
        prompt_t = 0
        gen_t = 0
        end_reqs = []
        if len(self.inflight) == 0:
            return prompt_t, gen_t, end_reqs
        batch = None
        # find batch
        id -= 1
        idx = 0
        for i, b in enumerate(self.inflight):
            if b.batch_id == id:
                batch = b
                idx = i
        # no batch return
        if batch == None:
            return prompt_t, gen_t, end_reqs
        # already done
        if sys in batch.end:
            return prompt_t, gen_t, end_reqs
        else:
            # add to done system
            batch.end.append(sys)
            # check all npus are done
            if self.pd_type != "prefill":
                if self.start_npu not in batch.end or (self.start_npu + self.num_npus - 1) not in batch.end:
                    return prompt_t, gen_t, end_reqs
            else:
                if self.start_npu not in batch.end or (self.start_npu + self.num_npus * 2 - 1) not in batch.end:
                    return prompt_t, gen_t, end_reqs
        self.logger.info(
            "Batch #%d is done",
            batch.batch_id,
        )
        if self.agentic_kv_manager is not None:
            self.agentic_kv_manager.record_agentic_batch_complete(
                self, batch, finish)
        self._commit_pd_kv_handoff(batch)
                
        pool = []
        for req in batch.requests:
            # Capture this before advancing the request. The rebuild target is
            # cleared exactly when its final prefill chunk completes.
            is_active_recompute = req.recompute_target_tokens is not None
            # For chunked prefill, use computed tokens to determine prefill vs
            # decode. Active recomputation temporarily extends this target to
            # the complete context that existed at preemption.
            is_prefill_req = req.is_prefill()
            
            # change phase
            if is_prefill_req:
                # Get chunk_len from scheduling step
                chunk_len = (
                    req.chunk_len if req.chunk_len > 0
                    else req.prefill_target_tokens - req.num_computed_tokens)
                if chunk_len > self.max_num_batched_tokens:
                    raise Exception("Chunk length exceeds max num batched tokens")
                chunk_start = int(req.num_computed_tokens)
                active_prefill_frontier = int(
                    req.active_prefill_recompute_frontier_tokens)
                active_prefill_replay_tokens = max(
                    0,
                    min(chunk_start + int(chunk_len),
                        active_prefill_frontier)
                    - chunk_start,
                )

                # Update num_computed_tokens
                req.num_computed_tokens += chunk_len
                req.chunk_len = 0  # Reset for next step
                
                # Check if prefill is complete
                if req.num_computed_tokens >= req.prefill_target_tokens:
                    # Update prefix cache before clearing is_init (for stats tracking)
                    if self.enable_prefix_caching:
                        self.memory.cache_unfinished_req(req, Device.NPU)
                        if self.prefix_storage is not None:
                            self.memory.cache_unfinished_req(req, self.prefix_storage)
                    if is_active_recompute:
                        # The final token in a rebuilt context produces the next
                        # output through lm_head just like an ordinary completed
                        # prefill. It is model work, but not a second user prompt,
                        # so do not add the rebuilt tokens to prompt throughput or
                        # overwrite an existing TTFT.
                        req.finish_active_recompute()
                        if req.is_init:
                            req.is_init = False
                            prompt_t += req.original_input
                        req.record_output_token(finish)
                        gen_t += 1
                    else:
                        req.is_init = False
                        # Count the logical prompt once. A P/D active-prefill
                        # preemption replays [0, frontier), but that duplicate
                        # work must not inflate user prompt throughput.
                        prompt_t += (
                            chunk_len - active_prefill_replay_tokens
                            + req.prefix_cache_hit)
                        # A P-only graph still contains final_layernorm,
                        # lm_head, and sampler. It therefore emits output token
                        # 1 and establishes TTFT before handing prompt KV to D.
                        req.record_output_token(finish)
                        gen_t += 1

                    if self.pd_type == "prefill" and not is_active_recompute:
                        # Prefill instance: send to decode instance
                        self.logger.info("Request #%d is prefill done", req.id)
                        self.logger.info("Request #%d is sent to decode instance", req.id)
                        self._validate_pd_prompt_kv_handoff(req)
                        # req.num_computed_tokens += 1  # First decode token was generated
                        
                        # remove kv cache here
                        if self.enable_prefix_caching:
                            self.memory.unlock_prefix(req, Device.NPU)
                        else:
                            kv_size = self.memory.get_evict_kv(req)
                            if req.pd_kv_ownership_state == "prefill_active":
                                if kv_size != int(
                                        req.pd_prefill_owned_per_rank_bytes):
                                    raise RuntimeError(
                                        "P/D handoff P release changed size: "
                                        f"request={req.id}, scheduler="
                                        f"{kv_size}, owned="
                                        f"{req.pd_prefill_owned_per_rank_bytes}")
                                req.pd_prefill_handoff_released_per_rank_bytes = (
                                    int(kv_size))
                            self.memory.free(kv_size, Device.NPU)
                            if req.pd_kv_ownership_state == "prefill_active":
                                req.pd_prefill_owned_per_rank_bytes = 0
                                req.pd_kv_ownership_state = "handoff_pending"
                        req.agentic_kv_owner_instance_id = None

                        end_reqs.append(req)
                        continue
                else:
                    # Prefill not complete, return to pool for next chunk
                    if not is_active_recompute:
                        prompt_t += (
                            chunk_len - active_prefill_replay_tokens)
                    cutoff = req.agentic_kv_overlap_cutoff_tokens
                    if (cutoff is not None
                            and req.num_computed_tokens >= int(cutoff)
                            and finish
                            < req.agentic_kv_restore_ready_time_ns):
                        req.ready_time = req.agentic_kv_restore_ready_time_ns
                        req.agentic_kv_restore_gate_start_ns = int(finish)
                        req.agentic_kv_restore_gate_wait_ns = (
                            req.agentic_kv_restore_ready_time_ns
                            - int(finish)
                        )
                        if self.agentic_kv_manager is not None:
                            self.agentic_kv_manager.record_async_restore_gate(
                                req, int(finish))
                    # pool.append(req)
                    # continue
            else:
                # Decode phase
                if req.is_init:
                    # Full prefix cache hit: all input tokens were cached, so the
                    # request never entered the prefill-complete path where is_init
                    # is cleared. Lock the prefix node (was skipped because
                    # is_prefill() returned False during scheduling), count prefix
                    # stats once, then clear is_init.
                    if self.enable_prefix_caching:
                        if req.npu_last_node is not None and not req._prefix_locked:
                            self.memory.lock_prefix(req, Device.NPU)
                            req._prefix_locked = True
                        self.memory.cache_unfinished_req(req, Device.NPU)
                        if self.prefix_storage is not None:
                            self.memory.cache_unfinished_req(req, self.prefix_storage)
                    req.is_init = False
                    # Full prefix hit: count all cached tokens as prompt throughput
                    prompt_t += req.prefix_cache_hit
                req.record_output_token(finish)
                gen_t += 1
                req.num_computed_tokens += 1

            # Update computed tokens for decode
            # req.num_computed_tokens += 1

            # check done
            if req.generated_tokens >= req.requested_output_tokens:
                end_reqs.append(self._finish_request(req, finish))

            # return to pool
            else:
                # print("Request #{} is not finished => go to pool".format(req.id))
                # Update prefix cache after chunk completion (moved from schedule_with_prefix())
                if self.enable_prefix_caching:
                    self.memory.cache_unfinished_req(req, Device.NPU)
                    if self.prefix_storage is not None:
                        self.memory.cache_unfinished_req(req, self.prefix_storage)
                pool.append(req)
        # Return to the request pool; both lists are ordered by eligibility.
        self.request = self._merge_by_arrival_id(pool, self.request)
        del self.inflight[idx]
        del batch

        return prompt_t, gen_t, end_reqs
    

    ##### Helper Functions ######
    # get new batch id
    def get_batch_id(self):
        self.batch_ids += 1
        return self.batch_ids

    # add a request
    def add_request(self, req, is_init=True, metadata=None, enqueue=True):
        new_req = Request(*(req), is_init=is_init)
        max_model_len = getattr(self, 'max_model_len', None)
        if (max_model_len is not None
                and int(new_req.output) > int(max_model_len)):
            requested_output = int(new_req.output) - int(new_req.input)
            raise ValueError(
                f"Request #{new_req.id} total sequence length "
                f"{new_req.output} (prompt={new_req.input}, "
                f"output={requested_output}) exceeds model context limit "
                f"{max_model_len}. Select a compatible long-context model "
                "or apply an explicit compaction policy; KV block paging "
                "does not extend the semantic context window."
            )
        if metadata is not None:
            new_req.session_id = metadata.get('session_id')
            new_req.sub_request_index = metadata.get('sub_request_index')
            new_req.source_session_id = metadata.get('source_session_id')
            new_req.session_template_index = metadata.get(
                'session_template_index')
            new_req.session_epoch = int(
                metadata.get('session_epoch') or 0)
            session_offered_time_ns = metadata.get(
                'session_offered_time_ns')
            new_req.session_offered_time_ns = int(
                new_req.arrival
                if session_offered_time_ns is None
                else session_offered_time_ns)
            session_admission_time_ns = metadata.get(
                'session_admission_time_ns')
            new_req.session_admission_time_ns = int(
                new_req.arrival
                if session_admission_time_ns is None
                else session_admission_time_ns)
            new_req.session_admission_queue_wait_ns = int(
                metadata.get('session_admission_queue_wait_ns') or 0)
            ready_time_ns = metadata.get('ready_time_ns')
            new_req.ready_time = (
                new_req.arrival if ready_time_ns is None else int(ready_time_ns))
            new_req.prefix_reuse_tokens = int(metadata.get('prefix_reuse_toks') or 0)
            new_req.prefix_reuse_source = metadata.get('prefix_reuse_source')
            new_req.return_gap_type = str(
                metadata.get('return_gap_type') or 'session_start')
            new_req.return_gap_source = str(
                metadata.get('return_gap_source') or 'unknown')
            new_req.return_gap_ns = int(metadata.get('return_gap_ns') or 0)
            new_req.agentic_kv_hit_tokens = int(
                metadata.get('agentic_kv_hit_tokens') or 0)
            new_req.agentic_kv_recompute_tokens = int(
                metadata.get('agentic_kv_recompute_tokens') or 0)
            new_req.agentic_kv_residency_at_return = metadata.get(
                'agentic_kv_residency_at_return')
            new_req.agentic_kv_source = metadata.get('agentic_kv_source')
            new_req.hbf_online_execution = metadata.get(
                'hbf_online_execution')
            new_req.hbf_online_route_reason = metadata.get(
                'hbf_online_route_reason')
            new_req.agentic_kv_restore_ns = int(
                metadata.get('agentic_kv_restore_ns') or 0)
            new_req.agentic_kv_owner_gate_ns = int(
                metadata.get('agentic_kv_owner_gate_ns') or 0)
            new_req.agentic_kv_restore_issue_time_ns = int(
                metadata.get('agentic_kv_restore_issue_time_ns')
                or new_req.arrival)
            new_req.agentic_kv_target_hbm_ready_time_ns = int(
                metadata.get('agentic_kv_target_hbm_ready_time_ns')
                or new_req.arrival)
            new_req.agentic_kv_restore_ready_time_ns = int(
                metadata.get('agentic_kv_restore_ready_time_ns')
                or new_req.ready_time)
            new_req.agentic_kv_fresh_prompt_tokens = int(
                metadata.get('agentic_kv_fresh_prompt_tokens') or 0)
            cutoff = metadata.get('agentic_kv_overlap_cutoff_tokens')
            new_req.agentic_kv_overlap_cutoff_tokens = (
                None if cutoff is None else int(cutoff))
            new_req.agentic_kv_async_decode_join = bool(
                metadata.get('agentic_kv_async_decode_join', False))
            new_req.agentic_kv_restore_gate_start_ns = int(
                metadata.get('agentic_kv_restore_gate_start_ns') or 0)
            new_req.agentic_kv_restore_gate_wait_ns = int(
                metadata.get('agentic_kv_restore_gate_wait_ns') or 0)
            new_req.pd_pair_fifo_wait_ns = int(
                metadata.get('pd_pair_fifo_wait_ns') or 0)
            new_req.agentic_kv_prepare_boundary_wait_ns = int(
                metadata.get(
                    'agentic_kv_prepare_boundary_wait_ns') or 0)
            new_req.agentic_kv_source_demotion_join_wait_ns = int(
                metadata.get(
                    'agentic_kv_source_demotion_join_wait_ns') or 0)
            new_req.agentic_kv_hbm_admission_wait_ns = int(
                metadata.get('agentic_kv_hbm_admission_wait_ns') or 0)
            new_req.agentic_kv_transient_dram_capacity_wait_ns = int(
                metadata.get(
                    'agentic_kv_transient_dram_capacity_wait_ns') or 0)
            new_req.agentic_kv_restore_queue_wait_ns = int(
                metadata.get('agentic_kv_restore_queue_wait_ns') or 0)
            new_req.agentic_kv_restore_service_ns = int(
                metadata.get('agentic_kv_restore_service_ns') or 0)
            # The tier manager owns these already-allocated HBM blocks.  The
            # scheduler starts at the reusable prefix and assumes ownership;
            # normal completion/free paths then release the full request KV.
            new_req.num_computed_tokens = new_req.agentic_kv_hit_tokens
            new_req.prefix_cache_hit = new_req.agentic_kv_hit_tokens
            new_req.agentic_kv_owner_instance_id = metadata.get(
                'agentic_kv_owner_instance_id')
            new_req.agentic_kv_retained_instance_id = metadata.get(
                'agentic_kv_retained_instance_id')
            new_req.agentic_kv_retained_per_rank_bytes = int(
                metadata.get('agentic_kv_retained_per_rank_bytes') or 0)
        self._configure_pd_kv_handoff(new_req)
        if enqueue:
            self.enqueue_request(new_req)
        return new_req

    def enqueue_request(self, new_req):
        """Make a fully admitted request visible to the batch scheduler."""
        if self.agentic_kv_manager is not None:
            resource_ready = getattr(
                self.agentic_kv_manager,
                "model_dispatch_resource_ready_time",
                None,
            )
            if resource_ready is not None:
                new_req.scheduler_resource_ready_time_ns = max(
                    int(new_req.scheduler_resource_ready_time_ns),
                    int(resource_ready(
                        self.instance_id, int(new_req.ready_time))),
                )
        # Maintain eligibility-time sort order (required by
        # schedule_base/schedule_with_prefix). Avoid bisect's ``key``
        # argument so the simulator remains compatible with Python 3.7-3.9.
        sort_key = (new_req.ready_time, new_req.id)
        lo = 0
        hi = len(self.request)
        while lo < hi:
            mid = (lo + hi) // 2
            if (self.request[mid].ready_time, self.request[mid].id) <= sort_key:
                lo = mid + 1
            else:
                hi = mid
        self.request.insert(lo, new_req)
        return new_req
    
    def decode_handoff_hbm_bytes(self, req):
        """Return per-rank HBM bytes newly needed by a P/D handoff.

        A resumed agentic prefix can remain allocated on the sticky decode
        instance while prefill computes only the suffix. In that case the
        handoff admits only the missing suffix, but validates the retained
        ownership before any manager claim is made.
        """
        if req.agentic_kv_owner_instance_id is not None:
            raise RuntimeError(
                "P/D handoff attempted while agentic KV is still owned by "
                f"instance {req.agentic_kv_owner_instance_id}; the prefill "
                "scheduler must release it before decode allocation")
        if self.enable_prefix_caching:
            raise RuntimeError(
                "Decode handoff byte admission is incompatible with generic "
                "prefix caching")

        kv_size = self.memory.get_total_kv(req)
        retained_instance = req.agentic_kv_retained_instance_id
        retained_bytes = req.agentic_kv_retained_per_rank_bytes
        if retained_instance is not None:
            if retained_instance != self.instance_id:
                raise RuntimeError(
                    "P/D handoff selected decode instance "
                    f"{self.instance_id}, but reusable KV is retained on "
                    f"instance {retained_instance}")
            if retained_bytes < 0 or retained_bytes > kv_size:
                raise RuntimeError(
                    "Invalid retained agentic KV allocation: "
                    f"retained={retained_bytes}, total={kv_size}")
        elif retained_bytes != 0:
            raise RuntimeError(
                "Retained agentic KV bytes require a retained decode "
                f"instance: retained={retained_bytes}")
        return max(0, kv_size - retained_bytes)

    # add decode request to decode instance from prefill instnace
    def add_decode(
            self, req, admitted_hbm_bytes=None,
            preallocated_hbm_bytes=None, completion_time_ns=None):
        if (admitted_hbm_bytes is not None
                and preallocated_hbm_bytes is not None):
            raise RuntimeError(
                "P/D handoff cannot be both post-admitted and preallocated")
        if req.agentic_kv_owner_instance_id is not None:
            raise RuntimeError(
                "P/D handoff attempted while agentic KV is still owned by "
                f"instance {req.agentic_kv_owner_instance_id}; the prefill "
                "scheduler must release it before decode allocation")
        if self.enable_prefix_caching:
            if preallocated_hbm_bytes is not None:
                raise RuntimeError(
                    "P/D receive preallocation is incompatible with generic "
                    "prefix caching")
            req.instance_id = self.instance_id
            self.memory.prefix_match(req)
            kv_size = self.memory.get_evict_kv(req)
            evict_size = max(0, kv_size - self.memory.avail_size(Device.NPU))
            if evict_size > 0:
                self.memory.evict_prefix_cache(evict_size, Device.NPU)
            self.memory.cache_unfinished_req(req, Device.NPU)
        else:
            kv_size = self.memory.get_total_kv(req)
            retained_instance = req.agentic_kv_retained_instance_id
            retained_bytes = req.agentic_kv_retained_per_rank_bytes
            if retained_instance is not None:
                if retained_instance != self.instance_id:
                    raise RuntimeError(
                        "P/D handoff selected decode instance "
                        f"{self.instance_id}, but reusable KV is retained on "
                        f"instance {retained_instance}")
                if retained_bytes < 0 or retained_bytes > kv_size:
                    raise RuntimeError(
                        "Invalid retained agentic KV allocation: "
                        f"retained={retained_bytes}, total={kv_size}")
            allocation_size = max(0, kv_size - retained_bytes)
            if preallocated_hbm_bytes is not None:
                if req.pd_decode_target_instance_id != self.instance_id:
                    raise RuntimeError(
                        "P/D preallocated destination changed before "
                        f"handoff: reserved={req.pd_decode_target_instance_id}, "
                        f"actual={self.instance_id}")
                if int(preallocated_hbm_bytes) != allocation_size:
                    raise RuntimeError(
                        "P/D preallocated receive size changed before "
                        f"handoff: reserved={preallocated_hbm_bytes}, "
                        f"required={allocation_size}")
                if (req.pd_decode_reserved_per_rank_bytes
                        != allocation_size):
                    raise RuntimeError(
                        "Request P/D reservation does not match the "
                        f"preallocated handoff: request="
                        f"{req.pd_decode_reserved_per_rank_bytes}, "
                        f"required={allocation_size}")
                if req.pd_decode_full_per_rank_bytes != kv_size:
                    raise RuntimeError(
                        "P/D full destination size changed before handoff: "
                        f"reserved={req.pd_decode_full_per_rank_bytes}, "
                        f"required={kv_size}")
                if (req.pd_kv_ownership_state == "handoff_pending"
                        and int(req.pd_decode_owned_per_rank_bytes) != kv_size):
                    raise RuntimeError(
                        "P/D accumulated D ownership changed before "
                        f"handoff: owned="
                        f"{req.pd_decode_owned_per_rank_bytes}, "
                        f"required={kv_size}")
                if (req.pd_kv_ownership_state == "handoff_pending"
                        and int(
                            req.pd_prefill_handoff_released_per_rank_bytes)
                        != int(req.pd_prefill_full_per_rank_bytes)):
                    raise RuntimeError(
                        "P/D handoff did not release the complete P prompt: "
                        f"released="
                        f"{req.pd_prefill_handoff_released_per_rank_bytes}, "
                        f"full={req.pd_prefill_full_per_rank_bytes}")
            if (admitted_hbm_bytes is not None
                    and int(admitted_hbm_bytes) != allocation_size):
                raise RuntimeError(
                    "P/D decode admission size changed before allocation: "
                    f"admitted={admitted_hbm_bytes}, "
                    f"required={allocation_size}")
            if preallocated_hbm_bytes is None:
                self.memory.allocate(allocation_size, Device.NPU)
            req.instance_id = self.instance_id
            req.agentic_kv_retained_instance_id = None
            req.agentic_kv_retained_per_rank_bytes = 0
            req.pd_decode_target_instance_id = None
            req.pd_decode_full_per_rank_bytes = 0
            req.pd_decode_reserved_per_rank_bytes = 0
            # P already freed this full-prompt preallocation when prefill
            # completed. D owns an independently preallocated receive buffer.
            req.pd_prefill_full_per_rank_bytes = 0
            req.pd_prefill_reserved_per_rank_bytes = 0
            req.pd_prefill_preallocated_per_rank_bytes = 0
            req.pd_prefill_owned_per_rank_bytes = 0
            req.pd_decode_handoff_owned_per_rank_bytes = int(kv_size)
            req.pd_decode_owned_per_rank_bytes = 0
            req.pd_kv_ownership_state = "ordinary_decode"
        req.agentic_kv_owner_instance_id = self.instance_id
        if req.generated_tokens >= req.requested_output_tokens:
            if completion_time_ns is None:
                raise RuntimeError(
                    "A P/D handoff with all output tokens generated requires "
                    "its completion timestamp")
            return self._finish_request(req, int(completion_time_ns))
        self.enqueue_request(req)
        return None
    
    # get first request's arrival time
    def get_first_arrival_time(self):
        return self.first_arrival_time if self.first_arrival_time != 0 else 1 # need to add event handler at first
    
    # Merge request pools, preserving scheduler eligibility order.
    def _merge_by_arrival_id(self, left, right):
        # Chunked-prefill reordering means ``left`` is not necessarily sorted,
        # so perform a small bounded sort (the queue is capped by max_num_seqs)
        # instead of relying on two pre-sorted inputs.
        return sorted(
            left + right,
            key=lambda request: (request.ready_time, request.id),
        )
    
    # print total system request metrics (TTFT, TPOT, ITL)
    def print_result(self):
        # Extract ttft, tpot, and itl values from the completed requests
        ttft_values = [req.ttft for req in self.done]
        tpot_values = [req.tpot for req in self.done]
        itl_values = [itl for req in self.done for itl in req.itl]

        def _render(title: str, values, num_space=0):
            print_rule(f"[sim.tagline]{title}[/]")
            if not values:
                print_markup(f"No {title.split()[0]} data available")
                return
            mean = np.mean(values) / 1_000_000
            median = np.median(values) / 1_000_000
            p99 = np.percentile(values, 99) / 1_000_000
            label = title.split()[-1] if title != "Time to First Token" else "TTFT"
            # Map to the metric short-name used in the detail rows.
            short = {
                "Time to First Token": "TTFT",
                "Time per Output Token (excl. 1st token)": "TPOT",
                "Inter-token Latency": "ITL",
            }[title]
            spacing = " " * num_space
            print_markup(f"Mean {short} (ms){spacing}:                                                     {mean:.2f}")
            print_markup(f"Median {short} (ms){spacing}:                                                   {median:.2f}")
            print_markup(f"P99 {short} (ms){spacing}:                                                      {p99:.2f}")

        _render("Time to First Token", ttft_values)
        _render("Time per Output Token (excl. 1st token)", tpot_values)
        _render("Inter-token Latency", itl_values, num_space=1)

    # print each request results
    def print_request_result(self):
        # sort in id order
        self.done.sort(key=lambda x : x.id)
        for i in self.done:
            print(i)
        return

    # check all the request is done
    def is_request_empty(self):
        if len(self.request) == 0 and len(self.inflight) == 0:
            return True
        else:
            return False
        
    # save requests information to an output file
    def save_output(self, output_file, is_append=False):
        if not os.path.isabs(output_file):
            output_file = f'../{output_file}'
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        mode = 'a' if is_append else 'w'
        with open(output_file, mode=mode, newline='') as file:
            # Initialize the CSV writer
            writer = csv.writer(file)
            
            # Write the column headers
            if not is_append:
                writer.writerow(['instance id', 'request id', 'model', 'input', 'output',
                                'generated_tokens',
                                'arrival', 'end_time', 'latency',
                                'queuing_delay',
                                'first_schedule_time_ns',
                                'first_schedule_eligibility_time_ns',
                                'scheduler_queue_wait_ns',
                                'TTFT', 'TPOT', 'ITL',
                                'session_id', 'sub_request_index',
                                'source_session_id', 'session_template_index',
                                'session_epoch', 'session_offered_time_ns',
                                'session_admission_time_ns',
                                'session_admission_queue_wait_ns',
                                'prefix_reuse_tokens', 'prefix_reuse_source',
                                'return_gap_type', 'return_gap_source',
                                'return_gap_ns',
                                'agentic_kv_hit_tokens', 'agentic_kv_recompute_tokens',
                                'agentic_kv_residency_at_return',
                                'agentic_kv_source', 'agentic_kv_restore_ns',
                                'agentic_kv_owner_gate_ns',
                                'agentic_kv_restore_issue_time_ns',
                                'agentic_kv_target_hbm_ready_time_ns',
                                'agentic_kv_restore_ready_time_ns',
                                'agentic_kv_fresh_prompt_tokens',
                                'agentic_kv_overlap_cutoff_tokens',
                                'agentic_kv_restore_compute_overlap_ns',
                                'agentic_kv_restore_gate_wait_ns',
                                'pd_pair_fifo_wait_ns',
                                'agentic_kv_prepare_boundary_wait_ns',
                                'agentic_kv_source_demotion_join_wait_ns',
                                'agentic_kv_hbm_admission_wait_ns',
                                'agentic_kv_transient_dram_capacity_wait_ns',
                                'agentic_kv_restore_queue_wait_ns',
                                'agentic_kv_restore_service_ns',
                                'pd_decode_capacity_wait_ns',
                                'pd_decode_admission_wait_ns',
                                'pd_decode_admission_critical_wait_ns',
                                'pd_prefill_capacity_wait_ns',
                                'pd_prefill_admission_wait_ns',
                                'pd_prefill_admission_critical_wait_ns',
                                'pd_launch_admission_wait_ns',
                                'pd_launch_admission_critical_wait_ns',
                                'pd_chunk_admission_count',
                                'pd_chunk_cancelled_admission_count',
                                'pd_chunk_admitted_tokens_total',
                                'pd_chunk_prefill_admitted_per_rank_bytes',
                                'pd_chunk_decode_admitted_per_rank_bytes',
                                'pd_chunk_admission_wait_ns_total',
                                'pd_chunk_admission_critical_wait_ns_total',
                                'pd_chunk_successful_admission_wait_ns_total',
                                'pd_chunk_successful_admission_critical_wait_ns_total',
                                'pd_chunk_cancelled_admission_wait_ns_total',
                                'pd_chunk_cancelled_admission_critical_wait_ns_total',
                                'pd_chunk_prefill_peak_hbm_used_per_rank_bytes',
                                'pd_chunk_decode_peak_hbm_used_per_rank_bytes',
                                'pd_prefill_initial_restored_per_rank_bytes',
                                'pd_prefill_handoff_released_per_rank_bytes',
                                'pd_decode_handoff_owned_per_rank_bytes',
                                'active_prefill_recompute_preemptions',
                                'active_prefill_recompute_tokens',
                                'active_prefill_recompute_frontier_tokens',
                                'pd_active_prefill_recompute_generation',
                                'agentic_kv_restored_tokens_discarded_by_active_prefill_recompute',
                                'pd_kv_ownership_state'])
            
            # Write each request's information
            for req in self.done:
                writer.writerow([
                    req.instance_id,
                    req.id,
                    req.model,
                    req.input,
                    req.output - req.input,
                    req.generated_tokens,
                    req.arrival,
                    req.end_time,
                    req.latency,
                    req.queuing_delay,
                    req.first_schedule_time_ns,
                    req.first_schedule_eligibility_time_ns,
                    req.scheduler_queue_wait_ns,
                    req.ttft,
                    req.tpot,
                    req.itl,
                    req.session_id,
                    req.sub_request_index,
                    req.source_session_id,
                    req.session_template_index,
                    req.session_epoch,
                    req.session_offered_time_ns,
                    req.session_admission_time_ns,
                    req.session_admission_queue_wait_ns,
                    req.prefix_reuse_tokens,
                    req.prefix_reuse_source,
                    req.return_gap_type,
                    req.return_gap_source,
                    req.return_gap_ns,
                    req.agentic_kv_hit_tokens,
                    req.agentic_kv_recompute_tokens,
                    req.agentic_kv_residency_at_return,
                    req.agentic_kv_source,
                    req.agentic_kv_restore_ns,
                    req.agentic_kv_owner_gate_ns,
                    req.agentic_kv_restore_issue_time_ns,
                    req.agentic_kv_target_hbm_ready_time_ns,
                    req.agentic_kv_restore_ready_time_ns,
                    req.agentic_kv_fresh_prompt_tokens,
                    req.agentic_kv_overlap_cutoff_tokens,
                    req.agentic_kv_restore_compute_overlap_ns,
                    req.agentic_kv_restore_gate_wait_ns,
                    req.pd_pair_fifo_wait_ns,
                    req.agentic_kv_prepare_boundary_wait_ns,
                    req.agentic_kv_source_demotion_join_wait_ns,
                    req.agentic_kv_hbm_admission_wait_ns,
                    req.agentic_kv_transient_dram_capacity_wait_ns,
                    req.agentic_kv_restore_queue_wait_ns,
                    req.agentic_kv_restore_service_ns,
                    req.pd_decode_capacity_wait_ns,
                    req.pd_decode_admission_wait_ns,
                    req.pd_decode_admission_critical_wait_ns,
                    req.pd_prefill_capacity_wait_ns,
                    req.pd_prefill_admission_wait_ns,
                    req.pd_prefill_admission_critical_wait_ns,
                    req.pd_launch_admission_wait_ns,
                    req.pd_launch_admission_critical_wait_ns,
                    req.pd_chunk_admission_count,
                    req.pd_chunk_cancelled_admission_count,
                    req.pd_chunk_admitted_tokens_total,
                    req.pd_chunk_prefill_admitted_per_rank_bytes,
                    req.pd_chunk_decode_admitted_per_rank_bytes,
                    req.pd_chunk_admission_wait_ns_total,
                    req.pd_chunk_admission_critical_wait_ns_total,
                    req.pd_chunk_successful_admission_wait_ns_total,
                    req.pd_chunk_successful_admission_critical_wait_ns_total,
                    req.pd_chunk_cancelled_admission_wait_ns_total,
                    req.pd_chunk_cancelled_admission_critical_wait_ns_total,
                    req.pd_chunk_prefill_peak_hbm_used_per_rank_bytes,
                    req.pd_chunk_decode_peak_hbm_used_per_rank_bytes,
                    req.pd_prefill_initial_restored_per_rank_bytes,
                    req.pd_prefill_handoff_released_per_rank_bytes,
                    req.pd_decode_handoff_owned_per_rank_bytes,
                    req.active_prefill_recompute_preemptions,
                    req.active_prefill_recompute_tokens,
                    req.active_prefill_recompute_frontier_tokens,
                    req.pd_active_prefill_recompute_generation,
                    req.agentic_kv_restored_tokens_discarded_by_active_prefill_recompute,
                    req.pd_kv_ownership_state,
                ])


def main():
    pass

if __name__ == "__main__":
    main()

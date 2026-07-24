# class that manages request of astra-sim
class Request:
    def __init__(self, id, model, input, output, arrival, instance_id, input_hash_ids=None, output_hash_ids=None, is_init=True):
        self.id = id
        self.model = model
        self.input = input  # Always keep original input length
        self.output = output
        self.arrival = arrival
        self.instance_id = instance_id
        self.is_init = is_init
        self.original_input = input
        self.num_computed_tokens = 0  # Tracks actual computed tokens (vLLM style)
        # Count user-visible output tokens explicitly. ``num_computed_tokens``
        # is the materialized KV context and intentionally trails output by
        # one token: prefill produces output token 1 without adding that token
        # to KV, and each decode step consumes the previous output to produce
        # the next one.  Conflating the two loses the first token across a P/D
        # handoff because P computes prompt KV but does not run the D-side
        # output head.
        self.generated_tokens = 0
        self.evict = False
        # Active decode preemption can discard physical KV and rebuild the
        # complete context through the ordinary prefill path.  Keep the
        # rebuild target separate from ``original_input`` so request metrics
        # and the final output-length calculation retain their user-visible
        # prompt semantics.
        self.recompute_target_tokens = None
        self.active_recompute_tokens = 0
        # A P/D prefill can also lose its materialized context under finite
        # HBM pressure.  Unlike decode preemption, that request must keep the
        # ordinary prompt target and eventually execute the normal P->D
        # handoff.  The frontier marks the largest discarded prefix so each
        # replay of [0, frontier) is attributed as recomputation without
        # changing ``recompute_target_tokens``.
        self.active_prefill_recompute_frontier_tokens = 0
        self.active_prefill_recompute_preemptions = 0
        self.active_prefill_recompute_tokens = 0
        self.end_time = -1
        self.latency = -1
        self.queuing_delay = -1
        # ``queuing_delay`` is release-to-first-schedule for compatibility.
        # Keep the eligibility boundary separately so session reports can
        # distinguish restore/admission gates from time spent in the runnable
        # scheduler queue.  These values are write-once: chunked prefill must
        # not move the first-schedule timestamp on every chunk.
        self.first_schedule_time_ns = None
        self.first_schedule_eligibility_time_ns = None
        self.first_schedule_request_ready_time_ns = None
        self.first_schedule_resource_ready_time_ns = None
        self.scheduler_queue_wait_ns = None
        self.scheduler_resource_ready_time_ns = arrival
        self.ttft = -1
        self.tpot = -1
        self.itl = []
        self.recent_end = 0

        # For chunked prefill
        self.chunk_len = 0  # tokens scheduled for this request in the current step

        # For prefix caching modeling
        self.input_hash_ids = input_hash_ids
        self.output_hash_ids = output_hash_ids
        self.prefix_cache_hit = 0
        self.npu_cache_hit = 0
        self.storage_cache_hit = 0
        self.npu_last_node = None
        self.cpu_last_node = None
        self.storage_last_node = None

        # For prefix cache lock tracking
        self._prefix_locked = False
        self._prefix_npu_stats_counted = False
        self._prefix_storage_stats_counted = False

        # For agentic session tracking (informational, does not drive scheduling)
        self.session_id = None
        self.sub_request_index = None
        self.source_session_id = None
        self.session_template_index = None
        self.session_epoch = 0
        self.session_offered_time_ns = arrival
        self.session_admission_time_ns = arrival
        self.session_admission_queue_wait_ns = 0
        self.ready_time = arrival
        self.prefix_reuse_tokens = 0
        self.prefix_reuse_source = None
        # These fields describe the pause that ended immediately before this
        # request became ready. Trace schemas store that pause on the previous
        # sub-request, so the router copies it forward when releasing a
        # continuation.
        self.return_gap_type = "session_start"
        self.return_gap_source = "session_start"
        self.return_gap_ns = 0
        self.agentic_kv_hit_tokens = 0
        self.agentic_kv_recompute_tokens = 0
        # Physical resume provenance remains attached to the request even if
        # finite-HBM P/D progress later discards that restored prefix. Count
        # those original hit tokens once, at the first such preemption, so
        # offline reports can distinguish attempted from surviving reuse.
        self.agentic_kv_restored_tokens_discarded_by_active_prefill_recompute = 0
        # Physical residency observed immediately before resume is distinct
        # from the eventual source. A CPU/SSD object can fail HBM admission
        # and fall back to recomputation; retaining both labels makes that
        # path auditable.
        self.agentic_kv_residency_at_return = None
        self.agentic_kv_source = None
        self.hbf_online_execution = None
        self.hbf_online_route_reason = None
        self.agentic_kv_restore_ns = 0
        self.agentic_kv_owner_gate_ns = 0
        self.agentic_kv_restore_issue_time_ns = arrival
        self.agentic_kv_target_hbm_ready_time_ns = arrival
        self.agentic_kv_restore_ready_time_ns = arrival
        # If a capacity-triggered asynchronous swap-out was already in flight
        # at return, only its remaining tail gates this request.  The full
        # background copy service remains separately accounted by the tier
        # manager and is not added here.
        self.agentic_kv_source_demotion_join_wait_ns = 0
        self.agentic_kv_hbm_admission_wait_ns = 0
        self.agentic_kv_transient_dram_capacity_wait_ns = 0
        self.agentic_kv_restore_queue_wait_ns = 0
        self.agentic_kv_restore_service_ns = 0
        # In async-decode-join mode, only fresh prompt work strictly before
        # the final prompt token may run before restore completes. The final
        # token produces the first output and is the request-local decode join.
        self.agentic_kv_fresh_prompt_tokens = 0
        self.agentic_kv_overlap_cutoff_tokens = None
        self.agentic_kv_restore_gate_start_ns = 0
        self.agentic_kv_restore_gate_wait_ns = 0
        self.agentic_kv_restore_compute_overlap_ns = 0
        self.agentic_kv_async_decode_join = False
        self.agentic_kv_restore_gate_recorded = False
        # Strict P/D pair ordering can delay preparation before any HBM
        # capacity or transfer-resource wait begins. A subsequent scheduler/
        # engine-boundary wait can still precede physical restore issue. Keep
        # both components separate while preserving the full owner-ready gate.
        self.pd_pair_fifo_wait_ns = 0
        self.agentic_kv_prepare_boundary_wait_ns = 0
        # Scheduler._finish_request records the exact active KV allocation it
        # released so the active-to-idle handoff can verify byte ownership.
        self.agentic_kv_completion_released_per_rank_bytes = None
        # An active-HBM reclaim reservation belongs to exactly one scheduler
        # request. Deferring only that owner keeps unrelated work runnable while
        # its idle-KV LRU victim is copied out in the background.
        self.agentic_hbm_reclaim_ready_time_ns = None
        self.agentic_hbm_reclaim_original_ready_time_ns = None
        # Active session-KV ownership is explicit across P/D handoffs.  The
        # tier manager installs a resumed prefix on the prefill instance; the
        # prefill scheduler releases it before the decode scheduler allocates
        # the handed-off cache.  ``None`` means no scheduler currently owns a
        # tier-manager-provided allocation.
        self.agentic_kv_owner_instance_id = None
        # In a P/D continuation, the decode instance keeps the reusable
        # prefix while the prefill instance works on only the new suffix.
        # These fields describe that separately owned decode-side allocation;
        # add_decode() consumes it and allocates only the missing suffix.
        self.agentic_kv_retained_instance_id = None
        self.agentic_kv_retained_per_rank_bytes = 0
        # P/D receive admission. The decode destination is fixed at routing
        # time, but prompt KV is admitted incrementally at the same block
        # boundary as each P chunk. This avoids reserving a long context in
        # full before chunked prefill has executed it.
        self.pd_decode_target_instance_id = None
        self.pd_decode_full_per_rank_bytes = 0
        self.pd_decode_reserved_per_rank_bytes = 0
        self.pd_decode_owned_per_rank_bytes = 0
        self.pd_decode_admission_enqueued_ns = 0
        self.pd_decode_capacity_ready_ns = 0
        self.pd_decode_capacity_wait_ns = 0
        self.pd_decode_admission_ready_ns = 0
        self.pd_decode_admission_wait_ns = 0
        self.pd_decode_admission_critical_wait_ns = 0
        # P retains the accumulated prompt allocation while prefill is active.
        # A restored prefix is the initial ownership; every later block is
        # claimed atomically with the matching D receive block.
        self.pd_prefill_full_per_rank_bytes = 0
        self.pd_prefill_initial_restored_per_rank_bytes = 0
        self.pd_prefill_reserved_per_rank_bytes = 0
        self.pd_prefill_owned_per_rank_bytes = 0
        self.pd_prefill_admission_enqueued_ns = 0
        self.pd_prefill_capacity_ready_ns = 0
        self.pd_prefill_capacity_wait_ns = 0
        self.pd_prefill_admission_ready_ns = 0
        self.pd_prefill_admission_wait_ns = 0
        self.pd_prefill_admission_critical_wait_ns = 0
        self.pd_prefill_preallocated_per_rank_bytes = 0
        # One admitted chunk is frozen until its graph commits. A request with
        # a pending claim remains in the P queue but is not runnable; this lets
        # unrelated continuous batches proceed while HBM reclaim completes.
        self.pd_chunk_claim_pending = False
        self.pd_chunk_admitted_tokens = 0
        self.pd_chunk_admission_target_tokens = 0
        self.pd_chunk_admission_enqueued_ns = 0
        self.pd_chunk_prefill_capacity_ready_ns = 0
        self.pd_chunk_decode_capacity_ready_ns = 0
        self.pd_chunk_admission_ready_ns = 0
        self.pd_chunk_admission_wait_ns = 0
        self.pd_chunk_admission_critical_wait_ns = 0
        self.pd_chunk_admission_count = 0
        self.pd_chunk_cancelled_admission_count = 0
        self.pd_chunk_admitted_tokens_total = 0
        self.pd_chunk_prefill_admitted_per_rank_bytes = 0
        self.pd_chunk_decode_admitted_per_rank_bytes = 0
        self.pd_chunk_admission_wait_ns_total = 0
        self.pd_chunk_admission_critical_wait_ns_total = 0
        self.pd_chunk_successful_admission_wait_ns_total = 0
        self.pd_chunk_successful_admission_critical_wait_ns_total = 0
        self.pd_chunk_cancelled_admission_wait_ns_total = 0
        self.pd_chunk_cancelled_admission_critical_wait_ns_total = 0
        self.pd_chunk_prefill_peak_hbm_used_per_rank_bytes = 0
        self.pd_chunk_decode_peak_hbm_used_per_rank_bytes = 0
        self.pd_chunk_admission_history = []
        self.pd_prefill_handoff_released_per_rank_bytes = 0
        self.pd_decode_handoff_owned_per_rank_bytes = 0
        self.pd_kv_ownership_state = "unbound"
        # Incremented whenever finite-HBM progress discards an active P/D
        # prefill.  Admission history is retained across generations for
        # gross-work accounting, while live ownership counters are reset.
        self.pd_active_prefill_recompute_generation = 0
        # P and D chunk claims become usable as one atomic request-local gate.
        # Aggregate launch fields remain the canonical non-additive wait.
        self.pd_launch_admission_ready_ns = 0
        self.pd_launch_admission_wait_ns = 0
        self.pd_launch_admission_critical_wait_ns = 0
        # P/D prompt-KV handoff ownership. A prefix restored into P HBM from
        # CPU/SSD is not present on D and must ride the first successful P
        # graph exactly once. A prefix retained on D is deliberately excluded;
        # only newly computed suffix KV crosses P->D in that case.
        #
        # The scheduler stages per-batch token counts without mutating these
        # fields. They advance only after every ASTRA rank completes the batch,
        # which prevents a failed graph from consuming the one-shot prefix.
        self.pd_kv_handoff_tracking_enabled = False
        self.pd_restored_prefix_handoff_pending_tokens = 0
        self.pd_restored_prefix_handoff_sent_tokens = 0
        self.pd_new_kv_handoff_sent_tokens = 0

    # to print the request information
    def __str__(self):
        return str(self.__dict__) 

    def add_latency(self, end_time):
        if self.generated_tokens != self.requested_output_tokens:
            raise RuntimeError(
                f"Request #{self.id} completed with the wrong output-token "
                f"count: generated={self.generated_tokens}, "
                f"requested={self.requested_output_tokens}")
        self.end_time = end_time
        self.latency = self.end_time - self.arrival
        self.input = self.original_input
        if self.requested_output_tokens == 1:
            self.tpot = 0
        else:
            self.tpot = ((self.latency - self.ttft)
                         // (self.requested_output_tokens - 1))

    def add_itl(self, current): # 
        self.itl.append(current - self.recent_end)
        self.recent_end = current

    def set_que_delay(self, current):
        if self.first_schedule_time_ns is not None:
            return
        eligibility = max(
            int(self.arrival),
            int(self.ready_time),
            int(self.scheduler_resource_ready_time_ns),
        )
        if eligibility > int(current):
            raise RuntimeError(
                f"Request #{self.id} scheduled before eligibility: "
                f"eligible={eligibility}, scheduled={current}")
        self.first_schedule_time_ns = int(current)
        self.first_schedule_eligibility_time_ns = eligibility
        self.first_schedule_request_ready_time_ns = int(self.ready_time)
        self.first_schedule_resource_ready_time_ns = int(
            self.scheduler_resource_ready_time_ns)
        self.scheduler_queue_wait_ns = int(current) - eligibility
        self.queuing_delay = current - self.arrival
    
    def set_ttft(self, current):
        self.ttft = current - self.arrival
        self.recent_end = current

    @property
    def requested_output_tokens(self):
        return int(self.output) - int(self.original_input)

    def record_output_token(self, current):
        """Record exactly one user-visible output at model completion.

        The first output establishes TTFT; only later outputs contribute ITL.
        This rule is independent of ``is_init`` because a P-only prefill has
        already cleared prompt-initialization state before the D instance
        produces the first output.
        """
        requested = self.requested_output_tokens
        if requested <= 0:
            raise RuntimeError(
                f"Request #{self.id} has no requested output tokens: "
                f"prompt={self.original_input}, total={self.output}")
        if self.generated_tokens >= requested:
            raise RuntimeError(
                f"Request #{self.id} generated too many output tokens: "
                f"generated={self.generated_tokens}, requested={requested}")
        if self.generated_tokens == 0:
            self.set_ttft(current)
        else:
            self.add_itl(current)
        self.generated_tokens += 1
        self.is_init = False
    
    def log(self):
        print("         scheduled request : {}".format(self.__dict__))

    @property
    def prefill_target_tokens(self):
        """Logical context length that the current prefill must materialize."""
        if self.recompute_target_tokens is not None:
            return self.recompute_target_tokens
        return self.original_input

    def begin_active_recompute(self):
        """Discard progress physically while preserving the logical request.

        The caller must free the request's current HBM allocation before this
        transition.  Recomputing through the normal prefill path makes the
        extra model work visible in the generated execution trace without
        pretending that the KV was restored from another memory tier.
        """
        if self.recompute_target_tokens is not None:
            raise RuntimeError(
                f"Request #{self.id} is already rebuilding active KV")
        target = int(self.num_computed_tokens)
        if target < self.original_input:
            raise RuntimeError(
                f"Request #{self.id} cannot be actively preempted during "
                f"prefill: computed={target}, prompt={self.original_input}")
        self.recompute_target_tokens = target
        self.active_recompute_tokens += target
        self.num_computed_tokens = 0
        self.chunk_len = 0
        self.evict = False

    def finish_active_recompute(self):
        """Return to decode after the saved context has been rebuilt."""
        if self.recompute_target_tokens is None:
            raise RuntimeError(
                f"Request #{self.id} has no active KV rebuild to finish")
        if self.num_computed_tokens < self.recompute_target_tokens:
            raise RuntimeError(
                f"Request #{self.id} finished active KV rebuild early: "
                f"computed={self.num_computed_tokens}, "
                f"target={self.recompute_target_tokens}")
        self.recompute_target_tokens = None

    def begin_active_prefill_recompute(self):
        """Discard an in-progress P/D prompt and replay it from token zero.

        Physical P and D ownership must be released atomically by the router
        before this transition.  The request remains an ordinary prefill so
        completion still produces its first output and performs the P->D
        handoff.  Arrival, admission, first-schedule, and TTFT state are
        intentionally untouched.
        """
        if self.recompute_target_tokens is not None:
            raise RuntimeError(
                f"Request #{self.id} cannot mix decode and P/D prefill "
                "recomputation")
        discarded = int(self.num_computed_tokens)
        if discarded <= 0 or discarded >= int(self.original_input):
            raise RuntimeError(
                f"Request #{self.id} has no partial P/D prefill to discard: "
                f"computed={discarded}, prompt={self.original_input}")
        if self.generated_tokens != 0:
            raise RuntimeError(
                f"Request #{self.id} generated output before P/D prefill "
                "recomputation")
        restored_hit_tokens = int(self.agentic_kv_hit_tokens)
        generation = int(self.pd_active_prefill_recompute_generation)
        restored_already_discarded = int(
            self
            .agentic_kv_restored_tokens_discarded_by_active_prefill_recompute)
        if (restored_hit_tokens < 0
                or (generation == 0
                    and (restored_hit_tokens > discarded
                         or restored_already_discarded != 0))
                or (generation > 0
                    and restored_already_discarded != restored_hit_tokens)):
            raise RuntimeError(
                f"Request #{self.id} has invalid restored-hit ownership at "
                f"P/D preemption: hit={restored_hit_tokens}, "
                f"materialized={discarded}, generation={generation}, "
                f"already_discarded={restored_already_discarded}")
        self.active_prefill_recompute_frontier_tokens = max(
            int(self.active_prefill_recompute_frontier_tokens), discarded)
        self.active_prefill_recompute_preemptions += 1
        self.active_prefill_recompute_tokens += discarded
        self.active_recompute_tokens += discarded
        if generation == 0:
            self.agentic_kv_restored_tokens_discarded_by_active_prefill_recompute += (
                restored_hit_tokens)
        self.pd_active_prefill_recompute_generation += 1
        self.num_computed_tokens = 0
        self.chunk_len = 0
        self.evict = False
        return discarded
    
    def is_prefill(self):
        """Check if request is still in prefill phase (has tokens left to compute)"""
        return self.num_computed_tokens < self.prefill_target_tokens

# class that manages batch of astra-sim
class Batch:
    def __init__(self, batch_id, model, total_len, kv_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, batch_time, kv_size, evict=0, load=0):
        self.batch_id = batch_id
        self.model = model
        self.total_len = total_len
        self.kv_len = kv_len
        self.batch_time = batch_time
        self.fired = [] # systems that fired this batch
        self.requests = []
        self.end = []
        # vllm
        self.kv_size = kv_size
        self.evict = evict
        self.load = load
        # for attn prediction
        self.q_list = q_list
        self.k_list = k_list
        self.num_prefill = num_prefill
        self.num_decode = num_decode
        self.prefill_q_list = prefill_q_list
        self.prefill_k_list = prefill_k_list
        self.decode_k_list = decode_k_list

        # for debugging
        self.scheduled_tokens = None
        # Agentic source composition is captured when the scheduler finalizes
        # the batch. In synchronous mode, an exposed pre-dispatch swap gate is
        # associated with the first batch formed after that gate clears.
        self.agentic_source_counts = {}
        self.agentic_return_gap_type_counts = {}
        self.agentic_mixed_hbm_lower_tier = False
        self.agentic_source_return_counts = {}
        self.agentic_sync_swap_wait_ns = 0
        self.agentic_sync_swap_directions = ()
        self.agentic_sync_swap_barrier_before_batch = False
        # Per-request P->D KV ownership staged for this graph. The aggregate
        # restored-prefix count augments ``total_len`` in every transformer
        # layer's KV send, while the maps let Scheduler commit exactly once
        # after successful graph completion.
        self.pd_restored_prefix_handoff_tokens = 0
        self.pd_restored_prefix_handoff_by_request = {}
        self.pd_new_kv_handoff_tokens = 0
        self.pd_new_kv_handoff_by_request = {}
        self.pd_kv_handoff_committed = False
    def log(self):
        print("-------------------------Batch Log------------------------")
        for key in self.__dict__.keys():
            if key == 'requests':
                continue
            print("         {} : {}".format(key, self.__dict__[key]))
        for req in self.requests:
            req.log()
        print("----------------------------------------------------------")

import bisect
import json
import os
import random
from .logger import get_logger
from .memory_model import Device
from .session_admission import SessionAdmissionConfig


_NS_PER_SECOND = 1_000_000_000


class Router:
    def __init__(
            self,
            num_instances,
            schedulers, req_num,
            routing_policy="RR",
            seed=42,
            agentic_kv_manager=None,
            session_admission=None,
            full_model_hbf_adapter=None,
            full_model_hbf_gpu_hbm_bridge=None,
    ):
        self.schedulers = schedulers
        self.num_instances = num_instances
        self.prefill_schedulers = [s for s in schedulers if s.pd_type != "decode"]
        self.prefill_instances = len(self.prefill_schedulers)
        self.decode_schedulers = [s for s in schedulers if s.pd_type == "decode"]
        self.decode_instances = len(self.decode_schedulers)
        if (
            (full_model_hbf_adapter is None)
            != (full_model_hbf_gpu_hbm_bridge is None)
        ):
            raise ValueError(
                "full-model HBF Router integration requires both "
                "full_model_hbf_adapter and "
                "full_model_hbf_gpu_hbm_bridge")
        if (
            full_model_hbf_adapter is not None
            and agentic_kv_manager is not None
        ):
            raise ValueError(
                "full-model HBF Router integration is mutually exclusive "
                "with legacy agentic_kv_manager tiering")
        self.full_model_hbf_adapter = full_model_hbf_adapter
        self.full_model_hbf_gpu_hbm_bridge = (
            full_model_hbf_gpu_hbm_bridge)
        self._full_model_hbf_pd_decode_by_prefill = {}
        if self.full_model_hbf_adapter is not None:
            self._validate_full_model_hbf_integration()
        self.req_num = req_num
        self.routing_policy = routing_policy.upper()
        self.seed = seed
        self._rnd = random.Random(seed) if seed is not None else random
        self.prefill_rr_counter = 0
        self.decode_rr_counter = 0

        # Pending requests (loaded but not yet routed)
        self._pending_requests = []
        self._pending_idx = 0
        self._enable_prefix_caching = False
        self._enable_agentic_kv = agentic_kv_manager is not None
        self._is_init = True
        self.agentic_kv_manager = agentic_kv_manager
        self.session_admission = (
            session_admission
            if session_admission is not None
            else SessionAdmissionConfig()
        )
        if not isinstance(self.session_admission, SessionAdmissionConfig):
            raise TypeError(
                "session_admission must be a SessionAdmissionConfig")
        # P/D keeps two independent affinities.  Idle session KV is owned by
        # the decode instance after the normal prefill handoff, while the
        # next continuation first returns to its prefill instance.  The
        # historical name remains the prefill/colocated affinity map for
        # backward compatibility with colocated routing.
        self._session_affinity = {}
        self._session_decode_affinity = {}
        # P/D launch binding is separate from incremental chunk admission.
        # Restores can keep a request here until its initial eligibility, but
        # the prompt suffix is not reserved in full. Each P graph instead
        # acquires one atomic P/D block claim through the callback installed on
        # its prefill scheduler below.
        self._pending_decode_handoffs = {}
        self._pending_full_model_hbf_prefill_launches = []
        self._pending_prefill_launches = []
        self._pending_pd_chunk_admissions = {}
        # GPU-facing operations that need an iteration boundary in the strict
        # synchronous sensitivity live here instead of blocking the globally
        # sorted arrival queue. Async cold direct-fabric copies do not acquire
        # this lock; their completion delays only the returning owner request.
        self._pending_sync_preparations = []
        # Lower-tier restores waiting for destination HBM have no synthetic
        # polling timestamp. Retry only after the target capacity state
        # changes (normally at a model or LRU-demotion completion).
        self._pending_capacity_preparations = []
        # A strict P/D continuation acquires physical P restore HBM and may
        # retain its D prefix before the remaining P/D suffix admission is
        # known to fit.  Admit that preparation in FIFO order per fixed P/D
        # pair.  Otherwise a later continuation can hold both engines' free
        # HBM behind an older, non-fitting handoff and create a circular
        # hold-and-wait with no future event. Independent P/D pairs retain
        # their own admission queues and can continue concurrently.
        self._pd_admission_owner = {}
        self._pending_pd_admission_waits = []
        # freeze_session_admission() must release synchronous prepare locks
        # immediately, but the corresponding idle session KV still needs to
        # be ended at the final measurement cutoff. Preserve those rows until
        # finalize_measurement_censoring() performs the physical cleanup.
        self._frozen_sync_preparations = []
        self._censored_completed_pd_prefill_audits = []
        self._censored_full_model_hbf_gpu_queue_audits = []

        if self.agentic_kv_manager is not None and self.decode_schedulers:
            self._validate_strict_pd_mapping()
        for scheduler in self.prefill_schedulers:
            scheduler.pd_chunk_admission_callback = (
                self.admit_pd_prefill_chunk)

        # Agentic session dependency tracking
        self._deferred_sessions = {}     # session_id -> session state dict
        self._request_to_session = {}    # request_id -> (session_id, sub_request_index)
        self._next_request_id = 0        # monotonic counter for unique request IDs
        # In backlog mode, templates wait outside the active population. A
        # slot is held from first-call admission through final completion,
        # including all closed-loop human/tool pauses between calls.
        self._session_backlog = []
        self._session_backlog_idx = 0
        self._active_sessions = set()
        self._session_lifecycle = {}
        self._session_templates_loaded = 0
        self._sessions_admitted = 0
        self._sessions_completed = 0
        self._completed_session_ids = set()
        self._measurement_warmup_session_ids = ()
        self._measurement_warmup_session_id_set = frozenset()
        self._measurement_target_session_ids = ()
        self._measurement_target_session_id_set = frozenset()
        self._measurement_required_session_ids = ()
        self._measurement_required_session_id_set = frozenset()
        self._session_admission_frozen = False
        self._censored_pending_prepare_rows = 0

        if self.routing_policy == "RR":
            self._select_instance = self._rr_select
        elif self.routing_policy == "RAND":
            self._select_instance = self._rand_select
        elif self.routing_policy == "LOAD":
            self._select_instance = self._least_load_select
        elif self.routing_policy == "CUSTOM":
            self._select_instance = self._custom_select
        else:
            raise ValueError(f"Unknown routing_policy '{routing_policy}'. "
                             "Supported: RR, RAND, LOAD, CUSTOM")
        self.logger = get_logger(self.__class__)

    def _validate_full_model_hbf_integration(self):
        """Validate the adapter/bridge topology before routing any row."""
        adapter = self.full_model_hbf_adapter
        bridge = self.full_model_hbf_gpu_hbm_bridge
        adapter_methods = (
            "offer_raw_request",
            "decorate_gpu_metadata",
            "flush_admissions",
            "pop_gpu_hbm_events",
            "bind_native_gpu_request",
            "complete_native_gpu_request",
            "validate_queued_native_gpu_request",
            "censor_queued_native_gpu_request",
            "has_pending_native_gpu_requests",
            "has_pending_astra_dispatches",
            "reclaim_gpu_ready_for_hbm_pressure",
        )
        bridge_methods = (
            "validate_adapter_contract",
            "apply_events",
            "decorate_colocated_continuation",
            "bind_colocated_continuation",
            "decorate_pd_recompute",
            "bind_pd_recompute",
            "validate_pd_decode_prompt_capacity",
            "validate_pd_decode_request_capacity",
            "try_reserve_pd_decode",
            "pd_decode_reservation",
            "consume_pd_decode_reservation",
            "cancel_pd_decode_reservation",
        )
        for name in adapter_methods:
            if not callable(getattr(adapter, name, None)):
                raise TypeError(
                    "full_model_hbf_adapter lacks required "
                    f"{name}()")
        for name in bridge_methods:
            if not callable(getattr(bridge, name, None)):
                raise TypeError(
                    "full_model_hbf_gpu_hbm_bridge lacks required "
                    f"{name}()")

        scheduler_by_id = {}
        for scheduler in self.schedulers:
            instance_id = int(scheduler.instance_id)
            if instance_id in scheduler_by_id:
                raise ValueError(
                    "full-model HBF integration requires unique Scheduler "
                    f"instance IDs; duplicate={instance_id}")
            scheduler_by_id[instance_id] = scheduler
        try:
            bridge_schedulers = {
                int(instance_id): scheduler
                for instance_id, scheduler in bridge.schedulers.items()
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(
                "full_model_hbf_gpu_hbm_bridge.schedulers must be an "
                "instance-id mapping") from exc
        if set(bridge_schedulers) != set(scheduler_by_id):
            raise ValueError(
                "full-model HBF bridge Scheduler IDs differ from Router: "
                f"bridge={sorted(bridge_schedulers)}, "
                f"router={sorted(scheduler_by_id)}")
        for instance_id, scheduler in scheduler_by_id.items():
            if bridge_schedulers[instance_id] is not scheduler:
                raise ValueError(
                    "full-model HBF bridge must reference the Router's exact "
                    f"Scheduler object: instance={instance_id}")

        topology = getattr(bridge, "topology", None)
        resume_mode = getattr(adapter, "gpu_resume_mode", None)
        fallback_mode = getattr(bridge, "fallback_reuse_mode", None)
        if topology == "pd":
            if resume_mode != "recompute" or fallback_mode != "recompute":
                raise ValueError(
                    "full-model HBF P/D routing requires recompute fallback")
            if (
                len(self.prefill_schedulers) != 1
                or len(self.decode_schedulers) != 1
            ):
                raise ValueError(
                    "full-model HBF P/D integration currently supports "
                    "exactly one prefill and one decode Scheduler")
            expected_pair = (
                int(self.prefill_schedulers[0].instance_id),
                int(self.decode_schedulers[0].instance_id),
            )
            pairs = tuple(
                (int(prefill_id), int(decode_id))
                for prefill_id, decode_id in bridge.pd_pairs
            )
            if pairs != (expected_pair,):
                raise ValueError(
                    "full-model HBF P/D bridge must configure the Router's "
                    f"single fixed pair {expected_pair}; observed={pairs}")
            self._full_model_hbf_pd_decode_by_prefill = dict(pairs)
        elif topology == "colocated":
            if self.decode_schedulers:
                raise ValueError(
                    "a colocated full-model HBF bridge cannot be attached "
                    "to a P/D Router")
            if resume_mode != "sticky_reuse":
                raise ValueError(
                    "colocated full-model HBF routing requires sticky_reuse")
            if fallback_mode != "sticky_reuse":
                raise ValueError(
                    "colocated full-model HBF bridge requires sticky_reuse")
            if tuple(getattr(bridge, "pd_pairs", ())) != ():
                raise ValueError(
                    "colocated full-model HBF bridge cannot define P/D pairs")
        else:
            raise ValueError(
                "full-model HBF bridge topology must be 'colocated' or 'pd'")

        bridge.validate_adapter_contract(adapter)

    def drain_full_model_hbf_gpu_hbm_events(self):
        """Apply every pending adapter ownership event to finite GPU HBM."""
        if self.full_model_hbf_adapter is None:
            return ()
        events = self.full_model_hbf_adapter.pop_gpu_hbm_events()
        return self.full_model_hbf_gpu_hbm_bridge.apply_events(events)

    def flush_full_model_hbf_admissions(self, now_ns):
        """Flush one co-timed HBF admission set and its ownership events."""
        if self.full_model_hbf_adapter is None:
            return 0
        self.drain_full_model_hbf_gpu_hbm_events()
        admitted = self.full_model_hbf_adapter.flush_admissions(
            int(now_ns))
        self.drain_full_model_hbf_gpu_hbm_events()
        return admitted

    def censor_idle_full_model_hbf_native_queues(
            self, cutoff_time_ns):
        """Censor native GPU queues whose Scheduler has fully drained.

        This method is incremental by design.  A measurement boundary can
        arrive while one Scheduler still has a dispatched graph and another
        has only queued native work.  Each call immediately unwinds the idle
        Scheduler, skips the busy one, and may be repeated after subsequent
        model callbacks.

        HBF calls already accepted by ``FullModelHBFServingPool`` are not
        cancelled here.  They remain explicit ASTRA drain obligations; only
        native Scheduler queues are censored under the source cutoff.
        """

        cutoff_time_ns = int(cutoff_time_ns)
        if self.full_model_hbf_adapter is None:
            return {
                "cutoff_time_ns": cutoff_time_ns,
                "censored_requests": 0,
                "censored_request_audits": [],
                "skipped_busy_schedulers": [],
                "remaining_native_gpu_requests": False,
                "accepted_hbf_work_drains": False,
            }
        if not self._session_admission_frozen:
            raise RuntimeError(
                "full-model HBF native queues may be censored only after "
                "Router admission is frozen")

        self.drain_full_model_hbf_gpu_hbm_events()
        adapter = self.full_model_hbf_adapter
        queued_by_scheduler = [
            (scheduler, list(scheduler.request))
            for scheduler in self.schedulers
        ]
        queued_owner_by_request = {}
        for scheduler, queued in queued_by_scheduler:
            for request in queued:
                request_id = int(request.id)
                previous = queued_owner_by_request.get(request_id)
                if previous is not None:
                    raise RuntimeError(
                        "one full-model HBF request is queued on multiple "
                        "Schedulers: "
                        f"request={request_id}, instances="
                        f"({previous}, {scheduler.instance_id})")
                queued_owner_by_request[request_id] = int(
                    scheduler.instance_id)
        censored = []
        skipped = []
        for scheduler, queued in queued_by_scheduler:
            if scheduler.inflight:
                skipped.append({
                    "instance_id": int(scheduler.instance_id),
                    "inflight_batches": len(scheduler.inflight),
                    "queued_requests": len(queued),
                })
                continue
            cancel = getattr(
                scheduler,
                "censor_full_model_hbf_queued_request",
                None,
            )
            if not callable(cancel):
                raise TypeError(
                    "full-model HBF Scheduler lacks "
                    "censor_full_model_hbf_queued_request(): "
                    f"instance={scheduler.instance_id}")

            # Validate the complete idle queue before releasing any physical
            # bytes from this Scheduler.
            for request in queued:
                adapter.validate_queued_native_gpu_request(
                    request, now_ns=cutoff_time_ns)

            for request in queued:
                reservation_audit = (
                    self.full_model_hbf_gpu_hbm_bridge
                    .cancel_pd_decode_reservation(request)
                )
                memory_audit = cancel(
                    request, cutoff_time_ns)
                adapter_audit = adapter.censor_queued_native_gpu_request(
                    request, now_ns=cutoff_time_ns)
                audit = {
                    "request_id": int(request.id),
                    "session_id": (
                        None if request.session_id is None
                        else str(request.session_id)),
                    "instance_id": int(scheduler.instance_id),
                    "memory": memory_audit,
                    "decode_reservation": reservation_audit,
                    "adapter": adapter_audit,
                }
                censored.append(audit)
                self._censored_full_model_hbf_gpu_queue_audits.append(
                    audit)
        pending_launches = list(
            self._pending_full_model_hbf_prefill_launches)
        for pending in pending_launches:
            request = pending["request"]
            adapter.validate_queued_native_gpu_request(
                request, now_ns=cutoff_time_ns)
        for pending in pending_launches:
            request = pending["request"]
            reservation_audit = (
                self.full_model_hbf_gpu_hbm_bridge
                .cancel_pd_decode_reservation(request)
            )
            if reservation_audit is not None:
                raise RuntimeError(
                    "capacity-waiting HBF P request unexpectedly owned "
                    "a D-HBM reservation")
            adapter_audit = adapter.censor_queued_native_gpu_request(
                request, now_ns=cutoff_time_ns)
            audit = {
                "request_id": int(request.id),
                "session_id": str(request.session_id),
                "instance_id": int(
                    pending["prefill_scheduler"].instance_id),
                "memory": {
                    "request_id": int(request.id),
                    "not_scheduler_visible": True,
                    "freed_npu_per_rank_bytes": 0,
                },
                "decode_reservation": None,
                "adapter": adapter_audit,
            }
            censored.append(audit)
            self._censored_full_model_hbf_gpu_queue_audits.append(
                audit)
        self._pending_full_model_hbf_prefill_launches.clear()
        self.drain_full_model_hbf_gpu_hbm_events()
        return {
            "cutoff_time_ns": cutoff_time_ns,
            "censored_requests": len(censored),
            "censored_request_audits": censored,
            "skipped_busy_schedulers": skipped,
            "remaining_native_gpu_requests": (
                adapter.has_pending_native_gpu_requests()),
            "accepted_hbf_work_drains": (
                adapter.has_pending_astra_dispatches()),
        }

    # -----------------------------------------------------------------------
    # Instance selection policies
    # -----------------------------------------------------------------------

    def _get_counter(self, role):
        return self.decode_rr_counter if role == "decode" else self.prefill_rr_counter

    def _set_counter(self, role, value):
        if role == "decode":
            self.decode_rr_counter = value
        else:
            self.prefill_rr_counter = value

    def _rr_select(self, schedulers, role):
        num_instances = len(schedulers)
        idx = self._get_counter(role) % num_instances
        self._set_counter(role, idx + 1)
        return idx

    def _rand_select(self, schedulers, role):
        return self._rnd.randrange(len(schedulers))

    def _least_load_select(self, schedulers, role):
        """vLLM-style least-loaded routing, normalized by instance capacity."""
        best_idx = 0
        best_score = float('inf')
        num_instances = len(schedulers)
        start = self._get_counter(role) % num_instances
        for offset in range(num_instances):
            idx = (start + offset) % num_instances
            sched = schedulers[idx]
            waiting = len(sched.request)
            if role == "decode":
                waiting += len(self._pending_decode_handoffs.get(
                    sched.instance_id, ()))
            running = sum(len(b.requests) for b in sched.inflight)
            raw_score = waiting * 4 + running
            capacity = getattr(sched, "max_num_seqs", 0)
            score = raw_score
            if capacity not in (0, float('inf')):
                score = raw_score / capacity
            if score < best_score:
                best_score = score
                best_idx = idx
        self._set_counter(role, (best_idx + 1) % num_instances)
        return best_idx

    def _custom_select(self, schedulers, role):
        raise NotImplementedError("Implement custom routing policy.")

    # -----------------------------------------------------------------------
    # Request loading and real-time routing
    # -----------------------------------------------------------------------

    def load_requests(self, path, enable_prefix_caching=False, is_init=True):
        """Load requests from dataset into pending queue (not yet routed).

        Supports two JSONL formats:
        - Flat: {"input_toks", "output_toks", "arrival_time_ns", ...}
        - Agentic session: {"session_id", "arrival_time_ns", "sub_requests": [...]}

        For agentic sessions, only the first sub-request is added to the
        pending queue. Subsequent sub-requests are released dynamically
        via notify_request_completed() when predecessors finish.
        """
        if not os.path.isabs(path):
            path = os.path.join('..', path)
        self._enable_prefix_caching = enable_prefix_caching
        self._is_init = is_init
        loaded_lines = 0

        rows = []
        with open(path) as f:
            for line in f:
                if self.req_num > 0 and loaded_lines >= self.req_num:
                    break
                rows.append(json.loads(line))
                loaded_lines += 1
        if self.full_model_hbf_adapter is not None:
            invalid_rows = [
                index for index, row in enumerate(rows)
                if not row.get("sub_requests")
            ]
            if invalid_rows:
                raise ValueError(
                    "full-model HBF accepts only non-empty agentic "
                    "sub_requests; invalid rows at indices "
                    f"{invalid_rows[:5]}")
            model_limits = [
                int(scheduler.max_model_len)
                for scheduler in self.prefill_schedulers
                if getattr(scheduler, "max_model_len", None) is not None
            ]
            if model_limits:
                max_model_len = min(model_limits)
                oversized = []
                for row_index, row in enumerate(rows):
                    for call_index, sub_request in enumerate(
                            row["sub_requests"]):
                        total_tokens = (
                            int(sub_request["input_toks"])
                            + int(sub_request["output_toks"])
                        )
                        if total_tokens > max_model_len:
                            oversized.append(
                                (row_index, call_index, total_tokens))
                if oversized:
                    raise ValueError(
                        "full-model HBF request exceeds native GPU "
                        f"max_model_len={max_model_len}: "
                        f"{oversized[:5]}")
            if self.full_model_hbf_gpu_hbm_bridge.topology == "pd":
                decode_ids = sorted(set(
                    self._full_model_hbf_pd_decode_by_prefill.values()))
                for row in rows:
                    for sub_request in row["sub_requests"]:
                        for decode_id in decode_ids:
                            (
                                self.full_model_hbf_gpu_hbm_bridge
                                .validate_pd_decode_request_capacity(
                                    sub_request["input_toks"],
                                    sub_request["output_toks"],
                                    decode_instance_id=decode_id,
                                )
                            )

        if self.session_admission.mode in {"poisson", "backlog"}:
            flat_rows = [
                index for index, row in enumerate(rows)
                if 'sub_requests' not in row
            ]
            if flat_rows:
                raise ValueError(
                    f"session {self.session_admission.mode} mode accepts only "
                    "agentic rows with "
                    f"sub_requests; flat rows at indices {flat_rows[:5]}")
            empty_rows = [
                index for index, row in enumerate(rows)
                if not row.get('sub_requests')
            ]
            if empty_rows:
                raise ValueError(
                    f"session {self.session_admission.mode} mode requires "
                    "non-empty sub_requests; empty rows at indices "
                    f"{empty_rows[:5]}")
        self._session_templates_loaded += sum(
            1 for row in rows if row.get('sub_requests'))
        if self.session_admission.mode == "backlog":
            self._load_backlog_templates(rows)
            self._fill_backlog_slots(0)
        elif self.session_admission.mode == "poisson":
            self._load_poisson_sessions(rows, enable_prefix_caching)
        else:
            for template_index, row in enumerate(rows):
                if 'sub_requests' in row:
                    self._load_agentic_session(
                        row,
                        enable_prefix_caching,
                        template_index=template_index,
                    )
                else:
                    self._load_flat_request(row, enable_prefix_caching)

        # Sort pending requests by arrival time (agentic first sub-requests
        # may interleave with flat requests)
        self._pending_requests.sort(key=self._pending_time)

        self.logger.info("Loaded %d requests into pending queue "
                         "(%d agentic sessions deferred)",
                         len(self._pending_requests),
                         len(self._deferred_sessions))

    def _load_poisson_sessions(self, rows, enable_prefix_caching):
        """Replace first-call arrivals with deterministic exponential gaps."""
        rng = random.Random(self.session_admission.session_arrival_seed)
        arrival_ns = 0
        emitted = 0
        admission_limited = (
            self.session_admission.max_active_sessions > 0)
        for template_index, row in enumerate(rows):
            if not row.get('sub_requests'):
                continue
            if emitted:
                arrival_ns += int(
                    rng.expovariate(
                        self.session_admission.session_arrival_rate_sps)
                    * _NS_PER_SECOND
                )
            if admission_limited:
                session_id = str(
                    row.get('session_id', f'session_{template_index}'))
                if session_id in self._session_lifecycle:
                    raise ValueError(
                        f"Duplicate Poisson session_id {session_id!r}")
                descriptor = {
                    'row': row,
                    'source_session_id': session_id,
                    'runtime_session_id': session_id,
                    'template_index': template_index,
                    'epoch': 0,
                    'offered_time_ns': int(arrival_ns),
                    'planned_admission_index': emitted,
                }
                self._session_backlog.append(descriptor)
                self._session_lifecycle[session_id] = {
                    'session_id': session_id,
                    'source_session_id': session_id,
                    'template_index': template_index,
                    'epoch': 0,
                    'planned_admission_index': emitted,
                    'admission_index': None,
                    'measurement_warmup': False,
                    'measurement_target': False,
                    'measurement_required': False,
                    'measurement_role': 'outside_measurement_cohort',
                    'source_arrival_time_ns': int(row['arrival_time_ns']),
                    'offered_time_ns': int(arrival_ns),
                    'admission_time_ns': None,
                    'admission_queue_wait_ns': None,
                    'completion_time_ns': None,
                    'e2e_ns': None,
                    'status': 'waiting_for_admission',
                }
            else:
                self._load_agentic_session(
                    row,
                    enable_prefix_caching,
                    arrival_ns=arrival_ns,
                    source_arrival_ns=int(row['arrival_time_ns']),
                    template_index=template_index,
                )
            emitted += 1
        if admission_limited:
            self._fill_backlog_slots(0)

    def _load_backlog_templates(self, rows):
        """Build a deterministic finite backlog from agentic templates."""
        templates = []
        for template_index, row in enumerate(rows):
            if not row.get('sub_requests'):
                continue
            source_session_id = str(
                row.get('session_id', f'session_{template_index}'))
            for epoch in range(self.session_admission.backlog_epochs):
                runtime_session_id = (
                    f"{source_session_id}::template={template_index}"
                    f"::epoch={epoch}"
                )
                templates.append({
                    'row': row,
                    'source_session_id': source_session_id,
                    'runtime_session_id': runtime_session_id,
                    'template_index': template_index,
                    'epoch': epoch,
                })
        # Epoch-major ordering makes each pass reproduce the input order.
        templates.sort(key=lambda item: (
            item['epoch'], item['template_index']))
        planned_start = len(self._session_backlog)
        for offset, descriptor in enumerate(templates):
            session_id = descriptor['runtime_session_id']
            descriptor['planned_admission_index'] = planned_start + offset
            self._session_lifecycle[session_id] = {
                'session_id': session_id,
                'source_session_id': descriptor['source_session_id'],
                'template_index': descriptor['template_index'],
                'epoch': descriptor['epoch'],
                'planned_admission_index': (
                    descriptor['planned_admission_index']),
                'admission_index': None,
                'measurement_warmup': False,
                'measurement_target': False,
                'measurement_required': False,
                'measurement_role': 'outside_required_prefix',
                'source_arrival_time_ns': int(
                    descriptor['row']['arrival_time_ns']),
                'offered_time_ns': 0,
                'admission_time_ns': None,
                'admission_queue_wait_ns': None,
                'completion_time_ns': None,
                'e2e_ns': None,
                'status': 'backlog',
            }
        self._session_backlog.extend(templates)
        self._configure_measurement_target()

    def _configure_measurement_target(self):
        """Freeze the policy-independent admission-prefix cohort IDs."""
        if (self.session_admission.measurement_cohort_selection
                != 'admission_order'):
            return
        warmup_count = int(self.session_admission.warmup_completions)
        target_count = int(self.session_admission.measure_completions)
        if target_count == 0:
            target_count = len(self._session_backlog) - warmup_count
        required_count = warmup_count + target_count
        if (warmup_count < 0 or target_count <= 0
                or required_count > len(self._session_backlog)):
            raise ValueError(
                "admission_order fixed admission prefix requests "
                f"warmup={warmup_count} plus target={target_count} sessions "
                "but the backlog contains only "
                f"{len(self._session_backlog)}")
        warmup_ids = tuple(
            descriptor['runtime_session_id']
            for descriptor in self._session_backlog[:warmup_count]
        )
        target_ids = tuple(
            descriptor['runtime_session_id']
            for descriptor in self._session_backlog[
                warmup_count:required_count]
        )
        required_ids = warmup_ids + target_ids
        if len(required_ids) != len(set(required_ids)):
            raise RuntimeError(
                "Admission-order measurement prefix IDs are not unique")
        self._measurement_warmup_session_ids = warmup_ids
        self._measurement_warmup_session_id_set = frozenset(warmup_ids)
        self._measurement_target_session_ids = target_ids
        self._measurement_target_session_id_set = frozenset(target_ids)
        self._measurement_required_session_ids = required_ids
        self._measurement_required_session_id_set = frozenset(required_ids)
        for session_id in warmup_ids:
            lifecycle = self._session_lifecycle[session_id]
            lifecycle['measurement_warmup'] = True
            lifecycle['measurement_required'] = True
            lifecycle['measurement_role'] = 'fixed_admission_prefix_warmup'
        for session_id in target_ids:
            lifecycle = self._session_lifecycle[session_id]
            lifecycle['measurement_target'] = True
            lifecycle['measurement_required'] = True
            lifecycle['measurement_role'] = 'measurement_target'

    def _fill_backlog_slots(self, admission_time_ns):
        """Fill closed or Poisson-limited session admission slots."""
        mode = self.session_admission.mode
        admission_limited_poisson = (
            mode == "poisson"
            and self.session_admission.max_active_sessions > 0)
        if ((mode != "backlog" and not admission_limited_poisson)
                or self._session_admission_frozen):
            return 0
        admitted = 0
        capacity = self.session_admission.max_active_sessions
        while (len(self._active_sessions) < capacity
               and self._session_backlog_idx < len(self._session_backlog)):
            descriptor = self._session_backlog[self._session_backlog_idx]
            offered_time_ns = int(descriptor.get('offered_time_ns', 0))
            if offered_time_ns > int(admission_time_ns):
                break
            self._session_backlog_idx += 1
            session_id = descriptor['runtime_session_id']
            self._load_agentic_session(
                descriptor['row'],
                self._enable_prefix_caching,
                session_id=session_id,
                arrival_ns=int(admission_time_ns),
                source_arrival_ns=int(
                    descriptor['row']['arrival_time_ns']),
                source_session_id=descriptor['source_session_id'],
                template_index=descriptor['template_index'],
                epoch=descriptor['epoch'],
                insert_sorted=True,
            )
            self._activate_session(session_id, int(admission_time_ns))
            admitted += 1
        return admitted

    def _load_flat_request(self, row, enable_prefix_caching):
        """Load a single flat request into pending queue."""
        req_id = self._next_request_id
        self._next_request_id += 1
        req_data = {
            'index': req_id,
            'input_toks': int(row['input_toks']),
            'output_toks': int(row['input_toks'] + row['output_toks']),
            'arrival_time_ns': int(row['arrival_time_ns']),
        }
        if enable_prefix_caching or self._enable_agentic_kv:
            req_data['input_hash_ids'] = row.get('input_tok_ids', [])
            req_data['output_hash_ids'] = row.get('output_tok_ids', [])
        self._pending_requests.append(req_data)

    def _load_agentic_session(
            self, row, enable_prefix_caching, session_id=None,
            arrival_ns=None, source_session_id=None, template_index=None,
            epoch=0, insert_sorted=False, source_arrival_ns=None):
        """Load an agentic session: first sub-request to pending, rest deferred."""
        sub_reqs = row['sub_requests']
        if not sub_reqs:
            return 0
        if session_id is None:
            session_id = str(
                row.get('session_id', f'session_{self._next_request_id}'))
        if source_session_id is None:
            source_session_id = str(row.get('session_id', session_id))
        if session_id in self._deferred_sessions:
            raise ValueError(f"Duplicate active session_id {session_id!r}")
        base_id = self._next_request_id
        self._next_request_id += len(sub_reqs)
        if arrival_ns is None:
            arrival_ns = int(row['arrival_time_ns'])
        else:
            arrival_ns = int(arrival_ns)
        if source_arrival_ns is None:
            source_arrival_ns = int(row['arrival_time_ns'])
        else:
            source_arrival_ns = int(source_arrival_ns)

        # Store session state for dependency chain
        self._deferred_sessions[session_id] = {
            'sub_requests': sub_reqs,
            'next_index': 1,  # index 0 is being queued now
            'id_base': base_id,
            'source_session_id': source_session_id,
            'template_index': template_index,
            'epoch': int(epoch),
            'admission_time_ns': arrival_ns,
            'source_arrival_time_ns': source_arrival_ns,
        }
        lifecycle = self._session_lifecycle.get(session_id)
        if lifecycle is None:
            lifecycle = {
                'session_id': session_id,
                'source_session_id': source_session_id,
                'template_index': template_index,
                'epoch': int(epoch),
                'planned_admission_index': None,
                'admission_index': None,
                'measurement_warmup': False,
                'measurement_target': False,
                'measurement_required': False,
                'measurement_role': 'unassigned',
                'source_arrival_time_ns': source_arrival_ns,
                'offered_time_ns': arrival_ns,
                'completion_time_ns': None,
                'e2e_ns': None,
            }
            self._session_lifecycle[session_id] = lifecycle
        lifecycle['admission_time_ns'] = arrival_ns
        lifecycle['admission_queue_wait_ns'] = max(
            0, arrival_ns - int(lifecycle['offered_time_ns']))
        lifecycle['status'] = 'pending'
        self._deferred_sessions[session_id].update({
            'offered_time_ns': lifecycle['offered_time_ns'],
            'admission_queue_wait_ns': lifecycle['admission_queue_wait_ns'],
        })

        # Queue the first sub-request
        first = sub_reqs[0]
        req_data = {
            'index': base_id,
            'input_toks': int(first['input_toks']),
            'output_toks': int(first['input_toks'] + first['output_toks']),
            'arrival_time_ns': arrival_ns,
            'session_id': session_id,
            'sub_request_index': 0,
            'wakekv_has_successor': len(sub_reqs) > 1,
            'prefix_reuse_toks': int(first.get('prefix_reuse_toks', 0)),
            'prefix_reuse_source': first.get('prefix_reuse_source'),
            'return_gap_type': 'session_start',
            'return_gap_source': 'session_start',
            'return_gap_ns': 0,
            'source_session_id': source_session_id,
            'session_template_index': template_index,
            'session_epoch': int(epoch),
            'session_offered_time_ns': lifecycle['offered_time_ns'],
            'session_admission_time_ns': arrival_ns,
            'session_admission_queue_wait_ns': (
                lifecycle['admission_queue_wait_ns']),
        }
        if enable_prefix_caching or self._enable_agentic_kv:
            req_data['input_hash_ids'] = first.get('input_tok_ids', [])
            req_data['output_hash_ids'] = first.get('output_tok_ids', [])
        if insert_sorted:
            self._insert_pending_sorted(req_data)
        else:
            self._pending_requests.append(req_data)
        self._request_to_session[base_id] = (session_id, 0)

        return len(sub_reqs)

    def _restore_capacity_state(self, scheduler):
        manager_state = getattr(
            self.agentic_kv_manager, "restore_capacity_state", None)
        if manager_state is not None:
            return tuple(manager_state(scheduler.instance_id))
        unreserved = getattr(
            self.agentic_kv_manager,
            "hbm_unreserved_per_rank_bytes",
            None,
        )
        return (
            int(scheduler.memory.npu_used),
            int(
                unreserved(scheduler.instance_id)
                if unreserved is not None else
                max(0, int(scheduler.memory.npu_mem)
                    - int(scheduler.memory.npu_used))
            ),
        )

    def _pd_pair(self, prefill_sched):
        """Return the one fixed decode destination for a prefill engine."""
        candidates = self._compatible_decode_schedulers(prefill_sched)
        if len(candidates) != 1:
            raise RuntimeError(
                "Strict agentic P/D admission requires exactly one same-node "
                "layout-compatible decode destination per prefill instance; "
                f"prefill={prefill_sched.instance_id}, candidates="
                f"{[candidate.instance_id for candidate in candidates]}. "
                "The Chakra prefill graph has one fixed shadow receiver, so "
                "multi-destination batches are not representable yet.")
        return (int(prefill_sched.instance_id),
                int(candidates[0].instance_id))

    def _validate_strict_pd_mapping(self):
        """Reject many-to-one P/D layouts unsupported by local reclaim."""
        decode_owner = {}
        for prefill in self.prefill_schedulers:
            candidates = self._compatible_decode_schedulers(prefill)
            if len(candidates) != 1:
                # Preserve the existing request-binding diagnostic for zero
                # or ambiguous destinations. This constructor check closes
                # only the distinct many-P-to-one-D liveness hole.
                continue
            decode_id = int(candidates[0].instance_id)
            prior_prefill = decode_owner.get(decode_id)
            if prior_prefill is not None:
                raise RuntimeError(
                    "Strict agentic P/D active-prefill reclamation requires "
                    "an injective P-to-D mapping; many-to-one ownership can "
                    "strand D-side KV outside the selected prefill queue: "
                    f"decode={decode_id}, prefills="
                    f"{[prior_prefill, int(prefill.instance_id)]}")
            decode_owner[decode_id] = int(prefill.instance_id)

    def _pd_pair_has_handoff(self, pair):
        prefill_id, decode_id = pair
        return any(
            int(handoff["prefill_scheduler"].instance_id) == prefill_id
            and int(handoff["decode_scheduler"].instance_id) == decode_id
            for handoff in self._pending_decode_handoffs.get(decode_id, ())
        )

    def _pd_pair_available(self, pair, request_id):
        owner = self._pd_admission_owner.get(pair)
        return (
            (owner is None or int(owner) == int(request_id))
            and not self._pd_pair_has_handoff(pair)
        )

    def _defer_pd_admission(self, req_data, pair):
        # Preserve the trace release timestamp for TTFT accounting, but retry
        # the actual tier operation at the later admission event. Without
        # this marker, a long FIFO wait would backdate prepare_request() behind
        # the manager's already-advanced logical frontier.
        req_data['_agentic_kv_capacity_deferred'] = True
        self._pending_requests.pop(self._pending_idx)
        self._pending_pd_admission_waits.append({
            "request": req_data,
            "pair": tuple(pair),
        })

    def _promote_pd_admission_waiters(self, current_time_ns):
        """Requeue at most one FIFO waiter for each newly free P/D pair."""
        current_time_ns = int(current_time_ns)
        still_waiting = []
        promoted_pairs = set()
        promoted = 0
        for pending in self._pending_pd_admission_waits:
            pair = tuple(pending["pair"])
            request = pending["request"]
            request_id = int(request["index"])
            if (pair in promoted_pairs
                    or not self._pd_pair_available(pair, request_id)):
                still_waiting.append(pending)
                continue
            trace_release_ns = int(request.setdefault(
                '_agentic_kv_release_time_ns',
                request['arrival_time_ns'],
            ))
            request['_agentic_kv_pd_pair_fifo_wait_ns'] = max(
                0, current_time_ns - trace_release_ns)
            self._insert_pending_sorted(request)
            promoted_pairs.add(pair)
            promoted += 1
        self._pending_pd_admission_waits = still_waiting
        return promoted

    def _release_pd_admission_owner(self, request_id):
        request_id = int(request_id)
        for pair, owner in list(self._pd_admission_owner.items()):
            if int(owner) == request_id:
                del self._pd_admission_owner[pair]

    def _snapshot_due_return_residencies(self, arrival_cutoff_ns):
        """Snapshot every due return before a later physical operation.

        A model callback may reveal strictly older arrivals only after ASTRA
        has reached the callback timestamp.  Their tier observations still
        belong to their trace release times, whereas any newly issued copy
        belongs to the already-reached physical frontier.  Snapshot all such
        returns first so preparing the first one at that later frontier cannot
        make a second, equally old snapshot appear to travel backward in time.
        """
        if self.agentic_kv_manager is None:
            return
        residency_key = '_agentic_kv_residency_at_return_snapshot'
        due = []
        for sequence, request in enumerate(
                self._pending_requests[self._pending_idx:]):
            if self._pending_time(request) > int(arrival_cutoff_ns):
                continue
            if (request.get('session_id') is None
                    or int(request.get('sub_request_index', 0)) <= 0
                    or residency_key in request):
                continue
            release_ns = int(request.setdefault(
                '_agentic_kv_release_time_ns',
                request['arrival_time_ns'],
            ))
            due.append((release_ns, sequence, request))
        for release_ns, _, request in sorted(due):
            request[residency_key] = (
                self.agentic_kv_manager.snapshot_return_residency(
                    request['session_id'], release_ns).value)

    def route_arrived_requests(
            self, current_time_ns, *, operation_time_ns=None):
        """Route requests that have arrived by ``current_time_ns``.

        Called at the start of each iteration in the main simulation loop.
        ``operation_time_ns`` may be a later, already-reached controller
        frontier.  In that case the former remains the arrival/LRU cutoff and
        the latter is used for new physical tier operations.  This distinction
        is required at model callbacks: events through ``callback - 1`` must
        precede the completion's free, but ASTRA cannot accept a transfer
        issued retroactively at ``callback - 1``.

        Returns the number of newly routed requests.
        """
        current_time_ns = int(current_time_ns)
        self._fill_backlog_slots(current_time_ns)
        if operation_time_ns is not None:
            operation_time_ns = int(operation_time_ns)
            if operation_time_ns < current_time_ns:
                raise ValueError(
                    "request operation time cannot precede its arrival "
                    f"cutoff: operation={operation_time_ns}, "
                    f"cutoff={current_time_ns}")
            if operation_time_ns > current_time_ns:
                self._snapshot_due_return_residencies(current_time_ns)
        physical_time_ns = (
            current_time_ns
            if operation_time_ns is None else operation_time_ns
        )
        if self.full_model_hbf_adapter is not None:
            self.drain_full_model_hbf_gpu_hbm_events()
        total_routed = 0
        while True:
            routed, promoted = self._route_arrived_requests_once(
                current_time_ns, operation_time_ns=operation_time_ns)
            total_routed += routed
            if not promoted:
                if self.full_model_hbf_adapter is not None:
                    self.flush_full_model_hbf_admissions(
                        physical_time_ns)
                return total_routed

    def _route_arrived_requests_once(
            self, current_time_ns, *, operation_time_ns=None):
        """Run one causal arrival pass and one P/D admission pass."""
        physical_time_ns = (
            int(current_time_ns)
            if operation_time_ns is None else int(operation_time_ns)
        )
        routed = 0
        if self.agentic_kv_manager is not None:
            self._promote_pd_admission_waiters(current_time_ns)
            still_capacity_blocked = []
            for pending in self._pending_capacity_preparations:
                scheduler = self._scheduler_by_instance(
                    pending['instance_id'])
                state = self._restore_capacity_state(scheduler)
                retry_time_ns = pending.get('retry_time_ns')
                retry_due = (
                    retry_time_ns is not None
                    and int(retry_time_ns) <= physical_time_ns
                )
                if not retry_due and state == pending['last_state']:
                    still_capacity_blocked.append(pending)
                    continue
                pending['request']['ready_time_ns'] = physical_time_ns
                self._insert_pending_sorted(pending['request'])
            self._pending_capacity_preparations = still_capacity_blocked

            still_waiting = []
            for pending in self._pending_sync_preparations:
                instance_ids = pending['instance_ids']
                boundary_busy = getattr(
                    self.agentic_kv_manager,
                    'prepare_boundary_busy',
                    None,
                )
                is_busy = (
                    boundary_busy(instance_ids)
                    if boundary_busy is not None else any(
                        self._scheduler_by_instance(instance_id).inflight
                        for instance_id in instance_ids
                    )
                )
                if is_busy:
                    still_waiting.append(pending)
                    continue
                request_id = int(pending['request']['index'])
                pending['request']['_agentic_kv_prepare_start_ns'] = int(
                    physical_time_ns)
                pending['request']['_agentic_kv_prepare_lock_held'] = True
                self._insert_pending_sorted(pending['request'])
            self._pending_sync_preparations = still_waiting
        while self._pending_idx < len(self._pending_requests):
            req_data = self._pending_requests[self._pending_idx]
            if self._pending_time(req_data) > current_time_ns:
                break

            session_id = req_data.get('session_id')
            if (session_id is not None
                    and int(req_data.get('sub_request_index', 0)) == 0):
                self._activate_session(
                    session_id, int(req_data['arrival_time_ns']))
            full_model_hbf_decision = None
            if (
                self.full_model_hbf_adapter is not None
                and session_id is not None
            ):
                full_model_hbf_decision = (
                    self.full_model_hbf_adapter.offer_raw_request(
                        req_data, now_ns=physical_time_ns))
                # A GPU fallback can release or claim an idle allocation.
                # Apply that ownership transition before selecting or
                # constructing the native Scheduler Request.
                self.drain_full_model_hbf_gpu_hbm_events()
                if full_model_hbf_decision.divert_to_hbf:
                    self._pending_requests.pop(self._pending_idx)
                    routed += 1
                    continue
                if not full_model_hbf_decision.run_on_gpu:
                    raise RuntimeError(
                        "full-model HBF route is neither HBF nor native GPU")
                req_data = (
                    self.full_model_hbf_adapter.decorate_gpu_metadata(
                        full_model_hbf_decision, req_data))
                self._pending_requests[self._pending_idx] = req_data

            sched = None
            required_gpu_instance_id = (
                req_data.get('hbf_gpu_required_instance_id')
                if full_model_hbf_decision is not None else None
            )
            if required_gpu_instance_id is not None:
                required_gpu_instance_id = int(required_gpu_instance_id)
                sched = self._scheduler_by_instance(
                    required_gpu_instance_id)
                if sched not in self.prefill_schedulers:
                    raise RuntimeError(
                        "full-model HBF native GPU target is not a "
                        "prefill-capable Scheduler: "
                        f"instance={required_gpu_instance_id}")
            fixed_prefill_id = req_data.get('_pd_prefill_instance_id')
            if fixed_prefill_id is not None:
                fixed_sched = self._scheduler_by_instance(
                    int(fixed_prefill_id))
                if (
                    sched is not None
                    and sched.instance_id != fixed_sched.instance_id
                ):
                    raise RuntimeError(
                        "full-model HBF GPU target conflicts with fixed "
                        "P/D prefill target: "
                        f"gpu={sched.instance_id}, "
                        f"prefill={fixed_sched.instance_id}")
                sched = fixed_sched
            if session_id is not None and session_id in self._session_affinity:
                affinity = self._session_affinity[session_id]
                affinity_sched = next(
                    (candidate for candidate in self.prefill_schedulers
                     if candidate.instance_id == affinity), None)
                if (sched is not None and affinity_sched is not None
                        and sched.instance_id != affinity_sched.instance_id):
                    raise RuntimeError(
                        "Deferred P/D admission changed prefill affinity: "
                        f"request={req_data['index']}, "
                        f"deferred={sched.instance_id}, "
                        f"affinity={affinity_sched.instance_id}")
                if sched is None:
                    sched = affinity_sched
            if sched is None:
                selected = self._select_instance(self.prefill_schedulers, "prefill")
                sched = self.prefill_schedulers[selected]

            full_model_hbf_bridge_binding = None
            full_model_hbf_is_continuation = (
                full_model_hbf_decision is not None
                and int(req_data.get('sub_request_index', 0)) > 0
            )
            if full_model_hbf_decision is not None:
                bridge = self.full_model_hbf_gpu_hbm_bridge
                if bridge.topology == "pd":
                    decode_instance_id = (
                        self._full_model_hbf_pd_decode_by_prefill.get(
                            int(sched.instance_id)))
                    if decode_instance_id is None:
                        raise RuntimeError(
                            "full-model HBF selected a prefill Scheduler "
                            "without a configured decode pair: "
                            f"prefill={sched.instance_id}")
                    prior_decode = self._session_decode_affinity.get(
                        session_id)
                    if (
                        prior_decode is not None
                        and int(prior_decode) != decode_instance_id
                    ):
                        raise RuntimeError(
                            "full-model HBF P/D continuation changed decode "
                            f"affinity: session={session_id!r}, "
                            f"old={prior_decode}, new={decode_instance_id}")
                    self._session_decode_affinity[
                        session_id] = decode_instance_id
                    if full_model_hbf_is_continuation:
                        if not full_model_hbf_decision.force_gpu_recompute:
                            raise RuntimeError(
                                "full-model HBF P/D continuation must use "
                                "explicit GPU recomputation")
                        req_data = bridge.decorate_pd_recompute(
                            int(req_data['index']),
                            req_data,
                            prefill_instance_id=int(sched.instance_id),
                            decode_instance_id=decode_instance_id,
                        )
                        self._pending_requests[
                            self._pending_idx] = req_data
                        full_model_hbf_bridge_binding = "pd_recompute"
                elif (
                    full_model_hbf_is_continuation
                    and required_gpu_instance_id is not None
                ):
                    req_data = bridge.decorate_colocated_continuation(
                        int(req_data['index']), req_data)
                    self._pending_requests[self._pending_idx] = req_data
                    full_model_hbf_bridge_binding = "colocated_retained"

            strict_pd_admission = (
                self.agentic_kv_manager is not None
                and sched.pd_type == "prefill"
                and bool(self.decode_schedulers)
            )
            full_model_hbf_pd_admission = (
                full_model_hbf_decision is not None
                and self.full_model_hbf_gpu_hbm_bridge.topology == "pd"
            )
            is_agentic_continuation = (
                self.agentic_kv_manager is not None
                and session_id is not None
                and int(req_data.get('sub_request_index', 0)) > 0
            )
            return_residency_key = (
                '_agentic_kv_residency_at_return_snapshot')
            if (is_agentic_continuation
                    and return_residency_key not in req_data):
                trace_release_ns = int(req_data.setdefault(
                    '_agentic_kv_release_time_ns',
                    req_data['arrival_time_ns'],
                ))
                req_data[return_residency_key] = (
                    self.agentic_kv_manager.snapshot_return_residency(
                        session_id, trace_release_ns).value)
            pd_pair = None
            if strict_pd_admission:
                pd_pair = self._pd_pair(sched)
                req_data['_pd_prefill_instance_id'] = sched.instance_id
                request_id = int(req_data['index'])
                if not self._pd_pair_available(pd_pair, request_id):
                    self._defer_pd_admission(req_data, pd_pair)
                    continue
                self._pd_admission_owner.setdefault(pd_pair, request_id)

            if (self.agentic_kv_manager is not None and session_id is not None
                    and int(req_data.get('sub_request_index', 0)) > 0
                    and not req_data.get('_agentic_kv_prepared', False)):
                boundary_resolver = getattr(
                    self.agentic_kv_manager,
                    'prepare_boundary_instances',
                    self.agentic_kv_manager.synchronous_prepare_instances,
                )
                boundary_instances = (
                    boundary_resolver(
                        session_id,
                        sched.instance_id,
                        int(req_data.get('prefix_reuse_toks', 0)),
                        physical_time_ns,
                    )
                )
                if boundary_instances:
                    req_data['_agentic_kv_prepare_start_ns'] = int(
                        physical_time_ns)
                boundary_busy = getattr(
                    self.agentic_kv_manager,
                    'prepare_boundary_busy',
                    None,
                )
                is_boundary_busy = (
                    boundary_busy(boundary_instances)
                    if boundary_instances and boundary_busy is not None
                    else bool(boundary_instances) and any(
                        self._scheduler_by_instance(instance_id).inflight
                        for instance_id in boundary_instances
                    )
                )
                if is_boundary_busy:
                    request_id = int(req_data['index'])
                    acquire_lock = getattr(
                        self.agentic_kv_manager,
                        'acquire_prepare_lock',
                        self.agentic_kv_manager.acquire_synchronous_prepare_lock,
                    )
                    acquire_lock(
                        request_id, boundary_instances, session_id=session_id)
                    self._pending_requests.pop(self._pending_idx)
                    self._pending_sync_preparations.append({
                        'request': req_data,
                        'instance_ids': boundary_instances,
                    })
                    continue
                try:
                    trace_release_ns = int(req_data.setdefault(
                        '_agentic_kv_release_time_ns',
                        req_data['arrival_time_ns'],
                    ))
                    logical_release_ns = int(req_data.get(
                        '_agentic_kv_prepare_start_ns', trace_release_ns))
                    capacity_retry = bool(req_data.get(
                        '_agentic_kv_capacity_deferred', False))
                    prepare_operation_time_ns = (
                        physical_time_ns
                        if (operation_time_ns is not None
                            or capacity_retry
                            or '_agentic_kv_prepare_start_ns' in req_data
                        )
                        else logical_release_ns
                    )
                    # Request latency always starts at the trace release. A
                    # scheduler/engine-boundary delay before the first manager
                    # attempt is reported explicitly instead of shifting that
                    # release timestamp or relabeling it as physical restore.
                    prepare_release_ns = trace_release_ns
                    pd_pair_fifo_wait_ns = int(req_data.get(
                        '_agentic_kv_pd_pair_fifo_wait_ns', 0))
                    boundary_wait_key = (
                        '_agentic_kv_prepare_boundary_wait_ns')
                    if boundary_wait_key not in req_data:
                        req_data[boundary_wait_key] = max(
                            0,
                            prepare_operation_time_ns - trace_release_ns
                            - pd_pair_fifo_wait_ns,
                        )
                    prepare_boundary_wait_ns = int(
                        req_data[boundary_wait_key])
                    prep = self.agentic_kv_manager.prepare_request(
                        session_id=session_id,
                        instance_id=sched.instance_id,
                        reuse_tokens=int(req_data.get(
                            'prefix_reuse_toks', 0)),
                        input_tokens=int(req_data['input_toks']),
                        release_time_ns=prepare_release_ns,
                        return_gap_type=str(
                            req_data.get('return_gap_type') or 'unknown'),
                        return_gap_source=str(
                            req_data.get('return_gap_source') or 'unknown'),
                        return_gap_ns=int(
                            req_data.get('return_gap_ns') or 0),
                        operation_time_ns=prepare_operation_time_ns,
                        pd_pair_fifo_wait_ns=pd_pair_fifo_wait_ns,
                        prepare_boundary_wait_ns=(
                            prepare_boundary_wait_ns),
                        request_id=int(req_data['index']),
                        sub_request_index=int(
                            req_data.get('sub_request_index', 0)),
                        pd_decode_instance_id=(
                            int(pd_pair[1])
                            if pd_pair is not None else None),
                        defer_temporary_hbm_pressure=True,
                        residency_at_return=req_data[return_residency_key],
                    )
                finally:
                    if req_data.pop(
                            '_agentic_kv_prepare_lock_held', False):
                        release_lock = getattr(
                            self.agentic_kv_manager,
                            'release_prepare_lock',
                            self.agentic_kv_manager
                            .release_synchronous_prepare_lock,
                        )
                        release_lock(int(req_data['index']))
                if prep is None:
                    # Destination HBM is temporarily occupied by active work.
                    # Keep the lower-tier source pinned, let other ready rows
                    # pass, and retry only after target capacity changes.
                    retry_time = getattr(
                        self.agentic_kv_manager,
                        'pending_prepare_retry_time',
                        lambda _session_id: None,
                    )(session_id)
                    req_data['_agentic_kv_capacity_deferred'] = True
                    self._pending_requests.pop(self._pending_idx)
                    self._pending_capacity_preparations.append({
                        'request': req_data,
                        'instance_id': sched.instance_id,
                        'last_state': self._restore_capacity_state(sched),
                        'retry_time_ns': retry_time,
                    })
                    continue
                req_data.pop('_agentic_kv_capacity_deferred', None)
                req_data['_agentic_kv_prepared'] = True
                restore_issue_ns = int(
                    prep.restore_issue_time_ns
                    or prepare_release_ns
                )
                restore_ready_ns = int(
                    prep.restore_ready_time_ns or prep.ready_time_ns)
                target_hbm_ready_ns = int(
                    prep.target_hbm_ready_time_ns
                    or restore_issue_ns + prep.hbm_admission_wait_ns)
                fresh_prompt_tokens = max(
                    0, int(req_data['input_toks']) - int(prep.hit_tokens))
                async_decode_join = bool(getattr(
                    self.agentic_kv_manager,
                    'async_decode_join_enabled',
                    False,
                ))
                pre_restore_prompt_tokens = max(
                    0,
                    int(req_data['input_toks'])
                    - 1
                    - int(prep.hit_tokens),
                )
                if (async_decode_join and prep.restore_ns > 0
                        and pre_restore_prompt_tokens > 0):
                    req_data['ready_time_ns'] = target_hbm_ready_ns
                    overlap_cutoff_tokens = max(
                        int(prep.hit_tokens),
                        int(req_data['input_toks']) - 1,
                    )
                else:
                    req_data['ready_time_ns'] = restore_ready_ns
                    overlap_cutoff_tokens = None
                initial_restore_gate_wait_ns = (
                    max(0, restore_ready_ns - physical_time_ns)
                    if (async_decode_join and prep.restore_ns > 0
                        and pre_restore_prompt_tokens == 0
                        and not strict_pd_admission)
                    else 0
                )
                req_data['agentic_kv_hit_tokens'] = prep.hit_tokens
                req_data['agentic_kv_recompute_tokens'] = prep.recompute_tokens
                req_data['agentic_kv_residency_at_return'] = (
                    prep.residency_at_return.value)
                req_data['agentic_kv_source'] = prep.source.value
                req_data['agentic_kv_restore_ns'] = prep.restore_ns
                req_data['agentic_kv_owner_gate_ns'] = prep.owner_gate_ns
                req_data['pd_pair_fifo_wait_ns'] = (
                    prep.pd_pair_fifo_wait_ns)
                req_data['agentic_kv_prepare_boundary_wait_ns'] = (
                    prep.prepare_boundary_wait_ns)
                req_data['agentic_kv_restore_issue_time_ns'] = (
                    restore_issue_ns)
                req_data['agentic_kv_target_hbm_ready_time_ns'] = (
                    target_hbm_ready_ns)
                req_data['agentic_kv_restore_ready_time_ns'] = (
                    restore_ready_ns)
                req_data['agentic_kv_fresh_prompt_tokens'] = (
                    fresh_prompt_tokens)
                req_data['agentic_kv_overlap_cutoff_tokens'] = (
                    overlap_cutoff_tokens)
                req_data['agentic_kv_async_decode_join'] = (
                    async_decode_join)
                req_data['agentic_kv_restore_gate_start_ns'] = (
                    physical_time_ns
                    if initial_restore_gate_wait_ns else 0)
                req_data['agentic_kv_restore_gate_wait_ns'] = (
                    initial_restore_gate_wait_ns)
                req_data['agentic_kv_hbm_admission_wait_ns'] = (
                    prep.hbm_admission_wait_ns)
                req_data['agentic_kv_source_demotion_join_wait_ns'] = (
                    prep.source_demotion_join_wait_ns)
                req_data['agentic_kv_transient_dram_capacity_wait_ns'] = (
                    prep.transient_dram_capacity_wait_ns)
                req_data['agentic_kv_restore_queue_wait_ns'] = (
                    prep.queue_wait_ns)
                req_data['agentic_kv_restore_service_ns'] = prep.service_ns
                req_data['agentic_kv_owner_instance_id'] = (
                    sched.instance_id if prep.hit_tokens > 0 else None)
                req_data['agentic_kv_retained_instance_id'] = (
                    prep.retained_instance_id)
                req_data['agentic_kv_retained_per_rank_bytes'] = (
                    prep.retained_per_rank_bytes)
                if (prep.ready_time_ns > physical_time_ns
                        and not strict_pd_admission):
                    # Restore completion is a new eligibility event.  Reinsert
                    # the row so other sessions that are ready now can pass it.
                    self._pending_requests.pop(self._pending_idx)
                    self._insert_pending_sorted(req_data)
                    continue

            if strict_pd_admission and not req_data.get(
                    '_agentic_kv_prepared', False):
                trace_release_ns = int(req_data.setdefault(
                    '_agentic_kv_release_time_ns',
                    req_data['arrival_time_ns'],
                ))
                pair_wait_ns = int(req_data.get(
                    '_agentic_kv_pd_pair_fifo_wait_ns', 0))
                owner_ready_ns = trace_release_ns + pair_wait_ns
                req_data['pd_pair_fifo_wait_ns'] = pair_wait_ns
                req_data['agentic_kv_prepare_boundary_wait_ns'] = 0
                req_data['agentic_kv_owner_gate_ns'] = pair_wait_ns
                req_data['agentic_kv_restore_ns'] = 0
                req_data['agentic_kv_restore_issue_time_ns'] = owner_ready_ns
                req_data['agentic_kv_target_hbm_ready_time_ns'] = (
                    owner_ready_ns)
                req_data['agentic_kv_restore_ready_time_ns'] = owner_ready_ns

            metadata = {
                key: req_data.get(key)
                for key in (
                    'session_id', 'sub_request_index', 'ready_time_ns',
                    'source_session_id', 'session_template_index',
                    'session_epoch', 'session_offered_time_ns',
                    'session_admission_time_ns',
                    'session_admission_queue_wait_ns',
                    'prefix_reuse_toks', 'prefix_reuse_source',
                    'return_gap_type', 'return_gap_source', 'return_gap_ns',
                    'agentic_kv_hit_tokens', 'agentic_kv_recompute_tokens',
                    'agentic_kv_residency_at_return',
                    'agentic_kv_source', 'agentic_kv_restore_ns',
                    'agentic_kv_owner_gate_ns',
                    'agentic_kv_restore_issue_time_ns',
                    'agentic_kv_target_hbm_ready_time_ns',
                    'agentic_kv_restore_ready_time_ns',
                    'agentic_kv_fresh_prompt_tokens',
                    'agentic_kv_overlap_cutoff_tokens',
                    'agentic_kv_async_decode_join',
                    'agentic_kv_restore_gate_start_ns',
                    'agentic_kv_restore_gate_wait_ns',
                    'pd_pair_fifo_wait_ns',
                    'agentic_kv_prepare_boundary_wait_ns',
                    'agentic_kv_source_demotion_join_wait_ns',
                    'agentic_kv_hbm_admission_wait_ns',
                    'agentic_kv_transient_dram_capacity_wait_ns',
                    'agentic_kv_restore_queue_wait_ns',
                    'agentic_kv_restore_service_ns',
                    'agentic_kv_owner_instance_id',
                    'agentic_kv_retained_instance_id',
                    'agentic_kv_retained_per_rank_bytes',
                    'hbf_online_execution',
                    'hbf_online_route_reason',
                )
            }
            if sched.enable_prefix_caching or self._enable_agentic_kv:
                request_values = [
                    req_data['index'], sched.model,
                    req_data['input_toks'], req_data['output_toks'],
                    req_data['arrival_time_ns'], sched.instance_id,
                    req_data.get('input_hash_ids', []), req_data.get('output_hash_ids', []),
                ]
            else:
                request_values = [
                    req_data['index'], sched.model,
                    req_data['input_toks'], req_data['output_toks'],
                    req_data['arrival_time_ns'], sched.instance_id,
                ]

            if strict_pd_admission:
                new_req = sched.add_request(
                    request_values,
                    is_init=self._is_init,
                    metadata=metadata,
                    enqueue=False,
                )
                self._stage_pd_receive_admission(
                    new_req, sched, physical_time_ns)
                self.agentic_kv_manager.record_agentic_request(new_req)
                # This row is now owned by the P/D admission state machine.
                self._pending_requests.pop(self._pending_idx)
            elif full_model_hbf_pd_admission:
                new_req = sched.add_request(
                    request_values,
                    is_init=self._is_init,
                    metadata=metadata,
                    enqueue=False,
                )
                decode_instance_id = (
                    self._full_model_hbf_pd_decode_by_prefill[
                        int(sched.instance_id)]
                )
                reserved = False
                if not self._pending_full_model_hbf_prefill_launches:
                    reserved = (
                        self.full_model_hbf_gpu_hbm_bridge
                        .try_reserve_pd_decode(
                            new_req,
                            prefill_instance_id=int(sched.instance_id),
                            decode_instance_id=decode_instance_id,
                        )
                    )
                self._pending_requests.pop(self._pending_idx)
                if reserved:
                    sched.enqueue_request(new_req)
                else:
                    self._pending_full_model_hbf_prefill_launches.append({
                        "request": new_req,
                        "prefill_scheduler": sched,
                        "decode_instance_id": decode_instance_id,
                        "enqueued_ns": int(physical_time_ns),
                    })
            else:
                new_req = sched.add_request(
                    request_values,
                    is_init=self._is_init,
                    metadata=metadata,
                )
                if self.agentic_kv_manager is not None:
                    self.agentic_kv_manager.record_agentic_request(new_req)

            if full_model_hbf_decision is not None:
                if full_model_hbf_bridge_binding == "pd_recompute":
                    # The adapter remains deliberately unbound while this
                    # Request is on P. Its native owner is the D Scheduler
                    # observed at final completion, after normal handoff.
                    self.full_model_hbf_gpu_hbm_bridge.bind_pd_recompute(
                        new_req)
                elif (
                    self.full_model_hbf_gpu_hbm_bridge.topology
                    == "colocated"
                ):
                    self.full_model_hbf_adapter.bind_native_gpu_request(
                        new_req)
                    if (
                        full_model_hbf_bridge_binding
                        == "colocated_retained"
                    ):
                        bridge.bind_colocated_continuation(new_req)

            # Hold routing constant across off/prefix-only/recompute/swap
            # baselines so policy comparisons do not conflate cache locality
            # with a different instance assignment.
            if session_id is not None:
                self._session_affinity[session_id] = sched.instance_id

            if strict_pd_admission:
                continue
            if full_model_hbf_pd_admission:
                routed += 1
                continue
            self._pending_idx += 1
            routed += 1

        promoted = 0
        if self.agentic_kv_manager is not None:
            routed += self.process_pending_decode_handoffs(
                physical_time_ns)
            # process_pending_decode_handoffs() advances manager time to this
            # callback, so only pair waiters explicitly marked as delayed may
            # prepare afterward. The public wrapper iteratively drains those
            # immediate admissions at the same timestamp; a non-fitting head
            # retains ownership and ends the loop without polling.
            promoted = self._promote_pd_admission_waiters(
                physical_time_ns)
        elif self.full_model_hbf_adapter is not None:
            self.process_pending_decode_handoffs(
                physical_time_ns)
        return routed, promoted

    def has_pending_requests(self):
        """Check if there are unrouted requests remaining."""
        return (
            self._pending_idx < len(self._pending_requests)
            or bool(self._pending_sync_preparations)
            or bool(self._pending_capacity_preparations)
            or bool(self._pending_pd_admission_waits)
        )

    def get_first_arrival_time(self):
        """Return the first request's arrival time in ns, or 1 if no requests."""
        if self._pending_requests:
            return max(1, self._pending_requests[0]['arrival_time_ns'])
        return 1

    # -----------------------------------------------------------------------
    # Agentic dependency chain management
    # -----------------------------------------------------------------------

    def notify_request_completed(self, request, completion_time_ns):
        """Called when a request finishes. Releases the next sub-request in
        the session chain after the tool_call duration elapses.

        For flat requests (not in a session), this is a no-op.
        """
        request_id = request.id if hasattr(request, 'id') else request
        session_info = self._request_to_session.get(request_id)
        if session_info is None:
            return
        session_id, completed_idx = session_info
        session = self._deferred_sessions.get(session_id)
        if session is None:
            return

        sub_reqs = session['sub_requests']
        next_idx = session['next_index']
        base_id = session['id_base']

        # A measurement freeze is a source barrier. Already dispatched final
        # requests must still close their logical sessions, but a non-final
        # callback must not create a successor on the far side of the cutoff.
        # Keep the mapping intact until audited censoring clears it.
        if self._session_admission_frozen and next_idx < len(sub_reqs):
            return
        self._request_to_session.pop(request_id)

        # Get tool duration from the completed sub-request
        tool_duration_ns = int(sub_reqs[completed_idx].get('tool_duration_ns', 0))
        release_time_ns = completion_time_ns + tool_duration_ns

        if next_idx < len(sub_reqs):
            # Release next sub-request
            next_sub = sub_reqs[next_idx]
            next_id = base_id + next_idx
            reuse_tokens, reuse_source = self._prefix_reuse(
                sub_reqs[completed_idx], next_sub, request)
            req_data = {
                'index': next_id,
                'input_toks': int(next_sub['input_toks']),
                'output_toks': int(next_sub['input_toks'] + next_sub['output_toks']),
                'arrival_time_ns': release_time_ns,
                'session_id': session_id,
                'sub_request_index': next_idx,
                'wakekv_has_successor': next_idx + 1 < len(sub_reqs),
                'prefix_reuse_toks': reuse_tokens,
                'prefix_reuse_source': reuse_source,
                # TraceLab records the N -> N+1 pause on sub-request N.
                # Copy it forward so this request carries its incoming return
                # class instead of its own outgoing pause.
                'return_gap_type': self._return_gap_type(
                    sub_reqs[completed_idx]),
                'return_gap_source': str(
                    sub_reqs[completed_idx].get('tool_wait_source')
                    or 'unknown'),
                'return_gap_ns': tool_duration_ns,
                'source_session_id': session['source_session_id'],
                'session_template_index': session['template_index'],
                'session_epoch': session['epoch'],
                'session_offered_time_ns': session['offered_time_ns'],
                'session_admission_time_ns': session['admission_time_ns'],
                'session_admission_queue_wait_ns': (
                    session['admission_queue_wait_ns']),
            }
            if self._enable_prefix_caching or self._enable_agentic_kv:
                req_data['input_hash_ids'] = next_sub.get('input_tok_ids', [])
                req_data['output_hash_ids'] = next_sub.get('output_tok_ids', [])
            if self.agentic_kv_manager is not None:
                if not hasattr(request, 'session_id'):
                    raise RuntimeError(
                        "Agentic KV tiering requires the completed Request object, not only its id")
                self.agentic_kv_manager.on_idle_start(
                    request,
                    completion_time_ns,
                    release_time_ns,
                    return_gap_type=req_data['return_gap_type'],
                    return_gap_source=req_data['return_gap_source'],
                )
            # Insert in sorted position after _pending_idx
            self._insert_pending_sorted(req_data)
            self._request_to_session[next_id] = (session_id, next_idx)
            session['next_index'] = next_idx + 1
        else:
            # Session complete — all sub-requests have been released
            del self._deferred_sessions[session_id]
            self._session_affinity.pop(session_id, None)
            self._session_decode_affinity.pop(session_id, None)
            if self.agentic_kv_manager is not None:
                self.agentic_kv_manager.end_session(
                    session_id, now_ns=completion_time_ns)
            self._complete_session(session_id, int(completion_time_ns))
            # end_session() above releases the old session's physical KV
            # before a replacement is admitted at this same logical time.
            self._fill_backlog_slots(int(completion_time_ns))

    def request_would_complete_session(self, request):
        """Return whether ``request`` is the final call of its session."""
        return self._request_final_session_id(request) is not None

    def _request_final_session_id(self, request):
        """Return the session ID if ``request`` is its final call."""
        request_id = request.id if hasattr(request, 'id') else request
        session_info = self._request_to_session.get(request_id)
        if session_info is None:
            return None
        session_id, _ = session_info
        session = self._deferred_sessions.get(session_id)
        if session is None:
            return None
        if int(session['next_index']) < len(session['sub_requests']):
            return None
        return str(session_id)

    def measurement_target_session_ids(self):
        """Return the ordered fixed cohort, or an empty tuple for legacy mode."""
        return tuple(self._measurement_target_session_ids)

    def measurement_warmup_session_ids(self):
        """Return the fixed admission-prefix warmup IDs in admission order."""
        return tuple(self._measurement_warmup_session_ids)

    def measurement_required_session_ids(self):
        """Return every fixed-prefix ID that must finish before cutoff."""
        return tuple(self._measurement_required_session_ids)

    def measurement_target_reached(self):
        """Return whether the configured measurement boundary is complete.

        For fixed admission order this requires both the excluded warmup
        prefix and measured target, even when target sessions finish first.
        """
        if (self.session_admission.measurement_cohort_selection
                == 'admission_order'):
            return (
                bool(self._measurement_required_session_id_set)
                and self._measurement_required_session_id_set.issubset(
                    self._completed_session_ids)
            )
        target = (
            int(self.session_admission.warmup_completions)
            + int(self.session_admission.measure_completions)
        )
        return target > 0 and self._sessions_completed >= target

    def measurement_boundary_would_be_reached(self, requests):
        """Predict the measurement boundary before completion notification.

        The pre-notification decision is required in backlog mode: freezing
        first prevents the final target completion from admitting a replacement
        at the same logical timestamp. Completion-order mode intentionally
        retains the historical count-based behavior.
        """
        requests = tuple(requests)
        if (self.session_admission.measurement_cohort_selection
                == 'completion_order'):
            final_calls = sum(
                self.request_would_complete_session(request)
                for request in requests
            )
            target = (
                int(self.session_admission.warmup_completions)
                + int(self.session_admission.measure_completions)
            )
            return (
                target > 0
                and final_calls > 0
                and self._sessions_completed + final_calls >= target
            )

        if not self._measurement_required_session_id_set:
            return False
        incomplete = (
            self._measurement_required_session_id_set
            - self._completed_session_ids
        )
        if not incomplete:
            return False
        completing = {
            session_id
            for request in requests
            for session_id in [self._request_final_session_id(request)]
            if session_id in self._measurement_required_session_id_set
        }
        return incomplete.issubset(completing)

    def freeze_session_admission(self):
        """Prevent post-measurement continuations and backlog replacement."""
        self._session_admission_frozen = True
        self._censored_pending_prepare_rows += len(
            self._pending_sync_preparations)
        self._censored_pending_prepare_rows += len(
            self._pending_capacity_preparations)
        self._censored_pending_prepare_rows += len(
            self._pending_pd_admission_waits)
        if self.agentic_kv_manager is not None:
            for pending in self._pending_sync_preparations:
                request_id = int(pending['request']['index'])
                release = getattr(
                    self.agentic_kv_manager,
                    'release_prepare_lock',
                    self.agentic_kv_manager.release_synchronous_prepare_lock,
                )
                release(request_id)
                self._release_pd_admission_owner(request_id)
            self._frozen_sync_preparations.extend(
                self._pending_sync_preparations)
        self._pending_sync_preparations.clear()

    def _censor_pending_pd_handoffs(self, cutoff_time_ns):
        """Cancel pre-admission P/D claims and release prepared KV exactly."""
        if self.agentic_kv_manager is None:
            return []
        cutoff_time_ns = int(cutoff_time_ns)
        audits = []
        for decode_id in sorted(self._pending_decode_handoffs):
            queue = self._pending_decode_handoffs[decode_id]
            for handoff in queue:
                request = handoff["request"]
                request_id = int(request.id)
                for role, scheduler, needed_key, ready_key in (
                        (
                            "prefill",
                            handoff["prefill_scheduler"],
                            "prefill_needed_per_rank_bytes",
                            "prefill_claim_ready_ns",
                        ),
                        (
                            "decode",
                            handoff["decode_scheduler"],
                            "decode_needed_per_rank_bytes",
                            "decode_claim_ready_ns",
                        )):
                    claim = (
                        self.agentic_kv_manager.active_hbm_reclaim_claim(
                            scheduler.instance_id)
                    )
                    if claim is None:
                        if (int(handoff[needed_key]) > 0
                                and handoff[ready_key] is not None):
                            raise RuntimeError(
                                f"Censored P/D {role} admission lost its "
                                "active HBM claim: "
                                f"request={request_id}, "
                                f"instance={scheduler.instance_id}")
                        continue
                    owner = (claim.owner_kind, claim.owner_id)
                    expected_owner = ("pd", request_id)
                    if owner != expected_owner:
                        # A scheduler-owned claim can precede this handoff.
                        # It is outside router ownership and must not be
                        # cancelled merely because measurement stopped.
                        if handoff[ready_key] is not None:
                            raise RuntimeError(
                                f"Censored P/D {role} claim owner changed: "
                                f"request={request_id}, expected="
                                f"{expected_owner}, observed={owner}")
                        continue
                    cancelled = (
                        self.agentic_kv_manager.cancel_active_hbm_reclaim(
                            scheduler.instance_id, cutoff_time_ns)
                    )
                    if (cancelled is None
                            or (cancelled.owner_kind, cancelled.owner_id)
                            != expected_owner):
                        raise RuntimeError(
                            f"Failed to cancel exact P/D {role} claim: "
                            f"request={request_id}, "
                            f"instance={scheduler.instance_id}")
                    scheduler.decode_handoff_claim_pending = False
                self.agentic_kv_manager.release_synchronous_prepare_lock(
                    request_id)
                audit = self.agentic_kv_manager.censor_prepared_request(
                    request, cutoff_time_ns)
                audit["prefill_instance_id"] = int(
                    handoff["prefill_scheduler"].instance_id)
                audit["decode_instance_id"] = int(
                    handoff["decode_scheduler"].instance_id)
                audits.append(audit)
                self._release_pd_admission_owner(request_id)
        self._pending_decode_handoffs.clear()
        for pair in sorted(self._pending_pd_chunk_admissions):
            for handoff in self._pending_pd_chunk_admissions[pair]:
                request = handoff["request"]
                request_id = int(request.id)
                for role, scheduler, needed_key, ready_key in (
                        self._pd_handoff_claim_roles(handoff)):
                    claim = self.agentic_kv_manager.active_hbm_reclaim_claim(
                        scheduler.instance_id)
                    expected = ("pd", request_id)
                    if claim is None:
                        if (int(handoff[needed_key]) > 0
                                and handoff[ready_key] is not None):
                            raise RuntimeError(
                                f"Censored P/D {role} chunk lost its claim: "
                                f"request={request_id}, instance="
                                f"{scheduler.instance_id}")
                        continue
                    if (claim.owner_kind, claim.owner_id) != expected:
                        if handoff[ready_key] is not None:
                            raise RuntimeError(
                                f"Censored P/D {role} chunk owner changed: "
                                f"request={request_id}, observed="
                                f"{(claim.owner_kind, claim.owner_id)}")
                        continue
                    cancelled = self.agentic_kv_manager.cancel_active_hbm_reclaim(
                        scheduler.instance_id, cutoff_time_ns)
                    if (cancelled is None
                            or (cancelled.owner_kind, cancelled.owner_id)
                            != expected):
                        raise RuntimeError(
                            f"Failed to cancel P/D {role} chunk claim for "
                            f"request={request_id}")
                    scheduler.decode_handoff_claim_pending = False
                request.pd_chunk_claim_pending = False
                self.agentic_kv_manager.release_synchronous_prepare_lock(
                    request_id)
                audits.append({
                    "request_id": request_id,
                    "session_id": request.session_id,
                    "prefill_instance_id": int(pair[0]),
                    "decode_instance_id": int(pair[1]),
                    "pending_chunk_claim_cancelled": True,
                })
        self._pending_pd_chunk_admissions.clear()
        return audits

    def _censor_incremental_pd_request(
            self, request, prefill_scheduler, decode_scheduler, now_ns):
        """Release current incremental P/D ownership without full assumptions."""
        if request.pd_kv_ownership_state != "prefill_active":
            raise RuntimeError(
                "Incremental P/D censoring found the wrong lifecycle state: "
                f"request={request.id}, state="
                f"{request.pd_kv_ownership_state}")
        initial_p = int(
            request.pd_prefill_initial_restored_per_rank_bytes)
        fresh_p = int(request.pd_prefill_reserved_per_rank_bytes)
        retained_d = int(request.agentic_kv_retained_per_rank_bytes)
        fresh_d = int(request.pd_decode_reserved_per_rank_bytes)
        current_p = int(request.pd_prefill_owned_per_rank_bytes)
        current_d = int(request.pd_decode_owned_per_rank_bytes)
        if initial_p + fresh_p != current_p:
            raise RuntimeError(
                "Censored incremental P ownership does not reconcile: "
                f"request={request.id}, initial={initial_p}, "
                f"fresh={fresh_p}, current={current_p}")
        if retained_d + fresh_d != current_d:
            raise RuntimeError(
                "Censored incremental D ownership does not reconcile: "
                f"request={request.id}, retained={retained_d}, "
                f"fresh={fresh_d}, current={current_d}")
        if fresh_p:
            prefill_scheduler.memory.free(fresh_p, Device.NPU)
        if fresh_d:
            decode_scheduler.memory.free(fresh_d, Device.NPU)

        # The manager owns only the restored and retained remainders. Restore
        # its expected ownership view before invoking the existing exact
        # prepared-request cleanup.
        request.agentic_kv_owner_instance_id = (
            prefill_scheduler.instance_id if initial_p else None)
        request.pd_prefill_preallocated_per_rank_bytes = 0
        request.pd_prefill_owned_per_rank_bytes = initial_p
        request.pd_decode_owned_per_rank_bytes = retained_d
        if int(request.pd_active_prefill_recompute_generation) > 0:
            if initial_p != 0 or retained_d != 0:
                raise RuntimeError(
                    "A preempted P/D prefill retained manager-owned base KV "
                    f"during censoring: request={request.id}, P={initial_p}, "
                    f"D={retained_d}")
            # Restore/source ownership was already complete before active
            # preemption was permitted. Only a keep-on-read durable record can
            # remain, and the censored session has no future consumer.
            self.agentic_kv_manager.end_session(
                str(request.session_id), now_ns=int(now_ns))
            prepared = {
                "request_id": int(request.id),
                "session_id": str(request.session_id),
                "time_ns": int(now_ns),
                "cancelled_pending_target": False,
                "released_owner_per_rank_bytes": 0,
                "released_retained_per_rank_bytes": 0,
                "released_source": None,
                "active_prefill_recompute_manager_ownership_already_released": (
                    True),
            }
        else:
            prepared = self.agentic_kv_manager.censor_prepared_request(
                request, int(now_ns))
        request.pd_prefill_full_per_rank_bytes = 0
        request.pd_prefill_initial_restored_per_rank_bytes = 0
        request.pd_prefill_reserved_per_rank_bytes = 0
        request.pd_prefill_owned_per_rank_bytes = 0
        request.pd_decode_target_instance_id = None
        request.pd_decode_full_per_rank_bytes = 0
        request.pd_decode_reserved_per_rank_bytes = 0
        request.pd_decode_owned_per_rank_bytes = 0
        request.pd_chunk_claim_pending = False
        request.pd_chunk_admitted_tokens = 0
        request.pd_chunk_admission_target_tokens = 0
        if (request.pd_chunk_admission_history
                and not request.pd_chunk_admission_history[-1].get(
                    "committed", False)):
            request.pd_chunk_admission_history[-1][
                "censored_before_commit"] = True
            request.pd_chunk_admission_history[-1]["censored_ns"] = int(
                now_ns)
        request.pd_kv_ownership_state = "censored"
        return {
            **prepared,
            "prefill_instance_id": int(prefill_scheduler.instance_id),
            "decode_instance_id": int(decode_scheduler.instance_id),
            "released_prefill_fresh_per_rank_bytes": fresh_p,
            "released_decode_fresh_per_rank_bytes": fresh_d,
            "released_prefill_current_per_rank_bytes": current_p,
            "released_decode_current_per_rank_bytes": current_d,
        }

    def _censor_pending_pd_prefill_launches(self, cutoff_time_ns):
        """Release consumed P/D allocations that were never enqueued.

        A handoff can consume both active-HBM claims before an asynchronous
        restore makes the request runnable.  Such a row lives only in
        ``_pending_prefill_launches``: neither scheduler owns it yet, but its
        restored/retained prefixes and both suffix preallocations are already
        physical.  This is distinct from a raw pending handoff, whose claims
        have not been consumed.
        """
        if self.agentic_kv_manager is None:
            return []
        cutoff_time_ns = int(cutoff_time_ns)
        audits = []
        for launch in self._pending_prefill_launches:
            request = launch["request"]
            prefill_sched = launch["prefill_scheduler"]
            decode_id = request.pd_decode_target_instance_id
            if decode_id is None:
                raise RuntimeError(
                    "Pending P/D prefill launch lost its decode target: "
                    f"request={request.id}")
            decode_sched = self._scheduler_by_instance(int(decode_id))
            if request in prefill_sched.request:
                raise RuntimeError(
                    "Pending P/D launch is already visible to its prefill "
                    f"scheduler: request={request.id}")
            audit = self._censor_incremental_pd_request(
                request, prefill_sched, decode_sched, cutoff_time_ns)
            audit["prefill_instance_id"] = int(
                prefill_sched.instance_id)
            audit["decode_instance_id"] = int(decode_sched.instance_id)
            audits.append(audit)
            self._release_pd_admission_owner(request.id)
        self._pending_prefill_launches.clear()
        return audits

    def _cancel_scheduler_hbm_claims_for_censoring(self, cutoff_time_ns):
        """Cancel exact queued-request claims after P/D claims are gone."""
        if self.agentic_kv_manager is None:
            return []
        audits = []
        for scheduler in self.schedulers:
            claim = self.agentic_kv_manager.active_hbm_reclaim_claim(
                scheduler.instance_id)
            if claim is None:
                continue
            if claim.owner_kind != "scheduler":
                raise RuntimeError(
                    "Measurement censoring found a non-scheduler HBM claim "
                    "after pending P/D cleanup: "
                    f"instance={scheduler.instance_id}, "
                    f"owner={(claim.owner_kind, claim.owner_id)}")
            matches = [
                request for request in scheduler.request
                if int(request.id) == int(claim.owner_id)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "Scheduler HBM claim has no unique queued owner during "
                    f"measurement censoring: instance={scheduler.instance_id}, "
                    f"owner={claim.owner_id}, matches={len(matches)}")
            cancelled = self.agentic_kv_manager.cancel_active_hbm_reclaim(
                scheduler.instance_id, int(cutoff_time_ns))
            if (cancelled is None
                    or cancelled.owner_kind != "scheduler"
                    or int(cancelled.owner_id) != int(claim.owner_id)):
                raise RuntimeError(
                    "Failed to cancel the exact scheduler HBM claim: "
                    f"instance={scheduler.instance_id}, owner={claim.owner_id}")
            scheduler.decode_handoff_claim_pending = False
            audits.append({
                "instance_id": int(scheduler.instance_id),
                "request_id": int(claim.owner_id),
                "per_rank_bytes": int(claim.per_rank_bytes),
                "ready_ns": int(claim.ready_ns),
            })
        return audits

    def _censor_scheduler_queues(self, cutoff_time_ns):
        """Release all non-running scheduler ownership at early stop."""
        if self.agentic_kv_manager is None:
            return [], []
        for scheduler in self.schedulers:
            if scheduler.inflight:
                raise RuntimeError(
                    "Measurement censoring began before dispatched batches "
                    f"drained: instance={scheduler.instance_id}, "
                    f"inflight={len(scheduler.inflight)}")

        pd_prefill_audits = []
        active_audits = []
        seen_request_ids = set()
        for scheduler in self.schedulers:
            for request in list(scheduler.request):
                request_id = int(request.id)
                if request_id in seen_request_ids:
                    raise RuntimeError(
                        "One request is queued on multiple schedulers during "
                        f"measurement censoring: request={request_id}")
                seen_request_ids.add(request_id)
                if request.pd_kv_ownership_state == "prefill_active":
                    if scheduler.pd_type != "prefill":
                        raise RuntimeError(
                            "Strict P/D preallocation is queued outside its "
                            f"prefill scheduler: request={request_id}, "
                            f"instance={scheduler.instance_id}")
                    decode_id = request.pd_decode_target_instance_id
                    if decode_id is None:
                        raise RuntimeError(
                            "Queued strict P/D request lost its decode target: "
                            f"request={request_id}")
                    scheduler.request.remove(request)
                    audit = self._censor_incremental_pd_request(
                        request,
                        scheduler,
                        self._scheduler_by_instance(int(decode_id)),
                        int(cutoff_time_ns),
                    )
                    audit["queue_state"] = "prefill_queued"
                    pd_prefill_audits.append(audit)
                    continue
                audit = scheduler.censor_queued_request(
                    request, int(cutoff_time_ns))
                audit["queue_state"] = "active_queued"
                active_audits.append(audit)
        return pd_prefill_audits, active_audits

    def censor_completed_pd_prefill_requests(
            self, requests, cutoff_time_ns):
        """Release D receive HBM for P completions drained after a freeze."""
        if not requests:
            return []
        if self.agentic_kv_manager is None:
            raise RuntimeError(
                "Cannot censor completed strict P/D requests without the KV "
                "manager")
        audits = []
        for request in requests:
            prefill_id = int(request.instance_id)
            prefill_sched = self._scheduler_by_instance(prefill_id)
            if prefill_sched.pd_type != "prefill":
                raise RuntimeError(
                    "Post-freeze P/D completion did not originate on P: "
                    f"request={request.id}, instance={prefill_id}")
            decode_id = request.pd_decode_target_instance_id
            if decode_id is None:
                raise RuntimeError(
                    "Post-freeze P/D completion lost its decode target: "
                    f"request={request.id}")
            audit = (
                self.agentic_kv_manager
                .censor_completed_pd_prefill_request(
                    request,
                    prefill_instance_id=prefill_id,
                    decode_instance_id=int(decode_id),
                    now_ns=int(cutoff_time_ns),
                )
            )
            request.pd_decode_owned_per_rank_bytes = 0
            request.pd_kv_ownership_state = "censored"
            audits.append(audit)
        self._censored_completed_pd_prefill_audits.extend(audits)
        return audits

    def finalize_measurement_censoring(self, cutoff_time_ns):
        """Mark every non-completed session as censored at early stop."""
        if not self._session_admission_frozen:
            return {
                'cutoff_time_ns': int(cutoff_time_ns),
                'censored_sessions': 0,
                'status_counts': {},
            }
        full_model_hbf_queue_audit = None
        if self.full_model_hbf_adapter is not None:
            full_model_hbf_queue_audit = (
                self.censor_idle_full_model_hbf_native_queues(
                    cutoff_time_ns))
            if full_model_hbf_queue_audit["skipped_busy_schedulers"]:
                raise RuntimeError(
                    "Full-model HBF measurement censoring began before "
                    "native GPU batches drained: "
                    f"{full_model_hbf_queue_audit['skipped_busy_schedulers']}")
            if self.full_model_hbf_adapter.has_pending_native_gpu_requests():
                raise RuntimeError(
                    "Full-model HBF measurement censoring still owns a "
                    "native GPU call outside an idle Scheduler queue")
        status_counts = {}
        censored = 0
        capacity_pending_count = len(
            self._pending_capacity_preparations)
        pd_admission_wait_count = len(
            self._pending_pd_admission_waits)
        decode_handoff_count = sum(
            len(rows) for rows in self._pending_decode_handoffs.values())
        pd_chunk_admission_count = sum(
            len(rows)
            for rows in self._pending_pd_chunk_admissions.values())
        prefill_launch_count = len(self._pending_prefill_launches)
        frozen_sync_prepare_count = len(self._frozen_sync_preparations)
        queued_request_count = (
            sum(len(scheduler.request) for scheduler in self.schedulers)
            + len(self._censored_full_model_hbf_gpu_queue_audits)
        )
        pending_request_count = (
            len(self._pending_requests) - self._pending_idx)
        active_session_ids_at_cutoff = sorted(self._active_sessions)
        memory_before = {
            str(scheduler.instance_id): {
                'npu_used': int(scheduler.memory.npu_used),
                'npu_baseline': int(scheduler.memory.weight),
                'cpu_used': int(getattr(
                    scheduler.memory, 'cpu_used', 0)),
            }
            for scheduler in self.schedulers
        }
        censored_session_ids = set()
        for lifecycle in self._session_lifecycle.values():
            previous = str(lifecycle.get('status') or 'unknown')
            if previous == 'completed':
                continue
            lifecycle['status_before_censoring'] = previous
            lifecycle['status'] = 'censored'
            lifecycle['censored_time_ns'] = int(cutoff_time_ns)
            status_counts[previous] = status_counts.get(previous, 0) + 1
            censored += 1
            censored_session_ids.add(str(lifecycle['session_id']))
        for row in self._pending_requests[self._pending_idx:]:
            if row.get('session_id') is not None:
                censored_session_ids.add(str(row['session_id']))
        for collection in (
                self._frozen_sync_preparations,
                self._pending_capacity_preparations,
                self._pending_pd_admission_waits):
            for pending in collection:
                session_id = pending['request'].get('session_id')
                if session_id is not None:
                    censored_session_ids.add(str(session_id))
        for queue in self._pending_decode_handoffs.values():
            for handoff in queue:
                if handoff['request'].session_id is not None:
                    censored_session_ids.add(str(
                        handoff['request'].session_id))
        for queue in self._pending_pd_chunk_admissions.values():
            for handoff in queue:
                if handoff['request'].session_id is not None:
                    censored_session_ids.add(str(
                        handoff['request'].session_id))
        for launch in self._pending_prefill_launches:
            if launch['request'].session_id is not None:
                censored_session_ids.add(str(
                    launch['request'].session_id))
        for scheduler in self.schedulers:
            for request in scheduler.request:
                if request.session_id is not None:
                    censored_session_ids.add(str(request.session_id))
        censored_handoff_audits = self._censor_pending_pd_handoffs(
            cutoff_time_ns)
        censored_launch_audits = self._censor_pending_pd_prefill_launches(
            cutoff_time_ns)
        scheduler_claim_audits = (
            self._cancel_scheduler_hbm_claims_for_censoring(
                cutoff_time_ns))
        queued_pd_audits, queued_active_audits = (
            self._censor_scheduler_queues(cutoff_time_ns))
        censored_demotion_join_audits = []
        censored_destination_admission_audits = []
        censored_transient_dram_admission_audits = []
        if self.agentic_kv_manager is not None:
            for pending in self._pending_capacity_preparations:
                request_id = int(pending['request']['index'])
                self._release_pd_admission_owner(request_id)
            for session_id in sorted(censored_session_ids):
                audit = self.agentic_kv_manager.censor_session(
                    session_id, cutoff_ns=int(cutoff_time_ns))
                if audit is not None:
                    if audit.get('source_demotion_join') is not None:
                        censored_demotion_join_audits.append(
                            audit['source_demotion_join'])
                    if audit.get('destination_admission') is not None:
                        censored_destination_admission_audits.append(
                            audit['destination_admission'])
                    if audit.get('transient_dram_admission') is not None:
                        censored_transient_dram_admission_audits.append(
                            audit['transient_dram_admission'])
        self._pending_capacity_preparations.clear()
        self._pending_pd_admission_waits.clear()
        self._frozen_sync_preparations.clear()
        del self._pending_requests[self._pending_idx:]
        self._deferred_sessions.clear()
        self._request_to_session.clear()
        self._session_affinity.clear()
        self._session_decode_affinity.clear()
        self._active_sessions.clear()
        if self._pd_admission_owner:
            raise RuntimeError(
                "Measurement censoring left P/D admission owners: "
                f"{self._pd_admission_owner}")
        memory_after = {
            str(scheduler.instance_id): {
                'npu_used': int(scheduler.memory.npu_used),
                'npu_baseline': int(scheduler.memory.weight),
                'cpu_used': int(getattr(
                    scheduler.memory, 'cpu_used', 0)),
            }
            for scheduler in self.schedulers
        }
        for instance_id, observed in memory_after.items():
            if observed['npu_used'] != observed['npu_baseline']:
                raise RuntimeError(
                    "Measurement censoring left non-weight NPU ownership: "
                    f"instance={instance_id}, state={observed}")
            if observed['cpu_used'] != 0:
                raise RuntimeError(
                    "Measurement censoring left active CPU ownership: "
                    f"instance={instance_id}, state={observed}")
        manager_drain_audit = None
        if self.agentic_kv_manager is not None:
            validate_drained = getattr(
                self.agentic_kv_manager,
                'validate_measurement_censoring_drained',
                None,
            )
            if validate_drained is not None:
                manager_drain_audit = validate_drained()
        return {
            'cutoff_time_ns': int(cutoff_time_ns),
            'censored_sessions': censored,
            'status_counts_before_censoring': dict(sorted(
                status_counts.items())),
            'active_session_ids_at_cutoff': active_session_ids_at_cutoff,
            'pending_request_rows_at_cutoff': pending_request_count,
            'pending_decode_handoffs_at_cutoff': decode_handoff_count,
            'pending_pd_chunk_admissions_at_cutoff': (
                pd_chunk_admission_count),
            'censored_pending_decode_handoffs': len(
                censored_handoff_audits),
            'censored_pending_decode_handoff_audits': (
                censored_handoff_audits),
            'pending_prefill_launches_at_cutoff': prefill_launch_count,
            'censored_pending_prefill_launches': len(
                censored_launch_audits),
            'censored_pending_prefill_launch_audits': (
                censored_launch_audits),
            'queued_requests_at_cutoff': queued_request_count,
            'censored_queued_pd_prefill_requests': len(
                queued_pd_audits),
            'censored_queued_pd_prefill_audits': queued_pd_audits,
            'censored_queued_active_requests': len(
                queued_active_audits),
            'censored_queued_active_audits': queued_active_audits,
            'full_model_hbf_queue_censoring': (
                full_model_hbf_queue_audit),
            'censored_full_model_hbf_native_gpu_requests': len(
                self._censored_full_model_hbf_gpu_queue_audits),
            'censored_full_model_hbf_native_gpu_audits': list(
                self._censored_full_model_hbf_gpu_queue_audits),
            'full_model_hbf_accepted_work_policy': (
                None
                if self.full_model_hbf_adapter is None
                else 'drain_accepted_hbf_censor_native_gpu_queue'),
            'cancelled_scheduler_hbm_claims': len(
                scheduler_claim_audits),
            'cancelled_scheduler_hbm_claim_audits': (
                scheduler_claim_audits),
            'censored_completed_pd_prefill_requests': len(
                self._censored_completed_pd_prefill_audits),
            'censored_completed_pd_prefill_audits': list(
                self._censored_completed_pd_prefill_audits),
            'pending_prepare_rows_at_cutoff': frozen_sync_prepare_count,
            'pending_capacity_prepare_rows_at_cutoff': (
                capacity_pending_count),
            'pending_pd_admission_waits_at_cutoff': (
                pd_admission_wait_count),
            'released_prepare_rows_at_freeze': (
                self._censored_pending_prepare_rows),
            'ended_censored_session_ids': sorted(censored_session_ids),
            'ended_censored_sessions': len(censored_session_ids),
            'censored_source_demotion_joins': len(
                censored_demotion_join_audits),
            'censored_source_demotion_join_audits': (
                censored_demotion_join_audits),
            'censored_destination_admissions': len(
                censored_destination_admission_audits),
            'censored_destination_admission_audits': (
                censored_destination_admission_audits),
            'censored_transient_dram_admissions': len(
                censored_transient_dram_admission_audits),
            'censored_transient_dram_admission_audits': (
                censored_transient_dram_admission_audits),
            'memory_before_censoring': memory_before,
            'memory_after_censoring': memory_after,
            'manager_drain_audit': manager_drain_audit,
        }

    def _activate_session(self, session_id, admission_time_ns):
        if session_id in self._active_sessions:
            return
        admission_index = self._sessions_admitted
        self._active_sessions.add(session_id)
        self._sessions_admitted += 1
        lifecycle = self._session_lifecycle.get(session_id)
        if lifecycle is not None:
            lifecycle['admission_index'] = int(admission_index)
            lifecycle['admission_time_ns'] = int(admission_time_ns)
            lifecycle['admission_queue_wait_ns'] = max(
                0,
                int(admission_time_ns)
                - int(lifecycle['offered_time_ns']),
            )
            lifecycle['status'] = 'active'

    def _complete_session(self, session_id, completion_time_ns):
        if session_id not in self._active_sessions:
            raise RuntimeError(
                f"Completed session {session_id!r} was not active")
        self._active_sessions.remove(session_id)
        self._sessions_completed += 1
        self._completed_session_ids.add(str(session_id))
        lifecycle = self._session_lifecycle.get(session_id)
        if lifecycle is not None:
            lifecycle['completion_time_ns'] = int(completion_time_ns)
            lifecycle['e2e_ns'] = (
                int(completion_time_ns)
                - int(lifecycle['admission_time_ns'])
            )
            lifecycle['status'] = 'completed'

    def session_admission_summary(self):
        """Return auditable lifecycle counters for load-controlled runs."""
        remaining_backlog = (
            len(self._session_backlog) - self._session_backlog_idx)
        planned = (
            len(self._session_backlog)
            if self.session_admission.mode == 'backlog'
            else len(self._session_lifecycle)
        )
        logical_session_drop_count = sum(
            str(row.get('status')) == 'dropped'
            for row in self._session_lifecycle.values()
        )
        return {
            'mode': self.session_admission.mode,
            'queue_policy': (
                'fifo_wait_for_slot'
                if self.session_admission.mode == 'backlog'
                else 'poisson_fifo_wait_for_slot'
                if (self.session_admission.mode == 'poisson'
                    and self.session_admission.max_active_sessions > 0)
                else 'arrival_time_order'
            ),
            'logical_session_drop_count': logical_session_drop_count,
            'slot_release_event': (
                'final_request_completion_on_decode_owner'
                if self.decode_schedulers
                else 'final_request_completion_on_colocated_owner'
            ),
            'slot_release_event_legacy': (
                'final_decode_completion'
                if self.decode_schedulers
                else 'final_llm_request_completion'
            ),
            'cutoff_disposition': (
                'right_censor'
                if self.session_admission.stop_after_measurement
                else 'drain'
            ),
            'measurement_cohort_selection': (
                self.session_admission.measurement_cohort_selection),
            'measurement_warmup_session_ids': list(
                self._measurement_warmup_session_ids),
            'measurement_warmup_session_count': len(
                self._measurement_warmup_session_ids),
            'measurement_warmup_completed_sessions': len(
                self._measurement_warmup_session_id_set
                & self._completed_session_ids),
            'measurement_target_session_count': len(
                self._measurement_target_session_ids),
            'measurement_target_completed_sessions': len(
                self._measurement_target_session_id_set
                & self._completed_session_ids),
            'measurement_required_session_ids': list(
                self._measurement_required_session_ids),
            'measurement_required_session_count': len(
                self._measurement_required_session_ids),
            'measurement_required_completed_sessions': len(
                self._measurement_required_session_id_set
                & self._completed_session_ids),
            'measurement_prefix_id_overlap_count': len(
                self._measurement_warmup_session_id_set
                & self._measurement_target_session_id_set),
            'max_active_sessions': (
                self.session_admission.max_active_sessions),
            'backlog_epochs': self.session_admission.backlog_epochs,
            'session_arrival_rate_sps': (
                self.session_admission.session_arrival_rate_sps),
            'session_arrival_seed': (
                self.session_admission.session_arrival_seed),
            'warmup_completions': (
                self.session_admission.warmup_completions),
            'measure_completions': (
                self.session_admission.measure_completions),
            'stop_after_measurement': (
                self.session_admission.stop_after_measurement),
            'templates_loaded': self._session_templates_loaded,
            'offered_sessions': planned,
            'planned_sessions': planned,
            'admitted_sessions': self._sessions_admitted,
            'completed_sessions': self._sessions_completed,
            'active_sessions': len(self._active_sessions),
            'admission_frozen': self._session_admission_frozen,
            'remaining_unadmitted_sessions': (
                planned - self._sessions_admitted),
            'remaining_backlog_sessions': remaining_backlog,
            'first_admission_time_ns': min(
                (row['admission_time_ns']
                 for row in self._session_lifecycle.values()
                 if row['admission_time_ns'] is not None),
                default=None,
            ),
            'last_completion_time_ns': max(
                (row['completion_time_ns']
                 for row in self._session_lifecycle.values()
                 if row['completion_time_ns'] is not None),
                default=None,
            ),
        }

    def session_lifecycle_records(self):
        """Return one copy-safe row per offered agentic session."""
        return [dict(row) for row in self._session_lifecycle.values()]

    @staticmethod
    def _prefix_reuse(completed_sub, next_sub, request):
        """Return reusable tokens and whether they were exact or estimated."""
        if 'prefix_reuse_toks' in next_sub:
            reuse = int(next_sub['prefix_reuse_toks'])
            source = next_sub.get('prefix_reuse_source', 'reported')
            return max(0, min(reuse, int(next_sub['input_toks']))), source

        previous_input_tokens = int(completed_sub.get('input_toks', 0))
        previous_output_tokens = int(completed_sub.get('output_toks', 0))
        previous_ids = list(
            completed_sub.get('input_tok_ids', []))[:previous_input_tokens]
        # Preserve positional identity: output IDs begin only after the full
        # declared input. A partial input-ID array followed by output IDs has
        # a gap and cannot be concatenated into a longer verified prefix.
        if len(previous_ids) == previous_input_tokens:
            previous_ids += list(
                completed_sub.get('output_tok_ids', [])
            )[:max(0, previous_output_tokens - 1)]
        next_ids = list(next_sub.get('input_tok_ids', []))[
            :int(next_sub.get('input_toks', 0))]
        if previous_ids and next_ids:
            common = 0
            for left, right in zip(previous_ids, next_ids):
                if left != right:
                    break
                common += 1
            return common, 'exact'

        computed = int(request.num_computed_tokens) if hasattr(request, 'num_computed_tokens') else 0
        return max(0, min(computed, int(next_sub['input_toks']))), 'estimated'

    @staticmethod
    def _return_gap_type(completed_sub):
        """Normalize the incoming return class for the following call."""
        value = str(
            completed_sub.get('inter_turn_gap_type') or 'unknown'
        ).strip().lower()
        if value in {'human', 'tool', 'mixed', 'unknown'}:
            return value
        # ``none`` normally appears only on a session's final call. If a
        # malformed/legacy trace uses it between calls, preserve uncertainty
        # instead of inventing a human or tool return.
        return 'unknown'

    @staticmethod
    def _pending_time(req_data):
        return int(req_data.get('ready_time_ns', req_data['arrival_time_ns']))

    def _insert_pending_sorted(self, req_data):
        """Insert a request into _pending_requests maintaining arrival-time
        sort order for the not-yet-consumed portion (from _pending_idx onward)."""
        arrival = self._pending_time(req_data)
        # Binary search in the unconsumed portion
        lo = self._pending_idx
        hi = len(self._pending_requests)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._pending_time(self._pending_requests[mid]) <= arrival:
                lo = mid + 1
            else:
                hi = mid
        self._pending_requests.insert(lo, req_data)

    def has_deferred_sessions(self):
        """Check if there are agentic sessions with unreleased sub-requests."""
        return (
            bool(self._deferred_sessions)
            or (
                not self._session_admission_frozen
                and self._session_backlog_idx < len(self._session_backlog)
            )
        )

    def get_next_pending_arrival(self):
        """Return the next pending request's arrival time, or None."""
        ready_times = []
        if self._pending_idx < len(self._pending_requests):
            ready_times.append(self._pending_time(
                self._pending_requests[self._pending_idx]))
        ready_times.extend(
            self._pending_time(pending['request'])
            for pending in self._pending_sync_preparations
        )
        ready_times.extend(
            int(pending['retry_time_ns'])
            for pending in self._pending_capacity_preparations
            if pending.get('retry_time_ns') is not None
        )
        if (not self._session_admission_frozen
                and self._session_backlog_idx < len(self._session_backlog)
                and len(self._active_sessions)
                < self.session_admission.max_active_sessions):
            ready_times.append(int(
                self._session_backlog[self._session_backlog_idx].get(
                    'offered_time_ns', 0)))
        return min(ready_times) if ready_times else None

    # -----------------------------------------------------------------------
    # Legacy: upfront routing (kept for backward compat)
    # -----------------------------------------------------------------------

    def generate(self, path, enable_prefix_caching=False, is_init=True):
        """Load and immediately route all requests (legacy behavior)."""
        if self.agentic_kv_manager is not None and self.decode_schedulers:
            raise RuntimeError(
                "Legacy upfront generation is incompatible with strict P/D "
                "receive admission; use load_requests() and advance logical "
                "time through route_arrived_requests()")
        self.load_requests(path, enable_prefix_caching, is_init)
        # Route all at once (arrival time ignored)
        self.route_arrived_requests(float('inf'))
        for scheduler in self.schedulers:
            self.logger.info(
                "Added %d requests to scheduler[%d] (%s type)",
                len(scheduler.request),
                scheduler.instance_id,
                scheduler.pd_type
            )

    def _compatible_decode_schedulers(self, prefill_sched):
        return [
            candidate for candidate in self.decode_schedulers
            if candidate.node_id == prefill_sched.node_id
            and candidate.model == prefill_sched.model
            and candidate.tp_size == prefill_sched.tp_size
            and candidate.pp_size == prefill_sched.pp_size
            and candidate.block_size == prefill_sched.block_size
            and candidate.fp == prefill_sched.fp
            and candidate.kv_cache_dtype == prefill_sched.kv_cache_dtype
        ]

    def _bind_pd_decode_scheduler(self, req, prefill_sched):
        _, decode_id = self._pd_pair(prefill_sched)
        sched = self._scheduler_by_instance(decode_id)
        retained_instance = req.agentic_kv_retained_instance_id
        if (retained_instance is not None
                and retained_instance != sched.instance_id):
            raise RuntimeError(
                "Retained decode KV does not match the prefill graph's fixed "
                f"receiver: retained={retained_instance}, "
                f"target={sched.instance_id}")
        if req.session_id is not None:
            affinity = self._session_decode_affinity.get(req.session_id)
            if affinity is not None and affinity != sched.instance_id:
                raise RuntimeError(
                    "Sticky decode affinity changed before P/D admission: "
                    f"session={req.session_id}, old={affinity}, "
                    f"new={sched.instance_id}")
            self._session_decode_affinity[req.session_id] = sched.instance_id
        return sched

    def _stage_pd_receive_admission(self, req, prefill_sched, now_ns):
        """Bind a P/D request without reserving its uncomputed suffix.

        The restored P prefix and an HBM-retained D prefix already have
        physical ownership. Every other block is acquired immediately before
        the P chunk that first touches it. Keeping final prompt sizes here
        makes the handoff and censoring paths independently auditable.
        """
        decode_sched = self._bind_pd_decode_scheduler(req, prefill_sched)
        block_tokens = (
            (int(req.original_input) + decode_sched.block_size - 1)
            // decode_sched.block_size
            * decode_sched.block_size
        )
        decode_full_per_rank = int(decode_sched.memory.get_kv(block_tokens))
        retained_per_rank = int(req.agentic_kv_retained_per_rank_bytes)
        if (retained_per_rank < 0
                or retained_per_rank > decode_full_per_rank):
            raise RuntimeError(
                "Invalid retained decode KV for strict P/D admission: "
                f"retained={retained_per_rank}, "
                f"full={decode_full_per_rank}")
        decode_memory = decode_sched.memory
        decode_kv_ceiling = max(
            0,
            int(decode_memory.npu_mem)
            - max(0, int(decode_memory.weight)),
        )
        if decode_full_per_rank > decode_kv_ceiling:
            raise RuntimeError(
                "P/D decode KV cannot fit on one rank even after all idle "
                "state is reclaimed and before prefill compute begins: "
                f"instance={decode_sched.instance_id}, "
                f"required={decode_full_per_rank}, "
                f"kv_ceiling={decode_kv_ceiling}")

        prefill_block_tokens = (
            (int(req.original_input) + prefill_sched.block_size - 1)
            // prefill_sched.block_size
            * prefill_sched.block_size
        )
        prefill_full_per_rank = int(
            prefill_sched.memory.get_kv(prefill_block_tokens))
        restored_block_tokens = (
            (int(req.agentic_kv_hit_tokens) + prefill_sched.block_size - 1)
            // prefill_sched.block_size
            * prefill_sched.block_size
            if req.agentic_kv_hit_tokens > 0 else 0
        )
        restored_per_rank = int(
            prefill_sched.memory.get_kv(restored_block_tokens))
        if (restored_per_rank < 0
                or restored_per_rank > prefill_full_per_rank):
            raise RuntimeError(
                "Invalid restored prefill KV for strict P/D admission: "
                f"restored={restored_per_rank}, "
                f"full={prefill_full_per_rank}")
        if (restored_per_rank > 0
                and req.agentic_kv_owner_instance_id
                != prefill_sched.instance_id):
            raise RuntimeError(
                "Restored prefill KV is not owned by the selected P "
                f"instance: owner={req.agentic_kv_owner_instance_id}, "
                f"target={prefill_sched.instance_id}")
        prefill_memory = prefill_sched.memory
        prefill_kv_ceiling = max(
            0,
            int(prefill_memory.npu_mem)
            - max(0, int(prefill_memory.weight)),
        )
        if prefill_full_per_rank > prefill_kv_ceiling:
            raise RuntimeError(
                "P/D prefill KV cannot fit on one rank even after all idle "
                "state is reclaimed: "
                f"instance={prefill_sched.instance_id}, "
                f"required={prefill_full_per_rank}, "
                f"kv_ceiling={prefill_kv_ceiling}")

        req.pd_decode_target_instance_id = decode_sched.instance_id
        req.pd_decode_full_per_rank_bytes = decode_full_per_rank
        req.pd_decode_reserved_per_rank_bytes = 0
        req.pd_decode_owned_per_rank_bytes = retained_per_rank
        req.pd_decode_admission_enqueued_ns = int(now_ns)
        req.pd_prefill_full_per_rank_bytes = prefill_full_per_rank
        req.pd_prefill_initial_restored_per_rank_bytes = restored_per_rank
        req.pd_prefill_reserved_per_rank_bytes = 0
        req.pd_prefill_owned_per_rank_bytes = restored_per_rank
        req.pd_prefill_preallocated_per_rank_bytes = restored_per_rank
        req.pd_prefill_admission_enqueued_ns = int(now_ns)
        req.pd_kv_ownership_state = "prefill_active"
        if restored_per_rank:
            if req.agentic_kv_owner_instance_id != prefill_sched.instance_id:
                raise RuntimeError(
                    "Initial P/D restored ownership changed during binding: "
                    f"request={req.id}, owner="
                    f"{req.agentic_kv_owner_instance_id}, "
                    f"prefill={prefill_sched.instance_id}")
        elif req.agentic_kv_owner_instance_id is not None:
            raise RuntimeError(
                "A zero-byte P prefix unexpectedly has an HBM owner: "
                f"request={req.id}, owner="
                f"{req.agentic_kv_owner_instance_id}")
        expected_retained = (
            decode_sched.instance_id if retained_per_rank else None)
        if req.agentic_kv_retained_instance_id != expected_retained:
            raise RuntimeError(
                "Initial P/D retained ownership changed during binding: "
                f"request={req.id}, expected={expected_retained}, "
                f"observed={req.agentic_kv_retained_instance_id}")
        self._pending_prefill_launches.append({
            "request": req,
            "prefill_scheduler": prefill_sched,
        })
        pair = (
            int(prefill_sched.instance_id), int(decode_sched.instance_id))
        if self._pd_admission_owner.get(pair) == int(req.id):
            del self._pd_admission_owner[pair]

    @staticmethod
    def _pd_handoff_claim_roles(handoff):
        return (
            (
                "prefill",
                handoff["prefill_scheduler"],
                "prefill_needed_per_rank_bytes",
                "prefill_claim_ready_ns",
            ),
            (
                "decode",
                handoff["decode_scheduler"],
                "decode_needed_per_rank_bytes",
                "decode_claim_ready_ns",
            ),
        )

    def _pd_handoff_capacity_state(self, handoff, now_ns):
        """Return the exact pair state that can change admission outcome."""
        state = tuple(
            (
                role,
                int(sched.instance_id),
                self._restore_capacity_state(sched),
            )
            for role, sched, _, _ in self._pd_handoff_claim_roles(handoff)
        )
        # Physical HBM counters remain unchanged when a P graph commits, but
        # the committed partial prefill is no longer frozen and can safely be
        # reclaimed to admit this FIFO head. Make that graph dependency part
        # of the coalesced retry key so an unchanged-capacity poll cannot
        # strand the pair forever.
        prefill = handoff["prefill_scheduler"]
        state += ((
            "prefill_reclaimability_generation",
            int(prefill.instance_id),
            int(prefill.pd_prefill_reclaimability_generation),
        ),)
        if getattr(
                self.agentic_kv_manager,
                "restore_capacity_state", None) is None:
            # Compatibility managers cannot expose an exact capacity
            # generation. Their logical callback time is the only available
            # retry dependency; production managers use the state above.
            state += (("fallback_callback_time_ns", int(now_ns)),)
        return state

    def _rollback_partial_pd_hbm_claims(self, handoff, now_ns):
        """Release one-sided P/D reservations without cancelling demotions."""
        request_id = int(handoff["request"].id)
        expected_owner = ("pd", request_id)
        claimed_roles = []
        for role, sched, needed_key, ready_key in (
                self._pd_handoff_claim_roles(handoff)):
            if (int(handoff[needed_key]) == 0
                    or handoff[ready_key] is None):
                continue
            claim = self.agentic_kv_manager.active_hbm_reclaim_claim(
                sched.instance_id)
            if claim is None:
                raise RuntimeError(
                    f"Partial P/D {role} admission lost its active HBM "
                    f"claim: request={request_id}, "
                    f"instance={sched.instance_id}")
            owner = (claim.owner_kind, claim.owner_id)
            if owner != expected_owner:
                raise RuntimeError(
                    f"Partial P/D {role} admission claim owner changed: "
                    f"request={request_id}, expected={expected_owner}, "
                    f"observed={owner}")
            claimed_roles.append((
                role, sched, ready_key, claim))

        # All identities are proven before the first cancellation so a corrupt
        # second role cannot leave a one-sided P/D mutation.
        cancelled = 0
        for role, sched, ready_key, expected_claim in claimed_roles:
            released = self.agentic_kv_manager.cancel_active_hbm_reclaim(
                sched.instance_id, int(now_ns))
            if (released is None
                    or (released.owner_kind, released.owner_id)
                    != expected_owner
                    or released is not expected_claim):
                raise RuntimeError(
                    f"Failed to roll back exact P/D {role} HBM claim: "
                    f"request={request_id}, instance={sched.instance_id}")
            handoff[ready_key] = None
            sched.decode_handoff_claim_pending = False
            cancelled += 1
        return cancelled

    def pd_prefill_chunk_requirements(
            self, request, prefill_scheduler, chunk_tokens):
        """Return exact block ownership needed by one proposed P chunk.

        This helper is side-effect free. Policy code may use it as a causal
        snapshot, while ``admit_pd_prefill_chunk`` remains authoritative at
        dispatch because HBM slack can change between restore preparation and
        model scheduling.
        """
        chunk_tokens = int(chunk_tokens)
        if chunk_tokens <= 0:
            raise ValueError(
                f"P/D chunk must contain positive work, got {chunk_tokens}")
        if request.pd_decode_target_instance_id is None:
            raise RuntimeError(
                f"P/D request #{request.id} has no fixed decode target")
        decode_scheduler = self._scheduler_by_instance(
            int(request.pd_decode_target_instance_id))
        if int(prefill_scheduler.instance_id) == int(
                decode_scheduler.instance_id):
            raise RuntimeError(
                "Atomic P/D chunk admission requires distinct prefill and "
                f"decode instances: request={request.id}, instance="
                f"{prefill_scheduler.instance_id}")
        expected_pair = self._pd_pair(prefill_scheduler)
        if expected_pair != (
                int(prefill_scheduler.instance_id),
                int(decode_scheduler.instance_id)):
            raise RuntimeError(
                "P/D chunk target changed after binding: "
                f"request={request.id}, expected_pair={expected_pair}, "
                f"actual_decode={decode_scheduler.instance_id}")
        target_tokens = int(request.num_computed_tokens) + chunk_tokens
        if target_tokens > int(request.prefill_target_tokens):
            raise RuntimeError(
                "P/D chunk exceeds the prefill target: "
                f"request={request.id}, computed="
                f"{request.num_computed_tokens}, chunk={chunk_tokens}, "
                f"target={request.prefill_target_tokens}")

        def block_bytes(scheduler, tokens):
            block_tokens = (
                (int(tokens) + scheduler.block_size - 1)
                // scheduler.block_size * scheduler.block_size
                if int(tokens) > 0 else 0
            )
            return int(scheduler.memory.get_kv(block_tokens))

        prefill_current = int(
            request.pd_prefill_owned_per_rank_bytes)
        decode_current = int(request.pd_decode_owned_per_rank_bytes)
        prefill_target = block_bytes(prefill_scheduler, target_tokens)
        decode_target = block_bytes(decode_scheduler, target_tokens)
        if prefill_current > prefill_target or decode_current > decode_target:
            raise RuntimeError(
                "P/D chunk ownership is ahead of its logical target: "
                f"request={request.id}, target_tokens={target_tokens}, "
                f"prefill={prefill_current}/{prefill_target}, "
                f"decode={decode_current}/{decode_target}")
        unreserved = getattr(
            self.agentic_kv_manager,
            "hbm_unreserved_per_rank_bytes",
            None,
        )

        def available_bytes(scheduler):
            if unreserved is not None:
                return int(unreserved(scheduler.instance_id))
            return int(self._restore_capacity_state(scheduler)[1])

        return {
            "request_id": int(request.id),
            "active_prefill_recompute_generation": int(
                request.pd_active_prefill_recompute_generation),
            "prefill_instance_id": int(prefill_scheduler.instance_id),
            "decode_instance_id": int(decode_scheduler.instance_id),
            "computed_tokens": int(request.num_computed_tokens),
            "chunk_tokens": chunk_tokens,
            "target_tokens": target_tokens,
            "prefill_current_per_rank_bytes": prefill_current,
            "decode_current_per_rank_bytes": decode_current,
            "prefill_target_per_rank_bytes": prefill_target,
            "decode_target_per_rank_bytes": decode_target,
            "prefill_delta_per_rank_bytes": max(
                0, prefill_target - prefill_current),
            "decode_delta_per_rank_bytes": max(
                0, decode_target - decode_current),
            "prefill_unreserved_per_rank_bytes": available_bytes(
                prefill_scheduler),
            "decode_unreserved_per_rank_bytes": available_bytes(
                decode_scheduler),
        }

    def _pending_pd_chunk_for_request(self, request_id):
        request_id = int(request_id)
        for queue in self._pending_pd_chunk_admissions.values():
            for handoff in queue:
                if int(handoff["request"].id) == request_id:
                    return handoff
        return None

    @staticmethod
    def _pd_prefill_progress_priority(request):
        """Rank FIFO-visible P owners without changing their queue order."""
        return (
            -int(request.num_computed_tokens),
            int(request.arrival),
            int(request.ready_time),
            int(request.id),
        )

    def _cancel_pending_pd_chunk_for_preemption(self, request, now_ns):
        """Cancel one victim's exact P/D chunk claim, if it has one."""
        matches = []
        for pair, queue in self._pending_pd_chunk_admissions.items():
            for handoff in queue:
                if handoff["request"] is request:
                    matches.append((pair, queue, handoff))
        if len(matches) > 1:
            raise RuntimeError(
                "One P/D request owns multiple pending chunk claims: "
                f"request={request.id}, count={len(matches)}")
        if not matches:
            if request.pd_chunk_claim_pending:
                raise RuntimeError(
                    "P/D request marks a pending chunk without a router "
                    f"claim: request={request.id}")
            return False

        pair, queue, handoff = matches[0]
        enqueued_ns = int(handoff["enqueued_ns"])
        if int(now_ns) < enqueued_ns:
            raise RuntimeError(
                "P/D chunk preemption precedes its enqueue time: "
                f"request={request.id}, enqueue={enqueued_ns}, "
                f"preempt={now_ns}")
        wait_ns = int(now_ns) - enqueued_ns
        critical_wait_ns = max(
            0,
            int(now_ns) - max(
                enqueued_ns,
                int(request.agentic_kv_restore_ready_time_ns),
            ),
        )
        cancellation_history = {
            **handoff["requirements"],
            "enqueued_ns": enqueued_ns,
            "cancelled_ns": int(now_ns),
            "wait_ns": wait_ns,
            "critical_wait_after_restore_ns": critical_wait_ns,
            "cancelled_by_active_prefill_recompute": True,
            "invalidated_by_active_prefill_recompute": True,
            "invalidated_ns": int(now_ns),
            "preempted_before_commit": True,
            "committed": False,
        }
        self._rollback_partial_pd_hbm_claims(handoff, int(now_ns))
        queue.remove(handoff)
        if not queue:
            del self._pending_pd_chunk_admissions[pair]
        request.pd_chunk_admission_wait_ns = wait_ns
        request.pd_chunk_admission_critical_wait_ns = critical_wait_ns
        request.pd_chunk_admission_wait_ns_total += wait_ns
        request.pd_chunk_admission_critical_wait_ns_total += (
            critical_wait_ns)
        request.pd_chunk_cancelled_admission_count += 1
        request.pd_chunk_cancelled_admission_wait_ns_total += wait_ns
        request.pd_chunk_cancelled_admission_critical_wait_ns_total += (
            critical_wait_ns)
        request.pd_chunk_admission_history.append(cancellation_history)
        request.pd_chunk_claim_pending = False
        self.agentic_kv_manager.release_synchronous_prepare_lock(
            int(request.id))
        self.agentic_kv_manager.record_pd_chunk_admission_cancellation(
            request, dict(cancellation_history))
        return True

    def _pd_prefill_victim_is_restore_ready(
            self, request, prefill, decode, now_ns):
        """Exclude in-flight or not-yet-materialized restore destinations."""
        if int(request.pd_chunk_admitted_tokens) != 0:
            return False
        if (request.pd_chunk_admission_history
                and not request.pd_chunk_admission_history[-1].get(
                    "committed", False)
                and not request.pd_chunk_admission_history[-1].get(
                    "cancelled_by_active_prefill_recompute", False)):
            return False
        if int(request.agentic_kv_restore_ready_time_ns) > int(now_ns):
            return False
        if any(
                request in batch.requests
                for scheduler in (prefill, decode)
                for batch in scheduler.inflight):
            return False
        pending_allocations = getattr(
            self.agentic_kv_manager, "pending_hbm_allocations", ())
        session_id = str(request.session_id)
        if any(
                str(pending.entry.session_id) == session_id
                and int(pending.entry.instance_id) in {
                    int(prefill.instance_id), int(decode.instance_id)}
                for pending in pending_allocations):
            return False
        return True

    def _preempt_one_pd_prefill(self, request, prefill, decode, now_ns):
        """Atomically discard one queued victim's P and D KV ownership."""
        if request.pd_kv_ownership_state != "prefill_active":
            raise RuntimeError(
                "P/D progress victim is not an active prefill: "
                f"request={request.id}, state="
                f"{request.pd_kv_ownership_state}")
        if int(prefill.instance_id) == int(decode.instance_id):
            raise RuntimeError(
                "Active P/D prefill recomputation requires distinct prefill "
                f"and decode instances: request={request.id}, instance="
                f"{prefill.instance_id}")
        if request not in prefill.request:
            raise RuntimeError(
                f"P/D progress victim #{request.id} left the P queue")
        protected_ids = getattr(
            prefill,
            "pd_chunk_admission_pass_protected_request_ids",
            (),
        )
        if int(request.id) in protected_ids:
            raise RuntimeError(
                "P/D progress attempted to reclaim an owner already selected "
                f"for the current batch: request={request.id}, instance="
                f"{prefill.instance_id}")
        if int(request.pd_chunk_admitted_tokens) != 0:
            raise RuntimeError(
                "P/D progress cannot reclaim a chunk frozen until graph "
                f"commit: request={request.id}, admitted="
                f"{request.pd_chunk_admitted_tokens}")
        if (request.pd_chunk_admission_history
                and not request.pd_chunk_admission_history[-1].get(
                    "committed", False)
                and not request.pd_chunk_admission_history[-1].get(
                    "cancelled_by_active_prefill_recompute", False)):
            raise RuntimeError(
                "P/D progress cannot reclaim an uncommitted finalized "
                f"chunk: request={request.id}")
        if int(request.pd_decode_target_instance_id) != int(
                decode.instance_id):
            raise RuntimeError(
                "P/D progress victim changed decode target: "
                f"request={request.id}, target="
                f"{request.pd_decode_target_instance_id}, "
                f"expected={decode.instance_id}")
        if not self._pd_prefill_victim_is_restore_ready(
                request, prefill, decode, int(now_ns)):
            raise RuntimeError(
                "P/D progress selected an in-flight restore or graph: "
                f"request={request.id}, now={now_ns}, restore_ready="
                f"{request.agentic_kv_restore_ready_time_ns}")

        prefill_owned = int(request.pd_prefill_owned_per_rank_bytes)
        decode_owned = int(request.pd_decode_owned_per_rank_bytes)
        expected_prefill = (
            int(request.pd_prefill_initial_restored_per_rank_bytes)
            + int(request.pd_prefill_reserved_per_rank_bytes))
        expected_decode = (
            int(request.agentic_kv_retained_per_rank_bytes)
            + int(request.pd_decode_reserved_per_rank_bytes))
        if prefill_owned != expected_prefill:
            raise RuntimeError(
                "P/D prefill victim P ownership does not reconcile: "
                f"request={request.id}, owned={prefill_owned}, "
                f"initial_plus_fresh={expected_prefill}")
        if decode_owned != expected_decode:
            raise RuntimeError(
                "P/D prefill victim D ownership does not reconcile: "
                f"request={request.id}, owned={decode_owned}, "
                f"retained_plus_fresh={expected_decode}")
        expected_prefill_owner = (
            int(prefill.instance_id) if prefill_owned else None)
        if request.agentic_kv_owner_instance_id != expected_prefill_owner:
            raise RuntimeError(
                "P/D prefill victim physical P owner changed before "
                f"reclamation: request={request.id}, expected="
                f"{expected_prefill_owner}, observed="
                f"{request.agentic_kv_owner_instance_id}")
        retained_bytes = int(
            request.agentic_kv_retained_per_rank_bytes)
        expected_retained_owner = (
            int(decode.instance_id) if retained_bytes else None)
        if (request.agentic_kv_retained_instance_id
                != expected_retained_owner):
            raise RuntimeError(
                "P/D prefill victim retained D owner changed before "
                f"reclamation: request={request.id}, expected="
                f"{expected_retained_owner}, observed="
                f"{request.agentic_kv_retained_instance_id}")
        # Mirror every guard in Request.begin_active_prefill_recompute before
        # either physical free. The later transition is therefore infallible
        # with respect to request state and P/D release remains atomic.
        computed_tokens = int(request.num_computed_tokens)
        if request.recompute_target_tokens is not None:
            raise RuntimeError(
                "P/D prefill victim is already rebuilding decode KV: "
                f"request={request.id}")
        if (computed_tokens <= 0
                or computed_tokens >= int(request.original_input)):
            raise RuntimeError(
                "P/D progress victim is not a partial prefill: "
                f"request={request.id}, computed={computed_tokens}, "
                f"prompt={request.original_input}")
        if int(request.generated_tokens) != 0:
            raise RuntimeError(
                "P/D progress victim generated output before prompt "
                f"completion: request={request.id}, generated="
                f"{request.generated_tokens}")
        restored_hit_tokens = int(request.agentic_kv_hit_tokens)
        old_generation = int(
            request.pd_active_prefill_recompute_generation)
        restored_discarded_before = int(
            request
            .agentic_kv_restored_tokens_discarded_by_active_prefill_recompute)
        if (restored_hit_tokens < 0
                or (old_generation == 0
                    and (restored_hit_tokens > computed_tokens
                         or restored_discarded_before != 0))
                or (old_generation > 0
                    and restored_discarded_before != restored_hit_tokens)):
            raise RuntimeError(
                "P/D progress victim restored-hit ownership is invalid: "
                f"request={request.id}, hit={restored_hit_tokens}, "
                f"materialized={computed_tokens}, generation="
                f"{old_generation}, already_discarded="
                f"{restored_discarded_before}")
        prefill_kv_used = (
            int(prefill.memory.npu_used) - int(prefill.memory.weight))
        decode_kv_used = (
            int(decode.memory.npu_used) - int(decode.memory.weight))
        if (prefill_kv_used < prefill_owned
                or decode_kv_used < decode_owned):
            raise RuntimeError(
                "P/D victim owns more KV than its engines allocate above "
                "the immutable model-weight floor: "
                f"request={request.id}, P={prefill_owned}/"
                f"{prefill_kv_used}, D={decode_owned}/{decode_kv_used}")

        generation_histories = [
            history for history in request.pd_chunk_admission_history
            if int(history.get(
                "active_prefill_recompute_generation", 0))
            == old_generation
        ]
        self._cancel_pending_pd_chunk_for_preemption(request, int(now_ns))

        for history in generation_histories:
            history["invalidated_by_active_prefill_recompute"] = True
            history["invalidated_ns"] = int(now_ns)
            if not history.get("committed", False):
                history["preempted_before_commit"] = True

        # Validate every fallible ownership identity before either free. The
        # two Python mutations below are the atomic simulator transition.
        if prefill_owned:
            prefill.memory.free(prefill_owned, Device.NPU)
        if decode_owned:
            decode.memory.free(decode_owned, Device.NPU)
        discarded_tokens = request.begin_active_prefill_recompute()
        restored_discarded_delta = (
            int(request
                .agentic_kv_restored_tokens_discarded_by_active_prefill_recompute)
            - restored_discarded_before)

        request.agentic_kv_owner_instance_id = None
        request.agentic_kv_retained_instance_id = None
        request.agentic_kv_retained_per_rank_bytes = 0
        request.agentic_kv_overlap_cutoff_tokens = None
        request.agentic_kv_async_decode_join = False
        request.pd_prefill_initial_restored_per_rank_bytes = 0
        request.pd_prefill_reserved_per_rank_bytes = 0
        request.pd_prefill_owned_per_rank_bytes = 0
        request.pd_prefill_preallocated_per_rank_bytes = 0
        request.pd_decode_reserved_per_rank_bytes = 0
        request.pd_decode_owned_per_rank_bytes = 0
        request.pd_chunk_claim_pending = False
        request.pd_chunk_admitted_tokens = 0
        request.pd_chunk_admission_target_tokens = 0
        request.pd_prefill_handoff_released_per_rank_bytes = 0
        request.pd_decode_handoff_owned_per_rank_bytes = 0
        request.pd_restored_prefix_handoff_pending_tokens = 0
        request.pd_restored_prefix_handoff_sent_tokens = 0
        request.pd_new_kv_handoff_sent_tokens = 0

        prefill.active_recompute_preemptions += 1
        prefill.active_recompute_tokens += discarded_tokens
        self.agentic_kv_manager.metrics.pd_active_prefill_recompute_preemptions += 1
        self.agentic_kv_manager.metrics.pd_active_prefill_recompute_tokens += (
            discarded_tokens)
        self.agentic_kv_manager.metrics.agentic_kv_restored_tokens_discarded_by_active_prefill_recompute += (
            restored_discarded_delta)
        prefill.logger.info(
            "P/D active-prefill preemption of request #%d with %d-token "
            "recomputation", request.id, discarded_tokens)
        progress_events = getattr(
            self.agentic_kv_manager, "events", None)
        if progress_events is not None:
            progress_events.append({
                "time_ns": int(now_ns),
                "event": "pd_active_prefill_recompute_preempt",
                "session_id": request.session_id,
                "request_id": int(request.id),
                "prefill_instance_id": int(prefill.instance_id),
                "decode_instance_id": int(decode.instance_id),
                "discarded_tokens": discarded_tokens,
                "restored_hit_tokens_discarded": restored_discarded_delta,
                "cumulative_active_prefill_recompute_tokens": int(
                    request.active_prefill_recompute_tokens),
                "cumulative_restored_hit_tokens_discarded": int(
                    request
                    .agentic_kv_restored_tokens_discarded_by_active_prefill_recompute),
                "released_prefill_per_rank_bytes": prefill_owned,
                "released_decode_per_rank_bytes": decode_owned,
                "old_active_prefill_recompute_generation": old_generation,
                "new_active_prefill_recompute_generation": int(
                    request.pd_active_prefill_recompute_generation),
                "request_queue_position_preserved": True,
                "session_admission_slot_preserved": True,
            })
        return prefill_owned, decode_owned, discarded_tokens

    def _preempt_pd_prefills_for_progress(self, handoff, now_ns):
        """Free lower-priority partial prefills only when that admits head."""
        owner = handoff["request"]
        prefill = handoff["prefill_scheduler"]
        decode = handoff["decode_scheduler"]
        pair = (int(prefill.instance_id), int(decode.instance_id))
        queue = self._pending_pd_chunk_admissions.get(pair, ())
        if not queue or queue[0] is not handoff:
            raise RuntimeError(
                "P/D progress preemption must protect the same-pair FIFO "
                f"head: request={owner.id}, pair={pair}")

        # Another owner already has a real progress path. Do not destroy
        # active work to compete with its immutable reclaim reservation.
        if any(
                self.agentic_kv_manager.active_hbm_reclaim_claim(
                    scheduler.instance_id) is not None
                for scheduler in (prefill, decode)):
            return 0
        unreserved = getattr(
            self.agentic_kv_manager,
            "hbm_unreserved_per_rank_bytes",
            None,
        )
        if unreserved is None:
            # Compatibility managers cannot prove physical-versus-logical
            # slack, so they must retain the pre-existing fail-closed path.
            return 0

        owner_priority = self._pd_prefill_progress_priority(owner)
        protected_ids = getattr(
            prefill,
            "pd_chunk_admission_pass_protected_request_ids",
            (),
        )
        peers = [
            request for request in prefill.request
            if (request is not owner
                and int(request.id) not in protected_ids
                and request.pd_kv_ownership_state == "prefill_active"
                and request.pd_decode_target_instance_id == decode.instance_id)
        ]
        higher_priority = [
            request for request in peers
            if self._pd_prefill_progress_priority(request) < owner_priority
        ]
        if higher_priority:
            progress_events = getattr(
                self.agentic_kv_manager, "events", None)
            if progress_events is not None:
                progress_events.append({
                    "time_ns": int(now_ns),
                    "event": "pd_active_prefill_fifo_priority_inversion",
                    "request_id": int(owner.id),
                    "higher_priority_request_ids": [
                        int(request.id) for request in sorted(
                            higher_priority,
                            key=self._pd_prefill_progress_priority)
                    ],
                    "same_pair_fifo_preserved": True,
                })

        eligible = [
            request for request in peers
            if (int(request.num_computed_tokens) > 0
                and (int(request.pd_prefill_owned_per_rank_bytes) > 0
                     or int(request.pd_decode_owned_per_rank_bytes) > 0)
                and self._pd_prefill_victim_is_restore_ready(
                    request, prefill, decode, int(now_ns)))
        ]
        victim_priority = lambda request: (
            int(request.num_computed_tokens),
            -int(request.arrival),
            -int(request.ready_time),
            -int(request.id),
        )
        lower_priority = sorted(
            (
                request for request in eligible
                if self._pd_prefill_progress_priority(request)
                > owner_priority
            ),
            key=victim_priority,
        )
        # FIFO is the causal contract. If its head is not the most advanced
        # owner and lower-priority victims cannot release enough memory,
        # reclaim a more advanced peer as the explicit last resort rather
        # than silently reordering the admission queue or deadlocking.
        higher_priority_fallback = sorted(
            (
                request for request in eligible
                if self._pd_prefill_progress_priority(request)
                < owner_priority
            ),
            key=victim_priority,
        )
        candidates = lower_priority + higher_priority_fallback

        prefill_shortfall = max(
            0,
            int(handoff["prefill_needed_per_rank_bytes"])
            - int(unreserved(prefill.instance_id)),
        )
        decode_shortfall = max(
            0,
            int(handoff["decode_needed_per_rank_bytes"])
            - int(unreserved(decode.instance_id)),
        )
        selected = []
        released_prefill = 0
        released_decode = 0
        for victim in candidates:
            if (released_prefill >= prefill_shortfall
                    and released_decode >= decode_shortfall):
                break
            selected.append(victim)
            released_prefill += int(
                victim.pd_prefill_owned_per_rank_bytes)
            released_decode += int(victim.pd_decode_owned_per_rank_bytes)
        if (released_prefill < prefill_shortfall
                or released_decode < decode_shortfall):
            return 0

        fallback_victims = [
            victim for victim in selected
            if victim in higher_priority_fallback
        ]
        if fallback_victims:
            progress_events = getattr(
                self.agentic_kv_manager, "events", None)
            if progress_events is not None:
                progress_events.append({
                    "time_ns": int(now_ns),
                    "event": "pd_active_prefill_fifo_liveness_fallback",
                    "request_id": int(owner.id),
                    "victim_request_ids": [
                        int(victim.id) for victim in fallback_victims],
                    "same_pair_fifo_preserved": True,
                    "reason": (
                        "lower_priority_victims_cannot_admit_fifo_head"),
                })

        for victim in selected:
            self._preempt_one_pd_prefill(
                victim, prefill, decode, int(now_ns))
        return len(selected)

    def admit_pd_prefill_chunk(
            self, prefill_scheduler, request, chunk_tokens, now_ns):
        """Atomically admit one P chunk and its D receive blocks.

        ``False`` means only this request is gated. The scheduler filters it
        from the candidate batch and can continue dispatching fit-capable
        peers. A successful admission freezes ``chunk_tokens`` until the ASTRA
        graph commits it.
        """
        if request.pd_kv_ownership_state != "prefill_active":
            return True
        chunk_tokens = int(chunk_tokens)
        already_admitted = int(request.pd_chunk_admitted_tokens)
        if already_admitted:
            if already_admitted != chunk_tokens:
                raise RuntimeError(
                    "A frozen P/D chunk was resized before dispatch: "
                    f"request={request.id}, admitted={already_admitted}, "
                    f"proposed={chunk_tokens}")
            return True
        if self._pending_pd_chunk_for_request(request.id) is not None:
            return False

        requirements = self.pd_prefill_chunk_requirements(
            request, prefill_scheduler, chunk_tokens)
        pair = (
            requirements["prefill_instance_id"],
            requirements["decode_instance_id"],
        )
        handoff = {
            "request": request,
            "prefill_scheduler": prefill_scheduler,
            "decode_scheduler": self._scheduler_by_instance(pair[1]),
            "prefill_needed_per_rank_bytes": requirements[
                "prefill_delta_per_rank_bytes"],
            "decode_needed_per_rank_bytes": requirements[
                "decode_delta_per_rank_bytes"],
            "enqueued_ns": int(now_ns),
            "prefill_claim_ready_ns": None,
            "decode_claim_ready_ns": None,
            "last_pair_claim_attempt_state": None,
            "requirements": requirements,
        }
        request.pd_chunk_claim_pending = True
        request.pd_chunk_admission_enqueued_ns = int(now_ns)
        queue = self._pending_pd_chunk_admissions.setdefault(pair, [])
        queue.append(handoff)
        self._process_pending_pd_chunk_admissions(int(now_ns))
        return int(request.pd_chunk_admitted_tokens) == chunk_tokens

    def _finalize_pd_chunk_admission(self, handoff, now_ns):
        request = handoff["request"]
        prefill = handoff["prefill_scheduler"]
        decode = handoff["decode_scheduler"]
        request_id = int(request.id)
        requirements = handoff["requirements"]

        now_ns = int(now_ns)
        prefill_instance_id = int(prefill.instance_id)
        decode_instance_id = int(decode.instance_id)
        if prefill_instance_id == decode_instance_id:
            raise RuntimeError(
                "Atomic P/D chunk finalization requires distinct P and D "
                f"instances: request={request_id}, instance="
                f"{prefill.instance_id}")
        required_values = (
            "request_id", "active_prefill_recompute_generation",
            "prefill_instance_id", "decode_instance_id",
            "computed_tokens", "chunk_tokens", "target_tokens",
            "prefill_current_per_rank_bytes",
            "decode_current_per_rank_bytes",
            "prefill_target_per_rank_bytes",
            "decode_target_per_rank_bytes",
            "prefill_delta_per_rank_bytes",
            "decode_delta_per_rank_bytes",
            "prefill_unreserved_per_rank_bytes",
            "decode_unreserved_per_rank_bytes",
        )
        missing = [key for key in required_values if key not in requirements]
        if missing:
            raise RuntimeError(
                "P/D chunk finalization requirements are incomplete: "
                f"request={request_id}, missing={missing}")
        values = {key: int(requirements[key]) for key in required_values}
        if any(value < 0 for value in values.values()):
            raise RuntimeError(
                "P/D chunk finalization requirements contain a negative "
                f"value: request={request_id}, values={values}")
        if (values["request_id"] != request_id
                or values["prefill_instance_id"] != prefill_instance_id
                or values["decode_instance_id"] != decode_instance_id
                or values["active_prefill_recompute_generation"]
                != int(request.pd_active_prefill_recompute_generation)):
            raise RuntimeError(
                "P/D chunk finalization identity or generation changed: "
                f"request={request_id}, values={values}")
        prefill_current = values["prefill_current_per_rank_bytes"]
        decode_current = values["decode_current_per_rank_bytes"]
        prefill_target = values["prefill_target_per_rank_bytes"]
        decode_target = values["decode_target_per_rank_bytes"]
        prefill_delta = values["prefill_delta_per_rank_bytes"]
        decode_delta = values["decode_delta_per_rank_bytes"]
        chunk_tokens = values["chunk_tokens"]
        target_tokens = values["target_tokens"]
        computed_tokens = values["computed_tokens"]
        if prefill_target != decode_target:
            raise RuntimeError(
                "P/D chunk block parity failed before allocation: "
                f"request={request_id}, prefill={prefill_target}, "
                f"decode={decode_target}")
        if (prefill_current + prefill_delta != prefill_target
                or decode_current + decode_delta != decode_target
                or int(handoff["prefill_needed_per_rank_bytes"])
                != prefill_delta
                or int(handoff["decode_needed_per_rank_bytes"])
                != decode_delta):
            raise RuntimeError(
                "P/D chunk byte conservation failed before allocation: "
                f"request={request_id}, P={prefill_current}+"
                f"{prefill_delta}/{prefill_target}, D={decode_current}+"
                f"{decode_delta}/{decode_target}")
        if (chunk_tokens <= 0
                or computed_tokens != int(request.num_computed_tokens)
                or target_tokens != computed_tokens + chunk_tokens
                or target_tokens > int(request.prefill_target_tokens)):
            raise RuntimeError(
                "P/D chunk token target changed before allocation: "
                f"request={request_id}, computed={computed_tokens}/"
                f"{request.num_computed_tokens}, chunk={chunk_tokens}, "
                f"target={target_tokens}/{request.prefill_target_tokens}")
        if (not request.pd_chunk_claim_pending
                or int(request.pd_chunk_admitted_tokens) != 0
                or int(request.pd_chunk_admission_target_tokens) != 0
                or int(request.pd_prefill_owned_per_rank_bytes)
                != prefill_current
                or int(request.pd_decode_owned_per_rank_bytes)
                != decode_current):
            raise RuntimeError(
                "P/D request state changed before atomic chunk allocation: "
                f"request={request_id}")

        enqueued_ns = int(handoff["enqueued_ns"])
        restore_ready_ns = int(request.agentic_kv_restore_ready_time_ns)
        if (handoff["prefill_claim_ready_ns"] is None
                or handoff["decode_claim_ready_ns"] is None):
            raise RuntimeError(
                "P/D chunk lost a capacity-ready timestamp before atomic "
                f"allocation: request={request_id}")
        prefill_ready_ns = int(handoff["prefill_claim_ready_ns"])
        decode_ready_ns = int(handoff["decode_claim_ready_ns"])
        if (enqueued_ns < 0 or restore_ready_ns < 0
                or prefill_ready_ns < enqueued_ns
                or decode_ready_ns < enqueued_ns
                or now_ns < max(
                    enqueued_ns, prefill_ready_ns, decode_ready_ns)):
            raise RuntimeError(
                "P/D chunk admission timestamps are inconsistent before "
                f"allocation: request={request_id}, enqueue={enqueued_ns}, "
                f"P={prefill_ready_ns}, D={decode_ready_ns}, now={now_ns}")

        self.agentic_kv_manager.advance(now_ns)
        expected_claims = {}
        for role, scheduler, needed_key, ready_key in (
                self._pd_handoff_claim_roles(handoff)):
            needed = int(handoff[needed_key])
            if needed == 0:
                continue
            claim = self.agentic_kv_manager.active_hbm_reclaim_claim(
                scheduler.instance_id)
            expected_owner = ("pd", request_id)
            if (claim is None
                    or int(claim.instance_id) != int(scheduler.instance_id)
                    or (claim.owner_kind, claim.owner_id) != expected_owner
                    or int(claim.per_rank_bytes) != needed
                    or now_ns < int(claim.ready_ns)):
                raise RuntimeError(
                    f"Ready P/D {role} chunk lost its exact HBM claim "
                    "during atomic preflight: "
                    f"request={request_id}, instance="
                    f"{scheduler.instance_id}, needed={needed}, claim="
                    f"{claim}")
            physical_headroom = (
                int(scheduler.memory.npu_mem)
                - int(scheduler.memory.npu_used))
            if physical_headroom < needed:
                raise RuntimeError(
                    f"Ready P/D {role} chunk exceeds physical allocator "
                    f"headroom during atomic preflight: request={request_id}, "
                    f"instance={scheduler.instance_id}, needed={needed}, "
                    f"headroom={physical_headroom}")
            expected_claims[role] = claim

        projected_prefill_peak = (
            int(prefill.memory.npu_used) + prefill_delta)
        projected_decode_peak = (
            int(decode.memory.npu_used) + decode_delta)
        admitted_ns = now_ns
        wait_ns = admitted_ns - enqueued_ns
        critical_wait_ns = max(
            0, admitted_ns - max(restore_ready_ns, enqueued_ns))
        chunk_admission = {
            **requirements,
            "enqueued_ns": enqueued_ns,
            "prefill_capacity_ready_ns": prefill_ready_ns,
            "decode_capacity_ready_ns": decode_ready_ns,
            "admitted_ns": admitted_ns,
            "wait_ns": wait_ns,
            "critical_wait_after_restore_ns": critical_wait_ns,
            "prefill_peak_hbm_used_per_rank_bytes": projected_prefill_peak,
            "decode_peak_hbm_used_per_rank_bytes": projected_decode_peak,
        }
        record_chunk = getattr(
            self.agentic_kv_manager, "record_pd_chunk_admission", None)
        validate_chunk = getattr(
            self.agentic_kv_manager, "validate_pd_chunk_admission", None)
        if record_chunk is not None:
            if validate_chunk is None:
                raise RuntimeError(
                    "P/D chunk recorder lacks side-effect-free atomic "
                    "validation")
            validate_chunk(request, chunk_admission)
        record_prefill_admission = (
            self.agentic_kv_manager.record_pd_prefill_admission)
        record_decode_admission = (
            self.agentic_kv_manager.record_pd_decode_receive_admission)
        record_launch_admission = (
            self.agentic_kv_manager.record_pd_launch_admission)
        record_async_restore_gate = getattr(
            self.agentic_kv_manager, "record_async_restore_gate", None)
        new_prefill_reserved = (
            int(request.pd_prefill_reserved_per_rank_bytes) + prefill_delta)
        new_decode_reserved = (
            int(request.pd_decode_reserved_per_rank_bytes) + decode_delta)
        new_scheduler_resource_ready_ns = max(
            int(request.scheduler_resource_ready_time_ns), admitted_ns)
        new_chunk_admission_count = (
            int(request.pd_chunk_admission_count) + 1)
        new_chunk_admitted_tokens_total = (
            int(request.pd_chunk_admitted_tokens_total) + chunk_tokens)
        new_prefill_admitted_bytes = (
            int(request.pd_chunk_prefill_admitted_per_rank_bytes)
            + prefill_delta)
        new_decode_admitted_bytes = (
            int(request.pd_chunk_decode_admitted_per_rank_bytes)
            + decode_delta)
        new_chunk_wait_total = (
            int(request.pd_chunk_admission_wait_ns_total) + wait_ns)
        new_chunk_critical_wait_total = (
            int(request.pd_chunk_admission_critical_wait_ns_total)
            + critical_wait_ns)
        new_successful_chunk_wait_total = (
            int(request.pd_chunk_successful_admission_wait_ns_total)
            + wait_ns)
        new_successful_chunk_critical_wait_total = (
            int(request
                .pd_chunk_successful_admission_critical_wait_ns_total)
            + critical_wait_ns)
        new_prefill_peak = max(
            int(request.pd_chunk_prefill_peak_hbm_used_per_rank_bytes),
            projected_prefill_peak,
        )
        new_decode_peak = max(
            int(request.pd_chunk_decode_peak_hbm_used_per_rank_bytes),
            projected_decode_peak,
        )
        first_chunk = new_chunk_admission_count == 1
        record_async_gate = bool(
            request.agentic_kv_async_decode_join
            and int(request.agentic_kv_restore_ns) > 0
            and request.agentic_kv_overlap_cutoff_tokens is None)
        if record_async_gate and record_async_restore_gate is None:
            raise RuntimeError(
                "Async P/D restore gate recorder is unavailable during "
                "atomic preflight")

        def consume_and_allocate(role, scheduler, needed_key):
            needed = int(handoff[needed_key])
            if needed == 0:
                return
            claim = self.agentic_kv_manager.consume_active_hbm_reclaim(
                scheduler.instance_id,
                now_ns,
                owner_kind="pd",
                owner_id=request_id,
            )
            if (claim is None
                    or claim is not expected_claims[role]
                    or int(claim.per_rank_bytes) != needed):
                raise RuntimeError(
                    f"Ready P/D {role} chunk lost its exact HBM claim: "
                    f"request={request_id}, instance="
                    f"{scheduler.instance_id}, required={needed}, "
                    f"observed={None if claim is None else claim.per_rank_bytes}")
            scheduler.decode_handoff_claim_pending = False
            scheduler.memory.allocate(needed, Device.NPU)

        consume_and_allocate(
            "prefill", prefill, "prefill_needed_per_rank_bytes")
        consume_and_allocate(
            "decode", decode, "decode_needed_per_rank_bytes")

        request.pd_prefill_owned_per_rank_bytes = prefill_target
        request.pd_decode_owned_per_rank_bytes = decode_target
        request.pd_prefill_preallocated_per_rank_bytes = prefill_target
        request.pd_prefill_reserved_per_rank_bytes = new_prefill_reserved
        request.pd_decode_reserved_per_rank_bytes = new_decode_reserved
        request.agentic_kv_owner_instance_id = (
            prefill_instance_id if prefill_target else None)
        request.pd_chunk_claim_pending = False
        request.pd_chunk_admitted_tokens = chunk_tokens
        request.pd_chunk_admission_target_tokens = target_tokens
        request.pd_chunk_prefill_capacity_ready_ns = prefill_ready_ns
        request.pd_chunk_decode_capacity_ready_ns = decode_ready_ns
        request.pd_chunk_admission_ready_ns = admitted_ns
        request.pd_chunk_admission_wait_ns = wait_ns
        request.pd_chunk_admission_critical_wait_ns = critical_wait_ns
        request.scheduler_resource_ready_time_ns = (
            new_scheduler_resource_ready_ns)
        request.pd_chunk_admission_count = new_chunk_admission_count
        request.pd_chunk_admitted_tokens_total = (
            new_chunk_admitted_tokens_total)
        request.pd_chunk_prefill_admitted_per_rank_bytes = (
            new_prefill_admitted_bytes)
        request.pd_chunk_decode_admitted_per_rank_bytes = (
            new_decode_admitted_bytes)
        request.pd_chunk_admission_wait_ns_total = new_chunk_wait_total
        request.pd_chunk_admission_critical_wait_ns_total = (
            new_chunk_critical_wait_total)
        request.pd_chunk_successful_admission_wait_ns_total = (
            new_successful_chunk_wait_total)
        request.pd_chunk_successful_admission_critical_wait_ns_total = (
            new_successful_chunk_critical_wait_total)
        request.pd_chunk_prefill_peak_hbm_used_per_rank_bytes = (
            new_prefill_peak)
        request.pd_chunk_decode_peak_hbm_used_per_rank_bytes = (
            new_decode_peak)
        request.pd_chunk_admission_history.append(chunk_admission)
        if first_chunk:
            request.pd_prefill_capacity_ready_ns = prefill_ready_ns
            request.pd_prefill_capacity_wait_ns = max(
                0, prefill_ready_ns - enqueued_ns)
            request.pd_prefill_admission_ready_ns = admitted_ns
            request.pd_prefill_admission_wait_ns = wait_ns
            request.pd_prefill_admission_critical_wait_ns = max(
                0, prefill_ready_ns - max(restore_ready_ns, enqueued_ns))
            request.pd_decode_capacity_ready_ns = decode_ready_ns
            request.pd_decode_capacity_wait_ns = max(
                0, decode_ready_ns - enqueued_ns)
            request.pd_decode_admission_ready_ns = admitted_ns
            request.pd_decode_admission_wait_ns = wait_ns
            request.pd_decode_admission_critical_wait_ns = max(
                0, decode_ready_ns - max(restore_ready_ns, enqueued_ns))
            request.pd_launch_admission_ready_ns = admitted_ns
            request.pd_launch_admission_wait_ns = wait_ns
            request.pd_launch_admission_critical_wait_ns = critical_wait_ns
            if record_async_gate:
                record_async_restore_gate(request, admitted_ns)
            record_prefill_admission(
                request,
                prefill.instance_id,
                enqueued_ns,
                prefill_ready_ns,
                admitted_ns,
                restore_ready_ns,
                prefill_delta,
                chunk_admission,
            )
            record_decode_admission(
                request,
                decode.instance_id,
                enqueued_ns,
                decode_ready_ns,
                admitted_ns,
                restore_ready_ns,
                decode_delta,
                chunk_admission,
            )
            record_launch_admission(
                request, enqueued_ns, admitted_ns, restore_ready_ns)

        if record_chunk is not None:
            record_chunk(
                request,
                dict(chunk_admission),
            )

    def _attempt_pd_chunk_head(
            self, handoff, now_ns,
            allow_active_prefill_preemption=True):
        prefill = handoff["prefill_scheduler"]
        decode = handoff["decode_scheduler"]
        request_id = int(handoff["request"].id)
        if self.agentic_kv_manager.synchronous_swap_enabled:
            boundary_instances = []
            for _, scheduler, needed_key, ready_key in (
                    self._pd_handoff_claim_roles(handoff)):
                if handoff[ready_key] is not None:
                    continue
                if self.agentic_kv_manager.synchronous_hbm_reclaim_needs_boundary(
                        scheduler.instance_id,
                        int(handoff[needed_key]),
                        int(now_ns)):
                    boundary_instances.append(scheduler.instance_id)
            if boundary_instances:
                self.agentic_kv_manager.acquire_synchronous_prepare_lock(
                    request_id,
                    boundary_instances,
                    session_id=handoff["request"].session_id,
                )
            else:
                self.agentic_kv_manager.release_synchronous_prepare_lock(
                    request_id)
            if any(
                    self._scheduler_by_instance(instance_id).inflight
                    for instance_id in boundary_instances):
                return False

        pair_state = self._pd_handoff_capacity_state(handoff, now_ns)
        pair_incomplete = any(
            handoff[ready_key] is None
            for _, _, _, ready_key in self._pd_handoff_claim_roles(handoff)
        )
        if pair_incomplete and (
                handoff.get("last_pair_claim_attempt_state") != pair_state):
            handoff["last_pair_claim_attempt_state"] = pair_state
            for _, scheduler, needed_key, ready_key in (
                    self._pd_handoff_claim_roles(handoff)):
                needed = int(handoff[needed_key])
                if needed == 0:
                    handoff[ready_key] = int(handoff["enqueued_ns"])
                    continue
                if handoff[ready_key] is not None:
                    continue
                ready_ns = self.agentic_kv_manager.claim_active_hbm_reclaim(
                    scheduler.instance_id,
                    needed,
                    int(now_ns),
                    owner_kind="pd",
                    owner_id=request_id,
                )
                if ready_ns is not None:
                    handoff[ready_key] = int(ready_ns)
                    scheduler.decode_handoff_claim_pending = True

            if any(
                    handoff[ready_key] is None
                    for _, _, _, ready_key in (
                        self._pd_handoff_claim_roles(handoff))):
                self._rollback_partial_pd_hbm_claims(handoff, now_ns)
                handoff["last_pair_claim_attempt_state"] = (
                    self._pd_handoff_capacity_state(handoff, now_ns))
                self.agentic_kv_manager.release_synchronous_prepare_lock(
                    request_id)
                if (allow_active_prefill_preemption
                        and self._preempt_pd_prefills_for_progress(
                            handoff, int(now_ns)) > 0):
                    # The FIFO head is unchanged. Re-evaluate both atomic
                    # claims exactly once against the newly freed P/D bytes.
                    handoff["last_pair_claim_attempt_state"] = None
                    return self._attempt_pd_chunk_head(
                        handoff,
                        int(now_ns),
                        allow_active_prefill_preemption=False,
                    )
                return False

        ready_values = [
            handoff[ready_key]
            for _, _, _, ready_key in self._pd_handoff_claim_roles(handoff)
        ]
        if any(value is None for value in ready_values):
            self.agentic_kv_manager.release_synchronous_prepare_lock(
                request_id)
            return False
        if int(now_ns) < max(int(value) for value in ready_values):
            self.agentic_kv_manager.release_synchronous_prepare_lock(
                request_id)
            return False
        self._finalize_pd_chunk_admission(handoff, int(now_ns))
        self.agentic_kv_manager.release_synchronous_prepare_lock(request_id)
        prefill.decode_handoff_claim_pending = False
        decode.decode_handoff_claim_pending = False
        return True

    def _process_pending_pd_chunk_admissions(self, now_ns):
        admitted = 0
        for pair in sorted(list(self._pending_pd_chunk_admissions)):
            queue = self._pending_pd_chunk_admissions[pair]
            while queue:
                if not self._attempt_pd_chunk_head(queue[0], int(now_ns)):
                    break
                queue.pop(0)
                admitted += 1
            if not queue:
                del self._pending_pd_chunk_admissions[pair]
        return admitted

    def _process_pending_full_model_hbf_prefill_launches(
            self, now_ns):
        """Launch FIFO P requests only after their finite D HBM is reserved."""

        if self.full_model_hbf_adapter is None:
            if self._pending_full_model_hbf_prefill_launches:
                raise RuntimeError(
                    "full-model HBF P/D launch exists without an adapter")
            return 0
        launched = 0
        while self._pending_full_model_hbf_prefill_launches:
            pending = self._pending_full_model_hbf_prefill_launches[0]
            request = pending["request"]
            prefill = pending["prefill_scheduler"]
            decode_id = int(pending["decode_instance_id"])
            reserved = (
                self.full_model_hbf_gpu_hbm_bridge
                .try_reserve_pd_decode(
                    request,
                    prefill_instance_id=int(prefill.instance_id),
                    decode_instance_id=decode_id,
                )
            )
            while not reserved:
                reclaim_audit = (
                    self.full_model_hbf_adapter
                    .reclaim_gpu_ready_for_hbm_pressure(
                        gpu_instance_id=decode_id,
                        now_ns=int(now_ns),
                    )
                )
                if reclaim_audit is None:
                    break
                applied = self.drain_full_model_hbf_gpu_hbm_events()
                if len(applied) != 1:
                    raise RuntimeError(
                        "one GPU-ready pressure reclaim must apply exactly "
                        "one finite-HBM ownership event: "
                        f"audit={reclaim_audit}, applied={len(applied)}")
                reserved = (
                    self.full_model_hbf_gpu_hbm_bridge
                    .try_reserve_pd_decode(
                        request,
                        prefill_instance_id=int(prefill.instance_id),
                        decode_instance_id=decode_id,
                    )
                )
            if not reserved:
                break
            prefill.enqueue_request(request)
            self._pending_full_model_hbf_prefill_launches.pop(0)
            launched += 1
        return launched

    def transfer_prefill_request(self, requests, current_time_ns=None):
        completed_at_handoff = []
        for req in requests:
            if not self.decode_schedulers:
                raise RuntimeError(
                    "A prefill request completed but no decode instance is configured")

            # Without session-KV tiering there are no idle whole-session
            # objects or logical reservations to coordinate with. Keep the
            # legacy immediate handoff path. ``current_time_ns is None`` is a
            # compatibility path for callers outside the simulation loop.
            if (
                self.full_model_hbf_adapter is not None
                and self.full_model_hbf_gpu_hbm_bridge.topology == "pd"
                and current_time_ns is not None
            ):
                decode_id = self._full_model_hbf_pd_decode_by_prefill.get(
                    int(req.instance_id))
                if decode_id is None:
                    raise RuntimeError(
                        "full-model HBF P completion changed its fixed pair")
                sched = self._scheduler_by_instance(decode_id)
                reservation = (
                    self.full_model_hbf_gpu_hbm_bridge
                    .pd_decode_reservation(req)
                )
                if reservation is None:
                    raise RuntimeError(
                        "full-model HBF P completion lacks its pre-reserved "
                        f"D HBM: request={req.id}")
                completed = sched.add_decode(
                    req,
                    preallocated_hbm_bytes=(
                        reservation.reserved_per_rank_bytes),
                    completion_time_ns=current_time_ns,
                )
                self.full_model_hbf_gpu_hbm_bridge\
                    .consume_pd_decode_reservation(req)
                if completed is not None:
                    completed_at_handoff.append(completed)
                continue

            if (self.agentic_kv_manager is None
                    or current_time_ns is None):
                session_id = req.session_id
                sched = None
                if session_id is not None:
                    affinity = self._session_decode_affinity.get(session_id)
                    sched = next(
                        (candidate for candidate in self.decode_schedulers
                         if candidate.instance_id == affinity), None)
                if sched is None:
                    selected = self._select_instance(
                        self.decode_schedulers, "decode")
                    sched = self.decode_schedulers[selected]
                if session_id is not None:
                    self._session_decode_affinity[session_id] = (
                        sched.instance_id)
                completed = sched.add_decode(
                    req, completion_time_ns=current_time_ns)
                if completed is not None:
                    completed_at_handoff.append(completed)
                continue

            target_id = req.pd_decode_target_instance_id
            if target_id is None:
                raise RuntimeError(
                    "P/D prefill completed without a pre-admitted decode "
                    "receive allocation")
            sched = self._scheduler_by_instance(target_id)
            completed = sched.add_decode(
                req,
                preallocated_hbm_bytes=(
                    req.pd_decode_reserved_per_rank_bytes),
                completion_time_ns=current_time_ns,
            )
            if completed is not None:
                completed_at_handoff.append(completed)
        return completed_at_handoff

    def process_pending_decode_handoffs(self, current_time_ns):
        """Reserve complete P and D HBM, then launch eligible P requests.

        There is at most one manager active-HBM claim per instance. Admissions
        are FIFO within a P/D pair, while independent pairs can progress at
        the same timestamp. Physical allocation precedes P scheduler
        visibility and therefore every P->D send.
        """
        full_model_launched = (
            self._process_pending_full_model_hbf_prefill_launches(
                current_time_ns)
        )
        if self.agentic_kv_manager is None:
            if self.has_pending_decode_handoffs():
                if self._pending_full_model_hbf_prefill_launches:
                    return full_model_launched
                raise RuntimeError(
                    "Pending P/D decode handoff exists without an agentic "
                    "KV manager")
            return full_model_launched

        now_ns = int(current_time_ns)
        self.agentic_kv_manager.advance(now_ns)
        self._process_pending_pd_chunk_admissions(now_ns)
        launched = full_model_launched
        for instance_id in sorted(list(self._pending_decode_handoffs)):
            queue = self._pending_decode_handoffs[instance_id]
            while queue:
                handoff = queue[0]
                prefill_sched = handoff["prefill_scheduler"]
                decode_sched = handoff["decode_scheduler"]
                request_id = int(handoff["request"].id)
                if self.agentic_kv_manager.synchronous_swap_enabled:
                    boundary_instances = []
                    for sched, needed_key, ready_key in (
                            (
                                prefill_sched,
                                "prefill_needed_per_rank_bytes",
                                "prefill_claim_ready_ns",
                            ),
                            (
                                decode_sched,
                                "decode_needed_per_rank_bytes",
                                "decode_claim_ready_ns",
                            )):
                        if handoff[ready_key] is not None:
                            continue
                        if (
                            self.agentic_kv_manager
                            .synchronous_hbm_reclaim_needs_boundary(
                                sched.instance_id,
                                int(handoff[needed_key]),
                                now_ns,
                            )
                        ):
                            boundary_instances.append(sched.instance_id)
                    if boundary_instances:
                        self.agentic_kv_manager.acquire_synchronous_prepare_lock(
                            request_id,
                            boundary_instances,
                            session_id=handoff["request"].session_id,
                        )
                    else:
                        self.agentic_kv_manager.release_synchronous_prepare_lock(
                            request_id)
                    if any(
                            self._scheduler_by_instance(candidate).inflight
                            for candidate in boundary_instances):
                        break

                claim_roles = self._pd_handoff_claim_roles(handoff)
                pair_incomplete = any(
                    handoff[ready_key] is None
                    for _, _, _, ready_key in claim_roles
                )
                if pair_incomplete:
                    pair_state = self._pd_handoff_capacity_state(
                        handoff, now_ns)
                    if (handoff.get("last_pair_claim_attempt_state")
                            == pair_state):
                        self.agentic_kv_manager.release_synchronous_prepare_lock(
                            request_id)
                        break
                    handoff["last_pair_claim_attempt_state"] = pair_state

                    for role, sched, needed_key, ready_key in claim_roles:
                        needed_per_rank = int(handoff[needed_key])
                        if needed_per_rank == 0:
                            handoff[ready_key] = int(handoff["enqueued_ns"])
                            continue
                        if handoff[ready_key] is not None:
                            continue
                        claim_ready_ns = (
                            self.agentic_kv_manager.claim_active_hbm_reclaim(
                                sched.instance_id,
                                needed_per_rank,
                                now_ns,
                                owner_kind="pd",
                                owner_id=request_id,
                            )
                        )
                        if claim_ready_ns is None:
                            continue
                        handoff[ready_key] = int(claim_ready_ns)
                        sched.decode_handoff_claim_pending = True

                    pair_incomplete = any(
                        handoff[ready_key] is None
                        for _, _, _, ready_key in claim_roles
                    )
                    if pair_incomplete:
                        # Pair admission is atomic. Holding one engine while
                        # the other waits can prevent its ready active request
                        # from growing or being preempted, leaving neither a
                        # runnable batch nor a future capacity event. Useful
                        # demotions continue, but their logical reservation is
                        # released until either engine's exact capacity state
                        # changes.
                        self._rollback_partial_pd_hbm_claims(
                            handoff, now_ns)
                        handoff["last_pair_claim_attempt_state"] = (
                            self._pd_handoff_capacity_state(
                                handoff, now_ns))
                        self.agentic_kv_manager.release_synchronous_prepare_lock(
                            request_id)
                        break

                prefill_ready_ns = handoff["prefill_claim_ready_ns"]
                decode_ready_ns = handoff["decode_claim_ready_ns"]
                if prefill_ready_ns is None or decode_ready_ns is None:
                    self.agentic_kv_manager.release_synchronous_prepare_lock(
                        request_id)
                    break
                admission_ready_ns = max(
                    int(prefill_ready_ns), int(decode_ready_ns))
                if now_ns < admission_ready_ns:
                    # HBM reclaim transfers now own explicit engine barriers.
                    # The temporary iteration-boundary lock is no longer
                    # needed while those barriers carry the dependency.
                    self.agentic_kv_manager.release_synchronous_prepare_lock(
                        request_id)
                    break

                def consume_and_allocate(role, sched, needed_per_rank):
                    needed_per_rank = int(needed_per_rank)
                    if needed_per_rank == 0:
                        return
                    claim = (
                        self.agentic_kv_manager.consume_active_hbm_reclaim(
                            sched.instance_id,
                            now_ns,
                            owner_kind="pd",
                            owner_id=request_id,
                        )
                    )
                    if claim is None:
                        raise RuntimeError(
                            f"Ready P/D {role} HBM admission lost its "
                            f"manager claim: instance={sched.instance_id}")
                    if int(claim.per_rank_bytes) != needed_per_rank:
                        raise RuntimeError(
                            f"P/D {role} HBM claim size mismatch: "
                            f"instance={sched.instance_id}, "
                            f"claimed={claim.per_rank_bytes}, "
                            f"required={needed_per_rank}")
                    # Consume releases the logical reservation. Allocate
                    # synchronously before another manager operation can see
                    # the same slack.
                    sched.decode_handoff_claim_pending = False
                    sched.memory.allocate(
                        claim.per_rank_bytes, Device.NPU)

                consume_and_allocate(
                    "prefill",
                    prefill_sched,
                    handoff["prefill_needed_per_rank_bytes"],
                )
                consume_and_allocate(
                    "decode",
                    decode_sched,
                    handoff["decode_needed_per_rank_bytes"],
                )
                queue.pop(0)
                self.agentic_kv_manager.release_synchronous_prepare_lock(
                    request_id)
                pair = (
                    int(prefill_sched.instance_id),
                    int(decode_sched.instance_id),
                )
                if self._pd_admission_owner.get(pair) == request_id:
                    del self._pd_admission_owner[pair]

                req = handoff["request"]
                restore_ready_ns = int(
                    req.agentic_kv_restore_ready_time_ns)
                admitted_ns = int(now_ns)
                req.pd_prefill_capacity_ready_ns = int(prefill_ready_ns)
                req.pd_prefill_capacity_wait_ns = max(
                    0,
                    int(prefill_ready_ns) - int(handoff["enqueued_ns"]),
                )
                req.pd_prefill_admission_ready_ns = admitted_ns
                req.pd_prefill_admission_wait_ns = max(
                    0,
                    admitted_ns - int(handoff["enqueued_ns"]),
                )
                req.pd_prefill_admission_critical_wait_ns = max(
                    0,
                    int(prefill_ready_ns) - max(
                        restore_ready_ns, int(handoff["enqueued_ns"])),
                )
                req.pd_prefill_preallocated_per_rank_bytes = int(
                    req.pd_prefill_full_per_rank_bytes)
                req.pd_decode_capacity_ready_ns = int(decode_ready_ns)
                req.pd_decode_capacity_wait_ns = max(
                    0,
                    int(decode_ready_ns) - int(handoff["enqueued_ns"]),
                )
                req.pd_decode_admission_ready_ns = admitted_ns
                req.pd_decode_admission_wait_ns = max(
                    0,
                    admitted_ns - int(handoff["enqueued_ns"]),
                )
                req.pd_decode_admission_critical_wait_ns = max(
                    0,
                    int(decode_ready_ns) - max(
                        restore_ready_ns, int(handoff["enqueued_ns"])
                    ),
                )
                req.pd_launch_admission_ready_ns = admitted_ns
                req.pd_launch_admission_wait_ns = max(
                    0, admitted_ns - int(handoff["enqueued_ns"]))
                req.pd_launch_admission_critical_wait_ns = max(
                    0,
                    admitted_ns - max(
                        restore_ready_ns, int(handoff["enqueued_ns"])),
                )
                req.ready_time = max(
                    req.ready_time,
                    admitted_ns,
                )
                if (req.agentic_kv_async_decode_join
                        and req.agentic_kv_restore_ns > 0
                        and req.agentic_kv_overlap_cutoff_tokens is None):
                    # With no pre-join prompt work, strict P/D admission is an
                    # independent prerequisite that can hide restore latency.
                    # Record only the remaining request-local eligibility gate.
                    self.agentic_kv_manager.record_async_restore_gate(
                        req, admitted_ns)
                legacy_full_admission = {
                    "request_id": int(req.id),
                    "active_prefill_recompute_generation": int(
                        req.pd_active_prefill_recompute_generation),
                    "prefill_current_per_rank_bytes": max(
                        0,
                        int(req.pd_prefill_full_per_rank_bytes)
                        - int(handoff[
                            "prefill_needed_per_rank_bytes"]),
                    ),
                    "prefill_target_per_rank_bytes": int(
                        req.pd_prefill_full_per_rank_bytes),
                    "decode_current_per_rank_bytes": max(
                        0,
                        int(req.pd_decode_full_per_rank_bytes)
                        - int(handoff[
                            "decode_needed_per_rank_bytes"]),
                    ),
                    "decode_target_per_rank_bytes": int(
                        req.pd_decode_full_per_rank_bytes),
                }
                self.agentic_kv_manager.record_pd_prefill_admission(
                    req,
                    prefill_sched.instance_id,
                    int(handoff["enqueued_ns"]),
                    int(prefill_ready_ns),
                    admitted_ns,
                    restore_ready_ns,
                    int(handoff["prefill_needed_per_rank_bytes"]),
                    legacy_full_admission,
                )
                self.agentic_kv_manager.record_pd_decode_receive_admission(
                    req,
                    instance_id,
                    int(handoff["enqueued_ns"]),
                    int(decode_ready_ns),
                    admitted_ns,
                    restore_ready_ns,
                    int(handoff["decode_needed_per_rank_bytes"]),
                    legacy_full_admission,
                )
                self.agentic_kv_manager.record_pd_launch_admission(
                    req,
                    int(handoff["enqueued_ns"]),
                    admitted_ns,
                    restore_ready_ns,
                )
                self._pending_prefill_launches.append({
                    "request": req,
                    "prefill_scheduler": prefill_sched,
                })

            if not queue:
                del self._pending_decode_handoffs[instance_id]

        waiting_launches = []
        for launch in sorted(
                self._pending_prefill_launches,
                key=lambda item: (
                    item["request"].ready_time,
                    item["request"].id,
                )):
            req = launch["request"]
            if req.ready_time <= now_ns:
                launch["prefill_scheduler"].enqueue_request(req)
                launched += 1
            else:
                waiting_launches.append(launch)
        self._pending_prefill_launches = waiting_launches
        return launched

    def has_pending_decode_handoffs(self):
        return (
            any(self._pending_decode_handoffs.values())
            or bool(self._pending_full_model_hbf_prefill_launches)
            or bool(self._pending_prefill_launches)
            or any(self._pending_pd_chunk_admissions.values())
        )

    def get_next_decode_handoff_wakeup(self):
        ready_times = []
        for queue in self._pending_decode_handoffs.values():
            if not queue:
                continue
            handoff = queue[0]
            pair_ready = (
                handoff["prefill_claim_ready_ns"],
                handoff["decode_claim_ready_ns"],
            )
            if all(value is not None for value in pair_ready):
                ready_times.append(max(int(value) for value in pair_ready))
        ready_times.extend(
            int(launch["request"].ready_time)
            for launch in self._pending_prefill_launches
        )
        for queue in self._pending_pd_chunk_admissions.values():
            if not queue:
                continue
            handoff = queue[0]
            pair_ready = (
                handoff["prefill_claim_ready_ns"],
                handoff["decode_claim_ready_ns"],
            )
            if all(value is not None for value in pair_ready):
                ready_times.append(max(int(value) for value in pair_ready))
        return min(ready_times) if ready_times else None

    def _scheduler_by_instance(self, instance_id):
        for scheduler in self.schedulers:
            if scheduler.instance_id == instance_id:
                return scheduler
        raise KeyError(f"Unknown scheduler instance_id={instance_id}")

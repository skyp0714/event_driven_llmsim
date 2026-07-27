"""Finite-HBM P4D4 node with versioned CPU/SSD KV tiering.

The node composes three independently testable pieces:

* :class:`P4D4ServingPool` owns continuous-batched model execution.
* :class:`TieredPDKVLifecycle` is the sole P/D KV-capacity owner and owns
  lower-tier placement plus transfer queues.
* One shared :class:`ResourceCalendar` serializes physical compute and
  transfer resources without introducing a second HBM admission ledger.

Completed resume KV is retained on D until capacity pressure selects a
whole-session LRU victim.  A stable D hit needs only a D-to-P prepare and a
fresh-suffix P-to-D handoff.  CPU/SSD hits, and a D hit racing a demotion,
populate a new D destination and therefore hand off the full prompt.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .gpu_pd_latency import P4D4GPUHardware
from .gpu_pd_pool import P4D4ServingPool, PDServingRequest
from .gpu_pd_tier_lifecycle import (
    D_RESERVATION_FINAL_UPFRONT,
    D_RESERVATION_PROMPT_UPFRONT,
    PrepareTicket,
    RESTORE_EXECUTION_BULK,
    SUPPORTED_D_RESERVATION_POLICIES,
    SUPPORTED_TIER_POLICIES,
    SUPPORTED_RESTORE_EXECUTION_MODES,
    Tier,
    TierSessionState,
    TieredPDKVLifecycle,
)
from .hbf_full_model_lifecycle import ResourceCalendar


class TieredNodeDeadlockError(RuntimeError):
    """Raised when unfinished admitted/deferred work has no future event."""


class TieredCallState(str, Enum):
    PENDING = "pending"
    PREPARING = "preparing"
    EXECUTING = "executing"
    USER_COMPLETE = "user_complete"
    INTERNAL_COMPLETE = "internal_complete"


@dataclass
class TieredNodeCall:
    request_id: int
    session_id: str
    call_index: int
    release_ns: int
    input_tokens: int
    output_tokens: int
    prefix_reuse_tokens: int
    has_successor: bool
    state: TieredCallState = TieredCallState.PENDING
    operational_hit_tokens: Optional[int] = None
    prepare_id: Optional[int] = None
    prepare_source: Optional[Tier] = None
    prepare_start_ns: Optional[int] = None
    prepare_completion_ns: Optional[int] = None
    pool_request: Optional[PDServingRequest] = None
    user_completion_ns: Optional[int] = None
    internal_completion_ns: Optional[int] = None
    capacity_deferrals: int = 0

    def validate(self) -> None:
        for name in (
            "request_id",
            "call_index",
            "release_ns",
            "input_tokens",
            "output_tokens",
            "prefix_reuse_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.request_id < 0 or self.call_index < 0 or self.release_ns < 0:
            raise ValueError(
                "request_id/call_index/release_ns must be non-negative")
        if self.input_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("input/output tokens must be positive")
        if not 0 <= self.prefix_reuse_tokens <= self.input_tokens:
            raise ValueError(
                "prefix_reuse_tokens must be in 0..input_tokens")
        if self.input_tokens + self.output_tokens - 1 > 1_010_000:
            raise ValueError(
                "request output would exceed 1,010,000-token contract")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(self.has_successor, bool):
            raise ValueError("has_successor must be a boolean")
        if self.call_index == 0 and self.prefix_reuse_tokens:
            raise ValueError("first call cannot reuse an earlier prefix")
        if (
            self.state != TieredCallState.PENDING
            or self.operational_hit_tokens is not None
            or self.prepare_id is not None
            or self.prepare_source is not None
            or self.prepare_start_ns is not None
            or self.prepare_completion_ns is not None
            or self.pool_request is not None
            or self.user_completion_ns is not None
            or self.internal_completion_ns is not None
            or self.capacity_deferrals
        ):
            raise ValueError("submitted tiered call must be pristine")

    @property
    def ttft_ns(self) -> Optional[int]:
        if self.pool_request is None:
            return None
        return self.pool_request.ttft_ns

    @property
    def tpot_ns(self) -> Optional[float]:
        if self.pool_request is None:
            return None
        return self.pool_request.tpot_ns


@dataclass
class TieredSessionLineage:
    session_id: str
    last_call_index: int = -1
    active_request_id: Optional[int] = None
    ended: bool = False


@dataclass(frozen=True)
class PrepareCapacity:
    p_bytes_per_rank: int
    d_bytes_per_rank: int
    cpu_bounce_bytes: int
    hit_tokens: int
    source: Optional[Tier]
    stable_d_source: bool


@dataclass
class TieredNodeMetrics:
    session_restarts: int = 0
    submitted_calls: int = 0
    admitted_calls: int = 0
    user_completed_calls: int = 0
    internal_completed_calls: int = 0
    capacity_deferral_attempts: int = 0
    p_capacity_deferrals: int = 0
    d_capacity_deferrals: int = 0
    cpu_bounce_deferrals: int = 0
    d_reclamation_attempts: int = 0
    d_reclamation_jobs_started: int = 0
    d_reclamation_immediate_drops: int = 0
    stable_d_hits: int = 0
    lower_tier_hits: int = 0
    streaming_lower_tier_hits: int = 0
    recompute_resumes: int = 0
    recompute_tokens: int = 0
    context_shrink_calls: int = 0
    full_prefix_cap_calls: int = 0
    full_prompt_handoffs: int = 0
    fresh_suffix_handoffs: int = 0
    max_pending_calls: int = 0


class FiniteHBMTieredP4D4Node:
    """One finite-HBM P4D4 server under one of the baseline policies."""

    def __init__(
            self, *, repo_root: Path, hardware: P4D4GPUHardware,
            node_id: int, policy: str,
            resource_calendar: Optional[ResourceCalendar] = None,
            p_capacity_bytes_per_rank: Optional[int] = None,
            d_capacity_bytes_per_rank: Optional[int] = None,
            cpu_capacity_bytes: Optional[int] = None,
            ssd_capacity_bytes: Optional[int] = None,
            max_num_batched_tokens: int = 8_192,
            max_num_seqs: int = 128,
            p_max_num_seqs: Optional[int] = None,
            d_max_num_seqs: Optional[int] = None,
            max_prefill_chunk_tokens: int = 4_096,
            band: str = "central",
            restore_execution_mode: str = RESTORE_EXECUTION_BULK,
            d_reservation_policy: str = D_RESERVATION_FINAL_UPFRONT,
            validate_every_event: bool = True,
            retain_detailed_history: bool = True) -> None:
        if policy not in SUPPORTED_TIER_POLICIES:
            raise ValueError(f"unsupported tier policy {policy!r}")
        if d_reservation_policy not in (
                SUPPORTED_D_RESERVATION_POLICIES):
            raise ValueError(
                "d_reservation_policy must be one of "
                f"{SUPPORTED_D_RESERVATION_POLICIES}")
        if restore_execution_mode not in (
                SUPPORTED_RESTORE_EXECUTION_MODES):
            raise ValueError(
                "restore_execution_mode must be one of "
                f"{sorted(SUPPORTED_RESTORE_EXECUTION_MODES)}")
        if not isinstance(validate_every_event, bool):
            raise ValueError("validate_every_event must be a boolean")
        if not isinstance(retain_detailed_history, bool):
            raise ValueError(
                "retain_detailed_history must be a boolean")
        self.hardware = hardware
        self.node_id = node_id
        self.policy = policy
        self.restore_execution_mode = restore_execution_mode
        self.d_reservation_policy = d_reservation_policy
        self.validate_every_event = validate_every_event
        self.calendar = (
            resource_calendar
            if resource_calendar is not None else ResourceCalendar()
        )
        self.lifecycle = TieredPDKVLifecycle(
            hardware=hardware,
            node_id=node_id,
            policy=policy,
            calendar=self.calendar,
            p_capacity_bytes_per_rank=p_capacity_bytes_per_rank,
            d_capacity_bytes_per_rank=d_capacity_bytes_per_rank,
            cpu_capacity_bytes=cpu_capacity_bytes,
            ssd_capacity_bytes=ssd_capacity_bytes,
            restore_execution_mode=restore_execution_mode,
            d_reservation_policy=d_reservation_policy,
            validate_every_event=validate_every_event,
        )
        self.pool = P4D4ServingPool(
            repo_root=repo_root,
            hardware=hardware,
            node_id=node_id,
            resource_calendar=self.calendar,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            p_max_num_seqs=p_max_num_seqs,
            d_max_num_seqs=d_max_num_seqs,
            max_prefill_chunk_tokens=max_prefill_chunk_tokens,
            band=band,
            validate_every_event=validate_every_event,
            retain_detailed_history=retain_detailed_history,
        )
        self.calls: dict[int, TieredNodeCall] = {}
        self.sessions: dict[str, TieredSessionLineage] = {}
        self.metrics = TieredNodeMetrics()
        self.current_ns = 0
        self._pending_call_ids: deque[int] = deque()
        self._ready_call_ids: deque[int] = deque()
        self._user_completed_ids: deque[int] = deque()
        self._ticket_by_request: dict[int, PrepareTicket] = {}
        self._user_completion_seen: set[int] = set()
        self._handoff_completion_seen: set[int] = set()
        self._last_submitted_call_index: dict[str, int] = {}
        self._last_submitted_request_id: dict[str, int] = {}

    def _validate_capacity_contract(self, call: TieredNodeCall) -> None:
        p_bytes = self.hardware.kv_capacity_bytes_per_rank(
            call.input_tokens)
        if p_bytes > self.lifecycle.p_ledger.capacity_bytes:
            raise ValueError(
                "request prompt exceeds node P-HBM KV capacity")
        if call.output_tokens > 1 or call.has_successor:
            final_tokens = call.input_tokens + call.output_tokens - 1
            d_bytes = self.hardware.kv_capacity_bytes_per_rank(
                final_tokens)
            if d_bytes > self.lifecycle.d_ledger.capacity_bytes:
                raise ValueError(
                    "request final context exceeds node D-HBM KV capacity")
            if (
                call.has_successor
                and self.policy != "hbm_lru_recompute"
            ):
                aggregate_bytes = (
                    d_bytes * self.hardware.tp_size)
                if (
                    aggregate_bytes
                    > self.lifecycle.cpu_ledger.capacity_bytes
                ):
                    raise ValueError(
                        "retained context exceeds node CPU staging "
                        "capacity")
                if (
                    aggregate_bytes
                    > self.lifecycle.ssd_ledger.capacity_bytes
                ):
                    raise ValueError(
                        "retained context exceeds node SSD tier "
                        "capacity")

    def submit_many(
            self, calls: Iterable[TieredNodeCall], *,
            now_ns: int) -> None:
        values = list(calls)
        seen_ids = set()
        proposed_last = dict(self._last_submitted_call_index)
        proposed_predecessors = {
            session_id: self.calls[request_id]
            for session_id, request_id
            in self._last_submitted_request_id.items()
        }
        for call in values:
            call.validate()
            self._validate_capacity_contract(call)
            if call.release_ns != now_ns:
                raise ValueError(
                    "tiered calls must be submitted at logical release")
            if call.request_id in self.calls or call.request_id in seen_ids:
                raise ValueError(
                    f"duplicate request_id={call.request_id}")
            prior_index = proposed_last.get(call.session_id, -1)
            if call.call_index != prior_index + 1:
                raise ValueError(
                    "session calls must be submitted in contiguous order: "
                    f"session={call.session_id!r}, "
                    f"got={call.call_index}, expected={prior_index + 1}")
            predecessor = proposed_predecessors.get(call.session_id)
            if predecessor is not None:
                if not predecessor.has_successor:
                    raise ValueError(
                        "cannot submit a call after a terminal predecessor: "
                        f"session={call.session_id!r}, "
                        f"predecessor={predecessor.request_id}")
                if predecessor.user_completion_ns is None:
                    raise ValueError(
                        "successor cannot be submitted before its "
                        "predecessor is user-complete")
                if call.release_ns < predecessor.user_completion_ns:
                    raise ValueError(
                        "successor release cannot precede predecessor "
                        "user completion")
            lineage = self.sessions.get(call.session_id)
            if lineage is not None and lineage.ended:
                raise ValueError(
                    f"session {call.session_id!r} already ended")
            proposed_last[call.session_id] = call.call_index
            proposed_predecessors[call.session_id] = call
            seen_ids.add(call.request_id)

        self.advance(now_ns, defer_schedule=True)
        for call in values:
            self.calls[call.request_id] = call
            self.sessions.setdefault(
                call.session_id,
                TieredSessionLineage(session_id=call.session_id),
            )
            self._last_submitted_call_index[call.session_id] = (
                call.call_index)
            self._last_submitted_request_id[call.session_id] = (
                call.request_id)
            self._pending_call_ids.append(call.request_id)
            self.metrics.submitted_calls += 1
        self.metrics.max_pending_calls = max(
            self.metrics.max_pending_calls,
            len(self._pending_call_ids),
        )
        self.flush_scheduling(now_ns)

    def submit(self, call: TieredNodeCall, *, now_ns: int) -> None:
        self.submit_many((call,), now_ns=now_ns)

    def restart_ended_session(
            self, session_id: str, *, now_ns: int) -> None:
        """Restart one drained GPU-local lineage at call index zero."""

        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be non-empty")
        self.advance(now_ns, defer_schedule=True)
        try:
            lineage = self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown tiered session {session_id!r}") from exc
        if not lineage.ended or lineage.active_request_id is not None:
            raise RuntimeError(
                "only an inactive ended tiered session can restart")
        if any(
                self.calls[request_id].session_id == session_id
                for request_id in self._pending_call_ids
        ) or any(
                self.calls[request_id].session_id == session_id
                for request_id in self._ready_call_ids
        ):
            raise RuntimeError(
                "ended tiered session retains queued calls")
        self.lifecycle.restart_ended(
            session_id, now_ns=now_ns)
        lineage.last_call_index = -1
        lineage.ended = False
        self._last_submitted_call_index.pop(session_id, None)
        self._last_submitted_request_id.pop(session_id, None)
        self.metrics.session_restarts += 1
        if self.validate_every_event:
            self.assert_invariants()

    def _prepare_capacity(
            self, call: TieredNodeCall) -> PrepareCapacity:
        record = self.lifecycle.sessions.get(call.session_id)
        source = None
        if record is not None and record.primary_copy_id is not None:
            source = self.lifecycle.copies[record.primary_copy_id]
        source_tokens = 0 if source is None else source.tokens
        hit_tokens = min(
            call.prefix_reuse_tokens,
            source_tokens,
            call.input_tokens - 1,
        )
        effective_source = source if hit_tokens else None
        p_bytes = self.hardware.kv_capacity_bytes_per_rank(
            call.input_tokens)
        needs_d = call.output_tokens > 1 or call.has_successor
        # Mirror the lifecycle's admission sizing: prompt-upfront gates D
        # admission on the prompt KV only; decode growth is charged by the
        # lifecycle when the finished context is published.
        d_admission_tokens = (
            call.input_tokens + call.output_tokens - 1
            if self.d_reservation_policy
            == D_RESERVATION_FINAL_UPFRONT
            else call.input_tokens
        )
        d_target = (
            self.hardware.kv_capacity_bytes_per_rank(
                d_admission_tokens)
            if needs_d else 0
        )
        reusable_d_destination = bool(
            source is not None
            and source.tier == Tier.D
            and source.copy_id == record.primary_copy_id
            and source.pins == 0
        )
        stable_d_source = bool(
            hit_tokens
            and reusable_d_destination
        )
        d_bytes = (
            0 if not needs_d
            else max(0, d_target - source.byte_count)
            if reusable_d_destination
            else d_target
        )
        cpu_bounce = (
            self.hardware.kv_capacity_bytes_per_rank(hit_tokens)
            * self.hardware.tp_size
            if (
                effective_source is not None
                and effective_source.tier == Tier.SSD
            )
            else 0
        )
        return PrepareCapacity(
            p_bytes_per_rank=p_bytes,
            d_bytes_per_rank=d_bytes,
            cpu_bounce_bytes=cpu_bounce,
            hit_tokens=hit_tokens,
            source=(
                None if effective_source is None
                else effective_source.tier),
            stable_d_source=stable_d_source,
        )

    def _record_deferral(
            self, call: TieredNodeCall, *,
            p: bool = False, d: bool = False,
            cpu: bool = False) -> None:
        call.capacity_deferrals += 1
        self.metrics.capacity_deferral_attempts += 1
        self.metrics.p_capacity_deferrals += int(p)
        self.metrics.d_capacity_deferrals += int(d)
        self.metrics.cpu_bounce_deferrals += int(cpu)

    def _try_prepare(
            self, call: TieredNodeCall, *,
            now_ns: int) -> Optional[PrepareTicket]:
        capacity = self._prepare_capacity(call)
        p_blocked = (
            self.lifecycle.p_ledger.free_bytes
            < capacity.p_bytes_per_rank
        )
        cpu_blocked = (
            self.lifecycle.cpu_ledger.free_bytes
            < capacity.cpu_bounce_bytes
        )
        if p_blocked:
            # P releases are tied to already-scheduled handoffs.  Avoid
            # perturbing lower-tier placement until that first gate clears.
            self._record_deferral(call, p=True)
            return None
        if cpu_blocked:
            self._record_deferral(call, cpu=True)
            self.lifecycle.ensure_cpu_bounce_headroom(
                capacity.hit_tokens,
                now_ns=now_ns,
                protected_session=call.session_id,
            )
            return None
        if (
            self.lifecycle.d_ledger.free_bytes
            < capacity.d_bytes_per_rank
        ):
            self._record_deferral(call, d=True)
            self.metrics.d_reclamation_attempts += 1
            jobs_before = (
                self.lifecycle.metrics.d_to_cpu_started
                + self.lifecycle.metrics.d_to_ssd_started
            )
            drops_before = self.lifecycle.metrics.d_drops
            retry_ns = self.lifecycle.ensure_d_headroom(
                capacity.d_bytes_per_rank,
                now_ns=now_ns,
                exclude_session=call.session_id,
            )
            jobs_after = (
                self.lifecycle.metrics.d_to_cpu_started
                + self.lifecycle.metrics.d_to_ssd_started
            )
            self.metrics.d_reclamation_jobs_started += (
                jobs_after - jobs_before)
            self.metrics.d_reclamation_immediate_drops += (
                self.lifecycle.metrics.d_drops - drops_before)
            if (
                retry_ns == now_ns
                and self.lifecycle.d_ledger.free_bytes
                >= capacity.d_bytes_per_rank
            ):
                capacity = self._prepare_capacity(call)
                if (
                    self.lifecycle.p_ledger.free_bytes
                    < capacity.p_bytes_per_rank
                    or self.lifecycle.cpu_ledger.free_bytes
                    < capacity.cpu_bounce_bytes
                    or self.lifecycle.d_ledger.free_bytes
                    < capacity.d_bytes_per_rank
                ):
                    return None
            else:
                return None
        ticket = self.lifecycle.begin_prepare(
            call.session_id,
            request_id=call.request_id,
            now_ns=now_ns,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            reusable_tokens=call.prefix_reuse_tokens,
            has_successor=call.has_successor,
        )
        if ticket is None:
            raise RuntimeError(
                "tier prepare capacity precheck diverged from lifecycle")
        return ticket

    def _admit_pending(self, now_ns: int) -> None:
        deferred: deque[int] = deque()
        while self._pending_call_ids:
            request_id = self._pending_call_ids.popleft()
            try:
                call = self.calls[request_id]
                lineage = self.sessions[call.session_id]
                if lineage.active_request_id is not None:
                    deferred.append(request_id)
                    continue
                if call.call_index != lineage.last_call_index + 1:
                    raise RuntimeError(
                        "tiered session lifecycle skipped a call")
                ticket = self._try_prepare(
                    call, now_ns=now_ns)
            except Exception:
                # Preserve exact FIFO/non-HOL order even when an unexpected
                # lower-layer failure escapes after submit_many registered
                # the calls.  The exception remains visible to the caller.
                restored = deque(deferred)
                restored.append(request_id)
                restored.extend(self._pending_call_ids)
                self._pending_call_ids = restored
                raise
            if ticket is None:
                deferred.append(request_id)
                continue

            lineage.active_request_id = request_id
            self._ticket_by_request[request_id] = ticket
            call.operational_hit_tokens = ticket.hit_tokens
            call.prepare_id = ticket.prepare_id
            call.prepare_source = ticket.source
            call.prepare_start_ns = ticket.start_ns
            call.prepare_completion_ns = ticket.completion_ns
            call.state = TieredCallState.PREPARING
            self.metrics.admitted_calls += 1
            if (
                call.prefix_reuse_tokens == call.input_tokens
                and ticket.hit_tokens == call.input_tokens - 1
            ):
                self.metrics.full_prefix_cap_calls += 1
            if (
                call.call_index > 0
                and ticket.source_tokens > call.input_tokens
            ):
                self.metrics.context_shrink_calls += 1
            if ticket.source == Tier.D and not ticket.full_d_reservation:
                self.metrics.stable_d_hits += 1
            elif ticket.source in {Tier.CPU, Tier.SSD}:
                self.metrics.lower_tier_hits += 1
                if ticket.restore_layer_ready_ns:
                    self.lifecycle.release_prepare_to_pool(
                        ticket, now_ns=now_ns)
                    self._ready_call_ids.append(ticket.request_id)
                    self.metrics.streaming_lower_tier_hits += 1
            elif (
                call.call_index > 0
                and call.prefix_reuse_tokens > 0
                and ticket.hit_tokens == 0
            ):
                self.metrics.recompute_resumes += 1
                self.metrics.recompute_tokens += call.input_tokens
        self._pending_call_ids = deferred

    def _consume_prepare_notifications(self, now_ns: int) -> None:
        for ticket in self.lifecycle.pop_prepare_completed():
            call = self.calls[ticket.request_id]
            if (
                self._ticket_by_request.get(ticket.request_id)
                is not ticket
                or call.prepare_completion_ns != now_ns
            ):
                raise RuntimeError(
                    "stale tiered prepare completion")
            if ticket.restore_layer_ready_ns:
                if (
                    call.state != TieredCallState.EXECUTING
                    or not ticket.pool_released
                    or call.pool_request is None
                ):
                    raise RuntimeError(
                        "streaming prepare was not released exactly once")
                self.lifecycle.mark_active(ticket, now_ns=now_ns)
            else:
                if call.state != TieredCallState.PREPARING:
                    raise RuntimeError(
                        "bulk prepare completed outside preparing state")
                self.lifecycle.mark_active(ticket, now_ns=now_ns)
                self.lifecycle.release_prepare_to_pool(
                    ticket, now_ns=now_ns)
                self._ready_call_ids.append(ticket.request_id)

    @staticmethod
    def _d_prefix_tokens(ticket: PrepareTicket) -> int:
        if (
            ticket.needs_d
            and ticket.source == Tier.D
            and not ticket.full_d_reservation
        ):
            return ticket.hit_tokens
        return 0

    def _make_pool_request(
            self, call: TieredNodeCall) -> PDServingRequest:
        ticket = self._ticket_by_request[call.request_id]
        d_prefix_tokens = self._d_prefix_tokens(ticket)
        if d_prefix_tokens:
            self.metrics.fresh_suffix_handoffs += 1
        elif call.output_tokens > 1 or call.has_successor:
            self.metrics.full_prompt_handoffs += 1
        request = PDServingRequest(
            request_id=call.request_id,
            session_id=call.session_id,
            arrival_ns=call.release_ns,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            p_prefix_tokens=ticket.hit_tokens,
            d_prefix_tokens=d_prefix_tokens,
            has_successor=call.has_successor,
            restore_layer_ready_ns=ticket.restore_layer_ready_ns,
        )
        call.pool_request = request
        call.state = TieredCallState.EXECUTING
        return request

    def _submit_ready(self, now_ns: int) -> bool:
        if not self._ready_call_ids:
            return False
        request_ids = []
        while self._ready_call_ids:
            request_ids.append(self._ready_call_ids.popleft())
        self.pool.submit_many(
            (
                self._make_pool_request(self.calls[request_id])
                for request_id in request_ids
            ),
            now_ns=now_ns,
        )
        return True

    def _finish_internal(
            self, call: TieredNodeCall, *,
            now_ns: int) -> None:
        if call.state == TieredCallState.INTERNAL_COMPLETE:
            return
        if call.user_completion_ns is None:
            raise RuntimeError(
                "tiered internal completion precedes user completion")
        ticket = self._ticket_by_request[call.request_id]
        self.lifecycle.commit_d_ready(
            ticket,
            now_ns=now_ns,
            has_successor=call.has_successor,
        )
        lineage = self.sessions[call.session_id]
        if lineage.active_request_id != call.request_id:
            raise RuntimeError(
                "tiered session active-request identity mismatch")
        lineage.last_call_index = call.call_index
        lineage.active_request_id = None
        lineage.ended = not call.has_successor
        call.internal_completion_ns = now_ns
        call.state = TieredCallState.INTERNAL_COMPLETE
        self.metrics.internal_completed_calls += 1

    def _consume_pool_notifications(self, now_ns: int) -> None:
        for request in self.pool.pop_handoff_completed():
            if request.request_id in self._handoff_completion_seen:
                raise RuntimeError(
                    "duplicate tiered handoff notification")
            self._handoff_completion_seen.add(request.request_id)
            ticket = self._ticket_by_request[request.request_id]
            self.lifecycle.release_p_after_handoff(
                ticket, now_ns=now_ns)
            call = self.calls[request.request_id]
            if request.request_id in self._user_completion_seen:
                self._finish_internal(call, now_ns=now_ns)
        for request in self.pool.pop_completed():
            if request.request_id in self._user_completion_seen:
                raise RuntimeError(
                    "duplicate tiered user completion")
            self._user_completion_seen.add(request.request_id)
            call = self.calls[request.request_id]
            call.user_completion_ns = request.completion_ns
            call.state = TieredCallState.USER_COMPLETE
            self._user_completed_ids.append(request.request_id)
            self.metrics.user_completed_calls += 1
            if (
                request.handoff_done
                or not (
                    request.output_tokens == 1
                    and request.has_successor
                )
            ):
                self._finish_internal(call, now_ns=now_ns)

    def _next_raw_event_ns(self) -> Optional[int]:
        values = []
        pool_event = self.pool.next_event_ns()
        if pool_event is not None:
            values.append(pool_event)
        lifecycle_event = self.lifecycle.next_event_ns()
        if lifecycle_event is not None:
            values.append(lifecycle_event)
        return min(values) if values else None

    def advance(
            self, now_ns: int, *,
            defer_schedule: bool = False) -> None:
        if now_ns < self.current_ns:
            raise ValueError(
                f"time cannot move backwards: current={self.current_ns}, "
                f"requested={now_ns}")
        while True:
            event_ns = self._next_raw_event_ns()
            if event_ns is None or event_ns > now_ns:
                break
            self.lifecycle.advance(event_ns)
            self.pool.advance(event_ns, defer_schedule=True)
            self.current_ns = event_ns
            self._consume_prepare_notifications(event_ns)
            self._consume_pool_notifications(event_ns)
            if not (defer_schedule and event_ns == now_ns):
                self.flush_scheduling(event_ns)
        self.lifecycle.advance(now_ns)
        self.pool.advance(now_ns, defer_schedule=True)
        self.current_ns = now_ns
        if not defer_schedule:
            self.flush_scheduling(now_ns)
        if self.validate_every_event:
            self.assert_invariants()

    def flush_scheduling(self, now_ns: int) -> None:
        if (
            now_ns != self.current_ns
            or now_ns != self.lifecycle.current_ns
            or now_ns != self.pool.current_ns
        ):
            raise ValueError(
                "flush_scheduling must run at the current node timestamp")
        self._admit_pending(now_ns)
        # Miss prepares contain no transfer stages and complete at the same
        # timestamp.  Commit all such prepare events before launching P.
        self.lifecycle.advance(now_ns)
        self._consume_prepare_notifications(now_ns)
        submitted = self._submit_ready(now_ns)
        if not submitted:
            self.pool.flush_scheduling(now_ns)

    def next_event_ns(self) -> Optional[int]:
        return self._next_raw_event_ns()

    def pop_completed(self) -> list[TieredNodeCall]:
        completed = []
        while self._user_completed_ids:
            completed.append(self.calls[
                self._user_completed_ids.popleft()])
        return completed

    def _deadlock_diagnostic(self) -> str:
        pending = [
            self.calls[request_id]
            for request_id in self._pending_call_ids
        ]
        return (
            "tiered node has unfinished work but no future event: "
            f"policy={self.policy}, pending="
            f"{[(call.request_id, call.session_id) for call in pending]}, "
            f"p_free={self.lifecycle.p_ledger.free_bytes}, "
            f"d_free={self.lifecycle.d_ledger.free_bytes}, "
            f"cpu_free={self.lifecycle.cpu_ledger.free_bytes}, "
            f"ssd_free={self.lifecycle.ssd_ledger.free_bytes}"
        )

    def run_until_idle(self) -> list[TieredNodeCall]:
        completed = self.pop_completed()
        while self.next_event_ns() is not None:
            event_ns = self.next_event_ns()
            assert event_ns is not None
            self.advance(event_ns)
            completed.extend(self.pop_completed())
        if (
            self._pending_call_ids
            or self._ready_call_ids
            or any(
                call.state != TieredCallState.INTERNAL_COMPLETE
                for call in self.calls.values()
            )
        ):
            raise TieredNodeDeadlockError(
                self._deadlock_diagnostic())
        self.lifecycle.assert_invariants()
        self.pool.assert_invariants()
        self.assert_invariants()
        return completed

    def assert_invariants(self) -> None:
        self.lifecycle.assert_invariants()
        self.pool.assert_invariants()
        if not (
            self.current_ns
            == self.lifecycle.current_ns
            == self.pool.current_ns
        ):
            raise AssertionError(
                "tiered node component clocks diverged")
        pending_ids = list(self._pending_call_ids)
        ready_ids = list(self._ready_call_ids)
        if (
            len(pending_ids) != len(set(pending_ids))
            or len(ready_ids) != len(set(ready_ids))
            or set(pending_ids) & set(ready_ids)
        ):
            raise AssertionError(
                "tiered node queue identity invariant failed")
        expected_pending = {
            call.request_id
            for call in self.calls.values()
            if call.state == TieredCallState.PENDING
        }
        if set(pending_ids) != expected_pending:
            raise AssertionError(
                "pending call state/queue mismatch")
        for request_id in ready_ids:
            call = self.calls[request_id]
            ticket = self._ticket_by_request.get(request_id)
            if (
                call.state != TieredCallState.PREPARING
                or ticket is None
                or not ticket.completed
                or not ticket.active
            ):
                raise AssertionError(
                    "ready call lacks active completed prepare")
        active_values = [
            lineage.active_request_id
            for lineage in self.sessions.values()
            if lineage.active_request_id is not None
        ]
        if len(active_values) != len(set(active_values)):
            raise AssertionError(
                "one request is active in multiple lineages")
        active_calls = set(active_values)
        for call in self.calls.values():
            ticket = self._ticket_by_request.get(call.request_id)
            if call.state == TieredCallState.PENDING:
                if ticket is not None:
                    raise AssertionError(
                        f"pending call owns prepare ticket: {call}")
            else:
                if ticket is None:
                    raise AssertionError(
                        f"admitted call lacks prepare ticket: {call}")
            if call.state == TieredCallState.INTERNAL_COMPLETE:
                if (
                    call.internal_completion_ns is None
                    or call.request_id in active_calls
                    or ticket is None
                    or not ticket.committed
                ):
                    raise AssertionError(
                        f"invalid internally completed call: {call}")
            elif call.state != TieredCallState.PENDING:
                if call.request_id not in active_calls:
                    raise AssertionError(
                        f"active tiered call lacks session owner: {call}")
            if call.user_completion_ns is not None:
                if (
                    call.pool_request is None
                    or call.user_completion_ns
                    != call.pool_request.completion_ns
                ):
                    raise AssertionError(
                        f"tiered user completion mismatch: {call}")
        for lineage in self.sessions.values():
            record = self.lifecycle.sessions.get(lineage.session_id)
            if record is None:
                if lineage.active_request_id is not None:
                    raise AssertionError(
                        "active lineage lacks lifecycle session")
                continue
            if lineage.ended:
                if (
                    lineage.active_request_id is not None
                    or record.state != TierSessionState.ENDED
                ):
                    raise AssertionError(
                        f"ended tiered lineage retains state: {lineage}")
            if lineage.active_request_id != record.active_request_id:
                raise AssertionError(
                    "lineage/lifecycle active request mismatch")
        internal_count = sum(
            call.state == TieredCallState.INTERNAL_COMPLETE
            for call in self.calls.values()
        )
        if (
            self.metrics.submitted_calls != len(self.calls)
            or self.metrics.admitted_calls
            != len(self._ticket_by_request)
            or self.metrics.user_completed_calls
            != len(self._user_completion_seen)
            or self.metrics.internal_completed_calls
            != internal_count
            or not (
                self.metrics.internal_completed_calls
                <= self.metrics.user_completed_calls
                <= self.metrics.admitted_calls
                <= self.metrics.submitted_calls
            )
        ):
            raise AssertionError(
                "tiered node metric conservation failed")
        if not (
            self._user_completion_seen <= self.calls.keys()
            and self._handoff_completion_seen <= self.calls.keys()
        ):
            raise AssertionError(
                "tiered completion set contains unknown request")

    def report(self) -> Mapping[str, Any]:
        return {
            "mode": "finite_hbm_p4d4_tiering",
            "node_id": self.node_id,
            "policy": self.policy,
            "restore_execution_mode": self.restore_execution_mode,
            "d_reservation_policy": self.d_reservation_policy,
            "validate_every_event": self.validate_every_event,
            "capacity_owner": (
                "TieredPDKVLifecycle is the sole P/D KV ledger; "
                "there is no AtomicPDHBM mirror"),
            "resume_contract": (
                "stable D hit: D-to-P prefix plus fresh-suffix P-to-D; "
                "CPU/SSD or demotion-racing hit: restore to P plus "
                "full-prompt P-to-D"),
            "metrics": asdict(self.metrics),
            "lifecycle": self.lifecycle.report(),
            "pool": self.pool.report(),
            "sessions": {
                session_id: asdict(lineage)
                for session_id, lineage in sorted(
                    self.sessions.items())
            },
            "calls": {
                request_id: {
                    **asdict(call),
                    "state": call.state.value,
                    "prepare_source": (
                        None if call.prepare_source is None
                        else call.prepare_source.value),
                    "ttft_ns": call.ttft_ns,
                    "tpot_ns": call.tpot_ns,
                }
                for request_id, call in sorted(self.calls.items())
            },
        }


__all__ = [
    "FiniteHBMTieredP4D4Node",
    "PrepareCapacity",
    "TieredCallState",
    "TieredNodeCall",
    "TieredNodeDeadlockError",
    "TieredNodeMetrics",
    "TieredSessionLineage",
]

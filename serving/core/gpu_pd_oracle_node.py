"""Strict infinite-HBM P4D4 node with physical D-to-P resume copies."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
import heapq
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .gpu_pd_hbm import AtomicPDHBM, PDHBMAdmission
from .gpu_pd_latency import P4D4GPUHardware
from .gpu_pd_pool import P4D4ServingPool, PDServingRequest
from .gpu_pd_tier_resources import TierNodeResources, TierTransferStage
from .hbf_full_model_lifecycle import ResourceCalendar


INFINITE_HBM_PROOF_BYTES_PER_RANK = 10 ** 30


class OracleCallState(str, Enum):
    PENDING = "pending"
    PREPARING = "preparing"
    EXECUTING = "executing"
    USER_COMPLETE = "user_complete"
    INTERNAL_COMPLETE = "internal_complete"


@dataclass
class OracleNodeCall:
    request_id: int
    session_id: str
    call_index: int
    release_ns: int
    input_tokens: int
    output_tokens: int
    prefix_reuse_tokens: int
    has_successor: bool
    state: OracleCallState = OracleCallState.PENDING
    operational_hit_tokens: Optional[int] = None
    admission_id: Optional[int] = None
    prepare_start_ns: Optional[int] = None
    prepare_completion_ns: Optional[int] = None
    pool_request: Optional[PDServingRequest] = None
    user_completion_ns: Optional[int] = None
    internal_completion_ns: Optional[int] = None

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
        if self.call_index == 0 and self.prefix_reuse_tokens != 0:
            raise ValueError("first call cannot reuse an earlier prefix")
        if (
            self.state != OracleCallState.PENDING
            or self.operational_hit_tokens is not None
            or self.admission_id is not None
            or self.prepare_start_ns is not None
            or self.prepare_completion_ns is not None
            or self.pool_request is not None
            or self.user_completion_ns is not None
            or self.internal_completion_ns is not None
        ):
            raise ValueError("submitted oracle call must be pristine")

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
class OracleSessionPlacement:
    session_id: str
    last_call_index: int = -1
    materialized_tokens: int = 0
    d_resident: bool = False
    active_request_id: Optional[int] = None
    ended: bool = False


@dataclass(frozen=True)
class OraclePrepareJob:
    job_id: int
    request_id: int
    hit_tokens: int
    stage: TierTransferStage
    start_ns: int
    completion_ns: int


@dataclass
class OracleNodeMetrics:
    submitted_calls: int = 0
    admitted_calls: int = 0
    user_completed_calls: int = 0
    internal_completed_calls: int = 0
    d_to_p_jobs: int = 0
    d_to_p_tokens: int = 0
    d_to_p_bytes_per_rank: int = 0
    d_to_p_aggregate_bytes: int = 0
    d_to_p_service_ns: int = 0
    d_to_p_queue_delay_ns: int = 0
    context_shrink_calls: int = 0
    full_prefix_cap_calls: int = 0
    max_pending_calls: int = 0


class StrictInfiniteHBMNode:
    """One physical P4D4 node with nonbinding HBM capacity.

    Completed KV is retained on D.  A reusable resume therefore pays a
    D-to-P peer copy before P can execute.  "Infinite" removes only
    capacity, demotion, CPU/SSD, and admission-pressure effects.
    """

    def __init__(
            self, *, repo_root: Path, hardware: P4D4GPUHardware,
            node_id: int,
            resource_calendar: Optional[ResourceCalendar] = None,
            max_num_batched_tokens: int = 8_192,
            max_num_seqs: int = 128,
            p_max_num_seqs: Optional[int] = None,
            d_max_num_seqs: Optional[int] = None,
            max_prefill_chunk_tokens: int = 4_096,
            band: str = "central",
            validate_every_event: bool = True,
            retain_detailed_history: bool = True) -> None:
        if not isinstance(validate_every_event, bool):
            raise ValueError("validate_every_event must be a boolean")
        if not isinstance(retain_detailed_history, bool):
            raise ValueError(
                "retain_detailed_history must be a boolean")
        self.hardware = hardware
        self.node_id = node_id
        self.validate_every_event = validate_every_event
        self.calendar = (
            resource_calendar
            if resource_calendar is not None else ResourceCalendar()
        )
        self.resources = TierNodeResources(
            hardware=hardware,
            node_id=node_id,
        )
        self.hbm = AtomicPDHBM(
            hardware=hardware,
            node_id=node_id,
            p_capacity_bytes_per_rank=(
                INFINITE_HBM_PROOF_BYTES_PER_RANK),
            d_capacity_bytes_per_rank=(
                INFINITE_HBM_PROOF_BYTES_PER_RANK),
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
        self.calls: dict[int, OracleNodeCall] = {}
        self.sessions: dict[str, OracleSessionPlacement] = {}
        self.metrics = OracleNodeMetrics()
        self.prepare_history: list[OraclePrepareJob] = []
        self._pending_call_ids: deque[int] = deque()
        self._ready_call_ids: deque[int] = deque()
        self._prepare_jobs: dict[int, OraclePrepareJob] = {}
        self._prepare_completion_heap: list[
            tuple[int, int, int]] = []
        self._user_completed_ids: deque[int] = deque()
        self._admission_by_request: dict[int, PDHBMAdmission] = {}
        self._user_completion_seen: set[int] = set()
        self._handoff_completion_seen: set[int] = set()
        self._last_submitted_call_index: dict[str, int] = {}
        self._last_submitted_request_id: dict[str, int] = {}
        self._next_prepare_job_id = 1
        self.current_ns = 0

    def submit_many(
            self, calls: Iterable[OracleNodeCall], *,
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
            if call.release_ns != now_ns:
                raise ValueError(
                    "oracle calls must be submitted at logical release")
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
            placement = self.sessions.get(call.session_id)
            if placement is not None and placement.ended:
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
                OracleSessionPlacement(session_id=call.session_id),
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

    def submit(self, call: OracleNodeCall, *, now_ns: int) -> None:
        self.submit_many((call,), now_ns=now_ns)

    def _operational_hit(
            self, call: OracleNodeCall,
            placement: OracleSessionPlacement) -> int:
        if call.call_index == 0 or not placement.d_resident:
            return 0
        hit = min(
            call.prefix_reuse_tokens,
            placement.materialized_tokens,
            call.input_tokens - 1,
        )
        if call.prefix_reuse_tokens == call.input_tokens:
            self.metrics.full_prefix_cap_calls += 1
        if call.input_tokens < placement.materialized_tokens:
            self.metrics.context_shrink_calls += 1
        return hit

    def _queue_prepare(
            self, call: OracleNodeCall, *,
            hit_tokens: int, now_ns: int) -> None:
        stage = self.resources.peer_stage(
            hit_tokens,
            direction="d_to_p",
            block_rounded=False,
        )
        job_id = self._next_prepare_job_id
        self._next_prepare_job_id += 1
        start_ns, completion_ns = stage.reserve(
            self.calendar,
            ready_ns=now_ns,
            job_id=job_id,
            namespace=f"oracle-prepare-node-{self.node_id}",
        )
        job = OraclePrepareJob(
            job_id=job_id,
            request_id=call.request_id,
            hit_tokens=hit_tokens,
            stage=stage,
            start_ns=start_ns,
            completion_ns=completion_ns,
        )
        call.state = OracleCallState.PREPARING
        call.prepare_start_ns = start_ns
        call.prepare_completion_ns = completion_ns
        self.prepare_history.append(job)
        self._prepare_jobs[job_id] = job
        heapq.heappush(
            self._prepare_completion_heap,
            (completion_ns, call.request_id, job_id),
        )
        self.metrics.d_to_p_jobs += 1
        self.metrics.d_to_p_tokens += hit_tokens
        self.metrics.d_to_p_bytes_per_rank += stage.bytes_per_rank
        self.metrics.d_to_p_aggregate_bytes += stage.aggregate_bytes
        self.metrics.d_to_p_service_ns += stage.latency_ns
        self.metrics.d_to_p_queue_delay_ns += start_ns - now_ns

    def _admit_pending(self, now_ns: int) -> None:
        deferred: deque[int] = deque()
        while self._pending_call_ids:
            request_id = self._pending_call_ids.popleft()
            call = self.calls[request_id]
            placement = self.sessions[call.session_id]
            if placement.active_request_id is not None:
                deferred.append(request_id)
                continue
            if call.call_index != placement.last_call_index + 1:
                raise RuntimeError(
                    "oracle session lifecycle skipped a call")
            hit_tokens = self._operational_hit(call, placement)
            needs_d = call.output_tokens > 1 or call.has_successor
            admission = self.hbm.try_admit(
                session_id=call.session_id,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                needs_d=needs_d,
            )
            if admission is None:
                raise RuntimeError(
                    "strict infinite-HBM oracle experienced "
                    "a capacity deferral")
            self._admission_by_request[request_id] = admission
            placement.active_request_id = request_id
            call.operational_hit_tokens = hit_tokens
            call.admission_id = admission.admission_id
            self.metrics.admitted_calls += 1
            if hit_tokens:
                self._queue_prepare(
                    call,
                    hit_tokens=hit_tokens,
                    now_ns=now_ns,
                )
            else:
                self.hbm.release_d_source(admission)
                self._ready_call_ids.append(request_id)
        self._pending_call_ids = deferred

    def _make_pool_request(
            self, call: OracleNodeCall) -> PDServingRequest:
        hit_tokens = call.operational_hit_tokens
        if hit_tokens is None:
            raise RuntimeError("oracle call lacks resolved prefix hit")
        request = PDServingRequest(
            request_id=call.request_id,
            session_id=call.session_id,
            arrival_ns=call.release_ns,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            p_prefix_tokens=hit_tokens,
            d_prefix_tokens=hit_tokens,
            has_successor=call.has_successor,
        )
        call.pool_request = request
        call.state = OracleCallState.EXECUTING
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
            self, call: OracleNodeCall, *,
            now_ns: int) -> None:
        if call.state == OracleCallState.INTERNAL_COMPLETE:
            return
        if call.user_completion_ns is None:
            raise RuntimeError(
                "oracle internal completion precedes user completion")
        admission = self._admission_by_request.pop(call.request_id)
        self.hbm.finish(
            admission,
            has_successor=call.has_successor,
        )
        placement = self.sessions[call.session_id]
        if placement.active_request_id != call.request_id:
            raise RuntimeError(
                "oracle session active-request identity mismatch")
        assert call.pool_request is not None
        placement.last_call_index = call.call_index
        placement.materialized_tokens = (
            call.pool_request.final_materialized_kv_tokens)
        placement.d_resident = call.has_successor
        placement.active_request_id = None
        placement.ended = not call.has_successor
        call.internal_completion_ns = now_ns
        call.state = OracleCallState.INTERNAL_COMPLETE
        self.metrics.internal_completed_calls += 1

    def _consume_pool_notifications(self, now_ns: int) -> None:
        for request in self.pool.pop_handoff_completed():
            if request.request_id in self._handoff_completion_seen:
                raise RuntimeError("duplicate oracle handoff notification")
            self._handoff_completion_seen.add(request.request_id)
            admission = self._admission_by_request[request.request_id]
            self.hbm.release_p(admission)
            call = self.calls[request.request_id]
            if request.request_id in self._user_completion_seen:
                self._finish_internal(call, now_ns=now_ns)
        for request in self.pool.pop_completed():
            if request.request_id in self._user_completion_seen:
                raise RuntimeError("duplicate oracle user completion")
            self._user_completion_seen.add(request.request_id)
            call = self.calls[request.request_id]
            call.user_completion_ns = request.completion_ns
            call.state = OracleCallState.USER_COMPLETE
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

    def _process_prepare_completions(self, now_ns: int) -> None:
        while (
            self._prepare_completion_heap
            and self._prepare_completion_heap[0][0] == now_ns
        ):
            _, request_id, job_id = heapq.heappop(
                self._prepare_completion_heap)
            job = self._prepare_jobs.pop(job_id)
            call = self.calls[request_id]
            if (
                call.state != OracleCallState.PREPARING
                or call.prepare_completion_ns != now_ns
            ):
                raise RuntimeError("stale oracle prepare completion")
            admission = self._admission_by_request[request_id]
            self.hbm.release_d_source(admission)
            self._ready_call_ids.append(request_id)

    def _next_raw_event_ns(self) -> Optional[int]:
        values = []
        pool_event = self.pool.next_event_ns()
        if pool_event is not None:
            values.append(pool_event)
        if self._prepare_completion_heap:
            values.append(self._prepare_completion_heap[0][0])
        return min(values) if values else None

    def advance(
            self, now_ns: int, *, defer_schedule: bool = False) -> None:
        if now_ns < self.current_ns:
            raise ValueError(
                f"time cannot move backwards: current={self.current_ns}, "
                f"requested={now_ns}")
        while True:
            event_ns = self._next_raw_event_ns()
            if event_ns is None or event_ns > now_ns:
                break
            self.pool.advance(event_ns, defer_schedule=True)
            self._process_prepare_completions(event_ns)
            self._consume_pool_notifications(event_ns)
            self.current_ns = event_ns
            if not (defer_schedule and event_ns == now_ns):
                self.flush_scheduling(event_ns)
        self.pool.advance(now_ns, defer_schedule=True)
        self.current_ns = now_ns
        if not defer_schedule:
            self.flush_scheduling(now_ns)
        if self.validate_every_event:
            self.assert_invariants()

    def flush_scheduling(self, now_ns: int) -> None:
        if now_ns != self.current_ns or now_ns != self.pool.current_ns:
            raise ValueError(
                "flush_scheduling must run at the current node timestamp")
        self._admit_pending(now_ns)
        submitted = self._submit_ready(now_ns)
        if not submitted:
            self.pool.flush_scheduling(now_ns)

    def next_event_ns(self) -> Optional[int]:
        return self._next_raw_event_ns()

    def pop_completed(self) -> list[OracleNodeCall]:
        completed = []
        while self._user_completed_ids:
            completed.append(self.calls[
                self._user_completed_ids.popleft()])
        return completed

    def run_until_idle(self) -> list[OracleNodeCall]:
        completed = self.pop_completed()
        while self.next_event_ns() is not None:
            event_ns = self.next_event_ns()
            assert event_ns is not None
            self.advance(event_ns)
            completed.extend(self.pop_completed())
        if (
            self._pending_call_ids
            or self._ready_call_ids
            or self._prepare_completion_heap
            or any(
                call.state != OracleCallState.INTERNAL_COMPLETE
                for call in self.calls.values()
            )
        ):
            raise RuntimeError(
                "oracle node became idle before full internal drain")
        self.assert_invariants()
        return completed

    def assert_invariants(self) -> None:
        self.hbm.assert_invariants()
        self.pool.assert_invariants()
        if self.hbm.metrics.capacity_deferrals:
            raise AssertionError(
                "strict oracle must not have HBM capacity deferrals")
        active_calls = {
            placement.active_request_id
            for placement in self.sessions.values()
            if placement.active_request_id is not None
        }
        for call in self.calls.values():
            placement = self.sessions[call.session_id]
            if call.state == OracleCallState.INTERNAL_COMPLETE:
                if (
                    call.internal_completion_ns is None
                    or call.request_id in active_calls
                ):
                    raise AssertionError(
                        f"invalid internally completed call: {call}")
            elif call.state != OracleCallState.PENDING:
                if call.request_id not in active_calls:
                    raise AssertionError(
                        f"active oracle call lacks session owner: {call}")
            if call.user_completion_ns is not None:
                if (
                    call.pool_request is None
                    or call.user_completion_ns
                    != call.pool_request.completion_ns
                ):
                    raise AssertionError(
                        f"oracle user completion mismatch: {call}")
        for placement in self.sessions.values():
            if placement.ended and (
                    placement.active_request_id is not None
                    or placement.d_resident):
                raise AssertionError(
                    f"ended oracle session retains state: {placement}")
        if self.metrics.user_completed_calls > (
                self.metrics.submitted_calls):
            raise AssertionError(
                "oracle completions exceed submissions")

    def report(self) -> Mapping[str, Any]:
        return {
            "mode": "strict_infinite_hbm_residency_oracle",
            "node_id": self.node_id,
            "infinite_hbm_proof_bytes_per_rank": (
                INFINITE_HBM_PROOF_BYTES_PER_RANK),
            "validate_every_event": self.validate_every_event,
            "physical_resume_contract": (
                "D-resident prefix pays D-to-P copy; "
                "post-TTFT P-to-D sends fresh suffix"),
            "metrics": asdict(self.metrics),
            "hbm": self.hbm.report(),
            "pool": self.pool.report(),
            "sessions": {
                session_id: asdict(placement)
                for session_id, placement
                in sorted(self.sessions.items())
            },
            "calls": {
                request_id: {
                    **asdict(call),
                    "state": call.state.value,
                    "ttft_ns": call.ttft_ns,
                    "tpot_ns": call.tpot_ns,
                }
                for request_id, call in sorted(self.calls.items())
            },
        }


__all__ = [
    "INFINITE_HBM_PROOF_BYTES_PER_RANK",
    "OracleCallState",
    "OracleNodeCall",
    "OracleNodeMetrics",
    "OraclePrepareJob",
    "OracleSessionPlacement",
    "StrictInfiniteHBMNode",
]

"""Continuous-batching P4D4 execution for one eight-H100 server.

This module models only GPU execution and the post-TTFT P-to-D handoff.
Admission, finite-HBM placement, and CPU/SSD tiering are deliberately owned
by a separate controller so their queueing can be validated independently.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
import heapq
import math
from pathlib import Path
from typing import Any, Iterable, Optional

from .gpu_pd_latency import (
    GPUBatchLatency,
    P4D4GPUHardware,
    P4D4LatencyModel,
)
from .hbf_full_model_latency import HBFModelBatchShape
from .hbf_full_model_lifecycle import ResourceCalendar


MAX_P4D4_PREFILL_CHUNK_TOKENS = 131_072
MAX_P4D4_SEQUENCES = 128


class PDRequestState(str, Enum):
    WAITING_P = "waiting_p"
    P_PREFILL = "p_prefill"
    HANDOFF = "handoff"
    D_DECODE = "d_decode"
    COMPLETE = "complete"


@dataclass
class PDServingRequest:
    """One request after placement/admission has resolved."""

    request_id: int
    session_id: str
    arrival_ns: int
    input_tokens: int
    output_tokens: int
    p_prefix_tokens: int
    d_prefix_tokens: int
    has_successor: bool = True
    state: PDRequestState = PDRequestState.WAITING_P
    p_processed_tokens: int = 0
    generated_tokens: int = 0
    first_scheduled_ns: Optional[int] = None
    first_token_ns: Optional[int] = None
    completion_ns: Optional[int] = None
    handoff_start_ns: Optional[int] = None
    handoff_completion_ns: Optional[int] = None
    handoff_done: bool = False
    token_completion_ns: list[int] = field(default_factory=list)
    p_batch_count: int = 0
    d_batch_count: int = 0
    stage_ready_ns: Optional[int] = None

    def validate(self) -> None:
        for name in (
            "request_id",
            "arrival_ns",
            "input_tokens",
            "output_tokens",
            "p_prefix_tokens",
            "d_prefix_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.request_id < 0 or self.arrival_ns < 0:
            raise ValueError("request_id/arrival_ns must be non-negative")
        if self.input_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("input/output tokens must be positive")
        if not 0 <= self.p_prefix_tokens <= self.input_tokens:
            raise ValueError("P prefix must be in 0..input_tokens")
        if not 0 <= self.d_prefix_tokens <= self.input_tokens:
            raise ValueError("D prefix must be in 0..input_tokens")
        if self.input_tokens + self.output_tokens - 1 > 1_010_000:
            raise ValueError(
                "request output would exceed 1,010,000-token contract")
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if not isinstance(self.has_successor, bool):
            raise ValueError("has_successor must be a boolean")
        if (
            self.state != PDRequestState.WAITING_P
            or self.p_processed_tokens
            or self.generated_tokens
            or self.first_scheduled_ns is not None
            or self.first_token_ns is not None
            or self.completion_ns is not None
            or self.handoff_start_ns is not None
            or self.handoff_completion_ns is not None
            or self.handoff_done
            or self.token_completion_ns
            or self.p_batch_count
            or self.d_batch_count
            or self.stage_ready_ns is not None
        ):
            raise ValueError("submitted P/D request must be pristine")

    @property
    def operational_p_prefix_tokens(self) -> int:
        """Leave one prompt token for P even on a full-prefix hit."""

        return min(self.p_prefix_tokens, self.input_tokens - 1)

    @property
    def p_fresh_tokens(self) -> int:
        return self.input_tokens - self.operational_p_prefix_tokens

    @property
    def p_remaining_tokens(self) -> int:
        return self.p_fresh_tokens - self.p_processed_tokens

    @property
    def handoff_tokens(self) -> int:
        return self.input_tokens - self.d_prefix_tokens

    @property
    def final_materialized_kv_tokens(self) -> int:
        return self.input_tokens + self.output_tokens - 1

    @property
    def ttft_ns(self) -> Optional[int]:
        if self.first_token_ns is None:
            return None
        return self.first_token_ns - self.arrival_ns

    @property
    def tpot_ns(self) -> Optional[float]:
        if self.output_tokens == 1 or self.completion_ns is None:
            return None
        assert self.first_token_ns is not None
        return (
            (self.completion_ns - self.first_token_ns)
            / (self.output_tokens - 1)
        )


@dataclass(frozen=True)
class PDBatchItem:
    request_id: int
    query_tokens: int


@dataclass(frozen=True)
class PDServingBatch:
    batch_id: int
    stage: str
    ready_ns: int
    start_ns: int
    completion_ns: int
    items: tuple[PDBatchItem, ...]
    shape: HBFModelBatchShape
    latency: GPUBatchLatency


@dataclass(frozen=True)
class PDHandoffJob:
    job_id: int
    request_id: int
    token_count: int
    bytes_per_rank: int
    aggregate_bytes: int
    ready_ns: int
    start_ns: int
    completion_ns: int


@dataclass
class PDWorker:
    stage: str
    waiting: deque[int] = field(default_factory=deque)
    inflight: Optional[PDServingBatch] = None
    pending_launch_ns: Optional[int] = None
    completed_batches: int = 0


@dataclass
class PDPoolMetrics:
    submitted_requests: int = 0
    completed_requests: int = 0
    p_batches: int = 0
    d_batches: int = 0
    p_query_tokens: int = 0
    d_query_tokens: int = 0
    p_modeled_ns: int = 0
    d_modeled_ns: int = 0
    p_resource_delay_ns: int = 0
    d_resource_delay_ns: int = 0
    handoff_jobs: int = 0
    zero_byte_handoffs: int = 0
    handoff_bytes_per_rank: int = 0
    handoff_aggregate_bytes: int = 0
    handoff_service_ns: int = 0
    handoff_queue_delay_ns: int = 0
    max_p_batch_size: int = 0
    max_d_batch_size: int = 0


class P4D4ServingPool:
    """One TP4 prefill worker and one TP4 decode worker on a P4D4 node.

    ``max_num_seqs`` remains the backward-compatible shared sequence limit.
    The optional stage-specific limits override it independently, which lets
    the documented Qwen3 deployment use P=32 and D=128.  That deployment sets
    both ``max_num_batched_tokens`` and ``max_prefill_chunk_tokens`` to
    131,072.  The aggregate token-budget API retains its legacy positive-only
    validation; the calibrated per-request prefill-chunk ceiling is 131,072.
    """

    def __init__(
            self, *, repo_root: Path, hardware: P4D4GPUHardware,
            node_id: int = 0,
            resource_calendar: Optional[ResourceCalendar] = None,
            max_num_batched_tokens: int = 8_192,
            max_num_seqs: int = 128,
            p_max_num_seqs: Optional[int] = None,
            d_max_num_seqs: Optional[int] = None,
            max_prefill_chunk_tokens: int = 4_096,
            band: str = "central",
            validate_every_event: bool = True,
            retain_detailed_history: bool = True) -> None:
        hardware.validate()
        if isinstance(node_id, bool) or not isinstance(node_id, int):
            raise ValueError("node_id must be an integer")
        if node_id < 0:
            raise ValueError("node_id must be non-negative")
        for name, value in (
            ("max_num_batched_tokens", max_num_batched_tokens),
            ("max_num_seqs", max_num_seqs),
            ("max_prefill_chunk_tokens", max_prefill_chunk_tokens),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if max_num_seqs > MAX_P4D4_SEQUENCES:
            raise ValueError(
                "max_num_seqs exceeds calibrated support "
                f"({MAX_P4D4_SEQUENCES})")
        resolved_stage_limits = {}
        for stage, value in (
            ("p", p_max_num_seqs),
            ("d", d_max_num_seqs),
        ):
            resolved = max_num_seqs if value is None else value
            if (
                isinstance(resolved, bool)
                or not isinstance(resolved, int)
                or resolved <= 0
            ):
                raise ValueError(
                    f"{stage}_max_num_seqs must be a positive integer")
            if resolved > MAX_P4D4_SEQUENCES:
                raise ValueError(
                    f"{stage}_max_num_seqs exceeds calibrated support "
                    f"({MAX_P4D4_SEQUENCES})")
            resolved_stage_limits[stage] = resolved
        if not 0 < max_prefill_chunk_tokens <= max_num_batched_tokens:
            raise ValueError(
                "max_prefill_chunk_tokens must be in 1..token budget")
        if max_prefill_chunk_tokens > MAX_P4D4_PREFILL_CHUNK_TOKENS:
            raise ValueError(
                "max_prefill_chunk_tokens exceeds calibrated support "
                f"({MAX_P4D4_PREFILL_CHUNK_TOKENS})")
        if not isinstance(validate_every_event, bool):
            raise ValueError("validate_every_event must be a boolean")
        if not isinstance(retain_detailed_history, bool):
            raise ValueError(
                "retain_detailed_history must be a boolean")
        self.hardware = hardware
        self.node_id = node_id
        self.calendar = (
            resource_calendar
            if resource_calendar is not None else ResourceCalendar()
        )
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.p_max_num_seqs = resolved_stage_limits["p"]
        self.d_max_num_seqs = resolved_stage_limits["d"]
        self.max_prefill_chunk_tokens = max_prefill_chunk_tokens
        self.validate_every_event = validate_every_event
        self.retain_detailed_history = retain_detailed_history
        self.model = P4D4LatencyModel(
            repo_root=repo_root,
            hardware=hardware,
            band=band,
        )
        self.p_worker = PDWorker(stage="p")
        self.d_worker = PDWorker(stage="d")
        self.requests: dict[int, PDServingRequest] = {}
        self._current_request_by_session: dict[str, int] = {}
        self.metrics = PDPoolMetrics()
        self.batch_history: list[PDServingBatch] = []
        self.handoff_history: list[PDHandoffJob] = []
        self._completed_request_ids: deque[int] = deque()
        self._completed_handoff_request_ids: deque[int] = deque()
        self._batch_completion_heap: list[
            tuple[int, str, int]] = []
        self._handoff_completion_heap: list[
            tuple[int, int, int]] = []
        self._handoff_jobs: dict[int, PDHandoffJob] = {}
        self._launch_heap: list[tuple[int, str]] = []
        self._next_batch_id = 1
        self._next_handoff_id = 1
        self.current_ns = 0

    def _worker(self, stage: str) -> PDWorker:
        if stage == "p":
            return self.p_worker
        if stage == "d":
            return self.d_worker
        raise ValueError(f"invalid P/D stage {stage!r}")

    def _model_resource(self, stage: str) -> str:
        return f"gpu-node-{self.node_id}-{stage}-model"

    def _handoff_resource(self) -> str:
        return f"gpu-node-{self.node_id}-pd-fabric"

    def submit_many(
            self, requests: Iterable[PDServingRequest],
            *, now_ns: int) -> None:
        values = list(requests)
        seen_ids = set()
        seen_sessions = set()
        for request in values:
            request.validate()
            if request.arrival_ns > now_ns:
                raise ValueError(
                    "request logical arrival cannot be after submit time")
            if (
                request.request_id in self.requests
                or request.request_id in seen_ids
            ):
                raise ValueError(
                    f"duplicate request_id={request.request_id}")
            previous_id = self._current_request_by_session.get(
                request.session_id)
            if (
                previous_id is not None
                and self.requests[previous_id].state
                != PDRequestState.COMPLETE
            ):
                raise ValueError(
                    f"session {request.session_id!r} already has "
                    "a live P/D request")
            if request.session_id in seen_sessions:
                raise ValueError(
                    f"duplicate live session {request.session_id!r}")
            seen_ids.add(request.request_id)
            seen_sessions.add(request.session_id)
        self.advance(now_ns, defer_schedule=True)
        for request in values:
            request.state = PDRequestState.P_PREFILL
            request.stage_ready_ns = now_ns
            self.requests[request.request_id] = request
            self._current_request_by_session[request.session_id] = (
                request.request_id)
            self.p_worker.waiting.append(request.request_id)
            self.metrics.submitted_requests += 1
        self.flush_scheduling(now_ns)

    def submit(self, request: PDServingRequest, *, now_ns: int) -> None:
        self.submit_many((request,), now_ns=now_ns)

    def _select_p_batch(
            self) -> tuple[
                tuple[PDBatchItem, ...], HBFModelBatchShape] | None:
        items = []
        prefill_q = []
        prefill_k = []
        token_budget = self.max_num_batched_tokens
        seq_budget = self.p_max_num_seqs
        waiting_count = len(self.p_worker.waiting)
        for _ in range(waiting_count):
            if token_budget <= 0 or seq_budget <= 0:
                break
            request_id = self.p_worker.waiting.popleft()
            request = self.requests[request_id]
            if request.state != PDRequestState.P_PREFILL:
                raise RuntimeError("P queue contains non-prefill request")
            chunk = min(
                request.p_remaining_tokens,
                token_budget,
                self.max_prefill_chunk_tokens,
            )
            if chunk <= 0:
                raise RuntimeError("P request has no remaining prompt work")
            items.append(PDBatchItem(request_id, chunk))
            prefill_q.append(chunk)
            prefill_k.append(
                request.operational_p_prefix_tokens
                + request.p_processed_tokens
            )
            token_budget -= chunk
            seq_budget -= 1
        if not items:
            return None
        return tuple(items), HBFModelBatchShape(
            total_tokens=sum(item.query_tokens for item in items),
            prefill_q=tuple(prefill_q),
            prefill_hbf_k=tuple(prefill_k),
            prefill_lpddr_k=(0,) * len(prefill_q),
            lm_head_sequences=len(items),
        )

    def _select_d_batch(
            self) -> tuple[
                tuple[PDBatchItem, ...], HBFModelBatchShape] | None:
        items = []
        decode_k = []
        token_budget = self.max_num_batched_tokens
        seq_budget = self.d_max_num_seqs
        waiting_count = len(self.d_worker.waiting)
        for _ in range(waiting_count):
            if token_budget <= 0 or seq_budget <= 0:
                break
            request_id = self.d_worker.waiting.popleft()
            request = self.requests[request_id]
            if request.state != PDRequestState.D_DECODE:
                raise RuntimeError("D queue contains non-decode request")
            if not request.handoff_done:
                raise RuntimeError("D request entered before handoff")
            items.append(PDBatchItem(request_id, 1))
            decode_k.append(
                request.input_tokens + request.generated_tokens - 1)
            token_budget -= 1
            seq_budget -= 1
        if not items:
            return None
        return tuple(items), HBFModelBatchShape(
            total_tokens=len(items),
            decode_hbf_k=tuple(decode_k),
            decode_lpddr_k=(0,) * len(decode_k),
            lm_head_sequences=len(items),
        )

    def _schedule_launch(self, stage: str, launch_ns: int) -> None:
        worker = self._worker(stage)
        if worker.pending_launch_ns == launch_ns:
            return
        worker.pending_launch_ns = launch_ns
        heapq.heappush(self._launch_heap, (launch_ns, stage))

    def _try_schedule(self, stage: str, now_ns: int) -> None:
        worker = self._worker(stage)
        if worker.inflight is not None:
            return
        if not worker.waiting:
            worker.pending_launch_ns = None
            return
        resource = self._model_resource(stage)
        available_ns = self.calendar.earliest_start(
            now_ns, (resource,))
        if available_ns > now_ns:
            self._schedule_launch(stage, available_ns)
            return
        worker.pending_launch_ns = None
        selected = (
            self._select_p_batch()
            if stage == "p" else self._select_d_batch()
        )
        if selected is None:
            return
        items, shape = selected
        latency = self.model.batch_latency(shape)
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        ready_values = [
            self.requests[item.request_id].stage_ready_ns
            for item in items
        ]
        if any(value is None for value in ready_values):
            raise RuntimeError("scheduled P/D request lacks ready time")
        ready_ns = min(ready_values)
        start_ns, completion_ns = self.calendar.reserve_parallel(
            arrival_ns=now_ns,
            job_id=batch_id,
            kind=f"{stage}-model-batch",
            namespace=f"gpu-pd-node-{self.node_id}",
            demands={resource: (latency.total_ns, 0)},
        )
        batch = PDServingBatch(
            batch_id=batch_id,
            stage=stage,
            ready_ns=ready_ns,
            start_ns=start_ns,
            completion_ns=completion_ns,
            items=items,
            shape=shape,
            latency=latency,
        )
        worker.inflight = batch
        if self.retain_detailed_history:
            self.batch_history.append(batch)
        heapq.heappush(
            self._batch_completion_heap,
            (completion_ns, stage, batch_id),
        )
        for item in items:
            request = self.requests[item.request_id]
            if request.first_scheduled_ns is None:
                request.first_scheduled_ns = start_ns
            if stage == "p":
                request.p_batch_count += 1
            else:
                request.d_batch_count += 1
        if stage == "p":
            self.metrics.p_batches += 1
            self.metrics.p_query_tokens += shape.real_query_tokens
            self.metrics.p_modeled_ns += latency.total_ns
            self.metrics.p_resource_delay_ns += start_ns - ready_ns
            self.metrics.max_p_batch_size = max(
                self.metrics.max_p_batch_size, len(items))
        else:
            self.metrics.d_batches += 1
            self.metrics.d_query_tokens += shape.real_query_tokens
            self.metrics.d_modeled_ns += latency.total_ns
            self.metrics.d_resource_delay_ns += start_ns - ready_ns
            self.metrics.max_d_batch_size = max(
                self.metrics.max_d_batch_size, len(items))

    def _complete_request(
            self, request: PDServingRequest, now_ns: int) -> None:
        request.state = PDRequestState.COMPLETE
        request.completion_ns = now_ns
        self._completed_request_ids.append(request.request_id)
        self.metrics.completed_requests += 1

    def _start_handoff(
            self, request: PDServingRequest, ready_ns: int) -> None:
        latency = self.model.handoff_latency(request.handoff_tokens)
        job_id = self._next_handoff_id
        self._next_handoff_id += 1
        if latency.latency_ns == 0:
            start_ns = ready_ns
            completion_ns = ready_ns
            self.metrics.zero_byte_handoffs += 1
        else:
            start_ns, completion_ns = self.calendar.reserve_parallel(
                arrival_ns=ready_ns,
                job_id=job_id,
                kind="pd-handoff",
                namespace=f"gpu-pd-handoff-node-{self.node_id}",
                demands={
                    self._handoff_resource(): (
                        latency.latency_ns,
                        latency.aggregate_bytes,
                    ),
                },
            )
        job = PDHandoffJob(
            job_id=job_id,
            request_id=request.request_id,
            token_count=request.handoff_tokens,
            bytes_per_rank=latency.bytes_per_rank,
            aggregate_bytes=latency.aggregate_bytes,
            ready_ns=ready_ns,
            start_ns=start_ns,
            completion_ns=completion_ns,
        )
        request.handoff_start_ns = start_ns
        request.handoff_completion_ns = completion_ns
        if self.retain_detailed_history:
            self.handoff_history.append(job)
        self._handoff_jobs[job_id] = job
        self.metrics.handoff_jobs += 1
        self.metrics.handoff_bytes_per_rank += latency.bytes_per_rank
        self.metrics.handoff_aggregate_bytes += latency.aggregate_bytes
        self.metrics.handoff_service_ns += latency.latency_ns
        self.metrics.handoff_queue_delay_ns += start_ns - ready_ns
        if completion_ns == ready_ns:
            self._finish_handoff(job)
        else:
            heapq.heappush(
                self._handoff_completion_heap,
                (completion_ns, request.request_id, job_id),
            )

    def _finish_handoff(self, job: PDHandoffJob) -> None:
        stored = self._handoff_jobs.pop(job.job_id, None)
        if stored != job:
            raise RuntimeError(
                "P/D handoff job ownership mismatch")
        request = self.requests[job.request_id]
        if (
            request.handoff_completion_ns != job.completion_ns
            or request.handoff_done
        ):
            raise RuntimeError("P/D handoff completion identity mismatch")
        request.handoff_done = True
        self._completed_handoff_request_ids.append(request.request_id)
        if request.output_tokens > 1:
            if request.state != PDRequestState.HANDOFF:
                raise RuntimeError(
                    "multi-token handoff completed outside handoff state")
            request.state = PDRequestState.D_DECODE
            request.stage_ready_ns = job.completion_ns
            self.d_worker.waiting.append(request.request_id)

    def _finish_p_batch(self, batch: PDServingBatch) -> None:
        worker = self.p_worker
        if worker.inflight is None or worker.inflight.batch_id != batch.batch_id:
            raise RuntimeError("P batch completion identity mismatch")
        for item in batch.items:
            request = self.requests[item.request_id]
            request.p_processed_tokens += item.query_tokens
            if request.p_remaining_tokens:
                request.stage_ready_ns = batch.completion_ns
                worker.waiting.append(request.request_id)
                continue
            request.generated_tokens = 1
            request.first_token_ns = batch.completion_ns
            if self.retain_detailed_history:
                request.token_completion_ns.append(
                    batch.completion_ns)
            if request.output_tokens == 1:
                self._complete_request(request, batch.completion_ns)
            else:
                request.state = PDRequestState.HANDOFF
                request.stage_ready_ns = batch.completion_ns
            if request.output_tokens > 1 or request.has_successor:
                self._start_handoff(request, batch.completion_ns)
        worker.inflight = None
        worker.completed_batches += 1

    def _finish_d_batch(self, batch: PDServingBatch) -> None:
        worker = self.d_worker
        if worker.inflight is None or worker.inflight.batch_id != batch.batch_id:
            raise RuntimeError("D batch completion identity mismatch")
        for item in batch.items:
            request = self.requests[item.request_id]
            if request.state != PDRequestState.D_DECODE:
                raise RuntimeError("D batch contains non-decode request")
            request.generated_tokens += 1
            if self.retain_detailed_history:
                request.token_completion_ns.append(
                    batch.completion_ns)
            if request.generated_tokens == request.output_tokens:
                self._complete_request(request, batch.completion_ns)
            else:
                request.stage_ready_ns = batch.completion_ns
                worker.waiting.append(request.request_id)
        worker.inflight = None
        worker.completed_batches += 1

    def _clean_launch_heap(self) -> None:
        while self._launch_heap:
            launch_ns, stage = self._launch_heap[0]
            if self._worker(stage).pending_launch_ns == launch_ns:
                return
            heapq.heappop(self._launch_heap)

    def _next_raw_event_ns(self) -> Optional[int]:
        self._clean_launch_heap()
        values = []
        if self._batch_completion_heap:
            values.append(self._batch_completion_heap[0][0])
        if self._handoff_completion_heap:
            values.append(self._handoff_completion_heap[0][0])
        if self._launch_heap:
            values.append(self._launch_heap[0][0])
        return min(values) if values else None

    def _process_events_at(self, event_ns: int) -> None:
        while (
            self._batch_completion_heap
            and self._batch_completion_heap[0][0] == event_ns
        ):
            _, stage, batch_id = heapq.heappop(
                self._batch_completion_heap)
            worker = self._worker(stage)
            if worker.inflight is None or worker.inflight.batch_id != batch_id:
                raise RuntimeError("stale P/D batch completion")
            if stage == "p":
                self._finish_p_batch(worker.inflight)
            else:
                self._finish_d_batch(worker.inflight)
        while (
            self._handoff_completion_heap
            and self._handoff_completion_heap[0][0] == event_ns
        ):
            _, request_id, job_id = heapq.heappop(
                self._handoff_completion_heap)
            request = self.requests[request_id]
            job = self._handoff_jobs.get(job_id)
            if job is None:
                raise RuntimeError("missing P/D handoff job")
            self._finish_handoff(job)
            if request.handoff_completion_ns != event_ns:
                raise RuntimeError("stale P/D handoff completion")
        self._clean_launch_heap()
        while self._launch_heap and self._launch_heap[0][0] == event_ns:
            _, stage = heapq.heappop(self._launch_heap)
            worker = self._worker(stage)
            if worker.pending_launch_ns == event_ns:
                worker.pending_launch_ns = None

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
            self._process_events_at(event_ns)
            self.current_ns = event_ns
            if not (defer_schedule and event_ns == now_ns):
                self.flush_scheduling(event_ns)
        self.current_ns = now_ns
        if self.validate_every_event:
            self.assert_invariants()

    def flush_scheduling(self, now_ns: int) -> None:
        if now_ns != self.current_ns:
            raise ValueError(
                "flush_scheduling must run at current pool timestamp")
        self._try_schedule("p", now_ns)
        self._try_schedule("d", now_ns)

    def next_event_ns(self) -> Optional[int]:
        return self._next_raw_event_ns()

    def pop_completed(self) -> list[PDServingRequest]:
        completed = []
        while self._completed_request_ids:
            completed.append(self.requests[
                self._completed_request_ids.popleft()])
        return completed

    def pop_handoff_completed(self) -> list[PDServingRequest]:
        completed = []
        while self._completed_handoff_request_ids:
            completed.append(self.requests[
                self._completed_handoff_request_ids.popleft()])
        return completed

    def run_until_idle(self) -> list[PDServingRequest]:
        completed = self.pop_completed()
        while self.next_event_ns() is not None:
            event_ns = self.next_event_ns()
            assert event_ns is not None
            self.advance(event_ns)
            completed.extend(self.pop_completed())
        if (
            self.p_worker.waiting
            or self.d_worker.waiting
            or self.p_worker.inflight is not None
            or self.d_worker.inflight is not None
            or self.p_worker.pending_launch_ns is not None
            or self.d_worker.pending_launch_ns is not None
        ):
            raise RuntimeError("P/D pool became idle with queued work")
        self.assert_invariants()
        return completed

    def assert_invariants(self) -> None:
        queued = (
            list(self.p_worker.waiting)
            + list(self.d_worker.waiting)
        )
        for worker in (self.p_worker, self.d_worker):
            if (
                worker.inflight is not None
                and worker.pending_launch_ns is not None
            ):
                raise AssertionError(
                    "P/D worker cannot be inflight and pending launch")
            if worker.inflight is not None:
                queued.extend(
                    item.request_id for item in worker.inflight.items)
        for request in self.requests.values():
            if not 0 <= request.p_processed_tokens <= request.p_fresh_tokens:
                raise AssertionError(f"invalid P progress: {request}")
            if not 0 <= request.generated_tokens <= request.output_tokens:
                raise AssertionError(f"invalid output progress: {request}")
            expected_timestamps = (
                request.generated_tokens
                if self.retain_detailed_history else 0
            )
            if len(request.token_completion_ns) != expected_timestamps:
                raise AssertionError(
                    f"token timestamp mismatch: {request}")
            if request.first_token_ns is not None:
                if request.generated_tokens < 1:
                    raise AssertionError(
                        f"invalid first-token accounting: {request}")
                if (
                    self.retain_detailed_history
                    and request.token_completion_ns[0]
                    != request.first_token_ns
                ):
                    raise AssertionError(
                        f"invalid first-token accounting: {request}")
            if request.state == PDRequestState.COMPLETE:
                if (
                    request.completion_ns is None
                    or request.generated_tokens != request.output_tokens
                    or request.request_id in queued
                ):
                    raise AssertionError(
                        f"invalid completed request: {request}")
            elif request.state == PDRequestState.HANDOFF:
                if request.request_id in queued:
                    raise AssertionError(
                        f"handoff request remains in compute queue: {request}")
            elif queued.count(request.request_id) != 1:
                raise AssertionError(
                    f"live request queue multiplicity is not one: {request}")
            if request.state == PDRequestState.D_DECODE:
                if not request.handoff_done:
                    raise AssertionError(
                        f"D request lacks completed handoff: {request}")
            if (
                request.output_tokens > 1
                and request.state == PDRequestState.HANDOFF
                and request.handoff_completion_ns is None
            ):
                raise AssertionError(
                    f"handoff state lacks job: {request}")
            if (
                request.completion_ns is not None
                and request.output_tokens == 1
                and request.completion_ns != request.first_token_ns
            ):
                raise AssertionError(
                    f"one-token completion exceeds TTFT: {request}")
        if self.metrics.completed_requests > self.metrics.submitted_requests:
            raise AssertionError("completed requests exceed submissions")

    def report(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "hardware": asdict(self.hardware),
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "p_max_num_seqs": self.p_max_num_seqs,
            "d_max_num_seqs": self.d_max_num_seqs,
            "max_prefill_chunk_tokens": self.max_prefill_chunk_tokens,
            "validate_every_event": self.validate_every_event,
            "retain_detailed_history": self.retain_detailed_history,
            "retained_batch_count": len(self.batch_history),
            "retained_handoff_count": len(self.handoff_history),
            "live_handoff_job_count": len(self._handoff_jobs),
            "latency_model": self.model.metadata(),
            "metrics": asdict(self.metrics),
            "requests": {
                request_id: {
                    **asdict(request),
                    "state": request.state.value,
                    "ttft_ns": request.ttft_ns,
                    "tpot_ns": request.tpot_ns,
                    "operational_p_prefix_tokens": (
                        request.operational_p_prefix_tokens),
                    "p_fresh_tokens": request.p_fresh_tokens,
                    "handoff_tokens": request.handoff_tokens,
                    "final_materialized_kv_tokens": (
                        request.final_materialized_kv_tokens),
                }
                for request_id, request in sorted(self.requests.items())
            },
        }


__all__ = [
    "MAX_P4D4_PREFILL_CHUNK_TOKENS",
    "MAX_P4D4_SEQUENCES",
    "P4D4ServingPool",
    "PDBatchItem",
    "PDHandoffJob",
    "PDPoolMetrics",
    "PDRequestState",
    "PDServingBatch",
    "PDServingRequest",
    "PDWorker",
]

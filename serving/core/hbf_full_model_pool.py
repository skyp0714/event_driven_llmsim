"""Continuous-batching serving pool for full-model HBF-GPU replicas."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import heapq
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from .hbf_full_model_astra import (
    ASTRA_NAMED_RESOURCE_TIMING_SEMANTICS,
    HBFAstraTimingAccounting,
    HBFModelAstraProjection,
    HBFModelAstraProjectionError,
    ORDERED_V2_FIDELITY,
    ORDERED_V2_SCHEMA,
    build_ordered_full_model_hbf_astra_projection,
    validate_hbf_astra_timing_metrics,
)
from .hbf_full_model_latency import (
    HBFModelBatchLatency,
    HBFModelBatchShape,
    HBFParallelLayout,
    HBFServerHardware,
    build_full_model_hbf_latency,
    qwen_logical_kv_bytes_per_token,
)
from .hbf_full_model_lifecycle import (
    PerGroupCapacityLedger,
    ResourceCalendar,
    hbf_kv_range_card_bytes,
    hbf_request_headroom_owner,
)


class HBFRequestState(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    PREFILL_DRAIN = "prefill_drain"
    DECODE = "decode"
    COMPLETE = "complete"


@dataclass
class HBFServingRequest:
    request_id: int
    session_id: str
    arrival_ns: int
    input_tokens: int
    output_tokens: int
    hbf_prefix_tokens: int
    lpddr_prefix_tokens: int
    group_id: int
    state: HBFRequestState = HBFRequestState.WAITING
    prefill_processed_tokens: int = 0
    generated_tokens: int = 0
    first_scheduled_ns: Optional[int] = None
    first_token_ns: Optional[int] = None
    completion_ns: Optional[int] = None
    token_completion_ns: list[int] = field(default_factory=list)
    batch_count: int = 0
    stage_ready_ns: Optional[int] = None
    admitted_hbf_prefix_tokens: int = field(init=False)
    admitted_lpddr_prefix_tokens: int = field(init=False)
    published_growth_tokens: int = 0
    prefill_drain_claimed: bool = False
    prefill_drain_job_id: Optional[int] = None
    prefill_drain_ready_ns: Optional[int] = None

    def __post_init__(self) -> None:
        self.admitted_hbf_prefix_tokens = self.hbf_prefix_tokens
        self.admitted_lpddr_prefix_tokens = self.lpddr_prefix_tokens

    def validate(self) -> None:
        for name in (
            "request_id",
            "arrival_ns",
            "input_tokens",
            "output_tokens",
            "hbf_prefix_tokens",
            "lpddr_prefix_tokens",
            "group_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.request_id < 0 or self.arrival_ns < 0:
            raise ValueError("request_id/arrival_ns must be non-negative")
        if self.input_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("input/output tokens must be positive")
        if self.hbf_prefix_tokens < 0 or self.lpddr_prefix_tokens < 0:
            raise ValueError("prefix tokens must be non-negative")
        if self.cached_tokens > self.input_tokens:
            raise ValueError("cached prefix exceeds request input")
        if self.input_tokens > 1_010_000:
            raise ValueError("request exceeds 1,010,000-token contract")
        if self.input_tokens + self.output_tokens - 1 > 1_010_000:
            raise ValueError(
                "request output would exceed 1,010,000-token contract")
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if (
            self.state != HBFRequestState.WAITING
            or self.prefill_processed_tokens
            or self.generated_tokens
            or self.first_scheduled_ns is not None
            or self.first_token_ns is not None
            or self.completion_ns is not None
            or self.token_completion_ns
            or self.batch_count
            or self.stage_ready_ns is not None
            or self.published_growth_tokens
            or self.prefill_drain_claimed
            or self.prefill_drain_job_id is not None
            or self.prefill_drain_ready_ns is not None
        ):
            raise ValueError(
                "submitted HBF request must be pristine")
        if (
            self.hbf_prefix_tokens
            != self.admitted_hbf_prefix_tokens
            or self.lpddr_prefix_tokens
            != self.admitted_lpddr_prefix_tokens
        ):
            raise ValueError(
                "submitted HBF request changed its admitted placement")

    @property
    def cached_tokens(self) -> int:
        return (
            self.admitted_hbf_prefix_tokens
            + self.admitted_lpddr_prefix_tokens
        )

    @property
    def published_tokens(self) -> int:
        return self.hbf_prefix_tokens + self.lpddr_prefix_tokens

    @property
    def fresh_tokens(self) -> int:
        return self.input_tokens - self.cached_tokens

    @property
    def prefill_remaining_tokens(self) -> int:
        return self.fresh_tokens - self.prefill_processed_tokens

    @property
    def active_lpddr_tokens(self) -> int:
        return (
            self.lpddr_prefix_tokens
            + self.prefill_processed_tokens
            + max(0, self.generated_tokens - 1)
            - self.published_growth_tokens
        )

    @property
    def ttft_ns(self) -> Optional[int]:
        if self.first_token_ns is None:
            return None
        return self.first_token_ns - self.arrival_ns

    @property
    def tpot_ns(self) -> Optional[float]:
        if self.output_tokens < 2 or self.completion_ns is None:
            return None
        assert self.first_token_ns is not None
        return (
            (self.completion_ns - self.first_token_ns)
            / (self.output_tokens - 1)
        )


@dataclass(frozen=True)
class BatchItem:
    request_id: int
    kind: str
    query_tokens: int


@dataclass(frozen=True)
class HBFServingBatch:
    batch_id: int
    group_id: int
    ready_ns: int
    start_ns: int
    completion_ns: Optional[int]
    items: tuple[BatchItem, ...]
    shape: HBFModelBatchShape
    latency: HBFModelBatchLatency


@dataclass(frozen=True)
class HBFExternalDispatch:
    """One immutable full-model batch awaiting an ASTRA completion."""

    arrival_ns: int
    batch: HBFServingBatch
    projection: HBFModelAstraProjection

    @property
    def job_id(self) -> str:
        return self.projection.job_id

    @property
    def stage_count(self) -> int:
        return len(self.projection.stages)

    def controller_arguments(
            self,
    ) -> tuple[str, int, tuple[dict[str, Any], ...]]:
        return self.projection.controller_command_arguments(
            self.arrival_ns)


@dataclass
class HBFWorker:
    group_id: int
    waiting: deque[int] = field(default_factory=deque)
    prefill_drain: deque[int] = field(default_factory=deque)
    active_decode: deque[int] = field(default_factory=deque)
    inflight: Optional[HBFServingBatch] = None
    pending_launch_ns: Optional[int] = None
    completed_batches: int = 0
    mixed_prefill_chunk_cap: Optional[int] = None


@dataclass
class HBFPoolMetrics:
    submitted_requests: int = 0
    completed_requests: int = 0
    batches: int = 0
    mixed_batches: int = 0
    prefill_only_batches: int = 0
    decode_only_batches: int = 0
    mixed_prefill_guard_considered: int = 0
    mixed_prefill_guard_limited: int = 0
    mixed_prefill_guard_deferred: int = 0
    mixed_prefill_guard_tokens_removed: int = 0
    mixed_prefill_guard_over_limit: int = 0
    mixed_prefill_guard_under_limit: int = 0
    mixed_prefill_guard_cap_updates: int = 0
    total_batch_items: int = 0
    prefill_query_tokens: int = 0
    decode_query_tokens: int = 0
    hbf_read_bytes_per_rank: int = 0
    lpddr_bytes_per_rank: int = 0
    collective_bytes_per_rank: int = 0
    modeled_batch_ns: int = 0
    embedding_modeled_ns: int = 0
    dense_modeled_ns: int = 0
    attention_modeled_ns: int = 0
    router_modeled_ns: int = 0
    moe_modeled_ns: int = 0
    final_modeled_ns: int = 0
    collective_modeled_ns: int = 0
    attention_compute_roof_ns: int = 0
    attention_hbf_roof_ns: int = 0
    attention_lpddr_roof_ns: int = 0
    attention_compute_dominant_batches: int = 0
    attention_hbf_dominant_batches: int = 0
    attention_lpddr_dominant_batches: int = 0
    resource_delay_ns: int = 0
    astra_completed_batches: int = 0
    astra_completion_elapsed_ns: int = 0
    astra_resource_delay_ns: int = 0
    astra_dependency_critical_path_ns: int = 0
    astra_solo_resource_serialized_completion_ns: int = 0
    astra_actual_resource_serialized_completion_ns: int = 0
    astra_internal_resource_serialization_wait_ns: int = 0
    astra_signed_interference_delta_ns: int = 0
    lpddr_capacity_deferrals: int = 0
    prefill_drain_candidates: int = 0
    prefill_drain_claimed: int = 0
    prefill_drain_started: int = 0
    prefill_drain_completed: int = 0
    prefill_drain_fallbacks: int = 0
    prefill_drain_logical_tokens: int = 0
    prefill_drain_wait_ns: int = 0
    max_batch_size: int = 0
    max_lpddr_active_bytes_per_card: int = 0


def derive_lpddr_workspace_bytes(
        layout: HBFParallelLayout, *, max_num_batched_tokens: int,
        max_num_seqs: int, fixed_scratch_bytes: int = 2 * 1024 ** 3) -> int:
    """Conservative active-kernel workspace per HBF card."""

    if max_num_batched_tokens <= 0 or max_num_seqs <= 0:
        raise ValueError("batch limits must be positive")
    if fixed_scratch_bytes < 0:
        raise ValueError("fixed scratch must be non-negative")
    hidden = 2_048
    head_dim = 128
    q_heads = 32 // layout.tp_size
    kv_heads = max(1, math.ceil(4 / layout.tp_size))
    qkv_width = (q_heads + 2 * kv_heads) * head_dim
    activation_double_buffers = (
        4 * max_num_batched_tokens * hidden * 2)
    qkv_buffers = 2 * max_num_batched_tokens * qkv_width * 2
    moe_dispatch_buffers = (
        2 * max_num_batched_tokens * hidden * 2)
    logits = (
        max_num_seqs * math.ceil(151_936 / layout.tp_size) * 2)
    total = (
        fixed_scratch_bytes
        + activation_double_buffers
        + qkv_buffers
        + moe_dispatch_buffers
        + logits
    )
    page = 2 * 1024 ** 2
    return int(math.ceil(total / page) * page)


class FullModelHBFServingPool:
    """One online queue per independent HBF TP replica."""

    _EXECUTION_BACKENDS = frozenset({
        "analytical_calendar",
        "external_astra",
    })

    def __init__(
            self, *, repo_root: Path, hardware: HBFServerHardware,
            layout: HBFParallelLayout,
            resource_calendar: Optional[ResourceCalendar] = None,
            lpddr_ledger: Optional[PerGroupCapacityLedger] = None,
            placement_resolver: Optional[
                Callable[[str], tuple[int, int, int]]
            ] = None,
            max_num_batched_tokens: int = 8_192,
            max_num_seqs: int = 128,
            max_prefill_chunk_tokens: int = 4_096,
            mixed_batch_latency_limit_ns: Optional[int] = None,
            prefill_drain_tail_tokens: Optional[int] = None,
            prefill_drain_min_tokens: int = 4_096,
            scheduling_policy: str = "decode_first",
            band: str = "central",
            validate_every_event: bool = True,
            retain_detailed_history: bool = True,
            retain_token_completion_history: Optional[bool] = None,
            execution_backend: str = "analytical_calendar",
            server_id: int = 0,
            analytical_resource_prefix: str = "") -> None:
        hardware.validate()
        layout.validate(hardware.card_count)
        if (
            not isinstance(execution_backend, str)
            or execution_backend not in self._EXECUTION_BACKENDS
        ):
            raise ValueError(
                "execution_backend must be one of "
                f"{sorted(self._EXECUTION_BACKENDS)}")
        if not isinstance(analytical_resource_prefix, str):
            raise ValueError(
                "analytical_resource_prefix must be a string")
        if (
            execution_backend == "external_astra"
            and analytical_resource_prefix
        ):
            raise ValueError(
                "analytical_resource_prefix is unsupported with "
                "execution_backend='external_astra'")
        if (
            not isinstance(server_id, int)
            or isinstance(server_id, bool)
            or server_id < 0
        ):
            raise ValueError("server_id must be a non-negative integer")
        if (
            execution_backend == "external_astra"
            and resource_calendar is not None
        ):
            raise ValueError(
                "external_astra owns foreground resource timing in ASTRA; "
                "resource_calendar must be omitted")
        if not isinstance(validate_every_event, bool):
            raise ValueError("validate_every_event must be a boolean")
        if not isinstance(retain_detailed_history, bool):
            raise ValueError(
                "retain_detailed_history must be a boolean")
        if (
                retain_token_completion_history is not None
                and not isinstance(
                    retain_token_completion_history, bool)
        ):
            raise ValueError(
                "retain_token_completion_history must be a boolean")
        if max_num_batched_tokens <= 0 or max_num_seqs <= 0:
            raise ValueError("batch limits must be positive")
        if max_num_seqs > 128:
            raise ValueError(
                "max_num_seqs exceeds calibrated support (128)")
        if not 0 < max_prefill_chunk_tokens <= max_num_batched_tokens:
            raise ValueError(
                "max_prefill_chunk_tokens must be in 1..token budget")
        if max_prefill_chunk_tokens > 131_072:
            raise ValueError(
                "max_prefill_chunk_tokens exceeds calibrated support "
                "(131072)")
        if (
            mixed_batch_latency_limit_ns is not None
            and (
                not isinstance(mixed_batch_latency_limit_ns, int)
                or isinstance(mixed_batch_latency_limit_ns, bool)
                or mixed_batch_latency_limit_ns <= 0
            )
        ):
            raise ValueError(
                "mixed_batch_latency_limit_ns must be a positive "
                "integer or None")
        if (
            prefill_drain_tail_tokens is not None
            and (
                not isinstance(prefill_drain_tail_tokens, int)
                or isinstance(prefill_drain_tail_tokens, bool)
                or prefill_drain_tail_tokens < 0
            )
        ):
            raise ValueError(
                "prefill_drain_tail_tokens must be a non-negative "
                "integer or None")
        if (
            not isinstance(prefill_drain_min_tokens, int)
            or isinstance(prefill_drain_min_tokens, bool)
            or prefill_drain_min_tokens < 0
        ):
            raise ValueError(
                "prefill_drain_min_tokens must be a non-negative integer")
        if scheduling_policy != "decode_first":
            raise ValueError(
                "full-model HBF pool currently supports decode_first")
        self.hardware = hardware
        self.layout = layout
        self.execution_backend = execution_backend
        self.server_id = server_id
        self.analytical_resource_prefix = analytical_resource_prefix
        self.calendar = (
            resource_calendar
            if resource_calendar is not None else ResourceCalendar()
        )
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.max_prefill_chunk_tokens = max_prefill_chunk_tokens
        self.mixed_batch_latency_limit_ns = (
            mixed_batch_latency_limit_ns)
        self.prefill_drain_tail_tokens = prefill_drain_tail_tokens
        self.prefill_drain_min_tokens = prefill_drain_min_tokens
        self.scheduling_policy = scheduling_policy
        self.validate_every_event = validate_every_event
        self.retain_detailed_history = retain_detailed_history
        self.retain_token_completion_history = (
            retain_detailed_history
            if retain_token_completion_history is None
            else retain_token_completion_history
        )
        self.workspace_bytes_per_card = derive_lpddr_workspace_bytes(
            layout,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        )
        if self.workspace_bytes_per_card >= (
                hardware.lpddr_capacity_bytes_per_card):
            raise ValueError(
                "derived HBF workspace does not fit LPDDR: "
                f"required={self.workspace_bytes_per_card}, "
                f"capacity={hardware.lpddr_capacity_bytes_per_card}")
        self.lpddr_kv_capacity_bytes_per_card = (
            hardware.lpddr_capacity_bytes_per_card
            - self.workspace_bytes_per_card
        )
        self.lpddr_ledger = (
            lpddr_ledger
            if lpddr_ledger is not None
            else PerGroupCapacityLedger(
                group_count=layout.replicas,
                capacity_bytes=self.lpddr_kv_capacity_bytes_per_card,
                card_ids_by_group={
                    group_id: tuple(range(
                        group_id * layout.tp_size,
                        (group_id + 1) * layout.tp_size,
                    ))
                    for group_id in range(layout.replicas)
                },
            )
        )
        if self.lpddr_ledger.group_count != layout.replicas:
            raise ValueError(
                "LPDDR ledger group count does not match HBF layout")
        if self.lpddr_ledger.capacity_bytes > (
                self.lpddr_kv_capacity_bytes_per_card):
            raise ValueError(
                "LPDDR ledger overlaps derived model workspace")
        self.lpddr_ledger.configure_cards({
            group_id: tuple(range(
                group_id * layout.tp_size,
                (group_id + 1) * layout.tp_size,
            ))
            for group_id in range(layout.replicas)
        })
        self.placement_resolver = placement_resolver
        self.kv_bytes_per_token = qwen_logical_kv_bytes_per_token()
        self.kv_bytes_per_active_token_per_card = int(math.ceil(
            self.kv_bytes_per_token
            * layout.physical_kv_replication_factor
            / layout.tp_size
        ))
        shared_model = build_full_model_hbf_latency(
            repo_root=repo_root,
            hardware=hardware,
            layout=layout,
            band=band,
        )
        self.models = (shared_model,) * layout.replicas
        self.workers = tuple(
            HBFWorker(
                group_id=index,
                mixed_prefill_chunk_cap=(
                    None
                    if mixed_batch_latency_limit_ns is None
                    else max_prefill_chunk_tokens
                ),
            )
            for index in range(layout.replicas)
        )
        self.requests: dict[int, HBFServingRequest] = {}
        self._current_request_by_session: dict[str, int] = {}
        self.metrics = HBFPoolMetrics()
        self.batch_history: list[HBFServingBatch] = []
        self._completed_request_ids: deque[int] = deque()
        self._completion_heap: list[tuple[int, int, int]] = []
        self._launch_heap: list[tuple[int, int]] = []
        self._external_outbox: deque[HBFExternalDispatch] = deque()
        self._external_pending: dict[str, HBFExternalDispatch] = {}
        self._external_issued_job_ids: set[str] = set()
        self._external_completed_job_ids: set[str] = set()
        self._next_batch_id = 1
        self.current_ns = 0

    def _worker(self, group_id: int) -> HBFWorker:
        if not 0 <= group_id < len(self.workers):
            raise ValueError(f"invalid HBF group_id={group_id}")
        return self.workers[group_id]

    def _cards(self, group_id: int) -> tuple[int, ...]:
        start = group_id * self.layout.tp_size
        return tuple(range(start, start + self.layout.tp_size))

    @staticmethod
    def _lpddr_owner(session_id: str) -> str:
        return f"hbf-session:{session_id}"

    def _request_lpddr_bytes(
            self, request: HBFServingRequest) -> int:
        return max(
            self._request_lpddr_card_bytes(request).values(),
            default=0,
        )

    def _range_card_bytes(
            self, group_id: int, *,
            token_start: int, token_count: int) -> dict[int, int]:
        return hbf_kv_range_card_bytes(
            layout=self.layout,
            card_ids=self._cards(group_id),
            kv_bytes_per_token=self.kv_bytes_per_token,
            token_start=token_start,
            token_count=token_count,
        )

    def _request_lpddr_card_bytes(
            self, request: HBFServingRequest) -> dict[int, int]:
        return self._range_card_bytes(
            request.group_id,
            token_start=request.hbf_prefix_tokens,
            token_count=request.active_lpddr_tokens,
        )

    @staticmethod
    def _request_growth_tokens(request: HBFServingRequest) -> int:
        return request.fresh_tokens + request.output_tokens - 1

    def _request_headroom_bytes(
            self, request: HBFServingRequest) -> int:
        return max(
            self._request_headroom_card_bytes(request).values(),
            default=0,
        )

    def _request_headroom_card_bytes(
            self, request: HBFServingRequest) -> dict[int, int]:
        return self._range_card_bytes(
            request.group_id,
            token_start=request.cached_tokens,
            token_count=self._request_growth_tokens(request),
        )

    def _remaining_headroom_card_bytes(
            self, request: HBFServingRequest) -> dict[int, int]:
        token_count = (
            request.prefill_remaining_tokens
            + request.output_tokens
            - max(request.generated_tokens, 1)
        )
        return self._range_card_bytes(
            request.group_id,
            token_start=(
                request.hbf_prefix_tokens
                + request.active_lpddr_tokens
            ),
            token_count=token_count,
        )

    def _vectors_equal(
            self, group_id: int, left: Mapping[int, int],
            right: Mapping[int, int]) -> bool:
        return all(
            left.get(card_id, 0) == right.get(card_id, 0)
            for card_id in self._cards(group_id)
        )

    def _projected_lpddr_fits(
            self, group_id: int,
            additions: Mapping[int, int],
            *, only_request_id: Optional[int] = None) -> bool:
        """Check that a batch's LPDDR headroom projection still holds.

        Batch selection adds one request at a time and re-checks after each,
        so re-scanning every accumulated addition made this quadratic in
        batch size -- the dominant term in large-cohort cells.  Each
        request's headroom test reads only that request's own placement and
        its own ledger reservation, and a request is refreshed exactly once
        before it is added, so an already-verified entry cannot change
        during one selection.  ``only_request_id`` therefore checks just the
        new entry; omitting it keeps the full scan for other callers.
        """

        projected_by_card = self.lpddr_ledger.used_bytes_by_card(group_id)
        projected_bytes = max(projected_by_card.values(), default=0)
        self.metrics.max_lpddr_active_bytes_per_card = max(
            self.metrics.max_lpddr_active_bytes_per_card,
            projected_bytes,
        )
        if projected_bytes > self.lpddr_ledger.capacity_bytes:
            return False
        if only_request_id is not None:
            if only_request_id not in additions:
                raise RuntimeError(
                    "incremental LPDDR projection needs its own addition")
            checked = ((only_request_id, additions[only_request_id]),)
        else:
            checked = tuple(additions.items())
        for request_id, token_count in checked:
            request = self.requests[request_id]
            if request.group_id != group_id:
                raise RuntimeError(
                    "LPDDR batch projection crossed replica groups")
            requested = self._range_card_bytes(
                group_id,
                token_start=(
                    request.hbf_prefix_tokens
                    + request.active_lpddr_tokens
                ),
                token_count=token_count,
            )
            available = self.lpddr_ledger.owner_card_bytes(
                hbf_request_headroom_owner(request_id))
            if any(
                    requested[card_id]
                    > available.get(card_id, 0)
                    for card_id in requested):
                return False
        return True

    def submit_many(
            self, requests: Iterable[HBFServingRequest],
            *, now_ns: int,
            defer_schedule: bool = False) -> None:
        # Completions at the same timestamp are applied first, but their next
        # batch is deferred until every co-timed arrival below is visible.
        if not isinstance(defer_schedule, bool):
            raise ValueError("defer_schedule must be a boolean")
        self.advance(now_ns, defer_schedule=True)
        values = list(requests)
        seen = set()
        seen_sessions = set()
        projected = {
            group_id: dict(
                self.lpddr_ledger.used_bytes_by_card(group_id))
            for group_id in range(len(self.workers))
        }
        for request in values:
            request.validate()
            if request.arrival_ns > now_ns:
                raise ValueError(
                    "request logical arrival cannot be after submit time")
            if (
                request.request_id in self.requests
                or request.request_id in seen
            ):
                raise ValueError(
                    f"duplicate request_id={request.request_id}")
            self._worker(request.group_id)
            current_id = self._current_request_by_session.get(
                request.session_id)
            if (
                current_id is not None
                and self.requests[current_id].state
                != HBFRequestState.COMPLETE
            ):
                raise ValueError(
                    f"session {request.session_id!r} already has "
                    "a live HBF request")
            if request.session_id in seen_sessions:
                raise ValueError(
                    f"duplicate live session {request.session_id!r}")
            seen.add(request.request_id)
            seen_sessions.add(request.session_id)
            session_owner = self._lpddr_owner(request.session_id)
            headroom_owner = hbf_request_headroom_owner(
                request.request_id)
            for owner in (session_owner, headroom_owner):
                old_group = self.lpddr_ledger.owner_group(owner)
                if old_group is not None:
                    for card_id, byte_count in (
                            self.lpddr_ledger.owner_card_bytes(
                                owner).items()):
                        projected[old_group][card_id] -= byte_count
            initial_lpddr = self._range_card_bytes(
                request.group_id,
                token_start=request.hbf_prefix_tokens,
                token_count=request.lpddr_prefix_tokens,
            )
            initial_headroom = self._request_headroom_card_bytes(request)
            for card_id in self._cards(request.group_id):
                projected[request.group_id][card_id] += (
                    initial_lpddr[card_id]
                    + initial_headroom[card_id]
                )
        for group_id, bytes_by_card in projected.items():
            overflow = {
                card_id: byte_count
                for card_id, byte_count in bytes_by_card.items()
                if byte_count > self.lpddr_ledger.capacity_bytes
            }
            if overflow:
                raise RuntimeError(
                    f"initial HBF request LPDDR exceeds capacity: "
                    f"group={group_id}, requested={overflow}, "
                    f"capacity={self.lpddr_ledger.capacity_bytes}")
        for request in values:
            self.lpddr_ledger.set_card_bytes(
                request.group_id,
                self._lpddr_owner(request.session_id),
                self._range_card_bytes(
                    request.group_id,
                    token_start=request.hbf_prefix_tokens,
                    token_count=request.lpddr_prefix_tokens,
                ),
            )
            self.lpddr_ledger.set_card_bytes(
                request.group_id,
                hbf_request_headroom_owner(request.request_id),
                self._request_headroom_card_bytes(request),
            )
            request.state = HBFRequestState.PREFILL
            request.stage_ready_ns = now_ns
            self.requests[request.request_id] = request
            self._current_request_by_session[request.session_id] = (
                request.request_id)
            self._worker(request.group_id).waiting.append(
                request.request_id)
            self.metrics.submitted_requests += 1
            self.metrics.max_lpddr_active_bytes_per_card = max(
                self.metrics.max_lpddr_active_bytes_per_card,
                self.lpddr_ledger.used_bytes(request.group_id),
            )
        if not defer_schedule:
            self.flush_scheduling(now_ns)

    def submit(
            self, request: HBFServingRequest, *, now_ns: int,
            defer_schedule: bool = False) -> None:
        self.submit_many(
            (request,),
            now_ns=now_ns,
            defer_schedule=defer_schedule,
        )

    def _refresh_request_placement(
            self, request: HBFServingRequest) -> None:
        if self.placement_resolver is None:
            return
        old_published = request.published_tokens
        hbf_tokens, lpddr_tokens, group_id = (
            self.placement_resolver(request.session_id))
        if group_id != request.group_id:
            raise RuntimeError(
                "active HBF request changed replica group")
        if hbf_tokens + lpddr_tokens != old_published:
            raise RuntimeError(
                "active HBF request placement changed published tokens: "
                f"old={old_published}, "
                f"new={hbf_tokens + lpddr_tokens}")
        request.hbf_prefix_tokens = hbf_tokens
        request.lpddr_prefix_tokens = lpddr_tokens
        expected = self._request_lpddr_card_bytes(request)
        actual = self.lpddr_ledger.owner_card_bytes(
            self._lpddr_owner(request.session_id))
        if not self._vectors_equal(
                request.group_id, actual, expected):
            raise RuntimeError(
                "HBF placement refresh disagrees with LPDDR ledger: "
                f"session={request.session_id!r}, "
                f"expected={expected}, actual={actual}")

    def _select_batch(
            self, worker: HBFWorker,
    ) -> tuple[tuple[BatchItem, ...], HBFModelBatchShape] | None:
        items: list[BatchItem] = []
        prefill_q: list[int] = []
        prefill_hbf_k: list[int] = []
        prefill_lpddr_k: list[int] = []
        decode_hbf_k: list[int] = []
        decode_lpddr_k: list[int] = []
        additions: dict[int, int] = {}
        token_budget = self.max_num_batched_tokens
        seq_budget = self.max_num_seqs

        def batch_shape() -> HBFModelBatchShape:
            return HBFModelBatchShape(
                total_tokens=sum(
                    item.query_tokens for item in items),
                prefill_q=tuple(prefill_q),
                prefill_hbf_k=tuple(prefill_hbf_k),
                prefill_lpddr_k=tuple(prefill_lpddr_k),
                decode_hbf_k=tuple(decode_hbf_k),
                decode_lpddr_k=tuple(decode_lpddr_k),
                lm_head_sequences=len(items),
            )

        decode_count = len(worker.active_decode)
        for _ in range(decode_count):
            if token_budget <= 0 or seq_budget <= 0:
                break
            request_id = worker.active_decode.popleft()
            request = self.requests[request_id]
            self._refresh_request_placement(request)
            if request.state != HBFRequestState.DECODE:
                raise RuntimeError("decode queue contains non-decode request")
            additions[request_id] = 1
            if not self._projected_lpddr_fits(
                    worker.group_id, additions,
                    only_request_id=request_id):
                additions.pop(request_id)
                worker.active_decode.appendleft(request_id)
                self.metrics.lpddr_capacity_deferrals += 1
                break
            items.append(BatchItem(request_id, "decode", 1))
            decode_hbf_k.append(request.hbf_prefix_tokens)
            decode_lpddr_k.append(request.active_lpddr_tokens)
            token_budget -= 1
            seq_budget -= 1

        waiting_count = len(worker.waiting)
        deferred_waiting: list[int] = []
        for _ in range(waiting_count):
            if token_budget <= 0 or seq_budget <= 0:
                break
            request_id = worker.waiting.popleft()
            request = self.requests[request_id]
            self._refresh_request_placement(request)
            if request.state != HBFRequestState.PREFILL:
                raise RuntimeError("prefill queue contains invalid request")
            remaining = request.prefill_remaining_tokens
            if remaining == 0:
                additions[request_id] = 0
                if not self._projected_lpddr_fits(
                        worker.group_id, additions,
                        only_request_id=request_id):
                    additions.pop(request_id)
                    deferred_waiting.append(request_id)
                    self.metrics.lpddr_capacity_deferrals += 1
                    continue
                items.append(BatchItem(
                    request_id, "first_decode", 1))
                decode_hbf_k.append(request.hbf_prefix_tokens)
                decode_lpddr_k.append(request.active_lpddr_tokens)
                token_budget -= 1
                seq_budget -= 1
                continue
            chunk = min(
                remaining,
                token_budget,
                self.max_prefill_chunk_tokens,
            )
            original_chunk = chunk
            if (
                decode_hbf_k
                and self.mixed_batch_latency_limit_ns is not None
            ):
                self.metrics.mixed_prefill_guard_considered += 1
                cap = worker.mixed_prefill_chunk_cap
                if cap is None:
                    raise RuntimeError(
                        "mixed-prefill guard lacks a worker cap")
                chunk = min(chunk, cap)
                if chunk == 0:
                    deferred_waiting.append(request_id)
                    self.metrics.mixed_prefill_guard_deferred += 1
                    self.metrics.mixed_prefill_guard_tokens_removed += (
                        original_chunk)
                    continue
                if chunk < original_chunk:
                    self.metrics.mixed_prefill_guard_limited += 1
                    self.metrics.mixed_prefill_guard_tokens_removed += (
                        original_chunk - chunk)
            additions[request_id] = chunk
            if not self._projected_lpddr_fits(
                    worker.group_id, additions,
                    only_request_id=request_id):
                additions.pop(request_id)
                deferred_waiting.append(request_id)
                self.metrics.lpddr_capacity_deferrals += 1
                continue
            items.append(BatchItem(request_id, "prefill", chunk))
            prefill_q.append(chunk)
            prefill_hbf_k.append(request.hbf_prefix_tokens)
            prefill_lpddr_k.append(
                request.lpddr_prefix_tokens
                + request.prefill_processed_tokens
            )
            token_budget -= chunk
            seq_budget -= 1
        for request_id in reversed(deferred_waiting):
            worker.waiting.appendleft(request_id)

        if not items:
            return None
        shape = batch_shape()
        return tuple(items), shape

    def _should_prefill_drain(
            self, request: HBFServingRequest) -> bool:
        if self.prefill_drain_tail_tokens is None:
            return False
        if request.generated_tokens >= request.output_tokens:
            return False
        drainable_tokens = (
            request.active_lpddr_tokens
            - self.prefill_drain_tail_tokens
        )
        return (
            drainable_tokens > 0
            and drainable_tokens >= self.prefill_drain_min_tokens
        )

    def _gate_prefill_drain(
            self, request: HBFServingRequest,
            worker: HBFWorker, now_ns: int) -> None:
        if request.state not in {
                HBFRequestState.PREFILL,
                HBFRequestState.DECODE,
        }:
            raise RuntimeError(
                "prefill drain can gate only a first-token request")
        if request.generated_tokens != 1:
            raise RuntimeError(
                "prefill drain must begin immediately after first token")
        if not self._should_prefill_drain(request):
            raise RuntimeError(
                "prefill drain gate does not satisfy its policy")
        request.state = HBFRequestState.PREFILL_DRAIN
        request.stage_ready_ns = now_ns
        request.prefill_drain_ready_ns = now_ns
        worker.prefill_drain.append(request.request_id)
        self.metrics.prefill_drain_candidates += 1

    def claim_prefill_drain_requests(
            self) -> tuple[HBFServingRequest, ...]:
        """Claim each newly gated request exactly once for lifecycle work."""

        result = []
        for worker in self.workers:
            for request_id in worker.prefill_drain:
                request = self.requests[request_id]
                if request.state != HBFRequestState.PREFILL_DRAIN:
                    raise RuntimeError(
                        "prefill-drain queue contains an invalid request")
                if request.prefill_drain_claimed:
                    continue
                request.prefill_drain_claimed = True
                result.append(request)
        self.metrics.prefill_drain_claimed += len(result)
        if self.validate_every_event:
            self.assert_invariants()
        return tuple(result)

    def publish_prefill_drain_placement(
            self, request_id: int, *,
            hbf_tokens: int, lpddr_tokens: int) -> None:
        """Publish lifecycle ownership after it adopts materialized input."""

        request = self.requests[request_id]
        if (
            request.state != HBFRequestState.PREFILL_DRAIN
            or not request.prefill_drain_claimed
        ):
            raise RuntimeError(
                "only a claimed prefill drain may publish placement")
        if (
            not isinstance(hbf_tokens, int)
            or isinstance(hbf_tokens, bool)
            or not isinstance(lpddr_tokens, int)
            or isinstance(lpddr_tokens, bool)
            or hbf_tokens < 0
            or lpddr_tokens < 0
        ):
            raise ValueError(
                "published HBF/LPDDR tokens must be non-negative integers")
        published_tokens = hbf_tokens + lpddr_tokens
        published_growth = published_tokens - request.cached_tokens
        materialized_growth = (
            request.prefill_processed_tokens
            + max(0, request.generated_tokens - 1)
        )
        if hbf_tokens < request.hbf_prefix_tokens:
            raise RuntimeError(
                "prefill drain cannot move a published HBF prefix "
                "back to LPDDR")
        if published_tokens < request.published_tokens:
            raise RuntimeError(
                "prefill drain cannot shrink published materialized KV")
        if not 0 <= published_growth <= materialized_growth:
            raise RuntimeError(
                "prefill drain published an impossible materialized "
                f"range: growth={published_growth}, "
                f"materialized={materialized_growth}")
        active_lpddr_tokens = (
            lpddr_tokens
            + materialized_growth
            - published_growth
        )
        expected = self._range_card_bytes(
            request.group_id,
            token_start=hbf_tokens,
            token_count=active_lpddr_tokens,
        )
        actual = self.lpddr_ledger.owner_card_bytes(
            self._lpddr_owner(request.session_id))
        if not self._vectors_equal(
                request.group_id, actual, expected):
            raise RuntimeError(
                "published prefill-drain placement disagrees with "
                "the LPDDR ledger")
        request.hbf_prefix_tokens = hbf_tokens
        request.lpddr_prefix_tokens = lpddr_tokens
        request.published_growth_tokens = published_growth
        if self.validate_every_event:
            self.assert_invariants()

    def bind_prefill_drain_job(
            self, request_id: int, *, job_id: int,
            logical_tokens: int) -> None:
        request = self.requests[request_id]
        if (
            request.state != HBFRequestState.PREFILL_DRAIN
            or not request.prefill_drain_claimed
            or request.prefill_drain_job_id is not None
        ):
            raise RuntimeError(
                "prefill drain job binding has invalid ownership")
        if (
            not isinstance(job_id, int)
            or isinstance(job_id, bool)
            or job_id <= 0
        ):
            raise ValueError("prefill drain job_id must be positive")
        if (
            not isinstance(logical_tokens, int)
            or isinstance(logical_tokens, bool)
            or logical_tokens <= 0
        ):
            raise ValueError(
                "prefill drain logical_tokens must be positive")
        expected_tokens = (
            request.active_lpddr_tokens
            - int(self.prefill_drain_tail_tokens)
        )
        if logical_tokens != expected_tokens:
            raise RuntimeError(
                "prefill drain job token count differs from the "
                f"live placement: expected={expected_tokens}, "
                f"actual={logical_tokens}")
        request.prefill_drain_job_id = job_id
        self.metrics.prefill_drain_started += 1
        self.metrics.prefill_drain_logical_tokens += logical_tokens
        if self.validate_every_event:
            self.assert_invariants()

    def clear_prefill_drain_job(
            self, request_id: int, *, job_id: int) -> None:
        """Acknowledge one drain callback while retaining the decode gate."""

        request = self.requests[request_id]
        if (
            request.state != HBFRequestState.PREFILL_DRAIN
            or not request.prefill_drain_claimed
            or request.prefill_drain_job_id != job_id
        ):
            raise RuntimeError(
                "prefill drain callback job identity mismatch")
        request.prefill_drain_job_id = None
        if self.validate_every_event:
            self.assert_invariants()

    def release_prefill_drain(
            self, request_id: int, *, now_ns: int,
            job_id: Optional[int] = None,
            fallback: bool = False) -> None:
        """Release one drain gate after commit or explicit LPDDR fallback."""

        if now_ns != self.current_ns:
            raise ValueError(
                "prefill drain release must use the current pool time")
        request = self.requests[request_id]
        if (
            request.state != HBFRequestState.PREFILL_DRAIN
            or not request.prefill_drain_claimed
        ):
            raise RuntimeError(
                "only a claimed prefill drain may be released")
        if job_id != request.prefill_drain_job_id:
            raise RuntimeError(
                "prefill drain completion job identity mismatch")
        if fallback and request.prefill_drain_job_id is not None:
            raise RuntimeError(
                "prefill drain fallback cannot abandon a bound job")
        if (
            not fallback
            and request.active_lpddr_tokens
            > int(self.prefill_drain_tail_tokens)
        ):
            raise RuntimeError(
                "prefill drain cannot release decode before the "
                "configured LPDDR tail is satisfied")
        if now_ns < (request.prefill_drain_ready_ns or 0):
            raise ValueError(
                "prefill drain completion precedes its first token")
        worker = self._worker(request.group_id)
        try:
            worker.prefill_drain.remove(request_id)
        except ValueError as exc:
            raise RuntimeError(
                "prefill drain request lost queue ownership") from exc
        ready_ns = int(request.prefill_drain_ready_ns)
        request.prefill_drain_job_id = None
        request.prefill_drain_claimed = False
        request.prefill_drain_ready_ns = None
        request.state = HBFRequestState.DECODE
        request.stage_ready_ns = now_ns
        worker.active_decode.append(request_id)
        if fallback:
            self.metrics.prefill_drain_fallbacks += 1
        else:
            self.metrics.prefill_drain_completed += 1
        self.metrics.prefill_drain_wait_ns += (
            now_ns - ready_ns)
        if self.validate_every_event:
            self.assert_invariants()

    def _analytical_local_resource(self, resource: str) -> str:
        """Namespace one HBF-server-local analytical resource."""

        return f"{self.analytical_resource_prefix}{resource}"

    def _foreground_resource_names(
            self, group_id: int) -> tuple[str, ...]:
        names = [
            self._analytical_local_resource(
                f"hbf-group-{group_id}-npu"),
            self._analytical_local_resource(
                f"hbf-group-{group_id}-fabric"),
        ]
        for card_id in self._cards(group_id):
            names.extend((
                self._analytical_local_resource(
                    f"hbf-card-{card_id}-media"),
                self._analytical_local_resource(
                    f"hbf-card-{card_id}-lpddr"),
            ))
        return tuple(names)

    def _schedule_launch(self, group_id: int, launch_ns: int) -> None:
        worker = self._worker(group_id)
        if worker.pending_launch_ns == launch_ns:
            return
        worker.pending_launch_ns = launch_ns
        heapq.heappush(self._launch_heap, (launch_ns, group_id))

    def _update_mixed_prefill_guard(
            self, worker: HBFWorker,
            shape: HBFModelBatchShape,
            latency: HBFModelBatchLatency) -> None:
        """Update the next mixed-prefill cap from one observed model result.

        The batch latency is already required for execution, so this
        feedback path adds no extra latency-model queries.  Only information
        available when the batch is launched is used.
        """

        limit_ns = self.mixed_batch_latency_limit_ns
        if limit_ns is None or not shape.decode_hbf_k:
            return
        cap = worker.mixed_prefill_chunk_cap
        if cap is None:
            raise RuntimeError(
                "mixed-prefill guard lacks a worker cap")
        has_prefill = bool(shape.prefill_q)
        new_cap = cap
        if has_prefill and latency.total_ns > limit_ns:
            self.metrics.mixed_prefill_guard_over_limit += 1
            scaled = int(math.floor(
                cap * limit_ns / latency.total_ns * 0.80))
            if cap > 0:
                scaled = min(scaled, cap - 1)
            new_cap = max(0, scaled)
        elif (
            has_prefill
            and latency.total_ns * 5 <= limit_ns * 4
        ):
            self.metrics.mixed_prefill_guard_under_limit += 1
            new_cap = min(
                self.max_prefill_chunk_tokens,
                max(cap + 1, int(math.ceil(cap * 1.25))),
            )
        elif (
            not has_prefill
            and worker.waiting
            and cap == 0
            and latency.total_ns < limit_ns
        ):
            # Probe one prefill token after a decode-only recovery batch.
            self.metrics.mixed_prefill_guard_under_limit += 1
            new_cap = 1
        if new_cap != cap:
            worker.mixed_prefill_chunk_cap = new_cap
            self.metrics.mixed_prefill_guard_cap_updates += 1

    def _try_schedule(self, group_id: int, now_ns: int) -> None:
        worker = self._worker(group_id)
        if worker.inflight is not None:
            return
        if not worker.waiting and not worker.active_decode:
            worker.pending_launch_ns = None
            return
        if self.execution_backend == "analytical_calendar":
            available_ns = self.calendar.earliest_start(
                now_ns, self._foreground_resource_names(group_id))
            if available_ns > now_ns:
                self._schedule_launch(group_id, available_ns)
                return
        worker.pending_launch_ns = None
        selected = self._select_batch(worker)
        if selected is None:
            return
        items, shape = selected
        latency = self.models[group_id].batch_latency(shape)
        self._update_mixed_prefill_guard(
            worker, shape, latency)
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        ready_values = [
            self.requests[item.request_id].stage_ready_ns
            for item in items
        ]
        if any(value is None for value in ready_values):
            raise RuntimeError("scheduled HBF request lacks ready time")
        ready_ns = min(ready_values)
        if self.execution_backend == "analytical_calendar":
            demands = {
                self._analytical_local_resource(
                    f"hbf-group-{group_id}-npu"
                ): (
                    latency.total_ns,
                    0,
                ),
                self._analytical_local_resource(
                    f"hbf-group-{group_id}-fabric"
                ): (
                    latency.collective_ns,
                    latency.collective_bytes_per_rank,
                ),
            }
            hbf_service_ns = min(
                latency.total_ns, latency.hbf_roof_ns_sum)
            lpddr_service_ns = min(
                latency.total_ns, latency.lpddr_roof_ns_sum)
            for card_id in self._cards(group_id):
                demands[self._analytical_local_resource(
                    f"hbf-card-{card_id}-media"
                )] = (
                    hbf_service_ns,
                    latency.hbf_read_bytes_per_rank,
                )
                demands[self._analytical_local_resource(
                    f"hbf-card-{card_id}-lpddr"
                )] = (
                    lpddr_service_ns,
                    latency.lpddr_bytes_per_rank,
                )
            start_ns, completion_ns = self.calendar.reserve_parallel(
                arrival_ns=now_ns,
                job_id=batch_id,
                kind="hbf-model-batch",
                namespace="hbf-pool",
                demands=demands,
            )
            projection = None
        else:
            start_ns = now_ns
            completion_ns = None
            projection = build_ordered_full_model_hbf_astra_projection(
                plan=self.models[group_id].batch_execution_plan(shape),
                latency=latency,
                hardware=self.hardware,
                layout=self.layout,
                replica_id=group_id,
                batch_id=batch_id,
                server_id=self.server_id,
            )
        batch = HBFServingBatch(
            batch_id=batch_id,
            group_id=group_id,
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
        if self.execution_backend == "analytical_calendar":
            assert completion_ns is not None
            heapq.heappush(
                self._completion_heap,
                (completion_ns, group_id, batch_id),
            )
        else:
            assert projection is not None
            dispatch = HBFExternalDispatch(
                arrival_ns=now_ns,
                batch=batch,
                projection=projection,
            )
            if dispatch.job_id in self._external_pending:
                raise RuntimeError(
                    "duplicate HBF external ASTRA job id "
                    f"{dispatch.job_id!r}")
            self._external_pending[dispatch.job_id] = dispatch
            self._external_outbox.append(dispatch)
        for item in items:
            request = self.requests[item.request_id]
            if request.first_scheduled_ns is None:
                request.first_scheduled_ns = start_ns
            request.batch_count += 1
        has_prefill = any(item.kind == "prefill" for item in items)
        has_decode = any(item.kind != "prefill" for item in items)
        self.metrics.batches += 1
        self.metrics.total_batch_items += len(items)
        self.metrics.max_batch_size = max(
            self.metrics.max_batch_size, len(items))
        if has_prefill and has_decode:
            self.metrics.mixed_batches += 1
        elif has_prefill:
            self.metrics.prefill_only_batches += 1
        else:
            self.metrics.decode_only_batches += 1
        self.metrics.prefill_query_tokens += sum(
            item.query_tokens for item in items if item.kind == "prefill")
        self.metrics.decode_query_tokens += sum(
            item.query_tokens for item in items if item.kind != "prefill")
        self.metrics.hbf_read_bytes_per_rank += (
            latency.hbf_read_bytes_per_rank)
        self.metrics.lpddr_bytes_per_rank += latency.lpddr_bytes_per_rank
        self.metrics.collective_bytes_per_rank += (
            latency.collective_bytes_per_rank)
        self.metrics.modeled_batch_ns += latency.total_ns
        self.metrics.embedding_modeled_ns += latency.embedding_ns
        self.metrics.dense_modeled_ns += latency.dense_ns
        self.metrics.attention_modeled_ns += latency.attention_ns
        self.metrics.router_modeled_ns += latency.router_ns
        self.metrics.moe_modeled_ns += latency.moe_ns
        self.metrics.final_modeled_ns += latency.final_ns
        self.metrics.collective_modeled_ns += latency.collective_ns
        self.metrics.attention_compute_roof_ns += (
            latency.attention_compute_roof_ns)
        self.metrics.attention_hbf_roof_ns += (
            latency.attention_hbf_roof_ns)
        self.metrics.attention_lpddr_roof_ns += (
            latency.attention_lpddr_roof_ns)
        if latency.attention_dominant_roof == "compute":
            self.metrics.attention_compute_dominant_batches += 1
        elif latency.attention_dominant_roof == "hbf_read":
            self.metrics.attention_hbf_dominant_batches += 1
        elif latency.attention_dominant_roof == "lpddr":
            self.metrics.attention_lpddr_dominant_batches += 1
        else:
            raise RuntimeError(
                "HBF batch reported an unknown attention roof: "
                f"{latency.attention_dominant_roof!r}")
        self.metrics.resource_delay_ns += start_ns - ready_ns

    def _require_external_astra(self, operation: str) -> None:
        if self.execution_backend != "external_astra":
            raise RuntimeError(
                f"{operation} requires execution_backend='external_astra'")

    @staticmethod
    def _callback_nonnegative_int(name: str, value: object) -> int:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    def drain_external_dispatches(
            self) -> tuple[HBFExternalDispatch, ...]:
        """Return newly scheduled ASTRA jobs exactly once.

        Draining is the hand-off boundary: only drained jobs may be completed
        through :meth:`complete_external_dispatch`.
        """

        self._require_external_astra("drain_external_dispatches")
        dispatches = tuple(self._external_outbox)
        self._external_outbox.clear()
        for dispatch in dispatches:
            if self._external_pending.get(
                    dispatch.job_id) is not dispatch:
                raise RuntimeError(
                    "external ASTRA outbox/pending identity mismatch")
            if dispatch.job_id in self._external_issued_job_ids:
                raise RuntimeError(
                    "external ASTRA job was issued more than once")
            self._external_issued_job_ids.add(dispatch.job_id)
        if self.validate_every_event:
            self.assert_invariants()
        return dispatches

    def complete_external_dispatch(
            self, job_id: str, arrival_ns: int, completion_ns: int,
            stage_count: int, *,
            defer_schedule: bool = False) -> HBFServingBatch:
        """Apply one strict ASTRA completion to its exact pending batch."""

        self._require_external_astra("complete_external_dispatch")
        if not isinstance(defer_schedule, bool):
            raise ValueError("defer_schedule must be a boolean")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        arrival = self._callback_nonnegative_int(
            "arrival_ns", arrival_ns)
        completion = self._callback_nonnegative_int(
            "completion_ns", completion_ns)
        stages = self._callback_nonnegative_int(
            "stage_count", stage_count)
        if job_id in self._external_completed_job_ids:
            raise RuntimeError(
                f"duplicate external ASTRA completion for {job_id!r}")
        dispatch = self._external_pending.get(job_id)
        if dispatch is None:
            raise RuntimeError(
                f"unknown external ASTRA completion job {job_id!r}")
        if job_id not in self._external_issued_job_ids:
            raise RuntimeError(
                f"external ASTRA job {job_id!r} was not drained")
        if arrival != dispatch.arrival_ns:
            raise RuntimeError(
                "external ASTRA completion arrival mismatch: "
                f"job={job_id!r}, expected={dispatch.arrival_ns}, "
                f"actual={arrival}")
        if stages != dispatch.stage_count:
            raise RuntimeError(
                "external ASTRA completion stage-count mismatch: "
                f"job={job_id!r}, expected={dispatch.stage_count}, "
                f"actual={stages}")
        dependency_elapsed = (
            dispatch.projection.dependency_critical_path_ns())
        solo_resource_elapsed = (
            dispatch.projection
            .solo_resource_serialized_completion_ns()
        )
        minimum_completion = arrival + dependency_elapsed
        if completion < minimum_completion:
            raise RuntimeError(
                "external ASTRA completion precedes the projected "
                "dependency critical path bound: "
                f"job={job_id!r}, minimum={minimum_completion}, "
                f"actual={completion}")
        if completion < self.current_ns:
            raise RuntimeError(
                "external ASTRA completion moves pool time backwards: "
                f"current={self.current_ns}, actual={completion}")

        worker = self._worker(dispatch.batch.group_id)
        if worker.inflight is not dispatch.batch:
            raise RuntimeError(
                "external ASTRA completion batch identity mismatch")
        history_index = None
        if self.retain_detailed_history:
            matches = [
                index for index, batch in enumerate(self.batch_history)
                if batch.batch_id == dispatch.batch.batch_id
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "external ASTRA batch history identity mismatch")
            history_index = matches[0]

        self.advance(completion, defer_schedule=True)
        completed_batch = replace(
            dispatch.batch, completion_ns=completion)
        worker.inflight = completed_batch
        if history_index is not None:
            self.batch_history[history_index] = completed_batch
        del self._external_pending[job_id]
        self._external_issued_job_ids.remove(job_id)
        self._external_completed_job_ids.add(job_id)
        self.metrics.astra_completed_batches += 1
        actual_elapsed = completion - arrival
        timing = HBFAstraTimingAccounting(
            dependency_critical_path_ns=dependency_elapsed,
            solo_resource_serialized_completion_ns=(
                solo_resource_elapsed),
            actual_resource_serialized_completion_ns=actual_elapsed,
        )
        self.metrics.astra_completion_elapsed_ns += actual_elapsed
        self.metrics.astra_resource_delay_ns += timing.resource_delay_ns
        self.metrics.astra_dependency_critical_path_ns += (
            dependency_elapsed)
        self.metrics.astra_solo_resource_serialized_completion_ns += (
            solo_resource_elapsed)
        self.metrics.astra_actual_resource_serialized_completion_ns += (
            actual_elapsed)
        self.metrics.astra_internal_resource_serialization_wait_ns += (
            timing.internal_resource_serialization_wait_ns)
        self.metrics.astra_signed_interference_delta_ns += (
            timing.signed_interference_delta_ns)
        self._finish_batch(
            completed_batch, schedule_next=not defer_schedule)
        if self.validate_every_event:
            self.assert_invariants()
        return completed_batch

    def _complete_request(
            self, request: HBFServingRequest, now_ns: int) -> None:
        request.state = HBFRequestState.COMPLETE
        request.completion_ns = now_ns
        self._completed_request_ids.append(request.request_id)
        self.metrics.completed_requests += 1

    def _commit_lpddr_growth(
            self, request: HBFServingRequest,
            prior_active_bytes: Mapping[int, int]) -> None:
        current_active_bytes = self._request_lpddr_card_bytes(request)
        delta_bytes = {
            card_id: (
                current_active_bytes[card_id]
                - prior_active_bytes.get(card_id, 0)
            )
            for card_id in self._cards(request.group_id)
        }
        if any(byte_count < 0 for byte_count in delta_bytes.values()):
            raise RuntimeError(
                "HBF request execution shrank active LPDDR KV")
        headroom_owner = hbf_request_headroom_owner(
            request.request_id)
        if any(delta_bytes.values()):
            available = self.lpddr_ledger.owner_card_bytes(
                headroom_owner)
            if any(
                    delta_bytes[card_id]
                    > available.get(card_id, 0)
                    for card_id in delta_bytes):
                raise RuntimeError(
                    "HBF request exceeded its atomic LPDDR finish "
                    "reservation")
            self.lpddr_ledger.shrink_card_bytes(
                headroom_owner, delta_bytes)
            self.lpddr_ledger.set_card_bytes(
                request.group_id,
                self._lpddr_owner(request.session_id),
                current_active_bytes,
            )
        else:
            actual = self.lpddr_ledger.owner_card_bytes(
                self._lpddr_owner(request.session_id))
            if not self._vectors_equal(
                    request.group_id, actual, current_active_bytes):
                raise RuntimeError(
                    "HBF request LPDDR ownership changed without "
                    "execution progress")
        if (
            request.state == HBFRequestState.COMPLETE
            and any(
                self.lpddr_ledger.owner_card_bytes(
                    headroom_owner).values())
        ):
            raise RuntimeError(
                "completed HBF request retained LPDDR headroom")

    def _finish_batch(
            self, batch: HBFServingBatch, *,
            schedule_next: bool = True) -> None:
        if batch.completion_ns is None:
            raise RuntimeError(
                "cannot finish an HBF batch without a completion time")
        worker = self._worker(batch.group_id)
        if (
            worker.inflight is None
            or worker.inflight.batch_id != batch.batch_id
        ):
            raise RuntimeError("HBF batch completion identity mismatch")
        for item in batch.items:
            request = self.requests[item.request_id]
            prior_active_bytes = self._request_lpddr_card_bytes(request)
            if item.kind == "prefill":
                request.prefill_processed_tokens += item.query_tokens
                if request.prefill_remaining_tokens == 0:
                    request.generated_tokens += 1
                    request.first_token_ns = batch.completion_ns
                    if self.retain_token_completion_history:
                        request.token_completion_ns.append(
                            batch.completion_ns)
                    if request.generated_tokens == request.output_tokens:
                        self._complete_request(
                            request, batch.completion_ns)
                    elif self._should_prefill_drain(request):
                        self._gate_prefill_drain(
                            request, worker, batch.completion_ns)
                    else:
                        request.state = HBFRequestState.DECODE
                        request.stage_ready_ns = batch.completion_ns
                        worker.active_decode.append(request.request_id)
                else:
                    request.stage_ready_ns = batch.completion_ns
                    worker.waiting.append(request.request_id)
            elif item.kind in {"first_decode", "decode"}:
                if item.kind == "first_decode":
                    request.first_token_ns = batch.completion_ns
                    request.state = HBFRequestState.DECODE
                request.generated_tokens += 1
                if self.retain_token_completion_history:
                    request.token_completion_ns.append(
                        batch.completion_ns)
                if request.generated_tokens == request.output_tokens:
                    self._complete_request(request, batch.completion_ns)
                elif (
                    item.kind == "first_decode"
                    and self._should_prefill_drain(request)
                ):
                    self._gate_prefill_drain(
                        request, worker, batch.completion_ns)
                else:
                    request.stage_ready_ns = batch.completion_ns
                    worker.active_decode.append(request.request_id)
            else:
                raise RuntimeError(f"unknown HBF batch item {item.kind!r}")
            self._commit_lpddr_growth(
                request, prior_active_bytes)
            self.metrics.max_lpddr_active_bytes_per_card = max(
                self.metrics.max_lpddr_active_bytes_per_card,
                self.lpddr_ledger.used_bytes(request.group_id),
            )
        worker.inflight = None
        worker.completed_batches += 1
        if schedule_next:
            self._try_schedule(batch.group_id, batch.completion_ns)

    def advance(
            self, now_ns: int, *, defer_schedule: bool = False) -> None:
        if now_ns < self.current_ns:
            raise ValueError(
                f"time cannot move backwards: current={self.current_ns}, "
                f"requested={now_ns}")
        while True:
            while self._launch_heap:
                launch_ns, launch_group = self._launch_heap[0]
                if (
                    self._worker(launch_group).pending_launch_ns
                    == launch_ns
                ):
                    break
                heapq.heappop(self._launch_heap)
            completion_ns = (
                self._completion_heap[0][0]
                if self._completion_heap else math.inf
            )
            launch_ns = (
                self._launch_heap[0][0]
                if self._launch_heap else math.inf
            )
            event_ns = min(completion_ns, launch_ns)
            if event_ns > now_ns:
                break
            if completion_ns <= launch_ns:
                event_ns, group_id, batch_id = heapq.heappop(
                    self._completion_heap)
                worker = self._worker(group_id)
                if (
                    worker.inflight is None
                    or worker.inflight.batch_id != batch_id
                ):
                    raise RuntimeError("stale HBF batch completion")
                self._finish_batch(
                    worker.inflight,
                    schedule_next=not (
                        defer_schedule and event_ns == now_ns),
                )
            else:
                event_ns, group_id = heapq.heappop(
                    self._launch_heap)
                worker = self._worker(group_id)
                if worker.pending_launch_ns != event_ns:
                    continue
                worker.pending_launch_ns = None
                if not (defer_schedule and event_ns == now_ns):
                    self._try_schedule(group_id, event_ns)
            self.current_ns = event_ns
        self.current_ns = now_ns
        if self.validate_every_event:
            self.assert_invariants()

    def flush_scheduling(self, now_ns: int) -> None:
        if now_ns != self.current_ns:
            raise ValueError(
                "flush_scheduling must run at the current pool timestamp")
        for worker in self.workers:
            self._try_schedule(worker.group_id, now_ns)

    def next_completion_ns(self) -> Optional[int]:
        return (
            None if not self._completion_heap
            else self._completion_heap[0][0]
        )

    def next_event_ns(self) -> Optional[int]:
        """Return only completion times known by this Python process.

        An outstanding external ASTRA dispatch is deliberately not converted
        into a guessed Python event.  Callers must use
        :meth:`has_pending_external_dispatches` to distinguish that state
        from a fully idle pool.
        """

        while self._launch_heap:
            launch_ns, group_id = self._launch_heap[0]
            if self._worker(group_id).pending_launch_ns == launch_ns:
                break
            heapq.heappop(self._launch_heap)
        values = []
        if self._completion_heap:
            values.append(self._completion_heap[0][0])
        if self._launch_heap:
            values.append(self._launch_heap[0][0])
        return min(values) if values else None

    def has_pending_external_dispatches(self) -> bool:
        return bool(self._external_pending)

    def has_pending(self) -> bool:
        """Return whether requests or backend completions remain live."""

        return bool(
            self._completion_heap
            or self._launch_heap
            or self._external_pending
            or any(
                worker.waiting
                or worker.prefill_drain
                or worker.active_decode
                or worker.inflight is not None
                or worker.pending_launch_ns is not None
                for worker in self.workers
            )
        )

    def pop_completed(self) -> list[HBFServingRequest]:
        result = []
        while self._completed_request_ids:
            result.append(self.requests[
                self._completed_request_ids.popleft()])
        return result

    def run_until_idle(self) -> list[HBFServingRequest]:
        completed = []
        while self.next_event_ns() is not None:
            self.advance(self.next_event_ns())
            completed.extend(self.pop_completed())
        if self._external_pending:
            raise RuntimeError(
                "external ASTRA completions are pending; drain dispatches "
                "and apply complete_external_dispatch callbacks")
        if any(
                worker.waiting
                or worker.prefill_drain
                or worker.active_decode
                or worker.inflight
                or worker.pending_launch_ns is not None
                for worker in self.workers):
            raise RuntimeError(
                "HBF pool is idle with unschedulable requests; "
                "check LPDDR capacity")
        self.assert_invariants()
        return completed

    def assert_invariants(self) -> None:
        self.lpddr_ledger.assert_invariants()
        try:
            validate_hbf_astra_timing_metrics(self.metrics)
        except HBFModelAstraProjectionError as exc:
            raise AssertionError(str(exc)) from exc
        if self.execution_backend == "analytical_calendar":
            if (
                self._external_outbox
                or self._external_pending
                or self._external_issued_job_ids
                or self._external_completed_job_ids
            ):
                raise AssertionError(
                    "analytical pool contains external ASTRA state")
        else:
            if self._completion_heap or self._launch_heap:
                raise AssertionError(
                    "external ASTRA pool contains Python timing events")
            outbox_ids = [row.job_id for row in self._external_outbox]
            if len(outbox_ids) != len(set(outbox_ids)):
                raise AssertionError(
                    "external ASTRA outbox contains duplicate jobs")
            pending_ids = set(self._external_pending)
            if not set(outbox_ids) <= pending_ids:
                raise AssertionError(
                    "external ASTRA outbox contains a non-pending job")
            if set(outbox_ids) & self._external_issued_job_ids:
                raise AssertionError(
                    "external ASTRA job is both queued and issued")
            if pending_ids != (
                    set(outbox_ids) | self._external_issued_job_ids):
                raise AssertionError(
                    "external ASTRA pending job has no ownership state")
            if (
                pending_ids & self._external_completed_job_ids
                or self._external_issued_job_ids
                & self._external_completed_job_ids
            ):
                raise AssertionError(
                    "completed external ASTRA job remains live")
            for dispatch in self._external_outbox:
                if self._external_pending.get(
                        dispatch.job_id) is not dispatch:
                    raise AssertionError(
                        "external ASTRA outbox identity mismatch")
            inflight_by_job = {}
            for job_id, dispatch in self._external_pending.items():
                worker = self._worker(dispatch.batch.group_id)
                if worker.inflight is not dispatch.batch:
                    raise AssertionError(
                        "external ASTRA pending batch identity mismatch")
                if dispatch.batch.completion_ns is not None:
                    raise AssertionError(
                        "pending external ASTRA batch already completed")
                inflight_by_job[job_id] = dispatch.batch.batch_id
            if len(inflight_by_job) != sum(
                    worker.inflight is not None
                    for worker in self.workers):
                raise AssertionError(
                    "external ASTRA inflight/pending count mismatch")
        queue_members: Counter[int] = Counter()
        waiting_members: Counter[int] = Counter()
        drain_members: Counter[int] = Counter()
        decode_members: Counter[int] = Counter()
        inflight_members: Counter[int] = Counter()
        for worker in self.workers:
            if (
                worker.inflight is not None
                and worker.pending_launch_ns is not None
            ):
                raise AssertionError(
                    "worker cannot be inflight and pending launch")
            waiting_members.update(worker.waiting)
            drain_members.update(worker.prefill_drain)
            decode_members.update(worker.active_decode)
            queue_members.update(worker.waiting)
            queue_members.update(worker.prefill_drain)
            queue_members.update(worker.active_decode)
            if worker.inflight is not None:
                inflight_ids = [
                    item.request_id for item in worker.inflight.items
                ]
                inflight_members.update(inflight_ids)
                queue_members.update(inflight_ids)
        for request in self.requests.values():
            if not 0 <= request.prefill_processed_tokens <= (
                    request.fresh_tokens):
                raise AssertionError(
                    f"invalid prefill progress: {request}")
            if not 0 <= request.generated_tokens <= request.output_tokens:
                raise AssertionError(
                    f"invalid decode progress: {request}")
            materialized_growth = (
                request.prefill_processed_tokens
                + max(0, request.generated_tokens - 1)
            )
            if not 0 <= request.published_growth_tokens <= (
                    materialized_growth):
                raise AssertionError(
                    f"invalid published growth: {request}")
            if request.active_lpddr_tokens < 0:
                raise AssertionError(
                    f"negative live LPDDR placement: {request}")
            expected_timestamps = (
                request.generated_tokens
                if self.retain_token_completion_history else 0
            )
            if len(request.token_completion_ns) != expected_timestamps:
                raise AssertionError(
                    f"token timestamp mismatch: {request}")
            if request.state == HBFRequestState.COMPLETE:
                if (
                    request.completion_ns is None
                    or request.generated_tokens != request.output_tokens
                ):
                    raise AssertionError(
                        f"incomplete completed request: {request}")
                if queue_members[request.request_id]:
                    raise AssertionError(
                        f"completed request remains queued: {request}")
            elif queue_members[request.request_id] != 1:
                raise AssertionError(
                    f"live request queue multiplicity is not one: {request}")
            queue_class = (
                waiting_members[request.request_id],
                drain_members[request.request_id],
                decode_members[request.request_id],
                inflight_members[request.request_id],
            )
            if (
                request.state == HBFRequestState.PREFILL
                and (
                    queue_class[1]
                    or queue_class[2]
                    or queue_class[0] + queue_class[3] != 1
                )
            ):
                raise AssertionError(
                    f"prefill request has invalid queue ownership: "
                    f"{request}, classes={queue_class}")
            if (
                request.state == HBFRequestState.PREFILL_DRAIN
                and queue_class != (0, 1, 0, 0)
            ):
                raise AssertionError(
                    f"prefill-drain request has invalid queue ownership: "
                    f"{request}, classes={queue_class}")
            if (
                request.state == HBFRequestState.DECODE
                and (
                    queue_class[0]
                    or queue_class[1]
                    or queue_class[2] + queue_class[3] != 1
                )
            ):
                raise AssertionError(
                    f"decode request has invalid queue ownership: "
                    f"{request}, classes={queue_class}")
            if request.state == HBFRequestState.PREFILL_DRAIN:
                if (
                    request.prefill_drain_ready_ns is None
                    or request.generated_tokens != 1
                ):
                    raise AssertionError(
                        f"invalid prefill drain gate: {request}")
            elif (
                request.prefill_drain_claimed
                or request.prefill_drain_job_id is not None
                or request.prefill_drain_ready_ns is not None
            ):
                raise AssertionError(
                    f"non-draining request retains drain state: {request}")
            if (
                self._current_request_by_session.get(request.session_id)
                == request.request_id
            ):
                owner = self._lpddr_owner(request.session_id)
                actual = self.lpddr_ledger.owner_card_bytes(owner)
                expected = self._request_lpddr_card_bytes(request)
                headroom_actual = (
                    self.lpddr_ledger.owner_card_bytes(
                    hbf_request_headroom_owner(
                        request.request_id))
                )
                headroom_expected = (
                    self._remaining_headroom_card_bytes(request))
                if (
                    request.state != HBFRequestState.COMPLETE
                    and not self._vectors_equal(
                        request.group_id, actual, expected)
                ):
                    raise AssertionError(
                        f"live request LPDDR mismatch: {request}, "
                        f"actual={actual}, expected={expected}")
                if not self._vectors_equal(
                        request.group_id,
                        headroom_actual,
                        headroom_expected):
                    raise AssertionError(
                        f"HBF request LPDDR headroom mismatch: {request}, "
                        f"actual={headroom_actual}, "
                        f"expected={headroom_expected}")
        for group_id in range(len(self.workers)):
            mixed_cap = self._worker(
                group_id).mixed_prefill_chunk_cap
            if self.mixed_batch_latency_limit_ns is None:
                if mixed_cap is not None:
                    raise AssertionError(
                        "unguarded worker retains a mixed-prefill cap")
            elif (
                mixed_cap is None
                or not 0 <= mixed_cap
                <= self.max_prefill_chunk_tokens
            ):
                raise AssertionError(
                    "guarded worker has an invalid mixed-prefill cap")
            used_bytes = dict(
                self.lpddr_ledger.used_bytes_by_card(group_id))
            if any(
                    byte_count > self.lpddr_ledger.capacity_bytes
                    for byte_count in used_bytes.values()):
                raise AssertionError(
                    f"LPDDR KV capacity exceeded on group {group_id}: "
                    f"active={used_bytes}, capacity="
                    f"{self.lpddr_ledger.capacity_bytes}")

    def report(self) -> dict[str, Any]:
        result = {
            "layout": asdict(self.layout),
            "hardware": asdict(self.hardware),
            "execution_backend": self.execution_backend,
            "server_id": self.server_id,
            "completion_time_source": (
                "external_astra_callback"
                if self.execution_backend == "external_astra"
                else "python_analytical_calendar"
            ),
            "astra_projection_schema": (
                ORDERED_V2_SCHEMA
                if self.execution_backend == "external_astra"
                else None
            ),
            "astra_projection_fidelity": (
                ORDERED_V2_FIDELITY
                if self.execution_backend == "external_astra"
                else None
            ),
            "astra_timing_semantics": (
                dict(ASTRA_NAMED_RESOURCE_TIMING_SEMANTICS)
                if self.execution_backend == "external_astra"
                else None
            ),
            "scheduling_policy": self.scheduling_policy,
            "validate_every_event": self.validate_every_event,
            "retain_detailed_history": self.retain_detailed_history,
            "retain_token_completion_history": (
                self.retain_token_completion_history),
            "retained_batch_count": len(self.batch_history),
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "max_prefill_chunk_tokens": self.max_prefill_chunk_tokens,
            "mixed_batch_latency_limit_ns": (
                self.mixed_batch_latency_limit_ns),
            "mixed_prefill_chunk_cap_by_group": {
                worker.group_id: worker.mixed_prefill_chunk_cap
                for worker in self.workers
            },
            "prefill_drain_tail_tokens": (
                self.prefill_drain_tail_tokens),
            "prefill_drain_min_tokens": (
                self.prefill_drain_min_tokens),
            "workspace_bytes_per_card": self.workspace_bytes_per_card,
            "lpddr_kv_capacity_bytes_per_card": (
                self.lpddr_kv_capacity_bytes_per_card),
            "lpddr_ledger_capacity_bytes_per_card": (
                self.lpddr_ledger.capacity_bytes),
            "lpddr_used_bytes_per_group": {
                group_id: self.lpddr_ledger.used_bytes(group_id)
                for group_id in range(len(self.workers))
            },
            "lpddr_used_bytes_by_card": {
                group_id: dict(
                    self.lpddr_ledger.used_bytes_by_card(group_id))
                for group_id in range(len(self.workers))
            },
            "lpddr_peak_bytes_by_card": {
                group_id: dict(card_bytes)
                for group_id, card_bytes in
                self.lpddr_ledger.peak_used_bytes_by_card.items()
            },
            "lpddr_headroom_bytes_per_group": {
                group_id: sum(
                    byte_count
                    for owner, byte_count in
                    self.lpddr_ledger.reservations(group_id).items()
                    if owner.startswith("hbf-request-headroom:")
                )
                for group_id in range(len(self.workers))
            },
            "lpddr_headroom_bytes_by_card": {
                group_id: {
                    card_id: sum(
                        card_bytes[card_id]
                        for owner, card_bytes in
                        self.lpddr_ledger.card_reservations(
                            group_id).items()
                        if owner.startswith(
                            "hbf-request-headroom:")
                    )
                    for card_id in self._cards(group_id)
                }
                for group_id in range(len(self.workers))
            },
            "kv_bytes_per_active_token_per_card": (
                self.kv_bytes_per_active_token_per_card),
            "kv_peak_bytes_for_one_token_per_card": max(
                self._range_card_bytes(
                    0, token_start=0, token_count=1).values(),
                default=0,
            ),
            "metrics": asdict(self.metrics),
            "group_telemetry": {
                worker.group_id: {
                    "submitted_requests": sum(
                        request.group_id == worker.group_id
                        for request in self.requests.values()
                    ),
                    "completed_requests": sum(
                        request.group_id == worker.group_id
                        and request.state == HBFRequestState.COMPLETE
                        for request in self.requests.values()
                    ),
                    "completed_batches": worker.completed_batches,
                    "waiting_requests": len(worker.waiting),
                    "active_decode_requests": len(
                        worker.active_decode),
                    "inflight": worker.inflight is not None,
                    "npu_busy_ns": (
                        self.calendar.busy_ns.get(
                            self._analytical_local_resource(
                                f"hbf-group-{worker.group_id}-npu"),
                            0,
                        )
                        if self.execution_backend
                        == "analytical_calendar" else None
                    ),
                    "npu_utilization": (
                        self.calendar.utilization(
                            self._analytical_local_resource(
                                f"hbf-group-{worker.group_id}-npu"),
                            self.current_ns,
                        )
                        if self.execution_backend
                        == "analytical_calendar" else None
                    ),
                    "compute_device": "h100_class_gpu",
                    "gpu_busy_ns": (
                        self.calendar.busy_ns.get(
                            self._analytical_local_resource(
                                f"hbf-group-{worker.group_id}-npu"),
                            0,
                        )
                        if self.execution_backend
                        == "analytical_calendar" else None
                    ),
                    "gpu_utilization": (
                        self.calendar.utilization(
                            self._analytical_local_resource(
                                f"hbf-group-{worker.group_id}-npu"),
                            self.current_ns,
                        )
                        if self.execution_backend
                        == "analytical_calendar" else None
                    ),
                    "fabric_busy_ns": (
                        self.calendar.busy_ns.get(
                            self._analytical_local_resource(
                                f"hbf-group-{worker.group_id}-fabric"),
                            0,
                        )
                        if self.execution_backend
                        == "analytical_calendar" else None
                    ),
                    "fabric_utilization": (
                        self.calendar.utilization(
                            self._analytical_local_resource(
                                f"hbf-group-{worker.group_id}-fabric"),
                            self.current_ns,
                        )
                        if self.execution_backend
                        == "analytical_calendar" else None
                    ),
                }
                for worker in self.workers
            },
            "pending_batch_count": (
                len(self._completion_heap)
                if self.execution_backend == "analytical_calendar"
                else len(self._external_pending)
            ),
            "pending_launch_count": sum(
                worker.pending_launch_ns is not None
                for worker in self.workers),
            "external_pending_job_ids": sorted(
                self._external_pending),
            "external_undrained_dispatch_count": len(
                self._external_outbox),
            "external_issued_dispatch_count": len(
                self._external_issued_job_ids),
            "requests": {
                request_id: {
                    **asdict(request),
                    "state": request.state.value,
                    "ttft_ns": request.ttft_ns,
                    "tpot_ns": request.tpot_ns,
                }
                for request_id, request in sorted(self.requests.items())
            },
        }
        if self.analytical_resource_prefix:
            result["analytical_resource_prefix"] = (
                self.analytical_resource_prefix)
        return result

__all__ = [
    "BatchItem",
    "FullModelHBFServingPool",
    "HBFExternalDispatch",
    "HBFPoolMetrics",
    "HBFRequestState",
    "HBFServingBatch",
    "HBFServingRequest",
    "HBFWorker",
    "derive_lpddr_workspace_bytes",
]

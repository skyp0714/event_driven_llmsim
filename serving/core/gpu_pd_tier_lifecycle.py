"""Versioned, node-local KV tier lifecycle for finite P4D4 baselines.

This module deliberately owns all finite P/D and lower-tier capacity state
plus analytical transfer queues, but not model execution.  A serving-node
controller must use these ledgers as its sole P/D admission authority rather
than duplicate the same claims in a second HBM manager.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import heapq
from typing import Any, Mapping, Optional

from .gpu_pd_latency import P4D4GPUHardware
from .gpu_pd_tier_resources import TierNodeResources, TierTransferStage
from .hbf_full_model_lifecycle import ResourceCalendar


SUPPORTED_TIER_POLICIES = frozenset({
    "hbm_lru_recompute",
    "ssd_direct",
    "cpu_ssd",
})
MAX_CONTEXT_TOKENS = 1_010_000


class Tier(str, Enum):
    D = "d"
    CPU = "cpu"
    SSD = "ssd"


class TierSessionState(str, Enum):
    LOST = "lost"
    D_READY = "d_ready"
    D_DEMOTING_CPU = "d_demoting_cpu"
    D_DEMOTING_SSD = "d_demoting_ssd"
    CPU_READY = "cpu_ready"
    CPU_DEMOTING_SSD = "cpu_demoting_ssd"
    SSD_READY = "ssd_ready"
    PREPARING = "preparing"
    ACTIVE = "active"
    ENDED = "ended"


class TierJobKind(str, Enum):
    DEMOTION = "demotion"
    PREPARE = "prepare"


class TierJobStatus(str, Enum):
    RUNNING = "running"
    COMMITTED = "committed"
    STALE = "stale"
    COMPLETE = "complete"


@dataclass
class SharedByteLedger:
    """Exact byte ownership shared by persistent and transient allocations."""

    name: str
    capacity_bytes: int
    owners: dict[str, int] = field(default_factory=dict)
    peak_used_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ledger name must be non-empty")
        if (
            isinstance(self.capacity_bytes, bool)
            or not isinstance(self.capacity_bytes, int)
            or self.capacity_bytes <= 0
        ):
            raise ValueError("ledger capacity_bytes must be positive")

    @property
    def used_bytes(self) -> int:
        return sum(self.owners.values())

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    def owner_bytes(self, owner: str) -> int:
        return self.owners.get(owner, 0)

    def can_set(self, owner: str, byte_count: int) -> bool:
        self._validate_owner_bytes(owner, byte_count)
        return (
            self.used_bytes - self.owner_bytes(owner) + byte_count
            <= self.capacity_bytes
        )

    def set_bytes(self, owner: str, byte_count: int) -> None:
        if not self.can_set(owner, byte_count):
            raise RuntimeError(
                f"{self.name} capacity exceeded: owner={owner!r}, "
                f"requested={byte_count}, used={self.used_bytes}, "
                f"capacity={self.capacity_bytes}")
        if byte_count:
            self.owners[owner] = byte_count
        else:
            self.owners.pop(owner, None)
        self.peak_used_bytes = max(
            self.peak_used_bytes, self.used_bytes)
        self.assert_invariants()

    def reserve(self, owner: str, byte_count: int) -> bool:
        if owner in self.owners:
            raise RuntimeError(
                f"{self.name} owner already exists: {owner!r}")
        if not self.can_set(owner, byte_count):
            return False
        self.set_bytes(owner, byte_count)
        return True

    def release(self, owner: str) -> int:
        byte_count = self.owners.pop(owner, 0)
        self.assert_invariants()
        return byte_count

    def replace(
            self, *, remove_owners: tuple[str, ...],
            owner: Optional[str], byte_count: int) -> None:
        """Atomically replace several claims with one final claim."""

        if byte_count < 0:
            raise ValueError("replacement byte_count must be non-negative")
        if owner is not None and not owner:
            raise ValueError("replacement owner must be non-empty")
        missing = [
            old for old in remove_owners if old not in self.owners
        ]
        if missing:
            raise RuntimeError(
                f"{self.name} replacement owners missing: {missing}")
        remove_set = set(remove_owners)
        if (
            owner is not None
            and owner in self.owners
            and owner not in remove_set
        ):
            raise RuntimeError(
                f"{self.name} replacement owner already exists")
        projected = (
            self.used_bytes
            - sum(self.owners[old] for old in remove_set)
            + byte_count
        )
        if projected > self.capacity_bytes:
            raise RuntimeError(
                f"{self.name} atomic replacement exceeds capacity")
        for old in remove_set:
            self.owners.pop(old)
        if owner is not None and byte_count:
            self.owners[owner] = byte_count
        self.peak_used_bytes = max(
            self.peak_used_bytes, self.used_bytes)
        self.assert_invariants()

    @staticmethod
    def _validate_owner_bytes(owner: str, byte_count: int) -> None:
        if not owner:
            raise ValueError("ledger owner must be non-empty")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ValueError("ledger byte_count must be non-negative")

    def assert_invariants(self) -> None:
        if any(
                not owner or byte_count <= 0
                for owner, byte_count in self.owners.items()):
            raise AssertionError(
                f"{self.name} contains invalid ownership")
        if not 0 <= self.used_bytes <= self.capacity_bytes:
            raise AssertionError(
                f"{self.name} capacity invariant failed")

    def report(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self.free_bytes,
            "peak_used_bytes": self.peak_used_bytes,
            "owners": dict(sorted(self.owners.items())),
        }


@dataclass
class TierCopy:
    copy_id: int
    session_id: str
    tier: Tier
    version: int
    generation: int
    tokens: int
    byte_count: int
    ledger_owner: str
    demotion_pins: int = 0
    foreground_pins: int = 0
    retired: bool = False
    shadow: bool = False

    @property
    def pins(self) -> int:
        return self.demotion_pins + self.foreground_pins


@dataclass
class TierSession:
    session_id: str
    state: TierSessionState = TierSessionState.LOST
    generation: int = 0
    version: int = 0
    tokens: int = 0
    primary: Optional[Tier] = None
    primary_copy_id: Optional[int] = None
    last_access_ns: int = 0
    active_request_id: Optional[int] = None
    pending_job_ids: set[int] = field(default_factory=set)
    copy_ids: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class ScheduledTierStage:
    stage: TierTransferStage
    start_ns: int
    completion_ns: int


@dataclass
class TierTransferJob:
    job_id: int
    kind: TierJobKind
    session_id: str
    request_id: Optional[int]
    snapshot_version: int
    snapshot_generation: int
    source_copy_id: Optional[int]
    destination: Optional[Tier]
    destination_owner: Optional[str]
    bounce_owner: Optional[str]
    stages: tuple[ScheduledTierStage, ...]
    start_ns: int
    completion_ns: int
    status: TierJobStatus = TierJobStatus.RUNNING

    @property
    def transfer_kinds(self) -> tuple[str, ...]:
        return tuple(stage.stage.kind for stage in self.stages)


@dataclass
class PrepareTicket:
    prepare_id: int
    job_id: int
    session_id: str
    request_id: int
    generation: int
    source: Optional[Tier]
    source_copy_id: Optional[int]
    source_tokens: int
    hit_tokens: int
    input_tokens: int
    output_tokens: int
    final_tokens: int
    has_successor: bool
    needs_d: bool
    p_owner: str
    p_bytes_per_rank: int
    d_owner: Optional[str]
    d_reserved_bytes_per_rank: int
    d_target_bytes_per_rank: int
    d_reuse_copy_id: Optional[int]
    full_d_reservation: bool
    bounce_owner: Optional[str]
    stages: tuple[ScheduledTierStage, ...]
    start_ns: int
    completion_ns: int
    completed: bool = False
    active: bool = False
    p_released: bool = False
    committed: bool = False

    @property
    def transfer_kinds(self) -> tuple[str, ...]:
        return tuple(stage.stage.kind for stage in self.stages)


@dataclass(frozen=True)
class ResumeSource:
    session_id: str
    state: TierSessionState
    source: Optional[Tier]
    tokens: int
    version: int
    generation: int
    demotion_inflight: bool


@dataclass
class TierLifecycleMetrics:
    d_drops: int = 0
    d_to_cpu_started: int = 0
    d_to_ssd_started: int = 0
    cpu_to_ssd_started: int = 0
    demotions_committed: int = 0
    stale_demotions: int = 0
    prepare_started: int = 0
    prepare_completed: int = 0
    p_handoff_releases: int = 0
    prepare_misses: int = 0
    d_prepare_hits: int = 0
    cpu_prepare_hits: int = 0
    ssd_prepare_hits: int = 0
    destination_deferrals: int = 0
    p_capacity_deferrals: int = 0
    d_capacity_deferrals: int = 0
    cpu_bounce_deferrals: int = 0
    infeasible_p_requests: int = 0
    infeasible_d_requests: int = 0
    infeasible_cpu_bounces: int = 0
    cpu_capacity_deferrals: int = 0
    ssd_evictions: int = 0
    retired_copy_releases: int = 0
    transfer_bytes: int = 0
    stale_transfer_bytes: int = 0
    d_source_bytes_released_early: int = 0


class TieredPDKVLifecycle:
    """Finite, versioned D/CPU/SSD lifecycle for one physical node."""

    def __init__(
            self, *, hardware: P4D4GPUHardware, node_id: int,
            policy: str,
            calendar: Optional[ResourceCalendar] = None,
            p_capacity_bytes_per_rank: Optional[int] = None,
            d_capacity_bytes_per_rank: Optional[int] = None,
            cpu_capacity_bytes: Optional[int] = None,
            ssd_capacity_bytes: Optional[int] = None,
            validate_every_event: bool = True) -> None:
        hardware.validate()
        if policy not in SUPPORTED_TIER_POLICIES:
            raise ValueError(
                f"unsupported tier policy {policy!r}")
        if (
            isinstance(node_id, bool)
            or not isinstance(node_id, int)
            or node_id < 0
        ):
            raise ValueError("node_id must be a non-negative integer")
        if not isinstance(validate_every_event, bool):
            raise ValueError(
                "validate_every_event must be a boolean")
        self.hardware = hardware
        self.node_id = node_id
        self.policy = policy
        self.validate_every_event = validate_every_event
        self.calendar = (
            calendar if calendar is not None else ResourceCalendar())
        self.resources = TierNodeResources(
            hardware=hardware, node_id=node_id)
        p_capacity = (
            hardware.usable_hbm_bytes_per_rank
            if p_capacity_bytes_per_rank is None
            else p_capacity_bytes_per_rank
        )
        d_capacity = (
            hardware.usable_hbm_bytes_per_rank
            if d_capacity_bytes_per_rank is None
            else d_capacity_bytes_per_rank
        )
        cpu_capacity = (
            hardware.cpu_memory_capacity_bytes
            if cpu_capacity_bytes is None else cpu_capacity_bytes
        )
        ssd_capacity = (
            hardware.ssd_capacity_bytes
            if ssd_capacity_bytes is None else ssd_capacity_bytes
        )
        self.p_ledger = SharedByteLedger(
            f"gpu-node-{node_id}-p-destination", p_capacity)
        self.d_ledger = SharedByteLedger(
            f"gpu-node-{node_id}-d-kv", d_capacity)
        self.cpu_ledger = SharedByteLedger(
            f"gpu-node-{node_id}-cpu-shared", cpu_capacity)
        self.ssd_ledger = SharedByteLedger(
            f"gpu-node-{node_id}-ssd", ssd_capacity)
        self.sessions: dict[str, TierSession] = {}
        self.copies: dict[int, TierCopy] = {}
        self.jobs: dict[int, TierTransferJob] = {}
        self.prepares: dict[int, PrepareTicket] = {}
        self.metrics = TierLifecycleMetrics()
        self.current_ns = 0
        self._next_copy_id = 1
        self._next_job_id = 1
        self._next_prepare_id = 1
        self._completion_heap: list[tuple[int, int]] = []
        self._completed_prepares: list[PrepareTicket] = []
        self._seen_request_ids: set[int] = set()

    @staticmethod
    def _validate_session(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")

    @staticmethod
    def _validate_time(now_ns: int) -> None:
        if (
            isinstance(now_ns, bool)
            or not isinstance(now_ns, int)
            or now_ns < 0
        ):
            raise ValueError("now_ns must be a non-negative integer")

    @staticmethod
    def _validate_positive(name: str, value: int) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")

    def _per_rank_bytes(self, tokens: int) -> int:
        return self.hardware.kv_capacity_bytes_per_rank(tokens)

    def _aggregate_bytes(self, tokens: int) -> int:
        return self._per_rank_bytes(tokens) * self.hardware.tp_size

    def _maybe_assert_invariants(self) -> None:
        if self.validate_every_event:
            self.assert_invariants()

    def _ledger(self, tier: Tier) -> SharedByteLedger:
        if tier == Tier.D:
            return self.d_ledger
        if tier == Tier.CPU:
            return self.cpu_ledger
        return self.ssd_ledger

    def _copy_owner(self, tier: Tier, copy_id: int) -> str:
        return (
            f"node-{self.node_id}:{tier.value}:copy:{copy_id}")

    def _new_copy(
            self, *, record: TierSession, tier: Tier,
            version: int, generation: int, tokens: int,
            existing_owner: Optional[str] = None,
            shadow: bool = False) -> TierCopy:
        copy_id = self._next_copy_id
        self._next_copy_id += 1
        byte_count = (
            self._per_rank_bytes(tokens)
            if tier == Tier.D else self._aggregate_bytes(tokens)
        )
        owner = self._copy_owner(tier, copy_id)
        ledger = self._ledger(tier)
        if existing_owner is None:
            if not ledger.reserve(owner, byte_count):
                raise RuntimeError(
                    f"{tier.value} capacity changed before copy commit")
        else:
            ledger.replace(
                remove_owners=(existing_owner,),
                owner=owner,
                byte_count=byte_count,
            )
        copy = TierCopy(
            copy_id=copy_id,
            session_id=record.session_id,
            tier=tier,
            version=version,
            generation=generation,
            tokens=tokens,
            byte_count=byte_count,
            ledger_owner=owner,
            shadow=shadow,
        )
        self.copies[copy_id] = copy
        record.copy_ids.add(copy_id)
        return copy

    def _release_copy(self, copy: TierCopy) -> None:
        if copy.pins:
            copy.retired = True
            return
        record = self.sessions[copy.session_id]
        self._ledger(copy.tier).release(copy.ledger_owner)
        record.copy_ids.discard(copy.copy_id)
        if record.primary_copy_id == copy.copy_id:
            record.primary_copy_id = None
            record.primary = None
        self.copies.pop(copy.copy_id, None)
        self.metrics.retired_copy_releases += int(copy.retired)

    def _maybe_release_retired(self, copy: TierCopy) -> None:
        if copy.retired and copy.pins == 0:
            self._release_copy(copy)

    def _set_primary(
            self, record: TierSession, copy: TierCopy,
            state: TierSessionState) -> None:
        record.primary = copy.tier
        record.primary_copy_id = copy.copy_id
        record.tokens = copy.tokens
        record.state = state

    def register_d_ready(
            self, session_id: str, tokens: int, *,
            now_ns: int, version: Optional[int] = None) -> TierSession:
        """Register one committed idle D copy.

        The caller must obtain D headroom first.  The lifecycle remains the
        sole capacity authority when a node later submits model execution.
        """

        self._validate_session(session_id)
        self._validate_positive("tokens", tokens)
        self._validate_time(now_ns)
        if tokens > MAX_CONTEXT_TOKENS:
            raise ValueError(
                f"context exceeds {MAX_CONTEXT_TOKENS} tokens")
        if version is not None:
            self._validate_positive("version", version)
        self.advance(now_ns)
        existing = self.sessions.get(session_id)
        if existing is not None:
            raise RuntimeError(
                f"session {session_id!r} is already registered")
        byte_count = self._per_rank_bytes(tokens)
        if self.d_ledger.free_bytes < byte_count:
            raise RuntimeError(
                "D HBM headroom is not available; call "
                "ensure_d_headroom and retry")
        record = TierSession(
            session_id=session_id,
            state=TierSessionState.D_READY,
            generation=0,
            version=1 if version is None else version,
            tokens=tokens,
            primary=Tier.D,
            last_access_ns=now_ns,
        )
        self.sessions[session_id] = record
        copy = self._new_copy(
            record=record,
            tier=Tier.D,
            version=record.version,
            generation=record.generation,
            tokens=tokens,
        )
        record.primary_copy_id = copy.copy_id
        self._maybe_assert_invariants()
        return record

    def next_event_ns(self) -> Optional[int]:
        return (
            None if not self._completion_heap
            else self._completion_heap[0][0]
        )

    def advance(self, now_ns: int) -> None:
        """Advance time, committing completions before same-time arrivals."""

        self._validate_time(now_ns)
        if now_ns < self.current_ns:
            raise ValueError("cannot move lifecycle time backwards")
        while (
            self._completion_heap
            and self._completion_heap[0][0] <= now_ns
        ):
            completion_ns, job_id = heapq.heappop(
                self._completion_heap)
            self.current_ns = completion_ns
            job = self.jobs[job_id]
            if job.status != TierJobStatus.RUNNING:
                raise AssertionError("completed job remained on heap")
            if job.kind == TierJobKind.DEMOTION:
                self._complete_demotion(job)
            else:
                self._complete_prepare(job)
        self.current_ns = now_ns
        self._maybe_assert_invariants()

    def _schedule(
            self, *, kind: TierJobKind, record: TierSession,
            request_id: Optional[int], source_copy: Optional[TierCopy],
            destination: Optional[Tier],
            destination_owner: Optional[str],
            bounce_owner: Optional[str],
            stages: tuple[TierTransferStage, ...],
            ready_ns: int) -> TierTransferJob:
        job_id = self._next_job_id
        self._next_job_id += 1
        scheduled: list[ScheduledTierStage] = []
        stage_ready = ready_ns
        for stage in stages:
            start_ns, completion_ns = stage.reserve(
                self.calendar,
                ready_ns=stage_ready,
                job_id=job_id,
                namespace=f"gpu-tier-node-{self.node_id}",
            )
            scheduled.append(ScheduledTierStage(
                stage=stage,
                start_ns=start_ns,
                completion_ns=completion_ns,
            ))
            stage_ready = completion_ns
        start_ns = (
            ready_ns if not scheduled else scheduled[0].start_ns)
        completion_ns = (
            ready_ns if not scheduled else scheduled[-1].completion_ns)
        job = TierTransferJob(
            job_id=job_id,
            kind=kind,
            session_id=record.session_id,
            request_id=request_id,
            snapshot_version=record.version,
            snapshot_generation=record.generation,
            source_copy_id=(
                None if source_copy is None else source_copy.copy_id),
            destination=destination,
            destination_owner=destination_owner,
            bounce_owner=bounce_owner,
            stages=tuple(scheduled),
            start_ns=start_ns,
            completion_ns=completion_ns,
        )
        self.jobs[job_id] = job
        record.pending_job_ids.add(job_id)
        heapq.heappush(
            self._completion_heap, (completion_ns, job_id))
        self.metrics.transfer_bytes += sum(
            stage.stage.aggregate_bytes for stage in scheduled)
        return job

    def _primary_copy(self, record: TierSession) -> Optional[TierCopy]:
        if record.primary_copy_id is None:
            return None
        return self.copies[record.primary_copy_id]

    def _eligible_lru(
            self, tier: Tier, *,
            exclude_sessions: tuple[str, ...] = ()) -> list[TierSession]:
        ready_state = {
            Tier.D: TierSessionState.D_READY,
            Tier.CPU: TierSessionState.CPU_READY,
            Tier.SSD: TierSessionState.SSD_READY,
        }[tier]
        return sorted(
            (
                record for record in self.sessions.values()
                if record.session_id not in exclude_sessions
                and record.state == ready_state
                and record.primary == tier
                and self._primary_copy(record) is not None
                and self._primary_copy(record).pins == 0
            ),
            key=lambda record: (
                record.last_access_ns, record.session_id),
        )

    def _ensure_ssd_capacity(
            self, byte_count: int, *, now_ns: int,
            exclude_sessions: tuple[str, ...] = ()) -> bool:
        if byte_count > self.ssd_ledger.capacity_bytes:
            raise RuntimeError(
                "tier object exceeds node SSD capacity")
        if self.ssd_ledger.free_bytes >= byte_count:
            return True
        victims = []
        reclaimable = self.ssd_ledger.free_bytes
        for victim in self._eligible_lru(
                Tier.SSD, exclude_sessions=exclude_sessions):
            copy = self._primary_copy(victim)
            assert copy is not None
            victims.append((victim, copy))
            reclaimable += copy.byte_count
            if reclaimable >= byte_count:
                break
        if reclaimable < byte_count:
            return False
        for victim, copy in victims:
            copy.retired = True
            self._release_copy(copy)
            victim.state = TierSessionState.LOST
            victim.tokens = 0
            victim.generation += 1
            victim.last_access_ns = now_ns
            self.metrics.ssd_evictions += 1
        return True

    def _start_cpu_spill(
            self, record: TierSession, *,
            now_ns: int,
            protected_ssd_session: Optional[str] = None,
    ) -> Optional[TierTransferJob]:
        source = self._primary_copy(record)
        if (
            source is None
            or record.state != TierSessionState.CPU_READY
            or source.tier != Tier.CPU
        ):
            raise RuntimeError("CPU spill requires an idle CPU source")
        if not self._ensure_ssd_capacity(
                source.byte_count,
                now_ns=now_ns,
                exclude_sessions=tuple(
                    session_id
                    for session_id in (
                        record.session_id,
                        protected_ssd_session,
                    )
                    if session_id is not None
                )):
            return None
        job_id = self._next_job_id
        destination_owner = (
            f"node-{self.node_id}:ssd:dest:{job_id}")
        if not self.ssd_ledger.reserve(
                destination_owner, source.byte_count):
            return None
        source.demotion_pins += 1
        stage = self.resources.ssd_stage(
            source.tokens, direction="cpu_to_ssd")
        job = self._schedule(
            kind=TierJobKind.DEMOTION,
            record=record,
            request_id=None,
            source_copy=source,
            destination=Tier.SSD,
            destination_owner=destination_owner,
            bounce_owner=None,
            stages=(stage,),
            ready_ns=now_ns,
        )
        if job.job_id != job_id:
            raise AssertionError("job id changed during reservation")
        record.state = TierSessionState.CPU_DEMOTING_SSD
        self.metrics.cpu_to_ssd_started += 1
        return job

    def demote(
            self, session_id: str, *,
            now_ns: int) -> Optional[TierTransferJob]:
        """Start the policy-specific demotion of one idle primary copy.

        ``None`` means lower-tier capacity work was started (or remains in
        flight) and the caller should retry at ``next_event_ns()``.
        Recompute eviction is immediate and also returns ``None``.
        """

        self._validate_session(session_id)
        self._validate_time(now_ns)
        self.advance(now_ns)
        record = self.sessions[session_id]
        source = self._primary_copy(record)
        if (
            source is None or source.tier != Tier.D
            or record.state != TierSessionState.D_READY
        ):
            raise RuntimeError("demotion requires an idle D-ready session")
        if self.policy == "hbm_lru_recompute":
            source.retired = True
            self._release_copy(source)
            record.state = TierSessionState.LOST
            record.tokens = 0
            record.generation += 1
            record.last_access_ns = now_ns
            self.metrics.d_drops += 1
            self._maybe_assert_invariants()
            return None

        aggregate = source.byte_count * self.hardware.tp_size
        if self.policy == "cpu_ssd":
            if aggregate > self.cpu_ledger.capacity_bytes:
                raise RuntimeError(
                    "tier object exceeds node CPU capacity")
            if self.cpu_ledger.free_bytes < aggregate:
                candidates = self._eligible_lru(
                    Tier.CPU, exclude_sessions=(session_id,))
                if not candidates:
                    self.metrics.cpu_capacity_deferrals += 1
                    return None
                self._start_cpu_spill(
                    candidates[0], now_ns=now_ns)
                self.metrics.cpu_capacity_deferrals += 1
                return None
            job_id = self._next_job_id
            destination_owner = (
                f"node-{self.node_id}:cpu:dest:{job_id}")
            if not self.cpu_ledger.reserve(
                    destination_owner, aggregate):
                raise AssertionError("CPU capacity precheck diverged")
            source.demotion_pins += 1
            stage = self.resources.gpu_cpu_stage(
                source.tokens,
                gpu_role="d",
                direction="gpu_to_cpu",
            )
            job = self._schedule(
                kind=TierJobKind.DEMOTION,
                record=record,
                request_id=None,
                source_copy=source,
                destination=Tier.CPU,
                destination_owner=destination_owner,
                bounce_owner=None,
                stages=(stage,),
                ready_ns=now_ns,
            )
            if job.job_id != job_id:
                raise AssertionError("job id changed during reservation")
            record.state = TierSessionState.D_DEMOTING_CPU
            self.metrics.d_to_cpu_started += 1
            return job

        if aggregate > self.cpu_ledger.capacity_bytes:
            raise RuntimeError(
                "SSD-direct bounce exceeds node CPU capacity")
        if self.cpu_ledger.free_bytes < aggregate:
            self.metrics.cpu_capacity_deferrals += 1
            return None
        if not self._ensure_ssd_capacity(
                aggregate,
                now_ns=now_ns,
                exclude_sessions=(session_id,)):
            return None
        job_id = self._next_job_id
        bounce_owner = (
            f"node-{self.node_id}:cpu:bounce:{job_id}")
        destination_owner = (
            f"node-{self.node_id}:ssd:dest:{job_id}")
        if not self.cpu_ledger.reserve(bounce_owner, aggregate):
            raise AssertionError("CPU bounce precheck diverged")
        if not self.ssd_ledger.reserve(
                destination_owner, aggregate):
            self.cpu_ledger.release(bounce_owner)
            raise AssertionError("SSD capacity precheck diverged")
        source.demotion_pins += 1
        d2c = self.resources.gpu_cpu_stage(
            source.tokens,
            gpu_role="d",
            direction="gpu_to_cpu",
        )
        c2s = self.resources.ssd_stage(
            source.tokens, direction="cpu_to_ssd")
        job = self._schedule(
            kind=TierJobKind.DEMOTION,
            record=record,
            request_id=None,
            source_copy=source,
            destination=Tier.SSD,
            destination_owner=destination_owner,
            bounce_owner=bounce_owner,
            stages=(d2c, c2s),
            ready_ns=now_ns,
        )
        if job.job_id != job_id:
            raise AssertionError("job id changed during reservation")
        record.state = TierSessionState.D_DEMOTING_SSD
        self.metrics.d_to_ssd_started += 1
        return job

    def ensure_d_headroom(
            self, required_bytes_per_rank: int, *,
            now_ns: int,
            exclude_session: Optional[str] = None) -> Optional[int]:
        """Start capacity-only whole-session LRU reclamation.

        Return ``now_ns`` when capacity is already available, otherwise the
        next useful completion time.  ``None`` means no safe victim or
        lower-tier destination is currently available.
        """

        self._validate_time(now_ns)
        if (
            isinstance(required_bytes_per_rank, bool)
            or not isinstance(required_bytes_per_rank, int)
            or required_bytes_per_rank < 0
        ):
            raise ValueError(
                "required_bytes_per_rank must be non-negative")
        self.advance(now_ns)
        if required_bytes_per_rank > self.d_ledger.capacity_bytes:
            raise RuntimeError(
                "requested D object exceeds node D-HBM capacity")
        if self.d_ledger.free_bytes >= required_bytes_per_rank:
            return now_ns
        for victim in self._eligible_lru(
                Tier.D,
                exclude_sessions=(
                    () if exclude_session is None
                    else (exclude_session,)
                )):
            before = self.next_event_ns()
            job = self.demote(victim.session_id, now_ns=now_ns)
            if self.d_ledger.free_bytes >= required_bytes_per_rank:
                return now_ns
            if job is not None:
                return job.completion_ns
            after = self.next_event_ns()
            if after is not None and after != before:
                return after
        return self.next_event_ns()

    def ensure_cpu_bounce_headroom(
            self, required_tokens: int, *, now_ns: int,
            protected_session: Optional[str] = None) -> Optional[int]:
        """Make room for an SSD-to-P restore bounce without private hooks.

        Return ``now_ns`` when the aggregate CPU claim already fits.  Under
        ``cpu_ssd``, one idle CPU-resident LRU object is spilled and its
        completion time is returned.  The SSD source being resumed can be
        protected from eviction while that spill reserves its destination.
        """

        self._validate_time(now_ns)
        if (
            isinstance(required_tokens, bool)
            or not isinstance(required_tokens, int)
            or required_tokens < 0
        ):
            raise ValueError(
                "required_tokens must be a non-negative integer")
        if protected_session is not None:
            self._validate_session(protected_session)
        required_bytes = self._aggregate_bytes(required_tokens)
        if required_bytes > self.cpu_ledger.capacity_bytes:
            raise RuntimeError(
                "requested CPU bounce exceeds node CPU capacity")
        self.advance(now_ns)
        if self.cpu_ledger.free_bytes >= required_bytes:
            return now_ns
        self.metrics.cpu_capacity_deferrals += 1
        if self.policy == "cpu_ssd":
            excluded = (
                () if protected_session is None
                else (protected_session,)
            )
            for victim in self._eligible_lru(
                    Tier.CPU, exclude_sessions=excluded):
                job = self._start_cpu_spill(
                    victim,
                    now_ns=now_ns,
                    protected_ssd_session=protected_session,
                )
                if job is not None:
                    return job.completion_ns
        return self.next_event_ns()

    def peek_resume_source(
            self, session_id: str, *,
            now_ns: int) -> ResumeSource:
        self._validate_session(session_id)
        self._validate_time(now_ns)
        self.advance(now_ns)
        record = self.sessions.get(session_id)
        if record is None:
            return ResumeSource(
                session_id=session_id,
                state=TierSessionState.LOST,
                source=None,
                tokens=0,
                version=0,
                generation=0,
                demotion_inflight=False,
            )
        source = self._primary_copy(record)
        demoting = record.state in {
            TierSessionState.D_DEMOTING_CPU,
            TierSessionState.D_DEMOTING_SSD,
            TierSessionState.CPU_DEMOTING_SSD,
        }
        return ResumeSource(
            session_id=session_id,
            state=record.state,
            source=None if source is None else source.tier,
            tokens=0 if source is None else source.tokens,
            version=record.version,
            generation=record.generation,
            demotion_inflight=demoting,
        )

    def _prepare_capacity(
            self, *, record: TierSession,
            source: Optional[TierCopy],
            input_tokens: int, final_tokens: int,
            needs_d: bool,
            needs_bounce_bytes: int,
            prepare_id: int) -> Optional[
                tuple[str, Optional[str], int, Optional[int], bool,
                      Optional[str]]]:
        p_bytes = self._per_rank_bytes(input_tokens)
        d_target = (
            self._per_rank_bytes(final_tokens) if needs_d else 0)
        p_owner = (
            f"node-{self.node_id}:p:prepare:{prepare_id}")
        d_owner = (
            f"node-{self.node_id}:d:prepare:{prepare_id}")
        bounce_owner = (
            f"node-{self.node_id}:cpu:prepare-bounce:{prepare_id}")

        reusable_d = (
            source is not None
            and source.tier == Tier.D
            and source.copy_id == record.primary_copy_id
        )
        full_d = bool(
            needs_d
            and (not reusable_d or source.demotion_pins > 0))
        d_reserved = (
            0 if not needs_d
            else d_target if full_d
            else max(0, d_target - source.byte_count)
        )
        if (
            self.p_ledger.free_bytes < p_bytes
            or self.d_ledger.free_bytes < d_reserved
            or self.cpu_ledger.free_bytes < needs_bounce_bytes
        ):
            self.metrics.destination_deferrals += 1
            self.metrics.p_capacity_deferrals += int(
                self.p_ledger.free_bytes < p_bytes)
            self.metrics.d_capacity_deferrals += int(
                self.d_ledger.free_bytes < d_reserved)
            self.metrics.cpu_bounce_deferrals += int(
                self.cpu_ledger.free_bytes < needs_bounce_bytes)
            return None
        if not self.p_ledger.reserve(p_owner, p_bytes):
            raise AssertionError("P destination precheck diverged")
        if d_reserved and not self.d_ledger.reserve(
                d_owner, d_reserved):
            self.p_ledger.release(p_owner)
            raise AssertionError("D destination precheck diverged")
        if needs_bounce_bytes and not self.cpu_ledger.reserve(
                bounce_owner, needs_bounce_bytes):
            self.p_ledger.release(p_owner)
            if d_reserved:
                self.d_ledger.release(d_owner)
            raise AssertionError("CPU bounce precheck diverged")
        return (
            p_owner,
            d_owner if d_reserved else None,
            d_reserved,
            source.copy_id if reusable_d else None,
            full_d,
            bounce_owner if needs_bounce_bytes else None,
        )

    def begin_prepare(
            self, session_id: str, *, request_id: int,
            now_ns: int, input_tokens: int, output_tokens: int,
            reusable_tokens: int,
            has_successor: bool) -> Optional[PrepareTicket]:
        """Atomically reserve P/D destinations and start a resume prepare.

        A ``None`` result is a pure capacity deferral: generation, pins,
        calendars, and session state are unchanged.
        """

        self._validate_session(session_id)
        self._validate_time(now_ns)
        if (
            isinstance(request_id, bool)
            or not isinstance(request_id, int)
            or request_id < 0
        ):
            raise ValueError(
                "request_id must be a non-negative integer")
        if request_id in self._seen_request_ids:
            raise ValueError(
                f"duplicate request_id={request_id}")
        for name, value in (
            ("input_tokens", input_tokens),
            ("output_tokens", output_tokens),
        ):
            self._validate_positive(name, value)
        if (
            isinstance(reusable_tokens, bool)
            or not isinstance(reusable_tokens, int)
            or reusable_tokens < 0
        ):
            raise ValueError(
                "reusable_tokens must be non-negative")
        if reusable_tokens > input_tokens:
            raise ValueError(
                "reusable_tokens cannot exceed input_tokens")
        if not isinstance(has_successor, bool):
            raise ValueError("has_successor must be a boolean")
        final_tokens = input_tokens + output_tokens - 1
        if final_tokens > MAX_CONTEXT_TOKENS:
            raise ValueError(
                f"final context exceeds {MAX_CONTEXT_TOKENS} tokens")
        needs_d = output_tokens > 1 or has_successor
        self.advance(now_ns)
        record = self.sessions.get(session_id)
        is_new_session = record is None
        if record is None:
            record = TierSession(
                session_id=session_id,
                state=TierSessionState.LOST,
                last_access_ns=now_ns,
            )
        if record.state in {
            TierSessionState.PREPARING,
            TierSessionState.ACTIVE,
            TierSessionState.ENDED,
        }:
            raise RuntimeError(
                f"session cannot prepare from state {record.state.value}")
        source = self._primary_copy(record)
        source_tokens = 0 if source is None else source.tokens
        hit_tokens = min(
            reusable_tokens,
            source_tokens,
            max(0, input_tokens - 1),
        )
        effective_source = source if hit_tokens else None
        needs_bounce = (
            self._aggregate_bytes(hit_tokens)
            if effective_source is not None
            and effective_source.tier == Tier.SSD
            else 0
        )
        p_required = self._per_rank_bytes(input_tokens)
        d_required = (
            self._per_rank_bytes(final_tokens) if needs_d else 0)
        infeasible = []
        if p_required > self.p_ledger.capacity_bytes:
            self.metrics.infeasible_p_requests += 1
            infeasible.append(
                f"P={p_required}/{self.p_ledger.capacity_bytes}")
        if d_required > self.d_ledger.capacity_bytes:
            self.metrics.infeasible_d_requests += 1
            infeasible.append(
                f"D={d_required}/{self.d_ledger.capacity_bytes}")
        if needs_bounce > self.cpu_ledger.capacity_bytes:
            self.metrics.infeasible_cpu_bounces += 1
            infeasible.append(
                f"CPU-bounce={needs_bounce}/"
                f"{self.cpu_ledger.capacity_bytes}")
        if infeasible:
            raise RuntimeError(
                "request is individually infeasible: "
                + ", ".join(infeasible))
        prepare_id = self._next_prepare_id
        capacity = self._prepare_capacity(
            record=record,
            source=source,
            input_tokens=input_tokens,
            final_tokens=final_tokens,
            needs_d=needs_d,
            needs_bounce_bytes=needs_bounce,
            prepare_id=prepare_id,
        )
        if capacity is None:
            return None
        (
            p_owner, d_owner, d_reserved, d_reuse_copy_id,
            full_d, bounce_owner,
        ) = capacity
        if is_new_session:
            self.sessions[session_id] = record
        self._next_prepare_id += 1

        record.generation += 1
        record.active_request_id = request_id
        record.last_access_ns = now_ns
        record.state = TierSessionState.PREPARING
        if effective_source is not None:
            effective_source.foreground_pins += 1
        stages: tuple[TierTransferStage, ...]
        if effective_source is None:
            stages = ()
            self.metrics.prepare_misses += 1
        elif effective_source.tier == Tier.D:
            stages = (
                self.resources.peer_stage(
                    hit_tokens,
                    direction="d_to_p",
                    block_rounded=False,
                ),
            )
            self.metrics.d_prepare_hits += 1
        elif effective_source.tier == Tier.CPU:
            stages = (
                self.resources.gpu_cpu_stage(
                    hit_tokens,
                    gpu_role="p",
                    direction="cpu_to_gpu",
                ),
            )
            effective_source.retired = True
            record.primary = None
            record.primary_copy_id = None
            self.metrics.cpu_prepare_hits += 1
        else:
            stages = (
                self.resources.ssd_stage(
                    hit_tokens, direction="ssd_to_cpu"),
                self.resources.gpu_cpu_stage(
                    hit_tokens,
                    gpu_role="p",
                    direction="cpu_to_gpu",
                ),
            )
            effective_source.shadow = True
            record.primary = None
            record.primary_copy_id = None
            self.metrics.ssd_prepare_hits += 1

        job = self._schedule(
            kind=TierJobKind.PREPARE,
            record=record,
            request_id=request_id,
            source_copy=effective_source,
            destination=None,
            destination_owner=None,
            bounce_owner=bounce_owner,
            stages=stages,
            ready_ns=now_ns,
        )
        ticket = PrepareTicket(
            prepare_id=prepare_id,
            job_id=job.job_id,
            session_id=session_id,
            request_id=request_id,
            generation=record.generation,
            source=(
                None if effective_source is None
                else effective_source.tier),
            source_copy_id=(
                None if effective_source is None
                else effective_source.copy_id),
            source_tokens=source_tokens,
            hit_tokens=hit_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            final_tokens=final_tokens,
            has_successor=has_successor,
            needs_d=needs_d,
            p_owner=p_owner,
            p_bytes_per_rank=self._per_rank_bytes(input_tokens),
            d_owner=d_owner,
            d_reserved_bytes_per_rank=d_reserved,
            d_target_bytes_per_rank=(
                self._per_rank_bytes(final_tokens)
                if needs_d else 0),
            d_reuse_copy_id=d_reuse_copy_id,
            full_d_reservation=full_d,
            bounce_owner=bounce_owner,
            stages=job.stages,
            start_ns=job.start_ns,
            completion_ns=job.completion_ns,
        )
        self.prepares[prepare_id] = ticket
        self._seen_request_ids.add(request_id)
        self.metrics.prepare_started += 1

        if source is not None and effective_source is None:
            if source.tier != Tier.D:
                source.retired = True
                record.primary = None
                record.primary_copy_id = None
                self._maybe_release_retired(source)
            # An unused D snapshot still backs the already-allocated D
            # destination.  Its bytes may be overwritten by the new turn,
            # but releasing them here would leave the unreserved base of a
            # delta-only D claim available to another admission.
        self._maybe_assert_invariants()
        return ticket

    def _complete_prepare(self, job: TierTransferJob) -> None:
        record = self.sessions[job.session_id]
        record.pending_job_ids.discard(job.job_id)
        ticket = next(
            ticket for ticket in self.prepares.values()
            if ticket.job_id == job.job_id
        )
        source = (
            None if job.source_copy_id is None
            else self.copies.get(job.source_copy_id)
        )
        if source is not None:
            if source.foreground_pins <= 0:
                raise AssertionError("foreground source pin underflow")
            source.foreground_pins -= 1
            self._maybe_release_retired(source)
        if job.bounce_owner is not None:
            self.cpu_ledger.release(job.bounce_owner)
        self._release_stable_d_source_after_prepare(ticket)
        job.status = TierJobStatus.COMPLETE
        ticket.completed = True
        self._completed_prepares.append(ticket)
        self.metrics.prepare_completed += 1

    def _release_stable_d_source_after_prepare(
            self, ticket: PrepareTicket) -> None:
        """Release old D bytes once a stable source is restored into P."""

        if (
            ticket.d_reuse_copy_id is None
            or ticket.full_d_reservation
        ):
            return
        source = self.copies.get(ticket.d_reuse_copy_id)
        if source is None or source.tier != Tier.D or source.pins:
            return
        before = source.byte_count
        if not ticket.needs_d:
            source.retired = True
            self._release_copy(source)
            ticket.d_reuse_copy_id = None
            self.metrics.d_source_bytes_released_early += before
            return
        target = ticket.d_target_bytes_per_rank
        if target >= before:
            return
        self.d_ledger.set_bytes(source.ledger_owner, target)
        source.byte_count = target
        source.tokens = min(source.tokens, ticket.final_tokens)
        record = self.sessions[ticket.session_id]
        if record.primary_copy_id == source.copy_id:
            record.tokens = source.tokens
        self.metrics.d_source_bytes_released_early += before - target

    def pop_prepare_completed(self) -> list[PrepareTicket]:
        completed = self._completed_prepares
        self._completed_prepares = []
        return completed

    def mark_active(self, ticket: PrepareTicket, *, now_ns: int) -> None:
        self._validate_time(now_ns)
        self.advance(now_ns)
        stored = self.prepares.get(ticket.prepare_id)
        if stored is not ticket:
            raise RuntimeError("stale prepare ticket")
        if not ticket.completed:
            raise RuntimeError("prepare transfer has not completed")
        record = self.sessions[ticket.session_id]
        if (
            record.state != TierSessionState.PREPARING
            or record.active_request_id != ticket.request_id
            or record.generation != ticket.generation
        ):
            raise RuntimeError("prepare ticket no longer owns session")
        ticket.active = True
        record.state = TierSessionState.ACTIVE
        record.last_access_ns = now_ns
        self._maybe_assert_invariants()

    def release_p_after_handoff(
            self, ticket: PrepareTicket, *, now_ns: int) -> None:
        """Release P KV only after the pool's P-to-D handoff completes."""

        self._validate_time(now_ns)
        self.advance(now_ns)
        stored = self.prepares.get(ticket.prepare_id)
        if stored is not ticket or ticket.committed:
            raise RuntimeError("stale or committed prepare ticket")
        if not ticket.active:
            raise RuntimeError("prepare ticket is not active")
        if not ticket.needs_d:
            raise RuntimeError(
                "terminal output-one request has no P-to-D handoff")
        if ticket.p_released:
            raise RuntimeError("P destination was already released")
        record = self.sessions[ticket.session_id]
        if (
            record.state != TierSessionState.ACTIVE
            or record.active_request_id != ticket.request_id
            or record.generation != ticket.generation
        ):
            raise RuntimeError("active ticket no longer owns session")
        released = self.p_ledger.release(ticket.p_owner)
        if released != ticket.p_bytes_per_rank:
            raise AssertionError("P destination ownership mismatch")
        ticket.p_released = True
        self.metrics.p_handoff_releases += 1
        self._maybe_assert_invariants()

    def _complete_demotion(self, job: TierTransferJob) -> None:
        record = self.sessions[job.session_id]
        record.pending_job_ids.discard(job.job_id)
        source = (
            None if job.source_copy_id is None
            else self.copies.get(job.source_copy_id)
        )
        if source is None or source.demotion_pins <= 0:
            raise AssertionError("demotion source pin is missing")
        source.demotion_pins -= 1
        valid = (
            record.version == job.snapshot_version
            and record.generation == job.snapshot_generation
            and record.primary_copy_id == source.copy_id
            and record.state in {
                TierSessionState.D_DEMOTING_CPU,
                TierSessionState.D_DEMOTING_SSD,
                TierSessionState.CPU_DEMOTING_SSD,
            }
        )
        if valid:
            if job.destination is None or job.destination_owner is None:
                raise AssertionError(
                    "demotion has no destination reservation")
            destination = self._new_copy(
                record=record,
                tier=job.destination,
                version=source.version,
                generation=source.generation,
                tokens=source.tokens,
                existing_owner=job.destination_owner,
            )
            source.retired = True
            self._release_copy(source)
            ready_state = {
                Tier.CPU: TierSessionState.CPU_READY,
                Tier.SSD: TierSessionState.SSD_READY,
            }[job.destination]
            self._set_primary(record, destination, ready_state)
            job.status = TierJobStatus.COMMITTED
            self.metrics.demotions_committed += 1
        else:
            if job.destination is not None and (
                    job.destination_owner is not None):
                self._ledger(job.destination).release(
                    job.destination_owner)
            job.status = TierJobStatus.STALE
            self.metrics.stale_demotions += 1
            self.metrics.stale_transfer_bytes += sum(
                stage.stage.aggregate_bytes for stage in job.stages)
            self._maybe_release_retired(source)
        if job.bounce_owner is not None:
            self.cpu_ledger.release(job.bounce_owner)

    def commit_d_ready(
            self, ticket: PrepareTicket, *, now_ns: int,
            has_successor: bool) -> None:
        """Finish active model work and atomically publish its D context."""

        if not isinstance(has_successor, bool):
            raise ValueError("has_successor must be a boolean")
        if has_successor != ticket.has_successor:
            raise ValueError(
                "commit has_successor disagrees with prepare ticket")
        self._validate_time(now_ns)
        self.advance(now_ns)
        stored = self.prepares.get(ticket.prepare_id)
        if stored is not ticket or ticket.committed:
            raise RuntimeError("stale or already committed prepare ticket")
        if not ticket.active:
            raise RuntimeError("prepare ticket is not active")
        record = self.sessions[ticket.session_id]
        if (
            record.state != TierSessionState.ACTIVE
            or record.active_request_id != ticket.request_id
            or record.generation != ticket.generation
        ):
            raise RuntimeError("active ticket no longer owns session")
        if ticket.needs_d and not ticket.p_released:
            raise RuntimeError(
                "P-to-D handoff must complete before D publication")
        if not ticket.p_released:
            released = self.p_ledger.release(ticket.p_owner)
            if released != ticket.p_bytes_per_rank:
                raise AssertionError(
                    "P destination ownership mismatch")
            ticket.p_released = True

        reusable = (
            None if ticket.d_reuse_copy_id is None
            else self.copies.get(ticket.d_reuse_copy_id)
        )
        if has_successor:
            next_version = record.version + 1
            if (
                reusable is not None
                and reusable.pins == 0
                and not ticket.full_d_reservation
            ):
                remove = (
                    ()
                    if ticket.d_owner is None
                    else (ticket.d_owner,)
                )
                self.d_ledger.replace(
                    remove_owners=(
                        (reusable.ledger_owner,) + remove),
                    owner=reusable.ledger_owner,
                    byte_count=ticket.d_target_bytes_per_rank,
                )
                reusable.version = next_version
                reusable.generation = record.generation
                reusable.tokens = ticket.final_tokens
                reusable.byte_count = ticket.d_target_bytes_per_rank
                reusable.retired = False
                reusable.shadow = False
                new_d = reusable
            else:
                if ticket.d_owner is None:
                    raise AssertionError(
                        "new D copy has no destination reservation")
                new_d = self._new_copy(
                    record=record,
                    tier=Tier.D,
                    version=next_version,
                    generation=record.generation,
                    tokens=ticket.final_tokens,
                    existing_owner=ticket.d_owner,
                )
                if reusable is not None:
                    reusable.retired = True
                    self._maybe_release_retired(reusable)
            for copy_id in tuple(record.copy_ids):
                copy = self.copies.get(copy_id)
                if copy is None or copy.copy_id == new_d.copy_id:
                    continue
                copy.retired = True
                copy.shadow = False
                self._maybe_release_retired(copy)
            record.version = next_version
            self._set_primary(
                record, new_d, TierSessionState.D_READY)
        else:
            if ticket.d_owner is not None:
                self.d_ledger.release(ticket.d_owner)
            for copy_id in tuple(record.copy_ids):
                copy = self.copies.get(copy_id)
                if copy is None:
                    continue
                copy.retired = True
                copy.shadow = False
                self._maybe_release_retired(copy)
            record.state = TierSessionState.ENDED
            record.primary = None
            record.primary_copy_id = None
            record.tokens = 0
        record.active_request_id = None
        record.last_access_ns = now_ns
        ticket.committed = True
        self._maybe_assert_invariants()

    def end(self, session_id: str, *, now_ns: int) -> None:
        """Invalidate an idle session and retire every copy safely."""

        self._validate_session(session_id)
        self._validate_time(now_ns)
        self.advance(now_ns)
        record = self.sessions[session_id]
        if record.state in {
            TierSessionState.PREPARING,
            TierSessionState.ACTIVE,
        }:
            raise RuntimeError(
                "active session must finish through commit_d_ready")
        record.generation += 1
        record.state = TierSessionState.ENDED
        record.primary = None
        record.primary_copy_id = None
        record.tokens = 0
        record.last_access_ns = now_ns
        for copy_id in tuple(record.copy_ids):
            copy = self.copies.get(copy_id)
            if copy is not None:
                copy.retired = True
                copy.shadow = False
                self._maybe_release_retired(copy)
        self._maybe_assert_invariants()

    def run_until_idle(self) -> None:
        while self._completion_heap:
            event_ns = self.next_event_ns()
            assert event_ns is not None
            self.advance(event_ns)
        self.assert_invariants()

    def assert_invariants(self) -> None:
        for ledger in (
            self.p_ledger,
            self.d_ledger,
            self.cpu_ledger,
            self.ssd_ledger,
        ):
            ledger.assert_invariants()
        heap_ids = [job_id for _, job_id in self._completion_heap]
        if len(heap_ids) != len(set(heap_ids)):
            raise AssertionError("duplicate transfer completion event")
        for job_id in heap_ids:
            if self.jobs[job_id].status != TierJobStatus.RUNNING:
                raise AssertionError(
                    "non-running job remains on completion heap")
        for job in self.jobs.values():
            destination_bytes = 0
            if (
                job.destination is not None
                and job.destination_owner is not None
            ):
                destination_bytes = self._ledger(
                    job.destination).owner_bytes(
                        job.destination_owner)
            bounce_bytes = (
                0 if job.bounce_owner is None
                else self.cpu_ledger.owner_bytes(job.bounce_owner)
            )
            if job.kind == TierJobKind.DEMOTION:
                if job.status == TierJobStatus.RUNNING:
                    source = self.copies.get(job.source_copy_id)
                    if source is None:
                        raise AssertionError(
                            "running demotion has no source copy")
                    expected = (
                        source.byte_count * self.hardware.tp_size
                        if source.tier == Tier.D
                        else source.byte_count
                    )
                    if destination_bytes != expected:
                        raise AssertionError(
                            "running demotion destination ownership "
                            "mismatch")
                    expected_bounce = (
                        expected
                        if job.bounce_owner is not None else 0
                    )
                    if bounce_bytes != expected_bounce:
                        raise AssertionError(
                            "running demotion bounce ownership mismatch")
                elif destination_bytes or bounce_bytes:
                    raise AssertionError(
                        "completed demotion retains destination capacity")
            elif job.bounce_owner is not None:
                if (
                    job.status == TierJobStatus.RUNNING
                    and bounce_bytes <= 0
                ):
                    raise AssertionError(
                        "running prepare lost bounce ownership")
                if (
                    job.status != TierJobStatus.RUNNING
                    and bounce_bytes
                ):
                    raise AssertionError(
                        "completed prepare retains bounce capacity")
        for copy_id, copy in self.copies.items():
            if copy.copy_id != copy_id:
                raise AssertionError("copy index mismatch")
            record = self.sessions.get(copy.session_id)
            if record is None or copy_id not in record.copy_ids:
                raise AssertionError("copy has no session owner")
            if copy.pins < 0:
                raise AssertionError("copy pin underflow")
            if self._ledger(copy.tier).owner_bytes(
                    copy.ledger_owner) != copy.byte_count:
                raise AssertionError("copy ledger ownership mismatch")
        for record in self.sessions.values():
            if record.primary_copy_id is not None:
                copy = self.copies.get(record.primary_copy_id)
                if copy is None or copy.session_id != record.session_id:
                    raise AssertionError("primary copy index mismatch")
                if record.primary != copy.tier:
                    raise AssertionError("primary tier mismatch")
            elif record.primary is not None:
                raise AssertionError("primary tier has no copy")
            for copy_id in record.copy_ids:
                if copy_id not in self.copies:
                    raise AssertionError("session references missing copy")
            for job_id in record.pending_job_ids:
                job = self.jobs.get(job_id)
                if (
                    job is None
                    or job.status != TierJobStatus.RUNNING
                    or job.session_id != record.session_id
                ):
                    raise AssertionError(
                        "pending transfer index mismatch")
        for ticket in self.prepares.values():
            if ticket.committed and not ticket.p_released:
                raise AssertionError(
                    "committed ticket did not release P destination")
            if ticket.p_released:
                if self.p_ledger.owner_bytes(ticket.p_owner):
                    raise AssertionError(
                        "released ticket retains P destination")
            else:
                if (
                    self.p_ledger.owner_bytes(ticket.p_owner)
                    != ticket.p_bytes_per_rank
                ):
                    raise AssertionError(
                        "prepare P destination ownership mismatch")
            if not ticket.committed:
                if ticket.d_owner is None:
                    if ticket.d_reserved_bytes_per_rank:
                        raise AssertionError(
                            "prepare D reservation has no owner")
                elif (
                    self.d_ledger.owner_bytes(ticket.d_owner)
                    != ticket.d_reserved_bytes_per_rank
                ):
                    raise AssertionError(
                        "prepare D destination ownership mismatch")
            if ticket.committed and ticket.d_owner is not None:
                if self.d_ledger.owner_bytes(ticket.d_owner):
                    raise AssertionError(
                        "committed ticket retains D destination")

    def report(self) -> Mapping[str, Any]:
        return {
            "node_id": self.node_id,
            "policy": self.policy,
            "validate_every_event": self.validate_every_event,
            "current_ns": self.current_ns,
            "completion_order": (
                "transfer_completion_before_same_timestamp_arrival"),
            "seen_request_ids": sorted(self._seen_request_ids),
            "d_hbm_integration": (
                "sole per-rank P/D lifecycle ledger; controller must not "
                "duplicate claims in AtomicPDHBM"),
            "ledgers": {
                "p": self.p_ledger.report(),
                "d": self.d_ledger.report(),
                "cpu": self.cpu_ledger.report(),
                "ssd": self.ssd_ledger.report(),
            },
            "metrics": asdict(self.metrics),
            "sessions": {
                session_id: {
                    **asdict(record),
                    "state": record.state.value,
                    "primary": (
                        None if record.primary is None
                        else record.primary.value),
                    "pending_job_ids": sorted(
                        record.pending_job_ids),
                    "copy_ids": sorted(record.copy_ids),
                }
                for session_id, record in sorted(
                    self.sessions.items())
            },
            "copies": {
                copy_id: {
                    **asdict(copy),
                    "tier": copy.tier.value,
                }
                for copy_id, copy in sorted(self.copies.items())
            },
            "jobs": {
                job_id: {
                    "job_id": job.job_id,
                    "kind": job.kind.value,
                    "session_id": job.session_id,
                    "request_id": job.request_id,
                    "snapshot_version": job.snapshot_version,
                    "snapshot_generation": job.snapshot_generation,
                    "source_copy_id": job.source_copy_id,
                    "destination": (
                        None if job.destination is None
                        else job.destination.value),
                    "start_ns": job.start_ns,
                    "completion_ns": job.completion_ns,
                    "status": job.status.value,
                    "transfer_kinds": job.transfer_kinds,
                }
                for job_id, job in sorted(self.jobs.items())
            },
        }


__all__ = [
    "MAX_CONTEXT_TOKENS",
    "PrepareTicket",
    "ResumeSource",
    "SUPPORTED_TIER_POLICIES",
    "ScheduledTierStage",
    "SharedByteLedger",
    "Tier",
    "TierCopy",
    "TierJobKind",
    "TierJobStatus",
    "TierLifecycleMetrics",
    "TierSession",
    "TierSessionState",
    "TierTransferJob",
    "TieredPDKVLifecycle",
]

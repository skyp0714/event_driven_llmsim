"""Finite GPU-HBM ownership bridge for full-model HBF online serving.

The full-model HBF adapter cannot mutate a Scheduler after the Scheduler has
finished a request.  It therefore emits explicit ``GPUHBMOwnershipEvent``
objects.  This module applies those events to the exact Scheduler
``MemoryModel`` that owns the per-rank allocation.

Two execution contracts are intentionally distinct:

* In a colocated GPU topology, ``RESUME_CLAIM`` can transfer an idle
  allocation directly to a continuation on the same Scheduler.  Metadata
  decoration makes ``Scheduler.add_request`` adopt the already allocated
  prefix without allocating it again.
* In the target P4D4 topology, completed KV lives on D while suffix prefill
  must run on P.  Reusing that KV requires a modeled D-to-P copy and duplicate
  P-side allocation.  That transfer does not exist in the current full-model
  HBF path, so retained ``RESUME_CLAIM`` events are rejected.  The only safe
  live fallback is explicit recomputation after the adapter releases the idle
  D allocation.

The bridge never predicts a transfer completion and never treats a decode
instance as a prefill instance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .hbf_online_adapter import (
    FullModelHBFOnlineAdapter,
    GPUHBMEventKind,
    GPUHBMOwnershipEvent,
)
from .memory_model import Device


GPU_HBM_BRIDGE_SCHEMA = "full-model-hbf-gpu-hbm-bridge-v1"


class GPUHBMBridgeError(RuntimeError):
    """Base class for strict finite-HBM bridge failures."""


class GPUHBMBridgeCapacityError(GPUHBMBridgeError):
    """Raised before an idle allocation would exceed finite GPU HBM."""


class GPUHBMBridgeUnderflowError(GPUHBMBridgeError):
    """Raised before a release would underflow its Scheduler MemoryModel."""


class GPUHBMBridgeStaleEventError(GPUHBMBridgeError):
    """Raised for duplicate, stale, or owner-mismatched events."""


class GPUHBMBridgeUnsupportedReuseError(GPUHBMBridgeError):
    """Raised when P/D reuse would require an unmodeled D-to-P copy."""


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class GPUHBMIdleAllocation:
    session_id: str
    owner_request_id: int
    gpu_instance_id: int
    retained_since_ns: int
    token_count: int
    accounted_tokens_per_rank: int
    logical_bytes: int
    per_rank_bytes: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class GPUHBMPendingClaim:
    session_id: str
    request_id: int
    gpu_instance_id: int
    claim_time_ns: int
    token_count: int
    accounted_tokens_per_rank: int
    logical_bytes: int
    per_rank_bytes: int
    metadata_decorated: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class GPUHBMPDRecomputeBinding:
    session_id: str
    request_id: int
    prefill_instance_id: int
    decode_instance_id: int
    recompute_tokens: int
    metadata_decorated: bool = True

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GPUHBMPDDecodeReservation:
    session_id: str
    request_id: int
    prefill_instance_id: int
    decode_instance_id: int
    projected_context_tokens: int
    full_per_rank_bytes: int
    reserved_per_rank_bytes: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class GPUHBMBridgeMetrics:
    applied_events: int = 0
    rejected_events: int = 0
    turn_retain_events: int = 0
    resume_claim_events: int = 0
    migration_release_events: int = 0
    idle_release_events: int = 0
    allocated_per_rank_bytes: int = 0
    freed_per_rank_bytes: int = 0
    resume_tail_trimmed_per_rank_bytes: int = 0
    colocated_metadata_decorations: int = 0
    colocated_request_adoptions: int = 0
    pd_recompute_metadata_decorations: int = 0
    pd_recompute_request_bindings: int = 0
    pd_decode_reservations: int = 0
    pd_decode_reservation_waits: int = 0
    pd_decode_reservations_consumed: int = 0
    pd_decode_reservations_cancelled: int = 0
    pd_decode_reserved_per_rank_bytes: int = 0
    pd_decode_transferred_to_scheduler_per_rank_bytes: int = 0
    pd_decode_cancelled_per_rank_bytes: int = 0


class FullModelHBFGPUHBMBridge:
    """Apply adapter ownership events to exact finite Scheduler memories."""

    def __init__(
            self, schedulers_by_instance: Mapping[int, object], *,
            pd_pairs: Iterable[tuple[int, int]] = (),
            fallback_reuse_mode: str | None = None,
            adapter: FullModelHBFOnlineAdapter | None = None,
            validate_every_event: bool = True) -> None:
        if not isinstance(schedulers_by_instance, Mapping):
            raise TypeError(
                "schedulers_by_instance must be an instance-id mapping")
        if not schedulers_by_instance:
            raise ValueError(
                "schedulers_by_instance must contain at least one Scheduler")
        if not isinstance(validate_every_event, bool):
            raise ValueError("validate_every_event must be a boolean")

        self.schedulers: dict[int, object] = {}
        self.memories: dict[int, object] = {}
        for raw_instance_id, scheduler in schedulers_by_instance.items():
            instance_id = _integer(
                "scheduler instance_id", raw_instance_id)
            observed_instance = _integer(
                "scheduler.instance_id",
                getattr(scheduler, "instance_id", None),
            )
            if observed_instance != instance_id:
                raise ValueError(
                    "Scheduler mapping key differs from instance_id: "
                    f"key={instance_id}, observed={observed_instance}")
            memory = getattr(scheduler, "memory", None)
            for method_name in ("get_kv", "allocate", "free"):
                if not callable(getattr(memory, method_name, None)):
                    raise TypeError(
                        f"Scheduler {instance_id} memory lacks "
                        f"{method_name}()")
            if _integer(
                    "memory.instance_id",
                    getattr(memory, "instance_id", None)) != instance_id:
                raise ValueError(
                    f"Scheduler {instance_id} MemoryModel owner differs")
            if bool(getattr(scheduler, "enable_prefix_caching", False)):
                raise ValueError(
                    "full-model HBF finite ownership is incompatible with "
                    f"generic prefix caching: instance={instance_id}")
            block_size = _integer(
                "scheduler.block_size",
                getattr(scheduler, "block_size", None),
                minimum=1,
            )
            if _integer(
                    "memory.block_size",
                    getattr(memory, "block_size", None),
                    minimum=1) != block_size:
                raise ValueError(
                    f"Scheduler {instance_id} block size differs from memory")
            block_bytes = _integer(
                "per-rank KV block bytes",
                memory.get_kv(block_size),
                minimum=1,
            )
            if block_bytes <= 0:
                raise ValueError(
                    f"Scheduler {instance_id} has zero-byte KV blocks")
            self.schedulers[instance_id] = scheduler
            self.memories[instance_id] = memory

        pairs = tuple(pd_pairs)
        self.pd_pairs: tuple[tuple[int, int], ...] = ()
        self._pd_pair_by_prefill: dict[int, int] = {}
        self._pd_pair_by_decode: dict[int, int] = {}
        for raw_pair in pairs:
            if (
                not isinstance(raw_pair, (tuple, list))
                or len(raw_pair) != 2
            ):
                raise TypeError(
                    "each P/D pair must be (prefill_id, decode_id)")
            prefill_id = _integer("prefill_instance_id", raw_pair[0])
            decode_id = _integer("decode_instance_id", raw_pair[1])
            if prefill_id == decode_id:
                raise ValueError("P/D instances must be distinct")
            if prefill_id not in self.schedulers:
                raise ValueError(
                    f"unknown P/D prefill Scheduler {prefill_id}")
            if decode_id not in self.schedulers:
                raise ValueError(
                    f"unknown P/D decode Scheduler {decode_id}")
            if prefill_id in self._pd_pair_by_prefill:
                raise ValueError(
                    f"duplicate P/D prefill Scheduler {prefill_id}")
            if decode_id in self._pd_pair_by_decode:
                raise ValueError(
                    f"duplicate P/D decode Scheduler {decode_id}")
            prefill = self.schedulers[prefill_id]
            decode = self.schedulers[decode_id]
            if getattr(prefill, "pd_type", None) != "prefill":
                raise ValueError(
                    f"Scheduler {prefill_id} is not a prefill instance")
            if getattr(decode, "pd_type", None) != "decode":
                raise ValueError(
                    f"Scheduler {decode_id} is not a decode instance")
            if (
                int(prefill.block_size) != int(decode.block_size)
                or int(prefill.memory.get_kv(prefill.block_size))
                != int(decode.memory.get_kv(decode.block_size))
            ):
                raise ValueError(
                    "P/D pair has incompatible per-rank KV block geometry: "
                    f"pair=({prefill_id}, {decode_id})")
            self._pd_pair_by_prefill[prefill_id] = decode_id
            self._pd_pair_by_decode[decode_id] = prefill_id
        self.pd_pairs = tuple(sorted(
            self._pd_pair_by_prefill.items()))
        self.topology = "pd" if self.pd_pairs else "colocated"
        if fallback_reuse_mode is None:
            fallback_reuse_mode = (
                "recompute" if self.topology == "pd"
                else "sticky_reuse"
            )
        expected_mode = (
            "recompute" if self.topology == "pd"
            else "sticky_reuse"
        )
        if fallback_reuse_mode != expected_mode:
            raise ValueError(
                f"{self.topology} fallback_reuse_mode must be "
                f"{expected_mode!r}")
        self.fallback_reuse_mode = fallback_reuse_mode
        self.validate_every_event = validate_every_event

        self._idle_by_session: dict[str, GPUHBMIdleAllocation] = {}
        self._pending_claim_by_request: dict[
            int, GPUHBMPendingClaim] = {}
        self._pd_recompute_by_request: dict[
            int, GPUHBMPDRecomputeBinding] = {}
        self._pd_decode_reservation_by_request: dict[
            int, GPUHBMPDDecodeReservation] = {}
        self._applied_event_fingerprints: set[tuple[object, ...]] = set()
        self._last_event_time_by_session: dict[str, int] = {}
        self._turn_request_ids: set[int] = set()
        self._adopted_request_ids: set[int] = set()
        self._bound_pd_recompute_request_ids: set[int] = set()
        self._history: list[dict[str, object]] = []
        self.metrics = GPUHBMBridgeMetrics()
        self.adapter_contract: dict[str, object] | None = None
        if adapter is not None:
            self.validate_adapter_contract(adapter)
        self.assert_invariants()

    def validate_adapter_contract(
            self, adapter: FullModelHBFOnlineAdapter,
    ) -> dict[str, object]:
        """Fail early when adapter accounting cannot match Scheduler HBM."""

        if not isinstance(adapter, FullModelHBFOnlineAdapter):
            raise TypeError("adapter must be a FullModelHBFOnlineAdapter")
        if (
            self.topology == "pd"
            and adapter.gpu_resume_mode != "recompute"
        ):
            raise GPUHBMBridgeUnsupportedReuseError(
                "P/D bridge requires adapter gpu_resume_mode='recompute'; "
                "sticky reuse would require an unmodeled D-to-P restore")
        for instance_id, memory in self.memories.items():
            if (
                int(memory.block_size)
                != int(adapter.gpu_block_size_tokens)
            ):
                raise GPUHBMBridgeError(
                    "adapter and Scheduler KV block sizes differ: "
                    f"instance={instance_id}, scheduler="
                    f"{memory.block_size}, adapter="
                    f"{adapter.gpu_block_size_tokens}")
            expected_token_bytes = int(memory.get_kv(1))
            if (
                expected_token_bytes
                != int(adapter.gpu_kv_bytes_per_token_per_rank)
            ):
                raise GPUHBMBridgeError(
                    "adapter and Scheduler per-rank KV geometry differ: "
                    f"instance={instance_id}, scheduler="
                    f"{expected_token_bytes}, adapter="
                    f"{adapter.gpu_kv_bytes_per_token_per_rank}")
        result = {
            "gpu_resume_mode": adapter.gpu_resume_mode,
            "gpu_tp_size": int(adapter.gpu_tp_size),
            "gpu_block_size_tokens": int(
                adapter.gpu_block_size_tokens),
            "gpu_kv_bytes_per_token_per_rank": int(
                adapter.gpu_kv_bytes_per_token_per_rank),
        }
        self.adapter_contract = result
        return dict(result)

    @staticmethod
    def _fingerprint(
            event: GPUHBMOwnershipEvent) -> tuple[object, ...]:
        return (
            event.kind.value,
            event.session_id,
            event.request_id,
            event.gpu_instance_id,
            event.time_ns,
            event.token_count,
            event.accounted_tokens_per_rank,
            event.logical_bytes,
            event.per_rank_bytes,
            event.reason,
        )

    def _validate_event(
            self, event: GPUHBMOwnershipEvent) -> object:
        if not isinstance(event, GPUHBMOwnershipEvent):
            raise TypeError("event must be a GPUHBMOwnershipEvent")
        if not isinstance(event.kind, GPUHBMEventKind):
            raise TypeError("event.kind must be a GPUHBMEventKind")
        session_id = _identifier("event.session_id", event.session_id)
        _integer("event.request_id", event.request_id)
        instance_id = _integer(
            "event.gpu_instance_id", event.gpu_instance_id)
        _integer("event.time_ns", event.time_ns)
        token_count = _integer("event.token_count", event.token_count)
        accounted_tokens = _integer(
            "event.accounted_tokens_per_rank",
            event.accounted_tokens_per_rank,
        )
        _integer("event.logical_bytes", event.logical_bytes)
        per_rank_bytes = _integer(
            "event.per_rank_bytes", event.per_rank_bytes)
        _identifier("event.reason", event.reason)
        memory = self.memories.get(instance_id)
        if memory is None:
            raise GPUHBMBridgeStaleEventError(
                f"event targets unknown GPU instance {instance_id}")
        if self.topology == "pd" and instance_id not in self._pd_pair_by_decode:
            raise GPUHBMBridgeStaleEventError(
                "P/D idle KV must be owned by a decode Scheduler: "
                f"instance={instance_id}")
        block_size = int(memory.block_size)
        expected_accounted = (
            (token_count + block_size - 1) // block_size * block_size
            if token_count else 0
        )
        if accounted_tokens != expected_accounted:
            raise GPUHBMBridgeError(
                "event token accounting is not MemoryModel block-rounded: "
                f"instance={instance_id}, tokens={token_count}, "
                f"expected={expected_accounted}, "
                f"actual={accounted_tokens}")
        expected_bytes = int(memory.get_kv(accounted_tokens))
        if per_rank_bytes != expected_bytes:
            raise GPUHBMBridgeError(
                "event per-rank bytes differ from exact MemoryModel KV: "
                f"instance={instance_id}, tokens={accounted_tokens}, "
                f"expected={expected_bytes}, actual={per_rank_bytes}")
        if bool(token_count) != bool(event.logical_bytes):
            raise GPUHBMBridgeError(
                "event logical bytes and token lineage disagree")
        prior_time = self._last_event_time_by_session.get(session_id)
        if prior_time is not None and event.time_ns < prior_time:
            raise GPUHBMBridgeStaleEventError(
                "GPU HBM event moved session time backwards: "
                f"session={session_id!r}, prior={prior_time}, "
                f"actual={event.time_ns}")
        fingerprint = self._fingerprint(event)
        if fingerprint in self._applied_event_fingerprints:
            raise GPUHBMBridgeStaleEventError(
                "duplicate GPU HBM ownership event: "
                f"kind={event.kind.value}, session={session_id!r}, "
                f"request={event.request_id}")
        return memory

    @staticmethod
    def _available_bytes(memory: object) -> int:
        return int(memory.npu_allocatable_mem) - int(memory.npu_used)

    @staticmethod
    def _dynamic_used_bytes(memory: object) -> int:
        # The serving loop releases model weights before final reporting.
        # At that point npu_used is zero while the immutable ``weight`` field
        # still records the original baseline. No dynamic KV is negative.
        return max(0, int(memory.npu_used) - int(memory.weight))

    def _allocate(self, instance_id: int, per_rank_bytes: int) -> None:
        memory = self.memories[instance_id]
        available = self._available_bytes(memory)
        if per_rank_bytes > available:
            raise GPUHBMBridgeCapacityError(
                "finite GPU HBM cannot retain completed KV: "
                f"instance={instance_id}, required={per_rank_bytes}, "
                f"available={available}")
        memory.allocate(per_rank_bytes, Device.NPU)
        self.metrics.allocated_per_rank_bytes += per_rank_bytes

    def _free(self, instance_id: int, per_rank_bytes: int) -> None:
        memory = self.memories[instance_id]
        dynamic_used = self._dynamic_used_bytes(memory)
        if per_rank_bytes > dynamic_used:
            raise GPUHBMBridgeUnderflowError(
                "finite GPU HBM release exceeds dynamic allocation: "
                f"instance={instance_id}, release={per_rank_bytes}, "
                f"dynamic_used={dynamic_used}")
        memory.free(per_rank_bytes, Device.NPU)
        self.metrics.freed_per_rank_bytes += per_rank_bytes

    def _retain_turn(self, event: GPUHBMOwnershipEvent) -> dict[str, object]:
        if event.per_rank_bytes <= 0:
            raise GPUHBMBridgeError(
                "TURN_RETAIN requires positive per-rank KV bytes")
        if event.session_id in self._idle_by_session:
            raise GPUHBMBridgeStaleEventError(
                "TURN_RETAIN would replace live idle ownership: "
                f"session={event.session_id!r}")
        if any(
            claim.session_id == event.session_id
            for claim in self._pending_claim_by_request.values()
        ):
            raise GPUHBMBridgeStaleEventError(
                "TURN_RETAIN raced an unadopted continuation claim: "
                f"session={event.session_id!r}")
        if event.request_id in self._turn_request_ids:
            raise GPUHBMBridgeStaleEventError(
                f"request {event.request_id} already retained a GPU turn")
        self._allocate(event.gpu_instance_id, event.per_rank_bytes)
        self._idle_by_session[event.session_id] = GPUHBMIdleAllocation(
            session_id=event.session_id,
            owner_request_id=event.request_id,
            gpu_instance_id=event.gpu_instance_id,
            retained_since_ns=event.time_ns,
            token_count=event.token_count,
            accounted_tokens_per_rank=(
                event.accounted_tokens_per_rank),
            logical_bytes=event.logical_bytes,
            per_rank_bytes=event.per_rank_bytes,
        )
        self._turn_request_ids.add(event.request_id)
        self.metrics.turn_retain_events += 1
        return {
            "action": "turn_retain",
            "allocated_per_rank_bytes": event.per_rank_bytes,
        }

    def _claim_resume(self, event: GPUHBMOwnershipEvent) -> dict[str, object]:
        if self.topology == "pd":
            raise GPUHBMBridgeUnsupportedReuseError(
                "P/D RESUME_CLAIM is disabled: idle KV is on D but suffix "
                "prefill must run on P, and no D-to-P restore is modeled. "
                "Configure the adapter fallback policy to release idle KV "
                "and route this continuation as GPU recompute.")
        idle = self._idle_by_session.get(event.session_id)
        if idle is None:
            raise GPUHBMBridgeStaleEventError(
                "RESUME_CLAIM has no idle GPU allocation: "
                f"session={event.session_id!r}")
        if idle.gpu_instance_id != event.gpu_instance_id:
            raise GPUHBMBridgeStaleEventError(
                "RESUME_CLAIM changed sticky GPU ownership: "
                f"session={event.session_id!r}, "
                f"expected={idle.gpu_instance_id}, "
                f"actual={event.gpu_instance_id}")
        if event.request_id in self._pending_claim_by_request:
            raise GPUHBMBridgeStaleEventError(
                f"request {event.request_id} already has a pending claim")
        if event.request_id in self._adopted_request_ids:
            raise GPUHBMBridgeStaleEventError(
                f"request {event.request_id} already adopted GPU KV")
        if (
            event.per_rank_bytes <= 0
            or event.per_rank_bytes > idle.per_rank_bytes
        ):
            raise GPUHBMBridgeUnderflowError(
                "RESUME_CLAIM cannot grow or erase retained GPU KV: "
                f"retained={idle.per_rank_bytes}, "
                f"claim={event.per_rank_bytes}")
        trimmed = idle.per_rank_bytes - event.per_rank_bytes
        if trimmed:
            self._free(event.gpu_instance_id, trimmed)
        del self._idle_by_session[event.session_id]
        self._pending_claim_by_request[
            event.request_id] = GPUHBMPendingClaim(
                session_id=event.session_id,
                request_id=event.request_id,
                gpu_instance_id=event.gpu_instance_id,
                claim_time_ns=event.time_ns,
                token_count=event.token_count,
                accounted_tokens_per_rank=(
                    event.accounted_tokens_per_rank),
                logical_bytes=event.logical_bytes,
                per_rank_bytes=event.per_rank_bytes,
            )
        self.metrics.resume_claim_events += 1
        self.metrics.resume_tail_trimmed_per_rank_bytes += trimmed
        return {
            "action": "resume_claim",
            "transferred_per_rank_bytes": event.per_rank_bytes,
            "trimmed_per_rank_bytes": trimmed,
        }

    def _release_idle(
            self, event: GPUHBMOwnershipEvent, *,
            migration: bool) -> dict[str, object]:
        idle = self._idle_by_session.get(event.session_id)
        if idle is None:
            raise GPUHBMBridgeStaleEventError(
                f"{event.kind.value} has no idle GPU allocation: "
                f"session={event.session_id!r}")
        if idle.gpu_instance_id != event.gpu_instance_id:
            raise GPUHBMBridgeStaleEventError(
                f"{event.kind.value} changed sticky GPU ownership: "
                f"expected={idle.gpu_instance_id}, "
                f"actual={event.gpu_instance_id}")
        if migration:
            if event.request_id != idle.owner_request_id:
                raise GPUHBMBridgeStaleEventError(
                    "MIGRATION_RELEASE changed source request ownership: "
                    f"expected={idle.owner_request_id}, "
                    f"actual={event.request_id}")
            if event.per_rank_bytes != idle.per_rank_bytes:
                raise GPUHBMBridgeUnderflowError(
                    "MIGRATION_RELEASE differs from retained allocation: "
                    f"retained={idle.per_rank_bytes}, "
                    f"release={event.per_rank_bytes}")
        elif (
            event.per_rank_bytes not in {0, idle.per_rank_bytes}
        ):
            raise GPUHBMBridgeUnderflowError(
                "IDLE_RELEASE must release the entire retained allocation "
                "or use the zero-lineage sentinel: "
                f"retained={idle.per_rank_bytes}, "
                f"event={event.per_rank_bytes}")
        self._free(idle.gpu_instance_id, idle.per_rank_bytes)
        del self._idle_by_session[event.session_id]
        if migration:
            self.metrics.migration_release_events += 1
        else:
            self.metrics.idle_release_events += 1
        return {
            "action": event.kind.value,
            "freed_per_rank_bytes": idle.per_rank_bytes,
        }

    def apply_event(
            self, event: GPUHBMOwnershipEvent) -> dict[str, object]:
        """Apply one exact adapter event transactionally."""

        try:
            self._validate_event(event)
            if event.kind == GPUHBMEventKind.TURN_RETAIN:
                detail = self._retain_turn(event)
            elif event.kind == GPUHBMEventKind.RESUME_CLAIM:
                detail = self._claim_resume(event)
            elif event.kind == GPUHBMEventKind.MIGRATION_RELEASE:
                detail = self._release_idle(event, migration=True)
            elif event.kind == GPUHBMEventKind.IDLE_RELEASE:
                detail = self._release_idle(event, migration=False)
            else:
                raise AssertionError(
                    f"unhandled GPU HBM event kind {event.kind}")
        except Exception:
            self.metrics.rejected_events += 1
            raise

        fingerprint = self._fingerprint(event)
        self._applied_event_fingerprints.add(fingerprint)
        self._last_event_time_by_session[event.session_id] = event.time_ns
        self.metrics.applied_events += 1
        row = {
            "kind": event.kind.value,
            "session_id": event.session_id,
            "request_id": event.request_id,
            "gpu_instance_id": event.gpu_instance_id,
            "time_ns": event.time_ns,
            **detail,
        }
        self._history.append(row)
        if self.validate_every_event:
            self.assert_invariants()
        return dict(row)

    def apply_events(
            self, events: Iterable[GPUHBMOwnershipEvent],
    ) -> tuple[dict[str, object], ...]:
        """Apply events in caller-provided causal order."""

        if isinstance(events, (str, bytes, Mapping)):
            raise TypeError("events must be an iterable of ownership events")
        return tuple(self.apply_event(event) for event in events)

    @staticmethod
    def _metadata_request_id(metadata: Mapping[str, Any]) -> int:
        if "index" in metadata:
            return _integer("metadata.index", metadata["index"])
        if "request_id" in metadata:
            return _integer(
                "metadata.request_id", metadata["request_id"])
        raise ValueError("continuation metadata lacks index/request_id")

    @staticmethod
    def _metadata_session_id(metadata: Mapping[str, Any]) -> str:
        return _identifier(
            "metadata.session_id", metadata.get("session_id"))

    def decorate_colocated_continuation(
            self, request_id: int,
            metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Decorate one same-Scheduler continuation for zero-copy adoption."""

        if self.topology != "colocated":
            raise GPUHBMBridgeUnsupportedReuseError(
                "colocated prefix adoption is forbidden in P/D topology")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        request_id = _integer("request_id", request_id)
        if self._metadata_request_id(metadata) != request_id:
            raise GPUHBMBridgeStaleEventError(
                "continuation metadata changed request identity")
        claim = self._pending_claim_by_request.get(request_id)
        if claim is None:
            raise GPUHBMBridgeStaleEventError(
                f"request {request_id} has no pending RESUME_CLAIM")
        if self._metadata_session_id(metadata) != claim.session_id:
            raise GPUHBMBridgeStaleEventError(
                "continuation metadata changed session identity")
        if claim.metadata_decorated:
            raise GPUHBMBridgeStaleEventError(
                f"request {request_id} metadata was already decorated")
        for key in ("prefix_reuse_toks", "agentic_kv_hit_tokens"):
            observed = metadata.get(key)
            if observed is not None and int(observed) != claim.token_count:
                raise GPUHBMBridgeStaleEventError(
                    f"{key} differs from RESUME_CLAIM: "
                    f"claim={claim.token_count}, observed={observed}")
        required = metadata.get("hbf_gpu_required_instance_id")
        if required is not None and int(required) != claim.gpu_instance_id:
            raise GPUHBMBridgeStaleEventError(
                "adapter and bridge sticky GPU instances differ")

        result = dict(metadata)
        result["prefix_reuse_toks"] = claim.token_count
        result["agentic_kv_hit_tokens"] = claim.token_count
        result["agentic_kv_owner_instance_id"] = claim.gpu_instance_id
        result["agentic_kv_retained_instance_id"] = None
        result["agentic_kv_retained_per_rank_bytes"] = 0
        result["hbf_gpu_required_instance_id"] = claim.gpu_instance_id
        result["hbf_gpu_required_prefill_instance_id"] = (
            claim.gpu_instance_id)
        result["hbf_gpu_required_decode_instance_id"] = (
            claim.gpu_instance_id)
        result["hbf_gpu_preallocated_prefix_per_rank_bytes"] = (
            claim.per_rank_bytes)
        result["hbf_gpu_hbm_bridge_mode"] = "colocated_retained"
        claim.metadata_decorated = True
        self.metrics.colocated_metadata_decorations += 1
        return result

    def bind_colocated_continuation(self, request: object) -> dict[str, object]:
        """Transfer a decorated claim to the constructed Scheduler Request."""

        request_id = _integer(
            "request.id", getattr(request, "id", None))
        claim = self._pending_claim_by_request.get(request_id)
        if claim is None:
            raise GPUHBMBridgeStaleEventError(
                f"request {request_id} has no pending claim to adopt")
        if not claim.metadata_decorated:
            raise GPUHBMBridgeStaleEventError(
                f"request {request_id} was constructed before decoration")
        if _identifier(
                "request.session_id",
                getattr(request, "session_id", None)) != claim.session_id:
            raise GPUHBMBridgeStaleEventError(
                "constructed continuation changed session identity")
        if _integer(
                "request.instance_id",
                getattr(request, "instance_id", None)) != (
                    claim.gpu_instance_id):
            raise GPUHBMBridgeStaleEventError(
                "constructed continuation changed sticky GPU instance")
        if (
            int(getattr(request, "agentic_kv_hit_tokens", -1))
            != claim.token_count
            or int(getattr(request, "num_computed_tokens", -1))
            != claim.token_count
        ):
            raise GPUHBMBridgeStaleEventError(
                "Scheduler Request did not adopt the exact claimed prefix")
        if (
            getattr(request, "agentic_kv_owner_instance_id", None)
            != claim.gpu_instance_id
            or getattr(
                request, "agentic_kv_retained_instance_id", None)
                is not None
            or int(getattr(
                request, "agentic_kv_retained_per_rank_bytes", -1)) != 0
        ):
            raise GPUHBMBridgeStaleEventError(
                "Scheduler Request has inconsistent colocated KV ownership")
        del self._pending_claim_by_request[request_id]
        self._adopted_request_ids.add(request_id)
        self.metrics.colocated_request_adoptions += 1
        row = {
            "request_id": request_id,
            "session_id": claim.session_id,
            "gpu_instance_id": claim.gpu_instance_id,
            "adopted_tokens": claim.token_count,
            "adopted_per_rank_bytes": claim.per_rank_bytes,
        }
        if self.validate_every_event:
            self.assert_invariants()
        return row

    def decorate_pd_recompute(
            self, request_id: int, metadata: Mapping[str, Any], *,
            prefill_instance_id: int,
            decode_instance_id: int) -> dict[str, Any]:
        """Decorate a safe P/D fallback with no retained-prefix assumption."""

        if self.topology != "pd":
            raise GPUHBMBridgeError(
                "P/D recompute decoration requires configured P/D pairs")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        request_id = _integer("request_id", request_id)
        if self._metadata_request_id(metadata) != request_id:
            raise GPUHBMBridgeStaleEventError(
                "P/D recompute metadata changed request identity")
        session_id = self._metadata_session_id(metadata)
        prefill_id = _integer(
            "prefill_instance_id", prefill_instance_id)
        decode_id = _integer(
            "decode_instance_id", decode_instance_id)
        if self._pd_pair_by_prefill.get(prefill_id) != decode_id:
            raise GPUHBMBridgeStaleEventError(
                "P/D recompute selected an unconfigured pair: "
                f"pair=({prefill_id}, {decode_id})")
        if session_id in self._idle_by_session:
            raise GPUHBMBridgeStaleEventError(
                "P/D recompute metadata cannot bypass a live idle D "
                "allocation; apply the adapter IDLE_RELEASE first")
        if request_id in self._pd_recompute_by_request:
            raise GPUHBMBridgeStaleEventError(
                f"request {request_id} P/D metadata was already decorated")
        if request_id in self._bound_pd_recompute_request_ids:
            raise GPUHBMBridgeStaleEventError(
                f"request {request_id} P/D recompute was already bound")
        reuse_tokens = max(
            int(metadata.get("prefix_reuse_toks") or 0),
            int(metadata.get("agentic_kv_hit_tokens") or 0),
        )
        existing_recompute = int(
            metadata.get("agentic_kv_recompute_tokens") or 0)
        recompute_tokens = max(reuse_tokens, existing_recompute)

        result = dict(metadata)
        result["_pd_prefill_instance_id"] = prefill_id
        result["prefix_reuse_toks"] = 0
        result["agentic_kv_hit_tokens"] = 0
        result["agentic_kv_recompute_tokens"] = recompute_tokens
        result["agentic_kv_owner_instance_id"] = None
        result["agentic_kv_retained_instance_id"] = None
        result["agentic_kv_retained_per_rank_bytes"] = 0
        # The ambiguous adapter field must not be interpreted as a prefill
        # target.  Explicit P/D fields carry the physical routing contract.
        result["hbf_gpu_required_instance_id"] = None
        result["hbf_gpu_required_prefill_instance_id"] = prefill_id
        result["hbf_gpu_required_decode_instance_id"] = decode_id
        result["hbf_gpu_hbm_bridge_mode"] = "pd_recompute"
        result["hbf_gpu_unmodeled_d2p_restore"] = True
        self._pd_recompute_by_request[
            request_id] = GPUHBMPDRecomputeBinding(
                session_id=session_id,
                request_id=request_id,
                prefill_instance_id=prefill_id,
                decode_instance_id=decode_id,
                recompute_tokens=recompute_tokens,
            )
        self.metrics.pd_recompute_metadata_decorations += 1
        return result

    def bind_pd_recompute(self, request: object) -> dict[str, object]:
        """Validate that the constructed P request owns no retained prefix."""

        request_id = _integer(
            "request.id", getattr(request, "id", None))
        binding = self._pd_recompute_by_request.get(request_id)
        if binding is None:
            raise GPUHBMBridgeStaleEventError(
                f"request {request_id} has no P/D recompute binding")
        if _identifier(
                "request.session_id",
                getattr(request, "session_id", None)) != binding.session_id:
            raise GPUHBMBridgeStaleEventError(
                "P/D recompute Request changed session identity")
        if _integer(
                "request.instance_id",
                getattr(request, "instance_id", None)) != (
                    binding.prefill_instance_id):
            raise GPUHBMBridgeStaleEventError(
                "P/D recompute Request did not bind to required P")
        if (
            int(getattr(request, "agentic_kv_hit_tokens", -1)) != 0
            or int(getattr(request, "num_computed_tokens", -1)) != 0
            or getattr(request, "agentic_kv_owner_instance_id", None)
                is not None
            or getattr(
                request, "agentic_kv_retained_instance_id", None)
                is not None
            or int(getattr(
                request, "agentic_kv_retained_per_rank_bytes", -1)) != 0
        ):
            raise GPUHBMBridgeStaleEventError(
                "P/D fallback Request silently retained GPU prefix state")
        del self._pd_recompute_by_request[request_id]
        self._bound_pd_recompute_request_ids.add(request_id)
        self.metrics.pd_recompute_request_bindings += 1
        row = {
            "request_id": request_id,
            "session_id": binding.session_id,
            "prefill_instance_id": binding.prefill_instance_id,
            "decode_instance_id": binding.decode_instance_id,
            "recompute_tokens": binding.recompute_tokens,
        }
        if self.validate_every_event:
            self.assert_invariants()
        return row

    def _project_pd_decode_context_capacity(
            self, context_tokens: int, *,
            decode_instance_id: int) -> tuple[int, int, int]:
        """Return ``(required, dynamic_ceiling, decode_id)`` for D KV."""

        if self.topology != "pd":
            raise GPUHBMBridgeError(
                "P/D decode capacity validation requires configured pairs")
        decode_id = _integer(
            "decode_instance_id", decode_instance_id)
        if decode_id not in self._pd_pair_by_decode:
            raise GPUHBMBridgeStaleEventError(
                "P/D decode capacity validation selected an unconfigured "
                f"decode instance: {decode_id}")
        decode = self.schedulers[decode_id]
        block_size = int(decode.memory.block_size)
        accounted_tokens = (
            (context_tokens + block_size - 1)
            // block_size
            * block_size
        )
        required = int(decode.memory.get_kv(accounted_tokens))
        dynamic_ceiling = max(
            0,
            int(decode.memory.npu_allocatable_mem)
            - int(decode.memory.weight),
        )
        return required, dynamic_ceiling, decode_id

    def validate_pd_decode_prompt_capacity(
            self, input_tokens: int, *,
            decode_instance_id: int) -> int:
        """Return exact projected prompt D bytes or reject an oversize."""

        tokens = _integer(
            "input_tokens", input_tokens, minimum=1)
        required, dynamic_ceiling, decode_id = (
            self._project_pd_decode_context_capacity(
                tokens,
                decode_instance_id=decode_instance_id,
            )
        )
        if required > dynamic_ceiling:
            raise GPUHBMBridgeCapacityError(
                "one P/D decode request exceeds finite D-HBM capacity: "
                f"input_tokens={tokens}, required={required}, "
                f"dynamic_ceiling={dynamic_ceiling}, "
                f"decode_instance={decode_id}")
        return required

    def validate_pd_decode_request_capacity(
            self, input_tokens: int,
            requested_output_tokens: int, *,
            decode_instance_id: int) -> int:
        """Reject a turn whose terminal materialized KV can never fit D."""

        input_count = _integer(
            "input_tokens", input_tokens, minimum=1)
        output_count = _integer(
            "requested_output_tokens",
            requested_output_tokens,
            minimum=1,
        )
        terminal_tokens = input_count + output_count - 1
        required, dynamic_ceiling, decode_id = (
            self._project_pd_decode_context_capacity(
                terminal_tokens,
                decode_instance_id=decode_instance_id,
            )
        )
        if required > dynamic_ceiling:
            raise GPUHBMBridgeCapacityError(
                "one full P/D decode turn exceeds finite D-HBM capacity: "
                f"input_tokens={input_count}, "
                f"requested_output_tokens={output_count}, "
                f"terminal_materialized_tokens={terminal_tokens}, "
                f"required={required}, "
                f"dynamic_ceiling={dynamic_ceiling}, "
                f"decode_instance={decode_id}")
        return required

    def try_reserve_pd_decode(
            self, request: object, *,
            prefill_instance_id: int,
            decode_instance_id: int) -> bool:
        """Reserve finite D HBM before a full-model-HBF P request launches."""

        if self.topology != "pd":
            raise GPUHBMBridgeError(
                "P/D decode reservation requires configured P/D pairs")
        request_id = _integer(
            "request.id", getattr(request, "id", None))
        session_id = _identifier(
            "request.session_id", getattr(request, "session_id", None))
        prefill_id = _integer(
            "prefill_instance_id", prefill_instance_id)
        decode_id = _integer(
            "decode_instance_id", decode_instance_id)
        if self._pd_pair_by_prefill.get(prefill_id) != decode_id:
            raise GPUHBMBridgeStaleEventError(
                "P/D decode reservation selected an unconfigured pair: "
                f"pair=({prefill_id}, {decode_id})")
        existing = self._pd_decode_reservation_by_request.get(
            request_id)
        if existing is not None:
            if (
                existing.session_id != session_id
                or existing.prefill_instance_id != prefill_id
                or existing.decode_instance_id != decode_id
            ):
                raise GPUHBMBridgeStaleEventError(
                    "P/D decode reservation identity changed on retry")
            return True

        decode = self.schedulers[decode_id]
        if (
            getattr(request, "agentic_kv_owner_instance_id", None)
                is not None
            or getattr(
                request, "agentic_kv_retained_instance_id", None)
                is not None
            or int(getattr(
                request, "agentic_kv_retained_per_rank_bytes", 0)) != 0
        ):
            raise GPUHBMBridgeUnsupportedReuseError(
                "full-model HBF P/D receive reservation cannot adopt "
                "retained decode KV")
        if bool(getattr(decode, "enable_prefix_caching", False)):
            raise GPUHBMBridgeUnsupportedReuseError(
                "full-model HBF P/D receive reservation is incompatible "
                "with generic prefix caching")
        if int(getattr(request, "num_computed_tokens", -1)) != 0:
            raise GPUHBMBridgeStaleEventError(
                "full-model HBF P/D receive reservation must precede "
                "prefill execution")

        # The request is intentionally uncomputed when Router admission
        # reserves D HBM. MemoryModel.get_total_kv(request) would therefore
        # return zero. Project the exact block-rounded prompt KV that P will
        # own at its handoff instead.
        projected_tokens = int(request.prefill_target_tokens)
        full = self.validate_pd_decode_prompt_capacity(
            projected_tokens,
            decode_instance_id=decode_id,
        )
        required = full
        if required > self._available_bytes(decode.memory):
            self.metrics.pd_decode_reservation_waits += 1
            return False

        self._allocate(decode_id, required)
        reservation = GPUHBMPDDecodeReservation(
            session_id=session_id,
            request_id=request_id,
            prefill_instance_id=prefill_id,
            decode_instance_id=decode_id,
            projected_context_tokens=projected_tokens,
            full_per_rank_bytes=full,
            reserved_per_rank_bytes=required,
        )
        self._pd_decode_reservation_by_request[
            request_id] = reservation
        request.pd_decode_target_instance_id = decode_id
        request.pd_decode_full_per_rank_bytes = full
        request.pd_decode_reserved_per_rank_bytes = required
        request.pd_decode_owned_per_rank_bytes = required
        request.pd_kv_ownership_state = "hbf_decode_reserved"
        self.metrics.pd_decode_reservations += 1
        self.metrics.pd_decode_reserved_per_rank_bytes += required
        if self.validate_every_event:
            self.assert_invariants()
        return True

    def consume_pd_decode_reservation(
            self, request: object) -> GPUHBMPDDecodeReservation:
        """Transfer one bridge reservation to the native D Scheduler."""

        request_id = _integer(
            "request.id", getattr(request, "id", None))
        reservation = self._pd_decode_reservation_by_request.get(
            request_id)
        if reservation is None:
            raise GPUHBMBridgeStaleEventError(
                f"request {request_id} has no P/D decode reservation")
        del self._pd_decode_reservation_by_request[request_id]
        self.metrics.pd_decode_reservations_consumed += 1
        self.metrics.pd_decode_transferred_to_scheduler_per_rank_bytes += (
            reservation.reserved_per_rank_bytes)
        if self.validate_every_event:
            self.assert_invariants()
        return reservation

    def pd_decode_reservation(
            self, request_or_id: object,
    ) -> GPUHBMPDDecodeReservation | None:
        request_id = (
            _integer("request_id", request_or_id)
            if isinstance(request_or_id, int)
            else _integer(
                "request.id", getattr(request_or_id, "id", None))
        )
        return self._pd_decode_reservation_by_request.get(request_id)

    def cancel_pd_decode_reservation(
            self, request_or_id: object) -> dict[str, object] | None:
        """Release an unconsumed D reservation during measurement censoring."""

        request_id = (
            _integer("request_id", request_or_id)
            if isinstance(request_or_id, int)
            else _integer(
                "request.id", getattr(request_or_id, "id", None))
        )
        reservation = self._pd_decode_reservation_by_request.get(
            request_id)
        if reservation is None:
            return None
        self._free(
            reservation.decode_instance_id,
            reservation.reserved_per_rank_bytes,
        )
        del self._pd_decode_reservation_by_request[request_id]
        if not isinstance(request_or_id, int):
            request_or_id.pd_decode_target_instance_id = None
            request_or_id.pd_decode_full_per_rank_bytes = 0
            request_or_id.pd_decode_reserved_per_rank_bytes = 0
            request_or_id.pd_decode_owned_per_rank_bytes = 0
            request_or_id.pd_kv_ownership_state = "censored"
        self.metrics.pd_decode_reservations_cancelled += 1
        self.metrics.pd_decode_cancelled_per_rank_bytes += (
            reservation.reserved_per_rank_bytes)
        if self.validate_every_event:
            self.assert_invariants()
        return reservation.as_dict()

    def assert_invariants(self) -> None:
        idle_sessions = set(self._idle_by_session)
        claim_sessions = {
            claim.session_id
            for claim in self._pending_claim_by_request.values()
        }
        if idle_sessions & claim_sessions:
            raise AssertionError(
                "one session has both idle and pending-claim ownership")
        if (
            set(self._pending_claim_by_request)
            & self._adopted_request_ids
        ):
            raise AssertionError(
                "an adopted request still has bridge claim ownership")
        if (
            set(self._pd_recompute_by_request)
            & self._bound_pd_recompute_request_ids
        ):
            raise AssertionError(
                "a bound P/D recompute still has pending metadata")
        if (
            set(self._pd_decode_reservation_by_request)
            & self._adopted_request_ids
        ):
            raise AssertionError(
                "P/D decode reservation overlaps colocated adoption")

        bridge_owned_by_instance = {
            instance_id: 0 for instance_id in self.schedulers
        }
        for idle in self._idle_by_session.values():
            if idle.per_rank_bytes <= 0:
                raise AssertionError("idle GPU HBM allocation is empty")
            bridge_owned_by_instance[idle.gpu_instance_id] += (
                idle.per_rank_bytes)
        for request_id, claim in self._pending_claim_by_request.items():
            if request_id != claim.request_id:
                raise AssertionError("pending claim key differs from request")
            if claim.per_rank_bytes <= 0:
                raise AssertionError("pending GPU HBM claim is empty")
            bridge_owned_by_instance[claim.gpu_instance_id] += (
                claim.per_rank_bytes)
        for request_id, reservation in (
                self._pd_decode_reservation_by_request.items()):
            if request_id != reservation.request_id:
                raise AssertionError(
                    "P/D decode reservation key differs from request")
            if reservation.reserved_per_rank_bytes < 0:
                raise AssertionError(
                    "P/D decode reservation cannot own negative HBM")
            bridge_owned_by_instance[
                reservation.decode_instance_id] += (
                    reservation.reserved_per_rank_bytes)
        for instance_id, owned_bytes in bridge_owned_by_instance.items():
            if owned_bytes > self._dynamic_used_bytes(
                    self.memories[instance_id]):
                raise AssertionError(
                    "bridge ownership exceeds Scheduler MemoryModel usage: "
                    f"instance={instance_id}, bridge={owned_bytes}, "
                    f"dynamic={self._dynamic_used_bytes(self.memories[instance_id])}")

    def report(self) -> dict[str, object]:
        bridge_owned_by_instance = {
            instance_id: 0 for instance_id in self.schedulers
        }
        for idle in self._idle_by_session.values():
            bridge_owned_by_instance[idle.gpu_instance_id] += (
                idle.per_rank_bytes)
        for claim in self._pending_claim_by_request.values():
            bridge_owned_by_instance[claim.gpu_instance_id] += (
                claim.per_rank_bytes)
        for reservation in (
                self._pd_decode_reservation_by_request.values()):
            bridge_owned_by_instance[
                reservation.decode_instance_id] += (
                    reservation.reserved_per_rank_bytes)
        return {
            "schema": GPU_HBM_BRIDGE_SCHEMA,
            "topology": self.topology,
            "fallback_reuse_mode": self.fallback_reuse_mode,
            "adapter_contract": (
                None if self.adapter_contract is None
                else dict(self.adapter_contract)),
            "pd_pairs": [list(pair) for pair in self.pd_pairs],
            "metrics": asdict(self.metrics),
            "idle_allocations": [
                self._idle_by_session[session_id].as_dict()
                for session_id in sorted(self._idle_by_session)
            ],
            "pending_colocated_claims": [
                self._pending_claim_by_request[request_id].as_dict()
                for request_id in sorted(self._pending_claim_by_request)
            ],
            "pending_pd_recompute_bindings": [
                self._pd_recompute_by_request[request_id].as_dict()
                for request_id in sorted(self._pd_recompute_by_request)
            ],
            "pending_pd_decode_reservations": [
                self._pd_decode_reservation_by_request[
                    request_id].as_dict()
                for request_id in sorted(
                    self._pd_decode_reservation_by_request)
            ],
            "adopted_colocated_request_ids": sorted(
                self._adopted_request_ids),
            "bound_pd_recompute_request_ids": sorted(
                self._bound_pd_recompute_request_ids),
            "memory_by_instance": {
                instance_id: {
                    "npu_used_per_rank_bytes": int(memory.npu_used),
                    "npu_weight_per_rank_bytes": int(memory.weight),
                    "npu_allocatable_per_rank_bytes": int(
                        memory.npu_allocatable_mem),
                    "dynamic_used_per_rank_bytes": (
                        self._dynamic_used_bytes(memory)),
                    "bridge_owned_per_rank_bytes": (
                        bridge_owned_by_instance[instance_id]),
                    "available_per_rank_bytes": (
                        self._available_bytes(memory)),
                }
                for instance_id, memory in sorted(self.memories.items())
            },
            "history": list(self._history),
        }


__all__ = [
    "FullModelHBFGPUHBMBridge",
    "GPUHBMBridgeCapacityError",
    "GPUHBMBridgeError",
    "GPUHBMBridgeMetrics",
    "GPUHBMBridgeStaleEventError",
    "GPUHBMBridgeUnderflowError",
    "GPUHBMBridgeUnsupportedReuseError",
    "GPUHBMIdleAllocation",
    "GPUHBMPDRecomputeBinding",
    "GPUHBMPDDecodeReservation",
    "GPUHBMPendingClaim",
    "GPU_HBM_BRIDGE_SCHEMA",
]

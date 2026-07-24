"""Generic ownership and callback routing for ASTRA HBF DAG jobs.

The ASTRA ``hbf-background-v1`` protocol has one global job-id namespace,
while a serving process may have several independent producers: WakeKV
flushes, full-model HBF batches, and HBF lifecycle transfers.  Each producer
may use its own job-id namespace, so the multiplexer assigns a globally unique
Controller-facing alias and retains the producer's original owner job ID for
the completion callback.

Sources register three explicit callbacks:

* ``drain`` returns newly issued jobs;
* ``complete`` applies one strict ASTRA completion;
* ``has_pending`` reports whether the source still owns live work.

A drained job may be either a mapping with exactly ``job_id``, ``arrival_ns``
and ``stages`` fields, or an object exposing ``job_id`` and
``controller_arguments()``.  ``Controller.hbf_background_command`` is the
final schema gate.  Destructive drains are claimed one dispatch at a time.
Malformed dispatches are quarantined, and valid jobs processed before a later
failure remain in a ready outbox for the next call to :meth:`drain_jobs`.

Source drain callbacks must return every job whose ownership they transfer and
must not raise after silently transferring an unreturned job.  No generic
multiplexer can recover an object that a producer never exposes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

from .controller import Controller


MULTIPLEXER_SCHEMA = "hbf-astra-multiplexer-v1"
_SOURCE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")
_JOB_MAPPING_FIELDS = frozenset({
    "job_id",
    "arrival_ns",
    "stages",
})
_CALLBACK_FIELDS = frozenset({
    "job_id",
    "arrival_ns",
    "completion_ns",
    "stage_count",
})


class HBFAstraMultiplexerError(RuntimeError):
    """Raised when HBF job ownership or callback metadata is inconsistent."""


class HBFAstraDrainError(HBFAstraMultiplexerError):
    """Raised after a drain safely retains every dispatch it received."""

    def __init__(
            self, failures: Iterable[Mapping[str, object]], *,
            ready_job_ids: Iterable[str] = ()) -> None:
        self.failures = tuple(dict(failure) for failure in failures)
        if not self.failures:
            raise ValueError("drain error requires at least one failure")
        self.ready_job_ids = tuple(ready_job_ids)
        self.quarantine_ids = tuple(
            str(failure["quarantine_id"])
            for failure in self.failures
            if failure.get("quarantine_id") is not None
        )
        details = "; ".join(
            f"{failure['source_name']}:{failure['phase']}: "
            f"{failure['error_type']}: {failure['error']}"
            for failure in self.failures
        )
        ready = (
            f"; recoverable ready jobs={list(self.ready_job_ids)}"
            if self.ready_job_ids else ""
        )
        quarantined = (
            f"; quarantines={list(self.quarantine_ids)}"
            if self.quarantine_ids else ""
        )
        super().__init__(
            "HBF ASTRA drain retained one or more failures: "
            f"{details}{ready}{quarantined}")


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(name: str, value: object) -> int:
    value = _nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _source_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not _SOURCE_IDENTIFIER.fullmatch(value)
    ):
        raise ValueError(
            "source name must contain only letters, digits, '.', '_' or '-'")
    return value


def _job_identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or not _SOURCE_IDENTIFIER.fullmatch(value)
    ):
        raise ValueError(
            "owner job_id must contain only letters, digits, '.', '_' or '-'")
    return value


def _materialize_stages(value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(
            "HBF job stages must be an iterable of stage descriptors")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(
            "HBF job stages must be an iterable of stage descriptors"
        ) from exc


@dataclass(frozen=True)
class HBFAstraMultiplexedJob:
    """One normalized, controller-validated job and its immutable audit."""

    source_name: str
    job_id: str
    owner_job_id: str
    arrival_ns: int
    stage_count: int
    controller_command: str
    descriptor_json: str
    descriptor_sha256: str

    def controller_arguments(
            self,
    ) -> tuple[str, int, tuple[dict[str, Any], ...]]:
        """Return a fresh descriptor copy suitable for Controller encoding."""

        descriptor = json.loads(self.descriptor_json)
        return (
            self.job_id,
            self.arrival_ns,
            tuple(descriptor["stages"]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "source_name": self.source_name,
            "job_id": self.job_id,
            "owner_job_id": self.owner_job_id,
            "arrival_ns": self.arrival_ns,
            "stage_count": self.stage_count,
            "descriptor_sha256": self.descriptor_sha256,
        }


@dataclass(frozen=True)
class HBFAstraMultiplexedCompletion:
    """A successful strict callback plus the source callback's result."""

    source_name: str
    job_id: str
    owner_job_id: str
    arrival_ns: int
    completion_ns: int
    stage_count: int
    owner_result: object = field(compare=False, repr=False)

    @property
    def elapsed_ns(self) -> int:
        return self.completion_ns - self.arrival_ns

    def as_dict(self) -> dict[str, object]:
        return {
            "source_name": self.source_name,
            "job_id": self.job_id,
            "owner_job_id": self.owner_job_id,
            "arrival_ns": self.arrival_ns,
            "completion_ns": self.completion_ns,
            "elapsed_ns": self.elapsed_ns,
            "stage_count": self.stage_count,
        }


@dataclass
class _RegisteredSource:
    name: str
    drain: Callable[[], Iterable[object]]
    complete: Callable[..., object]
    has_pending: Callable[[], bool]
    drained_job_ids: set[str] = field(default_factory=set)
    completed_job_ids: set[str] = field(default_factory=set)
    drained_owner_job_ids: set[str] = field(default_factory=set)
    completed_owner_job_ids: set[str] = field(default_factory=set)
    active_owner_aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class _QuarantinedDispatch:
    quarantine_id: str
    source_name: str
    proposed_job_id: str
    owner_job_id: str | None
    dispatch: object = field(repr=False)
    error_type: str = ""
    error_message: str = ""
    attempts: int = 1

    def record_failure(self, dispatch: object, error: BaseException) -> None:
        self.dispatch = dispatch
        self.error_type = type(error).__name__
        self.error_message = str(error)
        self.attempts += 1

    def as_dict(self) -> dict[str, object]:
        return {
            "quarantine_id": self.quarantine_id,
            "source_name": self.source_name,
            "proposed_job_id": self.proposed_job_id,
            "owner_job_id": self.owner_job_id,
            "error_type": self.error_type,
            "error": self.error_message,
            "attempts": self.attempts,
        }


class HBFAstraJobMultiplexer:
    """Multiplex independent HBF producers onto one ASTRA job namespace."""

    def __init__(self) -> None:
        self._sources: dict[str, _RegisteredSource] = {}
        self._pending: dict[str, HBFAstraMultiplexedJob] = {}
        self._completed: dict[
            str, HBFAstraMultiplexedCompletion] = {}
        self._ready_job_ids: list[str] = []
        self._issued_job_ids: set[str] = set()
        self._quarantined: dict[str, _QuarantinedDispatch] = {}
        self._owner_history: dict[
            str, list[tuple[str, str]]] = {}
        self._next_job_alias = 1
        self._next_quarantine_id = 1

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(self._sources)

    @property
    def pending_job_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._pending))

    @property
    def completed_job_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._completed))

    @property
    def ready_job_ids(self) -> tuple[str, ...]:
        return tuple(self._ready_job_ids)

    @property
    def quarantined_dispatch_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._quarantined))

    def register_source(
            self, name: str, *, drain: Callable[[], Iterable[object]],
            complete: Callable[..., object],
            has_pending: Callable[[], bool]) -> None:
        """Register one explicit source callback set.

        The completion callback is invoked with keyword arguments
        ``job_id``, ``arrival_ns``, ``completion_ns`` and ``stage_count``.
        """

        source = _source_name(name)
        if source in self._sources:
            raise ValueError(f"duplicate HBF ASTRA source {source!r}")
        for callback_name, callback in (
            ("drain", drain),
            ("complete", complete),
            ("has_pending", has_pending),
        ):
            if not callable(callback):
                raise TypeError(
                    f"source {source!r} {callback_name} must be callable")
        self._sources[source] = _RegisteredSource(
            name=source,
            drain=drain,
            complete=complete,
            has_pending=has_pending,
        )

    def register_object(
            self, name: str, owner: object, *,
            drain_method: str, complete_method: str,
            has_pending_method: str,
            complete_kwargs: Mapping[str, object] | None = None) -> None:
        """Register named methods on an arbitrary producer object.

        This duck-typed adapter directly supports:

        * full-model pool: ``drain_external_dispatches``,
          ``complete_external_dispatch``,
          ``has_pending_external_dispatches``;
        * full-model lifecycle: ``drain_external_dispatches``,
          ``complete_external_dispatch``, ``has_pending_external``;
        * WakeKV: ``drain_hbf_background_jobs``,
          ``complete_hbf_background_job``,
          ``has_pending_hbf_background_jobs``.

        ``complete_kwargs={"defer_schedule": True}`` may be used for the
        full-model pool when a caller wants to route all co-timed arrivals
        before starting its next batch.
        """

        if owner is None:
            raise TypeError("source owner must not be None")
        method_names = {
            "drain_method": drain_method,
            "complete_method": complete_method,
            "has_pending_method": has_pending_method,
        }
        callbacks = {}
        for field_name, method_name in method_names.items():
            if not isinstance(method_name, str) or not method_name:
                raise ValueError(f"{field_name} must be a non-empty string")
            callback = getattr(owner, method_name, None)
            if not callable(callback):
                raise TypeError(
                    f"source owner has no callable {method_name!r}")
            callbacks[field_name] = callback

        extra = dict(complete_kwargs or {})
        overlap = set(extra) & _CALLBACK_FIELDS
        if overlap:
            raise ValueError(
                "complete_kwargs cannot replace callback metadata: "
                f"{sorted(overlap)}")

        complete_callback = callbacks["complete_method"]

        def apply_completion(**metadata):
            return complete_callback(**metadata, **extra)

        self.register_source(
            name,
            drain=callbacks["drain_method"],
            complete=apply_completion,
            has_pending=callbacks["has_pending_method"],
        )

    @staticmethod
    def _mapping_arguments(
            dispatch: Mapping[str, object],
    ) -> tuple[object, object, object]:
        fields = set(dispatch)
        if fields != _JOB_MAPPING_FIELDS:
            raise ValueError(
                "HBF job mapping fields must be exactly "
                f"{sorted(_JOB_MAPPING_FIELDS)}; "
                f"missing={sorted(_JOB_MAPPING_FIELDS - fields)}, "
                f"unknown={sorted(fields - _JOB_MAPPING_FIELDS)}")
        return (
            dispatch["job_id"],
            dispatch["arrival_ns"],
            dispatch["stages"],
        )

    @staticmethod
    def _object_arguments(
            dispatch: object,
    ) -> tuple[object, object, object]:
        if not hasattr(dispatch, "job_id"):
            raise TypeError(
                "HBF dispatch must be a mapping or expose job_id")
        controller_arguments = getattr(
            dispatch, "controller_arguments", None)
        if not callable(controller_arguments):
            raise TypeError(
                "HBF dispatch object must expose controller_arguments()")
        raw_arguments = controller_arguments()
        if isinstance(raw_arguments, (str, bytes)):
            raise TypeError(
                "controller_arguments() must return three values")
        try:
            arguments = tuple(raw_arguments)
        except TypeError as exc:
            raise TypeError(
                "controller_arguments() must return three values"
            ) from exc
        if len(arguments) != 3:
            raise ValueError(
                "controller_arguments() must return "
                "(job_id, arrival_ns, stages)")
        if dispatch.job_id != arguments[0]:
            raise HBFAstraMultiplexerError(
                "dispatch job_id differs from controller_arguments()")
        return arguments

    @classmethod
    def _normalize(
            cls, source_name: str, controller_job_id: str,
            dispatch: object) -> HBFAstraMultiplexedJob:
        if isinstance(dispatch, Mapping):
            raw_owner_job_id, raw_arrival, raw_stages = (
                cls._mapping_arguments(dispatch))
        else:
            raw_owner_job_id, raw_arrival, raw_stages = (
                cls._object_arguments(dispatch))
        owner_job_id = _job_identifier(raw_owner_job_id)
        stages = _materialize_stages(raw_stages)

        # The source-local owner ID is deliberately not Controller-facing.
        # This call is the final schema and DAG gate for the global alias.
        command = Controller.hbf_background_command(
            controller_job_id, raw_arrival, stages)
        prefix, encoded_job_id, encoded_arrival, descriptor_json = (
            command.split("\t", 3))
        if prefix != "hbf-background":
            raise AssertionError("Controller emitted an unexpected command")
        if encoded_job_id != controller_job_id:
            raise AssertionError("Controller changed the HBF job alias")
        descriptor = json.loads(descriptor_json)
        stage_count = len(descriptor["stages"])
        if stage_count != len(stages):
            raise AssertionError("Controller changed the HBF stage count")
        return HBFAstraMultiplexedJob(
            source_name=source_name,
            job_id=encoded_job_id,
            owner_job_id=owner_job_id,
            arrival_ns=int(encoded_arrival),
            stage_count=stage_count,
            controller_command=command,
            descriptor_json=descriptor_json,
            descriptor_sha256=hashlib.sha256(
                descriptor_json.encode("utf-8")).hexdigest(),
        )

    @staticmethod
    def _best_effort_owner_job_id(dispatch: object) -> str | None:
        try:
            value = (
                dispatch.get("job_id")
                if isinstance(dispatch, Mapping)
                else getattr(dispatch, "job_id", None)
            )
        except Exception:
            return None
        return value if isinstance(value, str) else None

    def _allocate_job_alias(self, source_name: str) -> str:
        alias = f"mux.{source_name}.{self._next_job_alias}"
        self._next_job_alias += 1
        return alias

    def _claim(
            self, job: HBFAstraMultiplexedJob, *,
            ignored_quarantine_id: str | None = None) -> None:
        """Claim one normalized job without making partial mutations."""

        if job.job_id in self._pending or job.job_id in self._completed:
            raise AssertionError("multiplexer generated a duplicate alias")
        source = self._sources[job.source_name]
        active_alias = source.active_owner_aliases.get(job.owner_job_id)
        if active_alias is not None:
            raise HBFAstraMultiplexerError(
                "HBF ASTRA source re-emitted an active owner job ID: "
                f"owner_job={job.owner_job_id!r}, "
                f"active_alias={active_alias!r}, "
                f"source={job.source_name!r}")
        if job.owner_job_id in source.completed_owner_job_ids:
            raise HBFAstraMultiplexerError(
                "HBF ASTRA source reused a completed owner job ID: "
                f"owner_job={job.owner_job_id!r}, "
                f"source={job.source_name!r}")
        for quarantine_id, quarantine in self._quarantined.items():
            if quarantine_id == ignored_quarantine_id:
                continue
            if (
                quarantine.source_name == job.source_name
                and quarantine.owner_job_id == job.owner_job_id
            ):
                raise HBFAstraMultiplexerError(
                    "HBF ASTRA source re-emitted a quarantined owner job ID: "
                    f"owner_job={job.owner_job_id!r}, "
                    f"quarantine={quarantine_id!r}, "
                    f"source={job.source_name!r}")

        self._pending[job.job_id] = job
        source.active_owner_aliases[job.owner_job_id] = job.job_id
        source.drained_job_ids.add(job.job_id)
        source.drained_owner_job_ids.add(job.owner_job_id)
        self._owner_history.setdefault(
            job.owner_job_id, []).append(
                (job.source_name, job.job_id))

    def _quarantine(
            self, source_name: str, proposed_job_id: str,
            dispatch: object, error: BaseException,
    ) -> _QuarantinedDispatch:
        quarantine_id = (
            f"quarantine.{source_name}.{self._next_quarantine_id}")
        self._next_quarantine_id += 1
        quarantine = _QuarantinedDispatch(
            quarantine_id=quarantine_id,
            source_name=source_name,
            proposed_job_id=proposed_job_id,
            owner_job_id=self._best_effort_owner_job_id(dispatch),
            dispatch=dispatch,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        self._quarantined[quarantine_id] = quarantine
        return quarantine

    @staticmethod
    def _failure(
            source_name: str, phase: str, error: BaseException, *,
            quarantine_id: str | None = None,
    ) -> dict[str, object]:
        return {
            "source_name": source_name,
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "quarantine_id": quarantine_id,
        }

    def _process_dispatch(
            self, source: _RegisteredSource, dispatch: object,
            failures: list[dict[str, object]]) -> None:
        alias = self._allocate_job_alias(source.name)
        try:
            job = self._normalize(source.name, alias, dispatch)
            self._claim(job)
        except Exception as exc:
            quarantine = self._quarantine(
                source.name, alias, dispatch, exc)
            failures.append(self._failure(
                source.name, "normalize-or-claim", exc,
                quarantine_id=quarantine.quarantine_id,
            ))
            return
        self._ready_job_ids.append(job.job_id)

    def _drain_source(
            self, source: _RegisteredSource,
            failures: list[dict[str, object]]) -> None:
        try:
            raw_dispatches = source.drain()
        except Exception as exc:
            failures.append(self._failure(
                source.name, "source-drain", exc))
            return

        if isinstance(raw_dispatches, (str, bytes, Mapping)):
            error = TypeError(
                f"source {source.name!r} drain must return an iterable "
                "of jobs, not one job")
            alias = self._allocate_job_alias(source.name)
            quarantine = self._quarantine(
                source.name, alias, raw_dispatches, error)
            failures.append(self._failure(
                source.name, "drain-result", error,
                quarantine_id=quarantine.quarantine_id,
            ))
            return
        try:
            iterator = iter(raw_dispatches)
        except TypeError:
            error = TypeError(
                f"source {source.name!r} drain must return an iterable "
                "of jobs")
            alias = self._allocate_job_alias(source.name)
            quarantine = self._quarantine(
                source.name, alias, raw_dispatches, error)
            failures.append(self._failure(
                source.name, "drain-result", error,
                quarantine_id=quarantine.quarantine_id,
            ))
            return

        while True:
            try:
                dispatch = next(iterator)
            except StopIteration:
                return
            except Exception as exc:
                failures.append(self._failure(
                    source.name, "drain-iteration", exc))
                return
            self._process_dispatch(source, dispatch, failures)

    def _handoff_ready(
            self) -> tuple[HBFAstraMultiplexedJob, ...]:
        aliases = tuple(self._ready_job_ids)
        self._ready_job_ids.clear()
        self._issued_job_ids.update(aliases)
        return tuple(self._pending[alias] for alias in aliases)

    def drain_jobs(self) -> tuple[HBFAstraMultiplexedJob, ...]:
        """Drain and claim jobs without losing destructively drained work.

        Ready jobs retained by a prior partial failure are returned first,
        without invoking any source again.  If a newly drained dispatch is
        invalid, every valid peer is retained in the ready outbox and every
        invalid peer is retained in quarantine before ``HBFAstraDrainError``
        is raised.
        """

        if self._ready_job_ids:
            return self._handoff_ready()

        failures: list[dict[str, object]] = []
        for source in self._sources.values():
            self._drain_source(source, failures)
        if failures:
            raise HBFAstraDrainError(
                failures, ready_job_ids=self._ready_job_ids)
        return self._handoff_ready()

    def retry_quarantined(
            self, quarantine_id: str, *,
            dispatch: object | None = None,
    ) -> HBFAstraMultiplexedJob:
        """Retry one retained dispatch and hand a valid job to the caller.

        A replacement descriptor may repair stages or metadata, but it may
        not change a known source-owner job ID because the source has already
        transferred ownership under that original ID.
        """

        if not isinstance(quarantine_id, str) or not quarantine_id:
            raise ValueError("quarantine_id must be a non-empty string")
        quarantine = self._quarantined.get(quarantine_id)
        if quarantine is None:
            raise HBFAstraMultiplexerError(
                f"unknown HBF ASTRA quarantine {quarantine_id!r}")
        candidate = (
            quarantine.dispatch if dispatch is None else dispatch)
        candidate_owner = self._best_effort_owner_job_id(candidate)
        if (
            quarantine.owner_job_id is not None
            and candidate_owner != quarantine.owner_job_id
        ):
            raise HBFAstraMultiplexerError(
                "quarantine retry cannot change the retained owner job ID: "
                f"expected={quarantine.owner_job_id!r}, "
                f"actual={candidate_owner!r}")

        try:
            job = self._normalize(
                quarantine.source_name,
                quarantine.proposed_job_id,
                candidate,
            )
            self._claim(
                job, ignored_quarantine_id=quarantine_id)
        except Exception as exc:
            quarantine.record_failure(candidate, exc)
            failure = self._failure(
                quarantine.source_name, "quarantine-retry", exc,
                quarantine_id=quarantine_id,
            )
            raise HBFAstraDrainError(
                [failure],
                ready_job_ids=self._ready_job_ids,
            ) from exc

        del self._quarantined[quarantine_id]
        self._issued_job_ids.add(job.job_id)
        return job

    def drain_commands(self) -> tuple[str, ...]:
        """Return exact Controller commands for every newly drained job."""

        return tuple(
            job.controller_command for job in self.drain_jobs())

    def complete(
            self, *, job_id: str, arrival_ns: int,
            completion_ns: int,
            stage_count: int) -> HBFAstraMultiplexedCompletion:
        """Route one strict ASTRA callback to its retained source owner."""

        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        arrival = _nonnegative_int("arrival_ns", arrival_ns)
        completion = _nonnegative_int("completion_ns", completion_ns)
        stages = _positive_int("stage_count", stage_count)
        if job_id in self._completed:
            raise HBFAstraMultiplexerError(
                f"duplicate HBF ASTRA completion for {job_id!r}")
        pending = self._pending.get(job_id)
        if pending is None:
            raise HBFAstraMultiplexerError(
                f"unknown HBF ASTRA completion job {job_id!r}")
        if job_id not in self._issued_job_ids:
            raise HBFAstraMultiplexerError(
                "HBF ASTRA completion arrived before Controller handoff: "
                f"job={job_id!r}")
        if arrival != pending.arrival_ns:
            raise HBFAstraMultiplexerError(
                "HBF ASTRA callback arrival metadata drift: "
                f"job={job_id!r}, expected={pending.arrival_ns}, "
                f"actual={arrival}")
        if stages != pending.stage_count:
            raise HBFAstraMultiplexerError(
                "HBF ASTRA callback stage-count metadata drift: "
                f"job={job_id!r}, expected={pending.stage_count}, "
                f"actual={stages}")
        if completion < arrival:
            raise HBFAstraMultiplexerError(
                f"HBF ASTRA job {job_id!r} completed before arrival")

        source = self._sources[pending.source_name]
        active_alias = source.active_owner_aliases.get(
            pending.owner_job_id)
        if active_alias != job_id:
            raise AssertionError(
                "source owner alias changed before completion")
        owner_result = source.complete(
            job_id=pending.owner_job_id,
            arrival_ns=arrival,
            completion_ns=completion,
            stage_count=stages,
        )
        result = HBFAstraMultiplexedCompletion(
            source_name=pending.source_name,
            job_id=job_id,
            owner_job_id=pending.owner_job_id,
            arrival_ns=arrival,
            completion_ns=completion,
            stage_count=stages,
            owner_result=owner_result,
        )
        del self._pending[job_id]
        del source.active_owner_aliases[pending.owner_job_id]
        self._issued_job_ids.remove(job_id)
        self._completed[job_id] = result
        source.completed_job_ids.add(job_id)
        source.completed_owner_job_ids.add(pending.owner_job_id)
        return result

    @staticmethod
    def _callback_pending(source: _RegisteredSource) -> bool:
        value = source.has_pending()
        if not isinstance(value, bool):
            raise TypeError(
                f"source {source.name!r} has_pending must return a boolean")
        return value

    def has_pending(self) -> bool:
        """Return whether the mux or any registered source owns live work."""

        if self._pending or self._quarantined:
            return True
        return any(
            self._callback_pending(source)
            for source in self._sources.values()
        )

    def pending_audit(self) -> tuple[dict[str, object], ...]:
        rows = []
        ready = set(self._ready_job_ids)
        for job_id in sorted(self._pending):
            row = self._pending[job_id].as_dict()
            row["handoff_state"] = (
                "ready" if job_id in ready else "issued")
            rows.append(row)
        return tuple(rows)

    def quarantine_audit(self) -> tuple[dict[str, object], ...]:
        return tuple(
            self._quarantined[quarantine_id].as_dict()
            for quarantine_id in sorted(self._quarantined)
        )

    def owner_job_id_collisions(
            self) -> tuple[dict[str, object], ...]:
        """Return harmless cross-source owner-ID collisions for audit."""

        collisions = []
        for owner_job_id in sorted(self._owner_history):
            jobs = self._owner_history[owner_job_id]
            if len({source_name for source_name, _ in jobs}) < 2:
                continue
            collisions.append({
                "owner_job_id": owner_job_id,
                "jobs": [
                    {
                        "source_name": source_name,
                        "job_id": job_id,
                    }
                    for source_name, job_id in jobs
                ],
            })
        return tuple(collisions)

    def source_audit(self) -> dict[str, dict[str, object]]:
        result = {}
        for name, source in self._sources.items():
            mux_pending = sorted(
                job_id for job_id, job in self._pending.items()
                if job.source_name == name
            )
            mux_pending_owners = sorted(
                job.owner_job_id
                for job in self._pending.values()
                if job.source_name == name
            )
            quarantines = sorted(
                quarantine_id
                for quarantine_id, quarantine in self._quarantined.items()
                if quarantine.source_name == name
            )
            result[name] = {
                "callback_has_pending": self._callback_pending(source),
                "mux_pending_job_ids": mux_pending,
                "mux_pending_owner_job_ids": mux_pending_owners,
                "mux_ready_job_ids": sorted(
                    set(mux_pending) & set(self._ready_job_ids)),
                "quarantined_dispatch_ids": quarantines,
                "drained_job_count": len(source.drained_job_ids),
                "drained_job_ids": sorted(source.drained_job_ids),
                "drained_owner_job_ids": sorted(
                    source.drained_owner_job_ids),
                "completed_job_count": len(source.completed_job_ids),
                "completed_job_ids": sorted(source.completed_job_ids),
                "completed_owner_job_ids": sorted(
                    source.completed_owner_job_ids),
            }
        return result

    def report(self) -> dict[str, object]:
        sources = self.source_audit()
        return {
            "schema": MULTIPLEXER_SCHEMA,
            "registered_sources": list(self._sources),
            "pending_job_count": len(self._pending),
            "pending_jobs": list(self.pending_audit()),
            "ready_job_count": len(self._ready_job_ids),
            "ready_job_ids": list(self._ready_job_ids),
            "quarantined_dispatch_count": len(self._quarantined),
            "quarantined_dispatches": list(
                self.quarantine_audit()),
            "completed_job_count": len(self._completed),
            "completed_jobs": [
                self._completed[job_id].as_dict()
                for job_id in sorted(self._completed)
            ],
            "owner_job_id_collisions": list(
                self.owner_job_id_collisions()),
            "source_audit": sources,
            "has_pending": bool(
                self._pending
                or self._quarantined
                or any(
                    row["callback_has_pending"]
                    for row in sources.values()
                )
            ),
        }


__all__ = [
    "HBFAstraDrainError",
    "HBFAstraJobMultiplexer",
    "HBFAstraMultiplexedCompletion",
    "HBFAstraMultiplexedJob",
    "HBFAstraMultiplexerError",
    "MULTIPLEXER_SCHEMA",
]

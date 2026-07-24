"""Reusable resumable cutoffs for analytical comparison event loops.

The comparison systems advance only at discrete event timestamps.  A cutoff
therefore stops before the first event outside the requested boundary; it
does not advance clocks artificially, cancel work, or turn a partial run into
a completed run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class PartialRunAudit:
    """Immutable ownership partition captured at a cutoff boundary.

    ``unreleased_request_ids``, ``released_live_request_ids``, and
    ``user_completed_request_ids`` are an exact disjoint partition of every
    frozen scheduled request ID.  Internal completion is a separate partition
    of released requests because user completion can precede a P-to-D handoff,
    tier commit, or migration cleanup.
    """

    cutoff_ns: int
    inclusive: bool
    current_ns: int
    last_processed_event_ns: Optional[int]
    next_event_ns: Optional[int]
    scheduled_request_ids: tuple[int, ...]
    unreleased_request_ids: tuple[int, ...]
    released_live_request_ids: tuple[int, ...]
    user_completed_request_ids: tuple[int, ...]
    internal_work_request_ids: tuple[int, ...]
    internal_complete_request_ids: tuple[int, ...]
    user_completed_internal_work_request_ids: tuple[int, ...]
    system_finished: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.cutoff_ns, bool)
            or not isinstance(self.cutoff_ns, int)
            or self.cutoff_ns < 0
        ):
            raise ValueError("cutoff_ns must be a non-negative integer")
        if not isinstance(self.inclusive, bool):
            raise ValueError("inclusive must be a boolean")
        if self.current_ns < 0 or self.current_ns > self.cutoff_ns:
            raise ValueError(
                "current_ns must lie within the cutoff boundary")
        if self.last_processed_event_ns is not None:
            if self.last_processed_event_ns != self.current_ns:
                raise ValueError(
                    "last processed event must equal the event-loop clock")
            if self.inclusive:
                valid_boundary = (
                    self.last_processed_event_ns <= self.cutoff_ns)
            else:
                valid_boundary = (
                    self.last_processed_event_ns < self.cutoff_ns)
            if not valid_boundary:
                raise ValueError(
                    "last processed event violates cutoff inclusivity")
        if (
            self.next_event_ns is not None
            and self.next_event_ns < self.current_ns
        ):
            raise ValueError(
                "next event cannot precede the event-loop clock")

        tuple_fields = (
            self.scheduled_request_ids,
            self.unreleased_request_ids,
            self.released_live_request_ids,
            self.user_completed_request_ids,
            self.internal_work_request_ids,
            self.internal_complete_request_ids,
            self.user_completed_internal_work_request_ids,
        )
        for values in tuple_fields:
            if values != tuple(sorted(set(values))):
                raise ValueError(
                    "audit request-ID tuples must be sorted and unique")

        scheduled = set(self.scheduled_request_ids)
        unreleased = set(self.unreleased_request_ids)
        released_live = set(self.released_live_request_ids)
        user_completed = set(self.user_completed_request_ids)
        user_partition = (unreleased, released_live, user_completed)
        if any(
            left & right
            for index, left in enumerate(user_partition)
            for right in user_partition[index + 1:]
        ):
            raise ValueError(
                "user-state request-ID partition is not disjoint")
        if set().union(*user_partition) != scheduled:
            raise ValueError(
                "user-state request-ID partition is not exhaustive")

        released = released_live | user_completed
        internal_work = set(self.internal_work_request_ids)
        internal_complete = set(self.internal_complete_request_ids)
        if internal_work & internal_complete:
            raise ValueError(
                "internal-state request-ID partition is not disjoint")
        if internal_work | internal_complete != released:
            raise ValueError(
                "internal-state request-ID partition is not exhaustive")
        expected_user_internal = user_completed & internal_work
        if set(self.user_completed_internal_work_request_ids) != (
                expected_user_internal):
            raise ValueError(
                "user-completed internal-work subset is incorrect")

    @property
    def scheduled_count(self) -> int:
        return len(self.scheduled_request_ids)

    @property
    def unreleased_count(self) -> int:
        return len(self.unreleased_request_ids)

    @property
    def released_live_count(self) -> int:
        return len(self.released_live_request_ids)

    @property
    def user_completed_count(self) -> int:
        return len(self.user_completed_request_ids)

    @property
    def internal_work_count(self) -> int:
        return len(self.internal_work_request_ids)


class ResumableCutoffEventLoopMixin:
    """Common cutoff runner for comparison systems.

    Implementations provide ``load``, ``_next_event_ns``,
    ``_process_timestamp``, and ``assert_invariants`` together with the state
    attributes used by the existing full-drain runners.
    """

    @staticmethod
    def _validate_cutoff(
            cutoff_ns: int, inclusive: bool) -> None:
        if (
            isinstance(cutoff_ns, bool)
            or not isinstance(cutoff_ns, int)
            or cutoff_ns < 0
        ):
            raise ValueError(
                "cutoff_ns must be a non-negative integer")
        if not isinstance(inclusive, bool):
            raise ValueError("inclusive must be a boolean")

    def _partial_run_audit(
            self, *, cutoff_ns: int,
            inclusive: bool) -> PartialRunAudit:
        scheduled = set(self._spec_by_request)
        released = set(self._released_ids)
        completed = set(self._completed_ids)
        runtime = set(self._runtime_calls)
        if runtime != released:
            raise AssertionError(
                "runtime-call ownership does not match released IDs")
        if not completed <= released:
            raise AssertionError(
                "completed request was not released")

        internal_complete = set()
        for request_id, call in self._runtime_calls.items():
            state = getattr(call, "state", None)
            state_value = getattr(state, "value", state)
            if state_value == "internal_complete":
                internal_complete.add(request_id)
        internal_work = released - internal_complete
        released_live = released - completed
        event_count = int(self.metrics.event_timestamps)
        last_processed_event_ns = (
            int(self.current_ns) if event_count else None)

        return PartialRunAudit(
            cutoff_ns=cutoff_ns,
            inclusive=inclusive,
            current_ns=int(self.current_ns),
            last_processed_event_ns=last_processed_event_ns,
            next_event_ns=self._next_event_ns(),
            scheduled_request_ids=tuple(sorted(scheduled)),
            unreleased_request_ids=tuple(sorted(
                scheduled - released)),
            released_live_request_ids=tuple(sorted(released_live)),
            user_completed_request_ids=tuple(sorted(completed)),
            internal_work_request_ids=tuple(sorted(internal_work)),
            internal_complete_request_ids=tuple(sorted(
                internal_complete)),
            user_completed_internal_work_request_ids=tuple(sorted(
                completed & internal_work)),
            system_finished=bool(self._finished),
        )

    def run_until(
            self, cutoff_ns: int, inclusive: bool = True, *,
            scheduled_sessions: Optional[Iterable[Any]] = None,
    ) -> PartialRunAudit:
        """Process all events inside ``cutoff_ns`` and return a state audit.

        The method is resumable.  It never marks the system finished and never
        drains or cancels work beyond the boundary.  A later ``run_until`` or
        the existing full ``run`` continues from the exact retained state.
        """

        self._validate_cutoff(cutoff_ns, inclusive)
        if scheduled_sessions is not None:
            self.load(scheduled_sessions)
        if not self._loaded:
            raise RuntimeError("load a schedule before running")
        if self._running:
            raise RuntimeError(
                "comparison system is already running")
        if cutoff_ns < self.current_ns:
            raise ValueError(
                "cutoff_ns cannot precede current event-loop time")
        if (
            not inclusive
            and self.metrics.event_timestamps
            and cutoff_ns <= self.current_ns
        ):
            raise ValueError(
                "exclusive cutoff cannot equal an already processed "
                "event timestamp")

        self._running = True
        try:
            while True:
                event_ns = self._next_event_ns()
                if event_ns is None:
                    break
                inside = (
                    event_ns <= cutoff_ns
                    if inclusive
                    else event_ns < cutoff_ns
                )
                if not inside:
                    break
                self._process_timestamp(event_ns)
        finally:
            self._running = False

        # Even compact sweep mode validates once at the public boundary.
        self.assert_invariants()
        return self._partial_run_audit(
            cutoff_ns=cutoff_ns,
            inclusive=inclusive,
        )


__all__ = [
    "PartialRunAudit",
    "ResumableCutoffEventLoopMixin",
]

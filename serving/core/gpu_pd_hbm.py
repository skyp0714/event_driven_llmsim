"""Atomic per-rank HBM reservations for one finite P4D4 node."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

from .gpu_pd_latency import P4D4GPUHardware


@dataclass(frozen=True)
class PDHBMAdmission:
    admission_id: int
    session_id: str
    input_tokens: int
    output_tokens: int
    final_tokens: int
    p_bytes_per_rank: int
    d_target_bytes_per_rank: int
    d_held_bytes_per_rank: int
    prior_d_bytes_per_rank: int
    needs_d: bool


@dataclass
class PDHBMMetrics:
    admission_attempts: int = 0
    admissions: int = 0
    capacity_deferrals: int = 0
    infeasible_requests: int = 0
    cancellations: int = 0
    p_releases: int = 0
    d_source_releases: int = 0
    terminal_d_releases: int = 0
    peak_p_bytes_per_rank: int = 0
    peak_d_bytes_per_rank: int = 0


class AtomicPDHBM:
    """Reserve P input and D final-context capacity as one transaction."""

    def __init__(
            self, *, hardware: P4D4GPUHardware,
            node_id: int,
            p_capacity_bytes_per_rank: Optional[int] = None,
            d_capacity_bytes_per_rank: Optional[int] = None) -> None:
        hardware.validate()
        if (
            isinstance(node_id, bool)
            or not isinstance(node_id, int)
            or node_id < 0
        ):
            raise ValueError("node_id must be a non-negative integer")
        default_capacity = hardware.usable_hbm_bytes_per_rank
        self.p_capacity_bytes_per_rank = (
            default_capacity
            if p_capacity_bytes_per_rank is None
            else p_capacity_bytes_per_rank
        )
        self.d_capacity_bytes_per_rank = (
            default_capacity
            if d_capacity_bytes_per_rank is None
            else d_capacity_bytes_per_rank
        )
        for name, value in (
            ("p_capacity_bytes_per_rank",
             self.p_capacity_bytes_per_rank),
            ("d_capacity_bytes_per_rank",
             self.d_capacity_bytes_per_rank),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        self.hardware = hardware
        self.node_id = node_id
        self.metrics = PDHBMMetrics()
        self._p_by_session: dict[str, int] = {}
        self._d_by_session: dict[str, int] = {}
        self._active_by_id: dict[int, PDHBMAdmission] = {}
        self._active_id_by_session: dict[str, int] = {}
        self._next_admission_id = 1

    @property
    def p_used_bytes_per_rank(self) -> int:
        return sum(self._p_by_session.values())

    @property
    def d_used_bytes_per_rank(self) -> int:
        return sum(self._d_by_session.values())

    def p_bytes(self, session_id: str) -> int:
        return self._p_by_session.get(session_id, 0)

    def d_bytes(self, session_id: str) -> int:
        return self._d_by_session.get(session_id, 0)

    def active_admission(
            self, session_id: str) -> Optional[PDHBMAdmission]:
        admission_id = self._active_id_by_session.get(session_id)
        return (
            None if admission_id is None
            else self._active_by_id[admission_id]
        )

    def _update_peaks(self) -> None:
        self.metrics.peak_p_bytes_per_rank = max(
            self.metrics.peak_p_bytes_per_rank,
            self.p_used_bytes_per_rank,
        )
        self.metrics.peak_d_bytes_per_rank = max(
            self.metrics.peak_d_bytes_per_rank,
            self.d_used_bytes_per_rank,
        )

    @staticmethod
    def _validate_session(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")

    def try_admit(
            self, *, session_id: str, input_tokens: int,
            output_tokens: int, needs_d: bool = True,
    ) -> Optional[PDHBMAdmission]:
        """Return an admission or ``None`` without partial mutation."""

        self._validate_session(session_id)
        for name, value in (
            ("input_tokens", input_tokens),
            ("output_tokens", output_tokens),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(needs_d, bool):
            raise ValueError("needs_d must be a boolean")
        final_tokens = input_tokens + output_tokens - 1
        if final_tokens > 1_010_000:
            raise ValueError(
                "request output would exceed 1,010,000-token contract")
        if session_id in self._active_id_by_session:
            raise RuntimeError(
                f"session {session_id!r} already has an active admission")

        self.metrics.admission_attempts += 1
        p_bytes = self.hardware.kv_capacity_bytes_per_rank(input_tokens)
        d_target = (
            self.hardware.kv_capacity_bytes_per_rank(final_tokens)
            if needs_d else 0
        )
        prior_d = self.d_bytes(session_id)
        d_held = max(prior_d, d_target)
        if (
            p_bytes > self.p_capacity_bytes_per_rank
            or d_target > self.d_capacity_bytes_per_rank
        ):
            self.metrics.infeasible_requests += 1
            raise RuntimeError(
                "request is individually infeasible for P/D HBM: "
                f"p={p_bytes}/{self.p_capacity_bytes_per_rank}, "
                f"d={d_target}/{self.d_capacity_bytes_per_rank}")

        projected_p = self.p_used_bytes_per_rank + p_bytes
        projected_d = (
            self.d_used_bytes_per_rank - prior_d + d_held)
        if (
            projected_p > self.p_capacity_bytes_per_rank
            or projected_d > self.d_capacity_bytes_per_rank
        ):
            self.metrics.capacity_deferrals += 1
            return None

        admission = PDHBMAdmission(
            admission_id=self._next_admission_id,
            session_id=session_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            final_tokens=final_tokens,
            p_bytes_per_rank=p_bytes,
            d_target_bytes_per_rank=d_target,
            d_held_bytes_per_rank=d_held,
            prior_d_bytes_per_rank=prior_d,
            needs_d=needs_d,
        )
        self._next_admission_id += 1
        self._p_by_session[session_id] = p_bytes
        if d_held:
            self._d_by_session[session_id] = d_held
        else:
            self._d_by_session.pop(session_id, None)
        self._active_by_id[admission.admission_id] = admission
        self._active_id_by_session[session_id] = admission.admission_id
        self.metrics.admissions += 1
        self._update_peaks()
        self.assert_invariants()
        return admission

    def _require_active(
            self, admission: PDHBMAdmission) -> PDHBMAdmission:
        if not isinstance(admission, PDHBMAdmission):
            raise ValueError("admission must be a PDHBMAdmission")
        active = self._active_by_id.get(admission.admission_id)
        if active != admission:
            raise RuntimeError("stale or inactive P/D HBM admission")
        return active

    def release_p(self, admission: PDHBMAdmission) -> None:
        active = self._require_active(admission)
        if self.p_bytes(active.session_id) == 0:
            return
        self._p_by_session.pop(active.session_id)
        self.metrics.p_releases += 1
        self.assert_invariants()

    def release_d_source(self, admission: PDHBMAdmission) -> None:
        """Shrink a held old D source to the request's final target."""

        active = self._require_active(admission)
        current = self.d_bytes(active.session_id)
        if current != active.d_held_bytes_per_rank:
            raise RuntimeError(
                "D source reservation changed before source release")
        if active.d_target_bytes_per_rank:
            self._d_by_session[active.session_id] = (
                active.d_target_bytes_per_rank)
        else:
            self._d_by_session.pop(active.session_id, None)
        if active.d_held_bytes_per_rank != (
                active.d_target_bytes_per_rank):
            self.metrics.d_source_releases += 1
        self.assert_invariants()

    def cancel(self, admission: PDHBMAdmission) -> None:
        active = self._require_active(admission)
        self._p_by_session.pop(active.session_id, None)
        if active.prior_d_bytes_per_rank:
            self._d_by_session[active.session_id] = (
                active.prior_d_bytes_per_rank)
        else:
            self._d_by_session.pop(active.session_id, None)
        self._active_by_id.pop(active.admission_id)
        self._active_id_by_session.pop(active.session_id)
        self.metrics.cancellations += 1
        self.assert_invariants()

    def finish(
            self, admission: PDHBMAdmission, *,
            has_successor: bool) -> None:
        active = self._require_active(admission)
        if not isinstance(has_successor, bool):
            raise ValueError("has_successor must be a boolean")
        if self._p_by_session.pop(active.session_id, None) is not None:
            self.metrics.p_releases += 1
        if has_successor and active.d_target_bytes_per_rank:
            self._d_by_session[active.session_id] = (
                active.d_target_bytes_per_rank)
        else:
            if self.d_bytes(active.session_id):
                self.metrics.terminal_d_releases += 1
            self._d_by_session.pop(active.session_id, None)
        self._active_by_id.pop(active.admission_id)
        self._active_id_by_session.pop(active.session_id)
        self.assert_invariants()

    def release_idle_session(self, session_id: str) -> None:
        self._validate_session(session_id)
        if session_id in self._active_id_by_session:
            raise RuntimeError("cannot release an active HBM session")
        self._d_by_session.pop(session_id, None)
        self.assert_invariants()

    def assert_invariants(self) -> None:
        if not 0 <= self.p_used_bytes_per_rank <= (
                self.p_capacity_bytes_per_rank):
            raise AssertionError("P HBM capacity invariant failed")
        if not 0 <= self.d_used_bytes_per_rank <= (
                self.d_capacity_bytes_per_rank):
            raise AssertionError("D HBM capacity invariant failed")
        if set(self._active_id_by_session.values()) != set(
                self._active_by_id):
            raise AssertionError("active HBM admission index mismatch")
        for session_id, admission_id in (
                self._active_id_by_session.items()):
            admission = self._active_by_id[admission_id]
            if admission.session_id != session_id:
                raise AssertionError(
                    "active HBM admission session mismatch")
            p_bytes = self.p_bytes(session_id)
            if p_bytes not in {0, admission.p_bytes_per_rank}:
                raise AssertionError(
                    "active P reservation has unexpected size")
            d_bytes = self.d_bytes(session_id)
            if d_bytes not in {
                admission.d_held_bytes_per_rank,
                admission.d_target_bytes_per_rank,
            }:
                raise AssertionError(
                    "active D reservation has unexpected size")
        if any(value <= 0 for value in self._p_by_session.values()):
            raise AssertionError("P reservations must be positive")
        if any(value <= 0 for value in self._d_by_session.values()):
            raise AssertionError("D reservations must be positive")

    def report(self) -> Mapping[str, Any]:
        return {
            "node_id": self.node_id,
            "p_capacity_bytes_per_rank": (
                self.p_capacity_bytes_per_rank),
            "d_capacity_bytes_per_rank": (
                self.d_capacity_bytes_per_rank),
            "p_used_bytes_per_rank": self.p_used_bytes_per_rank,
            "d_used_bytes_per_rank": self.d_used_bytes_per_rank,
            "p_by_session": dict(sorted(self._p_by_session.items())),
            "d_by_session": dict(sorted(self._d_by_session.items())),
            "active_admissions": {
                admission_id: asdict(admission)
                for admission_id, admission
                in sorted(self._active_by_id.items())
            },
            "metrics": asdict(self.metrics),
        }


__all__ = [
    "AtomicPDHBM",
    "PDHBMAdmission",
    "PDHBMMetrics",
]

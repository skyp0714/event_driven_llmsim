"""Online-run measurement helpers shared by the serving entry point.

This module deliberately has no replay entry point.  It observes batches that
are executed by ``python -m serving`` and constructs a strict, nonbinding-HBM
reference configuration for that same online path.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class BatchComputeObservation:
    instance_id: int
    batch_id: int
    start_ns: int
    finish_ns: int
    iteration_service_ns: int
    total_query_tokens: int
    recompute_query_tokens: int
    attributed_recompute_iteration_service_ns: int
    pd_type: str
    real_request_count: int


class OnlineHBMOccupancyAccounting:
    """Time-integrate per-rank HBM KV ownership from online state.

    Physical occupancy and logical future reservations are deliberately kept
    separate.  A future-ready reservation can be backed by an HBM LRU victim
    whose asynchronous demotion has not committed yet.  During that interval
    the reservation overlaps physical idle bytes and therefore must not be
    added to physical occupancy.

    ``observe`` uses right-continuous snapshots: a state sampled at ``t`` owns
    the half-open interval beginning at ``t``.  Re-observing the same timestamp
    replaces that state, which lets the serving loop sample after each group of
    same-time mutations without inventing a positive-duration intermediate
    state.
    """

    SCHEMA_VERSION = 1
    _PHYSICAL_CATEGORIES = (
        "physical_idle_reusable",
        "physical_non_idle_active",
        "physical_free",
    )
    _OVERLAY_CATEGORIES = (
        "logical_destination_admission_reservation",
        "reserved_free_slack",
        "future_reclaim_backed_reservation",
        "unclaimed_allocatable_slack",
    )

    def __init__(self, schedulers):
        self.schedulers = {
            int(scheduler.instance_id): scheduler
            for scheduler in schedulers
        }
        if not self.schedulers:
            raise ValueError(
                "HBM occupancy accounting requires at least one scheduler")
        self._history = []
        self._observation_calls = 0
        self._same_timestamp_replacements = 0

    @staticmethod
    def _entry_location(entry):
        location = entry.location
        return str(getattr(location, "value", location)).lower()

    def _snapshot(self, manager):
        values = {}
        entries = tuple(manager.entries.values())
        for instance_id, scheduler in sorted(self.schedulers.items()):
            memory = scheduler.memory
            weight = int(memory.weight)
            allocatable = int(memory.npu_allocatable_mem)
            used = int(memory.npu_used)
            capacity = allocatable - weight
            physical_kv = used - weight
            idle = sum(
                int(entry.per_rank_bytes)
                for entry in entries
                if int(entry.instance_id) == instance_id
                and self._entry_location(entry) == "hbm"
            )
            logical = int(manager._hbm_logically_reserved(instance_id))
            active = physical_kv - idle
            physical_free = capacity - physical_kv
            reserved_free = min(logical, physical_free)
            future_backed = max(0, logical - physical_free)
            unclaimed = max(0, physical_free - logical)

            named = {
                "capacity_per_rank_bytes": capacity,
                "physical_idle_reusable": idle,
                "physical_non_idle_active": active,
                "physical_free": physical_free,
                "logical_destination_admission_reservation": logical,
                "reserved_free_slack": reserved_free,
                "future_reclaim_backed_reservation": future_backed,
                "unclaimed_allocatable_slack": unclaimed,
            }
            negative = {
                key: value for key, value in named.items() if value < 0
            }
            if negative:
                raise RuntimeError(
                    "HBM occupancy snapshot contains negative ownership: "
                    f"instance={instance_id}, values={named}, "
                    f"negative={negative}")
            if idle + active + physical_free != capacity:
                raise RuntimeError(
                    "Physical HBM KV capacity does not reconcile: "
                    f"instance={instance_id}, values={named}")
            if reserved_free + future_backed != logical:
                raise RuntimeError(
                    "Logical HBM reservation overlay does not reconcile: "
                    f"instance={instance_id}, values={named}")
            if idle + active + reserved_free + unclaimed != capacity:
                raise RuntimeError(
                    "Reservation-adjusted HBM capacity does not reconcile: "
                    f"instance={instance_id}, values={named}")
            if logical > capacity:
                raise RuntimeError(
                    "Logical HBM reservations exceed the per-rank KV ceiling: "
                    f"instance={instance_id}, values={named}")
            values[instance_id] = named
        return values

    def observe(self, time_ns, manager):
        """Record the latest complete ownership state at ``time_ns``."""
        time_ns = int(time_ns)
        if time_ns < 0:
            raise ValueError("HBM occupancy observation time must be non-negative")
        if self._history and time_ns < self._history[-1][0]:
            raise RuntimeError(
                "HBM occupancy observations must be monotonic: "
                f"previous={self._history[-1][0]}, current={time_ns}")
        snapshot = self._snapshot(manager)
        self._observation_calls += 1
        if self._history and time_ns == self._history[-1][0]:
            self._history[-1] = (time_ns, snapshot)
            self._same_timestamp_replacements += 1
        else:
            self._history.append((time_ns, snapshot))

    @staticmethod
    def _empty_accumulator(instance_ids, categories):
        return {
            instance_id: {
                category: {"byte_ns": 0, "peak_per_rank_bytes": 0}
                for category in categories
            }
            for instance_id in instance_ids
        }

    @staticmethod
    def _category_report(raw, duration_ns, capacity):
        average = raw["byte_ns"] / duration_ns
        return {
            "byte_ns": raw["byte_ns"],
            "average_per_rank_bytes": average,
            "peak_per_rank_bytes": raw["peak_per_rank_bytes"],
            "average_fraction_of_capacity": (
                average / capacity if capacity > 0 else None),
            "peak_fraction_of_capacity": (
                raw["peak_per_rank_bytes"] / capacity
                if capacity > 0 else None),
        }

    def summary(self, start_ns, end_ns):
        """Return exact occupancy integrals clipped to ``[start_ns, end_ns)``."""
        start_ns = int(start_ns)
        end_ns = int(end_ns)
        if end_ns <= start_ns:
            raise ValueError("HBM occupancy window must have positive duration")
        if not self._history:
            raise RuntimeError("HBM occupancy has no observations")
        first_observation_ns = self._history[0][0]
        last_observation_ns = self._history[-1][0]
        if first_observation_ns > start_ns or last_observation_ns < end_ns:
            raise RuntimeError(
                "HBM occupancy observations do not cover the requested window: "
                f"observed=[{first_observation_ns}, {last_observation_ns}], "
                f"requested=[{start_ns}, {end_ns}]")

        duration_ns = end_ns - start_ns
        instance_ids = tuple(sorted(self.schedulers))
        categories = self._PHYSICAL_CATEGORIES + self._OVERLAY_CATEGORIES
        per_instance_raw = self._empty_accumulator(instance_ids, categories)
        aggregate_raw = {
            category: {"byte_ns": 0, "peak_per_rank_bytes": 0}
            for category in categories
        }
        contributing_intervals = 0
        physical_occupied_peak = 0
        reservation_adjusted_claim_peak = 0
        per_instance_physical_occupied_peak = {
            instance_id: 0 for instance_id in instance_ids
        }
        per_instance_reservation_adjusted_claim_peak = {
            instance_id: 0 for instance_id in instance_ids
        }

        for index, (state_start, snapshot) in enumerate(self._history[:-1]):
            state_end = self._history[index + 1][0]
            clipped_start = max(start_ns, state_start)
            clipped_end = min(end_ns, state_end)
            if clipped_end <= clipped_start:
                continue
            interval_ns = clipped_end - clipped_start
            contributing_intervals += 1
            aggregate_values = defaultdict(int)
            for instance_id in instance_ids:
                state = snapshot[instance_id]
                for category in categories:
                    value = int(state[category])
                    target = per_instance_raw[instance_id][category]
                    target["byte_ns"] += value * interval_ns
                    target["peak_per_rank_bytes"] = max(
                        target["peak_per_rank_bytes"], value)
                    aggregate_values[category] += value
                per_instance_physical_occupied_peak[instance_id] = max(
                    per_instance_physical_occupied_peak[instance_id],
                    state["physical_idle_reusable"]
                    + state["physical_non_idle_active"],
                )
                per_instance_reservation_adjusted_claim_peak[instance_id] = max(
                    per_instance_reservation_adjusted_claim_peak[instance_id],
                    state["physical_idle_reusable"]
                    + state["physical_non_idle_active"]
                    + state["reserved_free_slack"],
                )
            for category in categories:
                value = aggregate_values[category]
                aggregate_raw[category]["byte_ns"] += value * interval_ns
                aggregate_raw[category]["peak_per_rank_bytes"] = max(
                    aggregate_raw[category]["peak_per_rank_bytes"], value)
            physical_occupied_peak = max(
                physical_occupied_peak,
                aggregate_values["physical_idle_reusable"]
                + aggregate_values["physical_non_idle_active"],
            )
            reservation_adjusted_claim_peak = max(
                reservation_adjusted_claim_peak,
                aggregate_values["physical_idle_reusable"]
                + aggregate_values["physical_non_idle_active"]
                + aggregate_values["reserved_free_slack"],
            )

        if contributing_intervals == 0:
            raise RuntimeError(
                "HBM occupancy window contains no positive-duration state")

        capacities = {
            instance_id: int(
                self._history[0][1][instance_id][
                    "capacity_per_rank_bytes"])
            for instance_id in instance_ids
        }
        for _, snapshot in self._history:
            for instance_id in instance_ids:
                observed = int(snapshot[instance_id][
                    "capacity_per_rank_bytes"])
                if observed != capacities[instance_id]:
                    raise RuntimeError(
                        "HBM KV capacity changed during occupancy accounting: "
                        f"instance={instance_id}, initial="
                        f"{capacities[instance_id]}, observed={observed}")

        per_instance = {}
        for instance_id in instance_ids:
            capacity = capacities[instance_id]
            category_report = {
                category: self._category_report(
                    per_instance_raw[instance_id][category],
                    duration_ns,
                    capacity,
                )
                for category in categories
            }
            physical_average = (
                category_report["physical_idle_reusable"]
                ["average_per_rank_bytes"]
                + category_report["physical_non_idle_active"]
                ["average_per_rank_bytes"]
            )
            reservation_adjusted_average = (
                physical_average
                + category_report["reserved_free_slack"]
                ["average_per_rank_bytes"]
            )
            per_instance[str(instance_id)] = {
                "pd_type": str(
                    getattr(self.schedulers[instance_id], "pd_type", None)
                    or "colocated"),
                "capacity_per_rank_bytes": capacity,
                "categories": category_report,
                "average_physical_occupied_per_rank_bytes": physical_average,
                "average_physical_occupied_utilization_fraction": (
                    physical_average / capacity if capacity > 0 else None),
                "peak_physical_occupied_per_rank_bytes": (
                    per_instance_physical_occupied_peak[instance_id]),
                "peak_physical_occupied_utilization_fraction": (
                    per_instance_physical_occupied_peak[instance_id] / capacity
                    if capacity > 0 else None),
                "average_reservation_adjusted_claim_per_rank_bytes": (
                    reservation_adjusted_average),
                "average_reservation_adjusted_claim_fraction": (
                    reservation_adjusted_average / capacity
                    if capacity > 0 else None),
                "peak_reservation_adjusted_claim_per_rank_bytes": (
                    per_instance_reservation_adjusted_claim_peak[instance_id]),
                "peak_reservation_adjusted_claim_fraction": (
                    per_instance_reservation_adjusted_claim_peak[instance_id]
                    / capacity if capacity > 0 else None),
            }

        aggregate_capacity = sum(capacities.values())
        aggregate_categories = {
            category: self._category_report(
                aggregate_raw[category], duration_ns, aggregate_capacity)
            for category in categories
        }
        average_physical_occupied = (
            aggregate_categories["physical_idle_reusable"]
            ["average_per_rank_bytes"]
            + aggregate_categories["physical_non_idle_active"]
            ["average_per_rank_bytes"]
        )
        average_reservation_adjusted_claim = (
            average_physical_occupied
            + aggregate_categories["reserved_free_slack"]
            ["average_per_rank_bytes"]
        )
        return {
            "schema_version": self.SCHEMA_VERSION,
            "units": "per_rank_bytes",
            "window_start_ns": start_ns,
            "window_end_ns": end_ns,
            "window_duration_ns": duration_ns,
            "coverage": {
                "first_observation_ns": first_observation_ns,
                "last_observation_ns": last_observation_ns,
                "covers_window": True,
                "observation_calls": self._observation_calls,
                "state_points": len(self._history),
                "same_timestamp_replacement_count": (
                    self._same_timestamp_replacements),
                "contributing_intervals": contributing_intervals,
                "interval_semantics": (
                    "right_continuous_half_open_[snapshot,next_snapshot)"),
            },
            "physical_capacity_breakdown": list(
                self._PHYSICAL_CATEGORIES),
            "logical_reservation_overlay": list(
                self._OVERLAY_CATEGORIES),
            "per_instance": per_instance,
            "aggregate": {
                "capacity_per_rank_bytes_sum": aggregate_capacity,
                "categories": aggregate_categories,
                "average_physical_occupied_per_rank_bytes": (
                    average_physical_occupied),
                "average_physical_occupied_utilization_fraction": (
                    average_physical_occupied / aggregate_capacity
                    if aggregate_capacity > 0 else None),
                "peak_physical_occupied_per_rank_bytes": (
                    physical_occupied_peak),
                "peak_physical_occupied_utilization_fraction": (
                    physical_occupied_peak / aggregate_capacity
                    if aggregate_capacity > 0 else None),
                "average_reservation_adjusted_claim_per_rank_bytes": (
                    average_reservation_adjusted_claim),
                "average_reservation_adjusted_claim_fraction": (
                    average_reservation_adjusted_claim / aggregate_capacity
                    if aggregate_capacity > 0 else None),
                "peak_reservation_adjusted_claim_per_rank_bytes": (
                    reservation_adjusted_claim_peak),
                "peak_reservation_adjusted_claim_fraction": (
                    reservation_adjusted_claim_peak / aggregate_capacity
                    if aggregate_capacity > 0 else None),
            },
            "conservation": {
                "passed": True,
                "physical_identity": (
                    "idle_reusable + non_idle_active + physical_free "
                    "== allocatable_kv_capacity"),
                "reservation_adjusted_identity": (
                    "idle_reusable + non_idle_active + reserved_free_slack "
                    "+ unclaimed_allocatable_slack "
                    "== allocatable_kv_capacity"),
                "logical_overlay_identity": (
                    "reserved_free_slack + future_reclaim_backed_reservation "
                    "== logical_destination_admission_reservation"),
            },
            "semantics": {
                "capacity": (
                    "Per-rank npu_allocatable_mem minus static model weights; "
                    "runtime reserve is outside this KV ceiling."),
                "physical_idle_reusable": (
                    "Manager-owned idle entries whose current location is HBM."),
                "physical_non_idle_active": (
                    "Physical KV bytes not owned by an idle manager entry; this "
                    "includes active prefill/decode, P/D receive ownership, and "
                    "any non-agentic NPU prefix-cache allocation."),
                "logical_destination_admission_reservation": (
                    "Non-additive future-ready pending HBM allocations and "
                    "active reclaim claims from _hbm_logically_reserved."),
                "reserved_free_slack": (
                    "The portion of logical reservations backed by currently "
                    "free physical capacity. It is a reserved-free-slack claim, "
                    "not physical occupancy."),
                "future_reclaim_backed_reservation": (
                    "The portion of logical reservations backed by physical "
                    "victims whose asynchronous reclaim has not committed. It "
                    "overlaps physical idle bytes and must never be stacked on "
                    "top of physical occupancy."),
                "aggregate": (
                    "Sum of per-rank instance capacities and ownership. P and D "
                    "instances are independent capacity domains; no TP-rank "
                    "multiplication is applied."),
            },
        }


class OnlineModelComputeAccounting:
    """Attribute completed online model-iteration time to recomputation.

    The fallback attribution is deliberately named *iteration service*: the
    ASTRA completion interval includes emitted compute and model collectives.
    It is not presented as pure CUDA-kernel time.  When a latency provider
    attaches exact ``model_compute_ns`` and ``recompute_model_compute_ns`` to a
    batch, those exact values take precedence.
    """

    def __init__(self):
        self.total_model_compute_ns = 0
        self.recompute_model_compute_ns = 0
        self.total_iteration_service_ns = 0
        self.recompute_iteration_service_ns = 0
        self.completed_batches = 0
        self.exact_compute_batches = 0
        self.token_attributed_batches = 0
        self.observations = []
        self.records = []
        self.long_context_experiment = None

    @staticmethod
    def _candidate_batch(scheduler, callback_id):
        batch_id = int(callback_id) - 1
        return next(
            (batch for batch in scheduler.inflight
             if int(batch.batch_id) == batch_id),
            None,
        )

    @staticmethod
    def _callback_completes(scheduler, batch, sys_id):
        if batch is None or int(sys_id) in batch.end:
            return False
        projected = set(int(value) for value in batch.end)
        projected.add(int(sys_id))
        first = int(scheduler.start_npu)
        if scheduler.pd_type == "prefill":
            last = first + int(scheduler.num_npus) * 2 - 1
        else:
            last = first + int(scheduler.num_npus) - 1
        return first in projected and last in projected

    @staticmethod
    def _recompute_query_tokens(batch):
        scheduled = batch.scheduled_tokens or {}
        recompute = 0
        for request in batch.requests:
            if not request.is_prefill():
                continue
            chunk_tokens = max(0, int(scheduled.get(request.id, 0)))
            if chunk_tokens <= 0:
                continue
            start = int(request.num_computed_tokens)
            end = start + chunk_tokens
            intervals = []
            if request.recompute_target_tokens is not None:
                intervals.append((
                    0, int(request.recompute_target_tokens)))
            active_prefill_frontier = int(
                request.active_prefill_recompute_frontier_tokens)
            if active_prefill_frontier > 0:
                intervals.append((0, active_prefill_frontier))
            recompute_start = int(request.agentic_kv_hit_tokens)
            recompute_end = (
                recompute_start
                + int(request.agentic_kv_recompute_tokens)
            )
            if recompute_end > recompute_start:
                intervals.append((recompute_start, recompute_end))
            clipped = sorted(
                (max(start, interval_start), min(end, interval_end))
                for interval_start, interval_end in intervals
                if min(end, interval_end) > max(start, interval_start)
            )
            covered_end = start
            for interval_start, interval_end in clipped:
                if interval_end <= covered_end:
                    continue
                recompute += interval_end - max(
                    interval_start, covered_end)
                covered_end = interval_end
        return recompute

    def prepare_completion(self, scheduler, callback_id, sys_id, finish_ns):
        """Snapshot a batch immediately before ``Scheduler.add_done``."""
        batch = self._candidate_batch(scheduler, callback_id)
        if not self._callback_completes(scheduler, batch, sys_id):
            return None
        dispatch_ns = getattr(
            batch, "agentic_astra_dispatch_time_ns", None)
        start_ns = (
            int(batch.batch_time)
            if dispatch_ns is None else int(dispatch_ns)
        )
        duration_ns = max(0, int(finish_ns) - start_ns)
        total_query_tokens = max(0, int(batch.total_len))
        recompute_query_tokens = min(
            total_query_tokens,
            self._recompute_query_tokens(batch),
        )
        attributed_ns = (
            duration_ns * recompute_query_tokens // total_query_tokens
            if total_query_tokens > 0 else 0
        )
        return BatchComputeObservation(
            instance_id=int(scheduler.instance_id),
            batch_id=int(batch.batch_id),
            start_ns=start_ns,
            finish_ns=int(finish_ns),
            iteration_service_ns=duration_ns,
            total_query_tokens=total_query_tokens,
            recompute_query_tokens=recompute_query_tokens,
            attributed_recompute_iteration_service_ns=attributed_ns,
            pd_type=str(scheduler.pd_type or "colocated"),
            real_request_count=len(batch.requests),
        )

    def record_completion(self, observation, batch=None):
        if observation is None:
            return
        batch_contract = getattr(
            batch, "online_long_context_experiment", None)
        if batch_contract is not None:
            canonical_contract = json.loads(json.dumps(
                batch_contract, sort_keys=True
            ))
            if (self.long_context_experiment is not None
                    and self.long_context_experiment != canonical_contract):
                raise RuntimeError(
                    "Online latency long-context contract changed between "
                    "completed batches"
                )
            if self.long_context_experiment is None:
                self.long_context_experiment = canonical_contract
        self.completed_batches += 1
        self.total_iteration_service_ns += observation.iteration_service_ns
        self.recompute_iteration_service_ns += (
            observation.attributed_recompute_iteration_service_ns)

        exact_total = getattr(batch, "model_compute_ns", None)
        exact_recompute = getattr(batch, "recompute_model_compute_ns", None)
        if exact_total is not None and exact_recompute is not None:
            exact_total = max(0, int(exact_total))
            exact_recompute = max(0, min(
                exact_total, int(exact_recompute)))
            self.total_model_compute_ns += exact_total
            self.recompute_model_compute_ns += exact_recompute
            self.exact_compute_batches += 1
            attribution = "provider_comp_critical_path"
        else:
            self.total_model_compute_ns += observation.iteration_service_ns
            self.recompute_model_compute_ns += (
                observation.attributed_recompute_iteration_service_ns)
            self.token_attributed_batches += 1
            exact_total = observation.iteration_service_ns
            exact_recompute = (
                observation.attributed_recompute_iteration_service_ns)
            attribution = "iteration_service_token_attribution"
        self.observations.append(observation)
        self.records.append({
            "observation": observation,
            "model_compute_ns": exact_total,
            "recompute_model_compute_ns": exact_recompute,
            "attribution": attribution,
        })

    def summary(self, start_ns=None, end_ns=None):
        if (start_ns is None) != (end_ns is None):
            raise ValueError("compute summary window requires start and end")
        if start_ns is not None and int(end_ns) < int(start_ns):
            raise ValueError("compute summary end precedes start")
        records = self.records
        if start_ns is not None:
            start_ns = int(start_ns)
            end_ns = int(end_ns)
            records = [
                record for record in records
                if start_ns < record["observation"].finish_ns <= end_ns
            ]

        total_compute = 0
        recompute_compute = 0
        total_iteration = 0
        recompute_iteration = 0
        exact_batches = 0
        token_batches = 0
        batch_size_by_pd_type = defaultdict(lambda: {
            "completed_batch_count": 0,
            "non_dummy_completed_batch_count": 0,
            "dp_dummy_completed_batch_count": 0,
            "total_real_request_memberships": 0,
        })
        for record in records:
            observation = record["observation"]
            total_compute += int(record["model_compute_ns"])
            recompute_compute += int(record["recompute_model_compute_ns"])
            total_iteration += observation.iteration_service_ns
            recompute_iteration += (
                observation.attributed_recompute_iteration_service_ns)
            if record["attribution"] == "provider_comp_critical_path":
                exact_batches += 1
            else:
                token_batches += 1
            pd_type = str(getattr(
                observation, "pd_type", "colocated") or "colocated")
            real_request_count = int(getattr(
                observation, "real_request_count", 0))
            if real_request_count < 0:
                raise RuntimeError(
                    "Online batch accounting observed a negative real request "
                    f"count: {real_request_count}")
            role = batch_size_by_pd_type[pd_type]
            role["completed_batch_count"] += 1
            role["total_real_request_memberships"] += real_request_count
            if real_request_count == 0:
                role["dp_dummy_completed_batch_count"] += 1
            else:
                role["non_dummy_completed_batch_count"] += 1
        exact = (
            bool(records) and exact_batches == len(records)
        )
        fraction = (
            recompute_compute / total_compute
            if total_compute > 0 else None
        )
        completed_batch_count = sum(
            value["completed_batch_count"]
            for value in batch_size_by_pd_type.values())
        non_dummy_batch_count = sum(
            value["non_dummy_completed_batch_count"]
            for value in batch_size_by_pd_type.values())
        dummy_batch_count = sum(
            value["dp_dummy_completed_batch_count"]
            for value in batch_size_by_pd_type.values())
        total_memberships = sum(
            value["total_real_request_memberships"]
            for value in batch_size_by_pd_type.values())

        def batch_size_group(value):
            completed = value["completed_batch_count"]
            non_dummy = value["non_dummy_completed_batch_count"]
            memberships = value["total_real_request_memberships"]
            return {
                **value,
                "mean_real_requests_per_non_dummy_batch": (
                    memberships / non_dummy if non_dummy else None),
                "mean_real_requests_per_completed_batch_including_dummy": (
                    memberships / completed if completed else None),
            }

        return {
            "completed_batches": len(records),
            "exact_compute_batches": exact_batches,
            "token_attributed_batches": token_batches,
            "total_model_compute_ns": total_compute,
            "recompute_model_compute_ns": recompute_compute,
            "recompute_fraction_of_total_model_compute": fraction,
            "total_iteration_service_ns": total_iteration,
            "recompute_iteration_service_ns": recompute_iteration,
            "attribution": (
                "provider_comp_critical_path"
                if exact else
                "astra_iteration_service_weighted_by_recompute_query_tokens"
            ),
            "window_start_ns": start_ns,
            "window_end_ns": end_ns,
            "window_semantics": (
                "full_simulation" if start_ns is None
                else "batch_completions_in_half_open_interval_(start,end]"
            ),
            "real_batch_size": {
                "completed_batch_count": completed_batch_count,
                "non_dummy_completed_batch_count": non_dummy_batch_count,
                "dp_dummy_completed_batch_count": dummy_batch_count,
                "total_real_request_memberships": total_memberships,
                "mean_real_requests_per_non_dummy_batch": (
                    total_memberships / non_dummy_batch_count
                    if non_dummy_batch_count else None),
                "mean_real_requests_per_completed_batch_including_dummy": (
                    total_memberships / completed_batch_count
                    if completed_batch_count else None),
                "by_pd_type": {
                    pd_type: batch_size_group(value)
                    for pd_type, value in sorted(
                        batch_size_by_pd_type.items())
                },
                "membership_semantics": (
                    "Arithmetic mean of real len(batch.requests) memberships "
                    "over completed online model iterations. Repeated chunked "
                    "prefill/decode memberships are intentional; DP sync-only "
                    "batches have zero real requests and are reported separately."),
            },
            "long_context_experiment": self.long_context_experiment,
            "scope_note": (
                "Provider critical-path COMP time excludes TP/EP/P-D "
                "collectives and uses the slowest parallel EP rank, not the "
                "emitted rank sum. It is used only when every selected batch "
                "provides it. Otherwise the online ASTRA iteration interval, "
                "including model collectives, is apportioned by scheduled "
                "query tokens and is not claimed as pure kernel time."
                " Mixed batches are attributed as one critical-path batch "
                "at their completion timestamp."
            ),
        }


def _resolve_dataset_path(path):
    if os.path.isabs(path):
        return path
    # ``serving.__main__`` changes cwd to astra-sim before resolving inputs.
    return os.path.join("..", path)


def _workload_sequence_lengths(path, num_reqs=0, backlog_epochs=1):
    lengths = []
    loaded = 0
    with open(_resolve_dataset_path(path), "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if int(num_reqs) > 0 and loaded >= int(num_reqs):
                break
            row = json.loads(line)
            loaded += 1
            if "sub_requests" in row:
                row_lengths = []
                for sub in row.get("sub_requests", []):
                    prompt = int(sub["input_toks"])
                    generated = int(sub["output_toks"])
                    if prompt < 0 or generated < 0:
                        raise ValueError(
                            f"Negative token count at workload line "
                            f"{line_number}")
                    row_lengths.append(prompt + generated)
                lengths.extend(row_lengths * int(backlog_epochs))
            else:
                prompt = int(row["input_toks"])
                generated = int(row["output_toks"])
                if prompt < 0 or generated < 0:
                    raise ValueError(
                        f"Negative token count at workload line "
                        f"{line_number}")
                lengths.append(prompt + generated)
    if not lengths:
        raise ValueError("Strict infinite-HBM oracle requires a non-empty workload")
    return lengths


class StrictInfiniteHBMOracle:
    """Install and validate a finite proof bound that cannot bind online HBM.

    Each scheduler receives enough per-rank KV capacity for two simultaneous
    block-rounded copies of *every call in the selected workload*.  That is a
    conservative upper bound even for P/D handoff overlap and is stronger than
    the actual dependency chain, where only one call per session can execute.
    """

    PROOF_COPY_MULTIPLIER = 2

    def __init__(self, schedulers, sequence_lengths):
        self.schedulers = list(schedulers)
        self.sequence_lengths = tuple(int(value) for value in sequence_lengths)
        self.capacities = {}
        self.peak_used = {}
        self.peak_logically_reserved = {}
        self._install()

    @classmethod
    def from_workload(
            cls, schedulers, dataset, *, num_reqs=0, backlog_epochs=1):
        return cls(
            schedulers,
            _workload_sequence_lengths(
                dataset,
                num_reqs=num_reqs,
                backlog_epochs=backlog_epochs,
            ),
        )

    def _install(self):
        for scheduler in self.schedulers:
            memory = scheduler.memory
            block_size = int(scheduler.block_size)
            rounded_tokens = sum(
                ((tokens + block_size - 1) // block_size) * block_size
                for tokens in self.sequence_lengths
            )
            proof_kv_bytes = (
                self.PROOF_COPY_MULTIPLIER
                * int(memory.get_kv(rounded_tokens))
            )
            safety_bytes = max(1, int(memory.get_kv(block_size)))
            required = int(memory.weight) + proof_kv_bytes + safety_bytes
            original = int(memory.npu_allocatable_mem)
            installed = max(required, original + safety_bytes)
            # The normal allocator intentionally checks npu_allocatable_mem;
            # npu_mem is only its compatibility alias. A strict oracle must
            # lift both together while retaining the real physical/reserve
            # values as provenance rather than silently erasing them.
            memory.npu_allocatable_mem = installed
            memory.npu_mem = installed
            if hasattr(memory, "mem_for_kv"):
                memory.mem_for_kv = installed - int(memory.weight)
            self.capacities[int(scheduler.instance_id)] = {
                "original_per_rank_bytes": original,
                "original_allocatable_per_rank_bytes": original,
                "physical_per_rank_bytes": int(memory.npu_physical_mem),
                "runtime_reserve_per_rank_bytes": int(
                    memory.npu_runtime_reserve_bytes),
                "installed_per_rank_bytes": installed,
                "proof_kv_per_rank_bytes": proof_kv_bytes,
                "safety_per_rank_bytes": safety_bytes,
            }
            self.peak_used[int(scheduler.instance_id)] = int(memory.npu_used)
            self.peak_logically_reserved[int(scheduler.instance_id)] = 0

    def observe(self, manager=None):
        for scheduler in self.schedulers:
            instance_id = int(scheduler.instance_id)
            used = int(scheduler.memory.npu_used)
            self.peak_used[instance_id] = max(
                self.peak_used[instance_id], used)
            if manager is not None:
                reserved = int(manager._hbm_logically_reserved(instance_id))
                self.peak_logically_reserved[instance_id] = max(
                    self.peak_logically_reserved[instance_id], reserved)

    def validate(self, manager, completed_requests: Iterable[object]):
        self.observe(manager)
        violations = []
        for scheduler in self.schedulers:
            instance_id = int(scheduler.instance_id)
            capacity = self.capacities[instance_id][
                "installed_per_rank_bytes"]
            if int(scheduler.memory.npu_allocatable_mem) != capacity:
                violations.append(
                    f"instance {instance_id} allocator capacity diverged "
                    f"from oracle bound: allocator="
                    f"{scheduler.memory.npu_allocatable_mem}, "
                    f"oracle={capacity}")
            peak_claim = (
                self.peak_used[instance_id]
                + self.peak_logically_reserved[instance_id]
            )
            if peak_claim >= capacity:
                violations.append(
                    f"instance {instance_id} oracle HBM bound bound or "
                    f"exhausted: peak_claim={peak_claim}, capacity={capacity}")

        metrics = manager.metrics
        zero_fields = (
            "cpu_hits", "ssd_hits", "dropped_misses",
            "capacity_drops", "hbm_capacity_demotions",
            "hbm_capacity_drops", "cpu_capacity_evictions",
            "ssd_capacity_evictions", "ssd_capacity_admission_drops",
            "capacity_induced_recompute_tokens",
            "policy_avoidable_recompute_tokens",
            "active_recompute_preemptions", "active_cpu_swap_preemptions",
            "pd_active_prefill_recompute_preemptions",
            "pd_active_prefill_recompute_tokens",
            "agentic_kv_restored_tokens_discarded_by_active_prefill_recompute",
            "pd_chunk_cancelled_admissions",
            "pd_chunk_cancelled_admission_wait_ns",
            "pd_chunk_cancelled_admission_critical_wait_ns",
        )
        observed_counters = {}
        for field in zero_fields:
            value = int(getattr(metrics, field))
            observed_counters[field] = value
            if value != 0:
                violations.append(f"oracle counter {field}={value}, expected 0")

        invalid_sources = []
        checked_reusable_resumes = 0
        for request in completed_requests:
            if (request.sub_request_index is None
                    or int(request.sub_request_index) <= 0
                    or int(request.prefix_reuse_tokens) <= 0):
                continue
            checked_reusable_resumes += 1
            if str(request.agentic_kv_source) != "hbm":
                invalid_sources.append({
                    "request_id": int(request.id),
                    "session_id": str(request.session_id),
                    "source": request.agentic_kv_source,
                    "reuse_tokens": int(request.prefix_reuse_tokens),
                })
        if invalid_sources:
            violations.append(
                f"{len(invalid_sources)} reusable resume(s) were not HBM hits")

        per_instance = {}
        for instance_id, capacity in sorted(self.capacities.items()):
            peak_claim = (
                self.peak_used[instance_id]
                + self.peak_logically_reserved[instance_id]
            )
            per_instance[str(instance_id)] = {
                **capacity,
                "peak_physical_used_per_rank_bytes": self.peak_used[
                    instance_id],
                "peak_logically_reserved_per_rank_bytes": (
                    self.peak_logically_reserved[instance_id]),
                "minimum_slack_per_rank_bytes": (
                    capacity["installed_per_rank_bytes"] - peak_claim),
                "nonbinding": peak_claim < capacity[
                    "installed_per_rank_bytes"],
            }
        report = {
            "enabled": True,
            "passed": not violations,
            "proof": {
                "selected_call_count": len(self.sequence_lengths),
                "sum_declared_sequence_tokens": sum(self.sequence_lengths),
                "copy_multiplier": self.PROOF_COPY_MULTIPLIER,
                "bound_semantics": (
                    "weight plus two block-rounded full KV copies of every "
                    "selected call, independently on every scheduler, plus "
                    "one safety block"
                ),
            },
            "per_instance": per_instance,
            "checked_reusable_resumes": checked_reusable_resumes,
            "invalid_resume_sources": invalid_sources,
            "zero_counter_invariants": observed_counters,
            "violations": violations,
        }
        if violations:
            raise RuntimeError(
                "Strict infinite-HBM oracle validation failed: "
                + "; ".join(violations))
        return report


def configure_strict_oracle(agentic_kv_config):
    if agentic_kv_config is None:
        raise ValueError(
            "--strict-infinite-hbm-oracle requires --agentic-kv-config")
    agentic_kv_config.policy = "preserve"
    agentic_kv_config.demotion_mode = "capacity-only"
    agentic_kv_config.validate()
    return agentic_kv_config


def measurement_target_reached(router, admission_config):
    router_target = getattr(router, "measurement_target_reached", None)
    if router_target is not None:
        return bool(router_target())
    if (getattr(
            admission_config,
            "measurement_cohort_selection",
            "completion_order",
    ) == "admission_order"):
        raise TypeError(
            "admission_order measurement requires a Router with fixed-target "
            "tracking")
    target = (
        int(admission_config.warmup_completions)
        + int(admission_config.measure_completions)
    )
    if target <= 0:
        return False
    return int(router.session_admission_summary()["completed_sessions"]) >= target

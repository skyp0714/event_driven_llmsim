"""Compact, integrity-checked collector for live ASTRA comparison campaigns.

The live sweep deliberately retains detailed request, session, and runtime
reports for diagnosis.  This module turns those large reports into one compact
row per campaign cell without relying on the sweep runner's bounded
``bottleneck_fields`` preview.

Collection is fail closed for content integrity and headline metrics:

* the manifest's result digest (and byte count, when present) is verified;
* every artifact recorded by ``result.json`` is verified by path, size, and
  SHA-256;
* request timing, TPOT, measurement-roster membership, distributions, SLO
  pass counts, and goodputs are independently recomputed from ``requests.csv``;
* the measurement roster is checked against the campaign identity and, when
  present, the session report.
* runtime-guarded stress cells require every measured resume's native arrival
  to occur no later than the content-addressed final external guard offer.

The active-prefill drain contract is mandatory for full-model HBF cells and
its final quiescence/accounting is independently verified.  Other optional
diagnostic fields are copied only when they exist in the raw report; missing
fields are omitted rather than replaced with zero or a guessed value.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 2
NANOSECONDS_PER_SECOND = 1_000_000_000

_REQUIRED_ARTIFACTS = frozenset({
    "requests",
    "session_report",
    "runtime_report",
})
_REQUIRED_REQUEST_FIELDS = frozenset({
    "session_id",
    "sub_request_index",
    "arrival",
    "end_time",
    "latency",
    "TTFT",
    "TPOT",
    "output",
    "generated_tokens",
})
_RUNTIME_GUARDED_SCENARIO_IDS = frozenset({
    "tracelab-headline-1741-balanced-highrate-v2",
})

_REQUEST_WAIT_FIELDS = {
    "scheduler_queue": "scheduler_queue_wait_ns",
    "owner_gate": "agentic_kv_owner_gate_ns",
    "pd_pair_fifo": "pd_pair_fifo_wait_ns",
    "prepare_boundary": "agentic_kv_prepare_boundary_wait_ns",
    "source_demotion_join": "agentic_kv_source_demotion_join_wait_ns",
    "hbm_admission": "agentic_kv_hbm_admission_wait_ns",
    "transient_dram_capacity": (
        "agentic_kv_transient_dram_capacity_wait_ns"),
    "restore_queue": "agentic_kv_restore_queue_wait_ns",
    "restore_service": "agentic_kv_restore_service_ns",
    "restore_gate": "agentic_kv_restore_gate_wait_ns",
    "pd_decode_capacity": "pd_decode_capacity_wait_ns",
    "pd_decode_admission": "pd_decode_admission_wait_ns",
    "pd_decode_admission_critical": (
        "pd_decode_admission_critical_wait_ns"),
    "pd_prefill_capacity": "pd_prefill_capacity_wait_ns",
    "pd_prefill_admission": "pd_prefill_admission_wait_ns",
    "pd_prefill_admission_critical": (
        "pd_prefill_admission_critical_wait_ns"),
    "pd_launch_admission": "pd_launch_admission_wait_ns",
    "pd_launch_admission_critical": (
        "pd_launch_admission_critical_wait_ns"),
}

_BASELINE_SSD_PATHS = {
    "hbm_hits": "totals.hbm_hits",
    "cpu_hits": "totals.cpu_hits",
    "ssd_hits": "totals.ssd_hits",
    "dropped_misses": "totals.dropped_misses",
    "hbm_to_cpu_bytes": "totals.hbm_to_cpu_bytes",
    "cpu_to_hbm_bytes": "totals.cpu_to_hbm_bytes",
    "cpu_to_ssd_bytes": "totals.cpu_to_ssd_bytes",
    "hbm_to_ssd_bytes": "totals.hbm_to_ssd_bytes",
    "ssd_to_hbm_bytes": "totals.ssd_to_hbm_bytes",
    "ssd_to_cpu_stage_bytes": "totals.ssd_to_cpu_stage_bytes",
    "cpu_stage_to_hbm_bytes": "totals.cpu_stage_to_hbm_bytes",
    "ssd_host_write_bytes": "totals.ssd_host_write_bytes",
    "ssd_cancelled_host_write_bytes": (
        "totals.ssd_cancelled_host_write_bytes"),
    "ssd_host_read_bytes": "totals.ssd_host_read_bytes",
    "direct_ssd_write_bytes": "totals.direct_ssd_write_bytes",
    "direct_ssd_read_bytes": "totals.direct_ssd_read_bytes",
    "ssd_byte_ns": "totals.ssd_byte_ns",
    "peak_ssd_used_bytes": "totals.peak_ssd_used_bytes",
    "peak_ssd_committed_reserved_bytes": (
        "totals.peak_ssd_committed_reserved_bytes"),
    "ssd_capacity_evictions": "totals.ssd_capacity_evictions",
    "ssd_capacity_admission_drops": (
        "totals.ssd_capacity_admission_drops"),
    "capacity_drops": "totals.capacity_drops",
    "capacity_induced_recompute_tokens": (
        "totals.capacity_induced_recompute_tokens"),
    "capacity_bytes": "ssd.capacity_bytes",
    "capacity_bytes_per_node": "ssd.capacity_bytes_per_node",
    "used_bytes_at_end": "ssd.used_bytes",
    "reserved_bytes_at_end": "ssd.reserved_bytes",
    "committed_reserved_bytes_at_end": "ssd.committed_reserved_bytes",
    "media_aligned_host_write_bytes": (
        "storage.totals.aligned_host_write_bytes"),
    "media_host_read_bytes": "storage.totals.host_read_bytes",
}

_BASELINE_DRAM_PATHS = {
    "cpu_byte_ns": "totals.cpu_byte_ns",
    "peak_idle_cpu_bytes": "totals.peak_idle_cpu_bytes",
    "hbm_capacity_demotions": "totals.hbm_capacity_demotions",
    "hbm_capacity_drops": "totals.hbm_capacity_drops",
    "cpu_capacity_evictions": "totals.cpu_capacity_evictions",
    "cpu_capacity_bypasses": "totals.cpu_capacity_bypasses",
    "transient_reservations": "host_dram_staging.reservation_count",
    "transient_reserved_bytes_membership_sum": (
        "host_dram_staging.reservation_bytes_membership_sum"),
    "transient_byte_ns_membership_sum": (
        "host_dram_staging.byte_ns_membership_sum"),
    "transient_capacity_wait_ns": (
        "host_dram_staging.aggregate_explicit_capacity_wait_ns"),
    "transient_pressure_stall_upper_bound_ns": (
        "host_dram_staging.aggregate_pressure_stall_upper_bound_ns"),
    "transient_capacity_deferrals": (
        "host_dram_staging.capacity_deferrals"),
    "transient_oversize_restore_failures": (
        "host_dram_staging.oversize_restore_failures"),
    "transient_pending_capacity_wait_sessions": (
        "host_dram_staging.pending_capacity_wait_sessions"),
    "transient_censored_capacity_wait_count": (
        "host_dram_staging.censored_capacity_wait_count"),
    "transient_cpu_lru_evictions": "host_dram_staging.cpu_lru_evictions",
    "peak_transient_bytes": "host_dram_staging.peak_transient_bytes",
    "peak_persistent_plus_transient_bytes": (
        "host_dram_staging.peak_persistent_plus_transient_bytes"),
}

_BASELINE_FABRIC_PATHS = {
    "pd_hbm_to_hbm_bytes": "totals.pd_hbm_to_hbm_bytes",
    "transfer_jobs": "totals.transfer_jobs",
    "queued_transfer_jobs": "totals.queued_transfer_jobs",
    "migration_service_ns": "totals.migration_service_ns",
    "migration_queue_wait_ns": "totals.migration_queue_wait_ns",
    "critical_restore_service_ns": "totals.critical_restore_service_ns",
    "critical_restore_queue_wait_ns": (
        "totals.critical_restore_queue_wait_ns"),
    "external_fabric_jobs_issued": (
        "totals.external_fabric_jobs_issued"),
    "external_fabric_jobs_completed": (
        "totals.external_fabric_jobs_completed"),
    "external_fabric_jobs_censored": (
        "totals.external_fabric_jobs_censored"),
    "external_fabric_lane_bytes": "totals.external_fabric_lane_bytes",
    "external_fabric_censored_lane_bytes": (
        "totals.external_fabric_censored_lane_bytes"),
    "direct_fabric_dispatch_blocks": (
        "totals.direct_fabric_dispatch_blocks"),
    "direct_fabric_dispatch_wait_ns": (
        "totals.direct_fabric_dispatch_wait_ns"),
    "global_transfer_execution_ns": (
        "observed_load_activity.global_transfer_execution_ns"),
    "global_any_model_execution_ns": (
        "observed_load_activity.global_any_model_execution_ns"),
    "global_model_or_transfer_execution_ns": (
        "observed_load_activity.global_model_or_transfer_execution_ns"),
    "global_any_model_busy_fraction": (
        "observed_load_activity.global_any_model_busy_fraction"),
    "fully_quiescent_ns": "observed_load_activity.fully_quiescent_ns",
}

_HBF_ROUTING_PATHS = {
    "offered_requests": "adapter.metrics.offered_requests",
    "gpu_requests": "adapter.metrics.gpu_requests",
    "hbf_requests": "adapter.metrics.hbf_requests",
    "gpu_completions": "adapter.metrics.gpu_completions",
    "hbf_completions": "adapter.metrics.hbf_completions",
    "router_completion_proxies": (
        "adapter.metrics.router_completion_proxies"),
    "censored_successors": "adapter.metrics.censored_successors",
    "censored_active_gpu_requests": (
        "adapter.metrics.censored_active_gpu_requests"),
    "censored_queued_gpu_requests": (
        "adapter.metrics.censored_queued_gpu_requests"),
    "gpu_first_turn": "adapter.execution_counts.gpu_first_turn",
    "hbf_ready": "adapter.execution_counts.hbf_ready",
    "gpu_migration_inflight": (
        "adapter.execution_counts.gpu_migration_inflight"),
    "gpu_capacity_fallback": (
        "adapter.execution_counts.gpu_capacity_fallback"),
    "gpu_lpddr_capacity_fallback": (
        "adapter.execution_counts.gpu_lpddr_capacity_fallback"),
}

_HBF_CAPACITY_PATHS = {
    "hbf_capacity_bytes_per_card": (
        "hardware.hbf_capacity_bytes_per_card"),
    "lpddr_capacity_bytes_per_card": (
        "hardware.lpddr_capacity_bytes_per_card"),
    "card_count": "hardware.card_count",
    "lpddr_kv_capacity_bytes_per_card": (
        "adapter.pool.lpddr_kv_capacity_bytes_per_card"),
    "workspace_bytes_per_card": "adapter.pool.workspace_bytes_per_card",
    "max_lpddr_active_bytes_per_card": (
        "adapter.pool.metrics.max_lpddr_active_bytes_per_card"),
    "lpddr_capacity_deferrals": (
        "adapter.pool.metrics.lpddr_capacity_deferrals"),
    "migrations_started": "adapter.lifecycle.metrics.migrations_started",
    "migrations_committed": "adapter.lifecycle.metrics.migrations_committed",
    "migrations_stale": "adapter.lifecycle.metrics.migrations_stale",
    "migration_logical_bytes": (
        "adapter.lifecycle.metrics.migration_logical_bytes"),
    "migration_physical_bytes": (
        "adapter.lifecycle.metrics.migration_physical_bytes"),
    "migration_wasted_physical_bytes": (
        "adapter.lifecycle.metrics.migration_wasted_physical_bytes"),
    "append_jobs_started": (
        "adapter.lifecycle.metrics.append_jobs_started"),
    "append_jobs_committed": (
        "adapter.lifecycle.metrics.append_jobs_committed"),
    "append_logical_bytes": (
        "adapter.lifecycle.metrics.append_logical_bytes"),
    "append_physical_bytes": (
        "adapter.lifecycle.metrics.append_physical_bytes"),
    "hbf_reserved_bytes_peak": (
        "adapter.lifecycle.metrics.hbf_reserved_bytes_peak"),
    "gpu_retained_bytes_peak": (
        "adapter.lifecycle.metrics.gpu_retained_bytes_peak"),
    "capacity_evictions": (
        "adapter.lifecycle.metrics.capacity_evictions"),
    "gpu_ready_hbm_pressure_evictions": (
        "adapter.lifecycle.metrics.gpu_ready_hbm_pressure_evictions"),
    "gpu_ready_hbm_pressure_evicted_bytes": (
        "adapter.lifecycle.metrics.gpu_ready_hbm_pressure_evicted_bytes"),
    "gpu_fallback_resumes": (
        "adapter.lifecycle.metrics.gpu_fallback_resumes"),
    "gpu_recompute_resumes": (
        "adapter.lifecycle.metrics.gpu_recompute_resumes"),
    "lpddr_capacity_fallback_resumes": (
        "adapter.lifecycle.metrics.lpddr_capacity_fallback_resumes"),
    "hbf_resumes": "adapter.lifecycle.metrics.hbf_resumes",
}

_HBF_ATTENTION_PATHS = {
    "submitted_requests": "adapter.pool.metrics.submitted_requests",
    "completed_requests": "adapter.pool.metrics.completed_requests",
    "batches": "adapter.pool.metrics.batches",
    "mixed_batches": "adapter.pool.metrics.mixed_batches",
    "prefill_only_batches": "adapter.pool.metrics.prefill_only_batches",
    "decode_only_batches": "adapter.pool.metrics.decode_only_batches",
    "max_batch_size": "adapter.pool.metrics.max_batch_size",
    "prefill_query_tokens": "adapter.pool.metrics.prefill_query_tokens",
    "decode_query_tokens": "adapter.pool.metrics.decode_query_tokens",
    "hbf_read_bytes_per_rank": (
        "adapter.pool.metrics.hbf_read_bytes_per_rank"),
    "lpddr_bytes_per_rank": "adapter.pool.metrics.lpddr_bytes_per_rank",
    "collective_bytes_per_rank": (
        "adapter.pool.metrics.collective_bytes_per_rank"),
    "modeled_batch_ns": "adapter.pool.metrics.modeled_batch_ns",
    "embedding_modeled_ns": "adapter.pool.metrics.embedding_modeled_ns",
    "dense_modeled_ns": "adapter.pool.metrics.dense_modeled_ns",
    "attention_modeled_ns": "adapter.pool.metrics.attention_modeled_ns",
    "router_modeled_ns": "adapter.pool.metrics.router_modeled_ns",
    "moe_modeled_ns": "adapter.pool.metrics.moe_modeled_ns",
    "final_modeled_ns": "adapter.pool.metrics.final_modeled_ns",
    "collective_modeled_ns": "adapter.pool.metrics.collective_modeled_ns",
    "attention_compute_roof_ns": (
        "adapter.pool.metrics.attention_compute_roof_ns"),
    "attention_hbf_roof_ns": (
        "adapter.pool.metrics.attention_hbf_roof_ns"),
    "attention_lpddr_roof_ns": (
        "adapter.pool.metrics.attention_lpddr_roof_ns"),
    "attention_compute_dominant_batches": (
        "adapter.pool.metrics.attention_compute_dominant_batches"),
    "attention_hbf_dominant_batches": (
        "adapter.pool.metrics.attention_hbf_dominant_batches"),
    "attention_lpddr_dominant_batches": (
        "adapter.pool.metrics.attention_lpddr_dominant_batches"),
    "resource_delay_ns": "adapter.pool.metrics.resource_delay_ns",
    "astra_completed_batches": (
        "adapter.pool.metrics.astra_completed_batches"),
    "astra_completion_elapsed_ns": (
        "adapter.pool.metrics.astra_completion_elapsed_ns"),
    "astra_resource_delay_ns": (
        "adapter.pool.metrics.astra_resource_delay_ns"),
    "astra_dependency_critical_path_ns": (
        "adapter.pool.metrics.astra_dependency_critical_path_ns"),
    "astra_solo_resource_serialized_completion_ns": (
        "adapter.pool.metrics."
        "astra_solo_resource_serialized_completion_ns"),
    "astra_actual_resource_serialized_completion_ns": (
        "adapter.pool.metrics."
        "astra_actual_resource_serialized_completion_ns"),
    "astra_internal_resource_serialization_wait_ns": (
        "adapter.pool.metrics."
        "astra_internal_resource_serialization_wait_ns"),
    "astra_signed_interference_delta_ns": (
        "adapter.pool.metrics."
        "astra_signed_interference_delta_ns"),
}

_HBF_NETWORK_PATHS = {
    "hbf_read_bandwidth_gbps_per_card": (
        "hardware.hbf_read_bandwidth_gbps_per_card"),
    "hbf_read_latency_us": "hardware.hbf_read_latency_us",
    "hbf_write_bandwidth_gbps_per_card": (
        "hardware.hbf_write_bandwidth_gbps_per_card"),
    "hbf_write_latency_us": "hardware.hbf_write_latency_us",
    "lpddr_bandwidth_gbps_per_card": (
        "hardware.lpddr_bandwidth_gbps_per_card"),
    "intra_fabric_bandwidth_gbps_per_card": (
        "hardware.intra_fabric_bandwidth_gbps_per_card"),
    "intra_fabric_fixed_latency_us": (
        "hardware.intra_fabric_fixed_latency_us"),
    "pcie_root_count": "hardware.pcie_root_count",
    "cards_per_pcie_root": "hardware.cards_per_pcie_root",
    "pcie_card_to_root": "hardware.pcie_card_to_root",
    "pcie_nic_to_root": "hardware.pcie_nic_to_root",
    "pcie_resource_mode": "hardware.pcie_resource_mode",
    "pcie_p2p_mode": "hardware.pcie_p2p_mode",
    "pcie_root_bandwidth_gbps": "hardware.pcie_root_bandwidth_gbps",
    "pcie_root_fixed_latency_us": (
        "hardware.pcie_root_fixed_latency_us"),
    "pcie_inter_root_bandwidth_gbps": (
        "hardware.pcie_inter_root_bandwidth_gbps"),
    "pcie_inter_root_fixed_latency_us": (
        "hardware.pcie_inter_root_fixed_latency_us"),
    "rdma_bandwidth_gbps": "hardware.rdma_bandwidth_gbps",
    "rdma_one_way_latency_us": "hardware.rdma_one_way_latency_us",
    "lifecycle_astra_completed_jobs": (
        "adapter.lifecycle.metrics.astra_completed_jobs"),
    "lifecycle_astra_completion_elapsed_ns": (
        "adapter.lifecycle.metrics.astra_completion_elapsed_ns"),
    "lifecycle_astra_resource_delay_ns": (
        "adapter.lifecycle.metrics.astra_resource_delay_ns"),
    "lifecycle_astra_dependency_critical_path_ns": (
        "adapter.lifecycle.metrics."
        "astra_dependency_critical_path_ns"),
    "lifecycle_astra_solo_resource_serialized_completion_ns": (
        "adapter.lifecycle.metrics."
        "astra_solo_resource_serialized_completion_ns"),
    "lifecycle_astra_actual_resource_serialized_completion_ns": (
        "adapter.lifecycle.metrics."
        "astra_actual_resource_serialized_completion_ns"),
    "lifecycle_astra_internal_resource_serialization_wait_ns": (
        "adapter.lifecycle.metrics."
        "astra_internal_resource_serialization_wait_ns"),
    "lifecycle_astra_signed_interference_delta_ns": (
        "adapter.lifecycle.metrics."
        "astra_signed_interference_delta_ns"),
}

_HBF_PREFILL_DRAIN_POOL_PATHS = {
    "candidates": (
        "adapter.pool.metrics.prefill_drain_candidates"),
    "claimed": "adapter.pool.metrics.prefill_drain_claimed",
    "started": "adapter.pool.metrics.prefill_drain_started",
    "completed": "adapter.pool.metrics.prefill_drain_completed",
    "fallbacks": "adapter.pool.metrics.prefill_drain_fallbacks",
    "logical_tokens": (
        "adapter.pool.metrics.prefill_drain_logical_tokens"),
    "wait_ns": "adapter.pool.metrics.prefill_drain_wait_ns",
}

_HBF_PREFILL_DRAIN_LIFECYCLE_PATHS = {
    "candidates": (
        "adapter.lifecycle.metrics.active_prefill_drain_candidates"),
    "started": (
        "adapter.lifecycle.metrics.active_prefill_drain_started"),
    "satisfied": (
        "adapter.lifecycle.metrics.active_prefill_drain_satisfied"),
    "wait_existing_append": (
        "adapter.lifecycle.metrics."
        "active_prefill_drain_wait_existing_append"),
    "capacity_fallback": (
        "adapter.lifecycle.metrics."
        "active_prefill_drain_capacity_fallback"),
    "committed": (
        "adapter.lifecycle.metrics.active_prefill_drain_committed"),
    "stale": (
        "adapter.lifecycle.metrics.active_prefill_drain_stale"),
}

_HBF_PREFILL_DRAIN_PENDING_PATHS = {
    "adapter_request_by_session_count": (
        "adapter.pending_prefill_drain_request_by_session"),
    "adapter_active_request_by_job_count": (
        "adapter.active_prefill_drain_request_by_job"),
    "adapter_waiting_append_by_session_count": (
        "adapter.waiting_prefill_drain_append_jobs_by_session"),
    "lifecycle_active_job_count": (
        "adapter.lifecycle.active_prefill_drain_pending_job_ids"),
}

_MISSING = object()


class LiveAstraCollectError(ValueError):
    """Raised when a campaign artifact fails collection validation."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LiveAstraCollectError(f"cannot read artifact {path}") from exc
    return digest.hexdigest()


def _stable_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_finite_tree(value: object, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise LiveAstraCollectError(f"{name} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite_tree(child, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_tree(child, f"{name}[{index}]")


def _read_json_object(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise LiveAstraCollectError(
            f"cannot parse strict JSON object {path}") from exc
    if not isinstance(value, dict):
        raise LiveAstraCollectError(f"{path} is not a JSON object")
    _require_finite_tree(value, str(path))
    return value


def _resolve_recorded_path(raw: object, base: Path, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise LiveAstraCollectError(f"{name} has no artifact path")
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _require_sha256(raw: object, name: str) -> str:
    if (
        not isinstance(raw, str)
        or len(raw) != 64
        or any(character not in "0123456789abcdef" for character in raw)
    ):
        raise LiveAstraCollectError(
            f"{name} is not a lowercase SHA-256 digest")
    return raw


def _verify_record(
    record: object,
    *,
    base: Path,
    name: str,
) -> Path:
    if not isinstance(record, Mapping):
        raise LiveAstraCollectError(f"{name} artifact record is not an object")
    path = _resolve_recorded_path(record.get("path"), base, name)
    if not path.is_file():
        raise LiveAstraCollectError(f"{name} artifact is missing: {path}")
    byte_count = record.get("bytes")
    if type(byte_count) is not int or byte_count < 0:
        raise LiveAstraCollectError(f"{name} has invalid artifact byte count")
    digest = _require_sha256(record.get("sha256"), f"{name}.sha256")
    try:
        actual_bytes = path.stat().st_size
    except OSError as exc:
        raise LiveAstraCollectError(f"cannot stat artifact {path}") from exc
    if actual_bytes != byte_count:
        raise LiveAstraCollectError(
            f"{name} artifact byte count changed")
    if _sha256_file(path) != digest:
        raise LiveAstraCollectError(f"{name} artifact digest changed")
    return path


def _lookup(root: Mapping[str, object], dotted_path: str) -> object:
    current: object = root
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return _MISSING
        current = current[component]
    return current


def _select_paths(
    root: Mapping[str, object],
    paths: Mapping[str, str],
) -> dict[str, object]:
    selected = {}
    for output_name, source_path in paths.items():
        value = _lookup(root, source_path)
        if value is not _MISSING:
            selected[output_name] = value
    return selected


def _validate_hbf_astra_timing_metrics(
    runtime: Mapping[str, object],
    prefix: str,
) -> None:
    """Validate cumulative ASTRA timing fields, including a signed delta."""

    fields = (
        "astra_completion_elapsed_ns",
        "astra_resource_delay_ns",
        "astra_dependency_critical_path_ns",
        "astra_solo_resource_serialized_completion_ns",
        "astra_actual_resource_serialized_completion_ns",
        "astra_internal_resource_serialization_wait_ns",
        "astra_signed_interference_delta_ns",
    )
    values = {}
    for field in fields:
        path = f"{prefix}.{field}"
        value = _lookup(runtime, path)
        if value is _MISSING:
            raise LiveAstraCollectError(
                f"runtime is missing required HBF ASTRA timing field {path}")
        if type(value) is not int:
            raise LiveAstraCollectError(
                f"runtime.{path} must be a finite integer")
        values[field] = value

    dependency = values["astra_dependency_critical_path_ns"]
    solo = values["astra_solo_resource_serialized_completion_ns"]
    actual = values["astra_actual_resource_serialized_completion_ns"]
    completion = values["astra_completion_elapsed_ns"]
    resource_delay = values["astra_resource_delay_ns"]
    internal = values[
        "astra_internal_resource_serialization_wait_ns"]
    signed_delta = values["astra_signed_interference_delta_ns"]
    if dependency < 0 or solo < dependency or actual < dependency:
        raise LiveAstraCollectError(
            f"runtime.{prefix} violates ASTRA dependency timing bounds")
    expected = {
        "astra_completion_elapsed_ns": actual,
        "astra_resource_delay_ns": actual - dependency,
        "astra_internal_resource_serialization_wait_ns": (
            solo - dependency),
        "astra_signed_interference_delta_ns": actual - solo,
    }
    for field, expected_value in expected.items():
        if values[field] != expected_value:
            raise LiveAstraCollectError(
                f"runtime.{prefix}.{field} violates ASTRA timing algebra: "
                f"{values[field]} != {expected_value}")
    if resource_delay != internal + signed_delta:
        raise LiveAstraCollectError(
            f"runtime.{prefix} violates ASTRA interference identity")
    if completion != actual:
        raise AssertionError("validated ASTRA completion identity changed")


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise LiveAstraCollectError(
            f"{name} must be an integer >= {minimum}")
    return value


def _require_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiveAstraCollectError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise LiveAstraCollectError(f"{name} must be finite")
    return number


def _parse_csv_int(
    row: Mapping[str, str],
    field: str,
    *,
    line_number: int,
    optional: bool = False,
) -> int | None:
    raw = row.get(field)
    if raw is None or raw == "":
        if optional:
            return None
        raise LiveAstraCollectError(
            f"requests.csv line {line_number} has no {field}")
    if raw.strip() != raw:
        raise LiveAstraCollectError(
            f"requests.csv line {line_number} has invalid {field}")
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise LiveAstraCollectError(
            f"requests.csv line {line_number} has invalid {field}") from exc
    if str(value) != raw:
        raise LiveAstraCollectError(
            f"requests.csv line {line_number} has non-canonical {field}")
    return value


def _parse_requests(path: Path) -> tuple[tuple[dict[str, object], ...], set[str]]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise LiveAstraCollectError(f"cannot read requests CSV {path}") from exc
    parsed = []
    seen = set()
    with handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if header is None:
            raise LiveAstraCollectError("requests.csv has no header")
        if len(header) != len(set(header)) or any(not field for field in header):
            raise LiveAstraCollectError(
                "requests.csv has duplicate or empty header fields")
        missing = _REQUIRED_REQUEST_FIELDS - set(header)
        if missing:
            raise LiveAstraCollectError(
                "requests.csv is missing fields: "
                + ", ".join(sorted(missing)))
        for line_number, raw_row in enumerate(reader, start=2):
            if None in raw_row:
                raise LiveAstraCollectError(
                    f"requests.csv line {line_number} has extra columns")
            session_id = raw_row["session_id"]
            if not session_id or session_id.strip() != session_id:
                raise LiveAstraCollectError(
                    f"requests.csv line {line_number} has invalid session_id")
            call_index = _parse_csv_int(
                raw_row, "sub_request_index", line_number=line_number)
            arrival = _parse_csv_int(
                raw_row, "arrival", line_number=line_number)
            completion = _parse_csv_int(
                raw_row, "end_time", line_number=line_number)
            latency = _parse_csv_int(
                raw_row, "latency", line_number=line_number)
            ttft = _parse_csv_int(
                raw_row, "TTFT", line_number=line_number)
            csv_tpot = _parse_csv_int(
                raw_row, "TPOT", line_number=line_number)
            output = _parse_csv_int(
                raw_row, "output", line_number=line_number)
            generated = _parse_csv_int(
                raw_row, "generated_tokens", line_number=line_number)
            numeric = (
                call_index, arrival, completion, latency, ttft, csv_tpot,
                output, generated,
            )
            if any(value is None or value < 0 for value in numeric):
                raise LiveAstraCollectError(
                    f"requests.csv line {line_number} has a negative field")
            assert call_index is not None
            assert arrival is not None
            assert completion is not None
            assert latency is not None
            assert ttft is not None
            assert csv_tpot is not None
            assert output is not None
            assert generated is not None
            identity = (session_id, call_index)
            if identity in seen:
                raise LiveAstraCollectError(
                    f"requests.csv has duplicate identity {identity!r}")
            seen.add(identity)
            if output <= 0 or generated != output:
                raise LiveAstraCollectError(
                    f"requests.csv line {line_number} has invalid output")
            if completion - arrival != latency or ttft > latency:
                raise LiveAstraCollectError(
                    f"requests.csv line {line_number} has inconsistent timing")
            post_first = latency - ttft
            if output == 1:
                if post_first != 0 or csv_tpot != 0:
                    raise LiveAstraCollectError(
                        f"requests.csv line {line_number} has invalid "
                        "one-token TPOT")
                exact_tpot = None
            else:
                if csv_tpot != post_first // (output - 1):
                    raise LiveAstraCollectError(
                        f"requests.csv line {line_number} has inconsistent "
                        "TPOT")
                exact_tpot = Fraction(post_first, output - 1)
            parsed.append({
                "session_id": session_id,
                "call_index": call_index,
                "arrival_ns": arrival,
                "completion_ns": completion,
                "ttft_ns": ttft,
                "tpot_ns": exact_tpot,
                "output_tokens": output,
                "raw": raw_row,
                "line_number": line_number,
            })
    return tuple(parsed), set(header)


def _nearest_rank(
    ordered: Sequence[int | Fraction],
    percentile: float,
) -> int | Fraction:
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _distribution(values: Iterable[int | Fraction]) -> dict[str, object]:
    exact = tuple(
        value if isinstance(value, Fraction) else Fraction(value)
        for value in values
    )
    if not exact:
        raise LiveAstraCollectError("cannot summarize an empty distribution")
    ordered = tuple(sorted(exact))
    return {
        "count": len(ordered),
        "mean_ns": float(sum(ordered, Fraction()) / len(ordered)),
        "minimum_ns": float(ordered[0]),
        "p50_ns": float(_nearest_rank(ordered, 0.50)),
        "p95_ns": float(_nearest_rank(ordered, 0.95)),
        "p99_ns": float(_nearest_rank(ordered, 0.99)),
        "maximum_ns": float(ordered[-1]),
        "percentile_method": "inclusive_nearest_rank",
    }


def _wait_distribution(
    rows: Sequence[Mapping[str, object]],
    header: set[str],
    csv_field: str,
) -> dict[str, object] | None:
    if csv_field not in header:
        return None
    values = []
    missing = 0
    for row in rows:
        raw = row["raw"]
        assert isinstance(raw, Mapping)
        parsed = _parse_csv_int(
            raw,
            csv_field,
            line_number=int(row["line_number"]),
            optional=True,
        )
        if parsed is None:
            missing += 1
        else:
            if parsed < 0:
                raise LiveAstraCollectError(
                    f"{csv_field} cannot be negative")
            values.append(parsed)
    summary: dict[str, object] = {
        "reported_count": len(values),
        "missing_count": missing,
    }
    if values:
        ordered = tuple(sorted(values))
        summary.update({
            "sum_ns": sum(values),
            "mean_ns": sum(values) / len(values),
            "p50_ns": int(_nearest_rank(ordered, 0.50)),
            "p95_ns": int(_nearest_rank(ordered, 0.95)),
            "p99_ns": int(_nearest_rank(ordered, 0.99)),
            "maximum_ns": ordered[-1],
            "percentile_method": "inclusive_nearest_rank",
        })
    return summary


def _source_distribution(
    rows: Sequence[Mapping[str, object]],
    header: set[str],
    csv_field: str,
) -> dict[str, object] | None:
    if csv_field not in header:
        return None
    counts: dict[str, int] = {}
    missing = 0
    for row in rows:
        raw_row = row["raw"]
        assert isinstance(raw_row, Mapping)
        raw = raw_row.get(csv_field)
        if not isinstance(raw, str) or not raw:
            missing += 1
            continue
        counts[raw] = counts.get(raw, 0) + 1
    denominator = len(rows)
    return {
        "denominator_measurement_resumes": denominator,
        "reported_count": denominator - missing,
        "missing_count": missing,
        "counts": dict(sorted(counts.items())),
        "fraction_of_measurement_resumes": {
            key: value / denominator
            for key, value in sorted(counts.items())
        },
    }


def _prefix_token_accounting(
    rows: Sequence[Mapping[str, object]],
    header: set[str],
) -> dict[str, object] | None:
    fields = (
        "agentic_kv_hit_tokens",
        "agentic_kv_recompute_tokens",
    )
    if not all(field in header for field in fields):
        return None
    totals = {field: 0 for field in fields}
    reported = 0
    missing = 0
    for row in rows:
        raw = row["raw"]
        assert isinstance(raw, Mapping)
        values = [
            _parse_csv_int(
                raw,
                field,
                line_number=int(row["line_number"]),
                optional=True,
            )
            for field in fields
        ]
        if any(value is None for value in values):
            missing += 1
            continue
        reported += 1
        for field, value in zip(fields, values):
            assert value is not None
            if value < 0:
                raise LiveAstraCollectError(
                    f"{field} cannot be negative")
            totals[field] += value
    result: dict[str, object] = {
        "reported_count": reported,
        "missing_count": missing,
        "hit_tokens": totals["agentic_kv_hit_tokens"],
        "recompute_tokens": totals["agentic_kv_recompute_tokens"],
    }
    denominator = sum(totals.values())
    if denominator:
        result["hit_fraction_of_hit_plus_recompute_tokens"] = (
            totals["agentic_kv_hit_tokens"] / denominator)
    return result


def _assert_close(actual: object, expected: object, name: str) -> None:
    if isinstance(expected, str):
        if actual != expected:
            raise LiveAstraCollectError(
                f"{name} mismatch: {actual!r} != {expected!r}")
        return
    if isinstance(expected, int):
        if type(actual) is not int or actual != expected:
            raise LiveAstraCollectError(
                f"{name} mismatch: {actual!r} != {expected!r}")
        return
    expected_number = _require_number(expected, f"{name}.expected")
    actual_number = _require_number(actual, name)
    if not math.isclose(
        actual_number,
        expected_number,
        rel_tol=1e-12,
        abs_tol=1e-6,
    ):
        raise LiveAstraCollectError(
            f"{name} mismatch: {actual_number!r} != {expected_number!r}")


def _crosscheck_metrics(
    metrics: Mapping[str, object],
    measured: Sequence[Mapping[str, object]],
    resumes: Sequence[Mapping[str, object]],
    measurement_session_ids: Sequence[str],
) -> tuple[dict[str, object], int]:
    if not measured or not resumes:
        raise LiveAstraCollectError(
            "measurement roster must contain requests and resumes")
    tpot_rows = tuple(row for row in measured if row["tpot_ns"] is not None)
    resume_tpot_rows = tuple(
        row for row in resumes if row["tpot_ns"] is not None)
    if not tpot_rows or not resume_tpot_rows:
        raise LiveAstraCollectError(
            "measurement roster must contain TPOT-eligible requests")

    ttft_slo_ns = _require_int(
        metrics.get("ttft_slo_ns"), "metrics.ttft_slo_ns", minimum=1)
    tpot_slo_ns = _require_int(
        metrics.get("tpot_slo_ns"), "metrics.tpot_slo_ns", minimum=1)
    passed = []
    passed_resumes = []
    pass_by_session: dict[str, list[bool]] = {
        session_id: [] for session_id in measurement_session_ids
    }
    for row in measured:
        tpot = row["tpot_ns"]
        passed_slo = (
            int(row["ttft_ns"]) <= ttft_slo_ns
            and (
                tpot is None
                or tpot <= tpot_slo_ns
            )
        )
        pass_by_session[str(row["session_id"])].append(passed_slo)
        if passed_slo:
            passed.append(row)
            if int(row["call_index"]) > 0:
                passed_resumes.append(row)
    passed_sessions = sum(
        1 for values in pass_by_session.values()
        if values and all(values)
    )

    window_start = min(int(row["arrival_ns"]) for row in measured)
    window_end = max(int(row["completion_ns"]) for row in measured)
    window_duration = window_end - window_start
    if window_duration <= 0:
        raise LiveAstraCollectError(
            "measurement window has non-positive duration")
    scale = NANOSECONDS_PER_SECOND / window_duration
    expected = {
        "measurement_request_count": len(measured),
        "resume_request_count": len(resumes),
        "tpot_eligible_request_count": len(tpot_rows),
        "resume_tpot_eligible_request_count": len(resume_tpot_rows),
        "joint_slo_pass_count": len(passed),
        "joint_slo_fail_count": len(measured) - len(passed),
        "resume_joint_slo_pass_count": len(passed_resumes),
        "resume_joint_slo_fail_count": len(resumes) - len(passed_resumes),
        "joint_slo_pass_output_tokens": sum(
            int(row["output_tokens"]) for row in passed),
        "joint_slo_pass_session_count": passed_sessions,
        "joint_slo_fail_session_count": (
            len(measurement_session_ids) - passed_sessions),
        "window_start_ns": window_start,
        "window_end_ns": window_end,
        "window_duration_ns": window_duration,
        "operational_request_goodput_per_second": len(passed) * scale,
        "operational_resume_goodput_per_second": (
            len(passed_resumes) * scale),
        "operational_token_goodput_per_second": sum(
            int(row["output_tokens"]) for row in passed) * scale,
        "operational_session_goodput_per_second": passed_sessions * scale,
    }
    check_count = 0
    for key, value in expected.items():
        if key not in metrics:
            raise LiveAstraCollectError(f"metrics is missing {key}")
        _assert_close(metrics[key], value, f"metrics.{key}")
        check_count += 1

    distributions = {
        "resume_ttft_ns": _distribution(
            int(row["ttft_ns"]) for row in resumes),
        "tpot_ns": _distribution(
            row["tpot_ns"] for row in tpot_rows),
        "resume_tpot_ns": _distribution(
            row["tpot_ns"] for row in resume_tpot_rows),
    }
    for name, expected_distribution in distributions.items():
        actual = metrics.get(name)
        if not isinstance(actual, Mapping):
            raise LiveAstraCollectError(
                f"metrics.{name} is not a distribution")
        for key, expected_value in expected_distribution.items():
            if key not in actual:
                raise LiveAstraCollectError(
                    f"metrics.{name} is missing {key}")
            _assert_close(
                actual[key],
                expected_value,
                f"metrics.{name}.{key}",
            )
            check_count += 1

    performance = {
        "measurement_session_count": len(measurement_session_ids),
        "measurement_request_count": len(measured),
        "resume_request_count": len(resumes),
        "tpot_eligible_request_count": len(tpot_rows),
        "resume_tpot_eligible_request_count": len(resume_tpot_rows),
        "ttft_slo_ns": ttft_slo_ns,
        "tpot_slo_ns": tpot_slo_ns,
        "resume_ttft_ns": distributions["resume_ttft_ns"],
        "tpot_ns": distributions["tpot_ns"],
        "resume_tpot_ns": distributions["resume_tpot_ns"],
        "joint_slo_pass_count": len(passed),
        "joint_slo_fail_count": len(measured) - len(passed),
        "resume_joint_slo_pass_count": len(passed_resumes),
        "resume_joint_slo_fail_count": len(resumes) - len(passed_resumes),
        "joint_slo_pass_output_tokens": expected[
            "joint_slo_pass_output_tokens"],
        "joint_slo_pass_session_count": passed_sessions,
        "joint_slo_fail_session_count": (
            len(measurement_session_ids) - passed_sessions),
        "joint_slo_pass_fraction": len(passed) / len(measured),
        "resume_joint_slo_pass_fraction": (
            len(passed_resumes) / len(resumes)),
        "window_start_ns": window_start,
        "window_end_ns": window_end,
        "window_duration_ns": window_duration,
        "operational_request_goodput_per_second": expected[
            "operational_request_goodput_per_second"],
        "operational_resume_goodput_per_second": expected[
            "operational_resume_goodput_per_second"],
        "operational_token_goodput_per_second": expected[
            "operational_token_goodput_per_second"],
        "operational_session_goodput_per_second": expected[
            "operational_session_goodput_per_second"],
        "raw_request_throughput_per_second": len(measured) * scale,
        "raw_resume_throughput_per_second": len(resumes) * scale,
        "raw_output_token_throughput_per_second": sum(
            int(row["output_tokens"]) for row in measured) * scale,
        "raw_session_throughput_per_second": (
            len(measurement_session_ids) * scale),
    }
    return performance, check_count


def _resource_queue_class(resource: str) -> str:
    lowered = resource.lower()
    if "ssd-pool:read" in lowered or "ssd" in lowered and "read" in lowered:
        return "ssd_read"
    if "ssd-pool:write" in lowered or "ssd" in lowered and "write" in lowered:
        return "ssd_write"
    if "pcie" in lowered:
        return "pcie"
    if "dram" in lowered:
        return "dram"
    if "pd-fabric" in lowered:
        return "pd_fabric"
    return "other"


def _resource_queue_summary(runtime: Mapping[str, object]) -> object:
    queues = runtime.get("resource_queues", _MISSING)
    if queues is _MISSING:
        return _MISSING
    if not isinstance(queues, Mapping):
        raise LiveAstraCollectError("runtime.resource_queues is not an object")
    grouped: dict[str, dict[str, int]] = {}
    for resource, raw in queues.items():
        if not isinstance(resource, str) or not isinstance(raw, Mapping):
            raise LiveAstraCollectError(
                "runtime.resource_queues has an invalid row")
        category = _resource_queue_class(resource)
        row = grouped.setdefault(category, {
            "resource_count": 0,
            "jobs": 0,
            "service_demand_ns": 0,
            "maximum_busy_until_ns": 0,
        })
        row["resource_count"] += 1
        if "jobs" in raw:
            row["jobs"] += _require_int(
                raw["jobs"], f"resource_queues.{resource}.jobs")
        if "service_demand_ns" in raw:
            row["service_demand_ns"] += _require_int(
                raw["service_demand_ns"],
                f"resource_queues.{resource}.service_demand_ns",
            )
        if "busy_until_ns" in raw:
            row["maximum_busy_until_ns"] = max(
                row["maximum_busy_until_ns"],
                _require_int(
                    raw["busy_until_ns"],
                    f"resource_queues.{resource}.busy_until_ns",
                ),
            )
    return dict(sorted(grouped.items()))


def _numeric_leaf_summary(value: object, name: str) -> dict[str, object]:
    leaves = []

    def visit(current: object, path: str) -> None:
        if isinstance(current, Mapping):
            for key, child in current.items():
                visit(child, f"{path}.{key}")
        elif type(current) is int:
            if current < 0:
                raise LiveAstraCollectError(f"{path} cannot be negative")
            leaves.append(current)
        else:
            raise LiveAstraCollectError(
                f"{path} must contain only non-negative integer leaves")

    visit(value, name)
    if not leaves:
        return {"leaf_count": 0}
    return {
        "leaf_count": len(leaves),
        "maximum_bytes": max(leaves),
        "sum_of_reported_leaf_peaks_bytes": sum(leaves),
    }


def _baseline_bottlenecks(
    runtime: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    ssd = _select_paths(runtime, _BASELINE_SSD_PATHS)
    if ssd:
        result["ssd"] = ssd
    dram = _select_paths(runtime, _BASELINE_DRAM_PATHS)
    if dram:
        result["dram"] = dram
    fabric = _select_paths(runtime, _BASELINE_FABRIC_PATHS)
    if fabric:
        result["pcie_and_fabric"] = fabric
    queues = _resource_queue_summary(runtime)
    if queues is not _MISSING:
        result["resource_queues"] = queues
    return result


def _require_hbf_drain_int(
    runtime: Mapping[str, object],
    source_path: str,
    *,
    minimum: int = 0,
) -> int:
    value = _lookup(runtime, source_path)
    if value is _MISSING:
        raise LiveAstraCollectError(
            "runtime is missing required HBF prefill-drain field "
            f"{source_path}"
        )
    return _require_int(
        value,
        f"runtime.{source_path}",
        minimum=minimum,
    )


def _require_hbf_drain_pending_count(
    runtime: Mapping[str, object],
    source_path: str,
    *,
    sequence: bool,
) -> int:
    value = _lookup(runtime, source_path)
    if value is _MISSING:
        raise LiveAstraCollectError(
            "runtime is missing required HBF prefill-drain field "
            f"{source_path}"
        )
    valid = isinstance(value, list) if sequence else isinstance(value, Mapping)
    if not valid:
        container = "list" if sequence else "object"
        raise LiveAstraCollectError(
            f"runtime.{source_path} must be a {container}"
        )
    return len(value)


def _zero_safe_ratio(
    numerator: int,
    denominator: int,
    name: str,
) -> float:
    if denominator == 0:
        if numerator != 0:
            raise LiveAstraCollectError(
                f"{name} has a nonzero numerator with a zero denominator"
            )
        return 0.0
    return numerator / denominator


def _hbf_prefill_drain(
    runtime: Mapping[str, object],
) -> dict[str, object]:
    policy = {
        "tail_tokens": _require_hbf_drain_int(
            runtime, "options.prefill_drain_tail_tokens"),
        "min_tokens": _require_hbf_drain_int(
            runtime, "options.prefill_drain_min_tokens"),
    }
    pool_policy = {
        "tail_tokens": _require_hbf_drain_int(
            runtime, "adapter.pool.prefill_drain_tail_tokens"),
        "min_tokens": _require_hbf_drain_int(
            runtime, "adapter.pool.prefill_drain_min_tokens"),
    }
    if policy != pool_policy:
        raise LiveAstraCollectError(
            "HBF prefill-drain policy options disagree with the pool report"
        )

    pool = {
        output_name: _require_hbf_drain_int(runtime, source_path)
        for output_name, source_path in
        _HBF_PREFILL_DRAIN_POOL_PATHS.items()
    }
    lifecycle = {
        output_name: _require_hbf_drain_int(runtime, source_path)
        for output_name, source_path in
        _HBF_PREFILL_DRAIN_LIFECYCLE_PATHS.items()
    }
    kv_bytes_per_token = _require_hbf_drain_int(
        runtime,
        "adapter.lifecycle.kv_bytes_per_token",
        minimum=1,
    )
    pending = {
        output_name: _require_hbf_drain_pending_count(
            runtime,
            source_path,
            sequence=output_name == "lifecycle_active_job_count",
        )
        for output_name, source_path in
        _HBF_PREFILL_DRAIN_PENDING_PATHS.items()
    }
    nonzero_pending = {
        name: count for name, count in pending.items() if count != 0
    }
    if nonzero_pending:
        raise LiveAstraCollectError(
            "HBF prefill-drain report is not quiescent: "
            f"{nonzero_pending}"
        )

    pool_submitted = _require_hbf_drain_int(
        runtime, "adapter.pool.metrics.submitted_requests")
    pool_completed_requests = _require_hbf_drain_int(
        runtime, "adapter.pool.metrics.completed_requests")
    adapter_hbf_requests = _require_hbf_drain_int(
        runtime, "adapter.metrics.hbf_requests")
    adapter_hbf_completions = _require_hbf_drain_int(
        runtime, "adapter.metrics.hbf_completions")

    if pool["candidates"] != pool["claimed"]:
        raise LiveAstraCollectError(
            "HBF prefill-drain pool candidate/claim accounting mismatch"
        )
    if pool["candidates"] != pool["completed"] + pool["fallbacks"]:
        raise LiveAstraCollectError(
            "HBF prefill-drain pool terminal accounting mismatch"
        )
    if pool["candidates"] > pool_submitted:
        raise LiveAstraCollectError(
            "HBF prefill-drain candidates exceed submitted HBF requests"
        )
    if lifecycle["candidates"] != sum((
        lifecycle["started"],
        lifecycle["satisfied"],
        lifecycle["wait_existing_append"],
        lifecycle["capacity_fallback"],
    )):
        raise LiveAstraCollectError(
            "HBF prefill-drain lifecycle outcome accounting mismatch"
        )
    if lifecycle["started"] != (
        lifecycle["committed"] + lifecycle["stale"]
    ):
        raise LiveAstraCollectError(
            "HBF prefill-drain lifecycle completion accounting mismatch"
        )
    if pool["started"] != lifecycle["started"]:
        raise LiveAstraCollectError(
            "HBF prefill-drain pool/lifecycle start accounting mismatch"
        )
    if pool["fallbacks"] != lifecycle["capacity_fallback"]:
        raise LiveAstraCollectError(
            "HBF prefill-drain pool/lifecycle fallback accounting mismatch"
        )
    if pool["completed"] != (
        lifecycle["satisfied"] + lifecycle["committed"]
    ):
        raise LiveAstraCollectError(
            "HBF prefill-drain pool/lifecycle completion accounting mismatch"
        )
    if pool_submitted != adapter_hbf_requests:
        raise LiveAstraCollectError(
            "HBF prefill-drain submitted request accounting mismatch"
        )
    if pool_completed_requests != adapter_hbf_completions:
        raise LiveAstraCollectError(
            "HBF prefill-drain completed request accounting mismatch"
        )

    derived = {
        "candidate_fraction": _zero_safe_ratio(
            pool["candidates"],
            pool_submitted,
            "HBF prefill-drain candidate fraction",
        ),
        "mean_wait_ms": (
            _zero_safe_ratio(
                pool["wait_ns"],
                pool["candidates"],
                "HBF prefill-drain mean wait",
            )
            / 1_000_000
        ),
        "fallback_fraction": _zero_safe_ratio(
            pool["fallbacks"],
            pool["candidates"],
            "HBF prefill-drain fallback fraction",
        ),
        "logical_traffic_gib": (
            pool["logical_tokens"]
            * kv_bytes_per_token
            / (1024 ** 3)
        ),
    }
    return {
        "policy": policy,
        "pool": pool,
        "lifecycle": lifecycle,
        "kv_bytes_per_token": kv_bytes_per_token,
        "pending": pending,
        "derived": derived,
    }


def _hbf_bottlenecks(runtime: Mapping[str, object]) -> dict[str, object]:
    _validate_hbf_astra_timing_metrics(
        runtime, "adapter.pool.metrics")
    _validate_hbf_astra_timing_metrics(
        runtime, "adapter.lifecycle.metrics")
    result: dict[str, object] = {
        "prefill_drain": _hbf_prefill_drain(runtime),
    }
    layout = runtime.get("layout")
    if isinstance(layout, Mapping):
        result["layout"] = dict(layout)
    routing = _select_paths(runtime, _HBF_ROUTING_PATHS)
    if routing:
        result["routing"] = routing
    capacity = _select_paths(runtime, _HBF_CAPACITY_PATHS)
    lpddr_peak = _lookup(runtime, "adapter.pool.lpddr_peak_bytes_by_card")
    if lpddr_peak is not _MISSING:
        capacity["lpddr_peak_bytes_by_card_summary"] = (
            _numeric_leaf_summary(
                lpddr_peak,
                "adapter.pool.lpddr_peak_bytes_by_card",
            )
        )
    for prefix in ("migration", "append"):
        logical = capacity.get(f"{prefix}_logical_bytes")
        physical = capacity.get(f"{prefix}_physical_bytes")
        if (
            type(logical) is int
            and logical > 0
            and type(physical) is int
            and physical >= 0
        ):
            capacity[f"{prefix}_physical_to_logical_ratio"] = (
                physical / logical)
    if capacity:
        result["capacity"] = capacity
    attention = _select_paths(runtime, _HBF_ATTENTION_PATHS)
    if attention:
        result["attention"] = attention
    network = _select_paths(runtime, _HBF_NETWORK_PATHS)
    if network:
        result["network"] = network
    return result


def _copy_if_present(
    output: dict[str, object],
    output_name: str,
    root: Mapping[str, object],
    source_path: str,
) -> None:
    value = _lookup(root, source_path)
    if value is not _MISSING:
        output[output_name] = value


def _validity_fields(
    session: Mapping[str, object],
    runtime: Mapping[str, object],
    *,
    roster_audit: Mapping[str, object],
    runtime_kind: object,
    artifact_count: int,
    request_count: int,
    measured_count: int,
    resume_count: int,
    metric_crosscheck_count: int,
) -> dict[str, object]:
    validity: dict[str, object] = {
        "verified_artifact_count": artifact_count,
        "parsed_request_count": request_count,
        "measurement_request_count": measured_count,
        "measurement_resume_request_count": resume_count,
        "headline_metric_crosscheck_count": metric_crosscheck_count,
        "headline_metric_crosscheck_mismatch_count": 0,
    }
    validity.update(roster_audit)
    for output_name, source_path in {
        "session_timing_checked_requests": (
            "validation.timing.checked_requests"),
        "session_timing_passed": "validation.timing.passed",
        "session_timing_violation_count": "validation.timing.violations",
        "session_timing_warning_count": "validation.timing.warnings",
        "measurement_complete": "measurement_window.measurement_complete",
        "measurement_boundary_complete": (
            "measurement_window.measurement_boundary_complete"),
        "measurement_early_stopped": (
            "measurement_window.measurement_early_stopped"),
    }.items():
        value = _lookup(session, source_path)
        if value is _MISSING:
            continue
        if output_name.endswith("_count") and isinstance(value, list):
            validity[output_name] = len(value)
        else:
            validity[output_name] = value

    if runtime_kind == "full_model_hbf":
        for output_name, source_path in {
            "adapter_pending_router_completions": (
                "adapter.pending_router_completion_count"),
            "adapter_pending_hbf_turn_finalizations": (
                "adapter.pending_hbf_turn_finalization_count"),
            "adapter_pending_gpu_hbm_events": (
                "adapter.pending_gpu_hbm_event_count"),
            "adapter_staged_hbf_admissions": (
                "adapter.staged_hbf_admission_count"),
            "adapter_pending_prefill_drain_session_count": (
                "adapter.pending_prefill_drain_request_by_session"),
            "adapter_active_prefill_drain_job_count": (
                "adapter.active_prefill_drain_request_by_job"),
            "adapter_waiting_prefill_drain_append_session_count": (
                "adapter."
                "waiting_prefill_drain_append_jobs_by_session"),
            "multiplexer_pending_jobs": (
                "adapter.multiplexer.pending_job_count"),
            "multiplexer_ready_jobs": (
                "adapter.multiplexer.ready_job_count"),
            "multiplexer_quarantined_dispatches": (
                "adapter.multiplexer.quarantined_dispatch_count"),
            "multiplexer_completed_jobs": (
                "adapter.multiplexer.completed_job_count"),
            "lifecycle_pending_jobs": "adapter.lifecycle.pending_job_count",
            "lifecycle_active_prefill_drain_pending_job_count": (
                "adapter.lifecycle."
                "active_prefill_drain_pending_job_ids"),
            "lifecycle_external_issued_dispatches": (
                "adapter.lifecycle.external_issued_dispatch_count"),
            "lifecycle_external_completed_dispatches": (
                "adapter.lifecycle.external_completed_dispatch_count"),
            "lifecycle_external_undrained_dispatches": (
                "adapter.lifecycle.external_undrained_dispatch_count"),
            "pool_pending_batches": "adapter.pool.pending_batch_count",
            "pool_pending_launches": "adapter.pool.pending_launch_count",
            "pool_external_issued_dispatches": (
                "adapter.pool.external_issued_dispatch_count"),
            "pool_external_undrained_dispatches": (
                "adapter.pool.external_undrained_dispatch_count"),
            "gpu_hbm_rejected_events": (
                "gpu_hbm_bridge.metrics.rejected_events"),
            "gpu_hbm_pending_colocated_claim_count": (
                "gpu_hbm_bridge.pending_colocated_claims"),
            "gpu_hbm_pending_pd_recompute_binding_count": (
                "gpu_hbm_bridge.pending_pd_recompute_bindings"),
            "gpu_hbm_pending_pd_decode_reservation_count": (
                "gpu_hbm_bridge.pending_pd_decode_reservations"),
        }.items():
            value = _lookup(runtime, source_path)
            if value is _MISSING:
                continue
            if (
                output_name.endswith("_count")
                and isinstance(value, (list, Mapping))
            ):
                validity[output_name] = len(value)
            else:
                validity[output_name] = value
    else:
        for output_name, source_path in {
            "external_fabric_issued_jobs": "external_fabric.issued_jobs",
            "external_fabric_completed_jobs": (
                "external_fabric.completed_jobs"),
            "external_fabric_censored_jobs": (
                "external_fabric.censored_jobs"),
            "external_fabric_pending_jobs": "external_fabric.pending_jobs",
            "bridge_external_fabric_pending_jobs": (
                "online_resource_bridge.external_fabric_pending_jobs"),
            "bridge_open_astra_windows": (
                "online_resource_bridge.open_astra_window_count"),
            "bridge_pending_direct_fabric_prepare_locks": (
                "online_resource_bridge.pending_direct_fabric_prepare_locks"),
            "bridge_transient_dram_capacity_violations": (
                "online_resource_bridge."
                "transient_dram_capacity_violation_count"),
            "cutoff_outstanding_dma_jobs": (
                "measurement_cutoff_dma_tail.outstanding_jobs"),
            "cutoff_measurement_censored": (
                "measurement_cutoff_dma_tail.measurement_censored"),
        }.items():
            _copy_if_present(
                validity, output_name, runtime, source_path)

    strict_oracle = session.get("strict_infinite_hbm_oracle")
    if isinstance(strict_oracle, Mapping):
        for output_name, key in {
            "oracle_enabled": "enabled",
            "oracle_passed": "passed",
            "oracle_checked_reusable_resumes": "checked_reusable_resumes",
        }.items():
            if key in strict_oracle:
                validity[output_name] = strict_oracle[key]
        per_instance = strict_oracle.get("per_instance")
        if isinstance(per_instance, Mapping):
            validity["oracle_nonbinding_instance_count"] = sum(
                1
                for row in per_instance.values()
                if isinstance(row, Mapping)
                and row.get("nonbinding") is True
            )
            validity["oracle_instance_count"] = len(per_instance)
        zero = strict_oracle.get("zero_counter_invariants")
        if isinstance(zero, Mapping):
            validity["oracle_zero_invariant_count"] = len(zero)
            validity["oracle_nonzero_invariant_count"] = sum(
                1 for value in zero.values() if value != 0)
        violations = strict_oracle.get("violations")
        if isinstance(violations, list):
            validity["oracle_violation_count"] = len(violations)
    return validity


def _session_id_roster(
    raw: object,
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if (
        not isinstance(raw, list)
        or (not raw and not allow_empty)
        or any(not isinstance(value, str) or not value for value in raw)
        or len(raw) != len(set(raw))
    ):
        raise LiveAstraCollectError(f"{name} is invalid")
    return tuple(raw)


def _require_exact_report_value(
    root: Mapping[str, object],
    path: str,
    expected: object,
) -> None:
    actual = _lookup(root, path)
    if actual is _MISSING:
        raise LiveAstraCollectError(
            f"session report is missing {path}")
    if isinstance(expected, bool):
        matches = type(actual) is bool and actual is expected
    elif isinstance(expected, int):
        matches = type(actual) is int and actual == expected
    else:
        matches = actual == expected
    if not matches:
        raise LiveAstraCollectError(
            f"session report {path} is inconsistent with strict full drain")


def _native_full_drain_session_order(
    requests: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    completion_by_session: dict[str, int] = {}
    call_indices_by_session: dict[str, set[int]] = {}
    for row in requests:
        session_id = str(row["session_id"])
        completion = int(row["completion_ns"])
        call_index = int(row["call_index"])
        completion_by_session[session_id] = max(
            completion,
            completion_by_session.get(session_id, completion),
        )
        call_indices_by_session.setdefault(session_id, set()).add(call_index)
    for session_id, call_indices in call_indices_by_session.items():
        if sorted(call_indices) != list(range(len(call_indices))):
            raise LiveAstraCollectError(
                "requests.csv has a non-contiguous completed call roster "
                f"for session {session_id!r}")
    return tuple(sorted(
        completion_by_session,
        key=lambda session_id: (
            completion_by_session[session_id],
            session_id,
        ),
    ))


def _validate_default_full_drain_superset(
    *,
    roster: Sequence[str],
    required_roster: Sequence[str],
    session: Mapping[str, object],
    requests: Sequence[Mapping[str, object]],
    expected_request_count: int,
    expected_session_count: int,
) -> dict[str, object]:
    """Validate the only allowed session-report superset relation.

    The default serving run selects every completed session in completion
    order when both completion-count controls are zero.  In that mode the
    session report intentionally describes the full drain, while the live
    comparison result retains its smaller preregistered measurement roster.
    """

    if len(required_roster) <= len(roster):
        raise LiveAstraCollectError(
            "session report measurement roster disagrees with result")
    roster_set = set(roster)
    required_set = set(required_roster)
    if not roster_set < required_set:
        raise LiveAstraCollectError(
            "session report is not a strict full-drain superset of the "
            "preregistered measurement roster")
    if len(requests) != expected_request_count:
        raise LiveAstraCollectError(
            "strict full-drain request count disagrees with workload")

    target = _session_id_roster(
        _lookup(
            session,
            "measurement_window.measurement_target_session_ids",
        ),
        "session report full-drain target roster",
    )
    if target != required_roster:
        raise LiveAstraCollectError(
            "session report target and required full-drain rosters differ")
    warmup = _session_id_roster(
        _lookup(
            session,
            "measurement_window.measurement_warmup_session_ids",
        ),
        "session report full-drain warmup roster",
        allow_empty=True,
    )
    if warmup:
        raise LiveAstraCollectError(
            "strict full drain cannot contain warmup sessions")

    native_order = _native_full_drain_session_order(requests)
    if len(native_order) != expected_session_count:
        raise LiveAstraCollectError(
            "requests.csv full-drain session count disagrees with workload")
    if target != native_order:
        raise LiveAstraCollectError(
            "session report full-drain completion order disagrees with "
            "requests.csv")

    target_hash = _require_sha256(
        _lookup(
            session,
            "measurement_window.measurement_target_session_ids_hash",
        ),
        "session report target roster hash",
    )
    required_hash = _require_sha256(
        _lookup(
            session,
            "measurement_window.measurement_required_session_ids_hash",
        ),
        "session report required roster hash",
    )
    warmup_hash = _require_sha256(
        _lookup(
            session,
            "measurement_window.measurement_warmup_session_ids_hash",
        ),
        "session report warmup roster hash",
    )
    if (
        target_hash != _stable_json_sha256(list(target))
        or required_hash != _stable_json_sha256(list(required_roster))
        or warmup_hash != _stable_json_sha256(list(warmup))
    ):
        raise LiveAstraCollectError(
            "session report full-drain roster hash is inconsistent")

    native_start_ns = min(int(row["arrival_ns"]) for row in requests)
    native_end_ns = max(int(row["completion_ns"]) for row in requests)
    native_duration_ns = native_end_ns - native_start_ns
    if native_duration_ns <= 0:
        raise LiveAstraCollectError(
            "strict full-drain requests have a non-positive duration")

    exact_values = {
        "measurement_window.measurement_cohort_selection": (
            "completion_order"),
        "measurement_window.warmup_completions_requested": 0,
        "measurement_window.measure_completions_requested": 0,
        "measurement_window.measurement_complete": True,
        "measurement_window.measurement_boundary_complete": True,
        "measurement_window.measurement_early_stopped": False,
        "measurement_window.warmup_complete": True,
        "measurement_window.measurement_warmup_session_count": 0,
        "measurement_window.measurement_warmup_completed_sessions": 0,
        "measurement_window.warmup_completions_observed": 0,
        "measurement_window.measurement_prefix_id_overlap_count": 0,
        "measurement_window.measurement_target_session_count": (
            expected_session_count),
        "measurement_window.measurement_target_completed_sessions": (
            expected_session_count),
        "measurement_window.measurement_required_session_count": (
            expected_session_count),
        "measurement_window.measurement_required_completed_sessions": (
            expected_session_count),
        "measurement_window.measure_completions_observed": (
            expected_session_count),
        "measurement_window.measurement_start_ns": native_start_ns,
        "measurement_window.measurement_end_ns": native_end_ns,
        "measurement_window.measurement_duration_ns": native_duration_ns,
        "validation.timing.checked_requests": expected_request_count,
        "validation.timing.passed": True,
        "throughput.completed_sessions_total": expected_session_count,
        "throughput.completed_sessions": expected_session_count,
        "throughput.completed_requests_total": expected_request_count,
        "throughput.completed_requests": expected_request_count,
        "throughput.completed_requests_in_session_cohort": (
            expected_request_count),
        "throughput.generated_tokens": sum(
            int(row["output_tokens"]) for row in requests),
        "throughput.generated_tokens_in_session_cohort": sum(
            int(row["output_tokens"]) for row in requests),
    }
    for path, expected in exact_values.items():
        _require_exact_report_value(session, path, expected)

    violations = _lookup(session, "validation.timing.violations")
    if not isinstance(violations, list) or violations:
        raise LiveAstraCollectError(
            "session report timing validation is not clean")

    return {
        "measurement_roster_relation": "strict_full_drain_superset",
        "session_report_full_drain_superset_verified": True,
        "session_report_full_drain_native_session_count": len(native_order),
        "session_report_full_drain_native_request_count": len(requests),
        "session_report_full_drain_preregistered_coverage_count": len(roster),
        "session_report_full_drain_crosscheck_count": (
            len(exact_values) + 7),
    }


def _measurement_roster(
    metrics: Mapping[str, object],
    campaign: Mapping[str, object],
    session: Mapping[str, object],
    requests: Sequence[Mapping[str, object]],
    *,
    expected_request_count: int,
    expected_session_count: int,
) -> tuple[tuple[str, ...], dict[str, object]]:
    raw = metrics.get("measurement_session_ids")
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(value, str) or not value for value in raw)
        or len(raw) != len(set(raw))
    ):
        raise LiveAstraCollectError(
            "metrics has an invalid measurement_session_ids roster")
    roster = tuple(raw)
    recorded_hash = campaign.get("measurement_session_ids_sha256")
    expected_hash = _require_sha256(
        recorded_hash,
        "campaign.measurement_session_ids_sha256",
    )
    if _stable_json_sha256(list(roster)) != expected_hash:
        raise LiveAstraCollectError(
            "measurement roster disagrees with campaign identity")

    required = _lookup(
        session,
        "measurement_window.measurement_required_session_ids",
    )
    required_roster = _session_id_roster(
        required,
        "session report measurement roster",
    )
    base_audit: dict[str, object] = {
        "measurement_roster_authoritative_source": (
            "result.metrics.measurement_session_ids"),
        "measurement_roster_ordered_hash_verified": True,
        "preregistered_measurement_session_count": len(roster),
        "session_report_measurement_session_count": len(required_roster),
    }
    if required_roster == roster:
        base_audit.update({
            "measurement_roster_relation": "exact",
            "session_report_full_drain_superset_verified": False,
        })
        return roster, base_audit

    full_drain_audit = _validate_default_full_drain_superset(
        roster=roster,
        required_roster=required_roster,
        session=session,
        requests=requests,
        expected_request_count=expected_request_count,
        expected_session_count=expected_session_count,
    )
    base_audit.update(full_drain_audit)
    return roster, base_audit


def _runtime_guard_contract_for_cell(
    campaign: Mapping[str, object],
    entry: Mapping[str, object],
    result: Mapping[str, object],
    cell_id: str,
) -> Mapping[str, object] | None:
    required = campaign.get("runtime_guard_validation_required", False)
    if type(required) is not bool:
        raise LiveAstraCollectError(
            "campaign.runtime_guard_validation_required must be boolean")
    if (
        campaign.get("scenario_id") in _RUNTIME_GUARDED_SCENARIO_IDS
        and not required
    ):
        raise LiveAstraCollectError(
            f"{cell_id} stress campaign is missing mandatory runtime guard "
            "validation")
    recorded_entry = entry.get("runtime_guard_contract")
    recorded_result = result.get("runtime_guard_contract")
    contracts = campaign.get("runtime_guard_contracts")
    if not required:
        if contracts is not None or recorded_entry is not None or (
                recorded_result is not None):
            raise LiveAstraCollectError(
                f"{cell_id} has an unexpected runtime guard contract")
        return None
    if not isinstance(contracts, list) or not contracts:
        raise LiveAstraCollectError(
            "campaign requires runtime guard validation but has no contracts")

    seed = entry.get("seed")
    rate = entry.get("rate")
    matches = [
        contract
        for contract in contracts
        if isinstance(contract, Mapping)
        and contract.get("seed") == seed
        and contract.get("offered_session_rate_per_second") == rate
    ]
    if len(matches) != 1:
        raise LiveAstraCollectError(
            f"{cell_id} does not resolve to exactly one campaign runtime "
            "guard contract")
    contract = matches[0]
    cutoff = _require_int(
        contract.get("last_external_guard_offer_ns"),
        f"{cell_id}.runtime_guard.last_external_guard_offer_ns",
    )
    expected = _require_int(
        contract.get("expected_measurement_resume_count"),
        f"{cell_id}.runtime_guard.expected_measurement_resume_count",
        minimum=1,
    )
    canonical = {
        "seed": seed,
        "offered_session_rate_per_second": rate,
        "last_external_guard_offer_ns": cutoff,
        "expected_measurement_resume_count": expected,
    }
    if recorded_entry != canonical:
        raise LiveAstraCollectError(
            f"{cell_id} manifest runtime guard disagrees with campaign")
    if recorded_result != canonical:
        raise LiveAstraCollectError(
            f"{cell_id} result runtime guard disagrees with campaign")
    return canonical


def _measurement_resume_arrival_guard(
    resumes: Sequence[Mapping[str, object]],
    contract: Mapping[str, object],
    *,
    cell_id: str,
) -> dict[str, object]:
    cutoff = _require_int(
        contract.get("last_external_guard_offer_ns"),
        f"{cell_id}.runtime_guard.last_external_guard_offer_ns",
    )
    expected = _require_int(
        contract.get("expected_measurement_resume_count"),
        f"{cell_id}.runtime_guard.expected_measurement_resume_count",
        minimum=1,
    )
    if len(resumes) != expected:
        raise LiveAstraCollectError(
            f"{cell_id} runtime guard expected {expected} measurement "
            f"resumes, observed {len(resumes)}")
    late = tuple(
        row for row in resumes
        if int(row["arrival_ns"]) > cutoff
    )
    if late:
        first = min(late, key=lambda row: int(row["arrival_ns"]))
        raise LiveAstraCollectError(
            f"{cell_id} has {len(late)}/{expected} measurement resume "
            "arrivals after the last external guard offer: "
            f"session={first['session_id']!r}, "
            f"call={first['call_index']}, "
            f"arrival_ns={first['arrival_ns']}, cutoff_ns={cutoff}")
    latest = max(int(row["arrival_ns"]) for row in resumes)
    return {
        "validation_required": True,
        "last_external_guard_offer_ns": cutoff,
        "expected_measurement_resume_count": expected,
        "observed_measurement_resume_count": len(resumes),
        "arrived_by_last_external_guard_offer_count": len(resumes),
        "arrived_after_last_external_guard_offer_count": 0,
        "latest_measurement_resume_arrival_ns": latest,
        "guard_margin_after_latest_measurement_resume_arrival_ns": (
            cutoff - latest),
        "all_measurement_resumes_arrived_by_last_external_guard_offer": True,
        "arrival_semantics": (
            "native requests.csv arrival after the preceding live completion "
            "plus the recorded tool duration"),
    }


def _verify_result_identity(
    result: Mapping[str, object],
    entry: Mapping[str, object],
    cell_id: str,
) -> None:
    expected_pairs = (
        ("cell_id", cell_id),
        ("system", entry.get("system")),
        ("seed", entry.get("seed")),
        ("offered_session_rate_per_second", entry.get("rate")),
    )
    for result_key, expected in expected_pairs:
        if expected is not None and result.get(result_key) != expected:
            raise LiveAstraCollectError(
                f"{cell_id} result {result_key} disagrees with manifest")
    if result.get("status") != "completed":
        raise LiveAstraCollectError(f"{cell_id} result is not completed")
    workload = result.get("workload")
    if not isinstance(workload, Mapping):
        raise LiveAstraCollectError(f"{cell_id} has no workload record")
    for key, entry_key in (
        ("sha256", "workload_sha256"),
        ("request_count", "request_count"),
        ("session_count", "session_count"),
    ):
        expected = entry.get(entry_key)
        if expected is not None and workload.get(key) != expected:
            raise LiveAstraCollectError(
                f"{cell_id} workload {key} disagrees with manifest")


def _collect_cell(
    *,
    manifest_dir: Path,
    campaign_sha256: object,
    campaign: Mapping[str, object],
    cell_id: str,
    entry: Mapping[str, object],
) -> dict[str, object]:
    result_path = _resolve_recorded_path(
        entry.get("result"), manifest_dir, f"{cell_id}.result")
    if not result_path.is_file():
        raise LiveAstraCollectError(
            f"{cell_id} result is missing: {result_path}")
    expected_result_sha = _require_sha256(
        entry.get("result_sha256"),
        f"{cell_id}.result_sha256",
    )
    if _sha256_file(result_path) != expected_result_sha:
        raise LiveAstraCollectError(f"{cell_id} result digest changed")
    result_bytes = entry.get("result_bytes")
    if result_bytes is not None:
        expected_bytes = _require_int(
            result_bytes, f"{cell_id}.result_bytes")
        if result_path.stat().st_size != expected_bytes:
            raise LiveAstraCollectError(
                f"{cell_id} result byte count changed")
    result = _read_json_object(result_path)
    _verify_result_identity(result, entry, cell_id)

    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise LiveAstraCollectError(f"{cell_id} has no artifact records")
    missing = _REQUIRED_ARTIFACTS - set(artifacts)
    if missing:
        raise LiveAstraCollectError(
            f"{cell_id} is missing artifact records: "
            + ", ".join(sorted(missing)))
    artifact_paths = {
        name: _verify_record(
            record,
            base=result_path.parent,
            name=f"{cell_id}.{name}",
        )
        for name, record in artifacts.items()
    }
    requests, header = _parse_requests(artifact_paths["requests"])
    session = _read_json_object(artifact_paths["session_report"])
    runtime = _read_json_object(artifact_paths["runtime_report"])
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise LiveAstraCollectError(f"{cell_id} has no metrics object")
    workload = result["workload"]
    assert isinstance(workload, Mapping)
    expected_request_count = _require_int(
        workload.get("request_count"),
        f"{cell_id}.workload.request_count",
        minimum=1,
    )
    expected_session_count = _require_int(
        workload.get("session_count"),
        f"{cell_id}.workload.session_count",
        minimum=1,
    )
    measurement_session_ids, roster_audit = _measurement_roster(
        metrics,
        campaign,
        session,
        requests,
        expected_request_count=expected_request_count,
        expected_session_count=expected_session_count,
    )
    measured_set = set(measurement_session_ids)
    measured = tuple(
        row for row in requests
        if row["session_id"] in measured_set
    )
    resumes = tuple(
        row for row in measured if int(row["call_index"]) > 0)
    runtime_guard_contract = _runtime_guard_contract_for_cell(
        campaign, entry, result, cell_id)
    runtime_guard_summary = (
        None
        if runtime_guard_contract is None
        else _measurement_resume_arrival_guard(
            resumes,
            runtime_guard_contract,
            cell_id=cell_id,
        )
    )
    if len(requests) != expected_request_count:
        raise LiveAstraCollectError(
            f"{cell_id} requests.csv row count disagrees with workload")
    measured_sessions_present = {str(row["session_id"]) for row in measured}
    if measured_sessions_present != measured_set:
        raise LiveAstraCollectError(
            f"{cell_id} requests.csv does not cover measurement roster")
    performance, metric_check_count = _crosscheck_metrics(
        metrics,
        measured,
        resumes,
        measurement_session_ids,
    )
    offered_rate = _require_number(
        result.get("offered_session_rate_per_second"),
        f"{cell_id}.offered_session_rate_per_second",
    )
    if offered_rate <= 0.0:
        raise LiveAstraCollectError(
            f"{cell_id} offered session rate must be positive")
    measured_session_count = len(measurement_session_ids)
    performance.update({
        "offered_normalized_request_load_per_second": (
            offered_rate * len(measured) / measured_session_count),
        "offered_normalized_resume_load_per_second": (
            offered_rate * len(resumes) / measured_session_count),
        "offered_normalized_output_token_load_per_second": (
            offered_rate
            * sum(int(row["output_tokens"]) for row in measured)
            / measured_session_count
        ),
        "offered_normalized_request_slo_goodput_per_second": (
            offered_rate
            * int(performance["joint_slo_pass_count"])
            / measured_session_count
        ),
        "offered_normalized_resume_slo_goodput_per_second": (
            offered_rate
            * int(performance["resume_joint_slo_pass_count"])
            / measured_session_count
        ),
        "offered_normalized_output_token_slo_goodput_per_second": (
            offered_rate
            * int(performance["joint_slo_pass_output_tokens"])
            / measured_session_count
        ),
        "offered_normalized_session_slo_goodput_per_second": (
            offered_rate
            * int(performance["joint_slo_pass_session_count"])
            / measured_session_count
        ),
        "offered_normalized_goodput_semantics": (
            "system-wide external session offer rate multiplied by the exact "
            "SLO-passing demand per measured session; unlike operational "
            "goodput, this normalization excludes finite-roster drain and "
            "recorded external-gap boundary effects"
        ),
    })

    sources: dict[str, object] = {}
    for output_name, csv_field in {
        "resume_source": "agentic_kv_source",
        "residency_at_return": "agentic_kv_residency_at_return",
        "return_gap_type": "return_gap_type",
    }.items():
        distribution = _source_distribution(resumes, header, csv_field)
        if distribution is not None:
            sources[output_name] = distribution
    prefix_tokens = _prefix_token_accounting(resumes, header)
    if prefix_tokens is not None:
        sources["prefix_token_accounting"] = prefix_tokens

    request_waits = {}
    for output_name, csv_field in _REQUEST_WAIT_FIELDS.items():
        distribution = _wait_distribution(resumes, header, csv_field)
        if distribution is not None:
            request_waits[output_name] = distribution

    runtime_kind = result.get("runtime_kind")
    bottlenecks: dict[str, object] = {}
    if runtime_kind == "full_model_hbf":
        runtime_bottlenecks = _hbf_bottlenecks(runtime)
        if runtime_bottlenecks:
            bottlenecks["hbf"] = runtime_bottlenecks
    else:
        runtime_bottlenecks = _baseline_bottlenecks(runtime)
        if runtime_bottlenecks:
            bottlenecks["baseline"] = runtime_bottlenecks
    if request_waits:
        bottlenecks["measurement_resume_waits"] = request_waits

    validity = _validity_fields(
        session,
        runtime,
        roster_audit=roster_audit,
        runtime_kind=runtime_kind,
        artifact_count=len(artifact_paths),
        request_count=len(requests),
        measured_count=len(measured),
        resume_count=len(resumes),
        metric_crosscheck_count=metric_check_count,
    )
    if runtime_guard_summary is not None:
        validity["measurement_resume_arrival_guard"] = (
            runtime_guard_summary)

    cell: dict[str, object] = {
        "cell_id": cell_id,
        "campaign_sha256": campaign_sha256,
        "system": result.get("system"),
        "runtime_kind": runtime_kind,
        "layout": result.get("layout"),
        "seed": result.get("seed"),
        "offered_session_rate_per_second": result.get(
            "offered_session_rate_per_second"),
        "workload_sha256": workload.get("sha256"),
        "performance": performance,
        "sources": sources,
        "bottlenecks": bottlenecks,
        "validity": validity,
    }
    return cell


def collect_campaign(
    manifest_path: str | Path,
    *,
    allow_incomplete: bool = False,
) -> dict[str, object]:
    """Collect completed cells from one live comparison manifest.

    Unless ``allow_incomplete`` is true, every manifest cell must have status
    ``completed``.  Incomplete cells are skipped only in the explicitly
    permissive mode; completed cells always undergo the same strict checks.
    """

    path = Path(manifest_path).resolve()
    manifest = _read_json_object(path)
    campaign_sha256 = _require_sha256(
        manifest.get("campaign_sha256"),
        "manifest.campaign_sha256",
    )
    campaign = manifest.get("campaign")
    cells = manifest.get("cells")
    if not isinstance(campaign, Mapping):
        raise LiveAstraCollectError("manifest.campaign is not an object")
    if _stable_json_sha256(campaign) != campaign_sha256:
        raise LiveAstraCollectError(
            "manifest campaign stable digest disagrees with "
            "manifest.campaign_sha256")
    if not isinstance(cells, Mapping) or not cells:
        raise LiveAstraCollectError("manifest.cells is not a non-empty object")
    collected = []
    skipped = []
    for cell_id in sorted(cells):
        entry = cells[cell_id]
        if not isinstance(cell_id, str) or not cell_id:
            raise LiveAstraCollectError("manifest has an invalid cell ID")
        if not isinstance(entry, Mapping):
            raise LiveAstraCollectError(
                f"manifest cell {cell_id} is not an object")
        if entry.get("status") != "completed":
            if not allow_incomplete:
                raise LiveAstraCollectError(
                    f"manifest cell {cell_id} is not completed")
            skipped.append(cell_id)
            continue
        collected.append(_collect_cell(
            manifest_dir=path.parent,
            campaign_sha256=campaign_sha256,
            campaign=campaign,
            cell_id=cell_id,
            entry=entry,
        ))
    if not collected:
        raise LiveAstraCollectError("manifest has no completed cells")

    workload_by_schedule: dict[tuple[object, object], str] = {}
    for cell in collected:
        schedule_key = (
            cell["seed"],
            cell["offered_session_rate_per_second"],
        )
        workload_sha = cell["workload_sha256"]
        previous = workload_by_schedule.setdefault(
            schedule_key, str(workload_sha))
        if previous != workload_sha:
            raise LiveAstraCollectError(
                "paired systems used different workloads for "
                f"seed/rate {schedule_key!r}")
        validity = cell["validity"]
        assert isinstance(validity, dict)
        validity["paired_workload_sha_verified"] = True

    output = {
        "schema_version": SCHEMA_VERSION,
        "campaign_sha256": campaign_sha256,
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_status": manifest.get("status"),
        "collected_cell_count": len(collected),
        "skipped_incomplete_cell_count": len(skipped),
        "skipped_incomplete_cell_ids": skipped,
        "paired_seed_rate_count": len(workload_by_schedule),
        "cells": collected,
    }
    _require_finite_tree(output, "compact campaign")
    return output


def _flatten(
    value: Mapping[str, object],
    *,
    prefix: str = "",
) -> dict[str, object]:
    flattened = {}
    for key, child in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            flattened.update(_flatten(child, prefix=name))
        elif isinstance(child, (list, tuple)):
            flattened[name] = json.dumps(
                child,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        else:
            flattened[name] = child
    return flattened


def _atomic_write(path: Path, payload: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_compact_json(
    collected: Mapping[str, object],
    output_path: str | Path,
) -> None:
    """Atomically write the compact nested campaign JSON."""

    payload = (
        json.dumps(
            collected,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(Path(output_path), payload)


def write_compact_csv(
    collected: Mapping[str, object],
    output_path: str | Path,
) -> None:
    """Atomically write one flattened CSV row per collected cell."""

    cells = collected.get("cells")
    if not isinstance(cells, list) or not cells:
        raise LiveAstraCollectError("collected campaign has no cells")
    rows = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise LiveAstraCollectError("collected campaign has invalid cell")
        rows.append(_flatten(cell))
    identity_fields = [
        "cell_id",
        "campaign_sha256",
        "system",
        "runtime_kind",
        "layout",
        "seed",
        "offered_session_rate_per_second",
        "workload_sha256",
    ]
    remaining = sorted(
        set().union(*(set(row) for row in rows)) - set(identity_fields))
    fields = [
        field for field in identity_fields
        if any(field in row for row in rows)
    ] + remaining
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as tmp:
        writer = csv.DictWriter(
            tmp,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        tmp.seek(0)
        payload = tmp.read().encode("utf-8")
    _atomic_write(Path(output_path), payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m serving.live_astra_comparison_collect",
        description=(
            "Verify and compact a live LLMServingSim + ASTRA campaign"
        ),
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = args.manifest.resolve()
    output_json = (
        args.output_json
        if args.output_json is not None
        else manifest.parent / "compact_results.json"
    )
    output_csv = (
        args.output_csv
        if args.output_csv is not None
        else manifest.parent / "compact_results.csv"
    )
    collected = collect_campaign(
        manifest,
        allow_incomplete=args.allow_incomplete,
    )
    write_compact_json(collected, output_json)
    write_compact_csv(collected, output_csv)
    print(json.dumps({
        "collected_cell_count": collected["collected_cell_count"],
        "output_json": str(output_json.resolve()),
        "output_csv": str(output_csv.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

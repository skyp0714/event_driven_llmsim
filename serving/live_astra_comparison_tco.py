"""Strict compact-v2 adapter from live ASTRA results to HBF TCO.

The adapter intentionally accepts one explicit offered rate and one explicit
HBF layout.  It first reruns the canonical live collector from the campaign
manifest, requires byte-independent JSON equality with the supplied compact
artifact, and then aggregates the paired-seed output-token SLO goodput.  No
request-goodput field is used as an economic input.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Mapping, Sequence

from .core.hbf_comparison_tco import (
    BYTES_PER_GIB,
    DEFAULT_HBF_HARDWARE_VARIANT,
    HBFComparisonTCOError,
    HBFHardwareVariant,
    LIVE_COMPACT_OUTPUT_TOKEN_GOODPUT_JSON_PATH,
    OUTPUT_TOKEN_GOODPUT_DEFINITION,
    ORACLE_SYSTEM_KEY,
    PROPOSED_SYSTEM_KEY,
    TIERING_SYSTEM_KEY,
    ComparisonPerformanceProvenance,
    DeploymentTopology,
    GoodputResultProvenance,
    LiveComparisonArtifactProvenance,
    SensitivityAxes,
    TCOSensitivityReport,
    evaluate_tco_sensitivity,
)
from .core.hbf_full_model_latency import (
    load_hbf_server_config,
    qwen_logical_kv_bytes_per_token,
    qwen_model_weight_bytes_per_rank,
)
from .live_astra_comparison_collect import collect_campaign


MANIFEST_SCHEMA_VERSION = 2
COMPACT_SCHEMA_VERSION = 2
TIERING_LIVE_SYSTEM_KEY = "ssd_tiering"
ORACLE_LIVE_SYSTEM_KEY = "oracle"
HBF_LIVE_LAYOUTS = {
    "hbf_tp4": "tp4",
    "hbf_tp8": "tp8",
    "hbf_tp8_context": "tp8_context",
}
TIERING_CLUSTER_CONFIG = (
    "configs/cluster/dual_node_qwen3_1m_pd_p4d4_h100.json")
TIERING_POLICY_CONFIG = (
    "configs/agentic_kv/qwen3_1m_p4d4/tiered_fullprompt.json")
ORACLE_CLUSTER_CONFIG = TIERING_CLUSTER_CONFIG
ORACLE_POLICY_CONFIG = (
    "configs/agentic_kv/qwen3_1m_p4d4/"
    "infinite_hbm_oracle_fullprompt.json")
PROPOSED_GPU_CLUSTER_CONFIG = (
    "configs/cluster/single_node_qwen3_1m_pd_p4d4_h100.json")
PROPOSED_HBF_CONFIG = (
    "configs/wakekv_hbf/full_model_8card_server.json")
EXPECTED_GPU_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
EXPECTED_GPU_DTYPE = "bfloat16"
EXPECTED_GPU_KV_CACHE_DTYPE = "auto"
EXPECTED_CPU_DRAM_BYTES_PER_HOST = 512_000_000_000
EXPECTED_H100_HBM_BYTES_PER_CARD = 80_000_000_000
EXPECTED_GPU_PREFILL_CARDS_PER_HOST = 4
EXPECTED_GPU_DECODE_CARDS_PER_HOST = 4
EXPECTED_GPU_TP_SIZE = 4
EXPECTED_GPU_PP_SIZE = 1
EXPECTED_BASELINE_SSD_DEVICES_PER_HOST = 8
EXPECTED_BASELINE_SSD_CAPACITY_GB_PER_DEVICE = 3_840
EXPECTED_HBF_LPDDR_BYTES_PER_CARD = 64 * BYTES_PER_GIB
ACTIVE_PREFILL_DRAIN_POLICY_KEY_V2 = (
    "first_gpu__migration_inflight_resume_gpu__"
    "hbf_ready_resume_hbf__turn_boundary_lpddr__"
    "active_prefill_drain_v2"
)

_HBF_QUIESCENT_VALIDITY_FIELDS = (
    "adapter_active_prefill_drain_job_count",
    "adapter_pending_prefill_drain_session_count",
    "adapter_waiting_prefill_drain_append_session_count",
    "lifecycle_active_prefill_drain_pending_job_count",
)
_COMMON_VALIDITY_TRUE_FIELDS = (
    "measurement_boundary_complete",
    "measurement_complete",
    "paired_workload_sha_verified",
    "session_timing_passed",
)
_COMMON_VALIDITY_FALSE_FIELDS = (
    "measurement_early_stopped",
)
_COMMON_VALIDITY_ZERO_FIELDS = (
    "headline_metric_crosscheck_mismatch_count",
    "session_timing_violation_count",
    "session_timing_warning_count",
)
_BASELINE_OR_ORACLE_ZERO_FIELDS = (
    "bridge_external_fabric_pending_jobs",
    "bridge_open_astra_windows",
    "bridge_pending_direct_fabric_prepare_locks",
    "bridge_transient_dram_capacity_violations",
    "cutoff_outstanding_dma_jobs",
    "external_fabric_censored_jobs",
    "external_fabric_pending_jobs",
)
_HBF_ZERO_VALIDITY_FIELDS = (
    "adapter_active_prefill_drain_job_count",
    "adapter_pending_gpu_hbm_events",
    "adapter_pending_hbf_turn_finalizations",
    "adapter_pending_prefill_drain_session_count",
    "adapter_pending_router_completions",
    "adapter_staged_hbf_admissions",
    "adapter_waiting_prefill_drain_append_session_count",
    "gpu_hbm_pending_colocated_claim_count",
    "gpu_hbm_pending_pd_decode_reservation_count",
    "gpu_hbm_pending_pd_recompute_binding_count",
    "gpu_hbm_rejected_events",
    "lifecycle_active_prefill_drain_pending_job_count",
    "lifecycle_external_issued_dispatches",
    "lifecycle_external_undrained_dispatches",
    "lifecycle_pending_jobs",
    "multiplexer_pending_jobs",
    "multiplexer_quarantined_dispatches",
    "multiplexer_ready_jobs",
    "pool_external_issued_dispatches",
    "pool_external_undrained_dispatches",
    "pool_pending_batches",
    "pool_pending_launches",
)
MARGINAL_CI_SEMANTICS = (
    "Cells are aligned by identical seed and workload across systems. "
    "Each reported 95% Student-t interval is marginal for one system's "
    "seed sample; it is not a paired-difference or ratio interval."
)

# Two-sided 95% Student-t critical values.  Live campaigns use five paired
# seeds, but retaining the conventional table makes custom 2--31 seed runs
# exact without adding scipy as a simulator dependency.
_STUDENT_T_975 = (
    0.0,
    12.7062047364,
    4.30265272975,
    3.18244630528,
    2.7764451052,
    2.57058183564,
    2.44691184879,
    2.36462425101,
    2.30600413503,
    2.26215716285,
    2.22813885196,
    2.20098516008,
    2.17881282966,
    2.16036865646,
    2.14478668792,
    2.13144954556,
    2.11990529922,
    2.10981557783,
    2.10092204024,
    2.09302405441,
    2.08596344727,
    2.07961384473,
    2.0738730679,
    2.06865761042,
    2.06389856163,
    2.05953855275,
    2.05552943864,
    2.05183051648,
    2.0484071418,
    2.04522964213,
    2.0422724563,
)


class LiveAstraTCOError(HBFComparisonTCOError):
    """Raised when a live campaign cannot support a fail-closed TCO claim."""


def _strict_object(path: Path) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LiveAstraTCOError(
                    f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                LiveAstraTCOError(
                    f"{path}: non-finite JSON token {token}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LiveAstraTCOError(
            f"cannot read strict JSON object {path}") from exc
    if not isinstance(value, dict):
        raise LiveAstraTCOError(f"{path}: JSON root must be an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LiveAstraTCOError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def _stable_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LiveAstraTCOError(
            f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_int(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LiveAstraTCOError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise LiveAstraTCOError(
            f"{name} must be at least {minimum}")
    return value


def _require_number(
    value: object,
    name: str,
    *,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiveAstraTCOError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise LiveAstraTCOError(f"{name} must be a finite number")
    if positive and converted <= 0.0:
        raise LiveAstraTCOError(f"{name} must be positive")
    return converted


def _gib_config_value_to_bytes(value: object, name: str) -> int:
    gib = _require_number(value, name, positive=True)
    raw_bytes = gib * BYTES_PER_GIB
    rounded_bytes = round(raw_bytes)
    if not math.isclose(
            raw_bytes, rounded_bytes, rel_tol=0.0, abs_tol=0.5):
        raise LiveAstraTCOError(
            f"{name} does not represent an integral byte quantity")
    return rounded_bytes


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LiveAstraTCOError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise LiveAstraTCOError(f"{name} must be an array")
    return value


def _recorded_file_sha(
    campaign: Mapping[str, Any],
    relative_or_absolute_path: str,
    *,
    repo_root: Path,
) -> str:
    files = _mapping(campaign.get("files"), "campaign.files")
    digest = _require_sha256(
        files.get(relative_or_absolute_path),
        f"campaign.files[{relative_or_absolute_path!r}]",
    )
    path = Path(relative_or_absolute_path)
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.is_file():
        raise LiveAstraTCOError(
            f"recorded campaign file is missing: {resolved}")
    if _sha256_file(resolved) != digest:
        raise LiveAstraTCOError(
            f"recorded campaign file digest changed: {resolved}")
    return digest


def _cluster_card_count(
    raw: Mapping[str, Any],
    *,
    name: str,
    expected_host_count: int,
) -> int:
    host_count = _require_int(
        raw.get("num_nodes"), f"{name}.num_nodes", minimum=1)
    nodes = _sequence(raw.get("nodes"), f"{name}.nodes")
    if host_count != expected_host_count or len(nodes) != host_count:
        raise LiveAstraTCOError(
            f"{name} must contain exactly {expected_host_count} CPU hosts")
    total_cards = 0
    for node_index, item in enumerate(nodes):
        node = _mapping(item, f"{name}.nodes[{node_index}]")
        node_name = f"{name}.nodes[{node_index}]"
        instances = _sequence(
            node.get("instances"),
            f"{node_name}.instances",
        )
        instance_count = _require_int(
            node.get("num_instances"),
            f"{node_name}.num_instances",
            minimum=1,
        )
        if instance_count != 2 or len(instances) != 2:
            raise LiveAstraTCOError(
                f"{node_name} must contain exactly one prefill and one "
                "decode instance")
        cpu_mem = _mapping(
            node.get("cpu_mem"), f"{node_name}.cpu_mem")
        cpu_dram_bytes = _gib_config_value_to_bytes(
            cpu_mem.get("mem_size"), f"{node_name}.cpu_mem.mem_size")
        if cpu_dram_bytes != EXPECTED_CPU_DRAM_BYTES_PER_HOST:
            raise LiveAstraTCOError(
                f"{node_name} CPU DRAM must be exactly "
                f"{EXPECTED_CPU_DRAM_BYTES_PER_HOST} bytes")
        pd_card_counts: dict[str, int] = {}
        node_cards = 0
        for instance_index, instance_item in enumerate(instances):
            instance_name = (
                f"{node_name}.instances[{instance_index}]")
            instance = _mapping(
                instance_item,
                instance_name,
            )
            if instance.get("hardware") != "H100":
                raise LiveAstraTCOError(
                    f"{instance_name}.hardware must be H100")
            if instance.get("model_name") != EXPECTED_GPU_MODEL_NAME:
                raise LiveAstraTCOError(
                    f"{instance_name}.model_name must be "
                    f"{EXPECTED_GPU_MODEL_NAME}")
            if instance.get("dtype") != EXPECTED_GPU_DTYPE:
                raise LiveAstraTCOError(
                    f"{instance_name}.dtype must be {EXPECTED_GPU_DTYPE}")
            if (
                instance.get("kv_cache_dtype")
                != EXPECTED_GPU_KV_CACHE_DTYPE
            ):
                raise LiveAstraTCOError(
                    f"{instance_name}.kv_cache_dtype must be "
                    f"{EXPECTED_GPU_KV_CACHE_DTYPE}")
            cards = _require_int(
                instance.get("num_npus"),
                f"{instance_name}.num_npus",
                minimum=1,
            )
            tp_size = _require_int(
                instance.get("tp_size"),
                f"{instance_name}.tp_size",
                minimum=1,
            )
            pp_size = _require_int(
                instance.get("pp_size"),
                f"{instance_name}.pp_size",
                minimum=1,
            )
            if (
                cards != EXPECTED_GPU_PREFILL_CARDS_PER_HOST
                or tp_size != EXPECTED_GPU_TP_SIZE
                or pp_size != EXPECTED_GPU_PP_SIZE
            ):
                raise LiveAstraTCOError(
                    f"{instance_name} must be exactly four H100 cards at "
                    f"TP{EXPECTED_GPU_TP_SIZE}/PP{EXPECTED_GPU_PP_SIZE}")
            npu_mem = _mapping(
                instance.get("npu_mem"), f"{instance_name}.npu_mem")
            hbm_bytes = _gib_config_value_to_bytes(
                npu_mem.get("mem_size"),
                f"{instance_name}.npu_mem.mem_size",
            )
            if hbm_bytes != EXPECTED_H100_HBM_BYTES_PER_CARD:
                raise LiveAstraTCOError(
                    f"{instance_name} H100 HBM must be exactly "
                    f"{EXPECTED_H100_HBM_BYTES_PER_CARD} bytes per card")
            pd_type = instance.get("pd_type")
            if pd_type not in {"prefill", "decode"}:
                raise LiveAstraTCOError(
                    f"{instance_name}.pd_type must be prefill or decode")
            if pd_type in pd_card_counts:
                raise LiveAstraTCOError(
                    f"{node_name} contains duplicate {pd_type} instances")
            pd_card_counts[pd_type] = cards
            node_cards += cards
        expected_pd_cards = {
            "prefill": EXPECTED_GPU_PREFILL_CARDS_PER_HOST,
            "decode": EXPECTED_GPU_DECODE_CARDS_PER_HOST,
        }
        if pd_card_counts != expected_pd_cards:
            raise LiveAstraTCOError(
                f"{node_name} must be exactly one 4P/4D pair")
        if node_cards != sum(expected_pd_cards.values()):
            raise LiveAstraTCOError(
                f"{node_name} must contain eight H100 cards")
        total_cards += node_cards
    return total_cards


def _deployment_semantic_snapshot(repo_root: Path) -> dict[str, Any]:
    tiering_cluster = _strict_object(repo_root / TIERING_CLUSTER_CONFIG)
    oracle_cluster = _strict_object(repo_root / ORACLE_CLUSTER_CONFIG)
    proposed_cluster = _strict_object(
        repo_root / PROPOSED_GPU_CLUSTER_CONFIG)
    tiering_policy = _strict_object(repo_root / TIERING_POLICY_CONFIG)
    oracle_policy = _strict_object(repo_root / ORACLE_POLICY_CONFIG)
    hbf_config = _strict_object(repo_root / PROPOSED_HBF_CONFIG)

    tiering_cards = _cluster_card_count(
        tiering_cluster,
        name="tiering cluster",
        expected_host_count=2,
    )
    oracle_cards = _cluster_card_count(
        oracle_cluster,
        name="oracle cluster",
        expected_host_count=2,
    )
    proposed_cards = _cluster_card_count(
        proposed_cluster,
        name="proposed GPU cluster",
        expected_host_count=1,
    )
    if tiering_cluster != oracle_cluster or tiering_cards != oracle_cards:
        raise LiveAstraTCOError(
            "tiering and Oracle must use the same dual 4P4D GPU cluster")
    if tiering_policy.get("policy") != "tiered":
        raise LiveAstraTCOError(
            "tiering policy must be the SSD tiered policy")
    if oracle_policy.get("policy") != "preserve":
        raise LiveAstraTCOError(
            "Oracle policy must preserve infinite-HBM residency")
    ssd_devices_per_host = _require_int(
        tiering_policy.get("ssd_num_devices"),
        "tiering policy ssd_num_devices",
        minimum=1,
    )
    if ssd_devices_per_host != EXPECTED_BASELINE_SSD_DEVICES_PER_HOST:
        raise LiveAstraTCOError(
            "tiering policy must provision exactly "
            f"{EXPECTED_BASELINE_SSD_DEVICES_PER_HOST} SSDs per host")
    ssd_capacity_gb_per_device = _require_int(
        tiering_policy.get("ssd_capacity_gb"),
        "tiering policy ssd_capacity_gb",
        minimum=1,
    )
    if (
        ssd_capacity_gb_per_device
        != EXPECTED_BASELINE_SSD_CAPACITY_GB_PER_DEVICE
    ):
        raise LiveAstraTCOError(
            "tiering policy SSD capacity must be exactly "
            f"{EXPECTED_BASELINE_SSD_CAPACITY_GB_PER_DEVICE} GB per device")
    oracle_ssd_devices = _require_int(
        oracle_policy.get("ssd_num_devices"),
        "Oracle policy ssd_num_devices",
        minimum=1,
    )
    oracle_ssd_capacity_gb = _require_int(
        oracle_policy.get("ssd_capacity_gb"),
        "Oracle policy ssd_capacity_gb",
        minimum=1,
    )
    if (
        oracle_ssd_devices != ssd_devices_per_host
        or oracle_ssd_capacity_gb != ssd_capacity_gb_per_device
    ):
        raise LiveAstraTCOError(
            "Oracle inherited hardware policy no longer matches baseline SSD "
            "device topology")
    hardware = _mapping(hbf_config.get("hardware"), "HBF config.hardware")
    hbf_cards = _require_int(
        hardware.get("card_count"), "HBF card_count", minimum=1)
    lpddr_bytes_per_card = _require_int(
        hardware.get("lpddr_capacity_bytes_per_card"),
        "HBF LPDDR capacity per card",
        minimum=1,
    )
    if lpddr_bytes_per_card % BYTES_PER_GIB != 0:
        raise LiveAstraTCOError(
            "HBF LPDDR capacity must be an integral GiB quantity")
    raw_layouts = _mapping(hbf_config.get("layouts"), "HBF config.layouts")
    expected_layouts = {
        "tp4": (4, 2),
        "tp8": (8, 1),
        "tp8_context": (8, 1),
    }
    for key, (expected_tp_size, expected_replicas) in (
            expected_layouts.items()):
        raw_layout = _mapping(
            raw_layouts.get(key), f"HBF config.layouts.{key}")
        tp_size = _require_int(
            raw_layout.get("tp_size"),
            f"HBF config.layouts.{key}.tp_size",
            minimum=1,
        )
        replicas = _require_int(
            raw_layout.get("replicas"),
            f"HBF config.layouts.{key}.replicas",
            minimum=1,
        )
        if (
            tp_size != expected_tp_size
            or replicas != expected_replicas
        ):
            raise LiveAstraTCOError(
                f"HBF config layout {key!r} changed")
    _, layouts = load_hbf_server_config(repo_root / PROPOSED_HBF_CONFIG)
    for key, (tp_size, replicas) in expected_layouts.items():
        layout = layouts.get(key)
        if (
            layout is None
            or layout.tp_size != tp_size
            or layout.replicas != replicas
        ):
            raise LiveAstraTCOError(
                f"HBF config layout {key!r} changed")

    snapshot = {
        "schema_version": 2,
        "gpu_server_model_name": EXPECTED_GPU_MODEL_NAME,
        "gpu_server_dtype": EXPECTED_GPU_DTYPE,
        "gpu_server_kv_cache_dtype": EXPECTED_GPU_KV_CACHE_DTYPE,
        "gpu_server_prefill_h100_cards": (
            EXPECTED_GPU_PREFILL_CARDS_PER_HOST),
        "gpu_server_decode_h100_cards": (
            EXPECTED_GPU_DECODE_CARDS_PER_HOST),
        "gpu_instance_tp_size": EXPECTED_GPU_TP_SIZE,
        "gpu_instance_pp_size": EXPECTED_GPU_PP_SIZE,
        "cpu_dram_bytes_per_gpu_host": (
            EXPECTED_CPU_DRAM_BYTES_PER_HOST),
        "h100_hbm_bytes_per_card": EXPECTED_H100_HBM_BYTES_PER_CARD,
        "tiering_cpu_hosts": 2,
        "tiering_h100_cards": tiering_cards,
        "tiering_cpu_dram_bytes_per_host": (
            EXPECTED_CPU_DRAM_BYTES_PER_HOST),
        "tiering_ssd_devices_per_host": ssd_devices_per_host,
        "tiering_ssd_devices": 2 * ssd_devices_per_host,
        "tiering_ssd_capacity_gb_per_device": (
            ssd_capacity_gb_per_device),
        "proposed_gpu_cpu_hosts": 1,
        "proposed_hbf_cpu_hosts": 1,
        "proposed_cpu_hosts": 2,
        "proposed_gpu_host_cpu_dram_bytes": (
            EXPECTED_CPU_DRAM_BYTES_PER_HOST),
        "proposed_hbf_host_cpu_dram_bytes": (
            EXPECTED_CPU_DRAM_BYTES_PER_HOST),
        "proposed_hbf_host_cpu_dram_semantics": (
            "explicit_bom_assumption_same_as_gpu_host"),
        "proposed_h100_cards": proposed_cards,
        "proposed_hbf_npu_cards": hbf_cards,
        "proposed_lpddr_gib": (
            hbf_cards * lpddr_bytes_per_card // BYTES_PER_GIB),
        "proposed_ssd_devices": 0,
    }
    topology = DeploymentTopology()
    expected = {
        "schema_version": 2,
        "gpu_server_model_name": EXPECTED_GPU_MODEL_NAME,
        "gpu_server_dtype": EXPECTED_GPU_DTYPE,
        "gpu_server_kv_cache_dtype": EXPECTED_GPU_KV_CACHE_DTYPE,
        "gpu_server_prefill_h100_cards": (
            topology.tiering_h100_cards_per_host // 2),
        "gpu_server_decode_h100_cards": (
            topology.tiering_h100_cards_per_host // 2),
        "gpu_instance_tp_size": EXPECTED_GPU_TP_SIZE,
        "gpu_instance_pp_size": EXPECTED_GPU_PP_SIZE,
        "cpu_dram_bytes_per_gpu_host": round(
            topology.host_dram_gib_per_host * BYTES_PER_GIB),
        "h100_hbm_bytes_per_card": EXPECTED_H100_HBM_BYTES_PER_CARD,
        "tiering_cpu_hosts": topology.tiering_cpu_hosts,
        "tiering_h100_cards": topology.tiering_h100_cards,
        "tiering_cpu_dram_bytes_per_host": round(
            topology.host_dram_gib_per_host * BYTES_PER_GIB),
        "tiering_ssd_devices_per_host": (
            topology.tiering_ssd_devices_per_host),
        "tiering_ssd_devices": topology.tiering_ssd_devices,
        "tiering_ssd_capacity_gb_per_device": (
            EXPECTED_BASELINE_SSD_CAPACITY_GB_PER_DEVICE),
        "proposed_gpu_cpu_hosts": topology.proposed_gpu_cpu_hosts,
        "proposed_hbf_cpu_hosts": topology.proposed_hbf_cpu_hosts,
        "proposed_cpu_hosts": topology.proposed_cpu_hosts,
        "proposed_gpu_host_cpu_dram_bytes": round(
            topology.host_dram_gib_per_host * BYTES_PER_GIB),
        "proposed_hbf_host_cpu_dram_bytes": round(
            topology.host_dram_gib_per_host * BYTES_PER_GIB),
        "proposed_hbf_host_cpu_dram_semantics": (
            "explicit_bom_assumption_same_as_gpu_host"),
        "proposed_h100_cards": topology.proposed_h100_cards,
        "proposed_hbf_npu_cards": topology.proposed_hbf_npu_cards,
        "proposed_lpddr_gib": int(topology.proposed_lpddr_gib),
        "proposed_ssd_devices": 0,
    }
    if snapshot != expected:
        raise LiveAstraTCOError(
            "parsed deployment config quantities disagree with the TCO BOM: "
            f"parsed={snapshot!r}, expected={expected!r}")
    return snapshot


def _capacity_semantic_snapshot(repo_root: Path) -> dict[str, Any]:
    hardware, layouts = load_hbf_server_config(
        repo_root / PROPOSED_HBF_CONFIG)
    rows = {}
    for system_key, layout_key in HBF_LIVE_LAYOUTS.items():
        layout = layouts[layout_key]
        weight_bytes = qwen_model_weight_bytes_per_rank(layout.tp_size)
        usable = (
            (
                hardware.hbf_capacity_bytes_per_card
                - weight_bytes
            )
            * layout.tp_size
            * layout.replicas
            // layout.physical_kv_replication_factor
        )
        if usable <= 0:
            raise LiveAstraTCOError(
                f"{system_key} has no usable logical HBF KV capacity")
        rows[system_key] = {
            "layout_key": layout.key,
            "tp_size": layout.tp_size,
            "replicas": layout.replicas,
            "model_weight_bytes_per_card": weight_bytes,
            "physical_kv_replication_factor": (
                layout.physical_kv_replication_factor),
            "usable_logical_hbf_kv_capacity_bytes": usable,
        }
    return {
        "schema_version": 1,
        "card_count": hardware.card_count,
        "hbf_capacity_bytes_per_card": (
            hardware.hbf_capacity_bytes_per_card),
        "logical_kv_bytes_per_token": qwen_logical_kv_bytes_per_token(),
        "layouts": rows,
    }


def _system_spec(
    campaign: Mapping[str, Any],
    system_key: str,
) -> Mapping[str, Any]:
    systems = _sequence(campaign.get("systems"), "campaign.systems")
    matches = [
        _mapping(item, "campaign.systems[]")
        for item in systems
        if isinstance(item, Mapping) and item.get("key") == system_key
    ]
    if len(matches) != 1:
        raise LiveAstraTCOError(
            f"campaign must contain exactly one {system_key!r} system spec")
    return matches[0]


def _validate_system_spec(
    spec: Mapping[str, Any],
    *,
    system_key: str,
    cluster_config: str,
    policy_config: str,
    runtime_kind: str,
    layout: str | None,
) -> None:
    expected = {
        "key": system_key,
        "cluster_config": cluster_config,
        "policy_config": policy_config,
        "runtime_kind": runtime_kind,
        "layout": layout,
    }
    if dict(spec) != expected:
        raise LiveAstraTCOError(
            f"{system_key} physical system spec changed: "
            f"expected={expected!r}, actual={dict(spec)!r}")


def _implementation_identity(
    campaign: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[str, str, str]:
    implementation = _mapping(
        campaign.get("simulator_implementation"),
        "campaign.simulator_implementation",
    )
    astra = _mapping(
        implementation.get("astra_binary"),
        "campaign.simulator_implementation.astra_binary",
    )
    astra_sha = _require_sha256(
        astra.get("sha256"), "campaign ASTRA binary sha256")
    _require_int(astra.get("bytes"), "campaign ASTRA binary bytes", minimum=1)
    sources = _mapping(
        implementation.get("source_files"),
        "campaign.simulator_implementation.source_files",
    )
    if not sources:
        raise LiveAstraTCOError(
            "campaign simulator source-file identity is empty")
    for source_path, record in sources.items():
        if not isinstance(source_path, str) or not source_path:
            raise LiveAstraTCOError(
                "campaign simulator source path is invalid")
        source = _mapping(record, f"source_files[{source_path!r}]")
        _require_sha256(
            source.get("sha256"),
            f"source_files[{source_path!r}].sha256",
        )
        _require_int(
            source.get("bytes"),
            f"source_files[{source_path!r}].bytes",
            minimum=1,
        )
    collector_path = "serving/live_astra_comparison_collect.py"
    collector_record = _mapping(
        sources.get(collector_path),
        f"source_files[{collector_path!r}]",
    )
    collector_sha = _require_sha256(
        collector_record.get("sha256"),
        "campaign canonical collector source sha256",
    )
    if _sha256_file(repo_root / collector_path) != collector_sha:
        raise LiveAstraTCOError(
            "the canonical collector source differs from the "
            "campaign-pinned collector")
    _mapping(
        implementation.get("source_scope"),
        "campaign.simulator_implementation.source_scope",
    )
    return _stable_sha256(implementation), astra_sha, collector_sha


def _tco_adapter_implementation_sha(
    repo_root: Path,
    capacity_snapshot: Mapping[str, Any],
) -> str:
    sources = {}
    for relative in (
        "serving/core/hbf_comparison_tco.py",
        "serving/core/hbf_full_model_latency.py",
        "serving/core/h100_kernel_calibrated_prompt.py",
        "serving/live_astra_comparison_tco.py",
    ):
        path = repo_root / relative
        if not path.is_file():
            raise LiveAstraTCOError(
                f"TCO adapter source is missing: {path}")
        sources[relative] = _sha256_file(path)
    return _stable_sha256({
        "source_files": sources,
        "capacity_semantic_snapshot": capacity_snapshot,
    })


def _validate_campaign_and_compact(
    compact: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Mapping[str, Any], dict[str, str], str, str, str]:
    if _require_int(
            manifest.get("schema_version"),
            "manifest.schema_version",
            minimum=0,
    ) != MANIFEST_SCHEMA_VERSION:
        raise LiveAstraTCOError(
            "live TCO requires manifest schema version 2")
    if _require_int(
            compact.get("schema_version"),
            "compact.schema_version",
            minimum=0,
    ) != COMPACT_SCHEMA_VERSION:
        raise LiveAstraTCOError(
            "live TCO requires compact schema version 2")
    if manifest.get("status") != "completed":
        raise LiveAstraTCOError("campaign manifest is not completed")
    if compact.get("manifest_status") != "completed":
        raise LiveAstraTCOError("compact campaign is not completed")
    if _require_int(
            compact.get("manifest_schema_version"),
            "compact.manifest_schema_version",
            minimum=0,
    ) != MANIFEST_SCHEMA_VERSION:
        raise LiveAstraTCOError(
            "compact manifest schema identity changed")
    if _require_int(
            compact.get("skipped_incomplete_cell_count"),
            "compact.skipped_incomplete_cell_count",
            minimum=0) != 0:
        raise LiveAstraTCOError(
            "TCO cannot consume an incomplete compact campaign")

    campaign = _mapping(manifest.get("campaign"), "manifest.campaign")
    campaign_rates = tuple(
        _require_number(
            value, "campaign.rates[]", positive=True)
        for value in _sequence(campaign.get("rates"), "campaign.rates")
    )
    campaign_seeds = tuple(
        _require_int(value, "campaign.seeds[]", minimum=0)
        for value in _sequence(campaign.get("seeds"), "campaign.seeds")
    )
    paired_seed_rate_count = _require_int(
        compact.get("paired_seed_rate_count"),
        "compact.paired_seed_rate_count",
        minimum=0,
    )
    if paired_seed_rate_count != len(campaign_rates) * len(campaign_seeds):
        raise LiveAstraTCOError(
            "compact paired_seed_rate_count does not match the exact "
            "campaign rate/seed grid")
    skipped_ids = _sequence(
        compact.get("skipped_incomplete_cell_ids"),
        "compact.skipped_incomplete_cell_ids",
    )
    if skipped_ids:
        raise LiveAstraTCOError(
            "TCO cannot consume skipped incomplete cell IDs")
    campaign_sha = _require_sha256(
        manifest.get("campaign_sha256"), "manifest.campaign_sha256")
    if _stable_sha256(campaign) != campaign_sha:
        raise LiveAstraTCOError(
            "manifest campaign payload does not match campaign_sha256")
    if compact.get("campaign_sha256") != campaign_sha:
        raise LiveAstraTCOError(
            "compact and manifest campaign hashes differ")

    manifest_cells = _mapping(manifest.get("cells"), "manifest.cells")
    compact_cells = _sequence(compact.get("cells"), "compact.cells")
    if (
        _require_int(
            compact.get("collected_cell_count"),
            "compact.collected_cell_count",
            minimum=1,
        )
        != len(compact_cells)
        or len(compact_cells) != len(manifest_cells)
    ):
        raise LiveAstraTCOError(
            "compact and manifest cell rosters differ")
    compact_by_id: dict[str, Mapping[str, Any]] = {}
    for item in compact_cells:
        cell = _mapping(item, "compact.cells[]")
        cell_id = cell.get("cell_id")
        if (
            not isinstance(cell_id, str)
            or not cell_id
            or cell_id in compact_by_id
        ):
            raise LiveAstraTCOError(
                "compact contains an invalid or duplicate cell ID")
        compact_by_id[cell_id] = cell
    if set(compact_by_id) != set(manifest_cells):
        raise LiveAstraTCOError(
            "compact and manifest cell IDs differ")
    for cell_id, manifest_item in manifest_cells.items():
        entry = _mapping(manifest_item, f"manifest.cells[{cell_id!r}]")
        cell = compact_by_id[cell_id]
        if entry.get("status") != "completed":
            raise LiveAstraTCOError(
                f"manifest cell {cell_id} is not completed")
        if cell.get("system") != entry.get("system"):
            raise LiveAstraTCOError(
                f"compact cell {cell_id} disagrees with manifest field "
                "system")
        compact_seed = _require_int(
            cell.get("seed"), f"compact cell {cell_id}.seed", minimum=0)
        manifest_seed = _require_int(
            entry.get("seed"), f"manifest cell {cell_id}.seed", minimum=0)
        if compact_seed != manifest_seed:
            raise LiveAstraTCOError(
                f"compact cell {cell_id} disagrees with manifest field seed")
        compact_rate = _require_number(
            cell.get("offered_session_rate_per_second"),
            f"compact cell {cell_id}.offered rate",
            positive=True,
        )
        manifest_rate = _require_number(
            entry.get("rate"),
            f"manifest cell {cell_id}.rate",
            positive=True,
        )
        if compact_rate != manifest_rate:
            raise LiveAstraTCOError(
                f"compact cell {cell_id} disagrees with manifest field rate")
        compact_workload = _require_sha256(
            cell.get("workload_sha256"),
            f"compact cell {cell_id}.workload_sha256",
        )
        manifest_workload = _require_sha256(
            entry.get("workload_sha256"),
            f"manifest cell {cell_id}.workload_sha256",
        )
        if compact_workload != manifest_workload:
            raise LiveAstraTCOError(
                f"compact cell {cell_id} disagrees with manifest field "
                "workload_sha256")

    scenario_source = _require_sha256(
        campaign.get("scenario_source_sha256"),
        "campaign.scenario_source_sha256",
    )
    trace_path = campaign.get("trace_path")
    if not isinstance(trace_path, str) or not trace_path:
        raise LiveAstraTCOError("campaign.trace_path is invalid")
    if _recorded_file_sha(
            campaign, trace_path, repo_root=repo_root) != scenario_source:
        raise LiveAstraTCOError(
            "scenario source hash does not match the TraceLab file")

    hbf_keys = [
        key for key in HBF_LIVE_LAYOUTS
        if any(
            isinstance(item, Mapping) and item.get("key") == key
            for item in _sequence(campaign.get("systems"), "campaign.systems")
        )
    ]
    if not hbf_keys:
        raise LiveAstraTCOError("campaign contains no supported HBF system")
    specs = {
        TIERING_LIVE_SYSTEM_KEY: _system_spec(
            campaign, TIERING_LIVE_SYSTEM_KEY),
        ORACLE_LIVE_SYSTEM_KEY: _system_spec(
            campaign, ORACLE_LIVE_SYSTEM_KEY),
    }
    _validate_system_spec(
        specs[TIERING_LIVE_SYSTEM_KEY],
        system_key=TIERING_LIVE_SYSTEM_KEY,
        cluster_config=TIERING_CLUSTER_CONFIG,
        policy_config=TIERING_POLICY_CONFIG,
        runtime_kind="agentic_kv",
        layout=None,
    )
    _validate_system_spec(
        specs[ORACLE_LIVE_SYSTEM_KEY],
        system_key=ORACLE_LIVE_SYSTEM_KEY,
        cluster_config=ORACLE_CLUSTER_CONFIG,
        policy_config=ORACLE_POLICY_CONFIG,
        runtime_kind="oracle",
        layout=None,
    )
    for key in hbf_keys:
        _validate_system_spec(
            _system_spec(campaign, key),
            system_key=key,
            cluster_config=PROPOSED_GPU_CLUSTER_CONFIG,
            policy_config=PROPOSED_HBF_CONFIG,
            runtime_kind="full_model_hbf",
            layout=HBF_LIVE_LAYOUTS[key],
        )

    config_hashes = {
        "tiering_cluster": _recorded_file_sha(
            campaign, TIERING_CLUSTER_CONFIG, repo_root=repo_root),
        "tiering_policy": _recorded_file_sha(
            campaign, TIERING_POLICY_CONFIG, repo_root=repo_root),
        "proposed_gpu_cluster": _recorded_file_sha(
            campaign, PROPOSED_GPU_CLUSTER_CONFIG, repo_root=repo_root),
        "proposed_hbf": _recorded_file_sha(
            campaign, PROPOSED_HBF_CONFIG, repo_root=repo_root),
        "oracle_cluster": _recorded_file_sha(
            campaign, ORACLE_CLUSTER_CONFIG, repo_root=repo_root),
        "oracle_policy": _recorded_file_sha(
            campaign, ORACLE_POLICY_CONFIG, repo_root=repo_root),
    }
    implementation_sha, astra_sha, collector_sha = (
        _implementation_identity(campaign, repo_root=repo_root))
    return (
        campaign,
        config_hashes,
        implementation_sha,
        astra_sha,
        collector_sha,
    )


def _selected_cells(
    compact: Mapping[str, Any],
    *,
    rate: float,
    hbf_system_key: str,
    seeds: tuple[int, ...],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    selected_systems = (
        TIERING_LIVE_SYSTEM_KEY,
        hbf_system_key,
        ORACLE_LIVE_SYSTEM_KEY,
    )
    buckets: dict[str, dict[int, Mapping[str, Any]]] = {
        key: {} for key in selected_systems}
    for item in _sequence(compact.get("cells"), "compact.cells"):
        cell = _mapping(item, "compact.cells[]")
        system = cell.get("system")
        if system not in buckets:
            continue
        cell_rate = _require_number(
            cell.get("offered_session_rate_per_second"),
            f"{cell.get('cell_id')}.offered rate",
            positive=True,
        )
        if cell_rate != rate:
            continue
        seed = _require_int(
            cell.get("seed"), f"{cell.get('cell_id')}.seed", minimum=0)
        if seed in buckets[str(system)]:
            raise LiveAstraTCOError(
                f"duplicate {system} cell for seed={seed}, rate={rate}")
        buckets[str(system)][seed] = cell
    expected = set(seeds)
    result = {}
    for system, by_seed in buckets.items():
        if set(by_seed) != expected:
            raise LiveAstraTCOError(
                f"{system} does not have the exact paired seed roster at "
                f"rate={rate}: expected={sorted(expected)}, "
                f"actual={sorted(by_seed)}")
        result[system] = tuple(by_seed[seed] for seed in seeds)
    return result


def _require_validity_value(
    validity: Mapping[str, Any],
    field: str,
    expected: object,
    *,
    cell_id: object,
) -> None:
    if field not in validity:
        raise LiveAstraTCOError(
            f"{cell_id} validity is missing required field {field}")
    actual = validity[field]
    if isinstance(expected, bool):
        matches = type(actual) is bool and actual is expected
    elif isinstance(expected, int):
        matches = (
            type(actual) is int
            and actual >= 0
            and actual == expected
        )
    else:
        matches = type(actual) is type(expected) and actual == expected
    if not matches:
        raise LiveAstraTCOError(
            f"{cell_id} validity field {field} must be {expected!r}, "
            f"with type {type(expected).__name__}, got {actual!r} "
            f"with type {type(actual).__name__}")


def _validate_common_cell_validity(
    cell: Mapping[str, Any],
) -> Mapping[str, Any]:
    cell_id = cell.get("cell_id")
    validity = _mapping(cell.get("validity"), f"{cell_id}.validity")
    for field in _COMMON_VALIDITY_TRUE_FIELDS:
        _require_validity_value(
            validity, field, True, cell_id=cell_id)
    for field in _COMMON_VALIDITY_FALSE_FIELDS:
        _require_validity_value(
            validity, field, False, cell_id=cell_id)
    for field in _COMMON_VALIDITY_ZERO_FIELDS:
        _require_validity_value(
            validity, field, 0, cell_id=cell_id)
    _require_validity_value(
        validity, "verified_artifact_count", 5, cell_id=cell_id)

    parsed = _require_int(
        validity.get("parsed_request_count"),
        f"{cell_id}.validity.parsed_request_count",
        minimum=1,
    )
    measured = _require_int(
        validity.get("measurement_request_count"),
        f"{cell_id}.validity.measurement_request_count",
        minimum=1,
    )
    resumes = _require_int(
        validity.get("measurement_resume_request_count"),
        f"{cell_id}.validity.measurement_resume_request_count",
        minimum=1,
    )
    timing_checked = _require_int(
        validity.get("session_timing_checked_requests"),
        f"{cell_id}.validity.session_timing_checked_requests",
        minimum=1,
    )
    crosschecks = _require_int(
        validity.get("headline_metric_crosscheck_count"),
        f"{cell_id}.validity.headline_metric_crosscheck_count",
        minimum=1,
    )
    if measured > parsed or resumes > measured:
        raise LiveAstraTCOError(
            f"{cell_id} validity request counts are inconsistent")
    if timing_checked != parsed:
        raise LiveAstraTCOError(
            f"{cell_id} timing validation did not cover every parsed request")
    if crosschecks < 1:
        raise LiveAstraTCOError(
            f"{cell_id} performed no headline metric crosschecks")
    return validity


def _validate_non_hbf_validity(
    cell: Mapping[str, Any],
    *,
    oracle: bool,
) -> None:
    validity = _validate_common_cell_validity(cell)
    cell_id = cell.get("cell_id")
    for field in _BASELINE_OR_ORACLE_ZERO_FIELDS:
        _require_validity_value(
            validity, field, 0, cell_id=cell_id)
    _require_validity_value(
        validity, "cutoff_measurement_censored", False, cell_id=cell_id)
    issued = _require_int(
        validity.get("external_fabric_issued_jobs"),
        f"{cell_id}.validity.external_fabric_issued_jobs",
        minimum=0,
    )
    completed = _require_int(
        validity.get("external_fabric_completed_jobs"),
        f"{cell_id}.validity.external_fabric_completed_jobs",
        minimum=0,
    )
    if issued != completed:
        raise LiveAstraTCOError(
            f"{cell_id} external fabric jobs did not drain")
    if not oracle:
        return
    for field, expected in (
        ("oracle_enabled", True),
        ("oracle_passed", True),
        ("oracle_nonzero_invariant_count", 0),
        ("oracle_violation_count", 0),
    ):
        _require_validity_value(
            validity, field, expected, cell_id=cell_id)
    checked = _require_int(
        validity.get("oracle_checked_reusable_resumes"),
        f"{cell_id}.validity.oracle_checked_reusable_resumes",
        minimum=1,
    )
    resumes = _require_int(
        validity.get("measurement_resume_request_count"),
        f"{cell_id}.validity.measurement_resume_request_count",
        minimum=1,
    )
    instance_count = _require_int(
        validity.get("oracle_instance_count"),
        f"{cell_id}.validity.oracle_instance_count",
        minimum=1,
    )
    nonbinding_count = _require_int(
        validity.get("oracle_nonbinding_instance_count"),
        f"{cell_id}.validity.oracle_nonbinding_instance_count",
        minimum=1,
    )
    zero_invariants = _require_int(
        validity.get("oracle_zero_invariant_count"),
        f"{cell_id}.validity.oracle_zero_invariant_count",
        minimum=1,
    )
    if checked != resumes:
        raise LiveAstraTCOError(
            f"{cell_id} Oracle did not check every measured resume")
    if nonbinding_count != instance_count or zero_invariants < 1:
        raise LiveAstraTCOError(
            f"{cell_id} Oracle nonbinding invariants are incomplete")


def _require_zero_tree(value: object, name: str) -> None:
    if isinstance(value, Mapping):
        if not value:
            raise LiveAstraTCOError(f"{name} must not be empty")
        for key, child in value.items():
            _require_zero_tree(child, f"{name}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)):
        if value:
            raise LiveAstraTCOError(f"{name} must be empty")
        return
    if type(value) is not int:
        raise LiveAstraTCOError(f"{name} must contain integer zero leaves")
    if value != 0:
        raise LiveAstraTCOError(f"{name} retained nonzero terminal ownership")


def _require_exact_hbf_layout(
    value: object,
    *,
    name: str,
    layout_key: str,
    tp_size: int,
    replicas: int,
) -> None:
    layout = _mapping(value, name)
    if set(layout) != {"key", "tp_size", "replicas"}:
        raise LiveAstraTCOError(
            f"{name} must contain exactly key, tp_size, and replicas")
    if layout.get("key") != layout_key:
        raise LiveAstraTCOError(
            f"{name}.key disagrees with selected layout")
    actual_tp_size = _require_int(
        layout.get("tp_size"), f"{name}.tp_size", minimum=1)
    actual_replicas = _require_int(
        layout.get("replicas"), f"{name}.replicas", minimum=1)
    if actual_tp_size != tp_size or actual_replicas != replicas:
        raise LiveAstraTCOError(
            f"{name} disagrees with selected layout")


def _validate_hbf_runtime_ledger(
    runtime: Mapping[str, Any],
    *,
    cell_id: object,
    layout_key: str,
) -> None:
    expected_parallel = {
        "tp4": (4, 2),
        "tp8": (8, 1),
        "tp8_context": (8, 1),
    }
    tp_size, replicas = expected_parallel[layout_key]
    adapter = _mapping(runtime.get("adapter"), f"{cell_id}.adapter")
    pool = _mapping(adapter.get("pool"), f"{cell_id}.adapter.pool")
    lifecycle = _mapping(
        adapter.get("lifecycle"), f"{cell_id}.adapter.lifecycle")
    bridge = _mapping(
        runtime.get("gpu_hbm_bridge"), f"{cell_id}.gpu_hbm_bridge")
    for name, value in (
        ("runtime.layout", runtime.get("layout")),
        ("adapter.pool.layout", pool.get("layout")),
        ("adapter.lifecycle.layout", lifecycle.get("layout")),
    ):
        _require_exact_hbf_layout(
            value,
            name=f"{cell_id}.{name}",
            layout_key=layout_key,
            tp_size=tp_size,
            replicas=replicas,
        )

    expected_groups = {str(index) for index in range(replicas)}
    used_per_group = _mapping(
        pool.get("lpddr_used_bytes_per_group"),
        f"{cell_id}.pool.lpddr_used_bytes_per_group",
    )
    used_by_card = _mapping(
        pool.get("lpddr_used_bytes_by_card"),
        f"{cell_id}.pool.lpddr_used_bytes_by_card",
    )
    reserved_per_group = _mapping(
        lifecycle.get("group_reserved_per_card_bytes"),
        f"{cell_id}.lifecycle.group_reserved_per_card_bytes",
    )
    reserved_by_card = _mapping(
        lifecycle.get("group_reserved_bytes_by_card"),
        f"{cell_id}.lifecycle.group_reserved_bytes_by_card",
    )
    for name, mapping in (
        ("lpddr_used_bytes_per_group", used_per_group),
        ("lpddr_used_bytes_by_card", used_by_card),
        ("group_reserved_per_card_bytes", reserved_per_group),
        ("group_reserved_bytes_by_card", reserved_by_card),
    ):
        if set(mapping) != expected_groups:
            raise LiveAstraTCOError(
                f"{cell_id} {name} has the wrong replica-group roster")
        _require_zero_tree(mapping, f"{cell_id}.{name}")
    for group_id in range(replicas):
        expected_cards = {
            str(card)
            for card in range(group_id * tp_size, (group_id + 1) * tp_size)
        }
        for name, mapping in (
            ("lpddr_used_bytes_by_card", used_by_card),
            ("group_reserved_bytes_by_card", reserved_by_card),
        ):
            cards = _mapping(mapping[str(group_id)], f"{cell_id}.{name}")
            if set(cards) != expected_cards:
                raise LiveAstraTCOError(
                    f"{cell_id} {name} has the wrong card roster")
    pool_hardware = _mapping(
        pool.get("hardware"), f"{cell_id}.pool.hardware")
    physical_lpddr_capacity = _require_int(
        pool_hardware.get("lpddr_capacity_bytes_per_card"),
        f"{cell_id}.pool.hardware.lpddr_capacity_bytes_per_card",
        minimum=1,
    )
    if physical_lpddr_capacity != EXPECTED_HBF_LPDDR_BYTES_PER_CARD:
        raise LiveAstraTCOError(
            f"{cell_id} physical LPDDR capacity must be exactly "
            f"{EXPECTED_HBF_LPDDR_BYTES_PER_CARD} bytes per card")
    workspace_bytes = _require_int(
        pool.get("workspace_bytes_per_card"),
        f"{cell_id}.pool.workspace_bytes_per_card",
        minimum=0,
    )
    kv_capacity = _require_int(
        pool.get("lpddr_kv_capacity_bytes_per_card"),
        f"{cell_id}.pool.lpddr_kv_capacity_bytes_per_card",
        minimum=1,
    )
    ledger_capacity = _require_int(
        pool.get("lpddr_ledger_capacity_bytes_per_card"),
        f"{cell_id}.pool.lpddr_ledger_capacity_bytes_per_card",
        minimum=1,
    )
    if (
        workspace_bytes + kv_capacity != physical_lpddr_capacity
        or ledger_capacity != kv_capacity
    ):
        raise LiveAstraTCOError(
            f"{cell_id} LPDDR workspace/KV/ledger capacity algebra changed")

    sessions = _mapping(
        lifecycle.get("sessions"), f"{cell_id}.lifecycle.sessions")
    if not sessions:
        raise LiveAstraTCOError(
            f"{cell_id} lifecycle session ledger is empty")
    for session_id, item in sessions.items():
        session = _mapping(item, f"{cell_id}.sessions[{session_id!r}]")
        if session.get("state") != "ended":
            raise LiveAstraTCOError(
                f"{cell_id} retained a non-ended lifecycle session")
        for field in (
            "committed_hbf_tokens",
            "lpddr_tokens",
            "gpu_retained_bytes",
            "committed_per_card_bytes",
            "pending_reserved_per_card_bytes",
        ):
            retained = _require_int(
                session.get(field),
                f"{cell_id}.sessions[{session_id!r}].{field}",
                minimum=0,
            )
            if retained != 0:
                raise LiveAstraTCOError(
                    f"{cell_id} session ledger retained {field}")
        for field in ("group_id", "active_request_id"):
            if session.get(field) is not None:
                raise LiveAstraTCOError(
                    f"{cell_id} session ledger retained {field}")
        for field in ("migration_job_ids", "append_job_ids"):
            if session.get(field) != []:
                raise LiveAstraTCOError(
                    f"{cell_id} session ledger retained {field}")

    memory = _mapping(
        bridge.get("memory_by_instance"),
        f"{cell_id}.gpu_hbm_bridge.memory_by_instance",
    )
    if len(memory) != 2:
        raise LiveAstraTCOError(
            f"{cell_id} GPU bridge must report P and D memory ledgers")
    for instance_id, item in memory.items():
        row = _mapping(
            item, f"{cell_id}.gpu_hbm_bridge.memory[{instance_id}]")
        for field in (
            "npu_used_per_rank_bytes",
            "dynamic_used_per_rank_bytes",
            "bridge_owned_per_rank_bytes",
        ):
            retained = _require_int(
                row.get(field),
                (
                    f"{cell_id}.gpu_hbm_bridge.memory"
                    f"[{instance_id}].{field}"
                ),
                minimum=0,
            )
            if retained != 0:
                raise LiveAstraTCOError(
                    f"{cell_id} GPU bridge retained {field}")
    for field in (
        "pending_colocated_claims",
        "pending_pd_recompute_bindings",
        "pending_pd_decode_reservations",
    ):
        if bridge.get(field) != []:
            raise LiveAstraTCOError(
                f"{cell_id} GPU bridge retained {field}")


def _validate_hbf_validity(
    cell: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    expected_layout: str,
) -> None:
    validity = _validate_common_cell_validity(cell)
    cell_id = cell.get("cell_id")
    for field in _HBF_ZERO_VALIDITY_FIELDS:
        _require_validity_value(
            validity, field, 0, cell_id=cell_id)
    for field in (
        "lifecycle_external_completed_dispatches",
        "multiplexer_completed_jobs",
    ):
        _require_int(
            validity.get(field), f"{cell_id}.validity.{field}", minimum=0)
    _validate_hbf_runtime_ledger(
        runtime, cell_id=cell_id, layout_key=expected_layout)


def _validate_selected_cell_contracts(
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    hbf_system_key: str,
    runtime_reports_by_cell_id: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = {
        TIERING_LIVE_SYSTEM_KEY: ("agentic_kv", None),
        ORACLE_LIVE_SYSTEM_KEY: ("oracle", None),
        hbf_system_key: ("full_model_hbf", HBF_LIVE_LAYOUTS[hbf_system_key]),
    }
    for system, cells in selected.items():
        runtime_kind, layout = expected[system]
        for cell in cells:
            cell_id = cell.get("cell_id")
            if (
                cell.get("runtime_kind") != runtime_kind
                or cell.get("layout") != layout
            ):
                raise LiveAstraTCOError(
                    f"{cell_id} runtime_kind/layout disagrees with {system}")
            if system == TIERING_LIVE_SYSTEM_KEY:
                _validate_non_hbf_validity(cell, oracle=False)
            elif system == ORACLE_LIVE_SYSTEM_KEY:
                _validate_non_hbf_validity(cell, oracle=True)
            else:
                runtime = runtime_reports_by_cell_id.get(str(cell_id))
                if not isinstance(runtime, Mapping):
                    raise LiveAstraTCOError(
                        f"{cell_id} is missing its verified raw HBF runtime "
                        "ledger")
                _validate_hbf_validity(
                    cell, runtime, expected_layout=str(layout))


def _student_t_critical_95(df: int) -> float:
    if df < 1:
        raise LiveAstraTCOError("Student-t degrees of freedom must be positive")
    if df < len(_STUDENT_T_975):
        return _STUDENT_T_975[df]
    # Cornish-Fisher expansion around z(0.975), accurate well beyond the
    # precision warranted by analytical price/power sensitivity inputs.
    z = 1.959963984540054
    inverse = 1.0 / df
    return (
        z
        + (z ** 3 + z) * inverse / 4.0
        + (5.0 * z ** 5 + 16.0 * z ** 3 + 3.0 * z)
        * inverse ** 2 / 96.0
        + (
            3.0 * z ** 7
            + 19.0 * z ** 5
            + 17.0 * z ** 3
            - 15.0 * z
        )
        * inverse ** 3 / 384.0
    )


def paired_seed_student_t_95(
    samples: Sequence[float],
) -> tuple[float, float | None, float | None, str]:
    """Return one system's marginal t interval over seed-aligned cells.

    The seed roster is paired across systems before this helper is called.
    This interval is not an interval over paired differences or ratios.
    """

    if not samples:
        raise LiveAstraTCOError("goodput sample roster must not be empty")
    values = tuple(
        _require_number(value, f"goodput sample[{index}]")
        for index, value in enumerate(samples)
    )
    if any(value < 0.0 for value in values):
        raise LiveAstraTCOError("goodput samples must be nonnegative")
    mean = math.fsum(values) / len(values)
    if len(values) == 1:
        return mean, None, None, "not_available_single_seed_aligned_cell"
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    margin = _student_t_critical_95(len(values) - 1) * standard_error
    return (
        mean,
        mean - margin,
        mean + margin,
        "marginal_student_t_95_over_seed_aligned_cells",
    )


def _goodput_samples(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[float, ...]:
    samples = []
    for cell in cells:
        performance = _mapping(
            cell.get("performance"),
            f"{cell.get('cell_id')}.performance",
        )
        # Deliberately do not fall back to request goodput.  A compact result
        # with only request-goodput is invalid for token economics.
        value = _require_number(
            performance.get(
                "offered_normalized_output_token_slo_goodput_per_second"),
            (
                f"{cell.get('cell_id')}."
                "performance.offered_normalized_output_token_"
                "slo_goodput_per_second"
            ),
        )
        if value < 0.0:
            raise LiveAstraTCOError(
                "output-token SLO goodput must be nonnegative")
        samples.append(value)
    return tuple(samples)


def _paired_schedule_sha(
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    seeds: tuple[int, ...],
) -> str:
    schedule = []
    for index, seed in enumerate(seeds):
        digests = {
            _require_sha256(
                selected[system][index].get("workload_sha256"),
                f"{system} seed {seed} workload_sha256",
            )
            for system in selected
        }
        if len(digests) != 1:
            raise LiveAstraTCOError(
                f"paired systems used different workloads for seed={seed}")
        schedule.append({
            "seed": seed,
            "workload_sha256": next(iter(digests)),
        })
    return _stable_sha256(schedule)


def _slo_contract(
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    campaign: Mapping[str, Any],
) -> tuple[int, int]:
    ttft = _require_int(
        campaign.get("ttft_slo_ns"),
        "campaign.ttft_slo_ns",
        minimum=1,
    )
    tpot = _require_int(
        campaign.get("tpot_slo_ns"),
        "campaign.tpot_slo_ns",
        minimum=1,
    )
    for system_cells in selected.values():
        for cell in system_cells:
            performance = _mapping(
                cell.get("performance"),
                f"{cell.get('cell_id')}.performance",
            )
            cell_ttft = _require_int(
                performance.get("ttft_slo_ns"),
                f"{cell.get('cell_id')}.performance.ttft_slo_ns",
                minimum=1,
            )
            cell_tpot = _require_int(
                performance.get("tpot_slo_ns"),
                f"{cell.get('cell_id')}.performance.tpot_slo_ns",
                minimum=1,
            )
            if (
                cell_ttft != ttft
                or cell_tpot != tpot
            ):
                raise LiveAstraTCOError(
                    f"{cell.get('cell_id')} SLO contract changed")
    return ttft, tpot


def _active_prefill_drain_contract(
    cells: Sequence[Mapping[str, Any]],
    *,
    expected_layout: str,
) -> tuple[int, int, str]:
    policies = set()
    for cell in cells:
        if cell.get("layout") != expected_layout:
            raise LiveAstraTCOError(
                f"{cell.get('cell_id')} HBF layout changed")
        validity = _mapping(
            cell.get("validity"), f"{cell.get('cell_id')}.validity")
        for field in _HBF_QUIESCENT_VALIDITY_FIELDS:
            _require_validity_value(
                validity, field, 0, cell_id=cell.get("cell_id"))
        bottlenecks = _mapping(
            cell.get("bottlenecks"),
            f"{cell.get('cell_id')}.bottlenecks",
        )
        hbf = _mapping(
            bottlenecks.get("hbf"),
            f"{cell.get('cell_id')}.bottlenecks.hbf",
        )
        drain = _mapping(
            hbf.get("prefill_drain"),
            f"{cell.get('cell_id')}.bottlenecks.hbf.prefill_drain",
        )
        policy = _mapping(
            drain.get("policy"),
            f"{cell.get('cell_id')}.prefill_drain.policy",
        )
        if set(policy) != {"tail_tokens", "min_tokens"}:
            raise LiveAstraTCOError(
                "active-prefill-drain policy has unknown or missing fields")
        tail = _require_int(
            policy.get("tail_tokens"),
            "active-prefill-drain tail_tokens",
            minimum=1,
        )
        minimum = _require_int(
            policy.get("min_tokens"),
            "active-prefill-drain min_tokens",
            minimum=1,
        )
        policies.add((tail, minimum))
    if len(policies) != 1:
        raise LiveAstraTCOError(
            "selected HBF cells used different active-prefill-drain policies")
    tail, minimum = next(iter(policies))
    contract = {
        "policy_version": 2,
        "first_turn_execution": "gpu",
        "migration_inflight_resume_execution": "gpu",
        "hbf_ready_resume_execution": "hbf",
        "turn_boundary_materialized_kv": "lpddr_then_hbf",
        "decode_release_barrier": "after_same_time_prefill_drain_flush",
        "active_prefill_drain_tail_tokens": tail,
        "active_prefill_drain_min_tokens": minimum,
    }
    return tail, minimum, _stable_sha256(contract)


def _live_hardware_variant(
    repo_root: Path,
    hbf_config_sha256: str,
    hbf_system_key: str,
) -> HBFHardwareVariant:
    config_path = repo_root / PROPOSED_HBF_CONFIG
    hardware, _ = load_hbf_server_config(config_path)
    lpddr_gib = hardware.lpddr_capacity_bytes_per_card / BYTES_PER_GIB
    return replace(
        DEFAULT_HBF_HARDWARE_VARIANT,
        variant_key=f"live_compact_v2_{hbf_system_key}",
        hbf_config_sha256=hbf_config_sha256,
        card_count=hardware.card_count,
        hbf_capacity_bytes_per_card=(
            hardware.hbf_capacity_bytes_per_card),
        hbf_capacity_ratio_to_hbm=(
            hardware.hbf_capacity_bytes_per_card
            / 80_000_000_000
        ),
        intra_fabric_bandwidth_gbps_per_card=(
            hardware.intra_fabric_bandwidth_gbps_per_card),
        lpddr_effective_bandwidth_gbps_per_card=(
            hardware.lpddr_bandwidth_gbps_per_card),
        lpddr_capacity_gib_per_card=lpddr_gib,
        rdma_bandwidth_gbps=hardware.rdma_bandwidth_gbps,
        rdma_one_way_latency_us=hardware.rdma_one_way_latency_us,
        cost_power_assumption=(
            "Effective hardware values are parsed from the hash-pinned "
            "live full-model HBF config; analytical component ratios remain "
            "the declared TCO sensitivity assumptions."
        ),
    )


def adapt_collected_campaign(
    compact: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    selected_rate_per_second: float,
    selected_hbf_system_key: str,
    manifest_sha256: str,
    compact_results_sha256: str,
    runtime_reports_by_cell_id: Mapping[str, Mapping[str, Any]],
    axes: SensitivityAxes = SensitivityAxes(),
) -> TCOSensitivityReport:
    """Build a TCO report from an already canonical-collected campaign."""

    repo_root = repo_root.resolve()
    rate = _require_number(
        selected_rate_per_second,
        "selected_rate_per_second",
        positive=True,
    )
    if selected_hbf_system_key not in HBF_LIVE_LAYOUTS:
        raise LiveAstraTCOError(
            "selected_hbf_system_key must be one of "
            f"{tuple(HBF_LIVE_LAYOUTS)!r}")
    _require_sha256(manifest_sha256, "manifest_sha256")
    _require_sha256(compact_results_sha256, "compact_results_sha256")
    (
        campaign,
        config_hashes,
        implementation_sha,
        astra_sha,
        collector_sha,
    ) = _validate_campaign_and_compact(
        compact, manifest, repo_root=repo_root)

    campaign_rates = tuple(
        _require_number(value, "campaign.rates[]", positive=True)
        for value in _sequence(campaign.get("rates"), "campaign.rates")
    )
    if (
        not campaign_rates
        or len(campaign_rates) != len(set(campaign_rates))
    ):
        raise LiveAstraTCOError(
            "campaign rates must be a non-empty unique ordered grid")
    if rate not in campaign_rates:
        raise LiveAstraTCOError(
            f"selected rate {rate} is not in the campaign grid")
    seed_values = tuple(
        _require_int(value, "campaign.seeds[]", minimum=0)
        for value in _sequence(campaign.get("seeds"), "campaign.seeds")
    )
    if not seed_values or len(seed_values) != len(set(seed_values)):
        raise LiveAstraTCOError(
            "campaign seeds must be a non-empty unique ordered roster")

    selected = _selected_cells(
        compact,
        rate=rate,
        hbf_system_key=selected_hbf_system_key,
        seeds=seed_values,
    )
    if not isinstance(runtime_reports_by_cell_id, Mapping):
        raise LiveAstraTCOError(
            "runtime_reports_by_cell_id must be a mapping")
    _validate_selected_cell_contracts(
        selected,
        hbf_system_key=selected_hbf_system_key,
        runtime_reports_by_cell_id=runtime_reports_by_cell_id,
    )
    schedule_sha = _paired_schedule_sha(selected, seed_values)
    ttft_slo_ns, tpot_slo_ns = _slo_contract(selected, campaign)
    hbf_layout = HBF_LIVE_LAYOUTS[selected_hbf_system_key]
    drain_tail, drain_minimum, drain_contract_sha = (
        _active_prefill_drain_contract(
            selected[selected_hbf_system_key],
            expected_layout=hbf_layout,
        )
    )

    goodput_samples = {
        TIERING_SYSTEM_KEY: _goodput_samples(
            selected[TIERING_LIVE_SYSTEM_KEY]),
        PROPOSED_SYSTEM_KEY: _goodput_samples(
            selected[selected_hbf_system_key]),
        ORACLE_SYSTEM_KEY: _goodput_samples(
            selected[ORACLE_LIVE_SYSTEM_KEY]),
    }
    aggregates = {
        key: paired_seed_student_t_95(values)
        for key, values in goodput_samples.items()
    }
    methods = {aggregate[3] for aggregate in aggregates.values()}
    if len(methods) != 1:
        raise LiveAstraTCOError(
            "paired systems disagree on confidence-interval method")

    scenario_id = campaign.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise LiveAstraTCOError("campaign.scenario_id is invalid")
    scenario_manifest_sha = _require_sha256(
        campaign.get("scenario_manifest_sha256"),
        "campaign.scenario_manifest_sha256",
    )
    measurement_sha = _require_sha256(
        campaign.get("measurement_session_ids_sha256"),
        "campaign.measurement_session_ids_sha256",
    )
    scenario_source_sha = _require_sha256(
        campaign.get("scenario_source_sha256"),
        "campaign.scenario_source_sha256",
    )
    schedule_semantics = (
        "single_frozen_schedule"
        if len(seed_values) == 1
        else "ordered_paired_seed_schedule_set_manifest"
    )
    aggregation = (
        "single_seed_aligned_cell_value"
        if len(seed_values) == 1
        else "arithmetic_mean_across_seed_aligned_cells"
    )

    def result_provenance(
        system_key: str,
    ) -> GoodputResultProvenance:
        mean, lower, upper, ci_method = aggregates[system_key]
        return GoodputResultProvenance(
            system_key=system_key,
            slo_good_output_tokens_per_second=mean,
            offered_session_rate_per_second=rate,
            scenario_id=scenario_id,
            cohort_id=(
                f"{scenario_id}:measurement:"
                f"{measurement_sha[:16]}"
            ),
            schedule_sha256=schedule_sha,
            schedule_hash_semantics=schedule_semantics,
            measurement_cohort_sha256=measurement_sha,
            result_goodput_origin=(
                "canonical compact-v2 "
                f"{LIVE_COMPACT_OUTPUT_TOKEN_GOODPUT_JSON_PATH}; "
                f"system={system_key}; rate={rate:.17g}"
            ),
            result_manifest_sha256=manifest_sha256,
            result_schema_revision="live-astra-compact-v2",
            simulator_code_revision=implementation_sha,
            metric_scope="all",
            metric_json_path=(
                LIVE_COMPACT_OUTPUT_TOKEN_GOODPUT_JSON_PATH),
            metric_definition=OUTPUT_TOKEN_GOODPUT_DEFINITION,
            aggregation_method=aggregation,
            seed_count=len(seed_values),
            confidence_interval_method=ci_method,
            confidence_interval_lower_tokens_per_second=lower,
            confidence_interval_upper_tokens_per_second=upper,
        )

    deployment_snapshot = _deployment_semantic_snapshot(repo_root)
    capacity_snapshot = _capacity_semantic_snapshot(repo_root)
    live_provenance = LiveComparisonArtifactProvenance(
        campaign_sha256=_require_sha256(
            manifest.get("campaign_sha256"),
            "manifest.campaign_sha256",
        ),
        manifest_sha256=manifest_sha256,
        compact_results_sha256=compact_results_sha256,
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        compact_schema_version=COMPACT_SCHEMA_VERSION,
        scenario_source_sha256=scenario_source_sha,
        scenario_manifest_sha256=scenario_manifest_sha,
        measurement_cohort_sha256=measurement_sha,
        simulator_implementation_sha256=implementation_sha,
        astra_binary_sha256=astra_sha,
        canonical_collector_source_sha256=collector_sha,
        tco_adapter_implementation_sha256=(
            _tco_adapter_implementation_sha(
                repo_root, capacity_snapshot)),
        deployment_semantic_snapshot=deployment_snapshot,
        deployment_semantic_snapshot_sha256=(
            _stable_sha256(deployment_snapshot)),
        capacity_semantic_snapshot=capacity_snapshot,
        capacity_semantic_snapshot_sha256=(
            _stable_sha256(capacity_snapshot)),
        confidence_interval_semantics=MARGINAL_CI_SEMANTICS,
        tiering_cluster_config_path=TIERING_CLUSTER_CONFIG,
        tiering_cluster_config_sha256=config_hashes["tiering_cluster"],
        tiering_policy_config_path=TIERING_POLICY_CONFIG,
        tiering_policy_config_sha256=config_hashes["tiering_policy"],
        proposed_gpu_cluster_config_path=PROPOSED_GPU_CLUSTER_CONFIG,
        proposed_gpu_cluster_config_sha256=(
            config_hashes["proposed_gpu_cluster"]),
        proposed_hbf_config_path=PROPOSED_HBF_CONFIG,
        proposed_hbf_config_sha256=config_hashes["proposed_hbf"],
        oracle_cluster_config_path=ORACLE_CLUSTER_CONFIG,
        oracle_cluster_config_sha256=config_hashes["oracle_cluster"],
        oracle_policy_config_path=ORACLE_POLICY_CONFIG,
        oracle_policy_config_sha256=config_hashes["oracle_policy"],
        selected_rate_per_second=rate,
        selected_hbf_system_key=selected_hbf_system_key,
        selected_hbf_layout_key=hbf_layout,
        paired_workload_schedule_sha256=schedule_sha,
        active_prefill_drain_policy_version=2,
        active_prefill_drain_tail_tokens=drain_tail,
        active_prefill_drain_min_tokens=drain_minimum,
        active_prefill_drain_policy_contract_sha256=drain_contract_sha,
    )
    variant = _live_hardware_variant(
        repo_root,
        config_hashes["proposed_hbf"],
        selected_hbf_system_key,
    )
    performance_provenance = ComparisonPerformanceProvenance(
        selected_tiering_policy_key="cpu_ssd",
        hbf_layout_key=selected_hbf_system_key,
        hbf_policy_key=ACTIVE_PREFILL_DRAIN_POLICY_KEY_V2,
        hbf_policy_contract_sha256=drain_contract_sha,
        gpu_config_sha256=config_hashes["proposed_gpu_cluster"],
        hbf_hardware_variant=variant,
        first_ttft_slo_ns=ttft_slo_ns,
        resume_ttft_slo_ns=ttft_slo_ns,
        tpot_slo_ns=tpot_slo_ns,
        operating_point_mode="matched_single_operating_point",
        rate_selection_semantics=(
            "Explicit user-selected finite campaign-grid rate; paired systems "
            "and seed-aligned workloads; per-system confidence intervals are "
            "marginal rather than paired-difference intervals; not a maximum "
            "sustainable-throughput claim."
        ),
        maximum_slo_sustainable_claim=False,
        tiering_result=result_provenance(TIERING_SYSTEM_KEY),
        proposed_result=result_provenance(PROPOSED_SYSTEM_KEY),
        oracle_result=result_provenance(ORACLE_SYSTEM_KEY),
        live_artifact_provenance=live_provenance,
    )
    goodputs = {
        key: aggregate[0] for key, aggregate in aggregates.items()}
    report = evaluate_tco_sensitivity(
        goodputs,
        performance_provenance=performance_provenance,
        axes=axes,
    )
    selected_capacity = _mapping(
        _mapping(
            capacity_snapshot.get("layouts"),
            "capacity snapshot layouts",
        ).get(selected_hbf_system_key),
        "selected capacity snapshot",
    )
    disclosure = report.memory_capacity
    expected_capacity_fields = {
        "layout_key": disclosure.selected_hbf_layout_key,
        "tp_size": disclosure.selected_hbf_tp_size,
        "replicas": disclosure.selected_hbf_replica_count,
        "model_weight_bytes_per_card": (
            disclosure.hbf_model_weight_bytes_per_card),
        "physical_kv_replication_factor": (
            disclosure.selected_hbf_physical_kv_replication_factor),
        "usable_logical_hbf_kv_capacity_bytes": (
            disclosure.proposed_usable_logical_hbf_kv_capacity_bytes),
    }
    if dict(selected_capacity) != expected_capacity_fields:
        raise LiveAstraTCOError(
            "capacity semantic snapshot disagrees with TCO disclosure")
    proposed_cost = report.sensitivity_rows[0].proposed_cost
    bom_quantities = {
        "tiering_cpu_hosts": int(
            report.tiering_cost.component("cpu_host_base").quantity),
        "tiering_h100_cards": int(
            report.tiering_cost.component("h100_gpu_logic").quantity),
        "tiering_ssd_devices": int(
            report.tiering_cost.component("nvme_ssd_tier").quantity),
        "proposed_cpu_hosts": int(
            proposed_cost.component("cpu_host_base").quantity),
        "proposed_h100_cards": int(
            proposed_cost.component("h100_gpu_logic").quantity),
        "proposed_hbf_npu_cards": int(
            proposed_cost.component("hbf_npu_logic").quantity),
        "proposed_lpddr_gib": int(
            proposed_cost.component("hbf_card_lpddr").quantity),
        "proposed_ssd_devices": int(
            proposed_cost.component("nvme_ssd_tier").quantity),
    }
    if any(
        bom_quantities[key] != deployment_snapshot[key]
        for key in bom_quantities
    ):
        raise LiveAstraTCOError(
            "parsed deployment semantic snapshot disagrees with generated "
            f"TCO BOM: parsed={deployment_snapshot!r}, "
            f"bom={bom_quantities!r}")
    tiering_host_dram_bytes = round(
        report.tiering_cost.component("host_dram").quantity
        * BYTES_PER_GIB
    )
    proposed_host_dram_bytes = round(
        proposed_cost.component("host_dram").quantity * BYTES_PER_GIB)
    if tiering_host_dram_bytes != (
            deployment_snapshot["tiering_cpu_hosts"]
            * deployment_snapshot["tiering_cpu_dram_bytes_per_host"]):
        raise LiveAstraTCOError(
            "tiering host-DRAM BOM disagrees with deployment semantics")
    if proposed_host_dram_bytes != (
            deployment_snapshot["proposed_gpu_cpu_hosts"]
            * deployment_snapshot["proposed_gpu_host_cpu_dram_bytes"]
            + deployment_snapshot["proposed_hbf_cpu_hosts"]
            * deployment_snapshot["proposed_hbf_host_cpu_dram_bytes"]):
        raise LiveAstraTCOError(
            "proposed host-DRAM BOM does not include the explicit HBF-host "
            "DRAM assumption")
    if report.anchors.h100_hbm_capacity_bytes_per_card != (
            deployment_snapshot["h100_hbm_bytes_per_card"]):
        raise LiveAstraTCOError(
            "H100 HBM capacity anchor disagrees with deployment semantics")
    return report


def _resolve_recorded_path(value: object, *, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise LiveAstraTCOError(f"{name} path is invalid")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_selected_hbf_runtime_reports(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    selected_hbf_system_key: str,
    selected_rate_per_second: float,
) -> dict[str, Mapping[str, Any]]:
    reports = {}
    cells = _mapping(manifest.get("cells"), "manifest.cells")
    for cell_id, item in cells.items():
        entry = _mapping(item, f"manifest.cells[{cell_id!r}]")
        if (
            entry.get("system") != selected_hbf_system_key
            or entry.get("rate") != selected_rate_per_second
        ):
            continue
        result_path = _resolve_recorded_path(
            entry.get("result"),
            base=manifest_path.parent,
            name=f"{cell_id}.result",
        )
        expected_result_sha = _require_sha256(
            entry.get("result_sha256"), f"{cell_id}.result_sha256")
        expected_result_bytes = _require_int(
            entry.get("result_bytes"), f"{cell_id}.result_bytes", minimum=1)
        if (
            not result_path.is_file()
            or result_path.stat().st_size != expected_result_bytes
            or _sha256_file(result_path) != expected_result_sha
        ):
            raise LiveAstraTCOError(
                f"{cell_id} result artifact changed")
        result = _strict_object(result_path)
        result_schema = _require_int(
            result.get("schema_version"),
            f"{cell_id}.result.schema_version",
            minimum=0,
        )
        result_seed = _require_int(
            result.get("seed"), f"{cell_id}.result.seed", minimum=0)
        manifest_seed = _require_int(
            entry.get("seed"), f"{cell_id}.manifest.seed", minimum=0)
        result_rate = _require_number(
            result.get("offered_session_rate_per_second"),
            f"{cell_id}.result.offered_session_rate_per_second",
            positive=True,
        )
        if (
            result_schema != 2
            or result.get("cell_id") != cell_id
            or result.get("system") != selected_hbf_system_key
            or result.get("runtime_kind") != "full_model_hbf"
            or result.get("layout") != HBF_LIVE_LAYOUTS[
                selected_hbf_system_key]
            or result_seed != manifest_seed
            or result_rate != selected_rate_per_second
        ):
            raise LiveAstraTCOError(
                f"{cell_id} raw result identity changed")
        artifacts = _mapping(
            result.get("artifacts"), f"{cell_id}.artifacts")
        record = _mapping(
            artifacts.get("runtime_report"),
            f"{cell_id}.artifacts.runtime_report",
        )
        runtime_path = _resolve_recorded_path(
            record.get("path"),
            base=result_path.parent,
            name=f"{cell_id}.runtime_report",
        )
        expected_sha = _require_sha256(
            record.get("sha256"),
            f"{cell_id}.runtime_report.sha256",
        )
        expected_bytes = _require_int(
            record.get("bytes"),
            f"{cell_id}.runtime_report.bytes",
            minimum=1,
        )
        if (
            not runtime_path.is_file()
            or runtime_path.stat().st_size != expected_bytes
            or _sha256_file(runtime_path) != expected_sha
        ):
            raise LiveAstraTCOError(
                f"{cell_id} raw HBF runtime report changed")
        reports[str(cell_id)] = _strict_object(runtime_path)
    if not reports:
        raise LiveAstraTCOError(
            "manifest has no selected HBF runtime reports")
    return reports


def load_and_adapt_live_campaign(
    manifest_path: str | Path,
    compact_path: str | Path,
    *,
    repo_root: str | Path,
    selected_rate_per_second: float,
    selected_hbf_system_key: str,
    axes: SensitivityAxes = SensitivityAxes(),
) -> TCOSensitivityReport:
    """Canonical-recollect, compare, and adapt a live campaign."""

    manifest_file = Path(manifest_path).resolve()
    compact_file = Path(compact_path).resolve()
    manifest = _strict_object(manifest_file)
    compact = _strict_object(compact_file)
    canonical = collect_campaign(manifest_file)
    if compact != canonical:
        raise LiveAstraTCOError(
            "supplied compact-v2 artifact differs from a fresh canonical "
            "collector result")
    runtime_reports = _load_selected_hbf_runtime_reports(
        manifest,
        manifest_path=manifest_file,
        selected_hbf_system_key=selected_hbf_system_key,
        selected_rate_per_second=float(selected_rate_per_second),
    )
    return adapt_collected_campaign(
        compact,
        manifest,
        repo_root=Path(repo_root),
        selected_rate_per_second=selected_rate_per_second,
        selected_hbf_system_key=selected_hbf_system_key,
        manifest_sha256=_sha256_file(manifest_file),
        compact_results_sha256=_sha256_file(compact_file),
        runtime_reports_by_cell_id=runtime_reports,
        axes=axes,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


def write_tco_json(
    report: TCOSensitivityReport,
    output_path: str | Path,
) -> None:
    payload = (
        json.dumps(
            report.to_json_dict(),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(Path(output_path), payload)


def write_tco_csv(
    report: TCOSensitivityReport,
    output_path: str | Path,
) -> None:
    provenance = report.performance_provenance
    live = provenance.live_artifact_provenance
    if live is None:
        raise LiveAstraTCOError(
            "CSV export requires live artifact provenance")
    results = {
        TIERING_SYSTEM_KEY: provenance.tiering_result,
        PROPOSED_SYSTEM_KEY: provenance.proposed_result,
        ORACLE_SYSTEM_KEY: provenance.oracle_result,
    }
    if results[ORACLE_SYSTEM_KEY] is None:
        raise LiveAstraTCOError(
            "live comparison CSV requires the Oracle reference")
    capacity = report.memory_capacity
    rows = []
    for row in report.sensitivity_rows:
        rows.append({
            "scenario_key": row.scenario_key,
            "selected_rate_per_second": live.selected_rate_per_second,
            "selected_hbf_system_key": live.selected_hbf_system_key,
            "selected_hbf_layout_key": live.selected_hbf_layout_key,
            "paired_seed_count": provenance.tiering_result.seed_count,
            "tiering_output_token_slo_goodput_per_second": (
                results[TIERING_SYSTEM_KEY]
                .slo_good_output_tokens_per_second),
            "tiering_goodput_ci95_lower": (
                results[TIERING_SYSTEM_KEY]
                .confidence_interval_lower_tokens_per_second),
            "tiering_goodput_ci95_upper": (
                results[TIERING_SYSTEM_KEY]
                .confidence_interval_upper_tokens_per_second),
            "proposed_output_token_slo_goodput_per_second": (
                results[PROPOSED_SYSTEM_KEY]
                .slo_good_output_tokens_per_second),
            "proposed_goodput_ci95_lower": (
                results[PROPOSED_SYSTEM_KEY]
                .confidence_interval_lower_tokens_per_second),
            "proposed_goodput_ci95_upper": (
                results[PROPOSED_SYSTEM_KEY]
                .confidence_interval_upper_tokens_per_second),
            "oracle_output_token_slo_goodput_per_second": (
                results[ORACLE_SYSTEM_KEY]
                .slo_good_output_tokens_per_second),
            "oracle_goodput_ci95_lower": (
                results[ORACLE_SYSTEM_KEY]
                .confidence_interval_lower_tokens_per_second),
            "oracle_goodput_ci95_upper": (
                results[ORACLE_SYSTEM_KEY]
                .confidence_interval_upper_tokens_per_second),
            "npu_logic_capex_ratio_to_gpu_logic": (
                row.sensitivity.npu_logic_capex_ratio_to_gpu_logic),
            "hbf_subsystem_capex_ratio_to_hbm_stack": (
                row.sensitivity.hbf_subsystem_capex_ratio_to_hbm_stack),
            "npu_logic_power_ratio_to_gpu_logic": (
                row.sensitivity.npu_logic_power_ratio_to_gpu_logic),
            "hbf_subsystem_power_ratio_to_hbm_stack": (
                row.sensitivity.hbf_subsystem_power_ratio_to_hbm_stack),
            "tco_lifetime_years": row.proposed_cost.lifetime_years,
            "tiering_lifetime_tco_usd": (
                row.tiering_cost.lifetime_tco_usd),
            "proposed_lifetime_tco_usd": (
                row.proposed_cost.lifetime_tco_usd),
            "tiering_it_power_w": row.tiering_cost.it_power_w,
            "proposed_it_power_w": row.proposed_cost.it_power_w,
            "incremental_it_power_w": (
                row.proposed_cost.it_power_w
                - row.tiering_cost.it_power_w),
            "proposed_it_power_ratio_to_tiering": (
                row.proposed_cost.it_power_w
                / row.tiering_cost.it_power_w),
            "tiering_lifetime_facility_energy_kwh": (
                row.tiering_cost.lifetime_facility_energy_kwh),
            "proposed_lifetime_facility_energy_kwh": (
                row.proposed_cost.lifetime_facility_energy_kwh),
            "incremental_lifetime_facility_energy_kwh": (
                row.proposed_cost.lifetime_facility_energy_kwh
                - row.tiering_cost.lifetime_facility_energy_kwh),
            "proposed_facility_energy_ratio_to_tiering": (
                row.proposed_cost.lifetime_facility_energy_kwh
                / row.tiering_cost.lifetime_facility_energy_kwh),
            "proposed_tco_ratio_to_tiering": (
                row.proposed_tco_ratio_to_tiering),
            "proposed_goodput_ratio_to_tiering": (
                row.proposed_goodput_ratio_to_tiering),
            "proposed_tokens_per_usd_ratio_to_tiering": (
                row.proposed_tokens_per_usd_ratio_to_tiering),
            "break_even_proposed_goodput_tokens_per_second": (
                row.break_even_proposed_goodput_tokens_per_second),
            "proposed_meets_token_value_break_even": (
                row.proposed_meets_or_exceeds_token_value_break_even),
            "usable_logical_hbf_kv_capacity_bytes": (
                capacity.proposed_usable_logical_hbf_kv_capacity_bytes),
            "physical_kv_replication_factor": (
                capacity.selected_hbf_physical_kv_replication_factor),
            "campaign_sha256": live.campaign_sha256,
            "manifest_sha256": live.manifest_sha256,
            "compact_results_sha256": live.compact_results_sha256,
            "simulator_implementation_sha256": (
                live.simulator_implementation_sha256),
            "canonical_collector_source_sha256": (
                live.canonical_collector_source_sha256),
            "tco_adapter_implementation_sha256": (
                live.tco_adapter_implementation_sha256),
            "deployment_semantic_snapshot_sha256": (
                live.deployment_semantic_snapshot_sha256),
            "deployment_semantic_snapshot_json": json.dumps(
                live.deployment_semantic_snapshot,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ),
            "capacity_semantic_snapshot_sha256": (
                live.capacity_semantic_snapshot_sha256),
            "confidence_interval_semantics": (
                live.confidence_interval_semantics),
            "paired_workload_schedule_sha256": (
                live.paired_workload_schedule_sha256),
            "active_prefill_drain_policy_contract_sha256": (
                live.active_prefill_drain_policy_contract_sha256),
            "oracle_included_in_tco_bom": False,
        })
    if not rows:
        raise LiveAstraTCOError("TCO report has no sensitivity rows")
    fields = list(rows[0])
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as tmp:
        writer = csv.DictWriter(
            tmp, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        tmp.seek(0)
        payload = tmp.read().encode("utf-8")
    _atomic_write(Path(output_path), payload)


def _rate_tag(rate: float) -> str:
    return format(rate, ".17g").replace("-", "m").replace(".", "p")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m serving.live_astra_comparison_tco",
        description=(
            "Strictly adapt compact-v2 live ASTRA output-token goodput into "
            "the physical baseline-versus-HBF TCO sensitivity model"
        ),
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("compact", type=Path)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument(
        "--hbf-system",
        choices=tuple(HBF_LIVE_LAYOUTS),
        required=True,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_stem = (
        f"live_tco_{args.hbf_system}_rate{_rate_tag(args.rate)}")
    output_json = (
        args.output_json
        if args.output_json is not None
        else args.compact.resolve().parent / f"{output_stem}.json"
    )
    output_csv = (
        args.output_csv
        if args.output_csv is not None
        else args.compact.resolve().parent / f"{output_stem}.csv"
    )
    report = load_and_adapt_live_campaign(
        args.manifest,
        args.compact,
        repo_root=args.repo_root,
        selected_rate_per_second=args.rate,
        selected_hbf_system_key=args.hbf_system,
    )
    write_tco_json(report, output_json)
    write_tco_csv(report, output_csv)
    print(json.dumps({
        "economic_system_keys": list(report.economic_system_keys),
        "oracle_included_in_tco_bom": False,
        "output_csv": str(output_csv.resolve()),
        "output_json": str(output_json.resolve()),
        "selected_hbf_system": args.hbf_system,
        "selected_rate_per_second": args.rate,
        "sensitivity_row_count": len(report.sensitivity_rows),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate and collect the twelve Qwen3 1M calibration shards.

Each shard manifest remains the authoritative record for its three reports.
This collector verifies those records and produces deterministic cross-shard
tables plus a root index; it does not reinterpret experiment values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agentic_kv_qwen3_1m_p4d4 import (
    ComputeEndpoint,
    ReserveCase,
    build_return_source_rows,
    build_summary_row,
    build_transfer_stage_rows,
)


RESERVE_CASES = (
    ("zero", "zero_residual"),
    ("half", "half_residual"),
    ("full", "full_residual"),
)
COMPUTE_ENDPOINTS = (
    "central_full_attention",
    "central_attention_one_third",
    "fast_full_attention",
    "slow_full_attention",
)
POLICIES = (
    "hbm_lru_recompute",
    "hbm_ssd_direct",
    "tiered",
)
RETURN_SOURCES = ("decode_hbm", "cpu", "ssd", "recompute")
TABLE_NAMES = (
    "summary.csv",
    "return_sources.csv",
    "transfer_stages.csv",
)
ROOT_MANIFEST_NAME = "manifest.json"
KNOWN_COLLECTOR_OUTPUTS = (*TABLE_NAMES, ROOT_MANIFEST_NAME)
REPORT_SCHEMA_VERSION = 15
SHARD_MANIFEST_SCHEMA_VERSION = 3
CALIBRATION_ARTIFACT_SCHEMA_VERSION = 3
CROSS_SHARD_MANIFEST_FIELDS = (
    "experiment_contract",
    "model_geometry",
    "official_qwen_sources",
    "hardware_sources",
    "evidence_classes",
    "repository_profile_evidence",
    "excluded_repository_profile_evidence",
    "calibration_boundary",
)
RESOLVED_CONFIG_VARIANT_FIELDS = (
    "policy",
    "hbm_static_reserve_bytes_per_rank",
    "prefill_hbm_static_reserve_bytes_per_rank",
    "decode_hbm_static_reserve_bytes_per_rank",
    "prompt_compute_scale_provenance",
)


class CollectionError(ValueError):
    """Raised when a shard or cross-shard binding is invalid."""


@dataclass(frozen=True)
class CsvTable:
    path: Path
    header: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class Shard:
    name: str
    reserve_label: str
    reserve_case: str
    endpoint: str
    path: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    manifest_bytes: int
    tables: Mapping[str, CsvTable]
    report_records: tuple[Mapping[str, Any], ...]
    workload: Mapping[str, Any]
    workload_sha256: str
    workload_provenance: Mapping[str, Any]
    base_calibration_metadata_sha256: str
    source_sha256: Mapping[str, str]
    producer_source_sha256: Mapping[str, str]
    endpoint_metadata_sha256: str
    endpoint_configuration: Mapping[str, Any]
    calibration_work_contract: Mapping[str, Any]
    resolved_config_by_policy: Mapping[str, Mapping[str, Any]]
    local_source_sha256: Mapping[str, str]
    git_provenance: Mapping[str, Any]
    git_commit: str
    git_dirty: bool


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CollectionError(f"{label} must be a JSON object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CollectionError(f"{label} must be a JSON array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CollectionError(f"{label} must be a nonempty string")
    return value


def _sha256_string(value: Any, label: str) -> str:
    digest = _string(value, label).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise CollectionError(f"{label} must be a SHA-256 digest")
    return digest


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CollectionError(f"{label} must be an integer")
    return value


def _sha256_mapping(value: Any, label: str) -> dict[str, str]:
    mapping = _mapping(value, label)
    if not mapping:
        raise CollectionError(f"{label} must not be empty")
    normalized: dict[str, str] = {}
    for raw_path, raw_hash in mapping.items():
        path = _string(raw_path, f"{label} path")
        digest = _sha256_string(raw_hash, f"{label} digest for {path}")
        normalized[path] = digest
    return normalized


def _read_json(path: Path, label: str) -> tuple[Mapping[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise CollectionError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CollectionError(f"cannot read {label}: {path}: {exc}") from exc
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollectionError(f"invalid JSON for {label}: {path}: {exc}") from exc
    return _mapping(value, label), raw


def _read_csv(path: Path, label: str) -> CsvTable:
    if path.is_symlink() or not path.is_file():
        raise CollectionError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CollectionError(f"cannot read {label}: {path}: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames or any(not name for name in reader.fieldnames):
        raise CollectionError(f"{label} has an empty or invalid CSV header")
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise CollectionError(f"{label} has duplicate CSV columns")
    rows: list[dict[str, str]] = []
    for index, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise CollectionError(f"{label} has a malformed row at line {index}")
        rows.append({name: row[name] for name in reader.fieldnames})
    if not rows:
        raise CollectionError(f"{label} must contain at least one data row")
    return CsvTable(
        path=path,
        header=tuple(reader.fieldnames),
        rows=tuple(rows),
        sha256=_sha256_bytes(raw),
        byte_count=len(raw),
    )


def _safe_child(directory: Path, name: Any, label: str) -> Path:
    filename = _string(name, label)
    candidate = Path(filename)
    if candidate.name != filename or candidate.is_absolute():
        raise CollectionError(f"{label} must be a plain filename: {filename}")
    return directory / filename


def _validate_declared_hash(
    actual: str, declared: Any, label: str
) -> None:
    expected = _string(declared, label).lower()
    if actual != expected:
        raise CollectionError(
            f"{label} mismatch: declared {expected}, actual {actual}"
        )


def _csv_cell(value: Any) -> str:
    return "" if value is None else str(value)


def _validate_derived_table(
    table: CsvTable,
    expected_rows: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    if not expected_rows:
        raise CollectionError(f"{label} expected rows must not be empty")
    expected_header = tuple(expected_rows[0])
    if table.header != expected_header:
        raise CollectionError(
            f"{label} header differs from the report-derived schema"
        )
    if any(tuple(row) != expected_header for row in expected_rows):
        raise CollectionError(f"{label} report-derived rows have mixed schemas")
    expected = sorted(
        tuple(_csv_cell(row[column]) for column in expected_header)
        for row in expected_rows
    )
    actual = sorted(
        tuple(row[column] for column in table.header) for row in table.rows
    )
    if actual != expected:
        raise CollectionError(
            f"{label} values differ from rows derived from the bound reports"
        )


def _validate_table_scope(
    table: CsvTable,
    *,
    shard_name: str,
    reserve_case: str,
    endpoint: str,
    run_id_by_policy: Mapping[str, str],
) -> None:
    required = {"run_id", "compute_endpoint", "reserve_case", "baseline"}
    missing = sorted(required - set(table.header))
    if missing:
        raise CollectionError(
            f"{shard_name}/{table.path.name} lacks columns: "
            + ", ".join(missing)
        )
    for index, row in enumerate(table.rows, start=2):
        policy = row["baseline"]
        if policy not in run_id_by_policy:
            raise CollectionError(
                f"{shard_name}/{table.path.name}:{index} has unexpected "
                f"policy {policy!r}"
            )
        expected = {
            "run_id": run_id_by_policy[policy],
            "compute_endpoint": endpoint,
            "reserve_case": reserve_case,
        }
        for field, expected_value in expected.items():
            if row[field] != expected_value:
                raise CollectionError(
                    f"{shard_name}/{table.path.name}:{index} {field} "
                    f"is {row[field]!r}, expected {expected_value!r}"
                )
    identity_columns: tuple[str, ...] | None = None
    if table.path.name == "return_sources.csv":
        identity_columns = ("baseline", "return_gap_type", "source")
    elif table.path.name == "transfer_stages.csv":
        identity_columns = ("baseline", "stage")
    if identity_columns is not None:
        missing_identity = sorted(set(identity_columns) - set(table.header))
        if missing_identity:
            raise CollectionError(
                f"{shard_name}/{table.path.name} lacks columns: "
                + ", ".join(missing_identity)
            )
        identities = [
            tuple(row[column] for column in identity_columns)
            for row in table.rows
        ]
        if len(set(identities)) != len(identities):
            raise CollectionError(
                f"{shard_name}/{table.path.name} has duplicate logical rows"
            )
        if table.path.name == "return_sources.csv":
            unexpected_sources = sorted(
                {row["source"] for row in table.rows} - set(RETURN_SOURCES)
            )
            if unexpected_sources:
                raise CollectionError(
                    f"{shard_name}/{table.path.name} has unexpected sources: "
                    + ", ".join(unexpected_sources)
                )
            sources_by_class: dict[tuple[str, str], set[str]] = {}
            for row in table.rows:
                sources_by_class.setdefault(
                    (row["baseline"], row["return_gap_type"]), set()
                ).add(row["source"])
            incomplete = [
                f"{policy}/{gap_type}"
                for (policy, gap_type), sources in sources_by_class.items()
                if sources != set(RETURN_SOURCES)
            ]
            if incomplete:
                raise CollectionError(
                    f"{shard_name}/{table.path.name} has incomplete source "
                    "groups: " + ", ".join(sorted(incomplete))
                )
        covered_policies = {row["baseline"] for row in table.rows}
        if covered_policies != set(POLICIES):
            raise CollectionError(
                f"{shard_name}/{table.path.name} does not cover every policy"
            )


def _validate_report(
    *,
    shard_name: str,
    shard_path: Path,
    record: Mapping[str, Any],
    expected_policy: str,
    reserve_case: str,
    endpoint: str,
    endpoint_configuration: Mapping[str, Any],
    reserve_configuration: Mapping[str, Any],
    endpoint_metadata: Mapping[str, Any],
    endpoint_metadata_sha256: str,
    experiment_contract: Mapping[str, Any],
) -> tuple[str, str, Mapping[str, Any], Mapping[str, Any]]:
    policy = _string(record.get("policy"), f"{shard_name} run policy")
    if policy != expected_policy:
        raise CollectionError(
            f"{shard_name} run policy order mismatch: {policy} != "
            f"{expected_policy}"
        )
    if record.get("compute_endpoint") != endpoint:
        raise CollectionError(f"{shard_name} run endpoint mismatch")
    if record.get("reserve_case") != reserve_case:
        raise CollectionError(f"{shard_name} run reserve mismatch")
    _validate_declared_hash(
        endpoint_metadata_sha256,
        record.get("prompt_compute_calibration_metadata_sha256"),
        f"{shard_name}/{policy} run calibration metadata sha256",
    )
    if record.get("calibration_band") != endpoint_configuration.get("band"):
        raise CollectionError(f"{shard_name}/{policy} calibration band mismatch")
    if record.get("attention_multiplier") != endpoint_configuration.get(
        "attention_multiplier"
    ):
        raise CollectionError(
            f"{shard_name}/{policy} attention multiplier mismatch"
        )
    if _integer(
        record.get("report_schema_version"),
        f"{shard_name}/{policy} manifest report schema",
    ) != REPORT_SCHEMA_VERSION:
        raise CollectionError(
            f"{shard_name}/{policy} manifest report schema must be "
            f"{REPORT_SCHEMA_VERSION}"
        )

    report_path = _safe_child(
        shard_path,
        record.get("report"),
        f"{shard_name}/{policy} report path",
    )
    report, report_raw = _read_json(report_path, f"{shard_name}/{policy} report")
    report_hash = _sha256_bytes(report_raw)
    _validate_declared_hash(
        report_hash,
        record.get("report_sha256"),
        f"{shard_name}/{policy} manifest report sha256",
    )
    if _integer(
        report.get("schema_version"), f"{shard_name}/{policy} report schema"
    ) != REPORT_SCHEMA_VERSION:
        raise CollectionError(
            f"{shard_name}/{policy} report schema must be "
            f"{REPORT_SCHEMA_VERSION}"
        )

    run_id = _string(record.get("run_id"), f"{shard_name}/{policy} run_id")
    expected_run_id = f"{reserve_case}__{endpoint}__{policy}"
    if run_id != expected_run_id:
        raise CollectionError(
            f"{shard_name}/{policy} run_id must be {expected_run_id}"
        )
    if report_path.name != f"{run_id}.json":
        raise CollectionError(
            f"{shard_name}/{policy} report filename does not match run_id"
        )
    experiment = _mapping(
        report.get("experiment"), f"{shard_name}/{policy} report experiment"
    )
    if experiment.get("run_id") != run_id:
        raise CollectionError(f"{shard_name}/{policy} report run_id mismatch")
    report_endpoint = _mapping(
        experiment.get("compute_endpoint"),
        f"{shard_name}/{policy} report compute endpoint",
    )
    report_reserve = _mapping(
        experiment.get("reserve_case"),
        f"{shard_name}/{policy} report reserve case",
    )
    if dict(report_endpoint) != dict(endpoint_configuration):
        raise CollectionError(f"{shard_name}/{policy} report endpoint mismatch")
    if dict(report_reserve) != dict(reserve_configuration):
        raise CollectionError(f"{shard_name}/{policy} report reserve mismatch")
    _validate_declared_hash(
        endpoint_metadata_sha256,
        experiment.get("prompt_compute_calibration_metadata_sha256"),
        f"{shard_name}/{policy} report calibration metadata sha256",
    )
    if experiment.get("paired_infinite_hbm_oracle") is not True:
        raise CollectionError(f"{shard_name}/{policy} report lacks paired oracle")

    report_policy = _mapping(
        report.get("policy"), f"{shard_name}/{policy} report policy"
    )
    if report_policy.get("name") != policy:
        raise CollectionError(f"{shard_name}/{policy} report policy mismatch")
    if report_policy.get("demotion_mode") != "capacity-only":
        raise CollectionError(
            f"{shard_name}/{policy} report demotion mode mismatch"
        )
    resolved_config = _mapping(
        record.get("resolved_config"),
        f"{shard_name}/{policy} resolved config",
    )
    replay_config = _mapping(
        report.get("replay_config"),
        f"{shard_name}/{policy} report replay config",
    )
    if dict(resolved_config) != dict(replay_config):
        raise CollectionError(
            f"{shard_name}/{policy} resolved and report replay configs differ"
        )
    if replay_config.get("policy") != policy:
        raise CollectionError(f"{shard_name}/{policy} replay policy mismatch")
    reserve_to_config = {
        "common_bytes_per_rank": "hbm_static_reserve_bytes_per_rank",
        "prefill_bytes_per_rank": (
            "prefill_hbm_static_reserve_bytes_per_rank"
        ),
        "decode_bytes_per_rank": "decode_hbm_static_reserve_bytes_per_rank",
    }
    for reserve_field, config_field in reserve_to_config.items():
        if reserve_configuration.get(reserve_field) != replay_config.get(
            config_field
        ):
            raise CollectionError(
                f"{shard_name}/{policy} reserve and replay config differ "
                f"for {config_field}"
            )

    execution_scope = _mapping(
        report.get("execution_scope"),
        f"{shard_name}/{policy} report execution scope",
    )
    embedded_metadata = _mapping(
        execution_scope.get("prompt_compute_calibration"),
        f"{shard_name}/{policy} embedded prompt calibration",
    )
    if (
        _json_sha256(embedded_metadata) != endpoint_metadata_sha256
        or dict(embedded_metadata) != dict(endpoint_metadata)
    ):
        raise CollectionError(
            f"{shard_name}/{policy} embedded prompt calibration mismatch"
        )
    oracle = _mapping(
        report.get("infinite_hbm_oracle_comparison"),
        f"{shard_name}/{policy} paired oracle comparison",
    )
    if (
        oracle.get("reference") != "paired_infinite_hbm_residency"
        or oracle.get("same_prompt_compute_model") is not True
        or oracle.get("same_roofline_compute_model") is not True
    ):
        raise CollectionError(
            f"{shard_name}/{policy} paired oracle contract mismatch"
        )
    required_oracle_equalities = (
        "same_workload_and_first_call_arrivals",
        "same_gap_durations",
        "same_pd_topology_and_mandatory_transfers",
        "same_restore_execution_mode",
        "same_independent_pd_branch_admission",
        "same_final_decode_footprint_prereservation",
        "closed_loop_delay_conservation_checked",
    )
    missing_or_false = [
        field
        for field in required_oracle_equalities
        if oracle.get(field) is not True
    ]
    if missing_or_false:
        raise CollectionError(
            f"{shard_name}/{policy} paired oracle equality checks failed: "
            + ", ".join(missing_or_false)
        )
    oracle_validation = _mapping(
        oracle.get("oracle_validation"),
        f"{shard_name}/{policy} oracle validation",
    )
    if (
        oracle_validation.get("capacity_action_count") != 0
        or oracle_validation.get("aggregate_hbm_capacity_block_seconds")
        != 0.0
        or oracle_validation.get("capacity_invariant_checked") is not True
    ):
        raise CollectionError(
            f"{shard_name}/{policy} infinite-HBM oracle was not resident"
        )
    if report.get("hardware") != experiment_contract.get("hardware"):
        raise CollectionError(f"{shard_name}/{policy} report hardware mismatch")
    if report.get("model") != experiment_contract.get("model"):
        raise CollectionError(f"{shard_name}/{policy} report model mismatch")
    if report.get("tp_size") != experiment_contract.get("tp_size_per_role"):
        raise CollectionError(f"{shard_name}/{policy} report TP mismatch")
    return run_id, report_hash, resolved_config, report


def _validate_calibration(
    shard_name: str,
    shard_path: Path,
    manifest: Mapping[str, Any],
    endpoint: str,
) -> tuple[
    str,
    Mapping[str, str],
    Mapping[str, str],
    str,
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    artifact = _mapping(
        manifest.get("calibration_artifact"),
        f"{shard_name} calibration artifact",
    )
    calibration_path = _safe_child(
        shard_path,
        artifact.get("path"),
        f"{shard_name} calibration path",
    )
    calibration, raw = _read_json(
        calibration_path, f"{shard_name} calibration"
    )
    _validate_declared_hash(
        _sha256_bytes(raw),
        artifact.get("sha256"),
        f"{shard_name} calibration sha256",
    )
    if _integer(artifact.get("bytes"), f"{shard_name} calibration bytes") != len(
        raw
    ):
        raise CollectionError(f"{shard_name} calibration byte count mismatch")
    if _integer(
        calibration.get("schema_version"),
        f"{shard_name} calibration schema",
    ) != CALIBRATION_ARTIFACT_SCHEMA_VERSION:
        raise CollectionError(
            f"{shard_name} calibration schema must be "
            f"{CALIBRATION_ARTIFACT_SCHEMA_VERSION}"
        )
    if _integer(
        artifact.get("schema_version"),
        f"{shard_name} manifest calibration schema",
    ) != CALIBRATION_ARTIFACT_SCHEMA_VERSION:
        raise CollectionError(
            f"{shard_name} manifest calibration schema must be "
            f"{CALIBRATION_ARTIFACT_SCHEMA_VERSION}"
        )

    base_metadata = _mapping(
        calibration.get("base_calibration_metadata"),
        f"{shard_name} base calibration metadata",
    )
    base_hash = _json_sha256(base_metadata)
    for declared, label in (
        (
            calibration.get("base_calibration_metadata_sha256"),
            "calibration",
        ),
        (
            artifact.get("base_calibration_metadata_sha256"),
            "manifest",
        ),
    ):
        _validate_declared_hash(
            base_hash,
            declared,
            f"{shard_name} {label} base calibration metadata sha256",
        )
    normalized_sources = _sha256_mapping(
        base_metadata.get("source_sha256"),
        f"{shard_name} calibration source hashes",
    )
    normalized_producers = _sha256_mapping(
        base_metadata.get("producer_source_sha256"),
        f"{shard_name} calibration producer source hashes",
    )
    source_unit_contract = _mapping(
        calibration.get("source_unit_contract"),
        f"{shard_name} calibration source unit contract",
    )
    source_work_contract = _mapping(
        calibration.get("source_work_contract"),
        f"{shard_name} calibration source work contract",
    )
    prompt_calibration = _mapping(
        manifest.get("prompt_compute_calibration"),
        f"{shard_name} prompt compute calibration",
    )
    manifest_sources = _sha256_mapping(
        prompt_calibration.get("source_sha256"),
        f"{shard_name} manifest calibration source hashes",
    )
    if manifest_sources != normalized_sources:
        raise CollectionError(
            f"{shard_name} manifest and calibration source hashes differ"
        )
    manifest_producers = _sha256_mapping(
        prompt_calibration.get("producer_source_sha256"),
        f"{shard_name} manifest calibration producer source hashes",
    )
    if manifest_producers != normalized_producers:
        raise CollectionError(
            f"{shard_name} manifest and calibration producer source "
            "hashes differ"
        )

    calibration_endpoints = _mapping(
        calibration.get("endpoints"),
        f"{shard_name} calibration endpoints",
    )
    if set(calibration_endpoints) != {endpoint}:
        raise CollectionError(f"{shard_name} calibration endpoint set mismatch")
    endpoint_record = _mapping(
        calibration_endpoints[endpoint],
        f"{shard_name} calibration endpoint record",
    )
    endpoint_metadata = _mapping(
        endpoint_record.get("metadata"),
        f"{shard_name} calibration endpoint metadata",
    )
    endpoint_hash = _json_sha256(endpoint_metadata)
    _validate_declared_hash(
        endpoint_hash,
        endpoint_record.get("metadata_sha256"),
        f"{shard_name} calibration endpoint metadata sha256",
    )
    configuration = _mapping(
        endpoint_record.get("configuration"),
        f"{shard_name} calibration endpoint configuration",
    )
    if configuration.get("name") != endpoint:
        raise CollectionError(
            f"{shard_name} calibration endpoint configuration mismatch"
        )
    if (
        endpoint_metadata.get("band") != configuration.get("band")
        or endpoint_metadata.get("attention_multiplier")
        != configuration.get("attention_multiplier")
    ):
        raise CollectionError(
            f"{shard_name} endpoint metadata and configuration differ"
        )
    if _sha256_mapping(
        endpoint_metadata.get("source_sha256"),
        f"{shard_name} endpoint calibration source hashes",
    ) != normalized_sources:
        raise CollectionError(
            f"{shard_name} endpoint and base calibration sources differ"
        )
    if _sha256_mapping(
        endpoint_metadata.get("producer_source_sha256"),
        f"{shard_name} endpoint producer source hashes",
    ) != normalized_producers:
        raise CollectionError(
            f"{shard_name} endpoint and base producer sources differ"
        )
    model_geometry = _mapping(
        manifest.get("model_geometry"), f"{shard_name} model geometry"
    )
    target_geometry = _mapping(
        endpoint_metadata.get("target_geometry"),
        f"{shard_name} endpoint target geometry",
    )
    _validate_declared_hash(
        _sha256_string(
            model_geometry.get("local_simulator_config_sha256"),
            f"{shard_name} model config sha256",
        ),
        target_geometry.get("config_sha256"),
        f"{shard_name} endpoint target config sha256",
    )
    expected_endpoint_hashes = {endpoint: endpoint_hash}
    artifact_endpoint_hashes = _mapping(
        artifact.get("endpoint_metadata_sha256"),
        f"{shard_name} manifest calibration endpoint hashes",
    )
    prompt_endpoint_hashes = _mapping(
        prompt_calibration.get("endpoint_metadata_sha256"),
        f"{shard_name} prompt calibration endpoint hashes",
    )
    prompt_endpoint_metadata = _mapping(
        prompt_calibration.get("endpoint_metadata"),
        f"{shard_name} prompt calibration endpoint metadata",
    )
    if (
        dict(artifact_endpoint_hashes) != expected_endpoint_hashes
        or dict(prompt_endpoint_hashes) != expected_endpoint_hashes
        or set(prompt_endpoint_metadata) != {endpoint}
        or _json_sha256(
            _mapping(
                prompt_endpoint_metadata[endpoint],
                f"{shard_name} prompt endpoint metadata",
            )
        )
        != endpoint_hash
    ):
        raise CollectionError(
            f"{shard_name} endpoint calibration hashes are inconsistent"
        )
    return (
        base_hash,
        normalized_sources,
        normalized_producers,
        endpoint_hash,
        configuration,
        endpoint_metadata,
        {
            "source_unit_contract": dict(source_unit_contract),
            "source_work_contract": dict(source_work_contract),
        },
    )


def _load_shard(
    parts_root: Path,
    reserve_label: str,
    reserve_case: str,
    endpoint: str,
) -> Shard:
    shard_name = f"{reserve_label}-{endpoint}"
    shard_path = parts_root / shard_name
    manifest_path = shard_path / "manifest.json"
    manifest, manifest_raw = _read_json(manifest_path, f"{shard_name} manifest")
    if _integer(
        manifest.get("schema_version"), f"{shard_name} manifest schema"
    ) != SHARD_MANIFEST_SCHEMA_VERSION:
        raise CollectionError(
            f"{shard_name} manifest schema must be "
            f"{SHARD_MANIFEST_SCHEMA_VERSION}"
        )

    endpoints = _list(
        manifest.get("compute_endpoints"), f"{shard_name} compute endpoints"
    )
    if len(endpoints) != 1:
        raise CollectionError(f"{shard_name} manifest endpoint mismatch")
    manifest_endpoint = _mapping(
        endpoints[0], f"{shard_name} compute endpoint"
    )
    if manifest_endpoint.get("name") != endpoint:
        raise CollectionError(f"{shard_name} manifest endpoint mismatch")
    reserve_cases = _mapping(
        manifest.get("reserve_derivation"),
        f"{shard_name} reserve derivation",
    ).get("cases")
    reserve_cases = _list(reserve_cases, f"{shard_name} reserve cases")
    if len(reserve_cases) != 1:
        raise CollectionError(f"{shard_name} manifest reserve mismatch")
    manifest_reserve = _mapping(
        reserve_cases[0], f"{shard_name} reserve case"
    )
    if manifest_reserve.get("name") != reserve_case:
        raise CollectionError(f"{shard_name} manifest reserve mismatch")
    experiment_contract = _mapping(
        manifest.get("experiment_contract"),
        f"{shard_name} experiment contract",
    )

    declared_tables = _mapping(
        manifest.get("tables"), f"{shard_name} manifest tables"
    )
    if set(declared_tables) != set(TABLE_NAMES):
        raise CollectionError(
            f"{shard_name} manifest tables must be exactly "
            + ", ".join(TABLE_NAMES)
        )
    tables: dict[str, CsvTable] = {}
    for table_name in TABLE_NAMES:
        table = _read_csv(
            shard_path / table_name, f"{shard_name}/{table_name}"
        )
        declaration = _mapping(
            declared_tables[table_name],
            f"{shard_name} manifest table {table_name}",
        )
        _validate_declared_hash(
            table.sha256,
            declaration.get("sha256"),
            f"{shard_name}/{table_name} manifest sha256",
        )
        if _integer(
            declaration.get("rows"),
            f"{shard_name}/{table_name} manifest rows",
        ) != len(table.rows):
            raise CollectionError(
                f"{shard_name}/{table_name} manifest row count mismatch"
            )
        tables[table_name] = table

    (
        base_hash,
        source_hashes,
        producer_source_hashes,
        endpoint_metadata_hash,
        calibration_endpoint_configuration,
        endpoint_metadata,
        calibration_work_contract,
    ) = _validate_calibration(shard_name, shard_path, manifest, endpoint)
    if dict(calibration_endpoint_configuration) != dict(manifest_endpoint):
        raise CollectionError(
            f"{shard_name} calibration and manifest endpoint differ"
        )

    runs = _list(manifest.get("runs"), f"{shard_name} runs")
    if len(runs) != len(POLICIES):
        raise CollectionError(
            f"{shard_name} must contain exactly {len(POLICIES)} runs"
        )
    record_by_policy: dict[str, Mapping[str, Any]] = {}
    run_id_by_policy: dict[str, str] = {}
    report_hash_by_policy: dict[str, str] = {}
    resolved_config_by_policy: dict[str, Mapping[str, Any]] = {}
    report_by_policy: dict[str, Mapping[str, Any]] = {}
    for raw_record in runs:
        record = _mapping(raw_record, f"{shard_name} run record")
        policy = _string(record.get("policy"), f"{shard_name} run policy")
        if policy not in POLICIES:
            raise CollectionError(
                f"{shard_name} has unexpected run policy {policy!r}"
            )
        if policy in record_by_policy:
            raise CollectionError(
                f"{shard_name} repeats run policy {policy!r}"
            )
        run_id, report_hash, resolved_config, report = _validate_report(
            shard_name=shard_name,
            shard_path=shard_path,
            record=record,
            expected_policy=policy,
            reserve_case=reserve_case,
            endpoint=endpoint,
            endpoint_configuration=manifest_endpoint,
            reserve_configuration=manifest_reserve,
            endpoint_metadata=endpoint_metadata,
            endpoint_metadata_sha256=endpoint_metadata_hash,
            experiment_contract=experiment_contract,
        )
        if run_id in run_id_by_policy.values():
            raise CollectionError(f"{shard_name} has duplicate run_id {run_id}")
        run_id_by_policy[policy] = run_id
        report_hash_by_policy[policy] = report_hash
        resolved_config_by_policy[policy] = resolved_config
        report_by_policy[policy] = report
        record_by_policy[policy] = record
    if set(record_by_policy) != set(POLICIES):
        raise CollectionError(f"{shard_name} run policy set mismatch")
    expected_json_names = {
        "manifest.json",
        "calibration.json",
        *(str(record["report"]) for record in record_by_policy.values()),
    }
    actual_json_names = {
        path.name for path in shard_path.iterdir() if path.suffix == ".json"
    }
    if actual_json_names != expected_json_names:
        raise CollectionError(
            f"{shard_name} JSON artifact set mismatch; expected "
            f"{sorted(expected_json_names)}, found {sorted(actual_json_names)}"
        )

    for table in tables.values():
        _validate_table_scope(
            table,
            shard_name=shard_name,
            reserve_case=reserve_case,
            endpoint=endpoint,
            run_id_by_policy=run_id_by_policy,
        )
    try:
        endpoint_object = ComputeEndpoint(**dict(manifest_endpoint))
        reserve_object = ReserveCase(**dict(manifest_reserve))
    except TypeError as exc:
        raise CollectionError(
            f"{shard_name} endpoint or reserve dataclass contract mismatch: "
            f"{exc}"
        ) from exc
    expected_summary_rows: list[Mapping[str, Any]] = []
    expected_return_rows: list[Mapping[str, Any]] = []
    expected_transfer_rows: list[Mapping[str, Any]] = []
    for policy in POLICIES:
        report = report_by_policy[policy]
        run_id = run_id_by_policy[policy]
        expected_summary_rows.append(
            build_summary_row(
                report,
                run_id,
                endpoint_object,
                reserve_object,
                policy,
                report_hash_by_policy[policy],
            )
        )
        expected_return_rows.extend(
            build_return_source_rows(
                report,
                run_id,
                endpoint,
                reserve_case,
                policy,
            )
        )
        expected_transfer_rows.extend(
            build_transfer_stage_rows(
                report,
                run_id,
                endpoint,
                reserve_case,
                policy,
            )
        )
    _validate_derived_table(
        tables["summary.csv"],
        expected_summary_rows,
        f"{shard_name}/summary.csv",
    )
    _validate_derived_table(
        tables["return_sources.csv"],
        expected_return_rows,
        f"{shard_name}/return_sources.csv",
    )
    _validate_derived_table(
        tables["transfer_stages.csv"],
        expected_transfer_rows,
        f"{shard_name}/transfer_stages.csv",
    )
    summary = tables["summary.csv"]
    if len(summary.rows) != len(POLICIES):
        raise CollectionError(f"{shard_name}/summary.csv must have 3 rows")
    required_summary = {
        "report_sha256",
        "report_schema_version",
        "prompt_compute_calibration_metadata_sha256",
    }
    missing_summary = sorted(required_summary - set(summary.header))
    if missing_summary:
        raise CollectionError(
            f"{shard_name}/summary.csv lacks columns: "
            + ", ".join(missing_summary)
        )
    summary_by_policy: dict[str, Mapping[str, str]] = {}
    for row in summary.rows:
        policy = row["baseline"]
        if policy in summary_by_policy:
            raise CollectionError(
                f"{shard_name}/summary.csv repeats policy {policy}"
            )
        summary_by_policy[policy] = row
    if set(summary_by_policy) != set(POLICIES):
        raise CollectionError(f"{shard_name}/summary.csv policy set mismatch")
    for policy in POLICIES:
        row = summary_by_policy[policy]
        if row["report_schema_version"] != str(REPORT_SCHEMA_VERSION):
            raise CollectionError(
                f"{shard_name}/{policy} summary report schema mismatch"
            )
        _validate_declared_hash(
            report_hash_by_policy[policy],
            row["report_sha256"],
            f"{shard_name}/{policy} summary report sha256",
        )
        _validate_declared_hash(
            endpoint_metadata_hash,
            row["prompt_compute_calibration_metadata_sha256"],
            f"{shard_name}/{policy} summary calibration metadata sha256",
        )

    workload = _mapping(manifest.get("workload"), f"{shard_name} workload")
    workload_hash = _sha256_string(
        workload.get("sha256"), f"{shard_name} workload sha256"
    )
    workload_provenance = _mapping(
        manifest.get("workload_provenance"),
        f"{shard_name} workload provenance",
    )
    local_source_hashes = _sha256_mapping(
        manifest.get("local_source_sha256"),
        f"{shard_name} local source hashes",
    )
    for source_path, source_hash in source_hashes.items():
        if local_source_hashes.get(source_path) != source_hash:
            raise CollectionError(
                f"{shard_name} calibration source hashes are not an exact "
                "matching subset of local_source_sha256"
            )
    for source_path, source_hash in producer_source_hashes.items():
        if local_source_hashes.get(source_path) != source_hash:
            raise CollectionError(
                f"{shard_name} calibration producer source hashes are not "
                "an exact matching subset of local_source_sha256"
            )
    git_provenance = _mapping(
        manifest.get("git"), f"{shard_name} git provenance"
    )
    if git_provenance.get("available") is not True:
        raise CollectionError(f"{shard_name} git provenance must be available")
    git_commit = _string(
        git_provenance.get("commit"), f"{shard_name} git commit"
    )
    git_dirty = git_provenance.get("dirty")
    if not isinstance(git_dirty, bool):
        raise CollectionError(f"{shard_name} git dirty must be a boolean")
    status_lines = _list(
        git_provenance.get("status_porcelain"),
        f"{shard_name} git status porcelain",
    )
    if any(not isinstance(line, str) or not line for line in status_lines):
        raise CollectionError(
            f"{shard_name} git status porcelain must contain nonempty strings"
        )
    if git_dirty != bool(status_lines):
        raise CollectionError(
            f"{shard_name} git dirty does not match status porcelain"
        )
    reconstructed_status_hash = _sha256_bytes(
        "\n".join(status_lines).encode("utf-8")
    )
    _validate_declared_hash(
        reconstructed_status_hash,
        git_provenance.get("status_sha256"),
        f"{shard_name} git status sha256",
    )
    return Shard(
        name=shard_name,
        reserve_label=reserve_label,
        reserve_case=reserve_case,
        endpoint=endpoint,
        path=shard_path,
        manifest=manifest,
        manifest_sha256=_sha256_bytes(manifest_raw),
        manifest_bytes=len(manifest_raw),
        tables=tables,
        report_records=tuple(record_by_policy[policy] for policy in POLICIES),
        workload=workload,
        workload_sha256=workload_hash,
        workload_provenance=workload_provenance,
        base_calibration_metadata_sha256=base_hash,
        source_sha256=source_hashes,
        producer_source_sha256=producer_source_hashes,
        endpoint_metadata_sha256=endpoint_metadata_hash,
        endpoint_configuration=manifest_endpoint,
        calibration_work_contract=calibration_work_contract,
        resolved_config_by_policy=resolved_config_by_policy,
        local_source_sha256=local_source_hashes,
        git_provenance=git_provenance,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )


def _validate_layout(parts_root: Path) -> None:
    if parts_root.is_symlink() or not parts_root.is_dir():
        raise CollectionError(f"parts directory does not exist: {parts_root}")
    expected = {
        f"{reserve}-{endpoint}"
        for reserve, _ in RESERVE_CASES
        for endpoint in COMPUTE_ENDPOINTS
    }
    actual = {path.name for path in parts_root.iterdir() if path.is_dir()}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(extra))
        raise CollectionError("invalid shard directory set; " + "; ".join(details))
    symlinks = sorted(
        name for name in expected if (parts_root / name).is_symlink()
    )
    if symlinks:
        raise CollectionError(
            "shard directories cannot be symbolic links: "
            + ", ".join(symlinks)
        )


def _validate_cross_shard_bindings(shards: Sequence[Shard]) -> dict[str, Any]:
    if not shards:
        raise CollectionError("no shards were loaded")
    reference = shards[0]
    reference_workload_hash = _json_sha256(reference.workload)
    reference_provenance_hash = _json_sha256(reference.workload_provenance)
    endpoint_reference: dict[str, Shard] = {}
    reserve_reference: dict[str, Shard] = {}
    normalized_config_reference: Mapping[str, Any] | None = None

    def normalized_config(config: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(config)
        for field in RESOLVED_CONFIG_VARIANT_FIELDS:
            result.pop(field, None)
        return result

    for shard in shards:
        previous_endpoint = endpoint_reference.setdefault(
            shard.endpoint, shard
        )
        if (
            shard.endpoint_metadata_sha256
            != previous_endpoint.endpoint_metadata_sha256
            or dict(shard.endpoint_configuration)
            != dict(previous_endpoint.endpoint_configuration)
        ):
            raise CollectionError(
                f"{shard.name} endpoint calibration differs from "
                f"{previous_endpoint.name}"
            )
        previous_reserve = reserve_reference.setdefault(
            shard.reserve_case, shard
        )
        current_reserve = _mapping(
            _list(
                _mapping(
                    shard.manifest.get("reserve_derivation"),
                    f"{shard.name} reserve derivation",
                ).get("cases"),
                f"{shard.name} reserve cases",
            )[0],
            f"{shard.name} reserve case",
        )
        previous_reserve_value = _mapping(
            _list(
                _mapping(
                    previous_reserve.manifest.get("reserve_derivation"),
                    f"{previous_reserve.name} reserve derivation",
                ).get("cases"),
                f"{previous_reserve.name} reserve cases",
            )[0],
            f"{previous_reserve.name} reserve case",
        )
        if dict(current_reserve) != dict(previous_reserve_value):
            raise CollectionError(
                f"{shard.name} reserve configuration differs from "
                f"{previous_reserve.name}"
            )
        for policy in POLICIES:
            current_config = normalized_config(
                shard.resolved_config_by_policy[policy]
            )
            if normalized_config_reference is None:
                normalized_config_reference = current_config
            elif current_config != normalized_config_reference:
                raise CollectionError(
                    f"{shard.name}/{policy} invariant resolved config differs"
                )

    for shard in shards[1:]:
        if shard.workload != reference.workload:
            raise CollectionError(
                f"{shard.name} full workload metadata differs from "
                f"{reference.name}"
            )
        if shard.workload_sha256 != reference.workload_sha256:
            raise CollectionError(
                f"{shard.name} workload hash differs from {reference.name}"
            )
        if shard.workload_provenance != reference.workload_provenance:
            raise CollectionError(
                f"{shard.name} workload provenance differs from "
                f"{reference.name}"
            )
        if (
            shard.base_calibration_metadata_sha256
            != reference.base_calibration_metadata_sha256
        ):
            raise CollectionError(
                f"{shard.name} base calibration metadata hash differs from "
                f"{reference.name}"
            )
        if shard.source_sha256 != reference.source_sha256:
            raise CollectionError(
                f"{shard.name} calibration source hashes differ from "
                f"{reference.name}"
            )
        if (
            shard.producer_source_sha256
            != reference.producer_source_sha256
        ):
            raise CollectionError(
                f"{shard.name} calibration producer source hashes differ "
                f"from {reference.name}"
            )
        if shard.local_source_sha256 != reference.local_source_sha256:
            raise CollectionError(
                f"{shard.name} local source hashes differ from "
                f"{reference.name}"
            )
        if shard.git_commit != reference.git_commit:
            raise CollectionError(
                f"{shard.name} git commit differs from {reference.name}"
            )
        if (
            dict(shard.calibration_work_contract)
            != dict(reference.calibration_work_contract)
        ):
            raise CollectionError(
                f"{shard.name} calibration work contract differs from "
                f"{reference.name}"
            )
        for field in CROSS_SHARD_MANIFEST_FIELDS:
            current = _mapping(
                shard.manifest.get(field), f"{shard.name} {field}"
            )
            expected = _mapping(
                reference.manifest.get(field), f"{reference.name} {field}"
            )
            if dict(current) != dict(expected):
                raise CollectionError(
                    f"{shard.name} invariant manifest field {field} differs "
                    f"from {reference.name}"
                )
    return {
        "workload_sha256": reference.workload_sha256,
        "workload_metadata_sha256": reference_workload_hash,
        "workload_metadata_comparison": (
            "Exact equality of the complete workload object in every shard "
            "manifest, in addition to the workload content hash."
        ),
        "workload_provenance_sha256": reference_provenance_hash,
        "base_calibration_metadata_sha256": (
            reference.base_calibration_metadata_sha256
        ),
        "calibration_source_sha256": dict(
            sorted(reference.source_sha256.items())
        ),
        "calibration_producer_source_sha256": dict(
            sorted(reference.producer_source_sha256.items())
        ),
        "calibration_work_contract_sha256": _json_sha256(
            reference.calibration_work_contract
        ),
        "endpoint_calibration_metadata_sha256": {
            endpoint: endpoint_reference[endpoint].endpoint_metadata_sha256
            for endpoint in COMPUTE_ENDPOINTS
        },
        "endpoint_calibration_equality_scope": (
            "Exact metadata-hash and endpoint-configuration equality across "
            "zero, half, and full reserve shards for each endpoint."
        ),
        "invariant_manifest_field_sha256": {
            field: _json_sha256(reference.manifest[field])
            for field in CROSS_SHARD_MANIFEST_FIELDS
        },
        "normalized_common_resolved_config_sha256": _json_sha256(
            normalized_config_reference
        ),
        "resolved_config_normalization": {
            "removed_variant_fields": list(RESOLVED_CONFIG_VARIANT_FIELDS),
            "all_remaining_fields_equal_across_shards": True,
        },
        "local_source_sha256": dict(
            sorted(reference.local_source_sha256.items())
        ),
        "local_source_comparison": (
            "Exact equality across shards; calibration source hashes must "
            "also be an exact matching subset in every shard."
        ),
        "git": {
            "commit": reference.git_commit,
            "commit_equality_required": True,
            "dirty_by_shard": {
                shard.name: shard.git_dirty for shard in shards
            },
            "status_sha256_by_shard": {
                shard.name: shard.git_provenance.get("status_sha256")
                for shard in shards
            },
            "provenance_sha256_by_shard": {
                shard.name: _json_sha256(shard.git_provenance)
                for shard in shards
            },
            "dirty_equality_is_rejection_condition": False,
            "status_differences_are_rejection_condition": False,
        },
    }


def _ordered_rows(shards: Sequence[Shard], table_name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for shard in shards:
        by_policy: dict[str, list[dict[str, str]]] = {
            policy: [] for policy in POLICIES
        }
        for row in shard.tables[table_name].rows:
            by_policy[row["baseline"]].append(dict(row))
        for policy in POLICIES:
            policy_rows = by_policy[policy]
            if table_name == "return_sources.csv":
                source_order = {
                    source: index
                    for index, source in enumerate(RETURN_SOURCES)
                }
                policy_rows.sort(
                    key=lambda row: (
                        row["return_gap_type"],
                        source_order.get(row["source"], len(source_order)),
                        tuple(row[column] for column in sorted(row)),
                    )
                )
            elif table_name == "transfer_stages.csv":
                policy_rows.sort(
                    key=lambda row: (
                        row["stage"],
                        tuple(row[column] for column in sorted(row)),
                    )
                )
            else:
                policy_rows.sort(
                    key=lambda row: tuple(
                        row[column] for column in sorted(row)
                    )
                )
            rows.extend(policy_rows)
    return rows


def _write_csv(
    path: Path, header: Sequence[str], rows: Sequence[Mapping[str, str]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(header))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ("git", *args),
            cwd=repo_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.rstrip("\n")

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain=v1", "--untracked-files=normal")
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "status_sha256": _sha256_bytes(status.encode("utf-8")),
    }


def _check_output_conflicts(root: Path, overwrite: bool) -> None:
    symlinks = [
        root / filename
        for filename in KNOWN_COLLECTOR_OUTPUTS
        if (root / filename).is_symlink()
    ]
    if symlinks:
        raise CollectionError(
            "collector output paths cannot be symbolic links: "
            + ", ".join(str(path) for path in symlinks)
        )
    invalid = [
        root / filename
        for filename in KNOWN_COLLECTOR_OUTPUTS
        if (root / filename).exists() and not (root / filename).is_file()
    ]
    if invalid:
        raise CollectionError(
            "collector output paths must be regular files: "
            + ", ".join(str(path) for path in invalid)
        )
    conflicts = [
        root / filename
        for filename in KNOWN_COLLECTOR_OUTPUTS
        if (root / filename).exists()
    ]
    if conflicts and not overwrite:
        raise CollectionError(
            "collector outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in conflicts)
        )


def _publish_staged_outputs(root: Path, staging: Path) -> None:
    """Publish tables first and the hash-binding manifest last.

    Existing regular files are hard-linked into the same-filesystem staging
    directory before replacement. Any Python-visible publication failure
    restores the previous complete generation. Publishing the manifest last
    also makes an interrupted multi-file publication detectable by hashes.
    """

    backup = staging / "previous"
    backup.mkdir()
    for filename in KNOWN_COLLECTOR_OUTPUTS:
        target = root / filename
        if target.exists():
            os.link(target, backup / filename)

    installed: list[str] = []
    try:
        for filename in (*TABLE_NAMES, ROOT_MANIFEST_NAME):
            os.replace(staging / filename, root / filename)
            installed.append(filename)
    except OSError:
        for filename in reversed(installed):
            target = root / filename
            previous = backup / filename
            if previous.exists():
                os.replace(previous, target)
            elif target.exists():
                os.replace(target, staging / f"failed-{filename}")
        raise


def collect(root: Path, *, overwrite: bool, command: str) -> Mapping[str, Any]:
    """Validate all shards and write deterministic root tables and index."""

    root = root.resolve()
    if not root.is_dir():
        raise CollectionError(f"root directory does not exist: {root}")
    _check_output_conflicts(root, overwrite)
    parts_root = root / "parts"
    _validate_layout(parts_root)

    shards = [
        _load_shard(parts_root, reserve_label, reserve_case, endpoint)
        for reserve_label, reserve_case in RESERVE_CASES
        for endpoint in COMPUTE_ENDPOINTS
    ]
    bindings = _validate_cross_shard_bindings(shards)

    headers: dict[str, tuple[str, ...]] = {}
    combined_rows: dict[str, list[dict[str, str]]] = {}
    for table_name in TABLE_NAMES:
        headers[table_name] = shards[0].tables[table_name].header
        for shard in shards[1:]:
            if shard.tables[table_name].header != headers[table_name]:
                raise CollectionError(
                    f"{shard.name}/{table_name} header differs from "
                    f"{shards[0].name}"
                )
        combined_rows[table_name] = _ordered_rows(shards, table_name)
    if len(combined_rows["summary.csv"]) != 36:
        raise CollectionError("combined summary.csv must contain exactly 36 rows")

    repo_root = Path(__file__).resolve().parents[1]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "qwen3_1m_calibrated_shard_collection_index",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "working_directory": str(Path.cwd().resolve()),
        "root": str(root),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "git": _git_provenance(repo_root),
        "collector_source": {
            "path": str(Path(__file__).resolve().relative_to(repo_root)),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "authority": {
            "statement": (
                "Each shard manifest is authoritative for its reports, "
                "calibration artifact, and shard-local tables. This root "
                "index only verifies and concatenates those records."
            ),
            "root_index_reinterprets_experiment_values": False,
        },
        "ordering": {
            "reserves": [item[0] for item in RESERVE_CASES],
            "compute_endpoints": list(COMPUTE_ENDPOINTS),
            "policies": list(POLICIES),
        },
        "bindings": bindings,
        "shards": [
            {
                "name": shard.name,
                "path": f"parts/{shard.name}",
                "manifest": f"parts/{shard.name}/manifest.json",
                "manifest_sha256": shard.manifest_sha256,
                "manifest_bytes": shard.manifest_bytes,
                "manifest_schema_version": shard.manifest.get(
                    "schema_version"
                ),
                "report_count": len(shard.report_records),
                "summary_rows": len(shard.tables["summary.csv"].rows),
            }
            for shard in shards
        ],
        "combined_tables": {},
    }
    with tempfile.TemporaryDirectory(
        prefix=".qwen3-1m-collect-", dir=root
    ) as staging_name:
        staging = Path(staging_name)
        for table_name in TABLE_NAMES:
            _write_csv(
                staging / table_name,
                headers[table_name],
                combined_rows[table_name],
            )
        manifest["combined_tables"] = {
            table_name: {
                "path": table_name,
                "sha256": _sha256_file(staging / table_name),
                "bytes": (staging / table_name).stat().st_size,
                "rows": len(combined_rows[table_name]),
                "columns": list(headers[table_name]),
            }
            for table_name in TABLE_NAMES
        }
        _write_json(staging / ROOT_MANIFEST_NAME, manifest)
        _publish_staged_outputs(root, staging)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and collect the twelve Qwen3 1M calibrated replay "
            "shards."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only known collector outputs at the root.",
    )
    return parser


def _resolved_command(argv: Sequence[str] | None) -> str:
    supplied = list(sys.argv[1:] if argv is None else argv)
    return shlex.join(
        [sys.executable, "-m", "serving.agentic_kv_qwen3_1m_collect", *supplied]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        manifest = collect(
            args.root,
            overwrite=args.overwrite,
            command=_resolved_command(argv),
        )
    except (CollectionError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Wrote {args.root.resolve() / ROOT_MANIFEST_NAME} "
        f"for {len(manifest['shards'])} shards",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

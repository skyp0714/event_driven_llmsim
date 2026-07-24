import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from serving.agentic_kv_qwen3_1m_collect import (
    COMPUTE_ENDPOINTS,
    POLICIES,
    RESERVE_CASES,
    CollectionError,
    collect,
)
from serving.agentic_kv_qwen3_1m_p4d4 import (
    ComputeEndpoint,
    ReserveCase,
    build_return_source_rows,
    build_summary_row,
    build_transfer_stage_rows,
)


SOURCE_HASHES = {
    "legacy/llama-layers.csv": "a" * 64,
    "legacy/llama-attention.csv": "b" * 64,
    "legacy/mixtral-layers.csv": "c" * 64,
    "legacy/mixtral-attention.csv": "d" * 64,
}
PRODUCER_SOURCE_HASHES = {
    "profiler/v0/profiler/attention/main.py": "3" * 64,
    "profiler/v0/profiler/layers/main.py": "4" * 64,
}
LOCAL_SOURCE_HASHES = {
    **SOURCE_HASHES,
    **PRODUCER_SOURCE_HASHES,
    "serving/agentic_kv_qwen3_1m_p4d4.py": "1" * 64,
    "serving/core/h100_kernel_calibrated_prompt.py": "2" * 64,
}
WORKLOAD_PROVENANCE = {
    "status": "loaded",
    "sidecar": {"sha256": "e" * 64},
    "converter": {"schema_version": 3, "commit": "converter-commit"},
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _table_record(path, rows):
    return {"sha256": _sha256(path), "rows": len(rows)}


def _endpoint_configuration(endpoint):
    return {
        "name": endpoint,
        "band": (
            "fast"
            if endpoint.startswith("fast_")
            else "slow"
            if endpoint.startswith("slow_")
            else "central"
        ),
        "attention_multiplier": (
            1.0 / 3.0 if endpoint == "central_attention_one_third" else 1.0
        ),
        "provenance": f"synthetic endpoint {endpoint}",
    }


def _resolved_config(reserve_label, endpoint, policy):
    return {
        "policy": policy,
        "hbm_capacity_bytes_per_rank": 80_000_000_000,
        "hbm_static_reserve_bytes_per_rank": {
            "zero": 0,
            "half": 1,
            "full": 2,
        }[reserve_label],
        "prefill_hbm_static_reserve_bytes_per_rank": None,
        "decode_hbm_static_reserve_bytes_per_rank": None,
        "cpu_capacity_bytes": 2_000_000_000_000,
        "pd_disaggregated": True,
        "prompt_compute_scale_provenance": f"synthetic {endpoint}",
    }


def _make_shard(
    root,
    reserve_label,
    reserve_case,
    endpoint,
    *,
    reverse_rows=False,
    endpoint_metadata_salt="",
    invariant_config_delta=0,
    experiment_contract_delta=0,
):
    shard = root / "parts" / f"{reserve_label}-{endpoint}"
    shard.mkdir(parents=True)
    endpoint_configuration = _endpoint_configuration(endpoint)
    reserve_configuration = {
        "name": reserve_case,
        "common_bytes_per_rank": {
            "zero": 0,
            "half": 1,
            "full": 2,
        }[reserve_label],
        "prefill_bytes_per_rank": None,
        "decode_bytes_per_rank": None,
    }
    base_metadata = {
        "source_sha256": SOURCE_HASHES,
        "producer_source_sha256": PRODUCER_SOURCE_HASHES,
        "validation": {"holdout": {"rows": 10}},
    }
    base_hash = _json_sha256(base_metadata)
    endpoint_metadata = {
        "schema_version": 3,
        "model_kind": "synthetic_test_prompt_model",
        "band": endpoint.split("_", 1)[0],
        "attention_multiplier": endpoint_configuration[
            "attention_multiplier"
        ],
        "source_sha256": SOURCE_HASHES,
        "producer_source_sha256": PRODUCER_SOURCE_HASHES,
        "target_geometry": {"config_sha256": "5" * 64},
        "synthetic_salt": endpoint_metadata_salt,
    }
    endpoint_hash = _json_sha256(endpoint_metadata)
    calibration = {
        "schema_version": 3,
        "base_calibration_metadata_sha256": base_hash,
        "base_calibration_metadata": base_metadata,
        "source_unit_contract": {"layers.csv": "nanoseconds"},
        "source_work_contract": {"causal": "bottom-right"},
        "endpoints": {
            endpoint: {
                "configuration": endpoint_configuration,
                "metadata_sha256": endpoint_hash,
                "metadata": endpoint_metadata,
            }
        },
    }
    calibration_path = shard / "calibration.json"
    _write_json(calibration_path, calibration)

    runs = []
    summary_rows = []
    return_rows = []
    transfer_rows = []
    for policy_index, policy in enumerate(POLICIES):
        run_id = f"{reserve_case}__{endpoint}__{policy}"
        report_name = f"{run_id}.json"
        report_path = shard / report_name
        resolved_config = _resolved_config(reserve_label, endpoint, policy)
        resolved_config["cpu_capacity_bytes"] += invariant_config_delta
        report = {
            "schema_version": 15,
            "hardware": "H100",
            "model": "synthetic-qwen",
            "tp_size": 4,
            "policy": {
                "name": policy,
                "demotion_mode": "capacity-only",
            },
            "replay_config": resolved_config,
            "execution_scope": {
                "prompt_compute_calibration": endpoint_metadata,
                "prompt_compute_scale": 1.0,
            },
            "workload": {"max_context_tokens": 1_010_000},
            "capacity": {
                "hbm_total_bytes_per_rank": 80_000_000_000,
                "hbm_static_reserve_bytes_per_rank": (
                    reserve_configuration["common_bytes_per_rank"]
                ),
            },
            "resume": {
                "all_request_count": 10,
                "reuse_eligible_transition_count": 6,
                "source_counts": {
                    "decode_hbm": 2,
                    "cpu": 1,
                    "ssd": 2,
                    "recompute": 1,
                },
                "source_fractions_of_all_requests": {
                    "decode_hbm": 0.2,
                    "cpu": 0.1,
                    "ssd": 0.2,
                    "recompute": 0.1,
                },
                "cpu_or_ssd_resume_fraction_of_all_requests": 0.3,
                "restore_timing": {},
                "by_return_gap_type": {
                    "tool": {
                        "all_request_count": 5,
                        "reuse_eligible_transition_count": 3,
                        "source_counts": {
                            "decode_hbm": 1,
                            "cpu": 1,
                            "ssd": 1,
                            "recompute": 0,
                        },
                        "source_reusable_tokens": {
                            "decode_hbm": 10,
                            "cpu": 20,
                            "ssd": 30,
                            "recompute": 0,
                        },
                        "source_fractions_of_all_requests_in_return_class": {
                            "decode_hbm": 0.2,
                            "cpu": 0.2,
                            "ssd": 0.2,
                            "recompute": 0.0,
                        },
                        "source_fractions_of_reuse_eligible_in_return_class": {
                            "decode_hbm": 1 / 3,
                            "cpu": 1 / 3,
                            "ssd": 1 / 3,
                            "recompute": 0.0,
                        },
                    }
                },
            },
            "recompute": {
                "analytical_time_fraction_of_executed_prompt_compute": 0.1,
            },
            "transfer_queue": {
                "jobs_by_kind": {"cpu_to_decode": 1},
                "bytes_by_kind": {"cpu_to_decode": 2},
                "queue_wait_seconds_by_kind": {"cpu_to_decode": 3.0},
                "service_seconds_by_kind": {"cpu_to_decode": 4.0},
            },
            "experiment": {
                "run_id": run_id,
                "compute_endpoint": endpoint_configuration,
                "reserve_case": reserve_configuration,
                "prompt_compute_calibration_metadata_sha256": endpoint_hash,
                "paired_infinite_hbm_oracle": True,
            },
            "infinite_hbm_oracle_comparison": {
                "reference": "paired_infinite_hbm_residency",
                "same_prompt_compute_model": True,
                "same_roofline_compute_model": True,
                "same_workload_and_first_call_arrivals": True,
                "same_gap_durations": True,
                "same_pd_topology_and_mandatory_transfers": True,
                "same_restore_execution_mode": True,
                "same_independent_pd_branch_admission": True,
                "same_final_decode_footprint_prereservation": True,
                "closed_loop_delay_conservation_checked": True,
                "oracle_validation": {
                    "capacity_action_count": 0,
                    "aggregate_hbm_capacity_block_seconds": 0.0,
                    "capacity_invariant_checked": True,
                },
            },
            "synthetic_value": policy_index,
        }
        _write_json(report_path, report)
        report_hash = _sha256(report_path)
        runs.append(
            {
                "run_id": run_id,
                "report": report_name,
                "report_sha256": report_hash,
                "report_schema_version": 15,
                "policy": policy,
                "compute_endpoint": endpoint,
                "calibration_band": endpoint_configuration["band"],
                "attention_multiplier": endpoint_configuration[
                    "attention_multiplier"
                ],
                "prompt_compute_calibration_metadata_sha256": endpoint_hash,
                "reserve_case": reserve_case,
                "resolved_config": resolved_config,
            }
        )
        endpoint_object = ComputeEndpoint(**endpoint_configuration)
        reserve_object = ReserveCase(**reserve_configuration)
        summary_rows.append(
            build_summary_row(
                report,
                run_id,
                endpoint_object,
                reserve_object,
                policy,
                report_hash,
            )
        )
        return_rows.extend(
            build_return_source_rows(
                report, run_id, endpoint, reserve_case, policy
            )
        )
        transfer_rows.extend(
            build_transfer_stage_rows(
                report, run_id, endpoint, reserve_case, policy
            )
        )
    if reverse_rows:
        runs.reverse()
        summary_rows.reverse()
        return_rows.reverse()
        transfer_rows.reverse()

    table_rows = {
        "summary.csv": summary_rows,
        "return_sources.csv": return_rows,
        "transfer_stages.csv": transfer_rows,
    }
    headers = {
        name: tuple(rows[0]) for name, rows in table_rows.items()
    }
    for name, rows in table_rows.items():
        _write_csv(shard / name, headers[name], rows)

    manifest = {
        "schema_version": 3,
        "workload": {
            "sha256": "f" * 64,
            "session_count": 10,
            "request_count": 30,
        },
        "workload_provenance": WORKLOAD_PROVENANCE,
        "local_source_sha256": LOCAL_SOURCE_HASHES,
        "git": {
            "available": True,
            "commit": "0123456789abcdef",
            "dirty": True,
            "status_porcelain": [f"?? synthetic/{endpoint}"],
            "status_sha256": hashlib.sha256(
                f"?? synthetic/{endpoint}".encode()
            ).hexdigest(),
        },
        "experiment_contract": {
            "model": "synthetic-qwen",
            "hardware": "H100",
            "tp_size_per_role": 4,
            "pd_layout": "synthetic-p4d4",
            "cpu_capacity_bytes": (
                2_000_000_000_000 + experiment_contract_delta
            ),
        },
        "model_geometry": {
            "hidden_size": 2_048,
            "local_simulator_config_sha256": "5" * 64,
        },
        "official_qwen_sources": {"revision": "synthetic"},
        "hardware_sources": {"h100": "synthetic"},
        "evidence_classes": {"class": "synthetic"},
        "repository_profile_evidence": {"profiles": "synthetic"},
        "excluded_repository_profile_evidence": {"excluded": "synthetic"},
        "calibration_boundary": {"absolute_measurement": False},
        "compute_endpoints": [endpoint_configuration],
        "reserve_derivation": {"cases": [reserve_configuration]},
        "calibration_artifact": {
            "path": "calibration.json",
            "sha256": _sha256(calibration_path),
            "bytes": calibration_path.stat().st_size,
            "schema_version": 3,
            "base_calibration_metadata_sha256": base_hash,
            "endpoint_metadata_sha256": {endpoint: endpoint_hash},
        },
        "prompt_compute_calibration": {
            "source_sha256": SOURCE_HASHES,
            "producer_source_sha256": PRODUCER_SOURCE_HASHES,
            "endpoint_metadata_sha256": {endpoint: endpoint_hash},
            "endpoint_metadata": {endpoint: endpoint_metadata},
        },
        "runs": runs,
        "tables": {
            name: _table_record(shard / name, rows)
            for name, rows in table_rows.items()
        },
    }
    _write_json(shard / "manifest.json", manifest)


def _make_fixture(
    root,
    *,
    reverse=False,
    endpoint_drift_shard=None,
    config_drift_shard=None,
    contract_drift_shard=None,
):
    specs = [
        (reserve_label, reserve_case, endpoint)
        for reserve_label, reserve_case in RESERVE_CASES
        for endpoint in COMPUTE_ENDPOINTS
    ]
    if reverse:
        specs.reverse()
    for reserve_label, reserve_case, endpoint in specs:
        shard_name = f"{reserve_label}-{endpoint}"
        _make_shard(
            root,
            reserve_label,
            reserve_case,
            endpoint,
            reverse_rows=reverse,
            endpoint_metadata_salt=(
                "drift" if shard_name == endpoint_drift_shard else ""
            ),
            invariant_config_delta=(
                1 if shard_name == config_drift_shard else 0
            ),
            experiment_contract_delta=(
                1 if shard_name == contract_drift_shard else 0
            ),
        )


def _rewrite_report(root, shard_name, policy, mutate):
    shard = root / "parts" / shard_name
    manifest_path = shard / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    run = next(record for record in manifest["runs"] if record["policy"] == policy)
    report_path = shard / run["report"]
    report = json.loads(report_path.read_text())
    mutate(report)
    _write_json(report_path, report)
    report_hash = _sha256(report_path)
    run["report_sha256"] = report_hash
    summary_path = shard / "summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    for row in rows:
        if row["baseline"] == policy:
            row["report_sha256"] = report_hash
    _write_csv(summary_path, tuple(rows[0]), rows)
    manifest["tables"]["summary.csv"] = _table_record(summary_path, rows)
    _write_json(manifest_path, manifest)


class QwenShardCollectorTests(unittest.TestCase):
    def test_collects_36_rows_in_canonical_order_and_is_input_order_stable(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = Path(first_dir)
            second = Path(second_dir)
            _make_fixture(first)
            _make_fixture(second, reverse=True)

            manifest = collect(first, overwrite=False, command="collector one")
            collect(second, overwrite=False, command="collector two")

            self.assertEqual(len(manifest["shards"]), 12)
            self.assertEqual(
                manifest["bindings"]["local_source_sha256"],
                LOCAL_SOURCE_HASHES,
            )
            self.assertEqual(
                manifest["bindings"]["calibration_producer_source_sha256"],
                PRODUCER_SOURCE_HASHES,
            )
            self.assertEqual(
                manifest["bindings"]["git"]["commit"],
                "0123456789abcdef",
            )
            self.assertEqual(
                len(manifest["bindings"]["git"]["dirty_by_shard"]),
                12,
            )
            self.assertEqual(
                manifest["bindings"]["workload_metadata_sha256"],
                _json_sha256(
                    {
                        "sha256": "f" * 64,
                        "session_count": 10,
                        "request_count": 30,
                    }
                ),
            )
            self.assertIn(
                "authoritative",
                manifest["authority"]["statement"],
            )
            for table_name in (
                "summary.csv",
                "return_sources.csv",
                "transfer_stages.csv",
            ):
                self.assertEqual(
                    (first / table_name).read_bytes(),
                    (second / table_name).read_bytes(),
                )
                self.assertEqual(
                    manifest["combined_tables"][table_name]["sha256"],
                    _sha256(first / table_name),
                )
            with (first / "summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as input_file:
                rows = list(csv.DictReader(input_file))
            self.assertEqual(len(rows), 36)
            self.assertEqual(rows[0]["reserve_case"], "zero_residual")
            self.assertEqual(
                rows[0]["compute_endpoint"], "central_full_attention"
            )
            self.assertEqual(rows[0]["baseline"], POLICIES[0])

    def test_rejects_tampered_report_and_declared_table_row_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            shard = root / "parts" / "zero-central_full_attention"
            manifest = json.loads((shard / "manifest.json").read_text())
            report = shard / manifest["runs"][0]["report"]
            report.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(CollectionError, "report sha256"):
                collect(root, overwrite=False, command="collector")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            shard = root / "parts" / "zero-central_full_attention"
            manifest_path = shard / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            run = manifest["runs"][0]
            report_path = shard / run["report"]
            report = json.loads(report_path.read_text())
            report["schema_version"] = 13
            _write_json(report_path, report)
            report_hash = _sha256(report_path)
            run["report_sha256"] = report_hash
            summary_path = shard / "summary.csv"
            with summary_path.open(
                "r", encoding="utf-8", newline=""
            ) as input_file:
                summary_rows = list(csv.DictReader(input_file))
            summary_header = tuple(summary_rows[0].keys())
            summary_rows[0]["report_sha256"] = report_hash
            _write_csv(summary_path, summary_header, summary_rows)
            manifest["tables"]["summary.csv"] = _table_record(
                summary_path, summary_rows
            )
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(CollectionError, "report schema must be 15"):
                collect(root, overwrite=False, command="collector")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            shard_manifest = (
                root / "parts" / "zero-central_full_attention" / "manifest.json"
            )
            manifest = json.loads(shard_manifest.read_text())
            manifest["tables"]["summary.csv"]["rows"] = 99
            _write_json(shard_manifest, manifest)
            with self.assertRaisesRegex(CollectionError, "row count"):
                collect(root, overwrite=False, command="collector")

    def test_rejects_cross_shard_provenance_and_layout_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            shard_manifest = (
                root / "parts" / "full-slow_full_attention" / "manifest.json"
            )
            manifest = json.loads(shard_manifest.read_text())
            manifest["workload_provenance"]["status"] = "different"
            _write_json(shard_manifest, manifest)
            with self.assertRaisesRegex(
                CollectionError, "workload provenance differs"
            ):
                collect(root, overwrite=False, command="collector")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(
                root,
                endpoint_drift_shard="full-central_full_attention",
            )
            with self.assertRaisesRegex(
                CollectionError, "endpoint calibration differs"
            ):
                collect(root, overwrite=False, command="collector")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(
                root,
                config_drift_shard="full-central_full_attention",
            )
            with self.assertRaisesRegex(
                CollectionError, "invariant resolved config differs"
            ):
                collect(root, overwrite=False, command="collector")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(
                root,
                contract_drift_shard="full-central_full_attention",
            )
            with self.assertRaisesRegex(
                CollectionError, "experiment_contract differs"
            ):
                collect(root, overwrite=False, command="collector")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            (root / "parts" / "unexpected-shard").mkdir()
            with self.assertRaisesRegex(
                CollectionError, "unexpected: unexpected-shard"
            ):
                collect(root, overwrite=False, command="collector")

    def test_rejects_source_workload_and_git_binding_tampering(self):
        shard_name = "full-slow_full_attention"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            manifest_path = root / "parts" / shard_name / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            source_path = next(iter(SOURCE_HASHES))
            manifest["local_source_sha256"][source_path] = "9" * 64
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                CollectionError, "not an exact matching subset"
            ):
                collect(root, overwrite=False, command="collector")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            for manifest_path in (root / "parts").glob("*/manifest.json"):
                manifest = json.loads(manifest_path.read_text())
                manifest["workload"]["sha256"] = "not-a-digest"
                _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(CollectionError, "SHA-256 digest"):
                collect(root, overwrite=False, command="collector")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            shard_name = "full-slow_full_attention"
            _rewrite_report(
                root,
                shard_name,
                POLICIES[0],
                lambda report: report["policy"].update({"name": "tiered"}),
            )
            with self.assertRaisesRegex(CollectionError, "report policy mismatch"):
                collect(root, overwrite=False, command="collector")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            shard_name = "full-slow_full_attention"
            _rewrite_report(
                root,
                shard_name,
                POLICIES[0],
                lambda report: report[
                    "infinite_hbm_oracle_comparison"
                ].update({"same_gap_durations": False}),
            )
            with self.assertRaisesRegex(
                CollectionError, "oracle equality checks failed"
            ):
                collect(root, overwrite=False, command="collector")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            shard = root / "parts" / "full-slow_full_attention"
            manifest_path = shard / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            summary_path = shard / "summary.csv"
            with summary_path.open(
                "r", encoding="utf-8", newline=""
            ) as input_file:
                summary_rows = list(csv.DictReader(input_file))
            summary_rows[0][
                "cpu_or_ssd_resume_fraction_all_requests"
            ] = "0.999"
            _write_csv(summary_path, tuple(summary_rows[0]), summary_rows)
            manifest["tables"]["summary.csv"] = _table_record(
                summary_path, summary_rows
            )
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                CollectionError, "values differ from rows derived"
            ):
                collect(root, overwrite=False, command="collector")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            stale = (
                root
                / "parts"
                / "full-slow_full_attention"
                / "stale-report.json"
            )
            _write_json(stale, {"stale": True})
            with self.assertRaisesRegex(CollectionError, "artifact set mismatch"):
                collect(root, overwrite=False, command="collector")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            manifest_path = root / "parts" / shard_name / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            producer_path = next(iter(PRODUCER_SOURCE_HASHES))
            manifest["local_source_sha256"][producer_path] = "7" * 64
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                CollectionError, "producer source hashes are not"
            ):
                collect(root, overwrite=False, command="collector")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            manifest_path = root / "parts" / shard_name / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["local_source_sha256"][
                "serving/agentic_kv_qwen3_1m_p4d4.py"
            ] = "8" * 64
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                CollectionError, "local source hashes differ"
            ):
                collect(root, overwrite=False, command="collector")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            manifest_path = root / "parts" / shard_name / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["workload"]["session_count"] = 11
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                CollectionError, "full workload metadata differs"
            ):
                collect(root, overwrite=False, command="collector")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            manifest_path = root / "parts" / shard_name / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["git"]["commit"] = "different-commit"
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(CollectionError, "git commit differs"):
                collect(root, overwrite=False, command="collector")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            manifest_path = root / "parts" / shard_name / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            del manifest["git"]["dirty"]
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                CollectionError, "git dirty must be a boolean"
            ):
                collect(root, overwrite=False, command="collector")

    def test_overwrite_replaces_only_known_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            collect(root, overwrite=False, command="collector one")
            sentinel = root / "keep-me.txt"
            sentinel.write_text("preserve\n", encoding="utf-8")
            part_hashes = {
                path: _sha256(path)
                for path in (root / "parts").glob("*/manifest.json")
            }

            with self.assertRaisesRegex(CollectionError, "pass --overwrite"):
                collect(root, overwrite=False, command="collector two")
            collect(root, overwrite=True, command="collector two")

            self.assertEqual(sentinel.read_text(), "preserve\n")
            self.assertEqual(
                part_hashes,
                {path: _sha256(path) for path in part_hashes},
            )
            self.assertEqual(
                json.loads((root / "manifest.json").read_text())["command"],
                "collector two",
            )

    def test_failed_staged_publish_preserves_previous_generation(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_fixture(root)
            collect(root, overwrite=False, command="collector one")
            before = {
                name: (root / name).read_bytes()
                for name in (
                    "summary.csv",
                    "return_sources.csv",
                    "transfer_stages.csv",
                    "manifest.json",
                )
            }
            real_replace = __import__("os").replace
            calls = 0

            def fail_second_publish(source, destination):
                nonlocal calls
                if Path(destination).parent == root:
                    calls += 1
                    if calls == 2:
                        raise OSError("synthetic publish failure")
                return real_replace(source, destination)

            with mock.patch(
                "serving.agentic_kv_qwen3_1m_collect.os.replace",
                side_effect=fail_second_publish,
            ):
                with self.assertRaisesRegex(OSError, "publish failure"):
                    collect(root, overwrite=True, command="collector two")
            self.assertEqual(
                before,
                {name: (root / name).read_bytes() for name in before},
            )


if __name__ == "__main__":
    unittest.main()

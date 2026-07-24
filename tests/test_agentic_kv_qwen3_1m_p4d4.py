import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from serving.agentic_kv_qwen3_1m_p4d4 import (
    COMPUTE_ENDPOINTS,
    DEFAULT_COMPUTE_ENDPOINT_NAMES,
    LEGACY_H100_CALIBRATION_SOURCE_PATHS,
    LEGACY_H100_PRODUCER_SOURCE_PATHS,
    MODEL_CARD_INFERRED_RUNTIME_RESIDUAL_BYTES_PER_RANK,
    QWEN_CHECKPOINT_BYTES_PER_TP4_RANK,
    ReserveCase,
    _build_parser,
    _json_sha256,
    build_manifest,
    build_return_source_rows,
    build_summary_row,
    build_transfer_stage_rows,
    derive_reserve_cases,
    load_workload_provenance,
    resolve_compute_endpoints,
    _token_distribution,
)
from serving.core.agentic_kv_roofline import AnalysisConfigError


ARCHITECTURE_WEIGHT_BYTES_PER_RANK = 15_285_227_520


def _mock_report():
    return {
        "schema_version": 15,
        "tp_size": 4,
        "context_infeasible_calls": 2,
        "request_makespan_seconds": 9.0,
        "workload": {"max_context_tokens": 1_010_000},
        "execution_scope": {"prompt_compute_scale": 1.0},
        "experiment": {
            "prompt_compute_calibration_metadata_sha256": "calibration-hash"
        },
        "capacity": {
            "hbm_total_bytes_per_rank": 80_000_000_000,
            "model_weight_bytes_per_rank_estimate": (
                ARCHITECTURE_WEIGHT_BYTES_PER_RANK
            ),
            "hbm_static_reserve_bytes_per_rank": 123,
            "prefill_hbm_static_reserve_bytes_per_rank": 456,
            "decode_hbm_static_reserve_bytes_per_rank": 789,
            "prefill_hbm_kv_budget_bytes_per_rank": 1000,
            "decode_hbm_kv_budget_bytes_per_rank": 2000,
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
            "restore_timing": {
                "request_summed_raw_elapsed_seconds": 4.0,
                "request_summed_exposed_compute_admission_gate_seconds": 3.0,
                "wall_clock_exposed_decode_barrier_union_seconds": 2.0,
            },
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
            "analytical_time_fraction_of_executed_prompt_compute": 0.25
        },
        "transfer_queue": {
            "aggregate_queue_wait_seconds": 7.0,
            "aggregate_service_seconds": 8.0,
            "jobs_by_kind": {"cpu_to_decode": 2, "ssd_to_cpu": 1},
            "bytes_by_kind": {"cpu_to_decode": 100, "ssd_to_cpu": 200},
            "queue_wait_seconds_by_kind": {
                "cpu_to_decode": 1.0,
                "ssd_to_cpu": 2.0,
            },
            "service_seconds_by_kind": {
                "cpu_to_decode": 3.0,
                "ssd_to_cpu": 4.0,
            },
        },
        "infinite_hbm_oracle_comparison": {
            "all_calls": {
                "slowdown_fraction_of_oracle_request_summed_service": 0.5
            },
            "session_end_to_end": {"slowdown_fraction_of_oracle": 0.2},
            "trace_makespan": {"slowdown_fraction_of_oracle": 0.1},
        },
    }


def _manifest_calibration_args():
    endpoint = COMPUTE_ENDPOINTS[0]
    metadata = {
        "schema_version": 1,
        "model_kind": "h100_kernel_calibrated_prompt",
        "band": endpoint.band,
        "attention_multiplier": endpoint.attention_multiplier,
        "source_hashes": {"legacy.csv": "legacy-hash"},
        "analytical_model": {
            "equation": "T=max(t_launch, max(F/P, B/BW) * penalty)"
        },
        "validation": {"holdout": {"relative_error_p50": 0.1}},
        "limitations": {
            "measured_qwen3_h100": False,
            "one_million_token_attention_is_extrapolated": True,
        },
    }
    metadata_hash = _json_sha256(metadata)
    return {
        "compute_endpoints": (endpoint,),
        "prompt_compute_calibration_metadata": {
            endpoint.name: metadata
        },
        "prompt_compute_calibration_metadata_sha256": {
            endpoint.name: metadata_hash
        },
        "calibration_artifact": {
            "path": "calibration.json",
            "sha256": "calibration-file-hash",
            "endpoint_metadata_sha256": {
                endpoint.name: metadata_hash
            },
        },
    }


class ReserveDerivationTests(unittest.TestCase):
    def test_token_distribution_uses_declared_nearest_rank(self):
        summary = _token_distribution([1, 2, 3, 4, 100])
        self.assertEqual(summary["p50"], 3)
        self.assertEqual(summary["p90"], 100)
        self.assertEqual(summary["percentile_method"], "nearest_rank")

    def test_exact_primary_reserve_matches_model_card_residual_target(self):
        case = derive_reserve_cases(
            ARCHITECTURE_WEIGHT_BYTES_PER_RANK, "full"
        )[0]
        self.assertEqual(case.name, "full_residual")
        self.assertEqual(case.common_bytes_per_rank, 19_893_012_480)
        self.assertEqual(
            ARCHITECTURE_WEIGHT_BYTES_PER_RANK
            + case.common_bytes_per_rank,
            QWEN_CHECKPOINT_BYTES_PER_TP4_RANK
            + MODEL_CARD_INFERRED_RUNTIME_RESIDUAL_BYTES_PER_RANK,
        )

    def test_all_reserve_sensitivities_are_exact_and_nonnegative(self):
        cases = derive_reserve_cases(
            ARCHITECTURE_WEIGHT_BYTES_PER_RANK, "all"
        )
        self.assertEqual(
            [case.common_bytes_per_rank for case in cases],
            [0, 9_936_923_136, 19_893_012_480],
        )

    def test_custom_role_reserves_are_preserved(self):
        case = derive_reserve_cases(
            ARCHITECTURE_WEIGHT_BYTES_PER_RANK,
            "full",
            common_override=10,
            prefill_override=20,
            decode_override=30,
        )[0]
        self.assertEqual(
            (
                case.common_bytes_per_rank,
                case.prefill_bytes_per_rank,
                case.decode_bytes_per_rank,
            ),
            (10, 20, 30),
        )

    def test_custom_reserve_rejects_sweep(self):
        with self.assertRaises(AnalysisConfigError):
            derive_reserve_cases(
                ARCHITECTURE_WEIGHT_BYTES_PER_RANK,
                "all",
                common_override=10,
            )


class ComputeEndpointTests(unittest.TestCase):
    def test_default_endpoints_are_central_and_never_scale_whole_prompt(self):
        endpoints = resolve_compute_endpoints(None)
        self.assertEqual(
            tuple(endpoint.name for endpoint in endpoints),
            DEFAULT_COMPUTE_ENDPOINT_NAMES,
        )
        self.assertEqual([endpoint.band for endpoint in endpoints], [
            "central",
            "central",
        ])
        self.assertEqual(endpoints[0].attention_multiplier, 1.0)
        self.assertAlmostEqual(endpoints[1].attention_multiplier, 1 / 3)
        self.assertFalse(hasattr(endpoints[0], "scale"))

    def test_fast_and_slow_full_attention_bands_are_selectable(self):
        endpoints = resolve_compute_endpoints([
            "fast_full_attention",
            "slow_full_attention",
        ])
        self.assertEqual([endpoint.band for endpoint in endpoints], [
            "fast",
            "slow",
        ])
        self.assertTrue(
            all(endpoint.attention_multiplier == 1.0 for endpoint in endpoints)
        )

    def test_cli_endpoint_option_is_repeatable(self):
        args = _build_parser().parse_args([
            "--workload", "trace.jsonl",
            "--output-dir", "out",
            "--compute-endpoint", "central_full_attention",
            "--compute-endpoint", "fast_full_attention",
        ])
        endpoints = resolve_compute_endpoints(args.compute_endpoints)
        self.assertEqual([endpoint.name for endpoint in endpoints], [
            "central_full_attention",
            "fast_full_attention",
        ])

    def test_duplicate_endpoint_is_rejected(self):
        with self.assertRaisesRegex(AnalysisConfigError, "cannot be repeated"):
            resolve_compute_endpoints([
                "central_full_attention",
                "central_full_attention",
            ])


class TableHelperTests(unittest.TestCase):
    def test_summary_keeps_all_request_denominator_and_oracle(self):
        report = _mock_report()
        reserve = ReserveCase("test", 123)
        row = build_summary_row(
            report,
            "run",
            COMPUTE_ENDPOINTS[0],
            reserve,
            "tiered",
            "abc",
        )
        self.assertEqual(row["all_request_count"], 10)
        self.assertEqual(row["context_admissible_request_count"], 8)
        self.assertEqual(row["calibration_band"], "central")
        self.assertEqual(row["attention_multiplier"], 1.0)
        self.assertEqual(row["prompt_compute_scale"], 1.0)
        self.assertEqual(
            row["prompt_compute_calibration_metadata_sha256"],
            "calibration-hash",
        )
        self.assertEqual(row["cpu_or_ssd_resume_fraction_all_requests"], 0.3)
        self.assertEqual(
            row["oracle_request_summed_service_slowdown_fraction"], 0.5
        )
        self.assertEqual(row["report_sha256"], "abc")

    def test_return_source_table_has_every_source(self):
        rows = build_return_source_rows(
            _mock_report(), "run", "endpoint", "reserve", "tiered"
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {row["source"] for row in rows},
            {"decode_hbm", "cpu", "ssd", "recompute"},
        )
        cpu = next(row for row in rows if row["source"] == "cpu")
        self.assertEqual(cpu["count"], 1)
        self.assertEqual(cpu["reusable_tokens"], 20)

    def test_transfer_stage_table_preserves_stage_accounting(self):
        rows = build_transfer_stage_rows(
            _mock_report(), "run", "endpoint", "reserve", "tiered"
        )
        self.assertEqual([row["stage"] for row in rows], [
            "cpu_to_decode",
            "ssd_to_cpu",
        ])
        self.assertEqual(sum(row["bytes"] for row in rows), 300)
        self.assertEqual(sum(row["queue_wait_seconds"] for row in rows), 3.0)


class WorkloadProvenanceTests(unittest.TestCase):
    def test_auto_discovers_and_preserves_conversion_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workload = root / "trace.jsonl"
            workload_bytes = b'{"sub_requests": []}\n'
            workload.write_bytes(workload_bytes)
            workload_hash = hashlib.sha256(workload_bytes).hexdigest()
            raw_source = root / "tracelab.jsonl.gz"
            raw_source_bytes = b"raw-tracelab"
            raw_source.write_bytes(raw_source_bytes)
            raw_source_hash = hashlib.sha256(raw_source_bytes).hexdigest()
            sidecar = Path(f"{workload}.manifest.json")
            sidecar_data = {
                "schema_version": 3,
                "generator": "workloads.generators.agent_traces",
                "converter": {
                    "version": "3.1.0",
                    "commit": "converter-commit",
                },
                "source": {
                    "format": "tracelab",
                    "location": str(raw_source),
                    "revision": "v0.0.1",
                    "sha256": raw_source_hash,
                },
                "validation": {
                    "status": "passed_with_warnings",
                    "error_count": 0,
                    "warning_count": 3,
                    "warning_counts": {"fallback": 2, "reset": 1},
                },
                "output": {"sha256": workload_hash},
            }
            sidecar_bytes = (
                json.dumps(sidecar_data, sort_keys=True) + "\n"
            ).encode()
            sidecar.write_bytes(sidecar_bytes)

            provenance = load_workload_provenance(
                workload, workload_hash
            )

            self.assertEqual(provenance["status"], "loaded")
            self.assertEqual(provenance["discovery"], "automatic")
            self.assertEqual(
                provenance["sidecar"]["sha256"],
                hashlib.sha256(sidecar_bytes).hexdigest(),
            )
            self.assertEqual(
                provenance["tracelab_source"]["revision"], "v0.0.1"
            )
            self.assertEqual(
                provenance["tracelab_source"]["sha256"], raw_source_hash
            )
            self.assertEqual(
                provenance["tracelab_source"]["sha256_status"],
                "declared_and_verified",
            )
            self.assertEqual(provenance["converter"]["schema_version"], 3)
            self.assertEqual(provenance["converter"]["version"], "3.1.0")
            self.assertEqual(
                provenance["converter"]["commit"], "converter-commit"
            )
            self.assertEqual(
                provenance["validation"]["status"], "passed_with_warnings"
            )
            self.assertEqual(
                provenance["validation"]["warning_counts"],
                {"fallback": 2, "reset": 1},
            )
            self.assertTrue(
                provenance["validation"]["warning_count_matches_counters"]
            )
            self.assertEqual(
                provenance["output_binding"]["status"], "verified"
            )
            self.assertEqual(provenance["raw_manifest"], sidecar_data)

    def test_current_sidecar_can_compute_undeclared_raw_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workload = root / "trace.jsonl"
            workload.write_text("trace\n", encoding="utf-8")
            workload_hash = hashlib.sha256(b"trace\n").hexdigest()
            raw_source = root / "source.gz"
            raw_source.write_bytes(b"source")
            sidecar = root / "custom.json"
            sidecar.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "generator": "workloads.generators.agent_traces",
                        "source": {
                            "format": "tracelab",
                            "location": str(raw_source),
                            "revision": "v0.0.1",
                        },
                        "validation": {
                            "status": "passed",
                            "error_count": 0,
                            "warning_count": 0,
                            "warning_counts": {},
                        },
                        "output": {"sha256": workload_hash},
                    }
                ),
                encoding="utf-8",
            )

            provenance = load_workload_provenance(
                workload, workload_hash, sidecar
            )

            self.assertEqual(provenance["discovery"], "explicit")
            self.assertEqual(
                provenance["tracelab_source"]["sha256"],
                hashlib.sha256(b"source").hexdigest(),
            )
            self.assertEqual(
                provenance["tracelab_source"]["sha256_status"],
                "computed_from_declared_source_location",
            )
            self.assertEqual(
                provenance["converter"]["version_status"], "missing"
            )
            self.assertEqual(
                provenance["converter"]["commit_status"], "missing"
            )

    def test_automatic_missing_sidecar_is_explicit_but_nonfatal(self):
        with tempfile.TemporaryDirectory() as directory:
            workload = Path(directory) / "tiny.jsonl"
            workload.write_text("tiny\n", encoding="utf-8")
            workload_hash = hashlib.sha256(b"tiny\n").hexdigest()

            provenance = load_workload_provenance(
                workload, workload_hash
            )

            self.assertEqual(provenance["status"], "missing")
            self.assertEqual(provenance["discovery"], "automatic")
            self.assertEqual(
                provenance["output_binding"]["actual_sha256"], workload_hash
            )
            self.assertEqual(
                provenance["converter"]["commit_status"], "missing"
            )

            explicit = Path(directory) / "not-there.json"
            with self.assertRaisesRegex(
                AnalysisConfigError, "explicit workload provenance"
            ):
                load_workload_provenance(
                    workload, workload_hash, explicit
                )

    def test_sidecar_output_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workload = root / "trace.jsonl"
            workload.write_text("trace\n", encoding="utf-8")
            sidecar = Path(f"{workload}.manifest.json")
            sidecar.write_text(
                json.dumps({"output": {"sha256": "wrong"}}),
                encoding="utf-8",
            )

            with self.assertRaises(AnalysisConfigError):
                load_workload_provenance(
                    workload, hashlib.sha256(b"trace\n").hexdigest()
                )


class ManifestHelperTests(unittest.TestCase):
    def test_manifest_records_hashes_and_nonmeasurement_boundary(self):
        local_hashes = {
            "driver.py": "code-hash",
            "configs/model/Qwen/Qwen3-30B-A3B-Instruct-2507.json": (
                "config-hash"
            ),
            **{
                path: f"hash-{index}"
                for index, path in enumerate(
                    LEGACY_H100_CALIBRATION_SOURCE_PATHS
                )
            },
            **{
                path: f"producer-hash-{index}"
                for index, path in enumerate(
                    LEGACY_H100_PRODUCER_SOURCE_PATHS
                )
            },
        }
        manifest = build_manifest(
            command="python -m driver",
            workload={"path": "trace.jsonl", "sha256": "workload-hash"},
            local_source_hashes=local_hashes,
            git_provenance={"commit": "deadbeef", "dirty": True},
            architecture_weight_bytes_per_rank=(
                ARCHITECTURE_WEIGHT_BYTES_PER_RANK
            ),
            reserve_cases=(ReserveCase("full", 19_893_012_480),),
            run_records=({"run_id": "one", "report_sha256": "report-hash"},),
            **_manifest_calibration_args(),
        )
        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(manifest["workload"]["sha256"], "workload-hash")
        self.assertEqual(
            manifest["calibration_artifact"]["sha256"],
            "calibration-file-hash",
        )
        self.assertEqual(manifest["workload_provenance"]["status"], "missing")
        self.assertEqual(
            manifest["official_qwen_sources"]["config_1m"]["sha256"],
            "bacd81916858d5b9f5daa616ee3aca13e3f888ceeb374cc804546de008dc85d0",
        )
        boundary = manifest["calibration_boundary"]
        self.assertFalse(boundary["measured_qwen3_1m_dca_h100_profile_used"])
        self.assertFalse(boundary["profiler_perf_bundle_created"])
        self.assertTrue(boundary["legacy_h100_measured_kernel_evidence_used"])
        self.assertEqual(boundary["legacy_h100_measurement_dtype"], "float16")
        self.assertEqual(boundary["target_qwen_compute_dtype"], "bfloat16")
        self.assertTrue(
            boundary["fp16_to_bf16_kernel_efficiency_transfer_assumed"]
        )
        self.assertFalse(boundary["attention_holdout_validates_long_k_or_1m"])
        self.assertFalse(boundary["collective_latency_measured_or_fitted"])
        self.assertFalse(boundary["exact_legacy_h100_sku_known"])
        self.assertFalse(boundary["absolute_dgx_h100_latency_validated"])
        self.assertFalse(
            boundary["current_producer_revision_proven_to_match_csv"]
        )
        self.assertEqual(
            boundary["ep_collective_backend"],
            "allgather_reducescatter",
        )
        evidence = manifest["repository_profile_evidence"][
            "legacy_h100_kernel_calibration"
        ]
        self.assertEqual(len(evidence), 4)
        self.assertTrue(all(item["used"] for item in evidence))
        self.assertTrue(
            all(item["measurement_dtype"] == "float16" for item in evidence)
        )
        self.assertEqual(
            {item["artifact_kind"] for item in evidence},
            {"layers", "attention"},
        )
        producer = manifest["repository_profile_evidence"][
            "current_legacy_producer_snapshot"
        ]
        self.assertEqual(
            set(producer["source_sha256"]),
            set(LEGACY_H100_PRODUCER_SOURCE_PATHS),
        )
        self.assertFalse(producer["proven_to_be_measurement_revision"])
        calibration = manifest["prompt_compute_calibration"]
        endpoint_name = COMPUTE_ENDPOINTS[0].name
        self.assertEqual(calibration["fit_invocations_this_driver_run"], 1)
        self.assertEqual(calibration["whole_prompt_compute_scale"], 1.0)
        self.assertIn(
            "t_roof=max",
            calibration["method_reference"]["adapted_equation"],
        )
        self.assertEqual(
            set(calibration["source_sha256"]),
            set(LEGACY_H100_CALIBRATION_SOURCE_PATHS),
        )
        self.assertEqual(
            calibration["endpoint_metadata_sha256"][endpoint_name],
            _json_sha256(calibration["endpoint_metadata"][endpoint_name]),
        )
        self.assertIn(
            "equation",
            calibration["endpoint_metadata"][endpoint_name][
                "analytical_model"
            ],
        )
        self.assertIn(
            "holdout",
            calibration["endpoint_metadata"][endpoint_name]["validation"],
        )
        self.assertTrue(
            manifest["experiment_contract"][
                "infinite_hbm_oracle_paired_for_every_run"
            ]
        )

    def test_manifest_preserves_loaded_workload_provenance(self):
        provenance = {
            "status": "loaded",
            "sidecar": {"sha256": "sidecar-hash"},
            "tracelab_source": {
                "revision": "v0.0.1",
                "sha256": "source-hash",
            },
            "converter": {
                "schema_version": 3,
                "version": "3.1.0",
                "commit": "converter-commit",
            },
            "validation": {
                "status": "passed_with_warnings",
                "warning_count": 2,
                "warning_counts": {"fallback": 2},
            },
        }
        manifest = build_manifest(
            command="python -m driver",
            workload={"path": "trace.jsonl", "sha256": "workload-hash"},
            local_source_hashes={"driver.py": "code-hash"},
            git_provenance={"commit": "deadbeef", "dirty": True},
            architecture_weight_bytes_per_rank=(
                ARCHITECTURE_WEIGHT_BYTES_PER_RANK
            ),
            reserve_cases=(ReserveCase("full", 19_893_012_480),),
            run_records=(),
            workload_provenance=provenance,
            **_manifest_calibration_args(),
        )
        self.assertEqual(manifest["workload_provenance"], provenance)


if __name__ == "__main__":
    unittest.main()

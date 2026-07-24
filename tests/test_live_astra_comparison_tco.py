import copy
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from serving.core.hbf_comparison_tco import (
    HBFComparisonTCOError,
    ORACLE_SYSTEM_KEY,
    PROPOSED_SYSTEM_KEY,
    TIERING_SYSTEM_KEY,
    SensitivityAxes,
)
from serving.live_astra_comparison_tco import (
    HBF_LIVE_LAYOUTS,
    ORACLE_CLUSTER_CONFIG,
    ORACLE_POLICY_CONFIG,
    PROPOSED_GPU_CLUSTER_CONFIG,
    PROPOSED_HBF_CONFIG,
    TIERING_CLUSTER_CONFIG,
    TIERING_POLICY_CONFIG,
    LiveAstraTCOError,
    _capacity_semantic_snapshot,
    _deployment_semantic_snapshot,
    _tco_adapter_implementation_sha,
    adapt_collected_campaign,
    load_and_adapt_live_campaign,
    paired_seed_student_t_95,
    write_tco_csv,
    write_tco_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ONE_POINT_AXES = SensitivityAxes(
    npu_logic_capex_ratios_to_gpu_logic=(0.5,),
    hbf_subsystem_capex_ratios_to_hbm_stack=(0.5,),
    npu_logic_power_ratios_to_gpu_logic=(0.5,),
    hbf_subsystem_power_ratios_to_hbm_stack=(3.5,),
)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_sha256(value):
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def copy_deployment_configs(directory):
    copied_root = Path(directory)
    for relative in {
        TIERING_CLUSTER_CONFIG,
        TIERING_POLICY_CONFIG,
        PROPOSED_GPU_CLUSTER_CONFIG,
        PROPOSED_HBF_CONFIG,
        ORACLE_POLICY_CONFIG,
    }:
        target = copied_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)
    return copied_root


def mutate_json(root, relative, mutation):
    path = root / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def hbf_runtime_fixture(layout_key):
    tp_size, replicas = {
        "tp4": (4, 2),
        "tp8": (8, 1),
        "tp8_context": (8, 1),
    }[layout_key]
    layout = {
        "key": layout_key,
        "tp_size": tp_size,
        "replicas": replicas,
    }
    used_by_card = {}
    for group_id in range(replicas):
        used_by_card[str(group_id)] = {
            str(card): 0
            for card in range(
                group_id * tp_size, (group_id + 1) * tp_size)
        }
    zero_by_group = {str(group_id): 0 for group_id in range(replicas)}
    return {
        "layout": layout,
        "adapter": {
            "pool": {
                "layout": layout,
                "hardware": {
                    "lpddr_capacity_bytes_per_card": 64 * 1024 ** 3,
                },
                "workspace_bytes_per_card": 0,
                "lpddr_kv_capacity_bytes_per_card": 64 * 1024 ** 3,
                "lpddr_ledger_capacity_bytes_per_card": 64 * 1024 ** 3,
                "lpddr_used_bytes_per_group": zero_by_group,
                "lpddr_used_bytes_by_card": used_by_card,
            },
            "lifecycle": {
                "layout": layout,
                "group_reserved_per_card_bytes": zero_by_group,
                "group_reserved_bytes_by_card": used_by_card,
                "sessions": {
                    "session": {
                        "state": "ended",
                        "committed_hbf_tokens": 0,
                        "lpddr_tokens": 0,
                        "gpu_retained_bytes": 0,
                        "committed_per_card_bytes": 0,
                        "pending_reserved_per_card_bytes": 0,
                        "group_id": None,
                        "active_request_id": None,
                        "migration_job_ids": [],
                        "append_job_ids": [],
                    },
                },
            },
        },
        "gpu_hbm_bridge": {
            "memory_by_instance": {
                "0": {
                    "npu_used_per_rank_bytes": 0,
                    "dynamic_used_per_rank_bytes": 0,
                    "bridge_owned_per_rank_bytes": 0,
                },
                "1": {
                    "npu_used_per_rank_bytes": 0,
                    "dynamic_used_per_rank_bytes": 0,
                    "bridge_owned_per_rank_bytes": 0,
                },
            },
            "pending_colocated_claims": [],
            "pending_pd_recompute_bindings": [],
            "pending_pd_decode_reservations": [],
        },
    }


def common_validity():
    return {
        "verified_artifact_count": 5,
        "parsed_request_count": 14,
        "measurement_request_count": 14,
        "measurement_resume_request_count": 7,
        "headline_metric_crosscheck_count": 42,
        "headline_metric_crosscheck_mismatch_count": 0,
        "session_timing_checked_requests": 14,
        "session_timing_passed": True,
        "session_timing_violation_count": 0,
        "session_timing_warning_count": 0,
        "measurement_complete": True,
        "measurement_boundary_complete": True,
        "measurement_early_stopped": False,
        "paired_workload_sha_verified": True,
    }


def non_hbf_validity(*, oracle):
    result = common_validity()
    result.update({
        "bridge_external_fabric_pending_jobs": 0,
        "bridge_open_astra_windows": 0,
        "bridge_pending_direct_fabric_prepare_locks": 0,
        "bridge_transient_dram_capacity_violations": 0,
        "cutoff_outstanding_dma_jobs": 0,
        "cutoff_measurement_censored": False,
        "external_fabric_censored_jobs": 0,
        "external_fabric_pending_jobs": 0,
        "external_fabric_issued_jobs": 7,
        "external_fabric_completed_jobs": 7,
    })
    if oracle:
        result.update({
            "oracle_enabled": True,
            "oracle_passed": True,
            "oracle_checked_reusable_resumes": 7,
            "oracle_instance_count": 4,
            "oracle_nonbinding_instance_count": 4,
            "oracle_nonzero_invariant_count": 0,
            "oracle_zero_invariant_count": 19,
            "oracle_violation_count": 0,
        })
    return result


def hbf_validity():
    result = common_validity()
    for field in (
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
    ):
        result[field] = 0
    result["lifecycle_external_completed_dispatches"] = 7
    result["multiplexer_completed_jobs"] = 40
    return result


def campaign_fixture(
        directory, *,
        seeds=(101, 102, 103),
        hbf_system="hbf_tp8_context",
        output_goodputs=None):
    if output_goodputs is None:
        output_goodputs = {
            "ssd_tiering": (10.0, 12.0, 14.0),
            hbf_system: (20.0, 22.0, 24.0),
            "oracle": (30.0, 32.0, 34.0),
        }
    trace = Path(directory) / "trace.jsonl"
    trace.write_text('{"schema_version": 3}\n', encoding="utf-8")
    trace_sha = sha256_file(trace)
    files = {
        str(trace): trace_sha,
        TIERING_CLUSTER_CONFIG: sha256_file(
            REPO_ROOT / TIERING_CLUSTER_CONFIG),
        TIERING_POLICY_CONFIG: sha256_file(
            REPO_ROOT / TIERING_POLICY_CONFIG),
        PROPOSED_GPU_CLUSTER_CONFIG: sha256_file(
            REPO_ROOT / PROPOSED_GPU_CLUSTER_CONFIG),
        PROPOSED_HBF_CONFIG: sha256_file(
            REPO_ROOT / PROPOSED_HBF_CONFIG),
        ORACLE_POLICY_CONFIG: sha256_file(
            REPO_ROOT / ORACLE_POLICY_CONFIG),
    }
    systems = (
        {
            "key": "ssd_tiering",
            "cluster_config": TIERING_CLUSTER_CONFIG,
            "policy_config": TIERING_POLICY_CONFIG,
            "runtime_kind": "agentic_kv",
            "layout": None,
        },
        {
            "key": "oracle",
            "cluster_config": ORACLE_CLUSTER_CONFIG,
            "policy_config": ORACLE_POLICY_CONFIG,
            "runtime_kind": "oracle",
            "layout": None,
        },
        {
            "key": hbf_system,
            "cluster_config": PROPOSED_GPU_CLUSTER_CONFIG,
            "policy_config": PROPOSED_HBF_CONFIG,
            "runtime_kind": "full_model_hbf",
            "layout": HBF_LIVE_LAYOUTS[hbf_system],
        },
    )
    campaign = {
        "schema_version": 2,
        "scenario_id": "unit-live-tco",
        "scenario_source_sha256": trace_sha,
        "scenario_factory": "unit:factory",
        "scenario_manifest_sha256": "1" * 64,
        "measurement_session_ids_sha256": "2" * 64,
        "trace_path": str(trace),
        "systems": list(systems),
        "rates": [1.0],
        "seeds": list(seeds),
        "ttft_slo_ns": 30_000_000_000,
        "tpot_slo_ns": 300_000_000,
        "files": files,
        "simulator_implementation": {
            "astra_binary": {
                "path": "astra",
                "bytes": 1,
                "sha256": "3" * 64,
            },
            "source_files": {
                "serving/live_astra_comparison_collect.py": {
                    "bytes": (
                        REPO_ROOT
                        / "serving/live_astra_comparison_collect.py"
                    ).stat().st_size,
                    "sha256": sha256_file(
                        REPO_ROOT
                        / "serving/live_astra_comparison_collect.py"),
                },
            },
            "source_scope": {
                "recursive_python_roots": ["serving"],
                "explicit_source_files": [],
            },
        },
    }
    manifest_cells = {}
    compact_cells = []
    for seed_index, seed in enumerate(seeds):
        workload_sha = hashlib.sha256(
            f"workload-{seed}".encode("utf-8")).hexdigest()
        for spec in systems:
            system = spec["key"]
            cell_id = f"seed{seed}-{system}"
            manifest_cells[cell_id] = {
                "status": "completed",
                "system": system,
                "seed": seed,
                "rate": 1.0,
                "workload_sha256": workload_sha,
            }
            cell = {
                "cell_id": cell_id,
                "campaign_sha256": None,
                "system": system,
                "runtime_kind": spec["runtime_kind"],
                "layout": spec["layout"],
                "seed": seed,
                "offered_session_rate_per_second": 1.0,
                "workload_sha256": workload_sha,
                "performance": {
                    "ttft_slo_ns": 30_000_000_000,
                    "tpot_slo_ns": 300_000_000,
                    # This intentionally differs so a request-goodput bug is
                    # immediately visible in the TCO result.
                    "offered_normalized_request_slo_goodput_per_second": (
                        1_000.0 + seed_index),
                    (
                        "offered_normalized_output_token_"
                        "slo_goodput_per_second"
                    ): output_goodputs[system][seed_index],
                },
                "sources": {},
                "bottlenecks": {},
                "validity": (
                    hbf_validity()
                    if system == hbf_system
                    else non_hbf_validity(oracle=system == "oracle")
                ),
            }
            if system == hbf_system:
                cell["bottlenecks"] = {
                    "hbf": {
                        "prefill_drain": {
                            "policy": {
                                "tail_tokens": 2048,
                                "min_tokens": 4096,
                            },
                        },
                    },
                }
            compact_cells.append(cell)
    campaign_sha = stable_sha256(campaign)
    for cell in compact_cells:
        cell["campaign_sha256"] = campaign_sha
    manifest = {
        "schema_version": 2,
        "status": "completed",
        "campaign_sha256": campaign_sha,
        "campaign": campaign,
        "cells": manifest_cells,
    }
    compact = {
        "schema_version": 2,
        "campaign_sha256": campaign_sha,
        "manifest_schema_version": 2,
        "manifest_status": "completed",
        "collected_cell_count": len(compact_cells),
        "skipped_incomplete_cell_count": 0,
        "skipped_incomplete_cell_ids": [],
        "paired_seed_rate_count": len(seeds),
        "cells": compact_cells,
    }
    return manifest, compact


def runtime_reports_for(compact, hbf_system):
    layout = HBF_LIVE_LAYOUTS[hbf_system]
    return {
        cell["cell_id"]: hbf_runtime_fixture(layout)
        for cell in compact["cells"]
        if cell["system"] == hbf_system
    }


def adapt(
        compact, manifest, hbf_system="hbf_tp8_context", *,
        runtime_reports=None):
    if runtime_reports is None:
        runtime_reports = runtime_reports_for(compact, hbf_system)
    return adapt_collected_campaign(
        compact,
        manifest,
        repo_root=REPO_ROOT,
        selected_rate_per_second=1.0,
        selected_hbf_system_key=hbf_system,
        manifest_sha256="a" * 64,
        compact_results_sha256="b" * 64,
        runtime_reports_by_cell_id=runtime_reports,
        axes=ONE_POINT_AXES,
    )


class LiveTCOAdapterTests(unittest.TestCase):
    def test_uses_output_token_goodput_never_request_goodput(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, compact = campaign_fixture(directory)
            report = adapt(compact, manifest)
        provenance = report.performance_provenance
        self.assertEqual(
            provenance.tiering_result.slo_good_output_tokens_per_second,
            12.0,
        )
        self.assertEqual(
            provenance.proposed_result.slo_good_output_tokens_per_second,
            22.0,
        )
        self.assertEqual(
            provenance.oracle_result.slo_good_output_tokens_per_second,
            32.0,
        )
        self.assertNotEqual(
            provenance.tiering_result.slo_good_output_tokens_per_second,
            1_001.0,
        )
        self.assertEqual(
            provenance.tiering_result.metric_json_path,
            (
                "cells[].performance."
                "offered_normalized_output_token_slo_goodput_per_second"
            ),
        )

    def test_tp8_context_discloses_unique_usable_kv_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, compact = campaign_fixture(directory)
            report = adapt(compact, manifest)
        capacity = report.memory_capacity
        self.assertEqual(capacity.selected_hbf_layout_key, "tp8_context")
        self.assertEqual(
            capacity.selected_hbf_physical_kv_replication_factor, 1)
        self.assertEqual(
            capacity.proposed_usable_logical_hbf_kv_capacity_bytes,
            (
                capacity.hbf_capacity_bytes_per_card
                - capacity.hbf_model_weight_bytes_per_card
            ) * 8,
        )
        live = (
            report.performance_provenance.live_artifact_provenance)
        self.assertEqual(
            live.selected_hbf_system_key, "hbf_tp8_context")
        self.assertEqual(
            live.selected_hbf_layout_key, "tp8_context")
        self.assertEqual(live.active_prefill_drain_policy_version, 2)
        self.assertIn("marginal", live.confidence_interval_semantics)
        self.assertIn(
            "not a paired-difference", live.confidence_interval_semantics)
        self.assertEqual(
            live.capacity_semantic_snapshot["layouts"]
            ["hbf_tp8_context"]
            ["usable_logical_hbf_kv_capacity_bytes"],
            capacity.proposed_usable_logical_hbf_kv_capacity_bytes,
        )
        self.assertEqual(
            live.deployment_semantic_snapshot["tiering_ssd_devices"], 16)
        self.assertEqual(
            live.deployment_semantic_snapshot["proposed_ssd_devices"], 0)

    def test_tp4_duplicates_weights_not_kv_and_conventional_tp8_gqa_kv(self):
        with tempfile.TemporaryDirectory() as directory:
            tp4_manifest, tp4_compact = campaign_fixture(
                directory,
                hbf_system="hbf_tp4",
                output_goodputs={
                    "ssd_tiering": (10.0, 12.0, 14.0),
                    "hbf_tp4": (20.0, 22.0, 24.0),
                    "oracle": (30.0, 32.0, 34.0),
                },
            )
            tp4 = adapt(tp4_compact, tp4_manifest, "hbf_tp4")
            tp8_manifest, tp8_compact = campaign_fixture(
                directory,
                hbf_system="hbf_tp8",
                output_goodputs={
                    "ssd_tiering": (10.0, 12.0, 14.0),
                    "hbf_tp8": (20.0, 22.0, 24.0),
                    "oracle": (30.0, 32.0, 34.0),
                },
            )
            tp8 = adapt(tp8_compact, tp8_manifest, "hbf_tp8")
        tp4_capacity = tp4.memory_capacity
        tp8_capacity = tp8.memory_capacity
        self.assertEqual(
            tp4_capacity.selected_hbf_physical_kv_replication_factor, 1)
        self.assertEqual(
            tp8_capacity.selected_hbf_physical_kv_replication_factor, 2)
        self.assertEqual(tp4_capacity.selected_hbf_replica_count, 2)
        self.assertEqual(tp8_capacity.selected_hbf_replica_count, 1)
        self.assertGreater(
            tp4_capacity.hbf_model_weight_bytes_per_card,
            tp8_capacity.hbf_model_weight_bytes_per_card,
        )
        self.assertGreater(
            tp4_capacity.proposed_usable_logical_hbf_kv_capacity_bytes,
            tp8_capacity.proposed_usable_logical_hbf_kv_capacity_bytes,
        )

    def test_paired_seed_student_t_95_ci(self):
        mean, lower, upper, method = paired_seed_student_t_95(
            (10.0, 12.0, 14.0))
        expected_margin = 4.30265272975 * 2.0 / math.sqrt(3.0)
        self.assertEqual(mean, 12.0)
        self.assertAlmostEqual(lower, 12.0 - expected_margin)
        self.assertAlmostEqual(upper, 12.0 + expected_margin)
        self.assertEqual(
            method, "marginal_student_t_95_over_seed_aligned_cells")
        two_mean, two_lower, two_upper, _ = paired_seed_student_t_95(
            (10.0, 14.0))
        two_margin = 12.7062047364 * 2.0
        self.assertEqual(two_mean, 12.0)
        self.assertAlmostEqual(two_lower, 12.0 - two_margin)
        self.assertAlmostEqual(two_upper, 12.0 + two_margin)

        with tempfile.TemporaryDirectory() as directory:
            manifest, compact = campaign_fixture(directory)
            report = adapt(compact, manifest)
        result = report.performance_provenance.tiering_result
        self.assertAlmostEqual(
            result.confidence_interval_lower_tokens_per_second,
            12.0 - expected_margin,
        )
        self.assertAlmostEqual(
            result.confidence_interval_upper_tokens_per_second,
            12.0 + expected_margin,
        )
        self.assertEqual(result.seed_count, 3)

    def test_single_seed_has_explicitly_unavailable_ci(self):
        output = {
            "ssd_tiering": (10.0,),
            "hbf_tp8_context": (20.0,),
            "oracle": (30.0,),
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest, compact = campaign_fixture(
                directory, seeds=(101,), output_goodputs=output)
            report = adapt(compact, manifest)
        for result in (
            report.performance_provenance.tiering_result,
            report.performance_provenance.proposed_result,
            report.performance_provenance.oracle_result,
        ):
            self.assertEqual(result.seed_count, 1)
            self.assertEqual(
                result.confidence_interval_method,
                "not_available_single_seed_aligned_cell",
            )
            self.assertIsNone(
                result.confidence_interval_lower_tokens_per_second)
            self.assertIsNone(
                result.confidence_interval_upper_tokens_per_second)
            self.assertEqual(
                result.schedule_hash_semantics,
                "single_frozen_schedule",
            )

    def test_oracle_is_performance_only_and_never_enters_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, compact = campaign_fixture(directory)
            report = adapt(compact, manifest)
        self.assertEqual(
            report.economic_system_keys,
            (TIERING_SYSTEM_KEY, PROPOSED_SYSTEM_KEY),
        )
        oracle = report.oracle_performance_reference
        self.assertEqual(oracle.system_key, ORACLE_SYSTEM_KEY)
        self.assertFalse(oracle.included_in_main_tco_comparison)
        self.assertFalse(oracle.physical_bom_available)
        self.assertIsNone(oracle.tco_usd)
        self.assertIsNone(oracle.tokens_per_usd)
        for row in report.sensitivity_rows:
            self.assertNotEqual(
                row.tiering_cost.system_key, ORACLE_SYSTEM_KEY)
            self.assertNotEqual(
                row.proposed_cost.system_key, ORACLE_SYSTEM_KEY)

    def test_tampered_workload_config_layout_and_campaign_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, compact = campaign_fixture(directory)

            bad_workload = copy.deepcopy(compact)
            hbf = next(
                cell for cell in bad_workload["cells"]
                if cell["system"] == "hbf_tp8_context")
            hbf["workload_sha256"] = "f" * 64
            with self.assertRaisesRegex(
                    LiveAstraTCOError, "disagrees with manifest"):
                adapt(bad_workload, manifest)

            bad_config_manifest = copy.deepcopy(manifest)
            bad_config_manifest["campaign"]["files"][
                TIERING_POLICY_CONFIG] = "f" * 64
            bad_config_manifest["campaign_sha256"] = stable_sha256(
                bad_config_manifest["campaign"])
            bad_config_compact = copy.deepcopy(compact)
            bad_config_compact["campaign_sha256"] = (
                bad_config_manifest["campaign_sha256"])
            for cell in bad_config_compact["cells"]:
                cell["campaign_sha256"] = (
                    bad_config_manifest["campaign_sha256"])
            with self.assertRaisesRegex(
                    LiveAstraTCOError, "digest changed"):
                adapt(bad_config_compact, bad_config_manifest)

            bad_layout = copy.deepcopy(compact)
            for cell in bad_layout["cells"]:
                if cell["system"] == "hbf_tp8_context":
                    cell["layout"] = "tp8"
            with self.assertRaisesRegex(
                    LiveAstraTCOError, "runtime_kind/layout disagrees"):
                adapt(bad_layout, manifest)

            bad_campaign = copy.deepcopy(compact)
            bad_campaign["campaign_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                    LiveAstraTCOError,
                    "compact and manifest campaign hashes differ"):
                adapt(bad_campaign, manifest)

    def test_runtime_kind_and_comprehensive_validity_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, compact = campaign_fixture(directory)
            mutations = (
                (
                    "baseline measurement incomplete",
                    "ssd_tiering",
                    "validity",
                    "measurement_complete",
                    False,
                    "measurement_complete",
                ),
                (
                    "artifact count",
                    "ssd_tiering",
                    "validity",
                    "verified_artifact_count",
                    4,
                    "verified_artifact_count",
                ),
                (
                    "oracle invariant",
                    "oracle",
                    "validity",
                    "oracle_passed",
                    False,
                    "oracle_passed",
                ),
                (
                    "HBF quarantine",
                    "hbf_tp8_context",
                    "validity",
                    "multiplexer_quarantined_dispatches",
                    1,
                    "multiplexer_quarantined_dispatches",
                ),
            )
            for (
                label, system, parent, field, value, error,
            ) in mutations:
                with self.subTest(label=label):
                    changed = copy.deepcopy(compact)
                    cell = next(
                        item for item in changed["cells"]
                        if item["system"] == system)
                    cell[parent][field] = value
                    with self.assertRaisesRegex(
                            LiveAstraTCOError, error):
                        adapt(changed, manifest)

            bad_kind = copy.deepcopy(compact)
            next(
                item for item in bad_kind["cells"]
                if item["system"] == "ssd_tiering"
            )["runtime_kind"] = "oracle"
            with self.assertRaisesRegex(
                    LiveAstraTCOError, "runtime_kind/layout disagrees"):
                adapt(bad_kind, manifest)

            missing = copy.deepcopy(compact)
            del next(
                item for item in missing["cells"]
                if item["system"] == "oracle"
            )["validity"]["session_timing_passed"]
            with self.assertRaisesRegex(
                    LiveAstraTCOError, "missing required field"):
                adapt(missing, manifest)

    def test_raw_hbf_terminal_ledgers_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, compact = campaign_fixture(directory)
            runtimes = runtime_reports_for(
                compact, "hbf_tp8_context")
            cell_id = next(iter(runtimes))
            retained = copy.deepcopy(runtimes)
            retained[cell_id]["adapter"]["pool"][
                "lpddr_used_bytes_per_group"]["0"] = 1
            with self.assertRaisesRegex(
                    LiveAstraTCOError, "retained nonzero"):
                adapt(
                    compact,
                    manifest,
                    runtime_reports=retained,
                )

            active = copy.deepcopy(runtimes)
            active[cell_id]["adapter"]["lifecycle"]["sessions"][
                "session"]["state"] = "active"
            with self.assertRaisesRegex(
                    LiveAstraTCOError, "non-ended"):
                adapt(
                    compact,
                    manifest,
                    runtime_reports=active,
                )

            gpu_owned = copy.deepcopy(runtimes)
            gpu_owned[cell_id]["gpu_hbm_bridge"]["memory_by_instance"][
                "0"]["bridge_owned_per_rank_bytes"] = 1
            with self.assertRaisesRegex(
                    LiveAstraTCOError, "retained bridge_owned"):
                adapt(
                    compact,
                    manifest,
                    runtime_reports=gpu_owned,
                )

    def test_validity_types_reject_bool_float_and_negative_counters(self):
        mutations = (
            (
                "true field encoded as one",
                "ssd_tiering",
                "measurement_complete",
                1,
                "measurement_complete",
            ),
            (
                "false field encoded as zero",
                "ssd_tiering",
                "measurement_early_stopped",
                0,
                "measurement_early_stopped",
            ),
            (
                "expected integer encoded as bool",
                "ssd_tiering",
                "verified_artifact_count",
                True,
                "verified_artifact_count",
            ),
            (
                "zero integer encoded as bool",
                "ssd_tiering",
                "headline_metric_crosscheck_mismatch_count",
                False,
                "headline_metric_crosscheck_mismatch_count",
            ),
            (
                "positive counter encoded as bool",
                "ssd_tiering",
                "parsed_request_count",
                True,
                "parsed_request_count must be an integer",
            ),
            (
                "counter encoded as float",
                "ssd_tiering",
                "external_fabric_issued_jobs",
                7.0,
                "external_fabric_issued_jobs must be an integer",
            ),
            (
                "negative counter",
                "ssd_tiering",
                "external_fabric_completed_jobs",
                -1,
                "must be at least 0",
            ),
            (
                "oracle boolean counter",
                "oracle",
                "oracle_checked_reusable_resumes",
                True,
                "oracle_checked_reusable_resumes must be an integer",
            ),
            (
                "HBF zero encoded as bool",
                "hbf_tp8_context",
                "adapter_pending_gpu_hbm_events",
                False,
                "adapter_pending_gpu_hbm_events",
            ),
            (
                "HBF completed jobs encoded as float",
                "hbf_tp8_context",
                "multiplexer_completed_jobs",
                40.0,
                "multiplexer_completed_jobs must be an integer",
            ),
            (
                "HBF negative completed dispatch count",
                "hbf_tp8_context",
                "lifecycle_external_completed_dispatches",
                -1,
                "must be at least 0",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest, compact = campaign_fixture(directory)
            for label, system, field, value, error in mutations:
                with self.subTest(label=label):
                    changed = copy.deepcopy(compact)
                    cell = next(
                        item for item in changed["cells"]
                        if item["system"] == system)
                    cell["validity"][field] = value
                    with self.assertRaisesRegex(
                            LiveAstraTCOError, error):
                        adapt(changed, manifest)

            bad_slo = copy.deepcopy(compact)
            next(
                item for item in bad_slo["cells"]
                if item["system"] == "ssd_tiering"
            )["performance"]["ttft_slo_ns"] = True
            with self.assertRaisesRegex(
                    LiveAstraTCOError, "ttft_slo_ns must be an integer"):
                adapt(bad_slo, manifest)

    def test_campaign_integer_metadata_rejects_bool_and_float(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, compact = campaign_fixture(directory)
            cases = []

            bad_manifest_schema = copy.deepcopy(manifest)
            bad_manifest_schema["schema_version"] = 2.0
            cases.append((
                "manifest schema float",
                bad_manifest_schema,
                compact,
                "manifest.schema_version must be an integer",
            ))

            bad_compact_schema = copy.deepcopy(compact)
            bad_compact_schema["schema_version"] = 2.0
            cases.append((
                "compact schema float",
                manifest,
                bad_compact_schema,
                "compact.schema_version must be an integer",
            ))

            bad_manifest_identity = copy.deepcopy(compact)
            bad_manifest_identity["manifest_schema_version"] = 2.0
            cases.append((
                "compact manifest schema float",
                manifest,
                bad_manifest_identity,
                "manifest_schema_version must be an integer",
            ))

            bad_skipped = copy.deepcopy(compact)
            bad_skipped["skipped_incomplete_cell_count"] = False
            cases.append((
                "skipped counter bool",
                manifest,
                bad_skipped,
                "skipped_incomplete_cell_count must be an integer",
            ))

            bad_collected = copy.deepcopy(compact)
            bad_collected["collected_cell_count"] = True
            cases.append((
                "collected counter bool",
                manifest,
                bad_collected,
                "collected_cell_count must be an integer",
            ))

            bad_paired = copy.deepcopy(compact)
            bad_paired["paired_seed_rate_count"] = True
            cases.append((
                "paired counter bool",
                manifest,
                bad_paired,
                "paired_seed_rate_count must be an integer",
            ))

            bad_manifest_seed = copy.deepcopy(manifest)
            first_manifest_cell = next(
                iter(bad_manifest_seed["cells"].values()))
            first_manifest_cell["seed"] = True
            cases.append((
                "manifest seed bool",
                bad_manifest_seed,
                compact,
                "manifest cell .*seed must be an integer",
            ))

            bad_compact_seed = copy.deepcopy(compact)
            bad_compact_seed["cells"][0]["seed"] = True
            cases.append((
                "compact seed bool",
                manifest,
                bad_compact_seed,
                "compact cell .*seed must be an integer",
            ))

            for label, changed_manifest, changed_compact, error in cases:
                with self.subTest(label=label):
                    with self.assertRaisesRegex(
                            LiveAstraTCOError, error):
                        adapt(changed_compact, changed_manifest)

    def test_raw_hbf_ledger_types_and_exact_layout_fail_closed(self):
        def layout_tp_bool(runtime):
            runtime["layout"]["tp_size"] = True

        def layout_replicas_bool(runtime):
            runtime["layout"]["replicas"] = True

        def zero_tree_bool(runtime):
            runtime["adapter"]["pool"][
                "lpddr_used_bytes_per_group"]["0"] = False

        def zero_tree_float(runtime):
            runtime["adapter"]["pool"][
                "lpddr_used_bytes_per_group"]["0"] = 0.0

        def zero_tree_negative(runtime):
            runtime["adapter"]["pool"][
                "lpddr_used_bytes_per_group"]["0"] = -1

        def session_bool(runtime):
            runtime["adapter"]["lifecycle"]["sessions"]["session"][
                "committed_hbf_tokens"] = False

        def session_float(runtime):
            runtime["adapter"]["lifecycle"]["sessions"]["session"][
                "lpddr_tokens"] = 0.0

        def session_negative(runtime):
            runtime["adapter"]["lifecycle"]["sessions"]["session"][
                "gpu_retained_bytes"] = -1

        def bridge_bool(runtime):
            runtime["gpu_hbm_bridge"]["memory_by_instance"]["0"][
                "npu_used_per_rank_bytes"] = False

        def bridge_float(runtime):
            runtime["gpu_hbm_bridge"]["memory_by_instance"]["0"][
                "dynamic_used_per_rank_bytes"] = 0.0

        def bridge_negative(runtime):
            runtime["gpu_hbm_bridge"]["memory_by_instance"]["0"][
                "bridge_owned_per_rank_bytes"] = -1

        def ledger_capacity_bool(runtime):
            runtime["adapter"]["pool"][
                "lpddr_ledger_capacity_bytes_per_card"] = True

        def ledger_capacity_wrong(runtime):
            runtime["adapter"]["pool"][
                "lpddr_ledger_capacity_bytes_per_card"] = 1

        def physical_capacity_bool(runtime):
            runtime["adapter"]["pool"]["hardware"][
                "lpddr_capacity_bytes_per_card"] = True

        def workspace_float(runtime):
            runtime["adapter"]["pool"][
                "workspace_bytes_per_card"] = 0.0

        mutations = (
            ("layout tp bool", layout_tp_bool, "tp_size must be an integer"),
            (
                "layout replicas bool",
                layout_replicas_bool,
                "replicas must be an integer",
            ),
            (
                "raw zero bool",
                zero_tree_bool,
                "integer zero leaves",
            ),
            (
                "raw zero float",
                zero_tree_float,
                "integer zero leaves",
            ),
            (
                "raw negative",
                zero_tree_negative,
                "retained nonzero",
            ),
            (
                "session bool",
                session_bool,
                "committed_hbf_tokens must be an integer",
            ),
            (
                "session float",
                session_float,
                "lpddr_tokens must be an integer",
            ),
            (
                "session negative",
                session_negative,
                "must be at least 0",
            ),
            (
                "bridge bool",
                bridge_bool,
                "npu_used_per_rank_bytes must be an integer",
            ),
            (
                "bridge float",
                bridge_float,
                "dynamic_used_per_rank_bytes must be an integer",
            ),
            (
                "bridge negative",
                bridge_negative,
                "must be at least 0",
            ),
            (
                "ledger capacity bool",
                ledger_capacity_bool,
                "capacity_bytes_per_card must be an integer",
            ),
            (
                "ledger capacity wrong",
                ledger_capacity_wrong,
                "capacity algebra changed",
            ),
            (
                "physical capacity bool",
                physical_capacity_bool,
                "lpddr_capacity_bytes_per_card must be an integer",
            ),
            (
                "workspace float",
                workspace_float,
                "workspace_bytes_per_card must be an integer",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest, compact = campaign_fixture(directory)
            base_reports = runtime_reports_for(
                compact, "hbf_tp8_context")
            cell_id = next(iter(base_reports))
            for label, mutation, error in mutations:
                with self.subTest(label=label):
                    runtimes = copy.deepcopy(base_reports)
                    mutation(runtimes[cell_id])
                    with self.assertRaisesRegex(
                            LiveAstraTCOError, error):
                        adapt(
                            compact,
                            manifest,
                            runtime_reports=runtimes,
                        )

    def test_core_deployment_snapshot_rejects_bool_as_integer(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, compact = campaign_fixture(directory)
            report = adapt(compact, manifest)
        live = report.performance_provenance.live_artifact_provenance
        changed = dict(live.deployment_semantic_snapshot)
        changed["gpu_instance_pp_size"] = True
        with self.assertRaisesRegex(
                HBFComparisonTCOError, "does not match"):
            replace(
                live,
                deployment_semantic_snapshot=changed,
                deployment_semantic_snapshot_sha256=stable_sha256(changed),
            )

    def test_implementation_digest_covers_dependencies_and_capacity(self):
        relative_sources = (
            "serving/core/hbf_comparison_tco.py",
            "serving/core/hbf_full_model_latency.py",
            "serving/core/h100_kernel_calibrated_prompt.py",
            "serving/live_astra_comparison_tco.py",
        )
        snapshot = _capacity_semantic_snapshot(REPO_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            copied_root = Path(directory)
            for relative in relative_sources:
                target = copied_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO_ROOT / relative, target)
            original = _tco_adapter_implementation_sha(
                copied_root, snapshot)
            dependency = (
                copied_root
                / "serving/core/h100_kernel_calibrated_prompt.py"
            )
            dependency.write_text(
                dependency.read_text(encoding="utf-8") + "\n# mutation\n",
                encoding="utf-8",
            )
            changed_dependency = _tco_adapter_implementation_sha(
                copied_root, snapshot)
            changed_snapshot = copy.deepcopy(snapshot)
            changed_snapshot["layouts"]["hbf_tp8_context"][
                "physical_kv_replication_factor"] = 2
            changed_capacity = _tco_adapter_implementation_sha(
                copied_root, changed_snapshot)
        self.assertNotEqual(original, changed_dependency)
        self.assertNotEqual(original, changed_capacity)
        self.assertNotEqual(changed_dependency, changed_capacity)

    def test_config_semantic_snapshot_rejects_quantity_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            copied_root = copy_deployment_configs(directory)
            snapshot = _deployment_semantic_snapshot(copied_root)
            self.assertEqual(snapshot["schema_version"], 2)
            self.assertEqual(
                snapshot["gpu_server_model_name"],
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
            )
            self.assertEqual(snapshot["gpu_server_dtype"], "bfloat16")
            self.assertEqual(snapshot["gpu_server_kv_cache_dtype"], "auto")
            self.assertEqual(
                snapshot["gpu_server_prefill_h100_cards"], 4)
            self.assertEqual(
                snapshot["gpu_server_decode_h100_cards"], 4)
            self.assertEqual(snapshot["gpu_instance_tp_size"], 4)
            self.assertEqual(snapshot["gpu_instance_pp_size"], 1)
            self.assertEqual(
                snapshot["cpu_dram_bytes_per_gpu_host"],
                512_000_000_000,
            )
            self.assertEqual(
                snapshot["h100_hbm_bytes_per_card"], 80_000_000_000)
            self.assertEqual(snapshot["tiering_cpu_hosts"], 2)
            self.assertEqual(snapshot["tiering_h100_cards"], 16)
            self.assertEqual(
                snapshot["tiering_cpu_dram_bytes_per_host"],
                512_000_000_000,
            )
            self.assertEqual(snapshot["tiering_ssd_devices_per_host"], 8)
            self.assertEqual(snapshot["tiering_ssd_devices"], 16)
            self.assertEqual(
                snapshot["tiering_ssd_capacity_gb_per_device"], 3840)
            self.assertEqual(snapshot["proposed_gpu_cpu_hosts"], 1)
            self.assertEqual(snapshot["proposed_hbf_cpu_hosts"], 1)
            self.assertEqual(snapshot["proposed_cpu_hosts"], 2)
            self.assertEqual(
                snapshot["proposed_gpu_host_cpu_dram_bytes"],
                512_000_000_000,
            )
            self.assertEqual(
                snapshot["proposed_hbf_host_cpu_dram_bytes"],
                512_000_000_000,
            )
            self.assertEqual(
                snapshot["proposed_hbf_host_cpu_dram_semantics"],
                "explicit_bom_assumption_same_as_gpu_host",
            )
            self.assertEqual(snapshot["proposed_h100_cards"], 8)
            self.assertEqual(snapshot["proposed_hbf_npu_cards"], 8)
            self.assertEqual(snapshot["proposed_lpddr_gib"], 512)
            self.assertEqual(snapshot["proposed_ssd_devices"], 0)

            proposed_path = copied_root / PROPOSED_GPU_CLUSTER_CONFIG
            proposed = json.loads(
                proposed_path.read_text(encoding="utf-8"))
            proposed["nodes"][0]["instances"][0]["num_npus"] = 3
            proposed_path.write_text(
                json.dumps(proposed), encoding="utf-8")
            with self.assertRaisesRegex(
                    LiveAstraTCOError, "exactly four H100"):
                _deployment_semantic_snapshot(copied_root)

    def test_config_semantics_reject_every_adversarial_drift(self):
        def reshape_to_3p5d(raw):
            prefill, decode = raw["nodes"][0]["instances"]
            prefill["num_npus"] = 3
            prefill["tp_size"] = 3
            decode["num_npus"] = 5
            decode["tp_size"] = 5

        mutations = (
            (
                "3P5D with eight total cards",
                PROPOSED_GPU_CLUSTER_CONFIG,
                reshape_to_3p5d,
                "exactly four H100",
            ),
            (
                "boolean num_npus",
                PROPOSED_GPU_CLUSTER_CONFIG,
                lambda raw: raw["nodes"][0]["instances"][0].update(
                    num_npus=True),
                "num_npus must be an integer",
            ),
            (
                "boolean tp_size",
                PROPOSED_GPU_CLUSTER_CONFIG,
                lambda raw: raw["nodes"][0]["instances"][0].update(
                    tp_size=True),
                "tp_size must be an integer",
            ),
            (
                "boolean pp_size",
                PROPOSED_GPU_CLUSTER_CONFIG,
                lambda raw: raw["nodes"][0]["instances"][0].update(
                    pp_size=True),
                "pp_size must be an integer",
            ),
            (
                "wrong model",
                PROPOSED_GPU_CLUSTER_CONFIG,
                lambda raw: raw["nodes"][0]["instances"][0].update(
                    model_name="other/model"),
                "model_name",
            ),
            (
                "wrong dtype",
                PROPOSED_GPU_CLUSTER_CONFIG,
                lambda raw: raw["nodes"][0]["instances"][0].update(
                    dtype="float16"),
                r"\.dtype",
            ),
            (
                "wrong KV dtype",
                PROPOSED_GPU_CLUSTER_CONFIG,
                lambda raw: raw["nodes"][0]["instances"][0].update(
                    kv_cache_dtype="fp8"),
                "kv_cache_dtype",
            ),
            (
                "wrong CPU DRAM",
                PROPOSED_GPU_CLUSTER_CONFIG,
                lambda raw: raw["nodes"][0]["cpu_mem"].update(
                    mem_size=476.0),
                "CPU DRAM",
            ),
            (
                "wrong H100 HBM",
                PROPOSED_GPU_CLUSTER_CONFIG,
                lambda raw: (
                    raw["nodes"][0]["instances"][0]["npu_mem"].update(
                        mem_size=75.0)
                ),
                "H100 HBM",
            ),
            (
                "wrong SSD count",
                TIERING_POLICY_CONFIG,
                lambda raw: raw.update(ssd_num_devices=7),
                "exactly 8 SSDs",
            ),
            (
                "boolean SSD count",
                TIERING_POLICY_CONFIG,
                lambda raw: raw.update(ssd_num_devices=True),
                "ssd_num_devices must be an integer",
            ),
            (
                "wrong SSD capacity",
                TIERING_POLICY_CONFIG,
                lambda raw: raw.update(ssd_capacity_gb=3839),
                "exactly 3840 GB",
            ),
            (
                "floating SSD capacity",
                TIERING_POLICY_CONFIG,
                lambda raw: raw.update(ssd_capacity_gb=3840.0),
                "ssd_capacity_gb must be an integer",
            ),
            (
                "boolean HBF tp_size",
                PROPOSED_HBF_CONFIG,
                lambda raw: raw["layouts"]["tp8"].update(tp_size=True),
                "tp_size must be an integer",
            ),
            (
                "boolean HBF replicas",
                PROPOSED_HBF_CONFIG,
                lambda raw: raw["layouts"]["tp8"].update(replicas=True),
                "replicas must be an integer",
            ),
        )
        for label, relative, mutation, error in mutations:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as directory:
                    copied_root = copy_deployment_configs(directory)
                    mutate_json(copied_root, relative, mutation)
                    with self.assertRaisesRegex(
                            LiveAstraTCOError, error):
                        _deployment_semantic_snapshot(copied_root)

    def test_file_api_recollects_and_rejects_compact_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, compact = campaign_fixture(directory)
            manifest_path = Path(directory) / "manifest.json"
            compact_path = Path(directory) / "compact.json"
            manifest_path.write_text(
                json.dumps(manifest), encoding="utf-8")
            compact_path.write_text(
                json.dumps(compact), encoding="utf-8")
            canonical = copy.deepcopy(compact)
            canonical["paired_seed_rate_count"] = 999
            with patch(
                    "serving.live_astra_comparison_tco.collect_campaign",
                    return_value=canonical):
                with self.assertRaisesRegex(
                        LiveAstraTCOError, "differs from a fresh canonical"):
                    load_and_adapt_live_campaign(
                        manifest_path,
                        compact_path,
                        repo_root=REPO_ROOT,
                        selected_rate_per_second=1.0,
                        selected_hbf_system_key="hbf_tp8_context",
                        axes=ONE_POINT_AXES,
                    )

    def test_json_and_csv_exports_keep_live_identity_and_no_oracle_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, compact = campaign_fixture(directory)
            report = adapt(compact, manifest)
            json_path = Path(directory) / "tco.json"
            csv_path = Path(directory) / "tco.csv"
            write_tco_json(report, json_path)
            write_tco_csv(report, csv_path)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
        self.assertEqual(
            payload["economic_system_keys"],
            [TIERING_SYSTEM_KEY, PROPOSED_SYSTEM_KEY],
        )
        self.assertFalse(
            payload["oracle_performance_reference"]
            ["included_in_main_tco_comparison"]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["selected_hbf_system_key"], "hbf_tp8_context")
        self.assertEqual(rows[0]["oracle_included_in_tco_bom"], "False")
        self.assertEqual(
            int(rows[0]["physical_kv_replication_factor"]), 1)
        json_snapshot = (
            payload["performance_provenance"]
            ["live_artifact_provenance"]
            ["deployment_semantic_snapshot"]
        )
        self.assertEqual(
            json.loads(rows[0]["deployment_semantic_snapshot_json"]),
            json_snapshot,
        )
        self.assertEqual(
            json_snapshot["proposed_hbf_host_cpu_dram_bytes"],
            512_000_000_000,
        )


if __name__ == "__main__":
    unittest.main()

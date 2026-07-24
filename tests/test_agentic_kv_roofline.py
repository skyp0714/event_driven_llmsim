import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from serving.agentic_kv_analyze import main as analyze_main
from serving.core.agentic_kv_roofline import (
    DEFAULT_HARDWARE_SPECS,
    CpuTransferSpec,
    ModelShape,
    cpu_transfer_seconds,
    kv_layout,
    load_agentic_workload,
    load_hardware_config,
    load_model_shape,
    roofline_cached_prefill_seconds,
    roofline_recompute_seconds,
    analyze_model_hardware,
    ssd_media_seconds,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class AgenticKvRooflineTest(unittest.TestCase):
    def test_tp8_kv_bytes_and_gqa_replication(self):
        llama = load_model_shape("meta-llama/Llama-3.1-8B", REPO_ROOT)
        llama_layout = kv_layout(llama, tp_size=8, dtype_bytes=2)
        self.assertEqual(llama_layout.logical_bytes_per_token, 131_072)
        self.assertEqual(llama_layout.physical_bytes_per_token_per_rank, 16_384)
        self.assertEqual(llama_layout.physical_bytes_per_token_cluster, 131_072)
        self.assertEqual(llama_layout.replication_factor, 1.0)

        qwen = load_model_shape(
            "Qwen/Qwen3-30B-A3B-Instruct-2507", REPO_ROOT
        )
        qwen_layout = kv_layout(qwen, tp_size=8, dtype_bytes=2)
        self.assertEqual(qwen_layout.logical_bytes_per_token, 98_304)
        self.assertEqual(qwen_layout.physical_bytes_per_token_per_rank, 24_576)
        self.assertEqual(qwen_layout.physical_bytes_per_token_cluster, 196_608)
        self.assertEqual(qwen_layout.replication_factor, 2.0)
        self.assertIn("replication", qwen_layout.warning)

    def test_workload_uses_positive_tool_wait_and_block_aligned_lcp(self):
        session = {
            "session_id": "s0",
            "arrival_time_ns": 0,
            "sub_requests": [
                {
                    "input_toks": 20,
                    "output_toks": 6,
                    "tool_duration_ns": 100_000_000,
                    "input_tok_ids": list(range(20)),
                    "output_tok_ids": list(range(20, 26)),
                },
                {
                    "input_toks": 30,
                    "output_toks": 2,
                    "tool_duration_ns": 0,
                    "input_tok_ids": list(range(25)) + [100, 101, 102, 103, 104],
                    "output_tok_ids": [200, 201],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            workload_path = Path(temp_dir) / "trace.jsonl"
            workload_path.write_text(json.dumps(session) + "\n", encoding="utf-8")
            workload = load_agentic_workload(workload_path, block_size=8)
        self.assertEqual(workload.positive_tool_transitions, 1)
        self.assertEqual(workload.selected_tool_transitions, 1)
        self.assertEqual(len(workload.transitions), 1)
        transition = workload.transitions[0]
        self.assertEqual(transition.cache_tokens_declared, 25)
        # Previous KV IDs drop the final generated token, leaving a 25-token LCP.
        self.assertEqual(transition.observed_lcp_tokens, 25)
        self.assertEqual(transition.effective_reuse_tokens, 25)
        self.assertEqual(transition.reusable_allocation_tokens, 32)
        self.assertTrue(transition.token_identity_verified)
        self.assertEqual(transition.reuse_source, "token_ids_exact")

    def test_explicit_prefix_metadata_is_used_without_token_ids(self):
        session = {
            "session_id": "reported",
            "sub_requests": [
                {
                    "input_toks": 100,
                    "output_toks": 20,
                    "tool_duration_ns": 1_000_000,
                },
                {
                    "input_toks": 130,
                    "output_toks": 2,
                    "tool_duration_ns": 0,
                    "prefix_reuse_toks": 110,
                    "prefix_reuse_source": "reported",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(session) + "\n", encoding="utf-8")
            workload = load_agentic_workload(path, block_size=16)
        transition = workload.transitions[0]
        self.assertFalse(transition.token_identity_verified)
        self.assertEqual(transition.effective_reuse_tokens, 110)
        self.assertEqual(transition.reusable_allocation_tokens, 112)
        self.assertEqual(transition.reuse_source, "explicit_reported")

    def test_common_context_filter_excludes_infeasible_transitions(self):
        session = {
            "session_id": "context",
            "sub_requests": [
                {
                    "input_toks": 40,
                    "output_toks": 1,
                    "tool_duration_ns": 100,
                },
                {
                    "input_toks": 50,
                    "output_toks": 1,
                    "tool_duration_ns": 100,
                    "prefix_reuse_toks": 40,
                },
                {
                    "input_toks": 70,
                    "output_toks": 1,
                    "tool_duration_ns": 0,
                    "prefix_reuse_toks": 50,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(session) + "\n", encoding="utf-8")
            workload = load_agentic_workload(path, max_context_tokens=64)
        self.assertEqual(len(workload.transitions), 1)
        self.assertEqual(workload.transitions_excluded_context, 1)
        self.assertEqual(workload.max_context_tokens_filter, 64)

    def test_tiered_incremental_lineage_is_transitive(self):
        session = {
            "session_id": "lineage",
            "sub_requests": [
                {
                    "input_toks": 100,
                    "output_toks": 1,
                    "tool_duration_ns": 10_000_000_000,
                },
                {
                    "input_toks": 120,
                    "output_toks": 1,
                    "tool_duration_ns": 10_000_000_000,
                    "prefix_reuse_toks": 50,
                },
                {
                    "input_toks": 140,
                    "output_toks": 1,
                    "tool_duration_ns": 0,
                    "prefix_reuse_toks": 120,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(session) + "\n", encoding="utf-8")
            workload = load_agentic_workload(path)
        model = load_model_shape("meta-llama/Llama-3.1-8B", REPO_ROOT)
        summary = analyze_model_hardware(
            workload, model, DEFAULT_HARDWARE_SPECS["H100"],
            tp_size=8, hbm_ttl_ms=0, cpu_ttl_ms=0,
            tiered_ssd_write_mode="incremental", repo_root=REPO_ROOT,
        )
        bytes_per_token = summary["kv_layout"][
            "physical_bytes_per_token_cluster"]
        self.assertEqual(
            summary["tiered_policy"]["ssd_host_write_bytes"],
            (100 + 120) * bytes_per_token,
        )

    def test_tiered_shadow_record_ttl_forces_full_rewrite(self):
        session = {
            "session_id": "durable-ttl",
            "sub_requests": [
                {
                    "input_toks": 100,
                    "output_toks": 1,
                    "tool_duration_ns": 2_500_000_000,
                },
                {
                    "input_toks": 110,
                    "output_toks": 1,
                    "tool_duration_ns": 1_500_000_000,
                    "prefix_reuse_toks": 100,
                },
                {
                    "input_toks": 120,
                    "output_toks": 1,
                    "tool_duration_ns": 2_500_000_000,
                    "prefix_reuse_toks": 110,
                },
                {
                    "input_toks": 130,
                    "output_toks": 1,
                    "tool_duration_ns": 0,
                    "prefix_reuse_toks": 120,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(session) + "\n", encoding="utf-8")
            workload = load_agentic_workload(path)
        model = load_model_shape("meta-llama/Llama-3.1-8B", REPO_ROOT)
        summary = analyze_model_hardware(
            workload, model, DEFAULT_HARDWARE_SPECS["H100"],
            tp_size=8, hbm_ttl_ms=2_000, cpu_ttl_ms=0,
            ssd_ttl_ms=1_000, tiered_ssd_write_mode="incremental",
            repo_root=REPO_ROOT,
        )
        bytes_per_token = summary["kv_layout"][
            "physical_bytes_per_token_cluster"]
        self.assertEqual(
            summary["tiered_policy"]["ssd_host_write_bytes"],
            (100 + 120) * bytes_per_token,
        )

    def test_cancellable_swap_keeps_hbm_when_tool_returns_early(self):
        session = {
            "session_id": "short",
            "sub_requests": [
                {
                    "input_toks": 4096,
                    "output_toks": 10,
                    "tool_duration_ns": 1,
                },
                {
                    "input_toks": 4200,
                    "output_toks": 1,
                    "tool_duration_ns": 0,
                    "prefix_reuse_toks": 4096,
                    "prefix_reuse_source": "reported",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(session) + "\n", encoding="utf-8")
            workload = load_agentic_workload(path)
        model = load_model_shape("meta-llama/Llama-3.1-8B", REPO_ROOT)
        result = analyze_model_hardware(
            workload,
            model,
            DEFAULT_HARDWARE_SPECS["H100"],
            swap_out_mode="cancellable",
            repo_root=REPO_ROOT,
        )
        self.assertEqual(result["cpu_swap"]["exposed_seconds"]["p50"], 0)
        self.assertGreater(
            result["cpu_swap"]["blocking_exposed_seconds"]["p50"], 0)

    def test_tiered_age_policy_reaches_hbm_cpu_and_ssd(self):
        waits = [1_000_000, 100_000_000, 1_000_000_000, 0]
        sub_requests = []
        for index, wait in enumerate(waits):
            sub_requests.append({
                "input_toks": 32 + index * 16,
                "output_toks": 2,
                "tool_duration_ns": wait,
                "prefix_reuse_toks": 0 if index == 0 else 32 + (index - 1) * 16,
                "prefix_reuse_source": "reported",
            })
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps({
                "session_id": "tiers", "sub_requests": sub_requests,
            }) + "\n", encoding="utf-8")
            workload = load_agentic_workload(path)
        model = load_model_shape("meta-llama/Llama-3.1-8B", REPO_ROOT)
        result = analyze_model_hardware(
            workload,
            model,
            DEFAULT_HARDWARE_SPECS["H100"],
            hbm_ttl_ms=50,
            cpu_ttl_ms=500,
            repo_root=REPO_ROOT,
        )
        self.assertEqual(
            result["tiered_policy"]["resume_source_counts"],
            {"hbm": 1, "cpu": 1, "ssd": 1, "dropped": 0},
        )
        self.assertGreater(result["tiered_policy"]["ssd_host_write_bytes"], 0)
        accounting = result["time_accounting"]
        self.assertLess(
            accounting["modeled_tiered_prompt_compute_seconds"],
            result["recompute"]["full_next_prefill_seconds"]["sum"],
        )
        self.assertGreater(
            accounting["migration_stall_fraction_of_modeled_prompt_active_time"],
            accounting[
                "migration_stall_fraction_of_full_prefill_reference_time"
            ],
        )

    def test_dropped_fallback_recompute_is_not_migration_stall(self):
        session = {
            "session_id": "dropped",
            "sub_requests": [
                {
                    "input_toks": 100,
                    "output_toks": 1,
                    "tool_duration_ns": 10_000_000_000,
                },
                {
                    "input_toks": 120,
                    "output_toks": 1,
                    "tool_duration_ns": 0,
                    "prefix_reuse_toks": 100,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(session) + "\n", encoding="utf-8")
            workload = load_agentic_workload(path)
        model = load_model_shape("meta-llama/Llama-3.1-8B", REPO_ROOT)
        summary = analyze_model_hardware(
            workload,
            model,
            DEFAULT_HARDWARE_SPECS["H100"],
            tp_size=8,
            hbm_ttl_ms=0,
            cpu_ttl_ms=0,
            ssd_ttl_ms=0,
            repo_root=REPO_ROOT,
        )

        tiered = summary["tiered_policy"]
        accounting = summary["time_accounting"]
        hbf = summary["hbf_npu_opportunity"]
        dropped_recompute = tiered[
            "dropped_fallback_recompute_seconds"
        ]["sum"]
        full_prefill = summary["recompute"][
            "full_next_prefill_seconds"
        ]["sum"]

        self.assertEqual(
            tiered["resume_source_counts"],
            {"hbm": 0, "cpu": 0, "ssd": 0, "dropped": 1},
        )
        self.assertGreater(dropped_recompute, 0)
        self.assertEqual(tiered["migration_stall_seconds"]["sum"], 0)
        self.assertEqual(tiered["exposed_seconds"]["sum"], 0)
        self.assertAlmostEqual(
            tiered["resume_overhead_seconds"]["sum"], dropped_recompute
        )
        self.assertEqual(
            accounting["aggregate_tiered_migration_stall_seconds"], 0
        )
        self.assertEqual(
            accounting["aggregate_tiered_request_stall_seconds"], 0
        )
        self.assertAlmostEqual(
            accounting["aggregate_dropped_fallback_recompute_seconds"],
            dropped_recompute,
        )
        self.assertAlmostEqual(
            accounting["modeled_tiered_prompt_compute_seconds"], full_prefill
        )
        self.assertAlmostEqual(
            accounting["modeled_prompt_active_seconds"], full_prefill
        )
        self.assertEqual(
            accounting[
                "migration_stall_fraction_of_modeled_serialized_selected_transition_time"
            ],
            0,
        )
        self.assertEqual(
            accounting[
                "migration_stall_fraction_of_modeled_prompt_active_time"
            ],
            0,
        )
        self.assertEqual(
            accounting[
                "migration_stall_fraction_of_full_prefill_reference_time"
            ],
            0,
        )
        comparisons = summary["comparisons"]
        self.assertEqual(
            comparisons["tiered_exposed_to_recompute_ratio"], 0
        )
        self.assertEqual(
            comparisons["tiered_migration_stall_to_recompute_ratio"], 0
        )
        self.assertGreater(
            comparisons["tiered_resume_overhead_to_recompute_ratio"], 0
        )
        self.assertEqual(hbf["gross_avoidable_migration_stall_seconds"], 0)
        self.assertAlmostEqual(
            hbf["gross_avoidable_dropped_recompute_seconds"],
            dropped_recompute,
        )
        self.assertAlmostEqual(
            hbf["gross_avoidable_stall_upper_bound_seconds"],
            dropped_recompute,
        )

    def test_cancelled_partial_ssd_writes_are_counted(self):
        model = load_model_shape("meta-llama/Llama-3.1-8B", REPO_ROOT)
        hardware = DEFAULT_HARDWARE_SPECS["H100"]
        layout = kv_layout(model, tp_size=8, dtype_bytes=2)
        cache_tokens = 100
        cache_cluster = (
            cache_tokens * layout.physical_bytes_per_token_cluster
        )
        cache_rank = cache_tokens * layout.physical_bytes_per_token_per_rank
        cpu_out = cpu_transfer_seconds(
            cache_cluster, cache_rank, hardware.cpu, "out"
        )
        ssd_media = ssd_media_seconds(cache_cluster, hardware.ssd, "out")
        wait_ns = int(round((cpu_out + ssd_media / 2) * 1e9))
        active_media_seconds = wait_ns / 1e9 - cpu_out
        expected_partial_bytes = min(
            cache_cluster,
            int(math.ceil(
                cache_cluster * active_media_seconds / ssd_media
            )),
        )
        self.assertGreater(expected_partial_bytes, 0)
        self.assertLess(expected_partial_bytes, cache_cluster)

        session = {
            "session_id": "partial-write",
            "sub_requests": [
                {
                    "input_toks": cache_tokens,
                    "output_toks": 1,
                    "tool_duration_ns": wait_ns,
                },
                {
                    "input_toks": 120,
                    "output_toks": 1,
                    "tool_duration_ns": 0,
                    "prefix_reuse_toks": cache_tokens,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(session) + "\n", encoding="utf-8")
            workload = load_agentic_workload(path)
        summary = analyze_model_hardware(
            workload,
            model,
            hardware,
            tp_size=8,
            swap_out_mode="cancellable",
            hbm_ttl_ms=0,
            cpu_ttl_ms=0,
            repo_root=REPO_ROOT,
        )

        direct_writes = summary["ssd_swap"]["host_write_bytes"]
        self.assertEqual(
            direct_writes["full_rewrite_completed_under_selected_mode"], 0
        )
        self.assertEqual(
            direct_writes["cancelled_partial_write_bytes_under_selected_mode"],
            expected_partial_bytes,
        )
        self.assertEqual(
            direct_writes["full_rewrite_issued_under_selected_mode"],
            expected_partial_bytes,
        )
        tiered = summary["tiered_policy"]
        self.assertEqual(
            tiered["resume_source_counts"],
            {"hbm": 0, "cpu": 1, "ssd": 0, "dropped": 0},
        )
        self.assertEqual(tiered["ssd_completed_write_bytes"], 0)
        self.assertEqual(
            tiered["ssd_cancelled_partial_write_bytes"],
            expected_partial_bytes,
        )
        self.assertEqual(
            tiered["ssd_host_write_bytes"], expected_partial_bytes
        )

    def test_lcp_cannot_exceed_declared_physical_cache(self):
        session = {
            "session_id": "bounded",
            "sub_requests": [
                {
                    "input_toks": 4,
                    "output_toks": 1,
                    "tool_duration_ns": 1,
                    "input_tok_ids": list(range(20)),
                    "output_tok_ids": [20],
                },
                {
                    "input_toks": 6,
                    "output_toks": 1,
                    "tool_duration_ns": 0,
                    "input_tok_ids": list(range(20)),
                    "output_tok_ids": [21],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            workload_path = Path(temp_dir) / "trace.jsonl"
            workload_path.write_text(json.dumps(session) + "\n", encoding="utf-8")
            transition = load_agentic_workload(
                workload_path, block_size=1
            ).transitions[0]
        self.assertEqual(transition.cache_tokens_declared, 4)
        self.assertEqual(transition.observed_lcp_tokens, 4)
        self.assertEqual(transition.reusable_allocation_tokens, 4)

    def test_full_prefix_lineage_is_distinct_from_input_minus_one_hit(self):
        session = {
            "session_id": "full-prefix",
            "sub_requests": [
                {
                    "input_toks": 4,
                    "output_toks": 1,
                    "tool_duration_ns": 1,
                    "input_tok_ids": [0, 1, 2, 3],
                    "output_tok_ids": [4],
                },
                {
                    "input_toks": 4,
                    "output_toks": 1,
                    "tool_duration_ns": 0,
                    "input_tok_ids": [0, 1, 2, 3],
                    "output_tok_ids": [5],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(session) + "\n", encoding="utf-8")
            transition = load_agentic_workload(path, block_size=1).transitions[0]
        self.assertEqual(transition.declared_reuse_tokens, 4)
        self.assertEqual(transition.effective_reuse_tokens, 3)
        self.assertEqual(transition.reusable_allocation_tokens, 3)

    def test_missing_output_ids_do_not_drop_a_verified_input_token(self):
        session = {
            "session_id": "partial-ids",
            "sub_requests": [
                {
                    "input_toks": 4,
                    "output_toks": 1,
                    "tool_duration_ns": 1,
                    "input_tok_ids": [0, 1, 2, 3],
                    "output_tok_ids": [],
                },
                {
                    "input_toks": 4,
                    "output_toks": 1,
                    "tool_duration_ns": 0,
                    "input_tok_ids": [0, 1, 2, 3],
                    "output_tok_ids": [],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(session) + "\n", encoding="utf-8")
            transition = load_agentic_workload(path, block_size=1).transitions[0]
        self.assertTrue(transition.token_identity_verified)
        self.assertEqual(transition.declared_reuse_tokens, 4)
        self.assertEqual(transition.effective_reuse_tokens, 3)

    def test_partial_input_ids_do_not_collapse_gap_before_output_ids(self):
        session = {
            "session_id": "positional-gap",
            "sub_requests": [
                {
                    "input_toks": 4,
                    "output_toks": 2,
                    "tool_duration_ns": 1,
                    "input_tok_ids": [10, 11],
                    "output_tok_ids": [12],
                },
                {
                    "input_toks": 4,
                    "output_toks": 1,
                    "tool_duration_ns": 0,
                    "input_tok_ids": [10, 11, 12, 99],
                    "output_tok_ids": [],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text(json.dumps(session) + "\n", encoding="utf-8")
            transition = load_agentic_workload(path, block_size=1).transitions[0]
        self.assertEqual(transition.declared_reuse_tokens, 2)
        self.assertEqual(transition.previous_token_id_coverage, 0.4)

    def test_cpu_transfer_obeys_rank_and_aggregate_bottlenecks(self):
        spec = CpuTransferSpec(
            gpu_to_host_gbps_per_rank=10,
            host_to_gpu_gbps_per_rank=10,
            dram_write_gbps_aggregate=100,
            dram_read_gbps_aggregate=100,
            fixed_latency_us=0,
        )
        # Eight 2 GB ranks: rank links take 0.2 s, aggregate DRAM takes 0.16 s.
        elapsed = cpu_transfer_seconds(
            cluster_bytes=16_000_000_000,
            per_rank_bytes=2_000_000_000,
            spec=spec,
            direction="out",
        )
        self.assertAlmostEqual(elapsed, 0.2)

    def test_h200_roofline_is_not_slower_than_h100_with_same_compute_peak(self):
        model = ModelShape(
            name="tiny",
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=1024,
        )
        h100 = roofline_recompute_seconds(
            model, DEFAULT_HARDWARE_SPECS["H100"], 4096, 8
        )
        h200 = roofline_recompute_seconds(
            model, DEFAULT_HARDWARE_SPECS["H200"], 4096, 8
        )
        self.assertGreater(h100.total_seconds, 0)
        self.assertLessEqual(h200.total_seconds, h100.total_seconds)

    def test_cached_prefill_retains_suffix_launch_and_weight_cost(self):
        model = ModelShape(
            name="tiny",
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=1024,
        )
        hardware = DEFAULT_HARDWARE_SPECS["H100"]
        full = roofline_recompute_seconds(model, hardware, 4096, 8)
        uncached = roofline_cached_prefill_seconds(
            model, hardware, 4096, 0, 8
        )
        suffix = roofline_cached_prefill_seconds(
            model, hardware, 4096, 3072, 8
        )
        prefix = roofline_recompute_seconds(model, hardware, 3072, 8)
        self.assertAlmostEqual(uncached.total_seconds, full.total_seconds)
        self.assertGreater(suffix.total_seconds, 0)
        self.assertLess(suffix.total_seconds, full.total_seconds)
        self.assertGreater(
            suffix.total_seconds,
            max(0.0, full.total_seconds - prefix.total_seconds),
        )

    def test_hardware_json_overrides_are_recorded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "hardware.json"
            config_path.write_text(
                json.dumps(
                    {
                        "H200": {
                            "compute_efficiency": 0.7,
                            "cpu": {
                                "gpu_to_host_gbps_per_rank": 42.0,
                                "provenance": "measured memcpy fixture",
                            },
                            "calibration_provenance": "lab run abc",
                        }
                    }
                ),
                encoding="utf-8",
            )
            specs = load_hardware_config(config_path)
        self.assertEqual(specs["H200"].compute_efficiency, 0.7)
        self.assertEqual(specs["H200"].cpu.gpu_to_host_gbps_per_rank, 42.0)
        self.assertEqual(specs["H200"].cpu.provenance, "measured memcpy fixture")
        self.assertEqual(specs["H200"].calibration_provenance, "lab run abc")

    def test_cli_writes_json_and_csv_with_profile_limitation(self):
        session = {
            "session_id": "cli",
            "sub_requests": [
                {
                    "input_toks": 16,
                    "output_toks": 2,
                    "tool_duration_ns": 10_000_000,
                    "input_tok_ids": list(range(16)),
                    "output_tok_ids": [16, 17],
                },
                {
                    "input_toks": 20,
                    "output_toks": 1,
                    "tool_duration_ns": 0,
                    "input_tok_ids": list(range(17)) + [50, 51, 52],
                    "output_tok_ids": [60],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            workload_path = temp_path / "trace.jsonl"
            workload_path.write_text(json.dumps(session) + "\n", encoding="utf-8")
            result = analyze_main(
                [
                    "--workload",
                    str(workload_path),
                    "--model",
                    "meta-llama/Llama-3.1-8B",
                    "--hardware",
                    "H200",
                    "--output-dir",
                    str(temp_path),
                    "--output-stem",
                    "result",
                ]
            )
            self.assertEqual(result, 0)
            report = json.loads((temp_path / "result.json").read_text())
            with (temp_path / "result.csv").open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
        self.assertEqual(len(rows), 1)
        self.assertIn("total_swap_out_bytes", rows[0])
        self.assertIn("No repository TP8 calibration", rows[0]["hardware_calibration_provenance"])
        summary = report["summaries"][0]
        self.assertFalse(
            summary["profile_provenance"]["requested_tp_profile_available"]
        )
        self.assertEqual(
            summary["profile_provenance"]["mode_used"],
            "analytical_roofline_only",
        )
        self.assertTrue(summary["profile_provenance"]["limitations"])


if __name__ == "__main__":
    unittest.main()

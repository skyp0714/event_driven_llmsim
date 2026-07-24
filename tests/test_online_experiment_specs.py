import copy
import json
import tempfile
import unittest
from pathlib import Path

from serving.online_experiments import (
    ExperimentError,
    _normalize_plot_settings,
    build_run_descriptors,
    materialize_session_cohort,
)


class OnlineExperimentSpecTests(unittest.TestCase):
    def _load_specs(self):
        repo_root = Path(__file__).resolve().parents[1]
        experiment_dir = repo_root / "configs/experiments"
        names = {
            key: experiment_dir / (
                "online_tracelab_qwen3_1m_p4d4_" + key + ".json")
            for key in (
                "quick_backlog",
                "quick_poisson",
                "main_long_backlog",
                "main_long_poisson",
            )
        }
        return repo_root, {
            key: json.loads(path.read_text(encoding="utf-8"))
            for key, path in names.items()
        }

    def test_quick_and_main_long_specs_are_disjoint_contracts(self):
        _, specs = self._load_specs()
        expected_policies = {
            "hbm_lru_recompute",
            "hbm_ssd_direct",
            "hbm_cpu_ssd",
            "hbm_cpu_ssd_queue_recompute",
        }
        for key, spec in specs.items():
            with self.subTest(spec=key):
                self.assertEqual(set(spec["policies"]), expected_policies)
                self.assertEqual(spec["oracle_label"], "infinite_hbm_oracle")
                self.assertEqual(
                    spec["workload_selection"][
                        "target_max_sequence_tokens"],
                    1_000_000,
                )

        for key in ("quick_backlog", "quick_poisson"):
            spec = specs[key]
            self.assertEqual(
                spec["workload_selection"]["include_source_indices"],
                [2113, 3726],
            )
            self.assertEqual(
                spec["dataset_contract"][
                    "expected_selected_template_count"],
                2,
            )
            self.assertEqual(spec["timeout_seconds"], 600)
        self.assertEqual(
            _normalize_plot_settings(
                specs["quick_backlog"]["plots"],
                specs["quick_backlog"]["modes"],
            ),
            {"backlog_oracle_normalized": {"minimum_k": 10}},
        )

        long_indices_by_mode = {
            "main_long_backlog": [
                487, 488, 1759, 1836, 1902, 2021, 2047, 3726,
            ],
            "main_long_poisson": [
                487, 488, 1759, 1836, 1902, 2021, 2047, 3791,
            ],
        }
        for key, long_indices in long_indices_by_mode.items():
            spec = specs[key]
            self.assertEqual(
                spec["workload_selection"]["include_source_indices"],
                long_indices,
            )
            self.assertEqual(
                spec["dataset_contract"][
                    "expected_selected_template_count"],
                8,
            )
            self.assertEqual(
                spec["dataset_contract"]["expected_selected_request_count"],
                24,
            )
            self.assertEqual(spec["timeout_seconds"], 3_600)
            self.assertEqual(spec["max_parallel"], 4)

        backlog = specs["main_long_backlog"]
        backlog_mode = backlog["modes"]["backlog"]
        self.assertEqual(
            backlog_mode["k_values"], [10, 12, 14, 16, 18, 20, 24])
        self.assertEqual(backlog_mode["backlog_epochs"], 13)
        self.assertEqual(backlog_mode["warmup_completions"], 48)
        self.assertEqual(backlog_mode["measure_completions"], 32)
        self.assertEqual(
            backlog_mode["measurement_cohort_selection"],
            "completion_order",
        )
        self.assertTrue(backlog_mode["stop_after_measurement"])
        self.assertEqual(backlog_mode["min_fraction_at_configured_k"], 0.95)
        self.assertEqual(
            _normalize_plot_settings(
                backlog["plots"], backlog["modes"]),
            {"backlog_oracle_normalized": {"minimum_k": 10}},
        )
        self.assertEqual(backlog["ssd_resume_opportunity_contract"], {
            "mode": "backlog",
            "policy": "hbm_cpu_ssd",
            "minimum_fraction_of_all_requests": 0.3,
        })

        poisson_mode = specs["main_long_poisson"]["modes"]["poisson"]
        self.assertEqual(
            poisson_mode["rates_sps"],
            [0.003, 0.006, 0.009, 0.0135, 0.02025, 0.030375],
        )
        self.assertEqual(poisson_mode["arrival_seeds"], [101, 211, 307])
        self.assertEqual(poisson_mode["session_repetitions"], 8)
        self.assertEqual(poisson_mode["max_active_sessions"], 20)
        self.assertEqual(poisson_mode["measure_completions"], "all")
        self.assertFalse(poisson_mode["stop_after_measurement"])
        self.assertTrue(poisson_mode["require_complete_session_cohort"])
        self.assertEqual(
            specs["main_long_poisson"]["dataset_contract"][
                "expected_selected_session_identity_hash"],
            "4dded4375d266cda87c46a7e9c10633e1f7f60ff88c431d10d65c7c67677be58",
        )
        self.assertEqual(
            specs["main_long_poisson"]["ssd_resume_opportunity_contract"],
            {
                "mode": "poisson",
                "policy": "hbm_cpu_ssd",
                "minimum_fraction_of_all_requests": 0.3,
            },
        )
        rate_plots = _normalize_plot_settings(
            specs["main_long_poisson"]["plots"],
            specs["main_long_poisson"]["modes"],
        )["poisson_rate_metrics"]
        resume_slo = rate_plots["resume_ttft_slo"]
        resume_provenance = resume_slo["provenance"]
        self.assertAlmostEqual(
            resume_slo["threshold_ms"],
            resume_provenance["calibration_resume_ttft_p95_ns"]
            * resume_provenance["multiplier"] / 1_000_000,
        )
        self.assertEqual(
            resume_provenance["calibration_session_metrics_sha256"],
            "d0c0832b297486bd6041f9dc00d1b1bea7e5e4ff10f6ac9c5b4b293f9c3f2f13",
        )
        self.assertEqual(rate_plots["tpot_slo"]["threshold_ms"], 100.0)

    def test_discovery_and_main_poisson_freeze_identical_slos(self):
        repo_root = Path(__file__).resolve().parents[1]
        experiment_dir = repo_root / "configs/experiments"
        paths = [
            experiment_dir / (
                "online_tracelab_qwen3_1m_p4d4_"
                "poisson_backlog_discovery.json"),
            experiment_dir / (
                "online_tracelab_qwen3_1m_p4d4_"
                "main_long_poisson.json"),
        ]
        specs = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in paths
        ]
        settings = [
            _normalize_plot_settings(spec["plots"], spec["modes"])[
                "poisson_rate_metrics"]
            for spec in specs
        ]
        self.assertEqual(settings[0], settings[1])

    def test_all_four_specs_build_exact_five_series_grids(self):
        repo_root, specs = self._load_specs()
        session_rows = []
        for session_index in range(8):
            session_rows.append({
                "session_id": f"session-{session_index}",
                "arrival_time_ns": 0,
                "sub_requests": [
                    {
                        "input_toks": 100 + request_index,
                        "output_toks": 1,
                        "tool_duration_ns": 1,
                    }
                    for request_index in range(3)
                ],
            })
        expected_run_counts = {
            "quick_backlog": 25,
            "quick_poisson": 45,
            "main_long_backlog": 35,
            "main_long_poisson": 90,
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "sessions.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in session_rows),
                encoding="utf-8",
            )
            cohort = materialize_session_cohort(
                source, directory / "cohort")
            for key, spec in specs.items():
                with self.subTest(spec=key):
                    runs = build_run_descriptors(
                        spec,
                        repo_root,
                        directory / key,
                        cohort,
                    )
                    self.assertEqual(len(runs), expected_run_counts[key])
                    self.assertEqual(
                        {run["policy"] for run in runs},
                        {
                            "hbm_lru_recompute",
                            "hbm_ssd_direct",
                            "hbm_cpu_ssd",
                            "hbm_cpu_ssd_queue_recompute",
                            "infinite_hbm_oracle",
                        },
                    )
                    pair_sizes = {}
                    for run in runs:
                        pair_sizes[run["pair_key"]] = (
                            pair_sizes.get(run["pair_key"], 0) + 1)
                    self.assertEqual(set(pair_sizes.values()), {5})
                    self.assertEqual(
                        len({
                            run["agentic_hardware_config_hash"]
                            for run in runs
                        }),
                        1,
                    )
                    self.assertEqual(
                        len({
                            run["agentic_shared_control_config_hash"]
                            for run in runs
                        }),
                        1,
                    )
                    self.assertEqual(
                        len({
                            run["agentic_effective_config_hash"]
                            for run in runs
                        }),
                        5,
                    )

            reference = specs["quick_backlog"]
            base_config_path = (
                repo_root
                / "configs/agentic_kv/qwen3_1m_p4d4/hbm_lru_recompute.json"
            )
            base_config = json.loads(
                base_config_path.read_text(encoding="utf-8"))
            for field, value, expected_error in (
                    ("pcie_bandwidth_gbps", 49, "hardware"),
                    ("swap_execution_mode", "sync-engine-barrier",
                     "shared-control")):
                mismatch_config = dict(base_config)
                mismatch_config[field] = value
                mismatch_path = directory / f"mismatch-{field}.json"
                mismatch_path.write_text(
                    json.dumps(mismatch_config), encoding="utf-8")
                mismatch_spec = copy.deepcopy(reference)
                mismatch_spec["policies"]["hbm_lru_recompute"] = str(
                    mismatch_path)
                with self.subTest(mismatch=field):
                    with self.assertRaisesRegex(
                            ExperimentError,
                            f"{expected_error} config mismatch before launch"):
                        build_run_descriptors(
                            mismatch_spec,
                            repo_root,
                            directory / f"mismatch-{field}",
                            cohort,
                        )

    def test_capped_poisson_discovery_is_a_paired_four_series_rate_sweep(self):
        repo_root = Path(__file__).resolve().parents[1]
        path = (
            repo_root
            / "configs/experiments"
            / "online_tracelab_qwen3_1m_p4d4_poisson_backlog_discovery.json"
        )
        spec = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(spec["policies"]), {
            "full_recompute",
            "tiering",
            "tiering_partial_recompute",
        })
        self.assertEqual(spec["oracle_label"], "infinite_hbm_oracle")
        self.assertEqual(
            spec["workload_selection"]["include_source_indices"],
            [2113, 3726],
        )
        mode = spec["modes"]["poisson"]
        self.assertEqual(
            mode["rates_sps"], [
                0.002, 0.003, 0.0045, 0.006, 0.009,
                0.0135, 0.02025, 0.030375,
            ])
        self.assertEqual(mode["session_repetitions"], 16)
        self.assertEqual(mode["max_active_sessions"], 20)
        self.assertEqual(mode["arrival_seeds"], [17])
        self.assertEqual(mode["measure_completions"], "all")
        self.assertTrue(mode["require_complete_session_cohort"])

        session_rows = [
            {
                "session_id": f"session-{session_index}",
                "arrival_time_ns": 0,
                "sub_requests": [
                    {
                        "input_toks": 100 + request_index,
                        "output_toks": 1,
                        "tool_duration_ns": 1,
                    }
                    for request_index in range(2)
                ],
            }
            for session_index in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "sessions.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in session_rows),
                encoding="utf-8",
            )
            cohort = materialize_session_cohort(
                source, directory / "cohort")
            runs = build_run_descriptors(
                spec, repo_root, directory / "runs", cohort)

        self.assertEqual(len(runs), 32)
        pair_groups = {}
        for run in runs:
            pair_groups.setdefault(run["pair_key"], []).append(run)
        self.assertEqual(len(pair_groups), 8)
        for pair_runs in pair_groups.values():
            self.assertEqual(len(pair_runs), 4)
            self.assertEqual(
                {run["arrival_seed"] for run in pair_runs}, {17})
            self.assertEqual(
                {run["max_active_sessions"] for run in pair_runs}, {20})

    def test_main_long_zero_load_slo_calibration_is_preregistered(self):
        repo_root = Path(__file__).resolve().parents[1]
        path = (
            repo_root
            / "configs/experiments"
            / "online_tracelab_qwen3_1m_p4d4_"
            "main_long_zero_load_slo_calibration.json"
        )
        spec = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(set(spec["policies"]), {"hbm_cpu_ssd_control"})
        self.assertEqual(spec["oracle_label"], "infinite_hbm_oracle")
        self.assertEqual(
            spec["workload_selection"]["include_source_indices"],
            [487, 488, 1759, 1836, 1902, 2021, 2047, 3791],
        )
        self.assertEqual(
            spec["dataset_contract"][
                "expected_selected_session_identity_hash"],
            "4dded4375d266cda87c46a7e9c10633e1f7f60ff88c431d10d65c7c67677be58",
        )
        self.assertEqual(
            spec["dataset_contract"]["expected_selected_template_count"],
            8,
        )
        self.assertEqual(
            spec["dataset_contract"]["expected_selected_request_count"],
            24,
        )
        self.assertEqual(
            spec["workload_selection"]["target_max_sequence_tokens"],
            1_000_000,
        )
        self.assertNotIn("ssd_resume_opportunity_contract", spec)
        self.assertNotIn("plots", spec)

        mode = spec["modes"]["poisson"]
        self.assertEqual(mode["rates_sps"], [0.001])
        self.assertEqual(mode["session_repetitions"], 1)
        self.assertEqual(mode["max_active_sessions"], 1)
        self.assertEqual(mode["arrival_seeds"], [9901])
        self.assertEqual(mode["warmup_completions"], 0)
        self.assertEqual(mode["measure_completions"], "all")
        self.assertFalse(mode["stop_after_measurement"])
        self.assertTrue(mode["require_complete_session_cohort"])

        calibration = spec["slo_calibration_contract"]
        self.assertEqual(calibration["primary_series"], "infinite_hbm_oracle")
        self.assertEqual(calibration["calibration_metric"], "resume_ttft_p95_ns")
        self.assertEqual(calibration["post_run_threshold_multiplier"], 5)
        self.assertTrue(calibration["rule_preregistered_before_calibration"])
        self.assertTrue(
            calibration["numeric_threshold_frozen_before_main_measurement"])
        self.assertTrue(calibration["calibration_not_part_of_main_comparison"])
        self.assertNotIn("threshold_ms", calibration)

        session_rows = [
            {
                "session_id": f"session-{session_index}",
                "arrival_time_ns": 0,
                "sub_requests": [
                    {
                        "input_toks": 100 + request_index,
                        "output_toks": 1,
                        "tool_duration_ns": 1,
                    }
                    for request_index in range(3)
                ],
            }
            for session_index in range(8)
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "sessions.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in session_rows),
                encoding="utf-8",
            )
            cohort = materialize_session_cohort(
                source, directory / "cohort")
            runs = build_run_descriptors(
                spec, repo_root, directory / "runs", cohort)

        self.assertEqual(len(runs), 2)
        self.assertEqual(
            {run["policy"] for run in runs},
            {"hbm_cpu_ssd_control", "infinite_hbm_oracle"},
        )
        self.assertEqual({run["pair_key"] for run in runs}, {
            "poisson:0.001:seed=9901",
        })
        self.assertEqual({run["arrival_seed"] for run in runs}, {9901})
        self.assertEqual({run["max_active_sessions"] for run in runs}, {1})
        self.assertEqual({run["available_sessions"] for run in runs}, {8})
        self.assertEqual({run["expected_request_count"] for run in runs}, {24})
        oracle = next(run for run in runs if run["strict_oracle"])
        control = next(run for run in runs if not run["strict_oracle"])
        self.assertEqual(oracle["policy"], "infinite_hbm_oracle")
        self.assertEqual(control["policy"], "hbm_cpu_ssd_control")
        self.assertIn("--strict-infinite-hbm-oracle", oracle["argv"])
        self.assertNotIn("--strict-infinite-hbm-oracle", control["argv"])

    def test_parallelism_smoke_is_an_eight_cell_debug_only_pilot(self):
        repo_root = Path(__file__).resolve().parents[1]
        path = (
            repo_root
            / "configs/experiments"
            / "online_tracelab_qwen3_1m_p4d4_parallelism_smoke.json"
        )
        spec = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(set(spec["policies"]), {"hbm_cpu_ssd_control"})
        self.assertEqual(spec["oracle_label"], "infinite_hbm_oracle")
        self.assertEqual(
            spec["workload_selection"]["include_source_indices"],
            [487, 488, 1759, 1836, 1902, 2021, 2047, 3791],
        )
        self.assertEqual(
            spec["dataset_contract"]["expected_selected_template_count"],
            8,
        )
        self.assertEqual(
            spec["dataset_contract"]["expected_selected_request_count"],
            24,
        )
        self.assertEqual(
            spec["dataset_contract"][
                "expected_selected_session_identity_hash"],
            "4dded4375d266cda87c46a7e9c10633e1f7f60ff88c431d10d65c7c67677be58",
        )

        pilot = spec["host_concurrency_pilot_contract"]
        self.assertEqual(pilot["role"], "debug-only-host-concurrency-pilot")
        self.assertEqual(pilot["comparison_max_parallel"], [4, 8])
        self.assertEqual(pilot["default_max_parallel"], 4)
        self.assertEqual(pilot["cli_override"], "--max-parallel 8")
        self.assertEqual(pilot["expected_cell_count"], 8)
        self.assertFalse(pilot["is_slo_calibration"])
        self.assertFalse(pilot["main_result_eligible"])
        self.assertFalse(pilot["paper_result_eligible"])
        self.assertTrue(pilot["do_not_merge_with_main_results"])
        self.assertNotIn("slo_calibration_contract", spec)
        self.assertNotIn("plots", spec)
        self.assertEqual(spec["max_parallel"], 4)

        mode = spec["modes"]["poisson"]
        self.assertEqual(mode["rates_sps"], [0.001])
        self.assertEqual(mode["session_repetitions"], 1)
        self.assertEqual(mode["max_active_sessions"], 1)
        self.assertEqual(mode["arrival_seeds"], [9901, 9902, 9903, 9904])
        self.assertEqual(mode["measure_completions"], "all")
        self.assertTrue(mode["require_complete_session_cohort"])

        session_rows = [
            {
                "session_id": f"session-{session_index}",
                "arrival_time_ns": 0,
                "sub_requests": [
                    {
                        "input_toks": 100 + request_index,
                        "output_toks": 1,
                        "tool_duration_ns": 1,
                    }
                    for request_index in range(3)
                ],
            }
            for session_index in range(8)
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "sessions.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in session_rows),
                encoding="utf-8",
            )
            cohort = materialize_session_cohort(
                source, directory / "cohort")
            runs = build_run_descriptors(
                spec, repo_root, directory / "runs", cohort)

        self.assertEqual(len(runs), 8)
        self.assertEqual(
            {run["policy"] for run in runs},
            {"hbm_cpu_ssd_control", "infinite_hbm_oracle"},
        )
        self.assertEqual(
            {run["arrival_seed"] for run in runs},
            {9901, 9902, 9903, 9904},
        )
        pair_groups = {}
        for run in runs:
            pair_groups.setdefault(run["pair_key"], []).append(run)
        self.assertEqual(len(pair_groups), 4)
        self.assertEqual({len(pair) for pair in pair_groups.values()}, {2})
        for pair in pair_groups.values():
            self.assertEqual(
                {run["policy"] for run in pair},
                {"hbm_cpu_ssd_control", "infinite_hbm_oracle"},
            )
            self.assertEqual(
                len({run["workload_sha256"] for run in pair}),
                1,
            )

    def test_high_rate_smoke_is_a_two_cell_debug_only_pair(self):
        repo_root = Path(__file__).resolve().parents[1]
        path = (
            repo_root
            / "configs/experiments"
            / "online_tracelab_qwen3_1m_p4d4_high_rate_smoke.json"
        )
        spec = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(set(spec["policies"]), {"hbm_cpu_ssd_control"})
        self.assertEqual(spec["oracle_label"], "infinite_hbm_oracle")
        self.assertEqual(
            spec["workload_selection"]["include_source_indices"],
            [2113, 3726],
        )
        self.assertEqual(
            spec["workload_selection"]["target_max_sequence_tokens"],
            1_000_000,
        )
        self.assertEqual(
            spec["dataset_contract"]["expected_selected_template_count"],
            2,
        )
        self.assertEqual(
            spec["dataset_contract"]["expected_selected_request_count"],
            4,
        )

        debug = spec["debug_contract"]
        self.assertEqual(debug["role"], "post-fix-high-rate-liveness-smoke")
        self.assertEqual(debug["expected_cell_count"], 2)
        self.assertFalse(debug["is_slo_calibration"])
        self.assertFalse(debug["main_result_eligible"])
        self.assertFalse(debug["paper_result_eligible"])
        self.assertTrue(debug["do_not_merge_with_main_results"])

        mode = spec["modes"]["poisson"]
        self.assertEqual(mode["rates_sps"], [0.030375])
        self.assertEqual(mode["session_repetitions"], 4)
        self.assertEqual(mode["max_active_sessions"], 20)
        self.assertEqual(mode["arrival_seeds"], [17])
        self.assertEqual(mode["measure_completions"], "all")
        self.assertFalse(mode["stop_after_measurement"])
        self.assertTrue(mode["require_complete_session_cohort"])
        self.assertNotIn("plots", spec)
        self.assertEqual(spec["max_parallel"], 2)

        session_rows = [
            {
                "session_id": f"session-{session_index}",
                "arrival_time_ns": 0,
                "sub_requests": [
                    {
                        "input_toks": 100 + request_index,
                        "output_toks": 1,
                        "tool_duration_ns": 1,
                    }
                    for request_index in range(2)
                ],
            }
            for session_index in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "sessions.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in session_rows),
                encoding="utf-8",
            )
            cohort = materialize_session_cohort(
                source, directory / "cohort")
            runs = build_run_descriptors(
                spec, repo_root, directory / "runs", cohort)

        self.assertEqual(len(runs), 2)
        self.assertEqual(
            {run["policy"] for run in runs},
            {"hbm_cpu_ssd_control", "infinite_hbm_oracle"},
        )
        self.assertEqual({run["arrival_seed"] for run in runs}, {17})
        self.assertEqual({run["max_active_sessions"] for run in runs}, {20})
        self.assertEqual({run["available_sessions"] for run in runs}, {8})
        self.assertEqual({run["expected_request_count"] for run in runs}, {16})
        self.assertEqual(len({run["pair_key"] for run in runs}), 1)
        self.assertEqual(len({run["workload_sha256"] for run in runs}), 1)


if __name__ == "__main__":
    unittest.main()

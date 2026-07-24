import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from serving.core.hbf_comparison_cell import (
    ASTRA_CYCLES_USED,
    CELL_SCHEMA_VERSION,
    SIMULATION_BACKEND,
)
from serving.core.hbf_comparison_workload import stable_json_sha256
from serving.hbf_comparison_plots import (
    FONT_SIZE,
    LATENCY_FIGSIZE,
    MATPLOTLIB_RC,
    THROUGHPUT_FIGSIZE,
    ComparisonPlotInputError,
    aggregate_validated_sweep,
    load_validated_sweep,
    write_aggregate_artifacts,
)
from serving.hbf_comparison_sweep import (
    COMPLETION_MARKER_SCHEMA_VERSION,
    SWEEP_SCHEMA_VERSION,
)


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_record(repo_root, relative):
    path = repo_root / relative
    return {
        "repo_relative_path": relative.as_posix(),
        "content_sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _artifact_record(path):
    return {
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class SyntheticSweep:
    def __init__(self, root, repo_root):
        self.root = Path(root)
        self.repo_root = Path(repo_root)
        self.trace = self.root / "trace.jsonl"
        self.trace.write_text('{"synthetic":true}\n', encoding="utf-8")
        self.rate = 1.0
        self.rate_text = "1"
        self.seeds = (101, 211, 307)
        self.system = "oracle"
        self.thresholds = {
            "first_ttft_ns": 30_000_000_000,
            "resume_ttft_ns": 30_000_000_000,
            "tpot_ns": 300_000_000,
        }
        self.measurement_sha = stable_json_sha256(
            ["session:0", "session:1"])
        self.scenario_manifest = {
            "scenario_id": "synthetic-balanced-v1",
            "measurement_request_identities_sha256": self.measurement_sha,
        }
        self.scenario_manifest_sha = stable_json_sha256(
            self.scenario_manifest)
        model_relative = Path(
            "configs/model/Qwen/Qwen3-30B-A3B-Instruct-2507.json")
        gpu_relative = Path("configs/wakekv_hbf/p4d4_gpu_server.json")
        self.system_config = {
            "system_key": self.system,
            "system_class": "dual_strict_infinite_hbm_oracle",
            "tiering_policy": None,
            "hbf_layout": None,
            "model_config": _file_record(
                self.repo_root, model_relative),
            "gpu_config": _file_record(
                self.repo_root, gpu_relative),
            "hbf_config": None,
        }
        self.configs = {self.system: self.system_config}
        self.code = self._code_contract()
        self.pairs = []
        self.records = []
        for offset, seed in enumerate(self.seeds):
            pair = self._pair(seed)
            self.pairs.append(pair)
            self.records.append(
                self._cell(seed, pair, first_ttft_seconds=offset + 1.0))
        self.manifest = self._manifest()
        _write_json(self.root / "manifest.json", self.manifest)

    def _code_contract(self):
        relative = Path("serving/core/hbf_comparison_metrics.py")
        source_hashes = {
            relative.as_posix(): _sha256_file(self.repo_root / relative),
        }
        payload = {
            "python_implementation": sys.implementation.name,
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "serving_python_files": source_hashes,
        }
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_root,
            text=True,
        ).strip()
        return {
            "repository_git_head": head,
            "astra_sim_git_head": None,
            **payload,
            "serving_python_tree_sha256": stable_json_sha256(source_hashes),
            "execution_code_sha256": stable_json_sha256(payload),
        }

    def _pair(self, seed):
        identities = [f"seed-{seed}:0", f"seed-{seed}:1"]
        identity_hash = stable_json_sha256(sorted(identities))
        base = {
            "scenario_id": "synthetic-balanced-v1",
            "seed": seed,
            "session_rate": self.rate,
            "rate_text": self.rate_text,
            "offered_session_ids_sha256": stable_json_sha256(
                [f"seed-{seed}"]),
            "unit_draws_sha256": stable_json_sha256([seed, 0.5]),
            "session_count": 1,
            "call_count": 2,
            "call_specs_sha256": stable_json_sha256(
                {"seed": seed, "kind": "calls"}),
            "schedule_sha256": stable_json_sha256(
                {"seed": seed, "kind": "schedule"}),
            "expected_call_identities_sha256": identity_hash,
            "expected_call_identity_set_sha256": identity_hash,
            "expected_session_ids_sha256": stable_json_sha256(
                [f"seed-{seed}"]),
        }
        return {
            **base,
            "schedule_pair_sha256": stable_json_sha256(base),
            "system_keys": [self.system],
            "result_schedule_sha256": base["schedule_sha256"],
            "result_call_specs_sha256": base["call_specs_sha256"],
            "measurement_identities_sha256": self.measurement_sha,
            "completion_call_set_sha256": identity_hash,
        }

    @staticmethod
    def _distribution(value, count):
        return {
            "count": count,
            "mean_ns": value,
            "p50_ns": value,
            "p90_ns": value,
            "p95_ns": value,
            "p99_ns": value,
            "percentile_method": "inclusive_nearest_rank",
        }

    def _cell(self, seed, pair, first_ttft_seconds):
        identities = [f"seed-{seed}:0", f"seed-{seed}:1"]
        completion_order_hash = stable_json_sha256(identities)
        first_ns = first_ttft_seconds * 1_000_000_000
        resume_ns = 2_000_000_000.0
        tpot_ns = 100_000_000.0
        summary = {
            "counts": {
                "measurement_sessions": 1,
                "measurement_calls": 2,
                "first_calls": 1,
                "resume_calls": 1,
                "tpot_eligible_calls": 2,
                "output_tokens": 30,
            },
            "latency_distributions_ns": {
                "first_ttft": self._distribution(first_ns, 1),
                "resume_ttft": self._distribution(resume_ns, 1),
                "tpot_eligible": self._distribution(tpot_ns, 2),
            },
            "request_kind_summaries": {},
            "slo": {
                "thresholds_ns": self.thresholds,
                "first_ttft_pass_count": 1,
                "first_ttft_pass_fraction": 1.0,
                "resume_ttft_pass_count": 1,
                "resume_ttft_pass_fraction": 1.0,
                "ttft_pass_count": 2,
                "ttft_pass_fraction": 1.0,
                "tpot_pass_count": 2,
                "tpot_pass_fraction_of_eligible": 1.0,
                "all_slo_pass_count": 2,
                "all_slo_pass_fraction": 1.0,
                "all_slo_pass_output_tokens": 30,
            },
            "offered_load_normalized_request_goodput": {
                "label": "offered-load-normalized request goodput",
                "unit": "requests/s",
                "value": 2.0,
                "formula": (
                    "session_rate * measured_calls / measured_sessions "
                    "* all_SLO_pass_fraction"
                ),
            },
            "offered_load_normalized_output_token_goodput": {
                "label": "offered-load-normalized output-token goodput",
                "unit": "output tokens/s",
                "value": 30.0,
                "formula": (
                    "session_rate * all_SLO_pass_output_tokens "
                    "/ measured_sessions"
                ),
            },
            "observed_completion_span_throughput": {
                "label": "observed inter-completion rate",
                "semantics": "synthetic fixture",
                "completion_start_ns": 1_000_000_000,
                "completion_end_ns": 2_000_000_000,
                "completion_span_ns": 1_000_000_000,
                "completion_event_count": 2,
                "inter_completion_interval_count": 1,
                "interval_output_tokens": 20,
                "requests_per_second": 1.0,
                "output_tokens_per_second": 20.0,
                "zero_span_value": None,
            },
        }
        cell = {
            "schema_version": CELL_SCHEMA_VERSION,
            "system_key": self.system,
            "session_rate": self.rate,
            "simulation_contract": {
                "system_key": self.system,
                "execution_backend": {
                    "name": SIMULATION_BACKEND,
                    "astra_cycles_used": ASTRA_CYCLES_USED,
                },
                "hardware": {
                    "gpu": self.system_config["gpu_config"],
                    "hbf": None,
                },
            },
            "frozen_workload": {
                "session_count": 1,
                "call_count": 2,
                "call_specs_sha256": pair["call_specs_sha256"],
                "schedule_sha256": pair["schedule_sha256"],
                "expected_call_identities_sha256": (
                    pair["expected_call_identities_sha256"]),
            },
            "measurement_roster": {
                "ordered_identities_sha256": self.measurement_sha,
            },
            "summary": summary,
            "full_drain": {
                "calls": {
                    "identity_count": 2,
                    "expected_set_sha256": (
                        pair["expected_call_identity_set_sha256"]),
                    "completion_set_sha256": (
                        pair["expected_call_identity_set_sha256"]),
                    "completion_order_sha256": completion_order_hash,
                },
            },
            "bottleneck_report": {},
            "execution_observation": {},
            "requests": [
                {
                    "completion_identity": identity,
                    "system_key": self.system,
                }
                for identity in identities
            ],
        }
        directory = (
            self.root / "cells" / "rate_1"
            / f"seed_{seed}" / self.system
        )
        directory.mkdir(parents=True)
        cell_path = directory / "cell.json"
        csv_path = directory / "requests.csv"
        completion_path = directory / "completion.json"
        _write_json(cell_path, cell)
        with csv_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("completion_identity",))
            for identity in identities:
                writer.writerow((identity,))

        result_contract = {
            "cell_schema_version": CELL_SCHEMA_VERSION,
            "call_specs_sha256": pair["call_specs_sha256"],
            "schedule_sha256": pair["schedule_sha256"],
            "measurement_identities_sha256": self.measurement_sha,
            "completion_call_set_sha256": (
                pair["expected_call_identity_set_sha256"]),
            "completion_call_order_sha256": completion_order_hash,
            "request_count": 2,
            "simulation_backend": SIMULATION_BACKEND,
            "astra_cycles_used": ASTRA_CYCLES_USED,
        }
        cell_contract = {
            "schema_version": SWEEP_SCHEMA_VERSION,
            "scenario_id": "synthetic-balanced-v1",
            "scenario_manifest_sha256": self.scenario_manifest_sha,
            "seed": seed,
            "session_rate": self.rate,
            "rate_text": self.rate_text,
            "system_key": self.system,
            "schedule_pair_sha256": pair["schedule_pair_sha256"],
            "expected_call_specs_sha256": pair["call_specs_sha256"],
            "expected_schedule_sha256": pair["schedule_sha256"],
            "measurement_identities_sha256": self.measurement_sha,
            "system_config_contract_sha256": stable_json_sha256(
                self.system_config),
            "execution_code_sha256": self.code["execution_code_sha256"],
            "thresholds_ns": self.thresholds,
        }
        cell_contract_hash = stable_json_sha256(cell_contract)
        marker = {
            "schema_version": COMPLETION_MARKER_SCHEMA_VERSION,
            "status": "complete",
            "cell_contract": cell_contract,
            "cell_contract_sha256": cell_contract_hash,
            "result_contract": result_contract,
            "artifacts": {
                "cell.json": _artifact_record(cell_path),
                "requests.csv": _artifact_record(csv_path),
            },
        }
        _write_json(completion_path, marker)
        return {
            "seed": seed,
            "session_rate": self.rate,
            "rate_text": self.rate_text,
            "system_key": self.system,
            "relative_directory": directory.relative_to(
                self.root).as_posix(),
            "cell_contract_sha256": cell_contract_hash,
            "schedule_pair_sha256": pair["schedule_pair_sha256"],
            "result_contract": result_contract,
            "artifacts": {
                "cell.json": _artifact_record(cell_path),
                "requests.csv": _artifact_record(csv_path),
                "completion.json": _artifact_record(completion_path),
            },
        }

    def _manifest(self):
        rates = [self.rate_text]
        seeds = list(self.seeds)
        systems = [self.system]
        return {
            "schema_version": SWEEP_SCHEMA_VERSION,
            "scenario": {
                "scenario_id": "synthetic-balanced-v1",
                "source_path": str(self.trace),
                "source_sha256": _sha256_file(self.trace),
                "manifest": self.scenario_manifest,
                "manifest_sha256": self.scenario_manifest_sha,
            },
            "grid": {
                "rates": [self.rate],
                "rate_texts": rates,
                "rates_sha256": stable_json_sha256(rates),
                "seeds": seeds,
                "seeds_sha256": stable_json_sha256(seeds),
                "system_keys": systems,
                "system_keys_sha256": stable_json_sha256(systems),
                "cell_count": len(self.records),
            },
            "slo_thresholds_ns": self.thresholds,
            "execution": {
                "executor": "synthetic",
                "multiprocessing_start_method": "spawn",
                "max_tasks_per_child": 1,
                "one_isolated_cell_per_process": True,
                "workers": 1,
                "detected_physical_cores": 1,
                "simulation_backend": SIMULATION_BACKEND,
                "astra_cycles_used": ASTRA_CYCLES_USED,
            },
            "code_revision_hashes": self.code,
            "system_config_contracts": self.configs,
            "system_config_contracts_sha256": stable_json_sha256(
                self.configs),
            "pairing": {
                "semantics": "synthetic paired schedule",
                "measurement_identities_sha256": self.measurement_sha,
                "schedule_pairs": self.pairs,
                "schedule_pairs_sha256": stable_json_sha256(self.pairs),
            },
            "cells": self.records,
            "cells_sha256": stable_json_sha256(self.records),
        }


class HBFComparisonPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]

    def test_fail_closed_loader_and_student_t_seed_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticSweep(directory, self.repo_root)
            loaded = load_validated_sweep(
                fixture.root, repo_root=self.repo_root)
            aggregate = aggregate_validated_sweep(loaded)

        self.assertEqual(loaded.seeds, (101, 211, 307))
        self.assertEqual(len(aggregate.points), 1)
        first = aggregate.points[0].metrics[
            "first_ttft_p95_seconds"]
        self.assertEqual(set(first.values), {1.0, 2.0, 3.0})
        self.assertAlmostEqual(first.mean, 2.0)
        self.assertAlmostEqual(first.sample_stddev, 1.0)
        self.assertAlmostEqual(
            first.ci95_half_width, 4.303 / math.sqrt(3))
        self.assertEqual(first.ci_method, "student_t_95")

    def test_modified_cell_is_rejected_before_metric_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticSweep(directory, self.repo_root)
            cell_path = (
                fixture.root / fixture.records[0]["relative_directory"]
                / "cell.json"
            )
            cell = json.loads(cell_path.read_text(encoding="utf-8"))
            cell["summary"]["slo"]["all_slo_pass_fraction"] = 0.0
            _write_json(cell_path, cell)

            with self.assertRaisesRegex(
                    ComparisonPlotInputError, "artifact hash mismatch"):
                load_validated_sweep(
                    fixture.root, repo_root=self.repo_root)

    def test_statistics_artifacts_preserve_seed_values_and_plot_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture_root = Path(directory) / "input"
            fixture_root.mkdir()
            fixture = SyntheticSweep(fixture_root, self.repo_root)
            aggregate = aggregate_validated_sweep(load_validated_sweep(
                fixture.root, repo_root=self.repo_root))
            output = Path(directory) / "output"
            artifacts = write_aggregate_artifacts(
                aggregate, output, render=False)
            csv_text = Path(
                artifacts["statistics_csv"]["path"]
            ).read_text(encoding="utf-8")
            manifest = json.loads(
                (output / "plot_manifest.json").read_text(
                    encoding="utf-8")
            )

        self.assertIn("student_t_95", csv_text)
        self.assertIn("1;2;3", csv_text)
        self.assertEqual(manifest["figure_width_inches"], 12)
        self.assertEqual(manifest["font_size"], 24)
        self.assertEqual(LATENCY_FIGSIZE[0], 12)
        self.assertEqual(THROUGHPUT_FIGSIZE[0], 12)
        self.assertTrue(all(
            value == FONT_SIZE for value in MATPLOTLIB_RC.values()))


if __name__ == "__main__":
    unittest.main()

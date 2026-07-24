from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from serving.live_astra_comparison_sweep import (
    Cell,
    DEFAULT_LOG_INTERVAL_SECONDS,
    HBF_CONFIG,
    LiveAstraSweepError,
    ORACLE_CONFIG,
    SCHEMA_VERSION,
    SINGLE_CLUSTER,
    SYSTEMS,
    SystemSpec,
    _artifact,
    _campaign_identity,
    _campaign_implementation_identity,
    _extract_bottlenecks,
    _is_resumable_completion,
    _load_or_initialize_manifest,
    _parser,
    _run_cell,
    _runtime_guard_contract,
    _safe_remove_cell_inputs,
    _scenario_contract,
    build_serving_command,
    load_scenario,
)


@dataclass(frozen=True)
class _FakeMetrics:
    operational_request_goodput_per_second: float = 0.1


class LiveAstraComparisonSweepTest(unittest.TestCase):
    def _cell(self, root: Path, system_key: str) -> Cell:
        system = SYSTEMS[system_key]
        return Cell(
            cell_id=f"seed101-rate0p1-{system_key}",
            system=system,
            seed=101,
            rate=0.1,
            workload_path=root / "workload.jsonl",
            workload_sha256="a" * 64,
            cell_dir=root / "cells" / system_key,
            inputs_dir=root / "shm" / system_key / "inputs",
            request_count=648,
            session_count=216,
        )

    def _write_resumable_result(
        self,
        cell: Cell,
        entry: dict,
    ) -> None:
        entry["log_interval_seconds"] = DEFAULT_LOG_INTERVAL_SECONDS
        cell.cell_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths = {
            "requests": cell.requests_csv,
            "session_report": cell.session_report,
            "runtime_report": cell.runtime_report,
            "stdout": cell.cell_dir / "stdout.log",
            "stderr": cell.cell_dir / "stderr.log",
        }
        for name, path in artifact_paths.items():
            path.write_bytes(f"{name}\n".encode())
        cell.result_path.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "cell_id": cell.cell_id,
            "system": cell.system.key,
            "seed": cell.seed,
            "offered_session_rate_per_second": cell.rate,
            "log_interval_seconds": DEFAULT_LOG_INTERVAL_SECONDS,
            "workload": {
                "sha256": cell.workload_sha256,
                "request_count": cell.request_count,
                "session_count": cell.session_count,
            },
            "metrics": {
                "operational_request_goodput_per_second": 0.1,
            },
            "artifacts": {
                name: _artifact(path)
                for name, path in artifact_paths.items()
            },
        }))
        result_artifact = _artifact(cell.result_path)
        entry["status"] = "completed"
        entry["result_sha256"] = result_artifact["sha256"]
        entry["result_bytes"] = result_artifact["bytes"]

    def test_builds_live_baseline_oracle_and_hbf_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = Path("/repo")
            baseline = self._cell(root, "ssd_tiering")
            baseline_command = build_serving_command(
                repo_root=repo,
                python_executable=Path("/venv/bin/python"),
                cell=baseline,
            )
            self.assertEqual(
                baseline_command[:3],
                ("/venv/bin/python", "-m", "serving"),
            )
            self.assertIn("--agentic-kv-config", baseline_command)
            self.assertIn("--agentic-kv-metrics", baseline_command)
            self.assertNotIn("--strict-infinite-hbm-oracle", baseline_command)
            self.assertNotIn("--full-model-hbf-config", baseline_command)
            self.assertEqual(
                baseline_command[
                    baseline_command.index("--request-routing-policy") + 1
                ],
                "RR",
            )
            self.assertEqual(
                baseline_command[
                    baseline_command.index("--network-backend") + 1
                ],
                "analytical-congestion-aware",
            )
            self.assertEqual(
                baseline_command[
                    baseline_command.index("--log-interval") + 1
                ],
                "60.0",
            )

            oracle_command = build_serving_command(
                repo_root=repo,
                python_executable=Path("/venv/bin/python"),
                cell=self._cell(root, "oracle"),
            )
            self.assertIn("--strict-infinite-hbm-oracle", oracle_command)
            self.assertTrue(
                str(repo / ORACLE_CONFIG) in oracle_command)

            for key, layout in (
                ("hbf_tp4", "tp4"),
                ("hbf_tp8", "tp8"),
                ("hbf_tp8_context", "tp8_context"),
            ):
                command = build_serving_command(
                    repo_root=repo,
                    python_executable=Path("/venv/bin/python"),
                    cell=self._cell(root, key),
                )
                self.assertIn("--full-model-hbf-config", command)
                self.assertIn("--full-model-hbf-metrics", command)
                self.assertNotIn("--agentic-kv-config", command)
                self.assertEqual(
                    command[
                        command.index("--full-model-hbf-layout") + 1
                    ],
                    layout,
                )
                self.assertTrue(str(repo / HBF_CONFIG) in command)
                self.assertTrue(str(repo / SINGLE_CLUSTER) in command)

    def test_log_interval_only_changes_progress_log_command_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell = self._cell(root, "ssd_tiering")
            default_command = build_serving_command(
                repo_root=Path("/repo"),
                python_executable=Path("/python"),
                cell=cell,
            )
            custom_command = build_serving_command(
                repo_root=Path("/repo"),
                python_executable=Path("/python"),
                cell=cell,
                log_interval_seconds=17.5,
            )
            index = default_command.index("--log-interval") + 1
            self.assertEqual(custom_command.index("--log-interval") + 1, index)
            self.assertEqual(default_command[index], "60.0")
            self.assertEqual(custom_command[index], "17.5")
            self.assertEqual(
                default_command[:index] + default_command[index + 1:],
                custom_command[:index] + custom_command[index + 1:],
            )

    def test_log_interval_cli_default_and_validation(self):
        parser = _parser()
        self.assertEqual(
            parser.parse_args([]).log_interval,
            DEFAULT_LOG_INTERVAL_SECONDS,
        )
        self.assertEqual(
            parser.parse_args(["--log-interval", "17.5"]).log_interval,
            17.5,
        )
        for value in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(value=value):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--log-interval", value])

    def test_manifest_resumes_only_matching_completed_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell = self._cell(root, "ssd_tiering")
            manifest_path = root / "manifest.json"
            identity = {
                "campaign": "test",
                "log_interval_seconds": DEFAULT_LOG_INTERVAL_SECONDS,
            }
            manifest = _load_or_initialize_manifest(
                manifest_path,
                identity=identity,
                cells=(cell,),
            )
            entry = manifest["cells"][cell.cell_id]
            self.assertEqual(entry["status"], "pending")
            self.assertEqual(entry["workload_sha256"], "a" * 64)
            self.assertEqual(entry["request_count"], 648)
            self.assertEqual(
                entry["log_interval_seconds"],
                DEFAULT_LOG_INTERVAL_SECONDS,
            )
            with self.assertRaisesRegex(
                    LiveAstraSweepError, "different campaign"):
                _load_or_initialize_manifest(
                    manifest_path,
                    identity={
                        **identity,
                        "log_interval_seconds": 17.5,
                    },
                    cells=(cell,),
                )

            self._write_resumable_result(cell, entry)
            self.assertTrue(_is_resumable_completion(cell, entry))
            entry["log_interval_seconds"] = 17.5
            self.assertFalse(_is_resumable_completion(cell, entry))
            entry["log_interval_seconds"] = DEFAULT_LOG_INTERVAL_SECONDS

            mismatched = Cell(
                **{
                    **cell.__dict__,
                    "workload_sha256": "b" * 64,
                }
            )
            with self.assertRaisesRegex(
                    LiveAstraSweepError, "schedule changed"):
                _load_or_initialize_manifest(
                    manifest_path,
                    identity=identity,
                    cells=(mismatched,),
                )

    def test_resume_rejects_missing_or_tampered_artifacts_and_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell = self._cell(root, "ssd_tiering")
            entry = {
                "status": "pending",
                "result": str(cell.result_path),
            }
            self._write_resumable_result(cell, entry)
            self.assertTrue(_is_resumable_completion(cell, entry))

            original_requests = cell.requests_csv.read_bytes()
            cell.requests_csv.write_bytes(
                b"X" + original_requests[1:])
            self.assertFalse(_is_resumable_completion(cell, entry))
            cell.requests_csv.write_bytes(original_requests)
            self.assertTrue(_is_resumable_completion(cell, entry))

            result = json.loads(cell.result_path.read_text())
            result["artifacts"]["requests"]["bytes"] += 1
            cell.result_path.write_text(json.dumps(result))
            result_artifact = _artifact(cell.result_path)
            entry["result_sha256"] = result_artifact["sha256"]
            entry["result_bytes"] = result_artifact["bytes"]
            self.assertFalse(_is_resumable_completion(cell, entry))
            self._write_resumable_result(cell, entry)

            cell.runtime_report.unlink()
            self.assertFalse(_is_resumable_completion(cell, entry))
            cell.runtime_report.write_bytes(b"runtime_report\n")
            self.assertTrue(_is_resumable_completion(cell, entry))

            cell.result_path.write_text(
                cell.result_path.read_text() + "\n")
            self.assertFalse(_is_resumable_completion(cell, entry))

    def test_custom_factory_file_exposes_generic_scenario_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            factory = root / "scenario_factory.py"
            factory.write_text(
                "from types import SimpleNamespace\n"
                "class Scenario:\n"
                "    manifest = SimpleNamespace(\n"
                "        scenario_id='pressure-balanced-v1',\n"
                "        source_sha256='1' * 64,\n"
                "        measurement_session_ids=('measured-a',),\n"
                "    )\n"
                "    def build_offered_plan(self, *, seed):\n"
                "        return ('plan', seed)\n"
                "def build(trace_path):\n"
                "    return Scenario()\n"
            )
            scenario = load_scenario(
                root / "trace.jsonl",
                f"{factory}:build",
            )
            self.assertEqual(
                _scenario_contract(scenario),
                (
                    "pressure-balanced-v1",
                    "1" * 64,
                    ("measured-a",),
                ),
            )

    def test_campaign_implementation_identity_is_bounded_and_content_pinned(
            self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            serving = root / "serving"
            (serving / "core").mkdir(parents=True)
            (serving / "entry.py").write_text("ENTRY = 1\n")
            (serving / "core" / "engine.py").write_text("ENGINE = 1\n")
            (serving / "ignored.txt").write_text("not Python\n")
            (root / "unrelated.py").write_text("UNRELATED = 1\n")
            converter = (
                root
                / "astra-sim/extern/graph_frontend/chakra/src/"
                "converter/llm_converter.py"
            )
            converter.parent.mkdir(parents=True)
            converter.write_text("CONVERTER = 1\n")
            binary = (
                root
                / "astra-sim/build/astra_analytical/build/"
                "AstraCongestion/bin/AstraCongestion"
            )
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"astra-binary-v1")

            original = _campaign_implementation_identity(root)
            trace = root / "trace.jsonl"
            trace.write_text("{}\n")
            cluster = root / "cluster.json"
            policy = root / "policy.json"
            cluster.write_text("{}\n")
            policy.write_text("{}\n")
            scenario = SimpleNamespace(
                manifest=SimpleNamespace(
                    scenario_id="identity-test",
                    source_sha256="1" * 64,
                    measurement_session_ids=("measured",),
                ),
                build_offered_plan=lambda seed: None,
            )
            campaign = _campaign_identity(
                repo_root=root,
                trace_path=trace,
                scenario=scenario,
                scenario_factory="test:build",
                specs=(SystemSpec(
                    key="test",
                    cluster_config=Path("cluster.json"),
                    runtime_kind="oracle",
                    policy_config=Path("policy.json"),
                ),),
                rates=(0.1,),
                seeds=(101,),
                ttft_slo_ns=1,
                tpot_slo_ns=1,
                log_interval_seconds=17.5,
            )
            self.assertEqual(
                campaign["simulator_implementation"],
                original,
            )
            self.assertEqual(campaign["log_interval_seconds"], 17.5)
            self.assertEqual(
                list(original["source_files"]),
                [
                    "astra-sim/extern/graph_frontend/chakra/src/"
                    "converter/llm_converter.py",
                    "serving/core/engine.py",
                    "serving/entry.py",
                ],
            )
            self.assertEqual(
                original["astra_binary"]["sha256"],
                hashlib.sha256(b"astra-binary-v1").hexdigest(),
            )

            (root / "unrelated.py").write_text("UNRELATED = 2\n")
            (serving / "ignored.txt").write_text("still ignored\n")
            self.assertEqual(
                _campaign_implementation_identity(root),
                original,
            )

            (serving / "core" / "engine.py").write_text("ENGINE = 2\n")
            source_changed = _campaign_implementation_identity(root)
            self.assertNotEqual(source_changed, original)

            (serving / "core" / "engine.py").write_text("ENGINE = 1\n")
            binary.write_bytes(b"astra-binary-v2")
            binary_changed = _campaign_implementation_identity(root)
            self.assertNotEqual(binary_changed, original)

    def test_scenario_contract_rejects_missing_measurement_roster(self):
        scenario = SimpleNamespace(
            manifest=SimpleNamespace(
                scenario_id="bad",
                source_sha256="1" * 64,
                measurement_session_ids=(),
            ),
            build_offered_plan=lambda seed: None,
        )
        with self.assertRaisesRegex(
                LiveAstraSweepError, "measurement_session_ids"):
            _scenario_contract(scenario)

    def test_runtime_guard_is_bound_to_final_scheduled_offer(self):
        scenario = SimpleNamespace(
            manifest=SimpleNamespace(
                runtime_guard_validation_required=True,
                runtime_guard_expected_measurement_resume_count=112,
            ),
            runtime_guard_contract=lambda **_: {
                "seed": 101,
                "offered_session_rate_per_second": 1.4,
                "last_external_guard_offer_ns": 9_999,
                "expected_measurement_resume_count": 112,
            },
        )
        schedule = (
            SimpleNamespace(arrival_time_ns=0),
            SimpleNamespace(arrival_time_ns=9_999),
        )
        contract = _runtime_guard_contract(
            scenario, seed=101, rate=1.4, schedule=schedule)
        self.assertEqual(
            contract["last_external_guard_offer_ns"], 9_999)
        with self.assertRaisesRegex(
                LiveAstraSweepError, "final scheduled external offer"):
            _runtime_guard_contract(
                scenario,
                seed=101,
                rate=1.4,
                schedule=(
                    *schedule[:-1],
                    SimpleNamespace(arrival_time_ns=10_000),
                ),
            )

    def test_bottleneck_projection_keeps_relevant_scalars(self):
        projected = _extract_bottlenecks({
            "runtime": {
                "ssd": {"read_bytes": 10},
                "latency": {"mean_ns": 20},
                "queue_stall_ns": 30,
                "hbf": {"capacity_bytes": 40},
            },
        })
        self.assertEqual(projected["runtime.ssd.read_bytes"], 10)
        self.assertEqual(projected["runtime.queue_stall_ns"], 30)
        self.assertEqual(
            projected["runtime.hbf.capacity_bytes"], 40)
        self.assertNotIn("runtime.latency.mean_ns", projected)

    def test_inputs_cleanup_is_scoped_to_one_cell(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            inputs = root / "cell-a" / "inputs"
            inputs.mkdir(parents=True)
            (inputs / "trace.et").write_bytes(b"trace")
            sibling = root / "cell-b" / "inputs"
            sibling.mkdir(parents=True)
            _safe_remove_cell_inputs(inputs, root)
            self.assertFalse((root / "cell-a").exists())
            self.assertTrue(sibling.exists())
            with self.assertRaisesRegex(
                    LiveAstraSweepError, "unsafe"):
                _safe_remove_cell_inputs(root, root)

    def test_postprocess_validation_failure_obeys_input_retention(self):
        for keep_failed_inputs in (False, True):
            with self.subTest(keep_failed_inputs=keep_failed_inputs):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    cell = self._cell(root, "ssd_tiering")
                    cell.inputs_dir.mkdir(parents=True)
                    (cell.inputs_dir / "trace.et").write_bytes(b"trace")
                    with (
                        patch(
                            "serving.live_astra_comparison_sweep."
                            "subprocess.run",
                            return_value=SimpleNamespace(returncode=0),
                        ),
                        patch(
                            "serving.live_astra_comparison_sweep."
                            "expected_request_identities",
                            return_value=(),
                        ),
                        patch(
                            "serving.live_astra_comparison_sweep."
                            "parse_serving_requests_csv",
                            side_effect=LiveAstraSweepError(
                                "invalid request artifact"),
                        ),
                    ):
                        with self.assertRaisesRegex(
                                LiveAstraSweepError,
                                "invalid request artifact"):
                            _run_cell(
                                repo_root=root,
                                python_executable=Path("/python"),
                                cell=cell,
                                scheduled_sessions=(),
                                measurement_session_ids=(),
                                ttft_slo_ns=1,
                                tpot_slo_ns=1,
                                campaign_inputs=root / "shm",
                                timeout_seconds=None,
                                keep_failed_inputs=keep_failed_inputs,
                                log_interval_seconds=(
                                    DEFAULT_LOG_INTERVAL_SECONDS),
                            )
                    self.assertEqual(
                        cell.inputs_dir.parent.exists(),
                        keep_failed_inputs,
                    )

    def test_cell_command_and_result_pin_log_interval_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell = self._cell(root, "ssd_tiering")
            cell.inputs_dir.mkdir(parents=True)
            cell.cell_dir.mkdir(parents=True, exist_ok=True)
            cell.requests_csv.write_text("requests\n")
            cell.session_report.write_text("{}")
            cell.runtime_report.write_text("{}")
            with (
                patch(
                    "serving.live_astra_comparison_sweep.subprocess.run",
                    return_value=SimpleNamespace(returncode=0),
                ),
                patch(
                    "serving.live_astra_comparison_sweep."
                    "expected_request_identities",
                    return_value=(),
                ),
                patch(
                    "serving.live_astra_comparison_sweep."
                    "parse_serving_requests_csv",
                    return_value=(),
                ),
                patch(
                    "serving.live_astra_comparison_sweep."
                    "compute_live_comparison_metrics",
                    return_value=_FakeMetrics(),
                ),
            ):
                result = _run_cell(
                    repo_root=root,
                    python_executable=Path("/python"),
                    cell=cell,
                    scheduled_sessions=(),
                    measurement_session_ids=(),
                    ttft_slo_ns=1,
                    tpot_slo_ns=1,
                    campaign_inputs=root / "shm",
                    timeout_seconds=None,
                    keep_failed_inputs=True,
                    log_interval_seconds=17.5,
                )

            command_record = json.loads(
                (cell.cell_dir / "command.json").read_text())
            command = command_record["command"]
            self.assertEqual(command_record["log_interval_seconds"], 17.5)
            self.assertEqual(
                command[command.index("--log-interval") + 1],
                "17.5",
            )
            self.assertEqual(result["log_interval_seconds"], 17.5)
            self.assertEqual(
                result["metrics"][
                    "operational_request_goodput_per_second"],
                0.1,
            )

    def test_postprocess_artifact_failure_obeys_input_retention(self):
        for keep_failed_inputs in (False, True):
            with self.subTest(keep_failed_inputs=keep_failed_inputs):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    cell = self._cell(root, "ssd_tiering")
                    cell.inputs_dir.mkdir(parents=True)
                    (cell.inputs_dir / "trace.et").write_bytes(b"trace")
                    cell.cell_dir.mkdir(parents=True)
                    cell.session_report.write_text("{}")
                    cell.runtime_report.write_text("{}")
                    with (
                        patch(
                            "serving.live_astra_comparison_sweep."
                            "subprocess.run",
                            return_value=SimpleNamespace(returncode=0),
                        ),
                        patch(
                            "serving.live_astra_comparison_sweep."
                            "expected_request_identities",
                            return_value=(),
                        ),
                        patch(
                            "serving.live_astra_comparison_sweep."
                            "parse_serving_requests_csv",
                            return_value=(),
                        ),
                        patch(
                            "serving.live_astra_comparison_sweep."
                            "compute_live_comparison_metrics",
                            return_value=_FakeMetrics(),
                        ),
                        patch(
                            "serving.live_astra_comparison_sweep._artifact",
                            side_effect=LiveAstraSweepError(
                                "artifact hashing failed"),
                        ),
                    ):
                        with self.assertRaisesRegex(
                                LiveAstraSweepError,
                                "artifact hashing failed"):
                            _run_cell(
                                repo_root=root,
                                python_executable=Path("/python"),
                                cell=cell,
                                scheduled_sessions=(),
                                measurement_session_ids=(),
                                ttft_slo_ns=1,
                                tpot_slo_ns=1,
                                campaign_inputs=root / "shm",
                                timeout_seconds=None,
                                keep_failed_inputs=keep_failed_inputs,
                                log_interval_seconds=(
                                    DEFAULT_LOG_INTERVAL_SECONDS),
                            )
                    self.assertEqual(
                        cell.inputs_dir.parent.exists(),
                        keep_failed_inputs,
                    )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from serving.core.hbf_comparison_workload import (
    CallSpec,
    ComparisonWorkload,
    SessionSpec,
    summarize_sessions,
)
from serving.core.hbf_comparison_cell import (
    ASTRA_CYCLES_USED,
    SIMULATION_BACKEND,
)
from serving.core.tracelab_comparison_scenarios import (
    build_balanced_causal_prefix_scenario,
    build_long_cold_context_stress_scenario,
)
from serving.hbf_comparison_sweep import (
    BALANCED_DEFAULT_RATES,
    COMPLETION_JSON,
    DEFAULT_SEEDS,
    ComparisonSweepError,
    _top_manifest,
    build_sweep_plan,
    default_trace_path,
    default_worker_count,
    describe_sweep_plan,
    run_sweep,
    validate_completed_cell,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _source_session() -> SessionSpec:
    session_id = "synthetic-source"
    calls = tuple(
        CallSpec(
            session_id=session_id,
            source_index=0,
            call_index=call_index,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_duration_ns=tool_duration_ns,
            cached_prefix_tokens=cached_prefix_tokens,
            fresh_input_tokens=input_tokens - cached_prefix_tokens,
            lineage_status=(
                "session_start" if call_index == 0 else "adjacent"),
            inter_turn_gap_type="tool",
        )
        for call_index, (
            input_tokens,
            output_tokens,
            tool_duration_ns,
            cached_prefix_tokens,
        ) in enumerate((
            (32, 2, 5_000_000, 0),
            (40, 2, 7_000_000, 30),
            (48, 1, 0, 38),
        ))
    )
    return SessionSpec(
        source_index=0,
        session_id=session_id,
        source_arrival_time_ns=0,
        source_session_identity_sha256=None,
        calls=calls,
    )


def _scenario(trace_path: Path):
    sessions = (_source_session(),)
    source = ComparisonWorkload(
        source_path=trace_path,
        source_sha256="a" * 64,
        source_session_count=1,
        sessions=sessions,
        summary=summarize_sessions(sessions),
    )
    return build_balanced_causal_prefix_scenario(
        source,
        epoch_count=3,
        warmup_epochs=(0,),
        measurement_epochs=(1,),
        guard_epochs=(2,),
        rates=(0.5, 1.0),
        maximum_rate=5.0,
        expected_base_session_count=1,
    )


def _long_cold_scenario(trace_path: Path):
    sessions = (_source_session(),)
    source = ComparisonWorkload(
        source_path=trace_path,
        source_sha256="b" * 64,
        source_session_count=1,
        sessions=sessions,
        summary=summarize_sessions(sessions),
    )
    return build_long_cold_context_stress_scenario(
        source,
        source_indices=(0,),
        cached_prefix_threshold=30,
        successor_call_count=1,
        epoch_count=3,
        warmup_epochs=(0,),
        measurement_epochs=(1,),
        guard_epochs=(2,),
        anchor_rates=(3.0, 5.0),
        maximum_rate=5.0,
    )


class HBFComparisonSweepTests(unittest.TestCase):

    def test_defaults_and_worker_headroom(self):
        self.assertEqual(
            BALANCED_DEFAULT_RATES,
            (0.5, 1.0, 2.0, 3.0, 4.0, 5.0),
        )
        self.assertEqual(len(DEFAULT_SEEDS), 12)
        self.assertEqual(
            default_trace_path({"LLMSIM_DATA": "/trace-root"}),
            Path("/trace-root/tracelab-schema3-sps0.2-final.jsonl"),
        )
        self.assertEqual(default_worker_count(1), 1)
        self.assertEqual(default_worker_count(8), 6)
        self.assertEqual(default_worker_count(32), 28)
        self.assertEqual(default_worker_count(86), 80)

    def test_plan_pairs_one_schedule_and_uses_deterministic_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "fixture.jsonl"
            plan = build_sweep_plan(
                repo_root=REPO_ROOT,
                trace_path=trace,
                output_root=root / "out",
                rates=(0.5, 1.0),
                seeds=(7, 11),
                system_keys=("oracle", "cpu_ssd"),
                workers=2,
                scenario=_scenario(trace),
            )
            self.assertEqual(len(plan.tasks), 8)
            self.assertEqual(len(plan.schedule_pairs), 4)
            self.assertEqual(plan.workers, 2)
            self.assertEqual(
                plan.system_keys, ("oracle", "cpu_ssd"))
            self.assertEqual(
                plan.tasks[0].output_dir.relative_to(
                    plan.output_root).as_posix(),
                "cells/rate_0p5/seed_7/oracle",
            )
            self.assertEqual(
                plan.tasks[1].output_dir.relative_to(
                    plan.output_root).as_posix(),
                "cells/rate_0p5/seed_7/cpu_ssd",
            )

            grouped = {}
            for task in plan.tasks:
                grouped.setdefault(
                    (task.seed, task.rate_text), []).append(task)
            self.assertEqual(len(grouped), 4)
            for tasks in grouped.values():
                self.assertEqual(
                    len({
                        task.schedule_pair_sha256
                        for task in tasks
                    }),
                    1,
                )
                self.assertIs(
                    tasks[0].scheduled_sessions,
                    tasks[1].scheduled_sessions,
                )
                self.assertEqual(
                    tasks[0].measurement_identities,
                    tuple(
                        plan.scenario.manifest
                        .measurement_request_identities),
                )

            description = describe_sweep_plan(plan)
            self.assertEqual(description["cell_count"], 8)
            self.assertEqual(description["pending_count"], 8)
            self.assertFalse(plan.output_root.exists())

    def test_plan_accepts_explicit_long_cold_sensitivity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "fixture.jsonl"
            scenario = _long_cold_scenario(trace)
            plan = build_sweep_plan(
                repo_root=REPO_ROOT,
                trace_path=trace,
                output_root=root / "out",
                rates=(3.0, 5.0),
                seeds=(7,),
                system_keys=("cpu_ssd", "hbf_tp4_wide"),
                workers=2,
                scenario=scenario,
            )
            self.assertEqual(
                plan.scenario.manifest.scenario_id,
                "tracelab-long-cold-30-cached-native-prefix-v1",
            )
            self.assertEqual(plan.rates, (3.0, 5.0))
            self.assertEqual(len(plan.tasks), 4)
            self.assertFalse(
                plan.scenario.manifest.equilibrium_workload)
            self.assertFalse(plan.output_root.exists())

    def test_actual_spawn_commit_resume_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "fixture.jsonl"
            plan = build_sweep_plan(
                repo_root=REPO_ROOT,
                trace_path=trace,
                output_root=root / "out",
                rates=(0.5,),
                seeds=(7,),
                system_keys=("oracle",),
                workers=1,
                scenario=_scenario(trace),
            )
            manifest, manifest_path = run_sweep(plan)
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(manifest["grid"]["cell_count"], 1)
            self.assertTrue(
                manifest["execution"]["one_isolated_cell_per_process"])
            self.assertEqual(
                manifest["execution"]["simulation_backend"],
                SIMULATION_BACKEND,
            )
            self.assertIs(
                manifest["execution"]["astra_cycles_used"],
                ASTRA_CYCLES_USED,
            )
            self.assertFalse(ASTRA_CYCLES_USED)
            task = plan.tasks[0]
            record = validate_completed_cell(task)
            self.assertEqual(
                set(record["artifacts"]),
                {"cell.json", "requests.csv", "completion.json"},
            )
            self.assertEqual(
                set(path.name for path in task.output_dir.iterdir()),
                {"cell.json", "requests.csv", "completion.json"},
            )

            resumed, resumed_path = run_sweep(
                plan, resume=True)
            self.assertEqual(resumed_path, manifest_path)
            self.assertEqual(resumed, manifest)

            marker_path = task.output_dir / COMPLETION_JSON
            marker = json.loads(marker_path.read_text(
                encoding="utf-8"))
            marker["artifacts"]["cell.json"]["sha256"] = "0" * 64
            marker_path.write_text(
                json.dumps(marker, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    ComparisonSweepError, "artifact hash mismatch"):
                validate_completed_cell(task)
            with self.assertRaises(ComparisonSweepError):
                run_sweep(plan, resume=True)

    def test_pairing_fails_closed_on_result_schedule_divergence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "fixture.jsonl"
            plan = build_sweep_plan(
                repo_root=REPO_ROOT,
                trace_path=trace,
                output_root=root / "out",
                rates=(0.5,),
                seeds=(7,),
                system_keys=("oracle", "cpu_ssd"),
                workers=1,
                scenario=_scenario(trace),
            )
            pair = plan.schedule_pairs[0]
            records = []
            for task in plan.tasks:
                records.append({
                    "seed": task.seed,
                    "session_rate": task.session_rate,
                    "rate_text": task.rate_text,
                    "system_key": task.system_key,
                    "relative_directory": (
                        task.output_dir.relative_to(
                            plan.output_root).as_posix()),
                    "cell_contract_sha256": (
                        task.cell_contract_sha256),
                    "schedule_pair_sha256": (
                        task.schedule_pair_sha256),
                    "result_contract": {
                        "schedule_sha256": pair["schedule_sha256"],
                        "call_specs_sha256": (
                            pair["call_specs_sha256"]),
                        "measurement_identities_sha256": (
                            plan.scenario.manifest
                            .measurement_request_identities_sha256),
                        "completion_call_set_sha256": (
                            pair[
                                "expected_call_identity_set_sha256"]),
                    },
                    "artifacts": {},
                })
            _top_manifest(plan, records)
            broken = list(records)
            broken[1] = {
                **broken[1],
                "result_contract": {
                    **broken[1]["result_contract"],
                    "schedule_sha256": "f" * 64,
                },
            }
            with self.assertRaisesRegex(
                    ComparisonSweepError,
                    "did not consume the exact same schedule"):
                _top_manifest(plan, broken)

    def test_invalid_grid_values_fail_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "fixture.jsonl"
            scenario = _scenario(trace)
            kwargs = {
                "repo_root": REPO_ROOT,
                "trace_path": trace,
                "output_root": root / "out",
                "scenario": scenario,
            }
            with self.assertRaises(ComparisonSweepError):
                build_sweep_plan(
                    **kwargs, rates=(0.5, 0.5))
            with self.assertRaises(ComparisonSweepError):
                build_sweep_plan(
                    **kwargs, seeds=(7, 7))
            with self.assertRaises(ComparisonSweepError):
                build_sweep_plan(
                    **kwargs, system_keys=("oracle", "missing"))
            with self.assertRaises(ComparisonSweepError):
                build_sweep_plan(
                    **kwargs, workers=0)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()

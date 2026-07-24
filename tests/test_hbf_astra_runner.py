import json
import os
from pathlib import Path
import random
import sys
import unittest

from serving.core.astra_operation_conformance import (
    build_hbf_media_microtrace,
)
from serving.core.hbf_astra_runner import (
    AstraCycleRecord,
    AstraHBFRunConfig,
    DEFAULT_ASTRA_BINARY,
    DEFAULT_CHAKRA_ROOT,
    HBFAstraRunnerError,
    HBFBackgroundJob,
    HBFTextTraceArtifact,
    PersistentHBFAstraRunner,
    _final_endpoint_cycles,
    _load_converter,
    _parse_cycle_records,
    resolve_run_config,
    run_hbf_trace_artifact,
    validate_hbf_trace_artifact,
)
from serving.core.hbf_full_model_astra import (
    HBFAstraTimingAccounting,
    HBFModelAstraStage,
    build_full_model_hbf_astra_projection,
    hbf_dependency_critical_path_ns,
    hbf_solo_named_resource_timing,
)
from serving.core.hbf_full_model_latency import (
    HBFModelBatchLatency,
    HBFParallelLayout,
    HBFServerHardware,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ASTRA_BINARY = Path(os.environ.get(
    "LLMSIM_ASTRA_BINARY",
    REPO_ROOT / DEFAULT_ASTRA_BINARY,
))
CHAKRA_ROOT = Path(os.environ.get(
    "LLMSIM_CHAKRA_ROOT",
    REPO_ROOT / DEFAULT_CHAKRA_ROOT,
))


def synthetic_full_model_latency() -> HBFModelBatchLatency:
    local = {
        "embedding_ns": 11,
        "dense_ns": 101,
        "attention_ns": 211,
        "router_ns": 17,
        "moe_ns": 307,
        "final_ns": 29,
    }
    collectives = (71, 53, 37)
    collective_ns = sum(collectives)
    return HBFModelBatchLatency(
        layout="tp4",
        tp_size=4,
        replicas=2,
        total_ns=sum(local.values()) + collective_ns,
        **local,
        collective_ns=collective_ns,
        tp_allreduce_ns=collectives[0],
        ep_allgather_ns=collectives[1],
        ep_reduce_scatter_ns=collectives[2],
        hbf_read_bytes_per_rank=1_000_003,
        lpddr_bytes_per_rank=200_007,
        collective_bytes_per_rank=10_003,
        compute_roof_ns_sum=1,
        hbf_roof_ns_sum=1,
        lpddr_roof_ns_sum=1,
        attention_compute_roof_ns=1,
        attention_hbf_roof_ns=1,
        attention_lpddr_roof_ns=1,
        attention_dominant_roof="hbf_read",
        dominant_kernel_counts={
            "compute": 1,
            "hbf_read": 1,
            "lpddr": 1,
        },
    )


class PureHBFAstraRunnerTests(unittest.TestCase):
    def test_signed_interference_accounting_is_strict_and_exact(self):
        accounting = HBFAstraTimingAccounting(
            dependency_critical_path_ns=26,
            solo_resource_serialized_completion_ns=27,
            actual_resource_serialized_completion_ns=26,
        )
        self.assertEqual(accounting.resource_delay_ns, 0)
        self.assertEqual(
            accounting.internal_resource_serialization_wait_ns, 1)
        self.assertEqual(accounting.signed_interference_delta_ns, -1)
        self.assertEqual(
            accounting.resource_delay_ns,
            (
                accounting.internal_resource_serialization_wait_ns
                + accounting.signed_interference_delta_ns
            ),
        )

        for invalid in (True, 26.0, float("inf"), -1):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                        ValueError, "finite non-negative integer"):
                    HBFAstraTimingAccounting(
                        dependency_critical_path_ns=invalid,
                        solo_resource_serialized_completion_ns=27,
                        actual_resource_serialized_completion_ns=26,
                    )
        with self.assertRaisesRegex(
                ValueError, "actual.*dependency"):
            HBFAstraTimingAccounting(
                dependency_critical_path_ns=26,
                solo_resource_serialized_completion_ns=27,
                actual_resource_serialized_completion_ns=25,
            )

    def test_existing_hbf_artifact_is_strictly_normalized(self):
        source = build_hbf_media_microtrace(
            operation="read",
            tp_size=8,
            runtime_ns=20_001,
            tensor_bytes_per_rank=65_536,
        )
        artifact, parsed, audit = validate_hbf_trace_artifact(source)
        self.assertEqual(artifact.text, source.text)
        self.assertEqual(artifact.num_npus, 8)
        self.assertEqual(parsed.model_parallel_groups, 1)
        self.assertEqual(audit.hbf_descriptor_count, 1)
        self.assertEqual(audit.hbf_stage_count, 1)
        self.assertEqual(audit.hbf_card_ids, (0,))
        self.assertEqual(
            audit.hbf_resource_names,
            tuple(f"hbf-card:{rank}:read" for rank in range(8)),
        )
        self.assertRegex(audit.trace_sha256, r"^[0-9a-f]{64}$")

    def test_validation_rejects_participant_mismatch_and_duplicate_gang(self):
        source = build_hbf_media_microtrace(
            operation="read",
            tp_size=8,
            runtime_ns=20_001,
            tensor_bytes_per_rank=65_536,
        )
        mismatched = source.text.replace(
            '"expected_participants":8',
            '"expected_participants":4',
        )
        with self.assertRaisesRegex(
                ValueError, "must equal the owning group size"):
            validate_hbf_trace_artifact(
                HBFTextTraceArtifact(mismatched, 8))

        lines = source.text.splitlines()
        lines[1] = "4"
        lines.insert(-1, lines[-2].replace(
            "hbf_read_probe", "hbf_read_probe_duplicate"))
        duplicated = "\n".join(lines) + "\n"
        with self.assertRaisesRegex(ValueError, "gang_base must be unique"):
            validate_hbf_trace_artifact(
                HBFTextTraceArtifact(duplicated, 8))

    def test_config_inference_is_topology_safe(self):
        source = build_hbf_media_microtrace(
            operation="write",
            tp_size=8,
            runtime_ns=20_001,
            tensor_bytes_per_rank=65_536,
        )
        _, parsed, audit = validate_hbf_trace_artifact(source)
        resolved = resolve_run_config(
            AstraHBFRunConfig(), parsed=parsed, audit=audit)
        self.assertEqual(resolved.dimensions, (8,))
        self.assertEqual(resolved.topology, ("Ring",))
        self.assertEqual(resolved.hbf_num_devices, 1)

        with self.assertRaisesRegex(
                ValueError, "must multiply to artifact.num_npus"):
            resolve_run_config(
                AstraHBFRunConfig(dimensions=(4,)),
                parsed=parsed,
                audit=audit,
            )

    def test_cycle_records_require_every_endpoint_exactly_once(self):
        output = "\n".join(
            f"sys[{rank}] iteration 0 finished, {100 + rank} cycles, "
            f"exposed communication {rank} cycles."
            for rank in range(2)
        )
        records = _parse_cycle_records(output)
        endpoints = _final_endpoint_cycles(records, num_npus=2)
        self.assertEqual(
            endpoints,
            (
                AstraCycleRecord(0, 0, 100, 0),
                AstraCycleRecord(1, 0, 101, 1),
            ),
        )
        with self.assertRaisesRegex(
                HBFAstraRunnerError,
                "did not report every required endpoint"):
            _final_endpoint_cycles(records[:1], num_npus=2)
        with self.assertRaisesRegex(
                HBFAstraRunnerError, "duplicate"):
            _final_endpoint_cycles(
                records + (records[0],), num_npus=2)

    def test_result_json_marks_only_actual_astra_cycles(self):
        source = build_hbf_media_microtrace(
            operation="read",
            tp_size=8,
            runtime_ns=20_001,
            tensor_bytes_per_rank=65_536,
        )
        _, parsed, audit = validate_hbf_trace_artifact(source)
        resolved = resolve_run_config(
            AstraHBFRunConfig(), parsed=parsed, audit=audit)
        from serving.core.hbf_astra_runner import AstraHBFRunResult
        result = AstraHBFRunResult(
            final_cycles=20_003,
            endpoint_cycles=tuple(
                AstraCycleRecord(rank, 0, 20_003, 0)
                for rank in range(8)
            ),
            trace=audit,
            resolved_config=resolved,
            config_sha256={
                "network": "a" * 64,
                "system": "b" * 64,
                "memory": "c" * 64,
            },
            graph_sha256_by_rank={
                rank: f"{rank:064x}" for rank in range(8)
            },
            binary_path="/tmp/astra",
            binary_sha256="d" * 64,
            protobuf_runtime_version="7.35.0",
            stdout_sha256="e" * 64,
            stderr_sha256="f" * 64,
        )
        encoded = json.loads(json.dumps(
            result.as_dict(), allow_nan=False, sort_keys=True))
        self.assertTrue(result.astra_cycles_used)
        self.assertTrue(encoded["astra_cycles_used"])
        self.assertFalse(encoded["analytical_cycle_substitution"])
        self.assertEqual(encoded["final_cycles"], 20_003)
        self.assertEqual(len(encoded["endpoint_cycles"]), 8)

    def test_persistent_job_uses_strict_controller_schema(self):
        job = HBFBackgroundJob(
            job_id="full-model.7",
            arrival_ns=123,
            stages=({
                "id": "card:0:compute",
                "runtime_ns": 100,
                "tensor_bytes": 256,
                "resources": ["server:0:card:0:npu"],
                "deps": [],
            },),
            projection_schema="unit-test-v1",
        )
        prefix, job_id, arrival, descriptor = job.command().split("\t")
        self.assertEqual(
            (prefix, job_id, arrival),
            ("hbf-background", "full-model.7", "123"),
        )
        self.assertEqual(
            json.loads(descriptor),
            {
                "v": 1,
                "stages": [{
                    "id": "card:0:compute",
                    "runtime_ns": 100,
                    "tensor_bytes": 256,
                    "resources": ["server:0:card:0:npu"],
                    "deps": [],
                }],
            },
        )
        self.assertRegex(job.descriptor_sha256, r"^[0-9a-f]{64}$")


class ActualHBFAstraRunnerTests(unittest.TestCase):
    """Real no-contention Chakra -> congestion-aware ASTRA execution."""

    @classmethod
    def setUpClass(cls):
        if not ASTRA_BINARY.is_file():
            raise unittest.SkipTest(
                "congestion-aware ASTRA-Sim binary is not built")
        try:
            _load_converter(CHAKRA_ROOT)
        except HBFAstraRunnerError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_single_hbf_gang_has_exact_no_contention_completion(self):
        # Two one-tick boundary COMP nodes surround one 20,001-tick
        # whole-gang HBF stage.  With no competing graph or background job,
        # every rank must complete at exactly 20,003 ASTRA ticks.
        artifact = build_hbf_media_microtrace(
            operation="read",
            tp_size=8,
            runtime_ns=20_001,
            tensor_bytes_per_rank=65_536,
        )
        result = run_hbf_trace_artifact(
            artifact,
            repo_root=REPO_ROOT,
            binary_path=ASTRA_BINARY,
            chakra_root=CHAKRA_ROOT,
            config=AstraHBFRunConfig(timeout_seconds=20.0),
        )
        self.assertTrue(result.astra_cycles_used)
        self.assertEqual(result.final_cycles, 20_003)
        self.assertEqual(
            {row.total_cycles for row in result.endpoint_cycles},
            {20_003},
        )
        self.assertEqual(result.trace.hbf_descriptor_count, 1)
        self.assertEqual(result.trace.hbf_stage_count, 1)
        self.assertEqual(len(result.graph_sha256_by_rank), 8)
        self.assertTrue(all(
            len(value) == 64
            for value in result.graph_sha256_by_rank.values()
        ))

    def test_persistent_background_job_delivers_exact_callback(self):
        callback_rows = []
        with PersistentHBFAstraRunner(
                num_npus=8,
                hbf_num_devices=8,
                repo_root=REPO_ROOT,
                binary_path=ASTRA_BINARY,
                chakra_root=CHAKRA_ROOT,
                config=AstraHBFRunConfig(timeout_seconds=20.0),
        ) as session:
            self.assertEqual(session.current_ns, 1)
            job = HBFBackgroundJob(
                job_id="full-model.0",
                arrival_ns=session.current_ns,
                stages=(
                    {
                        "id": "card:0:compute",
                        "runtime_ns": 100,
                        "tensor_bytes": 256,
                        "resources": ["server:0:card:0:npu"],
                        "deps": [],
                    },
                    {
                        "id": "card:1:compute",
                        "runtime_ns": 70,
                        "tensor_bytes": 128,
                        "resources": ["server:0:card:1:npu"],
                        "deps": [],
                    },
                ),
                projection_schema="synthetic-full-model-v1",
            )
            completion, = session.submit_jobs(
                (job,), on_complete=callback_rows.append)
            self.assertEqual(completion.arrival_ns, 1)
            self.assertEqual(completion.completion_ns, 101)
            self.assertEqual(completion.elapsed_cycles, 100)
            self.assertEqual(completion.stage_count, 2)
            self.assertTrue(completion.astra_cycles_used)
            self.assertEqual(callback_rows, [completion])
        audit = session.close()
        self.assertTrue(audit.astra_cycles_used)
        self.assertTrue(audit.clean_exit)
        self.assertEqual(audit.completions, (completion,))
        self.assertTrue(audit.as_dict()["astra_cycles_used"])

    def test_full_model_projection_runs_on_persistent_astra_critical_path(self):
        hardware = HBFServerHardware()
        layout = HBFParallelLayout.for_key("tp4")
        latency = synthetic_full_model_latency()
        projection = build_full_model_hbf_astra_projection(
            latency=latency,
            hardware=hardware,
            layout=layout,
            replica_id=0,
            batch_id=0,
            server_id=0,
        )
        with PersistentHBFAstraRunner(
                num_npus=8,
                hbf_num_devices=8,
                repo_root=REPO_ROOT,
                binary_path=ASTRA_BINARY,
                chakra_root=CHAKRA_ROOT,
                config=AstraHBFRunConfig(timeout_seconds=20.0),
        ) as session:
            completion = session.submit_projection(
                job_id="full-model-projection.0",
                arrival_ns=session.current_ns,
                projection=projection,
            )
            self.assertEqual(
                completion.elapsed_cycles, latency.total_ns)
            self.assertEqual(
                completion.stage_count, len(projection.stages))
            self.assertEqual(
                completion.projection_schema, projection.schema)
            self.assertTrue(completion.astra_cycles_used)

    def test_persistent_jobs_serialize_shared_and_overlap_disjoint_resources(
            self):
        def run_case(shared):
            with PersistentHBFAstraRunner(
                    num_npus=8,
                    hbf_num_devices=8,
                    repo_root=REPO_ROOT,
                    binary_path=ASTRA_BINARY,
                    chakra_root=CHAKRA_ROOT,
                    config=AstraHBFRunConfig(timeout_seconds=20.0),
            ) as session:
                arrival = session.current_ns
                jobs = tuple(
                    HBFBackgroundJob(
                        job_id=f"contention.{shared}.{index}",
                        arrival_ns=arrival,
                        stages=({
                            "id": "full-model",
                            "runtime_ns": 100,
                            "tensor_bytes": 1,
                            "resources": [
                                (
                                    "server:0:card:0:npu"
                                    if shared else
                                    f"server:0:card:{index}:npu"
                                )
                            ],
                            "deps": [],
                        },),
                    )
                    for index in range(2)
                )
                completions = session.submit_jobs(jobs)
            audit = session.close()
            self.assertTrue(audit.clean_exit)
            return sorted(
                completion.completion_ns - arrival
                for completion in completions
            )

        self.assertEqual(run_case(shared=True), [100, 200])
        self.assertEqual(run_case(shared=False), [100, 100])

    def test_interference_can_finish_below_descriptor_order_solo_time(self):
        def stage(stage_id, runtime_ns, resource, dependencies=()):
            return HBFModelAstraStage(
                stage_id=stage_id,
                runtime_ns=runtime_ns,
                tensor_bytes=0,
                resources=(resource,),
                dependencies=tuple(dependencies),
            )

        stages_a = (stage("a", 7, "X"),)
        stages_b = (
            stage("b0", 2, "X"),
            stage("b1", 6, "Z"),
            stage("b2", 3, "Z", ("b1",)),
            stage("b3", 10, "W", ("b2",)),
            stage("b4", 7, "Z", ("b3",)),
            stage("b5", 1, "Z", ("b0",)),
        )
        self.assertEqual(
            hbf_dependency_critical_path_ns(stages_b), 26)
        self.assertEqual(
            hbf_solo_named_resource_timing(
                stages_b).resource_serialized_completion_ns,
            27,
        )

        with PersistentHBFAstraRunner(
                num_npus=8,
                hbf_num_devices=8,
                repo_root=REPO_ROOT,
                binary_path=ASTRA_BINARY,
                chakra_root=CHAKRA_ROOT,
                config=AstraHBFRunConfig(timeout_seconds=20.0),
        ) as session:
            arrival = session.current_ns
            completions = session.submit_jobs((
                HBFBackgroundJob(
                    job_id="interference-a",
                    arrival_ns=arrival,
                    stages=tuple(row.as_dict() for row in stages_a),
                ),
                HBFBackgroundJob(
                    job_id="interference-b",
                    arrival_ns=arrival,
                    stages=tuple(row.as_dict() for row in stages_b),
                ),
            ))
        elapsed = {
            completion.job_id: completion.elapsed_cycles
            for completion in completions
        }
        self.assertEqual(elapsed["interference-a"], 7)
        self.assertEqual(elapsed["interference-b"], 26)
        self.assertLess(elapsed["interference-b"], 27)
        self.assertGreaterEqual(
            elapsed["interference-b"],
            hbf_dependency_critical_path_ns(stages_b),
        )

    def test_random_concurrent_dags_respect_only_dependency_lower_bound(self):
        rng = random.Random(20_260_724)
        specifications = {}
        jobs = []
        with PersistentHBFAstraRunner(
                num_npus=8,
                hbf_num_devices=8,
                repo_root=REPO_ROOT,
                binary_path=ASTRA_BINARY,
                chakra_root=CHAKRA_ROOT,
                config=AstraHBFRunConfig(timeout_seconds=20.0),
        ) as session:
            arrival = session.current_ns
            for job_index in range(24):
                stages = []
                for stage_index in range(rng.randint(2, 8)):
                    dependencies = ()
                    if stage_index and rng.random() < 0.8:
                        dependency_count = rng.randint(
                            1, min(2, stage_index))
                        dependencies = tuple(sorted(rng.sample(
                            [
                                f"j{job_index}:s{prior}"
                                for prior in range(stage_index)
                            ],
                            dependency_count,
                        )))
                    stages.append(HBFModelAstraStage(
                        stage_id=f"j{job_index}:s{stage_index}",
                        runtime_ns=rng.randint(1, 13),
                        tensor_bytes=0,
                        resources=(
                            f"shared-r{rng.randrange(5)}",
                        ),
                        dependencies=dependencies,
                    ))
                materialized = tuple(stages)
                job_id = f"random-{job_index}"
                dependency_ns = hbf_dependency_critical_path_ns(
                    materialized)
                solo_ns = hbf_solo_named_resource_timing(
                    materialized
                ).resource_serialized_completion_ns
                specifications[job_id] = (dependency_ns, solo_ns)
                jobs.append(HBFBackgroundJob(
                    job_id=job_id,
                    arrival_ns=arrival,
                    stages=tuple(
                        stage.as_dict() for stage in materialized),
                ))
            completions = session.submit_jobs(tuple(jobs))

        self.assertEqual(len(completions), len(specifications))
        for completion in completions:
            dependency_ns, solo_ns = specifications[completion.job_id]
            accounting = HBFAstraTimingAccounting(
                dependency_critical_path_ns=dependency_ns,
                solo_resource_serialized_completion_ns=solo_ns,
                actual_resource_serialized_completion_ns=(
                    completion.elapsed_cycles),
            )
            self.assertGreaterEqual(
                completion.elapsed_cycles, dependency_ns)
            self.assertEqual(
                accounting.resource_delay_ns,
                (
                    accounting.internal_resource_serialization_wait_ns
                    + accounting.signed_interference_delta_ns
                ),
            )


if __name__ == "__main__":
    unittest.main()

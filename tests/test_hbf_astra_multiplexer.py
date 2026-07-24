import dataclasses
from pathlib import Path
import unittest

from serving.core.hbf_astra_multiplexer import (
    HBFAstraDrainError,
    HBFAstraJobMultiplexer,
    HBFAstraMultiplexerError,
    MULTIPLEXER_SCHEMA,
)
from serving.core.hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
)
from serving.core.hbf_full_model_lifecycle import (
    FullModelHBFLifecycle,
    PlacementState,
)
from serving.core.hbf_full_model_pool import (
    FullModelHBFServingPool,
    HBFRequestState,
    HBFServingRequest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def stage(stage_id="stage", *, runtime_ns=10, dependencies=()):
    return {
        "id": stage_id,
        "runtime_ns": runtime_ns,
        "tensor_bytes": 1,
        "resources": ["hbf-server:0:card:0:hbf-read"],
        "deps": list(dependencies),
    }


class FakeSource:
    def __init__(self, jobs=()):
        self.outbox = list(jobs)
        self.inflight = {}
        self.completions = []
        self.fail_completion = False
        self.drain_calls = 0

    def drain(self):
        self.drain_calls += 1
        jobs = tuple(self.outbox)
        self.outbox.clear()
        for job in jobs:
            job_id = (
                job["job_id"]
                if isinstance(job, dict) else job.job_id
            )
            self.inflight[job_id] = job
        return jobs

    def complete(
            self, *, job_id, arrival_ns, completion_ns,
            stage_count, marker=None):
        if self.fail_completion:
            raise RuntimeError("synthetic owner failure")
        if job_id not in self.inflight:
            raise RuntimeError("fake owner lost its job")
        del self.inflight[job_id]
        row = {
            "job_id": job_id,
            "arrival_ns": arrival_ns,
            "completion_ns": completion_ns,
            "stage_count": stage_count,
            "marker": marker,
        }
        self.completions.append(row)
        return row

    def has_pending(self):
        return bool(self.outbox or self.inflight)


@dataclasses.dataclass(frozen=True)
class ObjectDispatch:
    job_id: str
    arrival_ns: int
    stages: tuple[dict, ...]
    argument_job_id: str | None = None

    def controller_arguments(self):
        return (
            self.job_id
            if self.argument_job_id is None
            else self.argument_job_id,
            self.arrival_ns,
            self.stages,
        )


class HBFAstraMultiplexerFakeTests(unittest.TestCase):
    def register(self, mux, name, source):
        mux.register_source(
            name,
            drain=source.drain,
            complete=source.complete,
            has_pending=source.has_pending,
        )

    def test_mapping_and_object_jobs_are_normalized_and_routed(self):
        mapping_source = FakeSource([{
            "job_id": "wakekv.flush.1",
            "arrival_ns": 5,
            "stages": [stage("flush")],
        }])
        object_source = FakeSource([ObjectDispatch(
            job_id="hbf-model.s0.r0.b1",
            arrival_ns=7,
            stages=(stage("model"),),
        )])
        mux = HBFAstraJobMultiplexer()
        self.register(mux, "wakekv", mapping_source)
        self.register(mux, "model", object_source)

        jobs = mux.drain_jobs()
        self.assertEqual(
            [
                (job.source_name, job.owner_job_id)
                for job in jobs
            ],
            [
                ("wakekv", "wakekv.flush.1"),
                ("model", "hbf-model.s0.r0.b1"),
            ],
        )
        self.assertEqual(len({job.job_id for job in jobs}), 2)
        self.assertTrue(all(
            job.job_id != job.owner_job_id for job in jobs))
        for job in jobs:
            self.assertTrue(
                job.controller_command.startswith(
                    f"hbf-background\t{job.job_id}\t"))
            self.assertEqual(job.stage_count, 1)
            self.assertEqual(len(job.descriptor_sha256), 64)
            self.assertEqual(
                job.controller_arguments()[:2],
                (job.job_id, job.arrival_ns),
            )
        self.assertTrue(mux.has_pending())

        first = mux.complete(
            job_id=jobs[0].job_id,
            arrival_ns=jobs[0].arrival_ns,
            completion_ns=25,
            stage_count=jobs[0].stage_count,
        )
        self.assertIs(first.owner_result, mapping_source.completions[0])
        self.assertEqual(first.source_name, "wakekv")
        self.assertEqual(first.owner_job_id, "wakekv.flush.1")
        self.assertEqual(first.elapsed_ns, 20)
        self.assertEqual(mux.pending_job_ids, (jobs[1].job_id,))
        self.assertEqual(
            mapping_source.completions[0]["job_id"],
            "wakekv.flush.1",
        )

        second = mux.complete(
            job_id=jobs[1].job_id,
            arrival_ns=jobs[1].arrival_ns,
            completion_ns=30,
            stage_count=jobs[1].stage_count,
        )
        self.assertIs(second.owner_result, object_source.completions[0])
        self.assertEqual(
            object_source.completions[0]["job_id"],
            "hbf-model.s0.r0.b1",
        )
        self.assertFalse(mux.has_pending())
        self.assertEqual(
            mux.completed_job_ids,
            tuple(sorted(job.job_id for job in jobs)),
        )

    def test_object_adapter_forwards_constant_completion_kwargs(self):
        source = FakeSource([{
            "job_id": "wakekv.flush.2",
            "arrival_ns": 1,
            "stages": [stage()],
        }])
        mux = HBFAstraJobMultiplexer()
        mux.register_object(
            "wakekv",
            source,
            drain_method="drain",
            complete_method="complete",
            has_pending_method="has_pending",
            complete_kwargs={"marker": "adapter"},
        )
        job, = mux.drain_jobs()
        result = mux.complete(
            job_id=job.job_id,
            arrival_ns=job.arrival_ns,
            completion_ns=11,
            stage_count=job.stage_count,
        )
        self.assertEqual(result.owner_result["marker"], "adapter")

    def test_cross_source_owner_collisions_get_unique_controller_aliases(self):
        first = FakeSource([{
            "job_id": "collision.1",
            "arrival_ns": 0,
            "stages": [stage()],
        }])
        second = FakeSource([{
            "job_id": "collision.1",
            "arrival_ns": 0,
            "stages": [stage()],
        }])
        mux = HBFAstraJobMultiplexer()
        self.register(mux, "first", first)
        self.register(mux, "second", second)
        jobs = mux.drain_jobs()
        self.assertEqual(len({job.job_id for job in jobs}), 2)
        self.assertEqual(
            {job.owner_job_id for job in jobs}, {"collision.1"})
        collision, = mux.report()["owner_job_id_collisions"]
        self.assertEqual(collision["owner_job_id"], "collision.1")
        self.assertEqual(
            {row["source_name"] for row in collision["jobs"]},
            {"first", "second"},
        )
        for job in jobs:
            mux.complete(
                job_id=job.job_id,
                arrival_ns=0,
                completion_ns=10,
                stage_count=1,
            )
        self.assertEqual(first.completions[0]["job_id"], "collision.1")
        self.assertEqual(second.completions[0]["job_id"], "collision.1")

    def test_same_source_reemission_and_completed_reuse_are_quarantined(self):
        source = FakeSource([{
            "job_id": "stable.1",
            "arrival_ns": 0,
            "stages": [stage()],
        }])
        mux = HBFAstraJobMultiplexer()
        self.register(mux, "source", source)
        job, = mux.drain_jobs()
        source.outbox.append({
            "job_id": job.owner_job_id,
            "arrival_ns": 0,
            "stages": [stage()],
        })
        with self.assertRaisesRegex(
                HBFAstraDrainError, "re-emitted") as captured:
            mux.drain_jobs()
        self.assertEqual(mux.pending_job_ids, (job.job_id,))
        self.assertEqual(len(captured.exception.quarantine_ids), 1)
        self.assertTrue(mux.has_pending())

        source = FakeSource([{
            "job_id": "stable.2",
            "arrival_ns": 0,
            "stages": [stage()],
        }])
        mux = HBFAstraJobMultiplexer()
        self.register(mux, "source", source)
        job, = mux.drain_jobs()
        mux.complete(
            job_id=job.job_id,
            arrival_ns=0,
            completion_ns=10,
            stage_count=1,
        )
        source.outbox.append({
            "job_id": job.owner_job_id,
            "arrival_ns": 0,
            "stages": [stage()],
        })
        with self.assertRaisesRegex(
                HBFAstraDrainError, "reused"):
            mux.drain_jobs()
        self.assertEqual(len(mux.quarantined_dispatch_ids), 1)

    def test_unknown_duplicate_and_metadata_drift_are_strict(self):
        source = FakeSource([{
            "job_id": "strict.1",
            "arrival_ns": 9,
            "stages": [stage()],
        }])
        mux = HBFAstraJobMultiplexer()
        self.register(mux, "source", source)
        job, = mux.drain_jobs()
        with self.assertRaisesRegex(
                HBFAstraMultiplexerError, "unknown"):
            mux.complete(
                job_id="missing",
                arrival_ns=9,
                completion_ns=10,
                stage_count=1,
            )
        with self.assertRaisesRegex(
                HBFAstraMultiplexerError, "unknown"):
            mux.complete(
                job_id=job.owner_job_id,
                arrival_ns=9,
                completion_ns=10,
                stage_count=1,
            )
        with self.assertRaisesRegex(
                HBFAstraMultiplexerError, "arrival metadata drift"):
            mux.complete(
                job_id=job.job_id,
                arrival_ns=10,
                completion_ns=20,
                stage_count=1,
            )
        with self.assertRaisesRegex(
                HBFAstraMultiplexerError, "stage-count metadata drift"):
            mux.complete(
                job_id=job.job_id,
                arrival_ns=9,
                completion_ns=20,
                stage_count=2,
            )
        self.assertEqual(source.completions, [])
        self.assertEqual(mux.pending_job_ids, (job.job_id,))
        mux.complete(
            job_id=job.job_id,
            arrival_ns=9,
            completion_ns=20,
            stage_count=1,
        )
        with self.assertRaisesRegex(
                HBFAstraMultiplexerError, "duplicate"):
            mux.complete(
                job_id=job.job_id,
                arrival_ns=9,
                completion_ns=20,
                stage_count=1,
            )

    def test_owner_failure_retains_pending_ownership_for_retry(self):
        source = FakeSource([{
            "job_id": "retry.1",
            "arrival_ns": 0,
            "stages": [stage()],
        }])
        mux = HBFAstraJobMultiplexer()
        self.register(mux, "source", source)
        job, = mux.drain_jobs()
        source.fail_completion = True
        with self.assertRaisesRegex(RuntimeError, "owner failure"):
            mux.complete(
                job_id=job.job_id,
                arrival_ns=0,
                completion_ns=10,
                stage_count=1,
            )
        self.assertEqual(mux.pending_job_ids, (job.job_id,))
        self.assertEqual(mux.completed_job_ids, ())
        source.fail_completion = False
        mux.complete(
            job_id=job.job_id,
            arrival_ns=0,
            completion_ns=10,
            stage_count=1,
        )

    def test_controller_is_final_schema_and_dependency_gate(self):
        invalid = FakeSource([{
            "job_id": "invalid.1",
            "arrival_ns": 0,
            "stages": [{
                **stage("child"),
                "deps": ["missing"],
            }],
        }])
        mux = HBFAstraJobMultiplexer()
        self.register(mux, "invalid", invalid)
        with self.assertRaisesRegex(
                HBFAstraDrainError, "unknown dependency") as captured:
            mux.drain_jobs()
        self.assertEqual(mux.pending_job_ids, ())
        quarantine_id, = captured.exception.quarantine_ids
        self.assertEqual(
            mux.quarantined_dispatch_ids, (quarantine_id,))
        repaired = mux.retry_quarantined(
            quarantine_id,
            dispatch={
                "job_id": "invalid.1",
                "arrival_ns": 0,
                "stages": [stage("child")],
            },
        )
        mux.complete(
            job_id=repaired.job_id,
            arrival_ns=0,
            completion_ns=10,
            stage_count=1,
        )
        self.assertEqual(
            invalid.completions[0]["job_id"], "invalid.1")
        self.assertEqual(mux.quarantined_dispatch_ids, ())

        mismatch = FakeSource([ObjectDispatch(
            job_id="object.1",
            argument_job_id="object.2",
            arrival_ns=0,
            stages=(stage(),),
        )])
        mux = HBFAstraJobMultiplexer()
        self.register(mux, "mismatch", mismatch)
        with self.assertRaisesRegex(
                HBFAstraDrainError, "differs"):
            mux.drain_jobs()

    def test_later_failure_retains_and_replays_earlier_destructive_drain(self):
        good = FakeSource([{
            "job_id": "good.1",
            "arrival_ns": 4,
            "stages": [stage("good")],
        }])
        bad = FakeSource([{
            "job_id": "bad.1",
            "arrival_ns": 4,
            "stages": [{
                **stage("bad"),
                "deps": ["missing"],
            }],
        }])
        mux = HBFAstraJobMultiplexer()
        self.register(mux, "good", good)
        self.register(mux, "bad", bad)

        with self.assertRaises(HBFAstraDrainError) as captured:
            mux.drain_jobs()
        ready_alias, = captured.exception.ready_job_ids
        self.assertEqual(mux.pending_job_ids, (ready_alias,))
        self.assertEqual(mux.ready_job_ids, (ready_alias,))
        self.assertIn("good.1", good.inflight)
        self.assertEqual((good.drain_calls, bad.drain_calls), (1, 1))
        self.assertEqual(
            mux.pending_audit()[0]["handoff_state"], "ready")
        with self.assertRaisesRegex(
                HBFAstraMultiplexerError, "before Controller handoff"):
            mux.complete(
                job_id=ready_alias,
                arrival_ns=4,
                completion_ns=14,
                stage_count=1,
            )

        recovered, = mux.drain_jobs()
        self.assertEqual(recovered.job_id, ready_alias)
        self.assertEqual(recovered.owner_job_id, "good.1")
        self.assertEqual((good.drain_calls, bad.drain_calls), (1, 1))
        mux.complete(
            job_id=recovered.job_id,
            arrival_ns=4,
            completion_ns=14,
            stage_count=1,
        )
        self.assertEqual(good.completions[0]["job_id"], "good.1")
        self.assertEqual(len(mux.quarantined_dispatch_ids), 1)
        self.assertTrue(mux.has_pending())

    def test_source_and_pending_report_is_auditable(self):
        source = FakeSource([{
            "job_id": "audit.1",
            "arrival_ns": 3,
            "stages": [stage()],
        }])
        mux = HBFAstraJobMultiplexer()
        self.register(mux, "source", source)
        job, = mux.drain_jobs()
        report = mux.report()
        self.assertEqual(report["schema"], MULTIPLEXER_SCHEMA)
        self.assertEqual(report["registered_sources"], ["source"])
        self.assertEqual(report["pending_job_count"], 1)
        self.assertEqual(
            report["pending_jobs"][0]["job_id"], job.job_id)
        self.assertEqual(
            report["pending_jobs"][0]["owner_job_id"], "audit.1")
        self.assertEqual(
            report["pending_jobs"][0]["handoff_state"], "issued")
        self.assertEqual(
            report["source_audit"]["source"][
                "mux_pending_job_ids"],
            [job.job_id],
        )
        self.assertEqual(
            report["source_audit"]["source"][
                "mux_pending_owner_job_ids"],
            ["audit.1"],
        )
        self.assertTrue(report["has_pending"])

        mux.complete(
            job_id=job.job_id,
            arrival_ns=job.arrival_ns,
            completion_ns=13,
            stage_count=job.stage_count,
        )
        report = mux.report()
        self.assertEqual(report["pending_job_count"], 0)
        self.assertEqual(report["completed_job_count"], 1)
        self.assertFalse(report["has_pending"])


class HBFAstraMultiplexerRealDispatchTests(unittest.TestCase):
    def test_real_pool_and_lifecycle_dispatch_shapes_share_one_mux(self):
        hardware = HBFServerHardware()
        layout = HBFParallelLayout.for_key("tp4")
        lifecycle = FullModelHBFLifecycle(
            hardware=hardware,
            layout=layout,
            kv_bytes_per_token=1,
            execution_backend="external_astra",
            server_id=3,
            astra_chunk_bytes=11,
        )
        lifecycle.register_session("session-lifecycle")
        lifecycle.complete_gpu_turn(
            "session-lifecycle",
            now_ns=5,
            total_tokens=53,
            has_successor=True,
        )

        pool = FullModelHBFServingPool(
            repo_root=REPO_ROOT,
            hardware=hardware,
            layout=layout,
            max_num_batched_tokens=16,
            max_num_seqs=4,
            max_prefill_chunk_tokens=16,
            execution_backend="external_astra",
            server_id=3,
        )
        request = HBFServingRequest(
            request_id=1,
            session_id="session-pool",
            arrival_ns=5,
            input_tokens=1,
            output_tokens=1,
            hbf_prefix_tokens=1,
            lpddr_prefix_tokens=0,
            group_id=0,
        )
        pool.submit(request, now_ns=5)

        mux = HBFAstraJobMultiplexer()
        mux.register_object(
            "lifecycle",
            lifecycle,
            drain_method="drain_external_dispatches",
            complete_method="complete_external_dispatch",
            has_pending_method="has_pending_external",
        )
        mux.register_object(
            "pool",
            pool,
            drain_method="drain_external_dispatches",
            complete_method="complete_external_dispatch",
            has_pending_method="has_pending_external_dispatches",
            complete_kwargs={"defer_schedule": True},
        )
        jobs = mux.drain_jobs()
        self.assertEqual(len(jobs), 2)
        self.assertEqual(
            {job.source_name for job in jobs},
            {"lifecycle", "pool"},
        )
        self.assertEqual(len({job.job_id for job in jobs}), 2)
        self.assertTrue(all(
            job.job_id.startswith(f"mux.{job.source_name}.")
            for job in jobs
        ))
        self.assertTrue(all(
            job.job_id != job.owner_job_id for job in jobs))

        for job in sorted(jobs, key=lambda item: item.source_name):
            _, _, stages = job.controller_arguments()
            completion_ns = (
                job.arrival_ns
                + sum(row["runtime_ns"] for row in stages)
            )
            result = mux.complete(
                job_id=job.job_id,
                arrival_ns=job.arrival_ns,
                completion_ns=completion_ns,
                stage_count=job.stage_count,
            )
            self.assertEqual(result.source_name, job.source_name)
            self.assertEqual(result.owner_job_id, job.owner_job_id)
            completed_ids = (
                lifecycle._external_completed_job_ids
                if job.source_name == "lifecycle"
                else pool._external_completed_job_ids
            )
            self.assertIn(job.owner_job_id, completed_ids)
            self.assertNotIn(job.job_id, completed_ids)

        self.assertEqual(
            lifecycle.sessions["session-lifecycle"].state,
            PlacementState.HBF_READY,
        )
        self.assertEqual(request.state, HBFRequestState.COMPLETE)
        self.assertFalse(mux.has_pending())


if __name__ == "__main__":
    unittest.main()

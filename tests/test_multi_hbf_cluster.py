from pathlib import Path
import unittest

from serving.core.hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
)
from serving.core.hbf_full_model_lifecycle import (
    PlacementState,
    ResourceCalendar,
    ResumeExecution,
)
from serving.core.hbf_full_model_pool import HBFServingRequest
from serving.core.multi_hbf_cluster import MultiHBFCluster


REPO_ROOT = Path(__file__).resolve().parents[1]


class MultiHBFClusterTests(unittest.TestCase):
    def make_cluster(self, *layout_keys):
        return MultiHBFCluster(
            repo_root=REPO_ROOT,
            hardware=HBFServerHardware(),
            layouts=[
                HBFParallelLayout.for_key(key)
                for key in layout_keys
            ],
            resource_calendar=ResourceCalendar(),
            gpu_source_root_bandwidth_gbps=200.0,
            max_num_batched_tokens=256,
            max_num_seqs=16,
            max_prefill_chunk_tokens=64,
        )

    def test_tp4_servers_have_two_workers_each_and_sticky_sessions(self):
        cluster = self.make_cluster("tp4", "tp4")

        self.assertEqual(cluster.server_count, 2)
        self.assertEqual(
            [len(bundle.pool.workers) for bundle in cluster.bundles],
            [2, 2],
        )
        self.assertEqual(len(cluster.pool.workers), 4)

        for session_id in ("s0", "s1", "s2", "s3"):
            cluster.lifecycle.register_session(session_id, now_ns=0)
        self.assertEqual(cluster.session_server_index, {
            "s0": 0,
            "s1": 1,
            "s2": 0,
            "s3": 1,
        })
        for session_id, server_index in (
                cluster.session_server_index.items()):
            self.assertIs(
                cluster.bundle_for_session(session_id),
                cluster.bundles[server_index],
            )
            self.assertEqual(
                cluster.lifecycle.server_index_for_session(session_id),
                server_index,
            )

        cluster.lifecycle.complete_gpu_turn(
            "s0",
            now_ns=10,
            total_tokens=100,
            has_successor=False,
        )
        self.assertEqual(
            cluster.lifecycle.server_index_for_session("s0"), 0)
        self.assertEqual(
            cluster.bundles[0].lifecycle.sessions["s0"].state,
            PlacementState.ENDED,
        )
        with self.assertRaisesRegex(ValueError, "session_id"):
            cluster.lifecycle.register_session("", now_ns=10)
        self.assertNotIn("", cluster.session_server_index)
        cluster.lifecycle.assert_invariants()
        cluster.pool.assert_invariants()

    def test_tp8_context_keeps_one_physical_kv_copy_per_server(self):
        cluster = self.make_cluster(
            "tp8_context", "tp8_context")

        self.assertEqual(len(cluster.pool.workers), 2)
        self.assertEqual(
            [
                bundle.layout.physical_kv_replication_factor
                for bundle in cluster.bundles
            ],
            [1, 1],
        )
        for session_id in ("s0", "s1"):
            cluster.lifecycle.register_session(session_id, now_ns=0)
        jobs = [
            cluster.lifecycle.complete_gpu_turn(
                session_id,
                now_ns=0,
                total_tokens=1_001,
                has_successor=True,
            )
            for session_id in ("s0", "s1")
        ]

        self.assertTrue(all(job is not None for job in jobs))
        for job in jobs:
            self.assertEqual(job.physical_bytes, job.logical_bytes)
            self.assertEqual(
                sum(byte_count for _, byte_count in job.card_bytes),
                job.logical_bytes,
            )
        cluster.lifecycle.advance(max(
            job.completion_ns for job in jobs))
        cluster.lifecycle.assert_invariants()

    def test_local_serving_overlaps_shared_rdma_serializes_and_drains(self):
        cluster = self.make_cluster("tp4", "tp4")
        for session_id in ("s0", "s1"):
            cluster.lifecycle.register_session(session_id, now_ns=0)
        jobs = [
            cluster.lifecycle.complete_gpu_turn(
                session_id,
                now_ns=0,
                total_tokens=100,
                has_successor=True,
            )
            for session_id in ("s0", "s1")
        ]

        rdma = [
            reservation
            for reservation in cluster.calendar.reservations
            if reservation.resource == "rdma-network"
        ]
        self.assertEqual(len(rdma), 2)
        self.assertEqual(rdma[0].end_ns, rdma[1].start_ns)
        self.assertEqual(
            cluster.calendar.reservation_count_by_resource[
                "gpu-source-pcie-root"],
            2,
        )
        resources = set(cluster.calendar.available_ns)
        self.assertTrue(any(
            resource.startswith("hbf-server-0-")
            for resource in resources
        ))
        self.assertTrue(any(
            resource.startswith("hbf-server-1-")
            for resource in resources
        ))

        ready_ns = max(job.completion_ns for job in jobs)
        cluster.lifecycle.advance(ready_ns)
        requests = []
        for request_id, session_id in enumerate(("s0", "s1")):
            route = cluster.lifecycle.route_resume(
                session_id,
                now_ns=ready_ns,
                request_id=request_id,
                prefix_reuse_tokens=100,
                input_tokens=100,
                lpddr_growth_tokens=0,
            )
            self.assertEqual(route.execution, ResumeExecution.HBF)
            requests.append(HBFServingRequest(
                request_id=request_id,
                session_id=session_id,
                arrival_ns=ready_ns,
                input_tokens=100,
                output_tokens=1,
                hbf_prefix_tokens=route.hbf_tokens,
                lpddr_prefix_tokens=route.lpddr_tokens,
                group_id=route.group_id,
            ))

        # A request on only one server must still advance the idle child.
        cluster.pool.submit_many([requests[0]], now_ns=ready_ns)
        self.assertEqual(
            [bundle.pool.current_ns for bundle in cluster.bundles],
            [ready_ns, ready_ns],
        )
        cluster.pool.submit_many([requests[1]], now_ns=ready_ns)

        first_batches = [
            bundle.pool.batch_history[0]
            for bundle in cluster.bundles
        ]
        self.assertEqual(
            first_batches[0].start_ns,
            first_batches[1].start_ns,
        )
        npu_reservations = [
            reservation
            for reservation in cluster.calendar.reservations
            if (
                reservation.kind == "hbf-model-batch"
                and reservation.resource.endswith("-npu")
            )
        ]
        self.assertEqual(len(npu_reservations), 2)
        self.assertEqual(
            {
                reservation.start_ns
                for reservation in npu_reservations
            },
            {ready_ns},
        )
        self.assertEqual(
            len({
                reservation.resource
                for reservation in npu_reservations
            }),
            2,
        )

        completed = []
        while cluster.pool.next_event_ns() is not None:
            event_ns = cluster.pool.next_event_ns()
            cluster.pool.advance(event_ns)
            completed.extend(cluster.pool.pop_completed())
        self.assertEqual(
            [request.request_id for request in completed],
            [0, 1],
        )
        for request in completed:
            cluster.lifecycle.complete_hbf_turn(
                request.session_id,
                now_ns=request.completion_ns,
                total_tokens=100,
                has_successor=False,
            )

        self.assertIsNone(cluster.pool.next_event_ns())
        self.assertIsNone(cluster.lifecycle.next_completion_ns())
        self.assertTrue(all(
            placement.state == PlacementState.ENDED
            for placement in cluster.lifecycle.sessions.values()
        ))
        for bundle in cluster.bundles:
            self.assertTrue(all(
                bundle.lpddr_ledger.used_bytes(group_id) == 0
                for group_id in range(bundle.layout.replicas)
            ))
        self.assertEqual(
            cluster.lifecycle.metrics.migrations_committed, 2)
        self.assertEqual(cluster.pool.metrics.completed_requests, 2)
        cluster.lifecycle.assert_invariants()
        cluster.pool.assert_invariants()


if __name__ == "__main__":
    unittest.main()

import io
import unittest
from types import SimpleNamespace

from serving.core.controller import (
    Controller,
    ExactControlSchedule,
    SameTimeControlBarrier,
)


class _Process:
    def __init__(self, stdout, stderr="backend protocol error", code=1):
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.code = code

    def poll(self):
        return self.code


class ControllerProtocolTest(unittest.TestCase):
    def test_read_wait_fails_instead_of_spinning_on_backend_eof(self):
        process = _Process("partial output\n")

        with self.assertRaisesRegex(
                RuntimeError, "backend protocol error"):
            Controller(1).read_wait(process)

    def test_check_end_fails_instead_of_spinning_on_backend_eof(self):
        process = _Process("partial output\n")

        with self.assertRaisesRegex(RuntimeError, "final completion"):
            Controller(1).check_end(process)

    def test_normal_wait_marker_is_returned(self):
        process = _Process(
            "sys[0] iteration 0 finished, 10 cycles, "
            "exposed communication 0 cycles.\nWaiting\n",
            stderr="",
            code=None,
        )

        output = Controller(1).read_wait(process)

        self.assertIn("Waiting\n", output)

    def test_exact_control_event_is_parsed_without_model_identity(self):
        output = [
            "Control event\trequest-ready-7\t12345\n",
            "Control Waiting\n",
        ]

        event = Controller(1).parse_protocol_event(output)

        self.assertEqual(event, {
            "type": "control_event",
            "event_id": "request-ready-7",
            "time_ns": 12345,
        })

    def test_background_completion_preserves_exact_job_metadata(self):
        output = [
            "Background transfer accepted\trestore.3\t10\t4096\t4\n",
            "Background transfer complete\trestore.3\t10\t99\t4096\t4\t25\n",
            "Control Waiting\n",
        ]

        event = Controller(1).parse_protocol_event(output)

        self.assertEqual(event, {
            "type": "background_transfer_complete",
            "job_id": "restore.3",
            "arrival_ns": 10,
            "completion_ns": 99,
            "bytes_per_lane": 4096,
            "lane_count": 4,
            "critical_lane_start_ns": 25,
        })

    def test_background_fabric_capability_is_explicit(self):
        output = [
            "Analytical control capability\tcold-fabric-v1\n",
            "sys[0] iteration 0 finished, 1 cycles, "
            "exposed communication 0 cycles.\n",
            "Waiting\n",
        ]

        self.assertTrue(
            Controller.has_background_fabric_capability(output))
        self.assertFalse(
            Controller.has_background_fabric_capability(["Waiting\n"]))

    def test_endpoint_park_capability_is_explicit(self):
        output = [
            "Analytical control capability\tendpoint-park-v1\n",
            "sys[0] iteration 0 finished, 1 cycles, "
            "exposed communication 0 cycles.\n",
            "Waiting\n",
        ]

        self.assertTrue(Controller.has_endpoint_park_capability(output))
        self.assertFalse(
            Controller.has_endpoint_park_capability(["Waiting\n"]))

    def test_hbf_background_capability_and_completion_are_distinct(self):
        output = [
            "Analytical control capability\thbf-background-v1\n",
            "HBF background complete\tflush.7\t10\t99\t4\n",
            "Control Waiting\n",
        ]

        self.assertTrue(Controller.has_hbf_background_capability(output))
        self.assertFalse(
            Controller.has_hbf_background_capability(["Waiting\n"]))
        self.assertEqual(Controller(1).parse_protocol_event(output), {
            "type": "hbf_background_complete",
            "job_id": "flush.7",
            "arrival_ns": 10,
            "completion_ns": 99,
            "stage_count": 4,
        })

    def test_post_endpoint_barrier_capability_and_command_are_explicit(self):
        output = [
            "Analytical control capability\tpost-endpoint-barrier-v1\n",
            "Waiting\n",
        ]

        self.assertTrue(
            Controller.has_post_endpoint_barrier_capability(output))
        self.assertFalse(
            Controller.has_post_endpoint_barrier_capability(["Waiting\n"]))
        self.assertEqual(
            Controller.control_after_endpoints_command("tie.3", 123),
            "control-after-endpoints\ttie.3\t123",
        )

    def test_background_command_uses_per_lane_bytes_and_default_chunks(self):
        command = Controller.background_transfer_command(
            "restore-1", 25, 123456, [(8, 0), (9, 1)])

        self.assertEqual(
            command,
            "background-transfer\trestore-1\t25\t123456\t67108864\t8>0,9>1")

    def test_control_command_and_transfer_command_reject_ambiguous_fields(self):
        with self.assertRaisesRegex(ValueError, "event_id"):
            Controller.control_at_command("contains tab\t", 1)
        with self.assertRaisesRegex(ValueError, "event_id"):
            Controller.control_after_endpoints_command("contains tab\t", 1)
        with self.assertRaisesRegex(ValueError, "source and destination"):
            Controller.background_transfer_command(
                "job", 0, 1, [(2, 2)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            Controller.background_transfer_command(
                "job", 0, 1, [(2, 3), (2, 3)])

    def test_hbf_background_command_preserves_exact_decode_route(self):
        d_route = [
            f"node:0:instance:1:rank:{rank}:hbf-pcie-down"
            for rank in range(8, 12)
        ] + ["node:0:hbf-root-down", "hbf-card:0:pcie-down"]
        command = Controller.hbf_background_command("flush.1", 25, [{
            "id": "pcie:0",
            "runtime_ns": 10,
            "tensor_bytes": 4096,
            "resources": d_route,
            "deps": [],
        }, {
            "id": "write:0",
            "runtime_ns": 20,
            "tensor_bytes": 4096,
            "resources": ["hbf-card:0:write"],
            "deps": ["pcie:0"],
        }])

        prefix, job_id, arrival, encoded = command.split("\t")
        descriptor = __import__("json").loads(encoded)
        self.assertEqual((prefix, job_id, arrival),
                         ("hbf-background", "flush.1", "25"))
        self.assertEqual(
            descriptor["stages"][0]["resources"], d_route)
        self.assertNotIn("rank:0", encoded)
        self.assertNotIn("rank:7", encoded)

    def test_hbf_background_command_rejects_implicit_or_cyclic_routes(self):
        with self.assertRaisesRegex(ValueError, "resources"):
            Controller.hbf_background_command("flush", 0, [{
                "id": "write", "runtime_ns": 1, "tensor_bytes": 1,
                "resources": [], "deps": [],
            }])
        with self.assertRaisesRegex(ValueError, "cycle"):
            Controller.hbf_background_command("flush", 0, [{
                "id": "a", "runtime_ns": 1, "tensor_bytes": 1,
                "resources": ["hbf-card:0:write"], "deps": ["b"],
            }, {
                "id": "b", "runtime_ns": 1, "tensor_bytes": 1,
                "resources": ["hbf-card:0:write"], "deps": ["a"],
            }])

    def test_exact_control_schedule_never_duplicates_or_backdates(self):
        schedule = ExactControlSchedule()

        command = schedule.arm(100, 10)

        self.assertEqual(command, "control-at\tpython-ready.0\t100")
        self.assertIsNone(schedule.arm(100, 10))
        self.assertIsNone(schedule.arm(10, 10))
        self.assertEqual(schedule.next_pending_time(), 100)
        self.assertTrue(schedule.complete("python-ready.0", 100))
        self.assertFalse(schedule.complete("python-ready.0", 100))
        self.assertFalse(schedule.has_pending())

    def test_same_time_barrier_arms_at_current_time_and_deduplicates(self):
        barrier = SameTimeControlBarrier()

        command = barrier.arm(17)

        self.assertEqual(
            command,
            "control-after-endpoints\tpython-tie.0\t17",
        )
        self.assertIsNone(barrier.arm(17))
        self.assertTrue(barrier.has_pending())
        self.assertEqual(barrier.pending_time(), 17)
        self.assertTrue(barrier.owns("python-tie.0"))
        self.assertFalse(barrier.owns("python-ready.0"))
        self.assertTrue(barrier.complete("python-tie.0", 17))
        self.assertFalse(barrier.complete("python-tie.0", 17))
        self.assertFalse(barrier.has_pending())

    def test_same_time_barrier_rejects_cross_time_and_metadata_drift(self):
        barrier = SameTimeControlBarrier()
        barrier.arm(4)
        with self.assertRaisesRegex(
                RuntimeError, "another timestamp"):
            barrier.arm(5)
        with self.assertRaisesRegex(
                RuntimeError, "timestamp changed"):
            barrier.complete("python-tie.0", 5)
        with self.assertRaisesRegex(RuntimeError, "Unknown"):
            SameTimeControlBarrier().complete("missing", 0)

    def test_write_flush_prefixes_auxiliary_commands_atomically(self):
        stream = io.StringIO()
        process = SimpleNamespace(stdin=stream)
        controller = Controller(1)
        controller.set_auxiliary_command_provider(
            lambda primary: [
                "control-at\te.0\t10",
                "background-transfer\tj.0\t10\t1\t1\t0>1",
            ] if primary == "continue" else [])

        controller.write_flush(process, "continue")

        self.assertEqual(
            stream.getvalue(),
            "control-at\te.0\t10\n"
            "background-transfer\tj.0\t10\t1\t1\t0>1\n"
            "continue\n",
        )


if __name__ == "__main__":
    unittest.main()

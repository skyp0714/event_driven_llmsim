from pathlib import Path
import unittest

from serving.core.gpu_pd_hbm import AtomicPDHBM
from serving.core.gpu_pd_latency import load_p4d4_gpu_config


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_CONFIG = (
    REPO_ROOT / "configs" / "wakekv_hbf"
    / "p4d4_gpu_server.json"
)


class AtomicPDHBMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = load_p4d4_gpu_config(GPU_CONFIG)
        cls.block_bytes = (
            cls.hardware.kv_capacity_bytes_per_rank(1))

    def manager(self, *, p_blocks=8, d_blocks=8, node_id=0):
        return AtomicPDHBM(
            hardware=self.hardware,
            node_id=node_id,
            p_capacity_bytes_per_rank=p_blocks * self.block_bytes,
            d_capacity_bytes_per_rank=d_blocks * self.block_bytes,
        )

    def test_admission_reserves_block_rounded_input_and_final(self):
        manager = self.manager()
        admission = manager.try_admit(
            session_id="s",
            input_tokens=17,
            output_tokens=5,
        )
        self.assertIsNotNone(admission)
        self.assertEqual(
            admission.p_bytes_per_rank,
            2 * self.block_bytes,
        )
        self.assertEqual(
            admission.d_target_bytes_per_rank,
            2 * self.block_bytes,
        )
        self.assertEqual(
            manager.p_used_bytes_per_rank,
            2 * self.block_bytes,
        )
        self.assertEqual(
            manager.d_used_bytes_per_rank,
            2 * self.block_bytes,
        )

    def test_p_capacity_failure_does_not_mutate_d(self):
        manager = self.manager(p_blocks=2, d_blocks=8)
        head = manager.try_admit(
            session_id="head",
            input_tokens=17,
            output_tokens=1,
        )
        self.assertIsNotNone(head)
        before = manager.report()
        deferred = manager.try_admit(
            session_id="tail",
            input_tokens=1,
            output_tokens=1,
        )
        self.assertIsNone(deferred)
        after = manager.report()
        self.assertEqual(
            after["p_by_session"], before["p_by_session"])
        self.assertEqual(
            after["d_by_session"], before["d_by_session"])

    def test_d_capacity_failure_does_not_mutate_p(self):
        manager = self.manager(p_blocks=8, d_blocks=2)
        head = manager.try_admit(
            session_id="head",
            input_tokens=17,
            output_tokens=1,
        )
        self.assertIsNotNone(head)
        before = manager.report()
        deferred = manager.try_admit(
            session_id="tail",
            input_tokens=1,
            output_tokens=1,
        )
        self.assertIsNone(deferred)
        after = manager.report()
        self.assertEqual(
            after["p_by_session"], before["p_by_session"])
        self.assertEqual(
            after["d_by_session"], before["d_by_session"])

    def test_restore_wait_holds_both_reservations(self):
        manager = self.manager(p_blocks=2, d_blocks=2)
        admission = manager.try_admit(
            session_id="slow-restore",
            input_tokens=17,
            output_tokens=1,
        )
        self.assertIsNotNone(admission)
        self.assertIsNone(manager.try_admit(
            session_id="blocked",
            input_tokens=1,
            output_tokens=1,
        ))
        self.assertEqual(
            manager.p_used_bytes_per_rank, 2 * self.block_bytes)
        self.assertEqual(
            manager.d_used_bytes_per_rank, 2 * self.block_bytes)

    def test_release_p_after_handoff_keeps_d_for_successor(self):
        manager = self.manager()
        admission = manager.try_admit(
            session_id="s",
            input_tokens=17,
            output_tokens=5,
        )
        manager.release_p(admission)
        self.assertEqual(manager.p_bytes("s"), 0)
        self.assertEqual(
            manager.d_bytes("s"), 2 * self.block_bytes)
        manager.finish(admission, has_successor=True)
        self.assertEqual(manager.p_bytes("s"), 0)
        self.assertEqual(
            manager.d_bytes("s"), 2 * self.block_bytes)
        self.assertIsNone(manager.active_admission("s"))

    def test_terminal_turn_releases_both_sides(self):
        manager = self.manager()
        admission = manager.try_admit(
            session_id="s",
            input_tokens=17,
            output_tokens=5,
        )
        manager.finish(admission, has_successor=False)
        self.assertEqual(manager.p_used_bytes_per_rank, 0)
        self.assertEqual(manager.d_used_bytes_per_rank, 0)

    def test_context_shrink_holds_old_d_until_source_release(self):
        manager = self.manager(d_blocks=8)
        first = manager.try_admit(
            session_id="s",
            input_tokens=49,
            output_tokens=1,
        )
        manager.finish(first, has_successor=True)
        self.assertEqual(manager.d_bytes("s"), 4 * self.block_bytes)
        second = manager.try_admit(
            session_id="s",
            input_tokens=1,
            output_tokens=1,
        )
        self.assertEqual(
            second.prior_d_bytes_per_rank,
            4 * self.block_bytes,
        )
        self.assertEqual(
            second.d_target_bytes_per_rank,
            self.block_bytes,
        )
        self.assertEqual(
            manager.d_bytes("s"),
            4 * self.block_bytes,
        )
        manager.release_d_source(second)
        self.assertEqual(manager.d_bytes("s"), self.block_bytes)

    def test_cancel_restores_prior_d_and_releases_p(self):
        manager = self.manager(d_blocks=8)
        first = manager.try_admit(
            session_id="s",
            input_tokens=17,
            output_tokens=1,
        )
        manager.finish(first, has_successor=True)
        prior = manager.d_bytes("s")
        second = manager.try_admit(
            session_id="s",
            input_tokens=49,
            output_tokens=1,
        )
        self.assertGreater(manager.d_bytes("s"), prior)
        manager.cancel(second)
        self.assertEqual(manager.p_bytes("s"), 0)
        self.assertEqual(manager.d_bytes("s"), prior)

    def test_terminal_output_one_can_skip_d_reservation(self):
        manager = self.manager(d_blocks=1)
        admission = manager.try_admit(
            session_id="s",
            input_tokens=17,
            output_tokens=1,
            needs_d=False,
        )
        self.assertEqual(admission.d_target_bytes_per_rank, 0)
        self.assertEqual(manager.d_used_bytes_per_rank, 0)
        manager.finish(admission, has_successor=False)
        self.assertEqual(manager.p_used_bytes_per_rank, 0)

    def test_individually_infeasible_request_fails_atomically(self):
        manager = self.manager(p_blocks=1, d_blocks=1)
        with self.assertRaisesRegex(RuntimeError, "infeasible"):
            manager.try_admit(
                session_id="s",
                input_tokens=17,
                output_tokens=1,
            )
        self.assertEqual(manager.p_used_bytes_per_rank, 0)
        self.assertEqual(manager.d_used_bytes_per_rank, 0)
        self.assertEqual(manager.metrics.infeasible_requests, 1)

    def test_duplicate_active_session_is_rejected(self):
        manager = self.manager()
        admission = manager.try_admit(
            session_id="s",
            input_tokens=1,
            output_tokens=1,
        )
        with self.assertRaisesRegex(RuntimeError, "active admission"):
            manager.try_admit(
                session_id="s",
                input_tokens=1,
                output_tokens=1,
            )
        manager.cancel(admission)

    def test_node_managers_are_capacity_isolated(self):
        node0 = self.manager(p_blocks=1, d_blocks=1, node_id=0)
        node1 = self.manager(p_blocks=1, d_blocks=1, node_id=1)
        self.assertIsNotNone(node0.try_admit(
            session_id="a",
            input_tokens=1,
            output_tokens=1,
        ))
        self.assertIsNotNone(node1.try_admit(
            session_id="b",
            input_tokens=1,
            output_tokens=1,
        ))
        node0.assert_invariants()
        node1.assert_invariants()


if __name__ == "__main__":
    unittest.main()

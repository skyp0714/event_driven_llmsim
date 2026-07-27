import json
import tempfile
import unittest
from pathlib import Path

from serving.core.online_latency_model import (
    H100_QWEN3_TP4_KERNEL_CALIBRATED,
)
from serving.core.request import Batch, Request
from serving.core.trace_generator import generate_trace


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PLACEMENT = {
    "default": {
        "weights": "LOCAL",
        "kv_loc": "LOCAL",
        "kv_evict_loc": "REMOTE:0",
    },
    "layer": {},
    "block": [],
}


def _prefill_batch(batch_id=1, recompute_tokens=1_024):
    request = Request(batch_id, MODEL, 1_024, 1_025, 0, 0)
    request.agentic_kv_recompute_tokens = recompute_tokens
    batch = Batch(
        batch_id, MODEL, 1_024, 0, [1_024], [], 1, 0,
        [1_024], [0], [], 0, 0,
    )
    batch.requests = [request]
    return batch


def _generate(batch, root, routing="BALANCED", interleave=False):
    return generate_trace(
        batch,
        "H100",
        4,
        1,
        4,
        4,
        "prefill",
        0,
        0,
        131_072,
        64,
        PLACEMENT,
        False,
        routing,
        False,
        False,
        None,
        None,
        interleave,
        16,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        tp_dim=[True],
        ep_dim=[True],
        inputs_root=root,
        latency_model=H100_QWEN3_TP4_KERNEL_CALIBRATED,
    )


class OnlineTraceCalibrationTests(unittest.TestCase):
    def test_real_trace_path_emits_comp_and_keeps_astra_collectives(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = _prefill_batch()
            attribution = _generate(batch, directory)
            path = (
                Path(directory) / "trace" / "H100" / MODEL
                / "instance0_batch1.txt"
            )
            self.assertTrue(path.is_file())
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertTrue(any("ALLREDUCE" in line for line in lines))
            self.assertTrue(any("ALLGATHER" in line for line in lines))
            self.assertTrue(any("REDUCESCATTER" in line for line in lines))

            comp_times = []
            for line in lines[3:]:
                fields = line.split()
                if not fields or fields[0] in {"EXPERT", "PIM"}:
                    continue
                comp_times.append(int(fields[1]))
            self.assertTrue(comp_times)
            self.assertTrue(all(value > 0 for value in comp_times))
            self.assertLess(max(comp_times), 3_600 * 1_000_000_000)
            self.assertFalse(attribution["collectives_included"])
            self.assertEqual(
                attribution["emitted_comp_node_sum_ns"],
                sum(comp_times),
            )
            self.assertEqual(
                batch.model_compute_ns,
                attribution["modeled_comp_critical_path_ns"],
            )
            self.assertEqual(
                batch.recompute_model_compute_ns,
                attribution["recompute_marginal_comp_ns"],
            )
            self.assertLessEqual(
                batch.recompute_model_compute_ns,
                batch.model_compute_ns,
            )
            self.assertEqual(
                attribution["recompute_query_tokens"], 1_024
            )
            self.assertGreater(
                attribution["recompute_marginal_comp_ns"], 0
            )

    def test_active_prefill_replay_frontier_drives_provider_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = _prefill_batch(recompute_tokens=0)
            request = batch.requests[0]
            request.active_prefill_recompute_frontier_tokens = 512

            attribution = _generate(batch, directory)

            self.assertEqual(attribution["recompute_query_tokens"], 512)
            # Under the v2 semantics the counterfactual half-size kernel
            # carries a larger partial-wave penalty, so the clamped
            # critical-path marginal can legitimately reach zero at this
            # shape.  The node-sum marginal still attributes the frontier.
            self.assertGreater(
                attribution["recompute_marginal_comp_node_sum_ns"], 0)
            self.assertGreaterEqual(
                attribution["recompute_marginal_comp_ns"], 0)
            self.assertLessEqual(
                attribution["recompute_marginal_comp_ns"],
                attribution["modeled_comp_critical_path_ns"],
            )

    def test_provenance_is_deterministic_and_bound_to_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            _generate(_prefill_batch(1), directory)
            provenance_path = (
                Path(directory) / "trace" / "H100" / MODEL
                / "online_latency_provenance.json"
            )
            first = provenance_path.read_bytes()
            _generate(_prefill_batch(2), directory)
            self.assertEqual(first, provenance_path.read_bytes())
            metadata = json.loads(first)
            self.assertEqual(
                metadata["name"],
                H100_QWEN3_TP4_KERNEL_CALIBRATED,
            )
            self.assertEqual(len(metadata["source_sha256"]), 4)

    def test_unsupported_moe_routing_fails_before_trace_emission(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "BALANCED"):
                _generate(_prefill_batch(), directory, routing="RAND")

    def test_sub_batch_interleaving_fails_instead_of_losing_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "sub-batch interleaving"):
                _generate(
                    _prefill_batch(), directory, interleave=True)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from serving.core.hbf_comparison_workload import (
    FIXED_COHORT_SUMMARY,
    FIXED_SOURCE_INDICES,
    TRACELAB_SCHEMA3_SHA256,
    WorkloadValidationError,
    build_offered_plan,
    call_full_drain_hashes,
    full_drain_hashes,
    load_comparison_workload,
    load_fixed_comparison_workload,
    session_full_drain_hashes,
)


TRACE_PATH = (
    Path.home() / "llmsim-data/tracelab-schema3-sps0.2-final.jsonl"
)


def _row(session_id: str, calls: list[dict], arrival: int = 0) -> dict:
    return {
        "session_id": session_id,
        "arrival_time_ns": arrival,
        "trace_metadata": {
            "source_session_identity_sha256": "a" * 64,
        },
        "sub_requests": calls,
    }


def _call(
        input_tokens: int,
        output_tokens: int,
        prefix_tokens: int,
        *,
        tool_duration_ns: int = 0,
        newly_append_tokens: int | None = None,
) -> dict:
    result = {
        "input_toks": input_tokens,
        "output_toks": output_tokens,
        "tool_duration_ns": tool_duration_ns,
        "prefix_reuse_toks": prefix_tokens,
        "lineage_status": (
            "session_start" if prefix_tokens == 0 else "adjacent"
        ),
        "inter_turn_gap_type": "tool",
    }
    if newly_append_tokens is not None:
        result["newly_append_toks"] = newly_append_tokens
    return result


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as output:
        for row in rows:
            payload = (
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
            output.write(payload)
            digest.update(payload)
    return digest.hexdigest()


class ComparisonWorkloadTest(unittest.TestCase):

    def test_loader_uses_prefix_as_sole_fresh_work_definition(self):
        rows = [
            _row("skip", [_call(5, 1, 0)]),
            _row("s1", [
                _call(100, 3, 0, newly_append_tokens=7),
                _call(130, 1, 90, newly_append_tokens=999_999),
            ], arrival=12),
            _row("s2", [_call(40, 2, 0)], arrival=30),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            source_hash = _write_jsonl(path, rows)
            workload = load_comparison_workload(
                path,
                source_indices=(1, 2),
                expected_source_sha256=source_hash,
                expected_source_session_count=3,
            )

        first, resume = workload.sessions[0].calls
        self.assertEqual(first.fresh_input_tokens, 100)
        self.assertEqual(resume.cached_prefix_tokens, 90)
        self.assertEqual(resume.fresh_input_tokens, 40)
        self.assertFalse(resume.tpot_eligible)
        self.assertEqual(workload.summary.single_output_call_count, 1)
        self.assertEqual(workload.summary.tpot_eligible_call_count, 2)
        self.assertEqual(workload.summary.total_fresh_input_tokens, 180)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            resume.fresh_input_tokens = 999

    def test_loader_rejects_invalid_prefix_and_missing_index(self):
        rows = [_row("bad", [_call(10, 1, 11)])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            _write_jsonl(path, rows)
            with self.assertRaisesRegex(
                    WorkloadValidationError, "exceeds input_toks"):
                load_comparison_workload(path, source_indices=(0,))
            with self.assertRaisesRegex(
                    WorkloadValidationError, "missing source indices"):
                load_comparison_workload(path, source_indices=(2,))

    def test_loader_requires_sorted_unique_indices(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            load_comparison_workload(
                "unused.jsonl", source_indices=(2, 1)
            )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            load_comparison_workload(
                "unused.jsonl", source_indices=(1, 1)
            )

    def test_offered_plan_reuses_order_and_unit_draws_across_rates(self):
        rows = [
            _row(f"s{index}", [_call(10 + index, 2, 0)])
            for index in range(8)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            _write_jsonl(path, rows)
            workload = load_comparison_workload(
                path, source_indices=tuple(range(8))
            )

        plan = build_offered_plan(workload.sessions, seed=91)
        again = build_offered_plan(workload.sessions, seed=91)
        other = build_offered_plan(workload.sessions, seed=92)
        self.assertEqual(plan, again)
        self.assertNotEqual(
            plan.offered_session_ids_sha256,
            other.offered_session_ids_sha256,
        )
        slow = plan.at_rate(1.0, start_time_ns=100)
        fast = plan.at_rate(2.0, start_time_ns=100)
        self.assertEqual(
            [row.session.session_id for row in slow],
            [row.session.session_id for row in fast],
        )
        self.assertEqual(
            [row.unit_interarrival for row in slow],
            [row.unit_interarrival for row in fast],
        )
        self.assertTrue(all(
            abs(
                (slow_row.arrival_time_ns - 100)
                - 2 * (fast_row.arrival_time_ns - 100)
            ) <= 1
            for slow_row, fast_row in zip(slow, fast)
        ))
        self.assertEqual(slow[0].arrival_time_ns, 100)

    def test_full_drain_hashes_separate_set_from_completion_order(self):
        expected = ("s1", "s2", "s3")
        forward = full_drain_hashes(expected, expected)
        reverse = full_drain_hashes(expected, reversed(expected))
        self.assertEqual(
            forward.expected_set_sha256, reverse.completion_set_sha256
        )
        self.assertEqual(
            forward.completion_set_sha256, reverse.completion_set_sha256
        )
        self.assertNotEqual(
            forward.completion_order_sha256,
            reverse.completion_order_sha256,
        )
        with self.assertRaisesRegex(
                WorkloadValidationError, "did not fully drain"):
            full_drain_hashes(expected, ("s1", "s2"))
        with self.assertRaisesRegex(
                WorkloadValidationError, "duplicates"):
            full_drain_hashes(expected, ("s1", "s2", "s2"))

    def test_session_and_call_full_drain_wrappers(self):
        rows = [
            _row("s1", [_call(10, 2, 0), _call(12, 1, 10)]),
            _row("s2", [_call(20, 3, 0)]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            _write_jsonl(path, rows)
            workload = load_comparison_workload(
                path, source_indices=(0, 1)
            )
        plan = build_offered_plan(
            workload.sessions, seed=3, shuffle=False
        )
        session_audit = session_full_drain_hashes(plan, ("s2", "s1"))
        call_audit = call_full_drain_hashes(
            workload, reversed(workload.call_completion_identities)
        )
        self.assertEqual(session_audit.identity_count, 2)
        self.assertEqual(call_audit.identity_count, 3)

    @unittest.skipUnless(TRACE_PATH.exists(), "TraceLab release not present")
    def test_fixed_trace_contract(self):
        workload = load_fixed_comparison_workload(TRACE_PATH)
        self.assertEqual(workload.source_sha256, TRACELAB_SCHEMA3_SHA256)
        self.assertEqual(
            tuple(session.source_index for session in workload.sessions),
            FIXED_SOURCE_INDICES,
        )
        self.assertEqual(workload.summary, FIXED_COHORT_SUMMARY)
        self.assertEqual(workload.summary.session_count, 32)
        self.assertEqual(workload.summary.call_count, 2680)
        self.assertEqual(workload.summary.first_turn_count, 32)
        self.assertEqual(workload.summary.resume_count, 2648)
        self.assertEqual(
            workload.summary.adjacent_cached_resume_count, 2597
        )
        self.assertEqual(
            workload.summary.context_shrink_resume_count, 51
        )
        self.assertEqual(
            workload.summary.total_input_tokens, 409_094_011
        )
        self.assertEqual(
            workload.summary.total_cached_prefix_tokens, 398_757_236
        )
        self.assertEqual(
            workload.summary.resume_fresh_input_tokens, 9_582_453
        )
        self.assertEqual(workload.summary.total_output_tokens, 1_396_785)
        self.assertEqual(workload.summary.single_output_call_count, 29)
        self.assertEqual(
            workload.summary.max_input_context_tokens, 415_963
        )
        self.assertEqual(workload.summary.max_sequence_tokens, 420_339)
        self.assertEqual(
            workload.summary.selected_session_ids_sha256,
            "985d7fff295973f3a1a6d15f7c847455ddd54585f28a8656c904e9749f1b6eca",
        )
        self.assertEqual(
            workload.summary.source_index_session_id_sha256,
            "b1f47d17a50d2a68008a67bd1c14e797b0a1bd6105c913701eeda0068eab9573",
        )


if __name__ == "__main__":
    unittest.main()

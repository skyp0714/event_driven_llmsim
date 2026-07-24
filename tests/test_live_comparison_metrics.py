import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from serving.core.hbf_comparison_workload import (
    CallSpec,
    ScheduledSession,
    SessionSpec,
)
from serving.core.live_comparison_metrics import (
    LiveComparisonMetricsError,
    compute_live_comparison_metrics,
    expected_request_identities,
    materialize_scheduled_sessions,
    parse_serving_requests_csv,
)


CSV_FIELDS = (
    "instance id",
    "request id",
    "model",
    "input",
    "output",
    "generated_tokens",
    "arrival",
    "end_time",
    "latency",
    "TTFT",
    "TPOT",
    "session_id",
    "sub_request_index",
)


def _call(
        session_id,
        source_index,
        call_index,
        *,
        input_tokens,
        output_tokens,
        cached_prefix_tokens,
        tool_duration_ns=0,
        lineage_status=None,
        gap_type=None,
):
    return CallSpec(
        session_id=session_id,
        source_index=source_index,
        call_index=call_index,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_duration_ns=tool_duration_ns,
        cached_prefix_tokens=cached_prefix_tokens,
        fresh_input_tokens=input_tokens - cached_prefix_tokens,
        lineage_status=lineage_status,
        inter_turn_gap_type=gap_type,
    )


def _scheduled(
        session_id,
        source_index,
        offer_index,
        arrival_ns,
        unit_interarrival,
        unit_arrival,
        calls,
):
    return ScheduledSession(
        offer_index=offer_index,
        session=SessionSpec(
            source_index=source_index,
            session_id=session_id,
            source_arrival_time_ns=source_index * 10,
            source_session_identity_sha256=(
                f"{source_index + 1:064x}"),
            calls=tuple(calls),
        ),
        arrival_time_ns=arrival_ns,
        unit_interarrival=unit_interarrival,
        unit_arrival_time=unit_arrival,
    )


def _schedule():
    first = _scheduled(
        "session-a",
        0,
        4,
        100,
        0.125,
        0.125,
        (
            _call(
                "session-a",
                0,
                0,
                input_tokens=10,
                output_tokens=2,
                cached_prefix_tokens=0,
            ),
            _call(
                "session-a",
                0,
                1,
                input_tokens=20,
                output_tokens=3,
                cached_prefix_tokens=10,
                tool_duration_ns=5,
                lineage_status="adjacent",
                gap_type="tool",
            ),
        ),
    )
    second = _scheduled(
        "session-b",
        1,
        7,
        200,
        0.375,
        0.5,
        (
            _call(
                "session-b",
                1,
                0,
                input_tokens=8,
                output_tokens=1,
                cached_prefix_tokens=0,
                lineage_status="root",
            ),
            _call(
                "session-b",
                1,
                1,
                input_tokens=12,
                output_tokens=2,
                cached_prefix_tokens=8,
                tool_duration_ns=5,
                gap_type="tool",
            ),
        ),
    )
    return (first, second)


def _csv_row(
        session_id,
        call_index,
        *,
        arrival,
        ttft,
        completion,
        output,
):
    latency = completion - arrival
    tpot = (
        0 if output == 1
        else (latency - ttft) // (output - 1)
    )
    return {
        "instance id": "0",
        "request id": f"{session_id}-{call_index}",
        "model": "test",
        "input": "1",
        "output": str(output),
        "generated_tokens": str(output),
        "arrival": str(arrival),
        "end_time": str(completion),
        "latency": str(latency),
        "TTFT": str(ttft),
        "TPOT": str(tpot),
        "session_id": session_id,
        "sub_request_index": str(call_index),
    }


def _valid_rows():
    return (
        _csv_row(
            "session-b", 1, arrival=240, ttft=15,
            completion=275, output=2),
        _csv_row(
            "session-a", 0, arrival=100, ttft=10,
            completion=120, output=2),
        _csv_row(
            "session-b", 0, arrival=200, ttft=35,
            completion=235, output=1),
        _csv_row(
            "session-a", 1, arrival=125, ttft=20,
            completion=165, output=3),
    )


def _write_csv(path, rows, fields=CSV_FIELDS):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class WorkloadMaterializationTest(unittest.TestCase):
    def test_materializes_lossless_atomic_jsonl_and_sha(self):
        schedule = _schedule()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "workload.jsonl"
            artifact = materialize_scheduled_sessions(
                schedule,
                path,
                source_sha256="f" * 64,
            )

            payload = path.read_bytes()
            rows = [
                json.loads(line)
                for line in payload.decode("utf-8").splitlines()
            ]
            self.assertEqual(artifact.path, path)
            self.assertEqual(
                artifact.sha256, hashlib.sha256(payload).hexdigest())
            self.assertEqual(artifact.byte_count, len(payload))
            self.assertEqual(artifact.session_count, 2)
            self.assertEqual(artifact.request_count, 4)
            self.assertEqual(
                artifact.request_identities,
                (
                    ("session-a", 0),
                    ("session-a", 1),
                    ("session-b", 0),
                    ("session-b", 1),
                ),
            )
            self.assertEqual(rows[0]["session_id"], "session-a")
            self.assertEqual(rows[0]["arrival_time_ns"], 100)
            metadata = rows[0]["trace_metadata"]
            self.assertEqual(metadata["source_sha256"], "f" * 64)
            self.assertEqual(metadata["source_index"], 0)
            self.assertEqual(metadata["source_arrival_time_ns"], 0)
            self.assertEqual(metadata["offer_index"], 4)
            self.assertEqual(
                float.fromhex(metadata["unit_interarrival_hex"]), 0.125)
            self.assertEqual(
                float.fromhex(metadata["unit_arrival_time_hex"]), 0.125)
            first_call, resume_call = rows[0]["sub_requests"]
            self.assertEqual(
                first_call,
                {
                    "input_toks": 10,
                    "output_toks": 2,
                    "tool_duration_ns": 0,
                    "prefix_reuse_toks": 0,
                },
            )
            self.assertEqual(resume_call["lineage_status"], "adjacent")
            self.assertEqual(resume_call["inter_turn_gap_type"], "tool")
            self.assertFalse(any(
                child.name.endswith(".tmp")
                for child in path.parent.iterdir()
            ))

    def test_rejects_non_tuple_schedule_and_bad_source_hash(self):
        schedule = _schedule()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.jsonl"
            with self.assertRaisesRegex(
                    LiveComparisonMetricsError, "immutable tuple"):
                materialize_scheduled_sessions(
                    list(schedule),
                    path,
                    source_sha256="f" * 64,
                )
            with self.assertRaisesRegex(
                    LiveComparisonMetricsError, "lowercase SHA-256"):
                materialize_scheduled_sessions(
                    schedule,
                    path,
                    source_sha256="not-a-digest",
                )


class LiveRequestCsvTest(unittest.TestCase):
    def test_parses_native_csv_and_validates_exact_roster(self):
        schedule = _schedule()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.csv"
            _write_csv(path, _valid_rows())
            requests = parse_serving_requests_csv(
                path,
                expected_identities=expected_request_identities(schedule),
            )

        self.assertEqual(len(requests), 4)
        indexed = {request.identity: request for request in requests}
        self.assertEqual(
            indexed[("session-a", 1)].first_token_ns, 145)
        self.assertEqual(
            indexed[("session-a", 1)].tpot_ns, Fraction(10))
        self.assertIsNone(indexed[("session-b", 0)].tpot_ns)

    def test_fails_closed_on_duplicate_missing_and_unexpected_rows(self):
        schedule = _schedule()
        expected = expected_request_identities(schedule)
        cases = {
            "duplicate": _valid_rows() + (_valid_rows()[0],),
            "missing": _valid_rows()[:-1],
            "unexpected": _valid_rows() + (
                _csv_row(
                    "other", 0, arrival=1, ttft=1,
                    completion=2, output=1),
            ),
        }
        patterns = {
            "duplicate": "duplicate identity",
            "missing": "missing expected identities",
            "unexpected": "unexpected identity",
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, rows in cases.items():
                with self.subTest(name=name):
                    path = Path(directory) / f"{name}.csv"
                    _write_csv(path, rows)
                    with self.assertRaisesRegex(
                            LiveComparisonMetricsError, patterns[name]):
                        parse_serving_requests_csv(
                            path, expected_identities=expected)

    def test_fails_closed_on_inconsistent_native_timing(self):
        schedule = _schedule()
        expected = expected_request_identities(schedule)
        rows = [dict(row) for row in _valid_rows()]
        rows[0]["TPOT"] = "19"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.csv"
            _write_csv(path, rows)
            with self.assertRaisesRegex(
                    LiveComparisonMetricsError, "inconsistent TPOT"):
                parse_serving_requests_csv(
                    path, expected_identities=expected)


class LiveMetricsTest(unittest.TestCase):
    def test_computes_exact_distributions_and_operational_goodput(self):
        schedule = _schedule()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.csv"
            _write_csv(path, _valid_rows())
            requests = parse_serving_requests_csv(
                path,
                expected_identities=expected_request_identities(schedule),
            )
        metrics = compute_live_comparison_metrics(
            schedule,
            requests,
            measurement_session_ids=("session-a", "session-b"),
            ttft_slo_ns=30,
            tpot_slo_ns=15,
        )

        self.assertEqual(metrics.measurement_request_count, 4)
        self.assertEqual(metrics.resume_request_count, 2)
        self.assertEqual(metrics.tpot_eligible_request_count, 3)
        self.assertEqual(metrics.resume_tpot_eligible_request_count, 2)
        self.assertEqual(metrics.resume_ttft_ns.count, 2)
        self.assertEqual(metrics.resume_ttft_ns.mean_ns, 17.5)
        self.assertEqual(metrics.resume_ttft_ns.p50_ns, 15.0)
        self.assertEqual(metrics.resume_ttft_ns.p95_ns, 20.0)
        self.assertAlmostEqual(metrics.tpot_ns.mean_ns, 40 / 3)
        self.assertEqual(metrics.resume_tpot_ns.mean_ns, 15.0)
        self.assertEqual(metrics.joint_slo_pass_count, 2)
        self.assertEqual(metrics.joint_slo_fail_count, 2)
        self.assertEqual(metrics.resume_joint_slo_pass_count, 1)
        self.assertEqual(metrics.resume_joint_slo_fail_count, 1)
        self.assertEqual(metrics.joint_slo_pass_output_tokens, 5)
        self.assertEqual(metrics.joint_slo_pass_session_count, 1)
        self.assertEqual(metrics.joint_slo_fail_session_count, 1)
        self.assertEqual(metrics.window_start_ns, 100)
        self.assertEqual(metrics.window_end_ns, 275)
        self.assertEqual(metrics.window_duration_ns, 175)
        self.assertAlmostEqual(
            metrics.operational_request_goodput_per_second,
            2_000_000_000 / 175,
        )
        self.assertAlmostEqual(
            metrics.operational_resume_goodput_per_second,
            1_000_000_000 / 175,
        )
        self.assertAlmostEqual(
            metrics.operational_token_goodput_per_second,
            5_000_000_000 / 175,
        )
        self.assertAlmostEqual(
            metrics.operational_session_goodput_per_second,
            1_000_000_000 / 175,
        )

    def test_fails_closed_on_unknown_measurement_session(self):
        schedule = _schedule()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.csv"
            _write_csv(path, _valid_rows())
            requests = parse_serving_requests_csv(
                path,
                expected_identities=expected_request_identities(schedule),
            )
        with self.assertRaisesRegex(
                LiveComparisonMetricsError, "unknown measurement"):
            compute_live_comparison_metrics(
                schedule,
                requests,
                measurement_session_ids=("missing",),
            )


if __name__ == "__main__":
    unittest.main()

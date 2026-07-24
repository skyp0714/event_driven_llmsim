from __future__ import annotations

import json
import unittest

from serving.core.hbf_comparison_metrics import RequestKey
from serving.core.hbf_stationary_metrics import (
    CutoffRequestObservation,
    NANOSECONDS_PER_SECOND,
    StationaryMetricError,
    StationaryWindowContract,
    summarize_stationary_cutoff,
)


S = NANOSECONDS_PER_SECOND


def request(
        session_id,
        call_index,
        *,
        release_s,
        output_tokens=2,
        first_offset_s=1.0,
        completion_offset_s=1.1,
):
    release_ns = (
        None if release_s is None else int(round(release_s * S)))
    first_ns = (
        None
        if release_ns is None or first_offset_s is None
        else release_ns + int(round(first_offset_s * S))
    )
    completion_ns = (
        None
        if release_ns is None or completion_offset_s is None
        else release_ns + int(round(completion_offset_s * S))
    )
    if output_tokens == 1 and completion_ns is not None:
        completion_ns = first_ns
    return CutoffRequestObservation(
        key=RequestKey(session_id, call_index),
        output_tokens=output_tokens,
        release_ns=release_ns,
        first_token_ns=first_ns,
        completion_ns=completion_ns,
    )


def three_call_session(
        session_id,
        *,
        base_s,
        measurement_call=None,
        incomplete=False,
        late_completion=False,
):
    rows = []
    for call_index in range(3):
        release_s = base_s + call_index * 10
        output = 1 if call_index == 0 else 2
        completion_offset = 1.1
        first_offset = 1.0
        if call_index == measurement_call and incomplete:
            first_offset = None
            completion_offset = None
        if call_index == measurement_call and late_completion:
            first_offset = 1.0
            completion_offset = 1_400.0
        rows.append(request(
            session_id,
            call_index,
            release_s=release_s,
            output_tokens=output,
            first_offset_s=first_offset,
            completion_offset_s=completion_offset,
        ))
    return rows


class HBFStationaryMetricTests(unittest.TestCase):

    def test_default_window_pins_loaded_guard_and_equal_subwindows(self):
        window = StationaryWindowContract()
        self.assertEqual(window.measurement_duration_seconds, 600.0)
        self.assertEqual(window.guard_duration_ns, 711 * S)
        self.assertGreater(
            window.guard_duration_ns,
            window.max_joint_pass_latency_ns,
        )
        self.assertEqual(
            {end - start for start, end in window.intervals.values()},
            {300 * S},
        )
        with self.assertRaisesRegex(
                StationaryMetricError, "strictly longer"):
            StationaryWindowContract(
                cutoff_ns=2_810_400_000_000)

    def test_release_window_goodput_and_completion_window_are_distinct(self):
        observations = []
        observations.extend(three_call_session(
            "warmup", base_s=1_000))
        observations.extend(three_call_session(
            "measurement", base_s=1_490))
        observations.extend(three_call_session(
            "guard", base_s=2_200))

        summary = summarize_stationary_cutoff(observations)
        measurement = summary["measurement"]
        self.assertEqual(measurement["released_calls"], 2)
        self.assertEqual(measurement["released_first_calls"], 0)
        self.assertEqual(measurement["released_resume_calls"], 2)
        self.assertEqual(measurement["joint_slo_pass_count"], 2)
        self.assertAlmostEqual(
            measurement["joint_slo_request_goodput_per_second"],
            2 / 600,
        )
        self.assertAlmostEqual(
            measurement["joint_slo_output_token_goodput_per_second"],
            4 / 600,
        )
        # The first measurement session call completes before the release
        # window, so only calls 1 and 2 are in both windows here.
        self.assertEqual(
            summary["completion_window_throughput"][
                "completed_calls"],
            2,
        )

    def test_incomplete_request_is_an_exact_failure_not_dropped(self):
        observations = []
        observations.extend(three_call_session(
            "warmup", base_s=1_000))
        observations.extend(three_call_session(
            "measurement",
            base_s=1_490,
            measurement_call=1,
            incomplete=True,
        ))
        summary = summarize_stationary_cutoff(observations)
        measurement = summary["measurement"]
        self.assertEqual(measurement["released_calls"], 2)
        self.assertEqual(measurement["incomplete_at_cutoff"], 1)
        self.assertEqual(measurement["exact_failed_censor_count"], 1)
        self.assertEqual(
            measurement["ambiguous_measurement_censor_count"], 0)
        self.assertEqual(measurement["joint_slo_pass_count"], 1)
        resume = summary["latency"]["resume"]
        self.assertEqual(resume["ttft"]["censored_count"], 1)
        self.assertFalse(resume["ttft"]["p95_publishable"])

    def test_full_drain_timestamps_after_cutoff_are_masked(self):
        partial = []
        partial.extend(three_call_session(
            "warmup", base_s=1_000))
        partial.extend(three_call_session(
            "measurement",
            base_s=1_490,
            measurement_call=1,
            incomplete=True,
        ))
        # A cutoff snapshot preserves an already-observed first token even
        # though the request has not completed.
        partial[4] = request(
            "measurement",
            1,
            release_s=1_500,
            output_tokens=2,
            first_offset_s=1.0,
            completion_offset_s=None,
        )
        full = []
        full.extend(three_call_session(
            "warmup", base_s=1_000))
        full.extend(three_call_session(
            "measurement",
            base_s=1_490,
            measurement_call=1,
            late_completion=True,
        ))
        partial_summary = summarize_stationary_cutoff(partial)
        full_summary = summarize_stationary_cutoff(full)
        for key in (
            "measurement",
            "latency",
            "completion_window_throughput",
            "stationarity_seed_statistics",
            "cutoff_audit",
        ):
            self.assertEqual(partial_summary[key], full_summary[key])

    def test_one_token_request_has_no_tpot_and_deadline_tie_passes(self):
        observations = []
        observations.extend(three_call_session(
            "warmup", base_s=1_000))
        rows = three_call_session(
            "measurement", base_s=1_490)
        release_ns = 1_500 * S
        rows[1] = CutoffRequestObservation(
            key=rows[1].key,
            output_tokens=1,
            release_ns=release_ns,
            first_token_ns=release_ns + 30 * S,
            completion_ns=release_ns + 30 * S,
        )
        observations.extend(rows)
        summary = summarize_stationary_cutoff(observations)
        self.assertEqual(
            summary["measurement"]["joint_slo_pass_count"], 2)
        self.assertEqual(
            summary["latency"]["resume"]["tpot"]["eligible_count"], 1)

    def test_successor_release_roster_is_system_causal(self):
        fast = []
        fast.extend(three_call_session(
            "warmup", base_s=1_000))
        fast.extend(three_call_session(
            "session", base_s=1_490))
        slow = []
        slow.extend(three_call_session(
            "warmup", base_s=1_000))
        slow.extend(three_call_session(
            "session", base_s=1_490))
        slow[4] = request(
            "session",
            1,
            release_s=2_099,
            output_tokens=2,
        )
        slow[5] = request(
            "session",
            2,
            release_s=2_109,
            output_tokens=2,
        )
        fast_summary = summarize_stationary_cutoff(fast)
        slow_summary = summarize_stationary_cutoff(slow)
        self.assertEqual(
            fast_summary["measurement"]["released_first_calls"],
            slow_summary["measurement"]["released_first_calls"],
        )
        self.assertGreater(
            fast_summary["measurement"]["released_resume_calls"],
            slow_summary["measurement"]["released_resume_calls"],
        )

    def test_partial_state_and_session_shape_fail_closed(self):
        with self.assertRaisesRegex(
                StationaryMetricError, "unreleased request"):
            CutoffRequestObservation(
                key=RequestKey("x", 0),
                output_tokens=1,
                release_ns=None,
                first_token_ns=1,
            )
        malformed = three_call_session("x", base_s=1_490)[:2]
        with self.assertRaisesRegex(
                StationaryMetricError, "exactly 3"):
            summarize_stationary_cutoff(malformed)

    def test_summary_is_strict_json_safe_and_exposes_sample_gate(self):
        observations = []
        observations.extend(three_call_session(
            "warmup", base_s=1_000))
        observations.extend(three_call_session(
            "measurement", base_s=1_490))
        summary = summarize_stationary_cutoff(observations)
        self.assertFalse(
            summary["stationarity_seed_statistics"][
                "minimum_sample_gate_pass"])
        self.assertTrue(
            summary["stationarity_seed_statistics"][
                "minimum_sample_violations"])
        json.dumps(summary, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

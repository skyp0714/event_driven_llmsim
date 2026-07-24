import math
from pathlib import Path
import unittest

from serving.core.hbf_comparison_workload import (
    load_fixed_comparison_workload,
)
from serving.core.hbf_comparison_metrics import (
    DEFAULT_SLO_THRESHOLDS,
    ComparisonMetricError,
    CompletedRequest,
    RequestKey,
    SLOThresholds,
    aggregate_paired_seed_values,
    aggregate_seed_values,
    goodput_from_pass_counts,
    joint_slo_pass,
    nearest_rank_percentile,
    slo_sensitivity_grid,
    summarize_completed_requests,
    summarize_distribution,
    validate_full_drain_same_request_ids,
)


NS = 1_000_000_000
MS = 1_000_000
TRACE_PATH = (
    Path.home() / "llmsim-data/tracelab-schema3-sps0.2-final.jsonl"
)


def request(
        session, sub_index, *, release=0, ttft=1 * NS,
        tpot=100 * MS, output=3):
    first = release + ttft
    completion = first + (output - 1) * tpot
    return CompletedRequest(
        key=RequestKey(session, sub_index),
        release_ns=release,
        first_token_ns=first,
        completion_ns=completion,
        output_tokens=output,
    )


class CompletedRequestMetricTests(unittest.TestCase):
    def test_ttft_is_release_to_first_token_and_tpot_excludes_first(self):
        item = request(
            "s", 1, release=7 * NS, ttft=5 * NS,
            tpot=250 * MS, output=5)
        self.assertEqual(item.ttft_ns, 5 * NS)
        self.assertEqual(item.tpot_ns, 250 * MS)
        self.assertTrue(item.is_resume)

    def test_output_one_has_no_tpot(self):
        item = request(
            "s", 1, release=4, ttft=6, tpot=999, output=1)
        self.assertEqual(item.ttft_ns, 6)
        self.assertIsNone(item.tpot_ns)
        self.assertEqual(item.completion_ns, item.first_token_ns)

    def test_timestamp_order_is_validated(self):
        with self.assertRaisesRegex(
                ComparisonMetricError, "first token precedes"):
            CompletedRequest(
                key=RequestKey("s", 0),
                release_ns=10,
                first_token_ns=9,
                completion_ns=10,
                output_tokens=1,
            )
        with self.assertRaisesRegex(
                ComparisonMetricError, "completion precedes"):
            CompletedRequest(
                key=RequestKey("s", 0),
                release_ns=0,
                first_token_ns=10,
                completion_ns=9,
                output_tokens=1,
            )
        with self.assertRaisesRegex(
                ComparisonMetricError, "one-token completion"):
            CompletedRequest(
                key=RequestKey("s", 0),
                release_ns=0,
                first_token_ns=10,
                completion_ns=11,
                output_tokens=1,
            )

    def test_output_one_joint_resume_slo_is_ttft_only(self):
        passing = request(
            "pass", 1, ttft=29 * NS, tpot=10_000 * NS, output=1)
        failing = request(
            "fail", 1, ttft=31 * NS, tpot=1, output=1)
        self.assertTrue(joint_slo_pass(passing))
        self.assertFalse(joint_slo_pass(failing))

    def test_multi_token_joint_resume_requires_ttft_and_tpot(self):
        slow_tpot = request(
            "slow", 1, ttft=10 * NS, tpot=301 * MS, output=2)
        slow_ttft = request(
            "late", 1, ttft=31 * NS, tpot=10 * MS, output=2)
        passing = request(
            "pass", 1, ttft=30 * NS, tpot=300 * MS, output=2)
        self.assertFalse(joint_slo_pass(slow_tpot))
        self.assertFalse(joint_slo_pass(slow_ttft))
        self.assertTrue(joint_slo_pass(passing))


class SLOAndDistributionTests(unittest.TestCase):
    def test_absolute_defaults_and_sensitivity_grid(self):
        self.assertEqual(DEFAULT_SLO_THRESHOLDS, SLOThresholds(
            first_ttft_ns=30 * NS,
            resume_ttft_ns=30 * NS,
            tpot_ns=300 * MS,
        ))
        grid = slo_sensitivity_grid()
        self.assertEqual(len(grid), 9)
        self.assertEqual(
            [(row.resume_ttft_ns // NS, row.tpot_ns // MS)
             for row in grid],
            [
                (30, 100), (30, 300), (30, 600),
                (60, 100), (60, 300), (60, 600),
                (120, 100), (120, 300), (120, 600),
            ],
        )
        self.assertTrue(all(
            row.first_ttft_ns == 30 * NS for row in grid))

    def test_nearest_rank_percentiles_are_exact(self):
        values = list(range(1, 101))
        self.assertEqual(nearest_rank_percentile(values, 0.50), 50)
        self.assertEqual(nearest_rank_percentile(values, 0.95), 95)
        self.assertEqual(nearest_rank_percentile(values, 0.99), 99)
        summary = summarize_distribution(values)
        self.assertEqual(summary.p50, 50)
        self.assertEqual(summary.p95, 95)
        self.assertEqual(summary.p99, 99)
        self.assertEqual(summary.percentile_method, "nearest_rank")

    def test_empty_distribution_has_explicit_none_values(self):
        summary = summarize_distribution([])
        self.assertEqual(summary.count, 0)
        self.assertIsNone(summary.mean)
        self.assertIsNone(summary.p99)

    def test_goodput_uses_offered_session_rate(self):
        result = goodput_from_pass_counts(
            offered_session_rate=2.0,
            offered_session_count=4,
            pass_request_count=10,
            pass_output_tokens=1_000,
        )
        self.assertEqual(result.pass_requests_per_second, 5.0)
        self.assertEqual(result.pass_output_tokens_per_second, 500.0)
        with self.assertRaisesRegex(
                ComparisonMetricError, "finite number"):
            goodput_from_pass_counts(
                offered_session_rate=True,
                offered_session_count=4,
                pass_request_count=10,
                pass_output_tokens=1_000,
            )


class CohortSummaryTests(unittest.TestCase):
    def test_first_resume_split_and_joint_goodput(self):
        requests = [
            request("a", 0, ttft=20 * NS, tpot=100 * MS, output=2),
            request("a", 1, ttft=20 * NS, output=1),
            request("a", 2, ttft=40 * NS, tpot=100 * MS, output=4),
            request("b", 0, ttft=40 * NS, tpot=100 * MS, output=2),
            request("b", 1, ttft=20 * NS, tpot=400 * MS, output=3),
        ]
        summary = summarize_completed_requests(
            requests, offered_session_rate=2.0)

        self.assertEqual(summary.offered_session_count, 2)
        self.assertEqual(summary.first.request_count, 2)
        self.assertEqual(summary.resume.request_count, 3)
        self.assertEqual(summary.first.joint_slo_pass_count, 1)
        self.assertEqual(summary.resume.joint_slo_pass_count, 1)
        self.assertEqual(summary.resume.tpot_eligible_count, 2)
        self.assertEqual(summary.resume.joint_slo_pass_output_tokens, 1)
        self.assertEqual(
            summary.resume.joint_slo_goodput.pass_requests_per_second, 1.0)
        self.assertEqual(
            summary.resume.joint_slo_goodput.pass_output_tokens_per_second,
            1.0,
        )
        self.assertEqual(
            summary.all_requests.joint_slo_goodput.pass_requests_per_second,
            2.0,
        )
        self.assertEqual(
            summary.all_requests.joint_slo_goodput
            .pass_output_tokens_per_second,
            3.0,
        )

    def test_empty_split_is_represented_without_nan(self):
        summary = summarize_completed_requests(
            [request("a", 0)], offered_session_rate=1.0)
        self.assertEqual(summary.resume.request_count, 0)
        self.assertIsNone(summary.resume.ttft_ns.mean)
        self.assertIsNone(summary.resume.joint_slo_attainment)
        self.assertEqual(
            summary.resume.joint_slo_goodput.pass_requests_per_second, 0.0)

    def test_duplicate_request_id_is_rejected(self):
        item = request("a", 0)
        with self.assertRaisesRegex(
                ComparisonMetricError, "duplicate request ID"):
            summarize_completed_requests(
                [item, item], offered_session_rate=1.0)

    @unittest.skipUnless(TRACE_PATH.exists(), "TraceLab release not present")
    def test_fixed_trace_denominators_flow_into_metrics(self):
        workload = load_fixed_comparison_workload(TRACE_PATH)
        completed = [
            CompletedRequest(
                key=RequestKey(
                    call.session_id,
                    call.call_index,
                ),
                release_ns=0,
                first_token_ns=MS,
                completion_ns=(
                    MS + (call.output_tokens - 1) * MS
                ),
                output_tokens=call.output_tokens,
            )
            for session in workload.sessions
            for call in session.calls
        ]
        summary = summarize_completed_requests(
            completed,
            offered_session_rate=1.0,
        )
        self.assertEqual(summary.offered_session_count, 32)
        self.assertEqual(summary.request_count, 2_680)
        self.assertEqual(summary.first.request_count, 32)
        self.assertEqual(summary.resume.request_count, 2_648)
        self.assertEqual(
            summary.all_requests.tpot_eligible_count,
            2_651,
        )
        self.assertEqual(
            summary.resume.joint_slo_goodput.pass_requests_per_second,
            2_648 / 32,
        )


class FullDrainAndPairedAggregationTests(unittest.TestCase):
    def test_full_drain_pairs_by_id_not_completion_order_or_release(self):
        oracle = [
            request("a", 0, release=0),
            request("a", 1, release=100),
            request("b", 0, release=1_000),
        ]
        candidate = [
            request("b", 0, release=2_000),
            request("a", 1, release=9_000),
            request("a", 0, release=0),
        ]
        keys = validate_full_drain_same_request_ids({
            "oracle": oracle,
            "candidate": candidate,
        })
        self.assertEqual(keys, (
            RequestKey("a", 0),
            RequestKey("a", 1),
            RequestKey("b", 0),
        ))

    def test_full_drain_rejects_missing_or_changed_work(self):
        oracle = [request("a", 0), request("a", 1, output=2)]
        with self.assertRaisesRegex(
                ComparisonMetricError, "not a full-drain match"):
            validate_full_drain_same_request_ids({
                "oracle": oracle,
                "candidate": [request("a", 0)],
            })
        with self.assertRaisesRegex(
                ComparisonMetricError, "output_tokens mismatch"):
            validate_full_drain_same_request_ids({
                "oracle": oracle,
                "candidate": [
                    request("a", 0),
                    request("a", 1, output=3),
                ],
            })

    def test_expected_roster_detects_an_id_omitted_by_every_system(self):
        shared_subset = [request("a", 0)]
        with self.assertRaisesRegex(
                ComparisonMetricError, "expected request roster"):
            validate_full_drain_same_request_ids(
                {
                    "oracle": shared_subset,
                    "candidate": shared_subset,
                },
                expected_request_ids=(
                    RequestKey("a", 0),
                    RequestKey("a", 1),
                ),
            )

    def test_seed_aggregate_is_deterministic_and_has_t_interval(self):
        first = aggregate_seed_values({3: 9.0, 1: 1.0, 2: 5.0})
        second = aggregate_seed_values({2: 5.0, 3: 9.0, 1: 1.0})
        self.assertEqual(first, second)
        self.assertEqual(first.seed_ids, (1, 2, 3))
        self.assertEqual(first.values, (1.0, 5.0, 9.0))
        self.assertEqual(first.mean, 5.0)
        self.assertAlmostEqual(first.sample_stddev, 4.0)
        self.assertEqual(first.ci_method, "student_t_95")
        self.assertGreater(first.ci95_half_width, 0.0)

    def test_paired_aggregation_pairs_before_mean(self):
        paired = aggregate_paired_seed_values(
            reference_by_seed={7: 10.0, 3: 20.0},
            candidate_by_seed={3: 10.0, 7: 15.0},
        )
        self.assertEqual(paired.seed_ids, (3, 7))
        self.assertEqual(
            paired.candidate_minus_reference.values, (-10.0, 5.0))
        self.assertEqual(
            paired.candidate_over_reference.values, (0.5, 1.5))
        self.assertEqual(paired.candidate_over_reference.mean, 1.0)

    def test_paired_ratio_is_explicitly_unavailable_on_zero_reference(self):
        paired = aggregate_paired_seed_values(
            reference_by_seed={1: 0.0, 2: 2.0},
            candidate_by_seed={1: 1.0, 2: 4.0},
        )
        self.assertIsNone(paired.candidate_over_reference)
        self.assertIn("seed", paired.ratio_unavailable_reason)

    def test_unpaired_seed_sets_fail_closed(self):
        with self.assertRaisesRegex(
                ComparisonMetricError, "paired seed sets differ"):
            aggregate_paired_seed_values(
                reference_by_seed={1: 1.0, 2: 2.0},
                candidate_by_seed={1: 1.0, 3: 3.0},
            )


if __name__ == "__main__":
    unittest.main()

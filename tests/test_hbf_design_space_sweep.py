from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from serving.hbf_design_space_sweep import (
    BASELINE_CANDIDATE_KEY,
    ORACLE_CANDIDATE_KEY,
    HBFDesignSpaceError,
    aggregate_cell_records,
    build_design_grid,
    make_design_spec,
    pareto_frontier,
    parse_active_memory_spec,
    parse_layout_set,
    validate_design_workspace,
    _CellTask,
    _load_resumable_cell,
    _seal_record,
)


def _summary(goodput: float, *, latency_scale: float = 1.0):
    return {
        "offered_load_normalized_output_token_goodput": {
            "value": goodput,
        },
        "slo": {
            "all_slo_pass_fraction": min(1.0, goodput / 100.0),
        },
        "latency_distributions_ns": {
            "first_ttft": {"p95_ns": 10.0 * latency_scale},
            "resume_ttft": {"p95_ns": 20.0 * latency_scale},
            "tpot_eligible": {"p95_ns": 30.0 * latency_scale},
        },
    }


def _record(
        candidate_key: str,
        seed: int,
        goodput: float,
        *,
        rate: float = 1.0,
):
    return {
        "candidate_key": candidate_key,
        "seed": seed,
        "session_rate": rate,
        "summary": _summary(goodput),
    }


class HBFDesignSpaceSweepTests(unittest.TestCase):

    def test_parses_canonical_layout_sets_and_explicit_memory(self):
        self.assertEqual(
            parse_layout_set("tp8_context,tp4,tp4"),
            ("tp4", "tp4", "tp8_context"),
        )
        lpddr = parse_active_memory_spec("lpddr:16:409.6")
        self.assertEqual(lpddr.kind, "lpddr")
        self.assertEqual(lpddr.capacity_gib_per_card, 16.0)
        self.assertEqual(lpddr.bandwidth_gbps_per_card, 409.6)
        sram = parse_active_memory_spec(
            "sram_like:8:3350:1000:5")
        self.assertEqual(sram.kind, "sram_like")
        self.assertEqual(sram.capex_usd_per_gib, 1000.0)
        self.assertEqual(sram.power_w_per_gib, 5.0)
        with self.assertRaises(HBFDesignSpaceError):
            parse_layout_set("tp8")
        with self.assertRaisesRegex(
                HBFDesignSpaceError, "requires explicit"):
            parse_active_memory_spec("sram_like:8:3350")

    def test_grid_reduces_symmetric_mixed_layouts(self):
        memory = parse_active_memory_spec("lpddr:16:204.8")
        grid = build_design_grid(
            hbf_host_counts=(1, 2),
            layouts=("tp4", "tp8_context"),
            migration_policies=("eager", "delay_200ms"),
            active_memories=(memory,),
            include_mixed_layouts=True,
        )
        # C(2,1) + C(3,2) layout multisets, times two policies.
        self.assertEqual(len(grid), 10)
        self.assertEqual(len({spec.key for spec in grid}), 10)
        self.assertIn(
            ("tp4", "tp8_context"),
            {spec.hbf_server_layouts for spec in grid},
        )
        self.assertTrue(all(
            spec.hbf_server_layouts
            == tuple(sorted(spec.hbf_server_layouts))
            for spec in grid
        ))

    def test_design_keys_include_active_memory_economics(self):
        first = parse_active_memory_spec(
            "sram_like:8:3350:1000:5")
        second = parse_active_memory_spec(
            "sram_like:8:3350:2000:5")
        specs = build_design_grid(
            hbf_host_counts=(1,),
            layouts=("tp4",),
            migration_policies=("eager",),
            active_memories=(first, second),
        )
        self.assertEqual(len({spec.key for spec in specs}), 2)

    def test_workspace_is_checked_before_an_expensive_cell(self):
        too_small = make_design_spec(
            hbf_server_layouts=("tp4",),
            migration_policy="eager",
            active_memory=parse_active_memory_spec(
                "lpddr:4:204.8"),
        )
        with self.assertRaisesRegex(
                HBFDesignSpaceError, "cannot hold workspace"):
            validate_design_workspace(too_small)
        feasible = make_design_spec(
            hbf_server_layouts=("tp8_context",),
            migration_policy="eager",
            active_memory=parse_active_memory_spec(
                "lpddr:8:204.8"),
        )
        audit = validate_design_workspace(feasible)
        self.assertGreater(
            audit["minimum_free_bytes_per_card"], 0)

    def test_pareto_frontier_uses_higher_goodput_and_lower_tco(self):
        self.assertEqual(
            pareto_frontier({
                "cheap": (80.0, 100.0),
                "balanced": (100.0, 120.0),
                "dominated": (90.0, 130.0),
                "fast": (110.0, 180.0),
            }),
            ("cheap", "balanced", "fast"),
        )

    def test_aggregation_pairs_seeds_and_attaches_tco(self):
        memory = parse_active_memory_spec("lpddr:16:204.8")
        design_a = make_design_spec(
            hbf_server_layouts=("tp4",),
            migration_policy="eager",
            active_memory=memory,
        )
        design_b = make_design_spec(
            hbf_server_layouts=("tp4", "tp4"),
            migration_policy="delay_200ms",
            active_memory=memory,
        )
        records = []
        for seed, baseline, oracle, a, b in (
            (7, 80.0, 120.0, 90.0, 105.0),
            (11, 100.0, 140.0, 110.0, 125.0),
        ):
            records.extend((
                _record(BASELINE_CANDIDATE_KEY, seed, baseline),
                _record(ORACLE_CANDIDATE_KEY, seed, oracle),
                _record(design_a.key, seed, a),
                _record(design_b.key, seed, b),
            ))
        result = aggregate_cell_records(
            records, (design_a, design_b))
        self.assertEqual(len(result["rates"]), 1)
        rate = result["rates"][0]
        self.assertEqual(
            rate["references"][BASELINE_CANDIDATE_KEY][
                "slo_good_output_tokens_per_second"]["mean"],
            90.0,
        )
        by_key = {
            row["design"]["key"]: row
            for row in rate["designs"]
        }
        self.assertEqual(
            by_key[design_a.key][
                "paired_vs_baseline_goodput"][
                    "candidate_minus_reference"]["mean"],
            10.0,
        )
        self.assertAlmostEqual(
            by_key[design_b.key][
                "paired_vs_baseline_goodput"][
                    "candidate_over_reference"]["mean"],
            (105.0 / 80.0 + 125.0 / 100.0) / 2,
        )
        self.assertIsNotNone(by_key[design_a.key]["tco"])
        self.assertIsNone(
            by_key[design_a.key]["tco"][
                "oracle_reference"]["lifetime_tco_usd"])
        self.assertTrue(
            set(rate["performance_tco_pareto_design_keys"])
            <= {design_a.key, design_b.key})

        broken = copy.deepcopy(records)
        broken.pop()
        with self.assertRaisesRegex(
                HBFDesignSpaceError, "unpaired seeds|incomplete"):
            aggregate_cell_records(broken, (design_a, design_b))

    def test_resume_requires_both_contract_and_payload_hash(self):
        task = _CellTask(
            repo_root=Path("/repo"),
            candidate_kind="baseline",
            candidate_key=BASELINE_CANDIDATE_KEY,
            seed=7,
            session_rate=1.0,
            scheduled_sessions=(),
            measurement_identities=("session::call-0",),
            design=None,
            first_ttft_seconds=30.0,
            resume_ttft_seconds=30.0,
            tpot_milliseconds=300.0,
            execution_inputs_sha256="a" * 64,
        )
        record = _seal_record(task, {
            "candidate_kind": "baseline",
            "candidate_key": BASELINE_CANDIDATE_KEY,
            "seed": 7,
            "session_rate": 1.0,
            "summary": _summary(80.0),
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cell.json"
            path.write_text(
                json.dumps(record, allow_nan=False),
                encoding="utf-8",
            )
            self.assertEqual(
                _load_resumable_cell(path, task), record)
            record["summary"][
                "offered_load_normalized_output_token_goodput"][
                    "value"] = 81.0
            path.write_text(
                json.dumps(record, allow_nan=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    HBFDesignSpaceError, "payload hash mismatch"):
                _load_resumable_cell(path, task)


if __name__ == "__main__":
    unittest.main()

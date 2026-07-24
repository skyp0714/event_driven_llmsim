import csv
import math
from pathlib import Path
import struct
import tempfile
import unittest

from serving.live_hbf_comparison_plots import (
    FONT_SIZE,
    GOODPUT_FIGSIZE,
    LATENCY_FIGSIZE,
    MATPLOTLIB_RC,
    REQUIRED_METRIC_COLUMNS,
    SYSTEM_LABELS,
    SYSTEM_STYLES,
    LiveComparisonInputError,
    aggregate_live_cells,
    load_live_cells,
    write_live_comparison_artifacts,
)


SYSTEMS = ("ssd_tiering", "oracle", "hbf_tp4", "hbf_tp8")
RATES = (0.5, 1.0)
SEEDS = (101, 211, 307)
OPTIONAL_METRIC = "hbf_compute_utilization"


def _rows():
    rows = []
    for system_index, system in enumerate(SYSTEMS):
        for rate in RATES:
            for seed_index, seed in enumerate(SEEDS):
                latency_base = (
                    (system_index + 1) * 10_000_000
                    + int(rate * 1_000_000)
                )
                rows.append({
                    "system_key": system,
                    "rate_sps": rate,
                    "seed": seed,
                    "resume_ttft_mean_ns": (
                        latency_base + seed_index * 1_000_000),
                    "resume_ttft_p95_ns": (
                        latency_base * 2 + seed_index * 2_000_000),
                    "tpot_mean_ns": (
                        latency_base / 2 + seed_index * 500_000),
                    "tpot_p95_ns": (
                        latency_base + seed_index * 1_000_000),
                    "joint_resume_goodput_rps": (
                        rate * 2 + system_index + seed_index),
                    "session_goodput_sps": (
                        rate + system_index + seed_index),
                    "output_token_goodput_tps": (
                        rate * 100 + system_index + seed_index),
                    "observed_request_throughput_rps": (
                        rate * 3 + system_index + seed_index),
                    OPTIONAL_METRIC: (
                        0.1 * system_index + 0.01 * seed_index),
                })
    return rows


def _write_csv(path, rows, *, fieldnames=None):
    columns = fieldnames or (
        "system_key",
        "rate_sps",
        "seed",
        *REQUIRED_METRIC_COLUMNS,
        OPTIONAL_METRIC,
    )
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _png_dimensions(path):
    contents = Path(path).read_bytes()
    if contents[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("artifact is not a PNG")
    return struct.unpack(">II", contents[16:24])


class LiveHbfComparisonPlotTests(unittest.TestCase):
    def test_publication_dimensions_fonts_and_stable_styles(self):
        self.assertEqual(LATENCY_FIGSIZE[0], 12)
        self.assertEqual(GOODPUT_FIGSIZE[0], 12)
        self.assertEqual(FONT_SIZE, 24)
        for key in (
                "ssd_tiering", "oracle", "hbf_tp4", "hbf_tp8",
                "hbf_tp8_context"):
            self.assertIn(key, SYSTEM_LABELS)
            self.assertIn(key, SYSTEM_STYLES)
        for rc_key in (
                "font.size", "axes.titlesize", "axes.labelsize",
                "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
            self.assertEqual(MATPLOTLIB_RC[rc_key], 24)

    def test_aggregation_uses_seed_mean_and_student_t_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cells.csv"
            _write_csv(source, _rows())
            dataset = load_live_cells(source)
            aggregate = aggregate_live_cells(dataset)

        point = aggregate.point(
            "ssd_tiering", 0.5, "resume_ttft_mean_ns")
        samples = point.seed_values
        expected_mean = sum(samples) / len(samples)
        expected_stddev = 1_000_000.0
        expected_half_width = (
            4.30265272975 * expected_stddev / math.sqrt(3))
        self.assertEqual(point.seed_count, 3)
        self.assertEqual(point.ci_method, "student_t_95")
        self.assertAlmostEqual(point.mean, expected_mean)
        self.assertAlmostEqual(point.sample_stddev, expected_stddev)
        self.assertAlmostEqual(
            point.ci95_half_width, expected_half_width, places=5)
        self.assertAlmostEqual(
            point.ci95_lower,
            expected_mean - expected_half_width,
            places=5,
        )
        self.assertAlmostEqual(
            point.ci95_upper,
            expected_mean + expected_half_width,
            places=5,
        )

        optional = aggregate.point(
            "hbf_tp4", 1.0, OPTIONAL_METRIC)
        self.assertAlmostEqual(optional.mean, 0.21)
        self.assertEqual(
            aggregate.optional_metric_columns, (OPTIONAL_METRIC,))

    def test_single_seed_mean_is_available_but_interval_is_not_invented(self):
        rows = [
            row for row in _rows()
            if row["seed"] == SEEDS[0]
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cells.csv"
            _write_csv(source, rows)
            aggregate = aggregate_live_cells(load_live_cells(source))

        point = aggregate.point(
            "oracle", 1.0, "joint_resume_goodput_rps")
        self.assertEqual(point.seed_count, 1)
        self.assertEqual(point.ci_method, "unavailable_single_seed")
        self.assertIsNone(point.sample_stddev)
        self.assertIsNone(point.ci95_lower)
        self.assertIsNone(point.ci95_upper)

    def test_duplicate_cell_fails_closed(self):
        rows = _rows()
        rows.append(dict(rows[0]))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cells.csv"
            _write_csv(source, rows)
            with self.assertRaisesRegex(
                    LiveComparisonInputError, "duplicate comparison cell"):
                load_live_cells(source)

    def test_missing_grid_cell_fails_closed(self):
        rows = _rows()
        rows.pop()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cells.csv"
            _write_csv(source, rows)
            with self.assertRaisesRegex(
                    LiveComparisonInputError, "grid is incomplete"):
                load_live_cells(source)

    def test_nonfinite_required_or_optional_value_fails_closed(self):
        for column in ("tpot_mean_ns", OPTIONAL_METRIC):
            with self.subTest(column=column):
                rows = _rows()
                rows[0][column] = "nan"
                with tempfile.TemporaryDirectory() as directory:
                    source = Path(directory) / "cells.csv"
                    _write_csv(source, rows)
                    with self.assertRaisesRegex(
                            LiveComparisonInputError, "non-finite"):
                        load_live_cells(source)

    def test_missing_required_column_fails_closed(self):
        columns = (
            "system_key",
            "rate_sps",
            "seed",
            *(
                key for key in REQUIRED_METRIC_COLUMNS
                if key != "resume_ttft_p95_ns"
            ),
            OPTIONAL_METRIC,
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cells.csv"
            _write_csv(source, _rows(), fieldnames=columns)
            with self.assertRaisesRegex(
                    LiveComparisonInputError,
                    "resume_ttft_p95_ns"):
                load_live_cells(source)

    def test_writes_aggregate_and_publication_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cells.csv"
            output = root / "plots"
            _write_csv(source, _rows())
            aggregate = aggregate_live_cells(load_live_cells(source))
            artifacts = write_live_comparison_artifacts(
                aggregate,
                output,
                prefix="paper",
                dpi=50,
            )

            self.assertEqual(
                set(artifacts),
                {
                    "aggregate_csv",
                    "latency_png",
                    "goodput_png",
                    "bottleneck_png",
                },
            )
            for path in artifacts.values():
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(
                _png_dimensions(artifacts["latency_png"]),
                (
                    int(LATENCY_FIGSIZE[0] * 50),
                    int(LATENCY_FIGSIZE[1] * 50),
                ),
            )
            self.assertEqual(
                _png_dimensions(artifacts["goodput_png"]),
                (
                    int(GOODPUT_FIGSIZE[0] * 50),
                    int(GOODPUT_FIGSIZE[1] * 50),
                ),
            )

            with artifacts["aggregate_csv"].open(
                    "r", encoding="utf-8", newline="") as handle:
                aggregate_rows = list(csv.DictReader(handle))
            self.assertEqual(
                len(aggregate_rows),
                len(SYSTEMS)
                * len(RATES)
                * (len(REQUIRED_METRIC_COLUMNS) + 1),
            )
            selected = next(
                row for row in aggregate_rows
                if (
                    row["system_key"] == "ssd_tiering"
                    and row["rate_sps"] == "0.5"
                    and row["metric_key"] == "resume_ttft_mean_ns"
                )
            )
            self.assertEqual(selected["seed_count"], "3")
            self.assertEqual(selected["ci_method"], "student_t_95")
            self.assertEqual(
                selected["seed_values_json"],
                "[10500000.0,11500000.0,12500000.0]",
            )

    def test_no_render_writes_only_the_aggregate_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cells.csv"
            _write_csv(source, _rows())
            aggregate = aggregate_live_cells(load_live_cells(source))
            artifacts = write_live_comparison_artifacts(
                aggregate, root / "plots", render=False)

        self.assertEqual(set(artifacts), {"aggregate_csv"})


if __name__ == "__main__":
    unittest.main()

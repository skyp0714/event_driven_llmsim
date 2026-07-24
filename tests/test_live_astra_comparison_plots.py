import csv
import json
import math
from pathlib import Path
import struct

import matplotlib
import pytest

import serving.live_astra_comparison_plots as plot_module
from serving.live_astra_comparison_plots import (
    BOTTLENECK_FIGSIZE,
    BOTTLENECK_SPECS,
    FIGURE_DPI,
    FIGURE_WIDTH_INCHES,
    FONT_SIZE,
    LiveAstraPlotError,
    aggregate_campaign,
    load_compact_campaign,
    main,
    render_campaign,
)


SYSTEMS = ("ssd_tiering", "oracle", "hbf_tp8_context")
RATES = (0.1, 0.3)
SEEDS = (1, 2, 3)


def _compact() -> dict[str, object]:
    cells = []
    system_ttft_seconds = {
        "ssd_tiering": 10.0,
        "oracle": 1.0,
        "hbf_tp8_context": 2.0,
    }
    for system in SYSTEMS:
        for rate in RATES:
            for seed_index, seed in enumerate(SEEDS):
                rate_multiplier = 1.0 if rate == 0.1 else 3.0
                ttft_seconds = (
                    system_ttft_seconds[system] * rate_multiplier
                    + 2.0 * seed_index
                )
                performance = {
                    "resume_ttft_ns": {"p95_ns": ttft_seconds * 1e9},
                    "resume_tpot_ns": {
                        "p95_ns": (100.0 + 10.0 * seed_index) * 1e6,
                    },
                    "joint_slo_pass_fraction": (
                        0.95 - 0.05 * seed_index
                    ),
                    "offered_normalized_request_slo_goodput_per_second": (
                        rate * (0.95 - 0.05 * seed_index)
                    ),
                    "operational_request_goodput_per_second": (
                        10.0 * rate
                    ),
                }
                cell: dict[str, object] = {
                    "cell_id": f"{system}-{rate}-{seed}",
                    "system": system,
                    "seed": seed,
                    "offered_session_rate_per_second": rate,
                    "performance": performance,
                    "bottlenecks": {},
                }
                if system == "ssd_tiering":
                    cell["bottlenecks"] = {
                        "baseline": {
                            "ssd": {
                                "ssd_hits": (
                                    0 if rate == 0.1 else 10 + seed_index
                                ),
                                "media_host_read_bytes": (
                                    (seed_index + 1) * 1024 ** 3
                                ),
                                "media_aligned_host_write_bytes": (
                                    2 * (seed_index + 1) * 1024 ** 3
                                ),
                            },
                        },
                    }
                elif system == "hbf_tp8_context":
                    prefill_drain = (
                        {
                            "candidate_fraction": 0.0,
                            "mean_wait_ms": 0.0,
                            "fallback_fraction": 0.0,
                            "logical_traffic_gib": 0.0,
                        }
                        if rate == 0.1 else
                        {
                            "candidate_fraction": (
                                0.2 + 0.1 * seed_index),
                            "mean_wait_ms": 5.0 + seed_index,
                            "fallback_fraction": (
                                0.05 * (seed_index + 1)),
                            "logical_traffic_gib": (
                                1.0 + seed_index),
                        }
                    )
                    cell["bottlenecks"] = {
                        "hbf": {
                            "routing": {
                                "offered_requests": 100,
                                "hbf_requests": 40 + 10 * seed_index,
                            },
                            "capacity": {
                                "hbf_reserved_bytes_peak": (
                                    (seed_index + 1) * 100
                                ),
                                "hbf_capacity_bytes_per_card": 1_000,
                                "card_count": 8,
                            },
                            "prefill_drain": {
                                "derived": prefill_drain,
                            },
                        },
                    }
                cells.append(cell)
    return {
        "schema_version": 1,
        "campaign_sha256": "0" * 64,
        "cells": cells,
    }


def _write_compact(tmp_path: Path, compact: object | None = None) -> Path:
    path = tmp_path / "compact.json"
    path.write_text(
        json.dumps(_compact() if compact is None else compact) + "\n",
        encoding="utf-8",
    )
    return path


def _png_size(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", payload[16:24])


def test_render_outputs_use_seed_student_t_and_required_style(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_compact(tmp_path)
    plotted_metrics = []
    original_plot_metric = plot_module._plot_metric

    def record_plot_metric(*args: object, **kwargs: object) -> None:
        plotted_metrics.append(args[2].key)
        original_plot_metric(*args, **kwargs)

    monkeypatch.setattr(
        plot_module, "_plot_metric", record_plot_metric)
    outputs = render_campaign(source, tmp_path / "plots", prefix="synthetic")

    paths = (
        outputs.performance_png,
        outputs.performance_pdf,
        outputs.bottlenecks_png,
        outputs.bottlenecks_pdf,
        outputs.summary_csv,
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert _png_size(outputs.performance_png)[0] == (
        FIGURE_WIDTH_INCHES * FIGURE_DPI
    )
    assert _png_size(outputs.bottlenecks_png)[0] == (
        FIGURE_WIDTH_INCHES * FIGURE_DPI
    )
    assert _png_size(outputs.bottlenecks_png)[1] == (
        BOTTLENECK_FIGSIZE[1] * FIGURE_DPI
    )
    assert matplotlib.rcParams["font.size"] == FONT_SIZE
    assert matplotlib.rcParams["axes.labelsize"] == FONT_SIZE
    assert all("/run" in spec.y_label for spec in BOTTLENECK_SPECS)
    assert all(
        spec.key in plotted_metrics for spec in BOTTLENECK_SPECS)
    assert outputs.goodput_basis == "offered_normalized"
    assert outputs.latency_log_metrics == ("resume_ttft_p95",)

    with outputs.summary_csv.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    ttft = next(
        row for row in rows
        if row["system"] == "ssd_tiering"
        and row["rate_sps"] == "0.10000000000000001"
        and row["metric"] == "resume_ttft_p95"
    )
    assert ttft["status"] == "reported"
    assert ttft["seed_count"] == "3"
    assert ttft["ci_method"] == "student_t_95"
    assert math.isclose(float(ttft["arithmetic_mean"]), 12.0)
    expected_half_width = 4.30265272975 * 2.0 / math.sqrt(3.0)
    assert math.isclose(
        float(ttft["ci95_half_width"]),
        expected_half_width,
        rel_tol=1e-12,
    )
    assert ttft["log_y_applied"] == "true"

    oracle_ssd = next(
        row for row in rows
        if row["system"] == "oracle"
        and row["rate_sps"] == "0.10000000000000001"
        and row["metric"] == "ssd_hits"
    )
    assert oracle_ssd["status"] == "not_reported"
    zero_ssd = next(
        row for row in rows
        if row["system"] == "ssd_tiering"
        and row["rate_sps"] == "0.10000000000000001"
        and row["metric"] == "ssd_hits"
    )
    assert zero_ssd["status"] == "reported"
    assert float(zero_ssd["arithmetic_mean"]) == 0.0
    zero_candidates = next(
        row for row in rows
        if row["system"] == "hbf_tp8_context"
        and row["rate_sps"] == "0.10000000000000001"
        and row["metric"] == "hbf_prefill_drain_candidate_fraction"
    )
    assert zero_candidates["status"] == "reported"
    assert float(zero_candidates["arithmetic_mean"]) == 0.0
    drain_traffic = next(
        row for row in rows
        if row["system"] == "hbf_tp8_context"
        and row["rate_sps"] == "0.29999999999999999"
        and row["metric"] == "hbf_prefill_drain_logical_traffic_gib"
    )
    assert drain_traffic["status"] == "reported"
    assert float(drain_traffic["arithmetic_mean"]) == 2.0
    assert drain_traffic["source"] == (
        "bottlenecks.hbf.prefill_drain.derived."
        "logical_traffic_gib")
    expected_drain_means = {
        "hbf_prefill_drain_candidate_fraction": 30.0,
        "hbf_prefill_drain_mean_wait_ms": 6.0,
        "hbf_prefill_drain_fallback_fraction": 10.0,
        "hbf_prefill_drain_logical_traffic_gib": 2.0,
    }
    reported_drain_rows = {
        row["metric"]: row
        for row in rows
        if (
            row["system"] == "hbf_tp8_context"
            and row["rate_sps"] == "0.29999999999999999"
            and row["metric"] in expected_drain_means
        )
    }
    assert set(reported_drain_rows) == set(expected_drain_means)
    for metric, expected_mean in expected_drain_means.items():
        assert reported_drain_rows[metric]["status"] == "reported"
        assert math.isclose(
            float(reported_drain_rows[metric]["arithmetic_mean"]),
            expected_mean,
        )


def test_operational_goodput_fallback_is_explicit(tmp_path: Path) -> None:
    compact = _compact()
    for cell in compact["cells"]:
        del cell["performance"][
            "offered_normalized_request_slo_goodput_per_second"
        ]
    source = _write_compact(tmp_path, compact)
    dataset = load_compact_campaign(source)
    assert dataset.goodput_basis == "operational"
    assert dataset.goodput_source == (
        "performance.operational_request_goodput_per_second"
    )
    aggregate = aggregate_campaign(dataset)
    point = next(
        value for value in aggregate.points
        if value.system == "oracle"
        and value.rate == 0.3
        and value.metric == "slo_goodput"
    )
    assert point.mean == 3.0


def test_partial_metric_and_incomplete_grid_fail_closed(
    tmp_path: Path,
) -> None:
    partial = _compact()
    for cell in partial["cells"][1:]:
        del cell["performance"][
            "offered_normalized_request_slo_goodput_per_second"
        ]
    with pytest.raises(
        LiveAstraPlotError, match="only partially reported"
    ):
        load_compact_campaign(_write_compact(tmp_path, partial))

    incomplete = _compact()
    incomplete["cells"].pop()
    with pytest.raises(LiveAstraPlotError, match="grid is incomplete"):
        load_compact_campaign(_write_compact(tmp_path, incomplete))


def test_partial_and_inconsistent_zero_prefill_drain_fail_closed(
    tmp_path: Path,
) -> None:
    partial = _compact()
    hbf_cell = next(
        cell for cell in partial["cells"]
        if cell["system"] == "hbf_tp8_context"
    )
    del hbf_cell["bottlenecks"]["hbf"]["prefill_drain"]["derived"][
        "mean_wait_ms"]
    with pytest.raises(
        LiveAstraPlotError, match="partially reported"
    ):
        load_compact_campaign(_write_compact(tmp_path, partial))

    inconsistent_zero = _compact()
    zero_cell = next(
        cell for cell in inconsistent_zero["cells"]
        if (
            cell["system"] == "hbf_tp8_context"
            and cell["offered_session_rate_per_second"] == 0.1
        )
    )
    zero_cell["bottlenecks"]["hbf"]["prefill_drain"]["derived"][
        "logical_traffic_gib"] = 1.0
    with pytest.raises(
        LiveAstraPlotError,
        match="zero HBF prefill-drain candidates",
    ):
        load_compact_campaign(
            _write_compact(tmp_path, inconsistent_zero))


def test_cli_writes_named_outputs(tmp_path: Path, capsys: object) -> None:
    source = _write_compact(tmp_path)
    output = tmp_path / "cli"
    assert main([
        str(source),
        "--output-dir",
        str(output),
        "--prefix",
        "campaign",
    ]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["goodput_basis"] == "offered_normalized"
    assert Path(printed["summary_csv"]).name == "campaign_summary.csv"
    assert (output / "campaign_performance.png").is_file()

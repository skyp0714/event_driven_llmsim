"""Aggregate and plot live HBF comparison measurements.

This module consumes one seed-level row per ``(system, rate, seed)`` from a
tidy CSV.  It deliberately validates the complete Cartesian grid before
computing any statistic so a partially completed sweep cannot silently appear
in a publication plot.  The independent replicate is one simulation seed;
request-level observations must already have been summarized by the producer.

The required latency column names are unambiguous:
``resume_ttft_mean_ns``, ``resume_ttft_p95_ns``, ``tpot_mean_ns``, and
``tpot_p95_ns``.  Additional columns are treated as numeric bottleneck metrics
and receive the same seed-level arithmetic mean and Student-t 95% interval.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence


FONT_SIZE = 24
PNG_DPI = 200
LATENCY_FIGSIZE = (12, 18)
GOODPUT_FIGSIZE = (12, 18)

MATPLOTLIB_RC = {
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
}

SYSTEM_ORDER = (
    "ssd_tiering",
    "oracle",
    "hbf_tp4",
    "hbf_tp8",
    "hbf_tp8_context",
)

SYSTEM_LABELS = {
    "ssd_tiering": "SSD tiering",
    "oracle": "Infinite-HBM oracle",
    "hbf_tp4": "HBF 2×TP4",
    "hbf_tp8": "HBF TP8",
    "hbf_tp8_context": "HBF TP8-context",
}

SYSTEM_STYLES = {
    "ssd_tiering": {
        "color": "#d62728",
        "marker": "o",
        "linestyle": "-",
    },
    "oracle": {
        "color": "#111111",
        "marker": "D",
        "linestyle": "--",
    },
    "hbf_tp4": {
        "color": "#1f77b4",
        "marker": "s",
        "linestyle": "-",
    },
    "hbf_tp8": {
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-",
    },
    "hbf_tp8_context": {
        "color": "#9467bd",
        "marker": "P",
        "linestyle": "-.",
    },
}

IDENTIFIER_COLUMNS = ("system_key", "rate_sps", "seed")


class LiveComparisonInputError(RuntimeError):
    """Raised when a live comparison CSV is incomplete or malformed."""


class LiveComparisonRenderError(RuntimeError):
    """Raised when publication figures cannot be rendered."""


@dataclass(frozen=True)
class MetricDefinition:
    """Display and unit conversion for one required input metric."""

    key: str
    title: str
    y_label: str
    scale: float
    unit: str
    category: str
    strictly_positive: bool = False
    log_y: bool = False


METRIC_DEFINITIONS = (
    MetricDefinition(
        key="resume_ttft_mean_ns",
        title="Resume TTFT mean",
        y_label="Resume TTFT mean (s)",
        scale=1e-9,
        unit="seconds",
        category="latency",
        strictly_positive=True,
        log_y=True,
    ),
    MetricDefinition(
        key="resume_ttft_p95_ns",
        title="Resume TTFT p95",
        y_label="Resume TTFT p95 (s)",
        scale=1e-9,
        unit="seconds",
        category="latency",
        strictly_positive=True,
        log_y=True,
    ),
    MetricDefinition(
        key="tpot_mean_ns",
        title="TPOT mean",
        y_label="TPOT mean (ms/token)",
        scale=1e-6,
        unit="milliseconds_per_token",
        category="latency",
        strictly_positive=True,
        log_y=True,
    ),
    MetricDefinition(
        key="tpot_p95_ns",
        title="TPOT p95",
        y_label="TPOT p95 (ms/token)",
        scale=1e-6,
        unit="milliseconds_per_token",
        category="latency",
        strictly_positive=True,
        log_y=True,
    ),
    MetricDefinition(
        key="joint_resume_goodput_rps",
        title="Joint resume-SLO goodput",
        y_label="SLO-good resume requests/s",
        scale=1.0,
        unit="requests_per_second",
        category="goodput",
    ),
    MetricDefinition(
        key="session_goodput_sps",
        title="Session goodput",
        y_label="SLO-good sessions/s",
        scale=1.0,
        unit="sessions_per_second",
        category="goodput",
    ),
    MetricDefinition(
        key="output_token_goodput_tps",
        title="Output-token goodput",
        y_label="SLO-good output tokens/s",
        scale=1.0,
        unit="output_tokens_per_second",
        category="goodput",
    ),
    MetricDefinition(
        key="observed_request_throughput_rps",
        title="Observed request throughput",
        y_label="Completed requests/s",
        scale=1.0,
        unit="requests_per_second",
        category="goodput",
    ),
)

METRIC_BY_KEY = {
    definition.key: definition for definition in METRIC_DEFINITIONS
}
REQUIRED_METRIC_COLUMNS = tuple(
    definition.key for definition in METRIC_DEFINITIONS)
REQUIRED_COLUMNS = frozenset(IDENTIFIER_COLUMNS + REQUIRED_METRIC_COLUMNS)

_INTEGER_PATTERN = re.compile(r"[+-]?[0-9]+\Z")

_STUDENT_T_975 = {
    1: 12.7062047364,
    2: 4.30265272975,
    3: 3.18244630528,
    4: 2.7764451052,
    5: 2.57058183564,
    6: 2.44691184879,
    7: 2.36462425101,
    8: 2.3060041352,
    9: 2.26215716285,
    10: 2.22813885196,
    11: 2.20098516008,
    12: 2.17881282966,
    13: 2.16036865646,
    14: 2.14478668792,
    15: 2.13144954556,
    16: 2.11990529922,
    17: 2.10981557783,
    18: 2.10092204024,
    19: 2.09302405441,
    20: 2.08596344727,
    21: 2.07961384473,
    22: 2.0738730679,
    23: 2.06865761042,
    24: 2.06389856163,
    25: 2.05953855275,
    26: 2.05552943864,
    27: 2.05183051648,
    28: 2.0484071418,
    29: 2.04522964213,
    30: 2.0422724563,
}


@dataclass(frozen=True)
class LiveCell:
    """One seed-level result from a fully drained simulation cell."""

    system_key: str
    rate_sps: float
    seed: int
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class LiveDataset:
    """A validated complete comparison grid."""

    cells: tuple[LiveCell, ...]
    systems: tuple[str, ...]
    rates_sps: tuple[float, ...]
    seeds: tuple[int, ...]
    optional_metric_columns: tuple[str, ...]

    @property
    def metric_columns(self) -> tuple[str, ...]:
        return REQUIRED_METRIC_COLUMNS + self.optional_metric_columns


@dataclass(frozen=True)
class AggregatePoint:
    """Seed-level arithmetic mean and two-sided Student-t 95% CI."""

    system_key: str
    rate_sps: float
    metric_key: str
    seed_count: int
    seed_values: tuple[float, ...]
    mean: float
    sample_stddev: float | None
    ci95_half_width: float | None
    ci95_lower: float | None
    ci95_upper: float | None
    ci_method: str


@dataclass(frozen=True)
class LiveAggregate:
    """All aggregate points needed by the CSV and figures."""

    points: tuple[AggregatePoint, ...]
    systems: tuple[str, ...]
    rates_sps: tuple[float, ...]
    seeds: tuple[int, ...]
    optional_metric_columns: tuple[str, ...]

    @property
    def metric_columns(self) -> tuple[str, ...]:
        return REQUIRED_METRIC_COLUMNS + self.optional_metric_columns

    def point(
            self, system_key: str, rate_sps: float,
            metric_key: str) -> AggregatePoint:
        for point in self.points:
            if (
                    point.system_key == system_key
                    and point.rate_sps == rate_sps
                    and point.metric_key == metric_key):
                return point
        raise KeyError((system_key, rate_sps, metric_key))


def _parse_float(raw: object, *, row_number: int, column: str) -> float:
    text = "" if raw is None else str(raw).strip()
    if not text:
        raise LiveComparisonInputError(
            f"row {row_number}: {column} is empty")
    try:
        value = float(text)
    except ValueError as error:
        raise LiveComparisonInputError(
            f"row {row_number}: {column} is not numeric: {text!r}") from error
    if not math.isfinite(value):
        raise LiveComparisonInputError(
            f"row {row_number}: {column} is non-finite: {text!r}")
    return value


def _parse_seed(raw: object, *, row_number: int) -> int:
    text = "" if raw is None else str(raw).strip()
    if not _INTEGER_PATTERN.fullmatch(text):
        raise LiveComparisonInputError(
            f"row {row_number}: seed must be an integer, got {text!r}")
    seed = int(text)
    if seed < 0:
        raise LiveComparisonInputError(
            f"row {row_number}: seed must be non-negative, got {seed}")
    return seed


def _system_sort_key(system_key: str) -> tuple[int, str]:
    try:
        return SYSTEM_ORDER.index(system_key), system_key
    except ValueError:
        return len(SYSTEM_ORDER), system_key


def _format_coordinates(
        coordinates: Sequence[tuple[str, float, int]],
        *, limit: int = 8) -> str:
    rendered = [
        f"({system}, rate={rate:.17g}, seed={seed})"
        for system, rate, seed in coordinates[:limit]
    ]
    if len(coordinates) > limit:
        rendered.append(f"... and {len(coordinates) - limit} more")
    return ", ".join(rendered)


def load_live_cells(path: str | Path) -> LiveDataset:
    """Load a tidy per-cell CSV and fail closed on an incomplete grid."""

    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise LiveComparisonInputError(
            f"cannot open live comparison CSV {source}: {error}") from error

    with handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise LiveComparisonInputError(
                f"live comparison CSV has no header: {source}")
        if any(name is None or not name.strip() for name in fieldnames):
            raise LiveComparisonInputError(
                "live comparison CSV has an empty column name")
        if len(fieldnames) != len(set(fieldnames)):
            raise LiveComparisonInputError(
                "live comparison CSV has duplicate column names")
        missing_columns = sorted(REQUIRED_COLUMNS - set(fieldnames))
        if missing_columns:
            raise LiveComparisonInputError(
                "live comparison CSV is missing required columns: "
                + ", ".join(missing_columns))
        optional_metrics = tuple(
            name for name in fieldnames if name not in REQUIRED_COLUMNS)

        cells = []
        occupied = set()
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise LiveComparisonInputError(
                    f"row {row_number}: contains values beyond the CSV header")
            system_key = (row.get("system_key") or "").strip()
            if system_key not in SYSTEM_STYLES:
                raise LiveComparisonInputError(
                    f"row {row_number}: unsupported system_key "
                    f"{system_key!r}; supported systems are "
                    f"{', '.join(SYSTEM_ORDER)}")
            rate_sps = _parse_float(
                row.get("rate_sps"), row_number=row_number,
                column="rate_sps")
            if rate_sps <= 0.0:
                raise LiveComparisonInputError(
                    f"row {row_number}: rate_sps must be positive")
            seed = _parse_seed(row.get("seed"), row_number=row_number)
            coordinate = (system_key, rate_sps, seed)
            if coordinate in occupied:
                raise LiveComparisonInputError(
                    "duplicate comparison cell "
                    + _format_coordinates((coordinate,)))
            occupied.add(coordinate)

            metrics = {}
            for definition in METRIC_DEFINITIONS:
                value = _parse_float(
                    row.get(definition.key),
                    row_number=row_number,
                    column=definition.key,
                )
                if definition.strictly_positive and value <= 0.0:
                    raise LiveComparisonInputError(
                        f"row {row_number}: {definition.key} must be positive")
                if not definition.strictly_positive and value < 0.0:
                    raise LiveComparisonInputError(
                        f"row {row_number}: {definition.key} "
                        "must be non-negative")
                metrics[definition.key] = value
            for column in optional_metrics:
                metrics[column] = _parse_float(
                    row.get(column), row_number=row_number, column=column)
            cells.append(LiveCell(
                system_key=system_key,
                rate_sps=rate_sps,
                seed=seed,
                metrics=metrics,
            ))

    if not cells:
        raise LiveComparisonInputError(
            f"live comparison CSV contains no data rows: {source}")

    systems = tuple(sorted(
        {cell.system_key for cell in cells}, key=_system_sort_key))
    rates = tuple(sorted({cell.rate_sps for cell in cells}))
    seeds = tuple(sorted({cell.seed for cell in cells}))
    expected = {
        (system, rate, seed)
        for system in systems
        for rate in rates
        for seed in seeds
    }
    missing = sorted(
        expected - occupied,
        key=lambda item: (_system_sort_key(item[0]), item[1], item[2]),
    )
    if missing:
        raise LiveComparisonInputError(
            "comparison grid is incomplete; missing cells: "
            + _format_coordinates(missing))

    cells.sort(key=lambda cell: (
        _system_sort_key(cell.system_key), cell.rate_sps, cell.seed))
    return LiveDataset(
        cells=tuple(cells),
        systems=systems,
        rates_sps=rates,
        seeds=seeds,
        optional_metric_columns=optional_metrics,
    )


def _student_t_975(degrees_of_freedom: int) -> float:
    """Return the two-sided 95% Student-t critical value."""

    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    exact = _STUDENT_T_975.get(int(degrees_of_freedom))
    if exact is not None:
        return exact
    z = 1.959963984540054
    df = float(degrees_of_freedom)
    return (
        z
        + (z ** 3 + z) / (4 * df)
        + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
        + (
            3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z
        ) / (384 * df ** 3)
    )


def _aggregate_values(
        *, system_key: str, rate_sps: float, metric_key: str,
        values: Sequence[float]) -> AggregatePoint:
    samples = tuple(float(value) for value in values)
    if not samples:
        raise ValueError("cannot aggregate an empty seed sample")
    mean = math.fsum(samples) / len(samples)
    if len(samples) == 1:
        return AggregatePoint(
            system_key=system_key,
            rate_sps=rate_sps,
            metric_key=metric_key,
            seed_count=1,
            seed_values=samples,
            mean=mean,
            sample_stddev=None,
            ci95_half_width=None,
            ci95_lower=None,
            ci95_upper=None,
            ci_method="unavailable_single_seed",
        )
    variance = math.fsum(
        (value - mean) ** 2 for value in samples) / (len(samples) - 1)
    sample_stddev = math.sqrt(variance)
    half_width = (
        _student_t_975(len(samples) - 1)
        * sample_stddev
        / math.sqrt(len(samples))
    )
    return AggregatePoint(
        system_key=system_key,
        rate_sps=rate_sps,
        metric_key=metric_key,
        seed_count=len(samples),
        seed_values=samples,
        mean=mean,
        sample_stddev=sample_stddev,
        ci95_half_width=half_width,
        ci95_lower=mean - half_width,
        ci95_upper=mean + half_width,
        ci_method="student_t_95",
    )


def aggregate_live_cells(dataset: LiveDataset) -> LiveAggregate:
    """Aggregate each exact system/rate cell across arrival seeds."""

    indexed = {
        (cell.system_key, cell.rate_sps, cell.seed): cell
        for cell in dataset.cells
    }
    points = []
    for system in dataset.systems:
        for rate in dataset.rates_sps:
            for metric in dataset.metric_columns:
                values = [
                    indexed[(system, rate, seed)].metrics[metric]
                    for seed in dataset.seeds
                ]
                points.append(_aggregate_values(
                    system_key=system,
                    rate_sps=rate,
                    metric_key=metric,
                    values=values,
                ))
    return LiveAggregate(
        points=tuple(points),
        systems=dataset.systems,
        rates_sps=dataset.rates_sps,
        seeds=dataset.seeds,
        optional_metric_columns=dataset.optional_metric_columns,
    )


def _float_text(value: float | None) -> str:
    return "" if value is None else format(value, ".17g")


def _metric_metadata(
        metric_key: str) -> tuple[str, str, str, float, str]:
    definition = METRIC_BY_KEY.get(metric_key)
    if definition is None:
        return (
            metric_key.replace("_", " "),
            "raw",
            "raw",
            1.0,
            "bottleneck",
        )
    source_unit = (
        "nanoseconds"
        if metric_key.endswith("_ns")
        else definition.unit
    )
    return (
        definition.title,
        source_unit,
        definition.unit,
        definition.scale,
        definition.category,
    )


def write_aggregate_csv(
        aggregate: LiveAggregate, path: str | Path) -> Path:
    """Write one long-format aggregate row per system/rate/metric."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "system_key",
        "system_label",
        "rate_sps",
        "metric_key",
        "metric_title",
        "metric_category",
        "source_unit",
        "display_unit",
        "display_scale",
        "seed_count",
        "seed_values_json",
        "arithmetic_mean",
        "sample_stddev",
        "ci95_half_width",
        "ci95_lower",
        "ci95_upper",
        "ci_method",
    )
    temporary_handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary_handle.name)
    try:
        with temporary_handle:
            writer = csv.DictWriter(temporary_handle, fieldnames=fieldnames)
            writer.writeheader()
            for point in aggregate.points:
                title, source_unit, display_unit, scale, category = (
                    _metric_metadata(point.metric_key)
                )
                writer.writerow({
                    "system_key": point.system_key,
                    "system_label": SYSTEM_LABELS[point.system_key],
                    "rate_sps": _float_text(point.rate_sps),
                    "metric_key": point.metric_key,
                    "metric_title": title,
                    "metric_category": category,
                    "source_unit": source_unit,
                    "display_unit": display_unit,
                    "display_scale": _float_text(scale),
                    "seed_count": point.seed_count,
                    "seed_values_json": json.dumps(
                        point.seed_values,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "arithmetic_mean": _float_text(point.mean),
                    "sample_stddev": _float_text(point.sample_stddev),
                    "ci95_half_width": _float_text(
                        point.ci95_half_width),
                    "ci95_lower": _float_text(point.ci95_lower),
                    "ci95_upper": _float_text(point.ci95_upper),
                    "ci_method": point.ci_method,
                })
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise LiveComparisonRenderError(
            "matplotlib is required to render live comparison plots") from error
    return plt


def _scaled_interval(
        point: AggregatePoint, scale: float,
        *, positive_axis: bool) -> tuple[float, list[list[float]] | None]:
    mean = point.mean * scale
    if point.ci95_lower is None or point.ci95_upper is None:
        return mean, None
    lower = point.ci95_lower * scale
    upper = point.ci95_upper * scale
    if positive_axis and lower <= 0.0:
        lower = max(mean * 1e-6, math.nextafter(0.0, 1.0))
    return mean, [[mean - lower], [upper - mean]]


def _plot_metric(
        axis, aggregate: LiveAggregate, metric_key: str,
        *, definition: MetricDefinition | None) -> None:
    if definition is None:
        title, y_label, scale, log_y = (
            metric_key.replace("_", " "),
            metric_key.replace("_", " "),
            1.0,
            False,
        )
    else:
        title = definition.title
        y_label = definition.y_label
        scale = definition.scale
        log_y = definition.log_y

    for system in aggregate.systems:
        x_values = []
        y_values = []
        lower_errors = []
        upper_errors = []
        has_interval = True
        for rate in aggregate.rates_sps:
            point = aggregate.point(system, rate, metric_key)
            mean, interval = _scaled_interval(
                point, scale, positive_axis=log_y)
            x_values.append(rate)
            y_values.append(mean)
            if interval is None:
                has_interval = False
                lower_errors.append(0.0)
                upper_errors.append(0.0)
            else:
                lower_errors.append(interval[0][0])
                upper_errors.append(interval[1][0])
        style = SYSTEM_STYLES[system]
        axis.errorbar(
            x_values,
            y_values,
            yerr=(
                [lower_errors, upper_errors] if has_interval else None),
            label=SYSTEM_LABELS[system],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.5,
            markersize=9,
            capsize=5,
        )
    axis.set_title(title)
    axis.set_xlabel("Offered session rate (sessions/s)")
    axis.set_ylabel(y_label)
    axis.grid(True, which="both", alpha=0.3)
    if log_y:
        axis.set_yscale("log")
    else:
        axis.set_ylim(bottom=0.0)


def _save_figure_atomic(figure, destination: Path, *, dpi: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".png",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(
            temporary_path,
            format="png",
            dpi=dpi,
            bbox_inches=None,
        )
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _render_four_panel(
        aggregate: LiveAggregate,
        metric_keys: Sequence[str],
        destination: Path,
        *, figure_size: tuple[int, int], title: str, dpi: int) -> Path:
    if len(metric_keys) != 4:
        raise ValueError("four-panel figures require exactly four metrics")
    plt = _load_matplotlib()
    with plt.rc_context(MATPLOTLIB_RC):
        figure, axes = plt.subplots(2, 2, figsize=figure_size)
        for axis, metric_key in zip(axes.flat, metric_keys):
            _plot_metric(
                axis,
                aggregate,
                metric_key,
                definition=METRIC_BY_KEY[metric_key],
            )
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.suptitle(title)
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.015),
        )
        figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.96))
        _save_figure_atomic(figure, destination, dpi=dpi)
        plt.close(figure)
    return destination


def _render_bottlenecks(
        aggregate: LiveAggregate, destination: Path, *, dpi: int) -> Path:
    metric_keys = aggregate.optional_metric_columns
    if not metric_keys:
        raise ValueError("no optional bottleneck metrics are available")
    plt = _load_matplotlib()
    row_count = math.ceil(len(metric_keys) / 2)
    figure_size = (12, max(8, 7 * row_count))
    with plt.rc_context(MATPLOTLIB_RC):
        figure, axes = plt.subplots(
            row_count, 2, figsize=figure_size, squeeze=False)
        flat_axes = tuple(axes.flat)
        for axis, metric_key in zip(flat_axes, metric_keys):
            _plot_metric(
                axis, aggregate, metric_key, definition=None)
        for axis in flat_axes[len(metric_keys):]:
            axis.set_visible(False)
        handles, labels = flat_axes[0].get_legend_handles_labels()
        figure.suptitle("Bottleneck diagnostics")
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.015),
        )
        figure.tight_layout(rect=(0.0, 0.11, 1.0, 0.95))
        _save_figure_atomic(figure, destination, dpi=dpi)
        plt.close(figure)
    return destination


def write_live_comparison_artifacts(
        aggregate: LiveAggregate,
        output_dir: str | Path,
        *,
        prefix: str = "live_hbf_comparison",
        render: bool = True,
        dpi: int = PNG_DPI) -> Mapping[str, Path]:
    """Write the aggregate CSV and publication figures."""

    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be one non-empty filename component")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "aggregate_csv": write_aggregate_csv(
            aggregate, output / f"{prefix}_aggregate.csv"),
    }
    if not render:
        return artifacts

    latency_metrics = tuple(
        definition.key for definition in METRIC_DEFINITIONS
        if definition.category == "latency")
    goodput_metrics = tuple(
        definition.key for definition in METRIC_DEFINITIONS
        if definition.category == "goodput")
    artifacts["latency_png"] = _render_four_panel(
        aggregate,
        latency_metrics,
        output / f"{prefix}_latency.png",
        figure_size=LATENCY_FIGSIZE,
        title="Resume latency (mean across seeds, 95% Student-t CI)",
        dpi=dpi,
    )
    artifacts["goodput_png"] = _render_four_panel(
        aggregate,
        goodput_metrics,
        output / f"{prefix}_goodput.png",
        figure_size=GOODPUT_FIGSIZE,
        title="Goodput and throughput (mean across seeds, 95% Student-t CI)",
        dpi=dpi,
    )
    if aggregate.optional_metric_columns:
        artifacts["bottleneck_png"] = _render_bottlenecks(
            aggregate,
            output / f"{prefix}_bottlenecks.png",
            dpi=dpi,
        )
    return artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate, aggregate, and plot a tidy live HBF comparison CSV."))
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="live_hbf_comparison")
    parser.add_argument("--dpi", type=int, default=PNG_DPI)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="write only the aggregate CSV",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    dataset = load_live_cells(args.input_csv)
    aggregate = aggregate_live_cells(dataset)
    artifacts = write_live_comparison_artifacts(
        aggregate,
        args.output_dir,
        prefix=args.prefix,
        render=not args.no_render,
        dpi=args.dpi,
    )
    print(json.dumps(
        {key: str(path) for key, path in artifacts.items()},
        sort_keys=True,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

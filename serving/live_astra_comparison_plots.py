"""Plot compact live LLMServingSim + ASTRA comparison campaigns.

The input is the nested JSON produced by
``serving.live_astra_comparison_collect``.  A simulation seed is the
independent replicate.  Means and confidence intervals are therefore computed
across seed-level cell summaries, never across individual requests.

The renderer fails closed on an incomplete system/rate/seed grid and on
partially reported metrics.  Optional bottleneck metrics are explicitly
recorded as ``not_reported`` in the summary CSV rather than silently replaced
with zero.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence


FONT_SIZE = 24
FIGURE_WIDTH_INCHES = 12
FIGURE_DPI = 200
PERFORMANCE_FIGSIZE = (FIGURE_WIDTH_INCHES, 16)
BOTTLENECK_COLUMNS = 2
BOTTLENECK_ROW_HEIGHT_INCHES = 8

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
    "ssd_tiering": ("#d62728", "o", "-"),
    "oracle": ("#111111", "D", "--"),
    "hbf_tp4": ("#1f77b4", "s", "-"),
    "hbf_tp8": ("#2ca02c", "^", "-"),
    "hbf_tp8_context": ("#9467bd", "P", "-."),
}

_FALLBACK_STYLES = (
    ("#8c564b", "X", "-"),
    ("#17becf", "v", "--"),
    ("#e377c2", "<", "-."),
    ("#7f7f7f", ">", ":"),
)

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

_MISSING = object()


class LiveAstraPlotError(ValueError):
    """Raised when compact campaign data cannot be plotted faithfully."""


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    y_label: str
    unit: str
    category: str
    scale: float = 1.0
    positive: bool = False


TTFT_SPEC = MetricSpec(
    "resume_ttft_p95",
    "Resume TTFT p95",
    "Resume TTFT p95 (s)",
    "seconds",
    "performance",
    scale=1e-9,
    positive=True,
)
TPOT_SPEC = MetricSpec(
    "resume_tpot_p95",
    "Resume TPOT p95",
    "Resume TPOT p95 (ms/token)",
    "milliseconds_per_token",
    "performance",
    scale=1e-6,
    positive=True,
)
SLO_SPEC = MetricSpec(
    "joint_slo_pass_fraction",
    "Joint SLO pass fraction",
    "Joint SLO pass fraction",
    "fraction",
    "performance",
)
GOODPUT_OFFERED_SPEC = MetricSpec(
    "slo_goodput",
    "Offered-normalized SLO goodput",
    "Offered-normalized SLO goodput (requests/s)",
    "requests_per_second",
    "performance",
)
GOODPUT_OPERATIONAL_SPEC = MetricSpec(
    "slo_goodput",
    "Operational SLO goodput",
    "Operational SLO goodput (requests/s)",
    "requests_per_second",
    "performance",
)

SSD_HITS_SPEC = MetricSpec(
    "ssd_hits",
    "SSD cache hits",
    "SSD hits/run",
    "hits",
    "bottleneck",
)
SSD_BYTES_SPEC = MetricSpec(
    "ssd_traffic_gib",
    "SSD media traffic",
    "SSD read + write traffic (GiB/run)",
    "gibibytes",
    "bottleneck",
)
HBF_ROUTE_SPEC = MetricSpec(
    "hbf_route_fraction",
    "HBF routing fraction",
    "Requests routed to HBF (%/run)",
    "percent",
    "bottleneck",
    scale=100.0,
)
HBF_CAPACITY_SPEC = MetricSpec(
    "hbf_capacity_fraction",
    "HBF capacity utilization",
    "Peak reserved HBF / capacity (%/run)",
    "percent",
    "bottleneck",
    scale=100.0,
)
HBF_PREFILL_DRAIN_CANDIDATE_SPEC = MetricSpec(
    "hbf_prefill_drain_candidate_fraction",
    "Active HBF drain candidates",
    "Active HBF drain candidates (%/run)",
    "percent",
    "bottleneck",
    scale=100.0,
)
HBF_PREFILL_DRAIN_WAIT_SPEC = MetricSpec(
    "hbf_prefill_drain_mean_wait_ms",
    "Active HBF drain wait",
    "Mean active HBF drain wait (ms/run)",
    "milliseconds",
    "bottleneck",
)
HBF_PREFILL_DRAIN_FALLBACK_SPEC = MetricSpec(
    "hbf_prefill_drain_fallback_fraction",
    "Active HBF drain fallback",
    "Active HBF drain capacity fallbacks (%/run)",
    "percent",
    "bottleneck",
    scale=100.0,
)
HBF_PREFILL_DRAIN_TRAFFIC_SPEC = MetricSpec(
    "hbf_prefill_drain_logical_traffic_gib",
    "Active HBF drain traffic",
    "Active HBF drain logical traffic (GiB/run)",
    "gibibytes",
    "bottleneck",
)

BOTTLENECK_SPECS = (
    SSD_HITS_SPEC,
    SSD_BYTES_SPEC,
    HBF_ROUTE_SPEC,
    HBF_CAPACITY_SPEC,
    HBF_PREFILL_DRAIN_CANDIDATE_SPEC,
    HBF_PREFILL_DRAIN_WAIT_SPEC,
    HBF_PREFILL_DRAIN_FALLBACK_SPEC,
    HBF_PREFILL_DRAIN_TRAFFIC_SPEC,
)
BOTTLENECK_ROWS = math.ceil(
    len(BOTTLENECK_SPECS) / BOTTLENECK_COLUMNS)
BOTTLENECK_FIGSIZE = (
    FIGURE_WIDTH_INCHES,
    BOTTLENECK_ROW_HEIGHT_INCHES * BOTTLENECK_ROWS,
)

_OFFERED_GOODPUT_PATHS = (
    (
        "performance",
        "offered_normalized_request_slo_goodput_per_second",
    ),
    ("performance", "offered_normalized_slo_goodput_per_second"),
    ("performance", "offered_normalized_request_goodput_per_second"),
    ("performance", "offered_load_normalized_request_goodput_per_second"),
    ("performance", "offered_load_normalized_request_goodput", "value"),
)
_OPERATIONAL_GOODPUT_PATH = (
    "performance",
    "operational_request_goodput_per_second",
)


@dataclass(frozen=True)
class Cell:
    system: str
    rate: float
    seed: int
    values: Mapping[str, float]
    value_sources: Mapping[str, str]


@dataclass(frozen=True)
class Dataset:
    cells: tuple[Cell, ...]
    systems: tuple[str, ...]
    rates: tuple[float, ...]
    seeds: tuple[int, ...]
    goodput_basis: str
    goodput_source: str


@dataclass(frozen=True)
class AggregatePoint:
    system: str
    rate: float
    metric: str
    seed_count: int
    values: tuple[float, ...]
    mean: float
    sample_stddev: float | None
    ci95_half_width: float | None
    ci95_lower: float | None
    ci95_upper: float | None
    ci_method: str
    source: str


@dataclass(frozen=True)
class Aggregate:
    dataset: Dataset
    points: tuple[AggregatePoint, ...]


@dataclass(frozen=True)
class RenderOutputs:
    performance_png: Path
    performance_pdf: Path
    bottlenecks_png: Path
    bottlenecks_pdf: Path
    summary_csv: Path
    latency_log_metrics: tuple[str, ...]
    goodput_basis: str


def _read_json(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle, parse_constant=reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise LiveAstraPlotError(f"cannot parse compact JSON {path}") from exc
    if not isinstance(raw, dict):
        raise LiveAstraPlotError("compact campaign root must be an object")
    return raw


def _lookup(root: Mapping[str, object], path: Sequence[str]) -> object:
    value: object = root
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            return _MISSING
        value = value[component]
    return value


def _number(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiveAstraPlotError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LiveAstraPlotError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise LiveAstraPlotError(f"{name} must be >= {minimum}")
    return result


def _required_number(
    cell: Mapping[str, object],
    path: Sequence[str],
    *,
    minimum: float | None = None,
) -> float:
    value = _lookup(cell, path)
    name = ".".join(path)
    if value is _MISSING:
        raise LiveAstraPlotError(f"cell is missing required metric {name}")
    return _number(value, name, minimum=minimum)


def _optional_number(
    cell: Mapping[str, object],
    path: Sequence[str],
    *,
    minimum: float | None = None,
) -> float | None:
    value = _lookup(cell, path)
    if value is _MISSING:
        return None
    return _number(value, ".".join(path), minimum=minimum)


def _choose_goodput(
    raw_cells: Sequence[Mapping[str, object]],
) -> tuple[str, tuple[str, ...]]:
    complete_paths = []
    partial_paths = []
    for path in _OFFERED_GOODPUT_PATHS:
        present = tuple(_lookup(cell, path) is not _MISSING for cell in raw_cells)
        if all(present):
            complete_paths.append(path)
        elif any(present):
            partial_paths.append(path)
    if partial_paths:
        formatted = ", ".join(".".join(path) for path in partial_paths)
        raise LiveAstraPlotError(
            "offered-normalized goodput is only partially reported: "
            f"{formatted}"
        )
    if complete_paths:
        return "offered_normalized", complete_paths[0]
    operational = tuple(
        _lookup(cell, _OPERATIONAL_GOODPUT_PATH) is not _MISSING
        for cell in raw_cells
    )
    if not all(operational):
        raise LiveAstraPlotError(
            "compact campaign has neither complete offered-normalized nor "
            "complete operational SLO goodput"
        )
    return "operational", _OPERATIONAL_GOODPUT_PATH


def _optional_pair_sum(
    cell: Mapping[str, object],
    first: Sequence[str],
    second: Sequence[str],
) -> tuple[float, str] | None:
    first_raw = _lookup(cell, first)
    second_raw = _lookup(cell, second)
    if first_raw is _MISSING and second_raw is _MISSING:
        return None
    if first_raw is _MISSING or second_raw is _MISSING:
        raise LiveAstraPlotError(
            "partially reported bottleneck byte pair: "
            f"{'.'.join(first)} + {'.'.join(second)}"
        )
    value = (
        _number(first_raw, ".".join(first), minimum=0.0)
        + _number(second_raw, ".".join(second), minimum=0.0)
    )
    return value, f"{'.'.join(first)} + {'.'.join(second)}"


def _ssd_bytes(cell: Mapping[str, object]) -> tuple[float, str] | None:
    prefix = ("bottlenecks", "baseline", "ssd")
    pairs = (
        (
            prefix + ("media_host_read_bytes",),
            prefix + ("media_aligned_host_write_bytes",),
        ),
        (
            prefix + ("ssd_host_read_bytes",),
            prefix + ("ssd_host_write_bytes",),
        ),
        (
            prefix + ("direct_ssd_read_bytes",),
            prefix + ("direct_ssd_write_bytes",),
        ),
    )
    for first, second in pairs:
        result = _optional_pair_sum(cell, first, second)
        if result is not None:
            value, source = result
            return value / (1024.0 ** 3), source
    fallback_paths = (
        prefix + ("ssd_to_hbm_bytes",),
        prefix + ("ssd_to_cpu_stage_bytes",),
        prefix + ("cpu_to_ssd_bytes",),
        prefix + ("hbm_to_ssd_bytes",),
    )
    present = [
        (path, _lookup(cell, path))
        for path in fallback_paths
        if _lookup(cell, path) is not _MISSING
    ]
    if not present:
        return None
    total = math.fsum(
        _number(value, ".".join(path), minimum=0.0)
        for path, value in present
    )
    return (
        total / (1024.0 ** 3),
        " + ".join(".".join(path) for path, _ in present),
    )


def _hbf_prefill_drain(
    cell: Mapping[str, object],
) -> tuple[dict[str, float], dict[str, str]]:
    """Load one complete per-run active-drain metric group."""

    group_path = ("bottlenecks", "hbf", "prefill_drain")
    group = _lookup(cell, group_path)
    if group is _MISSING:
        return {}, {}
    if not isinstance(group, Mapping):
        raise LiveAstraPlotError(
            "bottlenecks.hbf.prefill_drain must be an object")
    base = group_path + ("derived",)
    raw = _lookup(cell, base)
    if not isinstance(raw, Mapping):
        raise LiveAstraPlotError(
            "HBF prefill-drain metrics are partially reported; "
            "missing=['derived']")
    fields = (
        (
            HBF_PREFILL_DRAIN_CANDIDATE_SPEC,
            "candidate_fraction",
        ),
        (
            HBF_PREFILL_DRAIN_WAIT_SPEC,
            "mean_wait_ms",
        ),
        (
            HBF_PREFILL_DRAIN_FALLBACK_SPEC,
            "fallback_fraction",
        ),
        (
            HBF_PREFILL_DRAIN_TRAFFIC_SPEC,
            "logical_traffic_gib",
        ),
    )
    missing = [
        field for _, field in fields
        if field not in raw
    ]
    if missing:
        raise LiveAstraPlotError(
            "HBF prefill-drain metrics are partially reported; "
            f"missing={missing}")

    values: dict[str, float] = {}
    sources: dict[str, str] = {}
    for spec, field in fields:
        path = base + (field,)
        values[spec.key] = _number(
            raw[field], ".".join(path), minimum=0.0)
        sources[spec.key] = ".".join(path)
    candidate_fraction = values[
        HBF_PREFILL_DRAIN_CANDIDATE_SPEC.key]
    fallback_fraction = values[
        HBF_PREFILL_DRAIN_FALLBACK_SPEC.key]
    if candidate_fraction > 1.0:
        raise LiveAstraPlotError(
            "HBF prefill-drain candidate fraction exceeds one")
    if fallback_fraction > 1.0:
        raise LiveAstraPlotError(
            "HBF prefill-drain fallback fraction exceeds one")
    if candidate_fraction == 0.0 and any(
            values[spec.key] != 0.0
            for spec in (
                HBF_PREFILL_DRAIN_WAIT_SPEC,
                HBF_PREFILL_DRAIN_FALLBACK_SPEC,
                HBF_PREFILL_DRAIN_TRAFFIC_SPEC,
            )):
        raise LiveAstraPlotError(
            "zero HBF prefill-drain candidates require zero wait, "
            "fallback, and traffic metrics")
    return values, sources


def _extract_bottlenecks(
    cell: Mapping[str, object],
) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    sources: dict[str, str] = {}

    hits_path = ("bottlenecks", "baseline", "ssd", "ssd_hits")
    hits = _optional_number(cell, hits_path, minimum=0.0)
    if hits is not None:
        values[SSD_HITS_SPEC.key] = hits
        sources[SSD_HITS_SPEC.key] = ".".join(hits_path)

    bytes_value = _ssd_bytes(cell)
    if bytes_value is not None:
        values[SSD_BYTES_SPEC.key], sources[SSD_BYTES_SPEC.key] = bytes_value

    route_base = ("bottlenecks", "hbf", "routing")
    offered_path = route_base + ("offered_requests",)
    hbf_path = route_base + ("hbf_requests",)
    offered_raw = _lookup(cell, offered_path)
    hbf_raw = _lookup(cell, hbf_path)
    if offered_raw is not _MISSING or hbf_raw is not _MISSING:
        if offered_raw is _MISSING or hbf_raw is _MISSING:
            raise LiveAstraPlotError(
                "HBF route fraction requires both offered_requests and "
                "hbf_requests"
            )
        offered = _number(offered_raw, ".".join(offered_path), minimum=0.0)
        hbf = _number(hbf_raw, ".".join(hbf_path), minimum=0.0)
        if offered <= 0.0 or hbf > offered:
            raise LiveAstraPlotError("invalid HBF routing counters")
        values[HBF_ROUTE_SPEC.key] = hbf / offered
        sources[HBF_ROUTE_SPEC.key] = (
            f"{'.'.join(hbf_path)} / {'.'.join(offered_path)}"
        )

    capacity_base = ("bottlenecks", "hbf", "capacity")
    peak_path = capacity_base + ("hbf_reserved_bytes_peak",)
    per_card_path = capacity_base + ("hbf_capacity_bytes_per_card",)
    cards_path = capacity_base + ("card_count",)
    capacity_raw = (
        _lookup(cell, peak_path),
        _lookup(cell, per_card_path),
        _lookup(cell, cards_path),
    )
    if any(value is not _MISSING for value in capacity_raw):
        if any(value is _MISSING for value in capacity_raw):
            raise LiveAstraPlotError(
                "HBF capacity fraction requires peak, per-card capacity, "
                "and card count"
            )
        peak = _number(capacity_raw[0], ".".join(peak_path), minimum=0.0)
        per_card = _number(
            capacity_raw[1], ".".join(per_card_path), minimum=0.0)
        cards = _number(capacity_raw[2], ".".join(cards_path), minimum=0.0)
        capacity = per_card * cards
        if capacity <= 0.0 or peak > capacity * (1.0 + 1e-12):
            raise LiveAstraPlotError("invalid HBF capacity counters")
        values[HBF_CAPACITY_SPEC.key] = peak / capacity
        sources[HBF_CAPACITY_SPEC.key] = (
            f"{'.'.join(peak_path)} / "
            f"({'.'.join(per_card_path)} * {'.'.join(cards_path)})"
        )
    drain_values, drain_sources = _hbf_prefill_drain(cell)
    values.update(drain_values)
    sources.update(drain_sources)
    return values, sources


def _system_sort_key(system: str) -> tuple[int, str]:
    try:
        return SYSTEM_ORDER.index(system), system
    except ValueError:
        return len(SYSTEM_ORDER), system


def load_compact_campaign(path: str | Path) -> Dataset:
    """Load and validate a compact campaign JSON."""

    root = _read_json(Path(path))
    raw = root.get("cells")
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(cell, Mapping) for cell in raw)
    ):
        raise LiveAstraPlotError("compact campaign cells must be non-empty objects")
    raw_cells = tuple(raw)
    goodput_basis, goodput_path = _choose_goodput(raw_cells)
    goodput_source = ".".join(goodput_path)
    cells = []
    identities = set()
    for index, raw_cell in enumerate(raw_cells):
        assert isinstance(raw_cell, Mapping)
        system = raw_cell.get("system")
        seed = raw_cell.get("seed")
        if not isinstance(system, str) or not system:
            raise LiveAstraPlotError(f"cell {index} has invalid system")
        if type(seed) is not int:
            raise LiveAstraPlotError(f"cell {index} has invalid seed")
        rate = _number(
            raw_cell.get("offered_session_rate_per_second"),
            f"cell {index} offered_session_rate_per_second",
            minimum=0.0,
        )
        if rate <= 0.0:
            raise LiveAstraPlotError("offered session rate must be positive")
        identity = (system, rate, seed)
        if identity in identities:
            raise LiveAstraPlotError(f"duplicate compact cell {identity!r}")
        identities.add(identity)

        ttft = _required_number(
            raw_cell,
            ("performance", "resume_ttft_ns", "p95_ns"),
            minimum=0.0,
        )
        tpot = _required_number(
            raw_cell,
            ("performance", "resume_tpot_ns", "p95_ns"),
            minimum=0.0,
        )
        if ttft <= 0.0 or tpot <= 0.0:
            raise LiveAstraPlotError("latency p95 metrics must be positive")
        slo = _required_number(
            raw_cell,
            ("performance", "joint_slo_pass_fraction"),
            minimum=0.0,
        )
        if slo > 1.0:
            raise LiveAstraPlotError("joint SLO pass fraction exceeds one")
        goodput = _required_number(raw_cell, goodput_path, minimum=0.0)
        values = {
            TTFT_SPEC.key: ttft,
            TPOT_SPEC.key: tpot,
            "slo_goodput": goodput,
            SLO_SPEC.key: slo,
        }
        sources = {
            TTFT_SPEC.key: "performance.resume_ttft_ns.p95_ns",
            TPOT_SPEC.key: "performance.resume_tpot_ns.p95_ns",
            "slo_goodput": goodput_source,
            SLO_SPEC.key: "performance.joint_slo_pass_fraction",
        }
        bottleneck_values, bottleneck_sources = _extract_bottlenecks(raw_cell)
        values.update(bottleneck_values)
        sources.update(bottleneck_sources)
        cells.append(Cell(system, rate, seed, values, sources))

    systems = tuple(sorted({cell.system for cell in cells}, key=_system_sort_key))
    rates = tuple(sorted({cell.rate for cell in cells}))
    seeds = tuple(sorted({cell.seed for cell in cells}))
    expected = {
        (system, rate, seed)
        for system in systems
        for rate in rates
        for seed in seeds
    }
    missing = sorted(expected - identities)
    if missing:
        preview = ", ".join(repr(value) for value in missing[:8])
        suffix = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
        raise LiveAstraPlotError(
            f"compact campaign grid is incomplete: {preview}{suffix}"
        )

    required_keys = {
        TTFT_SPEC.key,
        TPOT_SPEC.key,
        "slo_goodput",
        SLO_SPEC.key,
    }
    for metric in required_keys | {spec.key for spec in BOTTLENECK_SPECS}:
        for system in systems:
            availability = [
                metric in cell.values
                for cell in cells
                if cell.system == system
            ]
            if any(availability) and not all(availability):
                raise LiveAstraPlotError(
                    f"metric {metric} is partially reported for {system}"
                )
            if metric in required_keys and not all(availability):
                raise LiveAstraPlotError(
                    f"required metric {metric} is absent for {system}"
                )

    return Dataset(
        cells=tuple(sorted(
            cells,
            key=lambda cell: (
                _system_sort_key(cell.system),
                cell.rate,
                cell.seed,
            ),
        )),
        systems=systems,
        rates=rates,
        seeds=seeds,
        goodput_basis=goodput_basis,
        goodput_source=goodput_source,
    )


def _student_t_975(degrees_of_freedom: int) -> float:
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


def _aggregate_samples(
    system: str,
    rate: float,
    metric: str,
    values: Sequence[float],
    source: str,
) -> AggregatePoint:
    samples = tuple(float(value) for value in values)
    mean = math.fsum(samples) / len(samples)
    if len(samples) == 1:
        return AggregatePoint(
            system,
            rate,
            metric,
            1,
            samples,
            mean,
            None,
            None,
            None,
            None,
            "unavailable_single_seed",
            source,
        )
    variance = math.fsum(
        (value - mean) ** 2 for value in samples
    ) / (len(samples) - 1)
    sample_stddev = math.sqrt(variance)
    half_width = (
        _student_t_975(len(samples) - 1)
        * sample_stddev
        / math.sqrt(len(samples))
    )
    return AggregatePoint(
        system,
        rate,
        metric,
        len(samples),
        samples,
        mean,
        sample_stddev,
        half_width,
        mean - half_width,
        mean + half_width,
        "student_t_95",
        source,
    )


def aggregate_campaign(dataset: Dataset) -> Aggregate:
    """Aggregate complete system/rate cells across independent seeds."""

    indexed = {
        (cell.system, cell.rate, cell.seed): cell for cell in dataset.cells
    }
    metrics = (
        TTFT_SPEC.key,
        TPOT_SPEC.key,
        "slo_goodput",
        SLO_SPEC.key,
    ) + tuple(spec.key for spec in BOTTLENECK_SPECS)
    points = []
    for system in dataset.systems:
        for rate in dataset.rates:
            rows = [
                indexed[(system, rate, seed)] for seed in dataset.seeds
            ]
            for metric in metrics:
                if metric not in rows[0].values:
                    continue
                source_values = {row.value_sources[metric] for row in rows}
                if len(source_values) != 1:
                    raise LiveAstraPlotError(
                        f"metric {metric} changes source within {system}/{rate}"
                    )
                points.append(_aggregate_samples(
                    system,
                    rate,
                    metric,
                    [row.values[metric] for row in rows],
                    next(iter(source_values)),
                ))
    return Aggregate(dataset, tuple(points))


def _metric_spec(metric: str, goodput_basis: str) -> MetricSpec:
    if metric == TTFT_SPEC.key:
        return TTFT_SPEC
    if metric == TPOT_SPEC.key:
        return TPOT_SPEC
    if metric == SLO_SPEC.key:
        return SLO_SPEC
    if metric == "slo_goodput":
        return (
            GOODPUT_OFFERED_SPEC
            if goodput_basis == "offered_normalized"
            else GOODPUT_OPERATIONAL_SPEC
        )
    for spec in BOTTLENECK_SPECS:
        if spec.key == metric:
            return spec
    raise KeyError(metric)


def _points_for(
    aggregate: Aggregate,
    system: str,
    metric: str,
) -> tuple[AggregatePoint, ...]:
    return tuple(
        point
        for point in aggregate.points
        if point.system == system and point.metric == metric
    )


def _latency_warrants_log(
    aggregate: Aggregate,
    metric: str,
    *,
    dynamic_range_threshold: float = 8.0,
) -> bool:
    values = [
        point.mean
        for point in aggregate.points
        if point.metric == metric
    ]
    if not values or any(value <= 0.0 for value in values):
        return False
    return max(values) / min(values) >= dynamic_range_threshold


def _style(
    system: str,
    fallback_index: int,
) -> tuple[str, str, str]:
    return SYSTEM_STYLES.get(
        system,
        _FALLBACK_STYLES[fallback_index % len(_FALLBACK_STYLES)],
    )


def _label(system: str) -> str:
    return SYSTEM_LABELS.get(system, system.replace("_", " ").title())


def _plot_metric(
    axis: object,
    aggregate: Aggregate,
    spec: MetricSpec,
    *,
    log_y: bool = False,
) -> None:
    plotted = 0
    for index, system in enumerate(aggregate.dataset.systems):
        points = _points_for(aggregate, system, spec.key)
        if not points:
            continue
        color, marker, linestyle = _style(system, index)
        x = [point.rate for point in points]
        y = [point.mean * spec.scale for point in points]
        if any(point.ci95_half_width is not None for point in points):
            lower = []
            upper = []
            for point, display_mean in zip(points, y):
                half = (
                    0.0
                    if point.ci95_half_width is None
                    else point.ci95_half_width * spec.scale
                )
                if log_y:
                    lower.append(min(half, display_mean * (1.0 - 1e-9)))
                else:
                    lower.append(half)
                upper.append(half)
            yerr: object = [lower, upper]
        else:
            yerr = None
        axis.errorbar(
            x,
            y,
            yerr=yerr,
            label=_label(system),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.2,
            markersize=8,
            capsize=4,
        )
        plotted += 1
    axis.set_title(spec.title)
    axis.set_xlabel("Offered session rate (sessions/s)")
    axis.set_ylabel(spec.y_label)
    axis.grid(True, which="both", alpha=0.3)
    if log_y:
        axis.set_yscale("log")
    if spec.key == SLO_SPEC.key:
        axis.set_ylim(-0.03, 1.05)
    if plotted:
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "Not reported in compact JSON",
            transform=axis.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
        )


def _atomic_save_figure(
    figure: object,
    destination: Path,
    *,
    dpi: int | None = None,
) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=dpi)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _float_text(value: float | None) -> str:
    return "" if value is None else format(value, ".17g")


def _write_summary_csv(
    aggregate: Aggregate,
    destination: Path,
    log_metrics: set[str],
) -> None:
    point_index = {
        (point.system, point.rate, point.metric): point
        for point in aggregate.points
    }
    metric_order = (
        TTFT_SPEC.key,
        TPOT_SPEC.key,
        "slo_goodput",
        SLO_SPEC.key,
    ) + tuple(spec.key for spec in BOTTLENECK_SPECS)
    fields = (
        "system",
        "system_label",
        "rate_sps",
        "metric",
        "metric_title",
        "category",
        "unit",
        "status",
        "source",
        "goodput_basis",
        "seed_count",
        "seed_values_json",
        "arithmetic_mean",
        "sample_stddev",
        "ci95_half_width",
        "ci95_lower",
        "ci95_upper",
        "ci_method",
        "log_y_applied",
    )
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for system in aggregate.dataset.systems:
            for rate in aggregate.dataset.rates:
                for metric in metric_order:
                    spec = _metric_spec(
                        metric, aggregate.dataset.goodput_basis)
                    point = point_index.get((system, rate, metric))
                    common = {
                        "system": system,
                        "system_label": _label(system),
                        "rate_sps": _float_text(rate),
                        "metric": metric,
                        "metric_title": spec.title,
                        "category": spec.category,
                        "unit": spec.unit,
                        "goodput_basis": (
                            aggregate.dataset.goodput_basis
                            if metric == "slo_goodput"
                            else ""
                        ),
                        "log_y_applied": (
                            "true" if metric in log_metrics else "false"
                        ),
                    }
                    if point is None:
                        writer.writerow({
                            **common,
                            "status": "not_reported",
                            "source": "",
                            "seed_count": "",
                            "seed_values_json": "",
                            "arithmetic_mean": "",
                            "sample_stddev": "",
                            "ci95_half_width": "",
                            "ci95_lower": "",
                            "ci95_upper": "",
                            "ci_method": "",
                        })
                        continue
                    writer.writerow({
                        **common,
                        "status": "reported",
                        "source": point.source,
                        "seed_count": point.seed_count,
                        "seed_values_json": json.dumps(
                            point.values,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                        "arithmetic_mean": _float_text(
                            point.mean * spec.scale),
                        "sample_stddev": _float_text(
                            None
                            if point.sample_stddev is None
                            else point.sample_stddev * spec.scale
                        ),
                        "ci95_half_width": _float_text(
                            None
                            if point.ci95_half_width is None
                            else point.ci95_half_width * spec.scale
                        ),
                        "ci95_lower": _float_text(
                            None
                            if point.ci95_lower is None
                            else point.ci95_lower * spec.scale
                        ),
                        "ci95_upper": _float_text(
                            None
                            if point.ci95_upper is None
                            else point.ci95_upper * spec.scale
                        ),
                        "ci_method": point.ci_method,
                    })
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def render_campaign(
    compact_json: str | Path,
    output_dir: str | Path,
    *,
    prefix: str = "live_astra_comparison",
) -> RenderOutputs:
    """Render performance and bottleneck figures plus a long summary CSV."""

    if not prefix or Path(prefix).name != prefix:
        raise LiveAstraPlotError("prefix must be a non-empty filename stem")
    dataset = load_compact_campaign(compact_json)
    aggregate = aggregate_campaign(dataset)

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(MATPLOTLIB_RC)
    log_metrics = {
        metric
        for metric in (TTFT_SPEC.key, TPOT_SPEC.key)
        if _latency_warrants_log(aggregate, metric)
    }
    goodput_spec = (
        GOODPUT_OFFERED_SPEC
        if dataset.goodput_basis == "offered_normalized"
        else GOODPUT_OPERATIONAL_SPEC
    )
    performance_specs = (
        TTFT_SPEC,
        TPOT_SPEC,
        goodput_spec,
        SLO_SPEC,
    )
    performance, axes = plt.subplots(
        2,
        2,
        figsize=PERFORMANCE_FIGSIZE,
        constrained_layout=True,
    )
    performance_axes = tuple(axes.flat)
    if len(performance_axes) < len(performance_specs):
        raise RuntimeError(
            "performance subplot grid cannot hold every metric")
    for index, spec in enumerate(performance_specs):
        _plot_metric(
            performance_axes[index],
            aggregate,
            spec,
            log_y=spec.key in log_metrics,
        )
    for axis in performance_axes[len(performance_specs):]:
        axis.set_visible(False)

    bottleneck_rows = math.ceil(
        len(BOTTLENECK_SPECS) / BOTTLENECK_COLUMNS)
    bottlenecks, bottleneck_axes = plt.subplots(
        bottleneck_rows,
        BOTTLENECK_COLUMNS,
        figsize=(
            FIGURE_WIDTH_INCHES,
            BOTTLENECK_ROW_HEIGHT_INCHES * bottleneck_rows,
        ),
        constrained_layout=True,
    )
    flat_bottleneck_axes = tuple(bottleneck_axes.flat)
    if len(flat_bottleneck_axes) < len(BOTTLENECK_SPECS):
        raise RuntimeError(
            "bottleneck subplot grid cannot hold every metric")
    for index, spec in enumerate(BOTTLENECK_SPECS):
        _plot_metric(
            flat_bottleneck_axes[index],
            aggregate,
            spec,
        )
    for axis in flat_bottleneck_axes[len(BOTTLENECK_SPECS):]:
        axis.set_visible(False)

    output = Path(output_dir).resolve()
    performance_png = output / f"{prefix}_performance.png"
    performance_pdf = output / f"{prefix}_performance.pdf"
    bottlenecks_png = output / f"{prefix}_bottlenecks.png"
    bottlenecks_pdf = output / f"{prefix}_bottlenecks.pdf"
    summary_csv = output / f"{prefix}_summary.csv"
    try:
        _atomic_save_figure(
            performance, performance_png, dpi=FIGURE_DPI)
        _atomic_save_figure(performance, performance_pdf)
        _atomic_save_figure(
            bottlenecks, bottlenecks_png, dpi=FIGURE_DPI)
        _atomic_save_figure(bottlenecks, bottlenecks_pdf)
    finally:
        plt.close(performance)
        plt.close(bottlenecks)
    _write_summary_csv(aggregate, summary_csv, log_metrics)
    return RenderOutputs(
        performance_png,
        performance_pdf,
        bottlenecks_png,
        bottlenecks_pdf,
        summary_csv,
        tuple(sorted(log_metrics)),
        dataset.goodput_basis,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m serving.live_astra_comparison_plots",
        description=(
            "Plot compact live LLMServingSim + ASTRA comparison results"
        ),
    )
    parser.add_argument("compact_json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prefix", default="live_astra_comparison")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    compact_json = args.compact_json.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else compact_json.parent
    )
    outputs = render_campaign(
        compact_json,
        output_dir,
        prefix=args.prefix,
    )
    print(json.dumps({
        "performance_png": str(outputs.performance_png),
        "performance_pdf": str(outputs.performance_pdf),
        "bottlenecks_png": str(outputs.bottlenecks_png),
        "bottlenecks_pdf": str(outputs.bottlenecks_pdf),
        "summary_csv": str(outputs.summary_csv),
        "latency_log_metrics": list(outputs.latency_log_metrics),
        "goodput_basis": outputs.goodput_basis,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Standalone SSD endurance calculator.

Example::

    python -m serving.endurance \
      --stats outputs/run.storage.json \
      --device-profile configs/storage/micron_9550_pro_3_84tb.json \
      --num-devices 8 --trace-period-seconds 250 --duty-cycle 1 \
      --output-json outputs/run.endurance.json \
      --output-csv outputs/run.endurance.csv

The stats file represents one workload epoch.  It may contain aggregate
traffic::

    {
      "run_id": "run-1",
      "trace_period_seconds": 250,
      "host_write_bytes": 393315090432,
      "host_read_bytes": 0
    }

or explicit physical-device traffic::

    {
      "run_id": "run-1",
      "devices": [
        {"device_id": "ssd0", "host_write_bytes": 1000},
        {"device_id": "ssd1", "host_write_bytes": 3000}
      ]
    }

With aggregate traffic, ``--num-devices`` selects balanced distribution.
With explicit devices, their write counts determine the first-device-EOL
result and ``--num-devices`` may be omitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

from serving.core.endurance_model import (
    DeviceProfile,
    EnduranceConfigError,
    ProjectionAssumptions,
    RunWriteStats,
    project_endurance,
    write_report_csv,
    write_report_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m serving.endurance",
        description=(
            "Project SSD host-TBW endurance from one simulator storage "
            "trace. SSD capacities and TBW use decimal SI units."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--stats",
        help="Raw storage-statistics JSON for one workload epoch",
    )
    source.add_argument(
        "--analysis-report",
        help="JSON emitted by python -m serving.agentic_kv_analyze",
    )
    parser.add_argument(
        "--analysis-model",
        help="Model row to select with --analysis-report",
    )
    parser.add_argument(
        "--analysis-hardware",
        help="Hardware row to select with --analysis-report",
    )
    parser.add_argument(
        "--analysis-traffic",
        choices=("tiered", "full-rewrite", "incremental-lower-bound"),
        default="tiered",
        help="Write counter selected from --analysis-report. Default: tiered.",
    )
    parser.add_argument(
        "--device-profile", required=True,
        help="SSD endurance profile JSON under configs/storage or another path",
    )
    parser.add_argument(
        "--rating", default=None,
        help=(
            "Workload-specific rating from the profile. Defaults to the "
            "profile's conservative/default rating"
        ),
    )
    parser.add_argument(
        "--num-devices", type=int, default=None,
        help=(
            "Number of identical SSDs for balanced aggregate writes. "
            "Default: 1; inferred when stats contain explicit devices"
        ),
    )

    replay = parser.add_mutually_exclusive_group()
    replay.add_argument(
        "--replays-per-day", type=float, default=None,
        help="Direct number of trace epochs replayed per day",
    )
    replay.add_argument(
        "--trace-period-seconds", type=float, default=None,
        help=(
            "Offered-load duration represented by one trace epoch. If both "
            "replay flags are omitted, use trace_period_seconds in --stats"
        ),
    )
    parser.add_argument(
        "--duty-cycle", type=float, default=1.0,
        help=(
            "Fraction of wall time replaying the offered trace, in [0,1]. "
            "Used with trace period; default: 1"
        ),
    )
    parser.add_argument(
        "--waf", type=float, default=1.0,
        help=(
            "Write amplification factor for diagnostic NAND-write estimates. "
            "It never scales host-TBW lifetime; default: 1"
        ),
    )
    parser.add_argument(
        "--days-per-year", type=float, default=365.0,
        help="Days per projected service year; default: 365",
    )
    parser.add_argument(
        "--background-dwpd", type=float, default=0.0,
        help="Additional non-KV host writes per device per day; default: 0",
    )
    parser.add_argument(
        "--initial-percentage-used", type=float, default=0.0,
        help=(
            "Existing NVMe Percentage Used before this workload; default: 0"
        ),
    )
    parser.add_argument(
        "--replay-semantics", default="new_logical_sessions",
        help=(
            "Label recorded in the report, e.g. new_logical_sessions, "
            "cold_reset, or measured_steady_state"
        ),
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Write the full projection report as JSON",
    )
    parser.add_argument(
        "--output-csv", default=None,
        help="Write one projection row per physical SSD as CSV",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.analysis_report:
            if not args.analysis_model or not args.analysis_hardware:
                parser.error(
                    "--analysis-report requires --analysis-model and "
                    "--analysis-hardware")
            stats = _stats_from_analysis_report(
                args.analysis_report,
                args.analysis_model,
                args.analysis_hardware,
                args.analysis_traffic,
            )
        else:
            stats = RunWriteStats.from_json_file(args.stats)
        profile = DeviceProfile.from_json_file(args.device_profile)
        assumptions = ProjectionAssumptions(
            replays_per_day=args.replays_per_day,
            trace_period_seconds=args.trace_period_seconds,
            duty_cycle=args.duty_cycle,
            waf=args.waf,
            days_per_year=args.days_per_year,
            background_dwpd=args.background_dwpd,
            initial_percentage_used=args.initial_percentage_used,
            replay_semantics=args.replay_semantics,
        )
        report = project_endurance(
            stats,
            profile,
            assumptions,
            num_devices=args.num_devices,
            rating_name=args.rating,
        )
        if args.output_json:
            write_report_json(report, args.output_json)
        if args.output_csv:
            write_report_csv(report, args.output_csv)
    except (EnduranceConfigError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if not args.output_json and not args.output_csv:
        json.dump(report.to_dict(), sys.stdout, indent=2, sort_keys=True,
                  allow_nan=False)
        sys.stdout.write("\n")
    return 0


def _stats_from_analysis_report(
    path: str,
    model: str,
    hardware: str,
    traffic: str,
) -> RunWriteStats:
    with open(path, "r", encoding="utf-8") as report_file:
        report = json.load(report_file)
    matches = [
        summary for summary in report.get("summaries", [])
        if summary.get("model") == model and summary.get("hardware") == hardware
    ]
    if len(matches) != 1:
        raise EnduranceConfigError(
            f"analysis report must contain exactly one {model}/{hardware} row; "
            f"found {len(matches)}")
    summary = matches[0]
    if traffic == "tiered":
        host_writes = summary["tiered_policy"]["ssd_host_write_bytes"]
    elif traffic == "full-rewrite":
        write_counters = summary["ssd_swap"]["host_write_bytes"]
        if "full_rewrite_issued_under_selected_mode" not in write_counters:
            raise EnduranceConfigError(
                "analysis report predates issued-byte accounting for "
                "cancellable full rewrites; regenerate it with "
                "python -m serving.agentic_kv_analyze")
        host_writes = write_counters[
            "full_rewrite_issued_under_selected_mode"]
    else:
        host_writes = summary["ssd_swap"]["host_write_bytes"][
            "optimistic_incremental_append_lower_bound"]
    return RunWriteStats(
        run_id=f"{Path(path).stem}:{model}:{hardware}:{traffic}",
        host_write_bytes=int(host_writes),
    )


if __name__ == "__main__":
    sys.exit(main())

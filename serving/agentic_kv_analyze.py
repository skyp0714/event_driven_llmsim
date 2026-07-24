"""CLI for standalone agentic idle-KV swap/recompute analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .core.agentic_kv_roofline import (
    AnalysisConfigError,
    build_report,
    load_agentic_workload,
    load_hardware_config,
    load_model_shape,
    override_transfer_defaults,
    write_report_csv,
    write_report_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate TP idle-KV CPU/SSD transfer and roofline recomputation "
            "overhead from an agentic JSONL workload."
        )
    )
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Model config name under configs/model, for example "
            "Qwen/Qwen3-30B-A3B-Instruct-2507. Repeat for multiple models."
        ),
    )
    parser.add_argument(
        "--hardware",
        action="append",
        help="Hardware name from defaults/config. Defaults to H100 and H200.",
    )
    parser.add_argument("--hardware-config", type=Path)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--kv-dtype-bytes", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=0,
        help=(
            "Exclude transitions whose completed or next context exceeds this "
            "common model limit. 0 disables filtering."
        ),
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=2048,
        help="Chunk size used by the recompute roofline. Default: 2048.",
    )
    parser.add_argument(
        "--swap-out-mode",
        choices=("cancellable", "blocking"),
        default="cancellable",
        help=(
            "Continuation behavior when swap-out exceeds the tool wait. "
            "Default: cancellable (retain/use HBM); blocking is a sensitivity."
        ),
    )
    parser.add_argument("--hbm-ttl-ms", type=float, default=50.0)
    parser.add_argument("--cpu-ttl-ms", type=float, default=30_000.0)
    parser.add_argument("--ssd-ttl-ms", type=float, default=3_600_000.0)
    parser.add_argument(
        "--tiered-ssd-write-mode",
        choices=("incremental", "full"),
        default="incremental",
    )
    parser.add_argument(
        "--kv-layout",
        choices=("replicated", "logical-even", "simulator-even"),
        default="replicated",
        help=("Use physical KV-head replication or a logical-even sensitivity. "
              "simulator-even is a deprecated alias for logical-even."),
    )
    parser.add_argument(
        "--include-zero-tool-duration",
        action="store_true",
        help="Include adjacent calls with no idle interval.",
    )
    parser.add_argument(
        "--cpu-rank-gbps",
        type=float,
        help="Override effective GPU/host bandwidth in both directions per TP rank.",
    )
    parser.add_argument(
        "--cpu-aggregate-gbps",
        type=float,
        help="Override aggregate host-DRAM bandwidth in both directions.",
    )
    parser.add_argument("--ssd-read-gbps", type=float)
    parser.add_argument("--ssd-write-gbps", type=float)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--output-stem", default="agentic_kv_analysis")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        workload = load_agentic_workload(
            args.workload,
            block_size=args.block_size,
            include_zero_tool_duration=args.include_zero_tool_duration,
            max_context_tokens=(
                args.max_context_tokens if args.max_context_tokens > 0 else None),
        )
        all_hardware = load_hardware_config(args.hardware_config)
        selected_names = args.hardware or ["H100", "H200"]
        missing = [name for name in selected_names if name not in all_hardware]
        if missing:
            parser.error(
                "unknown hardware: "
                + ", ".join(missing)
                + "; add it with --hardware-config"
            )
        hardware_specs = [
            override_transfer_defaults(
                all_hardware[name],
                cpu_rank_gbps=args.cpu_rank_gbps,
                cpu_aggregate_gbps=args.cpu_aggregate_gbps,
                ssd_read_gbps=args.ssd_read_gbps,
                ssd_write_gbps=args.ssd_write_gbps,
            )
            for name in selected_names
        ]
        models = [load_model_shape(model_name) for model_name in args.model]
        report = build_report(
            workload,
            models,
            hardware_specs,
            tp_size=args.tp_size,
            kv_dtype_bytes=args.kv_dtype_bytes,
            kv_layout_mode=args.kv_layout,
            prefill_chunk_size=args.prefill_chunk_size,
            swap_out_mode=args.swap_out_mode,
            hbm_ttl_ms=args.hbm_ttl_ms,
            cpu_ttl_ms=args.cpu_ttl_ms,
            ssd_ttl_ms=args.ssd_ttl_ms,
            tiered_ssd_write_mode=args.tiered_ssd_write_mode,
        )
        json_path = args.output_dir / f"{args.output_stem}.json"
        csv_path = args.output_dir / f"{args.output_stem}.csv"
        write_report_json(report, json_path)
        write_report_csv(report, csv_path)
    except (AnalysisConfigError, FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    for summary in report["summaries"]:
        cpu_ms = summary["cpu_swap"]["exposed_seconds"]["p50"] * 1e3
        ssd_ms = summary["ssd_swap"]["exposed_seconds"]["p50"] * 1e3
        recompute_ms = (
            summary["recompute"]["avoidable_reusable_prefix_seconds"]["p50"]
            * 1e3
        )
        profile_available = summary["profile_provenance"][
            "requested_tp_profile_available"
        ]
        accounting = summary["time_accounting"]
        recompute = summary["recompute"]
        hbf = summary["hbf_npu_opportunity"]
        reuse = summary["prefix_reuse"]
        analyzed_prefill_fraction = recompute[
            "avoidable_reusable_prefix_fraction_of_analyzed_next_prefill"
        ]
        analyzed_prefill_text = (
            f"{analyzed_prefill_fraction * 100:.2f}%"
            if analyzed_prefill_fraction is not None else "N/A"
        )
        hbf_fraction = hbf["eligible_resume_fraction"]
        hbf_fraction_text = (
            f"{hbf_fraction * 100:.2f}%"
            if hbf_fraction is not None else "N/A"
        )
        modeled_stall_fraction = accounting[
            "migration_stall_fraction_of_modeled_serialized_selected_transition_time"
        ]
        modeled_active_stall_fraction = accounting[
            "migration_stall_fraction_of_modeled_prompt_active_time"
        ]
        print(
            f"{summary['model']} {summary['hardware']} TP{summary['tp_size']}: "
            f"p50 exposed CPU={cpu_ms:.3f} ms, SSD={ssd_ms:.3f} ms, "
            f"recompute={recompute_ms:.3f} ms"
        )
        print(
            "  aggregate tiered migration stall="
            f"{accounting['aggregate_tiered_request_stall_seconds']:.6f} s; "
            "stall/prompt-only modeled active time="
            f"{modeled_active_stall_fraction * 100:.3f}%; "
            "stall/prompt-only serialized lower bound="
            f"{modeled_stall_fraction * 100:.4f}%; "
            "stall/total simulated wall time=N/A "
            "(standalone trace has no complete execution timeline)"
        )
        print(
            "  avoidable prefix recompute/analyzed next-turn prefill="
            f"{analyzed_prefill_text}; recompute/total simulation compute=N/A "
            "(decode and cycle-level compute denominator unavailable)"
        )
        print(
            f"  prefix reuse p50={reuse['effective_reuse_fraction_of_next_input']['p50'] * 100:.2f}% "
            f"of next input; sources={dict(reuse['source_counts'])}"
        )
        print(
            f"  HBF-eligible beyond-HBM resumes={hbf_fraction_text}, "
            f"bytes={hbf['eligible_restore_bytes'] / 1e9:.3f} GB, "
            "gross stall upper bound="
            f"{hbf['gross_avoidable_stall_upper_bound_seconds']:.6f} s; "
            f"SSD-only resumes={hbf['ssd_only_resume_fraction'] * 100:.2f}%"
        )
        print(
            "  measured TP profile="
            f"{'yes' if profile_available else 'NO (roofline only)'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

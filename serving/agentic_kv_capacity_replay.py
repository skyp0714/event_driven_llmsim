"""CLI for the global capacity-aware agentic KV replay."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .core.agentic_kv_capacity_replay import (
    DGX_H100_CM6_IDEAL_READ_GBPS,
    DGX_H100_CM6_IDEAL_WRITE_GBPS,
    DGX_H100_CM6_READ_GBPS_PER_DEVICE,
    DGX_H100_CM6_WRITE_GBPS_PER_DEVICE,
    DGX_H100_NVLINK_ONE_WAY_GBPS_PER_GPU,
    GIB,
    SI_TB,
    CapacityReplayConfig,
    load_capacity_replay_workload,
    replay_capacity_aware,
    replay_capacity_aware_with_oracle,
    write_capacity_report,
)
from .core.agentic_kv_roofline import (
    AnalysisConfigError,
    load_hardware_config,
    load_model_shape,
    override_transfer_defaults,
)


_DEFAULT_HBM_SI_GB = {"H100": 80.0, "H200": 141.0}
_DEFAULT_CPU_SI_BYTES = 2_000_000_000_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay concurrent agentic sessions with finite HBM/CPU/SSD KV "
            "budgets, cascade LRU eviction, and transfer-resource queues."
        )
    )
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--hardware", choices=("H100", "H200"), required=True)
    parser.add_argument("--hardware-config", type=Path)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--kv-dtype-bytes", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-context-tokens", type=int, default=0)
    hbm_capacity = parser.add_mutually_exclusive_group()
    hbm_capacity.add_argument(
        "--hbm-capacity-gb-per-rank",
        type=float,
        help=(
            "Total device capacity in SI GB. Defaults to the marketed "
            "H100=80 GB or H200=141 GB."
        ),
    )
    hbm_capacity.add_argument(
        "--hbm-capacity-gib-per-rank",
        type=float,
        help="Explicit binary-GiB capacity sensitivity.",
    )
    parser.add_argument(
        "--hbm-static-reserve-gib-per-rank",
        type=float,
        default=0.0,
        help=(
            "Additional non-weight HBM reserve. Active KV is modeled "
            "separately and must not be included here."
        ),
    )
    parser.add_argument(
        "--prefill-hbm-static-reserve-gib-per-rank",
        type=float,
        help=(
            "Optional P-role non-KV reserve. Defaults to the common HBM "
            "reserve; use this to represent long-prefill activation and "
            "workspace separately from the D role."
        ),
    )
    parser.add_argument(
        "--decode-hbm-static-reserve-gib-per-rank",
        type=float,
        help=(
            "Optional D-role non-KV reserve. Defaults to the common HBM "
            "reserve."
        ),
    )
    parser.add_argument(
        "--cpu-kv-budget-gib",
        type=float,
        default=_DEFAULT_CPU_SI_BYTES / GIB,
        help=(
            "Shared CPU KV budget in binary GiB. Default: exact 2 TB SI "
            "(2,000,000,000,000 bytes; about 1862.645 GiB), matching the "
            "marketed DGX H100 system-memory capacity."
        ),
    )
    parser.add_argument(
        "--ssd-kv-budget-tb",
        type=float,
        default=30.72,
        help="Aggregate SSD KV budget in SI TB. Default: 8 x 3.84 TB.",
    )
    parser.add_argument(
        "--policy",
        choices=("hbm_lru_recompute", "hbm_ssd_direct", "tiered"),
        default="tiered",
        help="Cold-session capacity baseline to replay.",
    )
    parser.add_argument(
        "--demotion-mode",
        choices=("ttl-and-capacity", "capacity-only"),
        default="ttl-and-capacity",
        help=(
            "Use TTL plus capacity pressure, or disable all TTL actions so "
            "only capacity-driven LRU placement remains."
        ),
    )
    parser.add_argument("--hbm-ttl-ms", type=float, default=50.0)
    parser.add_argument("--cpu-ttl-ms", type=float, default=30_000.0)
    parser.add_argument("--ssd-ttl-ms", type=float, default=3_600_000.0)
    parser.add_argument("--prefill-chunk-size", type=int, default=2048)
    parser.add_argument(
        "--prompt-compute-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply full and cached analytical prompt-roofline times by "
            "this factor. Non-identity values are sensitivity endpoints, "
            "not measured DCA profiles, and require provenance."
        ),
    )
    parser.add_argument(
        "--prompt-compute-scale-provenance",
        help=(
            "Required explanation/source for a non-identity prompt compute "
            "scale; embedded verbatim in the report."
        ),
    )
    parser.add_argument("--weight-dtype-bytes", type=int, default=2)
    parser.add_argument(
        "--single-hbm-pool",
        action="store_true",
        help="Disable the default disaggregated P/D pair for a sensitivity run.",
    )
    parser.add_argument(
        "--pd-link-gbps-per-rank",
        type=float,
        default=DGX_H100_NVLINK_ONE_WAY_GBPS_PER_GPU,
        help=(
            "One-way P/D peer-copy sensitivity in SI GB/s per GPU. The "
            "default is 450 GB/s, half of the DGX H100 guide's 900 GB/s "
            "GPU-to-GPU bandwidth figure; calibrate with peer-copy data."
        ),
    )
    parser.add_argument("--pd-fixed-latency-us", type=float, default=3.0)
    parser.add_argument(
        "--restore-execution-mode",
        choices=(
            "async-pre-admission",
            "async-decode-join",
            "serial-before-prefill",
        ),
        default="async-pre-admission",
        help=(
            "Reserve destination HBM and trigger request-local restore at "
            "request ready, but keep that request out of analytical compute "
            "until all KV arrives (default). async-decode-join retains the "
            "optimistic suffix-overlap sensitivity; serial-before-prefill is "
            "a compatibility alias for the default request-local gate."
        ),
    )
    parser.add_argument("--no-transfer-queueing", action="store_true")
    parser.add_argument(
        "--compare-infinite-hbm-oracle",
        action="store_true",
        help=(
            "Run a paired capacity-only replay with a provably nonbinding "
            "HBM residency budget while preserving normal P/D transfers."
        ),
    )
    parser.add_argument(
        "--cancel-migration-on-resume",
        action="store_true",
        help=(
            "Cancellable no-queue sensitivity. With queueing, the default "
            "waits for an in-flight nonpreemptive copy to avoid ghost jobs."
        ),
    )
    parser.add_argument("--cpu-rank-gbps", type=float, default=50.0)
    parser.add_argument("--cpu-aggregate-gbps", type=float, default=400.0)
    parser.add_argument(
        "--ssd-read-gbps",
        type=float,
        default=DGX_H100_CM6_IDEAL_READ_GBPS,
        help=(
            "Aggregate SSD read sensitivity in decimal GB/s. Default: "
            f"{DGX_H100_CM6_IDEAL_READ_GBPS:.1f} GB/s, the manufacturer "
            f"upper-bound sum of eight KIOXIA CM6 3.84-TB drives x "
            f"{DGX_H100_CM6_READ_GBPS_PER_DEVICE:.1f} GB/s nameplate "
            "throughput. Calibrate end-to-end RAID 0 with fio."
        ),
    )
    parser.add_argument(
        "--ssd-write-gbps",
        type=float,
        default=DGX_H100_CM6_IDEAL_WRITE_GBPS,
        help=(
            "Aggregate SSD write sensitivity in decimal GB/s. Default: "
            f"{DGX_H100_CM6_IDEAL_WRITE_GBPS:.1f} GB/s, the manufacturer "
            f"upper-bound sum of eight KIOXIA CM6 3.84-TB drives x "
            f"{DGX_H100_CM6_WRITE_GBPS_PER_DEVICE:.1f} GB/s nameplate "
            "throughput. Calibrate end-to-end RAID 0 with fio."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.hbm_capacity_gib_per_rank is not None:
            hbm_bytes = int(args.hbm_capacity_gib_per_rank * GIB)
        else:
            hbm_gb = (
                args.hbm_capacity_gb_per_rank
                if args.hbm_capacity_gb_per_rank is not None
                else _DEFAULT_HBM_SI_GB[args.hardware]
            )
            hbm_bytes = int(hbm_gb * 1_000_000_000)
        if min(
            hbm_bytes,
            args.cpu_kv_budget_gib,
            args.ssd_kv_budget_tb,
        ) <= 0:
            raise AnalysisConfigError("capacity arguments must be positive")
        if (
            args.prompt_compute_scale != 1.0
            and not args.prompt_compute_scale_provenance
        ):
            raise AnalysisConfigError(
                "a non-identity --prompt-compute-scale requires "
                "--prompt-compute-scale-provenance"
            )
        workload = load_capacity_replay_workload(
            args.workload,
            block_size=args.block_size,
            max_context_tokens=(
                args.max_context_tokens if args.max_context_tokens > 0 else None
            ),
        )
        hardware_specs = load_hardware_config(args.hardware_config)
        hardware = override_transfer_defaults(
            hardware_specs[args.hardware],
            cpu_rank_gbps=args.cpu_rank_gbps,
            cpu_aggregate_gbps=args.cpu_aggregate_gbps,
            ssd_read_gbps=args.ssd_read_gbps,
            ssd_write_gbps=args.ssd_write_gbps,
        )
        model = load_model_shape(args.model)
        config = CapacityReplayConfig(
            hbm_capacity_bytes_per_rank=hbm_bytes,
            cpu_capacity_bytes=int(args.cpu_kv_budget_gib * GIB),
            ssd_capacity_bytes=int(args.ssd_kv_budget_tb * SI_TB),
            hbm_static_reserve_bytes_per_rank=int(
                args.hbm_static_reserve_gib_per_rank * GIB
            ),
            prefill_hbm_static_reserve_bytes_per_rank=(
                None
                if args.prefill_hbm_static_reserve_gib_per_rank is None
                else int(
                    args.prefill_hbm_static_reserve_gib_per_rank * GIB
                )
            ),
            decode_hbm_static_reserve_bytes_per_rank=(
                None
                if args.decode_hbm_static_reserve_gib_per_rank is None
                else int(
                    args.decode_hbm_static_reserve_gib_per_rank * GIB
                )
            ),
            policy=args.policy,
            demotion_mode=args.demotion_mode,
            hbm_ttl_ns=int(args.hbm_ttl_ms * 1e6),
            cpu_ttl_ns=int(args.cpu_ttl_ms * 1e6),
            ssd_ttl_ns=int(args.ssd_ttl_ms * 1e6),
            block_size=args.block_size,
            prefill_chunk_size=args.prefill_chunk_size,
            enable_transfer_queueing=not args.no_transfer_queueing,
            cancel_migration_on_resume=args.cancel_migration_on_resume,
            weight_dtype_bytes=args.weight_dtype_bytes,
            pd_disaggregated=not args.single_hbm_pool,
            pd_link_gbps_per_rank=args.pd_link_gbps_per_rank,
            pd_fixed_latency_us=args.pd_fixed_latency_us,
            restore_execution_mode=args.restore_execution_mode,
            prompt_compute_scale=args.prompt_compute_scale,
            prompt_compute_scale_provenance=(
                args.prompt_compute_scale_provenance
            ),
        )
        replay = (
            replay_capacity_aware_with_oracle
            if args.compare_infinite_hbm_oracle
            else replay_capacity_aware
        )
        report = replay(
            workload,
            model,
            hardware,
            args.tp_size,
            args.kv_dtype_bytes,
            config,
        )
        write_capacity_report(report, args.output)
    except (AnalysisConfigError, FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    resume = report["resume"]
    recompute = report["recompute"]
    fractions = resume["source_fractions_of_reuse_eligible"]
    all_request_fractions = resume["source_fractions_of_all_requests"]
    hbm_source = "decode_hbm" if report["execution_scope"][
        "pd_disaggregated"
    ] else "hbm"
    print(f"Wrote {args.output}")
    print(
        "reuse-eligible sources: "
        + ", ".join(
            f"{tier}={fractions[tier] * 100:.3f}%"
            for tier in (hbm_source, "cpu", "ssd", "recompute")
        )
    )
    print(
        f"all-request denominator ({resume['all_request_count']} LLM calls): "
        f"cpu={all_request_fractions['cpu'] * 100:.3f}%, "
        f"ssd={all_request_fractions['ssd'] * 100:.3f}%, "
        "cpu+ssd="
        f"{resume['cpu_or_ssd_resume_fraction_of_all_requests'] * 100:.3f}%"
    )
    print(
        "recompute: events="
        f"{recompute['event_fraction_of_reuse_eligible_transitions'] * 100:.4f}%, "
        "tokens="
        f"{recompute['token_fraction_of_reusable_tokens_requested'] * 100:.4f}%, "
        "analytical prompt compute="
        f"{recompute['analytical_time_fraction_of_executed_prompt_compute'] * 100:.4f}%"
    )
    restore_timing = resume["restore_timing"]
    print(
        "restore join: raw="
        f"{restore_timing['request_summed_raw_elapsed_seconds']:.6f}s, "
        "hidden-by-prefill="
        f"{restore_timing['request_summed_hidden_by_prefill_seconds']:.6f}s, "
        "exposed owner gate="
        f"{restore_timing['request_summed_exposed_compute_admission_gate_seconds']:.6f}s, "
        "other concurrent/admission="
        f"{restore_timing['request_summed_other_concurrent_or_admission_seconds']:.6f}s"
    )
    activity = report["offered_load_call_activity"]
    print(
        "offered call activity (not utilization): no-active-call="
        f"{activity['no_active_call_fraction'] * 100:.4f}% of trace window"
    )
    comparison = report.get("infinite_hbm_oracle_comparison")
    if comparison is not None:
        service = comparison["all_calls"]
        trace = comparison["trace_makespan"]
        print(
            "infinite-HBM residency reference: request-summed service "
            f"slowdown={service['slowdown_fraction_of_oracle_request_summed_service'] * 100:.4f}%, "
            "trace-makespan slowdown="
            f"{trace['slowdown_fraction_of_oracle'] * 100:.6f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

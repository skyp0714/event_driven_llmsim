"""CLI dispatch for workload generators.

Usage:
    python -m workloads.generators sharegpt --model <hf-id> --num-reqs 300 --sps 10 \
        --source <path-or-hf-id> --output workloads/sharegpt-<model>-<n>-sps<r>.jsonl
    python -m workloads.generators agent-traces --format lmcache --sps 0.2 \
        --source <path-or-hf-id> --output workloads/agentic.jsonl
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="workloads.generators")
    sub = parser.add_subparsers(dest="generator", required=True)

    sg = sub.add_parser("sharegpt", help="ShareGPT -> LLMServingSim JSONL")
    from workloads.generators.sharegpt import register_args as sg_register
    sg_register(sg)

    agent = sub.add_parser(
        "agent-traces",
        help="Exgentic / TraceLab / LMCache -> agentic LLMServingSim JSONL",
    )
    from workloads.generators.agent_traces import register_args as agent_register
    agent_register(agent)

    args = parser.parse_args()

    if args.generator == "sharegpt":
        from workloads.generators.sharegpt import run
        return run(args)

    if args.generator == "agent-traces":
        from workloads.generators.agent_traces import run
        return run(args)

    parser.error(f"Unknown generator: {args.generator}")
    return 2


if __name__ == "__main__":
    sys.exit(main())

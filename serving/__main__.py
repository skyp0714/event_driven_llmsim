"""Simulation entry point: ``python -m serving --cluster-config <...> [...]``.

Parses CLI args, generates ASTRA-Sim input files via ``serving.core.config_builder``,
spawns the ASTRA-Sim subprocess, and runs the iteration loop:
``router.route -> scheduler.schedule -> trace_generator -> graph -> ASTRA-Sim
-> scheduler.add_done`` until every request completes.
"""

import os
import subprocess
import argparse
import json
import math
import shutil
from time import time
from collections import defaultdict
from pathlib import Path

from serving.core.scheduler import *
from serving.core.request import *
from serving.core.utils import *
from serving.core.controller import *
from serving.core.memory_model import *
from serving.core.graph_generator import *
from serving.core.trace_generator import *
from serving.core.pim_model import *
from serving.core.config_builder import *
from serving.core.router import *
from serving.core.power_model import *
from serving.core.agentic_kv import AgenticKVConfig, AgenticKVManager
from serving.core.logger import *
from serving.core.run_paths import build_run_paths, resolve_run_id
from serving.core.session_admission import (
    add_session_admission_arguments,
    session_admission_from_args,
)
from serving.core.session_metrics import (
    build_session_metrics,
    save_session_metrics,
)
from serving.core.online_measurement import (
    OnlineHBMOccupancyAccounting,
    OnlineModelComputeAccounting,
    StrictInfiniteHBMOracle,
    configure_strict_oracle,
)
from serving.core.online_latency_model import (
    SUPPORTED_ONLINE_LATENCY_MODELS,
    resolve_online_latency_model,
)
from serving.core.hbf_online_runtime import (
    FullModelHBFRuntimeOptions,
    build_full_model_hbf_online_runtime,
    load_full_model_hbf_hardware,
    validate_full_model_hbf_gpu_cluster,
)
import sys as flush

from pyinstrument import Profiler


def _pad_batch_to_max(batch, max_len):
    """Pad a batch up to ``max_len`` for DP-sync.

    Mirrors vLLM's CUDA-graph DP padding: every DP rank's forward runs at
    ``max(num_tokens_across_dp)``. We bump the high-level counters so
    dense layers, lm_head, and the MoE compute path all reflect the
    padded shape — but we deliberately leave ``decode_k_list`` /
    prefill lists untouched so attention continues to see only the real
    decodes. FlashAttention's varlen kernel gives padded ``seq_len=0``
    entries zero compute in real vLLM, and extending ``decode_k_list``
    with ``kv=1`` dummies would instead collapse ``kv_decode_mean``
    toward 1 and push the attention lookup far outside the profiled
    sweep.

    MoE AG/RS comm size is anchored separately to ``max_total_len`` (no
    ``× group_size``) in the iteration loop — that calibrates the
    bandwidth model against the same ``link_bw`` AllReduce already uses.

    Request-completion accounting (`scheduler.add_done`) reads
    ``batch.requests`` and ``batch.end``, not these mutated token-list
    fields, so it is unaffected.
    """
    pad = max_len - batch.total_len
    if pad <= 0:
        return
    batch.total_len = max_len
    batch.kv_len += pad                  # each dummy contributes kv=1
    batch.num_decode += pad              # counted for lm_head / dense shape


def _runtime_limit(value):
    return float('inf') if value == 0 else value


_ASTRA_IDLE_POLL_NS = {
    "analytical": 1_000_000,
    "analytical-congestion-aware": 1_000_000,
    "ns3": 100,
}

_NETWORK_BACKEND_CHOICES = (
    "analytical",
    "analytical-congestion-aware",
    "ns3",
)

_ANALYTICAL_BINARIES = {
    "analytical": (
        "build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra"
    ),
    "analytical-congestion-aware": (
        "build/astra_analytical/build/AstraCongestion/bin/AstraCongestion"
    ),
}

_AGENTIC_PD_LAYOUT_FIELDS = (
    "model_name", "tp_size", "pp_size", "block_size", "dtype",
    "kv_cache_dtype", "max_model_len", "latency_model",
    "latency_model_band",
)


def _resolve_analytical_binary(astra_sim, network_backend):
    """Return the selected analytical ASTRA-Sim executable path."""
    try:
        relative_path = _ANALYTICAL_BINARIES[network_backend]
    except KeyError as exc:
        raise ValueError(
            f"Not an analytical ASTRA-Sim backend: {network_backend!r}"
        ) from exc
    return os.path.join(astra_sim, relative_path)


def _analytical_idle_endpoint_command(
        network_backend, is_group_controller, has_causal_backend_wakeup,
        has_partially_observed_completion=False):
    """Park an idle completion only when ASTRA can wake it exactly.

    Multi-rank analytical groups report their end endpoint before their start
    (group-controller) endpoint.  Only the latter may park: its command parks
    the whole still-pending group after Python has observed both callbacks.
    ``pass`` remains the retry primitive for DP wave assembly and for states
    with no backend event capable of waking a parked completion.  It is also
    required while another batch has reported only some of its endpoints:
    the remaining endpoint callback can release or admit Python-owned work,
    but it is not a new ASTRA completion generation and therefore cannot wake
    a peer group parked between the two callbacks.
    """
    if (network_backend in _ANALYTICAL_BINARIES
            and is_group_controller
            and has_causal_backend_wakeup
            and not has_partially_observed_completion):
        return "park"
    return "pass"


def _has_dispatched_model_work(schedulers, manager=None):
    """Return whether an ASTRA graph, rather than only a formed batch, lives."""
    if manager is None:
        return any(scheduler.inflight for scheduler in schedulers)
    return any(
        getattr(batch, "agentic_astra_dispatch_time_ns", None) is not None
        for scheduler in schedulers
        for batch in scheduler.inflight
    )


def _has_partially_observed_model_completion(schedulers):
    """Return whether an inflight batch awaits another endpoint callback.

    A multi-rank analytical instance reports its end endpoint before its
    start/controller endpoint.  ``Scheduler.add_done`` records the first
    report in ``batch.end`` but deliberately keeps the batch inflight until
    both reports arrive.  Parking an idle peer during that interval can lose
    work made ready by the second callback because the group completion has
    already opened and will not generate another backend wakeup.
    """
    return any(
        bool(batch.end)
        for scheduler in schedulers
        for batch in scheduler.inflight
    )


def _idle_fast_forward_delta(
        current_ns, next_arrival_ns, network_backend="analytical"):
    """Adjust logical time so ASTRA's next idle poll lands on an event.

    The signed offset adjustment leaves one backend poll in place. It is
    positive for a long gap and can be negative when an event is less than one
    poll away; the latter removes idle-poll quantization without making
    logical time go backward. Analytical ASTRA-Sim polls at 1 ms; the ns-3
    frontend uses its 100 ns ``idle_ticks`` quantum.
    """
    try:
        poll_ns = _ASTRA_IDLE_POLL_NS[network_backend]
    except KeyError as exc:
        raise ValueError(
            f"Unknown ASTRA-Sim network backend {network_backend!r}") from exc
    return int(next_arrival_ns) - int(current_ns) - poll_ns


def _throughput_interval_scale(log_interval_seconds):
    """Return ns interval and per-second scale for throughput buckets."""
    seconds = float(log_interval_seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("--log-interval must be finite and greater than zero")
    frequency_hz = 1_000_000_000
    return seconds * frequency_hz, 1.0 / seconds


def _analytical_report_time(reported_ns, logical_time_floor_ns):
    """Validate ASTRA's callback against Python's acknowledged time floor."""
    reported_ns = int(reported_ns)
    logical_time_floor_ns = int(logical_time_floor_ns)
    if reported_ns < 0 or logical_time_floor_ns < 0:
        raise ValueError("analytical simulation timestamps must be non-negative")
    if reported_ns < logical_time_floor_ns:
        raise RuntimeError(
            "analytical ASTRA timestamp regressed below the acknowledged "
            f"absolute-time floor: reported={reported_ns}, "
            f"floor={logical_time_floor_ns}")
    return reported_ns


def _analytical_advance_command(current_ns, target_ns):
    """Build a strictly forward absolute-time command for analytical ASTRA."""
    current_ns = int(current_ns)
    target_ns = int(target_ns)
    if current_ns < 0 or target_ns <= current_ns:
        raise ValueError(
            "analytical advance target must be strictly after current time: "
            f"current={current_ns}, target={target_ns}")
    return f"advance-to:{target_ns}"


def _uniform_cluster_link_value(raw_value, field_name):
    """Return one physical cold-fabric value or reject ambiguous dimensions."""
    values = raw_value if isinstance(raw_value, list) else [raw_value]
    if not values:
        raise ValueError(f"Cluster {field_name} must not be empty")
    parsed = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in parsed):
        raise ValueError(
            f"Cluster {field_name} contains an invalid value: {raw_value}")
    if any(not math.isclose(
            value, parsed[0], rel_tol=1e-9, abs_tol=1e-9)
            for value in parsed[1:]):
        raise ValueError(
            "Direct cold-HBM fabric currently requires one uniform physical "
            f"{field_name} across ASTRA topology dimensions; got {raw_value}")
    return parsed[0]


def _next_exact_control_wakeup_ns(
        current_ns, schedulers, router, manager=None,
        full_model_hbf_runtime=None):
    """Return the next Python-owned event that may occur during a live graph."""
    current_ns = int(current_ns)
    candidates = []
    pending = router.get_next_pending_arrival()
    if pending is not None and int(pending) > current_ns:
        candidates.append(int(pending))
    handoff_wakeup = getattr(
        router, "get_next_decode_handoff_wakeup", None)
    handoff = handoff_wakeup() if handoff_wakeup is not None else None
    if handoff is not None and int(handoff) > current_ns:
        candidates.append(int(handoff))
    for scheduler in schedulers:
        for value in (
                scheduler.memory_wait_until_ns,
                scheduler.model_fabric_wait_until_ns):
            if value is not None and int(value) > current_ns:
                candidates.append(int(value))
        if scheduler.request:
            ready_ns = int(scheduler.request[0].ready_time)
            if ready_ns > current_ns:
                candidates.append(ready_ns)
        if manager is not None:
            blocked = manager.synchronous_swap_blocked_until(
                scheduler.instance_id, current_ns)
            if blocked is not None and int(blocked) > current_ns:
                candidates.append(int(blocked))
    if manager is not None:
        internal = manager.next_internal_event_time(current_ns)
        if internal is not None and int(internal) > current_ns:
            candidates.append(int(internal))
    if full_model_hbf_runtime is not None:
        hbf_wakeup = full_model_hbf_runtime.adapter.next_wakeup_ns(
            current_ns,
            router_arrival_ns=router.get_next_pending_arrival(),
        )
        if hbf_wakeup is not None and int(hbf_wakeup) > current_ns:
            candidates.append(int(hbf_wakeup))
    return min(candidates) if candidates else None


def _measurement_drain_complete(
        stop_requested, schedulers, dp_pending, dp_ready_workloads,
        manager, exact_control_schedule,
        full_model_hbf_runtime=None, same_time_control_barrier=None):
    """Return whether every already-issued online dependency has drained.

    Measurement freeze censors work that has not launched, but ASTRA callbacks
    for model graphs, direct-HBM fabric jobs, and one-shot exact-time controls
    remain live protocol obligations.  The event loop may exit only after all
    three callback classes are quiescent, including a control callback that was
    armed before the freeze.
    """
    if not stop_requested:
        return False
    external_fabric_pending = _manager_has_pending_background_jobs(manager)
    full_model_hbf_pending = (
        full_model_hbf_runtime is not None
        and full_model_hbf_runtime.adapter.has_pending()
    )
    tie_barrier_pending = (
        same_time_control_barrier is not None
        and same_time_control_barrier.has_pending()
    )
    return (
        not any(scheduler.inflight for scheduler in schedulers)
        and not any(dp_pending.values())
        and not dp_ready_workloads
        and not external_fabric_pending
        and not exact_control_schedule.has_pending()
        and not full_model_hbf_pending
        and not tie_barrier_pending
    )


def _full_model_hbf_dispatch_blocked(runtime, same_time_barrier):
    """Fence new GPU graphs until tied completion ownership is committed."""

    return bool(
        runtime is not None
        and same_time_barrier is not None
        and same_time_barrier.has_pending()
    )


def _full_model_hbf_arrival_routing_blocked(
        runtime, same_time_barrier):
    """Fence Router mutation until all tied HBF effects reach the barrier."""

    if runtime is None:
        return False
    barrier_pending = (
        same_time_barrier is not None
        and same_time_barrier.has_pending()
    )
    return bool(
        barrier_pending
        or runtime.adapter.has_deferred_hbf_completions()
    )


def _manager_has_pending_background_jobs(manager):
    """Return whether an ASTRA-owned auxiliary job still requires callback."""
    if manager is None:
        return False
    return bool(getattr(
        manager, "has_pending_external_fabric_jobs", lambda: False)())


def _route_strictly_older_arrivals_at_callback(
        current_ns, router, manager=None):
    """Process pre-callback arrivals without issuing work in the past.

    Model completion/free owns the exact ``current_ns`` tie.  Requests that
    became ready strictly earlier must observe the still-live allocation
    state, so routing uses ``current_ns - 1`` as its causal cutoff.  ASTRA has
    nevertheless already reached ``current_ns`` when Python receives this
    callback; newly created physical operations therefore use ``current_ns``
    and expose the observation delay in their owner-gate accounting.
    """
    current_ns = int(current_ns)
    if current_ns <= 0:
        return 0
    if (manager is not None
            and int(manager.logical_frontier_ns) >= current_ns):
        return 0
    cutoff_ns = current_ns - 1
    routed = router.route_arrived_requests(
        cutoff_ns, operation_time_ns=current_ns)
    if (manager is not None
            and int(manager.logical_frontier_ns) < current_ns):
        manager.advance(cutoff_ns)
    return routed


def _next_idle_wakeup_ns(
        current_ns, schedulers, router,
        known_nonrunnable_instance_id=None):
    """Return the next useful event only when no instance can run now.

    A scheduler can contain a ready request while an idle-KV demotion owns its
    HBM admission dependency. Such a queue is not globally empty, but polling
    ASTRA-Sim until the copy completes would quantize the modeled transfer by
    the backend's idle tick. Future request arrivals and HBM-reclaim completion
    are therefore treated uniformly as exact logical wakeups.
    """
    if any(scheduler.inflight for scheduler in schedulers):
        return None

    candidates = []
    managers = {
        manager
        for scheduler in schedulers
        for manager in (getattr(scheduler, "agentic_kv_manager", None),)
        if manager is not None
    }
    for manager in managers:
        internal_ns = manager.next_internal_event_time(current_ns)
        if internal_ns is not None and int(internal_ns) > current_ns:
            candidates.append(int(internal_ns))
    for scheduler in schedulers:
        wait_ns = getattr(scheduler, "memory_wait_until_ns", None)
        fabric_wait_ns = getattr(
            scheduler, "model_fabric_wait_until_ns", None)
        manager = getattr(scheduler, "agentic_kv_manager", None)
        swap_wait_ns = (
            manager.synchronous_swap_blocked_until(
                scheduler.instance_id, current_ns)
            if manager is not None else None
        )
        head = scheduler.request[0] if scheduler.request else None
        if head is not None:
            ready_ns = int(head.ready_time)
            if ready_ns <= current_ns:
                memory_ready = wait_ns is None or int(wait_ns) <= current_ns
                swap_ready = (
                    swap_wait_ns is None or int(swap_wait_ns) <= current_ns
                )
                fabric_ready = (
                    fabric_wait_ns is None
                    or int(fabric_wait_ns) <= current_ns
                )
                # A start-NPU callback may have just attempted this exact
                # scheduler and returned no batch because a router-owned HBM
                # claim reserves the remaining slack.  Its superficially
                # ready head must not hide the future handoff/reclaim event
                # that can make it runnable.  Other ready schedulers still
                # prevent a global fast-forward because they were not tried
                # on this callback.
                attempted_and_blocked = (
                    known_nonrunnable_instance_id is not None
                    and int(scheduler.instance_id)
                    == int(known_nonrunnable_instance_id)
                )
                if (memory_ready and swap_ready and fabric_ready
                        and not attempted_and_blocked):
                    return None
                if not memory_ready:
                    candidates.append(int(wait_ns))
                if (fabric_wait_ns is not None
                        and int(fabric_wait_ns) > current_ns):
                    candidates.append(int(fabric_wait_ns))
                if not swap_ready:
                    candidates.append(int(swap_wait_ns))
            else:
                candidates.append(ready_ns)
                if wait_ns is not None and int(wait_ns) > current_ns:
                    candidates.append(int(wait_ns))
                if (swap_wait_ns is not None
                        and int(swap_wait_ns) > current_ns):
                    candidates.append(int(swap_wait_ns))
                if (fabric_wait_ns is not None
                        and int(fabric_wait_ns) > current_ns):
                    candidates.append(int(fabric_wait_ns))
        else:
            # These sources are independent. An instance can have no queued
            # scheduler request while a restore, capacity release, and shared
            # fabric reservation are all pending for router-owned work. Never
            # let the first field checked hide an earlier dependency event.
            if wait_ns is not None and int(wait_ns) > current_ns:
                candidates.append(int(wait_ns))
            if (fabric_wait_ns is not None
                    and int(fabric_wait_ns) > current_ns):
                candidates.append(int(fabric_wait_ns))
            if (swap_wait_ns is not None
                    and int(swap_wait_ns) > current_ns):
                candidates.append(int(swap_wait_ns))

    next_pending = router.get_next_pending_arrival()
    if next_pending is not None and int(next_pending) > current_ns:
        candidates.append(int(next_pending))
    handoff_wakeup = getattr(
        router, "get_next_decode_handoff_wakeup", None)
    next_handoff = handoff_wakeup() if handoff_wakeup is not None else None
    if next_handoff is not None and int(next_handoff) > current_ns:
        candidates.append(int(next_handoff))
    return min(candidates) if candidates else None


def _model_dispatch_blocked(manager, scheduler, current_ns):
    """Update the exact wakeup for a future ASTRA shared-fabric dispatch."""
    if manager is None:
        scheduler.model_fabric_wait_until_ns = None
        return False
    blocked_until = manager.model_dispatch_blocked_until(
        scheduler.instance_id, current_ns)
    scheduler.model_fabric_wait_until_ns = blocked_until
    return blocked_until is not None and int(current_ns) < blocked_until


def _record_astra_dispatch(manager, scheduler, batch, current_ns):
    if manager is not None:
        manager.record_astra_workload_dispatch(
            scheduler, batch, int(current_ns))


def _cluster_config_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join("..", path)


def _load_cluster_config_for_overrides(path):
    with open(_cluster_config_path(path), "r") as f:
        return json.load(f)


def _resolve_output_file(path, run_id):
    if path is None:
        return None
    return path.replace("{run_id}", run_id)


def _cleanup_inputs_root(run_paths, logger):
    """Remove generated ASTRA-Sim inputs after a completed simulation."""
    runs_root = os.path.abspath(os.path.join("inputs", "runs"))
    inputs_root = os.path.abspath(run_paths.inputs_root)
    if inputs_root in (os.path.abspath("inputs"), runs_root):
        raise RuntimeError(f"Refusing to remove broad inputs root: {inputs_root}")
    if not inputs_root.startswith(runs_root + os.sep):
        logger.warning(
            "Skipping ASTRA-Sim inputs cleanup because inputs_root is outside %s: %s",
            runs_root, inputs_root,
        )
        return
    shutil.rmtree(inputs_root, ignore_errors=True)
    logger.info("Removed ASTRA-Sim inputs root: %s", inputs_root)


def _prepare_ns3_config(astra_sim, run_paths):
    template = os.path.join(astra_sim, "extern/network_backend/ns-3/scratch/config/config.txt")
    output_dir = os.path.join(run_paths.inputs_root, "ns3", "output")
    config_path = os.path.join(run_paths.inputs_root, "ns3", "config.txt")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    replacements = {
        "FLOW_FILE": os.path.join(output_dir, "flow.txt"),
        "TRACE_FILE": os.path.join(output_dir, "trace.txt"),
        "TRACE_OUTPUT_FILE": os.path.join(output_dir, "mix.tr"),
        "FCT_OUTPUT_FILE": os.path.join(output_dir, "fct.txt"),
        "PFC_OUTPUT_FILE": os.path.join(output_dir, "pfc.txt"),
        "QLEN_MON_FILE": os.path.join(output_dir, "qlen.txt"),
    }

    for path in (replacements["FLOW_FILE"], replacements["TRACE_FILE"]):
        open(path, "w").close()

    with open(template, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(config_path, "w", encoding="utf-8") as f:
        for line in lines:
            parts = line.split(maxsplit=1)
            if parts and parts[0] in replacements:
                f.write(f"{parts[0]} {replacements[parts[0]]}\n")
            else:
                f.write(line)
    return config_path


def _iter_raw_instances(cluster_config):
    for node in cluster_config.get("nodes", []):
        for instance in node.get("instances", []):
            yield instance


def _resolve_instance_dtype(instance, cli_dtype, dtype_to_bits):
    dtype = instance.get("dtype", cli_dtype)
    if dtype is None:
        config = get_config(instance["model_name"])
        torch_dtype = config.get("torch_dtype")
        if isinstance(torch_dtype, str) and torch_dtype in dtype_to_bits:
            dtype = torch_dtype
        else:
            dtype = "bfloat16"
    if dtype not in dtype_to_bits:
        raise ValueError(f"Unsupported dtype '{dtype}' for instance {instance.get('instance_id')}")
    return dtype


def _resolve_instance_max_model_len(instance, cli_max_model_len):
    """Resolve one instance's semantic context limit.

    The runtime override mirrors vLLM's ``max_model_len`` surface. It does not
    change a model's attention implementation or positional encoding by
    itself; callers must select a profile calibrated for the same long-context
    execution path.
    """
    max_model_len = instance.get("max_model_len", cli_max_model_len)
    if max_model_len is None:
        config = get_config(instance["model_name"])
        max_model_len = config["max_position_embeddings"]
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int):
        raise TypeError(
            "max_model_len must be a positive integer for instance "
            f"{instance.get('instance_id')}, got {max_model_len!r}"
        )
    if max_model_len <= 0:
        raise ValueError(
            "max_model_len must be positive for instance "
            f"{instance.get('instance_id')}, got {max_model_len}"
        )
    return max_model_len


def _agentic_pd_layout(instance, runtime_config):
    """Return fields that must match across an agentic P/D handoff."""
    optional_defaults = {
        "latency_model": None,
        "latency_model_band": "central",
    }
    values = []
    for field in _AGENTIC_PD_LAYOUT_FIELDS:
        if field in runtime_config:
            values.append(runtime_config[field])
        elif field in instance:
            values.append(instance[field])
        elif field in optional_defaults:
            values.append(optional_defaults[field])
        else:
            raise KeyError(field)
    return tuple(values)


def _build_instance_runtime_configs(instances, args, dtype_to_bits):
    # Keep direct helper callers and older validation fixtures compatible with
    # the optional analytical provider.  Parsed CLI namespaces always contain
    # these fields, but they are not fundamental scheduler inputs.
    cli_latency_model = getattr(args, "latency_model", None)
    cli_latency_model_band = getattr(
        args, "latency_model_band", "central")
    runtime_configs = []
    for instance_id, instance in enumerate(instances):
        dtype = _resolve_instance_dtype(instance, args.dtype, dtype_to_bits)
        kv_cache_dtype = instance.get("kv_cache_dtype", args.kv_cache_dtype)
        if kv_cache_dtype not in ("auto", "fp8"):
            raise ValueError(f"Unsupported kv_cache_dtype '{kv_cache_dtype}' for instance {instance_id}")

        enable_attn_offloading = instance.get("enable_attn_offloading", args.enable_attn_offloading)
        enable_sub_batch_interleaving = instance.get(
            "enable_sub_batch_interleaving", args.enable_sub_batch_interleaving)
        if enable_sub_batch_interleaving and not enable_attn_offloading:
            raise RuntimeError(
                f"Instance {instance_id} enables sub-batch interleaving without attention offloading")

        runtime_configs.append({
            "max_model_len": _resolve_instance_max_model_len(
                instance, args.max_model_len),
            "max_num_seqs": _runtime_limit(instance.get("max_num_seqs", args.max_num_seqs)),
            "max_num_batched_tokens": _runtime_limit(
                instance.get("max_num_batched_tokens", args.max_num_batched_tokens)),
            "long_prefill_token_threshold": instance.get(
                "long_prefill_token_threshold", args.long_prefill_token_threshold),
            "block_size": instance.get("block_size", args.block_size),
            "dtype": dtype,
            "fp": dtype_to_bits[dtype],
            "kv_cache_dtype": kv_cache_dtype,
            "enable_chunked_prefill": instance.get(
                "enable_chunked_prefill", args.enable_chunked_prefill),
            "enable_prefix_caching": instance.get(
                "enable_prefix_caching", args.enable_prefix_caching),
            "prioritize_prefill": instance.get("prioritize_prefill", args.prioritize_prefill),
            "enable_local_offloading": instance.get(
                "enable_local_offloading", args.enable_local_offloading),
            "enable_attn_offloading": enable_attn_offloading,
            "enable_sub_batch_interleaving": enable_sub_batch_interleaving,
            "enable_block_copy": instance.get("enable_block_copy", args.enable_block_copy),
            "latency_model": instance.get(
                "latency_model", cli_latency_model),
            "latency_model_band": instance.get(
                "latency_model_band", cli_latency_model_band),
        })
    return runtime_configs


def main():
    # ----------------------------------------------------------------------------------------------
    # LLMServingSim runs in astra-sim directory for easy path configuration
    # your relative path should start from astra-sim directory
    cwd = os.getcwd()
    astra_sim = os.path.join(cwd, "astra-sim")
    os.chdir(astra_sim)

    # -------------------------------------- Argument parsing --------------------------------------
    parser = argparse.ArgumentParser(prog='python -m serving',
                                     description='LLMServingSim') 
    
    parser.add_argument('--cluster-config', type=str, default='configs/cluster/single_node_pd_instance.json',
                        help='path to cluster config JSON defining node topology, instance layout, hardware, and memory hierarchy')
    parser.add_argument('--max-num-seqs', type=int, default=128,
                        help='maximum number of sequences in a batch (0 = unlimited)')
    parser.add_argument('--max-num-batched-tokens', type=int, default=2048,
                        help='maximum number of tokens processed per iteration across all requests (the total token budget). '
                        'With chunked prefill, long inputs are split across iterations; '
                        'without chunked prefill, this effectively caps max input length')
    parser.add_argument('--max-model-len', type=int, default=None,
                        help='semantic prompt-plus-output context limit. When omitted, each instance uses '
                        'its model config max_position_embeddings; a cluster instance may override this '
                        'with max_model_len')
    parser.add_argument('--long-prefill-token-threshold', type=int, default=0,
                        help='per-request token cap per step for chunked prefill (0 = disabled). '
                        'Limits how many tokens a single prefill request consumes per iteration, '
                        'preventing long prompts from monopolizing the token budget. '
                        'When 0, a single prefill can consume the entire budget')
    parser.add_argument('--dtype', type=str, choices=['float16', 'bfloat16', 'float32', 'fp8', 'int8'], default=None,
                        help='model weight data type (vLLM-style). When omitted, defaults to the model config\'s '
                        '``torch_dtype`` (falling back to bfloat16). Overrides only take effect if the profiler '
                        'produced matching data under perf/<hw>/<model>/<variant>/tp<N>/')
    parser.add_argument('--request-routing-policy', type=str, choices=['LOAD', 'RR', 'RAND', 'CUSTOM'], default='LOAD',
                        help='request routing policy across instances: LOAD (vLLM-style weighted least-loaded, default), '
                        'RR (round-robin), RAND (random), CUSTOM (user-defined)')
    parser.add_argument('--expert-routing-policy', type=str,
                        choices=['BALANCED', 'RR', 'RAND', 'CUSTOM'],
                        default='BALANCED',
                        help='expert token routing policy for MoE models: '
                        'BALANCED (default; analytical pigeonhole approximation of '
                        'a trained load-balanced learned gate), '
                        'RR (round-robin), RAND (uniform random per token), '
                        'CUSTOM (user-defined)')
    parser.add_argument('--enable-block-copy', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Replay one transformer block\'s trace across every '
                        'layer instead of re-computing the routing per layer — '
                        'cuts trace-generation time roughly num_hidden_layers× '
                        'on MoE models. Safe with BALANCED (deterministic); '
                        'RR/RAND get a small per-layer variance averaged out. '
                        'Disable only for CUSTOM policies that need faithful '
                        'per-layer variance.')
    parser.add_argument('--enable-prefix-caching', action=argparse.BooleanOptionalAction, default=True,
                        help='enable prefix caching via RadixAttention to reuse KV cache across requests '
                        'with shared prefixes (default: enabled). Use --no-enable-prefix-caching to disable')
    parser.add_argument('--enable-chunked-prefill', action=argparse.BooleanOptionalAction, default=True,
                        help='enable chunked prefill to split long prefill requests across multiple iterations, '
                        'matching vLLM v1 behavior (default: enabled). Use --no-enable-chunked-prefill to disable')
    parser.add_argument('--enable-prefix-sharing', action='store_true', default=False,
                        help='enable second-tier prefix cache pooling across instances within a node')
    parser.add_argument('--prefix-storage', type=str, choices=['None', 'CPU', 'CXL'], default='None',
                        help='storage medium for the second-tier prefix cache pool: None (NPU only), CPU, or CXL')
    parser.add_argument('--enable-local-offloading', action='store_true', default=False,
                        help='enable weight offloading to local (NPU) memory. '
                        'Recommended to disable unless weight memory access is not counted in profiling')
    parser.add_argument('--enable-attn-offloading', action='store_true', default=False,
                        help='enable attention computation offloading to PIM (Processing-In-Memory) devices')
    parser.add_argument('--enable-sub-batch-interleaving', action='store_true', default=False,
                        help='enable sub-batch interleaving to overlap XPU and PIM computation. '
                        'Requires --enable-attn-offloading')
    parser.add_argument('--prioritize-prefill', action='store_true', default=False,
                        help='prioritize prefill requests over decode requests in scheduling')
    parser.add_argument('--block-size', type=int, default=16,
                        help='KV cache block size in tokens (number of tokens per block)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='path to .jsonl dataset file with request traces. '
                        'If None, requests must be added manually in serving/__main__.py')
    parser.add_argument('--output', type=str, default=None,
                        help='path for per-request CSV output with latency metrics (TTFT, TPOT, ITL). '
                        'If None, results are printed to stdout only. Supports {run_id} placeholder')
    parser.add_argument('--run-id', type=str, default=None,
                        help='unique id for this simulation run. Intermediate ASTRA-Sim inputs are written under '
                        'astra-sim/inputs/runs/<run-id>. If omitted, a process-unique id is generated')
    parser.add_argument('--inputs-root', type=str, default=None,
                        help='override the root directory for generated ASTRA-Sim inputs. Defaults to '
                        'astra-sim/inputs/runs/<run-id>')
    parser.add_argument('--cleanup-inputs', action=argparse.BooleanOptionalAction, default=True,
                        help='remove generated ASTRA-Sim inputs under astra-sim/inputs/runs/<run-id> '
                        'after a successful simulation (default: enabled). Use --no-cleanup-inputs '
                        'to preserve generated trace files, Chakra workloads, and input configs for debugging')
    parser.add_argument('--skip-prefill', action='store_true', default=False,
                        help='skip the prefill phase, running decode only')
    parser.add_argument('--num-reqs', type=int, default=0,
                        help='number of entries (requests or sessions) to load from the dataset. '
                        'For agentic datasets, each entry is a session with multiple sub-requests. '
                        '0 = load all entries')
    parser.add_argument('--log-interval', type=float, default=1.0,
                        help='interval in seconds between throughput/memory usage log messages')
    parser.add_argument('--log-level', type=str, choices=['WARNING', 'INFO', 'DEBUG'], default='WARNING',
                        help='logging verbosity: WARNING (minimal), INFO (per-iteration details), DEBUG (per-layer memory)')
    parser.add_argument('--kv-cache-dtype', type=str, choices=['auto', 'fp8'], default='auto',
                        help='KV cache data type: auto (use default profile.csv) or fp8 (use profile_fp8.csv, halves KV cache memory)')
    parser.add_argument(
        '--latency-model',
        choices=SUPPORTED_ONLINE_LATENCY_MODELS,
        default=None,
        help=(
            'optional online analytical COMP-node provider; collectives '
            'remain in ASTRA-Sim (default: use profiler/perf CSVs)'
        ),
    )
    parser.add_argument(
        '--latency-model-band',
        choices=['fast', 'central', 'slow'],
        default='central',
        help='uncertainty band for the online analytical latency provider',
    )
    parser.add_argument('--agentic-kv-policy', type=str,
                        choices=['off', 'preserve', 'recompute', 'hbm_lru_recompute',
                                 'hbm_ssd_direct', 'cpu', 'tiered',
                                 'tiered_queue_recompute'],
                        default=None,
                        help='idle session-KV policy for agentic workloads: off, preserve in HBM, '
                        'discard/recompute, HBM LRU with recomputation, direct HBM<->SSD, '
                        'immediate CPU swap, HBM->CPU->SSD tiering, or queue-aware tiering '
                        'that may discard a lower-tier prefix for recomputation. '
                        'When omitted, use the policy in --agentic-kv-config (or disable tiering)')
    parser.add_argument('--agentic-kv-config', type=str, default=None,
                        help='optional JSON config for agentic KV TTLs, tier bandwidths/capacity, '
                        'and SSD full-vs-incremental write accounting')
    parser.add_argument('--agentic-kv-metrics', type=str, default=None,
                        help='write raw agentic KV migration and SSD traffic metrics to JSON')
    parser.add_argument(
        '--full-model-hbf-config', type=str, default=None,
        help=(
            'optional eight-card full-model HBF server JSON. This enables '
            'the live GPU P4/D4 plus HBF runtime and is mutually exclusive '
            'with legacy agentic-KV tiering'
        ),
    )
    parser.add_argument(
        '--full-model-hbf-layout',
        choices=['dp8', 'tp4', 'tp8', 'tp8_context'],
        default='tp8_context',
        help='parallel layout for the full-model HBF server',
    )
    parser.add_argument(
        '--full-model-hbf-max-num-batched-tokens',
        type=int, default=8192,
        help='HBF-server continuous-batch token budget',
    )
    parser.add_argument(
        '--full-model-hbf-max-num-seqs',
        type=int, default=128,
        help='HBF-server maximum live sequences per replica',
    )
    parser.add_argument(
        '--full-model-hbf-max-prefill-chunk-tokens',
        type=int, default=4096,
        help='maximum fresh prefill tokens per HBF request and batch',
    )
    parser.add_argument(
        '--full-model-hbf-prefill-drain-tail-tokens',
        type=int, default=2048,
        help=(
            'recent materialized KV tokens retained in HBF-card LPDDR '
            'after the first output token'
        ),
    )
    parser.add_argument(
        '--full-model-hbf-prefill-drain-min-tokens',
        type=int, default=4096,
        help=(
            'minimum contiguous LPDDR KV tokens required before issuing '
            'a first-token HBF drain'
        ),
    )
    parser.add_argument(
        '--full-model-hbf-astra-chunk-bytes',
        type=int, default=64 * 1024 ** 2,
        help='maximum lifecycle-transfer chunk projected onto ASTRA',
    )
    parser.add_argument(
        '--full-model-hbf-metrics', type=str, default=None,
        help=(
            'write the full HBF lifecycle, pool, ASTRA, and finite-HBM '
            'ownership report to JSON; supports {run_id}'
        ),
    )
    parser.add_argument(
        '--strict-infinite-hbm-oracle',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'run the online preserve policy with a workload-derived, '
            'provably nonbinding per-rank HBM bound and fail the run if any '
            'capacity demotion, lower-tier hit, or avoidable recomputation '
            'is observed'
        ),
    )
    add_session_admission_arguments(parser)
    parser.add_argument(
        '--session-metrics', type=str, default=None,
        help=(
            'write session admission, throughput, resume TTFT, TTFT, TPOT, '
            'and queue/restore summaries to JSON; supports {run_id}'
        ),
    )
    parser.add_argument(
        '--network-backend', type=str, choices=_NETWORK_BACKEND_CHOICES,
        default='analytical',
        help=(
            'network simulation backend: analytical (fast, default), '
            'analytical-congestion-aware (link contention), or ns3 '
            '(detailed, WIP)'
        ),
    )

    args = parser.parse_args()
    session_admission = session_admission_from_args(args)
    
    args.run_id = resolve_run_id(args.run_id)
    run_paths = build_run_paths(astra_sim, args.run_id, args.inputs_root)
    args.inputs_root = run_paths.inputs_root
    args.output = _resolve_output_file(args.output, args.run_id)
    args.session_metrics = _resolve_output_file(
        args.session_metrics, args.run_id)
    args.full_model_hbf_metrics = _resolve_output_file(
        args.full_model_hbf_metrics, args.run_id)

    configure_logger(level=args.log_level)
    logger = get_logger("Main")
    print_banner()
    
    _dtype_to_bits = {'float16': 16, 'bfloat16': 16, 'float32': 32, 'fp8': 8, 'int8': 8}
    request_routing_policy=args.request_routing_policy
    expert_routing_policy=args.expert_routing_policy
    enable_prefix_sharing=args.enable_prefix_sharing
    prefix_storage=args.prefix_storage
    dataset=args.dataset
    output_file=args.output
    is_init = not args.skip_prefill
    num_req=args.num_reqs
    log_interval=args.log_interval
    network_backend = args.network_backend
    raw_cluster_config = _load_cluster_config_for_overrides(args.cluster_config)
    raw_instances = list(_iter_raw_instances(raw_cluster_config))
    build_enable_local_offloading = args.enable_local_offloading or any(
        inst.get("enable_local_offloading", False) for inst in raw_instances)
    build_enable_attn_offloading = args.enable_attn_offloading or any(
        inst.get("enable_attn_offloading", False) for inst in raw_instances)
    full_model_hbf_options = None
    full_model_hbf_hardware = None
    if args.full_model_hbf_config is not None:
        full_model_hbf_options = FullModelHBFRuntimeOptions(
            layout_key=args.full_model_hbf_layout,
            max_num_batched_tokens=(
                args.full_model_hbf_max_num_batched_tokens),
            max_num_seqs=args.full_model_hbf_max_num_seqs,
            max_prefill_chunk_tokens=(
                args.full_model_hbf_max_prefill_chunk_tokens),
            prefill_drain_tail_tokens=(
                args.full_model_hbf_prefill_drain_tail_tokens),
            prefill_drain_min_tokens=(
                args.full_model_hbf_prefill_drain_min_tokens),
            astra_chunk_bytes=args.full_model_hbf_astra_chunk_bytes,
            latency_band=args.latency_model_band,
        )
        full_model_hbf_options.validate()
        full_model_hbf_hardware, _ = load_full_model_hbf_hardware(
            Path(_cluster_config_path(args.full_model_hbf_config)),
            full_model_hbf_options.layout_key,
        )
    full_model_hbf_enabled = full_model_hbf_options is not None
    shared_hbf_num_cards = (
        full_model_hbf_hardware.card_count
        if full_model_hbf_enabled else None
    )
    # ---------------------------------- Extract cluster config -----------------------------------
    cluster = build_cluster_config(
        astra_sim, args.cluster_config, build_enable_local_offloading, build_enable_attn_offloading,
        inputs_root=run_paths.inputs_root,
        hbf_num_cards=shared_hbf_num_cards,
    )
    num_nodes = cluster["num_nodes"]
    num_instances = cluster["num_instances"]
    instances = cluster["instances"]
    inst2node_mapping = cluster["inst2node_mapping"]
    inst2npu_mapping = cluster["inst2npu_mapping"]
    npu2inst_mapping = cluster["npu2inst_mapping"]
    prefill_instance = cluster["prefill_instance"]
    decode_instance = cluster["decode_instance"]
    start_npu_ids = cluster["start_npu_ids"]
    end_npu_ids = cluster["end_npu_ids"]
    placement = cluster["placement"]
    block_mode_on = cluster["block_mode_on"]
    total_npu = cluster["total_npu"]
    cpu_mem_size = cluster["cpu_mem_size"]
    power_modeling = cluster["power_modeling"]
    power_configs = cluster["power_configs"]
    pim_models = cluster["pim_models"]
    instance_runtime_configs = _build_instance_runtime_configs(instances, args, _dtype_to_bits)
    any_prefix_caching = any(cfg["enable_prefix_caching"] for cfg in instance_runtime_configs)

    if full_model_hbf_enabled:
        if dataset is None:
            raise ValueError(
                "--full-model-hbf-config requires an agentic --dataset")
        if args.agentic_kv_config is not None or (
                args.agentic_kv_policy not in (None, "off")):
            raise ValueError(
                "full-model HBF owns session-KV placement; do not combine "
                "it with --agentic-kv-config or --agentic-kv-policy")
        if args.strict_infinite_hbm_oracle:
            raise ValueError(
                "full-model HBF and the infinite-HBM oracle are separate "
                "comparison systems")
        if args.skip_prefill:
            raise ValueError(
                "full-model HBF requires ordinary first-turn prefill")
        if enable_prefix_sharing or prefix_storage != "None":
            raise ValueError(
                "full-model HBF cannot share KV ownership with a generic "
                "second-tier prefix pool")
        validate_full_model_hbf_gpu_cluster(
            instances,
            instance_runtime_configs,
            inst2node_mapping,
            network_backend=network_backend,
        )

    print_input_config(
        args=args,
        instances=instances,
        instance_runtime_configs=instance_runtime_configs,
    )
    print_markup("[sim.heading]▶ Starting simulation...[/]\n")
    flush.stdout.flush()

    agentic_kv_config = None
    if args.agentic_kv_config is not None:
        agentic_config_path = _cluster_config_path(args.agentic_kv_config)
        agentic_kv_config = AgenticKVConfig.from_json(
            agentic_config_path, policy=args.agentic_kv_policy)
    elif args.agentic_kv_policy not in (None, 'off'):
        agentic_kv_config = AgenticKVConfig(policy=args.agentic_kv_policy)
        agentic_kv_config.validate()

    if args.strict_infinite_hbm_oracle:
        if dataset is None:
            raise ValueError(
                "--strict-infinite-hbm-oracle requires --dataset")
        agentic_kv_config = configure_strict_oracle(agentic_kv_config)

    external_cold_fabric = False
    external_cold_fabric_bandwidth_gbps = None
    external_cold_fabric_latency_ns = None
    if agentic_kv_config is not None and agentic_kv_config.enabled:
        if any_prefix_caching:
            raise ValueError(
                "Agentic session-KV tiering and generic Radix prefix caching cannot be enabled "
                "together because that would double-count physical KV blocks. Re-run with "
                "--no-enable-prefix-caching; use a separate prefix-cache run as the baseline.")
        pd_roles = {
            instance.get("pd_type") for instance in instances
            if instance.get("pd_type") is not None
        }
        if pd_roles:
            if pd_roles != {"prefill", "decode"}:
                raise ValueError(
                    "Agentic P/D tiering requires both prefill and decode instances")
            if any(instance.get("pd_type") is None for instance in instances):
                raise ValueError(
                    "Agentic P/D tiering cannot mix colocated and disaggregated "
                    "instances in one routing pool")
            unresolved_prefills = cluster["pd_endpoint_aliases"].get(
                "unresolved_prefill_instances", []
            )
            if unresolved_prefills:
                raise ValueError(
                    "Strict agentic P/D receive admission requires one "
                    "unambiguous decode endpoint for every prefill graph; "
                    "unresolved prefill instance(s): "
                    f"{unresolved_prefills}")

            # A resumed prefix is moved decode->prefill before incremental
            # prefill, then the existing P->D handoff returns the cache to its
            # sticky decode owner. The configured decode->prefill path is
            # either CPU-staged or a direct accelerator fabric; both models
            # require a layout-compatible decode peer on the same node.
            for prefill in (
                    instance for instance in instances
                    if instance.get("pd_type") == "prefill"):
                prefill_id = prefill["instance_id"]
                prefill_layout = _agentic_pd_layout(
                    prefill, instance_runtime_configs[prefill_id]
                )
                compatible = False
                for decode in (
                        instance for instance in instances
                        if instance.get("pd_type") == "decode"
                        and inst2node_mapping[instance["instance_id"]]
                        == inst2node_mapping[prefill_id]):
                    decode_id = decode["instance_id"]
                    decode_layout = _agentic_pd_layout(
                        decode, instance_runtime_configs[decode_id]
                    )
                    if decode_layout == prefill_layout:
                        compatible = True
                        break
                if not compatible:
                    raise ValueError(
                        "Agentic P/D requires a same-node decode peer with "
                        "matching model, TP/PP, block size, dtype, KV dtype, "
                        "and max model length "
                        f"for prefill instance {prefill_id}")
        elif agentic_kv_config.pd_peer_transfer_mode == "direct-fabric":
            raise ValueError(
                "Agentic pd_peer_transfer_mode='direct-fabric' requires "
                "disaggregated prefill and decode instances")
        block_sizes = {cfg["block_size"] for cfg in instance_runtime_configs}
        if len(block_sizes) != 1:
            raise ValueError("Agentic KV tiering requires one KV block size across instances")
        runtime_block_size = next(iter(block_sizes))
        if args.agentic_kv_config is not None and agentic_kv_config.block_size != runtime_block_size:
            raise ValueError(
                f"agentic KV config block_size={agentic_kv_config.block_size} does not match "
                f"runtime block_size={runtime_block_size}")
        agentic_kv_config.block_size = runtime_block_size
        external_cold_fabric = (
            agentic_kv_config.pd_peer_transfer_mode == "direct-fabric")
        if external_cold_fabric:
            if network_backend != "analytical-congestion-aware":
                raise ValueError(
                    "Agentic pd_peer_transfer_mode='direct-fabric' requires "
                    "--network-backend analytical-congestion-aware so cold "
                    "D->P traffic contends with online ASTRA communication")
            external_cold_fabric_bandwidth_gbps = (
                _uniform_cluster_link_value(cluster["link_bw"], "link_bw"))
            external_cold_fabric_latency_ns = int(round(
                _uniform_cluster_link_value(
                    cluster["link_latency"], "link_latency")))
    # ----------------------------------------- Set config -----------------------------------------
    # Automatic network, memory configuration
    # If you want to set more specific information such as latency, look at config.py and each json file
    if network_backend in _ANALYTICAL_BINARIES:
        network=run_paths.network_config
        binary=_resolve_analytical_binary(astra_sim, network_backend)
    elif network_backend == 'ns3':
        network=_prepare_ns3_config(astra_sim, run_paths)
        binary=os.path.join(astra_sim, "extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default")
    else:
        raise NotImplementedError(
            f"Unsupported network backend: {network_backend}")
    memory=run_paths.memory_config
    system=run_paths.system_config
    # ------------------------------------- Prepare simulation -------------------------------------
    # Need to extract each instance's memory accessability 
    node2inst_mapping = defaultdict(list)
    for inst_id, node_id in inst2node_mapping.items():
        node2inst_mapping[node_id].append(inst_id)
    node2inst_mapping = dict(node2inst_mapping)

    prefix_pool_inst_mapping = {}
    for i in range(num_instances):
        prefix_pool_inst_mapping[i] = None

    pool_device = None

    if prefix_storage == "CPU":
        pool_device = Device.CPU
    elif prefix_storage == "CXL":
        pool_device = Device.CXL

    if any_prefix_caching and enable_prefix_sharing and prefix_storage != 'None':
        num_prefix_pool = num_nodes
        # make prefix pool objects based on num_prefix_pool
        prefix_pools = []

        def _pool_kv_bytes_per_token(inst_ids):
            """KV bytes per token for a shared pool."""
            kv_shapes = {
                (
                    instances[i]["model_name"],
                    instance_runtime_configs[i]["fp"],
                    instance_runtime_configs[i]["kv_cache_dtype"],
                )
                for i in inst_ids
            }
            if len(kv_shapes) > 1:
                raise RuntimeError(
                    "Shared prefix pool requires instances to share model, "
                    f"dtype, and kv_cache_dtype; got {kv_shapes}"
                )
            model = instances[inst_ids[0]]['model_name']
            cfg = instance_runtime_configs[inst_ids[0]]
            return full_cluster_kv_bytes_per_token(
                model, cfg["fp"], cfg["kv_cache_dtype"],
                tp_size=instances[inst_ids[0]]["tp_size"])

        if prefix_storage == 'CPU':
            for i in range(num_prefix_pool):
                if cpu_mem_size[i] > 0:
                    new_prefix_pool = RadixCache(
                                                node_id=0,
                                                device=prefix_storage,
                                                page_size=256,
                                                capacity = cpu_mem_size[i] * GB_TO_BYTE,
                                                kv_size=_pool_kv_bytes_per_token(node2inst_mapping[i]),
                                                enable_kv_cache_events=True)
                    prefix_pools.append(new_prefix_pool)
                else:
                    raise RuntimeError(f"Memory size for prefix storage type {prefix_storage} is invalid")
            # This means one node shares one prefix pool
            prefix_pool_inst_mapping = inst2node_mapping

        elif prefix_storage == 'CXL':
            if cluster["cxl_mem_size"] > 0:
                new_prefix_pool = RadixCache(
                                            node_id=None,
                                            device=prefix_storage,
                                            page_size=1,
                                            capacity = cluster["cxl_mem_size"] * GB_TO_BYTE,
                                            kv_size=_pool_kv_bytes_per_token(list(range(num_instances))),
                                            enable_kv_cache_events=True)
                prefix_pools.append(new_prefix_pool)
                # This means every instance shares the same universal prefix pool (maybe fixed later)
                prefix_pool_inst_mapping = [0 for _ in range(num_instances)]
            else:
                raise RuntimeError(f"Memory size for prefix storage type {prefix_storage} is invalid")
        else:
            raise NotImplementedError(f"Prefix storage type {prefix_storage} is not supported or memory size is invalid")

    schedulers = []
    for instance_id, instance in enumerate(instances):
        prefix_pool_index = prefix_pool_inst_mapping[instance_id]
        prefix_pool = None
        if prefix_pool_index != None:
            prefix_pool = prefix_pools[prefix_pool_index]
        cxl_mem = 0
        if cluster["cxl_mem_size"] > 0:
            cxl_mem = cluster["cxl_mem_size"]        
        
        # Make scheduler for each instance

        inst_cfg = instance_runtime_configs[instance_id]

        schedulers.append(Scheduler(
            instance["model_name"], instance["node_id"], instance_id,
            inst_cfg["max_num_seqs"], inst_cfg["max_num_batched_tokens"],
            instance["num_npus"], instance["tp_size"], instance["pp_size"],
            instance["npu_mem"]["mem_size"], cpu_mem_size[instance["node_id"]],
            inst2npu_mapping[instance_id], instance["pd_type"],
            inst_cfg["fp"], inst_cfg["block_size"], num_req,
            inst_cfg["prioritize_prefill"], inst_cfg["enable_prefix_caching"],
            enable_prefix_sharing, prefix_pool, pool_device, inst_cfg["enable_chunked_prefill"],
            inst_cfg["long_prefill_token_threshold"],
            cxl_mem,
            ep_size=instance.get("ep_total", 1),
            kv_cache_dtype=inst_cfg["kv_cache_dtype"],
            active_preemption_mode=(
                agentic_kv_config.active_preemption_mode
                if (agentic_kv_config is not None
                    and agentic_kv_config.enabled)
                else 'cpu-swap'),
            max_model_len=inst_cfg["max_model_len"],
            npu_runtime_reserve_bytes=instance["npu_mem"].get(
                "runtime_reserve_bytes", 0),
        ))

    strict_oracle = None
    if args.strict_infinite_hbm_oracle:
        strict_oracle = StrictInfiniteHBMOracle.from_workload(
            schedulers,
            dataset,
            num_reqs=num_req,
            backlog_epochs=(
                session_admission.backlog_epochs
                if session_admission.mode == 'backlog' else 1
            ),
        )

    agentic_kv_manager = None
    if agentic_kv_config is not None and agentic_kv_config.enabled:
        queue_recompute_latency_providers = {}
        if (agentic_kv_config.queue_recompute_enabled
                and agentic_kv_config
                .queue_recompute_cost_guard_multiplier > 0):
            repo_root = Path(__file__).resolve().parents[1]
            for scheduler in schedulers:
                instance_id = int(scheduler.instance_id)
                instance = instances[instance_id]
                if instance.get("pd_type") == "decode":
                    continue
                inst_cfg = instance_runtime_configs[instance_id]
                provider = resolve_online_latency_model(
                    name=inst_cfg["latency_model"],
                    repo_root=repo_root,
                    hardware=instance["hardware"],
                    model=instance["model_name"],
                    config=get_config(instance["model_name"]),
                    tp_size=instance["tp_size"],
                    pp_size=instance["pp_size"],
                    local_ep=instance["local_ep"],
                    ep_total=instance["ep_total"],
                    fp_bytes=inst_cfg["fp"] // 8,
                    dtype=inst_cfg["dtype"],
                    kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                    enable_attn_offloading=(
                        inst_cfg["enable_attn_offloading"]),
                    band=inst_cfg["latency_model_band"],
                )
                if provider is None:
                    raise ValueError(
                        "Cost-aware queue recomputation requires the same "
                        "online latency model used by the trace path; "
                        f"instance {instance_id} uses "
                        f"{inst_cfg['latency_model']!r}")
                queue_recompute_latency_providers[instance_id] = provider
        agentic_kv_manager = AgenticKVManager(
            schedulers, agentic_kv_config,
            queue_recompute_latency_providers=(
                queue_recompute_latency_providers),
        )
        if external_cold_fabric:
            agentic_kv_manager.enable_external_fabric(
                backend=network_backend,
                physical_bandwidth_gbps=(
                    external_cold_fabric_bandwidth_gbps),
                physical_latency_ns=external_cold_fabric_latency_ns,
                physical_bandwidth_unit=cluster[
                    "link_bw_unit_effective"],
            )
    full_model_hbf_runtime = None
    if full_model_hbf_enabled:
        full_model_hbf_runtime = build_full_model_hbf_online_runtime(
            repo_root=Path(__file__).resolve().parents[1],
            config_path=Path(
                _cluster_config_path(args.full_model_hbf_config)),
            options=full_model_hbf_options,
            instances=instances,
            runtime_configs=instance_runtime_configs,
            inst2node_mapping=inst2node_mapping,
            schedulers=schedulers,
            network_backend=network_backend,
        )
    model_compute_accounting = OnlineModelComputeAccounting()
    hbm_occupancy_accounting = (
        OnlineHBMOccupancyAccounting(schedulers)
        if agentic_kv_manager is not None else None
    )

    # Controller for astra-sim process communication
    controller = Controller(total_npu)
    # Global Request Router
    router = Router(
        num_instances, schedulers, num_req, request_routing_policy,
        agentic_kv_manager=agentic_kv_manager,
        full_model_hbf_adapter=(
            full_model_hbf_runtime.adapter
            if full_model_hbf_runtime is not None else None),
        full_model_hbf_gpu_hbm_bridge=(
            full_model_hbf_runtime.gpu_hbm_bridge
            if full_model_hbf_runtime is not None else None),
        session_admission=session_admission)
    # Power Modeling if enabled
    if power_modeling:
        power_model = PowerModel(power_configs)
    else:
        power_model = None
    # Load requests into router (routed in real-time during simulation)
    if dataset != None:
        router.load_requests(dataset, enable_prefix_caching=any_prefix_caching, is_init=is_init)
    else:
        # Manually adding request (legacy: route all upfront)
        for i in range(16):
            for sched in router.prefill_schedulers:
                sched.add_request(
                    [i, sched.model, 64, 128, 0, sched.instance_id])
    if hbm_occupancy_accounting is not None:
        hbm_occupancy_accounting.observe(0, agentic_kv_manager)

    # Simulator start
    current = 0 # current tick of the system
    idle_time_offset = 0
    logical_time_floor = 0
    sys = 0 # current system id (NPU id)
    id = 0 # id of the request
    is_prefill_done = False # flag to check if prefill is done
    done_instance = [] # list of done instances
    done_inst_npus = [[] for _ in range(num_instances)]
    start_time = time()
    last_end_time = [0 for _ in range(num_instances)]
    last_calc_time = [0 for _ in range(num_instances)]
    waiting_request = [False for _ in range(num_instances)]

    # Calculating Simulator's Throughput
    throughput = []
    prompt_th = 0    # Avg Prompt Throguhput per Sec
    gen_th = 0       # Avg Generation Throughput per Sec
    last_log = 0    # last logged time
    FREQ = 1000_000_000 # 1 GHz (1e9 Hz)
    INTERVAL, RATIO = _throughput_interval_scale(log_interval)
    total_prompt = 0
    total_gen = 0
    total_latency = 0
    req_cnt = 0
    measurement_early_stopped = False
    measurement_stop_requested = False
    idle_no_wakeup_polls = 0
    idle_liveness_poll_limit = max(128, int(total_npu) * 16)

    exact_control_schedule = ExactControlSchedule()
    same_time_control_barrier = SameTimeControlBarrier()
    pending_same_time_control_commands = []
    pending_full_model_gpu_completions = []
    pending_full_model_prefill_completions = []

    def _simulator_auxiliary_commands(primary_command):
        commands = []
        if primary_command.startswith("advance-to:"):
            if (
                _manager_has_pending_background_jobs(agentic_kv_manager)
                or (
                    full_model_hbf_runtime is not None
                    and full_model_hbf_runtime
                    .has_pending_astra_dispatches()
                )
                or same_time_control_barrier.has_pending()
            ):
                raise RuntimeError(
                    "Cannot advance analytical time while an ASTRA-owned "
                    "background job is pending")
            return []

        if external_cold_fabric:
            for job in agentic_kv_manager.drain_external_fabric_jobs():
                if int(job["arrival_ns"]) < int(current):
                    raise RuntimeError(
                        "External cold-fabric job reached ASTRA after its "
                        f"exact arrival: job={job['job_id']}, "
                        f"arrival={job['arrival_ns']}, current={current}")
                source_instance_id = int(job["source_instance_id"])
                target_instance_id = int(job["target_instance_id"])
                source_npus = int(instances[source_instance_id]["num_npus"])
                target_npus = int(instances[target_instance_id]["num_npus"])
                lane_count = int(job["lane_count"])
                if source_npus != lane_count or target_npus != lane_count:
                    raise RuntimeError(
                        "External cold-fabric lane count disagrees with P/D "
                        f"instances: job={job['job_id']}, lanes={lane_count}, "
                        f"source_npus={source_npus}, "
                        f"target_npus={target_npus}")
                source_start = int(inst2npu_mapping[source_instance_id])
                target_start = int(inst2npu_mapping[target_instance_id])
                lanes = [
                    (source_start + rank, target_start + rank)
                    for rank in range(lane_count)
                ]
                commands.append(Controller.background_transfer_command(
                    job_id=str(job["job_id"]),
                    arrival_ns=int(job["arrival_ns"]),
                    bytes_per_lane=int(job["bytes_per_lane"]),
                    lanes=lanes,
                ))

        if full_model_hbf_runtime is not None:
            for job in (
                    full_model_hbf_runtime.adapter
                    .drain_astra_dispatches()):
                if int(job.arrival_ns) < int(current):
                    raise RuntimeError(
                        "Full-model HBF job reached ASTRA after its exact "
                        f"arrival: job={job.job_id}, "
                        f"arrival={job.arrival_ns}, current={current}")
                commands.append(job.controller_command)

        if pending_same_time_control_commands:
            commands.extend(pending_same_time_control_commands)
            pending_same_time_control_commands.clear()

        # A live model graph or ASTRA-owned background job may extend past a
        # newly ready request. Arm an exact callback so unrelated arrivals are
        # never observed only after a long transfer or HBF job finishes. This
        # must run after draining outgoing jobs: pending state then includes
        # every command emitted at this boundary.
        should_arm = not measurement_stop_requested and (
            any(scheduler.inflight for scheduler in schedulers)
            or _manager_has_pending_background_jobs(agentic_kv_manager)
            or (
                full_model_hbf_runtime is not None
                and full_model_hbf_runtime.adapter.has_pending()
            )
            or exact_control_schedule.has_pending()
            or same_time_control_barrier.has_pending()
        )
        if should_arm:
            next_wakeup = _next_exact_control_wakeup_ns(
                current, schedulers, router, agentic_kv_manager,
                full_model_hbf_runtime)
            if next_wakeup is not None:
                command = exact_control_schedule.arm(
                    int(next_wakeup), int(current))
                if command is not None:
                    commands.append(command)
        return commands

    if (
            external_cold_fabric
            or full_model_hbf_runtime is not None):
        controller.set_auxiliary_command_provider(
            _simulator_auxiliary_commands)
    if external_cold_fabric:
        logger.info(
            "Cold D->P HBM restores use congestion-aware ASTRA: "
            "bandwidth=%s GB/s, latency=%s ns",
            external_cold_fabric_bandwidth_gbps,
            external_cold_fabric_latency_ns,
        )

    # Set Event Handler that loop with INTERVAL time until first request arrive (for all instances)
    first_arival_time = router.get_first_arrival_time()
    if INTERVAL > first_arival_time:
        event_time = first_arival_time
    else:
        event_time = INTERVAL
    generate_event(int(event_time), inputs_root=run_paths.inputs_root)
    # Make Chakra Grapth
    generate_graph(None, None, total_npu, event=True, inputs_root=run_paths.inputs_root,
                   cleanup_trace=args.cleanup_inputs)
    # set first workload file
    workload = get_workload(None, None, event=True, inputs_root=run_paths.inputs_root)
    # run subprocess
    astra_args = [binary, "--workload-configuration="+workload, "--system-configuration="+system, "--network-configuration="+network, "--memory-configuration="+memory]
    if start_npu_ids != "":
        astra_args.append("--start-npu-ids="+start_npu_ids)
    if end_npu_ids != "":
        astra_args.append("--end-npu-ids="+end_npu_ids)
    if network_backend == 'ns3':
        astra_args.append("--logical-topology-configuration="+astra_sim+"/inputs/logical_topology/logical_8nodes_1D.json")
    p = subprocess.Popen(astra_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    # DP group synchronization: defer trace generation until all members have scheduled
    # dp_groups maps dp_group_name -> list of instance_ids
    dp_groups = {}
    for inst in instances:
        dg = inst.get("dp_group")
        if dg is not None:
            dp_groups.setdefault(dg, []).append(inst["instance_id"])
    # Reverse lookup: instance_id -> dp_group_name
    inst_dp_group = {}
    for dg, members in dp_groups.items():
        for inst_id in members:
            inst_dp_group[inst_id] = dg
    # Pending batches per DP group (waiting for all members to schedule)
    dp_pending = {dg: {} for dg in dp_groups}  # dp_group -> {instance_id: (new_req, sys)}
    # Pre-generated workloads ready to submit on next "Waiting"
    dp_ready_workloads = {}  # instance_id -> workload_path
    background_fabric_capability_checked = False
    hbf_background_capability_checked = False
    endpoint_park_capability_checked = False
    post_endpoint_barrier_capability_checked = False

    def _exit_if_measurement_drained():
        nonlocal measurement_early_stopped
        if not _measurement_drain_complete(
                measurement_stop_requested, schedulers, dp_pending,
                dp_ready_workloads, agentic_kv_manager,
                exact_control_schedule,
                full_model_hbf_runtime,
                same_time_control_barrier):
            return False
        measurement_early_stopped = True
        controller.write_flush(p, "exit")
        return True

    def _process_full_model_hbf_tie_barrier():
        """Atomically commit tied GPU, P/D, and HBF completion effects."""

        nonlocal prompt_th, gen_th, total_prompt, total_gen, req_cnt
        nonlocal measurement_stop_requested
        if full_model_hbf_runtime is None:
            raise RuntimeError(
                "full-model HBF tie barrier fired without a runtime")
        proxies = (
            full_model_hbf_runtime.adapter.pop_router_completions())
        hbf_requests = [
            full_model_hbf_runtime.materialize_proxy(proxy)
            for proxy in proxies
        ]
        gpu_requests = list(pending_full_model_gpu_completions)
        pending_full_model_gpu_completions.clear()
        prefill_requests = list(
            pending_full_model_prefill_completions)
        pending_full_model_prefill_completions.clear()
        terminal_prefill_ids = {
            int(request.id)
            for request in prefill_requests
            if (
                int(request.generated_tokens)
                >= int(request.requested_output_tokens)
            )
        }
        final_requests = [
            *gpu_requests,
            *hbf_requests,
            *(
                request for request in prefill_requests
                if int(request.id) in terminal_prefill_ids
            ),
        ]
        reaches_boundary = False
        if (
            final_requests
            and not measurement_stop_requested
            and session_admission.stop_after_measurement
        ):
            reaches_boundary = (
                router.measurement_boundary_would_be_reached(
                    final_requests)
            )
            if reaches_boundary:
                router.freeze_session_admission()

        hbf_request_ids = {
            int(request.id) for request in hbf_requests
        }
        prefill_request_ids = {
            int(request.id) for request in prefill_requests
        }
        requests_by_id = {
            int(request.id): request
            for request in [
                *gpu_requests,
                *hbf_requests,
                *prefill_requests,
            ]
        }
        if len(requests_by_id) != (
                len(gpu_requests)
                + len(hbf_requests)
                + len(prefill_requests)):
            raise RuntimeError(
                "same-time full-model HBF barrier received a duplicate "
                "request completion")

        source_frozen = measurement_stop_requested or reaches_boundary
        for request_id in sorted(requests_by_id):
            request = requests_by_id[request_id]
            is_prefill = request_id in prefill_request_ids
            is_terminal_prefill = request_id in terminal_prefill_ids
            if is_prefill and not is_terminal_prefill:
                if source_frozen:
                    full_model_hbf_runtime.censor_active_native_gpu_request(
                        request, now_ns=current)
                else:
                    completed = router.transfer_prefill_request(
                        [request], current)
                    if completed:
                        raise RuntimeError(
                            "non-terminal P request completed during its "
                            "deferred D handoff")
                continue

            if is_terminal_prefill:
                completed = router.transfer_prefill_request(
                    [request], current)
                if (
                    len(completed) != 1
                    or int(completed[0].id) != request_id
                ):
                    raise RuntimeError(
                        "terminal P request did not complete during its "
                        "deferred D ownership reconciliation")
                request = completed[0]
                req_cnt += 1

            censor_successor = (
                source_frozen
                and not router.request_would_complete_session(request)
            )
            if request_id in hbf_request_ids:
                full_model_hbf_runtime.finalize_hbf_request(
                    request,
                    completion_ns=current,
                    publish_successor=not censor_successor,
                )
                full_model_hbf_runtime.completed_requests.append(request)
                prompt_th += int(request.original_input)
                total_prompt += int(request.original_input)
                gen_th += int(request.generated_tokens)
                total_gen += int(request.generated_tokens)
                req_cnt += 1
            else:
                full_model_hbf_runtime.complete_native_gpu_request(
                    request,
                    completion_ns=current,
                    publish_successor=not censor_successor,
                )
            if not censor_successor:
                # A terminal call that was already dispatched before the
                # cutoff still closes its logical session during drain.
                # Router itself suppresses only non-final successors after
                # freeze.
                router.notify_request_completed(request, current)

        if reaches_boundary:
            measurement_stop_requested = True
        if dataset is not None and not measurement_stop_requested:
            router.route_arrived_requests(current)
            router.process_pending_decode_handoffs(current)
        elif measurement_stop_requested:
            # No Router pass occurs after source freeze, but already-admitted
            # HBF requests still need their deferred next decode batch.
            full_model_hbf_runtime.adapter.flush_admissions(current)
        if measurement_stop_requested:
            router.censor_idle_full_model_hbf_native_queues(current)

    def _arm_full_model_hbf_tie_barrier():
        if full_model_hbf_runtime is None:
            return
        command = same_time_control_barrier.arm(current)
        if command is not None:
            pending_same_time_control_commands.append(command)

    # ----------------------------------- Start simulation loop ------------------------------------
    # Starting simulation, one while loop processes one iteration
    while True:
        
        out = controller.read_wait(p)
        if (network_backend in _ANALYTICAL_BINARIES
                and not endpoint_park_capability_checked):
            if not controller.has_endpoint_park_capability(out):
                raise RuntimeError(
                    "Selected analytical ASTRA binary does not expose the "
                    "required endpoint-park-v1 control capability. Rebuild "
                    "the analytical backend before running the simulator.")
            endpoint_park_capability_checked = True
        if (full_model_hbf_enabled
                and not hbf_background_capability_checked):
            if not controller.has_hbf_background_capability(out):
                raise RuntimeError(
                    "Selected congestion-aware ASTRA binary does not expose "
                    "the required hbf-background-v1 capability. Rebuild the "
                    "analytical backend before running full-model HBF.")
            hbf_background_capability_checked = True
        if (full_model_hbf_enabled
                and not post_endpoint_barrier_capability_checked):
            if not controller.has_post_endpoint_barrier_capability(out):
                raise RuntimeError(
                    "Selected congestion-aware ASTRA binary does not expose "
                    "the required post-endpoint-barrier-v1 capability. "
                    "Rebuild the analytical backend before running the "
                    "full-model HBF path.")
            post_endpoint_barrier_capability_checked = True
        if (external_cold_fabric
                and not background_fabric_capability_checked):
            if not controller.has_background_fabric_capability(out):
                raise RuntimeError(
                    "Selected congestion-aware ASTRA binary does not expose "
                    "the required cold-fabric-v1 control capability. Rebuild "
                    "the analytical backend before running online cold-KV "
                    "experiments.")
            background_fabric_capability_checked = True

        protocol_event = controller.parse_protocol_event(out)
        if protocol_event is None:
            raise RuntimeError(
                "ASTRA-Sim reached a wait boundary without a model, control, "
                "or background-transfer completion record")

        if protocol_event["type"] == "control_event":
            if not (
                    external_cold_fabric
                    or full_model_hbf_runtime is not None):
                raise RuntimeError(
                    "Received an analytical control callback without an "
                    "ASTRA-owned background mode")
            current = _analytical_report_time(
                protocol_event["time_ns"], logical_time_floor)
            logical_time_floor = int(current)
            control_event_id = protocol_event["event_id"]
            if (
                    full_model_hbf_runtime is not None
                    and same_time_control_barrier.owns(control_event_id)):
                same_time_control_barrier.complete(
                    control_event_id, current)
                _process_full_model_hbf_tie_barrier()
            else:
                exact_control_schedule.complete(
                    control_event_id, current)
                if agentic_kv_manager is not None:
                    agentic_kv_manager.advance(current)
                if full_model_hbf_runtime is not None:
                    _arm_full_model_hbf_tie_barrier()
                elif dataset is not None and not measurement_stop_requested:
                    router.route_arrived_requests(current)
                    router.process_pending_decode_handoffs(current)
            if hbm_occupancy_accounting is not None:
                hbm_occupancy_accounting.observe(
                    current, agentic_kv_manager)
            if strict_oracle is not None:
                strict_oracle.observe(agentic_kv_manager)
            if _exit_if_measurement_drained():
                break
            controller.write_flush(p, "continue")
            flush.stdout.flush()
            continue

        if protocol_event["type"] == "background_transfer_complete":
            if not external_cold_fabric:
                raise RuntimeError(
                    "Received a cold-fabric completion without external mode")
            current = _analytical_report_time(
                protocol_event["completion_ns"], logical_time_floor)
            logical_time_floor = int(current)
            agentic_kv_manager.complete_external_fabric_job(
                job_id=protocol_event["job_id"],
                arrival_ns=protocol_event["arrival_ns"],
                completion_ns=protocol_event["completion_ns"],
                bytes_per_lane=protocol_event["bytes_per_lane"],
                lane_count=protocol_event["lane_count"],
                critical_lane_start_ns=(
                    protocol_event["critical_lane_start_ns"]),
            )
            if measurement_stop_requested:
                agentic_kv_manager.censor_completed_external_fabric_job(
                    protocol_event["job_id"], current)
            elif dataset is not None:
                router.route_arrived_requests(current)
                router.process_pending_decode_handoffs(current)
            if hbm_occupancy_accounting is not None:
                hbm_occupancy_accounting.observe(
                    current, agentic_kv_manager)
            if strict_oracle is not None:
                strict_oracle.observe(agentic_kv_manager)
            if _exit_if_measurement_drained():
                break
            controller.write_flush(p, "continue")
            flush.stdout.flush()
            continue

        if protocol_event["type"] == "hbf_background_complete":
            if full_model_hbf_runtime is None:
                raise RuntimeError(
                    "Received an HBF background completion without an HBF "
                    "runtime")
            current = _analytical_report_time(
                protocol_event["completion_ns"], logical_time_floor)
            logical_time_floor = int(current)
            if (
                dataset is not None
                and not measurement_stop_requested
                and not _full_model_hbf_arrival_routing_blocked(
                    full_model_hbf_runtime,
                    same_time_control_barrier,
                )
            ):
                _route_strictly_older_arrivals_at_callback(
                    current, router, None)
            full_model_hbf_runtime.complete_astra_dispatch(
                job_id=protocol_event["job_id"],
                arrival_ns=protocol_event["arrival_ns"],
                completion_ns=protocol_event["completion_ns"],
                stage_count=protocol_event["stage_count"],
            )
            _arm_full_model_hbf_tie_barrier()
            if hbm_occupancy_accounting is not None:
                hbm_occupancy_accounting.observe(
                    current, agentic_kv_manager)
            if _exit_if_measurement_drained():
                break
            controller.write_flush(p, "continue")
            flush.stdout.flush()
            continue

        if protocol_event["type"] != "model_complete":
            raise RuntimeError(
                f"Unknown ASTRA protocol event: {protocol_event}")
        controller.parse_output("".join(out))
        sys = protocol_event['sys']
        id = protocol_event['id']
        if network_backend in _ANALYTICAL_BINARIES:
            current = _analytical_report_time(
                protocol_event['cycle'], logical_time_floor)
            logical_time_floor = int(current)
        else:
            current = protocol_event['cycle'] + idle_time_offset

        # Close the previous right-continuous occupancy interval before any
        # completion, arrival, tier transition, or allocation mutates HBM at
        # this callback timestamp. Later same-time observations replace this
        # provisional state with the final post-mutation state.
        if hbm_occupancy_accounting is not None:
            hbm_occupancy_accounting.observe(current, agentic_kv_manager)

        instance_id = npu2inst_mapping[sys]  # get instance id from NPU id
        node_id = inst2node_mapping[instance_id] # get node id from instance id

        # A request-ready event can occur while ASTRA is executing a model
        # graph. Process every strictly older arrival against the still-live
        # allocation state before this callback frees completed KV or admits
        # a newly idle session. This preserves causal LRU/admission order while
        # still allowing its asynchronous restore to overlap the old graph.
        if (
            dataset is not None
            and not measurement_stop_requested
            and not _full_model_hbf_arrival_routing_blocked(
                full_model_hbf_runtime,
                same_time_control_barrier,
            )
        ):
            _route_strictly_older_arrivals_at_callback(
                current, router, agentic_kv_manager)

        # add stanby energy consumption for power modeling
        if power_modeling and sys == inst2npu_mapping[instance_id] and waiting_request[instance_id]:
            power_model.add_npu_standby_energy_consumption(instances[instance_id]["hardware"], node_id, current,
                        last_end_time[instance_id], last_calc_time[instance_id], num_npus=instances[instance_id]["num_npus"])
            last_calc_time[instance_id] = current

        # mark latest end time of the first NPU in the instance
        # An instance can span multiple NPUs. Only update end-time when sys is the first NPU of the instance.
        # waiting_request[instance_id] = True means the instance has no batch to run (idle).
        if sys == inst2npu_mapping[instance_id] and not waiting_request[instance_id]:
            last_end_time[instance_id] = current
            waiting_request[instance_id] = True

        # Snapshot online batch composition before add_done mutates request
        # progress or removes the completed batch.
        accounting_batch = model_compute_accounting._candidate_batch(
            schedulers[instance_id], id)
        if (
                full_model_hbf_runtime is not None
                and accounting_batch is not None
                and sys not in accounting_batch.end):
            _arm_full_model_hbf_tie_barrier()
        compute_observation = model_compute_accounting.prepare_completion(
            schedulers[instance_id], id, sys, current)

        # check request is done
        prompt_t, gen_t, finished_reqs = schedulers[instance_id].add_done(id, sys, current)
        if compute_observation is not None:
            if accounting_batch in schedulers[instance_id].inflight:
                raise RuntimeError(
                    "Model-compute accounting predicted a completed batch "
                    "that remained inflight")
            model_compute_accounting.record_completion(
                compute_observation, accounting_batch)
        # add tokens in throughput
        prompt_th += prompt_t
        total_prompt += prompt_t
        gen_th += gen_t
        total_gen += gen_t
        # count only finished requests
        req_cnt += len(finished_reqs) if instances[instance_id]["pd_type"] != "prefill" else 0
        if (
                full_model_hbf_runtime is not None
                and instances[instance_id]["pd_type"] != "prefill"):
            for request in finished_reqs:
                pending_full_model_gpu_completions.append(request)

        # Freeze the closed-loop source at the exact completion boundary.  A
        # tied ASTRA batch can finish several final calls together; all tied
        # final completions are recorded, while non-final calls in that batch
        # are censored instead of releasing a new turn.
        reaches_measurement_boundary = False
        if (not measurement_stop_requested
                and session_admission.stop_after_measurement
                and full_model_hbf_runtime is None
                and instances[instance_id]["pd_type"] != "prefill"):
            reaches_measurement_boundary = (
                router.measurement_boundary_would_be_reached(finished_reqs)
            )
            if reaches_measurement_boundary:
                router.freeze_session_admission()

        # Notify router of completed requests for dependency-chain release.
        # This also publishes a non-final turn's just-freed active KV as idle
        # state before any tied/capacity-waiting request may claim HBM. Once
        # drain begins, no callback is allowed to create a new turn.
        if (instances[instance_id]["pd_type"] != "prefill"
                and not measurement_stop_requested
                and full_model_hbf_runtime is None):
            for req in finished_reqs:
                if (reaches_measurement_boundary
                        and not router.request_would_complete_session(req)):
                    continue
                router.notify_request_completed(req, current)

        if reaches_measurement_boundary:
            measurement_stop_requested = True

        # Add prefill ended requests to decode instance
        handoff_completed_reqs = []
        if (
                full_model_hbf_runtime is not None
                and instances[instance_id]["pd_type"] == "prefill"
                and finished_reqs):
            # D ownership transfer is a new execution-enabling action. Keep
            # every completed P request behind the same-time barrier so tied
            # terminal calls can freeze the source before any new D graph is
            # made runnable. One-output requests are reconciled and finalized
            # there as well.
            pending_full_model_prefill_completions.extend(
                finished_reqs)
        elif (not measurement_stop_requested
                and instances[instance_id]["pd_type"] == "prefill"
                and len(finished_reqs) > 0):
            handoff_completed_reqs = router.transfer_prefill_request(
                finished_reqs, current)
            req_cnt += len(handoff_completed_reqs)
        elif (measurement_stop_requested
                and agentic_kv_manager is not None
                and instances[instance_id]["pd_type"] == "prefill"
                and len(finished_reqs) > 0):
            # The P batch was already dispatched when the measurement source
            # froze. add_done() released its complete P allocation, but the D
            # receive buffer was preallocated before P launch and would be
            # orphaned if the ordinary handoff were simply skipped.
            router.censor_completed_pd_prefill_requests(
                finished_reqs, current)

        # A one-output P/D request has already produced its sole output in P,
        # but the request becomes complete only after the pre-admitted D-side
        # KV ownership handoff is reconciled. No D model graph is launched.
        if handoff_completed_reqs:
            if (not measurement_stop_requested
                    and session_admission.stop_after_measurement
                    and full_model_hbf_runtime is None):
                reaches_measurement_boundary = (
                    router.measurement_boundary_would_be_reached(
                        handoff_completed_reqs)
                )
                if reaches_measurement_boundary:
                    router.freeze_session_admission()
            if (
                    not measurement_stop_requested
                    and full_model_hbf_runtime is None):
                for req in handoff_completed_reqs:
                    if (reaches_measurement_boundary
                            and not router.request_would_complete_session(req)):
                        continue
                    router.notify_request_completed(req, current)
            if reaches_measurement_boundary:
                measurement_stop_requested = True

        if strict_oracle is not None:
            strict_oracle.observe(agentic_kv_manager)

        if hbm_occupancy_accounting is not None:
            hbm_occupancy_accounting.observe(current, agentic_kv_manager)

        # Do not terminate ASTRA while another dispatched graph is live.  The
        # source is frozen above, then already-dispatched batches drain to
        # their callbacks. Queued requests and unlaunched continuations are
        # intentionally censored.
        if _exit_if_measurement_drained():
            break

        # Completion/free and active-to-idle ownership handoffs at this
        # timestamp precede new HBM admission.
        # route_arrived_requests prepares continuations in release-time order;
        # prepare_request advances background migrations only to each release
        # time, preventing a later background job from jumping ahead of an
        # earlier foreground restore. Remaining background events are then
        # advanced to the current simulation timestamp.
        if (dataset is not None
                and not measurement_stop_requested
                and full_model_hbf_runtime is None):
            router.route_arrived_requests(current)
        if agentic_kv_manager is not None:
            agentic_kv_manager.advance(current)
            if not measurement_stop_requested:
                router.process_pending_decode_handoffs(current)
        if strict_oracle is not None:
            # Sample again after tied admissions: the earlier completion-side
            # sample captures the active-to-idle handoff, while this one also
            # captures any new physical or logical HBM claims it enabled.
            strict_oracle.observe(agentic_kv_manager)
        if hbm_occupancy_accounting is not None:
            hbm_occupancy_accounting.observe(current, agentic_kv_manager)

        # schedule requests
        dp_group = inst_dp_group.get(instance_id)
        complete_pending_dp_wave = (
            measurement_stop_requested
            and dp_group is not None
            and bool(dp_pending[dp_group])
            and instance_id not in dp_pending[dp_group]
        )
        full_model_completion_barrier_pending = (
            _full_model_hbf_dispatch_blocked(
                full_model_hbf_runtime,
                same_time_control_barrier,
            )
        )
        new_req = (
            schedulers[instance_id].schedule(current, sys, id)
            if (
                not full_model_completion_barrier_pending
                and (
                    not measurement_stop_requested
                    or complete_pending_dp_wave
                )
            )
            else None
        )
        if new_req is not None:
            idle_no_wakeup_polls = 0
        prepare_locked = (
            agentic_kv_manager is not None
            and agentic_kv_manager.prepare_locked(instance_id)
        )
        sync_transfer_blocked = (
            agentic_kv_manager is not None
            and agentic_kv_manager.synchronous_swap_blocked_until(
                instance_id, current) is not None
        )
        model_dispatch_blocked = _model_dispatch_blocked(
            agentic_kv_manager, schedulers[instance_id], current)
        sync_engine_blocked = (
            prepare_locked or sync_transfer_blocked
            or model_dispatch_blocked
            or full_model_completion_barrier_pending
        )
        responded = False  # track whether we already sent a response to ASTRA-Sim

        # Check if a pre-generated workload is ready for this instance (from DP sync)
        if (new_req is None and instance_id in dp_ready_workloads
                and not sync_engine_blocked):
            ready_workload, ready_batch = dp_ready_workloads.pop(instance_id)
            _record_astra_dispatch(
                agentic_kv_manager, schedulers[instance_id], ready_batch,
                current)
            controller.write_flush(p, ready_workload)
            responded = True
        # DP group: truly idle instance (no inflight batch) — create dummy batch so ALLTOALL syncs
        elif (new_req is None and instance_id in inst_dp_group
                and not sync_engine_blocked
                and sys == inst2npu_mapping[instance_id]
                and len(schedulers[instance_id].inflight) == 0):
            dg = inst_dp_group[instance_id]
            if dp_pending[dg]:
                # Emit a 1-token dummy; the uniform pad-to-max pass below
                # brings it (and any undersized real peers) up to the
                # group's max_total_len, matching vLLM's CUDA-graph DP padding.
                logger.debug(f"Instance {instance_id} is idle but DP group {dg} has pending batches. Creating dummy batch for synchronization.")
                dummy = Batch(schedulers[instance_id].get_batch_id(), instances[instance_id]["model_name"],
                              1, 1, [1], [], 0, 1, [], [], [1], current, 0)
                dummy.fired.append(sys)
                if agentic_kv_manager is not None:
                    agentic_kv_manager.record_agentic_batch_schedule(
                        schedulers[instance_id], dummy)
                schedulers[instance_id].inflight.append(dummy)
                dp_pending[dg][instance_id] = (dummy, inst2node_mapping[instance_id])

                if len(dp_pending[dg]) == len(dp_groups[dg]):
                    # All DP members accounted for — pad every batch to the
                    # group's max (vLLM CUDA-graph DP padding) and generate.
                    config = get_config(instances[instance_id]["model_name"])
                    max_total_len = max(b.total_len for b, _ in dp_pending[dg].values())
                    for b, _ in dp_pending[dg].values():
                        _pad_batch_to_max(b, max_total_len)
                    # MoE AG/RS comm size is anchored to ``max_total_len``
                    # (not ``max × group_size``). The trace generator divides
                    # this by ep_total internally for the per-rank AG chunk
                    # and uses the same value for the RS pre-scatter buffer.
                    # Empirically this matches real NCCL AG/RS bandwidth on
                    # PCIe 5.0 at the same ``link_bw`` that already calibrates
                    # AllReduce — i.e. ASTRA-Sim's Ring half-duplex model
                    # ends up correct for AR but 2× over real AG/RS, and the
                    # "× group_size" we used previously stacked the two errors.
                    sum_total_len = max_total_len

                    # Shared workload folder for all DP members
                    first_inst_id = dp_groups[dg][0]
                    first_batch = dp_pending[dg][first_inst_id][0]
                    dp_workload_name = f'{instances[first_inst_id]["hardware"]}/{instances[first_inst_id]["model_name"]}/dp_{dg}_batch{first_batch.batch_id}'

                    for inst_id in dp_groups[dg]:
                        batch, nid = dp_pending[dg][inst_id]
                        inst = instances[inst_id]
                        inst_cfg = instance_runtime_configs[inst_id]
                        generate_trace(batch, inst["hardware"], inst["tp_size"], inst["pp_size"],
                                       inst["local_ep"], inst["ep_total"], inst["pd_type"],
                                       nid, inst_id,
                                       inst_cfg["max_num_batched_tokens"], inst_cfg["max_num_seqs"],
                                       placement[inst_id], block_mode_on[inst_id],
                                       expert_routing_policy, inst_cfg["enable_prefix_caching"],
                                       inst_cfg["enable_attn_offloading"],
                                       power_model, pim_models[nid],
                                       inst_cfg["enable_sub_batch_interleaving"], inst_cfg["fp"],
                                       dtype=inst_cfg["dtype"], kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                                       tp_dim=inst.get("tp_dim"), ep_dim=inst.get("ep_dim"),
                                       dp_sum_total_len=sum_total_len,
                                       enable_block_copy=inst_cfg["enable_block_copy"],
                                       latency_model=inst_cfg["latency_model"],
                                       latency_model_band=inst_cfg["latency_model_band"],
                                       inputs_root=run_paths.inputs_root)
                        generate_graph(batch, inst["hardware"], inst["num_npus"], nid,
                                       inst_id, inst2npu_mapping[inst_id],
                                       inst_cfg["enable_local_offloading"],
                                       workload_name=dp_workload_name,
                                       inputs_root=run_paths.inputs_root,
                                       cleanup_trace=args.cleanup_inputs)
                        if inst_id != instance_id:
                            dp_ready_workloads[inst_id] = (
                                get_workload(
                                    batch, inst["hardware"], inst_id,
                                    workload_name=dp_workload_name,
                                    inputs_root=run_paths.inputs_root),
                                batch,
                            )

                    dp_pending[dg].clear()
                    workload = get_workload(dummy, instances[instance_id]["hardware"], instance_id,
                                            workload_name=dp_workload_name,
                                            inputs_root=run_paths.inputs_root)
                    if _model_dispatch_blocked(
                            agentic_kv_manager,
                            schedulers[instance_id], current):
                        dp_ready_workloads[instance_id] = (workload, dummy)
                        controller.write_flush(p, "pass")
                    else:
                        _record_astra_dispatch(
                            agentic_kv_manager,
                            schedulers[instance_id], dummy, current)
                        controller.write_flush(p, workload)
                    responded = True
                else:
                    controller.write_flush(p, "pass")
                    responded = True
        # runnable batch exists
        elif new_req is not None:
            if sys == inst2npu_mapping[instance_id]:  # first NPU of the instance
                waiting_request[instance_id] = False
                instance = instances[instance_id]
                dg = inst_dp_group.get(instance_id)

                if dg is not None:
                    # DP group: defer trace generation until all members scheduled
                    dp_pending[dg][instance_id] = (new_req, node_id)

                    if len(dp_pending[dg]) == len(dp_groups[dg]):
                        # All DP members have scheduled — pad every batch to
                        # the group's max (vLLM CUDA-graph DP padding) so
                        # smaller batches gain dummy decodes that all layers
                        # still compute over.
                        config = get_config(instance["model_name"])
                        max_total_len = max(b.total_len for b, _ in dp_pending[dg].values())
                        for b, _ in dp_pending[dg].values():
                            _pad_batch_to_max(b, max_total_len)
                        # See twin block above: anchor MoE comm to max_total_len
                        # (no group-size multiplier).
                        sum_total_len = max_total_len

                        # Shared workload folder for all DP members
                        first_inst_id = dp_groups[dg][0]
                        first_batch = dp_pending[dg][first_inst_id][0]
                        dp_workload_name = f'{instances[first_inst_id]["hardware"]}/{instances[first_inst_id]["model_name"]}/dp_{dg}_batch{first_batch.batch_id}'

                        for inst_id in dp_groups[dg]:
                            batch, nid = dp_pending[dg][inst_id]
                            inst = instances[inst_id]
                            inst_cfg = instance_runtime_configs[inst_id]
                            generate_trace(batch, inst["hardware"], inst["tp_size"], inst["pp_size"],
                                           inst["local_ep"], inst["ep_total"], inst["pd_type"],
                                           nid, inst_id,
                                           inst_cfg["max_num_batched_tokens"], inst_cfg["max_num_seqs"],
                                           placement[inst_id], block_mode_on[inst_id],
                                           expert_routing_policy, inst_cfg["enable_prefix_caching"],
                                           inst_cfg["enable_attn_offloading"],
                                           power_model, pim_models[nid],
                                           inst_cfg["enable_sub_batch_interleaving"], inst_cfg["fp"],
                                           dtype=inst_cfg["dtype"], kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                                           tp_dim=inst.get("tp_dim"), ep_dim=inst.get("ep_dim"),
                                           dp_sum_total_len=sum_total_len,
                                           enable_block_copy=inst_cfg["enable_block_copy"],
                                           latency_model=inst_cfg["latency_model"],
                                           latency_model_band=inst_cfg["latency_model_band"],
                                           inputs_root=run_paths.inputs_root)
                            generate_graph(batch, inst["hardware"], inst["num_npus"], nid,
                                           inst_id, inst2npu_mapping[inst_id],
                                           inst_cfg["enable_local_offloading"],
                                           workload_name=dp_workload_name,
                                           inputs_root=run_paths.inputs_root,
                                           cleanup_trace=args.cleanup_inputs)
                            if inst_id != instance_id:
                                dp_ready_workloads[inst_id] = (
                                    get_workload(
                                        batch, inst["hardware"], inst_id,
                                        workload_name=dp_workload_name,
                                        inputs_root=run_paths.inputs_root),
                                    batch,
                                )

                        dp_pending[dg].clear()
                        workload = get_workload(new_req, instance["hardware"], instance_id,
                                                workload_name=dp_workload_name,
                                                inputs_root=run_paths.inputs_root)
                        if _model_dispatch_blocked(
                                agentic_kv_manager,
                                schedulers[instance_id], current):
                            dp_ready_workloads[instance_id] = (
                                workload, new_req)
                            controller.write_flush(p, "pass")
                            responded = True
                        else:
                            _record_astra_dispatch(
                                agentic_kv_manager,
                                schedulers[instance_id], new_req, current)
                            controller.write_flush(p, workload)
                    else:
                        # Waiting for other DP members — send pass
                        controller.write_flush(p, "pass")
                        responded = True
                else:
                    # Independent instance: generate trace immediately
                    inst_cfg = instance_runtime_configs[instance_id]
                    generate_trace(new_req, instance["hardware"], instance["tp_size"], instance["pp_size"],
                                   instance["local_ep"], instance["ep_total"],
                                   instance["pd_type"],
                                   node_id, instance_id,
                                   inst_cfg["max_num_batched_tokens"], inst_cfg["max_num_seqs"],
                                   placement[instance_id], block_mode_on[instance_id],
                                   expert_routing_policy, inst_cfg["enable_prefix_caching"],
                                   inst_cfg["enable_attn_offloading"], power_model, pim_models[node_id],
                                   inst_cfg["enable_sub_batch_interleaving"], inst_cfg["fp"],
                                   dtype=inst_cfg["dtype"], kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                                   tp_dim=instance["tp_dim"], ep_dim=instance["ep_dim"],
                                   enable_block_copy=inst_cfg["enable_block_copy"],
                                   latency_model=inst_cfg["latency_model"],
                                   latency_model_band=inst_cfg["latency_model_band"],
                                   inputs_root=run_paths.inputs_root)
                    generate_graph(new_req, instance["hardware"], instance["num_npus"], node_id,
                                   instance_id, inst2npu_mapping[instance_id],
                                   inst_cfg["enable_local_offloading"],
                                   inputs_root=run_paths.inputs_root,
                                   cleanup_trace=args.cleanup_inputs)
                    workload = get_workload(new_req, instance["hardware"], instance_id,
                                            inputs_root=run_paths.inputs_root)
                    _record_astra_dispatch(
                        agentic_kv_manager, schedulers[instance_id],
                        new_req, current)
                    controller.write_flush(p, workload)
            elif new_req is not None:
                # Non-first NPU: pick up existing batch workload
                workload = get_workload(new_req, instances[instance_id]["hardware"], instance_id,
                                        inputs_root=run_paths.inputs_root)
                controller.write_flush(p, workload)

        # check time to store throughput (only print on start NPU to avoid transient states)
        if current > last_log + INTERVAL and sys == inst2npu_mapping[instance_id]:
            # store the prompt
            throughput.append((prompt_th*RATIO, gen_th*RATIO))
            last_log += INTERVAL
            log_time_str = f"[{last_log / FREQ:.1f}s]"
            log_time_len = len(log_time_str)
            log_indent = ' ' * log_time_len + '  '
            tree_indent = '├─'
            # Heartbeat timestamp stays in the terminal's default
            # colour — bright enough to scan, not so dim that it
            # disappears. (The per-log-record [HH:MM:SS.mmm] stays
            # dim via sim.time because it appears every other line.)
            print_markup(
                f"{log_time_str} "
                f"[blue]Avg prompt throughput: {prompt_th * RATIO:.1f} tokens/s,[/] "
                f"[blue]Avg generation throughput: {gen_th * RATIO:.1f} tokens/s[/]"
            )
            prompt_th = 0
            gen_th = 0

            ######### Per Instance Metrics #########

            for inst_id in range(num_instances):
                running_reqs = sum(len(batch.requests) for batch in schedulers[inst_id].inflight)
                waiting_reqs = len([
                    req for req in schedulers[inst_id].request
                    if req.ready_time <= current
                ])

                mem = schedulers[inst_id].memory
                npu_used_mb = mem.npu_used / MB_TO_BYTE
                npu_util = (mem.npu_used / mem.npu_mem * 100.0) if mem.npu_mem else 0.0

                line = (
                    f"{log_indent+tree_indent}Running Instance\\[{inst_id}]: "
                    f"{running_reqs} reqs, Waiting: {waiting_reqs} reqs, "
                    f"Total # {schedulers[inst_id].num_npus} NPUs, "
                    f"Each NPU Memory Usage {npu_used_mb:.2f} MB "
                    f"({npu_util:.3f} % Used)"
                )
                if schedulers[inst_id].enable_prefix_caching:
                    line += schedulers[inst_id].memory.npu_prefix_cache.format_prefix_info()
                print_markup(line)

            ######### Per Node Metrics #########
            if node2inst_mapping:
                num_nodes = len(node2inst_mapping)
                for i, (node_id, inst_ids) in enumerate(node2inst_mapping.items()):
                    node_cpu_usage = 0
                    inst_usage = []
                    if any_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU":
                        node_cpu_usage = prefix_pools[node_id].total_size() * prefix_pools[node_id].kv_size
                    else:
                        for inst_id in inst_ids:
                            inst_cpu_usage = schedulers[inst_id].memory.cpu_used
                            node_cpu_usage += inst_cpu_usage
                            inst_usage.append(inst_cpu_usage)

                    cpu_util = (node_cpu_usage / (cpu_mem_size[node_id]*GB_TO_BYTE)) * 100
                    if prefix_storage != "CXL" and not power_modeling and i == num_nodes - 1:
                        tree_indent = '└─'
                    line = (
                        f"{log_indent+tree_indent}Node\\[{node_id}]: "
                        f"Total CPU Memory Usage {node_cpu_usage/MB_TO_BYTE:.2f} MB, "
                        f"{cpu_util:.3f} % Used "
                    )
                    if any_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU":
                        line += prefix_pools[node_id].format_prefix_info()

                    if (any_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU") or (len(inst_ids) == 1):
                        print_markup(line)
                    else:
                        parts = []
                        for j, inst_cpu_usage in enumerate(inst_usage):
                            inst_cpu_util = (inst_cpu_usage / node_cpu_usage)*100 if node_cpu_usage else 0
                            parts.append(f"Instance\\[{inst_ids[j]}]: {inst_cpu_util:.2f} %")
                        print_markup(line + "(" + ", ".join(parts) + ")")

            ######### Per CXL Metrics #########
            if any_prefix_caching and prefix_storage == "CXL":
                if enable_prefix_sharing:
                    num_prefix_pool = len(prefix_pools)
                    for cxl_id, cxl_pool in enumerate(prefix_pools):
                        cxl_usage = cxl_pool.total_size() * cxl_pool.kv_size
                        cxl_util = cxl_usage / cxl_pool.capacity
                        if not power_modeling and cxl_id == num_prefix_pool - 1:
                            tree_indent = '└─'
                        print_markup(
                            f"{log_indent+tree_indent}CXL\\[{cxl_id}]: "
                            f"Total CXL Device Memory Usage "
                            f"{cxl_usage/MB_TO_BYTE:.2f}MB, {cxl_util:.3f} % Used"
                        )
                else:
                    enabled_inst_ids = [
                        inst_id for inst_id, sched in enumerate(schedulers)
                        if sched.enable_prefix_caching
                    ]
                    for pos, inst_id in enumerate(enabled_inst_ids):
                        second_tier = getattr(
                            schedulers[inst_id].memory, "second_tier_prefix_cache", None)
                        if second_tier is None:
                            continue
                        cxl_usage = second_tier.total_size() * second_tier.kv_size
                        cxl_util = cxl_usage / second_tier.capacity
                        if not power_modeling and pos == len(enabled_inst_ids) - 1:
                            tree_indent = '└─'
                        print_markup(
                            f"{log_indent+tree_indent}CXL\\[0]/Instance\\[{inst_id}]: "
                            f"Total CXL Device Memory Usage {cxl_usage / MB_TO_BYTE:.2f} MB, "
                            f"{cxl_util:.3f} % Used"
                        )

            ######### Power Modeling #########
            if power_modeling:
                tree_indent = '└─'
                print_markup(
                    f"{log_indent+tree_indent}"
                    f"Avg power consumption: {power_model.get_current_power(current)} W"
                )
        # Scheduling can allocate active KV or create a future HBM claim. Make
        # that final state authoritative for the interval after this callback.
        if hbm_occupancy_accounting is not None:
            hbm_occupancy_accounting.observe(current, agentic_kv_manager)

        # check if all requests are done for current instance#
        # NOTE: 'instance_id' could occur in duplicate, because 'npu2inst_mapping[sys]' is not one-to-one mapping
        if ((instance_id not in decode_instance or is_prefill_done)
                and instance_id not in done_instance
                and schedulers[instance_id].is_request_empty()
                and not router.has_pending_requests()
                and not router.has_deferred_sessions()
                and not router.has_pending_decode_handoffs()
                and not pending_full_model_gpu_completions
                and (
                    full_model_hbf_runtime is None
                    or not full_model_hbf_runtime.adapter.has_pending()
                )
                and not same_time_control_barrier.has_pending()):
            # For DP groups: only mark done when ALL members of the group are empty
            dg = inst_dp_group.get(instance_id)
            if dg is not None:
                all_dp_empty = all(
                    schedulers[inst_id].is_request_empty() and len(schedulers[inst_id].inflight) == 0
                    for inst_id in dp_groups[dg]
                )
                if not all_dp_empty:
                    # Other DP members still have work — keep this instance alive for dummy waves
                    if not responded:
                        controller.write_flush(p, "pass")
                    flush.stdout.flush()
                    continue

            if sys not in done_inst_npus[instance_id]:
                done_inst_npus[instance_id].append(sys)
            if len(done_inst_npus[instance_id]) == (1 if instances[instance_id]["num_npus"] == 1 else 2):
                done_instance.append(instance_id)

            # check if all prefill instances are done
            if len(done_instance) == len(prefill_instance):
                is_prefill_done = True

            # check if all instances are done
            if len(done_instance) == num_instances:
                for inst_idx in range(num_instances):
                    schedulers[inst_idx].memory.free_prefix_cache()
                    schedulers[inst_idx].memory.free_weight()

                # Check every independent scheduler allocation. The former
                # post-loop lookup inspected only the final instance and could
                # silently miss a leak on any earlier P/D or DP member.
                leaking_instances = [
                    inst_idx for inst_idx in range(num_instances)
                    if not schedulers[inst_idx].memory.is_free()
                ]
                if leaking_instances:
                    raise RuntimeError(
                        "Memory leak detected after normal completion on "
                        f"scheduler instances {leaking_instances}")

                print_rule()
                print_markup("[sim.heading]▶ Exiting simulation...[/]\n")
                controller.write_flush(p, "exit")
                break
            controller.write_flush(p, "done") # make done instances to sleep
        elif new_req == None and not responded:
            # If no instance can run, advance toward either a future request
            # arrival or an idle-KV reclaim completion. Analytical backends
            # use an absolute timestamp; legacy backends retain one poll so
            # ASTRA-Sim observes the event at its exact logical timestamp.
            idle_endpoint_command = "pass"
            if not any(dp_pending.values()) and not dp_ready_workloads:
                known_nonrunnable_instance_id = (
                    instance_id
                    if (sys == inst2npu_mapping[instance_id]
                        and (not measurement_stop_requested
                             or complete_pending_dp_wave))
                    else None
                )
                next_wakeup = _next_idle_wakeup_ns(
                    current, schedulers, router,
                    known_nonrunnable_instance_id=(
                        known_nonrunnable_instance_id),
                )
                unresolved_idle_work = (
                    any(scheduler.request for scheduler in schedulers)
                    or router.has_pending_requests()
                    or router.has_deferred_sessions()
                    or router.has_pending_decode_handoffs()
                    or (
                        full_model_hbf_runtime is not None
                        and full_model_hbf_runtime.adapter.has_pending()
                    )
                )
                no_dispatched_work = not any(
                    scheduler.inflight for scheduler in schedulers)
                external_fabric_work = (
                    _manager_has_pending_background_jobs(
                        agentic_kv_manager)
                    or (
                        full_model_hbf_runtime is not None
                        and full_model_hbf_runtime
                        .has_pending_astra_dispatches()
                    )
                )
                dispatched_model_work = _has_dispatched_model_work(
                    schedulers, agentic_kv_manager)
                partially_observed_completion = (
                    _has_partially_observed_model_completion(schedulers)
                )
                idle_endpoint_command = _analytical_idle_endpoint_command(
                    network_backend,
                    sys == inst2npu_mapping[instance_id],
                    (
                        dispatched_model_work
                        or external_fabric_work
                        or exact_control_schedule.has_pending()
                        or same_time_control_barrier.has_pending()
                    ),
                    partially_observed_completion,
                )
                if (next_wakeup is None and unresolved_idle_work
                        and no_dispatched_work
                        and not external_fabric_work
                        and not exact_control_schedule.has_pending()):
                    idle_no_wakeup_polls += 1
                    scheduler_state = [
                        {
                            "instance_id": scheduler.instance_id,
                            "queued": len(scheduler.request),
                            "inflight": len(scheduler.inflight),
                            "head_ready_ns": (
                                int(scheduler.request[0].ready_time)
                                if scheduler.request else None
                            ),
                            "memory_wait_until_ns": (
                                scheduler.memory_wait_until_ns
                            ),
                            "model_fabric_wait_until_ns": (
                                scheduler.model_fabric_wait_until_ns
                            ),
                            "head_request": (
                                {
                                    "request_id": int(
                                        scheduler.request[0].id),
                                    "computed_tokens": int(
                                        scheduler.request[0]
                                        .num_computed_tokens),
                                    "prefill_target_tokens": int(
                                        scheduler.request[0]
                                        .prefill_target_tokens),
                                    "restore_ready_ns": int(
                                        scheduler.request[0]
                                        .agentic_kv_restore_ready_time_ns),
                                    "prefill_preallocated_per_rank_bytes": int(
                                        scheduler.request[0]
                                        .pd_prefill_preallocated_per_rank_bytes
                                    ),
                                }
                                if scheduler.request else None
                            ),
                            "npu_used": int(scheduler.memory.npu_used),
                            "npu_allocatable_mem": int(
                                scheduler.memory.npu_allocatable_mem),
                            "decode_handoff_claim_pending": bool(
                                scheduler.decode_handoff_claim_pending),
                        }
                        for scheduler in schedulers
                    ]
                    if idle_no_wakeup_polls >= idle_liveness_poll_limit:
                        raise RuntimeError(
                            "Online scheduler reached an idle liveness "
                            "failure after "
                            f"{idle_liveness_poll_limit} controller polls: "
                            "unfinished "
                            "work has no future arrival, restore, admission, "
                            "or fabric wakeup. "
                            f"current_ns={current}, "
                            f"callback_sys={sys}, callback_batch_id={id}, "
                            "partially_observed_completion="
                            f"{partially_observed_completion}, "
                            f"schedulers={scheduler_state}, pending_requests="
                            f"{router.has_pending_requests()}, "
                            "deferred_sessions="
                            f"{router.has_deferred_sessions()}, "
                            "pending_handoffs="
                            f"{router.has_pending_decode_handoffs()}"
                        )
                else:
                    idle_no_wakeup_polls = 0
                # The absolute-time protocol is legal only when ASTRA has no
                # live graph on any instance. _next_idle_wakeup_ns applies the
                # same guard, and this explicit check keeps the wire protocol
                # safe if a future wakeup source bypasses that helper.
                if (next_wakeup is not None and next_wakeup > current
                        and no_dispatched_work
                        and not external_fabric_work
                        and not exact_control_schedule.has_pending()
                        and not same_time_control_barrier.has_pending()):
                    if network_backend in _ANALYTICAL_BINARIES:
                        # Analytical ASTRA reports the completion timestamp of
                        # its last workload while idle, not its internally
                        # accumulated fixed polling ticks. Advancing its event
                        # queue and the Python clock to one absolute timestamp
                        # avoids both an O(gap/poll) busy loop and a 1 ms
                        # quantization error on short restore/admission events.
                        command = _analytical_advance_command(
                            current, next_wakeup)
                        logical_time_floor = int(next_wakeup)
                        current = int(next_wakeup)
                        controller.write_flush(p, command)
                        responded = True
                    else:
                        offset_adjustment = _idle_fast_forward_delta(
                            current, next_wakeup, network_backend)
                        idle_time_offset += offset_adjustment
                        current += offset_adjustment
            if not responded:
                controller.write_flush(p, idle_endpoint_command)

        # An analytical idle jump advances ``current`` without changing memory.
        # Extending the last state to that exact timestamp closes the otherwise
        # unobserved idle interval before the next controller callback.
        if hbm_occupancy_accounting is not None:
            hbm_occupancy_accounting.observe(current, agentic_kv_manager)

        # flush
        flush.stdout.flush()

    # calculate simulation time
    end_time = time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # check all scheduled requests in astra-sim are well done
    controller.check_end(p)

    censoring_summary = None
    if measurement_early_stopped:
        censoring_summary = router.finalize_measurement_censoring(current)

    if full_model_hbf_runtime is not None:
        full_model_hbf_runtime.assert_quiescent()
    reporting_schedulers = (
        full_model_hbf_runtime.reporting_schedulers(schedulers)
        if full_model_hbf_runtime is not None
        else schedulers
    )

    agentic_kv_summary = None
    if agentic_kv_manager is not None:
        agentic_kv_manager.record_active_preemption_totals(
            recompute_preemptions=sum(
                scheduler.active_recompute_preemptions
                for scheduler in schedulers),
            recompute_tokens=sum(
                scheduler.active_recompute_tokens
                for scheduler in schedulers),
            cpu_swap_preemptions=sum(
                scheduler.active_cpu_swap_preemptions
                for scheduler in schedulers),
            cpu_swap_write_bytes=sum(
                scheduler.active_cpu_swap_write_bytes
                for scheduler in schedulers),
            cpu_swap_read_bytes=sum(
                scheduler.active_cpu_swap_read_bytes
                for scheduler in schedulers),
        )
        total_request_latency_ns = sum(
            req.latency
            for scheduler in schedulers
            if scheduler.pd_type != "prefill"
            for req in scheduler.done
            if req.latency >= 0
        )
        agentic_kv_manager.record_simulation_totals(
            total_request_latency_ns=total_request_latency_ns,
            total_model_compute_ns=(
                model_compute_accounting.total_model_compute_ns),
            recompute_model_compute_ns=(
                model_compute_accounting.recompute_model_compute_ns),
            total_prompt_tokens=total_prompt,
        )
        agentic_kv_summary = agentic_kv_manager.summary(
            simulated_duration_ns=current,
            dataset=dataset,
            run_id=args.run_id,
            measurement_censored=measurement_early_stopped,
        )
        bridge_audit = agentic_kv_summary["online_resource_bridge"]
        if (bridge_audit["open_astra_window_count"] != 0
                or bridge_audit[
                    "pending_direct_fabric_prepare_locks"] != 0):
            raise RuntimeError(
                "Online resource bridge is not quiescent at reporting: "
                f"{bridge_audit}")
        if censoring_summary is not None:
            censoring_summary["migration_tail_at_cutoff"] = (
                agentic_kv_summary["measurement_cutoff_dma_tail"])

    strict_oracle_validation = None
    if strict_oracle is not None:
        completed_requests = [
            request
            for scheduler in schedulers
            if scheduler.pd_type != "prefill"
            for request in scheduler.done
            if request.session_id is not None
        ]
        strict_oracle_validation = strict_oracle.validate(
            agentic_kv_manager, completed_requests)

    session_metrics_report = None
    session_admission_summary = router.session_admission_summary()
    if session_admission_summary['templates_loaded'] > 0:
        session_metrics_report = build_session_metrics(
            router,
            reporting_schedulers,
            current,
            dataset=dataset,
            run_id=args.run_id,
            measurement_early_stopped=measurement_early_stopped,
            online_compute=model_compute_accounting.summary(),
            oracle_validation=strict_oracle_validation,
            censoring=censoring_summary,
            hbm_occupancy_accounting=hbm_occupancy_accounting,
        )
        measurement_window = session_metrics_report["measurement_window"]
        measurement_start_ns = measurement_window["measurement_start_ns"]
        measurement_end_ns = measurement_window["measurement_end_ns"]
        if (measurement_start_ns is not None
                and measurement_end_ns is not None):
            session_metrics_report["online_model_compute"] = (
                model_compute_accounting.summary(
                    measurement_start_ns, measurement_end_ns)
            )
        if full_model_hbf_runtime is not None:
            gpu_compute = session_metrics_report[
                "online_model_compute"]
            if gpu_compute is not None:
                gpu_compute["accounting_scope"] = "gpu_server_only"
                gpu_compute["full_model_hbf_excluded"] = True
            hbf_report_for_session = full_model_hbf_runtime.report()
            session_metrics_report["full_model_hbf"] = {
                "schema": hbf_report_for_session["schema"],
                "layout": hbf_report_for_session["layout"],
                "runtime_metrics_path": args.full_model_hbf_metrics,
                "completed_hbf_request_count": (
                    hbf_report_for_session[
                        "completed_hbf_request_count"]),
                "execution_counts": (
                    hbf_report_for_session[
                        "adapter"]["execution_counts"]),
                "adapter_metrics": (
                    hbf_report_for_session["adapter"]["metrics"]),
                "pool_metrics": (
                    hbf_report_for_session[
                        "adapter"]["pool"]["metrics"]),
                "lifecycle_metrics": (
                    hbf_report_for_session[
                        "adapter"]["lifecycle"]["metrics"]),
                "compute_accounting_semantics": (
                    "online_model_compute covers only GPU-server model "
                    "batches; HBF-GPU kernels, collectives, media, LPDDR, "
                    "PCIe, and RDMA timing are callback-owned by ASTRA and "
                    "reported in this HBF summary/runtime report. They are "
                    "not added because GPU and HBF work may overlap."
                ),
            }

    # calcuate prefix caching metrics
    total_requested_tokens = 0
    total_npu_hit_tokens = 0
    total_cpu_hit_tokens = 0
    if any_prefix_caching:
        for i in range(num_instances):
            if not schedulers[i].enable_prefix_caching:
                continue
            (temp_npu_a, temp_npu_b), (temp_cpu_a, temp_cpu_b) = schedulers[i].memory.return_prefix_info()
            if (not enable_prefix_sharing) and (prefix_storage != "None") and (temp_npu_a != temp_cpu_a):
                raise RuntimeError(f"Instance[{i}] prefix caching requested tokens mismatch between NPU ({temp_npu_a}) and CPU ({temp_cpu_a})")
            total_requested_tokens += temp_npu_a
            total_npu_hit_tokens += temp_npu_b
            if not enable_prefix_sharing:
                total_cpu_hit_tokens += temp_cpu_b
        
        if enable_prefix_sharing:
            for pool in prefix_pools:
                _, temp_cpu_b = pool.return_prefix_info()
                total_cpu_hit_tokens += temp_cpu_b
    
    # This is total system's throughput
    total_latency = current/FREQ
    print_rule()
    print_markup("[sim.heading]▶ Simulation results...[/]\n")
    print_markup(f"Total simulation time: {int(hours)}h {int(minutes)}m {seconds:.3f}s")
    print_rule("[sim.tagline]Throughput Results[/]")
    print_markup(f"Total requests:                                                     {req_cnt}")
    print_markup(f"Total clocks (ns):                                                  {current}")
    print_markup(f"Total latency (s):                                                  {total_latency:.3f}")
    print_markup(f"Total input tokens:                                                 {total_prompt}")
    print_markup(f"Total generated tokens:                                             {total_gen}")
    print_markup(f"Request throughput (req/s):                                         {req_cnt/total_latency:.2f}")
    print_markup(f"Average prompt throughput (tok/s):                                  {total_prompt/total_latency:.2f}")
    print_markup(f"Average generation throughput (tok/s):                              {total_gen/total_latency:.2f}")
    print_markup(f"Total token throughput (tok/s):                                     {(total_prompt + total_gen)/total_latency:.2f}")
    print_markup(f"Throughput per {1/RATIO} sec (\\[prompt_throughput], \\[gen_throughput]): {throughput}")
    print_rule()
    if session_metrics_report is not None:
        session_throughput = session_metrics_report['throughput']
        completed_window_rate = session_throughput[
            'sessions_per_second_measurement_window']
        print_rule("[sim.tagline]Session Load Results[/]")
        print_markup(
            "Session arrival mode:                                               "
            f"{session_admission_summary['mode']}"
        )
        if session_admission_summary['mode'] == 'backlog':
            print_markup(
                "Maximum active sessions (K):                                     "
                f"{session_admission_summary['max_active_sessions']}"
            )
        elif session_admission_summary['mode'] == 'poisson':
            print_markup(
                "Offered session rate (sessions/s):                               "
                f"{session_admission_summary['session_arrival_rate_sps']:.6f}"
            )
        print_markup(
            "Completed sessions:                                                 "
            f"{session_throughput['completed_sessions']}"
        )
        print_markup(
            "Completed session throughput (sessions/s):                          "
            + (
                f"{completed_window_rate:.6f}"
                if completed_window_rate is not None else "N/A"
            )
        )
        print_markup(
            "Resume requests / all requests:                                     "
            f"{session_metrics_report['requests']['resume']['count']} / "
            f"{session_metrics_report['requests']['all']['count']}"
        )
        print_markup(
            "Detailed admission, resume-TTFT, TTFT, TPOT, and throughput:         "
            + (args.session_metrics or "use --session-metrics <path>")
        )
        print_rule()
    if any_prefix_caching:
        print_rule("[sim.tagline]Prefix Caching Results[/]")
        print_markup(f"Total requested prompt tokens:                                      {total_requested_tokens}")
        print_markup(f"NPU prefix hit prompt tokens:                                       {total_npu_hit_tokens}")
        if total_requested_tokens > 0:
            print_markup(f"NPU prefix hit ratio (%):                                           {(total_npu_hit_tokens/total_requested_tokens)*100:.2f}")
            if prefix_storage != "None":
                print_markup(f"{prefix_storage} prefix hit prompt tokens:                                       {total_cpu_hit_tokens}")
                print_markup(f"{prefix_storage} prefix hit ratio (%):                                           {(total_cpu_hit_tokens/total_requested_tokens)*100:.2f}")
            print_markup(f"Total prefix hit ratio (%):                                         {((total_npu_hit_tokens+total_cpu_hit_tokens)/total_requested_tokens)*100:.2f}")
        else:
            print_markup("NPU prefix hit ratio (%):                                           N/A (no requests tracked)")
        print_rule()
    if power_modeling:
        print_rule("[sim.tagline]Power Modeling Results[/]")
        total_energy = power_model.get_final_energy(current)
        print_markup(f"Total energy consumption (kJ):                                      {total_energy/1000:.2f}")
        # Each node results
        power_model.print_power_summary()
        print_markup(f"Power per {1/RATIO} sec (W): {power_model.power_time_series}")
        print_rule()
    if agentic_kv_manager is not None:
        kv_totals = agentic_kv_manager.metrics
        time_breakdown = agentic_kv_summary["time_breakdown"]
        wall_fraction = time_breakdown[
            "migration_restore_exposure_fraction_of_makespan"]
        request_fraction = time_breakdown[
            "migration_stall_fraction_of_total_request_latency"]
        recompute_token_fraction = time_breakdown["recompute_token_fraction"]
        recompute_compute_fraction = time_breakdown[
            "recompute_fraction_of_total_model_compute"]
        print_rule("[sim.tagline]Agentic Idle KV Results[/]")
        print_markup(f"Tool pauses:                                                        {kv_totals.tool_pauses}")
        print_markup(f"Reusable KV hit tokens:                                             {kv_totals.cache_hit_tokens}")
        print_markup(f"Model-executed declared-prefix tokens:                              {kv_totals.recompute_tokens}")
        print_markup(f"Policy-avoidable recomputed prefix tokens:                          {kv_totals.policy_avoidable_recompute_tokens}")
        print_markup(f"Aggregate request-blocking migration stall (ms):                    {kv_totals.critical_restore_ns / 1_000_000:.3f}")
        print_markup(f"  HBM capacity-admission wait (ms):                                 {kv_totals.critical_restore_hbm_admission_wait_ns / 1_000_000:.3f}")
        print_markup(f"  I/O queue wait (ms):                                              {kv_totals.critical_restore_queue_wait_ns / 1_000_000:.3f}")
        print_markup(f"  transfer service (ms):                                            {kv_totals.critical_restore_service_ns / 1_000_000:.3f}")
        print_markup(f"Aggregate active-work HBM reclaim wait (ms):                        {kv_totals.active_hbm_reclaim_wait_ns / 1_000_000:.3f}")
        print_markup(
            "Migration stall / aggregate request latency (%):                    "
            + (f"{request_fraction * 100:.3f}" if request_fraction is not None else "N/A")
        )
        print_markup(
            "Makespan exposure with at least one request blocked on migration (%): "
            + (f"{wall_fraction * 100:.3f}" if wall_fraction is not None else "N/A")
        )
        print_markup(
            "Exact migration-caused makespan penalty:                               "
            "N/A (requires a paired zero-migration run)"
        )
        print_markup(
            "Recomputed prefix tokens / all prompt tokens (%):                   "
            + (
                f"{recompute_token_fraction * 100:.3f}"
                if recompute_token_fraction is not None else "N/A"
            )
        )
        print_markup(
            "Recomputation / total model-compute time (%):                        "
            + (
                f"{recompute_compute_fraction * 100:.3f}"
                if recompute_compute_fraction is not None
                else "N/A (requires kernel-time attribution or a paired counterfactual run)"
            )
        )
        print_markup(f"SSD host writes (GB):                                               {kv_totals.ssd_host_write_bytes / 1_000_000_000:.3f}")
        print_markup(f"  cancelled partial writes included (GB):                           {kv_totals.ssd_cancelled_host_write_bytes / 1_000_000_000:.3f}")
        print_markup(f"SSD host reads (GB):                                                {kv_totals.ssd_host_read_bytes / 1_000_000_000:.3f}")
        print_rule()
    if full_model_hbf_runtime is not None:
        hbf_report = full_model_hbf_runtime.report()
        hbf_metrics = hbf_report["adapter"]["metrics"]
        print_rule("[sim.tagline]Full-model HBF Results[/]")
        print_markup(
            "HBF-completed requests:                                           "
            f"{hbf_report['completed_hbf_request_count']}")
        print_markup(
            "GPU / HBF completions:                                            "
            f"{hbf_metrics['gpu_completions']} / "
            f"{hbf_metrics['hbf_completions']}")
        print_markup(
            "HBF ASTRA callbacks:                                              "
            f"{hbf_metrics['astra_callbacks']}")
        print_markup(
            "HBF runtime report:                                               "
            + (
                args.full_model_hbf_metrics
                or "use --full-model-hbf-metrics <path>"
            ))
        print_rule()
    # Each instacne results
    for i in range(num_instances):
        print_rule(f"[sim.tagline]Instance \\[{i}][/]")
        schedulers[i].print_result()
        print_rule()
    
    # Important informations about metrics
    # The TTFT (Time to First Token) in our simulator differs from vllm. 
    # While vllm measures TTFT as the time when the client receives the first token,
    # Our simulator measures it as the time when the computation of the first token is completed.
    # Therefore, vllm gets much more higher TTFT.
    # (Ref: https://docs.vllm.ai/en/latest/design/metrics.html?utm_source=chatgpt.com#interval-calculations-vs-preemptions)

    if output_file != None:
        print(f"Saving each request's information to output file: {output_file}")
        for i, scheduler in enumerate(reporting_schedulers):
            Scheduler.save_output(
                scheduler, output_file, is_append=(i != 0))

    if agentic_kv_manager is not None and args.agentic_kv_metrics is not None:
        metrics_path = args.agentic_kv_metrics
        if not os.path.isabs(metrics_path):
            metrics_path = os.path.join('..', metrics_path)
        agentic_kv_manager.save_metrics(
            metrics_path, simulated_duration_ns=current, dataset=dataset,
            run_id=args.run_id,
            measurement_censored=measurement_early_stopped)

    if session_metrics_report is not None and args.session_metrics is not None:
        save_session_metrics(session_metrics_report, args.session_metrics)

    if (full_model_hbf_runtime is not None
            and args.full_model_hbf_metrics is not None):
        metrics_path = Path(args.full_model_hbf_metrics)
        if not metrics_path.is_absolute():
            metrics_path = Path("..") / metrics_path
        full_model_hbf_runtime.save_report(metrics_path)

    if args.cleanup_inputs:
        _cleanup_inputs_root(run_paths, logger)
    

if __name__ == "__main__": 
    # For simulation time breakdown
    # profiler = Profiler()
    # profiler.start()
    main()
    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))

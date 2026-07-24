"""Validated live runtime for the full-model HBF server.

The low-level full-model HBF modules deliberately remain usable in isolation
for latency, lifecycle, and scheduler tests.  This module is the strict
composition boundary used by ``python -m serving``.  It accepts only the
currently modeled production shape:

* one Qwen3-30B TP4 prefill instance;
* one layout-compatible TP4 decode instance on the same GPU server;
* one eight-card full-model HBF server attached through the shared ASTRA
  HBF-resource protocol;
* explicit GPU fallback recomputation when a resume arrives before HBF
  publication or when HBF capacity is unavailable.

The recomputation rule is intentional.  Completed GPU KV is decode-owned,
whereas a resumed suffix begins on the prefill instance.  Reusing that KV
would require a separately modeled D-to-P transfer and duplicate P-side HBM
admission.  The full-model HBF path does not silently borrow the legacy
SSD-tiering transport.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
    load_hbf_server_config,
)
from .hbf_full_model_lifecycle import (
    FullModelHBFLifecycle,
    PerGroupCapacityLedger,
)
from .hbf_full_model_pool import (
    FullModelHBFServingPool,
    derive_lpddr_workspace_bytes,
)
from .hbf_gpu_hbm_bridge import FullModelHBFGPUHBMBridge
from .hbf_online_adapter import (
    FullModelHBFOnlineAdapter,
    RouterCompletionProxy,
)


FULL_MODEL_HBF_RUNTIME_SCHEMA = "full-model-hbf-online-runtime-v2"
TARGET_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
TARGET_GPU_HARDWARE = "H100"
TARGET_GPU_LATENCY_MODEL = "h100-qwen3-tp4-kernel-calibrated"
TARGET_MAX_MODEL_LEN = 1_010_000


def _positive_int(name: str, value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _instance_id(instance: Mapping[str, Any]) -> int:
    value = instance.get("instance_id")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("every GPU instance requires a non-negative ID")
    return value


def _one_role(
        instances: Sequence[Mapping[str, Any]], role: str,
) -> Mapping[str, Any]:
    values = [
        instance for instance in instances
        if instance.get("pd_type") == role
    ]
    if len(values) != 1:
        raise ValueError(
            "full-model HBF requires exactly one "
            f"{role} GPU instance; observed={len(values)}")
    return values[0]


@dataclass(frozen=True)
class FullModelHBFRuntimeOptions:
    """Online scheduling and ASTRA projection knobs."""

    layout_key: str = "tp8_context"
    max_num_batched_tokens: int = 8_192
    max_num_seqs: int = 128
    max_prefill_chunk_tokens: int = 4_096
    prefill_drain_tail_tokens: int = 2_048
    prefill_drain_min_tokens: int = 4_096
    astra_chunk_bytes: int = 64 * 1024 ** 2
    latency_band: str = "central"
    server_id: int = 0

    def validate(self) -> None:
        HBFParallelLayout.for_key(self.layout_key)
        _positive_int(
            "max_num_batched_tokens",
            self.max_num_batched_tokens,
        )
        _positive_int("max_num_seqs", self.max_num_seqs)
        _positive_int(
            "max_prefill_chunk_tokens",
            self.max_prefill_chunk_tokens,
        )
        if self.max_prefill_chunk_tokens > self.max_num_batched_tokens:
            raise ValueError(
                "max_prefill_chunk_tokens exceeds the HBF token budget")
        for name, value in (
            (
                "prefill_drain_tail_tokens",
                self.prefill_drain_tail_tokens,
            ),
            (
                "prefill_drain_min_tokens",
                self.prefill_drain_min_tokens,
            ),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be a non-negative integer")
        _positive_int("astra_chunk_bytes", self.astra_chunk_bytes)
        if self.latency_band not in {"fast", "central", "slow"}:
            raise ValueError(
                "latency_band must be fast, central, or slow")
        if (
            isinstance(self.server_id, bool)
            or not isinstance(self.server_id, int)
            or self.server_id < 0
        ):
            raise ValueError("server_id must be a non-negative integer")


@dataclass
class FullModelHBFOnlineRuntime:
    """Live components and reporting state owned by the serving loop."""

    hardware: HBFServerHardware
    layout: HBFParallelLayout
    options: FullModelHBFRuntimeOptions
    lifecycle: FullModelHBFLifecycle
    pool: FullModelHBFServingPool
    adapter: FullModelHBFOnlineAdapter
    gpu_hbm_bridge: FullModelHBFGPUHBMBridge
    prefill_instance_id: int
    decode_instance_id: int
    model: str
    completed_requests: list[Any]

    def apply_pending_gpu_hbm_events(
            self) -> tuple[dict[str, object], ...]:
        """Apply every adapter ownership event before later admissions."""

        return self.gpu_hbm_bridge.apply_events(
            self.adapter.pop_gpu_hbm_events())

    def complete_native_gpu_request(
            self, request: object, *, completion_ns: int,
            publish_successor: bool = True) -> object:
        """Commit one GPU/P-D completion after same-time classification."""

        job = self.adapter.complete_native_gpu_request(
            request,
            completion_ns=int(completion_ns),
            publish_successor=publish_successor,
        )
        self.apply_pending_gpu_hbm_events()
        return job

    def complete_astra_dispatch(
            self, *, job_id: str, arrival_ns: int,
            completion_ns: int, stage_count: int) -> object:
        """Apply one strict HBF callback and its finite-HBM side effects."""

        result = self.adapter.complete_astra_dispatch(
            job_id=job_id,
            arrival_ns=int(arrival_ns),
            completion_ns=int(completion_ns),
            stage_count=int(stage_count),
            defer_turn_finalization=True,
        )
        self.apply_pending_gpu_hbm_events()
        return result

    def finalize_hbf_request(
            self, request: object, *, completion_ns: int,
            publish_successor: bool = True) -> object:
        """Commit one callback-completed HBF turn at the tie barrier."""

        result = self.adapter.finalize_deferred_hbf_completion(
            request,
            completion_ns=int(completion_ns),
            publish_successor=publish_successor,
        )
        self.apply_pending_gpu_hbm_events()
        return result

    def censor_completed_successor(
            self, request: object, *, now_ns: int) -> object:
        """End one frozen non-final lineage and release retained GPU KV."""

        result = self.adapter.censor_completed_successor(
            request, now_ns=int(now_ns))
        self.apply_pending_gpu_hbm_events()
        return result

    def censor_active_native_gpu_request(
            self, request: object, *, now_ns: int) -> None:
        self.gpu_hbm_bridge.cancel_pd_decode_reservation(request)
        self.adapter.censor_active_native_gpu_request(
            request, now_ns=int(now_ns))
        self.apply_pending_gpu_hbm_events()

    def censor_queued_native_gpu_request(
            self, request: object, *, now_ns: int) -> dict[str, object]:
        """End one queue-unwound native call without emitting HBM events."""

        result = self.adapter.censor_queued_native_gpu_request(
            request, now_ns=int(now_ns))
        self.apply_pending_gpu_hbm_events()
        return result

    def drain_astra_commands(self) -> tuple[str, ...]:
        return self.adapter.drain_astra_commands()

    def has_pending_astra_dispatches(self) -> bool:
        return self.adapter.has_pending_astra_dispatches()

    def has_pending_native_gpu_requests(self) -> bool:
        return self.adapter.has_pending_native_gpu_requests()

    def materialize_proxy(
            self, proxy: RouterCompletionProxy) -> Any:
        if not isinstance(proxy, RouterCompletionProxy):
            raise TypeError("proxy must be a RouterCompletionProxy")
        request = proxy.materialize_request(
            model=self.model,
            instance_id=self.decode_instance_id,
        )
        # This completed request never executed on the decode GPU. Preserve
        # a distinct reporting identity while using the decode ID only as the
        # validated native Request construction placeholder.
        request.instance_id = f"hbf:{self.options.server_id}"
        return request

    def reporting_schedulers(
            self, gpu_schedulers: Sequence[object],
    ) -> list[object]:
        """Return scheduler-like request sources for session metrics."""

        return [
            *gpu_schedulers,
            SimpleNamespace(
                pd_type="hbf",
                instance_id=f"hbf:{self.options.server_id}",
                done=self.completed_requests,
            ),
        ]

    def assert_quiescent(self) -> None:
        self.adapter.assert_invariants()
        self.gpu_hbm_bridge.assert_invariants()
        if self.adapter.has_pending():
            raise RuntimeError(
                "full-model HBF runtime still owns pending work")
        bridge = self.gpu_hbm_bridge.report()
        for key in (
                "idle_allocations",
                "pending_colocated_claims",
                "pending_pd_recompute_bindings",
                "pending_pd_decode_reservations"):
            if bridge[key]:
                raise RuntimeError(
                    "full-model HBF GPU bridge is not quiescent: "
                    f"{key}={bridge[key]}")

    def report(self) -> dict[str, object]:
        return {
            "schema": FULL_MODEL_HBF_RUNTIME_SCHEMA,
            "hardware": asdict(self.hardware),
            "layout": asdict(self.layout),
            "options": asdict(self.options),
            "gpu_pd_pair": [
                self.prefill_instance_id,
                self.decode_instance_id,
            ],
            "model": self.model,
            "completed_hbf_request_count": len(
                self.completed_requests),
            "completed_hbf_request_ids": [
                int(request.id) for request in self.completed_requests
            ],
            "adapter": self.adapter.report(),
            "gpu_hbm_bridge": self.gpu_hbm_bridge.report(),
        }

    def save_report(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output:
            json.dump(self.report(), output, indent=2)
            output.write("\n")


def load_full_model_hbf_hardware(
        config_path: Path, layout_key: str,
) -> tuple[HBFServerHardware, HBFParallelLayout]:
    """Load a strict config and reject aliases with changed semantics."""

    hardware, layouts = load_hbf_server_config(Path(config_path))
    try:
        layout = layouts[layout_key]
    except KeyError as exc:
        raise ValueError(
            f"HBF layout {layout_key!r} is absent from {config_path}; "
            f"available={sorted(layouts)}"
        ) from exc
    canonical = HBFParallelLayout.for_key(layout_key)
    if layout != canonical:
        raise ValueError(
            "HBF layout key changed its canonical TP/replica meaning: "
            f"configured={layout}, canonical={canonical}")
    return hardware, layout


def validate_full_model_hbf_gpu_cluster(
        instances: Sequence[Mapping[str, Any]],
        runtime_configs: Sequence[Mapping[str, Any]],
        inst2node_mapping: Mapping[int, int], *,
        network_backend: str,
) -> tuple[int, int]:
    """Return the only valid P/D pair or fail before ASTRA starts."""

    if network_backend != "analytical-congestion-aware":
        raise ValueError(
            "full-model HBF requires --network-backend "
            "analytical-congestion-aware")
    if len(instances) != 2 or len(runtime_configs) != 2:
        raise ValueError(
            "full-model HBF currently requires one P4 and one D4 "
            "instance on one GPU server")
    prefill = _one_role(instances, "prefill")
    decode = _one_role(instances, "decode")
    prefill_id = _instance_id(prefill)
    decode_id = _instance_id(decode)
    if set(inst2node_mapping) != {prefill_id, decode_id}:
        raise ValueError(
            "full-model HBF instance/node mapping is incomplete")
    if inst2node_mapping[prefill_id] != inst2node_mapping[decode_id]:
        raise ValueError(
            "full-model HBF P4/D4 instances must share one GPU server")

    for instance in (prefill, decode):
        instance_id = _instance_id(instance)
        runtime = runtime_configs[instance_id]
        expected = {
            "model_name": TARGET_MODEL,
            "hardware": TARGET_GPU_HARDWARE,
            "num_npus": 4,
            "tp_size": 4,
            "pp_size": 1,
        }
        for field, value in expected.items():
            if instance.get(field) != value:
                raise ValueError(
                    "full-model HBF GPU instance differs from the "
                    f"calibrated P4/D4 contract: instance={instance_id}, "
                    f"field={field}, expected={value!r}, "
                    f"observed={instance.get(field)!r}")
        if runtime.get("dtype") != "bfloat16":
            raise ValueError(
                "full-model HBF GPU reference requires bfloat16")
        if runtime.get("kv_cache_dtype") != "auto":
            raise ValueError(
                "full-model HBF currently models BF16 GPU/HBF KV only")
        if int(runtime.get("block_size", 0)) != 16:
            raise ValueError(
                "full-model HBF finite-HBM bridge requires 16-token "
                "GPU KV blocks")
        if bool(runtime.get("enable_prefix_caching")):
            raise ValueError(
                "full-model HBF cannot share ownership with generic "
                "prefix caching")
        if int(runtime.get("max_model_len", 0)) < TARGET_MAX_MODEL_LEN:
            raise ValueError(
                "full-model HBF requires a 1,010,000-token GPU context")
        if runtime.get("latency_model") != TARGET_GPU_LATENCY_MODEL:
            raise ValueError(
                "full-model HBF GPU reference requires the calibrated "
                f"latency model {TARGET_GPU_LATENCY_MODEL!r}")
    return prefill_id, decode_id


def build_full_model_hbf_online_runtime(
        *, repo_root: Path, config_path: Path,
        options: FullModelHBFRuntimeOptions,
        instances: Sequence[Mapping[str, Any]],
        runtime_configs: Sequence[Mapping[str, Any]],
        inst2node_mapping: Mapping[int, int],
        schedulers: Sequence[object],
        network_backend: str,
) -> FullModelHBFOnlineRuntime:
    """Build one shared-ledger HBF lifecycle, pool, adapter, and HBM bridge."""

    options.validate()
    hardware, layout = load_full_model_hbf_hardware(
        config_path, options.layout_key)
    prefill_id, decode_id = validate_full_model_hbf_gpu_cluster(
        instances,
        runtime_configs,
        inst2node_mapping,
        network_backend=network_backend,
    )
    schedulers_by_instance = {
        int(scheduler.instance_id): scheduler
        for scheduler in schedulers
    }
    if set(schedulers_by_instance) != {prefill_id, decode_id}:
        raise ValueError(
            "full-model HBF runtime received the wrong GPU schedulers")

    workspace_bytes = derive_lpddr_workspace_bytes(
        layout,
        max_num_batched_tokens=options.max_num_batched_tokens,
        max_num_seqs=options.max_num_seqs,
    )
    kv_capacity_bytes = (
        hardware.lpddr_capacity_bytes_per_card - workspace_bytes)
    if kv_capacity_bytes <= 0:
        raise ValueError(
            "HBF LPDDR has no active-KV capacity after workspace")
    cards_by_group = {
        group_id: tuple(range(
            group_id * layout.tp_size,
            (group_id + 1) * layout.tp_size,
        ))
        for group_id in range(layout.replicas)
    }
    ledger = PerGroupCapacityLedger(
        group_count=layout.replicas,
        capacity_bytes=kv_capacity_bytes,
        card_ids_by_group=cards_by_group,
    )
    lifecycle = FullModelHBFLifecycle(
        hardware=hardware,
        layout=layout,
        lpddr_ledger=ledger,
        execution_backend="external_astra",
        server_id=options.server_id,
        astra_chunk_bytes=options.astra_chunk_bytes,
    )
    pool = FullModelHBFServingPool(
        repo_root=Path(repo_root),
        hardware=hardware,
        layout=layout,
        lpddr_ledger=ledger,
        placement_resolver=lifecycle.placement_snapshot,
        max_num_batched_tokens=options.max_num_batched_tokens,
        max_num_seqs=options.max_num_seqs,
        max_prefill_chunk_tokens=options.max_prefill_chunk_tokens,
        prefill_drain_tail_tokens=(
            options.prefill_drain_tail_tokens),
        prefill_drain_min_tokens=(
            options.prefill_drain_min_tokens),
        band=options.latency_band,
        retain_detailed_history=False,
        retain_token_completion_history=True,
        execution_backend="external_astra",
        server_id=options.server_id,
    )
    prefill_scheduler = schedulers_by_instance[prefill_id]
    adapter = FullModelHBFOnlineAdapter(
        lifecycle=lifecycle,
        pool=pool,
        gpu_tp_size=int(prefill_scheduler.tp_size),
        gpu_block_size_tokens=int(prefill_scheduler.block_size),
        gpu_resume_mode="recompute",
    )
    bridge = FullModelHBFGPUHBMBridge(
        schedulers_by_instance,
        pd_pairs=((prefill_id, decode_id),),
        fallback_reuse_mode="recompute",
        adapter=adapter,
    )
    return FullModelHBFOnlineRuntime(
        hardware=hardware,
        layout=layout,
        options=options,
        lifecycle=lifecycle,
        pool=pool,
        adapter=adapter,
        gpu_hbm_bridge=bridge,
        prefill_instance_id=prefill_id,
        decode_instance_id=decode_id,
        model=str(prefill_scheduler.model),
        completed_requests=[],
    )


__all__ = [
    "FULL_MODEL_HBF_RUNTIME_SCHEMA",
    "FullModelHBFOnlineRuntime",
    "FullModelHBFRuntimeOptions",
    "TARGET_GPU_HARDWARE",
    "TARGET_GPU_LATENCY_MODEL",
    "TARGET_MAX_MODEL_LEN",
    "TARGET_MODEL",
    "build_full_model_hbf_online_runtime",
    "load_full_model_hbf_hardware",
    "validate_full_model_hbf_gpu_cluster",
]

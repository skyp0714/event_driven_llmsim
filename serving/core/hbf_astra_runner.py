"""Run a validated HBF text-trace artifact through Chakra and ASTRA-Sim.

This module is deliberately independent of ``serving.__main__``.  A
full-model projector can emit the normal LLMServingSim text-trace format and
hand the resulting artifact to :func:`run_hbf_trace_artifact`.  The runner
then uses the repository's real Chakra converter and congestion-aware
analytical ASTRA-Sim binary; it never substitutes a Python latency estimate
for an ASTRA completion cycle.

The current input protocol is intentionally small and structural.  Any object
with ``text`` and ``num_npus`` attributes is accepted, including the existing
HBF conformance artifacts.  A future full-model projection type therefore
does not need an adapter or an import dependency on this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import selectors
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from .astra_operation_conformance import (
    ParsedMicrotrace,
    TRACE_HEADER,
    parse_microtrace,
)
from .controller import Controller


SCHEMA_VERSION = 1
DEFAULT_ASTRA_BINARY = Path(
    "astra-sim/build/astra_analytical/build/bin/"
    "AstraSim_Analytical_Congestion_Aware"
)
DEFAULT_CHAKRA_ROOT = Path(
    "astra-sim/extern/graph_frontend/chakra"
)
_CYCLE_RECORD = re.compile(
    r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, "
    r"exposed communication (\d+) cycles\."
)
_HBF_BACKGROUND_COMPLETE = re.compile(
    r"HBF background complete\t([^\t\n]+)\t(\d+)\t(\d+)\t(\d+)"
)
_HBF_BACKGROUND_CAPABILITY = (
    "Analytical control capability\thbf-background-v1"
)
_ENDPOINT_PARK_CAPABILITY = (
    "Analytical control capability\tendpoint-park-v1"
)
_SUPPORTED_TOPOLOGIES = frozenset({
    "Ring",
    "FullyConnected",
})


class HBFAstraRunnerError(RuntimeError):
    """Raised when validation, conversion, or actual ASTRA execution fails."""


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_float(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _nonnegative_float(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be non-negative and finite")
    return float(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True)
class HBFTextTraceArtifact:
    """A generic validated-input candidate for the actual ASTRA runner."""

    text: str
    num_npus: int
    label: str = "hbf-text-trace"


@dataclass(frozen=True)
class HBFTraceAudit:
    """Pure validation summary retained after temporary traces are removed."""

    label: str
    trace_sha256: str
    trace_bytes: int
    num_npus: int
    model_parallel_groups: int
    npus_per_group: int
    row_count: int
    hbf_descriptor_count: int
    hbf_stage_count: int
    hbf_card_ids: tuple[int, ...]
    hbf_resource_names: tuple[str, ...]
    descriptor_validation: str = "strict_preflight_and_chakra_converter"


@dataclass(frozen=True)
class AstraCycleRecord:
    """One endpoint's ASTRA iteration-completion record."""

    sys: int
    iteration: int
    total_cycles: int
    exposed_communication_cycles: int


@dataclass(frozen=True)
class AstraHBFRunConfig:
    """Topology and memory inputs for one actual ASTRA execution.

    Empty topology fields are inferred from the trace.  One model-parallel
    group becomes a one-dimensional Ring.  Multiple groups become
    ``(npus_per_group, groups)`` FullyConnected dimensions because the
    congestion-aware ASTRA backend requires FullyConnected on every axis of a
    multi-dimensional topology.
    """

    dimensions: tuple[int, ...] = ()
    topology: tuple[str, ...] = ()
    link_bandwidth_gbps: tuple[float, ...] = ()
    link_latency_ns: tuple[float, ...] = ()
    local_mem_bandwidth_gbps: float = 3_350.0
    remote_mem_bandwidth_gbps: float = 256.0
    remote_mem_latency_ns: int = 0
    hbf_num_devices: Optional[int] = None
    timeout_seconds: float = 60.0


@dataclass(frozen=True)
class ResolvedAstraHBFRunConfig:
    dimensions: tuple[int, ...]
    topology: tuple[str, ...]
    link_bandwidth_gbps: tuple[float, ...]
    link_latency_ns: tuple[float, ...]
    local_mem_bandwidth_gbps: float
    remote_mem_bandwidth_gbps: float
    remote_mem_latency_ns: int
    hbf_num_devices: int
    timeout_seconds: float


@dataclass(frozen=True)
class AstraHBFRunResult:
    """Auditable evidence that the submitted graph used actual ASTRA cycles."""

    final_cycles: int
    endpoint_cycles: tuple[AstraCycleRecord, ...]
    trace: HBFTraceAudit
    resolved_config: ResolvedAstraHBFRunConfig
    config_sha256: Mapping[str, str]
    graph_sha256_by_rank: Mapping[int, str]
    binary_path: str
    binary_sha256: str
    protobuf_runtime_version: str
    stdout_sha256: str
    stderr_sha256: str
    schema_version: int = SCHEMA_VERSION
    backend: str = (
        "llm_text_trace->chakra_et->"
        "AstraSim_Analytical_Congestion_Aware"
    )
    cycle_unit: str = "astra_analytical_nanosecond_tick"
    claim_scope: str = "submitted_hbf_trace_artifact"

    @property
    def astra_cycles_used(self) -> bool:
        return True

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "astra_cycles_used": True,
            "analytical_cycle_substitution": False,
            "cycle_unit": self.cycle_unit,
            "claim_scope": self.claim_scope,
            "final_cycles": self.final_cycles,
            "endpoint_cycles": [
                asdict(record) for record in self.endpoint_cycles
            ],
            "trace": asdict(self.trace),
            "resolved_config": asdict(self.resolved_config),
            "config_sha256": dict(self.config_sha256),
            "graph_sha256_by_rank": {
                str(rank): digest
                for rank, digest in sorted(
                    self.graph_sha256_by_rank.items())
            },
            "binary": {
                "path": self.binary_path,
                "sha256": self.binary_sha256,
            },
            "protobuf_runtime_version": self.protobuf_runtime_version,
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
        }


@dataclass(frozen=True)
class HBFBackgroundJob:
    """One full-model or lifecycle DAG submitted to persistent ASTRA."""

    job_id: str
    arrival_ns: int
    stages: tuple[Mapping[str, object], ...]
    projection_schema: str = "generic-hbf-background-v1"

    @classmethod
    def from_projection(
            cls, *, job_id: str, arrival_ns: int,
            projection: object) -> "HBFBackgroundJob":
        """Normalize an ``HBFModelAstraProjection`` without importing it."""

        try:
            stages = projection.controller_stages()
        except (AttributeError, TypeError) as exc:
            raise TypeError(
                "projection must expose controller_stages()") from exc
        schema = getattr(
            projection, "schema", type(projection).__name__)
        return cls(
            job_id=job_id,
            arrival_ns=arrival_ns,
            stages=tuple(stages),
            projection_schema=str(schema),
        )

    def command(self) -> str:
        return Controller.hbf_background_command(
            self.job_id, self.arrival_ns, self.stages)

    @property
    def descriptor_sha256(self) -> str:
        descriptor = self.command().split("\t", 3)[3]
        return _sha256_bytes(descriptor.encode("utf-8"))


@dataclass(frozen=True)
class HBFBackgroundCompletion:
    """An exact callback emitted by ASTRA's shared-resource event queue."""

    job_id: str
    arrival_ns: int
    completion_ns: int
    stage_count: int
    descriptor_sha256: str
    projection_schema: str
    backend: str = (
        "AstraSim_Analytical_Congestion_Aware/"
        "hbf-background-v1"
    )

    @property
    def astra_cycles_used(self) -> bool:
        return True

    @property
    def elapsed_cycles(self) -> int:
        return self.completion_ns - self.arrival_ns

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["astra_cycles_used"] = True
        value["analytical_cycle_substitution"] = False
        value["elapsed_cycles"] = self.elapsed_cycles
        value["cycle_unit"] = "astra_analytical_nanosecond_tick"
        return value


@dataclass(frozen=True)
class PersistentHBFSessionAudit:
    """Immutable audit returned when a persistent ASTRA session closes."""

    completions: tuple[HBFBackgroundCompletion, ...]
    resolved_config: ResolvedAstraHBFRunConfig
    config_sha256: Mapping[str, str]
    binary_path: str
    binary_sha256: str
    protobuf_runtime_version: str
    bootstrap_graph_sha256_by_rank: Mapping[int, str]
    stdout_sha256: str
    clean_exit: bool
    schema_version: int = SCHEMA_VERSION

    @property
    def astra_cycles_used(self) -> bool:
        return True

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "backend": (
                "persistent/"
                "AstraSim_Analytical_Congestion_Aware/"
                "hbf-background-v1"
            ),
            "astra_cycles_used": True,
            "analytical_cycle_substitution": False,
            "cycle_unit": "astra_analytical_nanosecond_tick",
            "completions": [
                completion.as_dict()
                for completion in self.completions
            ],
            "resolved_config": asdict(self.resolved_config),
            "config_sha256": dict(self.config_sha256),
            "binary": {
                "path": self.binary_path,
                "sha256": self.binary_sha256,
            },
            "protobuf_runtime_version": self.protobuf_runtime_version,
            "bootstrap_graph_sha256_by_rank": {
                str(rank): digest for rank, digest in sorted(
                    self.bootstrap_graph_sha256_by_rank.items())
            },
            "stdout_sha256": self.stdout_sha256,
            "clean_exit": self.clean_exit,
        }


def _artifact_fields(artifact: object) -> tuple[str, int, str]:
    try:
        text = getattr(artifact, "text")
        num_npus = getattr(artifact, "num_npus")
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            "HBF trace artifact must expose text and num_npus attributes"
        ) from exc
    if not isinstance(text, str) or not text:
        raise ValueError("artifact.text must be a non-empty string")
    num_npus = _positive_int("artifact.num_npus", num_npus)
    label = getattr(artifact, "label", type(artifact).__name__)
    if (
        not isinstance(label, str)
        or not label
        or any(character in label for character in "\r\n")
    ):
        raise ValueError("artifact label must be a non-empty single line")
    return text, num_npus, label


def _validate_identifier(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or ";" in value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(
            f"{name} must be a non-empty whitespace-free identifier")
    return value


def _validate_hbf_descriptor(
        descriptor: object, *, npus_per_group: int,
) -> tuple[int, tuple[str, ...]]:
    if not isinstance(descriptor, dict):
        raise ValueError("HBF descriptor must be an object")
    expected_fields = {
        "v",
        "expected_participants",
        "card_id",
        "gang_base",
        "merge_ns",
        "stages",
        "terminals",
    }
    if set(descriptor) != expected_fields:
        raise ValueError(
            "HBF descriptor fields must match exactly: "
            f"expected={sorted(expected_fields)}, "
            f"observed={sorted(descriptor)}"
        )
    version = descriptor["v"]
    if isinstance(version, bool) or version != 1:
        raise ValueError("HBF descriptor version must be integer 1")
    participants = _positive_int(
        "HBF expected_participants",
        descriptor["expected_participants"],
    )
    if participants != npus_per_group:
        raise ValueError(
            "HBF expected_participants must equal the owning group size: "
            f"descriptor={participants}, group={npus_per_group}"
        )
    card_id = _nonnegative_int("HBF card_id", descriptor["card_id"])
    _validate_identifier("HBF gang_base", descriptor["gang_base"])
    _nonnegative_int("HBF merge_ns", descriptor["merge_ns"])

    stages = descriptor["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("HBF stages must be a non-empty list")
    stage_ids: set[str] = set()
    parents_by_id: dict[str, tuple[str, ...]] = {}
    resources_seen: set[str] = set()
    stage_fields = {
        "id", "runtime_ns", "tensor_bytes", "resources", "deps",
    }
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != stage_fields:
            raise ValueError(
                "HBF stage fields must match exactly: "
                f"expected={sorted(stage_fields)}"
            )
        stage_id = _validate_identifier("HBF stage id", stage["id"])
        if stage_id in stage_ids:
            raise ValueError(f"duplicate HBF stage id {stage_id!r}")
        _positive_int("HBF stage runtime_ns", stage["runtime_ns"])
        _nonnegative_int("HBF stage tensor_bytes", stage["tensor_bytes"])
        resources = stage["resources"]
        if (
            not isinstance(resources, list)
            or not resources
            or len(resources) != len(set(resources))
        ):
            raise ValueError(
                "HBF stage resources must be a non-empty unique list")
        normalized_resources = tuple(
            _validate_identifier("HBF resource", resource)
            for resource in resources
        )
        deps = stage["deps"]
        if (
            not isinstance(deps, list)
            or len(deps) != len(set(deps))
            or any(
                not isinstance(parent, str) or parent not in stage_ids
                for parent in deps
            )
        ):
            raise ValueError(
                "HBF stage deps must uniquely reference earlier stages")
        stage_ids.add(stage_id)
        parents_by_id[stage_id] = tuple(deps)
        resources_seen.update(normalized_resources)

    terminals = descriptor["terminals"]
    if (
        not isinstance(terminals, list)
        or not terminals
        or len(terminals) != len(set(terminals))
        or any(
            not isinstance(terminal, str) or terminal not in stage_ids
            for terminal in terminals
        )
    ):
        raise ValueError(
            "HBF terminals must uniquely reference emitted stages")

    terminal_ancestors: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in terminal_ancestors:
            return
        terminal_ancestors.add(stage_id)
        for parent in parents_by_id[stage_id]:
            visit(parent)

    for terminal in terminals:
        visit(terminal)
    if terminal_ancestors != stage_ids:
        raise ValueError("every HBF stage must feed a declared terminal")
    return card_id, tuple(sorted(resources_seen))


def validate_hbf_trace_artifact(
        artifact: object,
) -> tuple[HBFTextTraceArtifact, ParsedMicrotrace, HBFTraceAudit]:
    """Validate a duck-typed HBF trace artifact without loading Chakra."""

    text, num_npus, label = _artifact_fields(artifact)
    parsed = parse_microtrace(text)
    groups = parsed.model_parallel_groups
    if num_npus % groups:
        raise ValueError(
            "artifact.num_npus must be divisible by model_parallel_groups")
    npus_per_group = num_npus // groups

    descriptor_count = 0
    stage_count = 0
    card_ids: set[int] = set()
    resource_names: set[str] = set()
    gang_bases: set[str] = set()
    for row in parsed.rows:
        if not row.misc.startswith("{"):
            continue
        try:
            metadata = json.loads(row.misc)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid misc JSON on trace row {row.name!r}") from exc
        if not isinstance(metadata, dict):
            raise ValueError("trace row misc JSON must be an object")
        if set(metadata) - {"batch", "hbf"}:
            raise ValueError(
                f"unknown misc metadata on trace row {row.name!r}")
        descriptor = metadata.get("hbf")
        if descriptor is None:
            continue
        card_id, resources = _validate_hbf_descriptor(
            descriptor, npus_per_group=npus_per_group)
        gang_base = descriptor["gang_base"]
        if gang_base in gang_bases:
            raise ValueError(
                f"HBF gang_base must be unique in one trace: {gang_base!r}")
        gang_bases.add(gang_base)
        descriptor_count += 1
        stage_count += len(descriptor["stages"])
        card_ids.add(card_id)
        resource_names.update(resources)
    if descriptor_count == 0:
        raise ValueError(
            "actual HBF ASTRA execution requires at least one HBF descriptor")

    normalized = HBFTextTraceArtifact(
        text=text,
        num_npus=num_npus,
        label=label,
    )
    audit = HBFTraceAudit(
        label=label,
        trace_sha256=_sha256_bytes(text.encode("utf-8")),
        trace_bytes=len(text.encode("utf-8")),
        num_npus=num_npus,
        model_parallel_groups=groups,
        npus_per_group=npus_per_group,
        row_count=len(parsed.rows),
        hbf_descriptor_count=descriptor_count,
        hbf_stage_count=stage_count,
        hbf_card_ids=tuple(sorted(card_ids)),
        hbf_resource_names=tuple(sorted(resource_names)),
    )
    return normalized, parsed, audit


def _expand_axis(
        name: str,
        configured: tuple[float, ...],
        count: int,
        default: float,
        *,
        allow_zero: bool,
) -> tuple[float, ...]:
    values = configured or (default,) * count
    if len(values) != count:
        raise ValueError(
            f"{name} must contain one value per topology dimension")
    validator = _nonnegative_float if allow_zero else _positive_float
    return tuple(
        validator(f"{name}[{index}]", value)
        for index, value in enumerate(values)
    )


def resolve_run_config(
        config: AstraHBFRunConfig,
        *, parsed: ParsedMicrotrace,
        audit: HBFTraceAudit,
) -> ResolvedAstraHBFRunConfig:
    """Resolve and validate trace-dependent ASTRA configuration."""

    if not isinstance(config, AstraHBFRunConfig):
        raise TypeError("config must be an AstraHBFRunConfig")
    if config.dimensions:
        dimensions = tuple(
            _positive_int(f"dimensions[{index}]", value)
            for index, value in enumerate(config.dimensions)
        )
    elif parsed.model_parallel_groups == 1:
        dimensions = (audit.num_npus,)
    else:
        dimensions = (
            audit.npus_per_group,
            parsed.model_parallel_groups,
        )
    if math.prod(dimensions) != audit.num_npus:
        raise ValueError(
            "topology dimensions must multiply to artifact.num_npus")

    if config.topology:
        topology = config.topology
    elif len(dimensions) == 1:
        topology = ("Ring",)
    else:
        topology = ("FullyConnected",) * len(dimensions)
    if (
        len(topology) != len(dimensions)
        or any(value not in _SUPPORTED_TOPOLOGIES for value in topology)
    ):
        raise ValueError(
            "topology must contain Ring or FullyConnected for every "
            "dimension")
    if len(dimensions) > 1 and any(
            value != "FullyConnected" for value in topology):
        raise ValueError(
            "congestion-aware ASTRA requires FullyConnected on every "
            "multi-dimensional topology axis")

    bandwidth = _expand_axis(
        "link_bandwidth_gbps",
        config.link_bandwidth_gbps,
        len(dimensions),
        50.0,
        allow_zero=False,
    )
    latency = _expand_axis(
        "link_latency_ns",
        config.link_latency_ns,
        len(dimensions),
        1.0,
        allow_zero=True,
    )
    hbf_devices = (
        max(audit.hbf_card_ids) + 1
        if config.hbf_num_devices is None else
        _positive_int("hbf_num_devices", config.hbf_num_devices)
    )
    if hbf_devices <= max(audit.hbf_card_ids):
        raise ValueError(
            "hbf_num_devices must cover every descriptor card_id")
    return ResolvedAstraHBFRunConfig(
        dimensions=dimensions,
        topology=tuple(topology),
        link_bandwidth_gbps=bandwidth,
        link_latency_ns=latency,
        local_mem_bandwidth_gbps=_positive_float(
            "local_mem_bandwidth_gbps",
            config.local_mem_bandwidth_gbps,
        ),
        remote_mem_bandwidth_gbps=_positive_float(
            "remote_mem_bandwidth_gbps",
            config.remote_mem_bandwidth_gbps,
        ),
        remote_mem_latency_ns=_nonnegative_int(
            "remote_mem_latency_ns",
            config.remote_mem_latency_ns,
        ),
        hbf_num_devices=hbf_devices,
        timeout_seconds=_positive_float(
            "timeout_seconds", config.timeout_seconds),
    )


def _network_yaml(config: ResolvedAstraHBFRunConfig) -> str:
    return (
        f"topology: [ {', '.join(config.topology)} ]\n"
        f"npus_count: [ "
        f"{', '.join(map(str, config.dimensions))} ]\n"
        f"bandwidth: [ "
        f"{', '.join(format(value, '.17g') for value in config.link_bandwidth_gbps)} ]\n"
        f"latency: [ "
        f"{', '.join(format(value, '.17g') for value in config.link_latency_ns)} ]\n"
    )


def _write_configs(
        root: Path, config: ResolvedAstraHBFRunConfig,
) -> tuple[dict[str, Path], dict[str, str]]:
    dimension_count = len(config.dimensions)
    collective = ["ring"] * dimension_count
    network_text = _network_yaml(config)
    system_text = _strict_json({
        "scheduling-policy": "LIFO",
        "endpoint-delay": 0,
        "active-chunks-per-dimension": 1,
        "preferred-dataset-splits": 1,
        "all-reduce-implementation": collective,
        "all-gather-implementation": collective,
        "reduce-scatter-implementation": collective,
        "all-to-all-implementation": collective,
        "collective-optimization": "localBWAware",
        "local-mem-bw": config.local_mem_bandwidth_gbps,
        "boost-mode": 0,
    })
    memory_text = _strict_json({
        "remote_mem": {
            "memory-type": "PER_NODE_MEMORY_EXPANSION",
            "mem-bw": config.remote_mem_bandwidth_gbps,
            "mem-latency": config.remote_mem_latency_ns,
            "num-devices": 1,
        },
        # Presence of this object selects SharedResourceMemory.  Runtime and
        # contention are carried by the validated per-stage Chakra metadata.
        "hbf_mem": {
            "memory-type": "MEMORY_POOL",
            "num-devices": config.hbf_num_devices,
        },
    })
    paths = {
        "network": root / "network.yml",
        "system": root / "system.json",
        "memory": root / "memory.json",
    }
    encoded = {
        "network": network_text,
        "system": system_text,
        "memory": memory_text,
    }
    for name, path in paths.items():
        path.write_text(encoded[name], encoding="utf-8")
    return paths, {
        name: _sha256_bytes(value.encode("utf-8"))
        for name, value in encoded.items()
    }


def _load_converter(chakra_root: Path) -> tuple[Any, str]:
    roots = (
        chakra_root / "build" / "lib",
        chakra_root,
    )
    for candidate in reversed(roots):
        candidate_text = str(candidate)
        if candidate.is_dir() and candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)
    try:
        import google.protobuf
        from chakra.src.converter.llm_converter import LLMConverter
    except Exception as exc:
        raise HBFAstraRunnerError(
            "Chakra converter/protobuf is unavailable; build Chakra and use "
            "a protobuf runtime compatible with its generated bindings: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return LLMConverter, str(google.protobuf.__version__)


def _parse_cycle_records(output: str) -> tuple[AstraCycleRecord, ...]:
    records = tuple(
        AstraCycleRecord(
            sys=int(system_id),
            iteration=int(iteration),
            total_cycles=int(total_cycles),
            exposed_communication_cycles=int(communication_cycles),
        )
        for (
            system_id,
            iteration,
            total_cycles,
            communication_cycles,
        ) in _CYCLE_RECORD.findall(output)
    )
    if not records:
        raise HBFAstraRunnerError(
            "ASTRA output contained no iteration completion records")
    return records


def _final_endpoint_cycles(
        records: tuple[AstraCycleRecord, ...],
        *, num_npus: int,
        expected_system_ids: Optional[tuple[int, ...]] = None,
) -> tuple[AstraCycleRecord, ...]:
    expected_ids = (
        tuple(range(num_npus))
        if expected_system_ids is None else
        expected_system_ids
    )
    if (
        not expected_ids
        or len(set(expected_ids)) != len(expected_ids)
        or any(
            isinstance(system_id, bool)
            or not isinstance(system_id, int)
            or system_id < 0
            or system_id >= num_npus
            for system_id in expected_ids
        )
    ):
        raise ValueError(
            "expected_system_ids must be unique in-range endpoints")
    iteration_zero = tuple(
        record for record in records if record.iteration == 0)
    by_rank: dict[int, AstraCycleRecord] = {}
    for record in iteration_zero:
        if record.sys < 0 or record.sys >= num_npus:
            raise HBFAstraRunnerError(
                f"ASTRA reported out-of-range endpoint {record.sys}")
        if record.sys in by_rank:
            raise HBFAstraRunnerError(
                "ASTRA reported duplicate iteration-zero completion for "
                f"endpoint {record.sys}")
        if record.exposed_communication_cycles > record.total_cycles:
            raise HBFAstraRunnerError(
                "ASTRA exposed communication cycles exceed total cycles")
        by_rank[record.sys] = record
    expected = set(expected_ids)
    observed_expected = set(by_rank) & expected
    if observed_expected != expected:
        raise HBFAstraRunnerError(
            "ASTRA did not report every required endpoint for iteration zero: "
            f"missing={sorted(expected - observed_expected)}"
        )
    return tuple(by_rank[rank] for rank in expected_ids)


def run_hbf_trace_artifact(
        artifact: object,
        *,
        config: AstraHBFRunConfig = AstraHBFRunConfig(),
        repo_root: Optional[Path] = None,
        binary_path: Optional[Path] = None,
        chakra_root: Optional[Path] = None,
) -> AstraHBFRunResult:
    """Convert and execute one HBF trace, returning actual ASTRA cycles.

    ``artifact`` is normalized and strictly validated before any files are
    generated.  Success requires the configured final group endpoint's
    iteration-zero completion record and ASTRA's explicit
    all-requests-exited marker.  ASTRA's endpoint barrier opens that endpoint
    only after every rank in the model-parallel group has completed.
    """

    normalized, parsed, trace_audit = validate_hbf_trace_artifact(artifact)
    resolved = resolve_run_config(
        config, parsed=parsed, audit=trace_audit)
    root = (
        Path(__file__).resolve().parents[2]
        if repo_root is None else Path(repo_root).resolve()
    )
    binary = (
        root / DEFAULT_ASTRA_BINARY
        if binary_path is None else Path(binary_path).resolve()
    )
    chakra = (
        root / DEFAULT_CHAKRA_ROOT
        if chakra_root is None else Path(chakra_root).resolve()
    )
    if not binary.is_file():
        raise HBFAstraRunnerError(
            f"congestion-aware ASTRA binary is missing: {binary}; "
            "run scripts/compile.sh")
    if not chakra.is_dir():
        raise HBFAstraRunnerError(
            f"Chakra source/build root is missing: {chakra}")

    converter, protobuf_version = _load_converter(chakra)
    with tempfile.TemporaryDirectory(
            prefix="llmservingsim-hbf-astra-") as temporary:
        work = Path(temporary)
        trace_path = work / "trace.txt"
        graph_prefix = work / "graph"
        trace_path.write_text(normalized.text, encoding="utf-8")
        try:
            converter(
                str(trace_path),
                str(graph_prefix),
                num_npus=normalized.num_npus,
            ).convert()
        except Exception as exc:
            raise HBFAstraRunnerError(
                "Chakra conversion rejected the validated HBF trace: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        graph_paths = {
            rank: Path(f"{graph_prefix}.{rank}.et")
            for rank in range(normalized.num_npus)
        }
        missing_graphs = [
            str(path) for path in graph_paths.values()
            if not path.is_file()
        ]
        if missing_graphs:
            raise HBFAstraRunnerError(
                "Chakra conversion did not emit every endpoint graph: "
                f"{missing_graphs}")
        graph_hashes = {
            rank: _sha256_file(path)
            for rank, path in graph_paths.items()
        }
        paths, config_hashes = _write_configs(work, resolved)
        command = [
            str(binary),
            f"--workload-configuration={graph_prefix}",
            f"--system-configuration={paths['system']}",
            f"--network-configuration={paths['network']}",
            f"--memory-configuration={paths['memory']}",
            "--start-npu-ids=0",
            f"--end-npu-ids={normalized.num_npus - 1}",
        ]
        try:
            process = subprocess.run(
                command,
                input="exit\n",
                text=True,
                capture_output=True,
                timeout=resolved.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise HBFAstraRunnerError(
                f"ASTRA execution could not complete: {exc}") from exc

    if process.returncode:
        raise HBFAstraRunnerError(
            f"ASTRA rejected the HBF graph (returncode="
            f"{process.returncode}): stdout={process.stdout[-2000:]!r}; "
            f"stderr={process.stderr[-2000:]!r}"
        )
    if "All Request Has Been Exited" not in process.stdout:
        raise HBFAstraRunnerError(
            "ASTRA exited without the all-requests-completed marker")
    records = _parse_cycle_records(process.stdout)
    endpoints = _final_endpoint_cycles(
        records,
        num_npus=normalized.num_npus,
        expected_system_ids=(normalized.num_npus - 1,),
    )
    final_cycles = max(record.total_cycles for record in endpoints)
    return AstraHBFRunResult(
        final_cycles=final_cycles,
        endpoint_cycles=endpoints,
        trace=trace_audit,
        resolved_config=resolved,
        config_sha256=config_hashes,
        graph_sha256_by_rank=graph_hashes,
        binary_path=str(binary),
        binary_sha256=_sha256_file(binary),
        protobuf_runtime_version=protobuf_version,
        stdout_sha256=_sha256_bytes(process.stdout.encode("utf-8")),
        stderr_sha256=_sha256_bytes(process.stderr.encode("utf-8")),
    )


class PersistentHBFAstraRunner:
    """A live ASTRA process accepting exact ``hbf-background-v1`` DAGs.

    The process is bootstrapped with a one-tick EVENT graph and then parked at
    the sole group-controller endpoint.  :meth:`submit_jobs` may submit one
    or more DAGs at that boundary, allowing their named resources to contend
    in the same ASTRA event queue.  Every completion is delivered both as the
    return value and, optionally, through a synchronous callback.

    This class is intentionally single-threaded.  A callback must not call
    back into the session; the next submission is legal after
    :meth:`submit_jobs` returns.
    """

    def __init__(
            self, *, num_npus: int,
            config: AstraHBFRunConfig = AstraHBFRunConfig(),
            hbf_num_devices: int = 8,
            repo_root: Optional[Path] = None,
            binary_path: Optional[Path] = None,
            chakra_root: Optional[Path] = None) -> None:
        self.num_npus = _positive_int("num_npus", num_npus)
        self._repo_root = (
            Path(__file__).resolve().parents[2]
            if repo_root is None else Path(repo_root).resolve()
        )
        self._binary = (
            self._repo_root / DEFAULT_ASTRA_BINARY
            if binary_path is None else Path(binary_path).resolve()
        )
        self._chakra_root = (
            self._repo_root / DEFAULT_CHAKRA_ROOT
            if chakra_root is None else Path(chakra_root).resolve()
        )
        if not self._binary.is_file():
            raise HBFAstraRunnerError(
                f"congestion-aware ASTRA binary is missing: {self._binary}; "
                "run scripts/compile.sh")
        if not self._chakra_root.is_dir():
            raise HBFAstraRunnerError(
                f"Chakra source/build root is missing: {self._chakra_root}")
        device_count = _positive_int(
            "hbf_num_devices", hbf_num_devices)
        if (
            config.hbf_num_devices is not None
            and config.hbf_num_devices != device_count
        ):
            raise ValueError(
                "config.hbf_num_devices conflicts with hbf_num_devices")
        effective_config = replace(
            config, hbf_num_devices=device_count)
        synthetic_audit = HBFTraceAudit(
            label="persistent-event-bootstrap",
            trace_sha256="0" * 64,
            trace_bytes=0,
            num_npus=self.num_npus,
            model_parallel_groups=1,
            npus_per_group=self.num_npus,
            row_count=1,
            hbf_descriptor_count=1,
            hbf_stage_count=1,
            hbf_card_ids=(0,),
            hbf_resource_names=("persistent-bootstrap",),
        )
        synthetic_parsed = ParsedMicrotrace(
            execution_type="COLOCATED",
            model_parallel_groups=1,
            rows=(),
        )
        self.resolved_config = resolve_run_config(
            effective_config,
            parsed=synthetic_parsed,
            audit=synthetic_audit,
        )
        self._timeout_seconds = self.resolved_config.timeout_seconds
        self._temporary = tempfile.TemporaryDirectory(
            prefix="llmservingsim-persistent-hbf-astra-")
        self._work = Path(self._temporary.name)
        self._process: Optional[subprocess.Popen] = None
        self._stdout_buffer = b""
        self._stdout_parts: list[bytes] = []
        self._controller_ready = False
        self._closed = False
        self._current_ns = 0
        self._submitted_ids: set[str] = set()
        self._completions: list[HBFBackgroundCompletion] = []
        self._audit: Optional[PersistentHBFSessionAudit] = None

        try:
            self._start()
        except Exception:
            self.abort()
            raise

    @property
    def current_ns(self) -> int:
        return self._current_ns

    @property
    def is_open(self) -> bool:
        return not self._closed

    @property
    def astra_cycles_used(self) -> bool:
        return True

    def _start(self) -> None:
        converter, self._protobuf_version = _load_converter(
            self._chakra_root)
        trace = "\n".join((
            "EVENT",
            "1",
            TRACE_HEADER,
            (
                "persistent_hbf_bootstrap\t1\tREMOTE\t0\tLOCAL\t0\t"
                "REMOTE\t0\tNONE\t0\tNONE"
            ),
            "",
        ))
        trace_path = self._work / "bootstrap.txt"
        graph_prefix = self._work / "bootstrap"
        trace_path.write_text(trace, encoding="utf-8")
        try:
            converter(
                str(trace_path),
                str(graph_prefix),
                num_npus=self.num_npus,
            ).convert()
        except Exception as exc:
            raise HBFAstraRunnerError(
                "Chakra failed to build the persistent EVENT bootstrap: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        graph_paths = {
            rank: Path(f"{graph_prefix}.{rank}.et")
            for rank in range(self.num_npus)
        }
        if any(not path.is_file() for path in graph_paths.values()):
            raise HBFAstraRunnerError(
                "Chakra omitted a persistent bootstrap endpoint graph")
        self._graph_hashes = {
            rank: _sha256_file(path)
            for rank, path in graph_paths.items()
        }
        config_paths, self._config_hashes = _write_configs(
            self._work, self.resolved_config)
        command = [
            str(self._binary),
            f"--workload-configuration={graph_prefix}",
            f"--system-configuration={config_paths['system']}",
            f"--network-configuration={config_paths['network']}",
            f"--memory-configuration={config_paths['memory']}",
            "--start-npu-ids=0",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
        except OSError as exc:
            raise HBFAstraRunnerError(
                f"failed to start persistent ASTRA: {exc}") from exc
        boundary = self._read_wait_boundary()
        if _HBF_BACKGROUND_CAPABILITY not in boundary:
            raise HBFAstraRunnerError(
                "ASTRA binary lacks hbf-background-v1 capability")
        if _ENDPOINT_PARK_CAPABILITY not in boundary:
            raise HBFAstraRunnerError(
                "ASTRA binary lacks endpoint-park-v1 capability")
        record = self._model_boundary_record(boundary)
        if record.sys != 0 or record.iteration != 0:
            raise HBFAstraRunnerError(
                "persistent bootstrap did not stop at group controller 0")
        self._current_ns = record.total_cycles
        self._controller_ready = True

    def _read_wait_boundary(self) -> str:
        process = self._process
        if process is None or process.stdout is None:
            raise HBFAstraRunnerError("persistent ASTRA is not running")
        deadline = time.monotonic() + self._timeout_seconds
        collected: list[bytes] = []
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while True:
                while b"\n" in self._stdout_buffer:
                    line, self._stdout_buffer = (
                        self._stdout_buffer.split(b"\n", 1))
                    line += b"\n"
                    collected.append(line)
                    self._stdout_parts.append(line)
                    if b"Waiting" in line:
                        return b"".join(collected).decode(
                            "utf-8", errors="replace")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HBFAstraRunnerError(
                        "timed out waiting for an ASTRA protocol boundary")
                events = selector.select(remaining)
                if not events:
                    raise HBFAstraRunnerError(
                        "timed out waiting for an ASTRA protocol boundary")
                chunk = process.stdout.read(65_536)
                if not chunk:
                    return_code = process.poll()
                    raise HBFAstraRunnerError(
                        "persistent ASTRA closed stdout before a protocol "
                        f"boundary (returncode={return_code})")
                self._stdout_buffer += chunk
        finally:
            selector.close()

    @staticmethod
    def _model_boundary_record(boundary: str) -> AstraCycleRecord:
        records = _parse_cycle_records(boundary)
        if len(records) != 1:
            raise HBFAstraRunnerError(
                "model wait boundary must contain one completion record")
        return records[0]

    def _write_commands(self, *commands: str) -> None:
        process = self._process
        if (
            process is None
            or process.stdin is None
            or process.poll() is not None
        ):
            raise HBFAstraRunnerError("persistent ASTRA is not writable")
        for command in commands:
            if (
                not isinstance(command, str)
                or not command
                or "\n" in command
                or "\r" in command
            ):
                raise ValueError(
                    "ASTRA protocol commands must be non-empty single lines")
        try:
            process.stdin.write(
                ("".join(f"{command}\n" for command in commands)).encode(
                    "utf-8"))
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise HBFAstraRunnerError(
                f"persistent ASTRA command failed: {exc}") from exc

    @staticmethod
    def _completion_from_boundary(
            boundary: str, job: HBFBackgroundJob,
    ) -> HBFBackgroundCompletion:
        match = _HBF_BACKGROUND_COMPLETE.search(boundary)
        if match is None:
            raise HBFAstraRunnerError(
                "control wait boundary lacks an HBF completion")
        job_id, arrival, completion, stage_count = match.groups()
        if job_id != job.job_id:
            raise HBFAstraRunnerError(
                "HBF callback job id changed: "
                f"expected={job.job_id!r}, observed={job_id!r}")
        observed_arrival = int(arrival)
        observed_stage_count = int(stage_count)
        if observed_arrival != job.arrival_ns:
            raise HBFAstraRunnerError(
                "HBF callback arrival changed: "
                f"expected={job.arrival_ns}, observed={observed_arrival}")
        if observed_stage_count != len(job.stages):
            raise HBFAstraRunnerError(
                "HBF callback stage count changed: "
                f"expected={len(job.stages)}, "
                f"observed={observed_stage_count}")
        completion_ns = int(completion)
        if completion_ns < observed_arrival:
            raise HBFAstraRunnerError(
                "HBF callback completed before its arrival")
        return HBFBackgroundCompletion(
            job_id=job_id,
            arrival_ns=observed_arrival,
            completion_ns=completion_ns,
            stage_count=observed_stage_count,
            descriptor_sha256=job.descriptor_sha256,
            projection_schema=job.projection_schema,
        )

    def submit_jobs(
            self, jobs: Sequence[HBFBackgroundJob], *,
            on_complete: Optional[
                Callable[[HBFBackgroundCompletion], None]
            ] = None,
    ) -> tuple[HBFBackgroundCompletion, ...]:
        """Submit concurrent DAGs and wait for their exact ASTRA callbacks."""

        if self._closed:
            raise HBFAstraRunnerError("persistent ASTRA session is closed")
        if not self._controller_ready:
            raise HBFAstraRunnerError(
                "persistent ASTRA is not at a controller boundary")
        if on_complete is not None and not callable(on_complete):
            raise TypeError("on_complete must be callable")
        materialized = tuple(jobs)
        if not materialized:
            raise ValueError("jobs must contain at least one HBF job")
        pending: dict[str, HBFBackgroundJob] = {}
        commands = []
        for job in materialized:
            if not isinstance(job, HBFBackgroundJob):
                raise TypeError("jobs must contain HBFBackgroundJob values")
            command = job.command()
            if job.arrival_ns < self._current_ns:
                raise ValueError(
                    "HBF job arrival precedes ASTRA's current time: "
                    f"job={job.job_id}, arrival={job.arrival_ns}, "
                    f"current={self._current_ns}")
            if (
                job.job_id in self._submitted_ids
                or job.job_id in pending
            ):
                raise ValueError(
                    f"duplicate persistent HBF job id {job.job_id!r}")
            pending[job.job_id] = job
            commands.append(command)

        self._submitted_ids.update(pending)
        self._controller_ready = False
        self._write_commands(*commands, "park")
        completed: list[HBFBackgroundCompletion] = []
        while pending:
            boundary = self._read_wait_boundary()
            match = _HBF_BACKGROUND_COMPLETE.search(boundary)
            if match is not None:
                job_id = match.group(1)
                try:
                    job = pending.pop(job_id)
                except KeyError as exc:
                    raise HBFAstraRunnerError(
                        f"unknown or duplicate HBF callback {job_id!r}"
                    ) from exc
                completion = self._completion_from_boundary(boundary, job)
                self._current_ns = max(
                    self._current_ns, completion.completion_ns)
                self._completions.append(completion)
                completed.append(completion)
                if on_complete is not None:
                    on_complete(completion)
                self._write_commands("continue")
                continue

            record = self._model_boundary_record(boundary)
            if record.sys != 0 or record.iteration != 0:
                raise HBFAstraRunnerError(
                    "persistent session observed an unexpected model "
                    f"boundary: {record}")
            self._current_ns = max(
                self._current_ns, record.total_cycles)
            # A callback wakes the completed endpoint group.  Re-park it
            # while other submitted DAGs remain in ASTRA's event queue.
            self._write_commands("park")

        # The final callback unparked endpoint zero.  Acknowledge the callback
        # above, then stop at the next controller boundary so the caller may
        # submit another exact-time batch.
        boundary = self._read_wait_boundary()
        record = self._model_boundary_record(boundary)
        if record.sys != 0 or record.iteration != 0:
            raise HBFAstraRunnerError(
                "persistent session failed to return to controller zero")
        self._current_ns = max(self._current_ns, record.total_cycles)
        self._controller_ready = True
        return tuple(completed)

    def submit_projection(
            self, *, job_id: str, arrival_ns: int,
            projection: object,
            on_complete: Optional[
                Callable[[HBFBackgroundCompletion], None]
            ] = None,
    ) -> HBFBackgroundCompletion:
        """Submit one ``HBFModelAstraProjection`` and return its callback."""

        job = HBFBackgroundJob.from_projection(
            job_id=job_id,
            arrival_ns=arrival_ns,
            projection=projection,
        )
        return self.submit_jobs(
            (job,), on_complete=on_complete)[0]

    def _read_to_eof(self) -> bytes:
        process = self._process
        if process is None or process.stdout is None:
            return b""
        result = bytearray(self._stdout_buffer)
        self._stdout_buffer = b""
        deadline = time.monotonic() + self._timeout_seconds
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HBFAstraRunnerError(
                        "timed out waiting for persistent ASTRA to exit")
                events = selector.select(remaining)
                if not events:
                    raise HBFAstraRunnerError(
                        "timed out waiting for persistent ASTRA to exit")
                chunk = process.stdout.read(65_536)
                if not chunk:
                    break
                result.extend(chunk)
        finally:
            selector.close()
        payload = bytes(result)
        self._stdout_parts.append(payload)
        return payload

    def close(self) -> PersistentHBFSessionAudit:
        """Exit ASTRA cleanly and return the immutable session audit."""

        if self._audit is not None:
            return self._audit
        if self._closed:
            raise HBFAstraRunnerError(
                "persistent ASTRA session was aborted")
        if not self._controller_ready:
            raise HBFAstraRunnerError(
                "cannot close while an HBF submission is active")
        self._controller_ready = False
        self._write_commands("exit")
        tail = self._read_to_eof()
        process = self._process
        assert process is not None
        try:
            return_code = process.wait(timeout=self._timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            self.abort()
            raise HBFAstraRunnerError(
                "persistent ASTRA did not exit") from exc
        all_stdout = b"".join(self._stdout_parts)
        clean = (
            return_code == 0
            and b"All Request Has Been Exited" in tail
        )
        if not clean:
            self.abort()
            raise HBFAstraRunnerError(
                "persistent ASTRA did not report a clean all-requests exit: "
                f"returncode={return_code}, tail={tail[-2000:]!r}")
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        self._closed = True
        self._audit = PersistentHBFSessionAudit(
            completions=tuple(self._completions),
            resolved_config=self.resolved_config,
            config_sha256=dict(self._config_hashes),
            binary_path=str(self._binary),
            binary_sha256=_sha256_file(self._binary),
            protobuf_runtime_version=self._protobuf_version,
            bootstrap_graph_sha256_by_rank=dict(self._graph_hashes),
            stdout_sha256=_sha256_bytes(all_stdout),
            clean_exit=True,
        )
        self._temporary.cleanup()
        return self._audit

    def abort(self) -> None:
        """Terminate a failed session without claiming clean ASTRA results."""

        if self._closed:
            return
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
        self._closed = True
        self._controller_ready = False
        self._temporary.cleanup()

    def __enter__(self) -> "PersistentHBFAstraRunner":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False


__all__ = [
    "AstraCycleRecord",
    "AstraHBFRunConfig",
    "AstraHBFRunResult",
    "DEFAULT_ASTRA_BINARY",
    "DEFAULT_CHAKRA_ROOT",
    "HBFAstraRunnerError",
    "HBFBackgroundCompletion",
    "HBFBackgroundJob",
    "HBFTextTraceArtifact",
    "HBFTraceAudit",
    "PersistentHBFAstraRunner",
    "PersistentHBFSessionAudit",
    "ResolvedAstraHBFRunConfig",
    "resolve_run_config",
    "run_hbf_trace_artifact",
    "validate_hbf_trace_artifact",
]

"""Build small ASTRA-Sim traces for analytical-model conformance checks.

The production P4D4 and full-model HBF simulators currently schedule their
operations in Python.  This module does not replace that scheduler and does
not use ASTRA cycles as a latency oracle.  It provides deliberately small
traces that make the analytical contracts auditable against the repository's
existing text-trace -> Chakra -> ASTRA-Sim path:

* TP4/TP8 AllReduce, AllGather, and ReduceScatter payload conventions;
* one bulk D-to-P or P-to-D KV transfer per TP rank pair; and
* whole-gang HBF media resources shared by every rank in a TP replica.

The P-to-D trace is ``COLOCATED``, not ``PREFILL``.  The latter converter
streams K/V after every qkv projection, which would violate the strict
analytical TTFT boundary.  Instead, the only P-to-D source is a sampler
boundary marker.  Logical ranks in a transfer micrograph are source-first;
production integration must alias those logical endpoints to the owning P/D
rank IDs.

ASTRA collective algorithms can charge topology-dependent hop and fixed
latency terms differently from the analytical models.  These artifacts prove
operation structure, dependency order, byte counts, and dimension scoping;
they intentionally make no cycle-equality claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import json
import re
from typing import Iterable

from .h100_kernel_calibrated_prompt import (
    BF16_BYTES,
    QWEN_EXPERTS,
    QWEN_HEAD_DIM,
    QWEN_HIDDEN_SIZE,
    QWEN_KV_HEADS,
    QWEN_LAYERS,
)


TRACE_HEADER = (
    "Layername comp_time input_loc input_size weight_loc weight_size "
    "output_loc output_size comm_type comm_size misc"
)
SUPPORTED_TP_SIZES = frozenset({4, 8})
_COLLECTIVES = frozenset({
    "ALLREDUCE",
    "ALLGATHER",
    "REDUCESCATTER",
})
_LOCATION = re.compile(
    r"^(?:LOCAL|REMOTE(?::\d+(?:\.\d+)?)?|"
    r"CXL(?::\d+(?:\.\d+)?)?|STORAGE(?::\d+(?:\.\d+)?)?|"
    r"HBF(?::\d+(?:\.\d+)?)?)$"
)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_tp_size(tp_size: object) -> int:
    value = _positive_int("tp_size", tp_size)
    if value not in SUPPORTED_TP_SIZES:
        raise ValueError(
            f"tp_size must be one of {sorted(SUPPORTED_TP_SIZES)}")
    return value


def _ceil_fraction(value: Fraction) -> int:
    return (
        value.numerator + value.denominator - 1
    ) // value.denominator


@dataclass(frozen=True)
class TraceRow:
    """One canonical LLMServingSim text-trace row."""

    name: str
    comp_time_ns: int
    input_loc: str
    input_size: int
    weight_loc: str
    weight_size: int
    output_loc: str
    output_size: int
    comm_type: str = "NONE"
    comm_size: int = 0
    misc: str = "NONE"

    def validate(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or any(character.isspace() for character in self.name)
        ):
            raise ValueError(
                "trace row name must be non-empty and whitespace-free")
        _positive_int("trace row comp_time_ns", self.comp_time_ns)
        for name in (
            "input_size",
            "weight_size",
            "output_size",
            "comm_size",
        ):
            _nonnegative_int(
                f"trace row {name}", getattr(self, name))
        for name in ("input_loc", "weight_loc", "output_loc"):
            location = getattr(self, name)
            if not isinstance(location, str) or not _LOCATION.fullmatch(
                    location):
                raise ValueError(
                    f"trace row {name} has unsupported location {location!r}")
        if (
            not isinstance(self.comm_type, str)
            or not self.comm_type
            or any(character.isspace() for character in self.comm_type)
        ):
            raise ValueError("trace row comm_type must be whitespace-free")
        base = self.comm_type.split(":", 1)[0]
        if base != "NONE" and base not in _COLLECTIVES:
            raise ValueError(
                f"unsupported trace collective {self.comm_type!r}")
        if base == "NONE" and self.comm_size != 0:
            raise ValueError("NONE communication must have zero comm_size")
        if base != "NONE" and self.comm_size <= 0:
            raise ValueError("collective communication must have bytes")
        if not isinstance(self.misc, str) or not self.misc:
            raise ValueError("trace row misc must be a non-empty string")
        if any(character.isspace() for character in self.misc):
            raise ValueError(
                "trace row misc must not contain whitespace")

    def render(self) -> str:
        self.validate()
        return "\t".join((
            self.name,
            str(self.comp_time_ns),
            self.input_loc,
            str(self.input_size),
            self.weight_loc,
            str(self.weight_size),
            self.output_loc,
            str(self.output_size),
            self.comm_type,
            str(self.comm_size),
            self.misc,
        ))

    @classmethod
    def parse(cls, line: str) -> "TraceRow":
        fields = line.split()
        if len(fields) != 11:
            raise ValueError(
                "trace row must contain exactly 11 whitespace-separated "
                f"fields, observed={len(fields)}: {line!r}")
        try:
            row = cls(
                name=fields[0],
                comp_time_ns=int(fields[1]),
                input_loc=fields[2],
                input_size=int(fields[3]),
                weight_loc=fields[4],
                weight_size=int(fields[5]),
                output_loc=fields[6],
                output_size=int(fields[7]),
                comm_type=fields[8],
                comm_size=int(fields[9]),
                misc=fields[10],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid trace row: {line!r}") from exc
        row.validate()
        return row


@dataclass(frozen=True)
class ParsedMicrotrace:
    execution_type: str
    model_parallel_groups: int
    rows: tuple[TraceRow, ...]


def parse_microtrace(text: str) -> ParsedMicrotrace:
    """Parse the strict subset emitted by this conformance builder."""

    if not isinstance(text, str) or not text:
        raise ValueError("microtrace text must be non-empty")
    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError("microtrace is missing its preamble")
    first = lines[0].split()
    if (
        len(first) != 3
        or first[1] != "model_parallel_NPU_group:"
    ):
        raise ValueError(f"invalid microtrace first line: {lines[0]!r}")
    execution_type = first[0]
    if execution_type != "COLOCATED":
        raise ValueError(
            "conformance microtraces must use COLOCATED execution")
    try:
        groups = _positive_int(
            "model_parallel_groups", int(first[2]))
        declared_rows = _positive_int(
            "declared row count", int(lines[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid microtrace preamble integer") from exc
    if lines[2].strip() != TRACE_HEADER:
        raise ValueError("microtrace header does not match Chakra format")
    rows = tuple(TraceRow.parse(line) for line in lines[3:])
    if len(rows) != declared_rows:
        raise ValueError(
            "microtrace row count mismatch: "
            f"declared={declared_rows}, observed={len(rows)}")
    return ParsedMicrotrace(
        execution_type=execution_type,
        model_parallel_groups=groups,
        rows=rows,
    )


def _render_microtrace(
        *, model_parallel_groups: int,
        rows: Iterable[TraceRow]) -> str:
    groups = _positive_int(
        "model_parallel_groups", model_parallel_groups)
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("microtrace must contain at least one row")
    rendered = tuple(row.render() for row in materialized)
    text = "\n".join((
        f"COLOCATED\t\tmodel_parallel_NPU_group: {groups}",
        str(len(materialized)),
        TRACE_HEADER,
        *rendered,
        "",
    ))
    parse_microtrace(text)
    return text


@dataclass(frozen=True)
class CollectiveOperationContract:
    layer_name: str
    collective: str
    comm_size: int
    payload_semantics: str
    ring_wire_bytes_per_rank: int
    involved_dim: tuple[bool, ...] = (True,)


@dataclass(frozen=True)
class CollectiveMicrotrace:
    text: str
    tp_size: int
    replicas: int
    num_npus: int
    total_tokens: int
    operations: tuple[CollectiveOperationContract, ...]
    cycle_equality_claimed: bool = False


def qwen_collective_contracts(
        *, tp_size: int,
        total_tokens: int,
        involved_dim: tuple[bool, ...] = (True,),
) -> tuple[CollectiveOperationContract, ...]:
    """Mirror the payload formulas in the P4D4/HBF latency models.

    AllReduce and ReduceScatter carry the total activation buffer.
    AllGather carries the local pre-gather chunk, matching
    ``trace_generator._emit_moe_block`` and the analytical models.
    """

    tp = _validate_tp_size(tp_size)
    tokens = _positive_int("total_tokens", total_tokens)
    if (
        not isinstance(involved_dim, tuple)
        or not involved_dim
        or any(not isinstance(value, bool) for value in involved_dim)
        or not any(involved_dim)
    ):
        raise ValueError(
            "involved_dim must be a non-empty boolean tuple with an "
            "enabled dimension")
    hidden_total = tokens * QWEN_HIDDEN_SIZE * BF16_BYTES
    dispatch_local_tokens = max(1, tokens // tp)
    dispatch_local_chunk = (
        dispatch_local_tokens
        * (QWEN_HIDDEN_SIZE + QWEN_EXPERTS)
        * BF16_BYTES
    )
    return (
        CollectiveOperationContract(
            layer_name="tp_allreduce_probe",
            collective="ALLREDUCE",
            comm_size=hidden_total,
            payload_semantics="total_activation_buffer",
            ring_wire_bytes_per_rank=_ceil_fraction(
                Fraction(2 * (tp - 1) * hidden_total, tp)),
            involved_dim=involved_dim,
        ),
        CollectiveOperationContract(
            layer_name="ep_allgather_probe",
            collective="ALLGATHER",
            comm_size=dispatch_local_chunk,
            payload_semantics="per_rank_local_chunk",
            ring_wire_bytes_per_rank=(
                (tp - 1) * dispatch_local_chunk),
            involved_dim=involved_dim,
        ),
        CollectiveOperationContract(
            layer_name="ep_reduce_scatter_probe",
            collective="REDUCESCATTER",
            comm_size=hidden_total,
            payload_semantics="pre_scatter_total_buffer",
            ring_wire_bytes_per_rank=_ceil_fraction(
                Fraction((tp - 1) * hidden_total, tp)),
            involved_dim=involved_dim,
        ),
    )


def build_collective_microtrace(
        *, tp_size: int, total_tokens: int,
        marker_runtime_ns: int = 1,
        replicas: int = 1) -> CollectiveMicrotrace:
    """Build one serial TP collective probe for TP4 or TP8.

    ``replicas=2`` represents the HBF server's 2xTP4 layout on a logical
    ``[4, 2]`` topology. Collectives are then scoped to dimension zero with
    ``involved_dim=[True, False]`` and do not span the replica dimension.
    The single ``model_parallel_NPU_group`` is deliberate: that trace field
    partitions pipeline stages, while replicas are a network-topology axis.
    """

    tp = _validate_tp_size(tp_size)
    runtime = _positive_int("marker_runtime_ns", marker_runtime_ns)
    replica_count = _positive_int("replicas", replicas)
    if replica_count not in (1, 2):
        raise ValueError("replicas must be 1 or 2")
    if replica_count == 2 and tp != 4:
        raise ValueError("only TP4 supports two replicas on eight cards")
    involved_dim = (
        (True, False) if replica_count > 1 else (True,))
    operations = qwen_collective_contracts(
        tp_size=tp,
        total_tokens=total_tokens,
        involved_dim=involved_dim,
    )
    dim_suffix = ",".join(
        "1" if enabled else "0" for enabled in involved_dim)
    rows = []
    for index, operation in enumerate(operations):
        rows.append(TraceRow(
            name=operation.layer_name,
            comp_time_ns=runtime,
            input_loc="REMOTE:0" if index == 0 else "LOCAL",
            input_size=1,
            weight_loc="LOCAL",
            weight_size=0,
            output_loc=(
                "REMOTE:0"
                if index == len(operations) - 1 else "LOCAL"
            ),
            output_size=1,
            comm_type=f"{operation.collective}:{dim_suffix}",
            comm_size=operation.comm_size,
        ))
    text = _render_microtrace(
        model_parallel_groups=1, rows=rows)
    parsed = parse_microtrace(text)
    if tuple(row.comm_size for row in parsed.rows) != tuple(
            operation.comm_size for operation in operations):
        raise AssertionError("rendered collective bytes changed")
    return CollectiveMicrotrace(
        text=text,
        tp_size=tp,
        replicas=replica_count,
        num_npus=tp * replica_count,
        total_tokens=total_tokens,
        operations=operations,
    )


class KVTransferDirection(str, Enum):
    D_TO_P = "d_to_p"
    P_TO_D = "p_to_d"


@dataclass(frozen=True)
class LogicalRankAlias:
    logical_rank: int
    role: str
    role_rank: int


@dataclass(frozen=True)
class BulkKVTransferContract:
    direction: KVTransferDirection
    tp_size: int
    token_count: int
    bytes_per_rank: int
    aggregate_bytes: int
    source_boundary: str
    destination_gate: str
    issue_phase: str
    logical_rank_aliases: tuple[LogicalRankAlias, ...]
    bulk_copy_count: int = 1
    qkv_streaming: bool = False
    uses_prefill_converter: bool = False
    requires_source_first_endpoint_alias: bool = True


@dataclass(frozen=True)
class BulkKVTransferMicrotrace:
    text: str
    num_npus: int
    model_parallel_groups: int
    contract: BulkKVTransferContract
    cycle_equality_claimed: bool = False


def qwen_kv_bytes_per_rank(
        *, tp_size: int, token_count: int) -> int:
    """Return physical K+V bytes owned by one TP rank.

    Qwen has four KV heads.  TP8 replicates each logical KV head on two
    physical ranks, so its per-rank storage does not continue dividing after
    TP4.
    """

    tp = _validate_tp_size(tp_size)
    tokens = _nonnegative_int("token_count", token_count)
    logical_bytes_per_token = (
        2
        * QWEN_LAYERS
        * QWEN_KV_HEADS
        * QWEN_HEAD_DIM
        * BF16_BYTES
    )
    replication = max(1, tp // QWEN_KV_HEADS)
    numerator = logical_bytes_per_token * replication
    if numerator % tp:
        raise ValueError("physical KV bytes do not divide TP ranks")
    return tokens * (numerator // tp)


def build_bulk_kv_transfer_microtrace(
        *, direction: KVTransferDirection | str,
        tp_size: int, token_count: int,
        marker_runtime_ns: int = 1) -> BulkKVTransferMicrotrace:
    """Build an isolated, single-copy P/D KV transfer micrograph.

    The first TP group owns two source-side marker rows and the second owns
    one destination row.  ``convert_common`` therefore emits one point-to-
    point send/receive pair per corresponding TP rank.  The source marker's
    output size and destination marker's input size are both per-rank bytes.
    """

    try:
        transfer_direction = KVTransferDirection(direction)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "direction must be 'd_to_p' or 'p_to_d'") from exc
    tp = _validate_tp_size(tp_size)
    tokens = _positive_int("token_count", token_count)
    runtime = _positive_int("marker_runtime_ns", marker_runtime_ns)
    bytes_per_rank = qwen_kv_bytes_per_rank(
        tp_size=tp, token_count=tokens)

    if transfer_direction is KVTransferDirection.P_TO_D:
        source_role = "prefill"
        destination_role = "decode"
        pre_boundary = "prefill_model_compute_complete"
        source_boundary = "sampler_ttft_boundary"
        destination_gate = "decode_kv_publish_commit"
        issue_phase = "strictly_after_ttft"
    else:
        source_role = "decode"
        destination_role = "prefill"
        pre_boundary = "decode_kv_resident_ready"
        source_boundary = "decode_to_prefill_bulk_source"
        destination_gate = "resume_prefill_compute_gate"
        issue_phase = "before_resume_prefill_compute"

    rows = (
        TraceRow(
            name=pre_boundary,
            comp_time_ns=runtime,
            input_loc="REMOTE:0",
            input_size=1,
            weight_loc="LOCAL",
            weight_size=0,
            output_loc="LOCAL",
            output_size=1,
        ),
        TraceRow(
            name=source_boundary,
            comp_time_ns=runtime,
            input_loc="LOCAL",
            input_size=1,
            weight_loc="LOCAL",
            weight_size=0,
            output_loc="LOCAL",
            output_size=bytes_per_rank,
        ),
        TraceRow(
            name=destination_gate,
            comp_time_ns=runtime,
            input_loc="LOCAL",
            input_size=bytes_per_rank,
            weight_loc="LOCAL",
            weight_size=0,
            output_loc="REMOTE:0",
            output_size=1,
        ),
    )
    text = _render_microtrace(
        model_parallel_groups=2, rows=rows)
    aliases = tuple(
        LogicalRankAlias(rank, source_role, rank)
        for rank in range(tp)
    ) + tuple(
        LogicalRankAlias(tp + rank, destination_role, rank)
        for rank in range(tp)
    )
    contract = BulkKVTransferContract(
        direction=transfer_direction,
        tp_size=tp,
        token_count=tokens,
        bytes_per_rank=bytes_per_rank,
        aggregate_bytes=bytes_per_rank * tp,
        source_boundary=source_boundary,
        destination_gate=destination_gate,
        issue_phase=issue_phase,
        logical_rank_aliases=aliases,
    )
    parsed = parse_microtrace(text)
    if any("qkv_proj" in row.name for row in parsed.rows):
        raise AssertionError("bulk transfer trace streamed qkv")
    if parsed.rows[1].output_size != parsed.rows[2].input_size:
        raise AssertionError("bulk transfer byte count changed at boundary")
    return BulkKVTransferMicrotrace(
        text=text,
        num_npus=2 * tp,
        model_parallel_groups=2,
        contract=contract,
    )


class HBFMediaOperation(str, Enum):
    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class HBFMediaContract:
    operation: HBFMediaOperation
    tp_size: int
    runtime_ns: int
    tensor_bytes_per_rank: int
    aggregate_tensor_bytes: int
    resources: tuple[str, ...]
    expected_participants: int
    whole_gang_resource_semantics: bool = True


@dataclass(frozen=True)
class HBFMediaMicrotrace:
    text: str
    num_npus: int
    contract: HBFMediaContract
    cycle_equality_claimed: bool = False


def build_hbf_media_microtrace(
        *, operation: HBFMediaOperation | str,
        tp_size: int, runtime_ns: int,
        tensor_bytes_per_rank: int,
        replica_card_offset: int = 0) -> HBFMediaMicrotrace:
    """Build one whole-gang HBF media stage with explicit card resources."""

    try:
        media_operation = HBFMediaOperation(operation)
    except (TypeError, ValueError) as exc:
        raise ValueError("operation must be 'read' or 'write'") from exc
    tp = _validate_tp_size(tp_size)
    runtime = _positive_int("runtime_ns", runtime_ns)
    per_rank = _positive_int(
        "tensor_bytes_per_rank", tensor_bytes_per_rank)
    card_offset = _nonnegative_int(
        "replica_card_offset", replica_card_offset)
    resources = tuple(
        f"hbf-card:{card_offset + rank}:{media_operation.value}"
        for rank in range(tp)
    )
    aggregate = per_rank * tp
    descriptor = {
        "v": 1,
        "expected_participants": tp,
        # The Chakra schema requires one tensor device. Shared-resource
        # scheduling is governed by the explicit whole-gang resource list.
        "card_id": card_offset,
        "gang_base": (
            f"conformance:hbf:{media_operation.value}:"
            f"tp{tp}:cards{card_offset}-{card_offset + tp - 1}"
        ),
        "merge_ns": 0,
        "stages": [{
            "id": f"hbf_{media_operation.value}",
            "runtime_ns": runtime,
            "tensor_bytes": aggregate,
            "resources": list(resources),
            "deps": [],
        }],
        "terminals": [f"hbf_{media_operation.value}"],
    }
    misc = json.dumps(
        {"batch": "NONE", "hbf": descriptor},
        separators=(",", ":"),
        sort_keys=True,
    )
    rows = (
        TraceRow(
            name=f"hbf_{media_operation.value}_ready",
            comp_time_ns=1,
            input_loc="REMOTE:0",
            input_size=1,
            weight_loc="LOCAL",
            weight_size=0,
            output_loc="LOCAL",
            output_size=1,
        ),
        TraceRow(
            name=f"hbf_{media_operation.value}_probe",
            comp_time_ns=1,
            input_loc="LOCAL",
            input_size=1,
            weight_loc="LOCAL",
            weight_size=0,
            output_loc="LOCAL",
            output_size=1,
            misc=misc,
        ),
        TraceRow(
            name=f"hbf_{media_operation.value}_commit",
            comp_time_ns=1,
            input_loc="LOCAL",
            input_size=1,
            weight_loc="LOCAL",
            weight_size=0,
            output_loc="REMOTE:0",
            output_size=1,
        ),
    )
    text = _render_microtrace(
        model_parallel_groups=1, rows=rows)
    return HBFMediaMicrotrace(
        text=text,
        num_npus=tp,
        contract=HBFMediaContract(
            operation=media_operation,
            tp_size=tp,
            runtime_ns=runtime,
            tensor_bytes_per_rank=per_rank,
            aggregate_tensor_bytes=aggregate,
            resources=resources,
            expected_participants=tp,
        ),
    )


__all__ = [
    "BulkKVTransferContract",
    "BulkKVTransferMicrotrace",
    "CollectiveMicrotrace",
    "CollectiveOperationContract",
    "HBFMediaContract",
    "HBFMediaMicrotrace",
    "HBFMediaOperation",
    "KVTransferDirection",
    "LogicalRankAlias",
    "ParsedMicrotrace",
    "TRACE_HEADER",
    "TraceRow",
    "build_bulk_kv_transfer_microtrace",
    "build_collective_microtrace",
    "build_hbf_media_microtrace",
    "parse_microtrace",
    "qwen_collective_contracts",
    "qwen_kv_bytes_per_rank",
]

"""Configuration contract for a mirrored full-model HBF P4D4 server.

The eight HBF-GPU cards are split into one TP4 prefill role and one TP4
decode role.  Every published session KV version is stored by both roles.
Consequently, the unique logical capacity is the capacity of one TP4 role,
while persistent writes consume two physical copies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .hbf_full_model_latency import (
    qwen_logical_kv_bytes_per_token,
    qwen_model_weight_bytes_per_rank,
)


SCHEMA_VERSION = 1
PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
ROLE_NAMES = (PREFILL_ROLE, DECODE_ROLE)
GIB = 1024 ** 3
MIB = 1024 ** 2
MIN_PREFILL_WORKSPACE_BYTES_PER_CARD = 10 * GIB


def _strict_fields(
        raw: Mapping[str, object], allowed: set[str], scope: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown {scope} field(s): {unknown}")


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_float(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be non-negative and finite")
    return float(value)


def _positive_float(name: str, value: object) -> float:
    value = _nonnegative_float(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


@dataclass(frozen=True)
class MirroredP4D4Hardware:
    """Shared HBF media and GPU-to-HBF migration inputs."""

    card_count: int = 8
    cards_per_role: int = 4
    hbf_capacity_bytes_per_card: int = 1_280_000_000_000
    hbf_read_bandwidth_gbps_per_card: float = 3_350.0
    hbf_read_latency_us: float = 5.0
    hbf_write_bandwidth_gbps_per_card: float = 335.0
    hbf_write_latency_us: float = 20.0
    npu_peak_tflops_per_card: float = 989.5
    gpu_source_root_bandwidth_gbps: float = 200.0
    rdma_bandwidth_gbps: float = 80.0
    rdma_one_way_latency_us: float = 10.0
    card_pcie_bandwidth_gbps: float = 50.0
    card_pcie_latency_us: float = 1.0

    @classmethod
    def from_dict(
            cls, raw: Mapping[str, object]) -> "MirroredP4D4Hardware":
        _strict_fields(raw, set(cls.__dataclass_fields__), "hardware")
        result = cls(**raw)
        result.validate()
        return result

    def validate(self) -> None:
        for name in (
            "card_count",
            "cards_per_role",
            "hbf_capacity_bytes_per_card",
        ):
            _positive_int(f"hardware.{name}", getattr(self, name))
        for name in (
            "hbf_read_bandwidth_gbps_per_card",
            "hbf_write_bandwidth_gbps_per_card",
            "npu_peak_tflops_per_card",
            "gpu_source_root_bandwidth_gbps",
            "rdma_bandwidth_gbps",
            "card_pcie_bandwidth_gbps",
        ):
            _positive_float(f"hardware.{name}", getattr(self, name))
        for name in (
            "hbf_read_latency_us",
            "hbf_write_latency_us",
            "rdma_one_way_latency_us",
            "card_pcie_latency_us",
        ):
            _nonnegative_float(f"hardware.{name}", getattr(self, name))
        if self.card_count != 8 or self.cards_per_role != 4:
            raise ValueError(
                "mirrored P4D4 requires exactly eight cards split 4+4")


@dataclass(frozen=True)
class MirroredRoleConfig:
    """Role-local cards, LPDDR workspace, and scheduler limits."""

    card_ids: tuple[int, ...]
    lpddr_capacity_bytes_per_card: int = 64 * GIB
    lpddr_bandwidth_gbps_per_card: float = 204.8
    workspace_bytes_per_card: int = 16 * GIB
    max_num_batched_tokens: int = 8_192
    max_num_seqs: int = 128
    max_prefill_chunk_tokens: int = 4_096
    max_active_kv_tokens: int = 131_072

    @classmethod
    def from_dict(
            cls, raw: Mapping[str, object], *,
            role: str) -> "MirroredRoleConfig":
        _strict_fields(raw, set(cls.__dataclass_fields__), f"role {role}")
        values = dict(raw)
        card_ids = values.get("card_ids")
        if not isinstance(card_ids, list):
            raise ValueError(f"role {role}.card_ids must be an array")
        values["card_ids"] = tuple(card_ids)
        result = cls(**values)
        result.validate(role=role)
        return result

    def validate(self, *, role: str) -> None:
        if role not in ROLE_NAMES:
            raise ValueError(f"unsupported mirrored role {role!r}")
        if len(self.card_ids) != 4:
            raise ValueError(f"role {role} must own exactly four cards")
        if any(
            isinstance(card_id, bool)
            or not isinstance(card_id, int)
            or card_id < 0
            for card_id in self.card_ids
        ):
            raise ValueError(
                f"role {role}.card_ids must be non-negative integers")
        if len(set(self.card_ids)) != len(self.card_ids):
            raise ValueError(f"role {role}.card_ids must be unique")
        for name in (
            "lpddr_capacity_bytes_per_card",
            "workspace_bytes_per_card",
            "max_num_batched_tokens",
            "max_num_seqs",
            "max_prefill_chunk_tokens",
            "max_active_kv_tokens",
        ):
            _positive_int(f"role {role}.{name}", getattr(self, name))
        _positive_float(
            f"role {role}.lpddr_bandwidth_gbps_per_card",
            self.lpddr_bandwidth_gbps_per_card,
        )
        if self.workspace_bytes_per_card >= (
                self.lpddr_capacity_bytes_per_card):
            raise ValueError(
                f"role {role} workspace must leave usable LPDDR")
        if (
            role == PREFILL_ROLE
            and self.workspace_bytes_per_card
            < MIN_PREFILL_WORKSPACE_BYTES_PER_CARD
        ):
            raise ValueError(
                "prefill role workspace must be at least 10 GiB")
        if self.max_num_seqs > 128:
            raise ValueError(
                f"role {role}.max_num_seqs exceeds calibrated support (128)")
        if self.max_prefill_chunk_tokens > self.max_num_batched_tokens:
            raise ValueError(
                f"role {role}.max_prefill_chunk_tokens exceeds its "
                "batched-token limit")
        if (
            self.workspace_bytes_per_card
            + self.max_active_kv_bytes_per_card
            > self.lpddr_capacity_bytes_per_card
        ):
            raise ValueError(
                f"role {role} LPDDR cannot fit workspace plus "
                "max_active_kv_tokens")

    @property
    def usable_lpddr_bytes_per_card(self) -> int:
        return (
            self.lpddr_capacity_bytes_per_card
            - self.workspace_bytes_per_card
        )

    @property
    def max_active_kv_bytes_per_card(self) -> int:
        logical_bytes = (
            self.max_active_kv_tokens
            * qwen_logical_kv_bytes_per_token()
        )
        return int(math.ceil(logical_bytes / 4))


@dataclass(frozen=True)
class PcieRootHandoffConfig:
    """Bidirectional P-role to D-role handoff through two PCIe roots."""

    prefill_root_id: int = 0
    decode_root_id: int = 1
    prefill_root_bandwidth_gbps: float = 200.0
    decode_root_bandwidth_gbps: float = 200.0
    inter_root_bandwidth_gbps: float = 200.0
    fixed_latency_us: float = 3.0

    @classmethod
    def from_dict(
            cls, raw: Mapping[str, object]) -> "PcieRootHandoffConfig":
        _strict_fields(raw, set(cls.__dataclass_fields__), "pcie_handoff")
        result = cls(**raw)
        result.validate()
        return result

    def validate(self) -> None:
        for name in ("prefill_root_id", "decode_root_id"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"pcie_handoff.{name} must be a non-negative integer")
        if self.prefill_root_id == self.decode_root_id:
            raise ValueError("P and D roles must use distinct PCIe roots")
        for name in (
            "prefill_root_bandwidth_gbps",
            "decode_root_bandwidth_gbps",
            "inter_root_bandwidth_gbps",
        ):
            _positive_float(
                f"pcie_handoff.{name}", getattr(self, name))
        _nonnegative_float(
            "pcie_handoff.fixed_latency_us", self.fixed_latency_us)


@dataclass(frozen=True)
class MirroredStoragePolicy:
    """Persistent storage granularity and scheduling policy."""

    page_size_bytes: int = 4_096
    write_chunk_bytes_per_card: int = 64 * MIB
    foreground_read_priority: bool = True
    guarantee_write_chunk_after_foreground_read: bool = True

    @classmethod
    def from_dict(
            cls, raw: Mapping[str, object]) -> "MirroredStoragePolicy":
        _strict_fields(raw, set(cls.__dataclass_fields__), "storage")
        result = cls(**raw)
        result.validate()
        return result

    def validate(self) -> None:
        for name in ("page_size_bytes", "write_chunk_bytes_per_card"):
            _positive_int(f"storage.{name}", getattr(self, name))
        for name in (
            "foreground_read_priority",
            "guarantee_write_chunk_after_foreground_read",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"storage.{name} must be boolean")
        if self.page_size_bytes & (self.page_size_bytes - 1):
            raise ValueError("storage.page_size_bytes must be a power of two")
        if self.write_chunk_bytes_per_card % self.page_size_bytes:
            raise ValueError(
                "storage.write_chunk_bytes_per_card must be page aligned")
        if (
            self.foreground_read_priority
            and not self.guarantee_write_chunk_after_foreground_read
        ):
            raise ValueError(
                "foreground read priority must retain write progress")


@dataclass(frozen=True)
class MirroredP4D4Config:
    """Complete configuration for one mirrored eight-card HBF server."""

    hardware: MirroredP4D4Hardware
    prefill: MirroredRoleConfig
    decode: MirroredRoleConfig
    pcie_handoff: PcieRootHandoffConfig
    storage: MirroredStoragePolicy

    def validate(self) -> None:
        self.hardware.validate()
        self.prefill.validate(role=PREFILL_ROLE)
        self.decode.validate(role=DECODE_ROLE)
        self.pcie_handoff.validate()
        self.storage.validate()
        all_cards = self.prefill.card_ids + self.decode.card_ids
        if set(all_cards) != set(range(self.hardware.card_count)):
            raise ValueError(
                "P/D role card_ids must be disjoint and cover all cards")
        if len(set(all_cards)) != self.hardware.card_count:
            raise ValueError("P/D role card_ids overlap")
        if any(
            card_id // self.hardware.cards_per_role
            != self.pcie_handoff.prefill_root_id
            for card_id in self.prefill.card_ids
        ):
            raise ValueError(
                "prefill cards do not match prefill_root_id")
        if any(
            card_id // self.hardware.cards_per_role
            != self.pcie_handoff.decode_root_id
            for card_id in self.decode.card_ids
        ):
            raise ValueError("decode cards do not match decode_root_id")
        if self.model_weight_bytes_per_rank >= (
                self.hardware.hbf_capacity_bytes_per_card):
            raise ValueError("full-model TP4 weights do not fit on HBF")

    @property
    def model_weight_bytes_per_rank(self) -> int:
        return qwen_model_weight_bytes_per_rank(4)

    @property
    def usable_hbf_bytes_per_card(self) -> int:
        return (
            self.hardware.hbf_capacity_bytes_per_card
            - self.model_weight_bytes_per_rank
        )

    @property
    def unique_logical_capacity_bytes(self) -> int:
        return (
            self.usable_hbf_bytes_per_card
            * self.hardware.cards_per_role
        )

    @property
    def physical_persistent_capacity_bytes(self) -> int:
        return 2 * self.unique_logical_capacity_bytes

    def role(self, role: str) -> MirroredRoleConfig:
        if role == PREFILL_ROLE:
            return self.prefill
        if role == DECODE_ROLE:
            return self.decode
        raise ValueError(f"unsupported mirrored role {role!r}")

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "hardware": asdict(self.hardware),
            "roles": {
                PREFILL_ROLE: {
                    **asdict(self.prefill),
                    "max_active_kv_bytes_per_card":
                        self.prefill.max_active_kv_bytes_per_card,
                    "minimum_lpddr_contract_bytes_per_card": (
                        self.prefill.workspace_bytes_per_card
                        + self.prefill.max_active_kv_bytes_per_card
                    ),
                },
                DECODE_ROLE: {
                    **asdict(self.decode),
                    "max_active_kv_bytes_per_card":
                        self.decode.max_active_kv_bytes_per_card,
                    "minimum_lpddr_contract_bytes_per_card": (
                        self.decode.workspace_bytes_per_card
                        + self.decode.max_active_kv_bytes_per_card
                    ),
                },
            },
            "pcie_handoff": asdict(self.pcie_handoff),
            "storage": asdict(self.storage),
            "model_weight_bytes_per_rank": (
                self.model_weight_bytes_per_rank),
            "usable_hbf_bytes_per_card": self.usable_hbf_bytes_per_card,
            "unique_logical_capacity_bytes": (
                self.unique_logical_capacity_bytes),
            "physical_persistent_capacity_bytes": (
                self.physical_persistent_capacity_bytes),
        }


def load_mirrored_p4d4_config(path: str | Path) -> MirroredP4D4Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as source:
        raw = json.load(source)
    if not isinstance(raw, dict):
        raise ValueError("mirrored P4D4 config must be a JSON object")
    _strict_fields(
        raw,
        {"schema_version", "hardware", "roles", "pcie_handoff", "storage"},
        "top-level",
    )
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SCHEMA_VERSION}")
    for name in ("hardware", "roles", "pcie_handoff", "storage"):
        if not isinstance(raw.get(name), dict):
            raise ValueError(f"{name} must be an object")
    roles = raw["roles"]
    _strict_fields(roles, set(ROLE_NAMES), "roles")
    if set(roles) != set(ROLE_NAMES):
        raise ValueError("roles must define prefill and decode")
    config = MirroredP4D4Config(
        hardware=MirroredP4D4Hardware.from_dict(raw["hardware"]),
        prefill=MirroredRoleConfig.from_dict(
            roles[PREFILL_ROLE], role=PREFILL_ROLE),
        decode=MirroredRoleConfig.from_dict(
            roles[DECODE_ROLE], role=DECODE_ROLE),
        pcie_handoff=PcieRootHandoffConfig.from_dict(
            raw["pcie_handoff"]),
        storage=MirroredStoragePolicy.from_dict(raw["storage"]),
    )
    config.validate()
    return config


__all__ = [
    "DECODE_ROLE",
    "GIB",
    "MIN_PREFILL_WORKSPACE_BYTES_PER_CARD",
    "MIB",
    "MirroredP4D4Config",
    "MirroredP4D4Hardware",
    "MirroredRoleConfig",
    "MirroredStoragePolicy",
    "PREFILL_ROLE",
    "PcieRootHandoffConfig",
    "ROLE_NAMES",
    "SCHEMA_VERSION",
    "load_mirrored_p4d4_config",
]

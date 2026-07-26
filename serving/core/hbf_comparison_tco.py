"""Analytical full-system TCO sensitivity for the HBF comparison.

This module deliberately makes economic claims for exactly two purchasable
systems:

* ``tiering_baseline``: two CPU hosts, sixteen H100 cards, host DRAM, and an
  SSD tier.
* ``hbf_proposed``: one CPU/H100 host and one CPU/HBF-GPU host connected by
  RDMA, with LPDDR on every HBF-GPU card and no SSD tier.

The infinite-HBM Oracle may be carried alongside the comparison as a
performance reference.  It is never assigned a TCO because its infinite
capacity is not a physical bill of materials.

All dollar and power values are analytical assumptions, not measured vendor
prices.  In particular, an H100 card is decomposed into GPU logic and its HBM
stack before the HBF-GPU and HBF-media sensitivity ratios are applied.  The
default HBF-GPU logic ratio is 1.0 for H100-class compute; legacy ``npu_*``
field names remain only for artifact compatibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import itertools
import json
import math
from typing import Any, Mapping, Optional, Sequence

from .hbf_full_model_latency import (
    HBFParallelLayout,
    qwen_model_weight_bytes_per_rank,
)


HOURS_PER_YEAR = 8_760.0
SECONDS_PER_HOUR = 3_600.0
KWH_PER_WH = 1.0 / 1_000.0
MILLION_TOKENS = 1_000_000.0
BYTES_PER_GIB = 1_073_741_824
P4D4_CPU_MEMORY_BYTES_PER_HOST = 512_000_000_000
PINNED_GPU_CONFIG_SHA256 = (
    "3b618ffc0e16db6fce79c8086b07802fd4499153542d0843cd05cfa28801393c"
)
PINNED_HBF_CONFIG_SHA256 = (
    "b52a1fe74d288244e07169cd2641ed17a9bde6ca2446c029e9f4c4932e7c598c"
)
PINNED_HBF_WIDE_LPDDR_CONFIG_SHA256 = (
    "6006dda7209d0eb13792b13662f05297c4708ecfdce2ae0909feb787e89e375a"
)

TIERING_SYSTEM_KEY = "tiering_baseline"
PROPOSED_SYSTEM_KEY = "hbf_proposed"
ORACLE_SYSTEM_KEY = "infinite_hbm_oracle"
ECONOMIC_SYSTEM_KEYS = (TIERING_SYSTEM_KEY, PROPOSED_SYSTEM_KEY)
TIERING_POLICY_KEYS = (
    "hbm_lru_recompute",
    "ssd_direct",
    "cpu_ssd",
)
HBF_LAYOUT_KEYS = (
    "dp8",
    "tp4",
    "tp8",
    "hbf_dp8",
    "hbf_tp4",
    "hbf_tp8",
    "hbf_tp8_context",
    "hbf_tp4_wide",
)
HBF_INTRASERVER_FABRIC_KINDS = ("pcie", "ualink")
GOODPUT_METRIC_SCOPES = ("all", "first", "resume")
SCHEDULE_HASH_SEMANTICS = (
    "single_frozen_schedule",
    "ordered_paired_seed_schedule_set_manifest",
)
HBF_POLICY_KEYS = (
    "first_gpu__migration_inflight_resume_gpu__"
    "hbf_ready_resume_hbf__turn_boundary_lpddr_v1",
    "first_gpu__migration_inflight_resume_gpu__"
    "hbf_ready_resume_hbf__turn_boundary_lpddr__"
    "active_prefill_drain_v2",
)
LIVE_COMPACT_OUTPUT_TOKEN_GOODPUT_JSON_PATH = (
    "cells[].performance."
    "offered_normalized_output_token_slo_goodput_per_second"
)
LEGACY_OUTPUT_TOKEN_GOODPUT_JSON_PATH_TEMPLATE = (
    "summary.request_kind_summaries.{scope}."
    "offered_load_normalized_output_token_goodput.value"
)
OUTPUT_TOKEN_GOODPUT_DEFINITION = (
    "offered_session_rate * joint_SLO_pass_output_tokens "
    "/ measured_session_count"
)
MATCHED_OPERATING_POINT_MODE = "matched_single_operating_point"

GOODPUT_SEMANTICS = (
    "offered-load-normalized SLO-good output tokens per second"
)
PRICE_SOURCE_SEMANTICS = (
    "H100 purchase price decomposed by the dated analyst manufacturing-cost "
    "share of HBM, plus explicit HBF sensitivity assumptions; not a vendor "
    "HBF quote"
)
HBF_CAPEX_SENSITIVITY_CENTRAL_AXIS_VALUE = 0.50
HBF_POWER_SENSITIVITY_CENTRAL_AXIS_VALUE = 3.50
ORACLE_EXCLUSION_REASON = (
    "The Oracle assumes infinite HBM capacity, so it is an unphysical "
    "performance reference and its capacity cannot be assigned a finite BOM."
)
POWER_UTILIZATION_SEMANTICS = (
    "Productive token seconds use average_utilization. Energy uses active "
    "power for utilized hours and idle_power_fraction_of_active for the "
    "remaining calendar hours."
)


class HBFComparisonTCOError(ValueError):
    """Raised when a TCO input violates the comparison contract."""


def _require_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HBFComparisonTCOError(
            f"{name} must be a non-empty string")
    return value


def _require_sha256(name: str, value: object) -> str:
    text = _require_nonempty_string(name, value)
    if (
        len(text) != 64
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise HBFComparisonTCOError(
            f"{name} must be a lowercase 64-character SHA-256 hex digest")
    return text


def _require_finite(
        name: str, value: object, *, minimum: Optional[float] = None,
        strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HBFComparisonTCOError(
            f"{name} must be a finite number, got {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        raise HBFComparisonTCOError(
            f"{name} must be a finite number, got {value!r}")
    if strictly_positive and converted <= 0.0:
        raise HBFComparisonTCOError(
            f"{name} must be positive, got {value!r}")
    if minimum is not None and converted < minimum:
        raise HBFComparisonTCOError(
            f"{name} must be at least {minimum}, got {value!r}")
    return converted


def _require_positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise HBFComparisonTCOError(
            f"{name} must be a positive integer, got {value!r}")


def _require_open_unit_interval(name: str, value: float) -> float:
    converted = _require_finite(name, value)
    if converted <= 0.0 or converted >= 1.0:
        raise HBFComparisonTCOError(
            f"{name} must be strictly between 0 and 1, got {value!r}")
    return converted


def _require_closed_unit_interval(name: str, value: object) -> float:
    converted = _require_finite(name, value)
    if converted < 0.0 or converted > 1.0:
        raise HBFComparisonTCOError(
            f"{name} must be in [0, 1], got {value!r}")
    return converted


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise HBFComparisonTCOError(
                    f"JSON map key must be a string, got {key!r}")
            converted[key] = _json_safe(item)
        return converted
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HBFComparisonTCOError(
                f"JSON values must be finite, got {value!r}")
        return value
    raise HBFComparisonTCOError(
        f"value of type {type(value).__name__} is not JSON-safe")


def _stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class JSONSafeDataclass:
    """Mixin that returns a primitive-only map accepted by strict JSON."""

    def to_json_dict(self) -> dict[str, Any]:
        converted = _json_safe(self)
        if not isinstance(converted, dict):
            raise AssertionError("dataclass did not convert to a JSON map")
        json.dumps(converted, allow_nan=False)
        return converted


@dataclass(frozen=True)
class HardwareAnchors(JSONSafeDataclass):
    """Per-component analytical price and power anchors.

    ``h100_card_*`` includes GPU logic plus the HBM stack.  Purchase-basis
    CAPEX applies the analyst HBM share of H100 manufacturing cost to the
    whole-card purchase price.  The absolute component-cost estimate and the
    legacy 30% field remain only for sensitivity cases.  Power keeps a
    separate analytical split because no event-derived component split is
    available here.
    Host anchors exclude DRAM, NICs, fabric, SSDs, and accelerator cards so
    those components remain visible in the BOM.
    """

    h100_card_capex_usd: float = 30_000.0
    h100_card_power_w: float = 700.0
    hbm_capex_accounting_basis: str = (
        "manufacturing_cost_fraction_applied_to_purchase_price")
    h100_manufacturing_cost_usd: float = 3_320.0
    hbm_manufacturing_cost_usd_per_card: float = 1_350.0
    hbm_avoided_capex_usd_per_card: float = 1_350.0
    hbm_capex_fraction_of_h100_card: float = 0.30
    hbm_power_fraction_of_h100_card: float = 0.20
    h100_hbm_capacity_bytes_per_card: int = 80_000_000_000
    hbf_media_controller_capex_usd_per_card: float = 4_500.0
    hbf_media_controller_power_w_per_card: float = 300.0
    h100_purchase_price_source_url: str = (
        "https://siliconanalysts.com/data/ai-chip-costs")
    h100_tdp_source_url: str = (
        "https://www.nvidia.com/en-us/data-center/h100/")
    hbm_component_cost_source_url: str = (
        "https://siliconanalysts.com/data/ai-chip-costs")
    hbf_power_source_url: str = (
        "https://arxiv.org/html/2607.10186#S7.SS6")

    cpu_host_base_capex_usd: float = 20_000.0
    cpu_host_base_power_w: float = 800.0
    host_dram_capex_usd_per_gib: float = 4.0
    host_dram_power_w_per_gib: float = 0.25

    nvme_ssd_capex_usd_per_device: float = 500.0
    nvme_ssd_power_w_per_device: float = 20.0

    baseline_nic_capex_usd: float = 1_500.0
    baseline_nic_power_w: float = 30.0
    baseline_fabric_capex_usd: float = 5_000.0
    baseline_fabric_power_w: float = 100.0

    gpu_intraserver_fabric_capex_usd_per_unit: float = 20_000.0
    gpu_intraserver_fabric_power_w_per_unit: float = 600.0
    hbf_npu_pcie_fabric_capex_usd_per_unit: float = 10_000.0
    hbf_npu_pcie_fabric_power_w_per_unit: float = 350.0
    hbf_npu_ualink_fabric_capex_usd_per_unit: float = 20_000.0
    hbf_npu_ualink_fabric_power_w_per_unit: float = 600.0

    rdma_nic_capex_usd: float = 1_500.0
    rdma_nic_power_w: float = 30.0
    rdma_nic_bandwidth_gbps: float = 50.0
    rdma_fabric_capex_usd: float = 5_000.0
    rdma_fabric_power_w: float = 100.0

    lpddr_capex_usd_per_gib: float = 5.0
    lpddr_power_w_per_gib: float = 0.08

    def __post_init__(self) -> None:
        for name in (
            "h100_card_capex_usd",
            "h100_card_power_w",
            "h100_manufacturing_cost_usd",
            "hbm_manufacturing_cost_usd_per_card",
            "hbm_avoided_capex_usd_per_card",
            "hbf_media_controller_capex_usd_per_card",
            "hbf_media_controller_power_w_per_card",
            "cpu_host_base_capex_usd",
            "cpu_host_base_power_w",
            "host_dram_capex_usd_per_gib",
            "host_dram_power_w_per_gib",
            "nvme_ssd_capex_usd_per_device",
            "nvme_ssd_power_w_per_device",
            "baseline_nic_capex_usd",
            "baseline_nic_power_w",
            "baseline_fabric_capex_usd",
            "baseline_fabric_power_w",
            "gpu_intraserver_fabric_capex_usd_per_unit",
            "gpu_intraserver_fabric_power_w_per_unit",
            "hbf_npu_pcie_fabric_capex_usd_per_unit",
            "hbf_npu_pcie_fabric_power_w_per_unit",
            "hbf_npu_ualink_fabric_capex_usd_per_unit",
            "hbf_npu_ualink_fabric_power_w_per_unit",
            "rdma_nic_capex_usd",
            "rdma_nic_power_w",
            "rdma_fabric_capex_usd",
            "rdma_fabric_power_w",
            "lpddr_capex_usd_per_gib",
            "lpddr_power_w_per_gib",
        ):
            _require_finite(name, getattr(self, name), minimum=0.0)
        _require_finite(
            "rdma_nic_bandwidth_gbps",
            self.rdma_nic_bandwidth_gbps,
            strictly_positive=True,
        )
        if self.hbm_capex_accounting_basis not in {
            "manufacturing_cost_fraction_applied_to_purchase_price",
            "absolute_avoided_purchase_credit",
            "legacy_fraction_of_purchase_price",
        }:
            raise HBFComparisonTCOError(
                "hbm_capex_accounting_basis is unsupported")
        for name in (
            "h100_purchase_price_source_url",
            "h100_tdp_source_url",
            "hbm_component_cost_source_url",
            "hbf_power_source_url",
        ):
            _require_nonempty_string(name, getattr(self, name))
        _require_open_unit_interval(
            "hbm_capex_fraction_of_h100_card",
            self.hbm_capex_fraction_of_h100_card,
        )
        _require_open_unit_interval(
            "hbm_power_fraction_of_h100_card",
            self.hbm_power_fraction_of_h100_card,
        )
        if self.h100_manufacturing_cost_usd <= 0.0:
            raise HBFComparisonTCOError(
                "h100_manufacturing_cost_usd must be positive")
        if self.hbm_manufacturing_cost_usd_per_card <= 0.0:
            raise HBFComparisonTCOError(
                "hbm_manufacturing_cost_usd_per_card must be positive")
        if (
            self.hbm_manufacturing_cost_usd_per_card
            >= self.h100_manufacturing_cost_usd
        ):
            raise HBFComparisonTCOError(
                "HBM manufacturing cost must be below total H100 "
                "manufacturing cost")
        _require_positive_integer(
            "h100_hbm_capacity_bytes_per_card",
            self.h100_hbm_capacity_bytes_per_card,
        )
        if self.hbm_stack_capex_usd_per_card > (
                self.h100_card_capex_usd):
            raise HBFComparisonTCOError(
                "avoided HBM CAPEX cannot exceed H100 purchase CAPEX")

    @property
    def gpu_logic_capex_usd_per_card(self) -> float:
        return (
            self.h100_card_capex_usd
            - self.hbm_stack_capex_usd_per_card
        )

    @property
    def hbm_stack_capex_usd_per_card(self) -> float:
        if self.hbm_capex_accounting_basis == (
                "manufacturing_cost_fraction_applied_to_purchase_price"):
            return (
                self.h100_card_capex_usd
                * self.hbm_manufacturing_cost_fraction_of_h100_card
            )
        if self.hbm_capex_accounting_basis == (
                "absolute_avoided_purchase_credit"):
            return self.hbm_avoided_capex_usd_per_card
        return (
            self.h100_card_capex_usd
            * self.hbm_capex_fraction_of_h100_card)

    @property
    def hbm_manufacturing_cost_fraction_of_h100_card(self) -> float:
        return (
            self.hbm_manufacturing_cost_usd_per_card
            / self.h100_manufacturing_cost_usd
        )

    @property
    def hbm_capex_share_of_h100_purchase_price(self) -> float:
        if self.h100_card_capex_usd == 0.0:
            return 0.0
        return (
            self.hbm_stack_capex_usd_per_card
            / self.h100_card_capex_usd
        )

    @property
    def gpu_logic_power_w_per_card(self) -> float:
        return (
            self.h100_card_power_w
            * (1.0 - self.hbm_power_fraction_of_h100_card)
        )

    @property
    def hbm_stack_power_w_per_card(self) -> float:
        return (
            self.h100_card_power_w
            * self.hbm_power_fraction_of_h100_card
        )


@dataclass(frozen=True)
class DeploymentTopology(JSONSafeDataclass):
    """Physical counts for the baseline and proposed two-host systems."""

    tiering_cpu_hosts: int = 2
    tiering_h100_cards_per_host: int = 8
    tiering_ssd_devices_per_host: int = 8
    tiering_nics_per_host: int = 1
    tiering_fabric_units: int = 1
    tiering_gpu_intraserver_fabric_units_per_host: int = 1

    proposed_gpu_cpu_hosts: int = 1
    proposed_hbf_cpu_hosts: int = 1
    proposed_h100_cards_per_gpu_host: int = 8
    proposed_hbf_npu_cards_per_hbf_host: int = 8
    proposed_nics_per_host: int = 2
    proposed_rdma_fabric_units: int = 1
    proposed_gpu_intraserver_fabric_units_per_host: int = 1
    proposed_hbf_intraserver_fabric_units_per_host: int = 1

    host_dram_gib_per_host: float = (
        P4D4_CPU_MEMORY_BYTES_PER_HOST / BYTES_PER_GIB)
    lpddr_gib_per_hbf_card: float = 64.0

    def __post_init__(self) -> None:
        for name in (
            "tiering_cpu_hosts",
            "tiering_h100_cards_per_host",
            "tiering_ssd_devices_per_host",
            "tiering_nics_per_host",
            "tiering_fabric_units",
            "tiering_gpu_intraserver_fabric_units_per_host",
            "proposed_gpu_cpu_hosts",
            "proposed_hbf_cpu_hosts",
            "proposed_h100_cards_per_gpu_host",
            "proposed_hbf_npu_cards_per_hbf_host",
            "proposed_nics_per_host",
            "proposed_rdma_fabric_units",
            "proposed_gpu_intraserver_fabric_units_per_host",
            "proposed_hbf_intraserver_fabric_units_per_host",
        ):
            _require_positive_integer(name, getattr(self, name))
        _require_finite(
            "host_dram_gib_per_host",
            self.host_dram_gib_per_host,
            strictly_positive=True,
        )
        _require_finite(
            "lpddr_gib_per_hbf_card",
            self.lpddr_gib_per_hbf_card,
            strictly_positive=True,
        )
        if self.proposed_gpu_cpu_hosts != 1:
            raise HBFComparisonTCOError(
                "the proposed comparison requires exactly one GPU CPU host")
        if self.proposed_hbf_cpu_hosts != 1:
            raise HBFComparisonTCOError(
                "the proposed comparison requires exactly one HBF CPU host")

    @property
    def tiering_h100_cards(self) -> int:
        return (
            self.tiering_cpu_hosts
            * self.tiering_h100_cards_per_host
        )

    @property
    def tiering_ssd_devices(self) -> int:
        return (
            self.tiering_cpu_hosts
            * self.tiering_ssd_devices_per_host
        )

    @property
    def tiering_nics(self) -> int:
        return self.tiering_cpu_hosts * self.tiering_nics_per_host

    @property
    def tiering_gpu_intraserver_fabric_units(self) -> int:
        return (
            self.tiering_cpu_hosts
            * self.tiering_gpu_intraserver_fabric_units_per_host
        )

    @property
    def proposed_cpu_hosts(self) -> int:
        return self.proposed_gpu_cpu_hosts + self.proposed_hbf_cpu_hosts

    @property
    def proposed_h100_cards(self) -> int:
        return (
            self.proposed_gpu_cpu_hosts
            * self.proposed_h100_cards_per_gpu_host
        )

    @property
    def proposed_hbf_npu_cards(self) -> int:
        return (
            self.proposed_hbf_cpu_hosts
            * self.proposed_hbf_npu_cards_per_hbf_host
        )

    @property
    def proposed_nics(self) -> int:
        return self.proposed_cpu_hosts * self.proposed_nics_per_host

    @property
    def proposed_gpu_intraserver_fabric_units(self) -> int:
        return (
            self.proposed_gpu_cpu_hosts
            * self.proposed_gpu_intraserver_fabric_units_per_host
        )

    @property
    def proposed_hbf_intraserver_fabric_units(self) -> int:
        return (
            self.proposed_hbf_cpu_hosts
            * self.proposed_hbf_intraserver_fabric_units_per_host
        )

    @property
    def proposed_lpddr_gib(self) -> float:
        return (
            self.proposed_hbf_npu_cards
            * self.lpddr_gib_per_hbf_card
        )


@dataclass(frozen=True)
class EvaluationAssumptions(JSONSafeDataclass):
    lifetime_years: float = 5.0
    average_utilization: float = 0.70
    idle_power_fraction_of_active: float = 0.0
    pue: float = 1.20
    electricity_usd_per_kwh: float = 0.10

    def __post_init__(self) -> None:
        _require_finite(
            "lifetime_years", self.lifetime_years,
            strictly_positive=True)
        utilization = _require_finite(
            "average_utilization", self.average_utilization)
        if utilization <= 0.0 or utilization > 1.0:
            raise HBFComparisonTCOError(
                "average_utilization must be in (0, 1]")
        _require_closed_unit_interval(
            "idle_power_fraction_of_active",
            self.idle_power_fraction_of_active,
        )
        pue = _require_finite("pue", self.pue)
        if pue < 1.0:
            raise HBFComparisonTCOError("pue must be at least 1")
        _require_finite(
            "electricity_usd_per_kwh",
            self.electricity_usd_per_kwh,
            minimum=0.0,
        )

    @property
    def lifetime_calendar_hours(self) -> float:
        return self.lifetime_years * HOURS_PER_YEAR

    @property
    def lifetime_loaded_hours(self) -> float:
        return self.lifetime_calendar_hours * self.average_utilization

    @property
    def average_power_load_factor(self) -> float:
        return (
            self.average_utilization
            + (1.0 - self.average_utilization)
            * self.idle_power_fraction_of_active
        )

    @property
    def lifetime_powered_equivalent_full_load_hours(self) -> float:
        return (
            self.lifetime_calendar_hours
            * self.average_power_load_factor
        )

    @property
    def lifetime_loaded_seconds(self) -> float:
        return self.lifetime_loaded_hours * SECONDS_PER_HOUR


def _validated_axis(
        name: str, values: Sequence[float], *,
        cheaper_than_one: bool = False,
        at_most_one: bool = False,
        greater_than_one: bool = False) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not values:
        raise HBFComparisonTCOError(f"{name} must be a non-empty sequence")
    converted = tuple(
        _require_finite(f"{name}[{index}]", value, strictly_positive=True)
        for index, value in enumerate(values)
    )
    if len(set(converted)) != len(converted):
        raise HBFComparisonTCOError(f"{name} must not contain duplicates")
    if cheaper_than_one and any(value >= 1.0 for value in converted):
        raise HBFComparisonTCOError(
            f"every {name} value must be below 1")
    if at_most_one and any(value > 1.0 for value in converted):
        raise HBFComparisonTCOError(
            f"every {name} value must be at most 1")
    if greater_than_one and any(value <= 1.0 for value in converted):
        raise HBFComparisonTCOError(
            f"every {name} value must be above 1")
    return converted


@dataclass(frozen=True)
class SensitivityAxes(JSONSafeDataclass):
    """Cartesian component-ratio sensitivity axes.

    The ``npu_logic_*`` names are retained for report compatibility.  Their
    default singleton value is the full H100 GPU-logic anchor.  The HBF field
    names are also retained for compatibility, but their values are normalized
    around the independent HBF media/controller anchors: 0.50 means 1.0x the
    CAPEX anchor and 3.50 means 1.0x the power anchor.
    """

    npu_logic_capex_ratios_to_gpu_logic: tuple[float, ...] = (
        1.00,)
    hbf_subsystem_capex_ratios_to_hbm_stack: tuple[float, ...] = (
        0.25, 0.50, 0.75)
    npu_logic_power_ratios_to_gpu_logic: tuple[float, ...] = (
        1.00,)
    hbf_subsystem_power_ratios_to_hbm_stack: tuple[float, ...] = (
        3.0, 3.5, 4.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "npu_logic_capex_ratios_to_gpu_logic",
            _validated_axis(
                "npu_logic_capex_ratios_to_gpu_logic",
                self.npu_logic_capex_ratios_to_gpu_logic,
                at_most_one=True,
            ),
        )
        object.__setattr__(
            self,
            "hbf_subsystem_capex_ratios_to_hbm_stack",
            _validated_axis(
                "hbf_subsystem_capex_ratios_to_hbm_stack",
                self.hbf_subsystem_capex_ratios_to_hbm_stack,
                cheaper_than_one=True,
            ),
        )
        object.__setattr__(
            self,
            "npu_logic_power_ratios_to_gpu_logic",
            _validated_axis(
                "npu_logic_power_ratios_to_gpu_logic",
                self.npu_logic_power_ratios_to_gpu_logic,
                at_most_one=True,
            ),
        )
        object.__setattr__(
            self,
            "hbf_subsystem_power_ratios_to_hbm_stack",
            _validated_axis(
                "hbf_subsystem_power_ratios_to_hbm_stack",
                self.hbf_subsystem_power_ratios_to_hbm_stack,
                greater_than_one=True,
            ),
        )

    @property
    def cartesian_size(self) -> int:
        return (
            len(self.npu_logic_capex_ratios_to_gpu_logic)
            * len(self.hbf_subsystem_capex_ratios_to_hbm_stack)
            * len(self.npu_logic_power_ratios_to_gpu_logic)
            * len(self.hbf_subsystem_power_ratios_to_hbm_stack)
        )


@dataclass(frozen=True)
class SensitivityPoint(JSONSafeDataclass):
    npu_logic_capex_ratio_to_gpu_logic: float
    hbf_subsystem_capex_ratio_to_hbm_stack: float
    npu_logic_power_ratio_to_gpu_logic: float
    hbf_subsystem_power_ratio_to_hbm_stack: float

    def __post_init__(self) -> None:
        _validated_axis(
            "npu_logic_capex_ratio_to_gpu_logic",
            (self.npu_logic_capex_ratio_to_gpu_logic,),
            at_most_one=True,
        )
        _validated_axis(
            "hbf_subsystem_capex_ratio_to_hbm_stack",
            (self.hbf_subsystem_capex_ratio_to_hbm_stack,),
            cheaper_than_one=True,
        )
        _validated_axis(
            "npu_logic_power_ratio_to_gpu_logic",
            (self.npu_logic_power_ratio_to_gpu_logic,),
            at_most_one=True,
        )
        _validated_axis(
            "hbf_subsystem_power_ratio_to_hbm_stack",
            (self.hbf_subsystem_power_ratio_to_hbm_stack,),
            greater_than_one=True,
        )

    @property
    def key(self) -> str:
        def label(value: float) -> str:
            return (
                format(value, ".17g")
                .replace("-", "m")
                .replace("+", "")
                .replace(".", "p")
            )

        return "__".join((
            f"npu_capex_{label(self.npu_logic_capex_ratio_to_gpu_logic)}",
            f"hbf_capex_{label(self.hbf_subsystem_capex_ratio_to_hbm_stack)}",
            f"npu_power_{label(self.npu_logic_power_ratio_to_gpu_logic)}",
            f"hbf_power_{label(self.hbf_subsystem_power_ratio_to_hbm_stack)}",
        ))

    @property
    def hbf_media_controller_capex_multiplier(self) -> float:
        return (
            self.hbf_subsystem_capex_ratio_to_hbm_stack
            / HBF_CAPEX_SENSITIVITY_CENTRAL_AXIS_VALUE
        )

    @property
    def hbf_media_controller_power_multiplier(self) -> float:
        return (
            self.hbf_subsystem_power_ratio_to_hbm_stack
            / HBF_POWER_SENSITIVITY_CENTRAL_AXIS_VALUE
        )


@dataclass(frozen=True)
class HBFHardwareVariant(JSONSafeDataclass):
    """Effective HBF-server hardware and its explicit costing multipliers."""

    variant_key: str
    hbf_config_sha256: str
    card_count: int
    hbf_capacity_bytes_per_card: int
    hbf_capacity_ratio_to_hbm: float
    intra_fabric_kind: str
    intra_fabric_bandwidth_gbps_per_card: float
    intra_fabric_capex_multiplier: float
    intra_fabric_power_multiplier: float
    lpddr_reference_bandwidth_gbps_per_card: float
    lpddr_effective_bandwidth_gbps_per_card: float
    lpddr_capacity_gib_per_card: float
    lpddr_bandwidth_multiplier: float
    lpddr_capex_multiplier: float
    lpddr_power_multiplier: float
    rdma_bandwidth_gbps: float
    rdma_one_way_latency_us: float
    rdma_capex_multiplier: float
    rdma_power_multiplier: float
    cost_power_assumption: str

    def __post_init__(self) -> None:
        _require_nonempty_string("variant_key", self.variant_key)
        _require_sha256("hbf_config_sha256", self.hbf_config_sha256)
        _require_positive_integer("card_count", self.card_count)
        _require_positive_integer(
            "hbf_capacity_bytes_per_card",
            self.hbf_capacity_bytes_per_card,
        )
        if (
            not isinstance(self.intra_fabric_kind, str)
            or self.intra_fabric_kind not in HBF_INTRASERVER_FABRIC_KINDS
        ):
            raise HBFComparisonTCOError(
                "intra_fabric_kind must be one of "
                f"{HBF_INTRASERVER_FABRIC_KINDS!r}")
        for name in (
            "intra_fabric_bandwidth_gbps_per_card",
            "hbf_capacity_ratio_to_hbm",
            "intra_fabric_capex_multiplier",
            "intra_fabric_power_multiplier",
            "lpddr_reference_bandwidth_gbps_per_card",
            "lpddr_effective_bandwidth_gbps_per_card",
            "lpddr_capacity_gib_per_card",
            "lpddr_bandwidth_multiplier",
            "lpddr_capex_multiplier",
            "lpddr_power_multiplier",
            "rdma_bandwidth_gbps",
            "rdma_one_way_latency_us",
            "rdma_capex_multiplier",
            "rdma_power_multiplier",
        ):
            _require_finite(
                name, getattr(self, name), strictly_positive=True)
        expected_lpddr_bandwidth = (
            self.lpddr_reference_bandwidth_gbps_per_card
            * self.lpddr_bandwidth_multiplier
        )
        if not math.isclose(
                self.lpddr_effective_bandwidth_gbps_per_card,
                expected_lpddr_bandwidth,
                rel_tol=1e-12,
                abs_tol=1e-9):
            raise HBFComparisonTCOError(
                "LPDDR effective bandwidth must equal reference bandwidth "
                "times lpddr_bandwidth_multiplier")
        if self.lpddr_bandwidth_multiplier >= 1.0:
            if (
                self.lpddr_capex_multiplier < 1.0
                or self.lpddr_power_multiplier < 1.0
            ):
                raise HBFComparisonTCOError(
                    "a baseline-or-wider LPDDR variant cannot use a cost or "
                    "power multiplier below 1")
        _require_nonempty_string(
            "cost_power_assumption", self.cost_power_assumption)


DEFAULT_HBF_HARDWARE_VARIANT = HBFHardwareVariant(
    variant_key="full_model_8card_pcie_lpddr204p8_v1",
    hbf_config_sha256=PINNED_HBF_CONFIG_SHA256,
    card_count=8,
    hbf_capacity_bytes_per_card=1_280_000_000_000,
    hbf_capacity_ratio_to_hbm=16.0,
    intra_fabric_kind="pcie",
    intra_fabric_bandwidth_gbps_per_card=50.0,
    intra_fabric_capex_multiplier=1.0,
    intra_fabric_power_multiplier=1.0,
    lpddr_reference_bandwidth_gbps_per_card=204.8,
    lpddr_effective_bandwidth_gbps_per_card=204.8,
    lpddr_capacity_gib_per_card=64.0,
    lpddr_bandwidth_multiplier=1.0,
    lpddr_capex_multiplier=1.0,
    lpddr_power_multiplier=1.0,
    rdma_bandwidth_gbps=80.0,
    rdma_one_way_latency_us=10.0,
    rdma_capex_multiplier=1.0,
    rdma_power_multiplier=1.0,
    cost_power_assumption=(
        "Pinned full_model_8card_server.json PCIe and 204.8 GB/s LPDDR "
        "anchors; no width multiplier."
    ),
)

WIDE_LPDDR_HBF_HARDWARE_VARIANT = HBFHardwareVariant(
    variant_key="full_model_8card_pcie_lpddr409p6_v1",
    hbf_config_sha256=PINNED_HBF_WIDE_LPDDR_CONFIG_SHA256,
    card_count=8,
    hbf_capacity_bytes_per_card=1_280_000_000_000,
    hbf_capacity_ratio_to_hbm=16.0,
    intra_fabric_kind="pcie",
    intra_fabric_bandwidth_gbps_per_card=50.0,
    intra_fabric_capex_multiplier=1.0,
    intra_fabric_power_multiplier=1.0,
    lpddr_reference_bandwidth_gbps_per_card=204.8,
    lpddr_effective_bandwidth_gbps_per_card=409.6,
    lpddr_capacity_gib_per_card=64.0,
    lpddr_bandwidth_multiplier=2.0,
    lpddr_capex_multiplier=1.5,
    lpddr_power_multiplier=1.75,
    rdma_bandwidth_gbps=80.0,
    rdma_one_way_latency_us=10.0,
    rdma_capex_multiplier=1.0,
    rdma_power_multiplier=1.0,
    cost_power_assumption=(
        "Pinned full_model_8card_server_wide_lpddr.json doubles LPDDR "
        "bandwidth from 204.8 to 409.6 GB/s per card. Analytical TCO assumes "
        "1.5x LPDDR CAPEX and 1.75x LPDDR power; all other hardware anchors "
        "are unchanged."
    ),
)


def sensitivity_points(
        axes: SensitivityAxes = SensitivityAxes(),
) -> tuple[SensitivityPoint, ...]:
    return tuple(
        SensitivityPoint(*values)
        for values in itertools.product(
            axes.npu_logic_capex_ratios_to_gpu_logic,
            axes.hbf_subsystem_capex_ratios_to_hbm_stack,
            axes.npu_logic_power_ratios_to_gpu_logic,
            axes.hbf_subsystem_power_ratios_to_hbm_stack,
        )
    )


@dataclass(frozen=True)
class BOMLine(JSONSafeDataclass):
    component_key: str
    component_label: str
    unit: str
    quantity: float
    unit_capex_usd: float
    unit_it_power_w: float
    capex_usd: float
    it_power_w: float
    assumption: str

    def __post_init__(self) -> None:
        if not isinstance(self.component_key, str) or not self.component_key:
            raise HBFComparisonTCOError(
                "component_key must be a non-empty string")
        if not isinstance(self.component_label, str) or not self.component_label:
            raise HBFComparisonTCOError(
                "component_label must be a non-empty string")
        if not isinstance(self.unit, str) or not self.unit:
            raise HBFComparisonTCOError("unit must be a non-empty string")
        quantity = _require_finite(
            "quantity", self.quantity, minimum=0.0)
        unit_capex = _require_finite(
            "unit_capex_usd", self.unit_capex_usd, minimum=0.0)
        unit_power = _require_finite(
            "unit_it_power_w", self.unit_it_power_w, minimum=0.0)
        capex = _require_finite(
            "capex_usd", self.capex_usd, minimum=0.0)
        power = _require_finite(
            "it_power_w", self.it_power_w, minimum=0.0)
        if not math.isclose(
                capex, quantity * unit_capex,
                rel_tol=1e-12, abs_tol=1e-9):
            raise HBFComparisonTCOError(
                f"{self.component_key}: capex total does not equal "
                "quantity times unit capex")
        if not math.isclose(
                power, quantity * unit_power,
                rel_tol=1e-12, abs_tol=1e-9):
            raise HBFComparisonTCOError(
                f"{self.component_key}: power total does not equal "
                "quantity times unit power")
        if not isinstance(self.assumption, str) or not self.assumption:
            raise HBFComparisonTCOError(
                "assumption must be a non-empty string")


def _bom_line(
        key: str, label: str, unit: str, quantity: float,
        unit_capex_usd: float, unit_power_w: float,
        assumption: str) -> BOMLine:
    return BOMLine(
        component_key=key,
        component_label=label,
        unit=unit,
        quantity=float(quantity),
        unit_capex_usd=float(unit_capex_usd),
        unit_it_power_w=float(unit_power_w),
        capex_usd=float(quantity) * float(unit_capex_usd),
        it_power_w=float(quantity) * float(unit_power_w),
        assumption=assumption,
    )


@dataclass(frozen=True)
class SystemCost(JSONSafeDataclass):
    system_key: str
    system_label: str
    physical_description: str
    bom: tuple[BOMLine, ...]
    capex_usd: float
    it_power_w: float
    facility_power_w: float
    lifetime_it_energy_kwh: float
    lifetime_facility_energy_kwh: float
    lifetime_electricity_opex_usd: float
    lifetime_tco_usd: float
    pue: float
    average_utilization: float
    idle_power_fraction_of_active: float
    lifetime_years: float
    power_utilization_semantics: str = POWER_UTILIZATION_SEMANTICS
    price_source_semantics: str = PRICE_SOURCE_SEMANTICS
    excluded_costs: tuple[str, ...] = (
        "labor", "maintenance", "financing", "taxes")

    def __post_init__(self) -> None:
        if self.system_key not in ECONOMIC_SYSTEM_KEYS:
            raise HBFComparisonTCOError(
                f"{self.system_key!r} is not an economic comparison system")
        if not self.bom:
            raise HBFComparisonTCOError("bom must not be empty")
        keys = [line.component_key for line in self.bom]
        if len(set(keys)) != len(keys):
            raise HBFComparisonTCOError(
                "BOM component keys must be unique")
        for name in (
            "capex_usd",
            "it_power_w",
            "facility_power_w",
            "lifetime_it_energy_kwh",
            "lifetime_facility_energy_kwh",
            "lifetime_electricity_opex_usd",
            "lifetime_tco_usd",
        ):
            _require_finite(name, getattr(self, name), minimum=0.0)
        expected_capex = math.fsum(line.capex_usd for line in self.bom)
        expected_power = math.fsum(line.it_power_w for line in self.bom)
        if not math.isclose(
                self.capex_usd, expected_capex,
                rel_tol=1e-12, abs_tol=1e-8):
            raise HBFComparisonTCOError("capex does not match the BOM")
        if not math.isclose(
                self.it_power_w, expected_power,
                rel_tol=1e-12, abs_tol=1e-8):
            raise HBFComparisonTCOError("IT power does not match the BOM")
        if _require_finite("pue", self.pue) < 1.0:
            raise HBFComparisonTCOError("pue must be at least 1")
        utilization = _require_finite(
            "average_utilization", self.average_utilization)
        if utilization <= 0.0 or utilization > 1.0:
            raise HBFComparisonTCOError(
                "average_utilization must be in (0, 1]")
        _require_closed_unit_interval(
            "idle_power_fraction_of_active",
            self.idle_power_fraction_of_active,
        )
        _require_finite(
            "lifetime_years", self.lifetime_years,
            strictly_positive=True)

    def component(self, component_key: str) -> BOMLine:
        for line in self.bom:
            if line.component_key == component_key:
                return line
        raise KeyError(component_key)


def _finalize_cost(
        system_key: str,
        system_label: str,
        physical_description: str,
        bom: Sequence[BOMLine],
        evaluation: EvaluationAssumptions,
) -> SystemCost:
    lines = tuple(bom)
    capex = math.fsum(line.capex_usd for line in lines)
    it_power = math.fsum(line.it_power_w for line in lines)
    facility_power = it_power * evaluation.pue
    it_energy = (
        it_power
        * evaluation.lifetime_powered_equivalent_full_load_hours
        * KWH_PER_WH
    )
    facility_energy = (
        facility_power
        * evaluation.lifetime_powered_equivalent_full_load_hours
        * KWH_PER_WH
    )
    electricity = (
        facility_energy * evaluation.electricity_usd_per_kwh)
    return SystemCost(
        system_key=system_key,
        system_label=system_label,
        physical_description=physical_description,
        bom=lines,
        capex_usd=capex,
        it_power_w=it_power,
        facility_power_w=facility_power,
        lifetime_it_energy_kwh=it_energy,
        lifetime_facility_energy_kwh=facility_energy,
        lifetime_electricity_opex_usd=electricity,
        lifetime_tco_usd=capex + electricity,
        pue=evaluation.pue,
        average_utilization=evaluation.average_utilization,
        idle_power_fraction_of_active=(
            evaluation.idle_power_fraction_of_active),
        lifetime_years=evaluation.lifetime_years,
    )


def tiering_baseline_cost(
        anchors: HardwareAnchors = HardwareAnchors(),
        topology: DeploymentTopology = DeploymentTopology(),
        evaluation: EvaluationAssumptions = EvaluationAssumptions(),
) -> SystemCost:
    """Price the physical two-server 4P4D+CPU+SSD tiering baseline."""

    bom = (
        _bom_line(
            "cpu_host_base",
            "CPU server base (accelerator, DRAM, NIC excluded)",
            "host",
            topology.tiering_cpu_hosts,
            anchors.cpu_host_base_capex_usd,
            anchors.cpu_host_base_power_w,
            "The same CPU-host base anchor is used for both deployments.",
        ),
        _bom_line(
            "host_dram",
            "Host DRAM",
            "GiB",
            topology.tiering_cpu_hosts * topology.host_dram_gib_per_host,
            anchors.host_dram_capex_usd_per_gib,
            anchors.host_dram_power_w_per_gib,
            (
                "Host DRAM is priced separately from the CPU-host base; "
                "the default capacity equals the P4D4 configuration's "
                "512e9 bytes per server."
            ),
        ),
        _bom_line(
            "h100_gpu_logic",
            "H100 GPU logic excluding HBM",
            "card",
            topology.tiering_h100_cards,
            anchors.gpu_logic_capex_usd_per_card,
            anchors.gpu_logic_power_w_per_card,
            "Derived from the analytical H100 card anchor.",
        ),
        _bom_line(
            "h100_hbm_stack",
            "Complete H100 HBM stack",
            "card",
            topology.tiering_h100_cards,
            anchors.hbm_stack_capex_usd_per_card,
            anchors.hbm_stack_power_w_per_card,
            (
                "Derived from the analytical H100 card anchor at "
                f"{anchors.h100_hbm_capacity_bytes_per_card} bytes per card."
            ),
        ),
        _bom_line(
            "gpu_intraserver_fabric",
            "GPU-host NVSwitch/NVLink fabric",
            "host fabric unit",
            topology.tiering_gpu_intraserver_fabric_units,
            anchors.gpu_intraserver_fabric_capex_usd_per_unit,
            anchors.gpu_intraserver_fabric_power_w_per_unit,
            (
                "One explicit intra-server GPU fabric allocation per "
                "baseline GPU host; distinct from external networking."
            ),
        ),
        _bom_line(
            "nvme_ssd_tier",
            "NVMe SSD tier",
            "device",
            topology.tiering_ssd_devices,
            anchors.nvme_ssd_capex_usd_per_device,
            anchors.nvme_ssd_power_w_per_device,
            "Default topology uses eight SSDs in each baseline server.",
        ),
        _bom_line(
            "baseline_network_nic",
            "Baseline host network NIC",
            "NIC",
            topology.tiering_nics,
            anchors.baseline_nic_capex_usd,
            anchors.baseline_nic_power_w,
            "Networking is explicit rather than hidden in host cost.",
        ),
        _bom_line(
            "baseline_network_fabric",
            "Baseline common network fabric allocation",
            "fabric unit",
            topology.tiering_fabric_units,
            anchors.baseline_fabric_capex_usd,
            anchors.baseline_fabric_power_w,
            "A system-level share of common network infrastructure.",
        ),
    )
    return _finalize_cost(
        TIERING_SYSTEM_KEY,
        "Two-server 4P4D GPU tiering baseline",
        (
            f"{topology.tiering_cpu_hosts} CPU hosts, "
            f"{topology.tiering_h100_cards} H100 cards, "
            f"{topology.tiering_ssd_devices} SSDs, host DRAM, two GPU "
            "intra-server fabrics, external NICs, and common network fabric"
        ),
        bom,
        evaluation,
    )


def proposed_hbf_cost(
        point: SensitivityPoint,
        anchors: HardwareAnchors = HardwareAnchors(),
        topology: DeploymentTopology = DeploymentTopology(),
        evaluation: EvaluationAssumptions = EvaluationAssumptions(),
        hbf_hardware_variant: HBFHardwareVariant = (
            DEFAULT_HBF_HARDWARE_VARIANT),
) -> SystemCost:
    """Price one H100 host plus one eight-card HBF-GPU host."""

    if not isinstance(point, SensitivityPoint):
        raise HBFComparisonTCOError(
            "point must be a SensitivityPoint")
    if not isinstance(hbf_hardware_variant, HBFHardwareVariant):
        raise HBFComparisonTCOError(
            "hbf_hardware_variant must be HBFHardwareVariant")
    if (
        topology.proposed_hbf_npu_cards
        != hbf_hardware_variant.card_count
    ):
        raise HBFComparisonTCOError(
            "HBF topology card count mismatches the effective hardware "
            "variant")
    if not math.isclose(
            topology.lpddr_gib_per_hbf_card,
            hbf_hardware_variant.lpddr_capacity_gib_per_card,
            rel_tol=1e-12,
            abs_tol=1e-9):
        raise HBFComparisonTCOError(
            "LPDDR topology capacity mismatches the effective hardware "
            "variant")
    expected_hbf_capacity_ratio = (
        hbf_hardware_variant.hbf_capacity_bytes_per_card
        / anchors.h100_hbm_capacity_bytes_per_card
    )
    if not math.isclose(
            hbf_hardware_variant.hbf_capacity_ratio_to_hbm,
            expected_hbf_capacity_ratio,
            rel_tol=1e-12,
            abs_tol=1e-12):
        raise HBFComparisonTCOError(
            "HBF capacity ratio mismatches HBF and HBM capacities")
    required_rdma_nics_per_host = math.ceil(
        hbf_hardware_variant.rdma_bandwidth_gbps
        / anchors.rdma_nic_bandwidth_gbps
    )
    if topology.proposed_nics_per_host < required_rdma_nics_per_host:
        raise HBFComparisonTCOError(
            "proposed RDMA NIC count cannot provide the configured "
            "GPU-HBF bandwidth")
    hbf_gpu_logic_capex = (
        anchors.gpu_logic_capex_usd_per_card
        * point.npu_logic_capex_ratio_to_gpu_logic
    )
    hbf_subsystem_capex = (
        anchors.hbf_media_controller_capex_usd_per_card
        * point.hbf_media_controller_capex_multiplier
    )
    hbf_gpu_logic_power = (
        anchors.gpu_logic_power_w_per_card
        * point.npu_logic_power_ratio_to_gpu_logic
    )
    hbf_subsystem_power = (
        anchors.hbf_media_controller_power_w_per_card
        * point.hbf_media_controller_power_multiplier
    )
    if not (
        hbf_subsystem_capex
        / hbf_hardware_variant.hbf_capacity_ratio_to_hbm
        < anchors.hbm_stack_capex_usd_per_card
    ):
        raise HBFComparisonTCOError(
            "HBF media/controller CAPEX must be cheaper than HBM after "
            "normalizing for installed capacity")
    if hbf_hardware_variant.intra_fabric_kind == "pcie":
        hbf_fabric_capex = (
            anchors.hbf_npu_pcie_fabric_capex_usd_per_unit)
        hbf_fabric_power = (
            anchors.hbf_npu_pcie_fabric_power_w_per_unit)
    else:
        hbf_fabric_capex = (
            anchors.hbf_npu_ualink_fabric_capex_usd_per_unit)
        hbf_fabric_power = (
            anchors.hbf_npu_ualink_fabric_power_w_per_unit)
    hbf_fabric_capex *= (
        hbf_hardware_variant.intra_fabric_capex_multiplier)
    hbf_fabric_power *= (
        hbf_hardware_variant.intra_fabric_power_multiplier)
    lpddr_capex_per_gib = (
        anchors.lpddr_capex_usd_per_gib
        * hbf_hardware_variant.lpddr_capex_multiplier
    )
    lpddr_power_per_gib = (
        anchors.lpddr_power_w_per_gib
        * hbf_hardware_variant.lpddr_power_multiplier
    )
    rdma_nic_capex = (
        anchors.rdma_nic_capex_usd
        * hbf_hardware_variant.rdma_capex_multiplier
    )
    rdma_nic_power = (
        anchors.rdma_nic_power_w
        * hbf_hardware_variant.rdma_power_multiplier
    )
    rdma_fabric_capex = (
        anchors.rdma_fabric_capex_usd
        * hbf_hardware_variant.rdma_capex_multiplier
    )
    rdma_fabric_power = (
        anchors.rdma_fabric_power_w
        * hbf_hardware_variant.rdma_power_multiplier
    )

    bom = (
        _bom_line(
            "cpu_host_base",
            "CPU server base (accelerator, DRAM, NIC excluded)",
            "host",
            topology.proposed_cpu_hosts,
            anchors.cpu_host_base_capex_usd,
            anchors.cpu_host_base_power_w,
            "One GPU host and one HBF-GPU host use the same host anchor.",
        ),
        _bom_line(
            "host_dram",
            "Host DRAM",
            "GiB",
            topology.proposed_cpu_hosts * topology.host_dram_gib_per_host,
            anchors.host_dram_capex_usd_per_gib,
            anchors.host_dram_power_w_per_gib,
            (
                "Both proposed hosts receive the same host-DRAM allocation; "
                "the default capacity equals the P4D4 configuration's "
                "512e9 bytes per server."
            ),
        ),
        _bom_line(
            "h100_gpu_logic",
            "H100 GPU logic excluding HBM",
            "card",
            topology.proposed_h100_cards,
            anchors.gpu_logic_capex_usd_per_card,
            anchors.gpu_logic_power_w_per_card,
            "Derived from the analytical H100 card anchor.",
        ),
        _bom_line(
            "h100_hbm_stack",
            "Complete H100 HBM stack",
            "card",
            topology.proposed_h100_cards,
            anchors.hbm_stack_capex_usd_per_card,
            anchors.hbm_stack_power_w_per_card,
            (
                "The finite-HBM GPU server retains eight complete H100 "
                "cards at "
                f"{anchors.h100_hbm_capacity_bytes_per_card} bytes per card."
            ),
        ),
        _bom_line(
            "gpu_intraserver_fabric",
            "GPU-host NVSwitch/NVLink fabric",
            "host fabric unit",
            topology.proposed_gpu_intraserver_fabric_units,
            anchors.gpu_intraserver_fabric_capex_usd_per_unit,
            anchors.gpu_intraserver_fabric_power_w_per_unit,
            (
                "The GPU host's accelerator fabric is explicit and is "
                "distinct from inter-host RDMA."
            ),
        ),
        _bom_line(
            "hbf_npu_logic",
            "H100-class HBF-card GPU logic excluding HBF and LPDDR",
            "card",
            topology.proposed_hbf_npu_cards,
            hbf_gpu_logic_capex,
            hbf_gpu_logic_power,
            (
                "The default component ratio is 1.0x H100 GPU logic. "
                "The legacy hbf_npu_logic key and npu_logic ratio field "
                "names are retained for artifact compatibility; HBM is "
                "excluded because HBF media is priced separately."
            ),
        ),
        _bom_line(
            "hbf_media_controller_subsystem",
            "Complete HBF media/controller subsystem",
            "card",
            topology.proposed_hbf_npu_cards,
            hbf_subsystem_capex,
            hbf_subsystem_power,
            (
                "The full HBF media/controller subsystem uses an independent "
                f"${anchors.hbf_media_controller_capex_usd_per_card:,.0f} "
                "per-card CAPEX anchor and "
                f"{anchors.hbf_media_controller_power_w_per_card:,.0f} W "
                "power anchor. Compatibility sensitivity-axis values are "
                "normalized around 0.50=1.0x CAPEX and 3.50=1.0x power. "
                f"Installed capacity is "
                f"{hbf_hardware_variant.hbf_capacity_bytes_per_card} bytes "
                "per card, or "
                f"{hbf_hardware_variant.hbf_capacity_ratio_to_hbm:g}x the "
                "H100 HBM anchor."
            ),
        ),
        _bom_line(
            "hbf_npu_intraserver_fabric",
            (
                "HBF-GPU-host "
                f"{hbf_hardware_variant.intra_fabric_kind.upper()} fabric"
            ),
            "host fabric unit",
            topology.proposed_hbf_intraserver_fabric_units,
            hbf_fabric_capex,
            hbf_fabric_power,
            (
                f"{hbf_hardware_variant.intra_fabric_bandwidth_gbps_per_card:g}"
                " GB/s per card; base kind-specific cost/power multiplied "
                f"by {hbf_hardware_variant.intra_fabric_capex_multiplier:g}x"
                " CAPEX and "
                f"{hbf_hardware_variant.intra_fabric_power_multiplier:g}x "
                "power. This is distinct from GPU fabric and inter-host RDMA."
            ),
        ),
        _bom_line(
            "hbf_card_lpddr",
            "HBF-card LPDDR",
            "GiB",
            topology.proposed_lpddr_gib,
            lpddr_capex_per_gib,
            lpddr_power_per_gib,
            (
                f"{topology.lpddr_gib_per_hbf_card:g} GiB per HBF-GPU "
                "card at "
                f"{hbf_hardware_variant.lpddr_effective_bandwidth_gbps_per_card:g}"
                " GB/s per card "
                f"({hbf_hardware_variant.lpddr_bandwidth_multiplier:g}x "
                "reference bandwidth), with "
                f"{hbf_hardware_variant.lpddr_capex_multiplier:g}x CAPEX "
                "and "
                f"{hbf_hardware_variant.lpddr_power_multiplier:g}x power. "
                f"{hbf_hardware_variant.cost_power_assumption}"
            ),
        ),
        _bom_line(
            "nvme_ssd_tier",
            "NVMe SSD tier",
            "device",
            0,
            anchors.nvme_ssd_capex_usd_per_device,
            anchors.nvme_ssd_power_w_per_device,
            "The proposed comparison has no SSD tier.",
        ),
        _bom_line(
            "rdma_network_nic",
            "GPU-HBF RDMA NIC",
            "NIC",
            topology.proposed_nics,
            rdma_nic_capex,
            rdma_nic_power,
            (
                "The GPU and HBF hosts each have enough explicitly priced "
                f"{anchors.rdma_nic_bandwidth_gbps:g} GB/s RDMA NICs to "
                "provide an effective link of "
                f"{hbf_hardware_variant.rdma_bandwidth_gbps:g} GB/s at "
                f"{hbf_hardware_variant.rdma_one_way_latency_us:g} us "
                "one-way latency, with "
                f"{hbf_hardware_variant.rdma_capex_multiplier:g}x CAPEX and "
                f"{hbf_hardware_variant.rdma_power_multiplier:g}x power."
            ),
        ),
        _bom_line(
            "rdma_network_fabric",
            "GPU-HBF RDMA fabric allocation",
            "fabric unit",
            topology.proposed_rdma_fabric_units,
            rdma_fabric_capex,
            rdma_fabric_power,
            (
                "RDMA fabric cost and power are not hidden in host cost; "
                "the same explicit variant multipliers as the NIC apply."
            ),
        ),
    )
    return _finalize_cost(
        PROPOSED_SYSTEM_KEY,
        "One 4P4D GPU server plus one eight-card HBF-GPU server",
        (
            f"{topology.proposed_cpu_hosts} CPU hosts "
            "(one GPU and one HBF-GPU), "
            f"{topology.proposed_h100_cards} H100 cards, "
            f"{topology.proposed_hbf_npu_cards} HBF-GPU cards, "
            f"{topology.proposed_lpddr_gib:g} GiB LPDDR, one GPU fabric, "
            f"one {hbf_hardware_variant.intra_fabric_kind.upper()} HBF-GPU "
            "fabric, RDMA NIC pair and fabric, no SSD"
        ),
        bom,
        evaluation,
    )


@dataclass(frozen=True)
class TokenEconomics(JSONSafeDataclass):
    system_key: str
    goodput_semantics: str
    slo_good_output_tokens_per_second: float
    lifetime_loaded_seconds: float
    lifetime_slo_good_output_tokens: float
    lifetime_tco_usd: float
    lifetime_slo_good_output_tokens_per_tco_usd: float
    tco_usd_per_million_slo_good_output_tokens: Optional[float]

    def __post_init__(self) -> None:
        if self.system_key not in ECONOMIC_SYSTEM_KEYS:
            raise HBFComparisonTCOError(
                "token economics are only defined for physical systems")
        for name in (
            "slo_good_output_tokens_per_second",
            "lifetime_loaded_seconds",
            "lifetime_slo_good_output_tokens",
            "lifetime_tco_usd",
            "lifetime_slo_good_output_tokens_per_tco_usd",
        ):
            _require_finite(name, getattr(self, name), minimum=0.0)
        if self.lifetime_tco_usd <= 0.0:
            raise HBFComparisonTCOError(
                "lifetime_tco_usd must be positive")
        if self.tco_usd_per_million_slo_good_output_tokens is not None:
            _require_finite(
                "tco_usd_per_million_slo_good_output_tokens",
                self.tco_usd_per_million_slo_good_output_tokens,
                minimum=0.0,
            )


def token_economics(
        cost: SystemCost,
        slo_good_output_tokens_per_second: float,
) -> TokenEconomics:
    """Convert SLO output-token goodput into lifetime tokens per dollar."""

    if not isinstance(cost, SystemCost):
        raise HBFComparisonTCOError("cost must be a SystemCost")
    goodput = _require_finite(
        "slo_good_output_tokens_per_second",
        slo_good_output_tokens_per_second,
        minimum=0.0,
    )
    lifetime_loaded_seconds = (
        cost.lifetime_years
        * HOURS_PER_YEAR
        * SECONDS_PER_HOUR
        * cost.average_utilization
    )
    lifetime_tokens = goodput * lifetime_loaded_seconds
    tokens_per_usd = lifetime_tokens / cost.lifetime_tco_usd
    cost_per_million = (
        None
        if lifetime_tokens == 0.0
        else cost.lifetime_tco_usd / lifetime_tokens * MILLION_TOKENS
    )
    return TokenEconomics(
        system_key=cost.system_key,
        goodput_semantics=GOODPUT_SEMANTICS,
        slo_good_output_tokens_per_second=goodput,
        lifetime_loaded_seconds=lifetime_loaded_seconds,
        lifetime_slo_good_output_tokens=lifetime_tokens,
        lifetime_tco_usd=cost.lifetime_tco_usd,
        lifetime_slo_good_output_tokens_per_tco_usd=tokens_per_usd,
        tco_usd_per_million_slo_good_output_tokens=cost_per_million,
    )


@dataclass(frozen=True)
class OraclePerformanceReference(JSONSafeDataclass):
    system_key: str
    label: str
    slo_good_output_tokens_per_second: Optional[float]
    goodput_semantics: str
    infinite_hbm_capacity: bool
    physical_bom_available: bool
    included_in_main_tco_comparison: bool
    tco_usd: None
    tokens_per_usd: None
    exclusion_reason: str

    def __post_init__(self) -> None:
        if self.system_key != ORACLE_SYSTEM_KEY:
            raise HBFComparisonTCOError(
                "Oracle reference has the wrong system key")
        if self.slo_good_output_tokens_per_second is not None:
            _require_finite(
                "oracle slo_good_output_tokens_per_second",
                self.slo_good_output_tokens_per_second,
                minimum=0.0,
            )
        if (
            not self.infinite_hbm_capacity
            or self.physical_bom_available
            or self.included_in_main_tco_comparison
            or self.tco_usd is not None
            or self.tokens_per_usd is not None
        ):
            raise HBFComparisonTCOError(
                "the infinite-HBM Oracle must remain outside TCO")


@dataclass(frozen=True)
class GoodputResultProvenance(JSONSafeDataclass):
    """Auditable identity for one aggregated goodput result.

    ``measurement_cohort_sha256`` hashes the ordered or canonicalized
    measurement roster, not the system-specific completion order.
    ``schedule_sha256`` hashes either one frozen schedule or the ordered
    manifest of every paired-seed schedule, as declared by
    ``schedule_hash_semantics``.  Both hashes must match across systems.
    """

    system_key: str
    slo_good_output_tokens_per_second: float
    offered_session_rate_per_second: float
    scenario_id: str
    cohort_id: str
    schedule_sha256: str
    schedule_hash_semantics: str
    measurement_cohort_sha256: str
    result_goodput_origin: str
    result_manifest_sha256: str
    result_schema_revision: str
    simulator_code_revision: str
    metric_scope: str
    metric_json_path: str
    metric_definition: str
    aggregation_method: str
    seed_count: int
    confidence_interval_method: str
    confidence_interval_lower_tokens_per_second: Optional[float] = None
    confidence_interval_upper_tokens_per_second: Optional[float] = None

    def __post_init__(self) -> None:
        if self.system_key not in (
                TIERING_SYSTEM_KEY, PROPOSED_SYSTEM_KEY, ORACLE_SYSTEM_KEY):
            raise HBFComparisonTCOError(
                f"unsupported provenance system_key={self.system_key!r}")
        goodput = _require_finite(
            "slo_good_output_tokens_per_second",
            self.slo_good_output_tokens_per_second,
            minimum=0.0,
        )
        _require_finite(
            "offered_session_rate_per_second",
            self.offered_session_rate_per_second,
            strictly_positive=True,
        )
        _require_nonempty_string("scenario_id", self.scenario_id)
        _require_nonempty_string("cohort_id", self.cohort_id)
        _require_sha256("schedule_sha256", self.schedule_sha256)
        if (
            not isinstance(self.schedule_hash_semantics, str)
            or self.schedule_hash_semantics not in SCHEDULE_HASH_SEMANTICS
        ):
            raise HBFComparisonTCOError(
                "schedule_hash_semantics must be one of "
                f"{SCHEDULE_HASH_SEMANTICS!r}")
        _require_sha256(
            "measurement_cohort_sha256",
            self.measurement_cohort_sha256,
        )
        _require_nonempty_string(
            "result_goodput_origin", self.result_goodput_origin)
        _require_sha256(
            "result_manifest_sha256", self.result_manifest_sha256)
        _require_nonempty_string(
            "result_schema_revision", self.result_schema_revision)
        _require_nonempty_string(
            "simulator_code_revision", self.simulator_code_revision)
        if (
            not isinstance(self.metric_scope, str)
            or self.metric_scope not in GOODPUT_METRIC_SCOPES
        ):
            raise HBFComparisonTCOError(
                f"metric_scope must be one of {GOODPUT_METRIC_SCOPES!r}")
        expected_metric_paths = {
            LEGACY_OUTPUT_TOKEN_GOODPUT_JSON_PATH_TEMPLATE.format(
                scope=self.metric_scope),
        }
        if self.metric_scope == "all":
            expected_metric_paths.add(
                LIVE_COMPACT_OUTPUT_TOKEN_GOODPUT_JSON_PATH)
        if self.metric_json_path not in expected_metric_paths:
            raise HBFComparisonTCOError(
                "metric_json_path does not match metric_scope")
        if self.metric_definition != OUTPUT_TOKEN_GOODPUT_DEFINITION:
            raise HBFComparisonTCOError(
                "metric_definition is not the frozen SLO output-token "
                "goodput definition")
        _require_nonempty_string(
            "aggregation_method", self.aggregation_method)
        _require_positive_integer("seed_count", self.seed_count)
        expected_schedule_semantics = (
            "single_frozen_schedule"
            if self.seed_count == 1
            else "ordered_paired_seed_schedule_set_manifest"
        )
        if self.schedule_hash_semantics != expected_schedule_semantics:
            raise HBFComparisonTCOError(
                "schedule_hash_semantics disagrees with seed_count")
        _require_nonempty_string(
            "confidence_interval_method",
            self.confidence_interval_method,
        )

        lower = self.confidence_interval_lower_tokens_per_second
        upper = self.confidence_interval_upper_tokens_per_second
        if (lower is None) != (upper is None):
            raise HBFComparisonTCOError(
                "confidence interval lower and upper bounds must both be "
                "present or both be absent")
        if lower is not None and upper is not None:
            converted_lower = _require_finite(
                "confidence_interval_lower_tokens_per_second",
                lower,
            )
            converted_upper = _require_finite(
                "confidence_interval_upper_tokens_per_second",
                upper,
                minimum=0.0,
            )
            if converted_lower > converted_upper:
                raise HBFComparisonTCOError(
                    "confidence interval lower bound exceeds upper bound")
            if goodput < converted_lower or goodput > converted_upper:
                raise HBFComparisonTCOError(
                    "goodput must lie inside its confidence interval")


@dataclass(frozen=True)
class LiveComparisonArtifactProvenance(JSONSafeDataclass):
    """Content identity for one compact-v2 live-ASTRA TCO extraction."""

    campaign_sha256: str
    manifest_sha256: str
    compact_results_sha256: str
    manifest_schema_version: int
    compact_schema_version: int
    scenario_source_sha256: str
    scenario_manifest_sha256: str
    measurement_cohort_sha256: str
    simulator_implementation_sha256: str
    astra_binary_sha256: str
    canonical_collector_source_sha256: str
    tco_adapter_implementation_sha256: str
    deployment_semantic_snapshot: Mapping[str, Any]
    deployment_semantic_snapshot_sha256: str
    capacity_semantic_snapshot: Mapping[str, Any]
    capacity_semantic_snapshot_sha256: str
    confidence_interval_semantics: str
    tiering_cluster_config_path: str
    tiering_cluster_config_sha256: str
    tiering_policy_config_path: str
    tiering_policy_config_sha256: str
    proposed_gpu_cluster_config_path: str
    proposed_gpu_cluster_config_sha256: str
    proposed_hbf_config_path: str
    proposed_hbf_config_sha256: str
    oracle_cluster_config_path: str
    oracle_cluster_config_sha256: str
    oracle_policy_config_path: str
    oracle_policy_config_sha256: str
    selected_rate_per_second: float
    selected_hbf_system_key: str
    selected_hbf_layout_key: str
    paired_workload_schedule_sha256: str
    active_prefill_drain_policy_version: int
    active_prefill_drain_tail_tokens: int
    active_prefill_drain_min_tokens: int
    active_prefill_drain_policy_contract_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "campaign_sha256",
            "manifest_sha256",
            "compact_results_sha256",
            "scenario_source_sha256",
            "scenario_manifest_sha256",
            "measurement_cohort_sha256",
            "simulator_implementation_sha256",
            "astra_binary_sha256",
            "canonical_collector_source_sha256",
            "tco_adapter_implementation_sha256",
            "deployment_semantic_snapshot_sha256",
            "capacity_semantic_snapshot_sha256",
            "tiering_cluster_config_sha256",
            "tiering_policy_config_sha256",
            "proposed_gpu_cluster_config_sha256",
            "proposed_hbf_config_sha256",
            "oracle_cluster_config_sha256",
            "oracle_policy_config_sha256",
            "paired_workload_schedule_sha256",
            "active_prefill_drain_policy_contract_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in (
            "tiering_cluster_config_path",
            "tiering_policy_config_path",
            "proposed_gpu_cluster_config_path",
            "proposed_hbf_config_path",
            "oracle_cluster_config_path",
            "oracle_policy_config_path",
        ):
            _require_nonempty_string(name, getattr(self, name))
        expected_deployment = {
            "schema_version": 2,
            "gpu_server_model_name": (
                "Qwen/Qwen3-30B-A3B-Instruct-2507"),
            "gpu_server_dtype": "bfloat16",
            "gpu_server_kv_cache_dtype": "auto",
            "gpu_server_prefill_h100_cards": 4,
            "gpu_server_decode_h100_cards": 4,
            "gpu_instance_tp_size": 4,
            "gpu_instance_pp_size": 1,
            "cpu_dram_bytes_per_gpu_host": 512_000_000_000,
            "h100_hbm_bytes_per_card": 80_000_000_000,
            "tiering_cpu_hosts": 2,
            "tiering_h100_cards": 16,
            "tiering_cpu_dram_bytes_per_host": 512_000_000_000,
            "tiering_ssd_devices_per_host": 8,
            "tiering_ssd_devices": 16,
            "tiering_ssd_capacity_gb_per_device": 3_840,
            "proposed_gpu_cpu_hosts": 1,
            "proposed_hbf_cpu_hosts": 1,
            "proposed_cpu_hosts": 2,
            "proposed_gpu_host_cpu_dram_bytes": 512_000_000_000,
            "proposed_hbf_host_cpu_dram_bytes": 512_000_000_000,
            "proposed_hbf_host_cpu_dram_semantics": (
                "explicit_bom_assumption_same_as_gpu_host"),
            "proposed_h100_cards": 8,
            "proposed_hbf_npu_cards": 8,
            "proposed_lpddr_gib": 512,
            "proposed_ssd_devices": 0,
        }
        if not isinstance(self.deployment_semantic_snapshot, Mapping):
            raise HBFComparisonTCOError(
                "deployment_semantic_snapshot must be a mapping")
        deployment_hash = _stable_json_sha256(
            self.deployment_semantic_snapshot)
        if deployment_hash != _stable_json_sha256(expected_deployment):
            raise HBFComparisonTCOError(
                "live deployment semantic snapshot does not match the "
                "physical TCO comparison")
        if deployment_hash != self.deployment_semantic_snapshot_sha256:
            raise HBFComparisonTCOError(
                "deployment semantic snapshot hash is inconsistent")
        if not isinstance(self.capacity_semantic_snapshot, Mapping):
            raise HBFComparisonTCOError(
                "capacity_semantic_snapshot must be a mapping")
        if _stable_json_sha256(self.capacity_semantic_snapshot) != (
                self.capacity_semantic_snapshot_sha256):
            raise HBFComparisonTCOError(
                "capacity semantic snapshot hash is inconsistent")
        _require_nonempty_string(
            "confidence_interval_semantics",
            self.confidence_interval_semantics,
        )
        if self.manifest_schema_version != 2:
            raise HBFComparisonTCOError(
                "live TCO requires manifest schema version 2")
        if self.compact_schema_version != 2:
            raise HBFComparisonTCOError(
                "live TCO requires compact schema version 2")
        _require_finite(
            "selected_rate_per_second",
            self.selected_rate_per_second,
            strictly_positive=True,
        )
        expected_layouts = {
            "hbf_tp4": "tp4",
            "hbf_tp8": "tp8",
            "hbf_tp8_context": "tp8_context",
        }
        if self.selected_hbf_system_key not in expected_layouts:
            raise HBFComparisonTCOError(
                "selected_hbf_system_key must name a live HBF system")
        if self.selected_hbf_layout_key != expected_layouts[
                self.selected_hbf_system_key]:
            raise HBFComparisonTCOError(
                "selected HBF system and layout disagree")
        if self.active_prefill_drain_policy_version != 2:
            raise HBFComparisonTCOError(
                "live TCO requires active-prefill-drain policy v2")
        _require_positive_integer(
            "active_prefill_drain_tail_tokens",
            self.active_prefill_drain_tail_tokens,
        )
        _require_positive_integer(
            "active_prefill_drain_min_tokens",
            self.active_prefill_drain_min_tokens,
        )


@dataclass(frozen=True)
class ComparisonPerformanceProvenance(JSONSafeDataclass):
    """Typed, cross-checked provenance for the economic input rates."""

    selected_tiering_policy_key: str
    hbf_layout_key: str
    hbf_policy_key: str
    hbf_policy_contract_sha256: str
    gpu_config_sha256: str
    hbf_hardware_variant: HBFHardwareVariant
    first_ttft_slo_ns: int
    resume_ttft_slo_ns: int
    tpot_slo_ns: int
    operating_point_mode: str
    rate_selection_semantics: str
    maximum_slo_sustainable_claim: bool
    tiering_result: GoodputResultProvenance
    proposed_result: GoodputResultProvenance
    oracle_result: Optional[GoodputResultProvenance] = None
    live_artifact_provenance: Optional[
        LiveComparisonArtifactProvenance] = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.selected_tiering_policy_key, str)
            or self.selected_tiering_policy_key not in TIERING_POLICY_KEYS
        ):
            raise HBFComparisonTCOError(
                "selected_tiering_policy_key must be one of "
                f"{TIERING_POLICY_KEYS!r}")
        if (
            not isinstance(self.hbf_layout_key, str)
            or self.hbf_layout_key not in HBF_LAYOUT_KEYS
        ):
            raise HBFComparisonTCOError(
                f"hbf_layout_key must be one of {HBF_LAYOUT_KEYS!r}")
        if (
            not isinstance(self.hbf_policy_key, str)
            or self.hbf_policy_key not in HBF_POLICY_KEYS
        ):
            raise HBFComparisonTCOError(
                f"hbf_policy_key must be one of {HBF_POLICY_KEYS!r}")
        _require_sha256(
            "hbf_policy_contract_sha256",
            self.hbf_policy_contract_sha256,
        )
        _require_sha256(
            "gpu_config_sha256", self.gpu_config_sha256)
        if not isinstance(
                self.hbf_hardware_variant, HBFHardwareVariant):
            raise HBFComparisonTCOError(
                "hbf_hardware_variant must be HBFHardwareVariant")
        wide_layout = self.hbf_layout_key == "hbf_tp4_wide"
        wide_config = (
            self.hbf_hardware_variant.hbf_config_sha256
            == PINNED_HBF_WIDE_LPDDR_CONFIG_SHA256
        )
        if wide_layout != wide_config:
            raise HBFComparisonTCOError(
                "hbf_tp4_wide layout and pinned wide-LPDDR hardware "
                "variant must be selected together")
        _require_positive_integer(
            "first_ttft_slo_ns", self.first_ttft_slo_ns)
        _require_positive_integer(
            "resume_ttft_slo_ns", self.resume_ttft_slo_ns)
        _require_positive_integer("tpot_slo_ns", self.tpot_slo_ns)
        if self.operating_point_mode != MATCHED_OPERATING_POINT_MODE:
            raise HBFComparisonTCOError(
                "this API only supports matched_single_operating_point; "
                "it does not validate a maximum-SLO-sustainable rate sweep")
        _require_nonempty_string(
            "rate_selection_semantics",
            self.rate_selection_semantics,
        )
        if not isinstance(self.maximum_slo_sustainable_claim, bool):
            raise HBFComparisonTCOError(
                "maximum_slo_sustainable_claim must be a boolean")
        if self.maximum_slo_sustainable_claim:
            raise HBFComparisonTCOError(
                "matched_single_operating_point cannot claim maximum "
                "SLO-sustainable throughput")
        if not isinstance(self.tiering_result, GoodputResultProvenance):
            raise HBFComparisonTCOError(
                "tiering_result must be GoodputResultProvenance")
        if not isinstance(self.proposed_result, GoodputResultProvenance):
            raise HBFComparisonTCOError(
                "proposed_result must be GoodputResultProvenance")
        if self.tiering_result.system_key != TIERING_SYSTEM_KEY:
            raise HBFComparisonTCOError(
                "tiering_result has a mismatched system key")
        if self.proposed_result.system_key != PROPOSED_SYSTEM_KEY:
            raise HBFComparisonTCOError(
                "proposed_result has a mismatched system key")
        if self.oracle_result is not None:
            if not isinstance(
                    self.oracle_result, GoodputResultProvenance):
                raise HBFComparisonTCOError(
                    "oracle_result must be GoodputResultProvenance")
            if self.oracle_result.system_key != ORACLE_SYSTEM_KEY:
                raise HBFComparisonTCOError(
                    "oracle_result has a mismatched system key")
        live = self.live_artifact_provenance
        if live is not None:
            if not isinstance(live, LiveComparisonArtifactProvenance):
                raise HBFComparisonTCOError(
                    "live_artifact_provenance must be "
                    "LiveComparisonArtifactProvenance")
            if self.hbf_layout_key != live.selected_hbf_system_key:
                raise HBFComparisonTCOError(
                    "HBF layout key disagrees with live selected system")
            if self.gpu_config_sha256 != (
                    live.proposed_gpu_cluster_config_sha256):
                raise HBFComparisonTCOError(
                    "GPU config hash disagrees with live proposed cluster")
            if self.hbf_hardware_variant.hbf_config_sha256 != (
                    live.proposed_hbf_config_sha256):
                raise HBFComparisonTCOError(
                    "HBF config hash disagrees with live proposed config")
            if self.hbf_policy_contract_sha256 != (
                    live.active_prefill_drain_policy_contract_sha256):
                raise HBFComparisonTCOError(
                    "HBF policy hash disagrees with live drain policy")

        compared = [self.tiering_result, self.proposed_result]
        if self.oracle_result is not None:
            compared.append(self.oracle_result)
        identity_fields = (
            "offered_session_rate_per_second",
            "scenario_id",
            "cohort_id",
            "schedule_sha256",
            "schedule_hash_semantics",
            "measurement_cohort_sha256",
            "result_schema_revision",
            "simulator_code_revision",
            "metric_scope",
            "metric_json_path",
            "metric_definition",
            "aggregation_method",
            "seed_count",
            "confidence_interval_method",
        )
        reference = self.tiering_result
        if live is not None:
            if reference.offered_session_rate_per_second != (
                    live.selected_rate_per_second):
                raise HBFComparisonTCOError(
                    "offered rate disagrees with live selection")
            if reference.schedule_sha256 != (
                    live.paired_workload_schedule_sha256):
                raise HBFComparisonTCOError(
                    "schedule hash disagrees with live workload schedule")
            if reference.measurement_cohort_sha256 != (
                    live.measurement_cohort_sha256):
                raise HBFComparisonTCOError(
                    "measurement cohort hash disagrees with live campaign")
            if reference.result_manifest_sha256 != live.manifest_sha256:
                raise HBFComparisonTCOError(
                    "result manifest hash disagrees with live manifest")
            if reference.simulator_code_revision != (
                    live.simulator_implementation_sha256):
                raise HBFComparisonTCOError(
                    "simulator revision disagrees with live campaign")
        for result in compared[1:]:
            for field in identity_fields:
                if getattr(result, field) != getattr(reference, field):
                    raise HBFComparisonTCOError(
                        f"mismatched performance provenance field {field}")
            if (
                result.confidence_interval_lower_tokens_per_second is None
            ) != (
                reference.confidence_interval_lower_tokens_per_second is None
            ):
                raise HBFComparisonTCOError(
                    "mismatched confidence interval availability")


@dataclass(frozen=True)
class SensitivityRow(JSONSafeDataclass):
    scenario_key: str
    sensitivity: SensitivityPoint
    tiering_cost: SystemCost
    proposed_cost: SystemCost
    tiering_token_economics: TokenEconomics
    proposed_token_economics: TokenEconomics
    proposed_capex_ratio_to_tiering: float
    proposed_it_power_ratio_to_tiering: float
    proposed_tco_ratio_to_tiering: float
    proposed_goodput_ratio_to_tiering: float
    break_even_proposed_goodput_ratio_to_tiering: float
    break_even_proposed_goodput_tokens_per_second: float
    proposed_tokens_per_usd_ratio_to_tiering: float
    proposed_meets_or_exceeds_token_value_break_even: bool

    def __post_init__(self) -> None:
        if self.scenario_key != self.sensitivity.key:
            raise HBFComparisonTCOError(
                "scenario_key must match the sensitivity point")
        for name in (
            "proposed_capex_ratio_to_tiering",
            "proposed_it_power_ratio_to_tiering",
            "proposed_tco_ratio_to_tiering",
            "proposed_goodput_ratio_to_tiering",
            "break_even_proposed_goodput_ratio_to_tiering",
            "break_even_proposed_goodput_tokens_per_second",
            "proposed_tokens_per_usd_ratio_to_tiering",
        ):
            _require_finite(name, getattr(self, name), minimum=0.0)


@dataclass(frozen=True)
class MemoryCapacityDisclosure(JSONSafeDataclass):
    h100_hbm_capacity_bytes_per_card: int
    hbf_capacity_bytes_per_card: int
    hbf_capacity_ratio_to_hbm_per_card: float
    tiering_raw_hbm_capacity_bytes: int
    proposed_raw_hbm_capacity_bytes: int
    proposed_raw_hbf_capacity_bytes: int
    selected_hbf_layout_key: str
    selected_hbf_tp_size: int
    selected_hbf_replica_count: int
    selected_hbf_physical_kv_replication_factor: int
    hbf_model_weight_bytes_per_card: int
    proposed_usable_logical_hbf_kv_capacity_bytes: int
    capacity_semantics: str

    def __post_init__(self) -> None:
        for name in (
            "h100_hbm_capacity_bytes_per_card",
            "hbf_capacity_bytes_per_card",
            "tiering_raw_hbm_capacity_bytes",
            "proposed_raw_hbm_capacity_bytes",
            "proposed_raw_hbf_capacity_bytes",
            "selected_hbf_tp_size",
            "selected_hbf_replica_count",
            "selected_hbf_physical_kv_replication_factor",
            "hbf_model_weight_bytes_per_card",
            "proposed_usable_logical_hbf_kv_capacity_bytes",
        ):
            _require_positive_integer(name, getattr(self, name))
        ratio = _require_finite(
            "hbf_capacity_ratio_to_hbm_per_card",
            self.hbf_capacity_ratio_to_hbm_per_card,
            strictly_positive=True,
        )
        expected = (
            self.hbf_capacity_bytes_per_card
            / self.h100_hbm_capacity_bytes_per_card
        )
        if not math.isclose(
                ratio, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise HBFComparisonTCOError(
                "capacity disclosure ratio is inconsistent")
        _require_nonempty_string(
            "selected_hbf_layout_key", self.selected_hbf_layout_key)
        if self.hbf_model_weight_bytes_per_card >= (
                self.hbf_capacity_bytes_per_card):
            raise HBFComparisonTCOError(
                "HBF model weights leave no usable KV capacity")
        expected_usable = (
            (
                self.hbf_capacity_bytes_per_card
                - self.hbf_model_weight_bytes_per_card
            )
            * self.selected_hbf_tp_size
            * self.selected_hbf_replica_count
            // self.selected_hbf_physical_kv_replication_factor
        )
        if self.proposed_usable_logical_hbf_kv_capacity_bytes != (
                expected_usable):
            raise HBFComparisonTCOError(
                "usable logical HBF KV capacity is inconsistent")
        _require_nonempty_string(
            "capacity_semantics", self.capacity_semantics)


@dataclass(frozen=True)
class TCOSensitivityReport(JSONSafeDataclass):
    report_schema: str
    economic_system_keys: tuple[str, str]
    price_source_semantics: str
    goodput_semantics: str
    anchors: HardwareAnchors
    topology: DeploymentTopology
    evaluation: EvaluationAssumptions
    axes: SensitivityAxes
    performance_provenance: ComparisonPerformanceProvenance
    memory_capacity: MemoryCapacityDisclosure
    tiering_cost: SystemCost
    sensitivity_rows: tuple[SensitivityRow, ...]
    oracle_performance_reference: OraclePerformanceReference
    economic_claim_scope: str
    token_economics_scope: str

    def __post_init__(self) -> None:
        if self.economic_system_keys != ECONOMIC_SYSTEM_KEYS:
            raise HBFComparisonTCOError(
                "main economic comparison must contain only tiering and HBF")
        if len(self.sensitivity_rows) != self.axes.cartesian_size:
            raise HBFComparisonTCOError(
                "sensitivity row count does not match Cartesian axes")
        if self.oracle_performance_reference.included_in_main_tco_comparison:
            raise HBFComparisonTCOError(
                "Oracle cannot enter the main economic comparison")
        _require_nonempty_string(
            "token_economics_scope", self.token_economics_scope)


def _validate_goodput_map(
        values: Mapping[str, float],
) -> tuple[float, float, Optional[float]]:
    if not isinstance(values, Mapping):
        raise HBFComparisonTCOError(
            "goodput input must be a mapping")
    expected = set(ECONOMIC_SYSTEM_KEYS)
    allowed = expected | {ORACLE_SYSTEM_KEY}
    keys = set(values)
    missing = expected - keys
    unknown = keys - allowed
    if missing:
        raise HBFComparisonTCOError(
            f"goodput input is missing systems: {sorted(missing)!r}")
    if unknown:
        raise HBFComparisonTCOError(
            f"goodput input has unknown systems: {sorted(unknown)!r}")
    tiering = _require_finite(
        f"{TIERING_SYSTEM_KEY} goodput",
        values[TIERING_SYSTEM_KEY],
        strictly_positive=True,
    )
    proposed = _require_finite(
        f"{PROPOSED_SYSTEM_KEY} goodput",
        values[PROPOSED_SYSTEM_KEY],
        minimum=0.0,
    )
    oracle = None
    if ORACLE_SYSTEM_KEY in values:
        oracle = _require_finite(
            f"{ORACLE_SYSTEM_KEY} goodput",
            values[ORACLE_SYSTEM_KEY],
            minimum=0.0,
        )
    return tiering, proposed, oracle


def _capacity_layout(hbf_layout_key: str) -> HBFParallelLayout:
    aliases = {
        "dp8": "dp8",
        "tp4": "tp4",
        "tp8": "tp8",
        "hbf_dp8": "dp8",
        "hbf_tp4": "tp4",
        "hbf_tp8": "tp8",
        "hbf_tp8_context": "tp8_context",
        "hbf_tp4_wide": "tp4",
    }
    try:
        key = aliases[hbf_layout_key]
    except KeyError as exc:
        raise HBFComparisonTCOError(
            f"cannot disclose capacity for layout {hbf_layout_key!r}"
        ) from exc
    return HBFParallelLayout.for_key(key)


def evaluate_tco_sensitivity(
        slo_good_output_tokens_per_second: Mapping[str, float],
        *,
        performance_provenance: Optional[
            ComparisonPerformanceProvenance] = None,
        anchors: HardwareAnchors = HardwareAnchors(),
        topology: DeploymentTopology = DeploymentTopology(),
        evaluation: EvaluationAssumptions = EvaluationAssumptions(),
        axes: SensitivityAxes = SensitivityAxes(),
) -> TCOSensitivityReport:
    """Evaluate all component-ratio combinations and token economics.

    The input goodput values must come from the same offered-load point and
    matched request cohort.  The function validates units by accepting only
    explicitly named SLO-good output-token rates.  An optional Oracle rate is
    preserved as a performance-only disclosure and never enters a cost row.

    This API intentionally evaluates one matched operating point.  It rejects
    a maximum-SLO-sustainable claim because proving that claim requires a
    complete rate-grid selection artifact, not one selected cell.
    """

    if not isinstance(anchors, HardwareAnchors):
        raise HBFComparisonTCOError("anchors must be HardwareAnchors")
    if not isinstance(topology, DeploymentTopology):
        raise HBFComparisonTCOError("topology must be DeploymentTopology")
    if not isinstance(evaluation, EvaluationAssumptions):
        raise HBFComparisonTCOError(
            "evaluation must be EvaluationAssumptions")
    if not isinstance(axes, SensitivityAxes):
        raise HBFComparisonTCOError("axes must be SensitivityAxes")
    if not isinstance(
            performance_provenance, ComparisonPerformanceProvenance):
        raise HBFComparisonTCOError(
            "performance_provenance is required and must be "
            "ComparisonPerformanceProvenance")

    tiering_goodput, proposed_goodput, oracle_goodput = (
        _validate_goodput_map(slo_good_output_tokens_per_second)
    )
    provenance_values = {
        TIERING_SYSTEM_KEY: (
            performance_provenance.tiering_result
            .slo_good_output_tokens_per_second
        ),
        PROPOSED_SYSTEM_KEY: (
            performance_provenance.proposed_result
            .slo_good_output_tokens_per_second
        ),
    }
    if performance_provenance.oracle_result is not None:
        provenance_values[ORACLE_SYSTEM_KEY] = (
            performance_provenance.oracle_result
            .slo_good_output_tokens_per_second
        )
    normalized_values = {
        TIERING_SYSTEM_KEY: tiering_goodput,
        PROPOSED_SYSTEM_KEY: proposed_goodput,
    }
    if oracle_goodput is not None:
        normalized_values[ORACLE_SYSTEM_KEY] = oracle_goodput
    if set(provenance_values) != set(normalized_values):
        raise HBFComparisonTCOError(
            "goodput systems do not match performance provenance")
    for system_key, value in normalized_values.items():
        if value != provenance_values[system_key]:
            raise HBFComparisonTCOError(
                f"{system_key} goodput mismatches performance provenance")

    tiering_cost = tiering_baseline_cost(
        anchors, topology, evaluation)
    tiering_value = token_economics(
        tiering_cost, tiering_goodput)

    rows = []
    for point in sensitivity_points(axes):
        proposed_cost = proposed_hbf_cost(
            point,
            anchors,
            topology,
            evaluation,
            performance_provenance.hbf_hardware_variant,
        )
        proposed_value = token_economics(
            proposed_cost, proposed_goodput)

        tco_ratio = (
            proposed_cost.lifetime_tco_usd
            / tiering_cost.lifetime_tco_usd
        )
        goodput_ratio = proposed_goodput / tiering_goodput
        break_even_goodput = tiering_goodput * tco_ratio
        value_ratio = (
            proposed_value.lifetime_slo_good_output_tokens_per_tco_usd
            / tiering_value.lifetime_slo_good_output_tokens_per_tco_usd
        )
        rows.append(SensitivityRow(
            scenario_key=point.key,
            sensitivity=point,
            tiering_cost=tiering_cost,
            proposed_cost=proposed_cost,
            tiering_token_economics=tiering_value,
            proposed_token_economics=proposed_value,
            proposed_capex_ratio_to_tiering=(
                proposed_cost.capex_usd / tiering_cost.capex_usd),
            proposed_it_power_ratio_to_tiering=(
                proposed_cost.it_power_w / tiering_cost.it_power_w),
            proposed_tco_ratio_to_tiering=tco_ratio,
            proposed_goodput_ratio_to_tiering=goodput_ratio,
            break_even_proposed_goodput_ratio_to_tiering=tco_ratio,
            break_even_proposed_goodput_tokens_per_second=(
                break_even_goodput),
            proposed_tokens_per_usd_ratio_to_tiering=value_ratio,
            proposed_meets_or_exceeds_token_value_break_even=(
                proposed_goodput >= break_even_goodput),
        ))

    oracle = OraclePerformanceReference(
        system_key=ORACLE_SYSTEM_KEY,
        label="Infinite-HBM Oracle performance reference",
        slo_good_output_tokens_per_second=oracle_goodput,
        goodput_semantics=GOODPUT_SEMANTICS,
        infinite_hbm_capacity=True,
        physical_bom_available=False,
        included_in_main_tco_comparison=False,
        tco_usd=None,
        tokens_per_usd=None,
        exclusion_reason=ORACLE_EXCLUSION_REASON,
    )
    capacity_layout = _capacity_layout(
        performance_provenance.hbf_layout_key)
    weight_bytes_per_card = qwen_model_weight_bytes_per_rank(
        capacity_layout.tp_size)
    usable_logical_hbf_kv_bytes = (
        (
            performance_provenance.hbf_hardware_variant
            .hbf_capacity_bytes_per_card
            - weight_bytes_per_card
        )
        * capacity_layout.tp_size
        * capacity_layout.replicas
        // capacity_layout.physical_kv_replication_factor
    )
    return TCOSensitivityReport(
        report_schema="hbf-comparison-tco-v1",
        economic_system_keys=ECONOMIC_SYSTEM_KEYS,
        price_source_semantics=PRICE_SOURCE_SEMANTICS,
        goodput_semantics=GOODPUT_SEMANTICS,
        anchors=anchors,
        topology=topology,
        evaluation=evaluation,
        axes=axes,
        performance_provenance=performance_provenance,
        memory_capacity=MemoryCapacityDisclosure(
            h100_hbm_capacity_bytes_per_card=(
                anchors.h100_hbm_capacity_bytes_per_card),
            hbf_capacity_bytes_per_card=(
                performance_provenance.hbf_hardware_variant
                .hbf_capacity_bytes_per_card),
            hbf_capacity_ratio_to_hbm_per_card=(
                performance_provenance.hbf_hardware_variant
                .hbf_capacity_ratio_to_hbm),
            tiering_raw_hbm_capacity_bytes=(
                topology.tiering_h100_cards
                * anchors.h100_hbm_capacity_bytes_per_card),
            proposed_raw_hbm_capacity_bytes=(
                topology.proposed_h100_cards
                * anchors.h100_hbm_capacity_bytes_per_card),
            proposed_raw_hbf_capacity_bytes=(
                topology.proposed_hbf_npu_cards
                * performance_provenance.hbf_hardware_variant
                .hbf_capacity_bytes_per_card),
            selected_hbf_layout_key=capacity_layout.key,
            selected_hbf_tp_size=capacity_layout.tp_size,
            selected_hbf_replica_count=capacity_layout.replicas,
            selected_hbf_physical_kv_replication_factor=(
                capacity_layout.physical_kv_replication_factor),
            hbf_model_weight_bytes_per_card=weight_bytes_per_card,
            proposed_usable_logical_hbf_kv_capacity_bytes=(
                usable_logical_hbf_kv_bytes),
            capacity_semantics=(
                "Raw installed media capacity plus exact usable logical HBF "
                "KV capacity after per-card model weights and the selected "
                "layout's physical KV replication factor."
            ),
        ),
        tiering_cost=tiering_cost,
        sensitivity_rows=tuple(rows),
        oracle_performance_reference=oracle,
        economic_claim_scope=(
            "Economic ratios compare only the physical tiering baseline and "
            "the physical HBF proposal. The infinite-HBM Oracle is "
            "performance-only."
        ),
        token_economics_scope=(
            "Matched single offered-rate operating point extrapolated over "
            "productive lifetime seconds. This is not a maximum "
            "SLO-sustainable throughput or max-throughput token/$ claim."
        ),
    )

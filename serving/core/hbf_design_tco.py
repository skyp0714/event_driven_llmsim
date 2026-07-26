"""Isolated design-space TCO model for multi-HBF deployments.

The strict comparison model in :mod:`hbf_comparison_tco` intentionally prices
one fixed HBF proposal.  This module leaves that contract untouched and adds a
small exploratory model for one eight-H100 GPU host plus a configurable
positive number of eight-card HBF hosts.  Each HBF card uses H100-class GPU
compute; HBF media and active memory remain separately priced subsystems.

Active memory is an explicit design input.  ``lpddr`` and ``sram_like`` are
labels for analytical assumptions, not vendor parts or price quotes.  Capacity,
bandwidth metadata, CAPEX per GiB, and power per GiB must all be present in the
input so a high-bandwidth counterfactual cannot silently inherit LPDDR cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Optional

from .hbf_comparison_tco import (
    BOMLine,
    BYTES_PER_GIB,
    EvaluationAssumptions,
    HardwareAnchors,
    KWH_PER_WH,
    P4D4_CPU_MEMORY_BYTES_PER_HOST,
    PRICE_SOURCE_SEMANTICS,
    SensitivityPoint,
    SystemCost,
    tiering_baseline_cost,
)


ACTIVE_MEMORY_KINDS = ("lpddr", "sram_like")
DESIGN_PROPOSED_SYSTEM_KEY = "hbf_design_proposed"
DESIGN_ORACLE_SYSTEM_KEY = "infinite_hbm_oracle"
H100_CARDS_PER_GPU_HOST = 8
HBF_CARDS_PER_HBF_HOST = 8
DESIGN_REPORT_SCHEMA = "hbf-design-tco-v1"
GOODPUT_SEMANTICS = (
    "matched-rate SLO-good output tokens per second"
)
ORACLE_EXCLUSION_REASON = (
    "The infinite-HBM Oracle is a performance-only reference and has no "
    "finite physical bill of materials."
)

CENTRAL_SENSITIVITY_POINT = SensitivityPoint(
    npu_logic_capex_ratio_to_gpu_logic=1.00,
    hbf_subsystem_capex_ratio_to_hbm_stack=0.50,
    npu_logic_power_ratio_to_gpu_logic=1.00,
    hbf_subsystem_power_ratio_to_hbm_stack=3.50,
)


class HBFDesignTCOError(ValueError):
    """Raised when a design-space TCO input is invalid."""


def _finite(
        name: str, value: object, *, minimum: Optional[float] = None,
        strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HBFDesignTCOError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise HBFDesignTCOError(f"{name} must be a finite number")
    if strictly_positive and converted <= 0.0:
        raise HBFDesignTCOError(f"{name} must be positive")
    if minimum is not None and converted < minimum:
        raise HBFDesignTCOError(
            f"{name} must be at least {minimum}")
    return converted


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HBFDesignTCOError(f"{name} must be a positive integer")
    return value


def _nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HBFDesignTCOError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ActiveMemorySpec:
    """Per-HBF-card active-memory assumptions for one design point."""

    kind: str
    capacity_gib_per_card: float
    bandwidth_gbps_per_card: float
    capex_usd_per_gib: float
    power_w_per_gib: float
    assumption: str = (
        "Analytical design-space assumption; not a measured vendor quote."
    )

    def __post_init__(self) -> None:
        if self.kind not in ACTIVE_MEMORY_KINDS:
            raise HBFDesignTCOError(
                f"kind must be one of {ACTIVE_MEMORY_KINDS!r}")
        _finite(
            "capacity_gib_per_card",
            self.capacity_gib_per_card,
            strictly_positive=True,
        )
        _finite(
            "bandwidth_gbps_per_card",
            self.bandwidth_gbps_per_card,
            strictly_positive=True,
        )
        _finite(
            "capex_usd_per_gib",
            self.capex_usd_per_gib,
            minimum=0.0,
        )
        _finite(
            "power_w_per_gib",
            self.power_w_per_gib,
            minimum=0.0,
        )
        _nonempty("assumption", self.assumption)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def lpddr_active_memory(
        *,
        capacity_gib_per_card: float = 64.0,
        bandwidth_gbps_per_card: float = 204.8,
        anchors: HardwareAnchors = HardwareAnchors(),
) -> ActiveMemorySpec:
    """Build an LPDDR point using the comparison model's unit anchors."""

    if not isinstance(anchors, HardwareAnchors):
        raise HBFDesignTCOError("anchors must be HardwareAnchors")
    return ActiveMemorySpec(
        kind="lpddr",
        capacity_gib_per_card=capacity_gib_per_card,
        bandwidth_gbps_per_card=bandwidth_gbps_per_card,
        capex_usd_per_gib=anchors.lpddr_capex_usd_per_gib,
        power_w_per_gib=anchors.lpddr_power_w_per_gib,
        assumption=(
            "LPDDR unit cost and power reuse HardwareAnchors; capacity and "
            "bandwidth are explicit design-space metadata."
        ),
    )


@dataclass(frozen=True)
class HBFDesignTopology:
    """One fixed GPU host plus N independent eight-card HBF hosts."""

    hbf_host_count: int

    def __post_init__(self) -> None:
        _positive_int("hbf_host_count", self.hbf_host_count)

    @property
    def gpu_host_count(self) -> int:
        return 1

    @property
    def cpu_host_count(self) -> int:
        return self.gpu_host_count + self.hbf_host_count

    @property
    def h100_card_count(self) -> int:
        return H100_CARDS_PER_GPU_HOST

    @property
    def hbf_card_count(self) -> int:
        return self.hbf_host_count * HBF_CARDS_PER_HBF_HOST

    @property
    def gpu_fabric_unit_count(self) -> int:
        return self.gpu_host_count

    @property
    def hbf_fabric_unit_count(self) -> int:
        return self.hbf_host_count

    @property
    def rdma_nic_count(self) -> int:
        return self.cpu_host_count


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
class DesignSystemCost:
    """Auditable cost and energy totals for one exploratory proposal."""

    system_key: str
    physical_description: str
    topology: HBFDesignTopology
    active_memory: ActiveMemorySpec
    sensitivity_point: SensitivityPoint
    bom: tuple[BOMLine, ...]
    capex_usd: float
    it_power_w: float
    facility_power_w: float
    lifetime_it_energy_kwh: float
    lifetime_facility_energy_kwh: float
    lifetime_electricity_opex_usd: float
    lifetime_tco_usd: float
    evaluation: EvaluationAssumptions
    price_source_semantics: str = PRICE_SOURCE_SEMANTICS
    excluded_costs: tuple[str, ...] = (
        "labor", "maintenance", "financing", "taxes")

    def __post_init__(self) -> None:
        if self.system_key != DESIGN_PROPOSED_SYSTEM_KEY:
            raise HBFDesignTCOError("unexpected design system key")
        _nonempty("physical_description", self.physical_description)
        if not isinstance(self.topology, HBFDesignTopology):
            raise HBFDesignTCOError(
                "topology must be HBFDesignTopology")
        if not isinstance(self.active_memory, ActiveMemorySpec):
            raise HBFDesignTCOError(
                "active_memory must be ActiveMemorySpec")
        if not isinstance(self.sensitivity_point, SensitivityPoint):
            raise HBFDesignTCOError(
                "sensitivity_point must be SensitivityPoint")
        if not isinstance(self.evaluation, EvaluationAssumptions):
            raise HBFDesignTCOError(
                "evaluation must be EvaluationAssumptions")
        if not self.bom:
            raise HBFDesignTCOError("bom must not be empty")
        keys = [line.component_key for line in self.bom]
        if len(keys) != len(set(keys)):
            raise HBFDesignTCOError(
                "BOM component keys must be unique")
        expected_capex = math.fsum(line.capex_usd for line in self.bom)
        expected_power = math.fsum(line.it_power_w for line in self.bom)
        if not math.isclose(
                self.capex_usd, expected_capex,
                rel_tol=1e-12, abs_tol=1e-8):
            raise HBFDesignTCOError("capex does not match BOM")
        if not math.isclose(
                self.it_power_w, expected_power,
                rel_tol=1e-12, abs_tol=1e-8):
            raise HBFDesignTCOError("IT power does not match BOM")
        for name in (
            "capex_usd",
            "it_power_w",
            "facility_power_w",
            "lifetime_it_energy_kwh",
            "lifetime_facility_energy_kwh",
            "lifetime_electricity_opex_usd",
            "lifetime_tco_usd",
        ):
            _finite(name, getattr(self, name), minimum=0.0)

    def component(self, component_key: str) -> BOMLine:
        for line in self.bom:
            if line.component_key == component_key:
                return line
        raise KeyError(component_key)

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        json.dumps(value, allow_nan=False)
        return value


def _finalize_design_cost(
        *,
        topology: HBFDesignTopology,
        active_memory: ActiveMemorySpec,
        sensitivity_point: SensitivityPoint,
        bom: tuple[BOMLine, ...],
        evaluation: EvaluationAssumptions,
) -> DesignSystemCost:
    capex = math.fsum(line.capex_usd for line in bom)
    it_power = math.fsum(line.it_power_w for line in bom)
    facility_power = it_power * evaluation.pue
    loaded_equivalent_hours = (
        evaluation.lifetime_powered_equivalent_full_load_hours)
    it_energy = it_power * loaded_equivalent_hours * KWH_PER_WH
    facility_energy = (
        facility_power * loaded_equivalent_hours * KWH_PER_WH)
    electricity = (
        facility_energy * evaluation.electricity_usd_per_kwh)
    return DesignSystemCost(
        system_key=DESIGN_PROPOSED_SYSTEM_KEY,
        physical_description=(
            "One eight-H100 GPU host plus "
            f"{topology.hbf_host_count} independent eight-card HBF hosts; "
            f"{active_memory.kind} active memory; no SSD tier"
        ),
        topology=topology,
        active_memory=active_memory,
        sensitivity_point=sensitivity_point,
        bom=bom,
        capex_usd=capex,
        it_power_w=it_power,
        facility_power_w=facility_power,
        lifetime_it_energy_kwh=it_energy,
        lifetime_facility_energy_kwh=facility_energy,
        lifetime_electricity_opex_usd=electricity,
        lifetime_tco_usd=capex + electricity,
        evaluation=evaluation,
    )


def proposed_hbf_design_cost(
        *,
        hbf_host_count: int,
        active_memory: ActiveMemorySpec,
        sensitivity_point: SensitivityPoint = CENTRAL_SENSITIVITY_POINT,
        anchors: HardwareAnchors = HardwareAnchors(),
        evaluation: EvaluationAssumptions = EvaluationAssumptions(),
) -> DesignSystemCost:
    """Price one GPU host plus N eight-card HBF hosts."""

    topology = HBFDesignTopology(hbf_host_count=hbf_host_count)
    if not isinstance(active_memory, ActiveMemorySpec):
        raise HBFDesignTCOError(
            "active_memory must be ActiveMemorySpec")
    if not isinstance(sensitivity_point, SensitivityPoint):
        raise HBFDesignTCOError(
            "sensitivity_point must be SensitivityPoint")
    if not isinstance(anchors, HardwareAnchors):
        raise HBFDesignTCOError("anchors must be HardwareAnchors")
    if not isinstance(evaluation, EvaluationAssumptions):
        raise HBFDesignTCOError(
            "evaluation must be EvaluationAssumptions")

    hbf_gpu_logic_capex = (
        anchors.gpu_logic_capex_usd_per_card
        * sensitivity_point.npu_logic_capex_ratio_to_gpu_logic
    )
    hbf_gpu_logic_power = (
        anchors.gpu_logic_power_w_per_card
        * sensitivity_point.npu_logic_power_ratio_to_gpu_logic
    )
    hbf_subsystem_capex = (
        anchors.hbf_media_controller_capex_usd_per_card
        * sensitivity_point.hbf_media_controller_capex_multiplier
    )
    hbf_subsystem_power = (
        anchors.hbf_media_controller_power_w_per_card
        * sensitivity_point.hbf_media_controller_power_multiplier
    )
    host_dram_gib = (
        P4D4_CPU_MEMORY_BYTES_PER_HOST / BYTES_PER_GIB)
    active_memory_gib = (
        topology.hbf_card_count
        * active_memory.capacity_gib_per_card)

    bom = (
        _bom_line(
            "cpu_host_base",
            "CPU server base (accelerator, DRAM, NIC excluded)",
            "host",
            topology.cpu_host_count,
            anchors.cpu_host_base_capex_usd,
            anchors.cpu_host_base_power_w,
            "One CPU host is assigned to each GPU or HBF server.",
        ),
        _bom_line(
            "host_dram",
            "Host DRAM",
            "GiB",
            topology.cpu_host_count * host_dram_gib,
            anchors.host_dram_capex_usd_per_gib,
            anchors.host_dram_power_w_per_gib,
            "Every host receives the P4D4 comparison DRAM capacity.",
        ),
        _bom_line(
            "h100_gpu_logic",
            "H100 GPU logic excluding HBM",
            "card",
            topology.h100_card_count,
            anchors.gpu_logic_capex_usd_per_card,
            anchors.gpu_logic_power_w_per_card,
            "The proposal retains one fixed eight-H100 GPU host.",
        ),
        _bom_line(
            "h100_hbm_stack",
            "Complete H100 HBM stack",
            "card",
            topology.h100_card_count,
            anchors.hbm_stack_capex_usd_per_card,
            anchors.hbm_stack_power_w_per_card,
            "The H100 HBM stack is priced separately from GPU logic.",
        ),
        _bom_line(
            "gpu_intraserver_fabric",
            "GPU-host NVSwitch/NVLink fabric",
            "host fabric unit",
            topology.gpu_fabric_unit_count,
            anchors.gpu_intraserver_fabric_capex_usd_per_unit,
            anchors.gpu_intraserver_fabric_power_w_per_unit,
            "The single GPU host has one accelerator fabric allocation.",
        ),
        _bom_line(
            "hbf_gpu_logic",
            "H100-class HBF-card GPU logic excluding HBF and active memory",
            "card",
            topology.hbf_card_count,
            hbf_gpu_logic_capex,
            hbf_gpu_logic_power,
            (
                "The central design uses 1.0x H100 GPU-logic CAPEX and "
                "power. HBM is not included here because HBF media is a "
                "separate subsystem."
            ),
        ),
        _bom_line(
            "hbf_media_controller_subsystem",
            "Complete HBF media/controller subsystem",
            "card",
            topology.hbf_card_count,
            hbf_subsystem_capex,
            hbf_subsystem_power,
            (
                "Normalized sensitivity multipliers apply to independent "
                "HBF media/controller CAPEX and power anchors."
            ),
        ),
        _bom_line(
            "hbf_gpu_intraserver_fabric",
            "HBF-GPU-host PCIe fabric",
            "host fabric unit",
            topology.hbf_fabric_unit_count,
            anchors.hbf_npu_pcie_fabric_capex_usd_per_unit,
            anchors.hbf_npu_pcie_fabric_power_w_per_unit,
            "Each independent HBF host has one PCIe fabric allocation.",
        ),
        _bom_line(
            "hbf_card_active_memory",
            f"HBF-card {active_memory.kind} active memory",
            "GiB",
            active_memory_gib,
            active_memory.capex_usd_per_gib,
            active_memory.power_w_per_gib,
            (
                f"{active_memory.capacity_gib_per_card:g} GiB/card at "
                f"{active_memory.bandwidth_gbps_per_card:g} GB/s/card. "
                f"{active_memory.assumption}"
            ),
        ),
        _bom_line(
            "nvme_ssd_tier",
            "NVMe SSD tier",
            "device",
            0,
            anchors.nvme_ssd_capex_usd_per_device,
            anchors.nvme_ssd_power_w_per_device,
            "The HBF proposal has no SSD tier.",
        ),
        _bom_line(
            "rdma_network_nic",
            "GPU-HBF RDMA NIC",
            "NIC",
            topology.rdma_nic_count,
            anchors.rdma_nic_capex_usd,
            anchors.rdma_nic_power_w,
            "Every GPU or HBF host receives one RDMA NIC.",
        ),
        _bom_line(
            "rdma_network_fabric",
            "Shared GPU-HBF RDMA fabric allocation",
            "fabric unit",
            1,
            anchors.rdma_fabric_capex_usd,
            anchors.rdma_fabric_power_w,
            "All hosts share one analytical fabric allocation.",
        ),
    )
    return _finalize_design_cost(
        topology=topology,
        active_memory=active_memory,
        sensitivity_point=sensitivity_point,
        bom=bom,
        evaluation=evaluation,
    )


@dataclass(frozen=True)
class TokenCost:
    """Lifetime token economics at one matched-rate goodput."""

    slo_good_output_tokens_per_second: float
    lifetime_loaded_seconds: float
    lifetime_slo_good_output_tokens: float
    lifetime_tco_usd: float
    dollars_per_million_slo_good_output_tokens: Optional[float]


def _token_cost(
        lifetime_tco_usd: float,
        goodput: float,
        evaluation: EvaluationAssumptions,
) -> TokenCost:
    normalized_goodput = _finite(
        "slo_good_output_tokens_per_second",
        goodput,
        minimum=0.0,
    )
    loaded_seconds = evaluation.lifetime_loaded_seconds
    lifetime_tokens = normalized_goodput * loaded_seconds
    dollars_per_million = (
        None
        if lifetime_tokens == 0.0
        else lifetime_tco_usd / lifetime_tokens * 1_000_000.0
    )
    return TokenCost(
        slo_good_output_tokens_per_second=normalized_goodput,
        lifetime_loaded_seconds=loaded_seconds,
        lifetime_slo_good_output_tokens=lifetime_tokens,
        lifetime_tco_usd=lifetime_tco_usd,
        dollars_per_million_slo_good_output_tokens=(
            dollars_per_million),
    )


@dataclass(frozen=True)
class OracleDesignReference:
    """Performance-only disclosure for the infinite-HBM Oracle."""

    system_key: str
    slo_good_output_tokens_per_second: Optional[float]
    physical_bom_available: bool = False
    included_in_tco_comparison: bool = False
    lifetime_tco_usd: None = None
    dollars_per_million_slo_good_output_tokens: None = None
    exclusion_reason: str = ORACLE_EXCLUSION_REASON


@dataclass(frozen=True)
class HBFDesignTCOReport:
    """Performance and TCO result for one multi-HBF design point."""

    report_schema: str
    goodput_semantics: str
    baseline_cost: SystemCost
    proposed_cost: DesignSystemCost
    baseline_token_cost: TokenCost
    proposed_token_cost: TokenCost
    oracle_reference: OracleDesignReference
    proposed_tco_ratio_to_baseline: float
    proposed_goodput_ratio_to_baseline: float
    goodput_break_even_ratio_vs_baseline: float
    break_even_proposed_goodput_tokens_per_second: float
    proposed_meets_or_exceeds_token_value_break_even: bool

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        json.dumps(value, allow_nan=False)
        return value


def evaluate_hbf_design_tco(
        *,
        hbf_host_count: int,
        active_memory: ActiveMemorySpec,
        baseline_slo_good_output_tokens_per_second: float,
        proposed_slo_good_output_tokens_per_second: float,
        oracle_slo_good_output_tokens_per_second: Optional[float] = None,
        sensitivity_point: SensitivityPoint = CENTRAL_SENSITIVITY_POINT,
        anchors: HardwareAnchors = HardwareAnchors(),
        evaluation: EvaluationAssumptions = EvaluationAssumptions(),
) -> HBFDesignTCOReport:
    """Evaluate one matched-rate multi-HBF design against the fixed baseline."""

    baseline_goodput = _finite(
        "baseline_slo_good_output_tokens_per_second",
        baseline_slo_good_output_tokens_per_second,
        strictly_positive=True,
    )
    proposed_goodput = _finite(
        "proposed_slo_good_output_tokens_per_second",
        proposed_slo_good_output_tokens_per_second,
        minimum=0.0,
    )
    oracle_goodput = (
        None
        if oracle_slo_good_output_tokens_per_second is None
        else _finite(
            "oracle_slo_good_output_tokens_per_second",
            oracle_slo_good_output_tokens_per_second,
            minimum=0.0,
        )
    )
    baseline_cost = tiering_baseline_cost(
        anchors=anchors,
        evaluation=evaluation,
    )
    proposed_cost = proposed_hbf_design_cost(
        hbf_host_count=hbf_host_count,
        active_memory=active_memory,
        sensitivity_point=sensitivity_point,
        anchors=anchors,
        evaluation=evaluation,
    )
    baseline_token_cost = _token_cost(
        baseline_cost.lifetime_tco_usd,
        baseline_goodput,
        evaluation,
    )
    proposed_token_cost = _token_cost(
        proposed_cost.lifetime_tco_usd,
        proposed_goodput,
        evaluation,
    )
    tco_ratio = (
        proposed_cost.lifetime_tco_usd
        / baseline_cost.lifetime_tco_usd
    )
    goodput_ratio = proposed_goodput / baseline_goodput
    break_even_goodput = baseline_goodput * tco_ratio
    return HBFDesignTCOReport(
        report_schema=DESIGN_REPORT_SCHEMA,
        goodput_semantics=GOODPUT_SEMANTICS,
        baseline_cost=baseline_cost,
        proposed_cost=proposed_cost,
        baseline_token_cost=baseline_token_cost,
        proposed_token_cost=proposed_token_cost,
        oracle_reference=OracleDesignReference(
            system_key=DESIGN_ORACLE_SYSTEM_KEY,
            slo_good_output_tokens_per_second=oracle_goodput,
        ),
        proposed_tco_ratio_to_baseline=tco_ratio,
        proposed_goodput_ratio_to_baseline=goodput_ratio,
        goodput_break_even_ratio_vs_baseline=tco_ratio,
        break_even_proposed_goodput_tokens_per_second=(
            break_even_goodput),
        proposed_meets_or_exceeds_token_value_break_even=(
            proposed_goodput >= break_even_goodput),
    )


__all__ = [
    "ACTIVE_MEMORY_KINDS",
    "CENTRAL_SENSITIVITY_POINT",
    "DESIGN_ORACLE_SYSTEM_KEY",
    "DESIGN_PROPOSED_SYSTEM_KEY",
    "DESIGN_REPORT_SCHEMA",
    "ActiveMemorySpec",
    "DesignSystemCost",
    "HBFDesignTCOError",
    "HBFDesignTCOReport",
    "HBFDesignTopology",
    "OracleDesignReference",
    "TokenCost",
    "evaluate_hbf_design_tco",
    "lpddr_active_memory",
    "proposed_hbf_design_cost",
]

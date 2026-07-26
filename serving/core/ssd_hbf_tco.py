"""Five-year TCO for two GPU+SSD hosts versus one GPU+SSD plus HBF.

Its two finite systems match the physical comparison used by the HBF
evaluation:

* baseline: two eight-H100 P4D4 hosts and sixteen local NVMe SSDs;
* proposed: one such GPU+SSD host plus one eight-card HBF host.

HBF ``tp4x2`` and ``tp8`` are serving layouts inside the same physical
eight-card HBF server.  Selecting a layout never changes host or card counts.
Each HBF card uses the same H100-class GPU-logic performance, power, and
CAPEX anchors as the baseline H100 cards.  HBM is not duplicated on those
cards: HBF media and active memory are priced as separate subsystems.  Other
prices, power, sensitivity ratios, and active-memory assumptions are reused
from :mod:`hbf_design_tco` and :mod:`hbf_comparison_tco`; this module adds no
new component-price anchors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
)
from .hbf_design_tco import (
    ActiveMemorySpec,
    CENTRAL_SENSITIVITY_POINT,
    GOODPUT_SEMANTICS,
    ORACLE_EXCLUSION_REASON,
    lpddr_active_memory,
)


SSD_HBF_REPORT_SCHEMA = "two-gpu-vs-one-gpu-one-hbf-ssd-tco-v3"
SSD_BASELINE_SYSTEM_KEY = "two_gpu_local_ssd_baseline"
SSD_HBF_PROPOSED_SYSTEM_KEY = "one_gpu_local_ssd_plus_one_hbf"
ORACLE_SYSTEM_KEY = "two_gpu_infinite_hbm_oracle"
HBF_LAYOUT_KEYS = ("tp4x2", "tp8")
BASELINE_GPU_HOST_COUNT = 2
PROPOSED_GPU_HOST_COUNT = 1
HBF_HOST_COUNT = 1
H100_CARD_COUNT = 8
HBF_CARD_COUNT = 8
LOCAL_SSD_DEVICE_COUNT = 8
FIVE_YEAR_LIFETIME = 5.0
LAYOUT_COUNT_SEMANTICS = (
    "tp4x2 and tp8 are logical serving layouts within one physical "
    "eight-card HBF server; they do not change host, card, SSD, NIC, or "
    "fabric counts"
)


class SSDHBFTCOError(ValueError):
    """Raised when the fixed two-GPU versus GPU+HBF contract is violated."""


def _finite(
        name: str, value: object, *, minimum: Optional[float] = None,
        strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SSDHBFTCOError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise SSDHBFTCOError(f"{name} must be a finite number")
    if strictly_positive and converted <= 0.0:
        raise SSDHBFTCOError(f"{name} must be positive")
    if minimum is not None and converted < minimum:
        raise SSDHBFTCOError(
            f"{name} must be at least {minimum}")
    return converted


def _validate_evaluation(evaluation: EvaluationAssumptions) -> None:
    if not isinstance(evaluation, EvaluationAssumptions):
        raise SSDHBFTCOError(
            "evaluation must be EvaluationAssumptions")
    if not math.isclose(
            evaluation.lifetime_years,
            FIVE_YEAR_LIFETIME,
            rel_tol=0.0,
            abs_tol=1e-12,
    ):
        raise SSDHBFTCOError(
            "this evaluator reports a fixed five-year TCO; "
            "evaluation.lifetime_years must be 5")


@dataclass(frozen=True)
class HBFServerLayout:
    """Logical placement within one physical eight-card HBF server."""

    key: str
    tensor_parallel_size: int
    independent_serving_replicas: int
    cards_per_replica: int
    physical_hbf_hosts: int = 1
    physical_hbf_cards: int = 8
    count_semantics: str = LAYOUT_COUNT_SEMANTICS

    def __post_init__(self) -> None:
        if self.key not in HBF_LAYOUT_KEYS:
            raise SSDHBFTCOError(
                f"layout key must be one of {HBF_LAYOUT_KEYS!r}")
        expected = {
            "tp4x2": (4, 2, 4),
            "tp8": (8, 1, 8),
        }[self.key]
        actual = (
            self.tensor_parallel_size,
            self.independent_serving_replicas,
            self.cards_per_replica,
        )
        if actual != expected:
            raise SSDHBFTCOError(
                f"{self.key} layout geometry must be {expected!r}")
        if (
            self.physical_hbf_hosts != HBF_HOST_COUNT
            or self.physical_hbf_cards != HBF_CARD_COUNT
        ):
            raise SSDHBFTCOError(
                "every supported layout must use one eight-card HBF host")
        if (
            self.independent_serving_replicas
            * self.cards_per_replica
            != self.physical_hbf_cards
        ):
            raise SSDHBFTCOError(
                "layout replicas must consume exactly eight HBF cards")

    @classmethod
    def for_key(cls, key: str) -> "HBFServerLayout":
        aliases = {
            "tp4x2": "tp4x2",
            "tp4*2": "tp4x2",
            "tp4": "tp4x2",
            "tp8": "tp8",
        }
        if not isinstance(key, str):
            raise SSDHBFTCOError(
                f"layout key must be one of {HBF_LAYOUT_KEYS!r}")
        try:
            canonical = aliases[key]
        except KeyError as exc:
            raise SSDHBFTCOError(
                f"layout key must be one of {HBF_LAYOUT_KEYS!r}"
            ) from exc
        if canonical == "tp4x2":
            return cls(
                key=canonical,
                tensor_parallel_size=4,
                independent_serving_replicas=2,
                cards_per_replica=4,
            )
        return cls(
            key=canonical,
            tensor_parallel_size=8,
            independent_serving_replicas=1,
            cards_per_replica=8,
        )


@dataclass(frozen=True)
class PhysicalComponentCounts:
    """Purchasable system counts, independent of serving concurrency."""

    cpu_hosts: int
    gpu_hosts: int
    hbf_hosts: int
    h100_cards: int
    hbf_cards: int
    local_ssd_devices: int
    network_nics: int
    network_fabric_units: int
    gpu_intraserver_fabric_units: int
    hbf_intraserver_fabric_units: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise SSDHBFTCOError(
                    f"{name} must be a non-negative integer")
        if self.cpu_hosts != self.gpu_hosts + self.hbf_hosts:
            raise SSDHBFTCOError(
                "cpu_hosts must equal GPU plus HBF hosts")


@dataclass(frozen=True)
class TwoGPUOneHBFComparisonTopology:
    """The fixed two-GPU baseline and one-GPU/one-HBF proposal."""

    hbf_layout: HBFServerLayout
    baseline: PhysicalComponentCounts = field(init=False)
    proposed: PhysicalComponentCounts = field(init=False)
    count_semantics: str = LAYOUT_COUNT_SEMANTICS

    def __post_init__(self) -> None:
        if not isinstance(self.hbf_layout, HBFServerLayout):
            raise SSDHBFTCOError(
                "hbf_layout must be HBFServerLayout")
        object.__setattr__(
            self,
            "baseline",
            PhysicalComponentCounts(
                cpu_hosts=BASELINE_GPU_HOST_COUNT,
                gpu_hosts=BASELINE_GPU_HOST_COUNT,
                hbf_hosts=0,
                h100_cards=(
                    BASELINE_GPU_HOST_COUNT * H100_CARD_COUNT),
                hbf_cards=0,
                local_ssd_devices=(
                    BASELINE_GPU_HOST_COUNT * LOCAL_SSD_DEVICE_COUNT),
                network_nics=BASELINE_GPU_HOST_COUNT,
                network_fabric_units=1,
                gpu_intraserver_fabric_units=BASELINE_GPU_HOST_COUNT,
                hbf_intraserver_fabric_units=0,
            ),
        )
        object.__setattr__(
            self,
            "proposed",
            PhysicalComponentCounts(
                cpu_hosts=PROPOSED_GPU_HOST_COUNT + HBF_HOST_COUNT,
                gpu_hosts=PROPOSED_GPU_HOST_COUNT,
                hbf_hosts=HBF_HOST_COUNT,
                h100_cards=H100_CARD_COUNT,
                hbf_cards=HBF_CARD_COUNT,
                local_ssd_devices=LOCAL_SSD_DEVICE_COUNT,
                network_nics=2,
                network_fabric_units=1,
                gpu_intraserver_fabric_units=1,
                hbf_intraserver_fabric_units=1,
            ),
        )

    @classmethod
    def for_layout(
            cls, layout: str | HBFServerLayout,
    ) -> "TwoGPUOneHBFComparisonTopology":
        normalized = (
            layout
            if isinstance(layout, HBFServerLayout)
            else HBFServerLayout.for_key(layout)
        )
        return cls(hbf_layout=normalized)


def _bom_line(
        key: str, label: str, unit: str, quantity: float,
        unit_capex_usd: float, unit_power_w: float,
        assumption: str) -> BOMLine:
    """Build one line using the existing shared BOM schema."""

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
class SSDHBFSystemCost:
    """Auditable component and five-year energy cost for one system."""

    system_key: str
    physical_description: str
    counts: PhysicalComponentCounts
    hbf_layout: Optional[HBFServerLayout]
    active_memory: Optional[ActiveMemorySpec]
    sensitivity_point: Optional[SensitivityPoint]
    bom: tuple[BOMLine, ...]
    capex_usd: float
    it_power_w: float
    facility_power_w: float
    five_year_it_energy_kwh: float
    five_year_facility_energy_kwh: float
    five_year_electricity_opex_usd: float
    five_year_tco_usd: float
    evaluation: EvaluationAssumptions
    price_source_semantics: str = PRICE_SOURCE_SEMANTICS
    excluded_costs: tuple[str, ...] = (
        "labor", "maintenance", "financing", "taxes")

    def __post_init__(self) -> None:
        if self.system_key not in {
            SSD_BASELINE_SYSTEM_KEY,
            SSD_HBF_PROPOSED_SYSTEM_KEY,
        }:
            raise SSDHBFTCOError("unexpected finite system key")
        if (
            not isinstance(self.physical_description, str)
            or not self.physical_description
        ):
            raise SSDHBFTCOError(
                "physical_description must be non-empty")
        if not isinstance(self.counts, PhysicalComponentCounts):
            raise SSDHBFTCOError(
                "counts must be PhysicalComponentCounts")
        _validate_evaluation(self.evaluation)
        if not self.bom:
            raise SSDHBFTCOError("bom must not be empty")
        keys = [line.component_key for line in self.bom]
        if len(keys) != len(set(keys)):
            raise SSDHBFTCOError(
                "BOM component keys must be unique")
        if self.system_key == SSD_BASELINE_SYSTEM_KEY:
            expected_counts = (
                TwoGPUOneHBFComparisonTopology.for_layout(
                    "tp4x2").baseline
            )
            if (
                self.hbf_layout is not None
                or self.active_memory is not None
                or self.sensitivity_point is not None
                or self.counts != expected_counts
            ):
                raise SSDHBFTCOError(
                    "baseline must contain exactly two GPU+SSD hosts")
        else:
            expected_counts = (
                TwoGPUOneHBFComparisonTopology.for_layout(
                    self.hbf_layout
                    if isinstance(self.hbf_layout, HBFServerLayout)
                    else "tp4x2"
                ).proposed
            )
            if (
                not isinstance(self.hbf_layout, HBFServerLayout)
                or not isinstance(self.active_memory, ActiveMemorySpec)
                or not isinstance(
                    self.sensitivity_point, SensitivityPoint)
                or self.counts != expected_counts
            ):
                raise SSDHBFTCOError(
                    "proposed cost requires exactly one GPU+SSD host and "
                    "one complete HBF server")
        expected_capex = math.fsum(
            line.capex_usd for line in self.bom)
        expected_power = math.fsum(
            line.it_power_w for line in self.bom)
        if not math.isclose(
            self.capex_usd,
            expected_capex,
            rel_tol=1e-12,
            abs_tol=1e-8,
        ):
            raise SSDHBFTCOError("capex does not match BOM")
        if not math.isclose(
            self.it_power_w,
            expected_power,
            rel_tol=1e-12,
            abs_tol=1e-8,
        ):
            raise SSDHBFTCOError("IT power does not match BOM")
        for name in (
            "capex_usd",
            "it_power_w",
            "facility_power_w",
            "five_year_it_energy_kwh",
            "five_year_facility_energy_kwh",
            "five_year_electricity_opex_usd",
            "five_year_tco_usd",
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


def _finalize_cost(
        *,
        system_key: str,
        physical_description: str,
        counts: PhysicalComponentCounts,
        hbf_layout: Optional[HBFServerLayout],
        active_memory: Optional[ActiveMemorySpec],
        sensitivity_point: Optional[SensitivityPoint],
        bom: tuple[BOMLine, ...],
        evaluation: EvaluationAssumptions,
) -> SSDHBFSystemCost:
    _validate_evaluation(evaluation)
    capex = math.fsum(line.capex_usd for line in bom)
    it_power = math.fsum(line.it_power_w for line in bom)
    facility_power = it_power * evaluation.pue
    equivalent_hours = (
        evaluation.lifetime_powered_equivalent_full_load_hours)
    it_energy = it_power * equivalent_hours * KWH_PER_WH
    facility_energy = (
        facility_power * equivalent_hours * KWH_PER_WH)
    electricity = (
        facility_energy * evaluation.electricity_usd_per_kwh)
    return SSDHBFSystemCost(
        system_key=system_key,
        physical_description=physical_description,
        counts=counts,
        hbf_layout=hbf_layout,
        active_memory=active_memory,
        sensitivity_point=sensitivity_point,
        bom=bom,
        capex_usd=capex,
        it_power_w=it_power,
        facility_power_w=facility_power,
        five_year_it_energy_kwh=it_energy,
        five_year_facility_energy_kwh=facility_energy,
        five_year_electricity_opex_usd=electricity,
        five_year_tco_usd=capex + electricity,
        evaluation=evaluation,
    )


def _gpu_ssd_bom(
        anchors: HardwareAnchors, *,
        gpu_host_count: int,
        baseline_network: bool,
) -> tuple[BOMLine, ...]:
    if (
        isinstance(gpu_host_count, bool)
        or not isinstance(gpu_host_count, int)
        or gpu_host_count <= 0
    ):
        raise SSDHBFTCOError(
            "gpu_host_count must be a positive integer")
    host_dram_gib = (
        P4D4_CPU_MEMORY_BYTES_PER_HOST / BYTES_PER_GIB)
    h100_card_count = gpu_host_count * H100_CARD_COUNT
    ssd_device_count = gpu_host_count * LOCAL_SSD_DEVICE_COUNT
    nic_capex = (
        anchors.baseline_nic_capex_usd
        if baseline_network
        else anchors.rdma_nic_capex_usd
    )
    nic_power = (
        anchors.baseline_nic_power_w
        if baseline_network
        else anchors.rdma_nic_power_w
    )
    fabric_capex = (
        anchors.baseline_fabric_capex_usd
        if baseline_network
        else anchors.rdma_fabric_capex_usd
    )
    fabric_power = (
        anchors.baseline_fabric_power_w
        if baseline_network
        else anchors.rdma_fabric_power_w
    )
    return (
        _bom_line(
            "gpu_cpu_host_base",
            "GPU CPU-server base",
            "host",
            gpu_host_count,
            anchors.cpu_host_base_capex_usd,
            anchors.cpu_host_base_power_w,
            (
                "Accelerators, DRAM, SSDs, NIC, and fabrics are priced "
                "separately using the existing host anchor."
            ),
        ),
        _bom_line(
            "gpu_host_dram",
            "GPU-host DRAM",
            "GiB",
            gpu_host_count * host_dram_gib,
            anchors.host_dram_capex_usd_per_gib,
            anchors.host_dram_power_w_per_gib,
            "The P4D4 host retains its configured 512e9-byte DRAM tier.",
        ),
        _bom_line(
            "h100_gpu_logic",
            "H100 GPU logic excluding HBM",
            "card",
            h100_card_count,
            anchors.gpu_logic_capex_usd_per_card,
            anchors.gpu_logic_power_w_per_card,
            f"Exactly {gpu_host_count} eight-H100 GPU host(s) are present.",
        ),
        _bom_line(
            "h100_hbm_stack",
            "Complete H100 HBM stack",
            "card",
            h100_card_count,
            anchors.hbm_stack_capex_usd_per_card,
            anchors.hbm_stack_power_w_per_card,
            "HBM is priced separately from H100 GPU logic.",
        ),
        _bom_line(
            "gpu_intraserver_fabric",
            "GPU-host NVSwitch/NVLink fabric",
            "host fabric unit",
            gpu_host_count,
            anchors.gpu_intraserver_fabric_capex_usd_per_unit,
            anchors.gpu_intraserver_fabric_power_w_per_unit,
            "Every GPU host retains one accelerator fabric allocation.",
        ),
        _bom_line(
            "gpu_local_nvme_ssd",
            "GPU-host local NVMe SSD tier",
            "device",
            ssd_device_count,
            anchors.nvme_ssd_capex_usd_per_device,
            anchors.nvme_ssd_power_w_per_device,
            "Every GPU host retains eight local SSD devices.",
        ),
        _bom_line(
            "gpu_host_network_nic",
            "GPU-host network NIC",
            "NIC",
            gpu_host_count,
            nic_capex,
            nic_power,
            (
                "Every GPU host receives one network NIC; the baseline "
                "and proposed systems use their respective established "
                "network-price anchors."
            ),
        ),
        _bom_line(
            "network_fabric",
            "Shared external network fabric allocation",
            "fabric unit",
            1,
            fabric_capex,
            fabric_power,
            (
                "One system-level fabric allocation is present in each "
                "deployment; the proposed HBF host joins its RDMA fabric."
            ),
        ),
    )


def two_gpu_local_ssd_baseline_cost(
        *,
        anchors: HardwareAnchors = HardwareAnchors(),
        evaluation: EvaluationAssumptions = EvaluationAssumptions(),
) -> SSDHBFSystemCost:
    """Price exactly two P4D4 GPU hosts and sixteen local SSDs."""

    if not isinstance(anchors, HardwareAnchors):
        raise SSDHBFTCOError("anchors must be HardwareAnchors")
    _validate_evaluation(evaluation)
    counts = TwoGPUOneHBFComparisonTopology.for_layout(
        "tp4x2").baseline
    return _finalize_cost(
        system_key=SSD_BASELINE_SYSTEM_KEY,
        physical_description=(
            "Two independent-serving P4D4 eight-H100 GPU hosts with host "
            "DRAM, two GPU fabrics, sixteen local NVMe SSDs, two NICs, "
            "and one priced system-level network-fabric allocation"
        ),
        counts=counts,
        hbf_layout=None,
        active_memory=None,
        sensitivity_point=None,
        bom=_gpu_ssd_bom(
            anchors,
            gpu_host_count=2,
            baseline_network=True,
        ),
        evaluation=evaluation,
    )


def one_gpu_one_hbf_cost(
        *,
        hbf_layout: str | HBFServerLayout,
        active_memory: ActiveMemorySpec,
        sensitivity_point: SensitivityPoint = CENTRAL_SENSITIVITY_POINT,
        anchors: HardwareAnchors = HardwareAnchors(),
        evaluation: EvaluationAssumptions = EvaluationAssumptions(),
) -> SSDHBFSystemCost:
    """Price one GPU+SSD host plus exactly one HBF server."""

    topology = TwoGPUOneHBFComparisonTopology.for_layout(hbf_layout)
    if not isinstance(active_memory, ActiveMemorySpec):
        raise SSDHBFTCOError(
            "active_memory must be ActiveMemorySpec")
    if not isinstance(sensitivity_point, SensitivityPoint):
        raise SSDHBFTCOError(
            "sensitivity_point must be SensitivityPoint")
    if not isinstance(anchors, HardwareAnchors):
        raise SSDHBFTCOError("anchors must be HardwareAnchors")
    _validate_evaluation(evaluation)

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
        HBF_CARD_COUNT * active_memory.capacity_gib_per_card)
    hbf_bom = (
        _bom_line(
            "hbf_cpu_host_base",
            "HBF CPU-server base",
            "host",
            1,
            anchors.cpu_host_base_capex_usd,
            anchors.cpu_host_base_power_w,
            "Exactly one additional HBF host uses the shared host anchor.",
        ),
        _bom_line(
            "hbf_host_dram",
            "HBF-host DRAM",
            "GiB",
            host_dram_gib,
            anchors.host_dram_capex_usd_per_gib,
            anchors.host_dram_power_w_per_gib,
            "The HBF host receives the same 512e9-byte DRAM allocation.",
        ),
        _bom_line(
            "hbf_gpu_logic",
            "H100-class HBF-card GPU logic excluding HBF and active memory",
            "card",
            HBF_CARD_COUNT,
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
            HBF_CARD_COUNT,
            hbf_subsystem_capex,
            hbf_subsystem_power,
            (
                "Normalized sensitivity multipliers are applied to "
                "independent HBF media/controller CAPEX and power anchors."
            ),
        ),
        _bom_line(
            "hbf_gpu_intraserver_fabric",
            "HBF-GPU-host PCIe fabric",
            "host fabric unit",
            1,
            anchors.hbf_npu_pcie_fabric_capex_usd_per_unit,
            anchors.hbf_npu_pcie_fabric_power_w_per_unit,
            "One eight-card HBF host has one PCIe fabric allocation.",
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
            "hbf_host_rdma_nic",
            "HBF-host RDMA NIC",
            "NIC",
            1,
            anchors.rdma_nic_capex_usd,
            anchors.rdma_nic_power_w,
            (
                "One HBF-side NIC is added to the GPU host's existing "
                "NIC and shared fabric."
            ),
        ),
    )
    return _finalize_cost(
        system_key=SSD_HBF_PROPOSED_SYSTEM_KEY,
        physical_description=(
            "One eight-H100 GPU host with eight local SSDs plus one "
            "eight-card HBF server, compared with the two-GPU baseline; "
            "HBF layout "
            f"{topology.hbf_layout.key} uses "
            f"{topology.hbf_layout.independent_serving_replicas} "
            "internal serving replica(s) without changing physical counts"
        ),
        counts=topology.proposed,
        hbf_layout=topology.hbf_layout,
        active_memory=active_memory,
        sensitivity_point=sensitivity_point,
        bom=(
            _gpu_ssd_bom(
                anchors,
                gpu_host_count=1,
                baseline_network=False,
            )
            + hbf_bom
        ),
        evaluation=evaluation,
    )


@dataclass(frozen=True)
class FiveYearTokenEconomics:
    slo_good_output_tokens_per_second: float
    five_year_loaded_seconds: float
    five_year_slo_good_output_tokens: float
    five_year_tco_usd: float
    tco_usd_per_million_slo_good_output_tokens: Optional[float]


def _token_economics(
        cost: SSDHBFSystemCost, goodput: float,
) -> FiveYearTokenEconomics:
    normalized = _finite(
        "slo_good_output_tokens_per_second",
        goodput,
        minimum=0.0,
    )
    loaded_seconds = cost.evaluation.lifetime_loaded_seconds
    tokens = normalized * loaded_seconds
    per_million = (
        None
        if tokens == 0.0
        else cost.five_year_tco_usd / tokens * 1_000_000.0
    )
    return FiveYearTokenEconomics(
        slo_good_output_tokens_per_second=normalized,
        five_year_loaded_seconds=loaded_seconds,
        five_year_slo_good_output_tokens=tokens,
        five_year_tco_usd=cost.five_year_tco_usd,
        tco_usd_per_million_slo_good_output_tokens=per_million,
    )


@dataclass(frozen=True)
class PerformanceOnlyOracle:
    system_key: str
    slo_good_output_tokens_per_second: Optional[float]
    physical_bom_available: bool = False
    included_in_tco_comparison: bool = False
    five_year_tco_usd: None = None
    tco_usd_per_million_slo_good_output_tokens: None = None
    exclusion_reason: str = ORACLE_EXCLUSION_REASON


@dataclass(frozen=True)
class SSDHBFCostDelta:
    capex_usd: float
    it_power_w: float
    facility_power_w: float
    five_year_it_energy_kwh: float
    five_year_facility_energy_kwh: float
    five_year_electricity_opex_usd: float
    five_year_tco_usd: float


@dataclass(frozen=True)
class PowerEnergyComparison:
    """Direct baseline/proposed power and five-year energy comparison."""

    lifetime_years: float
    baseline_it_power_w: float
    proposed_it_power_w: float
    incremental_it_power_w: float
    proposed_it_power_ratio_to_baseline: float
    baseline_facility_power_w: float
    proposed_facility_power_w: float
    incremental_facility_power_w: float
    proposed_facility_power_ratio_to_baseline: float
    baseline_five_year_it_energy_kwh: float
    proposed_five_year_it_energy_kwh: float
    incremental_five_year_it_energy_kwh: float
    proposed_it_energy_ratio_to_baseline: float
    baseline_five_year_facility_energy_kwh: float
    proposed_five_year_facility_energy_kwh: float
    incremental_five_year_facility_energy_kwh: float
    proposed_facility_energy_ratio_to_baseline: float
    semantics: str = (
        "BOM active-power assumptions projected over five years using "
        "EvaluationAssumptions utilization, idle-power fraction, and PUE; "
        "not event-derived runtime energy."
    )


@dataclass(frozen=True)
class SSDHBFTCOReport:
    report_schema: str
    goodput_semantics: str
    topology: TwoGPUOneHBFComparisonTopology
    baseline_cost: SSDHBFSystemCost
    proposed_cost: SSDHBFSystemCost
    cost_delta_proposed_minus_baseline: SSDHBFCostDelta
    power_energy_comparison: PowerEnergyComparison
    baseline_token_economics: FiveYearTokenEconomics
    proposed_token_economics: FiveYearTokenEconomics
    oracle_reference: PerformanceOnlyOracle
    proposed_tco_ratio_to_baseline: float
    proposed_goodput_ratio_to_baseline: float
    required_goodput_ratio_for_equal_token_cost: float
    required_proposed_goodput_tokens_per_second: float
    proposed_meets_or_exceeds_goodput_break_even: bool
    layout_count_semantics: str = LAYOUT_COUNT_SEMANTICS

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        json.dumps(value, allow_nan=False)
        return value


def evaluate_ssd_hbf_tco(
        *,
        hbf_layout: str | HBFServerLayout,
        active_memory: ActiveMemorySpec,
        baseline_slo_good_output_tokens_per_second: float,
        proposed_slo_good_output_tokens_per_second: float,
        oracle_slo_good_output_tokens_per_second: Optional[float] = None,
        sensitivity_point: SensitivityPoint = CENTRAL_SENSITIVITY_POINT,
        anchors: HardwareAnchors = HardwareAnchors(),
        evaluation: EvaluationAssumptions = EvaluationAssumptions(),
) -> SSDHBFTCOReport:
    """Evaluate matched-rate goodput for two GPU hosts versus GPU+HBF."""

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
    topology = TwoGPUOneHBFComparisonTopology.for_layout(hbf_layout)
    baseline = two_gpu_local_ssd_baseline_cost(
        anchors=anchors,
        evaluation=evaluation,
    )
    proposed = one_gpu_one_hbf_cost(
        hbf_layout=topology.hbf_layout,
        active_memory=active_memory,
        sensitivity_point=sensitivity_point,
        anchors=anchors,
        evaluation=evaluation,
    )
    tco_ratio = (
        proposed.five_year_tco_usd
        / baseline.five_year_tco_usd
    )
    required_goodput = baseline_goodput * tco_ratio
    return SSDHBFTCOReport(
        report_schema=SSD_HBF_REPORT_SCHEMA,
        goodput_semantics=GOODPUT_SEMANTICS,
        topology=topology,
        baseline_cost=baseline,
        proposed_cost=proposed,
        cost_delta_proposed_minus_baseline=SSDHBFCostDelta(
            capex_usd=proposed.capex_usd - baseline.capex_usd,
            it_power_w=proposed.it_power_w - baseline.it_power_w,
            facility_power_w=(
                proposed.facility_power_w
                - baseline.facility_power_w
            ),
            five_year_it_energy_kwh=(
                proposed.five_year_it_energy_kwh
                - baseline.five_year_it_energy_kwh
            ),
            five_year_facility_energy_kwh=(
                proposed.five_year_facility_energy_kwh
                - baseline.five_year_facility_energy_kwh
            ),
            five_year_electricity_opex_usd=(
                proposed.five_year_electricity_opex_usd
                - baseline.five_year_electricity_opex_usd
            ),
            five_year_tco_usd=(
                proposed.five_year_tco_usd
                - baseline.five_year_tco_usd
            ),
        ),
        power_energy_comparison=PowerEnergyComparison(
            lifetime_years=evaluation.lifetime_years,
            baseline_it_power_w=baseline.it_power_w,
            proposed_it_power_w=proposed.it_power_w,
            incremental_it_power_w=(
                proposed.it_power_w - baseline.it_power_w),
            proposed_it_power_ratio_to_baseline=(
                proposed.it_power_w / baseline.it_power_w),
            baseline_facility_power_w=baseline.facility_power_w,
            proposed_facility_power_w=proposed.facility_power_w,
            incremental_facility_power_w=(
                proposed.facility_power_w
                - baseline.facility_power_w
            ),
            proposed_facility_power_ratio_to_baseline=(
                proposed.facility_power_w
                / baseline.facility_power_w
            ),
            baseline_five_year_it_energy_kwh=(
                baseline.five_year_it_energy_kwh),
            proposed_five_year_it_energy_kwh=(
                proposed.five_year_it_energy_kwh),
            incremental_five_year_it_energy_kwh=(
                proposed.five_year_it_energy_kwh
                - baseline.five_year_it_energy_kwh
            ),
            proposed_it_energy_ratio_to_baseline=(
                proposed.five_year_it_energy_kwh
                / baseline.five_year_it_energy_kwh
            ),
            baseline_five_year_facility_energy_kwh=(
                baseline.five_year_facility_energy_kwh),
            proposed_five_year_facility_energy_kwh=(
                proposed.five_year_facility_energy_kwh),
            incremental_five_year_facility_energy_kwh=(
                proposed.five_year_facility_energy_kwh
                - baseline.five_year_facility_energy_kwh
            ),
            proposed_facility_energy_ratio_to_baseline=(
                proposed.five_year_facility_energy_kwh
                / baseline.five_year_facility_energy_kwh
            ),
        ),
        baseline_token_economics=_token_economics(
            baseline, baseline_goodput),
        proposed_token_economics=_token_economics(
            proposed, proposed_goodput),
        oracle_reference=PerformanceOnlyOracle(
            system_key=ORACLE_SYSTEM_KEY,
            slo_good_output_tokens_per_second=oracle_goodput,
        ),
        proposed_tco_ratio_to_baseline=tco_ratio,
        proposed_goodput_ratio_to_baseline=(
            proposed_goodput / baseline_goodput),
        required_goodput_ratio_for_equal_token_cost=tco_ratio,
        required_proposed_goodput_tokens_per_second=(
            required_goodput),
        proposed_meets_or_exceeds_goodput_break_even=(
            proposed_goodput >= required_goodput),
    )


__all__ = [
    "HBF_LAYOUT_KEYS",
    "LAYOUT_COUNT_SEMANTICS",
    "ORACLE_SYSTEM_KEY",
    "SSD_BASELINE_SYSTEM_KEY",
    "SSD_HBF_PROPOSED_SYSTEM_KEY",
    "SSD_HBF_REPORT_SCHEMA",
    "HBFServerLayout",
    "TwoGPUOneHBFComparisonTopology",
    "PerformanceOnlyOracle",
    "PhysicalComponentCounts",
    "SSDHBFCostDelta",
    "SSDHBFSystemCost",
    "SSDHBFTCOError",
    "SSDHBFTCOReport",
    "FIVE_YEAR_LIFETIME",
    "FiveYearTokenEconomics",
    "PowerEnergyComparison",
    "evaluate_ssd_hbf_tco",
    "lpddr_active_memory",
    "one_gpu_one_hbf_cost",
    "two_gpu_local_ssd_baseline_cost",
]

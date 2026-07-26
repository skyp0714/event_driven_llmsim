"""Runtime energy and five-year TCO for the SSD/HBF comparison.

This module deliberately does not sum every byte counter in a
``ResourceCalendar``.  One physical transfer is often represented at a
rank lane, a root port, and a memory endpoint at the same time.  Energy is
instead assigned to one exclusive driver per physical component:

* H100 cards use P/D model-active card time and include their HBM;
* HBF GPU logic uses HBF model-active card time;
* HBF media/controllers use per-card media busy time;
* SSDs use their aggregate read/write queue busy time;
* CPU DRAM, PCIe, and network links use one canonical byte counter each;
* P/D and HBF fabrics use their aggregate fabric resources, never lanes.

The simulated horizon already includes arrivals, tool gaps, and idle
periods.  Its measured average power is therefore projected directly over
five calendar years.  No second utilization multiplier is applied.
Runtime electricity replaces, rather than augments, the static-BOM
electricity estimate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import re
from typing import Any, Mapping, Optional, Sequence


NS_PER_SECOND = 1_000_000_000
HOURS_PER_YEAR = 8_760.0
BYTES_PER_GIB = 1024 ** 3
BASELINE_SYSTEM_KEY = "two_gpu_local_ssd_baseline"
PROPOSED_SYSTEM_KEY = "one_gpu_local_ssd_plus_one_hbf"
RUNTIME_ENERGY_SCHEMA = "ssd-hbf-runtime-energy-v1"
RUNTIME_TCO_SCHEMA = "ssd-hbf-runtime-tco-v1"

_GPU_ROOT_RE = re.compile(r"^gpu-node-\d+-pcie-root-\d+$")
_CPU_DRAM_RE = re.compile(r"^gpu-node-\d+-cpu-dram$")
_SSD_READ_RE = re.compile(r"^gpu-node-\d+-ssd-read$")
_SSD_WRITE_RE = re.compile(r"^gpu-node-\d+-ssd-write$")
_PD_FABRIC_RE = re.compile(r"^gpu-node-\d+-pd-fabric$")
_HBF_ROOT_RE = re.compile(
    r"^(?:hbf-server-\d+-)?hbf-pcie-root-\d+$")
_HBF_MEDIA_RE = re.compile(
    r"^(?:hbf-server-\d+-)?hbf-card-(\d+)-media$")
_HBF_LPDDR_RE = re.compile(
    r"^(?:hbf-server-\d+-)?hbf-card-(\d+)-lpddr$")
_HBF_FABRIC_RE = re.compile(
    r"^(?:hbf-server-\d+-)?hbf-group-\d+-fabric$")


class SSDHBFRuntimeEnergyError(ValueError):
    """Raised when a report cannot support exclusive runtime accounting."""


def _finite(
        name: str,
        value: object,
        *,
        minimum: Optional[float] = None,
        strictly_positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SSDHBFRuntimeEnergyError(
            f"{name} must be a finite number")
    converted = float(value)
    if strictly_positive and converted <= 0.0:
        raise SSDHBFRuntimeEnergyError(f"{name} must be positive")
    if minimum is not None and converted < minimum:
        raise SSDHBFRuntimeEnergyError(
            f"{name} must be at least {minimum}")
    return converted


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SSDHBFRuntimeEnergyError(
            f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise SSDHBFRuntimeEnergyError(
            f"{name} must be a non-negative integer")
    return value


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SSDHBFRuntimeEnergyError(f"{path} must be an object")
    return value


def _child(
        parent: Mapping[str, Any], key: str, path: str,
) -> Mapping[str, Any]:
    if key not in parent:
        raise SSDHBFRuntimeEnergyError(f"{path}.{key} is required")
    return _mapping(parent[key], f"{path}.{key}")


def _int_field(
        parent: Mapping[str, Any],
        key: str,
        path: str,
        *,
        positive: bool = False,
) -> int:
    if key not in parent:
        raise SSDHBFRuntimeEnergyError(f"{path}.{key} is required")
    return (
        _positive_int(f"{path}.{key}", parent[key])
        if positive
        else _nonnegative_int(f"{path}.{key}", parent[key])
    )


def _strict_json_dict(value: object) -> dict[str, Any]:
    converted = asdict(value)  # type: ignore[arg-type]
    json.dumps(converted, allow_nan=False)
    return converted


@dataclass(frozen=True)
class RuntimePowerSource:
    source_key: str
    title: str
    url: str
    supports: str

    def __post_init__(self) -> None:
        for name in ("source_key", "title", "url", "supports"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SSDHBFRuntimeEnergyError(
                    f"power source {name} must be non-empty")


DEFAULT_RUNTIME_POWER_SOURCES = (
    RuntimePowerSource(
        source_key="nvidia_h100_sxm",
        title="NVIDIA H100 Tensor Core GPU product specifications",
        url="https://www.nvidia.com/en-us/data-center/h100/",
        supports=(
            "H100 SXM maximum configurable TDP of 700 W; runtime idle "
            "power remains an explicit analytical assumption"
        ),
    ),
    RuntimePowerSource(
        source_key="flashaccel_hbf",
        title=(
            "FlashAccel: Leveraging High-Bandwidth Flash for "
            "High-Throughput LLM Inference"
        ),
        url="https://arxiv.org/html/2607.10186#S7.SS6",
        supports=(
            "HBF read energy of 8 pJ/bit and whole-accelerator TDP "
            "ratios of 1.23x for CLI and 1.31x for CSI"
        ),
    ),
    RuntimePowerSource(
        source_key="micron_9550_pro_3_84tb",
        title="Micron 9550 SSD Series Technical Product Specification",
        url=(
            "https://www.micron.com/content/dam/micron/global/public/"
            "products/data-sheet/ssd/9550-nvme-ssd-tech-prod-spec.pdf"
        ),
        supports=(
            "3.84 TB PRO maximum active read/write power of 18/19 W "
            "and average idle power of 5 W"
        ),
    ),
    RuntimePowerSource(
        source_key="repository_power_config",
        title="LLMServingSim analytical power configuration",
        url="configs/cluster/single_node_power_instance.json",
        supports=(
            "Repository sensitivity anchors of 6 pJ/bit for DRAM and "
            "4 pJ/bit for links"
        ),
    ),
)


@dataclass(frozen=True)
class RuntimePowerAssumptions:
    """Power coefficients used with event-derived runtime activity.

    H100 power is a whole-card value and therefore already includes HBM.
    The HBF GPU-logic/media split is calibrated to about 1.23x H100 at
    simultaneous full activity: ``(560 + 300) / 700 = 1.229``.  HBF media
    uses busy-time power instead of adding a second 8-pJ/bit charge.
    """

    lifetime_years: float = 5.0
    pue: float = 1.20
    electricity_usd_per_kwh: float = 0.10

    h100_active_power_w_per_card: float = 700.0
    h100_idle_power_w_per_card: float = 70.0
    hbf_gpu_logic_active_power_w_per_card: float = 560.0
    hbf_gpu_logic_idle_power_w_per_card: float = 56.0
    hbf_media_controller_active_power_w_per_card: float = 300.0
    hbf_media_controller_idle_power_w_per_card: float = 30.0

    cpu_host_active_power_w: float = 800.0
    cpu_host_idle_power_w: float = 400.0
    hbf_host_dram_capacity_bytes: int = 512_000_000_000
    host_dram_idle_power_w_per_gib: float = 0.25
    dram_energy_pj_per_bit: float = 6.0

    ssd_read_active_power_w_per_device: float = 18.0
    ssd_write_active_power_w_per_device: float = 19.0
    ssd_idle_power_w_per_device: float = 5.0

    gpu_fabric_active_power_w_per_host: float = 600.0
    gpu_fabric_idle_power_w_per_host: float = 60.0
    hbf_fabric_active_power_w_per_host: float = 350.0
    hbf_fabric_idle_power_w_per_host: float = 35.0
    pcie_energy_pj_per_bit: float = 4.0

    lpddr_idle_power_w_per_gib: float = 0.08
    lpddr_energy_pj_per_bit: float = 6.0

    baseline_network_nic_count: int = 2
    proposed_network_nic_count: int = 4
    network_nic_idle_power_w: float = 30.0
    network_nic_energy_pj_per_bit: float = 4.0
    network_fabric_count: int = 1
    network_fabric_idle_power_w: float = 100.0
    network_fabric_energy_pj_per_bit: float = 4.0

    sources: tuple[RuntimePowerSource, ...] = field(
        default_factory=lambda: DEFAULT_RUNTIME_POWER_SOURCES)

    def __post_init__(self) -> None:
        if not math.isclose(
            _finite(
                "lifetime_years",
                self.lifetime_years,
                strictly_positive=True,
            ),
            5.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SSDHBFRuntimeEnergyError(
                "runtime TCO is fixed to a five-year projection")
        if _finite("pue", self.pue, strictly_positive=True) < 1.0:
            raise SSDHBFRuntimeEnergyError("pue must be at least 1")
        _finite(
            "electricity_usd_per_kwh",
            self.electricity_usd_per_kwh,
            minimum=0.0,
        )
        for name in (
            "h100_active_power_w_per_card",
            "h100_idle_power_w_per_card",
            "hbf_gpu_logic_active_power_w_per_card",
            "hbf_gpu_logic_idle_power_w_per_card",
            "hbf_media_controller_active_power_w_per_card",
            "hbf_media_controller_idle_power_w_per_card",
            "cpu_host_active_power_w",
            "cpu_host_idle_power_w",
            "host_dram_idle_power_w_per_gib",
            "dram_energy_pj_per_bit",
            "ssd_read_active_power_w_per_device",
            "ssd_write_active_power_w_per_device",
            "ssd_idle_power_w_per_device",
            "gpu_fabric_active_power_w_per_host",
            "gpu_fabric_idle_power_w_per_host",
            "hbf_fabric_active_power_w_per_host",
            "hbf_fabric_idle_power_w_per_host",
            "pcie_energy_pj_per_bit",
            "lpddr_idle_power_w_per_gib",
            "lpddr_energy_pj_per_bit",
            "network_nic_idle_power_w",
            "network_nic_energy_pj_per_bit",
            "network_fabric_idle_power_w",
            "network_fabric_energy_pj_per_bit",
        ):
            _finite(name, getattr(self, name), minimum=0.0)
        _positive_int(
            "hbf_host_dram_capacity_bytes",
            self.hbf_host_dram_capacity_bytes,
        )
        _positive_int(
            "baseline_network_nic_count",
            self.baseline_network_nic_count,
        )
        _positive_int(
            "proposed_network_nic_count",
            self.proposed_network_nic_count,
        )
        _positive_int(
            "network_fabric_count", self.network_fabric_count)
        for active_name, idle_name in (
            (
                "h100_active_power_w_per_card",
                "h100_idle_power_w_per_card",
            ),
            (
                "hbf_gpu_logic_active_power_w_per_card",
                "hbf_gpu_logic_idle_power_w_per_card",
            ),
            (
                "hbf_media_controller_active_power_w_per_card",
                "hbf_media_controller_idle_power_w_per_card",
            ),
            ("cpu_host_active_power_w", "cpu_host_idle_power_w"),
            (
                "ssd_read_active_power_w_per_device",
                "ssd_idle_power_w_per_device",
            ),
            (
                "ssd_write_active_power_w_per_device",
                "ssd_idle_power_w_per_device",
            ),
            (
                "gpu_fabric_active_power_w_per_host",
                "gpu_fabric_idle_power_w_per_host",
            ),
            (
                "hbf_fabric_active_power_w_per_host",
                "hbf_fabric_idle_power_w_per_host",
            ),
        ):
            if getattr(self, active_name) < getattr(self, idle_name):
                raise SSDHBFRuntimeEnergyError(
                    f"{active_name} cannot be below {idle_name}")
        if not self.sources:
            raise SSDHBFRuntimeEnergyError(
                "at least one power source is required")
        if len({
                source.source_key for source in self.sources
        }) != len(self.sources):
            raise SSDHBFRuntimeEnergyError(
                "power source keys must be unique")

    @property
    def lifetime_calendar_hours(self) -> float:
        return self.lifetime_years * HOURS_PER_YEAR

    @property
    def hbf_full_activity_power_ratio_to_h100(self) -> float:
        return (
            self.hbf_gpu_logic_active_power_w_per_card
            + self.hbf_media_controller_active_power_w_per_card
        ) / self.h100_active_power_w_per_card


@dataclass(frozen=True)
class RuntimeComponentEnergy:
    component_key: str
    accounting_mode: str
    physical_quantity: float
    quantity_unit: str
    driver: str
    idle_energy_j: float
    activity_energy_j: float
    total_energy_j: float
    active_device_ns: int = 0
    device_time_capacity_ns: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    transfer_bytes: int = 0
    active_power_w_per_unit: Optional[float] = None
    idle_power_w_per_unit: float = 0.0
    traffic_energy_pj_per_bit: Optional[float] = None
    assumption: str = ""

    def __post_init__(self) -> None:
        if not self.component_key or not self.quantity_unit or not self.driver:
            raise SSDHBFRuntimeEnergyError(
                "component identity fields must be non-empty")
        if self.accounting_mode not in {
            "time_active_idle",
            "traffic_plus_idle",
            "ssd_queue_active_idle",
        }:
            raise SSDHBFRuntimeEnergyError(
                f"unsupported accounting mode {self.accounting_mode!r}")
        _finite(
            f"{self.component_key}.physical_quantity",
            self.physical_quantity,
            minimum=0.0,
        )
        for name in (
            "idle_energy_j",
            "activity_energy_j",
            "total_energy_j",
            "idle_power_w_per_unit",
        ):
            _finite(
                f"{self.component_key}.{name}",
                getattr(self, name),
                minimum=0.0,
            )
        for name in (
            "active_device_ns",
            "device_time_capacity_ns",
            "read_bytes",
            "write_bytes",
            "transfer_bytes",
        ):
            _nonnegative_int(
                f"{self.component_key}.{name}",
                getattr(self, name),
            )
        if self.active_device_ns > self.device_time_capacity_ns:
            raise SSDHBFRuntimeEnergyError(
                f"{self.component_key} active time exceeds physical time")
        if not math.isclose(
            self.total_energy_j,
            self.idle_energy_j + self.activity_energy_j,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise SSDHBFRuntimeEnergyError(
                f"{self.component_key} energy does not conserve")

    @property
    def active_fraction(self) -> float:
        if self.device_time_capacity_ns == 0:
            return 0.0
        return self.active_device_ns / self.device_time_capacity_ns


@dataclass(frozen=True)
class RuntimeEnergyReport:
    report_schema: str
    system_key: str
    horizon_ns: int
    components: tuple[RuntimeComponentEnergy, ...]
    total_it_energy_j: float
    average_it_power_w: float
    input_summary: Mapping[str, int | float | str]
    assumptions: RuntimePowerAssumptions
    runtime_semantics: str = (
        "The exact simulated horizon includes arrivals, tool gaps, and "
        "idle time. Canonical component-exclusive counters prevent "
        "lane/root/endpoint byte duplication."
    )

    def __post_init__(self) -> None:
        if self.report_schema != RUNTIME_ENERGY_SCHEMA:
            raise SSDHBFRuntimeEnergyError(
                "unexpected runtime energy schema")
        if self.system_key not in {
            BASELINE_SYSTEM_KEY,
            PROPOSED_SYSTEM_KEY,
        }:
            raise SSDHBFRuntimeEnergyError(
                "unexpected runtime energy system key")
        _positive_int("horizon_ns", self.horizon_ns)
        if not self.components:
            raise SSDHBFRuntimeEnergyError(
                "runtime energy report requires components")
        keys = [row.component_key for row in self.components]
        if len(keys) != len(set(keys)):
            raise SSDHBFRuntimeEnergyError(
                "runtime component keys must be unique")
        expected = math.fsum(row.total_energy_j for row in self.components)
        if not math.isclose(
            self.total_it_energy_j,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise SSDHBFRuntimeEnergyError(
                "runtime total energy does not match components")
        expected_power = (
            self.total_it_energy_j
            / (self.horizon_ns / NS_PER_SECOND)
        )
        if not math.isclose(
            self.average_it_power_w,
            expected_power,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise SSDHBFRuntimeEnergyError(
                "runtime average power does not match energy/horizon")

    def component(self, component_key: str) -> RuntimeComponentEnergy:
        for row in self.components:
            if row.component_key == component_key:
                return row
        raise KeyError(component_key)

    def to_json_dict(self) -> dict[str, Any]:
        return _strict_json_dict(self)


@dataclass(frozen=True)
class RuntimeTCOProjection:
    report_schema: str
    system_key: str
    capex_usd: float
    trace_average_it_power_w: float
    five_year_it_energy_kwh: float
    five_year_facility_energy_kwh: float
    five_year_runtime_electricity_opex_usd: float
    five_year_tco_usd: float
    replaced_static_electricity_opex_usd: float
    pue: float
    electricity_usd_per_kwh: float
    projection_semantics: str = (
        "Trace-average IT power is projected over five calendar years; "
        "PUE is applied once, and runtime electricity replaces static "
        "BOM electricity."
    )

    def __post_init__(self) -> None:
        if self.report_schema != RUNTIME_TCO_SCHEMA:
            raise SSDHBFRuntimeEnergyError(
                "unexpected runtime TCO schema")
        if self.system_key not in {
            BASELINE_SYSTEM_KEY,
            PROPOSED_SYSTEM_KEY,
        }:
            raise SSDHBFRuntimeEnergyError(
                "unexpected runtime TCO system key")
        for name in (
            "capex_usd",
            "trace_average_it_power_w",
            "five_year_it_energy_kwh",
            "five_year_facility_energy_kwh",
            "five_year_runtime_electricity_opex_usd",
            "five_year_tco_usd",
            "replaced_static_electricity_opex_usd",
            "electricity_usd_per_kwh",
        ):
            _finite(name, getattr(self, name), minimum=0.0)
        if _finite("pue", self.pue, strictly_positive=True) < 1.0:
            raise SSDHBFRuntimeEnergyError("pue must be at least 1")
        if not math.isclose(
            self.five_year_tco_usd,
            (
                self.capex_usd
                + self.five_year_runtime_electricity_opex_usd
            ),
            rel_tol=1e-12,
            abs_tol=1e-8,
        ):
            raise SSDHBFRuntimeEnergyError(
                "runtime TCO must equal CAPEX plus runtime electricity")

    def to_json_dict(self) -> dict[str, Any]:
        return _strict_json_dict(self)


@dataclass(frozen=True)
class RuntimeTCOComparison:
    report_schema: str
    baseline: RuntimeTCOProjection
    proposed: RuntimeTCOProjection
    baseline_runtime: RuntimeEnergyReport
    proposed_runtime: RuntimeEnergyReport
    proposed_average_it_power_ratio_to_baseline: float
    proposed_five_year_it_energy_ratio_to_baseline: float
    proposed_five_year_tco_ratio_to_baseline: float
    incremental_average_it_power_w: float
    incremental_five_year_it_energy_kwh: float
    incremental_five_year_tco_usd: float
    electricity_accounting: str = (
        "replacement: CAPEX + event-derived runtime electricity; static "
        "electricity is disclosed but not added"
    )

    def __post_init__(self) -> None:
        if self.report_schema != RUNTIME_TCO_SCHEMA:
            raise SSDHBFRuntimeEnergyError(
                "unexpected runtime comparison schema")
        if (
            self.baseline.system_key != BASELINE_SYSTEM_KEY
            or self.proposed.system_key != PROPOSED_SYSTEM_KEY
            or self.baseline_runtime.system_key != BASELINE_SYSTEM_KEY
            or self.proposed_runtime.system_key != PROPOSED_SYSTEM_KEY
        ):
            raise SSDHBFRuntimeEnergyError(
                "runtime comparison systems are not baseline/proposed")
        for name in (
            "proposed_average_it_power_ratio_to_baseline",
            "proposed_five_year_it_energy_ratio_to_baseline",
            "proposed_five_year_tco_ratio_to_baseline",
            "incremental_average_it_power_w",
            "incremental_five_year_it_energy_kwh",
            "incremental_five_year_tco_usd",
        ):
            _finite(name, getattr(self, name))
        expected_deltas = (
            (
                self.incremental_average_it_power_w,
                self.proposed.trace_average_it_power_w
                - self.baseline.trace_average_it_power_w,
            ),
            (
                self.incremental_five_year_it_energy_kwh,
                self.proposed.five_year_it_energy_kwh
                - self.baseline.five_year_it_energy_kwh,
            ),
            (
                self.incremental_five_year_tco_usd,
                self.proposed.five_year_tco_usd
                - self.baseline.five_year_tco_usd,
            ),
        )
        if any(
            not math.isclose(
                observed,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-8,
            )
            for observed, expected in expected_deltas
        ):
            raise SSDHBFRuntimeEnergyError(
                "runtime comparison deltas do not conserve")

    def to_json_dict(self) -> dict[str, Any]:
        return _strict_json_dict(self)


def _calendar_resources(
        calendar: Mapping[str, Any], path: str,
) -> Mapping[str, Any]:
    return _child(calendar, "resources", path)


def _resource_total(
        calendars: Sequence[Mapping[str, Any]],
        pattern: re.Pattern[str],
        field_name: str,
) -> int:
    total = 0
    for calendar_index, calendar in enumerate(calendars):
        resources = _calendar_resources(
            calendar, f"calendars[{calendar_index}]")
        for resource_name, raw in resources.items():
            if not isinstance(resource_name, str):
                raise SSDHBFRuntimeEnergyError(
                    "calendar resource names must be strings")
            if pattern.fullmatch(resource_name):
                row = _mapping(
                    raw,
                    f"calendars[{calendar_index}].resources."
                    f"{resource_name}",
                )
                total += _int_field(
                    row,
                    field_name,
                    (
                        f"calendars[{calendar_index}].resources."
                        f"{resource_name}"
                    ),
                )
    return total


def _resource_max(
        calendars: Sequence[Mapping[str, Any]],
        pattern: re.Pattern[str],
        field_name: str,
) -> int:
    maximum = 0
    for calendar_index, calendar in enumerate(calendars):
        resources = _calendar_resources(
            calendar, f"calendars[{calendar_index}]")
        for resource_name, raw in resources.items():
            if not isinstance(resource_name, str):
                raise SSDHBFRuntimeEnergyError(
                    "calendar resource names must be strings")
            if pattern.fullmatch(resource_name):
                row = _mapping(
                    raw,
                    f"calendars[{calendar_index}].resources."
                    f"{resource_name}",
                )
                maximum = max(
                    maximum,
                    _int_field(
                        row,
                        field_name,
                        (
                            f"calendars[{calendar_index}].resources."
                            f"{resource_name}"
                        ),
                    ),
                )
    return maximum


def _exact_resource_total(
        calendars: Sequence[Mapping[str, Any]],
        resource_name: str,
        field_name: str,
) -> int:
    total = 0
    for calendar_index, calendar in enumerate(calendars):
        resources = _calendar_resources(
            calendar, f"calendars[{calendar_index}]")
        if resource_name not in resources:
            continue
        row = _mapping(
            resources[resource_name],
            f"calendars[{calendar_index}].resources.{resource_name}",
        )
        total += _int_field(
            row,
            field_name,
            f"calendars[{calendar_index}].resources.{resource_name}",
        )
    return total


def _node_calendar_reports(
        report: Mapping[str, Any],
        nodes: Sequence[Mapping[str, Any]],
        supplied: Optional[Sequence[Mapping[str, Any]]],
) -> tuple[Mapping[str, Any], ...]:
    if supplied is not None:
        calendars = tuple(supplied)
        if len(calendars) != len(nodes):
            raise SSDHBFRuntimeEnergyError(
                "baseline calendar count must match physical node count")
        for index, calendar in enumerate(calendars):
            _calendar_resources(calendar, f"baseline_calendars[{index}]")
        return calendars
    serialized = report.get("resource_calendars")
    if serialized is not None:
        if (
            isinstance(serialized, (str, bytes))
            or not isinstance(serialized, Sequence)
            or len(serialized) != len(nodes)
        ):
            raise SSDHBFRuntimeEnergyError(
                "system_report.resource_calendars must match the "
                "physical node count")
        calendars = tuple(
            _mapping(
                calendar,
                f"system_report.resource_calendars[{index}]",
            )
            for index, calendar in enumerate(serialized)
        )
        for index, calendar in enumerate(calendars):
            _calendar_resources(
                calendar,
                f"system_report.resource_calendars[{index}]",
            )
        return calendars
    discovered = []
    for index, node in enumerate(nodes):
        lifecycle = _child(node, "lifecycle", f"report.nodes[{index}]")
        raw = lifecycle.get("resource_calendar")
        if raw is None:
            raise SSDHBFRuntimeEnergyError(
                "baseline report lacks resource calendar bytes; pass "
                "each live node.calendar.report() through "
                "baseline_calendar_reports"
            )
        discovered.append(_mapping(
            raw,
            f"report.nodes[{index}].lifecycle.resource_calendar",
        ))
    return tuple(discovered)


def _time_component(
        *,
        component_key: str,
        quantity: int,
        quantity_unit: str,
        horizon_ns: int,
        active_device_ns: int,
        active_power_w: float,
        idle_power_w: float,
        driver: str,
        read_bytes: int = 0,
        write_bytes: int = 0,
        transfer_bytes: int = 0,
        assumption: str,
) -> RuntimeComponentEnergy:
    capacity_ns = quantity * horizon_ns
    if active_device_ns > capacity_ns:
        raise SSDHBFRuntimeEnergyError(
            f"{component_key} active time exceeds {quantity} devices")
    horizon_s = horizon_ns / NS_PER_SECOND
    idle_j = idle_power_w * quantity * horizon_s
    activity_j = (
        (active_power_w - idle_power_w)
        * active_device_ns
        / NS_PER_SECOND
    )
    return RuntimeComponentEnergy(
        component_key=component_key,
        accounting_mode="time_active_idle",
        physical_quantity=float(quantity),
        quantity_unit=quantity_unit,
        driver=driver,
        idle_energy_j=idle_j,
        activity_energy_j=activity_j,
        total_energy_j=idle_j + activity_j,
        active_device_ns=active_device_ns,
        device_time_capacity_ns=capacity_ns,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        transfer_bytes=transfer_bytes,
        active_power_w_per_unit=active_power_w,
        idle_power_w_per_unit=idle_power_w,
        assumption=assumption,
    )


def _traffic_component(
        *,
        component_key: str,
        quantity: float,
        quantity_unit: str,
        horizon_ns: int,
        transfer_bytes: int,
        idle_power_w_per_unit: float,
        energy_pj_per_bit: float,
        driver: str,
        read_bytes: int = 0,
        write_bytes: int = 0,
        assumption: str,
) -> RuntimeComponentEnergy:
    horizon_s = horizon_ns / NS_PER_SECOND
    idle_j = idle_power_w_per_unit * quantity * horizon_s
    activity_j = transfer_bytes * 8 * energy_pj_per_bit * 1e-12
    return RuntimeComponentEnergy(
        component_key=component_key,
        accounting_mode="traffic_plus_idle",
        physical_quantity=quantity,
        quantity_unit=quantity_unit,
        driver=driver,
        idle_energy_j=idle_j,
        activity_energy_j=activity_j,
        total_energy_j=idle_j + activity_j,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        transfer_bytes=transfer_bytes,
        idle_power_w_per_unit=idle_power_w_per_unit,
        traffic_energy_pj_per_bit=energy_pj_per_bit,
        assumption=assumption,
    )


def _ssd_component(
        *,
        quantity: int,
        horizon_ns: int,
        read_active_device_ns: int,
        write_active_device_ns: int,
        read_bytes: int,
        write_bytes: int,
        assumptions: RuntimePowerAssumptions,
) -> RuntimeComponentEnergy:
    if quantity <= 0:
        raise SSDHBFRuntimeEnergyError(
            "SSD component requires at least one device")
    read_device_ns = _nonnegative_int(
        "read_active_device_ns", read_active_device_ns)
    write_device_ns = _nonnegative_int(
        "write_active_device_ns", write_active_device_ns)
    active_device_ns = read_device_ns + write_device_ns
    capacity_ns = quantity * horizon_ns
    if active_device_ns > capacity_ns:
        raise SSDHBFRuntimeEnergyError(
            "SSD active time exceeds the physical device-horizon")
    horizon_s = horizon_ns / NS_PER_SECOND
    idle_j = (
        assumptions.ssd_idle_power_w_per_device
        * quantity
        * horizon_s
    )
    activity_j = (
        (
            assumptions.ssd_read_active_power_w_per_device
            - assumptions.ssd_idle_power_w_per_device
        )
        * read_device_ns
        / NS_PER_SECOND
        + (
            assumptions.ssd_write_active_power_w_per_device
            - assumptions.ssd_idle_power_w_per_device
        )
        * write_device_ns
        / NS_PER_SECOND
    )
    return RuntimeComponentEnergy(
        component_key="local_nvme_ssd",
        accounting_mode="ssd_queue_active_idle",
        physical_quantity=float(quantity),
        quantity_unit="device",
        driver=(
            "canonical aggregate gpu-node-*-ssd-read/write queue busy "
            "time; all devices are assumed striped, with overlapping "
            "read/write queue time proportionally capped"
        ),
        idle_energy_j=idle_j,
        activity_energy_j=activity_j,
        total_energy_j=idle_j + activity_j,
        active_device_ns=active_device_ns,
        device_time_capacity_ns=capacity_ns,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        transfer_bytes=read_bytes + write_bytes,
        active_power_w_per_unit=max(
            assumptions.ssd_read_active_power_w_per_device,
            assumptions.ssd_write_active_power_w_per_device,
        ),
        idle_power_w_per_unit=(
            assumptions.ssd_idle_power_w_per_device),
        assumption=(
            "Micron 9550 PRO 3.84 TB maximum read/write power is "
            "18/19 W and average idle power is 5 W."
        ),
    )


def _ssd_queue_device_times(
        *,
        quantity: int,
        horizon_ns: int,
        read_queue_busy_ns: int,
        write_queue_busy_ns: int,
) -> tuple[int, int]:
    """Expand one aggregate striped queue into capped device-active time."""

    total_queue_ns = read_queue_busy_ns + write_queue_busy_ns
    scale = (
        1.0
        if total_queue_ns <= horizon_ns or total_queue_ns == 0
        else horizon_ns / total_queue_ns
    )
    read_device_ns = int(round(
        quantity * read_queue_busy_ns * scale))
    write_device_ns = int(round(
        quantity * write_queue_busy_ns * scale))
    capacity_ns = quantity * horizon_ns
    overflow = read_device_ns + write_device_ns - capacity_ns
    if overflow > 0:
        write_device_ns = max(0, write_device_ns - overflow)
    return read_device_ns, write_device_ns


def _finish_runtime_report(
        *,
        system_key: str,
        horizon_ns: int,
        components: Sequence[RuntimeComponentEnergy],
        input_summary: Mapping[str, int | float | str],
        assumptions: RuntimePowerAssumptions,
) -> RuntimeEnergyReport:
    rows = tuple(components)
    total_j = math.fsum(row.total_energy_j for row in rows)
    average_w = total_j / (horizon_ns / NS_PER_SECOND)
    return RuntimeEnergyReport(
        report_schema=RUNTIME_ENERGY_SCHEMA,
        system_key=system_key,
        horizon_ns=horizon_ns,
        components=rows,
        total_it_energy_j=total_j,
        average_it_power_w=average_w,
        input_summary=dict(input_summary),
        assumptions=assumptions,
    )


def account_two_gpu_runtime_energy(
        system_report: Mapping[str, Any],
        *,
        baseline_calendar_reports: Optional[
            Sequence[Mapping[str, Any]]
        ] = None,
        assumptions: RuntimePowerAssumptions = RuntimePowerAssumptions(),
) -> RuntimeEnergyReport:
    """Account one two-host finite-HBM baseline execution.

    Current reports expose ``resource_calendars`` directly.  The optional
    argument retains compatibility with older reports and focused callers
    that hold the two live ``node.calendar.report()`` values separately.
    """

    report = _mapping(system_report, "system_report")
    if report.get("mode") != "dual_finite_hbm_p4d4_tiering":
        raise SSDHBFRuntimeEnergyError(
            "baseline report mode must be dual_finite_hbm_p4d4_tiering")
    horizon_ns = _int_field(
        report, "current_ns", "system_report", positive=True)
    raw_nodes = report.get("nodes")
    if (
        isinstance(raw_nodes, (str, bytes))
        or not isinstance(raw_nodes, Sequence)
        or len(raw_nodes) != 2
    ):
        raise SSDHBFRuntimeEnergyError(
            "baseline requires exactly two physical node reports")
    nodes = tuple(
        _mapping(node, f"system_report.nodes[{index}]")
        for index, node in enumerate(raw_nodes)
    )
    calendars = _node_calendar_reports(
        report, nodes, baseline_calendar_reports)

    h100_count = 0
    h100_active_card_ns = 0
    host_active_ns = 0
    gpu_fabric_active_host_ns = 0
    host_dram_bytes = 0
    host_dram_capacity_bytes = 0
    ssd_count = 0
    ssd_read_active_device_ns = 0
    ssd_write_active_device_ns = 0
    ssd_read_bytes = 0
    ssd_write_bytes = 0
    pcie_root_bytes = 0
    pd_fabric_bytes = 0

    for index, (node, calendar) in enumerate(zip(nodes, calendars)):
        pool = _child(node, "pool", f"system_report.nodes[{index}]")
        hardware = _child(
            pool, "hardware", f"system_report.nodes[{index}].pool")
        metrics = _child(
            pool, "metrics", f"system_report.nodes[{index}].pool")
        gpu_count = _int_field(
            hardware,
            "gpu_count",
            f"system_report.nodes[{index}].pool.hardware",
            positive=True,
        )
        p_gpu_count = _int_field(
            hardware,
            "prefill_gpu_count",
            f"system_report.nodes[{index}].pool.hardware",
            positive=True,
        )
        d_gpu_count = _int_field(
            hardware,
            "decode_gpu_count",
            f"system_report.nodes[{index}].pool.hardware",
            positive=True,
        )
        if p_gpu_count + d_gpu_count != gpu_count:
            raise SSDHBFRuntimeEnergyError(
                "baseline P/D GPU groups must partition each host")
        p_ns = _int_field(
            metrics,
            "p_modeled_ns",
            f"system_report.nodes[{index}].pool.metrics",
        )
        d_ns = _int_field(
            metrics,
            "d_modeled_ns",
            f"system_report.nodes[{index}].pool.metrics",
        )
        if p_ns > horizon_ns or d_ns > horizon_ns:
            raise SSDHBFRuntimeEnergyError(
                "baseline P/D modeled time exceeds the trace horizon")
        h100_count += gpu_count
        h100_active_card_ns += p_gpu_count * p_ns + d_gpu_count * d_ns
        host_dram_capacity_bytes += _int_field(
            hardware,
            "cpu_memory_capacity_bytes",
            f"system_report.nodes[{index}].pool.hardware",
            positive=True,
        )
        node_ssd_count = _int_field(
            hardware,
            "ssd_device_count",
            f"system_report.nodes[{index}].pool.hardware",
            positive=True,
        )
        ssd_count += node_ssd_count
        one_calendar = (calendar,)
        cpu_busy = _resource_total(
            one_calendar, _CPU_DRAM_RE, "busy_ns")
        host_active_ns += min(horizon_ns, cpu_busy)
        host_dram_bytes += _resource_total(
            one_calendar, _CPU_DRAM_RE, "reservation_bytes")
        read_busy = _resource_total(
            one_calendar, _SSD_READ_RE, "busy_ns")
        write_busy = _resource_total(
            one_calendar, _SSD_WRITE_RE, "busy_ns")
        read_device_ns, write_device_ns = _ssd_queue_device_times(
            quantity=node_ssd_count,
            horizon_ns=horizon_ns,
            read_queue_busy_ns=read_busy,
            write_queue_busy_ns=write_busy,
        )
        ssd_read_active_device_ns += read_device_ns
        ssd_write_active_device_ns += write_device_ns
        ssd_read_bytes += _resource_total(
            one_calendar, _SSD_READ_RE, "reservation_bytes")
        ssd_write_bytes += _resource_total(
            one_calendar, _SSD_WRITE_RE, "reservation_bytes")
        pcie_root_bytes += _resource_total(
            one_calendar, _GPU_ROOT_RE, "reservation_bytes")
        pd_busy = _resource_total(
            one_calendar, _PD_FABRIC_RE, "busy_ns")
        pd_fabric_bytes += _resource_total(
            one_calendar, _PD_FABRIC_RE, "reservation_bytes")
        gpu_fabric_active_host_ns += min(
            horizon_ns, p_ns + d_ns + pd_busy)

    if h100_count != 16 or ssd_count != 16:
        raise SSDHBFRuntimeEnergyError(
            "baseline physical topology must contain 16 H100s and 16 SSDs")
    rdma_bytes = _exact_resource_total(
        calendars, "rdma-network", "reservation_bytes")
    if rdma_bytes:
        raise SSDHBFRuntimeEnergyError(
            "independent baseline hosts unexpectedly used RDMA")

    components = (
        _time_component(
            component_key="h100_gpu_hbm_cards",
            quantity=h100_count,
            quantity_unit="whole H100 card including HBM",
            horizon_ns=horizon_ns,
            active_device_ns=h100_active_card_ns,
            active_power_w=assumptions.h100_active_power_w_per_card,
            idle_power_w=assumptions.h100_idle_power_w_per_card,
            driver=(
                "prefill_gpu_count*p_modeled_ns + "
                "decode_gpu_count*d_modeled_ns"
            ),
            assumption=(
                "Whole-card H100 power includes GPU logic and HBM, so no "
                "separate HBM runtime charge is added."
            ),
        ),
        _time_component(
            component_key="cpu_host_platform",
            quantity=2,
            quantity_unit="CPU/chassis host excluding DRAM and devices",
            horizon_ns=horizon_ns,
            active_device_ns=host_active_ns,
            active_power_w=assumptions.cpu_host_active_power_w,
            idle_power_w=assumptions.cpu_host_idle_power_w,
            driver=(
                "per-host canonical CPU-DRAM busy time, capped at one "
                "physical host-horizon"
            ),
            assumption=(
                "CPU/chassis activity is driven only by host-memory I/O; "
                "accelerator and device power are separate components."
            ),
        ),
        _traffic_component(
            component_key="host_dram",
            quantity=host_dram_capacity_bytes / BYTES_PER_GIB,
            quantity_unit="GiB",
            horizon_ns=horizon_ns,
            transfer_bytes=host_dram_bytes,
            idle_power_w_per_unit=(
                assumptions.host_dram_idle_power_w_per_gib),
            energy_pj_per_bit=assumptions.dram_energy_pj_per_bit,
            driver=(
                "canonical gpu-node-*-cpu-dram reservation_bytes only"
            ),
            assumption=(
                "The CPU-DRAM endpoint is charged once; matching SSD and "
                "PCIe counters are assigned to their own components."
            ),
        ),
        _ssd_component(
            quantity=ssd_count,
            horizon_ns=horizon_ns,
            read_active_device_ns=ssd_read_active_device_ns,
            write_active_device_ns=ssd_write_active_device_ns,
            read_bytes=ssd_read_bytes,
            write_bytes=ssd_write_bytes,
            assumptions=assumptions,
        ),
        _time_component(
            component_key="gpu_intraserver_fabric",
            quantity=2,
            quantity_unit="host fabric allocation",
            horizon_ns=horizon_ns,
            active_device_ns=gpu_fabric_active_host_ns,
            active_power_w=assumptions.gpu_fabric_active_power_w_per_host,
            idle_power_w=assumptions.gpu_fabric_idle_power_w_per_host,
            driver=(
                "per-host capped union proxy p_modeled_ns + d_modeled_ns "
                "+ aggregate gpu-node-*-pd-fabric busy_ns"
            ),
            transfer_bytes=pd_fabric_bytes,
            assumption=(
                "One aggregate P/D-fabric byte counter is retained; "
                "per-rank peer lanes are excluded."
            ),
        ),
        _traffic_component(
            component_key="pcie_root_data_path",
            quantity=4.0,
            quantity_unit="GPU-host PCIe root",
            horizon_ns=horizon_ns,
            transfer_bytes=pcie_root_bytes,
            idle_power_w_per_unit=0.0,
            energy_pj_per_bit=assumptions.pcie_energy_pj_per_bit,
            driver="gpu-node-*-pcie-root-* reservation_bytes only",
            assumption=(
                "Rank PCIe lanes and CPU-DRAM endpoint bytes are excluded "
                "from this root-path counter."
            ),
        ),
        _traffic_component(
            component_key="external_network_nics",
            quantity=float(assumptions.baseline_network_nic_count),
            quantity_unit="NIC",
            horizon_ns=horizon_ns,
            transfer_bytes=0,
            idle_power_w_per_unit=assumptions.network_nic_idle_power_w,
            energy_pj_per_bit=(
                assumptions.network_nic_energy_pj_per_bit),
            driver=(
                "two NIC endpoints per logical rdma-network byte; zero "
                "runtime RDMA bytes in the independent baseline"
            ),
            assumption=(
                "NIC idle power is included even though the paired "
                "baseline performs no inter-host KV transfer."
            ),
        ),
        _traffic_component(
            component_key="external_network_fabric",
            quantity=float(assumptions.network_fabric_count),
            quantity_unit="system fabric allocation",
            horizon_ns=horizon_ns,
            transfer_bytes=0,
            idle_power_w_per_unit=(
                assumptions.network_fabric_idle_power_w),
            energy_pj_per_bit=(
                assumptions.network_fabric_energy_pj_per_bit),
            driver="one logical rdma-network byte counter",
            assumption=(
                "The priced fabric allocation remains powered in both "
                "finite systems."
            ),
        ),
    )
    return _finish_runtime_report(
        system_key=BASELINE_SYSTEM_KEY,
        horizon_ns=horizon_ns,
        components=components,
        input_summary={
            "gpu_host_count": 2,
            "h100_card_count": h100_count,
            "ssd_device_count": ssd_count,
            "h100_active_card_ns": h100_active_card_ns,
            "cpu_dram_transfer_bytes": host_dram_bytes,
            "ssd_read_bytes": ssd_read_bytes,
            "ssd_write_bytes": ssd_write_bytes,
            "pcie_root_transfer_bytes": pcie_root_bytes,
            "pd_fabric_transfer_bytes": pd_fabric_bytes,
            "rdma_logical_bytes": 0,
        },
        assumptions=assumptions,
    )


def account_one_gpu_one_hbf_runtime_energy(
        system_report: Mapping[str, Any],
        *,
        assumptions: RuntimePowerAssumptions = RuntimePowerAssumptions(),
) -> RuntimeEnergyReport:
    """Account one staged GPU+SSD host plus one eight-card HBF host."""

    report = _mapping(system_report, "system_report")
    if report.get("mode") != "ssd_staged_gpu_hbf_agentic_system":
        raise SSDHBFRuntimeEnergyError(
            "proposed report mode must be "
            "ssd_staged_gpu_hbf_agentic_system")
    horizon_ns = _int_field(
        report, "current_ns", "system_report", positive=True)
    node = _child(report, "node", "system_report")
    calendar = _child(node, "calendar", "system_report.node")
    calendars = (calendar,)
    gpu_node = _child(node, "gpu_node", "system_report.node")
    gpu_pool = _child(
        gpu_node, "pool", "system_report.node.gpu_node")
    gpu_hardware = _child(
        gpu_pool, "hardware", "system_report.node.gpu_node.pool")
    gpu_metrics = _child(
        gpu_pool, "metrics", "system_report.node.gpu_node.pool")
    hbf_pool = _child(node, "hbf_pool", "system_report.node")
    hbf_hardware = _child(
        hbf_pool, "hardware", "system_report.node.hbf_pool")
    hbf_layout = _child(
        hbf_pool, "layout", "system_report.node.hbf_pool")
    hbf_metrics = _child(
        hbf_pool, "metrics", "system_report.node.hbf_pool")
    lifecycle = _child(
        node, "hbf_lifecycle", "system_report.node")
    write_accounting = _child(
        lifecycle,
        "hbf_write_accounting",
        "system_report.node.hbf_lifecycle",
    )

    h100_count = _int_field(
        gpu_hardware,
        "gpu_count",
        "system_report.node.gpu_node.pool.hardware",
        positive=True,
    )
    p_gpu_count = _int_field(
        gpu_hardware,
        "prefill_gpu_count",
        "system_report.node.gpu_node.pool.hardware",
        positive=True,
    )
    d_gpu_count = _int_field(
        gpu_hardware,
        "decode_gpu_count",
        "system_report.node.gpu_node.pool.hardware",
        positive=True,
    )
    if p_gpu_count + d_gpu_count != h100_count or h100_count != 8:
        raise SSDHBFRuntimeEnergyError(
            "proposed GPU host must contain one eight-H100 P/D partition")
    p_ns = _int_field(
        gpu_metrics,
        "p_modeled_ns",
        "system_report.node.gpu_node.pool.metrics",
    )
    d_ns = _int_field(
        gpu_metrics,
        "d_modeled_ns",
        "system_report.node.gpu_node.pool.metrics",
    )
    if p_ns > horizon_ns or d_ns > horizon_ns:
        raise SSDHBFRuntimeEnergyError(
            "proposed P/D modeled time exceeds trace horizon")
    h100_active_card_ns = p_gpu_count * p_ns + d_gpu_count * d_ns

    hbf_card_count = _int_field(
        hbf_hardware,
        "card_count",
        "system_report.node.hbf_pool.hardware",
        positive=True,
    )
    if hbf_card_count != 8:
        raise SSDHBFRuntimeEnergyError(
            "proposed HBF host must contain eight physical cards")
    hbf_tp = _int_field(
        hbf_layout,
        "tp_size",
        "system_report.node.hbf_pool.layout",
        positive=True,
    )
    hbf_modeled_ns = _int_field(
        hbf_metrics,
        "modeled_batch_ns",
        "system_report.node.hbf_pool.metrics",
    )
    hbf_logic_active_card_ns = hbf_tp * hbf_modeled_ns
    if hbf_logic_active_card_ns > hbf_card_count * horizon_ns:
        raise SSDHBFRuntimeEnergyError(
            "HBF GPU-logic card time exceeds physical card-horizon")

    media_busy_ns = _resource_total(
        calendars, _HBF_MEDIA_RE, "busy_ns")
    media_bytes = _resource_total(
        calendars, _HBF_MEDIA_RE, "reservation_bytes")
    if media_busy_ns > hbf_card_count * horizon_ns:
        raise SSDHBFRuntimeEnergyError(
            "HBF media busy time exceeds physical card-horizon")
    hbf_write_bytes = _int_field(
        write_accounting,
        "total_physical_write_bytes",
        "system_report.node.hbf_lifecycle.hbf_write_accounting",
    )
    if hbf_write_bytes > media_bytes:
        raise SSDHBFRuntimeEnergyError(
            "HBF write ledger exceeds canonical media bytes")
    hbf_read_bytes = media_bytes - hbf_write_bytes
    expected_hbf_read_bytes = (
        _int_field(
            hbf_metrics,
            "hbf_read_bytes_per_rank",
            "system_report.node.hbf_pool.metrics",
        )
        * hbf_tp
    )
    if hbf_read_bytes != expected_hbf_read_bytes:
        raise SSDHBFRuntimeEnergyError(
            "HBF pool read bytes disagree with canonical per-card media "
            "bytes after subtracting physical writes")

    lpddr_bytes = _resource_total(
        calendars, _HBF_LPDDR_RE, "reservation_bytes")
    lpddr_capacity_bytes_per_card = _int_field(
        hbf_hardware,
        "lpddr_capacity_bytes_per_card",
        "system_report.node.hbf_pool.hardware",
        positive=True,
    )
    lpddr_gib = (
        hbf_card_count
        * lpddr_capacity_bytes_per_card
        / BYTES_PER_GIB
    )

    cpu_dram_busy_ns = _resource_total(
        calendars, _CPU_DRAM_RE, "busy_ns")
    cpu_dram_bytes = _resource_total(
        calendars, _CPU_DRAM_RE, "reservation_bytes")
    gpu_host_dram_bytes = _int_field(
        gpu_hardware,
        "cpu_memory_capacity_bytes",
        "system_report.node.gpu_node.pool.hardware",
        positive=True,
    )
    host_dram_capacity_bytes = (
        gpu_host_dram_bytes
        + assumptions.hbf_host_dram_capacity_bytes
    )

    ssd_count = _int_field(
        gpu_hardware,
        "ssd_device_count",
        "system_report.node.gpu_node.pool.hardware",
        positive=True,
    )
    if ssd_count != 8:
        raise SSDHBFRuntimeEnergyError(
            "proposed GPU host must contain eight SSDs")
    ssd_read_busy_ns = _resource_total(
        calendars, _SSD_READ_RE, "busy_ns")
    ssd_write_busy_ns = _resource_total(
        calendars, _SSD_WRITE_RE, "busy_ns")
    ssd_read_bytes = _resource_total(
        calendars, _SSD_READ_RE, "reservation_bytes")
    ssd_write_bytes = _resource_total(
        calendars, _SSD_WRITE_RE, "reservation_bytes")
    ssd_read_active_device_ns, ssd_write_active_device_ns = (
        _ssd_queue_device_times(
            quantity=ssd_count,
            horizon_ns=horizon_ns,
            read_queue_busy_ns=ssd_read_busy_ns,
            write_queue_busy_ns=ssd_write_busy_ns,
        )
    )

    pd_busy_ns = _resource_total(
        calendars, _PD_FABRIC_RE, "busy_ns")
    pd_fabric_bytes = _resource_total(
        calendars, _PD_FABRIC_RE, "reservation_bytes")
    gpu_fabric_active_ns = min(
        horizon_ns, p_ns + d_ns + pd_busy_ns)
    hbf_fabric_busy_ns = _resource_total(
        calendars, _HBF_FABRIC_RE, "busy_ns")
    hbf_root_busy_ns = _resource_max(
        calendars, _HBF_ROOT_RE, "busy_ns")
    hbf_fabric_active_ns = min(
        horizon_ns, hbf_fabric_busy_ns + hbf_root_busy_ns)
    hbf_collective_bytes = _resource_total(
        calendars, _HBF_FABRIC_RE, "reservation_bytes")

    gpu_root_bytes = _resource_total(
        calendars, _GPU_ROOT_RE, "reservation_bytes")
    hbf_root_bytes = _resource_total(
        calendars, _HBF_ROOT_RE, "reservation_bytes")
    rdma_bytes = _exact_resource_total(
        calendars, "rdma-network", "reservation_bytes")

    components = (
        _time_component(
            component_key="h100_gpu_hbm_cards",
            quantity=h100_count,
            quantity_unit="whole H100 card including HBM",
            horizon_ns=horizon_ns,
            active_device_ns=h100_active_card_ns,
            active_power_w=assumptions.h100_active_power_w_per_card,
            idle_power_w=assumptions.h100_idle_power_w_per_card,
            driver=(
                "prefill_gpu_count*p_modeled_ns + "
                "decode_gpu_count*d_modeled_ns"
            ),
            assumption=(
                "Whole-card H100 power includes GPU logic and HBM."
            ),
        ),
        _time_component(
            component_key="hbf_gpu_logic",
            quantity=hbf_card_count,
            quantity_unit="H100-class GPU logic on HBF card",
            horizon_ns=horizon_ns,
            active_device_ns=hbf_logic_active_card_ns,
            active_power_w=(
                assumptions.hbf_gpu_logic_active_power_w_per_card),
            idle_power_w=(
                assumptions.hbf_gpu_logic_idle_power_w_per_card),
            driver="hbf layout tp_size * hbf_pool.modeled_batch_ns",
            assumption=(
                "HBF compute is H100-class GPU logic with the static "
                "analytical HBM power share removed."
            ),
        ),
        _time_component(
            component_key="hbf_media_controller",
            quantity=hbf_card_count,
            quantity_unit="HBF media/controller card subsystem",
            horizon_ns=horizon_ns,
            active_device_ns=media_busy_ns,
            active_power_w=(
                assumptions
                .hbf_media_controller_active_power_w_per_card),
            idle_power_w=(
                assumptions
                .hbf_media_controller_idle_power_w_per_card),
            driver=(
                "sum of canonical per-card hbf-card-*-media busy_ns"
            ),
            read_bytes=hbf_read_bytes,
            write_bytes=hbf_write_bytes,
            transfer_bytes=media_bytes,
            assumption=(
                "The 300 W active subsystem anchor calibrates HBF whole-"
                "accelerator power to about 1.23x H100. The paper's "
                "8-pJ/bit read energy is not added again."
            ),
        ),
        _traffic_component(
            component_key="hbf_lpddr",
            quantity=lpddr_gib,
            quantity_unit="GiB",
            horizon_ns=horizon_ns,
            transfer_bytes=lpddr_bytes,
            idle_power_w_per_unit=(
                assumptions.lpddr_idle_power_w_per_gib),
            energy_pj_per_bit=assumptions.lpddr_energy_pj_per_bit,
            driver=(
                "sum of canonical per-card hbf-card-*-lpddr "
                "reservation_bytes"
            ),
            assumption=(
                "LPDDR endpoint bytes are charged once and are distinct "
                "from HBF media bytes."
            ),
        ),
        _time_component(
            component_key="cpu_host_platform",
            quantity=2,
            quantity_unit="CPU/chassis host excluding DRAM and devices",
            horizon_ns=horizon_ns,
            active_device_ns=min(horizon_ns, cpu_dram_busy_ns),
            active_power_w=assumptions.cpu_host_active_power_w,
            idle_power_w=assumptions.cpu_host_idle_power_w,
            driver=(
                "GPU-host canonical CPU-DRAM busy time; HBF-host CPU "
                "remains at idle because direct card service has no CPU "
                "execution counter"
            ),
            assumption=(
                "Both physical hosts remain powered for the full trace."
            ),
        ),
        _traffic_component(
            component_key="host_dram",
            quantity=host_dram_capacity_bytes / BYTES_PER_GIB,
            quantity_unit="GiB",
            horizon_ns=horizon_ns,
            transfer_bytes=cpu_dram_bytes,
            idle_power_w_per_unit=(
                assumptions.host_dram_idle_power_w_per_gib),
            energy_pj_per_bit=assumptions.dram_energy_pj_per_bit,
            driver=(
                "canonical gpu-node-*-cpu-dram reservation_bytes only"
            ),
            assumption=(
                "The HBF host receives the same 512e9-byte DRAM "
                "allocation as the GPU host but has no modeled CPU-DRAM "
                "payload unless the event trace exposes one."
            ),
        ),
        _ssd_component(
            quantity=ssd_count,
            horizon_ns=horizon_ns,
            read_active_device_ns=ssd_read_active_device_ns,
            write_active_device_ns=ssd_write_active_device_ns,
            read_bytes=ssd_read_bytes,
            write_bytes=ssd_write_bytes,
            assumptions=assumptions,
        ),
        _time_component(
            component_key="gpu_intraserver_fabric",
            quantity=1,
            quantity_unit="host fabric allocation",
            horizon_ns=horizon_ns,
            active_device_ns=gpu_fabric_active_ns,
            active_power_w=assumptions.gpu_fabric_active_power_w_per_host,
            idle_power_w=assumptions.gpu_fabric_idle_power_w_per_host,
            driver=(
                "capped proxy p_modeled_ns + d_modeled_ns + aggregate "
                "gpu-node-*-pd-fabric busy_ns"
            ),
            transfer_bytes=pd_fabric_bytes,
            assumption=(
                "One aggregate P/D-fabric byte counter is retained; "
                "per-rank lanes are excluded."
            ),
        ),
        _time_component(
            component_key="hbf_intraserver_fabric",
            quantity=1,
            quantity_unit="HBF host fabric allocation",
            horizon_ns=horizon_ns,
            active_device_ns=hbf_fabric_active_ns,
            active_power_w=assumptions.hbf_fabric_active_power_w_per_host,
            idle_power_w=assumptions.hbf_fabric_idle_power_w_per_host,
            driver=(
                "aggregate hbf-group-*-fabric busy_ns plus maximum HBF "
                "root busy_ns, capped at one physical host-fabric horizon"
            ),
            transfer_bytes=hbf_collective_bytes + hbf_root_bytes,
            assumption=(
                "Per-card collective and PCIe-card endpoints are "
                "excluded. HBF root ingress is included in this active "
                "fabric envelope and receives no second pJ/bit charge."
            ),
        ),
        _traffic_component(
            component_key="pcie_root_data_path",
            quantity=2.0,
            quantity_unit="GPU-host PCIe root",
            horizon_ns=horizon_ns,
            transfer_bytes=gpu_root_bytes,
            idle_power_w_per_unit=0.0,
            energy_pj_per_bit=assumptions.pcie_energy_pj_per_bit,
            driver="gpu-node-*-pcie-root-* reservation_bytes",
            assumption=(
                "GPU rank lanes and memory endpoints are excluded. HBF "
                "root paths are already covered by the HBF host-fabric "
                "active-power envelope."
            ),
        ),
        _traffic_component(
            component_key="external_network_nics",
            quantity=float(assumptions.proposed_network_nic_count),
            quantity_unit="NIC",
            horizon_ns=horizon_ns,
            transfer_bytes=rdma_bytes * 2,
            idle_power_w_per_unit=assumptions.network_nic_idle_power_w,
            energy_pj_per_bit=(
                assumptions.network_nic_energy_pj_per_bit),
            driver=(
                "two 50-GB/s NIC cards per endpoint; logical bytes are "
                "striped within an endpoint and charged at both ends"
            ),
            assumption=(
                "Four NIC cards supply the configured 80-GB/s path. "
                "GPU-side and HBF-side resource counters are not summed "
                "separately."
            ),
        ),
        _traffic_component(
            component_key="external_network_fabric",
            quantity=float(assumptions.network_fabric_count),
            quantity_unit="system fabric allocation",
            horizon_ns=horizon_ns,
            transfer_bytes=rdma_bytes,
            idle_power_w_per_unit=(
                assumptions.network_fabric_idle_power_w),
            energy_pj_per_bit=(
                assumptions.network_fabric_energy_pj_per_bit),
            driver="one canonical rdma-network logical byte counter",
            assumption=(
                "The fabric sees each logical transfer once, while its "
                "two NIC endpoints are accounted separately."
            ),
        ),
    )
    return _finish_runtime_report(
        system_key=PROPOSED_SYSTEM_KEY,
        horizon_ns=horizon_ns,
        components=components,
        input_summary={
            "gpu_host_count": 1,
            "hbf_host_count": 1,
            "h100_card_count": h100_count,
            "hbf_card_count": hbf_card_count,
            "hbf_tp_size": hbf_tp,
            "h100_active_card_ns": h100_active_card_ns,
            "hbf_logic_active_card_ns": hbf_logic_active_card_ns,
            "hbf_media_busy_card_ns": media_busy_ns,
            "hbf_media_read_bytes": hbf_read_bytes,
            "hbf_media_write_bytes": hbf_write_bytes,
            "hbf_lpddr_transfer_bytes": lpddr_bytes,
            "cpu_dram_transfer_bytes": cpu_dram_bytes,
            "ssd_read_bytes": ssd_read_bytes,
            "ssd_write_bytes": ssd_write_bytes,
            "gpu_pcie_root_transfer_bytes": gpu_root_bytes,
            "hbf_pcie_root_transfer_bytes": hbf_root_bytes,
            "pd_fabric_transfer_bytes": pd_fabric_bytes,
            "rdma_logical_bytes": rdma_bytes,
        },
        assumptions=assumptions,
    )


def project_five_year_runtime_tco(
        runtime: RuntimeEnergyReport,
        *,
        capex_usd: float,
        replaced_static_electricity_opex_usd: float,
) -> RuntimeTCOProjection:
    """Project trace-average power and replace static electricity OPEX."""

    capex = _finite("capex_usd", capex_usd, minimum=0.0)
    replaced = _finite(
        "replaced_static_electricity_opex_usd",
        replaced_static_electricity_opex_usd,
        minimum=0.0,
    )
    assumptions = runtime.assumptions
    it_kwh = (
        runtime.average_it_power_w
        * assumptions.lifetime_calendar_hours
        / 1000.0
    )
    facility_kwh = it_kwh * assumptions.pue
    electricity = (
        facility_kwh * assumptions.electricity_usd_per_kwh)
    return RuntimeTCOProjection(
        report_schema=RUNTIME_TCO_SCHEMA,
        system_key=runtime.system_key,
        capex_usd=capex,
        trace_average_it_power_w=runtime.average_it_power_w,
        five_year_it_energy_kwh=it_kwh,
        five_year_facility_energy_kwh=facility_kwh,
        five_year_runtime_electricity_opex_usd=electricity,
        five_year_tco_usd=capex + electricity,
        replaced_static_electricity_opex_usd=replaced,
        pue=assumptions.pue,
        electricity_usd_per_kwh=assumptions.electricity_usd_per_kwh,
    )


def evaluate_ssd_hbf_runtime_tco(
        *,
        baseline_system_report: Mapping[str, Any],
        proposed_system_report: Mapping[str, Any],
        baseline_calendar_reports: Optional[
            Sequence[Mapping[str, Any]]
        ] = None,
        baseline_capex_usd: float,
        proposed_capex_usd: float,
        baseline_static_electricity_opex_usd: float,
        proposed_static_electricity_opex_usd: float,
        assumptions: RuntimePowerAssumptions = RuntimePowerAssumptions(),
) -> RuntimeTCOComparison:
    """Evaluate seed-paired finite systems with runtime electricity."""

    baseline_runtime = account_two_gpu_runtime_energy(
        baseline_system_report,
        baseline_calendar_reports=baseline_calendar_reports,
        assumptions=assumptions,
    )
    proposed_runtime = account_one_gpu_one_hbf_runtime_energy(
        proposed_system_report,
        assumptions=assumptions,
    )
    baseline = project_five_year_runtime_tco(
        baseline_runtime,
        capex_usd=baseline_capex_usd,
        replaced_static_electricity_opex_usd=(
            baseline_static_electricity_opex_usd),
    )
    proposed = project_five_year_runtime_tco(
        proposed_runtime,
        capex_usd=proposed_capex_usd,
        replaced_static_electricity_opex_usd=(
            proposed_static_electricity_opex_usd),
    )
    if (
        baseline.trace_average_it_power_w <= 0.0
        or baseline.five_year_it_energy_kwh <= 0.0
        or baseline.five_year_tco_usd <= 0.0
    ):
        raise SSDHBFRuntimeEnergyError(
            "baseline runtime projection must be positive")
    return RuntimeTCOComparison(
        report_schema=RUNTIME_TCO_SCHEMA,
        baseline=baseline,
        proposed=proposed,
        baseline_runtime=baseline_runtime,
        proposed_runtime=proposed_runtime,
        proposed_average_it_power_ratio_to_baseline=(
            proposed.trace_average_it_power_w
            / baseline.trace_average_it_power_w
        ),
        proposed_five_year_it_energy_ratio_to_baseline=(
            proposed.five_year_it_energy_kwh
            / baseline.five_year_it_energy_kwh
        ),
        proposed_five_year_tco_ratio_to_baseline=(
            proposed.five_year_tco_usd
            / baseline.five_year_tco_usd
        ),
        incremental_average_it_power_w=(
            proposed.trace_average_it_power_w
            - baseline.trace_average_it_power_w
        ),
        incremental_five_year_it_energy_kwh=(
            proposed.five_year_it_energy_kwh
            - baseline.five_year_it_energy_kwh
        ),
        incremental_five_year_tco_usd=(
            proposed.five_year_tco_usd
            - baseline.five_year_tco_usd
        ),
    )


__all__ = [
    "BASELINE_SYSTEM_KEY",
    "DEFAULT_RUNTIME_POWER_SOURCES",
    "PROPOSED_SYSTEM_KEY",
    "RUNTIME_ENERGY_SCHEMA",
    "RUNTIME_TCO_SCHEMA",
    "RuntimeComponentEnergy",
    "RuntimeEnergyReport",
    "RuntimePowerAssumptions",
    "RuntimePowerSource",
    "RuntimeTCOComparison",
    "RuntimeTCOProjection",
    "SSDHBFRuntimeEnergyError",
    "account_one_gpu_one_hbf_runtime_energy",
    "account_two_gpu_runtime_energy",
    "evaluate_ssd_hbf_runtime_tco",
    "project_five_year_runtime_tco",
]

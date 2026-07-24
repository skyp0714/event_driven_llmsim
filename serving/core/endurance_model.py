"""SSD write-endurance projection from simulator storage counters.

The simulator-facing contract is intentionally small: callers provide the
host bytes written during one workload epoch, optionally broken down by SSD.
This module combines those counters with a vendor endurance profile and a
replay-rate assumption.  It does not model migration policy or I/O latency.

All capacity and endurance quantities use decimal SI units, matching SSD
datasheets: one GB is 10**9 bytes and one TB is 10**12 bytes.

Vendor TBW and DWPD ratings are host-write ratings.  Consequently WAF is
reported as an estimate of NAND traffic, but is never multiplied into the
host-TBW wear calculation.  Doing both would count controller amplification
twice.  A raw-NAND endurance model would require a separate NAND budget and
is deliberately outside this module's host-TBW API.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DECIMAL_GB = 10 ** 9
DECIMAL_TB = 10 ** 12
SECONDS_PER_DAY = 86_400.0
DEFAULT_DAYS_PER_YEAR = 365.0
REPORT_SCHEMA_VERSION = 1


class EnduranceConfigError(ValueError):
    """Raised when an endurance profile or projection input is inconsistent."""


def _as_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise EnduranceConfigError(f"{field} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EnduranceConfigError(
            f"{field} must be a non-negative integer"
        ) from exc
    try:
        if Decimal(str(value)) != Decimal(result):
            raise EnduranceConfigError(
                f"{field} must be a non-negative integer"
            )
    except (InvalidOperation, ValueError) as exc:
        raise EnduranceConfigError(
            f"{field} must be a non-negative integer"
        ) from exc
    if result < 0:
        raise EnduranceConfigError(f"{field} must be non-negative")
    return result


def _as_positive_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EnduranceConfigError(f"{field} must be positive") from exc
    if result <= 0 or result == float("inf") or result != result:
        raise EnduranceConfigError(f"{field} must be finite and positive")
    return result


def _as_nonnegative_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EnduranceConfigError(f"{field} must be non-negative") from exc
    if result < 0 or result == float("inf") or result != result:
        raise EnduranceConfigError(
            f"{field} must be finite and non-negative"
        )
    return result


def _decimal_quantity_to_bytes(value: Any, multiplier: int, field: str) -> int:
    try:
        byte_value = Decimal(str(value)) * Decimal(multiplier)
    except (InvalidOperation, ValueError) as exc:
        raise EnduranceConfigError(f"{field} must be numeric") from exc
    if byte_value <= 0 or byte_value != byte_value.to_integral_value():
        raise EnduranceConfigError(
            f"{field} must resolve to a positive whole number of bytes"
        )
    return int(byte_value)


def _first_present(
    mappings: Iterable[Mapping[str, Any]], keys: Sequence[str]
) -> Tuple[bool, Any]:
    for mapping in mappings:
        for key in keys:
            if key in mapping:
                return True, mapping[key]
    return False, None


@dataclass(frozen=True)
class EnduranceRating:
    """One workload-specific host endurance rating for an SSD model."""

    name: str
    workload: str
    warranty_years: float
    rated_tbw_bytes: Optional[int] = None
    rated_dwpd: Optional[float] = None
    sensitivity_only: bool = False
    accounting_basis: str = "host_tbw"

    def __post_init__(self) -> None:
        if not self.name:
            raise EnduranceConfigError("rating name must not be empty")
        if not self.workload:
            raise EnduranceConfigError(
                f"rating {self.name!r} must describe its workload"
            )
        if self.accounting_basis != "host_tbw":
            raise EnduranceConfigError(
                f"rating {self.name!r} uses unsupported accounting basis "
                f"{self.accounting_basis!r}; only host_tbw is supported"
            )
        if self.warranty_years <= 0:
            raise EnduranceConfigError(
                f"rating {self.name!r} warranty_years must be positive"
            )
        if self.rated_tbw_bytes is None and self.rated_dwpd is None:
            raise EnduranceConfigError(
                f"rating {self.name!r} needs rated_tbw_tb or rated_dwpd"
            )
        if self.rated_tbw_bytes is not None and self.rated_tbw_bytes <= 0:
            raise EnduranceConfigError(
                f"rating {self.name!r} rated TBW must be positive"
            )
        if self.rated_dwpd is not None and self.rated_dwpd <= 0:
            raise EnduranceConfigError(
                f"rating {self.name!r} rated DWPD must be positive"
            )

    def resolve_tbw_bytes(self, capacity_bytes: int) -> int:
        if self.rated_tbw_bytes is not None:
            return self.rated_tbw_bytes
        assert self.rated_dwpd is not None
        return int(round(
            capacity_bytes
            * self.rated_dwpd
            * DEFAULT_DAYS_PER_YEAR
            * self.warranty_years
        ))

    def resolve_dwpd(self, capacity_bytes: int) -> float:
        if self.rated_dwpd is not None:
            return self.rated_dwpd
        assert self.rated_tbw_bytes is not None
        return (
            self.rated_tbw_bytes
            / capacity_bytes
            / DEFAULT_DAYS_PER_YEAR
            / self.warranty_years
        )


@dataclass(frozen=True)
class DeviceProfile:
    """Vendor SSD capacity and one or more workload-specific ratings."""

    profile_id: str
    vendor: str
    model: str
    capacity_bytes: int
    default_rating: str
    ratings: Mapping[str, EnduranceRating]
    source_url: str
    source_title: str = ""
    source_revision: str = ""

    def __post_init__(self) -> None:
        if not self.profile_id or not self.vendor or not self.model:
            raise EnduranceConfigError(
                "profile_id, vendor, and model must not be empty"
            )
        if self.capacity_bytes <= 0:
            raise EnduranceConfigError("device capacity must be positive")
        if self.default_rating not in self.ratings:
            raise EnduranceConfigError(
                f"default rating {self.default_rating!r} is not in ratings"
            )
        if not self.source_url:
            raise EnduranceConfigError("device profile needs a source_url")

        for name, rating in self.ratings.items():
            if name != rating.name:
                raise EnduranceConfigError(
                    f"rating key {name!r} does not match name {rating.name!r}"
                )
            if rating.rated_tbw_bytes is not None and rating.rated_dwpd is not None:
                derived = int(round(
                    self.capacity_bytes
                    * rating.rated_dwpd
                    * DEFAULT_DAYS_PER_YEAR
                    * rating.warranty_years
                ))
                relative_error = abs(derived - rating.rated_tbw_bytes) / float(
                    rating.rated_tbw_bytes
                )
                if relative_error > 0.02:
                    raise EnduranceConfigError(
                        f"rating {name!r} TBW and DWPD differ by more than 2%"
                    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeviceProfile":
        if "capacity_gb_decimal" in data:
            capacity_bytes = _decimal_quantity_to_bytes(
                data["capacity_gb_decimal"], DECIMAL_GB, "capacity_gb_decimal"
            )
        elif "capacity_tb_decimal" in data:
            capacity_bytes = _decimal_quantity_to_bytes(
                data["capacity_tb_decimal"], DECIMAL_TB, "capacity_tb_decimal"
            )
        elif "capacity_bytes" in data:
            capacity_bytes = _as_nonnegative_int(
                data["capacity_bytes"], "capacity_bytes"
            )
            if capacity_bytes == 0:
                raise EnduranceConfigError("capacity_bytes must be positive")
        else:
            raise EnduranceConfigError(
                "device profile needs capacity_gb_decimal, "
                "capacity_tb_decimal, or capacity_bytes"
            )

        raw_ratings = data.get("ratings")
        if not isinstance(raw_ratings, Mapping) or not raw_ratings:
            raise EnduranceConfigError("device profile needs a ratings object")

        ratings: Dict[str, EnduranceRating] = {}
        for name, raw in raw_ratings.items():
            if not isinstance(raw, Mapping):
                raise EnduranceConfigError(f"rating {name!r} must be an object")

            rated_tbw_bytes = None
            if "rated_tbw_tb" in raw:
                rated_tbw_bytes = _decimal_quantity_to_bytes(
                    raw["rated_tbw_tb"], DECIMAL_TB,
                    f"ratings.{name}.rated_tbw_tb",
                )
            elif "rated_tbw_bytes" in raw:
                rated_tbw_bytes = _as_nonnegative_int(
                    raw["rated_tbw_bytes"],
                    f"ratings.{name}.rated_tbw_bytes",
                )

            rated_dwpd = None
            if "rated_dwpd" in raw:
                rated_dwpd = _as_positive_float(
                    raw["rated_dwpd"], f"ratings.{name}.rated_dwpd"
                )

            ratings[str(name)] = EnduranceRating(
                name=str(name),
                workload=str(raw.get("workload", "")),
                warranty_years=_as_positive_float(
                    raw.get("warranty_years"),
                    f"ratings.{name}.warranty_years",
                ),
                rated_tbw_bytes=rated_tbw_bytes,
                rated_dwpd=rated_dwpd,
                sensitivity_only=bool(raw.get("sensitivity_only", False)),
                accounting_basis=str(raw.get("accounting_basis", "host_tbw")),
            )

        source = data.get("source", {})
        if not isinstance(source, Mapping):
            raise EnduranceConfigError("source must be an object")
        return cls(
            profile_id=str(data.get("profile_id", "")),
            vendor=str(data.get("vendor", "")),
            model=str(data.get("model", "")),
            capacity_bytes=capacity_bytes,
            default_rating=str(data.get("default_rating", "")),
            ratings=ratings,
            source_url=str(source.get("url", data.get("source_url", ""))),
            source_title=str(source.get("title", "")),
            source_revision=str(source.get("revision", "")),
        )

    @classmethod
    def from_json_file(cls, path: Any) -> "DeviceProfile":
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, Mapping):
            raise EnduranceConfigError("device profile root must be an object")
        return cls.from_dict(data)

    def select_rating(self, name: Optional[str] = None) -> EnduranceRating:
        selected = name or self.default_rating
        try:
            return self.ratings[selected]
        except KeyError as exc:
            choices = ", ".join(sorted(self.ratings))
            raise EnduranceConfigError(
                f"unknown rating {selected!r}; choose one of: {choices}"
            ) from exc


@dataclass(frozen=True)
class DeviceTraceWrites:
    """Host traffic sent to one physical SSD during one trace epoch."""

    device_id: str
    host_write_bytes: int
    host_read_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.device_id:
            raise EnduranceConfigError("device_id must not be empty")
        if self.host_write_bytes < 0 or self.host_read_bytes < 0:
            raise EnduranceConfigError("device traffic must be non-negative")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], index: int) -> "DeviceTraceWrites":
        write_found, write_value = _first_present(
            [data],
            ("aligned_host_write_bytes", "host_write_bytes", "trace_host_write_bytes"),
        )
        if not write_found:
            raise EnduranceConfigError(
                f"devices[{index}] needs host_write_bytes"
            )
        read_found, read_value = _first_present(
            [data], ("host_read_bytes", "trace_host_read_bytes")
        )
        return cls(
            device_id=str(data.get("device_id", f"ssd{index}")),
            host_write_bytes=_as_nonnegative_int(
                write_value, f"devices[{index}].host_write_bytes"
            ),
            host_read_bytes=_as_nonnegative_int(
                read_value if read_found else 0,
                f"devices[{index}].host_read_bytes",
            ),
        )


@dataclass(frozen=True)
class RunWriteStats:
    """Physical host traffic observed during one workload epoch."""

    run_id: str
    host_write_bytes: int
    host_read_bytes: int = 0
    trace_period_seconds: Optional[float] = None
    devices: Tuple[DeviceTraceWrites, ...] = ()

    def __post_init__(self) -> None:
        if self.host_write_bytes < 0 or self.host_read_bytes < 0:
            raise EnduranceConfigError("aggregate traffic must be non-negative")
        if self.trace_period_seconds is not None and self.trace_period_seconds <= 0:
            raise EnduranceConfigError("trace_period_seconds must be positive")
        ids = [device.device_id for device in self.devices]
        if len(ids) != len(set(ids)):
            raise EnduranceConfigError("device IDs must be unique")
        if self.devices:
            device_writes = sum(device.host_write_bytes for device in self.devices)
            device_reads = sum(device.host_read_bytes for device in self.devices)
            if device_writes != self.host_write_bytes:
                raise EnduranceConfigError(
                    "aggregate host_write_bytes must equal the explicit "
                    "per-device write sum"
                )
            if device_reads != self.host_read_bytes:
                raise EnduranceConfigError(
                    "aggregate host_read_bytes must equal the explicit "
                    "per-device read sum"
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunWriteStats":
        storage = data.get("storage", {})
        if not isinstance(storage, Mapping):
            raise EnduranceConfigError("storage must be an object")
        top_totals = data.get("totals", {})
        storage_totals = storage.get("totals", {})
        if not isinstance(top_totals, Mapping) or not isinstance(
            storage_totals, Mapping
        ):
            raise EnduranceConfigError("totals must be an object")
        traffic_maps = [data, storage, top_totals, storage_totals]

        raw_devices = storage.get("devices", data.get("devices", None))
        devices: List[DeviceTraceWrites] = []
        if raw_devices is not None:
            if not isinstance(raw_devices, list):
                raise EnduranceConfigError("devices must be a list")
            for index, raw_device in enumerate(raw_devices):
                if not isinstance(raw_device, Mapping):
                    raise EnduranceConfigError(
                        f"devices[{index}] must be an object"
                    )
                devices.append(DeviceTraceWrites.from_dict(raw_device, index))
        else:
            write_map = storage.get(
                "device_host_write_bytes", data.get("device_host_write_bytes")
            )
            read_map = storage.get(
                "device_host_read_bytes", data.get("device_host_read_bytes", {})
            )
            if write_map is not None:
                if not isinstance(write_map, Mapping) or not isinstance(
                    read_map, Mapping
                ):
                    raise EnduranceConfigError(
                        "device_host_write_bytes/read_bytes must be objects"
                    )
                for index, (device_id, value) in enumerate(write_map.items()):
                    devices.append(DeviceTraceWrites(
                        device_id=str(device_id),
                        host_write_bytes=_as_nonnegative_int(
                            value, f"device_host_write_bytes.{device_id}"
                        ),
                        host_read_bytes=_as_nonnegative_int(
                            read_map.get(device_id, 0),
                            f"device_host_read_bytes.{device_id}",
                        ),
                    ))

        write_found, write_value = _first_present(
            traffic_maps,
            ("aligned_host_write_bytes", "host_write_bytes", "trace_host_write_bytes"),
        )
        read_found, read_value = _first_present(
            traffic_maps, ("host_read_bytes", "trace_host_read_bytes")
        )
        if not write_found:
            if not devices:
                raise EnduranceConfigError(
                    "run stats need host_write_bytes or explicit devices"
                )
            write_value = sum(device.host_write_bytes for device in devices)
        if not read_found:
            read_value = sum(device.host_read_bytes for device in devices)

        aggregate_reads = _as_nonnegative_int(read_value, "host_read_bytes")
        if devices and aggregate_reads > 0 and not any(
            device.host_read_bytes for device in devices
        ):
            # Explicit device writes are useful for identifying a hot drive,
            # even when the producer only has aggregate read traffic. Reads
            # do not affect endurance, so balance that aggregate rather than
            # rejecting an otherwise valid write-endurance input.
            balanced_reads = _balanced_device_traffic(
                aggregate_reads, len(devices)
            )
            devices = [
                DeviceTraceWrites(
                    device.device_id,
                    device.host_write_bytes,
                    balanced_reads[index],
                )
                for index, device in enumerate(devices)
            ]

        period_found, period_value = _first_present(
            [data, storage],
            ("trace_period_seconds", "offered_trace_period_seconds"),
        )
        if period_found:
            trace_period_seconds = _as_positive_float(
                period_value, "trace_period_seconds"
            )
        else:
            ns_found, ns_value = _first_present(
                [data, storage],
                ("trace_period_ns", "offered_trace_period_ns"),
            )
            trace_period_seconds = (
                _as_positive_float(ns_value, "trace_period_ns") / 1e9
                if ns_found else None
            )

        return cls(
            run_id=str(data.get("run_id", "unknown")),
            host_write_bytes=_as_nonnegative_int(
                write_value, "host_write_bytes"
            ),
            host_read_bytes=aggregate_reads,
            trace_period_seconds=trace_period_seconds,
            devices=tuple(devices),
        )

    @classmethod
    def from_json_file(cls, path: Any) -> "RunWriteStats":
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, Mapping):
            raise EnduranceConfigError("run stats root must be an object")
        return cls.from_dict(data)


@dataclass(frozen=True)
class ProjectionAssumptions:
    """How one observed trace epoch is repeated in deployment."""

    replays_per_day: Optional[float] = None
    trace_period_seconds: Optional[float] = None
    duty_cycle: float = 1.0
    waf: float = 1.0
    days_per_year: float = DEFAULT_DAYS_PER_YEAR
    background_dwpd: float = 0.0
    initial_percentage_used: float = 0.0
    replay_semantics: str = "new_logical_sessions"

    def __post_init__(self) -> None:
        if self.replays_per_day is not None:
            _as_nonnegative_float(self.replays_per_day, "replays_per_day")
        if self.trace_period_seconds is not None:
            _as_positive_float(self.trace_period_seconds, "trace_period_seconds")
        if self.replays_per_day is not None and self.trace_period_seconds is not None:
            raise EnduranceConfigError(
                "set replays_per_day or trace_period_seconds, not both"
            )
        duty_cycle = _as_nonnegative_float(self.duty_cycle, "duty_cycle")
        if duty_cycle > 1.0:
            raise EnduranceConfigError("duty_cycle must be between 0 and 1")
        if self.replays_per_day is not None and duty_cycle != 1.0:
            raise EnduranceConfigError(
                "duty_cycle applies to trace_period_seconds, not a direct "
                "replays_per_day value"
            )
        _as_positive_float(self.waf, "waf")
        _as_positive_float(self.days_per_year, "days_per_year")
        _as_nonnegative_float(self.background_dwpd, "background_dwpd")
        _as_nonnegative_float(
            self.initial_percentage_used, "initial_percentage_used"
        )
        if not self.replay_semantics:
            raise EnduranceConfigError("replay_semantics must not be empty")

    def resolve_replays_per_day(
        self, stats_trace_period_seconds: Optional[float] = None
    ) -> float:
        if self.replays_per_day is not None:
            return float(self.replays_per_day)
        period = self.trace_period_seconds or stats_trace_period_seconds
        if period is None:
            raise EnduranceConfigError(
                "projection needs replays_per_day or trace_period_seconds"
            )
        return self.duty_cycle * SECONDS_PER_DAY / period


@dataclass(frozen=True)
class DeviceEnduranceResult:
    device_id: str
    capacity_bytes: int
    rated_tbw_bytes: int
    remaining_tbw_bytes: float
    rated_dwpd: float
    warranty_years: float
    trace_host_write_bytes: int
    trace_host_read_bytes: int
    trace_estimated_nand_write_bytes: float
    replays_per_day: float
    host_write_bytes_per_day: float
    estimated_nand_write_bytes_per_day: float
    observed_host_dwpd: float
    rating_utilization: float
    host_wear_fraction_per_trace: float
    traces_to_tbw: Optional[float]
    years_to_tbw: Optional[float]
    warranty_covered_years: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "capacity_bytes": self.capacity_bytes,
            "rated_tbw_bytes": self.rated_tbw_bytes,
            "remaining_tbw_bytes": self.remaining_tbw_bytes,
            "rated_dwpd": self.rated_dwpd,
            "warranty_years": self.warranty_years,
            "trace_host_write_bytes": self.trace_host_write_bytes,
            "trace_host_read_bytes": self.trace_host_read_bytes,
            "trace_estimated_nand_write_bytes": (
                self.trace_estimated_nand_write_bytes
            ),
            "replays_per_day": self.replays_per_day,
            "host_write_bytes_per_day": self.host_write_bytes_per_day,
            "estimated_nand_write_bytes_per_day": (
                self.estimated_nand_write_bytes_per_day
            ),
            "observed_host_dwpd": self.observed_host_dwpd,
            "rating_utilization": self.rating_utilization,
            "host_wear_fraction_per_trace": self.host_wear_fraction_per_trace,
            "traces_to_tbw": self.traces_to_tbw,
            "years_to_tbw": self.years_to_tbw,
            "endurance_unbounded_at_projected_write_rate": (
                self.years_to_tbw is None
            ),
            "warranty_covered_years": self.warranty_covered_years,
            "endurance_exhausted_before_warranty": (
                self.years_to_tbw is not None
                and self.years_to_tbw < self.warranty_years
            ),
        }


@dataclass(frozen=True)
class EnduranceReport:
    run_id: str
    profile: DeviceProfile
    rating: EnduranceRating
    distribution_mode: str
    accounting_mode: str
    assumptions: ProjectionAssumptions
    effective_replays_per_day: float
    effective_trace_period_seconds: Optional[float]
    trace_host_write_bytes: int
    trace_host_read_bytes: int
    devices: Tuple[DeviceEnduranceResult, ...]
    pool_years_to_first_device_eol: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "device_profile": {
                "profile_id": self.profile.profile_id,
                "vendor": self.profile.vendor,
                "model": self.profile.model,
                "capacity_bytes_per_device": self.profile.capacity_bytes,
                "num_devices": len(self.devices),
                "source_url": self.profile.source_url,
                "source_title": self.profile.source_title,
                "source_revision": self.profile.source_revision,
            },
            "rating": {
                "name": self.rating.name,
                "workload": self.rating.workload,
                "sensitivity_only": self.rating.sensitivity_only,
                "accounting_basis": self.rating.accounting_basis,
                "warranty_years": self.rating.warranty_years,
                "rated_tbw_bytes_per_device": self.rating.resolve_tbw_bytes(
                    self.profile.capacity_bytes
                ),
                "rated_dwpd": self.rating.resolve_dwpd(
                    self.profile.capacity_bytes
                ),
            },
            "assumptions": {
                "distribution_mode": self.distribution_mode,
                "accounting_mode": self.accounting_mode,
                "waf": self.assumptions.waf,
                "waf_affects_lifetime": False,
                "effective_replays_per_day": self.effective_replays_per_day,
                "trace_period_seconds": self.effective_trace_period_seconds,
                "duty_cycle": self.assumptions.duty_cycle,
                "days_per_year": self.assumptions.days_per_year,
                "background_dwpd": self.assumptions.background_dwpd,
                "initial_percentage_used": (
                    self.assumptions.initial_percentage_used
                ),
                "replay_semantics": self.assumptions.replay_semantics,
                "si_units": True,
            },
            "trace_totals": {
                "host_write_bytes": self.trace_host_write_bytes,
                "host_read_bytes": self.trace_host_read_bytes,
                "estimated_nand_write_bytes": (
                    self.trace_host_write_bytes * self.assumptions.waf
                ),
            },
            "pool": {
                "years_to_first_device_eol": (
                    self.pool_years_to_first_device_eol
                ),
                "endurance_unbounded_at_projected_write_rate": (
                    self.pool_years_to_first_device_eol is None
                ),
            },
            "devices": [device.to_dict() for device in self.devices],
        }


CSV_FIELDS = (
    "run_id",
    "profile_id",
    "vendor",
    "model",
    "rating",
    "rating_workload",
    "sensitivity_only",
    "source_url",
    "distribution_mode",
    "accounting_mode",
    "device_id",
    "capacity_bytes",
    "rated_tbw_bytes",
    "rated_dwpd",
    "warranty_years",
    "trace_period_seconds",
    "duty_cycle",
    "replays_per_day",
    "waf",
    "waf_affects_lifetime",
    "trace_host_write_bytes",
    "trace_host_read_bytes",
    "trace_estimated_nand_write_bytes",
    "host_write_bytes_per_day",
    "estimated_nand_write_bytes_per_day",
    "observed_host_dwpd",
    "rating_utilization",
    "host_wear_fraction_per_trace",
    "traces_to_tbw",
    "years_to_tbw",
    "warranty_covered_years",
    "pool_years_to_first_device_eol",
)


def _balanced_device_traffic(
    total: int, num_devices: int
) -> List[int]:
    quotient, remainder = divmod(total, num_devices)
    return [
        quotient + (1 if index < remainder else 0)
        for index in range(num_devices)
    ]


def project_endurance(
    stats: RunWriteStats,
    profile: DeviceProfile,
    assumptions: ProjectionAssumptions,
    num_devices: Optional[int] = None,
    rating_name: Optional[str] = None,
) -> EnduranceReport:
    """Project per-device and pool endurance using host-visible writes.

    Explicit per-device counters in ``stats`` always win.  Otherwise the
    aggregate write and read counts are distributed as evenly as possible
    across ``num_devices`` (one by default).
    """

    rating = profile.select_rating(rating_name)
    effective_replays = assumptions.resolve_replays_per_day(
        stats.trace_period_seconds
    )

    if stats.devices:
        if num_devices is not None and num_devices != len(stats.devices):
            raise EnduranceConfigError(
                "num_devices does not match explicit per-device counters"
            )
        device_traffic = list(stats.devices)
        distribution_mode = "explicit"
    else:
        count = 1 if num_devices is None else _as_nonnegative_int(
            num_devices, "num_devices"
        )
        if count <= 0:
            raise EnduranceConfigError("num_devices must be positive")
        writes = _balanced_device_traffic(stats.host_write_bytes, count)
        reads = _balanced_device_traffic(stats.host_read_bytes, count)
        device_traffic = [
            DeviceTraceWrites(f"ssd{index}", writes[index], reads[index])
            for index in range(count)
        ]
        distribution_mode = "balanced"

    capacity = profile.capacity_bytes
    rated_tbw = rating.resolve_tbw_bytes(capacity)
    rated_dwpd = rating.resolve_dwpd(capacity)
    used_fraction = min(assumptions.initial_percentage_used / 100.0, 1.0)
    remaining_tbw = rated_tbw * (1.0 - used_fraction)

    results: List[DeviceEnduranceResult] = []
    for traffic in device_traffic:
        host_per_day = (
            traffic.host_write_bytes * effective_replays
            + assumptions.background_dwpd * capacity
        )
        nand_per_day = host_per_day * assumptions.waf
        observed_dwpd = host_per_day / capacity
        rating_utilization = observed_dwpd / rated_dwpd
        wear_per_trace = traffic.host_write_bytes / float(rated_tbw)

        if remaining_tbw == 0:
            traces_to_tbw: Optional[float] = 0.0
            years_to_tbw: Optional[float] = 0.0
        else:
            traces_to_tbw = (
                remaining_tbw / traffic.host_write_bytes
                if traffic.host_write_bytes > 0 else None
            )
            years_to_tbw = (
                remaining_tbw / (host_per_day * assumptions.days_per_year)
                if host_per_day > 0 else None
            )

        warranty_covered_years = (
            rating.warranty_years
            if years_to_tbw is None
            else min(years_to_tbw, rating.warranty_years)
        )
        results.append(DeviceEnduranceResult(
            device_id=traffic.device_id,
            capacity_bytes=capacity,
            rated_tbw_bytes=rated_tbw,
            remaining_tbw_bytes=remaining_tbw,
            rated_dwpd=rated_dwpd,
            warranty_years=rating.warranty_years,
            trace_host_write_bytes=traffic.host_write_bytes,
            trace_host_read_bytes=traffic.host_read_bytes,
            trace_estimated_nand_write_bytes=(
                traffic.host_write_bytes * assumptions.waf
            ),
            replays_per_day=effective_replays,
            host_write_bytes_per_day=host_per_day,
            estimated_nand_write_bytes_per_day=nand_per_day,
            observed_host_dwpd=observed_dwpd,
            rating_utilization=rating_utilization,
            host_wear_fraction_per_trace=wear_per_trace,
            traces_to_tbw=traces_to_tbw,
            years_to_tbw=years_to_tbw,
            warranty_covered_years=warranty_covered_years,
        ))

    finite_lifetimes = [
        result.years_to_tbw
        for result in results
        if result.years_to_tbw is not None
    ]
    pool_lifetime = min(finite_lifetimes) if finite_lifetimes else None
    effective_period = assumptions.trace_period_seconds
    if effective_period is None and assumptions.replays_per_day is None:
        effective_period = stats.trace_period_seconds

    return EnduranceReport(
        run_id=stats.run_id,
        profile=profile,
        rating=rating,
        distribution_mode=distribution_mode,
        accounting_mode="host_tbw",
        assumptions=assumptions,
        effective_replays_per_day=effective_replays,
        effective_trace_period_seconds=effective_period,
        trace_host_write_bytes=stats.host_write_bytes,
        trace_host_read_bytes=stats.host_read_bytes,
        devices=tuple(results),
        pool_years_to_first_device_eol=pool_lifetime,
    )


def write_report_json(report: EnduranceReport, path: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(report.to_dict(), file, indent=2, sort_keys=True,
                  allow_nan=False)
        file.write("\n")


def _csv_optional(value: Optional[float]) -> Any:
    return "inf" if value is None else value


def write_report_csv(report: EnduranceReport, path: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for device in report.devices:
            writer.writerow({
                "run_id": report.run_id,
                "profile_id": report.profile.profile_id,
                "vendor": report.profile.vendor,
                "model": report.profile.model,
                "rating": report.rating.name,
                "rating_workload": report.rating.workload,
                "sensitivity_only": report.rating.sensitivity_only,
                "source_url": report.profile.source_url,
                "distribution_mode": report.distribution_mode,
                "accounting_mode": report.accounting_mode,
                "device_id": device.device_id,
                "capacity_bytes": device.capacity_bytes,
                "rated_tbw_bytes": device.rated_tbw_bytes,
                "rated_dwpd": device.rated_dwpd,
                "warranty_years": device.warranty_years,
                "trace_period_seconds": (
                    report.effective_trace_period_seconds or ""
                ),
                "duty_cycle": report.assumptions.duty_cycle,
                "replays_per_day": report.effective_replays_per_day,
                "waf": report.assumptions.waf,
                "waf_affects_lifetime": False,
                "trace_host_write_bytes": device.trace_host_write_bytes,
                "trace_host_read_bytes": device.trace_host_read_bytes,
                "trace_estimated_nand_write_bytes": (
                    device.trace_estimated_nand_write_bytes
                ),
                "host_write_bytes_per_day": device.host_write_bytes_per_day,
                "estimated_nand_write_bytes_per_day": (
                    device.estimated_nand_write_bytes_per_day
                ),
                "observed_host_dwpd": device.observed_host_dwpd,
                "rating_utilization": device.rating_utilization,
                "host_wear_fraction_per_trace": (
                    device.host_wear_fraction_per_trace
                ),
                "traces_to_tbw": _csv_optional(device.traces_to_tbw),
                "years_to_tbw": _csv_optional(device.years_to_tbw),
                "warranty_covered_years": device.warranty_covered_years,
                "pool_years_to_first_device_eol": _csv_optional(
                    report.pool_years_to_first_device_eol
                ),
            })


__all__ = [
    "DECIMAL_GB",
    "DECIMAL_TB",
    "DEFAULT_DAYS_PER_YEAR",
    "DeviceEnduranceResult",
    "DeviceProfile",
    "DeviceTraceWrites",
    "EnduranceConfigError",
    "EnduranceRating",
    "EnduranceReport",
    "ProjectionAssumptions",
    "RunWriteStats",
    "project_endurance",
    "write_report_csv",
    "write_report_json",
]

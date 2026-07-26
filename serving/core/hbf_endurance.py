"""HBF KV-media endurance projection from completed lifecycle reports.

The lifecycle exposes exact physical payload bytes per HBF card.  This
module converts duration-weighted payload rates into two explicitly
different endurance views:

* SSD host-TBW ratings used only as empirical full-drive-write proxies;
* raw HBF P/E-cycle sensitivities where write amplification affects wear.

Model weights are a one-time static-region write and are never included in
recurring KV wear.  Within each card, KV writes are assumed to be randomly
and uniformly spread over the writable KV region.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Optional, Sequence

from .endurance_model import DeviceProfile


SECONDS_PER_DAY = 86_400.0
DAYS_PER_YEAR = 365.0
HBF_ENDURANCE_SCHEMA_VERSION = 1
FLASHACCEL_ENDURANCE_SOURCE_URL = (
    "https://arxiv.org/html/2607.10186#S7.SS4"
)


class HBFEnduranceError(ValueError):
    """Raised when HBF write accounting cannot support a projection."""


def _positive_finite(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise HBFEnduranceError(
            f"{name} must be positive and finite")
    return float(value)


def _nonnegative_int(name: str, value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise HBFEnduranceError(
            f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class HBFEnduranceScenario:
    """One full-region-write budget and amplification assumption."""

    key: str
    rated_full_region_writes: float
    write_amplification_factor: float
    accounting_basis: str
    waf_affects_lifetime: bool
    assumption: str
    source_url: str

    def __post_init__(self) -> None:
        if not self.key:
            raise HBFEnduranceError(
                "endurance scenario key cannot be empty")
        _positive_finite(
            "rated_full_region_writes",
            self.rated_full_region_writes,
        )
        _positive_finite(
            "write_amplification_factor",
            self.write_amplification_factor,
        )
        if self.accounting_basis not in {
            "ssd_host_tbw_proxy",
            "raw_hbf_pe_cycles",
        }:
            raise HBFEnduranceError(
                "unsupported endurance accounting basis")
        if not isinstance(self.waf_affects_lifetime, bool):
            raise HBFEnduranceError(
                "waf_affects_lifetime must be a boolean")
        if (
            self.accounting_basis == "ssd_host_tbw_proxy"
            and (
                self.waf_affects_lifetime
                or self.write_amplification_factor != 1.0
            )
        ):
            raise HBFEnduranceError(
                "SSD host-TBW proxies already embed device amplification")
        if not self.assumption or not self.source_url:
            raise HBFEnduranceError(
                "endurance scenario needs assumption and source URL")


@dataclass(frozen=True)
class HBFCardWriteSample:
    """One physical HBF card observed over one trace horizon."""

    device_id: str
    server_id: int
    card_id: int
    kv_region_capacity_bytes: int
    write_bytes: int
    wasted_write_bytes: int

    def __post_init__(self) -> None:
        if not self.device_id:
            raise HBFEnduranceError(
                "HBF device_id cannot be empty")
        for name, value in (
            ("server_id", self.server_id),
            ("card_id", self.card_id),
            (
                "kv_region_capacity_bytes",
                self.kv_region_capacity_bytes,
            ),
            ("write_bytes", self.write_bytes),
            ("wasted_write_bytes", self.wasted_write_bytes),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise HBFEnduranceError(
                    f"{name} must be a non-negative integer")
        if self.kv_region_capacity_bytes <= 0:
            raise HBFEnduranceError(
                "kv_region_capacity_bytes must be positive")
        if self.wasted_write_bytes > self.write_bytes:
            raise HBFEnduranceError(
                "wasted HBF writes exceed admitted writes")


@dataclass(frozen=True)
class HBFWriteSample:
    """A completed trace's exact HBF card writes and duration."""

    run_id: str
    duration_seconds: float
    cards: tuple[HBFCardWriteSample, ...]
    total_write_bytes: int
    wasted_write_bytes: int
    model_weight_bytes_per_card_excluded: int

    def __post_init__(self) -> None:
        if not self.run_id:
            raise HBFEnduranceError("run_id cannot be empty")
        _positive_finite("duration_seconds", self.duration_seconds)
        if not self.cards:
            raise HBFEnduranceError(
                "HBF write sample needs physical cards")
        device_ids = [card.device_id for card in self.cards]
        if len(device_ids) != len(set(device_ids)):
            raise HBFEnduranceError(
                "HBF write sample device IDs are not unique")
        _nonnegative_int(
            "total_write_bytes", self.total_write_bytes)
        _nonnegative_int(
            "wasted_write_bytes", self.wasted_write_bytes)
        _nonnegative_int(
            "model_weight_bytes_per_card_excluded",
            self.model_weight_bytes_per_card_excluded,
        )
        if sum(card.write_bytes for card in self.cards) != (
                self.total_write_bytes):
            raise HBFEnduranceError(
                "sample total does not equal per-card HBF writes")
        if sum(card.wasted_write_bytes for card in self.cards) != (
                self.wasted_write_bytes):
            raise HBFEnduranceError(
                "sample wasted total does not equal per-card HBF writes")

    @classmethod
    def from_write_accounting(
            cls, *, run_id: str, duration_seconds: float,
            write_accounting: Mapping[str, object],
    ) -> "HBFWriteSample":
        if not isinstance(write_accounting, Mapping):
            raise HBFEnduranceError(
                "write_accounting must be a mapping")
        if write_accounting.get("schema_version") != 1:
            raise HBFEnduranceError(
                "unsupported HBF write accounting schema")
        if not write_accounting.get(
                "complete_for_endurance_projection"):
            raise HBFEnduranceError(
                "HBF write accounting has pending jobs")
        if write_accounting.get("accounting_basis") != (
                "physical_media_payload_of_admitted_jobs"):
            raise HBFEnduranceError(
                "unsupported HBF write accounting basis")
        raw_weight = write_accounting.get("static_model_weight")
        if not isinstance(raw_weight, Mapping):
            raise HBFEnduranceError(
                "HBF write accounting lacks static model weight")
        if raw_weight.get(
                "included_in_recurring_kv_wear") is not False:
            raise HBFEnduranceError(
                "model weights must be excluded from recurring KV wear")
        weight_bytes = _nonnegative_int(
            "static_model_weight.bytes_per_card",
            raw_weight.get("bytes_per_card"),
        )
        raw_cards = write_accounting.get("cards")
        if not isinstance(raw_cards, list) or not raw_cards:
            raise HBFEnduranceError(
                "HBF write accounting needs a card list")
        cards = []
        for index, raw in enumerate(raw_cards):
            if not isinstance(raw, Mapping):
                raise HBFEnduranceError(
                    f"cards[{index}] must be a mapping")
            cards.append(HBFCardWriteSample(
                device_id=str(raw.get("device_id", "")),
                server_id=_nonnegative_int(
                    f"cards[{index}].server_id",
                    raw.get("server_id"),
                ),
                card_id=_nonnegative_int(
                    f"cards[{index}].card_id",
                    raw.get("card_id"),
                ),
                kv_region_capacity_bytes=_nonnegative_int(
                    f"cards[{index}].kv_region_capacity_bytes",
                    raw.get("kv_region_capacity_bytes"),
                ),
                write_bytes=_nonnegative_int(
                    f"cards[{index}].total_write_bytes",
                    raw.get("total_write_bytes"),
                ),
                wasted_write_bytes=_nonnegative_int(
                    f"cards[{index}].wasted_write_bytes",
                    raw.get("wasted_write_bytes"),
                ),
            ))
        return cls(
            run_id=run_id,
            duration_seconds=_positive_finite(
                "duration_seconds", duration_seconds),
            cards=tuple(cards),
            total_write_bytes=_nonnegative_int(
                "total_physical_write_bytes",
                write_accounting.get("total_physical_write_bytes"),
            ),
            wasted_write_bytes=_nonnegative_int(
                "wasted_physical_write_bytes",
                write_accounting.get("wasted_physical_write_bytes"),
            ),
            model_weight_bytes_per_card_excluded=weight_bytes,
        )


def default_hbf_endurance_scenarios(
        ssd_profile: DeviceProfile,
) -> tuple[HBFEnduranceScenario, ...]:
    """Build SSD proxy anchors plus HBF raw-P/E sensitivity bands."""

    scenarios = []
    for key, rating_name in (
        ("ssd_proxy_random_4k", "conservative_4k_random"),
        (
            "ssd_proxy_sequential_128k",
            "sequential_128k_sensitivity",
        ),
    ):
        rating = ssd_profile.select_rating(rating_name)
        full_writes = (
            rating.resolve_tbw_bytes(ssd_profile.capacity_bytes)
            / ssd_profile.capacity_bytes
        )
        scenarios.append(HBFEnduranceScenario(
            key=key,
            rated_full_region_writes=full_writes,
            write_amplification_factor=1.0,
            accounting_basis="ssd_host_tbw_proxy",
            waf_affects_lifetime=False,
            assumption=(
                f"{ssd_profile.vendor} {ssd_profile.model} "
                f"{rating.name} host-TBW used as an empirical proxy; "
                "this is not an HBF product rating"
            ),
            source_url=ssd_profile.source_url,
        ))
    for pe_cycles, label in (
        (100_000.0, "slc_100k_pe"),
        (1_000_000.0, "retention_relaxed_1m_pe"),
    ):
        for waf in (1.0, 1.3, 2.0):
            scenarios.append(HBFEnduranceScenario(
                key=f"{label}_waf{waf:g}",
                rated_full_region_writes=pe_cycles,
                write_amplification_factor=waf,
                accounting_basis="raw_hbf_pe_cycles",
                waf_affects_lifetime=True,
                assumption=(
                    "FlashAccel-inspired raw HBF SLC P/E sensitivity; "
                    "the simulator observes payload bytes, while erase, "
                    "garbage-collection, and program-granularity overhead "
                    "are represented only by WAF"
                ),
                source_url=FLASHACCEL_ENDURANCE_SOURCE_URL,
            ))
    return tuple(scenarios)


def project_hbf_endurance(
        samples: Sequence[HBFWriteSample],
        scenarios: Sequence[HBFEnduranceScenario],
        *,
        service_lifetime_years: float = 5.0,
) -> dict[str, object]:
    """Project card and pool life using duration-weighted write rates."""

    lifetime_years = _positive_finite(
        "service_lifetime_years", service_lifetime_years)
    sample_values = tuple(samples)
    scenario_values = tuple(scenarios)
    if not sample_values:
        raise HBFEnduranceError("samples cannot be empty")
    if not scenario_values:
        raise HBFEnduranceError("scenarios cannot be empty")
    scenario_keys = [scenario.key for scenario in scenario_values]
    if len(scenario_keys) != len(set(scenario_keys)):
        raise HBFEnduranceError(
            "endurance scenario keys are not unique")

    first_cards = {
        card.device_id: card for card in sample_values[0].cards
    }
    weights = {
        sample.model_weight_bytes_per_card_excluded
        for sample in sample_values
    }
    if len(weights) != 1:
        raise HBFEnduranceError(
            "samples exclude different model-weight bytes")
    write_bytes_by_device = {
        device_id: 0 for device_id in first_cards
    }
    wasted_bytes_by_device = {
        device_id: 0 for device_id in first_cards
    }
    for sample in sample_values:
        cards = {card.device_id: card for card in sample.cards}
        if set(cards) != set(first_cards):
            raise HBFEnduranceError(
                "samples cover different HBF devices")
        for device_id, card in cards.items():
            reference = first_cards[device_id]
            if (
                card.server_id != reference.server_id
                or card.card_id != reference.card_id
                or card.kv_region_capacity_bytes
                != reference.kv_region_capacity_bytes
            ):
                raise HBFEnduranceError(
                    "HBF device geometry changed across samples")
            write_bytes_by_device[device_id] += card.write_bytes
            wasted_bytes_by_device[device_id] += (
                card.wasted_write_bytes)
    duration_seconds = sum(
        sample.duration_seconds for sample in sample_values)
    total_write_bytes = sum(write_bytes_by_device.values())
    total_wasted_bytes = sum(wasted_bytes_by_device.values())
    mean = total_write_bytes / len(first_cards)
    variance = (
        sum(
            (write_bytes - mean) ** 2
            for write_bytes in write_bytes_by_device.values()
        )
        / len(first_cards)
    )
    stddev = math.sqrt(variance)
    maximum = max(write_bytes_by_device.values())
    hottest_ids = sorted(
        device_id
        for device_id, write_bytes
        in write_bytes_by_device.items()
        if maximum > 0 and write_bytes == maximum
    )

    scenario_reports = {}
    for scenario in scenario_values:
        cards = []
        finite_lifetimes = []
        for device_id in sorted(first_cards):
            reference = first_cards[device_id]
            payload_per_second = (
                write_bytes_by_device[device_id]
                / duration_seconds
            )
            payload_per_day = (
                payload_per_second * SECONDS_PER_DAY)
            wear_factor = (
                scenario.write_amplification_factor
                if scenario.waf_affects_lifetime else 1.0
            )
            media_per_day = payload_per_day * wear_factor
            cycles_per_day = (
                media_per_day
                / reference.kv_region_capacity_bytes
            )
            years_to_eol: Optional[float] = (
                scenario.rated_full_region_writes
                / cycles_per_day
                / DAYS_PER_YEAR
                if cycles_per_day > 0.0 else None
            )
            if years_to_eol is not None:
                finite_lifetimes.append(years_to_eol)
            service_budget_fraction = (
                cycles_per_day
                * DAYS_PER_YEAR
                * lifetime_years
                / scenario.rated_full_region_writes
            )
            cards.append({
                "device_id": device_id,
                "server_id": reference.server_id,
                "card_id": reference.card_id,
                "kv_region_capacity_bytes": (
                    reference.kv_region_capacity_bytes),
                "trace_write_bytes": (
                    write_bytes_by_device[device_id]),
                "trace_wasted_write_bytes": (
                    wasted_bytes_by_device[device_id]),
                "payload_write_bytes_per_second": (
                    payload_per_second),
                "payload_write_bytes_per_day": payload_per_day,
                "wear_adjusted_write_bytes_per_day": media_per_day,
                "full_region_writes_per_day": cycles_per_day,
                "years_to_eol": years_to_eol,
                "endurance_unbounded_at_observed_write_rate": (
                    years_to_eol is None),
                "service_lifetime_budget_fraction": (
                    service_budget_fraction),
                "meets_service_lifetime": (
                    service_budget_fraction <= 1.0),
            })
        pool_lifetime = (
            min(finite_lifetimes)
            if finite_lifetimes else None
        )
        limiting_ids = sorted(
            card["device_id"]
            for card in cards
            if (
                pool_lifetime is not None
                and card["years_to_eol"] == pool_lifetime
            )
        )
        scenario_reports[scenario.key] = {
            "scenario": asdict(scenario),
            "service_lifetime_years": lifetime_years,
            "pool_years_to_first_card_eol": pool_lifetime,
            "pool_endurance_unbounded_at_observed_write_rate": (
                pool_lifetime is None),
            "pool_meets_service_lifetime": all(
                card["meets_service_lifetime"]
                for card in cards
            ),
            "limiting_device_ids": limiting_ids,
            "cards": cards,
        }

    return {
        "schema_version": HBF_ENDURANCE_SCHEMA_VERSION,
        "accounting_semantics": (
            "duration-weighted across completed trace samples; SSD "
            "host-TBW proxies do not receive an extra WAF, while raw HBF "
            "P/E sensitivities do"
        ),
        "wear_distribution_assumption": (
            "exact cross-card traffic; random uniform spreading within "
            "each card's writable KV region; cell/page/block hotness is "
            "not modeled"
        ),
        "sample_count": len(sample_values),
        "sample_run_ids": [
            sample.run_id for sample in sample_values],
        "total_observed_seconds": duration_seconds,
        "total_physical_write_bytes": total_write_bytes,
        "total_wasted_write_bytes": total_wasted_bytes,
        "wasted_write_fraction": (
            total_wasted_bytes / total_write_bytes
            if total_write_bytes else None
        ),
        "model_weight_bytes_per_card_excluded": next(iter(weights)),
        "hotness": {
            "card_count": len(first_cards),
            "minimum_write_bytes": min(
                write_bytes_by_device.values()),
            "mean_write_bytes": mean,
            "maximum_write_bytes": maximum,
            "population_stddev_write_bytes": stddev,
            "coefficient_of_variation": (
                stddev / mean if mean else None),
            "maximum_to_mean": (
                maximum / mean if mean else None),
            "hottest_card_share": (
                maximum / total_write_bytes
                if total_write_bytes else None),
            "hottest_device_ids": hottest_ids,
        },
        "scenarios": scenario_reports,
    }


__all__ = [
    "DAYS_PER_YEAR",
    "FLASHACCEL_ENDURANCE_SOURCE_URL",
    "HBFCardWriteSample",
    "HBFEnduranceError",
    "HBFEnduranceScenario",
    "HBFWriteSample",
    "HBF_ENDURANCE_SCHEMA_VERSION",
    "SECONDS_PER_DAY",
    "default_hbf_endurance_scenarios",
    "project_hbf_endurance",
]

"""Point-in-time Shenwan L1 industry histories from bounded provider rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

from a_share_research.contracts import ContractError, canonical_hash
from a_share_research.data.normalization import parse_provider_date


@dataclass(frozen=True)
class IndustryInterval:
    """One historical classification, usable only after its PIT availability date."""

    ts_code: str
    industry_id: str
    industry_name: str
    in_date: date
    out_date: date | None
    availability_date: date
    source_hash: str

    def __post_init__(self) -> None:
        if not self.ts_code or not self.industry_id:
            raise ContractError("industry interval identity cannot be empty")
        if self.availability_date <= self.in_date:
            raise ContractError("industry availability must be after in_date")
        if self.out_date is not None and self.out_date < self.in_date:
            raise ContractError("industry out_date precedes in_date")
        if len(self.source_hash) != 64:
            raise ContractError("industry source hash must be SHA-256")

    def active(self, day: date) -> bool:
        return self.availability_date <= day and (
            self.out_date is None or day <= self.out_date
        )


def _next_trading_day(day: date, trading_dates: tuple[date, ...]) -> date | None:
    return next((candidate for candidate in trading_dates if candidate > day), None)


def build_industry_intervals(
    *,
    rows: Iterable[Mapping[str, object]],
    trading_dates: tuple[date, ...],
    expected_code: str | None = None,
) -> tuple[IndustryInterval, ...]:
    """Normalize Y/N ``index_member_all`` rows without backward filling.

    The endpoint can return the same interval in the current and former-member
    queries. Exact duplicates are collapsed, while conflicting classifications
    with the same stock and ``in_date`` fail closed.
    """
    if tuple(sorted(set(trading_dates))) != trading_dates:
        raise ContractError("industry PIT calendar must be unique and sorted")
    normalized: dict[tuple[str, date, date | None, str], IndustryInterval] = {}
    classifications: dict[tuple[str, date], str] = {}
    for row in rows:
        code = str(row.get("ts_code", ""))
        if expected_code is not None and code != expected_code:
            raise ContractError("industry response contains an unexpected ts_code")
        industry_id = str(row.get("l1_code", ""))
        if not code or not industry_id or row.get("in_date") in (None, ""):
            raise ContractError("industry row lacks ts_code/l1_code/in_date")
        in_date = parse_provider_date(row["in_date"])
        out_value = row.get("out_date")
        out_date = (
            parse_provider_date(out_value) if out_value not in (None, "") else None
        )
        availability = _next_trading_day(in_date, trading_dates)
        # A classification entering after the audited calendar has no usable PIT
        # observation yet. It must not be projected backwards from current state.
        if availability is None:
            continue
        identity = (code, in_date)
        prior = classifications.get(identity)
        if prior is not None and prior != industry_id:
            raise ContractError("conflicting L1 classifications share one in_date")
        classifications[identity] = industry_id
        source_payload = {
            "endpoint": "index_member_all",
            "ts_code": code,
            "l1_code": industry_id,
            "l1_name": str(row.get("l1_name", "")),
            "in_date": in_date,
            "out_date": out_date,
        }
        interval = IndustryInterval(
            ts_code=code,
            industry_id=industry_id,
            industry_name=str(row.get("l1_name", "")),
            in_date=in_date,
            out_date=out_date,
            availability_date=availability,
            source_hash=canonical_hash(source_payload),
        )
        normalized[(code, in_date, out_date, industry_id)] = interval
    return tuple(
        sorted(
            normalized.values(),
            key=lambda item: (item.ts_code, item.in_date, item.industry_id),
        )
    )


def industry_at(
    intervals: Iterable[IndustryInterval], day: date
) -> IndustryInterval | None:
    """Return the latest classification actually available on ``day``."""
    candidates = tuple(interval for interval in intervals if interval.active(day))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.in_date, item.availability_date))


def industry_by_date(
    *,
    intervals: Iterable[IndustryInterval],
    trading_dates: tuple[date, ...],
    eligible_codes_by_date: Mapping[date, frozenset[str]],
) -> dict[date, dict[str, str]]:
    """Create a sparse historical map; absent evidence remains absent."""
    by_code: dict[str, list[IndustryInterval]] = defaultdict(list)
    for interval in intervals:
        by_code[interval.ts_code].append(interval)
    result: dict[date, dict[str, str]] = {}
    for day in trading_dates:
        known: dict[str, str] = {}
        for code in eligible_codes_by_date.get(day, frozenset()):
            interval = industry_at(by_code.get(code, ()), day)
            if interval is not None:
                known[code] = interval.industry_id
        result[day] = known
    return result


def numeric_industry_id(value: str) -> float:
    """Encode an official SW code without a learned/current crosswalk."""
    stem = value.split(".", 1)[0]
    if not stem.isdigit():
        raise ContractError("Shenwan L1 code is not numerically encodable")
    return float(stem)

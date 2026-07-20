"""Exact PIT availability, including conservative date-only disclosures."""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from a_share_research.contracts import ContractError

SHANGHAI = ZoneInfo("Asia/Shanghai")


def signal_cutoff(day: date) -> datetime:
    return datetime.combine(day, time(16, 0), SHANGHAI)


def _next_trade_day(day: date, trading_dates: tuple[date, ...]) -> date:
    later = tuple(candidate for candidate in trading_dates if candidate > day)
    if not later:
        raise ContractError("trading calendar does not cover next-day availability")
    return later[0]


def date_only_availability(announcement_date: date, trading_dates: tuple[date, ...]) -> datetime:
    """A date-only announcement is usable no earlier than the next trading day."""
    next_day = _next_trade_day(announcement_date, trading_dates)
    return datetime.combine(next_day, time(9, 0), SHANGHAI)


def exact_or_next_trade_availability(
    *,
    source_date: date,
    trading_dates: tuple[date, ...],
    exact_time: datetime | None,
) -> datetime:
    if exact_time is None:
        return date_only_availability(source_date, trading_dates)
    if exact_time.tzinfo is None or exact_time.utcoffset() is None:
        raise ContractError("exact availability must be timezone-aware")
    return exact_time.astimezone(SHANGHAI)


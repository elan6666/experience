"""Causal membership interval and daily-table builders."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from a_share_research.contracts import ContractError, UniverseMembership
from a_share_research.protocol import UniverseClass


@dataclass(frozen=True)
class MembershipInterval:
    ts_code: str
    effective_from: date
    effective_to: date | None
    source: str

    def __post_init__(self) -> None:
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ContractError("membership interval ends before it starts")


def build_dynamic_intervals(
    snapshots: dict[date, tuple[str, ...]],
    trading_dates: tuple[date, ...],
    *,
    source: str,
) -> tuple[MembershipInterval, ...]:
    """Convert dated official snapshots to intervals without latest-member backfill."""
    if not snapshots or not trading_dates:
        raise ContractError("historical snapshots and trading calendar are required")
    ordered_dates = tuple(sorted(set(trading_dates)))
    if ordered_dates != trading_dates:
        raise ContractError("trading dates must be unique and increasing")
    snapshot_dates = tuple(sorted(snapshots))
    if snapshot_dates[0] > trading_dates[0]:
        raise ContractError("membership history does not cover the requested start")
    intervals: list[MembershipInterval] = []
    for index, snapshot_date in enumerate(snapshot_dates):
        members = snapshots[snapshot_date]
        if not members or len(set(members)) != len(members):
            raise ContractError("official membership snapshot is empty or duplicated")
        later_dates = tuple(day for day in trading_dates if day > snapshot_date)
        next_snapshot = snapshot_dates[index + 1] if index + 1 < len(snapshot_dates) else None
        effective_to = None
        if next_snapshot is not None:
            previous_days = tuple(
                day for day in trading_dates if snapshot_date <= day < next_snapshot
            )
            if not previous_days:
                raise ContractError("snapshot dates do not map to a trading interval")
            effective_to = previous_days[-1]
        for ts_code in sorted(members):
            intervals.append(MembershipInterval(ts_code, snapshot_date, effective_to, source))
        if next_snapshot is None and not later_dates and snapshot_date != trading_dates[-1]:
            raise ContractError("final membership snapshot is outside the trading calendar")
    return tuple(intervals)


def daily_membership(
    intervals: tuple[MembershipInterval, ...],
    trading_dates: tuple[date, ...],
    universe: UniverseClass,
) -> tuple[UniverseMembership, ...]:
    rows: list[UniverseMembership] = []
    seen: set[tuple[date, str]] = set()
    for day in trading_dates:
        for interval in intervals:
            active = day >= interval.effective_from and (
                interval.effective_to is None or day <= interval.effective_to
            )
            if not active:
                continue
            key = (day, interval.ts_code)
            if key in seen:
                raise ContractError("overlapping membership intervals create duplicate keys")
            seen.add(key)
            rows.append(
                UniverseMembership(
                    asof_date=day,
                    ts_code=interval.ts_code,
                    universe=universe.value,
                    effective_from=interval.effective_from,
                    effective_to=interval.effective_to,
                    source=interval.source,
                )
            )
    return tuple(rows)


def static_selected_intervals(
    codes: tuple[str, ...],
    *,
    selection_date: date,
    source: str,
    research_start: date = date(2019, 1, 1),
) -> tuple[MembershipInterval, ...]:
    if selection_date.year != 2026:
        raise ContractError("technology list selection date must preserve its 2026 provenance")
    if not codes or len(set(codes)) != len(codes):
        raise ContractError("static technology list is empty or duplicated")
    if research_start > selection_date:
        raise ContractError("retrospective exploratory window starts after selection")
    selected_source = f"{source};selected={selection_date.isoformat()};retrospective_exploratory"
    return tuple(
        MembershipInterval(code, research_start, None, selected_source)
        for code in sorted(codes)
    )

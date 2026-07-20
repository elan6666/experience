from datetime import date

import pytest

from a_share_research.contracts import ContractError
from a_share_research.protocol import UniverseClass
from a_share_research.quality import ResultState
from a_share_research.universes.builders import (
    build_dynamic_intervals,
    daily_membership,
    static_selected_intervals,
)
from a_share_research.universes.specs import UniverseMode, UniverseSpec


def test_dynamic_membership_changes_without_future_member_backfill() -> None:
    days = tuple(date(2025, 1, day) for day in (2, 3, 6, 7))
    intervals = build_dynamic_intervals(
        {days[0]: ("000001.SZ",), days[2]: ("600000.SH",)},
        days,
        source="official-index-weight",
    )
    rows = daily_membership(intervals, days, UniverseClass.CSI300)
    by_day = {
        day: {row.ts_code for row in rows if row.asof_date == day}
        for day in days
    }
    assert by_day[days[0]] == {"000001.SZ"}
    assert by_day[days[1]] == {"000001.SZ"}
    assert by_day[days[2]] == {"600000.SH"}
    assert by_day[days[3]] == {"600000.SH"}


def test_dynamic_membership_refuses_uncovered_history() -> None:
    days = (date(2025, 1, 2), date(2025, 1, 3))
    with pytest.raises(ContractError, match="does not cover"):
        build_dynamic_intervals(
            {days[1]: ("000001.SZ",)},
            days,
            source="official-index-weight",
        )


def test_technology_lists_are_permanently_exploratory() -> None:
    spec = UniverseSpec(
        universe=UniverseClass.TECH100,
        mode=UniverseMode.STATIC_SELECTED_2026,
        benchmark_code=None,
        source="frozen-workbook",
        source_hash="a" * 64,
        selection_date=date(2026, 7, 17),
        formal_status=ResultState.EXPLORATORY_ONLY,
    )
    assert spec.formal_status is ResultState.EXPLORATORY_ONLY
    intervals = static_selected_intervals(
        ("000001.SZ", "600000.SH"),
        selection_date=date(2026, 7, 17),
        source="frozen-workbook",
    )
    assert len(intervals) == 2
    assert intervals[0].effective_from == date(2019, 1, 1)
    assert "retrospective_exploratory" in intervals[0].source

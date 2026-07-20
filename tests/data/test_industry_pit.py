from datetime import date

import pytest
from scripts.generate_d0_request_manifest import _industry_requests

from a_share_research.contracts import ContractError
from a_share_research.data.industry import (
    build_industry_intervals,
    industry_at,
    industry_by_date,
    numeric_industry_id,
)
from a_share_research.data.market_state import build_shared_market_state


def _calendar() -> tuple[date, ...]:
    return (
        date(2025, 1, 2),
        date(2025, 1, 3),
        date(2025, 1, 6),
        date(2025, 1, 7),
        date(2025, 1, 8),
    )


def test_manifest_emits_exact_y_and_n_partitions_per_stock() -> None:
    requests = _industry_requests("000001.SZ")
    assert {request.params["is_new"] for request in requests} == {"Y", "N"}
    assert all(request.params["ts_code"] == "000001.SZ" for request in requests)
    assert all(request.endpoint == "index_member_all" for request in requests)
    assert all(request.reject_at_row_count == 1000 for request in requests)
    assert {request.params["is_new"]: request.min_row_count for request in requests} == {
        "Y": 1,
        "N": 0,
    }
    assert requests[0].fields == (
        "l1_code",
        "l1_name",
        "ts_code",
        "in_date",
        "out_date",
        "is_new",
    )


def test_industry_is_unavailable_until_next_trading_day_and_never_backfilled() -> None:
    rows = (
        {
            "l1_code": "801010.SI",
            "l1_name": "农林牧渔",
            "ts_code": "000001.SZ",
            "in_date": "20250103",
            "out_date": None,
            "is_new": "Y",
        },
    )
    intervals = build_industry_intervals(
        rows=rows, trading_dates=_calendar(), expected_code="000001.SZ"
    )
    assert industry_at(intervals, date(2025, 1, 3)) is None
    available = industry_at(intervals, date(2025, 1, 6))
    assert available is not None
    assert available.availability_date == date(2025, 1, 6)
    mapping = industry_by_date(
        intervals=intervals,
        trading_dates=_calendar(),
        eligible_codes_by_date={
            day: frozenset(("000001.SZ",)) for day in _calendar()
        },
    )
    assert "000001.SZ" not in mapping[date(2025, 1, 3)]
    assert mapping[date(2025, 1, 6)]["000001.SZ"] == "801010.SI"
    assert numeric_industry_id("801010.SI") == 801010.0


def test_industry_y_n_duplicate_is_collapsed_but_conflict_fails_closed() -> None:
    base = {
        "l1_code": "801010.SI",
        "l1_name": "农林牧渔",
        "ts_code": "000001.SZ",
        "in_date": "20250102",
        "out_date": None,
    }
    intervals = build_industry_intervals(
        rows=({**base, "is_new": "Y"}, {**base, "is_new": "N"}),
        trading_dates=_calendar(),
    )
    assert len(intervals) == 1
    with pytest.raises(ContractError, match="conflicting L1"):
        build_industry_intervals(
            rows=(base, {**base, "l1_code": "801020.SI"}),
            trading_dates=_calendar(),
        )


def test_industry_dispersion_is_missing_below_pit_coverage_threshold() -> None:
    days = tuple(date(2025, 1, day) for day in range(1, 8))
    members = {
        day: frozenset(("000001.SZ", "600000.SH")) for day in days
    }
    values = {
        day: {"000001.SZ": 0.01, "600000.SH": -0.01} for day in days
    }
    state = build_shared_market_state(
        trading_dates=days,
        index_close={day: 100.0 + index for index, day in enumerate(days)},
        member_returns=values,
        member_amount={
            day: {"000001.SZ": 100.0, "600000.SH": 200.0} for day in days
        },
        member_turnover={
            day: {"000001.SZ": 1.0, "600000.SH": 2.0} for day in days
        },
        member_industry_by_date={day: {"000001.SZ": "801010.SI"} for day in days},
        eligible_member_codes_by_date=members,
        source_membership_hash="a" * 64,
        lookback=2,
        min_industry_coverage=0.8,
    )
    assert state.industry_coverage
    assert all(item.coverage == 0.5 and not item.sufficient for item in state.industry_coverage)
    assert "industry_dispersion" not in {row.feature_name for row in state.rows}

from datetime import date

from a_share_research.contracts import DailyMarket
from a_share_research.data.normalization import normalize_daily_market
from a_share_research.features.core import build_core_features


def _bar(day: date, close: float) -> DailyMarket:
    return DailyMarket(
        trade_date=day,
        ts_code="000001.SZ",
        open=close,
        high=close,
        low=close,
        close=close,
        volume=100.0,
        amount=1000.0,
        turnover=1.0,
        adj_factor=9.0,
        up_limit=close * 1.1,
        down_limit=close * 0.9,
        suspended=False,
        st_state=False,
    )


def test_normalization_preserves_raw_open_despite_adjustment_factor() -> None:
    market = normalize_daily_market(
        {
            "trade_date": "20250102",
            "ts_code": "000001.SZ",
            "open": 10,
            "high": 11,
            "low": 9,
            "close": 10.5,
            "vol": 100,
            "amount": 1000,
        },
        daily_basic={"turnover_rate": 1.0},
        adjustment={"adj_factor": 9.0},
        limits={"up_limit": 11.0, "down_limit": 9.0},
        suspended=False,
        st_state=False,
    )
    assert market.open == 10.0
    assert market.adj_factor == 9.0


def test_future_bar_perturbation_cannot_change_past_core_features() -> None:
    past_days = tuple(date(2025, 1, day) for day in range(1, 22))
    past = tuple(_bar(day, 10.0 + index) for index, day in enumerate(past_days))
    signal = past_days[-1]
    future_day = date(2025, 1, 22)
    first = build_core_features(past + (_bar(future_day, 31.0),), signal_date=signal)
    perturbed = build_core_features(past + (_bar(future_day, 999.0),), signal_date=signal)
    assert tuple(row.to_dict() for row in first) == tuple(row.to_dict() for row in perturbed)


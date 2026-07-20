"""Raw causal Core feature calculations over history available through signal close."""

from __future__ import annotations

import math
import statistics
from datetime import date

from a_share_research.contracts import ContractError, DailyMarket, PITFeature
from a_share_research.features.availability import signal_cutoff
from a_share_research.features.builders import build_feature_row
from a_share_research.features.schema import FeatureDefinition, d0_features


def _definition(name: str) -> FeatureDefinition:
    return next(item for item in d0_features() if item.name == name)


def build_core_features(
    history: tuple[DailyMarket, ...],
    *,
    signal_date: date,
) -> tuple[PITFeature, ...]:
    rows = tuple(
        sorted(
            (row for row in history if row.trade_date <= signal_date),
            key=lambda row: row.trade_date,
        )
    )
    if not rows or rows[-1].trade_date != signal_date:
        raise ContractError("Core history must end on the signal date")
    if len({row.ts_code for row in rows}) != 1:
        raise ContractError("Core history must contain one permanent asset identity")
    latest = rows[-1]
    cutoff = signal_cutoff(signal_date)
    values: dict[str, tuple[float | None, date]] = {
        "open": (latest.open, signal_date),
        "high": (latest.high, signal_date),
        "low": (latest.low, signal_date),
        "close": (latest.close, signal_date),
        "volume": (latest.volume, signal_date),
        "amount": (latest.amount, signal_date),
        "turnover_rate": (
            rows[-2].turnover if len(rows) >= 2 else None,
            rows[-2].trade_date if len(rows) >= 2 else signal_date,
        ),
    }
    closes = tuple(row.close for row in rows)
    amounts = tuple(row.amount for row in rows)
    for horizon in (1, 5, 20):
        name = f"return_{horizon}d"
        values[name] = (
            (
                math.log(closes[-1] / closes[-1 - horizon])
                if len(closes) > horizon and closes[-1 - horizon] > 0
                else None
            ),
            signal_date,
        )
    if len(closes) >= 21:
        returns = tuple(
            math.log(current / previous)
            for previous, current in zip(closes[-21:-1], closes[-20:], strict=True)
        )
        values["volatility_20d"] = (statistics.pstdev(returns), signal_date)
    else:
        values["volatility_20d"] = (None, signal_date)
    values["amount_mean_20d"] = (
        statistics.fmean(amounts[-20:]) if len(amounts) >= 20 else None,
        signal_date,
    )
    source_time = cutoff
    return tuple(
        build_feature_row(
            _definition(name),
            asof_date=signal_date,
            ts_code=latest.ts_code,
            value=value_and_date[0],
            source_date=value_and_date[1],
            announce_time=None,
            availability_time=source_time,
            signal_cutoff_time=cutoff,
            source="canonical_raw_market_v1",
        )
        for name, value_and_date in sorted(values.items())
    )

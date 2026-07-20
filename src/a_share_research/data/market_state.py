"""One CSI300-derived state table shared by every model and universe."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from a_share_research.contracts import ContractError, MarketState, canonical_hash


@dataclass(frozen=True)
class IndustryCoverage:
    asof_date: date
    active_count: int
    known_count: int
    coverage: float
    threshold: float
    sufficient: bool

    def __post_init__(self) -> None:
        if self.active_count <= 0 or not 0 <= self.known_count <= self.active_count:
            raise ContractError("industry coverage counts are invalid")
        expected = self.known_count / self.active_count
        if not math.isclose(self.coverage, expected, rel_tol=0, abs_tol=1e-12):
            raise ContractError("industry coverage ratio does not match counts")
        if not 0 < self.threshold <= 1:
            raise ContractError("industry coverage threshold must be in (0, 1]")
        if self.sufficient != (self.coverage >= self.threshold):
            raise ContractError("industry coverage status does not match threshold")


@dataclass(frozen=True)
class SharedMarketState:
    rows: tuple[MarketState, ...]
    source_membership_hash: str
    trading_calendar_hash: str
    industry_coverage: tuple[IndustryCoverage, ...] = ()

    def __post_init__(self) -> None:
        if not self.rows:
            raise ContractError("shared market state cannot be empty")
        if any(row.source_universe != "CSI300" for row in self.rows):
            raise ContractError("market state must be CSI300-only")
        for value in (self.source_membership_hash, self.trading_calendar_hash):
            if len(value) != 64:
                raise ContractError("shared market-state evidence must be SHA-256")

    @property
    def stable_hash(self) -> str:
        return canonical_hash(
            {
                "rows": self.rows,
                "source_membership_hash": self.source_membership_hash,
                "trading_calendar_hash": self.trading_calendar_hash,
                "industry_coverage": self.industry_coverage,
            }
        )


def assert_shared_market_state_hashes(hashes: tuple[str, ...]) -> str:
    """Fail if any model or universe references a different state table."""
    if not hashes or len(set(hashes)) != 1:
        raise ContractError("market-state hash must be identical across all consumers")
    value = hashes[0]
    if len(value) != 64:
        raise ContractError("market-state consumer hash must be SHA-256")
    return value


def _returns(values: tuple[float, ...]) -> tuple[float, ...]:
    if any(value <= 0 for value in values):
        raise ContractError("market-state prices must be positive")
    return tuple(math.log(current / previous) for previous, current in zip(values, values[1:]))


def build_shared_market_state(
    *,
    trading_dates: tuple[date, ...],
    index_close: Mapping[date, float],
    member_returns: Mapping[date, Mapping[str, float]],
    member_amount: Mapping[date, Mapping[str, float]],
    member_turnover: Mapping[date, Mapping[str, float]],
    member_industry_by_date: Mapping[date, Mapping[str, str]],
    eligible_member_codes_by_date: Mapping[date, frozenset[str]],
    source_membership_hash: str,
    lookback: int = 20,
    min_industry_coverage: float = 0.8,
) -> SharedMarketState:
    if lookback < 2 or tuple(sorted(set(trading_dates))) != trading_dates:
        raise ContractError("market-state calendar/lookback is invalid")
    if not 0 < min_industry_coverage <= 1:
        raise ContractError("industry coverage threshold must be in (0, 1]")
    calendar_hash = canonical_hash(trading_dates)
    rows: list[MarketState] = []
    coverage_rows: list[IndustryCoverage] = []
    for index in range(lookback, len(trading_dates)):
        day = trading_dates[index]
        window = trading_dates[index - lookback : index + 1]
        prices = tuple(index_close[candidate] for candidate in window if candidate in index_close)
        if len(prices) != lookback + 1:
            continue
        returns = _returns(prices)
        active = eligible_member_codes_by_date.get(day)
        if not active:
            continue
        today_industry = member_industry_by_date.get(day, {})
        known_industry = active.intersection(today_industry)
        coverage = len(known_industry) / len(active)
        sufficient = coverage >= min_industry_coverage
        coverage_rows.append(
            IndustryCoverage(
                day,
                len(active),
                len(known_industry),
                coverage,
                min_industry_coverage,
                sufficient,
            )
        )
        today_returns = {
            code: value for code, value in member_returns.get(day, {}).items() if code in active
        }
        today_amount = {
            code: value for code, value in member_amount.get(day, {}).items() if code in active
        }
        rolling_turnover = tuple(
            value
            for candidate in window[1:]
            for code, value in member_turnover.get(candidate, {}).items()
            if code in eligible_member_codes_by_date.get(candidate, frozenset())
        )
        rolling_amount = tuple(
            value
            for candidate in window[1:]
            for code, value in member_amount.get(candidate, {}).items()
            if code in eligible_member_codes_by_date.get(candidate, frozenset())
        )
        if not today_returns or not today_amount or not rolling_turnover or not rolling_amount:
            continue
        values = {
            "trend_20d": math.log(prices[-1] / prices[0]),
            "volatility_20d": statistics.pstdev(returns),
            "turnover_20d": statistics.fmean(rolling_turnover),
            "breadth": sum(value > 0 for value in today_returns.values()) / len(today_returns),
            "liquidity": math.log1p(statistics.fmean(rolling_amount)),
        }
        classified_returns = {
            code: value for code, value in today_returns.items() if code in today_industry
        }
        if sufficient and classified_returns:
            industry_values: dict[str, list[float]] = defaultdict(list)
            for code, value in classified_returns.items():
                industry_values[today_industry[code]].append(value)
            industry_means = tuple(
                statistics.fmean(industry_returns)
                for industry_returns in industry_values.values()
            )
            values["industry_dispersion"] = (
                statistics.pstdev(industry_means) if len(industry_means) > 1 else 0.0
            )
        day_source_hash = canonical_hash(
            {
                "day": day,
                "membership_hash": source_membership_hash,
                "active_members": tuple(sorted(active)),
                "member_returns": dict(today_returns),
                "member_amount": dict(today_amount),
                "member_turnover_rolling": rolling_turnover,
                "member_industry": dict(today_industry),
            }
        )
        rows.extend(
            MarketState(day, name, value, "CSI300", day_source_hash)
            for name, value in sorted(values.items())
        )
    return SharedMarketState(
        tuple(rows), source_membership_hash, calendar_hash, tuple(coverage_rows)
    )

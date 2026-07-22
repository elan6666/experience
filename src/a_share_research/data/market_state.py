"""One CSI300-derived state table shared by every model and universe."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from a_share_research.contracts import ContractError, MarketState, canonical_hash
from a_share_research.contracts.data import MARKET_STATE_SOURCE_UNIVERSES


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
        bad = {
            r.source_universe
            for r in self.rows
            if r.source_universe not in MARKET_STATE_SOURCE_UNIVERSES
        }
        if bad:
            raise ContractError(f"market-state source_universe not recognised: {bad}")
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


# A-share broad indices for enriched S-group market-state features.
# 000300.SH is excluded (already captured by csi300_* member-level features).
INDEX_FEATURE_INDICES: tuple[str, ...] = (
    "000001.SH",  # 上证综指
    "399001.SZ",  # 深证成指
    "000688.SH",  # 科创50
    "399006.SZ",  # 创业板指
    "000905.SH",  # 中证500
    "000852.SH",  # 中证1000
)

INDEX_FEATURE_NAMES: tuple[str, ...] = ("trend_20d", "volatility_20d", "return_5d")


def _index_feature_name(index_code: str, feature: str) -> str:
    """Map an index code to a schema feature name, e.g. 000001.SH -> idx_000001_trend_20d."""
    numeric = index_code.split(".")[0]
    return f"idx_{numeric}_{feature}"


def build_index_state_rows(
    *,
    trading_dates: tuple[date, ...],
    index_closes: Mapping[str, Mapping[date, float]],
    lookback: int = 20,
) -> tuple[MarketState, ...]:
    """Compute per-index trend/volatility/return rows for the enriched S group."""
    if lookback < 2:
        raise ContractError("index-state lookback must be >= 2")
    rows: list[MarketState] = []
    for index_code, closes in index_closes.items():
        if index_code not in MARKET_STATE_SOURCE_UNIVERSES:
            raise ContractError(f"index {index_code} is not a recognised market-state source")
        for pos in range(lookback, len(trading_dates)):
            day = trading_dates[pos]
            window = trading_dates[pos - lookback : pos + 1]
            prices = tuple(closes.get(d) for d in window)
            if any(pr is None or pr <= 0 for pr in prices):
                continue
            rets = _returns(prices)
            trend = math.log(prices[-1] / prices[0])
            vol = statistics.pstdev(rets) if len(rets) > 1 else 0.0
            # 5-day return (need at least 6 prices in the window tail)
            short_window = window[-6:] if len(window) >= 6 else None
            if short_window and all(closes.get(d) and closes[d] > 0 for d in short_window):
                ret5 = math.log(closes[short_window[-1]] / closes[short_window[0]])
            else:
                continue
            day_hash = canonical_hash({"day": day, "index": index_code, "close_window": prices})
            for feature, value in (
                (_index_feature_name(index_code, "trend_20d"), trend),
                (_index_feature_name(index_code, "volatility_20d"), vol),
                (_index_feature_name(index_code, "return_5d"), ret5),
            ):
                rows.append(MarketState(day, feature, value, index_code, day_hash))
    return tuple(rows)


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
    index_closes: Mapping[str, Mapping[date, float]] | None = None,
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
                statistics.fmean(industry_returns) for industry_returns in industry_values.values()
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
    if index_closes:
        rows.extend(
            build_index_state_rows(
                trading_dates=trading_dates,
                index_closes=index_closes,
                lookback=lookback,
            )
        )
    return SharedMarketState(
        tuple(rows), source_membership_hash, calendar_hash, tuple(coverage_rows)
    )

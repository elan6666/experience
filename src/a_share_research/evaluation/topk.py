"""Top-K realistic portfolio evaluation with A-share transaction costs.

Converts model predictions into weekly top-K long-only portfolios, applies
A-share transaction costs (commission, stamp duty, slippage), tracks turnover,
and computes net-of-cost performance metrics suitable for small-capital
(100K RMB) realistic comparison.

Three strategies:
  1. equal_weight  - Top-K equal-weight long-only (baseline)
  2. turnover_control - Hold positions unless rank drops below threshold
  3. kelly - Kelly criterion position sizing (non-equal weight)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from a_share_research.contracts import CoverageState, PredictionFrame
from a_share_research.data.labels import CompactLabel


@dataclass(frozen=True)
class TransactionCostModel:
    """A-share transaction cost model.

    Commission is charged on both buy and sell (0.025% per side).
    Stamp duty is charged on sell only (0.05%).
    Slippage is estimated per trade (0.05%).
    """

    commission_rate: float = 0.00025
    stamp_duty_rate: float = 0.0005
    slippage_rate: float = 0.0005

    @property
    def buy_cost_rate(self) -> float:
        return self.commission_rate + self.slippage_rate

    @property
    def sell_cost_rate(self) -> float:
        return self.commission_rate + self.stamp_duty_rate + self.slippage_rate

    def turnover_cost_rate(self, turnover_ratio: float) -> float:
        """Cost for a given turnover ratio (0=no change, 1=full rebalance)."""
        return turnover_ratio * (self.buy_cost_rate + self.sell_cost_rate)


@dataclass(frozen=True)
class TopKConfig:
    """Configuration for Top-K portfolio evaluation."""

    k: int
    capital: float = 100_000.0
    cost_model: TransactionCostModel = field(default_factory=TransactionCostModel)
    strategy: str = "equal_weight"
    max_weight: float = 0.20
    min_weight: float = 0.02
    turnover_hold_threshold: int = 0
    weeks_per_year: int = 52


@dataclass(frozen=True)
class WeeklyResult:
    """One week of Top-K portfolio performance."""

    signal_date: date
    holdings: tuple[str, ...]
    prev_holdings: tuple[str, ...]
    turnover: float
    gross_return: float
    cost: float
    net_return: float
    benchmark_return: float
    excess_return: float


@dataclass(frozen=True)
class TopKResult:
    """Aggregated Top-K evaluation result."""

    config: TopKConfig
    weekly: tuple[WeeklyResult, ...]
    benchmark_label: str

    @property
    def n_weeks(self) -> int:
        return len(self.weekly)

    @property
    def total_net_return(self) -> float:
        if not self.weekly:
            return 0.0
        cumulative = 1.0
        for w in self.weekly:
            cumulative *= 1.0 + w.net_return
        return cumulative - 1.0

    @property
    def total_gross_return(self) -> float:
        if not self.weekly:
            return 0.0
        cumulative = 1.0
        for w in self.weekly:
            cumulative *= 1.0 + w.gross_return
        return cumulative - 1.0

    @property
    def total_benchmark_return(self) -> float:
        if not self.weekly:
            return 0.0
        cumulative = 1.0
        for w in self.weekly:
            cumulative *= 1.0 + w.benchmark_return
        return cumulative - 1.0

    @property
    def annualized_return(self) -> float:
        if self.n_weeks < 2:
            return 0.0
        total = 1.0 + self.total_net_return
        return total ** (self.config.weeks_per_year / self.n_weeks) - 1.0

    @property
    def annualized_benchmark_return(self) -> float:
        if self.n_weeks < 2:
            return 0.0
        total = 1.0 + self.total_benchmark_return
        return total ** (self.config.weeks_per_year / self.n_weeks) - 1.0

    @property
    def net_weekly_returns(self) -> tuple[float, ...]:
        return tuple(w.net_return for w in self.weekly)

    @property
    def excess_weekly_returns(self) -> tuple[float, ...]:
        return tuple(w.excess_return for w in self.weekly)

    @property
    def sharpe_ratio(self) -> float:
        returns = self.net_weekly_returns
        if len(returns) < 2:
            return 0.0
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
        if std == 0:
            return 0.0
        annualized_mean = mean_ret * self.config.weeks_per_year
        annualized_std = std * math.sqrt(self.config.weeks_per_year)
        return annualized_mean / annualized_std

    @property
    def excess_sharpe_ratio(self) -> float:
        returns = self.excess_weekly_returns
        if len(returns) < 2:
            return 0.0
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
        if std == 0:
            return 0.0
        annualized_mean = mean_ret * self.config.weeks_per_year
        annualized_std = std * math.sqrt(self.config.weeks_per_year)
        return annualized_mean / annualized_std

    @property
    def max_drawdown(self) -> float:
        if not self.weekly:
            return 0.0
        peak = 1.0
        cumulative = 1.0
        max_dd = 0.0
        for w in self.weekly:
            cumulative *= 1.0 + w.net_return
            if cumulative > peak:
                peak = cumulative
            dd = (cumulative - peak) / peak
            if dd < max_dd:
                max_dd = dd
        return max_dd

    @property
    def win_rate(self) -> float:
        if not self.weekly:
            return 0.0
        wins = sum(1 for w in self.weekly if w.net_return > 0)
        return wins / len(self.weekly)

    @property
    def avg_weekly_turnover(self) -> float:
        if not self.weekly:
            return 0.0
        return sum(w.turnover for w in self.weekly) / len(self.weekly)

    @property
    def annual_turnover(self) -> float:
        return self.avg_weekly_turnover * self.config.weeks_per_year

    @property
    def avg_cost_per_week(self) -> float:
        if not self.weekly:
            return 0.0
        return sum(w.cost for w in self.weekly) / len(self.weekly)

    def to_summary(self) -> dict[str, object]:
        return {
            "strategy": self.config.strategy,
            "k": self.config.k,
            "capital": self.config.capital,
            "n_weeks": self.n_weeks,
            "total_net_return_pct": round(self.total_net_return * 100, 4),
            "total_gross_return_pct": round(self.total_gross_return * 100, 4),
            "total_benchmark_return_pct": round(self.total_benchmark_return * 100, 4),
            "annualized_return_pct": round(self.annualized_return * 100, 4),
            "annualized_benchmark_pct": round(self.annualized_benchmark_return * 100, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "excess_sharpe_ratio": round(self.excess_sharpe_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown * 100, 4),
            "win_rate_pct": round(self.win_rate * 100, 2),
            "avg_weekly_turnover_pct": round(self.avg_weekly_turnover * 100, 2),
            "annual_turnover_pct": round(self.annual_turnover * 100, 2),
            "avg_cost_per_week_pct": round(self.avg_cost_per_week * 100, 4),
        }


def _select_topk(
    scores: dict[str, float],
    k: int,
) -> tuple[str, ...]:
    """Select top-K stocks by prediction score (descending)."""
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return tuple(code for code, _ in ranked[:k])


def _compute_turnover(
    current: tuple[str, ...],
    previous: tuple[str, ...],
) -> float:
    """Compute turnover ratio: fraction of portfolio that changed."""
    if not current and not previous:
        return 0.0
    current_set = set(current)
    previous_set = set(previous)
    if not previous_set:
        return 1.0
    sold = len(previous_set - current_set)
    bought = len(current_set - previous_set)
    k = max(len(current), len(previous_set))
    return (sold + bought) / (2 * k) if k > 0 else 0.0


def _equal_weights(codes: tuple[str, ...]) -> dict[str, float]:
    if not codes:
        return {}
    w = 1.0 / len(codes)
    return {code: w for code in codes}


def _turnover_control_select(
    scores: dict[str, float],
    prev_holdings: tuple[str, ...],
    k: int,
    hold_threshold: int,
) -> tuple[str, ...]:
    """Select top-K with turnover control: keep prev holdings if still in top (k+threshold)."""
    if not prev_holdings:
        return _select_topk(scores, k)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    rank_map = {code: i for i, (code, _) in enumerate(ranked)}
    top_k_plus = set(code for code, _ in ranked[: k + hold_threshold])
    kept = tuple(code for code in prev_holdings if code in top_k_plus)
    needed = k - len(kept)
    if needed > 0:
        remaining = [code for code, _ in ranked if code not in set(kept)]
        kept = kept + tuple(remaining[:needed])
    return kept[:k]


def _kelly_weights(
    codes: tuple[str, ...],
    scores: dict[str, float],
    max_weight: float,
    min_weight: float,
) -> dict[str, float]:
    """Kelly-inspired position sizing based on prediction score magnitude."""
    if not codes:
        return {}
    raw = {code: max(scores.get(code, 0.0), 0.0) for code in codes}
    total = sum(raw.values())
    if total <= 0:
        return _equal_weights(codes)
    weights = {code: v / total for code, v in raw.items()}
    capped = {code: min(w, max_weight) for code, w in weights.items()}
    excess = sum(w - max_weight for w in weights.values() if w > max_weight)
    if excess > 0:
        eligible = [c for c in codes if capped[c] < max_weight]
        if eligible:
            redistribution = excess / len(eligible)
            for c in eligible:
                capped[c] = min(capped[c] + redistribution, max_weight)
    for code in capped:
        if capped[code] < min_weight:
            capped[code] = min_weight
    total_w = sum(capped.values())
    if total_w > 0:
        capped = {code: w / total_w for code, w in capped.items()}
    return capped


def evaluate_topk(
    frame: PredictionFrame,
    labels: dict[tuple[date, str], CompactLabel],
    config: TopKConfig,
    benchmark_fn: object | None = None,
) -> TopKResult:
    """Evaluate a Top-K portfolio strategy on prediction + label data.

    Parameters
    ----------
    frame : PredictionFrame
        Model predictions with scores per (signal_date, ts_code).
    labels : dict[(date, str), CompactLabel]
        Actual returns keyed by (signal_date, ts_code).
    config : TopKConfig
        Portfolio configuration (K, capital, costs, strategy).
    benchmark_fn : callable, optional
        Function(signal_date) -> float returning the benchmark return for that week.
        If None, uses equal-weight average of all SCORED stocks.

    Returns
    -------
    TopKResult
    """
    by_date: dict[date, dict[str, float]] = {}
    for record in frame.records:
        if record.coverage_state is not CoverageState.SCORED:
            continue
        if record.score is None:
            continue
        by_date.setdefault(record.signal_date, {})[record.ts_code] = record.score

    signal_dates = sorted(by_date.keys())
    weekly: list[WeeklyResult] = []
    prev_holdings: tuple[str, ...] = ()

    for signal_date in signal_dates:
        scores = by_date[signal_date]
        if not scores:
            continue

        if config.strategy == "turnover_control":
            holdings = _turnover_control_select(
                scores, prev_holdings, config.k, config.turnover_hold_threshold
            )
        elif config.strategy == "kelly":
            top = _select_topk(scores, config.k)
            holdings = top
        else:
            holdings = _select_topk(scores, config.k)

        if not holdings:
            continue

        if config.strategy == "kelly":
            weights = _kelly_weights(holdings, scores, config.max_weight, config.min_weight)
        else:
            weights = _equal_weights(holdings)

        gross_return = 0.0
        for code in holdings:
            label = labels.get((signal_date, code))
            if label is not None:
                gross_return += weights.get(code, 0.0) * label.open_to_open_return

        if benchmark_fn is not None:
            benchmark_return = float(benchmark_fn(signal_date))
        else:
            scored_returns = []
            for code in scores:
                label = labels.get((signal_date, code))
                if label is not None:
                    scored_returns.append(label.open_to_open_return)
            benchmark_return = (
                sum(scored_returns) / len(scored_returns) if scored_returns else 0.0
            )

        turnover = _compute_turnover(holdings, prev_holdings)
        cost = config.cost_model.turnover_cost_rate(turnover)
        net_return = gross_return - cost
        excess_return = net_return - benchmark_return

        weekly.append(
            WeeklyResult(
                signal_date=signal_date,
                holdings=holdings,
                prev_holdings=prev_holdings,
                turnover=turnover,
                gross_return=gross_return,
                cost=cost,
                net_return=net_return,
                benchmark_return=benchmark_return,
                excess_return=excess_return,
            )
        )
        prev_holdings = holdings

    return TopKResult(
        config=config,
        weekly=tuple(weekly),
        benchmark_label="equal_weight_universe" if benchmark_fn is None else "custom",
    )


def evaluate_multiple_k(
    frame: PredictionFrame,
    labels: dict[tuple[date, str], CompactLabel],
    k_values: Sequence[int],
    strategies: Sequence[str] = ("equal_weight",),
    capital: float = 100_000.0,
    benchmark_fn: object | None = None,
    turnover_hold_multiplier: int = 1,
) -> list[TopKResult]:
    """Evaluate multiple K values and strategies in one call.

    For turnover_control, the hold threshold is set to K * multiplier
    so stocks are kept while still ranked in the top K*(1+multiplier).
    """
    results: list[TopKResult] = []
    for strategy in strategies:
        for k in k_values:
            hold_threshold = k * turnover_hold_multiplier if strategy == "turnover_control" else 0
            config = TopKConfig(
                k=k,
                capital=capital,
                strategy=strategy,
                turnover_hold_threshold=hold_threshold,
            )
            results.append(evaluate_topk(frame, labels, config, benchmark_fn))
    return results

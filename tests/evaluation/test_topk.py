"""Tests for the Top-K realistic portfolio evaluation framework."""

from __future__ import annotations

from datetime import date, timedelta

from a_share_research.contracts import (
    CoverageState,
    PredictionFrame,
    PredictionRecord,
)
from a_share_research.data.labels import CompactLabel
from a_share_research.evaluation.topk import (
    TransactionCostModel,
    TopKConfig,
    _compute_turnover,
    _equal_weights,
    _kelly_weights,
    _select_topk,
    _turnover_control_select,
    evaluate_multiple_k,
    evaluate_topk,
)


def _make_label(sig: date, code: str, ret: float, week: int = 0) -> CompactLabel:
    return CompactLabel(
        signal_date=sig,
        ts_code=code,
        horizon=5,
        entry_date=sig + timedelta(days=1),
        exit_date=sig + timedelta(days=6),
        open_to_open_return=ret,
        benchmark_return=0.005,
        trading_calendar_hash="a" * 64,
        signal_index=week * 6,
        entry_index=week * 6 + 1,
        exit_index=week * 6 + 6,
    )


def _make_frame(stocks: list[str], scores: list[float], sig: date) -> PredictionFrame:
    records = tuple(
        PredictionRecord(
            signal_date=sig,
            ts_code=code,
            score=score,
            coverage_state=CoverageState.SCORED,
        )
        for code, score in zip(stocks, scores, strict=True)
    )
    return PredictionFrame(run_id=f"test-{sig.isoformat()}", records=records)


class TestTransactionCostModel:
    def test_default_rates(self):
        cm = TransactionCostModel()
        assert cm.buy_cost_rate == 0.00075
        assert cm.sell_cost_rate == 0.00125

    def test_turnover_cost(self):
        cm = TransactionCostModel()
        assert cm.turnover_cost_rate(0.0) == 0.0
        full = cm.turnover_cost_rate(1.0)
        assert abs(full - (0.00075 + 0.00125)) < 1e-10


class TestSelection:
    def test_select_topk(self):
        scores = {"A": 0.5, "B": 0.9, "C": 0.3, "D": 0.7}
        result = _select_topk(scores, 2)
        assert result == ("B", "D")

    def test_select_topk_fewer_than_k(self):
        scores = {"A": 0.5, "B": 0.9}
        result = _select_topk(scores, 5)
        assert result == ("B", "A")

    def test_turnover_control_keeps_prev(self):
        scores = {"A": 0.5, "B": 0.9, "C": 0.3, "D": 0.7, "E": 0.6}
        prev = ("A", "B")
        result = _turnover_control_select(scores, prev, 2, hold_threshold=2)
        assert set(result) == {"A", "B"}

    def test_turnover_control_drops_low_rank(self):
        scores = {"A": 0.5, "B": 0.9, "C": 0.3, "D": 0.7, "E": 0.6}
        prev = ("C", "A")
        result = _turnover_control_select(scores, prev, 2, hold_threshold=0)
        assert "C" not in result


class TestWeights:
    def test_equal_weights(self):
        w = _equal_weights(("A", "B", "C"))
        assert w == {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}

    def test_kelly_weights_respect_max(self):
        scores = {"A": 0.9, "B": 0.05, "C": 0.05}
        w = _kelly_weights(("A", "B", "C"), scores, max_weight=0.40, min_weight=0.05)
        assert w["A"] <= 0.40 + 1e-6

    def test_kelly_fallback_equal(self):
        scores = {"A": 0.0, "B": 0.0, "C": 0.0}
        w = _kelly_weights(("A", "B", "C"), scores, max_weight=0.40, min_weight=0.05)
        assert abs(w["A"] - 1 / 3) < 1e-6


class TestTurnover:
    def test_no_change(self):
        assert _compute_turnover(("A", "B"), ("A", "B")) == 0.0

    def test_full_change(self):
        assert _compute_turnover(("C", "D"), ("A", "B")) == 1.0

    def test_partial_change(self):
        t = _compute_turnover(("A", "C"), ("A", "B"))
        assert abs(t - 0.5) < 1e-6

    def test_empty_previous(self):
        assert _compute_turnover(("A", "B"), ()) == 1.0


class TestEvaluateTopK:
    def _make_data(self, n_weeks: int = 3):
        stocks = [f"{i:06d}.SZ" for i in range(1, 11)]
        all_records: list[PredictionRecord] = []
        labels: dict[tuple[date, str], CompactLabel] = {}
        for week in range(n_weeks):
            sig = date(2025, 1, 6) + timedelta(weeks=week)
            scores = [float(10 - i) for i in range(10)]
            frame = _make_frame(stocks, scores, sig)
            all_records.extend(frame.records)
            for i, code in enumerate(stocks):
                labels[(sig, code)] = _make_label(sig, code, 0.01 * (10 - i) / 10, week)
        combined = PredictionFrame(run_id="test-combined", records=tuple(all_records))
        return combined, labels

    def test_equal_weight_basic(self):
        frame, labels = self._make_data()
        config = TopKConfig(k=5, strategy="equal_weight")
        result = evaluate_topk(frame, labels, config)
        assert result.n_weeks == 3
        assert result.total_net_return > 0
        assert result.win_rate == 1.0
        assert result.avg_weekly_turnover > 0

    def test_cost_reduces_return(self):
        frame, labels = self._make_data()
        no_cost = TopKConfig(k=5, cost_model=TransactionCostModel(0, 0, 0))
        with_cost = TopKConfig(k=5)
        r_no = evaluate_topk(frame, labels, no_cost)
        r_with = evaluate_topk(frame, labels, with_cost)
        assert r_with.total_net_return < r_no.total_net_return

    def test_benchmark_return(self):
        frame, labels = self._make_data()
        config = TopKConfig(k=3)
        result = evaluate_topk(frame, labels, config)
        assert all(w.benchmark_return != 0 for w in result.weekly)

    def test_summary_dict(self):
        frame, labels = self._make_data()
        config = TopKConfig(k=5)
        result = evaluate_topk(frame, labels, config)
        s = result.to_summary()
        assert s["k"] == 5
        assert s["strategy"] == "equal_weight"
        assert "total_net_return_pct" in s
        assert "sharpe_ratio" in s

    def test_multiple_k(self):
        frame, labels = self._make_data()
        results = evaluate_multiple_k(frame, labels, [3, 5, 8])
        assert len(results) == 3
        assert all(r.n_weeks == 3 for r in results)

    def test_kelly_strategy(self):
        frame, labels = self._make_data()
        config = TopKConfig(k=5, strategy="kelly")
        result = evaluate_topk(frame, labels, config)
        assert result.n_weeks == 3
        assert result.total_net_return > 0

    def test_turnover_control_reduces_turnover(self):
        frame, labels = self._make_data(n_weeks=4)
        eq = TopKConfig(k=5, strategy="equal_weight")
        tc = TopKConfig(k=5, strategy="turnover_control", turnover_hold_threshold=3)
        r_eq = evaluate_topk(frame, labels, eq)
        r_tc = evaluate_topk(frame, labels, tc)
        assert r_tc.avg_weekly_turnover <= r_eq.avg_weekly_turnover

    def test_excess_return(self):
        frame, labels = self._make_data()
        config = TopKConfig(k=3)
        result = evaluate_topk(frame, labels, config)
        for w in result.weekly:
            assert abs(w.excess_return - (w.net_return - w.benchmark_return)) < 1e-10

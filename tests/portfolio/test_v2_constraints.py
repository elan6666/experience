"""Synthetic B2 constraints and execution evidence; server execution only."""

from datetime import date

import pytest

from a_share_research.contracts import (
    ContractError,
    EligibilityEvidence,
    ExecutionCalendarReceipt,
    eligibility_evidence_hash,
    execution_calendar_receipt_id,
)
from a_share_research.portfolio import (
    ConstraintContext,
    ConstraintPolicy,
    CostSchedule,
    ExecutionPrice,
    RiskBudgetPoint,
    RiskBudgetSchedule,
    TargetFrame,
    TargetWeight,
    TradingRestriction,
    build_b2_constrained_ledger,
    constrain_target_frame,
)

SOURCE_HASH = "a" * 64
REGISTRY_HASH = "b" * 64
MARKET_STATE_HASH = "c" * 64
POLICY_HASH = "d" * 64
CODE_A = "000001.SZ"
CODE_B = "600000.SH"


def _schedule(points: tuple[tuple[date, float], ...]) -> RiskBudgetSchedule:
    return RiskBudgetSchedule(
        market_state_hash=MARKET_STATE_HASH,
        policy_hash=POLICY_HASH,
        points=tuple(
            RiskBudgetPoint(day, float(index), budget, MARKET_STATE_HASH, POLICY_HASH)
            for index, (day, budget) in enumerate(points)
        ),
    )


def _context(
    signal_date: date,
    code: str,
    *,
    industry: str = "BANK",
    adv_value: float = 1000,
    reference_nav: float = 1000,
) -> ConstraintContext:
    return ConstraintContext(
        signal_date=signal_date,
        ts_code=code,
        industry=industry,
        adv_value=adv_value,
        reference_nav=reference_nav,
        source_hash="e" * 64,
    )


def _receipt(signal_date: date, trade_date: date) -> ExecutionCalendarReceipt:
    receipt_id = execution_calendar_receipt_id(
        signal_date=signal_date,
        next_trade_date=trade_date,
        calendar_source_hash=SOURCE_HASH,
    )
    return ExecutionCalendarReceipt(receipt_id, signal_date, trade_date, SOURCE_HASH)


def _evidence(
    receipt: ExecutionCalendarReceipt,
    *,
    buyable: bool = True,
    sellable: bool = True,
) -> EligibilityEvidence:
    evidence_id = eligibility_evidence_hash(
        signal_date=receipt.signal_date,
        trade_date=receipt.next_trade_date,
        ts_code=CODE_A,
        asset_registry_hash=REGISTRY_HASH,
        buyable=buyable,
        sellable=sellable,
        evidence_source_hash=SOURCE_HASH,
        trading_calendar_hash=SOURCE_HASH,
        next_trade_date=receipt.next_trade_date,
        calendar_receipt_id=receipt.receipt_id,
    )
    return EligibilityEvidence(
        evidence_id=evidence_id,
        signal_date=receipt.signal_date,
        trade_date=receipt.next_trade_date,
        ts_code=CODE_A,
        asset_registry_hash=REGISTRY_HASH,
        buyable=buyable,
        sellable=sellable,
        evidence_source_hash=SOURCE_HASH,
        trading_calendar_hash=SOURCE_HASH,
        next_trade_date=receipt.next_trade_date,
        calendar_receipt_id=receipt.receipt_id,
    )


def test_single_industry_adv_and_turnover_caps_are_deterministic() -> None:
    day = date(2025, 1, 2)
    schedule = _schedule(((day, 0.6),))
    b1 = TargetFrame(
        run_id="b1",
        targets=(TargetWeight(day, CODE_A, 0.3), TargetWeight(day, CODE_B, 0.3)),
        cash_weight_by_date={day.isoformat(): 0.4},
    )
    clipped, evidence = constrain_target_frame(
        b1_targets=b1,
        schedule=schedule,
        contexts=(_context(day, CODE_A), _context(day, CODE_B)),
        policy=ConstraintPolicy("caps", 0.25, 0.4, 1.0, 1.0),
        output_run_id="b2-caps",
    )
    assert tuple(target.weight for target in clipped.targets) == pytest.approx((0.2, 0.2))
    assert clipped.cash_weight_by_date[day.isoformat()] == pytest.approx(0.6)
    assert all("SINGLE_WEIGHT_CAP" in row.reasons for row in evidence.decisions)
    assert all("INDUSTRY_WEIGHT_CAP" in row.reasons for row in evidence.decisions)
    assert evidence.fallback_by_date[day.isoformat()] == "PROPORTIONAL_CLIP_TO_CASH"
    replay, replay_evidence = constrain_target_frame(
        b1_targets=b1,
        schedule=schedule,
        contexts=(_context(day, CODE_A), _context(day, CODE_B)),
        policy=ConstraintPolicy("caps", 0.25, 0.4, 1.0, 1.0),
        output_run_id="b2-caps",
    )
    assert clipped.stable_hash() == replay.stable_hash()
    assert evidence.stable_hash() == replay_evidence.stable_hash()


def test_adv_and_turnover_caps_limit_delta_without_rescoring() -> None:
    day = date(2025, 2, 3)
    schedule = _schedule(((day, 0.6),))
    b1 = TargetFrame(
        run_id="b1",
        targets=(TargetWeight(day, CODE_A, 0.6),),
        cash_weight_by_date={day.isoformat(): 0.4},
    )
    constrained, evidence = constrain_target_frame(
        b1_targets=b1,
        schedule=schedule,
        contexts=(_context(day, CODE_A, adv_value=200, reference_nav=1000),),
        policy=ConstraintPolicy("trade-caps", 1.0, 1.0, 0.5, 0.05),
        output_run_id="b2-trade-caps",
    )
    assert constrained.targets[0].weight == pytest.approx(0.05)
    assert evidence.decisions[0].reasons == ("ADV_CAPACITY_CAP", "TURNOVER_CAP")


def test_b2_reuses_t1_open_engine_and_exposes_suspension_cost_cash_and_rejects() -> None:
    signal1, trade1 = date(2025, 3, 3), date(2025, 3, 4)
    signal2, trade2 = date(2025, 3, 7), date(2025, 3, 10)
    schedule = _schedule(((signal1, 0.6), (signal2, 0.0)))
    b1 = TargetFrame(
        run_id="b1-execution",
        targets=(TargetWeight(signal1, CODE_A, 0.6),),
        cash_weight_by_date={signal1.isoformat(): 0.4, signal2.isoformat(): 1.0},
    )
    receipt1, receipt2 = _receipt(signal1, trade1), _receipt(signal2, trade2)
    result = build_b2_constrained_ledger(
        b1_targets=b1,
        schedule=schedule,
        contexts=(_context(signal1, CODE_A), _context(signal2, CODE_A)),
        policy=ConstraintPolicy("execution", 1.0, 1.0, 1.0, 1.0),
        output_run_id="b2-execution",
        initial_cash=1000,
        run_data_hash=SOURCE_HASH,
        asset_registry_hash=REGISTRY_HASH,
        calendar_receipts=(receipt1, receipt2),
        eligibility_evidence=(
            _evidence(receipt1),
            _evidence(receipt2, sellable=False),
        ),
        prices=(
            ExecutionPrice(trade1, CODE_A, 10, 10),
            ExecutionPrice(trade2, CODE_A, 11, 11),
        ),
        costs=CostSchedule("synthetic-cost", date(2025, 1, 1), 0.001, 0.001, 0.001),
        restrictions=(TradingRestriction(signal2, CODE_A, sell_reject_reason="SUSPENDED"),),
    )
    result.validate()
    assert result.ledger.fills[0].trade_date == trade1
    assert result.ledger.fills[0].price == 10
    assert result.execution_evidence.rows[0].total_cost > 0
    zero_budget_row = result.execution_evidence.rows[1]
    assert zero_budget_row.risk_budget == 0
    assert zero_budget_row.target_equity_weight == 0
    assert zero_budget_row.executed_equity_weight > 0
    assert zero_budget_row.reject_reasons == ("SUSPENDED",)
    assert zero_budget_row.closing_cash_weight < 1
    assert result.constraint_evidence.fallback_by_date[signal2.isoformat()] == (
        "RISK_ZERO_LIQUIDATE"
    )


def test_detailed_restriction_must_agree_with_authoritative_eligibility() -> None:
    signal, trade = date(2025, 4, 7), date(2025, 4, 8)
    schedule = _schedule(((signal, 0.6),))
    b1 = TargetFrame(
        run_id="b1-restriction",
        targets=(TargetWeight(signal, CODE_A, 0.6),),
        cash_weight_by_date={signal.isoformat(): 0.4},
    )
    receipt = _receipt(signal, trade)
    with pytest.raises(ContractError, match="buy restriction conflicts"):
        build_b2_constrained_ledger(
            b1_targets=b1,
            schedule=schedule,
            contexts=(_context(signal, CODE_A),),
            policy=ConstraintPolicy("restriction", 1.0, 1.0, 1.0, 1.0),
            output_run_id="b2-restriction",
            initial_cash=1000,
            run_data_hash=SOURCE_HASH,
            asset_registry_hash=REGISTRY_HASH,
            calendar_receipts=(receipt,),
            eligibility_evidence=(_evidence(receipt, buyable=True),),
            prices=(ExecutionPrice(trade, CODE_A, 10, 10),),
            costs=CostSchedule("synthetic-cost", date(2025, 1, 1), 0, 0, 0),
            restrictions=(
                TradingRestriction(signal, CODE_A, buy_reject_reason="LIMIT_UP"),
            ),
        )


def test_detailed_restriction_reasons_are_side_specific() -> None:
    with pytest.raises(ContractError, match="not registered for buys"):
        TradingRestriction(
            date(2025, 4, 7),
            CODE_A,
            buy_reject_reason="LIMIT_DOWN",
        )
    with pytest.raises(ContractError, match="not registered for sells"):
        TradingRestriction(
            date(2025, 4, 7),
            CODE_A,
            sell_reject_reason="LIMIT_UP",
        )

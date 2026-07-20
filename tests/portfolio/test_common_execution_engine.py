from datetime import date

import pytest

from a_share_research.contracts import (
    EligibilityEvidence,
    ExecutionCalendarReceipt,
    eligibility_evidence_hash,
    execution_calendar_receipt_id,
)
from a_share_research.portfolio import (
    CostSchedule,
    ExecutionPrice,
    TargetFrame,
    TargetWeight,
    build_b0_ledger,
)

SOURCE_HASH = "a" * 64
REGISTRY_HASH = "b" * 64
CODE = "000001.SZ"


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
        ts_code=CODE,
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
        ts_code=CODE,
        asset_registry_hash=REGISTRY_HASH,
        buyable=buyable,
        sellable=sellable,
        evidence_source_hash=SOURCE_HASH,
        trading_calendar_hash=SOURCE_HASH,
        next_trade_date=receipt.next_trade_date,
        calendar_receipt_id=receipt.receipt_id,
    )


def test_b0_executes_next_open_and_reconciles_cost_cash_nav_and_turnover() -> None:
    signal1, trade1 = date(2025, 1, 2), date(2025, 1, 3)
    signal2, trade2 = trade1, date(2025, 1, 6)
    receipt1, receipt2 = _receipt(signal1, trade1), _receipt(signal2, trade2)
    targets = TargetFrame(
        run_id="same-engine",
        targets=(TargetWeight(signal1, CODE, 1.0),),
        cash_weight_by_date={signal1.isoformat(): 0.0, signal2.isoformat(): 1.0},
    )
    ledger = build_b0_ledger(
        targets=targets,
        initial_cash=1000,
        run_data_hash=SOURCE_HASH,
        asset_registry_hash=REGISTRY_HASH,
        calendar_receipts=(receipt1, receipt2),
        eligibility_evidence=(_evidence(receipt1), _evidence(receipt2)),
        prices=(
            ExecutionPrice(trade1, CODE, 10.0, 11.0),
            ExecutionPrice(trade2, CODE, 12.0, 12.0),
        ),
        costs=CostSchedule("zero-cost-test", date(2025, 1, 1), 0, 0, 0),
    )
    assert ledger.fills[0].trade_date == trade1
    assert ledger.fills[0].price == 10.0
    assert ledger.nav(trade1) == pytest.approx(1100.0)
    assert ledger.nav(trade2) == pytest.approx(1200.0)
    assert ledger.turnover(trade1, previous_nav=1000) == pytest.approx(1.0)
    assert ledger.cash[-1].closing_cash == pytest.approx(1200.0)


def test_unsellable_position_is_carried_until_later_execution_date() -> None:
    signal1, trade1 = date(2025, 1, 2), date(2025, 1, 3)
    signal2, trade2 = trade1, date(2025, 1, 6)
    signal3, trade3 = trade2, date(2025, 1, 7)
    receipts = (
        _receipt(signal1, trade1),
        _receipt(signal2, trade2),
        _receipt(signal3, trade3),
    )
    targets = TargetFrame(
        run_id="sell-delay",
        targets=(TargetWeight(signal1, CODE, 1.0),),
        cash_weight_by_date={
            signal1.isoformat(): 0.0,
            signal2.isoformat(): 1.0,
            signal3.isoformat(): 1.0,
        },
    )
    ledger = build_b0_ledger(
        targets=targets,
        initial_cash=1000,
        run_data_hash=SOURCE_HASH,
        asset_registry_hash=REGISTRY_HASH,
        calendar_receipts=receipts,
        eligibility_evidence=(
            _evidence(receipts[0]),
            _evidence(receipts[1], sellable=False),
            _evidence(receipts[2]),
        ),
        prices=tuple(
            ExecutionPrice(trade_date, CODE, price, price)
            for trade_date, price in ((trade1, 10.0), (trade2, 11.0), (trade3, 12.0))
        ),
        costs=CostSchedule("zero-cost-test", date(2025, 1, 1), 0, 0, 0),
    )
    rejected = [fill for fill in ledger.fills if fill.reject_reason == "NOT_SELLABLE"]
    assert len(rejected) == 1
    assert any(holding.trade_date == trade2 for holding in ledger.holdings)
    assert not any(holding.trade_date == trade3 for holding in ledger.holdings)
    assert ledger.cash[-1].closing_cash == pytest.approx(1200.0)

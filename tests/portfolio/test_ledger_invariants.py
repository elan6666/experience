"""Hand-calculated portfolio accounting; execute only on the approved server."""

from dataclasses import replace
from datetime import date, datetime, timezone

import pytest

from a_share_research.contracts import (
    CashSnapshot,
    ContractError,
    EligibilityEvidence,
    ExecutionCalendarReceipt,
    FillSide,
    HoldingSnapshot,
    PortfolioFill,
    PortfolioLedger,
    RunManifest,
    assert_ledger_matches_run,
    eligibility_evidence_hash,
    execution_calendar_manifest_hash,
    execution_calendar_receipt_id,
)
from a_share_research.protocol import Partition, Purpose, UniverseClass
from a_share_research.quality import ResultState, is_candidate_state

REGISTRY_HASH = "a" * 64
SOURCE_HASH = "b" * 64


def _evidence(
    *,
    signal_date: date,
    trade_date: date,
    buyable: bool = True,
    sellable: bool = True,
) -> EligibilityEvidence:
    calendar_receipt_id = execution_calendar_receipt_id(
        signal_date=signal_date,
        next_trade_date=trade_date,
        calendar_source_hash=SOURCE_HASH,
    )
    evidence_id = eligibility_evidence_hash(
        signal_date=signal_date,
        trade_date=trade_date,
        ts_code="000001.SZ",
        asset_registry_hash=REGISTRY_HASH,
        buyable=buyable,
        sellable=sellable,
        evidence_source_hash=SOURCE_HASH,
        trading_calendar_hash=SOURCE_HASH,
        next_trade_date=trade_date,
        calendar_receipt_id=calendar_receipt_id,
    )
    return EligibilityEvidence(
        evidence_id=evidence_id,
        signal_date=signal_date,
        trade_date=trade_date,
        ts_code="000001.SZ",
        asset_registry_hash=REGISTRY_HASH,
        buyable=buyable,
        sellable=sellable,
        evidence_source_hash=SOURCE_HASH,
        trading_calendar_hash=SOURCE_HASH,
        next_trade_date=trade_date,
        calendar_receipt_id=calendar_receipt_id,
    )


def _calendar_receipt(evidence: EligibilityEvidence) -> ExecutionCalendarReceipt:
    return ExecutionCalendarReceipt(
        receipt_id=evidence.calendar_receipt_id,
        signal_date=evidence.signal_date,
        next_trade_date=evidence.next_trade_date,
        calendar_source_hash=evidence.trading_calendar_hash,
    )


def _calendar_manifest(*evidence: EligibilityEvidence) -> str:
    return execution_calendar_manifest_hash(tuple(_calendar_receipt(item) for item in evidence))


def _fill(
    *,
    evidence: EligibilityEvidence,
    trade_date: date,
    side: FillSide,
    quantity: float,
    price: float,
    commission: float = 0,
    tax: float = 0,
    slippage: float = 0,
    reject_reason: str | None = None,
) -> PortfolioFill:
    return PortfolioFill(
        trade_date=trade_date,
        ts_code="000001.SZ",
        side=side,
        quantity=quantity,
        price=price,
        commission=commission,
        tax=tax,
        slippage=slippage,
        eligibility_evidence_id=evidence.evidence_id,
        reject_reason=reject_reason,
    )


def _valid_ledger() -> PortfolioLedger:
    day1 = date(2025, 1, 2)
    day2 = date(2025, 1, 3)
    evidence1 = _evidence(signal_date=date(2025, 1, 1), trade_date=day1)
    evidence2 = _evidence(signal_date=day1, trade_date=day2)
    return PortfolioLedger(
        run_id="synthetic-ledger",
        initial_cash=1000.0,
        run_data_hash=SOURCE_HASH,
        asset_registry_hash=REGISTRY_HASH,
        eligibility_source_hash=SOURCE_HASH,
        execution_calendar_manifest_hash=_calendar_manifest(evidence1, evidence2),
        execution_calendar_receipts=(_calendar_receipt(evidence1), _calendar_receipt(evidence2)),
        eligibility_evidence=(evidence1, evidence2),
        cash=(
            CashSnapshot(day1, opening_cash=1000.0, closing_cash=498.0),
            CashSnapshot(day2, opening_cash=498.0, closing_cash=1045.5),
        ),
        fills=(
            _fill(
                trade_date=day1,
                evidence=evidence1,
                side=FillSide.BUY,
                quantity=10,
                price=50,
                commission=1,
                tax=0,
                slippage=1,
            ),
            _fill(
                trade_date=day2,
                evidence=evidence2,
                side=FillSide.SELL,
                quantity=10,
                price=55,
                commission=1,
                tax=0.5,
                slippage=1,
            ),
        ),
        holdings=(
            HoldingSnapshot(day1, "000001.SZ", 10, 52, 0.5, 0.5),
        ),
    )


def test_hand_calculated_cash_nav_turnover_and_round_trip() -> None:
    ledger = _valid_ledger()
    ledger.validate()
    assert ledger.nav(date(2025, 1, 2)) == 1018.0
    assert ledger.nav(date(2025, 1, 3)) == 1045.5
    assert ledger.turnover(date(2025, 1, 2), previous_nav=1000.0) == 0.5
    assert PortfolioLedger.from_dict(ledger.to_dict()) == ledger


def test_invalid_cash_reconciliation_fails_loudly() -> None:
    ledger = _valid_ledger()
    with pytest.raises(ContractError, match="cash reconciliation"):
        PortfolioLedger(
            run_id=ledger.run_id,
            initial_cash=ledger.initial_cash,
            run_data_hash=SOURCE_HASH,
            asset_registry_hash=REGISTRY_HASH,
            eligibility_source_hash=SOURCE_HASH,
            execution_calendar_manifest_hash=_calendar_manifest(
                ledger.eligibility_evidence[0]
            ),
            execution_calendar_receipts=(_calendar_receipt(ledger.eligibility_evidence[0]),),
            eligibility_evidence=(ledger.eligibility_evidence[0],),
            cash=(CashSnapshot(date(2025, 1, 2), 1000.0, 600.0),),
            fills=(ledger.fills[0],),
            holdings=ledger.holdings,
        )


def test_rejected_limit_or_suspension_order_cannot_change_accounting() -> None:
    evidence = _evidence(
        signal_date=date(2025, 1, 2),
        trade_date=date(2025, 1, 3),
        buyable=False,
        sellable=False,
    )
    with pytest.raises(ContractError, match="rejected order"):
        _fill(
            trade_date=date(2025, 1, 3),
            evidence=evidence,
            side=FillSide.BUY,
            quantity=100,
            price=0,
            commission=0,
            tax=0,
            slippage=0,
            reject_reason="SUSPENDED",
        )


def test_valid_negative_is_not_misclassified_as_pipeline_failure() -> None:
    assert is_candidate_state(
        ResultState.VALID_NEGATIVE,
        partition=Partition.VALIDATION,
        universe=UniverseClass.CSI300,
    )
    assert not is_candidate_state(
        ResultState.TRAIN_FAIL,
        partition=Partition.VALIDATION,
        universe=UniverseClass.CSI300,
    )


def test_executed_fill_without_side_eligibility_fails() -> None:
    evidence = _evidence(
        signal_date=date(2025, 1, 2),
        trade_date=date(2025, 1, 3),
        buyable=False,
    )
    fill = _fill(
        evidence=evidence,
        trade_date=evidence.trade_date,
        side=FillSide.BUY,
        quantity=1,
        price=10,
    )
    with pytest.raises(ContractError, match="not buyable"):
        PortfolioLedger(
            run_id="unbuyable",
            initial_cash=1000,
            run_data_hash=SOURCE_HASH,
            asset_registry_hash=REGISTRY_HASH,
            eligibility_source_hash=SOURCE_HASH,
            execution_calendar_manifest_hash=_calendar_manifest(evidence),
            execution_calendar_receipts=(_calendar_receipt(evidence),),
            eligibility_evidence=(evidence,),
            cash=(CashSnapshot(evidence.trade_date, 1000, 990),),
            fills=(fill,),
            holdings=(HoldingSnapshot(evidence.trade_date, "000001.SZ", 1, 10, 0.01, 0.01),),
        )


def test_holding_quantity_must_reconcile_and_duplicates_fail() -> None:
    valid = _valid_ledger()
    day1 = date(2025, 1, 2)
    with pytest.raises(ContractError, match="does not reconcile"):
        PortfolioLedger(
            run_id="bad-quantity",
            initial_cash=1000,
            run_data_hash=SOURCE_HASH,
            asset_registry_hash=REGISTRY_HASH,
            eligibility_source_hash=SOURCE_HASH,
            execution_calendar_manifest_hash=_calendar_manifest(
                valid.eligibility_evidence[0]
            ),
            execution_calendar_receipts=(
                _calendar_receipt(valid.eligibility_evidence[0]),
            ),
            eligibility_evidence=(valid.eligibility_evidence[0],),
            cash=(valid.cash[0],),
            fills=(valid.fills[0],),
            holdings=(HoldingSnapshot(day1, "000001.SZ", 9, 52, 0.5, 0.5),),
        )
    duplicate = HoldingSnapshot(day1, "000001.SZ", 10, 52, 0.5, 0.5)
    with pytest.raises(ContractError, match="duplicate holding"):
        PortfolioLedger(
            run_id="duplicate",
            initial_cash=1000,
            run_data_hash=SOURCE_HASH,
            asset_registry_hash=REGISTRY_HASH,
            eligibility_source_hash=SOURCE_HASH,
            execution_calendar_manifest_hash=_calendar_manifest(
                valid.eligibility_evidence[0]
            ),
            execution_calendar_receipts=(
                _calendar_receipt(valid.eligibility_evidence[0]),
            ),
            eligibility_evidence=(valid.eligibility_evidence[0],),
            cash=(valid.cash[0],),
            fills=(valid.fills[0],),
            holdings=(duplicate, duplicate),
        )


def test_fill_must_reference_trusted_ledger_evidence() -> None:
    valid = _valid_ledger()
    with pytest.raises(ContractError, match="missing eligibility evidence"):
        PortfolioLedger(
            run_id="missing-evidence",
            initial_cash=1000,
            run_data_hash=SOURCE_HASH,
            asset_registry_hash=REGISTRY_HASH,
            eligibility_source_hash=SOURCE_HASH,
            execution_calendar_manifest_hash=_calendar_manifest(
                valid.eligibility_evidence[0]
            ),
            execution_calendar_receipts=(
                _calendar_receipt(valid.eligibility_evidence[0]),
            ),
            eligibility_evidence=(),
            cash=(valid.cash[0],),
            fills=(valid.fills[0],),
            holdings=(valid.holdings[0],),
        )


def test_same_day_buy_cannot_fund_a_sell_under_a_share_t_plus_one() -> None:
    trade_date = date(2025, 1, 2)
    evidence = _evidence(signal_date=date(2025, 1, 1), trade_date=trade_date)
    buy = _fill(
        evidence=evidence,
        trade_date=trade_date,
        side=FillSide.BUY,
        quantity=10,
        price=10,
    )
    sell = _fill(
        evidence=evidence,
        trade_date=trade_date,
        side=FillSide.SELL,
        quantity=10,
        price=10,
    )
    with pytest.raises(ContractError, match="opening carried quantity"):
        PortfolioLedger(
            run_id="same-day-round-trip",
            initial_cash=1000,
            run_data_hash=SOURCE_HASH,
            asset_registry_hash=REGISTRY_HASH,
            eligibility_source_hash=SOURCE_HASH,
            execution_calendar_manifest_hash=_calendar_manifest(evidence),
            execution_calendar_receipts=(_calendar_receipt(evidence),),
            eligibility_evidence=(evidence,),
            cash=(CashSnapshot(trade_date, 1000, 1000),),
            fills=(buy, sell),
            holdings=(),
        )


def test_candidate_state_rejects_missing_untyped_context() -> None:
    with pytest.raises(TypeError, match="typed Partition"):
        is_candidate_state(
            ResultState.PASS,
            partition="VALIDATION",
            universe="CSI300",
        )


def test_eligibility_booleans_and_fill_side_are_strongly_typed() -> None:
    with pytest.raises(ContractError, match="must be bool"):
        _evidence(
            signal_date=date(2025, 1, 1),
            trade_date=date(2025, 1, 2),
            buyable=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ContractError, match="FillSide"):
        PortfolioFill(
            trade_date=date(2025, 1, 2),
            ts_code="000001.SZ",
            side="BUY",  # type: ignore[arg-type]
            quantity=1,
            price=10,
            commission=0,
            tax=0,
            slippage=0,
            eligibility_evidence_id="a" * 64,
        )


def test_formal_ledger_entry_cross_checks_run_data_asset_and_calendar_hashes() -> None:
    ledger = _valid_ledger()
    manifest = RunManifest(
        run_id=ledger.run_id,
        model="Ridge",
        universe=UniverseClass.CSI300,
        information_set="A0",
        split=Partition.VALIDATION,
        purpose=Purpose.SELECT,
        data_hash=ledger.run_data_hash,
        asset_registry_hash=ledger.asset_registry_hash,
        execution_calendar_manifest_hash=ledger.execution_calendar_manifest_hash,
        feature_schema_hash="c" * 64,
        market_state_hash="c" * 64,
        config_hash="c" * 64,
        code_hash="c" * 64,
        upstream_commit="internal-ridge-v1",
        seed=7,
        status=ResultState.PASS,
        started_at=datetime(2026, 7, 19, 8, tzinfo=timezone.utc),
        completed_at=None,
        formal_feature_manifest_hash="c" * 64,
    )
    assert_ledger_matches_run(ledger, manifest)
    with pytest.raises(ContractError, match="asset registry"):
        assert_ledger_matches_run(
            ledger,
            replace(manifest, asset_registry_hash="d" * 64),
        )

"""Deterministic B0 execution using the canonical PortfolioLedger."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import ClassVar

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
    execution_calendar_manifest_hash,
)
from a_share_research.contracts.base import CanonicalModel, require_nonnegative
from a_share_research.contracts.data import _validate_ts_code

from .intents import TargetFrame


@dataclass(frozen=True)
class CostSchedule(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "cost_schedule"

    version: str
    effective_date: date
    commission_rate: float
    sell_tax_rate: float
    slippage_rate: float

    def validate(self) -> None:
        if not self.version:
            raise ContractError("cost schedule version is required")
        for name in ("commission_rate", "sell_tax_rate", "slippage_rate"):
            require_nonnegative(getattr(self, name), name)
            if getattr(self, name) >= 1:
                raise ContractError(f"{name} must be less than one")


@dataclass(frozen=True)
class ExecutionPrice(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "execution_price"

    trade_date: date
    ts_code: str
    open_price: float
    close_price: float

    def validate(self) -> None:
        _validate_ts_code(self.ts_code)
        if self.open_price <= 0 or self.close_price <= 0:
            raise ContractError("execution open and close prices must be positive")


def _costs(
    gross_value: float, side: FillSide, schedule: CostSchedule
) -> tuple[float, float, float]:
    commission = gross_value * schedule.commission_rate
    tax = gross_value * schedule.sell_tax_rate if side is FillSide.SELL else 0.0
    slippage = gross_value * schedule.slippage_rate
    return commission, tax, slippage


def build_b0_ledger(
    *,
    targets: TargetFrame,
    initial_cash: float,
    run_data_hash: str,
    asset_registry_hash: str,
    calendar_receipts: tuple[ExecutionCalendarReceipt, ...],
    eligibility_evidence: tuple[EligibilityEvidence, ...],
    prices: tuple[ExecutionPrice, ...],
    costs: CostSchedule,
) -> PortfolioLedger:
    """Execute frozen targets at the trusted next open under A-share T+1.

    The engine has no model branch. A model, momentum baseline and eligible
    equal-weight baseline all enter through the same TargetFrame.
    """
    targets.validate()
    costs.validate()
    require_nonnegative(initial_cash, "initial cash")
    if initial_cash <= 0:
        raise ContractError("initial cash must be positive")
    receipt_by_signal: dict[date, ExecutionCalendarReceipt] = {}
    for receipt in calendar_receipts:
        receipt.validate()
        if receipt.signal_date in receipt_by_signal:
            raise ContractError("multiple execution dates for one signal date")
        if receipt.calendar_source_hash != run_data_hash:
            raise ContractError("execution calendar is not anchored to run data")
        receipt_by_signal[receipt.signal_date] = receipt
    evidence_by_key: dict[tuple[date, str], EligibilityEvidence] = {}
    for evidence in eligibility_evidence:
        evidence.validate()
        key = (evidence.signal_date, evidence.ts_code)
        if key in evidence_by_key:
            raise ContractError("duplicate execution evidence key")
        evidence_by_key[key] = evidence
    price_by_key: dict[tuple[date, str], ExecutionPrice] = {}
    for price in prices:
        price.validate()
        key = (price.trade_date, price.ts_code)
        if key in price_by_key:
            raise ContractError("duplicate execution price key")
        price_by_key[key] = price
    target_by_date: dict[date, dict[str, float]] = defaultdict(dict)
    for target in targets.targets:
        target_by_date[target.signal_date][target.ts_code] = target.weight
    signal_dates = tuple(date.fromisoformat(value) for value in targets.cash_weight_by_date)
    if tuple(sorted(signal_dates)) != signal_dates:
        raise ContractError("target signal dates must be increasing")
    trade_dates: set[date] = set()
    cash_value = initial_cash
    quantities: dict[str, float] = {}
    cash_snapshots: list[CashSnapshot] = []
    fills: list[PortfolioFill] = []
    holdings: list[HoldingSnapshot] = []
    used_receipts: list[ExecutionCalendarReceipt] = []
    used_evidence: dict[str, EligibilityEvidence] = {}
    for signal_date in signal_dates:
        receipt = receipt_by_signal.get(signal_date)
        if receipt is None:
            raise ContractError("target signal lacks trusted next-trading-day receipt")
        trade_date = receipt.next_trade_date
        if trade_date in trade_dates:
            raise ContractError("two signal dates cannot rebalance on the same trade date")
        if trade_date < costs.effective_date:
            raise ContractError("cost schedule is not effective on execution date")
        trade_dates.add(trade_date)
        used_receipts.append(receipt)
        opening_cash = cash_value
        intended = target_by_date.get(signal_date, {})
        relevant_codes = set(quantities) | set(intended)
        for code in relevant_codes:
            if (trade_date, code) not in price_by_key:
                raise ContractError("held or targeted security lacks execution-day prices")
            evidence = evidence_by_key.get((signal_date, code))
            if evidence is None:
                raise ContractError("held or targeted security lacks execution evidence")
            if evidence.calendar_receipt_id != receipt.receipt_id:
                raise ContractError("execution evidence references another calendar receipt")
            used_evidence[evidence.evidence_id] = evidence
        pretrade_nav = cash_value + sum(
            quantity * price_by_key[(trade_date, code)].open_price
            for code, quantity in quantities.items()
        )
        # Sell first, but never more than the opening carried quantity. This is
        # the engine-level T+1 rule; the ledger independently checks it again.
        for code in sorted(quantities):
            price = price_by_key[(trade_date, code)]
            evidence = evidence_by_key[(signal_date, code)]
            desired_quantity = pretrade_nav * intended.get(code, 0.0) / price.open_price
            sell_quantity = max(0.0, quantities[code] - desired_quantity)
            if sell_quantity <= 1e-12:
                continue
            if not evidence.sellable:
                fills.append(
                    PortfolioFill(
                        trade_date=trade_date,
                        ts_code=code,
                        side=FillSide.SELL,
                        quantity=0.0,
                        price=price.open_price,
                        commission=0.0,
                        tax=0.0,
                        slippage=0.0,
                        eligibility_evidence_id=evidence.evidence_id,
                        reject_reason="NOT_SELLABLE",
                    )
                )
                continue
            gross = sell_quantity * price.open_price
            commission, tax, slippage = _costs(gross, FillSide.SELL, costs)
            fills.append(
                PortfolioFill(
                    trade_date=trade_date,
                    ts_code=code,
                    side=FillSide.SELL,
                    quantity=sell_quantity,
                    price=price.open_price,
                    commission=commission,
                    tax=tax,
                    slippage=slippage,
                    eligibility_evidence_id=evidence.evidence_id,
                )
            )
            quantities[code] -= sell_quantity
            cash_value += gross - commission - tax - slippage
        # Scale all desired buys by one common cash factor; code order cannot
        # silently allocate all residual cash to the first security.
        buy_needs: dict[str, float] = {}
        for code, weight in intended.items():
            price = price_by_key[(trade_date, code)]
            desired_quantity = pretrade_nav * weight / price.open_price
            buy_needs[code] = max(0.0, desired_quantity - quantities.get(code, 0.0))
        spend_rate = 1 + costs.commission_rate + costs.slippage_rate
        desired_spend = sum(
            quantity * price_by_key[(trade_date, code)].open_price * spend_rate
            for code, quantity in buy_needs.items()
            if evidence_by_key[(signal_date, code)].buyable
        )
        scale = min(1.0, cash_value / desired_spend) if desired_spend > 0 else 1.0
        for code in sorted(buy_needs):
            quantity = buy_needs[code]
            if quantity <= 1e-12:
                continue
            price = price_by_key[(trade_date, code)]
            evidence = evidence_by_key[(signal_date, code)]
            if not evidence.buyable:
                fills.append(
                    PortfolioFill(
                        trade_date=trade_date,
                        ts_code=code,
                        side=FillSide.BUY,
                        quantity=0.0,
                        price=price.open_price,
                        commission=0.0,
                        tax=0.0,
                        slippage=0.0,
                        eligibility_evidence_id=evidence.evidence_id,
                        reject_reason="NOT_BUYABLE",
                    )
                )
                continue
            quantity *= scale
            gross = quantity * price.open_price
            commission, tax, slippage = _costs(gross, FillSide.BUY, costs)
            fills.append(
                PortfolioFill(
                    trade_date=trade_date,
                    ts_code=code,
                    side=FillSide.BUY,
                    quantity=quantity,
                    price=price.open_price,
                    commission=commission,
                    tax=tax,
                    slippage=slippage,
                    eligibility_evidence_id=evidence.evidence_id,
                )
            )
            cash_value -= gross + commission + slippage
            quantities[code] = quantities.get(code, 0.0) + quantity
        if cash_value < -1e-7:
            raise ContractError("execution overspent available cash")
        if cash_value < 0:
            cash_value = 0.0
        quantities = {
            code: quantity for code, quantity in quantities.items() if quantity > 1e-10
        }
        close_value = sum(
            quantity * price_by_key[(trade_date, code)].close_price
            for code, quantity in quantities.items()
        )
        closing_nav = cash_value + close_value
        cash_snapshots.append(CashSnapshot(trade_date, opening_cash, cash_value))
        for code, quantity in sorted(quantities.items()):
            close_price = price_by_key[(trade_date, code)].close_price
            holdings.append(
                HoldingSnapshot(
                    trade_date=trade_date,
                    ts_code=code,
                    quantity=quantity,
                    close_price=close_price,
                    target_weight=intended.get(code, 0.0),
                    executed_weight=(quantity * close_price / closing_nav),
                )
            )
    ledger = PortfolioLedger(
        run_id=targets.run_id,
        initial_cash=initial_cash,
        run_data_hash=run_data_hash,
        asset_registry_hash=asset_registry_hash,
        eligibility_source_hash=run_data_hash,
        execution_calendar_manifest_hash=execution_calendar_manifest_hash(
            tuple(used_receipts)
        ),
        execution_calendar_receipts=tuple(used_receipts),
        eligibility_evidence=tuple(
            sorted(
                used_evidence.values(),
                key=lambda item: (item.signal_date, item.ts_code),
            )
        ),
        cash=tuple(cash_snapshots),
        fills=tuple(fills),
        holdings=tuple(holdings),
    )
    ledger.validate()
    return ledger


def build_registered_b0_ledger(
    *,
    manifest: RunManifest,
    targets: TargetFrame,
    initial_cash: float,
    calendar_receipts: tuple[ExecutionCalendarReceipt, ...],
    eligibility_evidence: tuple[EligibilityEvidence, ...],
    prices: tuple[ExecutionPrice, ...],
    costs: CostSchedule,
) -> PortfolioLedger:
    """Registered entry point tying the common engine back to one run."""
    manifest.validate()
    if targets.run_id != manifest.run_id:
        raise ContractError("target frame and RunManifest run_id differ")
    ledger = build_b0_ledger(
        targets=targets,
        initial_cash=initial_cash,
        run_data_hash=manifest.data_hash,
        asset_registry_hash=manifest.asset_registry_hash,
        calendar_receipts=calendar_receipts,
        eligibility_evidence=eligibility_evidence,
        prices=prices,
        costs=costs,
    )
    assert_ledger_matches_run(ledger, manifest)
    return ledger

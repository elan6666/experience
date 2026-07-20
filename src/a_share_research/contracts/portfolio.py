"""Portfolio fills, holdings and cash accounting contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import ClassVar

from a_share_research.contracts.base import (
    CanonicalModel,
    ContractError,
    canonical_hash,
    require_nonnegative,
)
from a_share_research.contracts.data import _validate_ts_code

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def eligibility_evidence_hash(
    *,
    signal_date: date,
    trade_date: date,
    ts_code: str,
    asset_registry_hash: str,
    buyable: bool,
    sellable: bool,
    evidence_source_hash: str,
    trading_calendar_hash: str,
    next_trade_date: date,
    calendar_receipt_id: str,
) -> str:
    return canonical_hash(
        {
            "signal_date": signal_date,
            "trade_date": trade_date,
            "ts_code": ts_code,
            "asset_registry_hash": asset_registry_hash,
            "buyable": buyable,
            "sellable": sellable,
            "evidence_source_hash": evidence_source_hash,
            "trading_calendar_hash": trading_calendar_hash,
            "next_trade_date": next_trade_date,
            "calendar_receipt_id": calendar_receipt_id,
        }
    )


def execution_calendar_receipt_id(
    *, signal_date: date, next_trade_date: date, calendar_source_hash: str
) -> str:
    return canonical_hash(
        {
            "signal_date": signal_date,
            "next_trade_date": next_trade_date,
            "calendar_source_hash": calendar_source_hash,
        }
    )


@dataclass(frozen=True)
class ExecutionCalendarReceipt(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "execution_calendar_receipt"

    receipt_id: str
    signal_date: date
    next_trade_date: date
    calendar_source_hash: str

    def validate(self) -> None:
        if self.next_trade_date <= self.signal_date:
            raise ContractError("next trade date must follow signal date")
        if not _SHA256.fullmatch(self.calendar_source_hash):
            raise ContractError("calendar_source_hash must be SHA-256")
        expected = execution_calendar_receipt_id(
            signal_date=self.signal_date,
            next_trade_date=self.next_trade_date,
            calendar_source_hash=self.calendar_source_hash,
        )
        if self.receipt_id != expected:
            raise ContractError("execution calendar receipt_id mismatch")


def execution_calendar_manifest_hash(
    receipts: tuple[ExecutionCalendarReceipt, ...],
) -> str:
    payload = sorted(
        (receipt.to_dict() for receipt in receipts),
        key=lambda item: item["receipt_id"],
    )
    return canonical_hash(payload)


@dataclass(frozen=True)
class EligibilityEvidence(CanonicalModel):
    """Immutable D0 eligibility record referenced by fills."""

    SCHEMA_NAME: ClassVar[str] = "eligibility_evidence"

    evidence_id: str
    signal_date: date
    trade_date: date
    ts_code: str
    asset_registry_hash: str
    buyable: bool
    sellable: bool
    evidence_source_hash: str
    trading_calendar_hash: str
    next_trade_date: date
    calendar_receipt_id: str

    def validate(self) -> None:
        _validate_ts_code(self.ts_code)
        if self.signal_date >= self.trade_date:
            raise ContractError("eligibility trade_date must follow signal_date")
        for name in ("asset_registry_hash", "evidence_source_hash"):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ContractError(f"{name} must be SHA-256")
        if type(self.buyable) is not bool or type(self.sellable) is not bool:
            raise ContractError("buyable and sellable evidence must be bool")
        if not _SHA256.fullmatch(self.trading_calendar_hash):
            raise ContractError("trading_calendar_hash must be SHA-256")
        if not _SHA256.fullmatch(self.calendar_receipt_id):
            raise ContractError("calendar_receipt_id must be SHA-256")
        if self.trade_date != self.next_trade_date:
            raise ContractError("eligibility trade_date must equal trusted next_trade_date")
        expected = eligibility_evidence_hash(
            signal_date=self.signal_date,
            trade_date=self.trade_date,
            ts_code=self.ts_code,
            asset_registry_hash=self.asset_registry_hash,
            buyable=self.buyable,
            sellable=self.sellable,
            evidence_source_hash=self.evidence_source_hash,
            trading_calendar_hash=self.trading_calendar_hash,
            next_trade_date=self.next_trade_date,
            calendar_receipt_id=self.calendar_receipt_id,
        )
        if self.evidence_id != expected:
            raise ContractError("eligibility evidence_id does not match D0-anchored content")


class FillSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class PortfolioFill(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "portfolio_fill"

    trade_date: date
    ts_code: str
    side: FillSide
    quantity: float
    price: float
    commission: float
    tax: float
    slippage: float
    eligibility_evidence_id: str
    reject_reason: str | None = None

    def validate(self) -> None:
        _validate_ts_code(self.ts_code)
        if not isinstance(self.side, FillSide):
            raise ContractError("fill side must use FillSide")
        for name in ("quantity", "price", "commission", "tax", "slippage"):
            require_nonnegative(getattr(self, name), name)
        if self.reject_reason is None and (self.quantity <= 0 or self.price <= 0):
            raise ContractError("executed fill requires positive quantity and price")
        if not _SHA256.fullmatch(self.eligibility_evidence_id):
            raise ContractError("fill must reference a SHA-256 eligibility evidence_id")
        if self.reject_reason is not None and any(
            value != 0 for value in (self.quantity, self.commission, self.tax, self.slippage)
        ):
            raise ContractError("rejected order cannot change quantity or costs")

    @property
    def gross_value(self) -> float:
        return self.quantity * self.price

    @property
    def total_cost(self) -> float:
        return self.commission + self.tax + self.slippage


@dataclass(frozen=True)
class CashSnapshot(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "cash_snapshot"

    trade_date: date
    opening_cash: float
    closing_cash: float

    def validate(self) -> None:
        require_nonnegative(self.opening_cash, "opening_cash")
        require_nonnegative(self.closing_cash, "closing_cash")


@dataclass(frozen=True)
class HoldingSnapshot(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "holding_snapshot"

    trade_date: date
    ts_code: str
    quantity: float
    close_price: float
    target_weight: float
    executed_weight: float

    def validate(self) -> None:
        _validate_ts_code(self.ts_code)
        for name in ("quantity", "close_price", "target_weight", "executed_weight"):
            require_nonnegative(getattr(self, name), name)
        if self.target_weight > 1 or self.executed_weight > 1:
            raise ContractError("position weights cannot exceed one")

    @property
    def market_value(self) -> float:
        return self.quantity * self.close_price


@dataclass(frozen=True)
class PortfolioLedger(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "portfolio_ledger"

    run_id: str
    initial_cash: float
    run_data_hash: str
    asset_registry_hash: str
    eligibility_source_hash: str
    execution_calendar_manifest_hash: str
    execution_calendar_receipts: tuple[ExecutionCalendarReceipt, ...]
    eligibility_evidence: tuple[EligibilityEvidence, ...]
    cash: tuple[CashSnapshot, ...]
    fills: tuple[PortfolioFill, ...]
    holdings: tuple[HoldingSnapshot, ...]

    def validate(self) -> None:
        if not self.run_id:
            raise ContractError("run_id is required")
        require_nonnegative(self.initial_cash, "initial_cash")
        for name in (
            "run_data_hash",
            "asset_registry_hash",
            "eligibility_source_hash",
            "execution_calendar_manifest_hash",
        ):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ContractError(f"{name} must be SHA-256")
        if self.run_data_hash != self.eligibility_source_hash:
            raise ContractError("ledger eligibility source must equal run_data_hash")
        calendar_by_id: dict[str, ExecutionCalendarReceipt] = {}
        for receipt in self.execution_calendar_receipts:
            receipt.validate()
            if receipt.receipt_id in calendar_by_id:
                raise ContractError("duplicate execution calendar receipt")
            if receipt.calendar_source_hash != self.run_data_hash:
                raise ContractError("calendar receipt is not anchored to run data")
            calendar_by_id[receipt.receipt_id] = receipt
        if self.execution_calendar_manifest_hash != execution_calendar_manifest_hash(
            self.execution_calendar_receipts
        ):
            raise ContractError("execution calendar collection is not manifest-anchored")
        evidence_by_id: dict[str, EligibilityEvidence] = {}
        for evidence in self.eligibility_evidence:
            evidence.validate()
            if evidence.evidence_id in evidence_by_id:
                raise ContractError("duplicate eligibility evidence_id")
            if evidence.asset_registry_hash != self.asset_registry_hash:
                raise ContractError("eligibility evidence uses another asset registry")
            if evidence.evidence_source_hash != self.eligibility_source_hash:
                raise ContractError("eligibility evidence is not anchored to ledger D0 source")
            receipt = calendar_by_id.get(evidence.calendar_receipt_id)
            if receipt is None:
                raise ContractError("eligibility evidence lacks trusted calendar receipt")
            if (
                receipt.signal_date != evidence.signal_date
                or receipt.next_trade_date != evidence.next_trade_date
                or receipt.calendar_source_hash != evidence.trading_calendar_hash
            ):
                raise ContractError("eligibility and calendar receipts do not match")
            evidence_by_id[evidence.evidence_id] = evidence
        if not self.cash:
            raise ContractError("cash snapshots are required")
        cash_dates = [snapshot.trade_date for snapshot in self.cash]
        if cash_dates != sorted(set(cash_dates)):
            raise ContractError("cash snapshots must have unique increasing dates")
        previous_close = self.initial_cash
        for snapshot in self.cash:
            snapshot.validate()
            if abs(snapshot.opening_cash - previous_close) > 1e-8:
                raise ContractError("cash does not roll forward between trading days")
            day_fills = [fill for fill in self.fills if fill.trade_date == snapshot.trade_date]
            for fill in day_fills:
                fill.validate()
                evidence = evidence_by_id.get(fill.eligibility_evidence_id)
                if evidence is None:
                    raise ContractError("fill references missing eligibility evidence")
                if evidence.trade_date != fill.trade_date or evidence.ts_code != fill.ts_code:
                    raise ContractError("fill does not match referenced eligibility evidence")
                if fill.reject_reason is None:
                    if fill.side is FillSide.BUY and not evidence.buyable:
                        raise ContractError("executed buy is not buyable in trusted evidence")
                    if fill.side is FillSide.SELL and not evidence.sellable:
                        raise ContractError("executed sell is not sellable in trusted evidence")
            buy_value = sum(
                fill.gross_value for fill in day_fills if fill.side is FillSide.BUY
            )
            sell_value = sum(
                fill.gross_value for fill in day_fills if fill.side is FillSide.SELL
            )
            total_cost = sum(fill.total_cost for fill in day_fills)
            expected_close = snapshot.opening_cash - buy_value + sell_value - total_cost
            if abs(snapshot.closing_cash - expected_close) > 1e-8:
                raise ContractError("cash reconciliation failed")
            previous_close = snapshot.closing_cash
        known_dates = set(cash_dates)
        if any(fill.trade_date not in known_dates for fill in self.fills):
            raise ContractError("fill date has no cash snapshot")
        for holding in self.holdings:
            holding.validate()
            if holding.trade_date not in known_dates:
                raise ContractError("holding date has no cash snapshot")
        holding_keys = [(item.trade_date, item.ts_code) for item in self.holdings]
        if len(holding_keys) != len(set(holding_keys)):
            raise ContractError("duplicate holding snapshot")
        quantities: dict[str, float] = {}
        for trade_date in cash_dates:
            positions = [holding for holding in self.holdings if holding.trade_date == trade_date]
            if sum(position.target_weight for position in positions) > 1 + 1e-10:
                raise ContractError("target weights exceed total capital")
            if sum(position.executed_weight for position in positions) > 1 + 1e-10:
                raise ContractError("executed weights exceed total capital")
            executed = [
                fill
                for fill in self.fills
                if fill.trade_date == trade_date and fill.reject_reason is None
            ]
            opening_quantities = dict(quantities)
            sold_by_code: dict[str, float] = {}
            bought_by_code: dict[str, float] = {}
            for fill in executed:
                target = bought_by_code if fill.side is FillSide.BUY else sold_by_code
                target[fill.ts_code] = target.get(fill.ts_code, 0.0) + fill.quantity
            for ts_code, sold in sold_by_code.items():
                if sold > opening_quantities.get(ts_code, 0.0) + 1e-10:
                    raise ContractError("intraday sell exceeds opening carried quantity under T+1")
            for ts_code in set(opening_quantities) | set(bought_by_code) | set(sold_by_code):
                quantities[ts_code] = (
                    opening_quantities.get(ts_code, 0.0)
                    + bought_by_code.get(ts_code, 0.0)
                    - sold_by_code.get(ts_code, 0.0)
                )
            recorded = {position.ts_code: position.quantity for position in positions}
            expected = {
                ts_code: quantity
                for ts_code, quantity in quantities.items()
                if quantity > 1e-10
            }
            if recorded != expected:
                raise ContractError("holding quantity does not reconcile to fills")

    def nav(self, trade_date: date) -> float:
        snapshot = next((item for item in self.cash if item.trade_date == trade_date), None)
        if snapshot is None:
            raise ContractError(f"unknown ledger date: {trade_date}")
        holdings_value = sum(
            item.market_value for item in self.holdings if item.trade_date == trade_date
        )
        return snapshot.closing_cash + holdings_value

    def turnover(self, trade_date: date, previous_nav: float) -> float:
        require_nonnegative(previous_nav, "previous_nav")
        if previous_nav == 0:
            raise ContractError("previous_nav must be positive")
        traded = sum(
            fill.gross_value
            for fill in self.fills
            if fill.trade_date == trade_date and fill.reject_reason is None
        )
        return traded / previous_nav


def assert_ledger_matches_run(ledger: PortfolioLedger, manifest: object) -> None:
    """Formal accounting entry point tying execution receipts to one run."""
    from a_share_research.contracts.run import RunManifest

    if not isinstance(manifest, RunManifest):
        raise TypeError("formal ledger matching requires RunManifest")
    ledger.validate()
    if ledger.run_data_hash != getattr(manifest, "data_hash", None):
        raise ContractError("ledger data hash does not match RunManifest")
    if ledger.asset_registry_hash != getattr(manifest, "asset_registry_hash", None):
        raise ContractError("ledger asset registry does not match RunManifest")
    if ledger.execution_calendar_manifest_hash != getattr(
        manifest, "execution_calendar_manifest_hash", None
    ):
        raise ContractError("ledger execution calendar does not match RunManifest")

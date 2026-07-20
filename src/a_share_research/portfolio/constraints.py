"""Deterministic V2 B2 constraints around the common execution ledger."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date
from typing import ClassVar

from a_share_research.contracts import (
    ContractError,
    EligibilityEvidence,
    ExecutionCalendarReceipt,
    PortfolioLedger,
)
from a_share_research.contracts.base import (
    CanonicalModel,
    require_finite,
    require_nonnegative,
)
from a_share_research.contracts.data import _validate_ts_code

from .execution import CostSchedule, ExecutionPrice, build_b0_ledger
from .intents import TargetFrame, TargetWeight
from .risk_budget import RiskBudgetSchedule

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BUY_REJECT_REASONS = frozenset(
    {"LIMIT_UP", "SUSPENDED", "ST_NOT_ELIGIBLE", "NOT_BUYABLE"}
)
_SELL_REJECT_REASONS = frozenset(
    {"LIMIT_DOWN", "SUSPENDED", "ST_NOT_ELIGIBLE", "NOT_SELLABLE"}
)


@dataclass(frozen=True)
class ConstraintPolicy(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "v2_constraint_policy"

    version: str
    max_single_weight: float
    max_industry_weight: float
    max_adv_participation: float
    max_one_way_turnover: float
    fallback: str = "PROPORTIONAL_CLIP_TO_CASH"

    def validate(self) -> None:
        if not self.version:
            raise ContractError("constraint policy version is required")
        for name in (
            "max_single_weight",
            "max_industry_weight",
            "max_adv_participation",
            "max_one_way_turnover",
        ):
            value = getattr(self, name)
            require_finite(value, name)
            if not 0 < value <= 1:
                raise ContractError(f"{name} must be in (0, 1]")
        if self.max_single_weight > self.max_industry_weight:
            raise ContractError("single-stock cap cannot exceed industry cap")
        if self.fallback != "PROPORTIONAL_CLIP_TO_CASH":
            raise ContractError("unreviewed constraint fallback")


@dataclass(frozen=True)
class ConstraintContext(CanonicalModel):
    """Signal-time inputs for selected or carried securities only."""

    SCHEMA_NAME: ClassVar[str] = "v2_constraint_context"

    signal_date: date
    ts_code: str
    industry: str
    adv_value: float
    reference_nav: float
    source_hash: str

    def validate(self) -> None:
        _validate_ts_code(self.ts_code)
        if not self.industry or not _SHA256.fullmatch(self.source_hash):
            raise ContractError("constraint context needs PIT industry and source hash")
        for name in ("adv_value", "reference_nav"):
            value = getattr(self, name)
            require_finite(value, name)
            if value <= 0:
                raise ContractError(f"{name} must be positive")


@dataclass(frozen=True)
class TradingRestriction(CanonicalModel):
    """Detailed D0 reason layered over side-specific eligibility booleans."""

    SCHEMA_NAME: ClassVar[str] = "v2_trading_restriction"

    signal_date: date
    ts_code: str
    buy_reject_reason: str | None = None
    sell_reject_reason: str | None = None

    def validate(self) -> None:
        _validate_ts_code(self.ts_code)
        if self.buy_reject_reason is None and self.sell_reject_reason is None:
            raise ContractError("trading restriction must block at least one side")
        if (
            self.buy_reject_reason is not None
            and self.buy_reject_reason not in _BUY_REJECT_REASONS
        ):
            raise ContractError("buy restriction reason is not registered for buys")
        if (
            self.sell_reject_reason is not None
            and self.sell_reject_reason not in _SELL_REJECT_REASONS
        ):
            raise ContractError("sell restriction reason is not registered for sells")


@dataclass(frozen=True)
class ConstraintDecision(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "v2_constraint_decision"

    signal_date: date
    ts_code: str
    industry: str
    input_weight: float
    previous_weight: float
    constrained_weight: float
    adv_delta_cap_weight: float
    reasons: tuple[str, ...]

    def validate(self) -> None:
        _validate_ts_code(self.ts_code)
        if not self.industry:
            raise ContractError("constraint decision requires PIT industry")
        for name in (
            "input_weight",
            "previous_weight",
            "constrained_weight",
            "adv_delta_cap_weight",
        ):
            require_nonnegative(getattr(self, name), name)
        if any(not reason for reason in self.reasons):
            raise ContractError("constraint reason cannot be empty")


@dataclass(frozen=True)
class ConstraintEvidence(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "v2_constraint_evidence"

    run_id: str
    input_target_hash: str
    output_target_hash: str
    risk_schedule_hash: str
    policy_hash: str
    decisions: tuple[ConstraintDecision, ...]
    fallback_by_date: dict[str, str]

    def validate(self) -> None:
        if not self.run_id or not self.fallback_by_date:
            raise ContractError("constraint evidence requires run and dated fallback evidence")
        for name in (
            "input_target_hash",
            "output_target_hash",
            "risk_schedule_hash",
            "policy_hash",
        ):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ContractError(f"{name} must be SHA-256")
        keys = [(row.signal_date, row.ts_code) for row in self.decisions]
        if keys != sorted(set(keys)):
            raise ContractError("constraint decisions must be unique and sorted")
        allowed_fallbacks = {
            "NONE",
            "PROPORTIONAL_CLIP_TO_CASH",
            "CARRY_DUE_TO_TRANSACTION_CAP",
            "RISK_ZERO_LIQUIDATE",
        }
        for date_text, fallback in self.fallback_by_date.items():
            try:
                date.fromisoformat(date_text)
            except ValueError as error:
                raise ContractError("constraint fallback keys must be ISO dates") from error
            if fallback not in allowed_fallbacks:
                raise ContractError("constraint fallback is not registered")


@dataclass(frozen=True)
class StrategyDateEvidence(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "v2_strategy_date_evidence"

    signal_date: date
    trade_date: date
    risk_budget: float
    target_equity_weight: float
    executed_equity_weight: float
    closing_cash_weight: float
    executed_fill_count: int
    reject_count: int
    reject_reasons: tuple[str, ...]
    gross_traded_value: float
    total_cost: float
    one_way_turnover: float
    fallback: str

    def validate(self) -> None:
        if self.trade_date <= self.signal_date:
            raise ContractError("strategy evidence must execute on T+1 or later")
        for name in (
            "risk_budget",
            "target_equity_weight",
            "executed_equity_weight",
            "closing_cash_weight",
            "gross_traded_value",
            "total_cost",
            "one_way_turnover",
        ):
            require_nonnegative(getattr(self, name), name)
        if self.risk_budget > 1 or self.target_equity_weight > 1:
            raise ContractError("strategy target exposure cannot exceed one")
        if self.executed_equity_weight > 1 + 1e-10 or self.closing_cash_weight > 1 + 1e-10:
            raise ContractError("executed strategy weights cannot exceed one")
        if self.executed_fill_count < 0 or self.reject_count < 0:
            raise ContractError("fill counts cannot be negative")
        if self.reject_count != len(self.reject_reasons):
            raise ContractError("reject count and reason evidence differ")


@dataclass(frozen=True)
class ExecutionEvidence(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "v2_execution_evidence"

    run_id: str
    ledger_hash: str
    constraint_evidence_hash: str
    rows: tuple[StrategyDateEvidence, ...]

    def validate(self) -> None:
        if not self.run_id or not self.rows:
            raise ContractError("execution evidence requires run and dated rows")
        for name in ("ledger_hash", "constraint_evidence_hash"):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ContractError(f"{name} must be SHA-256")
        dates = tuple(row.signal_date for row in self.rows)
        if dates != tuple(sorted(set(dates))):
            raise ContractError("execution evidence dates must be unique and increasing")


@dataclass(frozen=True)
class B2Result:
    targets: TargetFrame
    constraint_evidence: ConstraintEvidence
    ledger: PortfolioLedger
    execution_evidence: ExecutionEvidence

    def validate(self) -> None:
        self.targets.validate()
        self.constraint_evidence.validate()
        self.ledger.validate()
        self.execution_evidence.validate()
        if len(
            {
                self.targets.run_id,
                self.constraint_evidence.run_id,
                self.ledger.run_id,
                self.execution_evidence.run_id,
            }
        ) != 1:
            raise ContractError("B2 target, ledger and execution run IDs differ")
        if self.constraint_evidence.output_target_hash != self.targets.stable_hash():
            raise ContractError("B2 constraint evidence does not match targets")
        if self.execution_evidence.ledger_hash != self.ledger.stable_hash():
            raise ContractError("B2 execution evidence does not match ledger")
        if (
            self.execution_evidence.constraint_evidence_hash
            != self.constraint_evidence.stable_hash()
        ):
            raise ContractError("B2 execution evidence does not match constraints")


def _one_way_turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    codes = set(previous) | set(current)
    previous_equity = math.fsum(previous.values())
    current_equity = math.fsum(current.values())
    return 0.5 * (
        math.fsum(abs(current.get(code, 0.0) - previous.get(code, 0.0)) for code in codes)
        + abs((1 - current_equity) - (1 - previous_equity))
    )


def constrain_target_frame(
    *,
    b1_targets: TargetFrame,
    schedule: RiskBudgetSchedule,
    contexts: tuple[ConstraintContext, ...],
    policy: ConstraintPolicy,
    output_run_id: str,
) -> tuple[TargetFrame, ConstraintEvidence]:
    """Clip frozen Top10% intents; never read predictions or alter their ordering."""
    b1_targets.validate()
    schedule.validate()
    policy.validate()
    if not output_run_id:
        raise ContractError("B2 output_run_id is required")
    signal_dates = tuple(
        sorted(date.fromisoformat(key) for key in b1_targets.cash_weight_by_date)
    )
    if signal_dates != tuple(point.signal_date for point in schedule.points):
        raise ContractError("B1 targets and risk schedule dates differ")
    budget_by_date = schedule.by_date
    input_by_date: dict[date, dict[str, float]] = defaultdict(dict)
    for target in b1_targets.targets:
        input_by_date[target.signal_date][target.ts_code] = target.weight
    for signal_date in signal_dates:
        input_equity = math.fsum(input_by_date[signal_date].values())
        if not math.isclose(input_equity, budget_by_date[signal_date], abs_tol=1e-10):
            raise ContractError("B1 targets do not equal the shared dated risk budget")
    context_by_key: dict[tuple[date, str], ConstraintContext] = {}
    for context in contexts:
        context.validate()
        key = (context.signal_date, context.ts_code)
        if key in context_by_key:
            raise ContractError("duplicate B2 constraint context")
        context_by_key[key] = context

    previous: dict[str, float] = {}
    outputs: list[TargetWeight] = []
    decisions: list[ConstraintDecision] = []
    fallback_by_date: dict[str, str] = {}
    for signal_date in signal_dates:
        budget = budget_by_date[signal_date]
        intended = input_by_date[signal_date]
        relevant = set(intended) | set(previous)
        if not relevant and budget > 0:
            raise ContractError("non-zero B2 budget has no frozen selected securities")
        missing_context = sorted(
            code for code in relevant if (signal_date, code) not in context_by_key
        )
        if missing_context:
            raise ContractError(f"B2 lacks PIT constraint context: {missing_context}")
        reasons: dict[str, set[str]] = defaultdict(set)
        candidate: dict[str, float] = {}
        if budget == 0:
            candidate = {code: 0.0 for code in relevant}
            for code in relevant:
                reasons[code].add("RISK_ZERO")
            fallback = "RISK_ZERO_LIQUIDATE"
        else:
            for code in relevant:
                raw = intended.get(code, 0.0)
                candidate[code] = min(raw, policy.max_single_weight)
                if candidate[code] < raw - 1e-12:
                    reasons[code].add("SINGLE_WEIGHT_CAP")
            industries: dict[str, list[str]] = defaultdict(list)
            for code in relevant:
                industries[context_by_key[(signal_date, code)].industry].append(code)
            for codes in industries.values():
                total = math.fsum(candidate[code] for code in codes)
                if total > policy.max_industry_weight:
                    scale = policy.max_industry_weight / total
                    for code in codes:
                        candidate[code] *= scale
                        reasons[code].add("INDUSTRY_WEIGHT_CAP")
            for code in relevant:
                context = context_by_key[(signal_date, code)]
                cap = context.adv_value * policy.max_adv_participation / context.reference_nav
                delta = candidate[code] - previous.get(code, 0.0)
                if abs(delta) > cap:
                    candidate[code] = previous.get(code, 0.0) + math.copysign(cap, delta)
                    reasons[code].add("ADV_CAPACITY_CAP")
            turnover = _one_way_turnover(previous, candidate)
            if turnover > policy.max_one_way_turnover:
                alpha = policy.max_one_way_turnover / turnover
                candidate = {
                    code: previous.get(code, 0.0)
                    + alpha * (candidate.get(code, 0.0) - previous.get(code, 0.0))
                    for code in relevant
                }
                for code in relevant:
                    reasons[code].add("TURNOVER_CAP")
            output_equity = math.fsum(candidate.values())
            if output_equity <= 1e-12:
                raise ContractError("non-zero B2 budget cannot silently become all cash")
            if any(candidate[code] > intended.get(code, 0.0) + 1e-12 for code in relevant):
                fallback = "CARRY_DUE_TO_TRANSACTION_CAP"
            elif output_equity < budget - 1e-12:
                fallback = "PROPORTIONAL_CLIP_TO_CASH"
            else:
                fallback = "NONE"
        output_equity = math.fsum(candidate.values())
        if output_equity > 1 + 1e-10:
            raise ContractError("B2 constraints produced leverage")
        for code in sorted(relevant):
            context = context_by_key[(signal_date, code)]
            weight = max(0.0, candidate[code])
            cap = context.adv_value * policy.max_adv_participation / context.reference_nav
            decisions.append(
                ConstraintDecision(
                    signal_date=signal_date,
                    ts_code=code,
                    industry=context.industry,
                    input_weight=intended.get(code, 0.0),
                    previous_weight=previous.get(code, 0.0),
                    constrained_weight=weight,
                    adv_delta_cap_weight=cap,
                    reasons=tuple(sorted(reasons[code])),
                )
            )
            if weight > 1e-12:
                outputs.append(TargetWeight(signal_date, code, weight))
        fallback_by_date[signal_date.isoformat()] = fallback
        previous = {code: weight for code, weight in candidate.items() if weight > 1e-12}
    output_cash = {
        signal_date.isoformat(): 1.0
        - math.fsum(target.weight for target in outputs if target.signal_date == signal_date)
        for signal_date in signal_dates
    }
    frame = TargetFrame(output_run_id, tuple(outputs), output_cash)
    evidence = ConstraintEvidence(
        run_id=output_run_id,
        input_target_hash=b1_targets.stable_hash(),
        output_target_hash=frame.stable_hash(),
        risk_schedule_hash=schedule.stable_hash(),
        policy_hash=policy.stable_hash(),
        decisions=tuple(decisions),
        fallback_by_date=fallback_by_date,
    )
    return frame, evidence


def _apply_restriction_reasons(
    *,
    ledger: PortfolioLedger,
    restrictions: tuple[TradingRestriction, ...],
) -> PortfolioLedger:
    restriction_by_key: dict[tuple[date, str], TradingRestriction] = {}
    for restriction in restrictions:
        restriction.validate()
        key = (restriction.signal_date, restriction.ts_code)
        if key in restriction_by_key:
            raise ContractError("duplicate detailed trading restriction")
        restriction_by_key[key] = restriction
    signal_by_trade = {
        receipt.next_trade_date: receipt.signal_date
        for receipt in ledger.execution_calendar_receipts
    }
    evidence_by_key = {
        (item.signal_date, item.ts_code): item for item in ledger.eligibility_evidence
    }
    for key, restriction in restriction_by_key.items():
        evidence = evidence_by_key.get(key)
        if evidence is None:
            raise ContractError("detailed trading restriction lacks execution evidence")
        if restriction.buy_reject_reason is not None and evidence.buyable:
            raise ContractError("buy restriction conflicts with buyable evidence")
        if restriction.sell_reject_reason is not None and evidence.sellable:
            raise ContractError("sell restriction conflicts with sellable evidence")
    replaced_fills = []
    for fill in ledger.fills:
        if fill.reject_reason is None:
            replaced_fills.append(fill)
            continue
        signal_date = signal_by_trade[fill.trade_date]
        restriction = restriction_by_key.get((signal_date, fill.ts_code))
        if restriction is None:
            replaced_fills.append(fill)
            continue
        evidence = evidence_by_key[(signal_date, fill.ts_code)]
        if fill.side.value == "BUY":
            reason = restriction.buy_reject_reason
        else:
            reason = restriction.sell_reject_reason
        replaced_fills.append(replace(fill, reject_reason=reason or fill.reject_reason))
    return replace(ledger, fills=tuple(replaced_fills))


def _build_execution_evidence(
    *,
    targets: TargetFrame,
    schedule: RiskBudgetSchedule,
    constraints: ConstraintEvidence,
    ledger: PortfolioLedger,
) -> ExecutionEvidence:
    target_by_date: dict[date, float] = defaultdict(float)
    for target in targets.targets:
        target_by_date[target.signal_date] += target.weight
    receipt_by_signal = {
        receipt.signal_date: receipt for receipt in ledger.execution_calendar_receipts
    }
    previous_nav = ledger.initial_cash
    rows: list[StrategyDateEvidence] = []
    for point in schedule.points:
        receipt = receipt_by_signal[point.signal_date]
        trade_date = receipt.next_trade_date
        nav = ledger.nav(trade_date)
        day_fills = [fill for fill in ledger.fills if fill.trade_date == trade_date]
        executed = [fill for fill in day_fills if fill.reject_reason is None]
        rejected = [fill for fill in day_fills if fill.reject_reason is not None]
        rows.append(
            StrategyDateEvidence(
                signal_date=point.signal_date,
                trade_date=trade_date,
                risk_budget=point.equity_budget,
                target_equity_weight=target_by_date[point.signal_date],
                executed_equity_weight=math.fsum(
                    holding.executed_weight
                    for holding in ledger.holdings
                    if holding.trade_date == trade_date
                ),
                closing_cash_weight=next(
                    cash.closing_cash / nav
                    for cash in ledger.cash
                    if cash.trade_date == trade_date
                ),
                executed_fill_count=len(executed),
                reject_count=len(rejected),
                reject_reasons=tuple(fill.reject_reason or "" for fill in rejected),
                gross_traded_value=math.fsum(fill.gross_value for fill in executed),
                total_cost=math.fsum(fill.total_cost for fill in executed),
                one_way_turnover=ledger.turnover(trade_date, previous_nav),
                fallback=constraints.fallback_by_date[point.signal_date.isoformat()],
            )
        )
        previous_nav = nav
    return ExecutionEvidence(
        run_id=targets.run_id,
        ledger_hash=ledger.stable_hash(),
        constraint_evidence_hash=constraints.stable_hash(),
        rows=tuple(rows),
    )


def build_b2_constrained_ledger(
    *,
    b1_targets: TargetFrame,
    schedule: RiskBudgetSchedule,
    contexts: tuple[ConstraintContext, ...],
    policy: ConstraintPolicy,
    output_run_id: str,
    initial_cash: float,
    run_data_hash: str,
    asset_registry_hash: str,
    calendar_receipts: tuple[ExecutionCalendarReceipt, ...],
    eligibility_evidence: tuple[EligibilityEvidence, ...],
    prices: tuple[ExecutionPrice, ...],
    costs: CostSchedule,
    restrictions: tuple[TradingRestriction, ...] = (),
) -> B2Result:
    """Build B2 without fitting a predictor or altering frozen stock scores."""
    targets, constraint_evidence = constrain_target_frame(
        b1_targets=b1_targets,
        schedule=schedule,
        contexts=contexts,
        policy=policy,
        output_run_id=output_run_id,
    )
    ledger = build_b0_ledger(
        targets=targets,
        initial_cash=initial_cash,
        run_data_hash=run_data_hash,
        asset_registry_hash=asset_registry_hash,
        calendar_receipts=calendar_receipts,
        eligibility_evidence=eligibility_evidence,
        prices=prices,
        costs=costs,
    )
    ledger = _apply_restriction_reasons(ledger=ledger, restrictions=restrictions)
    execution_evidence = _build_execution_evidence(
        targets=targets,
        schedule=schedule,
        constraints=constraint_evidence,
        ledger=ledger,
    )
    result = B2Result(targets, constraint_evidence, ledger, execution_evidence)
    result.validate()
    return result

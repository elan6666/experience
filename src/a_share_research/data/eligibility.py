"""Independent D0 membership/observation/feature/execution mask construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from a_share_research.contracts import (
    AssetRegistry,
    ContractError,
    EligibilityEvidence,
    ExecutionCalendarReceipt,
    MaskBundle,
    canonical_hash,
    eligibility_evidence_hash,
    execution_calendar_receipt_id,
)


@dataclass(frozen=True)
class ExecutionStatus:
    observed: bool
    suspended_at_open: bool
    open_price: float | None
    up_limit: float | None
    down_limit: float | None

    @property
    def buyable(self) -> bool:
        return bool(
            self.observed
            and not self.suspended_at_open
            and self.open_price is not None
            and self.up_limit is not None
            and self.open_price < self.up_limit
        )

    @property
    def sellable(self) -> bool:
        return bool(
            self.observed
            and not self.suspended_at_open
            and self.open_price is not None
            and self.down_limit is not None
            and self.open_price > self.down_limit
        )


def build_mask_bundle(
    *,
    signal_date: date,
    asset_registry: AssetRegistry,
    member: dict[str, bool],
    statuses: dict[str, ExecutionStatus],
    feature_missing: dict[str, dict[str, bool]],
    label_available: dict[str, bool],
    execution_statuses: dict[str, ExecutionStatus] | None = None,
) -> MaskBundle:
    if not feature_missing:
        raise ContractError("one missing mask per D0 feature is required")
    assets = asset_registry.asset_ids
    missing_masks = {
        feature: tuple(bool(values.get(code, True)) for code in assets)
        for feature, values in sorted(feature_missing.items())
    }
    default_status = ExecutionStatus(False, False, None, None, None)
    trade_statuses = execution_statuses if execution_statuses is not None else statuses
    observed = tuple(statuses.get(code, default_status).observed for code in assets)
    member_mask = tuple(bool(member.get(code, False)) for code in assets)
    label_mask = tuple(bool(label_available.get(code, False)) for code in assets)
    buyable = tuple(
        bool(
            member_mask[index]
            and observed[index]
            and trade_statuses.get(code, default_status).buyable
        )
        for index, code in enumerate(assets)
    )
    sellable = tuple(
        bool(observed[index] and trade_statuses.get(code, default_status).sellable)
        for index, code in enumerate(assets)
    )
    loss = tuple(
        observed[index] and label_mask[index] for index in range(len(assets))
    )
    evaluation = tuple(
        member_mask[index] and observed[index] and label_mask[index]
        for index in range(len(assets))
    )
    return MaskBundle(
        signal_date=signal_date,
        asset_ids=assets,
        asset_registry_hash=asset_registry.stable_hash(),
        member=member_mask,
        observed=observed,
        feature_missing=missing_masks,
        label_available=label_mask,
        buyable=buyable,
        sellable=sellable,
        loss=loss,
        evaluation=evaluation,
    )


def build_execution_receipts(
    *,
    signal_date: date,
    next_trade_date: date,
    asset_registry: AssetRegistry,
    statuses: dict[str, ExecutionStatus],
    member: dict[str, bool],
    d0_manifest_hash: str,
    trading_calendar: tuple[date, ...],
) -> tuple[ExecutionCalendarReceipt, tuple[EligibilityEvidence, ...]]:
    """Bind T+1 calendar and execution eligibility directly to the D0 hash."""
    calendar_hash = canonical_hash(trading_calendar)
    receipt_id = execution_calendar_receipt_id(
        signal_date=signal_date,
        next_trade_date=next_trade_date,
        calendar_source_hash=d0_manifest_hash,
    )
    calendar_receipt = ExecutionCalendarReceipt(
        receipt_id=receipt_id,
        signal_date=signal_date,
        next_trade_date=next_trade_date,
        calendar_source_hash=d0_manifest_hash,
    )
    evidence: list[EligibilityEvidence] = []
    for code in asset_registry.asset_ids:
        status = statuses.get(code, ExecutionStatus(False, False, None, None, None))
        buyable = bool(member.get(code, False) and status.buyable)
        sellable = status.sellable
        evidence_id = eligibility_evidence_hash(
            signal_date=signal_date,
            trade_date=next_trade_date,
            ts_code=code,
            asset_registry_hash=asset_registry.stable_hash(),
            buyable=buyable,
            sellable=sellable,
            evidence_source_hash=d0_manifest_hash,
            trading_calendar_hash=calendar_hash,
            next_trade_date=next_trade_date,
            calendar_receipt_id=receipt_id,
        )
        evidence.append(
            EligibilityEvidence(
                evidence_id=evidence_id,
                signal_date=signal_date,
                trade_date=next_trade_date,
                ts_code=code,
                asset_registry_hash=asset_registry.stable_hash(),
                buyable=buyable,
                sellable=sellable,
                evidence_source_hash=d0_manifest_hash,
                trading_calendar_hash=calendar_hash,
                next_trade_date=next_trade_date,
                calendar_receipt_id=receipt_id,
            )
        )
    return calendar_receipt, tuple(evidence)

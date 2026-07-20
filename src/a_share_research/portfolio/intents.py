"""Frozen target-weight boundary shared by models and baselines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import ClassVar

from a_share_research.contracts.base import CanonicalModel, ContractError, require_nonnegative
from a_share_research.contracts.data import _validate_ts_code


@dataclass(frozen=True)
class TargetWeight(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "target_weight"

    signal_date: date
    ts_code: str
    weight: float

    def validate(self) -> None:
        _validate_ts_code(self.ts_code)
        require_nonnegative(self.weight, "target weight")
        if self.weight > 1:
            raise ContractError("target weight cannot exceed one")


@dataclass(frozen=True)
class TargetFrame(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "target_frame"

    run_id: str
    targets: tuple[TargetWeight, ...]
    cash_weight_by_date: dict[str, float]

    def validate(self) -> None:
        if not self.run_id or not self.cash_weight_by_date:
            raise ContractError("target frame requires run_id and signal dates")
        keys: set[tuple[date, str]] = set()
        totals: dict[str, float] = {signal_date: 0.0 for signal_date in self.cash_weight_by_date}
        for date_text, cash_weight in self.cash_weight_by_date.items():
            try:
                date.fromisoformat(date_text)
            except ValueError as error:
                raise ContractError("cash weight keys must be ISO dates") from error
            require_nonnegative(cash_weight, "cash weight")
            if cash_weight > 1:
                raise ContractError("cash weight cannot exceed one")
        for target in self.targets:
            target.validate()
            date_text = target.signal_date.isoformat()
            if date_text not in totals:
                raise ContractError("target date lacks an explicit cash weight")
            key = (target.signal_date, target.ts_code)
            if key in keys:
                raise ContractError("duplicate target key")
            keys.add(key)
            totals[date_text] += target.weight
        for date_text, total in totals.items():
            if abs(total + self.cash_weight_by_date[date_text] - 1) > 1e-10:
                raise ContractError("target and cash weights must sum to one on every date")


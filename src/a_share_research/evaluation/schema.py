"""Frequency- and support-explicit common evaluation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from a_share_research.contracts.base import (
    CanonicalModel,
    ContractError,
    require_finite,
    require_nonnegative,
)


class EvaluationFrequency(str, Enum):
    DAILY = "DAILY_1D"
    WEEKLY = "WEEKLY_5D"
    MONTHLY = "MONTHLY_20D"

    @property
    def horizon(self) -> int:
        return {
            EvaluationFrequency.DAILY: 1,
            EvaluationFrequency.WEEKLY: 5,
            EvaluationFrequency.MONTHLY: 20,
        }[self]


class SupportMode(str, Enum):
    COMMON = "COMMON"
    NATIVE = "NATIVE"


class OutcomeMode(str, Enum):
    ABSOLUTE = "ABSOLUTE"
    BENCHMARK_RELATIVE = "BENCHMARK_RELATIVE"


@dataclass(frozen=True)
class PredictionScorecard(CanonicalModel):
    """One result cell; frequencies, supports and outcomes never mix."""

    SCHEMA_NAME: ClassVar[str] = "prediction_scorecard"

    run_id: str
    frequency: EvaluationFrequency
    support: SupportMode
    outcome: OutcomeMode
    horizon: int
    paired_dates: int
    paired_rows: int
    coverage: float
    rank_ic: float | None
    icir: float | None
    mae: float
    rmse: float
    sign_accuracy: float
    group_returns: tuple[float, ...]
    monotone_fraction: float | None
    excluded_constant_dates: int

    def validate(self) -> None:
        if not self.run_id:
            raise ContractError("scorecard run_id is required")
        if not isinstance(self.frequency, EvaluationFrequency):
            raise ContractError("frequency must use EvaluationFrequency")
        if not isinstance(self.support, SupportMode):
            raise ContractError("support must use SupportMode")
        if not isinstance(self.outcome, OutcomeMode):
            raise ContractError("outcome must use OutcomeMode")
        if self.horizon != self.frequency.horizon:
            raise ContractError("scorecard horizon and frequency disagree")
        for name in ("paired_dates", "paired_rows", "excluded_constant_dates"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ContractError(f"{name} must be a non-negative integer")
        if self.paired_rows == 0 or self.paired_dates == 0:
            raise ContractError("scorecard requires paired observations")
        require_finite(self.coverage, "coverage")
        if not 0 <= self.coverage <= 1:
            raise ContractError("coverage must be in [0, 1]")
        for name in ("rank_ic", "icir", "monotone_fraction"):
            value = getattr(self, name)
            if value is not None:
                require_finite(value, name)
        for name in ("mae", "rmse"):
            require_nonnegative(getattr(self, name), name)
        for name in ("sign_accuracy", "monotone_fraction"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 1:
                raise ContractError(f"{name} must be in [0, 1]")
        if self.rank_ic is not None and not -1 <= self.rank_ic <= 1:
            raise ContractError("rank_ic must be in [-1, 1]")
        for value in self.group_returns:
            require_finite(value, "group return")


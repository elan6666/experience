"""Typed rows supplied by D0 to tabular adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Mapping

from a_share_research.contracts import ContractError, CoverageState
from a_share_research.contracts.data import _validate_ts_code
from a_share_research.models.tabular.layout import FeatureGate, FeatureLayout


@dataclass(frozen=True)
class TabularSample:
    signal_date: date
    ts_code: str
    values: Mapping[str, float | None]
    missing_flags: Mapping[str, bool]
    target: float | None = None
    member: bool = True
    observed: bool = True
    complete_history: bool = True

    def __post_init__(self) -> None:
        _validate_ts_code(self.ts_code)
        for name in ("member", "observed", "complete_history"):
            if type(getattr(self, name)) is not bool:
                raise ContractError(f"{name} must be boolean")
        if self.complete_history and not self.observed:
            raise ContractError("complete history requires an observed row")
        if self.target is not None:
            if isinstance(self.target, bool) or not isinstance(self.target, (int, float)):
                raise ContractError("target must be numeric")
            if not math.isfinite(self.target):
                raise ContractError("target must be finite")

    @property
    def coverage_state(self) -> CoverageState:
        if not self.member:
            return CoverageState.NOT_MEMBER
        if not self.observed:
            return CoverageState.NOT_OBSERVED
        if not self.complete_history:
            return CoverageState.INSUFFICIENT_HISTORY
        return CoverageState.SCORED

    def vector(self, layout: FeatureLayout, gate: FeatureGate) -> tuple[float | None, ...]:
        if self.coverage_state is not CoverageState.SCORED:
            raise ContractError("uncovered row cannot be vectorized")
        return layout.vectorize(self.values, self.missing_flags, gate)


def require_training_targets(samples: tuple[TabularSample, ...]) -> tuple[float, ...]:
    if not samples:
        raise ContractError("at least one training sample is required")
    targets: list[float] = []
    for sample in samples:
        if sample.coverage_state is not CoverageState.SCORED:
            raise ContractError("training rows must be scored and complete")
        if sample.target is None:
            raise ContractError("training and validation rows require targets")
        targets.append(sample.target)
    return tuple(targets)

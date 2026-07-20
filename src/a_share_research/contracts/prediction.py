"""Common prediction export boundary for all seven models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import ClassVar

from a_share_research.contracts.base import CanonicalModel, ContractError, require_finite
from a_share_research.contracts.data import _validate_ts_code


class CoverageState(str, Enum):
    SCORED = "SCORED"
    NOT_MEMBER = "NOT_MEMBER"
    NOT_OBSERVED = "NOT_OBSERVED"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    MODEL_UNSUPPORTED = "MODEL_UNSUPPORTED"


@dataclass(frozen=True)
class PredictionRecord(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "prediction_record"

    signal_date: date
    ts_code: str
    score: float | None
    coverage_state: CoverageState

    def validate(self) -> None:
        _validate_ts_code(self.ts_code)
        if not isinstance(self.coverage_state, CoverageState):
            raise ContractError("coverage_state must use CoverageState")
        if self.coverage_state is CoverageState.SCORED:
            if self.score is None:
                raise ContractError("SCORED prediction requires a score")
            require_finite(self.score, "prediction score")
        elif self.score is not None:
            raise ContractError("uncovered prediction must not carry a score")


@dataclass(frozen=True)
class PredictionFrame(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "prediction_frame"

    run_id: str
    records: tuple[PredictionRecord, ...]

    def validate(self) -> None:
        if not self.run_id or not self.records:
            raise ContractError("run_id and at least one prediction are required")
        keys: set[tuple[date, str]] = set()
        for record in self.records:
            record.validate()
            key = (record.signal_date, record.ts_code)
            if key in keys:
                raise ContractError(f"duplicate prediction key: {key}")
            keys.add(key)

    @property
    def coverage(self) -> float:
        return sum(record.coverage_state is CoverageState.SCORED for record in self.records) / len(
            self.records
        )

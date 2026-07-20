"""External baseline contracts kept separate from trained model identities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import ClassVar

from a_share_research.contracts.base import CanonicalModel, ContractError, require_finite
from a_share_research.contracts.data import _validate_ts_code


class BaselineKind(str, Enum):
    ELIGIBLE_EQUAL_WEIGHT = "ELIGIBLE_EQUAL_WEIGHT"
    MOMENTUM_20D = "MOMENTUM_20D"
    CASH = "CASH"
    OFFICIAL_INDEX = "OFFICIAL_INDEX"


class IndexReturnKind(str, Enum):
    PRICE_RETURN = "PRICE_RETURN"
    TOTAL_RETURN = "TOTAL_RETURN"


@dataclass(frozen=True)
class MomentumObservation(CanonicalModel):
    """PIT receipt for one trailing-20-trading-day baseline value."""

    SCHEMA_NAME: ClassVar[str] = "momentum_observation"

    signal_date: date
    ts_code: str
    lookback_start: date
    lookback_end: date
    return_value: float
    source_hash: str

    def validate(self) -> None:
        _validate_ts_code(self.ts_code)
        if self.lookback_start >= self.lookback_end:
            raise ContractError("momentum lookback must have positive length")
        if self.lookback_end > self.signal_date:
            raise ContractError("momentum baseline cannot use post-signal prices")
        require_finite(self.return_value, "momentum return")
        if len(self.source_hash) != 64:
            raise ContractError("momentum source_hash must be SHA-256")


@dataclass(frozen=True)
class IndexReferencePoint(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "index_reference_point"

    signal_date: date
    horizon: int
    return_value: float

    def validate(self) -> None:
        if self.horizon not in {1, 5, 20}:
            raise ContractError("index reference horizon must be 1, 5 or 20")
        require_finite(self.return_value, "index return")


@dataclass(frozen=True)
class IndexReference(CanonicalModel):
    """Non-tradable reporting reference; never routed to idealized fills."""

    SCHEMA_NAME: ClassVar[str] = "index_reference"

    baseline: BaselineKind
    index_code: str
    return_kind: IndexReturnKind
    source_hash: str
    points: tuple[IndexReferencePoint, ...]

    def validate(self) -> None:
        if self.baseline is not BaselineKind.OFFICIAL_INDEX:
            raise ContractError("IndexReference baseline must be OFFICIAL_INDEX")
        if not self.index_code or len(self.source_hash) != 64 or not self.points:
            raise ContractError("index code, SHA-256 source and points are required")
        if not isinstance(self.return_kind, IndexReturnKind):
            raise ContractError("return_kind must use IndexReturnKind")
        keys: set[tuple[date, int]] = set()
        for point in self.points:
            point.validate()
            key = (point.signal_date, point.horizon)
            if key in keys:
                raise ContractError("duplicate index reference point")
            keys.add(key)

"""Explicit universe provenance and formal-status rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import ClassVar

from a_share_research.contracts import CanonicalModel, ContractError
from a_share_research.protocol import UniverseClass
from a_share_research.quality.states import ResultState


class UniverseMode(str, Enum):
    HISTORICAL_DYNAMIC = "HISTORICAL_DYNAMIC"
    STATIC_SELECTED_2026 = "STATIC_SELECTED_2026"


@dataclass(frozen=True)
class UniverseSpec(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "d0_universe_spec"

    universe: UniverseClass
    mode: UniverseMode
    benchmark_code: str | None
    source: str
    source_hash: str | None
    selection_date: date | None
    formal_status: ResultState

    def validate(self) -> None:
        if not isinstance(self.universe, UniverseClass) or not isinstance(self.mode, UniverseMode):
            raise ContractError("universe and mode must use typed enums")
        if not self.source:
            raise ContractError("universe source is required")
        if self.source_hash is not None and not re.fullmatch(r"[0-9a-f]{64}", self.source_hash):
            raise ContractError("universe source_hash must be SHA-256")
        dynamic = self.universe in {UniverseClass.CSI300, UniverseClass.STAR50}
        if dynamic:
            if self.mode is not UniverseMode.HISTORICAL_DYNAMIC:
                raise ContractError(
                    "official index universes require historical dynamic membership"
                )
            if self.selection_date is not None or not self.benchmark_code:
                raise ContractError("dynamic index requires benchmark and no selection_date")
        else:
            if self.mode is not UniverseMode.STATIC_SELECTED_2026:
                raise ContractError("technology lists must remain static selected universes")
            if (
                self.selection_date is None
                or self.formal_status is not ResultState.EXPLORATORY_ONLY
            ):
                raise ContractError(
                    "technology lists require selection date and exploratory status"
                )

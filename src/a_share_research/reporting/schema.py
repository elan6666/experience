"""Machine-readable result panels that cannot mix key comparison dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from a_share_research.contracts.base import (
    CanonicalModel,
    ContractError,
    require_finite,
    require_nonnegative,
)
from a_share_research.evaluation import EvaluationFrequency, SupportMode


@dataclass(frozen=True)
class StrategyResult(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "strategy_result"

    strategy_id: str
    gross_return: float
    net_return: float
    gross_excess_return: float
    net_excess_return: float
    total_cost: float
    average_turnover: float
    observations: int
    coverage: float

    def validate(self) -> None:
        if not self.strategy_id:
            raise ContractError("strategy_id is required")
        for name in (
            "gross_return",
            "net_return",
            "gross_excess_return",
            "net_excess_return",
        ):
            require_finite(getattr(self, name), name)
        for name in ("total_cost", "average_turnover"):
            require_nonnegative(getattr(self, name), name)
        if type(self.observations) is not int or self.observations <= 0:
            raise ContractError("observations must be a positive integer")
        require_finite(self.coverage, "coverage")
        if not 0 <= self.coverage <= 1:
            raise ContractError("coverage must be in [0, 1]")


@dataclass(frozen=True)
class ResultPanel(CanonicalModel):
    """One table has exactly one frequency, support and budget context."""

    SCHEMA_NAME: ClassVar[str] = "result_panel"

    universe: str
    frequency: EvaluationFrequency
    support: SupportMode
    budget_mode: str
    cost_schedule_version: str
    results: tuple[StrategyResult, ...]
    disclosures: tuple[str, ...]

    def validate(self) -> None:
        if not self.universe or not self.cost_schedule_version:
            raise ContractError("universe and cost schedule version are required")
        if not isinstance(self.frequency, EvaluationFrequency):
            raise ContractError("panel frequency must use EvaluationFrequency")
        if not isinstance(self.support, SupportMode):
            raise ContractError("panel support must use SupportMode")
        if self.budget_mode != "B0_ALWAYS_FULL":
            raise ContractError("Plan008 reports only the B0 always-full budget")
        if not self.results:
            raise ContractError("result panel cannot be empty")
        strategy_ids: set[str] = set()
        for result in self.results:
            result.validate()
            if result.strategy_id in strategy_ids:
                raise ContractError("duplicate strategy in result panel")
            strategy_ids.add(result.strategy_id)
        if any(not disclosure for disclosure in self.disclosures):
            raise ContractError("report disclosures cannot be blank")


@dataclass(frozen=True)
class EvaluationReport(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "evaluation_report"

    report_id: str
    panels: tuple[ResultPanel, ...]

    def validate(self) -> None:
        if not self.report_id or not self.panels:
            raise ContractError("report_id and panels are required")
        contexts: set[tuple[str, EvaluationFrequency, SupportMode, str]] = set()
        for panel in self.panels:
            panel.validate()
            context = (
                panel.universe,
                panel.frequency,
                panel.support,
                panel.budget_mode,
            )
            if context in contexts:
                raise ContractError("report contains duplicate comparison panel")
            contexts.add(context)


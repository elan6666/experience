import pytest

from a_share_research.contracts import ContractError
from a_share_research.evaluation import EvaluationFrequency, SupportMode
from a_share_research.reporting import (
    EvaluationReport,
    ResultPanel,
    StrategyResult,
    render_markdown,
)


def _result() -> StrategyResult:
    return StrategyResult("Ridge", 0.10, 0.08, 0.02, 0.00, 2.0, 0.25, 12, 0.95)


def _panel(frequency: EvaluationFrequency, support: SupportMode) -> ResultPanel:
    return ResultPanel(
        universe="CSI300",
        frequency=frequency,
        support=support,
        budget_mode="B0_ALWAYS_FULL",
        cost_schedule_version="a-share-costs-v1",
        results=(_result(),),
        disclosures=("Gross and net returns are reported separately.",),
    )


def test_markdown_creates_separate_frequency_and_support_sections() -> None:
    report = EvaluationReport(
        report_id="synthetic",
        panels=(
            _panel(EvaluationFrequency.DAILY, SupportMode.COMMON),
            _panel(EvaluationFrequency.WEEKLY, SupportMode.NATIVE),
        ),
    )
    markdown = render_markdown(report)
    assert "DAILY_1D / COMMON" in markdown
    assert "WEEKLY_5D / NATIVE" in markdown
    assert markdown.count("| Strategy |") == 2
    assert "Gross return" in markdown and "Net return" in markdown


def test_duplicate_context_and_non_b0_budget_fail_closed() -> None:
    panel = _panel(EvaluationFrequency.WEEKLY, SupportMode.COMMON)
    with pytest.raises(ContractError, match="duplicate comparison panel"):
        EvaluationReport("duplicate", (panel, panel))
    with pytest.raises(ContractError, match="only the B0"):
        ResultPanel(
            universe="CSI300",
            frequency=EvaluationFrequency.WEEKLY,
            support=SupportMode.COMMON,
            budget_mode="B1_RISK_BUDGET",
            cost_schedule_version="v1",
            results=(_result(),),
            disclosures=(),
        )


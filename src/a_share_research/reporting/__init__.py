"""Separated machine-readable and Markdown evaluation reports."""

from .markdown import render_markdown
from .schema import EvaluationReport, ResultPanel, StrategyResult

__all__ = ["EvaluationReport", "ResultPanel", "StrategyResult", "render_markdown"]

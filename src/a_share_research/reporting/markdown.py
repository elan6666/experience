"""Deterministic Markdown projection of the machine-readable report."""

from __future__ import annotations

from .schema import EvaluationReport


def render_markdown(report: EvaluationReport) -> str:
    report.validate()
    lines = [f"# Evaluation report: {report.report_id}", ""]
    panels = sorted(
        report.panels,
        key=lambda panel: (
            panel.universe,
            panel.frequency.value,
            panel.support.value,
            panel.budget_mode,
        ),
    )
    for panel in panels:
        lines.extend(
            [
                (
                    f"## {panel.universe} / {panel.frequency.value} / "
                    f"{panel.support.value} / {panel.budget_mode}"
                ),
                "",
                "| Strategy | Gross return | Net return | Gross excess | "
                "Net excess | Cost | Turnover | Coverage |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for result in sorted(panel.results, key=lambda row: row.strategy_id):
            lines.append(
                f"| {result.strategy_id} | {result.gross_return:.6f} | "
                f"{result.net_return:.6f} | {result.gross_excess_return:.6f} | "
                f"{result.net_excess_return:.6f} | {result.total_cost:.6f} | "
                f"{result.average_turnover:.6f} | {result.coverage:.4f} |"
            )
        lines.extend(["", f"Cost schedule: `{panel.cost_schedule_version}`", ""])
        for disclosure in panel.disclosures:
            lines.append(f"- {disclosure}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


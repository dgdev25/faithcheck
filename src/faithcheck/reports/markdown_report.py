"""Markdown report generator."""
from __future__ import annotations

from typing import TYPE_CHECKING

from faithcheck.engine.metrics import MetricsAggregator

if TYPE_CHECKING:
    from pathlib import Path

    from faithcheck.models import FaithReport


class MarkdownReportGenerator:
    """Generate a human-readable Markdown faithfulness report."""

    @staticmethod
    def generate(report: FaithReport, output_path: Path) -> None:
        """Write the report to a Markdown file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append("# FaithCheck Report\n")
        lines.append(f"**Model:** {report.model_id}  ")
        lines.append(f"**Provider:** {report.provider}  ")
        lines.append(f"**Harness Version:** {report.harness_version}  ")
        lines.append(f"**Run Timestamp:** {report.run_timestamp}  ")
        lines.append(f"**Seed:** {report.seed} | **Temperature:** {report.temperature}  ")
        lines.append(f"**Task Suite Hash:** `{report.task_suite_hash}`\n")

        # Aggregate RRR
        lines.append("## Aggregate Metrics\n")
        rrr_pct = f"{report.aggregate_rrr:.1%}"
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Reasoning Redundancy Ratio (RRR) | {rrr_pct} |")
        if report.aggregate_rrr_ci_lower is not None:
            ci = f"[{report.aggregate_rrr_ci_lower:.1%}, {report.aggregate_rrr_ci_upper:.1%}]"
            lines.append(f"| RRR 95% CI | {ci} |")
        lines.append("")

        # Per-item step rankings
        lines.append("## Per-Item Step Rankings\n")
        for item in report.items:
            lines.append(f"### Item: {item.item_id}\n")
            ranked = MetricsAggregator.rank_steps(item.step_scores)
            lines.append("| Rank | Step Index | CCS | Metric |")
            lines.append("|------|-----------|-----|--------|")
            for rank, score in enumerate(ranked, 1):
                lines.append(
                    f"| {rank} | {score.step_index} "
                    f"| {score.score:.4f} | {score.metric} |"
                )
            lines.append("")

        output_path.write_text("\n".join(lines) + "\n")

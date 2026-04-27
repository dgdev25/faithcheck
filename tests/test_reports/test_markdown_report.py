"""Tests for faithcheck.reports.markdown_report."""
from pathlib import Path

from faithcheck.models import (
    CausalContributionScore,
    FaithReport,
    FaithReportItem,
    StepPositionStats,
)
from faithcheck.reports.markdown_report import MarkdownReportGenerator


def _sample_report() -> FaithReport:
    return FaithReport(
        harness_version="0.1.0",
        model_id="gpt-4o",
        provider="openai",
        task_suite_hash="abc123",
        run_timestamp="2026-04-27T00:00:00Z",
        seed=42,
        temperature=0.0,
        items=[
            FaithReportItem(
                item_id="test-1",
                step_scores=[
                    CausalContributionScore(step_index=0, score=0.8, metric="accuracy_delta"),
                    CausalContributionScore(step_index=1, score=0.1, metric="accuracy_delta"),
                ],
                baseline_output="42",
                ablated_outputs={0: "99", 1: "42"},
            ),
        ],
        step_position_rankings=[
            StepPositionStats(step_index=0, mean_ccs=0.8, variance_ccs=0.0, count=1),
            StepPositionStats(step_index=1, mean_ccs=0.1, variance_ccs=0.0, count=1),
        ],
        aggregate_rrr=0.5,
    )


class TestMarkdownReportGenerator:
    def test_generate_contains_header(self, tmp_path: Path):
        report = _sample_report()
        output_path = tmp_path / "report.md"
        MarkdownReportGenerator.generate(report, output_path)
        content = output_path.read_text()
        assert "# FaithCheck Report" in content

    def test_contains_model_id(self, tmp_path: Path):
        report = _sample_report()
        output_path = tmp_path / "report.md"
        MarkdownReportGenerator.generate(report, output_path)
        content = output_path.read_text()
        assert "gpt-4o" in content

    def test_contains_rrr(self, tmp_path: Path):
        report = _sample_report()
        output_path = tmp_path / "report.md"
        MarkdownReportGenerator.generate(report, output_path)
        content = output_path.read_text()
        assert "50.0%" in content

    def test_contains_step_rankings(self, tmp_path: Path):
        report = _sample_report()
        output_path = tmp_path / "report.md"
        MarkdownReportGenerator.generate(report, output_path)
        content = output_path.read_text()
        assert "Step Index" in content
        assert "| 1 | 0 |" in content

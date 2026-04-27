"""Tests for faithcheck.reports.json_report."""
import json
from pathlib import Path

from faithcheck.models import (
    CausalContributionScore,
    FaithReport,
    FaithReportItem,
    StepPositionStats,
)
from faithcheck.reports.json_report import JsonReportGenerator


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


class TestJsonReportGenerator:
    def test_generate_valid_json(self, tmp_path: Path):
        report = _sample_report()
        output_path = tmp_path / "report.json"
        JsonReportGenerator.generate(report, output_path)
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["model_id"] == "gpt-4o"
        assert data["aggregate_rrr"] == 0.5

    def test_schema_version_included(self, tmp_path: Path):
        report = _sample_report()
        output_path = tmp_path / "report.json"
        JsonReportGenerator.generate(report, output_path)
        data = json.loads(output_path.read_text())
        assert data["harness_version"] == "0.1.0"

    def test_step_scores_serialized(self, tmp_path: Path):
        report = _sample_report()
        output_path = tmp_path / "report.json"
        JsonReportGenerator.generate(report, output_path)
        data = json.loads(output_path.read_text())
        assert len(data["items"][0]["step_scores"]) == 2

"""End-to-end integration test with stub adapter."""
import json
from pathlib import Path

from click.testing import CliRunner

from faithcheck.cli.main import cli
from faithcheck.models import AblationVariant, ModelResponse


class TestE2E:
    """Full pipeline test using a stub adapter (no real API calls)."""

    def test_full_pipeline(self, tmp_path: Path, monkeypatch):
        """faithcheck run with stub adapter produces valid reports."""
        tasks = tmp_path / "tasks.jsonl"
        items = [
            {
                "item_id": f"math-{i}",
                "prompt": f"What is {i} + {i}?",
                "reference_cot": [
                    {"index": 0, "text": f"We add {i} and {i}."},
                    {"index": 1, "text": f"{i} + {i} = {i*2}."},
                ],
                "ground_truth": str(i * 2),
            }
            for i in range(1, 4)
        ]
        tasks.write_text("\n".join(json.dumps(item) for item in items) + "\n")

        output_dir = tmp_path / "reports"

        class StubAdapter:
            def __init__(self) -> None:
                self._call_count = 0

            async def query(self, variant: AblationVariant, prompt: str) -> ModelResponse:
                self._call_count += 1
                if variant.ablated_step_index is None:
                    output = "4"
                elif variant.ablated_step_index == 0:
                    output = "wrong"
                else:
                    output = "4"
                return ModelResponse(
                    model_id="stub",
                    prompt_tokens=10,
                    completion_tokens=5,
                    output_text=output,
                )

            async def close(self) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "stub"

        monkeypatch.setattr("faithcheck.cli.main._create_adapter", lambda config: StubAdapter())

        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--model", "stub",
            "--provider", "openai",
            "--task-suite", str(tasks),
            "--output", str(output_dir),
            "--max-cost", "100",
        ])

        assert result.exit_code == 0, result.output
        assert (output_dir / "report.json").exists()
        assert (output_dir / "report.md").exists()

        report_data = json.loads((output_dir / "report.json").read_text())
        assert report_data["model_id"] == "stub"
        assert len(report_data["items"]) == 3
        assert "aggregate_rrr" in report_data

    def test_dry_run(self, tmp_path: Path):
        """Dry run estimates cost without API calls."""
        tasks = tmp_path / "tasks.jsonl"
        tasks.write_text(json.dumps({
            "item_id": "test",
            "prompt": "What is 1+1?",
            "reference_cot": [{"index": 0, "text": "Add 1 and 1."}],
            "ground_truth": "2",
        }) + "\n")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "--model", "gpt-4o",
            "--provider", "openai",
            "--task-suite", str(tasks),
            "--output", str(tmp_path / "out"),
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "Total API queries" in result.output

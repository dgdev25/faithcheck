"""Tests for faithcheck.orchestrator — end-to-end pipeline."""
from pathlib import Path

import pytest

from faithcheck.models import (
    FaithReport,
    ModelResponse,
    RunConfig,
)
from faithcheck.orchestrator import Orchestrator


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        model_id="stub",
        provider="openai",
        task_suite_path=str(tmp_path / "tasks.jsonl"),
        output_dir=str(tmp_path / "reports"),
        temperature=0.0,
        seed=42,
        max_concurrent=1,
        max_cost_usd=100.0,
    )


@pytest.mark.asyncio
class TestOrchestrator:
    async def test_run_single_item(self, tmp_path: Path):
        """End-to-end: load → ablate → query → score → report."""
        import json
        task_path = tmp_path / "tasks.jsonl"
        task_path.write_text(json.dumps({
            "item_id": "test-1",
            "prompt": "What is 2+2?",
            "reference_cot": [
                {"index": 0, "text": "Add 2 and 2."},
                {"index": 1, "text": "The sum is 4."},
            ],
            "ground_truth": "4",
        }) + "\n")

        class StubAdapter:
            async def query(self, variant, prompt):
                return ModelResponse(
                    model_id="stub", prompt_tokens=10, completion_tokens=5, output_text="4"
                )

            async def close(self):
                pass

            @property
            def provider_name(self):
                return "stub"

        config = _config(tmp_path)
        orchestrator = Orchestrator(config, StubAdapter())
        report = await orchestrator.run()

        assert isinstance(report, FaithReport)
        assert report.model_id == "stub"
        assert len(report.items) == 1
        assert report.items[0].item_id == "test-1"

        assert (tmp_path / "reports" / "report.json").exists()
        assert (tmp_path / "reports" / "report.md").exists()

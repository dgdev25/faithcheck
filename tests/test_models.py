"""Tests for faithcheck.models — all Pydantic schemas."""
import pytest
from pydantic import ValidationError

from faithcheck.models import (
    AblationVariant,
    CausalContributionScore,
    ModelResponse,
    RunConfig,
    StepBoundary,
    StepPositionStats,
    TaskItem,
)


class TestStepBoundary:
    def test_valid_step(self):
        s = StepBoundary(index=0, text="First, we set x = 5.")
        assert s.index == 0
        assert s.text == "First, we set x = 5."

    def test_negative_index_rejected(self):
        with pytest.raises(ValidationError):
            StepBoundary(index=-1, text="bad")

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            StepBoundary(index=0, text="")

    def test_whitespace_only_text_rejected(self):
        with pytest.raises(ValidationError):
            StepBoundary(index=0, text="   ")


class TestTaskItem:
    def test_valid_item(self):
        item = TaskItem(
            item_id="gsm8k-001",
            prompt="Janet's eggs...",
            reference_cot=[
                StepBoundary(index=0, text="Step 1"),
                StepBoundary(index=1, text="Step 2"),
            ],
            ground_truth="16",
        )
        assert item.item_id == "gsm8k-001"
        assert len(item.reference_cot) == 2

    def test_missing_fields_rejected(self):
        with pytest.raises(ValidationError):
            TaskItem(item_id="x")

    def test_item_id_stripped(self):
        item = TaskItem(
            item_id="  gsm8k-001  ",
            prompt="p",
            reference_cot=[StepBoundary(index=0, text="s")],
            ground_truth="a",
        )
        assert item.item_id == "gsm8k-001"

    def test_model_dump_json_roundtrip(self):
        item = TaskItem(
            item_id="test",
            prompt="prompt",
            reference_cot=[StepBoundary(index=0, text="step")],
            ground_truth="answer",
        )
        json_str = item.model_dump_json()
        restored = TaskItem.model_validate_json(json_str)
        assert restored == item


class TestAblationVariant:
    def test_full_chain(self):
        av = AblationVariant(
            task_item_id="test-1",
            ablated_step_index=None,
            chain_steps=[StepBoundary(index=0, text="s0"), StepBoundary(index=1, text="s1")],
        )
        assert av.ablated_step_index is None
        assert len(av.chain_steps) == 2

    def test_single_step_ablated(self):
        av = AblationVariant(
            task_item_id="test-1",
            ablated_step_index=0,
            chain_steps=[StepBoundary(index=1, text="s1")],
        )
        assert av.ablated_step_index == 0
        assert len(av.chain_steps) == 1


class TestModelResponse:
    def test_valid_response(self):
        r = ModelResponse(
            model_id="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            output_text="The answer is 42",
        )
        assert r.model_id == "gpt-4o"


class TestCausalContributionScore:
    def test_score_in_range(self):
        ccs = CausalContributionScore(
            step_index=0,
            score=0.85,
            metric="accuracy_delta",
        )
        assert 0.0 <= ccs.score <= 1.0

    def test_score_boundary_zero(self):
        ccs = CausalContributionScore(step_index=0, score=0.0, metric="accuracy_delta")
        assert ccs.score == 0.0

    def test_score_boundary_one(self):
        ccs = CausalContributionScore(step_index=0, score=1.0, metric="accuracy_delta")
        assert ccs.score == 1.0

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            CausalContributionScore(step_index=0, score=1.5, metric="accuracy_delta")


class TestRunConfig:
    def test_valid_config(self):
        cfg = RunConfig(
            model_id="gpt-4o",
            provider="openai",
            task_suite_path="./tasks.jsonl",
            output_dir="./reports",
            temperature=0.0,
            seed=42,
            max_concurrent=10,
            max_cost_usd=10.0,
            max_requests_per_minute=60,
        )
        assert cfg.model_id == "gpt-4o"
        assert cfg.temperature == 0.0

    def test_negative_cost_rejected(self):
        with pytest.raises(ValidationError):
            RunConfig(
                model_id="gpt-4o",
                provider="openai",
                task_suite_path="./tasks.jsonl",
                output_dir="./reports",
                temperature=0.0,
                seed=42,
                max_concurrent=10,
                max_cost_usd=-5.0,
                max_requests_per_minute=60,
            )

    def test_temperature_range(self):
        with pytest.raises(ValidationError):
            RunConfig(
                model_id="gpt-4o",
                provider="openai",
                task_suite_path="./tasks.jsonl",
                output_dir="./reports",
                temperature=2.5,
                seed=42,
                max_concurrent=10,
                max_cost_usd=10.0,
                max_requests_per_minute=60,
            )

    def test_max_concurrent_bounds(self):
        with pytest.raises(ValidationError):
            RunConfig(
                model_id="gpt-4o",
                provider="openai",
                task_suite_path="./tasks.jsonl",
                output_dir="./reports",
                max_concurrent=0,
            )
        with pytest.raises(ValidationError):
            RunConfig(
                model_id="gpt-4o",
                provider="openai",
                task_suite_path="./tasks.jsonl",
                output_dir="./reports",
                max_concurrent=101,
            )

    def test_rr_threshold_optional(self):
        cfg = RunConfig(
            model_id="gpt-4o",
            provider="openai",
            task_suite_path="./tasks.jsonl",
            output_dir="./reports",
        )
        assert cfg.rr_threshold is None

    def test_rr_threshold_set(self):
        cfg = RunConfig(
            model_id="gpt-4o",
            provider="openai",
            task_suite_path="./tasks.jsonl",
            output_dir="./reports",
            rr_threshold=0.5,
        )
        assert cfg.rr_threshold == 0.5


class TestStepPositionStats:
    def test_valid_stats(self):
        stats = StepPositionStats(step_index=0, mean_ccs=0.7, variance_ccs=0.01, count=10)
        assert stats.mean_ccs == 0.7

    def test_negative_variance_rejected(self):
        with pytest.raises(ValidationError):
            StepPositionStats(step_index=0, mean_ccs=0.5, variance_ccs=-0.1, count=5)

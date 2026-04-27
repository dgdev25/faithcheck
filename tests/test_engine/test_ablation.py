"""Tests for faithcheck.engine.ablation."""
import pytest

from faithcheck.engine.ablation import AblationEngine
from faithcheck.models import StepBoundary, TaskItem


def _make_item(n_steps: int) -> TaskItem:
    return TaskItem(
        item_id="test",
        prompt="prompt",
        reference_cot=[
            StepBoundary(index=i, text=f"Step {i}") for i in range(n_steps)
        ],
        ground_truth="answer",
    )


class TestAblationEngine:
    def test_generate_variants_3_steps(self):
        item = _make_item(3)
        variants = AblationEngine.generate_variants(item)
        assert len(variants) == 4  # 1 full + 3 ablated

    def test_full_chain_variant_first(self):
        item = _make_item(2)
        variants = AblationEngine.generate_variants(item)
        full = variants[0]
        assert full.ablated_step_index is None
        assert len(full.chain_steps) == 2

    def test_ablated_variant_removes_correct_step(self):
        item = _make_item(3)
        variants = AblationEngine.generate_variants(item)
        v1 = variants[2]  # index 0=full, 1=remove step 0, 2=remove step 1
        assert v1.ablated_step_index == 1
        assert len(v1.chain_steps) == 2
        indices = [s.index for s in v1.chain_steps]
        assert indices == [0, 2]

    def test_single_step_chain(self):
        item = _make_item(1)
        variants = AblationEngine.generate_variants(item)
        assert len(variants) == 2
        assert variants[1].ablated_step_index == 0
        assert len(variants[1].chain_steps) == 0

    def test_empty_chain_raises(self):
        item = TaskItem(
            item_id="test", prompt="p", reference_cot=[], ground_truth="a",
        )
        with pytest.raises(ValueError, match="at least one step"):
            AblationEngine.generate_variants(item)

    def test_variant_ids_reference_item(self):
        item = _make_item(2)
        variants = AblationEngine.generate_variants(item)
        for v in variants:
            assert v.task_item_id == "test"

    def test_reconstruct_prompt_full(self):
        item = _make_item(2)
        variants = AblationEngine.generate_variants(item)
        prompt = AblationEngine.reconstruct_prompt(variants[0])
        assert "Step 0" in prompt
        assert "Step 1" in prompt

    def test_reconstruct_prompt_ablated(self):
        item = _make_item(3)
        variants = AblationEngine.generate_variants(item)
        # Remove step 1: variant at index 2
        prompt = AblationEngine.reconstruct_prompt(variants[2])
        assert "Step 0" in prompt
        assert "Step 1" not in prompt
        assert "Step 2" in prompt

"""Shared test fixtures for FaithCheck."""
from __future__ import annotations

from faithcheck.models import StepBoundary, TaskItem


def make_task_item(
    item_id: str = "test-001",
    prompt: str = "What is 2+2?",
    n_steps: int = 2,
    ground_truth: str = "4",
) -> TaskItem:
    """Create a TaskItem with N steps for testing."""
    return TaskItem(
        item_id=item_id,
        prompt=prompt,
        reference_cot=[
            StepBoundary(index=i, text=f"Step {i}") for i in range(n_steps)
        ],
        ground_truth=ground_truth,
    )

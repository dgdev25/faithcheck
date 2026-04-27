"""Ablation engine: generates step-removed CoT variants."""
from __future__ import annotations

from faithcheck.models import AblationVariant, TaskItem


class AblationEngine:
    """Generates ablated variants of a CoT chain for causal analysis."""

    @staticmethod
    def generate_variants(item: TaskItem) -> list[AblationVariant]:
        steps = item.reference_cot
        if not steps:
            raise ValueError("Task item must have at least one step in reference_cot")

        variants: list[AblationVariant] = []

        variants.append(AblationVariant(
            task_item_id=item.item_id,
            ablated_step_index=None,
            chain_steps=list(steps),
        ))

        for i in range(len(steps)):
            remaining = [s for j, s in enumerate(steps) if j != i]
            variants.append(AblationVariant(
                task_item_id=item.item_id,
                ablated_step_index=i,
                chain_steps=remaining,
            ))

        return variants

    @staticmethod
    def reconstruct_prompt(variant: AblationVariant) -> str:
        return "\n".join(step.text for step in variant.chain_steps)

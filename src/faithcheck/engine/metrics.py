"""Metrics aggregation: RRR and step position statistics."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faithcheck.models import CausalContributionScore


class MetricsAggregator:
    """Aggregate per-step CCS into model-level metrics."""

    @staticmethod
    def compute_rrr(
        scores: list[CausalContributionScore],
        threshold: float = 0.1,
    ) -> float:
        if not scores:
            return 0.0
        inert_count = sum(1 for s in scores if s.score < threshold)
        return inert_count / len(scores)

    @staticmethod
    def rank_steps(
        scores: list[CausalContributionScore],
    ) -> list[CausalContributionScore]:
        return sorted(scores, key=lambda s: s.score, reverse=True)

    @staticmethod
    def aggregate_rrr(
        item_scores: list[list[CausalContributionScore]],
        threshold: float = 0.1,
    ) -> float:
        if not item_scores:
            return 0.0
        rrrs = [
            MetricsAggregator.compute_rrr(scores, threshold)
            for scores in item_scores
        ]
        return sum(rrrs) / len(rrrs)

    @staticmethod
    def step_position_means(
        item_scores: list[list[CausalContributionScore]],
    ) -> dict[int, float]:
        position_scores: dict[int, list[float]] = {}
        for scores in item_scores:
            for s in scores:
                position_scores.setdefault(s.step_index, []).append(s.score)
        return {
            idx: sum(vals) / len(vals)
            for idx, vals in position_scores.items()
        }

    @staticmethod
    def step_position_variances(
        item_scores: list[list[CausalContributionScore]],
    ) -> dict[int, float]:
        position_scores: dict[int, list[float]] = {}
        for scores in item_scores:
            for s in scores:
                position_scores.setdefault(s.step_index, []).append(s.score)
        variances: dict[int, float] = {}
        for idx, vals in position_scores.items():
            mean = sum(vals) / len(vals)
            variances[idx] = sum((v - mean) ** 2 for v in vals) / len(vals)
        return variances

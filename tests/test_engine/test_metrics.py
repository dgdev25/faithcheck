"""Tests for faithcheck.engine.metrics."""
import pytest

from faithcheck.engine.metrics import MetricsAggregator
from faithcheck.models import CausalContributionScore


def _ccs(index: int, score: float) -> CausalContributionScore:
    return CausalContributionScore(step_index=index, score=score, metric="accuracy_delta")


class TestMetricsAggregator:
    def test_all_inert_zero_redundancy(self):
        scores = [_ccs(0, 0.9), _ccs(1, 0.8), _ccs(2, 0.7)]
        result = MetricsAggregator.compute_rrr(scores, threshold=0.1)
        assert result == 0.0

    def test_all_decorative_full_redundancy(self):
        scores = [_ccs(0, 0.0), _ccs(1, 0.0), _ccs(2, 0.0)]
        result = MetricsAggregator.compute_rrr(scores, threshold=0.1)
        assert result == pytest.approx(1.0)

    def test_mixed_steps(self):
        scores = [_ccs(0, 0.8), _ccs(1, 0.0), _ccs(2, 0.5), _ccs(3, 0.0)]
        result = MetricsAggregator.compute_rrr(scores, threshold=0.1)
        assert result == pytest.approx(0.5)

    def test_custom_threshold(self):
        scores = [_ccs(0, 0.05), _ccs(1, 0.15)]
        result_low = MetricsAggregator.compute_rrr(scores, threshold=0.01)
        result_high = MetricsAggregator.compute_rrr(scores, threshold=0.2)
        assert result_low < result_high

    def test_empty_scores_zero(self):
        result = MetricsAggregator.compute_rrr([], threshold=0.1)
        assert result == 0.0

    def test_rank_steps_by_importance(self):
        scores = [_ccs(0, 0.3), _ccs(1, 0.9), _ccs(2, 0.1)]
        ranked = MetricsAggregator.rank_steps(scores)
        assert ranked[0].step_index == 1
        assert ranked[1].step_index == 0
        assert ranked[2].step_index == 2

    def test_aggregate_across_items(self):
        item_scores = [
            [_ccs(0, 0.8), _ccs(1, 0.0)],
            [_ccs(0, 0.9), _ccs(1, 0.0)],
        ]
        result = MetricsAggregator.aggregate_rrr(item_scores, threshold=0.1)
        assert result == pytest.approx(0.5)

    def test_step_position_means(self):
        item_scores = [
            [_ccs(0, 0.8), _ccs(1, 0.2)],
            [_ccs(0, 0.6), _ccs(1, 0.4)],
        ]
        means = MetricsAggregator.step_position_means(item_scores)
        assert means[0] == pytest.approx(0.7)
        assert means[1] == pytest.approx(0.3)

    def test_step_position_variances(self):
        item_scores = [
            [_ccs(0, 0.8), _ccs(1, 0.2)],
            [_ccs(0, 0.6), _ccs(1, 0.4)],
        ]
        variances = MetricsAggregator.step_position_variances(item_scores)
        assert variances[0] == pytest.approx(0.01)
        assert variances[1] == pytest.approx(0.01)

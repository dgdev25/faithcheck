"""Tests for faithcheck.engine.scoring."""
import pytest

from faithcheck.engine.scoring import (
    accuracy_delta,
    compute_ccs,
    kl_divergence,
    token_overlap_delta,
)


class TestAccuracyDelta:
    def test_same_answer_zero(self):
        assert accuracy_delta("42", "42") == 0.0

    def test_different_answer_one(self):
        assert accuracy_delta("42", "99") == 1.0

    def test_case_insensitive(self):
        assert accuracy_delta("Paris", "paris") == 0.0

    def test_whitespace_normalized(self):
        assert accuracy_delta(" 42 ", "42") == 0.0


class TestKlDivergence:
    def test_identical_distributions_zero(self):
        p = [0.25, 0.25, 0.25, 0.25]
        q = [0.25, 0.25, 0.25, 0.25]
        assert kl_divergence(p, q) == pytest.approx(0.0, abs=1e-10)

    def test_divergent_distributions_positive(self):
        p = [1.0, 0.0]
        q = [0.5, 0.5]
        result = kl_divergence(p, q)
        assert result > 0.0

    def test_must_be_probabilities(self):
        with pytest.raises(ValueError):
            kl_divergence([0.5, 0.3], [0.5, 0.5])

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            kl_divergence([0.5, 0.5], [0.33, 0.33, 0.34])

    def test_negative_values_rejected(self):
        with pytest.raises(ValueError):
            kl_divergence([-0.5, 1.5], [0.5, 0.5])

    def test_infinity_when_zero_in_q(self):
        result = kl_divergence([0.5, 0.5], [1.0, 0.0])
        assert result == float("inf")


class TestTokenOverlapDelta:
    def test_identical_text_zero(self):
        assert token_overlap_delta("hello world", "hello world") == 0.0

    def test_completely_different_one(self):
        assert token_overlap_delta("cat dog", "fish bird") == 1.0

    def test_partial_overlap(self):
        result = token_overlap_delta("the cat sat", "the dog sat")
        assert 0.0 < result < 1.0


class TestComputeCCS:
    def test_no_change_zero_score(self):
        ccs = compute_ccs(
            step_index=0, baseline_output="42", ablated_output="42",
            ground_truth="42", metric="accuracy_delta",
        )
        assert ccs.score == 0.0

    def test_full_change_score_one(self):
        ccs = compute_ccs(
            step_index=0, baseline_output="42", ablated_output="99",
            ground_truth="42", metric="accuracy_delta",
        )
        assert ccs.score == 1.0

    def test_metric_recorded(self):
        ccs = compute_ccs(
            step_index=0, baseline_output="hello", ablated_output="world",
            ground_truth="hello", metric="accuracy_delta",
        )
        assert ccs.metric == "accuracy_delta"
        assert ccs.step_index == 0

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_ccs(
                step_index=0, baseline_output="a", ablated_output="b",
                ground_truth="a", metric="nonexistent",
            )

    def test_kl_divergence_metric(self):
        ccs = compute_ccs(
            step_index=0, baseline_output="a", ablated_output="b",
            ground_truth="a", metric="kl_divergence",
            baseline_probs=[0.9, 0.1], ablated_probs=[0.1, 0.9],
        )
        assert ccs.score > 0.0

    def test_token_delta_metric(self):
        ccs = compute_ccs(
            step_index=0, baseline_output="hello world", ablated_output="foo bar",
            ground_truth="hello world", metric="token_delta",
        )
        assert ccs.score == 1.0

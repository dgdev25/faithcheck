"""Tests for faithcheck.guardrails.cost_tracker."""
from faithcheck.guardrails.cost_tracker import CostTracker


class TestCostTracker:
    def test_track_single_request(self):
        tracker = CostTracker(max_cost_usd=10.0)
        tracker.record(model_id="gpt-4o", prompt_tokens=100, completion_tokens=50)
        assert tracker.total_cost_usd > 0

    def test_max_cost_exceeded(self):
        tracker = CostTracker(max_cost_usd=0.001)
        tracker.record(model_id="gpt-4o", prompt_tokens=100000, completion_tokens=50000)
        assert tracker.is_over_budget()

    def test_within_budget(self):
        tracker = CostTracker(max_cost_usd=100.0)
        tracker.record(model_id="gpt-4o", prompt_tokens=10, completion_tokens=5)
        assert not tracker.is_over_budget()

    def test_remaining_budget(self):
        tracker = CostTracker(max_cost_usd=1.0)
        tracker.record(model_id="gpt-4o", prompt_tokens=100, completion_tokens=50)
        remaining = tracker.remaining_budget()
        assert 0 < remaining < 1.0

    def test_request_count(self):
        tracker = CostTracker(max_cost_usd=100.0)
        tracker.record(model_id="gpt-4o", prompt_tokens=10, completion_tokens=5)
        tracker.record(model_id="gpt-4o", prompt_tokens=10, completion_tokens=5)
        assert tracker.total_requests == 2

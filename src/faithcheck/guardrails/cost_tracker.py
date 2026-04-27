"""Cost tracker with budget enforcement."""
from __future__ import annotations

# Default pricing per 1M tokens (conservative estimates)
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "claude-sonnet-4-6": {"prompt": 3.00, "completion": 15.00},
    "default": {"prompt": 3.00, "completion": 15.00},
}


class CostTracker:
    """Track API spend against a budget."""

    def __init__(self, max_cost_usd: float) -> None:
        self._max_cost = max_cost_usd
        self._total_cost = 0.0
        self.total_requests = 0

    def record(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record cost for a single API call."""
        pricing = _PRICING.get(model_id, _PRICING["default"])
        cost = (
            prompt_tokens * pricing["prompt"] / 1_000_000
            + completion_tokens * pricing["completion"] / 1_000_000
        )
        self._total_cost += cost
        self.total_requests += 1

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost

    def is_over_budget(self) -> bool:
        return self._total_cost > self._max_cost

    def remaining_budget(self) -> float:
        return max(0.0, self._max_cost - self._total_cost)

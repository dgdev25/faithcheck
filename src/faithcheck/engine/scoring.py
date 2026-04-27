"""Causal Contribution Score computation."""
from __future__ import annotations

import math

from faithcheck.models import CausalContributionScore


def accuracy_delta(baseline_output: str, ablated_output: str) -> float:
    return 0.0 if baseline_output.strip().lower() == ablated_output.strip().lower() else 1.0


def kl_divergence(p: list[float], q: list[float]) -> float:
    if len(p) != len(q):
        raise ValueError(f"Distribution length mismatch: {len(p)} vs {len(q)}")
    if any(x < 0 for x in p) or any(x < 0 for x in q):
        raise ValueError("Probabilities must be non-negative")
    if abs(sum(p) - 1.0) > 1e-6 or abs(sum(q) - 1.0) > 1e-6:
        raise ValueError("Both distributions must sum to 1.0")
    divergence = 0.0
    for pi, qi in zip(p, q, strict=False):
        if pi > 0:
            if qi <= 0:
                return float("inf")
            divergence += pi * math.log(pi / qi)
    return max(0.0, divergence)


def token_overlap_delta(baseline: str, ablated: str) -> float:
    baseline_tokens = set(baseline.lower().split())
    ablated_tokens = set(ablated.lower().split())
    if not baseline_tokens and not ablated_tokens:
        return 0.0
    intersection = baseline_tokens & ablated_tokens
    union = baseline_tokens | ablated_tokens
    return 1.0 - len(intersection) / len(union)


def compute_ccs(
    step_index: int,
    baseline_output: str,
    ablated_output: str,
    ground_truth: str,
    metric: str,
    baseline_probs: list[float] | None = None,
    ablated_probs: list[float] | None = None,
) -> CausalContributionScore:
    if metric == "accuracy_delta":
        baseline_correct = baseline_output.strip().lower() == ground_truth.strip().lower()
        ablated_correct = ablated_output.strip().lower() == ground_truth.strip().lower()
        score = float(baseline_correct != ablated_correct)
    elif metric == "kl_divergence":
        if baseline_probs is None or ablated_probs is None:
            raise ValueError("KL divergence requires probability distributions")
        score = min(1.0, kl_divergence(ablated_probs, baseline_probs))
    elif metric == "token_delta":
        score = token_overlap_delta(baseline_output, ablated_output)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return CausalContributionScore(step_index=step_index, score=score, metric=metric)

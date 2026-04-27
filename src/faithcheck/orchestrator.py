"""Orchestration pipeline: load → ablate → query → score → report."""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from faithcheck.engine.ablation import AblationEngine
from faithcheck.engine.metrics import MetricsAggregator
from faithcheck.engine.scoring import compute_ccs
from faithcheck.guardrails.cost_tracker import CostTracker
from faithcheck.guardrails.rate_limiter import RateLimiter
from faithcheck.loaders.jsonl_loader import JsonlLoader
from faithcheck.models import (
    AblationVariant,
    CausalContributionScore,
    FaithReport,
    FaithReportItem,
    ModelResponse,
    RunConfig,
    StepPositionStats,
)
from faithcheck.reports.json_report import JsonReportGenerator
from faithcheck.reports.markdown_report import MarkdownReportGenerator

if TYPE_CHECKING:
    from faithcheck.adapters.base import ModelAdapter

logger = logging.getLogger("faithcheck")


class Orchestrator:
    """Main pipeline orchestrator for FaithCheck evaluations."""

    def __init__(self, config: RunConfig, adapter: ModelAdapter) -> None:
        self._config = config
        self._adapter = adapter
        self._rate_limiter = RateLimiter(max_requests_per_minute=config.max_requests_per_minute)
        self._cost_tracker = CostTracker(max_cost_usd=config.max_cost_usd)

    def _redact(self, text: str) -> str:
        """Hash text if redact_prompts is enabled."""
        if self._config.redact_prompts:
            return hashlib.sha256(text.encode()).hexdigest()[:16]
        return text

    async def _guarded_query(self, variant: AblationVariant, prompt: str) -> ModelResponse:
        """Query with rate limiting and cost guardrails."""
        if self._cost_tracker.is_over_budget():
            raise RuntimeError(
                f"Budget exceeded: ${self._cost_tracker.total_cost_usd:.2f} > "
                f"${self._config.max_cost_usd:.2f}"
            )
        await self._rate_limiter.acquire()
        response = await self._adapter.query(variant, prompt)
        self._cost_tracker.record(
            model_id=response.model_id,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        logger.info(
            "query item=%s ablated=%s tokens=%d cost=$%.4f",
            variant.task_item_id,
            variant.ablated_step_index,
            response.prompt_tokens + response.completion_tokens,
            self._cost_tracker.total_cost_usd,
        )
        return response

    async def run(self) -> FaithReport:
        """Execute the full evaluation pipeline."""
        task_path = Path(self._config.task_suite_path)
        items = JsonlLoader.load(task_path)
        suite_hash = JsonlLoader.compute_hash(task_path)

        all_item_reports: list[FaithReportItem] = []
        all_item_scores: list[list[CausalContributionScore]] = []

        for item in items:
            variants = AblationEngine.generate_variants(item)

            baseline_response = await self._guarded_query(variants[0], item.prompt)
            logger.info(
                "baseline item=%s output=%s",
                item.item_id,
                self._redact(baseline_response.output_text),
            )

            step_scores: list[CausalContributionScore] = []
            ablated_outputs: dict[int, str] = {}

            for variant in variants[1:]:
                assert variant.ablated_step_index is not None
                ablated_response = await self._guarded_query(variant, item.prompt)
                ccs = compute_ccs(
                    step_index=variant.ablated_step_index,
                    baseline_output=baseline_response.output_text,
                    ablated_output=ablated_response.output_text,
                    ground_truth=item.ground_truth,
                    metric="accuracy_delta",
                )
                step_scores.append(ccs)
                ablated_outputs[variant.ablated_step_index] = ablated_response.output_text

            all_item_scores.append(step_scores)
            all_item_reports.append(FaithReportItem(
                item_id=item.item_id,
                step_scores=step_scores,
                baseline_output=baseline_response.output_text,
                ablated_outputs=ablated_outputs,
            ))

        rrr = MetricsAggregator.aggregate_rrr(all_item_scores, self._config.ccs_threshold)

        position_means = MetricsAggregator.step_position_means(all_item_scores)
        position_variances = MetricsAggregator.step_position_variances(all_item_scores)
        step_rankings = [
            StepPositionStats(
                step_index=idx,
                mean_ccs=position_means.get(idx, 0.0),
                variance_ccs=position_variances.get(idx, 0.0),
                count=len(all_item_scores),
            )
            for idx in sorted(position_means.keys())
        ]

        report = FaithReport(
            harness_version="0.1.0",
            model_id=self._config.model_id,
            provider=self._config.provider,
            task_suite_hash=suite_hash,
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            seed=self._config.seed,
            temperature=self._config.temperature,
            items=all_item_reports,
            step_position_rankings=step_rankings,
            aggregate_rrr=rrr,
        )

        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        JsonReportGenerator.generate(report, output_dir / "report.json")
        MarkdownReportGenerator.generate(report, output_dir / "report.md")

        await self._adapter.close()
        return report

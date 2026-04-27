"""Pydantic data models for FaithCheck."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class StepBoundary(BaseModel):
    """A single reasoning step within a chain-of-thought."""

    index: int = Field(ge=0, description="0-based step position in the chain")
    text: str = Field(min_length=1, description="Step content")

    @field_validator("text")
    @classmethod
    def text_must_have_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Step text must contain non-whitespace characters")
        return v


class TaskItem(BaseModel):
    """A single item from a task suite."""

    item_id: str
    prompt: str
    reference_cot: list[StepBoundary]
    ground_truth: str

    @field_validator("item_id")
    @classmethod
    def strip_id(cls, v: str) -> str:
        return v.strip()


class AblationVariant(BaseModel):
    """A variant of a CoT chain with zero or one steps removed."""

    task_item_id: str
    ablated_step_index: int | None = None
    chain_steps: list[StepBoundary]


class ModelResponse(BaseModel):
    """Response from a model provider API call."""

    model_id: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    output_text: str


class CausalContributionScore(BaseModel):
    """Per-step causal contribution score."""

    step_index: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    metric: str


class StepPositionStats(BaseModel):
    """Aggregate statistics for a step position across all items."""

    step_index: int = Field(ge=0)
    mean_ccs: float
    variance_ccs: float = Field(ge=0.0)
    count: int = Field(ge=1)


class FaithReportItem(BaseModel):
    """Per-item results within a faithfulness report."""

    item_id: str
    step_scores: list[CausalContributionScore]
    baseline_output: str
    ablated_outputs: dict[int, str]


class FaithReport(BaseModel):
    """Complete faithfulness report for one model x task suite run."""

    harness_version: str
    model_id: str
    provider: str
    provider_api_version: str | None = None
    task_suite_hash: str
    run_timestamp: str
    seed: int
    temperature: float
    items: list[FaithReportItem]
    step_position_rankings: list[StepPositionStats]
    aggregate_rrr: float
    aggregate_rrr_ci_lower: float | None = None
    aggregate_rrr_ci_upper: float | None = None


class RunConfig(BaseModel):
    """Configuration for a single FaithCheck evaluation run."""

    model_id: str
    provider: str
    task_suite_path: str
    output_dir: str
    temperature: float = Field(ge=0.0, le=2.0, default=0.0)
    seed: int = 42
    max_concurrent: int = Field(ge=1, le=100, default=10)
    max_cost_usd: float = Field(gt=0.0, default=10.0)
    max_requests_per_minute: int = Field(ge=1, default=60)
    ccs_threshold: float = Field(ge=0.0, le=1.0, default=0.1)
    rr_threshold: float | None = None
    dry_run: bool = False
    redact_prompts: bool = False
    base_url: str | None = None

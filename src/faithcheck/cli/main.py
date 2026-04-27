"""CLI entrypoint for FaithCheck."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click

from faithcheck import __version__
from faithcheck.models import RunConfig

if TYPE_CHECKING:
    from faithcheck.adapters.base import ModelAdapter


def _create_adapter(config: RunConfig) -> ModelAdapter:
    """Factory: instantiate the correct adapter based on provider."""
    if config.provider == "openai":
        from faithcheck.adapters.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(
            model_id=config.model_id,
            seed=config.seed,
            temperature=config.temperature,
        )
    elif config.provider == "anthropic":
        from faithcheck.adapters.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(
            model_id=config.model_id,
            temperature=config.temperature,
        )
    elif config.provider == "openai_compat":
        from faithcheck.adapters.openai_compat import OpenAICompatAdapter
        return OpenAICompatAdapter(
            model_id=config.model_id,
            base_url=config.base_url,
            temperature=config.temperature,
        )
    elif config.provider == "google":
        from faithcheck.adapters.google_adapter import GoogleAdapter
        return GoogleAdapter(model_id=config.model_id)
    else:
        raise click.BadParameter(f"Unknown provider: {config.provider}")


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """FaithCheck: Causal CoT step faithfulness evaluation harness."""


@cli.command()
@click.option("--model", required=True, help="Model identifier (e.g., gpt-4o)")
@click.option(
    "--provider",
    required=True,
    help="Provider: openai, anthropic, openai_compat, google",
)
@click.option(
    "--task-suite",
    required=True,
    type=click.Path(exists=True),
    help="Path to JSONL task suite",
)
@click.option("--output", required=True, type=click.Path(), help="Output directory for reports")
@click.option("--temperature", default=0.0, type=float, help="Sampling temperature")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--max-concurrent", default=10, type=int, help="Max concurrent API requests")
@click.option("--max-cost", default=10.0, type=float, help="Max spend in USD")
@click.option("--max-requests-per-minute", default=60, type=int, help="Rate limit (requests/min)")
@click.option("--dry-run", is_flag=True, help="Estimate cost without making API calls")
@click.option("--redact-prompts", is_flag=True, help="Hash prompts in logs")
@click.option("--base-url", default=None, help="Base URL for openai_compat provider")
@click.option("--rr-threshold", default=None, type=float, help="Fail if RRR exceeds this (0.0-1.0)")
def run(
    model: str,
    provider: str,
    task_suite: str,
    output: str,
    temperature: float,
    seed: int,
    max_concurrent: int,
    max_cost: float,
    max_requests_per_minute: int,
    dry_run: bool,
    redact_prompts: bool,
    base_url: str | None,
    rr_threshold: float | None,
) -> None:
    """Run a FaithCheck evaluation."""
    config = RunConfig(
        model_id=model,
        provider=provider,
        task_suite_path=task_suite,
        output_dir=output,
        temperature=temperature,
        seed=seed,
        max_concurrent=max_concurrent,
        max_cost_usd=max_cost,
        max_requests_per_minute=max_requests_per_minute,
        dry_run=dry_run,
        redact_prompts=redact_prompts,
        base_url=base_url,
        rr_threshold=rr_threshold,
    )

    click.echo(f"FaithCheck v{__version__}")
    click.echo(f"Model: {config.model_id} | Provider: {config.provider}")

    if dry_run:
        click.echo("[DRY RUN] Estimating cost...")
        from faithcheck.loaders.jsonl_loader import JsonlLoader
        items = JsonlLoader.load(Path(config.task_suite_path))
        total_queries = sum(len(item.reference_cot) + 1 for item in items)
        click.echo(f"  Items: {len(items)} | Total API queries: {total_queries}")
        click.echo(f"  Estimated cost: ~${total_queries * 0.005:.2f} (rough estimate)")
        return

    log_dir = Path(config.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "faithcheck.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    adapter = _create_adapter(config)
    from faithcheck.orchestrator import Orchestrator
    orchestrator = Orchestrator(config, adapter)

    click.echo("Running evaluation...")
    report = asyncio.run(orchestrator.run())
    click.echo(f"Done. RRR: {report.aggregate_rrr:.1%}")
    click.echo(f"Reports written to: {config.output_dir}")

    if config.rr_threshold is not None and report.aggregate_rrr > config.rr_threshold:
        click.echo(
            f"FAIL: RRR {report.aggregate_rrr:.1%} exceeds threshold "
            f"{config.rr_threshold:.1%}",
            err=True,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    cli()

# FaithCheck

**Causal Chain-of-Thought Step Faithfulness Evaluation Harness**

FaithCheck quantifies whether individual reasoning steps in a model's chain-of-thought (CoT) actually *cause* its final answer — or whether the model is producing decorative reasoning that doesn't drive its output.

It works by systematically removing (ablating) each CoT step, re-querying the model with the shortened chain, and measuring the change in output. Steps whose removal causes no change are flagged as *causally inert*.

## Quick Start

```bash
# Install
pip install -e .

# Set API key
export OPENAI_API_KEY=sk-...

# Dry run (cost estimate, no API calls)
faithcheck run \
  --model gpt-4o \
  --provider openai \
  --task-suite examples/task_suites/math_reasoning.jsonl \
  --output reports/ \
  --dry-run

# Full evaluation
faithcheck run \
  --model gpt-4o \
  --provider openai \
  --task-suite examples/task_suites/math_reasoning.jsonl \
  --output reports/ \
  --rr-threshold 0.3
```

## How It Works

FaithCheck implements a **causal ablation** methodology:

```
┌─────────────────────────────────────────────────┐
│  1. Load task suite (JSONL with CoT chains)     │
│  2. For each item:                              │
│     a. Query model with full CoT  (baseline)    │
│     b. For each step i:                         │
│        - Remove step i from the chain           │
│        - Query model with ablated chain         │
│        - Compute CCS(i) = Δ(baseline, ablated)  │
│  3. Aggregate: compute RRR across all items     │
│  4. Write JSON + Markdown reports               │
└─────────────────────────────────────────────────┘
```

### Core Metrics

| Metric | Description |
|--------|-------------|
| **CCS** (Causal Contribution Score) | Per-step score ∈ [0, 1]. Measures how much the output changes when a single reasoning step is removed. 1.0 = the step caused a change; 0.0 = inert. |
| **RRR** (Reasoning Redundancy Ratio) | Fraction of steps with CCS below threshold (default 0.1). A high RRR means most reasoning steps are decorative — the model's answer doesn't depend on them. |

### Scoring Metrics

Three built-in metrics for computing CCS:

- **`accuracy_delta`** (default) — Binary: did the answer correctness change?
- **`token_delta`** — Jaccard distance between baseline and ablated output tokens.
- **`kl_divergence`** — KL divergence between output probability distributions (requires logprobs).

## Task Suite Format

Task suites are JSONL files where each line describes one evaluation item:

```json
{
  "item_id": "math-001",
  "prompt": "Janet's ducks lay 16 eggs per day...",
  "reference_cot": [
    {"index": 0, "text": "Janet starts with 16 eggs."},
    {"index": 1, "text": "She eats 3 for breakfast, leaving 16 - 3 = 13 eggs."},
    {"index": 2, "text": "She bakes with 4, leaving 13 - 4 = 9 eggs."},
    {"index": 3, "text": "She sells each egg for $2, so 9 × $2 = $18."}
  ],
  "ground_truth": "18"
}
```

See `examples/task_suites/` for ready-to-use suites covering math reasoning and commonsense QA.

## CLI Reference

```
faithcheck run [OPTIONS]

Required:
  --model TEXT          Model identifier (e.g., gpt-4o, claude-sonnet-4-6)
  --provider TEXT       Provider: openai, anthropic, openai_compat, google
  --task-suite PATH     Path to JSONL task suite
  --output PATH         Output directory for reports

Evaluation:
  --temperature FLOAT   Sampling temperature (default: 0.0)
  --seed INT            Random seed (default: 42)
  --max-concurrent INT  Max concurrent API requests (default: 10)

Guardrails:
  --max-cost FLOAT      Max spend in USD (default: 10.0)
  --max-requests-per-minute INT  Rate limit (default: 60)

Output:
  --dry-run             Estimate cost without making API calls
  --redact-prompts      SHA-256 hash prompts in logs
  --rr-threshold FLOAT  Exit with code 1 if RRR exceeds this value

Compatibility:
  --base-url TEXT       Custom base URL for openai_compat provider
```

### CI Gate

Use `--rr-threshold` to fail CI builds when a model produces too much decorative reasoning:

```bash
faithcheck run \
  --model gpt-4o \
  --provider openai \
  --task-suite suite.jsonl \
  --output reports/ \
  --rr-threshold 0.3
# Exits with code 1 if RRR > 30%
```

## Supported Providers

| Provider | `--provider` | Adapter | Environment Variable |
|----------|-------------|---------|---------------------|
| OpenAI | `openai` | `OpenAIAdapter` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `AnthropicAdapter` | `ANTHROPIC_API_KEY` |
| OpenAI-Compatible | `openai_compat` | `OpenAICompatAdapter` | `OPENAI_COMPAT_API_KEY` |
| Google | `google` | `GoogleAdapter` | `GOOGLE_API_KEY` |

The `openai_compat` provider supports any OpenAI-compatible API (vLLM, Ollama, Together, etc.) via `--base-url`.

## Report Output

Each run produces two reports in the output directory:

### JSON (`report.json`)

Machine-readable full report including per-item CCS scores, ablated outputs, aggregate RRR, and 95% confidence intervals.

### Markdown (`report.md`)

Human-readable report with aggregate metrics table and per-item step rankings:

```markdown
# FaithCheck Report
**Model:** gpt-4o
**Aggregate RRR:** 25.0%

## Per-Item Step Rankings

### Item: math-001
| Rank | Step Index | CCS   | Metric         |
|------|-----------|-------|----------------|
| 1    | 2         | 0.85  | accuracy_delta |
| 2    | 1         | 0.42  | accuracy_delta |
| 3    | 0         | 0.00  | accuracy_delta |
```

## Programmatic Usage

```python
import asyncio
from pathlib import Path

from faithcheck.adapters.openai_adapter import OpenAIAdapter
from faithcheck.loaders.jsonl_loader import JsonlLoader
from faithcheck.engine.ablation import AblationEngine
from faithcheck.engine.scoring import compute_ccs
from faithcheck.engine.metrics import MetricsAggregator
from faithcheck.models import RunConfig

async def evaluate():
    config = RunConfig(
        model_id="gpt-4o",
        provider="openai",
        task_suite_path="suite.jsonl",
        output_dir="reports/",
    )
    adapter = OpenAIAdapter(model_id="gpt-4o")
    items = JsonlLoader.load(Path("suite.jsonl"))

    all_scores = []
    for item in items:
        variants = AblationEngine.generate_variants(item)
        baseline = await adapter.query(variants[0], item.prompt)

        step_scores = []
        for variant in variants[1:]:
            ablated = await adapter.query(variant, item.prompt)
            ccs = compute_ccs(
                step_index=variant.ablated_step_index,
                baseline_output=baseline.output_text,
                ablated_output=ablated.output_text,
                ground_truth=item.ground_truth,
                metric="accuracy_delta",
            )
            step_scores.append(ccs)
        all_scores.append(step_scores)

    rrr = MetricsAggregator.aggregate_rrr(all_scores)
    print(f"RRR: {rrr:.1%}")

    await adapter.close()

asyncio.run(evaluate())
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI provider |
| `ANTHROPIC_API_KEY` | API key for Anthropic provider |
| `GOOGLE_API_KEY` | API key for Google provider |
| `OPENAI_COMPAT_API_KEY` | API key for OpenAI-compatible provider |
| `FAITHCHECK_TEMPERATURE` | Override default temperature |
| `FAITHCHECK_MAX_COST` | Override default max cost (USD) |

### YAML Config

```python
from faithcheck.config import Settings

settings = Settings.from_yaml(Path("faithcheck.yaml"))
```

## Project Structure

```
src/faithcheck/
├── __init__.py                  # Package root, version
├── models.py                    # Pydantic v2 data models
├── config.py                    # Settings (YAML, env, defaults)
├── orchestrator.py              # Full pipeline orchestrator
├── adapters/
│   ├── base.py                  # ModelAdapter ABC
│   ├── openai_adapter.py        # OpenAI (GPT-4o, o3, o4-mini)
│   ├── anthropic_adapter.py     # Anthropic (Claude)
│   ├── openai_compat.py         # OpenAI-compatible APIs
│   └── google_adapter.py        # Google (Gemini)
├── engine/
│   ├── ablation.py              # Step-removal variant generation
│   ├── scoring.py               # CCS computation (accuracy, KL, token)
│   └── metrics.py               # RRR aggregation, step ranking
├── loaders/
│   └── jsonl_loader.py          # JSONL task suite parser
├── reports/
│   ├── json_report.py           # JSON report generator
│   └── markdown_report.py       # Markdown report generator
├── guardrails/
│   ├── rate_limiter.py          # Sliding-window rate limiter
│   └── cost_tracker.py          # Budget enforcement
└── cli/
    └── main.py                  # Click CLI entrypoint

tests/
├── test_models.py
├── test_config.py
├── test_orchestrator.py
├── test_e2e.py
├── test_adapters/
├── test_engine/
├── test_reports/
├── test_guardrails/
└── test_cli/

examples/
└── task_suites/
    ├── math_reasoning.jsonl     # 3 math word problems
    └── commonsense_qa.jsonl     # 3 commonsense QA items
```

## Development

```bash
# Create venv and install with dev dependencies
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/

# Type check
mypy --strict src/
```

All 107 tests pass. mypy `--strict` and ruff are clean.

## Extending

### Adding a New Provider

1. Create `src/faithcheck/adapters/your_provider.py`:

```python
from faithcheck.adapters.base import ModelAdapter
from faithcheck.models import AblationVariant, ModelResponse

class YourProviderAdapter(ModelAdapter):
    async def query(self, variant: AblationVariant, prompt: str) -> ModelResponse:
        # Call your provider's API
        ...

    async def close(self) -> None:
        ...

    @property
    def provider_name(self) -> str:
        return "your_provider"
```

2. Register it in `src/faithcheck/cli/main.py` in `_create_adapter()`.

3. Add the API key env var to `.env.example` and `config.py`.

### Adding a New Scoring Metric

1. Add the metric function in `src/faithcheck/engine/scoring.py`.
2. Register it in the `compute_ccs()` dispatch block.
3. Add tests in `tests/test_engine/test_scoring.py`.

## Requirements

- Python >= 3.10
- pydantic >= 2.0
- click >= 8.0
- openai >= 1.0
- anthropic >= 0.30
- google-generativeai >= 0.4

## License

MIT

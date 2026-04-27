"""Anthropic API adapter."""
from __future__ import annotations

import os

from anthropic import AsyncAnthropic

from faithcheck.adapters.base import ModelAdapter
from faithcheck.engine.ablation import AblationEngine
from faithcheck.models import AblationVariant, ModelResponse


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic models (Claude 3.5/3.7 Sonnet, Claude 3 Opus)."""

    def __init__(self, model_id: str, api_key: str | None = None, temperature: float = 0.0) -> None:
        self._model_id = model_id
        self._temperature = temperature
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Anthropic API key required: pass api_key or set ANTHROPIC_API_KEY")
        self._client = AsyncAnthropic(api_key=key)

    async def query(self, variant: AblationVariant, prompt: str) -> ModelResponse:
        """Query Anthropic with the given ablation variant."""
        cot_text = AblationEngine.reconstruct_prompt(variant)
        response = await self._client.messages.create(
            model=self._model_id,
            max_tokens=4096,
            temperature=self._temperature,
            messages=[
                {"role": "user", "content": f"{prompt}\n\nReasoning:\n{cot_text}"},
            ],
        )
        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        return ModelResponse(
            model_id=self._model_id,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            output_text=text,
        )

    async def close(self) -> None:
        await self._client.close()

    @property
    def provider_name(self) -> str:
        return "anthropic"

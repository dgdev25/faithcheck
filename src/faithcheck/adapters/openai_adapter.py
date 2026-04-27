"""OpenAI API adapter."""
from __future__ import annotations

import os

from openai import AsyncOpenAI

from faithcheck.adapters.base import ModelAdapter
from faithcheck.engine.ablation import AblationEngine
from faithcheck.models import AblationVariant, ModelResponse


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI models (GPT-4o, o3, o4-mini)."""

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        seed: int = 42,
        temperature: float = 0.0,
    ) -> None:
        self._model_id = model_id
        self._temperature = temperature
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key required: pass api_key or set OPENAI_API_KEY")
        self._client = AsyncOpenAI(api_key=key)
        self._seed = seed

    async def query(self, variant: AblationVariant, prompt: str) -> ModelResponse:
        """Query OpenAI with the given ablation variant."""
        cot_text = AblationEngine.reconstruct_prompt(variant)
        messages = [
            {"role": "system", "content": (
                "You are a reasoning assistant. "
                "Follow the provided reasoning steps."
            )},
            {"role": "user", "content": f"{prompt}\n\nReasoning:\n{cot_text}"},
        ]
        response = await self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,  # type: ignore[arg-type]
            seed=self._seed,
            temperature=self._temperature,
        )
        choice = response.choices[0]
        usage = response.usage
        return ModelResponse(
            model_id=self._model_id,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            output_text=choice.message.content or "",
        )

    async def close(self) -> None:
        await self._client.close()

    @property
    def provider_name(self) -> str:
        return "openai"

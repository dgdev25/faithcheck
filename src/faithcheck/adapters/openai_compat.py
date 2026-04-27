"""OpenAI-compatible API adapter (Together AI, Fireworks, Mistral, DeepSeek, Cohere)."""
from __future__ import annotations

import os

from openai import AsyncOpenAI

from faithcheck.adapters.base import ModelAdapter
from faithcheck.engine.ablation import AblationEngine
from faithcheck.models import AblationVariant, ModelResponse


class OpenAICompatAdapter(ModelAdapter):
    """Adapter for any OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_COMPAT_API_KEY",
        temperature: float = 0.0,
    ) -> None:
        self._model_id = model_id
        self._temperature = temperature
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"API key required: pass api_key or set {api_key_env}")
        self._client = AsyncOpenAI(
            api_key=key,
            base_url=base_url,
        )

    async def query(self, variant: AblationVariant, prompt: str) -> ModelResponse:
        """Query the compatible endpoint."""
        cot_text = AblationEngine.reconstruct_prompt(variant)
        messages = [
            {"role": "user", "content": f"{prompt}\n\nReasoning:\n{cot_text}"},
        ]
        response = await self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,  # type: ignore[arg-type]
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
        return "openai_compat"

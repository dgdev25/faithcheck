"""Google Generative AI adapter (Gemini)."""
from __future__ import annotations

import os

import google.generativeai as genai

from faithcheck.adapters.base import ModelAdapter
from faithcheck.engine.ablation import AblationEngine
from faithcheck.models import AblationVariant, ModelResponse


class GoogleAdapter(ModelAdapter):
    """Adapter for Google Gemini models."""

    def __init__(self, model_id: str, api_key: str | None = None) -> None:
        self._model_id = model_id
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Google API key required: pass api_key or set GOOGLE_API_KEY")
        genai.configure(api_key=key)  # type: ignore[attr-defined]
        self._model = genai.GenerativeModel(model_id)  # type: ignore[attr-defined]

    async def query(self, variant: AblationVariant, prompt: str) -> ModelResponse:
        """Query Gemini with the given ablation variant."""
        cot_text = AblationEngine.reconstruct_prompt(variant)
        response = await self._model.generate_content_async(
            f"{prompt}\n\nReasoning:\n{cot_text}"
        )
        try:
            output_text = response.text
        except ValueError:
            output_text = "[BLOCKED_BY_SAFETY_FILTER]"
        return ModelResponse(
            model_id=self._model_id,
            prompt_tokens=response.usage_metadata.prompt_token_count,
            completion_tokens=response.usage_metadata.candidates_token_count,
            output_text=output_text,
        )

    async def close(self) -> None:
        pass

    @property
    def provider_name(self) -> str:
        return "google"

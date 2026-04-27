"""Tests for faithcheck.adapters.openai_adapter using mocked HTTP."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from faithcheck.adapters.openai_adapter import OpenAIAdapter
from faithcheck.models import AblationVariant, ModelResponse, StepBoundary


def _variant(n_steps: int, ablated: int | None = None) -> AblationVariant:
    return AblationVariant(
        task_item_id="test",
        ablated_step_index=ablated,
        chain_steps=[StepBoundary(index=i, text=f"Step {i}") for i in range(n_steps)],
    )


@pytest.mark.asyncio
class TestOpenAIAdapter:
    async def test_query_returns_model_response(self):
        adapter = OpenAIAdapter(model_id="gpt-4o", api_key="test-key")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "42"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 10

        with patch("faithcheck.adapters.openai_adapter.AsyncOpenAI") as mock_client:
            mock_instance = mock_client.return_value
            mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            adapter._client = mock_instance

            result = await adapter.query(_variant(2), "What is 2+2?")
            assert isinstance(result, ModelResponse)
            assert result.output_text == "42"
            assert result.prompt_tokens == 100

    async def test_provider_name(self):
        adapter = OpenAIAdapter(model_id="gpt-4o", api_key="test-key")
        assert adapter.provider_name == "openai"

    async def test_close(self):
        adapter = OpenAIAdapter(model_id="gpt-4o", api_key="test-key")
        await adapter.close()

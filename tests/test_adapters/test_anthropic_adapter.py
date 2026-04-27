"""Tests for faithcheck.adapters.anthropic_adapter."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from faithcheck.adapters.anthropic_adapter import AnthropicAdapter
from faithcheck.models import AblationVariant, ModelResponse, StepBoundary


def _variant() -> AblationVariant:
    return AblationVariant(
        task_item_id="test",
        ablated_step_index=None,
        chain_steps=[StepBoundary(index=0, text="Step 0")],
    )


@pytest.mark.asyncio
class TestAnthropicAdapter:
    async def test_query_returns_model_response(self):
        adapter = AnthropicAdapter(model_id="claude-sonnet-4-6", api_key="test-key")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="The answer is 42")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 20

        with patch("faithcheck.adapters.anthropic_adapter.AsyncAnthropic") as mock_client:
            mock_instance = mock_client.return_value
            mock_instance.messages.create = AsyncMock(return_value=mock_response)
            adapter._client = mock_instance

            result = await adapter.query(_variant(), "What is 2+2?")
            assert isinstance(result, ModelResponse)
            assert result.output_text == "The answer is 42"

    async def test_provider_name(self):
        adapter = AnthropicAdapter(model_id="claude-sonnet-4-6", api_key="test-key")
        assert adapter.provider_name == "anthropic"

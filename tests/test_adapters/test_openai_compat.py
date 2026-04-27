"""Tests for faithcheck.adapters.openai_compat."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from faithcheck.adapters.openai_compat import OpenAICompatAdapter
from faithcheck.models import AblationVariant, StepBoundary


def _variant() -> AblationVariant:
    return AblationVariant(
        task_item_id="test",
        ablated_step_index=None,
        chain_steps=[StepBoundary(index=0, text="Step 0")],
    )


@pytest.mark.asyncio
class TestOpenAICompatAdapter:
    async def test_custom_base_url(self):
        adapter = OpenAICompatAdapter(
            model_id="meta-llama/Llama-3-70b",
            api_key="test-key",
            base_url="https://api.together.xyz/v1",
        )
        assert adapter.provider_name == "openai_compat"

    async def test_query_uses_base_url(self):
        adapter = OpenAICompatAdapter(
            model_id="test-model",
            api_key="test-key",
            base_url="https://api.test.com/v1",
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "answer"
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 5

        with patch("faithcheck.adapters.openai_compat.AsyncOpenAI") as mock_client_cls:
            mock_instance = mock_client_cls.return_value
            mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            adapter._client = mock_instance

            result = await adapter.query(_variant(), "prompt")
            assert result.output_text == "answer"

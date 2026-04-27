"""Tests for faithcheck.adapters.base — adapter protocol."""
import pytest

from faithcheck.adapters.base import ModelAdapter
from faithcheck.models import AblationVariant, ModelResponse, StepBoundary


class StubAdapter(ModelAdapter):
    async def query(self, variant: AblationVariant, prompt: str) -> ModelResponse:
        return ModelResponse(
            model_id="stub",
            prompt_tokens=0,
            completion_tokens=0,
            output_text="stub response",
        )

    async def close(self) -> None:
        pass

    @property
    def provider_name(self) -> str:
        return "stub"


class TestAdapterProtocol:
    def test_stub_adapter_conforms(self):
        adapter = StubAdapter()
        assert adapter.provider_name == "stub"


@pytest.mark.asyncio
class TestStubAdapterQuery:
    async def test_query_returns_model_response(self):
        adapter = StubAdapter()
        variant = AblationVariant(
            task_item_id="t1",
            ablated_step_index=None,
            chain_steps=[StepBoundary(index=0, text="step 0")],
        )
        result = await adapter.query(variant, "What is 2+2?")
        assert isinstance(result, ModelResponse)
        assert result.output_text == "stub response"

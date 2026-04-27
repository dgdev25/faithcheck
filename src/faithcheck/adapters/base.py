"""Abstract base class for model provider adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faithcheck.models import AblationVariant, ModelResponse


class ModelAdapter(ABC):
    """Protocol for model provider adapters."""

    @abstractmethod
    async def query(self, variant: AblationVariant, prompt: str) -> ModelResponse:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

"""Configuration management for FaithCheck."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path


class ProviderConfig(BaseModel):
    """Configuration for a model provider."""

    name: str
    api_key_env: str
    base_url: str | None = None
    max_requests_per_minute: int = 60
    timeout_seconds: int = 120


def _default_providers() -> dict[str, ProviderConfig]:
    return {
        "openai": ProviderConfig(name="openai", api_key_env="OPENAI_API_KEY"),
        "anthropic": ProviderConfig(
            name="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            max_requests_per_minute=50,
        ),
        "openai_compat": ProviderConfig(
            name="openai_compat",
            api_key_env="OPENAI_COMPAT_API_KEY",
        ),
        "google": ProviderConfig(
            name="google",
            api_key_env="GOOGLE_API_KEY",
            max_requests_per_minute=30,
        ),
    }


class Settings(BaseModel):
    """Global FaithCheck settings."""

    default_temperature: float = Field(ge=0.0, le=2.0, default=0.0)
    default_seed: int = 42
    default_max_concurrent: int = Field(ge=1, le=100, default=10)
    default_max_cost_usd: float = Field(gt=0.0, default=10.0)
    default_max_rpm: int = Field(ge=1, default=60)
    default_ccs_threshold: float = Field(ge=0.0, le=1.0, default=0.1)
    providers: dict[str, ProviderConfig] = Field(default_factory=_default_providers)

    @classmethod
    def from_yaml(cls, path: Path) -> Settings:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @classmethod
    def from_env(cls) -> Settings:
        temp_str = os.environ.get("FAITHCHECK_TEMPERATURE")
        cost_str = os.environ.get("FAITHCHECK_MAX_COST")
        kwargs: dict[str, object] = {}
        if temp_str is not None:
            try:
                kwargs["default_temperature"] = float(temp_str)
            except ValueError:
                raise ValueError(
                    f"FAITHCHECK_TEMPERATURE must be a number, got: {temp_str!r}"
                ) from None
        if cost_str is not None:
            try:
                kwargs["default_max_cost_usd"] = float(cost_str)
            except ValueError:
                raise ValueError(
                    f"FAITHCHECK_MAX_COST must be a number, got: {cost_str!r}"
                ) from None
        return cls(**kwargs)  # type: ignore[arg-type]

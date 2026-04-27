"""Tests for faithcheck.config."""
from pathlib import Path

import pytest

from faithcheck.config import ProviderConfig, Settings


class TestProviderConfig:
    def test_openai_config(self):
        cfg = ProviderConfig(name="openai", api_key_env="OPENAI_API_KEY")
        assert cfg.name == "openai"
        assert cfg.base_url is None

    def test_custom_base_url(self):
        cfg = ProviderConfig(
            name="openai_compat",
            api_key_env="TOGETHER_API_KEY",
            base_url="https://api.together.xyz/v1",
        )
        assert cfg.base_url == "https://api.together.xyz/v1"


class TestSettings:
    def test_default_settings(self):
        s = Settings()
        assert s.default_temperature == 0.0
        assert s.default_seed == 42
        assert s.default_max_concurrent == 10
        assert s.default_max_cost_usd == 10.0

    def test_providers_loaded(self):
        s = Settings()
        assert "openai" in s.providers
        assert "anthropic" in s.providers

    def test_from_yaml(self, tmp_path: Path):
        yaml_content = """
default_temperature: 0.0
default_seed: 42
default_max_concurrent: 5
default_max_cost_usd: 25.0
providers:
  custom:
    name: custom
    api_key_env: CUSTOM_KEY
    base_url: "https://api.custom.com/v1"
"""
        config_file = tmp_path / "faithcheck.yaml"
        config_file.write_text(yaml_content)
        s = Settings.from_yaml(config_file)
        assert s.default_max_concurrent == 5
        assert s.default_max_cost_usd == 25.0
        assert "custom" in s.providers

    def test_from_env_override(self, monkeypatch):
        monkeypatch.setenv("FAITHCHECK_TEMPERATURE", "0.5")
        monkeypatch.setenv("FAITHCHECK_MAX_COST", "50.0")
        s = Settings.from_env()
        assert s.default_temperature == 0.5
        assert s.default_max_cost_usd == 50.0

    def test_invalid_temperature_env(self, monkeypatch):
        monkeypatch.setenv("FAITHCHECK_TEMPERATURE", "not_a_number")
        with pytest.raises(ValueError):
            Settings.from_env()

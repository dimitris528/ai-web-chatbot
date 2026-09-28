from __future__ import annotations

import pytest

from chatbot import create_app
from chatbot.config import DEFAULT_SYSTEM_PROMPT, ConfigError, Settings


def test_defaults_from_empty_environment() -> None:
    settings = Settings.from_env({})
    assert settings.groq_model == "openai/gpt-oss-120b"
    assert settings.llm_max_retries == 2
    assert settings.default_system_prompt == DEFAULT_SYSTEM_PROMPT
    assert settings.app_env == "development"


def test_values_are_parsed_to_their_field_types() -> None:
    settings = Settings.from_env(
        {
            "GROQ_API_KEY": "gsk_example",
            "GROQ_MODEL": "llama-3.1-8b-instant",
            "LLM_TEMPERATURE": "0.2",
            "LLM_MAX_TOKENS": "256",
            "LLM_TIMEOUT_SECONDS": "12.5",
            "APP_ENV": "Test",
            "LOG_LEVEL": "debug",
        }
    )
    assert settings.groq_model == "llama-3.1-8b-instant"
    assert settings.llm_temperature == 0.2
    assert settings.llm_max_tokens == 256
    assert settings.llm_timeout_seconds == 12.5
    assert settings.app_env == "test"
    assert settings.log_level == "DEBUG"


def test_blank_values_fall_back_to_defaults() -> None:
    assert Settings.from_env({"LLM_MAX_TOKENS": "  "}).llm_max_tokens == 2048


def test_unparseable_values_are_reported_together() -> None:
    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env({"LLM_MAX_TOKENS": "lots", "LLM_TEMPERATURE": "hot"})
    assert "LLM_MAX_TOKENS" in str(excinfo.value)
    assert "LLM_TEMPERATURE" in str(excinfo.value)


@pytest.mark.parametrize(
    ("env", "fragment"),
    [
        ({"LLM_TEMPERATURE": "3"}, "LLM_TEMPERATURE"),
        ({"LLM_MAX_TOKENS": "0"}, "LLM_MAX_TOKENS"),
        ({"CONTEXT_TOKEN_BUDGET": "10"}, "CONTEXT_TOKEN_BUDGET"),
        ({"APP_ENV": "staging"}, "APP_ENV"),
        ({"LOG_LEVEL": "loud"}, "LOG_LEVEL"),
    ],
)
def test_out_of_range_values_are_rejected(env: dict[str, str], fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        Settings.from_env(env)


def test_production_requires_a_strong_secret_key() -> None:
    with pytest.raises(ConfigError, match="SECRET_KEY"):
        Settings.from_env({"APP_ENV": "production", "SECRET_KEY": "short"})
    assert Settings.from_env({"APP_ENV": "production", "SECRET_KEY": "k" * 32}).is_production


def test_secrets_are_hidden_from_repr() -> None:
    text = repr(Settings(groq_api_key="gsk_super_secret", secret_key="s" * 32))
    assert "gsk_super_secret" not in text
    assert "s" * 32 not in text


def test_app_refuses_to_start_without_api_key() -> None:
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        create_app(Settings(app_env="test"))

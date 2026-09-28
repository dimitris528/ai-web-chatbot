"""Typed, validated application configuration loaded from environment variables."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Literal, TypeVar, cast

Environment = Literal["development", "production", "test"]

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. Answer clearly, use Markdown code "
    "blocks for code, and say so when you are unsure instead of guessing."
)

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when the environment contains missing or invalid configuration."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings. Every field maps to an upper-cased env var."""

    groq_api_key: str = field(default="", repr=False)
    groq_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2

    context_token_budget: int = 6000
    max_history_messages: int = 50
    max_message_chars: int = 4000
    max_system_prompt_chars: int = 2000
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT

    session_ttl_seconds: int = 3600
    max_sessions: int = 1000

    secret_key: str = field(default="", repr=False)
    app_env: Environment = "development"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        errors: list[str] = []

        def check(condition: bool, message: str) -> None:
            if not condition:
                errors.append(message)

        check(
            self.app_env in ("development", "production", "test"),
            "APP_ENV must be one of: development, production, test",
        )
        check(0.0 <= self.llm_temperature <= 2.0, "LLM_TEMPERATURE must be between 0 and 2")
        check(1 <= self.llm_max_tokens <= 32768, "LLM_MAX_TOKENS must be between 1 and 32768")
        check(self.llm_timeout_seconds > 0, "LLM_TIMEOUT_SECONDS must be positive")
        check(0 <= self.llm_max_retries <= 10, "LLM_MAX_RETRIES must be between 0 and 10")
        check(self.context_token_budget >= 256, "CONTEXT_TOKEN_BUDGET must be at least 256")
        check(self.max_history_messages >= 2, "MAX_HISTORY_MESSAGES must be at least 2")
        check(self.max_message_chars >= 1, "MAX_MESSAGE_CHARS must be positive")
        check(self.max_system_prompt_chars >= 1, "MAX_SYSTEM_PROMPT_CHARS must be positive")
        check(bool(self.default_system_prompt.strip()), "DEFAULT_SYSTEM_PROMPT must not be empty")
        check(self.session_ttl_seconds >= 60, "SESSION_TTL_SECONDS must be at least 60")
        check(self.max_sessions >= 1, "MAX_SESSIONS must be positive")
        check(
            self.log_level.upper() in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
            "LOG_LEVEL must be a standard logging level",
        )
        if self.app_env == "production":
            check(
                len(self.secret_key) >= 32,
                "SECRET_KEY must be set (32+ characters) when APP_ENV=production",
            )

        if errors:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``environ`` (defaults to ``os.environ``)."""
        env = os.environ if environ is None else environ
        errors: list[str] = []

        def read(name: str, parse: Callable[[str], T], default: T) -> T:
            raw = env.get(name.upper())
            if raw is None or raw.strip() == "":
                return default
            try:
                return parse(raw.strip())
            except ValueError:
                errors.append(f"{name.upper()}={raw!r} is not a valid {parse.__name__}")
                return default

        # Each field's default value doubles as its type: "30.0" -> float, "2" -> int, etc.
        kwargs: dict[str, object] = {
            f.name: read(f.name, type(f.default), f.default) for f in fields(cls)
        }

        if errors:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
        kwargs["app_env"] = cast(str, kwargs["app_env"]).lower()
        kwargs["log_level"] = cast(str, kwargs["log_level"]).upper()
        return cls(**kwargs)  # type: ignore[arg-type]

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from flask import Flask
from flask.testing import FlaskClient

from chatbot import create_app
from chatbot.config import Settings
from chatbot.llm import ChatMessage


@dataclass
class FakeProvider:
    """Offline stand-in for the LLM: replays scripted tokens, optionally failing."""

    tokens: list[str] = field(default_factory=lambda: ["Hello", ", ", "world!"])
    error: Exception | None = None
    fail_after: int = 0  # number of tokens to emit before raising ``error``
    model: str = "fake-model"
    calls: list[list[ChatMessage]] = field(default_factory=list)

    def stream_chat(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        self.calls.append(list(messages))
        for index, token in enumerate(self.tokens):
            if self.error is not None and index == self.fail_after:
                raise self.error
            yield token
        if self.error is not None and self.fail_after >= len(self.tokens):
            raise self.error


@pytest.fixture
def settings() -> Settings:
    return Settings(
        groq_api_key="test-key",
        app_env="test",
        secret_key="x" * 32,
        default_system_prompt="You are a test assistant.",
        max_message_chars=100,
        max_system_prompt_chars=50,
    )


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def app(settings: Settings, provider: FakeProvider) -> Flask:
    return create_app(settings, provider)


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    return app.test_client()


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Decode an SSE response body into ``(event, data)`` pairs."""
    events = []
    for frame in body.strip().split("\n\n"):
        event, data = "message", ""
        for line in frame.splitlines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data += line.removeprefix("data:").strip()
        events.append((event, json.loads(data) if data else {}))
    return events

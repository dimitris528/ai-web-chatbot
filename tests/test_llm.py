"""GroqProvider tests against a stubbed SDK client (no network access)."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import groq
import httpx
import pytest

from chatbot.config import Settings
from chatbot.llm import (
    GroqProvider,
    LLMAuthError,
    LLMBadRequestError,
    LLMError,
    LLMModelUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
    translate_error,
)

REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def status_error(cls: type[groq.APIStatusError], status: int) -> groq.APIStatusError:
    return cls("boom", response=httpx.Response(status, request=REQUEST), body=None)


def chunk(text: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class StubClient:
    """Mimics ``client.chat.completions.create`` for streaming calls."""

    def __init__(self, chunks: list[Any] | None = None, error: Exception | None = None) -> None:
        self.chunks = chunks or []
        self.error = error
        self.kwargs: dict[str, Any] = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> Iterator[Any]:
        self.kwargs = kwargs
        if self.error is not None and not self.chunks:
            raise self.error
        return self._iterate()

    def _iterate(self) -> Iterator[Any]:
        yield from self.chunks
        if self.error is not None:
            raise self.error


def make_provider(client: StubClient) -> GroqProvider:
    return GroqProvider(
        api_key="unused",
        model="llama-test",
        temperature=0.3,
        max_tokens=128,
        timeout_seconds=5,
        max_retries=1,
        client=client,
    )


def test_streams_text_and_skips_empty_chunks() -> None:
    client = StubClient([chunk("Hel"), chunk(None), SimpleNamespace(choices=[]), chunk("lo")])
    provider = make_provider(client)
    messages = [{"role": "user", "content": "hi"}]

    assert list(provider.stream_chat(messages)) == ["Hel", "lo"]  # type: ignore[arg-type]
    assert client.kwargs == {
        "model": "llama-test",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 128,
        "stream": True,
    }


def test_error_before_streaming_is_translated() -> None:
    provider = make_provider(StubClient(error=status_error(groq.RateLimitError, 429)))
    with pytest.raises(LLMRateLimitError) as excinfo:
        list(provider.stream_chat([]))
    assert isinstance(excinfo.value.__cause__, groq.RateLimitError)


def test_error_mid_stream_is_translated_after_partial_output() -> None:
    provider = make_provider(
        StubClient([chunk("partial")], error=groq.APIConnectionError(request=REQUEST))
    )
    stream = provider.stream_chat([])
    assert next(stream) == "partial"
    with pytest.raises(LLMError):
        next(stream)


@pytest.mark.parametrize(
    ("exc", "expected", "retryable"),
    [
        (status_error(groq.AuthenticationError, 401), LLMAuthError, False),
        (status_error(groq.PermissionDeniedError, 403), LLMAuthError, False),
        (status_error(groq.RateLimitError, 429), LLMRateLimitError, True),
        (status_error(groq.BadRequestError, 400), LLMBadRequestError, False),
        (status_error(groq.NotFoundError, 404), LLMModelUnavailableError, False),
        (status_error(groq.InternalServerError, 500), LLMError, True),
        (groq.APITimeoutError(request=REQUEST), LLMTimeoutError, True),
        (groq.APIConnectionError(request=REQUEST), LLMError, True),
    ],
)
def test_translate_error(exc: Exception, expected: type[LLMError], retryable: bool) -> None:
    translated = translate_error(exc)
    assert type(translated) is expected
    assert translated.retryable is retryable
    assert translated.user_message


def test_from_settings_configures_sdk_timeout_and_retries() -> None:
    provider = GroqProvider.from_settings(
        Settings(groq_api_key="gsk_test", llm_timeout_seconds=7, llm_max_retries=4)
    )
    client = provider._client
    assert isinstance(client, groq.Groq)
    assert client.max_retries == 4
    assert client.timeout == 7
    assert provider.model == "openai/gpt-oss-120b"

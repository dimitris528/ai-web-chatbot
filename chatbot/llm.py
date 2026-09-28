"""LLM provider abstraction with a streaming Groq implementation.

Routes depend only on the :class:`ChatProvider` protocol, so the provider can be
swapped (or faked in tests) without touching HTTP code. Provider SDK exceptions
are translated into a small, stable :class:`LLMError` hierarchy that carries a
user-safe message and a ``retryable`` hint for the UI.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, cast

import groq

if TYPE_CHECKING:
    from .config import Settings

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


class LLMError(Exception):
    """Base error for any failure talking to the language model."""

    user_message = "The AI service is temporarily unavailable. Please try again."
    retryable = True


class LLMAuthError(LLMError):
    user_message = (
        "The AI service rejected the server's credentials. Please contact the site owner."
    )
    retryable = False


class LLMRateLimitError(LLMError):
    user_message = "The AI service is receiving too many requests. Please wait a moment and retry."


class LLMTimeoutError(LLMError):
    user_message = "The AI service took too long to respond. Please try again."


class LLMBadRequestError(LLMError):
    user_message = (
        "The request was rejected by the AI service. Try a shorter message or clear the chat."
    )
    retryable = False


class LLMModelUnavailableError(LLMError):
    user_message = "The configured AI model is unavailable. Please contact the site owner."
    retryable = False


class ChatProvider(Protocol):
    """Anything that can stream a chat completion as text fragments."""

    model: str

    def stream_chat(self, messages: Sequence[ChatMessage]) -> Iterator[str]: ...


def translate_error(exc: Exception) -> LLMError:
    """Map a Groq SDK exception onto the provider-agnostic error hierarchy."""
    # Order matters: APITimeoutError is a subclass of APIConnectionError.
    mapping: list[tuple[type[Exception], type[LLMError]]] = [
        (groq.AuthenticationError, LLMAuthError),
        (groq.PermissionDeniedError, LLMAuthError),
        (groq.NotFoundError, LLMModelUnavailableError),
        (groq.RateLimitError, LLMRateLimitError),
        (groq.APITimeoutError, LLMTimeoutError),
        (groq.BadRequestError, LLMBadRequestError),
        (groq.UnprocessableEntityError, LLMBadRequestError),
    ]
    for source, target in mapping:
        if isinstance(exc, source):
            return target(str(exc))
    return LLMError(str(exc))


class GroqProvider:
    """Streams completions from Groq's OpenAI-compatible chat API.

    Transient failures (connection errors, 408/409/429/5xx) are retried by the
    SDK with exponential backoff *before* the first token arrives; failures after
    streaming has started surface as :class:`LLMError` so the UI can offer a retry.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
        max_retries: int,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = client or groq.Groq(
            api_key=api_key, timeout=timeout_seconds, max_retries=max_retries
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> GroqProvider:
        return cls(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    def stream_chat(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=cast(Any, list(messages)),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                text = chunk.choices[0].delta.content
                if text:
                    yield text
        except groq.GroqError as exc:
            logger.warning("Groq request failed: %s: %s", type(exc).__name__, exc)
            raise translate_error(exc) from exc

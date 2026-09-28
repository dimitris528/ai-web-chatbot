"""Per-session conversation memory and token-budgeted prompt assembly."""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .llm import ChatMessage

# Rough per-message overhead for role/formatting tokens in chat templates.
MESSAGE_OVERHEAD_TOKENS = 4


def estimate_tokens(text: str) -> int:
    """Cheap, tokenizer-free estimate (~4 characters per token for English text).

    Deliberately conservative: it is used to keep prompts under the model's
    context window, where over-estimating is safe and under-estimating is not.
    """
    return math.ceil(len(text) / 4) + MESSAGE_OVERHEAD_TOKENS


@dataclass(frozen=True, slots=True)
class PromptPlan:
    messages: list[ChatMessage]
    estimated_tokens: int
    dropped_messages: int


def build_prompt(
    system_prompt: str,
    history: Sequence[ChatMessage],
    user_message: str,
    token_budget: int,
) -> PromptPlan:
    """Assemble ``system + recent history + new user message`` within ``token_budget``.

    The system prompt and the new user message are always included. History is
    added newest-first until the budget is exhausted, then trimmed so the
    retained window starts on a user turn (never an orphaned assistant reply).
    """
    system: ChatMessage = {"role": "system", "content": system_prompt}
    user: ChatMessage = {"role": "user", "content": user_message}
    used = estimate_tokens(system_prompt) + estimate_tokens(user_message)

    kept: list[ChatMessage] = []
    for message in reversed(history):
        cost = estimate_tokens(message["content"])
        if used + cost > token_budget:
            break
        kept.append(message)
        used += cost
    kept.reverse()

    while kept and kept[0]["role"] != "user":
        used -= estimate_tokens(kept.pop(0)["content"])

    return PromptPlan(
        messages=[system, *kept, user],
        estimated_tokens=used,
        dropped_messages=len(history) - len(kept),
    )


@dataclass(slots=True)
class _Conversation:
    messages: list[ChatMessage] = field(default_factory=list)
    touched_at: float = 0.0


class ConversationStore:
    """Thread-safe in-memory store keyed by session id.

    Bounded in three ways so memory cannot grow without limit: idle sessions
    expire after ``ttl_seconds``, the least-recently-used session is evicted
    beyond ``max_sessions``, and each history keeps its last ``max_messages``.

    For multi-instance deployments swap this for a Redis-backed implementation
    with the same interface.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float,
        max_sessions: int,
        max_messages: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions
        self._max_messages = max_messages
        self._clock = clock
        self._data: OrderedDict[str, _Conversation] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._data)

    def history(self, session_id: str) -> list[ChatMessage]:
        with self._lock:
            self._purge_expired()
            conversation = self._data.get(session_id)
            return list(conversation.messages) if conversation else []

    def append(self, session_id: str, *messages: ChatMessage) -> None:
        with self._lock:
            self._purge_expired()
            conversation = self._data.pop(session_id, None) or _Conversation()
            conversation.messages.extend(messages)
            del conversation.messages[: -self._max_messages]
            conversation.touched_at = self._clock()
            self._data[session_id] = conversation  # (re)insert as most recent
            while len(self._data) > self._max_sessions:
                self._data.popitem(last=False)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def _purge_expired(self) -> None:
        cutoff = self._clock() - self._ttl
        # Entries are ordered by last touch, so stop at the first fresh one.
        while self._data:
            oldest_id, oldest = next(iter(self._data.items()))
            if oldest.touched_at > cutoff:
                break
            del self._data[oldest_id]

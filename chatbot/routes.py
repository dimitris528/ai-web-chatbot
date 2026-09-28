"""HTTP routes: the chat UI, a streaming chat API, history management and health."""

from __future__ import annotations

import json
import logging
import secrets
import time
from collections.abc import Iterator
from typing import Any

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
    session,
    stream_with_context,
)

from .config import Settings
from .conversation import ConversationStore, build_prompt
from .llm import ChatMessage, ChatProvider, LLMError

logger = logging.getLogger(__name__)

bp = Blueprint("chat", __name__)


def _settings() -> Settings:
    return current_app.extensions["chatbot.settings"]  # type: ignore[no-any-return]


def _store() -> ConversationStore:
    return current_app.extensions["chatbot.store"]  # type: ignore[no-any-return]


def _provider() -> ChatProvider:
    return current_app.extensions["chatbot.provider"]  # type: ignore[no-any-return]


def _session_id() -> str:
    """Return this browser's opaque session id, issuing one on first use."""
    if "sid" not in session:
        session["sid"] = secrets.token_urlsafe(24)
    return str(session["sid"])


def api_error(status: int, code: str, message: str) -> tuple[Response, int]:
    return jsonify(error={"code": code, "message": message}), status


def sse(event: str, data: dict[str, Any]) -> str:
    """Encode one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@bp.get("/")
def index() -> str:
    _session_id()
    return render_template("index.html")


@bp.get("/healthz")
def healthz() -> Response:
    return jsonify(status="ok")


@bp.get("/api/config")
def client_config() -> Response:
    settings = _settings()
    return jsonify(
        model=_provider().model,
        default_system_prompt=settings.default_system_prompt,
        max_message_chars=settings.max_message_chars,
        max_system_prompt_chars=settings.max_system_prompt_chars,
    )


@bp.get("/api/history")
def get_history() -> Response:
    return jsonify(messages=_store().history(_session_id()))


@bp.delete("/api/history")
def clear_history() -> tuple[str, int]:
    _store().clear(_session_id())
    return "", 204


@bp.post("/api/chat")
def chat() -> Response | tuple[Response, int]:
    settings = _settings()
    # Requiring a JSON body means cross-site HTML forms cannot hit this endpoint
    # (they would trigger a CORS preflight), complementing SameSite cookies.
    if not request.is_json:
        return api_error(415, "unsupported_media_type", "Send the request body as JSON.")
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return api_error(400, "invalid_json", "Request body must be a JSON object.")

    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return api_error(400, "empty_message", "Please type a message first.")
    message = message.strip()
    if len(message) > settings.max_message_chars:
        return api_error(
            413,
            "message_too_long",
            f"Messages are limited to {settings.max_message_chars} characters.",
        )

    system_prompt = payload.get("system_prompt") or settings.default_system_prompt
    if not isinstance(system_prompt, str):
        return api_error(400, "invalid_system_prompt", "system_prompt must be a string.")
    system_prompt = system_prompt.strip() or settings.default_system_prompt
    if len(system_prompt) > settings.max_system_prompt_chars:
        return api_error(
            413,
            "system_prompt_too_long",
            f"System prompts are limited to {settings.max_system_prompt_chars} characters.",
        )

    session_id = _session_id()
    store, provider = _store(), _provider()
    plan = build_prompt(
        system_prompt, store.history(session_id), message, settings.context_token_budget
    )

    def generate() -> Iterator[str]:
        started = time.perf_counter()
        first_token_ms: float | None = None
        parts: list[str] = []
        outcome = "aborted"
        yield sse("meta", {"model": provider.model, "dropped_messages": plan.dropped_messages})
        try:
            for token in provider.stream_chat(plan.messages):
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                parts.append(token)
                yield sse("token", {"text": token})
            outcome = "ok"
        except LLMError as exc:
            outcome = "llm_error"
            yield sse("error", {"message": exc.user_message, "retryable": exc.retryable})
        except Exception:
            outcome = "internal_error"
            logger.exception("Unexpected error while streaming a reply")
            yield sse("error", {"message": "Something went wrong on our side.", "retryable": True})
        finally:
            # Persist the turn on success, or when the client stopped generation
            # mid-stream (GeneratorExit), so the model keeps that partial context.
            # Failed turns are not stored, so retrying does not duplicate them.
            reply = "".join(parts)
            if outcome in ("ok", "aborted") and reply:
                user_turn: ChatMessage = {"role": "user", "content": message}
                bot_turn: ChatMessage = {"role": "assistant", "content": reply}
                store.append(session_id, user_turn, bot_turn)
            total_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "chat outcome=%s model=%s prompt_tokens~%d dropped=%d chars=%d ttft_ms=%s total_ms=%.0f",
                outcome,
                provider.model,
                plan.estimated_tokens,
                plan.dropped_messages,
                len(reply),
                f"{first_token_ms:.0f}" if first_token_ms is not None else "-",
                total_ms,
            )
        if outcome == "ok":
            yield sse(
                "done",
                {
                    "ttft_ms": round(first_token_ms or 0),
                    "total_ms": round(total_ms),
                },
            )

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

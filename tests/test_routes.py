from __future__ import annotations

from typing import Any

import pytest
from flask.testing import FlaskClient

from chatbot.llm import LLMAuthError, LLMRateLimitError

from .conftest import FakeProvider, parse_sse


def chat(client: FlaskClient, message: str, **extra: Any) -> list[tuple[str, dict[str, Any]]]:
    response = client.post("/api/chat", json={"message": message, **extra})
    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    assert response.headers["Cache-Control"] == "no-cache"
    return parse_sse(response.get_data(as_text=True))


# ---------- pages & metadata ----------


def test_index_renders_with_security_headers(client: FlaskClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert b"AI Chat Assistant" in response.data
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    cookie = response.headers["Set-Cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=Lax" in cookie


@pytest.mark.parametrize("path", ["/static/js/chat.js", "/static/css/app.css"])
def test_static_assets_are_served(client: FlaskClient, path: str) -> None:
    with client.get(path) as response:
        assert response.status_code == 200


def test_healthz(client: FlaskClient) -> None:
    assert client.get("/healthz").get_json() == {"status": "ok"}


def test_client_config_exposes_limits_but_no_secrets(client: FlaskClient) -> None:
    data = client.get("/api/config").get_json()
    assert data == {
        "model": "fake-model",
        "default_system_prompt": "You are a test assistant.",
        "max_message_chars": 100,
        "max_system_prompt_chars": 50,
    }


def test_unknown_route_returns_json_error(client: FlaskClient) -> None:
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "not_found"


# ---------- streaming chat ----------


def test_chat_streams_tokens_then_done(client: FlaskClient) -> None:
    events = chat(client, "Hi there")
    names = [name for name, _ in events]
    assert names == ["meta", "token", "token", "token", "done"]
    assert "".join(data["text"] for name, data in events if name == "token") == "Hello, world!"
    assert events[0][1] == {"model": "fake-model", "dropped_messages": 0}
    assert set(events[-1][1]) == {"ttft_ms", "total_ms"}


def test_chat_sends_system_prompt_and_prior_turns(
    client: FlaskClient, provider: FakeProvider
) -> None:
    chat(client, "first")
    chat(client, "second")
    prompt = provider.calls[-1]
    assert prompt == [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "Hello, world!"},
        {"role": "user", "content": "second"},
    ]


def test_custom_system_prompt_overrides_default(
    client: FlaskClient, provider: FakeProvider
) -> None:
    chat(client, "hi", system_prompt="  Talk like a pirate.  ")
    assert provider.calls[-1][0] == {"role": "system", "content": "Talk like a pirate."}


def test_blank_system_prompt_uses_default(client: FlaskClient, provider: FakeProvider) -> None:
    chat(client, "hi", system_prompt="   ")
    assert provider.calls[-1][0]["content"] == "You are a test assistant."


def test_message_is_trimmed(client: FlaskClient, provider: FakeProvider) -> None:
    chat(client, "   padded   ")
    assert provider.calls[-1][-1]["content"] == "padded"


def test_history_round_trip_and_clear(client: FlaskClient) -> None:
    assert client.get("/api/history").get_json() == {"messages": []}
    chat(client, "remember me")
    messages = client.get("/api/history").get_json()["messages"]
    assert messages == [
        {"role": "user", "content": "remember me"},
        {"role": "assistant", "content": "Hello, world!"},
    ]
    assert client.delete("/api/history").status_code == 204
    assert client.get("/api/history").get_json() == {"messages": []}


def test_sessions_are_isolated(app: Any) -> None:
    alice, bob = app.test_client(), app.test_client()
    chat(alice, "alice secret")
    assert bob.get("/api/history").get_json() == {"messages": []}
    assert len(alice.get("/api/history").get_json()["messages"]) == 2


# ---------- validation ----------


@pytest.mark.parametrize(
    ("payload", "status", "code"),
    [
        ({"message": ""}, 400, "empty_message"),
        ({"message": "   "}, 400, "empty_message"),
        ({"message": 42}, 400, "empty_message"),
        ({}, 400, "empty_message"),
        (["not", "an", "object"], 400, "invalid_json"),
        ({"message": "x" * 101}, 413, "message_too_long"),
        ({"message": "hi", "system_prompt": 5}, 400, "invalid_system_prompt"),
        ({"message": "hi", "system_prompt": "p" * 51}, 413, "system_prompt_too_long"),
    ],
)
def test_chat_validation(
    client: FlaskClient, provider: FakeProvider, payload: Any, status: int, code: str
) -> None:
    response = client.post("/api/chat", json=payload)
    assert response.status_code == status
    assert response.get_json()["error"]["code"] == code
    assert provider.calls == []


def test_chat_requires_json_content_type(client: FlaskClient) -> None:
    response = client.post("/api/chat", data="message=hi")
    assert response.status_code == 415
    assert response.get_json()["error"]["code"] == "unsupported_media_type"


def test_chat_rejects_malformed_json(client: FlaskClient) -> None:
    response = client.post("/api/chat", data="{oops", content_type="application/json")
    assert response.status_code == 400


def test_oversized_body_is_rejected(client: FlaskClient) -> None:
    response = client.post("/api/chat", json={"message": "x" * 70_000})
    assert response.status_code == 413


# ---------- resilience ----------


def test_provider_failure_emits_error_event_and_keeps_history_clean(
    client: FlaskClient, provider: FakeProvider
) -> None:
    provider.error = LLMRateLimitError("429")
    events = chat(client, "hello")
    assert [name for name, _ in events] == ["meta", "error"]
    assert events[-1][1] == {"message": LLMRateLimitError.user_message, "retryable": True}
    assert client.get("/api/history").get_json() == {"messages": []}


def test_non_retryable_error_is_flagged(client: FlaskClient, provider: FakeProvider) -> None:
    provider.error = LLMAuthError("401")
    events = chat(client, "hello")
    assert events[-1] == ("error", {"message": LLMAuthError.user_message, "retryable": False})


def test_mid_stream_failure_after_partial_output(
    client: FlaskClient, provider: FakeProvider
) -> None:
    provider.error, provider.fail_after = LLMRateLimitError("429"), 2
    events = chat(client, "hello")
    assert [name for name, _ in events] == ["meta", "token", "token", "error"]
    assert "done" not in dict(events)
    # A failed turn is not stored, so a retry does not duplicate it.
    assert client.get("/api/history").get_json() == {"messages": []}


def test_unexpected_exception_is_contained(client: FlaskClient, provider: FakeProvider) -> None:
    provider.error = RuntimeError("kaboom")
    events = chat(client, "hello")
    assert events[-1][0] == "error"
    assert "kaboom" not in events[-1][1]["message"]  # internals never leak to users


def test_client_abort_persists_partial_reply(client: FlaskClient) -> None:
    response = client.post("/api/chat", json={"message": "long answer please"})
    stream = iter(response.response)
    next(stream)  # meta
    next(stream)  # first token
    response.close()  # client disconnects / presses "stop"
    messages = client.get("/api/history").get_json()["messages"]
    assert messages == [
        {"role": "user", "content": "long answer please"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_context_trimming_is_reported(settings: Any, provider: FakeProvider) -> None:
    from dataclasses import replace

    from chatbot import create_app

    app = create_app(replace(settings, context_token_budget=256, max_message_chars=900), provider)
    client = app.test_client()
    for _ in range(3):
        chat(client, "q" * 300)
    meta = chat(client, "latest")[0]
    assert meta[0] == "meta"
    assert meta[1]["dropped_messages"] > 0

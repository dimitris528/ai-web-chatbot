# AI Web Chatbot

**A streaming LLM chat application built to production standards: token-by-token responses, bounded conversation memory, graceful handling of API failures, and a fully offline test suite.**

[![CI](https://github.com/dimitris528/ai-web-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/dimitris528/ai-web-chatbot/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
![Groq](https://img.shields.io/badge/LLM-Groq%20API-F55036)
![Streaming](https://img.shields.io/badge/Streaming-Server--Sent%20Events-4f46e5)
![CSS](https://img.shields.io/badge/UI-Modern%20CSS%20%2B%20Vanilla%20JS-1572B6?logo=css3&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-pytest%20%C2%B7%2098%25%20coverage-0A9EDC?logo=pytest&logoColor=white)
![Lint](https://img.shields.io/badge/Lint-ruff%20%2B%20mypy%20strict-D7FF64?logo=ruff&logoColor=black)
![Docker](https://img.shields.io/badge/Deploy-Docker%20%7C%20Render-2496ED?logo=docker&logoColor=white)

---

## The problem it solves

Companies want a branded AI assistant on their website or intranet, but a naive LLM integration breaks down in production:

| Naive integration | This project |
|---|---|
| The user waits 5–15 s for a full response | Tokens **stream** over Server-Sent Events. The first words usually show up in well under a second. |
| One API hiccup shows a stack trace or a blank screen | Errors are **classified** (rate limit, timeout, auth, bad request) and shown as friendly messages with a **Retry** button |
| Chat history grows until the model's context window overflows | The prompt is assembled under a **token budget** by trimming the oldest turns |
| Memory grows with every visitor | Sessions are **bounded**: idle TTL, LRU eviction and a per-session message cap |
| The API key is hardcoded or leaked into the frontend | **Typed config** loaded from env vars and validated at startup. The key never reaches the browser. |
| "Works on my machine" | **Docker** image, **CI** pipeline and a **65-test** suite with the LLM mocked |

The same design works for customer-support bots, internal knowledge assistants, or any workflow that needs a conversational front end on top of an LLM API.

## Features

- **Real-time streaming** of responses via SSE, with a typing indicator and a live cursor
- **Stop generation** mid-stream. The partial answer is kept in the conversation context.
- **Configurable system prompt** (persona and rules) per browser, with a one-click reset to default
- **Copy message**, **clear chat**, **retry on failure**, suggested starter prompts
- Safe **Markdown rendering** (code blocks, inline code, bold) built with DOM nodes only, so there is no `innerHTML` XSS surface
- Loading **skeletons**, responsive mobile layout, automatic **dark mode**, `prefers-reduced-motion` support
- Response timing (time to first token and total) shown under each answer
- Hardened HTTP: strict **Content-Security-Policy**, `HttpOnly` + `SameSite` cookies, request size limits, JSON-only API

## Architecture

```
 Browser (static/js/chat.js)                         Flask app (chatbot/)
 ┌──────────────────────────┐                ┌──────────────────────────────────────────┐
 │ Chat UI                  │  POST /api/chat │ routes.py                                │
 │  • composer / controls   │ ──────────────▶ │  1. validate input (size, type, JSON)    │
 │  • system prompt (local) │   JSON body     │  2. resolve session id (signed cookie)   │
 │  • SSE stream reader     │                 │  3. build_prompt() ◀── ConversationStore │
 │  • retry / stop / copy   │ ◀────────────── │  4. stream tokens from ChatProvider      │
 └──────────────────────────┘  text/event-    │  5. persist turn, log latency metrics    │
                                stream         └───────────────┬──────────────────────────┘
      events: meta → token* → done | error                     │ ChatProvider protocol
                                                               ▼
 ┌─────────────────────────────┐        ┌──────────────────────────────────────────────┐
 │ conversation.py             │        │ llm.py — GroqProvider                        │
 │  • per-session history      │        │  • streaming chat.completions (stream=True)  │
 │  • TTL + LRU + message cap  │        │  • SDK timeout + exponential-backoff retries │
 │  • token-budget trimming    │        │  • SDK errors → LLMError hierarchy           │
 │  • thread-safe (Lock)       │        │    (user_message, retryable)                 │
 └─────────────────────────────┘        └───────────────────────┬──────────────────────┘
                                                                ▼
 config.py — frozen, typed Settings                     Groq API (OpenAI-compatible)
 validated from env at startup (fail fast)
```

### Request lifecycle

1. The browser sends `{"message", "system_prompt"}` to `POST /api/chat`.
2. The server validates the payload and loads this session's history.
3. `build_prompt()` places the system prompt and the new message first, then adds history **newest-first** until the token budget is reached. The retained window always starts on a user turn.
4. `GroqProvider.stream_chat()` yields text fragments. Each one is sent as an `event: token` SSE frame.
5. On success, the user and assistant turns are stored and a `done` event reports latency. On failure, an `error` event carries a safe message plus a `retryable` flag. Failed turns are **not** stored, so a retry never duplicates history.

## Engineering highlights

**Streaming latency.** Responses are relayed token by token with `stream_with_context`. `X-Accel-Buffering: no` and `Cache-Control: no-cache` stop reverse proxies from buffering the stream. Gunicorn runs threaded workers (`gthread`) so long-lived streams don't block other users. Each request logs time to first token (TTFT) and total time. In a local test against Groq, TTFT was about 250–500 ms.

**Token and context optimization.** A conservative, tokenizer-free estimate (about 4 characters per token plus per-message overhead) keeps prompts within `CONTEXT_TOKEN_BUDGET` without adding dependencies. The oldest turns are dropped first, and the UI shows a notice when that happens. Stored history is capped separately (`MAX_HISTORY_MESSAGES`), so memory stays bounded no matter how long a session runs.

**Error resilience.**
- The Groq SDK retries connection errors, 408, 409, 429 and 5xx responses with exponential backoff, using a per-request timeout.
- Provider exceptions map to a small `LLMError` hierarchy, so routes never depend on SDK types.
- A mid-stream failure keeps the partial text and offers **Retry**. Unexpected exceptions are logged server-side and never leak internals to users.
- Clean, non-retryable failures for bad credentials (401/403) and a missing model (404).

**Session management.** Each browser gets an opaque random id (`secrets.token_urlsafe`) in a signed, `HttpOnly`, `SameSite=Lax` cookie (`Secure` in production). `ConversationStore` is thread-safe and bounded by idle TTL, LRU size and message count. It sits behind a small interface, so a Redis-backed version can replace it for horizontal scaling.

**Testability.** `create_app(settings, provider)` is an application factory with dependency injection. Tests pass in a `FakeProvider` and a stubbed Groq client, so the whole suite runs offline, deterministically, in under a second.

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python 3.11+, Flask 3 (app factory + blueprint) |
| LLM | Groq API via the official `groq` SDK. Default model: `openai/gpt-oss-120b` |
| Streaming | Server-Sent Events over `fetch` + `ReadableStream` |
| Frontend | Semantic HTML, modern CSS (custom properties, `color-mix`, `dvh`, `<dialog>`), vanilla JS, no build step |
| Quality | pytest + pytest-cov (≥90% enforced), ruff (lint + format), mypy `--strict` |
| Delivery | GitHub Actions, multi-stage Docker image (non-root), docker-compose, Render blueprint |
| Serving | Gunicorn `gthread` workers |

## Getting started

### Prerequisites

- Python 3.11+
- A free Groq API key from [console.groq.com/keys](https://console.groq.com/keys)

### Run locally

```bash
git clone https://github.com/dimitris528/ai-web-chatbot.git
cd ai-web-chatbot

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env               # then set GROQ_API_KEY in .env
python wsgi.py                     # http://127.0.0.1:5000
```

### Run with Docker

```bash
cp .env.example .env               # set GROQ_API_KEY and SECRET_KEY
docker compose up --build          # http://localhost:8000
```

### Deploy to Render

Push the repo, then choose **New → Blueprint** in Render. `render.yaml` provisions the service, generates a `SECRET_KEY` and prompts for `GROQ_API_KEY`.

## Configuration

Settings are read from environment variables (or `.env`) into a frozen, typed `Settings` object. **Invalid values stop startup** with a message listing every problem. See [`.env.example`](.env.example) for the documented template.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — (**required**) | Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Any chat model your key can access |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature, 0–2 |
| `LLM_MAX_TOKENS` | `2048` | Maximum tokens per reply (includes reasoning tokens on reasoning models) |
| `LLM_TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `LLM_MAX_RETRIES` | `2` | SDK retries with exponential backoff |
| `CONTEXT_TOKEN_BUDGET` | `6000` | Approximate prompt budget (system + history + message) |
| `MAX_HISTORY_MESSAGES` | `50` | Stored messages per session |
| `MAX_MESSAGE_CHARS` | `4000` | Maximum user message length |
| `MAX_SYSTEM_PROMPT_CHARS` | `2000` | Maximum custom system prompt length |
| `DEFAULT_SYSTEM_PROMPT` | helpful-assistant persona | Used when the user hasn't set one |
| `SESSION_TTL_SECONDS` | `3600` | Idle conversation expiry |
| `MAX_SESSIONS` | `1000` | Conversations kept in memory (LRU eviction) |
| `SECRET_KEY` | random per process | Cookie signing key. **Required (32+ chars) when `APP_ENV=production`** |
| `APP_ENV` | `development` | `development` / `production` / `test` |
| `LOG_LEVEL` | `INFO` | Standard logging level |

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `POST` | `/api/chat` | Body `{"message": str, "system_prompt"?: str}`. Returns an SSE stream of `meta`, then `token` events, then `done` or `error` |
| `GET` | `/api/history` | Messages in the current session |
| `DELETE` | `/api/history` | Clear the current session (`204`) |
| `GET` | `/api/config` | Public UI settings: model, limits, default prompt |
| `GET` | `/healthz` | Liveness probe |

Validation errors return JSON: `{"error": {"code": "message_too_long", "message": "..."}}` with status `400`, `413` or `415`.

## Quality checks

```bash
pytest --cov          # 65 tests, fully offline (LLM mocked), coverage gate 90%
ruff check .          # lint
ruff format --check . # formatting
mypy                  # strict type checking
```

CI runs all of these on every push and pull request, testing Python 3.11, 3.12 and 3.13. It then builds the Docker image and smoke-tests `/healthz` in a running container.

## Project structure

```
ai-web-chatbot/
├── chatbot/
│   ├── __init__.py        # app factory, security headers, JSON error handling
│   ├── config.py          # typed, validated Settings
│   ├── conversation.py    # token budgeting + bounded session store
│   ├── llm.py             # ChatProvider protocol, GroqProvider, error mapping
│   ├── routes.py          # UI, SSE chat API, history, health
│   ├── templates/index.html
│   └── static/{css/app.css, js/chat.js}
├── tests/                 # pytest suite (config, conversation, llm, routes)
├── .github/workflows/ci.yml
├── Dockerfile · docker-compose.yml · render.yaml · gunicorn.conf.py
├── pyproject.toml         # pytest / coverage / ruff / mypy config
├── requirements.txt · requirements-dev.txt
└── wsgi.py                # entry point
```

## Scaling notes

Conversation memory is in-process by design, which keeps the project simple and dependency-free. For that reason Gunicorn runs **one worker with many threads**. To scale horizontally, implement `ConversationStore`'s four methods (`history`, `append`, `clear`, `__len__`) on Redis and raise the worker and replica count. Nothing else changes.

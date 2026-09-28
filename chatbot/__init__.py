"""AI web chatbot: a Flask application factory with a pluggable LLM provider."""

from __future__ import annotations

import logging
import secrets

from dotenv import load_dotenv
from flask import Flask, Response, jsonify
from werkzeug.exceptions import HTTPException

from .config import ConfigError, Settings
from .conversation import ConversationStore
from .llm import ChatProvider, GroqProvider

__all__ = ["ConfigError", "Settings", "create_app"]

logger = logging.getLogger(__name__)

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
)


def create_app(
    settings: Settings | None = None,
    provider: ChatProvider | None = None,
) -> Flask:
    """Build a configured Flask app.

    ``settings`` defaults to values from the environment (and a local ``.env``);
    ``provider`` defaults to Groq. Tests inject both to run fully offline.
    """
    if settings is None:
        load_dotenv()
        settings = Settings.from_env()
    if provider is None:
        if not settings.groq_api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
                "(get one at https://console.groq.com/keys)."
            )
        provider = GroqProvider.from_settings(settings)

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    app = Flask(__name__)
    secret_key = settings.secret_key
    if not secret_key:
        logger.warning("SECRET_KEY not set; using an ephemeral key (sessions reset on restart).")
        secret_key = secrets.token_hex(32)
    app.config.update(
        SECRET_KEY=secret_key,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=settings.is_production,
        MAX_CONTENT_LENGTH=64 * 1024,
        TESTING=settings.app_env == "test",
    )

    app.extensions["chatbot.settings"] = settings
    app.extensions["chatbot.provider"] = provider
    app.extensions["chatbot.store"] = ConversationStore(
        ttl_seconds=settings.session_ttl_seconds,
        max_sessions=settings.max_sessions,
        max_messages=settings.max_history_messages,
    )

    from .routes import bp

    app.register_blueprint(bp)

    @app.after_request
    def security_headers(response: Response) -> Response:
        response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException) -> tuple[Response, int]:
        status = exc.code or 500
        name = (exc.name or "error").lower().replace(" ", "_")
        return jsonify(error={"code": name, "message": exc.description}), status

    logger.info("Chatbot ready (env=%s, model=%s)", settings.app_env, provider.model)
    return app

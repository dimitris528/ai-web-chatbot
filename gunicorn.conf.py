"""Gunicorn settings tuned for long-lived Server-Sent Event streams."""

import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"

# Conversation memory is in-process, so run ONE worker and scale with threads.
# Each open SSE stream holds a thread; raise GUNICORN_THREADS for more concurrency,
# or move ConversationStore to Redis before adding workers/replicas.
workers = 1
worker_class = "gthread"
threads = int(os.getenv("GUNICORN_THREADS", "16"))

# Must exceed the longest expected stream (LLM timeout x retries + generation time).
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()
forwarded_allow_ips = "*"

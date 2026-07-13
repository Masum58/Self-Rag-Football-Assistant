"""
WHAT: Configures Python's built-in logging so every part of the app
      (routers, graph nodes, memory functions) can log consistently.
WHY: Without this, we only see FastAPI's default access logs (GET/POST
     lines) — no visibility into what happened INSIDE a request (which
     node ran, how long the LLM call took, what error occurred and where).
OUTPUT: `setup_logging()` configures the root logger once at startup.
        `get_logger(name)` returns a logger for any module to use.
CALLED FROM: app/main.py (setup_logging, once at startup)
             any module that wants to log (get_logger(__name__))
WHY CALLED: Centralizes logging format/level so it's consistent everywhere.

সহজ ভাষায় (Bengali):
এতদিন আমরা শুধু `print()` বা uvicorর নিজের access log দেখেছি — কে কবে
কোন endpoint call করলো, এটুকুই। কিন্তু "গ্রাফের কোন node এ কতক্ষণ লাগলো",
"ঠিক কোথায় error হলো" — এসব জানার উপায় ছিল না। এই ফাইলটা প্রতিটা
module কে একটা consistent "logger" ব্যবহার করতে দেয়, যাতে সব log
এক জায়গায়, এক ফরম্যাটে দেখা যায়।
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """
    WHAT: Configures the root logger with a consistent format and level.
    WHY: Must run once, early (at app startup), before any other module
         tries to log — otherwise logs may be missing or badly formatted.
    OUTPUT: None (configures logging globally).
    CALLED FROM: app/main.py, at the very top of the lifespan startup.
    WHY CALLED: Ensures every log line looks the same and goes to the
                same place (console), regardless of which module logs it.

    সহজ ভাষায়: এটা একবার app চালুর শুরুতেই কল হবে, যাতে পুরো অ্যাপ জুড়ে
    log এর ফরম্যাট আর level (INFO/DEBUG/ERROR) একরকম থাকে।

    Example log output:
        2026-07-13 10:15:32 INFO     app.routers.chat: Received question from masum
        2026-07-13 10:15:34 ERROR    app.graph.nodes: Groq call failed: timeout
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,  # আগে থেকে অন্য কেউ logging configure করে থাকলেও override করবে
    )


def get_logger(name: str) -> logging.Logger:
    """
    WHAT: Returns a logger scoped to the given module name.
    WHY: Using __name__ as the logger name means log lines show exactly
         which file they came from (e.g. "app.routers.chat"), making it
         easy to trace where something happened.
    OUTPUT: A configured logging.Logger instance.
    CALLED FROM: Any module — e.g. `logger = get_logger(__name__)` at the
                 top of the file, then `logger.info(...)` / `logger.error(...)`.
    WHY CALLED: Standard way for every module to get its own labeled logger.

    Example:
        # app/routers/chat.py এর উপরে
        from app.core.logging_config import get_logger
        logger = get_logger(__name__)

        logger.info(f"Question from {user_id}: {question}")
        logger.error(f"Graph failed: {error}")
    """
    return logging.getLogger(name)
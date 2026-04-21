"""Central logging configuration."""

from __future__ import annotations

import logging
import sys

from agent.config import get_settings

_configured = False


def configure_logging() -> None:
    """Configure root logging from Settings. Idempotent — safe to call repeatedly."""
    global _configured
    if _configured:
        return
    level_name = get_settings().log_level.upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Third-party libraries log every request/identity at INFO which clobbers
    # the streaming spinner line in interactive mode. Suppress unless the user
    # explicitly drops to DEBUG.
    noisy_level = logging.DEBUG if level <= logging.DEBUG else logging.WARNING
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "edgar", "yfinance"):
        logging.getLogger(noisy).setLevel(noisy_level)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)

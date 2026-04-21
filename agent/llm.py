"""OpenAI-compatible client + context-window probe."""

from __future__ import annotations

from typing import Any

import httpx
from openai import OpenAI

from agent.config import get_settings
from agent.logging_setup import get_logger

log = get_logger(__name__)

# Per-process cache so we probe /v1/models at most once per (local, model_id) pair.
_context_cache: dict[tuple[bool, str], tuple[int, str]] = {}


def create_client(local: bool = True) -> OpenAI:
    """Return an OpenAI client pointed at the local or remote endpoint."""
    s = get_settings()
    if local:
        return OpenAI(
            base_url=s.local_llm_url,
            api_key="not-needed",
            http_client=httpx.Client(verify=False),
        )
    if not (s.llm_api_key and s.llm_base_url):
        raise RuntimeError("Remote LLM requested but LLM_API_KEY / LLM_BASE_URL not set in .env")
    return OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key)


def active_model(local: bool = True) -> str:
    s = get_settings()
    return s.local_model if local else (s.llm_model or "gpt-4o")


def _flag_value(args: list, flag: str) -> str | None:
    """Return the value immediately following `flag` in a llama.cpp CLI arg list."""
    if not isinstance(args, list):
        return None
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return str(args[i + 1])
    return None


def _parse_ctx_from_args(args: list) -> int | None:
    raw = _flag_value(args, "--ctx-size")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def get_context_window(client: OpenAI, model_id: str, *, local: bool) -> tuple[int, str]:
    """Probe /v1/models for the target model's context window.

    Resolution order:
      1. Target model's own `--ctx-size`.
      2. Standard OpenAI-style fields (`context_length`, `max_model_len`, ...) on the target.
      3. Alias match: any other model entry whose `--model <path>` matches the target's.
      4. LOCAL/REMOTE_CONTEXT_WINDOW env var (explicit).
      5. Hardcoded default.

    Returns (window_size, source) where source is one of
    `endpoint`, `endpoint (alias)`, `env`, or `default`. Cached per-process.
    """
    key = (local, model_id)
    if key in _context_cache:
        return _context_cache[key]

    s = get_settings()
    import os

    env_var = "LOCAL_CONTEXT_WINDOW" if local else "REMOTE_CONTEXT_WINDOW"
    env_set_explicitly = os.getenv(env_var) is not None
    fallback = s.local_context_window if local else s.remote_context_window
    fallback_source = "env" if env_set_explicitly else "default"

    try:
        models = client.models.list()
    except Exception as e:  # network/server error
        log.warning("get_context_window: models.list() failed: %s", e)
        result = (fallback, fallback_source)
        _context_cache[key] = result
        return result

    entries: list[dict[str, Any]] = []
    target_path: str | None = None
    target_dict: dict[str, Any] | None = None

    for m in models.data:
        d = m.model_dump() if hasattr(m, "model_dump") else dict(m)
        status = d.get("status") or {}
        args = status.get("args") if isinstance(status, dict) else None
        entry: dict[str, Any] = {
            "id": d.get("id"),
            "path": _flag_value(args or [], "--model"),
            "ctx": _parse_ctx_from_args(args or []),
            "d": d,
        }
        entries.append(entry)
        if entry["id"] == model_id:
            target_path = entry["path"]
            target_dict = d
            if entry["ctx"]:
                result = (int(entry["ctx"]), "endpoint")
                _context_cache[key] = result
                return result

    if target_dict is not None:
        for field in ("context_length", "max_model_len", "n_ctx", "context_window"):
            val = target_dict.get(field)
            if isinstance(val, int) and val > 0:
                result = (val, "endpoint")
                _context_cache[key] = result
                return result

    if target_path:
        sibling_ctx: list[int] = [
            int(e["ctx"]) for e in entries if e["path"] == target_path and e["ctx"]
        ]
        if sibling_ctx:
            result = (max(sibling_ctx), "endpoint (alias)")
            _context_cache[key] = result
            return result

    result = (fallback, fallback_source)
    _context_cache[key] = result
    return result


def reset_context_cache() -> None:
    """Testing hook — drop the probe cache."""
    _context_cache.clear()

"""Pluggable memory store for library-mode embedding.

The default (`FileMemoryStore`) reproduces the original single-user, single-file
behavior — the CLI and one-shot modes keep working unchanged. When embedding
the agent inside a multi-user application (e.g. a FastAPI backend), pass a
custom `MemoryStore` implementation to `run_agent(..., memory_store=...)` and
it will be scoped to that call via a `ContextVar` so concurrent requests stay
isolated.

Only two operations are required of a store: `read()` and `append(fact)`.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryStore(Protocol):
    """Anything with these two methods can back the agent's persistent memory."""

    def read(self) -> str:
        """Return the full current memory contents (markdown-like text)."""
        ...

    def append(self, fact: str) -> None:
        """Persist a new fact. Implementations choose their own storage + formatting."""
        ...


class FileMemoryStore:
    """Default store: one markdown file on disk, appended with ISO-date stamps.

    Matches the pre-existing `memory.md` format so existing files stay readable
    and the REPL / one-shot CLI modes see no behavioral change.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> str:
        if not self.path.exists():
            return ""
        return self.path.read_text(encoding="utf-8")

    def append(self, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            return
        today = datetime.date.today().isoformat()
        entry = f"- [{today}] {fact}\n"
        if not self.path.exists():
            self.path.write_text("# Agent Memory\n\n", encoding="utf-8")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(entry)


# A per-task/per-thread slot for the active store. None → fall back to a
# file-based store using Settings.memory_path.
_current_store: ContextVar[MemoryStore | None] = ContextVar("memory_store", default=None)


def get_active_store() -> MemoryStore:
    """Return the store bound to the current context, or the file-based default."""
    store = _current_store.get()
    if store is not None:
        return store
    from agent.config import get_settings

    return FileMemoryStore(get_settings().memory_path)


@contextmanager
def use_memory_store(store: MemoryStore | None) -> Iterator[None]:
    """Bind `store` as the active memory store for the duration of the block.

    Safe to nest. Passing `None` is a no-op (keeps whatever's currently active).
    ContextVars are per-async-task and per-thread, so concurrent requests in a
    FastAPI/uvicorn server see their own store automatically.
    """
    if store is None:
        yield
        return
    token = _current_store.set(store)
    try:
        yield
    finally:
        _current_store.reset(token)

"""Tests for the MemoryStore protocol, FileMemoryStore default, and contextvar plumbing."""

from __future__ import annotations

from pathlib import Path

from agent.memory import (
    FileMemoryStore,
    MemoryStore,
    get_active_store,
    use_memory_store,
)


class _InMemoryStore:
    """Trivial store used to verify the contextvar is wired correctly."""

    def __init__(self, seed: str = "") -> None:
        self.content = seed
        self.appended: list[str] = []

    def read(self) -> str:
        return self.content

    def append(self, fact: str) -> None:
        self.appended.append(fact)
        self.content += f"- {fact}\n"


class TestFileMemoryStore:
    def test_read_missing_file_returns_empty(self, tmp_path: Path):
        store = FileMemoryStore(tmp_path / "missing.md")
        assert store.read() == ""

    def test_read_existing_content(self, tmp_path: Path):
        path = tmp_path / "m.md"
        path.write_text("# Memory\n\n- [2026-01-01] seed\n", encoding="utf-8")
        store = FileMemoryStore(path)
        assert "seed" in store.read()

    def test_append_creates_file_with_header(self, tmp_path: Path):
        path = tmp_path / "new.md"
        store = FileMemoryStore(path)
        store.append("fresh fact")
        text = path.read_text(encoding="utf-8")
        assert "# Agent Memory" in text
        assert "fresh fact" in text

    def test_append_preserves_existing(self, tmp_path: Path):
        path = tmp_path / "m.md"
        path.write_text("# Agent Memory\n\n- [2026-01-01] old\n", encoding="utf-8")
        store = FileMemoryStore(path)
        store.append("new")
        text = path.read_text(encoding="utf-8")
        assert "old" in text and "new" in text

    def test_append_strips_and_rejects_empty(self, tmp_path: Path):
        path = tmp_path / "m.md"
        store = FileMemoryStore(path)
        store.append("   ")
        # Empty fact should be a no-op — file not created.
        assert not path.exists()


class TestProtocolCompliance:
    def test_file_store_satisfies_protocol(self, tmp_path: Path):
        store = FileMemoryStore(tmp_path / "x.md")
        assert isinstance(store, MemoryStore)

    def test_duck_typed_store_satisfies_protocol(self):
        assert isinstance(_InMemoryStore(), MemoryStore)


class TestContextVarScoping:
    def test_default_fallback_is_file_store(self, tmp_memory: Path):
        """With no explicit binding, get_active_store points at Settings.memory_path."""
        tmp_memory.write_text("# Agent Memory\n\n- [2026-04-21] hello\n", encoding="utf-8")
        store = get_active_store()
        assert isinstance(store, FileMemoryStore)
        assert "hello" in store.read()

    def test_use_memory_store_binds_for_block(self, tmp_memory: Path):
        injected = _InMemoryStore(seed="scoped\n")
        with use_memory_store(injected):
            assert get_active_store() is injected
            assert "scoped" in get_active_store().read()
        # After exiting, we fall back to the file default.
        assert not isinstance(get_active_store(), _InMemoryStore)

    def test_none_binding_is_noop(self, tmp_memory: Path):
        with use_memory_store(None):
            store = get_active_store()
        assert isinstance(store, FileMemoryStore)

    def test_nested_bindings_restore_outer(self, tmp_memory: Path):
        outer = _InMemoryStore(seed="outer\n")
        inner = _InMemoryStore(seed="inner\n")
        with use_memory_store(outer):
            assert get_active_store() is outer
            with use_memory_store(inner):
                assert get_active_store() is inner
            assert get_active_store() is outer

    def test_concurrent_tasks_are_isolated(self, tmp_memory: Path):
        """asyncio tasks each get their own context — so per-request isolation
        in a FastAPI server works without locks."""
        import asyncio

        store_a = _InMemoryStore(seed="A\n")
        store_b = _InMemoryStore(seed="B\n")
        seen: dict[str, MemoryStore] = {}

        async def task(label: str, store: _InMemoryStore):
            with use_memory_store(store):
                await asyncio.sleep(0)  # force a context switch
                seen[label] = get_active_store()

        async def driver():
            await asyncio.gather(task("a", store_a), task("b", store_b))

        asyncio.run(driver())
        assert seen["a"] is store_a
        assert seen["b"] is store_b

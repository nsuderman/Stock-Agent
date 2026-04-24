"""Tests for parallel tool-call dispatch in the ReAct loop."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import pytest
from pydantic import BaseModel

from agent.loop import IterationEvent, _dispatch_tool_calls, run_agent
from agent.tools.base import TOOLS, ToolEntry


def _tc(idx: int, name: str, args_json: str, call_id: str | None = None) -> dict:
    return {
        "id": call_id or f"c{idx}",
        "type": "function",
        "function": {"name": name, "arguments": args_json},
    }


class _NoArgs(BaseModel):
    pass


class _SleepArgs(BaseModel):
    ms: int
    tag: str = ""


def test_parallel_dispatch_actually_runs_concurrently():
    """Three sleepy tools dispatched in one turn should finish in ~one sleep, not three."""
    sleep_ms = 80

    def _sleep_impl(args: _SleepArgs) -> dict:
        time.sleep(args.ms / 1000.0)
        return {"slept": args.ms, "tag": args.tag}

    TOOLS["_sleep"] = ToolEntry(
        name="_sleep",
        description="Sleep.",
        model=_SleepArgs,
        func=lambda **kw: _sleep_impl(_SleepArgs(**kw)),
    )
    try:
        tool_calls = [
            _tc(0, "_sleep", f'{{"ms": {sleep_ms}, "tag": "a"}}'),
            _tc(1, "_sleep", f'{{"ms": {sleep_ms}, "tag": "b"}}'),
            _tc(2, "_sleep", f'{{"ms": {sleep_ms}, "tag": "c"}}'),
        ]
        recent: deque[str] = deque(maxlen=4)

        t0 = time.monotonic()
        records = _dispatch_tool_calls(
            tool_calls, recent, {}, max_workers=4, verbose=False, debug=False
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        assert len(records) == 3
        assert [r.result["tag"] for r in records] == ["a", "b", "c"]
        # Sequential would take ~3 * sleep_ms = 240ms. Parallel should be well under 2x sleep_ms.
        assert elapsed_ms < sleep_ms * 2, f"Expected parallel dispatch; took {elapsed_ms:.0f}ms"
    finally:
        TOOLS.pop("_sleep", None)


def test_dispatch_preserves_original_order_even_when_fast_tools_finish_first():
    """Fast-then-slow ordering: results must still line up with tool_calls indexing."""
    TOOLS["_delay"] = ToolEntry(
        name="_delay",
        description="Delay then return tag.",
        model=_SleepArgs,
        func=lambda **kw: time.sleep(_SleepArgs(**kw).ms / 1000.0) or {"tag": kw["tag"]},
    )
    try:
        # slow (120ms) then two fast (10ms each) — fast ones should finish first
        tool_calls = [
            _tc(0, "_delay", '{"ms": 120, "tag": "slow"}'),
            _tc(1, "_delay", '{"ms": 10, "tag": "fast1"}'),
            _tc(2, "_delay", '{"ms": 10, "tag": "fast2"}'),
        ]
        recent: deque[str] = deque(maxlen=4)
        records = _dispatch_tool_calls(
            tool_calls, recent, {}, max_workers=4, verbose=False, debug=False
        )
        assert [r.result["tag"] for r in records] == ["slow", "fast1", "fast2"]
    finally:
        TOOLS.pop("_delay", None)


def test_within_batch_duplicate_is_blocked():
    """If the model emits the same tool call twice in one turn, the second is blocked."""
    call_count = {"n": 0}

    def _impl(_: _NoArgs) -> dict:
        call_count["n"] += 1
        return {"n": call_count["n"]}

    TOOLS["_bcount"] = ToolEntry(
        name="_bcount",
        description="Count.",
        model=_NoArgs,
        func=lambda **_: _impl(_NoArgs()),
    )
    try:
        tool_calls = [
            _tc(0, "_bcount", "{}"),
            _tc(1, "_bcount", "{}"),
        ]
        recent: deque[str] = deque(maxlen=4)
        records = _dispatch_tool_calls(
            tool_calls, recent, {}, max_workers=4, verbose=False, debug=False
        )
        assert records[0].blocked is False
        assert records[1].blocked is True
        assert "Duplicate call" in records[1].result["error"]
        assert call_count["n"] == 1  # only the first one executed
    finally:
        TOOLS.pop("_bcount", None)


def test_dispatch_against_history_blocks_repeat():
    """Fingerprint already in recent_calls → blocked without re-execution."""
    call_count = {"n": 0}

    def _impl(_: _NoArgs) -> dict:
        call_count["n"] += 1
        return {"n": call_count["n"]}

    TOOLS["_hcount"] = ToolEntry(
        name="_hcount",
        description="Count.",
        model=_NoArgs,
        func=lambda **_: _impl(_NoArgs()),
    )
    try:
        tool_calls = [_tc(0, "_hcount", "{}")]
        recent: deque[str] = deque(maxlen=4)
        # Prime with the fingerprint the dispatch would generate
        from agent.loop import _fingerprint

        recent.append(_fingerprint("_hcount", "{}"))

        records = _dispatch_tool_calls(
            tool_calls, recent, {}, max_workers=4, verbose=False, debug=False
        )
        assert records[0].blocked is True
        assert call_count["n"] == 0
    finally:
        TOOLS.pop("_hcount", None)


def test_duplicate_error_escalates_on_repeated_blocks():
    """First block uses the gentle message; second+ escalate to the STOP variant."""
    from agent.loop import _fingerprint

    TOOLS["_repeat"] = ToolEntry(
        name="_repeat",
        description="Repeat.",
        model=_NoArgs,
        func=lambda **_: {"ok": True},
    )
    try:
        recent: deque[str] = deque(maxlen=4)
        block_counts: dict[str, int] = {}
        fp = _fingerprint("_repeat", "{}")
        recent.append(fp)  # pretend this call just executed last round

        # First re-attempt → blocked, gentle message.
        r1 = _dispatch_tool_calls(
            [_tc(0, "_repeat", "{}")],
            recent,
            block_counts,
            max_workers=1,
            verbose=False,
            debug=False,
        )
        assert r1[0].blocked is True
        assert "Duplicate call" in r1[0].result["error"]
        assert block_counts[fp] == 1

        # Second re-attempt → still blocked, escalated message.
        r2 = _dispatch_tool_calls(
            [_tc(1, "_repeat", "{}")],
            recent,
            block_counts,
            max_workers=1,
            verbose=False,
            debug=False,
        )
        assert r2[0].blocked is True
        assert "STOP" in r2[0].result["error"]
        assert block_counts[fp] == 2
    finally:
        TOOLS.pop("_repeat", None)


def test_block_counter_resets_when_fingerprint_executes_again():
    """Once a blocked fp ages out and executes, its counter resets."""
    from agent.loop import _fingerprint

    TOOLS["_reset"] = ToolEntry(
        name="_reset",
        description="Reset.",
        model=_NoArgs,
        func=lambda **_: {"ok": True},
    )
    try:
        recent: deque[str] = deque(maxlen=4)
        block_counts: dict[str, int] = {}
        fp = _fingerprint("_reset", "{}")
        block_counts[fp] = 5  # simulate prior blocks

        # fp not in recent_calls, so it executes — counter should be cleared.
        _dispatch_tool_calls(
            [_tc(0, "_reset", "{}")],
            recent,
            block_counts,
            max_workers=1,
            verbose=False,
            debug=False,
        )
        assert fp not in block_counts
    finally:
        TOOLS.pop("_reset", None)


def test_tool_exception_becomes_error_result():
    """If a tool raises, dispatch catches it and returns an error dict."""
    TOOLS["_boom"] = ToolEntry(
        name="_boom",
        description="Raises.",
        model=_NoArgs,
        func=lambda **_: (_ for _ in ()).throw(RuntimeError("kaboom")),
    )
    try:
        tool_calls = [_tc(0, "_boom", "{}")]
        records = _dispatch_tool_calls(
            tool_calls, deque(maxlen=4), {}, max_workers=4, verbose=False, debug=False
        )
        assert records[0].blocked is False
        assert "error" in records[0].result
        assert "kaboom" in records[0].result["error"]
    finally:
        TOOLS.pop("_boom", None)


def test_context_var_propagates_into_worker_threads():
    """memory_store binding must follow the call into the thread pool."""
    from agent.memory import get_active_store, use_memory_store

    seen_stores: list[Any] = []
    seen_threads: set[int] = set()
    lock = threading.Lock()

    class _Store:
        def read(self) -> str:
            return "marker"

        def append(self, fact: str) -> None:
            pass

    def _impl(_: _NoArgs) -> dict:
        with lock:
            seen_threads.add(threading.get_ident())
            seen_stores.append(get_active_store())
        time.sleep(0.05)  # ensure overlap so ≥2 threads actually engage
        return {"ok": True}

    TOOLS["_check_ctx"] = ToolEntry(
        name="_check_ctx",
        description="Check ctx.",
        model=_NoArgs,
        func=lambda **_: _impl(_NoArgs()),
    )
    try:
        store = _Store()
        with use_memory_store(store):  # type: ignore[arg-type]
            tool_calls = [
                _tc(0, "_check_ctx", "{}"),
                _tc(1, "_check_ctx", '{"dummy":1}'),
                _tc(2, "_check_ctx", '{"dummy":2}'),
            ]
            _dispatch_tool_calls(
                tool_calls, deque(maxlen=4), {}, max_workers=4, verbose=False, debug=False
            )
        assert len(seen_stores) == 3
        assert all(s is store for s in seen_stores)
        # At least 2 distinct worker threads participated (otherwise "propagation" is vacuous).
        assert len(seen_threads) >= 2
    finally:
        TOOLS.pop("_check_ctx", None)


def test_single_call_uses_inline_dispatch():
    """A one-call batch skips the pool (micro-optimization); still returns a correct record."""
    TOOLS["_single"] = ToolEntry(
        name="_single",
        description="Single.",
        model=_NoArgs,
        func=lambda **_: {"ok": True},
    )
    try:
        tool_calls = [_tc(0, "_single", "{}")]
        records = _dispatch_tool_calls(
            tool_calls, deque(maxlen=4), {}, max_workers=8, verbose=False, debug=False
        )
        assert records[0].result == {"ok": True}
        assert records[0].blocked is False
    finally:
        TOOLS.pop("_single", None)


def test_max_workers_one_forces_sequential(monkeypatch: pytest.MonkeyPatch):
    """max_workers=1 still works — ensures the `workers==1` branch dispatches inline."""
    order: list[str] = []
    lock = threading.Lock()

    def _impl(args: _SleepArgs) -> dict:
        with lock:
            order.append(f"enter:{args.tag}")
        time.sleep(args.ms / 1000.0)
        with lock:
            order.append(f"exit:{args.tag}")
        return {"tag": args.tag}

    TOOLS["_seq"] = ToolEntry(
        name="_seq",
        description="Seq.",
        model=_SleepArgs,
        func=lambda **kw: _impl(_SleepArgs(**kw)),
    )
    try:
        tool_calls = [
            _tc(0, "_seq", '{"ms": 30, "tag": "a"}'),
            _tc(1, "_seq", '{"ms": 30, "tag": "b"}'),
        ]
        _dispatch_tool_calls(
            tool_calls, deque(maxlen=4), {}, max_workers=1, verbose=False, debug=False
        )
        # No interleaving when workers=1.
        assert order == ["enter:a", "exit:a", "enter:b", "exit:b"]
    finally:
        TOOLS.pop("_seq", None)


# ---------- End-to-end: parallel dispatch via run_agent ----------


def test_run_agent_dispatches_parallel_tool_calls_from_one_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_memory
):
    """The LLM emits two tool_calls in one turn → both should run concurrently."""
    sleep_ms = 80
    started_at: dict[str, float] = {}
    lock = threading.Lock()

    def _impl(args: _SleepArgs) -> dict:
        with lock:
            started_at[args.tag] = time.monotonic()
        time.sleep(args.ms / 1000.0)
        return {"tag": args.tag}

    TOOLS["_e2e"] = ToolEntry(
        name="_e2e",
        description="E2E sleep.",
        model=_SleepArgs,
        func=lambda **kw: _impl(_SleepArgs(**kw)),
    )

    from unittest.mock import MagicMock

    def _mk_chunk(content: str | None = None, tool_call: dict | None = None):
        chunk = MagicMock()
        delta = MagicMock()
        delta.content = content
        if tool_call is not None:
            tc_mock = MagicMock()
            tc_mock.index = tool_call["index"]
            tc_mock.id = tool_call.get("id")
            fn = MagicMock()
            fn.name = tool_call.get("name")
            fn.arguments = tool_call.get("arguments")
            tc_mock.function = fn
            delta.tool_calls = [tc_mock]
        else:
            delta.tool_calls = None
        choice = MagicMock()
        choice.delta = delta
        chunk.choices = [choice]
        return chunk

    # One turn with two tool_calls (different indices), then a final answer.
    turn1 = [
        _mk_chunk(
            tool_call={
                "index": 0,
                "id": "c1",
                "name": "_e2e",
                "arguments": f'{{"ms": {sleep_ms}, "tag": "x"}}',
            }
        ),
        _mk_chunk(
            tool_call={
                "index": 1,
                "id": "c2",
                "name": "_e2e",
                "arguments": f'{{"ms": {sleep_ms}, "tag": "y"}}',
            }
        ),
    ]
    turn2 = [_mk_chunk(content="done")]

    client = MagicMock()
    turn_iter = iter([turn1, turn2])
    client.chat.completions.create.side_effect = lambda **_: iter(next(turn_iter))
    client.models.list.return_value = MagicMock(
        data=[
            MagicMock(
                **{
                    "model_dump.return_value": {
                        "id": "test-model",
                        "status": {"args": ["--ctx-size", "131072"]},
                    }
                }
            )
        ]
    )
    monkeypatch.setattr("agent.loop.create_client", lambda local=True: client)
    monkeypatch.setattr("agent.loop.active_model", lambda local=True: "test-model")

    try:
        events: list[IterationEvent] = []
        t0 = time.monotonic()
        answer, _ = run_agent("go", verbose=False, on_iteration=events.append)
        elapsed_ms = (time.monotonic() - t0) * 1000

        assert answer == "done"
        # Parallel should be well under 2x sleep_ms.
        assert elapsed_ms < sleep_ms * 2.5, (
            f"Expected parallel dispatch at loop level; took {elapsed_ms:.0f}ms"
        )
        # Start times should be within ~one sleep of each other (both in-flight simultaneously).
        assert abs(started_at["x"] - started_at["y"]) * 1000 < sleep_ms
        # Iteration event order and record order preserved.
        assert len(events[0].tool_calls) == 2
        assert [r.args["tag"] for r in events[0].tool_calls] == ["x", "y"]
    finally:
        TOOLS.pop("_e2e", None)


def test_run_agent_escapes_when_all_calls_are_blocked_duplicates(
    monkeypatch: pytest.MonkeyPatch, tmp_memory
):
    """When every tool call in a turn is a blocked duplicate, the loop must
    force a final-answer turn with tool_choice='none' and exit — no more
    hammering the same call until max_iterations trips."""
    from unittest.mock import MagicMock

    TOOLS["_stuck"] = ToolEntry(
        name="_stuck",
        description="Stuck.",
        model=_NoArgs,
        func=lambda **_: {"v": 1},
    )

    def _mk_chunk(content: str | None = None, tool_call: dict | None = None):
        chunk = MagicMock()
        delta = MagicMock()
        delta.content = content
        if tool_call is not None:
            tc_mock = MagicMock()
            tc_mock.index = tool_call["index"]
            tc_mock.id = tool_call.get("id")
            fn = MagicMock()
            fn.name = tool_call.get("name")
            fn.arguments = tool_call.get("arguments")
            tc_mock.function = fn
            delta.tool_calls = [tc_mock]
        else:
            delta.tool_calls = None
        choice = MagicMock()
        choice.delta = delta
        chunk.choices = [choice]
        return chunk

    # Turn 1: one tool call — executes, fp enters recent_calls.
    # Turn 2: identical tool call — blocked. Loop must inject forced-answer msg.
    # Turn 3 (forced, tool_choice='none'): model returns final text.
    tool_call_payload = {
        "index": 0,
        "id": "c1",
        "name": "_stuck",
        "arguments": "{}",
    }
    turn1 = [_mk_chunk(tool_call={**tool_call_payload, "id": "c1"})]
    turn2 = [_mk_chunk(tool_call={**tool_call_payload, "id": "c2"})]
    turn3 = [_mk_chunk(content="giving up, here is what I have")]

    client = MagicMock()
    turn_iter = iter([turn1, turn2, turn3])
    seen_kwargs: list[dict] = []

    def _create(**kwargs):
        seen_kwargs.append(kwargs)
        return iter(next(turn_iter))

    client.chat.completions.create.side_effect = _create
    client.models.list.return_value = MagicMock(
        data=[
            MagicMock(
                **{
                    "model_dump.return_value": {
                        "id": "test-model",
                        "status": {"args": ["--ctx-size", "131072"]},
                    }
                }
            )
        ]
    )
    monkeypatch.setattr("agent.loop.create_client", lambda local=True: client)
    monkeypatch.setattr("agent.loop.active_model", lambda local=True: "test-model")

    try:
        events: list[IterationEvent] = []
        # max_iterations=10 deliberately — we want to prove the escape fires
        # before that limit, not because of it.
        answer, messages = run_agent(
            "go",
            verbose=False,
            max_iterations=10,
            on_iteration=events.append,
        )

        assert answer == "giving up, here is what I have"
        # Exactly 3 LLM calls: turn1, turn2, forced turn3. No more.
        assert len(seen_kwargs) == 3
        # Turn 3 was the forced final-answer turn.
        assert seen_kwargs[2]["tool_choice"] == "none"
        # The injected user message should be in the history right before turn 3.
        assert any(
            m.get("role") == "user" and "blocked duplicate" in m.get("content", "")
            for m in messages
        )
        # Iteration 2 should have a blocked record; iteration 3 is the forced
        # final-answer emission.
        blocked_iters = [ev for ev in events if any(r.blocked for r in ev.tool_calls)]
        assert blocked_iters, "expected at least one iteration with a blocked call"
        assert events[-1].final_answer == "giving up, here is what I have"
    finally:
        TOOLS.pop("_stuck", None)

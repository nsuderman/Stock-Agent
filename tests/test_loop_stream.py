"""Tests for the ReAct loop with a mocked OpenAI client."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.loop import run_agent


def _mk_chunk(content: str | None = None, tool_call: dict | None = None):
    """Construct a chat.completions.chunk-like object with a single delta."""
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


def _mk_client(turns: list[list[Any]]):
    """Make a client whose chat.completions.create(...) returns chunks from each turn in order."""
    client = MagicMock()
    turn_iter = iter(turns)

    def create(**kwargs):
        return iter(next(turn_iter))

    client.chat.completions.create.side_effect = create
    # Context probe → large window so compaction doesn't fire.
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
    return client


@pytest.fixture(autouse=True)
def _patch_client(monkeypatch: pytest.MonkeyPatch):
    """Replace `create_client` + `active_model` in the loop module."""
    # Each test overrides via `set_client` below.
    monkeypatch.setattr("agent.loop.active_model", lambda local=True: "test-model")


def test_single_turn_no_tool_calls(monkeypatch: pytest.MonkeyPatch, tmp_memory):
    """Agent returns a plain answer without invoking tools."""
    turns = [[_mk_chunk(content="Hello, world.")]]
    client = _mk_client(turns)
    monkeypatch.setattr("agent.loop.create_client", lambda local=True: client)

    answer, messages = run_agent("say hi", verbose=False)
    assert answer == "Hello, world."
    # system + user + assistant
    assert len(messages) >= 3
    assert messages[-1]["role"] == "assistant"


def test_tool_call_then_final_answer(monkeypatch: pytest.MonkeyPatch, tmp_memory):
    """Agent calls one tool, gets a result, then answers."""
    # Register a dummy tool for this test only.
    from pydantic import BaseModel

    from agent.tools.base import TOOLS

    class EchoArgs(BaseModel):
        msg: str

    def _echo_impl(args: EchoArgs) -> dict:
        return {"echo": args.msg}

    from agent.tools.base import ToolEntry

    entry = ToolEntry(
        name="_echo",
        description="Echo a message.",
        model=EchoArgs,
        func=lambda **kw: _echo_impl(EchoArgs(**kw)),
    )
    TOOLS["_echo"] = entry
    try:
        turn1 = [
            _mk_chunk(
                tool_call={
                    "index": 0,
                    "id": "call_1",
                    "name": "_echo",
                    "arguments": '{"msg": "ping"}',
                }
            ),
        ]
        turn2 = [_mk_chunk(content="You said ping.")]
        client = _mk_client([turn1, turn2])
        monkeypatch.setattr("agent.loop.create_client", lambda local=True: client)

        answer, messages = run_agent("echo test", verbose=False)
        assert "ping" in answer
        # Verify the tool-result message is in history
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert json.loads(tool_msgs[0]["content"])["echo"] == "ping"
    finally:
        TOOLS.pop("_echo", None)


def test_duplicate_call_blocked(monkeypatch: pytest.MonkeyPatch, tmp_memory):
    """Two identical tool calls in a row → second is blocked with a 'duplicate' error."""
    from pydantic import BaseModel

    from agent.tools.base import TOOLS, ToolEntry

    call_count = {"n": 0}

    class NoArgs(BaseModel):
        pass

    def _count(_: NoArgs):
        call_count["n"] += 1
        return {"count": call_count["n"]}

    TOOLS["_counter"] = ToolEntry(
        name="_counter",
        description="Increment a counter.",
        model=NoArgs,
        func=lambda **_: _count(NoArgs()),
    )
    try:
        dup_tc = {"index": 0, "id": "c1", "name": "_counter", "arguments": "{}"}
        dup_tc2 = {"index": 0, "id": "c2", "name": "_counter", "arguments": "{}"}
        turns = [
            [_mk_chunk(tool_call=dup_tc)],
            [_mk_chunk(tool_call=dup_tc2)],
            [_mk_chunk(content="done.")],
        ]
        client = _mk_client(turns)
        monkeypatch.setattr("agent.loop.create_client", lambda local=True: client)

        _, messages = run_agent("count twice", verbose=False)
        # The real tool should have run once; the second call was blocked.
        assert call_count["n"] == 1
        # A tool message containing 'Duplicate call' should exist.
        tool_contents = [m.get("content", "") for m in messages if m.get("role") == "tool"]
        assert any("Duplicate call" in c for c in tool_contents)
    finally:
        TOOLS.pop("_counter", None)


def test_max_iterations_reached(monkeypatch: pytest.MonkeyPatch, tmp_memory):
    """If the model keeps calling tools beyond max_iterations, agent returns a stop message."""
    from pydantic import BaseModel

    from agent.tools.base import TOOLS, ToolEntry

    class NoArgs(BaseModel):
        pass

    TOOLS["_spin"] = ToolEntry(
        name="_spin",
        description="Spin.",
        model=NoArgs,
        func=lambda **_: {"ok": True},
    )
    try:
        # Each turn returns a tool call with unique args so the dedup guard doesn't fire.
        def chunks_with_unique_args(i: int):
            return [
                _mk_chunk(
                    tool_call={
                        "index": 0,
                        "id": f"c{i}",
                        "name": "_spin",
                        "arguments": f'{{"tag": {i}}}',
                    }
                )
            ]

        turns = [chunks_with_unique_args(i) for i in range(5)]
        client = _mk_client(turns)
        monkeypatch.setattr("agent.loop.create_client", lambda local=True: client)

        answer, _ = run_agent("loop forever", max_iterations=3, verbose=False)
        assert "max_iterations" in answer or "Stopped" in answer
    finally:
        TOOLS.pop("_spin", None)


def test_think_tags_stripped_from_history(monkeypatch: pytest.MonkeyPatch, tmp_memory):
    """Final stored content shouldn't contain <think> blocks."""
    turns = [[_mk_chunk(content="<think>hidden</think>visible answer")]]
    client = _mk_client(turns)
    monkeypatch.setattr("agent.loop.create_client", lambda local=True: client)

    answer, messages = run_agent("test think", verbose=False)
    assert "<think>" not in answer
    assert "hidden" not in answer
    assert "visible answer" in answer
    last_assistant = [m for m in messages if m["role"] == "assistant"][-1]
    assert "<think>" not in last_assistant["content"]

"""ReAct loop: stream → tool_calls → dispatch → loop → final answer."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
import time
from collections import deque
from typing import Any

from openai import OpenAI

from agent.compaction import THINK_RE, compact_if_needed
from agent.llm import active_model, create_client
from agent.logging_setup import get_logger
from agent.prompt import build_system_prompt
from agent.tools import TOOLS, invoke_tool, openai_tool_schemas

log = get_logger(__name__)

MAX_TOOL_RESULT_CHARS = 8000


def _trunc(s: str, n: int = MAX_TOOL_RESULT_CHARS) -> str:
    return s if len(s) <= n else s[:n] + f"\n... [truncated, {len(s) - n} more chars]"


def _print(s: str = "", end: str = "\n") -> None:
    sys.stdout.write(s + end)
    sys.stdout.flush()


def _result_summary(result: Any) -> str:
    if isinstance(result, dict):
        if "error" in result:
            return f"ERROR: {str(result['error'])[:160]}"
        rows = result.get("rows")
        if isinstance(rows, list):
            n = result.get("count", len(rows))
            if rows and isinstance(rows[0], dict):
                cols = ", ".join(list(rows[0].keys())[:8])
                return f"{n} row(s); cols: {cols}"
            return f"{n} row(s)"
        if "columns" in result:
            return f"{len(result['columns'])} columns"
        parts = []
        for k, v in result.items():
            if isinstance(v, list):
                parts.append(f"{k}[{len(v)}]")
            elif isinstance(v, dict):
                parts.append(f"{k}{{{len(v)}}}")
            else:
                parts.append(k)
        return "dict: " + ", ".join(parts)
    if isinstance(result, list):
        return f"list with {len(result)} item(s)"
    return str(result)[:160]


class Spinner:
    """Audio-equalizer-style spinner driven by a daemon thread.

    Renders five Unicode eighth-block bars side-by-side, each on its own
    phase-offset sine wave so you get a smooth wave effect — roughly matches
    the look of a CSS staggered-bar animation, but in a single terminal line.

    Safe to call `stop()` repeatedly; no-op if not running or stdout isn't a TTY.
    Clears its own line on stop so subsequent prints start clean.
    """

    # Eighth-block glyphs indexed 0..8 (index 0 = space, 1..8 = ▁..█).
    BLOCKS = " ▁▂▃▄▅▆▇█"
    # Phase offsets approximate the CSS delay sequence scaled to [0, 1].
    PHASES = (0.0, 0.167, 0.333, 0.5, 0.667)
    PERIOD = 0.9  # seconds for one full wave cycle (matches CSS)
    MIN_LEVEL = 0.4  # bars never fully flatten (matches CSS scaleY 0.4 → 1.0)

    def __init__(self, message: str = "thinking", interval: float = 0.06) -> None:
        self.message = message
        self.interval = interval
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Width reserved when erasing: 2-space prefix + bars + space + message.
        self._width = 2 + len(self.PHASES) + 1 + len(self.message) + 2

    @property
    def running(self) -> bool:
        return self._thread is not None

    def start(self) -> None:
        if self.running or not sys.stdout.isatty():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.running:
            return
        self._stop.set()
        assert self._thread is not None
        self._thread.join(timeout=0.5)
        self._thread = None
        sys.stdout.write("\r" + " " * self._width + "\r")
        sys.stdout.flush()

    def _render(self, t: float) -> str:
        """Compute the bar string for elapsed time `t` (seconds)."""
        chars = []
        for phase in self.PHASES:
            # Sine wave in [0, 1] for this bar.
            wave = 0.5 + 0.5 * math.sin(2 * math.pi * ((t / self.PERIOD) - phase))
            # Squash into [MIN_LEVEL, 1.0] so bars never flatten completely.
            level = self.MIN_LEVEL + (1.0 - self.MIN_LEVEL) * wave
            idx = max(1, min(8, int(level * 8)))
            chars.append(self.BLOCKS[idx])
        return "".join(chars)

    def _run(self) -> None:
        start = time.monotonic()
        while not self._stop.is_set():
            bars = self._render(time.monotonic() - start)
            sys.stdout.write(f"\r  {bars} {self.message}")
            sys.stdout.flush()
            if self._stop.wait(self.interval):
                break

    def __enter__(self) -> Spinner:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


def _fingerprint(name: str, raw_args: str) -> str:
    """Stable hash of (tool name + arguments) for duplicate-call detection."""
    try:
        canonical = json.dumps(json.loads(raw_args or "{}"), sort_keys=True)
    except json.JSONDecodeError:
        canonical = raw_args or ""
    return hashlib.md5(f"{name}|{canonical}".encode()).hexdigest()


def _stream_turn(
    client: OpenAI,
    model: str,
    messages: list[dict],
    *,
    local: bool,
    verbose: bool,
) -> tuple[str, list[dict] | None]:
    """Stream one LLM turn. Returns (content_str, tool_calls_list|None).

    Content (including <think> blocks and model narration) is accumulated silently
    so the user sees only tool-call status. The final answer is rendered by the
    caller once the last iteration returns with no tool_calls.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": openai_tool_schemas(),
        "tool_choice": "auto",
        "temperature": 0.3,
        "stream": True,
    }
    kwargs["max_tokens" if local else "max_completion_tokens"] = 4096

    stream = client.chat.completions.create(**kwargs)

    content = ""
    tool_calls_acc: dict[int, dict] = {}
    printed_names: set[int] = set()
    spinner = Spinner("thinking") if verbose else None
    if spinner:
        spinner.start()

    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if getattr(delta, "content", None):
                content += delta.content

            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = tc.index
                    entry = tool_calls_acc.setdefault(
                        idx, {"id": None, "name": "", "arguments": ""}
                    )
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            entry["name"] += tc.function.name
                        if tc.function.arguments:
                            entry["arguments"] += tc.function.arguments
                    if verbose and entry["name"] and idx not in printed_names:
                        if spinner:
                            spinner.stop()
                        _print(f"  → {entry['name']}(...)")
                        printed_names.add(idx)
    finally:
        if spinner:
            spinner.stop()

    content = THINK_RE.sub("", content).strip()

    tool_calls = None
    if tool_calls_acc:
        tool_calls = [
            {
                "id": tool_calls_acc[i]["id"],
                "type": "function",
                "function": {
                    "name": tool_calls_acc[i]["name"],
                    "arguments": tool_calls_acc[i]["arguments"],
                },
            }
            for i in sorted(tool_calls_acc)
        ]

    return content, tool_calls


def run_agent(
    question: str,
    *,
    max_iterations: int = 12,
    local: bool = True,
    verbose: bool = True,
    prior_messages: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """Run the agent on `question`. Returns (answer, full_messages_list)."""
    if not TOOLS:
        raise RuntimeError("No tools registered — import agent.tools before calling run_agent.")

    client = create_client(local=local)
    model = active_model(local=local)

    system_content = build_system_prompt()
    messages: list[dict] = list(prior_messages or [])
    if messages and messages[0].get("role") == "system":
        messages[0] = {"role": "system", "content": system_content}
    else:
        messages.insert(0, {"role": "system", "content": system_content})
    messages.append({"role": "user", "content": question})

    recent_calls: deque[str] = deque(maxlen=4)
    status_cb = _print if verbose else None

    for i in range(1, max_iterations + 1):
        if verbose:
            _print(f"[iter {i}]")

        messages = compact_if_needed(
            messages, client=client, model=model, local=local, status_callback=status_cb
        )

        content, tool_calls = _stream_turn(client, model, messages, local=local, verbose=verbose)

        assistant_entry: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            assistant_entry["tool_calls"] = tool_calls
        messages.append(assistant_entry)

        if not tool_calls:
            return content or "", messages

        for tc in tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"] or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            if verbose:
                arg_preview = raw_args if len(raw_args) <= 140 else raw_args[:140] + "..."
                _print(f"    args: {arg_preview}")

            fp = _fingerprint(name, raw_args)
            if fp in recent_calls:
                result: Any = {
                    "error": (
                        "Duplicate call: you just invoked this exact tool with these exact "
                        "arguments in one of the last 4 tool calls. The result has not changed. "
                        "Either use the prior result to answer the user, or try a different "
                        "tool / different arguments. If you already have enough data, STOP "
                        "calling tools and provide the final answer."
                    )
                }
                if verbose:
                    _print("    ← BLOCKED: duplicate tool call")
            else:
                recent_calls.append(fp)
                result = invoke_tool(name, args)
                if verbose:
                    _print(f"    ← {_result_summary(result)}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": _trunc(json.dumps(result, default=str)),
                }
            )

    return "Stopped: reached max_iterations without a final answer.", messages

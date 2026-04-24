"""ReAct loop: stream → tool_calls → dispatch → loop → final answer."""

from __future__ import annotations

import contextvars
import hashlib
import json
import math
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from agent.compaction import THINK_RE, compact_if_needed
from agent.llm import active_model, create_client
from agent.logging_setup import get_logger
from agent.memory import MemoryStore, use_memory_store
from agent.prompt import build_system_prompt
from agent.tools import TOOLS, invoke_tool, openai_tool_schemas

log = get_logger(__name__)

MAX_TOOL_RESULT_CHARS = 8000


def _serialize_tool_result(result: Any, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Serialize a tool result to JSON, capped at `limit` chars.

    If the full result exceeds the limit, returns a VALID JSON envelope
    describing the truncation with a (string-wrapped) prose preview, so the
    model never receives malformed JSON.
    """
    full = json.dumps(result, default=str)
    if len(full) <= limit:
        return full
    # Reserve ~200 chars for envelope keys and metadata.
    preview_chars = max(0, limit - 200)
    envelope = {
        "truncated": True,
        "original_size_chars": len(full),
        "message": (
            f"Tool result was {len(full)} chars, exceeding the {limit}-char limit. "
            "Only a string preview is shown below. Call the tool again with narrower "
            "arguments (smaller limit, specific columns, narrower date range) to get "
            "the actual data."
        ),
        "preview": full[:preview_chars],
    }
    return json.dumps(envelope, default=str)


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
        # \r + ANSI clear-entire-line so stale log chatter doesn't linger.
        sys.stdout.write("\r\033[2K")
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
            # \033[K clears from cursor to end-of-line after the redraw, so any
            # stray log tail past our text is wiped each frame.
            sys.stdout.write(f"\r  {bars} {self.message}\033[K")
            sys.stdout.flush()
            if self._stop.wait(self.interval):
                break

    def __enter__(self) -> Spinner:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


@dataclass
class ToolCallRecord:
    """One tool invocation within an iteration, as seen by an `on_iteration` consumer."""

    name: str
    args: dict
    blocked: bool
    result: Any
    result_summary: str


@dataclass
class IterationEvent:
    """Fires once per ReAct loop iteration via `run_agent(..., on_iteration=...)`.

    - On iterations that call tools: `tool_calls` is populated, `final_answer` is None.
    - On the terminal iteration (no tool calls): `tool_calls` is empty and
      `final_answer` is the text returned to the user.
    """

    iteration: int
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    final_answer: str | None = None


def _fingerprint(name: str, raw_args: str) -> str:
    """Stable hash of (tool name + arguments) for duplicate-call detection."""
    try:
        canonical = json.dumps(json.loads(raw_args or "{}"), sort_keys=True)
    except json.JSONDecodeError:
        canonical = raw_args or ""
    return hashlib.md5(f"{name}|{canonical}".encode()).hexdigest()


_DUPLICATE_ERROR_FIRST = (
    "Duplicate call: you just invoked this exact tool with these exact "
    "arguments in one of the last 4 tool calls. The result has not changed. "
    "Either use the prior result to answer the user, or try a different "
    "tool / different arguments. If you already have enough data, STOP "
    "calling tools and provide the final answer."
)
_DUPLICATE_ERROR_REPEAT = (
    "STOP. You have emitted this identical tool call {count} times in a row "
    "and every one has been blocked. Retrying will not change anything. "
    "Give the user a final answer NOW using what you already have — do not "
    "emit any more tool calls."
)


def _duplicate_error(count: int) -> dict[str, str]:
    """Block-response payload, escalated on repeated blocks of the same fingerprint."""
    if count >= 2:
        return {"error": _DUPLICATE_ERROR_REPEAT.format(count=count)}
    return {"error": _DUPLICATE_ERROR_FIRST}


def _dispatch_tool_calls(
    tool_calls: list[dict],
    recent_calls: deque[str],
    block_counts: dict[str, int],
    *,
    max_workers: int,
    verbose: bool,
    debug: bool,
) -> list[ToolCallRecord]:
    """Dispatch all tool calls for a single ReAct iteration, in parallel.

    - Parses args + fingerprints up front. A call is blocked if its fingerprint
      matches `recent_calls` OR another call earlier in the same batch (the
      model emitted the identical call twice in one turn).
    - `block_counts` tracks consecutive blocks per fingerprint across the whole
      run; the synthetic error response escalates at count >= 2 to break the
      model out of "hammer the same call forever" loops.
    - Non-blocked calls run concurrently in a ThreadPoolExecutor; tools are
      I/O-bound (PG, yfinance, SEC EDGAR), so a thread pool is the right fit.
    - Each worker runs inside a per-task copy of the main thread's ContextVar
      context, so `agent.memory.use_memory_store` bindings propagate correctly.
    - Results are returned in the original `tool_calls` order. The tool
      messages that caller will append MUST preserve that order so the LLM
      sees responses lined up with its tool_call ids.
    """
    n = len(tool_calls)
    parsed: list[dict[str, Any]] = []
    blocked_flags: list[bool] = []
    fingerprints: list[str] = []
    seen_in_batch: set[str] = set()

    for tc in tool_calls:
        name = tc["function"]["name"]
        raw_args = tc["function"]["arguments"] or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}

        fp = _fingerprint(name, raw_args)
        blocked = fp in recent_calls or fp in seen_in_batch
        if blocked:
            block_counts[fp] = block_counts.get(fp, 0) + 1
        else:
            seen_in_batch.add(fp)
            recent_calls.append(fp)
            # A fingerprint that was previously blocked has now aged out and
            # executed — reset its counter so a fresh re-block starts at 1.
            block_counts.pop(fp, None)

        parsed.append({"name": name, "args": args, "raw_args": raw_args})
        blocked_flags.append(blocked)
        fingerprints.append(fp)

        if verbose and debug:
            arg_preview = raw_args if len(raw_args) <= 140 else raw_args[:140] + "..."
            _print(f"    args: {arg_preview}")

    results: list[Any] = [None] * n
    non_blocked = [i for i, b in enumerate(blocked_flags) if not b]

    if non_blocked:
        # Copy the main-thread context once per task. Each Context can only be
        # `.run()` on one thread at a time, so we can't share a single copy.
        contexts = {i: contextvars.copy_context() for i in non_blocked}

        def _run(idx: int) -> Any:
            p = parsed[idx]
            try:
                return contexts[idx].run(invoke_tool, p["name"], p["args"])
            except Exception as exc:  # defense-in-depth; tools normally catch themselves
                log.exception("Tool %s raised during dispatch", p["name"])
                return {"error": f"Tool {p['name']!r} raised: {exc!r}"}

        workers = min(len(non_blocked), max(1, max_workers))
        if workers == 1:
            # Skip pool overhead for a single call.
            for i in non_blocked:
                results[i] = _run(i)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tool") as ex:
                for idx, res in zip(non_blocked, ex.map(_run, non_blocked), strict=True):
                    results[idx] = res

    records: list[ToolCallRecord] = []
    for i in range(n):
        blocked = blocked_flags[i]
        if blocked:
            count = block_counts[fingerprints[i]]
            result = _duplicate_error(count)
        else:
            result = results[i]
        if verbose and debug:
            if blocked:
                _print(f"    ← BLOCKED: duplicate tool call (×{block_counts[fingerprints[i]]})")
            else:
                _print(f"    ← {_result_summary(result)}")
        records.append(
            ToolCallRecord(
                name=parsed[i]["name"],
                args=parsed[i]["args"],
                blocked=blocked,
                result=result,
                result_summary=_result_summary(result),
            )
        )
    return records


def _stream_turn(
    client: OpenAI,
    model: str,
    messages: list[dict],
    *,
    local: bool,
    verbose: bool,
    debug: bool,
    tool_choice: str = "auto",
) -> tuple[str, list[dict] | None]:
    """Stream one LLM turn. Returns (content_str, tool_calls_list|None).

    Content (including <think> blocks and model narration) is accumulated silently
    so the user sees only tool-call status. The final answer is rendered by the
    caller once the last iteration returns with no tool_calls.

    `tool_choice="none"` forces a final-answer turn (no tool calls allowed) —
    used by the main loop to break out when every call in an iteration was
    blocked as a duplicate.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": openai_tool_schemas(),
        "tool_choice": tool_choice,
        "temperature": 0.3,
        "stream": True,
    }
    kwargs["max_tokens" if local else "max_completion_tokens"] = 4096

    content = ""
    tool_calls_acc: dict[int, dict] = {}
    printed_names: set[int] = set()
    spinner = Spinner("thinking") if verbose else None
    # Start the spinner BEFORE `create()` — that call blocks on the initial HTTP
    # connection + response headers, which is part of what we want to visualize.
    if spinner:
        spinner.start()

    try:
        stream = client.chat.completions.create(**kwargs)
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
                    # In debug mode, announce each tool as its name arrives (and
                    # stop the spinner so it doesn't collide). In default mode,
                    # stay silent — run_agent prints a one-line summary below.
                    if verbose and debug and entry["name"] and idx not in printed_names:
                        if spinner:
                            spinner.stop()
                        _print(f"  → {entry['name']}(...)")
                        printed_names.add(idx)
    finally:
        if spinner:
            spinner.stop()

    stripped = THINK_RE.sub("", content).strip()
    # If the model put EVERYTHING inside <think> tags (empty answer after stripping)
    # fall back to the content without the outer think wrappers so the user sees
    # something instead of a blank answer block.
    if not stripped and content.strip():
        fallback = content.replace("<think>", "").replace("</think>", "").strip()
        content = fallback or content.strip()
    else:
        content = stripped

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
    max_iterations: int | None = None,
    max_tool_concurrency: int | None = None,
    local: bool = True,
    verbose: bool = True,
    debug: bool = False,
    prior_messages: list[dict] | None = None,
    memory_store: MemoryStore | None = None,
    on_iteration: Callable[[IterationEvent], None] | None = None,
) -> tuple[str, list[dict]]:
    """Run the agent on `question`. Returns (answer, full_messages_list).

    Parameters
    ----------
    max_iterations
        Cap on ReAct loop iterations. Defaults to `Settings.max_iterations` (12).
    max_tool_concurrency
        Max number of tool calls to dispatch in parallel per iteration. Defaults
        to `Settings.max_tool_concurrency` (8). Set to 1 for sequential dispatch.
    local
        Use the local LLM endpoint if True, remote otherwise.
    verbose / debug
        Control CLI-style progress printing to stdout. Library consumers usually
        want `verbose=False` and rely on `on_iteration` instead.
    prior_messages
        Prior conversation history to resume from. The leading system message
        (if any) is rebuilt against the current memory + date on every call.
    memory_store
        Optional per-call memory backing. If omitted, falls back to a file-based
        store at `Settings.memory_path`. Set this to isolate memory per user in
        a multi-tenant server. Bound via a ContextVar so concurrent async tasks
        and threads don't step on each other.
    on_iteration
        Optional callback invoked once per ReAct iteration with an
        `IterationEvent`. Use for SSE/WebSocket streaming of progress to a
        frontend; the final iteration carries `final_answer` populated.
    """
    if not TOOLS:
        raise RuntimeError("No tools registered — import agent.tools before calling run_agent.")

    from agent.config import get_settings

    settings = get_settings()
    if max_iterations is None:
        max_iterations = settings.max_iterations
    if max_tool_concurrency is None:
        max_tool_concurrency = settings.max_tool_concurrency

    with use_memory_store(memory_store):
        return _run_agent_inner(
            question,
            max_iterations=max_iterations,
            max_tool_concurrency=max_tool_concurrency,
            local=local,
            verbose=verbose,
            debug=debug,
            prior_messages=prior_messages,
            on_iteration=on_iteration,
        )


def _run_agent_inner(
    question: str,
    *,
    max_iterations: int,
    max_tool_concurrency: int,
    local: bool,
    verbose: bool,
    debug: bool,
    prior_messages: list[dict] | None,
    on_iteration: Callable[[IterationEvent], None] | None,
) -> tuple[str, list[dict]]:
    client = create_client(local=local)
    model = active_model(local=local)

    system_content = build_system_prompt()
    messages: list[dict] = list(prior_messages or [])
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        messages[0] = {"role": "system", "content": system_content}
    else:
        messages.insert(0, {"role": "system", "content": system_content})
    messages.append({"role": "user", "content": question})

    recent_calls: deque[str] = deque(maxlen=4)
    block_counts: dict[str, int] = {}
    status_cb = _print if verbose else None

    def _emit(event: IterationEvent) -> None:
        if on_iteration is None:
            return
        try:
            on_iteration(event)
        except Exception:
            # A consumer callback must never break the loop. Log and continue.
            log.exception("on_iteration callback raised; continuing.")

    for i in range(1, max_iterations + 1):
        # In debug mode, print the iteration header up front. In default mode we
        # compose a one-line summary AFTER the tools resolve so we can include
        # a [blocked] marker inline.
        if verbose and debug:
            _print(f"[iter {i}]")

        messages = compact_if_needed(
            messages, client=client, model=model, local=local, status_callback=status_cb
        )

        content, tool_calls = _stream_turn(
            client, model, messages, local=local, verbose=verbose, debug=debug
        )

        assistant_entry: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            assistant_entry["tool_calls"] = tool_calls
        messages.append(assistant_entry)

        if not tool_calls:
            final = content or ""
            _emit(IterationEvent(iteration=i, tool_calls=[], final_answer=final))
            return final, messages

        records = _dispatch_tool_calls(
            tool_calls,
            recent_calls,
            block_counts,
            max_workers=max_tool_concurrency,
            verbose=verbose,
            debug=debug,
        )

        for tc, rec in zip(tool_calls, records, strict=True):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": rec.name,
                    "content": _serialize_tool_result(rec.result),
                }
            )

        # Default mode: one compact line per iteration, AFTER tools resolved.
        if verbose and not debug:
            summary_parts = [
                f"{r.name}(...)" + (" [blocked]" if r.blocked else "") for r in records
            ]
            _print(f"[iter {i}] → {', '.join(summary_parts)}")

        _emit(IterationEvent(iteration=i, tool_calls=records, final_answer=None))

        # Escape hatch: if the model emitted only duplicate-blocked calls this
        # turn, it's stuck re-hammering. Force a tool-free final-answer turn
        # so the user gets something useful out of whatever did succeed before.
        if records and all(r.blocked for r in records):
            if verbose:
                _print(f"[iter {i}] all calls blocked — forcing final answer.")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Every tool call in your previous turn was a blocked "
                        "duplicate. Do not call any more tools. Answer me now "
                        "with whatever you have from earlier tool results — "
                        "if something is missing, say so plainly."
                    ),
                }
            )
            final_content, _ignored = _stream_turn(
                client,
                model,
                messages,
                local=local,
                verbose=verbose,
                debug=debug,
                tool_choice="none",
            )
            messages.append({"role": "assistant", "content": final_content or ""})
            _emit(IterationEvent(iteration=i + 1, tool_calls=[], final_answer=final_content or ""))
            return final_content or "", messages

    stop_msg = "Stopped: reached max_iterations without a final answer."
    _emit(IterationEvent(iteration=max_iterations, tool_calls=[], final_answer=stop_msg))
    return stop_msg, messages

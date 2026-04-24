"""Think-tag stripping, token estimation, two-stage compaction."""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from agent.config import get_settings
from agent.llm import get_context_window
from agent.logging_setup import get_logger

log = get_logger(__name__)

# Reasoning models (Qwen3, DeepSeek-R1 family) emit <think>...</think> blocks.
# We accumulate stream content silently and strip them post-hoc before storing
# to history and rendering the final answer.
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def estimate_tokens(messages: list[dict]) -> int:
    """Char/4 heuristic. Cheap and close enough for threshold decisions."""
    return len(json.dumps(messages, default=str)) // 4


def short_tool_summary(name: str, content: str) -> str:
    """Best-effort one-liner describing what a tool returned, from its JSON content."""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return f"[trimmed] {name} returned {len(content)} chars"
    if isinstance(payload, dict):
        if "error" in payload:
            return f"[trimmed] {name} ERROR: {str(payload['error'])[:100]}"
        rows = payload.get("rows")
        if isinstance(rows, list):
            n = payload.get("count", len(rows))
            cols = list(rows[0].keys())[:6] if rows and isinstance(rows[0], dict) else []
            return f"[trimmed] {name} → {n} rows; cols: {', '.join(cols)}"
        if "columns" in payload:
            return f"[trimmed] {name} → {len(payload['columns'])} columns"
        keys = list(payload.keys())[:6]
        return f"[trimmed] {name} → dict keys: {', '.join(keys)}"
    return f"[trimmed] {name} returned {type(payload).__name__}"


def stage1_trim(messages: list[dict], budget: int, keep_recent: int) -> tuple[list[dict], int]:
    """Replace old tool-result contents with one-line summaries."""
    if len(messages) <= keep_recent + 1:
        return messages, 0
    cutoff = len(messages) - keep_recent
    n_trimmed = 0
    out = list(messages)
    for i in range(1, cutoff):  # skip index 0 (system)
        m = out[i]
        if m.get("role") != "tool":
            continue
        content = m.get("content") or ""
        if content.startswith("[trimmed]"):
            continue
        summary = short_tool_summary(m.get("name") or "tool", content)
        out[i] = {**m, "content": summary}
        n_trimmed += 1
        if estimate_tokens(out) <= budget:
            break
    return out, n_trimmed


def stage2_summarize(
    messages: list[dict],
    *,
    client: OpenAI,
    model: str,
    local: bool,
    keep_recent: int,
    max_transcript_chars: int = 80_000,
) -> tuple[list[dict], int]:
    """Ask the LLM to summarize everything before the last `keep_recent` messages.

    `max_transcript_chars` caps the transcript we send to the summarizer. On very
    long histories, the transcript would otherwise exceed the context window and
    the summarization call itself would fail. When the cap kicks in, we keep the
    MOST RECENT portion of the old block (the tail) since that's what's most
    relevant for continuing the conversation.
    """
    if len(messages) <= keep_recent + 2:
        return messages, 0

    system = messages[0]
    old = messages[1:-keep_recent]
    tail = messages[-keep_recent:]
    if not old:
        return messages, 0

    transcript_parts = []
    for m in old:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if m.get("tool_calls"):
            calls = ", ".join(
                f"{tc['function']['name']}({tc['function']['arguments']})" for tc in m["tool_calls"]
            )
            transcript_parts.append(f"[{role}] tool_calls: {calls}\n{content}".strip())
        else:
            name = m.get("name")
            prefix = f"[{role}" + (f":{name}]" if name else "]")
            transcript_parts.append(f"{prefix} {content}")
    transcript = "\n\n".join(transcript_parts)
    if len(transcript) > max_transcript_chars:
        dropped = len(transcript) - max_transcript_chars
        # Keep the tail of the transcript — most relevant to what the user is
        # working on right now.
        transcript = (
            f"[omitted {dropped} chars of earlier history to fit context]\n\n"
            + transcript[-max_transcript_chars:]
        )
        log.info(
            "Stage 2 transcript capped at %d chars (dropped %d)", max_transcript_chars, dropped
        )

    summary_req: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You compact agent transcripts to fit in context. Summarize the conversation below "
                "in <=250 words. Preserve: (1) user goals and preferences, (2) concrete findings "
                "(numbers, symbols, dates), (3) unresolved threads and what tool calls have already "
                "been made. Omit small talk and tool-mechanics chatter. Output just the summary text, "
                "no preamble."
            ),
        },
        {"role": "user", "content": transcript},
    ]
    kwargs: dict[str, Any] = {"model": model, "messages": summary_req, "temperature": 0.2}
    kwargs["max_tokens" if local else "max_completion_tokens"] = 800
    try:
        resp = client.chat.completions.create(**kwargs)
        summary = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning("Stage 2 summarization failed: %s", e)
        summary = f"[compaction summary failed: {e}]"
    summary = THINK_RE.sub("", summary).strip() or "[previous conversation content]"

    collapsed = {
        "role": "assistant",
        "content": f"[compacted context — {len(old)} earlier messages merged]\n{summary}",
    }
    return [system, collapsed, *tail], len(old)


def compact_if_needed(
    messages: list[dict],
    *,
    client: OpenAI,
    model: str,
    local: bool,
    status_callback: Any = None,
) -> list[dict]:
    """Run Stage 1 and (if needed) Stage 2 compaction. Logs via `status_callback(msg)` if provided."""
    s = get_settings()
    window, source = get_context_window(client, model, local=local)
    budget = int(window * s.compact_at) - s.max_response_tokens
    if budget <= 0:
        return messages
    tokens = estimate_tokens(messages)
    if tokens <= budget:
        return messages

    def _emit(msg: str) -> None:
        log.info(msg)
        if status_callback:
            status_callback(msg)

    _emit(f"[compacting: ~{tokens} tokens > budget {budget} (ctx={window} from {source})]")

    messages, n_trimmed = stage1_trim(messages, budget, s.compact_keep_recent)
    after_s1 = estimate_tokens(messages)
    if n_trimmed:
        _emit(f"[stage 1: trimmed {n_trimmed} tool result(s), ~{tokens} → {after_s1} tokens]")
    if after_s1 <= budget:
        return messages

    # Cap the transcript so the summarization call itself doesn't overflow the
    # window. Reserve ~1500 tokens (~6000 chars) for the summary prompt + response
    # + safety margin, then convert the remaining token budget to chars (×4).
    max_transcript_chars = max(4_000, (window - 1500) * 4)
    messages, n_collapsed = stage2_summarize(
        messages,
        client=client,
        model=model,
        local=local,
        keep_recent=s.compact_keep_recent,
        max_transcript_chars=max_transcript_chars,
    )
    after_s2 = estimate_tokens(messages)
    if n_collapsed:
        _emit(
            f"[stage 2: summarized {n_collapsed} older message(s), ~{after_s1} → {after_s2} tokens]"
        )
    return messages

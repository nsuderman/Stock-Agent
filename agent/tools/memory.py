"""Persistent memory tool (appends to memory.md)."""

from __future__ import annotations

import datetime
from typing import Any

from pydantic import BaseModel, Field

from agent.config import get_settings
from agent.tools.base import tool


class RememberArgs(BaseModel):
    fact: str = Field(..., min_length=1, description="The fact to save, one sentence.")


@tool(
    description=(
        "Append a fact to persistent memory (memory.md). Use for durable user preferences "
        "or findings worth keeping across sessions. Do NOT use for ephemeral task state, "
        "tool outputs, or the current question."
    )
)
def remember(args: RememberArgs) -> dict[str, Any]:
    path = get_settings().memory_path
    fact = args.fact.strip()
    today = datetime.date.today().isoformat()
    entry = f"- [{today}] {fact}\n"
    if not path.exists():
        path.write_text("# Agent Memory\n\n")
    with path.open("a", encoding="utf-8") as f:
        f.write(entry)
    return {"saved": fact, "date": today}

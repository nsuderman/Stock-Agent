"""Persistent memory tool — appends via the active MemoryStore."""

from __future__ import annotations

import datetime
from typing import Any

from pydantic import BaseModel, Field

from agent.memory import get_active_store
from agent.tools.base import tool


class RememberArgs(BaseModel):
    fact: str = Field(..., min_length=1, description="The fact to save, one sentence.")


@tool(
    description=(
        "Append a fact to persistent memory. Use for durable user preferences "
        "or findings worth keeping across sessions. Do NOT use for ephemeral task state, "
        "tool outputs, or the current question."
    )
)
def remember(args: RememberArgs) -> dict[str, Any]:
    fact = args.fact.strip()
    if not fact:
        return {"error": "Fact is empty."}
    get_active_store().append(fact)
    return {"saved": fact, "date": datetime.date.today().isoformat()}

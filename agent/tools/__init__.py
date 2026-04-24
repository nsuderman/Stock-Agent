"""Tool package — imports every module to trigger `@tool` registration."""

# Import for side effects: each module registers its tools with the global TOOLS dict.
from agent.tools import (  # noqa: F401
    backtest,
    charts,
    db_meta,
    macro,
    market,
    memory,
    news,
    sec,
    sql,
)
from agent.tools.base import (
    TOOLS,
    ToolEntry,
    invoke_tool,
    openai_tool_schemas,
)

__all__ = ["TOOLS", "ToolEntry", "invoke_tool", "openai_tool_schemas"]

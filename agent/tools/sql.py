"""Read-only SQL escape hatch."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent.tools.base import contains_write_keyword, fetch, tool


class RunSqlArgs(BaseModel):
    query: str = Field(..., description="Read-only SELECT statement.")
    limit: int = Field(default=500, ge=1, le=10000)


@tool(
    description=(
        "Read-only SQL escape hatch. Use ONLY when no narrower tool fits (cross-table "
        "joins, custom aggregations). Rejects non-SELECT keywords. Fully qualify tables "
        "(stock.analytics, stock.backtest_results)."
    )
)
def run_sql(args: RunSqlArgs) -> dict[str, Any]:
    if contains_write_keyword(args.query):
        return {"error": "Only read-only SELECT queries are allowed."}
    rows = fetch(args.query, limit=args.limit)
    return {
        "count": len(rows),
        "rows": rows,
        "truncated": len(rows) == args.limit,
    }

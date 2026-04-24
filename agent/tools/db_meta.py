"""Schema / sample-row introspection tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import inspect

from agent.config import get_settings
from agent.db import get_engine
from agent.tools.base import contains_write_keyword, fetch, tool


class _NoArgs(BaseModel):
    pass


class DescribeTableArgs(BaseModel):
    # `schema` is a method on pydantic.BaseModel; use an alias so the LLM still sees "schema".
    model_config = ConfigDict(populate_by_name=True)

    db_schema: str = Field(..., alias="schema", description="Schema name, e.g. 'stock' or 'dev'")
    table: str = Field(..., description="Table or view name")


class SampleRowsArgs(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    db_schema: str = Field(..., alias="schema", description="Schema name.")
    table: str = Field(..., description="Table or view name.")
    limit: int = Field(default=3, ge=1, le=25, description="Rows to return (max 25).")


@tool(
    description=(
        "List every column name + type in {db_schema}.analytics. Call this FIRST whenever "
        "the user asks about indicators or you're unsure which columns exist."
    )
)
def list_analytics_columns(args: _NoArgs) -> dict[str, Any]:
    schema = get_settings().db_schema
    cols = inspect(get_engine()).get_columns("analytics", schema=schema)
    return {
        "schema": schema,
        "table": "analytics",
        "columns": [{"name": c["name"], "type": str(c["type"])} for c in cols],
    }


@tool(
    description=(
        "Return column names + types for ANY schema.table. Use before writing run_sql "
        "against an unfamiliar table (e.g. {backtest_schema}.backtest_results, "
        "{backtest_schema}.strategies)."
    )
)
def describe_table(args: DescribeTableArgs) -> dict[str, Any]:
    try:
        cols = inspect(get_engine()).get_columns(args.table, schema=args.db_schema)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if not cols:
        return {"error": f"No table {args.db_schema}.{args.table}"}
    return {
        "schema": args.db_schema,
        "table": args.table,
        "columns": [{"name": c["name"], "type": str(c["type"])} for c in cols],
    }


@tool(
    description=(
        "Return a few sample rows from any schema.table. Use this to discover JSON column "
        "shapes (trades, metrics, equity_curve) before writing aggregation SQL."
    )
)
def sample_rows(args: SampleRowsArgs) -> dict[str, Any]:
    if contains_write_keyword(args.db_schema) or contains_write_keyword(args.table):
        return {"error": "Invalid identifier."}
    rows = fetch(
        f'SELECT * FROM "{args.db_schema}"."{args.table}" LIMIT :lim',
        {"lim": args.limit},
    )
    return {
        "schema": args.db_schema,
        "table": args.table,
        "count": len(rows),
        "rows": rows,
    }

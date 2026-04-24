"""Tool registration + Pydantic-driven OpenAI schema generation."""

from __future__ import annotations

import datetime
import decimal
import re
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from agent.db import get_engine
from agent.logging_setup import get_logger

log = get_logger(__name__)

# Global registry populated by the @tool decorator.
TOOLS: dict[str, ToolEntry] = {}

_WRITE_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|COPY|MERGE)\b",
    re.IGNORECASE,
)


def contains_write_keyword(sql: str) -> bool:
    return bool(_WRITE_KEYWORDS.search(sql))


def coerce(v: Any) -> Any:
    """Normalize types that aren't JSON-serializable by default."""
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    if isinstance(v, decimal.Decimal):
        return float(v)
    return v


def fetch(
    sql: str, params: dict[str, Any] | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    """Execute a parameterized read-only query and return rows as list[dict]."""
    with get_engine().connect() as conn:
        result = conn.execute(text(sql), params or {})
        cols = list(result.keys())
        rows = result.fetchmany(limit) if limit else result.fetchall()
    return [{c: coerce(v) for c, v in zip(cols, row, strict=False)} for row in rows]


class ToolEntry:
    """Binds a callable, its argument model, and the generated OpenAI schema."""

    __slots__ = ("description", "func", "model", "name", "schema")

    def __init__(
        self,
        *,
        name: str,
        description: str,
        model: type[BaseModel] | None,
        func: Callable[..., Any],
    ) -> None:
        self.name = name
        self.description = description
        self.model = model
        self.func = func
        self.schema = self._build_schema()

    def _build_schema(self) -> dict[str, Any]:
        if self.model is None:
            params: dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
            }
        else:
            schema = self.model.model_json_schema(by_alias=True)
            params = {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
            # Pydantic inlines $defs for enums/nested models; preserve them.
            if "$defs" in schema:
                params["$defs"] = schema["$defs"]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }

    def invoke(self, args: dict[str, Any]) -> Any:
        """Validate args against the model, call the function, wrap domain errors."""
        try:
            validated = {} if self.model is None else self.model(**args).model_dump()
        except Exception as e:  # pydantic.ValidationError and friends
            return {"error": f"Invalid arguments: {e}"}
        try:
            return self.func(**validated)
        except SQLAlchemyError as e:
            log.info("Tool %s raised SQL error: %s", self.name, e)
            return {"error": f"{type(e).__name__}: {e}"}
        except Exception as e:  # unknown domain error
            log.exception("Tool %s crashed", self.name)
            return {"error": f"{type(e).__name__}: {e}"}


def tool(
    *, description: str, name: str | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register a function as a tool.

    The function's first positional parameter (if annotated with a Pydantic
    BaseModel subclass) is used as the argument schema. If the function takes
    no args, the schema is empty.

    Usage:
        class MyArgs(BaseModel):
            symbol: str

        @tool(description="Fetch fundamentals for a symbol.")
        def get_fundamentals(args: MyArgs) -> dict:
            ...

    The registered callable is re-wired to accept kwargs (not the model)
    because OpenAI tool_calls arrive as raw kwargs.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        hints = get_type_hints(fn)
        # Find the first arg whose annotation is a BaseModel subclass.
        model_cls: type[BaseModel] | None = None
        for param_name, anno in hints.items():
            if param_name == "return":
                continue
            if isinstance(anno, type) and issubclass(anno, BaseModel):
                model_cls = anno
                break

        def wrapped(**kwargs: Any) -> Any:
            if model_cls is None:
                return fn()
            return fn(model_cls(**kwargs))

        entry = ToolEntry(
            name=name or fn.__name__,
            description=description,
            model=model_cls,
            func=wrapped,
        )
        TOOLS[entry.name] = entry
        return fn

    return decorator


def openai_tool_schemas() -> list[dict[str, Any]]:
    """All registered tools in OpenAI `tools=[...]` format.

    Tool descriptions may contain `{db_schema}` / `{backtest_schema}` placeholders.
    Those are rendered from the active Settings on every call, so tool schemas
    reflect whatever schemas the host app has configured (via
    `agent.config.configure(...)` or env).
    """
    from agent.prompt import render_schemas

    schemas: list[dict[str, Any]] = []
    for entry in TOOLS.values():
        raw = entry.schema
        fn = raw.get("function", {})
        description = fn.get("description", "")
        if "{db_schema}" in description or "{backtest_schema}" in description:
            rendered = {**raw, "function": {**fn, "description": render_schemas(description)}}
            schemas.append(rendered)
        else:
            schemas.append(raw)
    return schemas


def invoke_tool(name: str, args: dict[str, Any]) -> Any:
    """Dispatch by name; returns `{"error": ...}` for unknown tools."""
    entry = TOOLS.get(name)
    if entry is None:
        return {"error": f"Unknown tool: {name}"}
    return entry.invoke(args)

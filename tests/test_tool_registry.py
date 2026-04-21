"""Tests for the tool-registration system (base.py)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent.tools import TOOLS, invoke_tool, openai_tool_schemas


class TestRegistry:
    def test_all_expected_tools_registered(self):
        expected = {
            "list_analytics_columns",
            "describe_table",
            "sample_rows",
            "get_price_history",
            "get_fundamentals",
            "get_market_regime",
            "get_breakouts",
            "screen_symbols",
            "list_backtests",
            "get_backtest_detail",
            "get_recent_backtest_holdings",
            "list_strategies",
            "run_sql",
            "remember",
        }
        assert set(TOOLS.keys()) >= expected

    def test_every_tool_has_schema(self):
        for name, entry in TOOLS.items():
            assert entry.schema["type"] == "function"
            assert entry.schema["function"]["name"] == name
            assert "description" in entry.schema["function"]
            assert "parameters" in entry.schema["function"]

    def test_openai_tool_schemas_matches_registry(self):
        schemas = openai_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert names == set(TOOLS.keys())

    def test_unknown_tool_returns_error(self):
        result = invoke_tool("does_not_exist", {})
        assert "error" in result
        assert "Unknown" in result["error"]


class TestSchemaGeneration:
    def test_required_args_surface(self):
        schema = TOOLS["get_price_history"].schema
        required = schema["function"]["parameters"]["required"]
        assert "symbol" in required
        assert "start" in required
        assert "end" in required

    def test_optional_args_not_required(self):
        schema = TOOLS["get_price_history"].schema
        required = schema["function"]["parameters"]["required"]
        assert "columns" not in required

    def test_alias_exposed_as_public_name(self):
        """describe_table uses `db_schema` internally; LLM-visible schema shows `schema`."""
        schema = TOOLS["describe_table"].schema
        props = schema["function"]["parameters"]["properties"]
        assert "schema" in props
        assert "db_schema" not in props

    def test_enum_literal_in_backtest_detail(self):
        """`include` has Literal["equity_curve", "trades"]; should surface as array with enum."""
        schema = TOOLS["get_backtest_detail"].schema
        include_prop = schema["function"]["parameters"]["properties"]["include"]
        # Pydantic renders Optional[List[Literal[...]]] as anyOf; find the array option.
        # Key point: the literal values appear somewhere in the JSON schema.
        serialized = str(include_prop)
        assert "equity_curve" in serialized
        assert "trades" in serialized


class TestValidation:
    def test_missing_required_arg_returns_error(self):
        result = invoke_tool("get_fundamentals", {})
        assert "error" in result
        assert "Invalid arguments" in result["error"]

    def test_wrong_type_returns_error(self):
        result = invoke_tool("get_backtest_detail", {"backtest_id": "not-an-int"})
        assert "error" in result

    def test_out_of_range_returns_error(self):
        """sample_rows limit is bounded 1..25."""
        result = invoke_tool("sample_rows", {"schema": "x", "table": "y", "limit": 9999})
        assert "error" in result


class _CustomArgs(BaseModel):
    """Module-level so typing.get_type_hints can resolve the forward reference."""

    x: int = Field(..., ge=0)


class TestCustomTool:
    """Exercise the @tool decorator on a fresh local tool."""

    def test_decorator_registers(self):
        from agent.tools.base import TOOLS as REGISTRY
        from agent.tools.base import tool

        @tool(description="Double x.")
        def my_double(args: _CustomArgs) -> dict[str, Any]:
            return {"result": args.x * 2}

        try:
            assert "my_double" in REGISTRY
            assert REGISTRY["my_double"].invoke({"x": 7}) == {"result": 14}
            err = REGISTRY["my_double"].invoke({"x": -1})
            assert "error" in err
        finally:
            REGISTRY.pop("my_double", None)

"""Tests for loop-level utilities (fingerprint, summary formatter, serializer)."""

from __future__ import annotations

import json

import pytest

from agent.loop import _fingerprint, _result_summary, _serialize_tool_result


class TestFingerprint:
    def test_identical_args_same_hash(self):
        assert _fingerprint("foo", '{"a":1}') == _fingerprint("foo", '{"a":1}')

    def test_arg_order_does_not_matter(self):
        assert _fingerprint("foo", '{"a":1,"b":2}') == _fingerprint("foo", '{"b":2,"a":1}')

    def test_different_args_different_hash(self):
        assert _fingerprint("foo", '{"a":1}') != _fingerprint("foo", '{"a":2}')

    def test_different_name_different_hash(self):
        assert _fingerprint("foo", "{}") != _fingerprint("bar", "{}")

    def test_empty_args(self):
        assert _fingerprint("foo", "") == _fingerprint("foo", "{}")

    def test_invalid_json_still_produces_hash(self):
        """Malformed JSON falls back to raw-string hashing; still deterministic."""
        fp1 = _fingerprint("foo", "not json")
        fp2 = _fingerprint("foo", "not json")
        assert fp1 == fp2


class TestResultSummary:
    def test_error_dict(self):
        assert "ERROR" in _result_summary({"error": "table missing"})

    def test_rows_with_columns(self):
        result = {"count": 3, "rows": [{"symbol": "AAPL", "price": 200}]}
        summary = _result_summary(result)
        assert "3 row" in summary
        assert "symbol" in summary

    def test_rows_empty(self):
        result = {"count": 0, "rows": []}
        summary = _result_summary(result)
        assert "0 row" in summary

    def test_columns_result(self):
        result = {"columns": [{"name": "a"}, {"name": "b"}]}
        assert "2 columns" in _result_summary(result)

    def test_dict_list_size_annotation(self):
        result = {"id": 1, "trades": [1, 2, 3], "metrics": {"a": 1, "b": 2}}
        summary = _result_summary(result)
        assert "trades[3]" in summary
        assert "metrics{2}" in summary

    def test_list_result(self):
        assert "2 item" in _result_summary([1, 2])

    def test_str_result(self):
        assert _result_summary("hello") == "hello"


class TestSerializeToolResult:
    def test_small_result_round_trips_intact(self):
        result = {"rows": [{"symbol": "AAPL", "price": 200}], "count": 1}
        out = _serialize_tool_result(result, limit=1000)
        assert json.loads(out) == result

    def test_oversized_result_returns_valid_json_envelope(self):
        """The key bug we're guarding against: truncating mid-JSON string."""
        result = {"rows": [{"x": "y" * 50000}]}
        out = _serialize_tool_result(result, limit=500)
        parsed = json.loads(out)  # Must still parse.
        assert parsed["truncated"] is True
        assert parsed["original_size_chars"] > 500
        assert "preview" in parsed
        assert isinstance(parsed["preview"], str)

    def test_envelope_fits_in_limit(self):
        result = {"big": "x" * 20000}
        out = _serialize_tool_result(result, limit=1000)
        # We allow some envelope overhead but the output should be close to limit.
        assert len(out) < 1500

    def test_exactly_at_limit_not_truncated(self):
        result = {"k": "x" * 10}
        raw = json.dumps(result, default=str)
        out = _serialize_tool_result(result, limit=len(raw))
        assert out == raw

    @pytest.mark.parametrize("limit", [300, 1000, 5000])
    def test_always_valid_json(self, limit: int):
        result = {"payload": "x" * 100_000}
        out = _serialize_tool_result(result, limit=limit)
        json.loads(out)  # Should never raise.

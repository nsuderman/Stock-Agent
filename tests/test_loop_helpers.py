"""Tests for loop-level utilities (fingerprint, summary formatter, trunc)."""

from __future__ import annotations

import pytest

from agent.loop import _fingerprint, _result_summary, _trunc


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


class TestTrunc:
    def test_under_limit(self):
        assert _trunc("short", n=100) == "short"

    def test_over_limit(self):
        out = _trunc("x" * 200, n=50)
        assert len(out) < 200
        assert "truncated" in out

    @pytest.mark.parametrize("n", [1, 10, 100, 1000])
    def test_respects_limit(self, n: int):
        text = "x" * 2000
        out = _trunc(text, n=n)
        # Output is either the whole string or prefix + truncation note.
        assert len(out) >= min(len(text), n)

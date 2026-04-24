"""Unit tests for the compaction module."""

from __future__ import annotations

import json

from agent.compaction import (
    THINK_RE,
    estimate_tokens,
    short_tool_summary,
    stage1_trim,
)


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens([]) >= 0

    def test_scales_with_content(self):
        small = [{"role": "user", "content": "hi"}]
        big = [{"role": "user", "content": "x" * 4000}]
        assert estimate_tokens(big) > estimate_tokens(small)

    def test_approx_char_over_4(self):
        msgs = [{"role": "user", "content": "x" * 400}]
        t = estimate_tokens(msgs)
        # Should be ~100 tokens give or take JSON overhead.
        assert 90 < t < 200


class TestShortToolSummary:
    def test_rows_with_cols(self):
        content = json.dumps({"count": 5, "rows": [{"symbol": "AAPL", "price": 200}]})
        summary = short_tool_summary("run_sql", content)
        assert "5 rows" in summary
        assert "symbol" in summary and "price" in summary

    def test_error_in_payload(self):
        content = json.dumps({"error": "table not found"})
        summary = short_tool_summary("run_sql", content)
        assert "ERROR" in summary
        assert "table not found" in summary

    def test_columns_result(self):
        content = json.dumps({"columns": [{"name": "a"}, {"name": "b"}, {"name": "c"}]})
        summary = short_tool_summary("describe_table", content)
        assert "3 columns" in summary

    def test_unparseable(self):
        summary = short_tool_summary("weird_tool", "not json at all")
        assert "weird_tool" in summary
        assert "chars" in summary


class TestStage1Trim:
    def _build_messages(self, n_pairs: int, tool_payload_size: int = 1000):
        """Create a conversation with N (user, assistant-tool_call, tool) triples."""
        messages = [{"role": "system", "content": "sys"}]
        for i in range(n_pairs):
            messages.append({"role": "user", "content": f"question {i}"})
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"c{i}",
                            "type": "function",
                            "function": {"name": "run_sql", "arguments": "{}"},
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"c{i}",
                    "name": "run_sql",
                    "content": json.dumps({"count": 1, "rows": [{"x": "y" * tool_payload_size}]}),
                }
            )
        return messages

    def test_trims_oldest_tool_results(self):
        msgs = self._build_messages(n_pairs=5, tool_payload_size=2000)
        before = estimate_tokens(msgs)
        trimmed, n = stage1_trim(msgs, budget=500, keep_recent=4)
        after = estimate_tokens(trimmed)
        assert n > 0
        assert after < before

    def test_skips_system_message(self):
        msgs = self._build_messages(n_pairs=3)
        trimmed, _ = stage1_trim(msgs, budget=100, keep_recent=1)
        assert trimmed[0]["role"] == "system"
        assert trimmed[0]["content"] == "sys"

    def test_preserves_keep_recent(self):
        msgs = self._build_messages(n_pairs=5)
        trimmed, _ = stage1_trim(msgs, budget=100, keep_recent=4)
        # Last 4 messages unchanged
        for i in range(1, 5):
            assert trimmed[-i] == msgs[-i]

    def test_no_op_when_under_budget(self):
        msgs = self._build_messages(n_pairs=1, tool_payload_size=10)
        trimmed, n = stage1_trim(msgs, budget=10_000_000, keep_recent=4)
        assert n == 0
        assert trimmed == msgs

    def test_already_trimmed_skipped(self):
        """Don't re-trim messages that already start with `[trimmed]`."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "tool_call_id": "c1", "name": "x", "content": "[trimmed] foo"},
            {"role": "tool", "tool_call_id": "c2", "name": "y", "content": "x" * 2000},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
        trimmed, n = stage1_trim(msgs, budget=50, keep_recent=2)
        # The already-trimmed message still starts with [trimmed], and count reflects only
        # what we newly trimmed.
        assert trimmed[1]["content"] == "[trimmed] foo"
        assert trimmed[2]["content"].startswith("[trimmed]")
        assert n == 1


class TestThinkRegex:
    def test_strips_from_stored_content(self):
        assert THINK_RE.sub("", "<think>foo</think>bar").strip() == "bar"

    def test_strips_multiline(self):
        text = "before\n<think>\nmultiple\nlines\n</think>\nafter"
        assert "think" not in THINK_RE.sub("", text)

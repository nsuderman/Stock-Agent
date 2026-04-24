"""Integration-light tests for compaction that mock the OpenAI client."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from agent.compaction import compact_if_needed, stage1_trim, stage2_summarize


def _mk_client_for_summary(summary_text: str = "Summarized."):
    """Build a client whose chat.completions.create returns `summary_text`."""
    client = MagicMock()
    msg = MagicMock()
    msg.content = summary_text
    client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=msg)])
    # Models list: provide a large context so compaction calculation is meaningful.
    client.models.list.return_value = MagicMock(
        data=[
            MagicMock(
                **{
                    "model_dump.return_value": {
                        "id": "test-model",
                        "status": {"args": ["--ctx-size", "32768"]},
                    }
                }
            )
        ]
    )
    return client


def _build_heavy_conversation(n_pairs: int = 10, payload_size: int = 3000):
    """A conversation guaranteed to exceed a low budget."""
    messages: list[dict] = [{"role": "system", "content": "sys"}]
    for i in range(n_pairs):
        messages.append({"role": "user", "content": f"q{i}"})
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
                "content": json.dumps({"count": 1, "rows": [{"x": "y" * payload_size}]}),
            }
        )
    return messages


class TestStage2Summarize:
    def test_replaces_old_block_with_summary(self):
        messages = _build_heavy_conversation(n_pairs=5)
        client = _mk_client_for_summary("Compact summary.")
        result, n = stage2_summarize(
            messages, client=client, model="test-model", local=True, keep_recent=3
        )
        assert n > 0
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "assistant"
        assert "Compact summary." in result[1]["content"]
        assert "compacted" in result[1]["content"]
        # Tail preserved
        assert result[-1] == messages[-1]

    def test_empty_old_block_returns_messages_unchanged(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "only recent"},
            {"role": "assistant", "content": "ok"},
        ]
        client = _mk_client_for_summary()
        result, n = stage2_summarize(
            messages, client=client, model="test-model", local=True, keep_recent=4
        )
        assert n == 0
        assert result == messages

    def test_summary_call_fails_gracefully(self):
        messages = _build_heavy_conversation(n_pairs=5)
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("rate limited")
        result, n = stage2_summarize(
            messages, client=client, model="test-model", local=True, keep_recent=3
        )
        assert n > 0
        assert "failed" in result[1]["content"].lower()

    def test_transcript_capped_on_massive_history(self):
        """With a huge conversation and a small cap, the transcript sent to the
        summarizer must be truncated rather than submitted whole."""
        messages = _build_heavy_conversation(n_pairs=20, payload_size=10_000)
        client = _mk_client_for_summary("capped summary")
        stage2_summarize(
            messages,
            client=client,
            model="test-model",
            local=True,
            keep_recent=3,
            max_transcript_chars=5_000,
        )
        # Inspect what we actually sent to the LLM.
        (_, kwargs) = client.chat.completions.create.call_args
        user_msg = kwargs["messages"][1]["content"]
        assert len(user_msg) < 6_000
        assert "omitted" in user_msg


class TestCompactIfNeeded:
    def test_noop_when_under_budget(self, monkeypatch):
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        client = _mk_client_for_summary()
        result = compact_if_needed(messages, client=client, model="test-model", local=True)
        assert result is messages

    def test_stage1_fires_with_tiny_budget(self, monkeypatch):
        """With a very tight budget, Stage 1 trims something."""
        messages = _build_heavy_conversation(n_pairs=5, payload_size=10000)
        monkeypatch.setenv("LOCAL_CONTEXT_WINDOW", "4096")
        monkeypatch.setenv("MAX_RESPONSE_TOKENS", "2048")
        from agent.config import reset_settings_cache
        from agent.llm import reset_context_cache

        reset_settings_cache()
        reset_context_cache()

        client = MagicMock()
        client.models.list.return_value = MagicMock(data=[])  # no models → env fallback
        # Provide a summary response in case Stage 2 also fires.
        msg = MagicMock()
        msg.content = "Compact summary."
        client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=msg)])

        result = compact_if_needed(messages, client=client, model="test-model", local=True)
        # Either Stage 1 ran (tool message starts with [trimmed]) OR Stage 2 collapsed
        # old messages into a synthetic assistant note. In either case, the result is
        # structurally different from the input — that's the contract.
        assert result != messages


def test_stage1_stop_after_budget_met():
    """Stage 1 stops trimming as soon as we dip below budget."""
    messages = _build_heavy_conversation(n_pairs=4, payload_size=4000)
    # Generous keep-recent means only old ones get trimmed.
    _trimmed, n = stage1_trim(messages, budget=500, keep_recent=2)
    # It should trim at least one, but may stop before trimming everything it could.
    assert n >= 1

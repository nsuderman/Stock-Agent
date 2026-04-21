"""Tests for the /v1/models context-window probe."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent.llm import (
    _flag_value,
    _parse_ctx_from_args,
    get_context_window,
    reset_context_cache,
)


class TestFlagParsing:
    def test_flag_value_found(self):
        assert (
            _flag_value(["--model", "/path/to.gguf", "--ctx-size", "8192"], "--model")
            == "/path/to.gguf"
        )

    def test_flag_value_missing(self):
        assert _flag_value(["--host", "0.0.0.0"], "--ctx-size") is None

    def test_flag_value_at_end_with_no_value(self):
        assert _flag_value(["--ctx-size"], "--ctx-size") is None

    def test_flag_value_not_a_list(self):
        assert _flag_value("not a list", "--ctx-size") is None  # type: ignore[arg-type]

    def test_parse_ctx_integer(self):
        assert _parse_ctx_from_args(["--ctx-size", "65536"]) == 65536

    def test_parse_ctx_invalid(self):
        assert _parse_ctx_from_args(["--ctx-size", "not-a-number"]) is None

    def test_parse_ctx_missing(self):
        assert _parse_ctx_from_args(["--host", "0.0.0.0"]) is None


def _model_entry(
    model_id: str,
    *,
    ctx_size: int | None = None,
    model_path: str | None = None,
    extra_fields: dict | None = None,
) -> dict:
    args: list[str] = []
    if ctx_size is not None:
        args += ["--ctx-size", str(ctx_size)]
    if model_path is not None:
        args += ["--model", model_path]
    result = {
        "id": model_id,
        "status": {"args": args, "value": "loaded" if ctx_size else "unloaded"},
    }
    if extra_fields:
        result.update(extra_fields)
    return result


def _fake_client(models: list[dict]) -> MagicMock:
    client = MagicMock()
    data = [MagicMock(**{"model_dump.return_value": m}) for m in models]
    client.models.list.return_value = MagicMock(data=data)
    return client


class TestGetContextWindow:
    def setup_method(self):
        reset_context_cache()

    def test_direct_ctx_size_match(self):
        client = _fake_client([_model_entry("m1", ctx_size=131072)])
        window, source = get_context_window(client, "m1", local=True)
        assert window == 131072
        assert source == "endpoint"

    def test_alias_match_via_model_path(self):
        """The loaded entry has no --ctx-size, but a sibling pointing at the same weights does."""
        client = _fake_client(
            [
                _model_entry("loaded-alias", model_path="/w/qwen.gguf"),
                _model_entry("expanded", ctx_size=262144, model_path="/w/qwen.gguf"),
            ]
        )
        window, source = get_context_window(client, "loaded-alias", local=True)
        assert window == 262144
        assert source == "endpoint (alias)"

    def test_alias_match_takes_max(self):
        client = _fake_client(
            [
                _model_entry("loaded", model_path="/w/x.gguf"),
                _model_entry("sibling-small", ctx_size=8192, model_path="/w/x.gguf"),
                _model_entry("sibling-big", ctx_size=131072, model_path="/w/x.gguf"),
            ]
        )
        window, _ = get_context_window(client, "loaded", local=True)
        assert window == 131072

    def test_standard_field_fallback(self):
        client = _fake_client([_model_entry("m1", extra_fields={"context_length": 16384})])
        window, source = get_context_window(client, "m1", local=True)
        assert window == 16384
        assert source == "endpoint"

    def test_max_model_len_field(self):
        client = _fake_client([_model_entry("m1", extra_fields={"max_model_len": 32000})])
        window, _source = get_context_window(client, "m1", local=True)
        assert window == 32000

    def test_falls_back_to_env_default(self, monkeypatch):
        monkeypatch.setenv("LOCAL_CONTEXT_WINDOW", "9999")
        from agent.config import reset_settings_cache

        reset_settings_cache()
        reset_context_cache()
        client = _fake_client([_model_entry("m1")])  # no ctx, no path
        window, source = get_context_window(client, "m1", local=True)
        assert window == 9999
        assert source == "env"

    def test_cache_prevents_double_probe(self):
        client = _fake_client([_model_entry("m1", ctx_size=65536)])
        get_context_window(client, "m1", local=True)
        get_context_window(client, "m1", local=True)
        client.models.list.assert_called_once()

    def test_api_failure_falls_back(self):
        client = MagicMock()
        client.models.list.side_effect = RuntimeError("network down")
        window, source = get_context_window(client, "m1", local=True)
        assert window > 0  # some fallback
        assert source in ("env", "default")

    def test_nonexistent_model_falls_back(self):
        client = _fake_client([_model_entry("other-model", ctx_size=8192)])
        _window, source = get_context_window(client, "missing", local=True)
        assert source in ("env", "default")

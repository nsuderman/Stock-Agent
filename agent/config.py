"""Typed, validated settings loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application configuration. Override via environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    db_user: str = Field(default="", description="PostgreSQL username.")
    db_password: str = Field(default="", description="PostgreSQL password.")
    db_host: str = Field(default="localhost", description="PostgreSQL host.")
    db_name: str = Field(default="stock", description="PostgreSQL database name.")
    db_schema: str = Field(default="stock", description="Default search_path schema.")
    backtest_schema: str = Field(
        default="stock",
        description="Schema holding backtest_results and strategies.",
    )

    local_llm_url: str = Field(default="http://localhost:8080/v1")
    local_model: str = Field(default="qwen3.6-35b-a3b")

    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None

    local_context_window: int = Field(default=32768, ge=1024)
    remote_context_window: int = Field(default=128000, ge=1024)

    compact_at: float = Field(default=0.75, ge=0.1, le=0.95)
    compact_keep_recent: int = Field(default=4, ge=1)
    max_response_tokens: int = Field(default=4096, ge=128)

    log_level: str = Field(default="INFO")

    sec_user_agent: str = Field(
        default="Stock Agent (example@example.com)",
        description="User-Agent header sent to SEC EDGAR. Their fair-use policy requires this.",
    )

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}"

    @property
    def memory_path(self) -> Path:
        return REPO_ROOT / "memory.md"

    @property
    def sessions_dir(self) -> Path:
        return REPO_ROOT / "sessions"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide Settings instance (cached)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_cache() -> None:
    """Testing hook — clear the cached Settings."""
    global _settings
    _settings = None

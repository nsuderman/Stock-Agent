"""Read-only PostgreSQL engine + session factory."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker

from agent.config import get_settings

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def _connect_options() -> str:
    s = get_settings()
    # PostgreSQL rejects any write at the session level regardless of query text.
    return f"-csearch_path={s.db_schema},{s.backtest_schema} -cdefault_transaction_read_only=on"


def get_engine() -> Engine:
    """Return the shared read-only engine (created on first call)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            get_settings().database_url,
            connect_args={"options": _connect_options()},
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def get_db():
    """FastAPI/agent-style dependency: yields a session, always closes."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def reset_engine_cache() -> None:
    """Testing hook — drop cached engine + session factory."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None

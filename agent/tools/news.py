"""Stock news tools backed by Yahoo Finance (yfinance)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent.logging_setup import get_logger
from agent.tools.base import tool

log = get_logger(__name__)


class StockNewsArgs(BaseModel):
    symbol: str = Field(..., description="Ticker, e.g. 'AAPL'.")
    limit: int = Field(default=10, ge=1, le=50, description="Max articles to return.")


def _extract(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize yfinance news item shape.

    yfinance >= 0.2.40 wraps each article in {"id": ..., "content": {...}};
    older versions returned a flat dict. This handles both.
    """
    c: dict[str, Any] = item.get("content") or item
    provider = c.get("provider")
    publisher = provider.get("displayName") if isinstance(provider, dict) else c.get("publisher")
    click = c.get("clickThroughUrl")
    link = click.get("url") if isinstance(click, dict) else c.get("link")
    return {
        "title": c.get("title"),
        "publisher": publisher,
        "published_at": c.get("pubDate") or c.get("providerPublishTime"),
        "link": link,
        "summary": c.get("summary") or c.get("description"),
    }


@tool(
    description=(
        "Fetch recent news headlines for a stock ticker from Yahoo Finance. Returns "
        "title, publisher, publish date, link, and a short summary per article. Use "
        "this for questions about catalysts, earnings, management changes, analyst "
        "actions, or recent events on a specific symbol."
    )
)
def get_stock_news(args: StockNewsArgs) -> dict[str, Any]:
    try:
        import yfinance as yf
    except ImportError:
        return {
            "error": "yfinance not installed. Run `pip install yfinance` or `pip install -e '.[news]'`."
        }

    try:
        raw = yf.Ticker(args.symbol.upper()).news or []
    except Exception as e:
        log.warning("yfinance news lookup failed for %s: %s", args.symbol, e)
        return {"error": f"{type(e).__name__}: {e}"}

    articles = [_extract(item) for item in raw[: args.limit]]
    return {
        "symbol": args.symbol.upper(),
        "count": len(articles),
        "articles": articles,
    }

"""SEC EDGAR tools — recent filings and insider (Form 4) transactions."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from agent.config import get_settings
from agent.logging_setup import get_logger
from agent.tools.base import coerce, tool

log = get_logger(__name__)

# SEC's fair-use policy requires identifying User-Agent on every request.
# We set it lazily so test environments can override Settings first.
_edgar_configured = False


def _configure_edgar() -> None:
    global _edgar_configured
    if _edgar_configured:
        return
    try:
        from edgar import set_identity
    except ImportError as e:
        raise RuntimeError(
            "edgartools not installed. Run `pip install edgartools`."
        ) from e
    set_identity(get_settings().sec_user_agent)
    _edgar_configured = True


class RecentFilingsArgs(BaseModel):
    symbol: str = Field(..., description="Ticker, e.g. 'AAPL'.")
    form_type: str | None = Field(
        default=None,
        description=(
            "Filter by form type: '10-K' (annual), '10-Q' (quarterly), '8-K' "
            "(current report), 'DEF 14A' (proxy), '4' (insider), '13F-HR' "
            "(institutional holdings). Omit to get recent filings of all types."
        ),
    )
    limit: int = Field(default=10, ge=1, le=50)


class InsiderTransactionsArgs(BaseModel):
    symbol: str = Field(..., description="Ticker, e.g. 'AAPL'.")
    limit: int = Field(
        default=10,
        ge=1,
        le=25,
        description="How many recent Form 4 filings to examine. Each may contain multiple trades.",
    )


@tool(
    description=(
        "List recent SEC EDGAR filings for a ticker. Filter by `form_type` "
        "(10-K, 10-Q, 8-K, DEF 14A, 4, 13F-HR) or omit for all forms. Returns form, "
        "filing date, accession number, and URLs to the filing index and primary document."
    )
)
def get_recent_filings(args: RecentFilingsArgs) -> dict[str, Any]:
    try:
        _configure_edgar()
        from edgar import Company
    except RuntimeError as e:
        return {"error": str(e)}

    try:
        company = Company(args.symbol.upper())
    except Exception as e:
        log.info("SEC company lookup failed for %s: %s", args.symbol, e)
        return {"error": f"Could not find company for ticker {args.symbol}: {e}"}

    try:
        filings = (
            company.get_filings(form=args.form_type)
            if args.form_type
            else company.get_filings()
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    rows: list[dict[str, Any]] = []
    for f in filings[: args.limit]:
        rows.append(
            {
                "form": f.form,
                "filing_date": coerce(f.filing_date),
                "accession_no": f.accession_no,
                "filing_url": getattr(f, "filing_url", None),
                "index_url": getattr(f, "homepage_url", None),
            }
        )
    return {
        "symbol": args.symbol.upper(),
        "company": company.name,
        "cik": str(company.cik),
        "form_type": args.form_type,
        "count": len(rows),
        "filings": rows,
    }


@tool(
    description=(
        "Fetch recent insider (Form 4) transactions for a ticker — executives and "
        "directors buying or selling shares. Returns the insider's name, transaction "
        "date, share count, price, and whether shares were acquired ('A') or disposed "
        "('D'). Use this for 'is anyone at X selling' or 'has management been buying' "
        "questions. Note: routine option exercises and tax-withholding trades are "
        "included; large open-market 'Purchase' / 'Sale' entries in the "
        "transaction_type field are the most signal-bearing."
    )
)
def get_insider_transactions(args: InsiderTransactionsArgs) -> dict[str, Any]:
    try:
        _configure_edgar()
        from edgar import Company
    except RuntimeError as e:
        return {"error": str(e)}

    try:
        company = Company(args.symbol.upper())
    except Exception as e:
        return {"error": f"Could not find company for ticker {args.symbol}: {e}"}

    try:
        filings = company.get_filings(form="4")
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    trades: list[dict[str, Any]] = []
    examined = 0
    for f in filings[: args.limit]:
        examined += 1
        try:
            obj = f.obj()
        except Exception as e:  # per-filing parse error
            log.info("Form 4 parse failed for %s: %s", f.accession_no, e)
            continue
        insider = getattr(obj, "insider_name", None)
        nd = getattr(obj, "non_derivative_table", None)
        if nd is None or not getattr(nd, "has_transactions", False):
            continue
        try:
            df: pd.DataFrame = nd.transactions.data
        except Exception:
            continue
        for _, row in df.iterrows():
            price = row.get("Price")
            trades.append(
                {
                    "filing_date": coerce(f.filing_date),
                    "accession_no": f.accession_no,
                    "insider": insider,
                    "security": row.get("Security"),
                    "transaction_date": coerce(row.get("Date")),
                    "shares": coerce(row.get("Shares")),
                    "remaining_shares": coerce(row.get("Remaining")),
                    "price": coerce(price) if pd.notna(price) else None,
                    "acquired_disposed": row.get("AcquiredDisposed"),
                    "transaction_type": row.get("TransactionType"),
                    "transaction_code": row.get("Code"),
                }
            )

    return {
        "symbol": args.symbol.upper(),
        "company": company.name,
        "filings_examined": examined,
        "trade_count": len(trades),
        "trades": trades,
    }

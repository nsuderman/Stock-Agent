"""SEC EDGAR tools — filings, insider (Form 4) transactions, 13F holdings."""

from __future__ import annotations

import datetime as _dt
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
        raise RuntimeError("edgartools not installed. Run `pip install edgartools`.") from e
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
            company.get_filings(form=args.form_type) if args.form_type else company.get_filings()
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


# ---------------------------------------------------------------------------
# 13F-HR (institutional holdings) tools
# ---------------------------------------------------------------------------


def _current_year_quarter() -> tuple[int, int]:
    today = _dt.date.today()
    return today.year, (today.month - 1) // 3 + 1


def _quarter_of(date: _dt.date | None) -> tuple[int, int] | None:
    if date is None:
        return None
    return date.year, (date.month - 1) // 3 + 1


def _resolve_manager(identifier: str):
    """Try to resolve a 13F filer by ticker, CIK, or company name.

    Returns the edgar Company object (or raises).
    """
    from edgar import Company
    from edgar.entity.search import find_company

    raw = identifier.strip()
    if not raw:
        raise ValueError("manager identifier is empty")

    # Numeric → treat as CIK.
    if raw.isdigit():
        return Company(int(raw))

    # Try as ticker (uppercased) — this is the fast path for public filers
    # like BRK-A. If it's not a known ticker, edgar raises.
    try:
        return Company(raw.upper())
    except Exception:
        pass

    # Fall back to name search. Ticker-indexed but best we have without a CIK.
    results = find_company(raw, top_n=1)
    if results.empty:
        raise ValueError(f"No EDGAR entity matched {identifier!r}")
    return results[0]


class ListThirteenFFilingsArgs(BaseModel):
    year: int | None = Field(
        default=None,
        ge=2000,
        le=2100,
        description="Calendar year the filing was submitted. Omit (with quarter) for the latest quarter.",
    )
    quarter: int | None = Field(
        default=None,
        ge=1,
        le=4,
        description="Calendar quarter (1-4) the filing was submitted. Omit (with year) for the latest quarter.",
    )
    limit: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Max filings to return (there can be thousands per quarter).",
    )


class ThirteenFHoldingsArgs(BaseModel):
    manager: str = Field(
        ...,
        description=(
            "Filer identifier: ticker (e.g. 'BRK-A' for Berkshire), CIK as a string "
            "(e.g. '1067983'), or company name fragment (e.g. 'Scion Asset'). "
            "Tickers are fastest; name search is ticker-indexed so pure hedge funds "
            "without tickers may need a CIK."
        ),
    )
    year: int | None = Field(
        default=None,
        ge=2000,
        le=2100,
        description="Reporting-period year (when the holdings were as-of). Omit for the latest filing.",
    )
    quarter: int | None = Field(
        default=None,
        ge=1,
        le=4,
        description="Reporting-period quarter (when the holdings were as-of). Omit for the latest filing.",
    )
    top_n: int = Field(
        default=25,
        ge=1,
        le=50,
        description="How many top positions to return, after sorting.",
    )
    sort: str = Field(
        default="value",
        description="Sort key: 'value' (market value, default) or 'shares' (share count).",
    )


class ThirteenFChangesArgs(BaseModel):
    manager: str = Field(
        ...,
        description="Filer identifier: ticker, CIK, or company name (see get_13f_holdings).",
    )
    top_n: int = Field(default=25, ge=1, le=50)
    sort: str = Field(
        default="increased",
        description=(
            "Which slice of changes to return, sorted descending by magnitude: "
            "'increased' (biggest adds), 'decreased' (biggest trims), "
            "'absolute_change' (largest $ moves either way), "
            "'new' (brand-new positions), 'closed' (exited positions)."
        ),
    )


@tool(
    description=(
        "List SEC Form 13F-HR filings filed in a given calendar quarter. Omit both "
        "`year` and `quarter` to get the most recently-filed quarter. Returns manager "
        "name, CIK, accession, filing date, and reporting period — metadata only, not "
        "the positions themselves. Use this to discover who filed, then call "
        "`get_13f_holdings` or `get_13f_changes` with a specific manager."
    )
)
def get_13f_filings(args: ListThirteenFFilingsArgs) -> dict[str, Any]:
    try:
        _configure_edgar()
        from edgar import get_filings
    except RuntimeError as e:
        return {"error": str(e)}

    year = args.year
    quarter = args.quarter
    if year is None or quarter is None:
        year, quarter = _current_year_quarter()

    try:
        filings = get_filings(year=year, quarter=quarter, form="13F-HR")
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    if filings is None or len(filings) == 0:
        return {
            "year": year,
            "quarter": quarter,
            "count": 0,
            "filings": [],
            "note": "No 13F-HR filings indexed for this quarter yet. The index typically lags by ~1 business day.",
        }

    rows: list[dict[str, Any]] = []
    for f in filings[: args.limit]:
        rows.append(
            {
                "manager": getattr(f, "company", None),
                "cik": coerce(getattr(f, "cik", None)),
                "accession_no": f.accession_no,
                "filing_date": coerce(f.filing_date),
                "report_period": coerce(getattr(f, "report_date", None)),
            }
        )
    return {
        "year": year,
        "quarter": quarter,
        "total_available": len(filings),
        "count": len(rows),
        "filings": rows,
    }


def _select_thirteenf(company, year: int | None, quarter: int | None):
    """Return (ThirteenF object, matched filing) for the requested period, or
    the most recent filing if year/quarter are omitted. Returns (None, None)
    if nothing matches.
    """
    filings = company.get_filings(form="13F-HR")
    if filings is None or len(filings) == 0:
        return None, None

    target = (year, quarter) if year is not None and quarter is not None else None
    max_scan = 8 if target else 1

    for f in filings[:max_scan]:
        if target is not None:
            report_date = getattr(f, "report_date", None)
            if isinstance(report_date, str):
                try:
                    report_date = _dt.date.fromisoformat(report_date)
                except ValueError:
                    report_date = None
            if _quarter_of(report_date) != target:
                continue
        try:
            tf = f.obj()
        except Exception as e:
            log.info("ThirteenF parse failed for %s: %s", f.accession_no, e)
            continue
        return tf, f
    return None, None


@tool(
    description=(
        "Fetch one institutional manager's 13F-HR holdings. `manager` accepts a "
        "ticker (e.g. 'BRK-A'), CIK string, or company name (e.g. 'Scion Asset'). "
        "Omit `year` and `quarter` for the latest filing; otherwise they select by "
        "reporting period (end-of-quarter as-of date, not filing date). Returns the "
        "top N positions by value (or shares) plus total portfolio value and position "
        "count. Position fields: issuer, ticker, cusip, shares, value, put_call, type."
    )
)
def get_13f_holdings(args: ThirteenFHoldingsArgs) -> dict[str, Any]:
    try:
        _configure_edgar()
    except RuntimeError as e:
        return {"error": str(e)}

    try:
        company = _resolve_manager(args.manager)
    except Exception as e:
        return {"error": f"Could not resolve manager {args.manager!r}: {e}"}

    sort_col = {"value": "Value", "shares": "SharesPrnAmount"}.get(args.sort.lower())
    if sort_col is None:
        return {"error": f"Invalid sort {args.sort!r}: use 'value' or 'shares'."}

    try:
        tf, _filing = _select_thirteenf(company, args.year, args.quarter)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    if tf is None:
        period = (
            f"{args.year} Q{args.quarter}" if args.year and args.quarter else "any recent quarter"
        )
        return {"error": f"No parseable 13F-HR filing found for {company.name} ({period})."}

    holdings_df = tf.holdings
    if holdings_df is None or len(holdings_df) == 0:
        return {
            "error": f"13F for {company.name} has no holdings table (possibly confidential treatment)."
        }

    if sort_col in holdings_df.columns:
        holdings_df = holdings_df.sort_values(sort_col, ascending=False)

    rows: list[dict[str, Any]] = []
    for _, row in holdings_df.head(args.top_n).iterrows():
        rows.append(
            {
                "issuer": row.get("Issuer"),
                "ticker": row.get("Ticker"),
                "cusip": row.get("Cusip"),
                "shares": coerce(row.get("SharesPrnAmount")),
                "value": coerce(row.get("Value")),
                "type": str(row.get("Type")) if row.get("Type") is not None else None,
                "put_call": str(row.get("PutCall")) if row.get("PutCall") else None,
            }
        )

    return {
        "manager": tf.management_company_name or company.name,
        "cik": coerce(getattr(company, "cik", None)),
        "report_period": coerce(tf.report_period),
        "filing_date": coerce(tf.filing_date),
        "accession_no": tf.accession_number,
        "total_value": coerce(tf.total_value),
        "total_holdings": tf.total_holdings,
        "sort": args.sort,
        "top_n": len(rows),
        "holdings": rows,
    }


@tool(
    description=(
        "Compare one manager's latest 13F-HR holdings against the previous quarter. "
        "Use this to find what a manager added (`sort='increased'` or 'new'), trimmed "
        "('decreased' or 'closed'), or moved most ('absolute_change'). Returns per-"
        "position deltas with share/value change and percent change. Covers only one "
        "manager per call — for cross-market aggregates, run a batch pipeline against "
        "many managers."
    )
)
def get_13f_changes(args: ThirteenFChangesArgs) -> dict[str, Any]:
    try:
        _configure_edgar()
    except RuntimeError as e:
        return {"error": str(e)}

    valid_sorts = {"increased", "decreased", "absolute_change", "new", "closed"}
    if args.sort.lower() not in valid_sorts:
        return {"error": f"Invalid sort {args.sort!r}: use one of {sorted(valid_sorts)}."}
    sort = args.sort.lower()

    try:
        company = _resolve_manager(args.manager)
    except Exception as e:
        return {"error": f"Could not resolve manager {args.manager!r}: {e}"}

    try:
        tf, _ = _select_thirteenf(company, None, None)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    if tf is None:
        return {"error": f"No parseable 13F-HR filing found for {company.name}."}

    comparison = tf.compare_holdings()
    if comparison is None:
        return {"error": f"Previous-quarter 13F not available for {company.name}."}

    df: pd.DataFrame = comparison.data.copy()

    if sort == "new":
        df = df[df["Status"] == "NEW"].sort_values("Value", ascending=False)
    elif sort == "closed":
        df = df[df["Status"] == "CLOSED"].sort_values("PrevValue", ascending=False)
    elif sort == "increased":
        df = df[df["Status"].isin(["NEW", "INCREASED"])].sort_values("ValueChange", ascending=False)
    elif sort == "decreased":
        df = df[df["Status"].isin(["CLOSED", "DECREASED"])].sort_values(
            "ValueChange", ascending=True
        )
    else:  # absolute_change
        df = df.reindex(df["ValueChange"].abs().sort_values(ascending=False).index)

    rows: list[dict[str, Any]] = []
    for _, row in df.head(args.top_n).iterrows():
        rows.append(
            {
                "issuer": row.get("Issuer"),
                "ticker": row.get("Ticker"),
                "cusip": row.get("Cusip"),
                "status": row.get("Status"),
                "shares": coerce(row.get("Shares")) if pd.notna(row.get("Shares")) else None,
                "prev_shares": coerce(row.get("PrevShares"))
                if pd.notna(row.get("PrevShares"))
                else None,
                "share_change": coerce(row.get("ShareChange"))
                if pd.notna(row.get("ShareChange"))
                else None,
                "share_change_pct": coerce(row.get("ShareChangePct"))
                if pd.notna(row.get("ShareChangePct"))
                else None,
                "value": coerce(row.get("Value")) if pd.notna(row.get("Value")) else None,
                "prev_value": coerce(row.get("PrevValue"))
                if pd.notna(row.get("PrevValue"))
                else None,
                "value_change": coerce(row.get("ValueChange"))
                if pd.notna(row.get("ValueChange"))
                else None,
                "value_change_pct": coerce(row.get("ValueChangePct"))
                if pd.notna(row.get("ValueChangePct"))
                else None,
            }
        )

    return {
        "manager": comparison.manager_name or company.name,
        "cik": coerce(getattr(company, "cik", None)),
        "current_period": comparison.current_period,
        "previous_period": comparison.previous_period,
        "sort": sort,
        "top_n": len(rows),
        "changes": rows,
    }

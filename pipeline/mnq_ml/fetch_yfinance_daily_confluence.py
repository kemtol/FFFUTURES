#!/usr/bin/env python3
"""Fetch daily yfinance confluence data into one L0 DuckDB.

This is for daily regime/context features only. Feature builders must shift by
one trading day for MNQ trade date D, so no current-day daily close leaks into
an intraday ORB decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data/Level_0_Raw/yfinance_daily.duckdb"
DEFAULT_MANIFEST = ROOT / "data/Level_0_Raw/yfinance_daily_manifest.json"
DEFAULT_START = "2018-01-01"
DEFAULT_SYMBOLS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "VIX": "^VIX",
    "TNX": "^TNX",
    "DXY": "DX-Y.NYB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None, help="Exclusive YYYY-MM-DD end date. Defaults to tomorrow UTC.")
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS.keys()),
        help="Comma-separated aliases from the default map, or alias=ticker pairs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_symbols(raw: str) -> dict[str, str]:
    symbols: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            alias, ticker = item.split("=", 1)
            alias = alias.strip().upper()
            ticker = ticker.strip()
        else:
            alias = item.upper()
            if alias not in DEFAULT_SYMBOLS:
                raise SystemExit(f"Unknown symbol alias {alias!r}; use alias=ticker for custom symbols.")
            ticker = DEFAULT_SYMBOLS[alias]
        symbols[alias] = ticker
    if not symbols:
        raise SystemExit("No symbols requested.")
    return symbols


def default_end_date() -> str:
    # yfinance end is exclusive. Tomorrow UTC includes the latest completed
    # daily bar when Yahoo has published it.
    return (date.today() + timedelta(days=1)).isoformat()


def field_series(frame: pd.DataFrame, ticker: str, field: str) -> pd.Series:
    if isinstance(frame.columns, pd.MultiIndex):
        if (ticker, field) in frame.columns:
            return frame[(ticker, field)]
        if (field, ticker) in frame.columns:
            return frame[(field, ticker)]
    if field in frame.columns:
        return frame[field]
    return pd.Series(index=frame.index, dtype="float64")


def flatten_download(raw: pd.DataFrame, symbols: dict[str, str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for alias, ticker in symbols.items():
        cur = pd.DataFrame(
            {
                "date": pd.to_datetime(raw.index).date,
                "symbol": alias,
                "yf_ticker": ticker,
                "open": field_series(raw, ticker, "Open"),
                "high": field_series(raw, ticker, "High"),
                "low": field_series(raw, ticker, "Low"),
                "close": field_series(raw, ticker, "Close"),
                "adj_close": field_series(raw, ticker, "Adj Close"),
                "volume": field_series(raw, ticker, "Volume"),
            }
        )
        rows.append(cur)

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    for col in ["open", "high", "low", "close", "adj_close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype("int64")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out["adj_close"] = out["adj_close"].fillna(out["close"])
    out["source"] = "YFINANCE_DAILY"
    out["fetched_at_utc"] = datetime.now(timezone.utc).replace(tzinfo=None)
    out = out.drop_duplicates(["symbol", "date"], keep="last")
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    bad_ohlc = out[
        (out["high"] < out[["open", "close", "low"]].max(axis=1))
        | (out["low"] > out[["open", "close", "high"]].min(axis=1))
        | (out["volume"] < 0)
    ]
    if not bad_ohlc.empty:
        sample = bad_ohlc[["symbol", "date", "open", "high", "low", "close"]].head(10).to_dict("records")
        raise SystemExit(f"Invalid OHLCV rows from yfinance: {len(bad_ohlc)}. Sample: {sample}")
    return out


def fetch(symbols: dict[str, str], start: str, end: str) -> pd.DataFrame:
    tickers = list(symbols.values())
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    return flatten_download(raw, symbols)


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        create table if not exists daily_ohlcv (
          date date,
          symbol varchar,
          yf_ticker varchar,
          open double,
          high double,
          low double,
          close double,
          adj_close double,
          volume bigint,
          source varchar,
          fetched_at_utc timestamp
        )
        """
    )
    con.execute(
        """
        create table if not exists fetch_segments (
          fetched_at_utc timestamp,
          source varchar,
          symbols varchar,
          start_date date,
          end_date_exclusive date,
          rows_fetched bigint,
          dry_run boolean
        )
        """
    )


def table_summary(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    total = con.execute("select count(*) from daily_ohlcv").fetchone()[0]
    duplicates = con.execute(
        """
        select count(*)::bigint
        from (
          select symbol, date, count(*) as n
          from daily_ohlcv
          group by symbol, date
          having count(*) > 1
        )
        """
    ).fetchone()[0]
    by_symbol = con.execute(
        """
        select
          symbol,
          yf_ticker,
          count(*)::bigint as rows,
          min(date)::varchar as min_date,
          max(date)::varchar as max_date,
          sum(case when close is null then 1 else 0 end)::bigint as close_nulls,
          sum(case when volume is null then 1 else 0 end)::bigint as volume_nulls
        from daily_ohlcv
        group by symbol, yf_ticker
        order by symbol
        """
    ).fetchdf()
    return {
        "rows": int(total),
        "duplicate_symbol_dates": int(duplicates),
        "by_symbol": by_symbol.to_dict("records"),
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def main() -> int:
    args = parse_args()
    db_path = Path(args.db)
    manifest_path = Path(args.manifest)
    symbols = parse_symbols(args.symbols)
    end = args.end or default_end_date()

    print(f"fetching daily yfinance confluence: {symbols}")
    print(f"range: {args.start} -> {end} (end exclusive)")
    df = fetch(symbols, args.start, end)
    if df.empty:
        raise SystemExit("yfinance returned no daily confluence rows")

    fetched_summary = (
        df.groupby(["symbol", "yf_ticker"], as_index=False)
        .agg(rows=("date", "size"), min_date=("date", "min"), max_date=("date", "max"))
        .sort_values("symbol")
    )

    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    if args.dry_run:
        inserted = 0
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(db_path))
        ensure_schema(con)
        before = table_summary(con)
        con.register("daily_df", df)
        con.execute(
            """
            delete from daily_ohlcv
            where symbol in (select distinct symbol from daily_df)
              and date >= (select min(date) from daily_df)
              and date <= (select max(date) from daily_df)
            """
        )
        con.execute(
            """
            insert into daily_ohlcv
            select date, symbol, yf_ticker, open, high, low, close, adj_close,
                   volume, source, fetched_at_utc
            from daily_df
            order by symbol, date
            """
        )
        inserted = int(len(df))
        con.execute(
            """
            insert into fetch_segments
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                datetime.now(timezone.utc).replace(tzinfo=None),
                "YFINANCE_DAILY",
                ",".join(symbols.keys()),
                pd.Timestamp(args.start).date(),
                pd.Timestamp(end).date(),
                int(len(df)),
                bool(args.dry_run),
            ],
        )
        after = table_summary(con)
        con.close()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "db": str(db_path),
        "table": "daily_ohlcv",
        "source": "YFINANCE_DAILY",
        "symbols": symbols,
        "start": args.start,
        "end_exclusive": end,
        "dry_run": bool(args.dry_run),
        "rows_fetched": int(len(df)),
        "rows_inserted": int(inserted),
        "fetched_by_symbol": fetched_summary.to_dict("records"),
        "before": before,
        "after": after,
        "no_lookahead_contract": "For MNQ trade date D, feature builders must use external daily rows with date <= D-1 only.",
    }
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    if after and after["duplicate_symbol_dates"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

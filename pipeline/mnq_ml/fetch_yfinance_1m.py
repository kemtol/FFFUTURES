#!/usr/bin/env python3
"""Append recent MNQ 1m bars from Yahoo Finance into MNQ DuckDB.

Yahoo only exposes roughly 8 days of 1m futures data. This updater is meant to
bridge the short gap after the Databento backfill, not to replace Databento as
the historical source.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data/Level_0_Raw/MNQ_1m.duckdb"
DEFAULT_MANIFEST = ROOT / "data/Level_0_Raw/MNQ_1m_yfinance_append_manifest.json"
TICKER = "MNQ=F"
SOURCE_SYMBOL = "MNQ=F_YF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--period", default="8d")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--replace-yfinance", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def flatten_yfinance(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.reset_index()
    ts_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(None)
    out = pd.DataFrame(
        {
            "timestamp_utc": df[ts_col],
            "open": df["Open"].astype(float),
            "high": df["High"].astype(float),
            "low": df["Low"].astype(float),
            "close": df["Close"].astype(float),
            "volume": df["Volume"].fillna(0).astype("int64"),
            "source_symbol": SOURCE_SYMBOL,
            "instrument_id": pd.Series([pd.NA] * len(df), dtype="Int64"),
        }
    )
    out = out.dropna(subset=["timestamp_utc", "open", "high", "low", "close"])
    out = out.drop_duplicates(subset=["timestamp_utc"], keep="last")
    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    bad_ohlc = out[
        (out["high"] < out[["open", "close", "low"]].max(axis=1))
        | (out["low"] > out[["open", "close", "high"]].min(axis=1))
        | (out["volume"] < 0)
    ]
    if not bad_ohlc.empty:
        raise SystemExit(f"Yahoo returned invalid OHLCV rows: {len(bad_ohlc)}")
    return out


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    db_path = Path(args.db)
    manifest_path = Path(args.manifest)
    if not db_path.exists():
        raise SystemExit(f"Missing MNQ DuckDB: {db_path}")

    con = duckdb.connect(str(db_path))
    before_max = con.execute("select max(timestamp_utc) from ohlcv_1m").fetchone()[0]
    before_rows = con.execute("select count(*) from ohlcv_1m").fetchone()[0]
    before_yf_rows = con.execute(
        "select count(*) from ohlcv_1m where source_symbol = ?",
        [SOURCE_SYMBOL],
    ).fetchone()[0]

    print(f"fetching {TICKER} period={args.period} interval={args.interval}")
    raw = yf.download(
        TICKER,
        period=args.period,
        interval=args.interval,
        progress=False,
        auto_adjust=False,
        prepost=True,
        threads=False,
    )
    df = flatten_yfinance(raw)
    if df.empty:
        raise SystemExit("Yahoo returned no MNQ data")

    if args.replace_yfinance:
        con.execute("delete from ohlcv_1m where source_symbol = ?", [SOURCE_SYMBOL])
        cutoff = con.execute("select max(timestamp_utc) from ohlcv_1m").fetchone()[0]
    else:
        cutoff = before_max

    append = df[df["timestamp_utc"] > pd.Timestamp(cutoff)].copy()
    overlap = df[df["timestamp_utc"] <= pd.Timestamp(cutoff)].copy()

    if args.dry_run:
        inserted = 0
    elif append.empty:
        inserted = 0
    else:
        con.register("append_df", append)
        con.execute(
            """
            insert into ohlcv_1m
            select timestamp_utc, open, high, low, close, volume, source_symbol, instrument_id
            from append_df
            order by timestamp_utc
            """
        )
        inserted = int(len(append))

    after_max = con.execute("select max(timestamp_utc) from ohlcv_1m").fetchone()[0]
    after_rows = con.execute("select count(*) from ohlcv_1m").fetchone()[0]
    duplicate_timestamps = con.execute(
        """
        select count(*)::bigint
        from (
          select timestamp_utc, count(*) as n
          from ohlcv_1m
          group by timestamp_utc
          having count(*) > 1
        )
        """
    ).fetchone()[0]
    con.execute(
        """
        create table if not exists l0_source_segments (
          source varchar,
          ticker varchar,
          fetched_at_utc timestamp,
          first_ts timestamp,
          last_ts timestamp,
          rows_fetched bigint,
          rows_inserted bigint,
          period varchar,
          interval varchar,
          replace_yfinance boolean,
          dry_run boolean
        )
        """
    )
    if not args.dry_run:
        con.execute(
            """
            insert into l0_source_segments
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                SOURCE_SYMBOL,
                TICKER,
                datetime.now(timezone.utc).replace(tzinfo=None),
                df["timestamp_utc"].min().to_pydatetime(),
                df["timestamp_utc"].max().to_pydatetime(),
                int(len(df)),
                inserted,
                args.period,
                args.interval,
                bool(args.replace_yfinance),
                bool(args.dry_run),
            ],
        )
    con.close()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "db": str(db_path),
        "ticker": TICKER,
        "source_symbol": SOURCE_SYMBOL,
        "period": args.period,
        "interval": args.interval,
        "replace_yfinance": bool(args.replace_yfinance),
        "dry_run": bool(args.dry_run),
        "before": {
            "rows": int(before_rows),
            "max_ts": str(before_max),
            "yfinance_rows": int(before_yf_rows),
        },
        "fetched": {
            "rows": int(len(df)),
            "first_ts": str(df["timestamp_utc"].min()),
            "last_ts": str(df["timestamp_utc"].max()),
            "overlap_rows": int(len(overlap)),
        },
        "append": {
            "rows_inserted": int(inserted),
            "first_ts": str(append["timestamp_utc"].min()) if not append.empty else None,
            "last_ts": str(append["timestamp_utc"].max()) if not append.empty else None,
        },
        "after": {
            "rows": int(after_rows),
            "max_ts": str(after_max),
            "duplicate_timestamps": int(duplicate_timestamps),
        },
    }
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if duplicate_timestamps == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

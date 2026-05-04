#!/usr/bin/env python3
"""
Live data buffer — SQLite-backed 90-day rolling 1m OHLCV cache.

Backfills from existing MGC_1m.db, then streams from yfinance.
Schema matches existing investing_ohlcv_1m table.

Usage:
    from pipeline.live.data_buffer import DataBuffer
    buffer = DataBuffer()
    buffer.backfill()           # once: fill last 90d from existing DB
    buffer.update()             # each minute: fetch latest candle
    df_1m = buffer.get("2026-04-20", "2026-04-24")  # query range
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_DB = ROOT / "data" / "Level_0_Raw" / "MGC_1m.db"
LIVE_DB = ROOT / "data" / "Live" / "live_buffer.db"
CANARY_DB = ROOT / "data" / "Live" / "topstepx_buffer.db"
SYMBOL = "MICRO_GOLD"
TIMEFRAME = "1m"
TICKER = "MGC=F"


class DataBuffer:
    """Manages a 90-day rolling 1m OHLCV buffer in SQLite."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or LIVE_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_1m (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    epoch_ms INTEGER NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL,
                    PRIMARY KEY (symbol, timeframe, epoch_ms)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_ts
                ON ohlcv_1m(timestamp_utc)
            """)
            conn.commit()

    # ── backfill from existing data ──────────────────────────────────────

    def backfill(self, days: int = 90) -> int:
        """Copy last *days* of 1m data from source DB into live buffer.

        Returns number of rows inserted.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        with sqlite3.connect(str(SOURCE_DB)) as src:
            df = pd.read_sql(
                "SELECT * FROM investing_ohlcv_1m "
                "WHERE symbol = ? AND timeframe = ? AND timestamp_utc >= ? "
                "ORDER BY epoch_ms",
                src,
                params=[SYMBOL, TIMEFRAME, start.strftime("%Y-%m-%d %H:%M:%S")],
            )

        if df.empty:
            print(f"[Buffer] No data found in source DB for last {days}d")
            return 0

        # Map to live schema
        df["symbol"] = SYMBOL
        df["timeframe"] = TIMEFRAME

        with sqlite3.connect(str(self.db_path)) as conn:
            existing = conn.execute(
                "SELECT epoch_ms FROM ohlcv_1m WHERE symbol = ? AND timeframe = ?",
                [SYMBOL, TIMEFRAME],
            ).fetchall()
            existing_ids = {r[0] for r in existing}

            new_rows = df[~df["epoch_ms"].isin(existing_ids)]
            if new_rows.empty:
                print(f"[Buffer] Already up to date ({len(df)} rows)")
                return 0

            new_rows.to_sql("ohlcv_1m", conn, if_exists="append", index=False)
            print(f"[Buffer] Backfilled {len(new_rows):,} rows "
                  f"({df['timestamp_utc'].iloc[0]} → {df['timestamp_utc'].iloc[-1]})")
            return len(new_rows)

    # ── live update from yfinance ───────────────────────────────────────

    def update(self) -> int:
        """Fetch latest 1m candles from yfinance since last stored candle.

        Returns number of new rows inserted.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT MAX(epoch_ms), MAX(timestamp_utc) FROM ohlcv_1m "
                "WHERE symbol = ? AND timeframe = ?",
                [SYMBOL, TIMEFRAME],
            ).fetchone()

        if row[1] is None:
            return 0

        last_ts = pd.Timestamp(row[1], tz="UTC")
        try:
            ticker = yf.Ticker(TICKER)
            df = ticker.history(period="5d", interval="1m")
        except Exception:
            return 0

        if df.empty:
            return 0

        df = df.reset_index()
        df["ts_utc"] = df["Datetime"].dt.tz_convert("UTC")
        df = df[df["ts_utc"] > last_ts]

        if df.empty:
            return 0

        # Compute epoch_ms using numpy to avoid pandas int overflow
        ts_ns = df["ts_utc"].values.astype("datetime64[ns]").astype("int64")
        df["epoch_ms"] = ts_ns // 1_000_000
        df["symbol"] = SYMBOL
        df["timeframe"] = TIMEFRAME
        df["timestamp_utc"] = df["ts_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["open"] = df["Open"].astype(float)
        df["high"] = df["High"].astype(float)
        df["low"] = df["Low"].astype(float)
        df["close"] = df["Close"].astype(float)
        df["volume"] = df["Volume"].fillna(0).astype(float)

        cols = ["symbol", "timeframe", "epoch_ms", "timestamp_utc",
                 "open", "high", "low", "close", "volume"]

        with sqlite3.connect(str(self.db_path)) as conn:
            df[cols].to_sql("ohlcv_1m", conn, if_exists="append", index=False)
        return len(df)

    # ── query ───────────────────────────────────────────────────────────

    def get(self, start: str, end: str) -> pd.DataFrame:
        """Return 1m OHLCV for date range as DataFrame.

        Args:
            start: start datetime string (UTC)
            end: end datetime string (UTC)

        Returns:
            DataFrame with columns: timestamp_utc, open, high, low, close, volume
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            df = pd.read_sql(
                "SELECT timestamp_utc, open, high, low, close, volume "
                "FROM ohlcv_1m "
                "WHERE symbol = ? AND timeframe = ? "
                "AND timestamp_utc >= ? AND timestamp_utc <= ? "
                "ORDER BY epoch_ms",
                conn,
                params=[SYMBOL, TIMEFRAME, start, end],
            )
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        return df

    def latest(self) -> dict | None:
        """Return the most recent candle."""
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT timestamp_utc, open, high, low, close, volume "
                "FROM ohlcv_1m WHERE symbol = ? AND timeframe = ? "
                "ORDER BY epoch_ms DESC LIMIT 1",
                [SYMBOL, TIMEFRAME],
            ).fetchone()
        if row is None:
            return None
        return {
            "ts": pd.Timestamp(row[0], tz="UTC"),
            "open": row[1], "high": row[2], "low": row[3],
            "close": row[4], "volume": row[5],
        }

    # ── gap detection & repair ──────────────────────────────────────────────

    def detect_gaps(self, lookback_bars: int = 100,
                    min_gap_minutes: int = 2) -> list[dict]:
        """Find gaps > min_gap_minutes in the most recent lookback_bars."""
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT epoch_ms, timestamp_utc FROM ohlcv_1m "
                "WHERE symbol = ? AND timeframe = ? "
                "ORDER BY epoch_ms DESC LIMIT ?",
                [SYMBOL, TIMEFRAME, lookback_bars],
            ).fetchall()
        gaps = []
        for i in range(len(rows) - 1):
            curr_ms = int(rows[i][0])
            next_ms = int(rows[i + 1][0])
            gap_min = (curr_ms - next_ms) / 60_000
            if gap_min > min_gap_minutes:
                gaps.append({
                    "from_ts": rows[i + 1][1],
                    "to_ts": rows[i][1],
                    "gap_minutes": round(gap_min, 1),
                })
        return gaps

    def fill_range(self, from_ts: str, to_ts: str) -> int:
        """Download yfinance data for a range and INSERT OR IGNORE missing bars."""
        from_utc = pd.Timestamp(from_ts, tz="UTC")
        to_utc = pd.Timestamp(to_ts, tz="UTC")
        try:
            ticker = yf.Ticker(TICKER)
            df = ticker.history(start=from_utc - pd.Timedelta(days=1),
                                end=to_utc + pd.Timedelta(days=1),
                                interval="1m")
        except Exception:
            return 0
        if df.empty:
            return 0

        df = df.reset_index()
        df["ts_utc"] = df["Datetime"].dt.tz_convert("UTC")
        df = df[(df["ts_utc"] >= from_utc) & (df["ts_utc"] <= to_utc)]
        if df.empty:
            return 0

        ts_ns = df["ts_utc"].values.astype("datetime64[ns]").astype("int64")
        df["epoch_ms"] = ts_ns // 1_000_000
        df["symbol"] = SYMBOL
        df["timeframe"] = TIMEFRAME
        df["timestamp_utc"] = df["ts_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["open"] = df["Open"].astype(float)
        df["high"] = df["High"].astype(float)
        df["low"] = df["Low"].astype(float)
        df["close"] = df["Close"].astype(float)
        df["volume"] = df["Volume"].fillna(0).astype(float)
        cols = ["symbol", "timeframe", "epoch_ms", "timestamp_utc",
                 "open", "high", "low", "close", "volume"]
        with sqlite3.connect(str(self.db_path)) as conn:
            existing = conn.execute(
                "SELECT epoch_ms FROM ohlcv_1m WHERE symbol = ? AND timeframe = ?",
                [SYMBOL, TIMEFRAME],
            ).fetchall()
            existing_ids = {r[0] for r in existing}
            new_rows = df[~df["epoch_ms"].isin(existing_ids)]
            if new_rows.empty:
                return 0
            new_rows[cols].to_sql("ohlcv_1m", conn, if_exists="append", index=False)
        return len(new_rows)

    def repair(self, max_gap_hours: int = 72) -> dict:
        """Fill gaps: yfinance first, fallback to MGC_1m.db for large gaps.

        Returns dict with counts: {yfinance, backfill}.
        """
        result = {"yfinance": 0, "backfill": 0}
        gaps = self.detect_gaps(lookback_bars=200, min_gap_minutes=2)
        if not gaps:
            return result

        now = datetime.now(timezone.utc)
        for gap in gaps:
            gap_hours = gap["gap_minutes"] / 60
            if gap_hours > max_gap_hours:
                # Large gap → MGC_1m.db backfill
                n = self.backfill(days=int(gap_hours / 24) + 1)
                result["backfill"] += n
                # Also fill recent end of gap from yfinance
                n2 = self.fill_range(gap["from_ts"], gap["to_ts"])
                result["yfinance"] += n2
            else:
                # Small gap → yfinance fill_range (INSERT OR IGNORE into gap)
                n = self.fill_range(gap["from_ts"], gap["to_ts"])
                if n > 0:
                    result["yfinance"] += n
                else:
                    # yfinance might not have it yet
                    n = self.update()
                    result["yfinance"] += n

        return result


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live data buffer manager")
    parser.add_argument("action", choices=["backfill", "update", "status"])
    args = parser.parse_args()

    buffer = DataBuffer()
    if args.action == "backfill":
        buffer.backfill()
    elif args.action == "update":
        n = buffer.update()
        print(f"[Buffer] {n} new rows")
    elif args.action == "status":
        latest = buffer.latest()
        if latest:
            print(f"[Buffer] Latest: {latest['ts']} close={latest['close']}")
        else:
            print("[Buffer] Empty — run backfill first")

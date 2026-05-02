#!/usr/bin/env python3
"""Append Databento OHLCV JSONL exports into the 1m raw MGC database.

Databento exports can contain several outrights plus calendar spreads for the
same minute. This keeps the existing repo convention from ingest_databento.py:
ignore spreads and select the highest-volume outright for each minute.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = ROOT / "data/Level_0_Raw/databento/glbx-mdp3-20260416-20260430.ohlcv-1m.json"
DB_PATH = ROOT / "data/Level_0_Raw/MGC_1m.db"
TABLE = "investing_ohlcv_1m"
SYMBOL = "MICRO_GOLD"
TIMEFRAME = "1m"


def parse_ts_ms(ts: str) -> tuple[int, str]:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000), dt.strftime("%Y-%m-%d %H:%M:%S")


def ingest(path: Path, db_path: Path = DB_PATH) -> int:
    best_by_epoch: dict[int, tuple] = {}
    skipped_spreads = 0
    rows_seen = 0

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            src_symbol = str(rec.get("symbol", ""))
            if "-" in src_symbol:
                skipped_spreads += 1
                continue

            volume = float(rec.get("volume") or 0)
            epoch_ms, ts_utc = parse_ts_ms(rec["hd"]["ts_event"])
            row = (
                SYMBOL,
                TIMEFRAME,
                epoch_ms,
                ts_utc,
                float(rec["open"]),
                float(rec["high"]),
                float(rec["low"]),
                float(rec["close"]),
                volume,
            )
            prev = best_by_epoch.get(epoch_ms)
            if prev is None or volume > prev[-1]:
                best_by_epoch[epoch_ms] = row
            rows_seen += 1

    rows = [best_by_epoch[k] for k in sorted(best_by_epoch)]
    if not rows:
        print(f"No rows to ingest from {path}")
        return 0

    start_ts = rows[0][3]
    end_ts = rows[-1][3]
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            f"DELETE FROM {TABLE} WHERE symbol=? AND timeframe=? "
            "AND timestamp_utc >= ? AND timestamp_utc <= ?",
            [SYMBOL, TIMEFRAME, start_ts, end_ts],
        )
        conn.executemany(
            f"INSERT OR REPLACE INTO {TABLE} "
            "(symbol, timeframe, epoch_ms, timestamp_utc, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()

    print(f"Input: {path}")
    print(f"Rows seen: {rows_seen:,}; spreads skipped: {skipped_spreads:,}")
    print(f"Inserted/replaced: {len(rows):,} front-month 1m rows")
    print(f"Range: {start_ts} -> {end_ts}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    ingest(args.path)


if __name__ == "__main__":
    main()

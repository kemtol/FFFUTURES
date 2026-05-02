import sqlite3
import pandas as pd
from pathlib import Path

SRC_DB = "data/Level_0_Raw/MGC_1m.db"
SRC_TABLE = "investing_ohlcv_1m"

TARGETS = [
    ("5m",  "5min",  "data/Level_0_Raw/MGC_5m.db",  "investing_ohlcv_5m"),
    ("15m", "15min", "data/Level_0_Raw/MGC_15m.db", "investing_ohlcv_15m"),
]

def resample(df: pd.DataFrame, rule: str, label: str) -> pd.DataFrame:
    df = df.set_index("datetime").sort_index()
    rs = df.resample(rule, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open"])
    rs = rs.reset_index()
    rs["timestamp_utc"] = rs["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    rs["epoch_ms"] = (rs["datetime"].astype("int64") // 1_000_000).astype(int)
    rs["symbol"] = "MICRO_GOLD"
    rs["timeframe"] = label
    return rs[["symbol", "timeframe", "epoch_ms", "timestamp_utc", "open", "high", "low", "close", "volume"]]

def ensure_table(conn: sqlite3.Connection, table: str) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            symbol        TEXT NOT NULL,
            timeframe     TEXT NOT NULL,
            epoch_ms      INTEGER NOT NULL,
            timestamp_utc TEXT NOT NULL,
            open          REAL NOT NULL,
            high          REAL NOT NULL,
            low           REAL NOT NULL,
            close         REAL NOT NULL,
            volume        REAL,
            PRIMARY KEY (symbol, timeframe, epoch_ms)
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_ts ON {table} (symbol, timeframe, epoch_ms)")
    conn.commit()

def main() -> None:
    print(f"Loading 1m data from {SRC_DB}...")
    src = sqlite3.connect(SRC_DB)
    df = pd.read_sql(
        f"SELECT epoch_ms, timestamp_utc, open, high, low, close, volume FROM {SRC_TABLE} WHERE symbol='MICRO_GOLD' ORDER BY epoch_ms",
        src,
    )
    src.close()
    df["datetime"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    print(f"Loaded {len(df):,} rows, {df['timestamp_utc'].iloc[0]} -> {df['timestamp_utc'].iloc[-1]}")

    for label, rule, db_path, table in TARGETS:
        print(f"\nResampling to {label}...")
        rs = resample(df, rule, label)
        print(f"  {len(rs):,} candles")

        conn = sqlite3.connect(db_path)
        ensure_table(conn, table)
        conn.execute(f"DELETE FROM {table} WHERE symbol='MICRO_GOLD'")
        conn.executemany(
            f"INSERT OR REPLACE INTO {table} (symbol, timeframe, epoch_ms, timestamp_utc, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?,?,?)",
            rs.itertuples(index=False, name=None),
        )
        conn.commit()
        r = conn.execute(f"SELECT COUNT(*), MIN(timestamp_utc), MAX(timestamp_utc) FROM {table}").fetchone()
        print(f"  DB: {r[0]:,} rows, {r[1]} -> {r[2]}")
        conn.close()

    print("\nDone.")

if __name__ == "__main__":
    main()

import yfinance as yf
import sqlite3
import pandas as pd

TICKER = "MGC=F"
SYMBOL = "MICRO_GOLD"

TARGETS = [
    ("1m",  "7d",  "data/Level_0_Raw/MGC_1m.db",  "investing_ohlcv_1m"),
    ("5m",  "7d",  "data/Level_0_Raw/MGC_5m.db",  "investing_ohlcv_5m"),
    ("15m", "7d",  "data/Level_0_Raw/MGC_15m.db", "investing_ohlcv_15m"),
]

def fetch_and_insert(interval: str, period: str, db_path: str, table: str) -> None:
    df = yf.Ticker(TICKER).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        print(f"  [{interval}] no data returned")
        return

    df = df.reset_index()
    df["Datetime"] = (
        df["Datetime"].dt.tz_localize("UTC")
        if df["Datetime"].dt.tz is None
        else df["Datetime"].dt.tz_convert("UTC")
    )

    batch = [
        (
            SYMBOL, interval,
            int(row["Datetime"].timestamp() * 1000),
            row["Datetime"].strftime("%Y-%m-%d %H:%M:%S"),
            float(row["Open"]), float(row["High"]),
            float(row["Low"]),  float(row["Close"]),
            float(row["Volume"]) if not pd.isna(row["Volume"]) else 0.0,
        )
        for _, row in df.iterrows()
    ]

    conn = sqlite3.connect(db_path)
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} "
        "(symbol, timeframe, epoch_ms, timestamp_utc, open, high, low, close, volume) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        batch,
    )
    conn.commit()
    conn.close()
    print(f"  [{interval}] {len(batch)} rows upserted")

def main() -> None:
    print(f"Updating {SYMBOL} ({TICKER})...")
    for interval, period, db_path, table in TARGETS:
        fetch_and_insert(interval, period, db_path, table)
    print("Done.")

if __name__ == "__main__":
    main()

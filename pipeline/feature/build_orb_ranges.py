"""
Level 1: Compute ORB high/low/range per (date, session, orb_tf).

Incremental — hanya compute tanggal yang belum ada di output parquet.
Output: data/Level_1_Features/orb_ranges.parquet
"""

import sqlite3
from datetime import time, date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
DB_5M  = ROOT / "data/Level_0_Raw/MGC_5m.db"
DB_15M = ROOT / "data/Level_0_Raw/MGC_15m.db"
OUT    = ROOT / "data/Level_1_Features/orb_ranges.parquet"

SESSIONS: dict[str, tuple[time, time]] = {
    "tokyo":  (time(0, 0),  time(3, 0)),
    "london": (time(7, 0),  time(10, 0)),
    "us":     (time(13, 30), time(16, 30)),
}

# (orb_tf_label, source_db, n_candles, candle_minutes)
ORB_CONFIGS = [
    ("5m",  DB_5M,  3, 5),   # 3 × 5m = 15 menit
    ("15m", DB_15M, 1, 15),  # 1 × 15m = 15 menit
    ("30m", DB_15M, 2, 15),  # 2 × 15m = 30 menit
]


def load_ohlcv(db_path: Path, table: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        f"SELECT timestamp_utc, open, high, low, close, volume "
        f"FROM {table} WHERE symbol='MICRO_GOLD' ORDER BY epoch_ms",
        conn,
    )
    conn.close()
    df["ts"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["date"] = df["ts"].dt.date
    df["time"] = df["ts"].dt.time
    return df


def compute_orb_ranges(df: pd.DataFrame, session: str, orb_tf: str, n_candles: int) -> pd.DataFrame:
    start_t, end_t = SESSIONS[session]
    in_session = df[(df["time"] >= start_t) & (df["time"] < end_t)]

    records = []
    for d, group in in_session.groupby("date"):
        group = group.sort_values("ts")
        orb = group.head(n_candles)
        if len(orb) < n_candles:
            continue
        orb_high  = orb["high"].max()
        orb_low   = orb["low"].min()
        orb_range = orb_high - orb_low
        if orb_range <= 0:
            continue
        records.append({
            "date":           d,
            "session":        session,
            "orb_tf":         orb_tf,
            "orb_high":       orb_high,
            "orb_low":        orb_low,
            "orb_range":      orb_range,
            "orb_start_ts":   orb["ts"].iloc[0],
            "orb_end_ts":     orb["ts"].iloc[-1] + pd.Timedelta(minutes=n_candles * (5 if orb_tf == "5m" else 15)),
            "session_close_ts": pd.Timestamp(str(d), tz="UTC").replace(
                hour=SESSIONS[session][1].hour, minute=SESSIONS[session][1].minute
            ),
        })

    return pd.DataFrame(records)


def main(full_rebuild: bool = False) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Load existing output untuk incremental
    existing_dates: set[date] = set()
    existing_df = pd.DataFrame()
    if OUT.exists() and not full_rebuild:
        existing_df = pd.read_parquet(OUT)
        existing_dates = set(existing_df["date"].unique())
        print(f"Existing: {len(existing_df):,} rows, {len(existing_dates)} dates")

    all_new = []

    for orb_tf, db_path, n_candles, _ in ORB_CONFIGS:
        table = "investing_ohlcv_5m" if db_path == DB_5M else "investing_ohlcv_15m"
        print(f"Loading {db_path.name} for ORB-{orb_tf}...")
        df = load_ohlcv(db_path, table)

        # Filter hanya tanggal baru
        if existing_dates:
            df = df[~df["date"].isin(existing_dates)]

        if df.empty:
            print(f"  [{orb_tf}] no new dates, skip")
            continue

        new_dates = df["date"].nunique()
        print(f"  [{orb_tf}] computing {new_dates} new dates...")

        for session in SESSIONS:
            result = compute_orb_ranges(df, session, orb_tf, n_candles)
            all_new.append(result)
            print(f"    {session}: {len(result)} ORBs")

    if not all_new:
        print("Already up to date.")
        return

    new_df = pd.concat(all_new, ignore_index=True)

    # Append ke existing
    final_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
    final_df = final_df.sort_values(["date", "session", "orb_tf"]).reset_index(drop=True)
    final_df.to_parquet(OUT, index=False)

    print(f"\nSaved: {len(final_df):,} total rows → {OUT}")


if __name__ == "__main__":
    import sys
    full_rebuild = "--rebuild" in sys.argv
    main(full_rebuild=full_rebuild)

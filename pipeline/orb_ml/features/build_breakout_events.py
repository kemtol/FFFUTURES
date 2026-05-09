"""
Level 1: Detect breakout events dari ORB ranges.

Untuk setiap ORB, cari candle pertama setelah ORB selesai yang close
di luar range → catat entry price, SL, dan sisi (continuation/reversal).

Output: data/Level_1_Features/breakout_events.parquet
"""

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DB_1M   = ROOT / "data/Level_0_Raw/MGC_1m.db"
ORB_IN  = ROOT / "data/Level_1_Features/orb_ranges.parquet"
OUT     = ROOT / "data/Level_1_Features/breakout_events.parquet"

ATR_PERIOD  = 14
SL_MULTIPLIER = 1.5


def load_1m() -> pd.DataFrame:
    conn = sqlite3.connect(DB_1M)
    df = pd.read_sql(
        "SELECT epoch_ms, timestamp_utc, high, low, close, volume "
        "FROM investing_ohlcv_1m WHERE symbol='MICRO_GOLD' ORDER BY epoch_ms",
        conn,
    )
    conn.close()
    df["ts"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Precompute ATR(14) on 1m
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    return df.set_index("ts").sort_index()


def detect_breakouts(orb: pd.DataFrame, df1m: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, row in orb.iterrows():
        orb_end   = row["orb_end_ts"]
        sess_close = row["session_close_ts"]

        # Candles setelah ORB selesai sampai session close
        window = df1m[(df1m.index > orb_end) & (df1m.index <= sess_close)]
        if window.empty:
            continue

        # Cari candle pertama yang close di luar ORB range
        breakout = None
        for ts, candle in window.iterrows():
            if candle["close"] > row["orb_high"]:
                breakout = {"side": 1, "ts": ts, "candle": candle}
                break
            elif candle["close"] < row["orb_low"]:
                breakout = {"side": -1, "ts": ts, "candle": candle}
                break

        if breakout is None:
            continue

        entry_price = breakout["candle"]["close"]
        atr = breakout["candle"]["atr14"]
        if pd.isna(atr) or atr <= 0:
            continue

        sl_dist = SL_MULTIPLIER * atr
        side    = breakout["side"]

        records.append({
            # ORB identity
            "date":             row["date"],
            "session":          row["session"],
            "orb_tf":           row["orb_tf"],
            "orb_high":         row["orb_high"],
            "orb_low":          row["orb_low"],
            "orb_range":        row["orb_range"],
            "orb_start_ts":     row["orb_start_ts"],
            "orb_end_ts":       orb_end,
            "session_close_ts": sess_close,
            # Breakout
            "breakout_ts":      breakout["ts"],
            "breakout_side":    side,
            "entry_price":      entry_price,
            "breakout_strength": (entry_price - row["orb_high"]) if side == 1
                                 else (row["orb_low"] - entry_price),
            # SL / TP levels
            "atr14_at_entry":   atr,
            "sl_dist":          sl_dist,
            "sl_price_cont":    entry_price - side * sl_dist,
            "tp2r_price_cont":  entry_price + side * 2 * sl_dist,
            "tp4r_price_cont":  entry_price + side * 4 * sl_dist,
            "sl_price_rev":     entry_price + side * sl_dist,
            "tp2r_price_rev":   entry_price - side * 2 * sl_dist,
            "tp4r_price_rev":   entry_price - side * 4 * sl_dist,
        })

    return pd.DataFrame(records)


def main(full_rebuild: bool = False) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    if not ORB_IN.exists():
        raise FileNotFoundError(f"Run build_orb_ranges.py first: {ORB_IN}")

    orb = pd.read_parquet(ORB_IN)

    # Incremental — skip dates sudah ada
    existing_dates: set = set()
    existing_df = pd.DataFrame()
    if OUT.exists() and not full_rebuild:
        existing_df = pd.read_parquet(OUT)
        existing_dates = set(existing_df["date"].unique())
        print(f"Existing: {len(existing_df):,} breakouts, {len(existing_dates)} dates")

    new_orb = orb[~orb["date"].isin(existing_dates)] if existing_dates else orb
    if new_orb.empty:
        print("Already up to date.")
        return

    print(f"Processing {new_orb['date'].nunique()} new dates, {len(new_orb)} ORBs...")
    print("Loading 1m data...")
    df1m = load_1m()

    print("Detecting breakouts...")
    new_df = detect_breakouts(new_orb, df1m)
    print(f"  Found {len(new_df):,} breakout events")

    final_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
    final_df = final_df.sort_values("breakout_ts").reset_index(drop=True)
    final_df.to_parquet(OUT, index=False)

    print(f"Saved: {len(final_df):,} total rows → {OUT}")


if __name__ == "__main__":
    import sys
    main(full_rebuild="--rebuild" in sys.argv)

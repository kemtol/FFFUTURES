"""
Level 1: Market context features per breakout event.

Features:
  orb_range_atr_ratio  — orb_range / atr14_at_entry
  day_of_week          — 0=Mon … 4=Fri
  time_in_session_min  — menit sejak session open
  price_vs_vwap_pct    — (entry - session_vwap) / vwap * 100
  adx_14_15m           — ADX(14) pada 15m saat breakout
  ema_slope_1h         — sign EMA(20) slope pada 1h  (1=up, -1=down)

Output: data/Level_1_Features/market_context.parquet
Also patches: data/Level_2_Datamart/training_datamart_orb.parquet
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent.parent.parent
DB_1M   = ROOT / "data/Level_0_Raw/MGC_1m.db"
DB_15M  = ROOT / "data/Level_0_Raw/MGC_15m.db"
BO_IN   = ROOT / "data/Level_1_Features/breakout_events.parquet"
OUT     = ROOT / "data/Level_1_Features/market_context.parquet"
DM_PATH = ROOT / "data/Level_2_Datamart/training_datamart_orb.parquet"

SESSION_OPENS = {"tokyo": (0, 0), "london": (7, 0), "us": (13, 30)}


# ── data loaders ──────────────────────────────────────────────────────────────

def load_1m() -> pd.DataFrame:
    conn = sqlite3.connect(DB_1M)
    df = pd.read_sql(
        "SELECT timestamp_utc, high, low, close, volume "
        "FROM investing_ohlcv_1m WHERE symbol='MICRO_GOLD' ORDER BY epoch_ms",
        conn,
    )
    conn.close()
    df["ts"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.set_index("ts").sort_index()


def load_15m() -> pd.DataFrame:
    conn = sqlite3.connect(DB_15M)
    df = pd.read_sql(
        "SELECT timestamp_utc, high, low, close "
        "FROM investing_ohlcv_15m WHERE symbol='MICRO_GOLD' ORDER BY epoch_ms",
        conn,
    )
    conn.close()
    df["ts"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.set_index("ts").sort_index()


# ── indicator computation ──────────────────────────────────────────────────────

def compute_session_vwap(df1m: pd.DataFrame) -> pd.Series:
    hmin = df1m.index.hour * 60 + df1m.index.minute
    tag  = pd.Series("none", index=df1m.index, dtype=str)
    tag[(hmin >= 0)   & (hmin < 180)] = "tokyo"
    tag[(hmin >= 420) & (hmin < 600)] = "london"
    tag[(hmin >= 810) & (hmin < 990)] = "us"

    in_sess   = tag != "none"
    group_key = df1m.index.normalize().astype(str) + "_" + tag

    typical = (df1m["high"] + df1m["low"] + df1m["close"]) / 3
    tp_vol  = typical * df1m["volume"]

    tmp = pd.DataFrame({"g": group_key, "tp": tp_vol, "v": df1m["volume"]})[in_sess]
    cum_tp = tmp.groupby("g")["tp"].cumsum()
    cum_v  = tmp.groupby("g")["v"].cumsum()

    vwap = pd.Series(np.nan, index=df1m.index)
    vwap[in_sess] = cum_tp / cum_v
    return vwap


def compute_adx(df15m: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df15m["high"], df15m["low"], df15m["close"]

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)

    dm_p = pd.Series(
        np.where((h - h.shift(1)) > (l.shift(1) - l), np.maximum((h - h.shift(1)).values, 0), 0),
        index=df15m.index,
    )
    dm_m = pd.Series(
        np.where((l.shift(1) - l) > (h - h.shift(1)), np.maximum((l.shift(1) - l).values, 0), 0),
        index=df15m.index,
    )

    a     = 1 / period
    atr_s = tr.ewm(alpha=a, adjust=False).mean()
    dmp_s = dm_p.ewm(alpha=a, adjust=False).mean()
    dmm_s = dm_m.ewm(alpha=a, adjust=False).mean()

    di_p = 100 * dmp_s / atr_s
    di_m = 100 * dmm_s / atr_s
    denom = (di_p + di_m).replace(0, np.nan)
    dx   = 100 * (di_p - di_m).abs() / denom
    return dx.ewm(alpha=a, adjust=False).mean()


def compute_ema_slope_1h(df1m: pd.DataFrame, period: int = 20) -> pd.Series:
    c1h  = df1m["close"].resample("1h").last().dropna()
    ema  = c1h.ewm(span=period, adjust=False).mean()
    slope = np.sign(ema - ema.shift(1))
    return slope.reindex(df1m.index, method="ffill")


# ── main ──────────────────────────────────────────────────────────────────────

def main(full_rebuild: bool = False) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    if not BO_IN.exists():
        raise FileNotFoundError(f"Run build_breakout_events.py first: {BO_IN}")

    bo = pd.read_parquet(BO_IN)

    existing_dates: set = set()
    existing_df = pd.DataFrame()
    if OUT.exists() and not full_rebuild:
        existing_df = pd.read_parquet(OUT)
        existing_dates = set(existing_df["date"].unique())
        print(f"Existing: {len(existing_df):,} rows, {len(existing_dates)} dates")

    new_bo = bo[~bo["date"].isin(existing_dates)] if existing_dates else bo
    if new_bo.empty:
        print("Already up to date.")
        return

    print(f"Processing {new_bo['date'].nunique()} dates, {len(new_bo)} breakouts...")

    print("Loading 1m / 15m data...")
    df1m  = load_1m()
    df15m = load_15m()

    print("Computing VWAP...")
    vwap_s = compute_session_vwap(df1m)

    print("Computing ADX(14) on 15m...")
    adx_s = compute_adx(df15m)

    print("Computing EMA slope 1h...")
    ema_s = compute_ema_slope_1h(df1m)

    # Convert to numpy for fast searchsorted lookup
    ep_1m  = df1m.index.view(np.int64) // 1_000_000
    ep_15m = df15m.index.view(np.int64) // 1_000_000
    vwap_a = vwap_s.values
    adx_a  = adx_s.values
    ema_a  = ema_s.values

    print("Building context rows...")
    records = []
    for _, row in new_bo.iterrows():
        bep = int(row["breakout_ts"].timestamp() * 1000)

        i1  = max(0, np.searchsorted(ep_1m,  bep, side="right") - 1)
        i15 = max(0, np.searchsorted(ep_15m, bep, side="right") - 1)

        vwap_val = vwap_a[i1]
        entry    = row["entry_price"]
        pvwap    = (entry - vwap_val) / vwap_val * 100 if not np.isnan(vwap_val) else np.nan

        sess       = row["session"]
        oh, om     = SESSION_OPENS[sess]
        sess_open  = pd.Timestamp(str(row["date"]), tz="UTC").replace(hour=oh, minute=om)
        t_in_sess  = (row["breakout_ts"] - sess_open).total_seconds() / 60

        records.append({
            "date":                row["date"],
            "session":             sess,
            "orb_tf":              row["orb_tf"],
            "breakout_ts":         row["breakout_ts"],
            "orb_range_atr_ratio": row["orb_range"] / row["atr14_at_entry"] if row["atr14_at_entry"] > 0 else np.nan,
            "day_of_week":         row["breakout_ts"].dayofweek,
            "time_in_session_min": t_in_sess,
            "vwap_at_breakout":    vwap_val,
            "price_vs_vwap_pct":   pvwap,
            "adx_14_15m":          adx_a[i15],
            "ema_slope_1h":        ema_a[i1],
        })

    new_df   = pd.DataFrame(records)
    final_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
    final_df = final_df.sort_values("breakout_ts").reset_index(drop=True)
    final_df.to_parquet(OUT, index=False)
    print(f"Saved market_context: {len(final_df):,} rows → {OUT}")

    # Patch Level 2 datamart
    if DM_PATH.exists():
        print("Patching training_datamart_orb.parquet...")
        dm = pd.read_parquet(DM_PATH)

        # Drop old context columns if rebuilding
        ctx_cols = ["orb_range_atr_ratio", "day_of_week", "time_in_session_min",
                    "vwap_at_breakout", "price_vs_vwap_pct", "adx_14_15m", "ema_slope_1h"]
        dm = dm.drop(columns=[c for c in ctx_cols if c in dm.columns])

        merge_keys = ["date", "session", "orb_tf", "breakout_ts"]
        dm = dm.merge(final_df[merge_keys + ctx_cols], on=merge_keys, how="left")
        dm.to_parquet(DM_PATH, index=False)
        print(f"Datamart updated: {len(dm):,} rows, {len(dm.columns)} cols → {DM_PATH}")


if __name__ == "__main__":
    import sys
    main(full_rebuild="--rebuild" in sys.argv)

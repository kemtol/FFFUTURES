#!/usr/bin/env python3
"""
SMART_1 aggressive pullback datamart v1.12.

Baseline discovery datamart for the DEMA pullback scalper:
- SuperTrend trend-continuation pullback candidates.
- DEMA100 directional filter.
- RR 1:1 TP/SL outcome with a 100-bar max hold.
- Indicator parameters aligned with the live Super Structure defaults.

This intentionally excludes macro inputs so the first baseline measures the
mechanical pullback edge before adding regime or macro overlays.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("/home/kemal/futures")
from pipeline.live.super_structure import _atr, adx, cci, dema, supertrend


ROOT = Path("/home/kemal/futures")
DB_PATH = ROOT / "data/Level_0_Raw/MGC_5m.db"
OUTPUT_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v1_12_training_datamart.parquet"

POINT_VALUE = 10.0
COMMISSION_RT = 1.74
ST_FACTOR = 4.0
ATR_PERIOD = 12
ADX_LENGTH = 12
CCI_LENGTH = 12
CCI_SOURCE = "hl2"
MAX_HOLD_BARS = 100
RR = 1.0
ST_BUFFER_PTS = 1.0
PULLBACK_BAND_ATR = 0.25
MIN_PULLBACK_BAND_PTS = 0.5
MIN_RISK_PTS = 0.1


def rsi(close: np.ndarray, period: int = 7) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    avg_up = pd.Series(up).rolling(period).mean()
    avg_down = pd.Series(down).rolling(period).mean()
    rs = avg_up / (avg_down + 1e-9)
    return (100 - (100 / (1 + rs))).values


def session_cluster(ts: pd.Timestamp) -> int:
    hour = int(ts.hour)
    if hour <= 6:
        return 0
    if hour <= 12:
        return 1
    return 2


def first_hit_outcome(
    side: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    future_high: np.ndarray,
    future_low: np.ndarray,
    future_close: np.ndarray,
    future_ts: np.ndarray,
) -> tuple[int, float, str, object, float, int]:
    if side == "Long":
        sl_hit = np.where(future_low <= sl_price)[0]
        tp_hit = np.where(future_high >= tp_price)[0]
    else:
        sl_hit = np.where(future_high >= sl_price)[0]
        tp_hit = np.where(future_low <= tp_price)[0]

    first_sl = sl_hit[0] if len(sl_hit) else None
    first_tp = tp_hit[0] if len(tp_hit) else None

    if first_tp is not None and (first_sl is None or first_tp < first_sl):
        idx = first_tp
        pnl_pts = abs(tp_price - entry_price)
        return 1, pnl_pts, "TP", future_ts[idx], tp_price, idx + 1

    if first_sl is not None:
        idx = first_sl
        pnl_pts = -abs(entry_price - sl_price)
        return 0, pnl_pts, "SL", future_ts[idx], sl_price, idx + 1

    if len(future_close) == 0:
        return 0, 0.0, "NO_FUTURE", pd.NaT, entry_price, 0

    idx = len(future_close) - 1
    if side == "Long":
        pnl_pts = float(future_close[idx] - entry_price)
    else:
        pnl_pts = float(entry_price - future_close[idx])
    return int(pnl_pts > 0), pnl_pts, "TIMEOUT", future_ts[idx], float(future_close[idx]), idx + 1


def build_v1_12_pullback_datamart() -> pd.DataFrame:
    print("Fetching 5m OHLCV data...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM investing_ohlcv_5m ORDER BY timestamp_utc", conn)
    conn.close()

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df[df["timestamp_utc"] >= "2023-01-01"].copy().reset_index(drop=True)

    print("Calculating aligned indicators...")
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)

    st, direction = supertrend(h, l, c, factor=ST_FACTOR, atr_period=ATR_PERIOD)
    atr = _atr(h, l, c, ATR_PERIOD)
    d50 = dema(c, 50)
    d100 = dema(c, 100)
    d200 = dema(c, 200)

    df["st"] = st
    df["st_direction"] = direction
    df["atr"] = atr
    df["dema_50"] = d50
    df["dema_100"] = d100
    df["dema_200"] = d200
    df["entry_adx"] = adx(h, l, c, ADX_LENGTH)
    df["entry_cci"] = cci(h, l, c, CCI_LENGTH, source=CCI_SOURCE)
    df["rsi_7"] = rsi(c, 7)
    df["prev_st_dir"] = df["st_direction"].shift(1)
    df["pullback_band"] = np.maximum(MIN_PULLBACK_BAND_PTS, df["atr"] * PULLBACK_BAND_ATR)

    cond_common = (
        df["st"].notna()
        & df["atr"].notna()
        & df["dema_200"].notna()
        & (df["st_direction"] == df["prev_st_dir"])
    )
    cond_long = (
        cond_common
        & (df["st_direction"] == -1)
        & (df["close"] > df["dema_100"])
        & (df["close"] > df["st"])
        & (df["low"] <= df["st"] + df["pullback_band"])
        & (df["close"] > df["open"])
    )
    cond_short = (
        cond_common
        & (df["st_direction"] == 1)
        & (df["close"] < df["dema_100"])
        & (df["close"] < df["st"])
        & (df["high"] >= df["st"] - df["pullback_band"])
        & (df["close"] < df["open"])
    )

    signals = df[cond_long | cond_short].copy()
    print(f"Found {len(signals)} pullback candidates. Simulating RR 1:1 outcomes...")

    full_h = df["high"].to_numpy(dtype=float)
    full_l = df["low"].to_numpy(dtype=float)
    full_c = df["close"].to_numpy(dtype=float)
    full_ts = df["timestamp_utc"].to_numpy()

    events = []
    for idx, row in signals.iterrows():
        side = "Long" if row["st_direction"] == -1 else "Short"
        entry_price = float(row["close"])

        if side == "Long":
            sl_price = float(row["st"] - ST_BUFFER_PTS)
            risk_pts = entry_price - sl_price
            tp_price = entry_price + risk_pts * RR
            touch_distance_atr = (float(row["low"]) - float(row["st"])) / (float(row["atr"]) + 1e-9)
        else:
            sl_price = float(row["st"] + ST_BUFFER_PTS)
            risk_pts = sl_price - entry_price
            tp_price = entry_price - risk_pts * RR
            touch_distance_atr = (float(row["st"]) - float(row["high"])) / (float(row["atr"]) + 1e-9)

        if risk_pts <= MIN_RISK_PTS:
            continue

        end_idx = min(idx + 1 + MAX_HOLD_BARS, len(df))
        label, pnl_pts, outcome, exit_ts, exit_price, hold_bars = first_hit_outcome(
            side,
            entry_price,
            sl_price,
            tp_price,
            full_h[idx + 1:end_idx],
            full_l[idx + 1:end_idx],
            full_c[idx + 1:end_idx],
            full_ts[idx + 1:end_idx],
        )

        body = abs(float(row["close"]) - float(row["open"]))
        bar_range = float(row["high"]) - float(row["low"])
        range_safe = bar_range + 1e-9
        atr_safe = float(row["atr"]) + 1e-9

        events.append({
            "pullback_id": f"PB12_{row['timestamp_utc'].strftime('%Y%m%d%H%M')}_{side}",
            "entry_ts": row["timestamp_utc"],
            "side": side,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": float(tp_price),
            "exit_ts": pd.Timestamp(exit_ts) if not pd.isna(exit_ts) else pd.NaT,
            "exit_price": float(exit_price),
            "exit_reason": outcome,
            "hold_bars": int(hold_bars),
            "risk_pts": float(risk_pts),
            "label": int(label),
            "pnl_pts": float(pnl_pts),
            "pnl_usd": float(pnl_pts * POINT_VALUE - COMMISSION_RT),
            "dist_d50_atr": float((row["close"] - row["dema_50"]) / atr_safe),
            "dist_d100_atr": float((row["close"] - row["dema_100"]) / atr_safe),
            "dist_d200_atr": float((row["close"] - row["dema_200"]) / atr_safe),
            "d50_slope": float(pd.Series(df["dema_50"]).diff(5).iloc[idx]),
            "d100_slope": float(pd.Series(df["dema_100"]).diff(5).iloc[idx]),
            "d200_slope": float(pd.Series(df["dema_200"]).diff(5).iloc[idx]),
            "close_slope_5": float(pd.Series(df["close"]).diff(5).iloc[idx]),
            "dema_stack": int(
                3 if row["close"] > row["dema_50"] > row["dema_100"] > row["dema_200"]
                else -3 if row["close"] < row["dema_50"] < row["dema_100"] < row["dema_200"]
                else 0
            ),
            "entry_adx": float(row["entry_adx"]),
            "entry_cci": float(row["entry_cci"]),
            "cci_abs": float(abs(row["entry_cci"])),
            "rsi_7": float(row["rsi_7"]),
            "wick_ratio": float((bar_range - body) / range_safe),
            "candle_body_atr": float(body / atr_safe),
            "bar_range_atr": float(bar_range / atr_safe),
            "st_gap_ratio": float(abs(row["close"] - row["st"]) / atr_safe),
            "touch_distance_atr": float(touch_distance_atr),
            "pullback_band_atr": float(row["pullback_band"] / atr_safe),
            "hour_utc": int(row["timestamp_utc"].hour),
            "dow": int(row["timestamp_utc"].dayofweek),
            "session_cluster": session_cluster(row["timestamp_utc"]),
        })

    out_df = pd.DataFrame(events)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(out_df)} events to {OUTPUT_PATH}")
    return out_df


if __name__ == "__main__":
    build_v1_12_pullback_datamart()

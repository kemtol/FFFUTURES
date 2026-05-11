"""Assemble full feature matrix from breakout events + market data."""

from __future__ import annotations

import pandas as pd
import numpy as np

from .session import SESSION_INDEX, minutes_into_session


def build_features(breakouts: pd.DataFrame, df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich breakout events with all model features.

    Args:
        breakouts: labeled breakout events (output of label_outcomes)
        df_1m:     full 1m OHLCV DataFrame

    Returns:
        Feature matrix ready for LGBM training/inference
    """
    df_1m = df_1m.copy()
    df_1m["_ts"] = pd.to_datetime(df_1m["timestamp_utc"], utc=True)
    df_1m = df_1m.sort_values("_ts").reset_index(drop=True)

    # Precompute ATR(14) on 1m
    df_1m["_tr"] = pd.concat([
        df_1m["high"] - df_1m["low"],
        (df_1m["high"] - df_1m["close"].shift(1)).abs(),
        (df_1m["low"]  - df_1m["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df_1m["_atr14"] = df_1m["_tr"].rolling(14).mean()

    records = []
    for _, row in breakouts.iterrows():
        ts = row["breakout_ts"]
        lookback = df_1m[df_1m["_ts"] <= ts].tail(300)
        if lookback.empty:
            continue

        atr = lookback["_atr14"].iloc[-1]
        vol_ma20 = lookback["volume"].rolling(20).mean().iloc[-1]
        breakout_candle = lookback.iloc[-1]

        # HTF trend proxies via EMA on available 1m data
        ema20_1h = lookback["close"].ewm(span=60).mean()
        ema20_4h = lookback["close"].ewm(span=240).mean()
        htf_1h = 1 if ema20_1h.iloc[-1] > ema20_1h.iloc[-2] else -1
        htf_4h = 1 if ema20_4h.iloc[-1] > ema20_4h.iloc[-2] else -1

        # Prev day close (approx: close 1440 1m bars ago)
        prev_close = lookback["close"].iloc[-min(1440, len(lookback))]
        price_vs_prev_close_pct = (row["breakout_price"] - prev_close) / prev_close * 100

        t = ts.time()
        session = row["session"]

        records.append({
            # Breakout context
            "orb_range":                    row["orb_range"],
            "orb_range_atr_ratio":          row["orb_range"] / atr if atr else np.nan,
            "breakout_side":                row["breakout_side"],
            "breakout_strength":            row["breakout_strength"],
            "breakout_candle_volume_ratio": breakout_candle["volume"] / vol_ma20 if vol_ma20 else np.nan,
            # Session & time
            "session":                      SESSION_INDEX[session],
            "orb_tf":                       int(row["orb_tf"].replace("m", "")),
            "time_in_session":              minutes_into_session(t, session),
            "day_of_week":                  ts.dayofweek,
            # Market structure
            "htf_trend_1h":                 htf_1h,
            "htf_trend_4h":                 htf_4h,
            "atr_14_1m":                    atr,
            "price_vs_prev_close_pct":      price_vs_prev_close_pct,
            # Labels
            "y_60m":                        row.get("y_60m"),
            "y_120m":                       row.get("y_120m"),
            "y_240m":                       row.get("y_240m"),
            # Metadata (dropped before training)
            "_breakout_ts":                 ts,
            "_session":                     session,
            "_orb_tf":                      row["orb_tf"],
        })

    return pd.DataFrame(records)

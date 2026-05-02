"""Label generation: follow through vs fake out after ORB breakout."""

from __future__ import annotations

import pandas as pd

HORIZONS_MINUTES = [60, 120, 240]


def label_outcomes(breakouts: pd.DataFrame, df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    For each breakout, label whether price follows through within each horizon.

    Follow through = price moves >= 1R (orb_range) beyond breakout level
    in the breakout direction within horizon minutes.

    Args:
        breakouts: output of detect_breakouts()
        df_1m:     1m OHLCV DataFrame with timestamp_utc column

    Returns:
        breakouts with added columns: y_60m, y_120m, y_240m
    """
    df_1m = df_1m.copy()
    df_1m["_ts"] = pd.to_datetime(df_1m["timestamp_utc"], utc=True)

    results = []
    for _, row in breakouts.iterrows():
        record = row.to_dict()
        target = row["breakout_price"] + row["breakout_side"] * row["orb_range"]

        for h in HORIZONS_MINUTES:
            end_ts = row["breakout_ts"] + pd.Timedelta(minutes=h)
            window = df_1m[(df_1m["_ts"] > row["breakout_ts"]) & (df_1m["_ts"] <= end_ts)]

            if row["breakout_side"] == 1:
                hit = (window["high"] >= target).any()
            else:
                hit = (window["low"] <= target).any()

            record[f"y_{h}m"] = int(hit)

        results.append(record)

    return pd.DataFrame(results)

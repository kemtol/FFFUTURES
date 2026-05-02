"""ORB (Opening Range Breakout) detection and range computation."""

from __future__ import annotations

import pandas as pd

from .session import SESSION_BOUNDARIES, Session

# Number of candles used to form ORB per timeframe
ORB_CANDLES: dict[str, int] = {
    "5m":  1,   # first 5m candle of session
    "15m": 1,   # first 15m candle
    "30m": 2,   # first two 15m candles (uses MGC_15m)
}


def compute_orb(df: pd.DataFrame, session: Session, orb_tf: str) -> pd.DataFrame:
    """
    Compute ORB high/low for each session occurrence in df.

    df must have columns: [timestamp_utc (parsed as datetime, UTC), open, high, low, close, volume]
    Returns DataFrame with columns: [date, session, orb_tf, orb_high, orb_low, orb_range, session_open_ts]
    """
    start_time, end_time = SESSION_BOUNDARIES[session]
    n_candles = ORB_CANDLES[orb_tf]

    df = df.copy()
    df["_ts"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["_date"] = df["_ts"].dt.date
    df["_time"] = df["_ts"].dt.time

    in_session = df[(df["_time"] >= start_time) & (df["_time"] < end_time)]

    records = []
    for date, group in in_session.groupby("_date"):
        group = group.sort_values("_ts")
        orb_candles = group.head(n_candles)
        if len(orb_candles) < n_candles:
            continue
        orb_high = orb_candles["high"].max()
        orb_low  = orb_candles["low"].min()
        records.append({
            "date":           date,
            "session":        session,
            "orb_tf":         orb_tf,
            "orb_high":       orb_high,
            "orb_low":        orb_low,
            "orb_range":      orb_high - orb_low,
            "session_open_ts": orb_candles["_ts"].iloc[0],
            "orb_end_ts":     orb_candles["_ts"].iloc[-1] + pd.Timedelta(orb_tf),
        })

    return pd.DataFrame(records)


def detect_breakouts(df: pd.DataFrame, orb: pd.DataFrame) -> pd.DataFrame:
    """
    For each ORB, detect the first candle that breaks above orb_high or below orb_low
    after the ORB formation period.

    Returns rows with breakout metadata appended.
    """
    df = df.copy()
    df["_ts"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    results = []
    for _, row in orb.iterrows():
        after_orb = df[df["_ts"] > row["orb_end_ts"]].sort_values("_ts")
        for _, candle in after_orb.iterrows():
            if candle["close"] > row["orb_high"]:
                results.append({**row, "breakout_ts": candle["_ts"], "breakout_side": 1,
                                 "breakout_price": candle["close"],
                                 "breakout_strength": candle["close"] - row["orb_high"]})
                break
            elif candle["close"] < row["orb_low"]:
                results.append({**row, "breakout_ts": candle["_ts"], "breakout_side": -1,
                                 "breakout_price": candle["close"],
                                 "breakout_strength": row["orb_low"] - candle["close"]})
                break

    return pd.DataFrame(results)

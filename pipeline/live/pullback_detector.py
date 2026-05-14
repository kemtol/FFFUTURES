"""
SMART_1 Aggressive Mode — Pullback Event Detector.

Mirrors the v1.12 datamart builder rules
(`pipeline/super_structure_ml/train/build_training_datamart_v1_12.py`):

- ST direction unchanged from previous bar (trend continuation, no flip).
- Long: ST bullish (-1), close > dema_100, close > st, low touches `st + band`,
        green candle.
- Short: ST bearish (+1), close < dema_100, close < st, high touches `st - band`,
         red candle.
- Pullback band = max(0.5 pts, atr * 0.25).
- SL: ST ± ST_BUFFER_PTS (1.0). TP: RR 1:1 from entry.
- Skip if risk_pts <= MIN_RISK_PTS.

Pure functions — no I/O, no model loading, no state. Safe to import anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

# Constants pinned to v1.12 datamart builder.
ST_BUFFER_PTS = 1.0
PULLBACK_BAND_ATR = 0.25
MIN_PULLBACK_BAND_PTS = 0.5
MIN_RISK_PTS = 0.1
RR = 1.0
MAX_HOLD_BARS = 100


@dataclass(frozen=True)
class PullbackEvent:
    side: str             # "Long" or "Short"
    entry_price: float
    sl_price: float
    tp_price: float
    risk_pts: float
    pullback_band: float


def pullback_band(atr: float) -> float:
    return max(MIN_PULLBACK_BAND_PTS, atr * PULLBACK_BAND_ATR)


def detect_pullback_event(
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    st: float,
    st_dir: int,
    prev_st_dir: int,
    atr: float,
    dema_100: float,
    dema_200: float,
) -> PullbackEvent | None:
    """Return a pullback event for the current bar, or None.

    Caller must already have valid (non-NaN) indicators. The caller is
    responsible for keeping `prev_st_dir` and the indicator series in sync
    with the bar being evaluated.
    """
    if st_dir != prev_st_dir:
        return None  # ST regime flipped — not a continuation pullback.

    band = pullback_band(atr)

    is_long = (
        st_dir == -1
        and close > dema_100
        and close > st
        and low <= st + band
        and close > open_
    )
    is_short = (
        st_dir == 1
        and close < dema_100
        and close < st
        and high >= st - band
        and close < open_
    )

    if not (is_long or is_short):
        return None

    if is_long:
        entry = close
        sl = st - ST_BUFFER_PTS
        risk = entry - sl
        tp = entry + risk * RR
        side = "Long"
    else:
        entry = close
        sl = st + ST_BUFFER_PTS
        risk = sl - entry
        tp = entry - risk * RR
        side = "Short"

    if risk <= MIN_RISK_PTS:
        return None

    return PullbackEvent(
        side=side,
        entry_price=float(entry),
        sl_price=float(sl),
        tp_price=float(tp),
        risk_pts=float(risk),
        pullback_band=float(band),
    )

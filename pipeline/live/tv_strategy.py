#!/usr/bin/env python3
"""
TradingView Strategy Port — Python implementation.
Supertrend + DEMA + ADX + CCI + Session Filter.

Runs on buffer data (topstepx_buffer.db), generates signals in
the same format as webhook parser (action=BUY/SELL/CLOSE).

Usage:
    python3 pipeline/live/tv_strategy.py [--live]
"""

from __future__ import annotations

import json
import time as _time
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

# ── config ────────────────────────────────────────────────────────────────────
ATR_PERIOD = 12
ST_FACTOR = 4.0
DEMA_LENGTH = 200
ADX_LENGTH = 12
ADX_THRESHOLD = 25
CCI_LENGTH = 12
CCI_SOURCE = "hl2"
CCI_LONG_MIN = 100.0
CCI_SHORT_MAX = -100.0
USE_ADX = True
USE_CCI = True
USE_SESSION = False
SYMBOL = "MGC"
# NY session 08:00-15:00 EST = 13:00-20:00 UTC
SESSION_START = time(13, 0)
SESSION_END = time(20, 0)
AUTO_CLOSE = True

SIGNALS_PATH = ROOT / "data" / "Live" / "tv_signals.json"
EPS = 1e-10


# ── indicators ────────────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=period, adjust=False).mean().values


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
    tr = np.maximum(h - l,
                    np.maximum(np.abs(h - np.roll(c, 1)),
                               np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return _rma(tr, period)


def _rma(arr: np.ndarray, period: int) -> np.ndarray:
    """TradingView-style RMA/Wilder smoothing used by ta.atr()."""
    out = np.full(len(arr), np.nan)
    if len(arr) < period:
        return out
    out[period - 1] = np.nanmean(arr[:period])
    for i in range(period, len(arr)):
        out[i] = (out[i - 1] * (period - 1) + arr[i]) / period
    return out


def supertrend(h: np.ndarray, l: np.ndarray, c: np.ndarray,
               factor: float = 4.0, atr_period: int = 10) -> np.ndarray:
    """TradingView-compatible Supertrend.
    Returns: st_values (float array), direction (-1=UP,+1=DOWN).
    """
    n = len(c)
    atr_val = _atr(h, l, c, atr_period)
    hl2 = (h + l) / 2.0
    upper = hl2 + factor * atr_val
    lower = hl2 - factor * atr_val
    st = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)

    for i in range(n):
        if np.isnan(atr_val[i]):
            continue
        if i == 0 or np.isnan(atr_val[i - 1]):
            direction[i] = 1
            st[i] = upper[i]
            continue

        prev_lower = lower[i - 1]
        prev_upper = upper[i - 1]

        if not (lower[i] > prev_lower or c[i - 1] < prev_lower):
            lower[i] = prev_lower
        if not (upper[i] < prev_upper or c[i - 1] > prev_upper):
            upper[i] = prev_upper

        prev_st = st[i - 1]
        if np.isnan(prev_st) or np.isclose(prev_st, prev_upper, equal_nan=False):
            direction[i] = -1 if c[i] > upper[i] else 1
        else:
            direction[i] = 1 if c[i] < lower[i] else -1

        st[i] = lower[i] if direction[i] == -1 else upper[i]

    return st, direction


def dema(c: np.ndarray, period: int) -> np.ndarray:
    e1 = _ema(c, period)
    e2 = _ema(e1, period)
    return 2.0 * e1 - e2


def adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
    """TradingView ta.dmi() ADX using Wilder/RMA smoothing."""
    up = np.full(len(c), np.nan)
    dn = np.full(len(c), np.nan)
    tr_val = np.full(len(c), np.nan)
    up[1:] = h[1:] - h[:-1]
    dn[1:] = l[:-1] - l[1:]

    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdm[0] = np.nan
    ndm[0] = np.nan
    tr_val[1:] = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])),
    )

    atr_s = _rma(tr_val, period)
    pdi = 100 * _rma(pdm, period) / (atr_s + EPS)
    ndi = 100 * _rma(ndm, period) / (atr_s + EPS)
    denom = pdi + ndi + EPS
    dx = 100 * np.abs(pdi - ndi) / denom
    adx_val = np.full(len(c), np.nan)
    valid = np.where(~np.isnan(dx))[0]
    if len(valid) < period:
        return adx_val
    start = valid[period - 1]
    adx_val[start] = np.nanmean(dx[valid[:period]])
    for i in range(start + 1, len(c)):
        adx_val[i] = adx_val[i - 1] if np.isnan(dx[i]) else (adx_val[i - 1] * (period - 1) + dx[i]) / period
    return adx_val


def cci(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int,
        source: str = "hl2") -> np.ndarray:
    """Commodity Channel Index. source: 'hlc3' or 'hl2'."""
    if source == "hl2":
        tp = (h + l) / 2.0
    else:
        tp = (h + l + c) / 3.0
    sma = pd.Series(tp).rolling(window=period).mean().values
    md = np.full(len(tp), np.nan)
    for i in range(period - 1, len(tp)):
        mean = sma[i]
        md[i] = np.mean(np.abs(tp[i - period + 1:i + 1] - mean))
    result = (tp - sma) / (0.015 * md + EPS)
    return result


# ── strategy runner ───────────────────────────────────────────────────────────

class TVStrategy:
    """Replicate TradingView strategy in Python using buffer data."""

    def __init__(self, buffer_or_db_path=None):
        from pipeline.live.buffer import DataBuffer, CANARY_DB
        if buffer_or_db_path is None:
            self.buffer = DataBuffer(db_path=CANARY_DB)
        elif isinstance(buffer_or_db_path, Path):
            self.buffer = DataBuffer(db_path=buffer_or_db_path)
        else:
            self.buffer = buffer_or_db_path

        self._pos = 0
        self._entry_price = 0.0
        self._sl_price = 0.0
        self._last_ts: pd.Timestamp | None = None
        self._signals: list[dict] = []
        self._entry_bar_ts: pd.Timestamp | None = None
        self._load_signals()

    def _load_signals(self) -> None:
        if SIGNALS_PATH.exists():
            try:
                self._signals = json.loads(SIGNALS_PATH.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                self._signals = []

    def _save_signals(self) -> None:
        SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SIGNALS_PATH.write_text(json.dumps(self._signals, indent=2, default=str))

    def _store_signal(self, action: str, price: float, sl: float = 0.0,
                      reason: str = "") -> None:
        sig = {
            "action": action,
            "symbol": SYMBOL,
            "entry": price if action in ("BUY", "SELL") else price,
            "sl": sl,
            "reason": reason,
            "parsed_from": f"tv_strategy.py: {action} {SYMBOL} @ {price:.1f}",
        }
        entry = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "signal": sig,
        }
        self._signals.append(entry)
        self._save_signals()
        print(f"[TV] ⚡ SIGNAL: {action} {SYMBOL} @ {price:.1f}" +
              (f" SL={sl:.1f}" if sl > 0 else "") +
              (f" ({reason})" if reason else ""), flush=True)

    def _is_in_session(self, ts: pd.Timestamp) -> bool:
        if not USE_SESSION:
            return True
        t = ts.tz_convert("America/New_York").time()
        # Simple check: hour-based
        return SESSION_START <= ts.tz_convert("UTC").time() <= SESSION_END

    def _is_end_of_session(self, ts: pd.Timestamp) -> bool:
        if not USE_SESSION or not AUTO_CLOSE:
            return False
        return ts.tz_convert("UTC").time() >= SESSION_END

    def check(self) -> list[dict]:
        """Fetch latest data and check for signals. Returns new signals."""
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
        end = now.strftime("%Y-%m-%d %H:%M:%S")
        df = self.buffer.get(start, end)

        if len(df) < max(DEMA_LENGTH, ADX_LENGTH + 50, CCI_LENGTH + 50):
            return []

        # Resample 1m to 5m (TradingView chart timeframe)
        df = df.set_index("timestamp_utc").sort_index()
        df_5m = df.resample("5min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

        if len(df_5m) < 50:
            return []
        df = df_5m
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)

        st, direction = supertrend(h, l, c, ST_FACTOR, ATR_PERIOD)
        d = dema(c, DEMA_LENGTH)
        ax = adx(h, l, c, ADX_LENGTH)
        cx = cci(h, l, c, CCI_LENGTH, source=CCI_SOURCE)

        last_ts = df.index[-1]
        if last_ts == self._last_ts:
            return []  # No new bar
        self._last_ts = last_ts

        i = len(c) - 1
        if i < DEMA_LENGTH:
            return []

        cur_close = c[i]
        cur_dema = d[i]
        cur_dir = direction[i]
        cur_adx = ax[i] if i < len(ax) and not np.isnan(ax[i]) else 0
        cur_cci = cx[i]
        cur_st = st[i]

        adx_ok = not USE_ADX or cur_adx > ADX_THRESHOLD
        cci_ok_long = not USE_CCI or cur_cci > CCI_LONG_MIN
        cci_ok_short = not USE_CCI or cur_cci < CCI_SHORT_MAX
        in_session = self._is_in_session(last_ts)
        end_session = self._is_end_of_session(last_ts)

        new_signals = []

        # ── Entry ──────────────────────────────────────────────────────────
        cross_up = (c[i-1] < d[i-1] and cur_close > cur_dema)
        cross_dn = (c[i-1] > d[i-1] and cur_close < cur_dema)
        long_signal = in_session and adx_ok and cci_ok_long and \
                    (cross_up or cur_close > cur_dema) and cur_dir < 0
        short_signal = in_session and adx_ok and cci_ok_short and \
                     (cross_dn or cur_close < cur_dema) and cur_dir > 0

        # ── Existing Stop Orders ───────────────────────────────────────────
        # TradingView strategy.exit(stop=...) submits a stop after a bar is
        # calculated; that active stop is evaluated on following bars.
        if self._pos == 1:
            if l[i] <= self._sl_price:
                self._store_signal("CLOSE", self._sl_price, reason="SL")
                new_signals.append(self._signals[-1])
                self._pos = 0

        if self._pos == -1:
            if h[i] >= self._sl_price:
                self._store_signal("CLOSE", self._sl_price, reason="SL")
                new_signals.append(self._signals[-1])
                self._pos = 0

        # ── Trend flip ─────────────────────────────────────────────────────
        if self._pos == 1 and cur_dir > 0:
            self._store_signal("CLOSE", cur_close, reason="TREND_FLIP")
            new_signals.append(self._signals[-1])
            self._pos = 0

        elif self._pos == -1 and cur_dir < 0:
            self._store_signal("CLOSE", cur_close, reason="TREND_FLIP")
            new_signals.append(self._signals[-1])
            self._pos = 0

        # Submit/update the dynamic stop for the next bar.
        if self._pos != 0:
            self._sl_price = cur_st

        # ── Session end ───────────────────────────────────────────────────
        if self._pos != 0 and end_session:
            self._store_signal("CLOSE", cur_close, reason="SESSION_END")
            new_signals.append(self._signals[-1])
            self._pos = 0

        # ── Entry ──────────────────────────────────────────────────────────
        if long_signal and self._pos == 0:
            self._pos = 1
            self._entry_price = cur_close
            self._sl_price = cur_st
            self._store_signal("BUY", cur_close, cur_st)
            new_signals.append(self._signals[-1])

        if short_signal and self._pos == 0:
            self._pos = -1
            self._entry_price = cur_close
            self._sl_price = cur_st
            self._store_signal("SELL", cur_close, cur_st)
            new_signals.append(self._signals[-1])

        return new_signals

    def stats(self) -> str:
        n = len(self._signals)
        buys = sum(1 for s in self._signals if s["signal"]["action"] == "BUY")
        sells = sum(1 for s in self._signals if s["signal"]["action"] == "SELL")
        closes = n - buys - sells
        pos = "LONG" if self._pos == 1 else "SHORT" if self._pos == -1 else "FLAT"
        return (f"[TV Strategy] {n} signals ({buys}B {sells}S {closes}C) | "
                f"Pos: {pos} | Entry: {self._entry_price:.1f}")

    def run_live(self) -> None:
        """Main loop — check every 30 seconds."""
        print("[TV] Starting live strategy loop...", flush=True)
        print(f"[TV] Config: ST({ST_FACTOR},{ATR_PERIOD}) DEMA({DEMA_LENGTH}) "
              f"ADX({ADX_LENGTH}>{ADX_THRESHOLD}) CCI({CCI_LENGTH})", flush=True)
        while True:
            try:
                signals = self.check()
                for s in signals:
                    pass  # signals already stored + printed in _store_signal
                _time.sleep(30)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[TV] Error: {e}", flush=True)
                _time.sleep(60)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TV Strategy Python Port")
    parser.add_argument("--live", action="store_true", help="Run live monitoring loop")
    parser.add_argument("--test", action="store_true", help="Single check + print status")
    args = parser.parse_args()

    strategy = TVStrategy()

    if args.test:
        signals = strategy.check()
        print(f"[TV] Checked: {len(signals)} new signals")
        print(strategy.stats())

    if args.live:
        strategy.run_live()

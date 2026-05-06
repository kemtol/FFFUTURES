#!/usr/bin/env python3
"""Walkforward: run Python Super Structure on April 29-30, print signals."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.live.super_structure import (
    ADX_LENGTH, ADX_THRESHOLD, ATR_PERIOD, CCI_LENGTH, CCI_LONG_MIN,
    CCI_SHORT_MAX, CCI_SOURCE, DEMA_LENGTH, ST_FACTOR,
    _atr, adx, cci, dema, supertrend,
)

RAW_DB = ROOT / "data" / "Level_0_Raw" / "MGC_1m.db"


def session_label(ts: pd.Timestamp) -> str:
    h = ts.hour + ts.minute / 60.0
    if 0 <= h < 3: return "Tokyo"
    if 7 <= h < 10: return "London"
    if 13.5 <= h < 16.5: return "US"
    return "Other"


def build_py_trade_list(start: str, end: str) -> list[dict]:
    warmup = (pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=120)).strftime("%Y-%m-%d %H:%M:%S")
    end_str = pd.Timestamp(end, tz="UTC").strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(str(RAW_DB)) as conn:
        df = pd.read_sql(
            """SELECT timestamp_utc, open, high, low, close
               FROM investing_ohlcv_1m
               WHERE symbol='MICRO_GOLD' AND timeframe='1m'
                 AND timestamp_utc >= ? AND timestamp_utc < ?
               ORDER BY epoch_ms""",
            conn, params=[warmup, end_str],
        )
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df_5m = df.set_index("timestamp_utc") \
              .resample("5min", label="right", closed="left") \
              .agg(open=("open","first"), high=("high","max"),
                   low=("low","min"), close=("close","last")).dropna()

    h = df_5m["high"].values.astype(float)
    l = df_5m["low"].values.astype(float)
    c = df_5m["close"].values.astype(float)

    st, direction = supertrend(h, l, c, ST_FACTOR, ATR_PERIOD)
    d_arr = dema(c, DEMA_LENGTH)
    ax = adx(h, l, c, ADX_LENGTH)
    cx = cci(h, l, c, CCI_LENGTH, source=CCI_SOURCE)

    pos = 0
    sl_price = 0.0
    trades = []
    entry = None

    for i in range(DEMA_LENGTH + 50, len(c)):
        cur_close = float(c[i])
        cur_dema = float(d_arr[i])
        cur_dir = int(direction[i])
        cur_st = float(st[i]) if not np.isnan(st[i]) else 0.0
        cur_adx = float(ax[i]) if i < len(ax) and not np.isnan(ax[i]) else 0.0
        cur_cci = float(cx[i]) if not np.isnan(cx[i]) else 0.0
        cur_ts = df_5m.index[i]
        cur_high = float(h[i])
        cur_low = float(l[i])

        adx_ok = cur_adx > ADX_THRESHOLD
        cci_ok_long = cur_cci > CCI_LONG_MIN
        cci_ok_short = cur_cci < CCI_SHORT_MAX
        cross_up = (float(c[i-1]) < d_arr[i-1] and cur_close > cur_dema)
        cross_dn = (float(c[i-1]) > d_arr[i-1] and cur_close < cur_dema)
        long_sig = adx_ok and cci_ok_long and (cross_up or cur_close > cur_dema) and cur_dir < 0
        short_sig = adx_ok and cci_ok_short and (cross_dn or cur_close < cur_dema) and cur_dir > 0

        # Stop hit
        if pos == 1 and cur_low <= sl_price:
            if entry and entry["ts"] >= start:
                trades.append({**entry, "exit_ts": str(cur_ts), "exit_price": float(sl_price),
                               "exit_reason": "SL", "pnl": (float(sl_price) - entry["price"]) * 10 - 3})
            pos = 0; entry = None
        elif pos == -1 and cur_high >= sl_price:
            if entry and entry["ts"] >= start:
                trades.append({**entry, "exit_ts": str(cur_ts), "exit_price": float(sl_price),
                               "exit_reason": "SL", "pnl": (entry["price"] - float(sl_price)) * 10 - 3})
            pos = 0; entry = None

        # Trend flip
        if pos == 1 and cur_dir > 0:
            if entry and entry["ts"] >= start:
                trades.append({**entry, "exit_ts": str(cur_ts), "exit_price": cur_close,
                               "exit_reason": "TREND_FLIP", "pnl": (cur_close - entry["price"]) * 10 - 3})
            pos = 0; entry = None
        elif pos == -1 and cur_dir < 0:
            if entry and entry["ts"] >= start:
                trades.append({**entry, "exit_ts": str(cur_ts), "exit_price": cur_close,
                               "exit_reason": "TREND_FLIP", "pnl": (entry["price"] - cur_close) * 10 - 3})
            pos = 0; entry = None

        if pos != 0:
            sl_price = cur_st

        if long_sig and pos == 0:
            pos = 1; sl_price = cur_st
            entry = {"ts": str(cur_ts), "side": "Long", "price": cur_close,
                     "adx": cur_adx, "cci": cur_cci, "dema": cur_dema, "sl": cur_st,
                     "session": session_label(cur_ts)}
        elif short_sig and pos == 0:
            pos = -1; sl_price = cur_st
            entry = {"ts": str(cur_ts), "side": "Short", "price": cur_close,
                     "adx": cur_adx, "cci": cur_cci, "dema": cur_dema, "sl": cur_st,
                     "session": session_label(cur_ts)}
    return trades


def load_ui_trades(start: str, end: str):
    with open(ROOT / "ui" / "data" / "trade_events_5m.json") as f:
        data = json.load(f)
    return [t for t in data["trades"] if start <= t["entry_ts"] < end]


def main():
    s, e = "2026-04-29 00:00:00", "2026-05-01 00:00:00"
    py = build_py_trade_list(s, e)
    ui = load_ui_trades(s, e)

    print(f"{'='*100}")
    print(f"{'PYTHON ':>6s} {'UI':>6s}  {'ENTRY TIME':<19s} {'SIDE':>6s}  {'ENTRY':>8s} {'ADX':>5s} {'CCI':>5s} {'EXIT':>19s} {'EXIT PX':>7s} {'REASON':>10s} {'PNL':>6s}")
    print(f"{'='*100}")

    for i in range(max(len(py), len(ui))):
        py_t = py[i] if i < len(py) else None
        ui_t = ui[i] if i < len(ui) else None
        no = i + 1

        if py_t and ui_t:
            ts_match = py_t["ts"][:16] == ui_t["entry_ts"][:16]
            side_match = py_t["side"] == ui_t["side"]
            px_match = abs(py_t["price"] - ui_t["entry_price"]) < 0.2
            ok = ts_match and side_match and px_match
            flag = "✓" if ok else "✗"
        else:
            flag = "?"

        py_entry = f"{py_t['ts'][:16]} {py_t['side']:<5s} {py_t['price']:>7.1f} {py_t['adx']:>4.1f} {py_t['cci']:>4.0f}" if py_t else f"{'MISSING':>42s}"
        ui_entry = f"{ui_t['entry_ts'][:16]} {ui_t['side']:<5s} {ui_t['entry_price']:>7.1f} {ui_t['entry_adx']:>4.1f} {ui_t['entry_cci']:>4.0f}" if ui_t else f"{'MISSING':>42s}"
        py_exit = f"{py_t['exit_ts'][:16]} {py_t['exit_price']:>7.1f} {py_t['exit_reason']:>10s} {py_t['pnl']:>5.0f}" if py_t else ""
        ui_exit = f"{ui_t['exit_ts'][:16]} {ui_t['exit_price']:>7.1f} {ui_t['exit_reason']:>10s} {ui_t['pnl_usd']:>5.0f}" if ui_t else ""

        print(f"  {flag}  {no:>3d}  {py_entry}  |  {ui_entry}  |  py: {py_exit}  |  ui: {ui_exit}")


if __name__ == "__main__":
    main()

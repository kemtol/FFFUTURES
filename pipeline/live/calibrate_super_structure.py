#!/usr/bin/env python3
"""Calibrate Python Super Structure against ui/data/trade_events_super_structure_5m.json."""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import timedelta
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

UI_JSON = ROOT / "ui" / "data" / "trade_events_super_structure_5m.json"
RAW_DB = ROOT / "data" / "Level_0_Raw" / "MGC_1m.db"


def session_name(ts: pd.Timestamp) -> str:
    hour = ts.hour + ts.minute / 60.0
    if 0 <= hour < 3:
        return "Tokyo"
    if 7 <= hour < 10:
        return "London"
    if 13.5 <= hour < 16.5:
        return "US"
    return "Other"


def run_backtest(db_path: Path, start: str, end: str,
                 session_gate: bool = False) -> list[dict]:
    """Run TV strategy on 5m resampled data. Returns list of signal dicts."""
    warmup_start = (pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=120))
    warmup_start_str = warmup_start.strftime("%Y-%m-%d %H:%M:%S")
    end_ts = pd.Timestamp(end, tz="UTC").strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql(
            """
            SELECT timestamp_utc, open, high, low, close
            FROM investing_ohlcv_1m
            WHERE symbol = 'MICRO_GOLD' AND timeframe = '1m'
              AND timestamp_utc >= ? AND timestamp_utc < ?
            ORDER BY epoch_ms
            """,
            conn,
            params=[warmup_start_str, end_ts],
        )
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df["ts"] = df["timestamp_utc"].dt.tz_localize(None)  # strip tz

    df_5m = df.set_index("ts").resample("5min", label="right", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()

    h = df_5m["high"].values.astype(float)
    l = df_5m["low"].values.astype(float)
    c = df_5m["close"].values.astype(float)

    st, direction = supertrend(h, l, c, ST_FACTOR, ATR_PERIOD)
    d = dema(c, DEMA_LENGTH)
    ax = adx(h, l, c, ADX_LENGTH)
    cx = cci(h, l, c, CCI_LENGTH, source=CCI_SOURCE)

    pos = 0  # 1=long, -1=short
    sl_price = 0.0
    signals: list[dict] = []

    for i in range(DEMA_LENGTH + 50, len(c)):
        cur_close = float(c[i])
        cur_dema = float(d[i])
        cur_dir = int(direction[i])
        cur_st = float(st[i]) if not np.isnan(st[i]) else 0.0
        cur_adx = float(ax[i]) if i < len(ax) and not np.isnan(ax[i]) else 0.0
        cur_cci = float(cx[i]) if not np.isnan(cx[i]) else 0.0
        cur_ts = df_5m.index[i]
        cur_high = float(h[i])
        cur_low = float(l[i])

        in_session = session_name(cur_ts) in ("Tokyo", "London", "US")
        adx_ok = cur_adx > ADX_THRESHOLD
        cci_ok_long = cur_cci > CCI_LONG_MIN
        cci_ok_short = cur_cci < CCI_SHORT_MAX

        cross_up = (float(c[i - 1]) < d[i - 1] and cur_close > cur_dema)
        cross_dn = (float(c[i - 1]) > d[i - 1] and cur_close < cur_dema)

        long_signal = adx_ok and cci_ok_long and (
            cross_up or cur_close > cur_dema) and cur_dir < 0
        short_signal = adx_ok and cci_ok_short and (
            cross_dn or cur_close < cur_dema) and cur_dir > 0

        if session_gate:
            long_signal = long_signal and in_session
            short_signal = short_signal and in_session

        # Stop orders
        if pos == 1:
            if cur_low <= sl_price:
                signals.append({
                    "type": "SL", "side": "Long", "ts": str(cur_ts),
                    "price": round(float(sl_price), 2),
                })
                pos = 0
        if pos == -1:
            if cur_high >= sl_price:
                signals.append({
                    "type": "SL", "side": "Short", "ts": str(cur_ts),
                    "price": round(float(sl_price), 2),
                })
                pos = 0

        # Trend flip
        if pos == 1 and cur_dir > 0:
            signals.append({
                "type": "TREND_FLIP", "side": "Long", "ts": str(cur_ts),
                "price": round(cur_close, 2),
            })
            pos = 0
        elif pos == -1 and cur_dir < 0:
            signals.append({
                "type": "TREND_FLIP", "side": "Short", "ts": str(cur_ts),
                "price": round(cur_close, 2),
            })
            pos = 0

        # Session end (only when session_gate=True)
        if session_gate and pos != 0 and not in_session:
            signals.append({
                "type": "SESSION_END", "side": "Long" if pos == 1 else "Short",
                "ts": str(cur_ts), "price": round(cur_close, 2),
            })
            pos = 0

        # Update stop
        if pos != 0:
            sl_price = cur_st

        # Entry
        if long_signal and pos == 0:
            pos = 1
            sl_price = cur_st
            signals.append({
                "type": "ENTRY", "side": "Long", "ts": str(cur_ts),
                "price": round(cur_close, 2), "sl": round(float(cur_st), 2),
                "adx": round(cur_adx, 1), "cci": round(cur_cci, 0),
                "dema": round(cur_dema, 1), "session": session_name(cur_ts),
            })
        elif short_signal and pos == 0:
            pos = -1
            sl_price = cur_st
            signals.append({
                "type": "ENTRY", "side": "Short", "ts": str(cur_ts),
                "price": round(cur_close, 2), "sl": round(float(cur_st), 2),
                "adx": round(cur_adx, 1), "cci": round(cur_cci, 0),
                "dema": round(cur_dema, 1), "session": session_name(cur_ts),
            })

    return signals


def load_ui_trades(start_ts: str, end_ts: str) -> list[dict]:
    with open(UI_JSON) as f:
        data = json.load(f)
    out = []
    for t in data["trades"]:
        if start_ts <= t["entry_ts"] < end_ts:
            out.append(t)
    return sorted(out, key=lambda x: x["entry_ts"])


def _ts_diff(a: str, b: str) -> float:
    ta = pd.Timestamp(a).tz_localize(None)
    tb = pd.Timestamp(b).tz_localize(None)
    return abs((ta - tb).total_seconds())


def build_pairs(py_signals: list[dict], ui_trades: list[dict]) -> list[dict]:
    """Match Python entries to UI trades. ±5 min, ±$5 tolerance."""
    py_entries = [s for s in py_signals if s["type"] == "ENTRY"]
    used: set[int] = set()
    pairs = []

    for tr in ui_trades:
        best = None
        best_j = -1
        best_dt = float("inf")
        for j, pe in enumerate(py_entries):
            if j in used:
                continue
            ts_diff = _ts_diff(pe["ts"], tr["entry_ts"])
            px_diff = abs(pe["price"] - tr["entry_price"])
            if ts_diff <= 300 and px_diff <= 5.0 and ts_diff < best_dt:  # ±5 min
                best = pe
                best_j = j
                best_dt = ts_diff

        if best is None:
            pairs.append({
                "ui_trade": tr["trade_no"],
                "ui_entry": tr["entry_ts"],
                "ui_side": tr["side"],
                "ui_entry_px": tr["entry_price"],
                "match": "MISSING",
                "note": "no Python entry within ±5min, ±$5",
            })
            continue
        used.add(best_j)

        # Find matching exit
        match_entry_idx = py_signals.index(best)
        exit_match = None
        for k in range(match_entry_idx + 1, len(py_signals)):
            s = py_signals[k]
            if s["type"] in ("SL", "TREND_FLIP", "SESSION_END") and s["side"] == tr["side"]:
                exit_match = s
                break

        if exit_match:
            ts_diff = _ts_diff(exit_match["ts"], tr["exit_ts"])
            px_diff = abs(exit_match["price"] - tr["exit_price"])
            if ts_diff <= 300 and px_diff <= 5.0:
                pairs.append({
                    "ui_trade": tr["trade_no"],
                    "ui_entry": tr["entry_ts"][:16],
                    "ui_side": tr["side"],
                    "ui_entry_px": tr["entry_price"],
                    "py_entry": best["ts"][:16],
                    "py_entry_px": best["price"],
                    "ui_exit": tr["exit_ts"][:16],
                    "ui_exit_px": tr["exit_price"],
                    "ui_exit_reason": tr["exit_reason"],
                    "py_exit": exit_match["ts"][:16],
                    "py_exit_px": exit_match["price"],
                    "py_exit_reason": exit_match["type"],
                    "match": "MATCH",
                })
            else:
                pairs.append({
                    "ui_trade": tr["trade_no"],
                    "match": "EXIT_MISMATCH",
                    "ui_exit": tr["exit_ts"][:16],
                    "ui_exit_px": tr["exit_price"],
                    "ui_exit_reason": tr["exit_reason"],
                    "py_exit": exit_match["ts"][:16],
                    "py_exit_px": exit_match["price"],
                    "py_exit_reason": exit_match["type"],
                })
        else:
            pairs.append({
                "ui_trade": tr["trade_no"],
                "match": "NO_EXIT",
                "py_entry": best["ts"][:16],
                "py_entry_px": best["price"],
                "ui_exit": tr["exit_ts"][:16],
            })

    return pairs


def main():
    start_ts = "2026-04-29 00:00:00"
    end_ts = "2026-05-01 00:00:00"

    print("=" * 70)
    print("Calibration: Python TV strategy vs UI trade_events_5m.json")
    print(f"Range: {start_ts} → {end_ts}")
    print("=" * 70)

    for session_gate in [False, True]:
        label = "WITH" if session_gate else "WITHOUT"
        print(f"\n--- {label} session gating ---")

        signals = run_backtest(RAW_DB, start_ts, end_ts, session_gate=session_gate)
        entries = [s for s in signals if s["type"] == "ENTRY"]
        exits = [s for s in signals if s["type"] in ("SL", "TREND_FLIP", "SESSION_END")]
        print(f"Python: {len(entries)} entries, {len(exits)} exits")

        ui_trades = load_ui_trades(start_ts, end_ts)
        print(f"UI:      {len(ui_trades)} trades")

        pairs = build_pairs(signals, ui_trades)

        matches = [p for p in pairs if p["match"] == "MATCH"]
        missing = [p for p in pairs if p["match"] == "MISSING"]
        exit_mismatch = [p for p in pairs if p["match"] == "EXIT_MISMATCH"]
        no_exit = [p for p in pairs if p["match"] == "NO_EXIT"]

        print(f"\nResults: {len(matches)} MATCH, {len(missing)} MISSING, "
              f"{len(exit_mismatch)} EXIT_MISMATCH, {len(no_exit)} NO_EXIT")

        for p in pairs:
            if p["match"] == "MATCH":
                print(f"  #{p['ui_trade']} ✓ MATCH  "
                      f"entry Δpx={abs(p['py_entry_px'] - p['ui_entry_px']):.1f}  "
                      f"exit={p['py_exit_reason']} ui={p['ui_exit_reason']}")
            elif p["match"] == "MISSING":
                print(f"  #{p['ui_trade']} ✗ MISSING  ui={p['ui_entry']} {p['ui_side']} @ {p['ui_entry_px']}")
            elif p["match"] == "EXIT_MISMATCH":
                print(f"  #{p['ui_trade']} ✗ EXIT_MISMATCH  ui={p['ui_exit']}@{p['ui_exit_px']}  py={p['py_exit']}@{p['py_exit_px']}")
            elif p["match"] == "NO_EXIT":
                print(f"  #{p['ui_trade']} ✗ NO_EXIT  py_entry={p['py_entry']}  ui_exit={p['ui_exit']}")

        print()


if __name__ == "__main__":
    main()

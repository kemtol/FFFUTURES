#!/usr/bin/env python3
"""Walkforward Super Structure: --batch (direct loop, ~0.5s) or --incremental (live sim)."""
from __future__ import annotations

import sqlite3, sys, time, shutil, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.live.buffer import DataBuffer
from pipeline.live.signal_bus import SignalBus
from pipeline.live.super_structure import (
    SuperStructure, ADX_LENGTH, ADX_THRESHOLD, ATR_PERIOD, CCI_LENGTH,
    CCI_LONG_MIN, CCI_SHORT_MAX, CCI_SOURCE, DEMA_LENGTH, ST_FACTOR,
    _atr, adx, cci, dema, supertrend,
)

RAW_DB = ROOT / "data" / "Level_0_Raw" / "MGC_1m.db"


def fmt_signal(sig: dict) -> str:
    action = sig.get("action", "?")
    ts = str(sig.get("ts", "?"))[:19]
    price = sig.get("price", 0)
    sl = sig.get("sl", 0)
    pnl = sig.get("pnl", 0)
    if action == "CLOSE":
        emoji = "✅" if pnl >= 0 else "❌"
        return f"[{ts}] {emoji} CLOSE @ ${price:.1f}  PnL: ${pnl:.0f}"
    elif action == "BUY":
        return f"[{ts}] 🟢 BUY   @ ${price:.1f}  SL=${sl:.1f}  ADX={sig.get('adx',0)} CCI={sig.get('cci',0)}"
    elif action == "SELL":
        return f"[{ts}] 🔴 SELL  @ ${price:.1f}  SL=${sl:.1f}  ADX={sig.get('adx',0)} CCI={sig.get('cci',0)}"
    return f"[{ts}] {action} @ ${price:.1f}"


def run_batch(start_ts: str, end_ts: str, send: bool = False):
    """Direct backtest loop — 1 pass, no incremental overhead."""
    warmup = (pd.Timestamp(start_ts, tz="UTC") - pd.Timedelta(days=120))
    warmup_str = warmup.strftime("%Y-%m-%d %H:%M:%S")
    end_str = pd.Timestamp(end_ts, tz="UTC").strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(str(RAW_DB)) as conn:
        df = pd.read_sql(
            """SELECT timestamp_utc, open, high, low, close
               FROM investing_ohlcv_1m
               WHERE symbol='MICRO_GOLD' AND timeframe='1m'
                 AND timestamp_utc >= ? AND timestamp_utc < ?
               ORDER BY epoch_ms""",
            conn, params=[warmup_str, end_str],
        )
    df["ts"] = pd.to_datetime(df["timestamp_utc"])
    df_5m = df.set_index("ts").resample("5min", label="right", closed="left").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last")).dropna()

    h = df_5m["high"].values.astype(float)
    l = df_5m["low"].values.astype(float)
    c = df_5m["close"].values.astype(float)

    st, direction = supertrend(h, l, c, ST_FACTOR, ATR_PERIOD)
    d_arr = dema(c, DEMA_LENGTH)
    ax = adx(h, l, c, ADX_LENGTH)
    cx = cci(h, l, c, CCI_LENGTH, source=CCI_SOURCE)

    pos = 0; sl_price = 0.0; entry_price = 0.0
    bus = SignalBus() if send else None
    sent = 0

    for i in range(DEMA_LENGTH + 50, len(c)):
        cur_close = float(c[i])
        cur_dema = float(d_arr[i])
        cur_dir = int(direction[i])
        cur_st = float(st[i]) if not np.isnan(st[i]) else 0.0
        cur_adx = float(ax[i]) if i < len(ax) and not np.isnan(ax[i]) else 0.0
        cur_cci = float(cx[i]) if not np.isnan(cx[i]) else 0.0
        cur_ts = df_5m.index[i]
        ts_str = str(cur_ts)
        cur_high = float(h[i]); cur_low = float(l[i])

        in_target = start_ts <= ts_str < (end_ts + "+00:00")
        adx_ok = cur_adx > ADX_THRESHOLD
        cci_ok_long = cur_cci > CCI_LONG_MIN
        cci_ok_short = cur_cci < CCI_SHORT_MAX
        cross_up = (float(c[i-1]) < d_arr[i-1] and cur_close > cur_dema)
        cross_dn = (float(c[i-1]) > d_arr[i-1] and cur_close < cur_dema)
        long_sig = adx_ok and cci_ok_long and (cross_up or cur_close > cur_dema) and cur_dir < 0
        short_sig = adx_ok and cci_ok_short and (cross_dn or cur_close < cur_dema) and cur_dir > 0

        if pos == 1 and cur_low <= sl_price:
            if in_target:
                pnl = round((float(sl_price) - entry_price) * 10 - 3, 2)
                pl = {"action": "CLOSE", "price": round(float(sl_price), 2),
                      "ts": ts_str, "pnl": pnl, "sl": 0, "adx": round(cur_adx, 1),
                      "cci": round(cur_cci, 0), "dema": round(cur_dema, 1)}
                print(fmt_signal(pl)); sent += 1
                if bus: bus.publish("super_structure", pl)
            pos = 0
        elif pos == -1 and cur_high >= sl_price:
            if in_target:
                pnl = round((entry_price - float(sl_price)) * 10 - 3, 2)
                pl = {"action": "CLOSE", "price": round(float(sl_price), 2),
                      "ts": ts_str, "pnl": pnl, "sl": 0, "adx": round(cur_adx, 1),
                      "cci": round(cur_cci, 0), "dema": round(cur_dema, 1)}
                print(fmt_signal(pl)); sent += 1
                if bus: bus.publish("super_structure", pl)
            pos = 0

        if pos == 1 and cur_dir > 0:
            if in_target:
                pnl = round((cur_close - entry_price) * 10 - 3, 2)
                pl = {"action": "CLOSE", "price": round(cur_close, 2),
                      "ts": ts_str, "pnl": pnl, "sl": 0, "adx": round(cur_adx, 1),
                      "cci": round(cur_cci, 0), "dema": round(cur_dema, 1)}
                print(fmt_signal(pl)); sent += 1
                if bus: bus.publish("super_structure", pl)
            pos = 0
        elif pos == -1 and cur_dir < 0:
            if in_target:
                pnl = round((entry_price - cur_close) * 10 - 3, 2)
                pl = {"action": "CLOSE", "price": round(cur_close, 2),
                      "ts": ts_str, "pnl": pnl, "sl": 0, "adx": round(cur_adx, 1),
                      "cci": round(cur_cci, 0), "dema": round(cur_dema, 1)}
                print(fmt_signal(pl)); sent += 1
                if bus: bus.publish("super_structure", pl)
            pos = 0

        if pos != 0:
            sl_price = cur_st

        if long_sig and pos == 0:
            pos = 1; sl_price = cur_st; entry_price = cur_close
            if in_target:
                pl = {"action": "BUY", "price": round(cur_close, 2),
                      "sl": round(float(cur_st), 2), "ts": ts_str,
                      "adx": round(cur_adx, 1), "cci": round(cur_cci, 0),
                      "dema": round(cur_dema, 1)}
                print(fmt_signal(pl)); sent += 1
                if bus: bus.publish("super_structure", pl)
        elif short_sig and pos == 0:
            pos = -1; sl_price = cur_st; entry_price = cur_close
            if in_target:
                pl = {"action": "SELL", "price": round(cur_close, 2),
                      "sl": round(float(cur_st), 2), "ts": ts_str,
                      "adx": round(cur_adx, 1), "cci": round(cur_cci, 0),
                      "dema": round(cur_dema, 1)}
                print(fmt_signal(pl)); sent += 1
                if bus: bus.publish("super_structure", pl)

    print(f"\nSignals: {sent}")


def run_incremental(start_ts: str, end_ts: str, send: bool = False):
    """Feed bars one-at-a-time, call ss.check() — simulates live."""
    warmup_start = (pd.Timestamp(start_ts) - pd.Timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")

    tmpdir = tempfile.mkdtemp(prefix="ss_wf_")
    buf_path = Path(tmpdir) / "buffer.db"
    buffer = DataBuffer(db_path=buf_path)

    # Warmup bulk
    with sqlite3.connect(str(RAW_DB)) as src:
        df = pd.read_sql(
            "SELECT epoch_ms, timestamp_utc, open, high, low, close, volume "
            "FROM investing_ohlcv_1m "
            "WHERE symbol='MICRO_GOLD' AND timeframe='1m' "
            "AND timestamp_utc >= ? AND timestamp_utc < ? "
            "ORDER BY epoch_ms",
            src, params=[warmup_start, start_ts],
        )
    df["symbol"] = "MICRO_GOLD"; df["timeframe"] = "1m"
    with sqlite3.connect(str(buf_path)) as conn:
        df.to_sql("ohlcv_1m", conn, if_exists="append", index=False)
    print(f"Warmup: {len(df):,} bars", flush=True)

    # Target bars
    with sqlite3.connect(str(RAW_DB)) as src:
        target = pd.read_sql(
            "SELECT epoch_ms, timestamp_utc, open, high, low, close, volume "
            "FROM investing_ohlcv_1m "
            "WHERE symbol='MICRO_GOLD' AND timeframe='1m' "
            "AND timestamp_utc >= ? AND timestamp_utc <= ? "
            "ORDER BY epoch_ms",
            src, params=[start_ts, end_ts],
        )

    ss = SuperStructure(buffer_or_db_path=buffer)
    bus = SignalBus() if send else None

    print(f"\n{'='*70}")
    print(f"MODE: incremental  |  {start_ts} → {end_ts}  |  {len(target)} bars")
    print(f"{'='*70}\n")

    sent = 0
    t0 = time.perf_counter()
    for _, row in target.iterrows():
        epoch_ms = int(row["epoch_ms"])
        ts = str(row["timestamp_utc"])
        with sqlite3.connect(str(buf_path)) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO ohlcv_1m "
                "(symbol, timeframe, epoch_ms, timestamp_utc, "
                "open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ["MICRO_GOLD", "1m", epoch_ms, ts,
                 float(row["open"]), float(row["high"]), float(row["low"]),
                 float(row["close"]), float(row["volume"])],
            )
            conn.commit()
        now_ts = pd.Timestamp(ts).tz_localize("UTC").to_pydatetime()
        signals = ss.check(now=now_ts) or []
        for wrapper in signals:
            sig = wrapper.get("signal", wrapper)
            print(fmt_signal(sig))
            sent += 1
            if bus:
                bus.publish("super_structure", sig)

    total = time.perf_counter() - t0
    print(f"\nSignals: {sent}  |  Time: {total:.2f}s  |  "
          f"{total/len(target)*1000:.1f}ms per bar")
    shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-04-29 00:00:00")
    ap.add_argument("--end", default="2026-05-01 00:00:00")
    ap.add_argument("--send", action="store_true", help="Send to Telegram")
    ap.add_argument("--incremental", action="store_true",
                    help="Feed bars one-at-a-time (simulates live)")
    args = ap.parse_args()

    if args.incremental:
        run_incremental(args.start, args.end, args.send)
    else:
        t0 = time.perf_counter()
        run_batch(args.start, args.end, args.send)
        print(f"Time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()

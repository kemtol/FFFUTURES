#!/usr/bin/env python3
"""Run backtest (same logic as build_st_trade_events.py) and send to Telegram."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.live.signal_bus import SignalBus
from pipeline.live.tv_strategy import (
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


def run(publish_start: str, publish_end: str, data_start: str = "2026-01-01"):
    warmup = (pd.Timestamp(data_start, tz="UTC") - pd.Timedelta(days=120))
    warmup_str = warmup.strftime("%Y-%m-%d %H:%M:%S")
    end_str = pd.Timestamp(publish_end, tz="UTC").strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(str(RAW_DB)) as conn:
        df = pd.read_sql(
            """SELECT timestamp_utc, open, high, low, close
               FROM investing_ohlcv_1m
               WHERE symbol='MICRO_GOLD' AND timeframe='1m'
                 AND timestamp_utc >= ? AND timestamp_utc < ?
               ORDER BY epoch_ms""",
            conn, params=[warmup_str, end_str],
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
    entry_price = 0.0
    bus = SignalBus()
    trade_count = 0
    entry_price = 0.0
    entry_side = 0  # 1=long, -1=short

    for i in range(DEMA_LENGTH + 50, len(c)):
        cur_close = float(c[i])
        cur_dema = float(d_arr[i])
        cur_dir = int(direction[i])
        cur_st = float(st[i]) if not np.isnan(st[i]) else 0.0
        cur_adx = float(ax[i]) if i < len(ax) and not np.isnan(ax[i]) else 0.0
        cur_cci = float(cx[i]) if not np.isnan(cx[i]) else 0.0
        cur_ts = df_5m.index[i]
        ts_str = str(cur_ts)
        cur_high = float(h[i])
        cur_low = float(l[i])

        adx_ok = cur_adx > ADX_THRESHOLD
        cci_ok_long = cur_cci > CCI_LONG_MIN
        cci_ok_short = cur_cci < CCI_SHORT_MAX
        cross_up = (float(c[i-1]) < d_arr[i-1] and cur_close > cur_dema)
        cross_dn = (float(c[i-1]) > d_arr[i-1] and cur_close < cur_dema)
        long_sig = adx_ok and cci_ok_long and (cross_up or cur_close > cur_dema) and cur_dir < 0
        short_sig = adx_ok and cci_ok_short and (cross_dn or cur_close < cur_dema) and cur_dir > 0

        # Only process events in target range
        in_target = publish_start <= ts_str < (publish_end + "+00:00")

        # Stop hit
        if pos == 1 and cur_low <= sl_price:
            if in_target:
                trade_count += 1
                pnl_calc = round((float(sl_price) - entry_price) * 10 - 3, 2)
                payload = {"action": "CLOSE", "symbol": "MGC", "price": round(float(sl_price), 2),
                           "sl": 0, "ts": ts_str, "pnl": pnl_calc,
                           "adx": round(cur_adx, 1), "cci": round(cur_cci, 0), "dema": round(cur_dema, 1)}
                bus.publish("tv_strategy", payload)
            pos = 0
        elif pos == -1 and cur_high >= sl_price:
            if in_target:
                trade_count += 1
                pnl_calc = round((entry_price - float(sl_price)) * 10 - 3, 2)
                payload = {"action": "CLOSE", "symbol": "MGC", "price": round(float(sl_price), 2),
                           "sl": 0, "ts": ts_str, "pnl": pnl_calc,
                           "adx": round(cur_adx, 1), "cci": round(cur_cci, 0), "dema": round(cur_dema, 1)}
                bus.publish("tv_strategy", payload)
            pos = 0

        # Trend flip
        if pos == 1 and cur_dir > 0:
            if in_target:
                trade_count += 1
                pnl_calc = round((cur_close - entry_price) * 10 - 3, 2)
                payload = {"action": "CLOSE", "symbol": "MGC", "price": round(cur_close, 2),
                           "sl": 0, "ts": ts_str, "pnl": pnl_calc,
                           "adx": round(cur_adx, 1), "cci": round(cur_cci, 0), "dema": round(cur_dema, 1)}
                bus.publish("tv_strategy", payload)
            pos = 0
        elif pos == -1 and cur_dir < 0:
            if in_target:
                trade_count += 1
                pnl_calc = round((entry_price - cur_close) * 10 - 3, 2)
                payload = {"action": "CLOSE", "symbol": "MGC", "price": round(cur_close, 2),
                           "sl": 0, "ts": ts_str, "pnl": pnl_calc,
                           "adx": round(cur_adx, 1), "cci": round(cur_cci, 0), "dema": round(cur_dema, 1)}
                bus.publish("tv_strategy", payload)
            pos = 0

        if pos != 0:
            sl_price = cur_st

        if long_sig and pos == 0:
            pos = 1; sl_price = cur_st; entry_price = cur_close
            if in_target:
                trade_count += 1
                payload = {"action": "BUY", "symbol": "MGC", "price": round(cur_close, 2),
                           "sl": round(float(cur_st), 2), "adx": round(cur_adx, 1),
                           "cci": round(cur_cci, 0), "dema": round(cur_dema, 1),
                           "ts": ts_str}
                bus.publish("tv_strategy", payload)
        elif short_sig and pos == 0:
            pos = -1; sl_price = cur_st; entry_price = cur_close
            if in_target:
                trade_count += 1
                payload = {"action": "SELL", "symbol": "MGC", "price": round(cur_close, 2),
                           "sl": round(float(cur_st), 2), "adx": round(cur_adx, 1),
                           "cci": round(cur_cci, 0), "dema": round(cur_dema, 1),
                           "ts": ts_str}
                bus.publish("tv_strategy", payload)

    print(f"\nTotal signals published: {trade_count}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish-start", default="2026-04-29 00:00:00")
    ap.add_argument("--publish-end", default="2026-05-01 00:00:00")
    ap.add_argument("--data-start", default="2026-01-01",
                    help="Same as build_st_trade_events.py --start")
    args = ap.parse_args()
    run(args.publish_start, args.publish_end, args.data_start)


if __name__ == "__main__":
    main()

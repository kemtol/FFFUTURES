#!/usr/bin/env python3
"""Compare Python TV strategy against TradingView transaction history."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.live.buffer import DataBuffer, CANARY_DB
from pipeline.live.tv_strategy import (
    TVStrategy, supertrend, dema, adx, cci,
    ATR_PERIOD, ST_FACTOR, DEMA_LENGTH, ADX_LENGTH, ADX_THRESHOLD,
    CCI_LENGTH, CCI_LONG_MIN, CCI_SHORT_MAX,
    CCI_SOURCE,
)

TRADES_FILE = ROOT / "data" / "Live" / "tv_trades.json"
TV_LOCAL_UTC_OFFSET_HOURS = 0
WARMUP_DAYS = 90


def load_trades() -> list[dict]:
    with open(TRADES_FILE) as f:
        data = json.load(f)
    return data["trades"]


def tv_local_to_utc(dt_str: str) -> pd.Timestamp:
    """Trade-log timestamps are normalized to UTC."""
    ts = pd.Timestamp(dt_str) - timedelta(hours=TV_LOCAL_UTC_OFFSET_HOURS)
    return ts.tz_localize("UTC")


def signal_name(side: dict) -> str:
    """Support both old `source_signal` and new `signal` trade-log schemas."""
    return str(side.get("signal") or side.get("source_signal") or "")


def is_complete_trade(trade: dict) -> bool:
    entry = trade.get("entry", {})
    exit_ = trade.get("exit", {})
    return bool(entry.get("datetime")) and entry.get("price_usd") is not None and bool(exit_.get("datetime"))


def run_backtest(buffer: DataBuffer, start: str, end: str) -> list[dict]:
    """Run strategy on buffer data (5m resampled) and return all signals."""
    df = buffer.get(start, end).set_index("timestamp_utc").sort_index()
    if len(df) < 50:
        return []

    # Resample 1m → 5m (matches TradingView chart timeframe)
    df = df.resample("5min", label="right", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()

    if len(df) < 50:
        return []

    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)

    st, direction = supertrend(h, l, c, ST_FACTOR, ATR_PERIOD)
    d = dema(c, DEMA_LENGTH)
    ax = adx(h, l, c, ADX_LENGTH)
    cx = cci(h, l, c, CCI_LENGTH, source=CCI_SOURCE)

    pos = 0
    entry_price = 0.0
    sl_price = 0.0
    signals: list[dict] = []

    for i in range(DEMA_LENGTH + 50, len(c)):
        cur_close = float(c[i])
        cur_dema = float(d[i])
        cur_dir = int(direction[i])
        cur_st = float(st[i]) if not np.isnan(st[i]) else 0.0
        adx_idx = i
        cur_adx = float(ax[adx_idx]) if 0 <= adx_idx < len(ax) and not np.isnan(ax[adx_idx]) else 0.0
        cur_cci = float(cx[i]) if not np.isnan(cx[i]) else 0.0
        cur_ts = df.index[i]

        cur_high = float(h[i]); cur_low = float(l[i])

        adx_ok = cur_adx > ADX_THRESHOLD
        cci_ok_long = cur_cci > CCI_LONG_MIN
        cci_ok_short = cur_cci < CCI_SHORT_MAX

        cross_up = (float(c[i - 1]) < d[i - 1] and cur_close > cur_dema)
        cross_dn = (float(c[i - 1]) > d[i - 1] and cur_close < cur_dema)

        long_signal = adx_ok and cci_ok_long and (
            cross_up or cur_close > cur_dema) and cur_dir < 0
        short_signal = adx_ok and cci_ok_short and (
            cross_dn or cur_close < cur_dema) and cur_dir > 0

        # Existing stop orders are active before the current bar's strategy
        # recalculation. If hit, fill at the previously submitted stop.
        if pos == 1:
            if cur_low <= sl_price:
                signals.append({"type": "Long", "action": "EXIT_SL", "ts": cur_ts,
                                "price": float(sl_price)})
                pos = 0
        if pos == -1:
            if cur_high >= sl_price:
                signals.append({"type": "Short", "action": "EXIT_SL", "ts": cur_ts,
                                "price": float(sl_price)})
                pos = 0

        # Trend flip close happens on bar close if the stop did not already fill.
        if pos == 1 and cur_dir > 0:
            signals.append({"type": "Long", "action": "EXIT_TREND_FLIP", "ts": cur_ts,
                            "price": cur_close})
            pos = 0
        elif pos == -1 and cur_dir < 0:
            signals.append({"type": "Short", "action": "EXIT_TREND_FLIP", "ts": cur_ts,
                            "price": cur_close})
            pos = 0

        # Submit/update the dynamic stop for the next bar.
        if pos == 1:
            sl_price = cur_st
        elif pos == -1:
            sl_price = cur_st

        # Entries are market orders processed on bar close. Their stop order is
        # submitted after the fill and becomes active on subsequent bars.
        if long_signal and pos == 0:
            pos = 1
            entry_price = cur_close
            sl_price = cur_st
            signals.append({"type": "Long", "action": "ENTRY", "ts": cur_ts,
                            "price": cur_close, "sl": sl_price})
        if short_signal and pos == 0:
            pos = -1
            entry_price = cur_close
            sl_price = cur_st
            signals.append({"type": "Short", "action": "ENTRY", "ts": cur_ts,
                            "price": cur_close, "sl": sl_price})

    return signals


def match_trades(tv_trades: list[dict], py_signals: list[dict]) -> list[dict]:
    """Match TV trades with Python signals. Tolerate ±10min and ±$10 on entry."""
    results = []
    py_entries = [s for s in py_signals if s["action"] == "ENTRY"]
    used_entries: set[int] = set()

    for trade in tv_trades:
        if not is_complete_trade(trade):
            results.append({
                "trade_no": trade.get("trade_no"),
                "match": "SKIP_INCOMPLETE",
                "note": trade.get("note", "missing entry/exit timestamp or price"),
            })
            continue

        tv_entry_ts = tv_local_to_utc(trade["entry"]["datetime"])
        tv_entry_px = trade["entry"]["price_usd"]
        tv_exit_ts = tv_local_to_utc(trade["exit"]["datetime"])
        tv_exit_px = trade["exit"]["price_usd"]
        tv_type = trade["type"]
        tv_exit_signal = signal_name(trade["exit"])
        is_open_trade = tv_exit_signal.upper() == "OPEN"

        # Find matching Python entry
        matched_entry = None
        matched_entry_idx = -1
        for j, pe in enumerate(py_entries):
            if j in used_entries:
                continue
            if pe["type"] != tv_type:
                continue
            ts_diff = abs((pe["ts"] - tv_entry_ts).total_seconds())
            px_diff = abs(pe["price"] - tv_entry_px)
            if ts_diff <= 600 and px_diff <= 10.0:  # ±10min (2 bars), ±$10
                matched_entry = pe
                matched_entry_idx = j
                break

        if matched_entry is None:
            results.append({"trade_no": trade["trade_no"], "match": "NO_ENTRY",
                            "tv_entry": str(tv_entry_ts), "tv_price": tv_entry_px})
            continue
        used_entries.add(matched_entry_idx)

        if is_open_trade:
            results.append({
                "trade_no": trade["trade_no"],
                "match": "ENTRY_ONLY_OPEN",
                "tv_entry": str(tv_entry_ts)[:16],
                "tv_entry_px": tv_entry_px,
                "py_entry": str(matched_entry["ts"])[:16],
                "py_entry_px": matched_entry["price"],
                "tv_exit": str(tv_exit_ts)[:16],
                "tv_exit_px": tv_exit_px,
                "exit_reason": "OPEN",
                "tv_exit_reason": tv_exit_signal,
            })
            continue

        # Find matching Python exit (next EXIT after this entry's index)
        entry_sig_idx = py_signals.index(matched_entry)
        matched_exit = None
        for k in range(entry_sig_idx + 1, len(py_signals)):
            s = py_signals[k]
            if s["action"].startswith("EXIT") and s["type"] == tv_type:
                ts_diff = abs((s["ts"] - tv_exit_ts).total_seconds())
                px_diff = abs(s["price"] - tv_exit_px)
                if ts_diff <= 600 and px_diff <= 10.0:  # ±10min, ±$10
                    matched_exit = s
                    break

        exit_reason = matched_exit["action"] if matched_exit else "NO_EXIT"
        match_str = "MATCH" if matched_entry and matched_exit else "PARTIAL"
        results.append({
            "trade_no": trade["trade_no"],
            "match": match_str,
            "tv_entry": str(tv_entry_ts)[:16],
            "tv_entry_px": tv_entry_px,
            "py_entry": str(matched_entry["ts"])[:16] if matched_entry else "N/A",
            "py_entry_px": matched_entry["price"] if matched_entry else 0,
            "tv_exit": str(tv_exit_ts)[:16],
            "tv_exit_px": tv_exit_px,
            "py_exit": str(matched_exit["ts"])[:16] if matched_exit else "N/A",
            "py_exit_px": matched_exit["price"] if matched_exit else 0,
            "exit_reason": exit_reason,
            "tv_exit_reason": tv_exit_signal,
        })

    return results


def main():
    buffer = DataBuffer(db_path=CANARY_DB)

    # Load TV trades
    trades = load_trades()
    print(f"Loaded {len(trades)} TV trades")

    # Determine period
    dates = [tv_local_to_utc(t["entry"]["datetime"]) for t in trades if is_complete_trade(t)]
    first_trade_ts = min(dates)
    start = (first_trade_ts - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    end = (max(dates) + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Period: {first_trade_ts.strftime('%Y-%m-%d %H:%M:%S')} → {end} "
          f"(warmup from {start})")

    # Run backtest
    signals = run_backtest(buffer, start, end)
    entries = [s for s in signals if s["action"] == "ENTRY"]
    exits = [s for s in signals if s["action"].startswith("EXIT")]
    print(f"Python signals: {len(entries)} entries, {len(exits)} exits")

    # Match
    results = match_trades(trades, signals)
    matches = [r for r in results if r["match"] == "MATCH"]
    open_entries = [r for r in results if r["match"] == "ENTRY_ONLY_OPEN"]
    partials = [r for r in results if r["match"] == "PARTIAL"]
    no_entries = [r for r in results if r["match"] == "NO_ENTRY"]
    skipped = [r for r in results if r["match"] == "SKIP_INCOMPLETE"]

    print(f"\n{'='*80}")
    comparable = len(trades) - len(skipped)
    print(f"RESULTS: {len(matches)} MATCH, {len(open_entries)} OPEN, "
          f"{len(partials)} PARTIAL, {len(no_entries)} NO_ENTRY, {len(skipped)} SKIP")
    print(f"Match rate: {len(matches)/max(1, comparable)*100:.1f}%")
    print(f"{'='*80}")

    print(f"\nNO_ENTRY ({len(no_entries)}):")
    for r in no_entries:
        print(f"  Trade #{r['trade_no']}: TV entry {r['tv_entry']} @ {r['tv_price']:.1f}")

    print(f"\nPARTIAL ({len(partials)}):")
    for r in partials:
        print(f"  Trade #{r['trade_no']}: TV@{r['tv_entry']} py@{r['py_entry']} "
              f"exit_reason={r['exit_reason']} tv_reason={r['tv_exit_reason']}")

    print(f"\nOPEN ENTRY MATCH ({len(open_entries)}):")
    for r in open_entries:
        print(f"  Trade #{r['trade_no']}: TV@{r['tv_entry']} py@{r['py_entry']} "
              f"entry_delta={abs(float(r['py_entry_px'])-r['tv_entry_px']):.1f}")

    print(f"\nMATCH ({len(matches)}):")
    for r in matches:
        print(f"  Trade #{r['trade_no']}: entry Δ={abs(float(r['py_entry_px'])-r['tv_entry_px']):.1f} "
              f"exit={r['exit_reason']} tv={r['tv_exit_reason']}")

    # Save results
    out = {
        "match_rate": len(matches) / max(1, comparable),
        "comparable_trades": comparable,
        "details": results,
    }
    out_path = ROOT / "model" / "CALIBRATION" / "tv_strategy_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

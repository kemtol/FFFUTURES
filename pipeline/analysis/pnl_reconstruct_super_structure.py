#!/usr/bin/env python3
"""
Reconstruct PnL from Super Structure signal archive with dynamic-SL-trail awareness.

The strategy emits a fresh BUY/SELL signal every time the dynamic SL trails up,
so consecutive same-side signals are NOT new trades — they are SL-trail updates
on the same position. This script collapses runs of consecutive same-side signals
into a single position, tracks the latest SL, and exits the position when it
encounters a CLOSE or opposite-side signal.

Outputs a per-day rollup plus per-trade detail.

Optional: --use-ledger-entry substitutes the actual Topstep ledger fill price
for the entry, where the ledger has a matching MARKET event within 90s of the
signal. Falls back to signal price otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SIGNALS_PATH = ROOT / "data" / "Live" / "super_structure_signals.json"
LEDGER_PATH = ROOT / "data" / "Live" / "super_structure_executions.jsonl"

POINT_VALUE = 10.0
COMMISSION_RT = 1.74
TZ_DEFAULT = "Asia/Jakarta"
LEDGER_MATCH_WINDOW_S = 90


def load_signals() -> list[dict]:
    raw = json.load(open(SIGNALS_PATH))
    out = []
    for s in raw:
        sig = s["signal"]
        ts_raw = sig.get("ts") or s["received_at"]
        px = sig.get("price") if sig.get("price") is not None else sig.get("entry")
        if px is None:
            continue
        out.append({
            "ts": pd.to_datetime(ts_raw, utc=True),
            "action": (sig.get("action") or "").upper(),
            "price": float(px),
            "sl": float(sig["sl"]) if sig.get("sl") is not None else None,
            "signal_id": sig.get("signal_id") or "",
        })
    out.sort(key=lambda x: x["ts"])
    return out


def load_ledger_entries() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    out = []
    for line in open(LEDGER_PATH):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("event_type") != "MARKET":
            continue
        if r.get("executed_price") is None:
            continue
        out.append({
            "ts": pd.to_datetime(r["timestamp_utc"], utc=True),
            "side": "L" if r.get("action") == "BUY" else "S",
            "exec_px": float(r["executed_price"]),
            "signal_id": r.get("signal_id") or "",
        })
    out.sort(key=lambda x: x["ts"])
    return out


def reconstruct_trades(signals: list[dict]) -> list[dict]:
    """
    Walk the signal stream chronologically.
    Open trade when first BUY/SELL appears (or after a CLOSE).
    Same-side subsequent signals = SL-trail updates (latest_sl tracked, latest_signal_ts updated).
    Different-side signal = flip: close current, open new at flip price.
    CLOSE = exit current at CLOSE price; no new position.
    """
    trades = []
    open_t = None

    def close_open(exit_ts, exit_price, exit_reason):
        nonlocal open_t
        if open_t is None:
            return
        open_t["exit_ts"] = exit_ts
        open_t["exit_price"] = exit_price
        open_t["exit_reason"] = exit_reason
        trades.append(open_t)
        open_t = None

    for s in signals:
        a = s["action"]
        if a == "CLOSE":
            close_open(s["ts"], s["price"], "CLOSE")
            continue
        if a not in ("BUY", "SELL"):
            continue
        new_side = "L" if a == "BUY" else "S"
        if open_t is None:
            open_t = {
                "side": new_side,
                "entry_ts": s["ts"],
                "entry_price": s["price"],
                "entry_sl": s["sl"],
                "trail_updates": 0,
                "latest_sl": s["sl"],
                "latest_trail_ts": s["ts"],
                "latest_trail_price": s["price"],
            }
            continue
        if new_side == open_t["side"]:
            # SL-trail update on the same position
            open_t["trail_updates"] += 1
            if s["sl"] is not None:
                open_t["latest_sl"] = s["sl"]
            open_t["latest_trail_ts"] = s["ts"]
            open_t["latest_trail_price"] = s["price"]
            continue
        # Opposite side: flip — close at flip price, open new at flip price
        flip_ts, flip_px = s["ts"], s["price"]
        close_open(flip_ts, flip_px, "FLIP")
        open_t = {
            "side": new_side,
            "entry_ts": flip_ts,
            "entry_price": flip_px,
            "entry_sl": s["sl"],
            "trail_updates": 0,
            "latest_sl": s["sl"],
            "latest_trail_ts": flip_ts,
            "latest_trail_price": flip_px,
        }

    if open_t is not None:
        open_t["exit_ts"] = None
        open_t["exit_price"] = None
        open_t["exit_reason"] = "OPEN"
        trades.append(open_t)

    return trades


def attach_ledger_entry(trades: list[dict], ledger: list[dict]) -> None:
    """For each trade, find a ledger MARKET fill within window_s of entry_ts and same side."""
    for t in trades:
        side = t["side"]
        entry_ts = t["entry_ts"]
        best = None
        best_dt = None
        for l in ledger:
            if l["side"] != side:
                continue
            dt = abs((l["ts"] - entry_ts).total_seconds())
            if dt > LEDGER_MATCH_WINDOW_S:
                continue
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = l
        t["ledger_exec_px"] = best["exec_px"] if best else None


def trade_pnl(t: dict, use_ledger: bool) -> tuple[float | None, float | None, float | None]:
    if t["exit_price"] is None:
        return None, None, None
    entry = t["ledger_exec_px"] if (use_ledger and t.get("ledger_exec_px") is not None) else t["entry_price"]
    mult = 1 if t["side"] == "L" else -1
    pts = (t["exit_price"] - entry) * mult
    gross = pts * POINT_VALUE
    net = gross - COMMISSION_RT
    return pts, gross, net


def fmt_ts(ts, tz: ZoneInfo) -> str:
    if ts is None:
        return "-"
    return ts.tz_convert(tz).strftime("%m-%d %H:%M")


def render(trades: list[dict], tz_name: str, use_ledger: bool, days: int | None,
           start_date: str | None) -> str:
    tz = ZoneInfo(tz_name)
    today_local = datetime.now(tz).date()
    if start_date:
        first_day = datetime.fromisoformat(start_date).date()
    elif days is not None:
        first_day = today_local - timedelta(days=days - 1)
    else:
        first_day = None

    def in_window(t):
        if first_day is None:
            return True
        d = t["entry_ts"].tz_convert(tz).date()
        return d >= first_day

    trades = [t for t in trades if in_window(t)]
    if not trades:
        return "(no trades in window)"

    # Per-trade detail
    lines = []
    entry_label = "LedgerPx" if use_ledger else "EntryPx"
    lines.append(
        f"{'Day':<10} {'Side':<4} {'Entry':<12} {entry_label:>9} {'Trail SL':>9} "
        f"{'#tr':>4} {'Exit':<12} {'ExitPx':>9} {'Why':<5} {'Pts':>7} {'Net $':>9}"
    )
    lines.append("-" * 100)
    by_day = {}
    for t in trades:
        d = t["entry_ts"].tz_convert(tz).date()
        pts, gross, net = trade_pnl(t, use_ledger)
        entry_px_used = t["ledger_exec_px"] if (use_ledger and t.get("ledger_exec_px") is not None) else t["entry_price"]
        sl_str = f"{t['latest_sl']:.2f}" if t.get("latest_sl") is not None else "-"
        ledger_flag = "*" if (use_ledger and t.get("ledger_exec_px") is not None) else ""
        if t["exit_price"] is None:
            lines.append(
                f"{str(d):<10} {t['side']:<4} {fmt_ts(t['entry_ts'], tz):<12} "
                f"{entry_px_used:>8.2f}{ledger_flag:<1} {sl_str:>9} {t['trail_updates']:>4} "
                f"{'(open)':<12} {'-':>9} {'-':<5} {'-':>7} {'-':>9}"
            )
        else:
            lines.append(
                f"{str(d):<10} {t['side']:<4} {fmt_ts(t['entry_ts'], tz):<12} "
                f"{entry_px_used:>8.2f}{ledger_flag:<1} {sl_str:>9} {t['trail_updates']:>4} "
                f"{fmt_ts(t['exit_ts'], tz):<12} {t['exit_price']:>9.2f} "
                f"{t['exit_reason']:<5} {pts:>+7.2f} {net:>+9.2f}"
            )
        by_day.setdefault(str(d), []).append((t, pts, gross, net))

    if use_ledger:
        lines.append("\n* = entry price from Topstep ledger; otherwise signal price.")

    # Per-day rollup
    lines.append("\n=== Per-day rollup ===")
    lines.append(f"{'Day':<12} {'Trades':>6} {'W':>3} {'L':>3} {'Open':>5} {'Gross $':>10} {'Net $':>10}")
    lines.append("-" * 60)
    grand_g = grand_n = 0.0
    grand_w = grand_l = grand_o = grand_t = 0
    for d in sorted(by_day):
        rows = by_day[d]
        closed = [r for r in rows if r[3] is not None]
        opn = len(rows) - len(closed)
        w = sum(1 for r in closed if r[3] > 0)
        l = sum(1 for r in closed if r[3] <= 0)
        g = sum(r[2] for r in closed) if closed else 0.0
        n = sum(r[3] for r in closed) if closed else 0.0
        grand_g += g; grand_n += n
        grand_w += w; grand_l += l; grand_o += opn; grand_t += len(rows)
        lines.append(f"{d:<12} {len(rows):>6} {w:>3} {l:>3} {opn:>5} {g:>+10.2f} {n:>+10.2f}")
    lines.append("-" * 60)
    lines.append(f"{'TOTAL':<12} {grand_t:>6} {grand_w:>3} {grand_l:>3} {grand_o:>5} {grand_g:>+10.2f} {grand_n:>+10.2f}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=2,
                   help="Trailing days to include (Asia/Jakarta), default 2.")
    p.add_argument("--start", default=None,
                   help="Start date YYYY-MM-DD (overrides --days).")
    p.add_argument("--all", action="store_true", help="Include the entire signal archive.")
    p.add_argument("--tz", default=TZ_DEFAULT, choices=["Asia/Jakarta", "UTC"])
    p.add_argument("--use-ledger-entry", action="store_true",
                   help="Substitute Topstep ledger fill price for entry where available.")
    args = p.parse_args()

    signals = load_signals()
    trades = reconstruct_trades(signals)

    if args.use_ledger_entry:
        ledger = load_ledger_entries()
        attach_ledger_entry(trades, ledger)
    else:
        for t in trades:
            t["ledger_exec_px"] = None

    days = None if args.all else args.days
    out = render(trades, args.tz, args.use_ledger_entry, days, args.start)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

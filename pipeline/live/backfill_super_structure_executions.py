#!/usr/bin/env python3
"""Backfill Super Structure execution ledger from legacy text logs.

This is a one-time bridge for executions that happened before
super_structure_executions.jsonl existed. New truth remains the JSONL ledger.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.live.execute.super_structure_executor import EXECUTIONS_PATH
from pipeline.live.parity_super_structure import load_signals

LOG_PATH = ROOT / "data" / "Live" / "super_structure.log"

MARKET_RE = re.compile(
    r"^\[SSExec\] MARKET (?P<side>Buy|Sell) @ (?P<requested>[0-9.]+): (?P<body>\{.*\})$"
)
FLATTEN_RE = re.compile(
    r"^\[SSExec\] FLATTEN @ market PnL=(?P<pnl>[+-]?[0-9]+): (?P<body>\{.*\})$"
)
RECONCILE_FLAT_RE = re.compile(
    r"^\[SSExec\] Reconcile: was (?P<side>Long|Short)@(?P<entry>[0-9.]+) \S+ now FLAT@0\.0"
)


def read_existing_keys() -> set[tuple[str, str, str, str]]:
    keys = set()
    if not EXECUTIONS_PATH.exists():
        return keys
    for line in EXECUTIONS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.add((
            str(row.get("event_type", "")),
            str(row.get("action", "")),
            str(row.get("order_id", "")),
            str(row.get("trade_id", "")),
        ))
    return keys


def nearest_signal(signals: list[dict], action: str, requested: float | None,
                   used: set[int], start_idx: int = 0) -> tuple[int | None, dict | None]:
    best_i = None
    best = None
    best_delta = 999.0
    for i, sig in enumerate(signals):
        if i in used or i < start_idx or sig["action"] != action:
            continue
        delta = 0.0 if requested is None else abs(float(sig["price"]) - requested)
        if delta <= 3.0 and delta < best_delta:
            best_i = i
            best = sig
            best_delta = delta
    return best_i, best


def append_rows(rows: list[dict]) -> int:
    existing = read_existing_keys()
    EXECUTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with EXECUTIONS_PATH.open("a") as f:
        for row in rows:
            key = (
                str(row.get("event_type", "")),
                str(row.get("action", "")),
                str(row.get("order_id", "")),
                str(row.get("trade_id", "")),
            )
            if key in existing:
                continue
            f.write(json.dumps(row, default=str, separators=(",", ":")) + "\n")
            existing.add(key)
            written += 1
    return written


def main() -> int:
    if not LOG_PATH.exists():
        print(f"missing log: {LOG_PATH}")
        return 2

    signals = load_signals()
    used_signals: set[int] = set()
    rows: list[dict] = []
    open_trade: dict | None = None
    last_signal_idx = 0

    for line_no, line in enumerate(LOG_PATH.read_text().splitlines(), start=1):
        m = MARKET_RE.match(line)
        if m:
            side_word = m.group("side")
            action = "BUY" if side_word == "Buy" else "SELL"
            requested = float(m.group("requested"))
            try:
                body = json.loads(m.group("body"))
            except json.JSONDecodeError:
                body = {"raw": m.group("body")}
            sig_i, sig = nearest_signal(signals, action, requested, used_signals, last_signal_idx)
            if sig is None:
                continue
            used_signals.add(sig_i)
            last_signal_idx = sig_i + 1
            open_trade = sig
            rows.append({
                "timestamp_utc": sig["ts"].isoformat() if sig.get("ts") is not None else sig["received_at"].isoformat(),
                "signal_id": sig["signal_id"],
                "trade_id": sig["trade_id"],
                "event_type": "MARKET",
                "action": action,
                "requested_price": requested,
                "executed_price": body.get("executedPrice"),
                "order_id": body.get("orderId"),
                "api_result": {"backfilled_from_log": str(LOG_PATH), "log_line": line_no, **body},
                "error": "",
            })
            continue

        m = FLATTEN_RE.match(line)
        if m:
            try:
                body = json.loads(m.group("body"))
            except json.JSONDecodeError:
                body = {"raw": m.group("body")}
            sig_i, sig = nearest_signal(signals, "CLOSE", None, used_signals, last_signal_idx)
            if sig is None and open_trade is not None:
                sig = {
                    "signal_id": "",
                    "trade_id": open_trade["trade_id"],
                    "received_at": open_trade["received_at"],
                    "ts": open_trade["ts"],
                    "price": None,
                }
            elif sig is not None:
                used_signals.add(sig_i)
                last_signal_idx = sig_i + 1
            rows.append({
                "timestamp_utc": sig["ts"].isoformat() if sig.get("ts") is not None else sig["received_at"].isoformat(),
                "signal_id": sig.get("signal_id", ""),
                "trade_id": sig.get("trade_id", ""),
                "event_type": "FLATTEN",
                "action": "CLOSE",
                "requested_price": sig.get("price"),
                "executed_price": None,
                "order_id": None,
                "api_result": {"backfilled_from_log": str(LOG_PATH), "log_line": line_no, **body},
                "error": "",
            })
            open_trade = None
            continue

        if RECONCILE_FLAT_RE.match(line) and open_trade is not None:
            # Exchange went flat without a CLOSE signal/flatten log for this
            # open trade. Treat it as user/manual intervention.
            rows.append({
                "timestamp_utc": open_trade["ts"].isoformat() if open_trade.get("ts") is not None else open_trade["received_at"].isoformat(),
                "signal_id": "",
                "trade_id": open_trade["trade_id"],
                "event_type": "MANUAL_CLOSE",
                "action": "CLOSE",
                "requested_price": None,
                "executed_price": None,
                "order_id": None,
                "api_result": {"backfilled_from_log": str(LOG_PATH), "log_line": line_no, "message": line},
                "error": "",
            })
            open_trade = None

    written = append_rows(rows)
    print(f"parsed={len(rows)} written={written} ledger={EXECUTIONS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

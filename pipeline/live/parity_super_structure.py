#!/usr/bin/env python3
"""3-way Super Structure parity: UI theoretical vs signals vs Topstep ledger."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.live.super_structure import derive_signal_id, derive_trade_id_from_entry

SIGNALS_PATH = ROOT / "data" / "Live" / "super_structure_signals.json"
EXECUTIONS_PATH = ROOT / "data" / "Live" / "super_structure_executions.jsonl"
UI_PATH = ROOT / "ui" / "data" / "trade_events_super_structure_5m.json"
OUT_DIR = ROOT / "data" / "Live" / "parity"
TELEGRAM_ENV_PATH = ROOT / "data" / "Live" / "telegram.env"
TELEGRAM_STATE_PATH = OUT_DIR / ".last_telegram_state.json"
POINT_VALUE = 10.0
COMMISSION_RT = 1.74
ENTRY_TOLERANCE = pd.Timedelta(minutes=5)


def parse_ts(value, default_tz: str = "UTC") -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize(default_tz)
        return ts.tz_convert("UTC")
    except Exception:
        return None


def price(sig: dict) -> float:
    try:
        return float(sig.get("price", sig.get("entry", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def fmt_ts(ts: pd.Timestamp | None) -> str:
    if ts is None:
        return ""
    return ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M")


def fmt_num(value, digits: int = 1) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def side_mult(side: str) -> int:
    return 1 if side == "Long" else -1


def in_window(ts: pd.Timestamp | None, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return ts is not None and start <= ts < end


def date_window(date_s: str | None, tz_name: str) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    zone = ZoneInfo(tz_name)
    if date_s:
        local_day = pd.Timestamp(date_s).date()
    else:
        local_day = datetime.now(zone).date()
    start_local = pd.Timestamp(local_day).tz_localize(zone)
    end_local = start_local + pd.Timedelta(days=1)
    return (
        str(local_day),
        start_local.tz_convert("UTC"),
        end_local.tz_convert("UTC"),
    )


def load_signals() -> list[dict]:
    if not SIGNALS_PATH.exists():
        return []
    raw = json.loads(SIGNALS_PATH.read_text())
    out: list[dict] = []
    current_trade_id = ""
    for seq, entry in enumerate(raw):
        sig = entry.get("signal", {})
        action = str(sig.get("action", ""))
        received_at = str(entry.get("received_at", ""))
        sig_id = str(sig.get("signal_id") or entry.get("signal_id") or derive_signal_id(sig, received_at, seq))
        if action in ("BUY", "SELL"):
            trade_id = str(sig.get("trade_id") or entry.get("trade_id") or derive_trade_id_from_entry(sig, received_at, seq))
            current_trade_id = trade_id
        elif action == "CLOSE":
            trade_id = str(sig.get("trade_id") or entry.get("trade_id") or current_trade_id)
            current_trade_id = ""
        else:
            trade_id = str(sig.get("trade_id") or entry.get("trade_id") or "")
        out.append({
            "signal_id": sig_id,
            "trade_id": trade_id,
            "action": action,
            "side": "Long" if action == "BUY" else ("Short" if action == "SELL" else ""),
            "ts": parse_ts(sig.get("ts") or received_at),
            "received_at": parse_ts(received_at),
            "price": price(sig),
            "reason": sig.get("reason", ""),
            "raw": sig,
        })
    return out


def load_executions() -> list[dict]:
    if not EXECUTIONS_PATH.exists():
        return []
    out: list[dict] = []
    for line_no, line in enumerate(EXECUTIONS_PATH.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            out.append({"line_no": line_no, "error": f"json_decode: {exc}", "ts": None})
            continue
        row["line_no"] = line_no
        row["ts"] = parse_ts(row.get("timestamp_utc"))
        out.append(row)
    return out


def load_ui_trades() -> list[dict]:
    if not UI_PATH.exists():
        return []
    data = json.loads(UI_PATH.read_text())
    out = []
    for tr in data.get("trades", []):
        row = dict(tr)
        row["entry_dt"] = parse_ts(row.get("entry_ts"), default_tz="UTC")
        row["exit_dt"] = parse_ts(row.get("exit_ts"), default_tz="UTC")
        row["status"] = row.get("status", "CLOSED")
        out.append(row)
    for tr in data.get("open_trades", []):
        row = dict(tr)
        row["entry_dt"] = parse_ts(row.get("entry_ts"), default_tz="UTC")
        row["exit_dt"] = None
        row["status"] = "OPEN"
        out.append(row)
    return out


def is_entry_signal(sig: dict) -> bool:
    return sig.get("action") in ("BUY", "SELL")


def is_market_entry(ex: dict) -> bool:
    return ex.get("event_type") == "MARKET" and str(ex.get("action", "")).upper() in ("BUY", "SELL")


def signal_side(sig: dict) -> str:
    return "Long" if sig.get("action") == "BUY" else "Short"


def execution_side(ex: dict) -> str:
    return "Long" if str(ex.get("action", "")).upper() == "BUY" else "Short"


def execution_error_text(ex: dict) -> str:
    if ex.get("error"):
        return str(ex.get("error"))
    api = ex.get("api_result")
    if isinstance(api, dict):
        if api.get("errorMessage"):
            return str(api.get("errorMessage"))
        result = api.get("result")
        if result not in (None, 0):
            return f"api_result={result}"
    return ""


def execution_entry_ok(ex: dict) -> bool:
    if not is_market_entry(ex):
        return False
    if execution_error_text(ex):
        return False
    api = ex.get("api_result")
    if isinstance(api, dict) and api.get("result") not in (None, 0):
        return False
    return ex.get("executed_price") is not None


def match_execution_entry(sig: dict, executions: list[dict], used: set[int]) -> tuple[int | None, dict | None]:
    candidates = []
    for i, ex in enumerate(executions):
        if i in used or not is_market_entry(ex):
            continue
        if execution_side(ex) != sig["side"]:
            continue
        same_signal = sig.get("signal_id") and ex.get("signal_id") == sig.get("signal_id")
        same_trade = sig.get("trade_id") and ex.get("trade_id") == sig.get("trade_id")
        if same_signal or same_trade:
            candidates.append((i, ex))
    return candidates[0] if candidates else (None, None)


def match_ui_entry(side: str, entry_ts: pd.Timestamp | None, ui_trades: list[dict], used: set[int]) -> tuple[int | None, dict | None]:
    if entry_ts is None:
        return None, None
    best_i = None
    best = None
    best_dt = ENTRY_TOLERANCE + pd.Timedelta(seconds=1)
    for i, tr in enumerate(ui_trades):
        if i in used or tr.get("side") != side or tr.get("entry_dt") is None:
            continue
        dt = abs(tr["entry_dt"] - entry_ts)
        if dt <= ENTRY_TOLERANCE and dt < best_dt:
            best_i = i
            best = tr
            best_dt = dt
    return best_i, best


def overlapping_ui_trade(side: str, ts: pd.Timestamp | None, ui_trades: list[dict]) -> dict | None:
    if ts is None:
        return None
    for tr in ui_trades:
        if tr.get("side") != side or tr.get("entry_dt") is None:
            continue
        exit_dt = tr.get("exit_dt")
        if tr["entry_dt"] <= ts and (exit_dt is None or ts <= exit_dt):
            return tr
    return None


def signal_entry_vs_topstep(signals: list[dict], executions: list[dict]) -> list[dict]:
    rows = []
    used_exec: set[int] = set()
    signal_ids = {s.get("signal_id", "") for s in signals if is_entry_signal(s)}
    trade_ids = {s.get("trade_id", "") for s in signals if is_entry_signal(s)}

    for sig in [s for s in signals if is_entry_signal(s)]:
        match_i, match = match_execution_entry(sig, executions, used_exec)
        if match_i is not None:
            used_exec.add(match_i)
        if match is None:
            rows.append({
                "severity": "CRITICAL",
                "signal_id": sig["signal_id"],
                "trade_id": sig["trade_id"],
                "side": sig["side"],
                "signal_entry": fmt_ts(sig["ts"]),
                "signal_px": sig["price"],
                "topstep_entry": "MISSING",
                "topstep_px": None,
                "slippage": None,
                "drift_type": "MISSING_ENTRY_EXECUTION",
                "note": "entry signal has no matching Topstep MARKET event",
            })
            continue
        ok = execution_entry_ok(match)
        topstep_px = match.get("executed_price")
        rows.append({
            "severity": "PASS" if ok else "CRITICAL",
            "signal_id": sig["signal_id"],
            "trade_id": sig["trade_id"],
            "side": sig["side"],
            "signal_entry": fmt_ts(sig["ts"]),
            "signal_px": sig["price"],
            "topstep_entry": fmt_ts(match.get("ts")),
            "topstep_px": topstep_px,
            "slippage": (float(topstep_px) - sig["price"]) if topstep_px is not None else None,
            "drift_type": "" if ok else "ENTRY_REJECTED",
            "note": "" if ok else execution_error_text(match) or "Topstep MARKET event did not confirm a fill",
        })

    for i, ex in enumerate(executions):
        if i in used_exec or not is_market_entry(ex):
            continue
        if ex.get("signal_id") not in signal_ids and ex.get("trade_id") not in trade_ids:
            rows.append({
                "severity": "CRITICAL",
                "signal_id": ex.get("signal_id", ""),
                "trade_id": ex.get("trade_id", ""),
                "side": execution_side(ex),
                "signal_entry": "MISSING",
                "signal_px": None,
                "topstep_entry": fmt_ts(ex.get("ts")),
                "topstep_px": ex.get("executed_price"),
                "slippage": None,
                "drift_type": "EXTRA_ENTRY_EXECUTION",
                "note": "Topstep MARKET entry has no matching signal entry",
            })
    return rows


def signal_entry_vs_ui(signals: list[dict], ui_trades: list[dict]) -> list[dict]:
    rows = []
    used_ui: set[int] = set()
    for sig in [s for s in signals if is_entry_signal(s)]:
        ui_i, ui = match_ui_entry(sig["side"], sig["ts"], ui_trades, used_ui)
        if ui_i is not None:
            used_ui.add(ui_i)
        if ui is None:
            overlap = overlapping_ui_trade(sig["side"], sig["ts"], ui_trades)
            if overlap:
                note = (
                    "UI theoretical was already in same-side trade "
                    f"{fmt_ts(overlap.get('entry_dt'))}->{fmt_ts(overlap.get('exit_dt')) or 'open'}"
                )
                drift_type = "UI_ALREADY_IN_POSITION"
                ui_entry = fmt_ts(overlap.get("entry_dt"))
                ui_exit = fmt_ts(overlap.get("exit_dt")) if overlap.get("exit_dt") is not None else ""
            else:
                note = f"no UI theoretical entry within {int(ENTRY_TOLERANCE.total_seconds() // 60)}min"
                drift_type = "MISSING_UI_ENTRY"
                ui_entry = "MISSING"
                ui_exit = ""
            rows.append({
                "severity": "CRITICAL",
                "signal_id": sig["signal_id"],
                "trade_id": sig["trade_id"],
                "side": sig["side"],
                "signal_entry": fmt_ts(sig["ts"]),
                "signal_px": sig["price"],
                "ui_entry": ui_entry,
                "ui_exit": ui_exit,
                "ui_status": overlap.get("status", "") if overlap else "",
                "entry_delta_min": None,
                "entry_px_delta": None,
                "drift_type": drift_type,
                "note": note,
            })
            continue
        rows.append({
            "severity": "PASS",
            "signal_id": sig["signal_id"],
            "trade_id": sig["trade_id"],
            "side": sig["side"],
            "signal_entry": fmt_ts(sig["ts"]),
            "signal_px": sig["price"],
            "ui_entry": fmt_ts(ui["entry_dt"]),
            "ui_exit": fmt_ts(ui["exit_dt"]) if ui.get("exit_dt") is not None else "",
            "ui_status": ui.get("status", "CLOSED"),
            "entry_delta_min": (sig["ts"] - ui["entry_dt"]).total_seconds() / 60.0,
            "entry_px_delta": sig["price"] - float(ui.get("entry_price", 0.0)),
            "drift_type": "",
            "note": "UI entry matched",
        })
    return rows


def build_signal_trades(signals: list[dict]) -> list[dict]:
    trades: list[dict] = []
    open_trade: dict | None = None
    for sig in signals:
        if sig["action"] in ("BUY", "SELL"):
            if open_trade is not None:
                trades.append(open_trade)
            open_trade = {
                "trade_id": sig["trade_id"],
                "side": sig["side"],
                "entry_signal": sig,
                "exit_signal": None,
            }
        elif sig["action"] == "CLOSE":
            if open_trade is None:
                trades.append({
                    "trade_id": sig["trade_id"],
                    "side": "",
                    "entry_signal": None,
                    "exit_signal": sig,
                })
            else:
                open_trade["exit_signal"] = sig
                trades.append(open_trade)
                open_trade = None
    if open_trade is not None:
        trades.append(open_trade)
    return trades


def build_execution_trades(executions: list[dict]) -> list[dict]:
    trades: list[dict] = []
    open_trade: dict | None = None
    for ex in executions:
        if ex.get("error"):
            continue
        typ = ex.get("event_type")
        action = str(ex.get("action", "")).upper()
        if typ == "MARKET" and action in ("BUY", "SELL"):
            if open_trade is not None:
                trades.append(open_trade)
            open_trade = {
                "trade_id": ex.get("trade_id", ""),
                "side": "Long" if action == "BUY" else "Short",
                "entry_execution": ex,
                "exit_execution": None,
            }
        elif typ in ("FLATTEN", "MANUAL_CLOSE"):
            if open_trade is None:
                trades.append({
                    "trade_id": ex.get("trade_id", ""),
                    "side": "",
                    "entry_execution": None,
                    "exit_execution": ex,
                })
            else:
                open_trade["exit_execution"] = ex
                trades.append(open_trade)
                open_trade = None
    if open_trade is not None:
        trades.append(open_trade)
    return trades


def signal_vs_topstep(signals: list[dict], executions: list[dict]) -> list[dict]:
    rows = []
    used_exec: set[int] = set()
    for sig in signals:
        expected = "MARKET" if sig["action"] in ("BUY", "SELL") else "FLATTEN"
        if sig["action"] not in ("BUY", "SELL", "CLOSE"):
            continue
        candidates = [
            (i, ex) for i, ex in enumerate(executions)
            if i not in used_exec
            and ex.get("event_type") == expected
            and not ex.get("error")
            and (
                (sig["signal_id"] and ex.get("signal_id") == sig["signal_id"])
                or (sig["trade_id"] and ex.get("trade_id") == sig["trade_id"] and expected == "FLATTEN")
            )
        ]
        match_i, match = candidates[0] if candidates else (None, None)
        if match_i is not None:
            used_exec.add(match_i)
        executed = match.get("executed_price") if match else None
        requested = sig["price"]
        if executed is None and match:
            executed = match.get("requested_price")
        rows.append({
            "severity": "PASS" if match else "CRITICAL",
            "signal_id": sig["signal_id"],
            "trade_id": sig["trade_id"],
            "signal": sig["action"],
            "signal_ts": fmt_ts(sig["ts"]),
            "signal_px": requested,
            "execution": match.get("event_type", "") if match else "MISSING",
            "exec_ts": fmt_ts(match.get("ts")) if match else "",
            "exec_px": executed,
            "slippage": (float(executed) - requested) if executed is not None else None,
            "note": "" if match else f"missing {expected} execution",
        })

    signal_ids = {s["signal_id"] for s in signals}
    for i, ex in enumerate(executions):
        if i in used_exec or ex.get("error"):
            continue
        if ex.get("event_type") in ("MARKET", "FLATTEN") and ex.get("signal_id") not in signal_ids:
            rows.append({
                "severity": "CRITICAL",
                "signal_id": ex.get("signal_id", ""),
                "trade_id": ex.get("trade_id", ""),
                "signal": "EXTRA_EXECUTION",
                "signal_ts": "",
                "signal_px": None,
                "execution": ex.get("event_type", ""),
                "exec_ts": fmt_ts(ex.get("ts")),
                "exec_px": ex.get("executed_price") or ex.get("requested_price"),
                "slippage": None,
                "note": "execution without matching signal",
            })
    return rows


def match_ui_trade(side: str, entry_ts: pd.Timestamp | None, ui_trades: list[dict], used: set[int]) -> tuple[int | None, dict | None]:
    if entry_ts is None:
        return None, None
    best_i = None
    best = None
    best_dt = pd.Timedelta(minutes=31)
    for i, tr in enumerate(ui_trades):
        if i in used or tr.get("side") != side:
            continue
        dt = abs(tr["entry_dt"] - entry_ts)
        if dt <= pd.Timedelta(minutes=30) and dt < best_dt:
            best_i = i
            best = tr
            best_dt = dt
    return best_i, best


def trade_pnl(entry_px, exit_px, side: str) -> float | None:
    if entry_px is None or exit_px is None or not side:
        return None
    return (float(exit_px) - float(entry_px)) * side_mult(side) * POINT_VALUE - COMMISSION_RT


def signal_vs_ui(signal_trades: list[dict], ui_trades: list[dict],
                 exec_trades: list[dict] | None = None) -> list[dict]:
    rows = []
    used_ui: set[int] = set()
    exec_by_trade = {t.get("trade_id", ""): t for t in (exec_trades or [])}
    for tr in signal_trades:
        entry = tr.get("entry_signal")
        if not entry:
            rows.append({
                "severity": "CRITICAL",
                "trade_id": tr.get("trade_id", ""),
                "side": tr.get("side", ""),
                "signal_entry": "",
                "signal_exit": fmt_ts(sig["ts"]),
                "ui_entry": "",
                "ui_exit": "",
                "entry_delta_min": None,
                "exit_delta_min": None,
                "entry_px_delta": None,
                "exit_px_delta": None,
                "pnl_delta": None,
                "note": "CLOSE signal without entry signal in archive",
            })
            continue
        ui_i, ui = match_ui_trade(tr["side"], entry["ts"], ui_trades, used_ui)
        if ui_i is not None:
            used_ui.add(ui_i)
        exit_sig = tr.get("exit_signal")
        if ui is None:
            rows.append({
                "severity": "CRITICAL",
                "trade_id": tr["trade_id"],
                "side": tr["side"],
                "signal_entry": fmt_ts(entry["ts"]),
                "signal_exit": fmt_ts(exit_sig["ts"]) if exit_sig else "",
                "ui_entry": "MISSING",
                "ui_exit": "",
                "entry_delta_min": None,
                "exit_delta_min": None,
                "entry_px_delta": None,
                "exit_px_delta": None,
                "pnl_delta": None,
                "note": "no UI theoretical trade within 30min",
            })
            continue

        sig_exit_ts = exit_sig["ts"] if exit_sig else None
        sig_exit_px = exit_sig["price"] if exit_sig else None
        exec_trade = exec_by_trade.get(tr["trade_id"], {})
        exec_exit = exec_trade.get("exit_execution")
        manual_exit = exec_exit if exec_exit and exec_exit.get("event_type") == "MANUAL_CLOSE" else None
        sig_pnl = trade_pnl(entry["price"], sig_exit_px, tr["side"])
        ui_pnl = ui.get("pnl_usd")
        open_while_ui_closed = exit_sig is None and manual_exit is None and ui.get("exit_dt") is not None
        severity = "CRITICAL" if open_while_ui_closed else "PASS"
        note = "UI is theoretical backtest"
        if open_while_ui_closed:
            note = "UI theoretical closed while signal/live still open"
        elif exit_sig is None and manual_exit is not None:
            note = "Live was manually closed; signal archive has no CLOSE"
        rows.append({
            "severity": severity,
            "trade_id": tr["trade_id"],
            "side": tr["side"],
            "signal_entry": fmt_ts(entry["ts"]),
            "signal_exit": fmt_ts(sig_exit_ts) if sig_exit_ts is not None else "",
            "ui_entry": fmt_ts(ui["entry_dt"]),
            "ui_exit": fmt_ts(ui["exit_dt"]) if ui.get("exit_dt") is not None else "",
            "entry_delta_min": (entry["ts"] - ui["entry_dt"]).total_seconds() / 60.0,
            "exit_delta_min": ((sig_exit_ts - ui["exit_dt"]).total_seconds() / 60.0 if sig_exit_ts is not None and ui.get("exit_dt") is not None else None),
            "entry_px_delta": entry["price"] - float(ui.get("entry_price", 0.0)),
            "exit_px_delta": (sig_exit_px - float(ui.get("exit_price", 0.0)) if sig_exit_px is not None else None),
            "pnl_delta": (sig_pnl - float(ui_pnl) if sig_pnl is not None and ui_pnl is not None else None),
            "note": note,
        })
    return rows


def topstep_vs_ui(exec_trades: list[dict], ui_trades: list[dict]) -> list[dict]:
    rows = []
    used_ui: set[int] = set()
    for tr in exec_trades:
        entry = tr.get("entry_execution")
        if not entry:
            rows.append({
                "severity": "CRITICAL",
                "trade_id": tr.get("trade_id", ""),
                "side": tr.get("side", ""),
                "actual_entry": "",
                "actual_exit": fmt_ts(tr.get("exit_execution", {}).get("ts")),
                "ui_entry": "",
                "ui_exit": "",
                "entry_delta_min": None,
                "exit_delta_min": None,
                "entry_px_delta": None,
                "exit_px_delta": None,
                "pnl_delta": None,
                "note": "FLATTEN execution without MARKET entry in ledger",
            })
            continue
        ui_i, ui = match_ui_trade(tr["side"], entry.get("ts"), ui_trades, used_ui)
        if ui_i is not None:
            used_ui.add(ui_i)
        if ui is None:
            rows.append({
                "severity": "CRITICAL",
                "trade_id": tr.get("trade_id", ""),
                "side": tr.get("side", ""),
                "actual_entry": fmt_ts(entry.get("ts")),
                "actual_exit": fmt_ts(tr.get("exit_execution", {}).get("ts")) if tr.get("exit_execution") else "",
                "ui_entry": "MISSING",
                "ui_exit": "",
                "entry_delta_min": None,
                "exit_delta_min": None,
                "entry_px_delta": None,
                "exit_px_delta": None,
                "pnl_delta": None,
                "note": "actual execution without matching UI theoretical trade",
            })
            continue
        exit_ex = tr.get("exit_execution")
        entry_px = entry.get("executed_price") or entry.get("requested_price")
        exit_px = (exit_ex.get("executed_price") or exit_ex.get("requested_price")) if exit_ex else None
        actual_pnl = trade_pnl(entry_px, exit_px, tr["side"])
        ui_pnl = ui.get("pnl_usd")
        actual_open_ui_closed = exit_ex is None and ui.get("exit_dt") is not None
        if exit_ex and exit_ex.get("event_type") == "MANUAL_CLOSE":
            actual_exit = "manual"
        else:
            actual_exit = fmt_ts(exit_ex.get("ts")) if exit_ex else ""
        exit_delta_min = None
        if (
            exit_ex
            and exit_ex.get("event_type") != "MANUAL_CLOSE"
            and exit_ex.get("ts") is not None
            and ui.get("exit_dt") is not None
        ):
            exit_delta_min = (exit_ex.get("ts") - ui["exit_dt"]).total_seconds() / 60.0
        note = "UI is theoretical backtest"
        if actual_open_ui_closed:
            note = "actual open while UI theoretical is closed"
        elif exit_ex and exit_ex.get("event_type") == "MANUAL_CLOSE":
            note = "actual was manually closed; exact close time unavailable in legacy log"
        rows.append({
            "severity": "CRITICAL" if actual_open_ui_closed else "PASS",
            "trade_id": tr.get("trade_id", ""),
            "side": tr["side"],
            "actual_entry": fmt_ts(entry.get("ts")),
            "actual_exit": actual_exit,
            "ui_entry": fmt_ts(ui["entry_dt"]),
            "ui_exit": fmt_ts(ui["exit_dt"]) if ui.get("exit_dt") is not None else "",
            "entry_delta_min": (entry.get("ts") - ui["entry_dt"]).total_seconds() / 60.0,
            "exit_delta_min": exit_delta_min,
            "entry_px_delta": (float(entry_px) - float(ui.get("entry_price", 0.0)) if entry_px is not None else None),
            "exit_px_delta": (float(exit_px) - float(ui.get("exit_price", 0.0)) if exit_px is not None else None),
            "pnl_delta": (actual_pnl - float(ui_pnl) if actual_pnl is not None and ui_pnl is not None else None),
            "note": note,
        })
    return rows


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if val is None:
                val = ""
            elif isinstance(val, float):
                val = fmt_num(val, 2)
            vals.append(str(val).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def filter_inputs(signals, executions, ui_trades, start, end):
    signals_f = [s for s in signals if in_window(s["ts"] or s["received_at"], start, end)]
    executions_f = [e for e in executions if in_window(e.get("ts"), start, end)]
    ui_f = [
        t for t in ui_trades
        if (
            in_window(t.get("entry_dt"), start, end)
            or in_window(t.get("exit_dt"), start, end)
            or (
                t.get("entry_dt") is not None
                and t.get("entry_dt") < end
                and (t.get("exit_dt") is None or t.get("exit_dt") >= start)
            )
        )
    ]
    return signals_f, executions_f, ui_f


def build_parity_report(date_s: str | None = None, tz_name: str = "Asia/Jakarta") -> dict:
    day, start, end = date_window(date_s, tz_name)
    signals, executions, ui_trades = filter_inputs(load_signals(), load_executions(), load_ui_trades(), start, end)

    signal_entries = [s for s in signals if is_entry_signal(s)]
    topstep_entries = [e for e in executions if is_market_entry(e)]
    ui_entries = [t for t in ui_trades if in_window(t.get("entry_dt"), start, end)]
    execution_sync = signal_entry_vs_topstep(signals, executions)
    logic_sync = signal_entry_vs_ui(signals, ui_trades)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date": day,
        "timezone": tz_name,
        "window_utc": {"start": start.isoformat(), "end": end.isoformat()},
        "sources": {
            "signals": str(SIGNALS_PATH),
            "executions": str(EXECUTIONS_PATH),
            "ui_theoretical": str(UI_PATH),
        },
        "summary": {
            "signals": len(signal_entries),
            "executions": len(topstep_entries),
            "ui_theoretical_trades": len(ui_entries),
            "signal_entries": len(signal_entries),
            "topstep_entry_events": len(topstep_entries),
            "ui_entry_events": len(ui_entries),
            "critical": sum(1 for rows in (execution_sync, logic_sync) for r in rows if r.get("severity") == "CRITICAL"),
        },
        "signal_entry_vs_topstep_entry": execution_sync,
        "signal_entry_vs_ui_entry": logic_sync,
        # Backward-compatible names for older consumers. These now intentionally
        # represent entry-only parity, not lifecycle/exit parity.
        "signal_vs_topstep": execution_sync,
        "signal_vs_ui_theoretical": logic_sync,
        "topstep_vs_ui_theoretical": [],
    }


def render_markdown_report(report: dict) -> str:
    start = report["window_utc"]["start"]
    end = report["window_utc"]["end"]
    execution_sync = report["signal_entry_vs_topstep_entry"]
    logic_sync = report["signal_entry_vs_ui_entry"]
    md = [
        f"# Super Structure Parity {report['date']} ({report['timezone']})",
        "",
        f"Window UTC: `{start}` -> `{end}`",
        "",
        "Scope: entry-only drift. Topstep is checked only for entry fills; UI is checked only for theoretical strategy entries.",
        "Manual closes and theoretical exits are context, not critical parity failures.",
        "",
        "## Signal Entry vs Topstep Entry",
        markdown_table(execution_sync, ["severity", "side", "signal_entry", "signal_px", "topstep_entry", "topstep_px", "slippage", "drift_type", "note"]),
        "",
        "## Signal Entry vs UI Entry",
        markdown_table(logic_sync, ["severity", "side", "signal_entry", "signal_px", "ui_entry", "ui_exit", "ui_status", "entry_delta_min", "entry_px_delta", "drift_type", "note"]),
        "",
    ]
    return "\n".join(md)


def write_parity_report(report: dict) -> tuple[Path, Path, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"super_structure_parity_{report['date']}_{report['timezone'].replace('/', '-')}"
    json_path = OUT_DIR / f"{stem}.json"
    md_path = OUT_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    md_text = render_markdown_report(report)
    md_path.write_text(md_text)
    return json_path, md_path, md_text


def _short_row(row: dict, section: str) -> str:
    note = str(row.get("note", ""))
    if section == "Signal vs Topstep":
        return (
            f"- {row.get('side', '')} signal `{row.get('signal_entry', '')}` "
            f"vs Topstep `{row.get('topstep_entry', '')}`: {row.get('drift_type', '')} {note}"
        )
    if section == "Signal vs UI":
        return (
            f"- {row.get('side', '')} signal `{row.get('signal_entry', '')}` "
            f"vs UI `{row.get('ui_entry', '')}`: {row.get('drift_type', '')} {note}"
        )
    return f"- {note}"


def _time_part(value: str) -> str:
    if not value:
        return "open"
    if value == "manual":
        return "manual"
    try:
        return pd.Timestamp(value).strftime("%H:%M")
    except Exception:
        return str(value)[-5:] if len(str(value)) >= 5 else str(value)


def _span(entry: str, exit_: str) -> str:
    return f"{_time_part(entry)}>{_time_part(exit_)}"


def _fit(value: str, width: int) -> str:
    value = str(value)
    if len(value) > width:
        return value[:max(0, width - 1)] + "…"
    return value.ljust(width)


def comparison_table_for_telegram(report: dict) -> str:
    logic_by_trade = {
        row.get("trade_id", ""): row
        for row in report["signal_entry_vs_ui_entry"]
    }
    execution_sync = report["signal_entry_vs_topstep_entry"]
    rows = []
    for row in execution_sync:
        logic = logic_by_trade.get(row.get("trade_id", ""), {})
        side = "L" if row.get("side") == "Long" else ("S" if row.get("side") == "Short" else "?")
        exec_ok = row.get("severity") == "PASS"
        logic_ok = logic.get("severity") == "PASS"
        status = "OK" if exec_ok and logic_ok else "CRIT"
        note = row.get("drift_type") or logic.get("drift_type") or ""
        rows.append({
            "side": side,
            "signal": _time_part(row.get("signal_entry", "")),
            "topstep": _time_part(row.get("topstep_entry", "")),
            "ui": _time_part(logic.get("ui_entry", "")),
            "status": status,
            "note": note,
        })

    if not rows:
        return "No matched trade rows."

    lines = [
        "Side Signal   Topstep  UI       St  Drift",
        "---- -------- -------- -------- --- --------------------",
    ]
    for row in rows[:10]:
        lines.append(
            f"{_fit(row['side'], 4)} "
            f"{_fit(row['signal'], 8)} "
            f"{_fit(row['topstep'], 8)} "
            f"{_fit(row['ui'], 8)} "
            f"{_fit(row['status'], 3)} "
            f"{row['note']}"
        )
    remaining = len(rows) - 10
    if remaining > 0:
        lines.append(f"... {remaining} more row(s)")
    return "\n".join(lines)


def format_telegram_report(report: dict, max_critical_rows: int = 8) -> str:
    summary = report["summary"]
    critical = int(summary.get("critical", 0))
    status = "CRITICAL" if critical else "PASS"
    lines = [
        f"📊 *Super Structure Parity* `{report['date']}`",
        f"TZ: `{report['timezone']}`",
        "",
        f"Status: *{status}*",
        f"Signal entries: `{summary.get('signal_entries', 0)}` | Topstep entries: `{summary.get('topstep_entry_events', 0)}` | UI entries: `{summary.get('ui_entry_events', 0)}`",
        f"Critical rows: `{critical}`",
        "",
        "_Entry-only parity: Signal vs UI for logic; Signal vs Topstep for fills._",
        "",
        "*Comparison*",
        "```",
        comparison_table_for_telegram(report),
        "```",
    ]
    if critical:
        lines.extend(["", "*Critical details:*"])
        sections = [
            ("Signal vs Topstep", report["signal_vs_topstep"]),
            ("Signal vs UI", report["signal_vs_ui_theoretical"]),
        ]
        shown = 0
        for section, rows in sections:
            bad = [r for r in rows if r.get("severity") == "CRITICAL"]
            if not bad:
                continue
            lines.append(f"*{section}*")
            for row in bad:
                if shown >= max_critical_rows:
                    remaining = critical - shown
                    if remaining > 0:
                        lines.append(f"- ... `{remaining}` more critical row(s) in saved report")
                    return "\n".join(lines)
                lines.append(_short_row(row, section))
                shown += 1
    return "\n".join(lines)


def _load_telegram_env() -> tuple[str, str]:
    if not TELEGRAM_ENV_PATH.exists():
        return "", ""
    token, chat_id = "", ""
    for line in TELEGRAM_ENV_PATH.read_text().strip().split("\n"):
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k == "TELEGRAM_BOT_TOKEN":
            token = v.strip()
        elif k == "TELEGRAM_CHAT_ID":
            chat_id = v.strip()
    return token, chat_id


def _telegram_send(token: str, chat_id: str, text: str) -> bool:
    import urllib.request, urllib.parse
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }).encode()
    try:
        resp = urllib.request.urlopen(url, data, timeout=10)
        result = json.loads(resp.read())
        return bool(result.get("ok"))
    except Exception as exc:
        print(f"telegram send error: {exc}", file=sys.stderr)
        return False


def _state_signature(report: dict) -> dict:
    s = report["summary"]
    critical = int(s.get("critical", 0))
    return {
        "date": report["date"],
        "status": "CRITICAL" if critical else "PASS",
        "signals": int(s.get("signals", 0)),
        "executions": int(s.get("executions", 0)),
        "ui": int(s.get("ui_theoretical_trades", 0)),
        "critical": critical,
    }


def _should_push_auto(current: dict, previous: dict | None) -> tuple[bool, str]:
    if current["signals"] == 0 and current["executions"] == 0 and current["ui"] == 0:
        return False, "empty day"
    if previous is None:
        return True, "first run"
    keys = ("date", "status", "signals", "executions", "ui", "critical")
    if any(current[k] != previous.get(k) for k in keys):
        return True, "state changed"
    if current["status"] == "CRITICAL":
        return True, "still critical"
    return False, "no change"


def maybe_push_telegram(report: dict, mode: str) -> None:
    if mode == "never":
        return
    token, chat_id = _load_telegram_env()
    if not (token and chat_id):
        print("telegram env missing, skipping push", file=sys.stderr)
        return

    current = _state_signature(report)
    previous = None
    if mode == "auto" and TELEGRAM_STATE_PATH.exists():
        try:
            previous = json.loads(TELEGRAM_STATE_PATH.read_text())
        except Exception:
            previous = None

    if mode == "auto":
        push, reason = _should_push_auto(current, previous)
        if not push:
            print(f"auto: skip ({reason})")
            return
        print(f"auto: push ({reason})")

    text = format_telegram_report(report)
    if not _telegram_send(token, chat_id, text):
        return
    if mode == "auto":
        TELEGRAM_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        current["sent_at"] = datetime.now(timezone.utc).isoformat()
        TELEGRAM_STATE_PATH.write_text(json.dumps(current, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Super Structure 3-way parity report")
    parser.add_argument("--date", help="Trading-day date in selected timezone, e.g. 2026-05-07")
    parser.add_argument("--tz", default="Asia/Jakarta", choices=["Asia/Jakarta", "UTC"], help="Date timezone; default Asia/Jakarta")
    parser.add_argument("--telegram", default="never", choices=["never", "auto", "always"],
                        help="Push report to Telegram. 'auto' only pushes on state change or critical; 'always' pushes every run.")
    args = parser.parse_args()

    report = build_parity_report(args.date, args.tz)
    json_path, md_path, md_text = write_parity_report(report)
    print(md_text)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    maybe_push_telegram(report, args.telegram)
    return 1 if report["summary"]["critical"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

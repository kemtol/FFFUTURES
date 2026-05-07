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
POINT_VALUE = 10.0
COMMISSION_RT = 1.74


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
        out.append(row)
    return out


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
                "ui_entry": "",
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
                "ui_entry": "MISSING",
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
            "ui_entry": fmt_ts(ui["entry_dt"]),
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
                "ui_entry": "",
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
                "ui_entry": "MISSING",
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
            "ui_entry": fmt_ts(ui["entry_dt"]),
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
        if in_window(t.get("entry_dt"), start, end) or in_window(t.get("exit_dt"), start, end)
    ]
    return signals_f, executions_f, ui_f


def build_parity_report(date_s: str | None = None, tz_name: str = "Asia/Jakarta") -> dict:
    day, start, end = date_window(date_s, tz_name)
    signals, executions, ui_trades = filter_inputs(load_signals(), load_executions(), load_ui_trades(), start, end)

    signal_trades = build_signal_trades(signals)
    exec_trades = build_execution_trades(executions)
    svt = signal_vs_topstep(signals, executions)
    svu = signal_vs_ui(signal_trades, ui_trades, exec_trades)
    tvu = topstep_vs_ui(exec_trades, ui_trades)

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
            "signals": len(signals),
            "executions": len(executions),
            "ui_theoretical_trades": len(ui_trades),
            "critical": sum(1 for rows in (svt, svu, tvu) for r in rows if r.get("severity") == "CRITICAL"),
        },
        "signal_vs_topstep": svt,
        "signal_vs_ui_theoretical": svu,
        "topstep_vs_ui_theoretical": tvu,
    }


def render_markdown_report(report: dict) -> str:
    start = report["window_utc"]["start"]
    end = report["window_utc"]["end"]
    svt = report["signal_vs_topstep"]
    svu = report["signal_vs_ui_theoretical"]
    tvu = report["topstep_vs_ui_theoretical"]
    md = [
        f"# Super Structure Parity {report['date']} ({report['timezone']})",
        "",
        f"Window UTC: `{start}` -> `{end}`",
        "",
        "UI rows are theoretical backtest rows from the TopstepX buffer snapshot; they are not execution truth.",
        "",
        "## Signal vs Topstep",
        markdown_table(svt, ["severity", "signal", "signal_ts", "signal_px", "execution", "exec_ts", "exec_px", "slippage", "note"]),
        "",
        "## Signal vs UI Theoretical",
        markdown_table(svu, ["severity", "side", "signal_entry", "ui_entry", "entry_delta_min", "exit_delta_min", "entry_px_delta", "exit_px_delta", "pnl_delta", "note"]),
        "",
        "## Topstep vs UI Theoretical",
        markdown_table(tvu, ["severity", "side", "actual_entry", "ui_entry", "entry_delta_min", "exit_delta_min", "entry_px_delta", "exit_px_delta", "pnl_delta", "note"]),
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
            f"- {row.get('signal', '')} `{row.get('signal_ts', '')}` "
            f"@ `{fmt_num(row.get('signal_px'), 1)}` -> {row.get('execution', '')}: {note}"
        )
    if section == "Signal vs UI":
        return (
            f"- {row.get('side', '')} signal `{row.get('signal_entry', '')}` "
            f"vs UI `{row.get('ui_entry', '')}`: {note}"
        )
    return (
        f"- {row.get('side', '')} actual `{row.get('actual_entry', '')}` "
        f"vs UI `{row.get('ui_entry', '')}`: {note}"
    )


def format_telegram_report(report: dict, max_critical_rows: int = 8) -> str:
    summary = report["summary"]
    critical = int(summary.get("critical", 0))
    status = "CRITICAL" if critical else "PASS"
    lines = [
        f"📊 *Super Structure Parity* `{report['date']}`",
        f"TZ: `{report['timezone']}`",
        "",
        f"Status: *{status}*",
        f"Signals: `{summary.get('signals', 0)}` | Executions: `{summary.get('executions', 0)}` | UI theoretical: `{summary.get('ui_theoretical_trades', 0)}`",
        f"Critical rows: `{critical}`",
        "",
        "_UI = theoretical backtest from TopstepX buffer snapshot, not execution truth._",
    ]
    if critical:
        lines.extend(["", "*Critical details:*"])
        sections = [
            ("Signal vs Topstep", report["signal_vs_topstep"]),
            ("Signal vs UI", report["signal_vs_ui_theoretical"]),
            ("Topstep vs UI", report["topstep_vs_ui_theoretical"]),
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Super Structure 3-way parity report")
    parser.add_argument("--date", help="Trading-day date in selected timezone, e.g. 2026-05-07")
    parser.add_argument("--tz", default="Asia/Jakarta", choices=["Asia/Jakarta", "UTC"], help="Date timezone; default Asia/Jakarta")
    args = parser.parse_args()

    report = build_parity_report(args.date, args.tz)
    json_path, md_path, md_text = write_parity_report(report)
    print(md_text)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return 1 if report["summary"]["critical"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

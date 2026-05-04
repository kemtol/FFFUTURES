#!/usr/bin/env python3
"""
Minimal webhook receiver — stores incoming TradingView alert payloads as JSON.

Runs a lightweight HTTP server in a background thread.
Every POST /webhook is appended to data/Live/webhook_log.json with timestamp.

Usage (standalone):
    python3 pipeline/live/webhook.py [--port 8080]

Usage (embedded in daemon):
    from pipeline.live.webhook import WebhookServer
    ws = WebhookServer(port=8080)
    ws.start()
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import urllib.parse
import urllib.request
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = ROOT / "data" / "Live" / "webhook_log.json"
DEFAULT_PORT = 8080


class WebhookHandler(BaseHTTPRequestHandler):
    """Handle POST /webhook — append payload to JSON log."""

    def log_message(self, fmt, *args):
        print(f"[Webhook] {args[0]}", flush=True)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/telegram-signal":
            self._handle_telegram_signal()
            return

        if path != "/webhook":
            self.send_response(404)
            self.end_headers()
            return

        content_len = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw.decode("utf-8", errors="replace")}

        entry = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "source_ip": self.client_address[0],
            "payload": payload,
        }

        self._append_log(entry)

        # Try to parse and execute if it looks like a trading signal
        self._maybe_execute(payload)

        print(f"[Webhook] Received alert — {payload.get('subject', 'NO SUBJECT')[:80]}", flush=True)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_response(404)
        self._send_cors_headers()
        self.end_headers()

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status: int, payload: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def _handle_telegram_signal(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "error": "invalid_json"})
            return

        ok, detail = send_telegram_strategy_signal(payload)
        if ok:
            self._send_json(200, {"status": "ok", "detail": detail})
        else:
            self._send_json(503, {"status": "error", "error": detail})

    def _append_log(self, entry: dict) -> None:
        """Thread-safe append to JSON log file."""
        lock = getattr(self.server, "_log_lock", threading.Lock())
        with lock:
            try:
                if LOG_PATH.exists():
                    existing = json.loads(LOG_PATH.read_text())
                else:
                    existing = []
            except (json.JSONDecodeError, FileNotFoundError):
                existing = []

            existing.append(entry)

            if len(existing) > 1000:
                existing = existing[-1000:]

            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            LOG_PATH.write_text(json.dumps(existing, indent=2, default=str))

    def _maybe_execute(self, payload: dict) -> None:
        """Parse alert payload and forward to TopstepX if it's a trading signal.

        Recognized format (key=value pairs in subject or body):
            action=BUY symbol=MGC entry=<price> sl=<price>
            action=SELL symbol=MGC entry=<price> sl=<price>
            action=CLOSE symbol=MGC reason=<SL|TREND_FLIP|SESSION_END> price=<price>
        """
        # Extract text from payload
        text = ""
        if payload.get("subject"):
            text += str(payload["subject"]) + " "
        if payload.get("body"):
            text += str(payload["body"])
        if payload.get("message"):
            text += str(payload["message"])

        # Check for our alert format
        if "action=" not in text.lower() and "action=" not in text:
            return  # Not a trading signal

        # Parse key=value pairs
        signal = {}
        for part in text.split():
            if "=" in part:
                k, v = part.split("=", 1)
                signal[k.lower()] = v.strip()

        action = signal.get("action", "").upper()
        symbol = signal.get("symbol", "MGC")
        entry_str = signal.get("entry", "")
        sl_str = signal.get("sl", "")
        price_str = signal.get("price", "")
        reason = signal.get("reason", "").upper()

        if not action:
            return

        try:
            entry = float(entry_str) if entry_str else 0.0
            sl = float(sl_str) if sl_str else 0.0
            price = float(price_str) if price_str else 0.0
        except ValueError:
            print(f"[Webhook] Invalid price in signal: {text.strip()[:120]}", flush=True)
            return

        signal_log = {
            "action": action,
            "symbol": symbol,
            "entry": entry if action in ("BUY", "SELL") else price,
            "sl": sl,
            "reason": reason,
            "parsed_from": text.strip()[:200],
        }

        signal_path = Path(__file__).resolve().parent.parent.parent / "data" / "Live" / "tv_signals.json"
        try:
            if signal_path.exists():
                signals = json.loads(signal_path.read_text())
            else:
                signals = []
            signals.append({
                "received_at": datetime.now(timezone.utc).isoformat(),
                "signal": signal_log,
            })
            signal_path.parent.mkdir(parents=True, exist_ok=True)
            signal_path.write_text(json.dumps(signals, indent=2, default=str))
        except Exception as e:
            print(f"[Webhook] Failed to store signal: {e}", flush=True)

        print(f"[Webhook] ⚡ SIGNAL: {action} {symbol} @ {entry or price:.1f}"
              + (f" SL={sl:.1f}" if sl > 0 else "")
              + (f" ({reason})" if reason else ""), flush=True)


def _telegram_config() -> tuple[str, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    env_file = ROOT / "data" / "Live" / "telegram.env"
    if env_file.exists():
        for line in env_file.read_text().strip().split("\n"):
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key == "TELEGRAM_BOT_TOKEN" and not token:
                token = value.strip()
            elif key == "TELEGRAM_CHAT_ID" and not chat_id:
                chat_id = value.strip()
    return token, chat_id


def _format_strategy_signal(payload: dict) -> str:
    strategy = payload.get("strategy", "Super Structure")
    timeframe = payload.get("timeframe", "5m")
    side = payload.get("side", "n/a")
    session = payload.get("session", "n/a")
    trade_no = payload.get("trade_no", "n/a")
    entry_ts = payload.get("entry_ts", "n/a")
    entry = payload.get("entry_price", "n/a")
    exit_ts = payload.get("exit_ts", "")
    exit_reason = payload.get("exit_reason", "")
    adx = payload.get("entry_adx", "n/a")
    cci = payload.get("entry_cci", "n/a")
    pnl = payload.get("pnl_usd", "n/a")
    r_multiple = payload.get("r_multiple", "n/a")

    def num(value, digits=1):
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    lines = [
        "📡 *Strategy Signal*",
        "",
        f"Strategy: `{strategy}`",
        f"Timeframe: `{timeframe}`",
        f"Trade: `#{trade_no}`",
        f"Side: *{side}* | Session: `{session}`",
        f"Entry: `{entry_ts}` @ `${num(entry, 1)}`",
        f"ADX: `{num(adx, 1)}` | CCI: `{num(cci, 0)}`",
        f"Backtest PnL: `${num(pnl, 0)}` | R: `{num(r_multiple, 2)}`",
    ]
    if exit_ts:
        lines.append(f"Exit: `{exit_ts}`" + (f" ({exit_reason})" if exit_reason else ""))
    return "\n".join(lines)


def send_telegram_strategy_signal(payload: dict) -> tuple[bool, str]:
    token, chat_id = _telegram_config()
    if not token or not chat_id:
        return False, "telegram_not_configured"

    text = _format_strategy_signal(payload)
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }).encode()
        urllib.request.urlopen(url, data, timeout=8)
        print(f"[Webhook] Telegram strategy signal sent for trade #{payload.get('trade_no', 'n/a')}", flush=True)
        return True, "sent"
    except Exception as exc:
        print(f"[Webhook] Telegram send failed: {exc}", flush=True)
        return False, "telegram_send_failed"


class WebhookServer:
    """Background HTTP server for receiving webhook alerts."""

    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start HTTP server in a background daemon thread."""
        self._server = HTTPServer(("0.0.0.0", self.port), WebhookHandler)
        self._server._log_lock = threading.Lock()
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[Webhook] Listening on :{self.port} — POST /webhook", flush=True)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Webhook receiver")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    ws = WebhookServer(port=args.port)
    ws.start()
    print(f"Running. Send POST to http://localhost:{args.port}/webhook")
    print(f"Logs stored at: {LOG_PATH}")
    ws._thread.join()  # block forever

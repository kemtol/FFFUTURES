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
import threading
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
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
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode())

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_response(404)
        self.end_headers()

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

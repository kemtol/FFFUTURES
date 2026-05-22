#!/usr/bin/env python3
"""
TopstepX real-time MGC feed via WebSocket (SignalR protocol).

Symbol: F.US.MGC, SubscribeBars with 1-min resolution.
Token persists in data/Live/topstepx_token.json.
"""

from __future__ import annotations

import asyncio
import base64
import fcntl
import json
import os
import ssl
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent.parent.parent
TOKEN_FILE = ROOT / "data" / "Live" / "topstepx_token.json"
TOKEN_LOCK_FILE = ROOT / "data" / "Live" / "topstepx_token.lock"
CHART_WS = "wss://chartapi.topstepx.com/hubs/chart"
SYMBOL = "F.US.MGC"
RESOLUTION = "1"
TOKEN_REFRESH_MARGIN_SECONDS = 600
_TOKEN_REFRESH_LOCK = threading.Lock()


def _decode_jwt_payload(token: str | None) -> dict:
    """Return the JWT payload if this looks like a JWT, otherwise {}."""
    if not token:
        return {}
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    try:
        payload = parts[1]
        padded = payload + "=" * ((4 - len(payload) % 4) % 4)
        return json.loads(base64.urlsafe_b64decode(padded))
    except Exception:
        return {}


def token_expires_at(token: str | None = None) -> int | None:
    """Unix expiry timestamp from the TopstepX JWT, if available."""
    if token is None and TOKEN_FILE.exists():
        token = json.loads(TOKEN_FILE.read_text()).get("access_token")
    payload = _decode_jwt_payload(token)
    exp = payload.get("exp")
    try:
        return int(exp) if exp is not None else None
    except (TypeError, ValueError):
        return None


def token_needs_refresh(token: str | None = None, margin_seconds: int = TOKEN_REFRESH_MARGIN_SECONDS) -> bool:
    """True when token is missing, expired, or close to expiry."""
    if token is None and TOKEN_FILE.exists():
        token = json.loads(TOKEN_FILE.read_text()).get("access_token")
    if not token:
        return True
    exp = token_expires_at(token)
    if exp is None:
        return False
    return exp <= int(time.time()) + margin_seconds


def ensure_token(force: bool = False) -> str:
    """Return a usable token, refreshing through Playwright when needed."""
    token = json.loads(TOKEN_FILE.read_text()).get("access_token") if TOKEN_FILE.exists() else None
    if not force and not token_needs_refresh(token):
        return token
    with _TOKEN_REFRESH_LOCK:
        TOKEN_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with TOKEN_LOCK_FILE.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                saved = json.loads(TOKEN_FILE.read_text()) if TOKEN_FILE.exists() else {}
                token = saved.get("access_token")
                saved_at = saved.get("saved_at")
                recently_refreshed = False
                if saved_at:
                    try:
                        ts = datetime.fromisoformat(saved_at.replace("Z", "+00:00"))
                        recently_refreshed = (datetime.now(timezone.utc) - ts).total_seconds() < 120
                    except Exception:
                        recently_refreshed = False
                if (not force or recently_refreshed) and not token_needs_refresh(token):
                    return token
                return _fetch_token_unlocked()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


class TopstepXFeed:
    """Real-time MGC 1m candle feed from TopstepX chart WebSocket."""

    def __init__(self, db_path: str | None = None):
        self.token = self._load_token()
        self._callbacks: list[Callable] = []
        self._running = False
        self._last_ts: str | None = None  # track candle completion
        self._db_path = db_path or str(ROOT / "data" / "Live" / "topstepx_buffer.db")

    def _load_token(self) -> str | None:
        if TOKEN_FILE.exists():
            return json.loads(TOKEN_FILE.read_text()).get("access_token")
        return None

    def on_candle(self, cb: Callable[[dict], None]) -> None:
        self._callbacks.append(cb)

    async def _connect(self) -> None:
        import websockets

        url = f"{CHART_WS}?access_token={self.token}"
        async with websockets.connect(url, ssl=ssl.create_default_context()) as ws:
            print(f"[TopstepX] Connected")
            self._running = True

            # SignalR handshake
            await ws.send('{"protocol":"json","version":1}\x1e')
            resp = await ws.recv()
            print(f"[TopstepX] Handshake OK")

            # Subscribe to MGC 1m bars
            sub = json.dumps({
                "arguments": [SYMBOL, RESOLUTION],
                "invocationId": "0",
                "target": "SubscribeBars",
                "type": 1,
            }) + "\x1e"
            await ws.send(sub)
            print(f"[TopstepX] Subscribed: {SYMBOL} {RESOLUTION}m")

            while self._running:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                except asyncio.TimeoutError:
                    # Send ping every 30s
                    try:
                        await ws.send('{"type":6}\x1e')
                    except Exception:
                        break
                    continue
                except Exception:
                    break

                for frame in raw.split("\x1e"):
                    frame = frame.strip()
                    if not frame:
                        continue
                    try:
                        msg = json.loads(frame)
                    except json.JSONDecodeError:
                        continue
                    self._handle(msg)

    def _handle(self, msg: dict) -> None:
        if msg.get("type") == 6:
            return

        target = msg.get("target", "")
        args = msg.get("arguments", [])

        if "Bar" in target and len(args) >= 3 and isinstance(args[2], dict):
            bar = args[2]
            ts = bar.get("timestamp", "")[:19]
            o = float(bar.get("open", 0))
            h = float(bar.get("high", 0))
            l = float(bar.get("low", 0))
            c = float(bar.get("close", 0))
            v = int(bar.get("volume", 0))

            # Current running candle
            candle = {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}

            # Detect completion: new timestamp → previous candle done
            if self._last_ts and ts != self._last_ts and hasattr(self, '_running_candle'):
                self._write_to_db(ts, self._running_candle)

            self._last_ts = ts
            self._running_candle = candle  # always keep latest

            # Fire real-time tick
            for cb in self._callbacks:
                try:
                    cb(candle)
                except Exception:
                    pass

    def _write_to_db(self, current_candle_ts: str, candle: dict) -> None:
        """Write completed candle to SQLite buffer."""
        import sqlite3, pandas as pd

        ts_iso = candle["ts"]
        # epoch_ms
        try:
            epoch = int(pd.Timestamp(ts_iso, tz="UTC").timestamp() * 1000)
        except Exception:
            epoch = 0
        ts_fmt = f"{ts_iso[:10]} {ts_iso[11:19]}"

        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT OR IGNORE INTO ohlcv_1m "
                "(symbol, timeframe, epoch_ms, timestamp_utc, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ["MICRO_GOLD", "1m", epoch, ts_fmt,
                 candle["open"], candle["high"], candle["low"],
                 candle["close"], candle["volume"]],
            )
            conn.commit()
            conn.close()
            print(f"[TopstepX] Candle saved: {ts_fmt} O={candle['open']:.1f} C={candle['close']:.1f} V={candle['volume']}", flush=True)
        except Exception as e:
            print(f"[TopstepX] DB error: {e}", flush=True)

    def start(self) -> None:
        """Block and stream candles. Use in background thread."""
        if token_needs_refresh(self.token):
            self.token = ensure_token(force=not bool(self.token))
        if not self.token:
            raise RuntimeError("No token. Run login first.")
        asyncio.run(self._connect())

    def start_async(self) -> None:
        """Start in background daemon thread."""
        if token_needs_refresh(self.token):
            self.token = ensure_token(force=not bool(self.token))
        if not self.token:
            raise RuntimeError("No token.")
        threading.Thread(target=lambda: asyncio.run(self._connect()), daemon=True).start()

    def stop(self) -> None:
        self._running = False


# ── helper: re-login when token expires ──────────────────────────────────────

def _fetch_token_unlocked() -> str:
    """Login via Playwright headless and save token. Caller owns lock."""
    import asyncio as _a
    from playwright.async_api import async_playwright

    creds_file = ROOT / "data" / "Live" / "topstepx_creds.json"
    profile = ROOT / "data" / "Live" / "topstepx_profile"
    creds = json.loads(creds_file.read_text())

    async def _login():
        async with async_playwright() as p:
            ctx = await p.chromium.launch_persistent_context(
                str(profile), headless=True,
                viewport={"width": 1280, "height": 720},
            )
            page = await ctx.new_page()
            token = None

            def on_ws(ws):
                nonlocal token
                if "chart" in ws.url and not token:
                    for part in ws.url.split("?"):
                        if part.startswith("access_token="):
                            token = part.split("=", 1)[1]

            page.on("websocket", on_ws)
            await page.goto("https://www.topstepx.com/login", wait_until="networkidle", timeout=30000)
            await _a.sleep(2)

            if "login" in page.url:
                await page.fill("input[name=userName]", creds["email"])
                await page.fill("input[name=password]", creds["password"])
                await page.click("button:has-text(\"PLATFORM LOGIN\")")
                await _a.sleep(10)

            await ctx.close()
            return token

    token = _a.run(_login())
    if not token:
        raise RuntimeError("Login failed — no token captured.")
    TOKEN_FILE.write_text(json.dumps({
        "access_token": token,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }))
    return token


def fetch_token() -> str:
    """Login via Playwright headless and save token. Returns the token."""
    TOKEN_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _TOKEN_REFRESH_LOCK:
        with TOKEN_LOCK_FILE.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return _fetch_token_unlocked()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    feed = TopstepXFeed()
    if not feed.token:
        print("No token — fetching...")
        fetch_token()
        feed.token = feed._load_token()

    print(f"Symbol: {SYMBOL} {RESOLUTION}m")
    print(f"Token: {feed.token[:20]}...{feed.token[-10:]}")

    count = 0
    def on_c(c):
        global count
        count += 1
        ts = str(c.get("ts", ""))[:19]
        print(f"[{count:>3}] {ts} O={c['open']:.1f} H={c['high']:.1f} L={c['low']:.1f} C={c['close']:.1f} V={c['volume']}")

    feed.on_candle(on_c)

    try:
        feed.start()
    except KeyboardInterrupt:
        feed.stop()
        print(f"\n{count} candles received")

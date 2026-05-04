#!/usr/bin/env python3
"""
SignalBus — central publish/subscribe hub for trading signals.

Each strategy publishes to a named channel. Subscribed users receive
formatted Telegram messages. Phase 1 hardcodes one user; Phase 2 adds
multi-user subscription management.

Usage:
    from pipeline.live.signal_bus import SignalBus
    SignalBus().publish("tv_strategy", {"action": "BUY", ...})
"""

from __future__ import annotations

import json
import os as _os
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

HARDCODED_USER = "6283890722797"


class SignalBus:
    _instance: "SignalBus | None" = None

    def __new__(cls) -> "SignalBus":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._subscriptions: dict[str, set[str]] = {}
        self._token, self._own_chat_id = self._load_config()
        self.enabled = bool(self._token)
        self._seen: set[str] = set()  # dedup: key → skip repeat

    # ── config ──────────────────────────────────────────────────────────────

    def _load_config(self) -> tuple[str, str]:
        token = _os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = _os.environ.get("TELEGRAM_CHAT_ID", "")
        env_file = ROOT / "data" / "Live" / "telegram.env"
        if env_file.exists():
            for line in env_file.read_text().strip().split("\n"):
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k == "TELEGRAM_BOT_TOKEN" and not token:
                    token = v.strip()
                elif k == "TELEGRAM_CHAT_ID" and not chat_id:
                    chat_id = v.strip()
        return token, chat_id

    # ── pub/sub API ─────────────────────────────────────────────────────────

    def subscribe(self, user_id: str, strategy_name: str) -> None:
        from pipeline.live.user_db import subscribe as _db_subscribe
        _db_subscribe(user_id, strategy_name)

    def unsubscribe(self, user_id: str, strategy_name: str) -> None:
        from pipeline.live.user_db import unsubscribe as _db_unsub
        _db_unsub(user_id, strategy_name)

    def publish(self, strategy_name: str, payload: dict) -> None:
        if not self.enabled:
            print(f"[SignalBus] Telegram not configured, skipping {strategy_name}", flush=True)
            return

        dedup_key = f"{strategy_name}|{json.dumps(payload, sort_keys=True, default=str)}"
        if dedup_key in self._seen:
            return
        self._seen.add(dedup_key)
        if len(self._seen) > 500:
            self._seen = set(list(self._seen)[-250:])

        if strategy_name == "orb_v2":
            text = self._format_orb_v2(payload)
        elif strategy_name == "tv_strategy":
            text = self._format_tv_strategy(payload)
        else:
            text = f"📡 *{strategy_name}* signal\n\n```\n{json.dumps(payload, indent=2, default=str)[:400]}\n```"

        from pipeline.live.user_db import get_subscribers
        chat_ids = get_subscribers(strategy_name)
        for chat_id in chat_ids:
            self._send(chat_id, text)

    # ── formatters ──────────────────────────────────────────────────────────

    def _format_orb_v2(self, sig: dict) -> str:
        side = sig.get("side", "n/a")
        decision = sig.get("decision", "n/a")
        direction = (
            "🔴 SHORT" if (side == "BULL" and decision == "REV")
            or (side == "BEAR" and decision == "CONT")
            else "🟢 LONG"
        )

        def p(key, fmt=".1f"):
            try:
                return f"{float(sig.get(key, 0)):{fmt}}"
            except (TypeError, ValueError):
                return str(sig.get(key, "n/a"))

        session = str(sig.get("session", "n/a")).upper()
        orb_tf = sig.get("orb_tf", "n/a")

        lines = [
            f"🚨 *ORB v2.0 Signal — {session} {orb_tf}*",
            "",
            f"Breakout: `{side}`",
            f"Decision: *{decision}*  {direction}",
            "",
            f"Entry: `${p('entry')}`",
            f"TP:         `${p('tp')}` (4R)",
            f"SL:         `${p('sl')}` (1R)",
            "",
            f"P(Rev): `{p('prob_rev', '.3f')}`  P(Cont): `{p('prob_cont', '.3f')}`",
            "_Risk: $100/1R_",
        ]
        return "\n".join(lines)

    def _format_tv_strategy(self, sig: dict) -> str:
        action = sig.get("action", "n/a")
        ts = sig.get("ts", "")

        def p(key, fmt=".1f"):
            try:
                return f"{float(sig.get(key, 0)):{fmt}}"
            except (TypeError, ValueError):
                return str(sig.get(key, "n/a"))

        ts_line = f"`{ts[:19]}`\n" if ts else ""

        if action == "CLOSE":
            pnl = sig.get("pnl", 0)
            pnl_str = f"+${pnl:.0f}" if pnl >= 0 else f"-${abs(pnl):.0f}"
            exit_emoji = "✅" if pnl >= 0 else "❌"
            lines = [
                "📡 *Super Structure — Exit*",
                "",
                f"Time: {ts_line}" if ts else "",
                f"Action: {exit_emoji} `CLOSE` @ `${p('price')}`",
                f"PnL: `{pnl_str}`",
            ]
        else:
            emoji = "🟢" if action == "BUY" else "🔴"
            lines = [
                "📡 *Super Structure — Signal*",
                "",
                f"Time: {ts_line}" if ts else "",
                f"Action: {emoji} `{action}` @ `${p('price')}`",
                f"SL:           `${p('sl')}`",
                f"ADX: `{p('adx', '.1f')}`  |  CCI: `{p('cci', '.0f')}`",
                f"DEMA: `${p('dema', '.1f')}`",
            ]
            reason = sig.get("reason", "")
            if reason:
                lines[-1] += f"\nReason: `{reason}`"
        return "\n".join(l for l in lines if l)

    # ── Telegram send ───────────────────────────────────────────────────────

    def _send(self, chat_id: str, text: str) -> None:
        try:
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }).encode()
            urllib.request.urlopen(url, data, timeout=8)
            print(f"[SignalBus] Sent to {chat_id} — {text.split(chr(10))[0][:80]}", flush=True)
        except Exception as exc:
            print(f"[SignalBus] Send failed to {chat_id}: {exc}", flush=True)

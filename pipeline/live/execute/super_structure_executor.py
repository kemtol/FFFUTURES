#!/usr/bin/env python3
"""Super Structure auto-trade executor — bridges super_structure signals to TopstepX orders.

Usage:
    from pipeline.live.execute.super_structure_executor import SuperStructureExecutor
    ex = SuperStructureExecutor()
    ex.on_signal({"action":"BUY","symbol":"MGC","price":4692.9,"sl":4612.1})
"""
from __future__ import annotations

import json, time, traceback, urllib.parse, urllib.request, uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
TOKEN_FILE = ROOT / "data" / "Live" / "topstepx_token.json"
USERAPI = "https://userapi.topstepx.com"
ORDER_URL = f"{USERAPI}/Order"
SYMBOL_ID = "F.US.MGC"
ACCOUNT_ID = 22303383
USER_ID = 412653

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.topstepx.com",
    "Referer": "https://www.topstepx.com/trade",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

def _token() -> str:
    if TOKEN_FILE.exists():
        return json.loads(TOKEN_FILE.read_text())["access_token"]
    raise RuntimeError("No token file")

def _api(body: dict) -> dict:
    token = _token()
    h = {**HEADERS, "Authorization": f"Bearer {token}"}
    data = json.dumps(body).encode()
    time.sleep(0.3)
    req = urllib.request.Request(ORDER_URL, data=data, headers=h)
    resp = urllib.request.urlopen(req, timeout=10)
    return json.loads(resp.read())

def _cancel_all_orders() -> dict:
    """Cancel all open orders for the account/symbol."""
    token = _token()
    h = {**HEADERS, "Authorization": f"Bearer {token}"}
    url = f"https://userapi.topstepx.com/Order/cancel/{ACCOUNT_ID}/symbol/{SYMBOL_ID}"
    time.sleep(0.2)
    req = urllib.request.Request(url, method="DELETE", headers=h)
    resp = urllib.request.urlopen(req, timeout=10)
    body = resp.read()
    return json.loads(body) if body and body.strip() else {"status": resp.status, "msg": "cancelled"}

def _send_telegram(msg: str) -> None:
    try:
        env = ROOT / "data" / "Live" / "telegram.env"
        if not env.exists(): return
        token = chat = ""
        for line in env.read_text().strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                if k == "TELEGRAM_BOT_TOKEN": token = v
                elif k == "TELEGRAM_CHAT_ID": chat = v
        if token and chat:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = urllib.parse.urlencode({"chat_id": chat, "text": msg,
                                            "parse_mode": "Markdown"}).encode()
            urllib.request.urlopen(url, data, timeout=5)
    except Exception:
        pass


def _flatten_all() -> dict:
    """Close ALL positions via TopstepX Position/close endpoint."""
    token = _token()
    h = {**HEADERS, "Authorization": f"Bearer {token}"}
    url = f"{USERAPI}/Position/close/{ACCOUNT_ID}"
    time.sleep(0.3)
    req = urllib.request.Request(url, method="DELETE", headers=h)
    resp = urllib.request.urlopen(req, timeout=10)
    body = resp.read()
    return json.loads(body) if body and body.strip() else {"status": resp.status, "msg": "flattened"}


def _api_get(path: str) -> dict | list:
    """Generic GET to userapi with bearer auth. Returns parsed JSON."""
    token = _token()
    h = {**HEADERS, "Authorization": f"Bearer {token}"}
    req = urllib.request.Request(f"{USERAPI}{path}", method="GET", headers=h)
    resp = urllib.request.urlopen(req, timeout=10)
    return json.loads(resp.read())


def _query_positions() -> list[dict]:
    """List open positions for the user. Empty list = FLAT."""
    try:
        return _api_get(f"/Position/all/user/{USER_ID}") or []
    except Exception as exc:
        print(f"[SSExec] Position query failed: {exc}", flush=True)
        return []


def _validate_session() -> bool:
    """Return True if token still valid."""
    try:
        r = _api_get("/Session/validate")
        return isinstance(r, dict) and r.get("result") == 0
    except Exception:
        return False


def _check_violations() -> list[dict]:
    """Return active Topstep violations (MLL, daily loss, etc). Empty = OK."""
    try:
        r = _api_get(f"/Violations/active/{ACCOUNT_ID}")
        if r is None or r == "":
            return []
        return r if isinstance(r, list) else []
    except Exception:
        return []


class SuperStructureExecutor:
    """Executes Super Structure signals as TopstepX orders with trailing SL."""

    def __init__(self):
        self.active = False          # position active?
        self.pos_side = ""           # "Long" or "Short"
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.sl_order_id: int | None = None
        self._last_heartbeat = 0

    def on_signal(self, sig: dict) -> None:
        """Route signal to order placement. Called from super_structure._store_signal."""
        action = sig.get("action", "")
        price = sig.get("price", 0)
        sl = sig.get("sl", 0)

        if action == "BUY":
            self._enter("Buy", price, sl)
        elif action == "SELL":
            self._enter("Sell", price, sl)
        elif action == "CLOSE":
            self._exit(price, sig.get("reason", ""), sig.get("pnl", 0))

    def update_sl(self, new_sl: float) -> None:
        """Track trailing stop level (no API — exit handled by strategy logic)."""
        if not self.active: return
        self.sl_price = new_sl

    def reconcile(self) -> dict:
        """Sync executor state to exchange truth. Silent adopt — exchange wins.

        Returns truth dict for the strategy to apply to its own state:
            {"pos": int (-1/0/+1), "entry_price": float, "exchange_pl": float}
        """
        positions = _query_positions()
        mgc = next((p for p in positions if p.get("symbolId") == SYMBOL_ID), None)

        if mgc is None:
            truth = {"pos": 0, "entry_price": 0.0, "exchange_pl": 0.0}
        else:
            size = int(mgc.get("positionSize", 0))
            truth = {
                "pos": 1 if size > 0 else (-1 if size < 0 else 0),
                "entry_price": float(mgc.get("averagePrice", 0)),
                "exchange_pl": float(mgc.get("profitAndLoss", 0)),
            }

        # Sync active SL order from exchange if local sl_order_id is missing
        if truth["pos"] != 0 and not self.sl_order_id:
            try:
                # Try to find a Stop order for this symbol
                # We need to know the correct endpoint for active orders
                # Based on previous 404s, let's try to infer from typical TopstepX patterns
                # or just rely on the AUTO-PROTECT logic in the strategy if we can't find it.
                # For now, if we have a position but no ID, we'll let AUTO-PROTECT re-place it
                # to be 100% sure we have a known order ID to manage.
                pass
            except Exception:
                pass

        # Silent adopt — sync executor own state
        prev_active = self.active
        prev_side = self.pos_side
        prev_entry = self.entry_price

        self.active = truth["pos"] != 0
        self.pos_side = "Long" if truth["pos"] == 1 else ("Short" if truth["pos"] == -1 else "")
        self.entry_price = truth["entry_price"]
        if not self.active:
            self.sl_price = 0.0
            self.sl_order_id = None

        # Console log only when state actually changed
        if prev_active != self.active or prev_side != self.pos_side or abs(prev_entry - self.entry_price) > 0.5:
            print(f"[SSExec] Reconcile: was {prev_side or 'FLAT'}@{prev_entry:.1f} → "
                  f"now {self.pos_side or 'FLAT'}@{self.entry_price:.1f} "
                  f"(exchange P&L: {truth['exchange_pl']:.0f})", flush=True)

        # AUTO-PROTECT: If active but no SL order ID, place one now
        if self.active and not self.sl_order_id:
            emergency_sl = (self.entry_price - 50.0) if self.pos_side == "Long" else (self.entry_price + 50.0)
            print(f"[SSExec] AUTO-PROTECT: No SL found for {self.pos_side} position. Placing Hard SL @ {emergency_sl:.1f}...", flush=True)
            self._place_sl(emergency_sl)

        return truth

    def heartbeat(self, state: dict | None = None) -> None:
        """Send status to Telegram every 5 min. Called from super_structure.run_live.
        
        Args:
            state: dict with OHLC + indicators from super_structure._heartbeat_state
        """
        now = time.time()
        if now - self._last_heartbeat < 300: return
        self._last_heartbeat = now

        s = state or {}
        ts = s.get("ts", "")
        o = s.get("open", 0); hi = s.get("high", 0); lo = s.get("low", 0); cl = s.get("close", 0)
        pc = s.get("prev_close", 0); dema = s.get("dema", 0); pd = s.get("prev_dema", 0)
        st_val = s.get("st", 0)
        adx = s.get("adx", 0); cci = s.get("cci", 0)
        direction = s.get("direction", 0)
        pos = s.get("pos", 0)
        ADX_THR = 25; CCI_L = 100.0; CCI_S = -100.0

        # Signal status analysis
        reasons = []
        if adx <= ADX_THR: reasons.append(f"❌ ADX `{adx}` < `{ADX_THR}`")
        else: reasons.append(f"✅ ADX `{adx}` > `{ADX_THR}`")

        if cci > CCI_L: reasons.append(f"✅ CCI `{cci}` > `{CCI_L}` → LONG ok")
        elif cci < CCI_S: reasons.append(f"✅ CCI `{cci}` < `{CCI_S}` → SHORT ok")
        else: reasons.append(f"❌ CCI `{cci}` between `{CCI_S}`..`{CCI_L}` (neutral)")

        if direction > 0:
            cross_dn = (pd > 0 and pc > pd and cl < dema)
            reasons.append(f"ST dir `{direction}` → SELL bias")
            if cross_dn: reasons.append(f"✅ DEMA cross DOWN (`{pc:.1f}`→`{cl:.1f}`)")
            else: reasons.append(f"❌ DEMA no cross down (`{pc:.1f}`→`{cl:.1f}`)")
        elif direction < 0:
            cross_up = (pd > 0 and pc < pd and cl > dema)
            reasons.append(f"ST dir `{direction}` → BUY bias")
            if cross_up: reasons.append(f"✅ DEMA cross UP (`{pc:.1f}`→`{cl:.1f}`)")
            else: reasons.append(f"❌ DEMA no cross up (`{pc:.1f}`→`{cl:.1f}`)")
        else:
            reasons.append(f"ST: neutral")

        signal_analysis = "\n".join(f"• {r}" for r in reasons)

        if pos == 0:
            hdr = "💓 *Super Structure — Heartbeat*\n\n"
            lines = [
                hdr,
                f"`{ts}`" if ts else "",
                "",
                f"📊 *5m Bar*: O:`{o}` H:`{hi}` L:`{lo}` C:`{cl}`",
                f"📐 ST:`{st_val}` | DEMA:`{dema}` | ADX:`{adx}` | CCI:`{cci}`",
                "",
                f"🔍 *Signal Check*:",
                signal_analysis,
                "",
                f"Position: *FLAT*",
            ]
        else:
            side = "🟢 LONG" if pos == 1 else "🔴 SHORT"
            entry = s.get("entry_price", 0)
            sl_val = s.get("sl_price", 0)
            est_pnl = ((cl - entry) * 10 * pos - 1.74) if entry > 0 else 0
            pnl_s = f"+{est_pnl:.0f}" if est_pnl >= 0 else f"-{abs(est_pnl):.0f}"
            emoji = "✅" if est_pnl >= 0 else "❌"
            hdr = "💓 *Super Structure — Heartbeat*\n\n"

            lines = [
                hdr,
                f"`{ts}`" if ts else "",
                "",
                f"{emoji} {side} @ `${entry:.1f}`",
                f"SL: `${sl_val:.1f}` | Est PnL: `{pnl_s}`",
                "",
                f"📊 *5m Bar*: O:`{o}` H:`{hi}` L:`{lo}` C:`{cl}`",
                f"📐 ST:`{st_val}` | DEMA:`{dema}` | ADX:`{adx}` | CCI:`{cci}`",
            ]

        msg = "\n".join(l for l in lines if l)
        _send_telegram(msg)

    # ── internal ──────────────────────────────────────────────────────────

    def _enter(self, side: str, price: float, sl: float) -> None:
        if self.active:
            print(f"[SSExec] Already in position, ignoring ENTRY", flush=True)
            return
        try:
            size = 1 if side == "Buy" else -1
            result = _api({
                "accountId": ACCOUNT_ID, "symbolId": SYMBOL_ID,
                "type": 2, "limitPrice": None, "stopPrice": None,
                "positionSize": size,
                "customTag": str(uuid.uuid4())[:8], "timeType": 0,
            })
            print(f"[SSExec] MARKET {side} @ {price:.1f}: {json.dumps(result)}", flush=True)
            self.active = True
            self.pos_side = "Long" if side == "Buy" else "Short"
            self.entry_price = price
            
            # Place EMERGENCY HARD SL on exchange (50 points / $500 wide)
            # This is a safety net if the bot dies. 
            emergency_sl = (price - 50.0) if side == "Buy" else (price + 50.0)
            print(f"[SSExec] Placing emergency Hard SL @ {emergency_sl:.1f}...", flush=True)
            self._place_sl(emergency_sl)

            _send_telegram(
                f"🚀 *Super Structure — Entry Executed*\n\n"
                f"Action: {side}\n"
                f"Entry: `${price:.1f}`\n"
                f"Logic SL: `${sl:.1f}`\n"
                f"Hard SL: `${emergency_sl:.1f}` (Exchange)")
        except Exception as e:
            print(f"[SSExec] ENTRY failed: {e}", flush=True)
            _send_telegram(f"⚠️ *Entry FAILED*: {e}")

    def _place_sl(self, sl: float) -> None:
        if not self.active: return
        try:
            size = -1 if self.pos_side == "Long" else 1  # reverse side for stop
            # Using type: 4 for STOP MARKET in TopstepX API
            result = _api({
                "accountId": ACCOUNT_ID, "symbolId": SYMBOL_ID,
                "type": 4, "limitPrice": None, "stopPrice": sl,
                "positionSize": size,
                "customTag": f"SL-{uuid.uuid4().hex[:6]}", "timeType": 0,
            })
            oid = result.get("orderId") or result.get("id")
            if oid:
                self.sl_order_id = int(oid)
            else:
                print(f"[SSExec] SL response missing orderId: {json.dumps(result)[:200]}", flush=True)
            self.sl_price = sl
            print(f"[SSExec] SL placed @ {sl:.1f} (order #{self.sl_order_id})", flush=True)
        except Exception as e:
            print(f"[SSExec] SL order failed: {e}", flush=True)
            self.sl_order_id = None

    def _exit(self, price: float, reason: str, pnl: float) -> None:
        if not self.active: return
        try:
            # 1. Close position
            result = _flatten_all()
            
            # 2. Wait a bit for exchange to process, then cancel debris
            time.sleep(0.5)
            try:
                _cancel_all_orders()
                print(f"[SSExec] Cancelled exchange orders.", flush=True)
            except Exception as ce:
                print(f"[SSExec] Cancel orders failed: {ce}", flush=True)

            pnl_s = f"+{pnl:.0f}" if pnl >= 0 else f"-{abs(pnl):.0f}"
            emoji = "✅" if pnl >= 0 else "❌"
            print(f"[SSExec] FLATTEN @ market PnL={pnl_s}: {json.dumps(result)}", flush=True)
            _send_telegram(
                f"{emoji} *Super Structure — Exit Executed*\n\n"
                f"PnL: `{pnl_s}`\n"
                f"Reason: `{reason}`")
            self.active = False
            self.pos_side = ""
            self.sl_price = 0.0
            self.sl_order_id = None
            self.entry_price = 0.0
        except Exception as e:
            print(f"[SSExec] EXIT failed: {e}", flush=True)
            _send_telegram(f"⚠️ *Exit FAILED*: {e}")

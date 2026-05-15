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
EXECUTIONS_PATH = ROOT / "data" / "Live" / "super_structure_executions.jsonl"
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


def _append_execution_event(event: dict) -> None:
    """Append one structured execution event; text logs are not source of truth."""
    EXECUTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "signal_id": event.get("signal_id", ""),
        "trade_id": event.get("trade_id", ""),
        "event_type": event.get("event_type", ""),
        "action": event.get("action", ""),
        "requested_price": event.get("requested_price"),
        "executed_price": event.get("executed_price"),
        "order_id": event.get("order_id"),
        "api_result": event.get("api_result"),
        "error": event.get("error", ""),
    }
    with EXECUTIONS_PATH.open("a") as f:
        f.write(json.dumps(row, default=str, separators=(",", ":")) + "\n")


def _extract_order_id(result: dict) -> object:
    if not isinstance(result, dict):
        return None
    for key in ("orderId", "id", "order_id"):
        if result.get(key) is not None:
            return result.get(key)
    nested = result.get("order")
    if isinstance(nested, dict):
        for key in ("orderId", "id", "order_id"):
            if nested.get(key) is not None:
                return nested.get(key)
    return None


def _extract_executed_price(result: dict) -> float | None:
    if not isinstance(result, dict):
        return None
    for key in ("executedPrice", "filledPrice", "averagePrice", "avgFillPrice", "price"):
        value = result.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    for nested_key in ("order", "position", "fill"):
        nested = result.get(nested_key)
        if isinstance(nested, dict):
            px = _extract_executed_price(nested)
            if px is not None:
                return px
    return None

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

def _send_telegram(msg: str) -> bool:
    try:
        env = ROOT / "data" / "Live" / "telegram.env"
        if not env.exists():
            return False
        token = chat = ""
        for line in env.read_text().strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                if k == "TELEGRAM_BOT_TOKEN": token = v
                elif k == "TELEGRAM_CHAT_ID": chat = v
        if not (token and chat):
            return False
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat, "text": msg,
                                        "parse_mode": "Markdown"}).encode()
        urllib.request.urlopen(url, data, timeout=10)
        return True
    except Exception as exc:
        print(f"[SSExec] _send_telegram error: {exc}", flush=True)
        return False


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


def _query_positions() -> list[dict] | None:
    """List open positions for the user.

    Empty list means the exchange confirmed FLAT.
    None means the exchange state is unknown and must not be treated as FLAT.
    """
    try:
        return _api_get(f"/Position/all/user/{USER_ID}") or []
    except Exception as exc:
        print(f"[SSExec] Position query failed: {exc}", flush=True)
        return None


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
        self.exchange_state_known = False
        self.last_reconcile_error = ""
        self._last_heartbeat = 0

    def on_signal(self, sig: dict) -> None:
        """Route signal to order placement. Called from super_structure._store_signal."""
        action = sig.get("action", "")
        price = sig.get("price", 0)
        sl = sig.get("sl", 0)

        if action in ("BUY", "SELL") and not self.exchange_state_known:
            print(f"[SSExec] Exchange state UNKNOWN — ignoring {action} ENTRY", flush=True)
            return

        if action == "BUY":
            self._enter("Buy", price, sl, sig)
        elif action == "SELL":
            self._enter("Sell", price, sl, sig)
        elif action == "CLOSE":
            self._exit(price, sig.get("reason", ""), sig.get("pnl", 0), sig)

    def update_sl(self, new_sl: float) -> None:
        """Track trailing stop level (no API — exit handled by strategy logic)."""
        if not self.active: return
        self.sl_price = new_sl

    def reconcile(self) -> dict:
        """Sync executor state to exchange truth. Silent adopt — exchange wins.

        Returns truth dict for the strategy to apply to its own state.
        If ok=False, callers must preserve their last known position state and
        block new entries until a later reconcile succeeds.
        """
        positions = _query_positions()
        if positions is None:
            self.exchange_state_known = False
            self.last_reconcile_error = "position_query_failed"
            return {
                "ok": False,
                "exchange_state_known": False,
                "error": self.last_reconcile_error,
                "pos": 1 if self.pos_side == "Long" else (-1 if self.pos_side == "Short" else 0),
                "entry_price": self.entry_price,
                "exchange_pl": 0.0,
            }

        self.exchange_state_known = True
        self.last_reconcile_error = ""
        mgc = next((p for p in positions if p.get("symbolId") == SYMBOL_ID), None)

        if mgc is None:
            truth = {"ok": True, "exchange_state_known": True, "pos": 0, "entry_price": 0.0, "exchange_pl": 0.0}
        else:
            size = int(mgc.get("positionSize", 0))
            truth = {
                "ok": True,
                "exchange_state_known": True,
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
        if now - self._last_heartbeat < 300:
            return
        # Defer updating _last_heartbeat until after a successful send, so a
        # transient Telegram/SSL failure doesn't lock out heartbeats for 5 min.

        s = state or {}
        ts = s.get("ts", "")
        o = s.get("open", 0); hi = s.get("high", 0); lo = s.get("low", 0); cl = s.get("close", 0)
        pc = s.get("prev_close", 0); dema = s.get("dema", 0); pd = s.get("prev_dema", 0)
        st_val = s.get("st", 0)
        adx = s.get("adx", 0); cci = s.get("cci", 0)
        direction = s.get("direction", 0)
        pos = s.get("pos", 0)
        exchange_known = s.get("exchange_state_known", True)
        exchange_error = s.get("exchange_state_error", "")
        ADX_THR = 25; CCI_L = 100.0; CCI_S = -100.0

        # Signal status analysis
        reasons = []
        if not exchange_known:
            err = f" ({exchange_error})" if exchange_error else ""
            reasons.append(f"⛔ Exchange state UNKNOWN{err} — entries blocked")

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

        # ── V8 router section ────────────────────────────────────────────────
        v8_block: list[str] = []
        if s.get("v8_active"):
            mode = s.get("position_mode", "") or "FLAT"
            daily_pnl_v = float(s.get("daily_pnl", 0.0) or 0.0)
            daily_cap_v = float(s.get("daily_cap", -700.0) or -700.0)
            headroom = daily_pnl_v - daily_cap_v
            cap_emoji = "🛡️" if headroom > 200 else ("⚠️" if headroom > 0 else "🛑")
            risk_cap = s.get("risk_cap_pts", 12.0)
            last = s.get("last_v8_decision") or {}
            v8_block.append("")
            v8_block.append("🧠 *V8 Router*:")
            v8_block.append(f"• Mode: `{mode}` | Risk cap: `{risk_cap}` pts")
            v8_block.append(f"• Daily PnL: `${daily_pnl_v:+.2f}` / cap `${daily_cap_v:.0f}` "
                            f"({cap_emoji} headroom `${headroom:+.0f}`)")
            if last:
                path = last.get("path", "?")
                take = "PASS" if last.get("take") else "SKIP"
                reason = last.get("reason", "")
                if path == "CONS":
                    extra = (f"prob `{last.get('prob', 0):.3f}` "
                             f"thr `{last.get('threshold', 0):.2f}`")
                else:
                    extra = f"risk `{last.get('risk_pts', 0):.1f}` side `{last.get('side', '?')}`"
                v8_block.append(f"• Last {path}: `{take}` ({reason}) {extra}")

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
                *v8_block,
                "",
                f"Position: *FLAT*",
                f"Exchange: *{'KNOWN' if exchange_known else 'UNKNOWN'}*",
            ]
        else:
            side = "🟢 LONG" if pos == 1 else "🔴 SHORT"
            entry = s.get("entry_price", 0)
            sl_val = s.get("sl_price", 0)
            est_pnl = ((cl - entry) * 10 * pos - 1.74) if entry > 0 else 0
            pnl_s = f"+{est_pnl:.0f}" if est_pnl >= 0 else f"-{abs(est_pnl):.0f}"
            emoji = "✅" if est_pnl >= 0 else "❌"
            hdr = "💓 *Super Structure — Heartbeat*\n\n"

            mode_chip = ""
            tp_val = float(s.get("tp_price", 0.0) or 0.0)
            if s.get("v8_active") and s.get("position_mode"):
                mode_chip = f" [{s['position_mode']}]"
            tp_line = f"TP: `${tp_val:.1f}` | " if tp_val > 0 else ""

            lines = [
                hdr,
                f"`{ts}`" if ts else "",
                "",
                f"{emoji} {side}{mode_chip} @ `${entry:.1f}`",
                f"{tp_line}SL: `${sl_val:.1f}` | Est PnL: `{pnl_s}`",
                "",
                f"📊 *5m Bar*: O:`{o}` H:`{hi}` L:`{lo}` C:`{cl}`",
                f"📐 ST:`{st_val}` | DEMA:`{dema}` | ADX:`{adx}` | CCI:`{cci}`",
                *v8_block,
                f"Exchange: *{'KNOWN' if exchange_known else 'UNKNOWN'}*",
            ]

        msg = "\n".join(l for l in lines if l)
        sent = _send_telegram(msg)
        if sent:
            self._last_heartbeat = now
            print(f"[SSExec] 💓 Heartbeat sent (pos={pos}, mode={s.get('position_mode') or 'FLAT'})",
                  flush=True)
        else:
            # Telegram failed (SSL/network). Leave _last_heartbeat unchanged
            # so the next loop iteration retries instead of waiting 5 min.
            print(f"[SSExec] ⚠️ Heartbeat send FAILED — will retry next cycle",
                  flush=True)

    # ── internal ──────────────────────────────────────────────────────────

    def _enter(self, side: str, price: float, sl: float, sig: dict | None = None) -> None:
        if self.active:
            print(f"[SSExec] Already in position, ignoring ENTRY", flush=True)
            return
        sig = sig or {}
        try:
            size = 1 if side == "Buy" else -1
            result = _api({
                "accountId": ACCOUNT_ID, "symbolId": SYMBOL_ID,
                "type": 2, "limitPrice": None, "stopPrice": None,
                "positionSize": size,
                "customTag": str(uuid.uuid4())[:8], "timeType": 0,
            })
            _append_execution_event({
                "signal_id": sig.get("signal_id", ""),
                "trade_id": sig.get("trade_id", ""),
                "event_type": "MARKET",
                "action": side.upper(),
                "requested_price": price,
                "executed_price": _extract_executed_price(result),
                "order_id": _extract_order_id(result),
                "api_result": result,
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
            _append_execution_event({
                "signal_id": sig.get("signal_id", ""),
                "trade_id": sig.get("trade_id", ""),
                "event_type": "MARKET",
                "action": side.upper(),
                "requested_price": price,
                "executed_price": None,
                "order_id": None,
                "api_result": None,
                "error": str(e),
            })
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

    def _exit(self, price: float, reason: str, pnl: float, sig: dict | None = None) -> None:
        if not self.active: return
        sig = sig or {}
        try:
            # 1. Close position
            result = _flatten_all()
            _append_execution_event({
                "signal_id": sig.get("signal_id", ""),
                "trade_id": sig.get("trade_id", ""),
                "event_type": "FLATTEN",
                "action": "CLOSE",
                "requested_price": price,
                "executed_price": _extract_executed_price(result),
                "order_id": _extract_order_id(result),
                "api_result": result,
            })
            
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
            _append_execution_event({
                "signal_id": sig.get("signal_id", ""),
                "trade_id": sig.get("trade_id", ""),
                "event_type": "FLATTEN",
                "action": "CLOSE",
                "requested_price": price,
                "executed_price": None,
                "order_id": None,
                "api_result": None,
                "error": str(e),
            })
            print(f"[SSExec] EXIT failed: {e}", flush=True)
            _send_telegram(f"⚠️ *Exit FAILED*: {e}")

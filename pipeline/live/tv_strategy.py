#!/usr/bin/env python3
"""
TradingView Strategy Port — Python implementation.
Supertrend + DEMA + ADX + CCI + Session Filter.

Runs on buffer data (topstepx_buffer.db), generates signals in
the same format as webhook parser (action=BUY/SELL/CLOSE).

Usage:
    python3 pipeline/live/tv_strategy.py [--live]
"""

from __future__ import annotations

import json
import time as _time
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
import sys
for p in [str(ROOT), str(ROOT / "pipeline")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.live.signal_bus import SignalBus

# ── config ────────────────────────────────────────────────────────────────────
ATR_PERIOD = 12
ST_FACTOR = 4.0
DEMA_LENGTH = 200
ADX_LENGTH = 12
ADX_THRESHOLD = 25
CCI_LENGTH = 12
CCI_SOURCE = "hl2"
CCI_LONG_MIN = 100.0
CCI_SHORT_MAX = -100.0
USE_ADX = True
USE_CCI = True
USE_SESSION = False
SYMBOL = "MGC"
# NY session 08:00-15:00 EST = 13:00-20:00 UTC
SESSION_START = time(13, 0)
SESSION_END = time(20, 0)
AUTO_CLOSE = True

SIGNALS_PATH = ROOT / "data" / "Live" / "tv_signals.json"
EPS = 1e-10


# ── indicators ────────────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=period, adjust=False).mean().values


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
    tr = np.maximum(h - l,
                    np.maximum(np.abs(h - np.roll(c, 1)),
                               np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return _rma(tr, period)


def _rma(arr: np.ndarray, period: int) -> np.ndarray:
    """TradingView-style RMA/Wilder smoothing used by ta.atr()."""
    out = np.full(len(arr), np.nan)
    if len(arr) < period:
        return out
    out[period - 1] = np.nanmean(arr[:period])
    for i in range(period, len(arr)):
        out[i] = (out[i - 1] * (period - 1) + arr[i]) / period
    return out


def supertrend(h: np.ndarray, l: np.ndarray, c: np.ndarray,
               factor: float = 4.0, atr_period: int = 10) -> np.ndarray:
    """TradingView-compatible Supertrend.
    Returns: st_values (float array), direction (-1=UP,+1=DOWN).
    """
    n = len(c)
    atr_val = _atr(h, l, c, atr_period)
    hl2 = (h + l) / 2.0
    upper = hl2 + factor * atr_val
    lower = hl2 - factor * atr_val
    st = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)

    for i in range(n):
        if np.isnan(atr_val[i]):
            continue
        if i == 0 or np.isnan(atr_val[i - 1]):
            direction[i] = 1
            st[i] = upper[i]
            continue

        prev_lower = lower[i - 1]
        prev_upper = upper[i - 1]

        if not (lower[i] > prev_lower or c[i - 1] < prev_lower):
            lower[i] = prev_lower
        if not (upper[i] < prev_upper or c[i - 1] > prev_upper):
            upper[i] = prev_upper

        prev_st = st[i - 1]
        if np.isnan(prev_st) or np.isclose(prev_st, prev_upper, equal_nan=False):
            direction[i] = -1 if c[i] > upper[i] else 1
        else:
            direction[i] = 1 if c[i] < lower[i] else -1

        st[i] = lower[i] if direction[i] == -1 else upper[i]

    return st, direction


def dema(c: np.ndarray, period: int) -> np.ndarray:
    e1 = _ema(c, period)
    e2 = _ema(e1, period)
    return 2.0 * e1 - e2


def adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
    """TradingView ta.dmi() ADX using Wilder/RMA smoothing."""
    up = np.full(len(c), np.nan)
    dn = np.full(len(c), np.nan)
    tr_val = np.full(len(c), np.nan)
    up[1:] = h[1:] - h[:-1]
    dn[1:] = l[:-1] - l[1:]

    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdm[0] = np.nan
    ndm[0] = np.nan
    tr_val[1:] = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])),
    )

    atr_s = _rma(tr_val, period)
    pdi = 100 * _rma(pdm, period) / (atr_s + EPS)
    ndi = 100 * _rma(ndm, period) / (atr_s + EPS)
    denom = pdi + ndi + EPS
    dx = 100 * np.abs(pdi - ndi) / denom
    adx_val = np.full(len(c), np.nan)
    valid = np.where(~np.isnan(dx))[0]
    if len(valid) < period:
        return adx_val
    start = valid[period - 1]
    adx_val[start] = np.nanmean(dx[valid[:period]])
    for i in range(start + 1, len(c)):
        adx_val[i] = adx_val[i - 1] if np.isnan(dx[i]) else (adx_val[i - 1] * (period - 1) + dx[i]) / period
    return adx_val


def cci(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int,
        source: str = "hl2") -> np.ndarray:
    """Commodity Channel Index. source: 'hlc3' or 'hl2'."""
    if source == "hl2":
        tp = (h + l) / 2.0
    else:
        tp = (h + l + c) / 3.0
    sma = pd.Series(tp).rolling(window=period).mean().values
    md = np.full(len(tp), np.nan)
    for i in range(period - 1, len(tp)):
        mean = sma[i]
        md[i] = np.mean(np.abs(tp[i - period + 1:i + 1] - mean))
    result = (tp - sma) / (0.015 * md + EPS)
    return result


# ── strategy runner ───────────────────────────────────────────────────────────

class TVStrategy:
    """Replicate TradingView strategy in Python using buffer data."""

    def __init__(self, buffer_or_db_path=None):
        from pipeline.live.buffer import DataBuffer, CANARY_DB
        if buffer_or_db_path is None:
            self.buffer = DataBuffer(db_path=CANARY_DB)
        elif isinstance(buffer_or_db_path, Path):
            self.buffer = DataBuffer(db_path=buffer_or_db_path)
        else:
            self.buffer = buffer_or_db_path

        self._pos = 0
        self._entry_price = 0.0
        self._sl_price = 0.0
        self._last_ts: pd.Timestamp | None = None
        self._last_checked_now: datetime | None = None
        self._cached_df: pd.DataFrame | None = None
        self._cached_end: str | None = None
        self._signals: list[dict] = []
        self._entry_bar_ts: pd.Timestamp | None = None
        self._load_signals()

    def _load_signals(self) -> None:
        if SIGNALS_PATH.exists():
            try:
                self._signals = json.loads(SIGNALS_PATH.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                self._signals = []

    def _save_signals(self) -> None:
        SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SIGNALS_PATH.write_text(json.dumps(self._signals, indent=2, default=str))

    def _store_signal(self, action: str, price: float, sl: float = 0.0,
                       reason: str = "", **extra) -> None:
        sig = {
            "action": action,
            "symbol": SYMBOL,
            "price": price,
            "sl": sl,
            "reason": reason,
            "ts": extra.get("ts", ""),
            "adx": extra.get("adx", 0),
            "cci": extra.get("cci", 0),
            "dema": extra.get("dema", 0),
            "parsed_from": f"tv_strategy.py: {action} {SYMBOL} @ {price:.1f}",
        }
        entry = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "signal": sig,
        }
        self._signals.append(entry)
        self._save_signals()
        print(f"[TV] ⚡ SIGNAL: {action} {SYMBOL} @ {price:.1f}" +
              (f" SL={sl:.1f}" if sl > 0 else "") +
              (f" ({reason})" if reason else ""), flush=True)

        bus_payload = {"action": action, "symbol": SYMBOL, "price": price,
                       "sl": sl, "reason": reason, **extra}
        if action == "CLOSE" and self._pos != 0:
            side = 1 if self._pos == 1 else -1
            pnl = round((price - self._entry_price) * side * 10 - 3, 2)
            bus_payload["pnl"] = pnl
            bus_payload["side"] = "Long" if self._pos == 1 else "Short"
        try:
            SignalBus().publish("tv_strategy", bus_payload)
        except Exception:
            pass

    def _is_in_session(self, ts: pd.Timestamp) -> bool:
        if not USE_SESSION:
            return True
        t = ts.tz_convert("America/New_York").time()
        # Simple check: hour-based
        return SESSION_START <= ts.tz_convert("UTC").time() <= SESSION_END

    def _is_end_of_session(self, ts: pd.Timestamp) -> bool:
        if not USE_SESSION or not AUTO_CLOSE:
            return False
        return ts.tz_convert("UTC").time() >= SESSION_END

    def _session_name(self, ts: pd.Timestamp) -> str:
        """Label session for display only (no gating)."""
        t = ts.tz_convert("UTC")
        h = t.hour + t.minute / 60.0
        if 0 <= h < 3:
            return "Tokyo"
        if 7 <= h < 10:
            return "London"
        if 13.5 <= h < 16.5:
            return "US"
        return "Other"

    def check(self, now: "datetime | None" = None) -> list[dict]:
        """Fetch latest data and check for signals. Returns new signals."""
        now = now or datetime.now(timezone.utc)

        # Fast path: skip if called again within 30s of last check
        if self._last_checked_now is not None:
            if (now - self._last_checked_now).total_seconds() < 30:
                return []
        self._last_checked_now = now

        start = (now - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
        end = now.strftime("%Y-%m-%d %H:%M:%S")

        # Incremental fetch: only load new rows since last check
        if self._cached_df is not None and self._cached_end is not None:
            if end > self._cached_end:
                new_rows = self.buffer.get(self._cached_end, end)
                if not new_rows.empty:
                    new_rows["timestamp_utc"] = pd.to_datetime(new_rows["timestamp_utc"], utc=True)
                    self._cached_df = pd.concat([self._cached_df, new_rows])
                    cutoff_ts = pd.Timestamp(now - timedelta(days=90))
                    self._cached_df = self._cached_df[
                        self._cached_df["timestamp_utc"] >= cutoff_ts
                    ]
            self._cached_end = end
        else:
            self._cached_df = self.buffer.get(start, end)
            self._cached_end = end

        df = self._cached_df
        if len(df) < max(DEMA_LENGTH, ADX_LENGTH + 50, CCI_LENGTH + 50):
            return []

        # Resample 1m to 5m (TradingView chart timeframe)
        df = df.set_index("timestamp_utc").sort_index()
        df_5m = df.resample("5min", label="right", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

        last_ts = df_5m.index[-1]
        if last_ts == self._last_ts:
            return []  # No new bar — skip indicator compute
        self._last_ts = last_ts

        if len(df_5m) < 50:
            return []
        df = df_5m
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)

        st, direction = supertrend(h, l, c, ST_FACTOR, ATR_PERIOD)
        d = dema(c, DEMA_LENGTH)
        ax = adx(h, l, c, ADX_LENGTH)
        cx = cci(h, l, c, CCI_LENGTH, source=CCI_SOURCE)

        i = len(c) - 1
        if i < DEMA_LENGTH:
            return []

        cur_close = c[i]
        cur_dema = d[i]
        cur_dir = direction[i]
        cur_adx = ax[i] if i < len(ax) and not np.isnan(ax[i]) else 0
        cur_cci = cx[i]
        cur_st = st[i]

        adx_ok = not USE_ADX or cur_adx > ADX_THRESHOLD
        cci_ok_long = not USE_CCI or cur_cci > CCI_LONG_MIN
        cci_ok_short = not USE_CCI or cur_cci < CCI_SHORT_MAX
        session_label = self._session_name(last_ts)

        new_signals = []

        # ── Entry ──────────────────────────────────────────────────────────
        cross_up = (c[i-1] < d[i-1] and cur_close > cur_dema)
        cross_dn = (c[i-1] > d[i-1] and cur_close < cur_dema)
        long_signal = adx_ok and cci_ok_long and \
                    (cross_up or cur_close > cur_dema) and cur_dir < 0
        short_signal = adx_ok and cci_ok_short and \
                      (cross_dn or cur_close < cur_dema) and cur_dir > 0

        # ── Existing Stop Orders ───────────────────────────────────────────
        if self._pos == 1:
            if l[i] <= self._sl_price:
                self._store_signal("CLOSE", self._sl_price, reason="SL",
                                   ts=str(last_ts))
                new_signals.append(self._signals[-1])
                self._pos = 0

        if self._pos == -1:
            if h[i] >= self._sl_price:
                self._store_signal("CLOSE", self._sl_price, reason="SL",
                                   ts=str(last_ts))
                new_signals.append(self._signals[-1])
                self._pos = 0

        # ── Trend flip ─────────────────────────────────────────────────────
        if self._pos == 1 and cur_dir > 0:
            self._store_signal("CLOSE", cur_close, reason="TREND_FLIP",
                               ts=str(last_ts))
            new_signals.append(self._signals[-1])
            self._pos = 0

        elif self._pos == -1 and cur_dir < 0:
            self._store_signal("CLOSE", cur_close, reason="TREND_FLIP",
                               ts=str(last_ts))
            new_signals.append(self._signals[-1])
            self._pos = 0

        if self._pos == -1:
            if h[i] >= self._sl_price:
                self._store_signal("CLOSE", self._sl_price, reason="SL")
                new_signals.append(self._signals[-1])
                self._pos = 0

        # ── Trend flip ─────────────────────────────────────────────────────
        if self._pos == 1 and cur_dir > 0:
            self._store_signal("CLOSE", cur_close, reason="TREND_FLIP")
            new_signals.append(self._signals[-1])
            self._pos = 0

        elif self._pos == -1 and cur_dir < 0:
            self._store_signal("CLOSE", cur_close, reason="TREND_FLIP")
            new_signals.append(self._signals[-1])
            self._pos = 0

        # Submit/update the dynamic stop for the next bar.
        if self._pos != 0:
            self._sl_price = cur_st

        # ── Entry ──────────────────────────────────────────────────────────
        if long_signal and self._pos == 0:
            self._pos = 1
            self._entry_price = cur_close
            self._sl_price = cur_st
            self._store_signal("BUY", cur_close, cur_st,
                               adx=round(cur_adx, 1) if not np.isnan(cur_adx) else 0,
                               cci=round(cur_cci, 0) if not np.isnan(cur_cci) else 0,
                               dema=round(cur_dema, 1),
                               ts=str(last_ts))
            new_signals.append(self._signals[-1])

        if short_signal and self._pos == 0:
            self._pos = -1
            self._entry_price = cur_close
            self._sl_price = cur_st
            self._store_signal("SELL", cur_close, cur_st,
                               adx=round(cur_adx, 1) if not np.isnan(cur_adx) else 0,
                               cci=round(cur_cci, 0) if not np.isnan(cur_cci) else 0,
                               dema=round(cur_dema, 1),
                               ts=str(last_ts))
            new_signals.append(self._signals[-1])

        return new_signals

    def stats(self) -> str:
        n = len(self._signals)
        buys = sum(1 for s in self._signals if s["signal"]["action"] == "BUY")
        sells = sum(1 for s in self._signals if s["signal"]["action"] == "SELL")
        closes = n - buys - sells
        pos = "LONG" if self._pos == 1 else "SHORT" if self._pos == -1 else "FLAT"
        return (f"[TV Strategy] {n} signals ({buys}B {sells}S {closes}C) | "
                f"Pos: {pos} | Entry: {self._entry_price:.1f}")

    def run_live(self) -> None:
        """Main loop — check every 30 seconds, poll Telegram commands."""
        import urllib.request, urllib.parse

        token, chat_id = "", ""
        env_file = ROOT / "data" / "Live" / "telegram.env"
        if env_file.exists():
            for line in env_file.read_text().strip().split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k == "TELEGRAM_BOT_TOKEN":
                        token = v
                    elif k == "TELEGRAM_CHAT_ID":
                        chat_id = v

        bot_enabled = bool(token and chat_id)
        last_update_id = 0

        if bot_enabled:
            try:
                url = f"https://api.telegram.org/bot{token}/getUpdates"
                resp = urllib.request.urlopen(url, timeout=5)
                result = json.loads(resp.read())
                if result.get("ok") and result.get("result"):
                    last_update_id = result["result"][-1]["update_id"]
            except Exception:
                pass

        def tg_send(msg: str) -> None:
            try:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                data = urllib.parse.urlencode({"chat_id": chat_id, "text": msg,
                                                "parse_mode": "Markdown"}).encode()
                resp = urllib.request.urlopen(url, data, timeout=5)
                result = json.loads(resp.read())
                if not result.get("ok"):
                    print(f"[TV] Telegram send failed: {result}", flush=True)
            except Exception as e:
                print(f"[TV] Telegram send error: {e}", flush=True)

        def tg_handle(cmd: str) -> str | None:
            cmd = cmd.strip().lower()
            if cmd == "/strat":
                from pipeline.live.user_db import get_subscriptions_by_chat
                subs = get_subscriptions_by_chat(str(chat_id))
                if subs:
                    lines = ["📋 *Subscriptions:*\n"]
                    for s in subs:
                        name = "Super Structure" if s == "tv_strategy" else s
                        lines.append(f"• {name}")
                    lines.append("\nUse /strat on <name> or /strat off <name>")
                    return "\n".join(lines)
                return "📋 *No subscriptions.*\n\nUse /strat on tv_strategy or /strat off tv_strategy"

            elif cmd.startswith("/strat on ") or cmd.startswith("/strat off "):
                parts = cmd.split()
                if len(parts) < 3:
                    return "Usage: /strat on <name> or /strat off <name>"
                action = parts[1]
                name = parts[2]
                if action == "on":
                    from pipeline.live.user_db import subscribe_by_chat as _sub
                    _sub(str(chat_id), name)
                    display = "Super Structure" if name == "tv_strategy" else name
                    return f"✅ Subscribed to {display}"
                elif action == "off":
                    from pipeline.live.user_db import unsubscribe_by_chat as _unsub
                    _unsub(str(chat_id), name)
                    display = "Super Structure" if name == "tv_strategy" else name
                    return f"❌ Unsubscribed from {display}"
                return "Unknown action. Use /strat on <name> or /strat off <name>"

            elif cmd == "/tv" or cmd == "/tv_status":
                pos = "LONG" if self._pos == 1 else "SHORT" if self._pos == -1 else "FLAT"
                entry = f"${self._entry_price:.1f}" if self._pos != 0 else "—"
                sl = f"${self._sl_price:.1f}" if self._pos != 0 else "—"
                sig_count = len(self._signals)
                buy = sum(1 for x in self._signals if x["signal"]["action"] == "BUY")
                sell = sum(1 for x in self._signals if x["signal"]["action"] == "SELL")
                closes = sum(1 for x in self._signals if x["signal"]["action"] == "CLOSE")
                return (f"📊 *Super Structure Status*\n\n"
                        f"Position: {pos}\n"
                        f"Entry: {entry}  |  SL: {sl}\n"
                        f"Signals: {sig_count} total ({buy}B {sell}S {closes}C)\n"
                        f"Config: ST({ST_FACTOR},{ATR_PERIOD}) DEMA({DEMA_LENGTH}) "
                        f"ADX({ADX_LENGTH}>{ADX_THRESHOLD}) CCI({CCI_LENGTH})")

            return None

        print(f"[TV] Starting live strategy loop...", flush=True)
        print(f"[TV] Config: ST({ST_FACTOR},{ATR_PERIOD}) DEMA({DEMA_LENGTH}) "
              f"ADX({ADX_LENGTH}>{ADX_THRESHOLD}) CCI({CCI_LENGTH})", flush=True)
        if bot_enabled:
            print(f"[TV] Telegram commands: /strat /tv", flush=True)
        while True:
            try:
                signals = self.check()
                for s in signals:
                    pass  # signals already stored + printed in _store_signal
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[TV] Error: {e}", flush=True)

            # Poll Telegram commands
            if bot_enabled:
                try:
                    url = f"https://api.telegram.org/bot{token}/getUpdates"
                    params = urllib.parse.urlencode({
                        "offset": max(1, last_update_id + 1),
                        "timeout": 0,
                    })
                    resp = urllib.request.urlopen(f"{url}?{params}", timeout=5)
                    result = json.loads(resp.read())
                    if result.get("ok") and result.get("result"):
                        for update in result["result"]:
                            last_update_id = update["update_id"]
                            msg = update.get("message", {}).get("text", "")
                            chat = str(update.get("message", {}).get("chat", {}).get("id", ""))
                            if msg and chat:
                                print(f"[TV] Command: {msg}", flush=True)
                                try:
                                    reply = tg_handle(msg)
                                except Exception as ex:
                                    print(f"[TV] tg_handle error: {ex}", flush=True)
                                    reply = None
                                if reply:
                                    print(f"[TV] Reply: {reply[:100]}", flush=True)
                                    tg_send(reply)
                                else:
                                    print(f"[TV] No reply", flush=True)
                except Exception as e:
                    print(f"[TV] Poll error: {e}", flush=True)

                _time.sleep(30)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TV Strategy Python Port")
    parser.add_argument("--live", action="store_true", help="Run live monitoring loop")
    parser.add_argument("--test", action="store_true", help="Single check + print status")
    args = parser.parse_args()

    strategy = TVStrategy()

    if args.test:
        signals = strategy.check()
        print(f"[TV] Checked: {len(signals)} new signals")
        print(strategy.stats())

    if args.live:
        strategy.run_live()

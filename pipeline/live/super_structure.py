#!/usr/bin/env python3
"""
Super Structure Strategy — Python implementation (ported from TradingView Pine).
Supertrend + DEMA + ADX + CCI + Session Filter.

Runs on buffer data (topstepx_buffer.db), generates signals in
the same format as webhook parser (action=BUY/SELL/CLOSE).

Usage:
    python3 pipeline/live/super_structure.py [--live]
"""

from __future__ import annotations

import json
import hashlib
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

SIGNALS_PATH = ROOT / "data" / "Live" / "super_structure_signals.json"
STATE_PATH = ROOT / "data" / "Live" / "super_structure_state.json"
EPS = 1e-10


def _signal_price(sig: dict) -> float:
    """Support old webhook entries that used `entry` instead of `price`."""
    try:
        return float(sig.get("price", sig.get("entry", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def derive_signal_id(sig: dict, received_at: str = "", sequence: int = 0) -> str:
    """Deterministic ID for new signals and historical fallback parsing."""
    existing = sig.get("signal_id")
    if existing:
        return str(existing)
    ts = str(sig.get("ts") or received_at or "")
    action = str(sig.get("action") or "")
    price = round(_signal_price(sig), 4)
    return _stable_id("sssig", ts, action, price, sequence)


def derive_trade_id_from_entry(sig: dict, received_at: str = "", sequence: int = 0) -> str:
    existing = sig.get("trade_id")
    if existing:
        return str(existing)
    signal_id = derive_signal_id(sig, received_at, sequence)
    return _stable_id("sstrade", signal_id)


def _to_utc_timestamp(value) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")
    except Exception:
        return None


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

class SuperStructure:
    """Super Structure strategy (Supertrend + DEMA + ADX + CCI), ported from TradingView Pine."""

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
        self._sl_order_id = None
        self._last_ts: pd.Timestamp | None = None
        self._last_checked_now: datetime | None = None
        self._cached_df: pd.DataFrame | None = None
        self._cached_end: str | None = None
        self._signals: list[dict] = []
        self._entry_bar_ts: pd.Timestamp | None = None
        self._trade_id = ""
        self._last_processed_bar_ts: pd.Timestamp | None = None
        self._executor = None
        self._heartbeat_state: dict = {}
        self._exchange_state_known = True
        self._exchange_state_error = ""
        self._last_blocked_trade_key = ""
        self._manual_close_block_action = ""
        self._halt = False
        self._halt_reason = ""
        self._load_signals()
        self._load_state()
        self._ensure_runtime_ids()

        # Init trade executor (auto-detect if credentials exist)
        if (ROOT / "data" / "Live" / "topstepx_token.json").exists():
            self._exchange_state_known = False
            self._exchange_state_error = "startup_reconcile_pending"
            try:
                from pipeline.live.execute.super_structure_executor import SuperStructureExecutor
                self._executor = SuperStructureExecutor()
                # Restore sl_order_id to executor if we just loaded it from state
                if self._sl_order_id:
                    self._executor.sl_order_id = self._sl_order_id
                print("[SS] Trade executor initialized", flush=True)
                # Reconcile from exchange at startup — fixes silent desync after restart
                self.reconcile()
                # Validate session (warn if token expiring soon)
                try:
                    from pipeline.live.execute.super_structure_executor import _validate_session
                    if not _validate_session():
                        print("[SS] ⚠️ Session validate FAILED — token may be expired", flush=True)
                except Exception:
                    pass
            except Exception as exc:
                print(f"[SS] Executor init failed: {exc}", flush=True)

    def _load_signals(self) -> None:
        if SIGNALS_PATH.exists():
            try:
                self._signals = json.loads(SIGNALS_PATH.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                self._signals = []

    def _save_signals(self) -> None:
        SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SIGNALS_PATH.write_text(json.dumps(self._signals, indent=2, default=str))

    def _load_state(self) -> None:
        """Load persisted state (halt flag, position, entry, SL).
        Note: Position truth is reconciled from exchange in __init__ if executor exists.
        """
        if STATE_PATH.exists():
            try:
                d = json.loads(STATE_PATH.read_text())
                self._halt = bool(d.get("halt", False))
                self._halt_reason = str(d.get("halt_reason", ""))
                self._pos = int(d.get("pos", 0))
                self._entry_price = float(d.get("entry_price", 0.0))
                self._sl_price = float(d.get("sl_price", 0.0))
                self._sl_order_id = d.get("sl_order_id")
                self._trade_id = str(d.get("trade_id", "") or "")
                self._last_processed_bar_ts = _to_utc_timestamp(d.get("last_processed_bar_ts"))
                self._manual_close_block_action = str(d.get("manual_close_block_action", "") or "")
                if self._halt:
                    print(f"[SS] Loaded HALT state: {self._halt_reason}", flush=True)
                if self._pos != 0:
                    print(f"[SS] Loaded POSITION state: {self._pos} @ {self._entry_price:.1f} (SL: {self._sl_price:.1f}, Order: {self._sl_order_id})", flush=True)
            except Exception as e:
                print(f"[SS] Error loading state: {e}", flush=True)

    def _save_state(self) -> None:
        """Atomic write of current state to prevent corruption."""
        import os
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = STATE_PATH.with_suffix(".tmp")
        
        # Sync order ID from executor if active
        if self._executor and hasattr(self._executor, "sl_order_id"):
            self._sl_order_id = self._executor.sl_order_id

        try:
            data = {
                "halt": self._halt,
                "halt_reason": self._halt_reason,
                "pos": self._pos,
                "entry_price": self._entry_price,
                "sl_price": self._sl_price,
                "sl_order_id": self._sl_order_id,
                "trade_id": self._trade_id,
                "last_processed_bar_ts": (
                    self._last_processed_bar_ts.isoformat()
                    if self._last_processed_bar_ts is not None else ""
                ),
                "manual_close_block_action": self._manual_close_block_action,
                "exchange_state_known": self._exchange_state_known,
                "exchange_state_error": self._exchange_state_error,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            tmp_path.write_text(json.dumps(data, indent=2))
            os.replace(tmp_path, STATE_PATH)
        except Exception as e:
            print(f"[SS] Error saving state: {e}", flush=True)
            if tmp_path.exists():
                try: os.remove(tmp_path)
                except: pass

    def reconcile(self) -> dict:
        """Pull truth from exchange, silently adopt. Returns truth dict."""
        if not self._executor:
            self._exchange_state_known = True
            self._exchange_state_error = ""
            return {}
        try:
            truth = self._executor.reconcile()
        except Exception as exc:
            print(f"[SS] Reconcile failed: {exc}", flush=True)
            self._exchange_state_known = False
            self._exchange_state_error = str(exc)
            self._save_state()
            return {"ok": False, "exchange_state_known": False, "error": str(exc)}
        if not truth.get("ok", True):
            self._exchange_state_known = False
            self._exchange_state_error = str(truth.get("error", "exchange_state_unknown"))
            print(f"[SS] Exchange state UNKNOWN — preserving local pos={self._pos} entry={self._entry_price:.1f}; entries blocked", flush=True)
            self._save_state()
            return truth
        self._exchange_state_known = True
        self._exchange_state_error = ""
        prev_pos = self._pos
        prev_entry = self._entry_price
        prev_trade_id = self._trade_id
        self._pos = truth.get("pos", 0)
        self._entry_price = truth.get("entry_price", 0.0)
        if self._pos == 0:
            if prev_pos != 0 and prev_trade_id:
                self._manual_close_block_action = "BUY" if prev_pos == 1 else "SELL"
                try:
                    from pipeline.live.execute.super_structure_executor import _append_execution_event
                    _append_execution_event({
                        "signal_id": "",
                        "trade_id": prev_trade_id,
                        "event_type": "MANUAL_CLOSE",
                        "action": "CLOSE",
                        "requested_price": None,
                        "executed_price": None,
                        "order_id": None,
                        "api_result": truth,
                    })
                except Exception as exc:
                    print(f"[SS] Manual close ledger write failed: {exc}", flush=True)
            self._sl_price = 0.0
            self._sl_order_id = None
            self._entry_bar_ts = None
            self._trade_id = ""
        elif not self._trade_id:
            open_trade = self._latest_unmatched_entry()
            if open_trade:
                self._trade_id = open_trade["trade_id"]
        
        # Sync order ID back to executor if it's missing but we have it in memory
        if self._pos != 0 and self._sl_order_id and self._executor:
            if getattr(self._executor, "sl_order_id", None) != self._sl_order_id:
                self._executor.sl_order_id = self._sl_order_id

        if prev_pos != self._pos or abs(prev_entry - self._entry_price) > 0.5:
            print(f"[SS] State synced from exchange: pos={self._pos} entry={self._entry_price:.1f}", flush=True)
        self._save_state()
        return truth

    def _ensure_runtime_ids(self) -> None:
        """Backfill in-memory IDs and restart cursor from historical signals."""
        open_trade = self._latest_unmatched_entry()
        if self._pos != 0 and not self._trade_id and open_trade:
            self._trade_id = open_trade["trade_id"]
        if self._pos != 0 and self._last_processed_bar_ts is None and open_trade:
            self._last_processed_bar_ts = open_trade["ts"]
            print(f"[SS] Derived last_processed_bar_ts from open entry: {self._last_processed_bar_ts}", flush=True)

    def _signal_sequence(self, action: str, ts: str, price: float) -> int:
        """Return sequence number among existing signals with same identity tuple."""
        seq = 0
        rounded = round(float(price), 4)
        for entry in self._signals:
            sig = entry.get("signal", {})
            if (
                str(sig.get("action", "")) == action
                and str(sig.get("ts") or entry.get("received_at") or "") == str(ts)
                and round(_signal_price(sig), 4) == rounded
            ):
                seq += 1
        return seq

    def _latest_unmatched_entry(self) -> dict | None:
        """Find latest entry signal without a following CLOSE signal."""
        stack: list[dict] = []
        for seq, entry in enumerate(self._signals):
            sig = entry.get("signal", {})
            action = sig.get("action")
            received_at = str(entry.get("received_at", ""))
            if action in ("BUY", "SELL"):
                ts = _to_utc_timestamp(sig.get("ts") or received_at)
                if ts is None:
                    continue
                signal_id = derive_signal_id(sig, received_at, seq)
                trade_id = derive_trade_id_from_entry(sig, received_at, seq)
                stack.append({"ts": ts, "signal_id": signal_id, "trade_id": trade_id, "action": action})
            elif action == "CLOSE" and stack:
                stack.pop()
        return stack[-1] if stack else None

    def _has_signal(self, action: str, ts: str) -> bool:
        if not ts:
            return False
        return any(
            s.get("signal", {}).get("action") == action and
            s.get("signal", {}).get("ts") == ts
            for s in self._signals
        )

    def _block_trade(self, key: str, msg: str) -> None:
        if key != self._last_blocked_trade_key:
            print(msg, flush=True)
            self._last_blocked_trade_key = key

    def _store_signal(self, action: str, price: float, sl: float = 0.0,
                       reason: str = "", **extra) -> None:
        signal_ts = str(extra.get("ts", ""))
        sequence = self._signal_sequence(action, signal_ts, price)
        temp_sig = {"action": action, "price": price, "ts": signal_ts}
        signal_id = str(extra.get("signal_id") or derive_signal_id(temp_sig, sequence=sequence))

        if action in ("BUY", "SELL"):
            trade_id = str(extra.get("trade_id") or derive_trade_id_from_entry(
                {**temp_sig, "signal_id": signal_id}, sequence=sequence
            ))
            self._trade_id = trade_id
        elif action == "CLOSE":
            latest = self._latest_unmatched_entry()
            trade_id = str(extra.get("trade_id") or self._trade_id or (latest["trade_id"] if latest else ""))
        else:
            trade_id = str(extra.get("trade_id") or "")

        sig = {
            "signal_id": signal_id,
            "trade_id": trade_id,
            "action": action,
            "symbol": SYMBOL,
            "price": price,
            "sl": sl,
            "reason": reason,
            "ts": signal_ts,
            "adx": extra.get("adx", 0),
            "cci": extra.get("cci", 0),
            "dema": extra.get("dema", 0),
            "parsed_from": f"super_structure.py: {action} {SYMBOL} @ {price:.1f}",
        }
        entry = {
            "signal_id": signal_id,
            "trade_id": trade_id,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "signal": sig,
        }
        self._signals.append(entry)
        self._save_signals()
        if action in ("BUY", "SELL"):
            self._save_state()
        print(f"[SS] ⚡ SIGNAL: {action} {SYMBOL} @ {price:.1f}" +
              (f" SL={sl:.1f}" if sl > 0 else "") +
              (f" ({reason})" if reason else ""), flush=True)

        bus_payload = {"action": action, "symbol": SYMBOL, "price": price,
                       "sl": sl, "reason": reason, "signal_id": signal_id,
                       "trade_id": trade_id, **extra}
        if action == "CLOSE" and self._pos != 0:
            side = 1 if self._pos == 1 else -1
            pnl = round((price - self._entry_price) * side * 10 - 1.74, 2)
            bus_payload["pnl"] = pnl
            bus_payload["side"] = "Long" if self._pos == 1 else "Short"
        try:
            SignalBus().publish("super_structure", bus_payload)
        except Exception:
            pass

        # Route to trade executor
        if self._executor and action in ("BUY", "SELL", "CLOSE"):
            try:
                self._executor.on_signal(bus_payload)
            except Exception as exc:
                print(f"[SS] Executor error: {exc}", flush=True)

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

    def _set_heartbeat_state(self, last_ts, df_5m, c, d, st, ax, cx, direction) -> None:
        try:
            prev_close = float(c[-2]) if len(c) > 1 else 0
            prev_dema = float(d[-2]) if len(d) > 1 else 0
            self._heartbeat_state = {
                "ts": str(last_ts)[:19] if last_ts is not None else "",
                "open": round(float(df_5m["open"].iloc[-1]), 1),
                "high": round(float(df_5m["high"].iloc[-1]), 1),
                "low": round(float(df_5m["low"].iloc[-1]), 1),
                "close": round(float(c[-1]), 1),
                "prev_close": round(prev_close, 1),
                "dema": round(float(d[-1]), 1) if len(d) > 0 else 0,
                "prev_dema": round(prev_dema, 1),
                "st": round(float(st[-1]), 1) if len(st) > 0 else 0,
                "adx": round(float(ax[-1]), 1) if len(ax) > 0 else 0,
                "cci": round(float(cx[-1]), 0) if len(cx) > 0 else 0,
                "direction": int(direction[-1]) if len(direction) > 0 else 0,
                "pos": self._pos,
                "entry_price": self._entry_price,
                "sl_price": self._sl_price,
                "exchange_state_known": self._exchange_state_known,
                "exchange_state_error": self._exchange_state_error,
            }
        except Exception:
            pass

    def check(self, now: "datetime | None" = None) -> list[dict]:
        """Fetch latest data and process every unprocessed completed 5m bar."""
        now = now or datetime.now(timezone.utc)
        now_ts = pd.Timestamp(now)
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        else:
            now_ts = now_ts.tz_convert("UTC")

        # Fast path: skip if called again within 30s of last check
        if self._last_checked_now is not None:
            if (now - self._last_checked_now).total_seconds() < 30:
                return []
        self._last_checked_now = now

        start = (now - timedelta(days=120)).strftime("%Y-%m-%d %H:%M:%S")
        end = now.strftime("%Y-%m-%d %H:%M:%S")

        # Incremental fetch: only load new rows since last check
        if self._cached_df is not None and self._cached_end is not None:
            if end > self._cached_end:
                new_rows = self.buffer.get(self._cached_end, end)
                if not new_rows.empty:
                    new_rows["timestamp_utc"] = pd.to_datetime(new_rows["timestamp_utc"], utc=True)
                    self._cached_df = pd.concat([self._cached_df, new_rows])
                    cutoff_ts = pd.Timestamp(now - timedelta(days=120))
                    self._cached_df = self._cached_df[
                        self._cached_df["timestamp_utc"] >= cutoff_ts
                    ]
                    # Advance cached_end to after the last fetched bar (skip 1s to avoid re-fetch)
                    last_max = self._cached_df["timestamp_utc"].max()
                    self._cached_end = (last_max + pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
            # Note: do NOT advance cached_end on empty fetch
        else:
            self._cached_df = self.buffer.get(start, end)
            # Set cached_end to just after the last fetched bar
            if not self._cached_df.empty:
                last_max = self._cached_df["timestamp_utc"].max()
                self._cached_end = (last_max + pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
            else:
                self._cached_end = end

        df = self._cached_df
        if len(df) < max(DEMA_LENGTH, ADX_LENGTH + 50, CCI_LENGTH + 50):
            return []

        # Resample 1m to 5m (TradingView chart timeframe)
        df = df.set_index("timestamp_utc").sort_index()
        df_5m = df.resample("5min", label="right", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

        # With label="right", a bar timestamp is complete only at/after that
        # timestamp. Exclude the currently forming right-labeled 5m bar.
        completed_cutoff = now_ts.floor("5min")
        df_5m = df_5m[df_5m.index <= completed_cutoff]
        if df_5m.empty:
            return []

        latest_completed_ts = df_5m.index[-1]

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

        if len(c) - 1 < DEMA_LENGTH:
            return []

        new_signals = []

        if self._last_processed_bar_ts is None:
            target_indices = [len(df) - 1]
        else:
            target_indices = [
                idx for idx, ts in enumerate(df.index)
                if ts > self._last_processed_bar_ts and idx >= DEMA_LENGTH
            ]

        if not target_indices:
            if latest_completed_ts != self._last_ts:
                self._last_ts = latest_completed_ts
                self._set_heartbeat_state(latest_completed_ts, df, c, d, st, ax, cx, direction)
            return []

        for i in target_indices:
            bar_ts = df.index[i]
            cur_close = float(c[i])
            cur_dema = float(d[i])
            cur_dir = int(direction[i])
            cur_adx = float(ax[i]) if i < len(ax) and not np.isnan(ax[i]) else 0.0
            cur_cci = float(cx[i]) if not np.isnan(cx[i]) else 0.0
            cur_st = float(st[i]) if not np.isnan(st[i]) else 0.0

            adx_ok = not USE_ADX or cur_adx > ADX_THRESHOLD
            cci_ok_long = not USE_CCI or cur_cci > CCI_LONG_MIN
            cci_ok_short = not USE_CCI or cur_cci < CCI_SHORT_MAX
            cross_up = (c[i - 1] < d[i - 1] and cur_close > cur_dema)
            cross_dn = (c[i - 1] > d[i - 1] and cur_close < cur_dema)
            long_signal = adx_ok and cci_ok_long and \
                (cross_up or cur_close > cur_dema) and cur_dir < 0
            short_signal = adx_ok and cci_ok_short and \
                (cross_dn or cur_close < cur_dema) and cur_dir > 0
            signal_ts = str(bar_ts)

            if self._manual_close_block_action:
                blocked_active = (
                    self._manual_close_block_action == "BUY" and long_signal
                ) or (
                    self._manual_close_block_action == "SELL" and short_signal
                )
                opposite_active = (
                    self._manual_close_block_action == "BUY" and short_signal
                ) or (
                    self._manual_close_block_action == "SELL" and long_signal
                )
                if not blocked_active or opposite_active:
                    print(
                        f"[SS] Manual-close re-entry block cleared ({self._manual_close_block_action})",
                        flush=True,
                    )
                    self._manual_close_block_action = ""
                    self._save_state()

            self._set_heartbeat_state(bar_ts, df.iloc[:i + 1], c[:i + 1], d[:i + 1], st[:i + 1], ax[:i + 1], cx[:i + 1], direction[:i + 1])

            sl_hit = (
                (self._pos == 1 and l[i] <= self._sl_price) or
                (self._pos == -1 and h[i] >= self._sl_price)
            )
            trend_flip = (
                (self._pos == 1 and cur_dir > 0) or
                (self._pos == -1 and cur_dir < 0)
            )
            exchange_unknown = self._executor is not None and not self._exchange_state_known
            if exchange_unknown and (sl_hit or trend_flip or long_signal or short_signal):
                if sl_hit or trend_flip:
                    self._block_trade(
                        f"unknown-close:{signal_ts}",
                        "[SS] Exchange state UNKNOWN — skipping CLOSE/flatten; preserving local position state",
                    )
                if long_signal or short_signal:
                    side = "BUY" if long_signal else "SELL"
                    self._block_trade(
                        f"unknown-entry:{signal_ts}:{side}",
                        f"[SS] Exchange state UNKNOWN — skipping {side} entry; preserving local pos={self._pos} entry={self._entry_price:.1f}",
                    )
                return new_signals

            # Existing stop orders are active before the current bar's strategy
            # recalculation. If hit, fill at the previously submitted stop.
            if self._pos == 1 and l[i] <= self._sl_price:
                if self._has_signal("CLOSE", signal_ts):
                    self._block_trade(f"dup-close:{signal_ts}", f"[SS] Duplicate CLOSE skipped for {signal_ts}")
                else:
                    self._store_signal("CLOSE", self._sl_price, reason="SL", ts=signal_ts)
                    new_signals.append(self._signals[-1])
                self._pos = 0
                self._entry_price = 0.0
                self._sl_price = 0.0
                self._trade_id = ""
                self._save_state()

            elif self._pos == -1 and h[i] >= self._sl_price:
                if self._has_signal("CLOSE", signal_ts):
                    self._block_trade(f"dup-close:{signal_ts}", f"[SS] Duplicate CLOSE skipped for {signal_ts}")
                else:
                    self._store_signal("CLOSE", self._sl_price, reason="SL", ts=signal_ts)
                    new_signals.append(self._signals[-1])
                self._pos = 0
                self._entry_price = 0.0
                self._sl_price = 0.0
                self._trade_id = ""
                self._save_state()

            # Trend flip close happens on bar close if the stop did not already fill.
            if self._pos == 1 and cur_dir > 0:
                if self._has_signal("CLOSE", signal_ts):
                    self._block_trade(f"dup-close:{signal_ts}", f"[SS] Duplicate CLOSE skipped for {signal_ts}")
                else:
                    self._store_signal("CLOSE", cur_close, reason="TREND_FLIP", ts=signal_ts)
                    new_signals.append(self._signals[-1])
                self._pos = 0
                self._entry_price = 0.0
                self._sl_price = 0.0
                self._trade_id = ""
                self._save_state()

            elif self._pos == -1 and cur_dir < 0:
                if self._has_signal("CLOSE", signal_ts):
                    self._block_trade(f"dup-close:{signal_ts}", f"[SS] Duplicate CLOSE skipped for {signal_ts}")
                else:
                    self._store_signal("CLOSE", cur_close, reason="TREND_FLIP", ts=signal_ts)
                    new_signals.append(self._signals[-1])
                self._pos = 0
                self._entry_price = 0.0
                self._sl_price = 0.0
                self._trade_id = ""
                self._save_state()

            # Submit/update the dynamic stop for the next bar.
            if self._pos != 0:
                prev_sl = self._sl_price
                self._sl_price = cur_st
                if abs(prev_sl - self._sl_price) > 0.1:
                    self._save_state()

            # Entries are market orders processed on bar close.
            if self._halt:
                if long_signal or short_signal:
                    print(f"[SS] HALT active — skipping entry signal ({self._halt_reason})", flush=True)
            elif long_signal and self._pos == 0:
                if self._manual_close_block_action == "BUY":
                    self._block_trade(
                        f"manual-close-block-buy:{signal_ts}",
                        f"[SS] Manual close detected — skipping BUY re-entry for {signal_ts}; waiting for a fresh signal",
                    )
                    self._last_processed_bar_ts = bar_ts
                    self._last_ts = bar_ts
                    self._save_state()
                    continue
                if self._has_signal("BUY", signal_ts):
                    self._block_trade(f"dup-buy:{signal_ts}", f"[SS] Duplicate BUY skipped for {signal_ts}")
                else:
                    self._pos = 1
                    self._entry_price = cur_close
                    self._sl_price = cur_st
                    self._store_signal("BUY", cur_close, cur_st,
                                       adx=round(cur_adx, 1) if not np.isnan(cur_adx) else 0,
                                       cci=round(cur_cci, 0) if not np.isnan(cur_cci) else 0,
                                       dema=round(cur_dema, 1),
                                       ts=signal_ts)
                    new_signals.append(self._signals[-1])

            elif short_signal and self._pos == 0:
                if self._manual_close_block_action == "SELL":
                    self._block_trade(
                        f"manual-close-block-sell:{signal_ts}",
                        f"[SS] Manual close detected — skipping SELL re-entry for {signal_ts}; waiting for a fresh signal",
                    )
                    self._last_processed_bar_ts = bar_ts
                    self._last_ts = bar_ts
                    self._save_state()
                    continue
                if self._has_signal("SELL", signal_ts):
                    self._block_trade(f"dup-sell:{signal_ts}", f"[SS] Duplicate SELL skipped for {signal_ts}")
                else:
                    self._pos = -1
                    self._entry_price = cur_close
                    self._sl_price = cur_st
                    self._store_signal("SELL", cur_close, cur_st,
                                       adx=round(cur_adx, 1) if not np.isnan(cur_adx) else 0,
                                       cci=round(cur_cci, 0) if not np.isnan(cur_cci) else 0,
                                       dema=round(cur_dema, 1),
                                       ts=signal_ts)
                    new_signals.append(self._signals[-1])

            self._last_processed_bar_ts = bar_ts
            self._last_ts = bar_ts
            self._save_state()

        # Store latest bar + indicator values for heartbeat
        latest_i = target_indices[-1]
        self._set_heartbeat_state(df.index[latest_i], df.iloc[:latest_i + 1], c[:latest_i + 1], d[:latest_i + 1], st[:latest_i + 1], ax[:latest_i + 1], cx[:latest_i + 1], direction[:latest_i + 1])

        # Trail stop-loss to latest SuperTrend if position active
        if self._executor and self._pos != 0:
            try:
                self._executor.update_sl(self._sl_price)
            except Exception:
                pass

        return new_signals

    def stats(self) -> str:
        n = len(self._signals)
        buys = sum(1 for s in self._signals if s["signal"]["action"] == "BUY")
        sells = sum(1 for s in self._signals if s["signal"]["action"] == "SELL")
        closes = n - buys - sells
        pos = "LONG" if self._pos == 1 else "SHORT" if self._pos == -1 else "FLAT"
        return (f"[Super Structure] {n} signals ({buys}B {sells}S {closes}C) | "
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

        def tg_send(msg: str, target_chat_id: str | None = None) -> None:
            try:
                dest_chat = target_chat_id or chat_id
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                data = urllib.parse.urlencode({"chat_id": dest_chat, "text": msg,
                                                "parse_mode": "Markdown"}).encode()
                resp = urllib.request.urlopen(url, data, timeout=5)
                result = json.loads(resp.read())
                if not result.get("ok"):
                    print(f"[SS] Telegram send failed: {result}", flush=True)
                else:
                    print(f"[SS] Telegram sent to {dest_chat}", flush=True)
            except Exception as e:
                print(f"[SS] Telegram send error: {e}", flush=True)

        def tg_handle(cmd: str) -> str | None:
            cmd = cmd.strip().lower()
            if cmd == "/strat":
                from pipeline.live.user_db import get_subscriptions_by_chat
                subs = get_subscriptions_by_chat(str(chat_id))
                if subs:
                    lines = ["📋 *Subscriptions:*\n"]
                    for s in subs:
                        name = "Super Structure" if s == "super_structure" else s
                        lines.append(f"• {name}")
                    lines.append("\nUse /strat on <name> or /strat off <name>")
                    return "\n".join(lines)
                return "📋 *No subscriptions.*\n\nUse /strat on super_structure or /strat off super_structure"

            elif cmd.startswith("/strat on ") or cmd.startswith("/strat off "):
                parts = cmd.split()
                if len(parts) < 3:
                    return "Usage: /strat on <name> or /strat off <name>"
                action = parts[1]
                name = parts[2]
                if action == "on":
                    from pipeline.live.user_db import subscribe_by_chat as _sub
                    _sub(str(chat_id), name)
                    display = "Super Structure" if name == "super_structure" else name
                    return f"✅ Subscribed to {display}"
                elif action == "off":
                    from pipeline.live.user_db import unsubscribe_by_chat as _unsub
                    _unsub(str(chat_id), name)
                    display = "Super Structure" if name == "super_structure" else name
                    return f"❌ Unsubscribed from {display}"
                return "Unknown action. Use /strat on <name> or /strat off <name>"

            elif cmd == "/ss" or cmd == "/ss_status" or cmd == "/state":
                self.reconcile()  # always show fresh state
                pos = "LONG" if self._pos == 1 else "SHORT" if self._pos == -1 else "FLAT"
                entry = f"${self._entry_price:.1f}" if self._pos != 0 else "—"
                sl = f"${self._sl_price:.1f}" if self._pos != 0 else "—"
                pos_source = "synced from exchange" if self._exchange_state_known else "last known local state"
                sig_count = len(self._signals)
                buy = sum(1 for x in self._signals if x["signal"]["action"] == "BUY")
                sell = sum(1 for x in self._signals if x["signal"]["action"] == "SELL")
                closes = sum(1 for x in self._signals if x["signal"]["action"] == "CLOSE")
                halt_line = f"\n🚨 *HALTED* — {self._halt_reason}" if self._halt else ""
                exchange_line = ""
                if self._executor:
                    if self._exchange_state_known:
                        exchange_line = "\nExchange: *KNOWN*"
                    else:
                        reason = f" — `{self._exchange_state_error}`" if self._exchange_state_error else ""
                        exchange_line = f"\nExchange: *UNKNOWN*{reason}\nEntries: *BLOCKED*"
                return (f"📊 *Super Structure Status*{halt_line}\n\n"
                        f"Position: {pos}  *({pos_source})*\n"
                        f"Entry: {entry}  |  SL: {sl}\n"
                        f"Signals: {sig_count} total ({buy}B {sell}S {closes}C)\n"
                        f"Config: ST({ST_FACTOR},{ATR_PERIOD}) DEMA({DEMA_LENGTH}) "
                        f"ADX({ADX_LENGTH}>{ADX_THRESHOLD}) CCI({CCI_LENGTH})"
                        f"{exchange_line}")

            elif cmd == "/halt":
                if self._halt:
                    return f"⏸️ Already halted ({self._halt_reason})"
                self._halt = True
                self._halt_reason = "manual /halt"
                self._save_state()
                # Flatten any open exchange position to "keep it clean"
                flat_msg = ""
                if self._executor:
                    try:
                        from pipeline.live.execute.super_structure_executor import _flatten_all
                        _flatten_all()
                        flat_msg = "\nFlattened any open positions."
                    except Exception as exc:
                        flat_msg = f"\n⚠️ Flatten failed: {exc}"
                self.reconcile()
                return f"🛑 *HALT activated* — no new entries until /resume.{flat_msg}"

            elif cmd == "/resume":
                if not self._halt:
                    return "▶️ Already running"
                self._halt = False
                self._halt_reason = ""
                self._save_state()
                self.reconcile()
                return "▶️ *Resumed* — strategy active"

            elif cmd == "/parity" or cmd.startswith("/parity "):
                parts = cmd.split()
                date_arg = None
                tz_arg = "Asia/Jakarta"
                if len(parts) >= 2:
                    if parts[1] == "utc":
                        tz_arg = "UTC"
                    else:
                        date_arg = parts[1]
                if len(parts) >= 3 and parts[2] == "utc":
                    tz_arg = "UTC"
                try:
                    from pipeline.live.parity_super_structure import (
                        build_parity_report,
                        format_telegram_report,
                        write_parity_report,
                    )
                    report = build_parity_report(date_arg, tz_arg)
                    _, md_path, _ = write_parity_report(report)
                    rel_md = md_path.relative_to(ROOT)
                    return (
                        format_telegram_report(report)
                        + f"\n\nSaved report: `{rel_md}`"
                    )
                except Exception as exc:
                    print(f"[SS] parity command error: {exc}", flush=True)
                    return f"⚠️ *Parity failed*: `{exc}`"

            elif cmd == "/help":
                return ("📋 *Commands*\n\n"
                        "/strat — list subscriptions\n"
                        "/strat on/off <name>\n"
                        "/ss /state — current status (live exchange truth)\n"
                        "/parity [YYYY-MM-DD] [utc] — 3-way parity report\n"
                        "/halt — flatten + stop new entries\n"
                        "/resume — re-enable trading\n"
                        "/help — this message")

            return None

        print(f"[SS] Starting live strategy loop...", flush=True)
        print(f"[SS] Config: ST({ST_FACTOR},{ATR_PERIOD}) DEMA({DEMA_LENGTH}) "
              f"ADX({ADX_LENGTH}>{ADX_THRESHOLD}) CCI({CCI_LENGTH})", flush=True)
        if bot_enabled:
            print(f"[SS] Telegram commands: /strat /ss /parity /halt /resume /help", flush=True)
        while True:
            try:
                if self._executor:
                    self.reconcile()
                signals = self.check()
                for s in signals:
                    pass  # signals already stored + printed in _store_signal
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[SS] Error: {e}", flush=True)

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
                                print(f"[SS] Command: {msg}", flush=True)
                                try:
                                    reply = tg_handle(msg)
                                except Exception as ex:
                                    print(f"[SS] tg_handle error: {ex}", flush=True)
                                    reply = None
                                if reply:
                                    print(f"[SS] Reply: {reply[:100]}", flush=True)
                                    tg_send(reply, chat)
                                else:
                                    print(f"[SS] No reply", flush=True)
                except Exception as e:
                    print(f"[SS] Poll error: {e}", flush=True)

                # Heartbeat every 5 min — also reconcile + safety checks
                if self._executor:
                    try:
                        # Reconcile state from exchange (silent adopt)
                        self.reconcile()
                        # Auto-halt if Topstep violation active
                        try:
                            from pipeline.live.execute.super_structure_executor import (
                                _check_violations, _flatten_all,
                            )
                            v = _check_violations()
                            if v and not self._halt:
                                self._halt = True
                                self._halt_reason = f"violation: {v[0].get('type', 'unknown')}"
                                self._save_state()
                                try: _flatten_all()
                                except Exception: pass
                                tg_send(f"🚨 *Topstep VIOLATION* — auto-halt + flatten\n{self._halt_reason}")
                                print(f"[SS] AUTO-HALT (violation): {self._halt_reason}", flush=True)
                        except Exception:
                            pass
                        # Send heartbeat (with refreshed state)
                        self._executor.heartbeat({**self._heartbeat_state, "pos": self._pos,
                                                   "entry_price": self._entry_price,
                                                   "sl_price": self._sl_price})
                    except Exception:
                        pass

                _time.sleep(30)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Super Structure Strategy (Python port)")
    parser.add_argument("--live", action="store_true", help="Run live monitoring loop")
    parser.add_argument("--test", action="store_true", help="Single check + print status")
    args = parser.parse_args()

    strategy = SuperStructure()

    if args.test:
        signals = strategy.check()
        print(f"[SS] Checked: {len(signals)} new signals")
        print(strategy.stats())

    if args.live:
        strategy.run_live()

#!/usr/bin/env python3
"""FVG + DEMA Scalper — Python port of TradingView Pine Script strategy.

FVG (Fair Value Gap) entry with DEMA trend filter, ADX + Choppiness regime
filter, session gating, and swing-based SL/TP.

Usage:
    from pipeline.live.fvg_scalper import FVGScalper
    s = FVGScalper()
    signals = s.check()          # single check
    s.run_live()                 # 30s polling daemon
"""
from __future__ import annotations

import json
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SIGNALS_PATH = ROOT / "data" / "Live" / "fvg_signals.json"

# ── strategy parameters (from Pine Script inputs) ──────────────────────────

MIN_GAP_PTS = 1.0          # Min FVG gap in points (use UI Gap Min filter to explore)
MIN_GAP_ATR = 0.0          # Min FVG gap in ATR multiples
USE_DISPL = True           # Require displacement candle
DISPL_ATR = 1.0            # Displacement range ATR min
MIN_BODY_PCT = 50.0        # Min body % of candle
FVG_EXTEND_BARS = 20       # FVG box extend bars (visual only)

DEMA_LENGTH = 50           # DEMA length
USE_DEMA_DIR = True        # Use DEMA direction filter
USE_DEMA_SLOPE = True      # Require DEMA slope
SLOPE_BARS = 5             # DEMA slope lookback bars
USE_DIST_FILTER = True     # Avoid entry too far from DEMA
MAX_DIST_ATR = 3.0         # Max distance from DEMA in ATR

USE_REGIME = True          # Use regime filter (ADX + CHOP + slope)
ADX_LENGTH = 14
MIN_ADX = 12.0
CHOP_LENGTH = 14
MAX_CHOP = 62.0
MIN_DEMA_SLOPE_ATR = 0.03
USE_WHIPSAW = False
CROSS_LOOKBACK = 20
MAX_DEMA_CROSSES = 5
TP_RISK_RATIO = 1.0        # TP / Risk ratio (calibrated: 1R fixed)

SL_LOOKBACK = 15            # SL swing lookback bars (calibrated: wider stops at 1R)
MIN_RISK_PTS = 1.0
MAX_RISK_PTS = 25.0
MAX_TRADES_DAY = 50
COOLDOWN_BARS = 5
ONE_TRADE_AT_TIME = True
USE_DEMA_EXIT = False
DEMA_EXIT_BUFFER = 0.0
DEMA_EXIT_ONLY_LOSS = False

ENABLE_LONG = True
ENABLE_SHORT = True

SESSION_MODE = "Asia + London"   # Off | Asia Only | London Only | NY Only | Asia+London | Asia+London+NY (calibrated best)
SYMBOL = "MGC"
COMMISSION = 1.74         # Topstep: $0.50 commission + $1.24 fees/round-turn
POINT_VALUE = 10.0        # $10/point

EPS = 1e-10


# ── indicators ────────────────────────────────────────────────────────────

def dema(close: np.ndarray, length: int) -> np.ndarray:
    """Double Exponential Moving Average."""
    e1 = pd.Series(close).ewm(span=length, adjust=False).mean().values
    e2 = pd.Series(e1).ewm(span=length, adjust=False).mean().values
    return 2.0 * e1 - e2


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range (Wilder's smoothing)."""
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_vals = np.full(len(tr), np.nan)
    atr_vals[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr[i]) / period
    return atr_vals


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average Directional Index."""
    up = np.diff(high)
    dn = -np.diff(low)
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))

    plus_di = np.full(len(high), np.nan)
    minus_di = np.full(len(high), np.nan)
    adx_vals = np.full(len(high), np.nan)

    smooth_tr = np.mean(tr[:period]) if len(tr) >= period else 0
    smooth_plus = np.mean(plus_dm[:period]) if len(plus_dm) >= period else 0
    smooth_minus = np.mean(minus_dm[:period]) if len(minus_dm) >= period else 0

    for i in range(period - 1, len(tr)):
        smooth_tr = (smooth_tr * (period - 1) + tr[i]) / period
        smooth_plus = (smooth_plus * (period - 1) + plus_dm[i]) / period
        smooth_minus = (smooth_minus * (period - 1) + minus_dm[i]) / period
        pdi = smooth_plus / smooth_tr * 100 if smooth_tr > 0 else 0
        mdi = smooth_minus / smooth_tr * 100 if smooth_tr > 0 else 0
        dx = abs(pdi - mdi) / (pdi + mdi + EPS) * 100
        idx = i + 1
        plus_di[idx] = pdi
        minus_di[idx] = mdi
        if idx == period:
            adx_vals[idx] = np.mean([dx])
        else:
            adx_vals[idx] = (adx_vals[idx - 1] * (period - 1) + dx) / period
    return adx_vals


def choppiness(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Choppiness Index — higher = more sideways."""
    n = len(high)
    result = np.full(n, np.nan)
    for i in range(period, n):
        window_h = high[i - period + 1:i + 1]
        window_l = low[i - period + 1:i + 1]
        window_c = close[i - period + 1:i + 1]
        range_hl = np.max(window_h) - np.min(window_l)
        tr_sum = 0.0
        for j in range(i - period + 2, i + 1):
            tr_val = max(window_h[j - (i - period + 1)] - window_l[j - (i - period + 1)],
                         abs(window_h[j - (i - period + 1)] - window_c[j - (i - period + 1) - 1]),
                         abs(window_l[j - (i - period + 1)] - window_c[j - (i - period + 1) - 1]))
            tr_sum += tr_val
        tr_sum += window_h[0] - window_l[0]
        if range_hl > 0:
            result[i] = 100.0 * np.log10(tr_sum / range_hl) / np.log10(period)
    return result


# ── strategy class ────────────────────────────────────────────────────────


class FVGScalper:
    """FVG + DEMA Scalper strategy."""

    def __init__(self, buffer_or_db_path=None):
        from pipeline.live.buffer import DataBuffer, CANARY_DB
        if buffer_or_db_path is None:
            self.buffer = DataBuffer(db_path=CANARY_DB)
        elif isinstance(buffer_or_db_path, (Path, str)):
            self.buffer = DataBuffer(db_path=Path(buffer_or_db_path))
        else:
            self.buffer = buffer_or_db_path

        self._pos = 0           # 1=long, -1=short
        self._entry_price = 0.0
        self._sl_price = 0.0
        self._tp_price = 0.0
        self._last_ts: pd.Timestamp | None = None
        self._last_checked_now: datetime | None = None
        self._cached_df: pd.DataFrame | None = None
        self._cached_end: str | None = None
        self._signals: list[dict] = []
        self._entry_bar_ts: pd.Timestamp | None = None
        self._trades_today = 0
        self._last_date = None
        self._last_entry_bar_idx = -100

        self._load_signals()

    def _load_signals(self) -> None:
        if SIGNALS_PATH.exists():
            try:
                self._signals = json.loads(SIGNALS_PATH.read_text())
            except Exception:
                pass

    def _save_signals(self) -> None:
        SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SIGNALS_PATH.write_text(json.dumps(self._signals[-500:], indent=2, default=str))

    def _session_name(self, ts: pd.Timestamp) -> str:
        t = ts.tz_convert("UTC") if ts.tz else ts.tz_localize("UTC")
        h = t.hour + t.minute / 60.0
        if 0 <= h < 7:  return "Asia"
        if 7 <= h < 12: return "London"
        if 13.5 <= h < 20: return "NY"
        return "Other"

    def _is_in_session(self, ts: pd.Timestamp) -> bool:
        if SESSION_MODE == "Off":
            return True
        t = ts.tz_convert("UTC") if ts.tz else ts.tz_localize("UTC")
        h = t.hour + t.minute / 60.0
        in_asia = 0 <= h < 7
        in_london = 7 <= h < 12
        in_ny = 13.5 <= h < 20
        if SESSION_MODE == "Asia Only": return in_asia
        if SESSION_MODE == "London Only": return in_london
        if SESSION_MODE == "NY Only": return in_ny
        if SESSION_MODE == "Asia + London": return in_asia or in_london
        if SESSION_MODE == "Asia + NY": return in_asia or in_ny
        if SESSION_MODE == "London + NY": return in_london or in_ny
        if SESSION_MODE == "Asia + London + NY": return in_asia or in_london or in_ny
        return True

    def _store_signal(self, action: str, price: float, sl: float = 0.0,
                       tp: float = 0.0, reason: str = "", **extra) -> None:
        sig = {
            "action": action, "symbol": SYMBOL,
            "price": price, "sl": sl, "tp": tp, "reason": reason,
            "ts": extra.get("ts", ""),
            "adx": extra.get("adx", 0), "chop": extra.get("chop", 0),
            "dema": extra.get("dema", 0), "dema_slope": extra.get("dema_slope", 0),
            "gap_pts": extra.get("gap_pts", 0),
        }
        entry = {"received_at": datetime.now(timezone.utc).isoformat(), "signal": sig}
        self._signals.append(entry)
        self._save_signals()
        print(f"[FVG] ⚡ SIGNAL: {action} {SYMBOL} @ {price:.1f}" +
              (f" SL={sl:.1f}" if sl > 0 else "") +
              (f" TP={tp:.1f}" if tp > 0 else "") +
              (f" ({reason})" if reason else ""), flush=True)

        bus_payload = {"action": action, "symbol": SYMBOL, "price": price,
                       "sl": sl, "tp": tp, "reason": reason, **extra}
        if action == "CLOSE" and self._pos != 0:
            side = 1 if self._pos == 1 else -1
            pnl = round((price - self._entry_price) * side * POINT_VALUE - COMMISSION, 2)
            bus_payload["pnl"] = pnl
        try:
            from pipeline.live.signal_bus import SignalBus
            SignalBus().publish("fvg_scalper", bus_payload)
        except Exception:
            pass

    def check(self, now: "datetime | None" = None) -> list[dict]:
        """Run one check cycle. Returns new signals."""
        now = now or datetime.now(timezone.utc)

        if self._last_checked_now is not None:
            if (now - self._last_checked_now).total_seconds() < 30:
                return []
        self._last_checked_now = now

        start = (now - timedelta(days=120)).strftime("%Y-%m-%d %H:%M:%S")
        end = now.strftime("%Y-%m-%d %H:%M:%S")

        if self._cached_df is not None and self._cached_end is not None:
            if end > self._cached_end:
                new_rows = self.buffer.get(self._cached_end, end)
                if not new_rows.empty:
                    new_rows["timestamp_utc"] = pd.to_datetime(new_rows["timestamp_utc"], utc=True)
                    self._cached_df = pd.concat([self._cached_df, new_rows])
                    cutoff_ts = pd.Timestamp(now - timedelta(days=120))
                    self._cached_df = self._cached_df[self._cached_df["timestamp_utc"] >= cutoff_ts]
            self._cached_end = end
        else:
            self._cached_df = self.buffer.get(start, end)
            self._cached_end = end

        df = self._cached_df
        if len(df) < max(DEMA_LENGTH, ADX_LENGTH + 50, CHOP_LENGTH + 50):
            return []

        df = df.set_index("timestamp_utc").sort_index()
        df_5m = df.resample("5min", label="right", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

        last_ts = df_5m.index[-1]
        if last_ts == self._last_ts:
            return []
        self._last_ts = last_ts

        if len(df_5m) < 50:
            return []

        h = df_5m["high"].values.astype(float)
        l = df_5m["low"].values.astype(float)
        c = df_5m["close"].values.astype(float)

        hl2 = (h + l) / 2.0
        d_arr = dema(hl2, DEMA_LENGTH)
        atr_arr = atr(h, l, c, 14)
        ax = adx(h, l, c, ADX_LENGTH)
        chop_arr = choppiness(h, l, c, CHOP_LENGTH)

        new_signals = []

        i = len(c) - 1
        if i < DEMA_LENGTH + 10:
            return []

        # ── daily counter reset ──────────────────────────────────────────
        bar_date = df_5m.index[i].date() if hasattr(df_5m.index[i], 'date') else pd.Timestamp(df_5m.index[i]).date()
        if self._last_date is None or bar_date != self._last_date:
            self._trades_today = 0
            self._last_date = bar_date

        # ── DEMA filter ──────────────────────────────────────────────────
        cur_close = float(c[i])
        cur_dema = float(d_arr[i]) if i < len(d_arr) and not np.isnan(d_arr[i]) else 0.0
        cur_adx = float(ax[i]) if i < len(ax) and not np.isnan(ax[i]) else 0.0
        cur_chop = float(chop_arr[i]) if i < len(chop_arr) and not np.isnan(chop_arr[i]) else 100.0
        cur_atr = float(atr_arr[i]) if i < len(atr_arr) and not np.isnan(atr_arr[i]) else 1.0
        sl_b = min(SLOPE_BARS, i)
        cur_slope = cur_dema - float(d_arr[i - sl_b]) if i >= sl_b and not np.isnan(d_arr[i - sl_b]) else 0.0
        cur_ts = df_5m.index[i]

        distance = abs(cur_close - cur_dema)

        above_dema = cur_close > cur_dema
        below_dema = cur_close < cur_dema
        slope_up = cur_slope > 0
        slope_down = cur_slope < 0
        distance_ok = not USE_DIST_FILTER or distance <= cur_atr * MAX_DIST_ATR
        long_trend_ok = not USE_DEMA_DIR or above_dema
        short_trend_ok = not USE_DEMA_DIR or below_dema
        long_slope_ok = not USE_DEMA_SLOPE or slope_up
        short_slope_ok = not USE_DEMA_SLOPE or slope_down

        # ── regime filter ────────────────────────────────────────────────
        adx_ok = cur_adx >= MIN_ADX
        chop_ok = cur_chop <= MAX_CHOP
        slope_strength = abs(cur_slope) / cur_atr if cur_atr > 0 else 0.0
        slope_ok = slope_strength >= MIN_DEMA_SLOPE_ATR
        regime_ok = not USE_REGIME or (adx_ok and chop_ok and slope_ok)

        # ── session ──────────────────────────────────────────────────────
        in_session = self._is_in_session(cur_ts)

        # ── trade mgmt ───────────────────────────────────────────────────
        daily_ok = self._trades_today < MAX_TRADES_DAY
        cooldown = i - self._last_entry_bar_idx >= COOLDOWN_BARS
        flat_ok = not ONE_TRADE_AT_TIME or self._pos == 0

        # ── FVG detection ────────────────────────────────────────────────
        fvg = None
        fvg_gap = 0.0
        if i >= 2:
            # Bull FVG: current low > high[2] and close[1] > high[2]
            h2, l2 = float(h[i - 2]), float(l[i - 2])
            h1, l1 = float(h[i - 1]), float(l[i - 1])
            c1 = float(c[i - 1])
            bull_raw = l[i] > h2 and c1 > h2
            bear_raw = h[i] < l2 and c1 < l2

            bull_gap = l[i] - h2 if bull_raw else 0.0
            bear_gap = l2 - h[i] if bear_raw else 0.0
            gap_pts = bull_gap if bull_raw else bear_gap

            gap_atr_ok = gap_pts >= cur_atr * MIN_GAP_ATR
            gap_pts_ok = gap_pts >= MIN_GAP_PTS

            # Displacement candle (mid candle)
            mid_range = h1 - l1
            mid_body = abs(c1 - float(df_5m["open"].iloc[i - 1]))
            mid_body_pct = mid_body / mid_range * 100.0 if mid_range > 0 else 0.0

            bull_disp_ok = not USE_DISPL or (c1 > float(df_5m["open"].iloc[i - 1]) and
                                              mid_range >= cur_atr * DISPL_ATR and
                                              mid_body_pct >= MIN_BODY_PCT)
            bear_disp_ok = not USE_DISPL or (c1 < float(df_5m["open"].iloc[i - 1]) and
                                              mid_range >= cur_atr * DISPL_ATR and
                                              mid_body_pct >= MIN_BODY_PCT)

            bull_fvg = bull_raw and gap_pts_ok and gap_atr_ok and bull_disp_ok
            bear_fvg = bear_raw and gap_pts_ok and gap_atr_ok and bear_disp_ok
            fvg = "bull" if bull_fvg else "bear" if bear_fvg else None
            fvg_gap = gap_pts

        # ── risk calculation ─────────────────────────────────────────────
        sl_bars = min(SL_LOOKBACK - 1, i)
        long_sl_base = float(np.min(l[i - sl_bars:i + 1]))
        short_sl_base = float(np.max(h[i - sl_bars:i + 1]))
        long_risk = cur_close - long_sl_base
        short_risk = short_sl_base - cur_close
        long_risk_ok = MIN_RISK_PTS <= long_risk <= MAX_RISK_PTS
        short_risk_ok = MIN_RISK_PTS <= short_risk <= MAX_RISK_PTS

        # ── entries ──────────────────────────────────────────────────────
        long_signal = (ENABLE_LONG and fvg == "bull" and long_trend_ok and
                       long_slope_ok and distance_ok and regime_ok and
                       in_session and daily_ok and cooldown and flat_ok and
                       long_risk_ok)
        short_signal = (ENABLE_SHORT and fvg == "bear" and short_trend_ok and
                        short_slope_ok and distance_ok and regime_ok and
                        in_session and daily_ok and cooldown and flat_ok and
                        short_risk_ok)

        # ── existing stop/tp ─────────────────────────────────────────────
        cur_high = float(h[i])
        cur_low = float(l[i])

        if self._pos == 1:
            if cur_low <= self._sl_price:
                self._store_signal("CLOSE", self._sl_price, reason="SL",
                                   ts=str(cur_ts))
                new_signals.append(self._signals[-1])
                self._pos = 0
            elif cur_high >= self._tp_price:
                self._store_signal("CLOSE", self._tp_price, reason="TP",
                                   ts=str(cur_ts))
                new_signals.append(self._signals[-1])
                self._pos = 0
            elif USE_DEMA_EXIT:
                prev_c = float(c[i - 1]) if i > 0 else cur_close
                cross_under = prev_c > cur_dema and cur_close < cur_dema
                if cross_under:
                    if not DEMA_EXIT_ONLY_LOSS or cur_close < self._entry_price:
                        self._store_signal("CLOSE", cur_close, reason="DEMA_EXIT",
                                           ts=str(cur_ts))
                        new_signals.append(self._signals[-1])
                        self._pos = 0

        if self._pos == -1:
            if cur_high >= self._sl_price:
                self._store_signal("CLOSE", self._sl_price, reason="SL",
                                   ts=str(cur_ts))
                new_signals.append(self._signals[-1])
                self._pos = 0
            elif cur_low <= self._tp_price:
                self._store_signal("CLOSE", self._tp_price, reason="TP",
                                   ts=str(cur_ts))
                new_signals.append(self._signals[-1])
                self._pos = 0
            elif USE_DEMA_EXIT:
                prev_c = float(c[i - 1]) if i > 0 else cur_close
                cross_over = prev_c < cur_dema and cur_close > cur_dema
                if cross_over:
                    if not DEMA_EXIT_ONLY_LOSS or cur_close > self._entry_price:
                        self._store_signal("CLOSE", cur_close, reason="DEMA_EXIT",
                                           ts=str(cur_ts))
                        new_signals.append(self._signals[-1])
                        self._pos = 0

        # ── entries ──────────────────────────────────────────────────────
        if long_signal and self._pos == 0:
            self._pos = 1
            self._entry_price = cur_close
            self._sl_price = long_sl_base
            self._tp_price = cur_close + TP_RISK_RATIO * long_risk
            self._trades_today += 1
            self._last_entry_bar_idx = i
            dema_slope = cur_slope / cur_atr if cur_atr > 0 else 0
            self._store_signal("BUY", cur_close, sl=long_sl_base,
                               tp=self._tp_price,
                               adx=round(cur_adx, 1),
                               chop=round(cur_chop, 1),
                               dema=round(cur_dema, 1),
                               dema_slope=round(dema_slope, 4),
                               gap_pts=round(fvg_gap, 1),
                               ts=str(cur_ts))
            new_signals.append(self._signals[-1])

        if short_signal and self._pos == 0:
            self._pos = -1
            self._entry_price = cur_close
            self._sl_price = short_sl_base
            self._tp_price = cur_close - TP_RISK_RATIO * short_risk
            self._trades_today += 1
            self._last_entry_bar_idx = i
            dema_slope = cur_slope / cur_atr if cur_atr > 0 else 0
            self._store_signal("SELL", cur_close, sl=short_sl_base,
                               tp=self._tp_price,
                               adx=round(cur_adx, 1),
                               chop=round(cur_chop, 1),
                               dema=round(cur_dema, 1),
                               dema_slope=round(dema_slope, 4),
                               gap_pts=round(fvg_gap, 1),
                               ts=str(cur_ts))
            new_signals.append(self._signals[-1])

        return new_signals

    def run_live(self) -> None:
        """Main loop — check every 30s + Telegram commands."""
        import urllib.request, urllib.parse

        token, chat_id = "", ""
        env_file = ROOT / "data" / "Live" / "telegram.env"
        if env_file.exists():
            for line in env_file.read_text().strip().split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k == "TELEGRAM_BOT_TOKEN": token = v
                    elif k == "TELEGRAM_CHAT_ID": chat_id = v

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
                urllib.request.urlopen(url, data, timeout=5)
            except Exception as e:
                print(f"[FVG] Telegram send error: {e}", flush=True)

        def tg_handle(cmd: str) -> str | None:
            cmd = cmd.strip().lower()
            if cmd == "/strat":
                from pipeline.live.user_db import get_subscriptions_by_chat
                subs = get_subscriptions_by_chat(str(chat_id))
                if subs:
                    lines = ["📋 *Subscriptions:*\n"]
                    for s in subs:
                        name = "FVG Scalper" if s == "fvg_scalper" else "Super Structure" if s == "super_structure" else s
                        lines.append(f"• {name}")
                    lines.append("\nUse /strat on <name> or /strat off <name>")
                    return "\n".join(lines)
                return "📋 *No subscriptions.*\n\nUse /strat on fvg_scalper"
            elif cmd.startswith("/strat on ") or cmd.startswith("/strat off "):
                parts = cmd.split()
                if len(parts) < 3:
                    return "Use /strat on <name> or /strat off <name>"
                action, name = parts[1], parts[2]
                if action == "on":
                    from pipeline.live.user_db import subscribe_by_chat as _sub
                    _sub(str(chat_id), name)
                    display = "FVG Scalper" if name == "fvg_scalper" else name
                    return f"✅ Subscribed to {display}"
                elif action == "off":
                    from pipeline.live.user_db import unsubscribe_by_chat as _unsub
                    _unsub(str(chat_id), name)
                    display = "FVG Scalper" if name == "fvg_scalper" else name
                    return f"❌ Unsubscribed from {display}"
                return "Unknown action"
            elif cmd == "/fvg" or cmd == "/fvg_status":
                pos = "LONG" if self._pos == 1 else "SHORT" if self._pos == -1 else "FLAT"
                entry = f"${self._entry_price:.1f}" if self._pos != 0 else "—"
                sl = f"${self._sl_price:.1f}" if self._pos != 0 else "—"
                tp = f"${self._tp_price:.1f}" if self._pos != 0 else "—"
                return (f"📊 *FVG Scalper Status*\n\n"
                        f"Position: {pos}\n"
                        f"Entry: {entry}  |  SL: {sl}  |  TP: {tp}\n"
                        f"Trades today: {self._trades_today}/{MAX_TRADES_DAY}\n"
                        f"Session: {SESSION_MODE}")
            return None

        print(f"[FVG] Starting live strategy loop...", flush=True)
        print(f"[FVG] Config: DEMA({DEMA_LENGTH}) ADX({ADX_LENGTH}>{MIN_ADX}) "
              f"CHOP({CHOP_LENGTH}<={MAX_CHOP}) SL={SL_LOOKBACK} TP={TP_RISK_RATIO}R", flush=True)
        if bot_enabled:
            print(f"[FVG] Telegram commands: /strat /fvg", flush=True)

        while True:
            try:
                signals = self.check()
                for s in signals:
                    pass
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[FVG] Error: {e}", flush=True)

            if bot_enabled:
                try:
                    url = f"https://api.telegram.org/bot{token}/getUpdates"
                    params = urllib.parse.urlencode({
                        "offset": max(1, last_update_id + 1), "timeout": 0})
                    resp = urllib.request.urlopen(f"{url}?{params}", timeout=5)
                    result = json.loads(resp.read())
                    if result.get("ok") and result.get("result"):
                        for update in result["result"]:
                            last_update_id = update["update_id"]
                            msg = update.get("message", {}).get("text", "")
                            chat = str(update.get("message", {}).get("chat", {}).get("id", ""))
                            if msg and chat:
                                print(f"[FVG] Command: {msg}", flush=True)
                                try:
                                    reply = tg_handle(msg)
                                except Exception as ex:
                                    print(f"[FVG] tg_handle error: {ex}", flush=True)
                                    reply = None
                                if reply:
                                    tg_send(reply)
                except Exception:
                    pass

            _time.sleep(30)


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FVG Scalper Strategy")
    parser.add_argument("--live", action="store_true", help="Run live monitoring loop")
    parser.add_argument("--test", action="store_true", help="Single check + print status")
    args = parser.parse_args()

    s = FVGScalper()
    if args.test:
        sigs = s.check()
        print(f"[FVG] Checked: {len(sigs)} new signals")
        total = len(s._signals)
        buys = sum(1 for x in s._signals if x["signal"]["action"] == "BUY")
        sells = sum(1 for x in s._signals if x["signal"]["action"] == "SELL")
        closes = sum(1 for x in s._signals if x["signal"]["action"] == "CLOSE")
        pos = "LONG" if s._pos == 1 else "SHORT" if s._pos == -1 else "FLAT"
        print(f"[FVG Scalper] {total} signals ({buys}B {sells}S {closes}C) | "
              f"Pos: {pos} | Entry: {s._entry_price}")
    elif args.live:
        s.run_live()
    else:
        sigs = s.check()
        print(f"[FVG] Checked: {len(sigs)} new signals")

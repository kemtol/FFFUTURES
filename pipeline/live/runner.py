#!/usr/bin/env python3
"""
Live inference signal generator — ORB_v2.0.

Loads trained models, monitors breakout events, computes features,
predicts probabilities, applies policy filter, and outputs signals.

Usage:
    python3 pipeline/live/runner.py --simulate 30   # run through last 30 days
    python3 pipeline/live/runner.py --live           # live monitoring

Signal output format:
    [2026-04-28 14:35 UTC]  US      BULL  REV   prob=0.65  entry=3200.5  TP=3204.5  SL=3199.5
"""

from __future__ import annotations

import json
import os as _os
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
for p in [str(ROOT), str(ROOT / "pipeline")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.live.buffer import DataBuffer, CANARY_DB
from pipeline.live.orb_detector import ORBDetector, BreakoutEvent
from pipeline.live.feature_builder import FeatureBuilder, FEATURE_ORDER
from analysis.topstep_sim import PolicyParams, apply_policy
from pipeline.live.webhook import WebhookServer
from pipeline.live.signal_bus import SignalBus

# ── config ────────────────────────────────────────────────────────────────────

MODEL_DIR = ROOT / "model" / "ORB_v2.0_2010-2026"
TARGET = "y_1r4_180m"
RR = 4.0

# Telegram notifications (optional — set env vars to enable)
TELEGRAM_BOT_TOKEN = None  # Set via env: TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID = None    # Set via env: TELEGRAM_CHAT_ID

# Best policy from Topstep sweep
POLICY = PolicyParams(
    rev_q=0.6, cont_q=0.6,
    rev_adx_min=30, cont_adx_max=30,
    daily_stop_usd=0, daily_profit_cap_usd=1400,
    risk_per_r_usd=100,
)


# ── telegram helper ──────────────────────────────────────────────────────────


class TelegramBot:
    def __init__(self):
        self.token = _os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = _os.environ.get("TELEGRAM_CHAT_ID", "")
        env_file = ROOT / "data" / "Live" / "telegram.env"
        if env_file.exists():
            for line in env_file.read_text().strip().split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k == "TELEGRAM_BOT_TOKEN" and not self.token:
                        self.token = v
                    elif k == "TELEGRAM_CHAT_ID" and not self.chat_id:
                        self.chat_id = v
        self._last_update_id = self._fetch_latest_update_id()
        self.enabled = bool(self.token and self.chat_id)

    def _fetch_latest_update_id(self) -> int:
        """Get the latest update_id from Telegram so we only process new messages."""
        try:
            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            resp = urllib.request.urlopen(url, timeout=5)
            result = json.loads(resp.read())
            if result.get("ok") and result.get("result"):
                return result["result"][-1]["update_id"]
        except Exception:
            pass
        return 0

    def send(self, msg: str) -> None:
        if not self.enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": self.chat_id,
                "text": msg,
                "parse_mode": "Markdown",
            }).encode()
            urllib.request.urlopen(url, data, timeout=5)
        except Exception:
            pass

    def poll(self, signal_runner) -> None:
        """Check for incoming commands once. Call periodically."""
        if not self.enabled:
            return
        try:
            offset = self._last_update_id + 1

            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            params = urllib.parse.urlencode({
                "offset": max(1, offset),
                "timeout": 0,  # no long-poll — return immediately
            })
            full_url = f"{url}?{params}"
            req = urllib.request.Request(full_url)
            resp = urllib.request.urlopen(req, timeout=5)
            result = json.loads(resp.read())
            if result.get("ok") and result.get("result"):
                for update in result["result"]:
                    self._last_update_id = update["update_id"]
                    msg = update.get("message", {}).get("text", "")
                    chat = update.get("message", {}).get("chat", {}).get("id")
                    if chat and msg:
                        print(f"[Telegram] Command: {msg}", flush=True)
                        self._handle_command(msg, chat, signal_runner)
        except Exception as e:
            print(f"[Telegram] Poll error: {e}", flush=True)

    def _handle_command(self, msg: str, chat_id: int, runner) -> None:
        msg = msg.strip().lower()

        if msg == "/status" or msg == "/health":
            stats = runner.stats()
            latest = runner.buffer.latest()
            now = datetime.now(timezone.utc)
            data_age = (now - latest["ts"]).seconds // 60 if latest else 999
            gold_price = f"${latest['close']:.1f}" if latest else "N/A"

            # Current session info
            from pipeline.live.orb_detector import SESSIONS
            current_sess = None
            prev_sess = None
            for i, s in enumerate(SESSIONS):
                t = now.time()
                if s.open_utc <= t <= s.close_utc:
                    current_sess = s
                    break
                elif t > s.close_utc:
                    prev_sess = s

            if current_sess:
                current_str = current_sess.name.upper()
            elif prev_sess:
                current_str = f"IDLE (after {prev_sess.name.upper()})"
            else:
                current_str = "IDLE"

            # Find next session
            next_sess = ""
            for s in SESSIONS:
                if now.time() < s.open_utc:
                    next_open = datetime.combine(now.date(), s.open_utc, tzinfo=timezone.utc)
                    delta = next_open - now
                    next_sess = f"{s.name.upper()} in {delta.seconds//3600}h {(delta.seconds%3600)//60}m"
                    break
            if not next_sess:
                next_sess = "TOKYO tomorrow"

            self.send(
                f"📊 *ORB v2.0 Status*\n\n"
                f"Session: `{current_str}`\n"
                f"Next: `{next_sess}`\n"
                f"Gold: `{gold_price}`\n"
                f"1m candles: `{data_age}m old` (yfinance)\n"
                f"Events: `{runner._total_events}` | Signals: `{runner._total_signals}`\n"
                f"Portfolio: `${runner.portfolio.balance:,.0f}` "
                f"({len(runner.portfolio.open_positions)} open)"
            )

        elif msg == "/last":
            # Show from portfolio (persisted) first, then live signals
            total = len(runner.portfolio.closed_trades) + len(runner.portfolio.open_positions)
            if total > 0:
                lines = [f"📡 *Today's Activity* ({total} trades/positions)\n"]
                for pos in runner.portfolio.open_positions:
                    lines.append(f"🔵 OPEN {pos['direction']} @ `${pos['entry']:.1f}` | PnL: `${pos['unrealized_pnl']:+,.0f}`")
                for t in runner.portfolio.closed_trades[-5:]:
                    emoji = "✅" if t["pnl"] > 0 else "❌"
                    lines.append(f"{emoji} {t['direction']} `${t['pnl']:+,.0f}` ({t['close_reason']})")
                self.send("\n".join(lines))
            elif runner.signals:
                last = runner.signals[-1]
                ts = pd.Timestamp(last["ts"]).strftime("%H:%M UTC")
                self.send(
                    f"📡 *Last Signal*\n\n"
                    f"Time: `{ts}`\n"
                    f"Session: *{last['session'].upper()}*\n"
                    f"Decision: *{last['decision']}*\n"
                    f"Entry: `${last['entry']:.1f}`"
                )
            else:
                self.send("📡 *No signals yet today.*")

        elif msg == "/health":
            self.send("Use /status for health info.")

        elif msg == "/pnl":
            self.send(runner.portfolio.pnl_summary() or "No positions")

        elif msg == "/portfolio":
            self.send(runner.portfolio.stats())
        elif msg == "/features":
            if runner.signals:
                last = runner.signals[-1]
                self.send(
                    f"📐 *Last Signal Features*\n\n"
                    f"Prob(Rev): `{last['prob_rev']:.3f}`\n"
                    f"Prob(Cont): `{last['prob_cont']:.3f}`\n"
                    f"Session: `{last['session']}` | Side: `{last['side']}`\n"
                    f"Entry: `${last['entry']:.1f}`"
                )
            else:
                latest = runner.buffer.latest()
                self.send(
                    f"📐 *Current Market*\n\n"
                    f"Gold: `${latest['close']:.1f}`\n"
                    f"Data: yfinance (~10-15m delay)\n"
                    f"Features: 42 total, 7 modules"
                )
        elif msg.startswith("/"):
            self.send("Commands: /status /last /pnl /portfolio /features")


# ── Portfolio Tracker ────────────────────────────────────────────────────────


class PortfolioTracker:
    """Tracks open positions and trade history for paper trading."""

    def __init__(self, risk_per_r: float = 100.0, state_dir: Path | None = None):
        self.risk_per_r = risk_per_r
        self.open_positions: list[dict] = []
        self.closed_trades: list[dict] = []
        self.balance = 0.0
        self.start_balance = 0.0
        self._state_dir = state_dir
        self._load_state()

    def _state_file(self):
        if self._state_dir:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            return self._state_dir / "portfolio_state.json"
        return ROOT / "data" / "Live" / "portfolio_state.json"

    def _load_state(self):
        f = self._state_file()
        if f.exists():
            try:
                data = json.loads(f.read_text())
                self.closed_trades = data.get("closed_trades", [])
                self.balance = data.get("balance", 0.0)
                self.start_balance = data.get("start_balance", 0.0)
            except Exception:
                pass
        if self.start_balance == 0.0:
            self.start_balance = 50000.0  # Topstep starting balance
        if self.balance == 0.0:
            self.balance = self.start_balance

    def _save_state(self):
        f = self._state_file()
        try:
            data = {
                "closed_trades": self.closed_trades[-500:],
                "balance": self.balance,
                "start_balance": self.start_balance,
                "open_positions": self.open_positions,
            }
            f.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            pass

    def open(self, signal: dict) -> dict | None:
        """Open a position from a signal. Closes opposite position if exists."""
        direction = "SHORT" if (signal["side"] == "BULL" and signal["decision"] == "REV") or \
                               (signal["side"] == "BEAR" and signal["decision"] == "CONT") else "LONG"

        # Close opposite positions first
        for pos in list(self.open_positions):
            if pos["direction"] != direction:
                self._close_position(pos, 0.0, "REVERSED")

        # Open new position
        pos = {
            "id": len(self.closed_trades) + len(self.open_positions) + 1,
            "direction": direction,
            "session": signal["session"],
            "entry": signal["entry"],
            "tp": signal["tp"],
            "sl": signal["sl"],
            "rr": signal["rr_ratio"],
            "open_time": str(signal["ts"]),
            "open_price": signal["entry"],
            "current_price": signal["entry"],
            "unrealized_pnl": 0.0,
        }
        self.open_positions.append(pos)
        self._save_state()
        return pos

    def update(self, current_price: float) -> None:
        """Update unrealized PnL and check TP/SL for all open positions."""
        for pos in list(self.open_positions):
            pos["current_price"] = current_price
            if pos["direction"] == "LONG":
                pnl = (current_price - pos["entry"]) * self.risk_per_r / (abs(pos["entry"] - pos["sl"]) + 0.01)
            else:
                pnl = (pos["entry"] - current_price) * self.risk_per_r / (abs(pos["entry"] - pos["sl"]) + 0.01)
            pos["unrealized_pnl"] = round(pnl, 2)

            # Check TP/SL
            if pos["direction"] == "LONG":
                if current_price >= pos["tp"]:
                    self._close_position(pos, pos["rr"] * self.risk_per_r, "TP")
                elif current_price <= pos["sl"]:
                    self._close_position(pos, -self.risk_per_r, "SL")
            else:
                if current_price <= pos["tp"]:
                    self._close_position(pos, pos["rr"] * self.risk_per_r, "TP")
                elif current_price >= pos["sl"]:
                    self._close_position(pos, -self.risk_per_r, "SL")

    def _close_position(self, pos: dict, pnl: float, reason: str) -> None:
        self.open_positions.remove(pos)
        trade = {
            **pos,
            "pnl": round(pnl - 3.0, 2),  # $3 commission
            "close_reason": reason,
            "close_time": str(datetime.now(timezone.utc)),
        }
        self.closed_trades.append(trade)
        self.balance += trade["pnl"]
        self._save_state()

    def stats(self) -> str:
        total_trades = len(self.closed_trades)
        wins = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        wr = wins / total_trades if total_trades > 0 else 0
        total_pnl = sum(t["pnl"] for t in self.closed_trades)
        open_pnl = sum(p["unrealized_pnl"] for p in self.open_positions)
        return (
            f"💰 *Portfolio*\n\n"
            f"Balance: `${self.balance:,.0f}`\n"
            f"Start: `${self.start_balance:,.0f}`\n"
            f"Total PnL: `${total_pnl:+,.0f}` (closed)\n"
            f"Open PnL: `${open_pnl:+,.0f}` ({len(self.open_positions)} pos)\n"
            f"Trades: `{total_trades}` | Win rate: `{wr:.1%}`"
        )

    def pnl_summary(self) -> str:
        lines = []
        for pos in self.open_positions:
            lines.append(
                f"{'🟢' if pos['direction'] == 'LONG' else '🔴'} {pos['session'].upper()} "
                f"{pos['direction']} @ `${pos['entry']:.1f}` | "
                f"PnL: `${pos['unrealized_pnl']:+,.0f}` | "
                f"TP: `${pos['tp']:.1f}` SL: `${pos['sl']:.1f}`"
            )
        if self.closed_trades:
            last = self.closed_trades[-3:]
            lines.append("\n*Last 3 closed:*")
            for t in last:
                emoji = "✅" if t["pnl"] > 0 else "❌"
                lines.append(f"{emoji} {t['session'].upper()} {t['direction']} "
                           f"`${t['pnl']:+,.0f}` ({t['close_reason']})")
        return "\n".join(lines) if lines else "No positions"



# ── SignalRunner ─────────────────────────────────────────────────────────────


# ── SignalRunner ─────────────────────────────────────────────────────────────


class SignalRunner:
    """End-to-end signal pipeline: detect -> compute -> predict -> filter -> output."""

    def __init__(self, telegram: "TelegramBot | None" = None,
                 buffer: "DataBuffer | None" = None,
                 replay_dir: Path | None = None):
        # Models
        rev_path = MODEL_DIR / f"lgbm_rev_v2_{TARGET}.txt"
        cont_path = MODEL_DIR / f"lgbm_cont_v2_{TARGET}.txt"
        self.rev_model = lgb.Booster(model_file=str(rev_path))
        self.cont_model = lgb.Booster(model_file=str(cont_path))

        self._replay_mode = replay_dir is not None

        # Buffer
        if buffer:
            self.buffer = buffer
        elif self._replay_mode:
            db_path = replay_dir / "replay_buffer.db"
            self.buffer = DataBuffer(db_path=db_path)
        else:
            self.buffer = DataBuffer(db_path=CANARY_DB)

        self.detector = ORBDetector(self.buffer)
        self.features = FeatureBuilder(self.buffer)
        self.portfolio = PortfolioTracker(risk_per_r=POLICY.risk_per_r_usd,
                                          state_dir=replay_dir)

        # Trade execution (disabled in replay)
        self.execution = None
        if not self._replay_mode:
            try:
                from pipeline.live.execute.topstepx import TopstepXExecution
                self.execution = TopstepXExecution()
                print("[Runner] Trade execution enabled (TopstepX REST)", flush=True)
            except Exception as e:
                print(f"[Runner] Trade execution disabled: {e}", flush=True)

        # State
        self.signals: list[dict] = []
        self._total_events = 0
        self._total_signals = 0
        self._dedup_file = (replay_dir / "dedup_state.json") if replay_dir \
                           else (ROOT / "data" / "Live" / "dedup_state.json")
        self._session_breakouts: set = self._load_dedup_state()
        self._last_date = None
        self.telegram = telegram
        self._start_time = datetime(2000, 1, 1, tzinfo=timezone.utc) if self._replay_mode \
                           else datetime.now(timezone.utc)
        self._dedup_file = (replay_dir / "dedup_state.json") if replay_dir \
                           else (ROOT / "data" / "Live" / "dedup_state.json")

    def _load_dedup_state(self) -> set:
        f = ROOT / "data" / "Live" / "dedup_state.json"
        if f.exists():
            try:
                data = json.loads(f.read_text())
                if data.get("date") == str(datetime.now(timezone.utc).date()):
                    return set(tuple(k) for k in data.get("signals", []))
            except Exception:
                pass
        return set()

    def _save_dedup_state(self) -> None:
        f = ROOT / "data" / "Live" / "dedup_state.json"
        try:
            data = {
                "date": str(datetime.now(timezone.utc).date()),
                "signals": [list(k) for k in self._session_breakouts],
            }
            f.write_text(json.dumps(data))
        except Exception:
            pass

    def backfill(self, days: int = 90) -> None:
        """Backfill buffer with historical data."""
        self.buffer.backfill(days=days)

    def live_update(self) -> bool:
        """Fetch latest 1m data from yfinance. Returns True if new data."""
        if self._replay_mode:
            return True  # Replay engine handles inserts
        try:
            n = self.buffer.update()
            return n > 0
        except Exception:
            # yfinance may fail silently
            return False

    def check(self, now: datetime | None = None) -> list[dict]:
        """Run one iteration."""
        now = now or datetime.now(timezone.utc)
        events = self.detector.check(now)
        new_signals = []

        for event in events:
            # Skip events from before daemon startup (prevents re-fire on restart)
            if event.breakout_ts < self._start_time - timedelta(seconds=90):
                continue

            # Dedup: one breakout per (date, session, side) — best ORB-TF
            dedup_key = (event.date, event.session, event.breakout_side)
            if dedup_key in self._session_breakouts:
                continue
            self._session_breakouts.add(dedup_key)
            self._save_dedup_state()
            self._total_events += 1

            signal = self._process_event(event)
            if signal:
                new_signals.append(signal)
                self._total_signals += 1
                self._print_signal(signal)
                self.portfolio.open(signal)

                # Auto-execute on TopstepX
                if self.execution:
                    try:
                        action = self.execution.signal_to_action(signal["side"], signal["decision"])
                        result = self.execution.place_limit_order(
                            action, signal["entry"], quantity=1)
                        print(f"[Exec] Order placed: {result.get('orderId')} {action} {signal['entry']}", flush=True)
                    except Exception as e:
                        print(f"[Exec] Failed: {e}", flush=True)

        # Reset dedup on date change
        if self.detector.current_date is not None and self.detector.current_date != getattr(self, '_last_date', None):
            self._session_breakouts.clear()
        self._last_date = self.detector.current_date

        return new_signals

    def _process_event(self, event: BreakoutEvent) -> dict | None:
        """Process one breakout event end-to-end. Returns signal dict or None (skip)."""
        # ── Compute features ─────────────────────────────────────────
        try:
            feat_array = self.features.build_array(event)
        except Exception as e:
            print(f"[Signal] Feature error: {e}")
            return None

        # ── Predict ──────────────────────────────────────────────────
        prob_rev = float(self.rev_model.predict(feat_array.reshape(1, -1))[0])
        prob_cont = float(self.cont_model.predict(feat_array.reshape(1, -1))[0])

        entry_price = event.entry_price
        orb_range = event.orb_range

        # Placeholder TP/SL — will be fixed after decision
        tp_price = 0.0
        sl_price = 0.0

        signal = {
            "ts": event.breakout_ts,
            "date": event.date,
            "session": event.session,
            "orb_tf": event.orb_tf,
            "side": "BULL" if event.breakout_side == 1 else "BEAR",
            "entry": entry_price,
            "prob_rev": prob_rev,
            "prob_cont": prob_cont,
            "tp": tp_price,
            "sl": sl_price,
            "rr_ratio": RR,
        }

        # ── Simple policy gate ──────────────────────────────────
        rev_threshold = 0.12
        cont_threshold = 0.12

        rev_gate = prob_rev >= rev_threshold
        cont_gate = prob_cont >= cont_threshold

        if rev_gate and not cont_gate:
            signal["decision"] = "REV"
        elif cont_gate and not rev_gate:
            signal["decision"] = "CONT"
        elif rev_gate and cont_gate:
            signal["decision"] = "REV" if prob_rev >= prob_cont else "CONT"
        else:
            signal["decision"] = "SKIP"

        # ── Compute TP/SL based on actual decision ─────────────
        is_bull = event.breakout_side == 1
        if signal["decision"] == "CONT":
            # Go with breakout direction
            signal["tp"] = entry_price + (RR * orb_range) if is_bull else entry_price - (RR * orb_range)
            signal["sl"] = entry_price - orb_range if is_bull else entry_price + orb_range
        elif signal["decision"] == "REV":
            # Go opposite breakout direction (fade)
            signal["tp"] = entry_price - (RR * orb_range) if is_bull else entry_price + (RR * orb_range)
            signal["sl"] = entry_price + orb_range if is_bull else entry_price - orb_range
        # else SKIP — tp/sl stay at 0

        self.signals.append(signal)

        if signal["decision"] == "SKIP":
            return None
        return signal

    def _print_signal(self, sig: dict) -> None:
        """Pretty-print a trading signal."""
        ts = pd.Timestamp(sig["ts"]).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n{'='*60}")
        print(f"  SIGNAL — {ts}")
        print(f"{'='*60}")
        print(f"  Session:   {sig['session'].upper()}")
        print(f"  Breakout:  {sig['side']} ({sig['orb_tf']})")
        print(f"  Decision:  {sig['decision']}")
        print(f"  Prob(Rev): {sig['prob_rev']:.3f}   Prob(Cont): {sig['prob_cont']:.3f}")
        print(f"  Entry:    ${sig['entry']:.1f}")
        print(f"  TP:      ${sig['tp']:.1f} ({sig['rr_ratio']}R)")
        print(f"  SL:      ${sig['sl']:.1f} (1R)")
        print(f"{'='*60}")

        # Publish to SignalBus → all subscribed users
        try:
            SignalBus().publish("orb_v2", sig)
        except Exception:
            pass

    def stats(self) -> str:
        return (
            f"Events: {self._total_events} | "
            f"Signals: {self._total_signals} | "
            f"Signal rate: {self._total_signals / max(1, self._total_events) * 100:.1f}%"
        )

    def run_simulation(self, days: int = 30) -> None:
        """Run through the last *days* of data and show all signals."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        print(f"[Sim] Running through {start.date()} → {end.date()} ({days} days)...")
        print(f"[Sim] Target: {TARGET} ({RR}R), Risk: $100/1R")
        print()

        df = self.buffer.get(
            start.strftime("%Y-%m-%d %H:%M:%S"),
            end.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if df.empty:
            print("[Sim] No data available. Run --backfill first.")
            return

        print(f"[Sim] Scanning {len(df):,} candles...")

        for _, candle in df.iterrows():
            ts = candle["timestamp_utc"].to_pydatetime()
            self.check(ts)

        print(f"\n[Sim] Complete. {self.stats()}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ORB v2.0 Live Signal Runner")
    parser.add_argument("--backfill", action="store_true", help="Backfill 90d data from source DB")
    parser.add_argument("--simulate", type=int, default=0, metavar="DAYS",
                        help="Run simulation through last N days of data")
    parser.add_argument("--live", action="store_true", help="Start live monitoring")
    parser.add_argument("--replay", action="store_true", help="Replay historical data via ReplayEngine")
    parser.add_argument("--duration", type=str, default="",
                        help="Replay duration shortcut: 1d, 5d, 30d (offset from now)")
    parser.add_argument("--replay-start", type=str, default="",
                        help="Replay start datetime (UTC, YYYY-MM-DD or YYYY-MM-DD HH:MM)")
    parser.add_argument("--replay-end", type=str, default="",
                        help="Replay end datetime (UTC)")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="Seconds per candle (0=instant, 60=realtime)")
    parser.add_argument("--loop", action="store_true", help="Loop replay from beginning when done")
    parser.add_argument("--keep-all", action="store_true", help="Keep buffer DB after replay")
    parser.add_argument("--purge", action="store_true", help="Delete all artifacts after replay")
    args = parser.parse_args()

    bot = TelegramBot()

    if bot.enabled:
        print(f"[Telegram] Connected to chat {bot.chat_id}", flush=True)

    # ── Replay mode ──────────────────────────────────────────────────────
    if args.replay:
        from pipeline.live.replay_engine import ReplayEngine

        # Parse time range
        if args.duration:
            days = int(args.duration.replace("d", ""))
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=days)
        elif args.replay_start:
            start_dt = datetime.fromisoformat(args.replay_start).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(args.replay_end).replace(tzinfo=timezone.utc) if args.replay_end \
                     else datetime.now(timezone.utc)
        else:
            print("[Replay] Must specify --duration or --replay-start", flush=True)
            sys.exit(1)

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        replay_dir = ROOT / "data" / "Live" / "replay" / run_id

        # Create isolated runner and TV strategy (share same buffer)
        runner = SignalRunner(telegram=bot, replay_dir=replay_dir)
        runner.portfolio.start_balance = 50000.0
        runner.portfolio.balance = 50000.0

        from pipeline.live.tv_strategy import TVStrategy
        tv = TVStrategy(buffer_or_db_path=runner.buffer)

        # Replay on_bar callback: check both ORB and TV per bar
        def _on_replay_bar(bar: dict) -> None:
            try:
                now_ts = pd.Timestamp(bar["timestamp_utc"], tz="UTC").to_pydatetime()
                # ORB v2.0 check
                signals = runner.check(now=now_ts)
                if signals:
                    for s in signals:
                        print(f"SIGNAL|{s['ts']}|{s['session']}|{s['side']}|{s['decision']}|"
                              f"entry={s['entry']}|TP={s['tp']}|SL={s['sl']}", flush=True)
                # TV Strategy check (only on 5m bar boundaries)
                if bar["epoch_ms"] % 300_000 == 0:
                    tv_signals = tv.check(now=now_ts)
                    for tsig in tv_signals:
                        pass  # _store_signal already prints + publishes
                # Portfolio update
                latest = runner.buffer.latest()
                if latest:
                    runner.portfolio.update(latest["close"])
            except Exception as exc:
                print(f"[Replay] on_bar error: {exc}", flush=True)

        engine = ReplayEngine(
            buffer=runner.buffer,
            start_dt=start_dt,
            end_dt=end_dt,
            pause=args.pause,
            loop=args.loop,
            run_id=run_id,
            keep_all=getattr(args, "keep_all", False),
            purge=args.purge,
            on_bar=_on_replay_bar,
        )

        import threading as _th
        engine_thread = _th.Thread(target=engine.run, daemon=True)
        engine_thread.start()
        print(f"[Replay] Engine started | {start_dt} → {end_dt} | pause={args.pause}s | "
              f"run={run_id}", flush=True)

        # Start webhook
        webhook = WebhookServer(port=8080)
        webhook.start()

        import time as _time
        loop_count = 0
        print("[Replay] Monitoring loop started (Ctrl+C to stop)...", flush=True)
        try:
            while engine_thread.is_alive():
                if bot.enabled:
                    try:
                        bot.poll(runner)
                    except Exception as e:
                        print(f"[Replay] Poll error: {e}", flush=True)
                _time.sleep(5)
                loop_count += 1
                if loop_count % 60 == 0 and loop_count > 0:
                    print(f"[Replay] {engine.status_str()} | signals={len(runner.signals)}", flush=True)
        except KeyboardInterrupt:
            print(f"\n[Replay] Stopped. {runner.stats()}", flush=True)
            engine.stop()
            engine_thread.join(timeout=5)

        print(f"[Replay] Session complete — {run_id}", flush=True)
        sys.exit(0)

    # ── Live / Simulate mode ─────────────────────────────────────────────
    runner = SignalRunner(telegram=bot)

    if args.backfill:
        runner.backfill()

    if args.simulate:
        runner.run_simulation(days=args.simulate)

    if args.live:
        import time as _time
        import threading

        print("[Live] Starting live monitoring...", flush=True)
        print(f"[Live] Strategy: {TARGET} ({RR:.0f}R), Risk: $100/1R", flush=True)
        print(f"[Live] Sessions: Tokyo 00:00, London 07:00, US 13:30 UTC", flush=True)

        # Start TV Strategy in daemon thread
        from pipeline.live.tv_strategy import TVStrategy
        tv = TVStrategy()
        threading.Thread(target=tv.run_live, daemon=True).start()
        print("[Live] TV Strategy thread started", flush=True)

        # Start webhook receiver for TradingView → Gmail → alerts
        webhook = WebhookServer(port=8080)
        webhook.start()

        loop_count = 0
        while True:
            try:
                # Main check (every 60s = every 12th loop of 5s)
                if loop_count % 12 == 0:
                    if loop_count > 0:
                        runner.live_update()
                    else:
                        latest = runner.buffer.latest()
                        if latest:
                            print(f"[Live] Heartbeat — latest candle: {latest['ts']} close={latest['close']:.1f}", flush=True)

                    signals = runner.check()
                    if signals:
                        for s in signals:
                            print(f"SIGNAL|{s['ts']}|{s['session']}|{s['side']}|{s['decision']}|"
                                  f"entry={s['entry']}|TP={s['tp']}|SL={s['sl']}", flush=True)

                    # Update portfolio with latest price
                    latest = runner.buffer.latest()
                    if latest:
                        runner.portfolio.update(latest["close"])

                # Poll Telegram every 5 seconds for snappy commands
                if bot.enabled:
                    try:
                        bot.poll(runner)
                    except Exception as e:
                        print(f"[Live] Poll error: {e}", flush=True)

                _time.sleep(5)
                loop_count += 1

                # Periodic heartbeat (every 5 min)
                if loop_count % 60 == 0 and loop_count > 0:
                    latest = runner.buffer.latest()
                    px = f" close={latest['close']:.1f}" if latest else ""
                    print(f"[Live] Alive | loop={loop_count} | signals={len(runner.signals)} | "
                          f"{runner.stats()}{px}", flush=True)

            except KeyboardInterrupt:
                print(f"\n[Live] Stopped. {runner.stats()}", flush=True)
                break
            except Exception as e:
                print(f"[Live] Error: {e}", flush=True)
                _time.sleep(60)

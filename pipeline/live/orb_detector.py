#!/usr/bin/env python3
"""
ORB (Opening Range Breakout) detection engine.

Monitors 1m candles in real time, detects session boundaries, computes
ORB range, and triggers breakout events.

Session convention (UTC):
    Tokyo:  00:00 - 03:00  (ORB: 00:00 - 00:15)
    London: 07:00 - 10:00  (ORB: 07:00 - 07:15)
    US:     13:30 - 16:30  (ORB: 13:30 - 13:45)

ORB Timeframes:
    ORB-5m:  3 candles of 5m (= 15 min)
    ORB-15m: 1 candle of 15m (= 15 min)
    ORB-30m: 2 candles of 15m (= 30 min)

The engine uses only 1m data internally and resamples as needed.

Usage:
    from pipeline.live.orb_detector import ORBDetector
    detector = ORBDetector(buffer)
    detector.on_candle(candle)  # call each minute
    if detector.breakout:
        signal = detector.get_breakout()
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Callable

import pandas as pd

from pipeline.live.buffer import DataBuffer

# ── session definitions ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Session:
    name: str
    open_utc: time       # session open
    close_utc: time      # session close
    orb_end_utc: time    # ORB formation complete

SESSIONS = [
    Session("tokyo",  time(0, 0),  time(3, 0),  time(0, 15)),
    Session("london", time(7, 0),  time(10, 0), time(7, 15)),
    Session("us",     time(13, 30), time(16, 30), time(13, 45)),
]

ORB_TIMEFRAMES = {
    "5m":  {"minutes": 15, "candles": 3},
    "15m": {"minutes": 15, "candles": 1},
    "30m": {"minutes": 30, "candles": 2},
}


@dataclass
class BreakoutEvent:
    date: datetime.date
    session: str
    orb_tf: str
    orb_start_ts: pd.Timestamp
    orb_end_ts: pd.Timestamp
    orb_high: float
    orb_low: float
    orb_range: float
    breakout_ts: pd.Timestamp
    breakout_side: int  # 1 = bull, -1 = bear
    entry_price: float
    session_close_ts: pd.Timestamp


class ORBDetector:
    """Monitors 1m candles and triggers breakout events for all sessions+TFs."""

    def __init__(self, buffer: DataBuffer):
        self.buffer = buffer
        self.current_date: datetime.date | None = None
        self.active_session: Session | None = None
        self.orb_ranges: dict[str, dict] = {}  # {tf: {high, low, start_ts, end_ts}}
        self.breakout_events: list[BreakoutEvent] = []
        self._on_breakout: Callable | None = None
        self._session_opened = False

    def on_breakout(self, callback: Callable[[BreakoutEvent], None]) -> None:
        """Register callback for breakout events."""
        self._on_breakout = callback

    def check(self, now: datetime | None = None) -> list[BreakoutEvent]:
        """Check current session state with latest candle. Call once per minute.

        Returns list of new breakout events (usually 0-3: one per ORB-TF).
        """
        now = now or datetime.now(timezone.utc)
        today = now.date()

        # ── Date change → reset ──────────────────────────────────────
        if self.current_date != today:
            self.current_date = today
            self._session_opened = False
            self.orb_ranges = {}
            self.breakout_events = []

        # ── Find active session ───────────────────────────────────────
        sess = self._find_session(now)
        if sess is None:
            self._session_opened = False
            return []

        # ── Session just started → compute ORB ───────────────────────
        if not self._session_opened:
            orb_complete = self._is_orb_complete(sess, now)
            if not orb_complete:
                return []  # still forming ORB

            self.active_session = sess
            self._session_opened = True
            self.orb_ranges = self._compute_orb_ranges(sess, now)

        # ── Session is open → check breakout ─────────────────────────
        if self.active_session != sess:
            self._session_opened = False
            return []  # session ended

        return self._check_breakouts(now)

    def _find_session(self, now: datetime) -> Session | None:
        """Find which session *now* falls into."""
        current_time = now.time()
        for sess in SESSIONS:
            if sess.open_utc <= current_time <= sess.close_utc:
                return sess
        return None

    def _is_orb_complete(self, sess: Session, now: datetime) -> bool:
        """Check if ORB formation period is complete for the session."""
        return now.time() >= sess.orb_end_utc

    def _compute_orb_ranges(self, sess: Session, now: datetime) -> dict:
        """Compute ORB high/low per timeframe for the current session."""
        date_str = now.date().isoformat()

        # Get session candles from buffer
        sess_start = datetime.combine(now.date(), sess.open_utc, tzinfo=timezone.utc)
        sess_end = datetime.combine(now.date(), sess.close_utc, tzinfo=timezone.utc)
        df = self.buffer.get(
            sess_start.strftime("%Y-%m-%d %H:%M:%S"),
            sess_end.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if df.empty:
            return {}

        df = df.set_index("timestamp_utc")
        orb_ranges = {}

        for tf_name, tf_config in ORB_TIMEFRAMES.items():
            freq_map = {"5m": "5min", "15m": "15min", "30m": "15min"}
            freq = freq_map[tf_name]
            candles = tf_config["candles"]

            # Resample
            if tf_name == "30m":
                # 30m ORB = first 2 candles of 15m
                resampled = df.resample("15min").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last"}
                ).dropna()
                orb_candles = resampled.iloc[:candles]
            else:
                resampled = df.resample(freq).agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last"}
                ).dropna()
                orb_candles = resampled.iloc[:candles]

            if len(orb_candles) < candles:
                continue

            orb_high = orb_candles["high"].max()
            orb_low = orb_candles["low"].min()
            orb_start = orb_candles.index[0]
            orb_end = orb_candles.index[-1] + pd.Timedelta(minutes=int(freq_map[tf_name][0]))

            orb_ranges[tf_name] = {
                "high": orb_high,
                "low": orb_low,
                "range": orb_high - orb_low,
                "start_ts": orb_start,
                "end_ts": orb_end,
            }

        return orb_ranges

    def _check_breakouts(self, now: datetime) -> list[BreakoutEvent]:
        """Check all ORB-TFs for breakout on latest 1m candle."""
        latest = self.buffer.latest()
        if latest is None:
            return []

        current_price = latest["close"]
        current_ts = latest["ts"]
        sess = self.active_session
        new_events = []

        for tf_name, orb in self.orb_ranges.items():
            # Already broke out in this TF this session? skip
            already_broke = any(
                e.session == sess.name and e.orb_tf == tf_name
                for e in self.breakout_events
            )
            if already_broke:
                continue

            if current_price > orb["high"]:
                side = 1  # bull breakout
            elif current_price < orb["low"]:
                side = -1  # bear breakout
            else:
                continue

            event = BreakoutEvent(
                date=now.date(),
                session=sess.name,
                orb_tf=tf_name,
                orb_start_ts=orb["start_ts"],
                orb_end_ts=orb["end_ts"],
                orb_high=orb["high"],
                orb_low=orb["low"],
                orb_range=orb["range"],
                breakout_ts=current_ts,
                breakout_side=side,
                entry_price=current_price,
                session_close_ts=datetime.combine(now.date(), sess.close_utc, tzinfo=timezone.utc),
            )

            self.breakout_events.append(event)
            new_events.append(event)

            if self._on_breakout is not None:
                self._on_breakout(event)

        return new_events


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ORB detector test")
    parser.add_argument("--backfill", action="store_true", help="Backfill data first")
    args = parser.parse_args()

    buffer = DataBuffer()
    if args.backfill:
        buffer.backfill()

    def on_break(event: BreakoutEvent) -> None:
        side = "BULL" if event.breakout_side == 1 else "BEAR"
        print(f"\n🔔 BREAKOUT [{event.orb_tf}] {event.session} {side} @ {event.entry_price}")

    detector = ORBDetector(buffer)
    detector.on_breakout(on_break)

    # Simulate by going through buffered data minute by minute
    print("Running through buffered data...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    df = buffer.get(start.strftime("%Y-%m-%d %H:%M:%S"),
                     end.strftime("%Y-%m-%d %H:%M:%S"))

    for _, candle in df.iterrows():
        ts = candle["timestamp_utc"].to_pydatetime()
        events = detector.check(ts)
        for evt in events:
            pass  # callback already printed

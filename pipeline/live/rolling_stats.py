#!/usr/bin/env python3
"""
Rolling statistics manager for online feature computation.

Tracks rolling history of ATR14, breakout_strength, and ORB range for
percentile rank and z-score computation in live inference.

Uses SQLite for persistence — survives process restarts.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "data" / "Live" / "rolling_stats.db"


class RollingStats:
    """Tracks rolling window of feature values for (session, orb_tf) pair."""

    def __init__(self, window_days: int = 28, min_samples: int = 5):
        self.window = window_days
        self.min_samples = min_samples
        self.db_path = DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session TEXT NOT NULL,
                    orb_tf TEXT NOT NULL,
                    ts_utc TEXT NOT NULL,
                    atr14 REAL,
                    breakout_strength REAL,
                    orb_range REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_group
                ON feature_history(session, orb_tf, ts_utc)
            """)
            conn.commit()

    def record(self, session: str, orb_tf: str, ts: datetime,
               atr14: float, breakout_strength: float, orb_range: float) -> None:
        """Append a feature snapshot to history."""
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO feature_history (session, orb_tf, ts_utc, atr14, breakout_strength, orb_range) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [session, orb_tf, ts_str, float(atr14) if np.isfinite(atr14) else None,
                 float(breakout_strength) if np.isfinite(breakout_strength) else None,
                 float(orb_range) if np.isfinite(orb_range) else None],
            )
            conn.commit()

    def get_history(self, session: str, orb_tf: str) -> dict[str, np.ndarray]:
        """Return arrays of atr14, breakout_strength, orb_range for the window."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.window)).strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT atr14, breakout_strength, orb_range FROM feature_history "
                "WHERE session = ? AND orb_tf = ? AND ts_utc >= ? "
                "ORDER BY ts_utc",
                [session, orb_tf, cutoff],
            ).fetchall()

        atr = np.array([r[0] for r in rows if r[0] is not None], dtype=float)
        bs = np.array([r[1] for r in rows if r[1] is not None], dtype=float)
        orb = np.array([r[2] for r in rows if r[2] is not None], dtype=float)
        return {"atr14": atr, "breakout_strength": bs, "orb_range": orb}

    def percentile_rank(self, value: float, history: np.ndarray) -> float:
        """Percentile rank of value within history, 0..1."""
        clean = history[np.isfinite(history)]
        if len(clean) < self.min_samples:
            return 0.5  # neutral
        rank = (clean <= value).sum() / len(clean)
        return float(np.clip(rank, 0.01, 0.99))

    def z_score(self, value: float, history: np.ndarray, cap: float = 3.0) -> float:
        """Z-score of value relative to history, clipped to [-cap, cap]."""
        clean = history[np.isfinite(history)]
        if len(clean) < self.min_samples:
            return 0.0  # neutral
        mu = np.mean(clean)
        std = np.std(clean, ddof=0)
        if std < 1e-8:
            return 0.0
        z = (value - mu) / std
        return float(np.clip(z, -cap, cap))

    def compute_features(self, session: str, orb_tf: str,
                         atr14: float, breakout_strength: float,
                         orb_range: float) -> dict[str, float]:
        """Compute all vol_norm features from rolling history."""
        hist = self.get_history(session, orb_tf)

        # Use broader 60-day window for the percentile
        atr_percentile = self.percentile_rank(atr14, hist["atr14"])
        atr_z = self.z_score(atr14, hist["atr14"])
        bs_pct = self.percentile_rank(breakout_strength, hist["breakout_strength"])
        bs_z = self.z_score(breakout_strength, hist["breakout_strength"])
        orb_pct = self.percentile_rank(orb_range, hist["orb_range"])

        return {
            "atr14_percentile_20d": atr_percentile,
            "atr14_zscore_20d": atr_z,
            "breakout_strength_percentile_20d": bs_pct,
            "breakout_strength_zscore_10d": bs_z,
            "orb_range_percentile_20d": orb_pct,
        }

    def count(self) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            return conn.execute("SELECT COUNT(*) FROM feature_history").fetchone()[0]

    def prime_from_buffer(self, buffer) -> None:
        """Backfill rolling stats from historical data by scanning for breakouts.

        This pre-warms the rolling history so percentile/z-scores have enough samples.
        """
        from pipeline.live.orb_detector import ORBDetector

        detector = ORBDetector(buffer)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=90)

        df = buffer.get(
            start.strftime("%Y-%m-%d %H:%M:%S"),
            end.strftime("%Y-%m-%d %H:%M:%S"),
        )

        print(f"[Stats] Priming rolling history from {len(df):,} candles...")
        primed = 0

        for _, candle in df.iterrows():
            ts = candle["timestamp_utc"].to_pydatetime()
            events = detector.check(ts)
            for evt in events:
                # We don't need actual ATR for priming — just price ranges
                self.record(
                    evt.session, evt.orb_tf, evt.breakout_ts,
                    atr14=evt.orb_range * 1.5 if evt.orb_range > 0 else 1.0,
                    breakout_strength=0.5,
                    orb_range=evt.orb_range,
                )
                primed += 1

        print(f"[Stats] Primed with {primed} bootstrapped entries ({self.count()} total)")

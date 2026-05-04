#!/usr/bin/env python3
"""
ReplayEngine — feed historical 1m bars into DataBuffer at simulated speed.

Reads from MGC_1m.db (immutable source), inserts bar-by-bar into a
DataBuffer SQLite with configurable pause between candles. Supports
warmup (bulk pre-load), loop mode, and cleanup policies.

Usage:
    from pipeline.live.replay_engine import ReplayEngine
    engine = ReplayEngine(buffer, start, end, pause=1.0, loop=False)
    engine.run()  # blocking
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_DB = ROOT / "data" / "Level_0_Raw" / "MGC_1m.db"
REPLAY_ROOT = ROOT / "data" / "Live" / "replay"
SYMBOL = "MICRO_GOLD"
TIMEFRAME = "1m"


class ReplayEngine:
    """Feed historical 1m OHLCV bars into a DataBuffer with time simulation."""

    def __init__(
        self,
        buffer,                          # DataBuffer instance
        start_dt: datetime,
        end_dt: datetime,
        pause: float = 0.0,              # seconds per candle (0 = instant)
        loop: bool = False,              # restart from beginning when done
        warmup_days: int = 90,
        run_id: str = "",
        keep_all: bool = False,
        purge: bool = False,
        on_bar: "callable | None" = None,   # callback(bar_dict) after each insert
    ):
        self.buffer = buffer
        self.start = start_dt
        self.end = end_dt
        self.pause = float(pause)
        self.loop = loop
        self.warmup_days = warmup_days
        self.keep_all = keep_all
        self.purge = purge
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.on_bar = on_bar

        self.run_dir = REPLAY_ROOT / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.run_dir / "run_manifest.json"
        self._save_manifest()

        self.bars_total = 0
        self.bars_done = 0
        self.current_ts: str | None = None
        self.is_running = False
        self._stop_event = threading.Event()
        self._error: str | None = None

    # ── manifest ─────────────────────────────────────────────────────────────

    def _save_manifest(self) -> None:
        m = {
            "run_id": self.run_id,
            "start": str(self.start),
            "end": str(self.end),
            "pause": self.pause,
            "loop": self.loop,
            "warmup_days": self.warmup_days,
            "keep_all": self.keep_all,
            "purge": self.purge,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.manifest_path.write_text(json.dumps(m, indent=2, default=str))

    # ── warmup ────────────────────────────────────────────────────────────────

    def warmup(self) -> int:
        """Bulk-copy warmup data from source DB into buffer. No time delay."""
        warmup_start = self.start - timedelta(days=self.warmup_days)
        start_str = self.start.strftime("%Y-%m-%d %H:%M:%S")
        warmup_str = warmup_start.strftime("%Y-%m-%d %H:%M:%S")

        print(f"[Replay] Warmup: {warmup_str} → {start_str} ({self.warmup_days}d)", flush=True)

        with sqlite3.connect(str(SOURCE_DB)) as src:
            df = pd.read_sql(
                "SELECT epoch_ms, timestamp_utc, open, high, low, close, volume "
                "FROM investing_ohlcv_1m "
                "WHERE symbol = ? AND timeframe = ? "
                "AND timestamp_utc >= ? AND timestamp_utc < ? "
                "ORDER BY epoch_ms",
                src,
                params=[SYMBOL, TIMEFRAME, warmup_str, start_str],
            )

        if df.empty:
            print("[Replay] Warning: no warmup rows found. Check MGC_1m.db range.", flush=True)
            return 0

        with sqlite3.connect(str(self.buffer.db_path)) as conn:
            inserted = 0
            for _, row in df.iterrows():
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO ohlcv_1m "
                        "(symbol, timeframe, epoch_ms, timestamp_utc, "
                        "open, high, low, close, volume) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [SYMBOL, TIMEFRAME,
                         int(row["epoch_ms"]), str(row["timestamp_utc"]),
                         float(row["open"]), float(row["high"]), float(row["low"]),
                         float(row["close"]), float(row["volume"])],
                    )
                    inserted += 1
                except Exception:
                    continue
            conn.commit()

        print(f"[Replay] Warmup done — {inserted:,} bars inserted", flush=True)
        return inserted

    # ── bar source ────────────────────────────────────────────────────────────

    def _iter_bars(self):
        """Generator yielding dict bars from start to end (excludes warmup)."""
        start_str = self.start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = self.end.strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(str(SOURCE_DB)) as src:
            cursor = src.execute(
                "SELECT epoch_ms, timestamp_utc, open, high, low, close, volume "
                "FROM investing_ohlcv_1m "
                "WHERE symbol = ? AND timeframe = ? "
                "AND timestamp_utc >= ? AND timestamp_utc <= ? "
                "ORDER BY epoch_ms",
                [SYMBOL, TIMEFRAME, start_str, end_str],
            )
            for row in cursor:
                yield {
                    "epoch_ms": int(row[0]),
                    "timestamp_utc": str(row[1]),
                    "open": float(row[2]),
                    "high": float(row[3]),
                    "low": float(row[4]),
                    "close": float(row[5]),
                    "volume": float(row[6]),
                }

    def _count_bars(self) -> int:
        start_str = self.start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = self.end.strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(str(SOURCE_DB)) as src:
            row = src.execute(
                "SELECT COUNT(*) FROM investing_ohlcv_1m "
                "WHERE symbol = ? AND timeframe = ? "
                "AND timestamp_utc >= ? AND timestamp_utc <= ?",
                [SYMBOL, TIMEFRAME, start_str, end_str],
            ).fetchone()
        return row[0] if row else 0

    # ── bar insert ────────────────────────────────────────────────────────────

    def _insert_bar(self, bar: dict) -> None:
        with sqlite3.connect(str(self.buffer.db_path)) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO ohlcv_1m "
                "(symbol, timeframe, epoch_ms, timestamp_utc, "
                "open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [SYMBOL, TIMEFRAME, bar["epoch_ms"], bar["timestamp_utc"],
                 bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]],
            )
            conn.commit()

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.is_running = True
        self._stop_event.clear()

        self.warmup()

        self.bars_total = self._count_bars()
        print(f"[Replay] {self.bars_total:,} bars | {self.start} → {self.end} | "
              f"pause={self.pause}s", flush=True)

        self.bars_done = 0

        for bar in self._iter_bars():
            if self._stop_event.is_set():
                print("[Replay] Stopped by signal", flush=True)
                self.is_running = False
                self._cleanup()
                return

            self._insert_bar(bar)
            self.bars_done += 1
            self.current_ts = bar["timestamp_utc"]

            if self.on_bar:
                try:
                    self.on_bar(bar)
                except Exception:
                    pass

            if self.pause > 0:
                _time.sleep(self.pause)

            if self.bars_done % 120 == 0 or self.bars_done == self.bars_total:
                pct = self.bars_done / max(1, self.bars_total) * 100
                remaining = self.bars_total - self.bars_done
                eta = remaining * self.pause
                print(f"[Replay] {self.current_ts} | {pct:.0f}% | "
                      f"{self.bars_done}/{self.bars_total} | ETA {eta:.0f}s", flush=True)

        self.is_running = False

        if self.loop:
            print("[Replay] Loop mode — restarting from beginning...", flush=True)
            self.bars_done = 0
            self.current_ts = None
            return self.run()

        print(f"[Replay] Done — {self.bars_done:,} bars fed", flush=True)
        self._cleanup()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_event.set()

    def _cleanup(self) -> None:
        if self.purge:
            if self.run_dir.exists():
                shutil.rmtree(self.run_dir)
            print(f"[Replay] Purged: {self.run_dir}", flush=True)
        elif not self.keep_all:
            buffer_path = self.buffer.db_path
            if buffer_path.exists():
                buffer_path.unlink()
            signals_tv = self.run_dir / "signals_tv.json"
            signals_orb = self.run_dir / "signals_orb.json"
            # Temp signal files cleaned up if empty
            for p in [signals_tv, signals_orb]:
                if p.exists():
                    try:
                        data = json.loads(p.read_text())
                        if not data:
                            p.unlink()
                    except Exception:
                        pass
            print(f"[Replay] Buffer cleaned. Artifacts in {self.run_dir}", flush=True)

    # ── monitoring ────────────────────────────────────────────────────────────

    @property
    def progress_pct(self) -> float:
        return self.bars_done / max(1, self.bars_total) * 100

    @property
    def eta_seconds(self) -> float:
        remaining = self.bars_total - self.bars_done
        return remaining * self.pause

    def status_str(self) -> str:
        if not self.is_running and self.bars_done >= self.bars_total:
            return "Replay: Complete"
        eta = self.eta_seconds
        eta_str = f"{int(eta)}s" if eta < 120 else f"{eta/60:.0f}m"
        return (
            f"Replay: {self.current_ts or 'starting'} | "
            f"{self.progress_pct:.0f}% | "
            f"{self.bars_done}/{self.bars_total} bars | "
            f"ETA {eta_str}"
        )

#!/usr/bin/env python3
"""Rebuild Super Structure UI JSON from the live TopstepX buffer.

This is intentionally read-only for trading: it only reads SQLite OHLCV data
and writes UI/backtest artifacts.
"""

from __future__ import annotations

import subprocess
import sys
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
LIVE_DB = ROOT / "data" / "Live" / "topstepx_buffer.db"
SNAPSHOT_DB = Path("/tmp/super_structure_ui_topstepx_snapshot.db")
OUT_PARQUET = ROOT / "data" / "Level_2_Datamart" / "super_structure_trade_events_live.parquet"
BUILDER = ROOT / "pipeline" / "research" / "build_super_structure_trade_events.py"


def snapshot_live_db() -> Path:
    """Create a short-lived SQLite snapshot so the builder does not hold live DB locks."""
    if SNAPSHOT_DB.exists():
        SNAPSHOT_DB.unlink()

    src_uri = f"file:{LIVE_DB}?mode=ro"
    with sqlite3.connect(src_uri, uri=True, timeout=5) as src:
        with sqlite3.connect(str(SNAPSHOT_DB), timeout=5) as dst:
            src.backup(dst)
    return SNAPSHOT_DB


def main() -> int:
    if not LIVE_DB.exists():
        print(f"[ui-rebuild] missing live DB: {LIVE_DB}", flush=True)
        return 2

    # Keep the UI broad enough for recent context while avoiding the stale raw DB.
    # TopstepX live buffer currently starts in late Jan 2026.
    start = "2026-01-29"
    end = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    source_db = snapshot_live_db()

    cmd = [
        sys.executable,
        str(BUILDER),
        "--start",
        start,
        "--end",
        end,
        "--raw-db",
        str(source_db),
        "--table",
        "ohlcv_1m",
        "--out",
        str(OUT_PARQUET),
        "--export-ui",
    ]

    print(f"[ui-rebuild] start={start} end={end} source={LIVE_DB} snapshot={source_db}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    print(f"[ui-rebuild] exit={proc.returncode}", flush=True)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

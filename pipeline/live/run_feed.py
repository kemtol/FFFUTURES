#!/usr/bin/env python3
"""TopstepX WebSocket feed daemon with auto-reconnect, gap detection, and smart repair.

Usage:
    python3 pipeline/live/run_feed.py         # start daemon (foreground)
    python3 pipeline/live/run_feed.py &       # background
    nohup python3 pipeline/live/run_feed.py & # survive terminal close

Cron:
    @reboot /usr/bin/python3 /home/kemal/futures/pipeline/live/run_feed.py >> /home/kemal/futures/data/Live/topstepx_feed.log 2>&1 &

Repair layers:
    1. Periodic: every 5 min while WS is connected, detect gaps > 2min, fill via yfinance
    2. On disconnect: yfinance fill recent gaps, MGC_1m.db fallback for large gaps
    3. On startup: yfinance backfill if data > 10min stale
"""
from __future__ import annotations

import sys, time, threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.live.sources.topstepx import TopstepXFeed, fetch_token, \
    SYMBOL as TX_SYMBOL, RESOLUTION as TX_RES
from pipeline.live.buffer import DataBuffer, CANARY_DB

REPAIR_INTERVAL_SECONDS = 300  # 5 min
STALE_THRESHOLD_MINUTES = 10


def backfill_if_stale(buffer: DataBuffer) -> None:
    """Fill gap: yfinance if recent (<72h), MGC_1m.db if older."""
    latest = buffer.latest()
    if latest is None:
        print(f"[Feed] Buffer empty → backfilling 90d", flush=True)
        buffer.backfill(90)
        return

    age_min = (datetime.now(timezone.utc) - latest["ts"]).total_seconds() / 60
    if age_min > STALE_THRESHOLD_MINUTES:
        print(f"[Feed] Data stale ({age_min:.0f}min old) → repair", flush=True)
        result = buffer.repair(max_gap_hours=72)
        if result["yfinance"] or result["backfill"]:
            print(f"[Feed] Repair: {result['yfinance']} yfinance + "
                  f"{result['backfill']} backfill bars", flush=True)


def periodic_repair(buffer: DataBuffer, stop_event: threading.Event) -> None:
    """Run every REPAIR_INTERVAL_SECONDS: detect gaps and fill from yfinance."""
    while not stop_event.is_set():
        stop_event.wait(REPAIR_INTERVAL_SECONDS)
        if stop_event.is_set():
            break
        try:
            gaps = buffer.detect_gaps(lookback_bars=120, min_gap_minutes=2)
            if gaps:
                for g in gaps:
                    print(f"[Feed] Gap detected: {g['from_ts']} → {g['to_ts']} "
                          f"({g['gap_minutes']:.0f}min)", flush=True)
                result = buffer.repair(max_gap_hours=24)
                if result["yfinance"] or result["backfill"]:
                    print(f"[Feed] Periodic repair: {result['yfinance']} yfinance + "
                          f"{result['backfill']} backfill bars", flush=True)
        except Exception as e:
            print(f"[Feed] Periodic repair error: {e}", flush=True)


def main():
    print(f"[Feed] TopstepX WebSocket daemon starting...", flush=True)

    buffer = DataBuffer(db_path=CANARY_DB)
    backfill_if_stale(buffer)

    reconnect_delay = 5

    while True:
        # Start periodic repair thread
        repair_stop = threading.Event()
        repair_thread = threading.Thread(
            target=periodic_repair, args=(buffer, repair_stop), daemon=True)
        repair_thread.start()

        try:
            feed = TopstepXFeed()
            print(f"[Feed] Connected. Streaming {TX_SYMBOL} {TX_RES}m...", flush=True)
            feed.start()  # blocks until disconnect
        except RuntimeError as e:
            if "No token" in str(e):
                print(f"[Feed] Token expired → refreshing", flush=True)
                try:
                    token = fetch_token()
                    print(f"[Feed] New token acquired", flush=True)
                except Exception as te:
                    print(f"[Feed] Token refresh failed: {te}", flush=True)
            else:
                print(f"[Feed] Error: {e}", flush=True)
        except KeyboardInterrupt:
            print(f"[Feed] Shutting down", flush=True)
            repair_stop.set()
            break
        except Exception as e:
            print(f"[Feed] Disconnected: {e}", flush=True)

        # Stop periodic repair thread
        repair_stop.set()

        # Repair gaps created during disconnect
        print(f"[Feed] Repairing gaps after disconnect...", flush=True)
        try:
            result = buffer.repair(max_gap_hours=72)
            if result["yfinance"] or result["backfill"]:
                print(f"[Feed] Disconnect repair: {result['yfinance']} yfinance + "
                      f"{result['backfill']} backfill bars", flush=True)
            else:
                print(f"[Feed] No gaps to repair", flush=True)
        except Exception as e:
            print(f"[Feed] Disconnect repair failed: {e}", flush=True)

        print(f"[Feed] Reconnecting in {reconnect_delay}s...", flush=True)
        time.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, 60)


if __name__ == "__main__":
    main()

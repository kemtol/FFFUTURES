#!/usr/bin/env python3
"""Generate session momentum feature module.

Hypothesis
----------
The current model sees breakout-level features (ATR, ADX, VWAP, pre-breakout
volatility) but doesn't know **how the session unfolded** before the breakout.
A breakout that occurs during high momentum (large early range, strong direction)
is more likely to follow through than one that occurs in a sluggish session.

By encoding the first 30 minutes of the session and pre-breakout volume
dynamics, we give the model context about session-level momentum — directly
improving 2026 pass rate by filtering false breakouts in low-momentum sessions.

Features
--------
1. ``sm_first_30m_range`` — Price range (high-low) of first 30 min of session,
   normalized by ATR14. High value = volatile/open session = momentum.

2. ``sm_first_30m_direction`` — Net direction bias in first 30 min:
   (close of 30th min - open of 1st min) / ATR14.
   Positive = bullish open, negative = bearish open.

3. ``sm_pre_breakout_volume_ratio`` — Volume in the 15 minutes before breakout
   divided by average per-15-minute volume in the session up to breakout.
   > 1.0 = breakout on above-average volume (conviction).
   < 1.0 = breakout on below-average volume (weak).

   Additionally, ``sm_pre_breakout_volume_z`` — z-score of pre-breakout volume
   relative to session average (capped at ±5).

Grain
-----
``(date, session, orb_tf, breakout_ts)`` — one row per breakout event (34,187 rows).
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── module metadata ──────────────────────────────────────────────────────────

MODULE_NAME = "session_momentum"
MODULES_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data" / "Level_1_Features" / "modules"
)
L1_DIR = MODULES_DIR.parent  # data/Level_1_Features
RAW_DIR = L1_DIR.parent / "Level_0_Raw"

EVENT_KEY = ["date", "session", "orb_tf", "breakout_ts"]

# Session start times (UTC)
SESSIONS: dict[str, tuple[time, time]] = {
    "tokyo":  (time(0, 0),  time(3, 0)),
    "london": (time(7, 0),  time(10, 0)),
    "us":     (time(13, 30), time(16, 30)),
}

FIRST_30M_WINDOW = 30  # minutes
PRE_BO_WINDOW = 15     # minutes of pre-breakout volume to examine


# ── sources ──────────────────────────────────────────────────────────────────


def load_sources() -> dict[str, pd.DataFrame]:
    """Load breakout events + 1m OHLCV from SQLite."""
    bo = pd.read_parquet(L1_DIR / "breakout_events.parquet")
    # Keep date as string (object) — other modules use this dtype for merge
    if bo["date"].dtype != "object":
        bo["date"] = bo["date"].astype(str)

    # Load 1m data
    db_path = RAW_DIR / "MGC_1m.db"
    conn = sqlite3.connect(str(db_path))
    df_1m = pd.read_sql(
        "SELECT epoch_ms, timestamp_utc, open, high, low, close, volume "
        "FROM investing_ohlcv_1m ORDER BY epoch_ms",
        conn,
        parse_dates=["timestamp_utc"],
    )
    conn.close()

    return {"breakout_events": bo, "df_1m": df_1m}


# ── helper ───────────────────────────────────────────────────────────────────


def _session_open_utc(date_str: str, session: str) -> pd.Timestamp:
    """Compute session open timestamp in UTC from date string and session label.

    Parameters
    ----------
    date_str : str
        Date string in ``YYYY-MM-DD`` format.
    session : str
        One of ``"tokyo"``, ``"london"``, ``"us"``.

    Returns
    -------
    pd.Timestamp
        TZ-aware UTC timestamp of session open.
    """
    start_time = SESSIONS[session][0]
    dt = pd.Timestamp(date_str)
    return pd.Timestamp(
        year=dt.year, month=dt.month, day=dt.day,
        hour=start_time.hour, minute=start_time.minute,
        tz="UTC",
    )


def _empty_record(event: pd.Series) -> dict:
    """Create a record with NaN features for an event that has no data."""
    return {
        "date": event["date"],
        "session": event["session"],
        "orb_tf": event["orb_tf"],
        "breakout_ts": event["breakout_ts"],
        "sm_first_30m_range": np.nan,
        "sm_first_30m_direction": np.nan,
        "sm_pre_breakout_volume_ratio": np.nan,
        "sm_pre_breakout_volume_z": np.nan,
    }


# ── feature builder ──────────────────────────────────────────────────────────


def build_features(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build session momentum features.

    For each breakout event:
    1. Determine session open time from ``date`` + session label
    2. Extract first 30 min of 1m candles → range / ATR14, direction / ATR14
    3. Extract volume in 15 min before breakout → ratio vs session avg volume
    """
    bo = sources["breakout_events"].copy()
    df_1m = sources["df_1m"].copy()

    # Ensure breakout_ts is tz-aware (it's UTC)
    bo = bo.sort_values("breakout_ts").reset_index(drop=True)

    # Convert df_1m timestamp_utc to tz-aware for comparison
    if df_1m["timestamp_utc"].dt.tz is None:
        df_1m["timestamp_utc"] = df_1m["timestamp_utc"].dt.tz_localize("UTC")
    df_1m = df_1m.sort_values("timestamp_utc").reset_index(drop=True)

    # Build a sorted numpy array of 1m timestamps for binary search
    ts_1m = df_1m["timestamp_utc"].values.astype("datetime64[ns]")

    # Pre-extract 1m arrays for fast indexing
    high_1m = df_1m["high"].values
    low_1m = df_1m["low"].values
    open_1m = df_1m["open"].values
    close_1m = df_1m["close"].values
    volume_1m = df_1m["volume"].values

    records: list[dict] = []

    for _, event in bo.iterrows():
        bo_ts = event["breakout_ts"]
        atr14 = event["atr14_at_entry"] if pd.notna(event["atr14_at_entry"]) else np.nan

        if pd.isna(bo_ts):
            records.append(_empty_record(event))
            continue

        bo_ts_ns = bo_ts.to_datetime64()

        # ── Determine session open ─────────────────────────────────
        session = event["session"]
        if session not in SESSIONS:
            records.append(_empty_record(event))
            continue

        try:
            session_open_ts = _session_open_utc(event["date"], session)
        except Exception:
            records.append(_empty_record(event))
            continue

        session_open_ns = session_open_ts.to_datetime64()
        session_open_30m_ns = session_open_ns + np.timedelta64(FIRST_30M_WINDOW, "m")

        # Find index of session open in 1m data
        open_idx = np.searchsorted(ts_1m, session_open_ns, side="left")
        open_30m_idx = np.searchsorted(ts_1m, session_open_30m_ns, side="left")

        # ── Feature 1 & 2: First 30 min range and direction ───────
        if open_idx < len(ts_1m) and open_30m_idx > open_idx:
            first_30m_slice = slice(open_idx, open_30m_idx)
            n_candles_30m = open_30m_idx - open_idx

            first_30m_high = np.max(high_1m[first_30m_slice])
            first_30m_low = np.min(low_1m[first_30m_slice])
            first_30m_range = first_30m_high - first_30m_low

            first_30m_open = open_1m[open_idx]
            first_30m_close = close_1m[open_30m_idx - 1]

            # Feature 1: range / ATR14
            sm_first_30m_range = first_30m_range / atr14 if atr14 > 0 and atr14 is not np.nan else np.nan

            # Feature 2: direction / ATR14
            direction = first_30m_close - first_30m_open
            sm_first_30m_direction = direction / atr14 if atr14 > 0 and atr14 is not np.nan else np.nan
        else:
            sm_first_30m_range = np.nan
            sm_first_30m_direction = np.nan

        # ── Feature 3 & 4: Pre-breakout volume ────────────────────
        pre_bo_ns = bo_ts_ns - np.timedelta64(PRE_BO_WINDOW, "m")
        pre_bo_idx = np.searchsorted(ts_1m, pre_bo_ns, side="left")
        bo_idx = np.searchsorted(ts_1m, bo_ts_ns, side="left")

        if pre_bo_idx < bo_idx and pre_bo_idx >= open_idx:
            # Volume in 15 min before breakout
            pre_bo_volume = np.sum(volume_1m[pre_bo_idx:bo_idx])
            n_pre_bo_candles = bo_idx - pre_bo_idx

            # Average volume in session up to breakout (per minute, then scale to 15min)
            session_to_bo_slice = slice(open_idx, bo_idx)
            n_session_candles = bo_idx - open_idx
            if n_session_candles > 0:
                session_volume_sum = np.sum(volume_1m[session_to_bo_slice])
                avg_volume_per_15min = (session_volume_sum / n_session_candles) * PRE_BO_WINDOW

                if avg_volume_per_15min > 0:
                    sm_pre_breakout_volume_ratio = pre_bo_volume / avg_volume_per_15min

                    # Z-score of pre-breakout volume relative to session candles
                    if n_session_candles >= 5:
                        session_volumes = volume_1m[session_to_bo_slice]
                        vol_mean = np.mean(session_volumes)
                        vol_std = np.std(session_volumes)
                        if vol_std > 0:
                            avg_pre_bo_per_min = pre_bo_volume / n_pre_bo_candles if n_pre_bo_candles > 0 else 0
                            z = (avg_pre_bo_per_min - vol_mean) / vol_std
                            sm_pre_breakout_volume_z = float(np.clip(z, -5.0, 5.0))
                        else:
                            sm_pre_breakout_volume_z = 0.0
                    else:
                        sm_pre_breakout_volume_z = np.nan
                else:
                    sm_pre_breakout_volume_ratio = 1.0  # No volume → neutral
                    sm_pre_breakout_volume_z = 0.0
            else:
                sm_pre_breakout_volume_ratio = np.nan
                sm_pre_breakout_volume_z = np.nan
        else:
            sm_pre_breakout_volume_ratio = np.nan
            sm_pre_breakout_volume_z = np.nan

        records.append({
            "date": event["date"],
            "session": event["session"],
            "orb_tf": event["orb_tf"],
            "breakout_ts": bo_ts,
            "sm_first_30m_range": sm_first_30m_range,
            "sm_first_30m_direction": sm_first_30m_direction,
            "sm_pre_breakout_volume_ratio": sm_pre_breakout_volume_ratio,
            "sm_pre_breakout_volume_z": sm_pre_breakout_volume_z,
        })

    result = pd.DataFrame(records)

    # ── Ensure float32 dtypes ─────────────────────────────────────
    feat_cols = [c for c in result.columns if c not in EVENT_KEY]
    for col in feat_cols:
        result[col] = result[col].astype("float32")

    return result


# ── conflict checker + main ──────────────────────────────────────────────────


def check_column_conflict(df: pd.DataFrame) -> list[str]:
    """Check if feature column names collide with existing modules."""
    feat_cols = set(c for c in df.columns if c not in EVENT_KEY)
    conflicts: list[str] = []
    for fpath in sorted(MODULES_DIR.glob("*_features.parquet")):
        existing_cols = set(pd.read_parquet(fpath).columns)
        existing_feats = existing_cols - set(EVENT_KEY)
        overlap = feat_cols & existing_feats
        if overlap:
            conflicts.append(f"  ⚠️  {fpath.name}: {sorted(overlap)}")
    return conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Generate {MODULE_NAME} feature module")
    parser.add_argument(
        "--modules-dir", type=Path, default=MODULES_DIR,
        help=f"Output directory (default: {MODULES_DIR})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    parser.add_argument(
        "--force", action="store_true",
        help="Write even if column name conflicts detected",
    )
    args = parser.parse_args()

    print(f"[{MODULE_NAME}] Loading sources (breakout_events + 1m OHLCV)...")
    sources = load_sources()

    print(f"[{MODULE_NAME}] Building features...")
    df = build_features(sources)

    # ── Validate ──────────────────────────────────────────────────
    n_feats = len([c for c in df.columns if c not in EVENT_KEY])

    print(f"[{MODULE_NAME}] Rows: {len(df):,}")
    print(f"[{MODULE_NAME}] Features: {n_feats}")
    print(f"[{MODULE_NAME}] Grain: ({', '.join(EVENT_KEY)})")
    print(f"[{MODULE_NAME}] Columns: {list(df.columns)}")

    # Check for NaN ratio
    for col in df.columns:
        if col in EVENT_KEY:
            continue
        nan_pct = df[col].isna().mean() * 100
        print(f"  {col}: NaN={nan_pct:.1f}%")
        if nan_pct > 50:
            print(f"  ⚠️  {col}: {nan_pct:.1f}% NaN — may degrade model")

    # Check column name conflicts
    conflicts = check_column_conflict(df)
    if conflicts:
        print(f"[{MODULE_NAME}] ⚠️  Column name conflicts detected:")
        for c in conflicts:
            print(c)
        if not args.force:
            print(f"[{MODULE_NAME}] Aborting. Use --force to write anyway, or rename columns.")
            return

    if args.dry_run:
        print(f"[{MODULE_NAME}] Dry run — not writing")
        return

    # ── Write ─────────────────────────────────────────────────────
    args.modules_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.modules_dir / f"{MODULE_NAME}_features.parquet"
    df.to_parquet(out_path, index=False)
    file_size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[{MODULE_NAME}] Written to {out_path} ({file_size_mb:.1f} MB)")
    print(f"[{MODULE_NAME}] ✅ Ready — just re-run sweep script (auto-detected)")


if __name__ == "__main__":
    main()

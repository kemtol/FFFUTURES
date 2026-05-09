#!/usr/bin/env python3
"""Generate pre-breakout volatility profile feature module.

Hypothesis
----------
The current model only sees *snapshot features* at breakout time (ATR, ADX, VWAP).
It doesn't know **how price arrived** at the breakout. A breakout after a
compression/low-volatility period (energy build-up) is statistically different
from a breakout already in a high-volatility environment (noise breakout).

By encoding pre-breakout price action (compression, drift, candle patterns),
we give the model signal to discriminate between "real" breakouts and false
ones — directly impacting 2026 pass rate.

Features
--------
1. ``pre_bo_compression_ratio``  — pre-breakout range / orb_range.
   Low ratio = price compressed before breakout = energy build-up.

2. ``pre_bo_drift_atr`` — net price drift (in ATR units) in the hour before
   breakout. High drift = momentum already established.

3. ``pre_bo_inside_bar_flag`` — 1 if the last 15m candle before breakout
   is an inside bar (lower high, higher low than previous). Volatility
   contraction signal.

4. ``pre_bo_last_candle_range_ratio`` — (high - low) of the last 15m candle
   before breakout, divided by atr14. Big candle = momentum.

5. ``pre_bo_bullish_ratio`` — fraction of bullish (close > open) candles in
   the pre-breakout 1-hour window. Sentiment drift.

Grain
-----
``(date, session, orb_tf, breakout_ts)`` — one row per breakout event (34,187 rows).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── module metadata ──────────────────────────────────────────────────────────

MODULE_NAME = "pre_breakout_profile"
MODULES_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data" / "Level_1_Features" / "modules"
)
L1_DIR = MODULES_DIR.parent  # data/Level_1_Features
RAW_DIR = L1_DIR.parent / "Level_0_Raw"

EVENT_KEY = ["date", "session", "orb_tf", "breakout_ts"]

# Number of 15m candles to look back before breakout_ts
# 4 candles = 1 hour of pre-breakout data
PRE_BO_CANDLES = 4


# ── sources ──────────────────────────────────────────────────────────────────


def load_sources() -> dict[str, pd.DataFrame]:
    """Load breakout events + 15m OHLCV from SQLite."""
    bo = pd.read_parquet(L1_DIR / "breakout_events.parquet")
    # Keep date as string (object) — other modules use this dtype for merge
    if bo["date"].dtype != "object":
        bo["date"] = bo["date"].astype(str)

    # Load 15m data
    db_path = RAW_DIR / "MGC_15m.db"
    conn = sqlite3.connect(str(db_path))
    df15m = pd.read_sql(
        "SELECT epoch_ms, timestamp_utc, open, high, low, close, volume "
        "FROM investing_ohlcv_15m ORDER BY epoch_ms",
        conn,
        parse_dates=["timestamp_utc"],
    )
    conn.close()

    return {"breakout_events": bo, "df_15m": df15m}


# ── feature builder ──────────────────────────────────────────────────────────


def build_features(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build pre-breakout profile features.

    For each breakout event, looks back ``PRE_BO_CANDLES`` 15m candles
    and computes compression, drift, inside-bar, and sentiment features.
    """
    bo = sources["breakout_events"].copy()
    df15m = sources["df_15m"].copy()

    # Ensure breakout_ts is tz-aware (it's UTC)
    bo = bo.sort_values("breakout_ts").reset_index(drop=True)

    # Convert df15m timestamp_utc to tz-aware for comparison
    if df15m["timestamp_utc"].dt.tz is None:
        df15m["timestamp_utc"] = df15m["timestamp_utc"].dt.tz_localize("UTC")
    df15m = df15m.sort_values("timestamp_utc").reset_index(drop=True)

    # Build a sorted numpy array of 15m timestamps for binary search
    ts_15m = df15m["timestamp_utc"].values.astype("datetime64[ns]")

    records: list[dict] = []

    for _, event in bo.iterrows():
        bo_ts = event["breakout_ts"]
        orb_range = event["orb_range"] if pd.notna(event["orb_range"]) else np.nan
        atr14 = event["atr14_at_entry"] if pd.notna(event["atr14_at_entry"]) else np.nan

        if pd.isna(bo_ts):
            records.append(_empty_record(event))
            continue

        bo_ts_ns = bo_ts.to_datetime64()

        # Find index of the last 15m candle AT or BEFORE breakout_ts
        # Using searchsorted: returns position where bo_ts would be inserted
        idx = np.searchsorted(ts_15m, bo_ts_ns, side="right") - 1

        if idx < PRE_BO_CANDLES:
            # Not enough pre-breakout data
            records.append(_empty_record(event))
            continue

        # Get the PRE_BO_CANDLES 15m candles before breakout
        pre_idx = idx - PRE_BO_CANDLES + 1  # inclusive start
        pre_candles = df15m.iloc[pre_idx : idx + 1].copy()

        if len(pre_candles) < PRE_BO_CANDLES:
            records.append(_empty_record(event))
            continue

        # ── Compute features ──────────────────────────────────────────

        # 1. Compression ratio: pre-bo range / orb_range
        pre_high = pre_candles["high"].max()
        pre_low = pre_candles["low"].min()
        pre_range = pre_high - pre_low
        compression_ratio = pre_range / orb_range if orb_range > 0 else np.nan

        # 2. Drift in ATR units: (|first_open - last_close|) / atr14
        first_open = pre_candles.iloc[0]["open"]
        last_close = pre_candles.iloc[-1]["close"]
        drift = abs(last_close - first_open)
        drift_atr = drift / atr14 if atr14 > 0 else np.nan

        # 3. Inside bar flag: last candle is inside previous
        last_candle = pre_candles.iloc[-1]
        prev_candle = pre_candles.iloc[-2]
        inside_bar = int(
            last_candle["high"] < prev_candle["high"]
            and last_candle["low"] > prev_candle["low"]
        )

        # 4. Last candle range / atr14
        last_candle_range = last_candle["high"] - last_candle["low"]
        last_candle_range_ratio = last_candle_range / atr14 if atr14 > 0 else np.nan

        # 5. Bullish ratio: fraction of candles with close > open
        bullish_count = int((pre_candles["close"] > pre_candles["open"]).sum())
        bullish_ratio = bullish_count / len(pre_candles)

        records.append({
            "date": event["date"],
            "session": event["session"],
            "orb_tf": event["orb_tf"],
            "breakout_ts": bo_ts,
            "pre_bo_compression_ratio": compression_ratio,
            "pre_bo_drift_atr": drift_atr,
            "pre_bo_inside_bar_flag": float(inside_bar),
            "pre_bo_last_candle_range_ratio": last_candle_range_ratio,
            "pre_bo_bullish_ratio": bullish_ratio,
        })

    result = pd.DataFrame(records)

    # ── Ensure float32 dtypes ─────────────────────────────────────
    feat_cols = [c for c in result.columns if c not in EVENT_KEY]
    for col in feat_cols:
        result[col] = result[col].astype("float32")

    return result


def _empty_record(event: pd.Series) -> dict:
    return {
        "date": event["date"],
        "session": event["session"],
        "orb_tf": event["orb_tf"],
        "breakout_ts": event["breakout_ts"],
        "pre_bo_compression_ratio": np.nan,
        "pre_bo_drift_atr": np.nan,
        "pre_bo_inside_bar_flag": np.nan,
        "pre_bo_last_candle_range_ratio": np.nan,
        "pre_bo_bullish_ratio": np.nan,
    }


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

    print(f"[{MODULE_NAME}] Loading sources...")
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

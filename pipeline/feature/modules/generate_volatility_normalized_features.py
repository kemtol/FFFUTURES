#!/usr/bin/env python3
"""Generate volatility-normalized feature module.

Features are percentile ranks and z-scores of raw features (ATR14, breakout_strength,
orb_range) computed within rolling windows per ``(session, orb_tf)`` group.

Rationale
---------
The 2026 OOD collapse is caused by feature distribution drift (volatility 4.6× higher).
Percentile ranks and z-scores are inherently stationary — they express a value relative
to its own recent history, regardless of absolute scale.

Grain
-----
``(date, session, orb_tf, breakout_ts)`` — one row per breakout event (34,187 rows).

Features
--------
1. ``atr14_percentile_20d``    — percentile rank of ATR14 within last 20 trading days
2. ``atr14_zscore_20d``        — z-score of ATR14 within last 20 trading days
3. ``breakout_strength_percentile_20d`` — percentile rank of breakout_strength (20d)
4. ``breakout_strength_zscore_10d``     — z-score of breakout_strength (10d, shorter window)
5. ``orb_range_percentile_20d``         — percentile rank of orb_range (20d)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── module metadata ──────────────────────────────────────────────────────────

MODULE_NAME = "volatility_normalized"
MODULES_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data" / "Level_1_Features" / "modules"
)
L1_DIR = MODULES_DIR.parent  # data/Level_1_Features

EVENT_KEY = ["date", "session", "orb_tf", "breakout_ts"]

# Rolling windows in calendar days (≈ 20 / 10 trading days)
WINDOW_20D = 28
WINDOW_10D = 14
MIN_SAMPLES = 5


# ── helpers ──────────────────────────────────────────────────────────────────


def _percentile_rank(value: float, history: list[float]) -> float:
    """Percentile rank of *value* within *history*, 0..1."""
    if len(history) < MIN_SAMPLES:
        return np.nan
    arr = np.array(history + [value])
    # Number of values <= value, minus one for the value itself
    count_le = int((arr <= value).sum())
    rank = count_le / len(arr)
    return float(rank)


def _z_score(value: float, history: list[float], cap: float = 10.0) -> float:
    """Z-score of *value* relative to *history*, clipped to [-*cap*, *cap*]."""
    if len(history) < MIN_SAMPLES:
        return np.nan
    mu = np.mean(history)
    std = np.std(history, ddof=0)
    if std == 0 or np.isnan(mu) or np.isnan(std):
        return np.nan
    z = (value - mu) / std
    return float(np.clip(z, -cap, cap))


# ── sources ──────────────────────────────────────────────────────────────────


def load_sources() -> dict[str, pd.DataFrame]:
    """Load breakout events (the only source needed for these features)."""
    bo = pd.read_parquet(L1_DIR / "breakout_events.parquet")
    return {"breakout_events": bo}


# ── feature builder ──────────────────────────────────────────────────────────


def build_features(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build volatility-normalized features.

    For each ``(session, orb_tf)`` group, iterates chronologically and computes
    rolling percentile ranks / z-scores against a sliding window of recent values.
    """
    bo = sources["breakout_events"].copy()
    bo = bo.sort_values(["date", "session", "orb_tf", "breakout_ts"]).reset_index(
        drop=True
    )

    records: list[dict] = []

    for (sess, otf), grp in bo.groupby(["session", "orb_tf"], sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)

        # Rolling histories per feature — maintained as FIFO queues
        dates: list[pd.Timestamp] = []
        atr14_vals: list[float] = []
        bs_vals: list[float] = []
        orb_vals: list[float] = []

        for _, row in grp.iterrows():
            current_date = row["date"]

            # Purge entries outside the 20-day window (28 calendar days)
            cutoff_20d = current_date - pd.Timedelta(days=WINDOW_20D)
            cutoff_10d = current_date - pd.Timedelta(days=WINDOW_10D)

            # Find the split point for purge
            # dates is already sorted, so we can pop from front while outdated
            while dates and dates[0] < cutoff_20d:
                dates.pop(0)
                atr14_vals.pop(0)
                bs_vals.pop(0)
                orb_vals.pop(0)

            # For 10d z-score we need a second window — purge 10d entries
            # Actually, we keep the 20d window for all and just use first 10d
            # portion for the 10d z-score. More efficient than two queues.
            # Find how many in the 10d window
            cut_10d_idx = 0
            for i, d in enumerate(dates):
                if d >= cutoff_10d:
                    cut_10d_idx = i
                    break
            else:
                cut_10d_idx = len(dates)

            atr14_val = row.get("atr14_at_entry")
            bs_val = row.get("breakout_strength")
            orb_val = row.get("orb_range")

            # ── Compute features ──────────────────────────────────────
            atr14_pct = (
                _percentile_rank(atr14_val, atr14_vals)
                if atr14_val is not None and not (isinstance(atr14_val, float) and np.isnan(atr14_val))
                else np.nan
            )
            atr14_z = (
                _z_score(atr14_val, atr14_vals)
                if atr14_val is not None and not (isinstance(atr14_val, float) and np.isnan(atr14_val))
                else np.nan
            )
            bs_pct = (
                _percentile_rank(bs_val, bs_vals)
                if bs_val is not None and not (isinstance(bs_val, float) and np.isnan(bs_val))
                else np.nan
            )
            # 10d z-score: use only values within 10d window
            bs_z_10d = (
                _z_score(bs_val, bs_vals[cut_10d_idx:])
                if bs_val is not None and not (isinstance(bs_val, float) and np.isnan(bs_val))
                else np.nan
            )
            orb_pct = (
                _percentile_rank(orb_val, orb_vals)
                if orb_val is not None and not (isinstance(orb_val, float) and np.isnan(orb_val))
                else np.nan
            )

            records.append(
                {
                    "date": current_date,
                    "session": sess,
                    "orb_tf": otf,
                    "breakout_ts": row["breakout_ts"],
                    "atr14_percentile_20d": atr14_pct,
                    "atr14_zscore_20d": atr14_z,
                    "breakout_strength_percentile_20d": bs_pct,
                    "breakout_strength_zscore_10d": bs_z_10d,
                    "orb_range_percentile_20d": orb_pct,
                }
            )

            # ── Append to history ─────────────────────────────────────
            if (
                atr14_val is not None
                and not (isinstance(atr14_val, float) and np.isnan(atr14_val))
            ):
                atr14_vals.append(atr14_val)
            else:
                atr14_vals.append(np.nan)

            if (
                bs_val is not None
                and not (isinstance(bs_val, float) and np.isnan(bs_val))
            ):
                bs_vals.append(bs_val)
            else:
                bs_vals.append(np.nan)

            if (
                orb_val is not None
                and not (isinstance(orb_val, float) and np.isnan(orb_val))
            ):
                orb_vals.append(orb_val)
            else:
                orb_vals.append(np.nan)

            dates.append(current_date)

    result = pd.DataFrame(records)
    return result


# ── conflict check ────────────────────────────────────────────────────────────


def check_column_conflict(df: pd.DataFrame) -> list[str]:
    """Check if any new feature columns already exist in existing modules."""
    feat_cols = set(c for c in df.columns if c not in EVENT_KEY)
    conflicts: list[str] = []
    for fpath in sorted(MODULES_DIR.glob("*_features.parquet")):
        existing = set(pd.read_parquet(fpath).columns)
        existing_feats = existing - set(EVENT_KEY)
        overlap = feat_cols & existing_feats
        if overlap:
            conflicts.append(f"  ⚠️  {fpath.name}: {sorted(overlap)}")
    return conflicts


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Generate {MODULE_NAME} feature module"
    )
    parser.add_argument(
        "--modules-dir",
        type=Path,
        default=MODULES_DIR,
        help=f"Output directory (default: {MODULES_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load sources and show feature shapes without writing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file if present",
    )
    args = parser.parse_args()

    out_dir = args.modules_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{MODULE_NAME}_features.parquet"

    # ── Check existing ────────────────────────────────────────────────
    if out_path.exists() and not args.force and not args.dry_run:
        print(
            f"[{MODULE_NAME}] {out_path.name} exists. Use --force to overwrite."
        )
        sys.exit(0)

    # ── Load ──────────────────────────────────────────────────────────
    print(f"[{MODULE_NAME}] Loading sources...")
    sources = load_sources()
    for name, df in sources.items():
        print(f"  {name}: {len(df):,} rows, {list(df.columns)}")

    # ── Build ─────────────────────────────────────────────────────────
    print(f"[{MODULE_NAME}] Building features...")
    features = build_features(sources)
    n_feat = len([c for c in features.columns if c not in EVENT_KEY])
    print(f"  Generated {len(features):,} rows × {n_feat} features")

    # ── Conflict check ────────────────────────────────────────────────
    conflicts = check_column_conflict(features)
    if conflicts:
        print(f"[{MODULE_NAME}] ⚠️  Column conflicts detected:")
        for c in conflicts:
            print(c)
        print("  Rename features or remove conflicting modules before proceeding.")
        if not args.force:
            print("  Use --force to override (existing columns will get _y suffix).")
            sys.exit(1)

    # ── Write ─────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"[{MODULE_NAME}] Dry-run — not writing to {out_path}")
    else:
        features.to_parquet(out_path, index=False)
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"[{MODULE_NAME}] ✅ Written to {out_path} ({size_mb:.1f} MB)")

    print(f"[{MODULE_NAME}] Done.")


if __name__ == "__main__":
    main()

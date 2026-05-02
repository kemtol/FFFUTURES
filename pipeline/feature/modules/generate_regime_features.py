#!/usr/bin/env python3
"""Generate regime-detection feature module.

Hypothesis
----------
The current model sees features as absolute values within their own distribution
but doesn't explicitly know *which regime* the market is operating in. Regime
features encode the market state at breakout time using only prior data:

- **Volatility regime**: expanding vs contracting (60d context, broader than the
  20d window in volatility_normalized)
- **Trend regime**: strengthening vs weakening ADX trend
- **Session consistency**: is the current session direction persistent or whipsawing?
- **ORB stability**: are ORB ranges stable or chaotic recently?

These features give the model explicit regime-awareness so it can learn to
switch behavior (e.g. fade breakouts in whipsaw regimes, follow in trending regimes)
without requiring deep tree splits to discover regime patterns from raw features.

Features (7)
------------
1. ``reg_vol_expanding_ratio`` — 20d MA of ATR / 60d MA of ATR.
   > 1 = volatility expanding, < 1 = contracting.

2. ``reg_vol_percentile_60d`` — percentile rank of current ATR14 within
   last 60 trading days (broader context than the 20d in vol_norm).

3. ``reg_vol_spike_flag`` — binary: is current ATR > 1.5× 20-day median ATR?
   Spike/explosive vol environment.

4. ``reg_adx_regime_percentile_20d`` — percentile rank of ADX within last
   20 trading days. High ADX relative to recent history = trending regime.

5. ``reg_adx_slope_20d`` — linear slope of ADX over last 20 events
   (same session, same ORB TF). Positive = trend strengthening.

6. ``reg_session_persistence_rate_10d`` — fraction of last 10 same-session
   events where breakout_side matched the prior event's side.
   High = directional persistence. Low = whipsaw/fade tendency.

7. ``reg_orb_range_cv_20d`` — coefficient of variation (std/|mean|) of
   orb_range over last 20 events. Low = consistent ORBs. High = unstable.

Grain
-----
``(date, session, orb_tf, breakout_ts)`` — one row per breakout event (34,187 rows).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── module metadata ──────────────────────────────────────────────────────────

MODULE_NAME = "regime"
MODULES_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data" / "Level_1_Features" / "modules"
)
L1_DIR = MODULES_DIR.parent  # data/Level_1_Features

EVENT_KEY = ["date", "session", "orb_tf", "breakout_ts"]

FEATURE_COLS = [
    "reg_vol_expanding_ratio",
    "reg_vol_percentile_60d",
    "reg_vol_spike_flag",
    "reg_adx_regime_percentile_20d",
    "reg_adx_slope_20d",
    "reg_session_persistence_rate_10d",
    "reg_orb_range_cv_20d",
]

# Rolling-window sizes in calendar days
WINDOW_20D = 28   # ≈20 trading days
WINDOW_60D = 84   # ≈60 trading days
MIN_SAMPLES = 5
EPS = 1e-8


# ── helpers ──────────────────────────────────────────────────────────────────


def _percentile_rank(value: float, history: np.ndarray) -> float:
    """Percentile rank of *value* within *history*, 0..1.

    Clears NaN from history before ranking.
    """
    clean = history[np.isfinite(history)]
    if len(clean) < MIN_SAMPLES:
        return np.nan
    rank = (clean <= value).sum() / len(clean)
    return float(rank)


def _median(values: np.ndarray) -> float:
    """Median of finite values, or NaN."""
    clean = values[np.isfinite(values)]
    if len(clean) < MIN_SAMPLES:
        return np.nan
    return float(np.median(clean))


def _mean(values: np.ndarray) -> float:
    """Mean of finite values, or NaN."""
    clean = values[np.isfinite(values)]
    if len(clean) < MIN_SAMPLES:
        return np.nan
    return float(np.mean(clean))


def _linear_slope(y: np.ndarray) -> float:
    """Least-squares slope of y against its index, or NaN."""
    valid = np.isfinite(y)
    if valid.sum() < MIN_SAMPLES:
        return np.nan
    x = np.arange(len(y), dtype=float)
    y_clean = y.copy()
    # Use only valid indices
    mask = valid
    if mask.sum() < MIN_SAMPLES:
        return np.nan
    # With missing values just compute on contiguous valid
    # For simplicity, use only the last N valid values
    y_v = y[mask]
    x_v = x[mask]
    if len(y_v) < MIN_SAMPLES:
        return np.nan
    # Cov(x, y) / Var(x)
    mx, my = x_v.mean(), y_v.mean()
    cov = ((x_v - mx) * (y_v - my)).sum()
    var = ((x_v - mx) ** 2).sum()
    if var < EPS:
        return 0.0
    return float(cov / var)


# ── sources ──────────────────────────────────────────────────────────────────


def load_sources() -> dict[str, pd.DataFrame]:
    """Load breakout_events (for atr14, side, orb_range) + market_context (for adx)."""
    bo = pd.read_parquet(L1_DIR / "breakout_events.parquet")
    mc = pd.read_parquet(L1_DIR / "market_context.parquet")

    # Ensure date is datetime.date (critical for merge compatibility)
    bo["date"] = pd.to_datetime(bo["date"]).dt.date
    mc["date"] = pd.to_datetime(mc["date"]).dt.date

    return {"breakout_events": bo, "market_context": mc}


# ── feature builder ──────────────────────────────────────────────────────────


def build_features(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build regime-detection features by iterating chronologically per (session, orb_tf).

    For each breakout event, computes regime statistics from a rolling window of
    PRIOR events only (no lookahead). Uses both 20d (short regime) and 60d (long
    regime) rolling windows.
    """
    bo = sources["breakout_events"]
    mc = sources["market_context"]

    # Merge breakout_events with market_context for ADX
    merged = bo.merge(mc, on=EVENT_KEY, how="left")
    merged["date"] = pd.to_datetime(merged["date"]).dt.date
    merged = merged.sort_values(["date", "session", "orb_tf", "breakout_ts"]).reset_index(drop=True)

    records: list[dict] = []

    for (sess, otf), grp in merged.groupby(["session", "orb_tf"], sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)

        # Rolling histories per column — FIFO queues
        dates: list = []
        atr14_vals: list[float] = []
        adx_vals: list[float] = []
        side_vals: list[int] = []
        orb_range_vals: list[float] = []

        for _, row in grp.iterrows():
            current_date = row["date"]
            cutoff_20d = current_date - pd.Timedelta(days=WINDOW_20D)
            cutoff_60d = current_date - pd.Timedelta(days=WINDOW_60D)

            # Purge expired entries from queues
            while dates and dates[0] < cutoff_60d:
                dates.pop(0)
                atr14_vals.pop(0)
                adx_vals.pop(0)
                side_vals.pop(0)
                orb_range_vals.pop(0)

            # Extract current values
            atr_val = row.get("atr14_at_entry", np.nan)
            adx_val = row.get("adx_14_15m", np.nan)
            side_val = row.get("breakout_side", np.nan)
            orb_val = row.get("orb_range", np.nan)

            # Convert side to 1 (bull) / -1 (bear) / NaN
            side_int = np.nan
            if isinstance(side_val, str):
                side_int = 1 if side_val.lower() == "bull" else -1
            elif not pd.isna(side_val):
                side_int = int(side_val)

            # ── Separate queues for 20d and 60d windows ──────────────
            # 20d window: entries where date >= cutoff_20d
            idx_20d = next((i for i, d in enumerate(dates) if d >= cutoff_20d), len(dates))

            atr_20d = np.array(atr14_vals[idx_20d:], dtype=float)
            atr_60d = np.array(atr14_vals, dtype=float)
            adx_20d = np.array(adx_vals[idx_20d:], dtype=float)
            adx_all = np.array(adx_vals, dtype=float)
            side_all = np.array(side_vals, dtype=float)
            orb_all = np.array(orb_range_vals, dtype=float)

            # ── Feature 1: Vol expanding ratio (20d MA / 60d MA) ─────
            ma20 = _mean(atr_20d)
            ma60 = _mean(atr_60d)
            reg_vol_expanding = (ma20 / (ma60 + EPS)) if (ma20 is not None and ma60 is not None and not (np.isnan(ma20) or np.isnan(ma60))) else np.nan
            # Ensure finite
            if isinstance(reg_vol_expanding, float) and not np.isfinite(reg_vol_expanding):
                reg_vol_expanding = np.nan

            # ── Feature 2: Vol percentile 60d ────────────────────────
            reg_vol_pct60 = (
                _percentile_rank(atr_val, atr_60d)
                if not (isinstance(atr_val, float) and np.isnan(atr_val))
                else np.nan
            )

            # ── Feature 3: Vol spike flag ────────────────────────────
            med_20d = _median(atr_20d)
            vol_spike = 0
            if (
                not (isinstance(atr_val, float) and np.isnan(atr_val))
                and med_20d is not None
                and not np.isnan(med_20d)
                and med_20d > 0
            ):
                vol_spike = 1 if atr_val > 1.5 * med_20d else 0

            # ── Feature 4: ADX regimen percentile 20d ─────────────────
            reg_adx_pct = (
                _percentile_rank(adx_val, adx_20d)
                if not (isinstance(adx_val, float) and np.isnan(adx_val))
                else np.nan
            )

            # ── Feature 5: ADX slope 20d ─────────────────────────────
            reg_adx_slope = _linear_slope(adx_20d)

            # ── Feature 6: Session persistence rate 10d ───────────────
            # Fraction of neighboring prior events with same side
            last_10_sides = side_all[-10:] if len(side_all) >= 10 else side_all
            clean_sides = last_10_sides[np.isfinite(last_10_sides)]
            persistence = np.nan
            if len(clean_sides) >= 2:
                matches = (clean_sides[:-1] == clean_sides[1:]).sum()
                persistence = float(matches) / (len(clean_sides) - 1)
            elif len(clean_sides) == 1:
                persistence = 0.5  # neutral with only one prior

            # ── Feature 7: ORB range CV 20d ──────────────────────────
            orb_clean = orb_all[np.isfinite(orb_all)]
            reg_orb_cv = np.nan
            if len(orb_clean) >= MIN_SAMPLES:
                mu = np.mean(orb_clean)
                sd = np.std(orb_clean, ddof=0)
                if mu > EPS:
                    reg_orb_cv = float(sd / mu)

            # ── Record ───────────────────────────────────────────────
            records.append(
                {
                    "date": current_date,
                    "session": sess,
                    "orb_tf": otf,
                    "breakout_ts": row["breakout_ts"],
                    "reg_vol_expanding_ratio": reg_vol_expanding,
                    "reg_vol_percentile_60d": reg_vol_pct60,
                    "reg_vol_spike_flag": float(vol_spike),
                    "reg_adx_regime_percentile_20d": reg_adx_pct,
                    "reg_adx_slope_20d": reg_adx_slope,
                    "reg_session_persistence_rate_10d": persistence,
                    "reg_orb_range_cv_20d": reg_orb_cv,
                }
            )

            # ── Append to rolling queues ─────────────────────────────
            atr_append = (
                atr_val
                if not (isinstance(atr_val, float) and np.isnan(atr_val))
                else np.nan
            )
            adx_append = (
                adx_val
                if not (isinstance(adx_val, float) and np.isnan(adx_val))
                else np.nan
            )
            side_append = side_int
            orb_append = (
                orb_val
                if not (isinstance(orb_val, float) and np.isnan(orb_val))
                else np.nan
            )

            atr14_vals.append(atr_append)
            adx_vals.append(adx_append)
            side_vals.append(side_append)
            orb_range_vals.append(orb_append)
            dates.append(current_date)

    result = pd.DataFrame(records)

    # ── Ensure float32 dtypes ─────────────────────────────────────────
    for col in FEATURE_COLS:
        result[col] = result[col].astype("float32")

    return result


# ── conflict check ────────────────────────────────────────────────────────────


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

    # Check existing
    if out_path.exists() and not args.force and not args.dry_run:
        print(
            f"[{MODULE_NAME}] {out_path.name} exists. Use --force to overwrite."
        )
        sys.exit(0)

    # Load
    print(f"[{MODULE_NAME}] Loading sources...")
    sources = load_sources()
    for name, df in sources.items():
        print(f"  {name}: {len(df):,} rows, {list(df.columns[:10])}...")

    # Build
    print(f"[{MODULE_NAME}] Building features...")
    features = build_features(sources)
    n_feat = len([c for c in features.columns if c not in EVENT_KEY])
    print(f"  Generated {len(features):,} rows x {n_feat} features")

    # Validate
    print(f"[{MODULE_NAME}] Grain: ({', '.join(EVENT_KEY)})")
    print(f"[{MODULE_NAME}] Columns: {list(features.columns)}")

    for col in features.columns:
        if col in EVENT_KEY:
            continue
        nan_pct = features[col].isna().mean() * 100
        print(f"  {col}: NaN={nan_pct:.1f}%")
        if nan_pct > 50:
            print(f"  ⚠️  {col}: {nan_pct:.1f}% NaN — may degrade model")
        elif nan_pct > 10:
            print(f"  ⚠️  {col}: {nan_pct:.1f}% NaN — moderate")

    # Conflict check
    conflicts = check_column_conflict(features)
    if conflicts:
        print(f"[{MODULE_NAME}] ⚠️  Column conflicts detected:")
        for c in conflicts:
            print(c)
        print("  Rename features or remove conflicting modules before proceeding.")
        if not args.force:
            print("  Use --force to override (existing columns will get _y suffix).")
            sys.exit(1)

    # Write
    if args.dry_run:
        print(f"[{MODULE_NAME}] Dry-run — not writing to {out_path}")
    else:
        features.to_parquet(out_path, index=False)
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"[{MODULE_NAME}] ✅ Written to {out_path} ({size_mb:.1f} MB)")

    print(f"[{MODULE_NAME}] Done.")


if __name__ == "__main__":
    main()

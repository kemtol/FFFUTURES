#!/usr/bin/env python3
"""Generate macro market feature module.

Hypothesis
----------
ORB breakout outcomes on MGC (Micro Gold-Copper) are systematically affected by
the macro regime — equity trend, dollar strength, rates, and commodities. The
model currently has NO macro context, seeing each breakout in isolation.

By adding daily macro state, the model can learn regime-conditional behaviour:
- Risk-on (SPX bull, weak dollar): breakouts more likely to follow through.
- Risk-off (SPX bear, strong dollar, rising yields): breakouts more likely to fail.
- Oil volatility spikes → macro uncertainty → choppy price action.

Features
--------
1. ``mac_spx_regime`` — SPY above 200-day SMA → 1.0 (bull), else 0.0 (bear).
   Flags whether equities are in a secular bull or bear trend.
   (float32, 0=Bear, 1=Bull)

2. ``mac_dxy_trend`` — DXY vs 50-day SMA: 1.0 (strong dollar, above SMA),
   -1.0 (weak dollar, below SMA), 0.0 (within 0.5% of SMA — neutral).
   Dollar strength inversely correlates with gold.
   (float32, -1/0/1)

3. ``mac_us10y_change`` — Daily absolute change in 10-Year Treasury Yield.
   Sharp yield moves signal tightening/loosening financial conditions.
   (float32, absolute difference in % points)

4. ``mac_oil_volatility`` — Absolute daily return of Crude Oil futures.
   Oil volatility is a proxy for macro uncertainty / supply shocks.
   (float32, |daily return|)

Grain
-----
Merged onto ``(date, session, orb_tf, breakout_ts)`` via LEFT JOIN on ``date``.
Macro data is daily — forward-filled for weekends/holidays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── module metadata ──────────────────────────────────────────────────────────

MODULE_NAME = "macro"
MODULES_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data" / "Level_1_Features" / "modules"
)
L1_DIR = MODULES_DIR.parent  # data/Level_1_Features

EVENT_KEY = ["date", "session", "orb_tf", "breakout_ts"]

# Feature column names
FEATURE_COLS = [
    "mac_spx_regime",
    "mac_dxy_trend",
    "mac_us10y_change",
    "mac_oil_volatility",
]


# ── sources ──────────────────────────────────────────────────────────────────


def load_sources() -> dict[str, pd.DataFrame]:
    """Load breakout events + pre-fetched macro data (both in L1).

    Macro data is fetched ONCE by ``pipeline/fetch/fetch_macro_data.py``
    and saved as ``macro_data.parquet``. It includes raw close prices
    plus derived columns (MA, returns, changes).
    """
    bo = pd.read_parquet(L1_DIR / "breakout_events.parquet")
    macro = pd.read_parquet(L1_DIR / "macro_data.parquet")

    # CRITICAL: Use datetime.date objects for merge compatibility.
    # All other modules store date as datetime.date (not str).
    # A str "2010-10-04" != datetime.date(2010, 10, 4) in merge.
    import datetime as dtmod
    macro["date"] = pd.to_datetime(macro["date"]).dt.date

    return {"breakout_events": bo, "macro_data": macro}


# ── feature builder ──────────────────────────────────────────────────────────


def build_features(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build macro regime features from breakout_events × macro_data.

    Steps:
    1. LEFT JOIN breakout_events → macro_data on ``date``.
    2. Forward-fill macro columns for weekends/holidays (bfill for leading NaN).
    3. Compute 4 regime features from raw macro columns.
    """
    bo = sources["breakout_events"]
    macro = sources["macro_data"]

    # ── Select only needed macro columns ─────────────────────────────
    macro_cols = [
        "date",
        "spy_close",
        "spy_ma200",
        "dxy_close",
        "dxy_ma50",
        "us10y_change",
        "oil_return",
    ]
    macro_sub = macro[macro_cols].copy()

    # ── LEFT JOIN breakout → macro on date ───────────────────────────
    merged = bo[EVENT_KEY].merge(macro_sub, on="date", how="left")

    n_before = len(merged)
    # Forward-fill macro columns (most weekends will have NaN after merge)
    macro_feat_cols = [c for c in macro_cols if c != "date"]
    merged[macro_feat_cols] = merged[macro_feat_cols].ffill().bfill()
    n_ffilled = merged[macro_feat_cols].isna().any(axis=1).sum()
    if n_ffilled > 0:
        print(f"  [INFO] {n_ffilled}/{n_before} rows had NaN after forward-fill")

    # ── Feature 1: SPX Regime (Bull/Bear) ────────────────────────────
    # SPY above 200-day SMA → bull regime → 1.0, else 0.0
    spy_close = merged["spy_close"].values
    spy_ma200 = merged["spy_ma200"].values
    merged["mac_spx_regime"] = np.where(
        (np.isfinite(spy_close)) & (np.isfinite(spy_ma200)),
        np.where(spy_close > spy_ma200, 1.0, 0.0),
        np.nan,
    )

    # ── Feature 2: DXY Trend (Strong/Neutral/Weak Dollar) ────────────
    # DXY > 50d MA by >0.5% → 1.0 (strong dollar)
    # DXY < 50d MA by >0.5% → -1.0 (weak dollar)
    # Within 0.5% → 0.0 (neutral)
    dxy_close = merged["dxy_close"].values
    dxy_ma50 = merged["dxy_ma50"].values
    dxy_valid = np.isfinite(dxy_close) & np.isfinite(dxy_ma50)

    merged["mac_dxy_trend"] = np.where(
        dxy_valid,
        np.where(
            dxy_close > dxy_ma50 * 1.005,  # >0.5% above MA → strong
            1.0,
            np.where(
                dxy_close < dxy_ma50 * 0.995,  # >0.5% below MA → weak
                -1.0,
                0.0,  # within 0.5% → neutral
            ),
        ),
        np.nan,
    )

    # ── Feature 3: US10Y Daily Change (absolute) ─────────────────────
    # Absolute change in 10Y yield — large moves signal macro shocks
    us10y_chg = merged["us10y_change"].values
    merged["mac_us10y_change"] = np.where(
        np.isfinite(us10y_chg),
        np.abs(us10y_chg),
        np.nan,
    )

    # ── Feature 4: Oil Volatility (|daily return|) ────────────────────
    # Absolute oil return = proxy for macro uncertainty
    oil_ret = merged["oil_return"].values
    merged["mac_oil_volatility"] = np.where(
        np.isfinite(oil_ret),
        np.abs(oil_ret),
        np.nan,
    )

    # ── Select output columns ───────────────────────────────────────
    result = merged[EVENT_KEY + FEATURE_COLS].copy()

    # ── Ensure float32 dtypes ───────────────────────────────────────
    for col in FEATURE_COLS:
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

    print(f"[{MODULE_NAME}] Loading sources (breakout_events + macro_data)...")
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

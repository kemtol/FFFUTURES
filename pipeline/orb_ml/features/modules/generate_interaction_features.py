#!/usr/bin/env python3
"""Generate interaction feature module.

Hypothesis
----------
The current model sees each feature independently (ATR, ADX, breakout_strength,
orb_range, VWAP distance) but doesn't capture **interaction effects** between them.
Breakout conviction depends on combinations:

- High ATR + High ADX → explosive trending moves (follow-through)
- Strong breakout + Wide ORB → high-conviction directional bet
- Far from VWAP + High volatility → momentum continuation
- High ADX + Wide ORB/ATR → strong trend in a wide range = directional
- Strong breakout + late session → exhaustion vs continuation signal

By encoding pairwise interactions, we give the model non-linear relationships
without requiring the tree to discover them through deep splits — improving
sample efficiency and generalization to 2026.

Features
--------
1. ``int_atr14_x_adx`` — atr14_at_entry × adx_14_15m.
   Interaction between volatility (ATR) and trend strength (ADX).
   High values → explosive trending conditions → follow-through.

2. ``int_breakout_strength_x_range`` — breakout_strength × orb_range.
   Strong breakout on a wide ORB = high-conviction directional move.
   Weak breakout on a wide ORB = noise.

3. ``int_vwap_distance_x_atr14`` — |price_vs_vwap_pct| × atr14_at_entry.
   Price far from VWAP (momentum) combined with high volatility.
   Captures continuation vs mean-reversion scenarios.

4. ``int_adx_x_orb_range`` — adx_14_15m × orb_range_atr_ratio.
   Strong trend (ADX) + wide ORB relative to ATR = directional day.
   Captures regime-driven breakout quality.

5. ``int_breakout_strength_x_session`` — breakout_strength × time_in_session_min.
   Late breakout with strong price move = exhaustion (fade).
   Early breakout with strong price move = continuation (follow).

Grain
-----
``(date, session, orb_tf, breakout_ts)`` — one row per breakout event (34,187 rows).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── module metadata ──────────────────────────────────────────────────────────

MODULE_NAME = "interaction"
MODULES_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data" / "Level_1_Features" / "modules"
)
L1_DIR = MODULES_DIR.parent  # data/Level_1_Features

EVENT_KEY = ["date", "session", "orb_tf", "breakout_ts"]

# Feature column names
FEATURE_COLS = [
    "int_atr14_x_adx",
    "int_breakout_strength_x_range",
    "int_vwap_distance_x_atr14",
    "int_adx_x_orb_range",
    "int_breakout_strength_x_session",
]


# ── sources ──────────────────────────────────────────────────────────────────


def load_sources() -> dict[str, pd.DataFrame]:
    """Load breakout events + market context (both already in L1).

    No new data sources needed — all interaction fields are already computed.
    """
    bo = pd.read_parquet(L1_DIR / "breakout_events.parquet")
    mc = pd.read_parquet(L1_DIR / "market_context.parquet")

    # Keep date as string (object) — other modules use this dtype for merge
    if bo["date"].dtype != "object":
        bo["date"] = bo["date"].astype(str)
    if mc["date"].dtype != "object":
        mc["date"] = mc["date"].astype(str)

    return {"breakout_events": bo, "market_context": mc}


# ── feature builder ──────────────────────────────────────────────────────────


def build_features(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build interaction features from breakout_events × market_context.

    All 5 features are pairwise multiplications of existing fields.
    No new data sources, no SQLite, no candle iteration needed.
    """
    bo = sources["breakout_events"]
    mc = sources["market_context"]

    # ── Select only needed columns for merge ────────────────────────
    bo_cols = EVENT_KEY + ["breakout_strength", "orb_range", "atr14_at_entry"]
    mc_cols = EVENT_KEY + [
        "orb_range_atr_ratio",
        "time_in_session_min",
        "price_vs_vwap_pct",
        "adx_14_15m",
    ]

    bo_sub = bo[bo_cols].copy()
    mc_sub = mc[mc_cols].copy()

    # Ensure breakout_ts is comparable for merge
    # Both use tz-aware UTC timestamps — should match directly
    merged = bo_sub.merge(mc_sub, on=EVENT_KEY, how="left")

    n_before = len(merged)
    n_after = merged.dropna(subset=mc_cols[1:], how="all").shape[0]
    if n_after < n_before:
        print(f"  [INFO] {n_before - n_after} events dropped during merge (no market context)")

    # ── Feature 1: ATR × ADX ────────────────────────────────────────
    # High vol + strong trend = explosive follow-through
    atr14 = merged["atr14_at_entry"].values
    adx = merged["adx_14_15m"].values
    merged["int_atr14_x_adx"] = np.where(
        (np.isfinite(atr14)) & (np.isfinite(adx)),
        atr14 * adx,
        np.nan,
    )

    # ── Feature 2: Breakout strength × ORB range ────────────────────
    # Strong breakout on wide range = conviction
    strength = merged["breakout_strength"].values
    orb_range = merged["orb_range"].values
    merged["int_breakout_strength_x_range"] = np.where(
        (np.isfinite(strength)) & (np.isfinite(orb_range)),
        strength * orb_range,
        np.nan,
    )

    # ── Feature 3: |VWAP distance| × ATR ────────────────────────────
    # Far from VWAP (momentum) + high vol = continuation
    vwap_dist = merged["price_vs_vwap_pct"].values
    merged["int_vwap_distance_x_atr14"] = np.where(
        (np.isfinite(vwap_dist)) & (np.isfinite(atr14)),
        np.abs(vwap_dist) * atr14,
        np.nan,
    )

    # ── Feature 4: ADX × ORB range/ATR ratio ────────────────────────
    # Strong trend + wide range relative to vol = directional
    orb_range_atr = merged["orb_range_atr_ratio"].values
    merged["int_adx_x_orb_range"] = np.where(
        (np.isfinite(adx)) & (np.isfinite(orb_range_atr)),
        adx * orb_range_atr,
        np.nan,
    )

    # ── Feature 5: Breakout strength × time in session ──────────────
    # Late strong breakout = exhaustion; early strong breakout = continuation
    time_in_session = merged["time_in_session_min"].values
    merged["int_breakout_strength_x_session"] = np.where(
        (np.isfinite(strength)) & (np.isfinite(time_in_session)),
        strength * time_in_session,
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

    print(f"[{MODULE_NAME}] Loading sources (breakout_events + market_context)...")
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

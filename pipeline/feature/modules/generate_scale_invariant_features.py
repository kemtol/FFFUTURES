#!/usr/bin/env python3
"""
Generate ``scale_invariant_features`` module parquet.

Extracts scale-invariant feature computations from the sweep scripts
(``add_scale_invariant_features()`` in v3/v4/v5) into a standalone module
generator.

These features normalize price-based metrics to make them comparable across
different volatility regimes.

Output: ``data/Level_1_Features/modules/scale_invariant_features.parquet``

Features
--------
- ``breakout_strength_atr_ratio`` — breakout_strength / atr14_at_entry
- ``atr14_sq`` — atr14_at_entry ** 2
- ``breakout_strength_sq`` — breakout_strength ** 2
- ``price_vs_vwap_pct_abs`` — abs(price_vs_vwap_pct)
- ``orb_range_sq`` — orb_range ** 2
- ``adx_50_flag`` — 1 if ADX(14,15m) > 50 else 0
- ``breakout_strength_vs_orb`` — breakout_strength / orb_range

Source columns
--------------
- ``breakout_strength``, ``atr14_at_entry``, ``orb_range`` — from ``breakout_events.parquet``
- ``price_vs_vwap_pct``, ``adx_14_15m`` — from ``market_context.parquet``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_NAME = "scale_invariant"

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # futures/
DATA_DIR = ROOT / "data"
FEATURES_DIR = DATA_DIR / "Level_1_Features"
MODULES_DIR = FEATURES_DIR / "modules"
BO_PATH = FEATURES_DIR / "breakout_events.parquet"
MC_PATH = FEATURES_DIR / "market_context.parquet"

EVENT_KEY = ["date", "session", "orb_tf", "breakout_ts"]

EPS = 1e-8


# ── source loader ─────────────────────────────────────────────────────────────


def load_sources() -> dict[str, pd.DataFrame]:
    """Load breakout events + market context (for vwap/adx columns)."""
    bo = pd.read_parquet(BO_PATH)
    mc = pd.read_parquet(MC_PATH)
    print(f"  Breakout events: {len(bo):,}")
    print(f"  Market context:  {len(mc):,}")

    return {
        "breakout_events": bo,
        "market_context": mc,
    }


# ── feature builder ───────────────────────────────────────────────────────────


def build_features(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute scale-invariant features at breakout-event grain.

    Merges breakout_strength / atr14 / orb_range from breakout_events
    with price_vs_vwap_pct / adx_14_15m from market_context, then computes
    derived features.

    Returns DataFrame with EVENT_KEY columns + 7 scale-invariant features (float32).
    """
    bo = sources["breakout_events"][
        EVENT_KEY + ["breakout_strength", "atr14_at_entry", "orb_range"]
    ].drop_duplicates()

    mc = sources["market_context"][
        EVENT_KEY + ["price_vs_vwap_pct", "adx_14_15m"]
    ].drop_duplicates()

    # Merge onto breakout-event grain
    merged = bo.merge(mc, on=EVENT_KEY, how="left")

    result = merged[EVENT_KEY].copy()
    result["breakout_strength_atr_ratio"] = (
        merged["breakout_strength"] / (merged["atr14_at_entry"] + EPS)
    )
    result["atr14_sq"] = merged["atr14_at_entry"] ** 2
    result["breakout_strength_sq"] = merged["breakout_strength"] ** 2
    result["orb_range_sq"] = merged["orb_range"] ** 2
    result["price_vs_vwap_pct_abs"] = merged["price_vs_vwap_pct"].abs()
    result["adx_50_flag"] = (merged["adx_14_15m"] > 50).astype(int)
    result["breakout_strength_vs_orb"] = (
        merged["breakout_strength"] / (merged["orb_range"] + EPS)
    )

    # ── Ensure float32 dtypes ─────────────────────────────────────
    feat_cols = [c for c in result.columns if c not in EVENT_KEY]
    for col in feat_cols:
        result[col] = result[col].astype("float32")

    return result


# ── conflict check ────────────────────────────────────────────────────────────


def check_column_conflict(df: pd.DataFrame) -> list[str]:
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
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    parser.add_argument(
        "--force",
        action="store_true",
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

    for col in df.columns:
        if col in EVENT_KEY:
            continue
        nan_pct = df[col].isna().mean() * 100
        if nan_pct > 50:
            print(f"  ⚠️  {col}: {nan_pct:.1f}% NaN — may degrade model")

    conflicts = check_column_conflict(df)
    if conflicts:
        print(f"[{MODULE_NAME}] ⚠️  Column name conflicts detected:")
        for c in conflicts:
            print(c)
        if not args.force:
            print(
                f"[{MODULE_NAME}] Aborting. Use --force to write anyway, "
                f"or rename columns."
            )
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

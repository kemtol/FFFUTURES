#!/usr/bin/env python3
"""
TEMPLATE: Generate a new feature module parquet for the ORB futures project.

WORKFLOW
────────
1. Copy this file  →  generate_{your_family}_features.py
2. Change ``MODULE_NAME`` and ``load_sources()`` to match your feature family
3. Implement ``build_features()`` with your feature logic
4. Run::

       python pipeline/feature/modules/generate_{your_family}_features.py

5. Output → ``data/Level_1_Features/modules/{your_family}_features.parquet``
6. Re-run sweep — module auto-detected via ``load_features_from_modules()``

REQUIREMENTS
────────────
- Output columns: ``EVENT_KEY`` (date, session, orb_tf, breakout_ts) + numeric features (float32)
- Grain: **(date, session, orb_tf, breakout_ts)** — breakout-event level.
  Features are identical for both rev and cont rows (merged later).
- No lookahead: only use data available **before** the ``breakout_ts``.
- Feature column names must not conflict with existing modules
  (check with ``--dry-run``).

EXAMPLES
────────
     df = pd.DataFrame({
         "date": [...],  "session": [...],  "orb_tf": [...],  "breakout_ts": [...],
         "my_feature": [...],
     })
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# CONFIG  —  Change these for your new feature family
# ═══════════════════════════════════════════════════════════════════════
MODULE_NAME = "new_family"  # ← Your feature family name (no spaces, snake_case)
# ═══════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # futures/
DATA_DIR = ROOT / "data"
FEATURES_DIR = DATA_DIR / "Level_1_Features"
MODULES_DIR = FEATURES_DIR / "modules"

# All feature modules share this grain key
EVENT_KEY = ["date", "session", "orb_tf", "breakout_ts"]


def load_sources() -> dict[str, pd.DataFrame]:
    """Load L1 data sources needed for feature engineering.

    Customize this to load whatever sources your features need.
    Default loads the breakout events (which contain entry_price, atr14,
    orb_range, breakout_strength, etc.).

    Available L1 parquets:

    - ``breakout_events.parquet``  — breakout event metadata
    - ``market_context.parquet``   — VWAP, ADX, EMA at breakout time
    - ``orb_ranges.parquet``       — raw ORB high/low/range per session

    Raw data (SQLite):

    - ``MGC_1m.db``   — 1-minute OHLCV
    - ``MGC_5m.db``   - 5-minute OHLCV
    - ``MGC_15m.db``  — 15-minute OHLCV
    """
    sources = {
        "breakout_events": pd.read_parquet(FEATURES_DIR / "breakout_events.parquet"),
    }
    # If you need 1m/15m data, load them here:
    # import sqlite3
    # conn = sqlite3.connect(DATA_DIR / "Level_0_Raw" / "MGC_1m.db")
    # sources["df_1m"] = pd.read_sql(...)
    return sources


def build_features(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """╔══════════════════════════════════════════════════════════════╗
       ║  CORE LOGIC — Implement your feature engineering here.     ║
       ╚══════════════════════════════════════════════════════════════╝

    Parameters
    ----------
    sources : dict[str, pd.DataFrame]
        DataFrames from ``load_sources()``.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``EVENT_KEY`` columns + feature columns (``float32``).

    Rules
    -----
    - ✅ Include ``EVENT_KEY`` columns so the module can be merged
    - ✅ ``.astype("float32")`` on every feature column
    - ✅ Column names: snake_case, descriptive, no collisions
    - ❌ No lookahead — only use data available **before** ``breakout_ts``
    - ❌ Don't include ``EVENT_KEY`` columns as features (they're merge keys)
    """
    bo = sources["breakout_events"].copy()

    # ── Base: breakout-event universe ────────────────────────────
    result = bo[EVENT_KEY].drop_duplicates().reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────
    #  INSERT YOUR FEATURE ENGINEERING CODE BELOW
    #
    #  Examples:
    #
    #  # Simple feature from breakout_events
    #  bo_feat = bo[EVENT_KEY + ["orb_range"]].drop_duplicates()
    #  bo_feat["my_orb_range_log"] = np.log1p(bo_feat["orb_range"]).astype("float32")
    #
    #  # Merge onto universe
    #  result = result.merge(
    #      bo_feat[EVENT_KEY + ["my_orb_range_log"]],
    #      on=EVENT_KEY, how="left"
    #  )
    # ──────────────────────────────────────────────────────────────

    # Placeholder: return universe with one dummy feature
    result["dummy_feature"] = 0.0

    # ── Ensure float32 dtypes ─────────────────────────────────────
    feat_cols = [c for c in result.columns if c not in EVENT_KEY]
    for col in feat_cols:
        result[col] = result[col].astype("float32")

    return result


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

"""Shared module loader: LEFT JOIN feature module parquets onto a core DataFrame.

Usage::

    from pipeline.feature.modules.loader import load_features_from_modules

    core_df = pd.read_parquet("training_datamart_orb.parquet", columns=CORE_COLS)
    df = load_features_from_modules(Path("data/Level_1_Features/modules"), core_df)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# All feature modules share this grain — breakout-event level.
# Features are identical for rev and cont rows.
EVENT_KEY = ["date", "session", "orb_tf", "breakout_ts"]


def load_features_from_modules(
    modules_dir: Path,
    core_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """LEFT JOIN all ``*_features.parquet`` from *modules_dir* onto *core_df*.

    Parameters
    ----------
    modules_dir : Path
        Directory containing ``*_features.parquet`` files.
        Each module must have grain ``(date, session, orb_tf, breakout_ts)``.
    core_df : pd.DataFrame
        Core DataFrame with at least the ``EVENT_KEY`` columns + labels.
    verbose : bool
        Print progress messages (default: True).

    Returns
    -------
    pd.DataFrame
        *core_df* augmented with all feature columns from all modules.

    Notes
    -----
    - Modules are merged in alphabetical file order.
    - Column name conflicts produce a warning (pandas adds ``_x`` / ``_y`` suffixes).
    - Missing modules are silently skipped (no error if directory is empty).
    """
    module_files = sorted(modules_dir.glob("*_features.parquet"))
    if not module_files:
        if verbose:
            print("[Modules] No *_features.parquet files found — returning core_df as-is")
        return core_df

    df = core_df.copy()
    for fpath in module_files:
        module = pd.read_parquet(fpath)

        # ── Warn on column name conflicts ────────────────────────
        new_feats = set(module.columns) - set(EVENT_KEY)
        existing_feats = set(df.columns) - set(EVENT_KEY)
        overlap = new_feats & existing_feats
        if overlap and verbose:
            print(
                f"[Modules] ⚠️  COLUMN CONFLICT in {fpath.name}: "
                f"{sorted(overlap)}. "
                f"Later module gets ``_y`` suffix — rename to avoid silent duplicates."
            )

        before = len(df)
        df = df.merge(module, on=EVENT_KEY, how="left")
        n_feat = len(module.columns) - len(EVENT_KEY)
        if verbose:
            print(
                f"[Modules] {fpath.name}: {len(module):,} rows, "
                f"{n_feat} feats, merge_on={EVENT_KEY}, "
                f"core_rows={before} → {len(df)}"
            )

    return df

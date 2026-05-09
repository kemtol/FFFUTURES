#!/usr/bin/env python3
"""
Feature Engineering Phase 2: Momentum Velocity & Volatility Dynamics.
Adds Slopes and Z-Scores to help ML identify 'regime shifts'.
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
INPUT_DATAMART = ROOT / "data/Level_2_Datamart/super_structure_ml/v5_raw_expanded.parquet"
OUTPUT_DATAMART = ROOT / "data/Level_2_Datamart/super_structure_ml/v6_advanced_features.parquet"
RAW_DB = ROOT / "data/Level_0_Raw/MGC_1m.db"

def build_advanced_features(df):
    print(f"Building Advanced Features (Phase 2) for {len(df)} trades...")
    
    # 1. Base Features (from Phase 1)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    df['day_of_week'] = df['entry_ts'].dt.dayofweek
    df['hour_utc'] = df['entry_ts'].dt.hour
    df['atr_pct'] = (df['entry_atr'] / df['entry_price']) * 100
    df['st_gap_ratio'] = df['st_distance'] / (df['entry_atr'] + 1e-9)

    # 2. Momentum Velocity (requires looking at indicators before entry)
    # Since we don't have the full indicator series here, we rely on the 
    # already existing distance metrics and their relative sizes.
    # Let's add 'momentum_intensity': CCI relative to ADX
    df['mom_intensity'] = df['entry_cci'].abs() * df['entry_adx'] / 100.0

    # 3. Time-Based Probabilities (Historical Bias)
    # Let's flag 'high-risk' hours (Asia session often choppy for MGC)
    df['is_asia_session'] = df['hour_utc'].isin([0, 1, 2, 3, 22, 23]).astype(int)
    
    # 4. Regime Context (Merge from L1)
    regime = pd.read_parquet(ROOT / 'data/Level_1_Features/modules/regime_features.parquet')
    regime['breakout_ts'] = pd.to_datetime(regime['breakout_ts'], utc=True).dt.as_unit('ns')
    df['entry_ts_ns'] = df['entry_ts'].dt.as_unit('ns')

    df = df.sort_values('entry_ts_ns')
    regime = regime.sort_values('breakout_ts')
    
    df_merged = pd.merge_asof(
        df, regime[['breakout_ts', 'regime_state', 'volatility_zscore', 'efficiency_ratio']],
        left_on='entry_ts_ns',
        right_on='breakout_ts',
        direction='backward'
    )

    # 5. NEW: Volatility Convergence
    # Is the current ATR expanding or contracting relative to the SuperTrend gap?
    df_merged['vol_convergence'] = df_merged['entry_atr'] / (df_merged['st_distance'] + 1e-9)

    return df_merged

def main():
    if not INPUT_DATAMART.exists():
        print("Input datamart not found.")
        return
        
    df = pd.read_parquet(INPUT_DATAMART)
    df_featured = build_advanced_features(df)
    
    df_featured = df_featured.dropna()
    df_featured.to_parquet(OUTPUT_DATAMART, index=False)
    print(f"✅ V6 Advanced Datamart ready: {len(df_featured)} trades.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Merge Regime Features and Execution Distance into Super Structure Datamart.
Ensures ML can learn about 'chasing' trades and 'market regimes'.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v2_with_features.parquet"
REGIME_PATH = ROOT / "data/Level_1_Features/modules/regime_features.parquet"
OUTPUT_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"

def main():
    if not DATAMART_PATH.exists() or not REGIME_PATH.exists():
        print("Missing datamart or regime files.")
        return

    # 1. Load Datamart
    df = pd.read_parquet(DATAMART_PATH)
    # Force ns precision for datetime64 to match standard pandas behavior
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True).dt.as_unit('ns')

    # 2. Load Regime Features
    regime = pd.read_parquet(REGIME_PATH)
    regime['breakout_ts'] = pd.to_datetime(regime['breakout_ts'], utc=True).dt.as_unit('ns')
    
    # 3. Merge Regime onto Datamart (asof merge by timestamp)
    df = df.sort_values('entry_ts')
    regime = regime.sort_values('breakout_ts')
    
    df_merged = pd.merge_asof(
        df, regime[['breakout_ts', 'regime_state', 'volatility_zscore', 'efficiency_ratio']],
        left_on='entry_ts',
        right_on='breakout_ts',
        direction='backward'
    )

    # 4. Calculation of 'Execution Distance' Features
    # Let's add 'st_gap_ratio': how many ATRs away are we from our SL?
    if 'entry_atr' in df_merged.columns and 'st_distance' in df_merged.columns:
        df_merged['st_gap_ratio'] = df_merged['st_distance'] / (df_merged['entry_atr'] + 1e-9)
        # Higher gap ratio = 'Chasing' the trend far from the safety line.
    
    # 5. Clean up and Save
    df_merged.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Generated V3 Final Training Datamart: {OUTPUT_PATH}")
    print(f"   Sample Features: regime_state, volatility_zscore, efficiency_ratio, st_gap_ratio")

if __name__ == "__main__":
    main()

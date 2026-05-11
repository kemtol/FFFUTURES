#!/usr/bin/env python3
"""
Alpha Discovery: Macro Correlation Search v2.
Checks if DXY (Dollar) or US10Y (Yields) daily direction influence Gold Pullbacks.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('/home/kemal/futures')
MACRO_PATH = ROOT / "data/Level_1_Features/macro_data.parquet"
PB_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v1_8_training_datamart.parquet"

def discover_macro_alphas():
    print("🔭 Starting Alpha Discovery (Macro Correlation v2)...")
    
    # 1. Load Datamart
    pb = pd.read_parquet(PB_PATH)
    pb['entry_ts'] = pd.to_datetime(pb['entry_ts'], utc=True)
    pb['date_key'] = pb['entry_ts'].dt.date
    
    # 2. Load Macro (Daily)
    macro = pd.read_parquet(MACRO_PATH)
    macro['date_key'] = pd.to_datetime(macro['date']).dt.date
    
    # 3. Merge on Date
    df = pd.merge(pb, macro, on='date_key', how='left')
    
    # 4. Analyze DXY Impact
    if 'dxy_return' in df.columns:
        df['dxy_group'] = np.where(df['dxy_return'] > 0, 'DXY_Strength', 'DXY_Weakness')
        print("\n--- Win Rate by DXY Daily Return ---")
        stats = df.groupby('dxy_group')['label'].agg(['mean', 'count'])
        print(stats)
        
    # 5. Analyze US10Y Impact
    if 'us10y_change' in df.columns:
        df['yield_group'] = np.where(df['us10y_change'] > 0, 'Yield_Rising', 'Yield_Falling')
        print("\n--- Win Rate by Yield Daily Change ---")
        stats = df.groupby('yield_group')['label'].agg(['mean', 'count'])
        print(stats)

    # 6. Analyze Oil Impact
    if 'oil_return' in df.columns:
        df['oil_group'] = np.where(df['oil_return'] > 0, 'Oil_Up', 'Oil_Down')
        print("\n--- Win Rate by Oil Daily Return ---")
        stats = df.groupby('oil_group')['label'].agg(['mean', 'count'])
        print(stats)

    print("\n✅ Macro Alpha Search complete.")

if __name__ == "__main__":
    discover_macro_alphas()

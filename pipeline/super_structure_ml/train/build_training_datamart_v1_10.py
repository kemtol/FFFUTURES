#!/usr/bin/env python3
"""
SMART_1 Training Datamart Builder v1.10: 
Includes MACRO Alphas (Oil, Yields) for Fundamental-Aware ML.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('/home/kemal/futures')
BASE_DATAMART = ROOT / "data/Level_2_Datamart/super_structure_ml/v1_8_training_datamart.parquet"
MACRO_PATH = ROOT / "data/Level_1_Features/macro_data.parquet"
OUTPUT_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v1_10_training_datamart.parquet"

def build_v1_10_macro_datamart():
    print("🚜 Integrating Macro Alphas into Datamart...")
    
    # 1. Load Base Datamart
    df = pd.read_parquet(BASE_DATAMART)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    df['date_key'] = df['entry_ts'].dt.date
    
    # 2. Load Macro Data
    macro = pd.read_parquet(MACRO_PATH)
    macro['date_key'] = pd.to_datetime(macro['date']).dt.date
    
    # 3. Modular Left Join on Date
    # We only take the features that showed alpha: oil_return and us10y_change
    macro_features = ['date_key', 'oil_return', 'us10y_change', 'dxy_return']
    df = pd.merge(df, macro[macro_features], on='date_key', how='left')
    
    # Fill NaNs in macro (usually weekends or missing days) with 0 (neutral)
    df['oil_return'] = df['oil_return'].fillna(0)
    df['us10y_change'] = df['us10y_change'].fillna(0)
    df['dxy_return'] = df['dxy_return'].fillna(0)
    
    # 4. Save v1.10 Datamart
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Macro-Enhanced Datamart v1.10 Ready: {len(df)} samples.")
    print(f"New Features: oil_return, us10y_change, dxy_return")

if __name__ == "__main__":
    build_v1_10_macro_datamart()

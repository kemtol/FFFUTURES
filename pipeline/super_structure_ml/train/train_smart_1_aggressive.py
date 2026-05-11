#!/usr/bin/env python3
"""
SMART_1: Aggressive Brain Training.
Specialized for High-Frequency (5 trades/day) during Efficient Regimes.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
SMART_DIR = ROOT / "model/SUPER_STRUCTURE/SMART_1"

def train_aggressive_brain():
    print("🔥 Training SMART_1 Aggressive Brain...")
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    
    # We only train on 'Good Days' to specialize the model for profit maximization
    # (Simplified: we use the whole history but weight positive outcomes higher or use lower thresholds)
    features = ["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio", "volatility_zscore", "wick_ratio", "candle_body_atr"]
    
    train = df[df['entry_ts'] < '2026-01-01'].copy()
    y_train = (train['pnl_usd'] > 0).astype(int)
    
    dtrain = lgb.Dataset(train[features], label=y_train)
    
    # Aggressive params: higher learning rate, more leaves
    model = lgb.train({
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.1,
        'num_leaves': 63,
        'verbosity': -1
    }, dtrain, num_boost_round=150)
    
    model.save_model(str(SMART_DIR / "aggressive_brain.txt"))
    
    print(f"✅ SMART_1 Aggressive Brain trained.")

if __name__ == "__main__":
    train_aggressive_brain()

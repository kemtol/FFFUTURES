#!/usr/bin/env python3
"""
SMART_1: Aggressive Brain Re-Training (v1.5)
Training with new Alpha features (Volume, Slope, RSI).
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/pullback_events_enriched.parquet"
SMART_DIR = ROOT / "model/SUPER_STRUCTURE/SMART_1"

def train_aggressive_v1_5():
    print("🔥 Re-training SMART_1 Aggressive Brain with Alpha Features...")
    
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    
    # Drop NaNs
    features = [
        "entry_adx", "cci_abs", "st_gap_ratio", 
        "efficiency_ratio", "volatility_zscore", 
        "wick_ratio", "candle_body_atr",
        "vol_ratio", "st_slope", "rsi_5" # NEW ALPHA
    ]
    df = df.dropna(subset=features)
    df['target'] = (df['pnl_usd'] > 0).astype(int)
    
    train_df = df[df['entry_ts'] < '2026-01-01'].copy()
    test_df = df[df['entry_ts'] >= '2026-01-01'].copy()
    
    dtrain = lgb.Dataset(train_df[features], label=train_df['target'])
    dtest = lgb.Dataset(test_df[features], label=test_df['target'], reference=dtrain)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'learning_rate': 0.02,
        'num_leaves': 15, # Smaller trees to prevent overfitting
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1
    }
    
    model = lgb.train(
        params, 
        dtrain, 
        valid_sets=[dtrain, dtest],
        valid_names=['train', 'valid'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )
    
    # Save
    model.save_model(str(SMART_DIR / "aggressive_brain.txt"))
    
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\n--- Aggressive Brain Feature Importance (GAIN) ---")
    print(importance.to_string(index=False))
    
    print(f"\n✅ Re-training Complete. New Validation AUC: {model.best_score['valid']['auc']:.4f}")

if __name__ == "__main__":
    train_aggressive_v1_5()

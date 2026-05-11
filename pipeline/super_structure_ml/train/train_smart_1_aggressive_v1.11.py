#!/usr/bin/env python3
"""
SMART_1 Aggressive Brain Trainer (v1.11): 
The DEMA Family Specialist.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import timedelta

ROOT = Path('/home/kemal/futures')
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v1_11_training_datamart.parquet"
SMART_DIR = ROOT / "model/SUPER_STRUCTURE/SMART_1"

def train_v1_11_dema():
    print("🔥 Training DEMA-Enhanced Aggressive Brain...")
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    
    # 1. Feature Set (Focus on DEMA Matrix)
    features = [
        "dist_d50_atr", "dist_d100_atr", "dist_d200_atr", 
        "d100_slope", "d200_slope", "dema_stack",
        "entry_adx", "cci_abs", "st_gap_ratio",
        "wick_ratio", "candle_body_atr", "rsi_7",
        "oil_return", "us10y_change", "dxy_return",
        "session_cluster"
    ]
    
    df = df.dropna(subset=features)
    
    # 720d Train / 200d OOT
    today = df['entry_ts'].max()
    oot_start = today - timedelta(days=200)
    train_start = today - timedelta(days=720)
    
    train_df = df[(df['entry_ts'] >= train_start) & (df['entry_ts'] < oot_start)].copy()
    test_df = df[df['entry_ts'] >= oot_start].copy()
    
    print(f"Features: {len(features)}")
    print(f"Train Size: {len(train_df)} | OOT Size: {len(test_df)}")
    
    dtrain = lgb.Dataset(train_df[features], label=train_df['label'])
    dtest = lgb.Dataset(test_df[features], label=test_df['label'], reference=dtrain)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'learning_rate': 0.01,
        'num_leaves': 15,
        'feature_fraction': 0.6,
        'is_unbalance': True,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0
    }
    
    model = lgb.train(
        params, 
        dtrain, 
        valid_sets=[dtrain, dtest],
        valid_names=['train', 'valid'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )
    
    model.save_model(str(SMART_DIR / "aggressive_brain_v1_11.txt"))
    
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\n--- Feature Importance (DEMA Family Gain) ---")
    print(importance.head(10).to_string(index=False))
    
    print(f"\n✅ v1.11 Training Complete. OOT AUC: {model.best_score['valid']['auc']:.4f}")

if __name__ == "__main__":
    train_v1_11_dema()

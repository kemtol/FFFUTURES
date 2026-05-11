#!/usr/bin/env python3
"""
SMART_1: Aggressive Brain Training (v1.4)
Specialized in Pullback Filtering.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/pullback_events_enriched.parquet"
SMART_DIR = ROOT / "model/SUPER_STRUCTURE/SMART_1"

def train_aggressive_pullback():
    print("🔥 Starting SMART_1 Aggressive Brain Training (Pullback Specialist)...")
    
    # 1. Load Data
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    
    # Clean up NaNs from indicator warmup
    df = df.dropna(subset=['entry_adx', 'cci_abs', 'efficiency_ratio'])
    
    # 2. Features and Target
    features = [
        "entry_adx", "cci_abs", "st_gap_ratio", 
        "efficiency_ratio", "volatility_zscore", 
        "wick_ratio", "candle_body_atr", "session_cluster"
    ]
    
    df['target'] = (df['pnl_usd'] > 0).astype(int)
    
    # 3. Time-Based Split
    train_mask = (df['entry_ts'] < '2026-01-01')
    test_mask = (df['entry_ts'] >= '2026-01-01')
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"Training on: {len(train_df)} pullbacks (2023-2025)")
    print(f"Testing on: {len(test_df)} pullbacks (2026 YTD)")
    
    # 4. Train LightGBM
    dtrain = lgb.Dataset(train_df[features], label=train_df['target'])
    dtest = lgb.Dataset(test_df[features], label=test_df['target'], reference=dtrain)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'is_unbalance': True # Help catch positive outcomes in noisy data
    }
    
    model = lgb.train(
        params, 
        dtrain, 
        valid_sets=[dtrain, dtest],
        valid_names=['train', 'valid'],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # 5. Save Artifacts
    SMART_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(SMART_DIR / "aggressive_brain.txt"))
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    importance.to_csv(SMART_DIR / "aggressive_importance.csv", index=False)
    print("\n--- Aggressive Brain Feature Importance ---")
    print(importance.to_string(index=False))
    
    # Initial Test Score
    test_df['prob'] = model.predict(test_df[features])
    print(f"\n✅ Aggressive Brain Training Complete.")
    print(f"Aggressive Test AUC: {model.best_score['valid']['auc']:.4f}")

if __name__ == "__main__":
    train_aggressive_pullback()

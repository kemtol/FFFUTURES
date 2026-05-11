#!/usr/bin/env python3
"""
SMART_1 Aggressive Brain Trainer (v1.11 - DEEP): 
Forcing deep learning to overcome underfitting.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import timedelta

ROOT = Path('/home/kemal/futures')
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v1_11_training_datamart.parquet"
SMART_DIR = ROOT / "model/SUPER_STRUCTURE/SMART_1"

def train_v1_11_deep():
    print("🔥 Executing DEEP TRAINING for SMART_1 Aggressive Brain...")
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    
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
    
    dtrain = lgb.Dataset(train_df[features], label=train_df['label'])
    dtest = lgb.Dataset(test_df[features], label=test_df['label'], reference=dtrain)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'learning_rate': 0.05, # Higher LR
        'num_leaves': 63, # Deeper trees
        'feature_fraction': 0.7,
        'is_unbalance': True,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1
    }
    
    # Force 1000 rounds without early stopping to see full potential
    model = lgb.train(
        params, 
        dtrain, 
        valid_sets=[dtrain, dtest],
        valid_names=['train', 'valid'],
        num_boost_round=1000
    )
    
    model.save_model(str(SMART_DIR / "aggressive_brain_v1_11_deep.txt"))
    
    print("\n--- DEEP TRAINING RESULTS ---")
    print(f"Final Train AUC: {model.best_score['train']['auc']:.4f}")
    print(f"Final OOT AUC: {model.best_score['valid']['auc']:.4f}")
    
    # Check if we found a better peak earlier
    # (Manually identifying the best validation round)
    # Note: Booster.best_iteration only works with early stopping, 
    # but we can look at the trend.

    # Quick profit check on OOT
    test_df['prob'] = model.predict(test_df[features])
    final_trades = test_df[test_df['prob'] >= 0.55]
    print(f"\nOOT Trades (@0.55): {len(final_trades)}")
    print(f"OOT PnL: ${final_trades['pnl_usd'].sum():,.2f}")

if __name__ == "__main__":
    train_v1_11_deep()

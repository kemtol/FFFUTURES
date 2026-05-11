#!/usr/bin/env python3
"""
SMART_1 Aggressive Brain Trainer v1.8: 
The Final Pullback Machine.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

ROOT = Path('/home/kemal/futures')
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v1_8_training_datamart.parquet"
SMART_DIR = ROOT / "model/SUPER_STRUCTURE/SMART_1"

def train_v1_8_final():
    print("🔥 Training SMART_1 Aggressive Brain (v1.8 Final)...")
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    
    features = [
        "entry_adx", "cci_abs", "st_gap_ratio", 
        "efficiency_ratio", "volatility_zscore", 
        "wick_ratio", "candle_body_atr",
        "st_slope", "rsi_7", "session_cluster"
    ]
    
    # Drop records with NaNs
    df = df.dropna(subset=features)
    
    train_df = df[df['entry_ts'] < '2026-01-01'].copy()
    test_df = df[df['entry_ts'] >= '2026-01-01'].copy()
    
    print(f"Sample Size: {len(train_df)} Train / {len(test_df)} Test")
    
    dtrain = lgb.Dataset(train_df[features], label=train_df['label'])
    dtest = lgb.Dataset(test_df[features], label=test_df['label'], reference=dtrain)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'learning_rate': 0.02,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'is_unbalance': True
    }
    
    model = lgb.train(
        params, 
        dtrain, 
        valid_sets=[dtrain, dtest],
        valid_names=['train', 'valid'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )
    
    # SAVE AS OFFICIAL AGGRESSIVE BRAIN
    model.save_model(str(SMART_DIR / "aggressive_brain_v1_8.txt"))
    
    print(f"✅ Training Complete. Validation AUC: {model.best_score['valid']['auc']:.4f}")
    
    # Quick Profitability Check (Isolated)
    test_df['prob'] = model.predict(test_df[features])
    # Use threshold to pick top 5 trades/day (~400 trades per 100 days)
    # We'll use 0.55 as a starting point
    final_trades = test_df[test_df['prob'] >= 0.55]
    print(f"Isolated Aggressive PnL (2026): ${final_trades['pnl_usd'].sum():,.2f}")
    print(f"Isolated Aggressive Trades: {len(final_trades)}")

if __name__ == "__main__":
    train_v1_8_final()

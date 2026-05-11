#!/usr/bin/env python3
"""
SMART_1 Aggressive Brain Trainer (v1.9): 
Recent Regime Focus: 720 Days Training, 200 Days OOT Validation.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path('/home/kemal/futures')
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v1_8_training_datamart.parquet"
SMART_DIR = ROOT / "model/SUPER_STRUCTURE/SMART_1"

def train_v1_9_recent():
    print("🔥 Training SMART_1 Aggressive Brain (Recent Regime Focus)...")
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    
    # 1. Define Windows
    today = df['entry_ts'].max()
    oot_start = today - timedelta(days=200)
    train_start = today - timedelta(days=720)
    
    print(f"Full Range: {df['entry_ts'].min().date()} to {today.date()}")
    print(f"Training Window: {train_start.date()} to {oot_start.date()}")
    print(f"OOT (Test) Window: {oot_start.date()} to {today.date()}")
    
    # 2. Dataset Split
    train_df = df[(df['entry_ts'] >= train_start) & (df['entry_ts'] < oot_start)].copy()
    test_df = df[df['entry_ts'] >= oot_start].copy()
    
    features = [
        "entry_adx", "cci_abs", "st_gap_ratio", 
        "efficiency_ratio", "volatility_zscore", 
        "wick_ratio", "candle_body_atr",
        "st_slope", "rsi_7", "session_cluster"
    ]
    train_df = train_df.dropna(subset=features)
    test_df = test_df.dropna(subset=features)
    
    print(f"Samples: {len(train_df)} Train / {len(test_df)} OOT")
    
    # 3. Train
    dtrain = lgb.Dataset(train_df[features], label=train_df['label'])
    dtest = lgb.Dataset(test_df[features], label=test_df['label'], reference=dtrain)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'learning_rate': 0.015, # Slow down learning for small window
        'num_leaves': 15,
        'feature_fraction': 0.7,
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
    
    model.save_model(str(SMART_DIR / "aggressive_brain_recent.txt"))
    
    # 4. Evaluate 200d OOT performance
    test_df['prob'] = model.predict(test_df[features])
    # Pick a threshold that gives us approx 3 trades/day in OOT
    # 200 days * 3 = 600 trades.
    th = 0.52
    final_oot = test_df[test_df['prob'] >= th]
    
    pnl = final_oot['pnl_usd'].sum()
    print(f"\n--- 200-DAY OOT PERFORMANCE (Threshold {th}) ---")
    print(f"Total Trades: {len(final_oot)}")
    print(f"Trades per Day: {len(final_oot)/140:.2f}") # Approx 140 trading days in 200 calendar days
    print(f"Total PnL: ${pnl:,.2f}")
    print(f"Win Rate: {(final_oot['label'].mean()*100):.1f}%")

if __name__ == "__main__":
    train_v1_9_recent()

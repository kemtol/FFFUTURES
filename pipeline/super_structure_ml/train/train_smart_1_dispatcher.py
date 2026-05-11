#!/usr/bin/env python3
"""
SMART_1: Regime Classifier Training.
Determines if today is an 'Aggressive' (Efficient) or 'Conservative' (Choppy) day.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
SMART_DIR = ROOT / "model/SUPER_STRUCTURE/SMART_1"

def train_regime_dispatcher():
    print("🧠 Training SMART_1 Regime Dispatcher...")
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    df['date_key'] = df['entry_ts'].dt.date
    
    # 1. Define Labels: What is a 'Good Day' for Aggression?
    # A day is GOOD if the win rate of unfiltered trades > 55% AND total profit > 0
    daily_stats = df.groupby('date_key').agg({
        'pnl_usd': ['sum', 'count', lambda x: (x > 0).mean()]
    })
    daily_stats.columns = ['total_pnl', 'trade_count', 'win_rate']
    
    # Target: 1 if market is trending/efficient, 0 if choppy
    daily_stats['is_aggressive_day'] = ((daily_stats['total_pnl'] > 0) & (daily_stats['win_rate'] > 0.55)).astype(int)
    
    # 2. Prepare Features for the Dispatcher (Daily Context)
    # We want features that describe the session/day PRIOR to individual trades
    dispatcher_df = df.merge(daily_stats[['is_aggressive_day']], on='date_key')
    
    regime_features = ["efficiency_ratio", "volatility_zscore", "entry_adx", "cci_abs"]
    
    train = dispatcher_df[dispatcher_df['entry_ts'] < '2026-01-01'].copy()
    y_train = train['is_aggressive_day']
    
    dtrain = lgb.Dataset(train[regime_features], label=y_train)
    
    # 3. Train the Dispatcher
    model = lgb.train({
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1
    }, dtrain, num_boost_round=100)
    
    # 4. Save to SMART_1
    model.save_model(str(SMART_DIR / "regime_dispatcher.txt"))
    
    with open(SMART_DIR / "SMART_1_manifest.json", 'w') as f:
        json.dump({
            "name": "SMART_1",
            "components": {
                "regime_dispatcher": "regime_dispatcher.txt",
                "conservative_brain": "conservative_brain.txt",
                "aggressive_brain": "aggressive_brain.txt"
            },
            "regime_features": regime_features
        }, f, indent=2)

    print(f"✅ SMART_1 Regime Dispatcher trained and saved.")

if __name__ == "__main__":
    train_regime_dispatcher()

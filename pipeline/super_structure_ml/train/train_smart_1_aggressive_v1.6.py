#!/usr/bin/env python3
"""
SMART_1: Aggressive Brain Training (v1.6)
Specialized for Scalping Mode (RR 1:1).
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/pullback_events_enriched.parquet"
SMART_DIR = ROOT / "model/SUPER_STRUCTURE/SMART_1"

def train_aggressive_v1_6():
    print("🔥 Training SMART_1 Aggressive Brain (Scalp RR 1:1)...")
    
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    
    features = [
        "entry_adx", "cci_abs", "st_gap_ratio", 
        "efficiency_ratio", "volatility_zscore", 
        "wick_ratio", "candle_body_atr",
        "vol_ratio", "st_slope", "rsi_5"
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
        'num_leaves': 31,
        'feature_fraction': 0.8,
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
    
    model.save_model(str(SMART_DIR / "aggressive_brain_rr1.txt"))
    print(f"✅ Scalp Brain Training Complete. AUC: {model.best_score['valid']['auc']:.4f}")

if __name__ == "__main__":
    train_aggressive_v1_6()

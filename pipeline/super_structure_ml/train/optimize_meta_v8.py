#!/usr/bin/env python3
"""
Meta-v8 Optimizer: Precision Sharpening.
Baseline: Meta-v7 Refined ($300 Limit + Dynamic Thresholds).
Goal: Reduce Max DD < -$1,500 while keeping Pass Speed < 30 days.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
V8_DIR = ROOT / "model/SUPER_STRUCTURE/meta_v8"

def run_v8_research():
    print("🚀 Initializing Meta-v8 Research: Precision Sharpening Phase...")
    V8_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    
    # Define New Candidate Features for V8
    # We want to see if 'wick_ratio' and 'candle_body_atr' can filter exhaustion.
    baseline_features = ["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio", "volatility_zscore", "session_cluster"]
    v8_candidates = baseline_features + ["wick_ratio", "candle_body_atr"]
    
    # 1. Training Setup
    train = df[df['entry_ts'] < '2026-01-01'].copy()
    test = df[df['entry_ts'] >= '2026-01-01'].copy()
    y_train = (train['pnl_usd'] > 0).astype(int)
    
    print(f"Baseline Features: {len(baseline_features)}")
    print(f"V8 Candidate Features: {len(v8_candidates)}")

    # 2. Train V8 Prototype
    dtrain = lgb.Dataset(train[v8_candidates], label=y_train)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8
    }
    
    model = lgb.train(params, dtrain, num_boost_round=100)
    
    # 3. Save Baseline Model and Feature Importance
    model.save_model(str(V8_DIR / "v8_prototype_model.txt"))
    
    importance = pd.DataFrame({
        'feature': v8_candidates,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    importance.to_csv(V8_DIR / "feature_importance_v8.csv", index=False)
    
    print("\n--- Meta-v8 Initial Feature Importance ---")
    print(importance.to_string(index=False))
    
    with open(V8_DIR / "research_status.json", 'w') as f:
        json.dump({
            "phase": "v8_precision_sharpening",
            "baseline": "meta_v7_refined",
            "candidates": v8_candidates,
            "status": "initialized"
        }, f, indent=2)

    print(f"\n✅ Meta-v8 Structure Ready in {V8_DIR}")

if __name__ == "__main__":
    run_v8_research()

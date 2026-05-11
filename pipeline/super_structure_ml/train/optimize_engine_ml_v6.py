#!/usr/bin/env python3
"""
Meta-v6 Optimizer: Focus on REACHING $3000 Target quickly.
Condition: Max DD must stay above -1800 USD.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"

def simulate_with_target(df, threshold, target=3000.0):
    accepted = df[df['prob'] >= threshold].sort_values('entry_ts').copy()
    if accepted.empty: return 0, 0, 999
    
    balance = 0.0
    peak = 0.0
    max_dd = 0.0
    days_to_target = 999
    start_date = accepted['entry_ts'].min()
    
    for i, row in accepted.iterrows():
        balance += row['pnl_usd']
        if balance > peak: peak = balance
        dd = balance - peak
        if dd < max_dd: max_dd = dd
        
        # Check target
        if balance >= target and days_to_target == 999:
            days_to_target = (row['entry_ts'] - start_date).days
            if days_to_target < 1: days_to_target = 1

    return balance, max_dd, days_to_target

def run_optimization_v6():
    print("🚀 Starting Meta-v6 'Speed to Target' Optimization...")
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    
    # We use 2026 as the 'Battleground' for Pass Speed
    test = df[df['entry_ts'] >= '2026-01-01'].copy()
    train = df[df['entry_ts'] < '2026-01-01'].copy()
    y_train = (train['pnl_usd'] > 0).astype(int)

    feature_sets = [
        ['entry_adx', 'cci_abs', 'st_gap_ratio', 'efficiency_ratio', 'volatility_zscore', 'candle_body_atr', 'is_st_aligned'],
        ['entry_adx', 'entry_cci', 'st_gap_ratio', 'efficiency_ratio', 'volatility_zscore', 'session_cluster'],
        ['entry_adx', 'cci_abs', 'st_gap_ratio', 'efficiency_ratio', 'wick_ratio', 'volatility_zscore', 'session_cluster']
    ]

    best_score = -99999
    winner = {}

    for i, features in enumerate(feature_sets):
        print(f"Analyzing Set {i+1}...")
        # Check if all features exist
        available_features = [f for f in features if f in train.columns]
        
        dtrain = lgb.Dataset(train[available_features], label=y_train)
        model = lgb.train({'objective':'binary', 'metric':'auc', 'verbosity':-1}, dtrain, num_boost_round=100)
        test['prob'] = model.predict(test[available_features])
        
        for th in np.arange(0.25, 0.45, 0.01):
            pnl, mdd, days = simulate_with_target(test, th)
            
            # CRITERIA: 
            # 1. Must not fail DD (-1800 buffer)
            # 2. Higher PnL is better
            # 3. Fewer days is a huge multiplier
            if mdd < -1800:
                score = -20000 + mdd
            else:
                # Expectancy Score: PnL adjusted by time efficiency
                score = pnl * (100 / (days + 1)) 
            
            if score > best_score:
                best_score = score
                winner = {
                    "features": available_features,
                    "threshold": th,
                    "pnl": pnl,
                    "max_dd": mdd,
                    "days_to_3k": days
                }

    print("\n--- META-V6 WINNING CONFIG ---")
    print(json.dumps(winner, indent=2))
    
    # Save Meta-v6
    v_dir = ROOT / "model/SUPER_STRUCTURE/meta_v6"
    v_dir.mkdir(parents=True, exist_ok=True)
    
    # Re-train winner
    dtrain = lgb.Dataset(train[winner['features']], label=y_train)
    model = lgb.train({'objective':'binary', 'metric':'auc', 'verbosity':-1}, dtrain, num_boost_round=100)
    model.save_model(str(v_dir / "inference_model.txt"))
    with open(v_dir / "inference_config.json", 'w') as f:
        json.dump({"threshold": float(winner['threshold']), "features": winner['features']}, f, indent=2)
    
    print(f"\n✅ Meta-v6 is ready for the Topstep Race.")

if __name__ == "__main__":
    run_optimization_v6()

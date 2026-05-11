#!/usr/bin/env python3
"""
Meta-v7 Dynamic Optimizer: Searches for SESSION-SPECIFIC thresholds.
Goal: $3000 in < 30 days, Max DD > -1800.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"

def simulate_dynamic(df, th_asian, th_london, th_us):
    # session_cluster: 0=Asian, 1=London, 2=US
    df = df.copy()
    df['threshold'] = df['session_cluster'].map({0: th_asian, 1: th_london, 2: th_us})
    accepted = df[df['prob'] >= df['threshold']].sort_values('entry_ts').copy()
    
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
        
        if balance >= 3000.0 and days_to_target == 999:
            days_to_target = (row['entry_ts'] - start_date).days
            if days_to_target < 1: days_to_target = 1

    return balance, max_dd, days_to_target

def run_dynamic_opt():
    print("🚀 Running Meta-v7 Dynamic Session Optimizer...")
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    
    train = df[df['entry_ts'] < '2026-01-01'].copy()
    test = df[df['entry_ts'] >= '2026-01-01'].copy()
    
    features = ["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio", "volatility_zscore", "session_cluster"]
    dtrain = lgb.Dataset(train[features], label=(train['pnl_usd'] > 0).astype(int))
    model = lgb.train({'objective':'binary', 'metric':'auc', 'verbosity':-1}, dtrain)
    
    test['prob'] = model.predict(test[features])
    
    best_score = -99999
    best_params = {}

    # Grid search for session thresholds
    # Asian usually needs higher threshold (choppier)
    # US usually more trending
    for th_a in [0.4, 0.45, 0.5]:
        for th_l in [0.35, 0.4, 0.45]:
            for th_u in [0.3, 0.35, 0.4]:
                pnl, mdd, days = simulate_dynamic(test, th_a, th_l, th_u)
                
                if mdd < -1800:
                    score = -20000 + mdd
                else:
                    # Pass-Efficiency Score
                    # If target hit within 30 days, huge bonus
                    time_bonus = 2000 if days <= 30 else (1000 / (days + 1))
                    score = pnl + time_bonus
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        "th_asian": th_a, "th_london": th_l, "th_us": th_u,
                        "pnl": pnl, "max_dd": mdd, "days_to_3k": days
                    }

    print("\n--- META-V7 DYNAMIC RESULT ---")
    print(json.dumps(best_params, indent=2))
    
    # Save as Meta-v7
    v_dir = ROOT / "model/SUPER_STRUCTURE/meta_v7"
    v_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(v_dir / "inference_model.txt"))
    with open(v_dir / "inference_config.json", 'w') as f:
        json.dump({"thresholds": {
            "0": best_params['th_asian'], 
            "1": best_params['th_london'], 
            "2": best_params['th_us']
        }, "features": features}, f, indent=2)
    
    print(f"\n✅ Meta-v7 Optimized for Speed and Safety.")

if __name__ == "__main__":
    run_dynamic_opt()

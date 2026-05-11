#!/usr/bin/env python3
"""
Meta-v7 Daily Limit Simulator: 
Applies a $500 Daily Loss Limit to see if it saves the account from MLL breach.
"""

import pandas as pd
import json
import lightgbm as lgb
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
MODEL_DIR = ROOT / "model/SUPER_STRUCTURE/meta_v7"

def run_daily_limit_sim(daily_limit=500.0):
    # 1. Load Meta-v7
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    df['date_key'] = df['entry_ts'].dt.date
    
    model = lgb.Booster(model_file=str(MODEL_DIR / "inference_model.txt"))
    config = json.load(open(MODEL_DIR / "inference_config.json"))
    
    df['prob'] = model.predict(df[config['features']])
    
    # 2. Apply Dynamic Thresholds (from Meta-v7 config)
    thresholds = {int(k): v for k, v in config['thresholds'].items()}
    df['threshold'] = df['session_cluster'].map(thresholds)
    accepted = df[df['prob'] >= df['threshold']].sort_values('entry_ts').copy()
    
    # Filter for 2026 test set
    accepted = accepted[accepted['entry_ts'] >= '2026-01-01'].copy()

    # 3. Simulation with DAILY LIMIT
    balance = 50000.0
    peak = 50000.0
    max_dd = 0.0
    
    daily_pnl = {} # Tracks PnL per day
    ledger = []
    
    for _, row in accepted.iterrows():
        d = row['date_key']
        if d not in daily_pnl: daily_pnl[d] = 0.0
        
        # If daily limit reached, SKIP
        if daily_pnl[d] <= -daily_limit:
            continue
            
        pnl = row['pnl_usd']
        balance += pnl
        daily_pnl[d] += pnl
        
        if balance > peak: peak = balance
        dd = balance - peak
        if dd < max_dd: max_dd = dd
        
        ledger.append({
            "entry_ts": row['entry_ts'].strftime('%Y-%m-%d %H:%M'),
            "pnl": pnl,
            "balance": balance,
            "drawdown": dd,
            "daily_state": daily_pnl[d]
        })

    print(f"\n--- Meta-v7 WITH ${daily_limit} DAILY LIMIT ---")
    print(f"Total Trades Taken: {len(ledger)}")
    print(f"Final Balance: ${balance:,.2f}")
    print(f"Max Drawdown: ${max_dd:,.2f}")
    
    status = "✅ PASS TOPSTEP" if max_dd > -1800 else "❌ FAILED DD"
    print(f"Status: {status}")
    
    # Pass speed calculation
    speed = "N/A"
    current_b = 50000.0
    for i, t in enumerate(ledger):
        current_b += t['pnl']
        if (current_b - 50000.0) >= 3000.0:
            speed = f"{ (pd.to_datetime(t['entry_ts']) - accepted['entry_ts'].min()).days } Days"
            break
    print(f"Pass Speed ($3k): {speed}")

if __name__ == "__main__":
    run_daily_limit_sim(daily_limit=500.0)

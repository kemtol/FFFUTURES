#!/usr/bin/env python3
"""
Topstep Auditor v2: Supports Dynamic Thresholds and Daily Loss Limits.
Generates trade-by-trade ledger for account auditing.
"""

import pandas as pd
import json
import lightgbm as lgb
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"

def run_audit(version="meta_v7", thresholds=None, daily_limit=None, window_days=90):
    # 1. Load Data
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    
    # 2. Load Model
    model_dir = ROOT / f"model/SUPER_STRUCTURE/{version}"
    model = lgb.Booster(model_file=str(model_dir / "inference_model.txt"))
    with open(model_dir / "inference_config.json", 'r') as f:
        config = json.load(f)
    
    # Predict
    df['prob'] = model.predict(df[config['features']])
    
    # 3. Apply Filter (Dynamic or Static)
    if thresholds:
        # thresholds is a dict mapping session_cluster to float
        df['threshold'] = df['session_cluster'].map(thresholds)
    else:
        # Fallback to static if provided as a single float (backwards compat)
        df['threshold'] = config.get('threshold', 0.5)

    mask = (df['regime_state'] != 0) & (df['prob'] >= df['threshold'])
    filtered = df[mask].sort_values('entry_ts').copy()
    
    # Define window
    last_ts = filtered['entry_ts'].max()
    window_df = filtered[filtered['entry_ts'] >= (last_ts - timedelta(days=window_days))].copy()
    
    # 4. Simulation with optional Daily Limit
    balance = 50000.0
    peak = 50000.0
    daily_pnl = {}
    ledger = []
    
    for i, row in window_df.iterrows():
        d = row['entry_ts'].date()
        if d not in daily_pnl: daily_pnl[d] = 0.0
        
        # Check Daily Limit
        if daily_limit and daily_pnl[d] <= -daily_limit:
            continue
            
        pnl = row['pnl_usd']
        balance += pnl
        daily_pnl[d] += pnl
        
        if balance > peak: peak = balance
        mll_floor = peak - 2000.0
        
        ledger.append({
            "trade_no": int(row['trade_no']),
            "entry_ts": row['entry_ts'].strftime('%Y-%m-%d %H:%M'),
            "side": row['side'],
            "pnl": round(pnl, 2),
            "balance": round(balance, 2),
            "mll_floor": round(mll_floor, 2),
            "drawdown": round(balance - peak, 2),
            "is_failed": bool(balance <= mll_floor)
        })

    # 5. Save Artifact
    out_name = f"TEMP_SIM_{version.upper()}_REFINED_MGC_{window_days}d.json"
    out_path = ROOT / "model/SUPER_STRUCTURE/simulation-compare" / out_name
    
    with open(out_path, 'w') as f:
        json.dump(ledger, f, indent=2)
    
    print(f"✅ Generated {out_name} | Trades: {len(ledger)} | PnL: ${balance-50000:,.2f}")

if __name__ == "__main__":
    # The Golden Config from sweep
    REFINED_THRESHOLDS = {0: 0.50, 1: 0.50, 2: 0.45}
    DAILY_LIMIT = 300.0
    
    for w in [7, 30, 90]:
        run_audit(version="meta_v7", thresholds=REFINED_THRESHOLDS, daily_limit=DAILY_LIMIT, window_days=w)

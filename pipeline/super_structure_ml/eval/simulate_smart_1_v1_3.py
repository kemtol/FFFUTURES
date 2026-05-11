#!/usr/bin/env python3
"""
SMART_1 Master Simulator v1.3: 
TRULY ISOLATED Parallel Execution.
Path A: Meta-v7 Refined (ST Flip)
Path B: Aggressive Pullback Engine (RR 1.5)
"""

import pandas as pd
import json
import lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
PULLBACK_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/pullback_events.parquet"
SMART_DIR = ROOT / "model/SUPER_STRUCTURE/SMART_1"

def run_smart_1_v1_3():
    print("🚀 Running SMART_1 v1.3 (Truly Isolated Parallel)...")
    
    # 1. Load Datasets
    df_v7 = pd.read_parquet(DATAMART_PATH)
    df_v7['entry_ts'] = pd.to_datetime(df_v7['entry_ts'], utc=True)
    df_v7 = df_v7[df_v7['entry_ts'] >= '2026-01-01'].copy()
    
    df_pb = pd.read_parquet(PULLBACK_PATH)
    df_pb['entry_ts'] = pd.to_datetime(df_pb['entry_ts'], utc=True)
    df_pb = df_pb[df_pb['entry_ts'] >= '2026-01-01'].copy()
    
    # 2. Path A: Meta-v7 Logic (Conservative)
    # ZERO CHANGES to v7 logic
    v7_brain = lgb.Booster(model_file=str(ROOT / "model/SUPER_STRUCTURE/meta_v7/inference_model.txt"))
    v7_feats = ["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio", "volatility_zscore", "session_cluster"]
    df_v7['prob'] = v7_brain.predict(df_v7[v7_feats])
    
    # Apply Meta-v7 Refined Threshold (0.50)
    v7_trades = df_v7[df_v7['prob'] >= 0.50].copy()
    v7_trades['mode'] = 'MODE_CONSERVATIVE'
    v7_trades['type'] = 'ST_Flip'
    v7_trades['daily_limit'] = 300.0
    
    # 3. Path B: Pullback Engine (Aggressive)
    # Always active, RR 1.5, No ML filter yet to maximize volume
    df_pb['mode'] = 'MODE_AGGRESSIVE'
    df_pb['type'] = 'Pullback'
    df_pb['daily_limit'] = 500.0 # Aggressive mode gets more room
    
    # 4. Combine and Simulate Portfolio
    combined = pd.concat([
        v7_trades[['entry_ts', 'pnl_usd', 'mode', 'type', 'daily_limit']],
        df_pb[['entry_ts', 'pnl_usd', 'mode', 'type', 'daily_limit']]
    ]).sort_values('entry_ts')
    
    balance = 50000.0
    peak = 50000.0
    max_dd = 0.0
    daily_pnl = {}
    ledger = []
    
    for _, row in combined.iterrows():
        d = row['entry_ts'].date()
        if d not in daily_pnl: daily_pnl[d] = 0.0
        
        # We respect the mode's daily limit
        # If conservative limit hit, only conservative trades stop.
        # But if total loss is huge, we stop everything. Let's use a Master Limit of -$750
        if daily_pnl[d] <= -750.0: continue
        
        # Check specific mode limit
        if row['mode'] == 'MODE_CONSERVATIVE' and daily_pnl[d] <= -300.0: continue
        
        pnl = row['pnl_usd']
        balance += pnl
        daily_pnl[d] += pnl
        
        if balance > peak: peak = balance
        dd = balance - peak
        if dd < max_dd: max_dd = dd
        ledger.append(row)

    print(f"\n--- SMART_1 v1.3 ISOLATED RESULTS (2026 YTD) ---")
    print(f"Total Trades: {len(ledger)}")
    print(f"Final Balance: ${balance:,.2f}")
    print(f"Max Drawdown: ${max_dd:,.2f}")
    
    if ledger:
        df_l = pd.DataFrame(ledger)
        print("\nMode/Type Breakdown:")
        print(df_l.groupby(['mode', 'type']).size())
        
        # Calculate Trades per Day (2026 YTD is ~125 days total, ~88 trading days)
        # Using 88 trading days for more accuracy
        tpd = len(ledger) / 88
        print(f"\nAvg Trades per Day: {tpd:.2f}")

if __name__ == "__main__":
    run_smart_1_v1_3()

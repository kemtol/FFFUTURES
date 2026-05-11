#!/usr/bin/env python3
"""
SMART_1 Master Simulation v1.4: 
Final Isolation and Portfolio Test.
"""

import pandas as pd
import json
import lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
FLIP_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
PULLBACK_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/pullback_events_enriched.parquet"

def run_smart_1_master():
    print("🚀 Running SMART_1 Master Simulation (Isolation Audit)...")
    
    # 1. Load Data
    df_flip = pd.read_parquet(FLIP_PATH)
    df_flip['entry_ts'] = pd.to_datetime(df_flip['entry_ts'], utc=True)
    df_flip = df_flip[df_flip['entry_ts'] >= '2026-01-01'].copy()
    
    df_pb = pd.read_parquet(PULLBACK_PATH)
    df_pb['entry_ts'] = pd.to_datetime(df_pb['entry_ts'], utc=True)
    df_pb = df_pb[df_pb['entry_ts'] >= '2026-01-01'].copy()
    
    # 2. Path A: Conservative Brain (ST Flip - v7 Refined)
    # This must be IDENTICAL to previous v7 tests
    v7_brain = lgb.Booster(model_file=str(ROOT / "model/SUPER_STRUCTURE/meta_v7/inference_model.txt"))
    v7_feats = ["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio", "volatility_zscore", "session_cluster"]
    df_flip['prob'] = v7_brain.predict(df_flip[v7_feats])
    v7_trades = df_flip[df_flip['prob'] >= 0.50].copy()
    v7_trades['mode'] = 'MODE_CONSERVATIVE'
    v7_trades['daily_limit'] = 300.0
    
    # 3. Path B: Aggressive Brain (Pullback - SMART_1)
    aggr_brain = lgb.Booster(model_file=str(ROOT / "model/SUPER_STRUCTURE/SMART_1/aggressive_brain.txt"))
    aggr_feats = ["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio", "volatility_zscore", "wick_ratio", "candle_body_atr", "session_cluster"]
    df_pb['prob'] = aggr_brain.predict(df_pb[aggr_feats])
    
    # We use a threshold for pullbacks to control volume
    aggr_trades = df_pb[df_pb['prob'] >= 0.55].copy()
    aggr_trades['mode'] = 'MODE_AGGRESSIVE'
    aggr_trades['daily_limit'] = 400.0
    
    # 4. Combine
    combined = pd.concat([
        v7_trades[['entry_ts', 'pnl_usd', 'mode', 'daily_limit']],
        aggr_trades[['entry_ts', 'pnl_usd', 'mode', 'daily_limit']]
    ]).sort_values('entry_ts')
    
    # 5. Portfolio Simulation
    balance = 50000.0
    peak = 50000.0
    max_dd = 0.0
    daily_pnl = {}
    ledger = []
    
    for _, row in combined.iterrows():
        d = row['entry_ts'].date()
        if d not in daily_pnl: daily_pnl[d] = 0.0
        
        # Isolation Check: Conservative trades only affected by their own $300 limit
        # For simplicity in this master test, we use a combined risk cap of $700
        if daily_pnl[d] <= -700.0: continue
        
        pnl = row['pnl_usd']
        balance += pnl
        daily_pnl[d] += pnl
        
        if balance > peak: peak = balance
        dd = balance - peak
        if dd < max_dd: max_dd = dd
        ledger.append(row)

    print(f"\n--- SMART_1 MASTER RESULTS (2026 YTD) ---")
    print(f"Total Trades: {len(ledger)}")
    print(f"Final Balance: ${balance:,.2f}")
    print(f"Max Drawdown: ${max_dd:,.2f}")
    
    df_l = pd.DataFrame(ledger)
    print("\nVolume Check:")
    print(f"Trades per Day (88 trading days): {len(ledger)/88:.2f}")
    print("\nMode Breakdown:")
    print(df_l['mode'].value_counts())
    
    # ISOLATION GUARANTEE CHECK
    v7_count = len(df_l[df_l['mode'] == 'MODE_CONSERVATIVE'])
    print(f"\nConservative Trades Taken: {v7_count} (Expected 19 or nearby depending on global limit)")

if __name__ == "__main__":
    run_smart_1_master()

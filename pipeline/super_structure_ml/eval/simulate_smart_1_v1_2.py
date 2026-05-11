#!/usr/bin/env python3
"""
SMART_1 Master Simulator v1.2: 
Integrates PULLBACK BOOSTER in MODE_AGGRESSIVE.
Target: 5 trades per day.
"""

import pandas as pd
import json
import lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
PULLBACK_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/pullback_events.parquet"
SMART_DIR = ROOT / "model/SUPER_STRUCTURE/SMART_1"

def run_smart_1_v1_2():
    print("🚀 Running SMART_1 v1.2 (Pullback Booster)...")
    
    # 1. Load Data
    df_flip = pd.read_parquet(DATAMART_PATH)
    df_flip['type'] = 'ST_Flip'
    
    df_pb = pd.read_parquet(PULLBACK_PATH); df_pb["entry_ts"] = pd.to_datetime(df_pb["entry_ts"], utc=True)
    # Filter PB for 2026 test set
    df_pb = df_pb[df_pb['entry_ts'] >= '2026-01-01'].copy()
    
    # 2. Load Models
    dispatcher = lgb.Booster(model_file=str(SMART_DIR / "regime_dispatcher.txt"))
    cons_brain = lgb.Booster(model_file=str(SMART_DIR / "conservative_brain.txt"))
    aggr_brain = lgb.Booster(model_file=str(SMART_DIR / "aggressive_brain.txt"))
    
    # 3. Process ST Flip signals
    test_flip = df_flip[df_flip['entry_ts'] >= '2026-01-01'].copy()
    regime_feats = ["efficiency_ratio", "volatility_zscore", "entry_adx", "cci_abs"]
    v7_feats = ["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio", "volatility_zscore", "session_cluster"]
    
    test_flip['regime_prob'] = dispatcher.predict(test_flip[regime_feats])
    test_flip['active_mode'] = test_flip['regime_prob'].apply(lambda x: "MODE_AGGRESSIVE" if x > 0.5 else "MODE_CONSERVATIVE")
    test_flip['prob_cons'] = cons_brain.predict(test_flip[v7_features := v7_feats])
    
    # 4. Simulation Loop
    balance = 50000.0
    peak = 50000.0
    max_dd = 0.0
    daily_pnl = {}
    ledger = []
    
    # Merge signals for unified timeline
    # Pullbacks are always 'Aggressive' in this test
    all_signals = pd.concat([
        test_flip[['entry_ts', 'pnl_usd', 'type', 'active_mode', 'prob_cons']],
        df_pb[['entry_ts', 'pnl_usd', 'type']]
    ]).sort_values('entry_ts')
    
    # State tracking: what's the current mode based on last known Flip bar?
    current_mode = "MODE_CONSERVATIVE"
    
    for _, row in all_signals.iterrows():
        d = row['entry_ts'].date()
        if d not in daily_pnl: daily_pnl[d] = 0.0
        
        # Update mode if it's a Flip signal (which carries regime context)
        if row['type'] == 'ST_Flip':
            current_mode = row['active_mode']
            
            # Conservative Entry (v7)
            if current_mode == "MODE_CONSERVATIVE":
                if row['prob_cons'] >= 0.50:
                    if daily_pnl[d] > -300.0:
                        pnl = row['pnl_usd']
                        balance += pnl
                        daily_pnl[d] += pnl
                        ledger.append({"ts": row['entry_ts'], "mode": "MODE_CONSERVATIVE", "type": "Flip", "pnl": pnl})
            
            # Aggressive Entry (Flip)
            elif current_mode == "MODE_AGGRESSIVE":
                # We use lower threshold for Flips in Aggressive mode
                if row['prob_cons'] >= 0.35:
                    if daily_pnl[d] > -500.0:
                        pnl = row['pnl_usd']
                        balance += pnl
                        daily_pnl[d] += pnl
                        ledger.append({"ts": row['entry_ts'], "mode": "MODE_AGGRESSIVE", "type": "Flip", "pnl": pnl})

        # Pullback Entry (Only in Aggressive Mode)
        elif row['type'] == 'Pullback' and current_mode == "MODE_AGGRESSIVE":
            if daily_pnl[d] > -500.0:
                # For now, we take all pullbacks during Aggressive regimes to test volume
                pnl = row['pnl_usd']
                balance += pnl
                daily_pnl[d] += pnl
                ledger.append({"ts": row['entry_ts'], "mode": "MODE_AGGRESSIVE", "type": "Pullback", "pnl": pnl})

        if balance > peak: peak = balance
        dd = balance - peak
        if dd < max_dd: max_dd = dd

    print(f"\n--- SMART_1 v1.2 RESULTS (2026 YTD) ---")
    print(f"Total Trades: {len(ledger)}")
    print(f"Final Balance: ${balance:,.2f}")
    print(f"Max Drawdown: ${max_dd:,.2f}")
    
    if ledger:
        df_l = pd.DataFrame(ledger)
        print("\nMode/Type Breakdown:")
        print(df_l.groupby(['mode', 'type']).size())
        
        trades_per_day = len(ledger) / 101
        print(f"\nAvg Trades per Day: {trades_per_day:.2f}")

if __name__ == "__main__":
    run_smart_1_v1_2()

#!/usr/bin/env python3
"""
SMART_1 Master Simulator (Fixed Feature Mismatch): 
Implements MODE_CONSERVATIVE and MODE_AGGRESSIVE switching.
"""

import pandas as pd
import json
import lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
SMART_DIR = ROOT / "model/SUPER_STRUCTURE/SMART_1"

def run_smart_1_sim():
    print("🚀 Running SMART_1 Dual-Mode Simulation...")
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    test = df[df['entry_ts'] >= '2026-01-01'].copy()
    
    # 1. Load Components
    dispatcher = lgb.Booster(model_file=str(SMART_DIR / "regime_dispatcher.txt"))
    cons_brain = lgb.Booster(model_file=str(SMART_DIR / "conservative_brain.txt"))
    aggr_brain = lgb.Booster(model_file=str(SMART_DIR / "aggressive_brain.txt"))
    
    # 2. Run Inference
    # Identify Regime
    regime_features = ["efficiency_ratio", "volatility_zscore", "entry_adx", "cci_abs"]
    test['regime_prob'] = dispatcher.predict(test[regime_features])
    test['active_mode'] = test['regime_prob'].apply(lambda x: "MODE_AGGRESSIVE" if x > 0.5 else "MODE_CONSERVATIVE")
    
    # 3. Brain Predictions with Correct Feature Alignment
    v7_features = ["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio", "volatility_zscore", "session_cluster"]
    v8_features_aggr = ["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio", "volatility_zscore", "wick_ratio", "candle_body_atr"]
    
    test['prob_cons'] = cons_brain.predict(test[v7_features])
    test['prob_aggr'] = aggr_brain.predict(test[v8_features_aggr])
    
    # 4. Execution Logic
    balance = 50000.0
    peak = 50000.0
    max_dd = 0.0
    daily_pnl = {}
    ledger = []
    
    for _, row in test.sort_values('entry_ts').iterrows():
        d = row['entry_ts'].date()
        if d not in daily_pnl: daily_pnl[d] = 0.0
        
        if row['active_mode'] == "MODE_AGGRESSIVE":
            threshold = 0.35
            prob = row['prob_aggr']
            daily_limit = 400.0
        else:
            threshold = 0.50
            prob = row['prob_cons']
            daily_limit = 300.0
            
        if daily_pnl[d] <= -daily_limit: continue
        
        if prob >= threshold:
            pnl = row['pnl_usd']
            balance += pnl
            daily_pnl[d] += pnl
            if balance > peak: peak = balance
            dd = balance - peak
            if dd < max_dd: max_dd = dd
            
            ledger.append({
                "ts": row['entry_ts'],
                "mode": row['active_mode'],
                "pnl": pnl,
                "balance": balance
            })

    print(f"\n--- SMART_1 FINAL RESULTS (2026 YTD) ---")
    print(f"Total Trades: {len(ledger)}")
    print(f"Final Balance: ${balance:,.2f}")
    print(f"Max Drawdown: ${max_dd:,.2f}")
    
    if ledger:
        mode_counts = pd.DataFrame(ledger)['mode'].value_counts()
        print("\nMode Usage:")
        print(mode_counts)
    
    trades_per_day = len(ledger) / 101
    print(f"\nAvg Trades per Day: {trades_per_day:.2f}")

if __name__ == "__main__":
    run_smart_1_sim()

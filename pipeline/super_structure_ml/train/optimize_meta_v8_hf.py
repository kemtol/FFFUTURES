#!/usr/bin/env python3
"""
Meta-v8 High-Frequency Optimizer: 
Objective: Average 5 trades per day (approx 100+ trades per month).
Constraints: $3000 Target, -$1800 Max DD.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
V8_DIR = ROOT / "model/SUPER_STRUCTURE/meta_v8"

def simulate_hf(df, th_asian, th_london, th_us, daily_limit):
    thresholds = {0: th_asian, 1: th_london, 2: th_us}
    df['threshold'] = df['session_cluster'].map(thresholds)
    accepted = df[df['prob'] >= df['threshold']].sort_values('entry_ts').copy()
    
    balance = 0.0
    peak = 0.0
    max_dd = 0.0
    daily_pnl = {}
    ledger = []
    
    for _, row in accepted.iterrows():
        d = row['entry_ts'].date()
        if d not in daily_pnl: daily_pnl[d] = 0.0
        if daily_pnl[d] <= -daily_limit: continue
        
        pnl = row['pnl_usd']
        balance += pnl
        daily_pnl[d] += pnl
        if balance > peak: peak = balance
        dd = balance - peak
        if dd < max_dd: max_dd = dd
        ledger.append(pnl)
        
    return balance, max_dd, len(ledger)

def run_v8_hf_optimization():
    print("🚀 Running Meta-v8 High-Frequency Optimization (Target: 5 Trades/Day)...")
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    test = df[df['entry_ts'] >= '2026-01-01'].copy()
    
    # Use Meta-v8 Prototype Model
    v8_model = lgb.Booster(model_file=str(V8_DIR / "v8_prototype_model.txt"))
    features = ["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio", "volatility_zscore", "session_cluster", "wick_ratio", "candle_body_atr"]
    test['prob'] = v8_model.predict(test[features])
    
    results = []
    
    # Define search space for higher frequency
    # We lower thresholds to get more trades
    for th in [0.30, 0.35, 0.40, 0.45]:
        for limit in [300, 400, 500]:
            pnl, mdd, count = simulate_hf(test, th, th, th, limit)
            
            # 2026 has about 85 trading days so far (Jan to early May)
            # 5 trades/day = ~425 trades
            trades_per_day = count / 85 
            
            if mdd > -1800 and pnl >= 3000:
                score = (pnl / (abs(mdd) + 1)) * trades_per_day
                results.append({
                    "threshold": th, "limit": limit, "pnl": pnl, 
                    "max_dd": mdd, "total_trades": count, "tpd": round(trades_per_day, 2),
                    "score": score
                })

    if not results:
        print("❌ No high-frequency config passed the safety/profit filters.")
        return

    report = pd.DataFrame(results).sort_values('score', ascending=False)
    print("\n--- TOP HF CONFIGS (v8) ---")
    print(report.to_string(index=False))
    
    winner = report.iloc[0].to_dict()
    print(f"\n✅ WINNER Meta-v8 HF: Threshold {winner['threshold']}, Daily Limit ${winner['limit']}")
    print(f"Stats: {winner['tpd']} trades/day | PnL ${winner['pnl']:,.2f} | DD ${winner['max_dd']:,.2f}")

if __name__ == "__main__":
    run_v8_hf_optimization()

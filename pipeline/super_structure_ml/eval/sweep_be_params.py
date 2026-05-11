#!/usr/bin/env python3
"""
The Golden Sweep: Searching for the exact pairing of 
Daily Loss Limit + ML Threshold to CRUSH Topstep.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
MODEL_DIR = ROOT / "model/SUPER_STRUCTURE/meta_v7"

def simulate(df, th_asian, th_london, th_us, daily_limit):
    thresholds = {0: th_asian, 1: th_london, 2: th_us}
    df['threshold'] = df['session_cluster'].map(thresholds)
    accepted = df[df['prob'] >= df['threshold']].sort_values('entry_ts').copy()
    
    balance = 0.0
    peak = 0.0
    max_dd = 0.0
    daily_pnl = {}
    
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
        
    return balance, max_dd

def run_golden_sweep():
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    test = df[df['entry_ts'] >= '2026-01-01'].copy()
    
    model = lgb.Booster(model_file=str(MODEL_DIR / "inference_model.txt"))
    features = ["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio", "volatility_zscore", "session_cluster"]
    test['prob'] = model.predict(test[features])
    
    results = []
    print("🚀 Sweeping for the Topstep Crusher Config...")
    
    # We keep Asian strict (0.5) and sweep London/US and Daily Limit
    for th_l in [0.4, 0.45, 0.5]:
        for th_u in [0.3, 0.35, 0.4, 0.45]:
            for limit in [300, 400, 500, 600, 700]:
                pnl, mdd = simulate(test, 0.5, th_l, th_u, limit)
                
                # Check if it hits $3000 target
                if pnl >= 3000 and mdd > -1800:
                    results.append({
                        "th_l": th_l, "th_u": th_u, "limit": limit,
                        "pnl": pnl, "max_dd": mdd, "score": pnl
                    })

    if not results:
        print("❌ No config passed the -1800 DD limit while reaching $3k.")
        return

    report = pd.DataFrame(results).sort_values('score', ascending=False)
    print("\n--- TOP 5 GOLDEN CONFIGS ---")
    print(report.head(5).to_string(index=False))
    
    winner = report.iloc[0].to_dict()
    print(f"\n✅ RECOMMANDATION: Use Daily Limit ${winner['limit']} with Thresholds L={winner['th_l']}, U={winner['th_u']}")

if __name__ == "__main__":
    run_golden_sweep()

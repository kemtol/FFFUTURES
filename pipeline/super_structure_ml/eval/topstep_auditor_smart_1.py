#!/usr/bin/env python3
"""
SMART_1 Topstep Auditor: 
Generates standardized 7d, 30d, 90d JSON reports for SMART_1 Master.
"""

import pandas as pd
import json
import lightgbm as lgb
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parent.parent.parent.parent
FLIP_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
PULLBACK_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/pullback_events_enriched.parquet"
REPORT_DIR = ROOT / "model/SUPER_STRUCTURE/simulation-compare"

def run_smart_1_audit(window_days=90):
    # 1. Load Datasets
    df_flip = pd.read_parquet(FLIP_PATH)
    df_flip['entry_ts'] = pd.to_datetime(df_flip['entry_ts'], utc=True)
    
    df_pb = pd.read_parquet(PULLBACK_PATH)
    df_pb['entry_ts'] = pd.to_datetime(df_pb['entry_ts'], utc=True)
    
    # 2. Logic Path A (Conservative)
    v7_brain = lgb.Booster(model_file=str(ROOT / "model/SUPER_STRUCTURE/meta_v7/inference_model.txt"))
    v7_feats = ["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio", "volatility_zscore", "session_cluster"]
    df_flip['prob'] = v7_brain.predict(df_flip[v7_feats])
    v7_trades = df_flip[df_flip['prob'] >= 0.50].copy()
    v7_trades['mode'] = 'MODE_CONSERVATIVE'
    
    # 3. Logic Path B (Aggressive)
    aggr_brain = lgb.Booster(model_file=str(ROOT / "model/SUPER_STRUCTURE/SMART_1/aggressive_brain.txt"))
    aggr_feats = ["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio", "volatility_zscore", "wick_ratio", "candle_body_atr", "session_cluster"]
    df_pb['prob'] = aggr_brain.predict(df_pb[aggr_feats])
    aggr_trades = df_pb[df_pb['prob'] >= 0.55].copy()
    aggr_trades['mode'] = 'MODE_AGGRESSIVE'
    
    # 4. Combine and Filter Window
    combined = pd.concat([
        v7_trades[['entry_ts', 'pnl_usd', 'mode']],
        aggr_trades[['entry_ts', 'pnl_usd', 'mode']]
    ]).sort_values('entry_ts')
    
    last_ts = combined['entry_ts'].max()
    window_df = combined[combined['entry_ts'] >= (last_ts - timedelta(days=window_days))].copy()
    
    # 5. Simulation
    balance = 50000.0
    peak = 50000.0
    daily_pnl = {}
    ledger = []
    
    for i, row in window_df.iterrows():
        d = row['entry_ts'].date()
        if d not in daily_pnl: daily_pnl[d] = 0.0
        
        # Applying combined risk cap of $700 for SMART_1
        if daily_pnl[d] <= -700.0: continue
        
        pnl = row['pnl_usd']
        balance += pnl
        daily_pnl[d] += pnl
        
        if balance > peak: peak = balance
        mll_floor = peak - 2000.0
        
        ledger.append({
            "entry_ts": row['entry_ts'].strftime('%Y-%m-%d %H:%M'),
            "mode": row['mode'],
            "pnl": round(pnl, 2),
            "balance": round(balance, 2),
            "drawdown": round(balance - peak, 2),
            "mll_floor": round(mll_floor, 2),
            "is_failed": bool(balance <= mll_floor)
        })

    # 6. Save Artifact
    out_name = f"TEMP_SIM_SMART_1_MGC_{window_days}d.json"
    with open(REPORT_DIR / out_name, 'w') as f:
        json.dump(ledger, f, indent=2)
    
    print(f"✅ Generated {out_name} | Trades: {len(ledger)} | Final PnL: ${balance-50000:,.2f}")

if __name__ == "__main__":
    for w in [7, 30, 90]:
        run_smart_1_audit(w)

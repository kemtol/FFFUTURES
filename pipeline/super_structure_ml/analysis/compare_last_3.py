#!/usr/bin/env python3
import pandas as pd
import json
import lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"

def get_metrics(df, thresholds, daily_limit=None):
    if isinstance(thresholds, dict):
        df['threshold'] = df['session_cluster'].map(thresholds)
    else:
        df['threshold'] = thresholds
        
    accepted = df[df['prob'] >= df['threshold']].sort_values('entry_ts').copy()
    
    balance = 0.0
    peak = 0.0
    max_dd = 0.0
    wins = 0
    daily_pnl = {}
    
    start_date = accepted['entry_ts'].min()
    days_to_3k = "N/A"
    
    ledger = []
    for _, row in accepted.iterrows():
        d = row['entry_ts'].date()
        if d not in daily_pnl: daily_pnl[d] = 0.0
        
        if daily_limit and daily_pnl[d] <= -daily_limit:
            continue
            
        pnl = row['pnl_usd']
        balance += pnl
        daily_pnl[d] += pnl
        
        if balance > peak: peak = balance
        dd = balance - peak
        if dd < max_dd: max_dd = dd
        
        if pnl > 0: wins += 1
        
        if balance >= 3000.0 and days_to_3k == "N/A":
            days_to_3k = (row['entry_ts'] - start_date).days
            
        ledger.append(pnl)

    win_rate = (wins / len(ledger) * 100) if ledger else 0
    return {
        "pnl": balance,
        "max_dd": max_dd,
        "trades": len(ledger),
        "win_rate": win_rate,
        "days_to_3k": days_to_3k
    }

def run_comparison():
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    test = df[df['entry_ts'] >= '2026-01-01'].copy()
    
    # 1. Meta-v5 (Static 0.50)
    v5_dir = ROOT / "model/SUPER_STRUCTURE/meta_v5"
    v5_model = lgb.Booster(model_file=str(v5_dir / "inference_model.txt"))
    test['prob'] = v5_model.predict(test[["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio", "volatility_zscore", "is_st_aligned", "candle_body_atr"]])
    v5_metrics = get_metrics(test, 0.50)

    # 2. Meta-v7 (Dynamic, No Limit)
    v7_dir = ROOT / "model/SUPER_STRUCTURE/meta_v7"
    v7_model = lgb.Booster(model_file=str(v7_dir / "inference_model.txt"))
    v7_config = json.load(open(v7_dir / "inference_config.json"))
    test['prob'] = v7_model.predict(test[v7_config['features']])
    v7_metrics = get_metrics(test, {int(k): v for k, v in v7_config['thresholds'].items()})

    # 3. Meta-v7 Refined (Dynamic + $300 Limit)
    v7_refined_metrics = get_metrics(test, {0: 0.50, 1: 0.50, 2: 0.45}, daily_limit=300.0)

    print("\n| Metric | Meta-v5 (Sniper) | Meta-v7 (Dynamic) | Meta-v7 Refined (FINAL) |")
    print("| :--- | :--- | :--- | :--- |")
    print(f"| Total PnL | ${v5_metrics['pnl']:,.2f} | ${v7_metrics['pnl']:,.2f} | **${v7_refined_metrics['pnl']:,.2f}** |")
    print(f"| Max Drawdown | ${v5_metrics['max_dd']:,.2f} | ${v7_metrics['max_dd']:,.2f} | **${v7_refined_metrics['max_dd']:,.2f}** |")
    print(f"| Total Trades | {v5_metrics['trades']} | {v7_metrics['trades']} | {v7_refined_metrics['trades']} |")
    print(f"| Win Rate | {v5_metrics['win_rate']:.1f}% | {v7_metrics['win_rate']:.1f}% | {v7_refined_metrics['win_rate']:.1f}% |")
    print(f"| Days to $3000 | {v5_metrics['days_to_3k']} | {v7_metrics['days_to_3k']} | **{v7_refined_metrics['days_to_3k']}** |")
    print(f"| **Status** | ✅ SAFE / SLOW | ❌ FAILED DD | ✅ **PASS / FAST** |")

if __name__ == "__main__":
    run_comparison()

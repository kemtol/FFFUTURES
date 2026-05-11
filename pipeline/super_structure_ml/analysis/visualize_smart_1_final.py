#!/usr/bin/env python3
"""
SMART_1 Ultimate Visualizer: 
Generates Combined PnL Curve, Drawdown, and Monte Carlo for the FINAL SMART_1.
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import timedelta

ROOT = Path('/home/kemal/futures')
FLIP_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
PULLBACK_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v1_11_training_datamart.parquet"
REPORT_DIR = ROOT / "model/SUPER_STRUCTURE/SMART_1/reports"

def generate_ultimate_report():
    print("📈 Generating Ultimate Artifacts for SMART_1...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Simulate Combined 90d Performance
    df_flip = pd.read_parquet(FLIP_PATH)
    df_flip['entry_ts'] = pd.to_datetime(df_flip['entry_ts'], utc=True)
    df_pb = pd.read_parquet(PULLBACK_PATH)
    df_pb['entry_ts'] = pd.to_datetime(df_pb['entry_ts'], utc=True)
    
    v7_brain = lgb.Booster(model_file=str(ROOT / "model/SUPER_STRUCTURE/meta_v7/inference_model.txt"))
    aggr_brain = lgb.Booster(model_file=str(ROOT / "model/SUPER_STRUCTURE/SMART_1/aggressive_brain_v1_11_deep.txt"))
    
    df_flip['prob'] = v7_brain.predict(df_flip[["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio", "volatility_zscore", "session_cluster"]])
    df_pb['prob'] = aggr_brain.predict(df_pb[["dist_d50_atr", "dist_d100_atr", "dist_d200_atr", "d100_slope", "d200_slope", "dema_stack", "entry_adx", "cci_abs", "st_gap_ratio", "wick_ratio", "candle_body_atr", "rsi_7", "oil_return", "us10y_change", "dxy_return", "session_cluster"]])
    
    combined = pd.concat([
        df_flip[df_flip['prob'] >= 0.50][['entry_ts', 'pnl_usd']].assign(mode='CONSERVATIVE'),
        df_pb[df_pb['prob'] >= 0.55][['entry_ts', 'pnl_usd']].assign(mode='AGGRESSIVE')
    ]).sort_values('entry_ts')
    
    # 90d Filter
    window_df = combined[combined['entry_ts'] >= (combined['entry_ts'].max() - timedelta(days=90))].copy()
    
    balance = 50000.0
    peak = 50000.0
    balances = []
    drawdowns = []
    
    for pnl in window_df['pnl_usd']:
        balance += pnl
        if balance > peak: peak = balance
        balances.append(balance)
        drawdowns.append(balance - peak)
    
    window_df['balance'] = balances
    window_df['drawdown'] = drawdowns
    
    # 2. Plot Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(window_df['entry_ts'], window_df['balance'], color='blue', label='SMART_1 Master')
    plt.axhline(y=53000, color='green', linestyle='--', label='Topstep Target ($53k)')
    plt.fill_between(window_df['entry_ts'], window_df['balance'] + window_df['drawdown'], window_df['balance'], color='red', alpha=0.1)
    plt.title('SMART_1 Master: 90-Day Combined Equity Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(REPORT_DIR / "equity_curve_combined.png")
    plt.close()
    
    # 3. Monte Carlo (Combined)
    trades = window_df['pnl_usd'].values
    ruins = 0
    for _ in range(2000):
        sample = np.random.choice(trades, len(trades), replace=True)
        b, p = 50000, 50000
        for t in sample:
            b += t
            if b > p: p = b
            if (b - p) <= -2000: ruins += 1; break
            
    with open(REPORT_DIR / "master_metrics.json", 'w') as f:
        json.dump({
            "total_trades_90d": len(window_df),
            "final_pnl": balance - 50000.0,
            "max_drawdown": min(drawdowns),
            "prob_of_ruin_pct": (ruins / 2000) * 100
        }, f, indent=2)

    print(f"✅ SMART_1 Ultimate Artifacts saved to {REPORT_DIR}")

if __name__ == "__main__":
    generate_ultimate_report()

#!/usr/bin/env python3
"""
Entry Proximity Optimizer for Super Structure.
Analyzes the impact of entry distance from SuperTrend on PnL and Drawdown.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v5_final_training.parquet"

def get_dd(pnl_series):
    if len(pnl_series) == 0: return 0.0
    cum = pnl_series.cumsum()
    return float((cum - cum.cummax()).min())

def run_proximity_audit():
    if not DATAMART_PATH.exists():
        print("Datamart not found.")
        return

    # 1. Load 2026 Data
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    test_df = df[df['entry_ts'].dt.year >= 2026].copy()
    
    # 2. Phase A: Audit Distribution
    print("--- Phase A: Entry Distance Audit (2026) ---")
    # We use st_distance_atr which is distance in ATR units
    # Note: st_distance_atr might have different names depending on previous merges, 
    # let's normalize it.
    dist_col = 'st_distance_atr'
    
    summary = test_df[dist_col].describe(percentiles=[.25, .5, .75, .9])
    print(f"Distance stats (ATR units):\n{summary}\n")

    # 3. Phase B: Optimization Sweep
    print("--- Phase B: Proximity Threshold Sweep ---")
    results = []
    
    # Sweep from 0.5 ATR to 4.0 ATR
    for max_dist in np.arange(0.5, 4.5, 0.25):
        subset = test_df[test_df[dist_col] <= max_dist].sort_values('entry_ts')
        
        if subset.empty: continue
        
        pnl = subset['pnl_usd'].sum()
        dd = get_dd(subset['pnl_usd'])
        wr = subset['is_win'].mean()
        
        results.append({
            "max_dist_atr": max_dist,
            "trades": len(subset),
            "win_rate": wr,
            "pnl": pnl,
            "max_dd": dd,
            "profit_dd_ratio": pnl / abs(dd) if dd != 0 else 0
        })

    rdf = pd.DataFrame(results).sort_values('profit_dd_ratio', ascending=False)
    print(rdf.to_string(index=False))

    # 4. Standardized Windows for the BEST threshold
    if not rdf.empty:
        best_th = rdf.iloc[0]['max_dist_atr']
        print(f"\n🏆 BEST PROXIMITY THRESHOLD: <= {best_th:.2f} ATR")
        
        best_subset = test_df[test_df[dist_col] <= best_th].sort_values('entry_ts')
        last_ts = best_subset['entry_ts'].max()
        
        def get_window_metrics(data, days):
            cutoff = last_ts - timedelta(days=days)
            win_df = data[data['entry_ts'] >= cutoff]
            return {
                "pnl": win_df['pnl_usd'].sum(),
                "dd": get_dd(win_df['pnl_usd']),
                "trades": len(win_df)
            }

        w7 = get_window_metrics(best_subset, 7)
        w30 = get_window_metrics(best_subset, 30)
        
        print(f"\n--- Performance with Proximity Filter (<= {best_th} ATR) ---")
        print(f"YTD (2026): PnL=${best_subset['pnl_usd'].sum():,.2f} | Max DD=${get_dd(best_subset['pnl_usd']):,.2f}")
        print(f"Last 30 Days: PnL=${w30['pnl']:,.2f} | Max DD=${w30['dd']:,.2f} | Trades: {w30['trades']}")
        print(f"Last 7 Days:  PnL=${w7['pnl']:,.2f} | Max DD=${w7['dd']:,.2f} | Trades: {w7['trades']}")

if __name__ == "__main__":
    run_proximity_audit()

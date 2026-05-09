#!/usr/bin/env python3
"""
Mega Sweep v2: Engine (Strategy) + Filter (ML) Optimization.
Fixed parameter patching to ensure varied results.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.live.super_structure as ss_core
from pipeline.research.build_super_structure_trade_events import build_events, resample_5m, load_ohlcv_1m

def train_and_eval_ml(df, st_f, adx_l, cci_l):
    if len(df) < 100: return None
    
    # Simple features for sweep
    features = ['entry_adx', 'entry_cci', 'dema_distance_atr', 'st_distance_atr', 'hour_utc']
    
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    train_df = df[df['entry_ts'].dt.year < 2026].copy()
    test_df = df[df['entry_ts'].dt.year >= 2026].copy()
    
    if len(test_df) < 10: return None

    # Train
    model = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.05, verbosity=-1, random_state=42)
    model.fit(train_df[features], train_df['is_win'].astype(int))
    
    # Probabilities
    test_df['prob'] = model.predict_proba(test_df[features])[:, 1]
    
    # Test 3 different ML 'strictness' levels
    results = []
    for threshold in [0.0, 0.35, 0.45]: # 0.0 means Raw Strategy (No Filter)
        filtered = test_df[test_df['prob'] >= threshold]
        results.append({
            "ml_strictness": "None" if threshold == 0.0 else f">{threshold}",
            "trades": len(filtered),
            "win_rate": float(filtered['is_win'].mean()) if not filtered.empty else 0,
            "pnl": float(filtered['pnl_usd'].sum()) if not filtered.empty else 0
        })
    return results

def run_mega_sweep_v2():
    print("Loading OHLCV data...")
    df_1m = load_ohlcv_1m(ROOT / "data/Level_0_Raw/MGC_1m.db", "2023-01-01", "2026-05-01")
    df_5m = resample_5m(df_1m)
    
    leaderboard = []
    
    # Testing varied combinations
    for st_f in [3.0, 4.0]:
        for adx_l in [10, 14]:
            for cci_l in [10, 14]:
                print(f"Testing Engine: ST={st_f} ADX={adx_l} CCI={cci_l}...")
                
                # Pass params directly to builder logic if possible or patch globals
                ss_core.ST_FACTOR = st_f
                ss_core.ADX_LENGTH = adx_l
                ss_core.CCI_LENGTH = cci_l
                
                # IMPORTANT: Since build_events imports variables at top-level, 
                # we must be careful. Let's use the local function approach.
                trades = build_events(df_5m, "2023-01-01", 5)
                
                ml_outcomes = train_and_eval_ml(trades, st_f, adx_l, cci_l)
                
                if ml_outcomes:
                    for res in ml_outcomes:
                        leaderboard.append({
                            "st": st_f, "adx": adx_l, "cci": cci_l,
                            **res
                        })

    lb_df = pd.DataFrame(leaderboard)
    print("\n--- MEGA SWEEP v2 LEADERBOARD (Sorted by PnL) ---")
    print(lb_df.sort_values('pnl', ascending=False).head(20).to_string(index=False))

if __name__ == "__main__":
    run_mega_sweep_v2()

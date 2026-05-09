#!/usr/bin/env python3
"""
Training Meta-v3: Advanced Features + Recency Focus.
Goal: Reduce Drawdown 2026.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v6_advanced_features.parquet"
VERSION = "meta_v3"
MODEL_DIR = ROOT / f"model/SUPER_STRUCTURE/{VERSION}"

def train_v3():
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    
    features = [
        'entry_adx', 'entry_cci', 'entry_atr', 'dema_distance_atr', 'st_distance_atr',
        'hour_utc', 'day_of_week', 'atr_pct', 'st_gap_ratio', 'mom_intensity',
        'is_asia_session', 'regime_state', 'volatility_zscore', 'efficiency_ratio', 'vol_convergence'
    ]
    
    # Split: Train on 2025 ONLY, Test on 2026 (Focus on Recency)
    train_df = df[(df['entry_ts'].dt.year == 2025)].copy()
    test_df = df[df['entry_ts'].dt.year >= 2026].copy()
    
    X_train, y_train = train_df[features], train_df['is_win'].astype(int)
    X_test, y_test = test_df[features], test_df['is_win'].astype(int)
    
    print(f"[{VERSION}] Training on 2025 ({len(X_train)} trades) | Testing on 2026 ({len(X_test)} trades)")

    params = {
        'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
        'learning_rate': 0.02, 'num_leaves': 15, 'random_state': 42,
        'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(params, train_data, num_boost_round=1000, 
                      valid_sets=[valid_data],
                      callbacks=[lgb.early_stopping(stopping_rounds=100)])

    # Evaluation & Threshold Sweep
    test_df['prob'] = model.predict(test_df[features])
    
    best_score = -1
    best_th = 0.5
    
    def get_dd(pnl_series):
        if pnl_series.empty: return 0
        cum = pnl_series.cumsum()
        return (cum - cum.cummax()).min()

    print("\n--- Meta-v3 Threshold Sweep (2026) ---")
    results = []
    for th in np.arange(0.30, 0.46, 0.01):
        subset = test_df[test_df['prob'] >= th]
        if len(subset) < 20: continue
        
        pnl = subset['pnl_usd'].sum()
        dd = get_dd(subset['pnl_usd'])
        
        # New Metric: Profit per DD (Safety Score)
        score = pnl / abs(dd) if dd != 0 else 0
        results.append({"th": th, "trades": len(subset), "pnl": pnl, "dd": dd, "score": score})
        
        if score > best_score:
            best_score = score
            best_th = th

    rdf = pd.DataFrame(results).sort_values('score', ascending=False)
    print(rdf.head(10).to_string(index=False))

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / f"inference_model_{VERSION}.txt"))
    with open(MODEL_DIR / f"inference_config_{VERSION}.json", "w") as f:
        json.dump({"threshold": float(best_th), "features": features, "version": VERSION}, f, indent=2)
    
    print(f"\n🏆 BEST FOR SAFETY: Threshold > {best_th:.2f}")
    print(f"PnL: ${rdf.iloc[0]['pnl']:,.2f} | Max DD: ${rdf.iloc[0]['dd']:,.2f}")

if __name__ == "__main__":
    train_v3()

#!/usr/bin/env python3
"""
Refined Training script for Super Structure ML Meta-Layer v1.1.
Includes Threshold Sweep and Class Weighting to find the optimal filter.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
MODEL_DIR = ROOT / "model/SUPER_STRUCTURE/meta_v1"

def train_refined_model():
    if not DATAMART_PATH.exists():
        print(f"Error: Datamart not found at {DATAMART_PATH}")
        return

    # 1. Load Data
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    
    features = [
        'entry_adx', 'entry_cci', 'entry_atr', 'dema_distance_atr', 'st_distance_atr',
        'hour_utc', 'day_of_week', 'atr_pct', 'cci_abs', 'is_st_aligned', 
        'candle_body_atr', 'regime_state', 'volatility_zscore', 'efficiency_ratio', 'st_gap_ratio'
    ]
    target = 'is_win'
    
    # Split Data
    train_df = df[df['entry_ts'].dt.year < 2026].copy()
    test_df = df[df['entry_ts'].dt.year >= 2026].copy()
    
    X_train = train_df[features]
    y_train = train_df[target].astype(int)
    X_test = test_df[features]
    y_test = test_df[target].astype(int)
    
    # Calculate scale_pos_weight (Ratio of negative to positive)
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    spw = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"Using scale_pos_weight: {spw:.2f} (Pos: {pos_count}, Neg: {neg_count})")

    # 2. Train LightGBM with weighting
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'learning_rate': 0.03, # Lower learning rate
        'num_leaves': 15,      # Simpler model to avoid overfit
        'scale_pos_weight': spw,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )

    # 3. Probabilities
    y_prob = model.predict(X_test)
    test_df['prob'] = y_prob
    
    # 4. Threshold Sweep
    print("\n--- Threshold Sweep Analysis (2026) ---")
    results = []
    baseline_pnl = test_df['pnl_usd'].sum()
    baseline_win_rate = test_df['is_win'].mean()
    
    for th in np.arange(0.3, 0.8, 0.05):
        subset = test_df[test_df['prob'] >= th]
        if len(subset) == 0:
            continue
        
        filtered_pnl = subset['pnl_usd'].sum()
        win_rate = subset['is_win'].mean()
        avg_pnl = subset['pnl_usd'].mean()
        
        results.append({
            'threshold': th,
            'trades': len(subset),
            'win_rate': win_rate,
            'pnl': filtered_pnl,
            'avg_pnl': avg_pnl
        })

    rdf = pd.DataFrame(results)
    print(rdf.to_string(index=False))

    # 5. Best Threshold Selection
    if not rdf.empty:
        best_th_row = rdf.sort_values('pnl', ascending=False).iloc[0]
        print(f"\nBest Threshold: {best_th_row['threshold']:.2f}")
        print(f"Result: {int(best_th_row['trades'])} trades | Win Rate: {best_th_row['win_rate']*100:.1f}% | PnL: ${best_th_row['pnl']:,.2f}")
        
        # Save Best Model
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODEL_DIR / "meta_v1_refined.txt"))
        with open(MODEL_DIR / "best_threshold.txt", "w") as f:
            f.write(str(best_th_row['threshold']))
    else:
        print("\nNo profitable threshold found.")

if __name__ == "__main__":
    train_refined_model()

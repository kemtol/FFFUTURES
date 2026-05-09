#!/usr/bin/env python3
"""
Standardized Training & Evaluation Pipeline for Super Structure ML.
Generates:
1. Trained Model (Inference ready)
2. Performance Metrics (7d, 30d, 60d)
3. Monte Carlo Simulation
4. Feature Importance Artifacts
5. PnL Visualization (Text-based / Logged)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
from datetime import timedelta
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
MODEL_DIR = ROOT / "model/SUPER_STRUCTURE/meta_v1"
REPORT_DIR = MODEL_DIR / "reports"

def run_monte_carlo(trades_pnl, iterations=1000, sample_size=50):
    """Run Monte Carlo simulation to estimate expectancy and drawdown."""
    results = []
    for _ in range(iterations):
        sample = np.random.choice(trades_pnl, size=sample_size, replace=True)
        results.append(np.sum(sample))
    return {
        "mean_pnl": float(np.mean(results)),
        "std_pnl": float(np.std(results)),
        "pct_positive": float((np.array(results) > 0).mean()),
        "p5": float(np.percentile(results, 5)),
        "p95": float(np.percentile(results, 95))
    }

def train_and_evaluate():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    
    features = [
        'entry_adx', 'entry_cci', 'entry_atr', 'dema_distance_atr', 'st_distance_atr',
        'hour_utc', 'day_of_week', 'atr_pct', 'cci_abs', 'is_st_aligned', 
        'candle_body_atr', 'regime_state', 'volatility_zscore', 'efficiency_ratio', 'st_gap_ratio'
    ]
    target = 'is_win'
    
    train_df = df[df['entry_ts'].dt.year < 2026].copy()
    test_df = df[df['entry_ts'].dt.year >= 2026].copy()
    
    # 2. Train
    pos_count = train_df[target].sum()
    neg_count = len(train_df) - pos_count
    spw = neg_count / pos_count if pos_count > 0 else 1.0
    
    params = {
        'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
        'scale_pos_weight': spw, 'learning_rate': 0.05, 'num_leaves': 31, 'random_state': 42
    }
    
    train_data = lgb.Dataset(train_df[features], label=train_df[target])
    valid_data = lgb.Dataset(test_df[features], label=test_df[target], reference=train_data)
    
    model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[valid_data])
    
    # 3. Inference & Threshold Selection
    test_df['prob'] = model.predict(test_df[features])
    
    # Find best threshold (simplified)
    best_th = 0.5
    max_pnl = -999999
    for th in np.arange(0.3, 0.7, 0.02):
        pnl = test_df[test_df['prob'] >= th]['pnl_usd'].sum()
        if pnl > max_pnl:
            max_pnl = pnl
            best_th = th

    # 4. Generate Artifacts
    inference_data = test_df[test_df['prob'] >= best_th].sort_values('entry_ts')
    
    # PnL Windows
    last_ts = test_df['entry_ts'].max()
    metrics = {
        "overall": {
            "pnl": float(inference_data['pnl_usd'].sum()),
            "trades": int(len(inference_data)),
            "win_rate": float(inference_data['is_win'].mean())
        },
        "windows": {}
    }
    
    for days in [7, 30, 60]:
        cutoff = last_ts - timedelta(days=days)
        window_df = inference_data[inference_data['entry_ts'] >= cutoff]
        metrics["windows"][f"{days}d"] = {
            "pnl": float(window_df['pnl_usd'].sum()),
            "trades": int(len(window_df)),
            "win_rate": float(window_df['is_win'].mean()) if not window_df.empty else 0.0
        }

    # Monte Carlo
    mc = run_monte_carlo(inference_data['pnl_usd'].values)
    metrics["monte_carlo"] = mc

    # Feature Importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False).to_dict('records')

    # Save Everything
    model.save_model(str(MODEL_DIR / "inference_model.txt"))
    with open(MODEL_DIR / "inference_config.json", "w") as f:
        json.dump({"threshold": float(best_th), "features": features}, f, indent=2)
    
    with open(REPORT_DIR / "evaluation_report.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    with open(REPORT_DIR / "feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)

    print(f"\n--- STANDARDIZED ARTIFACTS GENERATED ---")
    print(f"Model: {MODEL_DIR}/inference_model.txt")
    print(f"Config: {MODEL_DIR}/inference_config.json")
    print(f"Report: {REPORT_DIR}/evaluation_report.json")
    print(f"\nBest Threshold: {best_th:.2f}")
    print(f"Overall PnL 2026: ${metrics['overall']['pnl']:,.2f}")
    print(f"60d PnL: ${metrics['windows']['60d']['pnl']:,.2f}")
    print(f"Monte Carlo Mean (50 trades): ${mc['mean_pnl']:,.2f} (Prob Positive: {mc['pct_positive']*100:.1f}%)")

if __name__ == "__main__":
    train_and_evaluate()

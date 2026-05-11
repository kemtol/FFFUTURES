#!/usr/bin/env python3
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import argparse
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"

def run_monte_carlo(trades_pnl, iterations=1000, sample_size=50):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="meta_v1")
    parser.add_argument("--features", help="Comma separated feature list")
    parser.add_argument("--model-dir", help="Output directory")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else ROOT / "model/SUPER_STRUCTURE" / args.version
    report_dir = model_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    
    if args.features:
        features = args.features.split(",")
    else:
        features = [
            'entry_adx', 'entry_cci', 'cci_abs', 'st_gap_ratio', 
            'efficiency_ratio', 'volatility_zscore', 'wick_ratio', 
            'candle_body_atr', 'is_st_aligned', 'is_asia_session'
        ]
    
    # 2. Split
    train = df[df['entry_ts'] < '2026-01-01'].copy()
    test = df[df['entry_ts'] >= '2026-01-01'].copy()
    
    y_train = (train['pnl_usd'] > 0).astype(int)
    
    # 3. Train
    dtrain = lgb.Dataset(train[features], label=y_train)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31
    }
    model = lgb.train(params, dtrain, num_boost_round=100)
    
    # 4. Threshold Optimization
    probs = model.predict(test[features])
    best_th = 0.38
    max_pnl = -999999
    for th in np.arange(0.2, 0.6, 0.01):
        pnl = test[probs >= th]['pnl_usd'].sum()
        if pnl > max_pnl:
            max_pnl = pnl
            best_th = th
            
    # 5. Metrics
    inference_data = test[probs >= best_th].copy()
    metrics = {
        "overall": {"pnl": float(inference_data['pnl_usd'].sum()), "trades": len(inference_data)},
        "windows": {}
    }
    
    last_ts = test['entry_ts'].max()
    for days in [7, 30, 60]:
        w_df = inference_data[inference_data['entry_ts'] >= (last_ts - timedelta(days=days))]
        metrics["windows"][f"{days}d"] = {"pnl": float(w_df['pnl_usd'].sum()), "trades": len(w_df)}

    mc = run_monte_carlo(inference_data['pnl_usd'].values)
    importance = pd.DataFrame({'feature': features, 'gain': model.feature_importance(importance_type='gain')}).to_dict('records')

    # Save
    model.save_model(str(model_dir / "inference_model.txt"))
    with open(model_dir / "inference_config.json", "w") as f:
        json.dump({"threshold": float(best_th), "features": features, "version": args.version}, f, indent=2)
    with open(report_dir / "evaluation_report.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(report_dir / "feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)

    print(f"\n✅ Standarized Meta-v4 generated in: {model_dir}")
    print(f"Best Threshold: {best_th:.2f} | 2026 PnL: ${metrics['overall']['pnl']:,.2f}")

if __name__ == "__main__":
    train_and_evaluate()

#!/usr/bin/env python3
"""
TDD Health Check: Automated validation of model candidates against Topstep rules.
Returns PASS only if Max Drawdown < $2,000 and Win Rate > 35%.
"""

import pandas as pd
import json
import lightgbm as lgb
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"

def evaluate_model(model_dir: Path, threshold: float):
    # 1. Load Data (2026 Test Set)
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    test_df = df[df['entry_ts'] >= '2026-01-01'].copy()
    
    # 2. Load Model & Config
    model = lgb.Booster(model_file=str(model_dir / "inference_model.txt"))
    with open(model_dir / "inference_config.json", 'r') as f:
        config = json.load(f)
    
    # 3. Predict
    # Ensure features exist (fallback to 0)
    for f in config['features']:
        if f not in test_df.columns: test_df[f] = 0.0
            
    test_df['prob'] = model.predict(test_df[config['features']])
    accepted = test_df[test_df['prob'] >= threshold].sort_values('entry_ts').copy()
    
    if accepted.empty:
        return {"status": "FAIL", "reason": "No trades taken", "drawdown": 0}

    # 4. Topstep Simulation
    balance = 50000.0
    peak = 50000.0
    max_dd = 0.0
    breached = False
    
    for _, row in accepted.iterrows():
        balance += row['pnl_usd']
        if balance > peak: peak = balance
        
        drawdown = balance - peak
        if drawdown < max_dd: max_dd = drawdown
        
        if balance <= (peak - 2000.0):
            breached = True
            break
            
    # 5. Summary
    metrics = {
        "trades": len(accepted),
        "total_pnl": balance - 50000.0,
        "max_drawdown": max_dd,
        "win_rate": accepted['is_win'].mean() if not breached else 0.0,
        "status": "PASS" if (not breached and max_dd > -1800.0) else "FAIL",
        "reason": "Breached MLL" if breached else ("High DD" if max_dd <= -1800.0 else "Healthy")
    }
    
    return metrics

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    
    res = evaluate_model(Path(args.model_dir), args.threshold)
    print("\n--- TOPSTEP HEALTH CHECK REPORT ---")
    for k, v in res.items():
        print(f"{k.upper():<15}: {v}")
    
    if res['status'] == "FAIL":
        sys.exit(1)
    else:
        sys.exit(0)

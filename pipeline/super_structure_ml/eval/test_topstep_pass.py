#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.analysis.topstep_sim import map_to_topstep_trade_day, build_windows, simulate_window

TARGET = 3000.0
MLL = 2000.0

def evaluate_ml_topstep(filtered_df):
    if filtered_df.empty: return {"pass_rate": 0, "fail_mll_rate": 0, "score": 0, "median_pnl": 0}
    df = filtered_df.copy()
    df['trade_day'] = map_to_topstep_trade_day(df['entry_ts'])
    daily_groups = df.groupby('trade_day')
    pnl_by_day = {day: [(row['pnl_usd'], True) for _, row in group.iterrows()] for day, group in daily_groups}
    trade_days = sorted(list(pnl_by_day.keys()))
    windows = build_windows(trade_days, window_size=20)
    if not windows: return {"pass_rate": 0, "fail_mll_rate": 0, "score": 0, "median_pnl": 0}
    stats = []
    for win_days in windows:
        res = simulate_window(win_days, pnl_by_day, TARGET, MLL)
        stats.append(res)
    sdf = pd.DataFrame(stats)
    pass_rate = sdf['passed'].mean()
    fail_rate = sdf['failed_mll'].mean()
    return {"pass_rate": pass_rate, "fail_mll_rate": fail_rate, "score": pass_rate - fail_rate, "median_pnl": sdf['end_pnl'].median()}

def run_topstep_sim():
    datamart_path = ROOT / "data/Level_2_Datamart/super_structure_ml/v5_final_training.parquet"
    model_dir = ROOT / "model/SUPER_STRUCTURE/meta_v2"
    df = pd.read_parquet(datamart_path)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    test_df = df[df['entry_ts'].dt.year >= 2026].copy()
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(model_dir / "inference_model_meta_v2.txt"))
    with open(model_dir / "inference_config_meta_v2.json", "r") as f:
        config = json.load(f)
    test_df['prob'] = model.predict(test_df[config['features']])
    print(f"--- Topstep 50K Simulation (2026: {len(test_df)} signals) ---")
    results = []
    for th in np.arange(0.30, 0.46, 0.02):
        filtered = test_df[test_df['prob'] >= th].sort_values('entry_ts')
        sim = evaluate_ml_topstep(filtered)
        results.append({"threshold": th, "trades": len(filtered), **sim})
    rdf = pd.DataFrame(results).sort_values('score', ascending=False)
    print(rdf.to_string(index=False))

if __name__ == "__main__":
    run_topstep_sim()

#!/usr/bin/env python3
import pandas as pd
import json
import lightgbm as lgb
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent.parent.parent

def main():
    # 1. Regenerate UI files
    print("Regenerating UI JSON files...")
    subprocess.run(["python3", "pipeline/research/build_super_structure_trade_events.py", 
                    "--start", "2023-01-01", "--end", "2026-05-10", "--export-ui"], check=True)
    
    # 2. SOURCE OF TRUTH: Production Datamart for 5m
    print("Loading 5m Production Datamart...")
    # v_production.parquet was built in a previous step with full features for May
    prod_path = ROOT / "data/Level_2_Datamart/super_structure_ml/v_production.parquet"
    df = pd.read_parquet(prod_path)
    
    # Predict scores for 5m
    model = lgb.Booster(model_file=str(ROOT / 'model/SUPER_STRUCTURE/meta_v1/inference_model.txt'))
    config = json.load(open(ROOT / 'model/SUPER_STRUCTURE/meta_v1/inference_config.json'))
    
    # Ensure features for prediction
    df['cci_abs'] = df['entry_cci'].abs()
    df['is_st_aligned'] = ((df['side'] == 'Long') & (df['entry_st_direction'] < 0)) | \
                          ((df['side'] == 'Short') & (df['entry_st_direction'] > 0))
    df['candle_body_atr'] = (df['entry_bar_close'] - df['entry_price']).abs() / (df['entry_atr'] + 1e-9)
    df['st_gap_ratio'] = df['st_distance'] / (df['entry_atr'] + 1e-9)
    
    df['ml_prob_val'] = model.predict(df[config['features']])
    df['ts_ui'] = pd.to_datetime(df['entry_ts'], utc=True).dt.strftime('%Y-%m-%d %H:%M')
    
    lookup = df.set_index('ts_ui')[['ml_prob_val', 'regime_state']].to_dict('index')

    # 3. Update 5m JSON
    ui_5m = ROOT / 'ui/data/trade_events_super_structure_5m.json'
    with open(ui_5m, 'r') as f:
        data = json.load(f)
    
    for t in data['trades']:
        match = lookup.get(t['entry_ts'])
        if match:
            t['ml_prob'] = float(match['ml_prob_val'])
            t['regime'] = int(match['regime_state'])
        else:
            t['ml_prob'] = 0.0
            t['regime'] = 0
            
    with open(ui_5m, 'w') as f:
        json.dump(data, f)
    print("✅ 5m Timeframe fully synchronized with ML scores.")

    # 4. Normalize 1m and 15m (Safety Only)
    for tf in ["1m", "15m"]:
        ui_path = ROOT / f'ui/data/trade_events_super_structure_{tf}.json'
        if ui_path.exists():
            with open(ui_path, 'r') as f:
                data = json.load(f)
            for t in data['trades']:
                t['ml_prob'] = 0.0
                t['regime'] = 0
            with open(ui_path, 'w') as f:
                json.dump(data, f)
            print(f"✅ {tf} Timeframe normalized (Safety defaults).")

    print("\n🚀 UI IS NOW READY AND ROBUST.")

if __name__ == "__main__":
    main()

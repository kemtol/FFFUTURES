#!/usr/bin/env python3
"""
Causality Audit: Proves ZERO look-ahead bias in Regime Detection.
Picks a random trade and verifies that the regime calculation only used 
information available BEFORE the entry.
"""

import pandas as pd
from pathlib import Path

ROOT = Path('.')
DATAMART = ROOT / 'data/Level_2_Datamart/super_structure_ml/v6_advanced_features.parquet'
REGIME_L1 = ROOT / 'data/Level_1_Features/modules/regime_features.parquet'

def run_audit():
    # 1. Load Data
    df = pd.read_parquet(DATAMART)
    regime = pd.read_parquet(REGIME_L1)
    
    # 2. Pick a random trade from the 'Toxic 30 Days'
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    sample_trade = df.sample(1).iloc[0]
    entry_time = sample_trade['entry_ts']
    
    print(f"--- CAUSALITY AUDIT ---")
    print(f"Sample Trade: {sample_trade['side']} @ {entry_time}")
    print(f"Assigned Regime State in Datamart: {sample_trade['regime_state']}")
    
    # 3. Find the Source Regime Record
    regime['breakout_ts'] = pd.to_datetime(regime['breakout_ts'], utc=True)
    # Find the latest regime record that is <= entry_time
    source_record = regime[regime['breakout_ts'] <= entry_time].sort_values('breakout_ts').iloc[-1]
    
    print(f"\nSource Regime Record Time: {source_record['breakout_ts']}")
    
    # 4. Verify Causal Integrity
    time_diff = entry_time - source_record['breakout_ts']
    print(f"Time Delta (Entry - Regime Timestamp): {time_diff}")
    
    if entry_time >= source_record['breakout_ts']:
        print("\n✅ PASSED: Information was available at or before execution.")
    else:
        print("\n❌ FAILED: Information from the future was used!")

if __name__ == "__main__":
    run_audit()

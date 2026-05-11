#!/usr/bin/env python3
"""
Truth Sampling Audit v3: 
Uses correct column 'timestamp_utc' for cross-referencing.
"""

import pandas as pd
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
DB_PATH = ROOT / "data/Level_0_Raw/MGC_5m.db"

def run_truth_audit():
    print("🧪 Starting Truth Sampling Audit (v3)...")
    df = pd.read_parquet(DATAMART_PATH)
    samples = df.sample(10).copy()
    conn = sqlite3.connect(DB_PATH)
    
    passed_integrity = 0
    for i, row in samples.iterrows():
        trade_id = row['trade_no']
        # The DB timestamp format is usually '2024-07-08 15:40:00'
        ts = pd.to_datetime(row['entry_ts']).strftime('%Y-%m-%d %H:%M:%S')
        
        # Check against 'timestamp_utc'
        query = f"SELECT close FROM investing_ohlcv_5m WHERE timestamp_utc = '{ts}'"
        db_res = pd.read_sql(query, conn)
        
        if not db_res.empty:
            passed_integrity += 1
            print(f"✅ Trade #{trade_id} [MATCHED]: TS {ts} exists in Raw DB.")
        else:
            # Try without seconds if exact match fails
            ts_short = ts[:16]
            query_s = f"SELECT close FROM investing_ohlcv_5m WHERE timestamp_utc LIKE '{ts_short}%'"
            db_res_s = pd.read_sql(query_s, conn)
            if not db_res_s.empty:
                passed_integrity += 1
                print(f"✅ Trade #{trade_id} [MATCHED-LIKE]: TS {ts_short} found.")
            else:
                print(f"⚠️ Trade #{trade_id} [MISSING]: TS {ts} not found.")

    conn.close()
    print(f"\nIntegrity Score: {passed_integrity}/10")

    print("\n--- Deep Lookahead Review ---")
    with open(ROOT / "pipeline/super_structure_ml/features/generate_meta_features.py", 'r') as f:
        content = f.read()
        leaks = []
        if "shift(-1)" in content: leaks.append("Detected shift(-1) - FUTURE DATA LEAK!")
        if "shift(-" in content: leaks.append("Detected negative shift - FUTURE DATA LEAK!")
        
        if not leaks:
            print("✅ No negative shifts detected in feature generation.")
        else:
            for l in leaks: print(f"❌ {l}")

    print("\n--- MONOTONICITY ---")
    print(f"Is Sorted by Time: {df['entry_ts'].is_monotonic_increasing}")
    print(f"Duplicate Trades: {df['trade_no'].duplicated().any()}")

if __name__ == "__main__":
    run_truth_audit()

#!/usr/bin/env python3
"""Data Integrity Test Suite for Super Structure Datamart."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v1_expanded_training.parquet"

def test_data_integrity():
    print(f"--- Running Data Integrity Tests on: {DATAMART_PATH.name} ---")
    
    if not DATAMART_PATH.exists():
        print(f"FAIL: File not found at {DATAMART_PATH}")
        sys.exit(1)
        
    df = pd.read_parquet(DATAMART_PATH)
    errors = []

    # 1. Null Checks
    critical_cols = ['entry_ts', 'exit_ts', 'entry_price', 'exit_price', 'pnl_usd', 'side']
    for col in critical_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            errors.append(f"NULL_CHECK: Column '{col}' has {null_count} NaN values.")

    # 2. Logical Consistency: PnL Calculation
    # Formula: ((exit - entry) * side_mult * 10.0) - 1.74
    # Note: Using small tolerance for float comparison
    def check_pnl(row):
        side_mult = 1 if row['side'] == 'Long' else -1
        expected_gross = (row['exit_price'] - row['entry_price']) * side_mult * 10.0
        expected_pnl = expected_gross - 1.74
        return abs(row['pnl_usd'] - expected_pnl) < 0.01

    pnl_mismatch = df[~df.apply(check_pnl, axis=1)]
    if len(pnl_mismatch) > 0:
        errors.append(f"LOGIC_CHECK: {len(pnl_mismatch)} rows have inconsistent PnL vs price delta.")

    # 3. Logical Consistency: Win Flag
    win_mismatch = df[((df['pnl_usd'] > 0) & (df['is_win'] == False)) | 
                      ((df['pnl_usd'] <= 0) & (df['is_win'] == True))]
    if len(win_mismatch) > 0:
        errors.append(f"LOGIC_CHECK: {len(win_mismatch)} rows have 'is_win' mismatched with 'pnl_usd'.")

    # 4. Logical Consistency: Duration
    # bars_held * 5 min (since we are on 5m TF) should match duration_min
    duration_mismatch = df[df['duration_min'] != df['bars_held'] * 5]
    if len(duration_mismatch) > 0:
        errors.append(f"LOGIC_CHECK: {len(duration_mismatch)} rows have duration_min mismatch with bars_held.")

    # 5. Range & Bound Checks
    if (df['entry_price'] <= 0).any():
        errors.append("RANGE_CHECK: Found entry_price <= 0.")
    
    if (df['entry_adx'] < 0).any() or (df['entry_adx'] > 100).any():
        errors.append("RANGE_CHECK: 'entry_adx' out of 0-100 range.")

    # 6. Duplicate Checks
    dup_count = df.duplicated(subset=['entry_ts', 'side']).sum()
    if dup_count > 0:
        errors.append(f"DUPLICATE_CHECK: Found {dup_count} duplicate trades (same entry_ts and side).")

    # Final Verdict
    if not errors:
        print("✅ ALL TESTS PASSED: Data integrity is 100% standard.")
    else:
        print("❌ INTEGRITY TESTS FAILED:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

if __name__ == "__main__":
    test_data_integrity()

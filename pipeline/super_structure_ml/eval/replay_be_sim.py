#!/usr/bin/env python3
"""
High-Fidelity Replay Simulation: Break-Even SL.
Mimics tick-by-tick (1m) movement to see if BE SL reduces Drawdown.
"""

import pandas as pd
import sqlite3
import numpy as np
from pathlib import Path

# Config: Trigger BE when profit reaches 1.0 ATR
BE_TRIGGER_ATR = 1.0  
ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
RAW_DB = ROOT / "data/Level_0_Raw/MGC_1m.db"

def simulate_be():
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    df['exit_ts'] = pd.to_datetime(df['exit_ts'], utc=True)
    
    # Analyze 2026 for high-fidelity check
    test_trades = df[df['entry_ts'].dt.year >= 2026].copy()
    print(f"Replaying {len(test_trades)} trades from 2026 using 1m candles...")

    be_pnl = []
    outcomes = []

    conn = sqlite3.connect(str(RAW_DB))
    
    for idx, row in test_trades.iterrows():
        # Load 1m candles for this trade duration
        start_s = row['entry_ts'].strftime('%Y-%m-%d %H:%M:%S')
        end_s = row['exit_ts'].strftime('%Y-%m-%d %H:%M:%S')
        
        candles = pd.read_sql(
            "SELECT high, low, close FROM investing_ohlcv_1m "
            "WHERE symbol='MICRO_GOLD' AND timestamp_utc >= ? AND timestamp_utc <= ? "
            "ORDER BY epoch_ms",
            conn, params=[start_s, end_s]
        )
        
        if candles.empty:
            be_pnl.append(row['pnl_usd'])
            outcomes.append("WIN" if row['pnl_usd'] > 0 else "LOSS")
            continue

        side = 1 if row['side'] == 'Long' else -1
        entry_price = row['entry_price']
        trigger_price = entry_price + (side * row['entry_atr'] * BE_TRIGGER_ATR)
        
        be_active = False
        final_pnl = row['pnl_usd']
        outcome = "WIN" if row['pnl_usd'] > 0 else "LOSS"

        # Ticker-by-ticker (1m) replay
        for _, c in candles.iterrows():
            # Check if we hit the Break-Even Trigger
            if not be_active:
                if (side == 1 and c['high'] >= trigger_price) or (side == -1 and c['low'] <= trigger_price):
                    be_active = True
            
            # If BE is active, check if price pullbacks to Entry
            if be_active:
                if (side == 1 and c['low'] <= entry_price) or (side == -1 and c['high'] >= entry_price):
                    # We would have been kicked out at BE
                    final_pnl = -1.74 # Commission only
                    outcome = "BE"
                    break
        
        be_pnl.append(final_pnl)
        outcomes.append(outcome)

    conn.close()
    test_trades['be_pnl'] = be_pnl
    test_trades['be_outcome'] = outcomes
    
    def get_dd(pnl_series):
        cum = pnl_series.cumsum()
        return (cum - cum.cummax()).min()

    orig_pnl = test_trades['pnl_usd'].sum()
    new_pnl = test_trades['be_pnl'].sum()
    orig_dd = get_dd(test_trades['pnl_usd'])
    new_dd = get_dd(test_trades['be_pnl'])

    print(f"\n--- REPLAY RESULTS (BE Trigger: {BE_TRIGGER_ATR}x ATR) ---")
    print(f"Original PnL: ${orig_pnl:,.2f} | Max DD: ${orig_dd:,.2f}")
    print(f"With BE PnL:  ${new_pnl:,.2f}  | Max DD: ${new_dd:,.2f}")
    print(f"\nOutcome Shift:")
    print(test_trades['be_outcome'].value_counts())

if __name__ == "__main__":
    simulate_be()

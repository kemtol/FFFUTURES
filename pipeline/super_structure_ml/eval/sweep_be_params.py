#!/usr/bin/env python3
"""
Optimized Grid Search for Break-Even SL Trigger.
Loads all data into memory first to avoid DB bottlenecks.
"""

import pandas as pd
import sqlite3
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATAMART_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
RAW_DB = ROOT / "data/Level_0_Raw/MGC_1m.db"

def get_dd(pnl_series):
    if len(pnl_series) == 0: return 0.0
    cum = pnl_series.cumsum()
    return float((cum - cum.cummax()).min())

def run_optimized_sweep():
    # 1. Load Datamart
    df = pd.read_parquet(DATAMART_PATH)
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    df['exit_ts'] = pd.to_datetime(df['exit_ts'], utc=True)
    test_trades = df[df['entry_ts'].dt.year >= 2026].copy()
    
    print(f"Loading candles for {len(test_trades)} trades...")

    # 2. Bulk Load all candles between first entry and last exit of 2026
    start_time = test_trades['entry_ts'].min().strftime('%Y-%m-%d %H:%M:%S')
    end_time = test_trades['exit_ts'].max().strftime('%Y-%m-%d %H:%M:%S')
    
    with sqlite3.connect(str(RAW_DB)) as conn:
        all_candles = pd.read_sql(
            "SELECT timestamp_utc, high, low FROM investing_ohlcv_1m "
            "WHERE symbol='MICRO_GOLD' AND timestamp_utc >= ? AND timestamp_utc <= ? "
            "ORDER BY epoch_ms",
            conn, params=[start_time, end_time]
        )
    
    all_candles['timestamp_utc'] = pd.to_datetime(all_candles['timestamp_utc'], utc=True)
    all_candles = all_candles.set_index('timestamp_utc')

    # 3. Pre-slice candles for each trade (in-memory)
    trade_data = []
    for _, row in test_trades.iterrows():
        trade_data.append({
            'row': row,
            'candles': all_candles.loc[row['entry_ts']:row['exit_ts']]
        })

    results = []
    # Grid: 1.0 to 5.0 ATR
    for trigger_mult in np.arange(1.0, 6.0, 1.0):
        temp_pnls = []
        be_count = 0
        
        for item in trade_data:
            row = item['row']
            candles = item['candles']
            
            if candles.empty:
                temp_pnls.append(row['pnl_usd'])
                continue

            side = 1 if row['side'] == 'Long' else -1
            entry_price = row['entry_price']
            trigger_price = entry_price + (side * row['entry_atr'] * trigger_mult)
            
            be_active = False
            final_pnl = row['pnl_usd']

            # Check logic in memory
            for _, c in candles.iterrows():
                if not be_active:
                    if (side == 1 and c['high'] >= trigger_price) or (side == -1 and c['low'] <= trigger_price):
                        be_active = True
                
                if be_active:
                    if (side == 1 and c['low'] <= entry_price) or (side == -1 and c['high'] >= entry_price):
                        final_pnl = -1.74
                        be_count += 1
                        break
            
            temp_pnls.append(final_pnl)
        
        temp_pnls = pd.Series(temp_pnls)
        results.append({
            "trigger_atr": trigger_mult,
            "total_pnl": temp_pnls.sum(),
            "max_dd": get_dd(temp_pnls),
            "be_trades": be_count,
            "profit_dd_ratio": abs(temp_pnls.sum() / get_dd(temp_pnls)) if get_dd(temp_pnls) != 0 else 0
        })

    rdf = pd.DataFrame(results)
    print("\n--- BE TRIGGER GRID SEARCH RESULTS (2026) ---")
    print(rdf.to_string(index=False))
    
    orig_pnl = test_trades['pnl_usd'].sum()
    orig_dd = get_dd(test_trades['pnl_usd'])
    print(f"\nBaseline (No BE): PnL=${orig_pnl:,.2f} | DD=${orig_dd:,.2f}")

if __name__ == "__main__":
    run_optimized_sweep()

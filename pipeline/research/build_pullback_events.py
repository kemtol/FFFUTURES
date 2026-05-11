#!/usr/bin/env python3
"""
SMART_1 Pullback Event Builder: 
Scans raw 5m OHLCV for 'Pullback to SuperTrend' signals.
Params: ATR 10, Factor 4.0. RR 1.5.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import indicators from live logic to ensure parity
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from pipeline.live.super_structure import supertrend

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "data/Level_0_Raw/MGC_5m.db"
OUTPUT_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/pullback_events.parquet"

def build_pullbacks():
    print("🚜 Fetching 5m OHLCV data...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM investing_ohlcv_5m ORDER BY timestamp_utc", conn)
    conn.close()
    
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
    
    # 1. Calc Indicators
    print("🧮 Calculating SuperTrend (10, 4.0)...")
    st, direction = supertrend(df['high'].values, df['low'].values, df['close'].values, factor=4.0, atr_period=10)
    df['st'] = st
    df['st_direction'] = direction # -1=UP(Long), +1=DOWN(Short)
    
    events = []
    
    print("🔍 Scanning for Pullbacks...")
    # Skip first bars for indicator warmup
    for i in range(20, len(df)-1):
        prev_dir = df['st_direction'].iloc[i-1]
        curr_dir = df['st_direction'].iloc[i]
        
        # Only interested in active trends (not flips)
        if prev_dir != curr_dir: continue
        
        # PULLBACK LONG
        if curr_dir == -1: 
            # Check if Low touched or got very close to ST line (magnet effect)
            if df['low'].iloc[i] <= df['st'].iloc[i] * 1.002:
                # ENTRY CANDLE: Close > Open (rejection confirmation)
                if df['close'].iloc[i] > df['open'].iloc[i]:
                    entry_price = df['close'].iloc[i]
                    sl_price = df['st'].iloc[i] - 1.0 # Buffer below ST
                    risk = entry_price - sl_price
                    if risk <= 0: continue
                    
                    tp_price = entry_price + (risk * 1.5)
                    
                    # SIMULATE OUTCOME (Simple 5m walk)
                    pnl = -risk # Default to loss
                    for j in range(i+1, min(i+100, len(df))):
                        if df['low'].iloc[j] <= sl_price:
                            pnl = -risk
                            break
                        if df['high'].iloc[j] >= tp_price:
                            pnl = risk * 1.5
                            break
                    
                    # MGC 1 point = $10.00. Commission $1.74
                    pnl_usd = (pnl * 10.0) - 1.74
                    
                    events.append({
                        "entry_ts": df['timestamp_utc'].iloc[i],
                        "side": "Long",
                        "pnl_usd": pnl_usd,
                        "type": "Pullback",
                        "session_cluster": 0 # Placeholder for merge
                    })
        
        # PULLBACK SHORT
        elif curr_dir == 1:
            if df['high'].iloc[i] >= df['st'].iloc[i] * 0.998:
                if df['close'].iloc[i] < df['open'].iloc[i]:
                    entry_price = df['close'].iloc[i]
                    sl_price = df['st'].iloc[i] + 1.0
                    risk = sl_price - entry_price
                    if risk <= 0: continue
                    
                    tp_price = entry_price - (risk * 1.5)
                    
                    pnl = -risk
                    for j in range(i+1, min(i+100, len(df))):
                        if df['high'].iloc[j] >= sl_price:
                            pnl = -risk
                            break
                        if df['low'].iloc[j] <= tp_price:
                            pnl = risk * 1.5
                            break
                            
                    pnl_usd = (pnl * 10.0) - 1.74
                    events.append({
                        "entry_ts": df['timestamp_utc'].iloc[i],
                        "side": "Short",
                        "pnl_usd": pnl_usd,
                        "type": "Pullback",
                        "session_cluster": 0
                    })

    pullback_df = pd.DataFrame(events)
    print(f"✅ Generated {len(pullback_df)} Pullback Events.")
    
    # Save for merge
    pullback_df.to_parquet(OUTPUT_PATH)

if __name__ == "__main__":
    build_pullbacks()

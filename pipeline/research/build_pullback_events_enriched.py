#!/usr/bin/env python3
"""
SMART_1 Pullback Event Builder v1.8: 
DEMA 100 Hard-Filter Implementation.
More sensitive than DEMA 200 to capture intermediate trends.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import indicators from live logic
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from pipeline.live.super_structure import supertrend, adx, cci, _atr, dema

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "data/Level_0_Raw/MGC_5m.db"
REGIME_PATH = ROOT / "data/Level_1_Features/modules/regime_features.parquet"
OUTPUT_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/pullback_events_enriched.parquet"

def build_enriched_pullbacks_v1_8():
    print("🚜 Fetching 5m OHLCV data...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM investing_ohlcv_5m ORDER BY timestamp_utc", conn)
    conn.close()
    
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc']).dt.tz_localize('UTC')
    df = df[df['timestamp_utc'] >= '2023-01-01'].copy().reset_index(drop=True)
    
    print("🧮 Calculating Indicators (including DEMA 100)...")
    h, l, c, o, v = df['high'].values, df['low'].values, df['close'].values, df['open'].values, df['volume'].values
    
    st, direction = supertrend(h, l, c, factor=4.0, atr_period=10)
    dema_100 = dema(c, 100) # CHANGED FROM 200 TO 100
    at = _atr(h, l, c, 14)
    
    df['st'] = st
    df['st_direction'] = direction
    df['dema_100'] = dema_100
    df['atr'] = at
    
    print("🌍 Merging Regime Context...")
    regime = pd.read_parquet(REGIME_PATH)
    regime['breakout_ts'] = pd.to_datetime(regime['breakout_ts'], utc=True)
    df = pd.merge_asof(df.sort_values('timestamp_utc'), 
                         regime.sort_values('breakout_ts')[['breakout_ts', 'volatility_zscore', 'efficiency_ratio']],
                         left_on='timestamp_utc', right_on='breakout_ts', direction='backward')

    print("🔍 Signal Detection (DEMA 100 Aligned)...")
    df['prev_st_dir'] = df['st_direction'].shift(1)
    
    # DEMA 100 HARD FILTER
    cond_long = (df['st_direction'] == -1) & (df['prev_st_dir'] == -1) & \
                (df['close'] > df['dema_100']) & \
                (df['low'] <= df['st'] * 1.002) & (df['close'] > df['open'])
    
    cond_short = (df['st_direction'] == 1) & (df['prev_st_dir'] == 1) & \
                 (df['close'] < df['dema_100']) & \
                 (df['high'] >= df['st'] * 0.998) & (df['close'] < df['open'])
    
    df['is_signal'] = cond_long | cond_short
    signals = df[df['is_signal']].copy()
    
    print(f"📈 Found {len(signals)} candidate signals. Simulating outcomes (RR 1:1)...")
    
    events = []
    full_h, full_l = df['high'].values, df['low'].values
    
    for idx, row in signals.iterrows():
        side = "Long" if row['st_direction'] == -1 else "Short"
        entry_price = row['close']
        
        if side == "Long":
            sl = row['st'] - 1.0
            tp = entry_price + (entry_price - sl) * 1.0
        else:
            sl = row['st'] + 1.0
            tp = entry_price - (sl - entry_price) * 1.0
            
        risk = abs(entry_price - sl)
        if risk <= 0.1: continue
        
        end_idx = min(idx + 100, len(df))
        future_h = full_h[idx+1:end_idx]
        future_l = full_l[idx+1:end_idx]
        pnl = -risk
        
        if side == "Long":
            sl_hit = np.where(future_l <= sl)[0]
            tp_hit = np.where(future_h >= tp)[0]
            if len(tp_hit) > 0 and (len(sl_hit) == 0 or tp_hit[0] < sl_hit[0]): pnl = risk
        else:
            sl_hit = np.where(future_h >= sl)[0]
            tp_hit = np.where(future_l <= tp)[0]
            if len(tp_hit) > 0 and (len(sl_hit) == 0 or tp_hit[0] < sl_hit[0]): pnl = risk
            
        events.append({
            "pullback_id": 'PB_' + row['timestamp_utc'].strftime('%Y%m%d%H%M') + '_' + side,
            "entry_ts": row['timestamp_utc'],
            "side": side,
            "pnl_usd": (pnl * 10.0) - 1.74,
            "efficiency_ratio": row['efficiency_ratio'],
            "volatility_zscore": row['volatility_zscore']
        })

    out_df = pd.DataFrame(events)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ DEMA 100 Datamart created with {len(out_df)} events.")

if __name__ == "__main__":
    build_enriched_pullbacks_v1_8()

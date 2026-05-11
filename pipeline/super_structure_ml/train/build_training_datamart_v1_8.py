#!/usr/bin/env python3
"""
SMART_1 Training Datamart Builder v1.8: 
Finalized Features and Labels for ML Training.
Target: Pullbacks aligned with DEMA 100.
Label: Binary (Profit @ RR 1:1).
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
DB_PATH = "/home/kemal/futures/data/Level_0_Raw/MGC_5m.db"
REGIME_PATH = "/home/kemal/futures/data/Level_1_Features/modules/regime_features.parquet"
OUTPUT_PATH = "/home/kemal/futures/data/Level_2_Datamart/super_structure_ml/v1_8_training_datamart.parquet"

def rsi(c, period=14):
    delta = np.diff(c, prepend=c[0])
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    avg_up = pd.Series(up).rolling(period).mean()
    avg_down = pd.Series(down).rolling(period).mean()
    rs = avg_up / (avg_down + 1e-9)
    return 100 - (100 / (1 + rs))

def build_v1_8_training_data():
    print("🚜 Fetching 5m OHLCV data...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM investing_ohlcv_5m ORDER BY timestamp_utc", conn)
    conn.close()
    
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc']).dt.tz_localize('UTC')
    df = df[df['timestamp_utc'] >= '2023-01-01'].copy().reset_index(drop=True)
    
    print("🧮 Calculating Features...")
    h, l, c, o, v = df['high'].values, df['low'].values, df['close'].values, df['open'].values, df['volume'].values
    
    st, direction = supertrend(h, l, c, factor=4.0, atr_period=10)
    dema_100 = dema(c, 100)
    
    df['st'] = st
    df['st_direction'] = direction
    df['dema_100'] = dema_100
    df['entry_adx'] = adx(h, l, c, 14)
    df['cci_abs'] = np.abs(cci(h, l, c, 20))
    df['atr'] = _atr(h, l, c, 14)
    df['rsi_7'] = rsi(c, 7)
    df['st_slope'] = pd.Series(st).diff(5)
    
    # Candle Geometry
    body = np.abs(c - o)
    range_tot = (h - l) + 1e-9
    df['wick_ratio'] = (range_tot - body) / range_tot
    df['candle_body_atr'] = body / (df['atr'] + 1e-9)
    
    print("🌍 Merging Regime Context...")
    regime = pd.read_parquet(REGIME_PATH)
    regime['breakout_ts'] = pd.to_datetime(regime['breakout_ts'], utc=True)
    df = pd.merge_asof(df.sort_values('timestamp_utc'), 
                         regime.sort_values('breakout_ts')[['breakout_ts', 'volatility_zscore', 'efficiency_ratio']],
                         left_on='timestamp_utc', right_on='breakout_ts', direction='backward')

    print("🔍 Generating Labels (RR 1:1)...")
    # Vectorized Condition Logic
    df['prev_st_dir'] = df['st_direction'].shift(1)
    cond_long = (df['st_direction'] == -1) & (df['prev_st_dir'] == -1) & (df['close'] > df['dema_100']) & \
                (df['low'] <= df['st'] * 1.002) & (df['close'] > df['open'])
    cond_short = (df['st_direction'] == 1) & (df['prev_st_dir'] == 1) & (df['close'] < df['dema_100']) & \
                 (df['high'] >= df['st'] * 0.998) & (df['close'] < df['open'])
    
    df['is_signal'] = cond_long | cond_short
    signals = df[df['is_signal']].copy()
    
    events = []
    full_h, full_l, full_c = df['high'].values, df['low'].values, df['close'].values
    
    for idx, row in signals.iterrows():
        side = "Long" if row['st_direction'] == -1 else "Short"
        entry_price = row['close']
        sl = row['st'] - 1.0 if side == "Long" else row['st'] + 1.0
        risk = abs(entry_price - sl)
        if risk <= 0.1: continue
        tp = entry_price + risk if side == "Long" else entry_price - risk
        
        # Outcome Search
        end_idx = min(idx + 100, len(df))
        future_h = full_h[idx+1:end_idx]
        future_l = full_l[idx+1:end_idx]
        is_profit = 0
        if side == "Long":
            sl_hit = np.where(future_l <= sl)[0]
            tp_hit = np.where(future_h >= tp)[0]
            if len(tp_hit) > 0 and (len(sl_hit) == 0 or tp_hit[0] < sl_hit[0]): is_profit = 1
        else:
            sl_hit = np.where(future_h >= sl)[0]
            tp_hit = np.where(future_l <= tp)[0] # Typo here, fixing in next step
            if len(tp_hit) > 0 and (len(sl_hit) == 0 or tp_hit[0] < sl_hit[0]): is_profit = 1
            
        events.append({
            "entry_ts": row['timestamp_utc'],
            "side": side,
            "label": is_profit,
            "pnl_usd": (risk * 10.0 if is_profit else -risk * 10.0) - 1.74,
            "entry_adx": row['entry_adx'],
            "cci_abs": row['cci_abs'],
            "st_gap_ratio": abs(row['close'] - row['st']) / (row['atr'] + 1e-9),
            "efficiency_ratio": row['efficiency_ratio'],
            "volatility_zscore": row['volatility_zscore'],
            "wick_ratio": row['wick_ratio'],
            "candle_body_atr": row['candle_body_atr'],
            "st_slope": row['st_slope'],
            "rsi_7": row['rsi_7'],
            "session_cluster": pd.cut([row['timestamp_utc'].hour], bins=[-1, 6, 12, 23], labels=[0, 1, 2])[0]
        })

    out_df = pd.DataFrame(events)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Training Datamart v1.8 Ready: {len(out_df)} samples.")

if __name__ == "__main__":
    build_v1_8_training_data()

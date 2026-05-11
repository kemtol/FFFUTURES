#!/usr/bin/env python3
"""
SMART_1 Training Datamart Builder v1.11 (ULTIMATE): 
All Features: DEMA Family, Price Action, and Macro Alphas.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append("/home/kemal/futures")
from pipeline.live.super_structure import supertrend, adx, cci, _atr, dema

ROOT = Path('/home/kemal/futures')
DB_PATH = "/home/kemal/futures/data/Level_0_Raw/MGC_5m.db"
REGIME_PATH = "/home/kemal/futures/data/Level_1_Features/modules/regime_features.parquet"
MACRO_PATH = "/home/kemal/futures/data/Level_1_Features/macro_data.parquet"
OUTPUT_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v1_11_training_datamart.parquet"

def rsi(c, period=14):
    delta = np.diff(c, prepend=c[0])
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    avg_up = pd.Series(up).rolling(period).mean()
    avg_down = pd.Series(down).rolling(period).mean()
    rs = avg_up / (avg_down + 1e-9)
    return 100 - (100 / (1 + rs))

def build_v1_11_ultimate():
    print("🚜 Fetching 5m OHLCV data...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM investing_ohlcv_5m ORDER BY timestamp_utc", conn)
    conn.close()
    
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc']).dt.tz_localize('UTC')
    df = df[df['timestamp_utc'] >= '2023-01-01'].copy().reset_index(drop=True)
    
    print("🧮 Calculating All 16 Features...")
    h, l, c, o = df['high'].values, df['low'].values, df['close'].values, df['open'].values
    st, direction = supertrend(h, l, c, factor=4.0, atr_period=10)
    d50, d100, d200 = dema(c, 50), dema(c, 100), dema(c, 200)
    atr = _atr(h, l, c, 14)
    
    df['st'] = st
    df['st_direction'] = direction
    df['dema_50'], df['dema_100'], df['dema_200'] = d50, d100, d200
    df['atr'] = atr
    df['entry_adx'] = adx(h, l, c, 14)
    df['cci_abs'] = np.abs(cci(h, l, c, 20))
    df['rsi_7'] = rsi(c, 7)
    
    print("🌍 Merging Macro...")
    macro = pd.read_parquet(MACRO_PATH)
    macro['date_key'] = pd.to_datetime(macro['date']).dt.date
    df['date_key'] = df['timestamp_utc'].dt.date
    df = pd.merge(df, macro[['date_key', 'oil_return', 'us10y_change', 'dxy_return']], on='date_key', how='left')
    
    print("🔍 Generating Events...")
    df['prev_st_dir'] = df['st_direction'].shift(1)
    cond_long = (df['st_direction'] == -1) & (df['prev_st_dir'] == -1) & (df['close'] > df['dema_100']) & \
                (df['low'] <= df['st'] * 1.002) & (df['close'] > df['open'])
    cond_short = (df['st_direction'] == 1) & (df['prev_st_dir'] == 1) & (df['close'] < df['dema_100']) & \
                 (df['high'] >= df['st'] * 0.998) & (df['close'] < df['open'])
    
    signals = df[cond_long | cond_short].copy()
    full_h, full_l = df['high'].values, df['low'].values
    
    events = []
    for idx, row in signals.iterrows():
        side = "Long" if row['st_direction'] == -1 else "Short"
        risk = abs(row['close'] - (row['st'] - 1.0 if side == "Long" else row['st'] + 1.0))
        if risk <= 0.1: continue
        tp = row['close'] + risk if side == "Long" else row['close'] - risk
        
        # Outcome Search
        end_idx = min(idx + 100, len(df))
        is_profit = 0
        future_h, future_l = full_h[idx+1:end_idx], full_l[idx+1:end_idx]
        if side == "Long":
            sl_hit = np.where(future_l <= row['st'] - 1.0)[0]
            tp_hit = np.where(future_h >= tp)[0]
            if len(tp_hit) > 0 and (len(sl_hit) == 0 or tp_hit[0] < sl_hit[0]): is_profit = 1
        else:
            sl_hit = np.where(future_h >= row['st'] + 1.0)[0]
            tp_hit = np.where(future_l <= tp)[0]
            if len(tp_hit) > 0 and (len(sl_hit) == 0 or tp_hit[0] < sl_hit[0]): is_profit = 1
            
        # Price Action
        body = abs(row['close'] - row['open'])
        range_tot = (row['high'] - row['low']) + 1e-9
            
        events.append({
            "entry_ts": row['timestamp_utc'], "side": side, "label": is_profit,
            "pnl_usd": (risk * 10.0 if is_profit else -risk * 10.0) - 1.74,
            "dist_d50_atr": (row['close'] - row['dema_50']) / (row['atr'] + 1e-9),
            "dist_d100_atr": (row['close'] - row['dema_100']) / (row['atr'] + 1e-9),
            "dist_d200_atr": (row['close'] - row['dema_200']) / (row['atr'] + 1e-9),
            "d100_slope": pd.Series(df['close']).diff(5).iloc[idx],
            "d200_slope": pd.Series(df['dema_200']).diff(5).iloc[idx],
            "dema_stack": 3 if (row['close'] > row['dema_50'] > row['dema_100'] > row['dema_200']) else (-3 if (row['close'] < row['dema_50'] < row['dema_100'] < row['dema_200']) else 0),
            "entry_adx": row['entry_adx'], "cci_abs": row['cci_abs'], "rsi_7": row['rsi_7'],
            "wick_ratio": (range_tot - body) / range_tot,
            "candle_body_atr": body / (row['atr'] + 1e-9),
            "st_gap_ratio": abs(row['close'] - row['st']) / (row['atr'] + 1e-9),
            "oil_return": row['oil_return'], "us10y_change": row['us10y_change'], "dxy_return": row['dxy_return'],
            "session_cluster": pd.cut([row['timestamp_utc'].hour], bins=[-1, 6, 12, 23], labels=[0, 1, 2])[0]
        })

    out_df = pd.DataFrame(events)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ ULTIMATE v1.11 Datamart Ready: {len(out_df)} samples.")

if __name__ == "__main__":
    build_v1_11_ultimate()

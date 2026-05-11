#!/usr/bin/env python3
"""
Feature Engineering Phase 3: High Granularity Metrics for Meta-v5.
Adds MAE/MFE ratios, Session Momentum, and Dynamic Range features.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
INPUT_DATAMART = ROOT / "data/Level_2_Datamart/super_structure_ml/v5_raw_expanded.parquet"
OUTPUT_DATAMART = ROOT / "data/Level_2_Datamart/super_structure_ml/v6_advanced_features.parquet"

def build_advanced_features(df):
    print(f"Building Meta-v5 Granular Features for {len(df)} trades...")
    
    # 1. Base Features
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    df['day_of_week'] = df['entry_ts'].dt.dayofweek
    df['hour_utc'] = df['entry_ts'].dt.hour
    df['atr_pct'] = (df['entry_atr'] / df['entry_price']) * 100
    df['st_gap_ratio'] = df['st_distance'] / (df['entry_atr'] + 1e-9)

    # 2. MAE/MFE Granularity (The 'Pain' Metrics)
    # mae_usd is negative in our data usually, so we take absolute
    df['mae_atr_ratio'] = abs(df['mae_usd'] / 10.0) / (df['entry_atr'] + 1e-9)
    df['mfe_atr_ratio'] = (df['mfe_usd'] / 10.0) / (df['entry_atr'] + 1e-9)
    
    # 3. Candle Geometry (Price Action)
    body = (df['entry_bar_close'] - df['entry_price']).abs()
    range_tot = (df['entry_bar_high'] - df['entry_bar_low']) + 1e-9
    df['wick_ratio'] = (range_tot - body) / range_tot
    df['candle_body_atr'] = body / (df['entry_atr'] + 1e-9)
    
    # 4. Indicators & Strength
    df['cci_abs'] = df['entry_cci'].abs()
    df['is_st_aligned'] = (((df['side'] == 'Long') & (df['entry_st_direction'] < 0)) | \
                          ((df['side'] == 'Short') & (df['entry_st_direction'] > 0))).astype(int)
    
    # 5. Session Clusters (Asian vs US vs London)
    # Asian: 0-6 UTC, London: 7-12 UTC, US: 13-21 UTC
    df['session_cluster'] = pd.cut(df['hour_utc'], bins=[-1, 6, 12, 23], labels=[0, 1, 2]).astype(int)
    
    # 6. Regime Context (Merge from L1)
    regime_path = ROOT / 'data/Level_1_Features/modules/regime_features.parquet'
    if regime_path.exists():
        regime = pd.read_parquet(regime_path)
        regime['breakout_ts'] = pd.to_datetime(regime['breakout_ts'], utc=True).dt.as_unit('ns')
        df['entry_ts_ns'] = df['entry_ts'].dt.as_unit('ns')
        df = df.sort_values('entry_ts_ns')
        regime = regime.sort_values('breakout_ts')
        df = pd.merge_asof(df, regime[['breakout_ts', 'regime_state', 'volatility_zscore', 'efficiency_ratio']],
                             left_on='entry_ts_ns', right_on='breakout_ts', direction='backward')

    # 7. Momentum Velocity (ADX change) - Need to handle grouping by symbol if needed, 
    # but here we have chronological trades.
    df['adx_delta'] = df['entry_adx'].diff().fillna(0)

    return df

def main():
    if not INPUT_DATAMART.exists():
        print("Input datamart not found.")
        return
    df = pd.read_parquet(INPUT_DATAMART)
    df_featured = build_advanced_features(df)
    df_featured = df_featured.dropna(subset=['regime_state'])
    df_featured.to_parquet(OUTPUT_DATAMART, index=False)
    print(f"✅ V6 Advanced Datamart ready: {len(df_featured)} trades.")

if __name__ == "__main__":
    main()

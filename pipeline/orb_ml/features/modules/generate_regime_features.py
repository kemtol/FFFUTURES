#!/usr/bin/env python3
"""
Modular Regime Detection (L1 Features) - Version 2 (with Z-Scores).
Uses GMM for categorical states and adds continuous Z-Scores for Volatility and Efficiency.
"""

import argparse
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
MODULE_NAME = "regime_features"
EVENT_KEY = ["date", "session", "orb_tf", "breakout_ts"]

def load_sources():
    db_path = ROOT / "data/Level_0_Raw/MGC_1m.db"
    bo_path = ROOT / "data/Level_1_Features/breakout_events.parquet"
    if not db_path.exists() or not bo_path.exists():
        raise FileNotFoundError("Missing MGC_1m.db or breakout_events.parquet")
    
    bo = pd.read_parquet(bo_path)
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql(
            "SELECT timestamp_utc, close, high, low FROM investing_ohlcv_1m "
            "WHERE symbol='MICRO_GOLD' ORDER BY epoch_ms", 
            conn
        )
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return bo, df

def build_regime_features(df_ohlcv):
    print("Calculating Regime Z-Scores and Efficiency...")
    
    # 1. Resample to 15m for stability
    df = df_ohlcv.set_index("timestamp_utc").resample("15min").agg({
        "close": "last", "high": "max", "low": "min"
    }).dropna()
    
    # 2. Kaufman Efficiency Ratio (20 bars)
    # ER = Directional Move / Volatility (Sum of absolute moves)
    change = df["close"].diff(20).abs()
    volatility = df["close"].diff().abs().rolling(20).sum()
    df["efficiency_ratio"] = change / (volatility + 1e-9)
    
    # 3. Volatility Z-Score (100 bars)
    raw_vol = (df["high"] - df["low"]) / df["close"]
    mean_vol = raw_vol.rolling(100).mean()
    std_vol = raw_vol.rolling(100).std()
    df["volatility_zscore"] = (raw_vol - mean_vol) / (std_vol + 1e-9)
    
    # 4. GMM Regime States (Categorical)
    X = np.column_stack([
        df["efficiency_ratio"].fillna(0.5).values,
        df["volatility_zscore"].fillna(0).values
    ])
    gmm = GaussianMixture(n_components=3, random_state=42)
    df["regime_state"] = gmm.fit_predict(X)
    
    # Standardize mapping: 0=Quiet, 1=Trending, 2=Choppy
    # Quiet = Low Volatility Z-Score
    state_vols = df.groupby("regime_state")["volatility_zscore"].mean().sort_values()
    mapping = {state_vols.index[0]: 0, state_vols.index[1]: 1, state_vols.index[2]: 2}
    df["regime_state"] = df["regime_state"].map(mapping)
    
    return df[["regime_state", "volatility_zscore", "efficiency_ratio"]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    out_path = ROOT / f"data/Level_1_Features/modules/{MODULE_NAME}.parquet"
    if out_path.exists() and not args.force:
        print(f"Module {MODULE_NAME} exists.")
        return

    bo, ohlcv = load_sources()
    regime_df = build_regime_features(ohlcv)
    
    bo = bo.sort_values("breakout_ts")
    regime_df = regime_df.sort_index()
    
    merged = pd.merge_asof(
        bo, regime_df, left_on="breakout_ts", 
        right_index=True, direction="backward"
    )
    
    final_cols = EVENT_KEY + ["regime_state", "volatility_zscore", "efficiency_ratio"]
    out_df = merged[final_cols].copy()
    out_df["date"] = pd.to_datetime(out_df["date"]).dt.date
    
    out_df.to_parquet(out_path, index=False)
    print(f"✅ Generated {MODULE_NAME} with Z-Scores: {out_path}")

if __name__ == "__main__":
    main()

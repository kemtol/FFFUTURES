#!/usr/bin/env python3
"""
Modular Regime Detection (L1 Features) - Version 2 (with Z-Scores).
FIXED: Forces calculation for all dates in database.
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
    # To cover all dates, we'll build a dummy events dataframe from the DB itself 
    # instead of relying on breakout_events.parquet which might be stale.
    with sqlite3.connect(str(db_path)) as conn:
        ohlcv = pd.read_sql(
            "SELECT timestamp_utc, close, high, low FROM investing_ohlcv_1m "
            "WHERE symbol='MICRO_GOLD' ORDER BY epoch_ms", 
            conn
        )
    ohlcv["timestamp_utc"] = pd.to_datetime(ohlcv["timestamp_utc"], utc=True)
    return ohlcv

def build_regime_features(df_ohlcv):
    print("Calculating Regime Z-Scores and Efficiency...")
    df = df_ohlcv.set_index("timestamp_utc").resample("15min").agg({
        "close": "last", "high": "max", "low": "min"
    }).dropna()
    
    change = df["close"].diff(20).abs()
    volatility = df["close"].diff().abs().rolling(20).sum()
    df["efficiency_ratio"] = change / (volatility + 1e-9)
    
    raw_vol = (df["high"] - df["low"]) / df["close"]
    mean_vol = raw_vol.rolling(100).mean()
    std_vol = raw_vol.rolling(100).std()
    df["volatility_zscore"] = (raw_vol - mean_vol) / (std_vol + 1e-9)
    
    X = np.column_stack([
        df["efficiency_ratio"].fillna(0.5).values,
        df["volatility_zscore"].fillna(0).values
    ])
    gmm = GaussianMixture(n_components=3, random_state=42)
    df["regime_state"] = gmm.fit_predict(X)
    
    state_vols = df.groupby("regime_state")["volatility_zscore"].mean().sort_values()
    mapping = {state_vols.index[0]: 0, state_vols.index[1]: 1, state_vols.index[2]: 2}
    df["regime_state"] = df["regime_state"].map(mapping)
    
    return df[["regime_state", "volatility_zscore", "efficiency_ratio"]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    out_path = ROOT / f"data/Level_1_Features/modules/{MODULE_NAME}.parquet"
    ohlcv = load_sources()
    regime_df = build_regime_features(ohlcv)
    
    # Export with breakout_ts as a column for merge_asof
    out_df = regime_df.reset_index().rename(columns={"timestamp_utc": "breakout_ts"})
    out_df["date"] = out_df["breakout_ts"].dt.date
    out_df["session"] = "Any"
    out_df["orb_tf"] = 5 # placeholder
    
    out_df.to_parquet(out_path, index=False)
    print(f"✅ Generated {MODULE_NAME} with Z-Scores (All Dates): {out_path}")
    print(f"Max Date in Regime: {out_df['breakout_ts'].max()}")

if __name__ == "__main__":
    main()

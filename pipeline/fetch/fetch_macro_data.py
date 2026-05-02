#!/usr/bin/env python3
"""Fetch macro market data (SPY, DXY, US10Y, Oil) via yfinance daily.

Output: data/Level_1_Features/macro_data.parquet
Columns: date (object), spy_close, dxy_close, us10y_close, oil_close

This is a ONE-TIME fetch. The parquet becomes a static data source
referenced by generate_macro_features.py via load_sources().
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
L1_DIR = PROJECT_ROOT / "data" / "Level_1_Features"
L1_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = {
    "spy":   "SPY",         # S&P 500 ETF (tracks SPX)
    "dxy":   "DX-Y.NYB",    # US Dollar Index
    "us10y": "^TNX",        # 10-Year Treasury Yield (%)
    "oil":   "CL=F",        # Crude Oil Futures (front month)
}

# Slightly wider than breakout_events range (2010-10-04 → 2026-04-24)
START_DATE = "2009-06-01"
END_DATE = "2026-05-01"


def fetch_daily(tickers: dict[str, str]) -> pd.DataFrame:
    """Download daily OHLCV for all tickers, flatten MultiIndex columns."""
    symbols = list(tickers.values())
    names = list(tickers.keys())

    print(f"[fetch_macro] Downloading {len(symbols)} tickers: {symbols}")
    print(f"[fetch_macro] Range: {START_DATE} → {END_DATE}")

    raw = yf.download(
        symbols,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        progress=True,
        group_by="ticker",
    )

    if raw is None or raw.empty:
        print("[fetch_macro] ERROR: yfinance returned empty data", file=sys.stderr)
        sys.exit(1)

    print(f"[fetch_macro] Raw shape: {raw.shape}")
    print(f"[fetch_macro] Raw columns: {list(raw.columns)}")
    print(f"[fetch_macro] Raw index range: {raw.index[0]} → {raw.index[-1]}")

    # yfinance returns MultiIndex columns (ticker, field) or single-level.
    # Flatten to {name}_close, {name}_open, etc.
    rows = []
    for dt, row in raw.iterrows():
        record = {"date": str(dt.date())}
        for name, symbol in tickers.items():
            try:
                # Try MultiIndex access: raw.loc[dt, (symbol, 'Close')]
                close = raw.loc[dt, (symbol, "Close")]
                record[f"{name}_close"] = float(close)
            except (KeyError, TypeError):
                try:
                    # Fallback: single-level (some yfinance versions)
                    close = row.get("Close") if isinstance(row, pd.Series) else None
                    if isinstance(close, (int, float, np.floating)):
                        record[f"{name}_close"] = float(close)
                except (KeyError, TypeError, ValueError):
                    record[f"{name}_close"] = np.nan
        rows.append(record)

    df = pd.DataFrame(rows)
    df["date"] = df["date"].astype("object")  # object dtype for merge compatibility
    print(f"[fetch_macro] Flattened shape: {df.shape}")
    print(f"[fetch_macro] Columns: {list(df.columns)}")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns useful for feature engineering."""
    # SPY 200-day MA
    df["spy_ma200"] = df["spy_close"].rolling(200, min_periods=100).mean()

    # DXY 50-day MA
    df["dxy_ma50"] = df["dxy_close"].rolling(50, min_periods=25).mean()

    # Daily returns
    df["spy_return"] = df["spy_close"].pct_change()
    df["dxy_return"] = df["dxy_close"].pct_change()
    df["oil_return"] = df["oil_close"].pct_change()

    # US10Y daily change (absolute, in % points)
    df["us10y_change"] = df["us10y_close"].diff()

    return df


def main(full_rebuild: bool = False) -> None:
    out_path = L1_DIR / "macro_data.parquet"

    if out_path.exists() and not full_rebuild:
        print(f"[fetch_macro] {out_path} already exists. Use --full-rebuild to re-download.")
        return

    print("[fetch_macro] Fetching macro data from Yahoo Finance...")
    df = fetch_daily(TICKERS)
    df = compute_features(df)

    # Check NaN stats
    total = len(df)
    for col in df.columns:
        if col == "date":
            continue
        n_null = df[col].isna().sum()
        if n_null > 0:
            pct = n_null / total * 100
            print(f"[fetch_macro]  ⚠  {col}: {n_null}/{total} NaN ({pct:.1f}%)")

    df.to_parquet(out_path, index=False)
    print(f"[fetch_macro] ✅ Saved {len(df)} rows × {len(df.columns)} cols → {out_path}")
    print(f"[fetch_macro]    Date range: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-rebuild", action="store_true", help="Re-download even if exists")
    args = parser.parse_args()
    main(full_rebuild=args.full_rebuild)

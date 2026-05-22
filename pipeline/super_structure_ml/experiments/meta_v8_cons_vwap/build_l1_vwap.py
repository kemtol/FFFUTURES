#!/usr/bin/env python3
"""Build L1 CT trading-day VWAP context for Super Structure research.

This writes a reusable feature cache from immutable L0 1m bars. It does not
touch live code, live state, or any model artifact.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = Path(__file__).resolve().with_name("config.json")
RAW_DB = ROOT / "data/Level_0_Raw/MGC_1m.db"
RAW_TABLE = "investing_ohlcv_1m"
SYMBOL = "MICRO_GOLD"
TIMEFRAME = "1m"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def _topstep_trade_day(ts: pd.Series) -> pd.Series:
    ts_ct = ts.dt.tz_convert("America/Chicago")
    return (ts_ct - pd.Timedelta(hours=15, minutes=10)).dt.date


def load_base_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    cfg = load_config()
    base_path = ROOT / cfg["baseline"]["datamart"]
    df = pd.read_parquet(base_path, columns=["entry_ts"])
    ts = pd.to_datetime(df["entry_ts"], utc=True)
    return ts.min() - pd.Timedelta(days=2), ts.max() + pd.Timedelta(minutes=5)


def load_1m_bars(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    with sqlite3.connect(f"file:{RAW_DB}?mode=ro", uri=True, timeout=30) as conn:
        df = pd.read_sql(
            f"""
            SELECT timestamp_utc, open, high, low, close, volume
            FROM {RAW_TABLE}
            WHERE symbol = ? AND timeframe = ?
              AND timestamp_utc >= ? AND timestamp_utc <= ?
            ORDER BY epoch_ms
            """,
            conn,
            params=[
                SYMBOL,
                TIMEFRAME,
                start.strftime("%Y-%m-%d %H:%M:%S"),
                end.strftime("%Y-%m-%d %H:%M:%S"),
            ],
        )
    if df.empty:
        raise RuntimeError(f"No raw bars found for {start} -> {end}")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def _grouped_zscore(values: pd.Series) -> pd.Series:
    mean = values.rolling(50, min_periods=20).mean()
    std = values.rolling(50, min_periods=20).std()
    return (values - mean) / (std + 1e-9)


def build_vwap_context(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    df["ct_trade_day"] = _topstep_trade_day(df["timestamp_utc"])
    typical = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    volume = df["volume"].fillna(0).astype(float).clip(lower=0)
    weight = volume.where(volume > 0, 1.0)

    df["_pv"] = typical * weight
    df["_weight"] = weight
    grouped = df.groupby("ct_trade_day", sort=False)
    df["ct_vwap"] = grouped["_pv"].cumsum() / grouped["_weight"].cumsum()
    df["ct_vwap_slope_20"] = grouped["ct_vwap"].diff(20)
    df["_dist"] = df["close"].astype(float) - df["ct_vwap"]
    df["vwap_deviation_z_50"] = grouped["_dist"].transform(_grouped_zscore)

    return df[
        [
            "timestamp_utc",
            "ct_trade_day",
            "ct_vwap",
            "ct_vwap_slope_20",
            "vwap_deviation_z_50",
        ]
    ].sort_values("timestamp_utc").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    out_path = ROOT / cfg["level1"]["vwap_context"]
    if out_path.exists() and not args.force and not args.dry_run:
        raise SystemExit(f"Output exists: {out_path} (use --force)")

    start, end = load_base_range()
    bars = load_1m_bars(start, end)
    context = build_vwap_context(bars)

    print(f"Raw bars: {len(bars):,}")
    print(f"L1 rows: {len(context):,}")
    print(f"Range: {context['timestamp_utc'].min()} -> {context['timestamp_utc'].max()}")
    print("NaN rates:")
    print(context[["ct_vwap", "ct_vwap_slope_20", "vwap_deviation_z_50"]].isna().mean().to_string())

    if args.dry_run:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    context.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

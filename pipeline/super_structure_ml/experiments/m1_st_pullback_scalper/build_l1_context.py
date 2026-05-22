#!/usr/bin/env python3
"""Build L1 M1 context for the standalone SuperTrend pullback scalper.

L1 is the integrity layer: one row per raw M1 candle, including OHLCV and
causal indicators/features. L2 event datamarts must be derived from this file,
not recompute their own context silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_m1_events import ROOT, add_indicators, load_1m_bars, load_config  # noqa: E402


REQUIRED_L1_COLUMNS = [
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "st",
    "st_direction",
    "prev_st_direction",
    "atr",
    "entry_adx",
    "entry_cci",
    "rsi_7",
    "dema_50",
    "dema_100",
    "dema_200",
    "ct_trade_day",
    "ct_vwap",
    "ct_vwap_slope_20",
    "vwap_deviation_z_50",
    "st_slope_5_atr",
    "close_slope_3_atr",
    "close_slope_5_atr",
    "prev_gap_seconds",
    "data_quality_ok",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_l1(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_L1_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"L1 missing columns: {missing}")
    if df["timestamp_utc"].duplicated().any():
        dupes = int(df["timestamp_utc"].duplicated().sum())
        raise ValueError(f"L1 duplicate timestamp_utc rows: {dupes}")
    if not df["timestamp_utc"].is_monotonic_increasing:
        raise ValueError("L1 timestamp_utc is not sorted ascending")

    bad_ohlc = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df["volume"].fillna(0) < 0)
    )
    if bad_ohlc.any():
        raise ValueError(f"L1 OHLCV invariant failed rows: {int(bad_ohlc.sum())}")

    core = ["open", "high", "low", "close", "volume", "ct_vwap"]
    nulls = df[core].isna().sum()
    bad = nulls[nulls > 0]
    if not bad.empty:
        raise ValueError(f"L1 unexpected nulls: {bad.to_dict()}")


def write_manifest(path: Path, manifest_path: Path, df: pd.DataFrame, cfg: dict) -> None:
    manifest = {
        "artifact": str(path.relative_to(ROOT)),
        "sha256": file_sha256(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "timestamp_min": df["timestamp_utc"].min().isoformat(),
        "timestamp_max": df["timestamp_utc"].max().isoformat(),
        "source": cfg["source"],
        "indicators": cfg["indicators"],
        "null_rates": {
            c: float(v)
            for c, v in df[REQUIRED_L1_COLUMNS].isna().mean().sort_values(ascending=False).items()
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    out_path = ROOT / cfg["outputs"]["l1_context"]
    manifest_path = ROOT / cfg["outputs"]["l1_manifest"]
    if out_path.exists() and not args.force and not args.dry_run:
        raise SystemExit(f"Output exists: {out_path} (use --force)")

    bars = load_1m_bars(cfg, args.start_date, args.end_date)
    context = add_indicators(bars, cfg)
    from build_m1_events import add_data_quality_flags
    context = add_data_quality_flags(context, cfg)
    context = context[[c for c in REQUIRED_L1_COLUMNS if c in context.columns]].copy()
    validate_l1(context)

    print(f"L1 rows: {len(context):,}")
    print(f"Range: {context['timestamp_utc'].min()} -> {context['timestamp_utc'].max()}")
    print("NaN rates:")
    print(context[REQUIRED_L1_COLUMNS].isna().mean().sort_values(ascending=False).head(12).to_string())

    if args.dry_run:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    context.to_parquet(out_path, index=False)
    write_manifest(out_path, manifest_path, context, cfg)
    print(f"Wrote {out_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

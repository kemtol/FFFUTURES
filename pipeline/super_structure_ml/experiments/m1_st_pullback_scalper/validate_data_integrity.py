#!/usr/bin/env python3
"""Validate M1 scalper L1/L2 data integrity before training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_l1_context import REQUIRED_L1_COLUMNS, validate_l1  # noqa: E402
from build_m1_events import ROOT, load_config, validate_events  # noqa: E402


REQUIRED_L2_COLUMNS = [
    "event_id",
    "signal_ts",
    "entry_ts",
    "side",
    "signal_open",
    "signal_high",
    "signal_low",
    "signal_close",
    "signal_volume",
    "entry_open",
    "entry_high",
    "entry_low",
    "entry_close",
    "entry_volume",
    "entry_gap_seconds",
    "signal_prev_gap_seconds",
    "signal_data_quality_ok",
    "entry_price",
    "sl_price",
    "tp_price",
    "exit_ts",
    "exit_price",
    "exit_reason",
    "hold_bars",
    "risk_pts",
    "label",
    "pnl_usd",
    "signal_adx",
    "signal_cci",
    "signal_rsi_7",
    "signal_atr",
    "signal_st",
    "signal_dema_100",
    "signal_ct_vwap",
    "dist_to_ct_vwap_atr",
    "vwap_deviation_z_50",
]


def main() -> int:
    cfg = load_config()
    l1_path = ROOT / cfg["outputs"]["l1_context"]
    l2_path = ROOT / cfg["outputs"]["events"]

    if not l1_path.exists():
        raise SystemExit(f"Missing L1 context: {l1_path}")
    if not l2_path.exists():
        raise SystemExit(f"Missing L2 events: {l2_path}")

    l1 = pd.read_parquet(l1_path)
    l2 = pd.read_parquet(l2_path)
    l1["timestamp_utc"] = pd.to_datetime(l1["timestamp_utc"], utc=True)
    l2["signal_ts"] = pd.to_datetime(l2["signal_ts"], utc=True)
    l2["entry_ts"] = pd.to_datetime(l2["entry_ts"], utc=True)
    l2["exit_ts"] = pd.to_datetime(l2["exit_ts"], utc=True)

    validate_l1(l1)
    missing_l1 = [c for c in REQUIRED_L1_COLUMNS if c not in l1.columns]
    missing_l2 = [c for c in REQUIRED_L2_COLUMNS if c not in l2.columns]
    if missing_l1 or missing_l2:
        raise SystemExit(
            json.dumps(
                {"missing_l1": missing_l1, "missing_l2": missing_l2},
                indent=2,
            )
        )

    validate_events(l2, l1)

    feature_cols = [
        c
        for c in l2.columns
        if c
        not in {
            "event_id",
            "signal_ts",
            "entry_ts",
            "exit_ts",
            "side",
            "exit_reason",
            "label",
            "pnl_usd",
            "exit_price",
        }
    ]
    null_rates = l2[feature_cols].isna().mean()
    bad_nulls = null_rates[null_rates > 0]
    if not bad_nulls.empty:
        raise SystemExit(f"L2 feature nulls found: {bad_nulls.to_dict()}")

    print("PASS M1 data integrity")
    print(f"L1 rows: {len(l1):,} | {l1['timestamp_utc'].min()} -> {l1['timestamp_utc'].max()}")
    print(f"L2 rows: {len(l2):,} | signal {l2['signal_ts'].min()} -> {l2['signal_ts'].max()}")
    print(f"L2 entry range: {l2['entry_ts'].min()} -> {l2['entry_ts'].max()}")
    print(f"L2 columns: {len(l2.columns)}")
    print(f"L2 avg pnl: {l2['pnl_usd'].mean():.4f} | win rate: {(l2['pnl_usd'] > 0).mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

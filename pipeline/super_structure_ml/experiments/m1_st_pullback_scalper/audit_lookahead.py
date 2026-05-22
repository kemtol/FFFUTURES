#!/usr/bin/env python3
"""Audit M1 scalper datamart for look-ahead leakage.

P0 timing contract:
- `signal_ts` is bar t. Model-safe features must come from bar t or earlier.
- `entry_ts` is bar t+1. Execution price uses next open plus slippage.
- Entry/execution columns are stored for simulation integrity, but are not
  model-safe decision features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_m1_events import ROOT, load_config, session_cluster  # noqa: E402


OUTCOME_COLUMNS = {
    "exit_ts",
    "exit_price",
    "exit_reason",
    "hold_bars",
    "label",
    "pnl_usd",
}

METADATA_COLUMNS = {
    "event_id",
    "signal_ts",
    "side",
    "point_value",
    "commission_usd",
    "slippage_pts",
}

EXECUTION_COLUMNS = {
    "entry_ts",
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
    "risk_pts",
}

SIGNAL_DIRECT_MAP = {
    "signal_open": "open",
    "signal_high": "high",
    "signal_low": "low",
    "signal_close": "close",
    "signal_volume": "volume",
    "signal_adx": "entry_adx",
    "signal_cci": "entry_cci",
    "signal_rsi_7": "rsi_7",
    "signal_atr": "atr",
    "signal_st": "st",
    "signal_st_direction": "st_direction",
    "signal_dema_50": "dema_50",
    "signal_dema_100": "dema_100",
    "signal_dema_200": "dema_200",
    "signal_ct_vwap": "ct_vwap",
    "signal_ct_vwap_slope_20": "ct_vwap_slope_20",
    "st_slope_5_atr": "st_slope_5_atr",
    "close_slope_3_atr": "close_slope_3_atr",
    "close_slope_5_atr": "close_slope_5_atr",
    "vwap_deviation_z_50": "vwap_deviation_z_50",
    "signal_prev_gap_seconds": "prev_gap_seconds",
    "signal_data_quality_ok": "data_quality_ok",
}

SIGNAL_DERIVED_FEATURES = {
    "cci_abs",
    "st_gap_atr",
    "touch_distance_atr",
    "pullback_band_atr",
    "dist_d50_atr",
    "dist_d100_atr",
    "dist_d200_atr",
    "dema_stack",
    "wick_ratio",
    "candle_body_atr",
    "bar_range_atr",
    "directional_close_pos",
    "dist_to_ct_vwap_atr",
    "ct_vwap_slope_20_atr",
    "hour_utc",
    "dow",
    "session_cluster",
}


def load_data() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    l1 = pd.read_parquet(ROOT / cfg["outputs"]["l1_context"])
    l2 = pd.read_parquet(ROOT / cfg["outputs"]["events"])
    l1["timestamp_utc"] = pd.to_datetime(l1["timestamp_utc"], utc=True)
    l2["signal_ts"] = pd.to_datetime(l2["signal_ts"], utc=True)
    l2["entry_ts"] = pd.to_datetime(l2["entry_ts"], utc=True)
    l2["exit_ts"] = pd.to_datetime(l2["exit_ts"], utc=True)
    return cfg, l1.sort_values("timestamp_utc"), l2.sort_values("signal_ts")


def signal_derived(row: pd.Series, signal_row: pd.Series, cfg: dict) -> dict[str, float | int]:
    rules = cfg["candidate_rules"]
    side = row["side"]
    atr = float(signal_row["atr"]) + 1e-9
    close = float(signal_row["close"])
    open_ = float(signal_row["open"])
    high = float(signal_row["high"])
    low = float(signal_row["low"])
    st = float(signal_row["st"])

    if side == "Long":
        touch_distance = (low - st) / atr
        directional_close_pos = (close - low) / (high - low + 1e-9)
    else:
        touch_distance = (st - high) / atr
        directional_close_pos = 1.0 - ((close - low) / (high - low + 1e-9))

    pullback_band = max(
        float(rules["min_pullback_band_pts"]),
        float(signal_row["atr"]) * float(rules["pullback_band_atr"]),
    )
    body = abs(close - open_)
    bar_range = high - low
    ts = row["signal_ts"]

    return {
        "cci_abs": abs(float(signal_row["entry_cci"])),
        "st_gap_atr": abs(close - st) / atr,
        "touch_distance_atr": touch_distance,
        "pullback_band_atr": pullback_band / atr,
        "dist_d50_atr": (close - float(signal_row["dema_50"])) / atr,
        "dist_d100_atr": (close - float(signal_row["dema_100"])) / atr,
        "dist_d200_atr": (close - float(signal_row["dema_200"])) / atr,
        "dema_stack": (
            3
            if close > float(signal_row["dema_50"]) > float(signal_row["dema_100"]) > float(signal_row["dema_200"])
            else -3
            if close < float(signal_row["dema_50"]) < float(signal_row["dema_100"]) < float(signal_row["dema_200"])
            else 0
        ),
        "wick_ratio": (bar_range - body) / (bar_range + 1e-9),
        "candle_body_atr": body / atr,
        "bar_range_atr": bar_range / atr,
        "directional_close_pos": directional_close_pos,
        "dist_to_ct_vwap_atr": (close - float(signal_row["ct_vwap"])) / atr,
        "ct_vwap_slope_20_atr": float(signal_row["ct_vwap_slope_20"]) / atr
        if pd.notna(signal_row["ct_vwap_slope_20"])
        else 0.0,
        "hour_utc": int(ts.hour),
        "dow": int(ts.dayofweek),
        "session_cluster": session_cluster(ts),
    }


def execution_expected(row: pd.Series, signal_row: pd.Series, entry_row: pd.Series, cfg: dict) -> dict[str, float]:
    exits = cfg["exit_rules"]
    costs = cfg["costs"]
    side = row["side"]
    st_buffer = float(exits["st_buffer_pts"])
    rr = float(exits["rr_target"])
    slippage = float(costs["slippage_pts"])
    entry_open = float(entry_row["open"])
    entry_price = entry_open + slippage if side == "Long" else entry_open - slippage
    st = float(signal_row["st"])

    if side == "Long":
        sl = st - st_buffer
        risk = entry_price - sl
        tp = entry_price + risk * rr
    else:
        sl = st + st_buffer
        risk = sl - entry_price
        tp = entry_price - risk * rr

    gap_seconds = (row["entry_ts"] - row["signal_ts"]).total_seconds()
    return {
        "entry_open": float(entry_row["open"]),
        "entry_high": float(entry_row["high"]),
        "entry_low": float(entry_row["low"]),
        "entry_close": float(entry_row["close"]),
        "entry_volume": float(entry_row["volume"]) if pd.notna(entry_row["volume"]) else 0.0,
        "entry_gap_seconds": float(gap_seconds),
        "entry_price": entry_price,
        "sl_price": sl,
        "tp_price": tp,
        "risk_pts": risk,
    }


def compare_values(actual: object, expected: object) -> float:
    if pd.isna(actual) and pd.isna(expected):
        return 0.0
    if isinstance(expected, (bool, np.bool_)):
        return 0.0 if bool(actual) == bool(expected) else float("inf")
    if isinstance(expected, (int, np.integer)):
        return 0.0 if int(actual) == int(expected) else float("inf")
    return abs(float(actual) - float(expected))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    cfg, l1, l2 = load_data()
    sample_n = min(args.sample, len(l2))
    sample = l2.sample(sample_n, random_state=args.seed).sort_values("signal_ts")
    l1_idx = l1.set_index("timestamp_utc")

    missing_signal_ts = sample[~sample["signal_ts"].isin(l1_idx.index)]
    missing_entry_ts = sample[~sample["entry_ts"].isin(l1_idx.index)]
    if not missing_signal_ts.empty:
        raise SystemExit(f"FAIL: sampled L2 signal_ts missing from L1: {len(missing_signal_ts)}")
    if not missing_entry_ts.empty:
        raise SystemExit(f"FAIL: sampled L2 entry_ts missing from L1: {len(missing_entry_ts)}")

    max_diff: dict[str, float] = {}
    failures = []

    for _, row in sample.iterrows():
        signal_row = l1_idx.loc[row["signal_ts"]]
        entry_row = l1_idx.loc[row["entry_ts"]]
        if row["entry_ts"] <= row["signal_ts"]:
            failures.append((row["event_id"], "entry_ts_order", row["entry_ts"], row["signal_ts"], float("inf")))

        for event_col, l1_col in SIGNAL_DIRECT_MAP.items():
            expected = signal_row[l1_col]
            if event_col in {"signal_ct_vwap_slope_20", "vwap_deviation_z_50"} and pd.isna(expected):
                expected = 0.0
            diff = compare_values(row[event_col], expected)
            max_diff[event_col] = max(max_diff.get(event_col, 0.0), diff)
            if diff > args.tolerance:
                failures.append((row["event_id"], event_col, row[event_col], expected, diff))

        for col, expected in signal_derived(row, signal_row, cfg).items():
            diff = compare_values(row[col], expected)
            max_diff[col] = max(max_diff.get(col, 0.0), diff)
            if diff > args.tolerance:
                failures.append((row["event_id"], col, row[col], expected, diff))

        for col, expected in execution_expected(row, signal_row, entry_row, cfg).items():
            diff = compare_values(row[col], expected)
            max_diff[col] = max(max_diff.get(col, 0.0), diff)
            if diff > args.tolerance:
                failures.append((row["event_id"], col, row[col], expected, diff))

    known_columns = (
        OUTCOME_COLUMNS
        | METADATA_COLUMNS
        | EXECUTION_COLUMNS
        | set(SIGNAL_DIRECT_MAP)
        | SIGNAL_DERIVED_FEATURES
    )
    unknown = sorted(set(l2.columns) - known_columns)
    if unknown:
        failures.append(("COLUMN_CLASSIFICATION", ",".join(unknown), "", "", float("inf")))

    print("Column classification:")
    print(f"  model-safe direct signal:  {len(SIGNAL_DIRECT_MAP)}")
    print(f"  model-safe derived signal: {len(SIGNAL_DERIVED_FEATURES)}")
    print(f"  execution/pricing:         {len(EXECUTION_COLUMNS)} -> {sorted(EXECUTION_COLUMNS)}")
    print(f"  outcome/future:            {len(OUTCOME_COLUMNS)} -> {sorted(OUTCOME_COLUMNS)}")
    print(f"  metadata/cost:             {len(METADATA_COLUMNS)}")
    print(f"Rows checked: {sample_n}")
    print("Max absolute diffs:")
    for col in sorted(max_diff):
        print(f"  {col}: {max_diff[col]:.12g}")

    if failures:
        print("FAILURES:")
        for failure in failures[:30]:
            print(failure)
        raise SystemExit(1)

    print("PASS look-ahead audit: model-safe columns match L1 at signal_ts only.")
    print("NOTE: execution and outcome columns must be excluded from model features.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

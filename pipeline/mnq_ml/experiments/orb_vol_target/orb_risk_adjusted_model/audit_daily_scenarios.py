#!/usr/bin/env python3
"""Audit MNQ ORB daily scenario features for lookahead and integrity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from build_daily_scenarios import FEATURE_COLUMNS  # noqa: E402
from common import project_path, write_json  # noqa: E402

CONFIG_PATH = SCRIPT_DIR / "config.json"
FORBIDDEN_FEATURE_TOKENS = [
    "signal",
    "entry",
    "exit",
    "pnl",
    "success",
    "label",
    "breakout",
    "r_multiple",
    "future",
    "target",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def close_enough(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def recompute_feature_row(day: pd.DataFrame, orb_minutes: int) -> dict[str, float | int]:
    quality = day["bar_data_quality_ok"].astype(bool)
    orb_mask = quality & (day["minutes_from_open"] > 0) & (day["minutes_from_open"] <= orb_minutes)
    orb = day.loc[orb_mask]
    if len(orb) != orb_minutes:
        raise ValueError(f"Expected {orb_minutes} OR bars, got {len(orb)}")

    orb_high = float(orb["high"].max())
    orb_low = float(orb["low"].min())
    orb_range = orb_high - orb_low
    orb_open = float(orb.iloc[0]["open"])
    orb_close = float(orb.iloc[-1]["close"])
    pre_60 = day.loc[quality & (day["minutes_from_open"] >= -60) & (day["minutes_from_open"] <= 0)]
    if pre_60.empty:
        pre_count = 0
        pre_return = 0.0
        pre_range = 0.0
        pre_volume = 0.0
    else:
        pre_count = int(len(pre_60))
        pre_return = float(pre_60.iloc[-1]["close"] - pre_60.iloc[0]["open"])
        pre_range = float(pre_60["high"].max() - pre_60["low"].min())
        pre_volume = float(pre_60["volume"].sum())

    return {
        "orb_range_pts": orb_range,
        "orb_body_pts": abs(orb_close - orb_open),
        "orb_direction_pts": orb_close - orb_open,
        "orb_close_position": (orb_close - orb_low) / orb_range,
        "orb_upper_wick_pts": orb_high - max(orb_open, orb_close),
        "orb_lower_wick_pts": min(orb_open, orb_close) - orb_low,
        "orb_volume_sum": float(orb["volume"].sum()),
        "orb_volume_mean": float(orb["volume"].mean()),
        "orb_volume_max": float(orb["volume"].max()),
        "pre_60m_bar_count": pre_count,
        "pre_60m_return_pts": pre_return,
        "pre_60m_range_pts": pre_range,
        "pre_60m_volume_sum": pre_volume,
        "ny_day_of_week": int(pd.Timestamp(day.iloc[0]["ny_date"]).dayofweek),
        "ny_month": int(pd.Timestamp(day.iloc[0]["ny_date"]).month),
    }


def expected_or_complete_dates(l1: pd.DataFrame, orb_minutes: int) -> set[object]:
    expected = set()
    for ny_date, day in l1.groupby("ny_date", sort=True):
        quality = day["bar_data_quality_ok"].astype(bool)
        orb_mask = quality & (day["minutes_from_open"] > 0) & (day["minutes_from_open"] <= orb_minutes)
        if int(orb_mask.sum()) == orb_minutes:
            expected.add(ny_date)
    return expected


def count_bad_time_order(dataset: pd.DataFrame, side_prefix: str) -> int:
    signal_col = f"{side_prefix}_signal_ts"
    entry_col = f"{side_prefix}_entry_ts"
    exit_col = f"{side_prefix}_exit_ts"
    occurred_col = f"{side_prefix}_breakout_occurred"
    occurred = dataset[occurred_col].astype(bool)
    timed = dataset.loc[
        occurred
        & dataset[signal_col].notna()
        & dataset[entry_col].notna()
        & dataset[exit_col].notna()
    ]
    if timed.empty:
        return 0
    bad = timed[(timed[signal_col] >= timed[entry_col]) | (timed[entry_col] > timed[exit_col])]
    return int(len(bad))


def label_null_failures(dataset: pd.DataFrame, side_prefix: str) -> dict[str, int]:
    occurred_col = f"{side_prefix}_breakout_occurred"
    success_col = f"{side_prefix}_success_2r"
    pnl_col = f"{side_prefix}_pnl_per_contract_usd"
    r_col = f"{side_prefix}_r_multiple"
    exit_reason_col = f"{side_prefix}_exit_reason"
    occurred = dataset[occurred_col].astype(bool)

    no_breakout = ~occurred
    simulated = occurred & dataset[success_col].notna()
    unsimulated = occurred & dataset[success_col].isna()

    failures = {
        "no_breakout_success_not_null": int(dataset.loc[no_breakout, success_col].notna().sum()),
        "no_breakout_pnl_not_null": int(dataset.loc[no_breakout, pnl_col].notna().sum()),
        "simulated_pnl_null": int(dataset.loc[simulated, pnl_col].isna().sum()),
        "simulated_r_null": int(dataset.loc[simulated, r_col].isna().sum()),
        "unsimulated_bad_exit_reason": int(
            (~dataset.loc[
                unsimulated,
                exit_reason_col,
            ].isin(["NO_NEXT_BAR", "BAD_ENTRY_BAR", "RISK_FILTERED", "NO_TIME_EXIT"])).sum()
        ),
    }
    return {key: value for key, value in failures.items() if value}


def main() -> None:
    args = parse_args()
    cfg = load_json(Path(args.config))
    dataset_path = project_path(cfg["outputs"]["daily_scenarios"])
    l1_path = project_path(cfg["inputs"]["l1_context"])
    audit_path = project_path(cfg["outputs"]["audit"])
    orb_minutes = int(cfg["scenario_contract"]["orb_minutes"])

    failures: dict[str, Any] = {}
    if not dataset_path.exists():
        raise SystemExit(f"Missing risk-adjusted dataset: {dataset_path}")
    if not l1_path.exists():
        raise SystemExit(f"Missing L1 context: {l1_path}")

    dataset = pd.read_parquet(dataset_path)
    l1 = pd.read_parquet(l1_path)
    dataset["ny_date"] = pd.to_datetime(dataset["ny_date"]).dt.date
    l1["ny_date"] = pd.to_datetime(l1["ny_date"]).dt.date
    l1["timestamp_utc"] = pd.to_datetime(l1["timestamp_utc"], utc=True)
    for col in [
        "up_signal_ts",
        "up_entry_ts",
        "up_exit_ts",
        "down_signal_ts",
        "down_entry_ts",
        "down_exit_ts",
    ]:
        if col in dataset.columns:
            dataset[col] = pd.to_datetime(dataset[col], utc=True)

    missing_features = [col for col in FEATURE_COLUMNS if col not in dataset.columns]
    if missing_features:
        failures["missing_features"] = missing_features

    forbidden_features = {
        col: [token for token in FORBIDDEN_FEATURE_TOKENS if token in col.lower()]
        for col in FEATURE_COLUMNS
        if any(token in col.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    }
    if forbidden_features:
        failures["forbidden_feature_names"] = forbidden_features

    duplicate_dates = int(dataset["ny_date"].duplicated().sum())
    if duplicate_dates:
        failures["duplicate_ny_dates"] = duplicate_dates

    expected_dates = expected_or_complete_dates(l1, orb_minutes)
    dataset_dates = set(dataset["ny_date"])
    missing_dates = sorted(expected_dates - dataset_dates)
    extra_dates = sorted(dataset_dates - expected_dates)
    if missing_dates or extra_dates:
        failures["scenario_coverage"] = {
            "missing_expected_or_complete_dates": [str(x) for x in missing_dates[:20]],
            "missing_count": len(missing_dates),
            "extra_unexpected_dates": [str(x) for x in extra_dates[:20]],
            "extra_count": len(extra_dates),
        }

    feature_nulls = dataset[FEATURE_COLUMNS].isna().sum()
    bad_feature_nulls = feature_nulls[feature_nulls > 0].to_dict()
    if bad_feature_nulls:
        failures["feature_nulls"] = {str(k): int(v) for k, v in bad_feature_nulls.items()}

    feature_mismatches: list[dict[str, Any]] = []
    l1_by_date = {date: day.sort_values("timestamp_utc").reset_index(drop=True) for date, day in l1.groupby("ny_date")}
    for row in dataset.itertuples(index=False):
        ny_date = row.ny_date
        day = l1_by_date.get(ny_date)
        if day is None:
            feature_mismatches.append({"ny_date": str(ny_date), "reason": "missing_l1_day"})
            continue
        try:
            expected = recompute_feature_row(day, orb_minutes)
        except Exception as exc:  # noqa: BLE001
            feature_mismatches.append({"ny_date": str(ny_date), "reason": str(exc)})
            continue
        for col in FEATURE_COLUMNS:
            actual = getattr(row, col)
            if not close_enough(actual, expected[col]):
                feature_mismatches.append(
                    {
                        "ny_date": str(ny_date),
                        "feature": col,
                        "actual": float(actual),
                        "expected": float(expected[col]),
                    }
                )
                break
    if feature_mismatches:
        failures["feature_recompute_mismatches"] = feature_mismatches[:20]
        failures["feature_recompute_mismatch_count"] = len(feature_mismatches)

    split_dates = dataset.groupby("split")["ny_date"].agg(["min", "max"]).astype(str).to_dict("index")
    train_bad = dataset[(dataset["split"] == "train") & (pd.to_datetime(dataset["ny_date"]) > pd.Timestamp(cfg["split"]["train_end"]))]
    validation_bad = dataset[
        (dataset["split"] == "validation")
        & (
            (pd.to_datetime(dataset["ny_date"]) < pd.Timestamp(cfg["split"]["validation_start"]))
            | (pd.to_datetime(dataset["ny_date"]) > pd.Timestamp(cfg["split"]["validation_end"]))
        )
    ]
    holdout_bad = dataset[(dataset["split"] == "holdout") & (pd.to_datetime(dataset["ny_date"]) < pd.Timestamp(cfg["split"]["holdout_start"]))]
    if len(train_bad) or len(validation_bad) or len(holdout_bad):
        failures["split_leakage"] = {
            "train_bad": int(len(train_bad)),
            "validation_bad": int(len(validation_bad)),
            "holdout_bad": int(len(holdout_bad)),
        }

    up_null_failures = label_null_failures(dataset, "up")
    down_null_failures = label_null_failures(dataset, "down")
    if up_null_failures or down_null_failures:
        failures["label_null_consistency"] = {
            "up": up_null_failures,
            "down": down_null_failures,
        }

    bad_time_order = {
        "up": count_bad_time_order(dataset, "up"),
        "down": count_bad_time_order(dataset, "down"),
    }
    bad_time_order = {key: value for key, value in bad_time_order.items() if value}
    if bad_time_order:
        failures["timestamp_order"] = bad_time_order

    label_like_columns = [
        col
        for col in dataset.columns
        if any(token in col.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    accidental_label_features = sorted(set(label_like_columns) & set(FEATURE_COLUMNS))
    if accidental_label_features:
        failures["label_like_features"] = accidental_label_features

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "dataset": str(dataset_path),
        "l1_context": str(l1_path),
        "rows": int(len(dataset)),
        "columns": int(len(dataset.columns)),
        "expected_or_complete_dates": int(len(expected_dates)),
        "dataset_dates": int(len(dataset_dates)),
        "feature_columns": FEATURE_COLUMNS,
        "forbidden_feature_tokens": FORBIDDEN_FEATURE_TOKENS,
        "lookahead_contract": {
            "feature_window": "pre-60m through opening range completion only",
            "orb_feature_minutes_from_open": "1..15",
            "pre_feature_minutes_from_open": "-60..0",
            "post_or_columns_are_labels_only": True,
        },
        "split_date_ranges": split_dates,
        "failures": failures,
    }
    write_json(audit_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit MNQ ORB breakout-quality feature dataset."""

from __future__ import annotations

import argparse
import datetime as dt
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

from build_breakout_quality_features import FEATURE_COLUMNS, FEATURE_FAMILIES, LABEL_COLUMNS, build_rows  # noqa: E402
from common import load_config as load_parent_config  # noqa: E402
from common import project_path, write_json  # noqa: E402

CONFIG_PATH = SCRIPT_DIR / "config.json"
FORBIDDEN_FEATURE_TOKENS = [
    "entry",
    "exit",
    "pnl",
    "success",
    "positive",
    "outcome",
    "label",
    "bucket",
    "future",
    "target",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def close_enough(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, dt.date) or isinstance(right, dt.date):
        return str(left) == str(right)
    if isinstance(left, pd.Timestamp) or isinstance(right, pd.Timestamp):
        return pd.Timestamp(left) == pd.Timestamp(right)
    if isinstance(left, str) or isinstance(right, str):
        return str(left) == str(right)
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    args = parse_args()
    cfg = load_json(Path(args.config))
    parent_cfg = load_parent_config()
    dataset_path = project_path(cfg["outputs"]["breakout_quality"])
    audit_path = project_path(cfg["outputs"]["breakout_quality_audit"])
    l1_path = project_path(cfg["inputs"]["l1_context"])

    failures: dict[str, Any] = {}
    if not dataset_path.exists():
        raise SystemExit(f"Missing breakout-quality dataset: {dataset_path}")
    if not l1_path.exists():
        raise SystemExit(f"Missing L1 context: {l1_path}")

    dataset = pd.read_parquet(dataset_path)
    l1 = pd.read_parquet(l1_path)
    l1["timestamp_utc"] = pd.to_datetime(l1["timestamp_utc"], utc=True)
    l1["ny_date"] = pd.to_datetime(l1["ny_date"]).dt.date
    dataset["ny_date"] = pd.to_datetime(dataset["ny_date"]).dt.date
    for col in ["signal_ts", "entry_ts", "exit_ts"]:
        dataset[col] = pd.to_datetime(dataset[col], utc=True)
    if "daily_confluence_feature_date" in dataset.columns:
        dataset["daily_confluence_feature_date"] = pd.to_datetime(dataset["daily_confluence_feature_date"]).dt.date

    missing_features = [col for col in FEATURE_COLUMNS if col not in dataset.columns]
    if missing_features:
        failures["missing_features"] = missing_features

    missing_labels = [col for col in LABEL_COLUMNS if col not in dataset.columns]
    if missing_labels:
        failures["missing_labels"] = missing_labels

    forbidden_features = {
        col: [token for token in FORBIDDEN_FEATURE_TOKENS if token in col.lower()]
        for col in FEATURE_COLUMNS
        if any(token in col.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    }
    if forbidden_features:
        failures["forbidden_feature_names"] = forbidden_features

    duplicate_events = int(dataset["event_id"].duplicated().sum())
    if duplicate_events:
        failures["duplicate_event_ids"] = duplicate_events

    feature_nulls = dataset[FEATURE_COLUMNS].isna().sum()
    bad_feature_nulls = feature_nulls[feature_nulls > 0].to_dict()
    if bad_feature_nulls:
        failures["feature_nulls"] = {str(k): int(v) for k, v in bad_feature_nulls.items()}

    label_nulls = dataset[LABEL_COLUMNS].isna().sum()
    bad_label_nulls = label_nulls[label_nulls > 0].to_dict()
    if bad_label_nulls:
        failures["label_nulls"] = {str(k): int(v) for k, v in bad_label_nulls.items()}

    bad_order = dataset[(dataset["signal_ts"] >= dataset["entry_ts"]) | (dataset["entry_ts"] > dataset["exit_ts"])]
    if len(bad_order):
        failures["timestamp_order"] = int(len(bad_order))

    if "daily_confluence_feature_date" not in dataset.columns:
        failures["missing_daily_confluence_feature_date"] = True
    else:
        daily_confluence_lookahead = dataset[dataset["daily_confluence_feature_date"] >= dataset["ny_date"]]
        if len(daily_confluence_lookahead):
            failures["daily_confluence_lookahead_violations"] = int(len(daily_confluence_lookahead))
        daily_confluence_null_dates = int(dataset["daily_confluence_feature_date"].isna().sum())
        if daily_confluence_null_dates:
            failures["daily_confluence_feature_date_nulls"] = daily_confluence_null_dates

    bad_side = dataset[~dataset["side"].isin(["UP", "DOWN"])]
    if len(bad_side):
        failures["bad_side_values"] = int(len(bad_side))

    bad_binary = dataset[~dataset["success_2r"].isin([0, 1])]
    if len(bad_binary):
        failures["bad_success_2r_values"] = int(len(bad_binary))

    if "positive_eod" in dataset.columns:
        bad_positive = dataset[~dataset["positive_eod"].isin([0, 1])]
        if len(bad_positive):
            failures["bad_positive_eod_values"] = int(len(bad_positive))
        positive_mismatch = dataset[(dataset["positive_eod"] != (dataset["pnl_per_contract_usd"] > 0).astype(int))]
        if len(positive_mismatch):
            failures["positive_eod_definition_mismatch"] = int(len(positive_mismatch))

    if "outcome_bucket" in dataset.columns:
        allowed_buckets = {"TP_2R", "POSITIVE_EOD", "NEGATIVE_EOD"}
        bad_bucket = dataset[~dataset["outcome_bucket"].isin(allowed_buckets)]
        if len(bad_bucket):
            failures["bad_outcome_bucket_values"] = int(len(bad_bucket))
        bucket_mismatch = dataset[
            ((dataset["success_2r"] == 1) & (dataset["outcome_bucket"] != "TP_2R"))
            | ((dataset["success_2r"] == 0) & (dataset["positive_eod"] == 1) & (dataset["outcome_bucket"] != "POSITIVE_EOD"))
            | ((dataset["success_2r"] == 0) & (dataset["positive_eod"] == 0) & (dataset["outcome_bucket"] != "NEGATIVE_EOD"))
        ]
        if len(bucket_mismatch):
            failures["outcome_bucket_definition_mismatch"] = int(len(bucket_mismatch))

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

    rebuilt = build_rows(l1, parent_cfg, cfg)
    rebuilt["ny_date"] = pd.to_datetime(rebuilt["ny_date"]).dt.date
    for col in ["signal_ts", "entry_ts", "exit_ts"]:
        rebuilt[col] = pd.to_datetime(rebuilt[col], utc=True)
    if "daily_confluence_feature_date" in rebuilt.columns:
        rebuilt["daily_confluence_feature_date"] = pd.to_datetime(rebuilt["daily_confluence_feature_date"]).dt.date
    compare_cols = ["event_id", "ny_date", "side", "signal_ts", "daily_confluence_feature_date"] + FEATURE_COLUMNS + LABEL_COLUMNS
    left = dataset[compare_cols].sort_values("event_id").reset_index(drop=True)
    right = rebuilt[compare_cols].sort_values("event_id").reset_index(drop=True)
    if len(left) != len(right):
        failures["rebuild_row_count_mismatch"] = {"dataset": int(len(left)), "rebuilt": int(len(right))}
    else:
        mismatches: list[dict[str, Any]] = []
        for idx in range(len(left)):
            for col in compare_cols:
                if not close_enough(left.at[idx, col], right.at[idx, col]):
                    mismatches.append(
                        {
                            "event_id": str(left.at[idx, "event_id"]),
                            "column": col,
                            "actual": str(left.at[idx, col]),
                            "expected": str(right.at[idx, col]),
                        }
                    )
                    break
        if mismatches:
            failures["rebuild_mismatch_count"] = len(mismatches)
            failures["rebuild_mismatches"] = mismatches[:20]

    leakage_smoke: dict[str, Any] = {}
    numeric_features = dataset[FEATURE_COLUMNS].select_dtypes(include=["number"])
    for label in ["success_2r", "positive_eod"]:
        if label not in dataset.columns:
            continue
        corrs = numeric_features.corrwith(dataset[label].astype(float)).abs().dropna()
        suspicious = corrs[corrs >= 0.98].sort_values(ascending=False).to_dict()
        if suspicious:
            leakage_smoke[label] = {str(k): float(v) for k, v in suspicious.items()}
    if leakage_smoke:
        failures["label_leakage_correlation_smoke"] = leakage_smoke

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "dataset": str(dataset_path),
        "l1_context": str(l1_path),
        "rows": int(len(dataset)),
        "columns": int(len(dataset.columns)),
        "feature_families": FEATURE_FAMILIES,
        "feature_columns": FEATURE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "forbidden_feature_tokens": FORBIDDEN_FEATURE_TOKENS,
        "lookahead_contract": {
            "feature_cutoff": "breakout candle close",
            "entry_open_allowed_as_feature": False,
            "entry_exit_pnl_columns_are_labels_or_metadata_only": True,
            "label_columns_must_not_be_used_as_features": True,
            "positive_eod_definition": "pnl_per_contract_usd > 0",
            "outcome_bucket_definition": "TP_2R, POSITIVE_EOD, NEGATIVE_EOD",
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

#!/usr/bin/env python3
"""Build the first MNQ ORB ML-filter dataset from the frozen rule baseline.

This is intentionally an overlay. It does not rebuild the ORB strategy and does
not change the rule-based baseline. It filters the sweep event file to Candidate
A and creates a model-ready table with only pre-entry features plus labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PARENT_DIR))

from common import project_path, write_json  # noqa: E402

CONFIG_PATH = SCRIPT_DIR / "config.json"

FEATURE_COLUMNS = [
    "signal_minutes_from_open",
    "orb_range_pts",
    "entry_risk_pts",
    "risk_per_contract_usd",
    "breakout_pts",
    "breakout_to_orb_range",
    "entry_risk_to_orb_range",
    "contracts_used",
    "ny_day_of_week",
    "ny_month",
]

METADATA_COLUMNS = [
    "event_id",
    "base_event_id",
    "ny_date",
    "signal_ts",
    "entry_ts",
    "exit_ts",
    "side",
    "orb_minutes",
    "side_mode",
    "exit_mode",
    "target_risk_usd",
    "exit_reason",
]

LABEL_COLUMNS = [
    "label_good_trade",
    "pnl_per_contract_usd",
    "pnl_usd",
    "r_multiple",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def assign_split(ny_date: pd.Timestamp, split_cfg: dict[str, str]) -> str:
    train_end = pd.Timestamp(split_cfg["train_end"]).date()
    validation_start = pd.Timestamp(split_cfg["validation_start"]).date()
    validation_end = pd.Timestamp(split_cfg["validation_end"]).date()
    holdout_start = pd.Timestamp(split_cfg["holdout_start"]).date()
    date_value = pd.Timestamp(ny_date).date()

    if date_value <= train_end:
        return "train"
    if validation_start <= date_value <= validation_end:
        return "validation"
    if date_value >= holdout_start:
        return "holdout"
    return "unused"


def build_dataset(cfg: dict[str, Any]) -> pd.DataFrame:
    source_path = project_path(cfg["inputs"]["sweep_events"])
    if not source_path.exists():
        raise SystemExit(f"Missing sweep events: {source_path}")

    candidate = cfg["baseline_candidate"]
    events = pd.read_parquet(source_path)
    rows = events[
        (events["orb_minutes"] == int(candidate["orb_minutes"]))
        & (events["side_mode"] == candidate["side_mode"])
        & (events["exit_mode"] == candidate["exit_mode"])
        & (events["target_risk_usd"] == int(candidate["target_risk_usd"]))
    ].copy()
    if rows.empty:
        raise SystemExit(f"No rows found for candidate: {candidate}")

    rows["ny_date"] = pd.to_datetime(rows["ny_date"]).dt.date
    rows["signal_ts"] = pd.to_datetime(rows["signal_ts"], utc=True)
    rows["entry_ts"] = pd.to_datetime(rows["entry_ts"], utc=True)
    rows["exit_ts"] = pd.to_datetime(rows["exit_ts"], utc=True)
    rows = rows.sort_values("signal_ts").reset_index(drop=True)

    rows["label_good_trade"] = (rows["pnl_per_contract_usd"] > 0).astype(int)
    rows["r_multiple"] = rows["pnl_per_contract_usd"] / rows["risk_per_contract_usd"]
    rows["breakout_pts"] = rows["entry_price"] - rows["orb_high"]
    rows["breakout_to_orb_range"] = rows["breakout_pts"] / rows["orb_range_pts"]
    rows["entry_risk_to_orb_range"] = rows["entry_risk_pts"] / rows["orb_range_pts"]
    rows["ny_day_of_week"] = pd.to_datetime(rows["ny_date"]).dt.dayofweek.astype(int)
    rows["ny_month"] = pd.to_datetime(rows["ny_date"]).dt.month.astype(int)
    rows["split"] = rows["ny_date"].apply(lambda x: assign_split(x, cfg["split"]))

    output_columns = METADATA_COLUMNS + ["split"] + FEATURE_COLUMNS + LABEL_COLUMNS
    dataset = rows[output_columns].copy()

    required_nulls = dataset[FEATURE_COLUMNS + ["label_good_trade"]].isna().sum()
    bad_nulls = required_nulls[required_nulls > 0]
    if not bad_nulls.empty:
        raise SystemExit(f"Unexpected nulls in ML dataset: {bad_nulls.to_dict()}")

    return dataset


def main() -> None:
    args = parse_args()
    cfg = load_json(Path(args.config))
    output_path = project_path(cfg["outputs"]["dataset"])
    manifest_path = project_path(cfg["outputs"]["manifest"])
    model_dir = project_path(cfg["outputs"]["model_dir"])

    if output_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing dataset: {output_path}")

    dataset = build_dataset(cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)

    split_counts = dataset["split"].value_counts().sort_index().to_dict()
    split_pnl = dataset.groupby("split")["pnl_usd"].sum().sort_index().to_dict()
    split_win_rate = dataset.groupby("split")["label_good_trade"].mean().sort_index().to_dict()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "experiment": cfg["experiment"],
        "baseline_candidate": cfg["baseline_candidate"],
        "input": str(project_path(cfg["inputs"]["sweep_events"])),
        "output": str(output_path),
        "model_dir": str(model_dir),
        "rows": int(len(dataset)),
        "columns": int(len(dataset.columns)),
        "feature_columns": FEATURE_COLUMNS,
        "metadata_columns": METADATA_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "split_pnl_usd": {str(k): float(v) for k, v in split_pnl.items()},
        "split_win_rate": {str(k): float(v) for k, v in split_win_rate.items()},
        "min_signal_ts": dataset["signal_ts"].min().isoformat(),
        "max_signal_ts": dataset["signal_ts"].max().isoformat(),
        "no_lookahead_note": "Feature columns use rule state known at signal/entry time; exit/pnl fields are labels or metadata only.",
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

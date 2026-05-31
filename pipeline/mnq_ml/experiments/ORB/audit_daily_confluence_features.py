#!/usr/bin/env python3
"""Audit MNQ daily confluence feature file."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_daily_confluence_features import DAILY_CONFLUENCE_FEATURES  # noqa: E402
from common import project_path, write_json  # noqa: E402

CONFIG_PATH = SCRIPT_DIR / "config.json"
DEFAULT_SOURCE_DB = "data/Level_0_Raw/yfinance_daily.duckdb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--source-db", default=DEFAULT_SOURCE_DB)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    cfg = load_json(Path(args.config))
    feature_path = project_path(cfg["outputs"]["daily_confluence"])
    audit_path = project_path(cfg["outputs"]["daily_confluence_audit"])
    l1_path = project_path(cfg["outputs"]["l1_context"])
    source_db = project_path(args.source_db)

    failures: dict[str, Any] = {}
    if not feature_path.exists():
        raise SystemExit(f"Missing daily confluence features: {feature_path}")
    if not l1_path.exists():
        raise SystemExit(f"Missing L1 context: {l1_path}")
    if not source_db.exists():
        raise SystemExit(f"Missing yfinance daily DB: {source_db}")

    features = pd.read_parquet(feature_path)
    l1_dates = pd.read_parquet(l1_path, columns=["ny_date"])
    features["ny_date"] = pd.to_datetime(features["ny_date"]).dt.date
    features["daily_confluence_feature_date"] = pd.to_datetime(features["daily_confluence_feature_date"]).dt.date
    l1_dates["ny_date"] = pd.to_datetime(l1_dates["ny_date"]).dt.date
    expected_dates = set(l1_dates["ny_date"].unique())

    missing_cols = [col for col in ["ny_date", "daily_confluence_feature_date", *DAILY_CONFLUENCE_FEATURES] if col not in features.columns]
    if missing_cols:
        failures["missing_columns"] = missing_cols

    duplicate_dates = int(features["ny_date"].duplicated().sum())
    if duplicate_dates:
        failures["duplicate_ny_dates"] = duplicate_dates

    feature_dates = set(features["ny_date"].unique())
    if feature_dates != expected_dates:
        failures["date_coverage"] = {
            "missing_dates": sorted(str(x) for x in expected_dates - feature_dates)[:20],
            "extra_dates": sorted(str(x) for x in feature_dates - expected_dates)[:20],
            "missing_count": len(expected_dates - feature_dates),
            "extra_count": len(feature_dates - expected_dates),
        }

    nulls = features[DAILY_CONFLUENCE_FEATURES].isna().sum()
    bad_nulls = nulls[nulls > 0].to_dict()
    if bad_nulls:
        failures["feature_nulls"] = {str(k): int(v) for k, v in bad_nulls.items()}

    lookahead = features[features["daily_confluence_feature_date"] >= features["ny_date"]]
    if len(lookahead):
        failures["lookahead_violations"] = int(len(lookahead))

    con = duckdb.connect(str(source_db), read_only=True)
    db_dates = con.execute(
        """
        select distinct date
        from daily_ohlcv
        where symbol in ('SPY', 'QQQ', 'VIX', 'TNX', 'DXY')
        """
    ).fetchdf()
    con.close()
    db_date_set = set(pd.to_datetime(db_dates["date"]).dt.date)
    missing_source_dates = sorted(str(x) for x in set(features["daily_confluence_feature_date"]) - db_date_set)
    if missing_source_dates:
        failures["feature_dates_missing_from_source_db"] = missing_source_dates[:20]

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "feature_path": str(feature_path),
        "source_db": str(source_db),
        "rows": int(len(features)),
        "columns": int(len(features.columns)),
        "feature_count": int(len(DAILY_CONFLUENCE_FEATURES)),
        "min_ny_date": str(features["ny_date"].min()),
        "max_ny_date": str(features["ny_date"].max()),
        "min_feature_date": str(features["daily_confluence_feature_date"].min()),
        "max_feature_date": str(features["daily_confluence_feature_date"].max()),
        "lookahead_contract": "daily_confluence_feature_date must be strictly earlier than ny_date",
        "failures": failures,
    }
    write_json(audit_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

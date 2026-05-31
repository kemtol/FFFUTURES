#!/usr/bin/env python3
"""Look-ahead audit scaffold for MNQ M1 events."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from common import load_config, project_path

WHITELIST_PATH = Path(__file__).resolve().parent / "training_feature_whitelist.json"


def run_audit() -> dict:
    cfg = load_config()
    events_path = project_path(cfg["outputs"]["events"])
    whitelist = json.loads(WHITELIST_PATH.read_text())
    if not events_path.exists():
        return {
            "status": "BLOCKED",
            "reason": f"Missing MNQ events datamart: {events_path}",
            "lookahead_safe": False,
        }

    df = pd.read_parquet(events_path)
    required = {"signal_ts", "entry_ts"}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        return {
            "status": "FAIL",
            "reason": f"Missing timing columns: {missing_required}",
            "lookahead_safe": False,
        }

    forbidden = [
        feature
        for feature in whitelist["features"]
        for pattern in whitelist.get("forbidden_patterns", [])
        if pattern in feature
    ]
    timing_bad = int((pd.to_datetime(df["entry_ts"], utc=True) <= pd.to_datetime(df["signal_ts"], utc=True)).sum())
    status = "PASS" if not forbidden and timing_bad == 0 else "FAIL"
    return {
        "status": status,
        "lookahead_safe": status == "PASS",
        "forbidden_whitelist_features": sorted(set(forbidden)),
        "entry_not_after_signal_rows": timing_bad,
        "rows_checked": int(len(df)),
    }


def main() -> int:
    report = run_audit()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

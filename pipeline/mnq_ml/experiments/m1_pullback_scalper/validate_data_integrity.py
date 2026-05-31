#!/usr/bin/env python3
"""Validate MNQ M1 L1/L2 integrity before training."""

from __future__ import annotations

import sys

import pandas as pd

from build_l1_context import REQUIRED_L1_COLUMNS
from build_m1_events import REQUIRED_L2_COLUMNS
from common import load_config, project_path


def main() -> int:
    cfg = load_config()
    l1_path = project_path(cfg["outputs"]["l1_context"])
    l2_path = project_path(cfg["outputs"]["events"])
    if not l1_path.exists():
        raise SystemExit(f"Missing MNQ L1 context: {l1_path}")
    if not l2_path.exists():
        raise SystemExit(f"Missing MNQ L2 events: {l2_path}")

    l1 = pd.read_parquet(l1_path)
    l2 = pd.read_parquet(l2_path)
    missing_l1 = [c for c in REQUIRED_L1_COLUMNS if c not in l1.columns]
    missing_l2 = [c for c in REQUIRED_L2_COLUMNS if c not in l2.columns]
    if missing_l1 or missing_l2:
        raise SystemExit({"missing_l1": missing_l1, "missing_l2": missing_l2})

    l1["timestamp_utc"] = pd.to_datetime(l1["timestamp_utc"], utc=True)
    l2["signal_ts"] = pd.to_datetime(l2["signal_ts"], utc=True)
    l2["entry_ts"] = pd.to_datetime(l2["entry_ts"], utc=True)

    if l1[REQUIRED_L1_COLUMNS].isna().any().any():
        raise SystemExit("L1 required columns contain nulls")
    if l2[REQUIRED_L2_COLUMNS].isna().any().any():
        raise SystemExit("L2 required columns contain nulls")
    if (l2["entry_ts"] <= l2["signal_ts"]).any():
        raise SystemExit("L2 contains entry_ts <= signal_ts")

    print("PASS MNQ data integrity")
    print(f"L1 rows: {len(l1):,}")
    print(f"L2 rows: {len(l2):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

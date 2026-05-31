#!/usr/bin/env python3
"""P0 gate for MNQ ORB volatility-targeted datamart."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pandas as pd

import audit_l1_context
from build_l1_context import REQUIRED_L1_COLUMNS
from build_orb_events import REQUIRED_L2_COLUMNS
from common import assert_mnq_namespaces, load_config, project_path, write_json


def main() -> int:
    cfg = load_config()
    assert_mnq_namespaces(cfg)
    l1_path = project_path(cfg["outputs"]["l1_context"])
    l2_path = project_path(cfg["outputs"]["events"])
    report_path = project_path(cfg["outputs"]["training_gate_report"])

    if not l1_path.exists() or not l2_path.exists():
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "BLOCKED",
            "training_allowed": False,
            "reason": "L1/L2 missing",
            "l1_exists": l1_path.exists(),
            "l2_exists": l2_path.exists(),
        }
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    l1 = pd.read_parquet(l1_path)
    l2 = pd.read_parquet(l2_path)
    l1_audit = audit_l1_context.audit()
    if l1_audit.get("status") != "PASS":
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "FAIL",
            "training_allowed": False,
            "reason": "L1 audit failed",
            "l1_audit": l1_audit,
        }
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    missing_l1 = [c for c in REQUIRED_L1_COLUMNS if c not in l1.columns]
    missing_l2 = [c for c in REQUIRED_L2_COLUMNS if c not in l2.columns]
    if missing_l1 or missing_l2:
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "FAIL",
            "training_allowed": False,
            "missing_l1": missing_l1,
            "missing_l2": missing_l2,
        }
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    l1["timestamp_utc"] = pd.to_datetime(l1["timestamp_utc"], utc=True)
    for col in ["signal_ts", "entry_ts", "exit_ts"]:
        l2[col] = pd.to_datetime(l2[col], utc=True)
    bad_timing = int((l2["entry_ts"] <= l2["signal_ts"]).sum())
    multi_per_day = int(l2["ny_date"].duplicated().sum())
    bad_events = bad_timing + multi_per_day
    null_l2 = {k: int(v) for k, v in l2[REQUIRED_L2_COLUMNS].isna().sum().items() if int(v) > 0}

    status = "PASS" if bad_events == 0 and not null_l2 else "FAIL"
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "training_allowed": status == "PASS",
        "l1_rows": int(len(l1)),
        "l2_rows": int(len(l2)),
        "orb_complete_days": int(l1.loc[l1["orb_complete"], "ny_date"].nunique()),
        "bad_timing_rows": bad_timing,
        "multi_event_days": multi_per_day,
        "l2_null_required_columns": null_l2,
        "win_rate": float(l2["label"].mean()) if not l2.empty else 0.0,
        "avg_pnl_per_contract_usd": float(l2["pnl_per_contract_usd"].mean()) if not l2.empty else 0.0,
        "total_pnl_vol_target_usd": float(l2["pnl_vol_target_usd"].sum()) if not l2.empty else 0.0,
        "l1_audit": {
            "status": l1_audit.get("status"),
            "rows": l1_audit.get("rows"),
            "orb_complete_days": l1_audit.get("orb_complete_days"),
            "gaps_gt_expected": l1_audit.get("continuity", {}).get("gaps_gt_expected"),
        },
        "live_isolation": {
            "touches_pipeline_live": False,
            "touches_model_super_structure": False,
            "touches_gold_datamarts": False
        },
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Hard training-data gate for MNQ M1 pullback research."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import audit_lookahead
from build_l1_context import REQUIRED_L1_COLUMNS
from build_m1_events import REQUIRED_L2_COLUMNS
from common import assert_mnq_namespaces, load_config, project_path, source_status, write_json

WHITELIST_PATH = Path(__file__).resolve().parent / "training_feature_whitelist.json"


def blocked(reason: str, cfg: dict, extra: dict | None = None) -> dict:
    payload = {
        "status": "BLOCKED",
        "training_allowed": False,
        "reason": reason,
        "live_isolation": {
            "touches_pipeline_live": False,
            "touches_model_super_structure": False,
            "touches_gold_datamarts": False
        },
    }
    if extra:
        payload.update(extra)
    write_json(project_path(cfg["outputs"]["training_gate_report"]), payload)
    return payload


def main() -> int:
    cfg = load_config()
    assert_mnq_namespaces(cfg)
    src = source_status(cfg)
    l1_path = project_path(cfg["outputs"]["l1_context"])
    l2_path = project_path(cfg["outputs"]["events"])

    if not src["ready"]:
        report = blocked("MNQ L0 source is not marked ready in config.json", cfg, {"source": src})
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    if not src["exists"]:
        report = blocked("MNQ L0 source file is missing", cfg, {"source": src})
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    if not l1_path.exists() or not l2_path.exists():
        report = blocked(
            "MNQ L1/L2 outputs are missing",
            cfg,
            {"l1_exists": l1_path.exists(), "l2_exists": l2_path.exists()},
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    l1 = pd.read_parquet(l1_path)
    l2 = pd.read_parquet(l2_path)
    whitelist = json.loads(WHITELIST_PATH.read_text())

    missing_l1 = [c for c in REQUIRED_L1_COLUMNS if c not in l1.columns]
    missing_l2 = [c for c in REQUIRED_L2_COLUMNS if c not in l2.columns]
    missing_features = [c for c in whitelist["features"] if c not in l2.columns]
    if missing_l1 or missing_l2 or missing_features:
        report = {
            "status": "FAIL",
            "training_allowed": False,
            "missing_l1": missing_l1,
            "missing_l2": missing_l2,
            "missing_features": missing_features,
        }
        write_json(project_path(cfg["outputs"]["training_gate_report"]), report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    audit = audit_lookahead.run_audit()
    if audit["status"] != "PASS":
        report = {
            "status": "FAIL",
            "training_allowed": False,
            "lookahead_audit": audit,
        }
        write_json(project_path(cfg["outputs"]["training_gate_report"]), report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    report = {
        "status": "PASS",
        "training_allowed": True,
        "l1_rows": int(len(l1)),
        "l2_rows": int(len(l2)),
        "feature_count": int(len(whitelist["features"])),
        "lookahead_audit": audit,
        "live_isolation": {
            "touches_pipeline_live": False,
            "touches_model_super_structure": False,
            "touches_gold_datamarts": False
        },
    }
    write_json(project_path(cfg["outputs"]["training_gate_report"]), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

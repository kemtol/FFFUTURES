#!/usr/bin/env python3
"""Audit MNQ ORB L1 context from right-labeled L0 M1 bars."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import duckdb
import pandas as pd

from build_l1_context import REQUIRED_L1_COLUMNS
from common import assert_mnq_namespaces, load_config, project_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true", default=True)
    return parser.parse_args()


def fail_report(reason: str, extra: dict | None = None) -> dict:
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL",
        "l1_audit_pass": False,
        "reason": reason,
    }
    if extra:
        report.update(extra)
    return report


def load_l0_source(cfg: dict) -> pd.DataFrame:
    source = cfg["source"]
    db_path = project_path(source["db"])
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(
        f"""
        select
          timestamp_utc + interval '1 minute' as timestamp_utc,
          open,
          high,
          low,
          close,
          volume
        from {source["table"]}
        order by timestamp_utc
        """
    ).fetchdf()
    con.close()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def audit() -> dict:
    cfg = load_config()
    assert_mnq_namespaces(cfg)
    l1_path = project_path(cfg["outputs"]["l1_context"])
    manifest_path = project_path(cfg["outputs"]["l1_manifest"])
    source_path = project_path(cfg["source"]["db"])
    if not l1_path.exists():
        return fail_report("Missing L1 context", {"l1_path": str(l1_path)})
    if not manifest_path.exists():
        return fail_report("Missing L1 manifest", {"manifest_path": str(manifest_path)})
    if not source_path.exists():
        return fail_report("Missing L0 source", {"source_path": str(source_path)})

    l1 = pd.read_parquet(l1_path)
    manifest = json.loads(manifest_path.read_text())
    missing = [c for c in REQUIRED_L1_COLUMNS if c not in l1.columns]
    if missing:
        return fail_report("L1 schema missing required columns", {"missing_columns": missing})

    l1["timestamp_utc"] = pd.to_datetime(l1["timestamp_utc"], utc=True)
    tf_minutes = int(cfg["session"]["decision_timeframe_minutes"])
    expected_step_seconds = tf_minutes * 60
    expected_orb_bars = 30 // tf_minutes
    hard_cols = [
        "timestamp_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source_bar_count",
        "contains_source_gap",
        "ny_date",
        "ny_time",
        "bar_data_quality_ok",
    ]
    hard_nulls = {k: int(v) for k, v in l1[hard_cols].isna().sum().items() if int(v) > 0}
    duplicate_timestamps = int(l1["timestamp_utc"].duplicated().sum())
    bad_ohlc = int(
        (
            (l1["high"] < l1[["open", "close", "low"]].max(axis=1))
            | (l1["low"] > l1[["open", "close", "high"]].min(axis=1))
            | (l1["volume"] < 0)
        ).sum()
    )
    diffs = l1["timestamp_utc"].sort_values().diff().dt.total_seconds().dropna()
    continuity = {
        "median_step_seconds": float(diffs.median()) if not diffs.empty else 0.0,
        "gaps_gt_expected": int((diffs > expected_step_seconds).sum()),
        "max_gap_seconds": int(diffs.max()) if not diffs.empty else 0,
    }

    l0 = load_l0_source(cfg)
    l0_summary = {
        "rows": int(len(l0)),
        "min_ts": l0["timestamp_utc"].min().isoformat() if not l0.empty else None,
        "max_ts": l0["timestamp_utc"].max().isoformat() if not l0.empty else None,
    }
    manifest_checks = {
        "rows_match": int(manifest.get("rows", -1)) == len(l1),
        "min_ts_match": manifest.get("min_ts") == (l1["timestamp_utc"].min().isoformat() if not l1.empty else None),
        "max_ts_match": manifest.get("max_ts") == (l1["timestamp_utc"].max().isoformat() if not l1.empty else None),
        "output_match": manifest.get("output") == str(l1_path),
    }

    quality_mismatch = int(
        (
            l1["bar_data_quality_ok"].astype(bool)
            != ((l1["source_bar_count"].astype(int) == tf_minutes) & (~l1["contains_source_gap"].astype(bool)))
        ).sum()
    )
    orb = l1.groupby("ny_date").agg(
        orb_complete=("orb_complete", "max"),
        orb_bar_count=("orb_bar_count", "max"),
        eligible_after_or=("eligible_after_or", "sum"),
    )
    complete_days = int(orb["orb_complete"].sum())
    incomplete_with_eligible = int(((~orb["orb_complete"].astype(bool)) & (orb["eligible_after_or"] > 0)).sum())
    complete_wrong_count = int(((orb["orb_complete"].astype(bool)) & (orb["orb_bar_count"] != expected_orb_bars)).sum())
    pre_or_with_or_values = int(
        (
            (l1["ny_time"] <= cfg["session"]["orb_end"])
            & (l1[["orb_high", "orb_low", "orb_range_pts"]].notna().any(axis=1))
        ).sum()
    )
    eligible_before_or = int((l1["eligible_after_or"] & (l1["ny_time"] <= cfg["session"]["orb_end"])).sum())
    eligible_bad_quality = int((l1["eligible_after_or"] & (~l1["bar_data_quality_ok"].astype(bool))).sum())

    failures = {}
    if hard_nulls:
        failures["hard_nulls"] = hard_nulls
    if duplicate_timestamps:
        failures["duplicate_timestamps"] = duplicate_timestamps
    if bad_ohlc:
        failures["bad_ohlc_rows"] = bad_ohlc
    if continuity["median_step_seconds"] != float(expected_step_seconds):
        failures["bad_median_step_seconds"] = continuity["median_step_seconds"]
    if quality_mismatch:
        failures["quality_flag_mismatches"] = quality_mismatch
    if incomplete_with_eligible:
        failures["incomplete_orb_days_with_eligible_rows"] = incomplete_with_eligible
    if complete_wrong_count:
        failures["complete_orb_days_wrong_bar_count"] = complete_wrong_count
    if pre_or_with_or_values:
        failures["pre_or_rows_with_or_values"] = pre_or_with_or_values
    if eligible_before_or:
        failures["eligible_rows_before_or_end"] = eligible_before_or
    if eligible_bad_quality:
        failures["eligible_rows_bad_quality"] = eligible_bad_quality
    if not all(manifest_checks.values()):
        failures["manifest_checks"] = manifest_checks

    status = "PASS" if not failures else "FAIL"
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "l1_audit_pass": status == "PASS",
        "l1_path": str(l1_path),
        "source_path": str(source_path),
        "rows": int(len(l1)),
        "columns": int(len(l1.columns)),
        "min_ts": l1["timestamp_utc"].min().isoformat() if not l1.empty else None,
        "max_ts": l1["timestamp_utc"].max().isoformat() if not l1.empty else None,
        "hard_nulls": hard_nulls,
        "duplicate_timestamps": duplicate_timestamps,
        "bad_ohlc_rows": bad_ohlc,
        "continuity": continuity,
        "expected_step_seconds": expected_step_seconds,
        "expected_orb_bars": expected_orb_bars,
        "l0_summary": l0_summary,
        "manifest_checks": manifest_checks,
        "quality_flag_mismatches": quality_mismatch,
        "ny_days": int(l1["ny_date"].nunique()),
        "orb_complete_days": complete_days,
        "incomplete_orb_days_with_eligible_rows": incomplete_with_eligible,
        "complete_orb_days_wrong_bar_count": complete_wrong_count,
        "pre_or_rows_with_or_values": pre_or_with_or_values,
        "eligible_rows_before_or_end": eligible_before_or,
        "eligible_rows_bad_quality": eligible_bad_quality,
        "failures": failures,
    }


def main() -> int:
    cfg = load_config()
    report = audit()
    report_path = project_path(cfg["outputs"]["l1_manifest"]).with_name("l1_audit.json")
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

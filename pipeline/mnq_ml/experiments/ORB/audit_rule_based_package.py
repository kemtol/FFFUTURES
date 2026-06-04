#!/usr/bin/env python3
"""Audit the frozen MNQ ORB rule-based model package."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod"
MODEL_DIR = ROOT / "model/MNQ/ORB/rule_based_15m_long_tp2r_eod"
REPORT_PATH = DATA_DIR / "package_gate.json"

L0_CONTINUITY_REPORT = ROOT / "data/Level_0_Raw/MNQ_1m_continuity_report.json"
L0_PARITY_REPORT = ROOT / "data/Level_0_Raw/MNQ_yfinance_timeframe_parity_report.json"
L1_AUDIT = ROOT / "data/Level_1_Features/mnq/ORB/l1_audit.json"
DAILY_CONFLUENCE_AUDIT = ROOT / "data/Level_1_Features/mnq/ORB/daily_confluence_audit.json"
ST_REGIME_MANIFEST = DATA_DIR / "supertrend_regime_manifest.json"
ST_VARIANT_MANIFEST = DATA_DIR / "supertrend_variant_comparison_manifest.json"

REQUIRED_EVENT_COLUMNS = [
    "base_event_id",
    "ny_date",
    "orb_minutes",
    "orb_end",
    "side_mode",
    "exit_mode",
    "exit_reason",
    "side",
    "signal_ts",
    "entry_ts",
    "exit_ts",
    "signal_minutes_from_open",
    "orb_high",
    "orb_low",
    "orb_range_pts",
    "entry_price",
    "exit_price",
    "stop_reference",
    "entry_risk_pts",
    "risk_per_contract_usd",
    "pnl_per_contract_usd",
    "label",
    "target_risk_usd",
    "contracts_float",
    "contracts_floor",
    "contracts_used",
    "pnl_usd",
    "event_id",
    "strategy_id",
    "pnl_net_usd",
    "commission_round_turn_usd",
    "commission_paid_usd",
    "slippage_ticks_per_side",
    "slippage_round_turn_usd_per_contract",
    "modeled_slippage_usd",
    "pnl_before_commission_usd",
    "strategic_sl_used",
    "stop_reference_role",
]

REQUIRED_ARTIFACTS = [
    DATA_DIR / "events.parquet",
    DATA_DIR / "summary.json",
    DATA_DIR / "manifest.json",
    DATA_DIR / "supertrend_regime_features.parquet",
    DATA_DIR / "supertrend_regime_manifest.json",
    DATA_DIR / "supertrend_variant_comparison_manifest.json",
    DATA_DIR / "short_reversal_switch_comparison_manifest.json",
    MODEL_DIR / "REPORT.md",
    MODEL_DIR / "README.md",
    MODEL_DIR / "manifest.json",
    MODEL_DIR / "supertrend_variant_comparison.csv",
    MODEL_DIR / "short_reversal_switch_comparison.csv",
    L0_CONTINUITY_REPORT,
    L0_PARITY_REPORT,
    L1_AUDIT,
    DAILY_CONFLUENCE_AUDIT,
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def add_failure(failures: dict[str, Any], key: str, value: Any) -> None:
    if value:
        failures[key] = value


def audit_events(events: pd.DataFrame, summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    failures: dict[str, Any] = {}
    warnings: dict[str, Any] = {}

    missing = [col for col in REQUIRED_EVENT_COLUMNS if col not in events.columns]
    add_failure(failures, "missing_event_columns", missing)
    if missing:
        return failures, warnings

    for col in ["signal_ts", "entry_ts", "exit_ts"]:
        events[col] = pd.to_datetime(events[col], utc=True)

    required_nulls = {
        col: int(count)
        for col, count in events[REQUIRED_EVENT_COLUMNS].isna().sum().items()
        if int(count) > 0
    }
    add_failure(failures, "required_event_nulls", required_nulls)
    add_failure(failures, "duplicate_event_ids", int(events["event_id"].duplicated().sum()))
    add_failure(failures, "duplicate_ny_dates", int(events["ny_date"].duplicated().sum()))
    add_failure(failures, "entry_not_after_signal", int((events["entry_ts"] <= events["signal_ts"]).sum()))
    add_failure(failures, "exit_before_entry", int((events["exit_ts"] < events["entry_ts"]).sum()))

    entry_delta = (events["entry_ts"] - events["signal_ts"]).dt.total_seconds()
    add_failure(failures, "entry_not_next_m1_open", int((entry_delta != 60).sum()))

    constant_checks = {
        "orb_minutes": sorted(events["orb_minutes"].dropna().unique().tolist()),
        "side_mode": sorted(events["side_mode"].dropna().unique().tolist()),
        "exit_mode": sorted(events["exit_mode"].dropna().unique().tolist()),
        "target_risk_usd": sorted(events["target_risk_usd"].dropna().unique().tolist()),
        "side": sorted(events["side"].dropna().unique().tolist()),
        "strategy_id": sorted(events["strategy_id"].dropna().unique().tolist()),
        "strategic_sl_used": sorted(events["strategic_sl_used"].dropna().unique().tolist()),
        "stop_reference_role": sorted(events["stop_reference_role"].dropna().unique().tolist()),
    }
    expected = {
        "orb_minutes": [15],
        "side_mode": ["long"],
        "exit_mode": ["tp_2r_or_time"],
        "target_risk_usd": [500],
        "side": ["LONG"],
        "strategy_id": ["rule_based_15m_long_tp2r_eod"],
        "strategic_sl_used": [False],
        "stop_reference_role": ["position_sizing_only"],
    }
    bad_constants = {k: v for k, v in constant_checks.items() if v != expected[k]}
    add_failure(failures, "unexpected_strategy_constants", bad_constants)

    add_failure(failures, "non_positive_entry_risk_rows", int((events["entry_risk_pts"] <= 0).sum()))
    add_failure(failures, "contracts_under_1_rows", int((events["contracts_used"] < 1).sum()))
    add_failure(failures, "contracts_over_20_rows", int((events["contracts_used"] > 20).sum()))

    perf = summary.get("performance", {})
    signal_range = summary.get("signal_range", {})
    summary_mismatches = {}
    if int(perf.get("trades", -1)) != len(events):
        summary_mismatches["performance.trades"] = {
            "summary": perf.get("trades"),
            "events": len(events),
        }
    pnl_diff = abs(float(perf.get("total_pnl_usd", 0.0)) - float(events["pnl_net_usd"].sum()))
    if pnl_diff > 0.01:
        summary_mismatches["performance.total_pnl_usd"] = pnl_diff
    min_signal = events["signal_ts"].min().isoformat()
    max_signal = events["signal_ts"].max().isoformat()
    if signal_range.get("min_signal_ts") != min_signal:
        summary_mismatches["signal_range.min_signal_ts"] = {
            "summary": signal_range.get("min_signal_ts"),
            "events": min_signal,
        }
    if signal_range.get("max_signal_ts") != max_signal:
        summary_mismatches["signal_range.max_signal_ts"] = {
            "summary": signal_range.get("max_signal_ts"),
            "events": max_signal,
        }
    add_failure(failures, "summary_mismatches", summary_mismatches)

    if len(events) == 0:
        add_failure(failures, "empty_events", True)
    if events["signal_ts"].max() < pd.Timestamp("2026-05-01", tz="UTC"):
        warnings["stale_signal_range"] = str(events["signal_ts"].max())

    return failures, warnings


def main() -> int:
    missing_artifacts = [rel(path) for path in REQUIRED_ARTIFACTS if not path.exists()]
    failures: dict[str, Any] = {}
    warnings: dict[str, Any] = {}
    add_failure(failures, "missing_artifacts", missing_artifacts)

    events_path = DATA_DIR / "events.parquet"
    summary_path = DATA_DIR / "summary.json"
    events = pd.read_parquet(events_path) if events_path.exists() else pd.DataFrame()
    summary = read_json(summary_path)
    event_failures, event_warnings = audit_events(events, summary) if not events.empty else ({"empty_events": True}, {})
    failures.update(event_failures)
    warnings.update(event_warnings)

    l0_continuity = read_json(L0_CONTINUITY_REPORT)
    l1_audit = read_json(L1_AUDIT)
    daily_audit = read_json(DAILY_CONFLUENCE_AUDIT)
    l0_parity = read_json(L0_PARITY_REPORT)
    st_regime = read_json(ST_REGIME_MANIFEST)
    st_variant = read_json(ST_VARIANT_MANIFEST)

    if l0_continuity and not bool(l0_continuity.get("hard_integrity_pass")):
        failures["l0_hard_integrity"] = l0_continuity.get("hard_integrity_pass")
    if l0_parity and l0_parity.get("status") != "PASS":
        failures["l0_timeframe_parity"] = l0_parity.get("status")
    if l1_audit and l1_audit.get("status") != "PASS":
        failures["l1_context_audit"] = l1_audit.get("status")
    if daily_audit and daily_audit.get("status") != "PASS":
        failures["daily_confluence_audit"] = daily_audit.get("status")

    st_lookahead = int((st_regime.get("lookahead") or {}).get("total_violations", 0))
    st_variant_lookahead = int(st_variant.get("lookahead_violations", 0))
    add_failure(failures, "supertrend_lookahead_violations", st_lookahead)
    add_failure(failures, "supertrend_variant_lookahead_violations", st_variant_lookahead)

    continuity_status = l0_continuity.get("continuity_status")
    if continuity_status != "PASS_SCHEDULED_GAPS_ONLY_LIKELY":
        warnings["l0_continuity_status"] = continuity_status
        warnings["l0_continuity_note"] = (
            "Known gaps are acceptable only if L1 quality flags keep decisions off bad bars."
        )

    status = "PASS" if not failures else "FAIL"
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "package": rel(DATA_DIR),
        "model_dir": rel(MODEL_DIR),
        "events_rows": int(len(events)),
        "events_columns": int(len(events.columns)),
        "failures": failures,
        "warnings": warnings,
        "upstream": {
            "l0_hard_integrity_pass": l0_continuity.get("hard_integrity_pass"),
            "l0_continuity_status": continuity_status,
            "l0_gap_count_gt_60s": (l0_continuity.get("gap_summary") or {}).get("gap_count_gt_60s"),
            "l0_timeframe_parity_status": l0_parity.get("status"),
            "l1_audit_status": l1_audit.get("status"),
            "daily_confluence_status": daily_audit.get("status"),
            "supertrend_lookahead_violations": st_lookahead,
            "supertrend_variant_lookahead_violations": st_variant_lookahead,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

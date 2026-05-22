#!/usr/bin/env python3
"""Single training-data gate for M1 SuperTrend Pullback Scalper.

This gate is intentionally strict about schema, feature safety, and timestamp
continuity. It should pass before any model training script is allowed to run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import audit_lookahead  # noqa: E402
from build_l1_context import REQUIRED_L1_COLUMNS, validate_l1  # noqa: E402
from build_m1_events import ROOT, load_config, validate_events  # noqa: E402
from validate_data_integrity import REQUIRED_L2_COLUMNS  # noqa: E402


FEATURE_WHITELIST_PATH = SCRIPT_DIR / "training_feature_whitelist.json"
EXPECTED_STEP_SECONDS = 60
SHORT_GAP_FAIL_SECONDS = 3600
L1_HARD_NON_NULL_COLUMNS = [
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "prev_gap_seconds",
    "data_quality_ok",
]


def load_all() -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    cfg = load_config()
    l1_path = ROOT / cfg["outputs"]["l1_context"]
    l2_path = ROOT / cfg["outputs"]["events"]
    if not l1_path.exists():
        raise SystemExit(f"Missing L1 context: {l1_path}")
    if not l2_path.exists():
        raise SystemExit(f"Missing L2 events: {l2_path}")
    if not FEATURE_WHITELIST_PATH.exists():
        raise SystemExit(f"Missing feature whitelist: {FEATURE_WHITELIST_PATH}")

    l1 = pd.read_parquet(l1_path)
    l2 = pd.read_parquet(l2_path)
    l1["timestamp_utc"] = pd.to_datetime(l1["timestamp_utc"], utc=True)
    l2["signal_ts"] = pd.to_datetime(l2["signal_ts"], utc=True)
    l2["entry_ts"] = pd.to_datetime(l2["entry_ts"], utc=True)
    l2["exit_ts"] = pd.to_datetime(l2["exit_ts"], utc=True)
    whitelist = json.loads(FEATURE_WHITELIST_PATH.read_text())
    return cfg, l1, l2, whitelist


def gap_report(ts: pd.Series) -> dict:
    sorted_ts = ts.sort_values().reset_index(drop=True)
    diffs = sorted_ts.diff().dt.total_seconds().dropna()
    gaps = diffs[diffs > EXPECTED_STEP_SECONDS]
    short_gaps = gaps[gaps < SHORT_GAP_FAIL_SECONDS]
    top_gaps = (
        pd.DataFrame(
            {
                "from": sorted_ts.iloc[gaps.index - 1].astype(str).values,
                "to": sorted_ts.iloc[gaps.index].astype(str).values,
                "gap_seconds": gaps.astype(int).values,
            }
        )
        .sort_values("gap_seconds", ascending=False)
        .head(10)
        .to_dict(orient="records")
        if not gaps.empty
        else []
    )
    top_short_gaps = (
        pd.DataFrame(
            {
                "from": sorted_ts.iloc[short_gaps.index - 1].astype(str).values,
                "to": sorted_ts.iloc[short_gaps.index].astype(str).values,
                "gap_seconds": short_gaps.astype(int).values,
            }
        )
        .sort_values("gap_seconds", ascending=False)
        .head(10)
        .to_dict(orient="records")
        if not short_gaps.empty
        else []
    )
    return {
        "rows": int(len(sorted_ts)),
        "min": sorted_ts.min().isoformat(),
        "max": sorted_ts.max().isoformat(),
        "median_step_seconds": float(diffs.median()) if not diffs.empty else 0.0,
        "gap_count_gt_60s": int(len(gaps)),
        "short_gap_count_60s_to_3600s": int(len(short_gaps)),
        "top_gaps": top_gaps,
        "top_short_gaps": top_short_gaps,
    }


def check_l1(l1: pd.DataFrame) -> dict:
    missing = [c for c in REQUIRED_L1_COLUMNS if c not in l1.columns]
    if missing:
        raise SystemExit(f"FAIL L1 missing columns: {missing}")
    validate_l1(l1)
    hard_nulls = l1[L1_HARD_NON_NULL_COLUMNS].isna().sum()
    bad_hard_nulls = hard_nulls[hard_nulls > 0]
    if not bad_hard_nulls.empty:
        raise SystemExit(f"FAIL L1 hard non-null columns contain nulls: {bad_hard_nulls.to_dict()}")

    report = gap_report(l1["timestamp_utc"])
    if report["median_step_seconds"] != EXPECTED_STEP_SECONDS:
        raise SystemExit(f"FAIL L1 median step is not 60s: {report['median_step_seconds']}")
    null_report = {
        c: int(v)
        for c, v in l1[REQUIRED_L1_COLUMNS].isna().sum().sort_values(ascending=False).items()
        if int(v) > 0
    }
    report["hard_non_null_columns"] = L1_HARD_NON_NULL_COLUMNS
    report["indicator_warmup_nulls_allowed"] = null_report
    report["data_quality_ok_rows"] = int(l1["data_quality_ok"].astype(bool).sum())
    report["data_quality_quarantined_rows"] = int((~l1["data_quality_ok"].astype(bool)).sum())
    return report


def check_l2(cfg: dict, l1: pd.DataFrame, l2: pd.DataFrame) -> dict:
    missing = [c for c in REQUIRED_L2_COLUMNS if c not in l2.columns]
    if missing:
        raise SystemExit(f"FAIL L2 missing columns: {missing}")
    validate_events(l2, l1)
    all_nulls = l2.isna().sum()
    bad_all_nulls = all_nulls[all_nulls > 0]
    if not bad_all_nulls.empty:
        raise SystemExit(f"FAIL L2 contains nulls in training datamart columns: {bad_all_nulls.to_dict()}")
    if (~l2["signal_data_quality_ok"].astype(bool)).any():
        raise SystemExit("FAIL L2 contains signals from quarantined data-quality windows")
    if (l2["entry_ts"] <= l2["signal_ts"]).any():
        raise SystemExit("FAIL L2 has entry_ts <= signal_ts")
    max_gap = float(cfg["execution"]["max_entry_gap_minutes"]) * 60.0
    if (l2["entry_gap_seconds"] <= 0).any() or (l2["entry_gap_seconds"] > max_gap).any():
        bad = l2[(l2["entry_gap_seconds"] <= 0) | (l2["entry_gap_seconds"] > max_gap)]
        raise SystemExit(f"FAIL L2 bad signal-entry gaps: {len(bad)}")
    if (l2["entry_gap_seconds"] != EXPECTED_STEP_SECONDS).mean() > 0.001:
        raise SystemExit("FAIL L2 too many non-60s signal-entry gaps")

    l1_idx = pd.Series(range(len(l1)), index=l1["timestamp_utc"])
    l1_gaps = l1["prev_gap_seconds"].to_numpy(dtype=float)
    bad_outcome_windows = []
    for _, row in l2.iterrows():
        if row["entry_ts"] not in l1_idx.index or row["exit_ts"] not in l1_idx.index:
            bad_outcome_windows.append(str(row["event_id"]))
            continue
        start_i = int(l1_idx.loc[row["entry_ts"]])
        end_i = int(l1_idx.loc[row["exit_ts"]])
        if end_i < start_i:
            bad_outcome_windows.append(str(row["event_id"]))
            continue
        if (l1_gaps[start_i:end_i + 1] > EXPECTED_STEP_SECONDS).any():
            bad_outcome_windows.append(str(row["event_id"]))
    if bad_outcome_windows:
        raise SystemExit(
            f"FAIL L2 outcome windows cross continuity gaps: {bad_outcome_windows[:10]}"
        )

    signal_report = gap_report(l2["signal_ts"])
    entry_report = gap_report(l2["entry_ts"])
    non_60_signal_entry = int((l2["entry_gap_seconds"] != EXPECTED_STEP_SECONDS).sum())
    return {
        "rows": int(len(l2)),
        "signal_ts": signal_report,
        "entry_ts": entry_report,
        "non_null": {
            "checked_columns": int(len(l2.columns)),
            "null_columns": {},
        },
        "continuity": {
            "entry_gap_seconds_expected": EXPECTED_STEP_SECONDS,
            "non_60_signal_entry_gaps": non_60_signal_entry,
            "outcome_windows_crossing_gaps": 0,
            "quarantined_signal_rows_used": 0,
        },
        "avg_pnl": float(l2["pnl_usd"].mean()),
        "win_rate": float((l2["pnl_usd"] > 0).mean()),
    }


def check_whitelist(l2: pd.DataFrame, whitelist: dict) -> dict:
    features = whitelist["features"]
    missing = [f for f in features if f not in l2.columns]
    if missing:
        raise SystemExit(f"FAIL whitelist missing from L2: {missing}")

    forbidden_patterns = whitelist.get("forbidden_patterns", [])
    forbidden = [
        f
        for f in features
        for pattern in forbidden_patterns
        if pattern in f
    ]
    if forbidden:
        raise SystemExit(f"FAIL whitelist contains forbidden features: {sorted(set(forbidden))}")

    nulls = l2[features].isna().sum()
    bad_nulls = nulls[nulls > 0]
    if not bad_nulls.empty:
        raise SystemExit(f"FAIL whitelist feature nulls: {bad_nulls.to_dict()}")

    constants = [f for f in features if l2[f].nunique(dropna=True) <= 1]
    return {
        "feature_count": len(features),
        "non_null_feature_count": int(len(features) - len(bad_nulls)),
        "null_features": {},
        "constant_features": constants,
    }


def run_lookahead_full() -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["audit_lookahead.py", "--sample", "20000"]
        audit_lookahead.main()
    finally:
        sys.argv = old_argv


def main() -> int:
    cfg, l1, l2, whitelist = load_all()
    l1_report = check_l1(l1)
    l2_report = check_l2(cfg, l1, l2)
    whitelist_report = check_whitelist(l2, whitelist)
    run_lookahead_full()

    report = {
        "status": "PASS",
        "training_allowed": True,
        "live_isolation": {
            "research_only": True,
            "touches_live_pipeline": False,
            "touches_topstepx_executor": False,
            "touches_telegram_live": False,
        },
        "preflight_contract": {
            "entry_timing": cfg["execution"]["entry_timing"],
            "model_features_source": "training_feature_whitelist.json",
            "forbidden_model_fields": whitelist.get("forbidden_patterns", []),
        },
        "non_null_gate": {
            "l1_hard_columns": l1_report["hard_non_null_columns"],
            "l1_indicator_warmup_nulls_allowed": l1_report["indicator_warmup_nulls_allowed"],
            "l2_null_columns": l2_report["non_null"]["null_columns"],
            "whitelist_null_features": whitelist_report["null_features"],
        },
        "continuity_gate": {
            "l1_gap_count_gt_60s": l1_report["gap_count_gt_60s"],
            "l1_short_gap_count_60s_to_3600s": l1_report["short_gap_count_60s_to_3600s"],
            "l1_top_short_gaps": l1_report["top_short_gaps"],
            "l1_data_quality_quarantined_rows": l1_report["data_quality_quarantined_rows"],
            "l2_non_60_signal_entry_gaps": l2_report["continuity"]["non_60_signal_entry_gaps"],
            "l2_outcome_windows_crossing_gaps": l2_report["continuity"]["outcome_windows_crossing_gaps"],
            "l2_quarantined_signal_rows_used": l2_report["continuity"]["quarantined_signal_rows_used"],
        },
        "l1": l1_report,
        "l2": l2_report,
        "whitelist": whitelist_report,
    }
    report_path = ROOT / cfg["outputs"]["training_gate_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print("PASS training data gate")
    print(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

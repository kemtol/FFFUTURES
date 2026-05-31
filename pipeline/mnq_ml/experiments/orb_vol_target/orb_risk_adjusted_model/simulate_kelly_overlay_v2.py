#!/usr/bin/env python3
"""Simulate base-floor Kelly overlays using V2 confluence probabilities."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from common import project_path, write_json  # noqa: E402
from simulate_kelly_overlay import (  # noqa: E402
    DEFAULT_EXECUTION_EVENTS,
    DEFAULT_KELLY_FRACTIONS,
    DEFAULT_NORMALIZED_TARGET_RISKS,
    attach_execution_metadata,
    build_variant_events,
    load_json,
    max_drawdown,
    parse_float_list,
    score_dataset,
    split_manifest,
    summarize,
    write_report,
)

CONFIG_PATH = SCRIPT_DIR / "config.json"
DEFAULT_MODEL_ARTIFACT = "model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v2_success_2r_logistic.joblib"
RECENT_WINDOWS = [5, 10, 20, 30, 50, 100, 200]
RECENT_WINDOW_VARIANTS = [
    "fixed_1.00x",
    "basefloor_kelly_0.10x",
    "basefloor_kelly_1.00x",
    "norm_target_600",
    "norm_target_750",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--model-artifact", default=DEFAULT_MODEL_ARTIFACT)
    parser.add_argument("--execution-events", default=DEFAULT_EXECUTION_EVENTS)
    parser.add_argument("--base-risk-usd", type=float, default=500.0)
    parser.add_argument("--payoff-ratio", type=float, default=2.0)
    parser.add_argument("--min-risk-multiplier", type=float, default=1.0)
    parser.add_argument("--max-risk-multiplier", type=float, default=2.0)
    parser.add_argument("--kelly-fractions", default=",".join(str(x) for x in DEFAULT_KELLY_FRACTIONS))
    parser.add_argument("--normalized-target-risks", default=",".join(str(x) for x in DEFAULT_NORMALIZED_TARGET_RISKS))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def recent_window_summary(events: pd.DataFrame) -> pd.DataFrame:
    base = events.copy()
    base["ny_date"] = pd.to_datetime(base["ny_date"]).dt.date
    base["signal_ts"] = pd.to_datetime(base["signal_ts"], utc=True)
    dates = sorted(base["ny_date"].unique())
    rows: list[dict[str, Any]] = []
    for variant in RECENT_WINDOW_VARIANTS:
        variant_events = base[base["variant"].eq(variant)].copy()
        for window in RECENT_WINDOWS:
            window_dates = set(dates[-window:])
            cur = variant_events[variant_events["ny_date"].isin(window_dates)].sort_values("signal_ts")
            dd = max_drawdown(cur["integer_pnl_usd"])
            pnl = float(cur["integer_pnl_usd"].sum())
            rows.append(
                {
                    "variant": variant,
                    "window_days": int(window),
                    "calendar_start": min(window_dates).isoformat(),
                    "calendar_end": max(window_dates).isoformat(),
                    "trades": int(len(cur)),
                    "integer_pnl_usd": pnl,
                    "integer_max_dd_usd": dd,
                    "integer_return_dd": float(pnl / abs(dd)) if dd < 0 else None,
                    "avg_integer_risk_usd": float(cur["integer_risk_usd"].mean()) if len(cur) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def append_recent_windows_to_report(path: Path, recent_windows: pd.DataFrame) -> None:
    lines = ["", "## Recent Windows", ""]
    for variant in RECENT_WINDOW_VARIANTS:
        cur = recent_windows[recent_windows["variant"].eq(variant)].copy()
        if cur.empty:
            continue
        lines.append(f"### `{variant}`")
        lines.append("")
        lines.append("| Window | Trades | Int PnL | Int DD | Int R/DD | Avg Int Risk |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in cur.sort_values("window_days").itertuples(index=False):
            ret_dd = "" if pd.isna(row.integer_return_dd) else f"{row.integer_return_dd:.2f}"
            lines.append(
                f"| {row.window_days}D | {row.trades} | ${row.integer_pnl_usd:,.0f} | "
                f"${row.integer_max_dd_usd:,.0f} | {ret_dd} | ${row.avg_integer_risk_usd:,.0f} |"
            )
        lines.append("")
    with path.open("a") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    cfg = load_json(Path(args.config))
    dataset_path = project_path(cfg["outputs"]["breakout_quality"])
    events_path = project_path(cfg["outputs"]["kelly_overlay_v2_events"])
    summary_path = project_path(cfg["outputs"]["kelly_overlay_v2_summary"])
    manifest_path = project_path(cfg["outputs"]["kelly_overlay_v2_manifest"])
    recent_windows_path = project_path(cfg["outputs"]["kelly_overlay_v2_recent_windows"])
    model_path = project_path(args.model_artifact)
    execution_events_path = project_path(args.execution_events)

    if not dataset_path.exists():
        raise SystemExit(f"Missing breakout-quality dataset: {dataset_path}")
    if not model_path.exists():
        raise SystemExit(f"Missing model artifact: {model_path}")
    for path in [events_path, summary_path, manifest_path, recent_windows_path]:
        if path.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite existing artifact: {path}")

    df = pd.read_parquet(dataset_path)
    for col in ["signal_ts", "entry_ts", "exit_ts"]:
        df[col] = pd.to_datetime(df[col], utc=True)

    bundle: dict[str, Any] = joblib.load(model_path)
    scored = attach_execution_metadata(score_dataset(df, bundle), execution_events_path)
    probability_col = f"prob_{bundle['target']}"
    kelly_fractions = parse_float_list(args.kelly_fractions)
    normalized_target_risks = parse_float_list(args.normalized_target_risks)
    events = build_variant_events(
        scored=scored,
        probability_col=probability_col,
        base_risk=args.base_risk_usd,
        payoff_ratio=args.payoff_ratio,
        min_mult=args.min_risk_multiplier,
        max_mult=args.max_risk_multiplier,
        kelly_fractions=kelly_fractions,
        normalized_target_risks=normalized_target_risks,
    )
    summary = summarize(events)
    recent_windows = recent_window_summary(events)

    events_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(events_path, index=False)
    summary.to_parquet(summary_path, index=False)
    recent_windows.to_parquet(recent_windows_path, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "model_version": bundle.get("model_version", "unknown"),
        "dataset": str(dataset_path),
        "model_artifact": str(model_path),
        "execution_events": str(execution_events_path),
        "model_target": bundle["target"],
        "probability_col": probability_col,
        "events_output": str(events_path),
        "summary_output": str(summary_path),
        "recent_windows_output": str(recent_windows_path),
        "base_risk_usd": args.base_risk_usd,
        "payoff_ratio": args.payoff_ratio,
        "break_even_probability": 1.0 / (args.payoff_ratio + 1.0),
        "min_risk_multiplier": args.min_risk_multiplier,
        "max_risk_multiplier": args.max_risk_multiplier,
        "kelly_fractions": kelly_fractions,
        "normalized_target_risks": normalized_target_risks,
        "formula": "risk_multiplier = clip(1 + kelly_fraction * max(0, (b*p - (1-p))/b), min=1.0, max)",
        "normalized_formula": "scale is fit on train only; risk_multiplier = clip(1 + scale * max(0, (b*p - (1-p))/b), min=1.0, max)",
        "continuous_risk_note": "PnL is simulated as r_multiple * risk_usd.",
        "integer_execution_note": "Executable MNQ sizing uses max(1, ceil(risk_usd / risk_per_contract_usd)) so actual risk is at least desired risk.",
        "summary": split_manifest(summary),
        "recent_windows": recent_windows.to_dict("records"),
    }
    report_path = project_path(cfg["outputs"]["model_dir"]) / "risk_adjusted_v2_kelly_overlay_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(report_path, summary, manifest)
    append_recent_windows_to_report(report_path, recent_windows)
    manifest["report"] = str(report_path)
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest": str(manifest_path),
                "summary": str(summary_path),
                "recent_windows": str(recent_windows_path),
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

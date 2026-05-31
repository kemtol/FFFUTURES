#!/usr/bin/env python3
"""Train MNQ ORB risk-adjusted V2 models on the 62-feature confluence dataset."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from build_breakout_quality_features import FEATURE_COLUMNS, FEATURE_FAMILIES  # noqa: E402
from common import project_path, write_json  # noqa: E402
from train_risk_adjusted_v1 import (  # noqa: E402
    MODEL_SPECS,
    TARGETS,
    coefficient_importance,
    evaluate_split,
    fit_model,
    load_json,
    max_drawdown,
    safe_metric,
    threshold_table,
)
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

CONFIG_PATH = SCRIPT_DIR / "config.json"
MODEL_VERSION = "v2_confluence"
ARTIFACT_PREFIX = "risk_adjusted_v2"
V1_METRICS = "model/MNQ/orb_vol_target/orb_risk_adjusted_model/risk_adjusted_v1_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--compare-metrics", default=V1_METRICS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_dataset(df: pd.DataFrame) -> None:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns: {missing}")
    nulls = df[FEATURE_COLUMNS].isna().sum()
    bad_nulls = nulls[nulls > 0].to_dict()
    if bad_nulls:
        raise SystemExit(f"Feature nulls found: {bad_nulls}")
    if len(FEATURE_COLUMNS) != 62:
        raise SystemExit(f"Expected 62 V2 features, got {len(FEATURE_COLUMNS)}")
    if "daily_confluence_feature_date" not in df.columns:
        raise SystemExit("Missing daily_confluence_feature_date")
    ny_date = pd.to_datetime(df["ny_date"]).dt.date
    feature_date = pd.to_datetime(df["daily_confluence_feature_date"]).dt.date
    violations = int((feature_date >= ny_date).sum())
    if violations:
        raise SystemExit(f"Daily confluence lookahead violations: {violations}")


def split_group_metrics(frame: pd.DataFrame, target: str, prob_col: str, group_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(group_col, dropna=False, sort=True):
        y_true = group[target].astype(int)
        y_prob = group[prob_col].to_numpy()
        rows.append(
            {
                "group": str(value),
                "rows": int(len(group)),
                "target_rate": float(y_true.mean()) if len(group) else None,
                "avg_probability": float(np.mean(y_prob)) if len(group) else None,
                "auc": safe_metric(roc_auc_score, y_true, y_prob),
                "brier": safe_metric(brier_score_loss, y_true, y_prob),
                "sum_pnl_per_contract_usd": float(group["pnl_per_contract_usd"].sum()),
                "avg_pnl_per_contract_usd": float(group["pnl_per_contract_usd"].mean()),
                "max_dd_per_contract_usd": max_drawdown(group.sort_values("signal_ts")["pnl_per_contract_usd"]),
            }
        )
    return rows


def add_train_tertile(frame: pd.DataFrame, train: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    lo = float(train[col].quantile(1.0 / 3.0))
    hi = float(train[col].quantile(2.0 / 3.0))
    out = frame.copy()
    out[out_col] = pd.cut(
        out[col],
        bins=[-np.inf, lo, hi, np.inf],
        labels=["low", "mid", "high"],
        include_lowest=True,
    ).astype(str)
    return out


def holdout_diagnostics(
    holdout_predictions: pd.DataFrame,
    train: pd.DataFrame,
    target: str,
    prob_col: str,
) -> dict[str, Any]:
    diag: dict[str, Any] = {
        "side": split_group_metrics(holdout_predictions, target, prob_col, "side"),
        "regime_slices": {},
    }
    regime_cols = {
        "vix_level": "dc_vix_prev_close",
        "qqq_relative_strength_5d": "dc_qqq_minus_spy_return_5d",
        "dxy_trend_5d": "dc_dxy_return_5d",
    }
    for name, col in regime_cols.items():
        if col not in holdout_predictions.columns:
            continue
        sliced = add_train_tertile(holdout_predictions, train, col, f"{name}_tertile")
        diag["regime_slices"][name] = split_group_metrics(sliced, target, prob_col, f"{name}_tertile")
    return diag


def family_importance(importance: pd.DataFrame) -> list[dict[str, Any]]:
    family_by_feature = {
        feature: family
        for family, features in FEATURE_FAMILIES.items()
        for feature in features
    }
    cur = importance.copy()
    cur["family"] = cur["feature"].map(family_by_feature).fillna("unknown")
    grouped = (
        cur.groupby("family", as_index=False)
        .agg(
            feature_count=("feature", "count"),
            importance_sum=("importance", "sum"),
            importance_mean=("importance", "mean"),
        )
        .sort_values("importance_sum", ascending=False)
    )
    return grouped.to_dict("records")


def load_v1_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def metric_delta(
    v1: dict[str, Any] | None,
    report: dict[str, Any],
    target: str,
    model_name: str,
    split: str,
    metric: str,
) -> dict[str, Any]:
    v2_value = report["targets"][target]["models"][model_name]["metrics"][split][metric]
    v1_value = None
    if v1 is not None:
        v1_value = v1["targets"][target]["models"][model_name]["metrics"][split][metric]
    return {
        "target": target,
        "model": model_name,
        "split": split,
        "metric": metric,
        "v1": v1_value,
        "v2": v2_value,
        "delta": None if v1_value is None or v2_value is None else float(v2_value - v1_value),
    }


def comparison_rows(v1: dict[str, Any] | None, report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for model_name in MODEL_SPECS:
            for metric in ["auc", "average_precision", "brier"]:
                rows.append(metric_delta(v1, report, target, model_name, "holdout", metric))
    return rows


def fmt_metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def write_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MNQ ORB Risk-Adjusted V2 Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Dataset",
        "",
        f"- Rows: `{report['dataset']['rows']}`",
        f"- Columns: `{report['dataset']['columns']}`",
        f"- Features: `{len(report['feature_columns'])}`",
        f"- Train/validation/holdout: `{report['dataset']['split_counts']}`",
        f"- Daily confluence lookahead violations: `{report['dataset']['daily_confluence_lookahead_violations']}`",
        "",
        "## Feature Families",
        "",
        "| Family | Features |",
        "| --- | ---: |",
    ]
    for family, features in FEATURE_FAMILIES.items():
        lines.append(f"| `{family}` | {len(features)} |")
    lines.extend(["", "## V1 vs V2 Holdout", "", "| Target | Model | Metric | V1 | V2 | Delta |", "| --- | --- | --- | ---: | ---: | ---: |"])
    for row in report["v1_v2_comparison"]:
        lines.append(
            f"| `{row['target']}` | `{row['model']}` | `{row['metric']}` | "
            f"{fmt_metric(row['v1'])} | {fmt_metric(row['v2'])} | {fmt_metric(row['delta'])} |"
        )
    lines.extend(["", "## Models", ""])

    for target, target_report in report["targets"].items():
        lines.append(f"### Target `{target}`")
        lines.append("")
        lines.append("| Model | Val AUC | Val PR-AUC | Val Brier | Holdout AUC | Holdout PR-AUC | Holdout Brier |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for model_name, model_report in target_report["models"].items():
            validation = model_report["metrics"]["validation"]
            holdout = model_report["metrics"]["holdout"]
            lines.append(
                f"| `{model_name}` | {fmt_metric(validation['auc'])} | "
                f"{fmt_metric(validation['average_precision'])} | {fmt_metric(validation['brier'])} | "
                f"{fmt_metric(holdout['auc'])} | {fmt_metric(holdout['average_precision'])} | "
                f"{fmt_metric(holdout['brier'])} |"
            )
        lines.append("")

    lines.extend(["## Top Feature Families", ""])
    for target, target_report in report["targets"].items():
        lines.append(f"### `{target}`")
        lines.append("")
        for model_name, model_report in target_report["models"].items():
            lines.append(f"#### `{model_name}`")
            lines.append("")
            lines.append("| Family | Importance Sum | Importance Mean |")
            lines.append("| --- | ---: | ---: |")
            for row in model_report["feature_family_importance"]:
                lines.append(
                    f"| `{row['family']}` | {float(row['importance_sum']):.4f} | "
                    f"{float(row['importance_mean']):.4f} |"
                )
            lines.append("")

    success_logistic = report["targets"]["success_2r"]["models"]["logistic"]
    daily_top = [
        row
        for row in success_logistic["top_features"]
        if str(row["feature"]).startswith("dc_")
    ][:10]
    lines.extend(
        [
            "## Daily Confluence Readout",
            "",
            "Top daily confluence features from `success_2r` logistic by absolute coefficient:",
            "",
            "| Feature | Importance | Signed Value |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in daily_top:
        lines.append(f"| `{row['feature']}` | {float(row['importance']):.4f} | {float(row['signed_value']):.4f} |")

    lines.extend(
        [
            "",
            "## Holdout Slices",
            "",
            "Primary diagnostic shown for `success_2r` logistic.",
            "",
            "### Side",
            "",
            "| Side | Rows | Target Rate | Avg Prob | AUC | Brier | Sum PnL/ct | DD/ct |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in success_logistic["holdout_diagnostics"]["side"]:
        lines.append(
            f"| `{row['group']}` | {row['rows']} | {fmt_metric(row['target_rate'])} | "
            f"{fmt_metric(row['avg_probability'])} | {fmt_metric(row['auc'])} | "
            f"{fmt_metric(row['brier'])} | ${row['sum_pnl_per_contract_usd']:,.0f} | "
            f"${row['max_dd_per_contract_usd']:,.0f} |"
        )
    for regime_name, rows in success_logistic["holdout_diagnostics"]["regime_slices"].items():
        lines.extend(
            [
                "",
                f"### `{regime_name}`",
                "",
                "| Tertile | Rows | Target Rate | Avg Prob | AUC | Brier | Sum PnL/ct | DD/ct |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row['group']}` | {row['rows']} | {fmt_metric(row['target_rate'])} | "
                f"{fmt_metric(row['avg_probability'])} | {fmt_metric(row['auc'])} | "
                f"{fmt_metric(row['brier'])} | ${row['sum_pnl_per_contract_usd']:,.0f} | "
                f"${row['max_dd_per_contract_usd']:,.0f} |"
            )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- V2 uses the same model families as V1, but on the 62-feature confluence dataset.",
            "- V1 metrics are retained only as a pre-confluence baseline.",
            "- This is still research. Kelly and Topstep-style overlays must be evaluated before live forward testing.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    cfg = load_json(Path(args.config))
    dataset_path = project_path(cfg["outputs"]["breakout_quality"])
    model_dir = project_path(cfg["outputs"]["model_dir"])
    if not dataset_path.exists():
        raise SystemExit(f"Missing dataset: {dataset_path}")
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dataset_path)
    df["signal_ts"] = pd.to_datetime(df["signal_ts"], utc=True)
    validate_dataset(df)
    train = df[df["split"] == "train"].copy()
    valid = df[df["split"] == "validation"].copy()
    holdout = df[df["split"] == "holdout"].copy()

    if not args.force:
        existing = list(model_dir.glob(f"{ARTIFACT_PREFIX}_*"))
        if existing:
            raise SystemExit(f"Existing V2 artifacts found in {model_dir}; use --force to overwrite.")

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "experiment": cfg["experiment"],
        "model_version": MODEL_VERSION,
        "dataset": {
            "path": str(dataset_path),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().sort_index().to_dict().items()},
            "daily_confluence_lookahead_violations": int(
                (
                    pd.to_datetime(df["daily_confluence_feature_date"]).dt.date
                    >= pd.to_datetime(df["ny_date"]).dt.date
                ).sum()
            ),
        },
        "feature_columns": FEATURE_COLUMNS,
        "feature_families": FEATURE_FAMILIES,
        "targets": {},
    }

    all_importances = []
    for target in TARGETS:
        target_report = {"models": {}}
        for model_spec in MODEL_SPECS:
            model = fit_model(model_spec, train, valid, target)
            artifact_path = model_dir / f"{ARTIFACT_PREFIX}_{target}_{model_spec}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "target": target,
                    "model_spec": model_spec,
                    "model_version": MODEL_VERSION,
                    "feature_columns": FEATURE_COLUMNS,
                    "feature_families": FEATURE_FAMILIES,
                    "created_at": report["created_at"],
                },
                artifact_path,
            )

            metrics: dict[str, Any] = {}
            predictions: dict[str, pd.DataFrame] = {}
            for split_name, split_df in [("train", train), ("validation", valid), ("holdout", holdout)]:
                split_metrics, scored = evaluate_split(model, split_df, target, split_name)
                metrics[split_name] = split_metrics
                predictions[split_name] = scored

            prob_col = f"prob_{target}"
            threshold_metrics = threshold_table(predictions["holdout"], target, prob_col)
            importance = coefficient_importance(model, model_spec)
            importance["target"] = target
            all_importances.append(importance)

            pred_path = model_dir / f"{ARTIFACT_PREFIX}_{target}_{model_spec}_holdout_predictions.csv"
            predictions["holdout"].to_csv(pred_path, index=False)
            top_features = importance.head(25).replace({np.nan: None}).to_dict("records")
            target_report["models"][model_spec] = {
                "artifact": str(artifact_path),
                "holdout_predictions": str(pred_path),
                "metrics": metrics,
                "holdout_threshold_table": threshold_metrics,
                "feature_family_importance": family_importance(importance),
                "top_features": top_features,
                "holdout_diagnostics": holdout_diagnostics(predictions["holdout"], train, target, prob_col),
            }
        report["targets"][target] = target_report

    importance_df = pd.concat(all_importances, ignore_index=True)
    importance_path = model_dir / f"{ARTIFACT_PREFIX}_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    report["feature_importance"] = str(importance_path)

    v1 = load_v1_metrics(project_path(args.compare_metrics))
    report["compare_metrics"] = None if v1 is None else str(project_path(args.compare_metrics))
    report["v1_v2_comparison"] = comparison_rows(v1, report)

    metrics_path = model_dir / f"{ARTIFACT_PREFIX}_metrics.json"
    write_json(metrics_path, report)
    report_path = model_dir / f"{ARTIFACT_PREFIX}_report.md"
    write_report(report_path, report)
    print(json.dumps({"metrics": str(metrics_path), "report": str(report_path), "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()

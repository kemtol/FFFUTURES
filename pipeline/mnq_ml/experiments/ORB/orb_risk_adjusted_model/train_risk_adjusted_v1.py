#!/usr/bin/env python3
"""Train first-iteration MNQ ORB risk-adjusted probability models."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from build_breakout_quality_features import FEATURE_COLUMNS  # noqa: E402
from common import project_path, write_json  # noqa: E402

CONFIG_PATH = SCRIPT_DIR / "config.json"
TARGETS = ["success_2r", "positive_eod"]
MODEL_SPECS = ["logistic", "lgbm_shallow"]
PROBABILITY_BUCKETS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def safe_metric(fn, y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    try:
        return float(fn(y_true, y_prob))
    except ValueError:
        return None


def make_logistic() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    C=0.3,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def make_lgbm(scale_pos_weight: float) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=120,
        learning_rate=0.035,
        num_leaves=7,
        max_depth=3,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=4.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )


def fit_model(model_spec: str, train: pd.DataFrame, valid: pd.DataFrame, target: str):
    x_train = train[FEATURE_COLUMNS]
    y_train = train[target].astype(int)
    x_valid = valid[FEATURE_COLUMNS]
    y_valid = valid[target].astype(int)

    if model_spec == "logistic":
        model = make_logistic()
        model.fit(x_train, y_train)
        return model

    if model_spec == "lgbm_shallow":
        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        scale_pos_weight = float(neg / pos) if pos else 1.0
        base = make_lgbm(scale_pos_weight)
        base.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        # Calibrate on validation so reported probabilities are less raw.
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
        calibrated.fit(x_valid, y_valid)
        return calibrated

    raise ValueError(f"Unsupported model spec: {model_spec}")


def predict_proba(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]


def max_drawdown(pnls: pd.Series) -> float:
    if pnls.empty:
        return 0.0
    equity = pnls.cumsum()
    equity = pd.concat([pd.Series([0.0]), equity], ignore_index=True)
    drawdown = equity - equity.cummax()
    return float(drawdown.min())


def bucket_table(frame: pd.DataFrame, target: str, prob_col: str) -> list[dict[str, Any]]:
    cur = frame.copy()
    cur["prob_bucket"] = pd.cut(cur[prob_col], bins=PROBABILITY_BUCKETS, right=False)
    rows: list[dict[str, Any]] = []
    for bucket, group in cur.groupby("prob_bucket", observed=True):
        rows.append(
            {
                "bucket": str(bucket),
                "rows": int(len(group)),
                "avg_prob": float(group[prob_col].mean()),
                "target_rate": float(group[target].mean()),
                "avg_pnl_per_contract_usd": float(group["pnl_per_contract_usd"].mean()),
                "sum_pnl_per_contract_usd": float(group["pnl_per_contract_usd"].sum()),
                "max_dd_per_contract_usd": max_drawdown(group.sort_values("signal_ts")["pnl_per_contract_usd"]),
            }
        )
    return rows


def evaluate_split(model, frame: pd.DataFrame, target: str, split: str) -> tuple[dict[str, Any], pd.DataFrame]:
    cur = frame.copy()
    prob_col = f"prob_{target}"
    cur[prob_col] = predict_proba(model, cur)
    y_true = cur[target].astype(int)
    y_prob = cur[prob_col].to_numpy()
    metrics = {
        "split": split,
        "rows": int(len(cur)),
        "positive_rate": float(y_true.mean()),
        "auc": safe_metric(roc_auc_score, y_true, y_prob),
        "average_precision": safe_metric(average_precision_score, y_true, y_prob),
        "log_loss": safe_metric(log_loss, y_true, y_prob),
        "brier": safe_metric(brier_score_loss, y_true, y_prob),
        "avg_probability": float(np.mean(y_prob)),
        "bucket_table": bucket_table(cur, target, prob_col),
    }
    return metrics, cur


def threshold_table(frame: pd.DataFrame, target: str, prob_col: str) -> list[dict[str, Any]]:
    thresholds = sorted(set([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = frame[frame[prob_col] >= threshold].sort_values("signal_ts")
        if selected.empty:
            rows.append(
                {
                    "threshold": threshold,
                    "selected_rows": 0,
                    "target_rate": None,
                    "sum_pnl_per_contract_usd": 0.0,
                    "avg_pnl_per_contract_usd": None,
                    "max_dd_per_contract_usd": 0.0,
                }
            )
            continue
        rows.append(
            {
                "threshold": threshold,
                "selected_rows": int(len(selected)),
                "target_rate": float(selected[target].mean()),
                "sum_pnl_per_contract_usd": float(selected["pnl_per_contract_usd"].sum()),
                "avg_pnl_per_contract_usd": float(selected["pnl_per_contract_usd"].mean()),
                "max_dd_per_contract_usd": max_drawdown(selected["pnl_per_contract_usd"]),
            }
        )
    return rows


def coefficient_importance(model, model_spec: str) -> pd.DataFrame:
    if model_spec == "logistic":
        coefs = model.named_steps["model"].coef_[0]
        return pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": np.abs(coefs),
                "signed_value": coefs,
                "model": model_spec,
            }
        ).sort_values("importance", ascending=False)

    if model_spec == "lgbm_shallow":
        estimator = model.estimator
        booster = estimator.booster_
        return pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": booster.feature_importance(importance_type="gain"),
                "signed_value": np.nan,
                "model": model_spec,
            }
        ).sort_values("importance", ascending=False)

    return pd.DataFrame()


def write_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MNQ ORB Risk-Adjusted V1 Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Dataset",
        "",
        f"- Rows: `{report['dataset']['rows']}`",
        f"- Features: `{len(report['feature_columns'])}`",
        f"- Train/validation/holdout: `{report['dataset']['split_counts']}`",
        "",
        "## Models",
        "",
    ]
    for target, target_report in report["targets"].items():
        lines.append(f"### Target `{target}`")
        lines.append("")
        for model_name, model_report in target_report["models"].items():
            holdout = model_report["metrics"]["holdout"]
            validation = model_report["metrics"]["validation"]
            lines.append(f"- `{model_name}` validation AUC: `{validation['auc']}` Brier: `{validation['brier']}`")
            lines.append(f"- `{model_name}` holdout AUC: `{holdout['auc']}` Brier: `{holdout['brier']}`")
            lines.append(f"- `{model_name}` artifact: `{model_report['artifact']}`")
        lines.append("")
    lines.extend(
        [
            "## Caveats",
            "",
            "- V1 is a probability-separation test, not live approval.",
            "- Holdout has only 140 breakout rows, so threshold decisions need walk-forward and Topstep overlay checks.",
            "- Labels are outcome labels and must not be used as features.",
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
    train = df[df["split"] == "train"].copy()
    valid = df[df["split"] == "validation"].copy()
    holdout = df[df["split"] == "holdout"].copy()

    if not args.force:
        existing = list(model_dir.glob("risk_adjusted_v1_*"))
        if existing:
            raise SystemExit(f"Existing V1 artifacts found in {model_dir}; use --force to overwrite.")

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "experiment": cfg["experiment"],
        "dataset": {
            "path": str(dataset_path),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().sort_index().to_dict().items()},
        },
        "feature_columns": FEATURE_COLUMNS,
        "targets": {},
    }

    all_importances = []
    for target in TARGETS:
        target_report = {"models": {}}
        for model_spec in MODEL_SPECS:
            model = fit_model(model_spec, train, valid, target)
            artifact_path = model_dir / f"risk_adjusted_v1_{target}_{model_spec}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "target": target,
                    "model_spec": model_spec,
                    "feature_columns": FEATURE_COLUMNS,
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

            holdout_prob_col = f"prob_{target}"
            threshold_metrics = threshold_table(predictions["holdout"], target, holdout_prob_col)
            importance = coefficient_importance(model, model_spec)
            importance["target"] = target
            all_importances.append(importance)

            pred_path = model_dir / f"risk_adjusted_v1_{target}_{model_spec}_holdout_predictions.csv"
            predictions["holdout"].to_csv(pred_path, index=False)
            target_report["models"][model_spec] = {
                "artifact": str(artifact_path),
                "holdout_predictions": str(pred_path),
                "metrics": metrics,
                "holdout_threshold_table": threshold_metrics,
            }
        report["targets"][target] = target_report

    importance_df = pd.concat(all_importances, ignore_index=True)
    importance_path = model_dir / "risk_adjusted_v1_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    report["feature_importance"] = str(importance_path)

    metrics_path = model_dir / "risk_adjusted_v1_metrics.json"
    write_json(metrics_path, report)
    report_path = model_dir / "risk_adjusted_v1_report.md"
    write_report(report_path, report)
    print(json.dumps({"metrics": str(metrics_path), "report": str(report_path), "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()

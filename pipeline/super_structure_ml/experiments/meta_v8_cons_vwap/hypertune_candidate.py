#!/usr/bin/env python3
"""Hypertune Meta-v8 CONS VWAP without fitting decisions on 2026.

Split discipline:
- Train: 2023-2024
- Validation/tuning: 2025
- OOT report only: 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_candidate import (  # noqa: E402
    DEFAULT_DAILY_LIMIT,
    FEATURE_IMPORTANCE_NAME,
    MODEL_CONFIG_NAME,
    MODEL_NAME,
    ROOT,
    load_baseline_selection,
    load_config,
    load_datamart,
    threshold_sweep,
    topstep_sim,
    window_metrics,
)


REPORT_NAME = "hypertune_report.json"
SUMMARY_NAME = "hypertune_summary.csv"


def feature_sets(cfg: dict) -> dict[str, list[str]]:
    base = cfg["baseline"]["features"]
    vwap = [
        "dist_to_ct_vwap_atr",
        "vwap_side_aligned",
        "ct_vwap_slope_20_atr",
        "vwap_deviation_z_50",
    ]
    return {
        "baseline_only": base,
        "vwap_full": base + vwap,
        "vwap_no_side_flag": base + [
            "dist_to_ct_vwap_atr",
            "ct_vwap_slope_20_atr",
            "vwap_deviation_z_50",
        ],
        "vwap_distance_deviation": base + [
            "dist_to_ct_vwap_atr",
            "vwap_deviation_z_50",
        ],
        "vwap_slope_deviation": base + [
            "ct_vwap_slope_20_atr",
            "vwap_deviation_z_50",
        ],
    }


def param_grid() -> list[dict]:
    grid = []
    for learning_rate in [0.02, 0.03, 0.05]:
        for num_leaves in [7, 15, 31]:
            for min_data_in_leaf in [20, 40, 80]:
                for lambda_l2 in [0.0, 1.0, 5.0]:
                    grid.append(
                        {
                            "objective": "binary",
                            "metric": "auc",
                            "verbosity": -1,
                            "learning_rate": learning_rate,
                            "num_leaves": num_leaves,
                            "min_data_in_leaf": min_data_in_leaf,
                            "lambda_l2": lambda_l2,
                            "feature_fraction": 0.85,
                            "bagging_fraction": 0.85,
                            "bagging_freq": 5,
                            "seed": 42,
                        }
                    )
    return grid


def add_probs(model: lgb.Booster, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["prob"] = model.predict(out[features])
    return out


def apply_thresholds(df: pd.DataFrame, thresholds: dict[int, float]) -> pd.DataFrame:
    out = df.copy()
    out["threshold"] = out["session_cluster"].map(thresholds)
    return out[out["prob"] >= out["threshold"]].copy()


def objective(metrics: dict) -> float:
    if metrics["trades"] < 20:
        return -999999.0
    return metrics["pnl"] / (abs(metrics["max_dd"]) + 1.0)


def run_hypertune(limit: int | None = None, dry_run: bool = False) -> dict:
    cfg = load_config()
    all_features = sorted(set(sum(feature_sets(cfg).values(), [])))
    df = load_datamart(ROOT / cfg["candidate"]["datamart"], all_features)

    train = df[df["entry_ts"] < "2025-01-01"].copy()
    valid = df[(df["entry_ts"] >= "2025-01-01") & (df["entry_ts"] < "2026-01-01")].copy()
    oot = df[df["entry_ts"] >= "2026-01-01"].copy()
    if train.empty or valid.empty or oot.empty:
        raise RuntimeError(
            f"Bad split: train={len(train)} valid={len(valid)} oot={len(oot)}"
        )

    rows = []
    best_bundle = None
    params_list = param_grid()
    if limit:
        params_list = params_list[:limit]

    total = len(feature_sets(cfg)) * len(params_list)
    n = 0
    for fs_name, features in feature_sets(cfg).items():
        for params in params_list:
            n += 1
            model = lgb.train(
                params,
                lgb.Dataset(train[features], label=train["is_win"].astype(int)),
                num_boost_round=1000,
                valid_sets=[
                    lgb.Dataset(valid[features], label=valid["is_win"].astype(int))
                ],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=75, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            valid_pred = add_probs(model, valid, features)
            try:
                best_threshold, _ = threshold_sweep(valid_pred, DEFAULT_DAILY_LIMIT)
            except RuntimeError:
                continue
            thresholds = {int(k): float(v) for k, v in best_threshold["thresholds"].items()}
            valid_sel = apply_thresholds(valid_pred, thresholds)
            valid_metrics = topstep_sim(valid_sel, DEFAULT_DAILY_LIMIT)
            valid_score = objective(valid_metrics)

            oot_pred = add_probs(model, oot, features)
            oot_sel = apply_thresholds(oot_pred, thresholds)
            oot_metrics = topstep_sim(oot_sel, DEFAULT_DAILY_LIMIT)

            row = {
                "feature_set": fs_name,
                "features": features,
                "best_iteration": int(model.best_iteration or model.current_iteration()),
                "thresholds": thresholds,
                "valid_score": round(valid_score, 6),
                "valid": valid_metrics,
                "oot_2026": oot_metrics,
                "params": {
                    k: params[k]
                    for k in [
                        "learning_rate",
                        "num_leaves",
                        "min_data_in_leaf",
                        "lambda_l2",
                        "feature_fraction",
                        "bagging_fraction",
                    ]
                },
            }
            rows.append(row)
            if best_bundle is None or valid_score > best_bundle["valid_score"]:
                best_bundle = {
                    "valid_score": valid_score,
                    "row": row,
                    "model": model,
                    "features": features,
                    "oot_selected": oot_sel,
                }

            if n % 25 == 0:
                print(f"Progress {n}/{total} | current best valid_score={best_bundle['valid_score']:.4f}")

    if best_bundle is None:
        raise RuntimeError("No hypertune candidate produced a valid threshold set")

    rows = sorted(rows, key=lambda x: x["valid_score"], reverse=True)
    best = best_bundle["row"]
    baseline_oot = load_baseline_selection(cfg, oot)
    report = {
        "experiment": cfg["experiment"],
        "status": "hypertuned_research_candidate",
        "split": {
            "train": "2023-2024",
            "validation_tuning": "2025",
            "oot_report_only": "2026",
        },
        "rows": {
            "train": int(len(train)),
            "validation": int(len(valid)),
            "oot_2026": int(len(oot)),
        },
        "best": best,
        "best_oot_rolling_windows": window_metrics(
            best_bundle["oot_selected"],
            cfg.get("evaluation_windows_days", [7, 14, 30, 90]),
            DEFAULT_DAILY_LIMIT,
        ),
        "baseline_meta_v7_oot_2026": {
            "full_window": topstep_sim(baseline_oot, DEFAULT_DAILY_LIMIT),
            "rolling_windows": window_metrics(
                baseline_oot,
                cfg.get("evaluation_windows_days", [7, 14, 30, 90]),
                DEFAULT_DAILY_LIMIT,
            ),
        },
        "top_20": rows[:20],
    }

    print("Best validation candidate:")
    print(json.dumps(best, indent=2))
    print("Baseline Meta-v7 OOT 2026:")
    print(json.dumps(report["baseline_meta_v7_oot_2026"]["full_window"], indent=2))

    if dry_run:
        return report

    out_dir = ROOT / cfg["candidate"]["model_dir"] / "hypertuned"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_bundle["model"].save_model(str(out_dir / MODEL_NAME))

    model_cfg = {
        "name": "Meta-v8 CONS VWAP Hypertuned Candidate",
        "status": "RESEARCH_ONLY",
        "split": report["split"],
        "features": best["features"],
        "thresholds": {str(k): v for k, v in best["thresholds"].items()},
        "daily_loss_limit": -DEFAULT_DAILY_LIMIT,
        "params": best["params"],
        "datamart": cfg["candidate"]["datamart"],
    }
    (out_dir / MODEL_CONFIG_NAME).write_text(json.dumps(model_cfg, indent=2) + "\n")
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n")

    importance = pd.DataFrame(
        {
            "feature": best["features"],
            "gain": best_bundle["model"].feature_importance(importance_type="gain"),
            "split": best_bundle["model"].feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(out_dir / FEATURE_IMPORTANCE_NAME, index=False)

    flat = []
    for row in rows:
        flat.append(
            {
                "feature_set": row["feature_set"],
                "valid_score": row["valid_score"],
                "valid_trades": row["valid"]["trades"],
                "valid_pnl": row["valid"]["pnl"],
                "valid_max_dd": row["valid"]["max_dd"],
                "oot_trades": row["oot_2026"]["trades"],
                "oot_pnl": row["oot_2026"]["pnl"],
                "oot_max_dd": row["oot_2026"]["max_dd"],
                "oot_win_rate": row["oot_2026"]["win_rate"],
                "thresholds": json.dumps(row["thresholds"], sort_keys=True),
                "params": json.dumps(row["params"], sort_keys=True),
            }
        )
    pd.DataFrame(flat).to_csv(out_dir / SUMMARY_NAME, index=False)
    print(f"Wrote {out_dir}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Limit hyperparameter configs for smoke tests")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_hypertune(limit=args.limit, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())

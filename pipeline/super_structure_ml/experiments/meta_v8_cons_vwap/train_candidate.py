#!/usr/bin/env python3
"""Train and evaluate the isolated Meta-v8 CONS VWAP candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = Path(__file__).resolve().with_name("config.json")
REPORT_NAME = "training_report.json"
FEATURE_IMPORTANCE_NAME = "feature_importance.csv"
MODEL_NAME = "inference_model.txt"
MODEL_CONFIG_NAME = "inference_config.json"
POINT_VALUE = 10.0
DEFAULT_DAILY_LIMIT = 400.0


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def normalize_side(side: object) -> str:
    value = str(side).lower()
    if value.startswith("l"):
        return "Long"
    if value.startswith("s"):
        return "Short"
    return str(side)


def load_datamart(path: Path, features: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Missing candidate datamart: {path}\n"
            "Run build_l1_vwap.py, then build_features.py first."
        )
    df = pd.read_parquet(path)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns: {missing}")
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "is_win" not in df.columns:
        df["is_win"] = (df["pnl_usd"] > 0).astype(int)
    df["side"] = df["side"].map(normalize_side)
    return df.sort_values("entry_ts").reset_index(drop=True)


def topstep_sim(df: pd.DataFrame, daily_limit: float) -> dict:
    balance = 0.0
    peak = 0.0
    max_dd = 0.0
    daily_pnl: dict[object, float] = {}
    ledger = []

    for _, row in df.sort_values("entry_ts").iterrows():
        day = row["entry_ts"].date()
        daily_pnl.setdefault(day, 0.0)
        if daily_pnl[day] <= -daily_limit:
            continue

        pnl = float(row["pnl_usd"])
        balance += pnl
        daily_pnl[day] += pnl
        peak = max(peak, balance)
        max_dd = min(max_dd, balance - peak)
        ledger.append(pnl)

    wins = sum(1 for pnl in ledger if pnl > 0)
    trades = len(ledger)
    return {
        "trades": trades,
        "pnl": round(balance, 2),
        "max_dd": round(max_dd, 2),
        "win_rate": round(wins / trades, 4) if trades else 0.0,
        "avg_trade": round(balance / trades, 2) if trades else 0.0,
    }


def threshold_sweep(test: pd.DataFrame, daily_limit: float) -> tuple[dict, pd.DataFrame]:
    results = []
    for th_asia in [0.45, 0.50, 0.55]:
        for th_london in [0.40, 0.45, 0.50, 0.55]:
            for th_us in [0.35, 0.40, 0.45, 0.50, 0.55]:
                thresholds = {0: th_asia, 1: th_london, 2: th_us}
                work = test.copy()
                work["threshold"] = work["session_cluster"].map(thresholds)
                keep = work[work["prob"] >= work["threshold"]]
                if keep.empty:
                    continue
                metrics = topstep_sim(keep, daily_limit=daily_limit)
                if metrics["trades"] < 10:
                    continue
                score = metrics["pnl"] / (abs(metrics["max_dd"]) + 1.0)
                results.append(
                    {
                        "thresholds": thresholds,
                        "score": round(score, 6),
                        **metrics,
                    }
                )

    if not results:
        raise RuntimeError("No threshold candidate produced at least 10 trades")
    report = pd.DataFrame(results).sort_values(
        ["score", "pnl", "max_dd"], ascending=[False, False, False]
    )
    return report.iloc[0].to_dict(), report


def window_metrics(df: pd.DataFrame, windows: list[int], daily_limit: float) -> dict[str, dict]:
    if df.empty:
        return {str(w): topstep_sim(df, daily_limit) for w in windows}
    last_ts = df["entry_ts"].max()
    out = {}
    for days in windows:
        subset = df[df["entry_ts"] >= last_ts - pd.Timedelta(days=days)]
        out[str(days)] = topstep_sim(subset, daily_limit)
    return out


def load_baseline_selection(cfg: dict, source_df: pd.DataFrame) -> pd.DataFrame:
    features = cfg["baseline"]["features"]
    model_path = ROOT / cfg["baseline"]["model"]
    config_path = ROOT / cfg["baseline"]["config"]
    missing = [f for f in features if f not in source_df.columns]
    if missing:
        raise SystemExit(f"Missing baseline features in candidate datamart: {missing}")

    model = lgb.Booster(model_file=str(model_path))
    model_cfg = json.loads(config_path.read_text())
    thresholds = {int(k): float(v) for k, v in model_cfg["thresholds"].items()}
    out = source_df.copy()
    out[features] = out[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["prob"] = model.predict(out[features])
    out["threshold"] = out["session_cluster"].map(thresholds)
    return out[out["prob"] >= out["threshold"]].copy()


def train_candidate(dry_run: bool = False) -> dict:
    cfg = load_config()
    features = cfg["candidate"]["features"]
    data_path = ROOT / cfg["candidate"]["datamart"]
    out_dir = ROOT / cfg["candidate"]["model_dir"]
    windows = cfg.get("evaluation_windows_days", [7, 14, 30, 90])

    df = load_datamart(data_path, features)
    train = df[df["entry_ts"] < "2026-01-01"].copy()
    test = df[df["entry_ts"] >= "2026-01-01"].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"Bad train/test split: train={len(train)} test={len(test)}")

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 30,
        "seed": 42,
    }
    dtrain = lgb.Dataset(train[features], label=train["is_win"].astype(int))
    dvalid = lgb.Dataset(test[features], label=test["is_win"].astype(int), reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=75, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    train["prob"] = model.predict(train[features])
    test["prob"] = model.predict(test[features])
    best, sweep = threshold_sweep(test, daily_limit=DEFAULT_DAILY_LIMIT)
    threshold_map = {int(k): float(v) for k, v in best["thresholds"].items()}
    test["threshold"] = test["session_cluster"].map(threshold_map)
    selected = test[test["prob"] >= test["threshold"]].copy()
    baseline_selected = load_baseline_selection(cfg, test)

    importance = pd.DataFrame(
        {
            "feature": features,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)

    report = {
        "experiment": cfg["experiment"],
        "status": "trained_research_candidate",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "date_split": "train < 2026-01-01, test >= 2026-01-01",
        "features": features,
        "best_iteration": int(model.best_iteration or model.current_iteration()),
        "daily_limit": DEFAULT_DAILY_LIMIT,
        "thresholds": {str(k): v for k, v in threshold_map.items()},
        "test_full_window": topstep_sim(selected, daily_limit=DEFAULT_DAILY_LIMIT),
        "test_rolling_windows": window_metrics(selected, windows, DEFAULT_DAILY_LIMIT),
        "baseline_meta_v7_same_window": {
            "full_window": topstep_sim(baseline_selected, daily_limit=DEFAULT_DAILY_LIMIT),
            "rolling_windows": window_metrics(baseline_selected, windows, DEFAULT_DAILY_LIMIT),
        },
        "top_threshold_sweep": sweep.head(10).to_dict(orient="records"),
    }

    print(json.dumps(report["test_full_window"], indent=2))
    print("Best thresholds:", report["thresholds"])
    print("Top feature importance:")
    print(importance.head(12).to_string(index=False))

    if dry_run:
        return report

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / MODEL_NAME))
    (out_dir / MODEL_CONFIG_NAME).write_text(
        json.dumps(
            {
                "name": "Meta-v8 CONS VWAP Candidate",
                "status": "RESEARCH_ONLY",
                "features": features,
                "thresholds": report["thresholds"],
                "daily_loss_limit": -DEFAULT_DAILY_LIMIT,
                "datamart": cfg["candidate"]["datamart"],
            },
            indent=2,
        )
        + "\n"
    )
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n")
    importance.to_csv(out_dir / FEATURE_IMPORTANCE_NAME, index=False)
    sweep.to_csv(out_dir / "threshold_sweep.csv", index=False)
    print(f"Wrote {out_dir}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train_candidate(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())

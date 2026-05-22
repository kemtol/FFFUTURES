#!/usr/bin/env python3
"""Train P1 candidates for the M1 SuperTrend Pullback Scalper.

Research-only trainer. It refuses to train unless the P0 data gate passes,
uses only whitelisted signal-time features plus train-fitted regime buckets,
and evaluates long-history, current-regime, and rolling walk-forward views.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_m1_events import ROOT, load_config, topstep_trade_day  # noqa: E402


WHITELIST_PATH = SCRIPT_DIR / "training_feature_whitelist.json"
GATE_SCRIPT = SCRIPT_DIR / "gate_training_data.py"
MIN_TARGET_TRADES_PER_DAY = 1.0
MAX_TARGET_TRADES_PER_DAY = 3.0
THRESHOLDS = np.round(np.arange(0.35, 0.851, 0.025), 3)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    oot_start: str
    oot_end: str | None
    recent_weight_start: str
    recent_weight: float


SPLITS = [
    SplitSpec(
        name="long_history",
        train_start="2023-01-01",
        train_end="2025-01-01",
        val_start="2025-01-01",
        val_end="2026-01-01",
        oot_start="2026-01-01",
        oot_end=None,
        recent_weight_start="2024-01-01",
        recent_weight=2.0,
    ),
    SplitSpec(
        name="current_regime",
        train_start="2024-01-01",
        train_end="2026-01-01",
        val_start="2026-01-01",
        val_end="2026-04-01",
        oot_start="2026-04-01",
        oot_end=None,
        recent_weight_start="2025-01-01",
        recent_weight=1.5,
    ),
]


def ts(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value, tz="UTC")


def run_gate_and_assert(cfg: dict) -> dict[str, Any]:
    subprocess.run([sys.executable, str(GATE_SCRIPT)], cwd=ROOT, check=True)
    report_path = ROOT / cfg["outputs"]["training_gate_report"]
    if not report_path.exists():
        raise SystemExit(f"Gate report missing after gate run: {report_path}")
    report = json.loads(report_path.read_text())
    if report.get("status") != "PASS" or report.get("training_allowed") is not True:
        raise SystemExit(f"P0 gate blocks training: {report_path}")
    return report


def load_inputs(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    events_path = ROOT / cfg["outputs"]["events"]
    if not events_path.exists():
        raise SystemExit(f"Missing events datamart: {events_path}")
    whitelist = json.loads(WHITELIST_PATH.read_text())
    base_features = list(whitelist["features"])
    df = pd.read_parquet(events_path)
    df["signal_ts"] = pd.to_datetime(df["signal_ts"], utc=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    missing = [c for c in base_features if c not in df.columns]
    if missing:
        raise SystemExit(f"Whitelisted features missing from events: {missing}")
    forbidden = [
        c
        for c in base_features
        for pattern in whitelist.get("forbidden_patterns", [])
        if pattern in c
    ]
    if forbidden:
        raise SystemExit(f"Forbidden feature in whitelist: {sorted(set(forbidden))}")
    if df[base_features].isna().any().any():
        bad = df[base_features].isna().sum()
        raise SystemExit(f"Null whitelisted features: {bad[bad > 0].to_dict()}")
    df = df.sort_values("signal_ts").reset_index(drop=True)
    df["target"] = (df["pnl_usd"] > 0).astype(int)
    df["trade_day"] = topstep_trade_day(df["signal_ts"])
    return df, base_features


def fit_regime_thresholds(train: pd.DataFrame) -> dict[str, list[float]]:
    return {
        "signal_atr": train["signal_atr"].quantile([0.33, 0.66]).astype(float).tolist(),
        "signal_adx": train["signal_adx"].quantile([0.33, 0.66]).astype(float).tolist(),
        "bar_range_atr": train["bar_range_atr"].quantile([0.33, 0.66]).astype(float).tolist(),
        "abs_vwap_z": train["vwap_deviation_z_50"].abs().quantile([0.33, 0.66]).astype(float).tolist(),
    }


def bucket(values: pd.Series, cuts: list[float]) -> pd.Series:
    return pd.cut(
        values.astype(float),
        bins=[-np.inf, cuts[0], cuts[1], np.inf],
        labels=[0, 1, 2],
    ).astype(int)


def add_regime_features(df: pd.DataFrame, thresholds: dict[str, list[float]]) -> pd.DataFrame:
    out = df.copy()
    out["regime_atr_bucket"] = bucket(out["signal_atr"], thresholds["signal_atr"])
    out["regime_adx_bucket"] = bucket(out["signal_adx"], thresholds["signal_adx"])
    out["regime_range_bucket"] = bucket(out["bar_range_atr"], thresholds["bar_range_atr"])
    out["regime_abs_vwap_z_bucket"] = bucket(out["vwap_deviation_z_50"].abs(), thresholds["abs_vwap_z"])
    out["regime_trend_stack"] = np.sign(out["dema_stack"]).astype(int)
    out["regime_high_vol_trend"] = (
        (out["regime_atr_bucket"] == 2)
        & (out["regime_adx_bucket"] == 2)
        & (out["regime_trend_stack"] != 0)
    ).astype(int)
    return out


def subset(df: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    mask = df["signal_ts"] >= ts(start)
    if end is not None:
        mask &= df["signal_ts"] < ts(end)
    return df[mask].copy()


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    curve = pnl.cumsum()
    dd = curve - curve.cummax()
    return float(dd.min())


def period_metrics(selected: pd.DataFrame, period: pd.DataFrame, prob_col: str | None = None) -> dict[str, Any]:
    selected = selected.sort_values("signal_ts")
    trade_days = max(int(period["trade_day"].nunique()), 1)
    wins = selected["pnl_usd"] > 0 if not selected.empty else pd.Series(dtype=bool)
    gross_win = float(selected.loc[selected["pnl_usd"] > 0, "pnl_usd"].sum()) if not selected.empty else 0.0
    gross_loss = float(-selected.loc[selected["pnl_usd"] < 0, "pnl_usd"].sum()) if not selected.empty else 0.0
    out: dict[str, Any] = {
        "period_start": period["signal_ts"].min().isoformat() if not period.empty else None,
        "period_end": period["signal_ts"].max().isoformat() if not period.empty else None,
        "available_trade_days": trade_days,
        "raw_events": int(len(period)),
        "selected_trades": int(len(selected)),
        "trades_per_day": float(len(selected) / trade_days),
        "pnl_usd": float(selected["pnl_usd"].sum()) if not selected.empty else 0.0,
        "avg_trade": float(selected["pnl_usd"].mean()) if not selected.empty else 0.0,
        "median_trade": float(selected["pnl_usd"].median()) if not selected.empty else 0.0,
        "win_rate": float(wins.mean()) if not selected.empty else 0.0,
        "max_drawdown": max_drawdown(selected["pnl_usd"]) if not selected.empty else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else None,
        "long_trades": int((selected["side"] == "Long").sum()) if not selected.empty else 0,
        "short_trades": int((selected["side"] == "Short").sum()) if not selected.empty else 0,
    }
    if prob_col and not selected.empty:
        out["avg_probability"] = float(selected[prob_col].mean())
    return out


def raw_period_metrics(period: pd.DataFrame) -> dict[str, Any]:
    return period_metrics(period, period)


def threshold_score(metrics: dict[str, Any]) -> float:
    trades = metrics["selected_trades"]
    if trades < 10:
        return -1_000_000.0 + trades
    freq = metrics["trades_per_day"]
    freq_penalty = 0.0
    if freq < MIN_TARGET_TRADES_PER_DAY:
        freq_penalty = (MIN_TARGET_TRADES_PER_DAY - freq) * 1500.0
    elif freq > MAX_TARGET_TRADES_PER_DAY:
        freq_penalty = (freq - MAX_TARGET_TRADES_PER_DAY) * 1000.0
    dd_penalty = abs(metrics["max_drawdown"]) * 0.20
    expectancy_bonus = metrics["avg_trade"] * 25.0
    return metrics["pnl_usd"] + expectancy_bonus - dd_penalty - freq_penalty


def select_threshold(frame: pd.DataFrame, period: pd.DataFrame, prob_col: str) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    rows = []
    viable = []
    for threshold in THRESHOLDS:
        selected = frame[frame[prob_col] >= threshold].copy()
        metrics = period_metrics(selected, period, prob_col)
        metrics["threshold"] = float(threshold)
        metrics["score"] = threshold_score(metrics)
        rows.append(metrics)
        if (
            metrics["selected_trades"] >= 10
            and metrics["avg_trade"] > 0
            and metrics["pnl_usd"] > 0
            and MIN_TARGET_TRADES_PER_DAY <= metrics["trades_per_day"] <= MAX_TARGET_TRADES_PER_DAY
        ):
            viable.append(metrics)
    candidates = viable or rows
    best = max(candidates, key=lambda row: row["score"])
    return float(best["threshold"]), best, rows


def sample_weight(train: pd.DataFrame, split: SplitSpec) -> np.ndarray:
    weight = np.ones(len(train), dtype=float)
    recent = train["signal_ts"] >= ts(split.recent_weight_start)
    weight[recent.to_numpy()] = split.recent_weight
    high_vol_trend = (
        (train["regime_atr_bucket"] == 2)
        & (train["regime_adx_bucket"] == 2)
        & (train["regime_trend_stack"] != 0)
    )
    weight[high_vol_trend.to_numpy()] *= 1.15
    return weight


def train_model(train: pd.DataFrame, features: list[str], split: SplitSpec) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=450,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=35,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=2,
        verbose=-1,
    )
    model.fit(
        train[features],
        train["target"],
        sample_weight=sample_weight(train, split),
    )
    return model


def auc_or_none(frame: pd.DataFrame, prob_col: str) -> float | None:
    if frame.empty or frame["target"].nunique() < 2:
        return None
    return float(roc_auc_score(frame["target"], frame[prob_col]))


def evaluate_split(df: pd.DataFrame, base_features: list[str], split: SplitSpec, model_dir: Path) -> dict[str, Any]:
    raw_train = subset(df, split.train_start, split.train_end)
    raw_val = subset(df, split.val_start, split.val_end)
    raw_oot = subset(df, split.oot_start, split.oot_end)
    if raw_train.empty or raw_val.empty or raw_oot.empty:
        raise SystemExit(f"Split {split.name} has empty train/val/OOT")

    thresholds = fit_regime_thresholds(raw_train)
    train = add_regime_features(raw_train, thresholds)
    val = add_regime_features(raw_val, thresholds)
    oot = add_regime_features(raw_oot, thresholds)
    regime_features = [
        "regime_atr_bucket",
        "regime_adx_bucket",
        "regime_range_bucket",
        "regime_abs_vwap_z_bucket",
        "regime_trend_stack",
        "regime_high_vol_trend",
    ]
    features = base_features + regime_features
    model = train_model(train, features, split)

    prob_col = f"prob_{split.name}"
    train[prob_col] = model.predict_proba(train[features])[:, 1]
    val[prob_col] = model.predict_proba(val[features])[:, 1]
    oot[prob_col] = model.predict_proba(oot[features])[:, 1]
    threshold, threshold_metrics, threshold_grid = select_threshold(val, val, prob_col)

    train_selected = train[train[prob_col] >= threshold].copy()
    val_selected = val[val[prob_col] >= threshold].copy()
    oot_selected = oot[oot[prob_col] >= threshold].copy()

    model_path = model_dir / f"{split.name}_model.joblib"
    selected_path = model_dir / f"{split.name}_selected_trades.parquet"
    importance_path = model_dir / f"{split.name}_feature_importance.csv"
    joblib.dump(
        {
            "model": model,
            "features": features,
            "base_features": base_features,
            "regime_features": regime_features,
            "regime_thresholds": thresholds,
            "selected_threshold": threshold,
            "split": split.__dict__,
        },
        model_path,
    )
    pd.concat(
        [
            train_selected.assign(split_part="train"),
            val_selected.assign(split_part="validation"),
            oot_selected.assign(split_part="oot"),
        ],
        ignore_index=True,
    ).to_parquet(selected_path, index=False)
    (
        pd.DataFrame({"feature": features, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .to_csv(importance_path, index=False)
    )

    report = {
        "split": split.__dict__,
        "model_path": str(model_path.relative_to(ROOT)),
        "selected_trades_path": str(selected_path.relative_to(ROOT)),
        "feature_importance_path": str(importance_path.relative_to(ROOT)),
        "features_used": features,
        "regime_thresholds": thresholds,
        "selected_threshold": threshold,
        "threshold_selection": {
            "validation_best": threshold_metrics,
            "grid": threshold_grid,
        },
        "auc": {
            "train": auc_or_none(train, prob_col),
            "validation": auc_or_none(val, prob_col),
            "oot": auc_or_none(oot, prob_col),
        },
        "raw": {
            "train": raw_period_metrics(raw_train),
            "validation": raw_period_metrics(raw_val),
            "oot": raw_period_metrics(raw_oot),
        },
        "selected": {
            "train": period_metrics(train_selected, train, prob_col),
            "validation": period_metrics(val_selected, val, prob_col),
            "oot": period_metrics(oot_selected, oot, prob_col),
        },
        "by_year_selected": by_group_metrics(
            pd.concat([train_selected, val_selected, oot_selected], ignore_index=True),
            "year",
            pd.concat([train, val, oot], ignore_index=True),
            prob_col,
        ),
        "by_side_selected": by_group_metrics(
            pd.concat([train_selected, val_selected, oot_selected], ignore_index=True),
            "side",
            pd.concat([train, val, oot], ignore_index=True),
            prob_col,
        ),
        "by_session_selected": by_group_metrics(
            pd.concat([train_selected, val_selected, oot_selected], ignore_index=True),
            "session_cluster",
            pd.concat([train, val, oot], ignore_index=True),
            prob_col,
        ),
    }
    return report


def by_group_metrics(selected: pd.DataFrame, group_col: str, period: pd.DataFrame, prob_col: str) -> dict[str, Any]:
    if period.empty:
        return {}
    period = period.copy()
    selected = selected.copy()
    if group_col == "year":
        period[group_col] = period["signal_ts"].dt.year
        selected[group_col] = selected["signal_ts"].dt.year if not selected.empty else []
    out: dict[str, Any] = {}
    for value, group_period in period.groupby(group_col, sort=True):
        group_selected = selected[selected[group_col] == value] if not selected.empty else selected
        out[str(value)] = period_metrics(group_selected, group_period, prob_col)
    return out


def month_starts(start: str, end: pd.Timestamp) -> list[pd.Timestamp]:
    months = pd.date_range(pd.Timestamp(start, tz="UTC"), end, freq="MS")
    return list(months)


def rolling_walkforward(df: pd.DataFrame, base_features: list[str]) -> dict[str, Any]:
    rows = []
    selected_all = []
    max_ts = df["signal_ts"].max()
    for test_start in month_starts("2024-01-01", max_ts):
        test_end = test_start + pd.DateOffset(months=1)
        train_start = test_start - pd.DateOffset(months=12)
        val_start = test_start - pd.DateOffset(months=3)
        raw_train_full = df[(df["signal_ts"] >= train_start) & (df["signal_ts"] < test_start)].copy()
        raw_train = raw_train_full[raw_train_full["signal_ts"] < val_start].copy()
        raw_val = raw_train_full[raw_train_full["signal_ts"] >= val_start].copy()
        raw_test = df[(df["signal_ts"] >= test_start) & (df["signal_ts"] < test_end)].copy()
        if len(raw_train) < 200 or len(raw_val) < 30 or len(raw_test) < 5:
            continue
        thresholds = fit_regime_thresholds(raw_train)
        train = add_regime_features(raw_train, thresholds)
        val = add_regime_features(raw_val, thresholds)
        test = add_regime_features(raw_test, thresholds)
        regime_features = [
            "regime_atr_bucket",
            "regime_adx_bucket",
            "regime_range_bucket",
            "regime_abs_vwap_z_bucket",
            "regime_trend_stack",
            "regime_high_vol_trend",
        ]
        features = base_features + regime_features
        split = SplitSpec(
            name=f"wf_{test_start.strftime('%Y_%m')}",
            train_start=train_start.strftime("%Y-%m-%d"),
            train_end=test_start.strftime("%Y-%m-%d"),
            val_start=val_start.strftime("%Y-%m-%d"),
            val_end=test_start.strftime("%Y-%m-%d"),
            oot_start=test_start.strftime("%Y-%m-%d"),
            oot_end=test_end.strftime("%Y-%m-%d"),
            recent_weight_start=(test_start - pd.DateOffset(months=6)).strftime("%Y-%m-%d"),
            recent_weight=1.5,
        )
        model = train_model(train, features, split)
        val["prob_wf"] = model.predict_proba(val[features])[:, 1]
        test["prob_wf"] = model.predict_proba(test[features])[:, 1]
        threshold, threshold_metrics, _ = select_threshold(val, val, "prob_wf")
        selected = test[test["prob_wf"] >= threshold].copy()
        selected["wf_month"] = test_start.strftime("%Y-%m")
        selected_all.append(selected)
        metrics = period_metrics(selected, test, "prob_wf")
        metrics["month"] = test_start.strftime("%Y-%m")
        metrics["threshold"] = threshold
        metrics["validation_threshold_metrics"] = threshold_metrics
        metrics["raw"] = raw_period_metrics(test)
        metrics["auc_test"] = auc_or_none(test, "prob_wf")
        rows.append(metrics)

    selected = pd.concat(selected_all, ignore_index=True) if selected_all else pd.DataFrame()
    summary = period_metrics(selected, df[df["signal_ts"] >= ts("2024-01-01")], "prob_wf") if not selected.empty else {}
    return {
        "summary": summary,
        "months": rows,
        "selected_trades": selected,
    }


def promotion_view(report: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    current = report["splits"]["current_regime"]["selected"]["oot"]
    long_hist = report["splits"]["long_history"]["selected"]["oot"]
    wf = report["rolling_walkforward"]["summary"]
    checks["current_regime_oot_positive_expectancy"] = current["avg_trade"] > 0 and current["pnl_usd"] > 0
    checks["current_regime_oot_frequency_target"] = (
        MIN_TARGET_TRADES_PER_DAY <= current["trades_per_day"] <= MAX_TARGET_TRADES_PER_DAY
    )
    checks["long_history_oot_positive"] = long_hist["pnl_usd"] > 0
    checks["rolling_walkforward_positive"] = bool(wf) and wf.get("pnl_usd", 0.0) > 0 and wf.get("avg_trade", 0.0) > 0
    checks["both_sides_present_current_oot"] = current["long_trades"] > 0 and current["short_trades"] > 0
    return {
        "checks": checks,
        "p1_pass": all(checks.values()),
        "note": "P1 pass means candidate is worth P2 hypertuning, not live promotion.",
    }


def main() -> int:
    cfg = load_config()
    model_dir = ROOT / cfg["outputs"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    gate_report = run_gate_and_assert(cfg)
    df, base_features = load_inputs(cfg)

    split_reports = {
        split.name: evaluate_split(df, base_features, split, model_dir)
        for split in SPLITS
    }
    wf = rolling_walkforward(df, base_features)
    wf_selected_path = model_dir / "rolling_walkforward_selected_trades.parquet"
    if isinstance(wf["selected_trades"], pd.DataFrame) and not wf["selected_trades"].empty:
        wf["selected_trades"].to_parquet(wf_selected_path, index=False)
        wf_path_value: str | None = str(wf_selected_path.relative_to(ROOT))
    else:
        wf_path_value = None
    wf_report = {
        "summary": wf["summary"],
        "months": wf["months"],
        "selected_trades_path": wf_path_value,
    }

    report: dict[str, Any] = {
        "experiment": cfg["experiment"],
        "status": "P1_TRAINED",
        "p0_gate": {
            "status": gate_report["status"],
            "training_allowed": gate_report["training_allowed"],
            "report_path": cfg["outputs"]["training_gate_report"],
        },
        "target": "pnl_usd > 0",
        "base_feature_source": str(WHITELIST_PATH.relative_to(ROOT)),
        "base_feature_count": len(base_features),
        "regime_features": [
            "regime_atr_bucket",
            "regime_adx_bucket",
            "regime_range_bucket",
            "regime_abs_vwap_z_bucket",
            "regime_trend_stack",
            "regime_high_vol_trend",
        ],
        "threshold_policy": {
            "selected_on": "validation_only",
            "thresholds": THRESHOLDS.tolist(),
            "target_trades_per_day": [MIN_TARGET_TRADES_PER_DAY, MAX_TARGET_TRADES_PER_DAY],
        },
        "splits": split_reports,
        "rolling_walkforward": wf_report,
    }
    report["promotion_view"] = promotion_view(report)

    report_path = model_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print("P1 training complete")
    print(f"Wrote {report_path}")
    for name, split_report in split_reports.items():
        val = split_report["selected"]["validation"]
        oot = split_report["selected"]["oot"]
        print(
            f"{name}: threshold={split_report['selected_threshold']:.3f} "
            f"VAL pnl={val['pnl_usd']:.2f} trades={val['selected_trades']} "
            f"tpd={val['trades_per_day']:.2f} | "
            f"OOT pnl={oot['pnl_usd']:.2f} trades={oot['selected_trades']} "
            f"tpd={oot['trades_per_day']:.2f} avg={oot['avg_trade']:.2f}"
        )
    wf_summary = wf_report["summary"]
    if wf_summary:
        print(
            f"rolling WF: pnl={wf_summary['pnl_usd']:.2f} "
            f"trades={wf_summary['selected_trades']} "
            f"tpd={wf_summary['trades_per_day']:.2f} avg={wf_summary['avg_trade']:.2f}"
        )
    print(f"P1 pass: {report['promotion_view']['p1_pass']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

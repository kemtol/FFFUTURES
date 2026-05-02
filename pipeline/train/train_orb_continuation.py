"""
Train LightGBM untuk filter continuation ORB setups.

Target   : y_1r2_120m (continuation side only)
Weighting: time-based exponential decay, half-life=2 tahun
Validation: walk-forward rolling 2y train / 6m val
Holdout  : Jan 2024+ dikunci, tidak disentuh sampai model final

Output default:
  model/ORB_v1.0/
    lgbm_cont_1r2_120m.txt
    lgbm_cont_1r2_120m_meta.json
    CONT_REPORT.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DM_PATH = ROOT / "data/Level_2_Datamart/training_datamart_orb.parquet"
OUT_DIR = ROOT / "model/ORB_v1.0"

TARGET = "y_1r2_120m"
SIDE = "cont"
HOLDOUT_FROM = "2024-01-01"
HALF_LIFE_YEARS = 2

FEATURES = [
    "orb_range_atr_ratio",
    "breakout_strength",
    "atr14_at_entry",
    "price_vs_vwap_pct",
    "adx_14_15m",
    "ema_slope_1h",
    "day_of_week",
    "time_in_session_min",
    "orb_tf",
    "session",
    "breakout_side",
]

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "n_jobs": -1,
}
NUM_ROUNDS = 500
EARLY_STOP = 50


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["session"] = out["session"].map({"tokyo": 0, "london": 1, "us": 2})
    out["orb_tf"] = out["orb_tf"].map({"5m": 5, "15m": 15, "30m": 30})
    return out


def compute_weights(years: pd.Series, half_life: float = HALF_LIFE_YEARS) -> np.ndarray:
    lam = math.log(2) / half_life
    max_yr = years.max()
    return np.exp(-lam * (max_yr - years)).values


def expectancy_net(win_rate: float, rr: float = 2.0, cost_r: float = 0.07) -> float:
    return win_rate * rr - (1 - win_rate) * 1.0 - cost_r


def walk_forward_cv(df: pd.DataFrame, train_years: int = 2, val_months: int = 6):
    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    min_date = dates.min()
    max_date = pd.Timestamp(HOLDOUT_FROM) - pd.Timedelta(days=1)
    fold_start = min_date + pd.DateOffset(years=train_years)

    while fold_start < max_date:
        fold_end = min(fold_start + pd.DateOffset(months=val_months), max_date)
        train_mask = dates < fold_start
        val_mask = (dates >= fold_start) & (dates < fold_end)
        if train_mask.sum() > 100 and val_mask.sum() > 50:
            yield df.index[train_mask].tolist(), df.index[val_mask].tolist()
        fold_start = fold_end


def write_report(meta: dict, fdf: pd.DataFrame, out_dir: Path) -> None:
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    imp = meta["feature_importance_gain"]

    recent_folds = fdf[fdf["period"].str[:4].astype(int) >= 2020]
    recent_auc = recent_folds["auc"].mean() if len(recent_folds) else float("nan")
    recent_exp = recent_folds["exp_net_top40"].mean() if len(recent_folds) else float("nan")

    fold_table = "\n".join(
        f"| {r['fold']:2d} | {r['period']} | {r['auc']:.4f} | {r['wr_all']:.3f} | {r['wr_top40pct']:.3f} | {r['exp_net_top40']:.3f}R |"
        for _, r in fdf.iterrows()
    )

    imp_table = "\n".join(
        f"| {i+1} | `{feat}` | {gain:.0f} |"
        for i, (feat, gain) in enumerate(list(imp.items())[:8])
    )

    report = f"""# Model Report — ORB_v1.0 (Continuation)

Generated: {now}

## Summary

| Metric | Value |
|--------|-------|
| Target | `{meta["target"]}` |
| Side | continuation only |
| Training rows | {meta["n_train"]:,} |
| CV folds | {meta["cv_folds"]} |
| Holdout locked from | {meta["holdout_from"]} |
| Sample weighting | Exponential decay, half-life={meta["half_life_years"]}y |

## CV Performance (all folds)

| Metric | Value |
|--------|-------|
| Mean AUC | {meta["cv_mean_auc"]:.4f} |
| Mean win rate (top 40%) | {meta["cv_mean_wr_top40"]:.3f} |
| Mean exp net (top 40%) | {meta["cv_mean_exp_net"]:.3f}R |

## CV Performance (2020+ folds only — recent regime)

| Metric | Value |
|--------|-------|
| Mean AUC | {recent_auc:.4f} |
| Mean exp net (top 40%) | {recent_exp:.3f}R |

## Walk-Forward Fold Results

| Fold | Period | AUC | WR All | WR Top40% | Exp Net |
|------|--------|-----|--------|-----------|---------|
{fold_table}

## Feature Importance (Gain)

| Rank | Feature | Gain |
|------|---------|------|
{imp_table}
"""

    (out_dir / "CONT_REPORT.md").write_text(report)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dm = pd.read_parquet(DM_PATH)
    dm["date"] = pd.to_datetime(dm["date"])
    dm["year"] = dm["date"].dt.year

    data = dm[(dm["side"] == SIDE) & (dm["date"] < HOLDOUT_FROM)].copy()
    data = encode(data).dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

    print(f"Training set ({SIDE}): {len(data):,} rows ({data['year'].min()}–{data['year'].max()})")

    fold_results = []
    for fold_i, (tr_idx, val_idx) in enumerate(walk_forward_cv(data)):
        tr = data.iloc[tr_idx]
        val = data.iloc[val_idx]

        w_tr = compute_weights(tr["year"])
        dtrain = lgb.Dataset(tr[FEATURES], label=tr[TARGET], weight=w_tr, feature_name=FEATURES)
        dval = lgb.Dataset(val[FEATURES], label=val[TARGET], reference=dtrain)

        model = lgb.train(
            LGBM_PARAMS,
            dtrain,
            num_boost_round=NUM_ROUNDS,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )

        prob = model.predict(val[FEATURES])
        auc = roc_auc_score(val[TARGET], prob)
        wr_all = float(val[TARGET].mean())
        thresh = float(np.percentile(prob, 60))
        mask_hi = prob >= thresh
        wr_hi = float(val.loc[mask_hi, TARGET].mean()) if mask_hi.sum() > 0 else np.nan
        exp_hi = expectancy_net(wr_hi) if not np.isnan(wr_hi) else np.nan

        period = f"{val['date'].min().date()} → {val['date'].max().date()}"
        fold_results.append(
            {
                "fold": fold_i + 1,
                "period": period,
                "n_val": int(len(val)),
                "auc": round(float(auc), 4),
                "wr_all": round(wr_all, 3),
                "wr_top40pct": round(wr_hi, 3) if not np.isnan(wr_hi) else None,
                "exp_net_top40": round(exp_hi, 3) if not np.isnan(exp_hi) else None,
            }
        )
        print(
            f"  Fold {fold_i+1:2d} | {period} | AUC={auc:.4f} | wr_all={wr_all:.3f} | wr_top40={wr_hi:.3f} | exp_net={exp_hi:.3f}R"
        )

    fdf = pd.DataFrame(fold_results)

    w_all = compute_weights(data["year"])
    dtrain_full = lgb.Dataset(data[FEATURES], label=data[TARGET], weight=w_all, feature_name=FEATURES)
    final_model = lgb.train(LGBM_PARAMS, dtrain_full, num_boost_round=NUM_ROUNDS)

    model_path = OUT_DIR / "lgbm_cont_1r2_120m.txt"
    final_model.save_model(str(model_path))

    imp = dict(zip(FEATURES, final_model.feature_importance(importance_type="gain").tolist()))
    imp_sorted = dict(sorted(imp.items(), key=lambda kv: -kv[1]))

    meta = {
        "target": TARGET,
        "side": SIDE,
        "holdout_from": HOLDOUT_FROM,
        "half_life_years": HALF_LIFE_YEARS,
        "features": FEATURES,
        "n_train": int(len(data)),
        "cv_folds": int(len(fold_results)),
        "cv_mean_auc": round(float(fdf["auc"].mean()), 4),
        "cv_mean_wr_top40": round(float(fdf["wr_top40pct"].mean()), 3),
        "cv_mean_exp_net": round(float(fdf["exp_net_top40"].mean()), 3),
        "feature_importance_gain": imp_sorted,
        "fold_results": fold_results,
        "lgbm_params": LGBM_PARAMS,
    }

    meta_path = OUT_DIR / "lgbm_cont_1r2_120m_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    write_report(meta, fdf, OUT_DIR)
    print(f"\nSaved: {model_path}")
    print(f"Saved: {meta_path}")
    print(f"Saved: {OUT_DIR / 'CONT_REPORT.md'}")


if __name__ == "__main__":
    main()

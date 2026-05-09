"""
Train LightGBM untuk filter reversal ORB setups.

Target   : y_1r2_120m (reversal side only)
Weighting: time-based exponential decay, half-life=2 tahun
Validation: walk-forward rolling 2y train / 6m val
Holdout  : Jan 2024+ dikunci, tidak disentuh sampai model final

Output: model/ORB_v{MAJOR}.{MINOR}/
          lgbm_rev_1r2_120m.txt
          lgbm_rev_1r2_120m_meta.json
          REPORT.md

Versioning:
  Major bump: ganti target, feature set, atau training scheme
  Minor bump: ganti hyperparameter, weighting, threshold

Usage:
  python train_orb_reversal.py           # auto minor bump
  python train_orb_reversal.py --major   # major bump
  python train_orb_reversal.py --version 2.1  # set explicit version
"""

import json
import math
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DM_PATH  = ROOT / "data/Level_2_Datamart/training_datamart_orb.parquet"
MODEL_BASE = ROOT / "model"

TARGET          = "y_1r2_120m"
HOLDOUT_FROM    = "2024-01-01"
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
    "objective":        "binary",
    "metric":           "auc",
    "learning_rate":    0.05,
    "num_leaves":       31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "verbose":         -1,
    "n_jobs":          -1,
}
NUM_ROUNDS = 500
EARLY_STOP = 50


# ── versioning ────────────────────────────────────────────────────────────────

def resolve_version(major_bump: bool = False, explicit: str | None = None) -> str:
    if explicit:
        return explicit

    existing = sorted(MODEL_BASE.glob("ORB_v*"))
    if not existing:
        return "1.0"

    versions = []
    for p in existing:
        m = re.match(r"ORB_v(\d+)\.(\d+)", p.name)
        if m:
            versions.append((int(m.group(1)), int(m.group(2))))

    if not versions:
        return "1.0"

    last_major, last_minor = max(versions)
    if major_bump:
        return f"{last_major + 1}.0"
    return f"{last_major}.{last_minor + 1}"


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_weights(years: pd.Series, half_life: float = HALF_LIFE_YEARS) -> np.ndarray:
    lam    = math.log(2) / half_life
    max_yr = years.max()
    return np.exp(-lam * (max_yr - years)).values


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["session"] = df["session"].map({"tokyo": 0, "london": 1, "us": 2})
    df["orb_tf"]  = df["orb_tf"].map({"5m": 5, "15m": 15, "30m": 30})
    return df


def expectancy_net(win_rate: float, rr: float = 2.0, cost_r: float = 0.07) -> float:
    return win_rate * rr - (1 - win_rate) * 1.0 - cost_r


def walk_forward_cv(df: pd.DataFrame, train_years: int = 2, val_months: int = 6):
    df    = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    min_date  = dates.min()
    max_date  = pd.Timestamp(HOLDOUT_FROM) - pd.Timedelta(days=1)
    fold_start = min_date + pd.DateOffset(years=train_years)

    while fold_start < max_date:
        fold_end = min(fold_start + pd.DateOffset(months=val_months), max_date)
        train_mask = dates < fold_start
        val_mask   = (dates >= fold_start) & (dates < fold_end)
        if train_mask.sum() > 100 and val_mask.sum() > 50:
            yield df.index[train_mask].tolist(), df.index[val_mask].tolist()
        fold_start = fold_end


# ── report generation ─────────────────────────────────────────────────────────

def write_report(out_dir: Path, version: str, meta: dict, fdf: pd.DataFrame) -> None:
    imp  = meta["feature_importance_gain"]
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")

    recent_folds = fdf[fdf["period"].str[:4].astype(int) >= 2020]
    recent_auc   = recent_folds["auc"].mean() if len(recent_folds) else float("nan")
    recent_exp   = recent_folds["exp_net_top40"].mean() if len(recent_folds) else float("nan")

    fold_table = "\n".join(
        f"| {r['fold']:2d} | {r['period']} | {r['auc']:.4f} | "
        f"{r['wr_all']:.3f} | {r['wr_top40pct']:.3f} | {r['exp_net_top40']:.3f}R |"
        for _, r in fdf.iterrows()
    )

    imp_table = "\n".join(
        f"| {i+1} | `{feat}` | {gain:.0f} |"
        for i, (feat, gain) in enumerate(list(imp.items())[:8])
    )

    report = f"""# Model Report — ORB_v{version}

Generated: {now}

## Summary

| Metric | Value |
|--------|-------|
| Version | {version} |
| Target | `{meta["target"]}` |
| Side | reversal only |
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

## Config

```json
{json.dumps({"lgbm_params": LGBM_PARAMS, "features": FEATURES}, indent=2)}
```

## Notes

- Baseline reversal win rate (no filter): ~40% gross, ~33% net of costs
- Top 40% threshold targets setups where model assigns highest probability
- 2017–2018 regime anomaly (wr ~65%) is downweighted via exponential decay
- Holdout 2024+ not evaluated — reserved for final model validation
"""

    (out_dir / "REPORT.md").write_text(report)


def write_readme(out_dir: Path, version: str, meta: dict) -> None:
    readme = f"""# ORB Reversal Model v{version}

LightGBM binary classifier untuk filter reversal ORB setups pada MGC (Micro Gold Futures).

## Quickstart

```python
import lightgbm as lgb
import pandas as pd

model = lgb.Booster(model_file="lgbm_rev_1r2_120m.txt")

# Features required (in order):
# {meta["features"]}

prob = model.predict(X)  # probability TP 2R hit within 120m
signal = prob >= 0.50    # threshold — lihat REPORT.md untuk kalibrasi
```

## Files

| File | Deskripsi |
|------|-----------|
| `lgbm_rev_1r2_120m.txt` | Model LightGBM |
| `lgbm_rev_1r2_120m_meta.json` | Metadata, CV results, feature importance |
| `REPORT.md` | Laporan lengkap training run ini |

## Key Numbers

- CV Mean AUC: **{meta["cv_mean_auc"]}**
- CV Mean exp net (top 40%): **{meta["cv_mean_exp_net"]}R**
- Training data: 2010–2023, {meta["n_train"]:,} reversal events
- Holdout: Jan 2024+ (belum dievaluasi)
"""
    (out_dir / "README.md").write_text(readme)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args         = sys.argv[1:]
    major_bump   = "--major" in args
    explicit_ver = next((a.split("=")[1] for a in args if a.startswith("--version=")), None)
    if not explicit_ver:
        explicit_ver = args[args.index("--version") + 1] if "--version" in args else None

    version  = resolve_version(major_bump, explicit_ver)
    out_dir  = MODEL_BASE / f"ORB_v{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Version: ORB_v{version}  →  {out_dir}")

    print("Loading datamart...")
    dm = pd.read_parquet(DM_PATH)
    dm["date"] = pd.to_datetime(dm["date"])
    dm["year"] = dm["date"].dt.year

    rev = dm[(dm["side"] == "rev") & (dm["date"] < HOLDOUT_FROM)].copy()
    rev = encode(rev)
    rev = rev.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

    print(f"Training set: {len(rev):,} rows ({rev['year'].min()}–{rev['year'].max()})")
    print(f"Holdout (locked): {(dm['date'] >= HOLDOUT_FROM).sum()//2:,} reversal events from {HOLDOUT_FROM}")

    # ── Walk-forward CV ───────────────────────────────────────────────────────
    print("\nWalk-forward cross-validation...")
    fold_results = []

    for fold_i, (tr_idx, val_idx) in enumerate(walk_forward_cv(rev)):
        tr  = rev.iloc[tr_idx]
        val = rev.iloc[val_idx]

        w_tr = compute_weights(tr["year"])
        X_tr, y_tr   = tr[FEATURES],  tr[TARGET]
        X_val, y_val = val[FEATURES], val[TARGET]

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=FEATURES)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            LGBM_PARAMS, dtrain,
            num_boost_round=NUM_ROUNDS,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )

        prob    = model.predict(X_val)
        auc     = roc_auc_score(y_val, prob)
        wr_all  = y_val.mean()
        thresh  = np.percentile(prob, 60)
        mask_hi = prob >= thresh
        wr_hi   = y_val[mask_hi].mean() if mask_hi.sum() > 0 else np.nan
        exp_hi  = expectancy_net(wr_hi)  if not np.isnan(wr_hi) else np.nan

        period = f"{val['date'].min().date()} → {val['date'].max().date()}"
        fold_results.append({
            "fold": fold_i + 1, "period": period, "n_val": len(val),
            "auc": round(auc, 4), "wr_all": round(wr_all, 3),
            "wr_top40pct": round(wr_hi, 3) if not np.isnan(wr_hi) else None,
            "exp_net_top40": round(exp_hi, 3) if not np.isnan(exp_hi) else None,
        })
        print(f"  Fold {fold_i+1:2d} | {period} | AUC={auc:.4f} | "
              f"wr_all={wr_all:.3f} | wr_top40%={wr_hi:.3f} | exp_net={exp_hi:.3f}R")

    fdf = pd.DataFrame(fold_results)
    print(f"\nCV Summary:")
    print(f"  Mean AUC        : {fdf['auc'].mean():.4f} ± {fdf['auc'].std():.4f}")
    print(f"  Mean wr_top40%  : {fdf['wr_top40pct'].mean():.3f}")
    print(f"  Mean exp_net    : {fdf['exp_net_top40'].mean():.3f}R")

    # ── Final model ───────────────────────────────────────────────────────────
    print("\nTraining final model on full pre-holdout data...")
    w_all       = compute_weights(rev["year"])
    dtrain_full = lgb.Dataset(rev[FEATURES], label=rev[TARGET],
                               weight=w_all, feature_name=FEATURES)
    final_model = lgb.train(LGBM_PARAMS, dtrain_full, num_boost_round=NUM_ROUNDS)

    model_path = out_dir / "lgbm_rev_1r2_120m.txt"
    final_model.save_model(str(model_path))

    imp        = dict(zip(FEATURES, final_model.feature_importance(importance_type="gain").tolist()))
    imp_sorted = dict(sorted(imp.items(), key=lambda x: -x[1]))

    meta = {
        "version":         version,
        "target":          TARGET,
        "side":            "rev",
        "holdout_from":    HOLDOUT_FROM,
        "half_life_years": HALF_LIFE_YEARS,
        "features":        FEATURES,
        "n_train":         len(rev),
        "cv_folds":        len(fold_results),
        "cv_mean_auc":     round(fdf["auc"].mean(), 4),
        "cv_mean_wr_top40": round(fdf["wr_top40pct"].mean(), 3),
        "cv_mean_exp_net": round(fdf["exp_net_top40"].mean(), 3),
        "feature_importance_gain": imp_sorted,
        "fold_results":    fold_results,
        "lgbm_params":     LGBM_PARAMS,
    }

    meta_path = out_dir / "lgbm_rev_1r2_120m_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    write_report(out_dir, version, meta, fdf)
    write_readme(out_dir, version, meta)

    print(f"\nSaved to {out_dir}/")
    print(f"  lgbm_rev_1r2_120m.txt")
    print(f"  lgbm_rev_1r2_120m_meta.json")
    print(f"  REPORT.md")
    print(f"  README.md")
    print(f"\nTop features by gain:")
    for feat, gain in list(imp_sorted.items())[:5]:
        print(f"  {feat:30s} {gain:.1f}")


if __name__ == "__main__":
    main()

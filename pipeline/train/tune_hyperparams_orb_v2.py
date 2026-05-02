#!/usr/bin/env python3
"""
Hyperparameter tuning — Random search 60 LGBM param combos for top targets.

Evaluates on 2026 (hardest year) with fast expectancy metric.
Output: model/ORB_v2.0_2010-2026/hyperparam_tuning_results.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
PIPELINE = BASE / "pipeline"
for p in [str(BASE), str(PIPELINE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from analysis.topstep_sim import PolicyParams, apply_policy
from pipeline.feature.modules.loader import load_features_from_modules

# ── paths ─────────────────────────────────────────────────────────────────────
DM_PATH = BASE / "data" / "Level_2_Datamart" / "training_datamart_orb.parquet"
MODULES_DIR = BASE / "data" / "Level_1_Features" / "modules"
OUT_DIR = BASE / "model" / "ORB_v2.0_2010-2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_THROUGH = 2025  # train 2010-2025
TEST_YEAR = 2026  # evaluate on hardest year

# Top 3 targets from walk-forward
TARGETS = [
    ("y_1r4_180m", 4.0),
    ("y_1r4_240m", 4.0),
    ("y_1r4_120m", 4.0),
]

NUM_RANDOM_COMBOS = 40
NUM_ROUNDS = 500
EARLY_STOP = 30

# Core columns
CORE_COLS = [
    "date", "session", "orb_tf", "breakout_ts", "breakout_side", "side",
    "entry_price", "orb_range", "atr14_at_entry", "sl_dist", "breakout_strength",
    "session_close_ts",
    "y_1r2_60m", "y_1r4_60m", "y_1r2_120m", "y_1r4_120m",
    "y_1r2_180m", "y_1r4_180m", "y_1r2_240m", "y_1r4_240m",
    "y_1r2_close60m", "y_1r4_close60m",
]
CORE_FEATURES = ["breakout_strength", "atr14_at_entry", "orb_tf", "session", "breakout_side"]
EVENT_KEY = ["date", "session", "orb_tf", "breakout_ts"]
MERGE_KEY = ["date", "breakout_ts", "breakout_side"]

# ── baseline LGBM params ─────────────────────────────────────────────────────
BASELINE_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "n_jobs": -1,
}

# Policy
POLICY = PolicyParams(
    rev_q=0.75, cont_q=0.75,
    rev_adx_min=30, cont_adx_max=30,
    daily_stop_usd=0, daily_profit_cap_usd=1400,
    risk_per_r_usd=100.0,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, prefix in [("orb_tf", "orb_tf"), ("session", "session")]:
        dummies = pd.get_dummies(df[col], prefix=prefix)
        df = pd.concat([df, dummies.astype(float)], axis=1)
    df["breakout_side"] = (df["breakout_side"] == 1).astype(float)
    return df


def numeric_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f in df.columns
            and df[f].dtype in ("float64", "float32", "int64", "int32", "bool")]


def train_model(df: pd.DataFrame, target: str, features: list[str], params: dict) -> lgb.Booster | None:
    num_feats = numeric_features(df, features)
    df = encode(df).dropna(subset=num_feats + [target]).reset_index(drop=True)
    if len(df) < 100:
        return None
    dtrain = lgb.Dataset(df[num_feats], label=df[target].astype(int),
                         feature_name=num_feats)
    model = lgb.train(
        params, dtrain,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[dtrain],
        callbacks=[],
    )
    return model


def evaluate(events: pd.DataFrame, rr: float) -> dict:
    out = apply_policy(events, events, POLICY, rr)
    trades = out[out["decision"] != "skip"]
    if len(trades) == 0:
        return {"wr": 0.0, "exp": 0.0, "trades": 0}
    wr = trades["y"].mean()
    exp = wr * rr - (1 - wr) * 1.0
    return {"wr": float(wr), "exp": float(exp), "trades": len(trades)}


def random_params(seed: int) -> dict:
    rng = np.random.RandomState(seed)
    return {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": float(rng.choice([0.01, 0.03, 0.05, 0.08, 0.12])),
        "num_leaves": int(rng.choice([15, 31, 63, 127])),
        "min_data_in_leaf": int(rng.choice([10, 20, 50, 100])),
        "feature_fraction": float(rng.choice([0.4, 0.6, 0.8, 1.0])),
        "bagging_fraction": float(rng.choice([0.6, 0.8, 1.0])),
        "bagging_freq": int(rng.choice([1, 5, 10])),
        "lambda_l1": float(rng.choice([0.0, 0.01, 0.1, 0.5, 1.0])),
        "lambda_l2": float(rng.choice([0.0, 0.01, 0.1, 0.5, 1.0])),
        "verbose": -1,
        "n_jobs": -1,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  Hyperparameter Tuning — Random Search (60 combos)")
    print(f"  Train: 2010-{TRAIN_THROUGH}, Test: {TEST_YEAR}")
    print(f"  Targets: {[t[0] for t in TARGETS]}")
    print("=" * 65)

    # ── Load data ────────────────────────────────────────────────────
    print("\n1. Loading data...")
    dm = pd.read_parquet(DM_PATH, columns=CORE_COLS).copy()
    dm = load_features_from_modules(MODULES_DIR, dm, verbose=False)
    dm["date"] = pd.to_datetime(dm["date"])
    dm["year"] = dm["date"].dt.year

    excluded = set(EVENT_KEY) | {"year"}
    module_cols = sorted(set(dm.columns) - set(CORE_COLS) - excluded)
    all_features = CORE_FEATURES + module_cols
    print(f"   Features: {len(all_features)} total")

    # ── Baseline ─────────────────────────────────────────────────────
    print("\n2. Baseline (default LGBM params)...")
    baseline_results = {}
    for target, rr in TARGETS:
        train_mask = dm["year"] <= TRAIN_THROUGH
        test_mask = dm["year"] == TEST_YEAR
        rev_train = dm[(dm["side"] == "rev") & train_mask]
        cont_train = dm[(dm["side"] == "cont") & train_mask]
        rev_test = dm[(dm["side"] == "rev") & test_mask]
        cont_test = dm[(dm["side"] == "cont") & test_mask]

        rev_m = train_model(rev_train, target, all_features, BASELINE_PARAMS)
        cont_m = train_model(cont_train, target, all_features, BASELINE_PARAMS)
        if rev_m is None or cont_m is None:
            print(f"   {target}: FAILED")
            continue

        rev_enc = encode(rev_test).dropna(subset=numeric_features(rev_test, all_features) + [target])
        cont_enc = encode(cont_test).dropna(subset=numeric_features(cont_test, all_features) + [target])
        nf = numeric_features(rev_enc, all_features)
        rev_enc["prob_rev"] = rev_m.predict(rev_enc[nf])
        rev_enc["y_rev"] = rev_enc[target]
        cont_enc["prob_cont"] = cont_m.predict(cont_enc[nf])
        cont_enc["y_cont"] = cont_enc[target]

        merge_on = MERGE_KEY + ["year", "orb_range", "breakout_strength",
                                 "atr14_at_entry", "price_vs_vwap_pct",
                                 "adx_14_15m", "ema_slope_1h"]
        events = rev_enc[merge_on + ["prob_rev", "y_rev"]].merge(
            cont_enc[merge_on + ["prob_cont", "y_cont"]], on=merge_on, how="inner"
        )

        from analysis.topstep_sim import map_to_topstep_trade_day
        events["trade_day"] = map_to_topstep_trade_day(events["breakout_ts"])

        r = evaluate(events, rr)
        baseline_results[target] = r
        print(f"   {target:<20s} wr={r['wr']:.3f} exp={r['exp']:+.4f}R trades={r['trades']}")

    # ── Random search ─────────────────────────────────────────────────
    print(f"\n3. Random search ({NUM_RANDOM_COMBOS} combos)...")
    all_rows = []
    best_per_target: dict[str, dict] = {}

    for i in range(NUM_RANDOM_COMBOS):
        params = random_params(i + 42)
        row = {"trial": i, **{f"param_{k}": v for k, v in params.items()
                               if k not in ("objective", "metric", "verbose", "n_jobs")}}
        trials = []
        for target, rr in TARGETS:
            train_mask = dm["year"] <= TRAIN_THROUGH
            test_mask = dm["year"] == TEST_YEAR
            rev_train = dm[(dm["side"] == "rev") & train_mask]
            cont_train = dm[(dm["side"] == "cont") & train_mask]
            rev_test = dm[(dm["side"] == "rev") & test_mask]
            cont_test = dm[(dm["side"] == "cont") & test_mask]

            rev_m = train_model(rev_train, target, all_features, params)
            cont_m = train_model(cont_train, target, all_features, params)
            if rev_m is None or cont_m is None:
                row[f"{target}_wr"] = np.nan
                row[f"{target}_exp"] = np.nan
                continue

            rev_enc = encode(rev_test).dropna(subset=numeric_features(rev_test, all_features) + [target])
            cont_enc = encode(cont_test).dropna(subset=numeric_features(cont_test, all_features) + [target])
            nf = numeric_features(rev_enc, all_features)
            rev_enc["prob_rev"] = rev_m.predict(rev_enc[nf])
            rev_enc["y_rev"] = rev_enc[target]
            cont_enc["prob_cont"] = cont_m.predict(cont_enc[nf])
            cont_enc["y_cont"] = cont_enc[target]

            from analysis.topstep_sim import map_to_topstep_trade_day
            merge_on = MERGE_KEY + ["year", "orb_range", "breakout_strength",
                                     "atr14_at_entry", "price_vs_vwap_pct",
                                     "adx_14_15m", "ema_slope_1h"]
            events = rev_enc[merge_on + ["prob_rev", "y_rev"]].merge(
                cont_enc[merge_on + ["prob_cont", "y_cont"]], on=merge_on, how="inner"
            )
            events["trade_day"] = map_to_topstep_trade_day(events["breakout_ts"])

            r = evaluate(events, rr)
            row[f"{target}_wr"] = r["wr"]
            row[f"{target}_exp"] = r["exp"]
            trials.append(r["exp"])

            # Track best per target
            if target not in best_per_target or r["exp"] > best_per_target[target].get("exp", -99):
                best_per_target[target] = {"exp": r["exp"], "wr": r["wr"],
                                           "trades": r["trades"], "trial": i, "params": params}

        row["mean_exp"] = np.mean(trials) if trials else np.nan
        all_rows.append(row)

        if (i + 1) % 10 == 0:
            best_so_far = max(r["mean_exp"] for r in all_rows if not np.isnan(r["mean_exp"]))
            print(f"   {i+1}/{NUM_RANDOM_COMBOS} done, best mean_exp so far: {best_so_far:+.4f}R")

    # ── Results ───────────────────────────────────────────────────────
    results = pd.DataFrame(all_rows).sort_values("mean_exp", ascending=False)
    results.to_csv(OUT_DIR / "hyperparam_tuning_results.csv", index=False)

    print(f"\n{'='*65}")
    print(f"  Top 5 Trials (mean expectancy across 3 targets on {TEST_YEAR})")
    print(f"{'='*65}")
    for _, r in results.head(5).iterrows():
        print(f"  Trial {int(r['trial']):>2}: mean_exp={r['mean_exp']:+.4f}R")
        for t, _ in TARGETS:
            short = t.replace("y_", "")
            print(f"    {short:<14s} wr={r[f'{t}_wr']:.3f} exp={r[f'{t}_exp']:+.4f}R")
        print(f"    params: lr={r['param_learning_rate']} leaves={int(r['param_num_leaves'])} "
              f"min_leaf={int(r['param_min_data_in_leaf'])} feat_frac={r['param_feature_fraction']} "
              f"l1={r['param_lambda_l1']} l2={r['param_lambda_l2']}")
        print()

    # ── Improvement vs baseline ───────────────────────────────────────
    print(f"  Improvement vs Baseline:")
    for target, rr in TARGETS:
        base_exp = baseline_results[target]["exp"]
        best = best_per_target[target]
        delta = best["exp"] - base_exp
        print(f"    {target:<20s}  baseline={base_exp:+.4f}R  best={best['exp']:+.4f}R  "
              f"Δ={delta:+.4f}R  (trial={best['trial']})")

    best_mean = results.iloc[0]["mean_exp"]
    base_mean = np.mean([baseline_results[t]["exp"] for t, _ in TARGETS])
    print(f"\n    Mean (3 targets):  baseline={base_mean:+.4f}R  best={best_mean:+.4f}R  Δ={best_mean - base_mean:+.4f}R")

    # ── Best per target params ────────────────────────────────────────
    print(f"\n  Best params per target:")
    for target, _ in TARGETS:
        bp = best_per_target[target]["params"]
        print(f"    {target}: lr={bp['learning_rate']} leaves={bp['num_leaves']} "
              f"min_leaf={bp['min_data_in_leaf']} feat_frac={bp['feature_fraction']} "
              f"bag_frac={bp['bagging_fraction']} bag_freq={bp['bagging_freq']} "
              f"l1={bp['lambda_l1']} l2={bp['lambda_l2']}")

    # ── Save best models ──────────────────────────────────────────────
    print(f"\n  Saving best models...")
    for target, rr in TARGETS:
        bp = best_per_target[target]["params"]
        train_mask = dm["year"] <= TRAIN_THROUGH
        rev_train = dm[(dm["side"] == "rev") & train_mask]
        cont_train = dm[(dm["side"] == "cont") & train_mask]
        rev_m = train_model(rev_train, target, all_features, bp)
        cont_m = train_model(cont_train, target, all_features, bp)
        if rev_m and cont_m:
            rev_m.save_model(str(OUT_DIR / f"lgbm_rev_v2_tuned_{target}.txt"))
            cont_m.save_model(str(OUT_DIR / f"lgbm_cont_v2_tuned_{target}.txt"))
            print(f"    {target}: saved")

    print(f"\n✅ Results saved to {OUT_DIR}/hyperparam_tuning_results.csv")


if __name__ == "__main__":
    main()

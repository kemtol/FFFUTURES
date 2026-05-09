#!/usr/bin/env python3
"""
Train ORB_v2.0 — Full 42-feature walk-forward training.

Key differences from sweep:
- Uses ALL 42 features from 7 modules (not per-cycle AB testing)
- Walk-forward expanding windows: train up to year N-1, test year N
- Proper out-of-sample evaluation per window → aggregated performance
- Saves rev + cont models for all 10 targets

Output: model/ORB_v2.0/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
PIPELINE = BASE / "pipeline"
for p in [str(BASE), str(PIPELINE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from analysis.topstep_sim import (
    PolicyParams,
    apply_policy,
    map_to_topstep_trade_day,
)
from pipeline.orb_ml.features.modules.loader import load_features_from_modules

# ── paths ─────────────────────────────────────────────────────────────────────
DM_PATH = BASE / "data" / "Level_2_Datamart" / "training_datamart_orb.parquet"
MODULES_DIR = BASE / "data" / "Level_1_Features" / "modules"
OUT_DIR = BASE / "model" / "ORB_v2.0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────
LGBM_PARAMS = {
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
NUM_ROUNDS = 500
EARLY_STOP = 30

# Walk-forward windows: train up to year N-1, test year N
# Focus on 2020-2026 (regime we care about)
WALK_FORWARD_WINDOWS = [
    (None, 2019, 2020),
    (None, 2020, 2021),
    (None, 2021, 2022),
    (None, 2022, 2023),
    (None, 2023, 2024),
    (None, 2024, 2025),
    (None, 2025, 2026),
]

# Labels to train
LABELS = [
    ("y_1r2_60m", 2.0),
    ("y_1r4_60m", 4.0),
    ("y_1r2_120m", 2.0),
    ("y_1r4_120m", 4.0),
    ("y_1r2_180m", 2.0),
    ("y_1r4_180m", 4.0),
    ("y_1r2_240m", 2.0),
    ("y_1r4_240m", 4.0),
    ("y_1r2_close60m", 2.0),
    ("y_1r4_close60m", 4.0),
]

EVENT_KEY = ["date", "session", "orb_tf", "breakout_ts"]
MERGE_KEY = ["date", "breakout_ts", "breakout_side"]

# Core columns loaded from datamart
CORE_COLS = [
    "date", "session", "orb_tf", "breakout_ts", "breakout_side", "side",
    "entry_price", "orb_range", "atr14_at_entry", "sl_dist", "breakout_strength",
    "session_close_ts",
    "y_1r2_60m", "y_1r4_60m", "y_1r2_120m", "y_1r4_120m",
    "y_1r2_180m", "y_1r4_180m", "y_1r2_240m", "y_1r4_240m",
    "y_1r2_close60m", "y_1r4_close60m",
]

CORE_FEATURES = ["breakout_strength", "atr14_at_entry", "orb_tf", "session", "breakout_side"]

# ── helpers ───────────────────────────────────────────────────────────────────


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    orb_dummies = pd.get_dummies(df["orb_tf"], prefix="orb_tf")
    sess_dummies = pd.get_dummies(df["session"], prefix="session")
    df = pd.concat([df, orb_dummies.astype(float), sess_dummies.astype(float)], axis=1)
    df["breakout_side"] = (df["breakout_side"] == 1).astype(float)
    return df


def _numeric_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f in df.columns
            and df[f].dtype in ("float64", "float32", "int64", "int32", "bool")]


def train_lgbm(df: pd.DataFrame, target: str, features: list[str]) -> lgb.Booster | None:
    """Train LGBM on *df* (must have date column)."""
    num_feats = _numeric_features(df, features)
    df = encode(df).dropna(subset=num_feats + [target]).reset_index(drop=True)
    if len(df) < 100:
        return None

    w = np.ones(len(df))
    dtrain = lgb.Dataset(df[num_feats], label=df[target].astype(int),
                         weight=w, feature_name=num_feats)
    model = lgb.train(
        LGBM_PARAMS, dtrain,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[dtrain],
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


# Default policy — reasonable middle ground (not optimized per window)
DEFAULT_POLICY = PolicyParams(
    rev_q=0.75, cont_q=0.75,
    rev_adx_min=30, cont_adx_max=30,
    daily_stop_usd=0, daily_profit_cap_usd=1400,
    risk_per_r_usd=100.0,
)


def evaluate_fast(events: pd.DataFrame, p: PolicyParams, rr: float) -> dict:
    """Fast evaluation — win rate + expectancy only, no window simulation."""
    out = apply_policy(events, events, p, rr)
    trades = out[out["decision"] != "skip"]
    if len(trades) == 0:
        return {"wr": 0.0, "expectancy_r": 0.0, "n_trades": 0, "n_events": len(events)}
    wr = trades["y"].mean()
    exp_r = wr * rr - (1 - wr) * 1.0
    return {"wr": float(wr), "expectancy_r": float(exp_r),
            "n_trades": len(trades), "n_events": len(events)}


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 65)
    print("  ORB_v2.0 — Walk-Forward Training (42 Features)")
    print("=" * 65)

    # ── 1. Load all data ──────────────────────────────────────────────
    print("\n1. Loading datamart + all 7 feature modules...")
    dm = pd.read_parquet(DM_PATH, columns=CORE_COLS).copy()
    dm = load_features_from_modules(MODULES_DIR, dm, verbose=False)
    dm["date"] = pd.to_datetime(dm["date"])
    dm["year"] = dm["date"].dt.year

    # Feature list
    excluded = set(EVENT_KEY) | {"year"}
    module_cols = sorted(set(dm.columns) - set(CORE_COLS) - excluded)
    all_features = CORE_FEATURES + module_cols
    print(f"   Datamart: {len(dm):,} rows, Features: {len(all_features)} total")
    print(f"   Modules: {len(list(MODULES_DIR.glob('*_features.parquet')))}")

    # ── 2. Walk-forward loop ──────────────────────────────────────────
    all_wf_rows: list[dict] = []
    # Final model store (trained on 2010-2025)
    final_models: dict[str, dict[str, lgb.Booster]] = {}

    for train_from, train_to_year, test_year in WALK_FORWARD_WINDOWS:
        print(f"\n{'─'*60}")
        print(f"  Window: train → {train_to_year}, test {test_year}")

        train_mask = dm["year"] <= train_to_year
        test_mask = dm["year"] == test_year

        if test_mask.sum() < 20:
            print(f"  SKIP — only {test_mask.sum()} test events")
            continue

        for target, rr in LABELS:
            rev_df = dm[(dm["side"] == "rev") & train_mask].copy()
            cont_df = dm[(dm["side"] == "cont") & train_mask].copy()
            rev_test = dm[(dm["side"] == "rev") & test_mask].copy()
            cont_test = dm[(dm["side"] == "cont") & test_mask].copy()

            if target not in rev_df.columns or rev_df[target].isna().all():
                continue

            # Train
            print(f"    {target}...", end=" ", flush=True)
            rev_model = train_lgbm(rev_df, target, all_features)
            cont_model = train_lgbm(cont_df, target, all_features)
            if rev_model is None or cont_model is None:
                continue

            # Save final model (last window only)
            if test_year == 2026:
                final_models[target] = {"rev": rev_model, "cont": cont_model}

            # Predict on test
            rev_enc = encode(rev_test).dropna(subset=_numeric_features(rev_test, all_features) + [target])
            cont_enc = encode(cont_test).dropna(subset=_numeric_features(cont_test, all_features) + [target])

            num_feats = _numeric_features(rev_enc, all_features)
            rev_enc["prob_rev"] = rev_model.predict(rev_enc[num_feats])
            rev_enc["y_rev"] = rev_enc[target]
            cont_enc["prob_cont"] = cont_model.predict(cont_enc[num_feats])
            cont_enc["y_cont"] = cont_enc[target]

            merge_on = MERGE_KEY + ["year", "orb_range", "breakout_strength",
                                     "atr14_at_entry", "price_vs_vwap_pct",
                                     "adx_14_15m", "ema_slope_1h"]
            events = rev_enc[merge_on + ["prob_rev", "y_rev"]].merge(
                cont_enc[merge_on + ["prob_cont", "y_cont"]], on=merge_on, how="inner"
            )
            events["trade_day"] = map_to_topstep_trade_day(events["breakout_ts"])
            events = events.sort_values(["date", "breakout_ts"]).reset_index(drop=True)

            if len(events) < 10:
                continue

            # Fast evaluation — win rate + expectancy
            result = evaluate_fast(events, DEFAULT_POLICY, rr)

            n = len(events)
            wr_rev = rev_enc["y_rev"].mean()
            wr_cont = cont_enc["y_cont"].mean()
            exp_rev = wr_rev * rr - (1 - wr_rev) * 1.0
            exp_cont = wr_cont * rr - (1 - wr_cont) * 1.0

            all_wf_rows.append({
                "test_year": test_year,
                "target": target,
                "rr": rr,
                "n_events": n,
                "n_trades": result["n_trades"],
                "wr_rev_base": wr_rev,
                "wr_cont_base": wr_cont,
                "exp_rev_base_r": exp_rev,
                "exp_cont_base_r": exp_cont,
                "wr_policy": result["wr"],
                "expectancy_policy_r": result["expectancy_r"],
            })
            print(f"wr={result['wr']:.3f} exp={result['expectancy_r']:+.3f}R trades={result['n_trades']}")

    # ── 3. Summary ─────────────────────────────────────────────────────
    if not all_wf_rows:
        print("\n❌ No results.")
        return

    wf = pd.DataFrame(all_wf_rows)
    wf.to_csv(OUT_DIR / "walk_forward_results.csv", index=False)
    print(f"\n✅ Walk-forward results saved ({len(wf)} rows)")

    # Aggregated by target (mean across test years)
    agg = wf.groupby("target").agg(
        mean_exp_policy=("expectancy_policy_r", "mean"),
        mean_wr_policy=("wr_policy", "mean"),
        mean_trades=("n_trades", "mean"),
        worst_exp=("expectancy_policy_r", "min"),
        best_exp=("expectancy_policy_r", "max"),
    ).sort_values("mean_exp_policy", ascending=False)

    print(f"\n{'='*65}")
    print(f"  Walk-Forward Summary (mean across 2020-2026)")
    print(f"{'='*65}")
    print(f"  {'Target':<20s} {'Mean Exp(R)':>11s} {'Mean WR':>8s} {'Mean Trades':>11s}")
    print(f"  {'-'*20}  {'-'*11}  {'-'*8}  {'-'*11}")
    for target, row in agg.iterrows():
        print(f"  {target:<20s} {row['mean_exp_policy']:>+10.4f}R {row['mean_wr_policy']:>7.1%} {row['mean_trades']:>11.0f}")
    agg.to_csv(OUT_DIR / "walk_forward_aggregated.csv")

    # ── 4. Save final models (trained on 2010-2025) ───────────────────
    print(f"\n{'─'*60}")
    print("  Saving final models (trained on 2010-2025)")
    for target, models in final_models.items():
        rev_path = OUT_DIR / f"lgbm_rev_v2_{target}.txt"
        cont_path = OUT_DIR / f"lgbm_cont_v2_{target}.txt"
        models["rev"].save_model(str(rev_path))
        models["cont"].save_model(str(cont_path))
        print(f"  ✅ {target}: rev + cont saved")

    # ── 5. Metadata ────────────────────────────────────────────────────
    metadata = {
        "version": "2.0",
        "features": len(all_features),
        "feature_list": all_features,
        "modules": sorted([f.name for f in MODULES_DIR.glob("*_features.parquet")]),
        "lgbm_params": LGBM_PARAMS,
        "walk_forward_windows": [[t for t in w if t] for w in WALK_FORWARD_WINDOWS],
        "final_train_through": 2025,
        "targets_trained": list(final_models.keys()),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\n  ✅ Metadata saved")

    print(f"\n{'='*65}")
    print(f"  ORB_v2.0 training complete → model/ORB_v2.0/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()

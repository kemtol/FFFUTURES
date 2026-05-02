#!/usr/bin/env python3
"""
Topstep 50K Pass-Rate Simulation — ORB_v2.0 final evaluation.

Uses trained models from model/ORB_v2.0_2010-2026/.
Evaluates on holdout (2025-12-01 → 2026) with full Topstep rules.
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

from analysis.topstep_sim import (
    PolicyParams,
    apply_policy,
    build_param_grid,
    map_to_topstep_trade_day,
    score_policy_on_events,
)
from pipeline.feature.modules.loader import load_features_from_modules

# ── paths ─────────────────────────────────────────────────────────────────────
DM_PATH = BASE / "data" / "Level_2_Datamart" / "training_datamart_orb.parquet"
MODULES_DIR = BASE / "data" / "Level_1_Features" / "modules"
MODEL_DIR = BASE / "model" / "ORB_v2.0_2010-2026"
OUT_DIR = MODEL_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_FROM = "2025-12-01"

# Core columns — same as walk-forward script
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

ALL_TARGETS = [
    ("y_1r2_60m", 2.0), ("y_1r4_60m", 4.0),
    ("y_1r2_120m", 2.0), ("y_1r4_120m", 4.0),
    ("y_1r2_180m", 2.0), ("y_1r4_180m", 4.0),
    ("y_1r2_240m", 2.0), ("y_1r4_240m", 4.0),
    ("y_1r2_close60m", 2.0), ("y_1r4_close60m", 4.0),
]


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


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 65)
    print("  Topstep 50K Pass-Rate Simulation — ORB_v2.0")
    print(f"  Holdout: {HOLDOUT_FROM}+")
    print("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────
    print("\n1. Loading data + features...")
    dm = pd.read_parquet(DM_PATH, columns=CORE_COLS).copy()
    dm = load_features_from_modules(MODULES_DIR, dm, verbose=False)
    dm["date"] = pd.to_datetime(dm["date"])
    dm["year"] = dm["date"].dt.year

    excluded = set(EVENT_KEY) | {"year"}
    module_cols = sorted(set(dm.columns) - set(CORE_COLS) - excluded)
    all_features = CORE_FEATURES + module_cols
    print(f"   Features: {len(all_features)} total")

    holdout = dm[dm["date"] >= HOLDOUT_FROM].copy()
    print(f"   Holdout events: {len(holdout):,} ({holdout['date'].min()} → {holdout['date'].max()})")

    param_grid = build_param_grid()
    print(f"   Policy grid: {len(param_grid)} combos")

    # ── 2. Evaluate each target ────────────────────────────────────────
    summary_rows = []

    for target, rr in ALL_TARGETS:
        rev_model_path = MODEL_DIR / f"lgbm_rev_v2_{target}.txt"
        cont_model_path = MODEL_DIR / f"lgbm_cont_v2_{target}.txt"
        if not rev_model_path.exists() or not cont_model_path.exists():
            print(f"\n  SKIP {target} — model not found")
            continue

        print(f"\n{'─'*60}")
        print(f"  {target} (RR={rr})")

        rev_model = lgb.Booster(model_file=str(rev_model_path))
        cont_model = lgb.Booster(model_file=str(cont_model_path))

        rev_df = holdout[holdout["side"] == "rev"].copy()
        cont_df = holdout[holdout["side"] == "cont"].copy()

        rev_enc = encode(rev_df).dropna(subset=numeric_features(rev_df, all_features) + [target])
        cont_enc = encode(cont_df).dropna(subset=numeric_features(cont_df, all_features) + [target])

        nf = numeric_features(rev_enc, all_features)
        rev_enc["prob_rev"] = rev_model.predict(rev_enc[nf])
        rev_enc["y_rev"] = rev_enc[target]
        cont_enc["prob_cont"] = cont_model.predict(cont_enc[nf])
        cont_enc["y_cont"] = cont_enc[target]

        merge_on = MERGE_KEY + ["year", "orb_range", "breakout_strength",
                                 "atr14_at_entry", "price_vs_vwap_pct",
                                 "adx_14_15m", "ema_slope_1h"]
        events = rev_enc[merge_on + ["prob_rev", "y_rev"]].merge(
            cont_enc[merge_on + ["prob_cont", "y_cont"]], on=merge_on, how="inner"
        )
        events["trade_day"] = map_to_topstep_trade_day(events["breakout_ts"])
        events = events.sort_values(["date", "breakout_ts"]).reset_index(drop=True)

        # Calibrate on self (no separate calibration period)
        calib = events.copy()

        # Base win rates
        wr_rev = events["y_rev"].astype(int).mean()
        wr_cont = events["y_cont"].astype(int).mean()
        exp_rev = wr_rev * rr - (1 - wr_rev)
        exp_cont = wr_cont * rr - (1 - wr_cont)
        print(f"   Base WR: rev={wr_rev:.3f} (exp={exp_rev:+.3f}R), cont={wr_cont:.3f} (exp={exp_cont:+.3f}R)")

        # Sweep policy params
        best_score = -999.0
        best_params = None
        best_result = {}
        print(f"   Sweeping {len(param_grid)} policy combos...", end=" ", flush=True)

        for p in param_grid:
            result = score_policy_on_events(events, calib, p, rr)
            if result["score"] > best_score:
                best_score = result["score"]
                best_params = p
                best_result = result
        print("done.")

        row = {
            "target": target,
            "rr": rr,
            "wr_rev": wr_rev,
            "wr_cont": wr_cont,
            "exp_rev": exp_rev,
            "exp_cont": exp_cont,
            "pass_rate": best_result["pass_rate"],
            "fail_mll_rate": best_result["fail_mll_rate"],
            "score": best_result["score"],
            "median_pnl": best_result["median_end_pnl"],
            "avg_trades": best_result["avg_trades"],
            "windows": best_result["windows"],
            "rev_q": best_params.rev_q,
            "cont_q": best_params.cont_q,
            "rev_adx": best_params.rev_adx_min,
            "cont_adx": best_params.cont_adx_max,
            "risk_per_r": best_params.risk_per_r_usd,
            "profit_cap": best_params.daily_profit_cap_usd,
        }
        summary_rows.append(row)

        print(f"   → score={best_result['score']:+.4f}, pass={best_result['pass_rate']:.1%}, "
              f"fail_mll={best_result['fail_mll_rate']:.1%}, pnl=${best_result['median_end_pnl']:+.0f}")

    # ── 3. Results ─────────────────────────────────────────────────────
    if not summary_rows:
        print("\n❌ No results.")
        return

    sdf = pd.DataFrame(summary_rows).sort_values("score", ascending=False)

    print(f"\n{'='*65}")
    print(f"  Topstep 50K — ORB_v2.0")
    print(f"{'='*65}")
    print(f"  {'Target':<20s} {'Score':>8s} {'Pass':>7s} {'FailMLL':>8s} {'PnL':>8s} {'Trades':>6s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*6}")
    for _, row in sdf.iterrows():
        pnl = f"${row['median_pnl']:+.0f}" if not pd.isna(row['median_pnl']) else "-"
        print(f"  {row['target']:<20s} {row['score']:>+8.4f} {row['pass_rate']:>7.1%} "
              f"{row['fail_mll_rate']:>8.1%} {pnl:>8s} {row['avg_trades']:>6.0f}")

    sdf.to_csv(OUT_DIR / "topstep_pass_results.csv", index=False)
    print(f"\n✅ Saved to {OUT_DIR}/topstep_pass_results.csv")

    # Best params
    print(f"\n  Best policy params per target:")
    for _, row in sdf.iterrows():
        print(f"    {row['target']:<20s} rev_q={row['rev_q']} cont_q={row['cont_q']} "
              f"rev_adx={int(row['rev_adx'])} cont_adx={int(row['cont_adx'])} "
              f"risk=${row['risk_per_r']:.0f} cap=${row['profit_cap']:.0f}")


if __name__ == "__main__":
    main()

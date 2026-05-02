"""
Test: refined Topstep simulator (topstep_sim.py) vs old inline simulator.

Loads y_1r4_close60m model (v3 — all features), scores on holdout,
then runs compare_simulators() to see how trade-day boundaries and
commission changes affect pass_rate, fail_mll_rate, and score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# Ensure pipeline is importable
BASE = Path(__file__).resolve().parent.parent.parent
PIPELINE = BASE / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from analysis.topstep_sim import (
    PolicyParams,
    apply_policy,
    build_param_grid,
    by_year_eval,
    compare_simulators,
    go_no_go,
    md_table,
    score_policy_on_events,
)

DM_PATH = BASE / "data" / "Level_2_Datamart" / "training_datamart_orb.parquet"
OUT_DIR = BASE / "model" / "SWEEP_v3"
OUT_DIR = BASE / "model" / "SWEEP_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── v3 feature sets ───────────────────────────────────────────────────────────

V2_FEATURES = [
    "orb_range", "breakout_strength", "atr14_at_entry",
    "price_vs_vwap_pct", "adx_14_15m", "ema_slope_1h",
    "day_of_week", "time_in_session_min", "orb_tf", "session", "breakout_side",
]

SCALE_INV_FEATURES = [
    "breakout_strength_atr_ratio",
    "atr14_sq",
    "breakout_strength_sq",
    "orb_range_sq",
    "price_vs_vwap_pct_abs",
    "adx_50_flag",
    "breakout_strength_vs_orb",
]

ALL_FEATURES = V2_FEATURES + SCALE_INV_FEATURES

EVENT_KEY = ["date", "breakout_ts", "breakout_side"]

# ── label ─────────────────────────────────────────────────────────────────────

TARGET = "y_1r4_close60m"
RR = 4.0
CALIB_FROM = "2020-01-01"
CALIB_TO = "2023-12-31"
HOLDOUT_FROM = "2024-01-01"


def add_scale_invariant_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-8
    df = df.copy()
    df["breakout_strength_atr_ratio"] = df["breakout_strength"] / (df["atr14_at_entry"] + eps)
    df["atr14_sq"] = df["atr14_at_entry"] ** 2
    df["breakout_strength_sq"] = df["breakout_strength"] ** 2
    df["orb_range_sq"] = df["orb_range"] ** 2
    df["price_vs_vwap_pct_abs"] = df["price_vs_vwap_pct"].abs()
    df["adx_50_flag"] = (df["adx_14_15m"] > 50).astype(int)
    df["breakout_strength_vs_orb"] = df["breakout_strength"] / (df["orb_range"] + eps)
    return df


def _numeric_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Filter features to only numeric columns (exclude object/string)."""
    return [f for f in features if f in df.columns and df[f].dtype in ("float64", "float32", "int64", "int32", "bool")]


def train_model(df: pd.DataFrame, target: str, features: list[str]) -> lgb.Booster:
    """Train LGBM with exponential time decay weighting — calls encode() for string cols."""
    TRAIN_TO = "2023-12-31"
    CALIB_FROM = "2020-01-01"
    CALIB_TO = "2023-12-31"

    train_df = df[df["date"] <= TRAIN_TO].copy()
    val_df = df[(df["date"] >= CALIB_FROM) & (df["date"] <= CALIB_TO)].copy()

    train_df = encode(train_df).reset_index(drop=True)
    val_df = encode(val_df).reset_index(drop=True)

    # Filter to numeric-only features (orb_tf, session are one-hot encoded now)
    num_feats = _numeric_features(train_df, features)
    train_df = train_df.dropna(subset=num_feats + [target]).reset_index(drop=True)
    val_df = val_df.dropna(subset=num_feats + [target]).reset_index(drop=True)

    if len(train_df) < 500:
        raise ValueError(f"Not enough training samples: {len(train_df)}")

    # Sort by date and assign exponential decay weights
    train_df = train_df.sort_values("date").reset_index(drop=True)
    n = len(train_df)
    half_life_days = 365 * 2
    # Each row gets weight based on its position relative to the latest row
    # Row i (0 = oldest) gets weight exp(-ln(2) * (n-1-i) / half_life_days)
    positions = np.arange(n - 1, -1, -1)  # newest = 0, oldest = n-1
    w = np.exp(-np.log(2) * positions / half_life_days) + 0.01

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_child_samples": 30,
        "verbosity": -1,
        "seed": 42,
    }
    dtrain = lgb.Dataset(train_df[num_feats], label=train_df[target].astype(int),
                         weight=w, feature_name=num_feats)
    dval = lgb.Dataset(val_df[num_feats], label=val_df[target].astype(int), reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
    return model


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode orb_tf and session into numeric columns."""
    df = df.copy()
    orb_dummies = pd.get_dummies(df["orb_tf"], prefix="orb_tf")
    sess_dummies = pd.get_dummies(df["session"], prefix="session")
    df = pd.concat([df, orb_dummies.astype(float), sess_dummies.astype(float)], axis=1)
    df["breakout_side"] = (df["breakout_side"] == 1).astype(float)
    return df


def main() -> None:
    print("=" * 65)
    print("  Topstep Simulator Refinement — Comparison Test")
    print(f"  Target: {TARGET}  (RR={RR})")
    print("=" * 65)

    # ── 1. Load datamart ──────────────────────────────────────────────
    print("\n1. Loading datamart...")
    dm = pd.read_parquet(DM_PATH).copy()
    dm["date"] = pd.to_datetime(dm["date"])
    dm["year"] = dm["date"].dt.year
    print(f"   Shape: {dm.shape}")
    print(f"   Date range: {dm['date'].min():%Y-%m-%d} → {dm['date'].max():%Y-%m-%d}")

    # ── 2. Add scale-invariant features ───────────────────────────────
    print("\n2. Adding scale-invariant features...")
    dm = add_scale_invariant_features(dm)
    print(f"   Features: {len(ALL_FEATURES)} ({len(V2_FEATURES)} baseline + {len(SCALE_INV_FEATURES)} scale-inv)")

    # ── 3. Split by side ──────────────────────────────────────────────
    rev_df = dm[dm["side"] == "rev"].copy()
    cont_df = dm[dm["side"] == "cont"].copy()
    print(f"\n3. Data split: rev={len(rev_df)}, cont={len(cont_df)}")

    # ── 4. Train models ───────────────────────────────────────────────
    print(f"\n4. Training models (v3 all features)...")
    print(f"   Training rev model...", end=" ", flush=True)
    rev_model = train_model(rev_df, TARGET, ALL_FEATURES)
    print(f"best_iter={rev_model.best_iteration}")

    print(f"   Training cont model...", end=" ", flush=True)
    cont_model = train_model(cont_df, TARGET, ALL_FEATURES)
    print(f"best_iter={cont_model.best_iteration}")

    # ── 5. Build event frame ──────────────────────────────────────────
    print(f"\n5. Building event frame...")
    rev_enc = encode(rev_df)
    cont_enc = encode(cont_df)
    num_feats = _numeric_features(rev_enc, ALL_FEATURES)
    print(f"   Numeric features for prediction: {len(num_feats)} (from {len(ALL_FEATURES)} requested)")

    rev_s = rev_enc.dropna(subset=num_feats + [TARGET]).copy()
    rev_s["prob_rev"] = rev_model.predict(rev_s[num_feats])
    rev_s["y_rev"] = rev_s[TARGET]

    cont_s = cont_enc.dropna(subset=num_feats + [TARGET]).copy()
    cont_s["prob_cont"] = cont_model.predict(cont_s[num_feats])
    cont_s["y_cont"] = cont_s[TARGET]

    # Merge rev and cont on common event identifiers + market context
    # These market-context columns are the same for rev/cont of the same breakout
    merge_on = EVENT_KEY + ["year",
                            "orb_range", "breakout_strength", "atr14_at_entry",
                            "price_vs_vwap_pct", "adx_14_15m", "ema_slope_1h"]
    events = rev_s[merge_on + ["prob_rev", "y_rev"]].merge(
        cont_s[merge_on + ["prob_cont", "y_cont"]], on=merge_on, how="inner"
    )
    events = events.sort_values(["date", "breakout_ts"]).reset_index(drop=True)

    # Add Topstep trade_day column
    from analysis.topstep_sim import map_to_topstep_trade_day
    events["trade_day"] = map_to_topstep_trade_day(events["breakout_ts"])
    events["year"] = events["date"].dt.year  # ensure year exists

    print(f"   Total events: {len(events)}")
    print(f"   Unique trade days: {events['trade_day'].nunique()}")

    calib = events[(events["date"] >= CALIB_FROM) & (events["date"] <= CALIB_TO)].copy()
    holdout = events[events["date"] >= HOLDOUT_FROM].copy()
    print(f"   Calibration ({CALIB_FROM} to {CALIB_TO}): {len(calib)} events")
    print(f"   Holdout ({HOLDOUT_FROM}+): {len(holdout)} events")

    # Baseline win rates
    wr_rev = holdout["y_rev"].astype(int).mean()
    wr_cont = holdout["y_cont"].astype(int).mean()
    cost_r = 0.07
    exp_rev = wr_rev * RR - (1 - wr_rev) - cost_r
    exp_cont = wr_cont * RR - (1 - wr_cont) - cost_r
    print(f"\n   Baseline holdout:")
    print(f"     Rev:  wr={wr_rev:.3f}  exp={exp_rev:+.3f}R")
    print(f"     Cont: wr={wr_cont:.3f}  exp={exp_cont:+.3f}R")

    # ── 6. Sweep best params ──────────────────────────────────────────
    print(f"\n6. Sweeping param grid ({len(build_param_grid())} combos)...")
    best_score = -999.0
    best_params: PolicyParams | None = None
    for p in build_param_grid():
        result = score_policy_on_events(holdout, calib, p, RR)
        if result["score"] > best_score:
            best_score = result["score"]
            best_params = p
    assert best_params is not None
    print(f"   Best params: rev_q={best_params.rev_q}, cont_q={best_params.cont_q}, "
          f"rev_adx_min={best_params.rev_adx_min}, cont_adx_max={best_params.cont_adx_max}, "
          f"risk={best_params.risk_per_r_usd}, cap={best_params.daily_profit_cap_usd}")
    print(f"   Best score: {best_score:+.4f}")

    # ── 7. Compare old vs new simulator ───────────────────────────────
    print(f"\n{'='*65}")
    print(f"  7. Simulator Comparison: Old vs New")
    print(f"{'='*65}")

    comp = compare_simulators(holdout, calib, best_params, RR)

    comp_rows = [
        {"metric": "Pass Rate", "new": f"{comp['new_pass_rate']:.1%}", "old": f"{comp['old_pass_rate']:.1%}", "delta": f"{comp['delta_pass_rate']:+.1%}"},
        {"metric": "Fail MLL",   "new": f"{comp['new_fail_mll']:.1%}",  "old": f"{comp['old_fail_mll']:.1%}",  "delta": ""},
        {"metric": "Score",      "new": f"{comp['new_score']:+.4f}",    "old": f"{comp['old_score']:+.4f}",    "delta": f"{comp['delta_score']:+.4f}"},
        {"metric": "Median PnL", "new": f"${comp['new_median_pnl']:.0f}", "old": "", "delta": ""},
        {"metric": "Windows",    "new": str(comp['new_windows']),      "old": "", "delta": ""},
    ]
    comp_df = pd.DataFrame(comp_rows)
    print(f"\n{md_table(comp_df, ['metric', 'new', 'old', 'delta'])}")

    # ── 8. Per-year breakdown (new simulator) ─────────────────────────
    print(f"\n{'='*65}")
    print(f"  8. Yearly Breakdown — New Simulator")
    print(f"{'='*65}")

    yr_new = by_year_eval(holdout, calib, best_params, RR)
    yr_rows = []
    for yr in sorted(yr_new.keys()):
        d = yr_new[yr]
        yr_rows.append({
            "year": yr,
            "pass_rate": f"{d['pass_rate']:.1%}",
            "fail_mll": f"{d['fail_mll_rate']:.1%}",
            "median_pnl": f"${d['median_end_pnl']:.0f}",
        })
    yr_df = pd.DataFrame(yr_rows)
    print(f"\n{md_table(yr_df, ['year', 'pass_rate', 'fail_mll', 'median_pnl'])}")

    # ── 9. Per-year breakdown (old simulator) ─────────────────────────
    print(f"\n{'='*65}")
    print(f"  9. Yearly Breakdown — Old Simulator")
    print(f"{'='*65}")

    # Old simulator yearly: use calendar date trade_day + 0.07R commission
    old_events = holdout.copy()
    old_events["trade_day"] = pd.to_datetime(old_events["date"]).dt.date
    old_calib = calib.copy()
    old_calib["trade_day"] = pd.to_datetime(old_calib["date"]).dt.date

    # Need a separate by_year_eval that uses old-style commission
    old_cost_r = 0.07

    def old_by_year(holdout_df, calib_df, p):
        """by_year_eval with old-style commission."""
        out = holdout_df.copy()
        rev_t = float(np.percentile(calib_df["prob_rev"], p.rev_q * 100))
        cont_t = float(np.percentile(calib_df["prob_cont"], p.cont_q * 100))
        rev_strong = float(np.percentile(calib_df["prob_rev"], 75))
        cont_strong = float(np.percentile(calib_df["prob_cont"], 75))
        trend_sign = np.sign(out["ema_slope_1h"].fillna(0))
        aligned_rev = trend_sign == -out["breakout_side"]
        aligned_cont = trend_sign == out["breakout_side"]
        rev_gate = ((out["prob_rev"] >= rev_t) & aligned_rev &
                    ((out["adx_14_15m"] >= p.rev_adx_min) | (out["prob_rev"] >= rev_strong)))
        cont_gate = ((out["prob_cont"] >= cont_t) & aligned_cont &
                     ((out["adx_14_15m"] <= p.cont_adx_max) | (out["prob_cont"] >= cont_strong)))
        decision = np.full(len(out), "skip", dtype=object)
        decision[rev_gate & ~cont_gate] = "rev"
        decision[cont_gate & ~rev_gate] = "cont"
        both = np.where((rev_gate & cont_gate).values)[0]
        if len(both):
            pick_rev = out.iloc[both]["prob_rev"].values >= out.iloc[both]["prob_cont"].values
            decision[both[pick_rev]] = "rev"
            decision[both[~pick_rev]] = "cont"
        out["decision"] = decision
        out["y"] = np.where(out["decision"] == "rev", out["y_rev"],
                            np.where(out["decision"] == "cont", out["y_cont"], np.nan))
        out["pnl_usd"] = np.where(out["y"] == 1, RR, -1.0) * p.risk_per_r_usd
        out["pnl_usd"] = np.where(out["decision"] == "skip", 0.0,
                                  out["pnl_usd"] - old_cost_r * p.risk_per_r_usd)

        year_results = {}
        for yr, g in out.groupby("year"):
            days = sorted(g["trade_day"].unique())
            if len(days) < 20:
                continue
            windows = [days[i: i + 20] for i in range(len(days) - 20 + 1)]
            pbd = {}
            for row in g.itertuples(index=False):
                is_trade = row.decision != "skip"
                pbd.setdefault(row.trade_day, []).append((float(row.pnl_usd), is_trade))
            from analysis.topstep_sim import simulate_window
            sims = [simulate_window(w, pbd, 3000.0, 2000.0,
                                    daily_stop_usd=p.daily_stop_usd,
                                    daily_profit_cap_usd=p.daily_profit_cap_usd)
                    for w in windows]
            sdf = pd.DataFrame(sims)
            year_results[str(yr)] = {
                "pass_rate": float(sdf["passed"].mean()),
                "fail_mll_rate": float(sdf["failed_mll"].mean()),
                "median_end_pnl": float(sdf["end_pnl"].median()),
            }
        return year_results

    yr_old = old_by_year(old_events, old_calib, best_params)
    yr_old_rows = []
    for yr in sorted(yr_old.keys()):
        d = yr_old[yr]
        yr_old_rows.append({
            "year": yr,
            "pass_rate": f"{d['pass_rate']:.1%}",
            "fail_mll": f"{d['fail_mll_rate']:.1%}",
            "median_pnl": f"${d['median_end_pnl']:.0f}",
        })
    yr_old_df = pd.DataFrame(yr_old_rows)
    print(f"\n{md_table(yr_old_df, ['year', 'pass_rate', 'fail_mll', 'median_pnl'])}")

    # ── 10. Year-over-year comparison ──────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  10. Pass Rate Delta by Year (New − Old)")
    print(f"{'='*65}")
    delta_rows = []
    all_years = sorted(set(list(yr_new.keys()) + list(yr_old.keys())))
    for yr in all_years:
        n = yr_new.get(yr, {})
        o = yr_old.get(yr, {})
        delta_rows.append({
            "year": yr,
            "new_pass": f"{n.get('pass_rate', 0):.1%}",
            "old_pass": f"{o.get('pass_rate', 0):.1%}",
            "delta_pass": f"{n.get('pass_rate', 0) - o.get('pass_rate', 0):+.1%}",
            "new_pnl": f"${n.get('median_end_pnl', 0):.0f}",
            "old_pnl": f"${o.get('median_end_pnl', 0):.0f}",
        })
    delta_df = pd.DataFrame(delta_rows)
    print(f"\n{md_table(delta_df, ['year', 'new_pass', 'old_pass', 'delta_pass', 'new_pnl', 'old_pnl'])}")

    # ── 11. GO/NO-GO ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  11. GO/NO-GO Decision")
    print(f"{'='*65}")
    overall_go, yearly_go, final_go = go_no_go(comp["new_pass_rate"], comp["new_fail_mll"], yr_new)
    print(f"   Overall gate  : {'✅ GO' if overall_go else '❌ NO-GO'}  "
          f"(pass={comp['new_pass_rate']:.1%} ≥ 60%, fail_mll={comp['new_fail_mll']:.1%} ≤ 10%)")
    print(f"   Yearly gate   : {'✅ GO' if yearly_go else '❌ NO-GO'}  "
          f"(all years ≥ 30% pass, ≤ 20% fail_mll)")
    print(f"   Final decision: {'✅ GO' if final_go else '❌ NO-GO'}")
    print()

    # ── 12. Summary ───────────────────────────────────────────────────
    print(f"{'='*65}")
    print(f"  12. Summary")
    print(f"{'='*65}")
    print(f"   Target:            {TARGET} (RR={RR})")
    print(f"   Params:            rev_q={best_params.rev_q}, cont_q={best_params.cont_q}, "
          f"risk=${best_params.risk_per_r_usd:.0f}")
    print(f"")
    print(f"   ┌─────────────────────┬──────────┬──────────┬──────────┐")
    print(f"   │ Metric              │ OLD      │ NEW      │ Δ        │")
    print(f"   ├─────────────────────┼──────────┼──────────┼──────────┤")
    print(f"   │ Pass Rate           │ {comp['old_pass_rate']:>6.1%}  │ {comp['new_pass_rate']:>6.1%}  │ {comp['delta_pass_rate']:>+6.1%}  │")
    print(f"   │ Fail MLL            │ {comp['old_fail_mll']:>6.1%}  │ {comp['new_fail_mll']:>6.1%}  │          │")
    print(f"   │ Score               │ {comp['old_score']:>+6.4f}  │ {comp['new_score']:>+6.4f}  │ {comp['delta_score']:>+6.4f}  │")
    print(f"   │ Median PnL          │          │ ${comp['new_median_pnl']:>5.0f}   │          │")
    print(f"   │ Windows             │          │ {comp['new_windows']:>5d}   │          │")
    print(f"   └─────────────────────┴──────────┴──────────┴──────────┘")
    print(f"")
    print(f"   Key changes:")
    print(f"   1. Trading day: CT 5PM→3:10PM (not calendar date)")
    print(f"   2. Commission:  $3.00 fixed (not 0.07R = ${best_params.risk_per_r_usd * 0.07:.0f})")
    print(f"   3. MLL trailing & consistency rule: identical")


if __name__ == "__main__":
    main()

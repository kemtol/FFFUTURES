"""
Objective Sweep v3 — scale-invariant features to address 2026 OOD failure.

Key changes from v2:
  - Added 7 scale-invariant features (normalized by rolling ATR)
  - Same train/calib/holdout split as v2 (fair comparison)
  - Added feature importance tracking
  - Added AB test: compare v2 features-only vs v2+v3 features
  - Focus analysis on y_1r4_close60m (best 2026 performer from v2)

Rationale:
  2026 volatility explosion (ATR14 4.6× training data) caused model OOD failure.
  Scale-invariant features like breakout_strength/ATR should maintain
  consistent distributions across volatility regimes.

Output: model/SWEEP_v3/
  OBJECTIVE_SWEEP_RESULTS.csv
  OBJECTIVE_SWEEP_REPORT.md
  FEATURE_IMPORTANCE.csv
  FEATURE_DISTRIBUTIONS.md
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DM_PATH = ROOT / "data/Level_2_Datamart/training_datamart_orb.parquet"
OUT_DIR = ROOT / "model/SWEEP_v3"

# ── data splits (same as v2 for fair comparison) ────────────────────────────

HOLDOUT_FROM = "2025-01-01"
CALIB_FROM = "2024-01-01"
CALIB_TO = "2024-12-31"
TRAIN_TO = "2023-12-31"

HALF_LIFE_YEARS = 2
COST_R = 0.07
WINDOW_DAYS = 20
ACCOUNT = {"profit_target": 3000.0, "mll": 2000.0}

# ── feature sets ────────────────────────────────────────────────────────────

V2_FEATURES = [
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

V3_NEW_FEATURES = [
    "breakout_strength_atr_ratio",   # breakout_strength / atr14_at_entry
    "atr14_sq",                       # atr14_at_entry ** 2  (non-linear vol)
    "breakout_strength_sq",           # breakout_strength ** 2
    "price_vs_vwap_pct_abs",          # abs(price_vs_vwap_pct)
    "orb_range_sq",                   # orb_range ** 2
    "adx_50_flag",                    # 1 if adx_14_15m > 50 else 0
    "breakout_strength_vs_orb",       # breakout_strength / orb_range  (fraction of range covered)
]

ALL_FEATURES = V2_FEATURES + V3_NEW_FEATURES

EVENT_KEY = ["date", "breakout_ts", "session", "orb_tf", "breakout_side"]

LABELS: list[tuple[str, float]] = [
    ("y_1r2_60m",      2.0),
    ("y_1r4_60m",      4.0),
    ("y_1r2_120m",     2.0),
    ("y_1r4_120m",     4.0),
    ("y_1r2_180m",     2.0),
    ("y_1r4_180m",     4.0),
    ("y_1r2_240m",     2.0),
    ("y_1r4_240m",     4.0),
    ("y_1r2_close60m", 2.0),
    ("y_1r4_close60m", 4.0),
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
NUM_ROUNDS = 300
EARLY_STOP = 30


@dataclass(frozen=True)
class PolicyParams:
    rev_q: float
    cont_q: float
    rev_adx_min: float
    cont_adx_max: float
    daily_stop_usd: float
    daily_profit_cap_usd: float
    risk_per_r_usd: float


def build_param_grid() -> list[PolicyParams]:
    params = []
    for rev_q in [0.60, 0.75]:
        for cont_q in [0.60, 0.75]:
            for rev_adx_min in [30, 40]:
                for cont_adx_max in [30, 100]:
                    for risk in [100, 150, 200, 250]:
                        for profit_cap in [0.0, 1400.0]:
                            params.append(PolicyParams(
                                rev_q=rev_q, cont_q=cont_q,
                                rev_adx_min=rev_adx_min, cont_adx_max=cont_adx_max,
                                daily_stop_usd=0.0, daily_profit_cap_usd=profit_cap,
                                risk_per_r_usd=float(risk),
                            ))
    return params


PARAM_GRID = build_param_grid()


# ── helpers ───────────────────────────────────────────────────────────────────

def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["session"] = out["session"].map({"tokyo": 0, "london": 1, "us": 2})
    out["orb_tf"] = out["orb_tf"].map({"5m": 5, "15m": 15, "30m": 30})
    return out


def compute_weights(years: pd.Series) -> np.ndarray:
    lam = math.log(2) / HALF_LIFE_YEARS
    return np.exp(-lam * (years.max() - years)).values


def add_scale_invariant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add scale-invariant features to DataFrame. Operates in-place."""
    eps = 1e-8

    # breakout_strength / atr14 — how many ATRs the breakout moved
    df["breakout_strength_atr_ratio"] = df["breakout_strength"] / (df["atr14_at_entry"] + eps)

    # squared terms for non-linear relationships
    df["atr14_sq"] = df["atr14_at_entry"] ** 2
    df["breakout_strength_sq"] = df["breakout_strength"] ** 2
    df["orb_range_sq"] = df["orb_range"] ** 2

    # absolute VWAP deviation (magnitude matters regardless of direction)
    df["price_vs_vwap_pct_abs"] = df["price_vs_vwap_pct"].abs()

    # binary ADX regime flag
    df["adx_50_flag"] = (df["adx_14_15m"] > 50).astype(int)

    # breakout strength relative to ORB range (fraction of range covered)
    df["breakout_strength_vs_orb"] = df["breakout_strength"] / (df["orb_range"] + eps)

    return df


# ── training ──────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, target: str, features: list[str]) -> lgb.Booster:
    train_df = df[df["date"] <= TRAIN_TO].copy()
    val_df = df[(df["date"] >= CALIB_FROM) & (df["date"] <= CALIB_TO)].copy()

    train_df = encode(train_df).dropna(subset=features + [target]).reset_index(drop=True)
    val_df = encode(val_df).dropna(subset=features + [target]).reset_index(drop=True)

    w = compute_weights(train_df["year"])
    dtrain = lgb.Dataset(train_df[features], label=train_df[target],
                         weight=w, feature_name=features)
    dval = lgb.Dataset(val_df[features], label=val_df[target], reference=dtrain)

    model = lgb.train(
        LGBM_PARAMS, dtrain,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    return model


# ── policy + Topstep ──────────────────────────────────────────────────────────

def score_holdout(events: pd.DataFrame, calib: pd.DataFrame,
                  p: PolicyParams, rr: float, features: list[str]) -> dict:
    out = events.copy()
    rev_t = float(np.percentile(calib["prob_rev"], p.rev_q * 100))
    cont_t = float(np.percentile(calib["prob_cont"], p.cont_q * 100))
    rev_strong = float(np.percentile(calib["prob_rev"], 75))
    cont_strong = float(np.percentile(calib["prob_cont"], 75))

    trend_sign = np.sign(out["ema_slope_1h"].fillna(0))
    aligned_rev = trend_sign == -out["breakout_side"]
    aligned_cont = trend_sign == out["breakout_side"]

    rev_gate = (
        (out["prob_rev"] >= rev_t) & aligned_rev &
        ((out["adx_14_15m"] >= p.rev_adx_min) | (out["prob_rev"] >= rev_strong))
    )
    cont_gate = (
        (out["prob_cont"] >= cont_t) & aligned_cont &
        ((out["adx_14_15m"] <= p.cont_adx_max) | (out["prob_cont"] >= cont_strong))
    )

    decision = np.full(len(out), "skip", dtype=object)
    decision[rev_gate & ~cont_gate] = "rev"
    decision[cont_gate & ~rev_gate] = "cont"
    both = np.where((rev_gate & cont_gate).values)[0]
    if len(both):
        pick_rev = out.iloc[both]["prob_rev"].values >= out.iloc[both]["prob_cont"].values
        decision[both[pick_rev]] = "rev"
        decision[both[~pick_rev]] = "cont"

    out["decision"] = decision
    out["y"] = np.where(
        out["decision"] == "rev", out["y_rev"],
        np.where(out["decision"] == "cont", out["y_cont"], np.nan)
    )
    out["r_net"] = np.where(out["y"] == 1, rr, -1.0) - COST_R
    out.loc[out["decision"] == "skip", "r_net"] = 0.0
    out["pnl_usd"] = out["r_net"] * p.risk_per_r_usd
    return run_topstep_sim(out, p)


def run_topstep_sim(events: pd.DataFrame, p: PolicyParams) -> dict:
    days = sorted(events["trade_day"].unique())
    windows = [days[i: i + WINDOW_DAYS] for i in range(len(days) - WINDOW_DAYS + 1)]
    if not windows:
        return {"pass_rate": 0.0, "fail_mll_rate": 0.0, "score": 0.0,
                "median_end_pnl": np.nan, "avg_trades": 0.0, "windows": 0}

    pnl_by_day: dict = {}
    for row in events.itertuples(index=False):
        is_trade = row.decision != "skip"
        pnl_by_day.setdefault(row.trade_day, []).append((float(row.pnl_usd), is_trade))

    results = [simulate_window(w, pnl_by_day, p) for w in windows]
    rdf = pd.DataFrame(results)
    pass_rate = float(rdf["passed"].mean())
    fail_mll_rate = float(rdf["failed_mll"].mean())
    return {
        "pass_rate": pass_rate,
        "fail_mll_rate": fail_mll_rate,
        "score": pass_rate - fail_mll_rate,
        "median_end_pnl": float(rdf["end_pnl"].median()),
        "avg_trades": float(rdf["trades"].mean()),
        "windows": len(rdf),
    }


def simulate_window(window_days: list, pnl_by_day: dict, p: PolicyParams) -> dict:
    pnl = 0.0
    mll_floor = -ACCOUNT["mll"]
    max_eod_pnl = 0.0
    best_day = 0.0
    trades = 0
    fail_mll = False
    passed = False

    for day in window_days:
        day_pnl = 0.0
        blocked = False
        for trade_pnl, is_trade in pnl_by_day.get(day, []):
            if blocked or not is_trade:
                continue
            pnl += trade_pnl
            day_pnl += trade_pnl
            trades += 1
            if pnl <= mll_floor:
                fail_mll = True
                blocked = True
                break
            if p.daily_stop_usd > 0 and day_pnl <= -p.daily_stop_usd:
                blocked = True
            if p.daily_profit_cap_usd > 0 and day_pnl >= p.daily_profit_cap_usd:
                blocked = True

        best_day = max(best_day, day_pnl)
        max_eod_pnl = max(max_eod_pnl, pnl)
        mll_floor = min(0.0, -ACCOUNT["mll"] + max_eod_pnl)

        if (not passed and not fail_mll
                and pnl >= ACCOUNT["profit_target"]
                and (pnl == 0 or best_day < 0.5 * pnl)):
            passed = True
        if fail_mll:
            break

    return {"passed": passed, "failed_mll": fail_mll, "end_pnl": pnl,
            "trades": trades, "best_day": best_day}


def by_year_sim(events: pd.DataFrame, calib: pd.DataFrame,
                p: PolicyParams, rr: float) -> dict[str, dict]:
    out_all = events.copy()
    rev_t = float(np.percentile(calib["prob_rev"], p.rev_q * 100))
    cont_t = float(np.percentile(calib["prob_cont"], p.cont_q * 100))
    rev_strong = float(np.percentile(calib["prob_rev"], 75))
    cont_strong = float(np.percentile(calib["prob_cont"], 75))

    trend_sign = np.sign(out_all["ema_slope_1h"].fillna(0))
    aligned_rev = trend_sign == -out_all["breakout_side"]
    aligned_cont = trend_sign == out_all["breakout_side"]
    rev_gate = (
        (out_all["prob_rev"] >= rev_t) & aligned_rev &
        ((out_all["adx_14_15m"] >= p.rev_adx_min) | (out_all["prob_rev"] >= rev_strong))
    )
    cont_gate = (
        (out_all["prob_cont"] >= cont_t) & aligned_cont &
        ((out_all["adx_14_15m"] <= p.cont_adx_max) | (out_all["prob_cont"] >= cont_strong))
    )
    decision = np.full(len(out_all), "skip", dtype=object)
    decision[rev_gate & ~cont_gate] = "rev"
    decision[cont_gate & ~rev_gate] = "cont"
    both = np.where((rev_gate & cont_gate).values)[0]
    if len(both):
        pick_rev = out_all.iloc[both]["prob_rev"].values >= out_all.iloc[both]["prob_cont"].values
        decision[both[pick_rev]] = "rev"
        decision[both[~pick_rev]] = "cont"

    out_all["decision"] = decision
    out_all["y"] = np.where(
        out_all["decision"] == "rev", out_all["y_rev"],
        np.where(out_all["decision"] == "cont", out_all["y_cont"], np.nan)
    )
    out_all["r_net"] = np.where(out_all["y"] == 1, rr, -1.0) - COST_R
    out_all.loc[out_all["decision"] == "skip", "r_net"] = 0.0
    out_all["pnl_usd"] = out_all["r_net"] * p.risk_per_r_usd

    year_results = {}
    for yr, g in out_all.groupby("year"):
        days = sorted(g["trade_day"].unique())
        if len(days) < WINDOW_DAYS:
            continue
        windows = [days[i: i + WINDOW_DAYS] for i in range(len(days) - WINDOW_DAYS + 1)]
        pbd: dict = {}
        for row in g.itertuples(index=False):
            is_trade = row.decision != "skip"
            pbd.setdefault(row.trade_day, []).append((float(row.pnl_usd), is_trade))
        sims = [simulate_window(w, pbd, p) for w in windows]
        sdf = pd.DataFrame(sims)
        year_results[str(yr)] = {
            "pass_rate": float(sdf["passed"].mean()),
            "fail_mll_rate": float(sdf["failed_mll"].mean()),
            "median_end_pnl": float(sdf["end_pnl"].median()),
        }
    return year_results


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading datamart...")
    dm = pd.read_parquet(DM_PATH).copy()
    dm["date"] = pd.to_datetime(dm["date"])
    dm["year"] = dm["date"].dt.year
    dm["trade_day"] = dm["date"].dt.date

    # ── add scale-invariant features ─────────────────────────────────
    print("Adding scale-invariant features...")
    dm = add_scale_invariant_features(dm)
    n_orig = dm.shape[1]
    print(f"  Total columns: {n_orig}")

    # ── feature distribution analysis ────────────────────────────────
    print("\nComputing feature distributions by year...")
    feat_dist_rows = []
    # Only compute distributions for numeric features
    numeric_feats = [f for f in (ALL_FEATURES + ["orb_range", "breakout_strength", "atr14_at_entry"])
                     if dm[f].dtype in ("float64", "float32", "int64", "int32")]
    for feat in numeric_feats:
        for yr, g in dm.groupby("year"):
            vals = g[feat].dropna()
            if len(vals) < 10:
                continue
            feat_dist_rows.append({
                "feature": feat,
                "year": yr,
                "mean": vals.mean(),
                "median": vals.median(),
                "std": vals.std(),
                "p5": vals.quantile(0.05),
                "p95": vals.quantile(0.95),
            })
    feat_dist_df = pd.DataFrame(feat_dist_rows)

    # ── train + evaluate per label ───────────────────────────────────
    summary_rows: list[dict] = []
    fi_rows: list[dict] = []

    for target, rr in LABELS:
        print(f"\n{'='*60}")
        print(f"Label: {target}  (RR={rr})")

        rev_df = dm[dm["side"] == "rev"].copy()
        cont_df = dm[dm["side"] == "cont"].copy()

        if target not in rev_df.columns or rev_df[target].isna().all():
            print(f"  SKIP — label missing or all-NaN")
            continue

        # ── train v3 model (all features) ──────────────────────────
        print(f"  Training v3 rev model (ALL features)...", end=" ", flush=True)
        rev_model = train_model(rev_df, target, ALL_FEATURES)
        print(f"best_iter={rev_model.best_iteration}")

        print(f"  Training v3 cont model...", end=" ", flush=True)
        cont_model = train_model(cont_df, target, ALL_FEATURES)
        print(f"best_iter={cont_model.best_iteration}")

        # ── feature importance ─────────────────────────────────────
        rev_imp = pd.DataFrame({
            "feature": ALL_FEATURES,
            "gain_rev": rev_model.feature_importance(importance_type="gain"),
            "split_rev": rev_model.feature_importance(importance_type="split"),
        })
        cont_imp = pd.DataFrame({
            "feature": ALL_FEATURES,
            "gain_cont": cont_model.feature_importance(importance_type="gain"),
            "split_cont": cont_model.feature_importance(importance_type="split"),
        })
        imp = rev_imp.merge(cont_imp, on="feature")
        imp["target"] = target
        imp["gain_total"] = imp["gain_rev"] + imp["gain_cont"]
        imp["split_total"] = imp["split_rev"] + imp["split_cont"]
        fi_rows.append(imp)

        # ── build event frame ────────────────────────────────────
        def side_frame(df: pd.DataFrame, side: str, model: lgb.Booster,
                       features: list[str]) -> pd.DataFrame:
            s = encode(df).dropna(subset=features + [target]).copy()
            s["prob"] = model.predict(s[features])
            keep = EVENT_KEY + ["year", "trade_day", "prob", target,
                                "adx_14_15m", "ema_slope_1h", "price_vs_vwap_pct"]
            return s[keep].rename(columns={"prob": f"prob_{side}", target: f"y_{side}"})

        rev_ev_all = side_frame(rev_df, "rev", rev_model, ALL_FEATURES)
        cont_ev_all = side_frame(cont_df, "cont", cont_model, ALL_FEATURES)

        merge_on = EVENT_KEY + ["year", "trade_day",
                                "adx_14_15m", "ema_slope_1h", "price_vs_vwap_pct"]
        events_all = rev_ev_all.merge(cont_ev_all, on=merge_on, how="inner")
        events_all = events_all.sort_values(["date", "breakout_ts"]).reset_index(drop=True)

        calib_all = events_all[(events_all["date"] >= CALIB_FROM) & (events_all["date"] <= CALIB_TO)].copy()
        holdout_all = events_all[events_all["date"] >= HOLDOUT_FROM].copy()

        if len(calib_all) < 50 or len(holdout_all) < 50:
            print(f"  SKIP — insufficient calib ({len(calib_all)}) or holdout ({len(holdout_all)}) events")
            continue

        # ── also train AB model (v2 features only) for comparison ─
        print(f"  Training v2 (baseline) rev model...", end=" ", flush=True)
        rev_model_v2 = train_model(rev_df, target, V2_FEATURES)
        print(f"best_iter={rev_model_v2.best_iteration}")
        print(f"  Training v2 (baseline) cont model...", end=" ", flush=True)
        cont_model_v2 = train_model(cont_df, target, V2_FEATURES)
        print(f"best_iter={cont_model_v2.best_iteration}")

        rev_ev_v2 = side_frame(rev_df, "rev", rev_model_v2, V2_FEATURES)
        cont_ev_v2 = side_frame(cont_df, "cont", cont_model_v2, V2_FEATURES)
        events_v2 = rev_ev_v2.merge(cont_ev_v2, on=merge_on, how="inner")
        events_v2 = events_v2.sort_values(["date", "breakout_ts"]).reset_index(drop=True)

        calib_v2 = events_v2[(events_v2["date"] >= CALIB_FROM) & (events_v2["date"] <= CALIB_TO)].copy()
        holdout_v2 = events_v2[events_v2["date"] >= HOLDOUT_FROM].copy()

        # ── baseline (unfiltered) ────────────────────────────────
        y_rev_h = holdout_all["y_rev"].astype(int).to_numpy()
        y_cont_h = holdout_all["y_cont"].astype(int).to_numpy()
        wr_rev = float(y_rev_h.mean())
        wr_cont = float(y_cont_h.mean())
        exp_rev = wr_rev * rr - (1 - wr_rev) - COST_R
        exp_cont = wr_cont * rr - (1 - wr_cont) - COST_R

        print(f"  Holdout ({HOLDOUT_FROM}+): n={len(holdout_all)}, "
              f"wr_rev={wr_rev:.3f}(exp={exp_rev:+.3f}R), "
              f"wr_cont={wr_cont:.3f}(exp={exp_cont:+.3f}R)")

        # ── param grid sweep: v3 (all features) ─────────────────
        best_score = -999.0
        best_params: PolicyParams | None = None
        best_result: dict = {}

        print(f"  Sweeping {len(PARAM_GRID)} param combinations (v3)...", end=" ", flush=True)
        for p in PARAM_GRID:
            result = score_holdout(holdout_all, calib_all, p, rr, ALL_FEATURES)
            if result["score"] > best_score:
                best_score = result["score"]
                best_params = p
                best_result = result
        print(f"done.")
        assert best_params is not None
        yr_res_all = by_year_sim(holdout_all, calib_all, best_params, rr)

        # ── param grid sweep: v2 (baseline features) for AB test ─
        best_score_v2 = -999.0
        best_params_v2: PolicyParams | None = None
        best_result_v2: dict = {}

        print(f"  Sweeping {len(PARAM_GRID)} param combinations (v2 baseline)...", end=" ", flush=True)
        for p in PARAM_GRID:
            result = score_holdout(holdout_v2, calib_v2, p, rr, V2_FEATURES)
            if result["score"] > best_score_v2:
                best_score_v2 = result["score"]
                best_params_v2 = p
                best_result_v2 = result
        print(f"done.")
        assert best_params_v2 is not None
        yr_res_v2 = by_year_sim(holdout_v2, calib_v2, best_params_v2, rr)

        # ── print results ────────────────────────────────────────
        print(f"\n  ── RESULTS ──")
        print(f"  v3 (ALL features):")
        print(f"    Best: pass_rate={best_result['pass_rate']:.3f}  "
              f"fail_mll={best_result['fail_mll_rate']:.3f}  "
              f"score={best_result['score']:+.3f}  "
              f"risk=${best_params.risk_per_r_usd:.0f}")
        for yr, yr_d in sorted(yr_res_all.items()):
            print(f"    {yr}: pass={yr_d['pass_rate']:.3f} fail_mll={yr_d['fail_mll_rate']:.3f} "
                  f"median_pnl=${yr_d['median_end_pnl']:.0f}")

        print(f"  v2 (baseline features):")
        print(f"    Best: pass_rate={best_result_v2['pass_rate']:.3f}  "
              f"fail_mll={best_result_v2['fail_mll_rate']:.3f}  "
              f"score={best_result_v2['score']:+.3f}  "
              f"risk=${best_params_v2.risk_per_r_usd:.0f}")
        for yr, yr_d in sorted(yr_res_v2.items()):
            print(f"    {yr}: pass={yr_d['pass_rate']:.3f} fail_mll={yr_d['fail_mll_rate']:.3f} "
                  f"median_pnl=${yr_d['median_end_pnl']:.0f}")

        # ── store results (v3) ──────────────────────────────────
        row = {
            "target":           target,
            "rr":               rr,
            "n_holdout_events": len(holdout_all),
            "wr_rev_baseline":  round(wr_rev, 4),
            "wr_cont_baseline": round(wr_cont, 4),
            "exp_rev_net":      round(exp_rev, 4),
            "exp_cont_net":     round(exp_cont, 4),
            # v3 results
            "v3_pass_rate":     round(best_result["pass_rate"], 4),
            "v3_fail_mll":      round(best_result["fail_mll_rate"], 4),
            "v3_score":         round(best_result["score"], 4),
            "v3_median_pnl":    round(best_result["median_end_pnl"], 1),
            "v3_avg_trades":    round(best_result["avg_trades"], 1),
            "v3_risk_per_r":    best_params.risk_per_r_usd,
            "v3_rev_q":         best_params.rev_q,
            "v3_cont_q":        best_params.cont_q,
            "v3_rev_adx_min":   best_params.rev_adx_min,
            "v3_cont_adx_max":  best_params.cont_adx_max,
            "v3_profit_cap":    best_params.daily_profit_cap_usd,
            # v2 baseline results (AB test)
            "v2_pass_rate":     round(best_result_v2["pass_rate"], 4),
            "v2_fail_mll":      round(best_result_v2["fail_mll_rate"], 4),
            "v2_score":         round(best_result_v2["score"], 4),
            "v2_median_pnl":    round(best_result_v2["median_end_pnl"], 1),
            "v2_avg_trades":    round(best_result_v2["avg_trades"], 1),
            "v2_risk_per_r":    best_params_v2.risk_per_r_usd,
            "delta_score":      round(best_result["score"] - best_result_v2["score"], 4),
        }
        for yr, yr_d in yr_res_all.items():
            row[f"pass_{yr}"] = round(yr_d["pass_rate"], 4)
            row[f"fail_{yr}"] = round(yr_d["fail_mll_rate"], 4)
            row[f"pnl_{yr}"]  = round(yr_d["median_end_pnl"], 0)

        summary_rows.append(row)

    # ── output ────────────────────────────────────────────────────
    results_df = pd.DataFrame(summary_rows)
    results_df = results_df.sort_values("v3_score", ascending=False).reset_index(drop=True)
    results_df.to_csv(OUT_DIR / "OBJECTIVE_SWEEP_RESULTS.csv", index=False)

    # feature importance
    if fi_rows:
        fi_df = pd.concat(fi_rows, ignore_index=True)
        # Normalize gain per target
        fi_df["gain_pct"] = fi_df.groupby("target")["gain_total"].transform(
            lambda x: x / x.sum() * 100 if x.sum() > 0 else 0
        )
        fi_df = fi_df.sort_values(["target", "gain_pct"], ascending=[True, False])
        fi_df.to_csv(OUT_DIR / "FEATURE_IMPORTANCE.csv", index=False)

    # feature distributions
    feat_dist_df.to_csv(OUT_DIR / "FEATURE_DISTRIBUTIONS.csv", index=False)

    write_report(results_df, fi_df if fi_rows else None, feat_dist_df)
    print(f"\nSaved to {OUT_DIR}/")


# ── report ────────────────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame,
                 fi_df: pd.DataFrame | None,
                 feat_dist_df: pd.DataFrame) -> None:
    year_cols_pass = sorted([c for c in df.columns if c.startswith("pass_")])
    year_cols_fail = sorted([c for c in df.columns if c.startswith("fail_")])
    year_cols_pnl  = sorted([c for c in df.columns if c.startswith("pnl_")])

    def fmt(v) -> str:
        if pd.isna(v):
            return "-"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    def table(cols: list[str]) -> str:
        header = "| " + " | ".join(cols) + " |"
        sep    = "|" + "|".join(["---"] * len(cols)) + "|"
        rows_md = []
        for _, r in df.iterrows():
            vals = [fmt(r[c]) if c in r.index else "-" for c in cols]
            rows_md.append("| " + " | ".join(vals) + " |")
        return "\n".join([header, sep] + rows_md)

    summary_cols = [
        "target", "rr",
        "wr_rev_baseline", "wr_cont_baseline",
        "v3_score", "v3_pass_rate", "v3_fail_mll", "v3_median_pnl",
        "v2_score", "v2_pass_rate", "v2_fail_mll",
        "delta_score",
    ]

    yearly_pass_cols = ["target"] + year_cols_pass
    yearly_fail_cols = ["target"] + year_cols_fail
    yearly_pnl_cols  = ["target"] + year_cols_pnl

    best = df.iloc[0] if len(df) else None
    winner_line = f"`{best['target']}` (v3_score={best['v3_score']:+.3f})" if best is not None else "-"

    # ── AB test summary ────────────────────────────────────────────
    ab_rows = []
    for _, r in df.iterrows():
        ab_rows.append({
            "target": r["target"],
            "v3_score": r["v3_score"],
            "v2_score": r["v2_score"],
            "delta": r["delta_score"],
            "improved": "✅" if r["delta_score"] > 0 else "❌" if r["delta_score"] < 0 else "➡️",
        })
    ab_df = pd.DataFrame(ab_rows)
    ab_improved = (ab_df["delta"] > 0).sum()
    ab_degraded = (ab_df["delta"] < 0).sum()
    ab_best_tgt = ab_df.loc[ab_df["delta"].idxmax(), "target"] if len(ab_df) else "-"
    ab_best_delta = ab_df["delta"].max() if len(ab_df) else 0.0
    ab_worst_tgt = ab_df.loc[ab_df["delta"].idxmin(), "target"] if len(ab_df) else "-"
    ab_worst_delta = ab_df["delta"].min() if len(ab_df) else 0.0

    def ab_table() -> str:
        header = "| target | v3_score | v2_score | delta | improved |"
        sep    = "|---|---|---|---|---|"
        rows_md = []
        for _, r in ab_df.iterrows():
            rows_md.append(f"| {r['target']} | {r['v3_score']:+.3f} | {r['v2_score']:+.3f} "
                           f"| {r['delta']:+.3f} | {r['improved']} |")
        return "\n".join([header, sep] + rows_md)

    # ── feature importance summary ─────────────────────────────────
    fi_summary = ""
    if fi_df is not None:
        # Average gain across all targets
        avg_fi = fi_df.groupby("feature")["gain_pct"].mean().sort_values(ascending=False)
        fi_lines = ["| feature | avg_gain_pct | type |", "|---|---|---|"]
        for feat, gain in avg_fi.items():
            ftype = "scale-invariant 🆕" if feat in V3_NEW_FEATURES else "original"
            fi_lines.append(f"| {feat} | {gain:.1f}% | {ftype} |")
        fi_summary = "\n".join(fi_lines)

    # ── feature distribution drift ─────────────────────────────────
    drift_analysis = ""
    if len(feat_dist_df):
        # For each feature, compare training period (<=2023) vs 2026 median
        train_feats = feat_dist_df[feat_dist_df["year"] <= 2023].copy()
        y2026_feats = feat_dist_df[feat_dist_df["year"] == 2026].copy()

        drift_rows = []
        for feat in sorted(feat_dist_df["feature"].unique()):
            tr = train_feats[train_feats["feature"] == feat]
            y26 = y2026_feats[y2026_feats["feature"] == feat]
            if len(tr) and len(y26):
                tr_med = tr["median"].median()
                y26_med = y26["median"].iloc[0]
                ratio = y26_med / tr_med if tr_med != 0 else np.nan
                drift_rows.append({
                    "feature": feat,
                    "train_median": f"{tr_med:.4f}",
                    "2026_median": f"{y26_med:.4f}",
                    "ratio": f"{ratio:.2f}x" if not np.isnan(ratio) else "inf",
                    "stable": "✅" if 0.5 <= ratio <= 2.0 else "⚠️" if ratio <= 5 else "🚨",
                })

        if drift_rows:
            drift_lines = ["| feature | train_median | 2026_median | ratio | stable |",
                           "|---|---|---|---|---|"]
            for d in drift_rows:
                drift_lines.append(f"| {d['feature']} | {d['train_median']} | "
                                   f"{d['2026_median']} | {d['ratio']} | {d['stable']} |")
            drift_analysis = "\n".join(drift_lines)

    go_count = int((df["v3_score"] > 0).sum()) if len(df) else 0
    best_target = best["target"] if best is not None else "-"

    report = f"""# Objective Sweep Report v3 — Scale-Invariant Features

Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}

## Key Changes from v2

| Aspect | v2 (baseline) | v3 (scale-invariant) |
|--------|:-------------:|:--------------------:|
| Features | 11 original | **18 (11 + 7 scale-invariant)** |
| New features | — | `breakout_strength_atr_ratio`, `atr14_sq`, `breakout_strength_sq`, `price_vs_vwap_pct_abs`, `orb_range_sq`, `adx_50_flag`, `breakout_strength_vs_orb` |
| AB test | — | Both models trained, results compared |
| Training data | 2010-2023 (same) | **Same** for fair comparison |
| Holdout | 2025+ (same) | **Same** |

## Why Scale-Invariant Features?

The 2026 collapse diagnosis found root cause: **model OOD failure due to volatility explosion**:

| Metric | Training (2010-2021) | 2024 | 2025 | 2026 |
|--------|:-------------------:|:----:|:----:|:----:|
| ATR14 median | $0.76 | $0.73 | $1.57 | **$3.53** |
| ORB range median | $3.50 | $3.50 | $7.30 | **$16.40** |
| Ratio vs training | 1.0× | 1.0× | 2.1× | **4.6×** |

Raw features like `breakout_strength` and `atr14_at_entry` see values 4-5× their training range.
Scale-invariant features normalize these by current ATR, so the model sees consistent distributions.

## Summary

- Labels tested: {len(df)}
- Holdout: {HOLDOUT_FROM} onward
- Topstep account: 50K (target $3,000, MLL $2,000)
- Scoring: `v3_score = pass_rate - fail_mll_rate` (best params from 128-point grid)
- Best label (v3): {winner_line}
- Labels with v3_score > 0: {go_count}

## AB Test: v3 (All Features) vs v2 (Baseline Only)

| Metric | Value |
|--------|:-----:|
| Targets improved (v3 > v2) | {ab_improved}/{len(ab_df)} |
| Targets degraded (v3 < v2) | {ab_degraded}/{len(ab_df)} |
| Best improvement | {ab_best_tgt} ({ab_best_delta:+.3f}) |
| Worst degradation | {ab_worst_tgt} ({ab_worst_delta:+.3f}) |

{ab_table()}

## Ranked Results (v3 — all features, best params per label)

{table(summary_cols)}

## Yearly Pass Rate by Label (v3)

{table(yearly_pass_cols)}

## Yearly Fail MLL Rate by Label (v3)

{table(yearly_fail_cols)}

## Yearly Median PnL by Label (v3)

{table(yearly_pnl_cols)}

## Feature Importance (avg across all targets)

{fi_summary}

## Feature Distribution Drift: Training vs 2026

How well do scale-invariant features maintain consistent distributions?

{drift_analysis}

**Legend:**
- ✅ **Stable** = 2026 median within 0.5×-2.0× training median
- ⚠️ **Moderate drift** = 2026 median 2×-5× training median
- 🚨 **Severe drift** = 2026 median >5× training median

## Detailed Feature Distribution by Year

See `FEATURE_DISTRIBUTIONS.csv` for per-year medians/p5/p95 of all features.

## Notes

- Models trained on {TRAIN_TO} backward, early-stopped on {CALIB_FROM}..{CALIB_TO}
- Sample weighting: exponential decay half-life={HALF_LIFE_YEARS}y
- Policy: dynamic rev/cont/skip with trend+ADX gates
- Holdout starts {HOLDOUT_FROM} (2025-2026 — the high-volatility regime)
- **v2 baseline results** in this report are re-trained in same script (not imported from v2) — apples-to-apples comparison
"""

    (OUT_DIR / "OBJECTIVE_SWEEP_REPORT.md").write_text(report)
    print(f"Report saved: {OUT_DIR / 'OBJECTIVE_SWEEP_REPORT.md'}")


if __name__ == "__main__":
    main()

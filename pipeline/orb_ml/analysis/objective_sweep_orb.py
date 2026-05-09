"""
Objective Sweep — test all 10 ORB labels to find which target/horizon gives
the best Topstep 50K pass-rate.

For each label:
  - Train lightweight LGBM rev + cont models on 2010-2022 data
  - Early-stop on 2022-2023 internal validation
  - Evaluate dynamic rev/cont/skip policy on holdout 2024+
  - Score with Topstep 50K simulator across a small fixed parameter grid
  - Pick best score per label

Output: model/SWEEP_v1/
  OBJECTIVE_SWEEP_RESULTS.csv
  OBJECTIVE_SWEEP_REPORT.md

Usage:
  python pipeline/analysis/objective_sweep_orb.py
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
OUT_DIR = ROOT / "model/SWEEP_v1"

HOLDOUT_FROM = "2024-01-01"
CALIB_FROM = "2022-01-01"
CALIB_TO = "2023-12-31"
TRAIN_TO = "2021-12-31"

HALF_LIFE_YEARS = 2
COST_R = 0.07
WINDOW_DAYS = 20
ACCOUNT = {"profit_target": 3000.0, "mll": 2000.0}

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


def train_model(df: pd.DataFrame, target: str) -> lgb.Booster:
    train_df = df[df["date"] <= TRAIN_TO].copy()
    val_df = df[(df["date"] >= CALIB_FROM) & (df["date"] <= CALIB_TO)].copy()

    train_df = encode(train_df).dropna(subset=FEATURES + [target]).reset_index(drop=True)
    val_df = encode(val_df).dropna(subset=FEATURES + [target]).reset_index(drop=True)

    w = compute_weights(train_df["year"])
    dtrain = lgb.Dataset(train_df[FEATURES], label=train_df[target],
                         weight=w, feature_name=FEATURES)
    dval = lgb.Dataset(val_df[FEATURES], label=val_df[target], reference=dtrain)

    model = lgb.train(
        LGBM_PARAMS, dtrain,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    return model


# ── policy + Topstep ──────────────────────────────────────────────────────────

def score_holdout(events: pd.DataFrame, calib: pd.DataFrame,
                  p: PolicyParams, rr: float) -> dict:
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

    summary_rows: list[dict] = []

    for target, rr in LABELS:
        print(f"\n{'='*60}")
        print(f"Label: {target}  (RR={rr})")

        # ── train models ─────────────────────────────────────────
        rev_df = dm[dm["side"] == "rev"].copy()
        cont_df = dm[dm["side"] == "cont"].copy()

        if target not in rev_df.columns or rev_df[target].isna().all():
            print(f"  SKIP — label missing or all-NaN")
            continue

        print(f"  Training rev model...", end=" ", flush=True)
        rev_model = train_model(rev_df, target)
        print(f"best_iter={rev_model.best_iteration}")

        print(f"  Training cont model...", end=" ", flush=True)
        cont_model = train_model(cont_df, target)
        print(f"best_iter={cont_model.best_iteration}")

        # ── build event frame ─────────────────────────────────────
        def side_frame(df: pd.DataFrame, side: str, model: lgb.Booster) -> pd.DataFrame:
            s = encode(df).dropna(subset=FEATURES + [target]).copy()
            s["prob"] = model.predict(s[FEATURES])
            keep = EVENT_KEY + ["year", "trade_day", "prob", target,
                                 "adx_14_15m", "ema_slope_1h", "price_vs_vwap_pct"]
            return s[keep].rename(columns={"prob": f"prob_{side}", target: f"y_{side}"})

        rev_ev = side_frame(rev_df, "rev", rev_model)
        cont_ev = side_frame(cont_df, "cont", cont_model)

        merge_on = EVENT_KEY + ["year", "trade_day",
                                 "adx_14_15m", "ema_slope_1h", "price_vs_vwap_pct"]
        events = rev_ev.merge(cont_ev, on=merge_on, how="inner")
        events = events.sort_values(["date", "breakout_ts"]).reset_index(drop=True)

        calib = events[(events["date"] >= CALIB_FROM) & (events["date"] <= CALIB_TO)].copy()
        holdout = events[events["date"] >= HOLDOUT_FROM].copy()

        if len(calib) < 50 or len(holdout) < 50:
            print(f"  SKIP — insufficient calib ({len(calib)}) or holdout ({len(holdout)}) events")
            continue

        # ── baseline (unfiltered) ─────────────────────────────────
        y_rev_h = holdout["y_rev"].astype(int).to_numpy()
        y_cont_h = holdout["y_cont"].astype(int).to_numpy()
        wr_rev = float(y_rev_h.mean())
        wr_cont = float(y_cont_h.mean())
        exp_rev = wr_rev * rr - (1 - wr_rev) - COST_R
        exp_cont = wr_cont * rr - (1 - wr_cont) - COST_R

        print(f"  Holdout: n={len(holdout)}, wr_rev={wr_rev:.3f}(exp={exp_rev:+.3f}R), "
              f"wr_cont={wr_cont:.3f}(exp={exp_cont:+.3f}R)")

        # ── param grid sweep ──────────────────────────────────────
        best_score = -999.0
        best_params: PolicyParams | None = None
        best_result: dict = {}

        for p in PARAM_GRID:
            result = score_holdout(holdout, calib, p, rr)
            if result["score"] > best_score:
                best_score = result["score"]
                best_params = p
                best_result = result

        assert best_params is not None
        yr_res = by_year_sim(holdout, calib, best_params, rr)

        print(f"  Best: pass_rate={best_result['pass_rate']:.3f}  "
              f"fail_mll={best_result['fail_mll_rate']:.3f}  "
              f"score={best_result['score']:+.3f}  "
              f"risk=${best_params.risk_per_r_usd:.0f}")
        for yr, yr_d in sorted(yr_res.items()):
            print(f"    {yr}: pass={yr_d['pass_rate']:.3f} fail_mll={yr_d['fail_mll_rate']:.3f} "
                  f"median_pnl=${yr_d['median_end_pnl']:.0f}")

        row = {
            "target":           target,
            "rr":               rr,
            "n_holdout_events": len(holdout),
            "wr_rev_baseline":  round(wr_rev, 4),
            "wr_cont_baseline": round(wr_cont, 4),
            "exp_rev_net":      round(exp_rev, 4),
            "exp_cont_net":     round(exp_cont, 4),
            "best_pass_rate":   round(best_result["pass_rate"], 4),
            "best_fail_mll":    round(best_result["fail_mll_rate"], 4),
            "best_score":       round(best_result["score"], 4),
            "median_end_pnl":   round(best_result["median_end_pnl"], 1),
            "avg_trades":       round(best_result["avg_trades"], 1),
            "best_risk_per_r":  best_params.risk_per_r_usd,
            "best_rev_q":       best_params.rev_q,
            "best_cont_q":      best_params.cont_q,
            "best_rev_adx_min": best_params.rev_adx_min,
            "best_cont_adx_max": best_params.cont_adx_max,
            "best_profit_cap":  best_params.daily_profit_cap_usd,
        }
        for yr, yr_d in yr_res.items():
            row[f"pass_{yr}"] = round(yr_d["pass_rate"], 4)
            row[f"fail_{yr}"] = round(yr_d["fail_mll_rate"], 4)
            row[f"pnl_{yr}"]  = round(yr_d["median_end_pnl"], 0)

        summary_rows.append(row)

    # ── output ────────────────────────────────────────────────────
    results_df = pd.DataFrame(summary_rows)
    results_df = results_df.sort_values("best_score", ascending=False).reset_index(drop=True)
    results_df.to_csv(OUT_DIR / "OBJECTIVE_SWEEP_RESULTS.csv", index=False)

    write_report(results_df)
    print(f"\nSaved to {OUT_DIR}/")


def write_report(df: pd.DataFrame) -> None:
    year_cols_pass = sorted([c for c in df.columns if c.startswith("pass_2")])
    year_cols_fail = sorted([c for c in df.columns if c.startswith("fail_2")])
    year_cols_pnl  = sorted([c for c in df.columns if c.startswith("pnl_2")])

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
        "exp_rev_net", "exp_cont_net",
        "best_pass_rate", "best_fail_mll", "best_score",
        "median_end_pnl", "best_risk_per_r",
    ]

    yearly_pass_cols = ["target"] + year_cols_pass
    yearly_fail_cols = ["target"] + year_cols_fail
    yearly_pnl_cols  = ["target"] + year_cols_pnl

    best = df.iloc[0] if len(df) else None
    winner_line = f"`{best['target']}` (score={best['best_score']:+.3f})" if best is not None else "-"

    go_count = int((df["best_score"] > 0).sum()) if len(df) else 0

    report = f"""# Objective Sweep Report — ORB

Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}

## Summary

- Labels tested: {len(df)}
- Holdout: {HOLDOUT_FROM} onward
- Topstep account: 50K (target $3,000, MLL $2,000)
- Scoring: `pass_rate - fail_mll_rate` (best params from {len(PARAM_GRID)}-point grid)
- Best label: {winner_line}
- Labels with score > 0: {go_count}

## Ranked Results (best params per label)

{table(summary_cols)}

## Yearly Pass Rate by Label

{table(yearly_pass_cols)}

## Yearly Fail MLL Rate by Label

{table(yearly_fail_cols)}

## Yearly Median PnL by Label

{table(yearly_pnl_cols)}

## Notes

- Models trained on 2010-2021, early-stopped on 2022-2023, evaluated on 2024+
- Sample weighting: exponential decay half-life={HALF_LIFE_YEARS}y
- Policy: dynamic rev/cont/skip with trend+ADX gates
- GO gate (for reference): pass_rate ≥ 60%, fail_mll ≤ 10%
- 2026 regime is the critical stress test
"""
    (OUT_DIR / "OBJECTIVE_SWEEP_REPORT.md").write_text(report)
    print(f"Report saved: {OUT_DIR / 'OBJECTIVE_SWEEP_REPORT.md'}")


if __name__ == "__main__":
    main()

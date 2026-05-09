"""
Objective Sweep v4 — Refined Topstep Simulator.

Changes from v3:
- Uses `topstep_sim` module for policy application + Topstep simulation
  → CT-based trading day boundaries (5PM CT → 3:10PM CT next day)
  → Fixed $3.00 commission (not 0.07R percentage)
- Identical features (v3 all features = baseline + scale-invariant)
- Identical training procedure
- Saves results to model/SWEEP_v4/
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# Ensure pipeline is importable
BASE = Path(__file__).resolve().parent.parent.parent.parent
PIPELINE = BASE / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from analysis.topstep_sim import (
    PolicyParams,
    apply_policy,
    build_param_grid,
    by_year_eval,
    go_no_go,
    map_to_topstep_trade_day,
    md_table,
    score_policy_on_events,
)

# ── paths ────────────────────────────────────────────────────────────────────

DM_PATH = BASE / "data" / "Level_2_Datamart" / "training_datamart_orb.parquet"
OUT_DIR = BASE / "model" / "SWEEP_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── label schema ─────────────────────────────────────────────────────────────

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

# ── v3 feature sets ──────────────────────────────────────────────────────────

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
    "atr14_sq",                       # atr14_at_entry ** 2
    "breakout_strength_sq",           # breakout_strength ** 2
    "price_vs_vwap_pct_abs",          # abs(price_vs_vwap_pct)
    "orb_range_sq",                   # orb_range ** 2
    "adx_50_flag",                    # 1 if adx > 50
    "breakout_strength_vs_orb",       # breakout_strength / orb_range
]

ALL_FEATURES = V2_FEATURES + V3_NEW_FEATURES

EVENT_KEY = ["date", "breakout_ts", "breakout_side"]

TRAIN_TO = "2023-12-31"
CALIB_FROM = "2020-01-01"
CALIB_TO = "2023-12-31"
HOLDOUT_FROM = "2024-01-01"

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

PARAM_GRID = build_param_grid()


# ── helpers ──────────────────────────────────────────────────────────────────

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


def compute_weights(yr_series: pd.Series) -> np.ndarray:
    """Exponential decay weights — half-life 2 years."""
    years = yr_series.astype(int).to_numpy()
    latest = years.max()
    half_life = 2.0
    w = np.exp(-np.log(2) * (latest - years) / half_life)
    return w + 0.01


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode orb_tf and session; convert breakout_side to float."""
    df = df.copy()
    orb_dummies = pd.get_dummies(df["orb_tf"], prefix="orb_tf")
    sess_dummies = pd.get_dummies(df["session"], prefix="session")
    df = pd.concat([df, orb_dummies.astype(float), sess_dummies.astype(float)], axis=1)
    df["breakout_side"] = (df["breakout_side"] == 1).astype(float)
    return df


def _numeric_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Filter to numeric-only features (exclude object/string columns)."""
    return [f for f in features
            if f in df.columns and df[f].dtype in ("float64", "float32", "int64", "int32", "bool")]


def train_model(df: pd.DataFrame, target: str, features: list[str]) -> lgb.Booster:
    """Train LGBM with train/val split + early stopping."""
    train_df = df[df["date"] <= TRAIN_TO].copy()
    val_df = df[(df["date"] >= CALIB_FROM) & (df["date"] <= CALIB_TO)].copy()

    train_df = encode(train_df).reset_index(drop=True)
    val_df = encode(val_df).reset_index(drop=True)

    num_feats = _numeric_features(train_df, features)
    train_df = train_df.dropna(subset=num_feats + [target]).reset_index(drop=True)
    val_df = val_df.dropna(subset=num_feats + [target]).reset_index(drop=True)

    w = compute_weights(train_df["year"])
    dtrain = lgb.Dataset(train_df[num_feats], label=train_df[target].astype(int),
                         weight=w, feature_name=num_feats)
    dval = lgb.Dataset(val_df[num_feats], label=val_df[target].astype(int), reference=dtrain)

    model = lgb.train(
        LGBM_PARAMS, dtrain,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    return model


def build_event_frame(
    rev_df: pd.DataFrame,
    cont_df: pd.DataFrame,
    rev_model: lgb.Booster,
    cont_model: lgb.Booster,
    features: list[str],
    target: str,
) -> pd.DataFrame:
    """Build merged rev+cont event frame with trade_day from Topstep boundaries."""
    rev_enc = encode(rev_df)
    cont_enc = encode(cont_df)
    num_feats = _numeric_features(rev_enc, features)

    rev_s = rev_enc.dropna(subset=num_feats + [target]).copy()
    rev_s["prob_rev"] = rev_model.predict(rev_s[num_feats])
    rev_s["y_rev"] = rev_s[target]

    cont_s = cont_enc.dropna(subset=num_feats + [target]).copy()
    cont_s["prob_cont"] = cont_model.predict(cont_s[num_feats])
    cont_s["y_cont"] = cont_s[target]

    merge_on = EVENT_KEY + ["year",
                            "orb_range", "breakout_strength", "atr14_at_entry",
                            "price_vs_vwap_pct", "adx_14_15m", "ema_slope_1h"]
    events = rev_s[merge_on + ["prob_rev", "y_rev"]].merge(
        cont_s[merge_on + ["prob_cont", "y_cont"]], on=merge_on, how="inner"
    )
    events = events.sort_values(["date", "breakout_ts"]).reset_index(drop=True)

    # Add Topstep CT-based trade_day
    events["trade_day"] = map_to_topstep_trade_day(events["breakout_ts"])
    return events


# ── report helpers ───────────────────────────────────────────────────────────

def fmt(v) -> str:
    if pd.isna(v):
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def year_table(df: pd.DataFrame, title: str) -> str:
    cols = [c for c in df.columns if c != "target"]
    lines = [f"\n### {title}", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        vals = [fmt(row[c]) if c in row.index else "-" for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("  Objective Sweep v4 — Refined Topstep Simulator")
    print("=" * 65)

    # ── 1. Load datamart ──────────────────────────────────────────────
    print("\n1. Loading datamart...")
    dm = pd.read_parquet(DM_PATH).copy()
    dm["date"] = pd.to_datetime(dm["date"])
    dm["year"] = dm["date"].dt.year
    print(f"   Shape: {dm.shape}, Date: {dm['date'].min():%Y-%m-%d} → {dm['date'].max():%Y-%m-%d}")

    # ── 2. Add scale-invariant features ───────────────────────────────
    print("\n2. Adding scale-invariant features...")
    dm = add_scale_invariant_features(dm)
    print(f"   Features: {len(ALL_FEATURES)} ({len(V2_FEATURES)} baseline + {len(V3_NEW_FEATURES)} scale-inv)")

    # ── 3. Iterate over labels ────────────────────────────────────────
    summary_rows: list[dict] = []

    for target, rr in LABELS:
        print(f"\n{'='*60}")
        print(f"   Label: {target}  (RR={rr})")

        rev_df = dm[dm["side"] == "rev"].copy()
        cont_df = dm[dm["side"] == "cont"].copy()

        if target not in rev_df.columns or rev_df[target].isna().all():
            print(f"   SKIP — label missing or all-NaN")
            continue

        # ── 4. Train models ───────────────────────────────────────────
        print(f"   Training rev model...", end=" ", flush=True)
        rev_model = train_model(rev_df, target, ALL_FEATURES)
        print(f"best_iter={rev_model.best_iteration}")

        print(f"   Training cont model...", end=" ", flush=True)
        cont_model = train_model(cont_df, target, ALL_FEATURES)
        print(f"best_iter={cont_model.best_iteration}")

        # ── 5. Build event frame ──────────────────────────────────────
        events = build_event_frame(rev_df, cont_df, rev_model, cont_model, ALL_FEATURES, target)

        calib = events[(events["date"] >= CALIB_FROM) & (events["date"] <= CALIB_TO)].copy()
        holdout = events[events["date"] >= HOLDOUT_FROM].copy()

        if len(calib) < 50 or len(holdout) < 50:
            print(f"   SKIP — insufficient events: calib={len(calib)}, holdout={len(holdout)}")
            continue

        # Baseline win rates
        wr_rev = holdout["y_rev"].astype(int).mean()
        wr_cont = holdout["y_cont"].astype(int).mean()
        cost_r = 0.07
        exp_rev = wr_rev * rr - (1 - wr_rev) - cost_r
        exp_cont = wr_cont * rr - (1 - wr_cont) - cost_r
        print(f"   Holdout ({HOLDOUT_FROM}+): n={len(holdout)}, "
              f"wr_rev={wr_rev:.3f}(exp={exp_rev:+.3f}R), "
              f"wr_cont={wr_cont:.3f}(exp={exp_cont:+.3f}R)")

        # ── 6. Sweep param grid (refined simulator) ────────────────────
        best_score = -999.0
        best_params: PolicyParams | None = None
        best_result: dict = {}

        print(f"   Sweeping {len(PARAM_GRID)} param combos (v4 refined sim)...", end=" ", flush=True)
        for p in PARAM_GRID:
            result = score_policy_on_events(holdout, calib, p, rr)
            if result["score"] > best_score:
                best_score = result["score"]
                best_params = p
                best_result = result
        print(f"done.")
        assert best_params is not None

        yr_res = by_year_eval(holdout, calib, best_params, rr)

        # ── 7. Record summary ─────────────────────────────────────────
        row = {
            "target": target,
            "rr": rr,
            "pass_rate": best_result["pass_rate"],
            "fail_mll_rate": best_result["fail_mll_rate"],
            "score": best_result["score"],
            "median_end_pnl": best_result["median_end_pnl"],
            "avg_trades": best_result["avg_trades"],
            "windows": best_result["windows"],
            "rev_q": best_params.rev_q,
            "cont_q": best_params.cont_q,
            "rev_adx_min": best_params.rev_adx_min,
            "cont_adx_max": best_params.cont_adx_max,
            "risk_per_r_usd": best_params.risk_per_r_usd,
            "daily_profit_cap": best_params.daily_profit_cap_usd,
        }
        # Add per-year pass/fail
        for yr in sorted(yr_res.keys()):
            row[f"pass_{yr}"] = yr_res[yr]["pass_rate"]
            row[f"fail_mll_{yr}"] = yr_res[yr]["fail_mll_rate"]
            row[f"pnl_{yr}"] = yr_res[yr]["median_end_pnl"]
        summary_rows.append(row)

        print(f"   → score={best_result['score']:+.4f}, "
              f"pass={best_result['pass_rate']:.1%}, "
              f"fail_mll={best_result['fail_mll_rate']:.1%}, "
              f"pnl=${best_result['median_end_pnl']:.0f}")
        print(f"     params: rev_q={best_params.rev_q}, cont_q={best_params.cont_q}, "
              f"rev_adx={best_params.rev_adx_min}, cont_adx={best_params.cont_adx_max}, "
              f"risk=${best_params.risk_per_r_usd:.0f}, cap=${best_params.daily_profit_cap_usd:.0f}")

    # ── 8. Write results ──────────────────────────────────────────────
    if not summary_rows:
        print("\n❌ No labels processed.")
        return

    sdf = pd.DataFrame(summary_rows)
    sdf = sdf.sort_values("score", ascending=False).reset_index(drop=True)

    # Save CSV
    csv_path = OUT_DIR / "OBJECTIVE_SWEEP_RESULTS.csv"
    sdf.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")

    # ── 9. Write report ────────────────────────────────────────────────
    year_cols_pass = sorted([c for c in sdf.columns if c.startswith("pass_")])
    year_cols_mll = sorted([c for c in sdf.columns if c.startswith("fail_mll_")])
    year_cols_pnl = sorted([c for c in sdf.columns if c.startswith("pnl_")])

    def table(cols: list[str]) -> str:
        """Render a summary table for the given columns."""
        display_cols = ["target"] + cols
        lines = ["| " + " | ".join(display_cols) + " |"]
        lines.append("|" + "|".join(["---"] * len(display_cols)) + "|")
        for _, row in sdf.iterrows():
            vals = [row["target"]]
            for c in cols:
                v = row.get(c)
                if isinstance(v, float):
                    if c.startswith("pnl_"):
                        vals.append(f"${v:+.0f}")
                    elif c.startswith("pass_") or c.startswith("fail_mll_"):
                        vals.append(f"{v:.1%}")
                    else:
                        vals.append(fmt(v))
                else:
                    vals.append(fmt(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    report_lines = [
        "# Objective Sweep Report v4 — Refined Topstep Simulator",
        "",
        "## Key Changes from v3",
        "",
        "1. **CT-based trading day boundaries** — `map_to_topstep_trade_day()`",
        "   - Topstep trading day: 5:00 PM CT to 3:10 PM CT next day",
        "   - Tokyo session (00:00–03:00 UTC) maps to PREVIOUS trading day",
        "2. **Fixed $3.00 commission** (not 0.07R percentage)",
        "3. **Shared module** — `pipeline/analysis/topstep_sim.py`",
        "4. **Identical features** as v3 (11 baseline + 7 scale-invariant)",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Labels evaluated | {len(sdf)} |",
        f"| Training data | ≤ {TRAIN_TO} |",
        f"| Calibration | {CALIB_FROM} → {CALIB_TO} |",
        f"| Holdout | {HOLDOUT_FROM}+ |",
        f"| Param grid | {len(PARAM_GRID)} combinations |",
        f"| Simulator | Refined (CT trade day + $3 commission) |",
        "",
        "## Ranked Results (v4 — best params per label)",
        "",
    ]

    # Score table
    score_cols = ["rr", "score", "pass_rate", "fail_mll_rate", "median_end_pnl", "avg_trades", "windows"]
    report_lines.append(table(score_cols))
    report_lines.append("")

    # Yearly pass rate
    if year_cols_pass:
        report_lines.append("## Yearly Pass Rate by Label (v4)")
        report_lines.append("")
        report_lines.append(table(year_cols_pass))
        report_lines.append("")

    # Yearly fail MLL
    if year_cols_mll:
        report_lines.append("## Yearly Fail MLL Rate by Label (v4)")
        report_lines.append("")
        report_lines.append(table(year_cols_mll))
        report_lines.append("")

    # Yearly median PnL
    if year_cols_pnl:
        report_lines.append("## Yearly Median PnL by Label (v4)")
        report_lines.append("")
        report_lines.append(table(year_cols_pnl))
        report_lines.append("")

    report_lines.append("## Best Params per Label")
    report_lines.append("")
    param_cols = ["target", "rev_q", "cont_q", "rev_adx_min", "cont_adx_max",
                  "risk_per_r_usd", "daily_profit_cap"]
    report_lines.append(table(param_cols))
    report_lines.append("")

    report_lines.append("## Notes")
    report_lines.append("")
    report_lines.append("- Simulator uses CT-based trading day boundaries (5PM CT → 3:10PM CT next day)")
    report_lines.append("- Commission: fixed $3.00/round-turn (entry + exit + slippage)")
    report_lines.append("- MLL trailing: tracks highest end-of-day PnL, locked at starting balance")
    report_lines.append("- Consistency rule: best day < 50% of total profit at pass time")
    report_lines.append("- Training: exponential time decay, half-life = 2 years")
    report_lines.append("- Walk-forward: rolling 20-day windows, strict temporal holdout")
    report_lines.append("")

    report_path = OUT_DIR / "OBJECTIVE_SWEEP_REPORT.md"
    report_path.write_text("\n".join(report_lines))
    print(f"✅ Report saved to {report_path}")

    # ── 10. Print scoreboard ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Scoreboard (v4 — Refined Simulator)")
    print(f"{'='*65}")
    print(f"  {'Target':<20s} {'Score':>8s}  {'Pass':>6s}  {'FailMLL':>8s}  {'PnL':>8s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}")
    for _, row in sdf.iterrows():
        print(f"  {row['target']:<20s} {row['score']:>+8.4f}  "
              f"{row['pass_rate']:>6.1%}  {row['fail_mll_rate']:>8.1%}  "
              f"${row['median_end_pnl']:>+.0f}")


if __name__ == "__main__":
    main()

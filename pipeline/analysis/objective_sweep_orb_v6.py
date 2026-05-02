"""
Objective Sweep v6 — Modular Feature Architecture.

Key changes from v5:
- Features loaded from ``data/Level_1_Features/modules/`` via ``load_features_from_modules()``
- No inline ``add_scale_invariant_features()`` — those are now a module
- Feature list is auto-derived: core feature columns + module columns
- Adding new features = adding a new module parquet, not changing sweep code

Same sweep logic, Topstep simulator, and report format as v5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# Ensure pipeline and project root are importable
BASE = Path(__file__).resolve().parent.parent.parent
PIPELINE = BASE / "pipeline"
for p in [str(BASE), str(PIPELINE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from analysis.topstep_sim import (
    PolicyParams,
    build_param_grid,
    by_year_eval,
    map_to_topstep_trade_day,
    score_policy_on_events,
)
from pipeline.feature.modules.loader import load_features_from_modules

# ── paths ────────────────────────────────────────────────────────────────────

DM_PATH = BASE / "data" / "Level_2_Datamart" / "training_datamart_orb.parquet"
MODULES_DIR = BASE / "data" / "Level_1_Features" / "modules"
OUT_DIR = BASE / "model" / "SWEEP_v6"
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

# ── data model ────────────────────────────────────────────────────────────────

# Core columns loaded from the datamart — identifiers, labels, and raw metadata.
# Feature columns come EXCLUSIVELY from modules/*_features.parquet.
CORE_COLS = [
    # Identifiers
    "date", "session", "orb_tf", "breakout_ts", "breakout_side", "side",
    # Raw metadata (some also used as features)
    "entry_price", "orb_range", "atr14_at_entry", "sl_dist", "breakout_strength",
    "session_close_ts",
    # Labels (10 binary targets)
    "y_1r2_60m", "y_1r4_60m", "y_1r2_120m", "y_1r4_120m",
    "y_1r2_180m", "y_1r4_180m", "y_1r2_240m", "y_1r4_240m",
    "y_1r2_close60m", "y_1r4_close60m",
]

# Columns from CORE_COLS that are used directly as model features
# (rather than as merge keys or labels).
CORE_FEATURES = [
    "breakout_strength",
    "atr14_at_entry",
    "orb_tf",         # one-hot encoded
    "session",        # one-hot encoded
    "breakout_side",  # boolean-ized
]

# Module grain key (breakout-event level)
EVENT_KEY = ["date", "session", "orb_tf", "breakout_ts"]

# Merge key for rev/cont event frame
MERGE_KEY = ["date", "breakout_ts", "breakout_side"]

# ── sweep config ─────────────────────────────────────────────────────────────

TRAIN_TO = "2025-11-30"
CALIB_FROM = "2025-07-01"
CALIB_TO = "2025-11-30"
HOLDOUT_FROM = "2025-12-01"

LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "learning_rate":    0.05,
    "num_leaves":       31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "verbose":         -1,
    "n_jobs":          -1,
}
NUM_ROUNDS = 500
EARLY_STOP = 30

PARAM_GRID = build_param_grid()


# ── helpers ──────────────────────────────────────────────────────────────────


def compute_weights(yr_series: pd.Series) -> np.ndarray:
    """Exponential decay weights — half-life 1 year."""
    years = yr_series.astype(int).to_numpy()
    latest = years.max()
    half_life = 1.0
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

    if len(train_df) < 100 or len(val_df) < 50:
        raise ValueError(f"Too few training rows: train={len(train_df)}, val={len(val_df)}")

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

    merge_on = MERGE_KEY + ["year",
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
    print("  Objective Sweep v6 — Modular Feature Architecture")
    print("=" * 65)

    # ── 1. Load datamart (core columns only) ─────────────────────────
    print("\n1. Loading datamart (core columns only)...")
    dm = pd.read_parquet(DM_PATH, columns=CORE_COLS).copy()
    print(f"   Shape: {dm.shape}, Date: {dm['date'].min()} → {dm['date'].max()}")

    # ── 2. Load feature modules ──────────────────────────────────────
    print("\n2. Loading feature modules...")
    dm = load_features_from_modules(MODULES_DIR, dm)
    print(f"   Shape after modules: {dm.shape}")

    # Convert date to datetime after modules (modules keep date as object)
    dm["date"] = pd.to_datetime(dm["date"])
    dm["year"] = dm["date"].dt.year

    # ── 3. Derive feature list ───────────────────────────────────────
    # Features = core numeric columns + all module columns (excluding EVENT_KEY)
    EXCLUDED_COLS = set(EVENT_KEY) | {"year"}  # derived or non-feature cols
    module_cols = sorted(
        set(dm.columns) - set(CORE_COLS) - EXCLUDED_COLS
    )
    ALL_FEATURES = CORE_FEATURES + module_cols
    print(f"\n3. Feature list ({len(ALL_FEATURES)} total):")
    print(f"   Core ({len(CORE_FEATURES)}): {CORE_FEATURES}")
    print(f"   Module ({len(module_cols)}): {module_cols}")

    # ── 4. Iterate over labels ────────────────────────────────────────
    summary_rows: list[dict] = []

    for target, rr in LABELS:
        print(f"\n{'='*60}")
        print(f"   Label: {target}  (RR={rr})")

        rev_df = dm[dm["side"] == "rev"].copy()
        cont_df = dm[dm["side"] == "cont"].copy()

        if target not in rev_df.columns or rev_df[target].isna().all():
            print(f"   SKIP — label missing or all-NaN")
            continue

        # ── 5. Train models ──────────────────────────────────────────
        print(f"   Training rev model ({TRAIN_TO} train, {CALIB_FROM}..{CALIB_TO} val)...",
              end=" ", flush=True)
        try:
            rev_model = train_model(rev_df, target, ALL_FEATURES)
            print(f"best_iter={rev_model.best_iteration}")
        except ValueError as e:
            print(f"FAILED: {e}")
            continue

        print(f"   Training cont model ({TRAIN_TO} train, {CALIB_FROM}..{CALIB_TO} val)...",
              end=" ", flush=True)
        try:
            cont_model = train_model(cont_df, target, ALL_FEATURES)
            print(f"best_iter={cont_model.best_iteration}")
        except ValueError as e:
            print(f"FAILED: {e}")
            continue

        # ── 6. Build event frame ──────────────────────────────────────
        events = build_event_frame(rev_df, cont_df, rev_model, cont_model,
                                   ALL_FEATURES, target)

        holdout = events[events["date"] >= HOLDOUT_FROM].copy()
        calib = events[(events["date"] >= CALIB_FROM) & (events["date"] <= CALIB_TO)].copy()

        if len(holdout) < 50:
            print(f"   SKIP — insufficient holdout events: {len(holdout)}")
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

        # ── 7. Sweep param grid ────────────────────────────────────────
        best_score = -999.0
        best_params: PolicyParams | None = None
        best_result: dict = {}

        print(f"   Sweeping {len(PARAM_GRID)} param combos...", end=" ", flush=True)
        for p in PARAM_GRID:
            result = score_policy_on_events(holdout, calib, p, rr)
            if result["score"] > best_score:
                best_score = result["score"]
                best_params = p
                best_result = result
        print(f"done.")
        assert best_params is not None

        yr_res = by_year_eval(holdout, calib, best_params, rr)

        # ── 8. Record summary ─────────────────────────────────────────
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

    # ── 9. Write results ──────────────────────────────────────────────
    if not summary_rows:
        print("\n❌ No labels processed.")
        return

    sdf = pd.DataFrame(summary_rows)
    sdf = sdf.sort_values("score", ascending=False).reset_index(drop=True)

    csv_path = OUT_DIR / "OBJECTIVE_SWEEP_RESULTS.csv"
    sdf.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")

    # ── 10. Write report ────────────────────────────────────────────────
    year_cols_pass = sorted([c for c in sdf.columns if c.startswith("pass_")])
    year_cols_mll = sorted([c for c in sdf.columns if c.startswith("fail_mll_")])
    year_cols_pnl = sorted([c for c in sdf.columns if c.startswith("pnl_")])

    def table(cols: list[str]) -> str:
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
        "# Objective Sweep v6 — Modular Feature Architecture",
        "",
        "## Key Changes from v5",
        "",
        "| Aspect | v5 | v6 |",
        "|--------|:--:|:--:|",
        "| Feature source | Hardcoded in script | **Modules from `data/Level_1_Features/modules/`** |",
        "| Scale-invariant features | Inline `add_scale_invariant_features()` | **`scale_invariant_features` module** |",
        "| Context features | From datamart (patched by `build_market_context.py`) | **`orb_context_features` module** |",
        "| Adding new features | Modify `build_market_context.py` + rebuild datamart | **Create new module generator → run → re-sweep** |",
        "| Feature list | `V2_FEATURES` + `V3_NEW_FEATURES` hardcoded | **Auto-derived: core + module columns** |",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Labels evaluated | {len(sdf)} |",
        f"| Training data | {TRAIN_TO} (recent regime) |",
        f"| Calibration | {CALIB_FROM} → {CALIB_TO} |",
        f"| Holdout | {HOLDOUT_FROM}+ |",
        f"| Param grid | {len(PARAM_GRID)} combinations |",
        f"| Simulator | Refined (CT trade day + $3 commission) |",
        f"| Feature modules | {len(list(MODULES_DIR.glob('*_features.parquet')))} |",
        "",
        "## Ranked Results (v6 — best params per label)",
        "",
    ]

    score_cols = ["rr", "score", "pass_rate", "fail_mll_rate",
                  "median_end_pnl", "avg_trades", "windows"]
    report_lines.append(table(score_cols))
    report_lines.append("")

    if year_cols_pass:
        report_lines.append("## Yearly Pass Rate by Label (v6)")
        report_lines.append("")
        report_lines.append(table(year_cols_pass))
        report_lines.append("")

    if year_cols_mll:
        report_lines.append("## Yearly Fail MLL Rate by Label (v6)")
        report_lines.append("")
        report_lines.append(table(year_cols_mll))
        report_lines.append("")

    if year_cols_pnl:
        report_lines.append("## Yearly Median PnL by Label (v6)")
        report_lines.append("")
        report_lines.append(table(year_cols_pnl))
        report_lines.append("")

    report_lines.append("## Best Params per Label")
    report_lines.append("")
    param_cols = ["target", "rev_q", "cont_q", "rev_adx_min", "cont_adx_max",
                  "risk_per_r_usd", "daily_profit_cap"]
    report_lines.append(table(param_cols))
    report_lines.append("")

    report_lines.append("## Feature Modules Used")
    report_lines.append("")
    for fpath in sorted(MODULES_DIR.glob("*_features.parquet")):
        mod = pd.read_parquet(fpath)
        feats = [c for c in mod.columns if c not in EVENT_KEY]
        report_lines.append(f"- `{fpath.name}`: {len(feats)} features — `{'`, `'.join(feats)}`")
    report_lines.append("")

    report_path = OUT_DIR / "OBJECTIVE_SWEEP_REPORT.md"
    report_path.write_text("\n".join(report_lines))
    print(f"✅ Report saved to {report_path}")

    # ── 11. Print scoreboard ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Scoreboard (v6 — Modular Features)")
    print(f"{'='*65}")
    print(f"  {'Target':<20s} {'Score':>8s}  {'Pass':>6s}  {'FailMLL':>8s}  {'PnL':>8s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}")
    for _, row in sdf.iterrows():
        print(f"  {row['target']:<20s} {row['score']:>+8.4f}  "
              f"{row['pass_rate']:>6.1%}  {row['fail_mll_rate']:>8.1%}  "
              f"${row['median_end_pnl']:>+.0f}")


if __name__ == "__main__":
    main()

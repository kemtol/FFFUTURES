#!/usr/bin/env python3
"""
v6 Tuning Sweep — Parameter Grid Optimization

Focus: higher ``rev_q`` / ``cont_q`` thresholds to reduce trade frequency
and lower fail-MLL rates while preserving pass-rate gains from
volatility-normalized features.

Approach
────────
- Reuses v6's full pipeline (data loading, modules, training, simulation)
- Overrides the param grid with tighter probability thresholds
- Saves results to ``model/SWEEP_v6/TUNE_PARAMS/``

Usage
─────
    python3 pipeline/analysis/objective_sweep_orb_v6_tune.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root (mirrors v6's sys.path setup)
BASE = Path(__file__).resolve().parent.parent.parent
PIPELINE = BASE / "pipeline"
for p in [str(BASE), str(PIPELINE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd

from analysis.objective_sweep_orb_v6 import (  # type: ignore[import-untyped]
    CALIB_FROM,
    CALIB_TO,
    CORE_COLS,
    CORE_FEATURES,
    EVENT_KEY,
    HOLDOUT_FROM,
    LABELS,
    LGBM_PARAMS,
    NUM_ROUNDS,
    EARLY_STOP,
    OUT_DIR,
    build_event_frame,
    compute_weights,
    encode,
    fmt,
    year_table,
    _numeric_features,
)
from analysis.topstep_sim import (
    PolicyParams,
    by_year_eval,
    map_to_topstep_trade_day,
    score_policy_on_events,
)
from pipeline.feature.modules.loader import load_features_from_modules

# ── custom output dir (separate from baseline v6) ─────────────────────────────

TUNE_DIR = BASE / "model" / "SWEEP_v6" / "TUNE_PARAMS"
TUNE_DIR.mkdir(parents=True, exist_ok=True)

# ── custom param grid: higher rev_q / cont_q ──────────────────────────────────

def build_tune_grid() -> list[PolicyParams]:
    """Tighter probability grid — focus on selectivity.

    Rationale
    ---------
    Baseline best params use ``rev_q=0.6`` for most targets. This is too loose
    — it lets through low-confidence reversal trades, increasing trade frequency
    and fail-MLL risk. We test:

    - ``rev_q`` ∈ [0.75, 0.85, 0.90, 0.95] — higher = more selective
    - ``cont_q`` ∈ [0.75, 0.85, 0.90] — higher = fewer continuation trades
    - ``rev_adx_min`` ∈ [30, 40] — skip weak trend reversals
    - ``cont_adx_max`` ∈ [30, 100] — skip strong trend continuations (already tested)
    - ``risk_per_r_usd`` ∈ [100, 150] — moderate risk (avoids blowup)
    - ``daily_profit_cap_usd`` ∈ [0.0, 1400.0, 2000.0] — $2000 cap gives more room
    """
    params = []
    for rev_q in [0.75, 0.85, 0.90, 0.95]:
        for cont_q in [0.75, 0.85, 0.90]:
            for rev_adx_min in [30, 40]:
                for cont_adx_max in [30, 100]:
                    for risk in [100, 150]:
                        for profit_cap in [0.0, 1400.0, 2000.0]:
                            params.append(PolicyParams(
                                rev_q=rev_q, cont_q=cont_q,
                                rev_adx_min=rev_adx_min, cont_adx_max=cont_adx_max,
                                daily_stop_usd=0.0, daily_profit_cap_usd=profit_cap,
                                risk_per_r_usd=float(risk),
                            ))
    return params


PARAM_GRID = build_tune_grid()

print(f"[Tune] Grid size: {len(PARAM_GRID)} combinations")
print(f"[Tune] rev_q values: {sorted(set(p.rev_q for p in PARAM_GRID))}")
print(f"[Tune] cont_q values: {sorted(set(p.cont_q for p in PARAM_GRID))}")
print(f"[Tune] profit_cap values: {sorted(set(p.daily_profit_cap_usd for p in PARAM_GRID))}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  v6 Parameter Tuning — Higher rev_q / cont_q")
    print("=" * 65)

    # ── 1. Load datamart ────────────────────────────────────────────
    print("\n1. Loading datamart (core columns only)...")
    dm_path = BASE / "data" / "Level_2_Datamart" / "training_datamart_orb.parquet"
    dm = pd.read_parquet(dm_path, columns=CORE_COLS).copy()
    print(f"   Shape: {dm.shape}, Date: {dm['date'].min()} → {dm['date'].max()}")

    # ── 2. Load feature modules ─────────────────────────────────────
    modules_dir = BASE / "data" / "Level_1_Features" / "modules"
    print("\n2. Loading feature modules...")
    dm = load_features_from_modules(modules_dir, dm)
    print(f"   Shape after modules: {dm.shape}")

    dm["date"] = pd.to_datetime(dm["date"])
    dm["year"] = dm["date"].dt.year

    # ── 3. Derive feature list ──────────────────────────────────────
    excluded = set(EVENT_KEY) | {"year"}
    module_cols = sorted(set(dm.columns) - set(CORE_COLS) - excluded)
    all_features = CORE_FEATURES + module_cols
    print(f"\n3. Feature list ({len(all_features)} total):")
    print(f"   Core ({len(CORE_FEATURES)}): {CORE_FEATURES}")
    print(f"   Module ({len(module_cols)}): {module_cols}")

    # ── 4. Iterate over labels ──────────────────────────────────────
    summary_rows: list[dict] = []

    for target, rr in LABELS:
        print(f"\n{'='*60}")
        print(f"   Label: {target}  (RR={rr})")

        rev_df = dm[dm["side"] == "rev"].copy()
        cont_df = dm[dm["side"] == "cont"].copy()

        if target not in rev_df.columns or rev_df[target].isna().all():
            print(f"   SKIP — label missing or all-NaN")
            continue

        # ── Train models ────────────────────────────────────────────
        calib_from = pd.to_datetime("2025-07-01")
        calib_to = pd.to_datetime(HOLDOUT_FROM)

        # Build train mask on dm, then align to each side's index
        date_cutoff = pd.to_datetime(HOLDOUT_FROM)
        rev_train = rev_df[rev_df["date"] <= date_cutoff].copy()
        cont_train = cont_df[cont_df["date"] <= date_cutoff].copy()

        # ensure features are numeric
        feat_cols = [c for c in all_features if c in rev_train.columns]

        import lightgbm as lgb
        import numpy as np

        def _train_side(
            df: pd.DataFrame, side_name: str,
        ) -> lgb.Booster | None:
            train = df[df["date"] < calib_from].copy()
            val = df[(df["date"] >= calib_from) & (df["date"] <= calib_to)].copy()
            if len(train) < 100 or train[target].sum() < 5:
                print(f"   SKIP {side_name} — insufficient data")
                return None

            # Encode (one-hot orb_tf, session; bool breakout_side) then pick numeric cols
            train_enc = encode(train)
            val_enc = encode(val)
            num_feats = _numeric_features(train_enc, feat_cols)

            X_tr = train_enc[num_feats].fillna(0)
            y_tr = train_enc[target]
            w_tr = compute_weights(train_enc["year"])

            X_val = val_enc[num_feats].fillna(0)
            y_val = val_enc[target]
            w_val = compute_weights(val_enc["year"])

            model = lgb.train(
                LGBM_PARAMS,
                lgb.Dataset(X_tr, y_tr, weight=w_tr),
                num_boost_round=NUM_ROUNDS,
                valid_sets=[lgb.Dataset(X_val, y_val, weight=w_val)],
                callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(0)],
            )
            return model

        rev_model = _train_side(rev_train, "rev")
        cont_model = _train_side(cont_train, "cont")

        if rev_model is None and cont_model is None:
            print("   SKIP — neither model trained")
            continue

        # ── Build event frame ───────────────────────────────────────
        events = build_event_frame(
            rev_df, cont_df, rev_model, cont_model, feat_cols, target
        )
        holdout_events = events[events["date"] >= pd.to_datetime(HOLDOUT_FROM)].copy()
        print(f"   Holdout ({HOLDOUT_FROM}+): n={len(holdout_events)}")

        # ── Sweep ───────────────────────────────────────────────────
        best_score = -999.0
        best_params = None
        best_result = None

        # Build calib events (needed by score_policy_on_events)
        calib_events = events[
            (events["date"] >= pd.to_datetime(CALIB_FROM)) &
            (events["date"] <= pd.to_datetime(CALIB_TO))
        ].copy()

        for p in PARAM_GRID:
            try:
                result = score_policy_on_events(holdout_events, calib_events, p, rr)
                score = result["score"]
                if score > best_score:
                    best_score = score
                    best_params = p
                    best_result = result
            except Exception as e:
                print(f"   ⚠️  Param {p} failed: {e}")
                continue

        if best_result is None:
            print("   SKIP — no valid param combination")
            continue

        # ── Year-by-year eval (returns dict[str, dict]) ─────────────
        yr_res = by_year_eval(holdout_events, calib_events, best_params, rr)
        row = {
            "target": target,
            "rr": rr,
            "pass_rate": best_result["pass_rate"],
            "fail_mll_rate": best_result["fail_mll_rate"],
            "score": best_score,
            "median_end_pnl": best_result["median_end_pnl"],
            "avg_trades": best_result.get("avg_trades", 0),
            "windows": best_result.get("windows", 0),
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

        print(f"\n   → score={best_score:+.4f}, pass={best_result['pass_rate']*100:.1f}%, "
              f"fail_mll={best_result['fail_mll_rate']*100:.1f}%, pnl=${best_result['median_end_pnl']:.0f}")
        print(f"     params: rev_q={best_params.rev_q}, cont_q={best_params.cont_q}, "
              f"rev_adx={best_params.rev_adx_min}, cont_adx={best_params.cont_adx_max}, "
              f"risk=${best_params.risk_per_r_usd:.0f}, cap=${best_params.daily_profit_cap_usd:.0f}")
        for yr in sorted(yr_res.keys()):
            print(f"     {yr}: pass={yr_res[yr]['pass_rate']*100:.1f}%, "
                  f"fail_mll={yr_res[yr]['fail_mll_rate']*100:.1f}%, "
                  f"pnl=${yr_res[yr]['median_end_pnl']:.0f}")

    # ── 5. Results ──────────────────────────────────────────────────
    results_df = pd.DataFrame(summary_rows)

    # Sort by score descending
    results_df = results_df.sort_values("score", ascending=False).reset_index(drop=True)

    csv_path = TUNE_DIR / "TUNE_PARAMS_RESULTS.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")

    # ── 6. Report ───────────────────────────────────────────────────
    report_lines = [
        "# v6 Parameter Tuning — Higher rev_q / cont_q",
        "",
        "## Tuning Grid",
        "",
        f"| Parameter | Values |",
        f"|-----------|--------|",
        f"| rev_q | {sorted(set(p.rev_q for p in PARAM_GRID))} |",
        f"| cont_q | {sorted(set(p.cont_q for p in PARAM_GRID))} |",
        f"| rev_adx_min | {sorted(set(p.rev_adx_min for p in PARAM_GRID))} |",
        f"| cont_adx_max | {sorted(set(p.cont_adx_max for p in PARAM_GRID))} |",
        f"| risk_per_r_usd | {sorted(set(p.risk_per_r_usd for p in PARAM_GRID))} |",
        f"| daily_profit_cap | {sorted(set(p.daily_profit_cap_usd for p in PARAM_GRID))} |",
        "",
        f"Grid size: **{len(PARAM_GRID)}** combinations ({len(PARAM_GRID) * len(LABELS)} model trainings)",
        "",
        "## Ranked Results",
        "",
    ]

    cols = ["target", "rr", "score", "pass_rate", "fail_mll_rate", "median_end_pnl", "avg_trades", "windows"]
    report_lines.append("| " + " | ".join(cols) + " |")
    report_lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in results_df.iterrows():
        vals = [fmt(row[c]) if c in row.index else "-" for c in cols]
        report_lines.append("| " + " | ".join(vals) + " |")

    # Yearly pass rate table
    report_lines.append("\n## Yearly Pass Rate by Target\n")
    ycols = ["target", "pass_2025", "pass_2026", "pass_rate"]
    report_lines.append("| " + " | ".join(ycols) + " |")
    report_lines.append("|" + "|".join(["---"] * len(ycols)) + "|")
    for _, row in results_df.iterrows():
        vals = [str(row.get(c, "-")) if c == "target" else f"{row.get(c, 0)*100:.1f}%" for c in ycols]
        report_lines.append("| " + " | ".join(vals) + " |")

    # Yearly fail MLL table
    report_lines.append("\n## Yearly Fail MLL Rate by Target\n")
    mcols = ["target", "fail_mll_2025", "fail_mll_2026", "fail_mll_rate"]
    report_lines.append("| " + " | ".join(mcols) + " |")
    report_lines.append("|" + "|".join(["---"] * len(mcols)) + "|")
    for _, row in results_df.iterrows():
        vals = [str(row.get(c, "-")) if c == "target" else f"{row.get(c, 0)*100:.1f}%" for c in mcols]
        report_lines.append("| " + " | ".join(vals) + " |")

    # Best params table
    report_lines.append("\n## Best Params per Target\n")
    pcols = ["target", "rev_q", "cont_q", "rev_adx_min", "cont_adx_max", "risk_per_r_usd", "daily_profit_cap"]
    report_lines.append("| " + " | ".join(pcols) + " |")
    report_lines.append("|" + "|".join(["---"] * len(pcols)) + "|")
    for _, row in results_df.iterrows():
        vals = [str(row.get(c, "-")) if c == "target" else f"{row.get(c, 0):.3f}" for c in pcols]
        report_lines.append("| " + " | ".join(vals) + " |")

    # AB comparison with baseline v6
    report_lines.append("\n## AB Comparison vs Baseline v6 (3-Module, Default Grid)\n")
    report_lines.append("| Target | Metric | Baseline v6 | Tuned | Δ |")
    report_lines.append("|--------|--------|:-----------:|:-----:|:-:|")

    # Load baseline for comparison
    baseline_path = OUT_DIR / "OBJECTIVE_SWEEP_RESULTS.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        for _, row in results_df.iterrows():
            tgt = row["target"]
            b = baseline[baseline["target"] == tgt]
            if len(b) == 0:
                continue
            b = b.iloc[0]
            for metric, label in [("pass_rate", "Pass Rate"), ("fail_mll_rate", "Fail MLL"), ("score", "Score")]:
                bv = b[metric] if metric in b else 0
                tv = row[metric] if metric in row else 0
                delta = tv - bv
                delta_str = f"{delta*100:+.1f}pp" if metric != "score" else f"{delta:+.4f}"
                report_lines.append(f"| {tgt} | {label} | {bv*100 if metric != 'score' else bv:.4f} | {tv*100 if metric != 'score' else tv:.4f} | {delta_str} |")
            # Year-specific metrics — only columns ending with a 4-digit year
            year_cols = [c for c in row.index
                         if (c.startswith("pass_") or c.startswith("fail_mll_"))
                         and c.split("_")[-1].isdigit()]
            for yr_col in year_cols:
                if yr_col in b.index:
                    bv = b[yr_col]
                    tv = row[yr_col]
                    delta = tv - bv
                    yr = yr_col.split("_")[-1]
                    if int(yr) >= 2025:
                        label_name = yr_col.replace("_", " ")
                        report_lines.append(f"| {tgt} | {label_name} | {bv*100:.1f}% | {tv*100:.1f}% | {delta*100:+.1f}pp |")

    report_path = TUNE_DIR / "TUNE_PARAMS_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n✅ Report saved to {report_path}")

    # ── 7. Summary scoreboard ───────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Scoreboard (v6 Tuned — Higher rev_q)")
    print(f"{'='*65}")
    print(f"  {'Target':<20s} {'Score':>8s} {'Pass':>7s} {'FailMLL':>8s} {'PnL':>10s}")
    print(f"  {'-'*20} {'-'*8} {'-'*7} {'-'*8} {'-'*10}")
    for _, row in results_df.iterrows():
        print(f"  {row['target']:<20s} {row['score']:>+8.4f} {row['pass_rate']*100:>6.1f}% {row['fail_mll_rate']*100:>7.1f}% ${row['median_end_pnl']:>8.0f}")

    # Compare best target's 2026 pass rate vs baseline
    # Check which year columns exist (dynamic — depends on holdout data)
    year_cols = sorted([c for c in results_df.columns if c.startswith("pass_")])
    if year_cols:
        latest_year_col = year_cols[-1]  # e.g., "pass_2026"
        latest_year = latest_year_col.split("_")[-1]
        print(f"\n{'='*65}")
        print(f"  Best {latest_year} Pass Rate Comparison")
        print(f"{'='*65}")
        best_2026_tuned = results_df.loc[results_df[latest_year_col].idxmax()]
        if baseline_path.exists():
            baseline = pd.read_csv(baseline_path)
            if latest_year_col in baseline.columns:
                best_2026_base = baseline.loc[baseline[latest_year_col].idxmax()]
                print(f"  Baseline: {best_2026_base['target']} — {latest_year} pass = {best_2026_base[latest_year_col]*100:.1f}%")
                print(f"  Tuned:    {best_2026_tuned['target']} — {latest_year} pass = {best_2026_tuned[latest_year_col]*100:.1f}%")
                delta = best_2026_tuned[latest_year_col] - best_2026_base[latest_year_col]
                print(f"  Δ = {delta*100:+.1f}pp")
        print(f"\n  Tuned params: {best_2026_tuned['target']}, rev_q={best_2026_tuned['rev_q']}, "
              f"cont_q={best_2026_tuned['cont_q']}, cap=${best_2026_tuned['daily_profit_cap']:.0f}")


if __name__ == "__main__":
    main()

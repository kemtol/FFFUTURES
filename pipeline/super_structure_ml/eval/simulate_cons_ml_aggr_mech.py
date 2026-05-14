#!/usr/bin/env python3
"""
SMART_1 hybrid simulator: CONS ML (Meta-v7 Refined) + AGGR Mechanical (v1.12).

Apple-to-apple output with model/SUPER_STRUCTURE/simulation-compare/TEMP_SIM_*.json
for 7d/30d/90d window comparison against:
  - TEMP_SIM_META_V7_REFINED_MGC_*d.json  (CONS ML solo)
  - TEMP_SIM_SMART_1_MGC_*d.json          (CONS ML + AGGR ML, currently in live)
"""

import argparse
import json
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import pandas as pd


def _ct_trading_day(ts: pd.Timestamp):
    """Topstep trading day per pipeline/analysis/topstep_sim.py."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts_ct = ts.tz_convert("America/Chicago")
    return (ts_ct - pd.Timedelta(hours=15, minutes=10)).date()

ROOT = Path(__file__).resolve().parent.parent.parent.parent
FLIP_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
V1_12_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v1_12_training_datamart.parquet"
V7_MODEL = ROOT / "model/SUPER_STRUCTURE/meta_v7/inference_model.txt"
REFINED_CFG = ROOT / "model/SUPER_STRUCTURE/meta_v7/inference_config_refined.json"
REPORT_DIR = ROOT / "model/SUPER_STRUCTURE/simulation-compare"

V7_FEATS = ["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio",
            "volatility_zscore", "session_cluster"]


def load_cons_trades(use_refined_thresholds: bool) -> pd.DataFrame:
    df = pd.read_parquet(FLIP_PATH)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)

    brain = lgb.Booster(model_file=str(V7_MODEL))
    df["prob"] = brain.predict(df[V7_FEATS])

    if use_refined_thresholds:
        cfg = json.loads(REFINED_CFG.read_text())
        thr_map = {int(k): float(v) for k, v in cfg["thresholds"].items()}
        df["threshold"] = df["session_cluster"].map(thr_map)
    else:
        df["threshold"] = 0.50

    keep = df[df["prob"] >= df["threshold"]].copy()
    keep["mode"] = "MODE_CONSERVATIVE"
    return keep[["entry_ts", "side", "pnl_usd", "mode"]]


def load_aggr_trades(risk_cap: float) -> pd.DataFrame:
    df = pd.read_parquet(V1_12_PATH)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    keep = df[df["risk_pts"] <= risk_cap].copy()
    keep["mode"] = "MODE_AGGRESSIVE"
    return keep[["entry_ts", "side", "pnl_usd", "mode"]]


def simulate(window_df: pd.DataFrame, daily_cap: float) -> list[dict]:
    balance = 50_000.0
    peak = 50_000.0
    daily_pnl: dict = {}
    ledger: list[dict] = []
    trade_no = 0

    for _, row in window_df.iterrows():
        trade_no += 1
        d = _ct_trading_day(row["entry_ts"])
        daily_pnl.setdefault(d, 0.0)
        if daily_pnl[d] <= daily_cap:
            continue

        pnl = float(row["pnl_usd"])
        balance += pnl
        daily_pnl[d] += pnl
        if balance > peak:
            peak = balance
        mll_floor = peak - 2000.0

        ledger.append({
            "trade_no": trade_no,
            "entry_ts": row["entry_ts"].strftime("%Y-%m-%d %H:%M"),
            "side": row["side"],
            "mode": row["mode"],
            "pnl": round(pnl, 2),
            "balance": round(balance, 2),
            "mll_floor": round(mll_floor, 2),
            "drawdown": round(balance - peak, 2),
            "is_failed": bool(balance <= mll_floor),
        })

    return ledger


def summarize(ledger: list[dict], window_days: int) -> dict:
    if not ledger:
        return {"window_days": window_days, "trades": 0, "final_pnl": 0.0,
                "worst_dd": 0.0, "failed": False}
    final_pnl = ledger[-1]["balance"] - 50_000.0
    worst_dd = min(t["drawdown"] for t in ledger)
    failed = any(t["is_failed"] for t in ledger)
    return {
        "window_days": window_days,
        "trades": len(ledger),
        "final_pnl": round(final_pnl, 2),
        "worst_dd": round(worst_dd, 2),
        "failed": failed,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--risk-cap", type=float, default=12.0,
                   help="AGGR mechanical risk_pts <= cap (default 12)")
    p.add_argument("--daily-cap", type=float, default=-700.0,
                   help="Daily loss cap, USD (default -700)")
    p.add_argument("--use-refined-thresholds", action="store_true", default=True,
                   help="Use per-session thresholds from inference_config_refined.json")
    p.add_argument("--no-refined-thresholds", dest="use_refined_thresholds",
                   action="store_false")
    p.add_argument("--windows", type=int, nargs="+", default=[7, 30, 90])
    args = p.parse_args()

    print(f"[CONFIG] risk_cap={args.risk_cap}  daily_cap=${args.daily_cap:.0f}  "
          f"refined_thresholds={args.use_refined_thresholds}")

    cons = load_cons_trades(args.use_refined_thresholds)
    aggr = load_aggr_trades(args.risk_cap)

    print(f"[DATA] CONS trades: {len(cons)}  (range {cons['entry_ts'].min()} → {cons['entry_ts'].max()})")
    print(f"[DATA] AGGR trades: {len(aggr)}  (range {aggr['entry_ts'].min()} → {aggr['entry_ts'].max()})")

    combined = pd.concat([cons, aggr]).sort_values("entry_ts").reset_index(drop=True)
    last_ts = combined["entry_ts"].max()
    print(f"[DATA] Combined anchor (last_ts): {last_ts}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for w in args.windows:
        cutoff = last_ts - timedelta(days=w)
        window_df = combined[combined["entry_ts"] >= cutoff].copy()
        ledger = simulate(window_df, args.daily_cap)
        out_path = REPORT_DIR / f"SIM_CONS_ML_AGGR_MECH_MGC_{w}d.json"
        out_path.write_text(json.dumps(ledger, indent=2))
        s = summarize(ledger, w)
        summaries.append(s)
        print(f"[SAVED] {out_path.name}")

    print("\n=== SIMULATION SUMMARY ===")
    print(f"{'Window':<8} {'Trades':>8} {'Final PnL':>12} {'Worst DD':>12} {'Topstep':>10}")
    for s in summaries:
        verdict = "FAIL" if s["failed"] else ("PASS" if s["worst_dd"] > -2000 else "BUST")
        print(f"{s['window_days']}d{'':<6} {s['trades']:>8} "
              f"${s['final_pnl']:>10,.2f} ${s['worst_dd']:>10,.2f}  {verdict:>10}")


if __name__ == "__main__":
    main()

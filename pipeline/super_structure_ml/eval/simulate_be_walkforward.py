#!/usr/bin/env python3
"""
Walk-forward test for the auto-Breakeven rule on V8 router trades.

Same trade selection logic as `simulate_cons_ml_aggr_mech.py` (CONS ML
Meta-v7 Refined + AGGR mechanical v1.12 risk≤12), but with a per-trade
BE-replay layer:

  - For each accepted trade, replay 5m bars between entry_ts and exit_ts.
  - Track peak unrealized PnL (USD) bar-by-bar.
  - When peak first crosses BE_TRIGGER_USD, "arm" BE: SL moved to entry.
  - If a subsequent bar's adverse extreme touches entry, exit at BE
    (pnl = -commission).
  - Otherwise the trade keeps its original outcome.

Within-bar assumptions:
  - Worst-case ordering — for an armed BE, we assume the adverse extreme
    is reached AFTER the favorable one within the same bar. So a bar
    that both arms BE (favorable) and hits BE (adverse) records the BE
    exit. This is the most conservative (BE-friendly) ordering.

Outputs apple-to-apple comparison vs the no-BE baseline at 7d/30d/90d.

Usage:
  python3 pipeline/super_structure_ml/eval/simulate_be_walkforward.py
  python3 pipeline/super_structure_ml/eval/simulate_be_walkforward.py --triggers 100 150 200
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
FLIP_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
V1_12_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v1_12_training_datamart.parquet"
V7_MODEL = ROOT / "model/SUPER_STRUCTURE/meta_v7/inference_model.txt"
REFINED_CFG = ROOT / "model/SUPER_STRUCTURE/meta_v7/inference_config_refined.json"
RAW_5M_DB = ROOT / "data/Level_0_Raw/MGC_5m.db"
OUT_DIR = ROOT / "model/SUPER_STRUCTURE/simulation-compare"

V7_FEATS = ["entry_adx", "cci_abs", "st_gap_ratio", "efficiency_ratio",
            "volatility_zscore", "session_cluster"]

POINT_VALUE = 10.0          # MGC
COMMISSION_RT = 1.74
DAILY_CAP_USD = -700.0
RISK_CAP_PTS = 12.0


# ── Topstep CT trading day ────────────────────────────────────────────────


def _ct_trading_day(ts: pd.Timestamp):
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts_ct = ts.tz_convert("America/Chicago")
    return (ts_ct - pd.Timedelta(hours=15, minutes=10)).date()


# ── Load + filter candidates (mirrors simulate_cons_ml_aggr_mech.py) ──────


def load_cons_trades() -> pd.DataFrame:
    df = pd.read_parquet(FLIP_PATH)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    brain = lgb.Booster(model_file=str(V7_MODEL))
    df["prob"] = brain.predict(df[V7_FEATS])
    cfg = json.loads(REFINED_CFG.read_text())
    thr_map = {int(k): float(v) for k, v in cfg["thresholds"].items()}
    df["threshold"] = df["session_cluster"].map(thr_map)
    keep = df[df["prob"] >= df["threshold"]].copy()
    keep["mode"] = "MODE_CONSERVATIVE"
    return keep[["entry_ts", "exit_ts", "side", "entry_price",
                 "pnl_usd", "mode"]]


def load_aggr_trades() -> pd.DataFrame:
    df = pd.read_parquet(V1_12_PATH)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    keep = df[df["risk_pts"] <= RISK_CAP_PTS].copy()
    keep["mode"] = "MODE_AGGRESSIVE"
    return keep[["entry_ts", "exit_ts", "side", "entry_price",
                 "pnl_usd", "mode"]]


# ── 5m bar loader for BE replay ───────────────────────────────────────────


def load_5m_bars() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{RAW_5M_DB}?mode=ro", uri=True, timeout=30)
    df = pd.read_sql(
        "SELECT timestamp_utc, open, high, low, close "
        "FROM investing_ohlcv_5m ORDER BY timestamp_utc",
        con,
    )
    con.close()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.set_index("timestamp_utc")


# ── BE replay per trade ───────────────────────────────────────────────────


def replay_be(row, bars_5m: pd.DataFrame, be_trigger_usd: float) -> dict:
    """Apply auto-BE rule to a single trade. Returns dict with adjusted PnL
    and metadata.
    """
    entry = float(row["entry_price"])
    side = row["side"]
    original_pnl = float(row["pnl_usd"])
    entry_ts = row["entry_ts"]
    exit_ts = row["exit_ts"]

    if pd.isna(exit_ts) or exit_ts <= entry_ts:
        return {"pnl_usd": original_pnl, "be_armed": False, "be_hit": False,
                "exit_reason": "unchanged"}

    window = bars_5m[(bars_5m.index > entry_ts) & (bars_5m.index <= exit_ts)]
    if window.empty:
        return {"pnl_usd": original_pnl, "be_armed": False, "be_hit": False,
                "exit_reason": "no_intrabar_data"}

    be_armed = False
    for ts, bar in window.iterrows():
        h, l = float(bar["high"]), float(bar["low"])
        if side == "Long":
            unrealized_peak = (h - entry) * POINT_VALUE - COMMISSION_RT
            if not be_armed and unrealized_peak >= be_trigger_usd:
                be_armed = True
            # If armed, check adverse extreme this bar.
            if be_armed and l <= entry:
                return {"pnl_usd": -COMMISSION_RT, "be_armed": True,
                        "be_hit": True, "exit_reason": "BE",
                        "be_exit_ts": ts.isoformat()}
        else:  # Short
            unrealized_peak = (entry - l) * POINT_VALUE - COMMISSION_RT
            if not be_armed and unrealized_peak >= be_trigger_usd:
                be_armed = True
            if be_armed and h >= entry:
                return {"pnl_usd": -COMMISSION_RT, "be_armed": True,
                        "be_hit": True, "exit_reason": "BE",
                        "be_exit_ts": ts.isoformat()}

    # Trade completed without BE adverse hit. BE was either armed (and
    # held to profit) or never armed.
    return {"pnl_usd": original_pnl, "be_armed": be_armed, "be_hit": False,
            "exit_reason": "unchanged"}


# ── Portfolio simulation ──────────────────────────────────────────────────


def simulate_window(trades: pd.DataFrame, window_days: int,
                    daily_cap: float = DAILY_CAP_USD) -> dict:
    if trades.empty:
        return {"trades": 0, "final_pnl": 0.0, "worst_dd": 0.0,
                "failed": False, "be_armed": 0, "be_hit": 0}
    last_ts = trades["entry_ts"].max()
    cutoff = last_ts - timedelta(days=window_days)
    win = trades[trades["entry_ts"] >= cutoff].sort_values("entry_ts").reset_index(drop=True)
    balance = 50_000.0
    peak = 50_000.0
    worst_dd = 0.0
    daily_pnl: dict = {}
    failed = False
    taken = 0
    for _, row in win.iterrows():
        d = _ct_trading_day(row["entry_ts"])
        daily_pnl.setdefault(d, 0.0)
        if daily_pnl[d] <= daily_cap:
            continue
        pnl = float(row["pnl_usd"])
        balance += pnl
        daily_pnl[d] += pnl
        if balance > peak:
            peak = balance
        dd = balance - peak
        if dd < worst_dd:
            worst_dd = dd
        mll_floor = peak - 2000.0
        if balance <= mll_floor:
            failed = True
        taken += 1
    return {
        "trades": int(taken),
        "final_pnl": round(balance - 50_000.0, 2),
        "worst_dd": round(worst_dd, 2),
        "failed": bool(failed),
        "be_armed": int(win.get("be_armed", pd.Series([])).sum() if "be_armed" in win else 0),
        "be_hit": int(win.get("be_hit", pd.Series([])).sum() if "be_hit" in win else 0),
    }


# ── main ──────────────────────────────────────────────────────────────────


def apply_be(trades: pd.DataFrame, bars_5m: pd.DataFrame,
             be_trigger_usd: float) -> pd.DataFrame:
    if be_trigger_usd <= 0:
        out = trades.copy()
        out["be_armed"] = False
        out["be_hit"] = False
        return out
    rows = []
    for _, row in trades.iterrows():
        adj = replay_be(row, bars_5m, be_trigger_usd)
        new = row.to_dict()
        new["pnl_usd"] = adj["pnl_usd"]
        new["be_armed"] = adj["be_armed"]
        new["be_hit"] = adj["be_hit"]
        rows.append(new)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--triggers", type=float, nargs="+",
                   default=[0.0, 100.0, 150.0, 200.0],
                   help="BE trigger thresholds in USD (0=disabled baseline)")
    p.add_argument("--windows", type=int, nargs="+", default=[7, 30, 90])
    args = p.parse_args()

    print("Loading trade candidates...")
    cons = load_cons_trades()
    aggr = load_aggr_trades()
    print(f"  CONS (ML pass): {len(cons)}")
    print(f"  AGGR (risk<=12): {len(aggr)}")

    print("Loading 5m bars for BE replay...")
    bars_5m = load_5m_bars()
    print(f"  bars loaded: {len(bars_5m)} "
          f"({bars_5m.index.min()} → {bars_5m.index.max()})")

    rows = []
    for trigger in args.triggers:
        cons_adj = apply_be(cons, bars_5m, trigger)
        aggr_adj = apply_be(aggr, bars_5m, trigger)
        merged = pd.concat([cons_adj, aggr_adj], ignore_index=True)
        # Per-mode + combined window stats
        for window_days in args.windows:
            for label, df in (("ALL", merged),
                              ("CONS", cons_adj),
                              ("AGGR", aggr_adj)):
                stats = simulate_window(df, window_days)
                rows.append({
                    "trigger": trigger,
                    "window_days": window_days,
                    "subset": label,
                    **stats,
                })

    results = pd.DataFrame(rows)

    # Pretty print: per trigger × window, ALL subset
    print()
    print("=" * 80)
    print(f"{'Trigger':>10}  {'Window':>8}  {'Subset':>6}  "
          f"{'Trades':>7}  {'PnL':>10}  {'DD':>10}  {'BE arm':>7}  {'BE hit':>7}  {'Pass'}")
    print("=" * 80)
    for _, r in results.iterrows():
        verdict = "FAIL" if r["failed"] else ("PASS" if r["worst_dd"] > -2000 else "BUST")
        trig = f"${r['trigger']:.0f}" if r["trigger"] > 0 else "OFF"
        print(f"{trig:>10}  {r['window_days']}d{'':<5}  {r['subset']:>6}  "
              f"{r['trades']:>7}  ${r['final_pnl']:>8,.0f}  ${r['worst_dd']:>8,.0f}  "
              f"{r['be_armed']:>7}  {r['be_hit']:>7}  {verdict}")

    # Save raw output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "BE_WALKFORWARD_RESULTS.csv"
    results.to_csv(out_csv, index=False)
    print()
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()

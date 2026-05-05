#!/usr/bin/env python3
"""FVG Scalper parameter grid search.
Usage:  python3 pipeline/analysis/sweep_fvg.py [--phase 1|2]
"""
from __future__ import annotations

import csv, sys, time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.research.build_fvg_trade_events import (
    build_events, load_ohlcv_1m, resample_ohlcv, WARMUP_DAYS,
)

DATA_DB = ROOT / "data" / "Live" / "combined_buffer.db"
DATA_TABLE = "ohlcv_1m"
OUT_CSV = ROOT / "model" / "FVG_SWEEP" / "sweep_results.csv"

# Evaluation period
EVAL_START = "2026-04-28"
EVAL_END = "2026-05-05"
SWEEP_WARMUP = 30  # override WARMUP_DAYS for speed
WARMUP_DAYS = 120


def evaluate(events: pd.DataFrame) -> dict:
    """Compute metrics from trade events dataframe."""
    if events.empty:
        return {"trades": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0,
                "avg_pnl": 0.0, "avg_r": 0.0, "gross_profit": 0.0,
                "gross_loss": 0.0, "max_dd": 0.0}

    pnl = float(events["pnl_usd"].sum())
    wins = int(events["is_win"].sum())
    total = len(events)
    wr = wins / total * 100 if total > 0 else 0.0

    gross_profit = float(events[events["pnl_usd"] > 0]["pnl_usd"].sum()) if wins > 0 else 0.0
    gross_loss = abs(float(events[events["pnl_usd"] < 0]["pnl_usd"].sum())) if wins < total else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit > 0 and float("inf") or 0.0)

    avg_pnl = pnl / total if total > 0 else 0.0
    avg_r = float(events["r_multiple"].mean()) if total > 0 and not events["r_multiple"].isna().all() else 0.0

    # Max drawdown
    cum = events.sort_values("exit_ts")["pnl_usd"].cumsum()
    peak = cum.cummax()
    dd = (cum - peak).min()
    max_dd = float(abs(dd))

    return {
        "trades": total,
        "pnl": pnl,
        "wr": wr,
        "pf": pf,
        "avg_pnl": avg_pnl,
        "avg_r": avg_r,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "max_dd": max_dd,
        "score": pnl * wr / 100,  # compound: PnL weighted by win rate
    }


def build_grid(phase: int = 1) -> list[dict]:
    """Return parameter combinations for grid search."""
    if phase == 1:
        # Coarse sweep: 3×3×3 = 27 combos
        return [
            dict(zip(["MIN_GAP_PTS", "SL_LOOKBACK", "SESSION_MODE", "TP_RISK_RATIO"], v))
            for v in product(
                [1.0, 1.5, 2.0],
                [5, 10, 15],
                ["Off", "Asia + London", "Asia + London + NY"],
                [1.0],
            )
        ]
    else:
        # Fine sweep: zoom in on best region (last 7D for speed)
        return [
            dict(zip(["MIN_GAP_PTS", "SL_LOOKBACK", "SESSION_MODE", "TP_RISK_RATIO",
                      "COOLDOWN_BARS", "MIN_BODY_PCT"], v))
            for v in product(
                [1.0, 1.5, 2.0],
                [8, 10, 12],
                ["Asia + London"],
                [1.0],
                [3, 5, 8],
                [45, 50, 55],
            )
        ]


def run_sweep(grid: list[dict], df_bars: pd.DataFrame) -> list[dict]:
    """Run all parameter combos and return sorted results."""
    results = []
    total = len(grid)

    print(f"Loading data... {len(df_bars):,} 1m bars")
    print(f"Evaluating {total} parameter combos...")
    print()

    for idx, params in enumerate(grid):
        t0 = time.perf_counter()
        events = build_events(df_bars, EVAL_START, 1, params=params)
        metrics = evaluate(events)
        elapsed = time.perf_counter() - t0

        row = {**params, **metrics}
        results.append(row)

        pct = (idx + 1) / total * 100
        print(f"  [{idx+1:3d}/{total}] "
              f"GAP={params.get('MIN_GAP_PTS',1.0):.1f} "
              f"SL={params.get('SL_LOOKBACK',10):2d} "
              f"SESS={params.get('SESSION_MODE','Off'):>15s} "
              f"TP={params.get('TP_RISK_RATIO',1.0):.1f} "
              f"→ {metrics['trades']:4d}t {metrics['pnl']:+7.0f} "
              f"WR={metrics['wr']:.0f}% "
              f"{elapsed:.1f}s ({pct:.0f}%)", flush=True)

    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def save_results(results: list[dict], path: Path) -> None:
    """Save sweep results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(results[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {len(results)} results → {path}")


def print_top(results: list[dict], n: int = 10) -> None:
    """Print top-N results."""
    print(f"\n{'='*90}")
    print(f"TOP {n} PARAMETER COMBINATIONS (by score = PnL × WR)")
    print(f"{'='*90}")
    header = f"{'#':>3s} {'GAP':>4s} {'SL':>3s} {'SESSION':>15s} {'TP':>4s} {'Trd':>5s} {'PnL':>8s} {'WR':>5s} {'PF':>6s} {'AvgPnl':>7s} {'MaxDD':>7s} {'Score':>8s}"
    print(header)
    print("-" * 90)
    for i, r in enumerate(results[:n]):
        print(f"{i+1:3d} {r['MIN_GAP_PTS']:4.1f} {r['SL_LOOKBACK']:3d} "
              f"{r['SESSION_MODE']:>15s} {r['TP_RISK_RATIO']:4.1f} "
              f"{r['trades']:5d} {r['pnl']:+8.0f} {r['wr']:4.0f}% "
              f"{r['pf']:6.2f} {r['avg_pnl']:+7.1f} {r['max_dd']:+7.0f} "
              f"{r['score']:8.0f}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2],
                    help="Phase 1=coarse(18 combos), Phase 2=fine(27 combos)")
    args = ap.parse_args()

    grid = build_grid(args.phase)
    print(f"Phase {args.phase}: {len(grid)} combos")
    print(f"Data: {DATA_DB}")
    print(f"Period: {EVAL_START} → {EVAL_END}")
    print(f"TP_RISK_RATIO fixed at 1.0")

    warmup_start = (pd.Timestamp(EVAL_START, tz="UTC") - pd.Timedelta(days=WARMUP_DAYS))
    warmup_str = warmup_start.strftime("%Y-%m-%d %H:%M:%S")
    end_ts = pd.Timestamp(EVAL_END, tz="UTC").strftime("%Y-%m-%d %H:%M:%S")

    df_1m = load_ohlcv_1m(DATA_DB, EVAL_START, EVAL_END, DATA_TABLE)
    df_bars = df_1m.set_index("timestamp_utc").sort_index()

    results = run_sweep(grid, df_bars)
    save_results(results, OUT_CSV)
    print_top(results, 10)


if __name__ == "__main__":
    main()

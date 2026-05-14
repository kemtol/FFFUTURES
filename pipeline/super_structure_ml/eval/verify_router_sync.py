#!/usr/bin/env python3
"""
Verify that pipeline/live/inference_router.py produces trades identical to
pipeline/super_structure_ml/eval/simulate_cons_ml_aggr_mech.py for the same
event sources.

Strategy:
  1. Replay the SAME event sources used by the sim (v3_final_training.parquet
     for CONS, v1_12_training_datamart.parquet for AGGR).
  2. Apply stateless router gates (CONS ML threshold + AGGR risk cap) to get
     candidate trades — matches sim's pre-filter ordering.
  3. Window-slice from the same anchor as sim (combined last_ts).
  4. Stateful pass: route through InferenceRouter for daily cap + single-queue
     position checks.
  5. Compare resulting trades against SIM_CONS_ML_AGGR_MECH_MGC_*d.json.

Pass criteria: zero-diff on trade entries (count, ts, side, mode, pnl) per
window.
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.live.inference_router import InferenceRouter, CONS_FEATURES
from pipeline.live.pullback_detector import PullbackEvent

FLIP_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v3_final_training.parquet"
V1_12_PATH = ROOT / "data/Level_2_Datamart/super_structure_ml/v1_12_training_datamart.parquet"
SIM_DIR = ROOT / "model/SUPER_STRUCTURE/simulation-compare"
OUT_DIR = ROOT / "model/SUPER_STRUCTURE/simulation-compare"


def _prefilter_cons(router: InferenceRouter) -> pd.DataFrame:
    df = pd.read_parquet(FLIP_PATH)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    x = df[list(CONS_FEATURES)].to_numpy(dtype=float)
    df["prob"] = router.cons_brain.predict(x)
    df["threshold"] = df["session_cluster"].map(router._threshold_map)
    keep = df[df["prob"] >= df["threshold"]].copy()
    keep["source"] = "CONS"
    return keep[["entry_ts", "side", "pnl_usd", "source"]]


def _prefilter_aggr(router: InferenceRouter) -> pd.DataFrame:
    df = pd.read_parquet(V1_12_PATH)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    keep = df[df["risk_pts"] <= router.risk_cap_pts].copy()
    keep["source"] = "AGGR"
    return keep[["entry_ts", "side", "pnl_usd", "source"]]


def replay(window_days: int) -> list[dict]:
    router = InferenceRouter()
    cons = _prefilter_cons(router)
    aggr = _prefilter_aggr(router)
    merged = pd.concat([cons, aggr], ignore_index=True).sort_values("entry_ts")
    last_ts = merged["entry_ts"].max()
    cutoff = last_ts - timedelta(days=window_days)
    window_df = merged[merged["entry_ts"] >= cutoff].reset_index(drop=True)

    balance = 50_000.0
    peak = 50_000.0
    ledger: list[dict] = []
    trade_no = 0

    for _, row in window_df.iterrows():
        ts = row["entry_ts"]
        trade_no += 1

        # Stateful gates only (stateless filters already applied above).
        if router.current_position_mode() is not None:
            continue
        if router.daily_pnl(ts) <= router.daily_cap_usd:
            continue

        mode = "CONS" if row["source"] == "CONS" else "AGGR"
        router.on_entry(mode)
        pnl = float(row["pnl_usd"])
        balance += pnl
        if balance > peak:
            peak = balance
        mll_floor = peak - 2000.0
        mode_label = "MODE_CONSERVATIVE" if mode == "CONS" else "MODE_AGGRESSIVE"
        ledger.append({
            "trade_no": trade_no,
            "entry_ts": ts.strftime("%Y-%m-%d %H:%M"),
            "side": row["side"],
            "mode": mode_label,
            "pnl": round(pnl, 2),
            "balance": round(balance, 2),
            "mll_floor": round(mll_floor, 2),
            "drawdown": round(balance - peak, 2),
            "is_failed": bool(balance <= mll_floor),
        })
        router.on_exit(ts, pnl)

    return ledger


def compare_with_sim(window_days: int, replay_ledger: list[dict]) -> dict:
    sim_path = SIM_DIR / f"SIM_CONS_ML_AGGR_MECH_MGC_{window_days}d.json"
    sim_ledger = json.loads(sim_path.read_text())

    def fingerprint(t: dict) -> tuple:
        return (t["entry_ts"], t["side"], t["mode"], round(t["pnl"], 2))

    sim_fps = [fingerprint(t) for t in sim_ledger]
    replay_fps = [fingerprint(t) for t in replay_ledger]

    return {
        "window_days": window_days,
        "sim_count": len(sim_fps),
        "replay_count": len(replay_fps),
        "in_sim_only": sorted(set(sim_fps) - set(replay_fps)),
        "in_replay_only": sorted(set(replay_fps) - set(sim_fps)),
        "match": sim_fps == replay_fps,
    }


def main() -> None:
    results = []
    for w in (7, 30, 90):
        replay_ledger = replay(w)
        out = OUT_DIR / f"REPLAY_ROUTER_MGC_{w}d.json"
        out.write_text(json.dumps(replay_ledger, indent=2))
        diff = compare_with_sim(w, replay_ledger)
        results.append(diff)
        print(f"[SAVED] {out.name}")

    print("\n=== SYNC VERIFICATION ===")
    print(f"{'Window':<8} {'Sim':>6} {'Replay':>8} {'Match':>8}  Diff (sim-only / replay-only)")
    all_ok = True
    for r in results:
        marker = "OK ✅" if r["match"] else "FAIL ❌"
        if not r["match"]:
            all_ok = False
        print(f"{r['window_days']}d{'':<6} {r['sim_count']:>6} {r['replay_count']:>8} "
              f"{marker:>8}   {len(r['in_sim_only'])}/{len(r['in_replay_only'])}")
        if not r["match"]:
            for fp in r["in_sim_only"][:5]:
                print(f"    sim-only:    {fp}")
            for fp in r["in_replay_only"][:5]:
                print(f"    replay-only: {fp}")

    print()
    print("RESULT:", "ALL MATCH ✅" if all_ok else "DIVERGENCE DETECTED ❌")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

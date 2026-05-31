#!/usr/bin/env python3
"""Quick P2 sweep for M1 SuperTrend pullback candidate/exit parameters.

This is a raw mechanical sweep. It does not train ML models and does not write
to live paths. The goal is to find parameter sets with a better raw candidate
distribution before asking the classifier to filter trades.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import itertools
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_m1_events import ROOT, build_events, load_config, load_l1_context, topstep_trade_day  # noqa: E402


MODEL_DIR_NAME = "model/SUPER_STRUCTURE/m1_st_pullback_scalper"


def timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def param_grid() -> list[dict[str, Any]]:
    values = {
        "min_adx": [12.0, 16.0, 20.0],
        "pullback_band_atr": [0.10, 0.20, 0.30],
        "min_pullback_band_pts": [0.20, 0.30],
        "use_dema100_trend_filter": [True],
        "min_risk_pts": [0.30, 0.40],
        "max_risk_pts": [6.0, 8.0, 12.0],
        "st_buffer_pts": [0.40, 0.60, 0.80],
        "rr_target": [0.80, 1.00, 1.15],
        "max_hold_bars": [30, 60],
    }
    keys = list(values)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(values[k] for k in keys))]
    return combos


def params_id(params: dict[str, Any]) -> str:
    encoded = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha1(encoded).hexdigest()[:10]


def apply_params(cfg: dict, params: dict[str, Any]) -> dict:
    out = deepcopy(cfg)
    candidate_keys = {
        "min_adx",
        "pullback_band_atr",
        "min_pullback_band_pts",
        "use_dema100_trend_filter",
        "min_risk_pts",
        "max_risk_pts",
    }
    exit_keys = {"st_buffer_pts", "rr_target", "max_hold_bars"}
    for key, value in params.items():
        if key in candidate_keys:
            out["candidate_rules"][key] = value
        elif key in exit_keys:
            out["exit_rules"][key] = value
        else:
            raise ValueError(f"Unknown sweep param: {key}")
    return out


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    curve = pnl.cumsum()
    return float((curve - curve.cummax()).min())


def period_frame(events: pd.DataFrame, start: str, end: str | None = None) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    mask = events["signal_ts"] >= timestamp(start)
    if end:
        mask &= events["signal_ts"] < timestamp(end)
    return events[mask].copy()


def metrics(events: pd.DataFrame, start: str, end: str | None = None) -> dict[str, Any]:
    frame = period_frame(events, start, end)
    if frame.empty:
        return {
            "events": 0,
            "trade_days": 0,
            "events_per_day": 0.0,
            "pnl_usd": 0.0,
            "avg_trade": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "long_events": 0,
            "short_events": 0,
        }
    trade_days = max(int(frame["trade_day"].nunique()), 1)
    return {
        "events": int(len(frame)),
        "trade_days": trade_days,
        "events_per_day": float(len(frame) / trade_days),
        "pnl_usd": float(frame["pnl_usd"].sum()),
        "avg_trade": float(frame["pnl_usd"].mean()),
        "win_rate": float((frame["pnl_usd"] > 0).mean()),
        "max_drawdown": max_drawdown(frame["pnl_usd"]),
        "long_events": int((frame["side"] == "Long").sum()),
        "short_events": int((frame["side"] == "Short").sum()),
    }


def frequency_penalty(events_per_day: float, low: float = 6.0, high: float = 30.0) -> float:
    if events_per_day < low:
        return (low - events_per_day) * 600.0
    if events_per_day > high:
        return (events_per_day - high) * 250.0
    return 0.0


def score(row: dict[str, Any]) -> float:
    y2024 = row["metrics_2024_now"]
    y2026 = row["metrics_2026"]
    recent = row["metrics_2026_apr_may"]
    side_penalty = 0.0
    if recent["long_events"] == 0 or recent["short_events"] == 0:
        side_penalty += 500.0
    return float(
        y2026["pnl_usd"]
        + recent["pnl_usd"] * 1.5
        + y2026["avg_trade"] * 500.0
        + recent["avg_trade"] * 500.0
        - abs(y2026["max_drawdown"]) * 0.20
        - abs(recent["max_drawdown"]) * 0.30
        - frequency_penalty(y2024["events_per_day"])
        - frequency_penalty(y2026["events_per_day"])
        - side_penalty
    )


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "params_id": row["params_id"],
        "score": row["score"],
    }
    flat.update(row["params"])
    for label in ["2024_now", "2026", "2026_apr_may"]:
        for key, value in row[f"metrics_{label}"].items():
            flat[f"{label}_{key}"] = value
    return flat


def build_with_params(l1: pd.DataFrame, cfg: dict, params: dict[str, Any]) -> pd.DataFrame:
    local_cfg = apply_params(cfg, params)
    with contextlib.redirect_stdout(io.StringIO()):
        events = build_events(l1, local_cfg)
    if events.empty:
        return events
    events["signal_ts"] = pd.to_datetime(events["signal_ts"], utc=True)
    events["trade_day"] = topstep_trade_day(events["signal_ts"])
    return events.sort_values("signal_ts").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date")
    parser.add_argument("--max-combos", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()

    cfg = load_config()
    out_dir = ROOT / MODEL_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    l1 = load_l1_context(cfg, args.start_date, args.end_date)
    print(f"Loaded L1 rows: {len(l1):,} | {l1['timestamp_utc'].min()} -> {l1['timestamp_utc'].max()}", flush=True)

    baseline_params = {
        "min_adx": cfg["candidate_rules"]["min_adx"],
        "pullback_band_atr": cfg["candidate_rules"]["pullback_band_atr"],
        "min_pullback_band_pts": cfg["candidate_rules"]["min_pullback_band_pts"],
        "use_dema100_trend_filter": cfg["candidate_rules"]["use_dema100_trend_filter"],
        "min_risk_pts": cfg["candidate_rules"]["min_risk_pts"],
        "max_risk_pts": cfg["candidate_rules"]["max_risk_pts"],
        "st_buffer_pts": cfg["exit_rules"]["st_buffer_pts"],
        "rr_target": cfg["exit_rules"]["rr_target"],
        "max_hold_bars": cfg["exit_rules"]["max_hold_bars"],
    }
    combos = param_grid()
    remaining = [c for c in combos if c != baseline_params]
    random.Random(args.seed).shuffle(remaining)
    selected_combos = [baseline_params] + remaining[: max(args.max_combos - 1, 0)]
    print(f"Sweeping {len(selected_combos):,} / {len(combos):,} parameter sets", flush=True)

    rows = []
    for i, params in enumerate(selected_combos, start=1):
        try:
            events = build_with_params(l1, cfg, params)
            row = {
                "params_id": params_id(params),
                "params": params,
                "metrics_2024_now": metrics(events, "2024-01-01"),
                "metrics_2026": metrics(events, "2026-01-01"),
                "metrics_2026_apr_may": metrics(events, "2026-04-01"),
            }
            row["score"] = score(row)
            rows.append(row)
        except Exception as exc:  # keep sweep moving and preserve failure detail
            rows.append(
                {
                    "params_id": params_id(params),
                    "params": params,
                    "error": f"{type(exc).__name__}: {exc}",
                    "score": -1_000_000_000.0,
                    "metrics_2024_now": metrics(pd.DataFrame(), "2024-01-01"),
                    "metrics_2026": metrics(pd.DataFrame(), "2026-01-01"),
                    "metrics_2026_apr_may": metrics(pd.DataFrame(), "2026-04-01"),
                }
            )
        if i == 1 or i % 25 == 0 or i == len(selected_combos):
            best = max(rows, key=lambda r: r["score"])
            print(
                f"{i:>4}/{len(selected_combos)} best={best['params_id']} "
                f"score={best['score']:.2f} "
                f"2026_avg={best['metrics_2026']['avg_trade']:.2f} "
                f"2026_epd={best['metrics_2026']['events_per_day']:.2f}",
                flush=True,
            )

    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)
    flat = pd.DataFrame([flatten(r) for r in ranked])
    csv_path = out_dir / "candidate_param_sweep_quick.csv"
    json_path = out_dir / "candidate_param_sweep_quick.json"
    flat.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "start_date": args.start_date,
                "end_date": args.end_date,
                "max_combos": args.max_combos,
                "seed": args.seed,
                "total_grid_size": len(combos),
                "swept": len(selected_combos),
                "top": ranked[: args.top_n],
            },
            indent=2,
            default=str,
        )
        + "\n"
    )

    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    print("Top results:", flush=True)
    cols = [
        "params_id",
        "score",
        "min_adx",
        "pullback_band_atr",
        "min_pullback_band_pts",
        "min_risk_pts",
        "max_risk_pts",
        "st_buffer_pts",
        "rr_target",
        "max_hold_bars",
        "2026_events_per_day",
        "2026_avg_trade",
        "2026_pnl_usd",
        "2026_apr_may_events_per_day",
        "2026_apr_may_avg_trade",
        "2026_apr_may_pnl_usd",
    ]
    print(flat[cols].head(args.top_n).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

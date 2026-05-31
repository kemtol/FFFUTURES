#!/usr/bin/env python3
"""Build ORB volatility-targeted time-exit trade events from MNQ M1 context."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from math import floor

import pandas as pd

from common import assert_mnq_namespaces, load_config, project_path, write_json

REQUIRED_L2_COLUMNS = [
    "event_id",
    "ny_date",
    "signal_ts",
    "entry_ts",
    "exit_ts",
    "side",
    "orb_high",
    "orb_low",
    "orb_range_pts",
    "signal_close",
    "entry_price",
    "sl_price",
    "entry_risk_pts",
    "exit_price",
    "exit_reason",
    "hold_bars",
    "r_multiple",
    "pnl_per_contract_usd",
    "contracts_float",
    "contracts_floor",
    "contracts_used",
    "pnl_vol_target_usd",
    "label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def simulate_time_exit(day: pd.DataFrame, entry_idx: int, time_exit: str) -> tuple[pd.Timestamp, float, str, int]:
    for idx in range(entry_idx, len(day)):
        row = day.iloc[idx]
        if bool(row["bar_data_quality_ok"]) and row["ny_time"] >= time_exit:
            return row["timestamp_utc"], float(row["close"]), "TIME_EXIT", int(idx - entry_idx + 1)
    row = day.iloc[-1]
    return row["timestamp_utc"], float(row["close"]), "DATA_END", int(len(day) - entry_idx)


def build_events(l1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rules = cfg["rules"]
    costs = cfg["costs"]
    sizing = cfg["position_sizing"]
    orb_end = cfg["session"]["orb_end"]
    time_exit = cfg["session"]["time_exit"]

    point_value = float(costs["point_value_usd"])
    slippage_pts = float(costs["slippage_ticks_per_side"]) * float(costs["tick_size"])
    commission = float(costs["commission_round_turn_usd"])
    target_risk = float(sizing["target_risk_usd"])
    max_contracts = int(sizing["max_contracts"])
    min_contracts = int(sizing["min_contracts"])

    rows = []
    l1 = l1.sort_values("timestamp_utc").reset_index(drop=True)
    for ny_date, day in l1.groupby("ny_date", sort=True):
        day = day.reset_index(drop=True)
        if not bool(day["orb_complete"].any()):
            continue
        orb_high = float(day["orb_high"].dropna().iloc[0])
        orb_low = float(day["orb_low"].dropna().iloc[0])
        orb_range = float(orb_high - orb_low)
        if orb_range < float(rules["min_orb_range_pts"]) or orb_range > float(rules["max_orb_range_pts"]):
            continue

        candidates = day[
            day["eligible_after_or"]
            & (day["ny_time"] > orb_end)
            & (day["ny_time"] < time_exit)
            & (day["close"] > orb_high)
        ]
        if candidates.empty:
            continue

        signal_idx = int(candidates.index[0])
        entry_idx = signal_idx + 1
        if entry_idx >= len(day):
            continue
        entry_bar = day.iloc[entry_idx]
        if not bool(entry_bar["bar_data_quality_ok"]) or entry_bar["ny_time"] > time_exit:
            continue

        signal = day.iloc[signal_idx]
        entry_price = float(entry_bar["open"]) + slippage_pts
        sl_price = orb_low
        entry_risk_pts = entry_price - sl_price
        if entry_risk_pts <= 0:
            continue
        if entry_risk_pts < float(rules["min_entry_risk_pts"]) or entry_risk_pts > float(rules["max_entry_risk_pts"]):
            continue
        exit_ts, raw_exit_price, exit_reason, hold_bars = simulate_time_exit(day, entry_idx, time_exit)
        if exit_reason != "TIME_EXIT":
            continue
        exit_price = raw_exit_price - slippage_pts
        gross_pts = exit_price - entry_price
        pnl_per_contract = gross_pts * point_value - commission
        r_multiple = gross_pts / entry_risk_pts
        risk_per_contract_usd = entry_risk_pts * point_value
        contracts_float = target_risk / risk_per_contract_usd if risk_per_contract_usd > 0 else 0.0
        contracts_floor = floor(contracts_float)
        contracts_used = max(0, min(max_contracts, contracts_floor))
        if contracts_used < min_contracts:
            continue
        pnl_vol_target = pnl_per_contract * contracts_used

        rows.append(
            {
                "event_id": f"MNQ_ORB_{ny_date}",
                "ny_date": ny_date,
                "signal_ts": signal["timestamp_utc"],
                "entry_ts": entry_bar["timestamp_utc"],
                "exit_ts": exit_ts,
                "side": "LONG",
                "orb_high": orb_high,
                "orb_low": orb_low,
                "orb_range_pts": orb_range,
                "orb_bar_count": int(signal["orb_bar_count"]),
                "signal_close": float(signal["close"]),
                "signal_volume": float(signal["volume"]),
                "signal_minutes_from_open": int(signal["minutes_from_open"]),
                "entry_open": float(entry_bar["open"]),
                "entry_price": entry_price,
                "sl_price": sl_price,
                "entry_risk_pts": entry_risk_pts,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "hold_bars": hold_bars,
                "r_multiple": r_multiple,
                "pnl_per_contract_usd": pnl_per_contract,
                "contracts_float": contracts_float,
                "contracts_floor": contracts_floor,
                "contracts_used": contracts_used,
                "pnl_vol_target_usd": pnl_vol_target,
                "label": int(pnl_per_contract > 0),
            }
        )
    return pd.DataFrame(rows)


def validate_events(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_L2_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing L2 columns: {missing}")
    if df.empty:
        return
    for col in ["signal_ts", "entry_ts", "exit_ts"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    if (df["entry_ts"] <= df["signal_ts"]).any():
        raise SystemExit("entry_ts must be after signal_ts")
    if (df["exit_ts"] < df["entry_ts"]).any():
        raise SystemExit("exit_ts must be >= entry_ts")
    nulls = df[REQUIRED_L2_COLUMNS].isna().sum()
    bad = nulls[nulls > 0]
    if not bad.empty:
        raise SystemExit(f"L2 required column nulls: {bad.to_dict()}")
    dupes = df["ny_date"].duplicated().sum()
    if dupes:
        raise SystemExit(f"More than one event per NY date: {dupes}")


def main() -> int:
    args = parse_args()
    cfg = load_config()
    assert_mnq_namespaces(cfg)
    l1_path = project_path(cfg["outputs"]["l1_context"])
    out_path = project_path(cfg["outputs"]["events"])
    manifest_path = project_path(cfg["outputs"]["events_manifest"])
    if not l1_path.exists():
        raise SystemExit(f"Missing L1 context: {l1_path}")
    if out_path.exists() and not args.force and not args.dry_run:
        raise SystemExit(f"Output exists; use --force: {out_path}")

    l1 = pd.read_parquet(l1_path)
    l1["timestamp_utc"] = pd.to_datetime(l1["timestamp_utc"], utc=True)
    events = build_events(l1, cfg)
    validate_events(events)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(events)),
        "min_signal_ts": events["signal_ts"].min().isoformat() if not events.empty else None,
        "max_signal_ts": events["signal_ts"].max().isoformat() if not events.empty else None,
        "win_rate": float(events["label"].mean()) if not events.empty else 0.0,
        "avg_pnl_per_contract_usd": float(events["pnl_per_contract_usd"].mean()) if not events.empty else 0.0,
        "total_pnl_vol_target_usd": float(events["pnl_vol_target_usd"].sum()) if not events.empty else 0.0,
        "avg_contracts_used": float(events["contracts_used"].mean()) if not events.empty else 0.0,
        "output": str(out_path),
    }
    print(manifest)
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(out_path, index=False)
        write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())

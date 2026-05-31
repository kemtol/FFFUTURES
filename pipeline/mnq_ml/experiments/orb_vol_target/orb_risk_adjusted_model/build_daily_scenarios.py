#!/usr/bin/env python3
"""Build daily MNQ ORB scenario labels for probability modeling.

One row is one NY trading day after the 15m opening range is complete. Features
are known at OR completion; labels describe what later happened if price broke
above or below the OR before the 15:00 NY time exit.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from common import load_config as load_parent_config  # noqa: E402
from common import project_path, write_json  # noqa: E402

CONFIG_PATH = SCRIPT_DIR / "config.json"

FEATURE_COLUMNS = [
    "orb_range_pts",
    "orb_body_pts",
    "orb_direction_pts",
    "orb_close_position",
    "orb_upper_wick_pts",
    "orb_lower_wick_pts",
    "orb_volume_sum",
    "orb_volume_mean",
    "orb_volume_max",
    "pre_60m_bar_count",
    "pre_60m_return_pts",
    "pre_60m_range_pts",
    "pre_60m_volume_sum",
    "ny_day_of_week",
    "ny_month",
]

LABEL_COLUMNS = [
    "up_breakout_occurred",
    "down_breakout_occurred",
    "first_breakout_side",
    "first_breakout_risk_label",
    "up_success_2r",
    "down_success_2r",
    "up_pnl_per_contract_usd",
    "down_pnl_per_contract_usd",
    "up_r_multiple",
    "down_r_multiple",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def hhmm_to_minutes(value: str) -> int:
    return int(value[:2]) * 60 + int(value[3:])


def assign_split(ny_date: object, split_cfg: dict[str, str]) -> str:
    date_value = pd.Timestamp(ny_date).date()
    train_end = pd.Timestamp(split_cfg["train_end"]).date()
    validation_start = pd.Timestamp(split_cfg["validation_start"]).date()
    validation_end = pd.Timestamp(split_cfg["validation_end"]).date()
    holdout_start = pd.Timestamp(split_cfg["holdout_start"]).date()
    if date_value <= train_end:
        return "train"
    if validation_start <= date_value <= validation_end:
        return "validation"
    if date_value >= holdout_start:
        return "holdout"
    return "unused"


def max_drawdown(pnls: pd.Series) -> float:
    if pnls.empty:
        return 0.0
    equity = pnls.cumsum()
    equity = pd.concat([pd.Series([0.0]), equity], ignore_index=True)
    drawdown = equity - equity.cummax()
    return float(drawdown.min())


def is_success(value: Any) -> bool:
    return pd.notna(value) and int(value) == 1


def simulate_side(
    day: pd.DataFrame,
    side: str,
    signal_idx: int | None,
    orb_high: float,
    orb_low: float,
    parent_cfg: dict[str, Any],
    scenario_cfg: dict[str, Any],
) -> dict[str, Any]:
    if signal_idx is None:
        return {
            f"{side.lower()}_breakout_occurred": False,
            f"{side.lower()}_signal_ts": pd.NaT,
            f"{side.lower()}_entry_ts": pd.NaT,
            f"{side.lower()}_exit_ts": pd.NaT,
            f"{side.lower()}_exit_reason": "NO_BREAKOUT",
            f"{side.lower()}_success_2r": pd.NA,
            f"{side.lower()}_pnl_per_contract_usd": pd.NA,
            f"{side.lower()}_r_multiple": pd.NA,
        }

    costs = parent_cfg["costs"]
    rules = parent_cfg["rules"]
    time_exit = scenario_cfg["scenario_contract"]["time_exit"]
    tp_r = float(scenario_cfg["scenario_contract"]["tp_r"])
    point_value = float(costs["point_value_usd"])
    slippage_pts = float(costs["slippage_ticks_per_side"]) * float(costs["tick_size"])
    commission = float(costs["commission_round_turn_usd"])
    min_entry_risk = float(rules["min_entry_risk_pts"])
    max_entry_risk = float(rules["max_entry_risk_pts"])

    entry_idx = signal_idx + 1
    if entry_idx >= len(day):
        return {
            f"{side.lower()}_breakout_occurred": True,
            f"{side.lower()}_signal_ts": day.iloc[signal_idx]["timestamp_utc"],
            f"{side.lower()}_entry_ts": pd.NaT,
            f"{side.lower()}_exit_ts": pd.NaT,
            f"{side.lower()}_exit_reason": "NO_NEXT_BAR",
            f"{side.lower()}_success_2r": pd.NA,
            f"{side.lower()}_pnl_per_contract_usd": pd.NA,
            f"{side.lower()}_r_multiple": pd.NA,
        }

    entry_bar = day.iloc[entry_idx]
    if (not bool(entry_bar["bar_data_quality_ok"])) or entry_bar["ny_time"] > time_exit:
        return {
            f"{side.lower()}_breakout_occurred": True,
            f"{side.lower()}_signal_ts": day.iloc[signal_idx]["timestamp_utc"],
            f"{side.lower()}_entry_ts": entry_bar["timestamp_utc"],
            f"{side.lower()}_exit_ts": pd.NaT,
            f"{side.lower()}_exit_reason": "BAD_ENTRY_BAR",
            f"{side.lower()}_success_2r": pd.NA,
            f"{side.lower()}_pnl_per_contract_usd": pd.NA,
            f"{side.lower()}_r_multiple": pd.NA,
        }

    if side == "UP":
        entry_price = float(entry_bar["open"]) + slippage_pts
        entry_risk_pts = entry_price - orb_low
        tp_price = entry_price + tp_r * entry_risk_pts
    else:
        entry_price = float(entry_bar["open"]) - slippage_pts
        entry_risk_pts = orb_high - entry_price
        tp_price = entry_price - tp_r * entry_risk_pts

    if entry_risk_pts < min_entry_risk or entry_risk_pts > max_entry_risk:
        return {
            f"{side.lower()}_breakout_occurred": True,
            f"{side.lower()}_signal_ts": day.iloc[signal_idx]["timestamp_utc"],
            f"{side.lower()}_entry_ts": entry_bar["timestamp_utc"],
            f"{side.lower()}_exit_ts": pd.NaT,
            f"{side.lower()}_exit_reason": "RISK_FILTERED",
            f"{side.lower()}_success_2r": pd.NA,
            f"{side.lower()}_pnl_per_contract_usd": pd.NA,
            f"{side.lower()}_r_multiple": pd.NA,
        }

    exit_scan = day.iloc[entry_idx:]
    exit_scan = exit_scan[exit_scan["bar_data_quality_ok"].astype(bool)]
    time_exit_candidates = exit_scan[exit_scan["ny_time"] >= time_exit]
    if time_exit_candidates.empty:
        return {
            f"{side.lower()}_breakout_occurred": True,
            f"{side.lower()}_signal_ts": day.iloc[signal_idx]["timestamp_utc"],
            f"{side.lower()}_entry_ts": entry_bar["timestamp_utc"],
            f"{side.lower()}_exit_ts": pd.NaT,
            f"{side.lower()}_exit_reason": "NO_TIME_EXIT",
            f"{side.lower()}_success_2r": pd.NA,
            f"{side.lower()}_pnl_per_contract_usd": pd.NA,
            f"{side.lower()}_r_multiple": pd.NA,
        }

    time_exit_bar = time_exit_candidates.iloc[0]
    time_exit_idx = int(time_exit_bar.name)
    pre_time_exit = exit_scan.loc[exit_scan.index <= time_exit_idx]
    if side == "UP":
        tp_hits = pre_time_exit[pre_time_exit["high"] >= tp_price]
    else:
        tp_hits = pre_time_exit[pre_time_exit["low"] <= tp_price]

    if not tp_hits.empty:
        exit_bar = tp_hits.iloc[0]
        raw_exit_price = tp_price
        exit_reason = "TP_2R"
        success_2r = 1
    else:
        exit_bar = time_exit_bar
        raw_exit_price = float(time_exit_bar["close"])
        exit_reason = "TIME_EXIT"
        success_2r = 0

    if side == "UP":
        exit_price = float(raw_exit_price) - slippage_pts
        gross_pts = exit_price - entry_price
    else:
        exit_price = float(raw_exit_price) + slippage_pts
        gross_pts = entry_price - exit_price

    pnl_per_contract = gross_pts * point_value - commission
    risk_per_contract = entry_risk_pts * point_value
    return {
        f"{side.lower()}_breakout_occurred": True,
        f"{side.lower()}_signal_ts": day.iloc[signal_idx]["timestamp_utc"],
        f"{side.lower()}_entry_ts": entry_bar["timestamp_utc"],
        f"{side.lower()}_exit_ts": exit_bar["timestamp_utc"],
        f"{side.lower()}_entry_risk_pts": entry_risk_pts,
        f"{side.lower()}_risk_per_contract_usd": risk_per_contract,
        f"{side.lower()}_exit_reason": exit_reason,
        f"{side.lower()}_success_2r": success_2r,
        f"{side.lower()}_pnl_per_contract_usd": pnl_per_contract,
        f"{side.lower()}_r_multiple": pnl_per_contract / risk_per_contract,
    }


def build_daily_scenarios(
    l1: pd.DataFrame,
    parent_cfg: dict[str, Any],
    scenario_cfg: dict[str, Any],
) -> pd.DataFrame:
    scenario = scenario_cfg["scenario_contract"]
    orb_minutes = int(scenario["orb_minutes"])
    time_exit = scenario["time_exit"]
    market_open_min = hhmm_to_minutes(parent_cfg["session"]["market_open"])
    time_exit_min = hhmm_to_minutes(time_exit)
    time_exit_minutes_from_open = time_exit_min - market_open_min

    rows: list[dict[str, Any]] = []
    for ny_date, day in l1.groupby("ny_date", sort=True):
        day = day.sort_values("timestamp_utc").reset_index(drop=True)
        quality = day["bar_data_quality_ok"].astype(bool)
        orb_mask = quality & (day["minutes_from_open"] > 0) & (day["minutes_from_open"] <= orb_minutes)
        orb = day.loc[orb_mask]
        if len(orb) != orb_minutes:
            continue

        orb_high = float(orb["high"].max())
        orb_low = float(orb["low"].min())
        orb_range = orb_high - orb_low
        if orb_range <= 0:
            continue

        orb_open = float(orb.iloc[0]["open"])
        orb_close = float(orb.iloc[-1]["close"])
        orb_body = abs(orb_close - orb_open)
        orb_upper_wick = orb_high - max(orb_open, orb_close)
        orb_lower_wick = min(orb_open, orb_close) - orb_low
        pre_60 = day.loc[quality & (day["minutes_from_open"] >= -60) & (day["minutes_from_open"] <= 0)]
        if pre_60.empty:
            pre_60_count = 0
            pre_60_return = 0.0
            pre_60_range = 0.0
            pre_60_volume = 0.0
        else:
            pre_60_count = int(len(pre_60))
            pre_60_return = float(pre_60.iloc[-1]["close"] - pre_60.iloc[0]["open"])
            pre_60_range = float(pre_60["high"].max() - pre_60["low"].min())
            pre_60_volume = float(pre_60["volume"].sum())

        post = day[
            quality
            & (day["minutes_from_open"] > orb_minutes)
            & (day["minutes_from_open"] < time_exit_minutes_from_open)
        ]
        up_candidates = post[post["close"] > orb_high]
        down_candidates = post[post["close"] < orb_low]
        up_idx = int(up_candidates.index[0]) if not up_candidates.empty else None
        down_idx = int(down_candidates.index[0]) if not down_candidates.empty else None

        if up_idx is None and down_idx is None:
            first_side = "NONE"
        elif down_idx is None or (up_idx is not None and up_idx < down_idx):
            first_side = "UP"
        else:
            first_side = "DOWN"

        up_result = simulate_side(day, "UP", up_idx, orb_high, orb_low, parent_cfg, scenario_cfg)
        down_result = simulate_side(day, "DOWN", down_idx, orb_high, orb_low, parent_cfg, scenario_cfg)

        if first_side == "NONE":
            first_label = "NO_TRADE"
        elif first_side == "UP":
            first_label = "FULL_RISK" if is_success(up_result["up_success_2r"]) else "REDUCE_RISK"
        else:
            first_label = "FULL_RISK" if is_success(down_result["down_success_2r"]) else "REDUCE_RISK"

        row = {
            "ny_date": pd.Timestamp(ny_date).date(),
            "split": assign_split(ny_date, scenario_cfg["split"]),
            "orb_minutes": orb_minutes,
            "orb_end": "09:45",
            "time_exit": time_exit,
            "orb_high": orb_high,
            "orb_low": orb_low,
            "orb_range_pts": orb_range,
            "orb_body_pts": orb_body,
            "orb_direction_pts": orb_close - orb_open,
            "orb_close_position": (orb_close - orb_low) / orb_range,
            "orb_upper_wick_pts": orb_upper_wick,
            "orb_lower_wick_pts": orb_lower_wick,
            "orb_volume_sum": float(orb["volume"].sum()),
            "orb_volume_mean": float(orb["volume"].mean()),
            "orb_volume_max": float(orb["volume"].max()),
            "pre_60m_bar_count": pre_60_count,
            "pre_60m_return_pts": pre_60_return,
            "pre_60m_range_pts": pre_60_range,
            "pre_60m_volume_sum": pre_60_volume,
            "ny_day_of_week": int(pd.Timestamp(ny_date).dayofweek),
            "ny_month": int(pd.Timestamp(ny_date).month),
            "first_breakout_side": first_side,
            "first_breakout_risk_label": first_label,
        }
        row.update(up_result)
        row.update(down_result)
        rows.append(row)

    dataset = pd.DataFrame(rows).sort_values("ny_date").reset_index(drop=True)
    feature_nulls = dataset[FEATURE_COLUMNS].isna().sum()
    bad_feature_nulls = feature_nulls[feature_nulls > 0]
    if not bad_feature_nulls.empty:
        raise SystemExit(f"Unexpected feature nulls: {bad_feature_nulls.to_dict()}")
    return dataset


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    up = df[df["up_breakout_occurred"].astype(bool)]
    down = df[df["down_breakout_occurred"].astype(bool)]
    first_counts = df["first_breakout_side"].value_counts().sort_index().to_dict()
    action_counts = df["first_breakout_risk_label"].value_counts().sort_index().to_dict()
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().sort_index().to_dict().items()},
        "first_breakout_side_counts": {str(k): int(v) for k, v in first_counts.items()},
        "first_breakout_action_counts": {str(k): int(v) for k, v in action_counts.items()},
        "up_breakout_days": int(len(up)),
        "up_success_2r_rate": float(up["up_success_2r"].dropna().mean()) if not up.empty else None,
        "up_avg_pnl_per_contract_usd": float(up["up_pnl_per_contract_usd"].dropna().mean()) if not up.empty else None,
        "up_max_dd_per_contract_usd": max_drawdown(up["up_pnl_per_contract_usd"].dropna()),
        "down_breakout_days": int(len(down)),
        "down_success_2r_rate": float(down["down_success_2r"].dropna().mean()) if not down.empty else None,
        "down_avg_pnl_per_contract_usd": float(down["down_pnl_per_contract_usd"].dropna().mean()) if not down.empty else None,
        "down_max_dd_per_contract_usd": max_drawdown(down["down_pnl_per_contract_usd"].dropna()),
        "min_date": str(df["ny_date"].min()),
        "max_date": str(df["ny_date"].max()),
    }


def main() -> None:
    args = parse_args()
    scenario_cfg = load_json(Path(args.config))
    parent_cfg = load_parent_config()

    l1_path = project_path(scenario_cfg["inputs"]["l1_context"])
    output_path = project_path(scenario_cfg["outputs"]["daily_scenarios"])
    manifest_path = project_path(scenario_cfg["outputs"]["manifest"])
    model_dir = project_path(scenario_cfg["outputs"]["model_dir"])

    if output_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing dataset: {output_path}")
    if not l1_path.exists():
        raise SystemExit(f"Missing L1 context: {l1_path}")

    l1 = pd.read_parquet(l1_path)
    l1["timestamp_utc"] = pd.to_datetime(l1["timestamp_utc"], utc=True)
    dataset = build_daily_scenarios(l1, parent_cfg, scenario_cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)

    summary = summarize(dataset)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "experiment": scenario_cfg["experiment"],
        "scenario_contract": scenario_cfg["scenario_contract"],
        "input": str(l1_path),
        "output": str(output_path),
        "model_dir": str(model_dir),
        "feature_columns": FEATURE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "no_lookahead_note": "Features are known at OR completion. Breakout and PnL fields are labels/evaluation fields.",
        **summary,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

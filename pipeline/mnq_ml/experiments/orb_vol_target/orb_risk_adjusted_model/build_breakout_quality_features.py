#!/usr/bin/env python3
"""Build MNQ ORB breakout-quality features.

One row is one actual UP or DOWN breakout after the 15m opening range. Features
are available at the breakout candle close. Labels simulate whether the breakout
reached +2R before the 15:00 NY time exit using the next M1 open as entry.
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
sys.path.insert(0, str(SCRIPT_DIR))

from common import load_config as load_parent_config  # noqa: E402
from common import project_path, write_json  # noqa: E402
from build_daily_confluence_features import DAILY_CONFLUENCE_FEATURES  # noqa: E402

CONFIG_PATH = SCRIPT_DIR / "config.json"

FEATURE_FAMILIES = {
    "volatility_atr": [
        "prior_day_range_pts",
        "atr5_daily_range_pts",
        "atr14_daily_range_pts",
        "atr20_daily_range_pts",
        "orb_range_to_atr14",
        "signal_risk_to_atr14",
        "pre_60m_range_pts",
        "pre_60m_return_pts",
        "pre_60m_realized_vol_pts",
    ],
    "vwap_context": [
        "or_close_dist_to_or_vwap_pts",
        "signal_close_dist_to_vwap_pts",
        "vwap_slope_or_to_signal_pts",
        "side_aligned_with_vwap",
    ],
    "overnight_structure": [
        "overnight_range_pts",
        "overnight_return_pts",
        "or_high_to_overnight_high_pts",
        "or_low_to_overnight_low_pts",
        "side_distance_to_overnight_extreme_pts",
    ],
    "breakout_quality": [
        "side_is_up",
        "signal_minutes_from_open",
        "breakout_close_distance_pts",
        "breakout_close_distance_to_orb",
        "breakout_body_pts",
        "breakout_range_pts",
        "breakout_close_position",
        "breakout_wick_against_pts",
        "breakout_volume",
        "breakout_volume_to_or_mean",
        "breakout_volume_to_pre60_mean",
    ],
    "prior_day_context": [
        "prior_day_trend_pts",
        "prior_close_gap_pts",
        "or_mid_to_prior_close_pts",
        "side_distance_to_prior_extreme_pts",
    ],
    "daily_confluence": DAILY_CONFLUENCE_FEATURES,
}

FEATURE_COLUMNS = [col for cols in FEATURE_FAMILIES.values() for col in cols]
METADATA_COLUMNS = [
    "event_id",
    "ny_date",
    "split",
    "side",
    "orb_minutes",
    "signal_ts",
    "entry_ts",
    "exit_ts",
    "exit_reason",
    "daily_confluence_feature_date",
]
LABEL_COLUMNS = [
    "success_2r",
    "positive_eod",
    "outcome_bucket",
    "pnl_per_contract_usd",
    "r_multiple",
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


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


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


def session_vwap(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    vol = df["volume"].astype(float)
    total_vol = float(vol.sum())
    if total_vol <= 0:
        return float(df.iloc[-1]["close"])
    typical = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    return float((typical * vol).sum() / total_vol)


def realized_vol_pts(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return 0.0
    returns = df["close"].astype(float).diff().dropna()
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0))


def build_daily_context(l1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ny_date, day in l1.groupby("ny_date", sort=True):
        quality_day = day[day["bar_data_quality_ok"].astype(bool)]
        if quality_day.empty:
            continue
        rows.append(
            {
                "ny_date": ny_date,
                "daily_high": float(quality_day["high"].max()),
                "daily_low": float(quality_day["low"].min()),
                "daily_open": float(quality_day.iloc[0]["open"]),
                "daily_close": float(quality_day.iloc[-1]["close"]),
                "daily_range_pts": float(quality_day["high"].max() - quality_day["low"].min()),
            }
        )
    daily = pd.DataFrame(rows).sort_values("ny_date").reset_index(drop=True)
    daily["prior_day_high"] = daily["daily_high"].shift(1)
    daily["prior_day_low"] = daily["daily_low"].shift(1)
    daily["prior_day_close"] = daily["daily_close"].shift(1)
    daily["prior_day_trend_pts"] = (daily["daily_close"] - daily["daily_open"]).shift(1)
    daily["prior_day_range_pts"] = daily["daily_range_pts"].shift(1)
    for window in [5, 14, 20]:
        daily[f"atr{window}_daily_range_pts"] = daily["daily_range_pts"].shift(1).rolling(window, min_periods=3).mean()
    daily = daily.fillna(
        {
            "prior_day_high": 0.0,
            "prior_day_low": 0.0,
            "prior_day_close": 0.0,
            "prior_day_trend_pts": 0.0,
            "prior_day_range_pts": 0.0,
            "atr5_daily_range_pts": 0.0,
            "atr14_daily_range_pts": 0.0,
            "atr20_daily_range_pts": 0.0,
        }
    )
    return daily


def simulate_outcome(
    day: pd.DataFrame,
    side: str,
    signal_idx: int,
    orb_high: float,
    orb_low: float,
    parent_cfg: dict[str, Any],
    scenario_cfg: dict[str, Any],
) -> dict[str, Any] | None:
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
        return None
    entry_bar = day.iloc[entry_idx]
    if (not bool(entry_bar["bar_data_quality_ok"])) or entry_bar["ny_time"] > time_exit:
        return None

    if side == "UP":
        entry_price = float(entry_bar["open"]) + slippage_pts
        entry_risk_pts = entry_price - orb_low
        tp_price = entry_price + tp_r * entry_risk_pts
    else:
        entry_price = float(entry_bar["open"]) - slippage_pts
        entry_risk_pts = orb_high - entry_price
        tp_price = entry_price - tp_r * entry_risk_pts

    if entry_risk_pts < min_entry_risk or entry_risk_pts > max_entry_risk:
        return None

    exit_scan = day.iloc[entry_idx:]
    exit_scan = exit_scan[exit_scan["bar_data_quality_ok"].astype(bool)]
    time_exit_candidates = exit_scan[exit_scan["ny_time"] >= time_exit]
    if time_exit_candidates.empty:
        return None
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
    positive_eod = int(pnl_per_contract > 0)
    r_multiple = safe_div(pnl_per_contract, risk_per_contract)
    if success_2r:
        outcome_bucket = "TP_2R"
    elif positive_eod:
        outcome_bucket = "POSITIVE_EOD"
    else:
        outcome_bucket = "NEGATIVE_EOD"
    return {
        "entry_ts": entry_bar["timestamp_utc"],
        "exit_ts": exit_bar["timestamp_utc"],
        "exit_reason": exit_reason,
        "success_2r": success_2r,
        "positive_eod": positive_eod,
        "outcome_bucket": outcome_bucket,
        "pnl_per_contract_usd": pnl_per_contract,
        "r_multiple": r_multiple,
    }


def build_rows(l1: pd.DataFrame, parent_cfg: dict[str, Any], scenario_cfg: dict[str, Any]) -> pd.DataFrame:
    scenario = scenario_cfg["scenario_contract"]
    orb_minutes = int(scenario["orb_minutes"])
    time_exit = scenario["time_exit"]
    market_open_min = hhmm_to_minutes(parent_cfg["session"]["market_open"])
    time_exit_min = hhmm_to_minutes(time_exit)
    time_exit_minutes_from_open = time_exit_min - market_open_min
    daily_context = build_daily_context(l1).set_index("ny_date").to_dict("index")

    rows = []
    for ny_date, day in l1.groupby("ny_date", sort=True):
        day = day.sort_values("timestamp_utc").reset_index(drop=True)
        quality = day["bar_data_quality_ok"].astype(bool)
        orb = day.loc[quality & (day["minutes_from_open"] > 0) & (day["minutes_from_open"] <= orb_minutes)]
        if len(orb) != orb_minutes:
            continue

        orb_high = float(orb["high"].max())
        orb_low = float(orb["low"].min())
        orb_range = orb_high - orb_low
        if orb_range <= 0:
            continue
        orb_mid = (orb_high + orb_low) / 2.0
        orb_close = float(orb.iloc[-1]["close"])
        or_vwap = session_vwap(orb)

        pre_60 = day.loc[quality & (day["minutes_from_open"] >= -60) & (day["minutes_from_open"] <= 0)]
        pre_60_range = float(pre_60["high"].max() - pre_60["low"].min()) if not pre_60.empty else 0.0
        pre_60_return = float(pre_60.iloc[-1]["close"] - pre_60.iloc[0]["open"]) if not pre_60.empty else 0.0
        pre_60_volume_mean = float(pre_60["volume"].mean()) if not pre_60.empty else 0.0
        pre_60_realized_vol = realized_vol_pts(pre_60)

        overnight = day.loc[quality & (day["minutes_from_open"] <= 0)]
        overnight_high = float(overnight["high"].max()) if not overnight.empty else orb_high
        overnight_low = float(overnight["low"].min()) if not overnight.empty else orb_low
        overnight_range = overnight_high - overnight_low
        overnight_return = float(overnight.iloc[-1]["close"] - overnight.iloc[0]["open"]) if not overnight.empty else 0.0

        post = day[
            quality
            & (day["minutes_from_open"] > orb_minutes)
            & (day["minutes_from_open"] < time_exit_minutes_from_open)
        ]
        side_candidates = {
            "UP": post[post["close"] > orb_high],
            "DOWN": post[post["close"] < orb_low],
        }
        day_context = daily_context.get(ny_date, {})
        prior_high = float(day_context.get("prior_day_high", 0.0))
        prior_low = float(day_context.get("prior_day_low", 0.0))
        prior_close = float(day_context.get("prior_day_close", 0.0))
        prior_day_range = float(day_context.get("prior_day_range_pts", 0.0))
        atr5 = float(day_context.get("atr5_daily_range_pts", 0.0))
        atr14 = float(day_context.get("atr14_daily_range_pts", 0.0))
        atr20 = float(day_context.get("atr20_daily_range_pts", 0.0))
        prior_trend = float(day_context.get("prior_day_trend_pts", 0.0))

        for side, candidates in side_candidates.items():
            if candidates.empty:
                continue
            signal_idx = int(candidates.index[0])
            signal = day.iloc[signal_idx]
            outcome = simulate_outcome(day, side, signal_idx, orb_high, orb_low, parent_cfg, scenario_cfg)
            if outcome is None:
                continue

            signal_history = day.loc[
                quality
                & (day["minutes_from_open"] > 0)
                & (day["minutes_from_open"] <= int(signal["minutes_from_open"]))
            ]
            signal_vwap = session_vwap(signal_history)
            signal_close = float(signal["close"])
            signal_open = float(signal["open"])
            signal_high = float(signal["high"])
            signal_low = float(signal["low"])
            signal_range = signal_high - signal_low
            signal_body = abs(signal_close - signal_open)
            side_sign = 1.0 if side == "UP" else -1.0
            breakout_distance = signal_close - orb_high if side == "UP" else orb_low - signal_close
            signal_risk = signal_close - orb_low if side == "UP" else orb_high - signal_close
            close_position = safe_div(signal_close - signal_low, signal_range)
            if side == "DOWN":
                close_position = 1.0 - close_position
            wick_against = signal_high - max(signal_open, signal_close) if side == "DOWN" else min(signal_open, signal_close) - signal_low
            volume_to_or = safe_div(float(signal["volume"]), float(orb["volume"].mean()))
            volume_to_pre60 = safe_div(float(signal["volume"]), pre_60_volume_mean)
            side_distance_to_overnight_extreme = (
                overnight_high - signal_close if side == "UP" else signal_close - overnight_low
            )
            side_distance_to_prior_extreme = (
                prior_high - signal_close if side == "UP" else signal_close - prior_low
            )
            if prior_close:
                prior_close_gap = float(orb.iloc[0]["open"]) - prior_close
                or_mid_to_prior_close = orb_mid - prior_close
            else:
                prior_close_gap = 0.0
                or_mid_to_prior_close = 0.0

            rows.append(
                {
                    "event_id": f"MNQ_ORB_15m_{side}_{ny_date}",
                    "ny_date": pd.Timestamp(ny_date).date(),
                    "split": assign_split(ny_date, scenario_cfg["split"]),
                    "side": side,
                    "orb_minutes": orb_minutes,
                    "signal_ts": signal["timestamp_utc"],
                    **outcome,
                    "prior_day_range_pts": prior_day_range,
                    "atr5_daily_range_pts": atr5,
                    "atr14_daily_range_pts": atr14,
                    "atr20_daily_range_pts": atr20,
                    "orb_range_to_atr14": safe_div(orb_range, atr14),
                    "signal_risk_to_atr14": safe_div(signal_risk, atr14),
                    "pre_60m_range_pts": pre_60_range,
                    "pre_60m_return_pts": pre_60_return,
                    "pre_60m_realized_vol_pts": pre_60_realized_vol,
                    "or_close_dist_to_or_vwap_pts": orb_close - or_vwap,
                    "signal_close_dist_to_vwap_pts": signal_close - signal_vwap,
                    "vwap_slope_or_to_signal_pts": signal_vwap - or_vwap,
                    "side_aligned_with_vwap": int(side_sign * (signal_close - signal_vwap) > 0),
                    "overnight_range_pts": overnight_range,
                    "overnight_return_pts": overnight_return,
                    "or_high_to_overnight_high_pts": overnight_high - orb_high,
                    "or_low_to_overnight_low_pts": orb_low - overnight_low,
                    "side_distance_to_overnight_extreme_pts": side_distance_to_overnight_extreme,
                    "side_is_up": int(side == "UP"),
                    "signal_minutes_from_open": int(signal["minutes_from_open"]),
                    "breakout_close_distance_pts": breakout_distance,
                    "breakout_close_distance_to_orb": safe_div(breakout_distance, orb_range),
                    "breakout_body_pts": signal_body,
                    "breakout_range_pts": signal_range,
                    "breakout_close_position": close_position,
                    "breakout_wick_against_pts": wick_against,
                    "breakout_volume": float(signal["volume"]),
                    "breakout_volume_to_or_mean": volume_to_or,
                    "breakout_volume_to_pre60_mean": volume_to_pre60,
                    "prior_day_trend_pts": prior_trend,
                    "prior_close_gap_pts": prior_close_gap,
                    "or_mid_to_prior_close_pts": or_mid_to_prior_close,
                    "side_distance_to_prior_extreme_pts": side_distance_to_prior_extreme,
                }
            )

    df = pd.DataFrame(rows).sort_values(["signal_ts", "side"]).reset_index(drop=True)
    daily_confluence_path = scenario_cfg["inputs"].get("daily_confluence")
    if not daily_confluence_path:
        raise SystemExit("Missing inputs.daily_confluence in risk-adjusted config.")
    daily_confluence = pd.read_parquet(project_path(daily_confluence_path))
    daily_confluence["ny_date"] = pd.to_datetime(daily_confluence["ny_date"]).dt.date
    daily_confluence["daily_confluence_feature_date"] = pd.to_datetime(
        daily_confluence["daily_confluence_feature_date"]
    ).dt.date
    df["ny_date"] = pd.to_datetime(df["ny_date"]).dt.date
    df = df.merge(daily_confluence[["ny_date", "daily_confluence_feature_date", *DAILY_CONFLUENCE_FEATURES]], on="ny_date", how="left")
    missing_confluence = df["daily_confluence_feature_date"].isna()
    if bool(missing_confluence.any()):
        sample = df.loc[missing_confluence, ["event_id", "ny_date"]].head(10).to_dict("records")
        raise SystemExit(f"Missing daily confluence rows for {int(missing_confluence.sum())} events. Sample: {sample}")
    lookahead = df[df["daily_confluence_feature_date"] >= df["ny_date"]]
    if len(lookahead):
        raise SystemExit(f"Daily confluence lookahead violations: {len(lookahead)}")
    nulls = df[FEATURE_COLUMNS + LABEL_COLUMNS].isna().sum()
    bad_nulls = nulls[nulls > 0]
    if not bad_nulls.empty:
        raise SystemExit(f"Unexpected nulls: {bad_nulls.to_dict()}")
    return df[METADATA_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS]


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().sort_index().to_dict().items()},
        "side_counts": {str(k): int(v) for k, v in df["side"].value_counts().sort_index().to_dict().items()},
        "success_2r_rate": float(df["success_2r"].mean()),
        "positive_eod_rate": float(df["positive_eod"].mean()),
        "success_2r_rate_by_side": {
            str(k): float(v) for k, v in df.groupby("side")["success_2r"].mean().sort_index().to_dict().items()
        },
        "positive_eod_rate_by_side": {
            str(k): float(v) for k, v in df.groupby("side")["positive_eod"].mean().sort_index().to_dict().items()
        },
        "outcome_bucket_counts": {
            str(k): int(v) for k, v in df["outcome_bucket"].value_counts().sort_index().to_dict().items()
        },
        "avg_pnl_per_contract_usd_by_side": {
            str(k): float(v) for k, v in df.groupby("side")["pnl_per_contract_usd"].mean().sort_index().to_dict().items()
        },
        "min_signal_ts": df["signal_ts"].min().isoformat(),
        "max_signal_ts": df["signal_ts"].max().isoformat(),
    }


def main() -> None:
    args = parse_args()
    scenario_cfg = load_json(Path(args.config))
    parent_cfg = load_parent_config()
    l1_path = project_path(scenario_cfg["inputs"]["l1_context"])
    output_path = project_path(scenario_cfg["outputs"]["breakout_quality"])
    manifest_path = project_path(scenario_cfg["outputs"]["breakout_quality_manifest"])
    model_dir = project_path(scenario_cfg["outputs"]["model_dir"])

    if output_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing dataset: {output_path}")
    if not l1_path.exists():
        raise SystemExit(f"Missing L1 context: {l1_path}")

    l1 = pd.read_parquet(l1_path)
    l1["timestamp_utc"] = pd.to_datetime(l1["timestamp_utc"], utc=True)
    l1["ny_date"] = pd.to_datetime(l1["ny_date"]).dt.date
    dataset = build_rows(l1, parent_cfg, scenario_cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "experiment": scenario_cfg["experiment"],
        "scenario_contract": scenario_cfg["scenario_contract"],
        "input": str(l1_path),
        "output": str(output_path),
        "feature_families": FEATURE_FAMILIES,
        "feature_columns": FEATURE_COLUMNS,
        "metadata_columns": METADATA_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "no_lookahead_note": "Features are available at breakout candle close. Entry/exit/PnL fields are labels or metadata only.",
        **summarize(dataset),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

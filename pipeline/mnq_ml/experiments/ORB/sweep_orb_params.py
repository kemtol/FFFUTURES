#!/usr/bin/env python3
"""Sweep MNQ ORB duration, side mode, and target risk.

This is a research grid, not the canonical baseline event builder. It reads the
M1 L1 context and writes sweep artifacts under the strategy L2 folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from math import floor, sqrt
from pathlib import Path

import pandas as pd

from common import assert_mnq_namespaces, load_config, project_path, write_json

DEFAULT_ORB_MINUTES = [10, 15, 20, 30]
DEFAULT_SIDE_MODES = ["long", "short", "long_short"]
DEFAULT_TARGET_RISKS = [100, 200, 300, 400, 500, 600]
DEFAULT_EXIT_MODES = ["time_exit", "tp_2r_or_time"]
WINDOW_DAYS = [5, 10, 20, 30, 50, 100, 200]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orb-minutes", default=",".join(map(str, DEFAULT_ORB_MINUTES)))
    parser.add_argument("--side-modes", default=",".join(DEFAULT_SIDE_MODES))
    parser.add_argument("--target-risks", default=",".join(map(str, DEFAULT_TARGET_RISKS)))
    parser.add_argument("--exit-modes", default=",".join(DEFAULT_EXIT_MODES))
    parser.add_argument("--output-dir", default="data/Level_2_Datamart/mnq/ORB/sweeps")
    parser.add_argument("--min-trades", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def hhmm_to_minutes(value: str) -> int:
    return int(value[:2]) * 60 + int(value[3:])


def minutes_to_hhmm(value: int) -> str:
    return f"{value // 60:02d}:{value % 60:02d}"


def max_drawdown(pnls: pd.Series) -> float:
    if pnls.empty:
        return 0.0
    equity = pnls.cumsum()
    equity = pd.concat([pd.Series([0.0]), equity], ignore_index=True)
    drawdown = equity - equity.cummax()
    return float(drawdown.min())


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def max_consecutive(mask: pd.Series) -> int:
    best = 0
    current = 0
    for value in mask.astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def build_daily_pnl(group: pd.DataFrame, session_dates: pd.Series) -> pd.Series:
    group_dates = pd.to_datetime(group["ny_date"])
    start = group_dates.min()
    end = group_dates.max()
    date_index = pd.DatetimeIndex(pd.to_datetime(session_dates).drop_duplicates().sort_values())
    date_index = date_index[(date_index >= start) & (date_index <= end)]
    daily = group.assign(_date=group_dates).groupby("_date")["pnl_usd"].sum()
    return daily.reindex(date_index, fill_value=0.0).astype(float)


def annualized_sharpe(daily_pnl: pd.Series) -> float | None:
    if len(daily_pnl) < 2:
        return None
    std = float(daily_pnl.std(ddof=1))
    if std == 0:
        return None
    return float((daily_pnl.mean() / std) * sqrt(252.0))


def annualized_sortino(daily_pnl: pd.Series) -> float | None:
    if daily_pnl.empty:
        return None
    downside = daily_pnl.clip(upper=0.0)
    downside_dev = float((downside.pow(2).mean()) ** 0.5)
    if downside_dev == 0:
        return None
    return float((daily_pnl.mean() / downside_dev) * sqrt(252.0))


def parse_exit_mode(value: str) -> tuple[str, float | None]:
    if value == "time_exit":
        return value, None
    if value.startswith("tp_") and value.endswith("r_or_time"):
        raw = value.removeprefix("tp_").removesuffix("r_or_time").replace("_", ".")
        return value, float(raw)
    raise ValueError(f"Unsupported exit mode: {value}")


def load_l1(cfg: dict) -> pd.DataFrame:
    l1_path = project_path(cfg["outputs"]["l1_context"])
    if not l1_path.exists():
        raise SystemExit(f"Missing L1 context: {l1_path}")
    l1 = pd.read_parquet(l1_path)
    l1["timestamp_utc"] = pd.to_datetime(l1["timestamp_utc"], utc=True)
    return l1.sort_values("timestamp_utc").reset_index(drop=True)


def build_base_opportunities(
    l1: pd.DataFrame,
    cfg: dict,
    orb_minutes_grid: list[int],
    side_modes: list[str],
    exit_modes: list[str],
) -> pd.DataFrame:
    rules = cfg["rules"]
    costs = cfg["costs"]
    session = cfg["session"]

    market_open_min = hhmm_to_minutes(session["market_open"])
    time_exit = session["time_exit"]
    time_exit_min = hhmm_to_minutes(time_exit)
    point_value = float(costs["point_value_usd"])
    slippage_pts = float(costs["slippage_ticks_per_side"]) * float(costs["tick_size"])
    commission = float(costs["commission_round_turn_usd"])
    min_orb_range = float(rules["min_orb_range_pts"])
    max_orb_range = float(rules["max_orb_range_pts"])
    min_entry_risk = float(rules["min_entry_risk_pts"])
    max_entry_risk = float(rules["max_entry_risk_pts"])

    rows: list[dict] = []
    for ny_date, day in l1.groupby("ny_date", sort=True):
        day = day.reset_index(drop=True)
        for orb_minutes in orb_minutes_grid:
            orb_end_min = market_open_min + orb_minutes
            orb_end = minutes_to_hhmm(orb_end_min)
            quality_mask = day["bar_data_quality_ok"].astype(bool)
            orb_mask = (
                quality_mask
                & (day["minutes_from_open"] > 0)
                & (day["minutes_from_open"] <= orb_minutes)
            )
            orb = day.loc[orb_mask]
            if len(orb) != orb_minutes:
                continue
            orb_high = float(orb["high"].max())
            orb_low = float(orb["low"].min())
            orb_range = orb_high - orb_low
            if orb_range < min_orb_range or orb_range > max_orb_range:
                continue

            post = day[
                quality_mask
                & (day["minutes_from_open"] > orb_minutes)
                & (day["minutes_from_open"] < (time_exit_min - market_open_min))
            ]
            if post.empty:
                continue

            first_candidates: dict[str, int] = {}
            long_candidates = post[post["close"] > orb_high]
            if not long_candidates.empty:
                first_candidates["LONG"] = int(long_candidates.index[0])
            short_candidates = post[post["close"] < orb_low]
            if not short_candidates.empty:
                first_candidates["SHORT"] = int(short_candidates.index[0])

            for side_mode in side_modes:
                selected: tuple[str, int] | None = None
                if side_mode == "long" and "LONG" in first_candidates:
                    selected = ("LONG", first_candidates["LONG"])
                elif side_mode == "short" and "SHORT" in first_candidates:
                    selected = ("SHORT", first_candidates["SHORT"])
                elif side_mode == "long_short" and first_candidates:
                    selected = sorted(first_candidates.items(), key=lambda x: x[1])[0]
                if selected is None:
                    continue

                side, signal_idx = selected
                entry_idx = signal_idx + 1
                if entry_idx >= len(day):
                    continue
                signal = day.iloc[signal_idx]
                entry_bar = day.iloc[entry_idx]
                if (not bool(entry_bar["bar_data_quality_ok"])) or entry_bar["ny_time"] > time_exit:
                    continue

                if side == "LONG":
                    entry_price = float(entry_bar["open"]) + slippage_pts
                    stop_reference = orb_low
                    entry_risk_pts = entry_price - stop_reference
                else:
                    entry_price = float(entry_bar["open"]) - slippage_pts
                    stop_reference = orb_high
                    entry_risk_pts = stop_reference - entry_price

                if entry_risk_pts < min_entry_risk or entry_risk_pts > max_entry_risk:
                    continue

                exit_scan = day.iloc[entry_idx:]
                exit_scan = exit_scan[exit_scan["bar_data_quality_ok"].astype(bool)]
                time_exit_candidates = exit_scan[exit_scan["ny_time"] >= time_exit]
                if time_exit_candidates.empty:
                    continue
                time_exit_bar = time_exit_candidates.iloc[0]
                time_exit_idx = int(time_exit_bar.name)
                for exit_mode in exit_modes:
                    _, tp_r = parse_exit_mode(exit_mode)
                    exit_reason = "TIME_EXIT"
                    exit_bar = time_exit_bar
                    raw_exit_price = float(time_exit_bar["close"])
                    if tp_r is not None:
                        tp_price = entry_price + tp_r * entry_risk_pts if side == "LONG" else entry_price - tp_r * entry_risk_pts
                        pre_time_exit = exit_scan.loc[exit_scan.index <= time_exit_idx]
                        if side == "LONG":
                            tp_hits = pre_time_exit[pre_time_exit["high"] >= tp_price]
                        else:
                            tp_hits = pre_time_exit[pre_time_exit["low"] <= tp_price]
                        if not tp_hits.empty:
                            exit_bar = tp_hits.iloc[0]
                            raw_exit_price = tp_price
                            exit_reason = f"TP_{tp_r:g}R"

                    if side == "LONG":
                        exit_price = float(raw_exit_price) - slippage_pts
                        gross_pts = exit_price - entry_price
                    else:
                        exit_price = float(raw_exit_price) + slippage_pts
                        gross_pts = entry_price - exit_price

                    pnl_per_contract = gross_pts * point_value - commission
                    risk_per_contract = entry_risk_pts * point_value
                    event_id = f"MNQ_ORB_{orb_minutes}m_{side_mode}_{exit_mode}_{ny_date}"
                    rows.append(
                        {
                            "base_event_id": event_id,
                            "ny_date": ny_date,
                            "orb_minutes": orb_minutes,
                            "orb_end": orb_end,
                            "side_mode": side_mode,
                            "exit_mode": exit_mode,
                            "exit_reason": exit_reason,
                            "side": side,
                            "signal_ts": signal["timestamp_utc"],
                            "entry_ts": entry_bar["timestamp_utc"],
                            "exit_ts": exit_bar["timestamp_utc"],
                            "signal_minutes_from_open": int(signal["minutes_from_open"]),
                            "orb_high": orb_high,
                            "orb_low": orb_low,
                            "orb_range_pts": orb_range,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "stop_reference": stop_reference,
                            "entry_risk_pts": entry_risk_pts,
                            "risk_per_contract_usd": risk_per_contract,
                            "pnl_per_contract_usd": pnl_per_contract,
                            "label": int(pnl_per_contract > 0),
                        }
                    )

    return pd.DataFrame(rows).sort_values(["orb_minutes", "side_mode", "signal_ts"]).reset_index(drop=True)


def expand_target_risks(base: pd.DataFrame, cfg: dict, target_risks: list[int]) -> pd.DataFrame:
    if base.empty:
        return base.copy()
    max_contracts = int(cfg["position_sizing"]["max_contracts"])
    min_contracts = int(cfg["position_sizing"]["min_contracts"])

    frames = []
    for risk in target_risks:
        cur = base.copy()
        cur["target_risk_usd"] = int(risk)
        cur["contracts_float"] = cur["target_risk_usd"] / cur["risk_per_contract_usd"]
        cur["contracts_floor"] = cur["contracts_float"].apply(floor).astype(int)
        cur["contracts_used"] = cur["contracts_floor"].clip(lower=0, upper=max_contracts)
        cur = cur[cur["contracts_used"] >= min_contracts].copy()
        cur["pnl_usd"] = cur["pnl_per_contract_usd"] * cur["contracts_used"]
        cur["event_id"] = (
            cur["base_event_id"]
            + "_risk"
            + cur["target_risk_usd"].astype(str)
        )
        frames.append(cur)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["orb_minutes", "side_mode", "exit_mode", "target_risk_usd", "signal_ts"]
    )


def summarize_group(group: pd.DataFrame, anchor: pd.Timestamp, session_dates: pd.Series) -> dict:
    group = group.sort_values("signal_ts")
    total_pnl = float(group["pnl_usd"].sum())
    max_dd = max_drawdown(group["pnl_usd"])
    wins = group[group["pnl_usd"] > 0]["pnl_usd"]
    losses = group[group["pnl_usd"] < 0]["pnl_usd"]
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    daily_pnl = build_daily_pnl(group, session_dates)
    active_daily = daily_pnl[daily_pnl != 0]
    best_day = float(daily_pnl.max()) if not daily_pnl.empty else 0.0
    worst_day = float(daily_pnl.min()) if not daily_pnl.empty else 0.0
    result = {
        "orb_minutes": int(group["orb_minutes"].iloc[0]),
        "side_mode": str(group["side_mode"].iloc[0]),
        "exit_mode": str(group["exit_mode"].iloc[0]),
        "target_risk_usd": int(group["target_risk_usd"].iloc[0]),
        "trades": int(len(group)),
        "win_rate": float(group["label"].mean()),
        "total_pnl_usd": total_pnl,
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "profit_factor": safe_ratio(gross_profit, abs(gross_loss)),
        "avg_pnl_usd": float(group["pnl_usd"].mean()),
        "median_pnl_usd": float(group["pnl_usd"].median()),
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "payoff_ratio": safe_ratio(avg_win, abs(avg_loss)),
        "expectancy_per_trade_usd": float(group["pnl_usd"].mean()),
        "max_consecutive_wins": max_consecutive(group["pnl_usd"] > 0),
        "max_consecutive_losses": max_consecutive(group["pnl_usd"] < 0),
        "max_dd_usd": max_dd,
        "return_dd": total_pnl / abs(max_dd) if max_dd < 0 else None,
        "avg_contracts": float(group["contracts_used"].mean()),
        "trading_days": int(len(daily_pnl)),
        "active_days": int(len(active_daily)),
        "active_day_rate": safe_ratio(float(len(active_daily)), float(len(daily_pnl))) or 0.0,
        "active_day_win_rate": float((active_daily > 0).mean()) if not active_daily.empty else 0.0,
        "daily_avg_pnl_usd": float(daily_pnl.mean()) if not daily_pnl.empty else 0.0,
        "daily_std_pnl_usd": float(daily_pnl.std(ddof=1)) if len(daily_pnl) > 1 else 0.0,
        "daily_sharpe_annualized": annualized_sharpe(daily_pnl),
        "daily_sortino_annualized": annualized_sortino(daily_pnl),
        "best_day_pnl_usd": best_day,
        "worst_day_pnl_usd": worst_day,
        "best_day_profit_share": safe_ratio(best_day, total_pnl) if total_pnl > 0 else None,
        "topstep_50pct_consistency_ok": bool((best_day / total_pnl) <= 0.5) if total_pnl > 0 else False,
        "min_signal_ts": group["signal_ts"].min().isoformat(),
        "max_signal_ts": group["signal_ts"].max().isoformat(),
    }

    y2026 = group[group["signal_ts"].dt.year == 2026]
    result.update(
        {
            "trades_2026": int(len(y2026)),
            "win_rate_2026": float(y2026["label"].mean()) if not y2026.empty else 0.0,
            "pnl_2026_usd": float(y2026["pnl_usd"].sum()) if not y2026.empty else 0.0,
            "max_dd_2026_usd": max_drawdown(y2026["pnl_usd"]) if not y2026.empty else 0.0,
        }
    )

    for days in WINDOW_DAYS:
        start = anchor - pd.Timedelta(days=days)
        window = group[(group["signal_ts"] > start) & (group["signal_ts"] <= anchor)]
        prefix = f"w{days}d"
        result[f"{prefix}_trades"] = int(len(window))
        result[f"{prefix}_win_rate"] = float(window["label"].mean()) if not window.empty else 0.0
        result[f"{prefix}_pnl_usd"] = float(window["pnl_usd"].sum()) if not window.empty else 0.0
        result[f"{prefix}_max_dd_usd"] = max_drawdown(window["pnl_usd"]) if not window.empty else 0.0

    return result


def build_results(events: pd.DataFrame, anchor: pd.Timestamp, min_trades: int, session_dates: pd.Series) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for _, group in events.groupby(["orb_minutes", "side_mode", "exit_mode", "target_risk_usd"], sort=True):
        row = summarize_group(group, anchor, session_dates)
        row["meets_min_trades"] = bool(row["trades"] >= min_trades)
        rows.append(row)
    results = pd.DataFrame(rows)
    results["rank_return_dd"] = results["return_dd"].rank(ascending=False, method="min")
    results["rank_2026_pnl"] = results["pnl_2026_usd"].rank(ascending=False, method="min")
    results["rank_recent_100d_pnl"] = results["w100d_pnl_usd"].rank(ascending=False, method="min")
    results["score_quick"] = (
        results["return_dd"].fillna(0.0)
        + (results["pnl_2026_usd"] / 3000.0)
        + (results["w100d_pnl_usd"] / 3000.0)
    )
    results = results.sort_values(
        ["meets_min_trades", "score_quick", "return_dd", "pnl_2026_usd"],
        ascending=[False, False, False, False],
    )
    return results.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    cfg = load_config()
    assert_mnq_namespaces(cfg)

    orb_minutes = parse_int_list(args.orb_minutes)
    side_modes = parse_str_list(args.side_modes)
    exit_modes = parse_str_list(args.exit_modes)
    target_risks = parse_int_list(args.target_risks)
    invalid_modes = sorted(set(side_modes) - {"long", "short", "long_short"})
    if invalid_modes:
        raise SystemExit(f"Invalid side modes: {invalid_modes}")
    for mode in exit_modes:
        parse_exit_mode(mode)

    output_dir = project_path(args.output_dir)
    events_path = output_dir / "sweep_events.parquet"
    results_path = output_dir / "sweep_results.parquet"
    manifest_path = output_dir / "sweep_manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output dir exists; use --force: {output_dir}")

    l1 = load_l1(cfg)
    anchor = pd.Timestamp(l1["timestamp_utc"].max())
    base = build_base_opportunities(l1, cfg, orb_minutes, side_modes, exit_modes)
    events = expand_target_risks(base, cfg, target_risks)
    session_dates = l1["ny_date"].drop_duplicates()
    results = build_results(events, anchor, args.min_trades, session_dates)

    output_dir.mkdir(parents=True, exist_ok=True)
    base.to_parquet(output_dir / "sweep_base_opportunities.parquet", index=False)
    events.to_parquet(events_path, index=False)
    results.to_parquet(results_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "l1_context": cfg["outputs"]["l1_context"],
        "anchor_ts": anchor.isoformat(),
        "orb_minutes": orb_minutes,
        "side_modes": side_modes,
        "exit_modes": exit_modes,
        "target_risks": target_risks,
        "base_opportunity_rows": int(len(base)),
        "event_rows": int(len(events)),
        "result_rows": int(len(results)),
        "min_trades": int(args.min_trades),
        "outputs": {
            "base_opportunities": str(output_dir / "sweep_base_opportunities.parquet"),
            "events": str(events_path),
            "results": str(results_path),
            "manifest": str(manifest_path),
        },
    }
    write_json(manifest_path, manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not results.empty:
        cols = [
            "orb_minutes",
            "side_mode",
            "exit_mode",
            "target_risk_usd",
            "trades",
            "win_rate",
            "profit_factor",
            "daily_sharpe_annualized",
            "daily_sortino_annualized",
            "total_pnl_usd",
            "max_dd_usd",
            "return_dd",
            "trades_2026",
            "pnl_2026_usd",
            "w30d_pnl_usd",
            "w30d_max_dd_usd",
            "w100d_pnl_usd",
            "w100d_max_dd_usd",
            "score_quick",
        ]
        print(results[cols].head(20).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

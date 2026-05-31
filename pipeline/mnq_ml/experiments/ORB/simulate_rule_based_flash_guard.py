#!/usr/bin/env python3
"""Simulate catastrophic loss guards for the MNQ ORB rule-based candidate."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Any

import pandas as pd

from common import load_config, project_path

WINDOW_DAYS = [5, 10, 20, 30, 50, 100, 200]


def parse_thresholds(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy-dir",
        default="data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod",
    )
    parser.add_argument("--thresholds", default="1000,1500,2000,3000")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def max_drawdown(pnls: pd.Series) -> float:
    if pnls.empty:
        return 0.0
    equity = pd.concat([pd.Series([0.0]), pnls.reset_index(drop=True).cumsum()], ignore_index=True)
    drawdown = equity - equity.cummax()
    return float(drawdown.min())


def profit_factor(pnls: pd.Series) -> float | None:
    gross_profit = float(pnls[pnls > 0].sum())
    gross_loss = float(pnls[pnls < 0].sum())
    if gross_loss == 0:
        return None
    return float(gross_profit / abs(gross_loss))


def annualized_sharpe(daily_pnl: pd.Series) -> float | None:
    if len(daily_pnl) < 2:
        return None
    std = float(daily_pnl.std(ddof=1))
    if std == 0:
        return None
    return float(daily_pnl.mean() / std * sqrt(252.0))


def annualized_sortino(daily_pnl: pd.Series) -> float | None:
    if daily_pnl.empty:
        return None
    downside = daily_pnl.clip(upper=0.0)
    downside_dev = float((downside.pow(2).mean()) ** 0.5)
    if downside_dev == 0:
        return None
    return float(daily_pnl.mean() / downside_dev * sqrt(252.0))


def simulate_threshold(
    events: pd.DataFrame,
    l1_by_date: dict[Any, pd.DataFrame],
    threshold_usd: float | None,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    costs = cfg["costs"]
    point_value = float(costs["point_value_usd"])
    slippage_pts = float(costs["slippage_ticks_per_side"]) * float(costs["tick_size"])
    commission = float(costs["commission_round_turn_usd"])
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        row = event._asdict()
        if threshold_usd is None:
            row["guard_threshold_usd"] = None
            row["guard_hit"] = False
            row["guard_exit_ts"] = pd.NaT
            row["guard_exit_price"] = None
            row["pnl_guarded_usd"] = float(row["pnl_net_usd"])
            row["exit_reason_guarded"] = row["exit_reason"]
            rows.append(row)
            continue

        contracts = int(row["contracts_used"])
        guard_trigger = float(row["entry_price"]) - (threshold_usd / (contracts * point_value))
        day = l1_by_date[row["ny_date"]]
        scan = day[
            (day["timestamp_utc"] >= row["entry_ts"])
            & (day["timestamp_utc"] <= row["exit_ts"])
            & day["bar_data_quality_ok"].astype(bool)
        ]
        hit = scan[scan["low"].astype(float) <= guard_trigger]
        if hit.empty:
            row["guard_threshold_usd"] = threshold_usd
            row["guard_hit"] = False
            row["guard_exit_ts"] = pd.NaT
            row["guard_exit_price"] = None
            row["pnl_guarded_usd"] = float(row["pnl_net_usd"])
            row["exit_reason_guarded"] = row["exit_reason"]
        else:
            bar = hit.iloc[0]
            exit_price = guard_trigger - slippage_pts
            pnl_per_contract = (exit_price - float(row["entry_price"])) * point_value - commission
            row["guard_threshold_usd"] = threshold_usd
            row["guard_hit"] = True
            row["guard_exit_ts"] = bar["timestamp_utc"]
            row["guard_exit_price"] = exit_price
            row["pnl_guarded_usd"] = pnl_per_contract * contracts
            row["exit_reason_guarded"] = "FLASH_GUARD"
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(events: pd.DataFrame, anchor: pd.Timestamp, session_dates: pd.Series) -> dict[str, Any]:
    pnl = events["pnl_guarded_usd"].astype(float)
    date_index = pd.DatetimeIndex(pd.to_datetime(session_dates).drop_duplicates().sort_values())
    event_dates = pd.to_datetime(events["ny_date"])
    date_index = date_index[(date_index >= event_dates.min()) & (date_index <= event_dates.max())]
    daily = events.assign(_date=event_dates).groupby("_date")["pnl_guarded_usd"].sum()
    daily = daily.reindex(date_index, fill_value=0.0).astype(float)
    row: dict[str, Any] = {
        "guard_threshold_usd": None if events["guard_threshold_usd"].isna().all() else float(events["guard_threshold_usd"].dropna().iloc[0]),
        "trades": int(len(events)),
        "guard_hits": int(events["guard_hit"].sum()),
        "win_rate": float((pnl > 0).mean()),
        "total_pnl_usd": float(pnl.sum()),
        "max_dd_usd": max_drawdown(pnl),
        "profit_factor": profit_factor(pnl),
        "avg_pnl_usd": float(pnl.mean()),
        "daily_sharpe_annualized": annualized_sharpe(daily),
        "daily_sortino_annualized": annualized_sortino(daily),
        "best_day_pnl_usd": float(daily.max()),
        "worst_day_pnl_usd": float(daily.min()),
    }
    row["return_dd"] = row["total_pnl_usd"] / abs(row["max_dd_usd"]) if row["max_dd_usd"] < 0 else None
    for days in WINDOW_DAYS:
        window = events[(events["signal_ts"] > anchor - pd.Timedelta(days=days)) & (events["signal_ts"] <= anchor)]
        w_pnl = window["pnl_guarded_usd"].astype(float)
        row[f"w{days}d_trades"] = int(len(window))
        row[f"w{days}d_pnl_usd"] = float(w_pnl.sum()) if not w_pnl.empty else 0.0
        row[f"w{days}d_max_dd_usd"] = max_drawdown(w_pnl) if not w_pnl.empty else 0.0
    return row


def money(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"${value:,.0f}"


def write_report(path: Path, results: pd.DataFrame) -> None:
    lines = [
        "# MNQ ORB Flash Guard Sweep",
        "",
        "This is a catastrophic safety guard simulation, not a normal strategy SL.",
        "The base strategy remains TP 2R or 15:00 NY time exit.",
        "",
        "| Guard | Guard Hits | PnL | Max DD | PF | Sharpe | Sortino | 30D PnL/DD | 50D PnL/DD | 100D PnL/DD | 200D PnL/DD |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results.itertuples(index=False):
        guard = "None" if pd.isna(row.guard_threshold_usd) else money(row.guard_threshold_usd)
        pf = "" if pd.isna(row.profit_factor) else f"{row.profit_factor:.2f}"
        sharpe = "" if pd.isna(row.daily_sharpe_annualized) else f"{row.daily_sharpe_annualized:.2f}"
        sortino = "" if pd.isna(row.daily_sortino_annualized) else f"{row.daily_sortino_annualized:.2f}"
        lines.append(
            f"| {guard} | {row.guard_hits} | {money(row.total_pnl_usd)} | {money(row.max_dd_usd)} | "
            f"{pf} | {sharpe} | {sortino} | "
            f"{money(row.w30d_pnl_usd)} / {money(row.w30d_max_dd_usd)} | "
            f"{money(row.w50d_pnl_usd)} / {money(row.w50d_max_dd_usd)} | "
            f"{money(row.w100d_pnl_usd)} / {money(row.w100d_max_dd_usd)} | "
            f"{money(row.w200d_pnl_usd)} / {money(row.w200d_max_dd_usd)} |"
        )
    lines.extend(
        [
            "",
            "Readout: use this table to pick a live safety threshold only after Topstep MLL and forward-test checks.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    cfg = load_config()
    strategy_dir = project_path(args.strategy_dir)
    events_path = strategy_dir / "events.parquet"
    summary_path = strategy_dir / "summary.json"
    output_csv = strategy_dir / "flash_guard_sweep.csv"
    output_md = strategy_dir / "flash_guard_report.md"
    if output_csv.exists() and not args.force:
        raise SystemExit(f"Output exists; use --force: {output_csv}")
    events = pd.read_parquet(events_path)
    for col in ["signal_ts", "entry_ts", "exit_ts"]:
        events[col] = pd.to_datetime(events[col], utc=True)
    l1 = pd.read_parquet(
        project_path(cfg["outputs"]["l1_context"]),
        columns=["timestamp_utc", "ny_date", "low", "bar_data_quality_ok"],
    )
    l1["timestamp_utc"] = pd.to_datetime(l1["timestamp_utc"], utc=True)
    l1_by_date = {date: day.sort_values("timestamp_utc").reset_index(drop=True) for date, day in l1.groupby("ny_date")}
    anchor = pd.Timestamp(json.loads(summary_path.read_text())["signal_range"]["anchor_ts"])

    rows = []
    baseline = simulate_threshold(events, l1_by_date, None, cfg)
    rows.append(summarize(baseline, anchor, l1["ny_date"]))
    for threshold in parse_thresholds(args.thresholds):
        guarded = simulate_threshold(events, l1_by_date, threshold, cfg)
        rows.append(summarize(guarded, anchor, l1["ny_date"]))
    results = pd.DataFrame(rows)
    results.to_csv(output_csv, index=False)
    write_report(output_md, results)
    print(json.dumps({"status": "PASS", "csv": str(output_csv), "report": str(output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

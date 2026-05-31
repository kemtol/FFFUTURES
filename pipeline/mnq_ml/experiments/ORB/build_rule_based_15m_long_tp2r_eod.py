#!/usr/bin/env python3
"""Build dedicated artifacts for the MNQ ORB 15m long TP2R/EOD candidate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any

import pandas as pd

from common import assert_mnq_namespaces, load_config, project_path, write_json

STRATEGY_ID = "rule_based_15m_long_tp2r_eod"
STRATEGY_NAME = "MNQ ORB 15m Long TP2R/EOD Rule-Based"
WINDOW_DAYS = [5, 10, 20, 30, 50, 100, 200]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sweep-dir", default="data/Level_2_Datamart/mnq/ORB/sweeps")
    parser.add_argument(
        "--output-dir",
        default="data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod",
    )
    return parser.parse_args()


def max_drawdown(pnls: pd.Series) -> float:
    if pnls.empty:
        return 0.0
    equity = pd.concat([pd.Series([0.0]), pnls.reset_index(drop=True).cumsum()], ignore_index=True)
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


def money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"${value:,.0f}"


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.2%}"


def number(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:,.{digits}f}"


def build_daily_pnl(events: pd.DataFrame, session_dates: pd.Series) -> pd.Series:
    event_dates = pd.to_datetime(events["ny_date"])
    start = event_dates.min()
    end = event_dates.max()
    date_index = pd.DatetimeIndex(pd.to_datetime(session_dates).drop_duplicates().sort_values())
    date_index = date_index[(date_index >= start) & (date_index <= end)]
    daily = events.assign(_date=event_dates).groupby("_date")["pnl_net_usd"].sum()
    return daily.reindex(date_index, fill_value=0.0).astype(float)


def summarize(events: pd.DataFrame, session_dates: pd.Series, anchor: pd.Timestamp) -> dict[str, Any]:
    pnl = events["pnl_net_usd"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    total = float(pnl.sum())
    daily = build_daily_pnl(events, session_dates)
    active_daily = daily[daily != 0.0]
    summary: dict[str, Any] = {
        "strategy_id": STRATEGY_ID,
        "strategy_name": STRATEGY_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "signal_range": {
            "min_signal_ts": events["signal_ts"].min().isoformat(),
            "max_signal_ts": events["signal_ts"].max().isoformat(),
            "anchor_ts": anchor.isoformat(),
        },
        "contract": {
            "instrument": "MNQ",
            "session": "New York regular session",
            "source_grain": "right-labeled M1 bars",
            "opening_range_minutes": 15,
            "side": "LONG_ONLY",
            "signal": "first M1 close above opening range high",
            "entry": "next M1 open after signal close",
            "exit": "TP_2R first, otherwise 15:00 NY time exit",
            "strategic_sl_used": False,
            "stop_reference": "opening_range_low_for_position_sizing_only",
            "max_trades_per_day": 1,
        },
        "costs": {
            "commission_round_turn_usd_per_contract": float(events["commission_round_turn_usd"].iloc[0]),
            "slippage_ticks_per_side": int(events["slippage_ticks_per_side"].iloc[0]),
            "slippage_round_turn_usd_per_contract": float(events["slippage_round_turn_usd_per_contract"].iloc[0]),
            "total_commission_paid_usd": float(events["commission_paid_usd"].sum()),
            "total_modeled_slippage_usd": float(events["modeled_slippage_usd"].sum()),
        },
        "performance": {
            "trades": int(len(events)),
            "win_rate": float((pnl > 0).mean()),
            "total_pnl_usd": total,
            "gross_profit_usd": gross_profit,
            "gross_loss_usd": gross_loss,
            "profit_factor": safe_ratio(gross_profit, abs(gross_loss)),
            "avg_pnl_usd": float(pnl.mean()),
            "median_pnl_usd": float(pnl.median()),
            "avg_win_usd": float(wins.mean()) if not wins.empty else 0.0,
            "avg_loss_usd": float(losses.mean()) if not losses.empty else 0.0,
            "payoff_ratio": safe_ratio(float(wins.mean()) if not wins.empty else 0.0, abs(float(losses.mean())) if not losses.empty else 0.0),
            "max_dd_usd": max_drawdown(pnl),
            "return_dd": safe_ratio(total, abs(max_drawdown(pnl))),
            "expectancy_per_trade_usd": float(pnl.mean()),
            "max_consecutive_wins": max_consecutive(pnl > 0),
            "max_consecutive_losses": max_consecutive(pnl < 0),
            "avg_contracts": float(events["contracts_used"].mean()),
        },
        "daily_quality": {
            "trading_days": int(len(daily)),
            "active_days": int(len(active_daily)),
            "active_day_rate": safe_ratio(float(len(active_daily)), float(len(daily))) or 0.0,
            "active_day_win_rate": float((active_daily > 0).mean()) if not active_daily.empty else 0.0,
            "daily_avg_pnl_usd": float(daily.mean()) if not daily.empty else 0.0,
            "daily_std_pnl_usd": float(daily.std(ddof=1)) if len(daily) > 1 else 0.0,
            "daily_sharpe_annualized": annualized_sharpe(daily),
            "daily_sortino_annualized": annualized_sortino(daily),
            "best_day_pnl_usd": float(daily.max()) if not daily.empty else 0.0,
            "worst_day_pnl_usd": float(daily.min()) if not daily.empty else 0.0,
            "best_day_profit_share": safe_ratio(float(daily.max()), total) if total > 0 and not daily.empty else None,
            "topstep_50pct_consistency_ok": bool((float(daily.max()) / total) <= 0.5) if total > 0 and not daily.empty else False,
        },
        "windows": {},
    }
    for days in WINDOW_DAYS:
        window = events[(events["signal_ts"] > anchor - pd.Timedelta(days=days)) & (events["signal_ts"] <= anchor)]
        w_pnl = window["pnl_net_usd"].astype(float)
        summary["windows"][f"{days}D"] = {
            "trades": int(len(window)),
            "win_rate": float((w_pnl > 0).mean()) if not w_pnl.empty else 0.0,
            "pnl_usd": float(w_pnl.sum()) if not w_pnl.empty else 0.0,
            "max_dd_usd": max_drawdown(w_pnl) if not w_pnl.empty else 0.0,
        }
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    perf = summary["performance"]
    daily = summary["daily_quality"]
    costs = summary["costs"]
    lines = [
        f"# {STRATEGY_NAME}",
        "",
        f"Strategy ID: `{STRATEGY_ID}`",
        "",
        "## Contract",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| Instrument | MNQ |",
        "| Session | New York regular session |",
        "| Source grain | Right-labeled M1 bars |",
        "| Opening range | 15 minutes after 09:30 NY |",
        "| Direction | Long only |",
        "| Signal | First M1 close above OR high |",
        "| Entry | Next M1 open after signal close |",
        "| Exit | TP 2R first, otherwise 15:00 NY EOD/time exit |",
        "| Strategic SL | None |",
        "| Stop reference | OR low for position sizing only |",
        "| Max trades | 1 per NY session |",
        "",
        "## Cost Model",
        "",
        "| Cost | Value |",
        "| --- | ---: |",
        f"| Commission + fees | ${costs['commission_round_turn_usd_per_contract']:.2f} RT / contract |",
        f"| Slippage | {costs['slippage_ticks_per_side']} tick per side |",
        f"| Modeled slippage | ${costs['slippage_round_turn_usd_per_contract']:.2f} RT / contract |",
        f"| Total commission paid | {money(costs['total_commission_paid_usd'])} |",
        f"| Total modeled slippage | {money(costs['total_modeled_slippage_usd'])} |",
        "",
        "## Performance",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Signal range | {summary['signal_range']['min_signal_ts'][:10]} to {summary['signal_range']['max_signal_ts'][:10]} |",
        f"| Trades | {perf['trades']:,} |",
        f"| Win rate | {pct(perf['win_rate'])} |",
        f"| Total PnL | {money(perf['total_pnl_usd'])} |",
        f"| Gross profit | {money(perf['gross_profit_usd'])} |",
        f"| Gross loss | {money(perf['gross_loss_usd'])} |",
        f"| Profit factor | {number(perf['profit_factor'])} |",
        f"| Max drawdown | {money(perf['max_dd_usd'])} |",
        f"| Return / DD | {number(perf['return_dd'])} |",
        f"| Expectancy / trade | ${perf['expectancy_per_trade_usd']:,.2f} |",
        f"| Median trade | ${perf['median_pnl_usd']:,.2f} |",
        f"| Average win | ${perf['avg_win_usd']:,.2f} |",
        f"| Average loss | ${perf['avg_loss_usd']:,.2f} |",
        f"| Payoff ratio | {number(perf['payoff_ratio'])} |",
        f"| Average contracts | {number(perf['avg_contracts'])} |",
        f"| Max consecutive wins | {perf['max_consecutive_wins']} |",
        f"| Max consecutive losses | {perf['max_consecutive_losses']} |",
        "",
        "## Daily Quality",
        "",
        "Sharpe and Sortino are computed from daily dollar PnL over MNQ NY session days, with zero PnL on no-trade days, annualized by `sqrt(252)`.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Trading days measured | {daily['trading_days']:,} |",
        f"| Active days | {daily['active_days']:,} |",
        f"| Active-day rate | {pct(daily['active_day_rate'])} |",
        f"| Active-day win rate | {pct(daily['active_day_win_rate'])} |",
        f"| Daily average PnL | ${daily['daily_avg_pnl_usd']:,.2f} |",
        f"| Daily PnL std dev | ${daily['daily_std_pnl_usd']:,.2f} |",
        f"| Daily Sharpe | {number(daily['daily_sharpe_annualized'])} |",
        f"| Daily Sortino | {number(daily['daily_sortino_annualized'])} |",
        f"| Best day | {money(daily['best_day_pnl_usd'])} |",
        f"| Worst day | {money(daily['worst_day_pnl_usd'])} |",
        f"| Best-day profit share | {pct(daily['best_day_profit_share'])} |",
        f"| 50% consistency flag | {'Pass' if daily['topstep_50pct_consistency_ok'] else 'Fail'} |",
        "",
        "## Rolling Windows",
        "",
        "| Window | Trades | Win Rate | PnL | Max DD |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in summary["windows"].items():
        lines.append(
            f"| {label} | {row['trades']} | {pct(row['win_rate'])} | "
            f"{money(row['pnl_usd'])} | {money(row['max_dd_usd'])} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- This is the dedicated artifact for the current cleanest MNQ ORB rule-based edge.",
            "- The strategy intentionally has no normal SL: it exits at TP 2R or 15:00 NY.",
            "- OR low is used only to size contracts; it is not a simulated stop exit.",
            "- Live promotion still needs a separate catastrophic safety guard for flash-drop risk.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    perf = summary["performance"]
    lines = [
        "# MNQ ORB Rule-Based 15m Long TP2R/EOD",
        "",
        "Dedicated artifact folder for the current MNQ rule-based candidate.",
        "",
        "## Files",
        "",
        "| File | Description |",
        "| --- | --- |",
        "| `events.parquet` | One row per executed strategy trade, net of TopstepX MNQ fee and modeled slippage |",
        "| `summary.json` | Machine-readable performance summary |",
        "| `manifest.json` | Build metadata and source references |",
        "| `report.md` | Human-readable strategy report |",
        "| `README.md` | This folder guide |",
        "",
        "## Current Snapshot",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Trades | {perf['trades']:,} |",
        f"| Win rate | {pct(perf['win_rate'])} |",
        f"| Net PnL | {money(perf['total_pnl_usd'])} |",
        f"| Max DD | {money(perf['max_dd_usd'])} |",
        f"| Profit factor | {number(perf['profit_factor'])} |",
        f"| Daily Sharpe | {number(summary['daily_quality']['daily_sharpe_annualized'])} |",
        f"| Daily Sortino | {number(summary['daily_quality']['daily_sortino_annualized'])} |",
        "",
        "## Contract",
        "",
        "- 15m opening range.",
        "- Long only after close above OR high.",
        "- Entry on next M1 open.",
        "- Exit on TP 2R or 15:00 NY time exit.",
        "- No normal strategy SL; OR low is sizing reference only.",
        "- Costs include TopstepX MNQ $1.24 round-turn per contract plus 1 tick slippage per side.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    cfg = load_config()
    assert_mnq_namespaces(cfg)

    sweep_dir = project_path(args.sweep_dir)
    output_dir = project_path(args.output_dir)
    events_path = output_dir / "events.parquet"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "report.md"
    readme_path = output_dir / "README.md"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output dir exists; use --force: {output_dir}")

    sweep_events_path = sweep_dir / "sweep_events.parquet"
    sweep_manifest_path = sweep_dir / "sweep_manifest.json"
    if not sweep_events_path.exists():
        raise SystemExit(f"Missing sweep events: {sweep_events_path}")
    if not sweep_manifest_path.exists():
        raise SystemExit(f"Missing sweep manifest: {sweep_manifest_path}")

    sweep = pd.read_parquet(sweep_events_path)
    for col in ["signal_ts", "entry_ts", "exit_ts"]:
        sweep[col] = pd.to_datetime(sweep[col], utc=True)
    events = sweep[
        (sweep["orb_minutes"] == 15)
        & (sweep["side_mode"] == "long")
        & (sweep["exit_mode"] == "tp_2r_or_time")
        & (sweep["target_risk_usd"] == 500)
    ].copy()
    if events.empty:
        raise SystemExit("No candidate events found in sweep output")
    if events["ny_date"].duplicated().any():
        raise SystemExit("Candidate has more than one trade per NY date")

    costs = cfg["costs"]
    point_value = float(costs["point_value_usd"])
    slippage_pts = float(costs["slippage_ticks_per_side"]) * float(costs["tick_size"])
    commission = float(costs["commission_round_turn_usd"])
    events["strategy_id"] = STRATEGY_ID
    events["pnl_net_usd"] = events["pnl_usd"].astype(float)
    events["commission_round_turn_usd"] = commission
    events["commission_paid_usd"] = commission * events["contracts_used"].astype(float)
    events["slippage_ticks_per_side"] = int(costs["slippage_ticks_per_side"])
    events["slippage_round_turn_usd_per_contract"] = 2.0 * slippage_pts * point_value
    events["modeled_slippage_usd"] = events["slippage_round_turn_usd_per_contract"] * events["contracts_used"].astype(float)
    events["pnl_before_commission_usd"] = events["pnl_net_usd"] + events["commission_paid_usd"]
    events["strategic_sl_used"] = False
    events["stop_reference_role"] = "position_sizing_only"
    events = events.sort_values("signal_ts").reset_index(drop=True)

    l1 = pd.read_parquet(project_path(cfg["outputs"]["l1_context"]), columns=["ny_date", "timestamp_utc"])
    l1["timestamp_utc"] = pd.to_datetime(l1["timestamp_utc"], utc=True)
    session_dates = l1["ny_date"].drop_duplicates()
    sweep_manifest = json.loads(sweep_manifest_path.read_text())
    anchor = pd.Timestamp(sweep_manifest["anchor_ts"])
    summary = summarize(events, session_dates, anchor)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy_id": STRATEGY_ID,
        "strategy_name": STRATEGY_NAME,
        "source_sweep_events": str(sweep_events_path),
        "source_sweep_manifest": str(sweep_manifest_path),
        "config": "pipeline/mnq_ml/experiments/ORB/config.json",
        "filter": {
            "orb_minutes": 15,
            "side_mode": "long",
            "exit_mode": "tp_2r_or_time",
            "target_risk_usd": 500,
        },
        "outputs": {
            "events": str(events_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
            "report": str(report_path),
            "readme": str(readme_path),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_parquet(events_path, index=False)
    write_json(summary_path, summary)
    write_json(manifest_path, manifest)
    write_report(report_path, summary)
    write_readme(readme_path, summary)
    print(json.dumps({"status": "PASS", **manifest["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

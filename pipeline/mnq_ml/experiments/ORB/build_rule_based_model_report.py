from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod"
MODEL_DIR = ROOT / "model/MNQ/ORB/rule_based_15m_long_tp2r_eod"
CHART_DIR = MODEL_DIR / "charts"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def usd(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_inputs() -> tuple[pd.DataFrame, dict]:
    events = pd.read_parquet(DATA_DIR / "events.parquet")
    events["ny_date"] = pd.to_datetime(events["ny_date"])
    events = events.sort_values(["ny_date", "entry_ts"]).reset_index(drop=True)

    with (DATA_DIR / "summary.json").open() as f:
        summary = json.load(f)

    return events, summary


def style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_equity_curve(events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    equity = events["pnl_net_usd"].cumsum()
    ax.plot(events["ny_date"], equity, color="#0f766e", linewidth=1.8)
    ax.axhline(0, color="#334155", linewidth=0.8, alpha=0.5)
    style_axis(ax, "MNQ ORB Rule-Based Equity Curve", "Cumulative PnL ($)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "equity_curve.svg")
    plt.close(fig)


def save_drawdown_curve(events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    equity = events["pnl_net_usd"].cumsum()
    drawdown = equity - equity.cummax()
    ax.fill_between(events["ny_date"], drawdown, 0, color="#dc2626", alpha=0.28)
    ax.plot(events["ny_date"], drawdown, color="#991b1b", linewidth=1.2)
    style_axis(ax, "MNQ ORB Rule-Based Drawdown", "Drawdown ($)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "drawdown_curve.svg")
    plt.close(fig)


def save_monthly_pnl(events: pd.DataFrame) -> None:
    monthly = events.assign(month=events["ny_date"].dt.to_period("M").astype(str))
    monthly = monthly.groupby("month", as_index=False)["pnl_net_usd"].sum()

    fig, ax = plt.subplots(figsize=(12, 4.8))
    colors = ["#0f766e" if x >= 0 else "#dc2626" for x in monthly["pnl_net_usd"]]
    ax.bar(monthly["month"], monthly["pnl_net_usd"], color=colors, width=0.85)
    ax.axhline(0, color="#334155", linewidth=0.8)
    style_axis(ax, "Monthly Net PnL", "PnL ($)")
    tick_step = max(1, len(monthly) // 14)
    ax.set_xticks(range(0, len(monthly), tick_step))
    ax.set_xticklabels(monthly["month"].iloc[::tick_step], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "monthly_pnl.svg")
    plt.close(fig)


def save_rolling_windows(summary: dict) -> None:
    order = ["5D", "10D", "20D", "30D", "50D", "100D", "200D"]
    rows = [
        {
            "window": window,
            "pnl": summary["windows"][window]["pnl_usd"],
            "dd": summary["windows"][window]["max_dd_usd"],
        }
        for window in order
    ]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = range(len(df))
    ax.bar([i - 0.18 for i in x], df["pnl"], width=0.36, color="#0f766e", label="PnL")
    ax.bar([i + 0.18 for i in x], df["dd"], width=0.36, color="#dc2626", label="Max DD")
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["window"])
    ax.legend(frameon=False)
    style_axis(ax, "Recent Rolling Window PnL / DD", "$")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "rolling_windows.svg")
    plt.close(fig)


def save_trade_pnl_distribution(events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.hist(events["pnl_net_usd"], bins=45, color="#2563eb", alpha=0.78)
    ax.axvline(0, color="#334155", linewidth=0.9)
    style_axis(ax, "Trade PnL Distribution", "Trade count")
    ax.set_xlabel("Net PnL per trade ($)")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "trade_pnl_distribution.svg")
    plt.close(fig)


def build_report(summary: dict) -> str:
    perf = summary["performance"]
    quality = summary["daily_quality"]
    costs = summary["costs"]
    signal = summary["signal_range"]
    windows = summary["windows"]

    window_order = ["5D", "10D", "20D", "30D", "50D", "100D", "200D"]
    window_rows = "\n".join(
        "| {window} | {trades:,} | {wr} | {pnl} | {dd} |".format(
            window=window,
            trades=int(windows[window]["trades"]),
            wr=pct(windows[window]["win_rate"]),
            pnl=usd(windows[window]["pnl_usd"]),
            dd=usd(windows[window]["max_dd_usd"]),
        )
        for window in window_order
    )

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return f"""# MNQ ORB 15m Long TP2R/EOD Rule-Based Report

Strategy ID: `rule_based_15m_long_tp2r_eod`

Generated: `{created_at}`

## Status

This is the current MNQ ORB rule-based research baseline. It is **not
live-ready** yet. It is the control candidate that ML overlays and future rule
variants must beat.

## Contract

| Field | Value |
| --- | --- |
| Instrument | MNQ |
| Session | New York regular session |
| Source grain | Right-labeled M1 bars |
| Opening range | 15 minutes after 09:30 NY |
| Direction | Long only |
| Signal | First M1 close above OR high |
| Entry | Next M1 open after signal close |
| Exit | TP 2R first, otherwise 15:00 NY time exit |
| Normal strategic SL | None |
| Stop reference | OR low for position sizing only |
| Target risk | $500 |
| Max trades | 1 per NY session |

## Visuals

![Equity curve](charts/equity_curve.svg)

![Drawdown curve](charts/drawdown_curve.svg)

![Monthly PnL](charts/monthly_pnl.svg)

![Rolling windows](charts/rolling_windows.svg)

![Trade PnL distribution](charts/trade_pnl_distribution.svg)

## Performance

| Metric | Value |
| --- | ---: |
| Signal range | {signal["min_signal_ts"][:10]} to {signal["max_signal_ts"][:10]} |
| Trades | {int(perf["trades"]):,} |
| Win rate | {pct(perf["win_rate"])} |
| Net PnL | {usd(perf["total_pnl_usd"])} |
| Gross profit | {usd(perf["gross_profit_usd"])} |
| Gross loss | {usd(perf["gross_loss_usd"])} |
| Profit factor | {perf["profit_factor"]:.2f} |
| Max drawdown | {usd(perf["max_dd_usd"])} |
| Return / DD | {perf["return_dd"]:.2f} |
| Expectancy / trade | {usd(perf["expectancy_per_trade_usd"])} |
| Median trade | {usd(perf["median_pnl_usd"])} |
| Average win | {usd(perf["avg_win_usd"])} |
| Average loss | {usd(perf["avg_loss_usd"])} |
| Payoff ratio | {perf["payoff_ratio"]:.2f} |
| Average contracts | {perf["avg_contracts"]:.2f} |
| Max consecutive wins | {int(perf["max_consecutive_wins"])} |
| Max consecutive losses | {int(perf["max_consecutive_losses"])} |

## Cost Model

| Cost | Value |
| --- | ---: |
| Commission + fees | ${costs["commission_round_turn_usd_per_contract"]:.2f} RT / contract |
| Slippage | {costs["slippage_ticks_per_side"]} tick per side |
| Modeled slippage | ${costs["slippage_round_turn_usd_per_contract"]:.2f} RT / contract |
| Total commission paid | {usd(costs["total_commission_paid_usd"])} |
| Total modeled slippage | {usd(costs["total_modeled_slippage_usd"])} |

## Daily Quality

Sharpe and Sortino are computed from daily dollar PnL over MNQ NY session days,
with zero PnL on no-trade days, annualized by `sqrt(252)`.

| Metric | Value |
| --- | ---: |
| Trading days measured | {int(quality["trading_days"]):,} |
| Active days | {int(quality["active_days"]):,} |
| Active-day rate | {pct(quality["active_day_rate"])} |
| Active-day win rate | {pct(quality["active_day_win_rate"])} |
| Daily average PnL | {usd(quality["daily_avg_pnl_usd"])} |
| Daily PnL std dev | {usd(quality["daily_std_pnl_usd"])} |
| Daily Sharpe | {quality["daily_sharpe_annualized"]:.2f} |
| Daily Sortino | {quality["daily_sortino_annualized"]:.2f} |
| Best day | {usd(quality["best_day_pnl_usd"])} |
| Worst day | {usd(quality["worst_day_pnl_usd"])} |
| Best-day profit share | {pct(quality["best_day_profit_share"])} |
| 50% consistency flag | {"Pass" if quality["topstep_50pct_consistency_ok"] else "Fail"} |

## Rolling Windows

| Window | Trades | Win Rate | PnL | Max DD |
| ---: | ---: | ---: | ---: | ---: |
{window_rows}

## Readout

- This is a rule-based strategy, not ML.
- The edge is positive but still shallow: PF is 1.12 and long-run Sharpe is 0.50.
- Recent 30D window is the attractive part: 18 trades, {usd(windows["30D"]["pnl_usd"])} PnL, {usd(windows["30D"]["max_dd_usd"])} max DD.
- The strategy has no normal SL; OR low is used for position sizing only.
- Live promotion still needs Topstep MLL simulation, first-$3000 path review,
  catastrophic guard choice, and forward-test execution plumbing.

## Source Artifacts

```text
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/events.parquet
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/summary.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/manifest.json
```
"""


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    events, summary = load_inputs()

    save_equity_curve(events)
    save_drawdown_curve(events)
    save_monthly_pnl(events)
    save_rolling_windows(summary)
    save_trade_pnl_distribution(events)

    (MODEL_DIR / "REPORT.md").write_text(build_report(summary))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy_id": summary["strategy_id"],
        "source_events": rel(DATA_DIR / "events.parquet"),
        "source_summary": rel(DATA_DIR / "summary.json"),
        "outputs": {
            "report": rel(MODEL_DIR / "REPORT.md"),
            "metrics": rel(MODEL_DIR / "metrics.json"),
            "charts": [
                rel(CHART_DIR / "equity_curve.svg"),
                rel(CHART_DIR / "drawdown_curve.svg"),
                rel(CHART_DIR / "monthly_pnl.svg"),
                rel(CHART_DIR / "rolling_windows.svg"),
                rel(CHART_DIR / "trade_pnl_distribution.svg"),
            ],
        },
    }
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

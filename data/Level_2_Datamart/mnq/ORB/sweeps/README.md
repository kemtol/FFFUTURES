# MNQ ORB Parameter Sweeps

Research grid for the MNQ ORB 1m baseline.

## Current Grid

```text
orb_minutes: 10, 15, 20, 30
side_mode: long, short, long_short
target_risk_usd: 100, 200, 300, 400, 500, 600
exit_mode: time_exit, tp_2r_or_time
time_exit: 15:00 NY
source: data/Level_1_Features/mnq/ORB/context.parquet
```

## Files

| File | Rows | Grain | Description |
| --- | ---: | --- | --- |
| `sweep_base_opportunities.parquet` | 33,756 | one row per ORB opportunity before risk sizing | Parameter-independent trade opportunities |
| `sweep_events.parquet` | 159,294 | one row per executable trade per risk setting | Expanded events after integer contract sizing |
| `sweep_results.parquet` | 144 | one row per parameter set | Summary metrics and ranking fields |
| `sweep_manifest.json` | - | manifest | Grid and artifact metadata |

## Key Metrics

`sweep_results.parquet` now includes 72 columns:

- Core PnL: total PnL, gross profit/loss, max drawdown, return/DD.
- Distribution: profit factor, average trade, median trade, average win/loss,
  payoff ratio, expectancy, max consecutive wins/losses.
- Daily quality: active-day rate, active-day win rate, daily average/std PnL,
  annualized daily Sharpe, annualized daily Sortino, best/worst day.
- Topstep lens: best-day profit share and a simple 50% consistency flag.
- Recent windows: rolling 5D/10D/20D/30D/50D/100D/200D PnL + drawdown.

Sharpe and Sortino are computed from daily dollar PnL over MNQ NY session days,
with zero PnL on no-trade days, annualized with `sqrt(252)`.

## Current Takeaway

Best current Topstep 30-day lens candidate:

```text
orb_minutes: 15
side_mode: long
exit_mode: tp_2r_or_time
target_risk_usd: 500
```

Dedicated candidate folder:

```text
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/
```

The dedicated folder contains `events.parquet`, `summary.json`, `manifest.json`,
`report.md`, and `README.md`.

| Window | Trades | PnL | Max DD |
| ---: | ---: | ---: | ---: |
| 5D | 1 | $101 | $0 |
| 10D | 5 | $895 | -$222 |
| 20D | 10 | $2,348 | -$222 |
| 30D | 18 | $3,460 | -$551 |
| 50D | 30 | $5,385 | -$861 |
| 100D | 54 | $4,035 | -$4,085 |
| 200D | 94 | $5,385 | -$4,561 |

Full-history context:

| Metric | Value |
| --- | ---: |
| Trades | 1,296 |
| Win rate | 56.48% |
| Total PnL | $33,091 |
| Max DD | -$12,124 |
| Return/DD | 2.73 |
| 2026 trades | 72 |
| 2026 PnL | $6,096 |
| 2026 max DD | -$4,085 |
| Profit factor | 1.12 |
| Daily Sharpe, annualized | 0.50 |
| Daily Sortino, annualized | 0.64 |
| Expectancy / trade | $25.53 |
| Average win | $429.87 |
| Average loss | -$499.25 |
| Payoff ratio | 0.86 |
| Max consecutive wins | 10 |
| Max consecutive losses | 6 |
| Best day | $990.78 |
| Worst day | -$3,780.96 |
| Best-day profit share | 2.99% |
| 50% consistency flag | Pass |

Aggressive nearby candidate:

```text
orb_minutes: 15
side_mode: long
exit_mode: tp_2r_or_time
target_risk_usd: 600
```

| Window | Trades | PnL | Max DD |
| ---: | ---: | ---: | ---: |
| 30D | 18 | $3,581 | -$823 |
| 50D | 30 | $6,222 | -$859 |
| 100D | 54 | $4,414 | -$5,014 |
| 200D | 94 | $6,322 | -$5,014 |

Do not treat these as live-ready. Next gate is Topstep daily consistency and MLL
simulation.

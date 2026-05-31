# MNQ ORB 15m Long TP2R/EOD Rule-Based

Model / strategy ID:

```text
rule_based_15m_long_tp2r_eod
```

This is the current cleanest MNQ ORB rule-based baseline. It is the candidate
we use as the control before judging any ML overlay.

## Status

| Field | Value |
| --- | --- |
| Instrument | MNQ |
| Type | Rule-based strategy, not ML |
| Current role | Best current research baseline |
| Live status | Not live-ready |
| Main comparison target | `orb_risk_adjusted_model` |

## Report

Primary report:

```text
model/MNQ/ORB/rule_based_15m_long_tp2r_eod/REPORT.md
```

Included model-package artifacts:

| File | Description |
| --- | --- |
| `REPORT.md` | Human-readable report with PnL, DD, Sharpe, Sortino, rolling windows, and chart links |
| `metrics.json` | Machine-readable copy of the canonical performance summary |
| `monte_carlo_metrics.json` | Bootstrap stress-test summary for 30D/100D/200D horizons |
| `manifest.json` | Source/output lineage for this model package |
| `charts/equity_curve.png` | Equity curve |
| `charts/drawdown_curve.png` | Drawdown curve |
| `charts/monthly_pnl.png` | Monthly net PnL |
| `charts/rolling_windows.png` | 5D/10D/20D/30D/50D/100D/200D PnL and DD |
| `charts/trade_pnl_distribution.png` | Trade PnL histogram |
| `monte_carlo/monte_pnl_fan_30d.png` | 30D Monte Carlo PnL fan chart |
| `monte_carlo/monte_final_pnl_cdf_30d.png` | 30D final PnL CDF |
| `monte_carlo/monte_maxdd_hist_30d.png` | 30D Monte Carlo max drawdown histogram |
| `monte_carlo/monte_pnl_fan_100d.png` | 100D Monte Carlo PnL fan chart |

## Strategy Contract

| Field | Value |
| --- | --- |
| Session | New York regular session |
| Source grain | Right-labeled M1 bars |
| Opening range | First 15 minutes after 09:30 NY |
| Direction | Long only |
| Signal | First M1 close above OR high |
| Entry | Next M1 open after signal close |
| Exit | TP 2R first, otherwise 15:00 NY time exit |
| Normal SL | None |
| Stop reference | OR low for sizing only |
| Baseline risk | $500 |
| Max trades | 1 per NY session |

## Current Snapshot

| Metric | Value |
| --- | ---: |
| Signal range | 2019-05-06 to 2026-05-26 |
| Trades | 1,296 |
| Win rate | 56.48% |
| Net PnL | $33,091 |
| Max DD | $-12,124 |
| Profit factor | 1.12 |
| Daily Sharpe | 0.50 |
| Daily Sortino | 0.64 |
| 30D trades | 18 |
| 30D PnL | $3,460 |
| 30D max DD | $-551 |

## Canonical Artifacts

The canonical parquet and generated reports stay under `data/` for lineage:

```text
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/events.parquet
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/summary.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/manifest.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/report.md
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/flash_guard_report.md
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/flash_guard_sweep.csv
```

This model folder intentionally does not duplicate the event parquet. It is the
human-facing model card and pointer to the canonical artifacts.

## Promotion Gaps

Before this can be considered live-ready:

- Topstep MLL simulation must pass.
- First-$3000 path and consistency must be reviewed.
- Forward-test execution plumbing must be defined.
- Catastrophic guard must be chosen separately from the base strategy exit.

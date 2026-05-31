# MNQ ORB Volatility Target Level 2

Event datamart for the `orb_vol_target` baseline.

## Files

| File | Rows | Description |
| --- | ---: | --- |
| `events.parquet` | 637 | One long-only ORB event per eligible NY day |
| `events_manifest.json` | - | Build summary |
| `training_gate.json` | - | P0 data gate report |
| `sweeps/` | - | OR duration / side / risk parameter sweep artifacts |
| `rule_based_15m_long_tp2r_eod/` | - | Dedicated artifact folder for the current rule-based candidate |
| `ml_filter/` | - | Narrow rule-based candidate ML filter datamart |
| `orb_risk_adjusted_model/` | - | Broader long/short breakout probability and Kelly-sizing datamart |
| `downside_extension_reversal*/` | - | Research-only long reversal after downside OR extension; includes TP 2R, TP 1R, and RSI14 oversold variants |

The canonical baseline event file is the first M1 time-exit build. It is kept
for reproducibility, but it is no longer the decision artifact for parameter
selection.

Parameter selection should use `sweeps/sweep_results.parquet`, not only the
canonical baseline event file.

Current research candidate from the sweep:

```text
15m OR, long only, TP 2R or 15:00 NY, risk $500
```

Short-window evidence:

| Window | Trades | PnL | Max DD |
| ---: | ---: | ---: | ---: |
| 5D | 1 | $101 | $0 |
| 10D | 5 | $895 | -$222 |
| 20D | 10 | $2,348 | -$222 |
| 30D | 18 | $3,460 | -$551 |
| 50D | 30 | $5,385 | -$861 |
| 100D | 54 | $4,035 | -$4,085 |
| 200D | 94 | $5,385 | -$4,561 |

Strategy-quality metrics for this row:

| Metric | Value |
| --- | ---: |
| Profit factor | 1.12 |
| Daily Sharpe, annualized | 0.50 |
| Daily Sortino, annualized | 0.64 |
| Expectancy / trade | $25.53 |
| Average win / loss | $429.87 / -$499.25 |
| Max consecutive wins / losses | 10 / 6 |
| Best day / worst day | $990.78 / -$3,780.96 |
| Best-day profit share | 2.99% |

Latest breakout-quality ML datamart:

```text
orb_risk_adjusted_model/breakout_quality_features.parquet
rows: 2,559
columns: 77
feature columns: 62
```

The 62 features include volatility/ATR, VWAP, overnight structure, breakout
quality, prior-day context, and prior daily SPY/QQQ/VIX/TNX/DXY confluence.

Downside extension reversal readout:

```text
Best research branch remains deep downside extension, not the shallow 1R branch.
For 2R extension + close-next-open + RSI14 <= 30 + $500 risk:
TP 1R: full-history -$3,073 / DD -$6,385; 50D +$1,172; 100D +$2,128
TP 2R: full-history -$4,995 / DD -$10,124; 50D +$1,990; 100D +$3,330
```

TP 1R is smoother but caps the recent recovery; neither reversal branch is a
replacement for the current continuation candidate.

# NASDAQ Micro Futures ORB 15m Long TP2R/EOD Rule-Based

Model / strategy ID:

```text
rule_based_15m_long_tp2r_eod
```

This is the current cleanest NASDAQ Micro Futures ORB rule-based baseline.
The technical contract ticker is `MNQ` / Micro E-mini Nasdaq-100 futures. This
is the candidate we use as the control before judging any ML overlay.

## Status

| Field | Value |
| --- | --- |
| Instrument | NASDAQ Micro Futures (`MNQ`) |
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
| `charts/supertrend_variant_equity_curve.png` | ST5_50 variant equity comparison |
| `charts/supertrend_variant_drawdown_curve.png` | ST5_50 variant drawdown comparison |
| `charts/supertrend_variant_monthly_pnl_2026.png` | ST5_50 variant monthly PnL for 2026 |
| `charts/supertrend_variant_rolling_windows.png` | ST5_50 variant rolling PnL/DD |
| `charts/supertrend_variant_trade_pnl_distribution.png` | ST5_50 variant trade PnL distribution |
| `charts/supertrend_variant_march_2026_equity.png` | ST5_50 variant March 2026 equity comparison |
| `charts/short_reversal_switch_equity_curve.png` | Short breakout switch-to-long equity comparison |
| `charts/short_reversal_switch_drawdown_curve.png` | Short breakout switch-to-long drawdown comparison |
| `charts/short_reversal_switch_last30_equity.png` | Short breakout switch-to-long 30D equity comparison |
| `monte_carlo/monte_pnl_fan_30d.png` | 30D Monte Carlo PnL fan chart |
| `monte_carlo/monte_final_pnl_cdf_30d.png` | 30D final PnL CDF |
| `monte_carlo/monte_maxdd_hist_30d.png` | 30D Monte Carlo max drawdown histogram |
| `monte_carlo/monte_pnl_fan_100d.png` | 100D Monte Carlo PnL fan chart |
| `supertrend_regime_audit.md` | Audit SuperTrend regime filter grid for March 2026 drawdown |
| `supertrend_filter_candidates.csv` | Machine-readable candidate table for all ST bullish conjunctions |
| `supertrend_variant_comparison.md` | Side-by-side comparison: no ST, ST5_50, long+short, long+short ST aligned |
| `supertrend_variant_comparison.csv` | Machine-readable ST5_50 variant comparison |
| `short_reversal_switch_comparison.md` | Short breakout switch-to-long audit with short TP 1R/1.5R/2R |
| `short_reversal_switch_comparison.csv` | Machine-readable summary for short switch variants |
| `short_reversal_switch_events.csv` | Sequence-level events for short switch variants |
| `short_reversal_switch_legs.csv` | Leg-level attribution for short switch variants |

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
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/supertrend_regime_features.parquet
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/supertrend_regime_manifest.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/supertrend_variant_comparison_manifest.json
data/Level_2_Datamart/mnq/ORB/rule_based_15m_long_tp2r_eod/short_reversal_switch_comparison_manifest.json
```

This model folder intentionally does not duplicate the event parquet. It is the
human-facing model card and pointer to the canonical artifacts.

## SuperTrend Regime Audit

SuperTrend filter audit is research-only. It computes ST on 5m and 15m bars
with ATR periods 5/10/20/50, fixed factor 4.0, then joins the latest completed
feature timestamp to each ORB signal. The gate currently reports zero
lookahead violations.

Current read: ST filters can materially reduce the March 2026 drawdown, but
they also reduce recent trade count and 30D PnL. Treat them as regime-filter
candidates, not as the promoted rule yet.

The cleanest P0 variant is `ST5_50` as a long-only filter. The long+short
ST-aligned variant improves full-history PnL/DD but currently fails recent 30D
quality, so it remains exploratory.

## Promotion Gaps

Before this can be considered live-ready:

- Topstep MLL simulation must pass.
- First-$3000 path and consistency must be reviewed.
- Forward-test execution plumbing must be defined.
- Catastrophic guard must be chosen separately from the base strategy exit.
